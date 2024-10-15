import Mathlib

namespace NUMINAMATH_CALUDE_growth_rate_correct_l1432_143215

/-- The average annual growth rate of vegetable production value from 2013 to 2015 -/
def average_growth_rate : ℝ := 0.25

/-- The initial production value in 2013 (in millions of yuan) -/
def initial_value : ℝ := 6.4

/-- The final production value in 2015 (in millions of yuan) -/
def final_value : ℝ := 10

/-- Theorem stating that the average annual growth rate correctly relates the initial and final values -/
theorem growth_rate_correct : initial_value * (1 + average_growth_rate)^2 = final_value := by
  sorry

end NUMINAMATH_CALUDE_growth_rate_correct_l1432_143215


namespace NUMINAMATH_CALUDE_sequence_120th_term_l1432_143253

/-- A function that returns the sum of digits of a positive integer -/
def digit_sum (n : ℕ) : ℕ := sorry

/-- A function that returns the nth term of the sequence of positive integers 
    whose digits sum to 10, arranged in ascending order -/
def sequence_term (n : ℕ) : ℕ := sorry

/-- The main theorem: The 120th term of the sequence is 2017 -/
theorem sequence_120th_term : sequence_term 120 = 2017 := by sorry

end NUMINAMATH_CALUDE_sequence_120th_term_l1432_143253


namespace NUMINAMATH_CALUDE_optimal_room_allocation_l1432_143205

theorem optimal_room_allocation (total_people : Nat) (large_room_capacity : Nat) 
  (h1 : total_people = 26) (h2 : large_room_capacity = 3) : 
  ∃ (small_room_capacity : Nat), 
    small_room_capacity = total_people - large_room_capacity ∧ 
    small_room_capacity > 0 ∧
    (∀ (x : Nat), x > 0 ∧ x < small_room_capacity → 
      (total_people - large_room_capacity) % x ≠ 0) :=
by sorry

end NUMINAMATH_CALUDE_optimal_room_allocation_l1432_143205


namespace NUMINAMATH_CALUDE_range_of_p_l1432_143293

def h (x : ℝ) : ℝ := 4 * x - 3

def p (x : ℝ) : ℝ := h (h (h x))

theorem range_of_p :
  ∀ y ∈ Set.range (fun x => p x), 1 ≤ y ∧ y ≤ 129 ∧
  ∀ y, 1 ≤ y ∧ y ≤ 129 → ∃ x, 1 ≤ x ∧ x ≤ 3 ∧ p x = y :=
by sorry

end NUMINAMATH_CALUDE_range_of_p_l1432_143293


namespace NUMINAMATH_CALUDE_starting_number_of_range_l1432_143252

/-- Given a sequence of 10 consecutive multiples of 5 ending with 65,
    prove that the first number in the sequence is 15. -/
theorem starting_number_of_range (seq : Fin 10 → ℕ) : 
  (∀ i : Fin 10, seq i % 5 = 0) →  -- All numbers are divisible by 5
  (∀ i : Fin 9, seq i.succ = seq i + 5) →  -- Consecutive multiples of 5
  seq 9 = 65 →  -- The last number is 65
  seq 0 = 15 := by  -- The first number is 15
sorry


end NUMINAMATH_CALUDE_starting_number_of_range_l1432_143252


namespace NUMINAMATH_CALUDE_brave_2022_first_appearance_l1432_143262

/-- The cycle length of the letters "BRAVE" -/
def letter_cycle_length : ℕ := 5

/-- The cycle length of the digits "2022" -/
def digit_cycle_length : ℕ := 4

/-- The line number where "BRAVE 2022" first appears -/
def first_appearance : ℕ := 20

theorem brave_2022_first_appearance :
  Nat.lcm letter_cycle_length digit_cycle_length = first_appearance :=
by sorry

end NUMINAMATH_CALUDE_brave_2022_first_appearance_l1432_143262


namespace NUMINAMATH_CALUDE_complex_real_condition_l1432_143272

theorem complex_real_condition (m : ℝ) :
  let z : ℂ := m^2 * (1 + Complex.I) - m * (m + Complex.I)
  (z.im = 0) ↔ (m = 0 ∨ m = 1) := by
  sorry

end NUMINAMATH_CALUDE_complex_real_condition_l1432_143272


namespace NUMINAMATH_CALUDE_similar_triangles_side_length_l1432_143295

theorem similar_triangles_side_length 
  (A₁ A₂ : ℕ) (k : ℕ) (side_small : ℝ) :
  A₁ > A₂ →
  A₁ - A₂ = 32 →
  A₁ = k^2 * A₂ →
  side_small = 4 →
  ∃ (side_large : ℝ), side_large = 12 :=
by sorry

end NUMINAMATH_CALUDE_similar_triangles_side_length_l1432_143295


namespace NUMINAMATH_CALUDE_unique_r_value_l1432_143217

/-- The polynomial function f(x) -/
def f (r : ℝ) (x : ℝ) : ℝ := 3 * x^4 + 2 * x^3 - x^2 - 5 * x + r

/-- Theorem stating that r = -5 is the unique value that satisfies f(-1) = 0 -/
theorem unique_r_value : ∃! r : ℝ, f r (-1) = 0 ∧ r = -5 := by sorry

end NUMINAMATH_CALUDE_unique_r_value_l1432_143217


namespace NUMINAMATH_CALUDE_max_k_value_l1432_143218

theorem max_k_value (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  (∀ x y, x > 0 → y > 0 → (x + 2*y) / (x*y) ≥ k / (2*x + y)) →
  k ≤ 9 :=
by sorry

end NUMINAMATH_CALUDE_max_k_value_l1432_143218


namespace NUMINAMATH_CALUDE_f_max_value_l1432_143202

/-- The quadratic function f(x) = -5x^2 + 25x - 1 -/
def f (x : ℝ) : ℝ := -5 * x^2 + 25 * x - 1

/-- The maximum value of f(x) is 129/4 -/
theorem f_max_value : ∃ (M : ℝ), M = 129 / 4 ∧ ∀ (x : ℝ), f x ≤ M := by
  sorry

end NUMINAMATH_CALUDE_f_max_value_l1432_143202


namespace NUMINAMATH_CALUDE_total_marks_calculation_l1432_143232

theorem total_marks_calculation (num_candidates : ℕ) (average_marks : ℕ) 
  (h1 : num_candidates = 120) (h2 : average_marks = 35) : 
  num_candidates * average_marks = 4200 := by
  sorry

end NUMINAMATH_CALUDE_total_marks_calculation_l1432_143232


namespace NUMINAMATH_CALUDE_three_in_all_curriculums_l1432_143244

/-- Represents the number of people in different curriculum groups -/
structure CurriculumGroups where
  yoga : ℕ
  cooking : ℕ
  weaving : ℕ
  cookingOnly : ℕ
  cookingAndYoga : ℕ
  cookingAndWeaving : ℕ

/-- Calculates the number of people participating in all curriculums -/
def allCurriculums (g : CurriculumGroups) : ℕ :=
  g.cooking - g.cookingOnly - g.cookingAndYoga - g.cookingAndWeaving

/-- Theorem stating that 3 people participate in all curriculums -/
theorem three_in_all_curriculums (g : CurriculumGroups) 
  (h1 : g.yoga = 35)
  (h2 : g.cooking = 20)
  (h3 : g.weaving = 15)
  (h4 : g.cookingOnly = 7)
  (h5 : g.cookingAndYoga = 5)
  (h6 : g.cookingAndWeaving = 5) :
  allCurriculums g = 3 := by
  sorry

end NUMINAMATH_CALUDE_three_in_all_curriculums_l1432_143244


namespace NUMINAMATH_CALUDE_gigi_jellybeans_l1432_143235

theorem gigi_jellybeans (gigi_jellybeans : ℕ) (rory_jellybeans : ℕ) (lorelai_jellybeans : ℕ) :
  rory_jellybeans = gigi_jellybeans + 30 →
  lorelai_jellybeans = 3 * (gigi_jellybeans + rory_jellybeans) →
  lorelai_jellybeans = 180 →
  gigi_jellybeans = 15 := by
sorry

end NUMINAMATH_CALUDE_gigi_jellybeans_l1432_143235


namespace NUMINAMATH_CALUDE_negative_reciprocal_of_opposite_of_neg_abs_three_l1432_143226

-- Definition of opposite numbers
def opposite (a b : ℝ) : Prop := a = -b

-- Definition of reciprocal
def reciprocal (a b : ℝ) : Prop := a * b = 1

-- Theorem to prove
theorem negative_reciprocal_of_opposite_of_neg_abs_three :
  ∃ x : ℝ, opposite (-|(-3)|) x ∧ reciprocal (-1/3) (-x) := by sorry

end NUMINAMATH_CALUDE_negative_reciprocal_of_opposite_of_neg_abs_three_l1432_143226


namespace NUMINAMATH_CALUDE_victoria_friends_l1432_143292

theorem victoria_friends (total_pairs : ℕ) (shoes_per_person : ℕ) (victoria_shoes : ℕ) : 
  total_pairs = 36 →
  shoes_per_person = 2 →
  victoria_shoes = 2 →
  (total_pairs * 2 - victoria_shoes) % shoes_per_person = 0 →
  (total_pairs * 2 - victoria_shoes) / shoes_per_person = 35 := by
  sorry

end NUMINAMATH_CALUDE_victoria_friends_l1432_143292


namespace NUMINAMATH_CALUDE_mph_to_fps_conversion_l1432_143245

/-- Conversion factor from miles per hour to feet per second -/
def mph_to_fps : ℝ := 1.5

/-- Cheetah's speed in miles per hour -/
def cheetah_speed : ℝ := 60

/-- Gazelle's speed in miles per hour -/
def gazelle_speed : ℝ := 40

/-- Initial distance between cheetah and gazelle in feet -/
def initial_distance : ℝ := 210

/-- Time for cheetah to catch up to gazelle in seconds -/
def catch_up_time : ℝ := 7

theorem mph_to_fps_conversion :
  (cheetah_speed * mph_to_fps * catch_up_time) - (gazelle_speed * mph_to_fps * catch_up_time) = initial_distance := by
  sorry

#check mph_to_fps_conversion

end NUMINAMATH_CALUDE_mph_to_fps_conversion_l1432_143245


namespace NUMINAMATH_CALUDE_grid_game_winner_l1432_143212

/-- Represents the possible outcomes of the game -/
inductive GameOutcome
  | Player1Wins
  | Player2Wins

/-- Represents the game state on a 1 × N grid strip -/
structure GameState (N : ℕ) where
  grid : Fin N → Option Bool
  turn : Bool

/-- Defines the game rules and winning conditions -/
def gameResult (N : ℕ) : GameOutcome :=
  if N = 1 then
    GameOutcome.Player1Wins
  else
    GameOutcome.Player2Wins

/-- Theorem stating the winning player based on the grid size -/
theorem grid_game_winner (N : ℕ) :
  (N = 1 → gameResult N = GameOutcome.Player1Wins) ∧
  (N > 1 → gameResult N = GameOutcome.Player2Wins) := by
  sorry

/-- Lemma: Player 1 wins when N = 1 -/
lemma player1_wins_n1 (N : ℕ) (h : N = 1) :
  gameResult N = GameOutcome.Player1Wins := by
  sorry

/-- Lemma: Player 2 wins when N > 1 -/
lemma player2_wins_n_gt1 (N : ℕ) (h : N > 1) :
  gameResult N = GameOutcome.Player2Wins := by
  sorry

end NUMINAMATH_CALUDE_grid_game_winner_l1432_143212


namespace NUMINAMATH_CALUDE_rita_swim_hours_l1432_143236

/-- The total number of hours Rita needs to swim --/
def total_swim_hours (backstroke breaststroke butterfly monthly_freestyle_sidestroke months : ℕ) : ℕ :=
  backstroke + breaststroke + butterfly + monthly_freestyle_sidestroke * months

/-- Theorem stating that Rita needs to swim 1500 hours in total --/
theorem rita_swim_hours :
  total_swim_hours 50 9 121 220 6 = 1500 :=
by sorry

end NUMINAMATH_CALUDE_rita_swim_hours_l1432_143236


namespace NUMINAMATH_CALUDE_correct_calculation_l1432_143248

theorem correct_calculation (a : ℝ) : (-a + 3) * (-3 - a) = a^2 - 9 := by
  sorry

end NUMINAMATH_CALUDE_correct_calculation_l1432_143248


namespace NUMINAMATH_CALUDE_lcm_18_24_l1432_143231

theorem lcm_18_24 : Nat.lcm 18 24 = 72 := by
  sorry

end NUMINAMATH_CALUDE_lcm_18_24_l1432_143231


namespace NUMINAMATH_CALUDE_scientific_notation_of_32_9_billion_l1432_143243

def billion : ℝ := 1000000000

theorem scientific_notation_of_32_9_billion :
  32.9 * billion = 3.29 * (10 : ℝ)^9 := by sorry

end NUMINAMATH_CALUDE_scientific_notation_of_32_9_billion_l1432_143243


namespace NUMINAMATH_CALUDE_perpendicular_transitivity_l1432_143279

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the perpendicular relation
variable (perp : Line → Plane → Prop)

-- Define the different relation
variable (different : ∀ {α : Type}, α → α → Prop)

theorem perpendicular_transitivity 
  (α β γ : Plane) (m n l : Line)
  (h_diff_planes : different α β ∧ different β γ ∧ different α γ)
  (h_diff_lines : different m n ∧ different n l ∧ different m l)
  (h_n_perp_α : perp n α)
  (h_n_perp_β : perp n β)
  (h_m_perp_α : perp m α) :
  perp m β :=
sorry

end NUMINAMATH_CALUDE_perpendicular_transitivity_l1432_143279


namespace NUMINAMATH_CALUDE_nested_f_result_l1432_143254

def f (p q : ℝ) (x : ℝ) : ℝ := x^2 + p*x + q

theorem nested_f_result (p q : ℝ) :
  (∀ x ∈ Set.Icc 1 3, |f p q x| ≤ 1/2) →
  (f p q)^[2017] ((3 + Real.sqrt 7) / 2) = (3 - Real.sqrt 7) / 2 :=
by sorry

end NUMINAMATH_CALUDE_nested_f_result_l1432_143254


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l1432_143296

theorem quadratic_inequality_solution_set (a b c : ℝ) :
  (∀ x : ℝ, a * x^2 + b * x + c > 0) →
  b^2 - 4*a*c < 0 ∧
  ¬(b^2 - 4*a*c < 0 → ∀ x : ℝ, a * x^2 + b * x + c > 0) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l1432_143296


namespace NUMINAMATH_CALUDE_tom_apple_purchase_l1432_143299

-- Define the given constants
def apple_price : ℝ := 70
def mango_amount : ℝ := 9
def mango_price : ℝ := 65
def total_paid : ℝ := 1145

-- Define the theorem
theorem tom_apple_purchase :
  ∃ (apple_amount : ℝ),
    apple_amount * apple_price + mango_amount * mango_price = total_paid ∧
    apple_amount = 8 := by
  sorry

end NUMINAMATH_CALUDE_tom_apple_purchase_l1432_143299


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l1432_143230

theorem quadratic_inequality_solution_set (x : ℝ) : x^2 + 3*x - 4 < 0 ↔ -4 < x ∧ x < 1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l1432_143230


namespace NUMINAMATH_CALUDE_comparison_of_powers_l1432_143203

theorem comparison_of_powers : 0.2^3 < 2^0.3 := by
  sorry

end NUMINAMATH_CALUDE_comparison_of_powers_l1432_143203


namespace NUMINAMATH_CALUDE_max_xy_value_l1432_143270

theorem max_xy_value (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : 4 * x^2 + 9 * y^2 + 3 * x * y = 30) : 
  x * y ≤ 2 ∧ ∃ (x₀ y₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ 4 * x₀^2 + 9 * y₀^2 + 3 * x₀ * y₀ = 30 ∧ x₀ * y₀ = 2 :=
sorry

end NUMINAMATH_CALUDE_max_xy_value_l1432_143270


namespace NUMINAMATH_CALUDE_cubic_quadratic_fraction_inequality_l1432_143266

theorem cubic_quadratic_fraction_inequality (s r : ℝ) (hs : 0 < s) (hr : 0 < r) (hsr : r < s) :
  (s^3 - r^3) / (s^3 + r^3) > (s^2 - r^2) / (s^2 + r^2) := by
  sorry

end NUMINAMATH_CALUDE_cubic_quadratic_fraction_inequality_l1432_143266


namespace NUMINAMATH_CALUDE_class_size_is_40_l1432_143265

/-- Represents the number of students who borrowed a specific number of books -/
structure BookBorrowers where
  zero : Nat
  one : Nat
  two : Nat
  threeOrMore : Nat

/-- Calculates the total number of students given the book borrowing data -/
def totalStudents (b : BookBorrowers) : Nat :=
  b.zero + b.one + b.two + b.threeOrMore

/-- Calculates the minimum number of books borrowed -/
def minBooksBorrowed (b : BookBorrowers) : Nat :=
  0 * b.zero + 1 * b.one + 2 * b.two + 3 * b.threeOrMore

/-- The given book borrowing data for the class -/
def classBorrowers : BookBorrowers := {
  zero := 2,
  one := 12,
  two := 10,
  threeOrMore := 16  -- This value is not given directly, but can be derived
}

theorem class_size_is_40 :
  totalStudents classBorrowers = 40 ∧
  (minBooksBorrowed classBorrowers : ℚ) / (totalStudents classBorrowers) = 2 := by
  sorry


end NUMINAMATH_CALUDE_class_size_is_40_l1432_143265


namespace NUMINAMATH_CALUDE_cube_surface_area_l1432_143234

/-- The surface area of a cube with edge length 7 cm is 294 square centimeters. -/
theorem cube_surface_area : 
  ∀ (edge_length : ℝ), 
  edge_length = 7 → 
  6 * edge_length^2 = 294 := by
  sorry

end NUMINAMATH_CALUDE_cube_surface_area_l1432_143234


namespace NUMINAMATH_CALUDE_overlap_length_l1432_143241

theorem overlap_length (total_length actual_distance : ℝ) (num_overlaps : ℕ) : 
  total_length = 98 → 
  actual_distance = 83 → 
  num_overlaps = 6 → 
  (total_length - actual_distance) / num_overlaps = 2.5 := by
  sorry

end NUMINAMATH_CALUDE_overlap_length_l1432_143241


namespace NUMINAMATH_CALUDE_expression_evaluation_l1432_143286

theorem expression_evaluation (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hxy : x - y^2 ≠ 0) :
  (y^2 - 1/x) / (x - y^2) = (x*y^2 - 1) / (x^2 - x*y^2) := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l1432_143286


namespace NUMINAMATH_CALUDE_complex_equation_solution_l1432_143264

theorem complex_equation_solution (z : ℂ) : (Complex.I - z = 2 - Complex.I) → z = -2 + 2 * Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l1432_143264


namespace NUMINAMATH_CALUDE_max_value_fraction_l1432_143206

theorem max_value_fraction (x y : ℝ) (hx : -6 ≤ x ∧ x ≤ -3) (hy : 1 ≤ y ∧ y ≤ 5) :
  (∀ a b : ℝ, -6 ≤ a ∧ a ≤ -3 → 1 ≤ b ∧ b ≤ 5 → (a + b + 1) / (a + 1) ≤ (x + y + 1) / (x + 1)) →
  (x + y + 1) / (x + 1) = 0 :=
by sorry

end NUMINAMATH_CALUDE_max_value_fraction_l1432_143206


namespace NUMINAMATH_CALUDE_fraction_inequality_l1432_143207

theorem fraction_inequality (a b c : ℝ) (h1 : a > b) (h2 : b > c) (h3 : c > 0) :
  a / b > (a + c) / (b + c) := by
  sorry

end NUMINAMATH_CALUDE_fraction_inequality_l1432_143207


namespace NUMINAMATH_CALUDE_tree_distance_l1432_143294

theorem tree_distance (n : ℕ) (d : ℝ) (h1 : n = 6) (h2 : d = 60) :
  (n - 1) * (d / 3) = 100 := by
  sorry

end NUMINAMATH_CALUDE_tree_distance_l1432_143294


namespace NUMINAMATH_CALUDE_smallest_number_divisible_by_three_prime_squares_l1432_143259

def is_divisible_by_three_prime_squares (n : ℕ) : Prop :=
  ∃ p q r : ℕ, 
    Nat.Prime p ∧ Nat.Prime q ∧ Nat.Prime r ∧ 
    p ≠ q ∧ p ≠ r ∧ q ≠ r ∧
    n % (p^2) = 0 ∧ n % (q^2) = 0 ∧ n % (r^2) = 0

theorem smallest_number_divisible_by_three_prime_squares :
  (∀ m : ℕ, m > 0 ∧ m < 900 → ¬(is_divisible_by_three_prime_squares m)) ∧
  is_divisible_by_three_prime_squares 900 := by
  sorry

end NUMINAMATH_CALUDE_smallest_number_divisible_by_three_prime_squares_l1432_143259


namespace NUMINAMATH_CALUDE_unread_fraction_of_book_l1432_143257

theorem unread_fraction_of_book (total : ℝ) (read : ℝ) : 
  total > 0 → read > total / 2 → read < total → (total - read) / total = 2 / 5 := by
  sorry

end NUMINAMATH_CALUDE_unread_fraction_of_book_l1432_143257


namespace NUMINAMATH_CALUDE_circle_center_sum_l1432_143251

theorem circle_center_sum (x y : ℝ) : 
  x^2 + y^2 = 4*x + 10*y - 12 → x + y = 7 :=
by sorry

end NUMINAMATH_CALUDE_circle_center_sum_l1432_143251


namespace NUMINAMATH_CALUDE_expression_equality_l1432_143263

theorem expression_equality (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (h : x = 1 / y) :
  (x - 1 / x) * (y + 1 / y) = x^2 - y^2 := by
  sorry

end NUMINAMATH_CALUDE_expression_equality_l1432_143263


namespace NUMINAMATH_CALUDE_triangle_with_angle_ratio_2_3_5_is_right_triangle_l1432_143216

theorem triangle_with_angle_ratio_2_3_5_is_right_triangle (a b c : ℝ) 
  (h_triangle : a + b + c = 180)
  (h_ratio : ∃ (x : ℝ), a = 2*x ∧ b = 3*x ∧ c = 5*x) :
  c = 90 := by
sorry

end NUMINAMATH_CALUDE_triangle_with_angle_ratio_2_3_5_is_right_triangle_l1432_143216


namespace NUMINAMATH_CALUDE_not_all_tetrahedra_altitudes_intersect_l1432_143275

/-- A tetrahedron is represented by its four vertices in 3D space -/
def Tetrahedron := Fin 4 → ℝ × ℝ × ℝ

/-- An altitude of a tetrahedron is a line segment from a vertex perpendicular to the opposite face -/
def Altitude (t : Tetrahedron) (v : Fin 4) : Set (ℝ × ℝ × ℝ) :=
  sorry

/-- Predicate to check if all altitudes of a tetrahedron intersect at a single point -/
def altitudesIntersectAtPoint (t : Tetrahedron) : Prop :=
  ∃ p : ℝ × ℝ × ℝ, ∀ v : Fin 4, p ∈ Altitude t v

/-- Theorem stating that not all tetrahedra have altitudes intersecting at a single point -/
theorem not_all_tetrahedra_altitudes_intersect :
  ∃ t : Tetrahedron, ¬ altitudesIntersectAtPoint t :=
sorry

end NUMINAMATH_CALUDE_not_all_tetrahedra_altitudes_intersect_l1432_143275


namespace NUMINAMATH_CALUDE_mothers_age_l1432_143233

theorem mothers_age (daughter_age mother_age : ℕ) 
  (h1 : 2 * daughter_age + mother_age = 70)
  (h2 : daughter_age + 2 * mother_age = 95) :
  mother_age = 40 := by
  sorry

end NUMINAMATH_CALUDE_mothers_age_l1432_143233


namespace NUMINAMATH_CALUDE_august_math_problems_l1432_143280

theorem august_math_problems (a1 a2 a3 : ℕ) : 
  a1 = 600 →
  a3 = a1 + a2 - 400 →
  a1 + a2 + a3 = 3200 →
  a2 / a1 = 2 := by
sorry

end NUMINAMATH_CALUDE_august_math_problems_l1432_143280


namespace NUMINAMATH_CALUDE_intersection_implies_a_value_l1432_143204

def M (a : ℤ) : Set ℤ := {a, 0}

def N : Set ℤ := {x : ℤ | x^2 - 3*x < 0}

theorem intersection_implies_a_value (a : ℤ) (h : (M a) ∩ N ≠ ∅) : a = 1 ∨ a = 2 := by
  sorry

end NUMINAMATH_CALUDE_intersection_implies_a_value_l1432_143204


namespace NUMINAMATH_CALUDE_lt_iff_forall_add_lt_l1432_143274

theorem lt_iff_forall_add_lt (a b : ℝ) : a < b ↔ ∀ x ∈ Set.Ioo 0 1, a + x < b := by sorry

end NUMINAMATH_CALUDE_lt_iff_forall_add_lt_l1432_143274


namespace NUMINAMATH_CALUDE_open_box_volume_l1432_143242

/-- The volume of an open box formed by cutting squares from the corners of a rectangular sheet. -/
theorem open_box_volume
  (sheet_length : ℝ)
  (sheet_width : ℝ)
  (cut_length : ℝ)
  (h1 : sheet_length = 48)
  (h2 : sheet_width = 36)
  (h3 : cut_length = 5) :
  (sheet_length - 2 * cut_length) * (sheet_width - 2 * cut_length) * cut_length = 9880 :=
by sorry

end NUMINAMATH_CALUDE_open_box_volume_l1432_143242


namespace NUMINAMATH_CALUDE_ball_color_probability_l1432_143256

theorem ball_color_probability : 
  let n : ℕ := 8
  let p : ℝ := 1/2
  let num_arrangements : ℕ := n.choose (n/2)
  Fintype.card {s : Finset (Fin n) | s.card = n/2} / 2^n = 35/128 :=
by sorry

end NUMINAMATH_CALUDE_ball_color_probability_l1432_143256


namespace NUMINAMATH_CALUDE_tangent_line_at_2_a_range_l1432_143298

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := x^3 - a*x^2 + 10

-- Part 1: Tangent line when a = 1
theorem tangent_line_at_2 :
  ∃ (m b : ℝ), ∀ (x y : ℝ),
    y = m*x + b ↔ y = 8*x - 2 ∧ 
    (∃ (h : ℝ), h ≠ 0 ∧ (f 1 (2 + h) - f 1 2) / h = m) :=
sorry

-- Part 2: Range of a
theorem a_range :
  ∀ (a : ℝ), (∃ (x : ℝ), x ∈ Set.Icc 1 2 ∧ f a x < 0) ↔ a > 9/2 :=
sorry

end NUMINAMATH_CALUDE_tangent_line_at_2_a_range_l1432_143298


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l1432_143289

theorem quadratic_inequality_solution_set :
  {x : ℝ | x^2 + x - 2 < 0} = {x : ℝ | -2 < x ∧ x < 1} := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l1432_143289


namespace NUMINAMATH_CALUDE_f_properties_l1432_143247

noncomputable def f (x : ℝ) := Real.cos x ^ 4 - 2 * Real.sin x * Real.cos x - Real.sin x ^ 4

theorem f_properties :
  (∃ (T : ℝ), T > 0 ∧ (∀ x, f (x + T) = f x) ∧ 
    (∀ T' > 0, (∀ x, f (x + T') = f x) → T ≤ T')) ∧
  (∀ x ∈ Set.Icc 0 (Real.pi / 2), f x ≥ -Real.sqrt 2) ∧
  f (3 * Real.pi / 8) = -Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_f_properties_l1432_143247


namespace NUMINAMATH_CALUDE_polygon_sides_count_l1432_143220

/-- Represents the number of degrees in a circle -/
def degrees_in_circle : ℝ := 360

/-- Represents the common difference in the arithmetic progression of angles -/
def common_difference : ℝ := 3

/-- Represents the measure of the largest angle in the polygon -/
def largest_angle : ℝ := 150

/-- Theorem: A convex polygon with interior angles in arithmetic progression,
    a common difference of 3°, and the largest angle of 150° has 48 sides -/
theorem polygon_sides_count :
  ∀ n : ℕ,
  (n > 2) →
  (n * (2 * largest_angle - (n - 1) * common_difference) / 2 = (n - 2) * degrees_in_circle / 2) →
  n = 48 := by
  sorry


end NUMINAMATH_CALUDE_polygon_sides_count_l1432_143220


namespace NUMINAMATH_CALUDE_naclo4_formation_l1432_143222

-- Define the chemical reaction
structure Reaction where
  naoh : ℝ
  hclo4 : ℝ
  naclo4 : ℝ
  h2o : ℝ

-- Define the balanced equation
def balanced_equation (r : Reaction) : Prop :=
  r.naoh = r.hclo4 ∧ r.naoh = r.naclo4 ∧ r.naoh = r.h2o

-- Define the initial conditions
def initial_conditions (initial_naoh initial_hclo4 : ℝ) (r : Reaction) : Prop :=
  initial_naoh = 3 ∧ initial_hclo4 = 3 ∧ r.naoh ≤ initial_naoh ∧ r.hclo4 ≤ initial_hclo4

-- Theorem statement
theorem naclo4_formation 
  (initial_naoh initial_hclo4 : ℝ) 
  (r : Reaction) 
  (h1 : balanced_equation r) 
  (h2 : initial_conditions initial_naoh initial_hclo4 r) :
  r.naclo4 = min initial_naoh initial_hclo4 :=
sorry

end NUMINAMATH_CALUDE_naclo4_formation_l1432_143222


namespace NUMINAMATH_CALUDE_savings_calculation_l1432_143278

theorem savings_calculation (income expenditure savings : ℕ) : 
  (income * 3 = expenditure * 5) →  -- Income and expenditure ratio is 5:3
  (income = 10000) →                -- Income is Rs. 10000
  (savings = income - expenditure) →  -- Definition of savings
  (savings = 4000) :=                -- Prove that savings are Rs. 4000
by
  sorry

#check savings_calculation

end NUMINAMATH_CALUDE_savings_calculation_l1432_143278


namespace NUMINAMATH_CALUDE_income_comparison_l1432_143267

theorem income_comparison (juan tim mart : ℝ) 
  (h1 : tim = juan * (1 - 0.4))
  (h2 : mart = tim * (1 + 0.4)) :
  mart = juan * 0.84 := by
sorry

end NUMINAMATH_CALUDE_income_comparison_l1432_143267


namespace NUMINAMATH_CALUDE_seojun_apple_fraction_l1432_143208

theorem seojun_apple_fraction :
  let total_apples : ℕ := 100
  let seojun_apples : ℕ := 11
  (seojun_apples : ℚ) / total_apples = 0.11 := by
  sorry

end NUMINAMATH_CALUDE_seojun_apple_fraction_l1432_143208


namespace NUMINAMATH_CALUDE_chemical_solution_concentration_l1432_143240

theorem chemical_solution_concentration 
  (initial_volume : ℝ) 
  (initial_concentration : ℝ) 
  (drained_volume : ℝ) 
  (added_concentration : ℝ) 
  (final_volume : ℝ)
  (h1 : initial_volume = 50)
  (h2 : initial_concentration = 0.60)
  (h3 : drained_volume = 35)
  (h4 : added_concentration = 0.40)
  (h5 : final_volume = initial_volume) :
  let remaining_volume := initial_volume - drained_volume
  let initial_chemical := initial_volume * initial_concentration
  let drained_chemical := drained_volume * initial_concentration
  let remaining_chemical := initial_chemical - drained_chemical
  let added_chemical := drained_volume * added_concentration
  let final_chemical := remaining_chemical + added_chemical
  let final_concentration := final_chemical / final_volume
  final_concentration = 0.46 := by
sorry

end NUMINAMATH_CALUDE_chemical_solution_concentration_l1432_143240


namespace NUMINAMATH_CALUDE_plot_area_in_acres_l1432_143250

/-- Conversion factor from square miles to acres -/
def miles_to_acres : ℝ := 640

/-- Length of the plot in miles -/
def length : ℝ := 12

/-- Width of the plot in miles -/
def width : ℝ := 8

/-- Theorem stating that the area of the rectangular plot in acres is 61440 -/
theorem plot_area_in_acres :
  length * width * miles_to_acres = 61440 := by sorry

end NUMINAMATH_CALUDE_plot_area_in_acres_l1432_143250


namespace NUMINAMATH_CALUDE_negation_of_universal_proposition_l1432_143214

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, 2 * x^2 + 1 > 0) ↔ (∃ x : ℝ, 2 * x^2 + 1 ≤ 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_universal_proposition_l1432_143214


namespace NUMINAMATH_CALUDE_vinnie_word_count_excess_l1432_143210

def word_limit : ℕ := 1000
def friday_words : ℕ := 450
def saturday_words : ℕ := 650
def sunday_words : ℕ := 300
def friday_articles : ℕ := 25
def saturday_articles : ℕ := 40
def sunday_articles : ℕ := 15

theorem vinnie_word_count_excess :
  (friday_words + saturday_words + sunday_words) -
  (friday_articles + saturday_articles + sunday_articles) -
  word_limit = 320 := by
  sorry

end NUMINAMATH_CALUDE_vinnie_word_count_excess_l1432_143210


namespace NUMINAMATH_CALUDE_erdos_binomial_prime_factors_l1432_143284

-- Define the number of distinct prime factors function
noncomputable def num_distinct_prime_factors (m : ℕ) : ℕ := sorry

-- State the theorem
theorem erdos_binomial_prime_factors :
  ∃ (c : ℝ), c > 1 ∧
  ∀ (n k : ℕ), n > 0 ∧ k > 0 →
  (n : ℝ) > c^k →
  num_distinct_prime_factors (Nat.choose n k) ≥ k :=
sorry

end NUMINAMATH_CALUDE_erdos_binomial_prime_factors_l1432_143284


namespace NUMINAMATH_CALUDE_woodworker_tables_l1432_143213

theorem woodworker_tables (total_legs : ℕ) (chairs : ℕ) (chair_legs : ℕ) (table_legs : ℕ) 
  (h1 : total_legs = 40)
  (h2 : chairs = 6)
  (h3 : chair_legs = 4)
  (h4 : table_legs = 4) :
  (total_legs - chairs * chair_legs) / table_legs = 4 :=
by
  sorry

end NUMINAMATH_CALUDE_woodworker_tables_l1432_143213


namespace NUMINAMATH_CALUDE_laptop_selection_problem_l1432_143249

def choose (n k : ℕ) : ℕ := Nat.choose n k

theorem laptop_selection_problem :
  let type_a : ℕ := 4
  let type_b : ℕ := 5
  let total_selection : ℕ := 3
  (choose type_a 2 * choose type_b 1) + (choose type_a 1 * choose type_b 2) = 70 :=
by sorry

end NUMINAMATH_CALUDE_laptop_selection_problem_l1432_143249


namespace NUMINAMATH_CALUDE_two_distinct_roots_iff_a_in_open_interval_l1432_143255

-- Define the logarithmic function
noncomputable def log_base (base : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log base

-- Define the main theorem
theorem two_distinct_roots_iff_a_in_open_interval (a : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ 
    x₁ > 0 ∧ x₁ + a > 0 ∧ x₁ + a ≠ 1 ∧
    x₂ > 0 ∧ x₂ + a > 0 ∧ x₂ + a ≠ 1 ∧
    log_base (x₁ + a) (2 * x₁) = 2 ∧
    log_base (x₂ + a) (2 * x₂) = 2) ↔
  (0 < a ∧ a < 1/2) :=
sorry

end NUMINAMATH_CALUDE_two_distinct_roots_iff_a_in_open_interval_l1432_143255


namespace NUMINAMATH_CALUDE_token_game_ends_in_37_rounds_l1432_143282

/-- Represents the state of the game at any given round -/
structure GameState where
  tokensA : ℕ
  tokensB : ℕ
  tokensC : ℕ

/-- Represents the rules of the game -/
def nextRound (state : GameState) : GameState :=
  match state with
  | ⟨a, b, c⟩ =>
    if a ≥ b ∧ a ≥ c then ⟨a - 3, b + 1, c + 1⟩
    else if b ≥ a ∧ b ≥ c then ⟨a + 1, b - 3, c + 1⟩
    else ⟨a + 1, b + 1, c - 3⟩

/-- Checks if the game has ended (i.e., if any player has run out of tokens) -/
def gameEnded (state : GameState) : Bool :=
  state.tokensA = 0 ∨ state.tokensB = 0 ∨ state.tokensC = 0

/-- Plays the game for a given number of rounds -/
def playGame (initialState : GameState) (rounds : ℕ) : GameState :=
  match rounds with
  | 0 => initialState
  | n + 1 => nextRound (playGame initialState n)

/-- The main theorem statement -/
theorem token_game_ends_in_37_rounds :
  let initialState := GameState.mk 15 14 13
  gameEnded (playGame initialState 37) ∧ ¬gameEnded (playGame initialState 36) := by
  sorry


end NUMINAMATH_CALUDE_token_game_ends_in_37_rounds_l1432_143282


namespace NUMINAMATH_CALUDE_final_temperature_l1432_143260

def initial_temp : Int := -3
def temp_rise : Int := 6
def temp_drop : Int := 7

theorem final_temperature : 
  initial_temp + temp_rise - temp_drop = -4 :=
by sorry

end NUMINAMATH_CALUDE_final_temperature_l1432_143260


namespace NUMINAMATH_CALUDE_lines_perpendicular_to_plane_are_parallel_l1432_143223

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the perpendicular and parallel relations
variable (perpendicular : Line → Plane → Prop)
variable (parallel : Line → Line → Prop)

-- State the theorem
theorem lines_perpendicular_to_plane_are_parallel 
  (l m : Line) (α : Plane) : 
  perpendicular l α → perpendicular m α → parallel l m :=
sorry

end NUMINAMATH_CALUDE_lines_perpendicular_to_plane_are_parallel_l1432_143223


namespace NUMINAMATH_CALUDE_correct_calculation_l1432_143200

theorem correct_calculation (a : ℝ) : 2 * a^3 * 3 * a^5 = 6 * a^8 := by
  sorry

end NUMINAMATH_CALUDE_correct_calculation_l1432_143200


namespace NUMINAMATH_CALUDE_min_garden_cost_l1432_143201

/-- Represents a rectangular region in the garden -/
structure Region where
  length : ℝ
  width : ℝ

/-- Represents a type of flower with its price -/
structure Flower where
  price : ℝ

/-- The garden layout -/
def garden : List Region := [
  ⟨5, 2⟩, -- Region 1
  ⟨3, 5⟩, -- Region 2
  ⟨2, 4⟩, -- Region 3
  ⟨5, 4⟩, -- Region 4
  ⟨5, 3⟩  -- Region 5
]

/-- Available flowers with their prices -/
def flowers : List Flower := [
  ⟨1.20⟩, -- Asters
  ⟨1.70⟩, -- Begonias
  ⟨2.20⟩, -- Cannas
  ⟨2.70⟩, -- Dahlias
  ⟨3.20⟩  -- Freesias
]

/-- Calculate the area of a region -/
def area (r : Region) : ℝ := r.length * r.width

/-- Calculate the total area of the garden -/
def totalArea : ℝ := List.sum (List.map area garden)

/-- The main theorem: prove that the minimum cost is $152.60 -/
theorem min_garden_cost :
  ∃ (assignment : List (Region × Flower)),
    (List.length assignment = List.length garden) ∧
    (∀ r ∈ garden, ∃ f ∈ flowers, (r, f) ∈ assignment) ∧
    (List.sum (List.map (λ (r, f) => area r * f.price) assignment) = 152.60) ∧
    (∀ other_assignment : List (Region × Flower),
      (List.length other_assignment = List.length garden) →
      (∀ r ∈ garden, ∃ f ∈ flowers, (r, f) ∈ other_assignment) →
      List.sum (List.map (λ (r, f) => area r * f.price) other_assignment) ≥ 152.60) :=
by sorry

end NUMINAMATH_CALUDE_min_garden_cost_l1432_143201


namespace NUMINAMATH_CALUDE_problem_1_l1432_143238

theorem problem_1 (a b : ℝ) : -2 * (a^2 - 4*b) + 3 * (2*a^2 - 4*b) = 4*a^2 - 4*b := by
  sorry

end NUMINAMATH_CALUDE_problem_1_l1432_143238


namespace NUMINAMATH_CALUDE_sum_of_multiples_plus_eleven_l1432_143277

theorem sum_of_multiples_plus_eleven : 3 * 13 + 3 * 14 + 3 * 17 + 11 = 143 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_multiples_plus_eleven_l1432_143277


namespace NUMINAMATH_CALUDE_no_three_similar_piles_l1432_143239

theorem no_three_similar_piles : ¬∃ (x a b c : ℝ), 
  (x > 0 ∧ a > 0 ∧ b > 0 ∧ c > 0) ∧ 
  (a + b + c = x) ∧
  (a ≤ Real.sqrt 2 * b ∧ b ≤ Real.sqrt 2 * a) ∧
  (a ≤ Real.sqrt 2 * c ∧ c ≤ Real.sqrt 2 * a) ∧
  (b ≤ Real.sqrt 2 * c ∧ c ≤ Real.sqrt 2 * b) := by
  sorry

end NUMINAMATH_CALUDE_no_three_similar_piles_l1432_143239


namespace NUMINAMATH_CALUDE_incorrect_observation_value_l1432_143246

/-- Given a set of observations with known properties, determine the value of an incorrectly recorded observation. -/
theorem incorrect_observation_value
  (n : ℕ)  -- Total number of observations
  (original_mean : ℝ)  -- Original mean of all observations
  (correct_value : ℝ)  -- The correct value of the misrecorded observation
  (new_mean : ℝ)  -- The new mean after correcting the misrecorded observation
  (h_n : n = 50)  -- There are 50 observations
  (h_original_mean : original_mean = 36)  -- The original mean was 36
  (h_correct_value : correct_value = 30)  -- The correct value should have been 30
  (h_new_mean : new_mean = 36.5)  -- The new mean after correction is 36.5
  : ∃ (incorrect_value : ℝ), incorrect_value = 55 := by
  sorry

end NUMINAMATH_CALUDE_incorrect_observation_value_l1432_143246


namespace NUMINAMATH_CALUDE_quadratic_expression_value_l1432_143258

theorem quadratic_expression_value (x y : ℝ) 
  (eq1 : 3 * x + 2 * y = 6) 
  (eq2 : 2 * x + 3 * y = 8) : 
  13 * x^2 + 22 * x * y + 13 * y^2 = 98.08 := by
sorry

end NUMINAMATH_CALUDE_quadratic_expression_value_l1432_143258


namespace NUMINAMATH_CALUDE_farmer_animals_l1432_143219

theorem farmer_animals (cows pigs goats chickens ducks sheep : ℕ) : 
  pigs = 3 * cows →
  cows = goats + 7 →
  chickens = 2 * (cows + pigs) →
  2 * ducks = goats + chickens →
  sheep = cows + chickens + 5 →
  cows + pigs + goats + chickens + ducks + sheep = 346 →
  goats = 6 := by
sorry

end NUMINAMATH_CALUDE_farmer_animals_l1432_143219


namespace NUMINAMATH_CALUDE_number_ratio_l1432_143290

theorem number_ratio (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : y = 3 * x) (h4 : x + y = 124) :
  x / y = 1 / 3 := by
sorry

end NUMINAMATH_CALUDE_number_ratio_l1432_143290


namespace NUMINAMATH_CALUDE_smallest_scalene_triangle_perimeter_l1432_143237

/-- A function that checks if a number is prime -/
def isPrime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

/-- A function that checks if three numbers form a valid triangle -/
def isTriangle (a b c : ℕ) : Prop := a + b > c ∧ b + c > a ∧ c + a > b

/-- A function that checks if three numbers are consecutive odd primes -/
def areConsecutiveOddPrimes (a b c : ℕ) : Prop :=
  isPrime a ∧ isPrime b ∧ isPrime c ∧
  a < b ∧ b < c ∧
  b = a + 2 ∧ c = b + 2

theorem smallest_scalene_triangle_perimeter :
  ∀ p q r : ℕ,
    areConsecutiveOddPrimes p q r →
    isTriangle p q r →
    isPrime (p + q + r) →
    p + q + r ≥ 23 :=
sorry

end NUMINAMATH_CALUDE_smallest_scalene_triangle_perimeter_l1432_143237


namespace NUMINAMATH_CALUDE_integral_abs_x_squared_minus_x_l1432_143287

theorem integral_abs_x_squared_minus_x : ∫ x in (-1)..1, |x^2 - x| = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_integral_abs_x_squared_minus_x_l1432_143287


namespace NUMINAMATH_CALUDE_sqrt_98_plus_sqrt_32_l1432_143211

theorem sqrt_98_plus_sqrt_32 : Real.sqrt 98 + Real.sqrt 32 = 11 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_98_plus_sqrt_32_l1432_143211


namespace NUMINAMATH_CALUDE_six_digit_divisible_by_7_8_9_l1432_143224

theorem six_digit_divisible_by_7_8_9 :
  ∃ n : ℕ, 523000 ≤ n ∧ n ≤ 523999 ∧ 7 ∣ n ∧ 8 ∣ n ∧ 9 ∣ n := by
  sorry

end NUMINAMATH_CALUDE_six_digit_divisible_by_7_8_9_l1432_143224


namespace NUMINAMATH_CALUDE_function_property_l1432_143291

theorem function_property (f : ℤ → ℤ) 
  (h : ∀ (a b : ℤ), a ≠ 0 → b ≠ 0 → f (a * b) ≥ f a + f b) :
  ∀ (a : ℤ), a ≠ 0 → (∀ (n : ℕ), f (a ^ n) = n * f a) ↔ f (a ^ 2) = 2 * f a :=
sorry

end NUMINAMATH_CALUDE_function_property_l1432_143291


namespace NUMINAMATH_CALUDE_original_number_proof_l1432_143261

theorem original_number_proof : ∃ N : ℕ, N > 0 ∧ N - 28 ≡ 0 [MOD 87] ∧ ∀ M : ℕ, M > 0 ∧ M - 28 ≡ 0 [MOD 87] → M ≥ N :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_original_number_proof_l1432_143261


namespace NUMINAMATH_CALUDE_largest_number_problem_l1432_143228

theorem largest_number_problem (a b c d : ℕ) 
  (sum_abc : a + b + c = 222)
  (sum_abd : a + b + d = 208)
  (sum_acd : a + c + d = 197)
  (sum_bcd : b + c + d = 180) :
  max a (max b (max c d)) = 89 := by
  sorry

end NUMINAMATH_CALUDE_largest_number_problem_l1432_143228


namespace NUMINAMATH_CALUDE_polygon_diagonal_division_l1432_143227

/-- 
For an n-sided polygon, if a diagonal drawn from a vertex can divide it into 
at most 2023 triangles, then n = 2025.
-/
theorem polygon_diagonal_division (n : ℕ) : 
  (∃ (d : ℕ), d ≤ 2023 ∧ d = n - 2) → n = 2025 := by
  sorry

end NUMINAMATH_CALUDE_polygon_diagonal_division_l1432_143227


namespace NUMINAMATH_CALUDE_sandwich_toppings_combinations_l1432_143288

def choose (n k : ℕ) : ℕ := (Nat.factorial n) / ((Nat.factorial k) * Nat.factorial (n - k))

theorem sandwich_toppings_combinations :
  choose 9 3 = 84 := by sorry

end NUMINAMATH_CALUDE_sandwich_toppings_combinations_l1432_143288


namespace NUMINAMATH_CALUDE_smallest_positive_integer_with_given_remainders_l1432_143209

theorem smallest_positive_integer_with_given_remainders :
  ∃ b : ℕ, b > 0 ∧ 
    b % 5 = 4 ∧ 
    b % 7 = 6 ∧ 
    b % 11 = 10 ∧ 
    (∀ c : ℕ, c > 0 ∧ c % 5 = 4 ∧ c % 7 = 6 ∧ c % 11 = 10 → b ≤ c) ∧
    b = 384 := by
  sorry

end NUMINAMATH_CALUDE_smallest_positive_integer_with_given_remainders_l1432_143209


namespace NUMINAMATH_CALUDE_broken_flagpole_l1432_143285

theorem broken_flagpole (h : ℝ) (d : ℝ) (x : ℝ) 
  (height_cond : h = 10)
  (distance_cond : d = 4) :
  (x^2 + d^2 = (h - x)^2) → x = 2 * Real.sqrt 22 :=
by sorry

end NUMINAMATH_CALUDE_broken_flagpole_l1432_143285


namespace NUMINAMATH_CALUDE_libby_igloo_bricks_l1432_143229

/-- Calculates the total number of bricks in an igloo -/
def igloo_bricks (total_rows : ℕ) (bottom_bricks_per_row : ℕ) (top_bricks_per_row : ℕ) : ℕ :=
  let bottom_rows := total_rows / 2
  let top_rows := total_rows - bottom_rows
  bottom_rows * bottom_bricks_per_row + top_rows * top_bricks_per_row

/-- Proves that Libby's igloo uses 100 bricks -/
theorem libby_igloo_bricks :
  igloo_bricks 10 12 8 = 100 := by
  sorry

end NUMINAMATH_CALUDE_libby_igloo_bricks_l1432_143229


namespace NUMINAMATH_CALUDE_smallest_gcd_qr_l1432_143281

theorem smallest_gcd_qr (p q r : ℕ+) (h1 : Nat.gcd p q = 240) (h2 : Nat.gcd p r = 540) :
  ∃ (q' r' : ℕ+), Nat.gcd q'.val r'.val = 60 ∧
    ∀ (q'' r'' : ℕ+), Nat.gcd q''.val r''.val ≥ 60 :=
sorry

end NUMINAMATH_CALUDE_smallest_gcd_qr_l1432_143281


namespace NUMINAMATH_CALUDE_intersection_points_form_line_l1432_143225

theorem intersection_points_form_line (s : ℝ) : 
  ∃ (x y : ℝ), 2*x + 3*y = 8*s + 4 ∧ 3*x - 4*y = 9*s - 3 → 
  y = (20/59)*x + 60/59 :=
sorry

end NUMINAMATH_CALUDE_intersection_points_form_line_l1432_143225


namespace NUMINAMATH_CALUDE_d_72_eq_22_l1432_143221

/-- D(n) is the number of ways to write n as an ordered product of integers greater than 1 -/
def D (n : ℕ) : ℕ := sorry

/-- The main theorem: D(72) = 22 -/
theorem d_72_eq_22 : D 72 = 22 := by sorry

end NUMINAMATH_CALUDE_d_72_eq_22_l1432_143221


namespace NUMINAMATH_CALUDE_scientific_notation_460_billion_l1432_143269

theorem scientific_notation_460_billion :
  (460 * 10^9 : ℝ) = 4.6 * 10^11 := by sorry

end NUMINAMATH_CALUDE_scientific_notation_460_billion_l1432_143269


namespace NUMINAMATH_CALUDE_exponent_addition_l1432_143276

theorem exponent_addition (a : ℝ) : a^3 * a^4 = a^7 := by
  sorry

end NUMINAMATH_CALUDE_exponent_addition_l1432_143276


namespace NUMINAMATH_CALUDE_dartboard_angle_l1432_143271

theorem dartboard_angle (p : ℝ) (θ : ℝ) : 
  0 < p → p < 1 → 0 ≤ θ → θ ≤ 360 →
  (p = θ / 360) → (p = 1 / 8) → θ = 45 := by
  sorry

end NUMINAMATH_CALUDE_dartboard_angle_l1432_143271


namespace NUMINAMATH_CALUDE_product_inequality_find_a_l1432_143273

-- Part I
theorem product_inequality (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 1) :
  (1 + 1/a) * (1 + 1/b) ≥ 9 := by sorry

-- Part II
theorem find_a (a : ℝ) (h : ∀ x, |x + 3| - |x - a| ≥ 2 ↔ x ≥ 1) :
  a = 2 := by sorry

end NUMINAMATH_CALUDE_product_inequality_find_a_l1432_143273


namespace NUMINAMATH_CALUDE_rectangle_area_l1432_143297

/-- Given a rectangle with width w and length L, where L = w^2 and L + w = 25,
    prove that the area of the rectangle is (√101 - 1)^3 / 8 square inches. -/
theorem rectangle_area (w L : ℝ) (h1 : L = w^2) (h2 : L + w = 25) :
  w * L = ((Real.sqrt 101 - 1)^3) / 8 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_l1432_143297


namespace NUMINAMATH_CALUDE_symmetry_sum_for_17gon_l1432_143283

/-- The number of sides in our regular polygon -/
def n : ℕ := 17

/-- The number of lines of symmetry in a regular n-gon -/
def L (n : ℕ) : ℕ := n

/-- The smallest positive angle of rotational symmetry (in degrees) for a regular n-gon -/
def R (n : ℕ) : ℚ := 360 / n

/-- Theorem: For a regular 17-gon, the sum of its number of lines of symmetry
    and its smallest positive angle of rotational symmetry (in degrees) is 649/17 -/
theorem symmetry_sum_for_17gon : L n + R n = 649 / 17 := by
  sorry

end NUMINAMATH_CALUDE_symmetry_sum_for_17gon_l1432_143283


namespace NUMINAMATH_CALUDE_smallest_non_odd_unit_digit_l1432_143268

def OddUnitDigits : Set ℕ := {1, 3, 5, 7, 9}

def IsOdd (n : ℕ) : Prop := n % 2 = 1

def UnitsDigit (n : ℕ) : ℕ := n % 10

theorem smallest_non_odd_unit_digit :
  (∀ n : ℕ, IsOdd n → UnitsDigit n ∈ OddUnitDigits) →
  (∀ d : ℕ, d < 0 → d ∉ OddUnitDigits) →
  (∀ d : ℕ, 0 < d → d < 10 → d ∉ OddUnitDigits → 0 < d) →
  (0 ∉ OddUnitDigits ∧ ∀ d : ℕ, d < 10 → d ∉ OddUnitDigits → 0 ≤ d) :=
by sorry

end NUMINAMATH_CALUDE_smallest_non_odd_unit_digit_l1432_143268
