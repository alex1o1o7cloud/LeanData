import Mathlib

namespace NUMINAMATH_CALUDE_quadratic_root_problem_l784_78434

theorem quadratic_root_problem (k : ℝ) : 
  (2 : ℝ)^2 + 2 - k = 0 → (-3 : ℝ)^2 + (-3) - k = 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_root_problem_l784_78434


namespace NUMINAMATH_CALUDE_hiker_first_day_distance_l784_78419

/-- A hiker's three-day journey --/
def HikersJourney (h : ℝ) : Prop :=
  let d1 := 3 * h  -- Distance on day 1
  let d2 := 4 * (h - 1)  -- Distance on day 2
  let d3 := 5 * 6  -- Distance on day 3
  d1 + d2 + d3 = 68  -- Total distance

/-- The hiker walked 18 miles on the first day --/
theorem hiker_first_day_distance :
  ∃ h : ℝ, HikersJourney h ∧ 3 * h = 18 :=
sorry

end NUMINAMATH_CALUDE_hiker_first_day_distance_l784_78419


namespace NUMINAMATH_CALUDE_quadratic_roots_sum_l784_78482

theorem quadratic_roots_sum (a b : ℝ) : 
  a ≠ b → 
  a^2 - 8*a + 5 = 0 → 
  b^2 - 8*b + 5 = 0 → 
  (b - 1) / (a - 1) + (a - 1) / (b - 1) = -20 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_sum_l784_78482


namespace NUMINAMATH_CALUDE_perpendicular_line_plane_condition_l784_78457

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the perpendicular relation between lines and between a line and a plane
variable (perp_line : Line → Line → Prop)
variable (perp_plane : Line → Plane → Prop)

-- Define the subset relation between a line and a plane
variable (subset : Line → Plane → Prop)

-- State the theorem
theorem perpendicular_line_plane_condition 
  (a l : Line) (α : Plane) (h_subset : subset a α) :
  (perp_plane l α → perp_line l a) ∧ 
  ∃ (l' : Line), perp_line l' a ∧ ¬perp_plane l' α :=
sorry

end NUMINAMATH_CALUDE_perpendicular_line_plane_condition_l784_78457


namespace NUMINAMATH_CALUDE_range_of_a_l784_78475

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, |x - 1| + |x + 1| ≥ 3 * a) →
  (∀ x y : ℝ, x < y → (2 * a - 1) ^ x > (2 * a - 1) ^ y) →
  1/2 < a ∧ a ≤ 2/3 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l784_78475


namespace NUMINAMATH_CALUDE_no_multiple_of_five_l784_78490

theorem no_multiple_of_five (n : ℕ) : 
  2 ≤ n → n ≤ 100 → ¬(5 ∣ (2 + 5*n + n^2 + 5*n^3 + 2*n^4)) :=
by sorry

end NUMINAMATH_CALUDE_no_multiple_of_five_l784_78490


namespace NUMINAMATH_CALUDE_large_square_area_l784_78463

theorem large_square_area (s : ℝ) (S : ℝ) 
  (h1 : S = s + 20)
  (h2 : S^2 - s^2 = 880) :
  S^2 = 1024 := by
  sorry

end NUMINAMATH_CALUDE_large_square_area_l784_78463


namespace NUMINAMATH_CALUDE_unique_solution_linear_system_l784_78476

theorem unique_solution_linear_system :
  ∃! (x y : ℝ), (2 * x - y = 1) ∧ (x + y = 2) :=
by
  -- The proof would go here
  sorry

end NUMINAMATH_CALUDE_unique_solution_linear_system_l784_78476


namespace NUMINAMATH_CALUDE_combined_sixth_grade_percent_l784_78413

-- Define the schools
structure School where
  name : String
  total_students : ℕ
  sixth_grade_percent : ℚ

-- Define the given data
def pineview : School := ⟨"Pineview", 150, 15/100⟩
def oakridge : School := ⟨"Oakridge", 180, 17/100⟩
def maplewood : School := ⟨"Maplewood", 170, 15/100⟩

def schools : List School := [pineview, oakridge, maplewood]

-- Function to calculate the number of 6th graders in a school
def sixth_graders (s : School) : ℚ :=
  s.total_students * s.sixth_grade_percent

-- Function to calculate the total number of students
def total_students (schools : List School) : ℕ :=
  schools.foldl (fun acc s => acc + s.total_students) 0

-- Function to calculate the total number of 6th graders
def total_sixth_graders (schools : List School) : ℚ :=
  schools.foldl (fun acc s => acc + sixth_graders s) 0

-- Theorem statement
theorem combined_sixth_grade_percent :
  (total_sixth_graders schools) / (total_students schools : ℚ) = 1572 / 10000 := by
  sorry

end NUMINAMATH_CALUDE_combined_sixth_grade_percent_l784_78413


namespace NUMINAMATH_CALUDE_tens_digit_of_23_pow_1987_l784_78478

theorem tens_digit_of_23_pow_1987 :
  (23^1987 / 10) % 10 = 4 := by
  sorry

end NUMINAMATH_CALUDE_tens_digit_of_23_pow_1987_l784_78478


namespace NUMINAMATH_CALUDE_complement_A_union_B_A_inter_complement_B_l784_78404

-- Define the sets A and B
def A : Set ℝ := {x | 0 ≤ x ∧ x < 8}
def B : Set ℝ := {x | 1 < x ∧ x < 9}

-- Theorem for the first part
theorem complement_A_union_B : 
  (Set.univ \ A) ∪ B = {x | x < 0 ∨ x > 1} := by sorry

-- Theorem for the second part
theorem A_inter_complement_B : 
  A ∩ (Set.univ \ B) = {x | 0 ≤ x ∧ x ≤ 1} := by sorry

end NUMINAMATH_CALUDE_complement_A_union_B_A_inter_complement_B_l784_78404


namespace NUMINAMATH_CALUDE_not_p_sufficient_not_necessary_for_not_q_l784_78406

-- Define p and q as predicates on real numbers
def p (x : ℝ) : Prop := x < -1 ∨ x > 1
def q (x : ℝ) : Prop := x < -2 ∨ x > 1

-- Theorem statement
theorem not_p_sufficient_not_necessary_for_not_q :
  (∀ x, ¬(p x) → ¬(q x)) ∧ 
  ¬(∀ x, ¬(q x) → ¬(p x)) :=
sorry

end NUMINAMATH_CALUDE_not_p_sufficient_not_necessary_for_not_q_l784_78406


namespace NUMINAMATH_CALUDE_log_drift_theorem_l784_78470

/-- The time it takes for a log to drift downstream -/
def log_drift_time (downstream_time upstream_time : ℝ) : ℝ :=
  6 * (upstream_time - downstream_time)

/-- Theorem: Given the downstream and upstream travel times of a boat, 
    the time for a log to drift downstream is 12 hours -/
theorem log_drift_theorem (downstream_time upstream_time : ℝ) 
  (h1 : downstream_time = 2)
  (h2 : upstream_time = 3) : 
  log_drift_time downstream_time upstream_time = 12 := by
  sorry

end NUMINAMATH_CALUDE_log_drift_theorem_l784_78470


namespace NUMINAMATH_CALUDE_complex_equation_modulus_l784_78423

theorem complex_equation_modulus : ∀ (x y : ℝ),
  (Complex.I * (x + 2 * Complex.I) = y - Complex.I) →
  Complex.abs (x - y * Complex.I) = Real.sqrt 5 := by sorry

end NUMINAMATH_CALUDE_complex_equation_modulus_l784_78423


namespace NUMINAMATH_CALUDE_coin_problem_l784_78494

theorem coin_problem (p n : ℕ) : 
  p + n = 32 →  -- Total number of coins
  p + 5 * n = 100 →  -- Total value in cents
  n = 17 :=
by sorry

end NUMINAMATH_CALUDE_coin_problem_l784_78494


namespace NUMINAMATH_CALUDE_fourth_student_id_l784_78487

/-- Represents a systematic sampling of students -/
structure SystematicSample where
  total_students : ℕ
  sample_size : ℕ
  first_id : ℕ
  step : ℕ

/-- Creates a systematic sample given the total number of students and sample size -/
def create_systematic_sample (total : ℕ) (size : ℕ) : SystematicSample :=
  { total_students := total
  , sample_size := size
  , first_id := 3  -- Given in the problem
  , step := (total - 2) / size }

/-- Checks if a given ID is in the systematic sample -/
def is_in_sample (sample : SystematicSample) (id : ℕ) : Prop :=
  ∃ k : ℕ, id = sample.first_id + k * sample.step ∧ k < sample.sample_size

/-- Main theorem: If 3, 29, and 42 are in the sample, then 16 is also in the sample -/
theorem fourth_student_id (sample : SystematicSample)
    (h_total : sample.total_students = 54)
    (h_size : sample.sample_size = 4)
    (h_3 : is_in_sample sample 3)
    (h_29 : is_in_sample sample 29)
    (h_42 : is_in_sample sample 42) :
    is_in_sample sample 16 := by
  sorry

end NUMINAMATH_CALUDE_fourth_student_id_l784_78487


namespace NUMINAMATH_CALUDE_derivative_f_at_zero_l784_78426

-- Define the function f
def f (x : ℝ) : ℝ := (2*x + 1)^3

-- State the theorem
theorem derivative_f_at_zero : 
  deriv f 0 = 6 := by sorry

end NUMINAMATH_CALUDE_derivative_f_at_zero_l784_78426


namespace NUMINAMATH_CALUDE_coffee_per_donut_l784_78460

/-- Proves that the number of ounces of coffee needed per donut is 2, given the specified conditions. -/
theorem coffee_per_donut (ounces_per_pot : ℕ) (cost_per_pot : ℕ) (dozen_donuts : ℕ) (total_coffee_cost : ℕ) :
  ounces_per_pot = 12 →
  cost_per_pot = 3 →
  dozen_donuts = 3 →
  total_coffee_cost = 18 →
  (total_coffee_cost / cost_per_pot * ounces_per_pot) / (dozen_donuts * 12) = 2 :=
by sorry

end NUMINAMATH_CALUDE_coffee_per_donut_l784_78460


namespace NUMINAMATH_CALUDE_sam_and_joan_books_l784_78455

/-- Given that Sam has 110 books and Joan has 102 books, prove that they have 212 books together. -/
theorem sam_and_joan_books : 
  let sam_books : ℕ := 110
  let joan_books : ℕ := 102
  sam_books + joan_books = 212 :=
by sorry

end NUMINAMATH_CALUDE_sam_and_joan_books_l784_78455


namespace NUMINAMATH_CALUDE_range_of_a_l784_78456

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, x^2 - 4*a*x + 3*a^2 < 0 → |x - 3| > 1) ∧ 
  (∃ x : ℝ, |x - 3| > 1 ∧ x^2 - 4*a*x + 3*a^2 ≥ 0) ∧
  (a > 0) →
  a ≥ 4 ∨ (0 < a ∧ a ≤ 2/3) :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l784_78456


namespace NUMINAMATH_CALUDE_alex_martin_games_l784_78449

/-- The number of players in the four-square league --/
def total_players : ℕ := 12

/-- The number of players in each game --/
def players_per_game : ℕ := 6

/-- The number of players to be chosen after Alex and Martin are included --/
def players_to_choose : ℕ := players_per_game - 2

/-- The number of remaining players after Alex and Martin are excluded --/
def remaining_players : ℕ := total_players - 2

/-- The number of times Alex plays in the same game as Martin --/
def games_together : ℕ := Nat.choose remaining_players players_to_choose

theorem alex_martin_games :
  games_together = 210 :=
sorry

end NUMINAMATH_CALUDE_alex_martin_games_l784_78449


namespace NUMINAMATH_CALUDE_sector_central_angle_l784_78405

/-- Given a sector with circumference 12 and area 8, its central angle in radians is either 1 or 4. -/
theorem sector_central_angle (r : ℝ) (l : ℝ) (α : ℝ) : 
  l + 2 * r = 12 → 
  1 / 2 * l * r = 8 → 
  α = l / r → 
  α = 1 ∨ α = 4 := by
sorry

end NUMINAMATH_CALUDE_sector_central_angle_l784_78405


namespace NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l784_78451

theorem imaginary_part_of_complex_fraction (z : ℂ) : z = (1 + 3*I) / (1 - I) → z.im = 2 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l784_78451


namespace NUMINAMATH_CALUDE_quadratic_vertex_coordinates_l784_78439

-- Define the quadratic function
def f (x : ℝ) : ℝ := -3 * x^2 - 6 * x + 5

-- Define the vertex coordinates
def vertex : ℝ × ℝ := (-1, 8)

-- Theorem statement
theorem quadratic_vertex_coordinates :
  (∀ x : ℝ, f x ≤ f (vertex.1)) ∧ f (vertex.1) = vertex.2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_vertex_coordinates_l784_78439


namespace NUMINAMATH_CALUDE_iphone_price_reduction_l784_78446

/-- 
Calculates the final price of an item after two consecutive price reductions.
-/
theorem iphone_price_reduction (initial_price : ℝ) 
  (first_reduction : ℝ) (second_reduction : ℝ) :
  initial_price = 1000 →
  first_reduction = 0.1 →
  second_reduction = 0.2 →
  initial_price * (1 - first_reduction) * (1 - second_reduction) = 720 := by
sorry

end NUMINAMATH_CALUDE_iphone_price_reduction_l784_78446


namespace NUMINAMATH_CALUDE_arithmetic_sequence_ratio_l784_78466

def arithmeticSum (a1 : ℚ) (d : ℚ) (an : ℚ) : ℚ :=
  let n := (an - a1) / d + 1
  n * (a1 + an) / 2

theorem arithmetic_sequence_ratio : 
  let numerator := arithmeticSum 3 3 39
  let denominator := arithmeticSum 4 4 64
  numerator / denominator = 1 / 2 := by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_ratio_l784_78466


namespace NUMINAMATH_CALUDE_recreation_area_tents_l784_78412

/-- Represents the number of tents in different parts of the campsite -/
structure Campsite where
  north : ℕ
  east : ℕ
  center : ℕ
  south : ℕ

/-- Calculates the total number of tents in the campsite -/
def total_tents (c : Campsite) : ℕ :=
  c.north + c.east + c.center + c.south

/-- Theorem stating the total number of tents in the recreation area -/
theorem recreation_area_tents : ∃ (c : Campsite), 
  c.north = 100 ∧ 
  c.east = 2 * c.north ∧ 
  c.center = 4 * c.north ∧ 
  c.south = 200 ∧ 
  total_tents c = 900 := by
  sorry


end NUMINAMATH_CALUDE_recreation_area_tents_l784_78412


namespace NUMINAMATH_CALUDE_number_of_teachers_l784_78444

/-- Represents the number of students at Queen Middle School -/
def total_students : ℕ := 1500

/-- Represents the number of classes each student takes per day -/
def classes_per_student : ℕ := 6

/-- Represents the number of classes each teacher teaches -/
def classes_per_teacher : ℕ := 5

/-- Represents the number of students in each class -/
def students_per_class : ℕ := 25

/-- Represents the number of teachers in each class -/
def teachers_per_class : ℕ := 1

/-- Theorem stating that the number of teachers at Queen Middle School is 72 -/
theorem number_of_teachers : 
  (total_students * classes_per_student) / students_per_class / classes_per_teacher = 72 := by
  sorry

end NUMINAMATH_CALUDE_number_of_teachers_l784_78444


namespace NUMINAMATH_CALUDE_freshman_class_size_l784_78441

theorem freshman_class_size :
  ∃! n : ℕ, 0 < n ∧ n < 450 ∧ n % 19 = 18 ∧ n % 17 = 10 ∧ n = 265 := by
  sorry

end NUMINAMATH_CALUDE_freshman_class_size_l784_78441


namespace NUMINAMATH_CALUDE_course_selection_schemes_l784_78472

theorem course_selection_schemes (pe art : ℕ) (total_courses : ℕ) : 
  pe = 4 → art = 4 → total_courses = pe + art →
  (Nat.choose pe 1 * Nat.choose art 1) + 
  (Nat.choose pe 2 * Nat.choose art 1 + Nat.choose pe 1 * Nat.choose art 2) = 64 :=
by sorry

end NUMINAMATH_CALUDE_course_selection_schemes_l784_78472


namespace NUMINAMATH_CALUDE_total_seashells_l784_78495

-- Define the variables
def seashells_given_to_tom : ℕ := 49
def seashells_left_with_mike : ℕ := 13

-- Define the theorem
theorem total_seashells :
  seashells_given_to_tom + seashells_left_with_mike = 62 := by
  sorry

end NUMINAMATH_CALUDE_total_seashells_l784_78495


namespace NUMINAMATH_CALUDE_five_circles_common_point_l784_78422

-- Define a circle type
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define a point type
structure Point where
  x : ℝ
  y : ℝ

-- Define a function to check if a point is on a circle
def pointOnCircle (p : Point) (c : Circle) : Prop :=
  (p.x - c.center.1)^2 + (p.y - c.center.2)^2 = c.radius^2

-- Define a function to check if four circles pass through a single point
def fourCirclesCommonPoint (c1 c2 c3 c4 : Circle) : Prop :=
  ∃ p : Point, pointOnCircle p c1 ∧ pointOnCircle p c2 ∧ pointOnCircle p c3 ∧ pointOnCircle p c4

-- Theorem statement
theorem five_circles_common_point 
  (c1 c2 c3 c4 c5 : Circle) 
  (h1234 : fourCirclesCommonPoint c1 c2 c3 c4)
  (h1235 : fourCirclesCommonPoint c1 c2 c3 c5)
  (h1245 : fourCirclesCommonPoint c1 c2 c4 c5)
  (h1345 : fourCirclesCommonPoint c1 c3 c4 c5)
  (h2345 : fourCirclesCommonPoint c2 c3 c4 c5) :
  ∃ p : Point, pointOnCircle p c1 ∧ pointOnCircle p c2 ∧ pointOnCircle p c3 ∧ pointOnCircle p c4 ∧ pointOnCircle p c5 :=
by
  sorry

end NUMINAMATH_CALUDE_five_circles_common_point_l784_78422


namespace NUMINAMATH_CALUDE_compound_interest_principal_exists_l784_78443

theorem compound_interest_principal_exists : ∃ (P r : ℝ), 
  P > 0 ∧ r > 0 ∧ 
  P * (1 + r)^2 = 8800 ∧ 
  P * (1 + r)^3 = 9261 := by
sorry

end NUMINAMATH_CALUDE_compound_interest_principal_exists_l784_78443


namespace NUMINAMATH_CALUDE_sum_of_powers_of_i_equals_zero_l784_78486

-- Define the imaginary unit i
def i : ℂ := Complex.I

-- State the theorem
theorem sum_of_powers_of_i_equals_zero :
  i^1234 + i^1235 + i^1236 + i^1237 = 0 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_powers_of_i_equals_zero_l784_78486


namespace NUMINAMATH_CALUDE_absolute_value_equality_l784_78473

theorem absolute_value_equality (a : ℝ) : 
  |a| = |5 + 1/3| → a = 5 + 1/3 ∨ a = -(5 + 1/3) := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_equality_l784_78473


namespace NUMINAMATH_CALUDE_sqrt_undefined_range_l784_78469

theorem sqrt_undefined_range (a : ℝ) : ¬ (∃ x : ℝ, x ^ 2 = 2 * a - 1) → a < 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_undefined_range_l784_78469


namespace NUMINAMATH_CALUDE_prob_black_then_red_standard_deck_l784_78417

/-- A deck of cards with black cards, red cards, and jokers. -/
structure Deck :=
  (total : ℕ)
  (black : ℕ)
  (red : ℕ)
  (jokers : ℕ)

/-- The probability of drawing a black card first and a red card second from a given deck. -/
def prob_black_then_red (d : Deck) : ℚ :=
  (d.black : ℚ) / d.total * (d.red : ℚ) / (d.total - 1)

/-- The standard deck with 54 cards including jokers. -/
def standard_deck : Deck :=
  { total := 54
  , black := 26
  , red := 26
  , jokers := 2 }

theorem prob_black_then_red_standard_deck :
  prob_black_then_red standard_deck = 338 / 1431 := by
  sorry

end NUMINAMATH_CALUDE_prob_black_then_red_standard_deck_l784_78417


namespace NUMINAMATH_CALUDE_remainder_sum_mod_five_l784_78496

theorem remainder_sum_mod_five (f y : ℤ) 
  (hf : f % 5 = 3) 
  (hy : y % 5 = 4) : 
  (f + y) % 5 = 2 :=
by sorry

end NUMINAMATH_CALUDE_remainder_sum_mod_five_l784_78496


namespace NUMINAMATH_CALUDE_sum_of_divisors_3600_l784_78461

/-- Sum of divisors function -/
def sum_of_divisors (n : ℕ) : ℕ := sorry

/-- Theorem: If the sum of divisors of 2^i * 3^j * 5^k is 3600, then i + j + k = 7 -/
theorem sum_of_divisors_3600 (i j k : ℕ) : 
  sum_of_divisors (2^i * 3^j * 5^k) = 3600 → i + j + k = 7 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_divisors_3600_l784_78461


namespace NUMINAMATH_CALUDE_negation_of_universal_proposition_l784_78411

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, x^2 + 2*x + 2 > 0) ↔ (∃ x : ℝ, x^2 + 2*x + 2 ≤ 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_universal_proposition_l784_78411


namespace NUMINAMATH_CALUDE_sequence_constant_l784_78485

theorem sequence_constant (a : ℕ → ℤ) (d : ℤ) :
  (∀ n : ℕ, Nat.Prime (Int.natAbs (a n))) →
  (∀ n : ℕ, a (n + 2) = a (n + 1) + a n + d) →
  (∀ n : ℕ, a n = 0) := by
sorry

end NUMINAMATH_CALUDE_sequence_constant_l784_78485


namespace NUMINAMATH_CALUDE_isosceles_triangle_perimeter_isosceles_triangle_perimeter_proof_l784_78421

/-- An isosceles triangle with side lengths 5 and 10 has a perimeter of 25 -/
theorem isosceles_triangle_perimeter : ℝ → ℝ → ℝ → Prop :=
  fun a b c =>
    (a = 5 ∨ a = 10) ∧ 
    (b = 5 ∨ b = 10) ∧ 
    (c = 5 ∨ c = 10) ∧ 
    (a = b ∨ b = c ∨ a = c) ∧ 
    (a + b > c ∧ b + c > a ∧ a + c > b) →
    a + b + c = 25

theorem isosceles_triangle_perimeter_proof : ∃ a b c : ℝ, isosceles_triangle_perimeter a b c := by
  sorry

end NUMINAMATH_CALUDE_isosceles_triangle_perimeter_isosceles_triangle_perimeter_proof_l784_78421


namespace NUMINAMATH_CALUDE_hcf_problem_l784_78479

theorem hcf_problem (a b : ℕ+) : 
  (∃ h : ℕ+, Nat.lcm a b = h * 13 * 14) →
  max a b = 322 →
  Nat.gcd a b = 7 := by
sorry

end NUMINAMATH_CALUDE_hcf_problem_l784_78479


namespace NUMINAMATH_CALUDE_local_arts_students_percentage_l784_78499

/-- Proves that the percentage of local arts students is 50% --/
theorem local_arts_students_percentage
  (total_arts : ℕ)
  (total_science : ℕ)
  (total_commerce : ℕ)
  (science_local_percentage : ℚ)
  (commerce_local_percentage : ℚ)
  (total_local_percentage : ℕ)
  (h_total_arts : total_arts = 400)
  (h_total_science : total_science = 100)
  (h_total_commerce : total_commerce = 120)
  (h_science_local : science_local_percentage = 25/100)
  (h_commerce_local : commerce_local_percentage = 85/100)
  (h_total_local : total_local_percentage = 327)
  : ∃ (arts_local_percentage : ℚ),
    arts_local_percentage = 50/100 ∧
    arts_local_percentage * total_arts +
    science_local_percentage * total_science +
    commerce_local_percentage * total_commerce =
    total_local_percentage := by
  sorry


end NUMINAMATH_CALUDE_local_arts_students_percentage_l784_78499


namespace NUMINAMATH_CALUDE_remainder_876539_mod_7_l784_78484

theorem remainder_876539_mod_7 : 876539 % 7 = 6 := by
  sorry

end NUMINAMATH_CALUDE_remainder_876539_mod_7_l784_78484


namespace NUMINAMATH_CALUDE_thirty_people_handshakes_l784_78459

/-- The number of handshakes in a group of n people where each person shakes hands
    with every other person exactly once. -/
def handshakes (n : ℕ) : ℕ := n * (n - 1) / 2

/-- Theorem stating that in a group of 30 people, the total number of handshakes is 435. -/
theorem thirty_people_handshakes :
  handshakes 30 = 435 := by sorry

end NUMINAMATH_CALUDE_thirty_people_handshakes_l784_78459


namespace NUMINAMATH_CALUDE_pizza_pooling_advantage_l784_78401

/-- Represents the size and price of a pizza --/
structure Pizza where
  side : ℕ
  price : ℕ

/-- Calculates the area of a square pizza --/
def pizzaArea (p : Pizza) : ℕ := p.side * p.side

/-- Represents the pizza options and money available --/
structure PizzaShop where
  smallPizza : Pizza
  largePizza : Pizza
  moneyPerPerson : ℕ
  numPeople : ℕ

/-- Calculates the maximum area of pizza that can be bought individually --/
def maxIndividualArea (shop : PizzaShop) : ℕ :=
  let smallArea := (shop.moneyPerPerson / shop.smallPizza.price) * pizzaArea shop.smallPizza
  let largeArea := (shop.moneyPerPerson / shop.largePizza.price) * pizzaArea shop.largePizza
  max smallArea largeArea * shop.numPeople

/-- Calculates the maximum area of pizza that can be bought by pooling money --/
def maxPooledArea (shop : PizzaShop) : ℕ :=
  let totalMoney := shop.moneyPerPerson * shop.numPeople
  let smallArea := (totalMoney / shop.smallPizza.price) * pizzaArea shop.smallPizza
  let largeArea := (totalMoney / shop.largePizza.price) * pizzaArea shop.largePizza
  max smallArea largeArea

theorem pizza_pooling_advantage (shop : PizzaShop) 
    (h1 : shop.smallPizza = ⟨6, 10⟩)
    (h2 : shop.largePizza = ⟨9, 20⟩)
    (h3 : shop.moneyPerPerson = 30)
    (h4 : shop.numPeople = 2) :
  maxPooledArea shop - maxIndividualArea shop = 27 := by
  sorry


end NUMINAMATH_CALUDE_pizza_pooling_advantage_l784_78401


namespace NUMINAMATH_CALUDE_drummer_drum_stick_usage_l784_78477

/-- Calculates the total number of drum stick sets used by a drummer over multiple shows. -/
def total_drum_stick_sets (sets_per_show : ℕ) (tossed_sets : ℕ) (num_shows : ℕ) : ℕ :=
  (sets_per_show + tossed_sets) * num_shows

/-- Theorem stating that a drummer using 5 sets per show, tossing 6 sets, for 30 shows uses 330 sets in total. -/
theorem drummer_drum_stick_usage :
  total_drum_stick_sets 5 6 30 = 330 := by
  sorry

end NUMINAMATH_CALUDE_drummer_drum_stick_usage_l784_78477


namespace NUMINAMATH_CALUDE_geometric_sequence_product_relation_l784_78438

/-- Represents a geometric sequence with 3n terms -/
structure GeometricSequence (α : Type*) [CommRing α] where
  n : ℕ
  terms : Fin (3 * n) → α
  is_geometric : ∀ i j k, i < j → j < k → terms j ^ 2 = terms i * terms k

/-- The product of n consecutive terms in a geometric sequence -/
def product_n_terms {α : Type*} [CommRing α] (seq : GeometricSequence α) (start : ℕ) : α :=
  (List.range seq.n).foldl (λ acc i => acc * seq.terms ⟨start * seq.n + i, sorry⟩) 1

/-- Theorem: In a geometric sequence with 3n terms, if A is the product of the first n terms,
    B is the product of the next n terms, and C is the product of the last n terms, then AC = B² -/
theorem geometric_sequence_product_relation {α : Type*} [CommRing α] (seq : GeometricSequence α) :
  let A := product_n_terms seq 0
  let B := product_n_terms seq 1
  let C := product_n_terms seq 2
  A * C = B ^ 2 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_product_relation_l784_78438


namespace NUMINAMATH_CALUDE_midpoint_segment_length_is_three_l784_78462

/-- A trapezoid with specific properties -/
structure Trapezoid where
  /-- The sum of the two base angles is 90° -/
  base_angles_sum : ℝ
  /-- The length of the upper base -/
  upper_base : ℝ
  /-- The length of the lower base -/
  lower_base : ℝ
  /-- The sum of the two base angles is 90° -/
  base_angles_sum_eq : base_angles_sum = 90
  /-- The length of the upper base is 5 -/
  upper_base_eq : upper_base = 5
  /-- The length of the lower base is 11 -/
  lower_base_eq : lower_base = 11

/-- The length of the segment connecting the midpoints of the two bases -/
def midpoint_segment_length (t : Trapezoid) : ℝ := 3

/-- Theorem: The length of the segment connecting the midpoints of the two bases is 3 -/
theorem midpoint_segment_length_is_three (t : Trapezoid) :
  midpoint_segment_length t = 3 := by
  sorry

#check midpoint_segment_length_is_three

end NUMINAMATH_CALUDE_midpoint_segment_length_is_three_l784_78462


namespace NUMINAMATH_CALUDE_min_sum_of_squares_l784_78400

theorem min_sum_of_squares (x y : ℝ) (h : (x + 3) * (y - 3) = 0) :
  ∃ (m : ℝ), m = 18 ∧ ∀ (a b : ℝ), (a + 3) * (b - 3) = 0 → x^2 + y^2 ≥ m :=
by sorry

end NUMINAMATH_CALUDE_min_sum_of_squares_l784_78400


namespace NUMINAMATH_CALUDE_tangent_slope_three_cubic_l784_78436

theorem tangent_slope_three_cubic (x : ℝ) : 
  (∃ y : ℝ, y = x^3 ∧ (3 * x^2 = 3)) ↔ (x = 1 ∨ x = -1) := by
  sorry

end NUMINAMATH_CALUDE_tangent_slope_three_cubic_l784_78436


namespace NUMINAMATH_CALUDE_sum_of_segments_is_81_l784_78440

/-- Represents the structure of triangles within the larger triangle -/
structure TriangleStructure where
  large_perimeter : ℝ
  small_side_length : ℝ
  small_triangle_count : ℕ

/-- The specific triangle structure from the problem -/
def problem_structure : TriangleStructure where
  large_perimeter := 24
  small_side_length := 1
  small_triangle_count := 27

/-- Calculates the sum of all segment lengths in the structure -/
def sum_of_segments (ts : TriangleStructure) : ℝ :=
  ts.small_triangle_count * (3 * ts.small_side_length)

/-- Theorem stating the sum of all segments in the given structure is 81 -/
theorem sum_of_segments_is_81 :
  sum_of_segments problem_structure = 81 := by sorry

end NUMINAMATH_CALUDE_sum_of_segments_is_81_l784_78440


namespace NUMINAMATH_CALUDE_cow_field_difference_l784_78429

theorem cow_field_difference (total : ℕ) (males : ℕ) (females : ℕ) : 
  total = 300 →
  females = 2 * males →
  total = males + females →
  (females / 2 : ℕ) - (males / 2 : ℕ) = 50 := by
  sorry

end NUMINAMATH_CALUDE_cow_field_difference_l784_78429


namespace NUMINAMATH_CALUDE_even_sum_condition_l784_78474

theorem even_sum_condition (m n : ℤ) :
  (∀ m n : ℤ, Even m ∧ Even n → Even (m + n)) ∧
  (∃ m n : ℤ, Even (m + n) ∧ (¬Even m ∨ ¬Even n)) :=
sorry

end NUMINAMATH_CALUDE_even_sum_condition_l784_78474


namespace NUMINAMATH_CALUDE_equation_solution_l784_78433

theorem equation_solution :
  ∃ (x : ℝ), 
    (3*x - 1 ≥ 0) ∧ 
    (x + 4 > 0) ∧ 
    (Real.sqrt ((3*x - 1) / (x + 4)) + 3 - 4 * Real.sqrt ((x + 4) / (3*x - 1)) = 0) ∧
    (x = 5/2) := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l784_78433


namespace NUMINAMATH_CALUDE_sqrt2_minus1_cf_infinite_sqrt3_minus1_cf_infinite_sqrt2_minus1_4th_convergent_error_sqrt3_minus1_4th_convergent_error_l784_78491

-- Define the continued fraction representation for √2 - 1
def sqrt2_minus1_cf (n : ℕ) : ℚ :=
  match n with
  | 0 => 0
  | n+1 => 1 / (2 + sqrt2_minus1_cf n)

-- Define the continued fraction representation for √3 - 1
def sqrt3_minus1_cf (n : ℕ) : ℚ :=
  match n with
  | 0 => 0
  | 1 => 1
  | n+2 => 1 / (1 + 1 / (2 + sqrt3_minus1_cf n))

-- Define the fourth convergent of √2 - 1
def sqrt2_minus1_4th_convergent : ℚ := 12 / 29

-- Define the fourth convergent of √3 - 1
def sqrt3_minus1_4th_convergent : ℚ := 8 / 11

theorem sqrt2_minus1_cf_infinite :
  ∀ n : ℕ, sqrt2_minus1_cf n ≠ sqrt2_minus1_cf (n+1) :=
sorry

theorem sqrt3_minus1_cf_infinite :
  ∀ n : ℕ, sqrt3_minus1_cf n ≠ sqrt3_minus1_cf (n+1) :=
sorry

theorem sqrt2_minus1_4th_convergent_error :
  |Real.sqrt 2 - 1 - sqrt2_minus1_4th_convergent| < 1 / 2000 :=
sorry

theorem sqrt3_minus1_4th_convergent_error :
  |Real.sqrt 3 - 1 - sqrt3_minus1_4th_convergent| < 1 / 209 :=
sorry

end NUMINAMATH_CALUDE_sqrt2_minus1_cf_infinite_sqrt3_minus1_cf_infinite_sqrt2_minus1_4th_convergent_error_sqrt3_minus1_4th_convergent_error_l784_78491


namespace NUMINAMATH_CALUDE_complex_multiplication_l784_78497

theorem complex_multiplication (i : ℂ) : i * i = -1 → 2 * i * (1 - i) = 2 + 2 * i := by
  sorry

end NUMINAMATH_CALUDE_complex_multiplication_l784_78497


namespace NUMINAMATH_CALUDE_regression_line_prediction_l784_78492

/-- Represents a linear regression line -/
structure RegressionLine where
  slope : ℝ
  intercept : ℝ

/-- Calculates the y-value for a given x on the regression line -/
def RegressionLine.predict (line : RegressionLine) (x : ℝ) : ℝ :=
  line.slope * x + line.intercept

theorem regression_line_prediction 
  (slope : ℝ) 
  (center_x center_y : ℝ) 
  (h_slope : slope = 1.23) 
  (h_center : center_y = slope * center_x + intercept) 
  (h_center_x : center_x = 4) 
  (h_center_y : center_y = 5) :
  let line : RegressionLine := {
    slope := slope,
    intercept := center_y - slope * center_x
  }
  line.predict 2 = 2.54 := by
  sorry

end NUMINAMATH_CALUDE_regression_line_prediction_l784_78492


namespace NUMINAMATH_CALUDE_illuminated_part_depends_on_position_l784_78428

/-- Represents a right circular cone on a plane -/
structure Cone where
  r : ℝ  -- radius of the base
  h : ℝ  -- height of the cone

/-- Represents the position of a light source -/
structure LightSource where
  H : ℝ  -- distance from the plane
  l : ℝ  -- distance from the height of the cone

/-- Represents the illuminated part of a circle -/
structure IlluminatedPart where
  angle : ℝ  -- angle of the illuminated arc

/-- Calculates the illuminated part of a circle with radius R on the plane -/
noncomputable def calculateIlluminatedPart (cone : Cone) (light : LightSource) (R : ℝ) : IlluminatedPart :=
  sorry

/-- Theorem stating that the illuminated part can be determined by the relative position of the light source -/
theorem illuminated_part_depends_on_position (cone : Cone) (light : LightSource) (R : ℝ) :
  ∃ (ip : IlluminatedPart), ip = calculateIlluminatedPart cone light R ∧
  (light.H > cone.h ∨ light.H = cone.h ∨ light.H < cone.h) :=
  sorry

end NUMINAMATH_CALUDE_illuminated_part_depends_on_position_l784_78428


namespace NUMINAMATH_CALUDE_rice_on_eighth_day_l784_78488

/-- Represents the number of laborers on a given day -/
def laborers (day : ℕ) : ℕ := 64 + 7 * (day - 1)

/-- The amount of rice given to each laborer per day -/
def ricePerLaborer : ℕ := 3

/-- The amount of rice given out on a specific day -/
def riceOnDay (day : ℕ) : ℕ := laborers day * ricePerLaborer

theorem rice_on_eighth_day : riceOnDay 8 = 339 := by
  sorry

end NUMINAMATH_CALUDE_rice_on_eighth_day_l784_78488


namespace NUMINAMATH_CALUDE_sphere_volume_l784_78408

theorem sphere_volume (r : ℝ) (d V : ℝ) (h1 : r = 1/3) (h2 : d = 2*r) (h3 : d = (16/9 * V)^(1/3)) : V = 1/6 := by
  sorry

end NUMINAMATH_CALUDE_sphere_volume_l784_78408


namespace NUMINAMATH_CALUDE_sqrt_product_plus_one_equals_271_l784_78427

theorem sqrt_product_plus_one_equals_271 : 
  Real.sqrt ((18 : ℝ) * 17 * 16 * 15 + 1) = 271 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_product_plus_one_equals_271_l784_78427


namespace NUMINAMATH_CALUDE_no_integer_roots_for_primes_l784_78403

theorem no_integer_roots_for_primes (p q : ℕ) : 
  Prime p → Prime q → ¬∃ (x : ℤ), x^2 + 3*p*x + 5*q = 0 := by
  sorry

end NUMINAMATH_CALUDE_no_integer_roots_for_primes_l784_78403


namespace NUMINAMATH_CALUDE_odd_binomial_coefficients_count_l784_78445

theorem odd_binomial_coefficients_count (n : ℕ) : 
  (∃ m : ℕ, (Finset.filter (fun k => Nat.choose n k % 2 = 1) (Finset.range (n + 1))).card = 2^m) := by
  sorry

end NUMINAMATH_CALUDE_odd_binomial_coefficients_count_l784_78445


namespace NUMINAMATH_CALUDE_tangent_circles_exist_l784_78407

/-- A circle in a 2D plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- A line in a 2D plane -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Predicate to check if a point is on a circle -/
def IsOnCircle (p : ℝ × ℝ) (c : Circle) : Prop :=
  (p.1 - c.center.1)^2 + (p.2 - c.center.2)^2 = c.radius^2

/-- Predicate to check if a point is on a line -/
def IsOnLine (p : ℝ × ℝ) (l : Line) : Prop :=
  l.a * p.1 + l.b * p.2 + l.c = 0

/-- Predicate to check if two circles are externally tangent -/
def AreExternallyTangent (c1 c2 : Circle) : Prop :=
  (c1.center.1 - c2.center.1)^2 + (c1.center.2 - c2.center.2)^2 = (c1.radius + c2.radius)^2

/-- Predicate to check if a circle touches another circle at a point -/
def CircleTouchesCircleAt (c1 c2 : Circle) (p : ℝ × ℝ) : Prop :=
  IsOnCircle p c1 ∧ IsOnCircle p c2 ∧ AreExternallyTangent c1 c2

/-- Predicate to check if a circle touches a line at a point -/
def CircleTouchesLineAt (c : Circle) (l : Line) (p : ℝ × ℝ) : Prop :=
  IsOnCircle p c ∧ IsOnLine p l

/-- The main theorem -/
theorem tangent_circles_exist (k : Circle) (e : Line) (P Q : ℝ × ℝ)
    (h_P : IsOnCircle P k) (h_Q : IsOnLine Q e) :
    ∃ (c1 c2 : Circle),
      c1.radius = c2.radius ∧
      AreExternallyTangent c1 c2 ∧
      CircleTouchesCircleAt c1 k P ∧
      CircleTouchesLineAt c2 e Q := by
  sorry

end NUMINAMATH_CALUDE_tangent_circles_exist_l784_78407


namespace NUMINAMATH_CALUDE_susie_babysitting_rate_l784_78415

/-- Susie's babysitting scenario -/
theorem susie_babysitting_rate :
  ∀ (rate : ℚ),
  (∀ (day : ℕ), day ≤ 7 → day * (3 * rate) = day * (3 * rate)) →  -- She works 3 hours every day
  (3/10 + 2/5) * (7 * (3 * rate)) + 63 = 7 * (3 * rate) →  -- Spent fractions and remaining money
  rate = 10 := by
sorry

end NUMINAMATH_CALUDE_susie_babysitting_rate_l784_78415


namespace NUMINAMATH_CALUDE_student_ranking_l784_78493

theorem student_ranking (n : ℕ) 
  (rank_from_right : ℕ) 
  (rank_from_left : ℕ) 
  (h1 : rank_from_right = 17) 
  (h2 : rank_from_left = 5) : 
  n = rank_from_right + rank_from_left - 1 :=
by sorry

end NUMINAMATH_CALUDE_student_ranking_l784_78493


namespace NUMINAMATH_CALUDE_page_lines_increase_percentage_correct_increase_percentage_l784_78465

theorem page_lines_increase_percentage : ℕ → ℝ → Prop :=
  fun original_lines increase_percentage =>
    let new_lines : ℕ := original_lines + 80
    new_lines = 240 →
    (increase_percentage * original_lines : ℝ) = 80 * 100

theorem correct_increase_percentage : 
  ∃ (original_lines : ℕ), page_lines_increase_percentage original_lines 50 := by
  sorry

end NUMINAMATH_CALUDE_page_lines_increase_percentage_correct_increase_percentage_l784_78465


namespace NUMINAMATH_CALUDE_problem_sequence_fifth_term_l784_78448

/-- A sequence where the differences between consecutive terms form an arithmetic sequence with a common difference of 3 -/
def SpecialSequence (a : ℕ → ℕ) : Prop :=
  ∀ n : ℕ, a (n + 1) - a n = 3 * n + 3

/-- The specific sequence from the problem -/
def ProblemSequence (a : ℕ → ℕ) : Prop :=
  a 0 = 2 ∧ a 1 = 5 ∧ a 2 = 11 ∧ a 3 = 20 ∧ a 5 = 47

theorem problem_sequence_fifth_term (a : ℕ → ℕ) 
  (h1 : SpecialSequence a) (h2 : ProblemSequence a) : a 4 = 32 := by
  sorry

end NUMINAMATH_CALUDE_problem_sequence_fifth_term_l784_78448


namespace NUMINAMATH_CALUDE_sqrt_fraction_simplification_l784_78489

theorem sqrt_fraction_simplification :
  Real.sqrt (8^2 + 15^2) / Real.sqrt (25 + 16) = 17 * Real.sqrt 41 / 41 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_fraction_simplification_l784_78489


namespace NUMINAMATH_CALUDE_x_squared_mod_20_l784_78442

theorem x_squared_mod_20 (x : ℤ) (h1 : 5 * x ≡ 10 [ZMOD 20]) (h2 : 2 * x ≡ 8 [ZMOD 20]) :
  x^2 ≡ 16 [ZMOD 20] := by
  sorry

end NUMINAMATH_CALUDE_x_squared_mod_20_l784_78442


namespace NUMINAMATH_CALUDE_inscribed_sphere_volume_l784_78437

/-- The volume of a sphere inscribed in a right circular cone -/
theorem inscribed_sphere_volume (d : ℝ) (θ : ℝ) (h_d : d = 12 * Real.sqrt 2) (h_θ : θ = π / 4) :
  let r := d / 4
  (4 / 3) * π * r^3 = 288 * π := by sorry

end NUMINAMATH_CALUDE_inscribed_sphere_volume_l784_78437


namespace NUMINAMATH_CALUDE_sum_of_seven_place_values_l784_78431

theorem sum_of_seven_place_values (n : ℚ) (h : n = 87953.0727) :
  (7000 : ℚ) + (7 / 100 : ℚ) + (7 / 10000 : ℚ) = 7000.0707 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_seven_place_values_l784_78431


namespace NUMINAMATH_CALUDE_parallel_line_plane_intersection_false_l784_78498

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the parallel and intersection relations
variable (parallel_line_plane : Line → Plane → Prop)
variable (parallel_lines : Line → Line → Prop)
variable (intersect_planes : Plane → Plane → Line → Prop)

-- Define our specific objects
variable (m n : Line)
variable (α β : Plane)

-- State that m and n are different lines
variable (m_neq_n : m ≠ n)

-- State that α and β are different planes
variable (α_neq_β : α ≠ β)

-- The theorem to be proved
theorem parallel_line_plane_intersection_false :
  ¬(∀ (m n : Line) (α β : Plane), 
    parallel_line_plane m α → intersect_planes α β n → parallel_lines m n) :=
sorry

end NUMINAMATH_CALUDE_parallel_line_plane_intersection_false_l784_78498


namespace NUMINAMATH_CALUDE_not_sufficient_nor_necessary_l784_78458

theorem not_sufficient_nor_necessary (p q : Prop) :
  ¬(((p ∧ q) → ¬p) ∧ (¬p → (p ∧ q))) := by sorry

end NUMINAMATH_CALUDE_not_sufficient_nor_necessary_l784_78458


namespace NUMINAMATH_CALUDE_exists_valid_marking_configuration_l784_78418

/-- A type representing a cell in the grid -/
structure Cell :=
  (row : Fin 19)
  (col : Fin 19)

/-- A type representing a marking configuration of the grid -/
def MarkingConfiguration := Cell → Bool

/-- A function to count marked cells in a 10x10 square -/
def countMarkedCells (config : MarkingConfiguration) (topLeft : Cell) : Nat :=
  sorry

/-- A predicate to check if all 10x10 squares have different counts -/
def allSquaresDifferent (config : MarkingConfiguration) : Prop :=
  ∀ s1 s2 : Cell, s1 ≠ s2 → 
    countMarkedCells config s1 ≠ countMarkedCells config s2

/-- The main theorem stating the existence of a valid marking configuration -/
theorem exists_valid_marking_configuration : 
  ∃ (config : MarkingConfiguration), allSquaresDifferent config :=
sorry

end NUMINAMATH_CALUDE_exists_valid_marking_configuration_l784_78418


namespace NUMINAMATH_CALUDE_mens_total_wages_l784_78483

/-- Proves that the men's total wages are 150 given the problem conditions --/
theorem mens_total_wages (W : ℕ) : 
  (12 : ℚ) * W = (20 : ℚ) → -- 12 men equal W women, W women equal 20 boys
  (12 : ℚ) * W * W + W * W * W + (20 : ℚ) * W = (450 : ℚ) → -- Total earnings equation
  (12 : ℚ) * ((450 : ℚ) / ((12 : ℚ) + W + (20 : ℚ))) = (150 : ℚ) -- Men's total wages
:= by sorry

end NUMINAMATH_CALUDE_mens_total_wages_l784_78483


namespace NUMINAMATH_CALUDE_nth_inequality_l784_78467

theorem nth_inequality (x : ℝ) (n : ℕ) (h : x > 0) : 
  x + (n^n : ℝ) / x^n ≥ n + 1 := by
sorry

end NUMINAMATH_CALUDE_nth_inequality_l784_78467


namespace NUMINAMATH_CALUDE_leg_equals_sum_of_radii_l784_78450

/-- An isosceles right triangle with its inscribed and circumscribed circles -/
structure IsoscelesRightTriangle where
  /-- The length of each leg of the triangle -/
  a : ℝ
  /-- The radius of the inscribed circle -/
  r : ℝ
  /-- The radius of the circumscribed circle -/
  R : ℝ
  /-- The leg length is positive -/
  a_pos : 0 < a
  /-- The inscribed circle radius is half the leg length -/
  r_def : r = a / 2
  /-- The circumscribed circle radius is (a√2)/2 -/
  R_def : R = (a * Real.sqrt 2) / 2

/-- 
The length of the legs of an isosceles right triangle is equal to 
the sum of the radii of its inscribed and circumscribed circles 
-/
theorem leg_equals_sum_of_radii (t : IsoscelesRightTriangle) : t.a = t.r + t.R := by
  sorry

end NUMINAMATH_CALUDE_leg_equals_sum_of_radii_l784_78450


namespace NUMINAMATH_CALUDE_evaluate_polynomial_l784_78430

theorem evaluate_polynomial (x : ℤ) (h : x = -2) : x^3 + x^2 + x + 1 = -5 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_polynomial_l784_78430


namespace NUMINAMATH_CALUDE_min_coin_tosses_l784_78409

theorem min_coin_tosses (n : ℕ) : (1 - (1/2)^n ≥ 15/16) ↔ n ≥ 4 := by sorry

end NUMINAMATH_CALUDE_min_coin_tosses_l784_78409


namespace NUMINAMATH_CALUDE_fixed_point_of_f_l784_78447

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a^(x - 2) + 2

-- State the theorem
theorem fixed_point_of_f (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  f a 2 = 3 :=
by sorry

end NUMINAMATH_CALUDE_fixed_point_of_f_l784_78447


namespace NUMINAMATH_CALUDE_savings_increase_percentage_l784_78464

theorem savings_increase_percentage (I : ℝ) (I_pos : I > 0) : 
  let regular_expense_ratio : ℝ := 0.75
  let additional_expense_ratio : ℝ := 0.10
  let income_increase_ratio : ℝ := 0.20
  let regular_expense_increase_ratio : ℝ := 0.10
  let additional_expense_increase_ratio : ℝ := 0.25
  
  let initial_savings := I * (1 - regular_expense_ratio - additional_expense_ratio)
  let new_income := I * (1 + income_increase_ratio)
  let new_regular_expense := I * regular_expense_ratio * (1 + regular_expense_increase_ratio)
  let new_additional_expense := I * additional_expense_ratio * (1 + additional_expense_increase_ratio)
  let new_savings := new_income - new_regular_expense - new_additional_expense
  
  (new_savings - initial_savings) / initial_savings = 2/3 :=
by
  sorry

end NUMINAMATH_CALUDE_savings_increase_percentage_l784_78464


namespace NUMINAMATH_CALUDE_find_y_l784_78468

theorem find_y : ∃ y : ℝ, y > 0 ∧ 0.02 * y * y = 18 ∧ y = 30 := by sorry

end NUMINAMATH_CALUDE_find_y_l784_78468


namespace NUMINAMATH_CALUDE_polynomial_factorization_l784_78416

theorem polynomial_factorization : 
  ∀ x : ℤ, x^15 + x^10 + 1 = (x^3 + x^2 + 1) * (x^12 - x^11 + x^9 - x^8 + x^6 - x^4 + x^2 - x + 1) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l784_78416


namespace NUMINAMATH_CALUDE_exactly_fourteen_plus_signs_l784_78410

/-- Represents a board with plus and minus signs -/
structure SignBoard where
  total_symbols : ℕ
  plus_signs : ℕ
  minus_signs : ℕ
  total_is_sum : total_symbols = plus_signs + minus_signs

/-- Predicate to check if any subset of size n contains at least one plus sign -/
def has_plus_in_subset (board : SignBoard) (n : ℕ) : Prop :=
  board.minus_signs < n

/-- Predicate to check if any subset of size n contains at least one minus sign -/
def has_minus_in_subset (board : SignBoard) (n : ℕ) : Prop :=
  board.plus_signs < n

/-- The main theorem to prove -/
theorem exactly_fourteen_plus_signs (board : SignBoard) 
  (h_total : board.total_symbols = 23)
  (h_plus_10 : has_plus_in_subset board 10)
  (h_minus_15 : has_minus_in_subset board 15) :
  board.plus_signs = 14 :=
sorry

end NUMINAMATH_CALUDE_exactly_fourteen_plus_signs_l784_78410


namespace NUMINAMATH_CALUDE_triangle_proof_l784_78424

-- Define the triangle ABC
structure Triangle where
  A : ℝ
  B : ℝ
  C : ℝ
  AB : ℝ
  BC : ℝ
  AC : ℝ

-- Define the properties of the triangle
def triangle_properties (t : Triangle) : Prop :=
  t.BC = 7 ∧ t.AB = 3 ∧ (Real.sin t.C) / (Real.sin t.B) = 3/5

-- Theorem statement
theorem triangle_proof (t : Triangle) (h : triangle_properties t) :
  t.AC = 5 ∧ t.A = Real.pi * 2/3 := by
  sorry

end NUMINAMATH_CALUDE_triangle_proof_l784_78424


namespace NUMINAMATH_CALUDE_cafeteria_pies_l784_78432

theorem cafeteria_pies (initial_apples : ℕ) (handed_out : ℕ) (apples_per_pie : ℕ) 
  (h1 : initial_apples = 50)
  (h2 : handed_out = 5)
  (h3 : apples_per_pie = 5) :
  (initial_apples - handed_out) / apples_per_pie = 9 := by
  sorry

end NUMINAMATH_CALUDE_cafeteria_pies_l784_78432


namespace NUMINAMATH_CALUDE_no_seven_edge_polyhedron_exists_polyhedron_with_n_edges_l784_78454

/-- A convex polyhedron is a three-dimensional geometric object with flat polygonal faces, straight edges and sharp corners or vertices. -/
structure ConvexPolyhedron where
  -- We don't need to specify the internal structure for this problem
  mk :: -- Constructor

/-- The number of edges in a convex polyhedron -/
def num_edges (p : ConvexPolyhedron) : ℕ := sorry

/-- Theorem stating that no convex polyhedron has exactly 7 edges -/
theorem no_seven_edge_polyhedron : ¬∃ (p : ConvexPolyhedron), num_edges p = 7 := by sorry

/-- Theorem stating that for all natural numbers n ≥ 6 and n ≠ 7, there exists a convex polyhedron with n edges -/
theorem exists_polyhedron_with_n_edges (n : ℕ) (h1 : n ≥ 6) (h2 : n ≠ 7) : 
  ∃ (p : ConvexPolyhedron), num_edges p = n := by sorry

end NUMINAMATH_CALUDE_no_seven_edge_polyhedron_exists_polyhedron_with_n_edges_l784_78454


namespace NUMINAMATH_CALUDE_max_value_of_x_l784_78425

/-- Given a > 0 and b > 0, x is defined as the minimum of {1, a, b / (a² + b²)}.
    This theorem states that the maximum value of x is √2 / 2. -/
theorem max_value_of_x (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  let x := min 1 (min a (b / (a^2 + b^2)))
  ∃ (max_x : ℝ), max_x = Real.sqrt 2 / 2 ∧ ∀ (a' b' : ℝ), a' > 0 → b' > 0 →
    min 1 (min a' (b' / (a'^2 + b'^2))) ≤ max_x :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_x_l784_78425


namespace NUMINAMATH_CALUDE_elvis_squares_l784_78402

theorem elvis_squares (total_matchsticks : ℕ) (elvis_square_size : ℕ) (ralph_square_size : ℕ) 
  (ralph_squares : ℕ) (leftover_matchsticks : ℕ) :
  total_matchsticks = 50 →
  elvis_square_size = 4 →
  ralph_square_size = 8 →
  ralph_squares = 3 →
  leftover_matchsticks = 6 →
  ∃ (elvis_squares : ℕ), 
    elvis_squares * elvis_square_size + ralph_squares * ralph_square_size + leftover_matchsticks = total_matchsticks ∧
    elvis_squares = 5 := by
  sorry

end NUMINAMATH_CALUDE_elvis_squares_l784_78402


namespace NUMINAMATH_CALUDE_T_forms_three_lines_closed_region_l784_78471

-- Define the set T
def T : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | let x := p.1; let y := p.2;
    (4 ≤ x - 1 ∧ 4 ≤ y + 3 ∧ y + 3 < x - 1) ∨
    (4 ≤ x - 1 ∧ 4 ≤ y + 3 ∧ x - 1 < y + 3) ∨
    (4 ≤ x - 1 ∧ y + 3 < 4 ∧ y + 3 < x - 1) ∨
    (x - 1 < 4 ∧ 4 ≤ y + 3 ∧ x - 1 < y + 3) ∨
    (x - 1 ≤ y + 3 ∧ 4 < x - 1 ∧ 4 < y + 3)}

-- Define the three lines
def line1 : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1 = 5 ∧ p.2 ≤ 1}
def line2 : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.2 = 1 ∧ p.1 ≤ 5}
def line3 : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.2 = p.1 - 4 ∧ p.1 ≥ 5}

-- Theorem statement
theorem T_forms_three_lines_closed_region :
  ∃ (point : ℝ × ℝ), 
    point ∈ T ∧
    point ∈ line1 ∧ point ∈ line2 ∧ point ∈ line3 ∧
    T = line1 ∪ line2 ∪ line3 :=
sorry


end NUMINAMATH_CALUDE_T_forms_three_lines_closed_region_l784_78471


namespace NUMINAMATH_CALUDE_complement_of_A_l784_78480

-- Define the set A
def A : Set ℝ := {x | x^2 - 2*x > 0}

-- State the theorem
theorem complement_of_A : 
  {x : ℝ | ¬ (x ∈ A)} = {x : ℝ | 0 ≤ x ∧ x ≤ 2} := by
  sorry

end NUMINAMATH_CALUDE_complement_of_A_l784_78480


namespace NUMINAMATH_CALUDE_vector_equation_solution_l784_78414

theorem vector_equation_solution (a b : ℝ × ℝ) (m n : ℝ) :
  a = (2, 1) →
  b = (1, -2) →
  m • a + n • b = (5, -5) →
  m - n = -2 := by
  sorry

end NUMINAMATH_CALUDE_vector_equation_solution_l784_78414


namespace NUMINAMATH_CALUDE_maple_trees_planted_l784_78452

theorem maple_trees_planted (initial_maple : ℕ) (final_maple : ℕ) :
  initial_maple = 2 →
  final_maple = 11 →
  final_maple - initial_maple = 9 :=
by sorry

end NUMINAMATH_CALUDE_maple_trees_planted_l784_78452


namespace NUMINAMATH_CALUDE_largest_multiple_of_seven_less_than_negative_hundred_l784_78481

theorem largest_multiple_of_seven_less_than_negative_hundred :
  ∀ n : ℤ, n * 7 < -100 → n * 7 ≤ -105 :=
by
  sorry

end NUMINAMATH_CALUDE_largest_multiple_of_seven_less_than_negative_hundred_l784_78481


namespace NUMINAMATH_CALUDE_import_export_scientific_notation_l784_78435

def billion : ℝ := 1000000000

theorem import_export_scientific_notation (volume : ℝ) (h : volume = 214.7 * billion) :
  ∃ (a : ℝ) (n : ℤ), volume = a * (10 : ℝ) ^ n ∧ 1 ≤ a ∧ a < 10 ∧ n = 10 := by
  sorry

end NUMINAMATH_CALUDE_import_export_scientific_notation_l784_78435


namespace NUMINAMATH_CALUDE_arithmetic_sequence_nth_term_l784_78453

/-- An arithmetic sequence is a sequence where the difference between
    consecutive terms is constant. -/
def ArithmeticSequence (a : ℕ → ℝ) (d : ℝ) :=
  ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_nth_term
  (a : ℕ → ℝ)
  (d : ℝ)
  (h : ArithmeticSequence a d)
  (h1 : a 1 = 11)
  (h2 : d = 2)
  (h3 : ∃ n : ℕ, a n = 2009) :
  ∃ n : ℕ, n = 1000 ∧ a n = 2009 :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_nth_term_l784_78453


namespace NUMINAMATH_CALUDE_line_tangent_to_log_curve_l784_78420

/-- A line y = x + 1 is tangent to the curve y = ln(x + a) if and only if a = 2 -/
theorem line_tangent_to_log_curve (a : ℝ) : 
  (∃ x : ℝ, x + 1 = Real.log (x + a) ∧ 
   ∀ y : ℝ, y ≠ x → y + 1 ≠ Real.log (y + a) ∧
   (1 : ℝ) = 1 / (x + a)) ↔ 
  a = 2 := by
  sorry

end NUMINAMATH_CALUDE_line_tangent_to_log_curve_l784_78420
