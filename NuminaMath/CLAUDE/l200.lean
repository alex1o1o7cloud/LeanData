import Mathlib

namespace NUMINAMATH_CALUDE_lucas_football_scores_l200_20097

def first_ten_games : List Nat := [5, 2, 6, 3, 10, 1, 3, 3, 4, 2]

def total_first_ten : Nat := first_ten_games.sum

theorem lucas_football_scores :
  ∃ (game11 game12 : Nat),
    game11 < 10 ∧
    game12 < 10 ∧
    (total_first_ten + game11) % 11 = 0 ∧
    (total_first_ten + game11 + game12) % 12 = 0 ∧
    game11 * game12 = 20 := by
  sorry

end NUMINAMATH_CALUDE_lucas_football_scores_l200_20097


namespace NUMINAMATH_CALUDE_gunther_free_time_l200_20099

/-- Represents the time required for cleaning tasks and available free time -/
structure CleaningTime where
  vacuum : ℕ
  dust : ℕ
  mop : ℕ
  brush_per_cat : ℕ
  num_cats : ℕ
  free_time : ℕ

/-- Calculates the remaining free time after cleaning -/
def remaining_free_time (ct : CleaningTime) : ℕ :=
  ct.free_time - (ct.vacuum + ct.dust + ct.mop + ct.brush_per_cat * ct.num_cats)

/-- Theorem: Given Gunther's cleaning tasks and available time, he will have 30 minutes left -/
theorem gunther_free_time :
  ∀ (ct : CleaningTime),
    ct.vacuum = 45 →
    ct.dust = 60 →
    ct.mop = 30 →
    ct.brush_per_cat = 5 →
    ct.num_cats = 3 →
    ct.free_time = 180 →
    remaining_free_time ct = 30 := by
  sorry

end NUMINAMATH_CALUDE_gunther_free_time_l200_20099


namespace NUMINAMATH_CALUDE_stratified_sampling_probability_l200_20035

theorem stratified_sampling_probability 
  (total_students : ℕ) 
  (first_year : ℕ) 
  (second_year : ℕ) 
  (third_year : ℕ) 
  (selected : ℕ) 
  (h1 : total_students = first_year + second_year + third_year)
  (h2 : total_students = 600)
  (h3 : first_year = 100)
  (h4 : second_year = 200)
  (h5 : third_year = 300)
  (h6 : selected = 30) :
  (selected : ℚ) / (total_students : ℚ) = 1 / 20 := by
sorry

end NUMINAMATH_CALUDE_stratified_sampling_probability_l200_20035


namespace NUMINAMATH_CALUDE_stratified_sampling_probability_l200_20024

theorem stratified_sampling_probability (students teachers support_staff sample_size : ℕ) 
  (h1 : students = 2500)
  (h2 : teachers = 350)
  (h3 : support_staff = 150)
  (h4 : sample_size = 300) :
  (sample_size * students) / ((students + teachers + support_staff) * students) = 1 / 10 := by
sorry

end NUMINAMATH_CALUDE_stratified_sampling_probability_l200_20024


namespace NUMINAMATH_CALUDE_exam_logic_l200_20046

-- Define the universe of students
variable (Student : Type)

-- Define predicates
variable (got_all_right : Student → Prop)
variable (received_A : Student → Prop)

-- State the theorem
theorem exam_logic (s : Student) 
  (h : ∀ x, got_all_right x → received_A x) :
  ¬(received_A s) → ¬(got_all_right s) := by
sorry

end NUMINAMATH_CALUDE_exam_logic_l200_20046


namespace NUMINAMATH_CALUDE_other_root_of_quadratic_l200_20033

theorem other_root_of_quadratic (b : ℝ) : 
  ((-1 : ℝ)^2 + b * (-1) - 5 = 0) → 
  (∃ x : ℝ, x ≠ -1 ∧ x^2 + b*x - 5 = 0 ∧ x = 5) :=
by sorry

end NUMINAMATH_CALUDE_other_root_of_quadratic_l200_20033


namespace NUMINAMATH_CALUDE_remaining_bird_families_l200_20021

/-- The number of bird families left near the mountain after some flew away -/
def bird_families_left (initial : ℕ) (flew_away : ℕ) : ℕ :=
  initial - flew_away

/-- Theorem stating that 237 bird families were left near the mountain -/
theorem remaining_bird_families :
  bird_families_left 709 472 = 237 := by
  sorry

end NUMINAMATH_CALUDE_remaining_bird_families_l200_20021


namespace NUMINAMATH_CALUDE_seating_arrangements_count_l200_20015

/-- Represents a theater with two rows of seats. -/
structure Theater :=
  (front_seats : Nat)
  (back_seats : Nat)

/-- Calculates the number of valid seating arrangements for two people in a theater. -/
def validArrangements (t : Theater) (middle_seats : Nat) : Nat :=
  sorry

/-- The theorem stating that the number of valid seating arrangements is 114. -/
theorem seating_arrangements_count (t : Theater) :
  t.front_seats = 9 ∧ t.back_seats = 8 →
  validArrangements t 3 = 114 :=
by sorry

end NUMINAMATH_CALUDE_seating_arrangements_count_l200_20015


namespace NUMINAMATH_CALUDE_incircle_excircle_center_distance_l200_20075

/-- Given a triangle DEF with side lengths, prove the distance between incircle and excircle centers --/
theorem incircle_excircle_center_distance (DE DF EF : ℝ) (h_DE : DE = 20) (h_DF : DF = 21) (h_EF : EF = 29) :
  let s := (DE + DF + EF) / 2
  let K := Real.sqrt (s * (s - DE) * (s - DF) * (s - EF))
  let r := K / s
  let I := Real.sqrt (DE^2 + r^2)
  let E := (DF * I) / DE
  E - I = Real.sqrt 232 / 14 := by sorry

end NUMINAMATH_CALUDE_incircle_excircle_center_distance_l200_20075


namespace NUMINAMATH_CALUDE_proposition_p_false_implies_a_range_l200_20064

theorem proposition_p_false_implies_a_range (a : ℝ) : 
  (¬ ∀ x : ℝ, a * x^2 + a * x + 1 ≥ 0) → 
  (a < 0 ∨ a > 4) := by
sorry

end NUMINAMATH_CALUDE_proposition_p_false_implies_a_range_l200_20064


namespace NUMINAMATH_CALUDE_decorative_window_area_ratio_l200_20058

theorem decorative_window_area_ratio :
  let base : ℝ := 40
  let length : ℝ := (4/3) * base
  let semi_major_axis : ℝ := base / 2
  let semi_minor_axis : ℝ := base / 4
  let rectangle_area : ℝ := length * base
  let ellipse_area : ℝ := π * semi_major_axis * semi_minor_axis
  let triangle_area : ℝ := (1/2) * base * semi_minor_axis
  rectangle_area / (ellipse_area + triangle_area) = 32 / (3 * (π + 1)) :=
by sorry

end NUMINAMATH_CALUDE_decorative_window_area_ratio_l200_20058


namespace NUMINAMATH_CALUDE_typist_salary_problem_l200_20044

/-- Proves that if a salary is increased by 10% and then decreased by 5%, 
    resulting in 1045, the original salary was 1000. -/
theorem typist_salary_problem (original_salary : ℝ) : 
  (original_salary * 1.1 * 0.95 = 1045) → original_salary = 1000 := by
  sorry

end NUMINAMATH_CALUDE_typist_salary_problem_l200_20044


namespace NUMINAMATH_CALUDE_min_value_zero_l200_20036

/-- The expression for which we want to find the minimum value -/
def f (k x y : ℝ) : ℝ := 9*x^2 - 12*k*x*y + (4*k^2 + 3)*y^2 - 6*x - 6*y + 9

/-- The theorem stating the condition for the minimum value of f to be 0 -/
theorem min_value_zero (k : ℝ) :
  (∀ x y : ℝ, f k x y ≥ 0) ∧ (∃ x y : ℝ, f k x y = 0) ↔ k = 4/3 := by sorry

end NUMINAMATH_CALUDE_min_value_zero_l200_20036


namespace NUMINAMATH_CALUDE_lucky_number_property_l200_20051

/-- A number is lucky if the sum of its digits is 7 -/
def IsLucky (n : ℕ) : Prop :=
  (n.digits 10).sum = 7

/-- The sequence of lucky numbers in ascending order -/
def LuckySequence : ℕ → ℕ :=
  sorry

theorem lucky_number_property (n : ℕ) :
  LuckySequence n = 2005 → LuckySequence (5 * n) = 30301 :=
by
  sorry

end NUMINAMATH_CALUDE_lucky_number_property_l200_20051


namespace NUMINAMATH_CALUDE_gcd_102_238_l200_20083

theorem gcd_102_238 : Nat.gcd 102 238 = 34 := by
  sorry

end NUMINAMATH_CALUDE_gcd_102_238_l200_20083


namespace NUMINAMATH_CALUDE_sum_after_removing_terms_l200_20020

theorem sum_after_removing_terms : 
  let sequence := [1/3, 1/6, 1/9, 1/12, 1/15, 1/18]
  let removed_terms := [1/12, 1/15]
  let remaining_terms := sequence.filter (λ x => x ∉ removed_terms)
  (remaining_terms.sum = 1) := by sorry

end NUMINAMATH_CALUDE_sum_after_removing_terms_l200_20020


namespace NUMINAMATH_CALUDE_rectangular_field_area_l200_20068

/-- A rectangular field with breadth 60% of length and perimeter 800 m has area 37500 m² -/
theorem rectangular_field_area (length breadth : ℝ) : 
  breadth = 0.6 * length →
  2 * (length + breadth) = 800 →
  length * breadth = 37500 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_field_area_l200_20068


namespace NUMINAMATH_CALUDE_unique_congruence_solution_l200_20089

theorem unique_congruence_solution : ∃! n : ℕ, 0 ≤ n ∧ n ≤ 8 ∧ n ≡ 1000000 [ZMOD 9] := by
  sorry

end NUMINAMATH_CALUDE_unique_congruence_solution_l200_20089


namespace NUMINAMATH_CALUDE_expression_perfect_square_iff_A_specific_values_l200_20071

/-- A monomial is a term of the form cx^n where c is a constant and n is a non-negative integer. -/
def Monomial (x : ℝ) := ℝ → ℝ

/-- The expression x^6 + x^4 + xA -/
def Expression (x : ℝ) (A : Monomial x) : ℝ := x^6 + x^4 + x * A x

/-- A perfect square is a number that is the square of an integer. -/
def IsPerfectSquare (n : ℝ) : Prop := ∃ m : ℝ, n = m^2

theorem expression_perfect_square_iff_A_specific_values (x : ℝ) (A : Monomial x) :
  IsPerfectSquare (Expression x A) ↔ 
  (A = λ x => 2 * x^4) ∨ 
  (A = λ x => -2 * x^4) ∨ 
  (A = λ x => (1/4) * x^7) ∨ 
  (A = λ x => (1/4) * x) :=
sorry

end NUMINAMATH_CALUDE_expression_perfect_square_iff_A_specific_values_l200_20071


namespace NUMINAMATH_CALUDE_winner_determined_by_parity_l200_20042

/-- Represents a player in the game -/
inductive Player
  | Anthelme
  | Brunehaut

/-- Represents the game state on an m × n chessboard -/
structure GameState (m n : ℕ) where
  kingPosition : ℕ × ℕ
  visitedSquares : Set (ℕ × ℕ)

/-- Determines the winner of the game based on the board dimensions -/
def determineWinner (m n : ℕ) : Player :=
  if m * n % 2 = 0 then Player.Anthelme else Player.Brunehaut

/-- Theorem stating that the winner is determined by the parity of m × n -/
theorem winner_determined_by_parity (m n : ℕ) :
  determineWinner m n = 
    if m * n % 2 = 0 then Player.Anthelme else Player.Brunehaut :=
by sorry

end NUMINAMATH_CALUDE_winner_determined_by_parity_l200_20042


namespace NUMINAMATH_CALUDE_sqrt_3600_equals_60_l200_20082

theorem sqrt_3600_equals_60 : Real.sqrt 3600 = 60 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_3600_equals_60_l200_20082


namespace NUMINAMATH_CALUDE_factorization_equality_l200_20026

theorem factorization_equality (a b : ℝ) : 2 * a^2 * b - 4 * a * b + 2 * b = 2 * b * (a - 1)^2 := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l200_20026


namespace NUMINAMATH_CALUDE_walking_speed_problem_l200_20055

theorem walking_speed_problem (slower_speed : ℝ) (faster_speed : ℝ) 
  (actual_distance : ℝ) (total_distance : ℝ) :
  faster_speed = 20 →
  actual_distance = 20 →
  total_distance = actual_distance + 20 →
  actual_distance / slower_speed = total_distance / faster_speed →
  slower_speed = 10 := by
  sorry

end NUMINAMATH_CALUDE_walking_speed_problem_l200_20055


namespace NUMINAMATH_CALUDE_necessary_but_not_sufficient_condition_l200_20067

theorem necessary_but_not_sufficient_condition :
  (∀ x : ℝ, -1 < x ∧ x < 3 → -2 < x ∧ x < 4) ∧
  (∃ x : ℝ, -2 < x ∧ x < 4 ∧ ¬(-1 < x ∧ x < 3)) :=
by sorry

end NUMINAMATH_CALUDE_necessary_but_not_sufficient_condition_l200_20067


namespace NUMINAMATH_CALUDE_y_equals_seven_l200_20052

/-- A shape composed entirely of right angles with specific side lengths -/
structure RightAngledShape where
  /-- Length of one side -/
  side1 : ℝ
  /-- Length of another side -/
  side2 : ℝ
  /-- Length of another side -/
  side3 : ℝ
  /-- Length of another side -/
  side4 : ℝ
  /-- Unknown length to be calculated -/
  Y : ℝ
  /-- The total horizontal lengths on the top and bottom sides are equal -/
  total_length_eq : side1 + side3 + Y + side2 = side4 + side2 + side3 + 5

/-- The theorem stating that Y equals 7 for the given shape -/
theorem y_equals_seven (shape : RightAngledShape) 
  (h1 : shape.side1 = 2) 
  (h2 : shape.side2 = 3) 
  (h3 : shape.side3 = 1) 
  (h4 : shape.side4 = 4) : 
  shape.Y = 7 := by
  sorry

end NUMINAMATH_CALUDE_y_equals_seven_l200_20052


namespace NUMINAMATH_CALUDE_set_intersection_and_union_l200_20088

theorem set_intersection_and_union (a : ℝ) : 
  let A : Set ℝ := {2, 3, a^2 + 4*a + 2}
  let B : Set ℝ := {0, 7, 2 - a, a^2 + 4*a - 2}
  (A ∩ B = {3, 7}) → 
  (a = 1 ∧ A ∪ B = {0, 1, 2, 3, 7}) :=
by sorry

end NUMINAMATH_CALUDE_set_intersection_and_union_l200_20088


namespace NUMINAMATH_CALUDE_subtract_negative_two_from_three_l200_20057

theorem subtract_negative_two_from_three : 3 - (-2) = 5 := by
  sorry

end NUMINAMATH_CALUDE_subtract_negative_two_from_three_l200_20057


namespace NUMINAMATH_CALUDE_min_points_for_12_monochromatic_triangles_l200_20092

/-- A coloring of edges in a complete graph with two colors -/
def TwoColoring (n : ℕ) := Fin n → Fin n → Bool

/-- The number of monochromatic triangles in a given coloring -/
def monochromaticTriangles (n : ℕ) (c : TwoColoring n) : ℕ := sorry

/-- The statement that for any two-coloring of Kn, there are at least 12 monochromatic triangles -/
def hasAtLeast12MonochromaticTriangles (n : ℕ) : Prop :=
  ∀ c : TwoColoring n, monochromaticTriangles n c ≥ 12

/-- The theorem stating that 9 is the minimum number of points satisfying the condition -/
theorem min_points_for_12_monochromatic_triangles :
  (hasAtLeast12MonochromaticTriangles 9) ∧ 
  (∀ m : ℕ, m < 9 → ¬(hasAtLeast12MonochromaticTriangles m)) :=
sorry

end NUMINAMATH_CALUDE_min_points_for_12_monochromatic_triangles_l200_20092


namespace NUMINAMATH_CALUDE_total_length_theorem_l200_20047

/-- Calculates the total length of ladders climbed by two workers in centimeters -/
def total_length_climbed (keaton_ladder_height : ℕ) (keaton_climbs : ℕ) 
  (reece_ladder_diff : ℕ) (reece_climbs : ℕ) : ℕ :=
  let reece_ladder_height := keaton_ladder_height - reece_ladder_diff
  let keaton_total := keaton_ladder_height * keaton_climbs
  let reece_total := reece_ladder_height * reece_climbs
  (keaton_total + reece_total) * 100

/-- The total length of ladders climbed by both workers is 422000 centimeters -/
theorem total_length_theorem : 
  total_length_climbed 60 40 8 35 = 422000 := by
  sorry

end NUMINAMATH_CALUDE_total_length_theorem_l200_20047


namespace NUMINAMATH_CALUDE_square_side_length_l200_20054

theorem square_side_length (area : ℚ) (side : ℚ) : 
  area = 9/16 → side^2 = area → side = 3/4 := by
  sorry

end NUMINAMATH_CALUDE_square_side_length_l200_20054


namespace NUMINAMATH_CALUDE_supplement_of_complement_of_63_degree_l200_20018

def complement (α : ℝ) : ℝ := 90 - α

def supplement (β : ℝ) : ℝ := 180 - β

theorem supplement_of_complement_of_63_degree :
  supplement (complement 63) = 153 := by sorry

end NUMINAMATH_CALUDE_supplement_of_complement_of_63_degree_l200_20018


namespace NUMINAMATH_CALUDE_tank_filling_time_l200_20011

theorem tank_filling_time (a b : ℝ) (ha : a > 0) (hb : b > 0) : 
  a = 60 → (15 / b + 15 * (1 / 60 + 1 / b) = 1) → b = 40 := by sorry

end NUMINAMATH_CALUDE_tank_filling_time_l200_20011


namespace NUMINAMATH_CALUDE_first_occurrence_is_lcm_l200_20072

/-- Represents the cycle length of letters -/
def letter_cycle : ℕ := 8

/-- Represents the cycle length of digits -/
def digit_cycle : ℕ := 4

/-- Represents the first occurrence of the original sequence -/
def first_occurrence : ℕ := 8

/-- Theorem stating that the first occurrence is the least common multiple of the cycle lengths -/
theorem first_occurrence_is_lcm :
  first_occurrence = Nat.lcm letter_cycle digit_cycle := by sorry

end NUMINAMATH_CALUDE_first_occurrence_is_lcm_l200_20072


namespace NUMINAMATH_CALUDE_unique_valid_set_l200_20022

/-- The sum of n consecutive integers starting from a -/
def consecutiveSum (a n : ℕ) : ℕ := n * (2 * a + n - 1) / 2

/-- A predicate that checks if a set of n consecutive integers starting from a sums to 30 -/
def isValidSet (a n : ℕ) : Prop :=
  a ≥ 3 ∧ n ≥ 2 ∧ consecutiveSum a n = 30

/-- The main theorem stating there is exactly one valid set -/
theorem unique_valid_set : ∃! p : ℕ × ℕ, isValidSet p.1 p.2 := by sorry

end NUMINAMATH_CALUDE_unique_valid_set_l200_20022


namespace NUMINAMATH_CALUDE_inequality_equivalence_l200_20073

theorem inequality_equivalence (θ : ℝ) (x : ℝ) :
  (|x + Real.cos θ ^ 2| ≤ Real.sin θ ^ 2) ↔ (-1 ≤ x ∧ x ≤ -Real.cos (2 * θ)) :=
by sorry

end NUMINAMATH_CALUDE_inequality_equivalence_l200_20073


namespace NUMINAMATH_CALUDE_design_area_is_16_l200_20013

/-- A point on a 2D grid --/
structure GridPoint where
  x : ℕ
  y : ℕ

/-- A right-angled triangle on a grid --/
structure RightTriangle where
  vertex1 : GridPoint
  vertex2 : GridPoint
  vertex3 : GridPoint

/-- The design formed by two right-angled triangles --/
structure Design where
  triangle1 : RightTriangle
  triangle2 : RightTriangle

/-- Function to calculate the area of a right-angled triangle using Pick's theorem --/
def triangleArea (t : RightTriangle) : ℕ := sorry

/-- Function to check if a design is symmetrical about the diagonal --/
def isSymmetrical (d : Design) : Prop := sorry

/-- The main theorem --/
theorem design_area_is_16 (d : Design) :
  d.triangle1.vertex1 = ⟨0, 0⟩ ∧
  d.triangle1.vertex2 = ⟨4, 0⟩ ∧
  d.triangle1.vertex3 = ⟨0, 4⟩ ∧
  d.triangle2.vertex1 = ⟨4, 0⟩ ∧
  d.triangle2.vertex2 = ⟨4, 4⟩ ∧
  d.triangle2.vertex3 = ⟨0, 4⟩ ∧
  isSymmetrical d →
  triangleArea d.triangle1 + triangleArea d.triangle2 = 16 := by
  sorry

end NUMINAMATH_CALUDE_design_area_is_16_l200_20013


namespace NUMINAMATH_CALUDE_xyz_mod_nine_l200_20086

theorem xyz_mod_nine (x y z : ℕ) : 
  x < 9 → y < 9 → z < 9 →
  (x + 3*y + 2*z) % 9 = 0 →
  (2*x + 2*y + z) % 9 = 7 →
  (x + 2*y + 3*z) % 9 = 5 →
  (x*y*z) % 9 = 5 := by
  sorry

end NUMINAMATH_CALUDE_xyz_mod_nine_l200_20086


namespace NUMINAMATH_CALUDE_circle_area_from_circumference_l200_20012

theorem circle_area_from_circumference :
  ∀ (C : ℝ) (r : ℝ) (A : ℝ),
  C = 36 →
  C = 2 * π * r →
  A = π * r^2 →
  A = 324 / π := by
sorry

end NUMINAMATH_CALUDE_circle_area_from_circumference_l200_20012


namespace NUMINAMATH_CALUDE_quadratic_function_general_form_l200_20040

/-- A quadratic function with the same shape as y = 5x² and vertex at (3, 7) -/
def quadratic_function (f : ℝ → ℝ) : Prop :=
  ∃ a : ℝ, 
    (a = 5 ∨ a = -5) ∧
    (∀ x : ℝ, f x = a * (x - 3)^2 + 7)

theorem quadratic_function_general_form (f : ℝ → ℝ) 
  (h : quadratic_function f) :
  (∀ x : ℝ, f x = 5 * x^2 - 30 * x + 52) ∨
  (∀ x : ℝ, f x = -5 * x^2 + 30 * x - 38) :=
sorry

end NUMINAMATH_CALUDE_quadratic_function_general_form_l200_20040


namespace NUMINAMATH_CALUDE_shirt_cost_l200_20063

theorem shirt_cost (J S X : ℝ) 
  (eq1 : 3 * J + 2 * S = X)
  (eq2 : 2 * J + 3 * S = 66)
  (eq3 : 3 * J + 2 * S = 2 * J + 3 * S) : 
  S = 13.20 := by
  sorry

end NUMINAMATH_CALUDE_shirt_cost_l200_20063


namespace NUMINAMATH_CALUDE_solution_set_quadratic_inequality_l200_20009

theorem solution_set_quadratic_inequality :
  {x : ℝ | 2 * x^2 - x - 3 > 0} = {x : ℝ | x < -1 ∨ x > 3/2} := by
  sorry

end NUMINAMATH_CALUDE_solution_set_quadratic_inequality_l200_20009


namespace NUMINAMATH_CALUDE_construction_material_order_l200_20005

theorem construction_material_order (concrete bricks stone total : ℝ) : 
  concrete = 0.17 →
  bricks = 0.17 →
  stone = 0.5 →
  total = concrete + bricks + stone →
  total = 0.84 := by
sorry

end NUMINAMATH_CALUDE_construction_material_order_l200_20005


namespace NUMINAMATH_CALUDE_shortest_rope_part_l200_20038

theorem shortest_rope_part (total_length : ℝ) (ratio1 ratio2 ratio3 : ℝ) 
  (h1 : total_length = 196.85)
  (h2 : ratio1 = 3.6)
  (h3 : ratio2 = 8.4)
  (h4 : ratio3 = 12) :
  let total_ratio := ratio1 + ratio2 + ratio3
  let shortest_part := (total_length / total_ratio) * ratio1
  shortest_part = 29.5275 := by
sorry

end NUMINAMATH_CALUDE_shortest_rope_part_l200_20038


namespace NUMINAMATH_CALUDE_angle_A_is_pi_over_six_l200_20043

-- Define a triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the theorem
theorem angle_A_is_pi_over_six (t : Triangle) :
  (2 * t.b - Real.sqrt 3 * t.c) * Real.cos t.A = Real.sqrt 3 * t.a * Real.cos t.C →
  t.A = π / 6 :=
by sorry

end NUMINAMATH_CALUDE_angle_A_is_pi_over_six_l200_20043


namespace NUMINAMATH_CALUDE_opposite_of_negative_one_fifth_l200_20002

theorem opposite_of_negative_one_fifth :
  -(-(1/5 : ℚ)) = 1/5 := by sorry

end NUMINAMATH_CALUDE_opposite_of_negative_one_fifth_l200_20002


namespace NUMINAMATH_CALUDE_chess_club_mixed_groups_l200_20037

/-- Represents the chess club structure and game information -/
structure ChessClub where
  total_children : ℕ
  total_groups : ℕ
  children_per_group : ℕ
  boy_boy_games : ℕ
  girl_girl_games : ℕ

/-- Calculates the number of mixed groups in the chess club -/
def mixed_groups (club : ChessClub) : ℕ :=
  let total_games := club.total_groups * (club.children_per_group.choose 2)
  let mixed_games := total_games - club.boy_boy_games - club.girl_girl_games
  mixed_games / 2

/-- Theorem stating the number of mixed groups in the given scenario -/
theorem chess_club_mixed_groups :
  let club : ChessClub := {
    total_children := 90,
    total_groups := 30,
    children_per_group := 3,
    boy_boy_games := 30,
    girl_girl_games := 14
  }
  mixed_groups club = 23 := by sorry

end NUMINAMATH_CALUDE_chess_club_mixed_groups_l200_20037


namespace NUMINAMATH_CALUDE_xiaoming_coins_l200_20078

theorem xiaoming_coins (first_day : ℕ) (second_day : ℕ)
  (h1 : first_day = 22)
  (h2 : second_day = 12) :
  first_day + second_day = 34 := by
  sorry

end NUMINAMATH_CALUDE_xiaoming_coins_l200_20078


namespace NUMINAMATH_CALUDE_minibus_boys_count_l200_20098

theorem minibus_boys_count : 
  ∀ (total boys girls : ℕ),
  total = 18 →
  boys + girls = total →
  boys = girls - 2 →
  boys = 8 :=
by
  sorry

end NUMINAMATH_CALUDE_minibus_boys_count_l200_20098


namespace NUMINAMATH_CALUDE_derivative_at_one_is_one_fourth_l200_20080

open Real

-- Define the function f
noncomputable def f (f'1 : ℝ) : ℝ → ℝ := λ x => log x - 3 * f'1 * x

-- State the theorem
theorem derivative_at_one_is_one_fourth (f'1 : ℝ) :
  (∀ x > 0, f f'1 x = log x - 3 * f'1 * x) →
  deriv (f f'1) 1 = 1/4 := by sorry

end NUMINAMATH_CALUDE_derivative_at_one_is_one_fourth_l200_20080


namespace NUMINAMATH_CALUDE_eldorado_license_plates_l200_20081

/-- The number of letters in the alphabet used for license plates -/
def alphabet_size : ℕ := 26

/-- The number of digits used for license plates -/
def digit_size : ℕ := 10

/-- The number of letter positions in a license plate -/
def letter_positions : ℕ := 3

/-- The number of digit positions in a license plate -/
def digit_positions : ℕ := 4

/-- The total number of possible valid license plates in Eldorado -/
def total_license_plates : ℕ := alphabet_size ^ letter_positions * digit_size ^ digit_positions

theorem eldorado_license_plates :
  total_license_plates = 175760000 :=
by sorry

end NUMINAMATH_CALUDE_eldorado_license_plates_l200_20081


namespace NUMINAMATH_CALUDE_dividend_calculation_l200_20031

theorem dividend_calculation (divisor quotient remainder : ℕ) 
  (h1 : divisor = 15) 
  (h2 : quotient = 9) 
  (h3 : remainder = 5) : 
  divisor * quotient + remainder = 140 := by
  sorry

end NUMINAMATH_CALUDE_dividend_calculation_l200_20031


namespace NUMINAMATH_CALUDE_polygon_angle_sum_l200_20065

theorem polygon_angle_sum (n : ℕ) (h : n = 5) :
  (n - 2) * 180 + 360 = 900 :=
sorry

end NUMINAMATH_CALUDE_polygon_angle_sum_l200_20065


namespace NUMINAMATH_CALUDE_linear_function_solution_l200_20000

/-- Represents a linear function y = ax + b -/
structure LinearFunction where
  a : ℝ
  b : ℝ

/-- Represents a point (x, y) on the linear function -/
structure Point where
  x : ℝ
  y : ℝ

/-- Given data points for the linear function -/
def dataPoints : List Point := [
  { x := -3, y := -4 },
  { x := -2, y := -2 },
  { x := -1, y := 0 },
  { x := 0, y := 2 },
  { x := 1, y := 4 },
  { x := 2, y := 6 }
]

/-- The linear function satisfies all given data points -/
def satisfiesDataPoints (f : LinearFunction) : Prop :=
  ∀ p ∈ dataPoints, f.a * p.x + f.b = p.y

theorem linear_function_solution (f : LinearFunction) 
  (h : satisfiesDataPoints f) : 
  f.a * 1 + f.b = 4 := by sorry

end NUMINAMATH_CALUDE_linear_function_solution_l200_20000


namespace NUMINAMATH_CALUDE_transfer_schemes_count_l200_20096

/-- The number of torchbearers and segments in the relay --/
def n : ℕ := 6

/-- The set of possible first torchbearers --/
inductive FirstTorchbearer
| A
| B
| C

/-- The set of possible last torchbearers --/
inductive LastTorchbearer
| A
| B

/-- A function to calculate the number of transfer schemes --/
def countTransferSchemes : ℕ :=
  let firstChoices := 3  -- A, B, or C
  let lastChoices := 2   -- A or B
  let middleArrangements := Nat.factorial (n - 2)
  firstChoices * lastChoices * middleArrangements

/-- Theorem stating that the number of transfer schemes is 96 --/
theorem transfer_schemes_count :
  countTransferSchemes = 96 := by
  sorry

end NUMINAMATH_CALUDE_transfer_schemes_count_l200_20096


namespace NUMINAMATH_CALUDE_sum_of_permutations_divisible_by_digit_sum_l200_20019

/-- A type representing a digit from 1 to 9 -/
def Digit := {n : ℕ // 1 ≤ n ∧ n ≤ 9}

/-- A function to calculate the sum of all permutations of a five-digit number -/
def sumOfPermutations (a b c d e : Digit) : ℕ :=
  24 * 11111 * (a.val + b.val + c.val + d.val + e.val)

/-- The theorem statement -/
theorem sum_of_permutations_divisible_by_digit_sum 
  (a b c d e : Digit) 
  (h_distinct : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ 
                b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ 
                c ≠ d ∧ c ≠ e ∧ 
                d ≠ e) : 
  (sumOfPermutations a b c d e) % (a.val + b.val + c.val + d.val + e.val) = 0 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_permutations_divisible_by_digit_sum_l200_20019


namespace NUMINAMATH_CALUDE_total_candies_l200_20093

theorem total_candies (chocolate_boxes caramel_boxes pieces_per_box : ℕ) :
  chocolate_boxes = 6 →
  caramel_boxes = 4 →
  pieces_per_box = 9 →
  chocolate_boxes * pieces_per_box + caramel_boxes * pieces_per_box = 90 :=
by
  sorry

#check total_candies

end NUMINAMATH_CALUDE_total_candies_l200_20093


namespace NUMINAMATH_CALUDE_fraction_value_decreases_as_denominator_increases_l200_20003

theorem fraction_value_decreases_as_denominator_increases 
  (ability : ℝ) (self_estimation : ℝ → ℝ) :
  ability > 0 → (∀ x y, x > 0 ∧ y > 0 ∧ x < y → self_estimation x > self_estimation y) →
  ∀ x y, x > 0 ∧ y > 0 ∧ x < y → ability / self_estimation x > ability / self_estimation y :=
sorry

end NUMINAMATH_CALUDE_fraction_value_decreases_as_denominator_increases_l200_20003


namespace NUMINAMATH_CALUDE_rain_probability_l200_20090

/-- The probability of rain on both Monday and Tuesday given specific conditions -/
theorem rain_probability (p_monday : ℝ) (p_tuesday : ℝ) (p_tuesday_given_no_monday : ℝ) :
  p_monday = 0.4 →
  p_tuesday = 0.3 →
  p_tuesday_given_no_monday = 0.5 →
  p_monday * p_tuesday = 0.12 :=
by sorry

end NUMINAMATH_CALUDE_rain_probability_l200_20090


namespace NUMINAMATH_CALUDE_salary_after_raise_l200_20048

theorem salary_after_raise (original_salary : ℝ) (percentage_increase : ℝ) (new_salary : ℝ) :
  original_salary = 60 →
  percentage_increase = 83.33333333333334 →
  new_salary = original_salary * (1 + percentage_increase / 100) →
  new_salary = 110 := by
  sorry

end NUMINAMATH_CALUDE_salary_after_raise_l200_20048


namespace NUMINAMATH_CALUDE_average_not_two_l200_20085

def data : List ℝ := [1, 1, 0, 2, 4]

theorem average_not_two : 
  (data.sum / data.length) ≠ 2 := by
  sorry

end NUMINAMATH_CALUDE_average_not_two_l200_20085


namespace NUMINAMATH_CALUDE_mean_of_seven_numbers_l200_20007

theorem mean_of_seven_numbers (x y : ℝ) :
  (6 + 14 + x + 17 + 9 + y + 10) / 7 = 13 → x + y = 35 := by
sorry

end NUMINAMATH_CALUDE_mean_of_seven_numbers_l200_20007


namespace NUMINAMATH_CALUDE_positive_A_value_l200_20045

-- Define the # relation
def hash (A B : ℝ) : ℝ := A^2 + B^2

-- Theorem statement
theorem positive_A_value (A : ℝ) (h : hash A 7 = 290) : A = Real.sqrt 241 := by
  sorry

end NUMINAMATH_CALUDE_positive_A_value_l200_20045


namespace NUMINAMATH_CALUDE_prob_two_even_out_of_six_l200_20062

/-- The probability of rolling an even number on a fair six-sided die -/
def prob_even : ℚ := 1/2

/-- The probability of rolling an odd number on a fair six-sided die -/
def prob_odd : ℚ := 1/2

/-- The number of dice rolled -/
def num_dice : ℕ := 6

/-- The number of dice showing even numbers -/
def num_even : ℕ := 2

/-- The number of ways to choose 2 dice out of 6 -/
def ways_to_choose : ℕ := Nat.choose num_dice num_even

theorem prob_two_even_out_of_six :
  (ways_to_choose : ℚ) * prob_even^num_even * prob_odd^(num_dice - num_even) = 15/64 := by
  sorry

end NUMINAMATH_CALUDE_prob_two_even_out_of_six_l200_20062


namespace NUMINAMATH_CALUDE_equal_volumes_condition_l200_20074

/-- Represents the side lengths of the square prisms to be removed from a cube --/
structure PrismSides where
  c : ℝ
  b : ℝ
  a : ℝ

/-- Calculates the volume of the remaining body after removing square prisms --/
def remainingVolume (sides : PrismSides) : ℝ :=
  1 - (sides.c^2 + (sides.b^2 - sides.c^2 * sides.b) + (sides.a^2 - sides.c^2 * sides.a - sides.b^2 * sides.a + sides.c^2 * sides.b))

/-- Theorem stating the conditions for equal volumes --/
theorem equal_volumes_condition (sides : PrismSides) : 
  sides.c = 1/2 ∧ 
  sides.b = (1 + Real.sqrt 17) / 8 ∧ 
  sides.a = (17 + Real.sqrt 17 + Real.sqrt (1202 - 94 * Real.sqrt 17)) / 64 ∧
  sides.c < sides.b ∧ sides.b < sides.a ∧ sides.a < 1 →
  remainingVolume sides = 1/4 :=
sorry

end NUMINAMATH_CALUDE_equal_volumes_condition_l200_20074


namespace NUMINAMATH_CALUDE_simplify_expression_find_a_value_independence_condition_l200_20010

-- Define A and B as functions of a and b
def A (a b : ℝ) : ℝ := 2 * a^2 + 3 * a * b - 2 * a - 1
def B (a b : ℝ) : ℝ := -a^2 + a * b - 1

-- Theorem 1: Simplification of 4A - (3A - 2B)
theorem simplify_expression (a b : ℝ) :
  4 * A a b - (3 * A a b - 2 * B a b) = 5 * a * b - 2 * a - 3 := by sorry

-- Theorem 2: Value of a when b = 1 and 4A - (3A - 2B) = b - 2a
theorem find_a_value (a : ℝ) :
  (4 * A a 1 - (3 * A a 1 - 2 * B a 1) = 1 - 2 * a) → a = 4/5 := by sorry

-- Theorem 3: A + 2B is independent of a iff b = 2/5
theorem independence_condition (b : ℝ) :
  (∀ a₁ a₂ : ℝ, A a₁ b + 2 * B a₁ b = A a₂ b + 2 * B a₂ b) ↔ b = 2/5 := by sorry

end NUMINAMATH_CALUDE_simplify_expression_find_a_value_independence_condition_l200_20010


namespace NUMINAMATH_CALUDE_stamp_revenue_calculation_l200_20008

/-- The total revenue generated from stamp sales --/
theorem stamp_revenue_calculation : 
  let color_price : ℚ := 15/100
  let bw_price : ℚ := 10/100
  let color_sold : ℕ := 578833
  let bw_sold : ℕ := 523776
  let total_revenue := (color_price * color_sold) + (bw_price * bw_sold)
  total_revenue = 139202551/10000 := by
  sorry

end NUMINAMATH_CALUDE_stamp_revenue_calculation_l200_20008


namespace NUMINAMATH_CALUDE_average_salary_l200_20049

def salary_A : ℕ := 8000
def salary_B : ℕ := 5000
def salary_C : ℕ := 16000
def salary_D : ℕ := 7000
def salary_E : ℕ := 9000

def total_salary : ℕ := salary_A + salary_B + salary_C + salary_D + salary_E
def num_people : ℕ := 5

theorem average_salary :
  (total_salary : ℚ) / num_people = 8800 := by sorry

end NUMINAMATH_CALUDE_average_salary_l200_20049


namespace NUMINAMATH_CALUDE_rainfall_volume_calculation_l200_20053

/-- Calculates the total rainfall volume given rainfall rates and area --/
def total_rainfall_volume (rate1 rate2 : ℝ) (area : ℝ) : ℝ :=
  (rate1 * area + rate2 * area) * 0.001

theorem rainfall_volume_calculation :
  let rate1 : ℝ := 5  -- mm/hour
  let rate2 : ℝ := 10 -- mm/hour
  let area : ℝ := 100 -- square meters
  total_rainfall_volume rate1 rate2 area = 1.5 := by
sorry

end NUMINAMATH_CALUDE_rainfall_volume_calculation_l200_20053


namespace NUMINAMATH_CALUDE_hash_example_l200_20028

def hash (a b c d : ℝ) : ℝ := d * b^2 - 4 * a * c

theorem hash_example : hash 2 3 1 (1/2) = -3.5 := by
  sorry

end NUMINAMATH_CALUDE_hash_example_l200_20028


namespace NUMINAMATH_CALUDE_matrix_determinant_solution_l200_20039

theorem matrix_determinant_solution (b : ℝ) (hb : b ≠ 0) :
  let y : ℝ := -b / 2
  ∃ (y : ℝ), Matrix.det
    ![![y + b, y, y],
      ![y, y + b, y],
      ![y, y, y + b]] = 0 ↔ y = -b / 2 := by
sorry

end NUMINAMATH_CALUDE_matrix_determinant_solution_l200_20039


namespace NUMINAMATH_CALUDE_perpendicular_line_equation_l200_20087

-- Define the given line
def given_line (x y : ℝ) : Prop := x + y - 5 = 0

-- Define the point that the perpendicular line passes through
def point : ℝ × ℝ := (2, -1)

-- Define the perpendicular line
def perpendicular_line (x y : ℝ) : Prop := x - y - 3 = 0

-- Theorem statement
theorem perpendicular_line_equation :
  ∃ (m b : ℝ),
    (∀ x y, perpendicular_line x y ↔ y = m * x + b) ∧
    perpendicular_line point.1 point.2 ∧
    (∀ x₁ y₁ x₂ y₂, given_line x₁ y₁ → given_line x₂ y₂ → x₁ ≠ x₂ →
      (y₂ - y₁) / (x₂ - x₁) * m = -1) :=
sorry

end NUMINAMATH_CALUDE_perpendicular_line_equation_l200_20087


namespace NUMINAMATH_CALUDE_june_bird_eggs_l200_20061

/-- The number of eggs in each nest in the first tree -/
def eggs_per_nest_tree1 : ℕ := 5

/-- The number of nests in the first tree -/
def nests_in_tree1 : ℕ := 2

/-- The number of eggs in the nest in the second tree -/
def eggs_in_tree2 : ℕ := 3

/-- The number of eggs in the nest in the front yard -/
def eggs_in_front_yard : ℕ := 4

/-- The total number of bird eggs June found -/
def total_eggs : ℕ := nests_in_tree1 * eggs_per_nest_tree1 + eggs_in_tree2 + eggs_in_front_yard

theorem june_bird_eggs : total_eggs = 17 := by
  sorry

end NUMINAMATH_CALUDE_june_bird_eggs_l200_20061


namespace NUMINAMATH_CALUDE_semicircle_radius_l200_20023

theorem semicircle_radius (a b c : ℝ) :
  a > 0 ∧ b > 0 ∧ c > 0 →
  a^2 + b^2 = c^2 →
  (1/2) * Real.pi * (a/2)^2 = 8 * Real.pi →
  Real.pi * (b/2) = 8.5 * Real.pi →
  c/2 = 7.5 := by
sorry

end NUMINAMATH_CALUDE_semicircle_radius_l200_20023


namespace NUMINAMATH_CALUDE_average_of_six_numbers_l200_20077

theorem average_of_six_numbers 
  (total_average : ℝ)
  (second_pair_average : ℝ)
  (third_pair_average : ℝ)
  (h1 : total_average = 6.40)
  (h2 : second_pair_average = 6.1)
  (h3 : third_pair_average = 6.9) :
  ∃ (first_pair_average : ℝ),
    first_pair_average = 6.2 ∧
    (first_pair_average + second_pair_average + third_pair_average) / 3 = total_average :=
by sorry

end NUMINAMATH_CALUDE_average_of_six_numbers_l200_20077


namespace NUMINAMATH_CALUDE_maria_josh_age_sum_l200_20006

/-- Proves that given the conditions about Maria and Josh's ages, the sum of their current ages is 31 years -/
theorem maria_josh_age_sum : 
  ∀ (maria josh : ℝ), 
  (maria = josh + 8) → 
  (maria + 6 = 3 * (josh - 3)) → 
  (maria + josh = 31) := by
sorry

end NUMINAMATH_CALUDE_maria_josh_age_sum_l200_20006


namespace NUMINAMATH_CALUDE_problem_solution_l200_20014

/-- The set of integers of the form m^k for integers m, k ≥ 2 -/
def S : Set ℕ := {n : ℕ | ∃ m k : ℕ, m ≥ 2 ∧ k ≥ 2 ∧ n = m^k}

/-- The number of ways to write n as the sum of distinct elements of S -/
def f (n : ℕ) : ℕ := sorry

/-- The set of integers for which f(n) = 3 -/
def T : Set ℕ := {n : ℕ | f n = 3}

theorem problem_solution :
  (f 30 = 0) ∧
  (∀ n : ℕ, n ≥ 31 → f n ≥ 1) ∧
  (T.Finite ∧ T.Nonempty) ∧
  (∃ m : ℕ, m ∈ T ∧ ∀ n ∈ T, n ≤ m) ∧
  (∃ m : ℕ, m ∈ T ∧ ∀ n ∈ T, n ≤ m ∧ m = 111) :=
by sorry

end NUMINAMATH_CALUDE_problem_solution_l200_20014


namespace NUMINAMATH_CALUDE_probability_of_flush_l200_20059

/-- Represents a standard deck of cards -/
def StandardDeck : ℕ := 52

/-- Represents the number of suits in a standard deck -/
def NumSuits : ℕ := 4

/-- Represents the number of cards chosen -/
def CardsChosen : ℕ := 6

/-- Represents the number of cards in each suit -/
def CardsPerSuit : ℕ := StandardDeck / NumSuits

/-- Calculates the number of ways to choose n items from k items -/
def choose (n k : ℕ) : ℕ := sorry

/-- Probability of forming a flush when choosing 6 cards at random from a standard 52-card deck -/
theorem probability_of_flush : 
  (NumSuits * choose CardsPerSuit CardsChosen) / choose StandardDeck CardsChosen = 3432 / 10179260 := by
  sorry

end NUMINAMATH_CALUDE_probability_of_flush_l200_20059


namespace NUMINAMATH_CALUDE_jogging_time_calculation_l200_20016

theorem jogging_time_calculation (total_distance : ℝ) (total_time : ℝ) (initial_speed : ℝ) (later_speed : ℝ)
  (h1 : total_distance = 160)
  (h2 : total_time = 8)
  (h3 : initial_speed = 15)
  (h4 : later_speed = 10) :
  ∃ (initial_time : ℝ),
    initial_time * initial_speed + (total_time - initial_time) * later_speed = total_distance ∧
    initial_time = 16 / 5 := by
  sorry

end NUMINAMATH_CALUDE_jogging_time_calculation_l200_20016


namespace NUMINAMATH_CALUDE_ellipse_equation_for_given_conditions_l200_20041

/-- Represents an ellipse with center at the origin -/
structure Ellipse where
  a : ℝ  -- Semi-major axis
  b : ℝ  -- Semi-minor axis
  c : ℝ  -- Semi-focal distance

/-- The standard equation of an ellipse -/
def Ellipse.equation (e : Ellipse) (x y : ℝ) : Prop :=
  x^2 / e.a^2 + y^2 / e.b^2 = 1

theorem ellipse_equation_for_given_conditions :
  ∀ e : Ellipse,
  e.a = 6 →                  -- Major axis is 12 (2a = 12)
  e.c / e.a = 1 / 3 →        -- Eccentricity is 1/3
  e.c = 2 →                  -- Derived from eccentricity and semi-major axis
  e.b^2 = e.a^2 - e.c^2 →    -- Relationship between a, b, and c
  ∀ x y : ℝ,
  e.equation x y ↔ x^2 / 36 + y^2 / 32 = 1 :=
by sorry

end NUMINAMATH_CALUDE_ellipse_equation_for_given_conditions_l200_20041


namespace NUMINAMATH_CALUDE_max_pieces_is_112_l200_20001

/-- Represents the dimensions of a rectangular cake cut into square pieces -/
structure CakeDimensions where
  m : ℕ
  n : ℕ

/-- Calculates the number of interior pieces in a cake -/
def interiorPieces (d : CakeDimensions) : ℕ :=
  if d.m > 2 ∧ d.n > 2 then (d.m - 2) * (d.n - 2) else 0

/-- Calculates the number of exterior pieces in a cake -/
def exteriorPieces (d : CakeDimensions) : ℕ :=
  d.m * d.n - interiorPieces d

/-- Checks if the cake satisfies the condition that exterior pieces are twice the interior pieces -/
def satisfiesCondition (d : CakeDimensions) : Prop :=
  exteriorPieces d = 2 * interiorPieces d

/-- The theorem stating that the maximum number of pieces under the given conditions is 112 -/
theorem max_pieces_is_112 :
  ∃ d : CakeDimensions, satisfiesCondition d ∧
  ∀ d' : CakeDimensions, satisfiesCondition d' → d.m * d.n ≥ d'.m * d'.n ∧ d.m * d.n = 112 :=
sorry

end NUMINAMATH_CALUDE_max_pieces_is_112_l200_20001


namespace NUMINAMATH_CALUDE_problem_solution_l200_20027

theorem problem_solution (x y : ℝ) (h : y = Real.sqrt (x - 4) - Real.sqrt (4 - x) + 2023) :
  y - x^2 + 17 = 2024 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l200_20027


namespace NUMINAMATH_CALUDE_modulus_product_complex_l200_20030

theorem modulus_product_complex : |(7 - 4*I)*(3 + 11*I)| = Real.sqrt 8450 := by
  sorry

end NUMINAMATH_CALUDE_modulus_product_complex_l200_20030


namespace NUMINAMATH_CALUDE_sally_seashell_money_l200_20069

/-- The number of seashells Sally picks on Monday -/
def monday_seashells : ℕ := 30

/-- The number of seashells Sally picks on Tuesday -/
def tuesday_seashells : ℕ := monday_seashells / 2

/-- The price of each seashell in dollars -/
def seashell_price : ℚ := 6/5

/-- The total number of seashells Sally picks -/
def total_seashells : ℕ := monday_seashells + tuesday_seashells

/-- The total money Sally can make by selling all her seashells -/
def total_money : ℚ := (total_seashells : ℚ) * seashell_price

theorem sally_seashell_money : total_money = 54 := by
  sorry

end NUMINAMATH_CALUDE_sally_seashell_money_l200_20069


namespace NUMINAMATH_CALUDE_additional_batches_is_seven_l200_20029

/-- Represents the number of cups of flour needed for one batch of cookies -/
def flour_per_batch : ℕ := 2

/-- Represents the number of batches of cookies Gigi baked -/
def batches_baked : ℕ := 3

/-- Represents the initial amount of flour in cups -/
def initial_flour : ℕ := 20

/-- Calculates the number of additional batches that can be made with the remaining flour -/
def additional_batches : ℕ :=
  (initial_flour - flour_per_batch * batches_baked) / flour_per_batch

/-- Theorem stating that the number of additional batches is 7 -/
theorem additional_batches_is_seven :
  additional_batches = 7 := by sorry

end NUMINAMATH_CALUDE_additional_batches_is_seven_l200_20029


namespace NUMINAMATH_CALUDE_tan_half_angle_less_than_one_l200_20070

theorem tan_half_angle_less_than_one (θ : Real) (h : 0 < θ ∧ θ < π / 2) : 
  Real.tan (θ / 2) < 1 := by
  sorry

end NUMINAMATH_CALUDE_tan_half_angle_less_than_one_l200_20070


namespace NUMINAMATH_CALUDE_isosceles_triangle_side_length_l200_20094

-- Define the triangle ABC
structure Triangle (α : Type*) [NormedAddCommGroup α] [InnerProductSpace ℝ α] :=
  (A B C : α)

-- Define an isosceles triangle
def IsIsosceles {α : Type*} [NormedAddCommGroup α] [InnerProductSpace ℝ α] (t : Triangle α) : Prop :=
  ‖t.A - t.B‖ = ‖t.A - t.C‖

-- Define the incenter
def Incenter {α : Type*} [NormedAddCommGroup α] [InnerProductSpace ℝ α] (t : Triangle α) : α :=
  sorry  -- Definition of incenter omitted for brevity

-- Define the distance from a point to a line segment
def DistanceToSegment {α : Type*} [NormedAddCommGroup α] [InnerProductSpace ℝ α] 
  (P : α) (A B : α) : ℝ :=
  sorry  -- Definition of distance to segment omitted for brevity

-- Theorem statement
theorem isosceles_triangle_side_length 
  {α : Type*} [NormedAddCommGroup α] [InnerProductSpace ℝ α] 
  (t : Triangle α) (I : α) :
  IsIsosceles t →
  I = Incenter t →
  ‖t.A - I‖ = 3 →
  DistanceToSegment I t.B t.C = 2 →
  ‖t.B - t.C‖ = 4 * Real.sqrt 5 :=
sorry

end NUMINAMATH_CALUDE_isosceles_triangle_side_length_l200_20094


namespace NUMINAMATH_CALUDE_percentage_increase_l200_20066

theorem percentage_increase (initial : ℝ) (final : ℝ) : 
  initial = 400 → final = 480 → (final - initial) / initial * 100 = 20 := by
  sorry

end NUMINAMATH_CALUDE_percentage_increase_l200_20066


namespace NUMINAMATH_CALUDE_triangle_max_area_l200_20091

theorem triangle_max_area (a b c : ℝ) (h1 : (a + b - c) * (a + b + c) = 3 * a * b) (h2 : c = 4) :
  ∃ (S : ℝ), S = (4 : ℝ) * Real.sqrt 3 ∧ ∀ (area : ℝ), area = 1/2 * a * b * Real.sin (Real.arccos ((a^2 + b^2 - c^2) / (2 * a * b))) → area ≤ S :=
sorry

end NUMINAMATH_CALUDE_triangle_max_area_l200_20091


namespace NUMINAMATH_CALUDE_increase_averages_possible_l200_20060

def group1 : List ℕ := [5, 3, 5, 3, 5, 4, 3, 4, 3, 4, 5, 5]
def group2 : List ℕ := [3, 4, 5, 2, 3, 2, 5, 4, 5, 3]

def average (l : List ℕ) : ℚ :=
  (l.sum : ℚ) / l.length

theorem increase_averages_possible :
  ∃ g ∈ group1,
    average (group1.filter (· ≠ g)) > average group1 ∧
    average (g :: group2) > average group2 := by
  sorry

end NUMINAMATH_CALUDE_increase_averages_possible_l200_20060


namespace NUMINAMATH_CALUDE_double_root_condition_l200_20017

/-- The equation has a double root when k is either 3 or 1/3 -/
theorem double_root_condition (k : ℝ) : 
  (∃ x : ℝ, (k - 1) / (x^2 - 1) - 1 / (x - 1) = k / (x + 1) ∧ 
   ∀ y : ℝ, (k - 1) / (y^2 - 1) - 1 / (y - 1) = k / (y + 1) → y = x) ↔ 
  (k = 3 ∨ k = 1/3) :=
sorry

end NUMINAMATH_CALUDE_double_root_condition_l200_20017


namespace NUMINAMATH_CALUDE_paul_lives_on_fifth_story_l200_20056

/-- The number of stories in Paul's apartment building -/
def S : ℕ := sorry

/-- The number of trips Paul makes each day -/
def trips_per_day : ℕ := 3

/-- The height of each story in feet -/
def story_height : ℕ := 10

/-- The number of days in a week -/
def days_in_week : ℕ := 7

/-- The total vertical distance Paul travels in a week in feet -/
def total_vertical_distance : ℕ := 2100

theorem paul_lives_on_fifth_story :
  S * story_height * trips_per_day * 2 * days_in_week = total_vertical_distance →
  S = 5 := by
  sorry

end NUMINAMATH_CALUDE_paul_lives_on_fifth_story_l200_20056


namespace NUMINAMATH_CALUDE_range_of_a_l200_20034

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, x > a → x > 2) ∧ (∃ x : ℝ, x > 2 ∧ x ≤ a) ↔ a > 2 :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l200_20034


namespace NUMINAMATH_CALUDE_product_of_reciprocal_differences_l200_20004

theorem product_of_reciprocal_differences (a b c : ℝ) 
  (h_pos_a : a > 0) (h_pos_b : b > 0) (h_pos_c : c > 0)
  (h_sum : a + b + c = 1) : 
  (1/a - 1) * (1/b - 1) * (1/c - 1) ≥ 8 := by
  sorry

end NUMINAMATH_CALUDE_product_of_reciprocal_differences_l200_20004


namespace NUMINAMATH_CALUDE_bucket_water_calculation_l200_20076

/-- Given an initial amount of water and an amount poured out, 
    calculate the remaining amount of water in the bucket. -/
def water_remaining (initial : ℝ) (poured_out : ℝ) : ℝ :=
  initial - poured_out

/-- Theorem stating that given 0.8 gallon initially and 0.2 gallon poured out,
    the remaining amount is 0.6 gallon. -/
theorem bucket_water_calculation :
  water_remaining 0.8 0.2 = 0.6 := by
  sorry

#eval water_remaining 0.8 0.2

end NUMINAMATH_CALUDE_bucket_water_calculation_l200_20076


namespace NUMINAMATH_CALUDE_sqrt_four_squared_l200_20079

theorem sqrt_four_squared : (Real.sqrt 4)^2 = 4 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_four_squared_l200_20079


namespace NUMINAMATH_CALUDE_digit_equation_sum_l200_20025

theorem digit_equation_sum : 
  ∀ (E M V Y : ℕ),
  (E < 10) → (M < 10) → (V < 10) → (Y < 10) →
  (V ≥ 1) →
  (Y ≠ 0) → (M ≠ 0) →
  (E ≠ M) → (E ≠ V) → (E ≠ Y) → 
  (M ≠ V) → (M ≠ Y) → 
  (V ≠ Y) →
  ((10 * Y + E) * (10 * M + E) = 111 * V) →
  (E + M + V + Y = 21) := by
sorry

end NUMINAMATH_CALUDE_digit_equation_sum_l200_20025


namespace NUMINAMATH_CALUDE_percentage_relation_l200_20084

theorem percentage_relation (A B C : ℝ) 
  (h1 : A = 0.05 * C) 
  (h2 : A = 0.20 * B) : 
  B = 0.25 * C := by
sorry

end NUMINAMATH_CALUDE_percentage_relation_l200_20084


namespace NUMINAMATH_CALUDE_order_of_powers_l200_20050

theorem order_of_powers : 5^56 < 31^28 ∧ 31^28 < 17^35 ∧ 17^35 < 10^51 := by
  sorry

end NUMINAMATH_CALUDE_order_of_powers_l200_20050


namespace NUMINAMATH_CALUDE_min_value_p_l200_20032

theorem min_value_p (p q r s : ℝ) 
  (sum_eq : p + q + r + s = 10)
  (sum_prod_eq : p*q + p*r + p*s + q*r + q*s + r*s = 20)
  (prod_sq_eq : p^2 * q^2 * r^2 * s^2 = 16) :
  ∃ (min_p : ℝ), min_p = 2 ∧ p ≥ min_p := by
  sorry

end NUMINAMATH_CALUDE_min_value_p_l200_20032


namespace NUMINAMATH_CALUDE_y_intercept_of_line_l200_20095

/-- The y-intercept of the line 3x - 5y = 15 is -3 -/
theorem y_intercept_of_line (x y : ℝ) : 3 * x - 5 * y = 15 → x = 0 → y = -3 := by
  sorry

end NUMINAMATH_CALUDE_y_intercept_of_line_l200_20095
