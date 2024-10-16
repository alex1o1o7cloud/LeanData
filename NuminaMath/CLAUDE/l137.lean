import Mathlib

namespace NUMINAMATH_CALUDE_max_shareholder_percentage_l137_13789

theorem max_shareholder_percentage (n : ℕ) (k : ℕ) (p : ℚ) (h1 : n = 100) (h2 : k = 66) (h3 : p = 1/2) :
  ∀ (f : ℕ → ℚ),
    (∀ i, 0 ≤ f i) →
    (∀ i, i < n → f i ≤ 1) →
    (∀ s : Finset ℕ, s.card = k → s.sum f ≥ p) →
    (∀ i, i < n → f i ≤ 1/4) :=
sorry

end NUMINAMATH_CALUDE_max_shareholder_percentage_l137_13789


namespace NUMINAMATH_CALUDE_sticks_at_20th_stage_l137_13791

/-- The number of sticks in the nth stage of the pattern -/
def sticks : ℕ → ℕ
| 0 => 5  -- Initial stage (indexed as 0)
| n + 1 => if n < 10 then sticks n + 3 else sticks n + 4

/-- The theorem stating that the 20th stage (indexed as 19) has 68 sticks -/
theorem sticks_at_20th_stage : sticks 19 = 68 := by
  sorry

end NUMINAMATH_CALUDE_sticks_at_20th_stage_l137_13791


namespace NUMINAMATH_CALUDE_janice_pebbles_l137_13700

/-- The number of pebbles each friend received -/
def pebbles_per_friend : ℕ := 4

/-- The number of friends who received pebbles -/
def number_of_friends : ℕ := 9

/-- The total number of pebbles Janice gave away -/
def total_pebbles : ℕ := pebbles_per_friend * number_of_friends

theorem janice_pebbles : total_pebbles = 36 := by
  sorry

end NUMINAMATH_CALUDE_janice_pebbles_l137_13700


namespace NUMINAMATH_CALUDE_triangle_case1_triangle_case2_l137_13773

-- Case 1
theorem triangle_case1 (AB AD HM : ℝ) (h1 : AB = 10) (h2 : AD = 4) (h3 : HM = 6/5) :
  let BD := Real.sqrt (AB^2 - AD^2)
  let DH := (4 * Real.sqrt 21) / 5
  let DC := BD - HM
  DC = (8 * Real.sqrt 21 - 12) / 5 := by sorry

-- Case 2
theorem triangle_case2 (AB AD HM : ℝ) (h1 : AB = 8 * Real.sqrt 2) (h2 : AD = 4) (h3 : HM = Real.sqrt 2) :
  let BD := Real.sqrt (AB^2 - AD^2)
  let DH := Real.sqrt 14
  let DC := BD - HM
  DC = 2 * Real.sqrt 14 - 2 * Real.sqrt 2 := by sorry

end NUMINAMATH_CALUDE_triangle_case1_triangle_case2_l137_13773


namespace NUMINAMATH_CALUDE_equilateral_is_peculiar_specific_right_triangle_is_peculiar_right_angled_peculiar_triangle_ratio_l137_13724

-- Definition of a peculiar triangle
def is_peculiar_triangle (a b c : ℝ) : Prop :=
  a^2 + b^2 = 2*c^2 ∨ a^2 + c^2 = 2*b^2 ∨ b^2 + c^2 = 2*a^2

-- Definition of an equilateral triangle
def is_equilateral_triangle (a b c : ℝ) : Prop :=
  a = b ∧ b = c

-- Definition of a right triangle
def is_right_triangle (a b c : ℝ) : Prop :=
  a^2 + b^2 = c^2

-- Theorem 1: An equilateral triangle is a peculiar triangle
theorem equilateral_is_peculiar (a b c : ℝ) :
  is_equilateral_triangle a b c → is_peculiar_triangle a b c :=
sorry

-- Theorem 2: A right triangle with sides 5√2, 10, and 5√6 is a peculiar triangle
theorem specific_right_triangle_is_peculiar :
  let a : ℝ := 5 * Real.sqrt 2
  let b : ℝ := 5 * Real.sqrt 6
  let c : ℝ := 10
  is_right_triangle a b c ∧ is_peculiar_triangle a b c :=
sorry

-- Theorem 3: In a right-angled peculiar triangle, the ratio of sides is 1:√2:√3
theorem right_angled_peculiar_triangle_ratio (a b c : ℝ) (h1 : a > 0) (h2 : b > a) :
  is_right_triangle a b c ∧ is_peculiar_triangle a b c →
  ∃ (k : ℝ), a = k ∧ b = k * Real.sqrt 2 ∧ c = k * Real.sqrt 3 :=
sorry

end NUMINAMATH_CALUDE_equilateral_is_peculiar_specific_right_triangle_is_peculiar_right_angled_peculiar_triangle_ratio_l137_13724


namespace NUMINAMATH_CALUDE_total_laundry_count_l137_13702

/-- Represents the number of items for each person and shared items -/
structure LaundryItems where
  cally : Nat
  danny : Nat
  emily : Nat
  cally_danny : Nat
  emily_danny : Nat
  cally_emily : Nat

/-- Calculates the total number of laundry items -/
def total_laundry (items : LaundryItems) : Nat :=
  items.cally + items.danny + items.emily + items.cally_danny + items.emily_danny + items.cally_emily

/-- Theorem: The total number of clothes and accessories washed is 141 -/
theorem total_laundry_count :
  ∃ (items : LaundryItems),
    items.cally = 40 ∧
    items.danny = 39 ∧
    items.emily = 39 ∧
    items.cally_danny = 8 ∧
    items.emily_danny = 6 ∧
    items.cally_emily = 9 ∧
    total_laundry items = 141 := by
  sorry

end NUMINAMATH_CALUDE_total_laundry_count_l137_13702


namespace NUMINAMATH_CALUDE_area_sin_3x_l137_13771

open Real MeasureTheory

/-- The area of a function f on [a, b] -/
noncomputable def area (f : ℝ → ℝ) (a b : ℝ) : ℝ := ∫ x in a..b, f x

/-- For any positive integer n, the area of sin(nx) on [0, π/n] is 2/n -/
axiom area_sin_nx (n : ℕ+) : area (fun x ↦ sin (n * x)) 0 (π / n) = 2 / n

/-- The area of sin(3x) on [0, π/3] is 2/3 -/
theorem area_sin_3x : area (fun x ↦ sin (3 * x)) 0 (π / 3) = 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_area_sin_3x_l137_13771


namespace NUMINAMATH_CALUDE_jane_quiz_score_l137_13769

/-- Represents the scoring system for a quiz -/
structure QuizScoring where
  correct : Int
  incorrect : Int
  unanswered : Int

/-- Represents a student's quiz results -/
structure QuizResults where
  total : Nat
  correct : Nat
  incorrect : Nat
  unanswered : Nat

/-- Calculates the final score based on quiz results and scoring system -/
def calculateScore (results : QuizResults) (scoring : QuizScoring) : Int :=
  results.correct * scoring.correct + 
  results.incorrect * scoring.incorrect + 
  results.unanswered * scoring.unanswered

/-- Theorem: Jane's final score in the quiz is 20 -/
theorem jane_quiz_score : 
  let scoring : QuizScoring := ⟨2, -1, 0⟩
  let results : QuizResults := ⟨30, 15, 10, 5⟩
  calculateScore results scoring = 20 := by
  sorry


end NUMINAMATH_CALUDE_jane_quiz_score_l137_13769


namespace NUMINAMATH_CALUDE_log_of_negative_one_not_real_l137_13737

/-- For b > 0 and b ≠ 1, log_b(-1) is not a real number -/
theorem log_of_negative_one_not_real (b : ℝ) (hb_pos : b > 0) (hb_ne_one : b ≠ 1) :
  ¬ ∃ (y : ℝ), b^y = -1 :=
sorry

end NUMINAMATH_CALUDE_log_of_negative_one_not_real_l137_13737


namespace NUMINAMATH_CALUDE_undefined_function_roots_sum_l137_13711

theorem undefined_function_roots_sum : 
  let f (x : ℝ) := 3 * x^2 - 9 * x + 6
  ∃ (r₁ r₂ : ℝ), f r₁ = 0 ∧ f r₂ = 0 ∧ r₁ ≠ r₂ ∧ r₁ + r₂ = 3 := by
  sorry

end NUMINAMATH_CALUDE_undefined_function_roots_sum_l137_13711


namespace NUMINAMATH_CALUDE_square_root_equation_solution_l137_13782

theorem square_root_equation_solution :
  ∃! z : ℚ, Real.sqrt (5 - 4 * z) = 7 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_square_root_equation_solution_l137_13782


namespace NUMINAMATH_CALUDE_squirrel_nut_difference_squirrel_nut_difference_example_l137_13787

theorem squirrel_nut_difference : ℕ → ℕ → ℕ
  | num_squirrels, num_nuts =>
    num_squirrels - num_nuts

theorem squirrel_nut_difference_example : squirrel_nut_difference 4 2 = 2 := by
  sorry

end NUMINAMATH_CALUDE_squirrel_nut_difference_squirrel_nut_difference_example_l137_13787


namespace NUMINAMATH_CALUDE_garage_spokes_count_l137_13707

/-- Represents a bicycle or tricycle -/
structure Vehicle where
  front_spokes : ℕ
  back_spokes : ℕ
  middle_spokes : Option ℕ

/-- The collection of vehicles in the garage -/
def garage : List Vehicle :=
  [
    { front_spokes := 12, back_spokes := 10, middle_spokes := none },
    { front_spokes := 14, back_spokes := 12, middle_spokes := none },
    { front_spokes := 10, back_spokes := 14, middle_spokes := none },
    { front_spokes := 14, back_spokes := 16, middle_spokes := some 12 }
  ]

/-- Calculates the total number of spokes for a single vehicle -/
def spokes_per_vehicle (v : Vehicle) : ℕ :=
  v.front_spokes + v.back_spokes + (v.middle_spokes.getD 0)

/-- Calculates the total number of spokes in the garage -/
def total_spokes : ℕ :=
  garage.map spokes_per_vehicle |>.sum

/-- Theorem stating that the total number of spokes in the garage is 114 -/
theorem garage_spokes_count : total_spokes = 114 := by
  sorry

end NUMINAMATH_CALUDE_garage_spokes_count_l137_13707


namespace NUMINAMATH_CALUDE_figure_area_solution_l137_13756

theorem figure_area_solution (x : ℝ) : 
  let square1_area := (3*x)^2
  let square2_area := (7*x)^2
  let triangle_area := (1/2) * (3*x) * (7*x)
  let total_area := square1_area + square2_area + triangle_area
  total_area = 1360 → x = Real.sqrt (2720/119) := by
sorry

end NUMINAMATH_CALUDE_figure_area_solution_l137_13756


namespace NUMINAMATH_CALUDE_distance_between_points_on_lines_l137_13705

/-- Given two lines and points A and B on these lines, prove that the distance between A and B is 5 -/
theorem distance_between_points_on_lines (a : ℝ) :
  let line1 := λ (x y : ℝ) => 3 * a * x - y - 2 = 0
  let line2 := λ (x y : ℝ) => (2 * a - 1) * x + 3 * a * y - 3 = 0
  let A : ℝ × ℝ := (0, -2)
  let B : ℝ × ℝ := (-3, 2)
  line1 A.1 A.2 ∧ line2 B.1 B.2 →
  Real.sqrt ((B.1 - A.1)^2 + (B.2 - A.2)^2) = 5 := by
sorry


end NUMINAMATH_CALUDE_distance_between_points_on_lines_l137_13705


namespace NUMINAMATH_CALUDE_least_possible_bananas_l137_13730

/-- Represents the distribution of bananas among four monkeys -/
structure BananaDistribution where
  total : ℕ
  first : ℕ
  second : ℕ
  third : ℕ
  fourth : ℕ

/-- Checks if the distribution satisfies the problem conditions -/
def isValidDistribution (d : BananaDistribution) : Prop :=
  d.total = d.first + d.second + d.third + d.fourth ∧
  (3 * d.first * 5 + (d.second + d.third + d.fourth)) % 20 = 0 ∧
  (d.second + (d.first + d.third + d.fourth)) % 4 = 0 ∧
  (d.third + (d.first + d.second + d.fourth)) % 4 = 0 ∧
  (d.fourth + 2 * (d.first + d.second + d.third)) % 8 = 0 ∧
  4 * (d.fourth + (d.first + d.second + d.third) / 4) =
  (3 * d.first * 5 + (d.second + d.third + d.fourth)) / 4 ∧
  3 * (d.fourth + (d.first + d.second + d.third) / 4) =
  (2 * d.second + (d.first + d.third + d.fourth)) / 2 ∧
  2 * (d.fourth + (d.first + d.second + d.third) / 4) =
  (d.third + (d.first + d.second + d.fourth)) / 4 ∧
  (d.fourth + (d.first + d.second + d.third) / 4) =
  (d.fourth / 8 + (d.first + d.second + d.third) / 4)

theorem least_possible_bananas :
  ∀ d : BananaDistribution,
    isValidDistribution d →
    d.total ≥ 600 :=
sorry

end NUMINAMATH_CALUDE_least_possible_bananas_l137_13730


namespace NUMINAMATH_CALUDE_two_from_ten_for_different_positions_l137_13751

/-- The number of ways to choose k items from n items where order matters -/
def permutations (n k : ℕ) : ℕ := (n - k + 1).factorial / (n - k).factorial

/-- The number of ways to choose 2 people from 10 for 2 different positions -/
theorem two_from_ten_for_different_positions : permutations 10 2 = 90 := by
  sorry

end NUMINAMATH_CALUDE_two_from_ten_for_different_positions_l137_13751


namespace NUMINAMATH_CALUDE_ac_length_l137_13734

/-- Given a line segment AB of length 4 with a point C on AB, 
    prove that if AC is the mean proportional between AB and BC, 
    then the length of AC is 2√5 - 2 -/
theorem ac_length (A B C : ℝ) (h1 : B - A = 4) (h2 : A ≤ C ∧ C ≤ B) 
  (h3 : (C - A)^2 = (B - A) * (B - C)) : C - A = 2 * Real.sqrt 5 - 2 := by
  sorry

end NUMINAMATH_CALUDE_ac_length_l137_13734


namespace NUMINAMATH_CALUDE_perpendicular_line_equation_l137_13717

/-- Given a line L1 with equation x - 2y - 2 = 0 and a point P (5, 3),
    the line L2 passing through P and perpendicular to L1 has equation 2x + y - 13 = 0 -/
theorem perpendicular_line_equation (L1 : Set (ℝ × ℝ)) (P : ℝ × ℝ) :
  (L1 = {(x, y) | x - 2*y - 2 = 0}) →
  (P = (5, 3)) →
  (∃ L2 : Set (ℝ × ℝ), 
    (P ∈ L2) ∧ 
    (∀ (v w : ℝ × ℝ), v ∈ L1 → w ∈ L1 → v ≠ w → 
      ∀ (p q : ℝ × ℝ), p ∈ L2 → q ∈ L2 → p ≠ q → 
        ((v.1 - w.1) * (p.1 - q.1) + (v.2 - w.2) * (p.2 - q.2) = 0)) ∧
    (L2 = {(x, y) | 2*x + y - 13 = 0})) :=
by
  sorry

end NUMINAMATH_CALUDE_perpendicular_line_equation_l137_13717


namespace NUMINAMATH_CALUDE_star_equal_is_diagonal_l137_13758

/-- The star operation defined on real numbers -/
def star (a b : ℝ) : ℝ := a * b * (a - b)

/-- The set of points (x, y) where x ★ y = y ★ x -/
def star_equal_set : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | star p.1 p.2 = star p.2 p.1}

/-- The line y = x in ℝ² -/
def diagonal_line : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1 = p.2}

theorem star_equal_is_diagonal :
  star_equal_set = diagonal_line := by sorry

end NUMINAMATH_CALUDE_star_equal_is_diagonal_l137_13758


namespace NUMINAMATH_CALUDE_intersection_M_N_l137_13714

def M : Set ℝ := {x | x > -3}
def N : Set ℝ := {x | x ≥ 2}

theorem intersection_M_N : M ∩ N = Set.Ici 2 := by sorry

end NUMINAMATH_CALUDE_intersection_M_N_l137_13714


namespace NUMINAMATH_CALUDE_munchausen_claim_correct_l137_13763

-- Define a function to calculate the sum of digits
def sumOfDigits (n : ℕ) : ℕ := sorry

-- Theorem statement
theorem munchausen_claim_correct : 
  ∃ (a b : ℕ), 
    (10^9 ≤ a ∧ a < 10^10) ∧ 
    (10^9 ≤ b ∧ b < 10^10) ∧ 
    a ≠ b ∧ 
    a % 10 ≠ 0 ∧ 
    b % 10 ≠ 0 ∧ 
    a + sumOfDigits (a^2) = b + sumOfDigits (b^2) := by
  sorry

end NUMINAMATH_CALUDE_munchausen_claim_correct_l137_13763


namespace NUMINAMATH_CALUDE_fraction_equality_l137_13795

theorem fraction_equality (m n r t : ℚ) 
  (h1 : m / n = 4 / 3) 
  (h2 : r / t = 9 / 14) : 
  (3 * m * r - n * t) / (4 * n * t - 7 * m * r) = -11 / 14 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l137_13795


namespace NUMINAMATH_CALUDE_buckingham_palace_visitors_l137_13732

theorem buckingham_palace_visitors (previous_day_visitors : ℕ) (additional_visitors : ℕ) 
  (h1 : previous_day_visitors = 295) 
  (h2 : additional_visitors = 22) : 
  previous_day_visitors + additional_visitors = 317 := by
  sorry

end NUMINAMATH_CALUDE_buckingham_palace_visitors_l137_13732


namespace NUMINAMATH_CALUDE_rectangular_box_volume_l137_13783

theorem rectangular_box_volume (x : ℕ) (h : x > 0) :
  (∃ (a b c : ℕ), a > 0 ∧ b > 0 ∧ c > 0 ∧ 
   a = x ∧ b = 3*x ∧ c = 4*x ∧
   a * b * c = 96) ↔ x = 2 :=
sorry

end NUMINAMATH_CALUDE_rectangular_box_volume_l137_13783


namespace NUMINAMATH_CALUDE_train_distance_45_minutes_l137_13743

/-- Represents the distance traveled by a train in miles -/
def train_distance (time : ℕ) : ℕ :=
  (time / 2 : ℕ)

/-- Proves that a train traveling 1 mile every 2 minutes will cover 22 miles in 45 minutes -/
theorem train_distance_45_minutes : train_distance 45 = 22 := by
  sorry

end NUMINAMATH_CALUDE_train_distance_45_minutes_l137_13743


namespace NUMINAMATH_CALUDE_susan_age_proof_l137_13790

def james_age_in_15_years : ℕ := 37

def james_current_age : ℕ := james_age_in_15_years - 15

def james_age_8_years_ago : ℕ := james_current_age - 8

def janet_age_8_years_ago : ℕ := james_age_8_years_ago / 2

def janet_current_age : ℕ := janet_age_8_years_ago + 8

def susan_current_age : ℕ := janet_current_age - 3

def susan_age_in_5_years : ℕ := susan_current_age + 5

theorem susan_age_proof : susan_age_in_5_years = 17 := by
  sorry

end NUMINAMATH_CALUDE_susan_age_proof_l137_13790


namespace NUMINAMATH_CALUDE_power_calculation_l137_13777

theorem power_calculation : (16^4 * 8^6) / 4^12 = 1024 := by
  sorry

end NUMINAMATH_CALUDE_power_calculation_l137_13777


namespace NUMINAMATH_CALUDE_invalid_votes_percentage_l137_13759

theorem invalid_votes_percentage
  (total_votes : ℕ)
  (winner_percentage : ℚ)
  (loser_votes : ℕ)
  (h_total : total_votes = 7000)
  (h_winner : winner_percentage = 55 / 100)
  (h_loser : loser_votes = 2520) :
  (total_votes - (loser_votes / (1 - winner_percentage))) / total_votes = 1 / 5 := by
  sorry

end NUMINAMATH_CALUDE_invalid_votes_percentage_l137_13759


namespace NUMINAMATH_CALUDE_solution_set_of_inequality_l137_13718

def increasing_function (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f x < f y

theorem solution_set_of_inequality
  (f : ℝ → ℝ)
  (h_increasing : increasing_function f)
  (h_f_0 : f 0 = -2)
  (h_f_3 : f 3 = 2) :
  {x : ℝ | |f (x + 1)| ≥ 2} = {x : ℝ | x ≤ -1 ∨ x ≥ 2} :=
sorry

end NUMINAMATH_CALUDE_solution_set_of_inequality_l137_13718


namespace NUMINAMATH_CALUDE_geometric_progression_solution_l137_13738

theorem geometric_progression_solution :
  ∀ (a b c d : ℝ),
  (∃ (q : ℝ), b = a * q ∧ c = a * q^2 ∧ d = a * q^3) →
  a + d = -49 →
  b + c = 14 →
  ((a = 7 ∧ b = -14 ∧ c = 28 ∧ d = -56) ∨
   (a = -56 ∧ b = 28 ∧ c = -14 ∧ d = 7)) :=
by sorry

end NUMINAMATH_CALUDE_geometric_progression_solution_l137_13738


namespace NUMINAMATH_CALUDE_meeting_point_divides_segment_l137_13740

/-- The meeting point of two people moving towards each other on a line --/
def meeting_point (x₁ y₁ x₂ y₂ : ℚ) (m n : ℕ) : ℚ × ℚ :=
  ((m * x₂ + n * x₁) / (m + n), (m * y₂ + n * y₁) / (m + n))

/-- Theorem stating that the meeting point divides the line segment in the correct ratio --/
theorem meeting_point_divides_segment : 
  let mark_start : ℚ × ℚ := (2, 6)
  let sandy_start : ℚ × ℚ := (4, -2)
  let speed_ratio : ℕ × ℕ := (2, 1)
  let meet_point := meeting_point mark_start.1 mark_start.2 sandy_start.1 sandy_start.2 speed_ratio.1 speed_ratio.2
  meet_point = (8/3, 10/3) :=
by sorry

end NUMINAMATH_CALUDE_meeting_point_divides_segment_l137_13740


namespace NUMINAMATH_CALUDE_curve_in_fourth_quadrant_implies_a_range_l137_13708

-- Define the curve
def curve (x y a : ℝ) : Prop :=
  x^2 + y^2 + 2*a*x - 4*a*y + 5*a^2 - 4 = 0

-- Define the fourth quadrant
def fourth_quadrant (x y : ℝ) : Prop :=
  x > 0 ∧ y < 0

-- Theorem statement
theorem curve_in_fourth_quadrant_implies_a_range :
  (∀ x y : ℝ, curve x y a → fourth_quadrant x y) →
  a < -2 ∧ a ∈ Set.Iio (-2 : ℝ) :=
sorry

end NUMINAMATH_CALUDE_curve_in_fourth_quadrant_implies_a_range_l137_13708


namespace NUMINAMATH_CALUDE_circle_radius_with_modified_area_formula_l137_13706

/-- Given a circle with a modified area formula, prove that its radius is 10√2 units. -/
theorem circle_radius_with_modified_area_formula 
  (area : ℝ) 
  (k : ℝ) 
  (h1 : area = 100 * Real.pi)
  (h2 : k = 0.5)
  (h3 : ∀ r, Real.pi * k * r^2 = area) :
  ∃ r, r = 10 * Real.sqrt 2 ∧ Real.pi * k * r^2 = area := by
  sorry

end NUMINAMATH_CALUDE_circle_radius_with_modified_area_formula_l137_13706


namespace NUMINAMATH_CALUDE_high_school_twelve_games_l137_13776

/-- The number of teams in the "High School Twelve" soccer conference -/
def num_teams : ℕ := 12

/-- The number of times each team plays every other conference team -/
def games_per_pair : ℕ := 3

/-- The number of non-conference games each team plays -/
def non_conference_games : ℕ := 5

/-- The total number of games in a season involving the "High School Twelve" teams -/
def total_games : ℕ := (num_teams.choose 2 * games_per_pair) + (num_teams * non_conference_games)

theorem high_school_twelve_games :
  total_games = 258 :=
by sorry

end NUMINAMATH_CALUDE_high_school_twelve_games_l137_13776


namespace NUMINAMATH_CALUDE_library_books_l137_13735

def books_per_student : ℕ := 5

def students_day1 : ℕ := 4
def students_day2 : ℕ := 5
def students_day3 : ℕ := 6
def students_day4 : ℕ := 9

def total_books : ℕ := books_per_student * (students_day1 + students_day2 + students_day3 + students_day4)

theorem library_books : total_books = 120 := by
  sorry

end NUMINAMATH_CALUDE_library_books_l137_13735


namespace NUMINAMATH_CALUDE_joy_meets_grandma_l137_13765

/-- The number of hours in a day -/
def hours_per_day : ℕ := 24

/-- The number of days until Joy sees her grandma -/
def days_until_meeting : ℕ := 2

/-- The time zone difference between Joy and her grandma in hours -/
def time_zone_difference : ℤ := 3

/-- The total number of hours until Joy sees her grandma -/
def total_hours : ℕ := hours_per_day * days_until_meeting

theorem joy_meets_grandma : total_hours = 48 := by sorry

end NUMINAMATH_CALUDE_joy_meets_grandma_l137_13765


namespace NUMINAMATH_CALUDE_profit_margin_properties_l137_13761

/-- Profit margin calculation --/
theorem profit_margin_properties
  (B E : ℝ)  -- Purchase price and selling price
  (hE : E > B)  -- Condition: selling price is greater than purchase price
  (a : ℝ := 100 * (E - B) / B)  -- Profit margin from bottom up
  (f : ℝ := 100 * (E - B) / E)  -- Profit margin from top down
  : 
  (f = 100 * a / (a + 100) ∧ a = 100 * f / (100 - f)) ∧  -- Conversion formulas
  (a - f = a * f / 100)  -- Difference property
  := by sorry

end NUMINAMATH_CALUDE_profit_margin_properties_l137_13761


namespace NUMINAMATH_CALUDE_geometric_sequence_fourth_term_l137_13780

/-- Given a geometric sequence with first term 1000 and sixth term 125, 
    the fourth term is 125. -/
theorem geometric_sequence_fourth_term :
  ∀ (a : ℝ → ℝ),
  (∀ n : ℕ, a (n + 1) = a n * (a 1)⁻¹ * a 0) →  -- Geometric sequence definition
  a 0 = 1000 →                                 -- First term is 1000
  a 5 = 125 →                                  -- Sixth term is 125
  a 3 = 125 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_fourth_term_l137_13780


namespace NUMINAMATH_CALUDE_points_from_lines_l137_13767

/-- The number of lines formed by n points on a plane, where no three points are collinear -/
def num_lines (n : ℕ) : ℕ := n * (n - 1) / 2

/-- Theorem stating that if 45 lines are formed by n points on a plane where no three are collinear, then n = 10 -/
theorem points_from_lines (n : ℕ) (h : num_lines n = 45) : n = 10 := by
  sorry

end NUMINAMATH_CALUDE_points_from_lines_l137_13767


namespace NUMINAMATH_CALUDE_correct_animal_count_l137_13731

def petting_zoo (mary_count : ℕ) (double_counted : ℕ) (forgotten : ℕ) : ℕ :=
  mary_count - double_counted + forgotten

theorem correct_animal_count :
  petting_zoo 60 7 3 = 56 := by
  sorry

end NUMINAMATH_CALUDE_correct_animal_count_l137_13731


namespace NUMINAMATH_CALUDE_intersection_A_complement_B_l137_13799

def A : Set ℝ := {x | x^2 - x - 2 ≤ 0}
def B : Set ℝ := {x | x < 1}

theorem intersection_A_complement_B : A ∩ (Set.univ \ B) = {x : ℝ | 1 ≤ x ∧ x ≤ 2} := by
  sorry

end NUMINAMATH_CALUDE_intersection_A_complement_B_l137_13799


namespace NUMINAMATH_CALUDE_tilted_cube_segment_length_l137_13704

/-- Represents a tilted cube container with liquid -/
structure TiltedCube where
  edge_length : ℝ
  initial_fill_ratio : ℝ
  kb_length : ℝ
  lc_length : ℝ

/-- The length of segment LC in the tilted cube -/
def segment_lc_length (cube : TiltedCube) : ℝ := cube.lc_length

theorem tilted_cube_segment_length 
  (cube : TiltedCube)
  (h1 : cube.edge_length = 12)
  (h2 : cube.initial_fill_ratio = 5/8)
  (h3 : cube.lc_length = 2 * cube.kb_length)
  (h4 : cube.edge_length * (cube.initial_fill_ratio * cube.edge_length) = 
        (cube.lc_length + cube.kb_length) * cube.edge_length / 2) :
  segment_lc_length cube = 10 := by
sorry

end NUMINAMATH_CALUDE_tilted_cube_segment_length_l137_13704


namespace NUMINAMATH_CALUDE_jason_pokemon_cards_jason_initial_cards_l137_13793

theorem jason_pokemon_cards : ℕ → Prop :=
  fun initial_cards =>
    let given_away := 9
    let remaining := 4
    initial_cards = given_away + remaining

theorem jason_initial_cards : ∃ x : ℕ, jason_pokemon_cards x ∧ x = 13 :=
sorry

end NUMINAMATH_CALUDE_jason_pokemon_cards_jason_initial_cards_l137_13793


namespace NUMINAMATH_CALUDE_base4_addition_l137_13778

/-- Addition in base 4 --/
def base4_add (a b c d : ℕ) : ℕ := sorry

/-- Convert a natural number to its base 4 representation --/
def to_base4 (n : ℕ) : List ℕ := sorry

theorem base4_addition :
  to_base4 (base4_add 1 13 313 1313) = [2, 0, 2, 0, 0] := by sorry

end NUMINAMATH_CALUDE_base4_addition_l137_13778


namespace NUMINAMATH_CALUDE_line_equation_l137_13774

/-- Represents a line in 2D space -/
structure Line where
  slope : ℝ
  point : ℝ × ℝ

/-- Checks if a given equation represents the line -/
def is_equation_of_line (l : Line) (f : ℝ → ℝ → Prop) : Prop :=
  ∀ x y, f x y ↔ y - l.point.snd = l.slope * (x - l.point.fst)

theorem line_equation (l : Line) :
  l.slope = 3 ∧ l.point = (1, 3) →
  is_equation_of_line l (fun x y ↦ y - 3 = 3 * (x - 1)) :=
sorry

end NUMINAMATH_CALUDE_line_equation_l137_13774


namespace NUMINAMATH_CALUDE_expression_value_l137_13736

theorem expression_value :
  let a : ℤ := 3
  let b : ℤ := 2
  let c : ℤ := 1
  (a + (b + c)^2) - ((a + b)^2 - c) = -12 :=
by sorry

end NUMINAMATH_CALUDE_expression_value_l137_13736


namespace NUMINAMATH_CALUDE_james_out_of_pocket_l137_13713

/-- Calculates the total amount James is out of pocket after his Amazon purchases and returns. -/
def total_out_of_pocket (initial_purchase : ℝ) (returned_tv_cost : ℝ) (returned_bike_cost : ℝ) (toaster_cost : ℝ) : ℝ :=
  let returned_items_value := returned_tv_cost + returned_bike_cost
  let after_returns := initial_purchase - returned_items_value
  let sold_bike_cost := returned_bike_cost * 1.2
  let sold_bike_price := sold_bike_cost * 0.8
  let loss_from_bike_sale := sold_bike_cost - sold_bike_price
  after_returns + loss_from_bike_sale + toaster_cost

/-- Theorem stating that James is out of pocket $2020 given the problem conditions. -/
theorem james_out_of_pocket :
  total_out_of_pocket 3000 700 500 100 = 2020 := by
  sorry

end NUMINAMATH_CALUDE_james_out_of_pocket_l137_13713


namespace NUMINAMATH_CALUDE_angle_of_inclination_range_l137_13781

theorem angle_of_inclination_range (θ : ℝ) (α : ℝ) :
  (∃ x y : ℝ, Real.sqrt 3 * x + y * Real.cos θ - 1 = 0) →
  (α = Real.arctan (Real.sqrt 3 / (-Real.cos θ))) →
  π / 3 ≤ α ∧ α ≤ 2 * π / 3 := by
  sorry

end NUMINAMATH_CALUDE_angle_of_inclination_range_l137_13781


namespace NUMINAMATH_CALUDE_power_of_power_l137_13786

theorem power_of_power (a : ℝ) : (a^3)^2 = a^6 := by
  sorry

end NUMINAMATH_CALUDE_power_of_power_l137_13786


namespace NUMINAMATH_CALUDE_solution_implies_a_equals_one_l137_13772

def f (x a : ℝ) : ℝ := |x - a| - 2

theorem solution_implies_a_equals_one (a : ℝ) :
  (∀ x : ℝ, |f x a| < 1 ↔ (x ∈ Set.Ioo (-2) 0 ∨ x ∈ Set.Ioo 2 4)) →
  a = 1 := by
sorry

end NUMINAMATH_CALUDE_solution_implies_a_equals_one_l137_13772


namespace NUMINAMATH_CALUDE_ice_cream_cost_l137_13779

theorem ice_cream_cost (ice_cream_cartons yoghurt_cartons : ℕ) 
  (yoghurt_cost : ℚ) (cost_difference : ℚ) :
  ice_cream_cartons = 19 →
  yoghurt_cartons = 4 →
  yoghurt_cost = 1 →
  cost_difference = 129 →
  ∃ (ice_cream_cost : ℚ), 
    ice_cream_cost * ice_cream_cartons = yoghurt_cost * yoghurt_cartons + cost_difference ∧
    ice_cream_cost = 7 := by
  sorry

end NUMINAMATH_CALUDE_ice_cream_cost_l137_13779


namespace NUMINAMATH_CALUDE_tangent_length_specific_tangent_length_l137_13721

/-- Given a circle with radius r, a point M at distance d from the center,
    and a line through M tangent to the circle at A, 
    the length of AM is sqrt(d^2 - r^2) -/
theorem tangent_length (r d : ℝ) (hr : r > 0) (hd : d > r) :
  let am := Real.sqrt (d^2 - r^2)
  am^2 = d^2 - r^2 := by sorry

/-- In a circle with radius 10, if a point M is 26 units away from the center
    and a line passing through M touches the circle at point A,
    then the length of AM is 24 units -/
theorem specific_tangent_length :
  let r : ℝ := 10
  let d : ℝ := 26
  let am := Real.sqrt (d^2 - r^2)
  am = 24 := by sorry

end NUMINAMATH_CALUDE_tangent_length_specific_tangent_length_l137_13721


namespace NUMINAMATH_CALUDE_tom_fruit_purchase_amount_l137_13766

/-- Represents a fruit purchase with quantity and rate --/
structure FruitPurchase where
  quantity : ℝ
  rate : ℝ

/-- Calculates the total cost of purchases before discount --/
def totalCost (purchases : List FruitPurchase) : ℝ :=
  purchases.foldl (fun acc p => acc + p.quantity * p.rate) 0

/-- Calculates the final amount after discount and tax --/
def finalAmount (purchases : List FruitPurchase) (discountRate : ℝ) (taxRate : ℝ) : ℝ :=
  let total := totalCost purchases
  let discountedPrice := total * (1 - discountRate)
  discountedPrice * (1 + taxRate)

theorem tom_fruit_purchase_amount :
  let purchases := [
    ⟨8, 70⟩,  -- Apples
    ⟨9, 55⟩,  -- Mangoes
    ⟨5, 40⟩,  -- Oranges
    ⟨12, 30⟩, -- Bananas
    ⟨7, 45⟩,  -- Grapes
    ⟨4, 80⟩   -- Cherries
  ]
  finalAmount purchases 0.1 0.05 = 2126.25 := by
  sorry


end NUMINAMATH_CALUDE_tom_fruit_purchase_amount_l137_13766


namespace NUMINAMATH_CALUDE_random_events_count_l137_13709

-- Define the type for events
inductive Event
| DiceRoll : Event
| PearFall : Event
| LotteryWin : Event
| SecondChild : Event
| WaterBoil : Event

-- Define a function to check if an event is random
def isRandom (e : Event) : Bool :=
  match e with
  | Event.DiceRoll => true
  | Event.PearFall => false
  | Event.LotteryWin => true
  | Event.SecondChild => true
  | Event.WaterBoil => false

-- Define the list of events
def eventList : List Event := [
  Event.DiceRoll,
  Event.PearFall,
  Event.LotteryWin,
  Event.SecondChild,
  Event.WaterBoil
]

-- Theorem: The number of random events in the list is 3
theorem random_events_count : 
  (eventList.filter isRandom).length = 3 := by
  sorry

end NUMINAMATH_CALUDE_random_events_count_l137_13709


namespace NUMINAMATH_CALUDE_sum_of_units_digits_equals_zero_l137_13715

-- Define a function to get the units digit of a number
def unitsDigit (n : ℕ) : ℕ := n % 10

-- Define the problem
theorem sum_of_units_digits_equals_zero :
  (unitsDigit (17 * 34) + unitsDigit (19 * 28)) % 10 = 0 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_units_digits_equals_zero_l137_13715


namespace NUMINAMATH_CALUDE_rope_cost_minimum_l137_13760

/-- The cost of one foot of rope in dollars -/
def cost_per_foot : ℚ := 5 / 4

/-- The length of rope needed in feet -/
def rope_length_needed : ℚ := 5

/-- The minimum cost to buy the required length of rope -/
def min_cost : ℚ := rope_length_needed * cost_per_foot

theorem rope_cost_minimum :
  min_cost = 25 / 4 := by sorry

end NUMINAMATH_CALUDE_rope_cost_minimum_l137_13760


namespace NUMINAMATH_CALUDE_jason_total_spent_l137_13744

/-- The amount Jason spent on the flute -/
def flute_cost : ℚ := 142.46

/-- The amount Jason spent on the music tool -/
def music_tool_cost : ℚ := 8.89

/-- The amount Jason spent on the song book -/
def song_book_cost : ℚ := 7

/-- The total amount Jason spent at the music store -/
def total_spent : ℚ := flute_cost + music_tool_cost + song_book_cost

/-- Theorem stating that the total amount Jason spent is $158.35 -/
theorem jason_total_spent : total_spent = 158.35 := by sorry

end NUMINAMATH_CALUDE_jason_total_spent_l137_13744


namespace NUMINAMATH_CALUDE_arithmetic_sequence_fifth_term_l137_13775

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_fifth_term
  (a : ℕ → ℝ)
  (h_arithmetic : arithmetic_sequence a)
  (h_sum : a 3 + a 8 = 22)
  (h_sixth : a 6 = 7) :
  a 5 = 15 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_fifth_term_l137_13775


namespace NUMINAMATH_CALUDE_max_quarters_problem_l137_13762

theorem max_quarters_problem :
  ∃! q : ℕ, 8 < q ∧ q < 60 ∧
  q % 4 = 2 ∧
  q % 7 = 3 ∧
  q % 9 = 2 ∧
  q = 38 :=
by sorry

end NUMINAMATH_CALUDE_max_quarters_problem_l137_13762


namespace NUMINAMATH_CALUDE_pants_cost_l137_13726

def initial_amount : ℕ := 109
def shirt_cost : ℕ := 11
def num_shirts : ℕ := 2
def remaining_amount : ℕ := 74

theorem pants_cost : 
  initial_amount - (shirt_cost * num_shirts) - remaining_amount = 13 := by
  sorry

end NUMINAMATH_CALUDE_pants_cost_l137_13726


namespace NUMINAMATH_CALUDE_garden_length_is_32_l137_13701

/-- Calculates the length of a garden with mango trees -/
def garden_length (num_columns : ℕ) (tree_distance : ℝ) (boundary : ℝ) : ℝ :=
  (num_columns - 1 : ℝ) * tree_distance + 2 * boundary

/-- Theorem: The length of the garden is 32 meters -/
theorem garden_length_is_32 :
  garden_length 12 2 5 = 32 := by
  sorry

end NUMINAMATH_CALUDE_garden_length_is_32_l137_13701


namespace NUMINAMATH_CALUDE_percentage_difference_l137_13797

theorem percentage_difference (original : ℝ) (result : ℝ) (h : result < original) :
  (original - result) / original * 100 = 50 :=
by
  -- Assuming original = 60 and result = 30
  have h1 : original = 60 := by sorry
  have h2 : result = 30 := by sorry
  
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_percentage_difference_l137_13797


namespace NUMINAMATH_CALUDE_ceiling_sqrt_225_l137_13757

theorem ceiling_sqrt_225 : ⌈Real.sqrt 225⌉ = 15 := by
  sorry

end NUMINAMATH_CALUDE_ceiling_sqrt_225_l137_13757


namespace NUMINAMATH_CALUDE_orphanage_donation_percentage_l137_13794

def total_income : ℝ := 1000000
def children_percentage : ℝ := 0.2
def num_children : ℕ := 3
def wife_percentage : ℝ := 0.3
def final_amount : ℝ := 50000

theorem orphanage_donation_percentage :
  let family_distribution := children_percentage * num_children + wife_percentage
  let remaining_before_donation := total_income * (1 - family_distribution)
  let donation_amount := remaining_before_donation - final_amount
  (donation_amount / remaining_before_donation) * 100 = 50 := by sorry

end NUMINAMATH_CALUDE_orphanage_donation_percentage_l137_13794


namespace NUMINAMATH_CALUDE_symmetry_line_equation_l137_13788

-- Define the curves C1 and C2
def C1 (x y : ℝ) : Prop := y = -x^2 + 4*x - 2
def C2 (x y : ℝ) : Prop := y^2 = x

-- Define symmetry about a line
def symmetric_about_line (l : ℝ → ℝ → Prop) (C1 C2 : ℝ → ℝ → Prop) : Prop :=
  ∀ (x1 y1 x2 y2 : ℝ), C1 x1 y1 → C2 x2 y2 → 
    ∃ (x' y' : ℝ), l x' y' ∧ 
      x' = (x1 + x2) / 2 ∧ 
      y' = (y1 + y2) / 2

-- Theorem statement
theorem symmetry_line_equation :
  ∀ (l : ℝ → ℝ → Prop),
  symmetric_about_line l C1 C2 →
  (∀ x y, l x y ↔ x + y - 2 = 0) :=
sorry

end NUMINAMATH_CALUDE_symmetry_line_equation_l137_13788


namespace NUMINAMATH_CALUDE_cubic_odd_and_increasing_l137_13728

-- Define the function
def f (x : ℝ) : ℝ := x^3

-- State the theorem
theorem cubic_odd_and_increasing :
  (∀ x, f (-x) = -f x) ∧ 
  (∀ x y, 0 < x → x < y → f x < f y) :=
by sorry

end NUMINAMATH_CALUDE_cubic_odd_and_increasing_l137_13728


namespace NUMINAMATH_CALUDE_probability_difference_l137_13716

def red_marbles : ℕ := 501
def black_marbles : ℕ := 1501
def blue_marbles : ℕ := 1000

def total_marbles : ℕ := red_marbles + black_marbles + blue_marbles

def same_color_probability : ℚ :=
  (red_marbles * (red_marbles - 1) + black_marbles * (black_marbles - 1) + blue_marbles * (blue_marbles - 1)) /
  (2 * (total_marbles * (total_marbles - 1)))

def different_color_probability : ℚ :=
  (2 * (red_marbles * black_marbles + red_marbles * blue_marbles + black_marbles * blue_marbles)) /
  (total_marbles * (total_marbles - 1))

theorem probability_difference :
  |same_color_probability - different_color_probability| = 2 / 9 := by
  sorry

end NUMINAMATH_CALUDE_probability_difference_l137_13716


namespace NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l137_13712

/-- An isosceles triangle with two sides of length 5 and one side of length 2 has perimeter 12 -/
theorem isosceles_triangle_perimeter : ∀ (a b c : ℝ),
  a = 5 → b = 5 → c = 2 →
  (a = b ∨ a = c ∨ b = c) →  -- isosceles condition
  a + b + c = 12 := by
sorry

end NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l137_13712


namespace NUMINAMATH_CALUDE_toad_frog_percentage_increase_l137_13754

/-- Represents the number of bugs eaten by each animal -/
structure BugsEaten where
  gecko : ℕ
  lizard : ℕ
  frog : ℕ
  toad : ℕ

/-- Conditions from the problem -/
def garden_conditions (b : BugsEaten) : Prop :=
  b.gecko = 12 ∧
  b.lizard = b.gecko / 2 ∧
  b.frog = 3 * b.lizard ∧
  b.gecko + b.lizard + b.frog + b.toad = 63

/-- Calculate percentage increase -/
def percentage_increase (old_value new_value : ℕ) : ℚ :=
  (new_value - old_value : ℚ) / old_value * 100

/-- Theorem stating the percentage increase in bugs eaten by toad compared to frog -/
theorem toad_frog_percentage_increase (b : BugsEaten) 
  (h : garden_conditions b) : 
  percentage_increase b.frog b.toad = 50 := by
  sorry

end NUMINAMATH_CALUDE_toad_frog_percentage_increase_l137_13754


namespace NUMINAMATH_CALUDE_degrees_to_radians_conversion_l137_13755

theorem degrees_to_radians_conversion (deg : ℝ) (rad : ℝ) : 
  deg = 50 → rad = deg * (π / 180) → rad = 5 * π / 18 := by
  sorry

end NUMINAMATH_CALUDE_degrees_to_radians_conversion_l137_13755


namespace NUMINAMATH_CALUDE_inverse_square_problem_l137_13725

/-- Represents the inverse square relationship between x and y -/
def inverse_square_relation (k : ℝ) (x y : ℝ) : Prop :=
  x = k / (y * y)

theorem inverse_square_problem (x₁ x₂ y₁ y₂ : ℝ) 
  (h₁ : y₁ = 6)
  (h₂ : x₁ = 0.1111111111111111)
  (h₃ : y₂ = 2)
  (h₄ : ∃ k, inverse_square_relation k x₁ y₁ ∧ inverse_square_relation k x₂ y₂) :
  x₂ = 1 := by
  sorry

end NUMINAMATH_CALUDE_inverse_square_problem_l137_13725


namespace NUMINAMATH_CALUDE_sum_equality_l137_13722

theorem sum_equality : 9548 + 7314 = 3362 + 13500 := by
  sorry

end NUMINAMATH_CALUDE_sum_equality_l137_13722


namespace NUMINAMATH_CALUDE_satellite_units_count_l137_13729

/-- Represents a satellite composed of modular units with sensors. -/
structure Satellite where
  /-- The number of modular units in the satellite. -/
  units : ℕ
  /-- The number of non-upgraded sensors per unit. -/
  non_upgraded_per_unit : ℕ
  /-- The total number of upgraded sensors on the entire satellite. -/
  total_upgraded : ℕ
  /-- The total number of sensors on the satellite. -/
  total_sensors : ℕ
  /-- Each unit contains the same number of non-upgraded sensors. -/
  non_upgraded_uniform : units * non_upgraded_per_unit = total_sensors - total_upgraded
  /-- The number of non-upgraded sensors on one unit is 1/3 the total number of upgraded sensors. -/
  non_upgraded_ratio : 3 * non_upgraded_per_unit = total_upgraded
  /-- The fraction of upgraded sensors is 1/9 of the total sensors. -/
  upgraded_fraction : 9 * total_upgraded = total_sensors

theorem satellite_units_count (s : Satellite) : s.units = 24 := by
  sorry

end NUMINAMATH_CALUDE_satellite_units_count_l137_13729


namespace NUMINAMATH_CALUDE_election_percentage_l137_13723

theorem election_percentage (total_votes : ℕ) (winner_votes : ℕ) (margin : ℕ) 
  (h1 : winner_votes = 744)
  (h2 : margin = 288)
  (h3 : total_votes = winner_votes + (winner_votes - margin)) :
  (winner_votes : ℚ) / total_votes * 100 = 62 := by
  sorry

end NUMINAMATH_CALUDE_election_percentage_l137_13723


namespace NUMINAMATH_CALUDE_cubic_division_theorem_l137_13798

theorem cubic_division_theorem (c d : ℝ) (hc : c = 7) (hd : d = 3) :
  (c^3 + d^3) / (c^2 - c*d + d^2) = 10 := by
  sorry

end NUMINAMATH_CALUDE_cubic_division_theorem_l137_13798


namespace NUMINAMATH_CALUDE_arthur_walked_six_miles_l137_13719

/-- Calculates the total distance walked in miles given the number of blocks walked east and north, 
    and the length of each block in miles. -/
def total_distance (blocks_east : ℕ) (blocks_north : ℕ) (miles_per_block : ℚ) : ℚ :=
  (blocks_east + blocks_north : ℚ) * miles_per_block

/-- Theorem stating that Arthur walked 6 miles given the problem conditions. -/
theorem arthur_walked_six_miles :
  let blocks_east : ℕ := 6
  let blocks_north : ℕ := 12
  let miles_per_block : ℚ := 1/3
  total_distance blocks_east blocks_north miles_per_block = 6 := by
  sorry

end NUMINAMATH_CALUDE_arthur_walked_six_miles_l137_13719


namespace NUMINAMATH_CALUDE_task_assignment_count_l137_13785

def number_of_ways (n m : ℕ) : ℕ :=
  Nat.choose n m

theorem task_assignment_count : 
  (number_of_ways 10 4) * (number_of_ways 4 2) * (number_of_ways 2 1) = 2520 := by
  sorry

end NUMINAMATH_CALUDE_task_assignment_count_l137_13785


namespace NUMINAMATH_CALUDE_sum_of_cubes_zero_l137_13703

theorem sum_of_cubes_zero (a b : ℝ) (h1 : a + b = 0) (h2 : a * b = -1) : 
  a^3 + b^3 = 0 := by
sorry

end NUMINAMATH_CALUDE_sum_of_cubes_zero_l137_13703


namespace NUMINAMATH_CALUDE_radio_contest_winner_l137_13784

theorem radio_contest_winner (n : ℕ) 
  (h1 : 35 % 5 = 0)
  (h2 : 35 % n = 0)
  (h3 : ∀ m : ℕ, m > 0 ∧ m < 35 → ¬(m % 5 = 0 ∧ m % n = 0)) : 
  n = 7 := by
sorry

end NUMINAMATH_CALUDE_radio_contest_winner_l137_13784


namespace NUMINAMATH_CALUDE_compare_a_and_b_l137_13748

theorem compare_a_and_b (a b : ℝ) (h : 5 * (a - 1) = b + a^2) : a > b := by
  sorry

end NUMINAMATH_CALUDE_compare_a_and_b_l137_13748


namespace NUMINAMATH_CALUDE_triangle_implies_s_range_l137_13745

-- Define the system of inequalities
def SystemOfInequalities : Type := Unit  -- Placeholder, as we don't have specific inequalities

-- Define what it means for a region to be a triangle
def IsTriangle (region : SystemOfInequalities) : Prop := sorry

-- Define the range of s
def SRange (s : ℝ) : Prop := (0 < s ∧ s ≤ 2) ∨ s ≥ 4

-- Theorem statement
theorem triangle_implies_s_range (region : SystemOfInequalities) :
  IsTriangle region → ∀ s, SRange s :=
sorry

end NUMINAMATH_CALUDE_triangle_implies_s_range_l137_13745


namespace NUMINAMATH_CALUDE_hyperbola_asymptote_l137_13752

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := y^2 / 4 - x^2 = 1

-- Define the asymptote equation
def asymptote (x y : ℝ) : Prop := y = 2*x ∨ y = -2*x

-- Theorem statement
theorem hyperbola_asymptote :
  ∀ x y : ℝ, hyperbola x y → (∃ x' y' : ℝ, x' ≠ x ∧ y' ≠ y ∧ hyperbola x' y' ∧ asymptote x' y') :=
sorry

end NUMINAMATH_CALUDE_hyperbola_asymptote_l137_13752


namespace NUMINAMATH_CALUDE_quadratic_properties_l137_13770

/-- The quadratic function f(x) = x^2 - 4x - 5 -/
def f (x : ℝ) : ℝ := x^2 - 4*x - 5

theorem quadratic_properties :
  (∀ x, f x ≥ -9) ∧ 
  (f 5 = 0 ∧ f (-1) = 0) :=
sorry

end NUMINAMATH_CALUDE_quadratic_properties_l137_13770


namespace NUMINAMATH_CALUDE_sin_cos_sixth_power_sum_l137_13720

theorem sin_cos_sixth_power_sum (θ : ℝ) (h : Real.sin (2 * θ) = 1 / 5) :
  Real.sin θ ^ 6 + Real.cos θ ^ 6 = 97 / 100 := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_sixth_power_sum_l137_13720


namespace NUMINAMATH_CALUDE_fence_decoration_combinations_l137_13747

def num_colors : ℕ := 6
def num_techniques : ℕ := 5

theorem fence_decoration_combinations :
  num_colors * num_techniques = 30 := by
  sorry

end NUMINAMATH_CALUDE_fence_decoration_combinations_l137_13747


namespace NUMINAMATH_CALUDE_greatest_b_quadratic_inequality_l137_13739

theorem greatest_b_quadratic_inequality :
  ∃ b : ℝ, b^2 - 10*b + 24 ≤ 0 ∧ ∀ x : ℝ, x^2 - 10*x + 24 ≤ 0 → x ≤ b :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_greatest_b_quadratic_inequality_l137_13739


namespace NUMINAMATH_CALUDE_water_left_over_l137_13727

/-- Calculates the amount of water left over after distributing to players and accounting for spillage -/
theorem water_left_over
  (total_players : ℕ)
  (initial_water_liters : ℕ)
  (water_per_player_ml : ℕ)
  (spilled_water_ml : ℕ)
  (h1 : total_players = 30)
  (h2 : initial_water_liters = 8)
  (h3 : water_per_player_ml = 200)
  (h4 : spilled_water_ml = 250) :
  initial_water_liters * 1000 - (total_players * water_per_player_ml + spilled_water_ml) = 1750 :=
by sorry

end NUMINAMATH_CALUDE_water_left_over_l137_13727


namespace NUMINAMATH_CALUDE_largest_divisor_of_even_squares_sum_l137_13710

theorem largest_divisor_of_even_squares_sum (m n : ℕ) : 
  Even m → Even n → n < m → (∀ k : ℕ, k > 4 → ∃ m' n' : ℕ, 
    Even m' ∧ Even n' ∧ n' < m' ∧ ¬(k ∣ m'^2 + n'^2)) ∧ 
  (∀ m' n' : ℕ, Even m' → Even n' → n' < m' → (4 ∣ m'^2 + n'^2)) := by
  sorry

end NUMINAMATH_CALUDE_largest_divisor_of_even_squares_sum_l137_13710


namespace NUMINAMATH_CALUDE_aquarium_visit_cost_difference_l137_13749

/-- Represents the cost structure and family composition for an aquarium visit -/
structure AquariumVisit where
  family_pass_cost : ℚ
  adult_ticket_cost : ℚ
  child_ticket_cost : ℚ
  num_adults : ℕ
  num_children : ℕ

/-- Calculates the cost of separate tickets with the special offer applied -/
def separate_tickets_cost (visit : AquariumVisit) : ℚ :=
  let discounted_adults := visit.num_children / 3
  let full_price_adults := visit.num_adults - discounted_adults
  let discounted_adult_cost := visit.adult_ticket_cost * (1/2)
  discounted_adults * discounted_adult_cost +
  full_price_adults * visit.adult_ticket_cost +
  visit.num_children * visit.child_ticket_cost

/-- Theorem stating the difference between separate tickets and family pass -/
theorem aquarium_visit_cost_difference (visit : AquariumVisit) 
  (h1 : visit.family_pass_cost = 150)
  (h2 : visit.adult_ticket_cost = 35)
  (h3 : visit.child_ticket_cost = 20)
  (h4 : visit.num_adults = 2)
  (h5 : visit.num_children = 5) :
  separate_tickets_cost visit - visit.family_pass_cost = 5/2 := by
  sorry

end NUMINAMATH_CALUDE_aquarium_visit_cost_difference_l137_13749


namespace NUMINAMATH_CALUDE_sum_of_square_roots_inequality_l137_13796

theorem sum_of_square_roots_inequality (a b c d e : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) (he : e > 0) : 
  Real.sqrt (a / (b + c + d + e)) + 
  Real.sqrt (b / (a + c + d + e)) + 
  Real.sqrt (c / (a + b + d + e)) + 
  Real.sqrt (d / (a + b + c + e)) + 
  Real.sqrt (e / (a + b + c + d)) > 2 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_square_roots_inequality_l137_13796


namespace NUMINAMATH_CALUDE_square_area_ratio_l137_13750

theorem square_area_ratio (big_side : ℝ) (small_side : ℝ) 
  (h1 : big_side = 12)
  (h2 : small_side = 6) : 
  (small_side ^ 2) / (big_side ^ 2 - small_side ^ 2) = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_square_area_ratio_l137_13750


namespace NUMINAMATH_CALUDE_number_of_divisors_36_l137_13742

theorem number_of_divisors_36 : Finset.card (Nat.divisors 36) = 9 := by
  sorry

end NUMINAMATH_CALUDE_number_of_divisors_36_l137_13742


namespace NUMINAMATH_CALUDE_clock_gains_seven_minutes_per_hour_l137_13792

/-- A clock that gains time -/
structure GainingClock where
  start_time : Nat  -- Start time in hours (24-hour format)
  end_time : Nat    -- End time in hours (24-hour format)
  total_gain : Nat  -- Total minutes gained

/-- Calculate the minutes gained per hour -/
def minutes_gained_per_hour (clock : GainingClock) : Rat :=
  clock.total_gain / (clock.end_time - clock.start_time)

/-- Theorem: A clock starting at 9 AM, ending at 6 PM, and gaining 63 minutes
    will gain 7 minutes per hour -/
theorem clock_gains_seven_minutes_per_hour 
  (clock : GainingClock) 
  (h1 : clock.start_time = 9)
  (h2 : clock.end_time = 18)
  (h3 : clock.total_gain = 63) :
  minutes_gained_per_hour clock = 7 := by
  sorry

end NUMINAMATH_CALUDE_clock_gains_seven_minutes_per_hour_l137_13792


namespace NUMINAMATH_CALUDE_complex_modulus_range_l137_13733

theorem complex_modulus_range (z : ℂ) (h : Complex.abs z = 1) :
  ∃ (x : ℝ), x = Complex.abs ((z - 2) * (z + 1)^2) ∧ 0 ≤ x ∧ x ≤ 3 * Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_complex_modulus_range_l137_13733


namespace NUMINAMATH_CALUDE_max_value_and_x_l137_13746

theorem max_value_and_x (x : ℝ) (y : ℝ) (h : x < 0) (h1 : y = 3*x + 4/x) :
  (∀ z, z < 0 → 3*z + 4/z ≤ y) → y = -4*Real.sqrt 3 ∧ x = -2*Real.sqrt 3/3 :=
sorry

end NUMINAMATH_CALUDE_max_value_and_x_l137_13746


namespace NUMINAMATH_CALUDE_equation_solutions_l137_13768

theorem equation_solutions :
  (∃ x : ℚ, 3 + 2 * x = 6 ∧ x = 3 / 2) ∧
  (∃ x : ℚ, 3 - 1 / 2 * x = 3 * x + 1 ∧ x = 4 / 7) := by
  sorry

end NUMINAMATH_CALUDE_equation_solutions_l137_13768


namespace NUMINAMATH_CALUDE_triangle_is_isosceles_l137_13741

theorem triangle_is_isosceles (a b c : ℝ) (A B C : ℝ) :
  a > 0 ∧ b > 0 ∧ c > 0 →
  A > 0 ∧ B > 0 ∧ C > 0 →
  A + B + C = π →
  a = 2 * b * Real.cos C →
  b = c :=
by sorry

end NUMINAMATH_CALUDE_triangle_is_isosceles_l137_13741


namespace NUMINAMATH_CALUDE_problem_statement_l137_13753

theorem problem_statement (x y : ℝ) (h : x^2 + y^2 - x*y = 1) : 
  (x + y ≥ -2) ∧ (x^2 + y^2 ≤ 2) := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l137_13753


namespace NUMINAMATH_CALUDE_water_flow_speed_l137_13764

/-- The speed of the water flow given the ship's travel times and distances -/
theorem water_flow_speed (x y : ℝ) : 
  (135 / (x + y) + 70 / (x - y) = 12.5) →
  (75 / (x + y) + 110 / (x - y) = 12.5) →
  y = 3.2 := by
sorry

end NUMINAMATH_CALUDE_water_flow_speed_l137_13764
