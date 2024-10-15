import Mathlib

namespace NUMINAMATH_CALUDE_anthony_balloons_l2866_286678

theorem anthony_balloons (tom_balloons luke_balloons anthony_balloons : ℕ) :
  tom_balloons = 3 * luke_balloons →
  luke_balloons = anthony_balloons / 4 →
  tom_balloons = 33 →
  anthony_balloons = 44 :=
by sorry

end NUMINAMATH_CALUDE_anthony_balloons_l2866_286678


namespace NUMINAMATH_CALUDE_cos_315_degrees_l2866_286642

theorem cos_315_degrees : Real.cos (315 * π / 180) = Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_315_degrees_l2866_286642


namespace NUMINAMATH_CALUDE_feed_lasts_longer_when_selling_feed_lasts_shorter_when_buying_nils_has_300_geese_l2866_286645

/-- Represents the number of geese Nils currently has -/
def current_geese : ℕ := sorry

/-- Represents the number of days the feed lasts with the current number of geese -/
def current_feed_duration : ℕ := sorry

/-- Represents the amount of feed one goose consumes per day -/
def feed_per_goose_per_day : ℚ := sorry

/-- Represents the total amount of feed available -/
def total_feed : ℚ := sorry

/-- The feed lasts 20 days longer when 75 geese are sold -/
theorem feed_lasts_longer_when_selling : 
  total_feed / (feed_per_goose_per_day * (current_geese - 75)) = current_feed_duration + 20 := by sorry

/-- The feed lasts 15 days shorter when 100 geese are bought -/
theorem feed_lasts_shorter_when_buying : 
  total_feed / (feed_per_goose_per_day * (current_geese + 100)) = current_feed_duration - 15 := by sorry

/-- The main theorem proving that Nils has 300 geese -/
theorem nils_has_300_geese : current_geese = 300 := by sorry

end NUMINAMATH_CALUDE_feed_lasts_longer_when_selling_feed_lasts_shorter_when_buying_nils_has_300_geese_l2866_286645


namespace NUMINAMATH_CALUDE_matrix_rank_theorem_l2866_286653

theorem matrix_rank_theorem (m n : ℕ) (A : Matrix (Fin m) (Fin n) ℚ) 
  (h : ∃ (S : Finset ℕ), S.card ≥ m + n ∧ 
    (∀ p ∈ S, Nat.Prime p ∧ ∃ (i : Fin m) (j : Fin n), |A i j| = p)) : 
  Matrix.rank A ≥ 2 := by
sorry

end NUMINAMATH_CALUDE_matrix_rank_theorem_l2866_286653


namespace NUMINAMATH_CALUDE_min_tiles_for_room_l2866_286696

/-- Represents the dimensions of a room in centimeters -/
structure Room where
  length : ℕ
  breadth : ℕ

/-- Represents a square tile with a given side length in centimeters -/
structure Tile where
  side : ℕ

/-- Calculates the number of tiles needed to cover a room, including wastage -/
def tilesNeeded (room : Room) (tile : Tile) : ℕ :=
  let roomArea := room.length * room.breadth
  let tileArea := tile.side * tile.side
  let baseTiles := (roomArea + tileArea - 1) / tileArea  -- Ceiling division
  let wastage := (baseTiles * 11 + 9) / 10  -- 10% wastage, rounded up
  baseTiles + wastage

/-- Theorem stating the minimum number of tiles required -/
theorem min_tiles_for_room (room : Room) (tile : Tile) :
  room.length = 888 ∧ room.breadth = 462 ∧ tile.side = 22 →
  tilesNeeded room tile ≥ 933 :=
by sorry

end NUMINAMATH_CALUDE_min_tiles_for_room_l2866_286696


namespace NUMINAMATH_CALUDE_no_money_left_l2866_286650

theorem no_money_left (total_money : ℝ) (total_items : ℝ) (h1 : total_money > 0) (h2 : total_items > 0) :
  (1 / 3 : ℝ) * total_money = (1 / 3 : ℝ) * total_items * (total_money / total_items) →
  total_money - total_items * (total_money / total_items) = 0 := by
sorry

end NUMINAMATH_CALUDE_no_money_left_l2866_286650


namespace NUMINAMATH_CALUDE_jack_marbles_l2866_286655

/-- Calculates the final number of marbles Jack has after sharing and finding more -/
def final_marbles (initial : ℕ) (shared : ℕ) (multiplier : ℕ) : ℕ :=
  let remaining := initial - shared
  let found := remaining * multiplier
  remaining + found

/-- Theorem stating that Jack ends up with 232 marbles -/
theorem jack_marbles :
  final_marbles 62 33 7 = 232 := by
  sorry

end NUMINAMATH_CALUDE_jack_marbles_l2866_286655


namespace NUMINAMATH_CALUDE_inequality_proof_l2866_286638

theorem inequality_proof (a b : ℝ) (ha : 0 < a) (hb : 0 < b) :
  (a^2 + b^2) / 2 ≥ ((a + b) / 2)^2 ∧ ((a + b) / 2)^2 ≥ a * b :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_l2866_286638


namespace NUMINAMATH_CALUDE_midpoint_trajectory_l2866_286644

/-- Given a curve defined by 2x^2 - y = 0, prove that the midpoint of the line segment
    connecting (0, -1) and any point on the curve satisfies y = 4x^2 - 1/2 -/
theorem midpoint_trajectory (x₁ y₁ x y : ℝ) :
  (2 * x₁^2 = y₁) →  -- P(x₁, y₁) is on the curve
  (x = x₁ / 2) →     -- x-coordinate of midpoint
  (y = (y₁ - 1) / 2) -- y-coordinate of midpoint
  → y = 4 * x^2 - 1/2 := by
sorry

end NUMINAMATH_CALUDE_midpoint_trajectory_l2866_286644


namespace NUMINAMATH_CALUDE_inequality_proof_l2866_286672

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (h_abc : a * b * c = 1) :
  (a^2014 / (1 + 2*b*c)) + (b^2014 / (1 + 2*a*c)) + (c^2014 / (1 + 2*a*b)) ≥ 3 / (a*b + b*c + c*a) :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_l2866_286672


namespace NUMINAMATH_CALUDE_range_of_f_l2866_286675

noncomputable def f (x : ℝ) : ℝ := 1 / (x^2 + 1)

theorem range_of_f :
  Set.range f = Set.Ioo 0 1 := by sorry

end NUMINAMATH_CALUDE_range_of_f_l2866_286675


namespace NUMINAMATH_CALUDE_correlation_identification_l2866_286693

-- Define the relationships
inductive Relationship
| AgeAndFat
| CurvePoints
| FruitProduction
| StudentAndID

-- Define the property of having a correlation
def HasCorrelation : Relationship → Prop :=
  fun r => match r with
  | Relationship.AgeAndFat => true
  | Relationship.CurvePoints => false
  | Relationship.FruitProduction => true
  | Relationship.StudentAndID => false

-- Define the property of being a functional relationship
def IsFunctionalRelationship : Relationship → Prop :=
  fun r => match r with
  | Relationship.AgeAndFat => false
  | Relationship.CurvePoints => true
  | Relationship.FruitProduction => false
  | Relationship.StudentAndID => true

-- Theorem statement
theorem correlation_identification :
  (∀ r : Relationship, HasCorrelation r ↔ ¬(IsFunctionalRelationship r)) ∧
  (HasCorrelation Relationship.AgeAndFat ∧ HasCorrelation Relationship.FruitProduction) ∧
  (¬HasCorrelation Relationship.CurvePoints ∧ ¬HasCorrelation Relationship.StudentAndID) :=
by sorry

end NUMINAMATH_CALUDE_correlation_identification_l2866_286693


namespace NUMINAMATH_CALUDE_total_chickens_and_ducks_prove_total_chickens_and_ducks_l2866_286608

theorem total_chickens_and_ducks : ℕ → ℕ → ℕ → Prop :=
  fun (chickens ducks total : ℕ) =>
    chickens = 45 ∧ 
    chickens = ducks + 8 ∧ 
    total = chickens + ducks → 
    total = 82

-- Proof
theorem prove_total_chickens_and_ducks : 
  ∃ (chickens ducks total : ℕ), total_chickens_and_ducks chickens ducks total :=
by
  sorry

end NUMINAMATH_CALUDE_total_chickens_and_ducks_prove_total_chickens_and_ducks_l2866_286608


namespace NUMINAMATH_CALUDE_simplify_and_evaluate_l2866_286691

theorem simplify_and_evaluate (a : ℝ) (h : a = 1 - Real.sqrt 2) :
  a * (a - 9) - (a + 3) * (a - 3) = 9 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_simplify_and_evaluate_l2866_286691


namespace NUMINAMATH_CALUDE_range_of_a_l2866_286694

-- Define the sets A and B
def A : Set ℝ := {x | x^2 - 4*x + 3 < 0}
def B (a : ℝ) : Set ℝ := {x | 2^(1-x) + a ≤ 0 ∧ x^2 - 2*(a + 7)*x + 5 ≤ 0}

-- State the theorem
theorem range_of_a (a : ℝ) : A ⊆ B a → -4 ≤ a ∧ a ≤ -1 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l2866_286694


namespace NUMINAMATH_CALUDE_sheets_per_ream_l2866_286684

theorem sheets_per_ream (cost_per_ream : ℕ) (sheets_needed : ℕ) (total_cost : ℕ) :
  cost_per_ream = 27 →
  sheets_needed = 5000 →
  total_cost = 270 →
  (sheets_needed / (total_cost / cost_per_ream) : ℕ) = 500 := by
  sorry

end NUMINAMATH_CALUDE_sheets_per_ream_l2866_286684


namespace NUMINAMATH_CALUDE_pigeonhole_principle_for_library_l2866_286629

/-- The number of different types of books available. -/
def num_book_types : ℕ := 4

/-- The maximum number of books a student can borrow. -/
def max_books_per_student : ℕ := 3

/-- The type representing a borrowing pattern (number and types of books borrowed). -/
def BorrowingPattern := Fin num_book_types → Fin (max_books_per_student + 1)

/-- The minimum number of students required to guarantee a repeated borrowing pattern. -/
def min_students_for_repeat : ℕ := 15

theorem pigeonhole_principle_for_library :
  ∀ (students : Fin min_students_for_repeat → BorrowingPattern),
  ∃ (i j : Fin min_students_for_repeat), i ≠ j ∧ students i = students j :=
sorry

end NUMINAMATH_CALUDE_pigeonhole_principle_for_library_l2866_286629


namespace NUMINAMATH_CALUDE_function_min_max_values_l2866_286647

/-- The function f(x) = x^3 - 3x + m has a minimum value of -1 and a maximum value of 3 -/
theorem function_min_max_values (m : ℝ) : 
  (∃ x₀ : ℝ, ∀ x : ℝ, x^3 - 3*x + m ≥ x₀^3 - 3*x₀ + m ∧ x₀^3 - 3*x₀ + m = -1) →
  (∃ x₁ : ℝ, ∀ x : ℝ, x^3 - 3*x + m ≤ x₁^3 - 3*x₁ + m ∧ x₁^3 - 3*x₁ + m = 3) :=
by sorry

end NUMINAMATH_CALUDE_function_min_max_values_l2866_286647


namespace NUMINAMATH_CALUDE_smallest_perfect_square_with_remainders_l2866_286606

theorem smallest_perfect_square_with_remainders : ∃ n : ℕ, 
  n > 1 ∧
  n % 3 = 2 ∧
  n % 7 = 2 ∧
  n % 8 = 2 ∧
  ∃ k : ℕ, n = k^2 ∧
  ∀ m : ℕ, m > 1 → m % 3 = 2 → m % 7 = 2 → m % 8 = 2 → (∃ j : ℕ, m = j^2) → m ≥ n :=
by sorry

end NUMINAMATH_CALUDE_smallest_perfect_square_with_remainders_l2866_286606


namespace NUMINAMATH_CALUDE_unique_zero_point_l2866_286616

open Real

noncomputable def f (x : ℝ) := exp x + x - 2 * exp 1

theorem unique_zero_point :
  ∃! x, f x = 0 :=
sorry

end NUMINAMATH_CALUDE_unique_zero_point_l2866_286616


namespace NUMINAMATH_CALUDE_units_digit_of_n_l2866_286624

/-- Given two natural numbers m and n, returns true if m has a units digit of 3 -/
def has_units_digit_3 (m : ℕ) : Prop :=
  m % 10 = 3

/-- Given a natural number n, returns its units digit -/
def units_digit (n : ℕ) : ℕ :=
  n % 10

theorem units_digit_of_n (m n : ℕ) (h1 : m * n = 31^5) (h2 : has_units_digit_3 m) :
  units_digit n = 7 := by
sorry

end NUMINAMATH_CALUDE_units_digit_of_n_l2866_286624


namespace NUMINAMATH_CALUDE_points_earned_l2866_286688

def points_per_enemy : ℕ := 3
def total_enemies : ℕ := 6
def enemies_not_defeated : ℕ := 2

theorem points_earned : 
  (total_enemies - enemies_not_defeated) * points_per_enemy = 12 := by
  sorry

end NUMINAMATH_CALUDE_points_earned_l2866_286688


namespace NUMINAMATH_CALUDE_catch_up_time_is_55_minutes_l2866_286649

/-- The time it takes for Bob to catch up with John -/
def catch_up_time (john_speed bob_speed initial_distance stop_time : ℚ) : ℚ :=
  let relative_speed := bob_speed - john_speed
  let time_without_stop := initial_distance / relative_speed
  (time_without_stop + stop_time / 60) * 60

theorem catch_up_time_is_55_minutes :
  catch_up_time 2 6 3 10 = 55 := by sorry

end NUMINAMATH_CALUDE_catch_up_time_is_55_minutes_l2866_286649


namespace NUMINAMATH_CALUDE_calculate_expression_l2866_286635

theorem calculate_expression : (8^5 / 8^2) * 3^6 = 373248 := by
  sorry

end NUMINAMATH_CALUDE_calculate_expression_l2866_286635


namespace NUMINAMATH_CALUDE_log_stacks_total_l2866_286646

def first_stack_start : ℕ := 15
def first_stack_end : ℕ := 4
def second_stack_start : ℕ := 5
def second_stack_end : ℕ := 10

def total_logs : ℕ := 159

theorem log_stacks_total :
  (first_stack_start - first_stack_end + 1) * (first_stack_start + first_stack_end) / 2 +
  (second_stack_end - second_stack_start + 1) * (second_stack_start + second_stack_end) / 2 =
  total_logs := by
  sorry

end NUMINAMATH_CALUDE_log_stacks_total_l2866_286646


namespace NUMINAMATH_CALUDE_modulus_of_complex_l2866_286614

theorem modulus_of_complex (z : ℂ) : (Complex.I * z = 3 + 4 * Complex.I) → Complex.abs z = 5 := by
  sorry

end NUMINAMATH_CALUDE_modulus_of_complex_l2866_286614


namespace NUMINAMATH_CALUDE_three_primes_sum_l2866_286615

theorem three_primes_sum (p q r : ℕ) : 
  Nat.Prime p → Nat.Prime q → Nat.Prime r →
  p * q * r = 31 * (p + q + r) →
  p + q + r = 51 := by
sorry

end NUMINAMATH_CALUDE_three_primes_sum_l2866_286615


namespace NUMINAMATH_CALUDE_chris_age_l2866_286689

theorem chris_age (c m : ℕ) : c = 3 * m - 22 → c + m = 70 → c = 47 := by
  sorry

end NUMINAMATH_CALUDE_chris_age_l2866_286689


namespace NUMINAMATH_CALUDE_third_degree_equation_roots_l2866_286604

theorem third_degree_equation_roots (x : ℝ) : 
  let f : ℝ → ℝ := λ x => 8 * x^3 - 4 * x^2 - 4 * x - 1
  let root1 := Real.sin (π / 14)
  let root2 := Real.sin (5 * π / 14)
  let root3 := Real.sin (-3 * π / 14)
  (f root1 = 0) ∧ (f root2 = 0) ∧ (f root3 = 0) := by
  sorry

end NUMINAMATH_CALUDE_third_degree_equation_roots_l2866_286604


namespace NUMINAMATH_CALUDE_student_journal_pages_l2866_286621

/-- Calculates the total number of journal pages written by a student over a given number of weeks. -/
def total_pages (sessions_per_week : ℕ) (pages_per_session : ℕ) (weeks : ℕ) : ℕ :=
  sessions_per_week * pages_per_session * weeks

/-- Theorem stating that given the specific conditions, a student writes 72 pages in 6 weeks. -/
theorem student_journal_pages :
  total_pages 3 4 6 = 72 := by
  sorry

end NUMINAMATH_CALUDE_student_journal_pages_l2866_286621


namespace NUMINAMATH_CALUDE_person_savings_l2866_286627

/-- Calculates a person's savings given their income and income-to-expenditure ratio --/
theorem person_savings (income : ℚ) (ratio_income ratio_expenditure : ℕ) 
  (h1 : income = 18000)
  (h2 : ratio_income = 9)
  (h3 : ratio_expenditure = 8) : 
  income - (income * ratio_expenditure / ratio_income) = 2000 := by
  sorry

end NUMINAMATH_CALUDE_person_savings_l2866_286627


namespace NUMINAMATH_CALUDE_exactly_two_absent_probability_l2866_286662

-- Define the probability of a student being absent
def prob_absent : ℚ := 1 / 20

-- Define the probability of a student being present
def prob_present : ℚ := 1 - prob_absent

-- Define the number of students we're considering
def num_students : ℕ := 3

-- Define the number of absent students we're looking for
def num_absent : ℕ := 2

-- Theorem statement
theorem exactly_two_absent_probability :
  (prob_absent ^ num_absent * prob_present ^ (num_students - num_absent)) * (num_students.choose num_absent) = 7125 / 1000000 := by
  sorry

end NUMINAMATH_CALUDE_exactly_two_absent_probability_l2866_286662


namespace NUMINAMATH_CALUDE_complex_number_problem_l2866_286673

def complex_i : ℂ := Complex.I

theorem complex_number_problem (z₁ z₂ : ℂ) 
  (h1 : (z₁ - 2) * (1 + complex_i) = 1 - complex_i)
  (h2 : z₂.im = 2)
  (h3 : (z₁ * z₂).im = 0) :
  z₁ = 2 - complex_i ∧ z₂ = 4 + 2 * complex_i := by
  sorry

end NUMINAMATH_CALUDE_complex_number_problem_l2866_286673


namespace NUMINAMATH_CALUDE_distance_is_1760_l2866_286676

/-- The distance between Péter's and Károly's houses in meters. -/
def distance_between_houses : ℝ := 1760

/-- The distance from Péter's house to the first meeting point in meters. -/
def first_meeting_distance : ℝ := 720

/-- The distance from Károly's house to the second meeting point in meters. -/
def second_meeting_distance : ℝ := 400

/-- Theorem stating that the distance between the houses is 1760 meters. -/
theorem distance_is_1760 :
  let x := distance_between_houses
  let d1 := first_meeting_distance
  let d2 := second_meeting_distance
  (d1 / (x - d1) = (x - d2) / (x + d2)) →
  x = 1760 := by
  sorry


end NUMINAMATH_CALUDE_distance_is_1760_l2866_286676


namespace NUMINAMATH_CALUDE_rectangle_area_theorem_l2866_286639

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a rectangle -/
structure Rectangle where
  A : Point
  B : Point
  C : Point
  D : Point

def Rectangle.area (r : Rectangle) : ℝ := sorry

def angle (p1 p2 p3 : Point) : ℝ := sorry

def distance (p1 p2 : Point) : ℝ := sorry

def foldPoint (p : Point) (line : Point × Point) : Point := sorry

theorem rectangle_area_theorem (ABCD : Rectangle) (E F : Point) (B' C' : Point) :
  E.x = ABCD.A.x ∧ F.x = ABCD.D.x →
  distance ABCD.B E < distance ABCD.C F →
  B' = foldPoint ABCD.B (E, F) →
  C' = foldPoint ABCD.C (E, F) →
  C'.x = ABCD.A.x →
  angle ABCD.A B' C' = 2 * angle B' E ABCD.A →
  distance ABCD.A B' = 8 →
  distance ABCD.B E = 15 →
  ∃ (a b c : ℕ), 
    Rectangle.area ABCD = a + b * Real.sqrt c ∧
    a = 100 ∧ b = 4 ∧ c = 23 ∧
    a + b + c = 127 ∧
    ∀ (p : ℕ), Prime p → c % (p * p) ≠ 0 :=
by sorry

end NUMINAMATH_CALUDE_rectangle_area_theorem_l2866_286639


namespace NUMINAMATH_CALUDE_number_puzzle_l2866_286620

theorem number_puzzle : ∃ x : ℝ, (2 * x) / 16 = 25 ∧ x = 200 := by sorry

end NUMINAMATH_CALUDE_number_puzzle_l2866_286620


namespace NUMINAMATH_CALUDE_system_solution_l2866_286683

theorem system_solution : 
  ∀ x y : ℚ, 
  x^2 - 9*y^2 = 0 ∧ x + y = 1 → 
  (x = 3/4 ∧ y = 1/4) ∨ (x = 3/2 ∧ y = -1/2) := by
sorry

end NUMINAMATH_CALUDE_system_solution_l2866_286683


namespace NUMINAMATH_CALUDE_total_movies_in_five_years_l2866_286685

-- Define the number of movies L&J Productions makes per year
def lj_movies_per_year : ℕ := 220

-- Define the percentage increase for Johnny TV
def johnny_tv_increase_percent : ℕ := 25

-- Define the number of years
def years : ℕ := 5

-- Statement to prove
theorem total_movies_in_five_years :
  (lj_movies_per_year + (lj_movies_per_year * johnny_tv_increase_percent) / 100 + lj_movies_per_year) * years = 2475 := by
  sorry

end NUMINAMATH_CALUDE_total_movies_in_five_years_l2866_286685


namespace NUMINAMATH_CALUDE_rational_function_sum_l2866_286618

-- Define p(x) and q(x) as functions
variable (p q : ℝ → ℝ)

-- State the conditions
variable (h1 : ∃ a b c : ℝ, ∀ x, q x = a * x^2 + b * x + c)
variable (h2 : q 2 = 0 ∧ q 4 = 0)
variable (h3 : p 1 = 2)
variable (h4 : q 3 = 3)

-- State the theorem
theorem rational_function_sum :
  ∃ f : ℝ → ℝ, (∀ x, f x = p x + q x) ∧ (∀ x, f x = -3 * x^2 + 18 * x - 22) :=
sorry

end NUMINAMATH_CALUDE_rational_function_sum_l2866_286618


namespace NUMINAMATH_CALUDE_roy_pens_total_l2866_286640

theorem roy_pens_total (blue : ℕ) (black : ℕ) (red : ℕ) : 
  blue = 2 → 
  black = 2 * blue → 
  red = 2 * black - 2 → 
  blue + black + red = 12 := by
  sorry

end NUMINAMATH_CALUDE_roy_pens_total_l2866_286640


namespace NUMINAMATH_CALUDE_margaret_score_is_86_l2866_286660

/-- Given an average test score, calculate Margaret's score based on the conditions -/
def margaret_score (average : ℝ) : ℝ :=
  let marco_score := average * 0.9
  marco_score + 5

/-- Theorem stating that Margaret's score is 86 given the conditions -/
theorem margaret_score_is_86 :
  margaret_score 90 = 86 := by
  sorry

end NUMINAMATH_CALUDE_margaret_score_is_86_l2866_286660


namespace NUMINAMATH_CALUDE_remainder_theorem_l2866_286659

theorem remainder_theorem : (9^6 + 5^7 + 3^8) % 7 = 4 := by sorry

end NUMINAMATH_CALUDE_remainder_theorem_l2866_286659


namespace NUMINAMATH_CALUDE_parabola_equation_correct_l2866_286603

-- Define the parabola
structure Parabola where
  focus : ℝ × ℝ
  directrix : ℝ → ℝ → ℝ

-- Define the equation of the parabola
def parabola_equation (p : Parabola) (x y : ℝ) : ℝ :=
  16 * x^2 + 25 * y^2 + 36 * x + 242 * y - 195

-- Theorem statement
theorem parabola_equation_correct (p : Parabola) :
  p.focus = (2, -1) ∧ p.directrix = (fun x y ↦ 5*x + 4*y - 20) →
  ∀ x y : ℝ, (x - 2)^2 + (y + 1)^2 = ((5*x + 4*y - 20)^2) / 41 ↔
  parabola_equation p x y = 0 :=
sorry

end NUMINAMATH_CALUDE_parabola_equation_correct_l2866_286603


namespace NUMINAMATH_CALUDE_exists_universal_friend_l2866_286698

-- Define a type for people
variable {Person : Type}

-- Define the friendship relation
variable (friends : Person → Person → Prop)

-- Define the property that every two people have exactly one friend in common
def one_common_friend (friends : Person → Person → Prop) : Prop :=
  ∀ a b : Person, a ≠ b →
    ∃! c : Person, friends a c ∧ friends b c

-- State the theorem
theorem exists_universal_friend
  [Finite Person]
  (h : one_common_friend friends) :
  ∃ x : Person, ∀ y : Person, y ≠ x → friends x y :=
sorry

end NUMINAMATH_CALUDE_exists_universal_friend_l2866_286698


namespace NUMINAMATH_CALUDE_corrected_mean_l2866_286681

theorem corrected_mean (n : ℕ) (incorrect_mean : ℚ) (incorrect_value : ℚ) (correct_value : ℚ) :
  n = 50 ∧ incorrect_mean = 36 ∧ incorrect_value = 23 ∧ correct_value = 45 →
  (n : ℚ) * incorrect_mean - incorrect_value + correct_value = 36.44 * n :=
by sorry

end NUMINAMATH_CALUDE_corrected_mean_l2866_286681


namespace NUMINAMATH_CALUDE_percentage_equation_l2866_286657

theorem percentage_equation (x : ℝ) : (65 / 100 * x = 20 / 100 * 747.50) → x = 230 := by
  sorry

end NUMINAMATH_CALUDE_percentage_equation_l2866_286657


namespace NUMINAMATH_CALUDE_max_value_a_l2866_286619

theorem max_value_a (a b c d : ℕ+) 
  (h1 : a < 2 * b)
  (h2 : b < 3 * c)
  (h3 : c < 4 * d)
  (h4 : d < 100) :
  a ≤ 2367 ∧ ∃ (a' b' c' d' : ℕ+), a' = 2367 ∧ 
    a' < 2 * b' ∧ b' < 3 * c' ∧ c' < 4 * d' ∧ d' < 100 :=
by sorry

end NUMINAMATH_CALUDE_max_value_a_l2866_286619


namespace NUMINAMATH_CALUDE_no_real_roots_l2866_286664

-- Define an arithmetic sequence
def isArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∀ n m : ℕ, a (n + m) - a n = m * (a 1 - a 0)

-- Define the problem statement
theorem no_real_roots
  (a : ℕ → ℝ)
  (h_arithmetic : isArithmeticSequence a)
  (h_sum : a 2 + a 5 + a 8 = 9) :
  ∀ x : ℝ, x^2 + (a 4 + a 6) * x + 10 ≠ 0 :=
by
  sorry


end NUMINAMATH_CALUDE_no_real_roots_l2866_286664


namespace NUMINAMATH_CALUDE_coordinate_axes_equiv_product_zero_l2866_286611

/-- The set of points on the coordinate axes in a Cartesian coordinate system -/
def CoordinateAxes : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1 = 0 ∨ p.2 = 0}

/-- The set of points where the product of coordinates is zero -/
def ProductZeroSet : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1 * p.2 = 0}

theorem coordinate_axes_equiv_product_zero :
  CoordinateAxes = ProductZeroSet :=
sorry

end NUMINAMATH_CALUDE_coordinate_axes_equiv_product_zero_l2866_286611


namespace NUMINAMATH_CALUDE_farm_water_consumption_l2866_286628

theorem farm_water_consumption 
  (num_cows : ℕ)
  (cow_daily_water : ℕ)
  (sheep_cow_ratio : ℕ)
  (sheep_water_ratio : ℚ)
  (days_in_week : ℕ)
  (h1 : num_cows = 40)
  (h2 : cow_daily_water = 80)
  (h3 : sheep_cow_ratio = 10)
  (h4 : sheep_water_ratio = 1/4)
  (h5 : days_in_week = 7) :
  (num_cows * cow_daily_water * days_in_week) + 
  (num_cows * sheep_cow_ratio * (sheep_water_ratio * cow_daily_water) * days_in_week) = 78400 :=
by sorry

end NUMINAMATH_CALUDE_farm_water_consumption_l2866_286628


namespace NUMINAMATH_CALUDE_system_of_equations_solution_l2866_286692

theorem system_of_equations_solution :
  ∃! (x y : ℝ), (6 * x - 3 * y = -3) ∧ (5 * x - 9 * y = -35) ∧ x = 2 ∧ y = 5 := by
  sorry

end NUMINAMATH_CALUDE_system_of_equations_solution_l2866_286692


namespace NUMINAMATH_CALUDE_shirt_price_ratio_l2866_286674

theorem shirt_price_ratio (marked_price : ℝ) (h1 : marked_price > 0) : 
  let discount_rate : ℝ := 2 / 5
  let selling_price : ℝ := marked_price * (1 - discount_rate)
  let cost_price : ℝ := selling_price * (4 / 5)
  cost_price / marked_price = 12 / 25 := by
sorry

end NUMINAMATH_CALUDE_shirt_price_ratio_l2866_286674


namespace NUMINAMATH_CALUDE_num_fm_pairs_is_four_l2866_286630

/-- The number of possible (f,m) pairs for 7 people at a round table -/
def num_fm_pairs : ℕ :=
  let people : ℕ := 7
  4

/-- Theorem: The number of possible (f,m) pairs for 7 people at a round table is 4 -/
theorem num_fm_pairs_is_four :
  num_fm_pairs = 4 := by sorry

end NUMINAMATH_CALUDE_num_fm_pairs_is_four_l2866_286630


namespace NUMINAMATH_CALUDE_joes_lift_l2866_286666

theorem joes_lift (first_lift second_lift : ℝ)
  (h1 : first_lift + second_lift = 1800)
  (h2 : 2 * first_lift = second_lift + 300) :
  first_lift = 700 := by
sorry

end NUMINAMATH_CALUDE_joes_lift_l2866_286666


namespace NUMINAMATH_CALUDE_marble_selection_ways_l2866_286601

def total_marbles : ℕ := 15
def red_marbles : ℕ := 2
def green_marbles : ℕ := 2
def blue_marbles : ℕ := 2
def marbles_to_choose : ℕ := 5

theorem marble_selection_ways :
  (red_marbles.choose 1) * (green_marbles.choose 1) *
  ((total_marbles - red_marbles - green_marbles + 2).choose (marbles_to_choose - 2)) = 660 := by
  sorry

end NUMINAMATH_CALUDE_marble_selection_ways_l2866_286601


namespace NUMINAMATH_CALUDE_pricing_theorem_l2866_286609

/-- Proves that for an item with a marked price 50% above its cost price,
    a discount of 23.33% on the marked price results in a 15% profit,
    and the final selling price is 115% of the cost price. -/
theorem pricing_theorem (cost_price : ℝ) (cost_price_pos : cost_price > 0) :
  let marked_price := cost_price * 1.5
  let discount_percentage := 23.33 / 100
  let selling_price := marked_price * (1 - discount_percentage)
  selling_price = cost_price * 1.15 ∧ 
  (selling_price - cost_price) / cost_price = 0.15 := by
  sorry

#check pricing_theorem

end NUMINAMATH_CALUDE_pricing_theorem_l2866_286609


namespace NUMINAMATH_CALUDE_dress_count_proof_l2866_286625

def total_dresses (emily melissa debora sophia : ℕ) : ℕ :=
  emily + melissa + debora + sophia

theorem dress_count_proof 
  (emily : ℕ) 
  (h_emily : emily = 16)
  (melissa : ℕ) 
  (h_melissa : melissa = emily / 2)
  (debora : ℕ)
  (h_debora : debora = melissa + 12)
  (sophia : ℕ)
  (h_sophia : sophia = debora * 3 / 4) :
  total_dresses emily melissa debora sophia = 59 := by
sorry

end NUMINAMATH_CALUDE_dress_count_proof_l2866_286625


namespace NUMINAMATH_CALUDE_complex_expression_simplification_l2866_286632

theorem complex_expression_simplification :
  (3/2)^0 - (1 - 0.5^(-2)) / ((27/8)^(2/3)) = 7/3 := by
  sorry

end NUMINAMATH_CALUDE_complex_expression_simplification_l2866_286632


namespace NUMINAMATH_CALUDE_specific_pyramid_height_l2866_286637

/-- A right pyramid with a square base -/
structure RightPyramid where
  /-- The perimeter of the square base in inches -/
  base_perimeter : ℝ
  /-- The distance from the apex to any vertex of the base in inches -/
  apex_to_vertex : ℝ

/-- The height of a right pyramid from its apex to the center of its square base -/
def pyramid_height (p : RightPyramid) : ℝ :=
  sorry

/-- Theorem stating the height of the specific pyramid -/
theorem specific_pyramid_height :
  let p := RightPyramid.mk 40 15
  pyramid_height p = 5 * Real.sqrt 7 := by
  sorry

end NUMINAMATH_CALUDE_specific_pyramid_height_l2866_286637


namespace NUMINAMATH_CALUDE_problem_statement_inequality_statement_l2866_286633

noncomputable section

def f (x : ℝ) : ℝ := x * Real.log x
def g (a x : ℝ) : ℝ := -x^2 + a*x - 3

theorem problem_statement (a : ℝ) : 
  (∀ x > 0, 2 * f x ≥ g a x) → a ≤ 4 :=
sorry

theorem inequality_statement : 
  ∀ x > 0, Real.log x > 1 / Real.exp x - 2 / (Real.exp 1 * x) :=
sorry

end NUMINAMATH_CALUDE_problem_statement_inequality_statement_l2866_286633


namespace NUMINAMATH_CALUDE_length_AB_l2866_286602

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 4*x

-- Define the focus of the parabola
def focus : ℝ × ℝ := (1, 0)

-- Define the line with slope 1 passing through the focus
def line (x y : ℝ) : Prop := y = x - 1

-- Define the intersection points A and B
def intersection_points (A B : ℝ × ℝ) : Prop :=
  parabola A.1 A.2 ∧ line A.1 A.2 ∧ parabola B.1 B.2 ∧ line B.1 B.2

-- Theorem statement
theorem length_AB (A B : ℝ × ℝ) (h : intersection_points A B) : 
  Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 8 := by sorry

end NUMINAMATH_CALUDE_length_AB_l2866_286602


namespace NUMINAMATH_CALUDE_train_journey_properties_l2866_286607

/-- Represents the properties of a train's journey -/
structure TrainJourney where
  duration : Real
  hourly_distance : Real

/-- Defines the concept of constant speed -/
def constant_speed (journey : TrainJourney) : Prop :=
  ∀ t : Real, 0 < t → t ≤ journey.duration → 
    (t * journey.hourly_distance) / t = journey.hourly_distance

/-- Calculates the total distance traveled -/
def total_distance (journey : TrainJourney) : Real :=
  journey.duration * journey.hourly_distance

/-- Main theorem about the train's journey -/
theorem train_journey_properties (journey : TrainJourney) 
  (h1 : journey.duration = 5.5)
  (h2 : journey.hourly_distance = 100) : 
  constant_speed journey ∧ total_distance journey = 550 := by
  sorry

#check train_journey_properties

end NUMINAMATH_CALUDE_train_journey_properties_l2866_286607


namespace NUMINAMATH_CALUDE_not_exp_ix_always_one_l2866_286667

open Complex

theorem not_exp_ix_always_one (x : ℝ) : ¬ ∀ x, exp (I * x) = 1 := by
  sorry

/-- e^(ix) is a periodic function with period 2π -/
axiom exp_ix_periodic : ∀ x : ℝ, exp (I * x) = exp (I * (x + 2 * Real.pi))

/-- e^(ix) = e^(i(x + 2πk)) for any integer k -/
axiom exp_ix_shift : ∀ (x : ℝ) (k : ℤ), exp (I * x) = exp (I * (x + 2 * Real.pi * ↑k))

end NUMINAMATH_CALUDE_not_exp_ix_always_one_l2866_286667


namespace NUMINAMATH_CALUDE_unique_integer_product_digits_l2866_286617

/-- ProductOfDigits calculates the product of digits for a given natural number -/
def ProductOfDigits (n : ℕ) : ℕ := sorry

/-- The theorem states that 84 is the unique positive integer k such that 
    the product of its digits is equal to (11k/4) - 199 -/
theorem unique_integer_product_digits : 
  ∃! (k : ℕ), k > 0 ∧ ProductOfDigits k = (11 * k) / 4 - 199 := by sorry

end NUMINAMATH_CALUDE_unique_integer_product_digits_l2866_286617


namespace NUMINAMATH_CALUDE_sqrt_difference_equals_eight_thirds_sqrt_three_l2866_286622

theorem sqrt_difference_equals_eight_thirds_sqrt_three :
  Real.sqrt 27 - Real.sqrt (1/3) = (8/3) * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_difference_equals_eight_thirds_sqrt_three_l2866_286622


namespace NUMINAMATH_CALUDE_simple_interest_rate_l2866_286663

/-- Given a principal amount and a simple interest rate, if the amount after 12 years
    is 9/6 of the principal, then the rate is 100/24 -/
theorem simple_interest_rate (P R : ℝ) (P_pos : P > 0) : 
  P * (1 + R * 12 / 100) = P * (9 / 6) → R = 100 / 24 := by
  sorry

end NUMINAMATH_CALUDE_simple_interest_rate_l2866_286663


namespace NUMINAMATH_CALUDE_sum_of_seven_odds_mod_twelve_l2866_286651

theorem sum_of_seven_odds_mod_twelve (n : ℕ) (h : n = 10331) : 
  (n + (n + 2) + (n + 4) + (n + 6) + (n + 8) + (n + 10) + (n + 12)) % 12 = 7 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_seven_odds_mod_twelve_l2866_286651


namespace NUMINAMATH_CALUDE_car_cost_proof_l2866_286697

/-- The cost of Gary's used car -/
def car_cost : ℝ := 6000

/-- The monthly payment difference between 2-year and 5-year loans -/
def monthly_difference : ℝ := 150

/-- The number of months in 2 years -/
def months_in_2_years : ℝ := 2 * 12

/-- The number of months in 5 years -/
def months_in_5_years : ℝ := 5 * 12

theorem car_cost_proof :
  (car_cost / months_in_2_years) - (car_cost / months_in_5_years) = monthly_difference :=
sorry

end NUMINAMATH_CALUDE_car_cost_proof_l2866_286697


namespace NUMINAMATH_CALUDE_equation_solutions_count_l2866_286687

theorem equation_solutions_count :
  let f : ℝ → ℝ := λ x => 2 * Real.sqrt 2 * (Real.sin (π * x / 4))^3 - Real.cos (π * (1 - x) / 4)
  ∃! (solutions : Finset ℝ),
    (∀ x ∈ solutions, f x = 0 ∧ 0 ≤ x ∧ x ≤ 2020) ∧
    (∀ x, f x = 0 ∧ 0 ≤ x ∧ x ≤ 2020 → x ∈ solutions) ∧
    Finset.card solutions = 505 :=
by sorry

end NUMINAMATH_CALUDE_equation_solutions_count_l2866_286687


namespace NUMINAMATH_CALUDE_lenas_collage_glue_drops_l2866_286605

/-- Calculates the total number of glue drops needed for a collage -/
def totalGlueDrops (clippings : List Nat) (gluePerClipping : Nat) : Nat :=
  (clippings.sum) * gluePerClipping

/-- Proves that the total number of glue drops for Lena's collage is 240 -/
theorem lenas_collage_glue_drops :
  let clippings := [4, 7, 5, 3, 5, 8, 2, 6]
  let gluePerClipping := 6
  totalGlueDrops clippings gluePerClipping = 240 := by
  sorry

#eval totalGlueDrops [4, 7, 5, 3, 5, 8, 2, 6] 6

end NUMINAMATH_CALUDE_lenas_collage_glue_drops_l2866_286605


namespace NUMINAMATH_CALUDE_inverse_exponential_point_l2866_286669

theorem inverse_exponential_point (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  (Function.invFun (fun x ↦ a^x) 9 = 2) → a = 3 := by
  sorry

end NUMINAMATH_CALUDE_inverse_exponential_point_l2866_286669


namespace NUMINAMATH_CALUDE_cone_from_sector_l2866_286658

/-- Represents a circular sector -/
structure CircularSector where
  radius : ℝ
  angle : ℝ

/-- Represents a cone -/
structure Cone where
  baseRadius : ℝ
  slantHeight : ℝ
  height : ℝ

/-- Checks if a cone can be formed from a given circular sector -/
def canFormCone (sector : CircularSector) (cone : Cone) : Prop :=
  -- The slant height of the cone equals the radius of the sector
  cone.slantHeight = sector.radius ∧
  -- The arc length of the sector equals the circumference of the cone's base
  (sector.angle / 360) * (2 * Real.pi * sector.radius) = 2 * Real.pi * cone.baseRadius ∧
  -- The Pythagorean theorem holds for the cone's dimensions
  cone.slantHeight ^ 2 = cone.baseRadius ^ 2 + cone.height ^ 2

/-- Theorem stating that a specific cone can be formed from a given sector -/
theorem cone_from_sector :
  let sector := CircularSector.mk 15 300
  let cone := Cone.mk 12 15 9
  canFormCone sector cone := by
  sorry

end NUMINAMATH_CALUDE_cone_from_sector_l2866_286658


namespace NUMINAMATH_CALUDE_alice_wins_second_attempt_prob_l2866_286679

-- Define the number of cards in the deck
def deckSize : ℕ := 20

-- Define the probability of a correct guess in each turn
def probFirst : ℚ := 1 / deckSize
def probSecond : ℚ := 1 / (deckSize - 1)
def probThird : ℚ := 1 / (deckSize - 2)

-- Define the probability of Alice winning on her second attempt
def aliceWinsSecondAttempt : ℚ := (1 - probFirst) * (1 - probSecond) * probThird

-- Theorem to prove
theorem alice_wins_second_attempt_prob :
  aliceWinsSecondAttempt = 1 / deckSize := by
  sorry


end NUMINAMATH_CALUDE_alice_wins_second_attempt_prob_l2866_286679


namespace NUMINAMATH_CALUDE_not_eventually_periodic_l2866_286680

/-- The rightmost non-zero digit in the decimal representation of n! -/
def rightmost_nonzero_digit (n : ℕ) : ℕ :=
  sorry

/-- The sequence of rightmost non-zero digits of factorials -/
def a : ℕ → ℕ := rightmost_nonzero_digit

/-- The sequence (a_n)_{n ≥ 0} is not periodic from any certain point onwards -/
theorem not_eventually_periodic :
  ∀ p q : ℕ, ∃ n : ℕ, n ≥ q ∧ a n ≠ a (n + p) :=
sorry

end NUMINAMATH_CALUDE_not_eventually_periodic_l2866_286680


namespace NUMINAMATH_CALUDE_solve_nested_function_l2866_286686

def f (p : ℝ) : ℝ := 2 * p + 20

theorem solve_nested_function : ∃ p : ℝ, f (f (f p)) = -4 ∧ p = -18 := by
  sorry

end NUMINAMATH_CALUDE_solve_nested_function_l2866_286686


namespace NUMINAMATH_CALUDE_dispatch_plans_count_l2866_286695

/-- The number of vehicles in the fleet -/
def total_vehicles : ℕ := 7

/-- The number of vehicles to be dispatched -/
def dispatched_vehicles : ℕ := 4

/-- The number of ways to arrange vehicles A and B with A before B -/
def arrange_A_B : ℕ := 6

/-- The number of remaining vehicles after A and B are selected -/
def remaining_vehicles : ℕ := total_vehicles - 2

/-- The number of additional vehicles to be selected after A and B -/
def additional_vehicles : ℕ := dispatched_vehicles - 2

/-- Calculate the number of permutations of n elements taken r at a time -/
def permutations (n : ℕ) (r : ℕ) : ℕ :=
  Nat.factorial n / Nat.factorial (n - r)

theorem dispatch_plans_count :
  arrange_A_B * permutations remaining_vehicles additional_vehicles = 120 :=
sorry

end NUMINAMATH_CALUDE_dispatch_plans_count_l2866_286695


namespace NUMINAMATH_CALUDE_flower_pots_theorem_l2866_286631

/-- Represents a set of items with increasing prices -/
structure IncreasingPriceSet where
  num_items : ℕ
  price_difference : ℚ
  total_cost : ℚ

/-- The cost of the most expensive item in the set -/
def most_expensive_item_cost (s : IncreasingPriceSet) : ℚ :=
  (s.total_cost - (s.num_items - 1) * s.num_items * s.price_difference / 2) / s.num_items + (s.num_items - 1) * s.price_difference

/-- Theorem: For a set of 6 items with $0.15 price difference and $8.25 total cost, 
    the most expensive item costs $1.75 -/
theorem flower_pots_theorem : 
  let s : IncreasingPriceSet := ⟨6, 15/100, 825/100⟩
  most_expensive_item_cost s = 175/100 := by
  sorry

end NUMINAMATH_CALUDE_flower_pots_theorem_l2866_286631


namespace NUMINAMATH_CALUDE_round_robin_tournament_teams_l2866_286690

/-- Represents the total points in a round-robin tournament -/
def totalPoints (n : ℕ) : ℕ := n * (n - 1)

/-- The set of reported total points and their averages -/
def reportedPoints : Finset ℕ := {3086, 2018, 1238, 2162, 2552, 1628, 2114}

/-- Theorem stating that if one of the reported points is correct, then there are 47 teams -/
theorem round_robin_tournament_teams :
  ∃ (p : ℕ), p ∈ reportedPoints ∧ totalPoints 47 = p :=
sorry

end NUMINAMATH_CALUDE_round_robin_tournament_teams_l2866_286690


namespace NUMINAMATH_CALUDE_bracelet_selling_price_l2866_286654

def number_of_bracelets : ℕ := 12
def cost_per_bracelet : ℚ := 1
def cost_of_cookies : ℚ := 3
def money_left : ℚ := 3

def total_cost : ℚ := number_of_bracelets * cost_per_bracelet
def total_revenue : ℚ := cost_of_cookies + money_left + total_cost

theorem bracelet_selling_price :
  (total_revenue / number_of_bracelets : ℚ) = 1.75 := by
  sorry

end NUMINAMATH_CALUDE_bracelet_selling_price_l2866_286654


namespace NUMINAMATH_CALUDE_min_magnitude_sum_vectors_l2866_286641

/-- Given two vectors a and b in a real inner product space, with magnitudes 8 and 12 respectively,
    the minimum value of the magnitude of their sum is 4. -/
theorem min_magnitude_sum_vectors {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V] 
  (a b : V) (ha : ‖a‖ = 8) (hb : ‖b‖ = 12) : 
  ∃ (sum : V), sum = a + b ∧ ‖sum‖ = 4 ∧ ∀ (x : V), x = a + b → ‖x‖ ≥ 4 := by
  sorry

end NUMINAMATH_CALUDE_min_magnitude_sum_vectors_l2866_286641


namespace NUMINAMATH_CALUDE_symmetric_reflection_theorem_l2866_286612

/-- Point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Reflect a point across the xOy plane -/
def reflectXOY (p : Point3D) : Point3D :=
  { x := p.x, y := p.y, z := -p.z }

/-- Reflect a point across the z axis -/
def reflectZ (p : Point3D) : Point3D :=
  { x := -p.x, y := -p.y, z := p.z }

theorem symmetric_reflection_theorem :
  let P : Point3D := { x := 1, y := 1, z := 1 }
  let R₁ : Point3D := reflectXOY P
  let p₂ : Point3D := reflectZ R₁
  p₂ = { x := -1, y := -1, z := -1 } :=
by sorry

end NUMINAMATH_CALUDE_symmetric_reflection_theorem_l2866_286612


namespace NUMINAMATH_CALUDE_no_right_obtuse_triangle_l2866_286661

-- Define a triangle type
structure Triangle where
  angle1 : ℝ
  angle2 : ℝ
  angle3 : ℝ

-- Define properties of triangles
def Triangle.isValid (t : Triangle) : Prop :=
  t.angle1 > 0 ∧ t.angle2 > 0 ∧ t.angle3 > 0 ∧ t.angle1 + t.angle2 + t.angle3 = 180

def Triangle.hasRightAngle (t : Triangle) : Prop :=
  t.angle1 = 90 ∨ t.angle2 = 90 ∨ t.angle3 = 90

def Triangle.hasObtuseAngle (t : Triangle) : Prop :=
  t.angle1 > 90 ∨ t.angle2 > 90 ∨ t.angle3 > 90

-- Theorem: A valid triangle cannot be both right-angled and obtuse
theorem no_right_obtuse_triangle :
  ∀ t : Triangle, t.isValid → ¬(t.hasRightAngle ∧ t.hasObtuseAngle) :=
by
  sorry


end NUMINAMATH_CALUDE_no_right_obtuse_triangle_l2866_286661


namespace NUMINAMATH_CALUDE_photo_arrangements_l2866_286668

def num_boys : ℕ := 4
def num_girls : ℕ := 3

def arrangements_girls_at_ends : ℕ := 720
def arrangements_no_adjacent_girls : ℕ := 1440
def arrangements_girl_A_right_of_B : ℕ := 2520

theorem photo_arrangements :
  (num_boys = 4 ∧ num_girls = 3) →
  (arrangements_girls_at_ends = 720 ∧
   arrangements_no_adjacent_girls = 1440 ∧
   arrangements_girl_A_right_of_B = 2520) :=
by sorry

end NUMINAMATH_CALUDE_photo_arrangements_l2866_286668


namespace NUMINAMATH_CALUDE_weight_of_new_person_l2866_286656

/-- Given a group of 4 persons with a total weight W, if replacing a person
    weighing 65 kg with a new person increases the average weight by 1.5 kg,
    then the weight of the new person is 71 kg. -/
theorem weight_of_new_person (W : ℝ) : 
  (W - 65 + 71) / 4 = W / 4 + 1.5 := by sorry

end NUMINAMATH_CALUDE_weight_of_new_person_l2866_286656


namespace NUMINAMATH_CALUDE_two_lines_at_constant_distance_l2866_286699

/-- A line in a plane -/
structure Line where
  -- Add necessary fields to define a line

/-- Distance between two lines in a plane -/
def distance (l1 l2 : Line) : ℝ :=
  sorry

/-- Theorem: There are exactly two lines at a constant distance of 2 from a given line -/
theorem two_lines_at_constant_distance (l : Line) :
  ∃! (pair : (Line × Line)), (distance l pair.1 = 2 ∧ distance l pair.2 = 2) :=
sorry

end NUMINAMATH_CALUDE_two_lines_at_constant_distance_l2866_286699


namespace NUMINAMATH_CALUDE_scientific_notation_of_2150_l2866_286670

def scientific_notation (x : ℝ) : ℝ × ℤ :=
  sorry

theorem scientific_notation_of_2150 :
  scientific_notation 2150 = (2.15, 3) := by
  sorry

end NUMINAMATH_CALUDE_scientific_notation_of_2150_l2866_286670


namespace NUMINAMATH_CALUDE_sum_of_roots_quadratic_undefined_values_sum_l2866_286671

theorem sum_of_roots_quadratic : 
  ∀ (a b c : ℝ) (y₁ y₂ : ℝ), 
  a ≠ 0 → 
  a * y₁^2 + b * y₁ + c = 0 → 
  a * y₂^2 + b * y₂ + c = 0 → 
  y₁ + y₂ = -b / a :=
sorry

theorem undefined_values_sum : 
  let y₁ := (3 + Real.sqrt 49) / 2
  let y₂ := (3 - Real.sqrt 49) / 2
  y₁^2 - 3*y₁ - 10 = 0 ∧ 
  y₂^2 - 3*y₂ - 10 = 0 ∧ 
  y₁ + y₂ = 3 :=
sorry

end NUMINAMATH_CALUDE_sum_of_roots_quadratic_undefined_values_sum_l2866_286671


namespace NUMINAMATH_CALUDE_lcm_problem_l2866_286626

theorem lcm_problem (a b c : ℕ+) (h1 : a = 10) (h2 : c = 20) (h3 : Nat.lcm a (Nat.lcm b c) = 140) : b = 7 := by
  sorry

end NUMINAMATH_CALUDE_lcm_problem_l2866_286626


namespace NUMINAMATH_CALUDE_eighteenth_term_of_equally_summed_sequence_l2866_286623

/-- An Equally Summed Sequence is a sequence where the sum of each term and its subsequent term is always constant. -/
def EquallyStandardSequence (a : ℕ → ℝ) (c : ℝ) :=
  ∀ n, a n + a (n + 1) = c

theorem eighteenth_term_of_equally_summed_sequence
  (a : ℕ → ℝ)
  (h1 : EquallyStandardSequence a 5)
  (h2 : a 1 = 2) :
  a 18 = 3 := by
sorry

end NUMINAMATH_CALUDE_eighteenth_term_of_equally_summed_sequence_l2866_286623


namespace NUMINAMATH_CALUDE_sparrow_grains_l2866_286636

theorem sparrow_grains : ∃ (x : ℕ), 
  (9 * x < 1001) ∧ 
  (10 * x > 1100) ∧ 
  (x = 111) := by
sorry

end NUMINAMATH_CALUDE_sparrow_grains_l2866_286636


namespace NUMINAMATH_CALUDE_bags_needed_is_17_l2866_286634

/-- Calculates the number of bags of special dog food needed for a puppy's first year --/
def bags_needed : ℕ :=
  let days_in_year : ℕ := 365
  let ounces_per_pound : ℕ := 16
  let bag_size : ℕ := 5 -- in pounds
  let initial_period : ℕ := 60 -- in days
  let initial_daily_amount : ℕ := 2 -- in ounces
  let later_daily_amount : ℕ := 4 -- in ounces
  
  let initial_total : ℕ := initial_period * initial_daily_amount
  let later_period : ℕ := days_in_year - initial_period
  let later_total : ℕ := later_period * later_daily_amount
  
  let total_ounces : ℕ := initial_total + later_total
  let total_pounds : ℕ := (total_ounces + ounces_per_pound - 1) / ounces_per_pound
  (total_pounds + bag_size - 1) / bag_size

theorem bags_needed_is_17 : bags_needed = 17 := by
  sorry

end NUMINAMATH_CALUDE_bags_needed_is_17_l2866_286634


namespace NUMINAMATH_CALUDE_geometric_series_properties_l2866_286610

theorem geometric_series_properties (q : ℝ) (b₁ : ℝ) (h_q : |q| < 1) :
  (b₁ / (1 - q) = 16) →
  (b₁^2 / (1 - q^2) = 153.6) →
  (b₁ * q^3 = 32/9 ∧ q = 2/3) :=
by sorry

end NUMINAMATH_CALUDE_geometric_series_properties_l2866_286610


namespace NUMINAMATH_CALUDE_triangle_incenter_properties_l2866_286665

/-- 
Given a right-angled triangle ABC with angle A = 90°, sides BC = a, AC = b, AB = c,
and a line d passing through the incenter intersecting AB at P and AC at Q.
-/
theorem triangle_incenter_properties 
  (a b c : ℝ) 
  (h_right_angle : a^2 = b^2 + c^2) 
  (h_positive : a > 0 ∧ b > 0 ∧ c > 0) 
  (P Q : ℝ × ℝ) 
  (h_P_on_AB : P.1 ≥ 0 ∧ P.1 ≤ c ∧ P.2 = 0)
  (h_Q_on_AC : Q.1 = 0 ∧ Q.2 ≥ 0 ∧ Q.2 ≤ b)
  (h_PQ_through_incenter : ∃ (t : ℝ), 0 < t ∧ t < 1 ∧ 
    P = (t * c, 0) ∧ 
    Q = (0, (1 - t) * b) ∧ 
    t * c / (a + b + c) = (1 - t) * b / (a + b + c)) :
  (b * (c - P.1) / P.1 + c * (b - Q.2) / Q.2 = a) ∧
  (∃ (m : ℝ), ∀ (x y : ℝ), 
    x ≥ 0 ∧ x ≤ c ∧ y ≥ 0 ∧ y ≤ b →
    ((c - x) / x)^2 + ((b - y) / y)^2 ≥ 1) := by
  sorry

end NUMINAMATH_CALUDE_triangle_incenter_properties_l2866_286665


namespace NUMINAMATH_CALUDE_average_difference_l2866_286613

theorem average_difference (a b c : ℝ) 
  (avg_ab : (a + b) / 2 = 45)
  (avg_bc : (b + c) / 2 = 60) : 
  c - a = 30 := by
sorry

end NUMINAMATH_CALUDE_average_difference_l2866_286613


namespace NUMINAMATH_CALUDE_smallest_sum_of_roots_l2866_286600

theorem smallest_sum_of_roots (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h1 : ∃ x : ℝ, x^2 + 3*a*x + 4*b = 0)
  (h2 : ∃ x : ℝ, x^2 + 4*b*x + 3*a = 0) :
  a + b ≥ 7/3 := by
  sorry

end NUMINAMATH_CALUDE_smallest_sum_of_roots_l2866_286600


namespace NUMINAMATH_CALUDE_nested_square_root_simplification_l2866_286677

theorem nested_square_root_simplification :
  Real.sqrt (25 * Real.sqrt (25 * Real.sqrt 25)) = 5 * Real.sqrt (5 * Real.sqrt 5) := by
  sorry

end NUMINAMATH_CALUDE_nested_square_root_simplification_l2866_286677


namespace NUMINAMATH_CALUDE_number_reversal_property_l2866_286648

/-- Reverses the digits of a natural number -/
def reverseDigits (n : ℕ) : ℕ := sorry

/-- Counts the number of zero digits in a natural number -/
def countZeroDigits (n : ℕ) : ℕ := sorry

/-- Checks if a number is of the form 1099...989 with k repetitions of 99 -/
def isSpecialForm (n : ℕ) : Prop := ∃ k : ℕ, n = 10^(2*k+1) + 9 * (10^(2*k) - 1) / 99

theorem number_reversal_property (N : ℕ) :
  (9 * N = reverseDigits N) ∧ (countZeroDigits N ≤ 1) ↔ N = 0 ∨ isSpecialForm N :=
sorry

end NUMINAMATH_CALUDE_number_reversal_property_l2866_286648


namespace NUMINAMATH_CALUDE_store_max_profit_l2866_286643

/-- Represents the profit function for a store selling seasonal goods -/
def profit_function (x : ℝ) : ℝ := -5 * x^2 + 500 * x + 20000

/-- The maximum profit achieved by the store -/
def max_profit : ℝ := 32500

theorem store_max_profit :
  ∃ (x : ℝ), 
    (∀ y : ℝ, profit_function y ≤ profit_function x) ∧ 
    profit_function x = max_profit :=
by sorry

end NUMINAMATH_CALUDE_store_max_profit_l2866_286643


namespace NUMINAMATH_CALUDE_zoo_animals_l2866_286682

theorem zoo_animals (X : ℕ) : 
  X - 6 + 1 + 3 + 8 + 16 = 90 → X = 68 := by
sorry

end NUMINAMATH_CALUDE_zoo_animals_l2866_286682


namespace NUMINAMATH_CALUDE_least_addition_for_divisibility_l2866_286652

theorem least_addition_for_divisibility : 
  (∃ x : ℕ, x ≥ 0 ∧ (228712 + x) % (2 * 3 * 5) = 0) ∧ 
  (∀ y : ℕ, y ≥ 0 ∧ (228712 + y) % (2 * 3 * 5) = 0 → y ≥ 8) :=
by sorry

end NUMINAMATH_CALUDE_least_addition_for_divisibility_l2866_286652
