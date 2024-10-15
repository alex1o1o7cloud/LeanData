import Mathlib

namespace NUMINAMATH_CALUDE_second_catch_size_l1095_109563

theorem second_catch_size (tagged_initial : ℕ) (tagged_second : ℕ) (total_fish : ℕ) :
  tagged_initial = 50 →
  tagged_second = 2 →
  total_fish = 1250 →
  (tagged_second : ℚ) / (tagged_initial : ℚ) = (tagged_second : ℚ) / x →
  x = 50 :=
by sorry

end NUMINAMATH_CALUDE_second_catch_size_l1095_109563


namespace NUMINAMATH_CALUDE_fraction_simplification_l1095_109520

theorem fraction_simplification (x : ℝ) (h : x ≠ 1) : (5 * x) / (x - 1) - 5 / (x - 1) = 5 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l1095_109520


namespace NUMINAMATH_CALUDE_large_box_chocolate_count_l1095_109578

/-- The number of chocolate bars in a large box -/
def total_chocolate_bars (num_small_boxes : ℕ) (bars_per_small_box : ℕ) : ℕ :=
  num_small_boxes * bars_per_small_box

/-- Theorem: The large box contains 475 chocolate bars -/
theorem large_box_chocolate_count :
  total_chocolate_bars 19 25 = 475 := by
  sorry

end NUMINAMATH_CALUDE_large_box_chocolate_count_l1095_109578


namespace NUMINAMATH_CALUDE_probability_two_ones_eight_dice_l1095_109506

/-- The probability of exactly two dice showing a 1 when rolling eight standard 6-sided dice -/
def probability_two_ones (n : ℕ) (k : ℕ) (p : ℚ) : ℚ :=
  ↑(Nat.choose n k) * p^k * (1 - p)^(n - k)

/-- Theorem stating the probability of exactly two ones when rolling eight 6-sided dice -/
theorem probability_two_ones_eight_dice :
  probability_two_ones 8 2 (1/6) = 28 * (1/6)^2 * (5/6)^6 :=
sorry

end NUMINAMATH_CALUDE_probability_two_ones_eight_dice_l1095_109506


namespace NUMINAMATH_CALUDE_coord_sum_of_point_on_line_l1095_109545

/-- Given two points A and B in a 2D plane, where A is at the origin and B is on the line y = 5,
    if the slope of segment AB is 3/4, then the sum of the x- and y-coordinates of B is 35/3. -/
theorem coord_sum_of_point_on_line (B : ℝ × ℝ) : 
  B.2 = 5 →  -- B is on the line y = 5
  (B.2 - 0) / (B.1 - 0) = 3/4 →  -- slope of AB is 3/4
  B.1 + B.2 = 35/3 := by
sorry

end NUMINAMATH_CALUDE_coord_sum_of_point_on_line_l1095_109545


namespace NUMINAMATH_CALUDE_perpendicular_necessary_not_sufficient_l1095_109511

-- Define the types for lines and relationships
def Line : Type := ℝ → ℝ → Prop
def Perpendicular (l₁ l₂ : Line) : Prop := sorry
def Parallel (l₁ l₂ : Line) : Prop := sorry

-- State the theorem
theorem perpendicular_necessary_not_sufficient 
  (a b c : Line) (h : Perpendicular a b) : 
  (∀ (a b c : Line), Parallel b c → Perpendicular a c) ∧ 
  (∃ (a b c : Line), Perpendicular a c ∧ ¬Parallel b c) := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_necessary_not_sufficient_l1095_109511


namespace NUMINAMATH_CALUDE_incorrect_statement_l1095_109556

theorem incorrect_statement : ¬(∀ (p q : Prop), (¬(p ∧ q)) → (p ∨ q)) := by
  sorry

end NUMINAMATH_CALUDE_incorrect_statement_l1095_109556


namespace NUMINAMATH_CALUDE_bales_in_barn_l1095_109581

/-- The number of bales in the barn after Tim's addition -/
def total_bales (initial_bales added_bales : ℕ) : ℕ :=
  initial_bales + added_bales

/-- Theorem stating that the total number of bales after Tim's addition is 54 -/
theorem bales_in_barn (initial_bales added_bales : ℕ) 
  (h1 : initial_bales = 28)
  (h2 : added_bales = 26) :
  total_bales initial_bales added_bales = 54 := by
  sorry

end NUMINAMATH_CALUDE_bales_in_barn_l1095_109581


namespace NUMINAMATH_CALUDE_edward_lawn_problem_l1095_109593

/-- The number of dollars Edward earns per lawn -/
def dollars_per_lawn : ℕ := 4

/-- The number of lawns Edward forgot to mow -/
def forgotten_lawns : ℕ := 9

/-- The amount of money Edward actually earned -/
def actual_earnings : ℕ := 32

/-- The original number of lawns Edward had to mow -/
def original_lawns : ℕ := 17

theorem edward_lawn_problem :
  dollars_per_lawn * (original_lawns - forgotten_lawns) = actual_earnings :=
by sorry

end NUMINAMATH_CALUDE_edward_lawn_problem_l1095_109593


namespace NUMINAMATH_CALUDE_alia_has_40_markers_l1095_109500

-- Define the number of markers for each person
def steve_markers : ℕ := 60
def austin_markers : ℕ := steve_markers / 3
def alia_markers : ℕ := 2 * austin_markers

-- Theorem statement
theorem alia_has_40_markers : alia_markers = 40 := by
  sorry

end NUMINAMATH_CALUDE_alia_has_40_markers_l1095_109500


namespace NUMINAMATH_CALUDE_pad_usage_duration_l1095_109560

/-- Represents the number of sheets in a pad of paper -/
def sheets_per_pad : ℕ := 60

/-- Represents the number of working days per week -/
def working_days_per_week : ℕ := 3

/-- Represents the number of sheets used per working day -/
def sheets_per_day : ℕ := 12

/-- Calculates the number of weeks it takes to use a full pad of paper -/
def weeks_per_pad : ℚ :=
  sheets_per_pad / (working_days_per_week * sheets_per_day)

/-- Theorem stating that the rounded-up number of weeks to use a pad is 2 -/
theorem pad_usage_duration :
  ⌈weeks_per_pad⌉ = 2 := by sorry

end NUMINAMATH_CALUDE_pad_usage_duration_l1095_109560


namespace NUMINAMATH_CALUDE_carol_peanuts_l1095_109596

/-- The total number of peanuts Carol has -/
def total_peanuts (tree_peanuts ground_peanuts bags bag_capacity : ℕ) : ℕ :=
  tree_peanuts + ground_peanuts + bags * bag_capacity

/-- Theorem: Carol has 976 peanuts in total -/
theorem carol_peanuts :
  total_peanuts 48 178 3 250 = 976 := by
  sorry

end NUMINAMATH_CALUDE_carol_peanuts_l1095_109596


namespace NUMINAMATH_CALUDE_base6_addition_problem_l1095_109542

-- Define a function to convert a base 6 number to base 10
def base6ToBase10 (d₂ d₁ d₀ : Nat) : Nat :=
  d₂ * 6^2 + d₁ * 6^1 + d₀ * 6^0

-- Define a function to convert a base 10 number to base 6
def base10ToBase6 (n : Nat) : Nat × Nat × Nat :=
  let d₂ := n / 36
  let r₂ := n % 36
  let d₁ := r₂ / 6
  let d₀ := r₂ % 6
  (d₂, d₁, d₀)

theorem base6_addition_problem :
  ∀ S H E : Nat,
    S ≠ 0 ∧ H ≠ 0 ∧ E ≠ 0 →
    S < 6 ∧ H < 6 ∧ E < 6 →
    S ≠ H ∧ S ≠ E ∧ H ≠ E →
    base6ToBase10 S H E + base6ToBase10 0 H E = base6ToBase10 E S H →
    S = 5 ∧ H = 4 ∧ E = 5 ∧ base10ToBase6 (S + H + E) = (2, 2, 0) := by
  sorry

end NUMINAMATH_CALUDE_base6_addition_problem_l1095_109542


namespace NUMINAMATH_CALUDE_max_S_n_is_three_halves_l1095_109580

/-- Given a geometric sequence {a_n} with first term 3/2 and common ratio -1/2,
    S_n is the sum of the first n terms. -/
def S_n (n : ℕ) : ℚ :=
  (3/2) * (1 - (-1/2)^n) / (1 - (-1/2))

/-- The maximum value of S_n is 3/2. -/
theorem max_S_n_is_three_halves :
  ∃ (M : ℚ), M = 3/2 ∧ ∀ (n : ℕ), S_n n ≤ M :=
sorry

end NUMINAMATH_CALUDE_max_S_n_is_three_halves_l1095_109580


namespace NUMINAMATH_CALUDE_x_total_time_is_20_l1095_109517

-- Define the work as a fraction of the total job
def Work := ℚ

-- Define the time y needs to finish the entire work
def y_total_time : ℕ := 16

-- Define the time y worked before leaving
def y_worked_time : ℕ := 12

-- Define the time x needed to finish the remaining work
def x_remaining_time : ℕ := 5

-- Theorem to prove
theorem x_total_time_is_20 : 
  ∃ (x_total_time : ℕ), 
    (y_worked_time : ℚ) / y_total_time + 
    (x_remaining_time : ℚ) / x_total_time = 1 ∧ 
    x_total_time = 20 := by sorry

end NUMINAMATH_CALUDE_x_total_time_is_20_l1095_109517


namespace NUMINAMATH_CALUDE_expression_equality_l1095_109505

theorem expression_equality : 
  (|(-1)|^2023 : ℝ) + (Real.sqrt 3)^2 - 2 * Real.sin (π / 6) + (1 / 2)⁻¹ = 5 := by
  sorry

end NUMINAMATH_CALUDE_expression_equality_l1095_109505


namespace NUMINAMATH_CALUDE_intersection_points_count_l1095_109527

/-- Represents a line in the form ax + by = c --/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Checks if two lines are parallel --/
def parallel (l1 l2 : Line) : Prop :=
  l1.a * l2.b = l1.b * l2.a

/-- Checks if a point (x, y) lies on a line --/
def point_on_line (x y : ℝ) (l : Line) : Prop :=
  l.a * x + l.b * y = l.c

/-- The three lines given in the problem --/
def line1 : Line := { a := -3, b := 4, c := 2 }
def line2 : Line := { a := 2, b := 4, c := 4 }
def line3 : Line := { a := 6, b := -8, c := 3 }

theorem intersection_points_count :
  ∃ (p1 p2 : ℝ × ℝ),
    p1 ≠ p2 ∧
    (point_on_line p1.1 p1.2 line1 ∨ point_on_line p1.1 p1.2 line2 ∨ point_on_line p1.1 p1.2 line3) ∧
    (point_on_line p1.1 p1.2 line1 ∨ point_on_line p1.1 p1.2 line2 ∨ point_on_line p1.1 p1.2 line3) ∧
    (point_on_line p2.1 p2.2 line1 ∨ point_on_line p2.1 p2.2 line2 ∨ point_on_line p2.1 p2.2 line3) ∧
    (point_on_line p2.1 p2.2 line1 ∨ point_on_line p2.1 p2.2 line2 ∨ point_on_line p2.1 p2.2 line3) ∧
    (∀ (p : ℝ × ℝ),
      p ≠ p1 → p ≠ p2 →
      ¬((point_on_line p.1 p.2 line1 ∧ point_on_line p.1 p.2 line2) ∨
        (point_on_line p.1 p.2 line1 ∧ point_on_line p.1 p.2 line3) ∨
        (point_on_line p.1 p.2 line2 ∧ point_on_line p.1 p.2 line3))) :=
by
  sorry

end NUMINAMATH_CALUDE_intersection_points_count_l1095_109527


namespace NUMINAMATH_CALUDE_roots_product_plus_one_l1095_109598

theorem roots_product_plus_one (a b c : ℝ) : 
  (a^3 - 15*a^2 + 25*a - 10 = 0) →
  (b^3 - 15*b^2 + 25*b - 10 = 0) →
  (c^3 - 15*c^2 + 25*c - 10 = 0) →
  (1+a)*(1+b)*(1+c) = 51 := by
  sorry

end NUMINAMATH_CALUDE_roots_product_plus_one_l1095_109598


namespace NUMINAMATH_CALUDE_regular_polygon_center_containment_l1095_109555

/-- A regular polygon with 2n sides -/
structure RegularPolygon (n : ℕ) where
  sideLength : ℝ
  center : ℝ × ℝ

/-- Predicate to check if one polygon is inside another -/
def isInside (p1 p2 : RegularPolygon n) : Prop := sorry

/-- Predicate to check if a point is inside a polygon -/
def containsPoint (p : RegularPolygon n) (point : ℝ × ℝ) : Prop := sorry

theorem regular_polygon_center_containment (n : ℕ) (a : ℝ) 
  (M₁ : RegularPolygon n) (M₂ : RegularPolygon n) 
  (h1 : M₁.sideLength = a) 
  (h2 : M₂.sideLength = 2 * a) 
  (h3 : isInside M₁ M₂) :
  containsPoint M₁ M₂.center := by
  sorry

end NUMINAMATH_CALUDE_regular_polygon_center_containment_l1095_109555


namespace NUMINAMATH_CALUDE_kangaroo_meeting_count_l1095_109537

def kangaroo_a_period : ℕ := 9
def kangaroo_b_period : ℕ := 6
def total_jumps : ℕ := 2017

def meeting_count (a_period b_period total_jumps : ℕ) : ℕ :=
  let lcm := Nat.lcm a_period b_period
  let meetings_per_cycle := 2  -- They meet twice in each LCM cycle
  let complete_cycles := total_jumps / lcm
  let remainder := total_jumps % lcm
  let meetings_in_complete_cycles := complete_cycles * meetings_per_cycle
  let initial_meeting := 1  -- They start at the same point
  let extra_meeting := if remainder ≥ 1 then 1 else 0
  meetings_in_complete_cycles + initial_meeting + extra_meeting

theorem kangaroo_meeting_count :
  meeting_count kangaroo_a_period kangaroo_b_period total_jumps = 226 := by
  sorry

end NUMINAMATH_CALUDE_kangaroo_meeting_count_l1095_109537


namespace NUMINAMATH_CALUDE_intersection_complement_theorem_l1095_109595

def U : Set Nat := {1, 2, 3, 4, 5, 6}
def A : Set Nat := {1, 3, 6}
def B : Set Nat := {2, 3, 4}

theorem intersection_complement_theorem : A ∩ (U \ B) = {1, 6} := by
  sorry

end NUMINAMATH_CALUDE_intersection_complement_theorem_l1095_109595


namespace NUMINAMATH_CALUDE_train_journey_constant_speed_time_l1095_109558

/-- Represents the journey of a train with uniform acceleration, constant speed, and uniform deceleration phases. -/
structure TrainJourney where
  totalDistance : ℝ  -- in km
  totalTime : ℝ      -- in hours
  constantSpeed : ℝ  -- in km/h

/-- Calculates the time spent at constant speed during the journey. -/
def timeAtConstantSpeed (journey : TrainJourney) : ℝ :=
  sorry

/-- Theorem stating that for the given journey parameters, the time at constant speed is 1/5 hours (12 minutes). -/
theorem train_journey_constant_speed_time 
  (journey : TrainJourney) 
  (h1 : journey.totalDistance = 21) 
  (h2 : journey.totalTime = 4/15)
  (h3 : journey.constantSpeed = 90) :
  timeAtConstantSpeed journey = 1/5 := by
  sorry

end NUMINAMATH_CALUDE_train_journey_constant_speed_time_l1095_109558


namespace NUMINAMATH_CALUDE_polynomial_divisibility_l1095_109564

theorem polynomial_divisibility (a b c d : ℤ) : 
  (∀ x : ℤ, ∃ k : ℤ, a * x^3 + b * x^2 + c * x + d = 5 * k) →
  (∃ (ka kb kc kd : ℤ), a = 5 * ka ∧ b = 5 * kb ∧ c = 5 * kc ∧ d = 5 * kd) :=
by sorry

end NUMINAMATH_CALUDE_polynomial_divisibility_l1095_109564


namespace NUMINAMATH_CALUDE_five_or_king_probability_l1095_109540

/-- Represents a standard deck of cards -/
structure Deck :=
  (cards : Finset (Fin 52))
  (card_count : cards.card = 52)

/-- Represents the ranks in a standard deck -/
inductive Rank
  | Ace | Two | Three | Four | Five | Six | Seven | Eight | Nine | Ten | Jack | Queen | King

/-- Represents the suits in a standard deck -/
inductive Suit
  | Spades | Hearts | Diamonds | Clubs

/-- A card in the deck -/
structure Card :=
  (rank : Rank)
  (suit : Suit)

/-- The probability of drawing a specific card from the deck -/
def draw_probability (d : Deck) (c : Card) : ℚ :=
  1 / 52

/-- The probability of drawing a card with a specific rank -/
def draw_rank_probability (d : Deck) (r : Rank) : ℚ :=
  4 / 52

/-- Theorem: The probability of drawing either a 5 or a King from a standard 52-card deck is 2/13 -/
theorem five_or_king_probability (d : Deck) : 
  draw_rank_probability d Rank.Five + draw_rank_probability d Rank.King = 2 / 13 := by
  sorry


end NUMINAMATH_CALUDE_five_or_king_probability_l1095_109540


namespace NUMINAMATH_CALUDE_mirasol_account_balance_l1095_109587

def remaining_balance (initial_balance : ℕ) (expense1 : ℕ) (expense2 : ℕ) : ℕ :=
  initial_balance - (expense1 + expense2)

theorem mirasol_account_balance : remaining_balance 50 10 30 = 10 := by
  sorry

end NUMINAMATH_CALUDE_mirasol_account_balance_l1095_109587


namespace NUMINAMATH_CALUDE_range_of_a_l1095_109510

theorem range_of_a (a : ℝ) : 
  (∀ x > a, x * (x - 1) > 0) ↔ a ≥ 1 := by
sorry

end NUMINAMATH_CALUDE_range_of_a_l1095_109510


namespace NUMINAMATH_CALUDE_magic_king_episodes_l1095_109554

/-- Calculates the total number of episodes for a TV show with a given number of seasons and episodes per season for each half. -/
def totalEpisodes (totalSeasons : ℕ) (episodesFirstHalf : ℕ) (episodesSecondHalf : ℕ) : ℕ :=
  let halfSeasons := totalSeasons / 2
  halfSeasons * episodesFirstHalf + halfSeasons * episodesSecondHalf

/-- Proves that a show with 10 seasons, 20 episodes per season for the first half, and 25 episodes per season for the second half has 225 total episodes. -/
theorem magic_king_episodes : totalEpisodes 10 20 25 = 225 := by
  sorry

end NUMINAMATH_CALUDE_magic_king_episodes_l1095_109554


namespace NUMINAMATH_CALUDE_initial_roses_l1095_109597

theorem initial_roses (initial : ℕ) (added : ℕ) (total : ℕ) : 
  added = 10 → total = 16 → total = initial + added → initial = 6 := by
  sorry

end NUMINAMATH_CALUDE_initial_roses_l1095_109597


namespace NUMINAMATH_CALUDE_abs_neg_one_third_l1095_109524

theorem abs_neg_one_third : |(-1 : ℚ) / 3| = 1 / 3 := by sorry

end NUMINAMATH_CALUDE_abs_neg_one_third_l1095_109524


namespace NUMINAMATH_CALUDE_dress_discount_percentage_l1095_109546

theorem dress_discount_percentage (d : ℝ) (x : ℝ) 
  (h1 : d > 0) 
  (h2 : 0.6 * d = d * (1 - x / 100) * 0.8) : 
  x = 25 := by sorry

end NUMINAMATH_CALUDE_dress_discount_percentage_l1095_109546


namespace NUMINAMATH_CALUDE_intersection_at_one_point_l1095_109591

theorem intersection_at_one_point (b : ℝ) : 
  (∃! x : ℝ, b * x^2 + 2 * x + 3 = 3 * x + 4) ↔ b = -1/4 := by
sorry

end NUMINAMATH_CALUDE_intersection_at_one_point_l1095_109591


namespace NUMINAMATH_CALUDE_twentieth_fisherman_catch_l1095_109574

theorem twentieth_fisherman_catch (total_fishermen : ℕ) (total_fish : ℕ) 
  (fishermen_with_400 : ℕ) (fish_per_fisherman : ℕ) :
  total_fishermen = 20 →
  total_fish = 10000 →
  fishermen_with_400 = 19 →
  fish_per_fisherman = 400 →
  total_fish - (fishermen_with_400 * fish_per_fisherman) = 2400 := by
sorry

end NUMINAMATH_CALUDE_twentieth_fisherman_catch_l1095_109574


namespace NUMINAMATH_CALUDE_line_bisects_and_perpendicular_l1095_109550

/-- The circle C in the xy-plane -/
def Circle (x y : ℝ) : Prop := x^2 + y^2 + x - 2*y + 1 = 0

/-- The line perpendicular to l -/
def PerpendicularLine (x y : ℝ) : Prop := x + 2*y + 3 = 0

/-- The line l -/
def Line_l (x y : ℝ) : Prop := 2*x - y + 2 = 0

/-- Theorem stating that line l bisects circle C and is perpendicular to the given line -/
theorem line_bisects_and_perpendicular :
  (∀ x y : ℝ, Line_l x y → (∃ x' y' : ℝ, Circle x' y' ∧ x = (x' + (-1/2))/2 ∧ y = (y' + 1)/2)) ∧ 
  (∀ x y : ℝ, Line_l x y → PerpendicularLine x y → x * 2 + y * 1 = 0) :=
sorry

end NUMINAMATH_CALUDE_line_bisects_and_perpendicular_l1095_109550


namespace NUMINAMATH_CALUDE_satisfactory_grades_fraction_l1095_109547

/-- Represents the grades in a class -/
structure ClassGrades where
  total_students : Nat
  grade_a : Nat
  grade_b : Nat
  grade_c : Nat
  grade_d : Nat
  grade_f : Nat

/-- Calculates the fraction of satisfactory grades -/
def satisfactory_fraction (grades : ClassGrades) : Rat :=
  (grades.grade_a + grades.grade_b + grades.grade_c : Rat) / grades.total_students

/-- The main theorem about the fraction of satisfactory grades -/
theorem satisfactory_grades_fraction :
  let grades : ClassGrades := {
    total_students := 30,
    grade_a := 8,
    grade_b := 7,
    grade_c := 6,
    grade_d := 5,
    grade_f := 4
  }
  satisfactory_fraction grades = 7 / 10 := by sorry

end NUMINAMATH_CALUDE_satisfactory_grades_fraction_l1095_109547


namespace NUMINAMATH_CALUDE_division_problem_l1095_109585

theorem division_problem (x y : ℤ) (k : ℕ) (h1 : x > 0) 
  (h2 : x = 11 * y + 4) 
  (h3 : 2 * x = k * (3 * y) + 1) 
  (h4 : 7 * y - x = 3) : 
  k = 6 := by sorry

end NUMINAMATH_CALUDE_division_problem_l1095_109585


namespace NUMINAMATH_CALUDE_league_games_count_l1095_109515

theorem league_games_count (n : ℕ) (h : n = 14) : 
  (n * (n - 1)) / 2 = 91 := by
  sorry

#check league_games_count

end NUMINAMATH_CALUDE_league_games_count_l1095_109515


namespace NUMINAMATH_CALUDE_final_score_l1095_109561

def bullseye_points : ℕ := 50

def dart_throws (bullseye half_bullseye miss : ℕ) : Prop :=
  bullseye = bullseye_points ∧
  half_bullseye = bullseye_points / 2 ∧
  miss = 0

theorem final_score (bullseye half_bullseye miss : ℕ) 
  (h : dart_throws bullseye half_bullseye miss) : 
  bullseye + half_bullseye + miss = 75 := by
  sorry

end NUMINAMATH_CALUDE_final_score_l1095_109561


namespace NUMINAMATH_CALUDE_minimum_students_in_class_l1095_109502

theorem minimum_students_in_class (b g : ℕ) : 
  (b ≠ 0 ∧ g ≠ 0) →  -- Ensure non-zero numbers of boys and girls
  (2 * (b / 2) = 3 * (g / 3)) →  -- Half of boys equals two-thirds of girls who passed
  (b / 2 = 2 * (g / 3)) →  -- Boys who failed is twice girls who failed
  7 ≤ b + g  -- The total number of students is at least 7
  ∧ ∃ (b' g' : ℕ), b' ≠ 0 ∧ g' ≠ 0 
     ∧ (2 * (b' / 2) = 3 * (g' / 3))
     ∧ (b' / 2 = 2 * (g' / 3))
     ∧ b' + g' = 7  -- There exists a solution with exactly 7 students
  := by sorry

end NUMINAMATH_CALUDE_minimum_students_in_class_l1095_109502


namespace NUMINAMATH_CALUDE_mine_locations_determinable_l1095_109518

/-- Represents the state of a cell in the grid -/
inductive CellState
  | Empty
  | Mine

/-- Represents the grid of cells -/
def Grid (n : ℕ) := Fin n → Fin n → CellState

/-- The number displayed in a cell, which is the count of mines in the cell and its surroundings -/
def CellNumber (n : ℕ) (grid : Grid n) (i j : Fin n) : Fin 10 :=
  sorry

/-- Checks if it's possible to uniquely determine mine locations given cell numbers -/
def CanDetermineMineLocations (n : ℕ) (cellNumbers : Fin n → Fin n → Fin 10) : Prop :=
  ∃! (grid : Grid n), ∀ (i j : Fin n), CellNumber n grid i j = cellNumbers i j

/-- Theorem stating that mine locations can be determined for n = 2009 and n = 2007 -/
theorem mine_locations_determinable :
  (∀ (cellNumbers : Fin 2009 → Fin 2009 → Fin 10), CanDetermineMineLocations 2009 cellNumbers) ∧
  (∀ (cellNumbers : Fin 2007 → Fin 2007 → Fin 10), CanDetermineMineLocations 2007 cellNumbers) :=
sorry

end NUMINAMATH_CALUDE_mine_locations_determinable_l1095_109518


namespace NUMINAMATH_CALUDE_similar_triangles_side_length_l1095_109588

/-- Represents a triangle with an area and a side length -/
structure Triangle where
  area : ℝ
  side : ℝ

/-- Proves that given two similar triangles with specific properties, 
    the corresponding side of the larger triangle is 12 feet -/
theorem similar_triangles_side_length 
  (t1 t2 : Triangle) 
  (h_area_diff : t1.area - t2.area = 32)
  (h_area_ratio : t1.area / t2.area = 9)
  (h_smaller_area_int : ∃ n : ℕ, t2.area = n)
  (h_smaller_side : t2.side = 4)
  (h_similar : ∃ k : ℝ, t1.side = k * t2.side ∧ t1.area = k^2 * t2.area) :
  t1.side = 12 := by
  sorry

#check similar_triangles_side_length

end NUMINAMATH_CALUDE_similar_triangles_side_length_l1095_109588


namespace NUMINAMATH_CALUDE_cos_eighteen_degrees_l1095_109582

theorem cos_eighteen_degrees : Real.cos (18 * π / 180) = (5 + Real.sqrt 5) / 8 := by
  sorry

end NUMINAMATH_CALUDE_cos_eighteen_degrees_l1095_109582


namespace NUMINAMATH_CALUDE_equilateral_triangle_perimeter_l1095_109557

/-- An equilateral triangle with area twice the side length has perimeter 8√3 -/
theorem equilateral_triangle_perimeter (s : ℝ) (h : s > 0) : 
  (s^2 * Real.sqrt 3) / 4 = 2 * s → 3 * s = 8 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_equilateral_triangle_perimeter_l1095_109557


namespace NUMINAMATH_CALUDE_unique_three_digit_number_l1095_109543

theorem unique_three_digit_number : ∃! (m g u : ℕ),
  m ≠ g ∧ m ≠ u ∧ g ≠ u ∧
  m < 10 ∧ g < 10 ∧ u < 10 ∧
  m ≥ 1 ∧
  100 * m + 10 * g + u = (m + g + u) * (m + g + u - 2) ∧
  100 * m + 10 * g + u = 195 := by
sorry

end NUMINAMATH_CALUDE_unique_three_digit_number_l1095_109543


namespace NUMINAMATH_CALUDE_polynomial_value_at_3_l1095_109552

def is_valid_coeff (b : ℤ) : Prop := 0 ≤ b ∧ b < 5

def P (b : Fin 6 → ℤ) (x : ℝ) : ℝ :=
  (b 0) + (b 1) * x + (b 2) * x^2 + (b 3) * x^3 + (b 4) * x^4 + (b 5) * x^5

theorem polynomial_value_at_3 (b : Fin 6 → ℤ) :
  (∀ i : Fin 6, is_valid_coeff (b i)) →
  P b (Real.sqrt 5) = 40 + 31 * Real.sqrt 5 →
  P b 3 = 381 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_value_at_3_l1095_109552


namespace NUMINAMATH_CALUDE_product_of_positive_real_solutions_l1095_109504

theorem product_of_positive_real_solutions : ∃ (S : Finset (ℂ)),
  (∀ z ∈ S, z^8 = -256 ∧ z.re > 0) ∧
  (∀ z : ℂ, z^8 = -256 ∧ z.re > 0 → z ∈ S) ∧
  (S.prod id = 8) :=
sorry

end NUMINAMATH_CALUDE_product_of_positive_real_solutions_l1095_109504


namespace NUMINAMATH_CALUDE_circle_intersection_range_l1095_109538

-- Define the circles
def circle1 (a : ℝ) (x y : ℝ) : Prop := (x - a)^2 + y^2 = 1
def circle2 (x y : ℝ) : Prop := x^2 + y^2 = 25

-- Define the intersection condition
def intersect (a : ℝ) : Prop := ∃ x y : ℝ, circle1 a x y ∧ circle2 x y

-- Define the range of a
def valid_range (a : ℝ) : Prop := (a > -6 ∧ a < -4) ∨ (a > 4 ∧ a < 6)

-- Theorem statement
theorem circle_intersection_range :
  ∀ a : ℝ, intersect a ↔ valid_range a := by sorry

end NUMINAMATH_CALUDE_circle_intersection_range_l1095_109538


namespace NUMINAMATH_CALUDE_two_envelopes_require_fee_l1095_109533

-- Define the envelope structure
structure Envelope where
  name : String
  length : ℚ
  height : ℚ

-- Define the condition for additional fee
def requiresAdditionalFee (e : Envelope) : Bool :=
  let ratio := e.length / e.height
  ratio < 1.5 || ratio > 2.8

-- Define the list of envelopes
def envelopes : List Envelope := [
  ⟨"E", 7, 5⟩,
  ⟨"F", 10, 4⟩,
  ⟨"G", 5, 5⟩,
  ⟨"H", 14, 5⟩
]

-- Theorem statement
theorem two_envelopes_require_fee :
  (envelopes.filter requiresAdditionalFee).length = 2 := by
  sorry

end NUMINAMATH_CALUDE_two_envelopes_require_fee_l1095_109533


namespace NUMINAMATH_CALUDE_anderson_shirts_theorem_l1095_109523

theorem anderson_shirts_theorem (total_clothing pieces_of_trousers : ℕ) 
  (h1 : total_clothing = 934)
  (h2 : pieces_of_trousers = 345) :
  total_clothing - pieces_of_trousers = 589 := by
  sorry

end NUMINAMATH_CALUDE_anderson_shirts_theorem_l1095_109523


namespace NUMINAMATH_CALUDE_recipe_flour_calculation_l1095_109521

/-- The amount of flour required for a recipe -/
def recipe_flour (initial : ℕ) (additional : ℕ) : ℕ :=
  initial + additional

theorem recipe_flour_calculation (initial : ℕ) (additional : ℕ) :
  recipe_flour initial additional = initial + additional :=
by sorry

end NUMINAMATH_CALUDE_recipe_flour_calculation_l1095_109521


namespace NUMINAMATH_CALUDE_direct_proportion_shift_right_l1095_109594

/-- A linear function represented by its slope and y-intercept -/
structure LinearFunction where
  slope : ℝ
  intercept : ℝ

/-- Shift a linear function horizontally -/
def shift_right (f : LinearFunction) (shift : ℝ) : LinearFunction :=
  { slope := f.slope, intercept := f.slope * shift + f.intercept }

theorem direct_proportion_shift_right :
  let f : LinearFunction := { slope := -2, intercept := 0 }
  let shifted_f := shift_right f 3
  shifted_f.slope = -2 ∧ shifted_f.intercept = 6 := by sorry

end NUMINAMATH_CALUDE_direct_proportion_shift_right_l1095_109594


namespace NUMINAMATH_CALUDE_tan_seventeen_pi_fourths_l1095_109507

theorem tan_seventeen_pi_fourths : Real.tan (17 * π / 4) = 1 := by
  sorry

end NUMINAMATH_CALUDE_tan_seventeen_pi_fourths_l1095_109507


namespace NUMINAMATH_CALUDE_money_division_l1095_109577

theorem money_division (alice bond charlie : ℕ) 
  (h1 : charlie = 495)
  (h2 : (alice - 10) * 18 * 24 = (bond - 20) * 11 * 24)
  (h3 : (alice - 10) * 24 * 18 = (charlie - 15) * 11 * 18)
  (h4 : (bond - 20) * 24 * 11 = (charlie - 15) * 18 * 11) :
  alice + bond + charlie = 1105 := by
  sorry

end NUMINAMATH_CALUDE_money_division_l1095_109577


namespace NUMINAMATH_CALUDE_at_most_three_lines_unique_line_through_two_points_l1095_109534

-- Define a point in a plane
structure Point :=
  (x : ℝ)
  (y : ℝ)

-- Define a line in a plane
structure Line :=
  (a : ℝ)
  (b : ℝ)
  (c : ℝ)

-- Define a function to check if a point is on a line
def isPointOnLine (p : Point) (l : Line) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

-- Define a function to create a line from two points
def lineFromPoints (p1 p2 : Point) : Line :=
  { a := p2.y - p1.y,
    b := p1.x - p2.x,
    c := p2.x * p1.y - p1.x * p2.y }

-- Theorem 1: At most three lines can be drawn through any two of three distinct points
theorem at_most_three_lines (p1 p2 p3 : Point) 
  (h1 : p1 ≠ p2) (h2 : p2 ≠ p3) (h3 : p1 ≠ p3) : 
  ∃ (l1 l2 l3 : Line), ∀ (l : Line), 
    (isPointOnLine p1 l ∧ isPointOnLine p2 l) ∨
    (isPointOnLine p2 l ∧ isPointOnLine p3 l) ∨
    (isPointOnLine p1 l ∧ isPointOnLine p3 l) →
    l = l1 ∨ l = l2 ∨ l = l3 :=
sorry

-- Theorem 2: Only one line can be drawn through two distinct points
theorem unique_line_through_two_points (p1 p2 : Point) (h : p1 ≠ p2) :
  ∃! (l : Line), isPointOnLine p1 l ∧ isPointOnLine p2 l :=
sorry

end NUMINAMATH_CALUDE_at_most_three_lines_unique_line_through_two_points_l1095_109534


namespace NUMINAMATH_CALUDE_orange_boxes_l1095_109553

theorem orange_boxes (total_oranges : ℕ) (oranges_per_box : ℕ) (h1 : total_oranges = 35) (h2 : oranges_per_box = 5) :
  total_oranges / oranges_per_box = 7 :=
by
  sorry

end NUMINAMATH_CALUDE_orange_boxes_l1095_109553


namespace NUMINAMATH_CALUDE_land_plot_area_land_plot_area_is_1267200_l1095_109571

/-- Calculates the total area of a land plot in acres given the dimensions in cm and conversion factors. -/
theorem land_plot_area 
  (triangle_base : ℝ) 
  (triangle_height : ℝ) 
  (rect_length : ℝ) 
  (rect_width : ℝ) 
  (scale_cm_to_miles : ℝ) 
  (acres_per_sq_mile : ℝ) : ℝ :=
  let triangle_area := (1/2) * triangle_base * triangle_height
  let rect_area := rect_length * rect_width
  let total_area_cm2 := triangle_area + rect_area
  let total_area_miles2 := total_area_cm2 * (scale_cm_to_miles^2)
  let total_area_acres := total_area_miles2 * acres_per_sq_mile
  total_area_acres

/-- Proves that the total area of the given land plot is 1267200 acres. -/
theorem land_plot_area_is_1267200 : 
  land_plot_area 20 12 20 5 3 640 = 1267200 := by
  sorry

end NUMINAMATH_CALUDE_land_plot_area_land_plot_area_is_1267200_l1095_109571


namespace NUMINAMATH_CALUDE_team_b_score_l1095_109590

/-- Given a trivia game where:
  * Team A scored 2 points
  * Team C scored 4 points
  * The total points scored by all teams is 15
  Prove that Team B scored 9 points -/
theorem team_b_score (team_a_score team_c_score total_score : ℕ)
  (h1 : team_a_score = 2)
  (h2 : team_c_score = 4)
  (h3 : total_score = 15) :
  total_score - (team_a_score + team_c_score) = 9 := by
  sorry

end NUMINAMATH_CALUDE_team_b_score_l1095_109590


namespace NUMINAMATH_CALUDE_gardening_club_membership_l1095_109519

theorem gardening_club_membership (initial_total : ℕ) 
  (h1 : initial_total > 0)
  (h2 : (60 : ℚ) / 100 * initial_total = (initial_total * 3) / 5) 
  (h3 : (((initial_total * 3) / 5 - 3 : ℚ) / initial_total) = 1 / 2) : 
  (initial_total * 3) / 5 = 18 := by
sorry

end NUMINAMATH_CALUDE_gardening_club_membership_l1095_109519


namespace NUMINAMATH_CALUDE_equal_roots_cubic_polynomial_l1095_109589

theorem equal_roots_cubic_polynomial (m : ℝ) :
  (∃ a b : ℝ, (3 * a^3 + 9 * a^2 - 135 * a + m = 0) ∧
              (3 * b^3 + 9 * b^2 - 135 * b + m = 0) ∧
              (a ≠ b)) ∧
  (∃ x : ℝ, (3 * x^3 + 9 * x^2 - 135 * x + m = 0) ∧
            (∃ y : ℝ, y ≠ x ∧ 3 * y^3 + 9 * y^2 - 135 * y + m = 0 ∧
                      (∀ z : ℝ, 3 * z^3 + 9 * z^2 - 135 * z + m = 0 → z = x ∨ z = y))) ∧
  m > 0 →
  m = 22275 :=
by sorry

end NUMINAMATH_CALUDE_equal_roots_cubic_polynomial_l1095_109589


namespace NUMINAMATH_CALUDE_range_of_f_l1095_109512

/-- The function f(c) defined as (c-a)(c-b) -/
def f (c a b : ℝ) : ℝ := (c - a) * (c - b)

/-- Theorem stating the range of f(c) -/
theorem range_of_f :
  ∀ c a b : ℝ,
  a + b = 1 - c →
  c ≥ 0 →
  a ≥ 0 →
  b ≥ 0 →
  ∃ y : ℝ, f c a b = y ∧ -1/8 ≤ y ∧ y ≤ 1 :=
by sorry

end NUMINAMATH_CALUDE_range_of_f_l1095_109512


namespace NUMINAMATH_CALUDE_rectangular_grid_toothpicks_l1095_109586

/-- Calculates the number of toothpicks in a rectangular grid. -/
def toothpick_count (height : ℕ) (width : ℕ) : ℕ :=
  (height + 1) * width + (width + 1) * height

/-- Theorem: A rectangular grid of toothpicks that is 20 high and 10 wide uses 430 toothpicks. -/
theorem rectangular_grid_toothpicks : toothpick_count 20 10 = 430 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_grid_toothpicks_l1095_109586


namespace NUMINAMATH_CALUDE_line_sum_m_b_l1095_109570

/-- A line passing through points (-2, 0) and (0, 2) can be represented by y = mx + b -/
def line_equation (x y m b : ℝ) : Prop := y = m * x + b

/-- The line passes through (-2, 0) -/
def point1_condition (m b : ℝ) : Prop := line_equation (-2) 0 m b

/-- The line passes through (0, 2) -/
def point2_condition (m b : ℝ) : Prop := line_equation 0 2 m b

/-- Theorem: For a line passing through (-2, 0) and (0, 2), represented by y = mx + b, m + b = 3 -/
theorem line_sum_m_b : ∀ m b : ℝ, point1_condition m b → point2_condition m b → m + b = 3 := by
  sorry

end NUMINAMATH_CALUDE_line_sum_m_b_l1095_109570


namespace NUMINAMATH_CALUDE_inequality_proof_l1095_109584

theorem inequality_proof (x : ℝ) (h : x ≠ 2) : x^2 / (x - 2)^2 ≥ 0 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1095_109584


namespace NUMINAMATH_CALUDE_salt_water_fraction_l1095_109529

theorem salt_water_fraction (small_capacity large_capacity : ℝ) 
  (h1 : large_capacity = 5 * small_capacity)
  (h2 : 0.3 * large_capacity = 0.2 * large_capacity + small_capacity * x) : x = 1/2 := by
  sorry

#check salt_water_fraction

end NUMINAMATH_CALUDE_salt_water_fraction_l1095_109529


namespace NUMINAMATH_CALUDE_diameter_of_circumscribing_circle_l1095_109566

/-- The diameter of a circle circumscribing six smaller tangent circles -/
theorem diameter_of_circumscribing_circle (r : ℝ) (h : r = 5) :
  let small_circle_radius : ℝ := r
  let small_circle_count : ℕ := 6
  let inner_hexagon_side : ℝ := 2 * small_circle_radius
  let inner_hexagon_radius : ℝ := inner_hexagon_side
  let large_circle_radius : ℝ := inner_hexagon_radius + small_circle_radius
  large_circle_radius * 2 = 30 := by sorry

end NUMINAMATH_CALUDE_diameter_of_circumscribing_circle_l1095_109566


namespace NUMINAMATH_CALUDE_top_card_is_eleven_l1095_109599

/-- Represents a card in the array -/
structure Card where
  row : Fin 3
  col : Fin 6
  number : Fin 18

/-- Represents the initial 3x6 array of cards -/
def initial_array : Array (Array Card) := sorry

/-- Folds the left third over the middle third -/
def fold_left_third (arr : Array (Array Card)) : Array (Array Card) := sorry

/-- Folds the right third over the overlapped left and middle thirds -/
def fold_right_third (arr : Array (Array Card)) : Array (Array Card) := sorry

/-- Folds the bottom half over the top half -/
def fold_bottom_half (arr : Array (Array Card)) : Array (Array Card) := sorry

/-- Performs all folding operations -/
def perform_all_folds (arr : Array (Array Card)) : Array (Array Card) :=
  arr |> fold_left_third |> fold_right_third |> fold_bottom_half

/-- The top card after all folds -/
def top_card (arr : Array (Array Card)) : Card := sorry

theorem top_card_is_eleven :
  (top_card (perform_all_folds initial_array)).number = 11 := by
  sorry

end NUMINAMATH_CALUDE_top_card_is_eleven_l1095_109599


namespace NUMINAMATH_CALUDE_largest_integer_K_l1095_109544

theorem largest_integer_K : ∃ K : ℕ, (∀ n : ℕ, n^200 < 5^300 → n ≤ K) ∧ K^200 < 5^300 ∧ K = 11 := by
  sorry

end NUMINAMATH_CALUDE_largest_integer_K_l1095_109544


namespace NUMINAMATH_CALUDE_a_minus_b_equals_plus_minus_eight_l1095_109509

theorem a_minus_b_equals_plus_minus_eight (a b : ℚ) : 
  (|a| = 5) → (|b| = 3) → (a * b < 0) → (a - b = 8 ∨ a - b = -8) := by
  sorry

end NUMINAMATH_CALUDE_a_minus_b_equals_plus_minus_eight_l1095_109509


namespace NUMINAMATH_CALUDE_magic_square_y_value_l1095_109565

/-- Represents a 3x3 magic square -/
def MagicSquare (a b c d e f g h i : ℚ) : Prop :=
  a + b + c = d + e + f ∧
  d + e + f = g + h + i ∧
  a + d + g = b + e + h ∧
  b + e + h = c + f + i ∧
  a + e + i = c + e + g

theorem magic_square_y_value :
  ∀ (y a b c d e : ℚ),
  MagicSquare y 7 24 8 a b c d e →
  y = 39.5 := by
  sorry

end NUMINAMATH_CALUDE_magic_square_y_value_l1095_109565


namespace NUMINAMATH_CALUDE_complex_fraction_third_quadrant_l1095_109516

/-- Given a complex fraction equal to 2-i, prove the resulting point is in the third quadrant -/
theorem complex_fraction_third_quadrant (a b : ℝ) : 
  (a + Complex.I) / (b - Complex.I) = 2 - Complex.I → 
  a < 0 ∧ b < 0 := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_third_quadrant_l1095_109516


namespace NUMINAMATH_CALUDE_girls_pass_percentage_l1095_109568

theorem girls_pass_percentage 
  (total_boys : ℕ) 
  (total_girls : ℕ) 
  (boys_pass_rate : ℚ) 
  (total_fail_rate : ℚ) :
  total_boys = 50 →
  total_girls = 100 →
  boys_pass_rate = 1/2 →
  total_fail_rate = 5667/10000 →
  (total_girls - (((total_boys + total_girls) * total_fail_rate).floor - (total_boys * (1 - boys_pass_rate)).floor)) / total_girls = 2/5 := by
  sorry

end NUMINAMATH_CALUDE_girls_pass_percentage_l1095_109568


namespace NUMINAMATH_CALUDE_book_store_problem_l1095_109549

/-- Represents the purchase and sale of books in a stationery store -/
structure BookStore where
  costA : ℝ  -- Cost price of type A book
  costB : ℝ  -- Cost price of type B book
  sellA : ℝ  -- Selling price of type A book
  sellB : ℝ  -- Selling price of type B book
  totalCost : ℝ  -- Total purchase cost
  profit : ℝ  -- Total profit from first sale

/-- Represents the second purchase and sale scenario -/
structure SecondPurchase where
  minSellA : ℝ  -- Minimum selling price for type A in second purchase
  minProfit : ℝ  -- Minimum required profit for second sale

/-- Theorem stating the solution to the book store problem -/
theorem book_store_problem (store : BookStore) (second : SecondPurchase) 
  (h1 : store.costA = 12)
  (h2 : store.costB = 10)
  (h3 : store.sellA = 15)
  (h4 : store.sellB = 12)
  (h5 : store.totalCost = 1200)
  (h6 : store.profit = 270)
  (h7 : second.minProfit = 340) :
  ∃ (x y : ℕ), 
    (x : ℝ) * store.costA + (y : ℝ) * store.costB = store.totalCost ∧ 
    (x : ℝ) * (store.sellA - store.costA) + (y : ℝ) * (store.sellB - store.costB) = store.profit ∧
    x = 50 ∧ 
    y = 60 ∧
    second.minSellA = 14 ∧
    (50 : ℝ) * (second.minSellA - store.costA) + 2 * (60 : ℝ) * (store.sellB - store.costB) ≥ second.minProfit := by
  sorry


end NUMINAMATH_CALUDE_book_store_problem_l1095_109549


namespace NUMINAMATH_CALUDE_max_value_x_plus_2y_l1095_109501

theorem max_value_x_plus_2y (x y : ℝ) (h : x^2 - 2*x + 4*y = 5) : 
  (∃ (z : ℝ), x + 2*y ≤ z) ∧ (∀ (w : ℝ), x + 2*y ≤ w → 9/2 ≤ w) :=
by sorry

end NUMINAMATH_CALUDE_max_value_x_plus_2y_l1095_109501


namespace NUMINAMATH_CALUDE_circle_radius_from_area_circumference_ratio_l1095_109514

/-- Given a circle with area Q and circumference P, if Q/P = 10, then the radius is 20 -/
theorem circle_radius_from_area_circumference_ratio (Q P : ℝ) (hQ : Q > 0) (hP : P > 0) :
  Q / P = 10 → ∃ (r : ℝ), r > 0 ∧ Q = π * r^2 ∧ P = 2 * π * r ∧ r = 20 := by
  sorry

end NUMINAMATH_CALUDE_circle_radius_from_area_circumference_ratio_l1095_109514


namespace NUMINAMATH_CALUDE_simplify_sqrt_sum_l1095_109576

theorem simplify_sqrt_sum : 2 * Real.sqrt 8 + 3 * Real.sqrt 32 = 16 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_simplify_sqrt_sum_l1095_109576


namespace NUMINAMATH_CALUDE_rectangle_square_overlap_l1095_109583

theorem rectangle_square_overlap (s w h : ℝ) 
  (h1 : 0.4 * s^2 = 0.25 * w * h) 
  (h2 : w = 4 * h) : 
  w / h = 4 := by
sorry

end NUMINAMATH_CALUDE_rectangle_square_overlap_l1095_109583


namespace NUMINAMATH_CALUDE_negation_of_all_geq_two_l1095_109569

theorem negation_of_all_geq_two :
  (¬ (∀ x : ℝ, x ≥ 2)) ↔ (∃ x₀ : ℝ, x₀ < 2) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_all_geq_two_l1095_109569


namespace NUMINAMATH_CALUDE_factorization_1_factorization_2_l1095_109579

-- Factorization of 4x^2 - 16
theorem factorization_1 (x : ℝ) : 4 * x^2 - 16 = 4 * (x - 2) * (x + 2) := by sorry

-- Factorization of a^2b - 4ab + 4b
theorem factorization_2 (a b : ℝ) : a^2 * b - 4 * a * b + 4 * b = b * (a - 2)^2 := by sorry

end NUMINAMATH_CALUDE_factorization_1_factorization_2_l1095_109579


namespace NUMINAMATH_CALUDE_sara_peaches_total_l1095_109573

/-- The total number of peaches Sara picked -/
def total_peaches (initial_peaches additional_peaches : Float) : Float :=
  initial_peaches + additional_peaches

/-- Theorem stating that Sara picked 85.0 peaches in total -/
theorem sara_peaches_total :
  let initial_peaches : Float := 61.0
  let additional_peaches : Float := 24.0
  total_peaches initial_peaches additional_peaches = 85.0 := by
  sorry

end NUMINAMATH_CALUDE_sara_peaches_total_l1095_109573


namespace NUMINAMATH_CALUDE_max_M_value_l1095_109572

theorem max_M_value (x y z u : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (hu : u > 0)
  (eq1 : x - 2*y = z - 2*u) (eq2 : 2*y*z = u*x) (hzy : z ≥ y) :
  ∃ (M : ℝ), M > 0 ∧ M ≤ z/y ∧ ∀ (N : ℝ), (N > 0 ∧ N ≤ z/y) → N ≤ 6 + 4*Real.sqrt 2 :=
sorry

end NUMINAMATH_CALUDE_max_M_value_l1095_109572


namespace NUMINAMATH_CALUDE_reflect_point_across_x_axis_l1095_109503

/-- Represents a point in a 2D Cartesian coordinate system -/
structure Point where
  x : ℝ
  y : ℝ

/-- Reflects a point across the x-axis -/
def reflectAcrossXAxis (p : Point) : Point :=
  ⟨p.x, -p.y⟩

theorem reflect_point_across_x_axis :
  let P : Point := ⟨2, -3⟩
  let P' : Point := reflectAcrossXAxis P
  P'.x = 2 ∧ P'.y = 3 := by
  sorry

end NUMINAMATH_CALUDE_reflect_point_across_x_axis_l1095_109503


namespace NUMINAMATH_CALUDE_intersection_M_N_l1095_109567

def M : Set ℝ := {x | x^2 = x}
def N : Set ℝ := {-1, 0, 1}

theorem intersection_M_N : M ∩ N = {0, 1} := by sorry

end NUMINAMATH_CALUDE_intersection_M_N_l1095_109567


namespace NUMINAMATH_CALUDE_shortest_side_length_l1095_109522

/-- A triangle with an inscribed circle -/
structure InscribedCircleTriangle where
  /-- The radius of the inscribed circle -/
  r : ℝ
  /-- The length of the first segment of the divided side -/
  a : ℝ
  /-- The length of the second segment of the divided side -/
  b : ℝ
  /-- The length of the shortest side of the triangle -/
  shortest_side : ℝ
  /-- Condition: The radius is positive -/
  r_pos : r > 0
  /-- Condition: The segments are positive -/
  a_pos : a > 0
  b_pos : b > 0
  /-- Condition: The shortest side is positive -/
  shortest_side_pos : shortest_side > 0

/-- Theorem: In a triangle with an inscribed circle of radius 5 units, 
    where one side is divided into segments of 9 and 5 units by the point of tangency, 
    the length of the shortest side is 16 units. -/
theorem shortest_side_length (t : InscribedCircleTriangle) 
    (h1 : t.r = 5)
    (h2 : t.a = 9)
    (h3 : t.b = 5) : 
  t.shortest_side = 16 := by
  sorry

end NUMINAMATH_CALUDE_shortest_side_length_l1095_109522


namespace NUMINAMATH_CALUDE_RS_length_value_l1095_109532

/-- Triangle ABC with given side lengths and angle bisectors -/
structure TriangleABC where
  -- Side lengths
  AB : ℝ
  BC : ℝ
  CA : ℝ
  -- Altitude AD
  AD : ℝ
  -- Points R and S on AD
  AR : ℝ
  AS : ℝ
  -- Conditions
  side_lengths : AB = 11 ∧ BC = 13 ∧ CA = 14
  altitude : AD > 0
  R_on_AD : 0 < AR ∧ AR < AD
  S_on_AD : 0 < AS ∧ AS < AD
  BE_bisector : AR / (AD - AR) = CA / BC
  CF_bisector : AS / (AD - AS) = AB / BC

/-- The length of RS in the given triangle -/
def RS_length (t : TriangleABC) : ℝ := t.AR - t.AS

/-- Theorem stating that RS length is equal to 645√95 / 4551 -/
theorem RS_length_value (t : TriangleABC) : RS_length t = 645 * Real.sqrt 95 / 4551 := by
  sorry

end NUMINAMATH_CALUDE_RS_length_value_l1095_109532


namespace NUMINAMATH_CALUDE_frequency_converges_to_probability_l1095_109531

-- Define a random event
def RandomEvent : Type := Unit

-- Define the probability of the event
def probability (e : RandomEvent) : ℝ := sorry

-- Define the observed frequency of the event after n experiments
def observedFrequency (e : RandomEvent) (n : ℕ) : ℝ := sorry

-- Statement: As the number of experiments increases, the frequency of the random event
-- will gradually stabilize at the probability of the random event occurring.
theorem frequency_converges_to_probability (e : RandomEvent) :
  ∀ ε > 0, ∃ N : ℕ, ∀ n ≥ N, |observedFrequency e n - probability e| < ε :=
sorry

end NUMINAMATH_CALUDE_frequency_converges_to_probability_l1095_109531


namespace NUMINAMATH_CALUDE_min_rb_selling_price_theorem_l1095_109528

/-- Represents the fruit sales problem -/
structure FruitSales where
  total_weight : ℝ
  total_cost : ℝ
  rb_purchase_price : ℝ
  rb_selling_price_last_week : ℝ
  xg_purchase_price : ℝ
  xg_selling_price : ℝ
  rb_damage_rate : ℝ

/-- Calculates the profit from last week's sales -/
def last_week_profit (fs : FruitSales) : ℝ := sorry

/-- Calculates the minimum selling price for Red Beauty this week -/
def min_rb_selling_price_this_week (fs : FruitSales) : ℝ := sorry

/-- Theorem stating the minimum selling price of Red Beauty this week -/
theorem min_rb_selling_price_theorem (fs : FruitSales) 
  (h1 : fs.total_weight = 300)
  (h2 : fs.total_cost = 3000)
  (h3 : fs.rb_purchase_price = 20)
  (h4 : fs.rb_selling_price_last_week = 35)
  (h5 : fs.xg_purchase_price = 5)
  (h6 : fs.xg_selling_price = 10)
  (h7 : fs.rb_damage_rate = 0.1)
  : min_rb_selling_price_this_week fs ≥ 36.7 ∧ 
    last_week_profit fs = 2500 := by sorry

end NUMINAMATH_CALUDE_min_rb_selling_price_theorem_l1095_109528


namespace NUMINAMATH_CALUDE_max_min_difference_z_l1095_109526

theorem max_min_difference_z (x y z : ℝ) 
  (sum_eq : x + y + z = 6)
  (sum_squares_eq : x^2 + y^2 + z^2 = 24) :
  ∃ (z_max z_min : ℝ),
    (∀ w, w = x ∨ w = y ∨ w = z → z_min ≤ w ∧ w ≤ z_max) ∧
    z_max - z_min = 4 :=
sorry

end NUMINAMATH_CALUDE_max_min_difference_z_l1095_109526


namespace NUMINAMATH_CALUDE_starting_lineup_combinations_l1095_109525

def team_size : ℕ := 15
def lineup_size : ℕ := 6
def guaranteed_players : ℕ := 2

theorem starting_lineup_combinations :
  Nat.choose (team_size - guaranteed_players) (lineup_size - guaranteed_players) = 715 := by
  sorry

end NUMINAMATH_CALUDE_starting_lineup_combinations_l1095_109525


namespace NUMINAMATH_CALUDE_cricket_average_increase_l1095_109562

def increase_average (current_innings : ℕ) (current_average : ℚ) (next_innings_runs : ℕ) : ℚ :=
  let total_runs := current_innings * current_average
  let new_total_runs := total_runs + next_innings_runs
  let new_average := new_total_runs / (current_innings + 1)
  new_average - current_average

theorem cricket_average_increase :
  increase_average 12 48 178 = 10 := by sorry

end NUMINAMATH_CALUDE_cricket_average_increase_l1095_109562


namespace NUMINAMATH_CALUDE_rectangular_prism_volume_l1095_109513

/-- The volume of a rectangular prism with length:width:height ratio of 4:3:1 and height √2 cm is 24√2 cm³ -/
theorem rectangular_prism_volume (height : ℝ) (length width : ℝ) : 
  height = Real.sqrt 2 →
  length = 4 * height →
  width = 3 * height →
  length * width * height = 24 * Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_rectangular_prism_volume_l1095_109513


namespace NUMINAMATH_CALUDE_perspective_right_angle_l1095_109559

-- Define the types for points and triangles
def Point : Type := ℝ × ℝ
def Triangle : Type := Point × Point × Point

-- Define the perspective transformation
def perspective_transform : Triangle → Triangle := sorry

-- Define the property of being horizontally placed
def is_horizontal (t : Triangle) : Prop := sorry

-- Define the property of a line being parallel to y' axis
def parallel_to_y_axis (p q : Point) : Prop := sorry

-- Define the property of a line being on x' axis
def on_x_axis (p q : Point) : Prop := sorry

-- Define the property of the angle formed by x'o'y' being 45°
def x_o_y_angle_45 (t : Triangle) : Prop := sorry

-- Define a right-angled triangle
def is_right_angled (t : Triangle) : Prop := sorry

-- The main theorem
theorem perspective_right_angle 
  (abc : Triangle) 
  (a'b'c' : Triangle) 
  (h1 : is_horizontal abc)
  (h2 : a'b'c' = perspective_transform abc)
  (h3 : parallel_to_y_axis a'b'c'.1 a'b'c'.2.1)
  (h4 : on_x_axis a'b'c'.2.1 a'b'c'.2.2)
  (h5 : x_o_y_angle_45 a'b'c') :
  is_right_angled abc :=
sorry

end NUMINAMATH_CALUDE_perspective_right_angle_l1095_109559


namespace NUMINAMATH_CALUDE_article_price_proof_l1095_109551

-- Define the selling price and loss percentage
def selling_price : ℝ := 800
def loss_percentage : ℝ := 33.33333333333333

-- Define the original price
def original_price : ℝ := 1200

-- Theorem statement
theorem article_price_proof :
  (selling_price = (1 - loss_percentage / 100) * original_price) → 
  (original_price = 1200) := by
  sorry

end NUMINAMATH_CALUDE_article_price_proof_l1095_109551


namespace NUMINAMATH_CALUDE_solution_to_equation_l1095_109548

theorem solution_to_equation : ∃! (x : ℝ), x ≠ 0 ∧ (9*x)^18 - (18*x)^9 = 0 ∧ x = 2 := by
  sorry

end NUMINAMATH_CALUDE_solution_to_equation_l1095_109548


namespace NUMINAMATH_CALUDE_modified_cube_edge_count_l1095_109541

/-- Represents a modified cube with smaller cubes removed from its corners -/
structure ModifiedCube where
  originalSideLength : ℕ
  removedCubeSideLength : ℕ

/-- Calculates the number of edges in a modified cube -/
def edgeCount (cube : ModifiedCube) : ℕ :=
  sorry

/-- Theorem stating that a cube of side length 4 with unit cubes removed from corners has 48 edges -/
theorem modified_cube_edge_count :
  let cube : ModifiedCube := ⟨4, 1⟩
  edgeCount cube = 48 := by
  sorry

end NUMINAMATH_CALUDE_modified_cube_edge_count_l1095_109541


namespace NUMINAMATH_CALUDE_absolute_value_equation_solution_l1095_109530

theorem absolute_value_equation_solution :
  ∀ x : ℝ, |2005 * x - 2005| = 2005 ↔ x = 2 ∨ x = 0 := by sorry

end NUMINAMATH_CALUDE_absolute_value_equation_solution_l1095_109530


namespace NUMINAMATH_CALUDE_gcd_45123_31207_l1095_109508

theorem gcd_45123_31207 : Nat.gcd 45123 31207 = 1 := by
  sorry

end NUMINAMATH_CALUDE_gcd_45123_31207_l1095_109508


namespace NUMINAMATH_CALUDE_division_problem_l1095_109536

theorem division_problem (A : ℕ) : 
  (A / 6 = 3) ∧ (A % 6 = 2) → A = 20 := by
  sorry

end NUMINAMATH_CALUDE_division_problem_l1095_109536


namespace NUMINAMATH_CALUDE_equation_solution_l1095_109575

theorem equation_solution :
  ∀ x : ℝ, (∀ y : ℝ, 10 * x * y - 15 * y + 3 * x - 4.5 = 0) ↔ x = 3/2 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1095_109575


namespace NUMINAMATH_CALUDE_integral_x_cubed_minus_reciprocal_x_fourth_l1095_109592

theorem integral_x_cubed_minus_reciprocal_x_fourth (f : ℝ → ℝ) :
  (∀ x, f x = x^3 - 1/x^4) →
  ∫ x in (-1)..1, f x = 2/3 := by sorry

end NUMINAMATH_CALUDE_integral_x_cubed_minus_reciprocal_x_fourth_l1095_109592


namespace NUMINAMATH_CALUDE_passengers_scientific_notation_l1095_109539

/-- Represents the number of passengers in millions -/
def passengers : ℝ := 1.446

/-- Represents the scientific notation of the number of passengers -/
def scientific_notation : ℝ := 1.446 * (10 ^ 6)

/-- Theorem stating that the number of passengers in millions 
    is equal to its scientific notation representation -/
theorem passengers_scientific_notation : 
  passengers * 1000000 = scientific_notation := by sorry

end NUMINAMATH_CALUDE_passengers_scientific_notation_l1095_109539


namespace NUMINAMATH_CALUDE_product_equality_l1095_109535

theorem product_equality (a b c d e f : ℝ) 
  (h1 : a * b * c = 65)
  (h2 : b * c * d = 65)
  (h3 : c * d * e = 1000)
  (h4 : (a * f) / (c * d) = 0.25) :
  d * e * f = 250 := by
sorry

end NUMINAMATH_CALUDE_product_equality_l1095_109535
