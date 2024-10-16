import Mathlib

namespace NUMINAMATH_CALUDE_solution_set_equivalence_l3983_398345

-- Define the set of real numbers greater than 1
def greater_than_one : Set ℝ := {x | x > 1}

-- Define the solution set of ax - 1 > 0
def solution_set_linear (a : ℝ) : Set ℝ := {x | a * x - 1 > 0}

-- Define the solution set of (ax - 1)(x + 2) ≥ 0
def solution_set_quadratic (a : ℝ) : Set ℝ := {x | (a * x - 1) * (x + 2) ≥ 0}

-- Define the set (-∞, -2] ∪ [1, +∞)
def target_set : Set ℝ := {x | x ≤ -2 ∨ x ≥ 1}

theorem solution_set_equivalence (a : ℝ) : 
  solution_set_linear a = greater_than_one → solution_set_quadratic a = target_set := by
  sorry

end NUMINAMATH_CALUDE_solution_set_equivalence_l3983_398345


namespace NUMINAMATH_CALUDE_pride_and_prejudice_watching_time_l3983_398330

/-- Calculates the total hours spent watching a TV series given the number of episodes and minutes per episode -/
def total_watching_hours (num_episodes : ℕ) (minutes_per_episode : ℕ) : ℚ :=
  (num_episodes * minutes_per_episode : ℚ) / 60

/-- Proves that watching 6 episodes of 50 minutes each takes 5 hours -/
theorem pride_and_prejudice_watching_time :
  total_watching_hours 6 50 = 5 := by
  sorry

#eval total_watching_hours 6 50

end NUMINAMATH_CALUDE_pride_and_prejudice_watching_time_l3983_398330


namespace NUMINAMATH_CALUDE_geometric_sequence_property_l3983_398363

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

theorem geometric_sequence_property 
  (a : ℕ → ℝ) (h : geometric_sequence a) (h5 : a 5 = 4) : 
  a 2 * a 8 = 16 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_property_l3983_398363


namespace NUMINAMATH_CALUDE_part_one_part_two_l3983_398329

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := |x - a| + 5 * x

-- Part I
theorem part_one :
  let a : ℝ := -1
  {x : ℝ | f a x ≤ 5 * x + 3} = Set.Icc (-4) 2 := by sorry

-- Part II
theorem part_two :
  {a : ℝ | ∀ x ≥ -1, f a x ≥ 0} = {a : ℝ | a ≥ 4 ∨ a ≤ -6} := by sorry

end NUMINAMATH_CALUDE_part_one_part_two_l3983_398329


namespace NUMINAMATH_CALUDE_cleaning_event_calculation_l3983_398385

def total_members : ℕ := 2000
def adult_men_percentage : ℚ := 30 / 100
def senior_percentage : ℚ := 5 / 100
def child_teen_ratio : ℚ := 3 / 2
def child_collection_rate : ℚ := 3 / 2
def teen_collection_rate : ℕ := 3
def senior_collection_rate : ℕ := 1

theorem cleaning_event_calculation :
  let adult_men := (adult_men_percentage * total_members).floor
  let adult_women := 2 * adult_men
  let seniors := (senior_percentage * total_members).floor
  let children_and_teens := total_members - (adult_men + adult_women + seniors)
  let children := ((child_teen_ratio * children_and_teens) / (1 + child_teen_ratio)).floor
  let teenagers := children_and_teens - children
  ∃ (children teenagers : ℕ) (recyclable mixed various : ℚ),
    children = 60 ∧
    teenagers = 40 ∧
    recyclable = child_collection_rate * children ∧
    mixed = teen_collection_rate * teenagers ∧
    various = senior_collection_rate * seniors ∧
    recyclable = 90 ∧
    mixed = 120 ∧
    various = 100 :=
by
  sorry

end NUMINAMATH_CALUDE_cleaning_event_calculation_l3983_398385


namespace NUMINAMATH_CALUDE_minimum_guests_proof_l3983_398370

-- Define the total food consumed
def total_food : ℝ := 319

-- Define the maximum individual consumption limits
def max_meat : ℝ := 1.5
def max_vegetables : ℝ := 0.3
def max_dessert : ℝ := 0.2

-- Define the consumption ratio
def meat_ratio : ℝ := 3
def vegetables_ratio : ℝ := 1
def dessert_ratio : ℝ := 1

-- Define the minimum number of guests
def min_guests : ℕ := 160

-- Theorem statement
theorem minimum_guests_proof :
  ∃ (guests : ℕ), guests ≥ min_guests ∧
  (guests : ℝ) * (max_meat + max_vegetables + max_dessert) ≥ total_food ∧
  ∀ (g : ℕ), g < guests →
    (g : ℝ) * (max_meat + max_vegetables + max_dessert) < total_food :=
sorry

end NUMINAMATH_CALUDE_minimum_guests_proof_l3983_398370


namespace NUMINAMATH_CALUDE_village_population_problem_l3983_398349

theorem village_population_problem (final_population : ℕ) : 
  final_population = 5265 → ∃ original : ℕ, 
    (original : ℚ) * (9/10) * (3/4) = final_population ∧ original = 7800 :=
by
  sorry

end NUMINAMATH_CALUDE_village_population_problem_l3983_398349


namespace NUMINAMATH_CALUDE_booklet_sheets_l3983_398367

/-- Given a booklet created from folded A4 sheets, prove the number of original sheets. -/
theorem booklet_sheets (n : ℕ) (h : 2 * n + 2 = 74) : n / 4 = 9 := by
  sorry

#check booklet_sheets

end NUMINAMATH_CALUDE_booklet_sheets_l3983_398367


namespace NUMINAMATH_CALUDE_range_of_x_in_negative_sqrt_l3983_398332

theorem range_of_x_in_negative_sqrt (x : ℝ) :
  (3 * x + 5 ≥ 0) ↔ (x ≥ -5/3) :=
by sorry

end NUMINAMATH_CALUDE_range_of_x_in_negative_sqrt_l3983_398332


namespace NUMINAMATH_CALUDE_bianca_tulips_l3983_398359

/-- The number of tulips Bianca picked -/
def tulips : ℕ := sorry

/-- The total number of flowers Bianca picked -/
def total_flowers : ℕ := sorry

/-- The number of roses Bianca picked -/
def roses : ℕ := 49

/-- The number of flowers Bianca used -/
def used_flowers : ℕ := 81

/-- The number of extra flowers -/
def extra_flowers : ℕ := 7

theorem bianca_tulips : 
  tulips = 39 ∧ 
  total_flowers = tulips + roses ∧ 
  total_flowers = used_flowers + extra_flowers :=
sorry

end NUMINAMATH_CALUDE_bianca_tulips_l3983_398359


namespace NUMINAMATH_CALUDE_board_game_investment_l3983_398395

/-- Proves that the investment in equipment for a board game business is $10,410 -/
theorem board_game_investment
  (manufacture_cost : ℝ)
  (selling_price : ℝ)
  (break_even_quantity : ℕ)
  (h1 : manufacture_cost = 2.65)
  (h2 : selling_price = 20)
  (h3 : break_even_quantity = 600)
  (h4 : selling_price * break_even_quantity = 
        manufacture_cost * break_even_quantity + investment) :
  investment = 10410 := by
  sorry

end NUMINAMATH_CALUDE_board_game_investment_l3983_398395


namespace NUMINAMATH_CALUDE_red_peaches_count_l3983_398368

theorem red_peaches_count (total : ℕ) (green : ℕ) (red : ℕ) : 
  total = 16 → green = 3 → total = red + green → red = 13 := by
  sorry

end NUMINAMATH_CALUDE_red_peaches_count_l3983_398368


namespace NUMINAMATH_CALUDE_square_position_2010_l3983_398372

/-- Represents the positions of the square's vertices -/
inductive SquarePosition
| ABCD
| CABD
| DACB
| BCAD
| ADCB
| CBDA
| BADC
| CDAB

/-- Applies the transformation sequence to a given position -/
def applyTransformation (pos : SquarePosition) : SquarePosition :=
  match pos with
  | SquarePosition.ABCD => SquarePosition.CABD
  | SquarePosition.CABD => SquarePosition.DACB
  | SquarePosition.DACB => SquarePosition.BCAD
  | SquarePosition.BCAD => SquarePosition.ADCB
  | SquarePosition.ADCB => SquarePosition.CBDA
  | SquarePosition.CBDA => SquarePosition.BADC
  | SquarePosition.BADC => SquarePosition.CDAB
  | SquarePosition.CDAB => SquarePosition.ABCD

/-- Returns the position after n transformations -/
def nthPosition (n : Nat) : SquarePosition :=
  match n % 8 with
  | 0 => SquarePosition.ABCD
  | 1 => SquarePosition.CABD
  | 2 => SquarePosition.DACB
  | 3 => SquarePosition.BCAD
  | 4 => SquarePosition.ADCB
  | 5 => SquarePosition.CBDA
  | 6 => SquarePosition.BADC
  | 7 => SquarePosition.CDAB
  | _ => SquarePosition.ABCD  -- This case should never occur due to % 8

theorem square_position_2010 :
  nthPosition 2010 = SquarePosition.CABD := by
  sorry

end NUMINAMATH_CALUDE_square_position_2010_l3983_398372


namespace NUMINAMATH_CALUDE_min_distance_parabola_to_line_l3983_398325

/-- The minimum distance from a point on y = x^2 to y = 2x - 2 is √5/5 -/
theorem min_distance_parabola_to_line :
  let f : ℝ → ℝ := λ x => x^2  -- The curve y = x^2
  let g : ℝ → ℝ := λ x => 2*x - 2  -- The line y = 2x - 2
  ∃ (P : ℝ × ℝ), P.2 = f P.1 ∧  -- Point P on the curve
  (∀ (Q : ℝ × ℝ), Q.2 = f Q.1 →  -- For all points Q on the curve
    Real.sqrt 5 / 5 ≤ Real.sqrt ((Q.1 - (Q.2 + 2) / 2)^2 + (Q.2 - g ((Q.2 + 2) / 2))^2)) ∧
  (∃ (P : ℝ × ℝ), P.2 = f P.1 ∧
    Real.sqrt 5 / 5 = Real.sqrt ((P.1 - (P.2 + 2) / 2)^2 + (P.2 - g ((P.2 + 2) / 2))^2)) :=
by
  sorry


end NUMINAMATH_CALUDE_min_distance_parabola_to_line_l3983_398325


namespace NUMINAMATH_CALUDE_min_value_on_line_l3983_398396

theorem min_value_on_line (m n : ℝ) : 
  m + 2 * n = 1 → 2^m + 4^n ≥ 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_min_value_on_line_l3983_398396


namespace NUMINAMATH_CALUDE_brick_wall_problem_l3983_398387

/-- Represents a brick wall with a specific structure -/
structure BrickWall where
  rows : Nat
  total_bricks : Nat
  bottom_row_bricks : Nat
  row_difference : Nat

/-- Calculates the sum of bricks in all rows of the wall -/
def sum_of_bricks (wall : BrickWall) : Nat :=
  wall.rows * wall.bottom_row_bricks - (wall.rows * (wall.rows - 1) * wall.row_difference) / 2

/-- Theorem stating the properties of the specific brick wall -/
theorem brick_wall_problem : ∃ (wall : BrickWall), 
  wall.rows = 5 ∧ 
  wall.total_bricks = 200 ∧ 
  wall.row_difference = 1 ∧
  sum_of_bricks wall = wall.total_bricks ∧
  wall.bottom_row_bricks = 42 := by
  sorry

end NUMINAMATH_CALUDE_brick_wall_problem_l3983_398387


namespace NUMINAMATH_CALUDE_ben_savings_proof_l3983_398343

/-- Represents the number of days that have elapsed -/
def days : ℕ := 7

/-- Ben's daily starting amount in cents -/
def daily_start : ℕ := 5000

/-- Ben's daily spending in cents -/
def daily_spend : ℕ := 1500

/-- Ben's dad's additional contribution in cents -/
def dad_contribution : ℕ := 1000

/-- Ben's final amount in cents -/
def final_amount : ℕ := 50000

theorem ben_savings_proof :
  2 * (days * (daily_start - daily_spend)) + dad_contribution = final_amount := by
  sorry

end NUMINAMATH_CALUDE_ben_savings_proof_l3983_398343


namespace NUMINAMATH_CALUDE_smallest_number_of_students_l3983_398334

/-- Represents the number of students in each grade -/
structure Students where
  ninth : ℕ
  tenth : ℕ
  eleventh : ℕ

/-- The ratios given in the problem -/
def ratio_9_10 : ℚ := 7 / 4
def ratio_9_11 : ℚ := 5 / 3

/-- The proposition that needs to be proved -/
theorem smallest_number_of_students :
  ∃ (s : Students),
    (s.ninth : ℚ) / s.tenth = ratio_9_10 ∧
    (s.ninth : ℚ) / s.eleventh = ratio_9_11 ∧
    s.ninth + s.tenth + s.eleventh = 76 ∧
    (∀ (t : Students),
      (t.ninth : ℚ) / t.tenth = ratio_9_10 →
      (t.ninth : ℚ) / t.eleventh = ratio_9_11 →
      t.ninth + t.tenth + t.eleventh ≥ 76) :=
sorry

end NUMINAMATH_CALUDE_smallest_number_of_students_l3983_398334


namespace NUMINAMATH_CALUDE_min_cuts_3x3x3_cube_l3983_398392

/-- Represents a 3D cube with side length n -/
structure Cube (n : ℕ) where
  side_length : n > 0

/-- Represents a cut along a plane -/
inductive Cut
  | X : ℕ → Cut
  | Y : ℕ → Cut
  | Z : ℕ → Cut

/-- The minimum number of cuts required to divide a cube into unit cubes -/
def min_cuts (c : Cube 3) : ℕ := 6

/-- Theorem stating that the minimum number of cuts to divide a 3x3x3 cube into 27 unit cubes is 6 -/
theorem min_cuts_3x3x3_cube (c : Cube 3) :
  min_cuts c = 6 :=
by sorry

end NUMINAMATH_CALUDE_min_cuts_3x3x3_cube_l3983_398392


namespace NUMINAMATH_CALUDE_absolute_value_equality_implication_not_always_true_l3983_398351

theorem absolute_value_equality_implication_not_always_true :
  ¬ (∀ a b : ℝ, |a| = |b| → a = b) := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_equality_implication_not_always_true_l3983_398351


namespace NUMINAMATH_CALUDE_max_volume_triangular_prism_l3983_398339

/-- Represents a triangular prism with rectangular bases -/
structure TriangularPrism where
  l : ℝ  -- length of the base
  w : ℝ  -- width of the base
  h : ℝ  -- height of the prism

/-- The sum of the areas of two lateral faces and one base is 30 -/
def area_constraint (p : TriangularPrism) : Prop :=
  2 * p.h * p.l + p.l * p.w = 30

/-- The volume of the prism -/
def volume (p : TriangularPrism) : ℝ :=
  p.l * p.w * p.h

/-- Theorem: The maximum volume of the triangular prism is 112.5 -/
theorem max_volume_triangular_prism :
  ∃ (p : TriangularPrism), area_constraint p ∧
    (∀ (q : TriangularPrism), area_constraint q → volume q ≤ volume p) ∧
    volume p = 112.5 :=
sorry

end NUMINAMATH_CALUDE_max_volume_triangular_prism_l3983_398339


namespace NUMINAMATH_CALUDE_shelf_fill_relation_l3983_398333

/-- Represents the number of books needed to fill a shelf. -/
structure ShelfFill :=
  (A H S M F : ℕ)
  (distinct : A ≠ H ∧ A ≠ S ∧ A ≠ M ∧ A ≠ F ∧
              H ≠ S ∧ H ≠ M ∧ H ≠ F ∧
              S ≠ M ∧ S ≠ F ∧
              M ≠ F)
  (positive : A > 0 ∧ H > 0 ∧ S > 0 ∧ M > 0 ∧ F > 0)
  (history_thicker : H < A ∧ M < S)

/-- Theorem stating the relation between the number of books needed to fill the shelf. -/
theorem shelf_fill_relation (sf : ShelfFill) : sf.F = (sf.A * sf.F - sf.S * sf.H) / (sf.M - sf.H) :=
  sorry

end NUMINAMATH_CALUDE_shelf_fill_relation_l3983_398333


namespace NUMINAMATH_CALUDE_total_trip_time_l3983_398307

/-- The total trip time given the specified conditions -/
theorem total_trip_time : ∀ (v : ℝ),
  v > 0 →
  20 / v = 40 →
  80 / (4 * v) + 40 = 80 :=
by
  sorry

end NUMINAMATH_CALUDE_total_trip_time_l3983_398307


namespace NUMINAMATH_CALUDE_average_monthly_income_is_69_l3983_398364

/-- Proves that the average monthly income for a 10-month period is 69 given specific income and expense conditions. -/
theorem average_monthly_income_is_69 
  (X : ℝ) -- Income base for first 6 months
  (Y : ℝ) -- Income for last 4 months
  (h1 : (6 * (1.1 * X) + 4 * Y) / 10 = 69) -- Average income condition
  (h2 : 4 * (Y - 60) - 6 * (70 - 1.1 * X) = 30) -- Debt and savings condition
  : (6 * (1.1 * X) + 4 * Y) / 10 = 69 := by
  sorry

#check average_monthly_income_is_69

end NUMINAMATH_CALUDE_average_monthly_income_is_69_l3983_398364


namespace NUMINAMATH_CALUDE_square_area_from_adjacent_points_l3983_398380

theorem square_area_from_adjacent_points :
  let p1 : ℝ × ℝ := (2, 3)
  let p2 : ℝ × ℝ := (5, 7)
  let distance := Real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2)
  let area := distance^2
  area = 25 := by sorry

end NUMINAMATH_CALUDE_square_area_from_adjacent_points_l3983_398380


namespace NUMINAMATH_CALUDE_gcd_of_256_196_560_l3983_398304

theorem gcd_of_256_196_560 : Nat.gcd 256 (Nat.gcd 196 560) = 28 := by sorry

end NUMINAMATH_CALUDE_gcd_of_256_196_560_l3983_398304


namespace NUMINAMATH_CALUDE_range_of_m_l3983_398322

theorem range_of_m : 
  (∀ x : ℝ, ∃ m : ℝ, 4^x - 2^(x+1) + m = 0) → 
  (∀ m : ℝ, (∃ x : ℝ, 4^x - 2^(x+1) + m = 0) → m ≤ 1) :=
by sorry

end NUMINAMATH_CALUDE_range_of_m_l3983_398322


namespace NUMINAMATH_CALUDE_gain_percent_calculation_l3983_398374

theorem gain_percent_calculation (cost_price selling_price : ℝ) 
  (h1 : cost_price = 10)
  (h2 : selling_price = 15) : 
  (selling_price - cost_price) / cost_price * 100 = 50 := by
  sorry

end NUMINAMATH_CALUDE_gain_percent_calculation_l3983_398374


namespace NUMINAMATH_CALUDE_square_difference_division_problem_solution_l3983_398358

theorem square_difference_division (a b : ℕ) (h : a > b) :
  (a^2 - b^2) / (a - b) = a + b :=
by sorry

theorem problem_solution :
  (275^2 - 245^2) / 30 = 520 :=
by sorry

end NUMINAMATH_CALUDE_square_difference_division_problem_solution_l3983_398358


namespace NUMINAMATH_CALUDE_average_speed_problem_l3983_398327

/-- Given a distance of 1800 meters and a time of 30 minutes, 
    prove that the average speed is 1 meter per second. -/
theorem average_speed_problem (distance : ℝ) (time_minutes : ℝ) :
  distance = 1800 ∧ time_minutes = 30 →
  (distance / (time_minutes * 60)) = 1 := by
sorry

end NUMINAMATH_CALUDE_average_speed_problem_l3983_398327


namespace NUMINAMATH_CALUDE_ninety_seventh_rising_number_l3983_398316

/-- A rising number is a positive integer where each digit is larger than each of the digits to its left. -/
def IsRisingNumber (n : ℕ) : Prop := sorry

/-- The total count of five-digit rising numbers. -/
def TotalFiveDigitRisingNumbers : ℕ := 126

/-- The nth five-digit rising number when arranged from smallest to largest. -/
def NthFiveDigitRisingNumber (n : ℕ) : ℕ := sorry

theorem ninety_seventh_rising_number :
  NthFiveDigitRisingNumber 97 = 24678 := by sorry

end NUMINAMATH_CALUDE_ninety_seventh_rising_number_l3983_398316


namespace NUMINAMATH_CALUDE_mushroom_remainder_l3983_398366

theorem mushroom_remainder (initial : ℕ) (consumed : ℕ) (remaining : ℕ) : 
  initial = 15 → consumed = 8 → remaining = initial - consumed → remaining = 7 := by
  sorry

end NUMINAMATH_CALUDE_mushroom_remainder_l3983_398366


namespace NUMINAMATH_CALUDE_square_root_one_ninth_l3983_398312

theorem square_root_one_ninth : Real.sqrt (1/9) = 1/3 ∨ Real.sqrt (1/9) = -1/3 := by
  sorry

end NUMINAMATH_CALUDE_square_root_one_ninth_l3983_398312


namespace NUMINAMATH_CALUDE_smallest_possible_d_l3983_398337

theorem smallest_possible_d : 
  let f : ℝ → ℝ := λ d => (5 * Real.sqrt 3)^2 + (2 * d + 6)^2 - (4 * d)^2
  ∃ d : ℝ, f d = 0 ∧ ∀ d' : ℝ, f d' = 0 → d ≤ d' ∧ d = 1 + Real.sqrt 41 / 2 :=
by sorry

end NUMINAMATH_CALUDE_smallest_possible_d_l3983_398337


namespace NUMINAMATH_CALUDE_jelly_bean_ratio_l3983_398393

def napoleon_jelly_beans : ℕ := 17
def mikey_jelly_beans : ℕ := 19

def sedrich_jelly_beans : ℕ := napoleon_jelly_beans + 4

def total_jelly_beans : ℕ := napoleon_jelly_beans + sedrich_jelly_beans

theorem jelly_bean_ratio :
  total_jelly_beans = 2 * mikey_jelly_beans :=
by sorry

end NUMINAMATH_CALUDE_jelly_bean_ratio_l3983_398393


namespace NUMINAMATH_CALUDE_largest_triangular_cross_section_area_l3983_398324

/-- The largest possible area of a triangular cross-section in a right circular cone -/
theorem largest_triangular_cross_section_area
  (slant_height : ℝ)
  (base_diameter : ℝ)
  (h_slant : slant_height = 5)
  (h_diameter : base_diameter = 8) :
  ∃ (area : ℝ), area = 12.5 ∧
  ∀ (other_area : ℝ),
    (∃ (a b c : ℝ),
      a ≤ slant_height ∧
      b ≤ slant_height ∧
      c ≤ base_diameter ∧
      other_area = (a * b) / 2) →
    other_area ≤ area :=
by sorry

end NUMINAMATH_CALUDE_largest_triangular_cross_section_area_l3983_398324


namespace NUMINAMATH_CALUDE_runner_speed_ratio_l3983_398321

theorem runner_speed_ratio (u₁ u₂ : ℝ) (h₁ : u₁ > u₂) (h₂ : u₁ + u₂ = 5) (h₃ : u₁ - u₂ = 5 / 3) :
  u₁ / u₂ = 2 := by
sorry

end NUMINAMATH_CALUDE_runner_speed_ratio_l3983_398321


namespace NUMINAMATH_CALUDE_set_relationships_l3983_398305

-- Define set A
def A : Set ℝ := {y | ∃ x, y = x^2 + 1}

-- Define set B
def B : Set (ℝ × ℝ) := {p | p.2 = p.1^2 + 1}

-- Theorem statement
theorem set_relationships :
  (1 ∉ B) ∧ (2 ∈ A) := by
  sorry

end NUMINAMATH_CALUDE_set_relationships_l3983_398305


namespace NUMINAMATH_CALUDE_last_digit_sum_powers_l3983_398389

theorem last_digit_sum_powers : (3^1991 + 1991^3) % 10 = 8 := by sorry

end NUMINAMATH_CALUDE_last_digit_sum_powers_l3983_398389


namespace NUMINAMATH_CALUDE_client_ladder_cost_l3983_398384

/-- The total cost for a set of ladders given the number of ladders, rungs per ladder, and cost per rung -/
def total_cost (num_ladders : ℕ) (rungs_per_ladder : ℕ) (cost_per_rung : ℕ) : ℕ :=
  num_ladders * rungs_per_ladder * cost_per_rung

/-- The theorem stating the total cost for the client's ladder order -/
theorem client_ladder_cost :
  let cost_per_rung := 2
  let cost_first_set := total_cost 10 50 cost_per_rung
  let cost_second_set := total_cost 20 60 cost_per_rung
  cost_first_set + cost_second_set = 3400 := by sorry

end NUMINAMATH_CALUDE_client_ladder_cost_l3983_398384


namespace NUMINAMATH_CALUDE_inheritance_calculation_l3983_398319

theorem inheritance_calculation (x : ℝ) : 
  (0.25 * x + 0.15 * (x - 0.25 * x) = 15000) → x = 41379 := by
  sorry

end NUMINAMATH_CALUDE_inheritance_calculation_l3983_398319


namespace NUMINAMATH_CALUDE_range_of_m_l3983_398338

theorem range_of_m (f g : ℝ → ℝ) (m : ℝ) : 
  (f = λ x => 1 + Real.sin (2 * x)) →
  (g = λ x => 2 * (Real.cos x)^2 + m) →
  (∃ x₀ ∈ Set.Icc 0 (Real.pi / 2), f x₀ ≥ g x₀) →
  m ≤ Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_range_of_m_l3983_398338


namespace NUMINAMATH_CALUDE_sin_double_angle_special_case_l3983_398328

theorem sin_double_angle_special_case (α : ℝ) (h : Real.cos (π / 4 - α) = 3 / 5) : 
  Real.sin (2 * α) = -7 / 25 := by
  sorry

end NUMINAMATH_CALUDE_sin_double_angle_special_case_l3983_398328


namespace NUMINAMATH_CALUDE_unique_solution_implies_a_greater_than_one_l3983_398357

-- Define the function f(x) = 2ax^2 - x - 1
def f (a : ℝ) (x : ℝ) : ℝ := 2 * a * x^2 - x - 1

-- State the theorem
theorem unique_solution_implies_a_greater_than_one :
  ∀ a : ℝ, (∃! x : ℝ, x ∈ (Set.Ioo 0 1) ∧ f a x = 0) → a > 1 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_implies_a_greater_than_one_l3983_398357


namespace NUMINAMATH_CALUDE_divisors_not_div_by_seven_l3983_398317

def number_to_factorize : ℕ := 420

-- Define a function to count divisors not divisible by 7
def count_divisors_not_div_by_seven (n : ℕ) : ℕ := sorry

-- Theorem statement
theorem divisors_not_div_by_seven :
  count_divisors_not_div_by_seven number_to_factorize = 12 := by sorry

end NUMINAMATH_CALUDE_divisors_not_div_by_seven_l3983_398317


namespace NUMINAMATH_CALUDE_modular_inverse_17_mod_1021_l3983_398346

theorem modular_inverse_17_mod_1021 (p : Nat) (prime_p : Nat.Prime p) (h : p = 1021) :
  (17 * 961) % p = 1 :=
by sorry

end NUMINAMATH_CALUDE_modular_inverse_17_mod_1021_l3983_398346


namespace NUMINAMATH_CALUDE_roots_not_real_l3983_398377

-- Define the complex number i
def i : ℂ := Complex.I

-- Define the quadratic equation
def quadratic_equation (z m : ℂ) : Prop :=
  5 * z^2 - 7 * i * z - m = 0

-- State the theorem
theorem roots_not_real (m : ℂ) :
  ∃ (z₁ z₂ : ℂ), quadratic_equation z₁ m ∧ quadratic_equation z₂ m ∧
  z₁ ≠ z₂ ∧ ¬(z₁.im = 0) ∧ ¬(z₂.im = 0) := by
  sorry

end NUMINAMATH_CALUDE_roots_not_real_l3983_398377


namespace NUMINAMATH_CALUDE_a_gt_one_iff_a_gt_zero_l3983_398386

theorem a_gt_one_iff_a_gt_zero {a : ℝ} : a > 1 ↔ a > 0 := by sorry

end NUMINAMATH_CALUDE_a_gt_one_iff_a_gt_zero_l3983_398386


namespace NUMINAMATH_CALUDE_merger_proportion_l3983_398313

/-- Represents the proportion of managers in a company -/
def ManagerProportion := Fin 101 → ℚ

/-- Represents the proportion of employees from one company in a merged company -/
def MergedProportion := Fin 101 → ℚ

theorem merger_proportion 
  (company_a_managers : ManagerProportion)
  (company_b_managers : ManagerProportion)
  (merged_managers : ManagerProportion)
  (h1 : company_a_managers 10 = 1)
  (h2 : company_b_managers 30 = 1)
  (h3 : merged_managers 25 = 1) :
  ∃ (result : MergedProportion), result 25 = 1 :=
sorry

end NUMINAMATH_CALUDE_merger_proportion_l3983_398313


namespace NUMINAMATH_CALUDE_geometric_series_first_term_l3983_398353

theorem geometric_series_first_term
  (a r : ℝ)
  (h_sum : a / (1 - r) = 30)
  (h_sum_squares : a^2 / (1 - r^2) = 120)
  (h_convergent : |r| < 1) :
  a = 120 / 17 := by
  sorry

end NUMINAMATH_CALUDE_geometric_series_first_term_l3983_398353


namespace NUMINAMATH_CALUDE_right_triangle_increase_sides_acute_l3983_398397

theorem right_triangle_increase_sides_acute (a b c x : ℝ) : 
  a > 0 → b > 0 → c > 0 → x > 0 → c^2 = a^2 + b^2 → 
  (a + x)^2 + (b + x)^2 > (c + x)^2 := by
sorry

end NUMINAMATH_CALUDE_right_triangle_increase_sides_acute_l3983_398397


namespace NUMINAMATH_CALUDE_logarithm_sum_simplification_l3983_398355

theorem logarithm_sum_simplification :
  let expr := (1 / (Real.log 3 / Real.log 6 + 1)) + 
              (1 / (Real.log 7 / Real.log 15 + 1)) + 
              (1 / (Real.log 4 / Real.log 12 + 1))
  expr = -Real.log 84 / Real.log 10 := by
  sorry

end NUMINAMATH_CALUDE_logarithm_sum_simplification_l3983_398355


namespace NUMINAMATH_CALUDE_candies_left_theorem_l3983_398309

def candies_left_to_share (initial_candies : ℕ) (siblings : ℕ) (candies_per_sibling : ℕ) (eat_self : ℕ) : ℕ :=
  let remaining_after_siblings := initial_candies - siblings * candies_per_sibling
  let given_to_friend := remaining_after_siblings / 2
  let remaining_after_friend := remaining_after_siblings - given_to_friend
  remaining_after_friend - eat_self

theorem candies_left_theorem : 
  candies_left_to_share 100 3 10 16 = 19 := by
  sorry

end NUMINAMATH_CALUDE_candies_left_theorem_l3983_398309


namespace NUMINAMATH_CALUDE_min_squares_128_343_l3983_398356

/-- Represents a rectangle with height and width -/
structure Rectangle where
  height : ℕ
  width : ℕ

/-- Represents a polyomino spanning a rectangle -/
def SpanningPolyomino (r : Rectangle) : Type := Unit

/-- The number of unit squares in a spanning polyomino -/
def num_squares (r : Rectangle) (p : SpanningPolyomino r) : ℕ := sorry

/-- The minimum number of unit squares in any spanning polyomino for a given rectangle -/
def min_spanning_squares (r : Rectangle) : ℕ := sorry

/-- Theorem: The minimum number of unit squares in a spanning polyomino for a 128-by-343 rectangle is 470 -/
theorem min_squares_128_343 :
  let r : Rectangle := { height := 128, width := 343 }
  min_spanning_squares r = 470 := by sorry

end NUMINAMATH_CALUDE_min_squares_128_343_l3983_398356


namespace NUMINAMATH_CALUDE_dance_pairs_correct_l3983_398394

/-- The number of ways to form dance pairs given specific knowledge constraints -/
def dance_pairs (n : ℕ) (r : ℕ) : ℕ :=
  if r ≤ n then
    Nat.choose n r * (Nat.factorial n / Nat.factorial (n - r))
  else 0

/-- Theorem stating the correct number of dance pairs -/
theorem dance_pairs_correct (n : ℕ) (r : ℕ) (h : r ≤ n) :
  dance_pairs n r = Nat.choose n r * (Nat.factorial n / Nat.factorial (n - r)) :=
by sorry

end NUMINAMATH_CALUDE_dance_pairs_correct_l3983_398394


namespace NUMINAMATH_CALUDE_lauryn_company_men_count_l3983_398365

theorem lauryn_company_men_count :
  ∀ (men women : ℕ),
    men + women = 180 →
    men = women - 20 →
    men = 80 := by
  sorry

end NUMINAMATH_CALUDE_lauryn_company_men_count_l3983_398365


namespace NUMINAMATH_CALUDE_jackies_pushup_count_l3983_398352

/-- Calculates the number of push-ups Jackie can do in one minute with breaks -/
def jackies_pushups (pushups_per_ten_seconds : ℕ) (total_time : ℕ) (break_duration : ℕ) (num_breaks : ℕ) : ℕ :=
  let total_break_time := break_duration * num_breaks
  let pushup_time := total_time - total_break_time
  let pushups_per_second := pushups_per_ten_seconds / 10
  pushup_time * pushups_per_second

/-- Proves that Jackie can do 22 push-ups in one minute with two 8-second breaks -/
theorem jackies_pushup_count : jackies_pushups 5 60 8 2 = 22 := by
  sorry

#eval jackies_pushups 5 60 8 2

end NUMINAMATH_CALUDE_jackies_pushup_count_l3983_398352


namespace NUMINAMATH_CALUDE_special_sale_discount_l3983_398347

theorem special_sale_discount (list_price : ℝ) (regular_discount_min : ℝ) (regular_discount_max : ℝ) (lowest_sale_price_ratio : ℝ) :
  list_price = 80 →
  regular_discount_min = 0.3 →
  regular_discount_max = 0.5 →
  lowest_sale_price_ratio = 0.4 →
  ∃ (additional_discount : ℝ),
    additional_discount = 0.2 ∧
    list_price * (1 - regular_discount_max) * (1 - additional_discount) = list_price * lowest_sale_price_ratio :=
by
  sorry

end NUMINAMATH_CALUDE_special_sale_discount_l3983_398347


namespace NUMINAMATH_CALUDE_distribute_five_to_three_l3983_398362

/-- The number of ways to distribute n distinct objects into k distinct groups,
    where each group must contain at least one object -/
def distribute (n k : ℕ) : ℕ :=
  sorry

/-- The number of ways to distribute 5 distinct objects into 3 distinct groups,
    where each group must contain at least one object, is 150 -/
theorem distribute_five_to_three : distribute 5 3 = 150 := by
  sorry

end NUMINAMATH_CALUDE_distribute_five_to_three_l3983_398362


namespace NUMINAMATH_CALUDE_meet_time_theorem_l3983_398350

def round_time_P : ℕ := 252
def round_time_Q : ℕ := 198
def round_time_R : ℕ := 315

theorem meet_time_theorem : 
  Nat.lcm (Nat.lcm round_time_P round_time_Q) round_time_R = 13860 := by
  sorry

end NUMINAMATH_CALUDE_meet_time_theorem_l3983_398350


namespace NUMINAMATH_CALUDE_euler_line_intersection_l3983_398354

structure Triangle where
  vertices : Fin 3 → ℝ × ℝ

def Triangle.isAcute (t : Triangle) : Prop := sorry

def Triangle.isObtuse (t : Triangle) : Prop := sorry

def Triangle.eulerLine (t : Triangle) : Set (ℝ × ℝ) := sorry

def Triangle.sides (t : Triangle) : Fin 3 → Set (ℝ × ℝ) := sorry

def Triangle.largestSide (t : Triangle) : Set (ℝ × ℝ) := sorry

def Triangle.smallestSide (t : Triangle) : Set (ℝ × ℝ) := sorry

def Triangle.medianSide (t : Triangle) : Set (ℝ × ℝ) := sorry

theorem euler_line_intersection (t : Triangle) :
  (t.isAcute → (t.eulerLine ∩ t.largestSide).Nonempty ∧ (t.eulerLine ∩ t.smallestSide).Nonempty) ∧
  (t.isObtuse → (t.eulerLine ∩ t.largestSide).Nonempty ∧ (t.eulerLine ∩ t.medianSide).Nonempty) :=
sorry

end NUMINAMATH_CALUDE_euler_line_intersection_l3983_398354


namespace NUMINAMATH_CALUDE_quadratic_equations_solutions_l3983_398388

theorem quadratic_equations_solutions :
  let eq1 : ℝ → Prop := λ x ↦ x^2 + 4*x - 5 = 0
  let eq2 : ℝ → Prop := λ x ↦ x^2 - 3*x + 1 = 0
  let solutions1 : Set ℝ := {1, -5}
  let solutions2 : Set ℝ := {(3 + Real.sqrt 5) / 2, (3 - Real.sqrt 5) / 2}
  (∀ x ∈ solutions1, eq1 x) ∧ (∀ y, eq1 y → y ∈ solutions1) ∧
  (∀ x ∈ solutions2, eq2 x) ∧ (∀ y, eq2 y → y ∈ solutions2) :=
by sorry


end NUMINAMATH_CALUDE_quadratic_equations_solutions_l3983_398388


namespace NUMINAMATH_CALUDE_vector_subtraction_and_scaling_l3983_398303

/-- Given two 2D vectors a and b, prove that a - 2b results in the specified coordinates. -/
theorem vector_subtraction_and_scaling (a b : Fin 2 → ℝ) (h1 : a = ![1, 2]) (h2 : b = ![-3, 2]) :
  a - 2 • b = ![7, -2] := by
  sorry

end NUMINAMATH_CALUDE_vector_subtraction_and_scaling_l3983_398303


namespace NUMINAMATH_CALUDE_geometric_sequence_condition_l3983_398306

/-- A geometric sequence with first term a and common ratio q -/
def geometric_sequence (a q : ℝ) : ℕ → ℝ := fun n => a * q ^ (n - 1)

theorem geometric_sequence_condition (a q : ℝ) (h : a > 0) :
  (∀ n : ℕ, geometric_sequence a q n = a * q ^ (n - 1)) →
  (geometric_sequence a q 1 < geometric_sequence a q 3 → geometric_sequence a q 3 < geometric_sequence a q 6) ∧
  ¬(geometric_sequence a q 1 < geometric_sequence a q 3 → geometric_sequence a q 3 < geometric_sequence a q 6) :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_condition_l3983_398306


namespace NUMINAMATH_CALUDE_not_power_of_two_l3983_398369

def lower_bound : Nat := 11111
def upper_bound : Nat := 99999

theorem not_power_of_two : ∃ (n : Nat), 
  (n = (upper_bound - lower_bound + 1) * upper_bound) ∧ 
  (n % 9 = 0) ∧
  (∀ (m : Nat), (2^m ≠ n)) := by
  sorry

end NUMINAMATH_CALUDE_not_power_of_two_l3983_398369


namespace NUMINAMATH_CALUDE_total_pies_sold_l3983_398390

/-- Represents the daily pie sales for a week -/
structure WeekSales where
  monday : ℕ
  tuesday : ℕ
  wednesday : ℕ
  thursday : ℕ
  friday : ℕ
  saturday : ℕ
  sunday : ℕ

/-- Calculates the total sales for a week -/
def totalSales (sales : WeekSales) : ℕ :=
  sales.monday + sales.tuesday + sales.wednesday + sales.thursday + sales.friday + sales.saturday + sales.sunday

/-- The actual sales data for the week -/
def actualSales : WeekSales := {
  monday := 8,
  tuesday := 12,
  wednesday := 14,
  thursday := 20,
  friday := 20,
  saturday := 20,
  sunday := 20
}

/-- Theorem: The total number of pies sold in the week is 114 -/
theorem total_pies_sold : totalSales actualSales = 114 := by
  sorry

end NUMINAMATH_CALUDE_total_pies_sold_l3983_398390


namespace NUMINAMATH_CALUDE_equal_debt_after_calculated_days_l3983_398311

/-- The number of days until Darren and Fergie owe the same amount -/
def days_until_equal_debt : ℝ := 53.75

/-- Darren's initial borrowed amount -/
def darren_initial_borrowed : ℝ := 200

/-- Fergie's initial borrowed amount -/
def fergie_initial_borrowed : ℝ := 300

/-- Darren's initial daily interest rate -/
def darren_initial_rate : ℝ := 0.08

/-- Darren's reduced daily interest rate after 10 days -/
def darren_reduced_rate : ℝ := 0.06

/-- Fergie's daily interest rate -/
def fergie_rate : ℝ := 0.04

/-- The number of days after which Darren's interest rate changes -/
def rate_change_days : ℝ := 10

/-- Theorem stating that Darren and Fergie owe the same amount after the calculated number of days -/
theorem equal_debt_after_calculated_days :
  let darren_debt := if days_until_equal_debt ≤ rate_change_days
    then darren_initial_borrowed * (1 + darren_initial_rate * days_until_equal_debt)
    else darren_initial_borrowed * (1 + darren_initial_rate * rate_change_days) *
      (1 + darren_reduced_rate * (days_until_equal_debt - rate_change_days))
  let fergie_debt := fergie_initial_borrowed * (1 + fergie_rate * days_until_equal_debt)
  darren_debt = fergie_debt := by sorry


end NUMINAMATH_CALUDE_equal_debt_after_calculated_days_l3983_398311


namespace NUMINAMATH_CALUDE_raft_journey_time_l3983_398300

/-- Represents the journey of a steamboat between two cities -/
structure SteamboatJourney where
  speed : ℝ  -- Speed of the steamboat
  current : ℝ  -- Speed of the river current
  time_ab : ℝ  -- Time from A to B
  time_ba : ℝ  -- Time from B to A

/-- Calculates the time taken by a raft to travel from A to B -/
def raft_time (journey : SteamboatJourney) : ℝ :=
  60  -- The actual calculation is omitted and replaced with the result

/-- Theorem stating the raft journey time given steamboat journey details -/
theorem raft_journey_time (journey : SteamboatJourney) 
  (h1 : journey.time_ab = 10)
  (h2 : journey.time_ba = 15)
  (h3 : journey.speed > 0)
  (h4 : journey.current > 0) :
  raft_time journey = 60 := by
  sorry


end NUMINAMATH_CALUDE_raft_journey_time_l3983_398300


namespace NUMINAMATH_CALUDE_simple_interest_rate_example_l3983_398378

/-- Calculates the simple interest rate given principal, amount, and time -/
def simple_interest_rate (principal amount : ℚ) (time : ℕ) : ℚ :=
  ((amount - principal) * 100) / (principal * time)

theorem simple_interest_rate_example :
  simple_interest_rate 750 1050 5 = 8 := by
  sorry

end NUMINAMATH_CALUDE_simple_interest_rate_example_l3983_398378


namespace NUMINAMATH_CALUDE_correct_proof_by_contradiction_components_l3983_398360

/-- Represents the components used in a proof by contradiction --/
inductive ProofByContradictionComponent
  | assumption
  | originalConditions
  | axiomTheoremsDefinitions
  | originalConclusion

/-- Defines the set of components used in a proof by contradiction --/
def proofByContradictionComponents : Set ProofByContradictionComponent :=
  {ProofByContradictionComponent.assumption,
   ProofByContradictionComponent.originalConditions,
   ProofByContradictionComponent.axiomTheoremsDefinitions}

/-- Theorem stating the correct components used in a proof by contradiction --/
theorem correct_proof_by_contradiction_components :
  proofByContradictionComponents =
    {ProofByContradictionComponent.assumption,
     ProofByContradictionComponent.originalConditions,
     ProofByContradictionComponent.axiomTheoremsDefinitions} :=
by
  sorry


end NUMINAMATH_CALUDE_correct_proof_by_contradiction_components_l3983_398360


namespace NUMINAMATH_CALUDE_shanghai_population_equality_l3983_398371

/-- The population of Shanghai in millions -/
def shanghai_population : ℝ := 16.3

/-- Scientific notation representation of Shanghai's population -/
def shanghai_population_scientific : ℝ := 1.63 * 10^7

/-- Theorem stating that the population of Shanghai expressed in millions 
    is equal to its representation in scientific notation -/
theorem shanghai_population_equality : 
  shanghai_population * 10^6 = shanghai_population_scientific := by
  sorry

end NUMINAMATH_CALUDE_shanghai_population_equality_l3983_398371


namespace NUMINAMATH_CALUDE_power_of_seven_mod_nine_l3983_398320

theorem power_of_seven_mod_nine : 7^15 % 9 = 1 := by
  sorry

end NUMINAMATH_CALUDE_power_of_seven_mod_nine_l3983_398320


namespace NUMINAMATH_CALUDE_det_of_specific_matrix_l3983_398310

/-- The determinant of the matrix [5 -2; 4 3] is 23. -/
theorem det_of_specific_matrix :
  let M : Matrix (Fin 2) (Fin 2) ℤ := !![5, -2; 4, 3]
  Matrix.det M = 23 := by
  sorry

end NUMINAMATH_CALUDE_det_of_specific_matrix_l3983_398310


namespace NUMINAMATH_CALUDE_solution_set_l3983_398315

theorem solution_set : ∀ x y : ℝ,
  (3/20 + |x - 15/40| < 7/20 ∧ y = 2*x + 1) ↔ 
  (7/20 < x ∧ x < 2/5 ∧ 17/10 ≤ y ∧ y ≤ 11/5) :=
by sorry

end NUMINAMATH_CALUDE_solution_set_l3983_398315


namespace NUMINAMATH_CALUDE_triangle_properties_l3983_398361

theorem triangle_properties (a b c : ℝ) (A B C : ℝ) :
  c = 2 →
  C = π / 3 →
  (∀ a' b' : ℝ, a' + b' + c ≤ 6) ∧
  (2 * Real.sin (2 * A) + Real.sin (2 * B + C) = Real.sin C →
   1/2 * a * b * Real.sin C = 2 * Real.sqrt 6 / 3) :=
by sorry

end NUMINAMATH_CALUDE_triangle_properties_l3983_398361


namespace NUMINAMATH_CALUDE_smallest_composite_no_small_factors_l3983_398379

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

def is_composite (n : ℕ) : Prop := n > 1 ∧ ∃ d : ℕ, d > 1 ∧ d < n ∧ n % d = 0

def has_no_prime_factors_less_than_20 (n : ℕ) : Prop :=
  ∀ p : ℕ, is_prime p → p < 20 → ¬(n % p = 0)

theorem smallest_composite_no_small_factors :
  ∃! n : ℕ, is_composite n ∧ has_no_prime_factors_less_than_20 n ∧
  ∀ m : ℕ, m < n → ¬(is_composite m ∧ has_no_prime_factors_less_than_20 m) :=
by sorry

end NUMINAMATH_CALUDE_smallest_composite_no_small_factors_l3983_398379


namespace NUMINAMATH_CALUDE_inequality_solution_range_function_minimum_value_l3983_398348

-- Part 1
theorem inequality_solution_range (a : ℝ) :
  (∀ x : ℝ, |x + 1| + |x - 2| ≥ a) → a ≤ 3 := by sorry

-- Part 2
theorem function_minimum_value (a : ℝ) :
  (∃ m : ℝ, m = 5 ∧ ∀ x : ℝ, |x + 1| + 2 * |x - a| ≥ m) →
  a = 4 ∨ a = -6 := by sorry

end NUMINAMATH_CALUDE_inequality_solution_range_function_minimum_value_l3983_398348


namespace NUMINAMATH_CALUDE_dawn_monthly_savings_l3983_398399

theorem dawn_monthly_savings (annual_salary : ℝ) (months_per_year : ℕ) (savings_rate : ℝ) : 
  annual_salary = 48000 ∧ 
  months_per_year = 12 ∧ 
  savings_rate = 0.1 → 
  (annual_salary / months_per_year) * savings_rate = 400 := by
sorry

end NUMINAMATH_CALUDE_dawn_monthly_savings_l3983_398399


namespace NUMINAMATH_CALUDE_box_volume_increase_l3983_398340

/-- Given a rectangular box with length l, width w, and height h, 
    if the volume is 3000, surface area is 1380, and sum of edges is 160,
    then increasing each dimension by 2 results in a volume of 4548 --/
theorem box_volume_increase (l w h : ℝ) 
  (volume : l * w * h = 3000)
  (surface_area : 2 * (l * w + w * h + l * h) = 1380)
  (edge_sum : 4 * (l + w + h) = 160) :
  (l + 2) * (w + 2) * (h + 2) = 4548 := by
  sorry


end NUMINAMATH_CALUDE_box_volume_increase_l3983_398340


namespace NUMINAMATH_CALUDE_min_soldiers_in_formation_l3983_398373

/-- Represents a rectangular formation of soldiers -/
structure SoldierFormation where
  columns : ℕ
  rows : ℕ
  new_uniforms : ℕ

/-- Checks if the formation satisfies the given conditions -/
def is_valid_formation (f : SoldierFormation) : Prop :=
  f.new_uniforms = (f.columns * f.rows) / 100 ∧
  f.new_uniforms ≥ (3 * f.columns) / 10 ∧
  f.new_uniforms ≥ (2 * f.rows) / 5

/-- The theorem stating the minimum number of soldiers -/
theorem min_soldiers_in_formation :
  ∀ f : SoldierFormation, is_valid_formation f → f.columns * f.rows ≥ 1200 :=
by sorry

end NUMINAMATH_CALUDE_min_soldiers_in_formation_l3983_398373


namespace NUMINAMATH_CALUDE_quadratic_coefficient_b_l3983_398391

theorem quadratic_coefficient_b (a b c y₁ y₂ y₃ : ℝ) : 
  y₁ = a + b + c →
  y₂ = a - b + c →
  y₃ = 4*a + 2*b + c →
  y₁ - y₂ = 8 →
  y₃ = y₁ + 2 →
  b = 4 := by
sorry


end NUMINAMATH_CALUDE_quadratic_coefficient_b_l3983_398391


namespace NUMINAMATH_CALUDE_strip_arrangement_area_l3983_398323

/-- Represents a rectangular paper strip -/
structure Strip where
  length : ℝ
  width : ℝ

/-- Calculates the area of a strip -/
def stripArea (s : Strip) : ℝ := s.length * s.width

/-- Calculates the overlap area between two perpendicular strips -/
def overlapArea (s1 s2 : Strip) : ℝ := s1.width * s2.width

/-- Represents the arrangement of strips on the table -/
structure StripArrangement where
  horizontalStrips : Fin 2 → Strip
  verticalStrips : Fin 2 → Strip

/-- Calculates the total area covered by the strips -/
def totalCoveredArea (arrangement : StripArrangement) : ℝ :=
  let totalStripArea := (Finset.sum (Finset.range 2) (λ i => stripArea (arrangement.horizontalStrips i))) +
                        (Finset.sum (Finset.range 2) (λ i => stripArea (arrangement.verticalStrips i)))
  let totalOverlapArea := Finset.sum (Finset.range 2) (λ i =>
                            Finset.sum (Finset.range 2) (λ j =>
                              overlapArea (arrangement.horizontalStrips i) (arrangement.verticalStrips j)))
  totalStripArea - totalOverlapArea

theorem strip_arrangement_area :
  ∀ (arrangement : StripArrangement),
    (∀ i : Fin 2, arrangement.horizontalStrips i = ⟨8, 1⟩) →
    (∀ i : Fin 2, arrangement.verticalStrips i = ⟨8, 1⟩) →
    totalCoveredArea arrangement = 28 := by
  sorry

end NUMINAMATH_CALUDE_strip_arrangement_area_l3983_398323


namespace NUMINAMATH_CALUDE_john_movie_count_l3983_398326

/-- The number of movies John has -/
def num_movies : ℕ := 100

/-- The trade-in value of each VHS in dollars -/
def vhs_value : ℕ := 2

/-- The cost of each DVD in dollars -/
def dvd_cost : ℕ := 10

/-- The total cost to replace all movies in dollars -/
def total_cost : ℕ := 800

theorem john_movie_count :
  (dvd_cost * num_movies) - (vhs_value * num_movies) = total_cost :=
by sorry

end NUMINAMATH_CALUDE_john_movie_count_l3983_398326


namespace NUMINAMATH_CALUDE_cosine_function_parameters_l3983_398308

theorem cosine_function_parameters (a b c d : ℝ) : 
  a > 0 → b > 0 → 
  (∀ x, ∃ y, y = a * Real.cos (b * x + c) + d) →
  (4 * Real.pi / b = 4 * Real.pi) →
  d = 3 →
  (∃ x, a * Real.cos (b * x + c) + d = 8) →
  (∃ x, a * Real.cos (b * x + c) + d = -2) →
  a = 5 ∧ b = 1 := by
sorry

end NUMINAMATH_CALUDE_cosine_function_parameters_l3983_398308


namespace NUMINAMATH_CALUDE_dividing_line_slope_l3983_398341

/-- Polygon in the xy-plane with given vertices -/
structure Polygon where
  vertices : List (ℝ × ℝ)

/-- Line passing through the origin with a given slope -/
structure Line where
  slope : ℝ

/-- Function to calculate the area of a polygon -/
def calculateArea (p : Polygon) : ℝ := sorry

/-- Function to check if a line divides a polygon into two equal areas -/
def dividesEqualArea (p : Polygon) (l : Line) : Prop := sorry

/-- The polygon with the given vertices -/
def givenPolygon : Polygon := {
  vertices := [(0, 0), (0, 4), (4, 4), (4, 2), (7, 2), (7, 0)]
}

/-- The theorem stating that the line with slope 2/7 divides the given polygon into two equal areas -/
theorem dividing_line_slope : 
  dividesEqualArea givenPolygon { slope := 2/7 } := by sorry

end NUMINAMATH_CALUDE_dividing_line_slope_l3983_398341


namespace NUMINAMATH_CALUDE_car_rental_cost_l3983_398383

theorem car_rental_cost (gas_gallons : ℕ) (gas_price : ℚ) (mile_cost : ℚ) (miles_driven : ℕ) (total_cost : ℚ) :
  gas_gallons = 8 →
  gas_price = 7/2 →
  mile_cost = 1/2 →
  miles_driven = 320 →
  total_cost = 338 →
  (total_cost - (↑gas_gallons * gas_price + ↑miles_driven * mile_cost) : ℚ) = 150 := by
sorry

end NUMINAMATH_CALUDE_car_rental_cost_l3983_398383


namespace NUMINAMATH_CALUDE_truck_speed_calculation_l3983_398342

/-- The average speed of Truck X in miles per hour -/
def truck_x_speed : ℝ := 57

/-- The average speed of Truck Y in miles per hour -/
def truck_y_speed : ℝ := 63

/-- The initial distance Truck X is ahead of Truck Y in miles -/
def initial_distance : ℝ := 14

/-- The final distance Truck Y is ahead of Truck X in miles -/
def final_distance : ℝ := 4

/-- The time it takes for Truck Y to overtake Truck X in hours -/
def overtake_time : ℝ := 3

theorem truck_speed_calculation :
  truck_x_speed = (truck_y_speed * overtake_time - initial_distance - final_distance) / overtake_time :=
by
  sorry

#check truck_speed_calculation

end NUMINAMATH_CALUDE_truck_speed_calculation_l3983_398342


namespace NUMINAMATH_CALUDE_company_female_employees_l3983_398344

theorem company_female_employees 
  (total_employees : ℕ) 
  (male_employees : ℕ) 
  (total_managers : ℕ) 
  (male_managers : ℕ) 
  (h1 : total_managers = (2 : ℕ) * total_employees / (5 : ℕ)) 
  (h2 : male_managers = (2 : ℕ) * male_employees / (5 : ℕ)) 
  (h3 : total_managers = male_managers + 200) :
  total_employees - male_employees = 500 := by
sorry

end NUMINAMATH_CALUDE_company_female_employees_l3983_398344


namespace NUMINAMATH_CALUDE_raghu_investment_l3983_398375

theorem raghu_investment (total_investment : ℝ) (vishal_investment : ℝ → ℝ) (trishul_investment : ℝ → ℝ) :
  total_investment = 7225 ∧
  (∀ r, vishal_investment r = 1.1 * (trishul_investment r)) ∧
  (∀ r, trishul_investment r = 0.9 * r) →
  ∃ r, r = 2500 ∧ r + trishul_investment r + vishal_investment r = total_investment :=
by sorry

end NUMINAMATH_CALUDE_raghu_investment_l3983_398375


namespace NUMINAMATH_CALUDE_no_real_roots_l3983_398301

theorem no_real_roots : ∀ x : ℝ, x^2 + 3*x + 5 ≠ 0 := by
  sorry

end NUMINAMATH_CALUDE_no_real_roots_l3983_398301


namespace NUMINAMATH_CALUDE_rhombus_diagonal_l3983_398382

theorem rhombus_diagonal (d1 : ℝ) (area : ℝ) (d2 : ℝ) : 
  d1 = 30 → area = 180 → area = (d1 * d2) / 2 → d2 = 12 := by
  sorry

end NUMINAMATH_CALUDE_rhombus_diagonal_l3983_398382


namespace NUMINAMATH_CALUDE_triangle_radius_ratio_l3983_398335

/-- Given a triangle with area S, circumradius R, and inradius r, 
    such that S^2 = 2R^2 + 8Rr + 3r^2, prove that R/r = 2 or R/r ≥ √2 + 1 -/
theorem triangle_radius_ratio (S R r : ℝ) (h : S^2 = 2*R^2 + 8*R*r + 3*r^2) :
  R / r = 2 ∨ R / r ≥ Real.sqrt 2 + 1 := by
  sorry

end NUMINAMATH_CALUDE_triangle_radius_ratio_l3983_398335


namespace NUMINAMATH_CALUDE_better_sequence_is_BAB_l3983_398302

/-- Represents a wrestler's opponent -/
inductive Opponent
  | A  -- Andrei
  | B  -- Boris

/-- Represents a sequence of three matches -/
def MatchSequence := List Opponent

/-- Calculates the probability of winning against an opponent -/
def winProbability (o : Opponent) : Real :=
  match o with
  | Opponent.A => 0.6  -- 1 - 0.4 (probability of losing to Andrei)
  | Opponent.B => 0.7  -- 1 - 0.3 (probability of losing to Boris)

/-- Calculates the probability of qualifying (winning at least two consecutive matches) -/
def qualifyingProbability (seq : MatchSequence) : Real :=
  match seq with
  | [o1, o2, o3] =>
    let p1 := winProbability o1
    let p2 := winProbability o2
    let p3 := winProbability o3
    p1 * p2 + p1 * (1 - p2) * p3 + (1 - p1) * p2 * p3
  | _ => 0  -- Invalid sequence

theorem better_sequence_is_BAB :
  let seqABA : MatchSequence := [Opponent.A, Opponent.B, Opponent.A]
  let seqBAB : MatchSequence := [Opponent.B, Opponent.A, Opponent.B]
  qualifyingProbability seqBAB > qualifyingProbability seqABA :=
by sorry

#eval qualifyingProbability [Opponent.B, Opponent.A, Opponent.B]

end NUMINAMATH_CALUDE_better_sequence_is_BAB_l3983_398302


namespace NUMINAMATH_CALUDE_abs_f_decreasing_on_4_6_l3983_398336

-- Define the properties of the function f
def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

def is_symmetric_about (f : ℝ → ℝ) (a : ℝ) : Prop :=
  ∀ x, f (a + x) = f (a - x)

def is_increasing_on (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → f x < f y

-- State the theorem
theorem abs_f_decreasing_on_4_6 
  (f : ℝ → ℝ) 
  (h_even : is_even_function f)
  (h_sym : is_symmetric_about f 2)
  (h_inc : is_increasing_on f (-2) 0)
  (h_nonneg : f (-2) ≥ 0) :
  ∀ x₁ x₂, 4 ≤ x₁ ∧ x₁ < x₂ ∧ x₂ ≤ 6 → |f x₁| > |f x₂| :=
sorry

end NUMINAMATH_CALUDE_abs_f_decreasing_on_4_6_l3983_398336


namespace NUMINAMATH_CALUDE_incorrect_equality_l3983_398331

theorem incorrect_equality (h : (12.5 / 12.5) = (2.4 / 2.4)) :
  ¬ (25 * (0.5 / 0.5) = 4 * (0.6 / 0.6)) := by
  sorry

end NUMINAMATH_CALUDE_incorrect_equality_l3983_398331


namespace NUMINAMATH_CALUDE_mary_cards_left_l3983_398318

/-- The number of baseball cards Mary has left after giving away promised cards -/
def cards_left (initial : ℕ) (promised_fred : ℕ) (promised_jane : ℕ) (promised_tom : ℕ) 
               (bought : ℕ) (received : ℕ) : ℕ :=
  initial + bought + received - (promised_fred + promised_jane + promised_tom)

/-- Theorem stating that Mary will have 6 cards left -/
theorem mary_cards_left : 
  cards_left 18 26 15 36 40 25 = 6 := by sorry

end NUMINAMATH_CALUDE_mary_cards_left_l3983_398318


namespace NUMINAMATH_CALUDE_fourth_root_over_sixth_root_of_seven_l3983_398398

theorem fourth_root_over_sixth_root_of_seven (x : ℝ) :
  (7 : ℝ)^(1/4) / (7 : ℝ)^(1/6) = (7 : ℝ)^(1/12) := by
  sorry

end NUMINAMATH_CALUDE_fourth_root_over_sixth_root_of_seven_l3983_398398


namespace NUMINAMATH_CALUDE_wilsborough_savings_l3983_398314

/-- Mrs. Wilsborough's concert ticket purchase problem -/
theorem wilsborough_savings (vip_price regular_price : ℕ) 
  (vip_count regular_count leftover : ℕ) :
  vip_price = 100 →
  regular_price = 50 →
  vip_count = 2 →
  regular_count = 3 →
  leftover = 150 →
  vip_count * vip_price + regular_count * regular_price + leftover = 500 :=
by sorry

end NUMINAMATH_CALUDE_wilsborough_savings_l3983_398314


namespace NUMINAMATH_CALUDE_roots_product_l3983_398376

theorem roots_product (a b : ℝ) : 
  a^2 + a - 2020 = 0 → b^2 + b - 2020 = 0 → (a - 1) * (b - 1) = -2018 := by
  sorry

end NUMINAMATH_CALUDE_roots_product_l3983_398376


namespace NUMINAMATH_CALUDE_ratio_xyz_l3983_398381

theorem ratio_xyz (x y z : ℚ) 
  (h1 : (3/4) * y = (1/2) * x) 
  (h2 : (3/10) * x = (1/5) * z) : 
  ∃ (k : ℚ), k > 0 ∧ x = 6*k ∧ y = 4*k ∧ z = 9*k := by
sorry

end NUMINAMATH_CALUDE_ratio_xyz_l3983_398381
