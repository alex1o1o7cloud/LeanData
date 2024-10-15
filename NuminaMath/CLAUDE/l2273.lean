import Mathlib

namespace NUMINAMATH_CALUDE_arrangements_count_l2273_227313

/-- The number of arrangements of 3 boys and 2 girls in a row with girls at both ends -/
def arrangements_with_girls_at_ends : ℕ :=
  let num_boys : ℕ := 3
  let num_girls : ℕ := 2
  let girl_arrangements : ℕ := 2  -- A_2^2
  let boy_arrangements : ℕ := 6  -- A_3^3
  girl_arrangements * boy_arrangements

/-- Theorem stating that the number of arrangements is 12 -/
theorem arrangements_count : arrangements_with_girls_at_ends = 12 := by
  sorry

end NUMINAMATH_CALUDE_arrangements_count_l2273_227313


namespace NUMINAMATH_CALUDE_luncheon_cost_l2273_227360

/-- Given two luncheon bills, prove the cost of one sandwich, one coffee, and one pie --/
theorem luncheon_cost (s c p : ℚ) : 
  (5 * s + 8 * c + 2 * p = 510/100) →
  (6 * s + 11 * c + 2 * p = 645/100) →
  (s + c + p = 135/100) := by
  sorry

#check luncheon_cost

end NUMINAMATH_CALUDE_luncheon_cost_l2273_227360


namespace NUMINAMATH_CALUDE_cross_product_result_l2273_227373

def u : ℝ × ℝ × ℝ := (3, 4, 2)
def v : ℝ × ℝ × ℝ := (1, -2, 5)

def cross_product (a b : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  (a.2.1 * b.2.2 - a.2.2 * b.2.1,
   a.2.2 * b.1 - a.1 * b.2.2,
   a.1 * b.2.1 - a.2.1 * b.1)

theorem cross_product_result : cross_product u v = (24, -13, -10) := by
  sorry

end NUMINAMATH_CALUDE_cross_product_result_l2273_227373


namespace NUMINAMATH_CALUDE_children_left_on_bus_l2273_227333

theorem children_left_on_bus (initial_children : ℕ) (difference : ℕ) : 
  initial_children = 41 →
  difference = 23 →
  initial_children - difference = 18 :=
by sorry

end NUMINAMATH_CALUDE_children_left_on_bus_l2273_227333


namespace NUMINAMATH_CALUDE_points_on_line_l2273_227335

/-- Three points are collinear if they lie on the same straight line -/
def collinear (p1 p2 p3 : ℝ × ℝ) : Prop :=
  (p3.2 - p1.2) * (p2.1 - p1.1) = (p2.2 - p1.2) * (p3.1 - p1.1)

/-- The problem statement -/
theorem points_on_line (k : ℝ) :
  collinear (1, 2) (3, -2) (4, k/3) → k = -12 := by
  sorry

end NUMINAMATH_CALUDE_points_on_line_l2273_227335


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_l2273_227339

theorem imaginary_part_of_z (z : ℂ) (h : Complex.I * z = (1/2 : ℂ) * (1 + Complex.I)) :
  z.im = -1/2 := by sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_l2273_227339


namespace NUMINAMATH_CALUDE_find_unknown_number_l2273_227343

theorem find_unknown_number : ∃ x : ℝ, 
  (14 + x + 53) / 3 = (21 + 47 + 22) / 3 + 3 ∧ x = 32 := by
  sorry

end NUMINAMATH_CALUDE_find_unknown_number_l2273_227343


namespace NUMINAMATH_CALUDE_twelve_gon_consecutive_sides_sum_l2273_227363

theorem twelve_gon_consecutive_sides_sum (sides : Fin 12 → ℕ) 
  (h1 : ∀ i : Fin 12, sides i = i.val + 1) : 
  ∃ i : Fin 12, sides i + sides (i + 1) + sides (i + 2) > 20 :=
by sorry

end NUMINAMATH_CALUDE_twelve_gon_consecutive_sides_sum_l2273_227363


namespace NUMINAMATH_CALUDE_expression_simplification_l2273_227378

theorem expression_simplification (m : ℝ) (h1 : m^2 - 4 = 0) (h2 : m ≠ 2) :
  (m^2 + 6*m + 9) / (m - 2) / (m + 2 + (3*m + 4) / (m - 2)) = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l2273_227378


namespace NUMINAMATH_CALUDE_power_two_100_mod_3_l2273_227369

theorem power_two_100_mod_3 : 2^100 ≡ 1 [ZMOD 3] := by sorry

end NUMINAMATH_CALUDE_power_two_100_mod_3_l2273_227369


namespace NUMINAMATH_CALUDE_total_fish_catch_l2273_227356

def fishing_competition (jackson_daily : ℕ) (jonah_daily : ℕ) (george_catches : List ℕ) 
  (lily_catches : List ℕ) (alex_diff : ℕ) : Prop :=
  george_catches.length = 5 ∧ 
  lily_catches.length = 4 ∧
  ∀ i, i < 5 → List.get? (george_catches) i ≠ none ∧
  ∀ i, i < 4 → List.get? (lily_catches) i ≠ none ∧
  (jackson_daily * 5 + jonah_daily * 5 + george_catches.sum + lily_catches.sum + 
    (george_catches.map (λ x => x - alex_diff)).sum) = 159

theorem total_fish_catch : 
  fishing_competition 6 4 [8, 12, 7, 9, 11] [5, 6, 9, 5] 2 := by
  sorry

end NUMINAMATH_CALUDE_total_fish_catch_l2273_227356


namespace NUMINAMATH_CALUDE_chicken_increase_l2273_227381

/-- The increase in chickens is the sum of chickens bought on two days -/
theorem chicken_increase (initial : ℕ) (day1 : ℕ) (day2 : ℕ) :
  day1 + day2 = (initial + day1 + day2) - initial :=
by sorry

end NUMINAMATH_CALUDE_chicken_increase_l2273_227381


namespace NUMINAMATH_CALUDE_complex_fraction_simplification_l2273_227321

theorem complex_fraction_simplification :
  let i : ℂ := Complex.I
  (1 - 2*i) / (1 + 2*i) = -3/5 - 4/5*i :=
by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_simplification_l2273_227321


namespace NUMINAMATH_CALUDE_unique_solution_l2273_227319

/-- Represents the number of students in a class -/
structure ClassSize where
  small : Nat
  large_min : Nat
  large_max : Nat

/-- Represents the number of classes for each school -/
structure SchoolClasses where
  shouchun_small : Nat
  binhu_small : Nat
  binhu_large : Nat

/-- Check if the given class distribution satisfies all conditions -/
def satisfies_conditions (cs : ClassSize) (sc : SchoolClasses) : Prop :=
  sc.shouchun_small + sc.binhu_small + sc.binhu_large = 45 ∧
  sc.binhu_small = 2 * sc.binhu_large ∧
  cs.small * (sc.shouchun_small + sc.binhu_small) + cs.large_min * sc.binhu_large ≤ 1800 ∧
  1800 ≤ cs.small * (sc.shouchun_small + sc.binhu_small) + cs.large_max * sc.binhu_large

theorem unique_solution (cs : ClassSize) (h_cs : cs.small = 36 ∧ cs.large_min = 70 ∧ cs.large_max = 75) :
  ∃! sc : SchoolClasses, satisfies_conditions cs sc ∧ 
    sc.shouchun_small = 30 ∧ sc.binhu_small = 10 ∧ sc.binhu_large = 5 :=
  sorry

end NUMINAMATH_CALUDE_unique_solution_l2273_227319


namespace NUMINAMATH_CALUDE_relay_race_second_leg_time_l2273_227390

theorem relay_race_second_leg_time 
  (first_leg_time : ℝ) 
  (average_time : ℝ) 
  (h1 : first_leg_time = 58) 
  (h2 : average_time = 42) : 
  let second_leg_time := 2 * average_time - first_leg_time
  second_leg_time = 26 := by
  sorry

end NUMINAMATH_CALUDE_relay_race_second_leg_time_l2273_227390


namespace NUMINAMATH_CALUDE_bacon_tomatoes_difference_l2273_227344

/-- The number of students who suggested mashed potatoes -/
def mashed_potatoes : ℕ := 228

/-- The number of students who suggested bacon -/
def bacon : ℕ := 337

/-- The number of students who suggested tomatoes -/
def tomatoes : ℕ := 23

/-- The difference between the number of students who suggested bacon and tomatoes -/
theorem bacon_tomatoes_difference : bacon - tomatoes = 314 := by
  sorry

end NUMINAMATH_CALUDE_bacon_tomatoes_difference_l2273_227344


namespace NUMINAMATH_CALUDE_consecutive_integers_product_sum_l2273_227367

theorem consecutive_integers_product_sum : ∃ (n : ℕ), 
  n > 0 ∧ n * (n + 1) = 1080 ∧ n + (n + 1) = 65 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_integers_product_sum_l2273_227367


namespace NUMINAMATH_CALUDE_central_cell_removed_theorem_corner_cell_removed_theorem_l2273_227392

-- Define a 7x7 grid
def Grid := Fin 7 → Fin 7 → Bool

-- Define a domino placement
structure Domino where
  x : Fin 7
  y : Fin 7
  horizontal : Bool

-- Define a tiling of the grid
def Tiling := List Domino

-- Function to check if a tiling is valid for a given grid
def is_valid_tiling (g : Grid) (t : Tiling) : Prop := sorry

-- Function to count horizontal dominoes in a tiling
def count_horizontal (t : Tiling) : Nat := sorry

-- Function to count vertical dominoes in a tiling
def count_vertical (t : Tiling) : Nat := sorry

-- Define a grid with the central cell removed
def central_removed_grid : Grid := sorry

-- Define a grid with a corner cell removed
def corner_removed_grid : Grid := sorry

theorem central_cell_removed_theorem :
  ∃ t : Tiling, is_valid_tiling central_removed_grid t ∧
    count_horizontal t = count_vertical t := sorry

theorem corner_cell_removed_theorem :
  ¬∃ t : Tiling, is_valid_tiling corner_removed_grid t ∧
    count_horizontal t = count_vertical t := sorry

end NUMINAMATH_CALUDE_central_cell_removed_theorem_corner_cell_removed_theorem_l2273_227392


namespace NUMINAMATH_CALUDE_completing_square_transform_l2273_227350

theorem completing_square_transform (x : ℝ) : 
  (x^2 - 2*x - 7 = 0) ↔ ((x - 1)^2 = 8) := by
  sorry

end NUMINAMATH_CALUDE_completing_square_transform_l2273_227350


namespace NUMINAMATH_CALUDE_books_per_day_l2273_227357

def total_books : ℕ := 15
def total_days : ℕ := 3

theorem books_per_day : (total_books / total_days : ℚ) = 5 := by
  sorry

end NUMINAMATH_CALUDE_books_per_day_l2273_227357


namespace NUMINAMATH_CALUDE_females_with_advanced_degrees_l2273_227393

theorem females_with_advanced_degrees 
  (total_employees : ℕ) 
  (total_females : ℕ) 
  (employees_with_advanced_degrees : ℕ) 
  (males_with_college_only : ℕ) 
  (h1 : total_employees = 160) 
  (h2 : total_females = 90) 
  (h3 : employees_with_advanced_degrees = 80) 
  (h4 : males_with_college_only = 40) :
  total_females - (total_employees - employees_with_advanced_degrees - males_with_college_only) = 50 :=
by sorry

end NUMINAMATH_CALUDE_females_with_advanced_degrees_l2273_227393


namespace NUMINAMATH_CALUDE_contractor_absent_days_l2273_227380

/-- Represents the contract details and calculates the number of absent days -/
def calculate_absent_days (total_days : ℕ) (pay_per_day : ℚ) (fine_per_day : ℚ) (total_pay : ℚ) : ℚ :=
  let worked_days := total_days - (total_pay - total_days * pay_per_day) / (pay_per_day + fine_per_day)
  total_days - worked_days

/-- Theorem stating that given the specific contract conditions, the number of absent days is 12 -/
theorem contractor_absent_days :
  let total_days : ℕ := 30
  let pay_per_day : ℚ := 25
  let fine_per_day : ℚ := 7.5
  let total_pay : ℚ := 360
  calculate_absent_days total_days pay_per_day fine_per_day total_pay = 12 := by
  sorry

#eval calculate_absent_days 30 25 7.5 360

end NUMINAMATH_CALUDE_contractor_absent_days_l2273_227380


namespace NUMINAMATH_CALUDE_equidistant_point_location_l2273_227386

-- Define a 2D point
structure Point2D where
  x : ℝ
  y : ℝ

-- Define a quadrilateral
structure Quadrilateral where
  A : Point2D
  B : Point2D
  C : Point2D
  D : Point2D

-- Define the property of being convex
def isConvex (q : Quadrilateral) : Prop := sorry

-- Define the distance between two points
def distance (p1 p2 : Point2D) : ℝ := sorry

-- Define the property of a point being equidistant from all vertices
def isEquidistant (p : Point2D) (q : Quadrilateral) : Prop :=
  distance p q.A = distance p q.B ∧
  distance p q.A = distance p q.C ∧
  distance p q.A = distance p q.D

-- Define the property of a point being inside a quadrilateral
def isInside (p : Point2D) (q : Quadrilateral) : Prop := sorry

-- Define the property of a point being outside a quadrilateral
def isOutside (p : Point2D) (q : Quadrilateral) : Prop := sorry

-- Define the property of a point being on the boundary of a quadrilateral
def isOnBoundary (p : Point2D) (q : Quadrilateral) : Prop := sorry

theorem equidistant_point_location (q : Quadrilateral) (h : isConvex q) :
  ∃ p : Point2D, isEquidistant p q ∧
    (isInside p q ∨ isOutside p q ∨ isOnBoundary p q) := by
  sorry

end NUMINAMATH_CALUDE_equidistant_point_location_l2273_227386


namespace NUMINAMATH_CALUDE_sample_probability_l2273_227389

/-- Simple random sampling with given conditions -/
def SimpleRandomSampling (n : ℕ) : Prop :=
  n > 0 ∧ 
  (1 : ℚ) / n = 1 / 8 ∧
  ∀ i : ℕ, i ≤ n → (1 - (1 : ℚ) / n)^3 = (n - 1 : ℚ)^3 / n^3

theorem sample_probability (n : ℕ) (h : SimpleRandomSampling n) : 
  n = 8 ∧ (1 - (7 : ℚ) / 8^3) = 169 / 512 := by
  sorry

#check sample_probability

end NUMINAMATH_CALUDE_sample_probability_l2273_227389


namespace NUMINAMATH_CALUDE_reinforcement_size_is_300_l2273_227377

/-- Calculates the reinforcement size given the initial garrison size, 
    initial provision duration, days passed, and remaining provision duration -/
def calculate_reinforcement (initial_garrison : ℕ) (initial_duration : ℕ) 
                             (days_passed : ℕ) (remaining_duration : ℕ) : ℕ :=
  let total_provisions := initial_garrison * initial_duration
  let provisions_left := initial_garrison * (initial_duration - days_passed)
  (provisions_left / remaining_duration) - initial_garrison

/-- Theorem stating that the reinforcement size is 300 given the problem conditions -/
theorem reinforcement_size_is_300 : 
  calculate_reinforcement 150 31 16 5 = 300 := by
  sorry

end NUMINAMATH_CALUDE_reinforcement_size_is_300_l2273_227377


namespace NUMINAMATH_CALUDE_trevor_placed_105_pieces_l2273_227315

/-- Represents the puzzle problem --/
def PuzzleProblem (total : ℕ) (border : ℕ) (missing : ℕ) (joeMultiplier : ℕ) :=
  {trevor : ℕ // 
    trevor + joeMultiplier * trevor + border + missing = total ∧
    trevor > 0 ∧
    joeMultiplier > 0}

theorem trevor_placed_105_pieces :
  ∃ (p : PuzzleProblem 500 75 5 3), p.val = 105 := by
  sorry

end NUMINAMATH_CALUDE_trevor_placed_105_pieces_l2273_227315


namespace NUMINAMATH_CALUDE_lila_sticker_count_l2273_227320

/-- The number of stickers each person has -/
structure StickerCount where
  kristoff : ℕ
  riku : ℕ
  lila : ℕ

/-- The conditions of the sticker problem -/
def sticker_problem (s : StickerCount) : Prop :=
  s.kristoff = 85 ∧
  s.riku = 25 * s.kristoff ∧
  s.lila = 2 * (s.kristoff + s.riku)

/-- The theorem stating that Lila has 4420 stickers -/
theorem lila_sticker_count (s : StickerCount) 
  (h : sticker_problem s) : s.lila = 4420 := by
  sorry

end NUMINAMATH_CALUDE_lila_sticker_count_l2273_227320


namespace NUMINAMATH_CALUDE_jacob_age_jacob_age_proof_l2273_227317

theorem jacob_age : ℕ → Prop :=
  fun j : ℕ =>
    ∃ t : ℕ,
      t = j / 2 ∧  -- Tony's age is half of Jacob's age
      t + 6 = 18 ∧ -- In 6 years, Tony will be 18 years old
      j = 24       -- Jacob's current age is 24

-- The proof of the theorem
theorem jacob_age_proof : ∃ j : ℕ, jacob_age j :=
  sorry

end NUMINAMATH_CALUDE_jacob_age_jacob_age_proof_l2273_227317


namespace NUMINAMATH_CALUDE_only_two_subsets_implies_a_zero_or_one_l2273_227391

-- Define the set A
def A (a : ℝ) : Set ℝ := {x : ℝ | a * x^2 + 2 * x + 1 = 0}

-- State the theorem
theorem only_two_subsets_implies_a_zero_or_one :
  ∀ a : ℝ, (∀ S : Set ℝ, S ⊆ A a → (S = ∅ ∨ S = A a)) → (a = 0 ∨ a = 1) := by
  sorry

end NUMINAMATH_CALUDE_only_two_subsets_implies_a_zero_or_one_l2273_227391


namespace NUMINAMATH_CALUDE_difference_of_squares_650_350_l2273_227395

theorem difference_of_squares_650_350 : 650^2 - 350^2 = 300000 := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_650_350_l2273_227395


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l2273_227366

def A : Set ℝ := {x | x + 1/2 ≥ 3/2}
def B : Set ℝ := {x | x^2 + x < 6}

theorem intersection_of_A_and_B : 
  A ∩ B = {x : ℝ | (-3 < x ∧ x ≤ -2) ∨ (1 ≤ x ∧ x < 2)} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l2273_227366


namespace NUMINAMATH_CALUDE_roy_blue_pens_l2273_227348

/-- The number of blue pens Roy has -/
def blue_pens : ℕ := 2

/-- The number of black pens Roy has -/
def black_pens : ℕ := 2 * blue_pens

/-- The number of red pens Roy has -/
def red_pens : ℕ := 2 * black_pens - 2

/-- The total number of pens Roy has -/
def total_pens : ℕ := 12

theorem roy_blue_pens :
  blue_pens = 2 ∧
  black_pens = 2 * blue_pens ∧
  red_pens = 2 * black_pens - 2 ∧
  total_pens = blue_pens + black_pens + red_pens ∧
  total_pens = 12 := by
  sorry

end NUMINAMATH_CALUDE_roy_blue_pens_l2273_227348


namespace NUMINAMATH_CALUDE_min_value_sum_reciprocals_l2273_227312

theorem min_value_sum_reciprocals (m n : ℝ) 
  (h1 : 2 * m + n = 2) 
  (h2 : m > 0) 
  (h3 : n > 0) : 
  (∀ x y : ℝ, x > 0 → y > 0 → 2 * x + y = 2 → 1 / m + 2 / n ≤ 1 / x + 2 / y) ∧ 
  (∃ x y : ℝ, x > 0 ∧ y > 0 ∧ 2 * x + y = 2 ∧ 1 / x + 2 / y = 4) := by
  sorry

end NUMINAMATH_CALUDE_min_value_sum_reciprocals_l2273_227312


namespace NUMINAMATH_CALUDE_new_encoding_of_original_message_l2273_227304

/-- Represents the encoding of a character in the old system -/
def OldEncoding : Char → String
| 'A' => "011"
| 'B' => "011"
| 'C' => "0"
| _ => ""

/-- Represents the encoding of a character in the new system -/
def NewEncoding : Char → String
| 'A' => "21"
| 'B' => "122"
| 'C' => "1"
| _ => ""

/-- Decodes a string from the old encoding system -/
def decodeOld (s : String) : String := sorry

/-- Encodes a string using the new encoding system -/
def encodeNew (s : String) : String := sorry

/-- The original message in the old encoding -/
def originalMessage : String := "011011010011"

/-- Theorem stating that the new encoding of the original message is "211221121" -/
theorem new_encoding_of_original_message :
  encodeNew (decodeOld originalMessage) = "211221121" := by sorry

end NUMINAMATH_CALUDE_new_encoding_of_original_message_l2273_227304


namespace NUMINAMATH_CALUDE_smallest_winning_number_l2273_227324

theorem smallest_winning_number : ∃ N : ℕ, 
  (N = 46) ∧ 
  (0 ≤ N) ∧ 
  (N ≤ 999) ∧ 
  (9 * N - 80 < 1000) ∧ 
  (27 * N - 240 ≥ 1000) ∧ 
  (∀ k : ℕ, k < N → (9 * k - 80 ≥ 1000 ∨ 27 * k - 240 < 1000)) :=
by sorry

end NUMINAMATH_CALUDE_smallest_winning_number_l2273_227324


namespace NUMINAMATH_CALUDE_lcm_of_40_90_150_l2273_227342

theorem lcm_of_40_90_150 : Nat.lcm 40 (Nat.lcm 90 150) = 1800 := by sorry

end NUMINAMATH_CALUDE_lcm_of_40_90_150_l2273_227342


namespace NUMINAMATH_CALUDE_certain_number_exists_l2273_227300

theorem certain_number_exists : ∃ x : ℝ, 220050 = (555 + x) * (2 * (x - 555)) + 50 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_exists_l2273_227300


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_l2273_227383

theorem quadratic_equation_roots : 
  let f : ℝ → ℝ := λ x ↦ x^2 - 4
  ∃ x₁ x₂ : ℝ, x₁ = 2 ∧ x₂ = -2 ∧ f x₁ = 0 ∧ f x₂ = 0 ∧ 
  ∀ x : ℝ, f x = 0 → x = x₁ ∨ x = x₂ := by
sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_l2273_227383


namespace NUMINAMATH_CALUDE_store_constraints_equivalence_l2273_227307

/-- Represents the constraints on product purchases in a store. -/
def StoreConstraints (x : ℝ) : Prop :=
  let productACost : ℝ := 8
  let productBCost : ℝ := 2
  let productBQuantity : ℝ := 2 * x - 4
  let totalItems : ℝ := x + productBQuantity
  let totalCost : ℝ := productACost * x + productBCost * productBQuantity
  (totalItems ≥ 32) ∧ (totalCost ≤ 148)

/-- Theorem stating that the given system of inequalities correctly represents the store constraints. -/
theorem store_constraints_equivalence (x : ℝ) :
  StoreConstraints x ↔ (x + (2 * x - 4) ≥ 32 ∧ 8 * x + 2 * (2 * x - 4) ≤ 148) :=
by sorry

end NUMINAMATH_CALUDE_store_constraints_equivalence_l2273_227307


namespace NUMINAMATH_CALUDE_batch_size_proof_l2273_227387

/-- The number of days it takes person A to complete the batch alone -/
def days_a : ℕ := 10

/-- The number of days it takes person B to complete the batch alone -/
def days_b : ℕ := 12

/-- The difference in parts processed by person A and B after working together for 1 day -/
def difference : ℕ := 40

/-- The total number of parts in the batch -/
def total_parts : ℕ := 2400

theorem batch_size_proof :
  (1 / days_a - 1 / days_b : ℚ) * total_parts = difference := by
  sorry

end NUMINAMATH_CALUDE_batch_size_proof_l2273_227387


namespace NUMINAMATH_CALUDE_distinct_arrangements_of_six_objects_l2273_227385

def factorial (n : ℕ) : ℕ := (List.range n).foldl (·*·) 1

theorem distinct_arrangements_of_six_objects : 
  factorial 6 = 720 := by sorry

end NUMINAMATH_CALUDE_distinct_arrangements_of_six_objects_l2273_227385


namespace NUMINAMATH_CALUDE_squirrel_nuts_theorem_l2273_227310

/-- The number of nuts found by Pizizubka -/
def pizizubka_nuts : ℕ := 48

/-- The number of nuts found by Zrzečka -/
def zrzecka_nuts : ℕ := 96

/-- The number of nuts found by Ouška -/
def ouska_nuts : ℕ := 144

/-- The fraction of nuts Pizizubka ate -/
def pizizubka_ate : ℚ := 1/2

/-- The fraction of nuts Zrzečka ate -/
def zrzecka_ate : ℚ := 1/3

/-- The fraction of nuts Ouška ate -/
def ouska_ate : ℚ := 1/4

/-- The total number of nuts left -/
def total_nuts_left : ℕ := 196

theorem squirrel_nuts_theorem :
  zrzecka_nuts = 2 * pizizubka_nuts ∧
  ouska_nuts = 3 * pizizubka_nuts ∧
  (1 - pizizubka_ate) * pizizubka_nuts +
  (1 - zrzecka_ate) * zrzecka_nuts +
  (1 - ouska_ate) * ouska_nuts = total_nuts_left :=
by sorry

end NUMINAMATH_CALUDE_squirrel_nuts_theorem_l2273_227310


namespace NUMINAMATH_CALUDE_trigonometric_expression_equality_l2273_227345

theorem trigonometric_expression_equality : 
  (Real.cos (190 * π / 180) * (1 + Real.sqrt 3 * Real.tan (10 * π / 180))) / 
  (Real.sin (290 * π / 180) * Real.sqrt (1 - Real.cos (40 * π / 180))) = 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_expression_equality_l2273_227345


namespace NUMINAMATH_CALUDE_x_varies_as_eighth_power_of_z_l2273_227361

/-- Given that x varies as the fourth power of y, and y varies as the square of z,
    prove that x varies as the 8th power of z. -/
theorem x_varies_as_eighth_power_of_z
  (k : ℝ) (j : ℝ) (x y z : ℝ → ℝ)
  (h1 : ∀ t, x t = k * (y t)^4)
  (h2 : ∀ t, y t = j * (z t)^2) :
  ∃ m : ℝ, ∀ t, x t = m * (z t)^8 := by
  sorry

end NUMINAMATH_CALUDE_x_varies_as_eighth_power_of_z_l2273_227361


namespace NUMINAMATH_CALUDE_system_solution_l2273_227314

theorem system_solution (x y z w : ℤ) : 
  x + y + z + w = 20 ∧
  y + 2*z - 3*w = 28 ∧
  x - 2*y + z = 36 ∧
  -7*x - y + 5*z + 3*w = 84 →
  x = 4 ∧ y = -6 ∧ z = 20 ∧ w = 2 := by
sorry

end NUMINAMATH_CALUDE_system_solution_l2273_227314


namespace NUMINAMATH_CALUDE_functional_equation_identity_l2273_227382

open Function

theorem functional_equation_identity (f : ℝ → ℝ) :
  (∀ x y : ℝ, f (y - f x) = f x - 2 * x + f (f y)) →
  (∀ x : ℝ, f x = x) :=
by sorry

end NUMINAMATH_CALUDE_functional_equation_identity_l2273_227382


namespace NUMINAMATH_CALUDE_total_cows_l2273_227349

theorem total_cows (cows_per_herd : ℕ) (num_herds : ℕ) 
  (h1 : cows_per_herd = 40) 
  (h2 : num_herds = 8) : 
  cows_per_herd * num_herds = 320 := by
  sorry

end NUMINAMATH_CALUDE_total_cows_l2273_227349


namespace NUMINAMATH_CALUDE_visitors_equal_cats_l2273_227330

/-- In a cat show scenario -/
structure CatShow where
  /-- The set of visitors -/
  visitors : Type
  /-- The set of cats -/
  cats : Type
  /-- The relation representing a visitor petting a cat -/
  pets : visitors → cats → Prop
  /-- Each visitor pets exactly three cats -/
  visitor_pets_three : ∀ v : visitors, ∃! (c₁ c₂ c₃ : cats), pets v c₁ ∧ pets v c₂ ∧ pets v c₃ ∧ c₁ ≠ c₂ ∧ c₁ ≠ c₃ ∧ c₂ ≠ c₃
  /-- Each cat is petted by exactly three visitors -/
  cat_petted_by_three : ∀ c : cats, ∃! (v₁ v₂ v₃ : visitors), pets v₁ c ∧ pets v₂ c ∧ pets v₃ c ∧ v₁ ≠ v₂ ∧ v₁ ≠ v₃ ∧ v₂ ≠ v₃

/-- The number of visitors is equal to the number of cats -/
theorem visitors_equal_cats (cs : CatShow) : Nonempty (Equiv cs.visitors cs.cats) := by
  sorry

end NUMINAMATH_CALUDE_visitors_equal_cats_l2273_227330


namespace NUMINAMATH_CALUDE_family_weight_gain_l2273_227341

/-- The total weight gained by three family members at a reunion --/
theorem family_weight_gain (orlando_gain jose_gain fernando_gain : ℕ) : 
  orlando_gain = 5 →
  jose_gain = 2 * orlando_gain + 2 →
  fernando_gain = jose_gain / 2 - 3 →
  orlando_gain + jose_gain + fernando_gain = 20 := by
sorry

end NUMINAMATH_CALUDE_family_weight_gain_l2273_227341


namespace NUMINAMATH_CALUDE_watermelon_seeds_l2273_227305

/-- Represents a watermelon slice with black and white seeds -/
structure WatermelonSlice where
  blackSeeds : ℕ
  whiteSeeds : ℕ

/-- Calculates the total number of seeds in a watermelon -/
def totalSeeds (slices : ℕ) (slice : WatermelonSlice) : ℕ :=
  slices * (slice.blackSeeds + slice.whiteSeeds)

/-- Theorem: The total number of seeds in the watermelon is 1600 -/
theorem watermelon_seeds :
  ∀ (slice : WatermelonSlice),
    slice.blackSeeds = 20 →
    slice.whiteSeeds = 20 →
    totalSeeds 40 slice = 1600 :=
by
  sorry

end NUMINAMATH_CALUDE_watermelon_seeds_l2273_227305


namespace NUMINAMATH_CALUDE_unique_campers_rowing_l2273_227340

theorem unique_campers_rowing (total_campers : ℕ) (morning : ℕ) (afternoon : ℕ) (evening : ℕ)
  (morning_and_afternoon : ℕ) (afternoon_and_evening : ℕ) (morning_and_evening : ℕ) (all_three : ℕ)
  (h1 : total_campers = 500)
  (h2 : morning = 235)
  (h3 : afternoon = 387)
  (h4 : evening = 142)
  (h5 : morning_and_afternoon = 58)
  (h6 : afternoon_and_evening = 23)
  (h7 : morning_and_evening = 15)
  (h8 : all_three = 8) :
  morning + afternoon + evening - (morning_and_afternoon + afternoon_and_evening + morning_and_evening) + all_three = 572 :=
by sorry

end NUMINAMATH_CALUDE_unique_campers_rowing_l2273_227340


namespace NUMINAMATH_CALUDE_swap_7_and_9_breaks_equality_l2273_227347

def original_number : ℕ := 271828
def swapped_number : ℕ := 291828
def target_sum : ℕ := 314159

def swap_digits (n : ℕ) (d1 d2 : ℕ) : ℕ := sorry

theorem swap_7_and_9_breaks_equality :
  swap_digits original_number 7 9 = swapped_number ∧
  swapped_number + original_number ≠ 2 * target_sum :=
sorry

end NUMINAMATH_CALUDE_swap_7_and_9_breaks_equality_l2273_227347


namespace NUMINAMATH_CALUDE_smallest_a_value_l2273_227399

/-- Given a polynomial x^3 - ax^2 + bx - 1890 with three positive integer roots,
    prove that the smallest possible value of a is 41 -/
theorem smallest_a_value (a b : ℤ) (x₁ x₂ x₃ : ℤ) : 
  (∀ x, x^3 - a*x^2 + b*x - 1890 = (x - x₁) * (x - x₂) * (x - x₃)) →
  x₁ > 0 → x₂ > 0 → x₃ > 0 →
  x₁ * x₂ * x₃ = 1890 →
  a = x₁ + x₂ + x₃ →
  ∀ a' : ℤ, (∃ b' x₁' x₂' x₃' : ℤ, 
    (∀ x, x^3 - a'*x^2 + b'*x - 1890 = (x - x₁') * (x - x₂') * (x - x₃')) ∧
    x₁' > 0 ∧ x₂' > 0 ∧ x₃' > 0 ∧
    x₁' * x₂' * x₃' = 1890) →
  a' ≥ 41 :=
by sorry

end NUMINAMATH_CALUDE_smallest_a_value_l2273_227399


namespace NUMINAMATH_CALUDE_sally_quarters_count_l2273_227334

/-- Given an initial number of quarters, quarters spent, and quarters found,
    calculate the final number of quarters Sally has. -/
def final_quarters (initial spent found : ℕ) : ℕ :=
  initial - spent + found

/-- Theorem stating that Sally's final number of quarters is 492 -/
theorem sally_quarters_count :
  final_quarters 760 418 150 = 492 := by
  sorry

end NUMINAMATH_CALUDE_sally_quarters_count_l2273_227334


namespace NUMINAMATH_CALUDE_inequality_and_equality_condition_l2273_227394

theorem inequality_and_equality_condition (a b c d : ℝ) 
  (pos_a : a > 0) (pos_b : b > 0) (pos_c : c > 0) (pos_d : d > 0)
  (h : a * b * c * d = 1) : 
  a^2 + b^2 + c^2 + d^2 + a*b + a*c + a*d + b*c + b*d + c*d ≥ 10 ∧ 
  (a^2 + b^2 + c^2 + d^2 + a*b + a*c + a*d + b*c + b*d + c*d = 10 ↔ a = 1 ∧ b = 1 ∧ c = 1 ∧ d = 1) :=
by sorry

end NUMINAMATH_CALUDE_inequality_and_equality_condition_l2273_227394


namespace NUMINAMATH_CALUDE_nanjing_visitors_scientific_notation_l2273_227331

/-- Represents a number in scientific notation -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  is_valid : 1 ≤ coefficient ∧ coefficient < 10

/-- Converts a real number to scientific notation -/
def toScientificNotation (x : ℝ) : ScientificNotation :=
  sorry

theorem nanjing_visitors_scientific_notation :
  toScientificNotation 44300000 = ScientificNotation.mk 4.43 7 sorry := by
  sorry

end NUMINAMATH_CALUDE_nanjing_visitors_scientific_notation_l2273_227331


namespace NUMINAMATH_CALUDE_f_properties_l2273_227375

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.exp x - (1/2) * (x + a)^2

theorem f_properties (a : ℝ) :
  (∀ x : ℝ, x ≥ 0 → f a x ≥ 0) →
  (HasDerivAt (f a) 1 0) →
  (∀ x : ℝ, StrictMono (f a)) ∧ 
  (a ≥ -Real.sqrt 2 ∧ a ≤ 2 - Real.log 2) := by
  sorry

end NUMINAMATH_CALUDE_f_properties_l2273_227375


namespace NUMINAMATH_CALUDE_total_bears_l2273_227355

/-- The number of bears in a national park --/
def bear_population (black white brown : ℕ) : ℕ :=
  black + white + brown

/-- Theorem: Given the conditions, the total bear population is 190 --/
theorem total_bears : ∀ (black white brown : ℕ),
  black = 60 →
  black = 2 * white →
  brown = black + 40 →
  bear_population black white brown = 190 := by
  sorry

end NUMINAMATH_CALUDE_total_bears_l2273_227355


namespace NUMINAMATH_CALUDE_absolute_value_of_negative_l2273_227332

theorem absolute_value_of_negative (a : ℝ) (h : a < 0) : |a| = -a := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_of_negative_l2273_227332


namespace NUMINAMATH_CALUDE_tangent_slope_at_one_l2273_227308

-- Define the function
def f (x : ℝ) : ℝ := x^2 + 2

-- State the theorem
theorem tangent_slope_at_one :
  (deriv f) 1 = 2 :=
sorry

end NUMINAMATH_CALUDE_tangent_slope_at_one_l2273_227308


namespace NUMINAMATH_CALUDE_sin_2alpha_value_l2273_227325

theorem sin_2alpha_value (α : Real) 
  (h1 : α ∈ Set.Ioo 0 (π/2)) 
  (h2 : 2 * Real.cos (2*α) = Real.cos (α - π/4)) : 
  Real.sin (2*α) = 7/8 := by
sorry

end NUMINAMATH_CALUDE_sin_2alpha_value_l2273_227325


namespace NUMINAMATH_CALUDE_min_value_of_transformed_sine_l2273_227362

theorem min_value_of_transformed_sine (φ : ℝ) (h : |φ| < π/2) :
  let f : ℝ → ℝ := λ x ↦ Real.sin (2*x - π/3)
  ∃ (x : ℝ), x ∈ Set.Icc 0 (π/2) ∧ f x = -Real.sqrt 3 / 2 ∧
    ∀ y ∈ Set.Icc 0 (π/2), f y ≥ -Real.sqrt 3 / 2 := by
  sorry


end NUMINAMATH_CALUDE_min_value_of_transformed_sine_l2273_227362


namespace NUMINAMATH_CALUDE_smallest_frood_number_l2273_227311

def sum_first_n (n : ℕ) : ℕ := n * (n + 1) / 2

theorem smallest_frood_number : 
  ∀ n : ℕ, n > 0 → (n < 10 → sum_first_n n ≤ 5 * n) ∧ (sum_first_n 10 > 5 * 10) :=
sorry

end NUMINAMATH_CALUDE_smallest_frood_number_l2273_227311


namespace NUMINAMATH_CALUDE_lcm_12_18_l2273_227388

theorem lcm_12_18 : Nat.lcm 12 18 = 36 := by
  sorry

end NUMINAMATH_CALUDE_lcm_12_18_l2273_227388


namespace NUMINAMATH_CALUDE_arithmetic_sequence_difference_l2273_227397

/-- An arithmetic sequence with specific properties -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, d > 0 ∧
  (∀ n, a n > 0) ∧
  (a 1 = 1) ∧
  (∀ n, a (n + 1) = a n + d) ∧
  (a 3 * a 11 = (a 4 + 5/2)^2)

/-- Theorem stating the difference between two terms -/
theorem arithmetic_sequence_difference
  (a : ℕ → ℝ) (m n : ℕ) (h : ArithmeticSequence a) (h_diff : m - n = 8) :
  a m - a n = 12 :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_difference_l2273_227397


namespace NUMINAMATH_CALUDE_remainder_76_pow_77_div_7_l2273_227374

theorem remainder_76_pow_77_div_7 : (76^77) % 7 = 6 := by
  sorry

end NUMINAMATH_CALUDE_remainder_76_pow_77_div_7_l2273_227374


namespace NUMINAMATH_CALUDE_right_triangle_arctan_sum_l2273_227370

/-- 
In a right triangle ABC with right angle at B, 
the sum of arctan(b/(a+c)) and arctan(c/(a+b)) equals π/4, 
where a, b, and c are the lengths of the sides opposite to angles A, B, and C respectively.
-/
theorem right_triangle_arctan_sum (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (right_angle : a^2 = b^2 + c^2) : 
  Real.arctan (b / (a + c)) + Real.arctan (c / (a + b)) = π / 4 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_arctan_sum_l2273_227370


namespace NUMINAMATH_CALUDE_simplify_fraction_l2273_227338

theorem simplify_fraction (a : ℝ) (h1 : a ≠ 1) (h2 : a ≠ -1) :
  ((3 * a / (a^2 - 1) - 1 / (a - 1)) / ((2 * a - 1) / (a + 1))) = 1 / (a - 1) := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_l2273_227338


namespace NUMINAMATH_CALUDE_triangle_side_length_l2273_227398

theorem triangle_side_length (A B C : Real) (a b c : Real) :
  b = 2 * Real.sqrt 3 →
  B = 2 * π / 3 →
  C = π / 6 →
  a = 2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_side_length_l2273_227398


namespace NUMINAMATH_CALUDE_solution_set_when_a_is_one_range_of_a_when_f_leq_one_l2273_227323

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := 5 - |x + a| - |x - 2|

theorem solution_set_when_a_is_one :
  {x : ℝ | f 1 x ≥ 0} = {x : ℝ | -2 ≤ x ∧ x ≤ 3} := by sorry

theorem range_of_a_when_f_leq_one :
  {a : ℝ | ∀ x, f a x ≤ 1} = {a : ℝ | a ≤ -6 ∨ a ≥ 2} := by sorry

end NUMINAMATH_CALUDE_solution_set_when_a_is_one_range_of_a_when_f_leq_one_l2273_227323


namespace NUMINAMATH_CALUDE_arithmetic_geometric_inequality_l2273_227351

theorem arithmetic_geometric_inequality (a b : ℝ) :
  (a + b) / 2 ≥ Real.sqrt (a * b) ∧
  ((a + b) / 2 = Real.sqrt (a * b) ↔ a = b) := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_inequality_l2273_227351


namespace NUMINAMATH_CALUDE_petes_number_l2273_227306

theorem petes_number : ∃ x : ℚ, 3 * (3 * x - 5) = 96 ∧ x = 111 / 9 := by
  sorry

end NUMINAMATH_CALUDE_petes_number_l2273_227306


namespace NUMINAMATH_CALUDE_line_segment_param_sum_l2273_227371

/-- Given a line segment connecting (1, -3) and (-4, 5), parameterized by x = pt + q and y = rt + s
    where 0 ≤ t ≤ 1 and t = 0 corresponds to (1, -3), prove that p^2 + q^2 + r^2 + s^2 = 99. -/
theorem line_segment_param_sum (p q r s : ℝ) : 
  (∀ t : ℝ, 0 ≤ t ∧ t ≤ 1 → ∃ x y : ℝ, x = p * t + q ∧ y = r * t + s) →
  (q = 1 ∧ s = -3) →
  (p + q = -4 ∧ r + s = 5) →
  p^2 + q^2 + r^2 + s^2 = 99 := by
sorry

end NUMINAMATH_CALUDE_line_segment_param_sum_l2273_227371


namespace NUMINAMATH_CALUDE_card_game_proof_l2273_227326

def deck_size : ℕ := 60
def hand_size : ℕ := 12

theorem card_game_proof :
  let combinations := Nat.choose deck_size hand_size
  ∃ (B : ℕ), 
    B = 7 ∧ 
    combinations = 17 * 10^10 + B * 10^9 + B * 10^7 + 5 * 10^6 + 2 * 10^5 + 9 * 10^4 + 8 * 10 + B ∧
    combinations % 6 = 0 := by
  sorry

end NUMINAMATH_CALUDE_card_game_proof_l2273_227326


namespace NUMINAMATH_CALUDE_stratified_sample_correct_l2273_227376

/-- Represents the three age groups in the population -/
inductive AgeGroup
  | Elderly
  | MiddleAged
  | Young

/-- Represents the population sizes for each age group -/
def populationSize (group : AgeGroup) : Nat :=
  match group with
  | .Elderly => 25
  | .MiddleAged => 35
  | .Young => 40

/-- The total population size -/
def totalPopulation : Nat := populationSize .Elderly + populationSize .MiddleAged + populationSize .Young

/-- The desired sample size -/
def sampleSize : Nat := 40

/-- Calculates the stratified sample size for a given age group -/
def stratifiedSampleSize (group : AgeGroup) : Nat :=
  (populationSize group * sampleSize) / totalPopulation

/-- Theorem stating that the stratified sample sizes are correct -/
theorem stratified_sample_correct :
  stratifiedSampleSize .Elderly = 10 ∧
  stratifiedSampleSize .MiddleAged = 14 ∧
  stratifiedSampleSize .Young = 16 ∧
  stratifiedSampleSize .Elderly + stratifiedSampleSize .MiddleAged + stratifiedSampleSize .Young = sampleSize :=
by sorry

end NUMINAMATH_CALUDE_stratified_sample_correct_l2273_227376


namespace NUMINAMATH_CALUDE_intersection_point_is_solution_l2273_227396

theorem intersection_point_is_solution (k : ℝ) (hk : k ≠ 0) :
  (∃ (x y : ℝ), y = 2 * x - 1 ∧ y = k * x ∧ x = 1 ∧ y = 1) →
  (∃! (x y : ℝ), 2 * x - y = 1 ∧ k * x - y = 0 ∧ x = 1 ∧ y = 1) := by
  sorry

end NUMINAMATH_CALUDE_intersection_point_is_solution_l2273_227396


namespace NUMINAMATH_CALUDE_base_with_six_digits_for_256_l2273_227365

theorem base_with_six_digits_for_256 :
  ∃! (b : ℕ), b > 0 ∧ b^5 ≤ 256 ∧ 256 < b^6 :=
by
  sorry

end NUMINAMATH_CALUDE_base_with_six_digits_for_256_l2273_227365


namespace NUMINAMATH_CALUDE_pyramid_base_edge_length_l2273_227353

theorem pyramid_base_edge_length 
  (r : ℝ) 
  (h : ℝ) 
  (hemisphere_radius : r = 3) 
  (pyramid_height : h = 4) 
  (hemisphere_tangent : True)  -- This represents the tangency condition
  : ∃ (s : ℝ), s = (12 * Real.sqrt 14) / 7 ∧ s > 0 := by
  sorry

end NUMINAMATH_CALUDE_pyramid_base_edge_length_l2273_227353


namespace NUMINAMATH_CALUDE_triangle_angle_measure_l2273_227354

theorem triangle_angle_measure (A B C : ℝ) : 
  A > 0 → B > 0 → C > 0 →
  A + B + C = 180 →
  C = 2 * B →
  B = A / 3 →
  A = 90 := by sorry

end NUMINAMATH_CALUDE_triangle_angle_measure_l2273_227354


namespace NUMINAMATH_CALUDE_circle_area_equals_square_perimeter_l2273_227379

theorem circle_area_equals_square_perimeter (side_length : ℝ) (radius : ℝ) : 
  side_length = 25 → 4 * side_length = Real.pi * radius^2 → Real.pi * radius^2 = 100 :=
by sorry

end NUMINAMATH_CALUDE_circle_area_equals_square_perimeter_l2273_227379


namespace NUMINAMATH_CALUDE_fred_red_marbles_l2273_227384

/-- Represents the number of marbles of each color --/
structure MarbleCount where
  red : ℕ
  green : ℕ
  blue : ℕ

/-- The conditions of Fred's marble collection --/
def fredMarbles (m : MarbleCount) : Prop :=
  m.red + m.green + m.blue = 63 ∧
  m.blue = 6 ∧
  m.green = m.red / 2

/-- Theorem stating that Fred has 38 red marbles --/
theorem fred_red_marbles :
  ∃ m : MarbleCount, fredMarbles m ∧ m.red = 38 := by
  sorry

end NUMINAMATH_CALUDE_fred_red_marbles_l2273_227384


namespace NUMINAMATH_CALUDE_book_club_monthly_books_l2273_227309

def prove_book_club_monthly_books (initial_books final_books bookstore_purchase yard_sale_purchase 
  daughter_gift mother_gift donated_books sold_books : ℕ) : Prop :=
  let total_acquired := bookstore_purchase + yard_sale_purchase + daughter_gift + mother_gift
  let total_removed := donated_books + sold_books
  let net_change := final_books - initial_books
  let book_club_total := net_change + total_removed - total_acquired
  (book_club_total % 12 = 0) ∧ (book_club_total / 12 = 1)

theorem book_club_monthly_books :
  prove_book_club_monthly_books 72 81 5 2 1 4 12 3 :=
sorry

end NUMINAMATH_CALUDE_book_club_monthly_books_l2273_227309


namespace NUMINAMATH_CALUDE_find_b_value_l2273_227301

theorem find_b_value (a b : ℝ) (eq1 : 3 * a + 2 = 2) (eq2 : b - 2 * a = 2) : b = 2 := by
  sorry

end NUMINAMATH_CALUDE_find_b_value_l2273_227301


namespace NUMINAMATH_CALUDE_golden_apples_weight_l2273_227368

theorem golden_apples_weight (kidney_apples : ℕ) (canada_apples : ℕ) (sold_apples : ℕ) (left_apples : ℕ) 
  (h1 : kidney_apples = 23)
  (h2 : canada_apples = 14)
  (h3 : sold_apples = 36)
  (h4 : left_apples = 38) :
  ∃ golden_apples : ℕ, 
    kidney_apples + golden_apples + canada_apples = sold_apples + left_apples ∧ 
    golden_apples = 37 := by
  sorry

end NUMINAMATH_CALUDE_golden_apples_weight_l2273_227368


namespace NUMINAMATH_CALUDE_r_plus_s_equals_six_l2273_227329

theorem r_plus_s_equals_six (r s : ℕ) (h1 : 2^r = 16) (h2 : 5^s = 25) : r + s = 6 := by
  sorry

end NUMINAMATH_CALUDE_r_plus_s_equals_six_l2273_227329


namespace NUMINAMATH_CALUDE_square_of_number_ending_in_five_l2273_227328

theorem square_of_number_ending_in_five (n : ℕ) :
  ∃ k : ℕ, n = 10 * k + 5 →
    (n^2 % 100 = 25) ∧
    (n^2 = 100 * (k * (k + 1)) + 25) := by
  sorry

end NUMINAMATH_CALUDE_square_of_number_ending_in_five_l2273_227328


namespace NUMINAMATH_CALUDE_total_people_in_program_l2273_227336

theorem total_people_in_program (parents pupils teachers : ℕ) 
  (h1 : parents = 73)
  (h2 : pupils = 724)
  (h3 : teachers = 744) :
  parents + pupils + teachers = 1541 := by
  sorry

end NUMINAMATH_CALUDE_total_people_in_program_l2273_227336


namespace NUMINAMATH_CALUDE_tripled_base_and_exponent_l2273_227303

theorem tripled_base_and_exponent (a b x : ℝ) (hb : b ≠ 0) :
  (3 * a) ^ (3 * b) = a ^ b * x ^ (2 * b) → x = 3 * Real.sqrt 3 * a := by
  sorry

end NUMINAMATH_CALUDE_tripled_base_and_exponent_l2273_227303


namespace NUMINAMATH_CALUDE_multiplicative_inverse_modulo_l2273_227327

def A : Nat := 123456
def B : Nat := 142857
def M : Nat := 1000000

theorem multiplicative_inverse_modulo :
  (892857 * (A * B)) % M = 1 := by sorry

end NUMINAMATH_CALUDE_multiplicative_inverse_modulo_l2273_227327


namespace NUMINAMATH_CALUDE_hex_BF02_eq_48898_l2273_227352

/-- Converts a single hexadecimal digit to its decimal value -/
def hex_to_dec (c : Char) : ℕ :=
  match c with
  | 'B' => 11
  | 'F' => 15
  | '0' => 0
  | '2' => 2
  | _ => 0  -- Default case, should not be reached for this problem

/-- Converts a hexadecimal number represented as a string to its decimal value -/
def hex_to_decimal (s : String) : ℕ :=
  s.foldr (fun c acc => hex_to_dec c + 16 * acc) 0

/-- The hexadecimal number BF02 is equal to 48898 in decimal -/
theorem hex_BF02_eq_48898 : hex_to_decimal "BF02" = 48898 := by
  sorry

end NUMINAMATH_CALUDE_hex_BF02_eq_48898_l2273_227352


namespace NUMINAMATH_CALUDE_min_value_theorem_l2273_227346

theorem min_value_theorem (x y z : ℝ) (h : x^3 + y^3 + z^3 - 3*x*y*z = 8) :
  ∃ (m : ℝ), m = 40/3 ∧ ∀ (a b c : ℝ), a^3 + b^3 + c^3 - 3*a*b*c = 8 → 
    a^2 + b^2 + c^2 + 2*a*c ≥ m ∧ 
    (∃ (p q r : ℝ), p^3 + q^3 + r^3 - 3*p*q*r = 8 ∧ p^2 + q^2 + r^2 + 2*p*r = m) :=
by sorry

end NUMINAMATH_CALUDE_min_value_theorem_l2273_227346


namespace NUMINAMATH_CALUDE_complex_magnitude_one_l2273_227337

theorem complex_magnitude_one (z : ℂ) (h : 11 * z^10 + 10*Complex.I * z^9 + 10*Complex.I * z - 11 = 0) : Complex.abs z = 1 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_one_l2273_227337


namespace NUMINAMATH_CALUDE_jessie_final_position_l2273_227364

/-- The number of steps Jessie takes in total -/
def total_steps : ℕ := 6

/-- The final position Jessie reaches -/
def final_position : ℕ := 24

/-- The number of steps to reach point x -/
def steps_to_x : ℕ := 4

/-- The number of steps from x to z -/
def steps_x_to_z : ℕ := 1

/-- The number of steps from z to y -/
def steps_z_to_y : ℕ := 1

/-- The length of each step -/
def step_length : ℚ := final_position / total_steps

/-- The position of point x -/
def x : ℚ := step_length * steps_to_x

/-- The position of point z -/
def z : ℚ := x + step_length * steps_x_to_z

/-- The position of point y -/
def y : ℚ := z + step_length * steps_z_to_y

theorem jessie_final_position : y = 24 := by
  sorry

end NUMINAMATH_CALUDE_jessie_final_position_l2273_227364


namespace NUMINAMATH_CALUDE_product_of_squares_l2273_227372

theorem product_of_squares (x y z : ℚ) 
  (hx : x = 1/4) 
  (hy : y = 1/2) 
  (hz : z = -8) : 
  x^2 * y^2 * z^2 = 1 := by sorry

end NUMINAMATH_CALUDE_product_of_squares_l2273_227372


namespace NUMINAMATH_CALUDE_equivalent_expression_l2273_227358

theorem equivalent_expression (x : ℝ) (hx : x < 0) :
  Real.sqrt ((x + 1) / (1 - (x - 2) / x)) = Complex.I * Real.sqrt (-((x^2 + x) / 2)) :=
by sorry

end NUMINAMATH_CALUDE_equivalent_expression_l2273_227358


namespace NUMINAMATH_CALUDE_zongzi_survey_measure_l2273_227302

-- Define the types of statistical measures
inductive StatMeasure
| Variance
| Mean
| Median
| Mode

-- Define a function that determines the most appropriate measure
def most_appropriate_measure (survey_goal : String) (data_type : String) : StatMeasure :=
  if survey_goal = "determine most preferred" && data_type = "categorical" then
    StatMeasure.Mode
  else
    StatMeasure.Mean  -- Default to mean for other cases

-- Theorem statement
theorem zongzi_survey_measure :
  most_appropriate_measure "determine most preferred" "categorical" = StatMeasure.Mode :=
by sorry

end NUMINAMATH_CALUDE_zongzi_survey_measure_l2273_227302


namespace NUMINAMATH_CALUDE_files_per_folder_l2273_227318

theorem files_per_folder (initial_files : ℝ) (additional_files : ℝ) (num_folders : ℝ) :
  let total_files := initial_files + additional_files
  total_files / num_folders = (initial_files + additional_files) / num_folders :=
by sorry

end NUMINAMATH_CALUDE_files_per_folder_l2273_227318


namespace NUMINAMATH_CALUDE_biff_hourly_rate_l2273_227322

/-- Biff's bus trip expenses and earnings -/
def biff_trip (hourly_rate : ℚ) : Prop :=
  let ticket : ℚ := 11
  let snacks : ℚ := 3
  let headphones : ℚ := 16
  let wifi_rate : ℚ := 2
  let trip_duration : ℚ := 3
  let total_expenses : ℚ := ticket + snacks + headphones + wifi_rate * trip_duration
  hourly_rate * trip_duration = total_expenses

/-- Theorem stating Biff's hourly rate for online work -/
theorem biff_hourly_rate : 
  ∃ (rate : ℚ), biff_trip rate ∧ rate = 12 := by
  sorry

end NUMINAMATH_CALUDE_biff_hourly_rate_l2273_227322


namespace NUMINAMATH_CALUDE_simplify_fraction_product_l2273_227316

theorem simplify_fraction_product : 5 * (14 / 9) * (27 / -63) = -30 := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_product_l2273_227316


namespace NUMINAMATH_CALUDE_simplify_sqrt_expression_l2273_227359

theorem simplify_sqrt_expression :
  Real.sqrt 12 + 3 * Real.sqrt (1/3) = 3 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_simplify_sqrt_expression_l2273_227359
