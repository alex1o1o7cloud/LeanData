import Mathlib

namespace NUMINAMATH_CALUDE_yard_area_l1492_149271

/-- The area of a rectangular yard with a rectangular cutout -/
theorem yard_area (length width cutout_length cutout_width : ℝ) 
  (h1 : length = 20)
  (h2 : width = 15)
  (h3 : cutout_length = 4)
  (h4 : cutout_width = 2) :
  length * width - cutout_length * cutout_width = 292 := by
  sorry

end NUMINAMATH_CALUDE_yard_area_l1492_149271


namespace NUMINAMATH_CALUDE_debate_only_count_l1492_149232

/-- Represents the number of pupils in a class with debate and singing activities -/
structure ClassActivities where
  total : ℕ
  singing_only : ℕ
  both : ℕ
  debate_only : ℕ

/-- The number of pupils in debate only is 37 -/
theorem debate_only_count (c : ClassActivities) 
  (h1 : c.total = 55)
  (h2 : c.singing_only = 18)
  (h3 : c.both = 17)
  (h4 : c.total = c.debate_only + c.singing_only + c.both) : 
  c.debate_only = 37 := by
  sorry

end NUMINAMATH_CALUDE_debate_only_count_l1492_149232


namespace NUMINAMATH_CALUDE_lost_episodes_proof_l1492_149249

/-- Represents the number of episodes lost per season after a computer failure --/
def episodes_lost_per_season (series1_seasons series2_seasons episodes_per_season remaining_episodes : ℕ) : ℕ :=
  let total_episodes := (series1_seasons + series2_seasons) * episodes_per_season
  let lost_episodes := total_episodes - remaining_episodes
  lost_episodes / (series1_seasons + series2_seasons)

/-- Theorem stating that given the problem conditions, 2 episodes were lost per season --/
theorem lost_episodes_proof :
  episodes_lost_per_season 12 14 16 364 = 2 := by
  sorry

end NUMINAMATH_CALUDE_lost_episodes_proof_l1492_149249


namespace NUMINAMATH_CALUDE_rectangular_prism_properties_l1492_149243

/-- A rectangular prism with dimensions 12, 16, and 21 inches has a diagonal length of 29 inches
    and a surface area of 1560 square inches. -/
theorem rectangular_prism_properties :
  let a : ℝ := 12
  let b : ℝ := 16
  let c : ℝ := 21
  let diagonal := Real.sqrt (a^2 + b^2 + c^2)
  let surface_area := 2 * (a*b + b*c + c*a)
  diagonal = 29 ∧ surface_area = 1560 := by
  sorry

#check rectangular_prism_properties

end NUMINAMATH_CALUDE_rectangular_prism_properties_l1492_149243


namespace NUMINAMATH_CALUDE_selling_price_equal_profit_loss_l1492_149277

def cost_price : ℕ := 49
def loss_price : ℕ := 42

def profit (selling_price : ℕ) : ℤ := selling_price - cost_price
def loss (selling_price : ℕ) : ℤ := cost_price - selling_price

theorem selling_price_equal_profit_loss : 
  ∃ (sp : ℕ), profit sp = loss loss_price ∧ profit sp > 0 := by
  sorry

end NUMINAMATH_CALUDE_selling_price_equal_profit_loss_l1492_149277


namespace NUMINAMATH_CALUDE_sphere_triangle_distance_theorem_l1492_149241

/-- The distance between the center of a sphere and the plane of a right triangle tangent to the sphere. -/
def sphere_triangle_distance (sphere_radius : ℝ) (triangle_side1 triangle_side2 triangle_side3 : ℝ) : ℝ :=
  sorry

/-- Theorem stating the distance between the center of a sphere and the plane of a right triangle tangent to the sphere. -/
theorem sphere_triangle_distance_theorem :
  sphere_triangle_distance 10 8 15 17 = Real.sqrt 91 := by
  sorry

end NUMINAMATH_CALUDE_sphere_triangle_distance_theorem_l1492_149241


namespace NUMINAMATH_CALUDE_ellen_lego_count_l1492_149235

/-- Calculates the final number of legos Ellen has after a series of transactions -/
def final_lego_count (initial : ℕ) : ℕ :=
  let after_week1 := initial - initial / 5
  let after_week2 := after_week1 + after_week1 / 4
  let after_week3 := after_week2 - 57
  after_week3 + after_week3 / 10

/-- Theorem stating that Ellen ends up with 355 legos -/
theorem ellen_lego_count : final_lego_count 380 = 355 := by
  sorry


end NUMINAMATH_CALUDE_ellen_lego_count_l1492_149235


namespace NUMINAMATH_CALUDE_exponent_division_l1492_149255

theorem exponent_division (a : ℝ) (h : a ≠ 0) : a^3 / a = a^2 :=
sorry

end NUMINAMATH_CALUDE_exponent_division_l1492_149255


namespace NUMINAMATH_CALUDE_ninth_grade_class_problem_l1492_149270

theorem ninth_grade_class_problem (total : ℕ) (science : ℕ) (arts : ℕ) 
  (h_total : total = 120)
  (h_science : science = 85)
  (h_arts : arts = 65)
  (h_covers_all : total ≤ science + arts) :
  science - (science + arts - total) = 55 := by
  sorry

end NUMINAMATH_CALUDE_ninth_grade_class_problem_l1492_149270


namespace NUMINAMATH_CALUDE_smallest_multiple_l1492_149269

theorem smallest_multiple (x : ℕ+) : (∀ y : ℕ+, 450 * y.val % 625 = 0 → x ≤ y) ∧ 450 * x.val % 625 = 0 := by
  sorry

end NUMINAMATH_CALUDE_smallest_multiple_l1492_149269


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l1492_149297

theorem sufficient_not_necessary (a b : ℝ) : 
  (a < 0 ∧ -1 < b ∧ b < 0) → 
  (∀ x y : ℝ, x < 0 ∧ -1 < y ∧ y < 0 → x + x * y < 0) ∧
  ¬(∀ x y : ℝ, x + x * y < 0 → x < 0 ∧ -1 < y ∧ y < 0) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l1492_149297


namespace NUMINAMATH_CALUDE_investment_growth_rate_l1492_149254

theorem investment_growth_rate (initial_investment : ℝ) (final_investment : ℝ) (x : ℝ) :
  initial_investment = 2500 ∧ 
  final_investment = 3600 ∧ 
  final_investment = initial_investment * (1 + x)^2 →
  2500 * (1 + x)^2 = 3600 :=
by sorry

end NUMINAMATH_CALUDE_investment_growth_rate_l1492_149254


namespace NUMINAMATH_CALUDE_range_of_a_l1492_149258

-- Define the function
def f (a : ℝ) (x : ℝ) : ℝ := x^2 - a*x + 65

-- State the theorem
theorem range_of_a (a : ℝ) : 
  (∀ x, f a x > 0) → -16 < a ∧ a < 16 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l1492_149258


namespace NUMINAMATH_CALUDE_loser_received_35_percent_l1492_149291

/-- Given a total number of votes and the difference between winner and loser,
    calculate the percentage of votes received by the losing candidate. -/
def loser_vote_percentage (total_votes : ℕ) (vote_difference : ℕ) : ℚ :=
  (total_votes - vote_difference) / (2 * total_votes) * 100

/-- Theorem stating that given 4500 total votes and a 1350 vote difference,
    the losing candidate received 35% of the votes. -/
theorem loser_received_35_percent :
  loser_vote_percentage 4500 1350 = 35 := by
  sorry

end NUMINAMATH_CALUDE_loser_received_35_percent_l1492_149291


namespace NUMINAMATH_CALUDE_sin_cos_shift_l1492_149218

theorem sin_cos_shift (x : ℝ) : 
  Real.sin (2 * x + π / 3) = Real.cos (2 * (x + π / 12) - π / 3) := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_shift_l1492_149218


namespace NUMINAMATH_CALUDE_geometric_figure_pieces_l1492_149261

/-- Calculates the sum of the first n natural numbers -/
def triangular_number (n : ℕ) : ℕ := n * (n + 1) / 2

/-- Represents the number of rows in the geometric figure -/
def num_rows : ℕ := 10

/-- Calculates the number of rods in the geometric figure -/
def num_rods : ℕ := 3 * triangular_number num_rows

/-- Calculates the number of connectors in the geometric figure -/
def num_connectors : ℕ := triangular_number (num_rows + 1)

/-- Calculates the number of unit squares in the geometric figure -/
def num_squares : ℕ := triangular_number num_rows

/-- The total number of pieces in the geometric figure -/
def total_pieces : ℕ := num_rods + num_connectors + num_squares

theorem geometric_figure_pieces :
  total_pieces = 286 :=
sorry

end NUMINAMATH_CALUDE_geometric_figure_pieces_l1492_149261


namespace NUMINAMATH_CALUDE_triangle_side_length_l1492_149231

/-- Given a triangle ABC with sides a, b, and c opposite to angles A, B, and C respectively,
    if a = 4√5, b = 5, and cos A = 3/5, then c = 11. -/
theorem triangle_side_length (a b c : ℝ) (A : ℝ) :
  a = 4 * Real.sqrt 5 →
  b = 5 →
  Real.cos A = 3 / 5 →
  c = 11 := by
  sorry

end NUMINAMATH_CALUDE_triangle_side_length_l1492_149231


namespace NUMINAMATH_CALUDE_coefficient_x_squared_l1492_149260

theorem coefficient_x_squared (n : ℕ) : 
  (2 : ℤ) * 4 * (Nat.choose 6 2) - 2 * (Nat.choose 6 1) + 1 = 109 := by
  sorry

#check coefficient_x_squared

end NUMINAMATH_CALUDE_coefficient_x_squared_l1492_149260


namespace NUMINAMATH_CALUDE_complement_implies_a_value_l1492_149246

def I (a : ℤ) : Set ℤ := {2, 4, a^2 - a - 3}
def A (a : ℤ) : Set ℤ := {4, 1 - a}

theorem complement_implies_a_value (a : ℤ) :
  (I a) \ (A a) = {-1} → a = -1 := by
  sorry

end NUMINAMATH_CALUDE_complement_implies_a_value_l1492_149246


namespace NUMINAMATH_CALUDE_place_values_and_names_l1492_149247

/-- Represents a place in a base-10 positional number system -/
inductive Place : Nat → Type where
  | units : Place 1
  | next (n : Nat) : Place n → Place (n + 1)

/-- The value of a place in a base-10 positional number system -/
def placeValue : ∀ n, Place n → Nat
  | _, Place.units => 1
  | _, Place.next _ p => 10 * placeValue _ p

/-- The name of a place in a base-10 positional number system -/
def placeName : ∀ n, Place n → String
  | _, Place.units => "units"
  | _, Place.next _ p =>
    let prev := placeName _ p
    if prev = "units" then "tens"
    else if prev = "tens" then "hundreds"
    else if prev = "hundreds" then "thousands"
    else if prev = "thousands" then "ten thousands"
    else if prev = "ten thousands" then "hundred thousands"
    else if prev = "hundred thousands" then "millions"
    else if prev = "millions" then "ten millions"
    else if prev = "ten millions" then "hundred millions"
    else "billion"

theorem place_values_and_names :
  ∃ (fifth tenth : Nat) (p5 : Place fifth) (p10 : Place tenth),
    fifth = 5 ∧
    tenth = 10 ∧
    placeName _ p5 = "ten thousands" ∧
    placeName _ p10 = "billion" ∧
    placeValue _ p5 = 10000 ∧
    placeValue _ p10 = 1000000000 := by
  sorry

end NUMINAMATH_CALUDE_place_values_and_names_l1492_149247


namespace NUMINAMATH_CALUDE_divisibility_by_37_l1492_149282

theorem divisibility_by_37 : ∃ k : ℤ, 333^555 + 555^333 = 37 * k := by sorry

end NUMINAMATH_CALUDE_divisibility_by_37_l1492_149282


namespace NUMINAMATH_CALUDE_sum_60_is_neg_300_l1492_149252

/-- An arithmetic progression with specific properties -/
structure ArithmeticProgression where
  /-- The first term of the progression -/
  a : ℚ
  /-- The common difference of the progression -/
  d : ℚ
  /-- The sum of the first 15 terms is 150 -/
  sum_15 : (15 : ℚ) / 2 * (2 * a + 14 * d) = 150
  /-- The sum of the first 45 terms is 0 -/
  sum_45 : (45 : ℚ) / 2 * (2 * a + 44 * d) = 0

/-- The sum of the first 60 terms of the arithmetic progression is -300 -/
theorem sum_60_is_neg_300 (ap : ArithmeticProgression) :
  (60 : ℚ) / 2 * (2 * ap.a + 59 * ap.d) = -300 := by
  sorry


end NUMINAMATH_CALUDE_sum_60_is_neg_300_l1492_149252


namespace NUMINAMATH_CALUDE_square_side_length_l1492_149227

theorem square_side_length (rectangle_width : ℝ) (rectangle_length : ℝ) 
  (h1 : rectangle_width = 3)
  (h2 : rectangle_length = 3)
  (h3 : square_area = rectangle_width * rectangle_length) : 
  ∃ (square_side : ℝ), square_side^2 = square_area ∧ square_side = 3 := by
  sorry

end NUMINAMATH_CALUDE_square_side_length_l1492_149227


namespace NUMINAMATH_CALUDE_matrix_equation_solution_l1492_149250

theorem matrix_equation_solution : 
  let M : Matrix (Fin 2) (Fin 2) ℝ := !![1, 2; 0.5, 1]
  M^4 - 3 • M^3 + 2 • M^2 = !![6, 12; 3, 6] := by
  sorry

end NUMINAMATH_CALUDE_matrix_equation_solution_l1492_149250


namespace NUMINAMATH_CALUDE_sister_chromatid_separation_in_second_division_sister_chromatid_separation_not_in_other_stages_l1492_149257

/-- Represents the stages of meiosis --/
inductive MeiosisStage
  | Interphase
  | TetradFormation
  | FirstDivision
  | SecondDivision

/-- Represents the events that occur during meiosis --/
inductive MeiosisEvent
  | ChromosomeReplication
  | HomologousPairing
  | ChromatidSeparation

/-- Defines the characteristics of each meiosis stage --/
def stageCharacteristics : MeiosisStage → List MeiosisEvent
  | MeiosisStage.Interphase => [MeiosisEvent.ChromosomeReplication]
  | MeiosisStage.TetradFormation => [MeiosisEvent.HomologousPairing]
  | MeiosisStage.FirstDivision => []
  | MeiosisStage.SecondDivision => [MeiosisEvent.ChromatidSeparation]

/-- Theorem: Sister chromatid separation occurs during the second meiotic division --/
theorem sister_chromatid_separation_in_second_division :
  MeiosisEvent.ChromatidSeparation ∈ stageCharacteristics MeiosisStage.SecondDivision :=
by sorry

/-- Corollary: Sister chromatid separation does not occur in other stages --/
theorem sister_chromatid_separation_not_in_other_stages :
  ∀ stage, stage ≠ MeiosisStage.SecondDivision →
    MeiosisEvent.ChromatidSeparation ∉ stageCharacteristics stage :=
by sorry

end NUMINAMATH_CALUDE_sister_chromatid_separation_in_second_division_sister_chromatid_separation_not_in_other_stages_l1492_149257


namespace NUMINAMATH_CALUDE_price_reduction_l1492_149221

theorem price_reduction (initial_price : ℝ) (first_reduction : ℝ) (second_reduction : ℝ) :
  first_reduction = 0.15 →
  second_reduction = 0.20 →
  let price_after_first := initial_price * (1 - first_reduction)
  let price_after_second := price_after_first * (1 - second_reduction)
  initial_price > 0 →
  (initial_price - price_after_second) / initial_price = 0.32 := by
  sorry

end NUMINAMATH_CALUDE_price_reduction_l1492_149221


namespace NUMINAMATH_CALUDE_middle_school_sample_size_l1492_149266

/-- Represents the number of schools to be sampled in a stratified sampling scenario -/
def stratified_sample (total : ℕ) (category : ℕ) (sample_size : ℕ) : ℕ :=
  (category * sample_size) / total

/-- Theorem stating the correct number of middle schools to be sampled -/
theorem middle_school_sample_size :
  let total_schools : ℕ := 700
  let middle_schools : ℕ := 200
  let sample_size : ℕ := 70
  stratified_sample total_schools middle_schools sample_size = 20 := by
  sorry


end NUMINAMATH_CALUDE_middle_school_sample_size_l1492_149266


namespace NUMINAMATH_CALUDE_can_measure_fifteen_minutes_l1492_149239

/-- Represents an hourglass with a specific duration. -/
structure Hourglass where
  duration : ℕ

/-- Represents the state of measuring time with two hourglasses. -/
structure MeasurementState where
  time : ℕ
  hg7 : ℕ
  hg11 : ℕ

/-- Defines a single step in the measurement process. -/
inductive MeasurementStep
  | FlipHg7
  | FlipHg11
  | Wait

/-- Applies a measurement step to the current state. -/
def applyStep (state : MeasurementState) (step : MeasurementStep) : MeasurementState :=
  sorry

/-- Checks if the given sequence of steps results in exactly 15 minutes. -/
def measuresFifteenMinutes (steps : List MeasurementStep) : Prop :=
  sorry

/-- Theorem stating that it's possible to measure 15 minutes with 7 and 11-minute hourglasses. -/
theorem can_measure_fifteen_minutes :
  ∃ (steps : List MeasurementStep), measuresFifteenMinutes steps :=
  sorry

end NUMINAMATH_CALUDE_can_measure_fifteen_minutes_l1492_149239


namespace NUMINAMATH_CALUDE_odd_decreasing_function_range_l1492_149267

def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

def is_decreasing (f : ℝ → ℝ) (a b : ℝ) : Prop := 
  ∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → f x > f y

theorem odd_decreasing_function_range 
  (f : ℝ → ℝ) 
  (h_odd : is_odd f) 
  (h_decreasing : is_decreasing f (-1) 1) 
  (h_condition : ∀ a, f (1 - a) + (f (1 - a))^2 > 0) :
  ∃ a, 1 < a ∧ a ≤ Real.sqrt 2 :=
sorry

end NUMINAMATH_CALUDE_odd_decreasing_function_range_l1492_149267


namespace NUMINAMATH_CALUDE_justin_and_tim_games_l1492_149289

/-- The number of players in the league -/
def total_players : ℕ := 10

/-- The number of players in each game -/
def players_per_game : ℕ := 5

/-- The number of games where two specific players play together -/
def games_together : ℕ := 56

/-- The total number of possible game combinations -/
def total_combinations : ℕ := Nat.choose total_players players_per_game

theorem justin_and_tim_games :
  games_together = (players_per_game - 1) * total_combinations / (total_players - 1) :=
sorry

end NUMINAMATH_CALUDE_justin_and_tim_games_l1492_149289


namespace NUMINAMATH_CALUDE_probability_of_perfect_square_l1492_149284

theorem probability_of_perfect_square :
  ∀ (P : ℝ),
  (P * 50 + 3 * P * 50 = 1) →
  (∃ (perfect_squares_le_50 perfect_squares_gt_50 : ℕ),
    perfect_squares_le_50 = 7 ∧ perfect_squares_gt_50 = 3) →
  (perfect_squares_le_50 * P + perfect_squares_gt_50 * 3 * P) / 100 = 0.08 :=
by sorry

end NUMINAMATH_CALUDE_probability_of_perfect_square_l1492_149284


namespace NUMINAMATH_CALUDE_sum_equals_932_l1492_149229

-- Define the value of a number in a given base
def value_in_base (digits : List ℕ) (base : ℕ) : ℕ :=
  digits.enum.foldr (fun (i, d) acc => acc + d * base^i) 0

-- Define the given numbers
def num1 : ℕ := value_in_base [3, 5, 1] 7
def num2 : ℕ := value_in_base [13, 12, 4] 13

-- Theorem to prove
theorem sum_equals_932 : num1 + num2 = 932 := by
  sorry

end NUMINAMATH_CALUDE_sum_equals_932_l1492_149229


namespace NUMINAMATH_CALUDE_unique_intersecting_line_l1492_149209

/-- A line in 3D space -/
structure Line3D where
  -- Define a line using two points
  point1 : ℝ × ℝ × ℝ
  point2 : ℝ × ℝ × ℝ
  ne : point1 ≠ point2

/-- Two lines are skew if they are not parallel and do not intersect -/
def are_skew (l1 l2 : Line3D) : Prop :=
  -- Definition of skew lines
  sorry

/-- A line intersects another line -/
def intersects (l1 l2 : Line3D) : Prop :=
  -- Definition of line intersection
  sorry

theorem unique_intersecting_line (a b c : Line3D) 
  (hab : are_skew a b) (hbc : are_skew b c) (hac : are_skew a c) :
  ∃! l : Line3D, intersects l a ∧ intersects l b ∧ intersects l c :=
sorry

end NUMINAMATH_CALUDE_unique_intersecting_line_l1492_149209


namespace NUMINAMATH_CALUDE_orange_seller_gain_percentage_l1492_149251

theorem orange_seller_gain_percentage 
  (loss_rate : ℝ) 
  (initial_sale_quantity : ℝ) 
  (new_sale_quantity : ℝ) 
  (loss_percentage : ℝ) : 
  loss_rate = 0.1 → 
  initial_sale_quantity = 10 → 
  new_sale_quantity = 6 → 
  loss_percentage = 10 → 
  ∃ (G : ℝ), G = 50 ∧ 
    (1 + G / 100) * (1 - loss_rate) * initial_sale_quantity / new_sale_quantity = 1 := by
  sorry

end NUMINAMATH_CALUDE_orange_seller_gain_percentage_l1492_149251


namespace NUMINAMATH_CALUDE_total_dogs_l1492_149279

/-- The number of dogs that can fetch -/
def fetch : ℕ := 55

/-- The number of dogs that can roll over -/
def roll : ℕ := 32

/-- The number of dogs that can play dead -/
def play : ℕ := 40

/-- The number of dogs that can fetch and roll over -/
def fetch_roll : ℕ := 20

/-- The number of dogs that can fetch and play dead -/
def fetch_play : ℕ := 18

/-- The number of dogs that can roll over and play dead -/
def roll_play : ℕ := 15

/-- The number of dogs that can do all three tricks -/
def all_tricks : ℕ := 12

/-- The number of dogs that can do no tricks -/
def no_tricks : ℕ := 14

/-- Theorem stating the total number of dogs in the center -/
theorem total_dogs : 
  fetch + roll + play - fetch_roll - fetch_play - roll_play + all_tricks + no_tricks = 100 := by
  sorry

end NUMINAMATH_CALUDE_total_dogs_l1492_149279


namespace NUMINAMATH_CALUDE_baseball_team_wins_l1492_149203

theorem baseball_team_wins (total_games : ℕ) (wins losses : ℕ) : 
  total_games = 130 →
  wins + losses = total_games →
  wins = 3 * losses + 14 →
  wins = 101 := by
sorry

end NUMINAMATH_CALUDE_baseball_team_wins_l1492_149203


namespace NUMINAMATH_CALUDE_staplers_left_proof_l1492_149268

/-- Represents the number of staplers used per report by Stacie -/
def stacie_rate : ℚ := 1

/-- Calculates the number of reports from dozens -/
def dozen_to_reports (dozens : ℕ) : ℕ := dozens * 12

theorem staplers_left_proof (initial_staplers : ℕ) 
  (stacie_dozens jack_dozens : ℕ) (laura_reports : ℕ) : 
  initial_staplers = 450 →
  stacie_dozens = 8 →
  jack_dozens = 9 →
  laura_reports = 50 →
  initial_staplers - 
    (stacie_rate * dozen_to_reports stacie_dozens +
     stacie_rate / 2 * dozen_to_reports jack_dozens +
     stacie_rate * 2 * laura_reports) = 200 := by
  sorry

end NUMINAMATH_CALUDE_staplers_left_proof_l1492_149268


namespace NUMINAMATH_CALUDE_paint_left_after_three_weeks_l1492_149224

def paint_calculation (initial_paint : ℚ) : ℚ :=
  let after_week1 := initial_paint - (1/4 * initial_paint)
  let after_week2 := after_week1 - (1/2 * after_week1)
  let after_week3 := after_week2 - (2/3 * after_week2)
  after_week3

theorem paint_left_after_three_weeks :
  paint_calculation 360 = 45 := by sorry

end NUMINAMATH_CALUDE_paint_left_after_three_weeks_l1492_149224


namespace NUMINAMATH_CALUDE_largest_solution_is_57_98_l1492_149259

/-- The floor function -/
noncomputable def floor (x : ℝ) : ℤ := Int.floor x

/-- The fractional part function -/
noncomputable def frac (x : ℝ) : ℝ := x - (floor x)

/-- The equation from the problem -/
def equation (x : ℝ) : Prop := (floor x : ℝ) = 8 + 50 * (frac x)

/-- The theorem statement -/
theorem largest_solution_is_57_98 :
  ∃ (x : ℝ), equation x ∧ (∀ (y : ℝ), equation y → y ≤ x) ∧ x = 57.98 :=
sorry

end NUMINAMATH_CALUDE_largest_solution_is_57_98_l1492_149259


namespace NUMINAMATH_CALUDE_inequality_proof_l1492_149210

theorem inequality_proof (x y z : ℝ) 
  (non_neg_x : x ≥ 0) (non_neg_y : y ≥ 0) (non_neg_z : z ≥ 0)
  (sum_squares : x^2 + y^2 + z^2 = 1) :
  x / (1 - x^2) + y / (1 - y^2) + z / (1 - z^2) ≥ 3 * Real.sqrt 3 / 2 :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_l1492_149210


namespace NUMINAMATH_CALUDE_line_through_points_l1492_149233

/-- Given two points A and D in 3D space, this theorem proves that the parametric equations
    of the line passing through these points are of the form x = -3 + 4t, y = 3t, z = 1 + t. -/
theorem line_through_points (A D : ℝ × ℝ × ℝ) (h : A = (-3, 0, 1) ∧ D = (1, 3, 2)) :
  ∃ (f : ℝ → ℝ × ℝ × ℝ), ∀ t : ℝ,
    f t = (-3 + 4*t, 3*t, 1 + t) ∧
    (∃ t₁ t₂ : ℝ, f t₁ = A ∧ f t₂ = D) :=
by sorry

end NUMINAMATH_CALUDE_line_through_points_l1492_149233


namespace NUMINAMATH_CALUDE_cafeteria_earnings_l1492_149293

/-- Calculates the total earnings of a cafeteria from selling fruits --/
theorem cafeteria_earnings
  (initial_apples initial_oranges initial_bananas : ℕ)
  (remaining_apples remaining_oranges remaining_bananas : ℕ)
  (apple_cost orange_cost banana_cost : ℚ) :
  initial_apples = 80 →
  initial_oranges = 60 →
  initial_bananas = 40 →
  remaining_apples = 25 →
  remaining_oranges = 15 →
  remaining_bananas = 5 →
  apple_cost = 1.20 →
  orange_cost = 0.75 →
  banana_cost = 0.55 →
  (initial_apples - remaining_apples) * apple_cost +
  (initial_oranges - remaining_oranges) * orange_cost +
  (initial_bananas - remaining_bananas) * banana_cost = 119 :=
by sorry

end NUMINAMATH_CALUDE_cafeteria_earnings_l1492_149293


namespace NUMINAMATH_CALUDE_geometric_sequence_fourth_term_l1492_149283

theorem geometric_sequence_fourth_term
  (a₁ a₅ : ℝ)
  (h₁ : a₁ = 5)
  (h₂ : a₅ = 10240)
  (h₃ : ∃ (r : ℝ), ∀ (n : ℕ), n ≤ 5 → a₁ * r^(n-1) = a₅^((n-1)/4) * a₁^(1-(n-1)/4)) :
  ∃ (a₄ : ℝ), a₄ = 2560 ∧ a₄ = a₁ * (a₅ / a₁)^(3/4) := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_fourth_term_l1492_149283


namespace NUMINAMATH_CALUDE_quadratic_inequality_l1492_149216

-- Define the quadratic function
def f (x : ℝ) : ℝ := -3 * x^2 + 6 * x - 5

-- State the theorem
theorem quadratic_inequality (x₁ x₂ y₁ y₂ : ℝ) 
  (h₁ : 0 ≤ x₁ ∧ x₁ < 1) 
  (h₂ : 2 ≤ x₂ ∧ x₂ < 3) 
  (hy₁ : y₁ = f x₁) 
  (hy₂ : y₂ = f x₂) : 
  y₁ ≥ y₂ := by
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_l1492_149216


namespace NUMINAMATH_CALUDE_solve_rental_problem_l1492_149248

def rental_problem (daily_rate : ℚ) (mileage_rate : ℚ) (days : ℕ) (miles : ℕ) : Prop :=
  daily_rate * days + mileage_rate * miles = 275

theorem solve_rental_problem :
  rental_problem 30 0.25 5 500 := by
  sorry

end NUMINAMATH_CALUDE_solve_rental_problem_l1492_149248


namespace NUMINAMATH_CALUDE_diophantine_equation_solution_l1492_149206

theorem diophantine_equation_solution : ∃ (a b c : ℕ+), a^3 + b^4 = c^5 ∧ a = 256 ∧ b = 64 ∧ c = 32 := by
  sorry

end NUMINAMATH_CALUDE_diophantine_equation_solution_l1492_149206


namespace NUMINAMATH_CALUDE_equation_solution_l1492_149205

theorem equation_solution :
  ∃ x : ℚ, (x^2 + 3*x + 4) / (x + 5) = x + 6 ∧ x = -13/4 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1492_149205


namespace NUMINAMATH_CALUDE_quadratic_inequality_range_l1492_149220

theorem quadratic_inequality_range (m : ℝ) : 
  (∀ x : ℝ, (m + 1) * x^2 + (m + 1) * x + (m + 2) ≥ 0) ↔ m ≥ -1 := by
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_range_l1492_149220


namespace NUMINAMATH_CALUDE_donation_relationship_l1492_149253

/-- Represents the relationship between the number of girls and the total donation in a class. -/
def donation_function (x : ℕ) : ℝ :=
  -5 * x + 1125

/-- Theorem stating the relationship between the number of girls and the total donation. -/
theorem donation_relationship (x : ℕ) (y : ℝ) 
  (h1 : x ≤ 45)  -- Ensure the number of girls is not more than the total number of students
  (h2 : y = 20 * x + 25 * (45 - x)) :  -- Total donation calculation
  y = donation_function x :=
by
  sorry

#check donation_relationship

end NUMINAMATH_CALUDE_donation_relationship_l1492_149253


namespace NUMINAMATH_CALUDE_acme_cheaper_at_min_shirts_l1492_149222

/-- Acme T-Shirt Company's setup fee -/
def acme_setup : ℕ := 60

/-- Acme T-Shirt Company's per-shirt cost -/
def acme_per_shirt : ℕ := 11

/-- Gamma T-Shirt Company's setup fee -/
def gamma_setup : ℕ := 10

/-- Gamma T-Shirt Company's per-shirt cost -/
def gamma_per_shirt : ℕ := 16

/-- The minimum number of shirts for which Acme is cheaper than Gamma -/
def min_shirts_acme_cheaper : ℕ := 11

theorem acme_cheaper_at_min_shirts :
  acme_setup + acme_per_shirt * min_shirts_acme_cheaper <
  gamma_setup + gamma_per_shirt * min_shirts_acme_cheaper ∧
  ∀ n : ℕ, n < min_shirts_acme_cheaper →
    acme_setup + acme_per_shirt * n ≥ gamma_setup + gamma_per_shirt * n :=
by sorry

end NUMINAMATH_CALUDE_acme_cheaper_at_min_shirts_l1492_149222


namespace NUMINAMATH_CALUDE_rate_of_discount_l1492_149296

/-- Calculate the rate of discount given the marked price and selling price -/
theorem rate_of_discount (marked_price selling_price : ℝ) 
  (h1 : marked_price = 150)
  (h2 : selling_price = 120) : 
  (marked_price - selling_price) / marked_price * 100 = 20 := by
  sorry

end NUMINAMATH_CALUDE_rate_of_discount_l1492_149296


namespace NUMINAMATH_CALUDE_waffle_cooking_time_l1492_149225

/-- The time it takes Carla to cook a batch of waffles -/
def waffle_time : ℕ := sorry

/-- The time it takes Carla to cook a chicken-fried steak -/
def steak_time : ℕ := 6

/-- The total time it takes Carla to cook 3 steaks and a batch of waffles -/
def total_time : ℕ := 28

/-- Theorem stating that the time to cook a batch of waffles is 10 minutes -/
theorem waffle_cooking_time : waffle_time = 10 := by sorry

end NUMINAMATH_CALUDE_waffle_cooking_time_l1492_149225


namespace NUMINAMATH_CALUDE_base4_division_theorem_l1492_149237

/-- Converts a number from base 4 to base 10 --/
def base4ToBase10 (n : ℕ) : ℕ := sorry

/-- Converts a number from base 10 to base 4 --/
def base10ToBase4 (n : ℕ) : ℕ := sorry

/-- Performs division in base 4 --/
def divBase4 (a b : ℕ) : ℕ := sorry

theorem base4_division_theorem :
  let dividend := 1302
  let divisor := 12
  let quotient := 103
  divBase4 dividend divisor = quotient := by sorry

end NUMINAMATH_CALUDE_base4_division_theorem_l1492_149237


namespace NUMINAMATH_CALUDE_evaluate_fraction_l1492_149290

theorem evaluate_fraction : (0.4 ^ 4) / (0.04 ^ 3) = 400 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_fraction_l1492_149290


namespace NUMINAMATH_CALUDE_train_crossing_time_l1492_149275

/-- Proves that a train with given length and speed takes the calculated time to cross a pole -/
theorem train_crossing_time (train_length : ℝ) (train_speed_kmh : ℝ) (crossing_time : ℝ) : 
  train_length = 135 →
  train_speed_kmh = 54 →
  crossing_time = train_length / (train_speed_kmh * 1000 / 3600) →
  crossing_time = 9 := by
  sorry

#check train_crossing_time

end NUMINAMATH_CALUDE_train_crossing_time_l1492_149275


namespace NUMINAMATH_CALUDE_difference_of_numbers_l1492_149207

theorem difference_of_numbers (x y : ℝ) (h1 : x + y = 10) (h2 : x^2 - y^2 = 24) :
  |x - y| = 12/5 := by
  sorry

end NUMINAMATH_CALUDE_difference_of_numbers_l1492_149207


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l1492_149226

/-- Given a hyperbola with equation x²/a² - y²/b² = 1 where a > 0 and b > 0,
    if one of its asymptotes is y = -√5/2 * x, then its eccentricity is 3/2. -/
theorem hyperbola_eccentricity (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h_asymptote : b / a = Real.sqrt 5 / 2) : 
  Real.sqrt (a^2 + b^2) / a = 3/2 := by sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l1492_149226


namespace NUMINAMATH_CALUDE_program_cost_calculation_l1492_149256

-- Define constants
def millisecond_to_second : Real := 0.001
def minute_to_millisecond : Nat := 60000
def os_overhead_cost : Real := 1.07
def computer_time_cost_per_ms : Real := 0.023
def data_tape_cost : Real := 5.35
def memory_cost_per_mb : Real := 0.15
def electricity_cost_per_kwh : Real := 0.02
def program_runtime_minutes : Nat := 45
def program_memory_gb : Real := 3.5
def program_electricity_kwh : Real := 2
def gb_to_mb : Nat := 1024

-- Define the theorem
theorem program_cost_calculation :
  let total_milliseconds := program_runtime_minutes * minute_to_millisecond
  let computer_time_cost := total_milliseconds * computer_time_cost_per_ms
  let memory_usage_mb := program_memory_gb * gb_to_mb
  let memory_cost := memory_usage_mb * memory_cost_per_mb
  let electricity_cost := program_electricity_kwh * electricity_cost_per_kwh
  let total_cost := os_overhead_cost + computer_time_cost + data_tape_cost + memory_cost + electricity_cost
  total_cost = 62644.06 := by
  sorry

end NUMINAMATH_CALUDE_program_cost_calculation_l1492_149256


namespace NUMINAMATH_CALUDE_surface_area_of_problem_solid_l1492_149236

/-- Represents an L-shaped solid formed by unit cubes -/
structure LShapedSolid where
  base_layer : ℕ
  top_layer : ℕ
  top_layer_start : ℕ

/-- Calculates the surface area of an L-shaped solid -/
def surface_area (solid : LShapedSolid) : ℕ :=
  let base_exposed := solid.base_layer - (solid.top_layer - (solid.top_layer_start - 1))
  let top_exposed := solid.top_layer
  let front_back := 2 * (solid.base_layer + solid.top_layer)
  let sides := 2 * 2
  let top_bottom := base_exposed + top_exposed + (solid.top_layer_start - 1)
  front_back + sides + top_bottom

/-- The specific L-shaped solid described in the problem -/
def problem_solid : LShapedSolid :=
  { base_layer := 8
  , top_layer := 6
  , top_layer_start := 5 }

theorem surface_area_of_problem_solid :
  surface_area problem_solid = 44 :=
sorry

end NUMINAMATH_CALUDE_surface_area_of_problem_solid_l1492_149236


namespace NUMINAMATH_CALUDE_range_of_a_l1492_149223

-- Define the sets S and T
def S : Set ℝ := {x | x < -1 ∨ x > 5}
def T (a : ℝ) : Set ℝ := {x | a < x ∧ x < a + 8}

-- State the theorem
theorem range_of_a : 
  ∀ a : ℝ, (S ∪ T a = Set.univ) → (-3 < a ∧ a < -1) :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l1492_149223


namespace NUMINAMATH_CALUDE_polynomial_division_theorem_l1492_149208

theorem polynomial_division_theorem (x : ℝ) :
  (x - 2) * (x^5 + 2*x^4 + 4*x^3 + 13*x^2 + 26*x + 52) + 96 = x^6 + 5*x^3 - 8 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_division_theorem_l1492_149208


namespace NUMINAMATH_CALUDE_airplane_seats_l1492_149262

theorem airplane_seats : ∃ (total : ℝ), 
  (30 : ℝ) + 0.2 * total + 0.75 * total = total ∧ total = 600 := by
  sorry

end NUMINAMATH_CALUDE_airplane_seats_l1492_149262


namespace NUMINAMATH_CALUDE_inequality_proof_l1492_149294

theorem inequality_proof (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hab : a > b) :
  1 / (a * b^2) > 1 / (a^2 * b) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1492_149294


namespace NUMINAMATH_CALUDE_disk_color_difference_l1492_149242

theorem disk_color_difference (total : ℕ) (blue_ratio yellow_ratio green_ratio : ℕ) :
  total = 144 →
  blue_ratio = 3 →
  yellow_ratio = 7 →
  green_ratio = 8 →
  let total_ratio := blue_ratio + yellow_ratio + green_ratio
  let disks_per_part := total / total_ratio
  let blue_disks := blue_ratio * disks_per_part
  let green_disks := green_ratio * disks_per_part
  green_disks - blue_disks = 40 :=
by
  sorry

end NUMINAMATH_CALUDE_disk_color_difference_l1492_149242


namespace NUMINAMATH_CALUDE_negation_of_all_squares_positive_l1492_149204

theorem negation_of_all_squares_positive :
  (¬ ∀ n : ℕ, n^2 > 0) ↔ (∃ n : ℕ, ¬(n^2 > 0)) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_all_squares_positive_l1492_149204


namespace NUMINAMATH_CALUDE_fraction_sum_equality_l1492_149265

theorem fraction_sum_equality (p q : ℚ) (h : p / q = 4 / 5) :
  4 / 7 + (2 * q - p) / (2 * q + p) = 1 := by sorry

end NUMINAMATH_CALUDE_fraction_sum_equality_l1492_149265


namespace NUMINAMATH_CALUDE_questions_left_blank_l1492_149281

/-- Represents the math test structure and Steve's performance --/
structure MathTest where
  totalQuestions : ℕ
  wordProblems : ℕ
  addSubProblems : ℕ
  algebraProblems : ℕ
  geometryProblems : ℕ
  totalTime : ℕ
  timePerWordProblem : ℚ
  timePerAddSubProblem : ℚ
  timePerAlgebraProblem : ℕ
  timePerGeometryProblem : ℕ
  wordProblemsAnswered : ℕ
  addSubProblemsAnswered : ℕ
  algebraProblemsAnswered : ℕ
  geometryProblemsAnswered : ℕ

/-- Theorem stating the number of questions left blank --/
theorem questions_left_blank (test : MathTest)
  (h1 : test.totalQuestions = 60)
  (h2 : test.wordProblems = 20)
  (h3 : test.addSubProblems = 25)
  (h4 : test.algebraProblems = 10)
  (h5 : test.geometryProblems = 5)
  (h6 : test.totalTime = 90)
  (h7 : test.timePerWordProblem = 2)
  (h8 : test.timePerAddSubProblem = 3/2)
  (h9 : test.timePerAlgebraProblem = 3)
  (h10 : test.timePerGeometryProblem = 4)
  (h11 : test.wordProblemsAnswered = 15)
  (h12 : test.addSubProblemsAnswered = 22)
  (h13 : test.algebraProblemsAnswered = 8)
  (h14 : test.geometryProblemsAnswered = 3) :
  test.totalQuestions - (test.wordProblemsAnswered + test.addSubProblemsAnswered + test.algebraProblemsAnswered + test.geometryProblemsAnswered) = 12 := by
  sorry

end NUMINAMATH_CALUDE_questions_left_blank_l1492_149281


namespace NUMINAMATH_CALUDE_officer_hopps_ticket_problem_l1492_149213

/-- Calculates the average number of tickets needed per day for the remaining days of the month -/
def average_tickets_remaining (total_tickets : ℕ) (days_in_month : ℕ) (first_period : ℕ) (first_period_average : ℕ) : ℚ :=
  let remaining_days := days_in_month - first_period
  let tickets_given := first_period * first_period_average
  let remaining_tickets := total_tickets - tickets_given
  (remaining_tickets : ℚ) / remaining_days

theorem officer_hopps_ticket_problem :
  average_tickets_remaining 200 31 15 8 = 5 := by
  sorry

end NUMINAMATH_CALUDE_officer_hopps_ticket_problem_l1492_149213


namespace NUMINAMATH_CALUDE_equation_solution_l1492_149234

theorem equation_solution : 
  let y : ℝ := -33/2
  ∀ x : ℝ, (8*x^2 + 78*x + 5) / (2*x + 19) = 4*x + 2 → x = y := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1492_149234


namespace NUMINAMATH_CALUDE_sams_initial_points_l1492_149287

theorem sams_initial_points :
  ∀ initial_points : ℕ,
  initial_points + 3 = 95 →
  initial_points = 92 :=
by
  sorry

end NUMINAMATH_CALUDE_sams_initial_points_l1492_149287


namespace NUMINAMATH_CALUDE_cupcakes_frosted_proof_l1492_149240

/-- Cagney's frosting rate in cupcakes per second -/
def cagney_rate : ℚ := 1 / 25

/-- Lacey's frosting rate in cupcakes per second -/
def lacey_rate : ℚ := 1 / 35

/-- Total working time in seconds -/
def total_time : ℕ := 600

/-- The number of cupcakes frosted when working together -/
def cupcakes_frosted : ℕ := 41

theorem cupcakes_frosted_proof :
  ⌊(cagney_rate + lacey_rate) * total_time⌋ = cupcakes_frosted := by
  sorry

end NUMINAMATH_CALUDE_cupcakes_frosted_proof_l1492_149240


namespace NUMINAMATH_CALUDE_square_sum_product_l1492_149244

theorem square_sum_product (x y : ℝ) (hx : x = Real.sqrt 5 + Real.sqrt 3) (hy : y = Real.sqrt 5 - Real.sqrt 3) :
  x^2 + x*y + y^2 = 18 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_product_l1492_149244


namespace NUMINAMATH_CALUDE_contrapositive_equivalence_l1492_149212

/-- The proposition "If a and b are both even, then the sum of a and b is even" -/
def original_proposition (a b : ℤ) : Prop :=
  (Even a ∧ Even b) → Even (a + b)

/-- The contrapositive of the original proposition -/
def contrapositive (a b : ℤ) : Prop :=
  ¬Even (a + b) → ¬(Even a ∧ Even b)

/-- Theorem stating that the contrapositive is equivalent to "If the sum of a and b is not even, then a and b are not both even" -/
theorem contrapositive_equivalence :
  ∀ a b : ℤ, contrapositive a b ↔ (¬Even (a + b) → ¬(Even a ∧ Even b)) :=
by sorry

end NUMINAMATH_CALUDE_contrapositive_equivalence_l1492_149212


namespace NUMINAMATH_CALUDE_a_alone_time_l1492_149201

-- Define the work rates of a, b, and c
variable (a b c : ℝ)

-- Define the conditions
axiom a_twice_b : a = 2 * b
axiom c_half_b : c = 0.5 * b
axiom combined_rate : a + b + c = 1 / 18
axiom c_alone_rate : c = 1 / 36

-- Theorem to prove
theorem a_alone_time : (1 / a) = 31.5 := by sorry

end NUMINAMATH_CALUDE_a_alone_time_l1492_149201


namespace NUMINAMATH_CALUDE_infinite_solutions_imply_b_value_l1492_149228

theorem infinite_solutions_imply_b_value :
  ∀ b : ℚ, (∀ x : ℚ, 4 * (3 * x - b) = 3 * (4 * x + 7)) → b = -21/4 := by
  sorry

end NUMINAMATH_CALUDE_infinite_solutions_imply_b_value_l1492_149228


namespace NUMINAMATH_CALUDE_least_congruent_number_proof_l1492_149286

/-- The least five-digit positive integer congruent to 7 (mod 18) and 4 (mod 9) -/
def least_congruent_number : ℕ := 10012

theorem least_congruent_number_proof :
  (least_congruent_number ≥ 10000) ∧
  (least_congruent_number < 100000) ∧
  (least_congruent_number % 18 = 7) ∧
  (least_congruent_number % 9 = 4) ∧
  (∀ n : ℕ, n ≥ 10000 ∧ n < 100000 ∧ n % 18 = 7 ∧ n % 9 = 4 → n ≥ least_congruent_number) :=
by sorry

end NUMINAMATH_CALUDE_least_congruent_number_proof_l1492_149286


namespace NUMINAMATH_CALUDE_arithmetic_sequence_ratio_l1492_149245

/-- An arithmetic sequence with its sum function -/
structure ArithmeticSequence where
  a : ℕ → ℝ  -- The sequence
  S : ℕ → ℝ  -- The sum function
  h_arithmetic : ∀ n, a (n + 1) - a n = a 2 - a 1  -- Arithmetic property
  h_sum : ∀ n, S n = n * (a 1 + a n) / 2  -- Sum formula

/-- Theorem: For an arithmetic sequence with S₁ = 1 and S₄/S₂ = 4, S₆/S₄ = 9/4 -/
theorem arithmetic_sequence_ratio (seq : ArithmeticSequence)
    (h1 : seq.S 1 = 1)
    (h2 : seq.S 4 / seq.S 2 = 4) :
  seq.S 6 / seq.S 4 = 9/4 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_ratio_l1492_149245


namespace NUMINAMATH_CALUDE_grain_storage_capacity_l1492_149219

theorem grain_storage_capacity (total_bins : ℕ) (large_bin_capacity : ℕ) (total_capacity : ℕ) (num_large_bins : ℕ) :
  total_bins = 30 →
  large_bin_capacity = 20 →
  total_capacity = 510 →
  num_large_bins = 12 →
  ∃ (small_bin_capacity : ℕ),
    small_bin_capacity * (total_bins - num_large_bins) + large_bin_capacity * num_large_bins = total_capacity ∧
    small_bin_capacity = 15 :=
by
  sorry

end NUMINAMATH_CALUDE_grain_storage_capacity_l1492_149219


namespace NUMINAMATH_CALUDE_volcano_ash_height_l1492_149295

theorem volcano_ash_height (radius : ℝ) (height : ℝ) : 
  radius = 2700 → 2 * radius = 18 * height → height = 300 := by
  sorry

end NUMINAMATH_CALUDE_volcano_ash_height_l1492_149295


namespace NUMINAMATH_CALUDE_mean_median_difference_l1492_149285

/-- Represents the score distribution of a math test -/
structure ScoreDistribution where
  score60 : Float
  score75 : Float
  score85 : Float
  score90 : Float
  score100 : Float
  sum_to_one : score60 + score75 + score85 + score90 + score100 = 1

/-- Calculates the mean score given a ScoreDistribution -/
def meanScore (d : ScoreDistribution) : Float :=
  60 * d.score60 + 75 * d.score75 + 85 * d.score85 + 90 * d.score90 + 100 * d.score100

/-- Calculates the median score given a ScoreDistribution -/
def medianScore (d : ScoreDistribution) : Float :=
  if d.score60 + d.score75 > 0.5 then 75
  else if d.score60 + d.score75 + d.score85 > 0.5 then 85
  else 90

/-- The main theorem stating the difference between mean and median -/
theorem mean_median_difference (d : ScoreDistribution) 
  (h1 : d.score60 = 0.15)
  (h2 : d.score75 = 0.20)
  (h3 : d.score85 = 0.25)
  (h4 : d.score90 = 0.25) :
  medianScore d - meanScore d = 2.25 := by
  sorry


end NUMINAMATH_CALUDE_mean_median_difference_l1492_149285


namespace NUMINAMATH_CALUDE_wickets_before_last_match_l1492_149276

/-- Represents the number of wickets taken before the last match -/
def W : ℕ := sorry

/-- The initial bowling average -/
def initial_average : ℚ := 12.4

/-- The number of wickets taken in the last match -/
def last_match_wickets : ℕ := 7

/-- The number of runs conceded in the last match -/
def last_match_runs : ℕ := 26

/-- The decrease in average after the last match -/
def average_decrease : ℚ := 0.4

/-- The new average after the last match -/
def new_average : ℚ := initial_average - average_decrease

theorem wickets_before_last_match :
  (initial_average * W + last_match_runs : ℚ) / (W + last_match_wickets) = new_average →
  W = 145 := by sorry

end NUMINAMATH_CALUDE_wickets_before_last_match_l1492_149276


namespace NUMINAMATH_CALUDE_cubic_fraction_equals_ten_l1492_149211

theorem cubic_fraction_equals_ten (a b : ℝ) (ha : a = 7) (hb : b = 3) :
  (a^3 + b^3) / (a^2 - a*b + b^2) = 10 := by
  sorry

end NUMINAMATH_CALUDE_cubic_fraction_equals_ten_l1492_149211


namespace NUMINAMATH_CALUDE_cone_properties_l1492_149217

/-- A cone with vertex P, base radius √3, and lateral area 2√3π -/
structure Cone where
  vertex : Point
  base_radius : ℝ
  lateral_area : ℝ
  h_base_radius : base_radius = Real.sqrt 3
  h_lateral_area : lateral_area = 2 * Real.sqrt 3 * Real.pi

/-- The length of the generatrix of the cone -/
def generatrix_length (c : Cone) : ℝ := sorry

/-- The angle between the generatrix and the base of the cone -/
def generatrix_base_angle (c : Cone) : ℝ := sorry

theorem cone_properties (c : Cone) : 
  generatrix_length c = 2 ∧ generatrix_base_angle c = Real.pi / 6 := by
  sorry

end NUMINAMATH_CALUDE_cone_properties_l1492_149217


namespace NUMINAMATH_CALUDE_same_range_implies_b_constraint_l1492_149278

-- Define the function f
def f (b : ℝ) (x : ℝ) : ℝ := x^2 + b*x + 2

-- Define the function g
def g (b : ℝ) (x : ℝ) : ℝ := f b (f b x)

-- State the theorem
theorem same_range_implies_b_constraint (b : ℝ) :
  (∀ y : ℝ, ∃ x : ℝ, f b x = y) ↔ (∀ y : ℝ, ∃ x : ℝ, g b x = y) →
  b ≥ 4 ∨ b ≤ -2 :=
sorry

end NUMINAMATH_CALUDE_same_range_implies_b_constraint_l1492_149278


namespace NUMINAMATH_CALUDE_chloe_final_score_is_86_l1492_149298

/-- Chloe's final score in a trivia game -/
def chloeFinalScore (firstRoundScore secondRoundScore lastRoundLoss : ℕ) : ℕ :=
  firstRoundScore + secondRoundScore - lastRoundLoss

/-- Theorem: Chloe's final score is 86 points -/
theorem chloe_final_score_is_86 :
  chloeFinalScore 40 50 4 = 86 := by
  sorry

#eval chloeFinalScore 40 50 4

end NUMINAMATH_CALUDE_chloe_final_score_is_86_l1492_149298


namespace NUMINAMATH_CALUDE_chef_wage_difference_chef_earns_less_l1492_149263

def manager_wage : ℚ := 17/2

theorem chef_wage_difference : ℚ :=
  let dishwasher_wage := manager_wage / 2
  let chef_wage := dishwasher_wage * (1 + 1/4)
  manager_wage - chef_wage

theorem chef_earns_less (h : chef_wage_difference = 255/80) : True := by
  sorry

end NUMINAMATH_CALUDE_chef_wage_difference_chef_earns_less_l1492_149263


namespace NUMINAMATH_CALUDE_cricket_team_average_age_l1492_149215

/-- The average age of a cricket team given specific conditions -/
theorem cricket_team_average_age :
  ∀ (team_size : ℕ) (captain_age : ℕ) (wicket_keeper_age_diff : ℕ) (team_average : ℝ),
  team_size = 11 →
  captain_age = 26 →
  wicket_keeper_age_diff = 3 →
  (team_size : ℝ) * team_average = 
    (team_size - 2 : ℝ) * (team_average - 1) + 
    (captain_age : ℝ) + (captain_age + wicket_keeper_age_diff : ℝ) →
  team_average = 32 := by
sorry

end NUMINAMATH_CALUDE_cricket_team_average_age_l1492_149215


namespace NUMINAMATH_CALUDE_count_monomials_l1492_149264

-- Define what a monomial is
def is_monomial (term : String) : Bool :=
  match term with
  | "0" => true  -- 0 is considered a monomial
  | t => (t.count '+' = 0) ∧ (t.count '-' ≤ 1) ∧ (t.count '/' = 0)  -- Simplified check for monomials

-- Define the list of terms in the expression
def expression : List String := ["1/x", "x+y", "0", "-a", "-3x^2y", "(x+1)/3"]

-- State the theorem
theorem count_monomials : 
  (expression.filter is_monomial).length = 3 := by sorry

end NUMINAMATH_CALUDE_count_monomials_l1492_149264


namespace NUMINAMATH_CALUDE_parallelogram_altitude_theorem_l1492_149202

-- Define the parallelogram ABCD
structure Parallelogram :=
  (A B C D : ℝ × ℝ)
  (is_parallelogram : sorry)

-- Define the properties of the parallelogram
def Parallelogram.DC (p : Parallelogram) : ℝ := sorry
def Parallelogram.EB (p : Parallelogram) : ℝ := sorry
def Parallelogram.DE (p : Parallelogram) : ℝ := sorry
def Parallelogram.DF (p : Parallelogram) : ℝ := sorry

-- State the theorem
theorem parallelogram_altitude_theorem (p : Parallelogram) 
  (h1 : p.DC = 15)
  (h2 : p.EB = 5)
  (h3 : p.DE = 9) :
  p.DF = 9 := by sorry

end NUMINAMATH_CALUDE_parallelogram_altitude_theorem_l1492_149202


namespace NUMINAMATH_CALUDE_arctan_difference_of_tans_l1492_149214

theorem arctan_difference_of_tans : 
  let result := Real.arctan (Real.tan (75 * π / 180) - 3 * Real.tan (20 * π / 180))
  0 ≤ result ∧ result ≤ π ∧ result = 15 * π / 180 := by
  sorry

end NUMINAMATH_CALUDE_arctan_difference_of_tans_l1492_149214


namespace NUMINAMATH_CALUDE_a_gt_b_not_sufficient_nor_necessary_for_a_sq_gt_b_sq_l1492_149274

theorem a_gt_b_not_sufficient_nor_necessary_for_a_sq_gt_b_sq :
  ¬(∀ a b : ℝ, a > b → a^2 > b^2) ∧ ¬(∀ a b : ℝ, a^2 > b^2 → a > b) := by
  sorry

end NUMINAMATH_CALUDE_a_gt_b_not_sufficient_nor_necessary_for_a_sq_gt_b_sq_l1492_149274


namespace NUMINAMATH_CALUDE_exchange_theorem_l1492_149292

/-- Represents the number of exchanges between Xiao Zhang and Xiao Li -/
def num_exchanges : ℕ := 4

/-- Initial number of pencils Xiao Zhang has -/
def initial_pencils : ℕ := 200

/-- Initial number of fountain pens Xiao Li has -/
def initial_pens : ℕ := 20

/-- Number of pencils exchanged per transaction -/
def pencils_per_exchange : ℕ := 6

/-- Number of pens exchanged per transaction -/
def pens_per_exchange : ℕ := 1

/-- Ratio of pencils to pens after exchanges -/
def final_ratio : ℕ := 11

theorem exchange_theorem : 
  initial_pencils - num_exchanges * pencils_per_exchange = 
  final_ratio * (initial_pens - num_exchanges * pens_per_exchange) :=
by sorry


end NUMINAMATH_CALUDE_exchange_theorem_l1492_149292


namespace NUMINAMATH_CALUDE_modulus_of_z_l1492_149280

/-- The modulus of the complex number z = 2/(1-i) + (1-i)^2 is equal to √2 -/
theorem modulus_of_z (i : ℂ) (h : i^2 = -1) :
  Complex.abs (2 / (1 - i) + (1 - i)^2) = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_modulus_of_z_l1492_149280


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l1492_149288

/-- A geometric sequence with positive terms -/
def geometric_sequence (a : ℕ+ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ+, a (n + 1) = r * a n

theorem geometric_sequence_sum (a : ℕ+ → ℝ) :
  geometric_sequence a →
  (∀ n : ℕ+, a n > 0) →
  a 3 * a 5 + 2 * a 4 * a 6 + a 5 * a 7 = 81 →
  a 4 + a 6 = 9 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l1492_149288


namespace NUMINAMATH_CALUDE_correct_num_buckets_l1492_149272

/-- The number of crab buckets Tom has -/
def num_buckets : ℕ := 56

/-- The number of crabs in each bucket -/
def crabs_per_bucket : ℕ := 12

/-- The price of each crab in dollars -/
def price_per_crab : ℕ := 5

/-- Tom's weekly earnings in dollars -/
def weekly_earnings : ℕ := 3360

/-- Theorem stating that the number of crab buckets is correct -/
theorem correct_num_buckets : 
  num_buckets = weekly_earnings / (crabs_per_bucket * price_per_crab) := by
  sorry

end NUMINAMATH_CALUDE_correct_num_buckets_l1492_149272


namespace NUMINAMATH_CALUDE_vertex_on_x_axis_l1492_149200

/-- The parabola equation -/
def parabola (x c : ℝ) : ℝ := x^2 - 8*x + c

/-- The x-coordinate of the vertex -/
def vertex_x : ℝ := 4

/-- Theorem: The vertex of the parabola y = x^2 - 8x + c lies on the x-axis if and only if c = 16 -/
theorem vertex_on_x_axis (c : ℝ) : 
  parabola vertex_x c = 0 ↔ c = 16 := by
sorry

end NUMINAMATH_CALUDE_vertex_on_x_axis_l1492_149200


namespace NUMINAMATH_CALUDE_tan_alpha_value_implies_expression_value_l1492_149230

theorem tan_alpha_value_implies_expression_value (α : Real) 
  (h : Real.tan α = -1/2) : 
  (Real.sin (2 * α) + 2 * Real.cos (2 * α)) / 
  (4 * Real.cos (2 * α) - 4 * Real.sin (2 * α)) = 1/14 := by
  sorry

end NUMINAMATH_CALUDE_tan_alpha_value_implies_expression_value_l1492_149230


namespace NUMINAMATH_CALUDE_remainder_of_3_pow_2n_plus_8_l1492_149299

theorem remainder_of_3_pow_2n_plus_8 (n : ℕ) : (3^(2*n) + 8) % 8 = 1 := by
  sorry

end NUMINAMATH_CALUDE_remainder_of_3_pow_2n_plus_8_l1492_149299


namespace NUMINAMATH_CALUDE_parabola_directrix_l1492_149238

/-- The equation of the directrix of the parabola y = 4x^2 is y = -1/16 -/
theorem parabola_directrix (x y : ℝ) : 
  y = 4 * x^2 → (∃ (k : ℝ), k = -1/16 ∧ y = k) :=
sorry

end NUMINAMATH_CALUDE_parabola_directrix_l1492_149238


namespace NUMINAMATH_CALUDE_base_4_minus_base_9_digits_l1492_149273

-- Define a function to calculate the number of digits in a given base
def num_digits (n : ℕ) (base : ℕ) : ℕ :=
  if n = 0 then 1 else Nat.log base n + 1

-- State the theorem
theorem base_4_minus_base_9_digits : 
  num_digits 1024 4 - num_digits 1024 9 = 2 := by
  sorry

end NUMINAMATH_CALUDE_base_4_minus_base_9_digits_l1492_149273
