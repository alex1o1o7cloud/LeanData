import Mathlib

namespace NUMINAMATH_CALUDE_michael_truck_meet_once_l2182_218222

-- Define the constants
def michael_speed : ℝ := 6
def truck_speed : ℝ := 12
def bench_distance : ℝ := 180
def truck_stop_time : ℝ := 40

-- Define the positions of Michael and the truck as functions of time
def michael_position (t : ℝ) : ℝ := michael_speed * t

-- The truck's position is more complex due to stops, so we'll define it as a noncomputable function
noncomputable def truck_position (t : ℝ) : ℝ := 
  let cycle_time := bench_distance / truck_speed + truck_stop_time
  let full_cycles := ⌊t / cycle_time⌋
  let remaining_time := t - full_cycles * cycle_time
  bench_distance * (full_cycles + 1) + 
    if remaining_time ≤ bench_distance / truck_speed 
    then truck_speed * remaining_time
    else bench_distance

-- Define the theorem
theorem michael_truck_meet_once :
  ∃! t : ℝ, t > 0 ∧ michael_position t = truck_position t :=
sorry


end NUMINAMATH_CALUDE_michael_truck_meet_once_l2182_218222


namespace NUMINAMATH_CALUDE_sine_graph_shift_l2182_218275

theorem sine_graph_shift (x : ℝ) :
  3 * Real.sin (2 * x - π / 6) = 3 * Real.sin (2 * (x - π / 12)) :=
by sorry

#check sine_graph_shift

end NUMINAMATH_CALUDE_sine_graph_shift_l2182_218275


namespace NUMINAMATH_CALUDE_purely_imaginary_fraction_l2182_218216

theorem purely_imaginary_fraction (a : ℝ) : 
  (∃ b : ℝ, (Complex.I * b : ℂ) = (2 * Complex.I - 1) / (1 + a * Complex.I)) → a = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_purely_imaginary_fraction_l2182_218216


namespace NUMINAMATH_CALUDE_units_digit_of_product_of_first_four_composites_l2182_218252

def first_four_composites : List Nat := [4, 6, 8, 9]

def product_of_list (l : List Nat) : Nat :=
  l.foldl (·*·) 1

def units_digit (n : Nat) : Nat :=
  n % 10

theorem units_digit_of_product_of_first_four_composites :
  units_digit (product_of_list first_four_composites) = 8 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_product_of_first_four_composites_l2182_218252


namespace NUMINAMATH_CALUDE_airport_distance_proof_l2182_218237

/-- The distance from David's home to the airport in miles -/
def airport_distance : ℝ := 155

/-- David's initial speed in miles per hour -/
def initial_speed : ℝ := 45

/-- The increase in speed for the remaining journey in miles per hour -/
def speed_increase : ℝ := 20

/-- The time David would be late if he continued at the initial speed, in hours -/
def late_time : ℝ := 0.75

theorem airport_distance_proof :
  ∃ (t : ℝ),
    -- t is the actual time needed to arrive on time
    t > 0 ∧
    -- The total distance equals the distance covered at the initial speed
    airport_distance = initial_speed * (t + late_time) ∧
    -- The remaining distance equals the distance covered at the increased speed
    airport_distance - initial_speed = (initial_speed + speed_increase) * (t - 1) :=
by sorry

#check airport_distance_proof

end NUMINAMATH_CALUDE_airport_distance_proof_l2182_218237


namespace NUMINAMATH_CALUDE_water_left_over_l2182_218278

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

end NUMINAMATH_CALUDE_water_left_over_l2182_218278


namespace NUMINAMATH_CALUDE_walter_hushpuppies_cooking_time_l2182_218203

/-- Calculates the time required to cook hushpuppies for a given number of guests -/
def cookingTime (guests : ℕ) (hushpuppiesPerGuest : ℕ) (hushpuppiesPerBatch : ℕ) (minutesPerBatch : ℕ) : ℕ :=
  let totalHushpuppies := guests * hushpuppiesPerGuest
  let batches := (totalHushpuppies + hushpuppiesPerBatch - 1) / hushpuppiesPerBatch
  batches * minutesPerBatch

/-- Proves that the cooking time for Walter's hushpuppies is 80 minutes -/
theorem walter_hushpuppies_cooking_time :
  cookingTime 20 5 10 8 = 80 := by
  sorry

end NUMINAMATH_CALUDE_walter_hushpuppies_cooking_time_l2182_218203


namespace NUMINAMATH_CALUDE_average_children_in_families_with_children_l2182_218219

/-- Given 15 families with an average of 3 children per family, 
    and exactly 3 of these families being childless, 
    prove that the average number of children in the families 
    that have children is 45/12. -/
theorem average_children_in_families_with_children 
  (total_families : ℕ) 
  (average_children : ℚ) 
  (childless_families : ℕ) 
  (h1 : total_families = 15)
  (h2 : average_children = 3)
  (h3 : childless_families = 3) :
  (total_families : ℚ) * average_children / 
  ((total_families : ℚ) - childless_families) = 45 / 12 := by
sorry

end NUMINAMATH_CALUDE_average_children_in_families_with_children_l2182_218219


namespace NUMINAMATH_CALUDE_total_cost_is_correct_l2182_218238

-- Define the cost of one t-shirt
def cost_per_shirt : ℚ := 9.95

-- Define the number of t-shirts bought
def num_shirts : ℕ := 25

-- Define the total cost
def total_cost : ℚ := cost_per_shirt * num_shirts

-- Theorem to prove
theorem total_cost_is_correct : total_cost = 248.75 := by
  sorry

end NUMINAMATH_CALUDE_total_cost_is_correct_l2182_218238


namespace NUMINAMATH_CALUDE_bridge_length_l2182_218213

/-- The length of a bridge given train specifications and crossing time -/
theorem bridge_length (train_length : Real) (train_speed_kmh : Real) (crossing_time_s : Real) :
  train_length = 100 →
  train_speed_kmh = 45 →
  crossing_time_s = 30 →
  (train_speed_kmh * 1000 / 3600 * crossing_time_s) - train_length = 275 := by
sorry

end NUMINAMATH_CALUDE_bridge_length_l2182_218213


namespace NUMINAMATH_CALUDE_josh_marbles_remaining_l2182_218215

def initial_marbles : ℕ := 19
def lost_marbles : ℕ := 11

theorem josh_marbles_remaining : initial_marbles - lost_marbles = 8 := by
  sorry

end NUMINAMATH_CALUDE_josh_marbles_remaining_l2182_218215


namespace NUMINAMATH_CALUDE_two_true_propositions_l2182_218204

theorem two_true_propositions (a b c : ℝ) : 
  (∃! n : Nat, n = 2 ∧ 
    (((a > b → a * c^2 > b * c^2) ∧ 
      (a * c^2 > b * c^2 → a > b) ∧ 
      (a ≤ b → a * c^2 ≤ b * c^2) ∧ 
      (a * c^2 ≤ b * c^2 → a ≤ b)) → n = 4) ∧
    ((¬(a > b → a * c^2 > b * c^2) ∧ 
      (a * c^2 > b * c^2 → a > b) ∧ 
      (a ≤ b → a * c^2 ≤ b * c^2) ∧ 
      ¬(a * c^2 ≤ b * c^2 → a ≤ b)) → n = 2) ∧
    ((¬(a > b → a * c^2 > b * c^2) ∧ 
      ¬(a * c^2 > b * c^2 → a > b) ∧ 
      ¬(a ≤ b → a * c^2 ≤ b * c^2) ∧ 
      ¬(a * c^2 ≤ b * c^2 → a ≤ b)) → n = 0)) :=
sorry

end NUMINAMATH_CALUDE_two_true_propositions_l2182_218204


namespace NUMINAMATH_CALUDE_same_terminal_side_l2182_218210

theorem same_terminal_side (a b : Real) : 
  a = -7 * π / 9 → b = 11 * π / 9 → ∃ k : Int, b - a = 2 * π * k := by
  sorry

#check same_terminal_side

end NUMINAMATH_CALUDE_same_terminal_side_l2182_218210


namespace NUMINAMATH_CALUDE_correct_sample_size_l2182_218229

/-- Represents the sampling strategy for a company's employee health survey. -/
structure CompanySampling where
  total_employees : ℕ
  young_employees : ℕ
  middle_aged_employees : ℕ
  elderly_employees : ℕ
  young_in_sample : ℕ

/-- The sample size for the company's health survey. -/
def sample_size (cs : CompanySampling) : ℕ := 15

theorem correct_sample_size (cs : CompanySampling) 
  (h1 : cs.total_employees = 750)
  (h2 : cs.young_employees = 350)
  (h3 : cs.middle_aged_employees = 250)
  (h4 : cs.elderly_employees = 150)
  (h5 : cs.young_in_sample = 7) :
  sample_size cs = 15 := by
  sorry

#check correct_sample_size

end NUMINAMATH_CALUDE_correct_sample_size_l2182_218229


namespace NUMINAMATH_CALUDE_percentage_difference_l2182_218283

theorem percentage_difference (original : ℝ) (result : ℝ) (h : result < original) :
  (original - result) / original * 100 = 50 :=
by
  -- Assuming original = 60 and result = 30
  have h1 : original = 60 := by sorry
  have h2 : result = 30 := by sorry
  
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_percentage_difference_l2182_218283


namespace NUMINAMATH_CALUDE_roots_exist_in_intervals_l2182_218214

-- Define the polynomial function
def f (x : ℝ) : ℝ := x^4 - (2*10^10 + 1)*x^2 - x + 10^20 + 10^10 - 1

-- State the theorem
theorem roots_exist_in_intervals : 
  ∃ (x₁ x₂ : ℝ), 
    f x₁ = 0 ∧ f x₂ = 0 ∧ 
    99999.9996 ≤ x₁ ∧ x₁ ≤ 99999.9998 ∧
    100000.0002 ≤ x₂ ∧ x₂ ≤ 100000.0004 :=
sorry

end NUMINAMATH_CALUDE_roots_exist_in_intervals_l2182_218214


namespace NUMINAMATH_CALUDE_arithmetic_sequence_fifth_term_l2182_218267

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

end NUMINAMATH_CALUDE_arithmetic_sequence_fifth_term_l2182_218267


namespace NUMINAMATH_CALUDE_triangle_angle_measure_l2182_218248

theorem triangle_angle_measure (P Q R : ℝ) : 
  P = 90 → 
  Q = 4 * R - 10 → 
  P + Q + R = 180 → 
  R = 20 := by sorry

end NUMINAMATH_CALUDE_triangle_angle_measure_l2182_218248


namespace NUMINAMATH_CALUDE_new_player_weight_l2182_218208

theorem new_player_weight (n : ℕ) (old_avg new_avg new_weight : ℝ) : 
  n = 20 →
  old_avg = 180 →
  new_avg = 181.42857142857142 →
  (n * old_avg + new_weight) / (n + 1) = new_avg →
  new_weight = 210 := by
sorry

end NUMINAMATH_CALUDE_new_player_weight_l2182_218208


namespace NUMINAMATH_CALUDE_certain_number_is_eight_l2182_218274

theorem certain_number_is_eight (x n : ℚ) : x = 6 ∧ 9 - n / x = 7 + 8 / x → n = 8 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_is_eight_l2182_218274


namespace NUMINAMATH_CALUDE_high_school_twelve_games_l2182_218290

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

end NUMINAMATH_CALUDE_high_school_twelve_games_l2182_218290


namespace NUMINAMATH_CALUDE_complex_1_2i_in_first_quadrant_l2182_218260

/-- A complex number is in the first quadrant if its real part is positive and its imaginary part is positive -/
def in_first_quadrant (z : ℂ) : Prop := 0 < z.re ∧ 0 < z.im

/-- The theorem states that the complex number 1+2i is in the first quadrant -/
theorem complex_1_2i_in_first_quadrant : in_first_quadrant (1 + 2*I) := by
  sorry

end NUMINAMATH_CALUDE_complex_1_2i_in_first_quadrant_l2182_218260


namespace NUMINAMATH_CALUDE_buckingham_palace_visitors_l2182_218223

theorem buckingham_palace_visitors (previous_day_visitors : ℕ) (additional_visitors : ℕ) 
  (h1 : previous_day_visitors = 295) 
  (h2 : additional_visitors = 22) : 
  previous_day_visitors + additional_visitors = 317 := by
  sorry

end NUMINAMATH_CALUDE_buckingham_palace_visitors_l2182_218223


namespace NUMINAMATH_CALUDE_right_triangle_area_l2182_218269

theorem right_triangle_area (hypotenuse : ℝ) (angle : ℝ) :
  hypotenuse = 8 * Real.sqrt 2 →
  angle = 45 * π / 180 →
  let area := (hypotenuse^2 / 4 : ℝ)
  area = 32 := by sorry

end NUMINAMATH_CALUDE_right_triangle_area_l2182_218269


namespace NUMINAMATH_CALUDE_abs_neg_five_l2182_218241

theorem abs_neg_five : |(-5 : ℤ)| = 5 := by
  sorry

end NUMINAMATH_CALUDE_abs_neg_five_l2182_218241


namespace NUMINAMATH_CALUDE_mike_marbles_l2182_218268

/-- Calculates the number of marbles Mike has after giving some to Sam. -/
def marblesLeft (initial : ℕ) (given : ℕ) : ℕ :=
  initial - given

/-- Proves that Mike has 4 marbles left after giving 4 out of his initial 8 marbles to Sam. -/
theorem mike_marbles : marblesLeft 8 4 = 4 := by
  sorry

end NUMINAMATH_CALUDE_mike_marbles_l2182_218268


namespace NUMINAMATH_CALUDE_solution_set_l2182_218206

/-- A function that checks if three positive real numbers can form a non-degenerate triangle -/
def is_triangle (x y z : ℝ) : Prop :=
  x > 0 ∧ y > 0 ∧ z > 0 ∧ x + y > z ∧ x + z > y ∧ y + z > x

/-- The property that n must satisfy -/
def satisfies_condition (n : ℕ) : Prop :=
  n > 0 ∧ ∀ (a b c : ℝ), a > 0 → b > 0 → c > 0 →
    ∃ (l j k : ℕ), is_triangle (a * n ^ k) (b * n ^ j) (c * n ^ l)

/-- The main theorem stating that only 2, 3, and 4 satisfy the condition -/
theorem solution_set : {n : ℕ | satisfies_condition n} = {2, 3, 4} := by sorry

end NUMINAMATH_CALUDE_solution_set_l2182_218206


namespace NUMINAMATH_CALUDE_product_of_repeating_decimals_l2182_218244

/-- Represents a repeating decimal with a single repeating digit -/
def repeatingDecimal (wholePart : ℚ) (repeatingDigit : ℕ) : ℚ :=
  wholePart + (repeatingDigit : ℚ) / 99

theorem product_of_repeating_decimals :
  (repeatingDecimal 0 3) * (repeatingDecimal 0 81) = 9 / 363 := by
  sorry

end NUMINAMATH_CALUDE_product_of_repeating_decimals_l2182_218244


namespace NUMINAMATH_CALUDE_sum_of_specific_polynomials_l2182_218272

/-- A linear polynomial -/
def LinearPolynomial (α : Type*) [Field α] := α → α

/-- A cubic polynomial -/
def CubicPolynomial (α : Type*) [Field α] := α → α

/-- The theorem statement -/
theorem sum_of_specific_polynomials 
  (p : LinearPolynomial ℝ) (q : CubicPolynomial ℝ)
  (h1 : p 1 = 1)
  (h2 : q (-1) = -3)
  (h3 : ∃ r : ℝ → ℝ, ∀ x, q x = r x * (x - 2)^2)
  (h4 : ∃ s t : ℝ → ℝ, (∀ x, p x = s x * (x + 1)) ∧ (∀ x, q x = t x * (x + 1)))
  : ∀ x, p x + q x = -1/3 * x^3 + 4/3 * x^2 + 1/3 * x + 13/6 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_specific_polynomials_l2182_218272


namespace NUMINAMATH_CALUDE_storage_unit_solution_l2182_218286

/-- Represents the storage unit problem -/
def storage_unit_problem (total_units : ℕ) (small_units : ℕ) (small_length : ℕ) (small_width : ℕ) (large_area : ℕ) : Prop :=
  let small_area : ℕ := small_length * small_width
  let large_units : ℕ := total_units - small_units
  let total_area : ℕ := small_units * small_area + large_units * large_area
  total_area = 5040

/-- Theorem stating the solution to the storage unit problem -/
theorem storage_unit_solution : storage_unit_problem 42 20 8 4 200 := by
  sorry


end NUMINAMATH_CALUDE_storage_unit_solution_l2182_218286


namespace NUMINAMATH_CALUDE_carpet_length_is_two_l2182_218202

/-- Represents a rectangular carpet with three concentric regions -/
structure Carpet where
  central_length : ℝ
  central_width : ℝ
  mid_width : ℝ
  outer_width : ℝ

/-- Calculates the area of the central region -/
def central_area (c : Carpet) : ℝ := c.central_length * c.central_width

/-- Calculates the area of the middle region -/
def middle_area (c : Carpet) : ℝ :=
  (c.central_length + 2 * c.mid_width) * (c.central_width + 2 * c.mid_width) - c.central_length * c.central_width

/-- Calculates the area of the outer region -/
def outer_area (c : Carpet) : ℝ :=
  (c.central_length + 2 * c.mid_width + 2 * c.outer_width) * (c.central_width + 2 * c.mid_width + 2 * c.outer_width) -
  (c.central_length + 2 * c.mid_width) * (c.central_width + 2 * c.mid_width)

/-- Checks if three areas form an arithmetic progression -/
def is_arithmetic_progression (a b c : ℝ) : Prop := 2 * b = a + c

theorem carpet_length_is_two (c : Carpet) 
  (h1 : c.central_width = 1)
  (h2 : c.mid_width = 1)
  (h3 : c.outer_width = 1)
  (h4 : is_arithmetic_progression (central_area c) (middle_area c) (outer_area c)) :
  c.central_length = 2 := by
  sorry

end NUMINAMATH_CALUDE_carpet_length_is_two_l2182_218202


namespace NUMINAMATH_CALUDE_scores_statistics_l2182_218230

def scores : List ℕ := [98, 88, 90, 92, 90, 94]

def mode (l : List ℕ) : ℕ := sorry

def median (l : List ℕ) : ℚ := sorry

def average (l : List ℕ) : ℚ := sorry

theorem scores_statistics :
  mode scores = 90 ∧
  median scores = 91 ∧
  average scores = 92 := by sorry

end NUMINAMATH_CALUDE_scores_statistics_l2182_218230


namespace NUMINAMATH_CALUDE_figure_area_solution_l2182_218217

theorem figure_area_solution (x : ℝ) : 
  let square1_area := (3*x)^2
  let square2_area := (7*x)^2
  let triangle_area := (1/2) * (3*x) * (7*x)
  let total_area := square1_area + square2_area + triangle_area
  total_area = 1360 → x = Real.sqrt (2720/119) := by
sorry

end NUMINAMATH_CALUDE_figure_area_solution_l2182_218217


namespace NUMINAMATH_CALUDE_gcd_459_357_l2182_218207

theorem gcd_459_357 : Nat.gcd 459 357 = 51 := by
  sorry

end NUMINAMATH_CALUDE_gcd_459_357_l2182_218207


namespace NUMINAMATH_CALUDE_tree_cutting_theorem_l2182_218246

/-- The number of trees James cuts per day -/
def james_trees_per_day : ℕ := 20

/-- The number of days James works alone -/
def solo_days : ℕ := 2

/-- The number of days James works with his brothers -/
def team_days : ℕ := 3

/-- The number of brothers helping James -/
def num_brothers : ℕ := 2

/-- The percentage of trees each brother cuts compared to James -/
def brother_efficiency : ℚ := 4/5

/-- The total number of trees cut down -/
def total_trees : ℕ := 196

theorem tree_cutting_theorem :
  james_trees_per_day * solo_days + 
  (james_trees_per_day + (james_trees_per_day * brother_efficiency).floor * num_brothers) * team_days = 
  total_trees :=
sorry

end NUMINAMATH_CALUDE_tree_cutting_theorem_l2182_218246


namespace NUMINAMATH_CALUDE_twice_one_fifth_of_ten_times_fifteen_l2182_218264

theorem twice_one_fifth_of_ten_times_fifteen : 2 * ((1 / 5 : ℚ) * (10 * 15)) = 60 := by
  sorry

end NUMINAMATH_CALUDE_twice_one_fifth_of_ten_times_fifteen_l2182_218264


namespace NUMINAMATH_CALUDE_cars_cannot_meet_between_intersections_l2182_218218

/-- Represents a point in the triangular grid --/
structure GridPoint where
  x : ℤ
  y : ℤ

/-- Represents a direction in the triangular grid --/
inductive Direction
  | Up
  | UpRight
  | DownRight

/-- Represents a car's state --/
structure CarState where
  position : GridPoint
  direction : Direction

/-- Represents the possible moves a car can make --/
inductive Move
  | Straight
  | Left
  | Right

/-- Function to update a car's state based on a move --/
def updateCarState (state : CarState) (move : Move) : CarState :=
  sorry

/-- Predicate to check if two cars are at the same position --/
def samePosition (car1 : CarState) (car2 : CarState) : Prop :=
  car1.position = car2.position

/-- Predicate to check if a point is an intersection --/
def isIntersection (point : GridPoint) : Prop :=
  sorry

/-- Theorem stating that two cars cannot meet between intersections --/
theorem cars_cannot_meet_between_intersections 
  (initialState : CarState) 
  (moves1 moves2 : List Move) : 
  let finalState1 := moves1.foldl updateCarState initialState
  let finalState2 := moves2.foldl updateCarState initialState
  samePosition finalState1 finalState2 → isIntersection finalState1.position :=
sorry

end NUMINAMATH_CALUDE_cars_cannot_meet_between_intersections_l2182_218218


namespace NUMINAMATH_CALUDE_discount_equation_l2182_218270

/-- Represents the discount rate as a real number between 0 and 1 -/
def discount_rate : ℝ := sorry

/-- The original price in yuan -/
def original_price : ℝ := 200

/-- The final selling price in yuan -/
def final_price : ℝ := 148

/-- Theorem stating the relationship between original price, discount rate, and final price -/
theorem discount_equation : 
  original_price * (1 - discount_rate)^2 = final_price := by sorry

end NUMINAMATH_CALUDE_discount_equation_l2182_218270


namespace NUMINAMATH_CALUDE_night_shift_nine_hours_l2182_218294

/-- Represents the number of hours worked by each guard -/
structure GuardShift :=
  (first : ℕ)
  (middle1 : ℕ)
  (middle2 : ℕ)
  (last : ℕ)

/-- Calculates the total length of the night shift -/
def nightShiftLength (shift : GuardShift) : ℕ :=
  shift.first + shift.middle1 + shift.middle2 + shift.last

/-- Theorem stating that the night shift length is 9 hours -/
theorem night_shift_nine_hours :
  ∃ (shift : GuardShift),
    shift.first = 3 ∧
    shift.middle1 = 2 ∧
    shift.middle2 = 2 ∧
    shift.last = 2 ∧
    nightShiftLength shift = 9 :=
by
  sorry

end NUMINAMATH_CALUDE_night_shift_nine_hours_l2182_218294


namespace NUMINAMATH_CALUDE_intersection_equals_M_l2182_218284

def M : Set ℝ := {y | ∃ x, y = 3^x}
def N : Set ℝ := {y | ∃ x, y = x^2 - 1}

theorem intersection_equals_M : M ∩ N = M := by sorry

end NUMINAMATH_CALUDE_intersection_equals_M_l2182_218284


namespace NUMINAMATH_CALUDE_first_divisor_problem_l2182_218226

theorem first_divisor_problem (m d : ℕ) : 
  (∃ q : ℕ, m = d * q + 47) →
  (∃ p : ℕ, m = 24 * p + 23) →
  (∀ x < d, ¬(∃ q : ℕ, m = x * q + 47)) →
  d = 72 := by
sorry

end NUMINAMATH_CALUDE_first_divisor_problem_l2182_218226


namespace NUMINAMATH_CALUDE_literate_female_percentage_approx_81_percent_l2182_218297

/-- Represents the demographics and literacy rates of a town -/
structure TownDemographics where
  total_inhabitants : ℕ
  adult_male_percent : ℚ
  adult_female_percent : ℚ
  children_percent : ℚ
  adult_male_literacy : ℚ
  adult_female_literacy : ℚ
  children_literacy : ℚ

/-- Calculates the percentage of literate females in the town -/
def literate_female_percentage (town : TownDemographics) : ℚ :=
  let adult_females := town.total_inhabitants * town.adult_female_percent
  let female_children := town.total_inhabitants * town.children_percent / 2
  let literate_adult_females := adult_females * town.adult_female_literacy
  let literate_female_children := female_children * town.children_literacy
  let total_literate_females := literate_adult_females + literate_female_children
  let total_females := adult_females + female_children
  total_literate_females / total_females

/-- Theorem stating that the percentage of literate females in the town is approximately 81% -/
theorem literate_female_percentage_approx_81_percent 
  (town : TownDemographics)
  (h1 : town.total_inhabitants = 3500)
  (h2 : town.adult_male_percent = 60 / 100)
  (h3 : town.adult_female_percent = 35 / 100)
  (h4 : town.children_percent = 5 / 100)
  (h5 : town.adult_male_literacy = 55 / 100)
  (h6 : town.adult_female_literacy = 80 / 100)
  (h7 : town.children_literacy = 95 / 100) :
  ∃ (ε : ℚ), ε > 0 ∧ ε < 1 / 100 ∧ 
  |literate_female_percentage town - 81 / 100| < ε :=
sorry

end NUMINAMATH_CALUDE_literate_female_percentage_approx_81_percent_l2182_218297


namespace NUMINAMATH_CALUDE_a_minus_c_equals_three_l2182_218224

theorem a_minus_c_equals_three
  (e f a b c d : ℝ)
  (h1 : e = a^2 + b^2)
  (h2 : f = c^2 + d^2)
  (h3 : a - b = c + d + 9)
  (h4 : a + b = c - d - 3)
  (h5 : f - e = 5*a + 2*b + 3*c + 4*d) :
  a - c = 3 := by
sorry

end NUMINAMATH_CALUDE_a_minus_c_equals_three_l2182_218224


namespace NUMINAMATH_CALUDE_james_total_score_l2182_218232

theorem james_total_score (field_goals : ℕ) (two_point_shots : ℕ) : field_goals = 13 → two_point_shots = 20 → field_goals * 3 + two_point_shots * 2 = 79 := by
  sorry

end NUMINAMATH_CALUDE_james_total_score_l2182_218232


namespace NUMINAMATH_CALUDE_equal_angles_45_degrees_l2182_218245

theorem equal_angles_45_degrees (α₁ α₂ α₃ : Real) : 
  0 < α₁ ∧ α₁ < π / 2 →
  0 < α₂ ∧ α₂ < π / 2 →
  0 < α₃ ∧ α₃ < π / 2 →
  Real.sin α₁ = Real.cos α₂ →
  Real.sin α₂ = Real.cos α₃ →
  Real.sin α₃ = Real.cos α₁ →
  α₁ = π / 4 ∧ α₂ = π / 4 ∧ α₃ = π / 4 := by
  sorry

end NUMINAMATH_CALUDE_equal_angles_45_degrees_l2182_218245


namespace NUMINAMATH_CALUDE_upstream_downstream_time_ratio_l2182_218211

def boat_speed : ℝ := 18
def stream_speed : ℝ := 6

def upstream_speed : ℝ := boat_speed - stream_speed
def downstream_speed : ℝ := boat_speed + stream_speed

theorem upstream_downstream_time_ratio :
  upstream_speed / downstream_speed = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_upstream_downstream_time_ratio_l2182_218211


namespace NUMINAMATH_CALUDE_sin_cos_sixth_power_sum_l2182_218257

theorem sin_cos_sixth_power_sum (θ : ℝ) (h : Real.sin (2 * θ) = 1 / 5) :
  Real.sin θ ^ 6 + Real.cos θ ^ 6 = 97 / 100 := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_sixth_power_sum_l2182_218257


namespace NUMINAMATH_CALUDE_tickets_sold_second_week_l2182_218299

/-- The number of tickets sold in the second week of a fair, given the total number of tickets,
    tickets sold in the first week, and tickets left to sell. -/
theorem tickets_sold_second_week
  (total_tickets : ℕ)
  (first_week_sales : ℕ)
  (tickets_left : ℕ)
  (h1 : total_tickets = 90)
  (h2 : first_week_sales = 38)
  (h3 : tickets_left = 35) :
  total_tickets - (first_week_sales + tickets_left) = 17 :=
by sorry

end NUMINAMATH_CALUDE_tickets_sold_second_week_l2182_218299


namespace NUMINAMATH_CALUDE_equal_cost_at_150_miles_unique_equal_cost_mileage_l2182_218263

-- Define the cost functions for both rental companies
def safety_cost (m : ℝ) : ℝ := 41.95 + 0.29 * m
def city_cost (m : ℝ) : ℝ := 38.95 + 0.31 * m

-- Theorem stating that the costs are equal at 150 miles
theorem equal_cost_at_150_miles : 
  safety_cost 150 = city_cost 150 := by
  sorry

-- Theorem stating that 150 miles is the unique solution
theorem unique_equal_cost_mileage :
  ∀ m : ℝ, safety_cost m = city_cost m ↔ m = 150 := by
  sorry

end NUMINAMATH_CALUDE_equal_cost_at_150_miles_unique_equal_cost_mileage_l2182_218263


namespace NUMINAMATH_CALUDE_negation_of_forall_gt_is_exists_leq_l2182_218255

theorem negation_of_forall_gt_is_exists_leq :
  (¬ ∀ x : ℝ, x^2 > 1 - 2*x) ↔ (∃ x : ℝ, x^2 ≤ 1 - 2*x) := by sorry

end NUMINAMATH_CALUDE_negation_of_forall_gt_is_exists_leq_l2182_218255


namespace NUMINAMATH_CALUDE_pants_cost_l2182_218277

def initial_amount : ℕ := 109
def shirt_cost : ℕ := 11
def num_shirts : ℕ := 2
def remaining_amount : ℕ := 74

theorem pants_cost : 
  initial_amount - (shirt_cost * num_shirts) - remaining_amount = 13 := by
  sorry

end NUMINAMATH_CALUDE_pants_cost_l2182_218277


namespace NUMINAMATH_CALUDE_quadratic_transformation_l2182_218254

theorem quadratic_transformation (x : ℝ) :
  x^2 - 10*x - 1 = 0 ↔ (x - 5)^2 = 26 := by sorry

end NUMINAMATH_CALUDE_quadratic_transformation_l2182_218254


namespace NUMINAMATH_CALUDE_f_8_equals_8_65_l2182_218201

/-- A function that takes a natural number and returns a rational number -/
def f (n : ℕ) : ℚ := n / (n^2 + 1)

/-- Theorem stating that f(8) equals 8/65 -/
theorem f_8_equals_8_65 : f 8 = 8 / 65 := by
  sorry

end NUMINAMATH_CALUDE_f_8_equals_8_65_l2182_218201


namespace NUMINAMATH_CALUDE_no_valid_class_composition_l2182_218261

theorem no_valid_class_composition : ¬ ∃ (n b g : ℕ+), 
  32 < n ∧ n < 40 ∧ 
  n = b + g ∧
  3 * b = 5 * g :=
by sorry

end NUMINAMATH_CALUDE_no_valid_class_composition_l2182_218261


namespace NUMINAMATH_CALUDE_fraction_pattern_l2182_218271

theorem fraction_pattern (n m k : ℕ) (h1 : m ≠ 0) (h2 : k ≠ 0) 
  (h3 : n / m = k * n / (k * m)) : 
  (n + m) / m = (k * n + k * m) / (k * m) := by
  sorry

end NUMINAMATH_CALUDE_fraction_pattern_l2182_218271


namespace NUMINAMATH_CALUDE_max_value_reciprocal_l2182_218240

theorem max_value_reciprocal (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 1) :
  (∀ x y : ℝ, x > 0 → y > 0 → x + y = 1 → 1 / (x + 2*y - 3*x*y) ≤ 3/2) ∧
  (∃ x y : ℝ, x > 0 ∧ y > 0 ∧ x + y = 1 ∧ 1 / (x + 2*y - 3*x*y) = 3/2) :=
by sorry

end NUMINAMATH_CALUDE_max_value_reciprocal_l2182_218240


namespace NUMINAMATH_CALUDE_zero_full_crates_l2182_218228

/-- Represents the number of berries picked for each type -/
structure BerriesPicked where
  blueberries : ℕ
  cranberries : ℕ
  raspberries : ℕ
  gooseberries : ℕ
  strawberries : ℕ

/-- Represents the fraction of rotten berries for each type -/
structure RottenFractions where
  blueberries : ℚ
  cranberries : ℚ
  raspberries : ℚ
  gooseberries : ℚ
  strawberries : ℚ

/-- Represents the number of berries required to fill one crate for each type -/
structure CrateCapacity where
  blueberries : ℕ
  cranberries : ℕ
  raspberries : ℕ
  gooseberries : ℕ
  strawberries : ℕ

/-- Calculates the number of full crates that can be sold -/
def calculateFullCrates (picked : BerriesPicked) (rotten : RottenFractions) (capacity : CrateCapacity) : ℕ :=
  sorry

/-- Theorem stating that the number of full crates that can be sold is 0 -/
theorem zero_full_crates : 
  let picked : BerriesPicked := ⟨30, 20, 10, 15, 25⟩
  let rotten : RottenFractions := ⟨1/3, 1/4, 1/5, 1/6, 1/7⟩
  let capacity : CrateCapacity := ⟨40, 50, 30, 60, 70⟩
  calculateFullCrates picked rotten capacity = 0 := by
  sorry

end NUMINAMATH_CALUDE_zero_full_crates_l2182_218228


namespace NUMINAMATH_CALUDE_uniform_motion_parametric_equation_l2182_218282

/-- Parametric equation of a point undergoing uniform linear motion -/
def parametric_equation (initial_x initial_y vx vy : ℝ) : ℝ → ℝ × ℝ :=
  λ t => (initial_x + vx * t, initial_y + vy * t)

/-- The correct parametric equation for the given conditions -/
theorem uniform_motion_parametric_equation :
  parametric_equation 1 1 9 12 = λ t => (1 + 9 * t, 1 + 12 * t) := by
  sorry

end NUMINAMATH_CALUDE_uniform_motion_parametric_equation_l2182_218282


namespace NUMINAMATH_CALUDE_plane_equation_proof_l2182_218225

/-- A plane in 3D space represented by its equation coefficients -/
structure Plane where
  a : ℤ
  b : ℤ
  c : ℤ
  d : ℤ

/-- A point in 3D space -/
structure Point where
  x : ℤ
  y : ℤ
  z : ℤ

/-- Check if a point lies on a plane -/
def pointOnPlane (plane : Plane) (point : Point) : Prop :=
  plane.a * point.x + plane.b * point.y + plane.c * point.z + plane.d = 0

/-- Check if two planes are parallel -/
def planesParallel (plane1 : Plane) (plane2 : Plane) : Prop :=
  ∃ (k : ℚ), k ≠ 0 ∧ plane1.a = k * plane2.a ∧ plane1.b = k * plane2.b ∧ plane1.c = k * plane2.c

/-- The greatest common divisor of four integers is 1 -/
def gcdOne (a b c d : ℤ) : Prop :=
  Nat.gcd (Nat.gcd (Nat.gcd a.natAbs b.natAbs) c.natAbs) d.natAbs = 1

theorem plane_equation_proof (givenPlane : Plane) (point : Point) :
  givenPlane.a = 3 ∧ givenPlane.b = -2 ∧ givenPlane.c = 4 ∧ givenPlane.d = 5 →
  point.x = 2 ∧ point.y = -3 ∧ point.z = 1 →
  ∃ (soughtPlane : Plane),
    soughtPlane.a = 3 ∧
    soughtPlane.b = -2 ∧
    soughtPlane.c = 4 ∧
    soughtPlane.d = -16 ∧
    soughtPlane.a > 0 ∧
    pointOnPlane soughtPlane point ∧
    planesParallel soughtPlane givenPlane ∧
    gcdOne soughtPlane.a soughtPlane.b soughtPlane.c soughtPlane.d :=
by sorry

end NUMINAMATH_CALUDE_plane_equation_proof_l2182_218225


namespace NUMINAMATH_CALUDE_inverse_proportion_intersection_l2182_218280

theorem inverse_proportion_intersection (b : ℝ) :
  ∃ k : ℝ, 1 < k ∧ k < 2 ∧
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧
    (6 - 3 * k) / x₁ = -7 * x₁ + b ∧
    (6 - 3 * k) / x₂ = -7 * x₂ + b ∧
    x₁ * x₂ > 0) :=
by sorry

end NUMINAMATH_CALUDE_inverse_proportion_intersection_l2182_218280


namespace NUMINAMATH_CALUDE_age_ratio_correct_l2182_218253

-- Define Sachin's age
def sachin_age : ℚ := 24.5

-- Define the age difference between Rahul and Sachin
def age_difference : ℚ := 7

-- Calculate Rahul's age
def rahul_age : ℚ := sachin_age + age_difference

-- Define the ratio of their ages
def age_ratio : ℚ × ℚ := (7, 9)

-- Theorem to prove
theorem age_ratio_correct : 
  (sachin_age / rahul_age) = (age_ratio.1 / age_ratio.2) := by
  sorry

end NUMINAMATH_CALUDE_age_ratio_correct_l2182_218253


namespace NUMINAMATH_CALUDE_vanessa_score_l2182_218227

/-- Vanessa's basketball score record problem -/
theorem vanessa_score (total_score : ℕ) (num_players : ℕ) (other_players_avg : ℚ) :
  total_score = 68 →
  num_players = 9 →
  other_players_avg = 4.5 →
  ∃ vanessa_score : ℕ,
    vanessa_score = 32 ∧
    vanessa_score = total_score - (num_players - 1) * (other_players_avg.num / other_players_avg.den) :=
by sorry

end NUMINAMATH_CALUDE_vanessa_score_l2182_218227


namespace NUMINAMATH_CALUDE_green_tea_price_decrease_proof_l2182_218279

/-- The percentage decrease in green tea price from June to July -/
def green_tea_price_decrease : ℝ := 90

/-- The cost per pound of green tea and coffee in June -/
def june_price : ℝ := 1

/-- The cost per pound of green tea in July -/
def july_green_tea_price : ℝ := 0.1

/-- The cost per pound of coffee in July -/
def july_coffee_price : ℝ := 2 * june_price

/-- The cost of 3 lbs of mixture containing equal quantities of green tea and coffee in July -/
def mixture_cost : ℝ := 3.15

theorem green_tea_price_decrease_proof :
  green_tea_price_decrease = (june_price - july_green_tea_price) / june_price * 100 ∧
  mixture_cost = 1.5 * july_green_tea_price + 1.5 * july_coffee_price :=
sorry

end NUMINAMATH_CALUDE_green_tea_price_decrease_proof_l2182_218279


namespace NUMINAMATH_CALUDE_petes_flag_has_128_shapes_l2182_218236

/-- Calculates the total number of shapes on Pete's flag given the number of stars and stripes on the US flag. -/
def petes_flag_shapes (us_stars : ℕ) (us_stripes : ℕ) : ℕ :=
  let circles := us_stars / 2 - 3
  let squares := 2 * us_stripes + 6
  let triangles := 2 * (us_stars - us_stripes)
  circles + squares + triangles

/-- Theorem stating that Pete's flag has 128 shapes given the US flag has 50 stars and 13 stripes. -/
theorem petes_flag_has_128_shapes :
  petes_flag_shapes 50 13 = 128 := by
  sorry

end NUMINAMATH_CALUDE_petes_flag_has_128_shapes_l2182_218236


namespace NUMINAMATH_CALUDE_twenty_fifth_triangular_number_l2182_218296

/-- Definition of triangular number -/
def triangular_number (n : ℕ) : ℕ := n * (n + 1) / 2

/-- The 25th triangular number is 325 -/
theorem twenty_fifth_triangular_number : triangular_number 25 = 325 := by
  sorry

end NUMINAMATH_CALUDE_twenty_fifth_triangular_number_l2182_218296


namespace NUMINAMATH_CALUDE_repeating_decimal_equals_fraction_l2182_218250

/-- The repeating decimal 0.4444... expressed as a real number -/
def repeating_decimal : ℚ := 0.4444444444

/-- The theorem states that the repeating decimal 0.4444... is equal to 4/9 -/
theorem repeating_decimal_equals_fraction : repeating_decimal = 4/9 := by
  sorry

end NUMINAMATH_CALUDE_repeating_decimal_equals_fraction_l2182_218250


namespace NUMINAMATH_CALUDE_fraction_simplification_l2182_218265

theorem fraction_simplification : 
  (3 + 6 - 12 + 24 + 48 - 96) / (6 + 12 - 24 + 48 + 96 - 192) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l2182_218265


namespace NUMINAMATH_CALUDE_singer_arrangements_l2182_218258

/-- The number of ways to arrange n objects. -/
def arrangements (n : ℕ) : ℕ := Nat.factorial n

/-- The number of ways to arrange k objects out of n objects. -/
def permutations (n k : ℕ) : ℕ := 
  if k ≤ n then Nat.factorial n / Nat.factorial (n - k) else 0

theorem singer_arrangements : 
  let total_singers : ℕ := 5
  let arrangements_case1 := permutations 4 4  -- when the singer who can't be last is first
  let arrangements_case2 := permutations 3 1 * permutations 3 1 * permutations 3 3  -- other cases
  arrangements_case1 + arrangements_case2 = 78 := by
  sorry

end NUMINAMATH_CALUDE_singer_arrangements_l2182_218258


namespace NUMINAMATH_CALUDE_factorial_prime_factorization_l2182_218276

theorem factorial_prime_factorization (x a k m p : ℕ) : 
  x = Nat.factorial 8 →
  x = 2^a * 3^k * 5^m * 7^p →
  a > 0 ∧ k > 0 ∧ m > 0 ∧ p > 0 →
  a + k + m + p = 11 →
  a = 7 := by
sorry

end NUMINAMATH_CALUDE_factorial_prime_factorization_l2182_218276


namespace NUMINAMATH_CALUDE_ones_digit_of_complex_expression_l2182_218281

-- Define a function to get the ones digit of a natural number
def ones_digit (n : ℕ) : ℕ := n % 10

-- Define the expression
def complex_expression : ℕ := 
  ones_digit ((73^567 % 10) * (47^123 % 10) + (86^784 % 10) - (32^259 % 10))

-- Theorem statement
theorem ones_digit_of_complex_expression :
  complex_expression = 9 := by sorry

end NUMINAMATH_CALUDE_ones_digit_of_complex_expression_l2182_218281


namespace NUMINAMATH_CALUDE_exists_m_divisible_by_1988_l2182_218231

def f (x : ℤ) : ℤ := 3 * x + 2

def iterate_f (n : ℕ) (x : ℤ) : ℤ :=
  match n with
  | 0 => x
  | n + 1 => f (iterate_f n x)

theorem exists_m_divisible_by_1988 :
  ∃ m : ℕ+, ∃ k : ℤ, iterate_f 100 m.val = 1988 * k := by
  sorry

end NUMINAMATH_CALUDE_exists_m_divisible_by_1988_l2182_218231


namespace NUMINAMATH_CALUDE_james_out_of_pocket_l2182_218292

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

end NUMINAMATH_CALUDE_james_out_of_pocket_l2182_218292


namespace NUMINAMATH_CALUDE_line_perpendicular_plane_parallel_l2182_218259

structure Space where
  Line : Type
  Plane : Type
  perpendicular : Line → Plane → Prop
  parallel : Line → Line → Prop

variable (S : Space)

theorem line_perpendicular_plane_parallel
  (l m : S.Line) (α : S.Plane)
  (h1 : l ≠ m)
  (h2 : S.perpendicular l α)
  (h3 : S.parallel l m) :
  S.perpendicular m α :=
sorry

end NUMINAMATH_CALUDE_line_perpendicular_plane_parallel_l2182_218259


namespace NUMINAMATH_CALUDE_probability_sum_10_three_dice_l2182_218205

/-- The number of faces on a standard die -/
def numFaces : ℕ := 6

/-- The number of dice rolled -/
def numDice : ℕ := 3

/-- The sum we're looking for -/
def targetSum : ℕ := 10

/-- The total number of possible outcomes when rolling three dice -/
def totalOutcomes : ℕ := numFaces ^ numDice

/-- The number of favorable outcomes (sum of 10) -/
def favorableOutcomes : ℕ := 27

/-- The probability of rolling a sum of 10 with three standard six-sided dice -/
theorem probability_sum_10_three_dice : 
  (favorableOutcomes : ℚ) / totalOutcomes = 1 / 8 := by
  sorry

end NUMINAMATH_CALUDE_probability_sum_10_three_dice_l2182_218205


namespace NUMINAMATH_CALUDE_sum_of_cubes_zero_l2182_218242

theorem sum_of_cubes_zero (a b : ℝ) (h1 : a + b = 0) (h2 : a * b = -1) : 
  a^3 + b^3 = 0 := by
sorry

end NUMINAMATH_CALUDE_sum_of_cubes_zero_l2182_218242


namespace NUMINAMATH_CALUDE_disobedient_pair_implies_ultra_disobedient_l2182_218239

/-- A function from natural numbers to positive real numbers -/
def IncreasingPositiveFunction : Type := 
  {f : ℕ → ℝ // (∀ m n, m < n → f m < f n) ∧ (∀ n, f n > 0)}

/-- Definition of a disobedient pair -/
def IsDisobedientPair (f : IncreasingPositiveFunction) (m n : ℕ) : Prop :=
  f.val (m * n) ≠ f.val m * f.val n

/-- Definition of an ultra-disobedient number -/
def IsUltraDisobedient (f : IncreasingPositiveFunction) (m : ℕ) : Prop :=
  ∀ N : ℕ, ∀ k : ℕ, ∃ n : ℕ, n > k ∧ 
    ∀ i : ℕ, i ≤ N → IsDisobedientPair f m (n + i)

/-- Main theorem: existence of a disobedient pair implies existence of an ultra-disobedient number -/
theorem disobedient_pair_implies_ultra_disobedient
  (f : IncreasingPositiveFunction)
  (h : ∃ m n : ℕ, IsDisobedientPair f m n) :
  ∃ m : ℕ, IsUltraDisobedient f m :=
sorry

end NUMINAMATH_CALUDE_disobedient_pair_implies_ultra_disobedient_l2182_218239


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l2182_218249

theorem sqrt_equation_solution (a b : ℕ+) (h : a < b) :
  Real.sqrt (1 + Real.sqrt (25 + 14 * Real.sqrt 3)) = Real.sqrt a + Real.sqrt b ↔ 
  a = 1 ∧ b = 3 := by
sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l2182_218249


namespace NUMINAMATH_CALUDE_arthur_walked_six_miles_l2182_218256

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

end NUMINAMATH_CALUDE_arthur_walked_six_miles_l2182_218256


namespace NUMINAMATH_CALUDE_correct_calculation_l2182_218262

theorem correct_calculation (x : ℝ) : 14 * x = 70 → x - 6 = -1 := by
  sorry

end NUMINAMATH_CALUDE_correct_calculation_l2182_218262


namespace NUMINAMATH_CALUDE_inscribed_circle_area_l2182_218273

theorem inscribed_circle_area (large_square_area : ℝ) (h : large_square_area = 80) :
  let large_side := Real.sqrt large_square_area
  let small_side := large_side / Real.sqrt 2
  let circle_radius := small_side / 2
  circle_radius ^ 2 * Real.pi = 10 * Real.pi :=
by sorry

end NUMINAMATH_CALUDE_inscribed_circle_area_l2182_218273


namespace NUMINAMATH_CALUDE_power_calculation_l2182_218291

theorem power_calculation : (16^4 * 8^6) / 4^12 = 1024 := by
  sorry

end NUMINAMATH_CALUDE_power_calculation_l2182_218291


namespace NUMINAMATH_CALUDE_georges_socks_l2182_218295

theorem georges_socks (initial_socks bought_socks dad_socks : ℝ) 
  (h1 : initial_socks = 28.0)
  (h2 : bought_socks = 36.0)
  (h3 : dad_socks = 4.0) :
  initial_socks + bought_socks + dad_socks = 68.0 :=
by sorry

end NUMINAMATH_CALUDE_georges_socks_l2182_218295


namespace NUMINAMATH_CALUDE_max_balloons_proof_l2182_218200

/-- Represents the maximum number of balloons that can be purchased given a budget and pricing scheme. -/
def max_balloons (budget : ℕ) (regular_price : ℕ) (set_price : ℕ) : ℕ :=
  (budget / set_price) * 3

/-- Proves that given $120 to spend, with balloons priced at $4 each, and a special sale where every set of 3 balloons costs $7, the maximum number of balloons that can be purchased is 51. -/
theorem max_balloons_proof :
  max_balloons 120 4 7 = 51 := by
  sorry

end NUMINAMATH_CALUDE_max_balloons_proof_l2182_218200


namespace NUMINAMATH_CALUDE_median_of_four_numbers_l2182_218288

theorem median_of_four_numbers (x : ℝ) : 
  (0 < 4) ∧ (4 < x) ∧ (x < 10) ∧  -- ascending order condition
  ((4 + x) / 2 = 5)                -- median condition
  → x = 6 := by
sorry

end NUMINAMATH_CALUDE_median_of_four_numbers_l2182_218288


namespace NUMINAMATH_CALUDE_vector_projection_l2182_218221

/-- Given two vectors a and e in a real inner product space, 
    where |a| = 4, e is a unit vector, and the angle between a and e is 2π/3,
    prove that the projection of a + e on a - e is 5√21 / 7 -/
theorem vector_projection (V : Type*) [NormedAddCommGroup V] [InnerProductSpace ℝ V] 
  (a e : V) (h1 : ‖a‖ = 4) (h2 : ‖e‖ = 1) 
  (h3 : Real.cos (Real.arccos (inner a e / (‖a‖ * ‖e‖))) = Real.cos (2 * Real.pi / 3)) :
  ‖a + e‖ * (inner (a + e) (a - e) / (‖a + e‖ * ‖a - e‖)) = 5 * Real.sqrt 21 / 7 := by
  sorry

end NUMINAMATH_CALUDE_vector_projection_l2182_218221


namespace NUMINAMATH_CALUDE_sum_divisible_by_31_l2182_218289

def geometric_sum (n : ℕ) : ℕ := (2^(5*n) - 1) / (2 - 1)

theorem sum_divisible_by_31 (n : ℕ+) : 
  31 ∣ geometric_sum n.val := by sorry

end NUMINAMATH_CALUDE_sum_divisible_by_31_l2182_218289


namespace NUMINAMATH_CALUDE_intersection_M_N_l2182_218293

def M : Set ℝ := {x | x > -3}
def N : Set ℝ := {x | x ≥ 2}

theorem intersection_M_N : M ∩ N = Set.Ici 2 := by sorry

end NUMINAMATH_CALUDE_intersection_M_N_l2182_218293


namespace NUMINAMATH_CALUDE_two_from_ten_for_different_positions_l2182_218234

/-- The number of ways to choose k items from n items where order matters -/
def permutations (n k : ℕ) : ℕ := (n - k + 1).factorial / (n - k).factorial

/-- The number of ways to choose 2 people from 10 for 2 different positions -/
theorem two_from_ten_for_different_positions : permutations 10 2 = 90 := by
  sorry

end NUMINAMATH_CALUDE_two_from_ten_for_different_positions_l2182_218234


namespace NUMINAMATH_CALUDE_change_percentage_difference_l2182_218247

/- Given conditions -/
def initial_yes : ℚ := 60 / 100
def initial_no : ℚ := 40 / 100
def final_yes : ℚ := 80 / 100
def final_no : ℚ := 20 / 100
def new_students : ℚ := 10 / 100

/- Theorem statement -/
theorem change_percentage_difference :
  let min_change := (final_yes - new_students) - initial_yes
  let max_change := min initial_no (final_yes - new_students) + min initial_yes final_no
  max_change - min_change = 40 / 100 := by
  sorry


end NUMINAMATH_CALUDE_change_percentage_difference_l2182_218247


namespace NUMINAMATH_CALUDE_least_possible_area_l2182_218209

/-- The least possible length of a side when measured as 4 cm to the nearest centimeter -/
def min_side_length : ℝ := 3.5

/-- The measured length of the square's side to the nearest centimeter -/
def measured_side_length : ℕ := 4

/-- The least possible area of the square -/
def min_area : ℝ := min_side_length ^ 2

theorem least_possible_area :
  min_area = 12.25 := by sorry

end NUMINAMATH_CALUDE_least_possible_area_l2182_218209


namespace NUMINAMATH_CALUDE_jodi_walked_3_miles_week3_l2182_218220

/-- Represents the walking schedule of Jodi over 4 weeks -/
structure WalkingSchedule where
  weeks : Nat
  days_per_week : Nat
  miles_week1 : Nat
  miles_week2 : Nat
  miles_week4 : Nat
  total_miles : Nat

/-- Calculates the miles walked per day in the third week -/
def miles_per_day_week3 (schedule : WalkingSchedule) : Nat :=
  let miles_weeks_124 := schedule.miles_week1 * schedule.days_per_week +
                         schedule.miles_week2 * schedule.days_per_week +
                         schedule.miles_week4 * schedule.days_per_week
  let miles_week3 := schedule.total_miles - miles_weeks_124
  miles_week3 / schedule.days_per_week

/-- Theorem stating that Jodi walked 3 miles per day in the third week -/
theorem jodi_walked_3_miles_week3 (schedule : WalkingSchedule) 
  (h1 : schedule.weeks = 4)
  (h2 : schedule.days_per_week = 6)
  (h3 : schedule.miles_week1 = 1)
  (h4 : schedule.miles_week2 = 2)
  (h5 : schedule.miles_week4 = 4)
  (h6 : schedule.total_miles = 60) :
  miles_per_day_week3 schedule = 3 := by
  sorry

end NUMINAMATH_CALUDE_jodi_walked_3_miles_week3_l2182_218220


namespace NUMINAMATH_CALUDE_sum_of_fractions_equals_seven_l2182_218233

theorem sum_of_fractions_equals_seven : 
  let S := 1 / (4 - Real.sqrt 15) - 1 / (Real.sqrt 15 - Real.sqrt 14) + 
           1 / (Real.sqrt 14 - Real.sqrt 13) - 1 / (Real.sqrt 13 - Real.sqrt 12) + 
           1 / (Real.sqrt 12 - 3)
  S = 7 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_fractions_equals_seven_l2182_218233


namespace NUMINAMATH_CALUDE_apple_sharing_l2182_218235

theorem apple_sharing (total_apples : ℕ) (num_friends : ℕ) (apples_per_friend : ℕ) :
  total_apples = 9 →
  num_friends = 3 →
  total_apples = num_friends * apples_per_friend →
  apples_per_friend = 3 := by
  sorry

end NUMINAMATH_CALUDE_apple_sharing_l2182_218235


namespace NUMINAMATH_CALUDE_problem_solution_l2182_218298

theorem problem_solution (x : ℝ) (h1 : x < 0) (h2 : 1 / (x + 1 / (x + 2)) = 2) : x + 7/2 = 2 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l2182_218298


namespace NUMINAMATH_CALUDE_vector_coordinates_proof_l2182_218212

theorem vector_coordinates_proof (a : ℝ × ℝ) (b : ℝ × ℝ) :
  let x := a.1
  let y := a.2
  b = (1, 2) →
  Real.sqrt (x^2 + y^2) = 3 →
  x * b.1 + y * b.2 = 0 →
  (x = -6 * Real.sqrt 5 / 5 ∧ y = 3 * Real.sqrt 5 / 5) ∨
  (x = 6 * Real.sqrt 5 / 5 ∧ y = -3 * Real.sqrt 5 / 5) :=
by sorry

end NUMINAMATH_CALUDE_vector_coordinates_proof_l2182_218212


namespace NUMINAMATH_CALUDE_total_commute_time_is_16_l2182_218287

/-- Time it takes Roque to walk to work (in hours) -/
def walk_time : ℕ := 2

/-- Time it takes Roque to bike to work (in hours) -/
def bike_time : ℕ := 1

/-- Number of times Roque walks to and from work per week -/
def walk_frequency : ℕ := 3

/-- Number of times Roque bikes to and from work per week -/
def bike_frequency : ℕ := 2

/-- Total time Roque spends commuting in a week -/
def total_commute_time : ℕ := (walk_time * walk_frequency * 2) + (bike_time * bike_frequency * 2)

theorem total_commute_time_is_16 : total_commute_time = 16 := by
  sorry

end NUMINAMATH_CALUDE_total_commute_time_is_16_l2182_218287


namespace NUMINAMATH_CALUDE_tangent_intersection_y_coord_l2182_218243

/-- Given two points on the parabola y = x^2 + 1 with perpendicular tangents,
    the y-coordinate of their intersection is 3/4 -/
theorem tangent_intersection_y_coord (a b : ℝ) : 
  (2 * a) * (2 * b) = -1 →  -- Perpendicular tangents condition
  (∃ (x : ℝ), (2 * a) * (x - a) + a^2 + 1 = (2 * b) * (x - b) + b^2 + 1) →  -- Intersection exists
  (2 * a) * ((a + b) / 2 - a) + a^2 + 1 = 3 / 4 :=
by sorry

end NUMINAMATH_CALUDE_tangent_intersection_y_coord_l2182_218243


namespace NUMINAMATH_CALUDE_pure_imaginary_complex_number_l2182_218251

theorem pure_imaginary_complex_number (m : ℝ) :
  let z : ℂ := (m^2 - 1) + (m + 1) * Complex.I
  (z.re = 0 ∧ z ≠ 0) → m = 1 := by sorry

end NUMINAMATH_CALUDE_pure_imaginary_complex_number_l2182_218251


namespace NUMINAMATH_CALUDE_missing_legos_l2182_218266

theorem missing_legos (total : ℕ) (in_box : ℕ) : 
  total = 500 → in_box = 245 → (total / 2 - in_box : ℤ) = 5 := by
  sorry

end NUMINAMATH_CALUDE_missing_legos_l2182_218266


namespace NUMINAMATH_CALUDE_range_of_f_l2182_218285

-- Define the function f
def f (x : ℝ) : ℝ := x^2 + 2*x

-- Define the domain
def domain : Set ℝ := {x | -2 ≤ x ∧ x ≤ 1}

-- Theorem statement
theorem range_of_f :
  {y | ∃ x ∈ domain, f x = y} = {y | -1 ≤ y ∧ y ≤ 3} := by sorry

end NUMINAMATH_CALUDE_range_of_f_l2182_218285
