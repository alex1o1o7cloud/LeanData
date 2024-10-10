import Mathlib

namespace perimeter_of_new_arrangement_l2505_250501

/-- Represents a square arrangement -/
structure SquareArrangement where
  rows : ℕ
  columns : ℕ

/-- Calculates the perimeter of a square arrangement -/
def perimeter (arrangement : SquareArrangement) : ℕ :=
  2 * (arrangement.rows + arrangement.columns)

/-- The original square arrangement -/
def original : SquareArrangement :=
  { rows := 3, columns := 5 }

/-- The new square arrangement with an additional row -/
def new : SquareArrangement :=
  { rows := original.rows + 1, columns := original.columns }

theorem perimeter_of_new_arrangement :
  perimeter new = 37 := by
  sorry


end perimeter_of_new_arrangement_l2505_250501


namespace sin_750_degrees_l2505_250533

theorem sin_750_degrees : Real.sin (750 * π / 180) = 1 / 2 := by
  sorry

end sin_750_degrees_l2505_250533


namespace minimum_value_of_fraction_l2505_250592

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, q > 0 ∧ ∀ n : ℕ, a (n + 1) = q * a n

theorem minimum_value_of_fraction (a : ℕ → ℝ) (m n : ℕ) :
  geometric_sequence a →
  (∀ k : ℕ, a k > 0) →
  a 7 = a 6 + 2 * a 5 →
  ∃ m n : ℕ, Real.sqrt (a m * a n) = 4 * a 1 →
  (1 : ℝ) / m + 9 / n ≥ 8 / 3 :=
sorry

end minimum_value_of_fraction_l2505_250592


namespace count_squares_with_six_black_l2505_250583

/-- Represents a square on the checkerboard -/
structure Square where
  size : Nat
  x : Nat
  y : Nat

/-- The size of the checkerboard -/
def boardSize : Nat := 10

/-- Checks if a square contains at least 6 black squares -/
def containsSixBlackSquares (s : Square) : Bool :=
  if s.size ≥ 5 then true
  else if s.size = 4 then (s.x + s.y) % 2 = 0
  else false

/-- Counts the number of squares containing at least 6 black squares -/
def countSquaresWithSixBlack : Nat :=
  let fourByFour := (boardSize - 3) * (boardSize - 3) / 2
  let fiveByFive := (boardSize - 4) * (boardSize - 4)
  let sixBySix := (boardSize - 5) * (boardSize - 5)
  let sevenBySeven := (boardSize - 6) * (boardSize - 6)
  let eightByEight := (boardSize - 7) * (boardSize - 7)
  let nineByNine := (boardSize - 8) * (boardSize - 8)
  let tenByTen := 1
  fourByFour + fiveByFive + sixBySix + sevenBySeven + eightByEight + nineByNine + tenByTen

theorem count_squares_with_six_black :
  countSquaresWithSixBlack = 115 := by
  sorry

end count_squares_with_six_black_l2505_250583


namespace households_with_only_bike_l2505_250584

-- Define the total number of households
def total_households : ℕ := 90

-- Define the number of households without car or bike
def households_without_car_or_bike : ℕ := 11

-- Define the number of households with both car and bike
def households_with_both : ℕ := 16

-- Define the number of households with a car
def households_with_car : ℕ := 44

-- Theorem to prove
theorem households_with_only_bike : 
  total_households - households_without_car_or_bike - households_with_car + households_with_both = 35 := by
  sorry

end households_with_only_bike_l2505_250584


namespace youseff_distance_to_office_l2505_250549

theorem youseff_distance_to_office (x : ℝ) 
  (walk_time : ℝ → ℝ) 
  (bike_time : ℝ → ℝ) 
  (h1 : ∀ d, walk_time d = d) 
  (h2 : ∀ d, bike_time d = d / 3) 
  (h3 : walk_time x = bike_time x + 14) : 
  x = 21 := by
sorry

end youseff_distance_to_office_l2505_250549


namespace garrett_peanut_granola_bars_l2505_250534

/-- The number of granola bars Garrett bought in total -/
def total_granola_bars : ℕ := 14

/-- The number of oatmeal raisin granola bars Garrett bought -/
def oatmeal_raisin_bars : ℕ := 6

/-- The number of peanut granola bars Garrett bought -/
def peanut_granola_bars : ℕ := total_granola_bars - oatmeal_raisin_bars

theorem garrett_peanut_granola_bars : peanut_granola_bars = 8 := by
  sorry

end garrett_peanut_granola_bars_l2505_250534


namespace sum_of_threes_place_values_l2505_250517

def number : ℕ := 63130

def first_three_place_value : ℕ := 3000
def second_three_place_value : ℕ := 30

theorem sum_of_threes_place_values :
  first_three_place_value + second_three_place_value = 3030 :=
by sorry

end sum_of_threes_place_values_l2505_250517


namespace complex_fraction_sum_zero_l2505_250528

theorem complex_fraction_sum_zero : 
  let i : ℂ := Complex.I
  ((1 + i) / (1 - i)) ^ 2017 + ((1 - i) / (1 + i)) ^ 2017 = 0 := by
  sorry

end complex_fraction_sum_zero_l2505_250528


namespace impossible_inequalities_l2505_250569

theorem impossible_inequalities (a b c : ℝ) : ¬(|a| < |b - c| ∧ |b| < |c - a| ∧ |c| < |a - b|) := by
  sorry

end impossible_inequalities_l2505_250569


namespace total_games_attended_l2505_250523

def games_this_month : ℕ := 11
def games_last_month : ℕ := 17
def games_next_month : ℕ := 16

theorem total_games_attended : games_this_month + games_last_month + games_next_month = 44 := by
  sorry

end total_games_attended_l2505_250523


namespace burrito_count_l2505_250563

theorem burrito_count (cheese_per_burrito cheese_per_taco total_cheese : ℕ) 
  (h1 : cheese_per_burrito = 4)
  (h2 : cheese_per_taco = 9)
  (h3 : total_cheese = 37) :
  ∃ (num_burritos : ℕ), 
    num_burritos * cheese_per_burrito + cheese_per_taco = total_cheese ∧ 
    num_burritos = 7 := by
  sorry

end burrito_count_l2505_250563


namespace pascals_triangle_15_numbers_4th_entry_l2505_250512

theorem pascals_triangle_15_numbers_4th_entry : 
  let n : ℕ := 14  -- The row number (15 numbers, so it's the 14th row)
  let k : ℕ := 4   -- The position of the number we're looking for
  Nat.choose (n - 1) (k - 1) = 286 := by
sorry

end pascals_triangle_15_numbers_4th_entry_l2505_250512


namespace coconut_trips_l2505_250541

/-- The number of trips needed to move coconuts -/
def num_trips (total_coconuts : ℕ) (barbie_capacity : ℕ) (bruno_capacity : ℕ) : ℕ :=
  total_coconuts / (barbie_capacity + bruno_capacity)

/-- Theorem stating that 12 trips are needed to move 144 coconuts -/
theorem coconut_trips : num_trips 144 4 8 = 12 := by
  sorry

end coconut_trips_l2505_250541


namespace ellipse_foci_l2505_250589

/-- The ellipse equation -/
def ellipse_equation (x y : ℝ) : Prop :=
  x^2 / 16 + y^2 / 25 = 1

/-- The coordinates of a focus of the ellipse -/
def is_focus (x y : ℝ) : Prop :=
  (x = 0 ∧ y = 3) ∨ (x = 0 ∧ y = -3)

/-- Theorem stating that the given coordinates are the foci of the ellipse -/
theorem ellipse_foci :
  ∀ x y : ℝ, ellipse_equation x y → is_focus x y :=
sorry

end ellipse_foci_l2505_250589


namespace max_value_of_f_on_interval_l2505_250577

-- Define the function f(x) = x^3 - 3x^2
def f (x : ℝ) : ℝ := x^3 - 3*x^2

-- Define the interval [-2, 4]
def interval : Set ℝ := Set.Icc (-2) 4

-- State the theorem
theorem max_value_of_f_on_interval :
  ∃ (c : ℝ), c ∈ interval ∧ ∀ (x : ℝ), x ∈ interval → f x ≤ f c ∧ f c = 16 :=
sorry

end max_value_of_f_on_interval_l2505_250577


namespace minor_arc_circumference_l2505_250545

theorem minor_arc_circumference (r : ℝ) (θ : ℝ) (h_r : r = 12) (h_θ : θ = 110 * π / 180) :
  let circle_circumference := 2 * π * r
  let arc_length := circle_circumference * θ / (2 * π)
  arc_length = 22 * π / 3 := by
  sorry

end minor_arc_circumference_l2505_250545


namespace skateboard_ramp_speeds_l2505_250530

theorem skateboard_ramp_speeds (S₁ S₂ S₃ : ℝ) :
  (S₁ + S₂ + S₃) / 3 + 4 = 40 →
  ∃ (T₁ T₂ T₃ : ℝ), (T₁ + T₂ + T₃) / 3 + 4 = 40 ∧ (T₁ ≠ S₁ ∨ T₂ ≠ S₂ ∨ T₃ ≠ S₃) :=
by sorry

end skateboard_ramp_speeds_l2505_250530


namespace apples_per_child_l2505_250598

theorem apples_per_child (total_apples : ℕ) (num_children : ℕ) (num_adults : ℕ) (apples_per_adult : ℕ)
  (h1 : total_apples = 450)
  (h2 : num_children = 33)
  (h3 : num_adults = 40)
  (h4 : apples_per_adult = 3) :
  (total_apples - num_adults * apples_per_adult) / num_children = 10 := by
sorry

end apples_per_child_l2505_250598


namespace car_replacement_cost_l2505_250581

/-- Given an old car worth $20,000 sold at 80% of its value and a new car with
    a sticker price of $30,000 bought at 90% of its value, prove that the
    difference in cost (out of pocket) is $11,000. -/
theorem car_replacement_cost (old_car_value : ℝ) (new_car_price : ℝ)
    (old_car_sale_percentage : ℝ) (new_car_buy_percentage : ℝ)
    (h1 : old_car_value = 20000)
    (h2 : new_car_price = 30000)
    (h3 : old_car_sale_percentage = 0.8)
    (h4 : new_car_buy_percentage = 0.9) :
    new_car_buy_percentage * new_car_price - old_car_sale_percentage * old_car_value = 11000 :=
by sorry

end car_replacement_cost_l2505_250581


namespace range_of_b_minus_a_l2505_250579

-- Define the function f
def f (x : ℝ) : ℝ := x^2 - 2*x

-- State the theorem
theorem range_of_b_minus_a (a b : ℝ) :
  (∀ x ∈ Set.Icc a b, f x ∈ Set.Icc (-1) 3) →
  (∃ x ∈ Set.Icc a b, f x = -1) →
  (∃ x ∈ Set.Icc a b, f x = 3) →
  b - a ∈ Set.Icc 2 4 :=
by sorry

end range_of_b_minus_a_l2505_250579


namespace correct_operation_l2505_250518

theorem correct_operation (a : ℝ) : 2 * a^2 * a = 2 * a^3 := by
  sorry

end correct_operation_l2505_250518


namespace smallest_N_proof_l2505_250536

/-- The smallest number of pies per batch that satisfies the conditions --/
def smallest_N : ℕ := 80

/-- The number of batches of pies --/
def num_batches : ℕ := 21

/-- The number of pies per tray --/
def pies_per_tray : ℕ := 70

theorem smallest_N_proof :
  (∀ N : ℕ, N > 70 → (num_batches * N) % pies_per_tray = 0 → N ≥ smallest_N) ∧
  smallest_N > 70 ∧
  (num_batches * smallest_N) % pies_per_tray = 0 :=
sorry

end smallest_N_proof_l2505_250536


namespace function_equality_implies_k_range_l2505_250510

open Real

/-- Given a function f(x) = 1 + ln x + kx where k is a real number,
    if there exists a positive x such that e^x = f(x)/x, then k ≥ 1 -/
theorem function_equality_implies_k_range (k : ℝ) :
  (∃ x > 0, exp x = (1 + log x + k * x) / x) → k ≥ 1 := by
  sorry

end function_equality_implies_k_range_l2505_250510


namespace log_function_range_l2505_250553

/-- The function f(x) = lg(ax^2 - 2x + 2) has a range of ℝ if and only if a ∈ (0, 1/2] -/
theorem log_function_range (a : ℝ) : 
  (∀ y : ℝ, ∃ x : ℝ, y = Real.log (a * x^2 - 2 * x + 2)) ↔ (0 < a ∧ a ≤ 1/2) :=
sorry

end log_function_range_l2505_250553


namespace janet_movie_cost_l2505_250504

/-- Calculates the total cost of filming Janet's newest movie given the following conditions:
  * Janet's previous movie was 2 hours long
  * The new movie is 60% longer than the previous movie
  * The previous movie cost $50 per minute to film
  * The new movie cost twice as much per minute to film as the previous movie
-/
def total_cost_newest_movie (previous_movie_length : Real) 
                            (length_increase_percent : Real)
                            (previous_cost_per_minute : Real)
                            (new_cost_multiplier : Real) : Real :=
  let new_movie_length := previous_movie_length * (1 + length_increase_percent)
  let new_movie_length_minutes := new_movie_length * 60
  let new_cost_per_minute := previous_cost_per_minute * new_cost_multiplier
  new_movie_length_minutes * new_cost_per_minute

theorem janet_movie_cost :
  total_cost_newest_movie 2 0.6 50 2 = 19200 := by
  sorry

end janet_movie_cost_l2505_250504


namespace equation_solution_l2505_250515

theorem equation_solution : ∃ X : ℝ, 
  1.5 * ((3.6 * X * 2.50) / (0.12 * 0.09 * 0.5)) = 1200.0000000000002 ∧ 
  X = 0.4800000000000001 := by
  sorry

end equation_solution_l2505_250515


namespace jacket_cost_l2505_250557

/-- Represents the cost of clothing items and shipments -/
structure ClothingCost where
  sweater : ℝ
  jacket : ℝ

/-- Represents a shipment of clothing items -/
structure Shipment where
  sweaters : ℕ
  jackets : ℕ
  totalCost : ℝ

/-- The problem statement -/
theorem jacket_cost (cost : ClothingCost) (shipment1 shipment2 : Shipment) :
  shipment1.sweaters = 10 →
  shipment1.jackets = 20 →
  shipment1.totalCost = 800 →
  shipment2.sweaters = 5 →
  shipment2.jackets = 15 →
  shipment2.totalCost = 550 →
  shipment1.sweaters * cost.sweater + shipment1.jackets * cost.jacket = shipment1.totalCost →
  shipment2.sweaters * cost.sweater + shipment2.jackets * cost.jacket = shipment2.totalCost →
  cost.jacket = 30 := by
  sorry


end jacket_cost_l2505_250557


namespace red_cards_taken_out_l2505_250561

/-- Represents a standard deck of cards -/
structure Deck :=
  (total_cards : ℕ)
  (red_cards : ℕ)
  (h_total : total_cards = 52)
  (h_red : red_cards = total_cards / 2)

/-- Represents the state after some red cards were taken out -/
structure RemainingCards :=
  (remaining_red : ℕ)
  (h_remaining : remaining_red = 16)

theorem red_cards_taken_out (d : Deck) (r : RemainingCards) :
  d.red_cards - r.remaining_red = 10 := by
  sorry

end red_cards_taken_out_l2505_250561


namespace wall_length_proof_l2505_250500

def men_group1 : ℕ := 20
def men_group2 : ℕ := 86
def days : ℕ := 8
def wall_length_group2 : ℝ := 283.8

def wall_length_group1 : ℝ := 65.7

theorem wall_length_proof :
  (men_group1 * days * wall_length_group2) / (men_group2 * days) = wall_length_group1 := by
  sorry

end wall_length_proof_l2505_250500


namespace two_solutions_set_equiv_l2505_250544

/-- The set of values for 'a' that satisfy the conditions for two distinct solutions -/
def TwoSolutionsSet : Set ℝ :=
  {a | 9 * (a - 2) > 0 ∧ 
       a > 0 ∧ 
       a^2 - 9*a + 18 > 0 ∧
       a ≠ 11 ∧
       ∃ (x y : ℝ), x ≠ y ∧ x = a + 3 * Real.sqrt (a - 2) ∧ y = a - 3 * Real.sqrt (a - 2)}

/-- The theorem stating the equivalence of the solution set -/
theorem two_solutions_set_equiv :
  TwoSolutionsSet = {a | (2 < a ∧ a < 3) ∨ (6 < a ∧ a < 11) ∨ (11 < a)} :=
by sorry

end two_solutions_set_equiv_l2505_250544


namespace prob_three_odd_dice_l2505_250548

/-- The number of dice being rolled -/
def num_dice : ℕ := 4

/-- The number of sides on each die -/
def num_sides : ℕ := 8

/-- The probability of rolling an odd number on a single die -/
def prob_odd : ℚ := 1/2

/-- The probability of rolling an even number on a single die -/
def prob_even : ℚ := 1/2

/-- The number of ways to choose 3 dice out of 4 -/
def choose_3_from_4 : ℕ := 4

theorem prob_three_odd_dice :
  (choose_3_from_4 : ℚ) * prob_odd^3 * prob_even^(num_dice - 3) = 1/4 := by
  sorry

end prob_three_odd_dice_l2505_250548


namespace solve_linear_equation_l2505_250532

theorem solve_linear_equation (x : ℝ) (h : 3*x - 4*x + 7*x = 120) : x = 20 := by
  sorry

end solve_linear_equation_l2505_250532


namespace vector_ratio_implies_k_l2505_250580

/-- Given vectors a and b in ℝ², if (a + 2b) / (3a - b) exists, then k = -6 -/
theorem vector_ratio_implies_k (a b : ℝ × ℝ) (k : ℝ) 
  (h1 : a = (1, 3))
  (h2 : b = (-2, k))
  (h3 : ∃ (r : ℝ), r • (3 • a - b) = a + 2 • b) :
  k = -6 := by
  sorry

end vector_ratio_implies_k_l2505_250580


namespace triangle_angle_side_inequality_l2505_250505

/-- Theorem: For any triangle, the weighted sum of angles divided by the sum of sides 
    is bounded between π/3 and π/2 -/
theorem triangle_angle_side_inequality (A B C a b c : ℝ) : 
  0 < a ∧ 0 < b ∧ 0 < c →  -- sides are positive
  a + b > c ∧ b + c > a ∧ c + a > b →  -- triangle inequality
  A + B + C = π →  -- sum of angles
  0 < A ∧ 0 < B ∧ 0 < C →  -- angles are positive
  π / 3 ≤ (A * a + B * b + C * c) / (a + b + c) ∧ 
  (A * a + B * b + C * c) / (a + b + c) < π / 2 := by
sorry

end triangle_angle_side_inequality_l2505_250505


namespace square_sum_reciprocal_l2505_250555

theorem square_sum_reciprocal (x : ℝ) (h : x + (1 / x) = 5) : x^2 + (1 / x)^2 = 23 := by
  sorry

end square_sum_reciprocal_l2505_250555


namespace factorial_ratio_l2505_250576

theorem factorial_ratio : Nat.factorial 5 / Nat.factorial (5 - 3) = 60 := by
  sorry

end factorial_ratio_l2505_250576


namespace joan_snow_volume_l2505_250538

/-- The volume of snow on a rectangular driveway -/
def snow_volume (length width depth : ℚ) : ℚ :=
  length * width * depth

/-- Proof that the volume of snow on Joan's driveway is 90 cubic feet -/
theorem joan_snow_volume :
  snow_volume 40 3 (3/4) = 90 := by
  sorry

end joan_snow_volume_l2505_250538


namespace alloy_mixture_problem_l2505_250513

/-- Proves that the amount of the first alloy used is 15 kg given the conditions of the problem -/
theorem alloy_mixture_problem (x : ℝ) : 
  (0.10 * x + 0.08 * 35 = 0.086 * (x + 35)) → x = 15 := by
  sorry

end alloy_mixture_problem_l2505_250513


namespace triangle_abc_proof_l2505_250511

theorem triangle_abc_proof (A B C : ℝ) (a b c : ℝ) (m n : ℝ × ℝ) :
  0 < A ∧ A < π →
  m = (Real.cos A, Real.sin A) →
  n = (Real.sqrt 2 - Real.sin A, Real.cos A) →
  Real.sqrt ((m.1 + n.1)^2 + (m.2 + n.2)^2) = 2 →
  b = 4 * Real.sqrt 2 →
  c = Real.sqrt 2 * a →
  A = π / 4 ∧ (1/2 * b * a = 16) := by sorry

end triangle_abc_proof_l2505_250511


namespace arithmetic_sequence_l2505_250551

def a (n : ℕ) : ℤ := 3 * n + 1

theorem arithmetic_sequence : ∀ n : ℕ, a (n + 1) - a n = 3 := by
  sorry

end arithmetic_sequence_l2505_250551


namespace rectangle_longer_side_length_l2505_250594

/-- Given a circle and rectangle with specific properties, prove the length of the rectangle's longer side --/
theorem rectangle_longer_side_length (r : ℝ) (circle_area rectangle_area : ℝ) (shorter_side longer_side : ℝ) : 
  r = 6 →  -- Circle radius is 6 cm
  circle_area = π * r^2 →  -- Area of the circle
  rectangle_area = 3 * circle_area →  -- Rectangle area is three times circle area
  shorter_side = 2 * r →  -- Shorter side is twice the radius
  rectangle_area = shorter_side * longer_side →  -- Rectangle area formula
  longer_side = 9 * π := by
  sorry

end rectangle_longer_side_length_l2505_250594


namespace geometric_sequence_problem_l2505_250527

theorem geometric_sequence_problem (a : ℝ) (h1 : a > 0) 
  (h2 : ∃ r : ℝ, r > 0 ∧ a = 30 * r ∧ 7/4 = a * r) : a = 7.5 := by
  sorry

end geometric_sequence_problem_l2505_250527


namespace prob_at_least_one_of_B_or_C_given_A_l2505_250526

/-- The probability of selecting at least one of boy B and girl C, given boy A is already selected -/
theorem prob_at_least_one_of_B_or_C_given_A (total_boys : Nat) (total_girls : Nat) 
  (representatives : Nat) (h1 : total_boys = 5) (h2 : total_girls = 2) (h3 : representatives = 3) :
  let remaining_boys := total_boys - 1
  let remaining_total := total_boys + total_girls - 1
  let total_ways := Nat.choose remaining_total (representatives - 1)
  let ways_without_B_or_C := Nat.choose (remaining_boys - 1) (representatives - 1) + 
                             Nat.choose (remaining_boys - 1) (representatives - 2) * total_girls
  (1 : ℚ) - (ways_without_B_or_C : ℚ) / total_ways = 1 / 15 := by
  sorry

end prob_at_least_one_of_B_or_C_given_A_l2505_250526


namespace tower_count_remainder_l2505_250578

/-- Represents a cube with an edge length --/
structure Cube where
  edge_length : ℕ

/-- Represents a tower of cubes --/
inductive Tower : Type
  | empty : Tower
  | cons : Cube → Tower → Tower

/-- Checks if a tower is valid according to the rules --/
def is_valid_tower : Tower → Bool
  | Tower.empty => true
  | Tower.cons c Tower.empty => true
  | Tower.cons c1 (Tower.cons c2 t) =>
    c1.edge_length ≤ c2.edge_length + 3 && is_valid_tower (Tower.cons c2 t)

/-- The set of cubes with edge lengths from 1 to 10 --/
def cube_set : List Cube :=
  List.map (λ k => ⟨k⟩) (List.range 10)

/-- Counts the number of valid towers that can be constructed --/
def count_valid_towers (cubes : List Cube) : ℕ :=
  sorry  -- Implementation details omitted

/-- The main theorem --/
theorem tower_count_remainder (U : ℕ) :
  U = count_valid_towers cube_set →
  U % 1000 = 536 :=
sorry

end tower_count_remainder_l2505_250578


namespace rectangle_perimeter_l2505_250506

/-- A rectangle with an inscribed ellipse -/
structure RectangleWithEllipse where
  -- Rectangle dimensions
  x : ℝ
  y : ℝ
  -- Ellipse semi-major and semi-minor axes
  a : ℝ
  b : ℝ
  -- Conditions
  rectangle_area : x * y = 4024
  ellipse_area : π * a * b = 4024 * π
  foci_distance : x^2 + y^2 = 4 * (a^2 - b^2)
  major_axis : x + y = 2 * a

/-- The perimeter of a rectangle with an inscribed ellipse is 8√2012 -/
theorem rectangle_perimeter (r : RectangleWithEllipse) : r.x + r.y = 8 * Real.sqrt 2012 := by
  sorry

end rectangle_perimeter_l2505_250506


namespace handshake_count_l2505_250546

/-- Represents a gathering of married couples -/
structure Gathering where
  couples : ℕ
  no_male_handshakes : Bool

/-- Calculates the total number of handshakes in the gathering -/
def total_handshakes (g : Gathering) : ℕ :=
  let women := g.couples
  let men := g.couples
  let women_handshakes := women.choose 2
  let men_women_handshakes := men * (women - 1)
  women_handshakes + men_women_handshakes

/-- Theorem stating that in a gathering of 15 married couples with the given conditions, 
    the total number of handshakes is 315 -/
theorem handshake_count (g : Gathering) :
  g.couples = 15 ∧ g.no_male_handshakes = true → total_handshakes g = 315 := by
  sorry

end handshake_count_l2505_250546


namespace colored_balls_theorem_l2505_250597

/-- Represents a box of colored balls -/
structure ColoredBalls where
  total : ℕ
  colors : ℕ
  min_same_color : ℕ → ℕ

/-- The problem statement -/
theorem colored_balls_theorem (box : ColoredBalls) 
  (h_total : box.total = 100)
  (h_colors : box.colors = 3)
  (h_min_same_color : box.min_same_color 26 ≥ 10) :
  box.min_same_color 66 ≥ 30 := by
  sorry

end colored_balls_theorem_l2505_250597


namespace product_b3_b17_l2505_250574

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

def geometric_sequence (b : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r ≠ 0 ∧ ∀ n : ℕ, b (n + 1) = b n * r

theorem product_b3_b17 (a b : ℕ → ℝ) 
    (h_arith : arithmetic_sequence a)
    (h_geom : geometric_sequence b)
    (h_eq : 3 * a 1 - (a 8)^2 + 3 * a 15 = 0)
    (h_a8_b10 : a 8 = b 10) :
  b 3 * b 17 = 36 := by
sorry

end product_b3_b17_l2505_250574


namespace cloth_cost_calculation_l2505_250570

theorem cloth_cost_calculation (length : Real) (price_per_meter : Real) :
  length = 9.25 ∧ price_per_meter = 43 → length * price_per_meter = 397.75 := by
  sorry

end cloth_cost_calculation_l2505_250570


namespace triangle_angle_sum_bound_l2505_250550

theorem triangle_angle_sum_bound (A B C : Real) (h_triangle : A + B + C = Real.pi)
  (h_sin_sum : Real.sin A + Real.sin B + Real.sin C ≤ 1) :
  min (A + B) (min (B + C) (C + A)) < Real.pi / 6 := by
  sorry

end triangle_angle_sum_bound_l2505_250550


namespace board_cut_theorem_l2505_250562

theorem board_cut_theorem (total_length : ℝ) (shorter_length : ℝ) :
  total_length = 20 ∧
  shorter_length > 0 ∧
  shorter_length < total_length ∧
  2 * shorter_length = (total_length - shorter_length) + 4 →
  shorter_length = 8 := by
  sorry

end board_cut_theorem_l2505_250562


namespace solve_for_P_l2505_250543

theorem solve_for_P : ∃ P : ℝ, (P ^ 3) ^ (1/2) = 9 * (81 ^ (1/6)) → P = 3 ^ (16/9) := by
  sorry

end solve_for_P_l2505_250543


namespace solution_set_inequality_l2505_250508

theorem solution_set_inequality (a : ℝ) (h : 0 < a ∧ a < 1) :
  {x : ℝ | (a - x) * (x - 1/a) > 0} = {x : ℝ | a < x ∧ x < 1/a} := by
  sorry

end solution_set_inequality_l2505_250508


namespace absolute_value_equation_product_l2505_250509

theorem absolute_value_equation_product (x : ℝ) : 
  (∃ x₁ x₂ : ℝ, (|5 * x₁| + 2 = 47 ∧ |5 * x₂| + 2 = 47) ∧ x₁ ≠ x₂ ∧ x₁ * x₂ = -81) := by
  sorry

end absolute_value_equation_product_l2505_250509


namespace common_chord_circle_equation_l2505_250514

-- Define the two circles
def circle_C1 (x y : ℝ) : Prop := x^2 + y^2 - 12*x - 2*y - 13 = 0
def circle_C2 (x y : ℝ) : Prop := x^2 + y^2 + 12*x + 16*y - 25 = 0

-- Define the result circle
def result_circle (x y : ℝ) : Prop := (x - 2)^2 + (y + 2)^2 = 25

-- Theorem statement
theorem common_chord_circle_equation :
  ∀ x y : ℝ,
  (∃ x₁ y₁ x₂ y₂ : ℝ,
    circle_C1 x₁ y₁ ∧ circle_C2 x₂ y₂ ∧
    (x₁ - x₂)^2 + (y₁ - y₂)^2 > 0 ∧
    (x - (x₁ + x₂)/2)^2 + (y - (y₁ + y₂)/2)^2 = ((x₁ - x₂)^2 + (y₁ - y₂)^2) / 4) →
  result_circle x y :=
by sorry

end common_chord_circle_equation_l2505_250514


namespace marble_count_l2505_250571

theorem marble_count (red_marbles : ℕ) (green_marbles : ℕ) (yellow_marbles : ℕ) (total_marbles : ℕ) :
  red_marbles = 20 →
  green_marbles = 3 * red_marbles →
  yellow_marbles = (20 * green_marbles) / 100 →
  total_marbles = green_marbles + 3 * green_marbles →
  total_marbles - red_marbles - green_marbles - yellow_marbles = 148 :=
by sorry

end marble_count_l2505_250571


namespace chris_birthday_money_l2505_250568

/-- Calculates the total amount of money Chris has after receiving birthday gifts -/
def total_money (initial_amount grandmother_gift aunt_uncle_gift parents_gift : ℕ) : ℕ :=
  initial_amount + grandmother_gift + aunt_uncle_gift + parents_gift

/-- Proves that Chris's total money after receiving gifts is correct -/
theorem chris_birthday_money :
  total_money 159 25 20 75 = 279 := by
  sorry

end chris_birthday_money_l2505_250568


namespace medicine_box_theorem_l2505_250502

/-- Represents the number of tablets of each medicine type in a box -/
structure MedicineBox where
  tabletA : ℕ
  tabletB : ℕ

/-- Calculates the minimum number of tablets to extract to ensure at least two of each type -/
def minExtract (box : MedicineBox) : ℕ :=
  box.tabletA + 6

theorem medicine_box_theorem (box : MedicineBox) 
  (h1 : box.tabletA = 10)
  (h2 : minExtract box = 16) :
  box.tabletB ≥ 4 := by
  sorry

end medicine_box_theorem_l2505_250502


namespace gcd_840_1764_l2505_250587

theorem gcd_840_1764 : Nat.gcd 840 1764 = 84 := by
  sorry

end gcd_840_1764_l2505_250587


namespace distance_sum_inequality_l2505_250585

theorem distance_sum_inequality (b : ℝ) (h : b > 0) :
  (∃ x : ℝ, |x - 5| + |x - 7| < b) ↔ b > 2 := by sorry

end distance_sum_inequality_l2505_250585


namespace negation_existence_gt_one_l2505_250565

theorem negation_existence_gt_one :
  (¬ ∃ x : ℝ, x > 1) ↔ (∀ x : ℝ, x ≤ 1) := by sorry

end negation_existence_gt_one_l2505_250565


namespace prob_A_shot_twice_correct_l2505_250573

def prob_A : ℚ := 3/4
def prob_B : ℚ := 4/5

def prob_A_shot_twice : ℚ := 19/400

theorem prob_A_shot_twice_correct :
  let p_A_miss := 1 - prob_A
  let p_B_miss := 1 - prob_B
  prob_A_shot_twice = p_A_miss * p_B_miss * prob_A + p_A_miss * p_B_miss * p_A_miss * prob_B :=
by sorry

end prob_A_shot_twice_correct_l2505_250573


namespace problem_1_problem_2_l2505_250596

theorem problem_1 (x y : ℝ) (h1 : x - y = 3) (h2 : x * y = 2) :
  x^2 + y^2 = 13 := by sorry

theorem problem_2 (a : ℝ) (h : (4 - a)^2 + (a + 3)^2 = 7) :
  (4 - a) * (a + 3) = 21 := by sorry

end problem_1_problem_2_l2505_250596


namespace tens_digit_of_N_power_20_l2505_250582

theorem tens_digit_of_N_power_20 (N : ℕ) (h1 : Even N) (h2 : ¬ (10 ∣ N)) :
  (N^20 % 100) / 10 = 7 := by
  sorry

end tens_digit_of_N_power_20_l2505_250582


namespace x_squared_eq_zero_is_quadratic_l2505_250537

/-- Definition of a quadratic equation in one variable -/
def is_quadratic_equation (f : ℝ → ℝ) : Prop :=
  ∃ (a b c : ℝ), a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

/-- The function representing x^2 = 0 -/
def f (x : ℝ) : ℝ := x^2

/-- Theorem: x^2 = 0 is a quadratic equation in one variable -/
theorem x_squared_eq_zero_is_quadratic : is_quadratic_equation f :=
sorry

end x_squared_eq_zero_is_quadratic_l2505_250537


namespace equation_solutions_l2505_250560

theorem equation_solutions : 
  (∃ x1 x2 : ℝ, x1 = 2 + Real.sqrt 5 ∧ x2 = 2 - Real.sqrt 5 ∧ 
    x1^2 - 4*x1 - 1 = 0 ∧ x2^2 - 4*x2 - 1 = 0) ∧ 
  (∃ x1 x2 : ℝ, x1 = -3 ∧ x2 = 6 ∧ 
    (x1 + 3)*(x1 - 3) = 3*(x1 + 3) ∧ (x2 + 3)*(x2 - 3) = 3*(x2 + 3)) :=
by sorry

end equation_solutions_l2505_250560


namespace harry_bought_apples_l2505_250599

/-- The number of apples Harry initially had -/
def initial_apples : ℕ := 79

/-- The number of apples Harry ended up with -/
def final_apples : ℕ := 84

/-- The number of apples Harry bought -/
def bought_apples : ℕ := final_apples - initial_apples

theorem harry_bought_apples :
  bought_apples = final_apples - initial_apples :=
by sorry

end harry_bought_apples_l2505_250599


namespace function_properties_l2505_250540

-- Define the function f
def f (x : ℝ) : ℝ := |2 * x - 3|

-- State the theorem
theorem function_properties :
  (∀ x : ℝ, f x ≤ 1 ↔ 1 ≤ x ∧ x ≤ 2) ∧
  (∀ a b c x : ℝ, a > 0 → b > 0 → c > 0 → a + b + c = 1 →
    f x - 2 * |x + 3| ≤ 1/a + 1/b + 1/c) :=
by sorry

end function_properties_l2505_250540


namespace arctan_equation_solution_l2505_250519

theorem arctan_equation_solution :
  ∃ y : ℝ, y > 0 ∧ Real.arctan (2 / y) + Real.arctan (1 / y^2) = π / 4 :=
by
  -- The proof would go here
  sorry

#check arctan_equation_solution

end arctan_equation_solution_l2505_250519


namespace square_sum_reciprocal_l2505_250591

theorem square_sum_reciprocal (x : ℝ) (h : x + (1 / x) = 2.5) : x^2 + (1 / x^2) = 4.25 := by
  sorry

end square_sum_reciprocal_l2505_250591


namespace triangle_third_side_length_l2505_250572

theorem triangle_third_side_length 
  (a b c : ℝ) 
  (θ : ℝ) 
  (h1 : a = 5) 
  (h2 : b = 12) 
  (h3 : θ = Real.pi / 3) -- 60° in radians
  (h4 : c^2 = a^2 + b^2 - 2*a*b*(Real.cos θ)) -- Law of Cosines
  : c = Real.sqrt 109 := by
  sorry

end triangle_third_side_length_l2505_250572


namespace cookies_per_bag_l2505_250516

theorem cookies_per_bag (total_cookies : ℕ) (num_bags : ℕ) (h1 : total_cookies = 703) (h2 : num_bags = 37) :
  total_cookies / num_bags = 19 := by
  sorry

end cookies_per_bag_l2505_250516


namespace cuboid_reduction_impossibility_l2505_250556

theorem cuboid_reduction_impossibility (a b c a' b' c' : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (ha' : a' > 0) (hb' : b' > 0) (hc' : c' > 0)
  (haa' : a ≥ a') (hbb' : b ≥ b') (hcc' : c ≥ c') :
  ¬(a' * b' * c' = (1/2) * a * b * c ∧ 
    2 * (a' * b' + b' * c' + c' * a') = 2 * (a * b + b * c + c * a)) := by
  sorry

end cuboid_reduction_impossibility_l2505_250556


namespace angle_measure_proof_l2505_250567

theorem angle_measure_proof (x : ℝ) : 
  (90 - x + 40 = (180 - x) / 2) → x = 80 := by
  sorry

end angle_measure_proof_l2505_250567


namespace savings_comparison_l2505_250531

theorem savings_comparison (last_year_salary : ℝ) (last_year_savings_rate : ℝ) 
  (salary_increase_rate : ℝ) (this_year_savings_rate : ℝ) 
  (h1 : last_year_savings_rate = 0.06)
  (h2 : salary_increase_rate = 0.20)
  (h3 : this_year_savings_rate = 0.05) :
  (this_year_savings_rate * (1 + salary_increase_rate) * last_year_salary) / 
  (last_year_savings_rate * last_year_salary) = 1 := by
sorry

end savings_comparison_l2505_250531


namespace expand_expression_l2505_250575

theorem expand_expression (x : ℝ) : (9*x + 4) * (2*x^2) = 18*x^3 + 8*x^2 := by
  sorry

end expand_expression_l2505_250575


namespace max_sum_abs_on_unit_sphere_l2505_250522

theorem max_sum_abs_on_unit_sphere :
  ∃ (M : ℝ), M = 2 ∧
  (∀ x y z : ℝ, x^2 + y^2 + z^2 = 1 → |x| + |y| + |z| ≤ M) ∧
  (∃ x y z : ℝ, x^2 + y^2 + z^2 = 1 ∧ |x| + |y| + |z| = M) := by
  sorry

end max_sum_abs_on_unit_sphere_l2505_250522


namespace jeffrey_bottle_caps_l2505_250593

/-- 
Given that Jeffrey can create 6 groups of bottle caps with 2 bottle caps in each group,
prove that the total number of bottle caps is 12.
-/
theorem jeffrey_bottle_caps : 
  let groups : ℕ := 6
  let caps_per_group : ℕ := 2
  groups * caps_per_group = 12 := by sorry

end jeffrey_bottle_caps_l2505_250593


namespace binary_to_decimal_l2505_250503

theorem binary_to_decimal (b : List Bool) :
  (b.reverse.enum.map (λ (i, x) => if x then 2^i else 0)).sum = 45 :=
sorry

end binary_to_decimal_l2505_250503


namespace polynomial_division_remainder_l2505_250539

theorem polynomial_division_remainder : ∃ q : Polynomial ℝ, 
  3 * X^5 - 2 * X^3 + 5 * X - 8 = (X^2 - 3 * X + 2) * q + (74 * X - 76) := by
  sorry

end polynomial_division_remainder_l2505_250539


namespace a_value_is_two_l2505_250588

/-- The quadratic function we're considering -/
def f (a : ℝ) (x : ℝ) : ℝ := -2 * x^2 + a * x + 6

/-- The condition that f(a, x) > 0 only when x ∈ (-∞, -2) ∪ (3, ∞) -/
def condition (a : ℝ) : Prop :=
  ∀ x : ℝ, f a x > 0 ↔ (x < -2 ∨ x > 3)

/-- The theorem stating that under the given condition, a = 2 -/
theorem a_value_is_two :
  ∃ a : ℝ, condition a ∧ a = 2 := by sorry

end a_value_is_two_l2505_250588


namespace baker_eggs_theorem_l2505_250595

/-- Calculates the number of eggs needed for a given amount of flour, based on a recipe ratio. -/
def eggs_needed (recipe_flour : ℚ) (recipe_eggs : ℚ) (available_flour : ℚ) : ℚ :=
  (available_flour / recipe_flour) * recipe_eggs

theorem baker_eggs_theorem (recipe_flour : ℚ) (recipe_eggs : ℚ) (available_flour : ℚ) 
  (h1 : recipe_flour = 2)
  (h2 : recipe_eggs = 3)
  (h3 : available_flour = 6) :
  eggs_needed recipe_flour recipe_eggs available_flour = 9 := by
  sorry

#eval eggs_needed 2 3 6

end baker_eggs_theorem_l2505_250595


namespace pregnant_fish_count_l2505_250586

theorem pregnant_fish_count (tanks : ℕ) (young_per_fish : ℕ) (total_young : ℕ) :
  tanks = 3 →
  young_per_fish = 20 →
  total_young = 240 →
  ∃ fish_per_tank : ℕ, fish_per_tank * tanks * young_per_fish = total_young ∧ fish_per_tank = 4 :=
by sorry

end pregnant_fish_count_l2505_250586


namespace cricketer_average_score_l2505_250564

/-- Proves that the overall average score for a cricketer who played 7 matches
    with given averages for the first 4 and last 3 matches is 56. -/
theorem cricketer_average_score 
  (total_matches : ℕ)
  (first_matches : ℕ)
  (last_matches : ℕ)
  (first_average : ℚ)
  (last_average : ℚ)
  (h1 : total_matches = 7)
  (h2 : first_matches = 4)
  (h3 : last_matches = 3)
  (h4 : first_matches + last_matches = total_matches)
  (h5 : first_average = 46)
  (h6 : last_average = 69333333333333 / 1000000000000) : 
  (first_average * first_matches + last_average * last_matches) / total_matches = 56 := by
sorry

#eval (46 * 4 + 69333333333333 / 1000000000000 * 3) / 7

end cricketer_average_score_l2505_250564


namespace equality_of_cyclic_system_l2505_250535

theorem equality_of_cyclic_system (x y z : ℝ) 
  (eq1 : x^3 = 2*y - 1)
  (eq2 : y^3 = 2*z - 1)
  (eq3 : z^3 = 2*x - 1) :
  x = y ∧ y = z := by
  sorry

end equality_of_cyclic_system_l2505_250535


namespace electricity_relationship_l2505_250554

/-- Represents the relationship between electricity consumption and fee -/
structure ElectricityRelation where
  consumption : ℝ  -- Electricity consumption in kWh
  fee : ℝ          -- Electricity fee in yuan
  linear : fee = 0.55 * consumption  -- Linear relationship

/-- Proves the functional relationship and calculates consumption for a given fee -/
theorem electricity_relationship (r : ElectricityRelation) :
  r.fee = 0.55 * r.consumption ∧ 
  (r.fee = 40.7 → r.consumption = 74) := by
  sorry

#check electricity_relationship

end electricity_relationship_l2505_250554


namespace two_face_painted_count_l2505_250507

/-- Represents a 3x3x3 cube made up of smaller cubes --/
structure Cube3x3x3 where
  /-- The total number of smaller cubes --/
  total_cubes : Nat
  /-- All outer faces of the large cube are painted --/
  outer_faces_painted : Bool

/-- Counts the number of smaller cubes painted on exactly two faces --/
def count_two_face_painted (c : Cube3x3x3) : Nat :=
  12

/-- Theorem stating that in a 3x3x3 painted cube, 12 smaller cubes are painted on exactly two faces --/
theorem two_face_painted_count (c : Cube3x3x3) 
    (h1 : c.total_cubes = 27) 
    (h2 : c.outer_faces_painted = true) : 
  count_two_face_painted c = 12 := by
  sorry

end two_face_painted_count_l2505_250507


namespace chromosomal_variations_l2505_250529

/-- Represents a biological process or condition -/
inductive BiologicalProcess
| AntherCulture
| DNABaseChange
| NonHomologousRecombination
| CrossingOver
| DownSyndrome

/-- Defines what constitutes a chromosomal variation -/
def isChromosomalVariation (p : BiologicalProcess) : Prop :=
  match p with
  | BiologicalProcess.AntherCulture => true
  | BiologicalProcess.DNABaseChange => false
  | BiologicalProcess.NonHomologousRecombination => false
  | BiologicalProcess.CrossingOver => false
  | BiologicalProcess.DownSyndrome => true

/-- The main theorem stating which processes are chromosomal variations -/
theorem chromosomal_variations :
  (isChromosomalVariation BiologicalProcess.AntherCulture) ∧
  (¬ isChromosomalVariation BiologicalProcess.DNABaseChange) ∧
  (¬ isChromosomalVariation BiologicalProcess.NonHomologousRecombination) ∧
  (¬ isChromosomalVariation BiologicalProcess.CrossingOver) ∧
  (isChromosomalVariation BiologicalProcess.DownSyndrome) :=
by sorry

end chromosomal_variations_l2505_250529


namespace smallest_number_with_remainder_two_l2505_250590

theorem smallest_number_with_remainder_two : ∃ n : ℕ,
  n > 2 ∧
  n % 9 = 2 ∧
  n % 10 = 2 ∧
  n % 11 = 2 ∧
  (∀ m : ℕ, m > 2 ∧ m % 9 = 2 ∧ m % 10 = 2 ∧ m % 11 = 2 → m ≥ n) ∧
  n = 992 :=
by sorry

end smallest_number_with_remainder_two_l2505_250590


namespace hyperbola_eccentricity_l2505_250547

/-- Hyperbola with given properties and intersecting circle -/
structure HyperbolaWithCircle where
  b : ℝ
  h_b_pos : b > 0
  hyperbola : ℝ × ℝ → Prop := fun (x, y) ↦ x^2 - y^2/b^2 = 1
  asymptote : ℝ → ℝ := fun x ↦ b * x
  circle : ℝ × ℝ → Prop := fun (x, y) ↦ x^2 + y^2 = 1
  intersection_area : ℝ := b

/-- The eccentricity of a hyperbola with the given properties is √3 -/
theorem hyperbola_eccentricity (h : HyperbolaWithCircle) : 
  ∃ (e : ℝ), e = Real.sqrt 3 ∧ e^2 = 1 + 1/h.b^2 :=
sorry

end hyperbola_eccentricity_l2505_250547


namespace horizontal_line_inclination_l2505_250552

def line (x y : ℝ) : Prop := y + 3 = 0

def angle_of_inclination (f : ℝ → ℝ → Prop) : ℝ := sorry

theorem horizontal_line_inclination :
  angle_of_inclination line = 0 := by sorry

end horizontal_line_inclination_l2505_250552


namespace number_of_pupils_is_40_l2505_250524

/-- The number of pupils in a class, given a specific mark entry error and its effect on the class average. -/
def number_of_pupils : ℕ :=
  let incorrect_mark : ℕ := 83
  let correct_mark : ℕ := 63
  let average_increase : ℚ := 1/2
  40

/-- Theorem stating that the number of pupils is 40 under the given conditions. -/
theorem number_of_pupils_is_40 :
  let n := number_of_pupils
  let incorrect_mark : ℕ := 83
  let correct_mark : ℕ := 63
  let mark_difference : ℕ := incorrect_mark - correct_mark
  let average_increase : ℚ := 1/2
  (mark_difference : ℚ) / n = average_increase → n = 40 := by
  sorry

end number_of_pupils_is_40_l2505_250524


namespace triangle_side_value_l2505_250558

open Real

/-- Prove that in triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    if a*sin(B) = √2*sin(C), cos(C) = 1/3, and the area of the triangle is 4, then c = 6. -/
theorem triangle_side_value (a b c : ℝ) (A B C : ℝ) :
  a * sin B = sqrt 2 * sin C →
  cos C = 1 / 3 →
  1 / 2 * a * b * sin C = 4 →
  c = 6 := by
sorry

end triangle_side_value_l2505_250558


namespace female_fraction_is_25_69_l2505_250520

/-- Represents the basketball club membership data --/
structure ClubData where
  maleLastYear : ℕ
  totalIncrease : ℚ
  maleIncrease : ℚ
  femaleIncrease : ℚ

/-- Calculates the fraction of female members this year --/
def femaleFraction (data : ClubData) : ℚ :=
  let maleThisYear := data.maleLastYear * (1 + data.maleIncrease)
  let femaleLastYear := (data.maleLastYear : ℚ) * (1 + data.totalIncrease - 1) / (data.femaleIncrease - 1)
  let femaleThisYear := femaleLastYear * (1 + data.femaleIncrease)
  let totalThisYear := maleThisYear + femaleThisYear
  femaleThisYear / totalThisYear

/-- Theorem stating that given the conditions, the fraction of female members this year is 25/69 --/
theorem female_fraction_is_25_69 (data : ClubData) 
  (h1 : data.maleLastYear = 30)
  (h2 : data.totalIncrease = 0.15)
  (h3 : data.maleIncrease = 0.10)
  (h4 : data.femaleIncrease = 0.25) :
  femaleFraction data = 25 / 69 := by
  sorry


end female_fraction_is_25_69_l2505_250520


namespace exists_different_reassembled_triangle_l2505_250521

/-- A triangle represented by its three vertices in 2D space -/
structure Triangle :=
  (A B C : ℝ × ℝ)

/-- A function that cuts a triangle into two parts -/
def cut (t : Triangle) : (Triangle × Triangle) :=
  sorry

/-- A function that reassembles two triangles into one -/
def reassemble (t1 t2 : Triangle) : Triangle :=
  sorry

/-- Theorem stating that there exists a triangle that can be cut and reassembled into a different triangle -/
theorem exists_different_reassembled_triangle :
  ∃ (t : Triangle), ∃ (t1 t2 : Triangle),
    (cut t = (t1, t2)) ∧ (reassemble t1 t2 ≠ t) := by
  sorry

end exists_different_reassembled_triangle_l2505_250521


namespace height_comparison_l2505_250525

theorem height_comparison (p q : ℝ) (h : p = 0.6 * q) :
  (q - p) / p = 2/3 := by sorry

end height_comparison_l2505_250525


namespace max_value_theorem_l2505_250566

/-- Represents a digit (0-9) -/
def Digit := Fin 10

/-- Checks if all given digits are distinct -/
def distinct (x y z w : Digit) : Prop :=
  x ≠ y ∧ x ≠ z ∧ x ≠ w ∧ y ≠ z ∧ y ≠ w ∧ z ≠ w

/-- Converts a four-digit number to its integer representation -/
def toInt (a b c d : Digit) : Nat :=
  1000 * a.val + 100 * b.val + 10 * c.val + d.val

/-- Main theorem -/
theorem max_value_theorem (x y z w v_1 v_2 v_3 v_4 : Digit) :
  distinct x y z w →
  (x.val * y.val * z.val + w.val = toInt v_1 v_2 v_3 v_4) →
  ∀ (a b c d : Digit), distinct a b c d →
    (a.val * b.val * c.val + d.val ≤ toInt v_1 v_2 v_3 v_4) →
  toInt v_1 v_2 v_3 v_4 = 9898 ∧ w.val = 98 := by
  sorry

end max_value_theorem_l2505_250566


namespace fraction_sum_l2505_250559

theorem fraction_sum (a b : ℝ) (h : a / b = 2 / 3) : (a + b) / b = 5 / 3 := by
  sorry

end fraction_sum_l2505_250559


namespace raisin_cost_fraction_l2505_250542

/-- The fraction of the total cost that is the cost of raisins in a mixture of raisins and nuts -/
theorem raisin_cost_fraction (raisin_pounds : ℝ) (nut_pounds : ℝ) (raisin_cost : ℝ) :
  raisin_pounds = 3 →
  nut_pounds = 4 →
  raisin_cost > 0 →
  (raisin_pounds * raisin_cost) / ((raisin_pounds * raisin_cost) + (nut_pounds * (2 * raisin_cost))) = 3 / 11 := by
  sorry

#check raisin_cost_fraction

end raisin_cost_fraction_l2505_250542
