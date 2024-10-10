import Mathlib

namespace poem_line_addition_l423_42380

theorem poem_line_addition (initial_lines : ℕ) (months : ℕ) (final_lines : ℕ) 
  (h1 : initial_lines = 24)
  (h2 : months = 22)
  (h3 : final_lines = 90) :
  (final_lines - initial_lines) / months = 3 :=
by
  sorry

end poem_line_addition_l423_42380


namespace brass_players_count_l423_42330

/-- Represents the composition of a marching band -/
structure MarchingBand where
  brass : ℕ
  woodwind : ℕ
  percussion : ℕ

/-- Checks if the given marching band composition is valid -/
def isValidBand (band : MarchingBand) : Prop :=
  band.woodwind = 2 * band.brass ∧
  band.percussion = 4 * band.woodwind ∧
  band.brass + band.woodwind + band.percussion = 110

theorem brass_players_count (band : MarchingBand) (h : isValidBand band) : band.brass = 10 := by
  sorry

#check brass_players_count

end brass_players_count_l423_42330


namespace bombay_express_speed_l423_42368

/-- The speed of Bombay Express in km/h -/
def speed_bombay_express : ℝ := 60

/-- The speed of Rajdhani Express in km/h -/
def speed_rajdhani_express : ℝ := 80

/-- The distance at which the trains meet in km -/
def meeting_distance : ℝ := 480

/-- The time difference between the departures of the two trains in hours -/
def time_difference : ℝ := 2

theorem bombay_express_speed :
  speed_bombay_express = meeting_distance / (meeting_distance / speed_rajdhani_express + time_difference) :=
by sorry

end bombay_express_speed_l423_42368


namespace train_speed_before_increase_l423_42315

/-- The average speed before a train's speed increase, given the travel times before and after
    the increase, and the amount of speed increase. -/
theorem train_speed_before_increase 
  (time_after : ℝ) 
  (time_before : ℝ) 
  (speed_increase : ℝ) 
  (h1 : time_after = 10) 
  (h2 : time_before = 12) 
  (h3 : speed_increase = 20) :
  let speed_before := (time_after * (time_before * speed_increase) / (time_before - time_after))
  speed_before = 100 := by
sorry

end train_speed_before_increase_l423_42315


namespace cyclic_sum_inequality_l423_42344

theorem cyclic_sum_inequality (a b c d : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) :
  (a - b) / (b + c) + (b - c) / (c + d) + (c - d) / (d + a) + (d - a) / (a + b) ≥ 0 ∧
  ((a - b) / (b + c) + (b - c) / (c + d) + (c - d) / (d + a) + (d - a) / (a + b) = 0 ↔ a = b ∧ b = c ∧ c = d) :=
by sorry

end cyclic_sum_inequality_l423_42344


namespace doll_difference_l423_42384

/-- The number of dolls Jane and Jill have together -/
def total_dolls : ℕ := 32

/-- The number of dolls Jane has -/
def jane_dolls : ℕ := 13

/-- The number of dolls Jill has -/
def jill_dolls : ℕ := total_dolls - jane_dolls

/-- Jill has more dolls than Jane -/
axiom jill_has_more : jill_dolls > jane_dolls

theorem doll_difference : jill_dolls - jane_dolls = 6 := by
  sorry

end doll_difference_l423_42384


namespace max_true_statements_l423_42311

theorem max_true_statements (y : ℝ) : 
  let statements := [
    (1 < y^2 ∧ y^2 < 4),
    (y^2 > 4),
    (-2 < y ∧ y < 0),
    (0 < y ∧ y < 2),
    (0 < y^3 - y^2 ∧ y^3 - y^2 < 4)
  ]
  ∃ (true_statements : List Bool), 
    (∀ i, true_statements.get! i = true → statements.get! i) ∧
    true_statements.count true ≤ 3 ∧
    ∀ (other_true_statements : List Bool),
      (∀ i, other_true_statements.get! i = true → statements.get! i) →
      other_true_statements.count true ≤ true_statements.count true :=
by sorry


end max_true_statements_l423_42311


namespace expression_equals_36_75_l423_42320

-- Define a function to convert a number from any base to base 10
def toBase10 (digits : List Nat) (base : Nat) : Nat :=
  digits.enum.foldr (fun (i, d) acc => acc + d * base ^ i) 0

-- Define the expression
def expression : ℚ :=
  (toBase10 [3, 4, 6] 8 : ℚ) / (toBase10 [1, 5] 3) +
  (toBase10 [2, 0, 4] 5 : ℚ) / (toBase10 [1, 2] 4) - 1

-- State the theorem
theorem expression_equals_36_75 : expression = 36.75 := by
  sorry

end expression_equals_36_75_l423_42320


namespace parallelogram_area_l423_42358

/-- The area of a parallelogram with base 32 cm and height 15 cm is 480 cm². -/
theorem parallelogram_area : 
  ∀ (base height area : ℝ), 
  base = 32 → height = 15 → area = base * height → area = 480 :=
by sorry

end parallelogram_area_l423_42358


namespace millionaire_allocation_l423_42325

/-- The number of ways to allocate millionaires to hotel rooms -/
def allocate_millionaires (n : ℕ) (k : ℕ) : ℕ :=
  Nat.choose (n - 1) (k - 1)

/-- Theorem: There are 36 ways to allocate 13 millionaires to 3 types of rooms -/
theorem millionaire_allocation :
  allocate_millionaires 10 3 = 36 :=
sorry

#eval allocate_millionaires 10 3

end millionaire_allocation_l423_42325


namespace path_count_l423_42310

/-- A simple directed graph with vertices A, B, C, and D -/
structure Graph :=
  (paths_AB : ℕ)
  (paths_BC : ℕ)
  (paths_CD : ℕ)
  (direct_AC : ℕ)

/-- The total number of paths from A to D in the graph -/
def total_paths (g : Graph) : ℕ :=
  g.paths_AB * g.paths_BC * g.paths_CD + g.direct_AC * g.paths_CD

/-- Theorem stating that the total number of paths from A to D is 15 -/
theorem path_count (g : Graph) 
  (h1 : g.paths_AB = 2)
  (h2 : g.paths_BC = 2)
  (h3 : g.paths_CD = 3)
  (h4 : g.direct_AC = 1) : 
  total_paths g = 15 := by
  sorry

end path_count_l423_42310


namespace f_properties_l423_42332

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x / a - Real.exp x

def hasExactlyTwoZeroPoints (f : ℝ → ℝ) : Prop :=
  ∃ x₁ x₂, x₁ ≠ x₂ ∧ f x₁ = 0 ∧ f x₂ = 0 ∧ ∀ x, f x = 0 → x = x₁ ∨ x = x₂

theorem f_properties (a : ℝ) (h : a > 0) :
  (∀ x ∈ Set.Icc 1 2, f a x ≤ (
    if a ≤ Real.exp (-2) then 2 / a - Real.exp 2
    else if a < Real.exp (-1) then (Real.log (1 / a)) / a - 1 / a
    else 1 / a - Real.exp 1
  )) ∧
  (hasExactlyTwoZeroPoints (f a) ↔ 0 < a ∧ a < Real.exp (-1)) := by
  sorry

end f_properties_l423_42332


namespace run_distance_proof_l423_42375

/-- Calculates the total distance run given a running speed and movie lengths -/
def total_distance_run (speed : ℚ) (movie_lengths : List ℚ) : ℚ :=
  (movie_lengths.sum) / speed

/-- Theorem stating that given the specified running speed and movie lengths, 
    the total distance run is 41 miles -/
theorem run_distance_proof : 
  let speed : ℚ := 12  -- 12 minutes per mile
  let movie_lengths : List ℚ := [96, 138, 108, 150]
  total_distance_run speed movie_lengths = 41 := by
  sorry

#eval total_distance_run 12 [96, 138, 108, 150]

end run_distance_proof_l423_42375


namespace sixth_number_is_52_when_i_7_l423_42390

/-- Represents the systematic sampling function for a population of 100 individuals -/
def systematicSample (i : Nat) (k : Nat) : Nat :=
  let drawnNumber := i + k
  if drawnNumber ≥ 10 then drawnNumber - 10 else drawnNumber

/-- Theorem stating that the 6th number drawn is 52 when i=7 -/
theorem sixth_number_is_52_when_i_7 :
  ∀ (populationSize : Nat) (segmentCount : Nat) (sampleSize : Nat) (i : Nat),
    populationSize = 100 →
    segmentCount = 10 →
    sampleSize = 10 →
    i = 7 →
    systematicSample i 5 = 52 := by
  sorry

end sixth_number_is_52_when_i_7_l423_42390


namespace triangle_side_ratio_l423_42389

theorem triangle_side_ratio (a b c k : ℝ) 
  (triangle_sides : a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b > c ∧ b + c > a ∧ c + a > b) 
  (side_ratio : a^2 + b^2 = k * c^2) : 
  k > 0.5 := by
sorry

end triangle_side_ratio_l423_42389


namespace bailey_towel_discount_percentage_l423_42306

/-- Calculates the discount percentage for Bailey's towel purchase. -/
theorem bailey_towel_discount_percentage : 
  let guest_sets : ℕ := 2
  let master_sets : ℕ := 4
  let guest_price : ℚ := 40
  let master_price : ℚ := 50
  let total_spent : ℚ := 224
  let original_total : ℚ := guest_sets * guest_price + master_sets * master_price
  let discount_amount : ℚ := original_total - total_spent
  let discount_percentage : ℚ := (discount_amount / original_total) * 100
  discount_percentage = 20 := by
  sorry

end bailey_towel_discount_percentage_l423_42306


namespace conference_handshakes_l423_42302

/-- The number of handshakes in a conference with n attendees -/
def handshakes (n : ℕ) : ℕ := n * (n - 1) / 2

/-- The number of attendees at the conference -/
def attendees : ℕ := 10

theorem conference_handshakes :
  handshakes attendees = 45 := by
  sorry


end conference_handshakes_l423_42302


namespace log_problem_l423_42387

theorem log_problem :
  let x := (Real.log 2 / Real.log 8) ^ (Real.log 8 / Real.log 2)
  Real.log x / Real.log 3 = -3 := by
  sorry

end log_problem_l423_42387


namespace lamp_arrangement_count_l423_42385

def number_of_lamps : ℕ := 10
def lamps_to_turn_off : ℕ := 3
def available_positions : ℕ := number_of_lamps - 2 - lamps_to_turn_off + 1

theorem lamp_arrangement_count : 
  Nat.choose available_positions lamps_to_turn_off = 20 := by
  sorry

end lamp_arrangement_count_l423_42385


namespace shaded_area_is_42_l423_42395

/-- Square ABCD with shaded regions -/
structure ShadedSquare where
  /-- Side length of the square ABCD -/
  side_length : ℕ
  /-- Side length of the first small square -/
  small_square_side : ℕ
  /-- Width of the second shaded region -/
  second_width : ℕ
  /-- Height of the second shaded region -/
  second_height : ℕ
  /-- Width of the third shaded region -/
  third_width : ℕ
  /-- Height of the third shaded region -/
  third_height : ℕ
  /-- Assumption that the side length is 7 -/
  h_side_length : side_length = 7
  /-- Assumption that the small square side length is 1 -/
  h_small_square : small_square_side = 1
  /-- Assumption about the second shaded region dimensions -/
  h_second : second_width = 2 ∧ second_height = 4
  /-- Assumption about the third shaded region dimensions -/
  h_third : third_width = 3 ∧ third_height = 6

/-- The shaded area of the square ABCD is 42 square units -/
theorem shaded_area_is_42 (sq : ShadedSquare) : 
  sq.small_square_side ^ 2 +
  (sq.small_square_side + sq.second_width) * (sq.small_square_side + sq.second_height) - sq.small_square_side ^ 2 +
  (sq.small_square_side + sq.second_width + sq.third_width) * (sq.small_square_side + sq.second_height + sq.third_height) -
  (sq.small_square_side + sq.second_width) * (sq.small_square_side + sq.second_height) = 42 :=
sorry

end shaded_area_is_42_l423_42395


namespace walk_time_calculation_l423_42391

/-- The time it takes Mark to walk into the courthouse each day -/
def walk_time : ℕ := sorry

/-- The number of work days in a week -/
def work_days : ℕ := 5

/-- The time it takes to find parking each day -/
def parking_time : ℕ := 5

/-- The time it takes to get through the metal detector on crowded days -/
def crowded_detector_time : ℕ := 30

/-- The time it takes to get through the metal detector on less crowded days -/
def less_crowded_detector_time : ℕ := 10

/-- The number of crowded days per week -/
def crowded_days : ℕ := 2

/-- The number of less crowded days per week -/
def less_crowded_days : ℕ := 3

/-- The total time spent on all activities in a week -/
def total_weekly_time : ℕ := 130

theorem walk_time_calculation : 
  walk_time = 3 ∧
  work_days * (parking_time + walk_time) + 
  crowded_days * crowded_detector_time +
  less_crowded_days * less_crowded_detector_time = 
  total_weekly_time :=
sorry

end walk_time_calculation_l423_42391


namespace cube_sum_over_product_equals_33_l423_42331

theorem cube_sum_over_product_equals_33 
  (a b c : ℂ) 
  (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0)
  (sum_eq : a + b + c = 30)
  (sq_diff_eq : (a - b)^2 + (a - c)^2 + (b - c)^2 = 2*a*b*c) :
  (a^3 + b^3 + c^3) / (a*b*c) = 33 := by
  sorry

end cube_sum_over_product_equals_33_l423_42331


namespace range_of_product_l423_42349

theorem range_of_product (a b : ℝ) (h1 : |a| ≤ 1) (h2 : |a + b| ≤ 1) :
  ∃ (min max : ℝ), min = -2 ∧ max = 9/4 ∧
  ∀ x, x = (a + 1) * (b + 1) → min ≤ x ∧ x ≤ max :=
sorry

end range_of_product_l423_42349


namespace absolute_sum_diff_equal_implies_product_zero_l423_42342

theorem absolute_sum_diff_equal_implies_product_zero (a b : ℝ) :
  |a + b| = |a - b| → a * b = 0 := by
  sorry

end absolute_sum_diff_equal_implies_product_zero_l423_42342


namespace regular_octagon_interior_angle_measure_l423_42317

-- Define a regular octagon
structure RegularOctagon where
  -- No specific fields needed for this problem

-- Define the measure of an interior angle of a regular octagon
def interior_angle_measure (o : RegularOctagon) : ℝ := 135

-- Theorem statement
theorem regular_octagon_interior_angle_measure (o : RegularOctagon) :
  interior_angle_measure o = 135 := by
  sorry

end regular_octagon_interior_angle_measure_l423_42317


namespace discount_difference_equals_582_l423_42351

def initial_bill : ℝ := 12000

def single_discount_rate : ℝ := 0.45
def successive_discount_rates : List ℝ := [0.30, 0.10, 0.05]

def apply_single_discount (bill : ℝ) (rate : ℝ) : ℝ :=
  bill * (1 - rate)

def apply_successive_discounts (bill : ℝ) (rates : List ℝ) : ℝ :=
  rates.foldl (fun acc rate => acc * (1 - rate)) bill

theorem discount_difference_equals_582 :
  apply_successive_discounts initial_bill successive_discount_rates -
  apply_single_discount initial_bill single_discount_rate = 582 := by
    sorry

end discount_difference_equals_582_l423_42351


namespace water_left_over_l423_42364

theorem water_left_over (total_water : ℕ) (num_players : ℕ) (water_per_player : ℕ) (water_spilled : ℕ) : 
  total_water = 8 →
  num_players = 30 →
  water_per_player = 200 →
  water_spilled = 250 →
  total_water * 1000 - (num_players * water_per_player + water_spilled) = 1750 :=
by
  sorry

end water_left_over_l423_42364


namespace condition_A_sufficient_not_necessary_for_B_l423_42355

theorem condition_A_sufficient_not_necessary_for_B :
  (∀ a b : ℝ, a > b ∧ b > 0 → 1 / a < 1 / b) ∧
  (∃ a b : ℝ, 1 / a < 1 / b ∧ ¬(a > b ∧ b > 0)) := by
  sorry

end condition_A_sufficient_not_necessary_for_B_l423_42355


namespace negation_of_existence_negation_of_quadratic_inequality_l423_42372

theorem negation_of_existence (P : ℝ → Prop) :
  (¬ ∃ x : ℝ, P x) ↔ (∀ x : ℝ, ¬ P x) := by sorry

theorem negation_of_quadratic_inequality :
  (¬ ∃ x : ℝ, x^2 + 1 > 3*x) ↔ (∀ x : ℝ, x^2 + 1 ≤ 3*x) := by sorry

end negation_of_existence_negation_of_quadratic_inequality_l423_42372


namespace necessary_not_sufficient_l423_42373

/-- A point in the 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Predicate for a point being on the curve y = 4x^2 -/
def OnCurve (p : Point) : Prop :=
  p.y = 4 * p.x^2

/-- Predicate for a point satisfying the equation √y = 2x -/
def SatisfiesEquation (p : Point) : Prop :=
  Real.sqrt p.y = 2 * p.x

theorem necessary_not_sufficient :
  (∀ p : Point, OnCurve p → SatisfiesEquation p) ∧
  (∃ p : Point, SatisfiesEquation p ∧ ¬OnCurve p) := by
  sorry

end necessary_not_sufficient_l423_42373


namespace T_increasing_T_binary_ones_M_properties_l423_42379

/-- A sequence of positive integers with exactly 9 ones in binary representation -/
def T : ℕ → ℕ := sorry

/-- The 1500th term in the sequence T -/
def M : ℕ := T 1500

/-- The remainder when M is divided by 500 -/
def r : ℕ := M % 500

theorem T_increasing (n : ℕ) : T n < T (n + 1) := sorry

theorem T_binary_ones (n : ℕ) : 
  (Nat.digits 2 (T n)).count 1 = 9 := sorry

theorem M_properties : 
  ∃ (M : ℕ), 
    M = T 1500 ∧ 
    (∀ n < 1500, T n < M) ∧
    (Nat.digits 2 M).count 1 = 9 ∧
    M % 500 = r := by sorry

end T_increasing_T_binary_ones_M_properties_l423_42379


namespace last_locker_opened_l423_42321

/-- Represents the state of a locker (open or closed) -/
inductive LockerState
| Open
| Closed

/-- Represents the corridor with lockers -/
def Corridor := Fin 512 → LockerState

/-- The initial state of the corridor with all lockers closed -/
def initialCorridor : Corridor := fun _ => LockerState.Closed

/-- Represents a single pass of opening lockers -/
def openLockersPass (c : Corridor) (start : Nat) (step : Nat) : Corridor :=
  fun n => if (n.val - start) % step = 0 then LockerState.Open else c n

/-- Represents the process of opening lockers in multiple passes -/
def openLockers (c : Corridor) : Corridor :=
  -- Implementation details omitted
  sorry

/-- The theorem stating that the last locker to be opened is 342 -/
theorem last_locker_opened (c : Corridor) :
  openLockers initialCorridor (⟨341, sorry⟩ : Fin 512) = LockerState.Closed ∧
  openLockers initialCorridor (⟨342, sorry⟩ : Fin 512) = LockerState.Open :=
by sorry

end last_locker_opened_l423_42321


namespace train_length_l423_42337

/-- Given a train that crosses a platform and a signal pole, prove its length. -/
theorem train_length (platform_length : ℝ) (platform_time : ℝ) (pole_time : ℝ)
  (h1 : platform_length = 550)
  (h2 : platform_time = 51)
  (h3 : pole_time = 18) :
  ∃ (train_length : ℝ), train_length = 300 ∧ 
    train_length + platform_length = (train_length / pole_time) * platform_time :=
by sorry

end train_length_l423_42337


namespace rectangular_plot_breadth_l423_42394

theorem rectangular_plot_breadth (length width area : ℝ) : 
  length = 3 * width →
  area = length * width →
  area = 432 →
  width = 12 := by
sorry

end rectangular_plot_breadth_l423_42394


namespace construct_incenter_l423_42363

-- Define the basic constructions available
class Constructible (α : Type*) where
  draw_line : α → α → Prop
  draw_circle : α → α → Prop
  mark_intersection : Prop

-- Define a triangle
structure Triangle (α : Type*) where
  A : α
  B : α
  C : α

-- Define the incenter
def Incenter (α : Type*) (t : Triangle α) : α := sorry

-- Theorem statement
theorem construct_incenter 
  {α : Type*} [Constructible α] (t : Triangle α) :
  ∃ (I : α), I = Incenter α t := by sorry

end construct_incenter_l423_42363


namespace modulo_seven_equivalence_l423_42323

theorem modulo_seven_equivalence : 
  ∃! n : ℤ, 0 ≤ n ∧ n < 7 ∧ -1234 ≡ n [ZMOD 7] ∧ n = 5 := by
  sorry

end modulo_seven_equivalence_l423_42323


namespace total_apple_weight_marta_apple_purchase_l423_42350

/-- The weight of one apple in ounces -/
def apple_weight : ℕ := 4

/-- The weight of one orange in ounces -/
def orange_weight : ℕ := 3

/-- The maximum weight a bag can hold in ounces -/
def bag_capacity : ℕ := 49

/-- The number of bags Marta wants to buy -/
def num_bags : ℕ := 3

/-- The number of apples in one bag -/
def apples_per_bag : ℕ := 7

theorem total_apple_weight :
  apple_weight * (apples_per_bag * num_bags) = 84 :=
by sorry

/-- The main theorem stating the total weight of apples Marta should buy -/
theorem marta_apple_purchase :
  ∃ (x : ℕ), x = apple_weight * (apples_per_bag * num_bags) ∧
  x ≤ bag_capacity * num_bags ∧
  x = 84 :=
by sorry

end total_apple_weight_marta_apple_purchase_l423_42350


namespace total_sewing_time_is_40_hours_l423_42334

/-- Represents the time (in hours) to sew a given number of items -/
def sewing_time (time_per_item : ℝ) (num_items : ℕ) : ℝ :=
  time_per_item * (num_items : ℝ)

/-- Proves that the total sewing time for skirts and coats is 40 hours -/
theorem total_sewing_time_is_40_hours 
  (skirt_time : ℝ) 
  (coat_time : ℝ) 
  (num_skirts : ℕ) 
  (num_coats : ℕ) 
  (h1 : skirt_time = 2)
  (h2 : coat_time = 7)
  (h3 : num_skirts = 6)
  (h4 : num_coats = 4) : 
  sewing_time skirt_time num_skirts + sewing_time coat_time num_coats = 40 := by
  sorry

#check total_sewing_time_is_40_hours

end total_sewing_time_is_40_hours_l423_42334


namespace number_problem_l423_42383

theorem number_problem : ∃ x : ℚ, x - (3/5) * x = 60 ∧ x = 150 := by
  sorry

end number_problem_l423_42383


namespace parabola_and_circle_equations_l423_42301

/-- Given a line l and a parabola C, prove the equations of C and a circle Q -/
theorem parabola_and_circle_equations 
  (l : Real → Real) 
  (C : Real → Real → Prop)
  (p : Real)
  (A B : Real × Real)
  (h1 : ∀ x y, l y = x - 1)
  (h2 : ∀ x y, C x y ↔ y^2 = 2*p*x)
  (h3 : p > 0)
  (h4 : C (A.1) (A.2) ∧ C (B.1) (B.2))
  (h5 : l (A.2) = A.1 - 1 ∧ l (B.2) = B.1 - 1)
  (h6 : l 0 = 1) :
  (∀ x y, C x y ↔ y^2 = 4*x) ∧
  (∀ x y, (x - 3)^2 + (y - 2)^2 = 16 ↔ 
    (x - (A.1 + B.1)/2)^2 + (y - (A.2 + B.2)/2)^2 = ((A.1 - B.1)^2 + (A.2 - B.2)^2)/4) := by
  sorry

end parabola_and_circle_equations_l423_42301


namespace real_part_of_i_squared_times_one_minus_two_i_l423_42382

theorem real_part_of_i_squared_times_one_minus_two_i (i : ℂ) : i ^ 2 = -1 → Complex.re (i ^ 2 * (1 - 2 * i)) = -1 := by
  sorry

end real_part_of_i_squared_times_one_minus_two_i_l423_42382


namespace line_segment_difference_l423_42360

theorem line_segment_difference (L₁ L₂ : ℝ) : 
  L₁ = 7 → 
  L₁^2 - L₂^2 = 32 → 
  L₁ - L₂ = 7 - Real.sqrt 17 := by
  sorry

end line_segment_difference_l423_42360


namespace simpson_paradox_possible_l423_42346

/-- Represents the number of patients and successful treatments for a medication in a hospital -/
structure HospitalData where
  patients : ℕ
  successes : ℕ
  hLe : successes ≤ patients

/-- Calculates the effectiveness rate of a medication -/
def effectivenessRate (data : HospitalData) : ℚ :=
  data.successes / data.patients

theorem simpson_paradox_possible 
  (h1A h1B h2A h2B : HospitalData) 
  (h1_effectiveness : effectivenessRate h1A > effectivenessRate h1B)
  (h2_effectiveness : effectivenessRate h2A > effectivenessRate h2B) :
  ∃ (h1A h1B h2A h2B : HospitalData),
    effectivenessRate h1A > effectivenessRate h1B ∧
    effectivenessRate h2A > effectivenessRate h2B ∧
    effectivenessRate (HospitalData.mk (h1A.patients + h2A.patients) (h1A.successes + h2A.successes) sorry) <
    effectivenessRate (HospitalData.mk (h1B.patients + h2B.patients) (h1B.successes + h2B.successes) sorry) :=
  sorry

end simpson_paradox_possible_l423_42346


namespace number_operations_problem_l423_42319

theorem number_operations_problem (x : ℚ) : 
  (((11 * x + 6) / 5) - 42 = 12) ↔ (x = 24) :=
by sorry

end number_operations_problem_l423_42319


namespace gcd_117_182_f_neg_one_l423_42333

-- Define the polynomial f(x)
def f (x : ℤ) : ℤ := 1 - 9*x + 8*x^2 - 4*x^4 + 5*x^5 + 3*x^6

-- Theorem for the GCD of 117 and 182
theorem gcd_117_182 : Nat.gcd 117 182 = 13 := by sorry

-- Theorem for the value of f(-1)
theorem f_neg_one : f (-1) = 12 := by sorry

end gcd_117_182_f_neg_one_l423_42333


namespace monotonic_functional_equation_solution_l423_42341

-- Define a monotonic function f from real numbers to real numbers
def monotonic_function (f : ℝ → ℝ) : Prop :=
  ∀ x y, x ≤ y → f x ≤ f y ∨ ∀ x y, x ≤ y → f x ≥ f y

-- Define the functional equation
def functional_equation (f : ℝ → ℝ) : Prop :=
  ∀ x y, f (x + y) = f x * f y

-- Theorem statement
theorem monotonic_functional_equation_solution :
  ∀ f : ℝ → ℝ, monotonic_function f → functional_equation f →
  ∃ a : ℝ, (a > 1 ∨ 0 < a ∧ a < 1) ∧ ∀ x, f x = a^x :=
sorry

end monotonic_functional_equation_solution_l423_42341


namespace remainder_sum_equality_l423_42388

/-- The remainder sum function -/
def r (n : ℕ) : ℕ := (Finset.range n).sum (λ i => n % (i + 1))

/-- Theorem: The remainder sum of 2^k - 1 equals the remainder sum of 2^k for all k ≥ 1 -/
theorem remainder_sum_equality (k : ℕ) (hk : k ≥ 1) : r (2^k - 1) = r (2^k) := by
  sorry

end remainder_sum_equality_l423_42388


namespace vector_collinearity_l423_42392

theorem vector_collinearity (m : ℝ) : 
  let a : Fin 2 → ℝ := ![2, 3]
  let b : Fin 2 → ℝ := ![-1, 2]
  (∃ (k : ℝ), k ≠ 0 ∧ (m • a + 4 • b) = k • (a - 2 • b)) →
  m = -2 := by
sorry

end vector_collinearity_l423_42392


namespace circle_tangent_axes_l423_42376

/-- Given a point M(x, y) in the first quadrant and a circle passing through M
    that is tangent to both coordinate axes, the product of the radii to the
    points of tangency equals x² + y². -/
theorem circle_tangent_axes (x y r₁ r₂ : ℝ) : 
  x > 0 → y > 0 → 
  ∃ (r : ℝ), (x - r)^2 + (y - r)^2 = r^2 ∧ 
             r₁ + r₂ = 2*r ∧ 
             r₁ * r₂ = r^2 → 
  r₁ * r₂ = x^2 + y^2 :=
sorry

end circle_tangent_axes_l423_42376


namespace mixed_number_difference_l423_42329

def digit_set : Finset ℕ := {1, 2, 3, 4, 5}

def is_valid_mixed_number (whole : ℕ) (numer : ℕ) (denom : ℕ) : Prop :=
  whole ∈ digit_set ∧ numer ∈ digit_set ∧ denom ∈ digit_set ∧
  whole ≠ numer ∧ whole ≠ denom ∧ numer ≠ denom ∧
  numer < denom

def mixed_number_to_rational (whole : ℕ) (numer : ℕ) (denom : ℕ) : ℚ :=
  (whole : ℚ) + (numer : ℚ) / (denom : ℚ)

def largest_mixed_number : ℚ :=
  mixed_number_to_rational 5 3 4

def smallest_mixed_number : ℚ :=
  mixed_number_to_rational 1 2 5

theorem mixed_number_difference :
  largest_mixed_number - smallest_mixed_number = 87 / 20 :=
sorry

end mixed_number_difference_l423_42329


namespace parallel_vectors_magnitude_l423_42397

/-- Given vectors a and b, if they are parallel, then the magnitude of a is 2. -/
theorem parallel_vectors_magnitude (x : ℝ) :
  let a : Fin 2 → ℝ := ![1, x]
  let b : Fin 2 → ℝ := ![x, 3]
  (∃ (k : ℝ), a = k • b) → ‖a‖ = 2 := by
  sorry

end parallel_vectors_magnitude_l423_42397


namespace centroid_vector_sum_centroid_line_ratio_l423_42371

variable {V : Type*} [AddCommGroup V] [Module ℝ V]

variable (A B O G M P Q : V) (a b : V) (m n : ℝ)

/-- G is the centroid of triangle ABO -/
def is_centroid (G A B O : V) : Prop :=
  G = (1/3 : ℝ) • (A + B + O)

/-- M is the midpoint of AB -/
def is_midpoint (M A B : V) : Prop :=
  M = (1/2 : ℝ) • (A + B)

/-- Line PQ passes through G -/
def line_passes_through (P Q G : V) : Prop :=
  ∃ t : ℝ, G = (1 - t) • P + t • Q

theorem centroid_vector_sum 
  (h1 : is_centroid G A B O)
  (h2 : is_midpoint M A B) :
  (G - A) + (G - B) + (G - O) = (0 : V) :=
sorry

theorem centroid_line_ratio
  (h1 : is_centroid G A B O)
  (h2 : O - A = a)
  (h3 : O - B = b)
  (h4 : O - P = m • a)
  (h5 : O - Q = n • b)
  (h6 : line_passes_through P Q G) :
  1/m + 1/n = 3 :=
sorry

end centroid_vector_sum_centroid_line_ratio_l423_42371


namespace second_person_average_pages_per_day_l423_42327

/-- The average number of pages read per day by the second person -/
def average_pages_per_day (summer_days : ℕ) (books_read : ℕ) (pages_per_book : ℕ) (second_person_percentage : ℚ) : ℚ :=
  (books_read * pages_per_book : ℚ) * second_person_percentage / summer_days

/-- Theorem stating that the average number of pages read per day by the second person is 180 -/
theorem second_person_average_pages_per_day :
  average_pages_per_day 80 60 320 (3/4) = 180 := by
  sorry

end second_person_average_pages_per_day_l423_42327


namespace class_exercise_result_l423_42381

theorem class_exercise_result (x : ℝ) : 2 * ((2 * (3 * x + 2) + 3) - 2) = 2 * (2 * (3 * x + 2) + 1) := by
  sorry

end class_exercise_result_l423_42381


namespace parallelogram_smaller_angle_measure_l423_42326

/-- 
Given a parallelogram where one angle exceeds the other by 50 degrees,
prove that the smaller angle measures 65 degrees.
-/
theorem parallelogram_smaller_angle_measure : 
  ∀ (small_angle large_angle : ℝ),
  small_angle > 0 →
  large_angle > 0 →
  large_angle = small_angle + 50 →
  small_angle + large_angle = 180 →
  small_angle = 65 :=
by
  sorry

end parallelogram_smaller_angle_measure_l423_42326


namespace thursday_max_attendance_l423_42339

/-- Represents the days of the week --/
inductive Day
| monday
| tuesday
| wednesday
| thursday
| friday

/-- Represents a team member --/
inductive Member
| dave
| elena
| fiona
| george
| hannah

/-- Returns whether a member can attend on a given day --/
def canAttend (m : Member) (d : Day) : Bool :=
  match m, d with
  | Member.dave, Day.wednesday => false
  | Member.dave, Day.thursday => false
  | Member.elena, Day.monday => false
  | Member.elena, Day.thursday => false
  | Member.elena, Day.friday => false
  | Member.fiona, Day.monday => false
  | Member.fiona, Day.tuesday => false
  | Member.fiona, Day.friday => false
  | Member.george, Day.tuesday => false
  | Member.george, Day.wednesday => false
  | Member.george, Day.friday => false
  | Member.hannah, Day.monday => false
  | Member.hannah, Day.wednesday => false
  | Member.hannah, Day.thursday => false
  | _, _ => true

/-- Counts the number of members who can attend on a given day --/
def countAttendees (d : Day) : Nat :=
  (List.filter (fun m => canAttend m d) [Member.dave, Member.elena, Member.fiona, Member.george, Member.hannah]).length

/-- Theorem: Thursday has the maximum number of attendees --/
theorem thursday_max_attendance :
  ∀ d : Day, countAttendees Day.thursday ≥ countAttendees d :=
by sorry


end thursday_max_attendance_l423_42339


namespace two_co_presidents_probability_l423_42356

def club_sizes : List Nat := [6, 8, 9, 10]
def co_presidents_per_club : Nat := 2

def probability_two_co_presidents (sizes : List Nat) (co_pres : Nat) : ℚ :=
  let probabilities := sizes.map (λ n =>
    (Nat.choose (n - co_pres) (co_pres)) / (Nat.choose n 4))
  (1 / 4 : ℚ) * (probabilities.sum)

theorem two_co_presidents_probability :
  probability_two_co_presidents club_sizes co_presidents_per_club = 2286/10000 := by
  sorry

end two_co_presidents_probability_l423_42356


namespace min_value_of_f_on_interval_l423_42305

def f (x : ℝ) := -x^2 + 4*x - 2

theorem min_value_of_f_on_interval :
  ∃ (m : ℝ), m = -2 ∧ ∀ x ∈ Set.Icc 1 4, f x ≥ m :=
sorry

end min_value_of_f_on_interval_l423_42305


namespace sum_fourth_fifth_terms_l423_42335

def geometric_sequence (a₀ : ℝ) (r : ℝ) (n : ℕ) : ℝ := a₀ * r ^ n

theorem sum_fourth_fifth_terms (a₀ r : ℝ) :
  a₀ > 0 →
  r > 0 →
  r < 1 →
  geometric_sequence a₀ r 0 = 4096 →
  geometric_sequence a₀ r 1 = 1024 →
  geometric_sequence a₀ r 2 = 256 →
  geometric_sequence a₀ r 5 = 4 →
  geometric_sequence a₀ r 6 = 1 →
  geometric_sequence a₀ r 7 = 1/4 →
  geometric_sequence a₀ r 3 + geometric_sequence a₀ r 4 = 80 := by
sorry

end sum_fourth_fifth_terms_l423_42335


namespace profit_distribution_l423_42347

theorem profit_distribution (P Q R : ℝ) (total_profit : ℝ) (R_profit : ℝ) (k : ℝ) :
  4 * P = 6 * Q ∧ 
  4 * P = k * R ∧
  total_profit = 4650 ∧
  R_profit = 900 →
  k = 2.4 := by
sorry

end profit_distribution_l423_42347


namespace min_stamps_for_50_cents_l423_42308

/-- Represents the number of stamps needed to make a certain value -/
def StampCombination := ℕ × ℕ

/-- Calculates the total value of a stamp combination -/
def value (c : StampCombination) : ℕ := 3 * c.1 + 4 * c.2

/-- Calculates the total number of stamps in a combination -/
def total_stamps (c : StampCombination) : ℕ := c.1 + c.2

/-- Checks if a stamp combination is valid (equals 50 cents) -/
def is_valid (c : StampCombination) : Prop := value c = 50

/-- Theorem: The minimum number of stamps needed to make 50 cents is 13 -/
theorem min_stamps_for_50_cents :
  ∃ (c : StampCombination), is_valid c ∧
    total_stamps c = 13 ∧
    ∀ (d : StampCombination), is_valid d → total_stamps c ≤ total_stamps d :=
by
  sorry

#check min_stamps_for_50_cents

end min_stamps_for_50_cents_l423_42308


namespace interview_selection_theorem_l423_42303

structure InterviewSelection where
  total_people : ℕ
  group3_size : ℕ
  group4_size : ℕ
  group5_size : ℕ
  interview_slots : ℕ

def stratified_sampling (s : InterviewSelection) : ℕ × ℕ × ℕ :=
  let total := s.group3_size + s.group4_size + s.group5_size
  ( (s.group3_size * s.interview_slots) / total,
    (s.group4_size * s.interview_slots) / total,
    (s.group5_size * s.interview_slots) / total )

def probability_at_least_one (group_size : ℕ) (selected : ℕ) : ℚ :=
  1 - (Nat.choose (group_size - 2) selected / Nat.choose group_size selected : ℚ)

theorem interview_selection_theorem (s : InterviewSelection) 
  (h1 : s.total_people = 36)
  (h2 : s.group3_size = 18)
  (h3 : s.group4_size = 12)
  (h4 : s.group5_size = 6)
  (h5 : s.interview_slots = 12) :
  let (g3, g4, g5) := stratified_sampling s
  g3 = 6 ∧ g4 = 4 ∧ g5 = 2 ∧ 
  probability_at_least_one s.group5_size g5 = 3/5 :=
sorry

end interview_selection_theorem_l423_42303


namespace quadratic_equation_result_l423_42336

theorem quadratic_equation_result (x : ℝ) (h : x^2 - x - 1 = 0) :
  1995 + 2*x - x^3 = 1994 := by
  sorry

end quadratic_equation_result_l423_42336


namespace power_of_power_three_l423_42353

theorem power_of_power_three : (3^3)^2 = 729 := by
  sorry

end power_of_power_three_l423_42353


namespace system_solution_implies_k_value_l423_42369

theorem system_solution_implies_k_value (k : ℝ) (x y : ℝ) : 
  x + y = 5 * k →
  x - y = 9 * k →
  x - 2 * y = 22 →
  k = 2 := by
sorry

end system_solution_implies_k_value_l423_42369


namespace percentage_of_seats_filled_l423_42398

theorem percentage_of_seats_filled (total_seats vacant_seats : ℕ) 
  (h1 : total_seats = 600) 
  (h2 : vacant_seats = 150) : 
  (((total_seats - vacant_seats : ℚ) / total_seats) * 100 : ℚ) = 75 := by
  sorry

end percentage_of_seats_filled_l423_42398


namespace counterexample_exponential_inequality_l423_42365

theorem counterexample_exponential_inequality :
  ∃ (a m n : ℝ), a > 0 ∧ a ≠ 1 ∧ a^m < a^n ∧ m ≥ n := by
  sorry

end counterexample_exponential_inequality_l423_42365


namespace problem_statement_l423_42307

theorem problem_statement (a b : ℝ) (h1 : a + b = 1) (h2 : a * b = -6) :
  a^3 * b - 2 * a^2 * b^2 + a * b^3 = -150 := by
  sorry

end problem_statement_l423_42307


namespace dragons_games_count_l423_42374

theorem dragons_games_count :
  ∀ (initial_games : ℕ) (initial_wins : ℕ),
    initial_wins = (2 * initial_games / 5) →
    ∀ (final_games : ℕ),
      final_games = initial_games + 8 →
      (initial_wins + 8) = (3 * final_games / 5) →
      final_games = 24 :=
by sorry

end dragons_games_count_l423_42374


namespace not_all_acquainted_l423_42314

/-- Represents a person in the company -/
inductive Person : Type
  | host : Person
  | son1 : Person
  | son2 : Person
  | son3 : Person
  | guest1 : Person
  | guest2 : Person
  | guest3 : Person

/-- Represents the acquaintance relation between two people -/
def acquainted : Person → Person → Prop := sorry

/-- The host is acquainted with all his sons -/
axiom host_knows_sons :
  acquainted Person.host Person.son1 ∧
  acquainted Person.host Person.son2 ∧
  acquainted Person.host Person.son3

/-- Each son is acquainted with exactly one guest -/
axiom sons_know_guests :
  acquainted Person.son1 Person.guest1 ∧
  acquainted Person.son2 Person.guest2 ∧
  acquainted Person.son3 Person.guest3

/-- No guest knows another guest -/
axiom guests_dont_know_each_other :
  ¬acquainted Person.guest1 Person.guest2 ∧
  ¬acquainted Person.guest2 Person.guest3 ∧
  ¬acquainted Person.guest3 Person.guest1

/-- The acquaintance relation is symmetric -/
axiom acquainted_symmetric :
  ∀ (p q : Person), acquainted p q → acquainted q p

/-- Theorem: There exists a pair of people who are not acquainted -/
theorem not_all_acquainted : ∃ (p q : Person), p ≠ q ∧ ¬acquainted p q := by
  sorry

end not_all_acquainted_l423_42314


namespace equation_represents_hyperbola_l423_42313

/-- Represents a conic section --/
inductive ConicSection
  | Parabola
  | Circle
  | Ellipse
  | Hyperbola
  | Point
  | Line
  | TwoLines
  | Empty

/-- Determines the type of conic section from the coefficients of the general equation --/
def determineConicSection (a b c d e f : ℝ) : ConicSection :=
  sorry

/-- The equation x^2 - 36y^2 - 12x + 36 = 0 represents a hyperbola --/
theorem equation_represents_hyperbola :
  determineConicSection 1 (-36) 0 (-12) 0 36 = ConicSection.Hyperbola :=
sorry

end equation_represents_hyperbola_l423_42313


namespace output_for_input_3_l423_42345

-- Define the function for calculating y
def calculate_y (x : ℝ) : ℝ := 3 * x^2 - 5 * x

-- Define the output function
def output (x y : ℝ) : ℝ × ℝ := (x, y)

-- Theorem statement
theorem output_for_input_3 :
  let x : ℝ := 3
  let y : ℝ := calculate_y x
  output x y = (3, 12) := by sorry

end output_for_input_3_l423_42345


namespace cube_sum_equality_l423_42362

theorem cube_sum_equality (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h : 1 / a - 1 / b - 1 / (a + b) = 0) : 
  (b / a)^3 + (a / b)^3 = 2 * Real.sqrt 5 := by
  sorry

end cube_sum_equality_l423_42362


namespace price_change_after_increase_and_discounts_l423_42338

theorem price_change_after_increase_and_discounts :
  let initial_price : ℝ := 100
  let increased_price := initial_price * 1.5
  let price_after_first_discount := increased_price * 0.9
  let price_after_second_discount := price_after_first_discount * 0.85
  let final_price := price_after_second_discount * 0.8
  (final_price - initial_price) / initial_price = -0.082 :=
by sorry

end price_change_after_increase_and_discounts_l423_42338


namespace pure_imaginary_condition_l423_42370

theorem pure_imaginary_condition (a : ℝ) : 
  (∃ b : ℝ, (a - 1) * (a + 1 + Complex.I) = Complex.I * b) → a = -1 := by
  sorry

end pure_imaginary_condition_l423_42370


namespace alley_width_equals_height_l423_42367

/-- Represents a ladder in an alley scenario -/
structure LadderInAlley where
  a : ℝ  -- length of the ladder
  w : ℝ  -- width of the alley
  k : ℝ  -- height at 45° angle
  h : ℝ  -- height at 75° angle

/-- The theorem stating that the width of the alley equals the height h -/
theorem alley_width_equals_height (l : LadderInAlley) : 
  l.w = l.h ∧ 
  Real.cos (45 * π / 180) * l.a = l.k ∧ 
  Real.cos (75 * π / 180) * l.a = l.h :=
by sorry

end alley_width_equals_height_l423_42367


namespace inequality_B_is_linear_one_var_inequality_A_is_not_linear_one_var_inequality_C_is_not_linear_one_var_only_B_is_linear_one_var_l423_42378

-- Define the inequalities
def inequality_A (x : ℝ) := 3 * x^2 > 45 - 9 * x
def inequality_B (x : ℝ) := 3 * x - 2 < 4
def inequality_C (x : ℝ) := 1 / x < 2
def inequality_D (x y : ℝ) := 4 * x - 3 < 2 * y - 7

-- Define what it means for an inequality to be linear with one variable
def is_linear_one_var (f : ℝ → Prop) : Prop :=
  ∃ (a b : ℝ), a ≠ 0 ∧ ∀ x, f x ↔ a * x < b ∨ a * x > b

-- Theorem stating that inequality_B is linear with one variable
theorem inequality_B_is_linear_one_var :
  is_linear_one_var inequality_B :=
sorry

-- Theorems stating that the other inequalities are not linear with one variable
theorem inequality_A_is_not_linear_one_var :
  ¬ is_linear_one_var inequality_A :=
sorry

theorem inequality_C_is_not_linear_one_var :
  ¬ is_linear_one_var inequality_C :=
sorry

-- Note: inequality_D is not included as it has two variables

-- Main theorem
theorem only_B_is_linear_one_var :
  is_linear_one_var inequality_B ∧
  ¬ is_linear_one_var inequality_A ∧
  ¬ is_linear_one_var inequality_C :=
sorry

end inequality_B_is_linear_one_var_inequality_A_is_not_linear_one_var_inequality_C_is_not_linear_one_var_only_B_is_linear_one_var_l423_42378


namespace extended_box_with_hemispheres_volume_l423_42312

/-- The volume of a region formed by extending a rectangular parallelepiped and adding hemispheres at its vertices -/
theorem extended_box_with_hemispheres_volume 
  (l w h : ℝ) 
  (hl : l = 5) 
  (hw : w = 6) 
  (hh : h = 7) 
  (extension : ℝ) 
  (hemisphere_radius : ℝ) 
  (he : extension = 2) 
  (hr : hemisphere_radius = 2) : 
  (l + 2 * extension) * (w + 2 * extension) * (h + 2 * extension) + 
  8 * ((2 / 3) * π * hemisphere_radius^3) = 
  990 + (128 / 3) * π :=
sorry

end extended_box_with_hemispheres_volume_l423_42312


namespace representatives_selection_count_l423_42340

def num_boys : ℕ := 5
def num_girls : ℕ := 4
def total_students : ℕ := num_boys + num_girls
def num_representatives : ℕ := 3

theorem representatives_selection_count :
  (Nat.choose total_students num_representatives) - (Nat.choose num_boys num_representatives) = 74 := by
  sorry

end representatives_selection_count_l423_42340


namespace savings_is_240_l423_42361

/-- Represents the window purchase scenario -/
structure WindowPurchase where
  regularPrice : ℕ
  discountThreshold : ℕ
  freeWindows : ℕ
  georgeNeeds : ℕ
  anneNeeds : ℕ

/-- Calculates the cost for a given number of windows -/
def calculateCost (wp : WindowPurchase) (windows : ℕ) : ℕ :=
  let freeWindowSets := windows / wp.discountThreshold
  let paidWindows := windows - freeWindowSets * wp.freeWindows
  paidWindows * wp.regularPrice

/-- Calculates the savings when purchasing together vs separately -/
def calculateSavings (wp : WindowPurchase) : ℕ :=
  let separateCost := calculateCost wp wp.georgeNeeds + calculateCost wp wp.anneNeeds
  let togetherCost := calculateCost wp (wp.georgeNeeds + wp.anneNeeds)
  separateCost - togetherCost

/-- Theorem stating that the savings is $240 -/
theorem savings_is_240 (wp : WindowPurchase) 
  (h1 : wp.regularPrice = 120)
  (h2 : wp.discountThreshold = 10)
  (h3 : wp.freeWindows = 2)
  (h4 : wp.georgeNeeds = 9)
  (h5 : wp.anneNeeds = 11) :
  calculateSavings wp = 240 := by
  sorry


end savings_is_240_l423_42361


namespace original_paint_intensity_l423_42324

/-- Given a paint mixture where a fraction of 1.5 times the original amount is replaced
    with a 25% solution of red paint, and the resulting mixture has a red paint
    intensity of 30%, prove that the original intensity of the red paint was 15%. -/
theorem original_paint_intensity
  (replacement_fraction : ℝ)
  (replacement_solution_intensity : ℝ)
  (final_mixture_intensity : ℝ)
  (h1 : replacement_fraction = 1.5)
  (h2 : replacement_solution_intensity = 0.25)
  (h3 : final_mixture_intensity = 0.30)
  : ∃ (original_intensity : ℝ),
    original_intensity * (1 - replacement_fraction) +
    replacement_solution_intensity * replacement_fraction = final_mixture_intensity ∧
    original_intensity = 0.15 :=
by sorry

end original_paint_intensity_l423_42324


namespace last_two_digits_of_sum_l423_42304

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

def last_two_digits (n : ℕ) : ℕ := n % 100

def sum_series : ℕ := 
  let terms := List.range 16 |>.map (fun i => 6 * i + 6)
  terms.map (fun n => last_two_digits ((factorial n) + 1)) |>.sum

theorem last_two_digits_of_sum : last_two_digits sum_series = 36 := by
  sorry

end last_two_digits_of_sum_l423_42304


namespace strictly_increasing_function_bounds_l423_42377

theorem strictly_increasing_function_bounds (k : ℕ) (f : ℕ → ℕ) 
  (h_increasing : ∀ m n, m < n → f m < f n)
  (h_property : ∀ n, f (f n) = k * n) :
  ∀ n, (2 * k : ℚ) / (k + 1) * n ≤ f n ∧ (f n : ℚ) ≤ (k + 1) / 2 * n :=
by sorry

end strictly_increasing_function_bounds_l423_42377


namespace work_completion_time_l423_42309

theorem work_completion_time (a_time b_time : ℝ) (work_left : ℝ) (days_worked : ℝ) : 
  a_time = 15 →
  b_time = 20 →
  work_left = 0.5333333333333333 →
  (1 / a_time + 1 / b_time) * days_worked = 1 - work_left →
  days_worked = 4 := by
sorry

end work_completion_time_l423_42309


namespace geometric_sequence_proof_l423_42354

theorem geometric_sequence_proof :
  ∃ (q : ℚ) (n : ℕ),
    let a₁ : ℚ := 6
    let S : ℚ := (a₁ * (1 - q^n)) / (1 - q)
    let R : ℚ := ((1 / a₁) * (1 - (1/q)^n)) / (1 - (1/q))
    S = 45/4 ∧
    R = 5/2 ∧
    n = 4 ∧
    q = 1/2 ∧
    [a₁, a₁ * q, a₁ * q^2, a₁ * q^3] = [6, 3, 3/2, 3/4] :=
by sorry

end geometric_sequence_proof_l423_42354


namespace imaginary_part_of_complex_fraction_l423_42343

theorem imaginary_part_of_complex_fraction (i : ℂ) : 
  i * i = -1 → 
  Complex.im ((1 - i)^2 / (1 + i)) = -1 := by sorry

end imaginary_part_of_complex_fraction_l423_42343


namespace intersection_A_complement_B_l423_42359

-- Define the universal set U as ℝ
def U : Set ℝ := Set.univ

-- Define set A
def A : Set ℝ := {x | -2 ≤ x ∧ x ≤ 2}

-- Define set B
def B : Set ℝ := {x | 0 < x ∧ x < 1}

-- State the theorem
theorem intersection_A_complement_B :
  A ∩ (U \ B) = {x | (-2 ≤ x ∧ x ≤ 0) ∨ (1 ≤ x ∧ x ≤ 2)} := by sorry

end intersection_A_complement_B_l423_42359


namespace simplify_expression_l423_42316

theorem simplify_expression : (2^8 + 5^5) * (2^3 - (-2)^3)^7 = 9077567990336 := by
  sorry

end simplify_expression_l423_42316


namespace gcd_12347_9876_l423_42386

theorem gcd_12347_9876 : Nat.gcd 12347 9876 = 7 := by
  sorry

end gcd_12347_9876_l423_42386


namespace count_even_perfect_square_factors_l423_42396

/-- The number of even perfect square factors of 2^6 * 7^9 * 3^2 -/
def evenPerfectSquareFactors : ℕ :=
  30

/-- The exponent of 2 in the given number -/
def exponent2 : ℕ := 6

/-- The exponent of 7 in the given number -/
def exponent7 : ℕ := 9

/-- The exponent of 3 in the given number -/
def exponent3 : ℕ := 2

/-- Theorem stating that the number of even perfect square factors
    of 2^6 * 7^9 * 3^2 is equal to 30 -/
theorem count_even_perfect_square_factors :
  (∀ a b c : ℕ,
    0 ≤ a ∧ a ≤ exponent2 ∧
    0 ≤ b ∧ b ≤ exponent7 ∧
    0 ≤ c ∧ c ≤ exponent3 ∧
    Even a ∧ Even b ∧ Even c ∧
    a ≥ 1) →
  evenPerfectSquareFactors = 30 := by
  sorry

end count_even_perfect_square_factors_l423_42396


namespace no_real_solutions_l423_42322

theorem no_real_solutions :
  ∀ x : ℝ, ((x - 4*x + 15)^2 + 3)^2 + 1 ≠ -(abs x)^2 := by
  sorry

end no_real_solutions_l423_42322


namespace point_on_line_iff_concyclic_l423_42393

-- Define the points
variable (A B C D E F M : Point)

-- Define the concyclic property
def are_concyclic (P Q R S : Point) : Prop := sorry

-- Define the collinear property
def are_collinear (P Q R : Point) : Prop := sorry

-- State the theorem
theorem point_on_line_iff_concyclic :
  (are_concyclic D M F B) →
  (are_concyclic M A E F) →
  (are_collinear A M D) ↔ (are_concyclic B F E C) := by
  sorry

end point_on_line_iff_concyclic_l423_42393


namespace pet_store_cages_l423_42357

theorem pet_store_cages (initial_puppies : ℕ) (sold_puppies : ℕ) (puppies_per_cage : ℕ) : 
  initial_puppies = 56 →
  sold_puppies = 24 →
  puppies_per_cage = 4 →
  (initial_puppies - sold_puppies) / puppies_per_cage = 8 :=
by
  sorry

end pet_store_cages_l423_42357


namespace fathers_age_l423_42399

theorem fathers_age (sebastian_age : ℕ) (sister_age : ℕ) (father_age : ℕ) :
  sebastian_age = 40 →
  sebastian_age = sister_age + 10 →
  (sebastian_age - 5 + sister_age - 5 : ℚ) = 3/4 * (father_age - 5) →
  father_age = 85 := by
  sorry

end fathers_age_l423_42399


namespace valid_arrangements_count_l423_42348

/-- Represents the number of advertisements --/
def total_ads : ℕ := 5

/-- Represents the number of commercial advertisements --/
def commercial_ads : ℕ := 3

/-- Represents the number of public service advertisements --/
def public_service_ads : ℕ := 2

/-- Calculates the number of ways to arrange the advertisements --/
def arrangement_count : ℕ := 36

/-- Theorem stating that the number of valid arrangements is 36 --/
theorem valid_arrangements_count :
  (total_ads = commercial_ads + public_service_ads) →
  (public_service_ads > 0) →
  (arrangement_count = 36) := by
  sorry

#check valid_arrangements_count

end valid_arrangements_count_l423_42348


namespace range_of_a_l423_42366

theorem range_of_a (a : ℝ) : 
  (∀ x > 0, a - x - |Real.log x| ≤ 0) ↔ 0 < a ∧ a ≤ 1 := by
sorry

end range_of_a_l423_42366


namespace defective_items_probability_l423_42352

theorem defective_items_probability 
  (p_zero : ℝ) 
  (p_one : ℝ) 
  (p_two : ℝ) 
  (p_three : ℝ) 
  (h1 : p_zero = 0.18) 
  (h2 : p_one = 0.53) 
  (h3 : p_two = 0.27) 
  (h4 : p_three = 0.02) : 
  (p_two + p_three = 0.29) ∧ (p_zero + p_one = 0.71) := by
  sorry

end defective_items_probability_l423_42352


namespace power_mod_seventeen_l423_42300

theorem power_mod_seventeen : 3^2000 % 17 = 1 := by
  sorry

end power_mod_seventeen_l423_42300


namespace range_of_a_l423_42318

theorem range_of_a (f : ℝ → ℝ) (a : ℝ) :
  (∀ x ∈ Set.Icc (-π/3) a, f x = Real.sin (x + π/6)) →
  Set.range f = Set.Icc (-1/2) 1 →
  a ∈ Set.Icc (π/3) π :=
by sorry

end range_of_a_l423_42318


namespace polynomial_factorization_l423_42328

theorem polynomial_factorization (x : ℝ) : 
  x^4 + 2*x^3 - 9*x^2 - 2*x + 8 = (x + 4)*(x - 2)*(x + 1)*(x - 1) :=
by sorry

end polynomial_factorization_l423_42328
