import Mathlib

namespace NUMINAMATH_CALUDE_intersected_cells_count_l3467_346707

/-- Represents a grid cell -/
structure Cell where
  x : Int
  y : Int

/-- Represents a grid -/
structure Grid where
  width : Nat
  height : Nat

/-- Checks if a point (x, y) is inside the grid -/
def Grid.contains (g : Grid) (x y : Int) : Prop :=
  -g.width / 2 ≤ x ∧ x < g.width / 2 ∧ -g.height / 2 ≤ y ∧ y < g.height / 2

/-- Counts the number of cells intersected by the line y = mx -/
def countIntersectedCells (g : Grid) (m : ℚ) : ℕ :=
  sorry

/-- Theorem stating that the number of cells intersected by y = 0.83x on a 60x70 grid is 108 -/
theorem intersected_cells_count :
  let g : Grid := { width := 60, height := 70 }
  let m : ℚ := 83 / 100
  countIntersectedCells g m = 108 := by
  sorry

end NUMINAMATH_CALUDE_intersected_cells_count_l3467_346707


namespace NUMINAMATH_CALUDE_mean_of_five_numbers_with_sum_one_third_l3467_346777

theorem mean_of_five_numbers_with_sum_one_third :
  ∀ (a b c d e : ℚ), 
    a + b + c + d + e = 1/3 →
    (a + b + c + d + e) / 5 = 1/15 := by
sorry

end NUMINAMATH_CALUDE_mean_of_five_numbers_with_sum_one_third_l3467_346777


namespace NUMINAMATH_CALUDE_fish_in_pond_l3467_346749

theorem fish_in_pond (initial_tagged : ℕ) (second_catch : ℕ) (tagged_in_second : ℕ) 
  (h1 : initial_tagged = 30)
  (h2 : second_catch = 50)
  (h3 : tagged_in_second = 2)
  (h4 : (tagged_in_second : ℚ) / second_catch = initial_tagged / total_fish) :
  total_fish = 750 :=
by sorry

#check fish_in_pond

end NUMINAMATH_CALUDE_fish_in_pond_l3467_346749


namespace NUMINAMATH_CALUDE_complement_of_A_in_U_l3467_346780

def U : Set ℕ := {x | 1 < x ∧ x < 5}
def A : Set ℕ := {2, 3}

theorem complement_of_A_in_U : (U \ A) = {4} := by sorry

end NUMINAMATH_CALUDE_complement_of_A_in_U_l3467_346780


namespace NUMINAMATH_CALUDE_alice_bracelet_profit_l3467_346730

def bracelet_profit (initial_bracelets : ℕ) (material_cost : ℚ) 
                    (given_away : ℕ) (selling_price : ℚ) : ℚ :=
  let remaining_bracelets := initial_bracelets - given_away
  let revenue := (remaining_bracelets : ℚ) * selling_price
  revenue - material_cost

theorem alice_bracelet_profit :
  bracelet_profit 52 3 8 (1/4) = 8 := by
  sorry

end NUMINAMATH_CALUDE_alice_bracelet_profit_l3467_346730


namespace NUMINAMATH_CALUDE_unique_solution_quadratic_l3467_346716

theorem unique_solution_quadratic (m : ℝ) : 
  (∃! x : ℝ, (x + 5) * (x + 2) = m + 3 * x) ↔ m = 6 :=
by sorry

end NUMINAMATH_CALUDE_unique_solution_quadratic_l3467_346716


namespace NUMINAMATH_CALUDE_zhaos_estimate_l3467_346708

theorem zhaos_estimate (x y ε : ℝ) (h1 : x > y) (h2 : y > 0) (h3 : ε > 0) :
  (x + ε) - (y - 2*ε) > x - y :=
sorry

end NUMINAMATH_CALUDE_zhaos_estimate_l3467_346708


namespace NUMINAMATH_CALUDE_complex_magnitude_equation_l3467_346771

theorem complex_magnitude_equation (t : ℝ) : 
  0 < t → t < 4 → Complex.abs (t + 3 * Complex.I * Real.sqrt 2) * Complex.abs (7 - 5 * Complex.I) = 35 * Real.sqrt 2 → 
  t = Real.sqrt (559 / 37) := by
sorry

end NUMINAMATH_CALUDE_complex_magnitude_equation_l3467_346771


namespace NUMINAMATH_CALUDE_torch_relay_probability_l3467_346774

/-- The number of torchbearers --/
def n : ℕ := 18

/-- The common difference of the arithmetic sequence --/
def d : ℕ := 3

/-- The probability of selecting three numbers from 1 to n that form an arithmetic 
    sequence with common difference d --/
def probability (n d : ℕ) : ℚ :=
  (3 * (n - 2 * d)) / (n * (n - 1) * (n - 2))

/-- The main theorem: the probability for the given problem is 1/68 --/
theorem torch_relay_probability : probability n d = 1 / 68 := by
  sorry


end NUMINAMATH_CALUDE_torch_relay_probability_l3467_346774


namespace NUMINAMATH_CALUDE_leftover_tarts_l3467_346751

/-- The number of leftover tarts in a restaurant, given the fractions of different flavored tarts. -/
theorem leftover_tarts (cherry : ℝ) (blueberry : ℝ) (peach : ℝ) 
  (h_cherry : cherry = 0.08)
  (h_blueberry : blueberry = 0.75)
  (h_peach : peach = 0.08) :
  cherry + blueberry + peach = 0.91 := by
  sorry

end NUMINAMATH_CALUDE_leftover_tarts_l3467_346751


namespace NUMINAMATH_CALUDE_diagonal_path_shorter_than_sides_l3467_346790

theorem diagonal_path_shorter_than_sides (ε : ℝ) (h : ε > 0) : ∃ δ : ℝ, 
  0 < δ ∧ δ < ε ∧ 
  |(2 - Real.sqrt 2) / 2 - 0.293| < δ :=
sorry

end NUMINAMATH_CALUDE_diagonal_path_shorter_than_sides_l3467_346790


namespace NUMINAMATH_CALUDE_set_A_is_empty_l3467_346735

def set_A : Set ℝ := {x : ℝ | x^2 + 2 = 0}
def set_B : Set ℝ := {0}
def set_C : Set ℝ := {x : ℝ | x > 8 ∨ x < 4}
def set_D : Set (Set ℝ) := {∅}

theorem set_A_is_empty : set_A = ∅ := by
  sorry

end NUMINAMATH_CALUDE_set_A_is_empty_l3467_346735


namespace NUMINAMATH_CALUDE_sphere_radii_difference_l3467_346728

theorem sphere_radii_difference (R r : ℝ) : 
  R > r → 
  4 * Real.pi * R^2 - 4 * Real.pi * r^2 = 48 * Real.pi → 
  2 * Real.pi * R + 2 * Real.pi * r = 12 * Real.pi → 
  R - r = 2 :=
by sorry

end NUMINAMATH_CALUDE_sphere_radii_difference_l3467_346728


namespace NUMINAMATH_CALUDE_overlapping_squares_area_l3467_346799

/-- Given two overlapping squares with side length 12, where the overlap forms an equilateral triangle,
    prove that the area of the overlapping region is 108√3, and m + n = 111 -/
theorem overlapping_squares_area (side_length : ℝ) (m n : ℕ) :
  side_length = 12 →
  (m : ℝ) * Real.sqrt n = 108 * Real.sqrt 3 →
  n.Prime →
  m + n = 111 :=
by sorry

end NUMINAMATH_CALUDE_overlapping_squares_area_l3467_346799


namespace NUMINAMATH_CALUDE_d_sufficient_not_necessary_for_a_l3467_346789

-- Define the propositions
variable (A B C D : Prop)

-- Define the relationships between the propositions
variable (h1 : A → B ∧ ¬(B → A))
variable (h2 : (B → C) ∧ (C → B))
variable (h3 : C → D ∧ ¬(D → C))

-- Theorem statement
theorem d_sufficient_not_necessary_for_a :
  D → A ∧ ¬(A → D) :=
sorry

end NUMINAMATH_CALUDE_d_sufficient_not_necessary_for_a_l3467_346789


namespace NUMINAMATH_CALUDE_lawyer_upfront_payment_l3467_346741

theorem lawyer_upfront_payment
  (hourly_rate : ℕ)
  (court_time : ℕ)
  (prep_time_multiplier : ℕ)
  (total_payment : ℕ)
  (h1 : hourly_rate = 100)
  (h2 : court_time = 50)
  (h3 : prep_time_multiplier = 2)
  (h4 : total_payment = 8000) :
  let prep_time := prep_time_multiplier * court_time
  let total_hours := court_time + prep_time
  let total_fee := hourly_rate * total_hours
  let johns_share := total_payment / 2
  let upfront_payment := johns_share
  upfront_payment = 4000 := by
sorry

end NUMINAMATH_CALUDE_lawyer_upfront_payment_l3467_346741


namespace NUMINAMATH_CALUDE_point_on_y_axis_l3467_346770

/-- 
Given a point P with coordinates (1-x, 2x+1) that lies on the y-axis,
prove that its coordinates are (0, 3).
-/
theorem point_on_y_axis (x : ℝ) :
  (1 - x = 0) ∧ (∃ y, y = 2*x + 1) → (1 - x = 0 ∧ 2*x + 1 = 3) :=
by sorry

end NUMINAMATH_CALUDE_point_on_y_axis_l3467_346770


namespace NUMINAMATH_CALUDE_ball_passing_game_l3467_346733

/-- Probability of the ball returning to player A after n passes in a three-player game --/
def P (n : ℕ) : ℚ :=
  1/3 - 1/3 * (-1/2)^(n-1)

theorem ball_passing_game :
  (P 2 = 1/2) ∧
  (∀ n : ℕ, P (n+1) = 1/2 * (1 - P n)) ∧
  (∀ n : ℕ, P n = 1/3 - 1/3 * (-1/2)^(n-1)) :=
by sorry

end NUMINAMATH_CALUDE_ball_passing_game_l3467_346733


namespace NUMINAMATH_CALUDE_gate_change_probability_l3467_346759

/-- The number of gates at the airport -/
def num_gates : ℕ := 15

/-- The distance between adjacent gates in feet -/
def gate_distance : ℕ := 100

/-- The maximum walking distance we're interested in -/
def max_walk_distance : ℕ := 300

/-- The total number of possible gate change scenarios -/
def total_scenarios : ℕ := num_gates * (num_gates - 1)

/-- The number of gates within the maximum walking distance on each side -/
def gates_within_distance : ℕ := max_walk_distance / gate_distance

/-- The number of valid scenarios for gates at the extremities -/
def extreme_gate_scenarios : ℕ := 4 * gates_within_distance

/-- The number of valid scenarios for gates next to extremities -/
def next_to_extreme_scenarios : ℕ := 2 * (gates_within_distance + 1)

/-- The number of valid scenarios for middle gates -/
def middle_gate_scenarios : ℕ := (num_gates - 4) * (2 * gates_within_distance + 1)

/-- The total number of valid scenarios -/
def valid_scenarios : ℕ := extreme_gate_scenarios + next_to_extreme_scenarios + middle_gate_scenarios

/-- The probability of walking 300 feet or less -/
theorem gate_change_probability :
  (valid_scenarios : ℚ) / total_scenarios = 37 / 105 := by
  sorry

end NUMINAMATH_CALUDE_gate_change_probability_l3467_346759


namespace NUMINAMATH_CALUDE_quadratic_inequality_l3467_346781

theorem quadratic_inequality (x : ℝ) : 3 * x^2 - 4 * x - 4 < 0 ↔ -2/3 < x ∧ x < 2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_l3467_346781


namespace NUMINAMATH_CALUDE_recurrence_sequence_b8_l3467_346701

/-- An increasing sequence of positive integers satisfying the given recurrence relation -/
def RecurrenceSequence (b : ℕ → ℕ) : Prop :=
  (∀ n, b n < b (n + 1)) ∧ 
  (∀ n, 1 ≤ n → b (n + 2) = b (n + 1) + b n)

/-- The theorem statement -/
theorem recurrence_sequence_b8 (b : ℕ → ℕ) 
  (h : RecurrenceSequence b) (h7 : b 7 = 198) : b 8 = 321 := by
  sorry

end NUMINAMATH_CALUDE_recurrence_sequence_b8_l3467_346701


namespace NUMINAMATH_CALUDE_total_candies_l3467_346714

/-- The number of boxes Linda has -/
def x : ℕ := 3

/-- The number of candy bags Chloe has -/
def y : ℕ := 2

/-- The number of candy bars Olivia has -/
def z : ℕ := 5

/-- The number of candies in each of Linda's boxes -/
def candies_per_box : ℕ := 2

/-- The number of candies in each of Chloe's bags -/
def candies_per_bag : ℕ := 4

/-- The number of candies equivalent to each of Olivia's candy bars -/
def candies_per_bar : ℕ := 3

/-- The number of candies Linda has -/
def linda_candies : ℕ := 2 * x + 6

/-- The number of candies Chloe has -/
def chloe_candies : ℕ := 4 * y + 7

/-- The number of candies Olivia has -/
def olivia_candies : ℕ := 3 * z - 5

theorem total_candies : linda_candies + chloe_candies + olivia_candies = 37 := by
  sorry

end NUMINAMATH_CALUDE_total_candies_l3467_346714


namespace NUMINAMATH_CALUDE_divisor_calculation_l3467_346798

theorem divisor_calculation (dividend quotient remainder divisor : ℕ) : 
  dividend = 15968 ∧ quotient = 89 ∧ remainder = 37 ∧ 
  dividend = divisor * quotient + remainder → 
  divisor = 179 := by
  sorry

end NUMINAMATH_CALUDE_divisor_calculation_l3467_346798


namespace NUMINAMATH_CALUDE_playground_girls_l3467_346767

theorem playground_girls (total_children : ℕ) (boys : ℕ) (girls : ℕ) : 
  total_children = 117 → boys = 40 → girls = total_children - boys → girls = 77 := by
  sorry

end NUMINAMATH_CALUDE_playground_girls_l3467_346767


namespace NUMINAMATH_CALUDE_union_of_sets_l3467_346768

def A (a : ℝ) : Set ℝ := {0, a}
def B (a : ℝ) : Set ℝ := {3^a, 1}

theorem union_of_sets (a : ℝ) (h : A a ∩ B a = {1}) : A a ∪ B a = {0, 1, 3} := by
  sorry

end NUMINAMATH_CALUDE_union_of_sets_l3467_346768


namespace NUMINAMATH_CALUDE_insurance_premium_theorem_l3467_346769

/-- Represents an insurance policy -/
structure InsurancePolicy where
  payout : ℝ  -- The amount paid out if the event occurs
  probability : ℝ  -- The probability of the event occurring
  premium : ℝ  -- The premium charged to the customer

/-- Calculates the expected revenue for an insurance policy -/
def expectedRevenue (policy : InsurancePolicy) : ℝ :=
  policy.premium - policy.payout * policy.probability

/-- Theorem: Given an insurance policy with payout 'a' and event probability 'p',
    if the company wants an expected revenue of 10% of 'a',
    then the required premium is a(p + 0.1) -/
theorem insurance_premium_theorem (a p : ℝ) (h1 : 0 ≤ p) (h2 : p ≤ 1) :
  let policy := InsurancePolicy.mk a p (a * (p + 0.1))
  expectedRevenue policy = 0.1 * a := by
  sorry

#check insurance_premium_theorem

end NUMINAMATH_CALUDE_insurance_premium_theorem_l3467_346769


namespace NUMINAMATH_CALUDE_uncolored_area_rectangle_with_circles_l3467_346784

/-- The uncolored area of a rectangle with four tangent circles --/
theorem uncolored_area_rectangle_with_circles (w h r : Real) 
  (hw : w = 30) 
  (hh : h = 50) 
  (hr : r = w / 4) 
  (circles_fit : 4 * r = w) 
  (circles_tangent : 2 * r = h / 2) : 
  w * h - 4 * Real.pi * r^2 = 1500 - 225 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_uncolored_area_rectangle_with_circles_l3467_346784


namespace NUMINAMATH_CALUDE_kolya_can_prevent_divisibility_by_nine_l3467_346779

def digits : Set Nat := {1, 2, 3, 4, 5}

def alternating_sum (n : Nat) (f : Nat → Nat) : Nat :=
  List.sum (List.range n |>.map f)

theorem kolya_can_prevent_divisibility_by_nine :
  ∃ (kolya : Nat → Nat), ∀ (vasya : Nat → Nat),
    (∀ i, kolya i ∈ digits ∧ vasya i ∈ digits) →
    ¬(alternating_sum 10 kolya + alternating_sum 10 vasya) % 9 = 0 :=
sorry

end NUMINAMATH_CALUDE_kolya_can_prevent_divisibility_by_nine_l3467_346779


namespace NUMINAMATH_CALUDE_solution_set_part1_solution_set_part2_l3467_346762

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |x - a| + 3 * x

-- Part 1
theorem solution_set_part1 :
  {x : ℝ | f 1 x ≥ 3 * x + 2} = {x : ℝ | x ≥ 3 ∨ x ≤ -1} := by sorry

-- Part 2
theorem solution_set_part2 (a : ℝ) (h : a > 0) :
  ({x : ℝ | f a x ≤ 0} = {x : ℝ | x ≤ -1}) → a = 2 := by sorry

end NUMINAMATH_CALUDE_solution_set_part1_solution_set_part2_l3467_346762


namespace NUMINAMATH_CALUDE_weekly_training_cost_l3467_346726

/-- Proves that the weekly training cost is $250, given the adoption fee, training duration, certification cost, insurance coverage, and total out-of-pocket cost. -/
theorem weekly_training_cost
  (adoption_fee : ℝ)
  (training_weeks : ℕ)
  (certification_cost : ℝ)
  (insurance_coverage : ℝ)
  (total_out_of_pocket : ℝ)
  (h1 : adoption_fee = 150)
  (h2 : training_weeks = 12)
  (h3 : certification_cost = 3000)
  (h4 : insurance_coverage = 0.9)
  (h5 : total_out_of_pocket = 3450)
  : ∃ (weekly_cost : ℝ),
    weekly_cost = 250 ∧
    total_out_of_pocket = adoption_fee + training_weeks * weekly_cost + (1 - insurance_coverage) * certification_cost :=
by sorry

end NUMINAMATH_CALUDE_weekly_training_cost_l3467_346726


namespace NUMINAMATH_CALUDE_count_five_digit_integers_l3467_346702

/-- The set of digits to be used -/
def digits : Multiset ℕ := {3, 3, 6, 6, 6, 7, 8, 8}

/-- The number of digits required for each integer -/
def required_digits : ℕ := 5

/-- The function to count valid integers -/
def count_valid_integers (d : Multiset ℕ) (r : ℕ) : ℕ :=
  (d.card.factorial) / ((d.count 3).factorial * (d.count 6).factorial * (d.count 8).factorial)

/-- The main theorem -/
theorem count_five_digit_integers : 
  count_valid_integers digits required_digits = 1680 :=
sorry

end NUMINAMATH_CALUDE_count_five_digit_integers_l3467_346702


namespace NUMINAMATH_CALUDE_plants_eaten_third_day_l3467_346752

theorem plants_eaten_third_day 
  (initial_plants : ℕ)
  (eaten_first_day : ℕ)
  (fraction_eaten_second_day : ℚ)
  (final_plants : ℕ)
  (h1 : initial_plants = 30)
  (h2 : eaten_first_day = 20)
  (h3 : fraction_eaten_second_day = 1/2)
  (h4 : final_plants = 4)
  : initial_plants - eaten_first_day - 
    (initial_plants - eaten_first_day) * fraction_eaten_second_day - 
    final_plants = 1 := by
  sorry

end NUMINAMATH_CALUDE_plants_eaten_third_day_l3467_346752


namespace NUMINAMATH_CALUDE_greatest_digit_sum_base_seven_l3467_346715

/-- Represents a positive integer in base 7 --/
def BaseSevenRepresentation := List Nat

/-- Converts a natural number to its base-seven representation --/
def toBaseSeven (n : Nat) : BaseSevenRepresentation :=
  sorry

/-- Calculates the sum of digits in a base-seven representation --/
def sumDigits (repr : BaseSevenRepresentation) : Nat :=
  sorry

/-- The upper bound for the problem --/
def upperBound : Nat := 2401

theorem greatest_digit_sum_base_seven :
  ∃ (max : Nat), ∀ (n : Nat), n < upperBound →
    sumDigits (toBaseSeven n) ≤ max ∧
    ∃ (m : Nat), m < upperBound ∧ sumDigits (toBaseSeven m) = max ∧
    max = 12 :=
  sorry

end NUMINAMATH_CALUDE_greatest_digit_sum_base_seven_l3467_346715


namespace NUMINAMATH_CALUDE_inequality_proof_l3467_346718

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (Real.sqrt (a * b) + Real.sqrt (b * c) + Real.sqrt (c * a) ≤ a + b + c) ∧
  (a + b + c = 1 → (2 * a * b) / (a + b) + (2 * b * c) / (b + c) + (2 * a * c) / (a + c) ≤ 1) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3467_346718


namespace NUMINAMATH_CALUDE_inequality_exists_n_l3467_346732

theorem inequality_exists_n : ∃ n : ℕ+, ∀ x : ℝ, x ≥ 0 → (x - 1) * (x^2005 - 2005*x^(n.val + 1) + 2005*x^n.val - 1) ≥ 0 := by
  sorry

end NUMINAMATH_CALUDE_inequality_exists_n_l3467_346732


namespace NUMINAMATH_CALUDE_extreme_value_implies_params_l3467_346794

/-- A cubic function with parameters a and b -/
def f (a b x : ℝ) : ℝ := a * x^3 + b * x

/-- The derivative of f with respect to x -/
def f' (a b x : ℝ) : ℝ := 3 * a * x^2 + b

theorem extreme_value_implies_params (a b : ℝ) :
  (f a b 1 = -2) ∧ (f' a b 1 = 0) → a = 1 ∧ b = -3 := by
  sorry

end NUMINAMATH_CALUDE_extreme_value_implies_params_l3467_346794


namespace NUMINAMATH_CALUDE_difference_of_squares_example_l3467_346786

theorem difference_of_squares_example : (15 + 12)^2 - (15 - 12)^2 = 720 := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_example_l3467_346786


namespace NUMINAMATH_CALUDE_line_parallel_to_plane_relationship_l3467_346782

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relationships
variable (parallelToPlane : Line → Plane → Prop)
variable (withinPlane : Line → Plane → Prop)
variable (parallel : Line → Line → Prop)
variable (skew : Line → Line → Prop)

-- State the theorem
theorem line_parallel_to_plane_relationship 
  (a b : Line) (α : Plane)
  (h1 : parallelToPlane a α)
  (h2 : withinPlane b α) :
  parallel a b ∨ skew a b :=
sorry

end NUMINAMATH_CALUDE_line_parallel_to_plane_relationship_l3467_346782


namespace NUMINAMATH_CALUDE_cake_distribution_l3467_346712

theorem cake_distribution (total_pieces : ℕ) (eaten_percentage : ℚ) (num_sisters : ℕ) : 
  total_pieces = 240 →
  eaten_percentage = 60 / 100 →
  num_sisters = 3 →
  (total_pieces - (eaten_percentage * total_pieces).floor) / num_sisters = 32 := by
sorry

end NUMINAMATH_CALUDE_cake_distribution_l3467_346712


namespace NUMINAMATH_CALUDE_polyhedron_edge_length_bound_l3467_346753

/-- A polyhedron is represented as a set of points in ℝ³ -/
def Polyhedron : Type := Set (Fin 3 → ℝ)

/-- The edge length of a polyhedron -/
def edgeLength (P : Polyhedron) : ℝ := sorry

/-- The sum of all edge lengths of a polyhedron -/
def sumEdgeLengths (P : Polyhedron) : ℝ := sorry

/-- The distance between two points in ℝ³ -/
def distance (a b : Fin 3 → ℝ) : ℝ := sorry

/-- The maximum distance between any two points in a polyhedron -/
def maxDistance (P : Polyhedron) : ℝ := sorry

/-- Theorem: The sum of edge lengths is at least 3 times the maximum distance -/
theorem polyhedron_edge_length_bound (P : Polyhedron) :
  sumEdgeLengths P ≥ 3 * maxDistance P := by sorry

end NUMINAMATH_CALUDE_polyhedron_edge_length_bound_l3467_346753


namespace NUMINAMATH_CALUDE_ad_length_l3467_346758

/-- A simple quadrilateral with specific side lengths and angle properties -/
structure SimpleQuadrilateral where
  -- Sides
  AB : ℝ
  BC : ℝ
  CD : ℝ
  AD : ℝ
  -- Angles (in radians)
  B : ℝ
  C : ℝ
  -- Properties
  simple : Prop
  AB_length : AB = 4
  BC_length : BC = 5
  CD_length : CD = 20
  B_obtuse : π / 2 < B ∧ B < π
  C_obtuse : π / 2 < C ∧ C < π
  angle_relation : Real.sin C = -Real.cos B ∧ Real.sin C = 3/5

/-- The theorem stating the length of AD in the specific quadrilateral -/
theorem ad_length (q : SimpleQuadrilateral) : q.AD = Real.sqrt 674 := by
  sorry

end NUMINAMATH_CALUDE_ad_length_l3467_346758


namespace NUMINAMATH_CALUDE_arithmetic_sequence_10th_term_l3467_346709

/-- An arithmetic sequence is defined by its first term and common difference -/
structure ArithmeticSequence :=
  (a : ℤ)  -- First term
  (d : ℤ)  -- Common difference

/-- Get the nth term of an arithmetic sequence -/
def ArithmeticSequence.nthTerm (seq : ArithmeticSequence) (n : ℕ) : ℤ :=
  seq.a + (n - 1) * seq.d

theorem arithmetic_sequence_10th_term
  (seq : ArithmeticSequence)
  (h4 : seq.nthTerm 4 = 23)
  (h8 : seq.nthTerm 8 = 55) :
  seq.nthTerm 10 = 71 := by
  sorry

#check arithmetic_sequence_10th_term

end NUMINAMATH_CALUDE_arithmetic_sequence_10th_term_l3467_346709


namespace NUMINAMATH_CALUDE_evaluate_expression_l3467_346764

theorem evaluate_expression (x z : ℝ) (hx : x = 4) (hz : z = 2) : 
  z * (z - 4 * x) = -28 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l3467_346764


namespace NUMINAMATH_CALUDE_remainder_theorem_l3467_346706

def polynomial (x : ℝ) : ℝ := 5*x^5 - 12*x^4 + 3*x^3 - 7*x + 15

def divisor (x : ℝ) : ℝ := 3*x - 6

theorem remainder_theorem :
  ∃ (q : ℝ → ℝ), ∀ x, polynomial x = (divisor x) * q x + (-7) :=
by
  sorry

end NUMINAMATH_CALUDE_remainder_theorem_l3467_346706


namespace NUMINAMATH_CALUDE_common_tangent_sum_l3467_346773

-- Define the parabolas
def P₁ (x y : ℝ) : Prop := y = x^2 + 51/50
def P₂ (x y : ℝ) : Prop := x = y^2 + 95/8

-- Define the common tangent line
def CommonTangent (a b c : ℕ) (x y : ℝ) : Prop :=
  (a : ℝ) * x + (b : ℝ) * y = c

-- Main theorem
theorem common_tangent_sum :
  ∃ (a b c : ℕ),
    (a > 0 ∧ b > 0 ∧ c > 0) ∧
    (Nat.gcd a (Nat.gcd b c) = 1) ∧
    (∃ (m : ℚ), ∀ (x y : ℝ),
      CommonTangent a b c x y → y = m * x + (c / b : ℝ)) ∧
    (∀ (x y : ℝ),
      (P₁ x y → ∃ (x₀ y₀ : ℝ), P₁ x₀ y₀ ∧ CommonTangent a b c x₀ y₀) ∧
      (P₂ x y → ∃ (x₀ y₀ : ℝ), P₂ x₀ y₀ ∧ CommonTangent a b c x₀ y₀)) ∧
    a + b + c = 59 := by
  sorry


end NUMINAMATH_CALUDE_common_tangent_sum_l3467_346773


namespace NUMINAMATH_CALUDE_trig_computation_l3467_346722

theorem trig_computation : 
  1 / Real.cos (70 * π / 180) - Real.sqrt 3 / Real.sin (70 * π / 180) = 4 := by
  sorry

end NUMINAMATH_CALUDE_trig_computation_l3467_346722


namespace NUMINAMATH_CALUDE_last_digit_of_one_over_three_to_ten_l3467_346787

theorem last_digit_of_one_over_three_to_ten (n : ℕ) :
  n = 10 →
  ∃ (k : ℕ), (1 : ℚ) / 3^n = k / 10^10 ∧ k % 10 = 3 :=
by sorry

end NUMINAMATH_CALUDE_last_digit_of_one_over_three_to_ten_l3467_346787


namespace NUMINAMATH_CALUDE_huahuan_initial_cards_l3467_346717

/-- Represents the card distribution among the three players -/
structure CardDistribution where
  huahuan : ℕ
  yingying : ℕ
  nini : ℕ

/-- Represents one round of operations -/
def performRound (dist : CardDistribution) : CardDistribution :=
  sorry

/-- Check if the distribution forms an arithmetic sequence -/
def isArithmeticSequence (dist : CardDistribution) : Prop :=
  dist.yingying - dist.huahuan = dist.nini - dist.yingying

/-- The main theorem -/
theorem huahuan_initial_cards 
  (initial : CardDistribution)
  (h1 : initial.huahuan + initial.yingying + initial.nini = 2712)
  (h2 : ∃ (final : CardDistribution), 
    (performRound^[50] initial = final) ∧ 
    (isArithmeticSequence final)) :
  initial.huahuan = 754 := by
  sorry


end NUMINAMATH_CALUDE_huahuan_initial_cards_l3467_346717


namespace NUMINAMATH_CALUDE_stream_speed_l3467_346703

/-- Given a canoe that rows upstream at 8 km/hr and downstream at 12 km/hr,
    prove that the speed of the stream is 2 km/hr. -/
theorem stream_speed (upstream_speed downstream_speed : ℝ)
  (h_upstream : upstream_speed = 8)
  (h_downstream : downstream_speed = 12) :
  ∃ (canoe_speed stream_speed : ℝ),
    canoe_speed - stream_speed = upstream_speed ∧
    canoe_speed + stream_speed = downstream_speed ∧
    stream_speed = 2 := by
  sorry

end NUMINAMATH_CALUDE_stream_speed_l3467_346703


namespace NUMINAMATH_CALUDE_sin_alpha_value_l3467_346737

theorem sin_alpha_value (α : Real) :
  let point : Real × Real := (2 * Real.sin (60 * π / 180), -2 * Real.cos (60 * π / 180))
  (∃ k : Real, k > 0 ∧ k * point.1 = Real.cos α ∧ k * point.2 = Real.sin α) →
  Real.sin α = -1/2 := by
sorry

end NUMINAMATH_CALUDE_sin_alpha_value_l3467_346737


namespace NUMINAMATH_CALUDE_negation_of_existence_negation_of_inequality_l3467_346724

theorem negation_of_existence (p : ℝ → Prop) :
  (¬∃ x : ℝ, p x) ↔ (∀ x : ℝ, ¬p x) :=
by sorry

theorem negation_of_inequality :
  (¬∃ x : ℝ, 2^x ≥ 2*x + 1) ↔ (∀ x : ℝ, 2^x < 2*x + 1) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_existence_negation_of_inequality_l3467_346724


namespace NUMINAMATH_CALUDE_trapezoid_perimeter_is_228_l3467_346792

/-- A trapezoid with given side lengths -/
structure Trapezoid where
  EF : ℝ
  GH : ℝ
  EG : ℝ
  FH : ℝ
  h_EF_longer : EF > GH

/-- The perimeter of a trapezoid -/
def perimeter (t : Trapezoid) : ℝ := t.EF + t.GH + t.EG + t.FH

/-- Theorem stating that the perimeter of the given trapezoid is 228 -/
theorem trapezoid_perimeter_is_228 : 
  ∀ (t : Trapezoid), t.EF = 90 ∧ t.GH = 40 ∧ t.EG = 53 ∧ t.FH = 45 → perimeter t = 228 := by
  sorry

end NUMINAMATH_CALUDE_trapezoid_perimeter_is_228_l3467_346792


namespace NUMINAMATH_CALUDE_expansion_properties_l3467_346748

theorem expansion_properties (n : ℕ) (hn : n ≥ 3) :
  (∃ k : ℕ, k ≥ 1 ∧ k ≤ n ∧ (14 - 3 * k) % 4 = 0) ∧
  (∀ k : ℕ, k ≥ 0 → k ≤ n → Nat.choose n k * ((-1/2)^k : ℚ) ≤ 21/4) :=
by sorry

end NUMINAMATH_CALUDE_expansion_properties_l3467_346748


namespace NUMINAMATH_CALUDE_g_range_l3467_346745

noncomputable def g (x : ℝ) : ℝ :=
  (Real.arccos (x / 3))^2 + (Real.pi / 2) * Real.arcsin (x / 3) - (Real.arcsin (x / 3))^2 + 
  (Real.pi^2 / 18) * (x^2 + 9*x + 27)

theorem g_range :
  ∀ y ∈ Set.range g, π^2 / 6 ≤ y ∧ y ≤ 4*π^2 / 3 ∧
  ∃ x ∈ Set.Icc (-3 : ℝ) 3, g x = π^2 / 6 ∧
  ∃ x ∈ Set.Icc (-3 : ℝ) 3, g x = 4*π^2 / 3 :=
by sorry

end NUMINAMATH_CALUDE_g_range_l3467_346745


namespace NUMINAMATH_CALUDE_coefficient_x_squared_is_120_l3467_346760

/-- The coefficient of x^2 in the expansion of (1+x)^2 + (1+x)^3 + ... + (1+x)^9 -/
def coefficient_x_squared : ℕ :=
  (Finset.range 8).sum (λ n => Nat.choose (n + 2) 2)

/-- Theorem stating that the coefficient of x^2 in the expansion is 120 -/
theorem coefficient_x_squared_is_120 : coefficient_x_squared = 120 := by
  sorry

end NUMINAMATH_CALUDE_coefficient_x_squared_is_120_l3467_346760


namespace NUMINAMATH_CALUDE_perfect_square_trinomial_condition_l3467_346723

/-- A trinomial ax^2 + bx + c is a perfect square if there exist real numbers p and q
    such that ax^2 + bx + c = (px + q)^2 for all real x. -/
def is_perfect_square_trinomial (a b c : ℝ) : Prop :=
  ∃ p q : ℝ, ∀ x : ℝ, a * x^2 + b * x + c = (p * x + q)^2

/-- If x^2 + kx + 9 is a perfect square trinomial, then k = 6 or k = -6. -/
theorem perfect_square_trinomial_condition (k : ℝ) :
  is_perfect_square_trinomial 1 k 9 → k = 6 ∨ k = -6 := by
  sorry

end NUMINAMATH_CALUDE_perfect_square_trinomial_condition_l3467_346723


namespace NUMINAMATH_CALUDE_intersection_points_on_circle_l3467_346710

-- Define the parabolas
def Parabola1 (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c
def Parabola2 (d e f : ℝ) (y : ℝ) : ℝ := d * y^2 + e * y + f

-- Define the intersection points
def IntersectionPoints (a b c d e f : ℝ) : Set (ℝ × ℝ) :=
  {(x, y) | Parabola1 a b c x = y ∧ Parabola2 d e f y = x}

-- Define a circle
def Circle (center : ℝ × ℝ) (radius : ℝ) : Set (ℝ × ℝ) :=
  {(x, y) | (x - center.1)^2 + (y - center.2)^2 = radius^2}

-- Theorem statement
theorem intersection_points_on_circle 
  (a b c d e f : ℝ) (ha : a > 0) (hd : d > 0) :
  ∃ (center : ℝ × ℝ) (radius : ℝ),
    IntersectionPoints a b c d e f ⊆ Circle center radius :=
sorry

end NUMINAMATH_CALUDE_intersection_points_on_circle_l3467_346710


namespace NUMINAMATH_CALUDE_solve_quadratic_equation_solve_linear_equation_l3467_346756

-- Equation 1
theorem solve_quadratic_equation (x : ℝ) :
  2 * x^2 - 5 * x + 1 = 0 ↔ x = (5 + Real.sqrt 17) / 4 ∨ x = (5 - Real.sqrt 17) / 4 :=
sorry

-- Equation 2
theorem solve_linear_equation (x : ℝ) :
  3 * x * (x - 2) = 2 * (2 - x) ↔ x = 2 ∨ x = -2/3 :=
sorry

end NUMINAMATH_CALUDE_solve_quadratic_equation_solve_linear_equation_l3467_346756


namespace NUMINAMATH_CALUDE_double_room_percentage_l3467_346744

theorem double_room_percentage (total_students : ℝ) (h : total_students > 0) :
  let students_in_double_rooms := 0.75 * total_students
  let double_rooms := students_in_double_rooms / 2
  let students_in_single_rooms := 0.25 * total_students
  let single_rooms := students_in_single_rooms
  let total_rooms := double_rooms + single_rooms
  (double_rooms / total_rooms) * 100 = 60 := by
  sorry

end NUMINAMATH_CALUDE_double_room_percentage_l3467_346744


namespace NUMINAMATH_CALUDE_probability_divisor_of_12_l3467_346721

/-- A fair 6-sided die -/
def Die : Finset ℕ := {1, 2, 3, 4, 5, 6}

/-- The set of divisors of 12 that appear on the die -/
def DivisorsOf12OnDie : Finset ℕ := {1, 2, 3, 4, 6}

/-- The probability of an event on a fair die -/
def probability (event : Finset ℕ) : ℚ :=
  (event ∩ Die).card / Die.card

theorem probability_divisor_of_12 :
  probability DivisorsOf12OnDie = 5 / 6 := by
  sorry

end NUMINAMATH_CALUDE_probability_divisor_of_12_l3467_346721


namespace NUMINAMATH_CALUDE_octahedron_cube_volume_ratio_l3467_346743

/-- The volume ratio of an octahedron formed by connecting the centers of adjacent faces of a cube
    to the volume of the cube itself is 1/6, given that the cube has an edge length of 2 units. -/
theorem octahedron_cube_volume_ratio :
  let cube_edge : ℝ := 2
  let cube_volume : ℝ := cube_edge ^ 3
  let octahedron_edge : ℝ := Real.sqrt 8
  let octahedron_volume : ℝ := (Real.sqrt 2 / 3) * octahedron_edge ^ 3
  octahedron_volume / cube_volume = 1 / 6 := by
sorry

end NUMINAMATH_CALUDE_octahedron_cube_volume_ratio_l3467_346743


namespace NUMINAMATH_CALUDE_digit_sum_equation_l3467_346783

/-- Given that a000 + a998 + a999 = 22997, prove that a = 7 -/
theorem digit_sum_equation (a : ℕ) : 
  a * 1000 + a * 998 + a * 999 = 22997 → a = 7 := by
  sorry

end NUMINAMATH_CALUDE_digit_sum_equation_l3467_346783


namespace NUMINAMATH_CALUDE_quadratic_root_form_n_l3467_346705

def quadratic_equation (x : ℝ) : Prop := 3 * x^2 - 8 * x - 5 = 0

def root_form (x m n p : ℝ) : Prop :=
  x = (m + Real.sqrt n) / p ∨ x = (m - Real.sqrt n) / p

theorem quadratic_root_form_n :
  ∃ (m n p : ℕ+),
    (∀ x : ℝ, quadratic_equation x → root_form x m n p) ∧
    Nat.gcd m.val (Nat.gcd n.val p.val) = 1 ∧
    n = 124 :=
sorry

end NUMINAMATH_CALUDE_quadratic_root_form_n_l3467_346705


namespace NUMINAMATH_CALUDE_three_zeros_properties_l3467_346763

variable (a : ℝ) (x₁ x₂ x₃ : ℝ)

def f (x : ℝ) := a * (2 * x - 1) * abs (x + 1) - 2 * x - 1

theorem three_zeros_properties 
  (h_zeros : f a x₁ = 0 ∧ f a x₂ = 0 ∧ f a x₃ = 0)
  (h_order : x₁ < x₂ ∧ x₂ < x₃) :
  (1 / a < x₃ ∧ x₃ < 1 / a + 1 / x₃) ∧ a * (x₂ - x₁) < 1 := by
  sorry

end NUMINAMATH_CALUDE_three_zeros_properties_l3467_346763


namespace NUMINAMATH_CALUDE_prob_white_ball_l3467_346778

/-- Probability of drawing a white ball from a box with inaccessible balls -/
theorem prob_white_ball (total : ℕ) (white : ℕ) (locked : ℕ) : 
  total = 17 → white = 7 → locked = 3 → 
  (white : ℚ) / (total - locked : ℚ) = 1 / 2 := by
sorry

end NUMINAMATH_CALUDE_prob_white_ball_l3467_346778


namespace NUMINAMATH_CALUDE_number_difference_l3467_346766

theorem number_difference (x y : ℝ) (h1 : x + y = 40) (h2 : 3 * y - 4 * x = 10) (h3 : y > x) :
  y - x = 8.58 := by
  sorry

end NUMINAMATH_CALUDE_number_difference_l3467_346766


namespace NUMINAMATH_CALUDE_next_chime_together_l3467_346797

def town_hall_interval : ℕ := 18
def library_interval : ℕ := 24
def railway_interval : ℕ := 30

def minutes_in_hour : ℕ := 60

theorem next_chime_together (start_hour : ℕ) : 
  ∃ (hours : ℕ), 
    hours * minutes_in_hour = Nat.lcm town_hall_interval (Nat.lcm library_interval railway_interval) ∧ 
    hours = 6 := by
  sorry

end NUMINAMATH_CALUDE_next_chime_together_l3467_346797


namespace NUMINAMATH_CALUDE_m_cubed_plus_two_m_squared_minus_2001_l3467_346775

theorem m_cubed_plus_two_m_squared_minus_2001 (m : ℝ) (h : m^2 + m - 1 = 0) :
  m^3 + 2*m^2 - 2001 = -2000 := by
  sorry

end NUMINAMATH_CALUDE_m_cubed_plus_two_m_squared_minus_2001_l3467_346775


namespace NUMINAMATH_CALUDE_w_squared_value_l3467_346765

theorem w_squared_value (w : ℝ) (h : (2*w + 19)^2 = (4*w + 9)*(3*w + 13)) :
  w^2 = ((6 + Real.sqrt 524) / 4)^2 := by
  sorry

end NUMINAMATH_CALUDE_w_squared_value_l3467_346765


namespace NUMINAMATH_CALUDE_triangle_count_l3467_346796

theorem triangle_count (num_circles : ℕ) (num_triangles : ℕ) : 
  num_circles = 5 → num_triangles = 2 * num_circles → num_triangles = 10 := by
  sorry

end NUMINAMATH_CALUDE_triangle_count_l3467_346796


namespace NUMINAMATH_CALUDE_benjie_current_age_l3467_346704

/-- Benjie's age in years -/
def benjie_age : ℕ := 6

/-- Margo's age in years -/
def margo_age : ℕ := 1

/-- The age difference between Benjie and Margo in years -/
def age_difference : ℕ := 5

/-- The number of years until Margo is 4 years old -/
def years_until_margo_4 : ℕ := 3

theorem benjie_current_age :
  (benjie_age = margo_age + age_difference) ∧
  (margo_age + years_until_margo_4 = 4) →
  benjie_age = 6 := by sorry

end NUMINAMATH_CALUDE_benjie_current_age_l3467_346704


namespace NUMINAMATH_CALUDE_trigonometric_identity_l3467_346742

theorem trigonometric_identity : 
  1 / Real.cos (70 * π / 180) + Real.sqrt 3 / Real.sin (70 * π / 180) = 4 * Real.tan (10 * π / 180) := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_identity_l3467_346742


namespace NUMINAMATH_CALUDE_subset_implies_a_equals_one_l3467_346795

-- Define the sets A and B
def A : Set ℝ := {-1, 0, 2}
def B (a : ℝ) : Set ℝ := {2^a}

-- State the theorem
theorem subset_implies_a_equals_one (a : ℝ) :
  B a ⊆ A → a = 1 := by sorry

end NUMINAMATH_CALUDE_subset_implies_a_equals_one_l3467_346795


namespace NUMINAMATH_CALUDE_problem_1_problem_2_problem_3_problem_4_l3467_346719

-- Problem 1
theorem problem_1 : 3 * (13 / 15) - 2 * (13 / 14) + 5 * (2 / 15) - 1 * (1 / 14) = 5 := by sorry

-- Problem 2
theorem problem_2 : (1 / 9) / (2 / (3 / 4 - 2 / 3)) = 1 / 216 := by sorry

-- Problem 3
theorem problem_3 : 99 * 78.6 + 786 * 0.3 - 7.86 * 20 = 7860 := by sorry

-- Problem 4
theorem problem_4 : 2015 / (2015 * 2015 / 2016) = 2016 / 2017 := by sorry

end NUMINAMATH_CALUDE_problem_1_problem_2_problem_3_problem_4_l3467_346719


namespace NUMINAMATH_CALUDE_last_four_average_l3467_346720

/-- Given a list of seven real numbers where the average of all seven is 62
    and the average of the first three is 58, prove that the average of the
    last four numbers is 65. -/
theorem last_four_average (list : List ℝ) : 
  list.length = 7 →
  list.sum / 7 = 62 →
  (list.take 3).sum / 3 = 58 →
  (list.drop 3).sum / 4 = 65 := by
sorry

end NUMINAMATH_CALUDE_last_four_average_l3467_346720


namespace NUMINAMATH_CALUDE_book_arrangement_count_l3467_346793

theorem book_arrangement_count :
  let total_books : ℕ := 9
  let arabic_books : ℕ := 2
  let german_books : ℕ := 3
  let spanish_books : ℕ := 4
  let arabic_unit : ℕ := 1
  let spanish_unit : ℕ := 1
  let total_units : ℕ := arabic_unit + spanish_unit + german_books

  (total_books = arabic_books + german_books + spanish_books) →
  (Nat.factorial total_units * Nat.factorial arabic_books * Nat.factorial spanish_books = 5760) :=
by sorry

end NUMINAMATH_CALUDE_book_arrangement_count_l3467_346793


namespace NUMINAMATH_CALUDE_alberts_number_l3467_346785

theorem alberts_number (a b c : ℚ) 
  (h1 : a = 2 * b + 1)
  (h2 : b = 2 * c + 1)
  (h3 : c = 2 * a + 2) :
  a = -11/7 := by
sorry

end NUMINAMATH_CALUDE_alberts_number_l3467_346785


namespace NUMINAMATH_CALUDE_cutlery_added_l3467_346736

def initial_forks : ℕ := 6

def initial_knives (forks : ℕ) : ℕ := forks + 9

def initial_spoons (knives : ℕ) : ℕ := 2 * knives

def initial_teaspoons (forks : ℕ) : ℕ := forks / 2

def total_initial_cutlery (forks knives spoons teaspoons : ℕ) : ℕ :=
  forks + knives + spoons + teaspoons

def final_total_cutlery : ℕ := 62

theorem cutlery_added :
  final_total_cutlery - total_initial_cutlery initial_forks
    (initial_knives initial_forks)
    (initial_spoons (initial_knives initial_forks))
    (initial_teaspoons initial_forks) = 8 := by
  sorry

end NUMINAMATH_CALUDE_cutlery_added_l3467_346736


namespace NUMINAMATH_CALUDE_common_term_formula_l3467_346711

def x (n : ℕ) : ℕ := 2 * n - 1
def y (n : ℕ) : ℕ := n ^ 2

def is_common_term (m : ℕ) : Prop :=
  ∃ n k : ℕ, x n = m ∧ y k = m

def c (n : ℕ) : ℕ := (2 * n - 1) ^ 2

theorem common_term_formula :
  ∀ n : ℕ, is_common_term (c n) ∧
  (∀ m : ℕ, m < c n → is_common_term m → ∃ k < n, c k = m) :=
sorry

end NUMINAMATH_CALUDE_common_term_formula_l3467_346711


namespace NUMINAMATH_CALUDE_quadrilateral_point_D_l3467_346713

-- Define a structure for a point in 2D space
structure Point2D where
  x : ℝ
  y : ℝ

-- Define a structure for a quadrilateral
structure Quadrilateral where
  A : Point2D
  B : Point2D
  C : Point2D
  D : Point2D

-- Define a property for parallel sides
def parallel_sides (q : Quadrilateral) : Prop :=
  (q.A.x - q.B.x) * (q.C.y - q.D.y) = (q.A.y - q.B.y) * (q.C.x - q.D.x) ∧
  (q.A.x - q.D.x) * (q.B.y - q.C.y) = (q.A.y - q.D.y) * (q.B.x - q.C.x)

-- Theorem statement
theorem quadrilateral_point_D (q : Quadrilateral) :
  q.A = Point2D.mk (-2) 0 ∧
  q.B = Point2D.mk 6 8 ∧
  q.C = Point2D.mk 8 6 ∧
  parallel_sides q →
  q.D = Point2D.mk 0 (-2) := by
  sorry

end NUMINAMATH_CALUDE_quadrilateral_point_D_l3467_346713


namespace NUMINAMATH_CALUDE_shanghai_score_is_75_l3467_346738

/-- The score of Yao Ming in the basketball game -/
def yao_ming_score : ℕ := 30

/-- The winning margin of the Shanghai team over the Beijing team -/
def shanghai_margin : ℕ := 10

/-- Calculates the total score of both teams based on Yao Ming's score -/
def total_score (yao_score : ℕ) : ℕ := 5 * yao_score - 10

/-- The score of the Shanghai team -/
def shanghai_score : ℕ := 75

/-- The score of the Beijing team -/
def beijing_score : ℕ := shanghai_score - shanghai_margin

theorem shanghai_score_is_75 :
  shanghai_score = 75 ∧
  shanghai_score - beijing_score = shanghai_margin ∧
  shanghai_score + beijing_score = total_score yao_ming_score :=
by sorry

end NUMINAMATH_CALUDE_shanghai_score_is_75_l3467_346738


namespace NUMINAMATH_CALUDE_reflection_of_point_across_x_axis_l3467_346754

/-- Represents a point in the 2D Cartesian coordinate system -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- Reflects a point across the x-axis -/
def reflectAcrossXAxis (p : Point2D) : Point2D :=
  { x := p.x, y := -p.y }

/-- Theorem: The reflection of the point P(-3,1) across the x-axis is (-3,-1) -/
theorem reflection_of_point_across_x_axis :
  let P : Point2D := { x := -3, y := 1 }
  reflectAcrossXAxis P = { x := -3, y := -1 } := by
  sorry

end NUMINAMATH_CALUDE_reflection_of_point_across_x_axis_l3467_346754


namespace NUMINAMATH_CALUDE_arithmetic_geometric_ratio_l3467_346772

/-- An arithmetic sequence with common difference d ≠ 0 -/
def arithmetic_sequence (a : ℕ → ℚ) (d : ℚ) : Prop :=
  d ≠ 0 ∧ ∀ n : ℕ, a (n + 1) = a n + d

/-- Three terms form a geometric sequence -/
def geometric_sequence (x y z : ℚ) : Prop :=
  y * y = x * z

/-- The main theorem -/
theorem arithmetic_geometric_ratio
  (a : ℕ → ℚ) (d : ℚ)
  (h_arith : arithmetic_sequence a d)
  (h_geom : geometric_sequence (a 1) (a 3) (a 9)) :
  (a 1 + a 3 + a 5) / (a 2 + a 4 + a 6) = 3 / 4 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_ratio_l3467_346772


namespace NUMINAMATH_CALUDE_replaced_person_age_l3467_346729

theorem replaced_person_age 
  (n : ℕ) 
  (original_total_age : ℕ) 
  (new_person_age : ℕ) 
  (average_decrease : ℕ) :
  n = 10 →
  new_person_age = 10 →
  average_decrease = 3 →
  (original_total_age : ℚ) / n - average_decrease = 
    (original_total_age - (original_total_age / n * n - new_person_age) : ℚ) / n →
  (original_total_age / n * n - new_person_age : ℚ) / n = 40 :=
by sorry

end NUMINAMATH_CALUDE_replaced_person_age_l3467_346729


namespace NUMINAMATH_CALUDE_f_positive_before_root_l3467_346739

noncomputable def f (x : ℝ) : ℝ := (1/3)^x - Real.log x / Real.log 2

theorem f_positive_before_root (x₀ a : ℝ) 
  (h_root : f x₀ = 0)
  (h_decreasing : ∀ x y, x < y → f x > f y)
  (h_a_pos : 0 < a)
  (h_a_lt_x₀ : a < x₀) : 
  f a > 0 := by
  sorry

end NUMINAMATH_CALUDE_f_positive_before_root_l3467_346739


namespace NUMINAMATH_CALUDE_angle_through_point_l3467_346700

theorem angle_through_point (α : Real) :
  (∃ (x y : Real), x = -1 ∧ y = 2 ∧ x = Real.cos α * Real.sqrt (x^2 + y^2) ∧ y = Real.sin α * Real.sqrt (x^2 + y^2)) →
  Real.sin α = 2 * Real.sqrt 5 / 5 ∧
  Real.cos α = -Real.sqrt 5 / 5 ∧
  Real.tan α = -2 ∧
  Real.tan (α - Real.pi/4) = 3 := by
sorry

end NUMINAMATH_CALUDE_angle_through_point_l3467_346700


namespace NUMINAMATH_CALUDE_max_product_sum_constraint_l3467_346761

theorem max_product_sum_constraint (w x y z : ℝ) : 
  w ≥ 0 → x ≥ 0 → y ≥ 0 → z ≥ 0 → w + x + y + z = 200 → 
  (w + x) * (y + z) ≤ 10000 := by
sorry

end NUMINAMATH_CALUDE_max_product_sum_constraint_l3467_346761


namespace NUMINAMATH_CALUDE_jerry_field_hours_eq_96_l3467_346734

/-- The number of hours Jerry spends at the field watching his daughters play and practice -/
def jerry_field_hours : ℕ :=
  let num_daughters : ℕ := 2
  let games_per_daughter : ℕ := 8
  let practice_hours_per_game : ℕ := 4
  let game_duration_hours : ℕ := 2
  
  let game_hours_per_daughter : ℕ := games_per_daughter * game_duration_hours
  let practice_hours_per_daughter : ℕ := games_per_daughter * practice_hours_per_game
  
  num_daughters * (game_hours_per_daughter + practice_hours_per_daughter)

theorem jerry_field_hours_eq_96 : jerry_field_hours = 96 := by
  sorry

end NUMINAMATH_CALUDE_jerry_field_hours_eq_96_l3467_346734


namespace NUMINAMATH_CALUDE_identity_element_is_one_zero_l3467_346725

-- Define the operation ⊕
def oplus (a b c d : ℝ) : ℝ × ℝ := (a * c + b * d, a * d + b * c)

-- State the theorem
theorem identity_element_is_one_zero :
  (∀ a b : ℝ, oplus a b x y = (a, b)) → (x, y) = (1, 0) := by
  sorry

end NUMINAMATH_CALUDE_identity_element_is_one_zero_l3467_346725


namespace NUMINAMATH_CALUDE_sum_of_integers_l3467_346788

theorem sum_of_integers (x y z w : ℤ) 
  (eq1 : x - y + z = 7)
  (eq2 : y - z + w = 8)
  (eq3 : z - w + x = 4)
  (eq4 : w - x + y = 3) :
  x + y + z + w = 11 := by
sorry

end NUMINAMATH_CALUDE_sum_of_integers_l3467_346788


namespace NUMINAMATH_CALUDE_total_seeds_in_garden_l3467_346757

/-- Represents the number of beds of each type in the garden -/
def num_beds : ℕ := 2

/-- Represents the number of rows in a top bed -/
def top_rows : ℕ := 4

/-- Represents the number of seeds per row in a top bed -/
def top_seeds_per_row : ℕ := 25

/-- Represents the number of rows in a medium bed -/
def medium_rows : ℕ := 3

/-- Represents the number of seeds per row in a medium bed -/
def medium_seeds_per_row : ℕ := 20

/-- Calculates the total number of seeds that can be planted in Grace's raised bed garden -/
theorem total_seeds_in_garden : 
  num_beds * (top_rows * top_seeds_per_row + medium_rows * medium_seeds_per_row) = 320 := by
  sorry

end NUMINAMATH_CALUDE_total_seeds_in_garden_l3467_346757


namespace NUMINAMATH_CALUDE_area_circle_inscribed_equilateral_triangle_l3467_346731

theorem area_circle_inscribed_equilateral_triangle (p : ℝ) (h : p > 0) :
  ∃ (R : ℝ), R > 0 ∧
  ∃ (s : ℝ), s > 0 ∧
  p = 3 * s ∧
  R = s / Real.sqrt 3 ∧
  π * R^2 = π * p^2 / 27 :=
by sorry

end NUMINAMATH_CALUDE_area_circle_inscribed_equilateral_triangle_l3467_346731


namespace NUMINAMATH_CALUDE_polygon_sides_l3467_346727

theorem polygon_sides (n : ℕ) (h : (n - 2) * 180 = 4 * 360 + 180) : n = 9 := by
  sorry

end NUMINAMATH_CALUDE_polygon_sides_l3467_346727


namespace NUMINAMATH_CALUDE_train_speed_fraction_l3467_346755

theorem train_speed_fraction (usual_time : ℝ) (delay : ℝ) : 
  usual_time = 12 → delay = 9 → 
  (usual_time / (usual_time + delay)) = (4 : ℝ) / 7 := by
sorry

end NUMINAMATH_CALUDE_train_speed_fraction_l3467_346755


namespace NUMINAMATH_CALUDE_soda_bottle_count_l3467_346747

theorem soda_bottle_count (regular_soda : ℕ) (diet_soda : ℕ) 
  (h1 : regular_soda = 49) (h2 : diet_soda = 40) : 
  regular_soda + diet_soda = 89 := by
  sorry

end NUMINAMATH_CALUDE_soda_bottle_count_l3467_346747


namespace NUMINAMATH_CALUDE_binomial_prob_half_l3467_346791

/-- A random variable following a binomial distribution -/
structure BinomialRV where
  n : ℕ
  p : ℝ
  h1 : 0 ≤ p ∧ p ≤ 1

/-- Expected value of a binomial random variable -/
def expected_value (ξ : BinomialRV) : ℝ := ξ.n * ξ.p

/-- Variance of a binomial random variable -/
def variance (ξ : BinomialRV) : ℝ := ξ.n * ξ.p * (1 - ξ.p)

theorem binomial_prob_half (ξ : BinomialRV) 
  (h_exp : expected_value ξ = 2)
  (h_var : variance ξ = 1) : 
  ξ.p = 0.5 := by
  sorry

end NUMINAMATH_CALUDE_binomial_prob_half_l3467_346791


namespace NUMINAMATH_CALUDE_simplify_expression_l3467_346740

theorem simplify_expression (x : ℝ) : (x + 15) + (150 * x + 20) = 151 * x + 35 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l3467_346740


namespace NUMINAMATH_CALUDE_negative_x_gt_1_is_inequality_l3467_346750

-- Define what an inequality is
def is_inequality (expr : Prop) : Prop :=
  ∃ (a b : ℝ), (expr = (a > b) ∨ expr = (a < b) ∨ expr = (a ≥ b) ∨ expr = (a ≤ b))

-- Theorem to prove
theorem negative_x_gt_1_is_inequality :
  is_inequality (-x > 1) :=
sorry

end NUMINAMATH_CALUDE_negative_x_gt_1_is_inequality_l3467_346750


namespace NUMINAMATH_CALUDE_water_students_l3467_346776

theorem water_students (total : ℕ) (juice_percent : ℚ) (water_percent : ℚ) (juice_students : ℕ) : ℕ :=
  sorry

end NUMINAMATH_CALUDE_water_students_l3467_346776


namespace NUMINAMATH_CALUDE_symmetry_point_of_sine_function_l3467_346746

/-- Given a function f(x) = sin(ωx + π/6) with ω > 0, if the distance between adjacent
    symmetry axes is π/2 and the graph is symmetrical about (x₀, 0) where x₀ ∈ [0, π/2],
    then x₀ = 5π/12 -/
theorem symmetry_point_of_sine_function (ω : ℝ) (x₀ : ℝ) :
  ω > 0 →
  (2 * π) / ω = π →
  x₀ ∈ Set.Icc 0 (π / 2) →
  (∀ x, Real.sin (ω * x + π / 6) = Real.sin (ω * (2 * x₀ - x) + π / 6)) →
  x₀ = 5 * π / 12 := by
  sorry

end NUMINAMATH_CALUDE_symmetry_point_of_sine_function_l3467_346746
