import Mathlib

namespace NUMINAMATH_CALUDE_total_cinnamon_swirls_l3384_338434

/-- The number of people eating cinnamon swirls -/
def num_people : ℕ := 3

/-- The number of pieces Jane ate -/
def janes_pieces : ℕ := 4

/-- Theorem: If there are 3 people eating an equal number of cinnamon swirls, 
    and one person ate 4 pieces, then the total number of pieces is 12. -/
theorem total_cinnamon_swirls : 
  num_people * janes_pieces = 12 := by sorry

end NUMINAMATH_CALUDE_total_cinnamon_swirls_l3384_338434


namespace NUMINAMATH_CALUDE_pizza_slices_l3384_338416

theorem pizza_slices : ∃ S : ℕ,
  S > 0 ∧
  (3 * S / 4 : ℚ) > 0 ∧
  (9 * S / 16 : ℚ) > 4 ∧
  (9 * S / 16 : ℚ) - 4 = 5 ∧
  S = 16 := by
  sorry

end NUMINAMATH_CALUDE_pizza_slices_l3384_338416


namespace NUMINAMATH_CALUDE_volcano_count_l3384_338442

theorem volcano_count (total : ℕ) (intact : ℕ) : 
  (intact : ℝ) = total * (1 - 0.2) * (1 - 0.4) * (1 - 0.5) ∧ intact = 48 → 
  total = 200 := by
sorry

end NUMINAMATH_CALUDE_volcano_count_l3384_338442


namespace NUMINAMATH_CALUDE_fraction_equivalence_l3384_338493

theorem fraction_equivalence : ∃ x : ℚ, (4 + x) / (7 + x) = 3 / 4 := by
  use 5
  sorry

end NUMINAMATH_CALUDE_fraction_equivalence_l3384_338493


namespace NUMINAMATH_CALUDE_train_crossing_time_l3384_338402

/-- Proves that a train with the given length and speed takes the calculated time to cross an electric pole -/
theorem train_crossing_time (train_length : Real) (train_speed_kmh : Real) : 
  train_length = 120 ∧ train_speed_kmh = 72 → 
  (train_length / (train_speed_kmh * 1000 / 3600)) = 6 := by
  sorry

#check train_crossing_time

end NUMINAMATH_CALUDE_train_crossing_time_l3384_338402


namespace NUMINAMATH_CALUDE_divisible_by_15_20_25_between_1000_and_2000_l3384_338447

theorem divisible_by_15_20_25_between_1000_and_2000 : 
  ∃! n : ℕ, (∀ k : ℕ, 1000 < k ∧ k < 2000 ∧ 15 ∣ k ∧ 20 ∣ k ∧ 25 ∣ k → k ∈ Finset.range n) ∧ 
  (∀ k : ℕ, k ∈ Finset.range n → 1000 < k ∧ k < 2000 ∧ 15 ∣ k ∧ 20 ∣ k ∧ 25 ∣ k) :=
by sorry

end NUMINAMATH_CALUDE_divisible_by_15_20_25_between_1000_and_2000_l3384_338447


namespace NUMINAMATH_CALUDE_first_dog_takes_one_more_than_second_l3384_338453

def dog_bone_problem (second_dog_bones : ℕ) : Prop :=
  let first_dog_bones := 3
  let third_dog_bones := 2 * second_dog_bones
  let fourth_dog_bones := 1
  let fifth_dog_bones := 2 * fourth_dog_bones
  first_dog_bones + second_dog_bones + third_dog_bones + fourth_dog_bones + fifth_dog_bones = 12

theorem first_dog_takes_one_more_than_second :
  ∃ (second_dog_bones : ℕ), dog_bone_problem second_dog_bones ∧ 3 = second_dog_bones + 1 := by
  sorry

end NUMINAMATH_CALUDE_first_dog_takes_one_more_than_second_l3384_338453


namespace NUMINAMATH_CALUDE_intersection_equidistant_l3384_338411

-- Define the quadrilateral ABCD
structure Quadrilateral :=
  (A B C D : ℝ × ℝ)

-- Define the condition AB = CD
def equal_sides (q : Quadrilateral) : Prop :=
  dist q.A q.B = dist q.C q.D

-- Define the intersection point O of diagonals AC and BD
def intersection_point (q : Quadrilateral) : ℝ × ℝ :=
  sorry

-- Define a line passing through O
structure Line :=
  (slope : ℝ)
  (point : ℝ × ℝ)

-- Define the intersection points of a line with the quadrilateral sides
def intersection_points (q : Quadrilateral) (l : Line) :
  (ℝ × ℝ) × (ℝ × ℝ) × (ℝ × ℝ) × (ℝ × ℝ) :=
  sorry

-- Define the intersection of a line with BD
def intersection_with_diagonal (q : Quadrilateral) (l : Line) : ℝ × ℝ :=
  sorry

-- Main theorem
theorem intersection_equidistant (q : Quadrilateral) (l1 l2 : Line)
  (h : equal_sides q) :
  let O := intersection_point q
  let I := intersection_with_diagonal q l1
  let J := intersection_with_diagonal q l2
  dist O I = dist O J :=
sorry

end NUMINAMATH_CALUDE_intersection_equidistant_l3384_338411


namespace NUMINAMATH_CALUDE_g_1994_of_4_l3384_338463

def g (x : ℚ) : ℚ := (2 + x) / (2 - 4 * x)

def g_n : ℕ → (ℚ → ℚ)
  | 0 => id
  | n + 1 => λ x => g (g_n n x)

theorem g_1994_of_4 : g_n 1994 4 = 87 / 50 := by
  sorry

end NUMINAMATH_CALUDE_g_1994_of_4_l3384_338463


namespace NUMINAMATH_CALUDE_add_decimals_l3384_338412

theorem add_decimals : (7.45 : ℝ) + 2.56 = 10.01 := by
  sorry

end NUMINAMATH_CALUDE_add_decimals_l3384_338412


namespace NUMINAMATH_CALUDE_num_valid_sequences_is_377_l3384_338460

/-- Represents a sequence of A's and B's -/
inductive ABSequence
  | A : ABSequence
  | B : ABSequence
  | cons : ABSequence → ABSequence → ABSequence

/-- Returns true if the given sequence satisfies the run length conditions -/
def validSequence : ABSequence → Bool :=
  sorry

/-- Returns the length of the given sequence -/
def sequenceLength : ABSequence → Nat :=
  sorry

/-- Returns true if the given sequence has length 15 and satisfies the run length conditions -/
def validSequenceOfLength15 (s : ABSequence) : Bool :=
  validSequence s ∧ sequenceLength s = 15

/-- The number of valid sequences of length 15 -/
def numValidSequences : Nat :=
  sorry

theorem num_valid_sequences_is_377 : numValidSequences = 377 := by
  sorry

end NUMINAMATH_CALUDE_num_valid_sequences_is_377_l3384_338460


namespace NUMINAMATH_CALUDE_chord_equation_l3384_338491

/-- A chord of the hyperbola x^2 - y^2 = 1 with midpoint (2, 1) -/
structure Chord where
  /-- First endpoint of the chord -/
  p1 : ℝ × ℝ
  /-- Second endpoint of the chord -/
  p2 : ℝ × ℝ
  /-- The chord endpoints lie on the hyperbola -/
  h1 : p1.1^2 - p1.2^2 = 1
  h2 : p2.1^2 - p2.2^2 = 1
  /-- The midpoint of the chord is (2, 1) -/
  h3 : (p1.1 + p2.1) / 2 = 2
  h4 : (p1.2 + p2.2) / 2 = 1

/-- The equation of the line containing the chord is y = 2x - 3 -/
theorem chord_equation (c : Chord) : 
  ∃ (m b : ℝ), m = 2 ∧ b = -3 ∧ ∀ (x y : ℝ), y = m * x + b ↔ 
  ∃ (t : ℝ), x = (1 - t) * c.p1.1 + t * c.p2.1 ∧ y = (1 - t) * c.p1.2 + t * c.p2.2 :=
sorry

end NUMINAMATH_CALUDE_chord_equation_l3384_338491


namespace NUMINAMATH_CALUDE_all_divisible_by_nine_l3384_338476

/-- A five-digit number represented as a tuple of five natural numbers -/
def FiveDigitNumber := (ℕ × ℕ × ℕ × ℕ × ℕ)

/-- The sum of digits in a five-digit number -/
def digitSum (n : FiveDigitNumber) : ℕ :=
  n.1 + n.2.1 + n.2.2.1 + n.2.2.2.1 + n.2.2.2.2

/-- Predicate for a valid five-digit number -/
def isValidFiveDigitNumber (n : FiveDigitNumber) : Prop :=
  1 ≤ n.1 ∧ n.1 ≤ 9 ∧ 0 ≤ n.2.1 ∧ n.2.1 ≤ 9 ∧
  0 ≤ n.2.2.1 ∧ n.2.2.1 ≤ 9 ∧ 0 ≤ n.2.2.2.1 ∧ n.2.2.2.1 ≤ 9 ∧
  0 ≤ n.2.2.2.2 ∧ n.2.2.2.2 ≤ 9

/-- The set of all valid five-digit numbers with digit sum 36 -/
def S : Set FiveDigitNumber :=
  {n | isValidFiveDigitNumber n ∧ digitSum n = 36}

/-- The numeric value of a five-digit number -/
def numericValue (n : FiveDigitNumber) : ℕ :=
  10000 * n.1 + 1000 * n.2.1 + 100 * n.2.2.1 + 10 * n.2.2.2.1 + n.2.2.2.2

theorem all_divisible_by_nine :
  ∀ n ∈ S, (numericValue n) % 9 = 0 := by
  sorry

end NUMINAMATH_CALUDE_all_divisible_by_nine_l3384_338476


namespace NUMINAMATH_CALUDE_zongzi_production_theorem_l3384_338471

/-- The average daily production of zongzi for Team A -/
def team_a_production : ℝ := 200

/-- The average daily production of zongzi for Team B -/
def team_b_production : ℝ := 150

/-- Theorem stating that given the conditions, the average daily production
    of zongzi for Team A is 200 bags and for Team B is 150 bags -/
theorem zongzi_production_theorem :
  (team_a_production + team_b_production = 350) ∧
  (2 * team_a_production - team_b_production = 250) →
  team_a_production = 200 ∧ team_b_production = 150 := by
  sorry

end NUMINAMATH_CALUDE_zongzi_production_theorem_l3384_338471


namespace NUMINAMATH_CALUDE_students_passed_both_tests_l3384_338439

theorem students_passed_both_tests
  (total : ℕ)
  (passed_chinese : ℕ)
  (passed_english : ℕ)
  (failed_both : ℕ)
  (h1 : total = 50)
  (h2 : passed_chinese = 40)
  (h3 : passed_english = 31)
  (h4 : failed_both = 4) :
  total - failed_both = passed_chinese + passed_english - (passed_chinese + passed_english - (total - failed_both)) :=
by sorry

end NUMINAMATH_CALUDE_students_passed_both_tests_l3384_338439


namespace NUMINAMATH_CALUDE_digit_equation_solution_l3384_338461

theorem digit_equation_solution :
  ∃ (a b c d e : ℕ),
    a < 10 ∧ b < 10 ∧ c < 10 ∧ d < 10 ∧ e < 10 ∧
    (5^a) + (100*b + 10*c + 3) = (1000*d + 100*e + 1) := by
  sorry

end NUMINAMATH_CALUDE_digit_equation_solution_l3384_338461


namespace NUMINAMATH_CALUDE_intersection_of_three_lines_l3384_338498

/-- Given three lines that intersect at the same point, prove the value of k -/
theorem intersection_of_three_lines (x y : ℝ) :
  y = 4 * x + 5 ∧ 
  y = -3 * x + 10 ∧ 
  y = 2 * x + k →
  k = 45 / 7 := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_three_lines_l3384_338498


namespace NUMINAMATH_CALUDE_solution_set_quadratic_inequality_l3384_338433

theorem solution_set_quadratic_inequality :
  {x : ℝ | x^2 + x - 2 < 0} = {x : ℝ | -2 < x ∧ x < 1} := by
sorry

end NUMINAMATH_CALUDE_solution_set_quadratic_inequality_l3384_338433


namespace NUMINAMATH_CALUDE_complex_number_location_l3384_338448

theorem complex_number_location : ∃ (z : ℂ), z = (5 - 6*I) + (-2 - I) - (3 + 4*I) ∧ z = -11*I := by
  sorry

end NUMINAMATH_CALUDE_complex_number_location_l3384_338448


namespace NUMINAMATH_CALUDE_cosine_midline_l3384_338487

/-- Given a cosine function y = a cos(bx + c) + d where a, b, c, and d are positive constants,
    if the graph oscillates between 5 and 1, then d = 3 -/
theorem cosine_midline (a b c d : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) :
  (∀ x, 1 ≤ a * Real.cos (b * x + c) + d ∧ a * Real.cos (b * x + c) + d ≤ 5) →
  d = 3 := by
  sorry

end NUMINAMATH_CALUDE_cosine_midline_l3384_338487


namespace NUMINAMATH_CALUDE_inequality_solution_l3384_338465

theorem inequality_solution (x : ℝ) : 
  3 - 2 / (3 * x + 2) < 5 ↔ x < -1 ∨ x > -2/3 :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_l3384_338465


namespace NUMINAMATH_CALUDE_ellipse_focus_x_axis_l3384_338403

/-- 
Given an ellipse with equation x²/(1-k) + y²/(2+k) = 1,
if its focus lies on the x-axis, then k ∈ (-2, -1/2)
-/
theorem ellipse_focus_x_axis (k : ℝ) : 
  (∃ (x y : ℝ), x^2 / (1 - k) + y^2 / (2 + k) = 1) →
  (∃ (c : ℝ), c > 0 ∧ c^2 = (1 - k) - (2 + k)) →
  k ∈ Set.Ioo (-2 : ℝ) (-1/2 : ℝ) :=
by sorry

end NUMINAMATH_CALUDE_ellipse_focus_x_axis_l3384_338403


namespace NUMINAMATH_CALUDE_sum_to_n_432_l3384_338484

theorem sum_to_n_432 : ∃ n : ℕ, (∀ m : ℕ, m > n → (m * (m + 1)) / 2 > 432) ∧ (n * (n + 1)) / 2 ≤ 432 := by
  sorry

end NUMINAMATH_CALUDE_sum_to_n_432_l3384_338484


namespace NUMINAMATH_CALUDE_intersection_implies_a_values_l3384_338432

def A (a : ℝ) : Set ℝ := {a^2, a+1, 3}
def B (a : ℝ) : Set ℝ := {a-3, 2*a-1, a^2+1}

theorem intersection_implies_a_values :
  ∀ a : ℝ, A a ∩ B a = {3} → a = 6 ∨ a = Real.sqrt 2 ∨ a = -Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_intersection_implies_a_values_l3384_338432


namespace NUMINAMATH_CALUDE_square_sum_of_powers_l3384_338441

theorem square_sum_of_powers (a b : ℕ+) : 
  (∃ n : ℕ, 2^(a : ℕ) + 3^(b : ℕ) = n^2) ↔ a = 4 ∧ b = 2 := by
sorry

end NUMINAMATH_CALUDE_square_sum_of_powers_l3384_338441


namespace NUMINAMATH_CALUDE_min_lcm_x_z_l3384_338421

theorem min_lcm_x_z (x y z : ℕ) (h1 : Nat.lcm x y = 18) (h2 : Nat.lcm y z = 20) :
  ∃ (x' z' : ℕ), Nat.lcm x' z' = 90 ∧ ∀ (x'' z'' : ℕ), 
    Nat.lcm x'' y = 18 → Nat.lcm y z'' = 20 → Nat.lcm x'' z'' ≥ 90 := by
  sorry

end NUMINAMATH_CALUDE_min_lcm_x_z_l3384_338421


namespace NUMINAMATH_CALUDE_guests_who_stayed_l3384_338490

-- Define the initial conditions
def total_guests : ℕ := 50
def men_guests : ℕ := 15

-- Define the number of guests who left
def men_left : ℕ := men_guests / 5
def children_left : ℕ := 4

-- Theorem to prove
theorem guests_who_stayed :
  let women_guests := total_guests / 2
  let children_guests := total_guests - women_guests - men_guests
  let guests_who_stayed := total_guests - men_left - children_left
  guests_who_stayed = 43 := by sorry

end NUMINAMATH_CALUDE_guests_who_stayed_l3384_338490


namespace NUMINAMATH_CALUDE_multiply_33333_33334_l3384_338492

theorem multiply_33333_33334 : 33333 * 33334 = 1111122222 := by
  sorry

end NUMINAMATH_CALUDE_multiply_33333_33334_l3384_338492


namespace NUMINAMATH_CALUDE_c_gains_thousand_l3384_338409

/-- Represents the financial state of a person -/
structure FinancialState where
  cash : Int
  house_value : Option Int

/-- Represents a house transaction -/
inductive Transaction
  | Buy (price : Int)
  | Sell (price : Int)

def initial_c : FinancialState := { cash := 15000, house_value := some 12000 }
def initial_d : FinancialState := { cash := 16000, house_value := none }

def house_appreciation : Int := 13000
def house_depreciation : Int := 11000

def apply_transaction (state : FinancialState) (t : Transaction) : FinancialState :=
  match t with
  | Transaction.Buy price => { cash := state.cash - price, house_value := some price }
  | Transaction.Sell price => { cash := state.cash + price, house_value := none }

def net_worth (state : FinancialState) : Int :=
  state.cash + state.house_value.getD 0

theorem c_gains_thousand (c d : FinancialState → FinancialState) :
  c = (λ s => apply_transaction s (Transaction.Sell house_appreciation)) ∘
      (λ s => apply_transaction s (Transaction.Buy house_depreciation)) ∘
      (λ s => { s with house_value := some house_appreciation }) →
  d = (λ s => apply_transaction s (Transaction.Buy house_appreciation)) ∘
      (λ s => apply_transaction s (Transaction.Sell house_depreciation)) →
  net_worth (c initial_c) - net_worth initial_c = 1000 :=
sorry

end NUMINAMATH_CALUDE_c_gains_thousand_l3384_338409


namespace NUMINAMATH_CALUDE_volume_of_inscribed_sphere_l3384_338413

/-- The volume of a sphere inscribed in a cube with edge length 6 inches is 36π cubic inches. -/
theorem volume_of_inscribed_sphere (π : ℝ) : ℝ := by
  -- Define the edge length of the cube
  let cube_edge : ℝ := 6

  -- Define the radius of the inscribed sphere
  let sphere_radius : ℝ := cube_edge / 2

  -- Define the volume of the sphere
  let sphere_volume : ℝ := (4 / 3) * π * sphere_radius ^ 3

  -- Prove that the volume equals 36π
  sorry

#check volume_of_inscribed_sphere

end NUMINAMATH_CALUDE_volume_of_inscribed_sphere_l3384_338413


namespace NUMINAMATH_CALUDE_problem_solution_l3384_338473

def p₁ : Prop := ∃ x : ℝ, x^2 + x + 1 < 0

def p₂ : Prop := ∀ x : ℝ, x ∈ Set.Icc 1 2 → x^2 - 1 ≥ 0

theorem problem_solution : ¬p₁ ∧ p₂ := by sorry

end NUMINAMATH_CALUDE_problem_solution_l3384_338473


namespace NUMINAMATH_CALUDE_log_difference_cubes_l3384_338423

theorem log_difference_cubes (x y : ℝ) (a : ℝ) (h : Real.log x - Real.log y = a) :
  Real.log ((x / 2) ^ 3) - Real.log ((y / 2) ^ 3) = 3 * a := by
  sorry

end NUMINAMATH_CALUDE_log_difference_cubes_l3384_338423


namespace NUMINAMATH_CALUDE_unique_number_with_specific_remainders_l3384_338437

theorem unique_number_with_specific_remainders :
  ∃! n : ℕ, ∀ k ∈ Finset.range 11, n % (k + 2) = k + 1 := by
  sorry

end NUMINAMATH_CALUDE_unique_number_with_specific_remainders_l3384_338437


namespace NUMINAMATH_CALUDE_binomial_product_and_evaluation_l3384_338428

theorem binomial_product_and_evaluation :
  ∀ x : ℝ,
  (4 * x + 3) * (2 * x - 6) = 8 * x^2 - 18 * x - 18 ∧
  (8 * (-1)^2 - 18 * (-1) - 18) = 8 := by
  sorry

end NUMINAMATH_CALUDE_binomial_product_and_evaluation_l3384_338428


namespace NUMINAMATH_CALUDE_smallest_angle_for_complete_circle_sufficient_angle_for_complete_circle_exact_smallest_angle_for_complete_circle_l3384_338443

/-- The smallest angle needed to plot the entire circle for r = sin θ -/
theorem smallest_angle_for_complete_circle : 
  ∀ t : ℝ, (∀ θ : ℝ, 0 ≤ θ ∧ θ ≤ t → ∃ r : ℝ, r = Real.sin θ ∧ 
    (∃ x y : ℝ, x = r * Real.cos θ ∧ y = r * Real.sin θ)) →
  (∀ x y : ℝ, x^2 + y^2 ≤ 1 → 
    ∃ θ : ℝ, 0 ≤ θ ∧ θ ≤ t ∧ x = (Real.sin θ) * (Real.cos θ) ∧ y = (Real.sin θ) * (Real.sin θ)) →
  t ≥ 3 * π / 2 :=
by sorry

/-- 3π/2 is sufficient to plot the entire circle for r = sin θ -/
theorem sufficient_angle_for_complete_circle :
  ∀ x y : ℝ, x^2 + y^2 ≤ 1 → 
    ∃ θ : ℝ, 0 ≤ θ ∧ θ ≤ 3 * π / 2 ∧ x = (Real.sin θ) * (Real.cos θ) ∧ y = (Real.sin θ) * (Real.sin θ) :=
by sorry

/-- The smallest angle needed to plot the entire circle for r = sin θ is exactly 3π/2 -/
theorem exact_smallest_angle_for_complete_circle :
  (∀ t : ℝ, t < 3 * π / 2 → 
    ∃ x y : ℝ, x^2 + y^2 ≤ 1 ∧ 
      ∀ θ : ℝ, 0 ≤ θ ∧ θ ≤ t → x ≠ (Real.sin θ) * (Real.cos θ) ∨ y ≠ (Real.sin θ) * (Real.sin θ)) ∧
  (∀ x y : ℝ, x^2 + y^2 ≤ 1 → 
    ∃ θ : ℝ, 0 ≤ θ ∧ θ ≤ 3 * π / 2 ∧ x = (Real.sin θ) * (Real.cos θ) ∧ y = (Real.sin θ) * (Real.sin θ)) :=
by sorry

end NUMINAMATH_CALUDE_smallest_angle_for_complete_circle_sufficient_angle_for_complete_circle_exact_smallest_angle_for_complete_circle_l3384_338443


namespace NUMINAMATH_CALUDE_balloon_purchase_theorem_l3384_338430

/-- The price of a regular balloon -/
def regular_price : ℚ := 1

/-- The price of a discounted balloon -/
def discounted_price : ℚ := 1/2

/-- The total budget available -/
def budget : ℚ := 30

/-- The cost of a set of three balloons -/
def set_cost : ℚ := 2 * regular_price + discounted_price

/-- The number of balloons in a set -/
def balloons_per_set : ℕ := 3

/-- The maximum number of balloons that can be purchased -/
def max_balloons : ℕ := 36

theorem balloon_purchase_theorem : 
  (budget / set_cost : ℚ).floor * balloons_per_set = max_balloons :=
sorry

end NUMINAMATH_CALUDE_balloon_purchase_theorem_l3384_338430


namespace NUMINAMATH_CALUDE_number_puzzle_l3384_338474

theorem number_puzzle (x : ℝ) : (1/2 : ℝ) * x + (1/3 : ℝ) * x = (1/4 : ℝ) * x + 7 → x = 12 := by
  sorry

end NUMINAMATH_CALUDE_number_puzzle_l3384_338474


namespace NUMINAMATH_CALUDE_runner_lap_time_l3384_338499

/-- Given two runners on a circular track, prove that if one runner completes 9 laps
    in the time the other completes 8, and the first runner's lap time is 40 seconds,
    then the second runner's lap time is 45 seconds. -/
theorem runner_lap_time (M D : ℕ) (h1 : M = 40) (h2 : 9 * M = 8 * D) : D = 45 := by
  sorry

end NUMINAMATH_CALUDE_runner_lap_time_l3384_338499


namespace NUMINAMATH_CALUDE_line_intercepts_minimum_sum_l3384_338445

theorem line_intercepts_minimum_sum (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h_point : a + b = a * b) : 
  (a / b + b / a) ≥ 4 ∧ ∃ (a₀ b₀ : ℝ), a₀ > 0 ∧ b₀ > 0 ∧ a₀ + b₀ = a₀ * b₀ ∧ a₀ / b₀ + b₀ / a₀ = 4 :=
by sorry

end NUMINAMATH_CALUDE_line_intercepts_minimum_sum_l3384_338445


namespace NUMINAMATH_CALUDE_sugar_recipe_problem_l3384_338401

/-- The number of recipes that can be accommodated given a certain amount of sugar and recipe requirement -/
def recipes_accommodated (total_sugar : ℚ) (sugar_per_recipe : ℚ) : ℚ :=
  total_sugar / sugar_per_recipe

/-- The problem statement -/
theorem sugar_recipe_problem :
  let total_sugar : ℚ := 56 / 3  -- 18⅔ cups
  let sugar_per_recipe : ℚ := 3 / 2  -- 1½ cups
  recipes_accommodated total_sugar sugar_per_recipe = 112 / 9 :=
by
  sorry

#eval (112 : ℚ) / 9  -- Should output 12⁴⁄₉

end NUMINAMATH_CALUDE_sugar_recipe_problem_l3384_338401


namespace NUMINAMATH_CALUDE_partial_fraction_decomposition_l3384_338435

theorem partial_fraction_decomposition :
  ∀ x : ℝ, x ≠ 6 ∧ x ≠ -3 →
  (4 * x + 7) / (x^2 - 3*x - 18) = (31/9) / (x - 6) + (5/9) / (x + 3) := by
sorry

end NUMINAMATH_CALUDE_partial_fraction_decomposition_l3384_338435


namespace NUMINAMATH_CALUDE_leastDivisorTheorem_l3384_338479

/-- The least positive integer that divides 16800 to get a number that is both a perfect square and a perfect cube -/
def leastDivisor : ℕ := 8400

/-- 16800 divided by the least divisor is a perfect square -/
def isPerfectSquare : Prop :=
  ∃ m : ℕ, (16800 / leastDivisor) = m * m

/-- 16800 divided by the least divisor is a perfect cube -/
def isPerfectCube : Prop :=
  ∃ m : ℕ, (16800 / leastDivisor) = m * m * m

/-- The main theorem stating that leastDivisor is the smallest positive integer
    that divides 16800 to get both a perfect square and a perfect cube -/
theorem leastDivisorTheorem :
  isPerfectSquare ∧ isPerfectCube ∧
  ∀ n : ℕ, 0 < n ∧ n < leastDivisor →
    ¬(∃ m : ℕ, (16800 / n) = m * m) ∨ ¬(∃ m : ℕ, (16800 / n) = m * m * m) :=
sorry

end NUMINAMATH_CALUDE_leastDivisorTheorem_l3384_338479


namespace NUMINAMATH_CALUDE_ball_radius_from_hole_dimensions_l3384_338467

/-- 
Given a spherical ball that leaves a circular hole when removed from a frozen surface,
this theorem proves that if the hole has a diameter of 30 cm and a depth of 10 cm,
then the radius of the ball is 16.25 cm.
-/
theorem ball_radius_from_hole_dimensions (hole_diameter : ℝ) (hole_depth : ℝ) 
    (h_diameter : hole_diameter = 30) 
    (h_depth : hole_depth = 10) : 
    ∃ (ball_radius : ℝ), ball_radius = 16.25 := by
  sorry

end NUMINAMATH_CALUDE_ball_radius_from_hole_dimensions_l3384_338467


namespace NUMINAMATH_CALUDE_symmetry_condition_l3384_338400

def f (x a : ℝ) : ℝ := |x + 1| + |x - 1| + |x - a|

theorem symmetry_condition (a : ℝ) : 
  (∃ k : ℝ, ∀ x : ℝ, f x a = f (2*k - x) a) ↔ a ∈ ({-3, 0, 3} : Set ℝ) := by
sorry

end NUMINAMATH_CALUDE_symmetry_condition_l3384_338400


namespace NUMINAMATH_CALUDE_complex_polynomial_roots_l3384_338464

theorem complex_polynomial_roots (c : ℂ) : 
  (∃ (P : ℂ → ℂ), P = (fun x ↦ (x^2 - 2*x + 2) * (x^2 - c*x + 4) * (x^2 - 4*x + 8)) ∧ 
   (∃ (r1 r2 r3 r4 : ℂ), r1 ≠ r2 ∧ r1 ≠ r3 ∧ r1 ≠ r4 ∧ r2 ≠ r3 ∧ r2 ≠ r4 ∧ r3 ≠ r4 ∧
    ∀ x, P x = 0 ↔ x = r1 ∨ x = r2 ∨ x = r3 ∨ x = r4)) →
  Complex.abs c = Real.sqrt 10 := by
sorry

end NUMINAMATH_CALUDE_complex_polynomial_roots_l3384_338464


namespace NUMINAMATH_CALUDE_old_supervisor_salary_proof_l3384_338483

/-- Calculates the old supervisor's salary given the initial and new average salaries,
    number of workers, and new supervisor's salary. -/
def old_supervisor_salary (initial_avg : ℚ) (new_avg : ℚ) (num_workers : ℕ) 
  (new_supervisor_salary : ℚ) : ℚ :=
  (initial_avg * (num_workers + 1) - new_avg * (num_workers + 1) + new_supervisor_salary)

/-- Proves that the old supervisor's salary was $870 given the problem conditions. -/
theorem old_supervisor_salary_proof :
  old_supervisor_salary 430 440 8 960 = 870 := by
  sorry

#eval old_supervisor_salary 430 440 8 960

end NUMINAMATH_CALUDE_old_supervisor_salary_proof_l3384_338483


namespace NUMINAMATH_CALUDE_f_properties_l3384_338405

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * Real.exp (2 * x) + (a - 2) * Real.exp x - x

theorem f_properties (a : ℝ) :
  (∀ x, a ≤ 0 → (deriv (f a)) x < 0) ∧
  (a > 0 → ∀ x, x < -Real.log a → (deriv (f a)) x < 0) ∧
  (a > 0 → ∀ x, x > -Real.log a → (deriv (f a)) x > 0) ∧
  (a ≥ 1 → ∀ x, f a x ≥ 0) :=
by sorry

end NUMINAMATH_CALUDE_f_properties_l3384_338405


namespace NUMINAMATH_CALUDE_trigonometric_identities_l3384_338489

theorem trigonometric_identities (α : ℝ) 
  (h : (3 * Real.sin α - Real.cos α) / (2 * Real.sin α + 3 * Real.cos α) = 5 / 7) :
  (Real.tan (π / 2 - α) = 1 / 2) ∧
  (3 * Real.cos α * Real.sin (α + π) + 2 * (Real.cos (α + π / 2))^2 = 2 / 5) := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_identities_l3384_338489


namespace NUMINAMATH_CALUDE_A_intersection_B_equals_A_l3384_338406

-- Define the function f
def f (x : ℝ) : ℝ := x^2

-- Define set A
def A : Set ℝ := {x | f x = x}

-- Define set B
def B : Set ℝ := {x | f (f x) = x}

-- Theorem statement
theorem A_intersection_B_equals_A : A ∩ B = A := by sorry

end NUMINAMATH_CALUDE_A_intersection_B_equals_A_l3384_338406


namespace NUMINAMATH_CALUDE_brian_stones_l3384_338475

theorem brian_stones (total : ℕ) (white black grey green : ℕ) : 
  total = 100 → 
  white + black = total → 
  grey = 40 → 
  green = 60 → 
  white * green = black * grey → 
  white > black → 
  white = 60 := by sorry

end NUMINAMATH_CALUDE_brian_stones_l3384_338475


namespace NUMINAMATH_CALUDE_range_of_a_minus_abs_b_l3384_338418

-- Define the conditions
def a_condition (a : ℝ) : Prop := 1 < a ∧ a < 3
def b_condition (b : ℝ) : Prop := -4 < b ∧ b < 2

-- Define the range of a - |b|
def range_a_minus_abs_b (x : ℝ) : Prop :=
  ∃ (a b : ℝ), a_condition a ∧ b_condition b ∧ x = a - |b|

-- Theorem statement
theorem range_of_a_minus_abs_b :
  ∀ x, range_a_minus_abs_b x ↔ -3 < x ∧ x < 3 :=
sorry

end NUMINAMATH_CALUDE_range_of_a_minus_abs_b_l3384_338418


namespace NUMINAMATH_CALUDE_quadratic_root_l3384_338481

theorem quadratic_root (a b c : ℝ) (h1 : 4 * a - 2 * b + c = 0) (h2 : a ≠ 0) :
  a * (-2)^2 + b * (-2) + c = 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_root_l3384_338481


namespace NUMINAMATH_CALUDE_total_money_made_l3384_338452

/-- The total money made from selling items is the sum of the products of price and quantity for each item. -/
theorem total_money_made 
  (smoothie_price cake_price : ℚ) 
  (smoothie_quantity cake_quantity : ℕ) :
  smoothie_price * smoothie_quantity + cake_price * cake_quantity =
  (smoothie_price * smoothie_quantity + cake_price * cake_quantity : ℚ) :=
by sorry

/-- Scott's total earnings from selling smoothies and cakes -/
def scotts_earnings : ℚ :=
  let smoothie_price : ℚ := 3
  let cake_price : ℚ := 2
  let smoothie_quantity : ℕ := 40
  let cake_quantity : ℕ := 18
  smoothie_price * smoothie_quantity + cake_price * cake_quantity

#eval scotts_earnings -- Expected output: 156

end NUMINAMATH_CALUDE_total_money_made_l3384_338452


namespace NUMINAMATH_CALUDE_economic_indicator_p_l3384_338454

/-- Given the economic indicators equation and values, prove the value of p -/
theorem economic_indicator_p (f w p : ℂ) : 
  f * p - w = 15000 ∧ f = 8 ∧ w = 10 + 200 * Complex.I → p = 1876.25 + 25 * Complex.I := by
  sorry

end NUMINAMATH_CALUDE_economic_indicator_p_l3384_338454


namespace NUMINAMATH_CALUDE_triangle_ABC_properties_l3384_338480

-- Define the triangle ABC
def A : ℝ × ℝ := (4, 0)
def B : ℝ × ℝ := (8, 10)
def C : ℝ × ℝ := (0, 6)

-- Define the line parallel to BC passing through A
def line_parallel_BC (x y : ℝ) : Prop :=
  x + 2*y - 8 = 0

-- Define the lines passing through B equidistant from A and C
def line_equidistant_1 (x y : ℝ) : Prop :=
  3*x - 2*y - 4 = 0

def line_equidistant_2 (x y : ℝ) : Prop :=
  3*x + 2*y - 44 = 0

theorem triangle_ABC_properties :
  (∀ x y, line_parallel_BC x y ↔ (x - A.1) * (C.2 - B.2) = (y - A.2) * (C.1 - B.1)) ∧
  (∀ x y, (line_equidistant_1 x y ∨ line_equidistant_2 x y) ↔
    ((x - A.1)^2 + (y - A.2)^2 = (x - C.1)^2 + (y - C.2)^2 ∧ x = B.1 ∧ y = B.2)) :=
by sorry

end NUMINAMATH_CALUDE_triangle_ABC_properties_l3384_338480


namespace NUMINAMATH_CALUDE_intimate_functions_range_l3384_338420

theorem intimate_functions_range (f g : ℝ → ℝ) (m : ℝ) :
  (∀ x ∈ Set.Icc 2 3, f x = x^3 - 2*x + 7) →
  (∀ x ∈ Set.Icc 2 3, g x = x + m) →
  (∀ x ∈ Set.Icc 2 3, |f x - g x| ≤ 10) →
  15 ≤ m ∧ m ≤ 19 := by
sorry

end NUMINAMATH_CALUDE_intimate_functions_range_l3384_338420


namespace NUMINAMATH_CALUDE_nine_gon_diagonals_l3384_338444

/-- The number of diagonals in a regular nine-sided polygon -/
def num_diagonals_nine_gon : ℕ :=
  (9 * (9 - 1)) / 2 - 9

theorem nine_gon_diagonals :
  num_diagonals_nine_gon = 27 := by
  sorry

end NUMINAMATH_CALUDE_nine_gon_diagonals_l3384_338444


namespace NUMINAMATH_CALUDE_bank_interest_rate_problem_l3384_338417

theorem bank_interest_rate_problem
  (initial_investment : ℝ)
  (additional_investment : ℝ)
  (additional_rate : ℝ)
  (total_rate : ℝ)
  (h1 : initial_investment = 2200)
  (h2 : additional_investment = 1099.9999999999998)
  (h3 : additional_rate = 0.08)
  (h4 : total_rate = 0.06)
  (h5 : total_rate * (initial_investment + additional_investment) =
        initial_investment * x + additional_investment * additional_rate) :
  x = 0.05 :=
by sorry

end NUMINAMATH_CALUDE_bank_interest_rate_problem_l3384_338417


namespace NUMINAMATH_CALUDE_sum_of_segments_equals_radius_l3384_338477

/-- A regular (4k+2)-gon inscribed in a circle -/
structure RegularPolygon (k : ℕ) :=
  (center : ℝ × ℝ)
  (radius : ℝ)
  (vertices : Fin (4*k+2) → ℝ × ℝ)

/-- The segments cut by angle A₍ₖ₎OA₍ₖ₊₁₎ on the lines A₁A₍₂ₖ₎, A₂A₍₂ₖ₋₁₎, ..., A₍ₖ₎A₍ₖ₊₁₎ -/
def cut_segments (p : RegularPolygon k) : List (ℝ × ℝ) :=
  sorry

/-- The sum of the lengths of the cut segments -/
def sum_of_segment_lengths (segments : List (ℝ × ℝ)) : ℝ :=
  sorry

/-- Theorem: The sum of the lengths of the cut segments is equal to the radius -/
theorem sum_of_segments_equals_radius (k : ℕ) (p : RegularPolygon k) :
  sum_of_segment_lengths (cut_segments p) = p.radius :=
sorry

end NUMINAMATH_CALUDE_sum_of_segments_equals_radius_l3384_338477


namespace NUMINAMATH_CALUDE_infinitely_many_solutions_l3384_338431

theorem infinitely_many_solutions (d : ℝ) : 
  (∀ x : ℝ, 3 * (5 + d * x) = 15 * x + 15) ↔ d = 5 :=
sorry

end NUMINAMATH_CALUDE_infinitely_many_solutions_l3384_338431


namespace NUMINAMATH_CALUDE_additional_apples_needed_l3384_338466

def apples_needed (pies : ℕ) (apples_per_pie : ℕ) (available_apples : ℕ) : ℕ :=
  pies * apples_per_pie - available_apples

theorem additional_apples_needed : 
  apples_needed 10 8 50 = 30 := by
  sorry

end NUMINAMATH_CALUDE_additional_apples_needed_l3384_338466


namespace NUMINAMATH_CALUDE_jump_ratio_l3384_338429

def hattie_first_round : ℕ := 180
def lorelei_first_round : ℕ := (3 * hattie_first_round) / 4
def total_jumps : ℕ := 605

def hattie_second_round : ℕ := (total_jumps - hattie_first_round - lorelei_first_round - 50) / 2

theorem jump_ratio : 
  (hattie_second_round : ℚ) / hattie_first_round = 2 / 3 := by sorry

end NUMINAMATH_CALUDE_jump_ratio_l3384_338429


namespace NUMINAMATH_CALUDE_sqrt_sum_equation_solutions_l3384_338425

theorem sqrt_sum_equation_solutions :
  ∀ x : ℝ, (Real.sqrt ((3 + 2 * Real.sqrt 2) ^ x) + Real.sqrt ((3 - 2 * Real.sqrt 2) ^ x) = 6) ↔ (x = 2 ∨ x = -2) := by
  sorry

end NUMINAMATH_CALUDE_sqrt_sum_equation_solutions_l3384_338425


namespace NUMINAMATH_CALUDE_frequency_of_six_is_nineteen_hundredths_l3384_338449

/-- Represents the outcome of rolling a fair six-sided die multiple times -/
structure DieRollOutcome where
  total_rolls : ℕ
  sixes_count : ℕ

/-- Calculates the frequency of rolling a 6 -/
def frequency_of_six (outcome : DieRollOutcome) : ℚ :=
  outcome.sixes_count / outcome.total_rolls

/-- Theorem stating that for the given die roll outcome, the frequency of rolling a 6 is 0.19 -/
theorem frequency_of_six_is_nineteen_hundredths 
  (outcome : DieRollOutcome) 
  (h1 : outcome.total_rolls = 100) 
  (h2 : outcome.sixes_count = 19) : 
  frequency_of_six outcome = 19 / 100 := by
  sorry

end NUMINAMATH_CALUDE_frequency_of_six_is_nineteen_hundredths_l3384_338449


namespace NUMINAMATH_CALUDE_trigonometric_equality_l3384_338419

theorem trigonometric_equality (α β : ℝ) 
  (h : (Real.cos α)^6 / (Real.cos β)^3 + (Real.sin α)^6 / (Real.sin β)^3 = 1) :
  (Real.sin β)^6 / (Real.sin α)^3 + (Real.cos β)^6 / (Real.cos α)^3 = 1 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_equality_l3384_338419


namespace NUMINAMATH_CALUDE_subtracted_value_l3384_338422

theorem subtracted_value (N V : ℝ) (h1 : N = 740) (h2 : N / 4 - V = 10) : V = 175 := by
  sorry

end NUMINAMATH_CALUDE_subtracted_value_l3384_338422


namespace NUMINAMATH_CALUDE_k_at_one_l3384_338414

-- Define the polynomials h and k
def h (p : ℝ) (x : ℝ) : ℝ := x^3 + p*x^2 + 2*x + 20
def k (q r : ℝ) (x : ℝ) : ℝ := x^4 + 2*x^3 + q*x^2 + 50*x + r

-- State the theorem
theorem k_at_one (p q r : ℝ) : 
  (∃ x y z : ℝ, x ≠ y ∧ y ≠ z ∧ x ≠ z ∧ 
    h p x = 0 ∧ h p y = 0 ∧ h p z = 0 ∧
    k q r x = 0 ∧ k q r y = 0 ∧ k q r z = 0) →
  k q r 1 = -155 := by
sorry

end NUMINAMATH_CALUDE_k_at_one_l3384_338414


namespace NUMINAMATH_CALUDE_unique_solution_fifth_power_equation_l3384_338446

theorem unique_solution_fifth_power_equation :
  ∃! (x : ℝ), x ≠ 0 ∧ (5 * x)^10 = (10 * x)^5 :=
by
  -- The unique solution is x = 2/5
  use 2/5
  sorry

end NUMINAMATH_CALUDE_unique_solution_fifth_power_equation_l3384_338446


namespace NUMINAMATH_CALUDE_cube_volume_ratio_l3384_338462

theorem cube_volume_ratio (edge_q : ℝ) (edge_p : ℝ) (h : edge_p = 3 * edge_q) :
  (edge_q ^ 3) / (edge_p ^ 3) = 1 / 27 := by
sorry

end NUMINAMATH_CALUDE_cube_volume_ratio_l3384_338462


namespace NUMINAMATH_CALUDE_complex_pure_imaginary_m_l3384_338468

def is_pure_imaginary (z : ℂ) : Prop := z.re = 0 ∧ z.im ≠ 0

theorem complex_pure_imaginary_m (m : ℝ) :
  let z : ℂ := Complex.mk (m^2 - 4) (m + 2)
  is_pure_imaginary z → m = 2 :=
by sorry

end NUMINAMATH_CALUDE_complex_pure_imaginary_m_l3384_338468


namespace NUMINAMATH_CALUDE_parametric_curve_length_l3384_338415

/-- The parametric curve described by (x,y) = (3 sin t, 3 cos t) for t ∈ [0, 2π] -/
def parametric_curve : Set (ℝ × ℝ) :=
  {p | ∃ t ∈ Set.Icc 0 (2 * Real.pi), p = (3 * Real.sin t, 3 * Real.cos t)}

/-- The length of a curve -/
noncomputable def curve_length (c : Set (ℝ × ℝ)) : ℝ := sorry

theorem parametric_curve_length :
  curve_length parametric_curve = 6 * Real.pi := by sorry

end NUMINAMATH_CALUDE_parametric_curve_length_l3384_338415


namespace NUMINAMATH_CALUDE_fish_pond_population_l3384_338485

theorem fish_pond_population (initial_tagged : ℕ) (second_catch : ℕ) (tagged_in_second : ℕ) :
  initial_tagged = 80 →
  second_catch = 80 →
  tagged_in_second = 2 →
  (tagged_in_second : ℚ) / second_catch = initial_tagged / (initial_tagged * second_catch / tagged_in_second) :=
by
  sorry

#check fish_pond_population

end NUMINAMATH_CALUDE_fish_pond_population_l3384_338485


namespace NUMINAMATH_CALUDE_D_72_l3384_338458

/-- D(n) is the number of ways to express n as a product of integers greater than 1, considering order as distinct -/
def D (n : ℕ) : ℕ := sorry

/-- The prime factorization of 72 -/
def prime_factorization_72 : List (ℕ × ℕ) := [(2, 3), (3, 2)]

theorem D_72 : D 72 = 22 := by sorry

end NUMINAMATH_CALUDE_D_72_l3384_338458


namespace NUMINAMATH_CALUDE_toy_cost_price_l3384_338451

def toy_problem (num_toys : ℕ) (selling_price : ℚ) (gain_ratio : ℕ) :=
  (num_toys : ℚ) * (selling_price / num_toys) / (num_toys + gain_ratio : ℚ)

theorem toy_cost_price :
  toy_problem 25 62500 5 = 2083 + 1/3 :=
by sorry

end NUMINAMATH_CALUDE_toy_cost_price_l3384_338451


namespace NUMINAMATH_CALUDE_parabola_tangent_intersection_l3384_338469

/-- Parabola defined by x^2 = 2y -/
def parabola (x y : ℝ) : Prop := x^2 = 2*y

/-- Tangent line to the parabola at a given point (a, a^2/2) -/
def tangent_line (a x y : ℝ) : Prop := y - (a^2/2) = a*(x - a)

/-- Point of intersection of two lines -/
def intersection (m₁ b₁ m₂ b₂ x y : ℝ) : Prop :=
  y = m₁*x + b₁ ∧ y = m₂*x + b₂

theorem parabola_tangent_intersection :
  ∃ (x y : ℝ),
    intersection 4 (-8) (-2) (-2) x y ∧
    y = -4 :=
sorry

end NUMINAMATH_CALUDE_parabola_tangent_intersection_l3384_338469


namespace NUMINAMATH_CALUDE_wengs_hourly_rate_l3384_338426

/-- Weng's hourly rate given her earnings and work duration --/
theorem wengs_hourly_rate (work_duration : ℚ) (earnings : ℚ) : 
  work_duration = 50 / 60 → earnings = 10 → (earnings / work_duration) = 12 := by
  sorry

end NUMINAMATH_CALUDE_wengs_hourly_rate_l3384_338426


namespace NUMINAMATH_CALUDE_valid_numbers_l3384_338495

def is_valid_number (n : ℕ) : Prop :=
  ∃ (k : ℕ) (a : Fin 10) (m n : ℕ),
    n = m + 10^k * a.val + 10^(k+1) * n ∧
    m < 10^k ∧
    m + 10^k * n = (m + 10^k * a.val + 10^(k+1) * n) / 6 ∧
    n % 10 ≠ 0

theorem valid_numbers :
  {n : ℕ | is_valid_number n} = {12, 24, 36, 48, 108} :=
sorry

end NUMINAMATH_CALUDE_valid_numbers_l3384_338495


namespace NUMINAMATH_CALUDE_at_most_one_equal_area_point_l3384_338472

/-- A point in a 2D plane -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- A convex quadrilateral in a 2D plane -/
structure ConvexQuadrilateral where
  A : Point2D
  B : Point2D
  C : Point2D
  D : Point2D
  convex : Bool  -- Assumption that the quadrilateral is convex

/-- Calculate the area of a triangle given its three vertices -/
def triangleArea (p1 p2 p3 : Point2D) : ℝ :=
  sorry

/-- Check if four triangles have equal areas -/
def equalAreaTriangles (p : Point2D) (quad : ConvexQuadrilateral) : Prop :=
  let areaABP := triangleArea quad.A quad.B p
  let areaBCP := triangleArea quad.B quad.C p
  let areaCDP := triangleArea quad.C quad.D p
  let areaDPA := triangleArea quad.D quad.A p
  areaABP = areaBCP ∧ areaBCP = areaCDP ∧ areaCDP = areaDPA

/-- Main theorem: There exists at most one point P that satisfies the equal area condition -/
theorem at_most_one_equal_area_point (quad : ConvexQuadrilateral) :
  ∃! p : Point2D, equalAreaTriangles p quad :=
sorry

end NUMINAMATH_CALUDE_at_most_one_equal_area_point_l3384_338472


namespace NUMINAMATH_CALUDE_intersection_trig_functions_l3384_338482

theorem intersection_trig_functions (x : ℝ) (h1 : 0 < x) (h2 : x < π / 2) 
  (h3 : 6 * Real.cos x = 9 * Real.tan x) : Real.sin x = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_intersection_trig_functions_l3384_338482


namespace NUMINAMATH_CALUDE_box_filled_with_large_cubes_l3384_338436

/-- Represents the dimensions of a rectangular box -/
structure BoxDimensions where
  length : ℕ
  width : ℕ
  depth : ℕ

/-- Represents a cube with a given side length -/
structure Cube where
  sideLength : ℕ

/-- Calculates the volume of a box given its dimensions -/
def boxVolume (b : BoxDimensions) : ℕ :=
  b.length * b.width * b.depth

/-- Calculates the volume of a cube given its side length -/
def cubeVolume (c : Cube) : ℕ :=
  c.sideLength * c.sideLength * c.sideLength

/-- Theorem: A box with dimensions 50 × 60 × 43 inches can be filled completely with 1032 cubes of size 5 × 5 × 5 inches -/
theorem box_filled_with_large_cubes :
  let box := BoxDimensions.mk 50 60 43
  let largeCube := Cube.mk 5
  boxVolume box = 1032 * cubeVolume largeCube := by
  sorry


end NUMINAMATH_CALUDE_box_filled_with_large_cubes_l3384_338436


namespace NUMINAMATH_CALUDE_two_correct_statements_l3384_338455

theorem two_correct_statements :
  (∀ x > 0, x > Real.sin x) ∧
  (¬(∀ x > 0, x - Real.log x > 0) ↔ ∃ x₀ > 0, x₀ - Real.log x₀ ≤ 0) ∧
  ¬((∀ p q : Prop, p ∨ q → p ∧ q) ∧ ¬(∀ p q : Prop, p ∧ q → p ∨ q)) := by
  sorry

end NUMINAMATH_CALUDE_two_correct_statements_l3384_338455


namespace NUMINAMATH_CALUDE_sum_remainder_mod_nine_l3384_338440

theorem sum_remainder_mod_nine : (8357 + 8358 + 8359 + 8360 + 8361) % 9 = 3 := by
  sorry

end NUMINAMATH_CALUDE_sum_remainder_mod_nine_l3384_338440


namespace NUMINAMATH_CALUDE_min_difference_f_g_l3384_338407

noncomputable def f (x : ℝ) := Real.exp x

noncomputable def g (x : ℝ) := Real.log (x / 2) + 1 / 2

theorem min_difference_f_g :
  ∀ a : ℝ, ∃ b : ℝ, b > 0 ∧ f a = g b ∧
  (∀ c : ℝ, c > 0 ∧ f a = g c → b - a ≤ c - a) ∧
  b - a = 2 + Real.log 2 :=
sorry

end NUMINAMATH_CALUDE_min_difference_f_g_l3384_338407


namespace NUMINAMATH_CALUDE_accessory_time_is_ten_l3384_338410

/-- Represents the production details of a doll factory --/
structure DollFactory where
  num_dolls : ℕ
  time_per_doll : ℕ
  total_time : ℕ
  shoes_per_doll : ℕ
  bags_per_doll : ℕ
  cosmetics_per_doll : ℕ
  hats_per_doll : ℕ

/-- Calculates the time taken to make each accessory --/
def time_per_accessory (factory : DollFactory) : ℕ :=
  let total_accessories := factory.num_dolls * (factory.shoes_per_doll + factory.bags_per_doll + 
                           factory.cosmetics_per_doll + factory.hats_per_doll)
  let time_for_dolls := factory.num_dolls * factory.time_per_doll
  let time_for_accessories := factory.total_time - time_for_dolls
  time_for_accessories / total_accessories

/-- Theorem stating that the time to make each accessory is 10 seconds --/
theorem accessory_time_is_ten (factory : DollFactory) 
  (h1 : factory.num_dolls = 12000)
  (h2 : factory.time_per_doll = 45)
  (h3 : factory.total_time = 1860000)
  (h4 : factory.shoes_per_doll = 2)
  (h5 : factory.bags_per_doll = 3)
  (h6 : factory.cosmetics_per_doll = 1)
  (h7 : factory.hats_per_doll = 5) :
  time_per_accessory factory = 10 := by
  sorry

#eval time_per_accessory { 
  num_dolls := 12000, 
  time_per_doll := 45, 
  total_time := 1860000, 
  shoes_per_doll := 2, 
  bags_per_doll := 3, 
  cosmetics_per_doll := 1, 
  hats_per_doll := 5 
}

end NUMINAMATH_CALUDE_accessory_time_is_ten_l3384_338410


namespace NUMINAMATH_CALUDE_bells_toll_together_l3384_338404

theorem bells_toll_together (bell1 bell2 bell3 bell4 : ℕ) 
  (h1 : bell1 = 9) (h2 : bell2 = 10) (h3 : bell3 = 14) (h4 : bell4 = 18) :
  Nat.lcm bell1 (Nat.lcm bell2 (Nat.lcm bell3 bell4)) = 630 := by
  sorry

end NUMINAMATH_CALUDE_bells_toll_together_l3384_338404


namespace NUMINAMATH_CALUDE_total_sequences_value_l3384_338459

/-- The number of students in the first class -/
def students_class1 : ℕ := 12

/-- The number of students in the second class -/
def students_class2 : ℕ := 13

/-- The number of meetings per week for each class -/
def meetings_per_week : ℕ := 3

/-- The total number of different sequences of students solving problems for both classes in a week -/
def total_sequences : ℕ := (students_class1 * students_class2) ^ meetings_per_week

theorem total_sequences_value : total_sequences = 3796416 := by sorry

end NUMINAMATH_CALUDE_total_sequences_value_l3384_338459


namespace NUMINAMATH_CALUDE_parabola_properties_l3384_338488

/-- A parabola with the given properties -/
structure Parabola where
  a : ℝ
  b : ℝ
  intersectsXAxis : (a * (-1)^2 + b * (-1) + 2 = 0) ∧ (a ≠ 0)
  distanceAB : ∃ x, x ≠ -1 ∧ a * x^2 + b * x + 2 = 0 ∧ |x - (-1)| = 3
  increasingAfterA : ∀ x > -1, ∀ y > -1, 
    a * x^2 + b * x + 2 > a * y^2 + b * y + 2 → x > y

/-- The axis of symmetry and point P for the parabola -/
theorem parabola_properties (p : Parabola) :
  (∃ x, x = -(p.a + 2) / (2 * p.a) ∧ 
    ∀ y, p.a * (x + y)^2 + p.b * (x + y) + 2 = p.a * (x - y)^2 + p.b * (x - y) + 2) ∧
  (∃ x y, (x = -3 ∨ x = -2) ∧ y = -1 ∧ 
    p.a * x^2 + p.b * x + 2 = y ∧ 
    y < 0 ∧
    ∃ xB yC, p.a * xB^2 + p.b * xB + 2 = 0 ∧ yC = 2 ∧
    2 * (yC - y) = xB - x) := by
  sorry

end NUMINAMATH_CALUDE_parabola_properties_l3384_338488


namespace NUMINAMATH_CALUDE_apple_pear_worth_l3384_338494

-- Define the worth of apples in terms of pears
def apple_worth (x : ℚ) : Prop := (3/4 : ℚ) * 16 * x = 6

-- Theorem to prove
theorem apple_pear_worth (x : ℚ) (h : apple_worth x) : (1/3 : ℚ) * 9 * x = (3/2 : ℚ) :=
by sorry

end NUMINAMATH_CALUDE_apple_pear_worth_l3384_338494


namespace NUMINAMATH_CALUDE_dartboard_region_angle_l3384_338470

def circular_dartboard (total_area : ℝ) : Prop := total_area > 0

def region_probability (prob : ℝ) : Prop := prob = 1 / 8

def central_angle (angle : ℝ) : Prop := 
  0 ≤ angle ∧ angle ≤ 360

theorem dartboard_region_angle 
  (total_area : ℝ) 
  (prob : ℝ) 
  (angle : ℝ) :
  circular_dartboard total_area →
  region_probability prob →
  central_angle angle →
  prob = angle / 360 →
  angle = 45 := by sorry

end NUMINAMATH_CALUDE_dartboard_region_angle_l3384_338470


namespace NUMINAMATH_CALUDE_fish_count_after_21_days_l3384_338478

/-- Represents the state of the aquarium --/
structure AquariumState where
  days : ℕ
  fish : ℕ

/-- Calculates the number of fish eaten in a given number of days --/
def fishEaten (days : ℕ) : ℕ :=
  (2 + 3) * days

/-- Calculates the number of fish born in a given number of days --/
def fishBorn (days : ℕ) : ℕ :=
  2 * (days / 3)

/-- Updates the aquarium state for a given number of days --/
def updateAquarium (state : AquariumState) (days : ℕ) : AquariumState :=
  let newFish := max 0 (state.fish - fishEaten days + fishBorn days)
  { days := state.days + days, fish := newFish }

/-- Adds a specified number of fish to the aquarium --/
def addFish (state : AquariumState) (amount : ℕ) : AquariumState :=
  { state with fish := state.fish + amount }

/-- The final state of the aquarium after 21 days --/
def finalState : AquariumState :=
  let initialState : AquariumState := { days := 0, fish := 60 }
  let afterTwoWeeks := updateAquarium initialState 14
  let withAddedFish := addFish afterTwoWeeks 8
  updateAquarium withAddedFish 7

/-- The theorem stating that the number of fish after 21 days is 4 --/
theorem fish_count_after_21_days :
  finalState.fish = 4 := by sorry

end NUMINAMATH_CALUDE_fish_count_after_21_days_l3384_338478


namespace NUMINAMATH_CALUDE_inequality_proof_l3384_338497

theorem inequality_proof (a b x : ℝ) (ha : a > 0) (hb : b > 0) :
  a * b ≤ (a * Real.sin x ^ 2 + b * Real.cos x ^ 2) * (a * Real.cos x ^ 2 + b * Real.sin x ^ 2) ∧
  (a * Real.sin x ^ 2 + b * Real.cos x ^ 2) * (a * Real.cos x ^ 2 + b * Real.sin x ^ 2) ≤ (a + b) ^ 2 / 4 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3384_338497


namespace NUMINAMATH_CALUDE_larger_number_given_hcf_lcm_factors_l3384_338424

theorem larger_number_given_hcf_lcm_factors (a b : ℕ) : 
  a > 0 ∧ b > 0 ∧ 
  Nat.gcd a b = 63 ∧ 
  ∃ (k : ℕ), Nat.lcm a b = 63 * 11 * 17 * k →
  max a b = 1071 := by
  sorry

end NUMINAMATH_CALUDE_larger_number_given_hcf_lcm_factors_l3384_338424


namespace NUMINAMATH_CALUDE_arithmetic_sequence_middle_term_l3384_338486

theorem arithmetic_sequence_middle_term (a b c : ℤ) : 
  (a = 3^2 ∧ c = 3^4 ∧ b - a = c - b) → b = 45 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_middle_term_l3384_338486


namespace NUMINAMATH_CALUDE_cafe_location_l3384_338450

/-- Represents a point in 2D space -/
structure Point where
  x : ℚ
  y : ℚ

/-- Checks if a point divides a line segment in a given ratio -/
def divides_segment (p1 p2 p : Point) (m n : ℚ) : Prop :=
  p.x = (n * p1.x + m * p2.x) / (m + n) ∧
  p.y = (n * p1.y + m * p2.y) / (m + n)

theorem cafe_location :
  let mark := Point.mk 1 8
  let sandy := Point.mk (-5) 0
  let cafe := Point.mk (-3) (8/3)
  divides_segment mark sandy cafe 1 2 := by
  sorry

end NUMINAMATH_CALUDE_cafe_location_l3384_338450


namespace NUMINAMATH_CALUDE_planes_parallel_iff_skew_lines_parallel_l3384_338456

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the parallel relation for planes and lines
variable (parallel : Plane → Plane → Prop)
variable (parallel_line_plane : Line → Plane → Prop)

-- Define the "contained in" relation for lines and planes
variable (contained_in : Line → Plane → Prop)

-- Define the "skew" relation for lines
variable (skew : Line → Line → Prop)

-- Theorem statement
theorem planes_parallel_iff_skew_lines_parallel (α β : Plane) :
  parallel α β ↔
  ∃ (a b : Line),
    skew a b ∧
    contained_in a α ∧
    contained_in b β ∧
    parallel_line_plane a β ∧
    parallel_line_plane b α :=
sorry

end NUMINAMATH_CALUDE_planes_parallel_iff_skew_lines_parallel_l3384_338456


namespace NUMINAMATH_CALUDE_waxing_time_is_36_minutes_l3384_338427

/-- Represents the time spent on different parts of car washing -/
structure CarWashTime where
  windows : ℕ
  body : ℕ
  tires : ℕ

/-- Calculates the total waxing time for all cars -/
def calculate_waxing_time (normal_car_time : CarWashTime) (normal_car_count : ℕ) (suv_count : ℕ) (total_time : ℕ) : ℕ :=
  let normal_car_wash_time := normal_car_time.windows + normal_car_time.body + normal_car_time.tires
  let total_wash_time_without_waxing := normal_car_wash_time * normal_car_count + (normal_car_wash_time * 2 * suv_count)
  total_time - total_wash_time_without_waxing

/-- Theorem stating that the waxing time is 36 minutes given the problem conditions -/
theorem waxing_time_is_36_minutes :
  let normal_car_time : CarWashTime := ⟨4, 7, 4⟩
  let normal_car_count : ℕ := 2
  let suv_count : ℕ := 1
  let total_time : ℕ := 96
  calculate_waxing_time normal_car_time normal_car_count suv_count total_time = 36 := by
  sorry

end NUMINAMATH_CALUDE_waxing_time_is_36_minutes_l3384_338427


namespace NUMINAMATH_CALUDE_parabola_f_value_l3384_338408

-- Define the parabola equation
def parabola (d e f y : ℝ) : ℝ := d * y^2 + e * y + f

-- Theorem statement
theorem parabola_f_value :
  ∀ d e f : ℝ,
  -- Vertex condition
  (∀ y : ℝ, parabola d e f (-3) = 2) ∧
  -- Point (7, 0) condition
  parabola d e f 0 = 7 →
  -- Conclusion: f = 7
  f = 7 := by
  sorry

end NUMINAMATH_CALUDE_parabola_f_value_l3384_338408


namespace NUMINAMATH_CALUDE_remainder_of_large_power_l3384_338496

theorem remainder_of_large_power (n : ℕ) : 
  4^(4^(4^4)) ≡ 656 [ZMOD 1000] :=
sorry

end NUMINAMATH_CALUDE_remainder_of_large_power_l3384_338496


namespace NUMINAMATH_CALUDE_total_arc_length_is_900_l3384_338438

/-- A triangle with its circumcircle -/
structure CircumscribedTriangle where
  /-- The radius of the circumcircle -/
  radius : ℝ
  /-- The perimeter of the triangle -/
  perimeter : ℝ

/-- The total length of arcs XX', YY', and ZZ' in a circumscribed triangle -/
def total_arc_length (t : CircumscribedTriangle) : ℝ := sorry

/-- Theorem: The total length of arcs XX', YY', and ZZ' is 900° -/
theorem total_arc_length_is_900 (t : CircumscribedTriangle) 
  (h1 : t.radius = 5) 
  (h2 : t.perimeter = 24) : 
  total_arc_length t = 900 := by sorry

end NUMINAMATH_CALUDE_total_arc_length_is_900_l3384_338438


namespace NUMINAMATH_CALUDE_product_sign_l3384_338457

theorem product_sign (a b c d e : ℝ) (h : a * b^2 * c^3 * d^4 * e^5 < 0) : 
  a * b^2 * c * d^4 * e < 0 := by
sorry

end NUMINAMATH_CALUDE_product_sign_l3384_338457
