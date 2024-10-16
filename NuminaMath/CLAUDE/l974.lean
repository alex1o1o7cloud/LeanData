import Mathlib

namespace NUMINAMATH_CALUDE_least_positive_integer_divisible_by_53_l974_97411

theorem least_positive_integer_divisible_by_53 :
  ∃ (x : ℕ), x > 0 ∧ 
  (∀ (y : ℕ), y > 0 → y < x → ¬(53 ∣ (3*y)^2 + 2*41*3*y + 41^2)) ∧
  (53 ∣ (3*x)^2 + 2*41*3*x + 41^2) ∧ 
  x = 4 := by
  sorry

end NUMINAMATH_CALUDE_least_positive_integer_divisible_by_53_l974_97411


namespace NUMINAMATH_CALUDE_cookie_count_l974_97487

theorem cookie_count (bags : ℕ) (cookies_per_bag : ℕ) (h1 : bags = 37) (h2 : cookies_per_bag = 19) :
  bags * cookies_per_bag = 703 := by
  sorry

end NUMINAMATH_CALUDE_cookie_count_l974_97487


namespace NUMINAMATH_CALUDE_possible_winning_scores_for_A_l974_97415

/-- Represents the outcome of a single question for a team -/
inductive QuestionOutcome
  | Correct
  | Incorrect
  | NoBuzz

/-- Calculates the score for a single question based on the outcome -/
def scoreQuestion (outcome : QuestionOutcome) : Int :=
  match outcome with
  | QuestionOutcome.Correct => 1
  | QuestionOutcome.Incorrect => -1
  | QuestionOutcome.NoBuzz => 0

/-- Calculates the total score for a team based on their outcomes for three questions -/
def calculateScore (q1 q2 q3 : QuestionOutcome) : Int :=
  scoreQuestion q1 + scoreQuestion q2 + scoreQuestion q3

/-- Defines a winning condition for team A -/
def teamAWins (scoreA scoreB : Int) : Prop :=
  scoreA > scoreB

/-- The main theorem stating the possible winning scores for team A -/
theorem possible_winning_scores_for_A :
  ∀ (q1A q2A q3A q1B q2B q3B : QuestionOutcome),
    let scoreA := calculateScore q1A q2A q3A
    let scoreB := calculateScore q1B q2B q3B
    teamAWins scoreA scoreB →
    (scoreA = -1 ∨ scoreA = 0 ∨ scoreA = 1 ∨ scoreA = 3) :=
  sorry


end NUMINAMATH_CALUDE_possible_winning_scores_for_A_l974_97415


namespace NUMINAMATH_CALUDE_water_bottle_cost_l974_97462

/-- Proves that the cost of a water bottle is $2 given the conditions of Adam's shopping trip. -/
theorem water_bottle_cost (num_sandwiches : ℕ) (sandwich_price total_cost : ℚ) : 
  num_sandwiches = 3 →
  sandwich_price = 3 →
  total_cost = 11 →
  total_cost - (num_sandwiches : ℚ) * sandwich_price = 2 :=
by sorry

end NUMINAMATH_CALUDE_water_bottle_cost_l974_97462


namespace NUMINAMATH_CALUDE_phi_function_form_l974_97429

/-- A direct proportion function -/
def DirectProportion (f : ℝ → ℝ) : Prop :=
  ∃ m : ℝ, m ≠ 0 ∧ ∀ x, f x = m * x

/-- An inverse proportion function -/
def InverseProportion (g : ℝ → ℝ) : Prop :=
  ∃ n : ℝ, n ≠ 0 ∧ ∀ x, x ≠ 0 → g x = n / x

/-- The main theorem -/
theorem phi_function_form (f g : ℝ → ℝ) (φ : ℝ → ℝ) :
  DirectProportion f →
  InverseProportion g →
  (∀ x, φ x = f x + g x) →
  φ 1 = 8 →
  (∃ x, φ x = 16) →
  ∀ x, x ≠ 0 → φ x = 3 * x + 5 / x := by
  sorry

end NUMINAMATH_CALUDE_phi_function_form_l974_97429


namespace NUMINAMATH_CALUDE_carlton_zoo_total_l974_97491

/-- Represents the number of animals in each zoo -/
structure ZooAnimals :=
  (rhinoceroses : ℕ)
  (elephants : ℕ)
  (lions : ℕ)
  (monkeys : ℕ)
  (penguins : ℕ)

/-- Defines the relationship between Bell Zoo and Carlton Zoo -/
def zoo_relationship (bell : ZooAnimals) (carlton : ZooAnimals) : Prop :=
  bell.rhinoceroses = carlton.lions ∧
  bell.elephants = carlton.lions + 3 ∧
  bell.elephants = carlton.rhinoceroses ∧
  carlton.elephants = carlton.rhinoceroses + 2 ∧
  carlton.monkeys = 2 * (carlton.rhinoceroses + carlton.elephants + carlton.lions) ∧
  carlton.penguins = carlton.monkeys + 2 ∧
  bell.monkeys = 2 * carlton.penguins / 3 ∧
  bell.penguins = bell.monkeys + 2 ∧
  bell.lions * 2 = bell.penguins ∧
  bell.rhinoceroses + bell.elephants + bell.lions + bell.monkeys + bell.penguins = 48

theorem carlton_zoo_total (bell : ZooAnimals) (carlton : ZooAnimals) 
  (h : zoo_relationship bell carlton) : 
  carlton.rhinoceroses + carlton.elephants + carlton.lions + carlton.monkeys + carlton.penguins = 57 :=
by sorry


end NUMINAMATH_CALUDE_carlton_zoo_total_l974_97491


namespace NUMINAMATH_CALUDE_valid_configurations_l974_97485

/-- A configuration of lines and points on a plane -/
structure PlaneConfiguration where
  n : ℕ  -- number of points
  lines : Fin 3 → Set (ℝ × ℝ)  -- three lines represented as sets of points
  points : Fin n → ℝ × ℝ  -- n points

/-- Predicate to check if a point is on a line -/
def isOnLine (p : ℝ × ℝ) (l : Set (ℝ × ℝ)) : Prop :=
  p ∈ l

/-- Predicate to check if a point is on either side of a line -/
def isOnEitherSide (p : ℝ × ℝ) (l : Set (ℝ × ℝ)) : Prop :=
  ¬(isOnLine p l)

/-- The main theorem stating the possible values of n -/
theorem valid_configurations (c : PlaneConfiguration) :
  (∀ l : Fin 3, ∃! (s₁ s₂ : Finset (Fin c.n)),
    s₁.card = 2 ∧ s₂.card = 2 ∧ 
    (∀ i ∈ s₁, isOnEitherSide (c.points i) (c.lines l)) ∧
    (∀ i ∈ s₂, isOnEitherSide (c.points i) (c.lines l)) ∧
    (∀ i : Fin c.n, i ∉ s₁ ∧ i ∉ s₂ → isOnLine (c.points i) (c.lines l))) →
  c.n = 0 ∨ c.n = 1 ∨ c.n = 3 ∨ c.n = 4 ∨ c.n = 6 ∨ c.n = 7 :=
by sorry

end NUMINAMATH_CALUDE_valid_configurations_l974_97485


namespace NUMINAMATH_CALUDE_consecutive_negative_integers_sum_l974_97463

theorem consecutive_negative_integers_sum (n : ℤ) : 
  n < 0 ∧ n * (n + 1) = 1224 → n + (n + 1) = -69 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_negative_integers_sum_l974_97463


namespace NUMINAMATH_CALUDE_count_divisible_by_four_l974_97439

theorem count_divisible_by_four : 
  (Finset.filter (fun n : Fin 10 => (748 * 10 + n : ℕ) % 4 = 0) Finset.univ).card = 3 :=
by sorry

end NUMINAMATH_CALUDE_count_divisible_by_four_l974_97439


namespace NUMINAMATH_CALUDE_circle_positions_l974_97484

theorem circle_positions (a b d : ℝ) (h1 : a = 4) (h2 : b = 10) (h3 : b > a) :
  (∃ d, d = b - a) ∧
  (∃ d, d = b + a) ∧
  (∃ d, d > b + a) ∧
  (∃ d, d > b - a) :=
by sorry

end NUMINAMATH_CALUDE_circle_positions_l974_97484


namespace NUMINAMATH_CALUDE_even_decreasing_function_ordering_l974_97428

def is_even (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f x = f (-x)

def monotone_decreasing_on_pos (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, 0 < x → 0 < y → x < y → f y < f x

theorem even_decreasing_function_ordering (f : ℝ → ℝ) 
  (h_even : is_even f) 
  (h_decreasing : monotone_decreasing_on_pos f) :
  f 3 < f (-2) ∧ f (-2) < f 1 :=
sorry

end NUMINAMATH_CALUDE_even_decreasing_function_ordering_l974_97428


namespace NUMINAMATH_CALUDE_groups_with_pair_fraction_l974_97432

-- Define the number of people
def n : ℕ := 6

-- Define the size of each group
def k : ℕ := 3

-- Define the function to calculate combinations
def C (n k : ℕ) : ℕ := Nat.choose n k

-- Theorem statement
theorem groups_with_pair_fraction :
  C (n - 2) (k - 2) / C n k = 1 / 5 := by sorry

end NUMINAMATH_CALUDE_groups_with_pair_fraction_l974_97432


namespace NUMINAMATH_CALUDE_min_value_expression_min_value_achievable_l974_97441

open Real

theorem min_value_expression (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  (x + 1/y)^2 * (x + 1/y - 2023) + (y + 1/x)^2 * (y + 1/x - 2023) ≥ -1814505489.667 :=
sorry

theorem min_value_achievable :
  ∃ (x y : ℝ), x > 0 ∧ y > 0 ∧
  (x + 1/y)^2 * (x + 1/y - 2023) + (y + 1/x)^2 * (y + 1/x - 2023) = -1814505489.667 :=
sorry

end NUMINAMATH_CALUDE_min_value_expression_min_value_achievable_l974_97441


namespace NUMINAMATH_CALUDE_sum_of_rectangle_areas_l974_97486

def rectangle_width : ℕ := 3

def odd_numbers : List ℕ := [1, 3, 5, 7, 9, 11, 13]

def rectangle_lengths : List ℕ := odd_numbers.map (λ x => x * x)

def rectangle_areas : List ℕ := rectangle_lengths.map (λ x => rectangle_width * x)

theorem sum_of_rectangle_areas :
  rectangle_areas.sum = 1365 := by sorry

end NUMINAMATH_CALUDE_sum_of_rectangle_areas_l974_97486


namespace NUMINAMATH_CALUDE_cone_volume_l974_97471

/-- Given a cone with slant height 1 and lateral surface area 2π/3, its volume is 4√5π/81 -/
theorem cone_volume (s : Real) (A : Real) (V : Real) : 
  s = 1 → A = (2/3) * Real.pi → V = (4 * Real.sqrt 5 / 81) * Real.pi :=
by sorry

end NUMINAMATH_CALUDE_cone_volume_l974_97471


namespace NUMINAMATH_CALUDE_sum_odd_integers_eq_1040_l974_97492

/-- The sum of odd integers from 15 to 65, inclusive -/
def sum_odd_integers : ℕ :=
  let first := 15
  let last := 65
  let n := (last - first) / 2 + 1
  n * (first + last) / 2

theorem sum_odd_integers_eq_1040 : sum_odd_integers = 1040 := by
  sorry

end NUMINAMATH_CALUDE_sum_odd_integers_eq_1040_l974_97492


namespace NUMINAMATH_CALUDE_cubic_root_sum_l974_97496

theorem cubic_root_sum (p q r : ℝ) : 
  (3 * p^3 - 5 * p^2 + 12 * p - 7 = 0) →
  (3 * q^3 - 5 * q^2 + 12 * q - 7 = 0) →
  (3 * r^3 - 5 * r^2 + 12 * r - 7 = 0) →
  (p + q + r = 5/3) →
  (p + q - 2)^3 + (q + r - 2)^3 + (r + p - 2)^3 = -35/3 := by
  sorry

end NUMINAMATH_CALUDE_cubic_root_sum_l974_97496


namespace NUMINAMATH_CALUDE_initial_amount_theorem_l974_97446

/-- The amount of money in Olivia's wallet before visiting the supermarket. -/
def initial_amount : ℕ := sorry

/-- The amount of money Olivia spent at the supermarket. -/
def amount_spent : ℕ := 16

/-- The amount of money left in Olivia's wallet after visiting the supermarket. -/
def amount_left : ℕ := 78

/-- Theorem stating that the initial amount in Olivia's wallet was $94. -/
theorem initial_amount_theorem : initial_amount = 94 :=
by
  sorry

end NUMINAMATH_CALUDE_initial_amount_theorem_l974_97446


namespace NUMINAMATH_CALUDE_octal_difference_multiple_of_seven_fifty_six_possible_difference_l974_97400

-- Define a two-digit number in base 8
def octal_number (tens units : Nat) : Nat :=
  8 * tens + units

-- Define the reversed number
def reversed_octal_number (tens units : Nat) : Nat :=
  8 * units + tens

-- Define the difference between the original and reversed number
def octal_difference (tens units : Nat) : Int :=
  (octal_number tens units : Int) - (reversed_octal_number tens units : Int)

-- Theorem stating that the difference is always a multiple of 7
theorem octal_difference_multiple_of_seven (tens units : Nat) :
  ∃ k : Int, octal_difference tens units = 7 * k :=
sorry

-- Theorem stating that 56 is a possible difference
theorem fifty_six_possible_difference :
  ∃ tens units : Nat, octal_difference tens units = 56 :=
sorry

end NUMINAMATH_CALUDE_octal_difference_multiple_of_seven_fifty_six_possible_difference_l974_97400


namespace NUMINAMATH_CALUDE_geometric_proof_l974_97436

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the perpendicular relation between planes
variable (perp_planes : Plane → Plane → Prop)

-- Define the perpendicular relation between a line and a plane
variable (perp_line_plane : Line → Plane → Prop)

-- Define the intersection of two planes resulting in a line
variable (intersect_planes : Plane → Plane → Line)

-- Define the relation of a line being contained in a plane
variable (line_in_plane : Line → Plane → Prop)

-- Define the perpendicular relation between lines
variable (perp_lines : Line → Line → Prop)

-- State the theorem
theorem geometric_proof 
  (m n : Line) (α β γ : Plane)
  (h1 : perp_planes α β)
  (h2 : m = intersect_planes α β)
  (h3 : perp_line_plane n α)
  (h4 : line_in_plane n γ) :
  perp_lines m n ∧ perp_planes α γ := by
  sorry

end NUMINAMATH_CALUDE_geometric_proof_l974_97436


namespace NUMINAMATH_CALUDE_prob_at_least_two_tails_in_three_flips_prob_at_least_two_tails_in_three_flips_is_half_l974_97454

/-- The probability of getting at least two tails in three independent flips of a fair coin -/
theorem prob_at_least_two_tails_in_three_flips : ℝ :=
  let p_head : ℝ := 1/2  -- probability of getting heads on a single flip
  let p_tail : ℝ := 1 - p_head  -- probability of getting tails on a single flip
  let p_all_heads : ℝ := p_head ^ 3  -- probability of getting all heads
  let p_one_tail : ℝ := 3 * p_head ^ 2 * p_tail  -- probability of getting exactly one tail
  1 - (p_all_heads + p_one_tail)

/-- The probability of getting at least two tails in three independent flips of a fair coin is 1/2 -/
theorem prob_at_least_two_tails_in_three_flips_is_half :
  prob_at_least_two_tails_in_three_flips = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_prob_at_least_two_tails_in_three_flips_prob_at_least_two_tails_in_three_flips_is_half_l974_97454


namespace NUMINAMATH_CALUDE_max_servings_is_ten_l974_97475

/-- Represents the number of chunks per serving for each fruit type -/
structure FruitRatio where
  cantaloupe : ℕ
  honeydew : ℕ
  pineapple : ℕ
  watermelon : ℕ

/-- Represents the available chunks of each fruit type -/
structure AvailableFruit where
  cantaloupe : ℕ
  honeydew : ℕ
  pineapple : ℕ
  watermelon : ℕ

/-- Calculates the maximum number of servings that can be made -/
def maxServings (ratio : FruitRatio) (available : AvailableFruit) : ℕ :=
  min
    (available.cantaloupe / ratio.cantaloupe)
    (min
      (available.honeydew / ratio.honeydew)
      (min
        (available.pineapple / ratio.pineapple)
        (available.watermelon / ratio.watermelon)))

/-- The given fruit ratio -/
def givenRatio : FruitRatio :=
  { cantaloupe := 3
  , honeydew := 2
  , pineapple := 1
  , watermelon := 4 }

/-- The available fruit chunks -/
def givenAvailable : AvailableFruit :=
  { cantaloupe := 30
  , honeydew := 42
  , pineapple := 12
  , watermelon := 56 }

theorem max_servings_is_ten :
  maxServings givenRatio givenAvailable = 10 := by
  sorry


end NUMINAMATH_CALUDE_max_servings_is_ten_l974_97475


namespace NUMINAMATH_CALUDE_sally_balloons_l974_97416

/-- 
Given that Sally has x orange balloons initially, finds 2 more orange balloons,
and ends up with 11 orange balloons in total, prove that x = 9.
-/
theorem sally_balloons (x : ℝ) : x + 2 = 11 → x = 9 := by
  sorry

end NUMINAMATH_CALUDE_sally_balloons_l974_97416


namespace NUMINAMATH_CALUDE_nancy_clay_pots_l974_97478

/-- The number of clay pots Nancy created on Monday -/
def monday_pots : ℕ := 12

/-- The number of clay pots Nancy created on Tuesday -/
def tuesday_pots : ℕ := 2 * monday_pots

/-- The number of clay pots Nancy created on Wednesday -/
def wednesday_pots : ℕ := 14

/-- The total number of clay pots Nancy created by the end of the week -/
def total_pots : ℕ := monday_pots + tuesday_pots + wednesday_pots

theorem nancy_clay_pots : total_pots = 50 := by sorry

end NUMINAMATH_CALUDE_nancy_clay_pots_l974_97478


namespace NUMINAMATH_CALUDE_f_properties_l974_97405

/-- An odd function f(x) with a parameter a ≠ 0 -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log ((a * x) / (1 - x) + 1)

/-- The function f is odd -/
axiom f_odd (a : ℝ) (h : a ≠ 0) : ∀ x, f a (-x) = -(f a x)

/-- Theorem stating the properties of the function f -/
theorem f_properties :
  ∃ (a : ℝ), a ≠ 0 ∧ 
  (a = 2) ∧ 
  (∀ x, f a x ≠ 0 ↔ -1 < x ∧ x < 1) ∧
  (∀ x y, -1 < x ∧ x < y ∧ y < 1 → f a x < f a y) :=
sorry

end NUMINAMATH_CALUDE_f_properties_l974_97405


namespace NUMINAMATH_CALUDE_intersection_equals_open_closed_interval_l974_97472

-- Define set A
def A : Set ℝ := {x | x^2 - 1 ≤ 0}

-- Define set B
def B : Set ℝ := {x | 0 < x ∧ x ≤ 2}

-- Define the intersection of A and B
def A_intersect_B : Set ℝ := A ∩ B

-- Theorem statement
theorem intersection_equals_open_closed_interval :
  A_intersect_B = Set.Ioo 0 1 ∪ Set.Ioc 1 1 :=
sorry

end NUMINAMATH_CALUDE_intersection_equals_open_closed_interval_l974_97472


namespace NUMINAMATH_CALUDE_total_cans_collected_l974_97494

def saturday_bags : ℕ := 3
def sunday_bags : ℕ := 4
def cans_per_bag : ℕ := 9

theorem total_cans_collected :
  saturday_bags * cans_per_bag + sunday_bags * cans_per_bag = 63 :=
by sorry

end NUMINAMATH_CALUDE_total_cans_collected_l974_97494


namespace NUMINAMATH_CALUDE_area_of_triangle_MOI_l974_97456

/-- Triangle ABC with given side lengths --/
structure Triangle :=
  (A B C : ℝ × ℝ)
  (AB_length : Real.sqrt ((B.1 - A.1)^2 + (B.2 - A.2)^2) = 15)
  (AC_length : Real.sqrt ((C.1 - A.1)^2 + (C.2 - A.2)^2) = 8)
  (BC_length : Real.sqrt ((C.1 - B.1)^2 + (C.2 - B.2)^2) = 17)

/-- Circumcenter of a triangle --/
def circumcenter (t : Triangle) : ℝ × ℝ := sorry

/-- Incenter of a triangle --/
def incenter (t : Triangle) : ℝ × ℝ := sorry

/-- Center of circle tangent to AC, BC, and circumcircle --/
def tangent_circle_center (t : Triangle) : ℝ × ℝ := sorry

/-- Check if a point lies on the internal bisector of angle A --/
def on_angle_bisector (t : Triangle) (p : ℝ × ℝ) : Prop := sorry

/-- Area of a triangle given its vertices --/
def triangle_area (p q r : ℝ × ℝ) : ℝ := sorry

/-- Main theorem --/
theorem area_of_triangle_MOI (t : Triangle) :
  let O := circumcenter t
  let I := incenter t
  let M := tangent_circle_center t
  on_angle_bisector t M →
  triangle_area M O I = 4.5 := by sorry

end NUMINAMATH_CALUDE_area_of_triangle_MOI_l974_97456


namespace NUMINAMATH_CALUDE_range_of_m_l974_97408

theorem range_of_m (x y m : ℝ) 
  (hx : x > 0) 
  (hy : y > 0) 
  (h_eq : 2/x + 1/y = 1) 
  (h_ineq : ∀ (x y : ℝ), x > 0 → y > 0 → 2/x + 1/y = 1 → x + 2*y > m^2 + 2*m) : 
  -4 < m ∧ m < 2 := by
sorry

end NUMINAMATH_CALUDE_range_of_m_l974_97408


namespace NUMINAMATH_CALUDE_double_inequality_proof_l974_97426

theorem double_inequality_proof (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  (0 < 1 / (x + y + z + 1) - 1 / ((x + 1) * (y + 1) * (z + 1))) ∧
  (1 / (x + y + z + 1) - 1 / ((x + 1) * (y + 1) * (z + 1)) ≤ 1 / 8) ∧
  (1 / (x + y + z + 1) - 1 / ((x + 1) * (y + 1) * (z + 1)) = 1 / 8 ↔ x = 1 ∧ y = 1 ∧ z = 1) :=
by sorry

end NUMINAMATH_CALUDE_double_inequality_proof_l974_97426


namespace NUMINAMATH_CALUDE_pascal_ratio_98_l974_97490

/-- Binomial coefficient -/
def binomial (n k : ℕ) : ℕ := sorry

/-- Pascal's Triangle property: each entry is the sum of the two entries directly above it -/
axiom pascal_property (n k : ℕ) : binomial (n + 1) k = binomial n (k - 1) + binomial n k

/-- Three consecutive entries in Pascal's Triangle -/
def consecutive_entries (n r : ℕ) : (ℕ × ℕ × ℕ) :=
  (binomial n r, binomial n (r + 1), binomial n (r + 2))

/-- Ratio of three numbers -/
def in_ratio (a b c : ℕ) (x y z : ℕ) : Prop :=
  a * y = b * x ∧ b * z = c * y

theorem pascal_ratio_98 : ∃ r : ℕ, in_ratio (binomial 98 r) (binomial 98 (r + 1)) (binomial 98 (r + 2)) 4 5 6 := by
  sorry

end NUMINAMATH_CALUDE_pascal_ratio_98_l974_97490


namespace NUMINAMATH_CALUDE_trajectory_of_tangent_circles_l974_97401

-- Define the circles
def C1 (x y : ℝ) : Prop := x^2 + (y + 1)^2 = 25
def C2 (x y : ℝ) : Prop := x^2 + (y - 2)^2 = 1

-- Define the trajectory equation
def trajectory_equation (x y : ℝ) : Prop := y^2 / 9 + x^2 / 5 = 1 ∧ y ≠ 3

-- Theorem statement
theorem trajectory_of_tangent_circles :
  ∀ x y : ℝ, 
  (∃ r : ℝ, (x - 0)^2 + (y - (-1))^2 = (5 - r)^2 ∧ (x - 0)^2 + (y - 2)^2 = (r + 1)^2) →
  trajectory_equation x y :=
sorry

end NUMINAMATH_CALUDE_trajectory_of_tangent_circles_l974_97401


namespace NUMINAMATH_CALUDE_solution_satisfies_equation_l974_97461

/-- The general solution to the differential equation (4y - 3x - 5)y' + 7x - 3y + 2 = 0 -/
def general_solution (x y : ℝ) (C : ℝ) : Prop :=
  2 * y^2 - 3 * x * y + (7/2) * x^2 + 2 * x - 5 * y = C

/-- The differential equation (4y - 3x - 5)y' + 7x - 3y + 2 = 0 -/
def differential_equation (x y : ℝ) (y' : ℝ → ℝ) : Prop :=
  (4 * y - 3 * x - 5) * (y' x) + 7 * x - 3 * y + 2 = 0

theorem solution_satisfies_equation :
  ∀ (x y : ℝ) (C : ℝ),
  general_solution x y C →
  ∃ (y' : ℝ → ℝ), differential_equation x y y' :=
sorry

end NUMINAMATH_CALUDE_solution_satisfies_equation_l974_97461


namespace NUMINAMATH_CALUDE_equal_sum_sequence_properties_l974_97443

/-- An equal sum sequence is a sequence where each term plus the previous term
    equals the same constant, starting from the second term. -/
def EqualSumSequence (a : ℕ → ℝ) :=
  ∃ k : ℝ, ∀ n : ℕ, n ≥ 1 → a n + a (n + 1) = k

theorem equal_sum_sequence_properties (a : ℕ → ℝ) (h : EqualSumSequence a) :
  (∀ n : ℕ, n ≥ 1 → a n = a (n + 2)) ∧
  (∀ m n : ℕ, m ≥ 1 → n ≥ 1 → Odd m ∧ Odd n → a m = a n) ∧
  (∀ m n : ℕ, m ≥ 1 → n ≥ 1 → Even m ∧ Even n → a m = a n) :=
by sorry

end NUMINAMATH_CALUDE_equal_sum_sequence_properties_l974_97443


namespace NUMINAMATH_CALUDE_tiffany_lives_l974_97476

/-- Calculates the final number of lives in a video game scenario -/
def final_lives (initial : ℕ) (lost : ℕ) (gained : ℕ) : ℕ :=
  initial - lost + gained

/-- Proves that for the given scenario, the final number of lives is 56 -/
theorem tiffany_lives : final_lives 43 14 27 = 56 := by
  sorry

end NUMINAMATH_CALUDE_tiffany_lives_l974_97476


namespace NUMINAMATH_CALUDE_sum_of_fractions_minus_ten_equals_zero_l974_97406

theorem sum_of_fractions_minus_ten_equals_zero :
  5 / 3 + 10 / 6 + 20 / 12 + 40 / 24 + 80 / 48 + 160 / 96 - 10 = 0 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_fractions_minus_ten_equals_zero_l974_97406


namespace NUMINAMATH_CALUDE_parallelogram_area_14_24_l974_97425

/-- The area of a parallelogram with given base and height -/
def parallelogram_area (base height : ℝ) : ℝ := base * height

/-- Theorem: The area of a parallelogram with base 14 cm and height 24 cm is 336 cm² -/
theorem parallelogram_area_14_24 :
  parallelogram_area 14 24 = 336 := by sorry

end NUMINAMATH_CALUDE_parallelogram_area_14_24_l974_97425


namespace NUMINAMATH_CALUDE_opposite_abs_power_l974_97489

theorem opposite_abs_power (x y : ℝ) : 
  |x - 2| + |y + 3| = 0 → (x + y)^2023 = -1 := by
  sorry

end NUMINAMATH_CALUDE_opposite_abs_power_l974_97489


namespace NUMINAMATH_CALUDE_sphere_volume_l974_97452

theorem sphere_volume (d : ℝ) (a : ℝ) (h1 : d = 2) (h2 : a = π) :
  let r := Real.sqrt (1^2 + d^2)
  (4 / 3) * π * r^3 = (20 * Real.sqrt 5 * π) / 3 := by
sorry

end NUMINAMATH_CALUDE_sphere_volume_l974_97452


namespace NUMINAMATH_CALUDE_sum_reciprocals_bound_l974_97412

theorem sum_reciprocals_bound (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x^2 + y^2 = 1) :
  1/x + 1/y ≥ 8 ∧ ∀ ε > 0, ∃ (x' y' : ℝ), x' > 0 ∧ y' > 0 ∧ x'^2 + y'^2 = 1 ∧ 1/x' + 1/y' > 1/ε :=
sorry

end NUMINAMATH_CALUDE_sum_reciprocals_bound_l974_97412


namespace NUMINAMATH_CALUDE_student_number_problem_l974_97467

theorem student_number_problem (x : ℝ) : (3/2 : ℝ) * x + 53.4 = -78.9 → x = -88.2 := by
  sorry

end NUMINAMATH_CALUDE_student_number_problem_l974_97467


namespace NUMINAMATH_CALUDE_parallel_resistances_solutions_l974_97498

theorem parallel_resistances_solutions : 
  ∀ x y z : ℕ+, 
    (1 : ℚ) / z = 1 / x + 1 / y → 
    ((x = 3 ∧ y = 6 ∧ z = 2) ∨ 
     (x = 4 ∧ y = 4 ∧ z = 2) ∨ 
     (x = 4 ∧ y = 12 ∧ z = 3) ∨ 
     (x = 6 ∧ y = 6 ∧ z = 3)) :=
by sorry

end NUMINAMATH_CALUDE_parallel_resistances_solutions_l974_97498


namespace NUMINAMATH_CALUDE_complex_parts_of_z_l974_97477

theorem complex_parts_of_z : ∃ z : ℂ, z = Complex.I ^ 2 + Complex.I ∧ z.re = -1 ∧ z.im = 1 := by
  sorry

end NUMINAMATH_CALUDE_complex_parts_of_z_l974_97477


namespace NUMINAMATH_CALUDE_simplify_and_evaluate_evaluate_expression_l974_97422

theorem simplify_and_evaluate (a b : ℝ) :
  (a - b)^2 - 2*a*(a + b) + (a + 2*b)*(a - 2*b) = -4*a*b - 3*b^2 :=
by sorry

theorem evaluate_expression :
  let a : ℝ := -1
  let b : ℝ := 4
  (a - b)^2 - 2*a*(a + b) + (a + 2*b)*(a - 2*b) = -32 :=
by sorry

end NUMINAMATH_CALUDE_simplify_and_evaluate_evaluate_expression_l974_97422


namespace NUMINAMATH_CALUDE_inequality_proof_l974_97444

theorem inequality_proof (w x y z : ℝ) (h : w^2 + y^2 ≤ 1) :
  (w*x + y*z - 1)^2 ≥ (w^2 + y^2 - 1)*(x^2 + z^2 - 1) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l974_97444


namespace NUMINAMATH_CALUDE_basketball_probability_l974_97404

theorem basketball_probability (p_no_make : ℝ) (num_tries : ℕ) : 
  p_no_make = 1/3 → num_tries = 3 → 
  let p_make := 1 - p_no_make
  (num_tries.choose 1) * p_make * p_no_make^2 = 2/9 := by
sorry

end NUMINAMATH_CALUDE_basketball_probability_l974_97404


namespace NUMINAMATH_CALUDE_chelsea_cupcake_time_l974_97453

/-- The time it takes to make cupcakes given the number of batches and time per batch -/
def cupcake_time (num_batches : ℕ) (bake_time : ℕ) (ice_time : ℕ) : ℕ :=
  num_batches * (bake_time + ice_time)

/-- Theorem: Chelsea's cupcake-making time -/
theorem chelsea_cupcake_time :
  cupcake_time 4 20 30 = 200 := by
  sorry

end NUMINAMATH_CALUDE_chelsea_cupcake_time_l974_97453


namespace NUMINAMATH_CALUDE_pancake_cost_l974_97473

/-- The cost of a stack of pancakes satisfies the given conditions -/
theorem pancake_cost (pancake_stacks : ℕ) (bacon_slices : ℕ) (bacon_price : ℚ) (total_raised : ℚ) :
  pancake_stacks = 60 →
  bacon_slices = 90 →
  bacon_price = 2 →
  total_raised = 420 →
  ∃ (P : ℚ), P * pancake_stacks + bacon_price * bacon_slices = total_raised ∧ P = 4 :=
by sorry

end NUMINAMATH_CALUDE_pancake_cost_l974_97473


namespace NUMINAMATH_CALUDE_intersection_M_N_l974_97466

-- Define the sets M and N
def M : Set ℝ := {x | 0 < x ∧ x < 3}
def N : Set ℝ := {x | x^2 - 5*x + 4 ≥ 0}

-- State the theorem
theorem intersection_M_N : M ∩ N = {x | 0 < x ∧ x ≤ 1} := by
  sorry

end NUMINAMATH_CALUDE_intersection_M_N_l974_97466


namespace NUMINAMATH_CALUDE_inscribed_trapezoids_equal_diagonals_l974_97480

-- Define a circle
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define an isosceles trapezoid
structure IsoscelesTrapezoid where
  vertices : Fin 4 → ℝ × ℝ

-- Define the property of being inscribed in a circle
def inscribed (t : IsoscelesTrapezoid) (c : Circle) : Prop := sorry

-- Define the property of sides being parallel
def parallel_sides (t1 t2 : IsoscelesTrapezoid) : Prop := sorry

-- Define the length of a diagonal
def diagonal_length (t : IsoscelesTrapezoid) : ℝ := sorry

-- Main theorem
theorem inscribed_trapezoids_equal_diagonals 
  (c : Circle) (t1 t2 : IsoscelesTrapezoid) 
  (h1 : inscribed t1 c) (h2 : inscribed t2 c) 
  (h3 : parallel_sides t1 t2) : 
  diagonal_length t1 = diagonal_length t2 := by sorry

end NUMINAMATH_CALUDE_inscribed_trapezoids_equal_diagonals_l974_97480


namespace NUMINAMATH_CALUDE_number_equation_solution_l974_97403

theorem number_equation_solution :
  ∃ x : ℝ, x + 5 * 12 / (180 / 3) = 51 ∧ x = 50 := by
sorry

end NUMINAMATH_CALUDE_number_equation_solution_l974_97403


namespace NUMINAMATH_CALUDE_complex_magnitude_inequality_l974_97413

theorem complex_magnitude_inequality (a : ℝ) : 
  let z₁ : ℂ := a + 2*Complex.I
  let z₂ : ℂ := -2 + Complex.I
  Complex.abs z₁ < Complex.abs z₂ → -1 < a ∧ a < 1 := by
sorry

end NUMINAMATH_CALUDE_complex_magnitude_inequality_l974_97413


namespace NUMINAMATH_CALUDE_geometric_sequence_first_term_l974_97481

/-- Given a geometric sequence where the third term is 24 and the fourth term is 36, 
    the first term of the sequence is 32/3. -/
theorem geometric_sequence_first_term (a : ℚ) (r : ℚ) : 
  a * r^2 = 24 ∧ a * r^3 = 36 → a = 32/3 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_first_term_l974_97481


namespace NUMINAMATH_CALUDE_wood_length_problem_l974_97414

theorem wood_length_problem (first_set second_set : ℝ) :
  second_set = 5 * first_set →
  second_set = 20 →
  first_set = 4 :=
by
  sorry

end NUMINAMATH_CALUDE_wood_length_problem_l974_97414


namespace NUMINAMATH_CALUDE_gift_wrap_sales_l974_97474

theorem gift_wrap_sales (solid_price print_price total_rolls total_amount : ℝ) 
  (h1 : solid_price = 4)
  (h2 : print_price = 6)
  (h3 : total_rolls = 480)
  (h4 : total_amount = 2340)
  : ∃ (solid_rolls print_rolls : ℝ),
    solid_rolls + print_rolls = total_rolls ∧
    solid_price * solid_rolls + print_price * print_rolls = total_amount ∧
    print_rolls = 210 := by
  sorry

end NUMINAMATH_CALUDE_gift_wrap_sales_l974_97474


namespace NUMINAMATH_CALUDE_power_of_product_l974_97470

theorem power_of_product (a b : ℝ) : (-a * b^2)^2 = a^2 * b^4 := by
  sorry

end NUMINAMATH_CALUDE_power_of_product_l974_97470


namespace NUMINAMATH_CALUDE_expression_simplification_l974_97450

theorem expression_simplification (a b : ℤ) (h1 : a = -2) (h2 : b = 1) :
  a^3 * (-b^3)^2 + (-1/2 * a * b^2)^3 = -7 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l974_97450


namespace NUMINAMATH_CALUDE_roots_of_polynomial_l974_97437

def p (x : ℝ) : ℝ := 3*x^4 + 7*x^3 - 13*x^2 + 11*x - 6

theorem roots_of_polynomial :
  (p (-3) = 0) ∧ (p (-2) = 0) ∧ (p (-1) = 0) ∧ (p (1/3) = 0) :=
by sorry

end NUMINAMATH_CALUDE_roots_of_polynomial_l974_97437


namespace NUMINAMATH_CALUDE_tour_group_size_l974_97409

theorem tour_group_size (initial_groups : ℕ) (initial_avg : ℕ) (remaining_groups : ℕ) (remaining_avg : ℕ) :
  initial_groups = 10 →
  initial_avg = 9 →
  remaining_groups = 9 →
  remaining_avg = 8 →
  (initial_groups * initial_avg) - (remaining_groups * remaining_avg) = 18 :=
by
  sorry

end NUMINAMATH_CALUDE_tour_group_size_l974_97409


namespace NUMINAMATH_CALUDE_tricia_age_l974_97451

/-- Represents the ages of individuals in the problem -/
structure Ages where
  tricia : ℕ
  amilia : ℕ
  yorick : ℕ
  eugene : ℕ
  khloe : ℕ
  rupert : ℕ
  vincent : ℕ
  selena : ℕ
  cora : ℕ
  brody : ℕ

/-- Defines the relationships between ages as given in the problem -/
def valid_ages (a : Ages) : Prop :=
  a.tricia = a.amilia / 3 ∧
  a.amilia = a.yorick / 4 ∧
  a.yorick = 2 * a.eugene ∧
  a.khloe = a.eugene / 3 ∧
  a.rupert = a.khloe + 10 ∧
  a.rupert = a.vincent - 2 ∧
  a.vincent = 22 ∧
  a.yorick = a.selena + 5 ∧
  a.selena = a.amilia + 3 ∧
  a.cora = (a.vincent + a.amilia) / 2 ∧
  a.brody = a.tricia + a.vincent

/-- Theorem stating that if the ages satisfy the given relationships, then Tricia's age is 5 -/
theorem tricia_age (a : Ages) (h : valid_ages a) : a.tricia = 5 := by
  sorry


end NUMINAMATH_CALUDE_tricia_age_l974_97451


namespace NUMINAMATH_CALUDE_divisibility_and_smallest_m_l974_97410

def E (x y m : ℕ) : ℤ := (72 / x)^m + (72 / y)^m - x^m - y^m

theorem divisibility_and_smallest_m :
  ∀ k : ℕ,
  let m := 400 * k + 200
  2005 ∣ E 3 12 m ∧
  2005 ∣ E 9 6 m ∧
  (∀ m' : ℕ, m' > 0 ∧ m' < 200 → ¬(2005 ∣ E 3 12 m' ∧ 2005 ∣ E 9 6 m')) :=
by sorry

end NUMINAMATH_CALUDE_divisibility_and_smallest_m_l974_97410


namespace NUMINAMATH_CALUDE_largest_integer_inequality_l974_97459

theorem largest_integer_inequality (y : ℤ) : (y / 4 : ℚ) + 3 / 7 < 7 / 4 ↔ y ≤ 5 := by
  sorry

#check largest_integer_inequality

end NUMINAMATH_CALUDE_largest_integer_inequality_l974_97459


namespace NUMINAMATH_CALUDE_polynomial_value_theorem_l974_97458

/-- Given a polynomial function g(x) = px^4 + qx^3 + rx^2 + sx + t 
    where g(-1) = 4, prove that 16p - 8q + 4r - 2s + t = 64 -/
theorem polynomial_value_theorem 
  (p q r s t : ℝ) 
  (g : ℝ → ℝ)
  (h1 : ∀ x, g x = p * x^4 + q * x^3 + r * x^2 + s * x + t)
  (h2 : g (-1) = 4) :
  16 * p - 8 * q + 4 * r - 2 * s + t = 64 := by
sorry

end NUMINAMATH_CALUDE_polynomial_value_theorem_l974_97458


namespace NUMINAMATH_CALUDE_bob_pie_count_l974_97469

/-- The radius of Tom's circular pies in cm -/
def tom_radius : ℝ := 8

/-- The number of pies Tom can make in one batch -/
def tom_batch_size : ℕ := 6

/-- The length of one leg of Bob's right-angled triangular pies in cm -/
def bob_leg1 : ℝ := 6

/-- The length of the other leg of Bob's right-angled triangular pies in cm -/
def bob_leg2 : ℝ := 8

/-- The number of pies Bob can make with the same amount of dough as Tom -/
def bob_batch_size : ℕ := 50

theorem bob_pie_count :
  bob_batch_size = ⌊(tom_radius^2 * Real.pi * tom_batch_size) / (bob_leg1 * bob_leg2 / 2)⌋ := by
  sorry

end NUMINAMATH_CALUDE_bob_pie_count_l974_97469


namespace NUMINAMATH_CALUDE_ratio_problem_l974_97483

theorem ratio_problem (x : ℝ) : (20 / 1 = x / 10) → x = 200 := by
  sorry

end NUMINAMATH_CALUDE_ratio_problem_l974_97483


namespace NUMINAMATH_CALUDE_cans_collected_l974_97497

theorem cans_collected (solomon juwan levi : ℕ) : 
  solomon = 66 →
  solomon = 3 * juwan →
  levi = juwan / 2 →
  solomon + juwan + levi = 99 :=
by
  sorry

end NUMINAMATH_CALUDE_cans_collected_l974_97497


namespace NUMINAMATH_CALUDE_expression_simplification_l974_97407

theorem expression_simplification (x y : ℚ) (hx : x = 1/2) (hy : y = -2) :
  ((2*x + y)^2 - (2*x - y)*(x + y) - 2*(x - 2*y)*(x + 2*y)) / y = -37/2 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l974_97407


namespace NUMINAMATH_CALUDE_faster_car_speed_l974_97482

/-- Given two cars traveling in opposite directions for 5 hours, with one car
    traveling 10 mi/h faster than the other, and ending up 500 miles apart,
    prove that the speed of the faster car is 55 mi/h. -/
theorem faster_car_speed (slower_speed faster_speed : ℝ) : 
  faster_speed = slower_speed + 10 →
  5 * slower_speed + 5 * faster_speed = 500 →
  faster_speed = 55 := by sorry

end NUMINAMATH_CALUDE_faster_car_speed_l974_97482


namespace NUMINAMATH_CALUDE_john_needs_four_planks_l974_97447

/-- The number of planks John needs for the house wall -/
def num_planks (total_nails : ℕ) (nails_per_plank : ℕ) (additional_nails : ℕ) : ℕ :=
  (total_nails - additional_nails) / nails_per_plank

/-- Theorem stating that John needs 4 planks for the house wall -/
theorem john_needs_four_planks :
  num_planks 43 7 15 = 4 := by
  sorry

end NUMINAMATH_CALUDE_john_needs_four_planks_l974_97447


namespace NUMINAMATH_CALUDE_fixed_point_of_square_minus_600_l974_97445

theorem fixed_point_of_square_minus_600 :
  ∃! (x : ℕ), x = x^2 - 600 :=
by
  -- The unique natural number satisfying the equation is 25
  use 25
  constructor
  · -- Prove that 25 satisfies the equation
    norm_num
  · -- Prove that any natural number satisfying the equation must be 25
    intro y hy
    -- Here we would prove that y = 25
    sorry

#eval (25 : ℕ)^2 - 600  -- This should evaluate to 25

end NUMINAMATH_CALUDE_fixed_point_of_square_minus_600_l974_97445


namespace NUMINAMATH_CALUDE_total_money_proof_l974_97430

/-- Represents the ratio of money shares for Jonah, Kira, and Liam respectively -/
def money_ratio : Fin 3 → ℕ
| 0 => 2  -- Jonah's ratio
| 1 => 3  -- Kira's ratio
| 2 => 8  -- Liam's ratio

/-- Kira's share of the money -/
def kiras_share : ℕ := 45

/-- The total amount of money shared -/
def total_money : ℕ := 195

/-- Theorem stating that given the conditions, the total amount of money shared is $195 -/
theorem total_money_proof :
  (∃ (multiplier : ℚ), 
    (multiplier * money_ratio 1 = kiras_share) ∧ 
    (multiplier * (money_ratio 0 + money_ratio 1 + money_ratio 2) = total_money)) :=
by sorry

end NUMINAMATH_CALUDE_total_money_proof_l974_97430


namespace NUMINAMATH_CALUDE_expr_is_monomial_of_degree_3_l974_97464

/-- A monomial is an algebraic expression consisting of one term. This term can be a constant, a variable, or a product of constants and variables raised to whole number powers. -/
def is_monomial (e : Expr) : Prop := sorry

/-- The degree of a monomial is the sum of the exponents of all its variables. -/
def monomial_degree (e : Expr) : ℕ := sorry

/-- An algebraic expression. -/
inductive Expr
| const : ℚ → Expr
| var : String → Expr
| mul : Expr → Expr → Expr
| pow : Expr → ℕ → Expr

/-- The expression -x^2y -/
def expr : Expr :=
  Expr.mul (Expr.const (-1))
    (Expr.mul (Expr.pow (Expr.var "x") 2) (Expr.var "y"))

theorem expr_is_monomial_of_degree_3 :
  is_monomial expr ∧ monomial_degree expr = 3 := by sorry

end NUMINAMATH_CALUDE_expr_is_monomial_of_degree_3_l974_97464


namespace NUMINAMATH_CALUDE_cafeteria_pies_correct_l974_97424

def cafeteria_pies (initial_apples : ℕ) (handed_out : ℕ) (apples_per_pie : ℕ) : ℕ :=
  (initial_apples - handed_out) / apples_per_pie

theorem cafeteria_pies_correct : cafeteria_pies 47 27 4 = 5 := by
  sorry

end NUMINAMATH_CALUDE_cafeteria_pies_correct_l974_97424


namespace NUMINAMATH_CALUDE_marys_potatoes_l974_97468

/-- 
Given that Mary has some initial number of potatoes, rabbits ate 3 potatoes, 
and Mary now has 5 potatoes left, prove that Mary initially had 8 potatoes.
-/
theorem marys_potatoes (initial : ℕ) (eaten : ℕ) (remaining : ℕ) : 
  eaten = 3 → remaining = 5 → initial = eaten + remaining → initial = 8 := by
sorry

end NUMINAMATH_CALUDE_marys_potatoes_l974_97468


namespace NUMINAMATH_CALUDE_quadratic_equation_m_value_l974_97448

theorem quadratic_equation_m_value (m : ℝ) : 
  (∀ x, ∃ a b c : ℝ, (m + 2) * x^(m^2 - 2) + 2 * x + 1 = a * x^2 + b * x + c) ∧ 
  (m + 2 ≠ 0) → 
  m = 2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_m_value_l974_97448


namespace NUMINAMATH_CALUDE_factorization_of_cubic_l974_97465

theorem factorization_of_cubic (a : ℝ) : 
  -2 * a^3 + 12 * a^2 - 18 * a = -2 * a * (a - 3)^2 := by sorry

end NUMINAMATH_CALUDE_factorization_of_cubic_l974_97465


namespace NUMINAMATH_CALUDE_texas_migration_l974_97499

/-- The number of people moving to Texas in four days -/
def people_moving : ℕ := 3600

/-- The number of days -/
def num_days : ℕ := 4

/-- The number of hours in a day -/
def hours_per_day : ℕ := 24

/-- Calculates the average number of people moving per hour -/
def avg_people_per_hour : ℚ :=
  people_moving / (num_days * hours_per_day)

/-- Rounds a rational number to the nearest integer -/
def round_to_nearest (q : ℚ) : ℤ :=
  ⌊q + 1/2⌋

theorem texas_migration :
  round_to_nearest avg_people_per_hour = 38 := by
  sorry

end NUMINAMATH_CALUDE_texas_migration_l974_97499


namespace NUMINAMATH_CALUDE_two_solutions_iff_a_gt_neg_one_l974_97438

/-- The equation has exactly two solutions if and only if a > -1 -/
theorem two_solutions_iff_a_gt_neg_one (a : ℝ) :
  (∃! x y, x ≠ y ∧ x^2 + 2*x + 2*|x+1| = a ∧ y^2 + 2*y + 2*|y+1| = a) ↔ a > -1 := by
  sorry

end NUMINAMATH_CALUDE_two_solutions_iff_a_gt_neg_one_l974_97438


namespace NUMINAMATH_CALUDE_system_one_solution_system_two_solution_l974_97417

-- System 1
theorem system_one_solution (x y : ℚ) : 
  x + y = 4 ∧ 5 * (x - y) - 2 * (x + y) = -1 → x = 27/10 ∧ y = 13/10 := by sorry

-- System 2
theorem system_two_solution (x y : ℚ) :
  2 * (x - y) / 3 - (x + y) / 4 = -1/12 ∧ 3 * (x + y) - 2 * (2 * x - y) = 3 → x = 2 ∧ y = 1 := by sorry

end NUMINAMATH_CALUDE_system_one_solution_system_two_solution_l974_97417


namespace NUMINAMATH_CALUDE_min_value_3m_plus_n_l974_97423

/-- Given a triangle ABC with point G satisfying the centroid condition,
    and points M on AB and N on AC with specific vector relationships,
    prove that the minimum value of 3m + n is 4/3 + 2√3/3 -/
theorem min_value_3m_plus_n (A B C G M N : ℝ × ℝ) (m n : ℝ) :
  (G.1 - A.1 + G.1 - B.1 + G.1 - C.1 = 0 ∧
   G.2 - A.2 + G.2 - B.2 + G.2 - C.2 = 0) →
  (∃ t : ℝ, M = (1 - t) • A + t • B ∧
            N = (1 - t) • A + t • C) →
  (M.1 - A.1 = m * (B.1 - A.1) ∧
   M.2 - A.2 = m * (B.2 - A.2)) →
  (N.1 - A.1 = n * (C.1 - A.1) ∧
   N.2 - A.2 = n * (C.2 - A.2)) →
  m > 0 →
  n > 0 →
  (∀ m' n' : ℝ, m' > 0 → n' > 0 → 3 * m + n ≤ 3 * m' + n') →
  3 * m + n = 4/3 + 2 * Real.sqrt 3 / 3 :=
by sorry

end NUMINAMATH_CALUDE_min_value_3m_plus_n_l974_97423


namespace NUMINAMATH_CALUDE_gcd_75_225_l974_97455

theorem gcd_75_225 : Nat.gcd 75 225 = 75 := by
  sorry

end NUMINAMATH_CALUDE_gcd_75_225_l974_97455


namespace NUMINAMATH_CALUDE_three_balls_four_boxes_l974_97440

theorem three_balls_four_boxes :
  let num_balls : ℕ := 3
  let num_boxes : ℕ := 4
  num_boxes ^ num_balls = 64 :=
by sorry

end NUMINAMATH_CALUDE_three_balls_four_boxes_l974_97440


namespace NUMINAMATH_CALUDE_election_vote_ratio_l974_97418

theorem election_vote_ratio (Vx Vy : ℝ) 
  (h1 : 0.72 * Vx + 0.36 * Vy = 0.6 * (Vx + Vy)) 
  (h2 : Vx > 0) 
  (h3 : Vy > 0) : 
  Vx / Vy = 2 := by
sorry

end NUMINAMATH_CALUDE_election_vote_ratio_l974_97418


namespace NUMINAMATH_CALUDE_kolya_optimal_strategy_l974_97433

/-- Represents the three methods Kolya can choose from -/
inductive Method
  | largest_smallest
  | two_middle
  | choice_with_payment

/-- Represents a division of nuts -/
structure NutDivision where
  a₁ : ℕ
  a₂ : ℕ
  b₁ : ℕ
  b₂ : ℕ

/-- Calculates the number of nuts Kolya gets for a given method and division -/
def nuts_for_kolya (m : Method) (d : NutDivision) : ℕ :=
  match m with
  | Method.largest_smallest => max d.a₁ d.b₁ + min d.a₂ d.b₂
  | Method.two_middle => d.a₁ + d.a₂ + d.b₁ + d.b₂ - (max d.a₁ (max d.a₂ (max d.b₁ d.b₂))) - (min d.a₁ (min d.a₂ (min d.b₁ d.b₂)))
  | Method.choice_with_payment => max (max d.a₁ d.b₁ + min d.a₂ d.b₂) (d.a₁ + d.a₂ + d.b₁ + d.b₂ - (max d.a₁ (max d.a₂ (max d.b₁ d.b₂))) - (min d.a₁ (min d.a₂ (min d.b₁ d.b₂)))) - 1

/-- Theorem stating the existence of most and least advantageous methods for Kolya -/
theorem kolya_optimal_strategy (n : ℕ) (h : n ≥ 2) :
  ∃ (best worst : Method) (d : NutDivision),
    (d.a₁ + d.a₂ + d.b₁ + d.b₂ = 2*n + 1) ∧
    (d.a₁ ≥ 1 ∧ d.a₂ ≥ 1 ∧ d.b₁ ≥ 1 ∧ d.b₂ ≥ 1) ∧
    (∀ m : Method, nuts_for_kolya best d ≥ nuts_for_kolya m d) ∧
    (∀ m : Method, nuts_for_kolya worst d ≤ nuts_for_kolya m d) :=
  sorry

end NUMINAMATH_CALUDE_kolya_optimal_strategy_l974_97433


namespace NUMINAMATH_CALUDE_number_of_values_l974_97421

theorem number_of_values (initial_mean : ℚ) (incorrect_value : ℚ) (correct_value : ℚ) (correct_mean : ℚ) : 
  initial_mean = 250 →
  incorrect_value = 135 →
  correct_value = 165 →
  correct_mean = 251 →
  ∃ n : ℕ, n > 0 ∧ 
    (n : ℚ) * initial_mean + correct_value - incorrect_value = (n : ℚ) * correct_mean ∧
    n = 30 :=
by sorry

end NUMINAMATH_CALUDE_number_of_values_l974_97421


namespace NUMINAMATH_CALUDE_sin_cos_sum_equals_half_l974_97402

theorem sin_cos_sum_equals_half : 
  Real.sin (21 * π / 180) * Real.cos (9 * π / 180) + 
  Real.sin (69 * π / 180) * Real.sin (9 * π / 180) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_sum_equals_half_l974_97402


namespace NUMINAMATH_CALUDE_derivative_at_pi_half_l974_97442

/-- Given a function f where f(x) = sin x + 2x * f'(0), prove that f'(π/2) = -2 -/
theorem derivative_at_pi_half (f : ℝ → ℝ) (hf : ∀ x, f x = Real.sin x + 2 * x * (deriv f 0)) :
  deriv f (π/2) = -2 := by
  sorry

end NUMINAMATH_CALUDE_derivative_at_pi_half_l974_97442


namespace NUMINAMATH_CALUDE_power_of_square_l974_97435

theorem power_of_square (x : ℝ) : (x^2)^3 = x^6 := by
  sorry

end NUMINAMATH_CALUDE_power_of_square_l974_97435


namespace NUMINAMATH_CALUDE_group_age_calculation_l974_97493

theorem group_age_calculation (total_members : ℕ) (total_average_age : ℚ) (zero_age_members : ℕ) : 
  total_members = 50 →
  total_average_age = 5 →
  zero_age_members = 10 →
  let non_zero_members : ℕ := total_members - zero_age_members
  let total_age : ℚ := total_members * total_average_age
  let non_zero_average_age : ℚ := total_age / non_zero_members
  non_zero_average_age = 25/4 := by
sorry

#eval (25 : ℚ) / 4  -- This should output 6.25

end NUMINAMATH_CALUDE_group_age_calculation_l974_97493


namespace NUMINAMATH_CALUDE_valve_flow_rate_difference_l974_97457

/-- The problem of calculating the difference in water flow rates between two valves filling a pool. -/
theorem valve_flow_rate_difference (pool_capacity : ℝ) (both_valves_time : ℝ) (first_valve_time : ℝ) :
  pool_capacity = 12000 ∧ 
  both_valves_time = 48 ∧ 
  first_valve_time = 120 →
  (pool_capacity / both_valves_time) - (pool_capacity / first_valve_time) = 50 := by
sorry

end NUMINAMATH_CALUDE_valve_flow_rate_difference_l974_97457


namespace NUMINAMATH_CALUDE_horner_method_equals_polynomial_f_at_5_equals_4881_l974_97427

def f (x : ℝ) : ℝ := x^5 + 2*x^4 + 3*x^3 + 4*x^2 + 5*x + 6

def horner_method (a : List ℝ) (x : ℝ) : ℝ :=
  a.foldl (fun acc c => acc * x + c) 0

theorem horner_method_equals_polynomial (x : ℝ) :
  horner_method [6, 5, 4, 3, 2, 1] x = f x :=
sorry

theorem f_at_5_equals_4881 :
  f 5 = 4881 :=
sorry

end NUMINAMATH_CALUDE_horner_method_equals_polynomial_f_at_5_equals_4881_l974_97427


namespace NUMINAMATH_CALUDE_bee_paths_count_l974_97449

/-- Represents the number of beehives in the row -/
def n : ℕ := 6

/-- Represents the possible moves of the bee -/
inductive BeeMove
  | Right
  | UpperRight
  | LowerRight

/-- Represents a path of the bee as a list of moves -/
def BeePath := List BeeMove

/-- Checks if a path is valid (ends at hive number 6) -/
def isValidPath (path : BeePath) : Bool :=
  sorry

/-- Counts the number of valid paths to hive number 6 -/
def countValidPaths : ℕ :=
  sorry

/-- Theorem: The number of valid paths to hive number 6 is 21 -/
theorem bee_paths_count : countValidPaths = 21 := by
  sorry

end NUMINAMATH_CALUDE_bee_paths_count_l974_97449


namespace NUMINAMATH_CALUDE_total_fish_l974_97479

theorem total_fish (lilly_fish rosy_fish : ℕ) 
  (h1 : lilly_fish = 10) 
  (h2 : rosy_fish = 8) : 
  lilly_fish + rosy_fish = 18 := by
  sorry

end NUMINAMATH_CALUDE_total_fish_l974_97479


namespace NUMINAMATH_CALUDE_unique_assignment_l974_97495

-- Define the students and authors as enums
inductive Student : Type
| ZhangBoyuan : Student
| GaoJiaming : Student
| LiuYuheng : Student

inductive Author : Type
| Shakespeare : Author
| Hugo : Author
| CaoXueqin : Author

-- Define the assignment of authors to students
def Assignment := Student → Author

-- Define the condition that each student has a different author
def all_different (a : Assignment) : Prop :=
  ∀ s1 s2 : Student, s1 ≠ s2 → a s1 ≠ a s2

-- Define Teacher Liu's guesses
def guess1 (a : Assignment) : Prop := a Student.ZhangBoyuan = Author.Shakespeare
def guess2 (a : Assignment) : Prop := a Student.LiuYuheng ≠ Author.CaoXueqin
def guess3 (a : Assignment) : Prop := a Student.GaoJiaming ≠ Author.Shakespeare

-- Define the condition that only one guess is correct
def only_one_correct (a : Assignment) : Prop :=
  (guess1 a ∧ ¬guess2 a ∧ ¬guess3 a) ∨
  (¬guess1 a ∧ guess2 a ∧ ¬guess3 a) ∨
  (¬guess1 a ∧ ¬guess2 a ∧ guess3 a)

-- The main theorem
theorem unique_assignment :
  ∃! a : Assignment,
    all_different a ∧
    only_one_correct a ∧
    a Student.ZhangBoyuan = Author.CaoXueqin ∧
    a Student.GaoJiaming = Author.Shakespeare ∧
    a Student.LiuYuheng = Author.Hugo :=
  sorry

end NUMINAMATH_CALUDE_unique_assignment_l974_97495


namespace NUMINAMATH_CALUDE_recurring_decimal_sum_l974_97431

theorem recurring_decimal_sum : 
  (2 : ℚ) / 3 + 7 / 9 = 13 / 9 := by sorry

end NUMINAMATH_CALUDE_recurring_decimal_sum_l974_97431


namespace NUMINAMATH_CALUDE_initial_average_age_l974_97419

theorem initial_average_age (n : ℕ) (new_person_age : ℕ) (new_average : ℚ) :
  n = 17 ∧ new_person_age = 32 ∧ new_average = 15 →
  ∃ initial_average : ℚ, 
    initial_average * n + new_person_age = new_average * (n + 1) ∧
    initial_average = 14 :=
by sorry

end NUMINAMATH_CALUDE_initial_average_age_l974_97419


namespace NUMINAMATH_CALUDE_supplementary_angles_ratio_l974_97434

theorem supplementary_angles_ratio (a b : ℝ) : 
  a + b = 180 →  -- The angles are supplementary
  a / b = 4 / 5 →  -- The ratio of the angles is 4:5
  b = 100 :=  -- The larger angle is 100°
by sorry

end NUMINAMATH_CALUDE_supplementary_angles_ratio_l974_97434


namespace NUMINAMATH_CALUDE_age_difference_l974_97460

theorem age_difference : ∃ (a b : ℕ), 
  (a ≤ 9 ∧ b ≤ 9) ∧ 
  (10 * a + b + 5 = 2 * (10 * b + a + 5)) ∧
  ((10 * a + b) - (10 * b + a) = 18) := by
  sorry

end NUMINAMATH_CALUDE_age_difference_l974_97460


namespace NUMINAMATH_CALUDE_stacy_berries_l974_97488

theorem stacy_berries (skylar_berries steve_berries stacy_berries : ℕ) : 
  skylar_berries = 20 →
  steve_berries = skylar_berries / 2 →
  stacy_berries = 3 * steve_berries + 2 →
  stacy_berries = 32 := by
  sorry

end NUMINAMATH_CALUDE_stacy_berries_l974_97488


namespace NUMINAMATH_CALUDE_eulers_formula_modulus_l974_97420

theorem eulers_formula_modulus (i : ℂ) (π : ℝ) : 
  Complex.abs (Complex.exp (i * π / 3)) = 1 :=
by
  sorry

end NUMINAMATH_CALUDE_eulers_formula_modulus_l974_97420
