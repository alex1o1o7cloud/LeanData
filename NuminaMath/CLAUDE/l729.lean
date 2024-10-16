import Mathlib

namespace NUMINAMATH_CALUDE_wire_cutting_problem_l729_72982

theorem wire_cutting_problem :
  let total_length : ℕ := 102
  let piece_length_1 : ℕ := 15
  let piece_length_2 : ℕ := 12
  ∀ x y : ℕ, piece_length_1 * x + piece_length_2 * y = total_length →
    (x = 2 ∧ y = 6) ∨ (x = 6 ∧ y = 1) :=
by sorry

end NUMINAMATH_CALUDE_wire_cutting_problem_l729_72982


namespace NUMINAMATH_CALUDE_unknown_rate_is_225_l729_72932

/-- The unknown rate of two blankets given the following conditions:
    - 3 blankets at Rs. 100 each
    - 6 blankets at Rs. 150 each
    - 2 blankets at an unknown rate
    - The average price of all blankets is Rs. 150
-/
def unknown_rate : ℕ := by
  -- Define the known quantities
  let blankets_100 : ℕ := 3
  let price_100 : ℕ := 100
  let blankets_150 : ℕ := 6
  let price_150 : ℕ := 150
  let blankets_unknown : ℕ := 2
  let average_price : ℕ := 150

  -- Calculate the total number of blankets
  let total_blankets : ℕ := blankets_100 + blankets_150 + blankets_unknown

  -- Calculate the total cost of all blankets
  let total_cost : ℕ := average_price * total_blankets

  -- Calculate the cost of known blankets
  let cost_known : ℕ := blankets_100 * price_100 + blankets_150 * price_150

  -- Calculate the cost of unknown blankets
  let cost_unknown : ℕ := total_cost - cost_known

  -- Calculate the rate of each unknown blanket
  exact cost_unknown / blankets_unknown

theorem unknown_rate_is_225 : unknown_rate = 225 := by
  sorry

end NUMINAMATH_CALUDE_unknown_rate_is_225_l729_72932


namespace NUMINAMATH_CALUDE_ellipse_chord_slope_l729_72934

/-- Given an ellipse defined by 4x^2 + 9y^2 = 144 and a point P(3, 2) inside it,
    the slope of the line containing the chord with P as its midpoint is -2/3 -/
theorem ellipse_chord_slope (x y : ℝ) :
  4 * x^2 + 9 * y^2 = 144 →  -- Ellipse equation
  ∃ (x1 y1 x2 y2 : ℝ),       -- Endpoints of the chord
    4 * x1^2 + 9 * y1^2 = 144 ∧   -- First endpoint on ellipse
    4 * x2^2 + 9 * y2^2 = 144 ∧   -- Second endpoint on ellipse
    (x1 + x2) / 2 = 3 ∧           -- P is midpoint (x-coordinate)
    (y1 + y2) / 2 = 2 →           -- P is midpoint (y-coordinate)
    (y2 - y1) / (x2 - x1) = -2/3  -- Slope of the chord
:= by sorry

end NUMINAMATH_CALUDE_ellipse_chord_slope_l729_72934


namespace NUMINAMATH_CALUDE_no_5_6_8_multiplier_l729_72927

/-- Function to get the number of digits in a positive integer -/
def num_digits (n : ℕ) : ℕ :=
  if n < 10 then 1 else 1 + num_digits (n / 10)

/-- Function to get the leading digit of a positive integer -/
def leading_digit (n : ℕ) : ℕ :=
  if n < 10 then n else leading_digit (n / 10)

/-- Function to move the leading digit to the end -/
def move_leading_digit (n : ℕ) : ℕ :=
  let d := num_digits n
  let lead := leading_digit n
  (n - lead * 10^(d-1)) * 10 + lead

/-- Theorem stating that no integer becomes 5, 6, or 8 times larger when its leading digit is moved to the end -/
theorem no_5_6_8_multiplier (n : ℕ) (h : n ≥ 10) : 
  let m := move_leading_digit n
  m ≠ 5*n ∧ m ≠ 6*n ∧ m ≠ 8*n :=
sorry

end NUMINAMATH_CALUDE_no_5_6_8_multiplier_l729_72927


namespace NUMINAMATH_CALUDE_function_value_order_l729_72999

-- Define the function f
def f (a x : ℝ) : ℝ := -x^2 + 6*x + a^2 - 1

-- State the theorem
theorem function_value_order (a : ℝ) : 
  f a (Real.sqrt 2) < f a 4 ∧ f a 4 < f a 3 := by
  sorry

end NUMINAMATH_CALUDE_function_value_order_l729_72999


namespace NUMINAMATH_CALUDE_bacteria_fill_count_l729_72910

/-- Fibonacci sequence -/
def fib : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | n + 2 => fib (n + 1) + fib n

/-- Number of ways to fill a table with bacteria -/
def bacteriaFillWays (m n : ℕ) : ℕ :=
  2^(n-1) * (fib (2*n+1))^(m-1)

/-- Theorem: The number of ways to fill an m×n table with non-overlapping bacteria -/
theorem bacteria_fill_count (m n : ℕ) :
  bacteriaFillWays m n = 2^(n-1) * (fib (2*n+1))^(m-1) :=
by
  sorry

/-- Property: Bacteria have horizontal bodies of natural length -/
axiom bacteria_body_natural_length : True

/-- Property: Bacteria have nonnegative number of vertical feet -/
axiom bacteria_feet_nonnegative : True

/-- Property: Bacteria feet have nonnegative natural length -/
axiom bacteria_feet_natural_length : True

/-- Property: Bacteria do not overlap in the table -/
axiom bacteria_no_overlap : True

end NUMINAMATH_CALUDE_bacteria_fill_count_l729_72910


namespace NUMINAMATH_CALUDE_binomial_coefficient_equality_l729_72979

theorem binomial_coefficient_equality (n : ℕ+) :
  (Nat.choose n 2 = Nat.choose (n - 1) 2 + Nat.choose (n - 1) 3) → n = 5 := by
  sorry

end NUMINAMATH_CALUDE_binomial_coefficient_equality_l729_72979


namespace NUMINAMATH_CALUDE_tom_weekly_fee_l729_72947

/-- Represents Tom's car leasing scenario -/
structure CarLease where
  miles_mon_wed_fri : ℕ  -- Miles driven on Monday, Wednesday, Friday
  miles_other_days : ℕ   -- Miles driven on other days
  cost_per_mile : ℚ      -- Cost per mile in dollars
  total_annual_payment : ℚ -- Total annual payment in dollars

/-- Calculate the weekly fee given a car lease scenario -/
def weekly_fee (lease : CarLease) : ℚ :=
  let weekly_miles := 3 * lease.miles_mon_wed_fri + 4 * lease.miles_other_days
  let weekly_mileage_cost := weekly_miles * lease.cost_per_mile
  let annual_mileage_cost := 52 * weekly_mileage_cost
  (lease.total_annual_payment - annual_mileage_cost) / 52

/-- Theorem stating that the weekly fee for Tom's scenario is $95 -/
theorem tom_weekly_fee :
  let tom_lease := CarLease.mk 50 100 (1/10) 7800
  weekly_fee tom_lease = 95 := by sorry

end NUMINAMATH_CALUDE_tom_weekly_fee_l729_72947


namespace NUMINAMATH_CALUDE_cube_root_inequality_l729_72941

theorem cube_root_inequality (x : ℝ) (h : x > 0) :
  Real.rpow x (1/3) < 3 - x ↔ x < 3 := by sorry

end NUMINAMATH_CALUDE_cube_root_inequality_l729_72941


namespace NUMINAMATH_CALUDE_inequality_proof_l729_72913

theorem inequality_proof (a b c d : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) :
  (a / (b + 2*c + 3*d)) + (b / (c + 2*d + 3*a)) + (c / (d + 2*a + 3*b)) + (d / (a + 2*b + 3*c)) ≥ 2/3 :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_l729_72913


namespace NUMINAMATH_CALUDE_prob_same_number_four_dice_l729_72944

/-- The number of sides on a standard die -/
def standard_die_sides : ℕ := 6

/-- The number of dice being tossed -/
def num_dice : ℕ := 4

/-- The probability of getting the same number on all dice -/
def prob_same_number : ℚ := 1 / (standard_die_sides ^ (num_dice - 1))

/-- Theorem: The probability of getting the same number on all four standard six-sided dice is 1/216 -/
theorem prob_same_number_four_dice : 
  prob_same_number = 1 / 216 := by sorry

end NUMINAMATH_CALUDE_prob_same_number_four_dice_l729_72944


namespace NUMINAMATH_CALUDE_line_intersects_extension_l729_72978

/-- Given a line l: Ax + By + C = 0 and two points P₁ and P₂, 
    prove that l intersects with the extension of P₁P₂ under certain conditions. -/
theorem line_intersects_extension (A B C x₁ y₁ x₂ y₂ : ℝ) 
  (hAB : A ≠ 0 ∨ B ≠ 0)
  (hSameSide : (A * x₁ + B * y₁ + C) * (A * x₂ + B * y₂ + C) > 0)
  (hDistance : |A * x₁ + B * y₁ + C| > |A * x₂ + B * y₂ + C|) :
  ∃ (t : ℝ), t > 1 ∧ A * (x₁ + t * (x₂ - x₁)) + B * (y₁ + t * (y₂ - y₁)) + C = 0 :=
sorry

end NUMINAMATH_CALUDE_line_intersects_extension_l729_72978


namespace NUMINAMATH_CALUDE_largest_integer_less_than_100_with_remainder_5_mod_8_l729_72931

theorem largest_integer_less_than_100_with_remainder_5_mod_8 :
  ∀ n : ℕ, n < 100 → n % 8 = 5 → n ≤ 93 :=
by
  sorry

end NUMINAMATH_CALUDE_largest_integer_less_than_100_with_remainder_5_mod_8_l729_72931


namespace NUMINAMATH_CALUDE_parallelogram_base_length_l729_72987

/-- Given a parallelogram with area 44 cm² and height 11 cm, its base length is 4 cm. -/
theorem parallelogram_base_length (area : ℝ) (height : ℝ) (base : ℝ) 
    (h1 : area = 44) 
    (h2 : height = 11) 
    (h3 : area = base * height) : base = 4 := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_base_length_l729_72987


namespace NUMINAMATH_CALUDE_negation_equivalence_l729_72953

theorem negation_equivalence : 
  (¬ ∃ x₀ : ℝ, x₀^2 - x₀ + 1 ≤ 0) ↔ (∀ x : ℝ, x^2 - x + 1 > 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_equivalence_l729_72953


namespace NUMINAMATH_CALUDE_tangent_product_simplification_l729_72956

theorem tangent_product_simplification :
  (1 + Real.tan (15 * π / 180)) * (1 + Real.tan (30 * π / 180)) = 2 := by
  sorry

end NUMINAMATH_CALUDE_tangent_product_simplification_l729_72956


namespace NUMINAMATH_CALUDE_triangle_positive_number_placement_l729_72914

theorem triangle_positive_number_placement 
  (A B C : ℝ × ℝ) -- Vertices of the triangle
  (AB BC CA : ℝ)  -- Lengths of the sides
  (h_pos_AB : AB > 0)
  (h_pos_BC : BC > 0)
  (h_pos_CA : CA > 0)
  (h_triangle : AB + BC > CA ∧ BC + CA > AB ∧ CA + AB > BC) -- Triangle inequality
  : ∃ x y z : ℝ, 
    x > 0 ∧ y > 0 ∧ z > 0 ∧
    AB = x + y ∧
    BC = y + z ∧
    CA = z + x :=
sorry

end NUMINAMATH_CALUDE_triangle_positive_number_placement_l729_72914


namespace NUMINAMATH_CALUDE_soma_cube_combination_l729_72936

/-- Represents a Soma cube piece -/
structure SomaPiece where
  number : Nat
  cubes : Nat

/-- The set of all Soma cube pieces -/
def somaPieces : List SomaPiece :=
  [⟨1, 3⟩, ⟨2, 4⟩, ⟨3, 4⟩, ⟨4, 4⟩, ⟨5, 4⟩, ⟨6, 4⟩, ⟨7, 4⟩]

/-- Checks if a pair of Soma pieces forms a valid combination -/
def isValidCombination (p1 p2 : SomaPiece) : Prop :=
  p1.number ≠ p2.number ∧ p1.cubes + p2.cubes = 8

/-- The theorem to be proved -/
theorem soma_cube_combination :
  ∀ p1 p2 : SomaPiece,
    p1 ∈ somaPieces → p2 ∈ somaPieces →
    isValidCombination p1 p2 →
    ((p1.number = 2 ∧ p2.number = 5) ∨ (p1.number = 2 ∧ p2.number = 6) ∨
     (p1.number = 5 ∧ p2.number = 2) ∨ (p1.number = 6 ∧ p2.number = 2)) :=
by sorry

end NUMINAMATH_CALUDE_soma_cube_combination_l729_72936


namespace NUMINAMATH_CALUDE_divisibility_by_120_l729_72957

theorem divisibility_by_120 (n : ℕ) : ∃ k : ℤ, (n ^ 7 : ℤ) - (n ^ 3 : ℤ) = 120 * k := by
  sorry

end NUMINAMATH_CALUDE_divisibility_by_120_l729_72957


namespace NUMINAMATH_CALUDE_stripe_area_on_cylinder_l729_72973

/-- The area of a stripe on a cylindrical tank -/
theorem stripe_area_on_cylinder (diameter : Real) (stripe_width : Real) (revolutions : Real) :
  diameter = 40 ∧ stripe_width = 4 ∧ revolutions = 3 →
  stripe_width * revolutions * π * diameter = 480 * π := by
  sorry

end NUMINAMATH_CALUDE_stripe_area_on_cylinder_l729_72973


namespace NUMINAMATH_CALUDE_tangent_line_at_point_l729_72963

-- Define the curve
def f (x : ℝ) : ℝ := x^3 - 3*x

-- Define the derivative of the curve
def f' (x : ℝ) : ℝ := 3*x^2 - 3

-- Define the point of tangency
def point : ℝ × ℝ := (2, 2)

-- Define the two possible tangent lines
def line1 (x : ℝ) : ℝ := 2
def line2 (x : ℝ) : ℝ := 9*x - 16

theorem tangent_line_at_point :
  (∀ x, line1 x = f x → x = 2) ∧
  (∀ x, line2 x = f x → x = 2) ∧
  line1 (point.1) = point.2 ∧
  line2 (point.1) = point.2 ∧
  f' (point.1) = 9 :=
sorry

end NUMINAMATH_CALUDE_tangent_line_at_point_l729_72963


namespace NUMINAMATH_CALUDE_balls_given_to_partner_l729_72939

/-- Represents the number of tennis games played by Bertha -/
def games : ℕ := 20

/-- Represents the number of games after which one ball wears out -/
def wear_out_rate : ℕ := 10

/-- Represents the number of games after which Bertha loses a ball -/
def lose_rate : ℕ := 5

/-- Represents the number of games after which Bertha buys a canister of balls -/
def buy_rate : ℕ := 4

/-- Represents the number of balls in each canister -/
def balls_per_canister : ℕ := 3

/-- Represents the number of balls Bertha started with -/
def initial_balls : ℕ := 2

/-- Represents the number of balls Bertha has after 20 games -/
def final_balls : ℕ := 10

/-- Calculates the number of balls worn out during the games -/
def balls_worn_out : ℕ := games / wear_out_rate

/-- Calculates the number of balls lost during the games -/
def balls_lost : ℕ := games / lose_rate

/-- Calculates the number of balls bought during the games -/
def balls_bought : ℕ := (games / buy_rate) * balls_per_canister

/-- Theorem stating that Bertha gave 1 ball to her partner -/
theorem balls_given_to_partner :
  initial_balls + balls_bought - balls_worn_out - balls_lost - final_balls = 1 := by
  sorry

end NUMINAMATH_CALUDE_balls_given_to_partner_l729_72939


namespace NUMINAMATH_CALUDE_largest_divisor_of_expression_l729_72962

theorem largest_divisor_of_expression (x : ℤ) (h : Even x) :
  (∃ (k : ℤ), (10*x + 1) * (10*x + 5) * (5*x + 3) = 3 * k) ∧
  (∀ (d : ℤ), d > 3 → ∃ (y : ℤ), Even y ∧ ¬(∃ (k : ℤ), (10*y + 1) * (10*y + 5) * (5*y + 3) = d * k)) :=
by sorry

end NUMINAMATH_CALUDE_largest_divisor_of_expression_l729_72962


namespace NUMINAMATH_CALUDE_yw_equals_five_l729_72917

/-- Triangle with sides a, b, c --/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Point on the perimeter of a triangle --/
structure PerimeterPoint where
  distanceFromY : ℝ

/-- Definition of the meeting point of two ants crawling from X in opposite directions --/
def meetingPoint (t : Triangle) : PerimeterPoint :=
  { distanceFromY := 5 }

/-- Theorem stating that YW = 5 for the given triangle and ant movement --/
theorem yw_equals_five (t : Triangle) 
    (h1 : t.a = 7) 
    (h2 : t.b = 8) 
    (h3 : t.c = 9) : 
  (meetingPoint t).distanceFromY = 5 := by
  sorry

end NUMINAMATH_CALUDE_yw_equals_five_l729_72917


namespace NUMINAMATH_CALUDE_total_rectangles_count_l729_72919

/-- Represents the number of rectangles a single cell can form -/
structure CellRectangles where
  count : Nat

/-- Represents a group of cells with the same rectangle-forming property -/
structure CellGroup where
  cells : Nat
  rectangles : CellRectangles

/-- Calculates the total number of rectangles for a cell group -/
def totalRectangles (group : CellGroup) : Nat :=
  group.cells * group.rectangles.count

/-- The main theorem stating the total number of rectangles -/
theorem total_rectangles_count 
  (total_cells : Nat)
  (group1 : CellGroup)
  (group2 : CellGroup)
  (h1 : total_cells = group1.cells + group2.cells)
  (h2 : total_cells = 40)
  (h3 : group1.cells = 36)
  (h4 : group1.rectangles.count = 4)
  (h5 : group2.cells = 4)
  (h6 : group2.rectangles.count = 8) :
  totalRectangles group1 + totalRectangles group2 = 176 := by
  sorry

#check total_rectangles_count

end NUMINAMATH_CALUDE_total_rectangles_count_l729_72919


namespace NUMINAMATH_CALUDE_students_taking_neither_music_nor_art_l729_72904

theorem students_taking_neither_music_nor_art 
  (total : ℕ) (music : ℕ) (art : ℕ) (both : ℕ) : 
  total = 500 → music = 30 → art = 20 → both = 10 →
  total - (music + art - both) = 460 := by
sorry

end NUMINAMATH_CALUDE_students_taking_neither_music_nor_art_l729_72904


namespace NUMINAMATH_CALUDE_power_product_equals_four_l729_72954

theorem power_product_equals_four : 4^2020 * (1/4)^2019 = 4 := by
  sorry

end NUMINAMATH_CALUDE_power_product_equals_four_l729_72954


namespace NUMINAMATH_CALUDE_fraction_equality_l729_72993

theorem fraction_equality (m n p q : ℚ) 
  (h1 : m / n = 18)
  (h2 : p / n = 9)
  (h3 : p / q = 1 / 15) :
  m / q = 2 / 15 := by
sorry

end NUMINAMATH_CALUDE_fraction_equality_l729_72993


namespace NUMINAMATH_CALUDE_exam_average_problem_l729_72942

theorem exam_average_problem :
  ∀ (N : ℕ),
  (15 : ℝ) * 70 + (10 : ℝ) * 95 = (N : ℝ) * 80 →
  N = 25 :=
by
  sorry

end NUMINAMATH_CALUDE_exam_average_problem_l729_72942


namespace NUMINAMATH_CALUDE_slowest_racer_time_l729_72923

/-- Represents the time taken by each person to reach the top floor -/
structure RaceTime where
  lola : ℕ
  tara : ℕ
  sam : ℕ

/-- Calculates the race times given the building parameters -/
def calculateRaceTimes (
  totalStories : ℕ
  ) (lolaTimePerStory : ℕ
  ) (samTimePerStory : ℕ
  ) (elevatorTimePerStory : ℕ
  ) (elevatorStopTime : ℕ
  ) (samSwitchFloor : ℕ
  ) (elevatorWaitTime : ℕ
  ) : RaceTime :=
  { lola := totalStories * lolaTimePerStory,
    tara := totalStories * elevatorTimePerStory + (totalStories - 1) * elevatorStopTime,
    sam := samSwitchFloor * samTimePerStory + elevatorWaitTime +
           (totalStories - samSwitchFloor) * elevatorTimePerStory +
           (totalStories - samSwitchFloor - 1) * elevatorStopTime }

/-- The main theorem to prove -/
theorem slowest_racer_time (
  totalStories : ℕ
  ) (lolaTimePerStory : ℕ
  ) (samTimePerStory : ℕ
  ) (elevatorTimePerStory : ℕ
  ) (elevatorStopTime : ℕ
  ) (samSwitchFloor : ℕ
  ) (elevatorWaitTime : ℕ
  ) (h1 : totalStories = 50
  ) (h2 : lolaTimePerStory = 12
  ) (h3 : samTimePerStory = 15
  ) (h4 : elevatorTimePerStory = 10
  ) (h5 : elevatorStopTime = 4
  ) (h6 : samSwitchFloor = 25
  ) (h7 : elevatorWaitTime = 20
  ) : (
    let times := calculateRaceTimes totalStories lolaTimePerStory samTimePerStory
                   elevatorTimePerStory elevatorStopTime samSwitchFloor elevatorWaitTime
    max times.lola (max times.tara times.sam) = 741
  ) := by
  sorry

end NUMINAMATH_CALUDE_slowest_racer_time_l729_72923


namespace NUMINAMATH_CALUDE_marble_probability_l729_72938

/-- The probability of drawing either a green or black marble from a bag -/
theorem marble_probability (green black white : ℕ) 
  (h_green : green = 4)
  (h_black : black = 3)
  (h_white : white = 6) :
  (green + black : ℚ) / (green + black + white) = 7 / 13 := by
  sorry

end NUMINAMATH_CALUDE_marble_probability_l729_72938


namespace NUMINAMATH_CALUDE_triangle_equality_l729_72995

/-- Given a triangle ABC with sides a, b, c opposite to angles α, β, γ respectively,
    and circumradius R, prove that if the given equation holds, then the triangle is equilateral. -/
theorem triangle_equality (a b c R : ℝ) (α β γ : ℝ) : 
  a > 0 → b > 0 → c > 0 → R > 0 →
  0 < α ∧ α < π → 0 < β ∧ β < π → 0 < γ ∧ γ < π →
  α + β + γ = π →
  (a * Real.cos α + b * Real.cos β + c * Real.cos γ) / (a * Real.sin β + b * Real.sin γ + c * Real.sin α) = (a + b + c) / (9 * R) →
  α = β ∧ β = γ ∧ γ = π / 3 := by
  sorry

end NUMINAMATH_CALUDE_triangle_equality_l729_72995


namespace NUMINAMATH_CALUDE_lcm_45_75_l729_72906

theorem lcm_45_75 : Nat.lcm 45 75 = 225 := by sorry

end NUMINAMATH_CALUDE_lcm_45_75_l729_72906


namespace NUMINAMATH_CALUDE_elevator_capacity_l729_72959

/-- Proves that the number of people in an elevator is 20, given the weight limit,
    average weight, and excess weight. -/
theorem elevator_capacity
  (weight_limit : ℝ)
  (average_weight : ℝ)
  (excess_weight : ℝ)
  (h1 : weight_limit = 1500)
  (h2 : average_weight = 80)
  (h3 : excess_weight = 100)
  : (weight_limit + excess_weight) / average_weight = 20 := by
  sorry

#check elevator_capacity

end NUMINAMATH_CALUDE_elevator_capacity_l729_72959


namespace NUMINAMATH_CALUDE_geometric_arithmetic_sequence_l729_72996

theorem geometric_arithmetic_sequence (a : ℕ → ℝ) (q : ℝ) :
  (∀ n, a (n + 1) = q * a n) →  -- geometric sequence condition
  (4 * a 1 - a 3 = a 3 - 2 * a 2) →  -- arithmetic sequence condition
  q = -1 ∨ q = 2 := by
sorry

end NUMINAMATH_CALUDE_geometric_arithmetic_sequence_l729_72996


namespace NUMINAMATH_CALUDE_matrix_equation_solution_l729_72981

def B : Matrix (Fin 3) (Fin 3) ℤ := !![0, -1, 1; -1, 2, -1; 1, -1, 0]

theorem matrix_equation_solution :
  ∃ (s t u : ℤ), 
    B^3 + s • B^2 + t • B + u • (1 : Matrix (Fin 3) (Fin 3) ℤ) = 0 ∧ 
    s = -1 ∧ t = 0 ∧ u = 2 := by
  sorry

end NUMINAMATH_CALUDE_matrix_equation_solution_l729_72981


namespace NUMINAMATH_CALUDE_min_angle_line_equation_l729_72974

/-- A circle in the 2D plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- A line in the 2D plane -/
structure Line where
  slope : ℝ
  yIntercept : ℝ

/-- The angle between two points and a center point -/
def angle (center : ℝ × ℝ) (p1 : ℝ × ℝ) (p2 : ℝ × ℝ) : ℝ := sorry

/-- Check if a line intersects a circle at two points -/
def intersectsCircle (l : Line) (c : Circle) : Prop := sorry

/-- Check if a point lies on a line -/
def pointOnLine (p : ℝ × ℝ) (l : Line) : Prop := sorry

/-- The main theorem -/
theorem min_angle_line_equation 
  (c : Circle)
  (m : ℝ × ℝ)
  (l : Line) :
  c.center = (3, 4) →
  c.radius = 5 →
  m = (1, 2) →
  pointOnLine m l →
  intersectsCircle l c →
  (∀ l' : Line, intersectsCircle l' c → angle c.center (1, 2) (3, 4) ≤ angle c.center (1, 2) (3, 4)) →
  l.slope = -1 ∧ l.yIntercept = 3 :=
sorry

end NUMINAMATH_CALUDE_min_angle_line_equation_l729_72974


namespace NUMINAMATH_CALUDE_soda_price_calculation_soda_price_proof_l729_72971

/-- Given the cost of sandwiches and total cost, calculate the price of each soda -/
theorem soda_price_calculation (sandwich_price : ℚ) (num_sandwiches : ℕ) (num_sodas : ℕ) (total_cost : ℚ) : ℚ :=
  let sandwich_total := sandwich_price * num_sandwiches
  let soda_total := total_cost - sandwich_total
  soda_total / num_sodas

/-- Prove that the price of each soda is $1.87 given the problem conditions -/
theorem soda_price_proof :
  soda_price_calculation 2.49 2 4 12.46 = 1.87 := by
  sorry

end NUMINAMATH_CALUDE_soda_price_calculation_soda_price_proof_l729_72971


namespace NUMINAMATH_CALUDE_sum_leq_fourth_powers_over_product_l729_72980

theorem sum_leq_fourth_powers_over_product (a b c : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) : 
  a + b + c ≤ (a^4 + b^4 + c^4) / (a * b * c) := by
  sorry

end NUMINAMATH_CALUDE_sum_leq_fourth_powers_over_product_l729_72980


namespace NUMINAMATH_CALUDE_lily_pad_growth_rate_l729_72935

/-- The growth rate of a lily pad patch in a lake -/
def growth_rate (full_coverage_days : ℕ) (half_coverage_days : ℕ) : ℝ :=
  -- Definition to be proved
  1

/-- Theorem stating the growth rate of the lily pad patch -/
theorem lily_pad_growth_rate :
  growth_rate 34 33 = 1 := by
  sorry

#check lily_pad_growth_rate

end NUMINAMATH_CALUDE_lily_pad_growth_rate_l729_72935


namespace NUMINAMATH_CALUDE_square_dissection_l729_72908

/-- Given two squares with side lengths a and b respectively, prove that:
    1. The square with side a can be dissected into 3 identical squares.
    2. The square with side b can be dissected into 7 identical squares. -/
theorem square_dissection (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  ∃ (sa sb : ℝ),
    sa = a / Real.sqrt 3 ∧
    sb = b / Real.sqrt 7 ∧
    3 * sa ^ 2 = a ^ 2 ∧
    7 * sb ^ 2 = b ^ 2 :=
by sorry

end NUMINAMATH_CALUDE_square_dissection_l729_72908


namespace NUMINAMATH_CALUDE_upstream_speed_is_27_l729_72970

/-- Represents the speed of a man rowing in different conditions -/
structure RowingSpeed where
  downstream : ℝ
  stillWater : ℝ
  upstream : ℝ

/-- Calculates the upstream speed given downstream and still water speeds -/
def calculateUpstreamSpeed (downstream stillWater : ℝ) : ℝ :=
  2 * stillWater - downstream

/-- Theorem stating that given the conditions, the upstream speed is 27 kmph -/
theorem upstream_speed_is_27 (speed : RowingSpeed) 
    (h1 : speed.downstream = 35)
    (h2 : speed.stillWater = 31) :
    speed.upstream = 27 :=
  by sorry

end NUMINAMATH_CALUDE_upstream_speed_is_27_l729_72970


namespace NUMINAMATH_CALUDE_cookie_box_duration_l729_72924

/-- Proves that a box of cookies lasts 9 days given the specified conditions -/
theorem cookie_box_duration (oldest_son_cookies : ℕ) (youngest_son_cookies : ℕ) (total_cookies : ℕ) : 
  oldest_son_cookies = 4 → 
  youngest_son_cookies = 2 → 
  total_cookies = 54 → 
  (total_cookies / (oldest_son_cookies + youngest_son_cookies) : ℕ) = 9 := by
  sorry

#check cookie_box_duration

end NUMINAMATH_CALUDE_cookie_box_duration_l729_72924


namespace NUMINAMATH_CALUDE_power_of_three_mod_ten_l729_72989

theorem power_of_three_mod_ten (k : ℕ) : 3^(4*k + 3) % 10 = 7 := by
  sorry

end NUMINAMATH_CALUDE_power_of_three_mod_ten_l729_72989


namespace NUMINAMATH_CALUDE_power_eight_mod_five_l729_72994

theorem power_eight_mod_five : 8^2023 % 5 = 2 := by
  sorry

end NUMINAMATH_CALUDE_power_eight_mod_five_l729_72994


namespace NUMINAMATH_CALUDE_round_robin_tournament_matches_l729_72965

theorem round_robin_tournament_matches (n : ℕ) (h : n = 6) : 
  (n * (n - 1)) / 2 = Nat.choose n 2 := by
  sorry

end NUMINAMATH_CALUDE_round_robin_tournament_matches_l729_72965


namespace NUMINAMATH_CALUDE_least_common_multiple_of_2_3_4_5_6_8_l729_72937

theorem least_common_multiple_of_2_3_4_5_6_8 : ∃ n : ℕ, 
  (∀ m : ℕ, m > 0 ∧ 2 ∣ m ∧ 3 ∣ m ∧ 4 ∣ m ∧ 5 ∣ m ∧ 6 ∣ m ∧ 8 ∣ m → n ≤ m) ∧
  2 ∣ n ∧ 3 ∣ n ∧ 4 ∣ n ∧ 5 ∣ n ∧ 6 ∣ n ∧ 8 ∣ n ∧
  n = 120 :=
by sorry

end NUMINAMATH_CALUDE_least_common_multiple_of_2_3_4_5_6_8_l729_72937


namespace NUMINAMATH_CALUDE_inequality_relationship_l729_72920

theorem inequality_relationship (a b : ℝ) (h : a < 1 / b) :
  (a > 0 ∧ b > 0 → 1 / a > b) ∧
  (a < 0 ∧ b < 0 → 1 / a > b) ∧
  (a < 0 ∧ b > 0 → 1 / a < b) :=
by sorry

end NUMINAMATH_CALUDE_inequality_relationship_l729_72920


namespace NUMINAMATH_CALUDE_janet_key_search_time_l729_72997

/-- The number of minutes Janet spends looking for her keys every day -/
def key_search_time : ℝ := 8

/-- The number of minutes Janet spends complaining after finding her keys -/
def complain_time : ℝ := 3

/-- The number of days in a week -/
def days_per_week : ℝ := 7

/-- The number of minutes Janet would save per week if she stops losing her keys -/
def time_saved_per_week : ℝ := 77

theorem janet_key_search_time :
  key_search_time = (time_saved_per_week - days_per_week * complain_time) / days_per_week :=
by sorry

end NUMINAMATH_CALUDE_janet_key_search_time_l729_72997


namespace NUMINAMATH_CALUDE_solution_set_f_geq_2_range_of_a_for_full_solution_set_l729_72964

-- Define the function f
def f (x : ℝ) : ℝ := |x + 1| - |x - 2|

-- Theorem for part I
theorem solution_set_f_geq_2 :
  {x : ℝ | f x ≥ 2} = {x : ℝ | x ≥ 3/2} :=
sorry

-- Theorem for part II
theorem range_of_a_for_full_solution_set :
  {a : ℝ | ∀ x, f x ≤ |a - 2|} = {a : ℝ | a ≤ -1 ∨ a ≥ 5} :=
sorry

end NUMINAMATH_CALUDE_solution_set_f_geq_2_range_of_a_for_full_solution_set_l729_72964


namespace NUMINAMATH_CALUDE_sum_remainder_l729_72921

theorem sum_remainder (x y z : ℕ) 
  (hx : x % 53 = 31) 
  (hy : y % 53 = 45) 
  (hz : z % 53 = 6) : 
  (x + y + z) % 53 = 29 := by
sorry

end NUMINAMATH_CALUDE_sum_remainder_l729_72921


namespace NUMINAMATH_CALUDE_sum_is_composite_l729_72948

theorem sum_is_composite (m n : ℕ) (h : 88 * m = 81 * n) : 
  ∃ (k : ℕ), k > 1 ∧ k < m + n ∧ k ∣ (m + n) :=
by sorry

end NUMINAMATH_CALUDE_sum_is_composite_l729_72948


namespace NUMINAMATH_CALUDE_range_of_m_l729_72969

-- Define propositions p and q
def p (m : ℝ) : Prop := ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ ∀ (x y : ℝ), x^2 / (2*m) - y^2 / (m-1) = 1 → x^2 / (a^2) + y^2 / (b^2) = 1 ∧ a > b

def q (m : ℝ) : Prop := ∃ (e : ℝ), 1 < e ∧ e < 2 ∧ ∀ (x y : ℝ), y^2 / 5 - x^2 / m = 1 → x^2 / (5*e^2) - y^2 / 5 = 1

-- Theorem statement
theorem range_of_m :
  ∀ m : ℝ, (¬(p m) ∧ ¬(q m)) ∧ (p m ∨ q m) → 1/3 ≤ m ∧ m < 15 :=
sorry

end NUMINAMATH_CALUDE_range_of_m_l729_72969


namespace NUMINAMATH_CALUDE_inequality_solution_l729_72943

theorem inequality_solution (a : ℝ) : 
  (∀ x : ℝ, (x < 1 ∨ x > 3) ↔ (a * x) / (x - 1) < 1) → 
  a = 2/3 := by
sorry

end NUMINAMATH_CALUDE_inequality_solution_l729_72943


namespace NUMINAMATH_CALUDE_z₁z₂_value_a_value_when_sum_real_l729_72977

-- Define complex numbers z₁ and z₂
def z₁ (a : ℝ) : ℂ := 2 + a * Complex.I
def z₂ : ℂ := 3 - 4 * Complex.I

-- Theorem 1: When a = 1, z₁z₂ = 10 - 5i
theorem z₁z₂_value : z₁ 1 * z₂ = 10 - 5 * Complex.I := by sorry

-- Theorem 2: When z₁ + z₂ is a real number, a = 4
theorem a_value_when_sum_real : 
  ∃ (a : ℝ), (z₁ a + z₂).im = 0 → a = 4 := by sorry

end NUMINAMATH_CALUDE_z₁z₂_value_a_value_when_sum_real_l729_72977


namespace NUMINAMATH_CALUDE_turkey_cost_per_employee_l729_72929

/-- The cost of turkeys for employees --/
theorem turkey_cost_per_employee (num_employees : ℕ) (total_cost : ℚ) : 
  num_employees = 85 → total_cost = 2125 → (total_cost / num_employees : ℚ) = 25 := by
  sorry

end NUMINAMATH_CALUDE_turkey_cost_per_employee_l729_72929


namespace NUMINAMATH_CALUDE_one_large_one_small_capacity_l729_72986

/-- Represents the capacity of a large truck in tons -/
def large_truck_capacity : ℝ := sorry

/-- Represents the capacity of a small truck in tons -/
def small_truck_capacity : ℝ := sorry

/-- The total capacity of 3 large trucks and 4 small trucks is 22 tons -/
axiom condition1 : 3 * large_truck_capacity + 4 * small_truck_capacity = 22

/-- The total capacity of 2 large trucks and 6 small trucks is 23 tons -/
axiom condition2 : 2 * large_truck_capacity + 6 * small_truck_capacity = 23

/-- Theorem: One large truck and one small truck can transport 6.5 tons together -/
theorem one_large_one_small_capacity : 
  large_truck_capacity + small_truck_capacity = 6.5 := by sorry

end NUMINAMATH_CALUDE_one_large_one_small_capacity_l729_72986


namespace NUMINAMATH_CALUDE_equation_is_parabola_l729_72903

/-- Represents a point in a 2D plane -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- Defines the equation |y - 3| = √((x+4)² + (y-1)²) -/
def equation (p : Point2D) : Prop :=
  |p.y - 3| = Real.sqrt ((p.x + 4)^2 + (p.y - 1)^2)

/-- Defines a parabola as a set of points satisfying a quadratic equation -/
def isParabola (S : Set Point2D) : Prop :=
  ∃ (a b c d e : ℝ), a ≠ 0 ∧ 
    ∀ p ∈ S, a * p.x^2 + b * p.x * p.y + c * p.y^2 + d * p.x + e * p.y = 0

/-- Theorem stating that the given equation represents a parabola -/
theorem equation_is_parabola :
  isParabola {p : Point2D | equation p} :=
sorry

end NUMINAMATH_CALUDE_equation_is_parabola_l729_72903


namespace NUMINAMATH_CALUDE_average_of_five_numbers_l729_72951

theorem average_of_five_numbers (numbers : Fin 5 → ℝ) 
  (sum_of_three : ∃ (i j k : Fin 5), i ≠ j ∧ j ≠ k ∧ i ≠ k ∧ numbers i + numbers j + numbers k = 48)
  (avg_of_two : ∃ (l m : Fin 5), l ≠ m ∧ (numbers l + numbers m) / 2 = 26) :
  (numbers 0 + numbers 1 + numbers 2 + numbers 3 + numbers 4) / 5 = 20 := by
sorry

end NUMINAMATH_CALUDE_average_of_five_numbers_l729_72951


namespace NUMINAMATH_CALUDE_subtraction_rearrangement_l729_72950

theorem subtraction_rearrangement (a b c : ℤ) : a - b - c = a - (b + c) := by
  sorry

end NUMINAMATH_CALUDE_subtraction_rearrangement_l729_72950


namespace NUMINAMATH_CALUDE_square_plus_reciprocal_square_l729_72984

theorem square_plus_reciprocal_square (a : ℝ) (h : a + 1/a = 3) : a^2 + 1/a^2 = 7 := by
  sorry

end NUMINAMATH_CALUDE_square_plus_reciprocal_square_l729_72984


namespace NUMINAMATH_CALUDE_regular_polygon_interior_angle_sum_l729_72952

theorem regular_polygon_interior_angle_sum (n : ℕ) (h1 : n > 2) : 
  (360 / 24 : ℝ) = n → (180 * (n - 2) : ℝ) = 2340 := by
  sorry

end NUMINAMATH_CALUDE_regular_polygon_interior_angle_sum_l729_72952


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l729_72916

/-- An arithmetic sequence. -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- The sum of the 9th, 52nd, and 95th terms of the sequence. -/
def sum_terms (a : ℕ → ℝ) : ℝ := a 9 + a 52 + a 95

theorem arithmetic_sequence_sum (a : ℕ → ℝ) :
  arithmetic_sequence a → a 3 = 4 → a 101 = 36 → sum_terms a = 60 :=
by
  sorry

#check arithmetic_sequence_sum

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l729_72916


namespace NUMINAMATH_CALUDE_fifth_month_sale_l729_72900

theorem fifth_month_sale
  (target_average : ℕ)
  (num_months : ℕ)
  (sales : Fin 4 → ℕ)
  (sixth_month_sale : ℕ)
  (h1 : target_average = 6000)
  (h2 : num_months = 5)
  (h3 : sales 0 = 5420)
  (h4 : sales 1 = 5660)
  (h5 : sales 2 = 6200)
  (h6 : sales 3 = 6350)
  (h7 : sixth_month_sale = 5870) :
  ∃ (fifth_month_sale : ℕ),
    fifth_month_sale = target_average * num_months - (sales 0 + sales 1 + sales 2 + sales 3) :=
by sorry

end NUMINAMATH_CALUDE_fifth_month_sale_l729_72900


namespace NUMINAMATH_CALUDE_binomial_10_choose_5_l729_72928

theorem binomial_10_choose_5 : Nat.choose 10 5 = 252 := by
  sorry

end NUMINAMATH_CALUDE_binomial_10_choose_5_l729_72928


namespace NUMINAMATH_CALUDE_square_root_problem_l729_72988

theorem square_root_problem (n : ℝ) (x : ℝ) (h1 : n > 0) (h2 : Real.sqrt n = x + 3) (h3 : Real.sqrt n = 2*x - 6) :
  x = 1 ∧ n = 16 := by
  sorry

end NUMINAMATH_CALUDE_square_root_problem_l729_72988


namespace NUMINAMATH_CALUDE_adam_shelf_count_l729_72967

/-- The number of shelves in Adam's room -/
def num_shelves : ℕ := sorry

/-- The number of action figures that can fit on each shelf -/
def figures_per_shelf : ℕ := 9

/-- The total number of action figures that can be held by all shelves -/
def total_figures : ℕ := 27

/-- Theorem stating that the number of shelves is 3 -/
theorem adam_shelf_count : num_shelves = 3 := by sorry

end NUMINAMATH_CALUDE_adam_shelf_count_l729_72967


namespace NUMINAMATH_CALUDE_one_fourth_difference_product_sum_l729_72946

theorem one_fourth_difference_product_sum : 
  (1 / 4 : ℚ) * ((9 * 5) - (7 + 3)) = 35 / 4 := by
  sorry

end NUMINAMATH_CALUDE_one_fourth_difference_product_sum_l729_72946


namespace NUMINAMATH_CALUDE_rectangular_prism_diagonal_l729_72912

theorem rectangular_prism_diagonal (length width height : ℝ) 
  (h_length : length = 12) 
  (h_width : width = 15) 
  (h_height : height = 8) : 
  Real.sqrt (length^2 + width^2 + height^2) = Real.sqrt 433 :=
by sorry

end NUMINAMATH_CALUDE_rectangular_prism_diagonal_l729_72912


namespace NUMINAMATH_CALUDE_scale_division_l729_72992

-- Define the length of the scale in inches
def scale_length : ℕ := 6 * 12 + 8

-- Define the number of parts
def num_parts : ℕ := 4

-- Theorem to prove
theorem scale_division :
  scale_length / num_parts = 20 := by
  sorry

end NUMINAMATH_CALUDE_scale_division_l729_72992


namespace NUMINAMATH_CALUDE_factory_sample_theorem_l729_72975

/-- Given a factory with total products, a sample size, and products from one workshop,
    calculate the number of products drawn from this workshop in a stratified sampling. -/
def stratifiedSampleSize (totalProducts sampleSize workshopProducts : ℕ) : ℕ :=
  (workshopProducts * sampleSize) / totalProducts

/-- Theorem stating that for the given values, the stratified sample size is 16. -/
theorem factory_sample_theorem :
  stratifiedSampleSize 2048 128 256 = 16 := by
  sorry

end NUMINAMATH_CALUDE_factory_sample_theorem_l729_72975


namespace NUMINAMATH_CALUDE_complex_equation_solution_l729_72905

theorem complex_equation_solution (z : ℂ) :
  (3 + Complex.I) * z = 4 - 2 * Complex.I →
  z = 1 - Complex.I := by
sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l729_72905


namespace NUMINAMATH_CALUDE_net_effect_theorem_l729_72907

/-- Calculates the net effect on sale given price reduction, sale increase, tax, and discount -/
def net_effect_on_sale (price_reduction : ℝ) (sale_increase : ℝ) (tax : ℝ) (discount : ℝ) : ℝ :=
  let new_price_factor := 1 - price_reduction
  let new_quantity_factor := 1 + sale_increase
  let after_tax_factor := 1 + tax
  let after_discount_factor := 1 - discount
  new_price_factor * new_quantity_factor * after_tax_factor * after_discount_factor

/-- Theorem stating the net effect on sale given specific conditions -/
theorem net_effect_theorem :
  net_effect_on_sale 0.60 1.50 0.10 0.05 = 1.045 := by
  sorry

end NUMINAMATH_CALUDE_net_effect_theorem_l729_72907


namespace NUMINAMATH_CALUDE_circle_and_tangent_properties_l729_72918

-- Define the circle C
def circle_C (x y : ℝ) : Prop :=
  (x - 1)^2 + (y - 2)^2 = 9

-- Define the center of the circle
def center (x y : ℝ) : Prop :=
  y = 2 * x ∧ x > 0 ∧ y > 0

-- Define the tangent lines
def tangent_line_1 (x : ℝ) : Prop := x = 4
def tangent_line_2 (x y : ℝ) : Prop := 4 * x + 3 * y = 25

theorem circle_and_tangent_properties :
  ∃ (cx cy : ℝ),
    -- The center lies on y = 2x in the first quadrant
    center cx cy ∧
    -- The circle passes through (1, -1)
    circle_C 1 (-1) ∧
    -- (4, 3) is outside the circle
    ¬ circle_C 4 3 ∧
    -- The tangent lines touch the circle at exactly one point each
    (∃ (tx ty : ℝ), circle_C tx ty ∧ tangent_line_1 tx) ∧
    (∃ (tx ty : ℝ), circle_C tx ty ∧ tangent_line_2 tx ty) :=
by sorry

end NUMINAMATH_CALUDE_circle_and_tangent_properties_l729_72918


namespace NUMINAMATH_CALUDE_inequality_solution_set_l729_72902

theorem inequality_solution_set (x : ℝ) : 
  (x - 2) / (x - 4) ≥ 3 ↔ x ∈ Set.Ioo 4 5 ∪ {5} :=
sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l729_72902


namespace NUMINAMATH_CALUDE_board_numbers_prime_or_one_l729_72949

theorem board_numbers_prime_or_one (a : ℕ) (h_a_odd : Odd a) (h_a_gt_100 : a > 100)
  (h_prime : ∀ n : ℕ, n ≤ Real.sqrt (a / 5) → Nat.Prime ((a - n^2) / 4)) :
  ∀ n : ℕ, Nat.Prime ((a - n^2) / 4) ∨ (a - n^2) / 4 = 1 :=
by sorry

end NUMINAMATH_CALUDE_board_numbers_prime_or_one_l729_72949


namespace NUMINAMATH_CALUDE_tangent_four_implies_expression_l729_72961

theorem tangent_four_implies_expression (α : Real) (h : Real.tan α = 4) :
  (1 + Real.cos (2 * α) + 8 * Real.sin α ^ 2) / Real.sin (2 * α) = 65 / 4 := by
  sorry

end NUMINAMATH_CALUDE_tangent_four_implies_expression_l729_72961


namespace NUMINAMATH_CALUDE_equation_d_is_quadratic_l729_72925

/-- A polynomial equation in x is quadratic if it can be written in the form ax² + bx + c = 0,
    where a ≠ 0 and a, b, c are constants. -/
def is_quadratic_equation (f : ℝ → ℝ) : Prop :=
  ∃ a b c : ℝ, a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

/-- The equation 3(x+1)² = 2(x+1) is a quadratic equation in terms of x. -/
theorem equation_d_is_quadratic :
  is_quadratic_equation (λ x => 3 * (x + 1)^2 - 2 * (x + 1)) :=
sorry

end NUMINAMATH_CALUDE_equation_d_is_quadratic_l729_72925


namespace NUMINAMATH_CALUDE_simple_interest_principal_calculation_l729_72991

/-- Simple interest calculation -/
theorem simple_interest_principal_calculation
  (rate : ℝ) (interest : ℝ) (time : ℝ) :
  rate = 5 →
  interest = 160 →
  time = 4 →
  160 = (rate * time / 100) * (interest * 100 / (rate * time)) :=
by
  sorry

#check simple_interest_principal_calculation

end NUMINAMATH_CALUDE_simple_interest_principal_calculation_l729_72991


namespace NUMINAMATH_CALUDE_cube_root_inequality_l729_72901

theorem cube_root_inequality (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  Real.rpow ((a + 1) * (b + 1) * (c + 1)) (1/3) ≥ Real.rpow (a * b * c) (1/3) + 1 := by
  sorry

end NUMINAMATH_CALUDE_cube_root_inequality_l729_72901


namespace NUMINAMATH_CALUDE_arithmetic_sequence_transformation_l729_72983

/-- An arithmetic sequence -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- Given arithmetic sequence and its transformation -/
theorem arithmetic_sequence_transformation
  (a : ℕ → ℝ) (b : ℕ → ℝ) (d : ℝ)
  (h1 : ArithmeticSequence a)
  (h2 : ∀ n : ℕ, a (n + 1) = a n + d)
  (h3 : ∀ n : ℕ, b n = 3 * a n + 4) :
  ArithmeticSequence b :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_transformation_l729_72983


namespace NUMINAMATH_CALUDE_min_value_fraction_l729_72985

theorem min_value_fraction (a b : ℝ) (h1 : a > b) (h2 : a * b = 1) :
  (a^2 + b^2) / (a - b) ≥ 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_min_value_fraction_l729_72985


namespace NUMINAMATH_CALUDE_fraction_product_l729_72909

theorem fraction_product (a b c d e f : ℝ) 
  (h1 : a / b = 5 / 2)
  (h2 : b / c = 1 / 2)
  (h3 : c / d = 1)
  (h4 : d / e = 3 / 2)
  (h5 : e / f = 4 / 3) :
  a * b * c / (d * e * f) = 2.5 := by
  sorry

end NUMINAMATH_CALUDE_fraction_product_l729_72909


namespace NUMINAMATH_CALUDE_monomial_exponent_equality_l729_72976

theorem monomial_exponent_equality (a b : ℤ) : 
  (1 : ℤ) = a - 2 → b + 1 = 3 → (a - b)^(2023 : ℕ) = 1 := by
  sorry

end NUMINAMATH_CALUDE_monomial_exponent_equality_l729_72976


namespace NUMINAMATH_CALUDE_middle_school_enrollment_l729_72960

theorem middle_school_enrollment (band_percentage : Real) (sports_percentage : Real)
  (band_count : Nat) (sports_count : Nat)
  (h1 : band_percentage = 0.20)
  (h2 : sports_percentage = 0.30)
  (h3 : band_count = 168)
  (h4 : sports_count = 252) :
  ∃ (total : Nat), (band_count : Real) / band_percentage = total ∧
                   (sports_count : Real) / sports_percentage = total ∧
                   total = 840 := by
  sorry

end NUMINAMATH_CALUDE_middle_school_enrollment_l729_72960


namespace NUMINAMATH_CALUDE_candy_distribution_l729_72911

theorem candy_distribution (x : ℝ) (x_pos : x > 0) : 
  let al_share := (4/9 : ℝ) * x
  let bert_share := (1/3 : ℝ) * (x - al_share)
  let carl_share := (2/9 : ℝ) * (x - al_share - bert_share)
  al_share + bert_share + carl_share = x :=
by sorry

end NUMINAMATH_CALUDE_candy_distribution_l729_72911


namespace NUMINAMATH_CALUDE_linear_congruence_solvability_and_solutions_l729_72940

/-- 
For integers a, b, and m > 0, this theorem states:
1. The congruence ax ≡ b (mod m) has solutions if and only if gcd(a,m) | b.
2. If solutions exist, they are of the form x = x₀ + k(m/d) for all integers k, 
   where d = gcd(a,m) and x₀ is a particular solution to (a/d)x ≡ (b/d) (mod m/d).
-/
theorem linear_congruence_solvability_and_solutions 
  (a b m : ℤ) (hm : m > 0) : 
  (∃ x, a * x ≡ b [ZMOD m]) ↔ (gcd a m ∣ b) ∧
  (∀ x, (a * x ≡ b [ZMOD m]) ↔ 
    ∃ (x₀ k : ℤ), x = x₀ + k * (m / gcd a m) ∧ 
    (a / gcd a m) * x₀ ≡ (b / gcd a m) [ZMOD (m / gcd a m)]) :=
by sorry

end NUMINAMATH_CALUDE_linear_congruence_solvability_and_solutions_l729_72940


namespace NUMINAMATH_CALUDE_integer_tuple_solution_l729_72930

theorem integer_tuple_solution :
  ∀ a b c x y z : ℕ,
    a + b + c = x * y * z →
    x + y + z = a * b * c →
    a ≥ b →
    b ≥ c →
    c ≥ 1 →
    x ≥ y →
    y ≥ z →
    z ≥ 1 →
    ((a = 2 ∧ b = 2 ∧ c = 2 ∧ x = 6 ∧ y = 1 ∧ z = 1) ∨
     (a = 5 ∧ b = 2 ∧ c = 1 ∧ x = 8 ∧ y = 1 ∧ z = 1) ∨
     (a = 3 ∧ b = 3 ∧ c = 1 ∧ x = 7 ∧ y = 1 ∧ z = 1) ∨
     (a = 3 ∧ b = 2 ∧ c = 1 ∧ x = 3 ∧ y = 2 ∧ z = 1)) :=
by
  sorry

end NUMINAMATH_CALUDE_integer_tuple_solution_l729_72930


namespace NUMINAMATH_CALUDE_complex_square_root_minus_100_plus_44i_l729_72945

theorem complex_square_root_minus_100_plus_44i :
  {z : ℂ | z^2 = -100 + 44*I} = {2 + 11*I, -2 - 11*I} := by sorry

end NUMINAMATH_CALUDE_complex_square_root_minus_100_plus_44i_l729_72945


namespace NUMINAMATH_CALUDE_chocolate_bars_count_l729_72998

theorem chocolate_bars_count (bar_price : ℕ) (remaining_bars : ℕ) (total_sales : ℕ) : 
  bar_price = 6 →
  remaining_bars = 6 →
  total_sales = 42 →
  ∃ (total_bars : ℕ), total_bars = 13 ∧ bar_price * (total_bars - remaining_bars) = total_sales :=
by sorry

end NUMINAMATH_CALUDE_chocolate_bars_count_l729_72998


namespace NUMINAMATH_CALUDE_max_sum_with_reciprocals_l729_72926

theorem max_sum_with_reciprocals (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : x + y + 1/x + 1/y = 5) : 
  ∀ a b : ℝ, a > 0 → b > 0 → a + b + 1/a + 1/b = 5 → x + y ≥ a + b ∧ x + y ≤ 4 :=
by sorry

end NUMINAMATH_CALUDE_max_sum_with_reciprocals_l729_72926


namespace NUMINAMATH_CALUDE_expand_expression_l729_72968

theorem expand_expression (x : ℝ) : (15 * x^2 + 5) * 3 * x^3 = 45 * x^5 + 15 * x^3 := by
  sorry

end NUMINAMATH_CALUDE_expand_expression_l729_72968


namespace NUMINAMATH_CALUDE_aftershave_dilution_l729_72955

/-- Proves that 6 ounces of water are needed to dilute 12 ounces of 60% alcohol aftershave to 40% alcohol concentration -/
theorem aftershave_dilution (initial_volume : ℝ) (initial_concentration : ℝ) 
  (final_concentration : ℝ) (water_added : ℝ) : 
  initial_volume = 12 →
  initial_concentration = 0.6 →
  final_concentration = 0.4 →
  water_added = 6 →
  initial_volume * initial_concentration = 
    final_concentration * (initial_volume + water_added) := by
  sorry

#check aftershave_dilution

end NUMINAMATH_CALUDE_aftershave_dilution_l729_72955


namespace NUMINAMATH_CALUDE_geometric_sequence_ratio_l729_72915

/-- Given a geometric sequence {a_n} with sum of first n terms S_n,
    if a_1 + a_3 = 5/4 and a_2 + a_4 = 5/2, then S_6 / S_3 = 9 -/
theorem geometric_sequence_ratio (a : ℕ → ℝ) (S : ℕ → ℝ)
  (h1 : a 1 + a 3 = 5/4)
  (h2 : a 2 + a 4 = 5/2)
  (h_geom : ∀ n : ℕ, a (n+1) / a n = a 2 / a 1)
  (h_sum : ∀ n : ℕ, S n = a 1 * (1 - (a 2 / a 1)^n) / (1 - a 2 / a 1)) :
  S 6 / S 3 = 9 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_ratio_l729_72915


namespace NUMINAMATH_CALUDE_divisor_sum_property_l729_72922

def divisors (n : ℕ) : List ℕ := sorry

def D (n : ℕ) : ℕ := sorry

theorem divisor_sum_property (n : ℕ) (h : n > 1) :
  let d := divisors n
  D n < n^2 ∧ (D n ∣ n^2 ↔ Nat.Prime n) := by sorry

end NUMINAMATH_CALUDE_divisor_sum_property_l729_72922


namespace NUMINAMATH_CALUDE_max_potential_salary_is_440000_l729_72966

/-- Represents a soccer team with its payroll constraints -/
structure SoccerTeam where
  numPlayers : ℕ
  minSalary : ℕ
  maxPayroll : ℕ

/-- Calculates the maximum potential salary for an individual player on a team -/
def maxPotentialSalary (team : SoccerTeam) : ℕ :=
  team.maxPayroll - (team.numPlayers - 1) * team.minSalary

/-- Theorem stating the maximum potential salary for an individual player -/
theorem max_potential_salary_is_440000 :
  let team : SoccerTeam := ⟨19, 20000, 800000⟩
  maxPotentialSalary team = 440000 := by
  sorry

#eval maxPotentialSalary ⟨19, 20000, 800000⟩

end NUMINAMATH_CALUDE_max_potential_salary_is_440000_l729_72966


namespace NUMINAMATH_CALUDE_min_benches_in_hall_l729_72990

/-- The minimum number of benches required in a school hall -/
def min_benches (male_students : ℕ) (female_ratio : ℕ) (students_per_bench : ℕ) : ℕ :=
  ((male_students * (female_ratio + 1) + students_per_bench - 1) / students_per_bench : ℕ)

/-- Theorem: Given the conditions, the minimum number of benches required is 29 -/
theorem min_benches_in_hall :
  min_benches 29 4 5 = 29 := by
  sorry

end NUMINAMATH_CALUDE_min_benches_in_hall_l729_72990


namespace NUMINAMATH_CALUDE_rhombus_area_rhombus_area_is_88_l729_72958

/-- The area of a rhombus with vertices at (0, 5.5), (8, 0), (0, -5.5), and (-8, 0) is 88 square units. -/
theorem rhombus_area : ℝ → Prop :=
  fun area =>
    let v1 : ℝ × ℝ := (0, 5.5)
    let v2 : ℝ × ℝ := (8, 0)
    let v3 : ℝ × ℝ := (0, -5.5)
    let v4 : ℝ × ℝ := (-8, 0)
    let d1 : ℝ := v1.2 - v3.2
    let d2 : ℝ := v2.1 - v4.1
    area = (d1 * d2) / 2 ∧ area = 88

theorem rhombus_area_is_88 : rhombus_area 88 := by
  sorry

end NUMINAMATH_CALUDE_rhombus_area_rhombus_area_is_88_l729_72958


namespace NUMINAMATH_CALUDE_students_per_group_l729_72933

/-- Given a total of 64 students, with 36 not picked, and divided into 4 groups,
    prove that there are 7 students in each group. -/
theorem students_per_group :
  ∀ (total : ℕ) (not_picked : ℕ) (groups : ℕ),
    total = 64 →
    not_picked = 36 →
    groups = 4 →
    (total - not_picked) / groups = 7 :=
by
  sorry

end NUMINAMATH_CALUDE_students_per_group_l729_72933


namespace NUMINAMATH_CALUDE_rationalize_denominator_l729_72972

-- Define the original expression
def original_expr := (4 : ℚ) / (3 * (7 : ℚ)^(1/3))

-- Define the rationalized expression
def rationalized_expr := (4 * (49 : ℚ)^(1/3)) / 21

-- Define the property that 49 is not divisible by the cube of any prime
def not_cube_divisible (n : ℕ) : Prop :=
  ∀ p : ℕ, Prime p → ¬(p^3 ∣ n)

-- Theorem statement
theorem rationalize_denominator :
  original_expr = rationalized_expr ∧ not_cube_divisible 49 := by sorry

end NUMINAMATH_CALUDE_rationalize_denominator_l729_72972
