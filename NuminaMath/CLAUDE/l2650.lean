import Mathlib

namespace NUMINAMATH_CALUDE_candy_challenge_solution_l2650_265075

/-- Represents the candy-eating challenge over three days -/
def candy_challenge (initial_candies : ℚ) : Prop :=
  let day1_after_eating := (3/4) * initial_candies
  let day1_remaining := day1_after_eating - 3
  let day2_after_eating := (4/5) * day1_remaining
  let day2_remaining := day2_after_eating - 5
  day2_remaining = 10

theorem candy_challenge_solution :
  ∃ (x : ℚ), candy_challenge x ∧ x = 52 :=
sorry

end NUMINAMATH_CALUDE_candy_challenge_solution_l2650_265075


namespace NUMINAMATH_CALUDE_denominator_numerator_difference_l2650_265036

/-- Represents a base-12 number as a pair of integers (numerator, denominator) -/
def Base12Fraction := ℤ × ℤ

/-- Converts a repeating decimal in base 12 to a fraction -/
def repeating_decimal_to_fraction (digits : List ℕ) : Base12Fraction := sorry

/-- Simplifies a fraction to its lowest terms -/
def simplify_fraction (f : Base12Fraction) : Base12Fraction := sorry

/-- The infinite repeating decimal 0.127127127... in base 12 -/
def G : Base12Fraction := repeating_decimal_to_fraction [1, 2, 7]

theorem denominator_numerator_difference :
  let simplified_G := simplify_fraction G
  (simplified_G.2 - simplified_G.1) = 342 := by sorry

end NUMINAMATH_CALUDE_denominator_numerator_difference_l2650_265036


namespace NUMINAMATH_CALUDE_find_S_l2650_265067

-- Define the sets A and B
def A : Set ℝ := {x | x^2 - 7*x + 10 ≤ 0}
def B : Set ℝ := {x | ∃ (a b : ℝ), x^2 + a*x + b < 0}

-- Define the union of A and B
def AUnionB : Set ℝ := {x | x - 3 < 4 ∧ 4 ≤ 2*x}

-- State the theorem
theorem find_S (a b : ℝ) : 
  A ∩ B = ∅ → 
  A ∪ B = AUnionB → 
  {x | x = a + b} = {23} := by sorry

end NUMINAMATH_CALUDE_find_S_l2650_265067


namespace NUMINAMATH_CALUDE_units_digit_34_plus_47_base_8_l2650_265059

def base_8_addition (a b : Nat) : Nat :=
  (a + b) % 8

theorem units_digit_34_plus_47_base_8 :
  base_8_addition (34 % 8) (47 % 8) = 3 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_34_plus_47_base_8_l2650_265059


namespace NUMINAMATH_CALUDE_sequence_transformation_l2650_265095

/-- Represents a sequence of letters 'A' and 'B' -/
def Sequence := List Char

/-- An operation that can be performed on a sequence -/
inductive Operation
| Insert (c : Char) (pos : Nat) (count : Nat)
| Remove (pos : Nat) (count : Nat)

/-- Applies an operation to a sequence -/
def applyOperation (s : Sequence) (op : Operation) : Sequence :=
  match op with
  | Operation.Insert c pos count => sorry
  | Operation.Remove pos count => sorry

/-- Checks if a sequence contains only 'A' and 'B' -/
def isValidSequence (s : Sequence) : Prop :=
  s.all (fun c => c = 'A' ∨ c = 'B')

/-- Theorem: Any two valid sequences of length 100 can be transformed
    into each other using at most 100 operations -/
theorem sequence_transformation
  (s1 s2 : Sequence)
  (h1 : s1.length = 100)
  (h2 : s2.length = 100)
  (v1 : isValidSequence s1)
  (v2 : isValidSequence s2) :
  ∃ (ops : List Operation),
    ops.length ≤ 100 ∧
    (ops.foldl applyOperation s1 = s2) :=
  sorry

end NUMINAMATH_CALUDE_sequence_transformation_l2650_265095


namespace NUMINAMATH_CALUDE_max_xyz_value_l2650_265071

theorem max_xyz_value (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h : x * y + z = (x + z) * (y + z)) :
  x * y * z ≤ 1 / 27 := by
sorry

end NUMINAMATH_CALUDE_max_xyz_value_l2650_265071


namespace NUMINAMATH_CALUDE_solve_equation_l2650_265096

theorem solve_equation (x : ℝ) : 2*x - 3*x + 4*x = 120 → x = 40 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l2650_265096


namespace NUMINAMATH_CALUDE_archery_score_distribution_l2650_265003

theorem archery_score_distribution :
  ∃! (a b c d : ℕ),
    a + b + c + d = 10 ∧
    a ≥ 1 ∧ b ≥ 1 ∧ c ≥ 1 ∧ d ≥ 1 ∧
    8*a + 12*b + 14*c + 18*d = 110 :=
by
  sorry

end NUMINAMATH_CALUDE_archery_score_distribution_l2650_265003


namespace NUMINAMATH_CALUDE_square_root_of_36_l2650_265040

theorem square_root_of_36 : ∃ (x : ℝ), x^2 = 36 ↔ x = 6 ∨ x = -6 := by
  sorry

end NUMINAMATH_CALUDE_square_root_of_36_l2650_265040


namespace NUMINAMATH_CALUDE_undefined_values_count_l2650_265030

theorem undefined_values_count : 
  ∃! (s : Finset ℝ), (∀ x ∈ s, (x^2 + 2*x - 3) * (x - 3) = 0) ∧ Finset.card s = 3 := by
  sorry

end NUMINAMATH_CALUDE_undefined_values_count_l2650_265030


namespace NUMINAMATH_CALUDE_idaho_to_nevada_distance_l2650_265061

/-- Represents the road trip from Washington to Nevada via Idaho -/
structure RoadTrip where
  wash_to_idaho : ℝ     -- Distance from Washington to Idaho
  idaho_to_nevada : ℝ   -- Distance from Idaho to Nevada (to be proven)
  speed_to_idaho : ℝ    -- Speed from Washington to Idaho
  speed_to_nevada : ℝ   -- Speed from Idaho to Nevada
  total_time : ℝ        -- Total travel time

/-- The road trip satisfies the given conditions -/
def satisfies_conditions (trip : RoadTrip) : Prop :=
  trip.wash_to_idaho = 640 ∧
  trip.speed_to_idaho = 80 ∧
  trip.speed_to_nevada = 50 ∧
  trip.total_time = 19 ∧
  trip.total_time = trip.wash_to_idaho / trip.speed_to_idaho + trip.idaho_to_nevada / trip.speed_to_nevada

theorem idaho_to_nevada_distance (trip : RoadTrip) 
  (h : satisfies_conditions trip) : trip.idaho_to_nevada = 550 := by
  sorry

end NUMINAMATH_CALUDE_idaho_to_nevada_distance_l2650_265061


namespace NUMINAMATH_CALUDE_b_4_lt_b_7_l2650_265007

def b (n : ℕ) (α : ℕ → ℕ) : ℚ :=
  match n with
  | 0 => 0
  | 1 => 1 + 1 / α 1
  | n + 1 => 1 + 1 / (α 1 + 1 / b n (fun k => α (k + 1)))

theorem b_4_lt_b_7 (α : ℕ → ℕ) (h : ∀ k, α k ≥ 1) : b 4 α < b 7 α := by
  sorry

end NUMINAMATH_CALUDE_b_4_lt_b_7_l2650_265007


namespace NUMINAMATH_CALUDE_number_of_b_objects_l2650_265044

theorem number_of_b_objects (total : ℕ) (a : ℕ) (b : ℕ) : 
  total = 35 →
  total = a + b →
  a = 17 →
  b = 18 := by
sorry

end NUMINAMATH_CALUDE_number_of_b_objects_l2650_265044


namespace NUMINAMATH_CALUDE_maddies_mom_milk_consumption_l2650_265099

/-- Represents the weekly coffee consumption scenario of Maddie's mom -/
structure CoffeeConsumption where
  cups_per_day : ℕ
  ounces_per_cup : ℚ
  ounces_per_bag : ℚ
  price_per_bag : ℚ
  price_per_gallon_milk : ℚ
  weekly_coffee_budget : ℚ

/-- Calculates the amount of milk in gallons used per week -/
def milk_gallons_per_week (c : CoffeeConsumption) : ℚ :=
  sorry

/-- Theorem stating that given the specific conditions, 
    the amount of milk used per week is 0.5 gallons -/
theorem maddies_mom_milk_consumption :
  let c : CoffeeConsumption := {
    cups_per_day := 2,
    ounces_per_cup := 3/2,
    ounces_per_bag := 21/2,
    price_per_bag := 8,
    price_per_gallon_milk := 4,
    weekly_coffee_budget := 18
  }
  milk_gallons_per_week c = 1/2 :=
by sorry

end NUMINAMATH_CALUDE_maddies_mom_milk_consumption_l2650_265099


namespace NUMINAMATH_CALUDE_initial_deposit_l2650_265069

theorem initial_deposit (P R : ℝ) : 
  P + (P * R * 3) / 100 = 9200 →
  P + (P * (R + 2.5) * 3) / 100 = 9800 →
  P = 8000 := by
sorry

end NUMINAMATH_CALUDE_initial_deposit_l2650_265069


namespace NUMINAMATH_CALUDE_five_regular_polyhedra_l2650_265078

/-- A convex regular polyhedron with n edges meeting at each vertex and k sides on each face. -/
structure ConvexRegularPolyhedron where
  n : ℕ
  k : ℕ
  n_ge_three : n ≥ 3
  k_ge_three : k ≥ 3

/-- The inequality that must be satisfied by a convex regular polyhedron. -/
def satisfies_inequality (p : ConvexRegularPolyhedron) : Prop :=
  (1 : ℚ) / p.n + (1 : ℚ) / p.k > (1 : ℚ) / 2

/-- The theorem stating that there are only five types of convex regular polyhedra. -/
theorem five_regular_polyhedra :
  ∀ p : ConvexRegularPolyhedron, satisfies_inequality p ↔
    (p.n = 3 ∧ p.k = 3) ∨
    (p.n = 3 ∧ p.k = 4) ∨
    (p.n = 3 ∧ p.k = 5) ∨
    (p.n = 4 ∧ p.k = 3) ∨
    (p.n = 5 ∧ p.k = 3) :=
by sorry

end NUMINAMATH_CALUDE_five_regular_polyhedra_l2650_265078


namespace NUMINAMATH_CALUDE_weight_of_A_l2650_265008

/-- Given the weights of five people A, B, C, D, and E, prove that A weighs 77 kg -/
theorem weight_of_A (A B C D E : ℝ) : 
  (A + B + C) / 3 = 84 →
  (A + B + C + D) / 4 = 80 →
  E = D + 5 →
  (B + C + D + E) / 4 = 79 →
  A = 77 := by
sorry

end NUMINAMATH_CALUDE_weight_of_A_l2650_265008


namespace NUMINAMATH_CALUDE_remainder_of_470521_div_5_l2650_265016

theorem remainder_of_470521_div_5 : 470521 % 5 = 1 := by
  sorry

end NUMINAMATH_CALUDE_remainder_of_470521_div_5_l2650_265016


namespace NUMINAMATH_CALUDE_largest_integer_k_for_distinct_roots_l2650_265028

theorem largest_integer_k_for_distinct_roots (k : ℝ) : 
  (∃ x y : ℝ, x ≠ y ∧ 
   (k - 2) * x^2 - 4 * x + 4 = 0 ∧ 
   (k - 2) * y^2 - 4 * y + 4 = 0) →
  (∀ m : ℤ, m > 1 → (m : ℝ) > k) :=
by sorry

#check largest_integer_k_for_distinct_roots

end NUMINAMATH_CALUDE_largest_integer_k_for_distinct_roots_l2650_265028


namespace NUMINAMATH_CALUDE_complex_fraction_product_l2650_265011

theorem complex_fraction_product (a b : ℝ) :
  (1 + 7 * Complex.I) / (2 - Complex.I) = Complex.mk a b →
  a * b = -3 := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_product_l2650_265011


namespace NUMINAMATH_CALUDE_sqrt_450_simplified_l2650_265065

theorem sqrt_450_simplified : Real.sqrt 450 = 15 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_450_simplified_l2650_265065


namespace NUMINAMATH_CALUDE_average_and_difference_l2650_265024

theorem average_and_difference (y : ℝ) : 
  (35 + y) / 2 = 44 → |35 - y| = 18 := by
  sorry

end NUMINAMATH_CALUDE_average_and_difference_l2650_265024


namespace NUMINAMATH_CALUDE_rectangle_side_length_l2650_265054

theorem rectangle_side_length (perimeter width : ℝ) (h1 : perimeter = 40) (h2 : width = 8) :
  perimeter / 2 - width = 12 :=
by sorry

end NUMINAMATH_CALUDE_rectangle_side_length_l2650_265054


namespace NUMINAMATH_CALUDE_smallest_integer_satisfying_inequality_l2650_265037

theorem smallest_integer_satisfying_inequality : 
  (∀ x : ℤ, x < 11 → 2*x ≥ 3*x - 10) ∧ (2*11 < 3*11 - 10) := by
  sorry

end NUMINAMATH_CALUDE_smallest_integer_satisfying_inequality_l2650_265037


namespace NUMINAMATH_CALUDE_correct_arrangements_l2650_265013

/-- The number of people standing in a row -/
def n : ℕ := 7

/-- Calculates the number of arrangements given specific conditions -/
noncomputable def arrangements (condition : ℕ) : ℕ :=
  match condition with
  | 1 => 3720  -- A cannot stand at the head, and B cannot stand at the tail
  | 2 => 720   -- A, B, and C must stand next to each other
  | 3 => 1440  -- A, B, and C must not stand next to each other
  | 4 => 1200  -- There is exactly one person between A and B
  | 5 => 840   -- A, B, and C must stand in order from left to right
  | _ => 0     -- Invalid condition

/-- Theorem stating the correct number of arrangements for each condition -/
theorem correct_arrangements :
  (arrangements 1 = 3720) ∧
  (arrangements 2 = 720) ∧
  (arrangements 3 = 1440) ∧
  (arrangements 4 = 1200) ∧
  (arrangements 5 = 840) :=
by sorry

end NUMINAMATH_CALUDE_correct_arrangements_l2650_265013


namespace NUMINAMATH_CALUDE_tip_amount_is_24_l2650_265081

-- Define the cost of haircuts
def womens_haircut_cost : ℚ := 48
def childrens_haircut_cost : ℚ := 36

-- Define the number of each type of haircut
def num_womens_haircuts : ℕ := 1
def num_childrens_haircuts : ℕ := 2

-- Define the tip percentage
def tip_percentage : ℚ := 20 / 100

-- Theorem statement
theorem tip_amount_is_24 :
  let total_cost := womens_haircut_cost * num_womens_haircuts + childrens_haircut_cost * num_childrens_haircuts
  tip_percentage * total_cost = 24 := by
  sorry

end NUMINAMATH_CALUDE_tip_amount_is_24_l2650_265081


namespace NUMINAMATH_CALUDE_rectangle_perimeter_l2650_265025

theorem rectangle_perimeter (area : ℝ) (length width : ℝ) : 
  area = 450 ∧ length = 2 * width ∧ area = length * width → 
  2 * (length + width) = 90 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_perimeter_l2650_265025


namespace NUMINAMATH_CALUDE_phone_profit_fraction_l2650_265060

theorem phone_profit_fraction (num_phones : ℕ) (initial_investment : ℚ) (selling_price : ℚ) :
  num_phones = 200 →
  initial_investment = 3000 →
  selling_price = 20 →
  (num_phones * selling_price - initial_investment) / initial_investment = 1/3 := by
sorry

end NUMINAMATH_CALUDE_phone_profit_fraction_l2650_265060


namespace NUMINAMATH_CALUDE_corrected_mean_l2650_265019

theorem corrected_mean (n : ℕ) (original_mean : ℚ) (wrong_value : ℚ) (correct_value : ℚ) :
  n = 50 ∧ original_mean = 36 ∧ wrong_value = 23 ∧ correct_value = 46 →
  ((n : ℚ) * original_mean - wrong_value + correct_value) / n = 36.46 := by
  sorry

end NUMINAMATH_CALUDE_corrected_mean_l2650_265019


namespace NUMINAMATH_CALUDE_monotonicity_intervals_k_range_l2650_265094

/-- The function f(x) = xe^(kx) -/
noncomputable def f (k : ℝ) (x : ℝ) : ℝ := x * Real.exp (k * x)

/-- Monotonicity intervals for f(x) when k > 0 -/
theorem monotonicity_intervals (k : ℝ) (h : k > 0) :
  (∀ x₁ x₂, x₁ < x₂ ∧ - 1 / k < x₁ → f k x₁ < f k x₂) ∧
  (∀ x₁ x₂, x₁ < x₂ ∧ x₂ < - 1 / k → f k x₁ > f k x₂) :=
sorry

/-- Range of k when f(x) is monotonically increasing in (-1, 1) -/
theorem k_range (k : ℝ) (h : k ≠ 0) :
  (∀ x₁ x₂, -1 < x₁ ∧ x₁ < x₂ ∧ x₂ < 1 → f k x₁ < f k x₂) →
  (k ∈ Set.Icc (-1) 0 ∪ Set.Ioc 0 1) :=
sorry

end NUMINAMATH_CALUDE_monotonicity_intervals_k_range_l2650_265094


namespace NUMINAMATH_CALUDE_meet_after_four_turns_l2650_265080

-- Define the number of points on the circular track
def num_points : ℕ := 15

-- Define Alice's clockwise movement per turn
def alice_move : ℕ := 4

-- Define Bob's counterclockwise movement per turn
def bob_move : ℕ := 11

-- Define the starting point for both Alice and Bob
def start_point : ℕ := 15

-- Function to calculate the new position after a move
def new_position (current : ℕ) (move : ℕ) : ℕ :=
  ((current + move - 1) % num_points) + 1

-- Function to calculate Alice's position after n turns
def alice_position (n : ℕ) : ℕ :=
  new_position start_point (n * alice_move)

-- Function to calculate Bob's position after n turns
def bob_position (n : ℕ) : ℕ :=
  new_position start_point (n * (num_points - bob_move))

-- Theorem stating that Alice and Bob meet after 4 turns
theorem meet_after_four_turns :
  ∃ n : ℕ, n = 4 ∧ alice_position n = bob_position n :=
sorry

end NUMINAMATH_CALUDE_meet_after_four_turns_l2650_265080


namespace NUMINAMATH_CALUDE_circle_mass_is_one_kg_l2650_265017

/-- Given three balanced scales and the mass of λ, prove that the circle has a mass of 1 kg. -/
theorem circle_mass_is_one_kg (x y z : ℝ) : 
  3 * y = 2 * x →  -- First scale
  x + z = 3 * y →  -- Second scale
  2 * y = x + 1 →  -- Third scale (λ has mass 1)
  z = 1 :=         -- Mass of circle is 1 kg
by sorry

end NUMINAMATH_CALUDE_circle_mass_is_one_kg_l2650_265017


namespace NUMINAMATH_CALUDE_light_path_length_l2650_265070

-- Define the cube side length
def cube_side : ℝ := 10

-- Define the reflection point coordinates relative to the face
def reflect_x : ℝ := 4
def reflect_y : ℝ := 6

-- Define the number of reflections needed
def num_reflections : ℕ := 10

-- Theorem statement
theorem light_path_length :
  let path_length := (num_reflections : ℝ) * Real.sqrt (cube_side^2 + reflect_x^2 + reflect_y^2)
  path_length = 10 * Real.sqrt 152 :=
by sorry

end NUMINAMATH_CALUDE_light_path_length_l2650_265070


namespace NUMINAMATH_CALUDE_simplify_expression_1_simplify_expression_2_l2650_265058

-- Problem 1
theorem simplify_expression_1 (x : ℝ) : 
  x^2 + (3*x - 5) - (4*x - 1) = x^2 - x - 4 := by sorry

-- Problem 2
theorem simplify_expression_2 (a b : ℝ) : 
  7*a + 3*(a - 3*b) - 2*(b - a) = 12*a - 11*b := by sorry

end NUMINAMATH_CALUDE_simplify_expression_1_simplify_expression_2_l2650_265058


namespace NUMINAMATH_CALUDE_screw_nut_production_l2650_265076

theorem screw_nut_production (total_workers : ℕ) (screws_per_worker : ℕ) (nuts_per_worker : ℕ) 
  (screw_workers : ℕ) (nut_workers : ℕ) : 
  total_workers = 22 →
  screws_per_worker = 1200 →
  nuts_per_worker = 2000 →
  screw_workers = 10 →
  nut_workers = 12 →
  screw_workers + nut_workers = total_workers ∧
  2 * (screw_workers * screws_per_worker) = nut_workers * nuts_per_worker :=
by
  sorry

#check screw_nut_production

end NUMINAMATH_CALUDE_screw_nut_production_l2650_265076


namespace NUMINAMATH_CALUDE_baker_donuts_l2650_265014

theorem baker_donuts (total_donuts : ℕ) (boxes : ℕ) (extra_donuts : ℕ) : 
  boxes = 7 → 
  extra_donuts = 6 → 
  ∃ (n : ℕ), n > 0 ∧ total_donuts = 7 * n + 6 := by
  sorry

end NUMINAMATH_CALUDE_baker_donuts_l2650_265014


namespace NUMINAMATH_CALUDE_point_coordinates_l2650_265057

-- Define a point in 2D space
structure Point where
  x : ℝ
  y : ℝ

-- Define the second quadrant
def secondQuadrant (p : Point) : Prop :=
  p.x < 0 ∧ p.y > 0

-- Define distance to x-axis
def distToXAxis (p : Point) : ℝ :=
  |p.y|

-- Define distance to y-axis
def distToYAxis (p : Point) : ℝ :=
  |p.x|

theorem point_coordinates (M : Point) :
  secondQuadrant M ∧ distToXAxis M = 1 ∧ distToYAxis M = 2 →
  M.x = -2 ∧ M.y = 1 := by sorry

end NUMINAMATH_CALUDE_point_coordinates_l2650_265057


namespace NUMINAMATH_CALUDE_parabola_intersection_difference_l2650_265068

def parabola1 (x : ℝ) : ℝ := 3 * x^2 - 6 * x + 6
def parabola2 (x : ℝ) : ℝ := -2 * x^2 - 4 * x + 6

def intersection_points : Set ℝ := {x : ℝ | parabola1 x = parabola2 x}

theorem parabola_intersection_difference :
  ∃ (a c : ℝ), a ∈ intersection_points ∧ c ∈ intersection_points ∧ c ≥ a ∧ c - a = 2/5 :=
sorry

end NUMINAMATH_CALUDE_parabola_intersection_difference_l2650_265068


namespace NUMINAMATH_CALUDE_max_prob_div_by_10_min_nonzero_prob_div_by_10_l2650_265053

/-- A segment of natural numbers -/
structure Segment where
  start : ℕ
  length : ℕ
  h : length > 0

/-- The probability of a number in the segment being divisible by 10 -/
def prob_div_by_10 (s : Segment) : ℚ :=
  (s.length.divisors.filter (· % 10 = 0)).card / s.length

/-- The maximum probability of a number in any segment being divisible by 10 is 1 -/
theorem max_prob_div_by_10 : ∃ s : Segment, prob_div_by_10 s = 1 :=
  sorry

/-- The minimum non-zero probability of a number in any segment being divisible by 10 is 1/19 -/
theorem min_nonzero_prob_div_by_10 : 
  ∀ s : Segment, prob_div_by_10 s ≠ 0 → prob_div_by_10 s ≥ 1/19 :=
  sorry

end NUMINAMATH_CALUDE_max_prob_div_by_10_min_nonzero_prob_div_by_10_l2650_265053


namespace NUMINAMATH_CALUDE_gathering_drinks_l2650_265041

/-- Represents the number of people who took both wine and soda at a gathering -/
def people_took_both (total : ℕ) (wine : ℕ) (soda : ℕ) : ℕ :=
  wine + soda - total

theorem gathering_drinks (total : ℕ) (wine : ℕ) (soda : ℕ) 
  (h_total : total = 31) 
  (h_wine : wine = 26) 
  (h_soda : soda = 22) :
  people_took_both total wine soda = 17 := by
  sorry

#eval people_took_both 31 26 22

end NUMINAMATH_CALUDE_gathering_drinks_l2650_265041


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l2650_265050

def A : Set ℕ := {1, 2, 3}
def B : Set ℕ := {1, 3, 4}

theorem intersection_of_A_and_B : A ∩ B = {1, 3} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l2650_265050


namespace NUMINAMATH_CALUDE_expression_simplification_l2650_265097

theorem expression_simplification (x : ℝ) (h1 : x ≠ 2) (h2 : x ≠ -2) :
  ((((x + 2)^2 * (x^2 - 2*x + 2)^2) / (x^3 + 8)^2)^2 * 
   (((x - 2)^2 * (x^2 + 2*x + 2)^2) / (x^3 - 8)^2)^2) = 1 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l2650_265097


namespace NUMINAMATH_CALUDE_rectangle_tiling_l2650_265018

theorem rectangle_tiling (m n a b : ℕ) (hm : m > 0) (hn : n > 0) 
  (h_tiling : ∃ (h v : ℕ), a * b = h * m + v * n) :
  n ∣ a ∨ m ∣ b :=
sorry

end NUMINAMATH_CALUDE_rectangle_tiling_l2650_265018


namespace NUMINAMATH_CALUDE_triangle_properties_l2650_265000

theorem triangle_properties (a b c : ℝ) (A B C : ℝ) :
  0 < A ∧ 0 < B ∧ 0 < C ∧  -- Acute triangle condition
  A + B + C = Real.pi ∧
  a / Real.sin A = 2 * c / Real.sqrt 3 ∧
  c = Real.sqrt 7 ∧
  1/2 * a * b * Real.sin C = 3 * Real.sqrt 3 / 2 →
  C = Real.pi / 3 ∧ a^2 + b^2 = 13 := by
sorry

end NUMINAMATH_CALUDE_triangle_properties_l2650_265000


namespace NUMINAMATH_CALUDE_bakers_sales_l2650_265087

/-- Baker's cake and pastry sales problem -/
theorem bakers_sales (cakes_made pastries_made cakes_sold pastries_sold : ℕ) 
  (h1 : cakes_made = 475)
  (h2 : pastries_made = 539)
  (h3 : cakes_sold = 358)
  (h4 : pastries_sold = 297) :
  cakes_sold - pastries_sold = 61 := by
  sorry

end NUMINAMATH_CALUDE_bakers_sales_l2650_265087


namespace NUMINAMATH_CALUDE_cone_generatrix_length_l2650_265056

/-- Represents a cone with specific properties -/
structure Cone where
  r : ℝ  -- radius of the base
  h : ℝ  -- height of the cone
  l : ℝ  -- length of the generatrix
  lateral_area_eq : π * r * l = 2 * π * r^2  -- lateral surface area is twice the base area
  volume_eq : (1/3) * π * r^2 * h = 9 * Real.sqrt 3 * π  -- volume is 9√3π

/-- Theorem stating that a cone with the given properties has a generatrix of length 6 -/
theorem cone_generatrix_length (c : Cone) : c.l = 6 := by
  sorry

end NUMINAMATH_CALUDE_cone_generatrix_length_l2650_265056


namespace NUMINAMATH_CALUDE_building_meets_safety_regulations_l2650_265032

/-- Represents the school building configuration and safety requirements -/
structure SchoolBuilding where
  floors : Nat
  classrooms_per_floor : Nat
  main_doors : Nat
  side_doors : Nat
  students_all_doors_2min : Nat
  students_half_doors_4min : Nat
  emergency_efficiency_decrease : Rat
  evacuation_time_limit : Nat
  students_per_classroom : Nat

/-- Calculates the flow rate of students through doors -/
def calculate_flow_rates (building : SchoolBuilding) : Nat × Nat :=
  sorry

/-- Checks if the building meets safety regulations -/
def meets_safety_regulations (building : SchoolBuilding) : Bool :=
  sorry

/-- Theorem stating that the given building configuration meets safety regulations -/
theorem building_meets_safety_regulations :
  let building : SchoolBuilding := {
    floors := 4,
    classrooms_per_floor := 8,
    main_doors := 2,
    side_doors := 2,
    students_all_doors_2min := 560,
    students_half_doors_4min := 800,
    emergency_efficiency_decrease := 1/5,
    evacuation_time_limit := 5,
    students_per_classroom := 45
  }
  meets_safety_regulations building = true :=
sorry

end NUMINAMATH_CALUDE_building_meets_safety_regulations_l2650_265032


namespace NUMINAMATH_CALUDE_negation_of_forall_is_exists_not_l2650_265039

variable (S : Set ℝ)

-- Define the original property
def P (x : ℝ) : Prop := x^2 = x

-- State the theorem
theorem negation_of_forall_is_exists_not (h : ∀ x ∈ S, P x) : 
  ¬(∀ x ∈ S, P x) ↔ ∃ x ∈ S, ¬(P x) := by sorry

end NUMINAMATH_CALUDE_negation_of_forall_is_exists_not_l2650_265039


namespace NUMINAMATH_CALUDE_intersection_A_complement_B_l2650_265077

def U : Set ℝ := Set.univ

def A : Set ℝ := {-3, -1, 0, 1, 3}

def B : Set ℝ := {x | |x - 1| > 1}

theorem intersection_A_complement_B : A ∩ (U \ B) = {0, 1} := by
  sorry

end NUMINAMATH_CALUDE_intersection_A_complement_B_l2650_265077


namespace NUMINAMATH_CALUDE_remainder_problem_l2650_265047

theorem remainder_problem (n : ℤ) (h : n ≡ 3 [ZMOD 4]) : 7 * n ≡ 1 [ZMOD 5] := by
  sorry

end NUMINAMATH_CALUDE_remainder_problem_l2650_265047


namespace NUMINAMATH_CALUDE_existence_of_mn_l2650_265043

theorem existence_of_mn : ∃ (m n : ℕ), ∀ (a b : ℝ), 
  ((-2 * a^n * b^n)^m + (3 * a^m * b^m)^n) = a^6 * b^6 := by
  sorry

end NUMINAMATH_CALUDE_existence_of_mn_l2650_265043


namespace NUMINAMATH_CALUDE_vending_machine_probability_l2650_265090

/-- The number of toys in the vending machine -/
def num_toys : ℕ := 10

/-- The price of the cheapest toy in cents -/
def min_price : ℕ := 50

/-- The price increment between toys in cents -/
def price_increment : ℕ := 25

/-- The price of Sam's favorite toy in cents -/
def favorite_toy_price : ℕ := 225

/-- The number of quarters Sam has initially -/
def initial_quarters : ℕ := 12

/-- The value of Sam's bill in cents -/
def bill_value : ℕ := 2000

/-- The probability that Sam has to break his twenty-dollar bill -/
def probability_break_bill : ℚ := 8/9

theorem vending_machine_probability :
  let total_permutations := Nat.factorial num_toys
  let favorable_permutations := Nat.factorial (num_toys - 1) + Nat.factorial (num_toys - 2)
  probability_break_bill = 1 - (favorable_permutations : ℚ) / total_permutations :=
by sorry

end NUMINAMATH_CALUDE_vending_machine_probability_l2650_265090


namespace NUMINAMATH_CALUDE_grunters_win_probability_l2650_265035

theorem grunters_win_probability (p : ℝ) (n : ℕ) (h1 : p = 2/3) (h2 : n = 6) :
  p^n = 64/729 := by
  sorry

end NUMINAMATH_CALUDE_grunters_win_probability_l2650_265035


namespace NUMINAMATH_CALUDE_equal_area_split_line_slope_l2650_265083

/-- Represents a circle in 2D space -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Represents a line in 2D space -/
structure Line where
  slope : ℝ
  passesThrough : ℝ × ℝ

/-- Checks if a line splits the area of circles equally -/
def splitAreaEqually (line : Line) (circles : List Circle) : Prop :=
  sorry

/-- The main theorem -/
theorem equal_area_split_line_slope :
  let circles : List Circle := [
    { center := (10, 100), radius := 4 },
    { center := (13, 82),  radius := 4 },
    { center := (15, 90),  radius := 4 }
  ]
  let line : Line := { slope := 0.5, passesThrough := (13, 82) }
  splitAreaEqually line circles ∧ 
  ∀ (m : ℝ), splitAreaEqually { slope := m, passesThrough := (13, 82) } circles → 
    |m| = 0.5 := by
  sorry

end NUMINAMATH_CALUDE_equal_area_split_line_slope_l2650_265083


namespace NUMINAMATH_CALUDE_complex_number_location_l2650_265052

theorem complex_number_location : ∃ (x y : ℝ), x > 0 ∧ y > 0 ∧ (Complex.I : ℂ) / (3 + Complex.I) = ⟨x, y⟩ := by
  sorry

end NUMINAMATH_CALUDE_complex_number_location_l2650_265052


namespace NUMINAMATH_CALUDE_arithmetic_sequence_problem_l2650_265029

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_problem (a : ℕ → ℝ) 
  (h_arithmetic : arithmetic_sequence a)
  (h_a5 : a 5 = 6)
  (h_a8 : a 8 = 15) :
  (∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d ∧ d = 3) ∧ a 11 = 24 :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_problem_l2650_265029


namespace NUMINAMATH_CALUDE_expand_binomials_l2650_265031

theorem expand_binomials (a : ℝ) : (a + 2) * (a - 3) = a^2 - a - 6 := by
  sorry

end NUMINAMATH_CALUDE_expand_binomials_l2650_265031


namespace NUMINAMATH_CALUDE_typing_time_proof_l2650_265033

def original_speed : ℕ := 212
def speed_reduction : ℕ := 40
def document_length : ℕ := 3440

theorem typing_time_proof :
  let new_speed := original_speed - speed_reduction
  document_length / new_speed = 20 := by sorry

end NUMINAMATH_CALUDE_typing_time_proof_l2650_265033


namespace NUMINAMATH_CALUDE_square_area_ratio_l2650_265042

theorem square_area_ratio (side_C side_D : ℝ) (h1 : side_C = 48) (h2 : side_D = 60) :
  (side_C ^ 2) / (side_D ^ 2) = 16 / 25 := by
  sorry

end NUMINAMATH_CALUDE_square_area_ratio_l2650_265042


namespace NUMINAMATH_CALUDE_lost_in_mountains_second_group_size_l2650_265051

theorem lost_in_mountains (initial_people : ℕ) (initial_days : ℕ) (days_after_sharing : ℕ) : ℕ :=
  let initial_portions := initial_people * initial_days
  let remaining_portions := initial_portions - initial_people
  let total_people := initial_people + (remaining_portions / (days_after_sharing + 1) - initial_people)
  remaining_portions / (days_after_sharing + 1) - initial_people

theorem second_group_size :
  lost_in_mountains 9 5 3 = 3 := by
  sorry

end NUMINAMATH_CALUDE_lost_in_mountains_second_group_size_l2650_265051


namespace NUMINAMATH_CALUDE_conference_duration_l2650_265045

theorem conference_duration (h₁ : 9 > 0) (h₂ : 11 > 0) (h₃ : 12 > 0) :
  Nat.lcm (Nat.lcm 9 11) 12 = 396 := by
  sorry

end NUMINAMATH_CALUDE_conference_duration_l2650_265045


namespace NUMINAMATH_CALUDE_point_distance_on_line_l2650_265021

/-- Given a line with equation x - 5/2y + 1 = 0 and two points on this line,
    if the x-coordinate of the second point is 1/2 unit more than the x-coordinate of the first point,
    then the difference between their x-coordinates is 1/2. -/
theorem point_distance_on_line (m n a : ℝ) : 
  (m - (5/2) * n + 1 = 0) →  -- First point (m, n) satisfies the line equation
  (m + a - (5/2) * (n + 1) + 1 = 0) →  -- Second point (m + a, n + 1) satisfies the line equation
  (m + a = m + 1/2) →  -- x-coordinate of second point is 1/2 more than first point
  a = 1/2 := by
sorry

end NUMINAMATH_CALUDE_point_distance_on_line_l2650_265021


namespace NUMINAMATH_CALUDE_inequality_impossibility_l2650_265093

theorem inequality_impossibility (a b c d : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) :
  ¬(a + b < c + d ∧ (a + b) * (c + d) < a * b + c * d ∧ (a + b) * c * d < a * b * (c + d)) := by
  sorry

end NUMINAMATH_CALUDE_inequality_impossibility_l2650_265093


namespace NUMINAMATH_CALUDE_min_rows_correct_l2650_265038

/-- The minimum number of rows required to seat students under given conditions -/
def min_rows (total_students : ℕ) (max_per_school : ℕ) (seats_per_row : ℕ) : ℕ :=
  -- Definition to be proved
  15

theorem min_rows_correct (total_students max_per_school seats_per_row : ℕ) 
  (h1 : total_students = 2016)
  (h2 : max_per_school = 40)
  (h3 : seats_per_row = 168)
  (h4 : ∀ (school_size : ℕ), school_size ≤ max_per_school → school_size ≤ seats_per_row) :
  min_rows total_students max_per_school seats_per_row = 15 := by
  sorry

#eval min_rows 2016 40 168

end NUMINAMATH_CALUDE_min_rows_correct_l2650_265038


namespace NUMINAMATH_CALUDE_symmetry_about_xoz_plane_l2650_265026

-- Define a point in 3D space
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

-- Define the symmetry operation about the xOz plane
def symmetryAboutXOZ (p : Point3D) : Point3D :=
  { x := p.x, y := -p.y, z := p.z }

-- Theorem statement
theorem symmetry_about_xoz_plane :
  let A : Point3D := { x := 3, y := -2, z := 5 }
  let A_sym : Point3D := symmetryAboutXOZ A
  A_sym = { x := 3, y := 2, z := 5 } := by sorry

end NUMINAMATH_CALUDE_symmetry_about_xoz_plane_l2650_265026


namespace NUMINAMATH_CALUDE_triangle_area_from_smaller_triangles_l2650_265074

/-- Given a triangle divided into six parts by lines parallel to its sides,
    this theorem states that the area of the original triangle is equal to
    (√t₁ + √t₂ + √t₃)², where t₁, t₂, and t₃ are the areas of three of the
    smaller triangles formed. -/
theorem triangle_area_from_smaller_triangles 
  (t₁ t₂ t₃ : ℝ) 
  (h₁ : t₁ > 0) 
  (h₂ : t₂ > 0) 
  (h₃ : t₃ > 0) :
  ∃ T : ℝ, T > 0 ∧ T = (Real.sqrt t₁ + Real.sqrt t₂ + Real.sqrt t₃)^2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_from_smaller_triangles_l2650_265074


namespace NUMINAMATH_CALUDE_find_number_l2650_265084

theorem find_number (n x : ℝ) (h1 : n * (x - 1) = 21) (h2 : x = 4) : n = 7 := by
  sorry

end NUMINAMATH_CALUDE_find_number_l2650_265084


namespace NUMINAMATH_CALUDE_min_value_on_ellipse_l2650_265022

/-- The minimum value of d for points on the given ellipse --/
theorem min_value_on_ellipse :
  let ellipse := {P : ℝ × ℝ | (P.1^2 / 4) + (P.2^2 / 3) = 1}
  let d (P : ℝ × ℝ) := Real.sqrt (P.1^2 + P.2^2 + 4*P.2 + 4) - P.1/2
  ∀ P ∈ ellipse, d P ≥ 2 * Real.sqrt 2 - 1 ∧ ∃ Q ∈ ellipse, d Q = 2 * Real.sqrt 2 - 1 :=
by sorry


end NUMINAMATH_CALUDE_min_value_on_ellipse_l2650_265022


namespace NUMINAMATH_CALUDE_mark_fruit_count_l2650_265088

/-- The number of pieces of fruit Mark had at the beginning of the week -/
def total_fruit (kept_for_next_week : ℕ) (brought_to_school : ℕ) (eaten_first_four_days : ℕ) : ℕ :=
  kept_for_next_week + brought_to_school + eaten_first_four_days

/-- Theorem stating that Mark had 10 pieces of fruit at the beginning of the week -/
theorem mark_fruit_count : total_fruit 2 3 5 = 10 := by
  sorry

end NUMINAMATH_CALUDE_mark_fruit_count_l2650_265088


namespace NUMINAMATH_CALUDE_inequality_chain_l2650_265010

theorem inequality_chain (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  a + b + c ≤ (a^2 + b^2)/(2*c) + (a^2 + c^2)/(2*b) + (b^2 + c^2)/(2*a) ∧
  (a^2 + b^2)/(2*c) + (a^2 + c^2)/(2*b) + (b^2 + c^2)/(2*a) ≤ a^3/(b*c) + b^3/(a*c) + c^3/(a*b) :=
by sorry

end NUMINAMATH_CALUDE_inequality_chain_l2650_265010


namespace NUMINAMATH_CALUDE_final_dislikes_is_300_l2650_265091

/-- Represents the number of likes and dislikes on a YouTube video -/
structure VideoStats where
  likes : ℕ
  dislikes : ℕ

/-- Calculates the final number of dislikes after changes -/
def finalDislikes (original : VideoStats) : ℕ :=
  3 * original.dislikes

/-- Theorem: Given the conditions, the final number of dislikes is 300 -/
theorem final_dislikes_is_300 (original : VideoStats) 
    (h1 : original.likes = 3 * original.dislikes)
    (h2 : original.likes = 100 + 2 * original.dislikes) : 
  finalDislikes original = 300 := by
  sorry

#eval finalDislikes {likes := 300, dislikes := 100}

end NUMINAMATH_CALUDE_final_dislikes_is_300_l2650_265091


namespace NUMINAMATH_CALUDE_forester_count_impossible_l2650_265002

/-- Represents a circle in the forest --/
structure Circle where
  id : Nat
  trees : Finset Nat

/-- Represents the forest with circles and pine trees --/
structure Forest where
  circles : Finset Circle
  total_trees : Finset Nat

/-- The property that each circle contains exactly 3 distinct trees --/
def validCount (f : Forest) : Prop :=
  ∀ c ∈ f.circles, c.trees.card = 3

/-- The property that all trees in circles are from the total set of trees --/
def validTrees (f : Forest) : Prop :=
  ∀ c ∈ f.circles, c.trees ⊆ f.total_trees

/-- The main theorem stating the impossibility of the forester's count --/
theorem forester_count_impossible (f : Forest) :
  f.circles.card = 5 → validCount f → validTrees f → False := by
  sorry

end NUMINAMATH_CALUDE_forester_count_impossible_l2650_265002


namespace NUMINAMATH_CALUDE_pie_chart_most_suitable_for_air_composition_l2650_265063

/-- Represents different types of graphs -/
inductive GraphType
  | BarGraph
  | LineGraph
  | PieChart
  | Histogram

/-- Represents a component of air -/
structure AirComponent where
  name : String
  percentage : Float

/-- Determines if a graph type is suitable for representing percentage composition -/
def isSuitableForPercentageComposition (graphType : GraphType) : Prop :=
  match graphType with
  | GraphType.PieChart => True
  | _ => False

/-- The air composition representation problem -/
theorem pie_chart_most_suitable_for_air_composition 
  (components : List AirComponent) 
  (hComponents : components.all (λ c => c.percentage ≥ 0 ∧ c.percentage ≤ 100)) 
  (hTotalPercentage : components.foldl (λ acc c => acc + c.percentage) 0 = 100) :
  isSuitableForPercentageComposition GraphType.PieChart ∧ 
  (∀ g : GraphType, isSuitableForPercentageComposition g → g = GraphType.PieChart) :=
sorry

end NUMINAMATH_CALUDE_pie_chart_most_suitable_for_air_composition_l2650_265063


namespace NUMINAMATH_CALUDE_quadratic_real_roots_l2650_265020

theorem quadratic_real_roots (a b c : ℤ) (h : a ≠ b ∧ b ≠ c ∧ a ≠ c) :
  ∃ x : ℝ, x^2 + 2*x*(a : ℝ) + 3*((b : ℝ) + (c : ℝ)) = 0 :=
sorry

end NUMINAMATH_CALUDE_quadratic_real_roots_l2650_265020


namespace NUMINAMATH_CALUDE_cab_cost_for_week_long_event_l2650_265009

/-- Calculates the total cost of cab rides for a week-long event -/
def total_cab_cost (days : ℕ) (distance : ℝ) (cost_per_mile : ℝ) : ℝ :=
  2 * days * distance * cost_per_mile

/-- Proves that the total cost of cab rides for the given conditions is $7000 -/
theorem cab_cost_for_week_long_event :
  total_cab_cost 7 200 2.5 = 7000 := by
  sorry

end NUMINAMATH_CALUDE_cab_cost_for_week_long_event_l2650_265009


namespace NUMINAMATH_CALUDE_tangent_line_segment_region_area_l2650_265048

theorem tangent_line_segment_region_area (r : ℝ) (h : r = 3) : 
  let outer_radius := r * Real.sqrt 2
  let inner_area := π * r^2
  let outer_area := π * outer_radius^2
  outer_area - inner_area = 9 * π :=
by sorry

end NUMINAMATH_CALUDE_tangent_line_segment_region_area_l2650_265048


namespace NUMINAMATH_CALUDE_wedding_drinks_l2650_265079

theorem wedding_drinks (total_guests : ℕ) (num_drink_types : ℕ) 
  (champagne_glasses_per_guest : ℕ) (champagne_servings_per_bottle : ℕ)
  (wine_glasses_per_guest : ℕ) (wine_servings_per_bottle : ℕ)
  (juice_glasses_per_guest : ℕ) (juice_servings_per_bottle : ℕ)
  (h1 : total_guests = 120)
  (h2 : num_drink_types = 3)
  (h3 : champagne_glasses_per_guest = 2)
  (h4 : champagne_servings_per_bottle = 6)
  (h5 : wine_glasses_per_guest = 1)
  (h6 : wine_servings_per_bottle = 5)
  (h7 : juice_glasses_per_guest = 1)
  (h8 : juice_servings_per_bottle = 4) :
  let guests_per_drink_type := total_guests / num_drink_types
  let juice_bottles_needed := (guests_per_drink_type * juice_glasses_per_guest + juice_servings_per_bottle - 1) / juice_servings_per_bottle
  juice_bottles_needed = 10 := by
sorry

end NUMINAMATH_CALUDE_wedding_drinks_l2650_265079


namespace NUMINAMATH_CALUDE_contractor_problem_l2650_265027

theorem contractor_problem (total_days : ℕ) (absent_workers : ℕ) (actual_days : ℕ) :
  total_days = 6 →
  absent_workers = 7 →
  actual_days = 10 →
  ∃ (original_workers : ℕ), 
    original_workers * total_days = (original_workers - absent_workers) * actual_days ∧ 
    original_workers = 18 :=
by sorry

end NUMINAMATH_CALUDE_contractor_problem_l2650_265027


namespace NUMINAMATH_CALUDE_x_squared_minus_y_squared_l2650_265098

theorem x_squared_minus_y_squared (x y : ℚ) 
  (h1 : x + y = 8 / 15) (h2 : x - y = 1 / 45) : x^2 - y^2 = 8 / 675 := by
  sorry

end NUMINAMATH_CALUDE_x_squared_minus_y_squared_l2650_265098


namespace NUMINAMATH_CALUDE_average_of_abc_is_three_l2650_265086

theorem average_of_abc_is_three (A B C : ℝ) 
  (eq1 : 1503 * C - 3006 * A = 6012)
  (eq2 : 1503 * B + 4509 * A = 7509) :
  (A + B + C) / 3 = 3 := by
sorry

end NUMINAMATH_CALUDE_average_of_abc_is_three_l2650_265086


namespace NUMINAMATH_CALUDE_max_x_minus_y_l2650_265006

theorem max_x_minus_y (x y : ℝ) (h : x^2 + y^2 - 4*x - 2*y - 4 = 0) :
  ∃ (z : ℝ), z = x - y ∧ z ≤ 1 + 3 * Real.sqrt 2 ∧
  ∀ (w : ℝ), w = x - y → w ≤ z :=
sorry

end NUMINAMATH_CALUDE_max_x_minus_y_l2650_265006


namespace NUMINAMATH_CALUDE_sum_subfixed_points_ln_exp_is_zero_l2650_265023

/-- A sub-fixed point of a function f is a real number t such that f(t) = -t -/
def SubFixedPoint (f : ℝ → ℝ) (t : ℝ) : Prop := f t = -t

/-- The natural logarithm function -/
noncomputable def ln : ℝ → ℝ := Real.log

/-- The exponential function -/
noncomputable def exp : ℝ → ℝ := Real.exp

/-- The sub-fixed point of the natural logarithm function -/
noncomputable def t : ℝ := sorry

/-- Statement: The sum of sub-fixed points of ln and exp is zero -/
theorem sum_subfixed_points_ln_exp_is_zero :
  SubFixedPoint ln t ∧ SubFixedPoint exp (-t) → t + (-t) = 0 := by sorry

end NUMINAMATH_CALUDE_sum_subfixed_points_ln_exp_is_zero_l2650_265023


namespace NUMINAMATH_CALUDE_kelly_string_cheese_problem_l2650_265012

/-- The number of string cheeses Kelly's youngest child eats per day -/
def youngest_daily_cheese : ℕ := by sorry

theorem kelly_string_cheese_problem :
  let days_per_week : ℕ := 5
  let oldest_daily_cheese : ℕ := 2
  let cheeses_per_pack : ℕ := 30
  let weeks : ℕ := 4
  let packs_needed : ℕ := 2

  youngest_daily_cheese = 1 := by sorry

end NUMINAMATH_CALUDE_kelly_string_cheese_problem_l2650_265012


namespace NUMINAMATH_CALUDE_opposite_numbers_sum_power_l2650_265073

/-- If a and b are opposite numbers, then (a+b)^2023 = 0 -/
theorem opposite_numbers_sum_power (a b : ℝ) : a = -b → (a + b)^2023 = 0 := by
  sorry

end NUMINAMATH_CALUDE_opposite_numbers_sum_power_l2650_265073


namespace NUMINAMATH_CALUDE_radar_coverage_l2650_265015

noncomputable def n : ℕ := 7
def r : ℝ := 41
def w : ℝ := 18

theorem radar_coverage (n : ℕ) (r w : ℝ) 
  (h_n : n = 7) 
  (h_r : r = 41) 
  (h_w : w = 18) :
  ∃ (max_distance area : ℝ),
    max_distance = 40 / Real.sin (180 / n * π / 180) ∧
    area = 1440 * π / Real.tan (180 / n * π / 180) := by
  sorry

end NUMINAMATH_CALUDE_radar_coverage_l2650_265015


namespace NUMINAMATH_CALUDE_quadratic_rewrite_product_l2650_265066

/-- Given a quadratic equation 16x^2 - 40x - 24 that can be rewritten as (dx + e)^2 + f,
    where d, e, and f are integers, prove that de = -20 -/
theorem quadratic_rewrite_product (d e f : ℤ) : 
  (∀ x, 16 * x^2 - 40 * x - 24 = (d * x + e)^2 + f) → d * e = -20 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_rewrite_product_l2650_265066


namespace NUMINAMATH_CALUDE_odd_function_sum_l2650_265082

-- Define an odd function
def OddFunction (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

-- Main theorem
theorem odd_function_sum (f : ℝ → ℝ) (h1 : OddFunction f) (h2 : f (-1) = 2) :
  f 0 + f 1 = -2 := by
  sorry

end NUMINAMATH_CALUDE_odd_function_sum_l2650_265082


namespace NUMINAMATH_CALUDE_residue_mod_13_l2650_265089

theorem residue_mod_13 : (250 * 11 - 20 * 6 + 5^2) % 13 = 3 := by sorry

end NUMINAMATH_CALUDE_residue_mod_13_l2650_265089


namespace NUMINAMATH_CALUDE_mens_wages_l2650_265005

theorem mens_wages (men women boys : ℕ) (total_earnings : ℚ) : 
  men = 5 →
  5 * women = men * 8 →
  total_earnings = 60 →
  total_earnings / (3 * men) = 4 :=
by sorry

end NUMINAMATH_CALUDE_mens_wages_l2650_265005


namespace NUMINAMATH_CALUDE_cafeteria_discussion_participation_l2650_265049

theorem cafeteria_discussion_participation 
  (students_like : ℕ) 
  (students_dislike : ℕ) 
  (h1 : students_like = 383) 
  (h2 : students_dislike = 431) : 
  students_like + students_dislike = 814 := by
sorry

end NUMINAMATH_CALUDE_cafeteria_discussion_participation_l2650_265049


namespace NUMINAMATH_CALUDE_closest_integer_to_cube_root_1728_l2650_265055

theorem closest_integer_to_cube_root_1728 : 
  ∀ n : ℤ, |n - (1728 : ℝ)^(1/3)| ≥ |12 - (1728 : ℝ)^(1/3)| :=
by
  sorry

end NUMINAMATH_CALUDE_closest_integer_to_cube_root_1728_l2650_265055


namespace NUMINAMATH_CALUDE_water_storage_calculation_l2650_265064

/-- Calculates the total volume of water stored in jars of different sizes -/
theorem water_storage_calculation (total_jars : ℕ) (h1 : total_jars = 24) :
  let jars_per_size := total_jars / 3
  let quart_volume := jars_per_size * (1 / 4 : ℚ)
  let half_gallon_volume := jars_per_size * (1 / 2 : ℚ)
  let gallon_volume := jars_per_size * 1
  quart_volume + half_gallon_volume + gallon_volume = 14 := by
  sorry

#check water_storage_calculation

end NUMINAMATH_CALUDE_water_storage_calculation_l2650_265064


namespace NUMINAMATH_CALUDE_exam_score_problem_l2650_265062

theorem exam_score_problem (total_questions : ℕ) (correct_score : ℤ) (wrong_score : ℤ) (total_score : ℤ) :
  total_questions = 60 →
  correct_score = 4 →
  wrong_score = -1 →
  total_score = 150 →
  ∃ (correct_answers : ℕ),
    correct_answers ≤ total_questions ∧
    correct_score * correct_answers + wrong_score * (total_questions - correct_answers) = total_score ∧
    correct_answers = 42 :=
by sorry

end NUMINAMATH_CALUDE_exam_score_problem_l2650_265062


namespace NUMINAMATH_CALUDE_bridge_length_l2650_265034

/-- The length of a bridge given train specifications and crossing time -/
theorem bridge_length
  (train_length : ℝ)
  (train_speed_kmh : ℝ)
  (crossing_time : ℝ)
  (h1 : train_length = 90)
  (h2 : train_speed_kmh = 45)
  (h3 : crossing_time = 30) :
  train_speed_kmh * (1000 / 3600) * crossing_time - train_length = 285 :=
by sorry

end NUMINAMATH_CALUDE_bridge_length_l2650_265034


namespace NUMINAMATH_CALUDE_jesse_book_reading_l2650_265072

theorem jesse_book_reading (pages_read pages_left : ℕ) 
  (h1 : pages_read = 83) 
  (h2 : pages_left = 166) : 
  (pages_read : ℚ) / (pages_read + pages_left) = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_jesse_book_reading_l2650_265072


namespace NUMINAMATH_CALUDE_large_balls_can_make_l2650_265092

/-- The number of rubber bands in a small ball -/
def small_ball_bands : ℕ := 50

/-- The number of rubber bands in a large ball -/
def large_ball_bands : ℕ := 300

/-- The total number of rubber bands Michael brought to class -/
def total_bands : ℕ := 5000

/-- The number of small balls Michael has already made -/
def small_balls_made : ℕ := 22

/-- The number of large balls Michael can make with the remaining rubber bands -/
theorem large_balls_can_make : ℕ := by
  sorry

end NUMINAMATH_CALUDE_large_balls_can_make_l2650_265092


namespace NUMINAMATH_CALUDE_mixing_ways_count_l2650_265046

/-- Represents a container used in the mixing process -/
inductive Container
| Barrel : Container  -- 12-liter barrel
| Small : Container   -- 2-liter container
| Medium : Container  -- 8-liter container

/-- Represents a liquid type -/
inductive Liquid
| Wine : Liquid
| Water : Liquid

/-- Represents a mixing operation -/
structure MixingOperation :=
(source : Container)
(destination : Container)
(liquid : Liquid)
(amount : ℕ)

/-- The set of all valid mixing operations -/
def valid_operations : Set MixingOperation := sorry

/-- A mixing sequence is a list of mixing operations -/
def MixingSequence := List MixingOperation

/-- Checks if a mixing sequence results in the correct final mixture -/
def is_valid_mixture (seq : MixingSequence) : Prop := sorry

/-- The number of distinct valid mixing sequences -/
def num_valid_sequences : ℕ := sorry

/-- Main theorem: There are exactly 32 ways to mix the liquids -/
theorem mixing_ways_count :
  num_valid_sequences = 32 := by sorry

end NUMINAMATH_CALUDE_mixing_ways_count_l2650_265046


namespace NUMINAMATH_CALUDE_factorization_of_polynomial_l2650_265085

theorem factorization_of_polynomial (b : ℝ) :
  348 * b^2 + 87 * b + 261 = 87 * (4 * b^2 + b + 3) := by
  sorry

end NUMINAMATH_CALUDE_factorization_of_polynomial_l2650_265085


namespace NUMINAMATH_CALUDE_company_production_l2650_265004

/-- The number of bottles produced daily by a company -/
def bottles_produced (cases_required : ℕ) (bottles_per_case : ℕ) : ℕ :=
  cases_required * bottles_per_case

/-- Theorem stating the company's daily water bottle production -/
theorem company_production :
  bottles_produced 10000 12 = 120000 := by
  sorry

end NUMINAMATH_CALUDE_company_production_l2650_265004


namespace NUMINAMATH_CALUDE_triangle_area_l2650_265001

/-- The area of a triangle with base 4 and height 5 is 10 -/
theorem triangle_area : 
  ∀ (base height area : ℝ), 
  base = 4 → 
  height = 5 → 
  area = (base * height) / 2 → 
  area = 10 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_l2650_265001
