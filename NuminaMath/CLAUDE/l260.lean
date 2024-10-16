import Mathlib

namespace NUMINAMATH_CALUDE_real_part_of_z_l260_26070

theorem real_part_of_z (z : ℂ) (h : (z + 1).re = 0) : z.re = -1 := by
  sorry

end NUMINAMATH_CALUDE_real_part_of_z_l260_26070


namespace NUMINAMATH_CALUDE_ellipse_dist_to_directrix_l260_26034

/-- An ellipse with semi-major axis a and semi-minor axis b -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h : 0 < b ∧ b < a

/-- A point on an ellipse -/
structure PointOnEllipse (E : Ellipse) where
  x : ℝ
  y : ℝ
  h : x^2 / E.a^2 + y^2 / E.b^2 = 1

/-- The distance from a point to the left focus of an ellipse -/
def distToLeftFocus (E : Ellipse) (P : PointOnEllipse E) : ℝ := sorry

/-- The distance from a point to the right directrix of an ellipse -/
def distToRightDirectrix (E : Ellipse) (P : PointOnEllipse E) : ℝ := sorry

/-- Theorem: For the given ellipse, if a point on the ellipse is at distance 8 from the left focus,
    then its distance to the right directrix is 5/2 -/
theorem ellipse_dist_to_directrix (E : Ellipse) (P : PointOnEllipse E) :
  E.a = 5 ∧ E.b = 3 ∧ distToLeftFocus E P = 8 → distToRightDirectrix E P = 5/2 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_dist_to_directrix_l260_26034


namespace NUMINAMATH_CALUDE_music_listening_time_l260_26092

/-- Given music with 200 beats per minute and 168000 beats heard per week,
    prove that the number of hours of music listened to per day is 2. -/
theorem music_listening_time (beats_per_minute : ℕ) (beats_per_week : ℕ) 
  (h1 : beats_per_minute = 200)
  (h2 : beats_per_week = 168000) :
  (beats_per_week / 7) / beats_per_minute / 60 = 2 := by
  sorry

#check music_listening_time

end NUMINAMATH_CALUDE_music_listening_time_l260_26092


namespace NUMINAMATH_CALUDE_chips_in_bag_is_81_l260_26053

/-- Represents the number of chocolate chips in a bag -/
def chips_in_bag : ℕ := sorry

/-- Represents the number of batches made from one bag of chips -/
def batches_per_bag : ℕ := 3

/-- Represents the number of cookies in each batch -/
def cookies_per_batch : ℕ := 3

/-- Represents the number of chocolate chips in each cookie -/
def chips_per_cookie : ℕ := 9

/-- Theorem stating that the number of chips in a bag is 81 -/
theorem chips_in_bag_is_81 : chips_in_bag = 81 := by sorry

end NUMINAMATH_CALUDE_chips_in_bag_is_81_l260_26053


namespace NUMINAMATH_CALUDE_exists_idempotent_l260_26065

/-- A custom binary operation on a finite set -/
class CustomOperation (α : Type*) [Fintype α] where
  op : α → α → α

/-- Axioms for the custom operation -/
class CustomOperationAxioms (α : Type*) [Fintype α] [CustomOperation α] where
  closure : ∀ (a b : α), CustomOperation.op a b ∈ (Finset.univ : Finset α)
  property : ∀ (a b : α), CustomOperation.op (CustomOperation.op a b) a = b

/-- Theorem: There exists an element that is idempotent under the custom operation -/
theorem exists_idempotent (α : Type*) [Fintype α] [CustomOperation α] [CustomOperationAxioms α] :
  ∃ (a : α), CustomOperation.op a a = a :=
sorry

end NUMINAMATH_CALUDE_exists_idempotent_l260_26065


namespace NUMINAMATH_CALUDE_money_distribution_l260_26061

theorem money_distribution (A B C : ℕ) 
  (total : A + B + C = 500)
  (AC_sum : A + C = 200)
  (BC_sum : B + C = 340) :
  C = 40 := by sorry

end NUMINAMATH_CALUDE_money_distribution_l260_26061


namespace NUMINAMATH_CALUDE_eventB_mutually_exclusive_not_complementary_to_eventA_l260_26043

/-- Represents the possible outcomes when drawing balls from a bag -/
inductive BallDraw
  | TwoBlack
  | ThreeBlack
  | OneBlack
  | NoBlack

/-- The total number of balls in the bag -/
def totalBalls : ℕ := 6

/-- The number of black balls in the bag -/
def blackBalls : ℕ := 3

/-- The number of red balls in the bag -/
def redBalls : ℕ := 3

/-- The number of balls drawn -/
def ballsDrawn : ℕ := 3

/-- Event A: At least 2 black balls are drawn -/
def eventA : Set BallDraw := {BallDraw.TwoBlack, BallDraw.ThreeBlack}

/-- Event B: Exactly 1 black ball is drawn -/
def eventB : Set BallDraw := {BallDraw.OneBlack}

/-- Two events are mutually exclusive if their intersection is empty -/
def mutuallyExclusive (S T : Set BallDraw) : Prop := S ∩ T = ∅

/-- Two events are complementary if their union is the entire sample space -/
def complementary (S T : Set BallDraw) : Prop := S ∪ T = Set.univ

theorem eventB_mutually_exclusive_not_complementary_to_eventA :
  mutuallyExclusive eventA eventB ∧ ¬complementary eventA eventB := by sorry

end NUMINAMATH_CALUDE_eventB_mutually_exclusive_not_complementary_to_eventA_l260_26043


namespace NUMINAMATH_CALUDE_rationalize_denominator_l260_26042

theorem rationalize_denominator : 
  (Real.sqrt 12 + Real.sqrt 5) / (Real.sqrt 3 + Real.sqrt 5) = (Real.sqrt 15 - 1) / 2 := by
  sorry

end NUMINAMATH_CALUDE_rationalize_denominator_l260_26042


namespace NUMINAMATH_CALUDE_difference_multiple_of_nine_l260_26079

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

def reverse_digits (n : ℕ) : ℕ :=
  (n % 10) * 10 + (n / 10)

theorem difference_multiple_of_nine (q r : ℕ) :
  is_two_digit q ∧ 
  is_two_digit r ∧ 
  r = reverse_digits q ∧
  (∀ x y : ℕ, is_two_digit x ∧ is_two_digit y ∧ y = reverse_digits x → x - y ≤ 27) →
  ∃ k : ℕ, q - r = 9 * k ∨ r - q = 9 * k :=
sorry

end NUMINAMATH_CALUDE_difference_multiple_of_nine_l260_26079


namespace NUMINAMATH_CALUDE_smaller_angle_at_5_oclock_l260_26044

/-- The number of hour marks on a clock. -/
def num_hours : ℕ := 12

/-- The number of degrees in a full circle. -/
def full_circle : ℕ := 360

/-- The time in hours. -/
def time : ℕ := 5

/-- The angle between adjacent hour marks on a clock. -/
def angle_per_hour : ℕ := full_circle / num_hours

/-- The angle between the hour hand and 12 o'clock position at the given time. -/
def hour_hand_angle : ℕ := time * angle_per_hour

/-- The smaller angle between the hour hand and minute hand at 5 o'clock. -/
theorem smaller_angle_at_5_oclock : hour_hand_angle = 150 := by
  sorry

end NUMINAMATH_CALUDE_smaller_angle_at_5_oclock_l260_26044


namespace NUMINAMATH_CALUDE_sequence_property_l260_26020

def sequence_sum (a : ℕ+ → ℚ) (n : ℕ+) : ℚ :=
  (Finset.range n.val).sum (λ i => a ⟨i + 1, Nat.succ_pos i⟩)

theorem sequence_property (a : ℕ+ → ℚ) :
  (∀ n : ℕ+, sequence_sum a n + a n = 4 - 1 / (2 ^ (n.val - 2))) →
  (∀ n : ℕ+, a n = n.val / (2 ^ (n.val - 1))) :=
sorry

end NUMINAMATH_CALUDE_sequence_property_l260_26020


namespace NUMINAMATH_CALUDE_exists_polygon_9_exists_polygon_8_l260_26062

/-- A polygon is represented as a list of points in the plane -/
def Polygon := List (ℝ × ℝ)

/-- Check if three points are collinear -/
def collinear (p q r : ℝ × ℝ) : Prop :=
  (r.2 - p.2) * (q.1 - p.1) = (q.2 - p.2) * (r.1 - p.1)

/-- Property: each side of the polygon lies on a line containing at least one additional vertex -/
def satisfiesProperty (poly : Polygon) : Prop :=
  ∀ i j : Fin poly.length,
    i ≠ j →
    ∃ k : Fin poly.length, k ≠ i ∧ k ≠ j ∧
      collinear (poly.get i) (poly.get j) (poly.get k)

/-- Theorem: There exists a polygon with at most 9 vertices satisfying the property -/
theorem exists_polygon_9 : ∃ poly : Polygon, poly.length ≤ 9 ∧ satisfiesProperty poly :=
  sorry

/-- Theorem: There exists a polygon with at most 8 vertices satisfying the property -/
theorem exists_polygon_8 : ∃ poly : Polygon, poly.length ≤ 8 ∧ satisfiesProperty poly :=
  sorry

end NUMINAMATH_CALUDE_exists_polygon_9_exists_polygon_8_l260_26062


namespace NUMINAMATH_CALUDE_g_difference_l260_26086

/-- Given g(x) = 3x^2 + 4x + 5, prove that g(x + h) - g(x) = h(6x + 3h + 4) for all real x and h. -/
theorem g_difference (x h : ℝ) : 
  let g : ℝ → ℝ := λ t ↦ 3 * t^2 + 4 * t + 5
  g (x + h) - g x = h * (6 * x + 3 * h + 4) := by
  sorry

end NUMINAMATH_CALUDE_g_difference_l260_26086


namespace NUMINAMATH_CALUDE_range_of_a_l260_26085

-- Define the sets A and B
def A : Set ℝ := {x | x ≤ -1 ∨ x > 2}
def B (a : ℝ) : Set ℝ := {x | x < a ∨ x > a + 1}

-- State the theorem
theorem range_of_a (a : ℝ) : A ∪ B a = B a → -1 < a ∧ a ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l260_26085


namespace NUMINAMATH_CALUDE_subway_ways_l260_26099

theorem subway_ways (total : ℕ) (bus : ℕ) (subway : ℕ) : 
  total = 7 → bus = 4 → total = bus + subway → subway = 3 := by
  sorry

end NUMINAMATH_CALUDE_subway_ways_l260_26099


namespace NUMINAMATH_CALUDE_lemon_price_increase_l260_26049

/-- Proves that the increase in lemon price is $4 given the conditions of Erick's fruit sale --/
theorem lemon_price_increase :
  ∀ (x : ℝ),
    (80 * (8 + x) + 140 * (7 + x / 2) = 2220) →
    x = 4 := by
  sorry

end NUMINAMATH_CALUDE_lemon_price_increase_l260_26049


namespace NUMINAMATH_CALUDE_function_bounds_l260_26046

-- Define the functions F and G
def F (a b c x : ℝ) : ℝ := a * x^2 + b * x + c
def G (a b c x : ℝ) : ℝ := c * x^2 + b * x + a

-- State the theorem
theorem function_bounds (a b c : ℝ) 
  (h1 : |F a b c 0| ≤ 1)
  (h2 : |F a b c 1| ≤ 1)
  (h3 : |F a b c (-1)| ≤ 1) :
  (∀ x : ℝ, |x| ≤ 1 → |F a b c x| ≤ 5/4) ∧
  (∀ x : ℝ, |x| ≤ 1 → |G a b c x| ≤ 2) :=
by sorry

end NUMINAMATH_CALUDE_function_bounds_l260_26046


namespace NUMINAMATH_CALUDE_part_one_part_two_l260_26056

-- Define the function f
def f (x : ℝ) : ℝ := |2*x - 1|

-- Part 1
theorem part_one (m : ℝ) (h1 : m > 0) 
  (h2 : ∀ x, x ∈ Set.Icc (-2 : ℝ) 2 ↔ f (x + 1/2) ≤ 2*m + 1) : 
  m = 3/2 := by sorry

-- Part 2
theorem part_two : 
  (∃ a : ℝ, ∀ x y : ℝ, f x ≤ 2^y + a/(2^y) + |2*x + 3|) ∧ 
  (∀ a : ℝ, (∀ x y : ℝ, f x ≤ 2^y + a/(2^y) + |2*x + 3|) → a ≥ 4) := by sorry

end NUMINAMATH_CALUDE_part_one_part_two_l260_26056


namespace NUMINAMATH_CALUDE_bus_trip_distance_l260_26014

/-- Given a bus trip with specific conditions, prove that the trip distance is 550 miles. -/
theorem bus_trip_distance (speed : ℝ) (distance : ℝ) : 
  speed = 50 →  -- The actual average speed is 50 mph
  distance / speed = distance / (speed + 5) + 1 →  -- The trip would take 1 hour less if speed increased by 5 mph
  distance = 550 := by
sorry

end NUMINAMATH_CALUDE_bus_trip_distance_l260_26014


namespace NUMINAMATH_CALUDE_original_model_cost_l260_26008

/-- The original cost of a model before the price increase -/
def original_cost : ℝ := sorry

/-- The amount Kirsty saved -/
def saved_amount : ℝ := 30 * original_cost

/-- The new cost of a model after the price increase -/
def new_cost : ℝ := original_cost + 0.50

theorem original_model_cost :
  (saved_amount = 27 * new_cost) → original_cost = 0.45 := by
  sorry

end NUMINAMATH_CALUDE_original_model_cost_l260_26008


namespace NUMINAMATH_CALUDE_scott_smoothie_sales_l260_26054

/-- Proves that Scott sold 40 cups of smoothies given the conditions of the problem -/
theorem scott_smoothie_sales :
  let smoothie_price : ℕ := 3
  let cake_price : ℕ := 2
  let cakes_sold : ℕ := 18
  let total_money : ℕ := 156
  let smoothies_sold : ℕ := (total_money - cake_price * cakes_sold) / smoothie_price
  smoothies_sold = 40 := by sorry

end NUMINAMATH_CALUDE_scott_smoothie_sales_l260_26054


namespace NUMINAMATH_CALUDE_square_difference_equality_l260_26073

theorem square_difference_equality : 1010^2 - 994^2 - 1008^2 + 996^2 = 8016 := by sorry

end NUMINAMATH_CALUDE_square_difference_equality_l260_26073


namespace NUMINAMATH_CALUDE_complement_A_inter_B_wrt_U_l260_26069

def U : Set ℤ := {x | -1 ≤ x ∧ x ≤ 2}
def A : Set ℤ := {x | x^2 - x = 0}
def B : Set ℤ := {x | -1 < x ∧ x < 2}

theorem complement_A_inter_B_wrt_U : (U \ (A ∩ B)) = {-1, 2} := by sorry

end NUMINAMATH_CALUDE_complement_A_inter_B_wrt_U_l260_26069


namespace NUMINAMATH_CALUDE_calculate_expression_l260_26057

theorem calculate_expression : (1000 * 0.09999) / 10 * 999 = 998001 := by
  sorry

end NUMINAMATH_CALUDE_calculate_expression_l260_26057


namespace NUMINAMATH_CALUDE_inequality_solution_set_l260_26023

theorem inequality_solution_set (x : ℝ) : (1 - x > x - 1) ↔ (x < 1) := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l260_26023


namespace NUMINAMATH_CALUDE_train_speed_calculation_l260_26022

/-- Calculates the speed of a train crossing a bridge -/
theorem train_speed_calculation (train_length bridge_length : ℝ) (crossing_time : ℝ) :
  train_length = 100 →
  bridge_length = 140 →
  crossing_time = 23.998080153587715 →
  ∃ (speed : ℝ), abs (speed - 36) < 0.1 ∧ speed = (train_length + bridge_length) / crossing_time * 3.6 :=
by sorry

end NUMINAMATH_CALUDE_train_speed_calculation_l260_26022


namespace NUMINAMATH_CALUDE_container_volume_ratio_l260_26093

theorem container_volume_ratio : 
  ∀ (v1 v2 : ℝ), v1 > 0 → v2 > 0 → 
  (3/4 : ℝ) * v1 = (5/8 : ℝ) * v2 → 
  v1 / v2 = (5/6 : ℝ) :=
by
  sorry

end NUMINAMATH_CALUDE_container_volume_ratio_l260_26093


namespace NUMINAMATH_CALUDE_mass_calculation_l260_26025

/-- Calculate the mass of a substance given its concentration and volume --/
theorem mass_calculation (C V : ℝ) (hC : C = 4) (hV : V = 8) : C * V = 32 := by
  sorry

end NUMINAMATH_CALUDE_mass_calculation_l260_26025


namespace NUMINAMATH_CALUDE_functional_equation_solution_l260_26012

theorem functional_equation_solution (f : ℝ → ℝ) 
  (h : ∀ x y : ℝ, (f x + x * y) * f (x - 3 * y) + (f y + x * y) * f (3 * x - y) = (f (x + y))^2) : 
  (∀ x : ℝ, f x = x^2) ∨ (∀ x : ℝ, f x = 0) := by
  sorry

end NUMINAMATH_CALUDE_functional_equation_solution_l260_26012


namespace NUMINAMATH_CALUDE_leaves_per_sub_branch_l260_26004

/-- Given a farm with trees, branches, and sub-branches, calculate the number of leaves per sub-branch. -/
theorem leaves_per_sub_branch 
  (num_trees : ℕ) 
  (branches_per_tree : ℕ) 
  (sub_branches_per_branch : ℕ) 
  (total_leaves : ℕ) 
  (h1 : num_trees = 4)
  (h2 : branches_per_tree = 10)
  (h3 : sub_branches_per_branch = 40)
  (h4 : total_leaves = 96000) :
  total_leaves / (num_trees * branches_per_tree * sub_branches_per_branch) = 60 := by
  sorry

#check leaves_per_sub_branch

end NUMINAMATH_CALUDE_leaves_per_sub_branch_l260_26004


namespace NUMINAMATH_CALUDE_bruno_pen_units_l260_26011

/-- Given that Bruno buys 2.5 units of pens and ends up with 30 pens in total,
    prove that the unit he is using is 12 pens per unit. -/
theorem bruno_pen_units (units : ℝ) (total_pens : ℕ) :
  units = 2.5 ∧ total_pens = 30 → (total_pens : ℝ) / units = 12 := by
  sorry

end NUMINAMATH_CALUDE_bruno_pen_units_l260_26011


namespace NUMINAMATH_CALUDE_min_distance_point_to_curve_l260_26018

theorem min_distance_point_to_curve (α : Real) (h : α ∈ Set.Icc 0 Real.pi) :
  let P : Prod Real Real := (1 + Real.cos α, Real.sin α)
  let C : Set (Prod Real Real) := {Q : Prod Real Real | Q.1 + Q.2 = 9}
  (∃ (d : Real), d = 4 * Real.sqrt 2 - 1 ∧
    ∀ Q ∈ C, Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2) ≥ d) :=
by sorry

end NUMINAMATH_CALUDE_min_distance_point_to_curve_l260_26018


namespace NUMINAMATH_CALUDE_min_value_reciprocal_sum_l260_26071

theorem min_value_reciprocal_sum (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 1) :
  (1 / a + 1 / (b + 1) ≥ 2) ∧
  (1 / a + 1 / (b + 1) = 2 ↔ a = 1 / 2 ∧ b = 1 / 2) := by
sorry

end NUMINAMATH_CALUDE_min_value_reciprocal_sum_l260_26071


namespace NUMINAMATH_CALUDE_xenia_earnings_and_wage_l260_26017

/-- Xenia's work schedule and earnings --/
structure WorkSchedule where
  week1_hours : ℝ
  week2_hours : ℝ
  week3_hours : ℝ
  week2_extra_earnings : ℝ
  week3_bonus : ℝ

/-- Calculate Xenia's total earnings and hourly wage --/
def calculate_earnings_and_wage (schedule : WorkSchedule) : ℝ × ℝ := by
  sorry

/-- Theorem stating Xenia's total earnings and hourly wage --/
theorem xenia_earnings_and_wage (schedule : WorkSchedule)
  (h1 : schedule.week1_hours = 18)
  (h2 : schedule.week2_hours = 25)
  (h3 : schedule.week3_hours = 28)
  (h4 : schedule.week2_extra_earnings = 60)
  (h5 : schedule.week3_bonus = 30) :
  let (total_earnings, hourly_wage) := calculate_earnings_and_wage schedule
  total_earnings = 639.47 ∧ hourly_wage = 8.57 := by
  sorry

end NUMINAMATH_CALUDE_xenia_earnings_and_wage_l260_26017


namespace NUMINAMATH_CALUDE_m_range_l260_26076

-- Define propositions p and q
def p (x m : ℝ) : Prop := x^2 - 4*m*x + 3*m^2 < 0
def q (x : ℝ) : Prop := |x - 3| ≤ 1

-- Define the condition that q is sufficient but not necessary for p
def q_sufficient_not_necessary (m : ℝ) : Prop :=
  (∀ x, q x → p x m) ∧ (∃ x, p x m ∧ ¬q x)

-- Main theorem
theorem m_range (m : ℝ) (h1 : m > 0) (h2 : q_sufficient_not_necessary m) :
  m > 4/3 ∧ m < 2 :=
sorry

end NUMINAMATH_CALUDE_m_range_l260_26076


namespace NUMINAMATH_CALUDE_wire_cutting_l260_26088

theorem wire_cutting (total_length : ℝ) (ratio : ℝ) (shorter_piece : ℝ) : 
  total_length = 70 →
  ratio = 2 / 3 →
  shorter_piece + ratio * shorter_piece = total_length →
  shorter_piece = 42 := by
sorry

end NUMINAMATH_CALUDE_wire_cutting_l260_26088


namespace NUMINAMATH_CALUDE_circles_externally_tangent_l260_26087

/-- Two circles are externally tangent if the distance between their centers
    equals the sum of their radii -/
def externally_tangent (c1 c2 : ℝ × ℝ) (r1 r2 : ℝ) : Prop :=
  Real.sqrt ((c1.1 - c2.1)^2 + (c1.2 - c2.2)^2) = r1 + r2

theorem circles_externally_tangent :
  let c1 : ℝ × ℝ := (0, 0)
  let c2 : ℝ × ℝ := (3, 4)
  let r1 : ℝ := 2
  let r2 : ℝ := 3
  externally_tangent c1 c2 r1 r2 := by
  sorry

#check circles_externally_tangent

end NUMINAMATH_CALUDE_circles_externally_tangent_l260_26087


namespace NUMINAMATH_CALUDE_jackpot_probability_correct_l260_26033

/-- The total number of numbers in the lottery -/
def total_numbers : ℕ := 45

/-- The number of numbers to be chosen in each ticket -/
def numbers_per_ticket : ℕ := 6

/-- The number of tickets bought by the player -/
def tickets_bought : ℕ := 100

/-- The probability of hitting the jackpot with the given number of tickets -/
def jackpot_probability : ℚ :=
  tickets_bought / Nat.choose total_numbers numbers_per_ticket

theorem jackpot_probability_correct :
  jackpot_probability = tickets_bought / Nat.choose total_numbers numbers_per_ticket :=
by sorry

end NUMINAMATH_CALUDE_jackpot_probability_correct_l260_26033


namespace NUMINAMATH_CALUDE_range_sum_l260_26082

noncomputable def f (x : ℝ) : ℝ := 1 + (2^(x+1))/(2^x + 1) + Real.sin x

theorem range_sum (k : ℝ) (h : k > 0) :
  ∃ (m n : ℝ), (∀ x ∈ Set.Icc (-k) k, m ≤ f x ∧ f x ≤ n) ∧
                (∀ y, y ∈ Set.Icc m n ↔ ∃ x ∈ Set.Icc (-k) k, f x = y) ∧
                m + n = 4 :=
sorry

end NUMINAMATH_CALUDE_range_sum_l260_26082


namespace NUMINAMATH_CALUDE_chocolate_doughnut_students_correct_l260_26026

/-- The number of students wanting chocolate doughnuts given the conditions -/
def chocolate_doughnut_students : ℕ :=
  let total_students : ℕ := 25
  let chocolate_cost : ℕ := 2
  let glazed_cost : ℕ := 1
  let total_cost : ℕ := 35
  -- The number of students wanting chocolate doughnuts
  10

/-- Theorem stating that the number of students wanting chocolate doughnuts is correct -/
theorem chocolate_doughnut_students_correct :
  let c := chocolate_doughnut_students
  let g := 25 - c
  c + g = 25 ∧ 2 * c + g = 35 := by sorry

end NUMINAMATH_CALUDE_chocolate_doughnut_students_correct_l260_26026


namespace NUMINAMATH_CALUDE_leg_length_theorem_l260_26066

/-- An isosceles triangle with a median on one leg dividing the perimeter -/
structure IsoscelesTriangleWithMedian where
  leg : ℝ
  base : ℝ
  median_divides_perimeter : leg + leg + base = 12 + 18
  isosceles : leg > 0
  base_positive : base > 0

/-- The theorem stating the possible lengths of the leg -/
theorem leg_length_theorem (triangle : IsoscelesTriangleWithMedian) :
  triangle.leg = 8 ∨ triangle.leg = 12 := by
  sorry

#check leg_length_theorem

end NUMINAMATH_CALUDE_leg_length_theorem_l260_26066


namespace NUMINAMATH_CALUDE_complex_equation_result_l260_26030

theorem complex_equation_result (x y : ℝ) (i : ℂ) 
  (h1 : x * i + 2 = y - i) 
  (h2 : i^2 = -1) : 
  x - y = -3 := by
sorry

end NUMINAMATH_CALUDE_complex_equation_result_l260_26030


namespace NUMINAMATH_CALUDE_chocolate_division_l260_26051

theorem chocolate_division (total_chocolate : ℚ) (num_piles : ℕ) (multiply_factor : ℕ) 
  (h1 : total_chocolate = 60 / 7)
  (h2 : num_piles = 5)
  (h3 : multiply_factor = 3) :
  (total_chocolate / num_piles) * multiply_factor = 36 / 7 := by
  sorry

end NUMINAMATH_CALUDE_chocolate_division_l260_26051


namespace NUMINAMATH_CALUDE_expression_equality_l260_26009

theorem expression_equality : (1/4)⁻¹ - |Real.sqrt 3 - 2| + 2 * (-Real.sqrt 3) = 2 - Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_expression_equality_l260_26009


namespace NUMINAMATH_CALUDE_hotdogs_sold_l260_26083

theorem hotdogs_sold (initial : ℕ) (final : ℕ) (h1 : initial = 99) (h2 : final = 97) :
  initial - final = 2 := by
  sorry

end NUMINAMATH_CALUDE_hotdogs_sold_l260_26083


namespace NUMINAMATH_CALUDE_jellybean_probability_l260_26055

def total_jellybeans : ℕ := 15
def red_jellybeans : ℕ := 5
def blue_jellybeans : ℕ := 3
def white_jellybeans : ℕ := 4
def green_jellybeans : ℕ := 3
def picked_jellybeans : ℕ := 4

theorem jellybean_probability :
  (Nat.choose red_jellybeans 2 * Nat.choose green_jellybeans 1 * Nat.choose (total_jellybeans - red_jellybeans - green_jellybeans) 1) / 
  Nat.choose total_jellybeans picked_jellybeans = 2 / 13 :=
sorry

end NUMINAMATH_CALUDE_jellybean_probability_l260_26055


namespace NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l260_26001

theorem arithmetic_sequence_common_difference 
  (a : ℕ → ℝ) 
  (h_arithmetic : ∀ n : ℕ, a (n + 1) - a n = a (n + 2) - a (n + 1)) 
  (h_first_term : a 1 = 1) 
  (h_sum : a 3 + a 5 = 14) : 
  ∃ d : ℝ, (∀ n : ℕ, a (n + 1) = a n + d) ∧ d = 2 :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l260_26001


namespace NUMINAMATH_CALUDE_coprime_n_minus_two_and_n_squared_minus_n_minus_one_part2_solutions_part3_solution_l260_26013

-- Part 1
theorem coprime_n_minus_two_and_n_squared_minus_n_minus_one (n : ℕ) :
  Nat.gcd (n - 2) (n^2 - n - 1) = 1 :=
sorry

-- Part 2
def is_valid_solution_part2 (n m : ℕ) : Prop :=
  n^3 - 3*n^2 + n + 2 = 5^m

theorem part2_solutions :
  ∀ n m : ℕ, is_valid_solution_part2 n m ↔ (n = 3 ∧ m = 1) ∨ (n = 1 ∧ m = 0) :=
sorry

-- Part 3
def is_valid_solution_part3 (n m : ℕ) : Prop :=
  2*n^3 - n^2 + 2*n + 1 = 3^m

theorem part3_solution :
  ∀ n m : ℕ, is_valid_solution_part3 n m ↔ (n = 0 ∧ m = 0) :=
sorry

end NUMINAMATH_CALUDE_coprime_n_minus_two_and_n_squared_minus_n_minus_one_part2_solutions_part3_solution_l260_26013


namespace NUMINAMATH_CALUDE_triangle_properties_l260_26059

open Real

theorem triangle_properties (a b c : ℝ) (A B C : ℝ) :
  -- Triangle conditions
  0 < a ∧ 0 < b ∧ 0 < c ∧
  0 < A ∧ A < π ∧
  0 < B ∧ B < π ∧
  0 < C ∧ C < π ∧
  A + B + C = π ∧
  a / sin A = b / sin B ∧
  b / sin B = c / sin C →
  -- Part 1
  (b = a * cos C + (1/2) * c → A = π/3) ∧
  -- Part 2
  (b * cos C + c * cos B = Real.sqrt 7 ∧ b = 2 → c = 3) :=
by sorry

end NUMINAMATH_CALUDE_triangle_properties_l260_26059


namespace NUMINAMATH_CALUDE_smallest_palindrome_div_by_7_l260_26097

/-- A function that checks if a number is a four-digit palindrome -/
def is_four_digit_palindrome (n : ℕ) : Prop :=
  1000 ≤ n ∧ n ≤ 9999 ∧ (n / 1000 = n % 10) ∧ ((n / 100) % 10 = (n / 10) % 10)

/-- A function that checks if a number has an odd first digit -/
def has_odd_first_digit (n : ℕ) : Prop :=
  (n / 1000) % 2 = 1

/-- The theorem stating that 1661 is the smallest four-digit palindrome divisible by 7 with an odd first digit -/
theorem smallest_palindrome_div_by_7 :
  (∀ n : ℕ, is_four_digit_palindrome n ∧ has_odd_first_digit n ∧ n % 7 = 0 → n ≥ 1661) ∧
  is_four_digit_palindrome 1661 ∧ has_odd_first_digit 1661 ∧ 1661 % 7 = 0 := by
  sorry

end NUMINAMATH_CALUDE_smallest_palindrome_div_by_7_l260_26097


namespace NUMINAMATH_CALUDE_polynomial_division_remainder_l260_26047

theorem polynomial_division_remainder : ∃ q : Polynomial ℝ, 
  X^4 + 3*X + 1 = (X - 3)^2 * q + (81*X - 161) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_division_remainder_l260_26047


namespace NUMINAMATH_CALUDE_second_smallest_divisible_by_all_less_than_8_sum_of_digits_l260_26006

def is_divisible_by_all_less_than_8 (n : ℕ) : Prop :=
  ∀ k : ℕ, 0 < k ∧ k < 8 → k ∣ n

def second_smallest (P : ℕ → Prop) (n : ℕ) : Prop :=
  P n ∧ ∃ m : ℕ, P m ∧ m < n ∧ ∀ k : ℕ, P k → k = m ∨ n ≤ k

def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + sum_of_digits (n / 10)

theorem second_smallest_divisible_by_all_less_than_8_sum_of_digits :
  ∃ M : ℕ, second_smallest is_divisible_by_all_less_than_8 M ∧ sum_of_digits M = 12 :=
sorry

end NUMINAMATH_CALUDE_second_smallest_divisible_by_all_less_than_8_sum_of_digits_l260_26006


namespace NUMINAMATH_CALUDE_range_of_a_l260_26094

-- Define the sets A and B
def A : Set ℝ := {x : ℝ | x^2 + x - 6 < 0}
def B (a : ℝ) : Set ℝ := {x : ℝ | x > a}

-- Define the sufficient condition
def sufficient_condition (a : ℝ) : Prop := ∀ x, x ∈ A → x ∈ B a

-- Theorem statement
theorem range_of_a :
  ∀ a : ℝ, sufficient_condition a ↔ a ≤ -3 :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l260_26094


namespace NUMINAMATH_CALUDE_sufficient_but_not_necessary_condition_l260_26058

theorem sufficient_but_not_necessary_condition (a : ℝ) :
  (a ≥ 5) →
  (∀ x ∈ Set.Icc 1 2, x^2 ≤ a) ∧
  ¬(∀ a : ℝ, (∀ x ∈ Set.Icc 1 2, x^2 ≤ a) → (a ≥ 5)) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_but_not_necessary_condition_l260_26058


namespace NUMINAMATH_CALUDE_linear_function_slope_condition_l260_26037

/-- Given two points on a linear function, if x-coordinate increases while y-coordinate decreases, then the slope is less than 2 -/
theorem linear_function_slope_condition (a x₁ y₁ x₂ y₂ : ℝ) : 
  y₁ = (a - 2) * x₁ + 1 →   -- Point A lies on the graph
  y₂ = (a - 2) * x₂ + 1 →   -- Point B lies on the graph
  (x₁ > x₂ → y₁ < y₂) →     -- When x₁ > x₂, y₁ < y₂
  a < 2 := by
sorry

end NUMINAMATH_CALUDE_linear_function_slope_condition_l260_26037


namespace NUMINAMATH_CALUDE_problem_solution_l260_26028

theorem problem_solution (a b c : ℤ) 
  (h1 : a + b = c) 
  (h2 : b + 2*c = 10) 
  (h3 : c = 4) : 
  a = 2 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l260_26028


namespace NUMINAMATH_CALUDE_sum_of_ages_l260_26081

/-- Given that Jed is 10 years older than Matt and in 10 years, Jed will be 25 years old,
    prove that the sum of their present ages is 20. -/
theorem sum_of_ages (jed_age matt_age : ℕ) : 
  jed_age = matt_age + 10 →  -- Jed is 10 years older than Matt
  jed_age + 10 = 25 →        -- In 10 years, Jed will be 25 years old
  jed_age + matt_age = 20 :=   -- The sum of their present ages is 20
by sorry

end NUMINAMATH_CALUDE_sum_of_ages_l260_26081


namespace NUMINAMATH_CALUDE_P_initial_investment_l260_26002

/-- Represents the initial investment of P in rupees -/
def P_investment : ℕ := sorry

/-- Represents Q's investment in rupees -/
def Q_investment : ℕ := 9000

/-- Represents the number of months P's investment was active -/
def P_months : ℕ := 12

/-- Represents the number of months Q's investment was active -/
def Q_months : ℕ := 8

/-- Represents P's share in the profit ratio -/
def P_share : ℕ := 2

/-- Represents Q's share in the profit ratio -/
def Q_share : ℕ := 3

/-- Theorem stating that P's initial investment is 4000 rupees -/
theorem P_initial_investment :
  (P_investment * P_months) * Q_share = (Q_investment * Q_months) * P_share ∧
  P_investment = 4000 := by
  sorry

end NUMINAMATH_CALUDE_P_initial_investment_l260_26002


namespace NUMINAMATH_CALUDE_weekly_income_proof_l260_26015

/-- Proves that a weekly income of $500 satisfies the given conditions -/
theorem weekly_income_proof (income : ℝ) : 
  income - 0.2 * income - 55 = 345 → income = 500 := by
  sorry

end NUMINAMATH_CALUDE_weekly_income_proof_l260_26015


namespace NUMINAMATH_CALUDE_second_number_calculation_l260_26003

theorem second_number_calculation (a b : ℝ) (h1 : a = 1600) (h2 : 0.20 * a = 0.20 * b + 190) : b = 650 := by
  sorry

end NUMINAMATH_CALUDE_second_number_calculation_l260_26003


namespace NUMINAMATH_CALUDE_division_remainder_l260_26060

theorem division_remainder (x y : ℕ+) (h1 : x = 7 * y + 3) (h2 : 11 * y - x = 1) : 
  2 * x ≡ 2 [ZMOD 6] := by
sorry

end NUMINAMATH_CALUDE_division_remainder_l260_26060


namespace NUMINAMATH_CALUDE_savings_calculation_l260_26067

theorem savings_calculation (savings : ℝ) (furniture_fraction : ℝ) (tv_cost : ℝ) : 
  furniture_fraction = 3 / 4 →
  tv_cost = 230 →
  (1 - furniture_fraction) * savings = tv_cost →
  savings = 920 := by
sorry

end NUMINAMATH_CALUDE_savings_calculation_l260_26067


namespace NUMINAMATH_CALUDE_arithmetic_geometric_mean_inequality_l260_26052

theorem arithmetic_geometric_mean_inequality {a b : ℝ} (ha : a ≥ 0) (hb : b ≥ 0) :
  (a + b) / 2 ≥ Real.sqrt (a * b) ∧ ((a + b) / 2 = Real.sqrt (a * b) ↔ a = b) := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_mean_inequality_l260_26052


namespace NUMINAMATH_CALUDE_peter_banana_purchase_l260_26019

def problem (initial_amount : ℕ) 
            (potato_price potato_quantity : ℕ)
            (tomato_price tomato_quantity : ℕ)
            (cucumber_price cucumber_quantity : ℕ)
            (banana_price : ℕ)
            (remaining_amount : ℕ) : Prop :=
  let potato_cost := potato_price * potato_quantity
  let tomato_cost := tomato_price * tomato_quantity
  let cucumber_cost := cucumber_price * cucumber_quantity
  let total_cost := potato_cost + tomato_cost + cucumber_cost
  let banana_cost := initial_amount - remaining_amount - total_cost
  banana_cost / banana_price = 14

theorem peter_banana_purchase :
  problem 500 2 6 3 9 4 5 5 426 := by
  sorry

end NUMINAMATH_CALUDE_peter_banana_purchase_l260_26019


namespace NUMINAMATH_CALUDE_arithmetic_pattern_l260_26074

theorem arithmetic_pattern (n : ℕ) : 
  (10^n - 1) * 9 + (n + 1) = 10^(n+1) - 1 :=
sorry

end NUMINAMATH_CALUDE_arithmetic_pattern_l260_26074


namespace NUMINAMATH_CALUDE_trig_identity_l260_26000

theorem trig_identity (θ : Real) (h : Real.tan (θ - Real.pi) = 2) :
  Real.sin θ ^ 2 + Real.sin θ * Real.cos θ - 2 * Real.cos θ ^ 2 = 4 / 5 := by
  sorry

end NUMINAMATH_CALUDE_trig_identity_l260_26000


namespace NUMINAMATH_CALUDE_duke_record_breaking_l260_26075

/-- Duke's basketball record breaking proof --/
theorem duke_record_breaking (points_to_tie : ℕ) (old_record : ℕ) 
  (free_throws : ℕ) (regular_baskets : ℕ) (normal_three_pointers : ℕ) :
  points_to_tie = 17 →
  old_record = 257 →
  free_throws = 5 →
  regular_baskets = 4 →
  normal_three_pointers = 2 →
  (free_throws * 1 + regular_baskets * 2 + (normal_three_pointers + 1) * 3) - points_to_tie = 5 := by
  sorry

#check duke_record_breaking

end NUMINAMATH_CALUDE_duke_record_breaking_l260_26075


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l260_26035

-- Define set A
def A : Set ℝ := {x : ℝ | (x - 3) * (x + 1) ≤ 0}

-- Define set B
def B : Set ℝ := {x : ℝ | 2 * x > 2}

-- Theorem statement
theorem intersection_of_A_and_B : 
  A ∩ B = {x : ℝ | 1 < x ∧ x ≤ 3} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l260_26035


namespace NUMINAMATH_CALUDE_sum_of_a_and_b_l260_26048

theorem sum_of_a_and_b (a b : ℚ) 
  (eq1 : 2 * a + 5 * b = 40) 
  (eq2 : 3 * a + 4 * b = 38) : 
  a + b = 74 / 7 := by
sorry

end NUMINAMATH_CALUDE_sum_of_a_and_b_l260_26048


namespace NUMINAMATH_CALUDE_a_range_theorem_l260_26036

/-- Sequence a_n defined as n^2 - 2an for n ∈ ℕ+ -/
def a_n (a : ℝ) (n : ℕ+) : ℝ := n.val^2 - 2*a*n.val

/-- Proposition: Given a_n = n^2 - 2an for n ∈ ℕ+, and a_n > a_4 for all n ≠ 4,
    the range of values for a is (7/2, 9/2) -/
theorem a_range_theorem (a : ℝ) : 
  (∀ (n : ℕ+), n ≠ 4 → a_n a n > a_n a 4) ↔ 
  (7/2 < a ∧ a < 9/2) :=
sorry

end NUMINAMATH_CALUDE_a_range_theorem_l260_26036


namespace NUMINAMATH_CALUDE_expression_one_equals_negative_one_expression_two_equals_five_l260_26064

-- Expression 1
theorem expression_one_equals_negative_one :
  (9/4)^(1/2) - (-8.6)^0 - (8/27)^(-1/3) = -1 := by sorry

-- Expression 2
theorem expression_two_equals_five :
  Real.log 25 / Real.log 10 + Real.log 4 / Real.log 10 + 7^(Real.log 2 / Real.log 7) + 2 * (Real.log 3 / (2 * Real.log 3)) = 5 := by sorry

end NUMINAMATH_CALUDE_expression_one_equals_negative_one_expression_two_equals_five_l260_26064


namespace NUMINAMATH_CALUDE_tan_sin_ratio_thirty_degrees_l260_26072

theorem tan_sin_ratio_thirty_degrees :
  let tan_30_sq := sin_30_sq / cos_30_sq
  let sin_30_sq := (1 : ℝ) / 4
  let cos_30_sq := (3 : ℝ) / 4
  (tan_30_sq - sin_30_sq) / (tan_30_sq * sin_30_sq) = 1 :=
by
  sorry

end NUMINAMATH_CALUDE_tan_sin_ratio_thirty_degrees_l260_26072


namespace NUMINAMATH_CALUDE_total_non_hot_peppers_l260_26095

/-- Represents the days of the week -/
inductive Day
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday

/-- Represents the pepper subtypes -/
inductive PepperSubtype
  | Jalapeno
  | Habanero
  | Bell
  | Banana
  | Poblano
  | Anaheim

/-- Returns the number of peppers picked for a given subtype on a given day -/
def peppers_picked (day : Day) (subtype : PepperSubtype) : Nat :=
  match day, subtype with
  | Day.Sunday,    PepperSubtype.Jalapeno  => 3
  | Day.Sunday,    PepperSubtype.Habanero  => 4
  | Day.Sunday,    PepperSubtype.Bell      => 6
  | Day.Sunday,    PepperSubtype.Banana    => 4
  | Day.Sunday,    PepperSubtype.Poblano   => 7
  | Day.Sunday,    PepperSubtype.Anaheim   => 6
  | Day.Monday,    PepperSubtype.Jalapeno  => 6
  | Day.Monday,    PepperSubtype.Habanero  => 6
  | Day.Monday,    PepperSubtype.Bell      => 4
  | Day.Monday,    PepperSubtype.Banana    => 4
  | Day.Monday,    PepperSubtype.Poblano   => 5
  | Day.Monday,    PepperSubtype.Anaheim   => 5
  | Day.Tuesday,   PepperSubtype.Jalapeno  => 7
  | Day.Tuesday,   PepperSubtype.Habanero  => 7
  | Day.Tuesday,   PepperSubtype.Bell      => 10
  | Day.Tuesday,   PepperSubtype.Banana    => 9
  | Day.Tuesday,   PepperSubtype.Poblano   => 4
  | Day.Tuesday,   PepperSubtype.Anaheim   => 3
  | Day.Wednesday, PepperSubtype.Jalapeno  => 6
  | Day.Wednesday, PepperSubtype.Habanero  => 6
  | Day.Wednesday, PepperSubtype.Bell      => 3
  | Day.Wednesday, PepperSubtype.Banana    => 2
  | Day.Wednesday, PepperSubtype.Poblano   => 12
  | Day.Wednesday, PepperSubtype.Anaheim   => 11
  | Day.Thursday,  PepperSubtype.Jalapeno  => 3
  | Day.Thursday,  PepperSubtype.Habanero  => 2
  | Day.Thursday,  PepperSubtype.Bell      => 10
  | Day.Thursday,  PepperSubtype.Banana    => 10
  | Day.Thursday,  PepperSubtype.Poblano   => 3
  | Day.Thursday,  PepperSubtype.Anaheim   => 2
  | Day.Friday,    PepperSubtype.Jalapeno  => 9
  | Day.Friday,    PepperSubtype.Habanero  => 9
  | Day.Friday,    PepperSubtype.Bell      => 8
  | Day.Friday,    PepperSubtype.Banana    => 7
  | Day.Friday,    PepperSubtype.Poblano   => 6
  | Day.Friday,    PepperSubtype.Anaheim   => 6
  | Day.Saturday,  PepperSubtype.Jalapeno  => 6
  | Day.Saturday,  PepperSubtype.Habanero  => 6
  | Day.Saturday,  PepperSubtype.Bell      => 4
  | Day.Saturday,  PepperSubtype.Banana    => 4
  | Day.Saturday,  PepperSubtype.Poblano   => 15
  | Day.Saturday,  PepperSubtype.Anaheim   => 15

/-- Returns true if the pepper subtype is non-hot (sweet or mild) -/
def is_non_hot (subtype : PepperSubtype) : Bool :=
  match subtype with
  | PepperSubtype.Bell    => true
  | PepperSubtype.Banana  => true
  | PepperSubtype.Poblano => true
  | PepperSubtype.Anaheim => true
  | _                     => false

/-- Theorem: The total number of non-hot peppers picked throughout the week is 185 -/
theorem total_non_hot_peppers :
  (List.sum (List.map
    (fun day =>
      List.sum (List.map
        (fun subtype =>
          if is_non_hot subtype then peppers_picked day subtype else 0)
        [PepperSubtype.Jalapeno, PepperSubtype.Habanero, PepperSubtype.Bell,
         PepperSubtype.Banana, PepperSubtype.Poblano, PepperSubtype.Anaheim]))
    [Day.Sunday, Day.Monday, Day.Tuesday, Day.Wednesday, Day.Thursday, Day.Friday, Day.Saturday]))
  = 185 := by
  sorry

end NUMINAMATH_CALUDE_total_non_hot_peppers_l260_26095


namespace NUMINAMATH_CALUDE_roots_of_unity_count_l260_26029

theorem roots_of_unity_count (a b c : ℤ) : 
  ∃ (roots : Finset ℂ), 
    (∀ z ∈ roots, z^3 = 1 ∧ z^3 + a*z^2 + b*z + c = 0) ∧ 
    Finset.card roots = 3 :=
sorry

end NUMINAMATH_CALUDE_roots_of_unity_count_l260_26029


namespace NUMINAMATH_CALUDE_basketball_teams_l260_26063

theorem basketball_teams (n : ℕ) (h : n * (n - 1) / 2 = 28) : n = 8 := by
  sorry

end NUMINAMATH_CALUDE_basketball_teams_l260_26063


namespace NUMINAMATH_CALUDE_cubic_sum_theorem_l260_26021

theorem cubic_sum_theorem (a b c : ℝ) (ha : a ≠ b) (hb : b ≠ c) (hc : c ≠ a)
  (h : (a^3 + 9)/a = (b^3 + 9)/b ∧ (b^3 + 9)/b = (c^3 + 9)/c) :
  a^3 + b^3 + c^3 = -27 := by sorry

end NUMINAMATH_CALUDE_cubic_sum_theorem_l260_26021


namespace NUMINAMATH_CALUDE_nicky_run_time_l260_26080

/-- The time Nicky runs before Cristina catches up to him in a 200-meter race --/
theorem nicky_run_time (race_distance : ℝ) (head_start : ℝ) (cristina_speed : ℝ) (nicky_speed : ℝ)
  (h1 : race_distance = 200)
  (h2 : head_start = 12)
  (h3 : cristina_speed = 5)
  (h4 : nicky_speed = 3) :
  let catch_up_time := (nicky_speed * head_start) / (cristina_speed - nicky_speed)
  head_start + catch_up_time = 30 := by
  sorry

#check nicky_run_time

end NUMINAMATH_CALUDE_nicky_run_time_l260_26080


namespace NUMINAMATH_CALUDE_brother_lower_limit_l260_26096

-- Define Arun's weight
def W : ℝ := sorry

-- Define brother's lower limit
def B : ℝ := sorry

-- Arun's opinion
axiom arun_opinion : 64 < W ∧ W < 72

-- Brother's opinion
axiom brother_opinion : B < W ∧ W < 70

-- Mother's opinion
axiom mother_opinion : W ≤ 67

-- Average weight
axiom average_weight : (W + 67) / 2 = 66

-- Theorem to prove
theorem brother_lower_limit : B > 64 := by sorry

end NUMINAMATH_CALUDE_brother_lower_limit_l260_26096


namespace NUMINAMATH_CALUDE_curve_touches_x_axis_and_area_l260_26084

noncomputable def curve (a : ℝ) (t : ℝ) : ℝ × ℝ :=
  (t + Real.exp (a * t), -t + Real.exp (a * t))

theorem curve_touches_x_axis_and_area (a : ℝ) (h : a > 0) :
  (∃ t : ℝ, (curve a t).2 = 0 ∧ 
    (∀ s : ℝ, s ≠ t → (curve a s).2 ≠ 0 ∨ (curve a s).2 < 0)) →
  a = 1 / Real.exp 1 ∧
  (∫ t in (0)..(Real.exp 1), (curve a t).2 - min (curve a t).1 (curve a t).2) = Real.exp 2 / 2 - Real.exp 1 := by
  sorry

end NUMINAMATH_CALUDE_curve_touches_x_axis_and_area_l260_26084


namespace NUMINAMATH_CALUDE_system_solution_unique_l260_26050

theorem system_solution_unique :
  ∃! (x y z : ℝ), 5 * x + 3 * y = 65 ∧ 2 * y - z = 11 ∧ 3 * x + 4 * z = 57 :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_system_solution_unique_l260_26050


namespace NUMINAMATH_CALUDE_polynomial_equality_l260_26089

theorem polynomial_equality : 7^4 + 4*(7^3) + 6*(7^2) + 4*7 + 1 = 4096 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_equality_l260_26089


namespace NUMINAMATH_CALUDE_wall_area_calculation_l260_26090

theorem wall_area_calculation (regular_area : ℝ) (jumbo_ratio : ℝ) (length_ratio : ℝ) :
  regular_area = 70 →
  jumbo_ratio = 1 / 3 →
  length_ratio = 3 →
  (regular_area + jumbo_ratio / (1 - jumbo_ratio) * regular_area * length_ratio) = 175 :=
by
  sorry

end NUMINAMATH_CALUDE_wall_area_calculation_l260_26090


namespace NUMINAMATH_CALUDE_right_triangular_prism_relation_l260_26098

/-- 
Given a right triangular prism with mutually perpendicular lateral edges of lengths a, b, and c,
and base height h, prove that 1/h^2 = 1/a^2 + 1/b^2 + 1/c^2.
-/
theorem right_triangular_prism_relation (a b c h : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) (hh : h > 0) :
  1 / h^2 = 1 / a^2 + 1 / b^2 + 1 / c^2 := by
  sorry

end NUMINAMATH_CALUDE_right_triangular_prism_relation_l260_26098


namespace NUMINAMATH_CALUDE_modular_congruence_l260_26041

theorem modular_congruence (n : ℤ) : 
  0 ≤ n ∧ n < 103 ∧ 99 * n ≡ 65 [ZMOD 103] → n ≡ 68 [ZMOD 103] := by
  sorry

end NUMINAMATH_CALUDE_modular_congruence_l260_26041


namespace NUMINAMATH_CALUDE_reflection_envelope_is_half_nephroid_l260_26045

/-- A point on the complex plane -/
def ComplexPoint := ℂ

/-- A line in the complex plane -/
def Line := ComplexPoint → Prop

/-- The unit circle centered at the origin -/
def UnitCircle : Set ℂ := {z : ℂ | Complex.abs z = 1}

/-- A bundle of parallel rays -/
def ParallelRays : Set Line := sorry

/-- The reflection of a ray off the unit circle -/
def ReflectedRay (ray : Line) : Line := sorry

/-- The envelope of a family of lines -/
def Envelope (family : Set Line) : Set ComplexPoint := sorry

/-- Half of a nephroid -/
def HalfNephroid : Set ComplexPoint := sorry

/-- The theorem statement -/
theorem reflection_envelope_is_half_nephroid :
  Envelope (ReflectedRay '' ParallelRays) = HalfNephroid := by sorry

end NUMINAMATH_CALUDE_reflection_envelope_is_half_nephroid_l260_26045


namespace NUMINAMATH_CALUDE_father_age_twice_marika_l260_26077

/-- Marika's birth year -/
def marika_birth_year : ℕ := 1996

/-- Marika's father's birth year -/
def father_birth_year : ℕ := 1956

/-- The year when the father's age is twice Marika's age -/
def target_year : ℕ := 2036

theorem father_age_twice_marika (year : ℕ) :
  year = target_year ↔ 
  (year - father_birth_year = 2 * (year - marika_birth_year)) ∧
  (year > marika_birth_year) ∧
  (year > father_birth_year) := by
sorry

end NUMINAMATH_CALUDE_father_age_twice_marika_l260_26077


namespace NUMINAMATH_CALUDE_gasohol_calculation_l260_26091

/-- The amount of gasohol initially in the tank -/
def initial_gasohol : ℝ := 54

/-- The fraction of ethanol in the initial mixture -/
def initial_ethanol_fraction : ℝ := 0.05

/-- The fraction of ethanol in the desired mixture -/
def desired_ethanol_fraction : ℝ := 0.10

/-- The amount of pure ethanol added to achieve the desired mixture -/
def added_ethanol : ℝ := 3

theorem gasohol_calculation :
  initial_gasohol * initial_ethanol_fraction + added_ethanol =
  desired_ethanol_fraction * (initial_gasohol + added_ethanol) :=
by sorry

end NUMINAMATH_CALUDE_gasohol_calculation_l260_26091


namespace NUMINAMATH_CALUDE_smallest_five_digit_congruent_to_11_mod_14_l260_26024

theorem smallest_five_digit_congruent_to_11_mod_14 :
  ∀ n : ℕ, 
    10000 ≤ n ∧ n < 100000 ∧ n ≡ 11 [ZMOD 14] →
    10007 ≤ n :=
by sorry

end NUMINAMATH_CALUDE_smallest_five_digit_congruent_to_11_mod_14_l260_26024


namespace NUMINAMATH_CALUDE_three_correct_probability_l260_26038

/-- The probability of exactly 3 out of 5 packages being delivered to their correct houses -/
def probability_three_correct (n : ℕ) : ℚ :=
  if n = 5 then 1 / 12 else 0

/-- Theorem stating the probability of exactly 3 out of 5 packages being delivered correctly -/
theorem three_correct_probability :
  probability_three_correct 5 = 1 / 12 :=
by sorry

end NUMINAMATH_CALUDE_three_correct_probability_l260_26038


namespace NUMINAMATH_CALUDE_layla_nahima_score_difference_l260_26039

theorem layla_nahima_score_difference :
  ∀ (total_points layla_score nahima_score : ℕ),
    total_points = 112 →
    layla_score = 70 →
    total_points = layla_score + nahima_score →
    layla_score - nahima_score = 28 :=
by
  sorry

end NUMINAMATH_CALUDE_layla_nahima_score_difference_l260_26039


namespace NUMINAMATH_CALUDE_functional_equation_solution_l260_26031

/-- A function satisfying the given functional equation -/
def SatisfiesFunctionalEquation (f : ℝ → ℝ) : Prop :=
  f 0 = 0 ∧
  ∀ x y : ℝ, (x - y) * (f (f (x^2)) - f (f (y^2))) = (f x + f y) * (f x - f y)^2

/-- The main theorem stating that any function satisfying the functional equation
    is of the form f(x) = cx for some constant c -/
theorem functional_equation_solution :
  ∀ f : ℝ → ℝ, SatisfiesFunctionalEquation f →
  ∃ c : ℝ, ∀ x : ℝ, f x = c * x :=
sorry

end NUMINAMATH_CALUDE_functional_equation_solution_l260_26031


namespace NUMINAMATH_CALUDE_min_ratio_bounds_l260_26068

/-- An equiangular hexagon with alternating side lengths 1 and a -/
structure EquiangularHexagon :=
  (a : ℝ)

/-- A circle intersecting the hexagon at 12 distinct points -/
structure IntersectingCircle (h : EquiangularHexagon) :=
  (exists_intersection : True)

/-- The bounds M and N for the side length a -/
structure Bounds (h : EquiangularHexagon) (c : IntersectingCircle h) :=
  (M N : ℝ)
  (lower_bound : M < h.a)
  (upper_bound : h.a < N)

/-- The theorem stating the minimum possible value of N/M -/
theorem min_ratio_bounds 
  (h : EquiangularHexagon) 
  (c : IntersectingCircle h) 
  (b : Bounds h c) : 
  ∃ (M N : ℝ), M < h.a ∧ h.a < N ∧ 
  ∀ (M' N' : ℝ), M' < h.a → h.a < N' → (3 * Real.sqrt 3 + 3) / 2 ≤ N' / M' :=
sorry

end NUMINAMATH_CALUDE_min_ratio_bounds_l260_26068


namespace NUMINAMATH_CALUDE_no_cyclic_knight_tour_5x5_l260_26005

/-- Represents a chessboard --/
structure Chessboard :=
  (rows : Nat)
  (cols : Nat)

/-- Represents a knight's move --/
inductive KnightMove
  | move : Nat → Nat → KnightMove

/-- Represents a tour on the chessboard --/
structure Tour :=
  (moves : List KnightMove)
  (cyclic : Bool)

/-- Defines a valid knight's move --/
def isValidKnightMove (m : KnightMove) : Prop :=
  match m with
  | KnightMove.move x y => (x = 2 ∧ y = 1) ∨ (x = 1 ∧ y = 2)

/-- Defines if a tour visits each square exactly once --/
def visitsEachSquareOnce (t : Tour) (b : Chessboard) : Prop :=
  t.moves.length = b.rows * b.cols

/-- Theorem: It's impossible for a knight to make a cyclic tour on a 5x5 chessboard
    visiting each square exactly once --/
theorem no_cyclic_knight_tour_5x5 :
  ∀ (t : Tour),
    t.cyclic →
    (∀ (m : KnightMove), m ∈ t.moves → isValidKnightMove m) →
    visitsEachSquareOnce t (Chessboard.mk 5 5) →
    False :=
sorry

end NUMINAMATH_CALUDE_no_cyclic_knight_tour_5x5_l260_26005


namespace NUMINAMATH_CALUDE_jims_journey_distance_l260_26027

/-- The total distance of Jim's journey -/
def total_distance (driven : ℕ) (remaining : ℕ) : ℕ :=
  driven + remaining

/-- Theorem stating the total distance of Jim's journey -/
theorem jims_journey_distance :
  total_distance 215 985 = 1200 := by
  sorry

end NUMINAMATH_CALUDE_jims_journey_distance_l260_26027


namespace NUMINAMATH_CALUDE_triangle_existence_condition_l260_26078

-- Define the triangle ABC
structure Triangle :=
  (A B C : ℝ × ℝ)

-- Define the conditions
def angle_ABC (t : Triangle) : ℝ := sorry
def side_AC (t : Triangle) : ℝ := sorry
def side_BC (t : Triangle) : ℝ := sorry

-- Theorem statement
theorem triangle_existence_condition (t : Triangle) (k : ℝ) :
  (∃! t, angle_ABC t = π/3 ∧ side_AC t = 12 ∧ side_BC t = k) ↔
  (0 < k ∧ k ≤ 12) ∨ k = 8 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_triangle_existence_condition_l260_26078


namespace NUMINAMATH_CALUDE_smallest_number_with_remainder_two_l260_26032

theorem smallest_number_with_remainder_two : ∃ n : ℕ, 
  (n > 1) ∧ 
  (n % 3 = 2) ∧ 
  (n % 4 = 2) ∧ 
  (n % 5 = 2) ∧ 
  (∀ m : ℕ, m > 1 → (m % 3 = 2) → (m % 4 = 2) → (m % 5 = 2) → m ≥ n) ∧
  (n = 62) := by
  sorry

end NUMINAMATH_CALUDE_smallest_number_with_remainder_two_l260_26032


namespace NUMINAMATH_CALUDE_manuscript_typing_cost_l260_26016

/-- Calculates the total cost of typing a manuscript with given parameters. -/
def manuscriptCost (totalPages : ℕ) (initialCost revisonCost : ℚ) 
  (revisedOnce revisedTwice : ℕ) : ℚ :=
  let initialTypingCost := totalPages * initialCost
  let firstRevisionCost := revisedOnce * revisonCost
  let secondRevisionCost := revisedTwice * (2 * revisonCost)
  initialTypingCost + firstRevisionCost + secondRevisionCost

/-- Theorem stating that the total cost of typing the manuscript is $1360. -/
theorem manuscript_typing_cost : 
  manuscriptCost 200 5 3 80 20 = 1360 := by
  sorry

end NUMINAMATH_CALUDE_manuscript_typing_cost_l260_26016


namespace NUMINAMATH_CALUDE_cos_330_degrees_l260_26007

theorem cos_330_degrees : Real.cos (330 * π / 180) = Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_330_degrees_l260_26007


namespace NUMINAMATH_CALUDE_roots_sum_of_squares_l260_26040

theorem roots_sum_of_squares (a b : ℝ) : 
  (∀ x, x^2 - 4*x + 4 = 0 ↔ x = a ∨ x = b) → a^2 + b^2 = 8 := by
sorry

end NUMINAMATH_CALUDE_roots_sum_of_squares_l260_26040


namespace NUMINAMATH_CALUDE_water_purifier_filtration_layers_l260_26010

theorem water_purifier_filtration_layers (initial_impurities : ℝ) (target_impurities : ℝ) 
  (filter_efficiency : ℝ) (h1 : initial_impurities = 80) (h2 : target_impurities = 2) 
  (h3 : filter_efficiency = 1/3) : 
  ∃ n : ℕ, (initial_impurities * (1 - filter_efficiency)^n ≤ target_impurities ∧ 
  ∀ m : ℕ, m < n → initial_impurities * (1 - filter_efficiency)^m > target_impurities) :=
sorry

end NUMINAMATH_CALUDE_water_purifier_filtration_layers_l260_26010
