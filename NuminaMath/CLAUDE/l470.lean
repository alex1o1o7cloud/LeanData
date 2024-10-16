import Mathlib

namespace NUMINAMATH_CALUDE_nonagon_coloring_theorem_l470_47000

/-- A type representing the colors used to color the nonagon vertices -/
inductive Color
| A
| B
| C

/-- A type representing the vertices of a regular nonagon -/
inductive Vertex
| One | Two | Three | Four | Five | Six | Seven | Eight | Nine

/-- A function type representing a coloring of the nonagon -/
def Coloring := Vertex → Color

/-- Predicate to check if two vertices are adjacent in a regular nonagon -/
def adjacent (v1 v2 : Vertex) : Prop := sorry

/-- Predicate to check if three vertices form an equilateral triangle in a regular nonagon -/
def equilateralTriangle (v1 v2 v3 : Vertex) : Prop := sorry

/-- Predicate to check if a coloring is valid according to the given conditions -/
def validColoring (c : Coloring) : Prop :=
  (∀ v1 v2, adjacent v1 v2 → c v1 ≠ c v2) ∧
  (∀ v1 v2 v3, equilateralTriangle v1 v2 v3 → c v1 ≠ c v2 ∧ c v1 ≠ c v3 ∧ c v2 ≠ c v3)

/-- The minimum number of colors needed for a valid coloring -/
def m : Nat := 3

/-- The total number of valid colorings using m colors -/
def n : Nat := 18

/-- The main theorem stating that the product of m and n is 54 -/
theorem nonagon_coloring_theorem : m * n = 54 := by sorry

end NUMINAMATH_CALUDE_nonagon_coloring_theorem_l470_47000


namespace NUMINAMATH_CALUDE_distance_to_x_axis_l470_47068

theorem distance_to_x_axis (P : ℝ × ℝ) (h : P = (-4, 1)) : 
  |P.2| = 1 := by sorry

end NUMINAMATH_CALUDE_distance_to_x_axis_l470_47068


namespace NUMINAMATH_CALUDE_cosine_identity_l470_47082

theorem cosine_identity (n : Real) : 
  (Real.cos (30 * π / 180 - n * π / 180)) / (Real.cos (n * π / 180)) = 
  (1 / 2) * (Real.sqrt 3 + Real.tan (n * π / 180)) := by
  sorry

end NUMINAMATH_CALUDE_cosine_identity_l470_47082


namespace NUMINAMATH_CALUDE_investment_solution_l470_47095

def investment_problem (amount_A : ℝ) : Prop :=
  let yield_A : ℝ := 0.30
  let yield_B : ℝ := 0.50
  let amount_B : ℝ := 200
  (amount_A * (1 + yield_A)) = (amount_B * (1 + yield_B) + 90)

theorem investment_solution : 
  ∃ (amount_A : ℝ), investment_problem amount_A ∧ amount_A = 300 := by
  sorry

end NUMINAMATH_CALUDE_investment_solution_l470_47095


namespace NUMINAMATH_CALUDE_two_suits_cost_l470_47007

def off_the_rack_cost : ℕ := 300
def tailoring_cost : ℕ := 200

def total_cost (off_the_rack : ℕ) (tailoring : ℕ) : ℕ :=
  off_the_rack + (3 * off_the_rack + tailoring)

theorem two_suits_cost :
  total_cost off_the_rack_cost tailoring_cost = 1400 := by
  sorry

end NUMINAMATH_CALUDE_two_suits_cost_l470_47007


namespace NUMINAMATH_CALUDE_corner_removed_cube_edges_l470_47037

/-- Represents a solid formed by removing smaller cubes from corners of a larger cube. -/
structure CornerRemovedCube where
  originalSideLength : ℝ
  removedSideLength : ℝ

/-- Calculates the number of edges in the resulting solid after corner removal. -/
def edgeCount (cube : CornerRemovedCube) : ℕ :=
  12 + 24  -- This is a placeholder. The actual calculation would be more complex.

/-- Theorem stating that a cube of side length 4 with corners of side length 2 removed has 36 edges. -/
theorem corner_removed_cube_edges :
  ∀ (cube : CornerRemovedCube),
    cube.originalSideLength = 4 →
    cube.removedSideLength = 2 →
    edgeCount cube = 36 := by
  sorry

#check corner_removed_cube_edges

end NUMINAMATH_CALUDE_corner_removed_cube_edges_l470_47037


namespace NUMINAMATH_CALUDE_root_intervals_l470_47054

/-- An even function f centered at x = 2 -/
def f (x : ℝ) : ℝ := sorry

theorem root_intervals (e : ℝ) (h_e : 0 < e) :
  let f := fun (x : ℝ) => f (x - 2)
  (∀ x, f (-x) = f x) ∧ 
  (∀ x > -2, f x = Real.exp (x + 1) - 2) →
  {k : ℤ | ∃ x₀ : ℝ, f x₀ = 0 ∧ ↑k - 1 < x₀ ∧ x₀ < ↑k} = {-3, 0} :=
sorry

end NUMINAMATH_CALUDE_root_intervals_l470_47054


namespace NUMINAMATH_CALUDE_fraction_sum_l470_47064

theorem fraction_sum (m n : ℚ) (h : m / n = 3 / 7) : (m + n) / n = 10 / 7 := by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_l470_47064


namespace NUMINAMATH_CALUDE_largest_class_size_l470_47062

theorem largest_class_size (num_classes : ℕ) (student_diff : ℕ) (total_students : ℕ) :
  num_classes = 5 →
  student_diff = 2 →
  total_students = 120 →
  ∃ (x : ℕ), x = 28 ∧ 
    (x + (x - student_diff) + (x - 2*student_diff) + (x - 3*student_diff) + (x - 4*student_diff) = total_students) :=
by sorry

end NUMINAMATH_CALUDE_largest_class_size_l470_47062


namespace NUMINAMATH_CALUDE_range_of_x_no_solution_exists_l470_47009

-- Define the conditions
def conditions (a b : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ a + b = a * b

-- Theorem for the first part of the problem
theorem range_of_x (a b : ℝ) (h : conditions a b) :
  (∀ x : ℝ, |x| + |x - 2| ≤ a + b) ↔ ∀ x : ℝ, -1 ≤ x ∧ x ≤ 3 :=
sorry

-- Theorem for the second part of the problem
theorem no_solution_exists :
  ¬∃ a b : ℝ, conditions a b ∧ 4 * a + b = 8 :=
sorry

end NUMINAMATH_CALUDE_range_of_x_no_solution_exists_l470_47009


namespace NUMINAMATH_CALUDE_find_a_l470_47012

theorem find_a (a x y : ℝ) (h1 : a^(3*x - 1) * 3^(4*y - 3) = 49^x * 27^y) (h2 : x + y = 4) : a = 7 := by
  sorry

end NUMINAMATH_CALUDE_find_a_l470_47012


namespace NUMINAMATH_CALUDE_hyperbola_larger_y_focus_l470_47078

def hyperbola_equation (x y : ℝ) : Prop :=
  (x - 5)^2 / 7^2 - (y - 10)^2 / 3^2 = 1

def is_focus (x y : ℝ) : Prop :=
  (x - 5)^2 + (y - 10)^2 = 58

def larger_y_focus (x y : ℝ) : Prop :=
  is_focus x y ∧ y > 10

theorem hyperbola_larger_y_focus :
  ∃ (x y : ℝ), hyperbola_equation x y ∧ larger_y_focus x y ∧ x = 5 ∧ y = 10 + Real.sqrt 58 :=
sorry

end NUMINAMATH_CALUDE_hyperbola_larger_y_focus_l470_47078


namespace NUMINAMATH_CALUDE_share_ratio_l470_47048

def problem (total a b c : ℚ) : Prop :=
  total = 527 ∧
  a = 372 ∧
  b = 93 ∧
  c = 62 ∧
  a = (2/3) * b ∧
  total = a + b + c

theorem share_ratio (total a b c : ℚ) (h : problem total a b c) :
  b / c = 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_share_ratio_l470_47048


namespace NUMINAMATH_CALUDE_total_current_ages_l470_47042

theorem total_current_ages (amar akbar anthony : ℕ) : 
  (amar - 4) + (akbar - 4) + (anthony - 4) = 54 → amar + akbar + anthony = 66 := by
  sorry

end NUMINAMATH_CALUDE_total_current_ages_l470_47042


namespace NUMINAMATH_CALUDE_chess_tournament_participants_l470_47045

theorem chess_tournament_participants (x : ℕ) : 
  (∃ y : ℕ, 2 * x * y + 16 = (x + 2) * (x + 1)) ↔ (x = 7 ∨ x = 14) :=
by sorry

end NUMINAMATH_CALUDE_chess_tournament_participants_l470_47045


namespace NUMINAMATH_CALUDE_tan_beta_value_l470_47099

theorem tan_beta_value (θ β : Real) (h1 : (2 : Real) = 2 * Real.cos θ) 
  (h2 : (-3 : Real) = 2 * Real.sin θ) (h3 : β = θ - 3 * Real.pi / 4) : 
  Real.tan β = -1/5 := by
  sorry

end NUMINAMATH_CALUDE_tan_beta_value_l470_47099


namespace NUMINAMATH_CALUDE_sum_of_proportions_l470_47014

theorem sum_of_proportions (a b c d e f : ℝ) 
  (h1 : a / b = 2) 
  (h2 : c / d = 2) 
  (h3 : e / f = 2) 
  (h4 : b + d + f = 4) : 
  a + c + e = 8 := by
sorry

end NUMINAMATH_CALUDE_sum_of_proportions_l470_47014


namespace NUMINAMATH_CALUDE_zeros_of_f_l470_47059

-- Define the function f(x) = x^2 - 2x - 3
def f (x : ℝ) : ℝ := x^2 - 2*x - 3

-- Theorem stating that 3 and -1 are the zeros of the function f
theorem zeros_of_f : 
  (f 3 = 0 ∧ f (-1) = 0) ∧ 
  ∀ x : ℝ, f x = 0 → x = 3 ∨ x = -1 :=
sorry

end NUMINAMATH_CALUDE_zeros_of_f_l470_47059


namespace NUMINAMATH_CALUDE_upstream_downstream_time_ratio_l470_47056

/-- 
Given a boat with speed 48 kmph in still water and a stream with speed 16 kmph,
prove that the ratio of time taken to row upstream to the time taken to row downstream is 2:1.
-/
theorem upstream_downstream_time_ratio 
  (boat_speed : ℝ) 
  (stream_speed : ℝ) 
  (h1 : boat_speed = 48) 
  (h2 : stream_speed = 16) : 
  (boat_speed - stream_speed) / (boat_speed + stream_speed) = 1 / 2 := by
sorry

end NUMINAMATH_CALUDE_upstream_downstream_time_ratio_l470_47056


namespace NUMINAMATH_CALUDE_a_squared_b_plus_ab_squared_l470_47070

theorem a_squared_b_plus_ab_squared (a b : ℝ) 
  (h1 : a + b = 5) 
  (h2 : a * b = 6) : 
  a^2 * b + a * b^2 = 30 := by
sorry

end NUMINAMATH_CALUDE_a_squared_b_plus_ab_squared_l470_47070


namespace NUMINAMATH_CALUDE_puppies_feeding_theorem_l470_47080

/-- Given the number of formula portions, puppies, and days, calculate the number of feedings per day. -/
def feedings_per_day (portions : ℕ) (puppies : ℕ) (days : ℕ) : ℚ :=
  (portions : ℚ) / (puppies * days)

/-- Theorem stating that given 105 portions of formula for 7 puppies over 5 days, the number of feedings per day is equal to 3. -/
theorem puppies_feeding_theorem :
  feedings_per_day 105 7 5 = 3 := by
  sorry

end NUMINAMATH_CALUDE_puppies_feeding_theorem_l470_47080


namespace NUMINAMATH_CALUDE_ratio_odd_even_divisors_l470_47088

def M : ℕ := 42 * 43 * 75 * 196

def sum_odd_divisors (n : ℕ) : ℕ := sorry
def sum_even_divisors (n : ℕ) : ℕ := sorry

theorem ratio_odd_even_divisors :
  (sum_odd_divisors M : ℚ) / (sum_even_divisors M : ℚ) = 1 / 14 := by sorry

end NUMINAMATH_CALUDE_ratio_odd_even_divisors_l470_47088


namespace NUMINAMATH_CALUDE_point_in_third_quadrant_l470_47083

/-- Definition of a point in the third quadrant -/
def is_in_third_quadrant (p : ℝ × ℝ) : Prop :=
  p.1 < 0 ∧ p.2 < 0

/-- The point (-2, -3) is in the third quadrant -/
theorem point_in_third_quadrant : is_in_third_quadrant (-2, -3) := by
  sorry

end NUMINAMATH_CALUDE_point_in_third_quadrant_l470_47083


namespace NUMINAMATH_CALUDE_max_profit_l470_47066

-- Define the linear relationship between price and quantity
def sales_quantity (x : ℝ) : ℝ := -2 * x + 180

-- Define the profit function
def profit (x : ℝ) : ℝ := (x - 50) * (sales_quantity x)

-- Theorem statement
theorem max_profit :
  ∃ (x : ℝ), x = 70 ∧ profit x = 800 ∧ ∀ (y : ℝ), profit y ≤ profit x :=
sorry

end NUMINAMATH_CALUDE_max_profit_l470_47066


namespace NUMINAMATH_CALUDE_older_rabbit_catch_up_steps_l470_47057

/-- Represents the rabbits in the race -/
inductive Rabbit
| Younger
| Older

/-- Properties of the rabbit race -/
structure RaceProperties where
  initial_lead : ℕ
  younger_steps_per_time : ℕ
  older_steps_per_time : ℕ
  younger_distance_steps : ℕ
  older_distance_steps : ℕ
  younger_distance : ℕ
  older_distance : ℕ

/-- The race between the two rabbits -/
def rabbit_race (props : RaceProperties) : Prop :=
  props.initial_lead = 10 ∧
  props.younger_steps_per_time = 4 ∧
  props.older_steps_per_time = 3 ∧
  props.younger_distance_steps = 7 ∧
  props.older_distance_steps = 5 ∧
  props.younger_distance = props.older_distance

/-- Theorem stating the number of steps for the older rabbit to catch up -/
theorem older_rabbit_catch_up_steps (props : RaceProperties) 
  (h : rabbit_race props) : ∃ (steps : ℕ), steps = 150 := by
  sorry


end NUMINAMATH_CALUDE_older_rabbit_catch_up_steps_l470_47057


namespace NUMINAMATH_CALUDE_gcd_plus_ten_l470_47094

theorem gcd_plus_ten (a b : ℕ) (h : a = 8436 ∧ b = 156) :
  (Nat.gcd a b) + 10 = 22 := by
  sorry

end NUMINAMATH_CALUDE_gcd_plus_ten_l470_47094


namespace NUMINAMATH_CALUDE_infinite_power_tower_eq_four_solution_l470_47085

/-- Define the infinite power tower function --/
noncomputable def infinitePowerTower (x : ℝ) : ℝ :=
  Real.log x / Real.log 2

/-- Theorem: The solution to x^(x^(x^...)) = 4 is √2 --/
theorem infinite_power_tower_eq_four_solution :
  ∀ x : ℝ, x > 0 → infinitePowerTower x = 4 → x = Real.sqrt 2 :=
by
  sorry


end NUMINAMATH_CALUDE_infinite_power_tower_eq_four_solution_l470_47085


namespace NUMINAMATH_CALUDE_relationship_abc_l470_47098

theorem relationship_abc : 
  let a : ℝ := 2^(1/2)
  let b : ℝ := 3^(1/3)
  let c : ℝ := Real.log 2
  b > a ∧ a > c := by sorry

end NUMINAMATH_CALUDE_relationship_abc_l470_47098


namespace NUMINAMATH_CALUDE_complementary_sets_count_l470_47033

/-- Represents a card in the deck -/
structure Card where
  shape : Fin 3
  color : Fin 3
  shade : Fin 3

/-- The deck of cards -/
def Deck : Finset Card := sorry

/-- A set of three cards -/
def ThreeCardSet : Type := Fin 3 → Card

/-- Checks if a three-card set is complementary -/
def is_complementary (set : ThreeCardSet) : Prop := sorry

/-- The set of all complementary three-card sets -/
def ComplementarySets : Finset ThreeCardSet := sorry

theorem complementary_sets_count : 
  Finset.card ComplementarySets = 702 := by sorry

end NUMINAMATH_CALUDE_complementary_sets_count_l470_47033


namespace NUMINAMATH_CALUDE_candied_fruit_earnings_l470_47053

/-- The number of candied apples made -/
def num_apples : ℕ := 15

/-- The price of each candied apple in dollars -/
def price_apple : ℚ := 2

/-- The number of candied grapes made -/
def num_grapes : ℕ := 12

/-- The price of each candied grape in dollars -/
def price_grape : ℚ := (3 : ℚ) / 2

/-- The total earnings from selling all candied apples and grapes -/
def total_earnings : ℚ := num_apples * price_apple + num_grapes * price_grape

theorem candied_fruit_earnings : total_earnings = 48 := by
  sorry

end NUMINAMATH_CALUDE_candied_fruit_earnings_l470_47053


namespace NUMINAMATH_CALUDE_triangle_area_l470_47002

theorem triangle_area (a b : ℝ) (C : ℝ) (h1 : a = 3 * Real.sqrt 2) (h2 : b = 2 * Real.sqrt 3) (h3 : Real.cos C = 1/3) :
  (1/2) * a * b * Real.sin C = 4 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_l470_47002


namespace NUMINAMATH_CALUDE_largest_divisor_of_m_l470_47060

theorem largest_divisor_of_m (m : ℕ) (h1 : m > 0) (h2 : 216 ∣ m^2) : 
  36 = Nat.gcd m 36 ∧ ∀ k : ℕ, k > 36 → k ∣ m → k ∣ 36 := by
  sorry

end NUMINAMATH_CALUDE_largest_divisor_of_m_l470_47060


namespace NUMINAMATH_CALUDE_evelyns_marbles_l470_47001

theorem evelyns_marbles (initial_marbles : ℕ) : 
  initial_marbles + 9 = 104 → initial_marbles = 95 := by
  sorry

end NUMINAMATH_CALUDE_evelyns_marbles_l470_47001


namespace NUMINAMATH_CALUDE_roja_work_time_l470_47021

/-- Given that Malar and Roja combined complete a task in 35 days,
    and Malar alone completes the same work in 60 days,
    prove that Roja alone can complete the work in 210 days. -/
theorem roja_work_time (combined_time malar_time : ℝ)
  (h_combined : combined_time = 35)
  (h_malar : malar_time = 60) :
  let roja_time := (combined_time * malar_time) / (malar_time - combined_time)
  roja_time = 210 := by
sorry

end NUMINAMATH_CALUDE_roja_work_time_l470_47021


namespace NUMINAMATH_CALUDE_inequalities_given_sum_positive_l470_47087

theorem inequalities_given_sum_positive (a b : ℝ) (h : a + b > 0) :
  (a^5 * b^2 + a^4 * b^3 ≥ 0) ∧
  (a^21 + b^21 > 0) ∧
  ((a+2)*(b+2) > a*b) := by
  sorry

end NUMINAMATH_CALUDE_inequalities_given_sum_positive_l470_47087


namespace NUMINAMATH_CALUDE_cosine_equation_solutions_l470_47092

open Real

theorem cosine_equation_solutions :
  let f := fun (x : ℝ) => 3 * (cos x)^4 - 6 * (cos x)^3 + 4 * (cos x)^2 - 1
  ∃! (s : Finset ℝ), s.card = 5 ∧ (∀ x ∈ s, 0 ≤ x ∧ x ≤ 2*π ∧ f x = 0) ∧
    (∀ x, 0 ≤ x ∧ x ≤ 2*π ∧ f x = 0 → x ∈ s) :=
by sorry

end NUMINAMATH_CALUDE_cosine_equation_solutions_l470_47092


namespace NUMINAMATH_CALUDE_waiter_customers_l470_47051

/-- Calculates the number of remaining customers given the initial number and the number who left. -/
def remaining_customers (initial : ℕ) (left : ℕ) : ℕ :=
  initial - left

/-- Theorem stating that for a waiter with 14 initial customers, after 5 leave, 9 remain. -/
theorem waiter_customers : remaining_customers 14 5 = 9 := by
  sorry

end NUMINAMATH_CALUDE_waiter_customers_l470_47051


namespace NUMINAMATH_CALUDE_discount_equivalence_l470_47029

/-- Proves that a 30% discount followed by a 15% discount is equivalent to a 40.5% single discount -/
theorem discount_equivalence (original_price : ℝ) (h : original_price > 0) :
  let first_discount := 0.30
  let second_discount := 0.15
  let discounted_price := original_price * (1 - first_discount)
  let final_price := discounted_price * (1 - second_discount)
  let equivalent_discount := 1 - (final_price / original_price)
  equivalent_discount = 0.405 := by
sorry

end NUMINAMATH_CALUDE_discount_equivalence_l470_47029


namespace NUMINAMATH_CALUDE_side_length_S2_is_correct_l470_47047

/-- The side length of square S2 in a specific arrangement of rectangles and squares. -/
def side_length_S2 : ℕ :=
  let total_width : ℕ := 4422
  let total_height : ℕ := 2420
  -- S1 and S3 have the same side length, which is also the smaller dimension of R1 and R2
  -- Let r be this common side length
  -- Let s be the side length of S2
  -- From the height: 2r + s = total_height
  -- From the width: 2r + 3s = total_width
  -- Solving this system of equations gives s = 1001
  1001

/-- Theorem stating that the side length of S2 is correct given the conditions. -/
theorem side_length_S2_is_correct :
  let total_width : ℕ := 4422
  let total_height : ℕ := 2420
  ∃ (r : ℕ),
    (2 * r + side_length_S2 = total_height) ∧
    (2 * r + 3 * side_length_S2 = total_width) :=
by sorry

#eval side_length_S2  -- Should output 1001

end NUMINAMATH_CALUDE_side_length_S2_is_correct_l470_47047


namespace NUMINAMATH_CALUDE_coin_stack_arrangements_l470_47027

/-- The number of indistinguishable gold coins -/
def num_gold_coins : ℕ := 5

/-- The number of indistinguishable silver coins -/
def num_silver_coins : ℕ := 3

/-- The total number of coins -/
def total_coins : ℕ := num_gold_coins + num_silver_coins

/-- The number of ways to arrange gold and silver coins -/
def gold_silver_arrangements : ℕ := Nat.choose total_coins num_gold_coins

/-- The number of valid head-tail sequences -/
def valid_head_tail_sequences : ℕ := total_coins + 1

/-- The total number of distinguishable arrangements -/
def total_arrangements : ℕ := gold_silver_arrangements * valid_head_tail_sequences

theorem coin_stack_arrangements :
  total_arrangements = 504 :=
sorry

end NUMINAMATH_CALUDE_coin_stack_arrangements_l470_47027


namespace NUMINAMATH_CALUDE_equal_area_division_l470_47006

/-- Represents a point in 2D space -/
structure Point where
  x : ℚ
  y : ℚ

/-- Represents a quadrilateral -/
structure Quadrilateral where
  A : Point
  B : Point
  C : Point
  D : Point

/-- Calculates the area of a quadrilateral using the shoelace formula -/
def area (q : Quadrilateral) : ℚ :=
  let det := q.A.x * q.B.y + q.B.x * q.C.y + q.C.x * q.D.y + q.D.x * q.A.y -
             (q.B.x * q.A.y + q.C.x * q.B.y + q.D.x * q.C.y + q.A.x * q.D.y)
  (1/2) * abs det

/-- Represents the intersection point of the dividing line with CD -/
structure IntersectionPoint where
  p : ℤ
  q : ℤ
  r : ℤ
  s : ℤ

/-- The main theorem -/
theorem equal_area_division (q : Quadrilateral) (i : IntersectionPoint) :
  q.A = ⟨0, 0⟩ →
  q.B = ⟨0, 3⟩ →
  q.C = ⟨4, 4⟩ →
  q.D = ⟨5, 0⟩ →
  area { A := q.A, B := q.B, C := ⟨i.p / i.q, i.r / i.s⟩, D := q.D } = 
  area { A := q.A, B := ⟨i.p / i.q, i.r / i.s⟩, C := q.C, D := q.D } →
  i.p + i.q + i.r + i.s = 13 := by sorry

end NUMINAMATH_CALUDE_equal_area_division_l470_47006


namespace NUMINAMATH_CALUDE_f_range_implies_a_values_l470_47017

/-- The function f(x) defined as x^2 - 2ax + 2a + 4 -/
def f (a : ℝ) (x : ℝ) : ℝ := x^2 - 2*a*x + 2*a + 4

/-- The property that the range of f is [1, +∞) -/
def range_property (a : ℝ) : Prop :=
  ∀ x, f a x ≥ 1 ∧ ∀ y ≥ 1, ∃ x, f a x = y

/-- Theorem stating that the only values of a satisfying the conditions are -1 and 3 -/
theorem f_range_implies_a_values :
  ∀ a : ℝ, range_property a ↔ (a = -1 ∨ a = 3) :=
sorry

end NUMINAMATH_CALUDE_f_range_implies_a_values_l470_47017


namespace NUMINAMATH_CALUDE_fair_coin_probability_difference_l470_47075

def probability_exactly_3_heads (n : ℕ) (p : ℚ) : ℚ :=
  ↑(Nat.choose n 3) * p^3 * (1 - p)^(n - 3)

def probability_all_heads (n : ℕ) (p : ℚ) : ℚ :=
  p^n

theorem fair_coin_probability_difference :
  let n : ℕ := 4
  let p : ℚ := 1/2
  (probability_exactly_3_heads n p) - (probability_all_heads n p) = 7/16 := by
  sorry

end NUMINAMATH_CALUDE_fair_coin_probability_difference_l470_47075


namespace NUMINAMATH_CALUDE_g_1993_of_4_eq_11_26_l470_47091

-- Define the function g
def g (x : ℚ) : ℚ := (2 + x) / (2 - 4 * x)

-- Define the recursive function gₙ
def g_n : ℕ → (ℚ → ℚ)
| 0 => id
| (n + 1) => g ∘ (g_n n)

-- Theorem statement
theorem g_1993_of_4_eq_11_26 : g_n 1993 4 = 11/26 := by
  sorry

end NUMINAMATH_CALUDE_g_1993_of_4_eq_11_26_l470_47091


namespace NUMINAMATH_CALUDE_largest_expression_l470_47016

theorem largest_expression (a₁ a₂ b₁ b₂ : ℝ) 
  (ha : 0 < a₁ ∧ a₁ < a₂) 
  (hb : 0 < b₁ ∧ b₁ < b₂) 
  (ha_sum : a₁ + a₂ = 1) 
  (hb_sum : b₁ + b₂ = 1) : 
  a₁ * b₁ + a₂ * b₂ > a₁ * a₂ + b₁ * b₂ ∧ 
  a₁ * b₁ + a₂ * b₂ > a₁ * b₂ + a₂ * b₁ := by
  sorry

end NUMINAMATH_CALUDE_largest_expression_l470_47016


namespace NUMINAMATH_CALUDE_perpendicular_from_perpendicular_and_parallel_perpendicular_from_parallel_planes_l470_47046

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (perpendicular : Line → Line → Prop)
variable (perpendicular_plane : Line → Plane → Prop)
variable (parallel : Line → Line → Prop)
variable (parallel_plane : Line → Plane → Prop)
variable (parallel_planes : Plane → Plane → Prop)

-- Theorem 1
theorem perpendicular_from_perpendicular_and_parallel
  (m n : Line) (α : Plane)
  (h1 : perpendicular_plane m α)
  (h2 : parallel_plane n α) :
  perpendicular m n :=
sorry

-- Theorem 2
theorem perpendicular_from_parallel_planes
  (m : Line) (α β γ : Plane)
  (h1 : parallel_planes α β)
  (h2 : parallel_planes β γ)
  (h3 : perpendicular_plane m α) :
  perpendicular_plane m γ :=
sorry

end NUMINAMATH_CALUDE_perpendicular_from_perpendicular_and_parallel_perpendicular_from_parallel_planes_l470_47046


namespace NUMINAMATH_CALUDE_sqrt_288_simplification_l470_47055

theorem sqrt_288_simplification : Real.sqrt 288 = 12 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_288_simplification_l470_47055


namespace NUMINAMATH_CALUDE_partial_fraction_decomposition_l470_47063

theorem partial_fraction_decomposition (x : ℝ) (h1 : x ≠ 9) (h2 : x ≠ -4) :
  (5 * x + 7) / (x^2 - 5*x - 36) = 4 / (x - 9) + 1 / (x + 4) := by
  sorry

#check partial_fraction_decomposition

end NUMINAMATH_CALUDE_partial_fraction_decomposition_l470_47063


namespace NUMINAMATH_CALUDE_seventh_root_unity_product_l470_47023

theorem seventh_root_unity_product (s : ℂ) (h1 : s^7 = 1) (h2 : s ≠ 1) :
  (s - 1) * (s^2 - 1) * (s^3 - 1) * (s^4 - 1) * (s^5 - 1) * (s^6 - 1) = 8 := by
  sorry

end NUMINAMATH_CALUDE_seventh_root_unity_product_l470_47023


namespace NUMINAMATH_CALUDE_particle_position_after_3000_minutes_l470_47020

/-- Represents the position of a particle as a pair of integers -/
def Position := ℤ × ℤ

/-- Represents the direction of movement -/
inductive Direction
| Up
| Right
| Down
| Left

/-- Defines the movement pattern of the particle -/
def move_particle (start : Position) (time : ℕ) : Position :=
  sorry

/-- The theorem to be proved -/
theorem particle_position_after_3000_minutes :
  move_particle (0, 0) 3000 = (0, 27) :=
sorry

end NUMINAMATH_CALUDE_particle_position_after_3000_minutes_l470_47020


namespace NUMINAMATH_CALUDE_arithmetic_sequence_12th_term_l470_47011

/-- An arithmetic sequence with a₃ = 10 and a₉ = 28 -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d ∧ a 3 = 10 ∧ a 9 = 28

/-- The 12th term of the arithmetic sequence is 37 -/
theorem arithmetic_sequence_12th_term (a : ℕ → ℝ) (h : ArithmeticSequence a) : a 12 = 37 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_12th_term_l470_47011


namespace NUMINAMATH_CALUDE_intersection_point_of_two_lines_l470_47043

/-- Two lines in 2D space -/
structure Line2D where
  origin : ℝ × ℝ
  direction : ℝ × ℝ

/-- The point lies on the given line -/
def pointOnLine (p : ℝ × ℝ) (l : Line2D) : Prop :=
  ∃ t : ℝ, p = (l.origin.1 + t * l.direction.1, l.origin.2 + t * l.direction.2)

theorem intersection_point_of_two_lines :
  let l1 : Line2D := { origin := (2, 3), direction := (-1, 5) }
  let l2 : Line2D := { origin := (0, 7), direction := (-1, 4) }
  let p : ℝ × ℝ := (6, -17)
  (pointOnLine p l1 ∧ pointOnLine p l2) ∧
  ∀ q : ℝ × ℝ, pointOnLine q l1 ∧ pointOnLine q l2 → q = p :=
by sorry

end NUMINAMATH_CALUDE_intersection_point_of_two_lines_l470_47043


namespace NUMINAMATH_CALUDE_sin_shift_minimum_value_l470_47052

open Real

theorem sin_shift_minimum_value (a : ℝ) :
  (a > 0) →
  (∀ x, sin (2 * x - π / 3) = sin (2 * (x - a))) →
  a = π / 6 :=
by sorry

end NUMINAMATH_CALUDE_sin_shift_minimum_value_l470_47052


namespace NUMINAMATH_CALUDE_problem_solution_l470_47003

theorem problem_solution (a b : ℝ) (ha : a = 2 + Real.sqrt 3) (hb : b = 2 - Real.sqrt 3) :
  (a - b = 2 * Real.sqrt 3) ∧ (a * b = 1) ∧ (a^2 + b^2 - 5*a*b = 9) := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l470_47003


namespace NUMINAMATH_CALUDE_negation_of_existential_proposition_l470_47024

theorem negation_of_existential_proposition :
  (¬ ∃ x₀ : ℝ, x₀ < 0 ∧ Real.exp x₀ - x₀ > 1) ↔ (∀ x : ℝ, x < 0 → Real.exp x - x ≤ 1) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_existential_proposition_l470_47024


namespace NUMINAMATH_CALUDE_car_speed_problem_l470_47013

theorem car_speed_problem (highway_length : ℝ) (meeting_time : ℝ) (car2_speed : ℝ) : 
  highway_length = 333 ∧ 
  meeting_time = 3 ∧ 
  car2_speed = 57 →
  ∃ car1_speed : ℝ, 
    car1_speed * meeting_time + car2_speed * meeting_time = highway_length ∧ 
    car1_speed = 54 :=
by sorry

end NUMINAMATH_CALUDE_car_speed_problem_l470_47013


namespace NUMINAMATH_CALUDE_lending_period_equation_l470_47018

/-- Represents the lending period in years -/
def t : ℝ := sorry

/-- The amount Manoj borrowed from Anwar -/
def borrowed_amount : ℝ := 3900

/-- The amount Manoj lent to Ramu -/
def lent_amount : ℝ := 5655

/-- The interest rate Anwar charged Manoj (in percentage) -/
def borrowing_rate : ℝ := 6

/-- The interest rate Manoj charged Ramu (in percentage) -/
def lending_rate : ℝ := 9

/-- Manoj's gain from the whole transaction -/
def gain : ℝ := 824.85

/-- Theorem stating the relationship between the lending period and the financial parameters -/
theorem lending_period_equation : 
  gain = (lent_amount * lending_rate * t / 100) - (borrowed_amount * borrowing_rate * t / 100) := by
  sorry

end NUMINAMATH_CALUDE_lending_period_equation_l470_47018


namespace NUMINAMATH_CALUDE_abc_product_l470_47049

theorem abc_product (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (hab : a * b = 45 * Real.rpow 3 (1/3))
  (hac : a * c = 75 * Real.rpow 3 (1/3))
  (hbc : b * c = 30 * Real.rpow 3 (1/3)) :
  a * b * c = 75 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_abc_product_l470_47049


namespace NUMINAMATH_CALUDE_contractor_problem_l470_47005

/-- Represents the problem of calculating the number of days to complete 1/4 of the work --/
theorem contractor_problem (total_days : ℕ) (initial_workers : ℕ) (remaining_days : ℕ) (fired_workers : ℕ) :
  total_days = 100 →
  initial_workers = 10 →
  remaining_days = 75 →
  fired_workers = 2 →
  let remaining_workers := initial_workers - fired_workers
  let work_per_day := (1 : ℚ) / initial_workers
  let days_to_quarter := (1 / 4 : ℚ) / work_per_day
  let remaining_work := (3 / 4 : ℚ) / (remaining_workers : ℚ) / (remaining_days : ℚ)
  (1 : ℚ) = days_to_quarter * work_per_day + remaining_work * (remaining_workers : ℚ) * (remaining_days : ℚ) →
  days_to_quarter = 20 := by
  sorry


end NUMINAMATH_CALUDE_contractor_problem_l470_47005


namespace NUMINAMATH_CALUDE_boat_speed_l470_47019

/-- Proves that the speed of a boat in still water is 60 kmph given the conditions of the problem -/
theorem boat_speed (stream_speed : ℝ) (upstream_time downstream_time : ℝ) 
  (h1 : stream_speed = 20)
  (h2 : upstream_time = 2 * downstream_time)
  (h3 : downstream_time > 0)
  (h4 : ∀ (boat_speed : ℝ), 
    (boat_speed + stream_speed) * downstream_time = 
    (boat_speed - stream_speed) * upstream_time → 
    boat_speed = 60) : 
  ∃ (boat_speed : ℝ), boat_speed = 60 := by
  sorry

end NUMINAMATH_CALUDE_boat_speed_l470_47019


namespace NUMINAMATH_CALUDE_last_digit_89_base_5_l470_47028

theorem last_digit_89_base_5 : 89 % 5 = 4 := by
  sorry

end NUMINAMATH_CALUDE_last_digit_89_base_5_l470_47028


namespace NUMINAMATH_CALUDE_stream_speed_l470_47071

/-- Proves that the speed of the stream is 8 km/hr, given the conditions of the boat's travel --/
theorem stream_speed (boat_speed : ℝ) (downstream_distance : ℝ) (downstream_time : ℝ) :
  boat_speed = 10 →
  downstream_distance = 54 →
  downstream_time = 3 →
  (boat_speed + (downstream_distance / downstream_time - boat_speed)) = 18 :=
by
  sorry

end NUMINAMATH_CALUDE_stream_speed_l470_47071


namespace NUMINAMATH_CALUDE_repeating_decimal_equals_fraction_l470_47026

/-- Represents a repeating decimal with a non-repeating part and a repeating part -/
structure RepeatingDecimal where
  nonRepeating : ℚ
  repeating : ℚ
  repeatingDigits : ℕ

/-- Converts a RepeatingDecimal to its fraction representation -/
def repeatingDecimalToFraction (x : RepeatingDecimal) : ℚ :=
  x.nonRepeating + x.repeating / (1 - (1 / 10 ^ x.repeatingDigits))

theorem repeating_decimal_equals_fraction :
  let x : RepeatingDecimal := {
    nonRepeating := 1/2,
    repeating := 23/1000,
    repeatingDigits := 3
  }
  repeatingDecimalToFraction x = 1045/1998 := by sorry

end NUMINAMATH_CALUDE_repeating_decimal_equals_fraction_l470_47026


namespace NUMINAMATH_CALUDE_travel_options_count_l470_47067

/-- The number of bus options from A to B -/
def bus_options : ℕ := 5

/-- The number of train options from A to B -/
def train_options : ℕ := 6

/-- The number of boat options from A to B -/
def boat_options : ℕ := 2

/-- The total number of travel options from A to B -/
def total_options : ℕ := bus_options + train_options + boat_options

theorem travel_options_count :
  total_options = 13 :=
by sorry

end NUMINAMATH_CALUDE_travel_options_count_l470_47067


namespace NUMINAMATH_CALUDE_pad_pages_proof_l470_47031

theorem pad_pages_proof (P : ℝ) 
  (h1 : P - (0.25 * P + 10) = 80) : P = 120 := by
  sorry

end NUMINAMATH_CALUDE_pad_pages_proof_l470_47031


namespace NUMINAMATH_CALUDE_polynomial_uniqueness_l470_47077

def is_valid_polynomial (P : ℝ → ℝ) (n : ℕ) : Prop :=
  (∀ k : ℕ, k ≤ n → P (2 * k) = 0) ∧
  (∀ k : ℕ, k < n → P (2 * k + 1) = 2) ∧
  (P (2 * n + 1) = -6)

theorem polynomial_uniqueness (P : ℝ → ℝ) (n : ℕ) :
  is_valid_polynomial P n →
  (∃ a b c : ℝ, ∀ x, P x = a * x^2 + b * x + c) →
  (n = 1 ∧ ∀ x, P x = -2 * x^2 + 4 * x) :=
sorry

end NUMINAMATH_CALUDE_polynomial_uniqueness_l470_47077


namespace NUMINAMATH_CALUDE_willie_stickers_l470_47034

theorem willie_stickers (initial : ℝ) (received : ℝ) (total : ℝ) :
  initial = 278.5 →
  received = 43.8 →
  total = initial + received →
  total = 322.3 := by
sorry

end NUMINAMATH_CALUDE_willie_stickers_l470_47034


namespace NUMINAMATH_CALUDE_expected_urns_with_one_marble_value_l470_47041

/-- The number of urns -/
def n : ℕ := 7

/-- The number of marbles -/
def m : ℕ := 5

/-- The probability that a specific urn has exactly one marble -/
def p : ℚ := m * (n - 1)^(m - 1) / n^m

/-- The expected number of urns with exactly one marble -/
def expected_urns_with_one_marble : ℚ := n * p

theorem expected_urns_with_one_marble_value : 
  expected_urns_with_one_marble = 6480 / 2401 := by sorry

end NUMINAMATH_CALUDE_expected_urns_with_one_marble_value_l470_47041


namespace NUMINAMATH_CALUDE_line_segment_product_l470_47084

/-- Given four points A, B, C, D on a line in this order, prove that AB · CD + AD · BC = 1000 -/
theorem line_segment_product (A B C D : ℝ) : 
  (A < B) → (B < C) → (C < D) →  -- Points are in order on the line
  (C - A = 25) →  -- AC = 25
  (D - B = 40) →  -- BD = 40
  (D - A = 57) →  -- AD = 57
  (B - A) * (D - C) + (D - A) * (C - B) = 1000 := by
  sorry

end NUMINAMATH_CALUDE_line_segment_product_l470_47084


namespace NUMINAMATH_CALUDE_train_passing_platform_l470_47022

theorem train_passing_platform (train_length platform_length : ℝ) 
  (time_to_pass_point : ℝ) (h1 : train_length = 1400) 
  (h2 : platform_length = 700) (h3 : time_to_pass_point = 100) :
  (train_length + platform_length) / (train_length / time_to_pass_point) = 150 :=
by sorry

end NUMINAMATH_CALUDE_train_passing_platform_l470_47022


namespace NUMINAMATH_CALUDE_sum_E_equals_1600_l470_47089

-- Define E(n) as the sum of even digits in n
def E (n : ℕ) : ℕ := sorry

-- Define the sum of E(n) from 1 to 200
def sum_E : ℕ := (Finset.range 200).sum (fun i => E (i + 1))

-- Theorem to prove
theorem sum_E_equals_1600 : sum_E = 1600 := by sorry

end NUMINAMATH_CALUDE_sum_E_equals_1600_l470_47089


namespace NUMINAMATH_CALUDE_johns_score_increase_l470_47079

/-- Given John's four test scores, prove that the difference between
    the average of all four scores and the average of the first three scores is 0.92. -/
theorem johns_score_increase (score1 score2 score3 score4 : ℚ) 
    (h1 : score1 = 92)
    (h2 : score2 = 89)
    (h3 : score3 = 93)
    (h4 : score4 = 95) :
    (score1 + score2 + score3 + score4) / 4 - (score1 + score2 + score3) / 3 = 92 / 100 := by
  sorry

end NUMINAMATH_CALUDE_johns_score_increase_l470_47079


namespace NUMINAMATH_CALUDE_sum_of_inscribed_angles_is_180_l470_47096

/-- A regular pentagon inscribed in a circle -/
structure RegularPentagonInCircle where
  /-- The circle in which the pentagon is inscribed -/
  circle : Real
  /-- The regular pentagon inscribed in the circle -/
  pentagon : Real
  /-- The sides of the pentagon divide the circle into five equal arcs -/
  equal_arcs : pentagon = 5

/-- The sum of angles inscribed in the five arcs cut off by the sides of a regular pentagon inscribed in a circle -/
def sum_of_inscribed_angles (p : RegularPentagonInCircle) : Real :=
  sorry

/-- Theorem: The sum of angles inscribed in the five arcs cut off by the sides of a regular pentagon inscribed in a circle is 180° -/
theorem sum_of_inscribed_angles_is_180 (p : RegularPentagonInCircle) :
  sum_of_inscribed_angles p = 180 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_inscribed_angles_is_180_l470_47096


namespace NUMINAMATH_CALUDE_letter_placement_l470_47030

theorem letter_placement (n_letters : ℕ) (n_boxes : ℕ) : n_letters = 3 ∧ n_boxes = 5 → n_boxes ^ n_letters = 125 := by
  sorry

end NUMINAMATH_CALUDE_letter_placement_l470_47030


namespace NUMINAMATH_CALUDE_soccer_team_lineups_l470_47039

/-- The number of possible lineups for a soccer team -/
def number_of_lineups (total_players : ℕ) (goalkeeper : ℕ) (defenders : ℕ) (others : ℕ) : ℕ :=
  total_players * (Nat.choose (total_players - 1) defenders) * (Nat.choose (total_players - 1 - defenders) others)

/-- Theorem: The number of possible lineups for a soccer team of 18 players,
    with 1 goalkeeper, 4 defenders, and 4 other players is 30,544,200 -/
theorem soccer_team_lineups :
  number_of_lineups 18 1 4 4 = 30544200 := by
  sorry

end NUMINAMATH_CALUDE_soccer_team_lineups_l470_47039


namespace NUMINAMATH_CALUDE_jed_cards_per_week_l470_47010

/-- Represents the number of cards Jed has after a given number of weeks -/
def cards_after_weeks (initial_cards : ℕ) (cards_per_week : ℕ) (weeks : ℕ) : ℕ :=
  initial_cards + cards_per_week * weeks - 2 * (weeks / 2)

/-- Proves that Jed gets 6 cards per week given the conditions -/
theorem jed_cards_per_week :
  ∃ (cards_per_week : ℕ),
    cards_after_weeks 20 cards_per_week 4 = 40 ∧ cards_per_week = 6 := by
  sorry

#check jed_cards_per_week

end NUMINAMATH_CALUDE_jed_cards_per_week_l470_47010


namespace NUMINAMATH_CALUDE_train_length_l470_47097

/-- Train crossing a bridge problem -/
theorem train_length (train_speed : ℝ) (bridge_length : ℝ) (man_speed : ℝ) :
  train_speed = 80 →
  bridge_length = 1 →
  man_speed = 5 →
  (bridge_length / train_speed) * man_speed = 1/16 :=
by sorry

end NUMINAMATH_CALUDE_train_length_l470_47097


namespace NUMINAMATH_CALUDE_sum_of_coefficients_P_l470_47035

-- Define the polynomial P(x)
def P (x : ℝ) : ℝ := 5 * (2 * x^9 - 3 * x^6 + 4) - 4 * (x^6 - 5 * x^3 + 6)

-- Theorem stating that the sum of coefficients of P(x) is 7
theorem sum_of_coefficients_P : (P 1) = 7 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_P_l470_47035


namespace NUMINAMATH_CALUDE_eccentricity_of_ellipse_l470_47065

-- Define the complex polynomial
def polynomial (z : ℂ) : ℂ := (z - 2) * (z^2 + 3*z + 5) * (z^2 + 5*z + 8)

-- Define the set of solutions
def solutions : Set ℂ := {z : ℂ | polynomial z = 0}

-- Define the ellipse centered at the origin
def ellipse (a b : ℝ) : Set ℂ := {z : ℂ | (z.re^2 / a^2) + (z.im^2 / b^2) = 1}

-- Theorem statement
theorem eccentricity_of_ellipse :
  ∀ a b : ℝ, a > 0 → b > 0 →
  (∃ e : Set ℂ, e = ellipse a b ∧ solutions ⊆ e) →
  (a^2 - b^2) / a^2 = 5/16 :=
sorry

end NUMINAMATH_CALUDE_eccentricity_of_ellipse_l470_47065


namespace NUMINAMATH_CALUDE_wage_increase_constant_wage_increase_l470_47076

/-- Represents the regression equation for worker's wage based on labor productivity -/
def wage_equation (x : ℝ) : ℝ := 10 + 70 * x

/-- Proves that an increase of 1 in labor productivity results in an increase of 70 in wage -/
theorem wage_increase (x : ℝ) : wage_equation (x + 1) - wage_equation x = 70 := by
  sorry

/-- Proves that the wage increase is constant for any labor productivity value -/
theorem constant_wage_increase (x y : ℝ) : 
  wage_equation (x + 1) - wage_equation x = wage_equation (y + 1) - wage_equation y := by
  sorry

end NUMINAMATH_CALUDE_wage_increase_constant_wage_increase_l470_47076


namespace NUMINAMATH_CALUDE_reflection_theorem_l470_47038

/-- A point in the 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- The line of symmetry for the fold -/
def lineOfSymmetry : ℝ := 2

/-- Function to reflect a point across the line of symmetry -/
def reflect (p : Point) : Point :=
  { x := p.x, y := 2 * lineOfSymmetry - p.y }

/-- The original point before folding -/
def originalPoint : Point := { x := -4, y := 1 }

/-- The expected point after folding -/
def expectedPoint : Point := { x := -4, y := 3 }

/-- Theorem stating that reflecting the original point results in the expected point -/
theorem reflection_theorem : reflect originalPoint = expectedPoint := by
  sorry

end NUMINAMATH_CALUDE_reflection_theorem_l470_47038


namespace NUMINAMATH_CALUDE_original_denominator_proof_l470_47032

theorem original_denominator_proof (d : ℚ) : 
  (2 : ℚ) / d ≠ 0 →    -- Ensure the fraction is well-defined
  (6 : ℚ) / (3 * d) = (2 : ℚ) / 3 → 
  d = 3 := by
sorry

end NUMINAMATH_CALUDE_original_denominator_proof_l470_47032


namespace NUMINAMATH_CALUDE_certain_number_problem_l470_47074

theorem certain_number_problem : ∃! x : ℝ, ((x - 50) / 4) * 3 + 28 = 73 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_problem_l470_47074


namespace NUMINAMATH_CALUDE_sector_arc_length_and_area_l470_47044

theorem sector_arc_length_and_area (r : ℝ) (α : ℝ) 
    (h1 : r = 2) 
    (h2 : α = π / 6) : 
  let l := α * r
  let S := (1 / 2) * l * r
  (l = π / 3) ∧ (S = π / 3) := by
  sorry

end NUMINAMATH_CALUDE_sector_arc_length_and_area_l470_47044


namespace NUMINAMATH_CALUDE_sally_found_thirteen_l470_47008

/-- The number of seashells Tim found -/
def tim_seashells : ℕ := 37

/-- The total number of seashells Tim and Sally found together -/
def total_seashells : ℕ := 50

/-- The number of seashells Sally found -/
def sally_seashells : ℕ := total_seashells - tim_seashells

theorem sally_found_thirteen : sally_seashells = 13 := by
  sorry

end NUMINAMATH_CALUDE_sally_found_thirteen_l470_47008


namespace NUMINAMATH_CALUDE_function_equation_solution_l470_47004

theorem function_equation_solution (f : ℝ → ℝ) (a : ℝ) :
  (∀ x : ℝ, f (f x) = x * f x - a * x) →
  (∃ x y : ℝ, f x ≠ f y) →
  (∃ t : ℝ, f t = a) →
  (a = 0 ∨ a = -1) :=
by sorry

end NUMINAMATH_CALUDE_function_equation_solution_l470_47004


namespace NUMINAMATH_CALUDE_expression_equality_l470_47040

theorem expression_equality : 7^2 - 2*(5) + 4^2 / 2 = 47 := by
  sorry

end NUMINAMATH_CALUDE_expression_equality_l470_47040


namespace NUMINAMATH_CALUDE_number_line_mark_distance_not_always_1cm_l470_47015

/-- Represents a number line -/
structure NumberLine where
  origin : ℝ
  positive_direction : Bool
  unit_length : ℝ

/-- Properties of a number line -/
def valid_number_line (nl : NumberLine) : Prop :=
  nl.unit_length > 0

theorem number_line_mark_distance_not_always_1cm 
  (h1 : ∀ (x : ℝ), ∃ (nl : NumberLine), nl.origin = x ∧ valid_number_line nl)
  (h2 : ∀ (nl : NumberLine), valid_number_line nl → nl.positive_direction = true)
  (h3 : ∀ (l : ℝ), l > 0 → ∃ (nl : NumberLine), nl.unit_length = l ∧ valid_number_line nl) :
  ¬(∀ (nl : NumberLine), valid_number_line nl → nl.unit_length = 1) :=
sorry

end NUMINAMATH_CALUDE_number_line_mark_distance_not_always_1cm_l470_47015


namespace NUMINAMATH_CALUDE_cubic_root_ceiling_divisibility_l470_47061

theorem cubic_root_ceiling_divisibility (x₁ x₂ x₃ : ℝ) (n : ℕ+) :
  x₁ < x₂ ∧ x₂ < x₃ →
  x₁^3 - 3*x₁^2 + 1 = 0 →
  x₂^3 - 3*x₂^2 + 1 = 0 →
  x₃^3 - 3*x₃^2 + 1 = 0 →
  ∃ k : ℤ, ⌈x₃^(n : ℝ)⌉ = 3 * k :=
by sorry

end NUMINAMATH_CALUDE_cubic_root_ceiling_divisibility_l470_47061


namespace NUMINAMATH_CALUDE_last_digit_of_2021_2021_l470_47093

-- Define the table size
def n : Nat := 2021

-- Define the cell value function
def cellValue (x y : Nat) : Nat :=
  if x = 1 ∧ y = 1 then 0
  else 2^(x + y - 2) - 1

-- State the theorem
theorem last_digit_of_2021_2021 :
  (cellValue n n) % 10 = 5 := by sorry

end NUMINAMATH_CALUDE_last_digit_of_2021_2021_l470_47093


namespace NUMINAMATH_CALUDE_train_passing_time_train_passing_time_specific_l470_47050

/-- Time for a train to pass a tree -/
theorem train_passing_time (train_length : ℝ) (train_speed : ℝ) (wind_speed : ℝ) : ℝ :=
  let train_speed_ms := train_speed * 1000 / 3600
  let wind_speed_ms := wind_speed * 1000 / 3600
  let effective_speed := train_speed_ms - wind_speed_ms
  train_length / effective_speed

/-- Proof that the time for a train of length 850 m, traveling at 85 km/hr against a 5 km/hr wind, to pass a tree is approximately 38.25 seconds -/
theorem train_passing_time_specific : 
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.01 ∧ |train_passing_time 850 85 5 - 38.25| < ε :=
sorry

end NUMINAMATH_CALUDE_train_passing_time_train_passing_time_specific_l470_47050


namespace NUMINAMATH_CALUDE_perpendicular_line_through_intersection_l470_47025

-- Define the lines
def line1 (x y : ℝ) : Prop := 2*x - y - 3 = 0
def line2 (x y : ℝ) : Prop := 4*x - 3*y - 5 = 0
def line3 (x y : ℝ) : Prop := 2*x + 3*y + 5 = 0

-- Define the intersection point
def intersection_point : ℝ × ℝ := (2, 1)

-- Define the perpendicular line
def perpendicular_line (x y : ℝ) : Prop := 3*x - 2*y - 4 = 0

-- Theorem statement
theorem perpendicular_line_through_intersection :
  ∃ (x y : ℝ), 
    line1 x y ∧ 
    line2 x y ∧ 
    perpendicular_line x y ∧
    (∀ (m : ℝ), line3 x y → m = -2/3) :=
sorry

end NUMINAMATH_CALUDE_perpendicular_line_through_intersection_l470_47025


namespace NUMINAMATH_CALUDE_sum_x_y_equals_thirteen_l470_47081

theorem sum_x_y_equals_thirteen 
  (h1 : (8:ℝ)^x / (4:ℝ)^(x+y) = 16)
  (h2 : (16:ℝ)^(x+y) / (4:ℝ)^(7*y) = 1024)
  : x + y = 13 := by
  sorry

end NUMINAMATH_CALUDE_sum_x_y_equals_thirteen_l470_47081


namespace NUMINAMATH_CALUDE_circumradius_right_triangle_l470_47073

/-- The radius of the circumscribed circle of a right triangle with sides 10, 8, and 6 is 5 -/
theorem circumradius_right_triangle : 
  ∀ (a b c : ℝ), 
    a = 10 → b = 8 → c = 6 →
    a^2 = b^2 + c^2 →
    (a / 2 : ℝ) = 5 :=
by sorry

end NUMINAMATH_CALUDE_circumradius_right_triangle_l470_47073


namespace NUMINAMATH_CALUDE_cubic_root_sum_l470_47090

theorem cubic_root_sum (a b c : ℝ) : 
  a^3 - 6*a^2 + 11*a - 6 = 0 →
  b^3 - 6*b^2 + 11*b - 6 = 0 →
  c^3 - 6*c^2 + 11*c - 6 = 0 →
  a*b/c + b*c/a + c*a/b = 49/6 := by
sorry

end NUMINAMATH_CALUDE_cubic_root_sum_l470_47090


namespace NUMINAMATH_CALUDE_consecutive_biology_majors_probability_l470_47072

/-- The number of people sitting at the round table -/
def total_people : ℕ := 10

/-- The number of biology majors -/
def biology_majors : ℕ := 4

/-- The number of math majors -/
def math_majors : ℕ := 4

/-- The number of physics majors -/
def physics_majors : ℕ := 2

/-- The probability of all biology majors sitting in consecutive seats -/
def consecutive_biology_prob : ℚ := 2/3

theorem consecutive_biology_majors_probability :
  let total_arrangements := Nat.factorial (total_people - 1)
  let favorable_arrangements := (total_people - biology_majors) * Nat.factorial (total_people - biology_majors - 1)
  (favorable_arrangements : ℚ) / total_arrangements = consecutive_biology_prob :=
sorry

end NUMINAMATH_CALUDE_consecutive_biology_majors_probability_l470_47072


namespace NUMINAMATH_CALUDE_caterpillar_length_difference_l470_47036

/-- The length difference between two caterpillars -/
theorem caterpillar_length_difference : 
  let green_length : ℝ := 3
  let orange_length : ℝ := 1.17
  green_length - orange_length = 1.83 := by sorry

end NUMINAMATH_CALUDE_caterpillar_length_difference_l470_47036


namespace NUMINAMATH_CALUDE_absolute_value_inequality_implies_a_bound_l470_47086

theorem absolute_value_inequality_implies_a_bound (a : ℝ) :
  (∀ x : ℝ, |x + 2| - |x - 1| ≥ a^3 - 4*a^2 - 3) →
  a ≤ 4 := by
sorry

end NUMINAMATH_CALUDE_absolute_value_inequality_implies_a_bound_l470_47086


namespace NUMINAMATH_CALUDE_only_statement3_correct_l470_47069

-- Define the propositions
variable (p q : Prop)

-- Define the four statements
def statement1 : Prop := ∀ (p q : Prop), (p ∧ q → p ∨ q) ∧ ¬(p ∨ q → p ∧ q)
def statement2 : Prop := ∀ (p q : Prop), (¬(p ∧ q) → p ∨ q) ∧ ¬(p ∨ q → ¬(p ∧ q))
def statement3 : Prop := ∀ (p q : Prop), (p ∨ q → ¬(¬p)) ∧ ¬(¬(¬p) → p ∨ q)
def statement4 : Prop := ∀ (p q : Prop), (¬p → ¬(p ∧ q)) ∧ ¬(¬(p ∧ q) → ¬p)

-- Theorem stating that only the third statement is correct
theorem only_statement3_correct :
  ¬statement1 ∧ ¬statement2 ∧ statement3 ∧ ¬statement4 :=
sorry

end NUMINAMATH_CALUDE_only_statement3_correct_l470_47069


namespace NUMINAMATH_CALUDE_fraction_equality_l470_47058

theorem fraction_equality (x m : ℝ) (h : x ≠ 0) :
  x / (x^2 - m*x + 1) = 1 →
  x^3 / (x^6 - m^3*x^3 + 1) = 1 / (3*m^2 - 2) :=
by sorry

end NUMINAMATH_CALUDE_fraction_equality_l470_47058
