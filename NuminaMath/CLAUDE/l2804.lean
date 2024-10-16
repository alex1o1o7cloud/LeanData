import Mathlib

namespace NUMINAMATH_CALUDE_chocolate_factory_order_completion_l2804_280421

/-- Represents the number of days required to complete an order of candies. -/
def days_to_complete_order (candies_per_hour : ℕ) (hours_per_day : ℕ) (total_candies : ℕ) : ℕ :=
  (total_candies / candies_per_hour + hours_per_day - 1) / hours_per_day

/-- Theorem stating that it takes 8 days to complete the order under given conditions. -/
theorem chocolate_factory_order_completion :
  days_to_complete_order 50 10 4000 = 8 := by
  sorry

#eval days_to_complete_order 50 10 4000

end NUMINAMATH_CALUDE_chocolate_factory_order_completion_l2804_280421


namespace NUMINAMATH_CALUDE_probability_qualified_product_l2804_280485

/-- The probability of buying a qualified product from a market with two factories -/
theorem probability_qualified_product
  (factory_a_share : ℝ)
  (factory_b_share : ℝ)
  (factory_a_qualification_rate : ℝ)
  (factory_b_qualification_rate : ℝ)
  (h1 : factory_a_share = 0.6)
  (h2 : factory_b_share = 0.4)
  (h3 : factory_a_qualification_rate = 0.95)
  (h4 : factory_b_qualification_rate = 0.9)
  (h5 : factory_a_share + factory_b_share = 1) :
  factory_a_share * factory_a_qualification_rate +
  factory_b_share * factory_b_qualification_rate = 0.93 :=
by sorry

end NUMINAMATH_CALUDE_probability_qualified_product_l2804_280485


namespace NUMINAMATH_CALUDE_jerry_has_49_feathers_l2804_280490

/-- The number of feathers Jerry has left after his adventure -/
def jerrys_remaining_feathers : ℕ :=
  let hawk_feathers : ℕ := 6
  let eagle_feathers : ℕ := 17 * hawk_feathers
  let total_feathers : ℕ := hawk_feathers + eagle_feathers
  let feathers_after_giving : ℕ := total_feathers - 10
  (feathers_after_giving / 2 : ℕ)

/-- Theorem stating that Jerry has 49 feathers left -/
theorem jerry_has_49_feathers : jerrys_remaining_feathers = 49 := by
  sorry

end NUMINAMATH_CALUDE_jerry_has_49_feathers_l2804_280490


namespace NUMINAMATH_CALUDE_ten_person_round_robin_matches_l2804_280481

/-- Calculates the number of matches in a round-robin tournament -/
def roundRobinMatches (n : ℕ) : ℕ :=
  n * (n - 1) / 2

/-- Theorem: A 10-person round-robin tournament has 45 matches -/
theorem ten_person_round_robin_matches :
  roundRobinMatches 10 = 45 := by
  sorry

#eval roundRobinMatches 10  -- Should output 45

end NUMINAMATH_CALUDE_ten_person_round_robin_matches_l2804_280481


namespace NUMINAMATH_CALUDE_ship_length_in_emily_steps_l2804_280454

/-- The length of the ship in terms of Emily's steps -/
def ship_length : ℕ := 70

/-- The number of steps Emily takes from back to front of the ship -/
def steps_back_to_front : ℕ := 210

/-- The number of steps Emily takes from front to back of the ship -/
def steps_front_to_back : ℕ := 42

/-- Emily's walking speed is faster than the ship's speed -/
axiom emily_faster : ∃ (e s : ℝ), e > s ∧ e > 0 ∧ s > 0

/-- Theorem stating the length of the ship in terms of Emily's steps -/
theorem ship_length_in_emily_steps :
  ∃ (e s : ℝ), e > s ∧ e > 0 ∧ s > 0 →
  (steps_back_to_front : ℝ) * e = ship_length + steps_back_to_front * s ∧
  (steps_front_to_back : ℝ) * e = ship_length - steps_front_to_back * s →
  ship_length = 70 := by
  sorry

end NUMINAMATH_CALUDE_ship_length_in_emily_steps_l2804_280454


namespace NUMINAMATH_CALUDE_probability_three_red_balls_l2804_280437

def total_balls : ℕ := 15
def red_balls : ℕ := 10
def blue_balls : ℕ := 5
def drawn_balls : ℕ := 5
def target_red : ℕ := 3

theorem probability_three_red_balls :
  (Nat.choose red_balls target_red * Nat.choose blue_balls (drawn_balls - target_red)) /
  Nat.choose total_balls drawn_balls = 1200 / 3003 :=
sorry

end NUMINAMATH_CALUDE_probability_three_red_balls_l2804_280437


namespace NUMINAMATH_CALUDE_percentage_change_condition_l2804_280489

theorem percentage_change_condition (r s N : ℝ) 
  (hr : r > 0) (hs : s > 0) (hN : N > 0) (hs_bound : s < 50) :
  (N * (1 + r / 100) * (1 - s / 100) > N) ↔ (r > 100 * s / (100 - s)) :=
sorry

end NUMINAMATH_CALUDE_percentage_change_condition_l2804_280489


namespace NUMINAMATH_CALUDE_inverse_proportion_problem_l2804_280480

/-- Given that x and y are inversely proportional, and when their sum is 50, x is three times y,
    prove that y = -39.0625 when x = -12 -/
theorem inverse_proportion_problem (x y : ℝ) (k : ℝ) : 
  (∀ x y, x * y = k) →  -- x and y are inversely proportional
  (∃ x y, x + y = 50 ∧ x = 3 * y) →  -- when their sum is 50, x is three times y
  (x = -12 → y = -39.0625) :=  -- prove that y = -39.0625 when x = -12
by sorry

end NUMINAMATH_CALUDE_inverse_proportion_problem_l2804_280480


namespace NUMINAMATH_CALUDE_greatest_integer_prime_quadratic_l2804_280469

theorem greatest_integer_prime_quadratic : 
  ∃ (x : ℤ), (∀ (y : ℤ), y > x → ¬(Nat.Prime (Int.natAbs (4*y^2 - 35*y + 21)))) ∧ 
  (Nat.Prime (Int.natAbs (4*x^2 - 35*x + 21))) ∧ 
  x = 8 :=
by sorry

end NUMINAMATH_CALUDE_greatest_integer_prime_quadratic_l2804_280469


namespace NUMINAMATH_CALUDE_prob_red_then_green_l2804_280407

/-- A bag containing one red ball and one green ball -/
structure Bag :=
  (red : Nat)
  (green : Nat)

/-- The initial state of the bag -/
def initial_bag : Bag :=
  { red := 1, green := 1 }

/-- A draw from the bag -/
inductive Draw
  | Red
  | Green

/-- The probability of drawing a specific sequence of two balls -/
def prob_draw (first second : Draw) : ℚ :=
  1 / 4

/-- Theorem: The probability of drawing a red ball first and a green ball second is 1/4 -/
theorem prob_red_then_green :
  prob_draw Draw.Red Draw.Green = 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_prob_red_then_green_l2804_280407


namespace NUMINAMATH_CALUDE_f_definition_f_max_min_on_interval_l2804_280424

-- Define the function f
noncomputable def f : ℝ → ℝ := λ x => 2 / (x - 1)

-- Theorem for the function definition
theorem f_definition (x : ℝ) (h : x ≠ 1) : 
  f ((x - 1) / (x + 1)) = -x - 1 := by sorry

-- Theorem for the maximum and minimum values
theorem f_max_min_on_interval : 
  ∃ (max min : ℝ), (∀ x ∈ Set.Icc 2 6, f x ≤ max ∧ min ≤ f x) ∧ 
  (∃ x₁ ∈ Set.Icc 2 6, f x₁ = max) ∧ 
  (∃ x₂ ∈ Set.Icc 2 6, f x₂ = min) ∧ 
  max = 2 ∧ min = 2/5 := by sorry

end NUMINAMATH_CALUDE_f_definition_f_max_min_on_interval_l2804_280424


namespace NUMINAMATH_CALUDE_marbles_given_to_dylan_l2804_280450

/-- Given that Cade had 87 marbles initially and was left with 79 marbles after giving some to Dylan,
    prove that Cade gave 8 marbles to Dylan. -/
theorem marbles_given_to_dylan :
  ∀ (initial_marbles remaining_marbles marbles_given : ℕ),
    initial_marbles = 87 →
    remaining_marbles = 79 →
    initial_marbles = remaining_marbles + marbles_given →
    marbles_given = 8 :=
by
  sorry

end NUMINAMATH_CALUDE_marbles_given_to_dylan_l2804_280450


namespace NUMINAMATH_CALUDE_multiplication_proof_l2804_280403

theorem multiplication_proof :
  ∀ (a b c : ℕ),
  a = 60 + b →
  c = 14 →
  a * c = 882 ∧
  68 * 14 = 952 :=
by
  sorry

end NUMINAMATH_CALUDE_multiplication_proof_l2804_280403


namespace NUMINAMATH_CALUDE_inscribing_square_area_is_one_l2804_280471

/-- The circle equation -/
def circle_equation (x y : ℝ) : Prop :=
  3 * x^2 + 3 * y^2 - 15 * x + 9 * y + 27 = 0

/-- The square that inscribes the circle -/
structure InscribingSquare where
  side_length : ℝ
  center_x : ℝ
  center_y : ℝ
  parallel_to_y_axis : Prop

/-- The theorem stating that the area of the inscribing square is 1 -/
theorem inscribing_square_area_is_one :
  ∃ (s : InscribingSquare), s.side_length ^ 2 = 1 ∧
  (∀ (x y : ℝ), circle_equation x y →
    (|x - s.center_x| ≤ s.side_length / 2 ∧
     |y - s.center_y| ≤ s.side_length / 2)) :=
sorry

end NUMINAMATH_CALUDE_inscribing_square_area_is_one_l2804_280471


namespace NUMINAMATH_CALUDE_light_travel_distance_l2804_280474

/-- The distance light travels in one year in miles -/
def light_year_distance : ℝ := 5870000000000

/-- The number of years we want to calculate the light travel distance for -/
def years : ℝ := 150

/-- Theorem stating the distance light travels in 150 years -/
theorem light_travel_distance : light_year_distance * years = 8.805e14 := by
  sorry

end NUMINAMATH_CALUDE_light_travel_distance_l2804_280474


namespace NUMINAMATH_CALUDE_cube_fraction_product_l2804_280400

theorem cube_fraction_product : 
  (((7^3 - 1) / (7^3 + 1)) * 
   ((8^3 - 1) / (8^3 + 1)) * 
   ((9^3 - 1) / (9^3 + 1)) * 
   ((10^3 - 1) / (10^3 + 1)) * 
   ((11^3 - 1) / (11^3 + 1))) = 931 / 946 := by
  sorry

end NUMINAMATH_CALUDE_cube_fraction_product_l2804_280400


namespace NUMINAMATH_CALUDE_apples_packed_in_two_weeks_l2804_280402

/-- Calculates the total number of apples packed in two weeks under specific conditions -/
theorem apples_packed_in_two_weeks
  (apples_per_box : ℕ)
  (boxes_per_day : ℕ)
  (days_per_week : ℕ)
  (fewer_apples_second_week : ℕ)
  (h1 : apples_per_box = 40)
  (h2 : boxes_per_day = 50)
  (h3 : days_per_week = 7)
  (h4 : fewer_apples_second_week = 500) :
  apples_per_box * boxes_per_day * days_per_week +
  (apples_per_box * boxes_per_day - fewer_apples_second_week) * days_per_week = 24500 :=
by sorry

#check apples_packed_in_two_weeks

end NUMINAMATH_CALUDE_apples_packed_in_two_weeks_l2804_280402


namespace NUMINAMATH_CALUDE_sum_even_factors_1176_l2804_280493

def sum_even_factors (n : ℕ) : ℕ := sorry

theorem sum_even_factors_1176 : sum_even_factors 1176 = 3192 := by sorry

end NUMINAMATH_CALUDE_sum_even_factors_1176_l2804_280493


namespace NUMINAMATH_CALUDE_dog_speed_calculation_l2804_280475

/-- Calculates the speed of a dog running between two people moving towards each other. -/
theorem dog_speed_calculation (distance : ℝ) (people_speed : ℝ) (dog_distance : ℝ) : 
  distance = 240 → 
  people_speed = 3 → 
  dog_distance = 400 → 
  (dog_distance / (distance / (2 * people_speed))) = 10 := by
  sorry

end NUMINAMATH_CALUDE_dog_speed_calculation_l2804_280475


namespace NUMINAMATH_CALUDE_betty_sugar_purchase_l2804_280494

/-- The minimum amount of sugar Betty can buy given the constraints on flour and sugar purchases. -/
theorem betty_sugar_purchase (f s : ℝ) : 
  (f ≥ 10 + 3 * s) → (f ≤ 4 * s) → s ≥ 10 := by sorry

end NUMINAMATH_CALUDE_betty_sugar_purchase_l2804_280494


namespace NUMINAMATH_CALUDE_integer_solutions_l2804_280497

def is_integer (x : ℚ) : Prop := ∃ (n : ℤ), x = n

def expression_a (n : ℤ) : ℚ := (n^4 + 3) / (n^2 + n + 1)
def expression_b (n : ℤ) : ℚ := (n^3 + n + 1) / (n^2 - n + 1)

theorem integer_solutions :
  (∀ n : ℤ, is_integer (expression_a n) ↔ n = -3 ∨ n = -1 ∨ n = 0) ∧
  (∀ n : ℤ, is_integer (expression_b n) ↔ n = 0 ∨ n = 1) :=
sorry

end NUMINAMATH_CALUDE_integer_solutions_l2804_280497


namespace NUMINAMATH_CALUDE_exists_concave_to_convex_map_not_exists_convex_to_concave_map_l2804_280404

-- Define the plane
def Plane := ℝ × ℝ

-- Define a polygon as a list of points in the plane
def Polygon := List Plane

-- Define a simple polygon
def SimplePolygon (p : Polygon) : Prop := sorry

-- Define a convex polygon
def ConvexPolygon (p : Polygon) : Prop := sorry

-- Define a concave polygon
def ConcavePolygon (p : Polygon) : Prop := ¬ConvexPolygon p

-- State the existence of the function for part (a)
theorem exists_concave_to_convex_map :
  ∃ (f : Plane → Plane), ∀ (n : ℕ) (p : Polygon),
    n ≥ 4 →
    SimplePolygon p →
    ConcavePolygon p →
    p.length = n →
    ∃ (q : Polygon), SimplePolygon q ∧ ConvexPolygon q ∧ q = p.map f :=
sorry

-- State the non-existence of the function for part (b)
theorem not_exists_convex_to_concave_map :
  ¬∃ (f : Plane → Plane), ∀ (n : ℕ) (p : Polygon),
    n ≥ 4 →
    SimplePolygon p →
    ConvexPolygon p →
    p.length = n →
    ∃ (q : Polygon), SimplePolygon q ∧ ConcavePolygon q ∧ q = p.map f :=
sorry

end NUMINAMATH_CALUDE_exists_concave_to_convex_map_not_exists_convex_to_concave_map_l2804_280404


namespace NUMINAMATH_CALUDE_outfit_count_is_688_l2804_280447

/-- Represents the number of items of clothing -/
structure ClothingCounts where
  redShirts : Nat
  greenShirts : Nat
  bluePants : Nat
  greenPants : Nat
  greenHats : Nat
  redHats : Nat

/-- Calculates the number of valid outfits given clothing counts -/
def countOutfits (c : ClothingCounts) : Nat :=
  (c.redShirts * c.greenHats * c.greenPants) + (c.greenShirts * c.redHats * c.bluePants)

/-- Theorem stating that the number of outfits is 688 given the specific clothing counts -/
theorem outfit_count_is_688 :
  let counts : ClothingCounts := {
    redShirts := 5,
    greenShirts := 7,
    bluePants := 8,
    greenPants := 6,
    greenHats := 8,
    redHats := 8
  }
  countOutfits counts = 688 := by
  sorry

end NUMINAMATH_CALUDE_outfit_count_is_688_l2804_280447


namespace NUMINAMATH_CALUDE_yellow_flowers_count_l2804_280457

/-- Represents a garden with yellow, green, and red flowers. -/
structure Garden where
  yellow : ℕ
  green : ℕ
  red : ℕ

/-- Proves that a garden with the given conditions has 12 yellow flowers. -/
theorem yellow_flowers_count (g : Garden) : 
  g.yellow + g.green + g.red = 78 → 
  g.red = 42 → 
  g.green = 2 * g.yellow → 
  g.yellow = 12 := by
  sorry

#check yellow_flowers_count

end NUMINAMATH_CALUDE_yellow_flowers_count_l2804_280457


namespace NUMINAMATH_CALUDE_function_inequality_l2804_280463

/-- A function satisfying the given condition -/
def SatisfiesCondition (f : ℝ → ℝ) : Prop :=
  ∀ x, x * (deriv (deriv f) x) + f x > 0

theorem function_inequality {f : ℝ → ℝ} (hf : Differentiable ℝ f) 
    (hf' : Differentiable ℝ (deriv f)) (hcond : SatisfiesCondition f) 
    {a b : ℝ} (ha : a > 0) (hb : b > 0) (hab : a > b) : 
    a * f a > b * f b := by
  sorry

end NUMINAMATH_CALUDE_function_inequality_l2804_280463


namespace NUMINAMATH_CALUDE_triangle_similarity_theorem_l2804_280425

-- Define the triangle XYZ
structure Triangle :=
  (X Y Z : ℝ × ℝ)

-- Define the line segment MN
structure LineSegment :=
  (M N : ℝ × ℝ)

-- Define the parallel relation
def parallel (l1 l2 : LineSegment) : Prop := sorry

-- Define the length of a line segment
def length (p1 p2 : ℝ × ℝ) : ℝ := sorry

-- Theorem statement
theorem triangle_similarity_theorem (XYZ : Triangle) (MN : LineSegment) :
  parallel MN (LineSegment.mk XYZ.X XYZ.Y) →
  length XYZ.X (MN.M) = 5 →
  length (MN.M) XYZ.Y = 8 →
  length (MN.N) XYZ.Z = 7 →
  length XYZ.X XYZ.Z = 18.2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_similarity_theorem_l2804_280425


namespace NUMINAMATH_CALUDE_squirrel_problem_l2804_280451

/-- Theorem: Given the conditions of the squirrel problem, prove the original number of squirrels on each tree. -/
theorem squirrel_problem (s b j : ℕ) : 
  s + b + j = 34 ∧ 
  b + 7 = j + s - 7 ∧ 
  b + 12 = 2 * j → 
  s = 13 ∧ b = 10 ∧ j = 11 := by
sorry

end NUMINAMATH_CALUDE_squirrel_problem_l2804_280451


namespace NUMINAMATH_CALUDE_inverse_variation_problem_l2804_280464

-- Define the inverse relationship between y and x^2
def inverse_relation (k : ℝ) (x y : ℝ) : Prop := y = k / (x^2)

-- Theorem statement
theorem inverse_variation_problem (k : ℝ) :
  (inverse_relation k 1 8) →
  (inverse_relation k 4 0.5) :=
by
  sorry

end NUMINAMATH_CALUDE_inverse_variation_problem_l2804_280464


namespace NUMINAMATH_CALUDE_max_xy_on_circle_l2804_280440

theorem max_xy_on_circle (x y : ℝ) (h : x^2 + y^2 = 4) : 
  ∃ (max : ℝ), (∀ a b : ℝ, a^2 + b^2 = 4 → a * b ≤ max) ∧ (∃ c d : ℝ, c^2 + d^2 = 4 ∧ c * d = max) ∧ max = 2 := by
  sorry

end NUMINAMATH_CALUDE_max_xy_on_circle_l2804_280440


namespace NUMINAMATH_CALUDE_polynomial_product_equality_l2804_280472

theorem polynomial_product_equality (a b c : ℝ) :
  a * (b - c)^3 + b * (c - a)^3 + c * (a - b)^3 = (a - b) * (b - c) * (c - a) * (a + b + c) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_product_equality_l2804_280472


namespace NUMINAMATH_CALUDE_range_of_a_l2804_280438

-- Define proposition p
def p (a : ℝ) : Prop :=
  ∃ x : ℝ, x ∈ Set.Icc (-1) 1 ∧ 2 * x^2 + a * x - a^2 = 0

-- Define proposition q
def q (a : ℝ) : Prop :=
  ∃! x : ℝ, x^2 + 2 * a * x + 2 * a ≤ 0

-- Theorem statement
theorem range_of_a (a : ℝ) : ¬(p a ∨ q a) ↔ a > 2 ∨ a < -2 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l2804_280438


namespace NUMINAMATH_CALUDE_journey_distance_is_70_l2804_280496

-- Define the journey parameters
def journey_time_at_40 : Real := 1.75
def journey_time_at_35 : Real := 2

-- Theorem statement
theorem journey_distance_is_70 :
  ∃ (distance : Real),
    distance = 40 * journey_time_at_40 ∧
    distance = 35 * journey_time_at_35 ∧
    distance = 70 := by
  sorry

end NUMINAMATH_CALUDE_journey_distance_is_70_l2804_280496


namespace NUMINAMATH_CALUDE_third_group_men_count_l2804_280488

/-- The work rate of one man -/
def man_rate : ℝ := sorry

/-- The work rate of one woman -/
def woman_rate : ℝ := sorry

/-- The number of men in the third group -/
def x : ℝ := sorry

theorem third_group_men_count : x = 4 :=
  by
  have h1 : 3 * man_rate + 8 * woman_rate = 6 * man_rate + 2 * woman_rate := by sorry
  have h2 : x * man_rate + 5 * woman_rate = 0.9285714285714286 * (6 * man_rate + 2 * woman_rate) := by sorry
  sorry

end NUMINAMATH_CALUDE_third_group_men_count_l2804_280488


namespace NUMINAMATH_CALUDE_complex_average_calculation_l2804_280458

def avg2 (a b : ℚ) : ℚ := (a + b) / 2

def avg4 (a b c d : ℚ) : ℚ := (a + b + c + d) / 4

theorem complex_average_calculation :
  avg4 (avg4 2 2 (-1) (avg2 1 3)) 7 (avg2 4 (5 - 2)) = 27 / 8 := by
  sorry

end NUMINAMATH_CALUDE_complex_average_calculation_l2804_280458


namespace NUMINAMATH_CALUDE_max_chickens_and_chicks_optimal_chicken_count_l2804_280449

/-- Represents the chicken coop problem -/
structure ChickenCoop where
  area : ℝ
  chicken_space : ℝ
  chick_space : ℝ
  chicken_feed : ℝ
  chick_feed : ℝ
  max_feed : ℝ

/-- Defines the conditions of the problem -/
def problem_conditions : ChickenCoop :=
  { area := 240
  , chicken_space := 4
  , chick_space := 2
  , chicken_feed := 160
  , chick_feed := 40
  , max_feed := 8000
  }

/-- Checks if a given number of chickens and chicks satisfies the constraints -/
def satisfies_constraints (coop : ChickenCoop) (chickens : ℕ) (chicks : ℕ) : Prop :=
  (chickens : ℝ) * coop.chicken_space + (chicks : ℝ) * coop.chick_space ≤ coop.area ∧
  (chickens : ℝ) * coop.chicken_feed + (chicks : ℝ) * coop.chick_feed ≤ coop.max_feed

/-- Theorem stating the maximum number of chickens and chicks -/
theorem max_chickens_and_chicks (coop : ChickenCoop := problem_conditions) :
  satisfies_constraints coop 40 40 ∧
  (∀ c : ℕ, c > 40 → ¬satisfies_constraints coop c 40) ∧
  satisfies_constraints coop 0 120 ∧
  (∀ k : ℕ, k > 120 → ¬satisfies_constraints coop 0 k) := by
  sorry

/-- Theorem stating that 40 chickens and 40 chicks is optimal when maximizing chickens -/
theorem optimal_chicken_count (coop : ChickenCoop := problem_conditions) :
  ∀ c k : ℕ, satisfies_constraints coop c k →
    c ≤ 40 ∧ (c = 40 → k ≤ 40) := by
  sorry

end NUMINAMATH_CALUDE_max_chickens_and_chicks_optimal_chicken_count_l2804_280449


namespace NUMINAMATH_CALUDE_isosceles_triangle_m_value_l2804_280417

/-- An isosceles triangle with side length 8 and other sides as roots of x^2 - 10x + m = 0 -/
structure IsoscelesTriangle where
  m : ℝ
  BC : ℝ
  AB_AC_eq : x^2 - 10*x + m = 0 → x = AB ∨ x = AC
  BC_eq : BC = 8
  isosceles : AB = AC

/-- The value of m in the isosceles triangle is either 25 or 16 -/
theorem isosceles_triangle_m_value (t : IsoscelesTriangle) : t.m = 25 ∨ t.m = 16 := by
  sorry

end NUMINAMATH_CALUDE_isosceles_triangle_m_value_l2804_280417


namespace NUMINAMATH_CALUDE_shielas_classmates_l2804_280466

theorem shielas_classmates (total_stars : ℕ) (stars_per_bottle : ℕ) (num_classmates : ℕ) : 
  total_stars = 45 → stars_per_bottle = 5 → num_classmates = total_stars / stars_per_bottle → num_classmates = 9 := by
  sorry

end NUMINAMATH_CALUDE_shielas_classmates_l2804_280466


namespace NUMINAMATH_CALUDE_unique_function_property_l2804_280429

theorem unique_function_property (f : ℝ → ℝ) 
  (h1 : ∀ x, f (-x) = -f x)
  (h2 : ∀ x, f (x + 1) = f x + 1)
  (h3 : ∀ x ≠ 0, f (1 / x) = f x / x^2) :
  ∀ x, f x = x := by
sorry

end NUMINAMATH_CALUDE_unique_function_property_l2804_280429


namespace NUMINAMATH_CALUDE_total_items_l2804_280434

theorem total_items (bread : ℕ) (milk : ℕ) (cookies : ℕ) 
  (h1 : bread = 58)
  (h2 : bread = milk + 18)
  (h3 : bread = cookies - 27) : 
  bread + milk + cookies = 183 := by
  sorry

end NUMINAMATH_CALUDE_total_items_l2804_280434


namespace NUMINAMATH_CALUDE_wrappers_found_at_park_l2804_280468

-- Define the variables
def bottle_caps_found : ℕ := 15
def total_wrappers : ℕ := 67
def total_bottle_caps : ℕ := 35
def wrapper_excess : ℕ := 32

-- Define the theorem
theorem wrappers_found_at_park :
  total_wrappers = total_bottle_caps + wrapper_excess →
  total_wrappers - (total_bottle_caps + wrapper_excess - bottle_caps_found) = 0 :=
by sorry

end NUMINAMATH_CALUDE_wrappers_found_at_park_l2804_280468


namespace NUMINAMATH_CALUDE_systematic_sampling_distribution_l2804_280423

/-- Represents a building in the summer camp -/
inductive Building
| A
| B
| C

/-- Calculates the number of students selected from each building using systematic sampling -/
def systematic_sampling (total_students : ℕ) (sample_size : ℕ) (start : ℕ) : Building → ℕ :=
  λ b =>
    match b with
    | Building.A => sorry
    | Building.B => sorry
    | Building.C => sorry

theorem systematic_sampling_distribution :
  let total_students := 400
  let sample_size := 50
  let start := 5
  (systematic_sampling total_students sample_size start Building.A = 25) ∧
  (systematic_sampling total_students sample_size start Building.B = 12) ∧
  (systematic_sampling total_students sample_size start Building.C = 13) :=
by sorry

end NUMINAMATH_CALUDE_systematic_sampling_distribution_l2804_280423


namespace NUMINAMATH_CALUDE_min_product_abc_l2804_280487

theorem min_product_abc (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (hsum : a + b + c = 2)
  (hbound1 : a ≤ 3*b ∧ a ≤ 3*c)
  (hbound2 : b ≤ 3*a ∧ b ≤ 3*c)
  (hbound3 : c ≤ 3*a ∧ c ≤ 3*b) :
  a * b * c ≥ 2/9 :=
sorry

end NUMINAMATH_CALUDE_min_product_abc_l2804_280487


namespace NUMINAMATH_CALUDE_derivative_f_at_1_l2804_280408

-- Define the function f
def f (x : ℝ) : ℝ := x^3 + x^2 - 2*x - 2

-- State the theorem
theorem derivative_f_at_1 :
  HasDerivAt f 3 1 := by sorry

end NUMINAMATH_CALUDE_derivative_f_at_1_l2804_280408


namespace NUMINAMATH_CALUDE_magic_square_solution_l2804_280495

/-- Represents a 3x3 magic square -/
structure MagicSquare :=
  (a11 a12 a13 a21 a22 a23 a31 a32 a33 : ℤ)

/-- The sum of any row, column, or diagonal in a magic square is the same -/
def is_magic (s : MagicSquare) : Prop :=
  let sum := s.a11 + s.a12 + s.a13
  sum = s.a21 + s.a22 + s.a23 ∧
  sum = s.a31 + s.a32 + s.a33 ∧
  sum = s.a11 + s.a21 + s.a31 ∧
  sum = s.a12 + s.a22 + s.a32 ∧
  sum = s.a13 + s.a23 + s.a33 ∧
  sum = s.a11 + s.a22 + s.a33 ∧
  sum = s.a13 + s.a22 + s.a31

theorem magic_square_solution :
  ∀ (s : MagicSquare),
    is_magic s →
    s.a11 = s.a11 ∧ s.a12 = 25 ∧ s.a13 = 75 ∧ s.a21 = 5 →
    s.a11 = 310 := by
  sorry

end NUMINAMATH_CALUDE_magic_square_solution_l2804_280495


namespace NUMINAMATH_CALUDE_mandy_reading_age_ratio_l2804_280411

/-- Represents Mandy's reading progression over time -/
structure ReadingProgression where
  starting_age : ℕ
  starting_pages : ℕ
  middle_age_multiplier : ℕ
  middle_pages_multiplier : ℕ
  later_years : ℕ
  later_pages_multiplier : ℕ
  current_pages_multiplier : ℕ
  current_pages : ℕ

/-- Theorem stating the ratio of Mandy's age when she started reading 40-page books to her starting age -/
theorem mandy_reading_age_ratio 
  (rp : ReadingProgression)
  (h1 : rp.starting_age = 6)
  (h2 : rp.starting_pages = 8)
  (h3 : rp.middle_pages_multiplier = 5)
  (h4 : rp.later_pages_multiplier = 3)
  (h5 : rp.later_years = 8)
  (h6 : rp.current_pages_multiplier = 4)
  (h7 : rp.current_pages = 480) :
  (rp.starting_age * rp.middle_pages_multiplier) / rp.starting_age = 5 := by
  sorry

#check mandy_reading_age_ratio

end NUMINAMATH_CALUDE_mandy_reading_age_ratio_l2804_280411


namespace NUMINAMATH_CALUDE_intersection_A_complement_B_l2804_280427

def U : Set Nat := {1, 2, 3, 4, 5, 6, 7}
def A : Set Nat := {2, 4, 6}
def B : Set Nat := {1, 3, 5, 7}

theorem intersection_A_complement_B : A ∩ (U \ B) = {2, 4, 6} := by
  sorry

end NUMINAMATH_CALUDE_intersection_A_complement_B_l2804_280427


namespace NUMINAMATH_CALUDE_nike_cost_l2804_280479

theorem nike_cost (total_goal : ℝ) (adidas_cost reebok_cost : ℝ) 
  (nike_sold adidas_sold reebok_sold : ℕ) (excess : ℝ)
  (h1 : total_goal = 1000)
  (h2 : adidas_cost = 45)
  (h3 : reebok_cost = 35)
  (h4 : nike_sold = 8)
  (h5 : adidas_sold = 6)
  (h6 : reebok_sold = 9)
  (h7 : excess = 65) :
  ∃ (nike_cost : ℝ), 
    nike_cost * nike_sold + adidas_cost * adidas_sold + reebok_cost * reebok_sold 
    = total_goal + excess ∧ nike_cost = 60 :=
by sorry

end NUMINAMATH_CALUDE_nike_cost_l2804_280479


namespace NUMINAMATH_CALUDE_remainder_three_power_twenty_mod_five_l2804_280460

theorem remainder_three_power_twenty_mod_five : 3^20 ≡ 1 [ZMOD 5] := by
  sorry

end NUMINAMATH_CALUDE_remainder_three_power_twenty_mod_five_l2804_280460


namespace NUMINAMATH_CALUDE_geometric_sequence_property_l2804_280445

/-- A geometric sequence -/
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

/-- Theorem: In a geometric sequence where a_3 = 1 and a_5 = 4, a_7 = 16 -/
theorem geometric_sequence_property (a : ℕ → ℝ) 
  (h_geom : geometric_sequence a) 
  (h_3 : a 3 = 1) 
  (h_5 : a 5 = 4) : 
  a 7 = 16 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_property_l2804_280445


namespace NUMINAMATH_CALUDE_smallest_sum_four_consecutive_composites_l2804_280442

def isComposite (n : ℕ) : Prop := n > 1 ∧ ∃ m : ℕ, m > 1 ∧ m < n ∧ n % m = 0

def fourConsecutiveComposites (n : ℕ) : Prop :=
  isComposite n ∧ isComposite (n + 1) ∧ isComposite (n + 2) ∧ isComposite (n + 3)

theorem smallest_sum_four_consecutive_composites :
  ∃ n : ℕ, fourConsecutiveComposites n ∧
    (∀ m : ℕ, fourConsecutiveComposites m → n ≤ m) ∧
    n + (n + 1) + (n + 2) + (n + 3) = 102 :=
sorry

end NUMINAMATH_CALUDE_smallest_sum_four_consecutive_composites_l2804_280442


namespace NUMINAMATH_CALUDE_tunneled_cube_surface_area_l2804_280486

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a cube with a tunnel drilled through it -/
structure TunneledCube where
  sideLength : ℝ
  tunnelStart : Point3D
  tunnelEnd : Point3D

/-- Calculates the surface area of a tunneled cube -/
def surfaceArea (cube : TunneledCube) : ℝ := sorry

/-- Checks if a number is square-free (not divisible by the square of any prime) -/
def isSquareFree (n : ℕ) : Prop := sorry

/-- Main theorem statement -/
theorem tunneled_cube_surface_area :
  ∃ (cube : TunneledCube) (u v w : ℕ),
    cube.sideLength = 10 ∧
    surfaceArea cube = u + v * Real.sqrt w ∧
    isSquareFree w ∧
    u + v + w = 472 := by sorry

end NUMINAMATH_CALUDE_tunneled_cube_surface_area_l2804_280486


namespace NUMINAMATH_CALUDE_g_at_neg_three_l2804_280467

-- Define the property of g
def satisfies_equation (g : ℚ → ℚ) : Prop :=
  ∀ x : ℚ, x ≠ 0 → 4 * g (1 / x) + 3 * g x / x = x^3

-- State the theorem
theorem g_at_neg_three (g : ℚ → ℚ) (h : satisfies_equation g) : g (-3) = -6565 / 189 := by
  sorry

end NUMINAMATH_CALUDE_g_at_neg_three_l2804_280467


namespace NUMINAMATH_CALUDE_function_symmetry_l2804_280422

theorem function_symmetry (f : ℝ → ℝ) (x : ℝ) : f (x - 1) = f (1 - x) := by
  sorry

end NUMINAMATH_CALUDE_function_symmetry_l2804_280422


namespace NUMINAMATH_CALUDE_i_power_sum_l2804_280428

-- Define the imaginary unit i
noncomputable def i : ℂ := Complex.I

-- Define the property of i
axiom i_squared : i^2 = -1

-- Define the periodicity of i
axiom i_period (n : ℕ) : i^(n + 4) = i^n

-- State the theorem
theorem i_power_sum : i^45 + i^123 = 0 := by sorry

end NUMINAMATH_CALUDE_i_power_sum_l2804_280428


namespace NUMINAMATH_CALUDE_greatest_K_inequality_l2804_280465

theorem greatest_K_inequality : 
  ∃ (K : ℝ), K = 16 ∧ 
  (∀ (u v w : ℝ), u > 0 → v > 0 → w > 0 → u^2 > 4*v*w → 
    (u^2 - 4*v*w)^2 > K*(2*v^2 - u*w)*(2*w^2 - u*v)) ∧
  (∀ (K' : ℝ), K' > K → 
    ∃ (u v w : ℝ), u > 0 ∧ v > 0 ∧ w > 0 ∧ u^2 > 4*v*w ∧ 
      (u^2 - 4*v*w)^2 ≤ K'*(2*v^2 - u*w)*(2*w^2 - u*v)) :=
by sorry

end NUMINAMATH_CALUDE_greatest_K_inequality_l2804_280465


namespace NUMINAMATH_CALUDE_inequality_equivalence_l2804_280419

theorem inequality_equivalence (x : ℝ) : 
  (2*x + 3)/(3*x + 5) > (4*x + 1)/(x + 4) ↔ -4 < x ∧ x < -5/3 := by
  sorry

end NUMINAMATH_CALUDE_inequality_equivalence_l2804_280419


namespace NUMINAMATH_CALUDE_travis_payment_l2804_280409

/-- Calculates the payment for Travis given the specified conditions --/
def calculate_payment (total_bowls : ℕ) (fixed_fee : ℕ) (safe_delivery_fee : ℕ) (penalty : ℕ) (lost_bowls : ℕ) (broken_bowls : ℕ) : ℕ :=
  let damaged_bowls := lost_bowls + broken_bowls
  let safe_bowls := total_bowls - damaged_bowls
  let safe_delivery_payment := safe_bowls * safe_delivery_fee
  let total_payment := safe_delivery_payment + fixed_fee
  let penalty_amount := damaged_bowls * penalty
  total_payment - penalty_amount

/-- Theorem stating that Travis should be paid $1825 given the specified conditions --/
theorem travis_payment :
  calculate_payment 638 100 3 4 12 15 = 1825 := by
  sorry

end NUMINAMATH_CALUDE_travis_payment_l2804_280409


namespace NUMINAMATH_CALUDE_base4_addition_theorem_l2804_280456

/-- Addition of numbers in base 4 -/
def base4_add (a b c d : ℕ) : ℕ := sorry

/-- Conversion from base 4 to decimal -/
def base4_to_decimal (n : ℕ) : ℕ := sorry

theorem base4_addition_theorem :
  base4_add (base4_to_decimal 2) (base4_to_decimal 23) (base4_to_decimal 132) (base4_to_decimal 1320) = base4_to_decimal 20200 := by
  sorry

end NUMINAMATH_CALUDE_base4_addition_theorem_l2804_280456


namespace NUMINAMATH_CALUDE_divisibility_by_five_l2804_280462

theorem divisibility_by_five (x y : ℕ) (h1 : 2 * x ^ 2 - 1 = y ^ 15) (h2 : x > 1) :
  5 ∣ x := by
  sorry

end NUMINAMATH_CALUDE_divisibility_by_five_l2804_280462


namespace NUMINAMATH_CALUDE_derivative_f_at_one_l2804_280473

noncomputable def f (x : ℝ) : ℝ := x^2 + Real.exp x

theorem derivative_f_at_one :
  deriv f 1 = 2 + Real.exp 1 := by sorry

end NUMINAMATH_CALUDE_derivative_f_at_one_l2804_280473


namespace NUMINAMATH_CALUDE_sum_factorials_mod_15_l2804_280478

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

def sum_factorials (n : ℕ) : ℕ := (List.range n).map factorial |>.sum

theorem sum_factorials_mod_15 : sum_factorials 50 % 15 = 3 := by sorry

end NUMINAMATH_CALUDE_sum_factorials_mod_15_l2804_280478


namespace NUMINAMATH_CALUDE_impossible_placement_l2804_280426

/-- A function representing the placement of integers on a 35x35 table -/
def TablePlacement := Fin 35 → Fin 35 → ℤ

/-- The property that all integers in the table are different -/
def AllDifferent (t : TablePlacement) : Prop :=
  ∀ i j k l, t i j = t k l → (i = k ∧ j = l)

/-- The property that adjacent cells differ by at most 18 -/
def AdjacentDifference (t : TablePlacement) : Prop :=
  ∀ i j k l, (i = k ∧ (j.val + 1 = l.val ∨ j.val = l.val + 1)) ∨
             (j = l ∧ (i.val + 1 = k.val ∨ i.val = k.val + 1)) →
             |t i j - t k l| ≤ 18

/-- The main theorem stating the impossibility of the required placement -/
theorem impossible_placement : ¬∃ t : TablePlacement, AllDifferent t ∧ AdjacentDifference t := by
  sorry

end NUMINAMATH_CALUDE_impossible_placement_l2804_280426


namespace NUMINAMATH_CALUDE_cherries_theorem_l2804_280484

def cherries_problem (initial_cherries : ℕ) (difference : ℕ) : ℕ :=
  initial_cherries - difference

theorem cherries_theorem (initial_cherries : ℕ) (difference : ℕ) 
  (h1 : initial_cherries = 16) (h2 : difference = 10) :
  cherries_problem initial_cherries difference = 6 := by
  sorry

end NUMINAMATH_CALUDE_cherries_theorem_l2804_280484


namespace NUMINAMATH_CALUDE_geometric_sequence_problem_l2804_280448

/-- Given a geometric sequence with common ratio 2 and all positive terms,
    if the product of the 4th and 12th terms is 64, then the 7th term is 4. -/
theorem geometric_sequence_problem (a : ℕ → ℝ) : 
  (∀ n, a (n + 1) = 2 * a n) →  -- Common ratio is 2
  (∀ n, a n > 0) →              -- All terms are positive
  a 4 * a 12 = 64 →             -- Product of 4th and 12th terms is 64
  a 7 = 4 := by                 -- The 7th term is 4
sorry

end NUMINAMATH_CALUDE_geometric_sequence_problem_l2804_280448


namespace NUMINAMATH_CALUDE_sin_2alpha_proof_l2804_280416

theorem sin_2alpha_proof (α : ℝ) (h : Real.cos (π / 4 - α) = 3 / 5) : 
  Real.sin (2 * α) = -7 / 25 := by
  sorry

end NUMINAMATH_CALUDE_sin_2alpha_proof_l2804_280416


namespace NUMINAMATH_CALUDE_area_equals_perimeter_count_l2804_280435

-- Define a rectangle with integer sides
structure Rectangle where
  a : ℕ
  b : ℕ

-- Define a right triangle with integer sides
structure RightTriangle where
  a : ℕ
  b : ℕ

-- Function to check if a rectangle's area equals its perimeter
def Rectangle.areaEqualsPerimeter (r : Rectangle) : Prop :=
  r.a * r.b = 2 * (r.a + r.b)

-- Function to check if a right triangle's area equals its perimeter
def RightTriangle.areaEqualsPerimeter (t : RightTriangle) : Prop :=
  t.a * t.b / 2 = t.a + t.b + Int.sqrt (t.a^2 + t.b^2)

-- The main theorem
theorem area_equals_perimeter_count :
  (∃! (r₁ r₂ : Rectangle), r₁ ≠ r₂ ∧ r₁.areaEqualsPerimeter ∧ r₂.areaEqualsPerimeter) ∧
  (∃! (t₁ t₂ : RightTriangle), t₁ ≠ t₂ ∧ t₁.areaEqualsPerimeter ∧ t₂.areaEqualsPerimeter) :=
sorry

end NUMINAMATH_CALUDE_area_equals_perimeter_count_l2804_280435


namespace NUMINAMATH_CALUDE_exists_common_point_l2804_280455

/-- Represents a rectangular map with a scale factor -/
structure Map where
  scale : ℝ
  width : ℝ
  height : ℝ

/-- Represents a point on a map -/
structure MapPoint where
  x : ℝ
  y : ℝ

/-- Theorem stating that there exists a common point on two maps of different scales -/
theorem exists_common_point (map1 map2 : Map) (h_scale : map2.scale = 5 * map1.scale) :
  ∃ (p1 : MapPoint) (p2 : MapPoint),
    p1.x / map1.width = p2.x / map2.width ∧
    p1.y / map1.height = p2.y / map2.height :=
sorry

end NUMINAMATH_CALUDE_exists_common_point_l2804_280455


namespace NUMINAMATH_CALUDE_geometric_progression_and_quadratic_vertex_l2804_280415

/-- Given a, b, c, d in geometric progression and (b,c) is the vertex of y=x^2-2x+3, prove a+d = 9/2 -/
theorem geometric_progression_and_quadratic_vertex (a b c d : ℝ) : 
  (∃ (q : ℝ), q ≠ 0 ∧ b = a * q ∧ c = b * q ∧ d = c * q) →  -- geometric progression
  (∀ x : ℝ, x^2 - 2*x + 3 = (x - b)^2 + c) →               -- vertex form of quadratic
  a + d = 9/2 := by
sorry

end NUMINAMATH_CALUDE_geometric_progression_and_quadratic_vertex_l2804_280415


namespace NUMINAMATH_CALUDE_solution_set_for_a_4_range_of_a_for_all_x_geq_4_l2804_280483

def f (a : ℝ) (x : ℝ) : ℝ := |x - 1| + |x - a|

theorem solution_set_for_a_4 :
  {x : ℝ | f 4 x ≥ 5} = {x : ℝ | x ≤ 0 ∨ x ≥ 5} := by sorry

theorem range_of_a_for_all_x_geq_4 :
  (∀ x : ℝ, f a x ≥ 4) → (a ≤ -3 ∨ a ≥ 5) := by sorry

end NUMINAMATH_CALUDE_solution_set_for_a_4_range_of_a_for_all_x_geq_4_l2804_280483


namespace NUMINAMATH_CALUDE_initial_investment_rate_is_five_percent_l2804_280420

/-- Proves that given specific investment conditions, the initial investment rate is 5% --/
theorem initial_investment_rate_is_five_percent
  (initial_investment : ℝ)
  (additional_investment : ℝ)
  (additional_rate : ℝ)
  (total_income_rate : ℝ)
  (h1 : initial_investment = 2800)
  (h2 : additional_investment = 1400)
  (h3 : additional_rate = 0.08)
  (h4 : total_income_rate = 0.06)
  (h5 : initial_investment * x + additional_investment * additional_rate = 
        (initial_investment + additional_investment) * total_income_rate) :
  x = 0.05 := by
  sorry

end NUMINAMATH_CALUDE_initial_investment_rate_is_five_percent_l2804_280420


namespace NUMINAMATH_CALUDE_first_catchup_theorem_second_catchup_theorem_l2804_280441

/-- Represents a circular track with two runners -/
structure Track :=
  (length : ℝ)
  (speed_fast : ℝ)
  (speed_slow : ℝ)

/-- Calculates the distance covered by each runner at first catch-up -/
def first_catchup (track : Track) : ℝ × ℝ := sorry

/-- Calculates the number of laps completed by each runner at second catch-up -/
def second_catchup (track : Track) : ℕ × ℕ := sorry

/-- Theorem for the first catch-up distances -/
theorem first_catchup_theorem (track : Track) 
  (h1 : track.length = 400)
  (h2 : track.speed_fast = 7)
  (h3 : track.speed_slow = 5) :
  first_catchup track = (1400, 1000) := by sorry

/-- Theorem for the second catch-up laps -/
theorem second_catchup_theorem (track : Track)
  (h1 : track.length = 400)
  (h2 : track.speed_fast = 7)
  (h3 : track.speed_slow = 5) :
  second_catchup track = (7, 5) := by sorry

end NUMINAMATH_CALUDE_first_catchup_theorem_second_catchup_theorem_l2804_280441


namespace NUMINAMATH_CALUDE_cos_angle_between_vectors_l2804_280431

def a : ℝ × ℝ := (3, -1)
def b : ℝ × ℝ := (2, 0)

theorem cos_angle_between_vectors :
  let θ := Real.arccos ((a.1 * b.1 + a.2 * b.2) / (Real.sqrt (a.1^2 + a.2^2) * Real.sqrt (b.1^2 + b.2^2)))
  (θ.cos : ℝ) = 3 * Real.sqrt 10 / 10 := by
  sorry

end NUMINAMATH_CALUDE_cos_angle_between_vectors_l2804_280431


namespace NUMINAMATH_CALUDE_smallest_brownie_pan_dimension_l2804_280459

def is_valid_brownie_pan (m n : ℕ) : Prop :=
  m > 0 ∧ n > 0 ∧ (m - 2) * (n - 2) = 2 * (2 * m + 2 * n - 4)

def smallest_dimension : ℕ := 12

theorem smallest_brownie_pan_dimension :
  (is_valid_brownie_pan smallest_dimension smallest_dimension) ∧
  (∀ k : ℕ, k < smallest_dimension → ¬(is_valid_brownie_pan k k) ∧ ¬(∃ l : ℕ, is_valid_brownie_pan k l ∨ is_valid_brownie_pan l k)) :=
by sorry

end NUMINAMATH_CALUDE_smallest_brownie_pan_dimension_l2804_280459


namespace NUMINAMATH_CALUDE_intersection_equals_Q_intersection_empty_l2804_280446

-- Define the sets P and Q
def P : Set ℝ := {x | 2 * x^2 - 5 * x - 3 < 0}
def Q (a : ℝ) : Set ℝ := {x | a ≤ x ∧ x ≤ a + 1}

-- Theorem for the first part
theorem intersection_equals_Q (a : ℝ) :
  (P ∩ Q a) = Q a ↔ a ∈ Set.Ioo (-1/2) 2 :=
sorry

-- Theorem for the second part
theorem intersection_empty (a : ℝ) :
  (P ∩ Q a) = ∅ ↔ a ∈ Set.Iic (-3/2) ∪ Set.Ici 3 :=
sorry

end NUMINAMATH_CALUDE_intersection_equals_Q_intersection_empty_l2804_280446


namespace NUMINAMATH_CALUDE_dvd_book_theorem_l2804_280410

/-- Represents a DVD book with a given capacity and current number of DVDs -/
structure DVDBook where
  capacity : ℕ
  current : ℕ

/-- Calculates the number of additional DVDs that can be put in the book -/
def additionalDVDs (book : DVDBook) : ℕ :=
  book.capacity - book.current

theorem dvd_book_theorem (book : DVDBook) 
  (h1 : book.capacity = 126) 
  (h2 : book.current = 81) : 
  additionalDVDs book = 45 := by
  sorry

end NUMINAMATH_CALUDE_dvd_book_theorem_l2804_280410


namespace NUMINAMATH_CALUDE_cos_2alpha_problem_l2804_280491

theorem cos_2alpha_problem (α : Real) (h1 : 0 < α ∧ α < π / 2) 
  (h2 : Real.sin α - Real.cos α = Real.sqrt 10 / 5) : 
  Real.cos (2 * α) = -4/5 := by
  sorry

end NUMINAMATH_CALUDE_cos_2alpha_problem_l2804_280491


namespace NUMINAMATH_CALUDE_exactly_one_project_not_selected_l2804_280413

/-- The number of employees and projects -/
def n : ℕ := 4

/-- The probability of exactly one project not being selected -/
def probability : ℚ := 9/16

/-- Theorem stating the probability of exactly one project not being selected -/
theorem exactly_one_project_not_selected :
  (n : ℚ)^n * probability = (n.choose 2) * n! :=
sorry

end NUMINAMATH_CALUDE_exactly_one_project_not_selected_l2804_280413


namespace NUMINAMATH_CALUDE_concrete_wall_width_l2804_280432

theorem concrete_wall_width
  (r : ℝ)  -- radius of the pool
  (w : ℝ)  -- width of the concrete wall
  (h1 : r = 20)  -- radius of the pool is 20 ft
  (h2 : π * ((r + w)^2 - r^2) = (11/25) * (π * r^2))  -- area of wall is 11/25 of pool area
  : w = 4 := by
  sorry

end NUMINAMATH_CALUDE_concrete_wall_width_l2804_280432


namespace NUMINAMATH_CALUDE_max_length_sum_l2804_280430

/-- The length of an integer is the number of positive prime factors, not necessarily distinct, whose product is equal to the integer. -/
def length (k : ℕ) : ℕ := sorry

/-- A number is prime if it has exactly two factors -/
def isPrime (p : ℕ) : Prop := sorry

theorem max_length_sum :
  ∀ x y z : ℕ,
  x > 1 → y > 1 → z > 1 →
  (∃ p q : ℕ, isPrime p ∧ isPrime q ∧ p ≠ q ∧ x = p * q) →
  (∃ p q r : ℕ, isPrime p ∧ isPrime q ∧ isPrime r ∧ p ≠ q ∧ q ≠ r ∧ p ≠ r ∧ y = p * q * r) →
  x + 3 * y + 5 * z < 5000 →
  length x + length y + length z ≤ 14 :=
sorry

end NUMINAMATH_CALUDE_max_length_sum_l2804_280430


namespace NUMINAMATH_CALUDE_triangle_identity_l2804_280453

/-- Operation △ between ordered pairs of real numbers -/
def triangle (a b c d : ℝ) : ℝ × ℝ := (a*c + b*d, a*d + b*c)

/-- Theorem: If (u,v) △ (x,y) = (u,v) for all real u and v, then (x,y) = (1,0) -/
theorem triangle_identity (x y : ℝ) :
  (∀ u v : ℝ, triangle u v x y = (u, v)) → (x = 1 ∧ y = 0) := by
  sorry

end NUMINAMATH_CALUDE_triangle_identity_l2804_280453


namespace NUMINAMATH_CALUDE_point_transformation_l2804_280414

/-- Reflect a point (x, y) across the line y = x -/
def reflect_across_y_eq_x (p : ℝ × ℝ) : ℝ × ℝ :=
  (p.2, p.1)

/-- Rotate a point (x, y) by 180° around a center (h, k) -/
def rotate_180_around (p : ℝ × ℝ) (center : ℝ × ℝ) : ℝ × ℝ :=
  (2 * center.1 - p.1, 2 * center.2 - p.2)

/-- The main theorem -/
theorem point_transformation (a b : ℝ) :
  let Q : ℝ × ℝ := (a, b)
  let reflected := reflect_across_y_eq_x Q
  let rotated := rotate_180_around reflected (1, 5)
  rotated = (-8, 2) → a - b = -2 := by
  sorry

end NUMINAMATH_CALUDE_point_transformation_l2804_280414


namespace NUMINAMATH_CALUDE_bucket_capacity_reduction_l2804_280499

theorem bucket_capacity_reduction (original_buckets : ℕ) (capacity_ratio : ℚ) : 
  original_buckets = 42 →
  capacity_ratio = 2 / 5 →
  (original_buckets : ℚ) / capacity_ratio = 105 := by sorry

end NUMINAMATH_CALUDE_bucket_capacity_reduction_l2804_280499


namespace NUMINAMATH_CALUDE_chocolate_bars_count_l2804_280406

/-- The number of small boxes in the large box -/
def num_small_boxes : ℕ := 18

/-- The number of chocolate bars in each small box -/
def bars_per_small_box : ℕ := 28

/-- The total number of chocolate bars in the large box -/
def total_chocolate_bars : ℕ := num_small_boxes * bars_per_small_box

theorem chocolate_bars_count : total_chocolate_bars = 504 := by
  sorry

end NUMINAMATH_CALUDE_chocolate_bars_count_l2804_280406


namespace NUMINAMATH_CALUDE_complement_B_A_when_m_2_range_of_m_for_necessary_not_sufficient_l2804_280482

def A : Set ℝ := {x | 4 < x ∧ x ≤ 8}
def B (m : ℝ) : Set ℝ := {x | 5 - m^2 ≤ x ∧ x ≤ 5 + m^2}

theorem complement_B_A_when_m_2 :
  {x : ℝ | 1 ≤ x ∧ x ≤ 4 ∨ 8 < x ∧ x ≤ 9} = (B 2) \ A := by sorry

theorem range_of_m_for_necessary_not_sufficient :
  {m : ℝ | A ⊆ B m ∧ A ≠ B m} = {m : ℝ | -1 < m ∧ m < 1} := by sorry

end NUMINAMATH_CALUDE_complement_B_A_when_m_2_range_of_m_for_necessary_not_sufficient_l2804_280482


namespace NUMINAMATH_CALUDE_jack_weight_l2804_280401

theorem jack_weight (total_weight sam_weight jack_weight : ℕ) : 
  total_weight = 96 →
  jack_weight = sam_weight + 8 →
  total_weight = sam_weight + jack_weight →
  jack_weight = 52 := by
sorry

end NUMINAMATH_CALUDE_jack_weight_l2804_280401


namespace NUMINAMATH_CALUDE_principal_amount_l2804_280412

/-- Proves that given the specified conditions, the principal amount is 1300 --/
theorem principal_amount (P : ℝ) : 
  P * ((1 + 0.1)^2 - 1) - P * (0.1 * 2) = 13 → P = 1300 := by
  sorry

end NUMINAMATH_CALUDE_principal_amount_l2804_280412


namespace NUMINAMATH_CALUDE_intersection_point_unique_l2804_280439

/-- The intersection point of two lines -/
def intersection_point : ℚ × ℚ := (54/5, -26/5)

/-- First line equation -/
def line1 (x y : ℚ) : Prop := 3*y = -2*x + 6

/-- Second line equation -/
def line2 (x y : ℚ) : Prop := 7*y = -3*x - 4

theorem intersection_point_unique :
  (∃! p : ℚ × ℚ, line1 p.1 p.2 ∧ line2 p.1 p.2) ∧
  (line1 intersection_point.1 intersection_point.2) ∧
  (line2 intersection_point.1 intersection_point.2) :=
sorry

end NUMINAMATH_CALUDE_intersection_point_unique_l2804_280439


namespace NUMINAMATH_CALUDE_pastries_cakes_difference_l2804_280461

theorem pastries_cakes_difference (pastries_sold : ℕ) (cakes_sold : ℕ) 
  (h1 : pastries_sold = 154) (h2 : cakes_sold = 78) : 
  pastries_sold - cakes_sold = 76 := by
  sorry

end NUMINAMATH_CALUDE_pastries_cakes_difference_l2804_280461


namespace NUMINAMATH_CALUDE_trigonometric_calculations_l2804_280498

theorem trigonometric_calculations :
  (((Real.pi - 2) ^ 0 - |1 - Real.tan (60 * Real.pi / 180)| - (1/2)⁻¹ + 6 / Real.sqrt 3) = Real.sqrt 3) ∧
  ((Real.sin (45 * Real.pi / 180) - Real.cos (30 * Real.pi / 180) * Real.tan (60 * Real.pi / 180)) = (Real.sqrt 2 - 3) / 2) :=
by sorry

end NUMINAMATH_CALUDE_trigonometric_calculations_l2804_280498


namespace NUMINAMATH_CALUDE_arithmetic_sequence_special_case_l2804_280470

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- The theorem statement -/
theorem arithmetic_sequence_special_case (a : ℕ → ℝ) :
  arithmetic_sequence a →
  (a 4)^2 - 4*(a 4) - 1 = 0 →
  (a 8)^2 - 4*(a 8) - 1 = 0 →
  a 6 = 2 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_special_case_l2804_280470


namespace NUMINAMATH_CALUDE_price_reduction_equation_correct_l2804_280433

/-- Represents the price reduction scenario -/
structure PriceReduction where
  initial_price : ℝ
  final_price : ℝ
  num_reductions : ℕ
  
/-- The price reduction equation is correct for the given scenario -/
theorem price_reduction_equation_correct (pr : PriceReduction) 
  (h1 : pr.initial_price = 560)
  (h2 : pr.final_price = 315)
  (h3 : pr.num_reductions = 2) :
  ∃ x : ℝ, pr.initial_price * (1 - x)^pr.num_reductions = pr.final_price :=
sorry

end NUMINAMATH_CALUDE_price_reduction_equation_correct_l2804_280433


namespace NUMINAMATH_CALUDE_blackboard_division_l2804_280443

theorem blackboard_division : (96 : ℕ) / 8 = 12 := by
  sorry

end NUMINAMATH_CALUDE_blackboard_division_l2804_280443


namespace NUMINAMATH_CALUDE_arithmetic_mean_of_special_set_l2804_280418

theorem arithmetic_mean_of_special_set : 
  let S : Finset ℕ := Finset.range 9
  let special_number (n : ℕ) : ℕ := n * ((10^n - 1) / 9)
  let sum_of_set : ℕ := S.sum special_number
  sum_of_set / 9 = 123456790 := by sorry

end NUMINAMATH_CALUDE_arithmetic_mean_of_special_set_l2804_280418


namespace NUMINAMATH_CALUDE_identity_proof_l2804_280436

theorem identity_proof (a b c : ℝ) 
  (h_pos_a : a > 0) (h_pos_b : b > 0) (h_pos_c : c > 0)
  (h_sum : a + b + c = 1) : 
  Real.sqrt ((a + b * c) * (b + c * a) / (c + a * b)) + 
  Real.sqrt ((b + c * a) * (c + a * b) / (a + b * c)) + 
  Real.sqrt ((c + a * b) * (a + b * c) / (b + c * a)) = 2 := by
  sorry

end NUMINAMATH_CALUDE_identity_proof_l2804_280436


namespace NUMINAMATH_CALUDE_equation_solution_l2804_280444

theorem equation_solution : ∃ x : ℝ, 3 * x - 6 = |(-23 + 5)| ∧ x = 8 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2804_280444


namespace NUMINAMATH_CALUDE_arithmetic_geometric_ratio_l2804_280492

/-- An arithmetic sequence with common difference d -/
def arithmetic_sequence (a : ℕ → ℚ) (d : ℚ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n + d

/-- Three terms form a geometric sequence -/
def geometric_sequence (x y z : ℚ) : Prop :=
  y * y = x * z

/-- The common ratio of a geometric sequence -/
def common_ratio (x y : ℚ) : ℚ :=
  y / x

theorem arithmetic_geometric_ratio 
  (a : ℕ → ℚ) (d : ℚ) 
  (h_arith : arithmetic_sequence a d)
  (h_geom : geometric_sequence (a 1) (a 3) (a 4)) :
  (common_ratio (a 1) (a 3) = 1/2) ∨ (common_ratio (a 1) (a 3) = 1) :=
sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_ratio_l2804_280492


namespace NUMINAMATH_CALUDE_zacks_marbles_l2804_280452

theorem zacks_marbles (n : ℕ) : 
  (∃ k : ℕ, n = 3 * k + 5) → 
  (n = 3 * 20 + 5) → 
  n = 65 := by
sorry

end NUMINAMATH_CALUDE_zacks_marbles_l2804_280452


namespace NUMINAMATH_CALUDE_mango_distribution_l2804_280476

/-- Given 560 mangoes, if half are sold and the remainder is distributed evenly among 8 neighbors,
    each neighbor receives 35 mangoes. -/
theorem mango_distribution (total_mangoes : ℕ) (neighbors : ℕ) 
    (h1 : total_mangoes = 560) 
    (h2 : neighbors = 8) : 
  (total_mangoes / 2) / neighbors = 35 := by
  sorry

end NUMINAMATH_CALUDE_mango_distribution_l2804_280476


namespace NUMINAMATH_CALUDE_seventh_root_unity_product_l2804_280405

theorem seventh_root_unity_product (r : ℂ) (h1 : r^7 = 1) (h2 : r ≠ 1) :
  (r - 1) * (r^2 - 1) * (r^3 - 1) * (r^4 - 1) * (r^5 - 1) * (r^6 - 1) = 7 := by
  sorry

end NUMINAMATH_CALUDE_seventh_root_unity_product_l2804_280405


namespace NUMINAMATH_CALUDE_tangency_point_difference_l2804_280477

/-- A quadrilateral inscribed in a circle with an inscribed circle --/
structure InscribedQuadrilateral where
  -- Side lengths
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ
  -- Conditions
  positive : 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d
  cyclic : True  -- Represents that the quadrilateral is cyclic
  inscribed : True  -- Represents that there's an inscribed circle

/-- The specific quadrilateral from the problem --/
def specificQuad : InscribedQuadrilateral where
  a := 60
  b := 110
  c := 140
  d := 90
  positive := by simp
  cyclic := True.intro
  inscribed := True.intro

/-- The point of tangency divides the side of length 140 into m and n --/
def tangencyPoint (q : InscribedQuadrilateral) : ℝ × ℝ :=
  sorry

/-- The theorem to prove --/
theorem tangency_point_difference (q : InscribedQuadrilateral) 
  (h : q = specificQuad) : 
  let (m, n) := tangencyPoint q
  |m - n| = 120 := by
  sorry

end NUMINAMATH_CALUDE_tangency_point_difference_l2804_280477
