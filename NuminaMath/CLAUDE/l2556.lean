import Mathlib

namespace NUMINAMATH_CALUDE_beetle_distance_theorem_l2556_255682

def beetle_crawl (start : ℤ) (stop1 : ℤ) (stop2 : ℤ) : ℕ :=
  (Int.natAbs (stop1 - start)) + (Int.natAbs (stop2 - stop1))

theorem beetle_distance_theorem :
  beetle_crawl 3 (-5) 7 = 20 := by
  sorry

end NUMINAMATH_CALUDE_beetle_distance_theorem_l2556_255682


namespace NUMINAMATH_CALUDE_factory_B_is_better_l2556_255646

/-- Represents a chicken leg factory --/
structure ChickenFactory where
  name : String
  mean : ℝ
  median : ℝ
  mode : ℝ
  variance : ℝ

/-- Determines if a factory is better based on its statistics --/
def isBetterFactory (f1 f2 : ChickenFactory) : Prop :=
  f1.mean = f2.mean ∧
  f1.variance < f2.variance ∧
  f1.median = f1.mean ∧
  f1.mode = f1.mean ∧
  (f2.median ≠ f2.mean ∨ f2.mode ≠ f2.mean)

/-- Factory A data --/
def factoryA : ChickenFactory :=
  { name := "A"
    mean := 75
    median := 74.5
    mode := 74
    variance := 3.4 }

/-- Factory B data --/
def factoryB : ChickenFactory :=
  { name := "B"
    mean := 75
    median := 75
    mode := 75
    variance := 2 }

/-- Theorem stating that Factory B is better than Factory A --/
theorem factory_B_is_better : isBetterFactory factoryB factoryA := by
  sorry

#check factory_B_is_better

end NUMINAMATH_CALUDE_factory_B_is_better_l2556_255646


namespace NUMINAMATH_CALUDE_sum_of_union_equals_31_l2556_255665

def A : Finset ℕ := {2, 0, 1, 8}

def B : Finset ℕ := Finset.image (· * 2) A

theorem sum_of_union_equals_31 : (A ∪ B).sum id = 31 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_union_equals_31_l2556_255665


namespace NUMINAMATH_CALUDE_add_4500_seconds_to_10_45_00_l2556_255657

/-- Represents time in hours, minutes, and seconds -/
structure Time where
  hours : Nat
  minutes : Nat
  seconds : Nat

/-- Adds seconds to a given time -/
def addSeconds (t : Time) (s : Nat) : Time :=
  sorry

/-- The starting time 10:45:00 -/
def startTime : Time :=
  { hours := 10, minutes := 45, seconds := 0 }

/-- The number of seconds to add -/
def secondsToAdd : Nat := 4500

/-- The expected end time 12:00:00 -/
def endTime : Time :=
  { hours := 12, minutes := 0, seconds := 0 }

theorem add_4500_seconds_to_10_45_00 :
  addSeconds startTime secondsToAdd = endTime := by
  sorry

end NUMINAMATH_CALUDE_add_4500_seconds_to_10_45_00_l2556_255657


namespace NUMINAMATH_CALUDE_triangle_longest_side_range_l2556_255660

/-- Given a triangle with perimeter 12 and b as the longest side, prove the range of b -/
theorem triangle_longest_side_range (a b c : ℝ) : 
  a > 0 → b > 0 → c > 0 →  -- positive side lengths
  a + b + c = 12 →         -- perimeter is 12
  b ≥ a ∧ b ≥ c →          -- b is the longest side
  4 < b ∧ b < 6 :=         -- range of b
by sorry

end NUMINAMATH_CALUDE_triangle_longest_side_range_l2556_255660


namespace NUMINAMATH_CALUDE_store_a_cheaper_for_15_boxes_store_a_cheaper_for_x_boxes_l2556_255612

/-- Represents the cost of purchasing table tennis equipment from a store. -/
structure StoreCost where
  ballCost : ℝ  -- Cost per box of balls
  racketCost : ℝ  -- Cost per racket
  numRackets : ℕ  -- Number of rackets needed
  discount : ℝ  -- Discount factor (1 for no discount, 0.9 for 10% discount)
  freeBoxes : ℕ  -- Number of free boxes of balls

/-- Calculates the total cost for a given number of ball boxes. -/
def totalCost (s : StoreCost) (x : ℕ) : ℝ :=
  s.discount * (s.ballCost * (x - s.freeBoxes) + s.racketCost * s.numRackets)

/-- Store A's cost structure -/
def storeA : StoreCost :=
  { ballCost := 5
  , racketCost := 30
  , numRackets := 5
  , discount := 1
  , freeBoxes := 5 }

/-- Store B's cost structure -/
def storeB : StoreCost :=
  { ballCost := 5
  , racketCost := 30
  , numRackets := 5
  , discount := 0.9
  , freeBoxes := 0 }

/-- Theorem stating that Store A is cheaper than or equal to Store B for 15 boxes of balls -/
theorem store_a_cheaper_for_15_boxes :
  totalCost storeA 15 ≤ totalCost storeB 15 :=
by
  sorry

/-- Theorem stating that Store A is cheaper than or equal to Store B for any number of boxes ≥ 5 -/
theorem store_a_cheaper_for_x_boxes (x : ℕ) (h : x ≥ 5) :
  totalCost storeA x ≤ totalCost storeB x :=
by
  sorry

end NUMINAMATH_CALUDE_store_a_cheaper_for_15_boxes_store_a_cheaper_for_x_boxes_l2556_255612


namespace NUMINAMATH_CALUDE_rectangular_field_diagonal_shortcut_l2556_255642

theorem rectangular_field_diagonal_shortcut (x y : ℝ) (hxy : 0 < x ∧ x < y) :
  x + y - Real.sqrt (x^2 + y^2) = (1/3) * y →
  x / y = 5/12 := by
sorry

end NUMINAMATH_CALUDE_rectangular_field_diagonal_shortcut_l2556_255642


namespace NUMINAMATH_CALUDE_sqrt_eight_and_nine_sixteenths_l2556_255616

theorem sqrt_eight_and_nine_sixteenths (x : ℝ) : 
  x = Real.sqrt (8 + 9/16) → x = Real.sqrt 137 / 4 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_eight_and_nine_sixteenths_l2556_255616


namespace NUMINAMATH_CALUDE_circle_radius_with_tangent_parabola_l2556_255635

theorem circle_radius_with_tangent_parabola :
  ∀ r : ℝ,
  (∃ x : ℝ, x^2 + r = x) →  -- Parabola y = x^2 + r is tangent to line y = x
  (∀ x : ℝ, x^2 + r ≥ x) →  -- Parabola lies above or on the line
  r = (1 : ℝ) / 4 :=
by sorry

end NUMINAMATH_CALUDE_circle_radius_with_tangent_parabola_l2556_255635


namespace NUMINAMATH_CALUDE_function_properties_l2556_255662

def f (a b x : ℝ) : ℝ := a * x^2 + b * x - 2 * b

theorem function_properties :
  ∀ (a b : ℝ),
  (a > 0 ∧ a = b → {x : ℝ | f a b x < 0} = Set.Ioo (-2) 1) ∧
  (a = 1 ∧ (∀ x < 2, f a b x ≥ 1) → b ≤ 2 * Real.sqrt 3 - 4) ∧
  (|f a b (-1)| ≤ 1 ∧ |f a b 1| ≤ 3 → 5/3 ≤ |a| + |b + 2| ∧ |a| + |b + 2| ≤ 9) :=
by sorry

end NUMINAMATH_CALUDE_function_properties_l2556_255662


namespace NUMINAMATH_CALUDE_solution_difference_l2556_255666

theorem solution_difference (r s : ℝ) : 
  ((5 * r - 20) / (r^2 + 3*r - 18) = r + 3) →
  ((5 * s - 20) / (s^2 + 3*s - 18) = s + 3) →
  (r ≠ s) →
  (r > s) →
  (r - s = Real.sqrt 29) :=
by sorry

end NUMINAMATH_CALUDE_solution_difference_l2556_255666


namespace NUMINAMATH_CALUDE_m_range_condition_l2556_255654

def A : Set ℝ := Set.Ioo (-2) 2
def B (m : ℝ) : Set ℝ := Set.Ici (m - 1)

theorem m_range_condition (m : ℝ) : A ⊆ B m ↔ m ≤ -1 := by
  sorry

end NUMINAMATH_CALUDE_m_range_condition_l2556_255654


namespace NUMINAMATH_CALUDE_regular_ticket_cost_l2556_255623

theorem regular_ticket_cost (total_tickets : ℕ) (senior_ticket_cost : ℕ) (total_sales : ℕ) (regular_tickets_sold : ℕ) :
  total_tickets = 65 →
  senior_ticket_cost = 10 →
  total_sales = 855 →
  regular_tickets_sold = 41 →
  ∃ (regular_ticket_cost : ℕ),
    regular_ticket_cost * regular_tickets_sold + senior_ticket_cost * (total_tickets - regular_tickets_sold) = total_sales ∧
    regular_ticket_cost = 15 :=
by
  sorry

end NUMINAMATH_CALUDE_regular_ticket_cost_l2556_255623


namespace NUMINAMATH_CALUDE_quadratic_roots_properties_l2556_255641

-- Define the quadratic equation and its roots
def quadratic_eq (p q x : ℝ) := x^2 + p*x + q = 0

theorem quadratic_roots_properties (p q : ℝ) (x₁ x₂ : ℝ) 
  (h₁ : quadratic_eq p q x₁) (h₂ : quadratic_eq p q x₂) (h₃ : x₁ ≠ x₂) :
  (1/x₁ + 1/x₂ = -p/q) ∧
  (1/x₁^2 + 1/x₂^2 = (p^2 - 2*q)/q^2) ∧
  (x₁^3 + x₂^3 = -p^3 + 3*p*q) ∧
  (1/(x₁ + p)^2 + 1/(x₂ + p)^2 = (p^2 - 2*q)/q^2) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_roots_properties_l2556_255641


namespace NUMINAMATH_CALUDE_wall_width_is_100cm_l2556_255610

/-- Represents the dimensions of a brick in centimeters -/
structure BrickDimensions where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Represents the dimensions of a wall in centimeters -/
structure WallDimensions where
  length : ℝ
  width : ℝ
  thickness : ℝ

/-- Calculates the volume of a brick given its dimensions -/
def brickVolume (d : BrickDimensions) : ℝ :=
  d.length * d.width * d.height

/-- Calculates the volume of a wall given its dimensions -/
def wallVolume (d : WallDimensions) : ℝ :=
  d.length * d.width * d.thickness

/-- Theorem stating that the width of the wall is 100 cm -/
theorem wall_width_is_100cm
  (brick : BrickDimensions)
  (wall : WallDimensions)
  (h1 : brick.length = 25)
  (h2 : brick.width = 11)
  (h3 : brick.height = 6)
  (h4 : wall.length = 800)
  (h5 : wall.thickness = 5)
  (h6 : 242.42424242424244 * brickVolume brick = wallVolume wall) :
  wall.width = 100 := by
  sorry

end NUMINAMATH_CALUDE_wall_width_is_100cm_l2556_255610


namespace NUMINAMATH_CALUDE_x_gt_one_sufficient_not_necessary_for_x_squared_gt_one_l2556_255673

theorem x_gt_one_sufficient_not_necessary_for_x_squared_gt_one :
  (∃ x : ℝ, x > 1 → x^2 > 1) ∧
  (∃ x : ℝ, x^2 > 1 ∧ ¬(x > 1)) := by
  sorry

end NUMINAMATH_CALUDE_x_gt_one_sufficient_not_necessary_for_x_squared_gt_one_l2556_255673


namespace NUMINAMATH_CALUDE_stephanie_oranges_l2556_255650

/-- The number of times Stephanie went to the store last month -/
def store_visits : ℕ := 8

/-- The number of oranges Stephanie buys each time she goes to the store -/
def oranges_per_visit : ℕ := 2

/-- The total number of oranges Stephanie bought last month -/
def total_oranges : ℕ := store_visits * oranges_per_visit

theorem stephanie_oranges : total_oranges = 16 := by
  sorry

end NUMINAMATH_CALUDE_stephanie_oranges_l2556_255650


namespace NUMINAMATH_CALUDE_max_value_of_p_l2556_255684

theorem max_value_of_p (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) 
  (h : a * b * c + a + c = b) :
  ∃ (p : ℝ), p = (2 / (a^2 + 1)) - (2 / (b^2 + 1)) + (3 / (c^2 + 1)) ∧ 
  p ≤ 10/3 ∧ 
  (∃ (a' b' c' : ℝ) (ha' : 0 < a') (hb' : 0 < b') (hc' : 0 < c') 
    (h' : a' * b' * c' + a' + c' = b'), 
    (2 / (a'^2 + 1)) - (2 / (b'^2 + 1)) + (3 / (c'^2 + 1)) = 10/3) :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_p_l2556_255684


namespace NUMINAMATH_CALUDE_johns_remaining_money_l2556_255687

/-- The amount of money John has left after buying pizzas and sodas -/
theorem johns_remaining_money (q : ℝ) : 
  (3 : ℝ) * (2 : ℝ) * q + -- cost of small pizzas
  (2 : ℝ) * (3 : ℝ) * q + -- cost of medium pizzas
  (4 : ℝ) * q             -- cost of sodas
  ≤ (50 : ℝ) →
  (50 : ℝ) - ((3 : ℝ) * (2 : ℝ) * q + (2 : ℝ) * (3 : ℝ) * q + (4 : ℝ) * q) = 50 - 16 * q :=
by sorry

end NUMINAMATH_CALUDE_johns_remaining_money_l2556_255687


namespace NUMINAMATH_CALUDE_lcm_gcd_problem_l2556_255617

theorem lcm_gcd_problem (a b : ℕ+) 
  (h1 : Nat.lcm a b = 2520)
  (h2 : Nat.gcd a b = 30)
  (h3 : a = 150) :
  b = 504 := by
  sorry

end NUMINAMATH_CALUDE_lcm_gcd_problem_l2556_255617


namespace NUMINAMATH_CALUDE_estimate_red_balls_l2556_255677

theorem estimate_red_balls (num_black : ℕ) (total_draws : ℕ) (black_draws : ℕ) :
  num_black = 4 →
  total_draws = 100 →
  black_draws = 40 →
  ∃ (num_red : ℕ),
    (num_black : ℚ) / (num_black + num_red : ℚ) = (black_draws : ℚ) / (total_draws : ℚ) ∧
    num_red = 6 := by
  sorry

end NUMINAMATH_CALUDE_estimate_red_balls_l2556_255677


namespace NUMINAMATH_CALUDE_no_upper_bound_for_y_l2556_255604

-- Define the equation
def hyperbola_equation (x y : ℝ) : Prop :=
  x^2 / 25 - (y - 3)^2 / 9 = 1

-- Theorem stating that there is no upper bound for y
theorem no_upper_bound_for_y :
  ∀ M : ℝ, ∃ x y : ℝ, hyperbola_equation x y ∧ y > M :=
by sorry

end NUMINAMATH_CALUDE_no_upper_bound_for_y_l2556_255604


namespace NUMINAMATH_CALUDE_negation_of_existence_negation_of_inequality_l2556_255609

theorem negation_of_existence (p : ℝ → Prop) : 
  (¬ ∃ x, p x) ↔ ∀ x, ¬ p x :=
by sorry

theorem negation_of_inequality : 
  (¬ ∃ x : ℝ, 2^x ≥ 2*x + 1) ↔ (∀ x : ℝ, 2^x < 2*x + 1) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_existence_negation_of_inequality_l2556_255609


namespace NUMINAMATH_CALUDE_five_sixteenths_decimal_l2556_255663

theorem five_sixteenths_decimal : (5 : ℚ) / 16 = 0.3125 := by sorry

end NUMINAMATH_CALUDE_five_sixteenths_decimal_l2556_255663


namespace NUMINAMATH_CALUDE_factorial_ratio_l2556_255637

theorem factorial_ratio : Nat.factorial 50 / Nat.factorial 48 = 2450 := by
  sorry

end NUMINAMATH_CALUDE_factorial_ratio_l2556_255637


namespace NUMINAMATH_CALUDE_solve_equation_l2556_255656

theorem solve_equation (x : ℝ) (h1 : x > 5) 
  (h2 : Real.sqrt (x - 3 * Real.sqrt (x - 5)) + 3 = Real.sqrt (x + 3 * Real.sqrt (x - 5)) - 3) : 
  x = 41 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l2556_255656


namespace NUMINAMATH_CALUDE_freds_remaining_cards_l2556_255647

/-- Calculates the number of baseball cards Fred has after Melanie's purchase. -/
def remaining_cards (initial : ℕ) (bought : ℕ) : ℕ :=
  initial - bought

/-- Theorem stating that Fred's remaining cards is the difference between his initial cards and those bought by Melanie. -/
theorem freds_remaining_cards :
  remaining_cards 5 3 = 2 := by
  sorry

end NUMINAMATH_CALUDE_freds_remaining_cards_l2556_255647


namespace NUMINAMATH_CALUDE_total_cost_is_240000_l2556_255697

/-- The total cost of three necklaces and a set of earrings -/
def total_cost (necklace_price : ℕ) : ℕ :=
  3 * necklace_price + 3 * necklace_price

/-- Proof that the total cost is $240,000 -/
theorem total_cost_is_240000 :
  total_cost 40000 = 240000 := by
  sorry

#eval total_cost 40000

end NUMINAMATH_CALUDE_total_cost_is_240000_l2556_255697


namespace NUMINAMATH_CALUDE_cone_sphere_volume_ratio_l2556_255658

/-- Given an equilateral cone and a sphere with surface area ratio 3:1, 
    the volume ratio of the cone to the sphere is 2√3 -/
theorem cone_sphere_volume_ratio (r : ℝ) (h : r > 0) : 
  let R := 2 * r
  let cone_surface := 3 * R^2 * Real.pi
  let sphere_surface := 4 * r^2 * Real.pi
  let cone_volume := (1/3) * Real.pi * R^2 * (R * Real.sqrt 3)
  let sphere_volume := (4/3) * Real.pi * r^3
  (cone_surface / sphere_surface = 3) → (cone_volume / sphere_volume = 2 * Real.sqrt 3) := by
sorry


end NUMINAMATH_CALUDE_cone_sphere_volume_ratio_l2556_255658


namespace NUMINAMATH_CALUDE_pie_baking_difference_l2556_255693

def alice_bake_time : ℕ := 5
def bob_bake_time : ℕ := 6
def charlie_bake_time : ℕ := 7
def total_time : ℕ := 90

def pies_baked (bake_time : ℕ) : ℕ := total_time / bake_time

theorem pie_baking_difference :
  (pies_baked alice_bake_time) - (pies_baked bob_bake_time) + 
  (pies_baked alice_bake_time) - (pies_baked charlie_bake_time) = 9 := by
  sorry

end NUMINAMATH_CALUDE_pie_baking_difference_l2556_255693


namespace NUMINAMATH_CALUDE_product_multiple_of_three_probability_l2556_255622

/-- The probability of rolling a multiple of 3 on a standard die -/
def prob_multiple_of_three : ℚ := 1/3

/-- The number of rolls -/
def num_rolls : ℕ := 8

/-- The probability that the product of all rolls is a multiple of 3 -/
def prob_product_multiple_of_three : ℚ :=
  1 - (1 - prob_multiple_of_three) ^ num_rolls

theorem product_multiple_of_three_probability :
  prob_product_multiple_of_three = 6305/6561 := by
  sorry

end NUMINAMATH_CALUDE_product_multiple_of_three_probability_l2556_255622


namespace NUMINAMATH_CALUDE_equal_segments_iff_proportion_l2556_255664

/-- A triangle with side lengths a, b, and c where a ≤ c -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  a_le_c : a ≤ c

/-- The internal bisector of the incenter divides the median from point B into three equal segments -/
def has_equal_segments (t : Triangle) : Prop :=
  ∃ (x : ℝ), x > 0 ∧ 
    let m := (t.a^2 + t.c^2 - t.b^2/2) / 2
    (3*x)^2 = m ∧
    ((t.a + t.c - t.b)/2)^2 = 2*x^2 ∧
    ((t.c - t.a)/2)^2 = 2*x^2

/-- The side lengths satisfy the proportion a/5 = b/10 = c/13 -/
def satisfies_proportion (t : Triangle) : Prop :=
  ∃ (k : ℝ), k > 0 ∧ t.a = 5*k ∧ t.b = 10*k ∧ t.c = 13*k

theorem equal_segments_iff_proportion (t : Triangle) :
  has_equal_segments t ↔ satisfies_proportion t := by
  sorry

end NUMINAMATH_CALUDE_equal_segments_iff_proportion_l2556_255664


namespace NUMINAMATH_CALUDE_sqrt_simplification_l2556_255676

theorem sqrt_simplification :
  Real.sqrt 75 + Real.sqrt 8 - Real.sqrt 18 - Real.sqrt 6 * Real.sqrt 2 = 3 * Real.sqrt 3 - Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_simplification_l2556_255676


namespace NUMINAMATH_CALUDE_exists_valid_coloring_l2556_255671

/-- A coloring of the plane using seven colors -/
def Coloring := ℝ × ℝ → Fin 7

/-- The property that no two points of the same color are exactly 1 unit apart -/
def ValidColoring (c : Coloring) : Prop :=
  ∀ x y : ℝ × ℝ, c x = c y → (x.1 - y.1)^2 + (x.2 - y.2)^2 ≠ 1

/-- There exists a coloring of the plane using seven colors such that
    no two points of the same color are exactly 1 unit apart -/
theorem exists_valid_coloring : ∃ c : Coloring, ValidColoring c := by
  sorry

end NUMINAMATH_CALUDE_exists_valid_coloring_l2556_255671


namespace NUMINAMATH_CALUDE_root_sum_reciprocal_products_l2556_255632

theorem root_sum_reciprocal_products (p q r s : ℂ) : 
  (p^4 + 10*p^3 + 20*p^2 + 15*p + 6 = 0) →
  (q^4 + 10*q^3 + 20*q^2 + 15*q + 6 = 0) →
  (r^4 + 10*r^3 + 20*r^2 + 15*r + 6 = 0) →
  (s^4 + 10*s^3 + 20*s^2 + 15*s + 6 = 0) →
  1/(p*q) + 1/(p*r) + 1/(p*s) + 1/(q*r) + 1/(q*s) + 1/(r*s) = 10/3 :=
by sorry

end NUMINAMATH_CALUDE_root_sum_reciprocal_products_l2556_255632


namespace NUMINAMATH_CALUDE_sin_690_degrees_l2556_255690

theorem sin_690_degrees :
  Real.sin (690 * π / 180) = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_sin_690_degrees_l2556_255690


namespace NUMINAMATH_CALUDE_smallest_divisible_by_8_9_11_l2556_255667

theorem smallest_divisible_by_8_9_11 : ∀ n : ℕ, n > 0 ∧ 8 ∣ n ∧ 9 ∣ n ∧ 11 ∣ n → n ≥ 792 :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_divisible_by_8_9_11_l2556_255667


namespace NUMINAMATH_CALUDE_cubic_equation_solution_l2556_255651

theorem cubic_equation_solution :
  let f : ℝ → ℝ := λ x => (x + 1)^3 + (3 - x)^3
  ∃ (x₁ x₂ : ℝ), 
    (f x₁ = 35 ∧ f x₂ = 35) ∧ 
    (x₁ = 1 + Real.sqrt (19/3) / 2) ∧ 
    (x₂ = 1 - Real.sqrt (19/3) / 2) ∧
    (∀ x : ℝ, f x = 35 → x = x₁ ∨ x = x₂) :=
by sorry

end NUMINAMATH_CALUDE_cubic_equation_solution_l2556_255651


namespace NUMINAMATH_CALUDE_complex_equation_solution_l2556_255607

theorem complex_equation_solution (c d : ℂ) (x : ℝ) :
  Complex.abs c = 3 →
  Complex.abs d = 5 →
  c * d = x - 3 * Complex.I →
  x = 6 * Real.sqrt 6 :=
by sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l2556_255607


namespace NUMINAMATH_CALUDE_no_rectangle_satisfies_conditions_l2556_255661

theorem no_rectangle_satisfies_conditions (p q : ℝ) (hp : p > q) (hq : q > 0) :
  ¬∃ x y : ℝ, x < p ∧ y < q ∧ x + y = (p + q) / 2 ∧ x * y = p * q / 4 := by
  sorry

end NUMINAMATH_CALUDE_no_rectangle_satisfies_conditions_l2556_255661


namespace NUMINAMATH_CALUDE_sequence_ratio_range_l2556_255652

theorem sequence_ratio_range (x y a₁ a₂ b₁ b₂ : ℝ) 
  (h_arith : a₁ - x = a₂ - a₁ ∧ a₂ - a₁ = y - a₂)
  (h_geom : b₁ / x = b₂ / b₁ ∧ b₂ / b₁ = y / b₂) :
  (a₁ + a₂)^2 / (b₁ * b₂) ≥ 4 ∨ (a₁ + a₂)^2 / (b₁ * b₂) ≤ 0 :=
by sorry

end NUMINAMATH_CALUDE_sequence_ratio_range_l2556_255652


namespace NUMINAMATH_CALUDE_inequality_proof_l2556_255645

theorem inequality_proof (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (h : x * y * z ≥ 1) :
  (x^5 - x^2) / (x^5 + y^2 + z^2) + (y^5 - y^2) / (y^5 + x^2 + z^2) + (z^5 - z^2) / (z^5 + x^2 + y^2) ≥ 0 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2556_255645


namespace NUMINAMATH_CALUDE_bottle_production_l2556_255615

/-- Given that 6 identical machines produce 270 bottles per minute at a constant rate,
    prove that 20 such machines will produce 3600 bottles in 4 minutes. -/
theorem bottle_production (rate : ℕ) (h1 : 6 * rate = 270) : 20 * rate * 4 = 3600 := by
  sorry

end NUMINAMATH_CALUDE_bottle_production_l2556_255615


namespace NUMINAMATH_CALUDE_coin_problem_l2556_255686

theorem coin_problem (total : ℕ) (difference : ℕ) (heads : ℕ) : 
  total = 128 → difference = 12 → heads = (total + difference) / 2 → heads = 70 := by
  sorry

end NUMINAMATH_CALUDE_coin_problem_l2556_255686


namespace NUMINAMATH_CALUDE_perpendicular_vectors_l2556_255689

theorem perpendicular_vectors (a b : ℝ × ℝ) :
  a = (-2, 1) →
  b = (3, 2) →
  ∃ k : ℝ, k = 5/4 ∧ (a.1 * (a.1 + k * b.1) + a.2 * (a.2 + k * b.2) = 0) :=
by sorry

end NUMINAMATH_CALUDE_perpendicular_vectors_l2556_255689


namespace NUMINAMATH_CALUDE_part_one_part_two_l2556_255678

-- Define the function f
def f (x m : ℝ) : ℝ := |x + m| + |2 * x - 3|

-- Part I
theorem part_one : 
  {x : ℝ | f x (-3) < 9} = {x : ℝ | -1 < x ∧ x < 5} :=
sorry

-- Part II
theorem part_two :
  (∃ x ∈ Set.Icc 2 4, f x m ≤ 3) → m ∈ Set.Icc (-2) 2 :=
sorry

end NUMINAMATH_CALUDE_part_one_part_two_l2556_255678


namespace NUMINAMATH_CALUDE_jake_peaches_l2556_255618

/-- Given the number of peaches Steven, Jill, and Jake have, prove that Jake has 8 peaches. -/
theorem jake_peaches (steven jill jake : ℕ) 
  (h1 : steven = 15)
  (h2 : steven = jill + 14)
  (h3 : jake + 7 = steven) : 
  jake = 8 := by
sorry

end NUMINAMATH_CALUDE_jake_peaches_l2556_255618


namespace NUMINAMATH_CALUDE_fewer_bees_than_flowers_l2556_255625

theorem fewer_bees_than_flowers : 
  let flowers : ℕ := 5
  let bees : ℕ := 3
  flowers - bees = 2 := by sorry

end NUMINAMATH_CALUDE_fewer_bees_than_flowers_l2556_255625


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_property_l2556_255699

/-- An arithmetic sequence with its sum function -/
structure ArithmeticSequence where
  S : ℕ → ℝ  -- Sum function

/-- Theorem: For an arithmetic sequence, if S_30 = S_60, then S_90 = 0 -/
theorem arithmetic_sequence_sum_property (seq : ArithmeticSequence) 
  (h : seq.S 30 = seq.S 60) : seq.S 90 = 0 := by
  sorry


end NUMINAMATH_CALUDE_arithmetic_sequence_sum_property_l2556_255699


namespace NUMINAMATH_CALUDE_expression_value_l2556_255639

theorem expression_value (x y : ℝ) (hx : x = 3) (hy : y = 4) :
  (3 * x^3 + 4 * y^3) / 9 = 37 + 4/9 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l2556_255639


namespace NUMINAMATH_CALUDE_unique_parallel_line_l2556_255633

-- Define the types for our geometric objects
variable (Point Line Plane : Type)

-- Define the relationships between geometric objects
variable (parallel : Plane → Plane → Prop)
variable (contains : Plane → Line → Prop)
variable (in_plane : Point → Plane → Prop)
variable (passes_through : Line → Point → Prop)
variable (line_in_plane : Line → Plane → Prop)
variable (line_parallel : Line → Line → Prop)

-- State the theorem
theorem unique_parallel_line 
  (α β : Plane) (a : Line) (B : Point)
  (h1 : parallel α β)
  (h2 : contains α a)
  (h3 : in_plane B β) :
  ∃! l : Line, line_in_plane l β ∧ passes_through l B ∧ line_parallel l a :=
sorry

end NUMINAMATH_CALUDE_unique_parallel_line_l2556_255633


namespace NUMINAMATH_CALUDE_grassy_plot_width_l2556_255695

/-- Given a rectangular grassy plot with a gravel path, calculate its width -/
theorem grassy_plot_width : ℝ :=
  let plot_length : ℝ := 110
  let path_width : ℝ := 2.5
  let gravelling_cost_per_sq_meter : ℝ := 0.80
  let total_gravelling_cost : ℝ := 680
  let plot_width : ℝ := 97.5
  
  have h1 : plot_length > 0 := by sorry
  have h2 : path_width > 0 := by sorry
  have h3 : gravelling_cost_per_sq_meter > 0 := by sorry
  have h4 : total_gravelling_cost > 0 := by sorry
  
  have path_area : ℝ := 
    (plot_length + 2 * path_width) * (plot_width + 2 * path_width) - 
    plot_length * plot_width
  
  have total_cost_equation : 
    gravelling_cost_per_sq_meter * path_area = total_gravelling_cost := by sorry
  
  plot_width

end NUMINAMATH_CALUDE_grassy_plot_width_l2556_255695


namespace NUMINAMATH_CALUDE_bug_prob_after_8_meters_l2556_255613

/-- Represents the probability of the bug being at vertex A after n meters -/
def Q (n : ℕ) : ℚ :=
  match n with
  | 0 => 1
  | n + 1 => (1 - Q n) / 3

/-- The vertices of the tetrahedron -/
inductive Vertex
| A | B | C | D

/-- The probability of the bug being at vertex A after 8 meters -/
def prob_at_A_after_8 : ℚ := Q 8

theorem bug_prob_after_8_meters :
  prob_at_A_after_8 = 547 / 2187 :=
sorry

end NUMINAMATH_CALUDE_bug_prob_after_8_meters_l2556_255613


namespace NUMINAMATH_CALUDE_shaded_area_is_30_l2556_255603

/-- An isosceles right triangle with legs of length 12 -/
structure IsoscelesRightTriangle where
  leg_length : ℝ
  leg_length_eq : leg_length = 12

/-- A partition of the triangle into 36 congruent smaller triangles -/
structure TrianglePartition (t : IsoscelesRightTriangle) where
  num_parts : ℕ
  num_parts_eq : num_parts = 36

/-- A shaded region consisting of 15 smaller triangles -/
structure ShadedRegion (p : TrianglePartition t) where
  num_shaded : ℕ
  num_shaded_eq : num_shaded = 15

/-- The theorem stating that the shaded area is 30 -/
theorem shaded_area_is_30 (t : IsoscelesRightTriangle) 
  (p : TrianglePartition t) (s : ShadedRegion p) : 
  Real.sqrt ((t.leg_length ^ 2) / 2) * s.num_shaded = 30 := by
  sorry

end NUMINAMATH_CALUDE_shaded_area_is_30_l2556_255603


namespace NUMINAMATH_CALUDE_power_of_negative_one_product_l2556_255634

theorem power_of_negative_one_product (n : ℕ) : 
  ((-1 : ℤ) ^ n) * ((-1 : ℤ) ^ (2 * n + 1)) * ((-1 : ℤ) ^ (n + 1)) = 1 := by
  sorry

end NUMINAMATH_CALUDE_power_of_negative_one_product_l2556_255634


namespace NUMINAMATH_CALUDE_blue_segments_count_l2556_255643

/-- Represents a 10x10 grid with colored points and segments -/
structure ColoredGrid :=
  (red_points : Nat)
  (red_corners : Nat)
  (red_edges : Nat)
  (green_segments : Nat)

/-- Calculates the number of blue segments in the grid -/
def blue_segments (grid : ColoredGrid) : Nat :=
  -- The actual calculation would go here
  sorry

/-- Theorem stating the number of blue segments in the given conditions -/
theorem blue_segments_count (grid : ColoredGrid) :
  grid.red_points = 52 →
  grid.red_corners = 2 →
  grid.red_edges = 16 →
  grid.green_segments = 98 →
  blue_segments grid = 37 := by
  sorry

end NUMINAMATH_CALUDE_blue_segments_count_l2556_255643


namespace NUMINAMATH_CALUDE_circle_ratio_l2556_255679

theorem circle_ratio (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h : π * b^2 - π * a^2 = 5 * (π * a^2)) : 
  a / b = 1 / Real.sqrt 6 := by
  sorry

end NUMINAMATH_CALUDE_circle_ratio_l2556_255679


namespace NUMINAMATH_CALUDE_power_of_three_mod_thirteen_l2556_255644

theorem power_of_three_mod_thirteen : 3^3021 % 13 = 1 := by
  sorry

end NUMINAMATH_CALUDE_power_of_three_mod_thirteen_l2556_255644


namespace NUMINAMATH_CALUDE_c_2017_value_l2556_255631

/-- Sequence a_n -/
def a : ℕ → ℕ
  | 0 => 3
  | n + 1 => a n + 3

/-- Sequence b_n -/
def b : ℕ → ℕ
  | 0 => 3
  | n + 1 => 3 * b n

/-- Sequence c_n -/
def c (n : ℕ) : ℕ := b (a n - 1)

theorem c_2017_value : c 2016 = 27^2017 := by sorry

end NUMINAMATH_CALUDE_c_2017_value_l2556_255631


namespace NUMINAMATH_CALUDE_distribute_negative_two_l2556_255602

theorem distribute_negative_two (m n : ℝ) : -2 * (m - n) = -2 * m + 2 * n := by
  sorry

end NUMINAMATH_CALUDE_distribute_negative_two_l2556_255602


namespace NUMINAMATH_CALUDE_percentage_difference_l2556_255691

theorem percentage_difference (p t j : ℝ) 
  (h1 : j = 0.75 * p) 
  (h2 : j = 0.8 * t) : 
  t = (15/16) * p := by
  sorry

end NUMINAMATH_CALUDE_percentage_difference_l2556_255691


namespace NUMINAMATH_CALUDE_fourth_sample_seat_number_l2556_255605

/-- Represents a systematic sampling scheme -/
structure SystematicSample where
  population_size : ℕ
  sample_size : ℕ
  first_sample : ℕ

/-- Calculates the nth element in a systematic sample -/
def nth_sample (s : SystematicSample) (n : ℕ) : ℕ :=
  s.first_sample + (n - 1) * (s.population_size / s.sample_size)

theorem fourth_sample_seat_number 
  (sample : SystematicSample)
  (h1 : sample.population_size = 56)
  (h2 : sample.sample_size = 4)
  (h3 : sample.first_sample = 4)
  (h4 : nth_sample sample 2 = 18)
  (h5 : nth_sample sample 3 = 46) :
  nth_sample sample 4 = 32 := by
  sorry

end NUMINAMATH_CALUDE_fourth_sample_seat_number_l2556_255605


namespace NUMINAMATH_CALUDE_sum_of_repeating_decimals_l2556_255611

-- Define the repeating decimals
def repeating_decimal_1 : ℚ := 2/9
def repeating_decimal_2 : ℚ := 2/99

-- Theorem statement
theorem sum_of_repeating_decimals :
  repeating_decimal_1 + repeating_decimal_2 = 8/33 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_repeating_decimals_l2556_255611


namespace NUMINAMATH_CALUDE_sphere_surface_area_rectangular_solid_l2556_255653

/-- The surface area of a sphere containing all vertices of a rectangular solid -/
theorem sphere_surface_area_rectangular_solid (l w h : ℝ) (r : ℝ) : 
  l = 2 → w = 1 → h = 2 → 
  r^2 = (l^2 + w^2 + h^2) / 4 →
  4 * Real.pi * r^2 = 9 * Real.pi := by sorry

end NUMINAMATH_CALUDE_sphere_surface_area_rectangular_solid_l2556_255653


namespace NUMINAMATH_CALUDE_batsman_average_after_31st_inning_l2556_255698

/-- Represents a batsman's cricket statistics -/
structure Batsman where
  innings : ℕ
  totalRuns : ℕ
  average : ℚ

/-- Calculates the new average after an innings -/
def newAverage (b : Batsman) (runsScored : ℕ) : ℚ :=
  (b.totalRuns + runsScored : ℚ) / (b.innings + 1 : ℚ)

theorem batsman_average_after_31st_inning 
  (b : Batsman)
  (h1 : b.innings = 30)
  (h2 : newAverage b 105 = b.average + 3) :
  newAverage b 105 = 15 := by
  sorry

#check batsman_average_after_31st_inning

end NUMINAMATH_CALUDE_batsman_average_after_31st_inning_l2556_255698


namespace NUMINAMATH_CALUDE_number_times_one_sixth_squared_l2556_255630

theorem number_times_one_sixth_squared (x : ℝ) : x * (1/6)^2 = 6^3 ↔ x = 7776 := by
  sorry

end NUMINAMATH_CALUDE_number_times_one_sixth_squared_l2556_255630


namespace NUMINAMATH_CALUDE_bac_is_105_l2556_255668

/-- Represents the encoding of a base-5 digit --/
inductive Encoding
  | A
  | B
  | C
  | D
  | E

/-- Converts an Encoding to its corresponding base-5 digit --/
def encoding_to_digit (e : Encoding) : Nat :=
  match e with
  | Encoding.A => 1
  | Encoding.B => 4
  | Encoding.C => 0
  | Encoding.D => 3
  | Encoding.E => 4

/-- Converts a sequence of Encodings to its base-10 representation --/
def encodings_to_base10 (encodings : List Encoding) : Nat :=
  encodings.enum.foldl (fun acc (i, e) => acc + encoding_to_digit e * (5 ^ (encodings.length - 1 - i))) 0

/-- The main theorem stating that BAC in the given encoding system represents 105 in base 10 --/
theorem bac_is_105 (h1 : encodings_to_base10 [Encoding.A, Encoding.B, Encoding.E] + 1 = encodings_to_base10 [Encoding.A, Encoding.B, Encoding.D])
                   (h2 : encodings_to_base10 [Encoding.A, Encoding.B, Encoding.D] + 1 = encodings_to_base10 [Encoding.A, Encoding.A, Encoding.C]) :
  encodings_to_base10 [Encoding.B, Encoding.A, Encoding.C] = 105 := by
  sorry

end NUMINAMATH_CALUDE_bac_is_105_l2556_255668


namespace NUMINAMATH_CALUDE_arithmetic_mean_after_removal_l2556_255655

/-- Given a set of 60 numbers with an arithmetic mean of 42, 
    prove that removing 50 and 60 results in a new arithmetic mean of 41.5 -/
theorem arithmetic_mean_after_removal (S : Finset ℝ) (x y : ℝ) : 
  S.card = 60 → 
  x ∈ S → 
  y ∈ S → 
  x = 50 → 
  y = 60 → 
  (S.sum id) / S.card = 42 → 
  ((S.sum id) - x - y) / (S.card - 2) = 41.5 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_mean_after_removal_l2556_255655


namespace NUMINAMATH_CALUDE_sausage_division_ratio_l2556_255629

/-- Represents the length of the sausage after each bite -/
def remaining_length (n : ℕ) : ℚ :=
  match n with
  | 0 => 1
  | n + 1 => if n % 2 = 0 then 3/4 * remaining_length n else 2/3 * remaining_length n

/-- Theorem stating that the sausage should be divided in a 1:1 ratio -/
theorem sausage_division_ratio :
  ∀ ε > 0, ∃ N : ℕ, ∀ n ≥ N, |remaining_length n - 1/2| < ε :=
sorry

end NUMINAMATH_CALUDE_sausage_division_ratio_l2556_255629


namespace NUMINAMATH_CALUDE_trapezoid_area_sum_l2556_255601

/-- Represents a trapezoid with four side lengths -/
structure Trapezoid where
  side1 : ℝ
  side2 : ℝ
  side3 : ℝ
  side4 : ℝ

/-- Calculates the sum of all possible areas of a trapezoid -/
def sum_of_areas (t : Trapezoid) : ℝ := sorry

/-- Checks if a number is not divisible by the square of any prime -/
def not_divisible_by_square_prime (n : ℕ) : Prop := sorry

/-- Main theorem statement -/
theorem trapezoid_area_sum (t : Trapezoid) :
  t.side1 = 4 ∧ t.side2 = 6 ∧ t.side3 = 8 ∧ t.side4 = 10 →
  ∃ (r₁ r₂ r₃ : ℚ) (n₁ n₂ : ℕ),
    sum_of_areas t = r₁ * Real.sqrt n₁ + r₂ * Real.sqrt n₂ + r₃ ∧
    not_divisible_by_square_prime n₁ ∧
    not_divisible_by_square_prime n₂ ∧
    ⌊r₁ + r₂ + r₃ + n₁ + n₂⌋ = 18 := by
  sorry

end NUMINAMATH_CALUDE_trapezoid_area_sum_l2556_255601


namespace NUMINAMATH_CALUDE_complex_equation_solution_l2556_255669

theorem complex_equation_solution (z : ℂ) : 2 * z * Complex.I = 1 + 3 * Complex.I → z = (3 / 2 : ℂ) - (1 / 2 : ℂ) * Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l2556_255669


namespace NUMINAMATH_CALUDE_function_satisfies_equation_l2556_255659

theorem function_satisfies_equation (x : ℝ) : 
  let y : ℝ → ℝ := λ x => (1 + x) / (1 - x)
  deriv y x = (1 + (y x)^2) / (1 + x^2) := by sorry

end NUMINAMATH_CALUDE_function_satisfies_equation_l2556_255659


namespace NUMINAMATH_CALUDE_tank_initial_water_l2556_255606

def tank_capacity : ℚ := 100
def day1_collection : ℚ := 15
def day2_collection : ℚ := 20
def day3_overflow : ℚ := 25

theorem tank_initial_water (initial_water : ℚ) :
  initial_water + day1_collection + day2_collection = tank_capacity ∧
  (initial_water / tank_capacity = 13 / 20) := by
  sorry

end NUMINAMATH_CALUDE_tank_initial_water_l2556_255606


namespace NUMINAMATH_CALUDE_min_complex_value_is_zero_l2556_255675

theorem min_complex_value_is_zero 
  (a b c : ℤ) 
  (ω : ℂ) 
  (h_distinct : a ≠ b ∧ b ≠ c ∧ a ≠ c) 
  (h_omega_power : ω^4 = 1) 
  (h_omega_not_one : ω ≠ 1) :
  ∃ (a' b' c' : ℤ) (h_distinct' : a' ≠ b' ∧ b' ≠ c' ∧ a' ≠ c'),
    Complex.abs (a' + b' * ω + c' * ω^3) = 0 ∧
    ∀ (x y z : ℤ) (h_xyz_distinct : x ≠ y ∧ y ≠ z ∧ x ≠ z),
      Complex.abs (x + y * ω + z * ω^3) ≥ 0 :=
by sorry

end NUMINAMATH_CALUDE_min_complex_value_is_zero_l2556_255675


namespace NUMINAMATH_CALUDE_OPQRS_shape_l2556_255688

/-- A point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- The figure formed by points O, P, Q, R, S -/
inductive Figure
  | Parallelepiped
  | Plane
  | Line
  | General3D

/-- The theorem stating that OPQRS can only be a parallelepiped or a plane -/
theorem OPQRS_shape (P Q R S : Point3D)
  (hP : P ≠ Q ∧ P ≠ R ∧ P ≠ S)
  (hQ : Q ≠ R ∧ Q ≠ S)
  (hR : R ≠ S)
  (hR_sum : R = Point3D.mk (P.x + Q.x) (P.y + Q.y) (P.z + Q.z))
  (O : Point3D := Point3D.mk 0 0 0) :
  ∃ (f : Figure), (f = Figure.Parallelepiped ∨ f = Figure.Plane) ∧
    (∀ (g : Figure), g ≠ Figure.Line ∧ (g ≠ Figure.General3D ∨ g = Figure.Plane ∨ g = Figure.Parallelepiped)) :=
sorry

end NUMINAMATH_CALUDE_OPQRS_shape_l2556_255688


namespace NUMINAMATH_CALUDE_jason_picked_ten_plums_l2556_255600

def alyssa_plums : ℕ := 17
def total_plums : ℕ := 27

def jason_plums : ℕ := total_plums - alyssa_plums

theorem jason_picked_ten_plums : jason_plums = 10 := by
  sorry

end NUMINAMATH_CALUDE_jason_picked_ten_plums_l2556_255600


namespace NUMINAMATH_CALUDE_tim_has_203_balloons_l2556_255608

/-- The number of violet balloons Dan has -/
def dan_balloons : ℕ := 29

/-- The factor by which Tim's balloons exceed Dan's -/
def tim_factor : ℕ := 7

/-- The number of violet balloons Tim has -/
def tim_balloons : ℕ := dan_balloons * tim_factor

/-- Theorem: Tim has 203 violet balloons -/
theorem tim_has_203_balloons : tim_balloons = 203 := by
  sorry

end NUMINAMATH_CALUDE_tim_has_203_balloons_l2556_255608


namespace NUMINAMATH_CALUDE_crossword_solvable_l2556_255649

-- Define the structure of a crossword puzzle
structure Crossword :=
  (grid : List (List Char))
  (vertical_clues : List String)
  (horizontal_clues : List String)

-- Define the words for the crossword
def words : List String := ["счет", "евро", "доллар", "вклад", "золото", "ломбард", "обмен", "система"]

-- Define the clues for the crossword
def vertical_clues : List String := [
  "What a bank opens for a person who wants to become its client",
  "This currency is used in Italy and other places",
  "One of the most well-known international currencies, accepted for payment in many countries",
  "The way to store and gradually increase family money in the bank",
  "A precious metal, whose reserves are accounted for by the Bank of Russia",
  "An organization from which you can borrow money and pay a small interest"
]

def horizontal_clues : List String := [
  "To pay abroad, you need to carry out ... of currency",
  "In Russia, there is a multi-level banking ...: the Central Bank of the Russian Federation, banks with a universal license, and with a basic one",
  "The place where you can take jewelry and get a loan for it"
]

-- Define the function to check if the crossword is valid
def is_valid_crossword (c : Crossword) : Prop :=
  c.vertical_clues.length = 6 ∧
  c.horizontal_clues.length = 3 ∧
  c.grid.all (λ row => row.length = 6) ∧
  c.grid.length = 7

-- Define the theorem to prove
theorem crossword_solvable :
  ∃ (c : Crossword), is_valid_crossword c ∧
    (∀ w ∈ words, w.length ≤ 7) ∧
    (∀ clue ∈ c.vertical_clues ++ c.horizontal_clues, ∃ w ∈ words, clue.length > 0 ∧ w.length > 0) :=
sorry

end NUMINAMATH_CALUDE_crossword_solvable_l2556_255649


namespace NUMINAMATH_CALUDE_total_selling_price_proof_l2556_255683

def calculate_selling_price (cost : ℝ) (profit_percentage : ℝ) : ℝ :=
  cost * (1 + profit_percentage)

theorem total_selling_price_proof :
  let cost_prices : List ℝ := [280, 350, 500]
  let profit_percentages : List ℝ := [0.30, 0.45, 0.25]
  let selling_prices := List.zipWith calculate_selling_price cost_prices profit_percentages
  List.sum selling_prices = 1496.50 := by
sorry

end NUMINAMATH_CALUDE_total_selling_price_proof_l2556_255683


namespace NUMINAMATH_CALUDE_large_triangle_toothpicks_l2556_255636

/-- The number of small triangles in the base row of the large triangle -/
def base_triangles : ℕ := 1001

/-- The total number of small triangles in the large triangle -/
def total_triangles : ℕ := base_triangles * (base_triangles + 1) / 2

/-- The number of toothpicks needed to construct the large triangle -/
def toothpicks : ℕ := (3 * total_triangles) / 2

theorem large_triangle_toothpicks :
  toothpicks = 752252 :=
sorry

end NUMINAMATH_CALUDE_large_triangle_toothpicks_l2556_255636


namespace NUMINAMATH_CALUDE_max_value_of_f_l2556_255628

/-- Given a function f(x) = (x^2 - 4)(x - a) where a is a real number and f'(1) = 0,
    the maximum value of f(x) on the interval [-2, 2] is 50/27. -/
theorem max_value_of_f (a : ℝ) (f : ℝ → ℝ) (h1 : ∀ x, f x = (x^2 - 4) * (x - a)) 
    (h2 : deriv f 1 = 0) : 
    ∃ x ∈ Set.Icc (-2 : ℝ) 2, ∀ y ∈ Set.Icc (-2 : ℝ) 2, f y ≤ f x ∧ f x = 50/27 := by
  sorry

end NUMINAMATH_CALUDE_max_value_of_f_l2556_255628


namespace NUMINAMATH_CALUDE_initial_ratio_men_to_women_l2556_255674

/-- Proves that the initial ratio of men to women in a room was 4:5 --/
theorem initial_ratio_men_to_women :
  ∀ (initial_men initial_women : ℕ),
  (initial_women - 3) * 2 = 24 →
  initial_men + 2 = 14 →
  (initial_men : ℚ) / initial_women = 4 / 5 := by
sorry

end NUMINAMATH_CALUDE_initial_ratio_men_to_women_l2556_255674


namespace NUMINAMATH_CALUDE_zara_sheep_count_l2556_255619

/-- The number of sheep Zara bought -/
def num_sheep : ℕ := 7

/-- The number of cows Zara bought -/
def num_cows : ℕ := 24

/-- The number of goats Zara bought -/
def num_goats : ℕ := 113

/-- The number of groups for transporting animals -/
def num_groups : ℕ := 3

/-- The number of animals in each group -/
def animals_per_group : ℕ := 48

theorem zara_sheep_count :
  num_sheep + num_cows + num_goats = num_groups * animals_per_group := by
  sorry

end NUMINAMATH_CALUDE_zara_sheep_count_l2556_255619


namespace NUMINAMATH_CALUDE_mean_proportion_of_3_and_4_l2556_255681

theorem mean_proportion_of_3_and_4 :
  ∃ x : ℝ, (3 : ℝ) / x = x / 4 ∧ (x = 2 * Real.sqrt 3 ∨ x = -2 * Real.sqrt 3) :=
by sorry

end NUMINAMATH_CALUDE_mean_proportion_of_3_and_4_l2556_255681


namespace NUMINAMATH_CALUDE_ice_cream_ratio_is_two_to_one_l2556_255648

/-- The ratio of Victoria's ice cream scoops to Oli's ice cream scoops -/
def ice_cream_ratio : ℚ := by
  -- Define Oli's number of scoops
  let oli_scoops : ℕ := 4
  -- Define Victoria's number of scoops
  let victoria_scoops : ℕ := oli_scoops + 4
  -- Calculate the ratio
  exact (victoria_scoops : ℚ) / oli_scoops

/-- Theorem stating that the ice cream ratio is 2:1 -/
theorem ice_cream_ratio_is_two_to_one : ice_cream_ratio = 2 := by
  sorry

end NUMINAMATH_CALUDE_ice_cream_ratio_is_two_to_one_l2556_255648


namespace NUMINAMATH_CALUDE_polynomial_equality_l2556_255692

-- Define the polynomials P and Q
variable (P Q : ℝ → ℝ)

-- Define the property of being a nonconstant polynomial
def IsNonconstantPolynomial (f : ℝ → ℝ) : Prop := sorry

-- Define the theorem
theorem polynomial_equality
  (hP : IsNonconstantPolynomial P)
  (hQ : IsNonconstantPolynomial Q)
  (h : ∀ y : ℝ, ⌊P y⌋ = ⌊Q y⌋) :
  ∀ x : ℝ, P x = Q x := by sorry

end NUMINAMATH_CALUDE_polynomial_equality_l2556_255692


namespace NUMINAMATH_CALUDE_min_value_greater_than_five_l2556_255621

/-- The function f(x) defined in the problem -/
def f (a : ℝ) (x : ℝ) : ℝ := x^2 + (a + 1)^2 + |x + a - 1|

/-- The theorem statement -/
theorem min_value_greater_than_five (a : ℝ) :
  (∀ x, f a x > 5) ↔ a < (1 - Real.sqrt 14) / 2 ∨ a > Real.sqrt 6 / 2 := by sorry

end NUMINAMATH_CALUDE_min_value_greater_than_five_l2556_255621


namespace NUMINAMATH_CALUDE_equation_solution_l2556_255694

theorem equation_solution :
  let f : ℝ → ℝ := λ x => (x^2 + 1)^2 - 4*(x^2 + 1) - 12
  ∀ x : ℝ, f x = 0 ↔ x = Real.sqrt 5 ∨ x = -Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2556_255694


namespace NUMINAMATH_CALUDE_expected_collectors_is_120_l2556_255627

-- Define the number of customers
def num_customers : ℕ := 3000

-- Define the probability of a customer collecting a prize
def prob_collect : ℝ := 0.04

-- Define the expected number of prize collectors
def expected_collectors : ℝ := num_customers * prob_collect

-- Theorem statement
theorem expected_collectors_is_120 : expected_collectors = 120 := by
  sorry

end NUMINAMATH_CALUDE_expected_collectors_is_120_l2556_255627


namespace NUMINAMATH_CALUDE_not_divisible_by_five_l2556_255685

theorem not_divisible_by_five (n : ℤ) : ¬ (5 ∣ (n^3 + 2*n - 1)) := by
  sorry

end NUMINAMATH_CALUDE_not_divisible_by_five_l2556_255685


namespace NUMINAMATH_CALUDE_rope_length_problem_l2556_255696

theorem rope_length_problem (L : ℝ) : 
  (L / 3 + 0.3 * (2 * L / 3)) - (L - (L / 3 + 0.3 * (2 * L / 3))) = 0.4 → L = 6 := by
  sorry

end NUMINAMATH_CALUDE_rope_length_problem_l2556_255696


namespace NUMINAMATH_CALUDE_age_sum_proof_l2556_255624

theorem age_sum_proof (p q : ℕ) : 
  (p : ℚ) / q = 3 / 4 →
  p - 8 = (q - 8) / 2 →
  p + q = 28 := by
sorry

end NUMINAMATH_CALUDE_age_sum_proof_l2556_255624


namespace NUMINAMATH_CALUDE_negation_of_x_squared_plus_two_gt_zero_is_false_l2556_255640

theorem negation_of_x_squared_plus_two_gt_zero_is_false :
  ¬(∃ x : ℝ, x^2 + 2 ≤ 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_x_squared_plus_two_gt_zero_is_false_l2556_255640


namespace NUMINAMATH_CALUDE_arithmetic_calculation_l2556_255626

theorem arithmetic_calculation : 4 * 6 * 8 + 24 / 4 = 198 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_calculation_l2556_255626


namespace NUMINAMATH_CALUDE_camel_cannot_move_to_adjacent_l2556_255670

def Board := Fin 10 × Fin 10

def adjacent (a b : Board) : Prop :=
  (a.1 = b.1 ∧ (a.2 = b.2 + 1 ∨ a.2 = b.2 - 1)) ∨
  (a.2 = b.2 ∧ (a.1 = b.1 + 1 ∨ a.1 = b.1 - 1))

def camel_move (a b : Board) : Prop :=
  (a.1 = b.1 + 1 ∧ (a.2 = b.2 + 3 ∨ a.2 = b.2 - 3)) ∨
  (a.1 = b.1 - 1 ∧ (a.2 = b.2 + 3 ∨ a.2 = b.2 - 3)) ∨
  (a.2 = b.2 + 1 ∧ (a.1 = b.1 + 3 ∨ a.1 = b.1 - 3)) ∨
  (a.2 = b.2 - 1 ∧ (a.1 = b.1 + 3 ∨ a.1 = b.1 - 3))

theorem camel_cannot_move_to_adjacent :
  ∀ (start finish : Board), adjacent start finish → ¬ camel_move start finish :=
by sorry

end NUMINAMATH_CALUDE_camel_cannot_move_to_adjacent_l2556_255670


namespace NUMINAMATH_CALUDE_repeating_decimal_quotient_l2556_255672

/-- Represents a repeating decimal with a two-digit repetend -/
def RepeatingDecimal (n : ℕ) : ℚ :=
  n / 99

theorem repeating_decimal_quotient :
  (RepeatingDecimal 54) / (RepeatingDecimal 18) = 3 := by
  sorry

end NUMINAMATH_CALUDE_repeating_decimal_quotient_l2556_255672


namespace NUMINAMATH_CALUDE_square_of_negative_sum_l2556_255680

theorem square_of_negative_sum (a b : ℝ) : (-a - b)^2 = a^2 + 2*a*b + b^2 := by
  sorry

end NUMINAMATH_CALUDE_square_of_negative_sum_l2556_255680


namespace NUMINAMATH_CALUDE_girls_percentage_after_adding_boy_l2556_255620

def initial_boys : ℕ := 11
def initial_girls : ℕ := 13
def added_boys : ℕ := 1

def total_students : ℕ := initial_boys + initial_girls + added_boys

def girls_percentage : ℚ := (initial_girls : ℚ) / (total_students : ℚ) * 100

theorem girls_percentage_after_adding_boy :
  girls_percentage = 52 := by sorry

end NUMINAMATH_CALUDE_girls_percentage_after_adding_boy_l2556_255620


namespace NUMINAMATH_CALUDE_strawberry_jam_earnings_l2556_255638

/-- Represents the number of strawberries picked by each person and the jam-making process. -/
structure StrawberryPicking where
  betty : ℕ
  matthew : ℕ
  natalie : ℕ
  strawberries_per_jar : ℕ
  price_per_jar : ℕ

/-- Calculates the total money earned from selling jam made from picked strawberries. -/
def total_money_earned (sp : StrawberryPicking) : ℕ :=
  let total_strawberries := sp.betty + sp.matthew + sp.natalie
  let jars_of_jam := total_strawberries / sp.strawberries_per_jar
  jars_of_jam * sp.price_per_jar

/-- Theorem stating that under the given conditions, the total money earned is $40. -/
theorem strawberry_jam_earnings : ∀ (sp : StrawberryPicking),
  sp.betty = 16 ∧
  sp.matthew = sp.betty + 20 ∧
  sp.matthew = 2 * sp.natalie ∧
  sp.strawberries_per_jar = 7 ∧
  sp.price_per_jar = 4 →
  total_money_earned sp = 40 :=
by
  sorry


end NUMINAMATH_CALUDE_strawberry_jam_earnings_l2556_255638


namespace NUMINAMATH_CALUDE_power_equality_l2556_255614

theorem power_equality (p : ℕ) : 16^5 = 4^p → p = 10 := by
  sorry

end NUMINAMATH_CALUDE_power_equality_l2556_255614
