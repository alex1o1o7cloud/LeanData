import Mathlib

namespace NUMINAMATH_CALUDE_min_intersection_distance_l2841_284170

/-- The minimum distance between intersection points of a line and a circle --/
theorem min_intersection_distance (k : ℝ) : 
  let l := {(x, y) : ℝ × ℝ | y = k * x + 1}
  let C := {(x, y) : ℝ × ℝ | x^2 + y^2 - 2*x - 3 = 0}
  ∃ (A B : ℝ × ℝ), A ∈ l ∧ A ∈ C ∧ B ∈ l ∧ B ∈ C ∧
    ∀ (P Q : ℝ × ℝ), P ∈ l ∧ P ∈ C ∧ Q ∈ l ∧ Q ∈ C →
      Real.sqrt 8 ≤ Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2) :=
by
  sorry

end NUMINAMATH_CALUDE_min_intersection_distance_l2841_284170


namespace NUMINAMATH_CALUDE_race_even_distance_l2841_284102

/-- The distance Alex and Max were even at the beginning of the race -/
def even_distance : ℕ := sorry

/-- The total race distance in feet -/
def total_race_distance : ℕ := 5000

/-- The distance left for Max to catch up at the end of the race -/
def distance_left : ℕ := 3890

/-- Alex's first lead over Max in feet -/
def alex_first_lead : ℕ := 300

/-- Max's lead over Alex in feet -/
def max_lead : ℕ := 170

/-- Alex's final lead over Max in feet -/
def alex_final_lead : ℕ := 440

theorem race_even_distance :
  even_distance = 540 ∧
  even_distance + alex_first_lead - max_lead + alex_final_lead = total_race_distance - distance_left :=
by sorry

end NUMINAMATH_CALUDE_race_even_distance_l2841_284102


namespace NUMINAMATH_CALUDE_smallest_sum_of_abs_l2841_284189

def matrix_squared (a b c d : ℤ) : Matrix (Fin 2) (Fin 2) ℤ :=
  !![a^2 + b*c, a*b + b*d;
     a*c + c*d, b*c + d^2]

theorem smallest_sum_of_abs (a b c d : ℤ) : 
  a ≠ 0 → b ≠ 0 → c ≠ 0 → d ≠ 0 →
  matrix_squared a b c d = !![9, 0; 0, 9] →
  (∃ (w x y z : ℤ), w ≠ 0 ∧ x ≠ 0 ∧ y ≠ 0 ∧ z ≠ 0 ∧
    matrix_squared w x y z = !![9, 0; 0, 9] ∧
    |w| + |x| + |y| + |z| < |a| + |b| + |c| + |d|) ∨
  |a| + |b| + |c| + |d| = 8 :=
by sorry

end NUMINAMATH_CALUDE_smallest_sum_of_abs_l2841_284189


namespace NUMINAMATH_CALUDE_max_value_sqrt_sum_l2841_284181

theorem max_value_sqrt_sum (x : ℝ) (h : -49 ≤ x ∧ x ≤ 49) :
  Real.sqrt (49 + x) + Real.sqrt (49 - x) ≤ 14 ∧
  ∃ y : ℝ, -49 ≤ y ∧ y ≤ 49 ∧ Real.sqrt (49 + y) + Real.sqrt (49 - y) = 14 :=
by sorry

end NUMINAMATH_CALUDE_max_value_sqrt_sum_l2841_284181


namespace NUMINAMATH_CALUDE_alyssa_plums_count_l2841_284193

/-- The number of plums picked by Jason -/
def jason_plums : ℕ := 10

/-- The total number of plums picked -/
def total_plums : ℕ := 27

/-- The number of plums picked by Alyssa -/
def alyssa_plums : ℕ := total_plums - jason_plums

theorem alyssa_plums_count : alyssa_plums = 17 := by
  sorry

end NUMINAMATH_CALUDE_alyssa_plums_count_l2841_284193


namespace NUMINAMATH_CALUDE_chocolate_bar_reduction_l2841_284123

theorem chocolate_bar_reduction 
  (m n : ℕ) 
  (h_lt : m < n) 
  (a b : ℕ) 
  (h_div_a : n^5 ∣ a) 
  (h_div_b : n^5 ∣ b) : 
  ∃ (x y : ℕ), 
    x ≤ a ∧ 
    y ≤ b ∧ 
    x * y = a * b * (m / n)^10 := by
  sorry

end NUMINAMATH_CALUDE_chocolate_bar_reduction_l2841_284123


namespace NUMINAMATH_CALUDE_min_value_expression_l2841_284168

theorem min_value_expression (a b : ℝ) (h1 : a > b) (h2 : b > 0) :
  a^2 + 1 / (b * (a - b)) ≥ 5 := by
  sorry

end NUMINAMATH_CALUDE_min_value_expression_l2841_284168


namespace NUMINAMATH_CALUDE_vacation_cost_division_l2841_284173

theorem vacation_cost_division (total_cost : ℕ) (initial_people : ℕ) (cost_reduction : ℕ) (n : ℕ) : 
  total_cost = 360 →
  initial_people = 3 →
  (total_cost / initial_people) - (total_cost / n) = cost_reduction →
  cost_reduction = 30 →
  n = 4 :=
by sorry

end NUMINAMATH_CALUDE_vacation_cost_division_l2841_284173


namespace NUMINAMATH_CALUDE_ginger_flower_sales_l2841_284139

/-- Represents the number of flowers sold of each type -/
structure FlowerSales where
  lilacs : ℕ
  roses : ℕ
  gardenias : ℕ

/-- Calculates the total number of flowers sold -/
def totalFlowers (sales : FlowerSales) : ℕ :=
  sales.lilacs + sales.roses + sales.gardenias

/-- Theorem: Given the conditions of Ginger's flower sales, the total number of flowers sold is 45 -/
theorem ginger_flower_sales :
  ∀ (sales : FlowerSales),
    sales.lilacs = 10 →
    sales.roses = 3 * sales.lilacs →
    sales.gardenias = sales.lilacs / 2 →
    totalFlowers sales = 45 := by
  sorry


end NUMINAMATH_CALUDE_ginger_flower_sales_l2841_284139


namespace NUMINAMATH_CALUDE_parallel_vectors_x_value_l2841_284134

/-- Given two 2D vectors a and b, where a = (2, -1) and b = (-4, x),
    if a and b are parallel, then x = 2. -/
theorem parallel_vectors_x_value (x : ℝ) :
  let a : Fin 2 → ℝ := ![2, -1]
  let b : Fin 2 → ℝ := ![-4, x]
  (∃ (k : ℝ), k ≠ 0 ∧ b = k • a) →
  x = 2 := by
sorry

end NUMINAMATH_CALUDE_parallel_vectors_x_value_l2841_284134


namespace NUMINAMATH_CALUDE_man_speed_in_still_water_l2841_284164

/-- The speed of a man rowing in still water, given his upstream and downstream speeds -/
theorem man_speed_in_still_water (upstream_speed downstream_speed : ℝ) 
  (h1 : upstream_speed = 34)
  (h2 : downstream_speed = 48) :
  (upstream_speed + downstream_speed) / 2 = 41 := by
  sorry

end NUMINAMATH_CALUDE_man_speed_in_still_water_l2841_284164


namespace NUMINAMATH_CALUDE_laptop_sticker_price_is_750_l2841_284156

/-- The sticker price of the laptop -/
def sticker_price : ℝ := 750

/-- Store A's pricing strategy -/
def store_A_price (x : ℝ) : ℝ := 0.80 * x - 100

/-- Store B's pricing strategy -/
def store_B_price (x : ℝ) : ℝ := 0.70 * x

/-- The theorem stating that the sticker price is correct -/
theorem laptop_sticker_price_is_750 :
  store_B_price sticker_price - store_A_price sticker_price = 25 := by
  sorry

end NUMINAMATH_CALUDE_laptop_sticker_price_is_750_l2841_284156


namespace NUMINAMATH_CALUDE_max_large_chips_l2841_284121

theorem max_large_chips (total : ℕ) (small large : ℕ) (h1 : total = 100) 
  (h2 : total = small + large) (h3 : ∃ p : ℕ, Prime p ∧ Even p ∧ small = large + p) : 
  large ≤ 49 := by
  sorry

end NUMINAMATH_CALUDE_max_large_chips_l2841_284121


namespace NUMINAMATH_CALUDE_semicircle_perimeter_l2841_284151

/-- The perimeter of a semicircle with radius 20 is equal to 20π + 40. -/
theorem semicircle_perimeter : 
  ∀ (r : ℝ), r = 20 → (r * π + r) = 20 * π + 40 := by
  sorry

end NUMINAMATH_CALUDE_semicircle_perimeter_l2841_284151


namespace NUMINAMATH_CALUDE_set_union_problem_l2841_284198

theorem set_union_problem (m : ℝ) : 
  let A : Set ℝ := {1, 2^m}
  let B : Set ℝ := {0, 2}
  A ∪ B = {0, 1, 2, 8} → m = 3 := by
sorry

end NUMINAMATH_CALUDE_set_union_problem_l2841_284198


namespace NUMINAMATH_CALUDE_max_edges_cube_plane_intersection_l2841_284172

/-- A cube is a three-dimensional solid object with six square faces -/
structure Cube where
  -- We don't need to define the internal structure of a cube for this problem

/-- A plane is a flat, two-dimensional surface that extends infinitely far -/
structure Plane where
  -- We don't need to define the internal structure of a plane for this problem

/-- A polygon is a plane figure with straight sides -/
structure Polygon where
  edges : ℕ

/-- The result of intersecting a cube with a plane is a polygon -/
def intersect (c : Cube) (p : Plane) : Polygon :=
  sorry

/-- Theorem: The maximum number of edges in a polygon formed by the intersection of a cube and a plane is 6 -/
theorem max_edges_cube_plane_intersection (c : Cube) (p : Plane) :
  (intersect c p).edges ≤ 6 ∧ ∃ (c' : Cube) (p' : Plane), (intersect c' p').edges = 6 :=
sorry

end NUMINAMATH_CALUDE_max_edges_cube_plane_intersection_l2841_284172


namespace NUMINAMATH_CALUDE_negation_of_conjunction_l2841_284160

theorem negation_of_conjunction (p q : Prop) : ¬(p ∧ q) ↔ (¬p ∨ ¬q) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_conjunction_l2841_284160


namespace NUMINAMATH_CALUDE_symmetric_point_coordinates_l2841_284194

/-- Given two points are symmetric with respect to the origin -/
def symmetric_wrt_origin (A B : ℝ × ℝ) : Prop :=
  B.1 = -A.1 ∧ B.2 = -A.2

theorem symmetric_point_coordinates :
  let A : ℝ × ℝ := (-2, 1)
  let B : ℝ × ℝ := (2, -1)
  symmetric_wrt_origin A B → B = (2, -1) := by
  sorry

end NUMINAMATH_CALUDE_symmetric_point_coordinates_l2841_284194


namespace NUMINAMATH_CALUDE_statements_equivalence_l2841_284113

variable (α : Type)
variable (A B : α → Prop)

theorem statements_equivalence :
  (∀ x, A x → B x) ↔
  (∀ x, A x → B x) ∧
  (∀ x, A x → B x) ∧
  (∀ x, A x → B x) ∧
  (∀ x, ¬ B x → ¬ A x) :=
by sorry

end NUMINAMATH_CALUDE_statements_equivalence_l2841_284113


namespace NUMINAMATH_CALUDE_absolute_value_inequality_l2841_284178

theorem absolute_value_inequality (x a : ℝ) (ha : a > 0) :
  (|x - 3| + |x - 4| + |x - 5| < a) ↔ (a > 4) := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_inequality_l2841_284178


namespace NUMINAMATH_CALUDE_square_sum_given_difference_and_product_l2841_284110

theorem square_sum_given_difference_and_product (x y : ℝ) 
  (h1 : x - y = 12) 
  (h2 : x * y = 9) : 
  x^2 + y^2 = 162 := by
sorry

end NUMINAMATH_CALUDE_square_sum_given_difference_and_product_l2841_284110


namespace NUMINAMATH_CALUDE_sally_coin_problem_l2841_284131

/-- Represents the number and value of coins in Sally's bank -/
structure CoinBank where
  pennies : ℕ
  nickels : ℕ
  pennyValue : ℕ
  nickelValue : ℕ

/-- Calculates the total value of coins in cents -/
def totalValue (bank : CoinBank) : ℕ :=
  bank.pennies * bank.pennyValue + bank.nickels * bank.nickelValue

/-- Represents gifts of nickels -/
structure NickelGift where
  fromDad : ℕ
  fromMom : ℕ

theorem sally_coin_problem (initialBank : CoinBank) (gift : NickelGift) :
  initialBank.pennies = 8 ∧
  initialBank.nickels = 7 ∧
  initialBank.pennyValue = 1 ∧
  initialBank.nickelValue = 5 ∧
  gift.fromDad = 9 ∧
  gift.fromMom = 2 →
  let finalBank : CoinBank := {
    pennies := initialBank.pennies,
    nickels := initialBank.nickels + gift.fromDad + gift.fromMom,
    pennyValue := initialBank.pennyValue,
    nickelValue := initialBank.nickelValue
  }
  finalBank.nickels = 18 ∧ totalValue finalBank = 98 := by
  sorry

end NUMINAMATH_CALUDE_sally_coin_problem_l2841_284131


namespace NUMINAMATH_CALUDE_sphere_tangency_relation_l2841_284192

/-- Given three mutually tangent spheres touching a plane at three points on a circle of radius R,
    and two spheres of radii r and ρ (ρ > r) each tangent to the three given spheres and the plane,
    prove that 1/r - 1/ρ = 2√3/R. -/
theorem sphere_tangency_relation (R r ρ : ℝ) (h1 : r > 0) (h2 : ρ > 0) (h3 : ρ > r) :
  1 / r - 1 / ρ = 2 * Real.sqrt 3 / R :=
by sorry

end NUMINAMATH_CALUDE_sphere_tangency_relation_l2841_284192


namespace NUMINAMATH_CALUDE_parabola_point_distance_l2841_284103

theorem parabola_point_distance (a : ℝ) : 
  (∀ x y : ℝ, y^2 = 4*x → (x - a)^2 + y^2 ≥ a^2) → a ≤ 2 := by
  sorry

end NUMINAMATH_CALUDE_parabola_point_distance_l2841_284103


namespace NUMINAMATH_CALUDE_prob_at_least_two_evens_eq_247_256_l2841_284183

/-- Probability of getting an even number on a single roll of a standard die -/
def p_even : ℚ := 1/2

/-- Number of rolls -/
def n : ℕ := 8

/-- Probability of getting exactly k even numbers in n rolls -/
def prob_k_evens (k : ℕ) : ℚ :=
  (n.choose k) * (p_even ^ k) * ((1 - p_even) ^ (n - k))

/-- Probability of getting at least two even numbers in n rolls -/
def prob_at_least_two_evens : ℚ :=
  1 - (prob_k_evens 0 + prob_k_evens 1)

theorem prob_at_least_two_evens_eq_247_256 :
  prob_at_least_two_evens = 247/256 := by sorry

end NUMINAMATH_CALUDE_prob_at_least_two_evens_eq_247_256_l2841_284183


namespace NUMINAMATH_CALUDE_smallest_four_digit_mod_8_l2841_284166

theorem smallest_four_digit_mod_8 : 
  ∀ n : ℕ, 
    1000 ≤ n ∧ n ≡ 3 [MOD 8] → 
    1003 ≤ n :=
by sorry

end NUMINAMATH_CALUDE_smallest_four_digit_mod_8_l2841_284166


namespace NUMINAMATH_CALUDE_sarah_eli_age_ratio_l2841_284179

/-- Given the ages and relationships between Kaylin, Sarah, Eli, and Freyja, 
    prove that the ratio of Sarah's age to Eli's age is 2:1 -/
theorem sarah_eli_age_ratio :
  ∀ (kaylin_age sarah_age eli_age freyja_age : ℕ),
    kaylin_age = 33 →
    freyja_age = 10 →
    sarah_age = kaylin_age + 5 →
    eli_age = freyja_age + 9 →
    ∃ (n : ℕ), sarah_age = n * eli_age →
    sarah_age / eli_age = 2 := by
  sorry

end NUMINAMATH_CALUDE_sarah_eli_age_ratio_l2841_284179


namespace NUMINAMATH_CALUDE_inequality_proof_l2841_284154

theorem inequality_proof (x y z : ℝ) 
  (h_nonneg_x : 0 ≤ x) (h_nonneg_y : 0 ≤ y) (h_nonneg_z : 0 ≤ z)
  (h_sum : x + y + z = 1) :
  0 ≤ y * z + z * x + x * y - 2 * x * y * z ∧ 
  y * z + z * x + x * y - 2 * x * y * z ≤ 7 / 27 := by
sorry

end NUMINAMATH_CALUDE_inequality_proof_l2841_284154


namespace NUMINAMATH_CALUDE_seating_arrangement_l2841_284191

theorem seating_arrangement (total_people : ℕ) (max_rows : ℕ) 
  (h1 : total_people = 57)
  (h2 : max_rows = 8) : 
  ∃ (rows_with_9 rows_with_6 : ℕ),
    rows_with_9 + rows_with_6 ≤ max_rows ∧
    9 * rows_with_9 + 6 * rows_with_6 = total_people ∧
    rows_with_9 = 5 := by
  sorry

end NUMINAMATH_CALUDE_seating_arrangement_l2841_284191


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l2841_284175

def A : Set ℝ := {-1, 1, 2, 4}
def B : Set ℝ := {x : ℝ | -1 < x ∧ x < 3}

theorem intersection_of_A_and_B : A ∩ B = {1, 2} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l2841_284175


namespace NUMINAMATH_CALUDE_unique_twisty_divisible_by_12_l2841_284159

/-- A function that checks if a number is twisty -/
def is_twisty (n : ℕ) : Prop :=
  ∃ (a b : ℕ), a ≠ b ∧ 
  n = a * 10000 + b * 1000 + a * 100 + b * 10 + a

/-- A function that checks if a number is five digits long -/
def is_five_digit (n : ℕ) : Prop :=
  n ≥ 10000 ∧ n < 100000

/-- The main theorem -/
theorem unique_twisty_divisible_by_12 : 
  ∃! (n : ℕ), is_twisty n ∧ is_five_digit n ∧ n % 12 = 0 :=
sorry

end NUMINAMATH_CALUDE_unique_twisty_divisible_by_12_l2841_284159


namespace NUMINAMATH_CALUDE_spatial_relationships_l2841_284106

/-- Two lines are non-coincident -/
def non_coincident_lines (l m : Line) : Prop := l ≠ m

/-- Two planes are non-coincident -/
def non_coincident_planes (α β : Plane) : Prop := α ≠ β

/-- A line is perpendicular to a plane -/
def line_perp_plane (l : Line) (α : Plane) : Prop := sorry

/-- A line is parallel to a plane -/
def line_parallel_plane (l : Line) (α : Plane) : Prop := sorry

/-- A line is contained in a plane -/
def line_in_plane (l : Line) (α : Plane) : Prop := sorry

/-- Two planes are perpendicular -/
def planes_perp (α β : Plane) : Prop := sorry

/-- Two lines are perpendicular -/
def lines_perp (l m : Line) : Prop := sorry

theorem spatial_relationships (l m : Line) (α β : Plane) 
  (h1 : non_coincident_lines l m) (h2 : non_coincident_planes α β) :
  (lines_perp l m ∧ line_perp_plane l α ∧ line_perp_plane m β → planes_perp α β) ∧
  (line_perp_plane l β ∧ planes_perp α β → line_parallel_plane l α ∨ line_in_plane l α) :=
sorry

end NUMINAMATH_CALUDE_spatial_relationships_l2841_284106


namespace NUMINAMATH_CALUDE_coke_cost_l2841_284108

def cheeseburger_cost : ℚ := 3.65
def milkshake_cost : ℚ := 2
def fries_cost : ℚ := 4
def cookie_cost : ℚ := 0.5
def tax : ℚ := 0.2
def toby_initial : ℚ := 15
def toby_change : ℚ := 7

theorem coke_cost (coke_price : ℚ) : coke_price = 1 := by
  sorry

end NUMINAMATH_CALUDE_coke_cost_l2841_284108


namespace NUMINAMATH_CALUDE_roots_quadratic_equation_l2841_284153

theorem roots_quadratic_equation (a b : ℝ) : 
  (a^2 + a - 2014 = 0) → 
  (b^2 + b - 2014 = 0) → 
  a^2 + 2*a + b = 2013 := by
sorry

end NUMINAMATH_CALUDE_roots_quadratic_equation_l2841_284153


namespace NUMINAMATH_CALUDE_completing_square_sum_l2841_284177

theorem completing_square_sum (a b : ℝ) : 
  (∀ x, x^2 - 4*x = 5 ↔ (x + a)^2 = b) → a + b = 7 := by
  sorry

end NUMINAMATH_CALUDE_completing_square_sum_l2841_284177


namespace NUMINAMATH_CALUDE_faire_percentage_calculation_dirk_faire_percentage_l2841_284145

/-- Calculates the percentage of revenue given to the faire for Dirk's amulet sales --/
theorem faire_percentage_calculation (days : Nat) (amulets_per_day : Nat) 
  (selling_price : Nat) (cost_price : Nat) (final_profit : Nat) : ℚ :=
  let total_amulets := days * amulets_per_day
  let revenue := total_amulets * selling_price
  let total_cost := total_amulets * cost_price
  let profit_before_fee := revenue - total_cost
  let faire_fee := profit_before_fee - final_profit
  (faire_fee : ℚ) / revenue * 100

/-- Proves that Dirk gave 10% of his revenue to the faire --/
theorem dirk_faire_percentage : 
  faire_percentage_calculation 2 25 40 30 300 = 10 := by
  sorry

end NUMINAMATH_CALUDE_faire_percentage_calculation_dirk_faire_percentage_l2841_284145


namespace NUMINAMATH_CALUDE_consecutive_integers_square_sum_l2841_284180

theorem consecutive_integers_square_sum : 
  ∃ (n : ℤ), 
    (n + 1)^2 + (n + 2)^2 = (n - 2)^2 + (n - 1)^2 + n^2 ∧
    n = 12 :=
by sorry

end NUMINAMATH_CALUDE_consecutive_integers_square_sum_l2841_284180


namespace NUMINAMATH_CALUDE_tangent_line_equation_l2841_284126

-- Define the curve
def f (x : ℝ) : ℝ := x^3 - 3*x^2

-- Define the slope of the line parallel to 3x + y = 0
def m : ℝ := -3

-- Define the point of tangency
def a : ℝ := 1
def b : ℝ := f a

-- State the theorem
theorem tangent_line_equation :
  ∃ (c : ℝ), ∀ x y : ℝ,
    (y - b = m * (x - a)) ↔ (y = -3*x + c) :=
sorry

end NUMINAMATH_CALUDE_tangent_line_equation_l2841_284126


namespace NUMINAMATH_CALUDE_yue_bao_scientific_notation_l2841_284196

theorem yue_bao_scientific_notation :
  ∃ (a : ℝ) (n : ℤ), 
    1 ≤ a ∧ a < 10 ∧ 
    (1853 * 1000000000 : ℝ) = a * (10 : ℝ) ^ n ∧
    a = 1.853 ∧ n = 11 := by
  sorry

end NUMINAMATH_CALUDE_yue_bao_scientific_notation_l2841_284196


namespace NUMINAMATH_CALUDE_special_function_properties_l2841_284169

/-- A function satisfying the given properties -/
def special_function (f : ℝ → ℝ) : Prop :=
  (∀ x y : ℝ, f (x + y) = f x + f y) ∧
  (∀ x : ℝ, x < 0 → f x > 0)

/-- Main theorem encapsulating all parts of the problem -/
theorem special_function_properties (f : ℝ → ℝ) (hf : special_function f) :
  (∀ x : ℝ, f (-x) = -f x) ∧
  (∀ x₁ x₂ : ℝ, x₁ < x₂ → f x₁ > f x₂) ∧
  (∀ a x : ℝ, f (x^2) + 3 * f a > 3 * f x + f (a * x) ↔
    (a ≠ 0 ∧ ((a > 3 ∧ 3 < x ∧ x < a) ∨ (a < 3 ∧ a < x ∧ x < 3)))) :=
by sorry

end NUMINAMATH_CALUDE_special_function_properties_l2841_284169


namespace NUMINAMATH_CALUDE_f_properties_l2841_284132

open Real

noncomputable def f (x : ℝ) := Real.log (Real.exp (2 * x) + 1) - x

theorem f_properties :
  (∀ x, ∃ y, f x = y) ∧
  (∀ x, f (-x) = f x) ∧
  (∀ x y, 0 ≤ x ∧ x < y → f x < f y) :=
by sorry

end NUMINAMATH_CALUDE_f_properties_l2841_284132


namespace NUMINAMATH_CALUDE_joels_age_proof_l2841_284167

/-- Joel's current age -/
def joels_current_age : ℕ := 5

/-- Joel's dad's current age -/
def dads_current_age : ℕ := 32

/-- The age Joel will be when his dad is twice his age -/
def joels_future_age : ℕ := 27

theorem joels_age_proof :
  joels_current_age = 5 ∧
  dads_current_age = 32 ∧
  joels_future_age = 27 ∧
  dads_current_age + (joels_future_age - joels_current_age) = 2 * joels_future_age :=
by sorry

end NUMINAMATH_CALUDE_joels_age_proof_l2841_284167


namespace NUMINAMATH_CALUDE_max_pieces_is_seven_l2841_284150

/-- Represents a mapping of letters to digits -/
def LetterDigitMap := Char → Nat

/-- Checks if a mapping is valid (each letter maps to a unique digit) -/
def is_valid_mapping (m : LetterDigitMap) : Prop :=
  ∀ c1 c2, c1 ≠ c2 → m c1 ≠ m c2

/-- Converts a string to a number using the given mapping -/
def string_to_number (s : String) (m : LetterDigitMap) : Nat :=
  s.foldl (fun acc c => acc * 10 + m c) 0

/-- Represents the equation PIE = n * PIECE -/
def satisfies_equation (pie : String) (piece : String) (n : Nat) (m : LetterDigitMap) : Prop :=
  string_to_number pie m = n * string_to_number piece m

theorem max_pieces_is_seven :
  ∃ (pie piece : String) (m : LetterDigitMap),
    pie.length = 5 ∧
    piece.length = 5 ∧
    is_valid_mapping m ∧
    satisfies_equation pie piece 7 m ∧
    (∀ (pie' piece' : String) (m' : LetterDigitMap) (n : Nat),
      pie'.length = 5 →
      piece'.length = 5 →
      is_valid_mapping m' →
      satisfies_equation pie' piece' n m' →
      n ≤ 7) :=
sorry

end NUMINAMATH_CALUDE_max_pieces_is_seven_l2841_284150


namespace NUMINAMATH_CALUDE_sine_inequality_l2841_284127

theorem sine_inequality (x : ℝ) : 
  (9.2894 * Real.sin x * Real.sin (2 * x) * Real.sin (3 * x) > Real.sin (4 * x)) ↔ 
  (∃ n : ℤ, (-π/8 + π * n < x ∧ x < π * n) ∨ 
            (π/8 + π * n < x ∧ x < 3*π/8 + π * n) ∨ 
            (π/2 + π * n < x ∧ x < 5*π/8 + π * n)) := by
  sorry

end NUMINAMATH_CALUDE_sine_inequality_l2841_284127


namespace NUMINAMATH_CALUDE_gcd_of_polynomial_and_multiple_l2841_284111

theorem gcd_of_polynomial_and_multiple : ∀ y : ℤ, 
  9240 ∣ y → 
  Int.gcd ((5*y+3)*(11*y+2)*(17*y+8)*(4*y+7)) y = 168 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_polynomial_and_multiple_l2841_284111


namespace NUMINAMATH_CALUDE_game_board_probability_l2841_284133

/-- Represents a triangle on the game board --/
structure GameTriangle :=
  (is_isosceles_right : Bool)
  (num_subdivisions : Nat)
  (num_shaded : Nat)

/-- Calculates the probability of landing in a shaded region --/
def probability_shaded (t : GameTriangle) : ℚ :=
  t.num_shaded / t.num_subdivisions

/-- The main theorem stating the probability for the specific game board configuration --/
theorem game_board_probability (t : GameTriangle) :
  t.is_isosceles_right = true →
  t.num_subdivisions = 6 →
  t.num_shaded = 2 →
  probability_shaded t = 1/3 := by
  sorry


end NUMINAMATH_CALUDE_game_board_probability_l2841_284133


namespace NUMINAMATH_CALUDE_midpoint_ratio_range_l2841_284148

-- Define the lines
def line1 (x y : ℝ) : Prop := x + 2*y - 1 = 0
def line2 (x y : ℝ) : Prop := x + 2*y + 3 = 0

-- Define the midpoint condition
def is_midpoint (x₀ y₀ xp yp xq yq : ℝ) : Prop :=
  x₀ = (xp + xq) / 2 ∧ y₀ = (yp + yq) / 2

-- Main theorem
theorem midpoint_ratio_range 
  (P : ℝ × ℝ) (A : ℝ × ℝ) (Q : ℝ × ℝ) (x₀ y₀ : ℝ)
  (h1 : line1 P.1 P.2)
  (h2 : line2 A.1 A.2)
  (h3 : is_midpoint x₀ y₀ P.1 P.2 Q.1 Q.2)
  (h4 : y₀ > x₀ + 2) :
  -1/2 < y₀/x₀ ∧ y₀/x₀ < -1/5 :=
sorry

end NUMINAMATH_CALUDE_midpoint_ratio_range_l2841_284148


namespace NUMINAMATH_CALUDE_midpoint_between_fractions_l2841_284197

theorem midpoint_between_fractions :
  (1 / 12 + 1 / 20) / 2 = 1 / 15 := by
  sorry

end NUMINAMATH_CALUDE_midpoint_between_fractions_l2841_284197


namespace NUMINAMATH_CALUDE_simple_interest_problem_l2841_284137

/-- Given a principal amount P and an unknown interest rate R,
    if increasing the rate by 1% results in Rs. 72 more interest over 3 years,
    then P must be Rs. 2400. -/
theorem simple_interest_problem (P R : ℝ) : 
  (P * (R + 1) * 3) / 100 - (P * R * 3) / 100 = 72 → P = 2400 := by
  sorry

end NUMINAMATH_CALUDE_simple_interest_problem_l2841_284137


namespace NUMINAMATH_CALUDE_rectangle_tiling_existence_l2841_284155

/-- A rectangle is represented by its width and height -/
structure Rectangle where
  width : ℕ
  height : ℕ

/-- Represents a tiling of a rectangle using smaller rectangles -/
def CanTile (r : Rectangle) (tiles : List Rectangle) : Prop :=
  sorry

/-- The main theorem: there exists an N such that all rectangles with sides > N can be tiled -/
theorem rectangle_tiling_existence : 
  ∃ N : ℕ, ∀ m n : ℕ, m > N → n > N → 
    CanTile ⟨m, n⟩ [⟨4, 6⟩, ⟨5, 7⟩] :=
  sorry

end NUMINAMATH_CALUDE_rectangle_tiling_existence_l2841_284155


namespace NUMINAMATH_CALUDE_exp_greater_equal_linear_l2841_284128

theorem exp_greater_equal_linear : ∀ x : ℝ, Real.exp x ≥ Real.exp 1 * x := by sorry

end NUMINAMATH_CALUDE_exp_greater_equal_linear_l2841_284128


namespace NUMINAMATH_CALUDE_shadow_boundary_is_constant_l2841_284115

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a sphere in 3D space -/
structure Sphere where
  center : Point3D
  radius : ℝ

/-- The equation of the shadow boundary for a sphere -/
def shadowBoundary (s : Sphere) (lightSource : Point3D) : ℝ → ℝ := fun x => -2

/-- Theorem stating that the shadow boundary is y = -2 for the given sphere and light source -/
theorem shadow_boundary_is_constant (s : Sphere) (lightSource : Point3D) :
  s.center = Point3D.mk 0 0 2 →
  s.radius = 2 →
  lightSource = Point3D.mk 0 1 3 →
  ∀ x, shadowBoundary s lightSource x = -2 := by
  sorry

#check shadow_boundary_is_constant

end NUMINAMATH_CALUDE_shadow_boundary_is_constant_l2841_284115


namespace NUMINAMATH_CALUDE_expression_simplification_l2841_284100

theorem expression_simplification (b y : ℝ) (hb : b > 0) (hy : y > 0) :
  (Real.sqrt (b^2 + y^2) - (y^2 - b^2) / Real.sqrt (b^2 + y^2)) / (b + y^2) = 2 * b^2 / (b + y^2) := by
sorry

end NUMINAMATH_CALUDE_expression_simplification_l2841_284100


namespace NUMINAMATH_CALUDE_height_of_pillar_D_l2841_284117

/-- Regular hexagon with pillars -/
structure HexagonWithPillars where
  -- Side length of the hexagon
  side_length : ℝ
  -- Heights of pillars at A, B, C
  height_A : ℝ
  height_B : ℝ
  height_C : ℝ

/-- Theorem: Height of pillar at D in a regular hexagon with given pillar heights -/
theorem height_of_pillar_D (h : HexagonWithPillars) 
  (h_side : h.side_length = 10)
  (h_A : h.height_A = 8)
  (h_B : h.height_B = 11)
  (h_C : h.height_C = 12) : 
  ∃ (z : ℝ), z = 5 ∧ 
  ((-15 * Real.sqrt 3) * (-10) + 20 * 0 + (50 * Real.sqrt 3) * z = 400 * Real.sqrt 3) := by
  sorry

#check height_of_pillar_D

end NUMINAMATH_CALUDE_height_of_pillar_D_l2841_284117


namespace NUMINAMATH_CALUDE_fraction_meaningful_l2841_284174

theorem fraction_meaningful (x : ℝ) : 
  (∃ y : ℝ, y = x / (x - 1)) ↔ x ≠ 1 :=
sorry

end NUMINAMATH_CALUDE_fraction_meaningful_l2841_284174


namespace NUMINAMATH_CALUDE_line_BM_equation_angle_ABM_equals_ABN_l2841_284118

-- Define the parabola C
def C (x y : ℝ) : Prop := y^2 = 2*x

-- Define points A and B
def A : ℝ × ℝ := (2, 0)
def B : ℝ × ℝ := (-2, 0)

-- Define a line l passing through A
def l (t : ℝ) (x y : ℝ) : Prop := x = t*y + 2

-- Define points M and N as intersections of l and C
def M (t : ℝ) : ℝ × ℝ := sorry
def N (t : ℝ) : ℝ × ℝ := sorry

-- Theorem 1: When l is perpendicular to x-axis, equation of BM
theorem line_BM_equation (t : ℝ) : 
  t = 0 → (
    let (x₁, y₁) := M t
    (x₁ - 2*y₁ + 2 = 0) ∨ (x₁ + 2*y₁ + 2 = 0)
  ) := by sorry

-- Theorem 2: ∠ABM = ∠ABN for any line l
theorem angle_ABM_equals_ABN (t : ℝ) :
  let (x₁, y₁) := M t
  let (x₂, y₂) := N t
  (y₁ / (x₁ + 2)) + (y₂ / (x₂ + 2)) = 0 := by sorry

end NUMINAMATH_CALUDE_line_BM_equation_angle_ABM_equals_ABN_l2841_284118


namespace NUMINAMATH_CALUDE_projectile_meeting_time_l2841_284109

/-- Given two objects traveling towards each other, calculate the time it takes for them to meet. -/
theorem projectile_meeting_time 
  (initial_distance : ℝ) 
  (speed1 : ℝ) 
  (speed2 : ℝ) 
  (h1 : initial_distance = 1182) 
  (h2 : speed1 = 460) 
  (h3 : speed2 = 525) : 
  (initial_distance / (speed1 + speed2)) * 60 = 72 := by
  sorry

end NUMINAMATH_CALUDE_projectile_meeting_time_l2841_284109


namespace NUMINAMATH_CALUDE_swim_time_ratio_l2841_284195

/-- The ratio of time taken to swim upstream to downstream -/
theorem swim_time_ratio (v_m : ℝ) (v_s : ℝ) (h1 : v_m = 4.5) (h2 : v_s = 1.5) :
  (v_m + v_s) / (v_m - v_s) = 2 := by
  sorry

#check swim_time_ratio

end NUMINAMATH_CALUDE_swim_time_ratio_l2841_284195


namespace NUMINAMATH_CALUDE_ice_cream_scoop_arrangements_l2841_284141

theorem ice_cream_scoop_arrangements :
  (Finset.univ.filter (fun σ : Equiv.Perm (Fin 5) => true)).card = 120 := by
  sorry

end NUMINAMATH_CALUDE_ice_cream_scoop_arrangements_l2841_284141


namespace NUMINAMATH_CALUDE_sum_of_roots_l2841_284144

theorem sum_of_roots (c d : ℝ) 
  (hc : c^3 - 12*c^2 + 15*c - 36 = 0) 
  (hd : 6*d^3 - 36*d^2 - 150*d + 1350 = 0) : 
  c + d = 7 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_roots_l2841_284144


namespace NUMINAMATH_CALUDE_stock_price_change_l2841_284190

theorem stock_price_change (initial_price : ℝ) (initial_price_positive : initial_price > 0) :
  let price_after_decrease := initial_price * (1 - 0.08)
  let final_price := price_after_decrease * (1 + 0.10)
  let net_change_percentage := (final_price - initial_price) / initial_price * 100
  net_change_percentage = 1.2 := by
sorry

end NUMINAMATH_CALUDE_stock_price_change_l2841_284190


namespace NUMINAMATH_CALUDE_sum_second_largest_second_smallest_l2841_284101

/-- A function that generates all valid three-digit numbers using digits 0 to 9 (each digit used only once) -/
def generateNumbers : Finset Nat := sorry

/-- The second smallest number in the set of generated numbers -/
def secondSmallest : Nat := sorry

/-- The second largest number in the set of generated numbers -/
def secondLargest : Nat := sorry

/-- Theorem stating that the sum of the second largest and second smallest numbers is 1089 -/
theorem sum_second_largest_second_smallest :
  secondLargest + secondSmallest = 1089 := by sorry

end NUMINAMATH_CALUDE_sum_second_largest_second_smallest_l2841_284101


namespace NUMINAMATH_CALUDE_sum_lent_is_350_l2841_284130

/-- Proves that the sum lent is 350 Rs. given the specified conditions --/
theorem sum_lent_is_350 (P : ℚ) : 
  (∀ (I : ℚ), I = P * (4 : ℚ) * (8 : ℚ) / 100) →  -- Simple interest formula
  (∀ (I : ℚ), I = P - 238) →                      -- Interest is 238 less than principal
  P = 350 := by
  sorry

end NUMINAMATH_CALUDE_sum_lent_is_350_l2841_284130


namespace NUMINAMATH_CALUDE_average_age_of_joans_kittens_l2841_284140

/-- Represents the number of days in each month (simplified to 30 for all months) -/
def daysInMonth : ℕ := 30

/-- Calculates the age of kittens in days given their birth month -/
def kittenAge (birthMonth : ℕ) : ℕ :=
  (4 - birthMonth) * daysInMonth + 15

/-- Represents Joan's original number of kittens -/
def joansOriginalKittens : ℕ := 8

/-- Represents the number of kittens Joan gave away -/
def joansGivenAwayKittens : ℕ := 2

/-- Represents the number of neighbor's kittens Joan adopted -/
def adoptedNeighborKittens : ℕ := 3

/-- Represents the number of friend's kittens Joan adopted -/
def adoptedFriendKittens : ℕ := 1

/-- Calculates the total number of kittens Joan has after all transactions -/
def totalJoansKittens : ℕ :=
  joansOriginalKittens - joansGivenAwayKittens + adoptedNeighborKittens + adoptedFriendKittens

/-- Theorem stating that the average age of Joan's kittens on April 15th is 90 days -/
theorem average_age_of_joans_kittens :
  (joansOriginalKittens - joansGivenAwayKittens) * kittenAge 1 +
  adoptedNeighborKittens * kittenAge 2 +
  adoptedFriendKittens * kittenAge 3 =
  90 * totalJoansKittens := by sorry

end NUMINAMATH_CALUDE_average_age_of_joans_kittens_l2841_284140


namespace NUMINAMATH_CALUDE_problem_solution_l2841_284165

theorem problem_solution (m n : ℝ) 
  (h1 : m^2 + 2*m*n = 384) 
  (h2 : 3*m*n + 2*n^2 = 560) : 
  2*m^2 + 13*m*n + 6*n^2 - 444 = 2004 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l2841_284165


namespace NUMINAMATH_CALUDE_units_digit_factorial_sum_2010_l2841_284163

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

def units_digit (n : ℕ) : ℕ := n % 10

def factorial_sum (n : ℕ) : ℕ := (List.range n).map factorial |>.sum

theorem units_digit_factorial_sum_2010 :
  units_digit (factorial_sum 2010) = 3 := by sorry

end NUMINAMATH_CALUDE_units_digit_factorial_sum_2010_l2841_284163


namespace NUMINAMATH_CALUDE_larger_integer_problem_l2841_284107

theorem larger_integer_problem (x y : ℕ) (h1 : 0 < x) (h2 : 0 < y) (h3 : x < y) (h4 : y - x = 8) (h5 : x * y = 272) : y = 17 := by
  sorry

end NUMINAMATH_CALUDE_larger_integer_problem_l2841_284107


namespace NUMINAMATH_CALUDE_angle_alpha_properties_l2841_284116

def angle_alpha (α : Real) : Prop :=
  ∃ (x y : Real), x = 1 ∧ y = Real.sqrt 3 ∧ x = Real.cos α ∧ y = Real.sin α

theorem angle_alpha_properties (α : Real) (h : angle_alpha α) :
  (Real.sin (π - α) - Real.sin (π / 2 + α) = (Real.sqrt 3 - 1) / 2) ∧
  (∃ k : ℤ, α = 2 * π * (k : Real) + π / 3) :=
sorry

end NUMINAMATH_CALUDE_angle_alpha_properties_l2841_284116


namespace NUMINAMATH_CALUDE_remainder_67_power_67_plus_67_mod_68_l2841_284182

theorem remainder_67_power_67_plus_67_mod_68 : (67^67 + 67) % 68 = 66 := by
  sorry

end NUMINAMATH_CALUDE_remainder_67_power_67_plus_67_mod_68_l2841_284182


namespace NUMINAMATH_CALUDE_tea_milk_mixture_l2841_284104

/-- Represents a cup with a certain amount of liquid -/
structure Cup where
  tea : ℚ
  milk : ℚ

/-- The problem setup and solution -/
theorem tea_milk_mixture : 
  let initial_cup1 : Cup := { tea := 6, milk := 0 }
  let initial_cup2 : Cup := { tea := 0, milk := 6 }
  let cup_size : ℚ := 12

  -- Transfer 1/3 of tea from Cup 1 to Cup 2
  let transfer1_amount : ℚ := initial_cup1.tea / 3
  let after_transfer1_cup1 : Cup := { tea := initial_cup1.tea - transfer1_amount, milk := initial_cup1.milk }
  let after_transfer1_cup2 : Cup := { tea := initial_cup2.tea + transfer1_amount, milk := initial_cup2.milk }

  -- Transfer 1/4 of mixture from Cup 2 back to Cup 1
  let total_liquid_cup2 : ℚ := after_transfer1_cup2.tea + after_transfer1_cup2.milk
  let transfer2_amount : ℚ := total_liquid_cup2 / 4
  let tea_ratio_cup2 : ℚ := after_transfer1_cup2.tea / total_liquid_cup2
  let milk_ratio_cup2 : ℚ := after_transfer1_cup2.milk / total_liquid_cup2
  let final_cup1 : Cup := {
    tea := after_transfer1_cup1.tea + transfer2_amount * tea_ratio_cup2,
    milk := after_transfer1_cup1.milk + transfer2_amount * milk_ratio_cup2
  }

  -- The fraction of milk in Cup 1 at the end
  let milk_fraction : ℚ := final_cup1.milk / (final_cup1.tea + final_cup1.milk)

  milk_fraction = 1/4 := by sorry

end NUMINAMATH_CALUDE_tea_milk_mixture_l2841_284104


namespace NUMINAMATH_CALUDE_g_inverse_equals_g_l2841_284188

noncomputable def g (k : ℝ) (x : ℝ) : ℝ := (3 * x + 4) / (k * x - 3)

theorem g_inverse_equals_g (k : ℝ) :
  k ≠ -4/3 →
  ∀ x : ℝ, g k (g k x) = x :=
sorry

end NUMINAMATH_CALUDE_g_inverse_equals_g_l2841_284188


namespace NUMINAMATH_CALUDE_solve_temperature_problem_l2841_284136

def temperature_problem (temps : List ℝ) (avg : ℝ) : Prop :=
  temps.length = 6 ∧
  temps = [99.1, 98.2, 98.7, 99.3, 99.8, 99] ∧
  avg = 99 ∧
  ∃ (saturday_temp : ℝ),
    (temps.sum + saturday_temp) / 7 = avg ∧
    saturday_temp = 98.9

theorem solve_temperature_problem (temps : List ℝ) (avg : ℝ)
  (h : temperature_problem temps avg) : 
  ∃ (saturday_temp : ℝ), saturday_temp = 98.9 := by
  sorry

end NUMINAMATH_CALUDE_solve_temperature_problem_l2841_284136


namespace NUMINAMATH_CALUDE_candy_making_time_l2841_284161

/-- Candy-making process time calculation -/
theorem candy_making_time
  (initial_temp : ℝ)
  (target_temp : ℝ)
  (final_temp : ℝ)
  (heating_rate : ℝ)
  (cooling_rate : ℝ)
  (h1 : initial_temp = 60)
  (h2 : target_temp = 240)
  (h3 : final_temp = 170)
  (h4 : heating_rate = 5)
  (h5 : cooling_rate = 7) :
  (target_temp - initial_temp) / heating_rate + (target_temp - final_temp) / cooling_rate = 46 :=
by sorry

end NUMINAMATH_CALUDE_candy_making_time_l2841_284161


namespace NUMINAMATH_CALUDE_group_selection_count_l2841_284114

theorem group_selection_count : 
  let total_students : ℕ := 7
  let male_students : ℕ := 4
  let female_students : ℕ := 3
  let group_size : ℕ := 3
  (Nat.choose total_students group_size) - 
  (Nat.choose male_students group_size) - 
  (Nat.choose female_students group_size) = 30 := by
sorry

end NUMINAMATH_CALUDE_group_selection_count_l2841_284114


namespace NUMINAMATH_CALUDE_percentage_increase_decrease_l2841_284171

theorem percentage_increase_decrease (α β p q : ℝ) 
  (h_pos_α : α > 0) (h_pos_β : β > 0) (h_pos_p : p > 0) (h_pos_q : q > 0) (h_q_lt_50 : q < 50) :
  (α * β * (1 + p / 100) * (1 - q / 100) > α * β) ↔ (p > 100 * q / (100 - q)) :=
by sorry

end NUMINAMATH_CALUDE_percentage_increase_decrease_l2841_284171


namespace NUMINAMATH_CALUDE_all_analogies_correct_correct_analogies_count_l2841_284142

-- Define the structure for a hyperbola
structure Hyperbola where
  focal_length : ℝ
  real_axis_length : ℝ
  eccentricity : ℝ

-- Define the structure for an ellipse
structure Ellipse where
  focal_length : ℝ
  major_axis_length : ℝ
  eccentricity : ℝ

-- Define the structure for an arithmetic sequence
structure ArithmeticSequence where
  first_term : ℝ
  second_term : ℝ
  third_term : ℝ

-- Define the structure for a geometric sequence
structure GeometricSequence where
  first_term : ℝ
  second_term : ℝ
  third_term : ℝ

-- Define the structure for an equilateral triangle
structure EquilateralTriangle where
  side_length : ℝ
  area : ℝ

-- Define the structure for a regular tetrahedron
structure RegularTetrahedron where
  edge_length : ℝ
  volume : ℝ

def analogy1_correct (h : Hyperbola) (e : Ellipse) : Prop :=
  (h.focal_length = 2 * h.real_axis_length → h.eccentricity = 2) →
  (e.focal_length = 1/2 * e.major_axis_length → e.eccentricity = 1/2)

def analogy2_correct (a : ArithmeticSequence) (g : GeometricSequence) : Prop :=
  (a.first_term + a.second_term + a.third_term = 1 → a.second_term = 1/3) →
  (g.first_term * g.second_term * g.third_term = 1 → g.second_term = 1)

def analogy3_correct (t1 t2 : EquilateralTriangle) (tet1 tet2 : RegularTetrahedron) : Prop :=
  (t2.side_length = 2 * t1.side_length → t2.area = 4 * t1.area) →
  (tet2.edge_length = 2 * tet1.edge_length → tet2.volume = 8 * tet1.volume)

theorem all_analogies_correct 
  (h : Hyperbola) (e : Ellipse) 
  (a : ArithmeticSequence) (g : GeometricSequence)
  (t1 t2 : EquilateralTriangle) (tet1 tet2 : RegularTetrahedron) : 
  analogy1_correct h e ∧ analogy2_correct a g ∧ analogy3_correct t1 t2 tet1 tet2 := by
  sorry

theorem correct_analogies_count : ∃ (n : ℕ), n = 3 ∧ 
  ∀ (h : Hyperbola) (e : Ellipse) 
     (a : ArithmeticSequence) (g : GeometricSequence)
     (t1 t2 : EquilateralTriangle) (tet1 tet2 : RegularTetrahedron),
  (analogy1_correct h e → n ≥ 1) ∧
  (analogy2_correct a g → n ≥ 2) ∧
  (analogy3_correct t1 t2 tet1 tet2 → n = 3) := by
  sorry

end NUMINAMATH_CALUDE_all_analogies_correct_correct_analogies_count_l2841_284142


namespace NUMINAMATH_CALUDE_smallest_next_divisor_after_437_l2841_284185

theorem smallest_next_divisor_after_437 (m : ℕ) (h1 : 1000 ≤ m ∧ m ≤ 9999) 
  (h2 : Even m) (h3 : m % 437 = 0) : 
  (∀ d : ℕ, d ∣ m → 437 < d → d ≥ 874) ∧ 874 ∣ m :=
sorry

end NUMINAMATH_CALUDE_smallest_next_divisor_after_437_l2841_284185


namespace NUMINAMATH_CALUDE_value_of_M_l2841_284184

theorem value_of_M : ∃ M : ℝ, (0.25 * M = 0.55 * 1500) ∧ (M = 3300) := by
  sorry

end NUMINAMATH_CALUDE_value_of_M_l2841_284184


namespace NUMINAMATH_CALUDE_sum_of_reciprocals_roots_l2841_284122

theorem sum_of_reciprocals_roots (x : ℝ) : 
  (x^2 - 13*x + 4 = 0) → 
  (∃ r₁ r₂ : ℝ, r₁ ≠ r₂ ∧ x^2 - 13*x + 4 = (x - r₁) * (x - r₂) ∧ 
    (1 / r₁ + 1 / r₂ = 13 / 4)) :=
by sorry

end NUMINAMATH_CALUDE_sum_of_reciprocals_roots_l2841_284122


namespace NUMINAMATH_CALUDE_distribution_and_points_correct_l2841_284135

/-- Represents a comparison between two tanks -/
structure TankComparison where
  siyan_name : String
  siyan_quality : Nat
  zvezda_quality : Nat
  zvezda_name : String

/-- Calculates the oil distribution and rating points -/
def calculate_distribution_and_points (comparisons : List TankComparison) (oil_quantity : Real) :
  (Real × Real × Nat × Nat) :=
  let process := λ (acc : Real × Real × Nat × Nat) (comp : TankComparison) =>
    let (hv_22, lv_426, siyan_points, zvezda_points) := acc
    let new_hv_22 := hv_22 + 
      (if comp.siyan_quality > 2 then oil_quantity else 0) +
      (if comp.zvezda_quality > 2 then oil_quantity else 0)
    let new_lv_426 := lv_426 + 
      (if comp.siyan_quality ≤ 2 then oil_quantity else 0) +
      (if comp.zvezda_quality ≤ 2 then oil_quantity else 0)
    let new_siyan_points := siyan_points +
      (if comp.siyan_quality > comp.zvezda_quality then 3
       else if comp.siyan_quality = comp.zvezda_quality then 1
       else 0)
    let new_zvezda_points := zvezda_points +
      (if comp.zvezda_quality > comp.siyan_quality then 3
       else if comp.siyan_quality = comp.zvezda_quality then 1
       else 0)
    (new_hv_22, new_lv_426, new_siyan_points, new_zvezda_points)
  comparisons.foldl process (0, 0, 0, 0)

/-- Theorem stating the correctness of the calculation -/
theorem distribution_and_points_correct (comparisons : List TankComparison) (oil_quantity : Real) :
  let (hv_22, lv_426, siyan_points, zvezda_points) := calculate_distribution_and_points comparisons oil_quantity
  (hv_22 ≥ 0 ∧ lv_426 ≥ 0 ∧ 
   hv_22 + lv_426 = oil_quantity * comparisons.length * 2 ∧
   siyan_points + zvezda_points = comparisons.length * 3) :=
by sorry

end NUMINAMATH_CALUDE_distribution_and_points_correct_l2841_284135


namespace NUMINAMATH_CALUDE_number_of_dimes_l2841_284143

/-- Given a total of 11 coins, including 2 nickels and 7 quarters, prove that the number of dimes is 2 -/
theorem number_of_dimes (total : ℕ) (nickels : ℕ) (quarters : ℕ) (h1 : total = 11) (h2 : nickels = 2) (h3 : quarters = 7) :
  total - nickels - quarters = 2 := by
sorry

end NUMINAMATH_CALUDE_number_of_dimes_l2841_284143


namespace NUMINAMATH_CALUDE_quadratic_inequality_max_value_l2841_284138

theorem quadratic_inequality_max_value (a b c : ℝ) (ha : a > 0) :
  (∀ x : ℝ, a * x^2 + b * x + c ≥ 2 * a * x + b) →
  (∃ M : ℝ, M = 2 * Real.sqrt 2 - 2 ∧ 
    ∀ k : ℝ, k * (a^2 + c^2) ≤ b^2 → k ≤ M) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_max_value_l2841_284138


namespace NUMINAMATH_CALUDE_polynomial_multiplication_l2841_284158

theorem polynomial_multiplication (x : ℝ) :
  (3 * x^2 - 4 * x + 5) * (-2 * x^2 + 3 * x - 7) =
  -6 * x^4 + 17 * x^3 - 43 * x^2 + 43 * x - 35 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_multiplication_l2841_284158


namespace NUMINAMATH_CALUDE_trig_abs_sum_diff_ge_one_l2841_284149

theorem trig_abs_sum_diff_ge_one (x : ℝ) : 
  max (|Real.cos x - Real.sin x|) (|Real.sin x + Real.cos x|) ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_trig_abs_sum_diff_ge_one_l2841_284149


namespace NUMINAMATH_CALUDE_expression_evaluation_l2841_284199

theorem expression_evaluation (x y : ℝ) (h : (x - 2)^2 + |y - 3| = 0) :
  ((x - 2*y) * (x + 2*y) - (x - y)^2 + y * (y + 2*x)) / (-2*y) = 2 := by
sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2841_284199


namespace NUMINAMATH_CALUDE_permutation_preserves_lines_l2841_284120

-- Define a type for points in a plane
variable {Point : Type*}

-- Define a permutation of points
variable (f : Point → Point)

-- Define what it means for three points to be collinear
def collinear (A B C : Point) : Prop := sorry

-- Define what it means for three points to lie on a circle
def on_circle (A B C : Point) : Prop := sorry

-- State the theorem
theorem permutation_preserves_lines 
  (h : ∀ A B C : Point, on_circle A B C → on_circle (f A) (f B) (f C)) :
  (∀ A B C : Point, collinear A B C ↔ collinear (f A) (f B) (f C)) :=
sorry

end NUMINAMATH_CALUDE_permutation_preserves_lines_l2841_284120


namespace NUMINAMATH_CALUDE_angle_between_vectors_l2841_284129

theorem angle_between_vectors (a b : ℝ × ℝ) :
  (∀ x y : ℝ, a.1 * x + a.2 * y = 1) →  -- a is a unit vector
  b = (2, 2 * Real.sqrt 3) →           -- b = (2, 2√3)
  a.1 * (2 * a.1 + b.1) + a.2 * (2 * a.2 + b.2) = 0 →  -- a ⟂ (2a + b)
  Real.arccos ((a.1 * b.1 + a.2 * b.2) / (Real.sqrt (a.1^2 + a.2^2) * Real.sqrt (b.1^2 + b.2^2))) = 2 * Real.pi / 3 :=
by sorry

end NUMINAMATH_CALUDE_angle_between_vectors_l2841_284129


namespace NUMINAMATH_CALUDE_intersection_point_k_value_l2841_284162

theorem intersection_point_k_value (x y k : ℝ) : 
  x = -6.3 →
  3 * x + y = k →
  -0.75 * x + y = 25 →
  k = 1.375 := by
sorry

end NUMINAMATH_CALUDE_intersection_point_k_value_l2841_284162


namespace NUMINAMATH_CALUDE_point_M_coordinates_l2841_284112

-- Define the coordinates of points M and N
def M (m : ℝ) : ℝ × ℝ := (4*m + 4, 3*m - 6)
def N : ℝ × ℝ := (-8, 12)

-- Define the condition for MN being parallel to x-axis
def parallel_to_x_axis (M N : ℝ × ℝ) : Prop := M.2 = N.2

-- Theorem statement
theorem point_M_coordinates :
  ∃ m : ℝ, parallel_to_x_axis (M m) N ∧ M m = (28, 12) := by
  sorry

end NUMINAMATH_CALUDE_point_M_coordinates_l2841_284112


namespace NUMINAMATH_CALUDE_inequality_proof_l2841_284125

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (sum_one : a + b + c = 1) :
  (a/b + a/c + b/a + b/c + c/a + c/b + 6) ≥ 2 * Real.sqrt 2 * (Real.sqrt ((1-a)/a) + Real.sqrt ((1-b)/b) + Real.sqrt ((1-c)/c)) ∧
  ((a/b + a/c + b/a + b/c + c/a + c/b + 6) = 2 * Real.sqrt 2 * (Real.sqrt ((1-a)/a) + Real.sqrt ((1-b)/b) + Real.sqrt ((1-c)/c)) ↔ a = 1/3 ∧ b = 1/3 ∧ c = 1/3) :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_l2841_284125


namespace NUMINAMATH_CALUDE_problem_solution_l2841_284176

theorem problem_solution : 
  (∃ n : ℕ, 25 = 5 * n) ∧ 
  (∃ m : ℕ, 209 = 19 * m) ∧ ¬(∃ k : ℕ, 63 = 19 * k) ∧
  (∃ p : ℕ, 180 = 9 * p) := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l2841_284176


namespace NUMINAMATH_CALUDE_morning_campers_l2841_284187

theorem morning_campers (afternoon evening total : ℕ) 
  (h1 : afternoon = 13)
  (h2 : evening = 49)
  (h3 : total = 98)
  : total - afternoon - evening = 36 := by
  sorry

end NUMINAMATH_CALUDE_morning_campers_l2841_284187


namespace NUMINAMATH_CALUDE_emilys_average_speed_l2841_284105

-- Define the parameters of Emily's trip
def distance1 : ℝ := 450  -- miles
def time1 : ℝ := 7.5      -- hours (7 hours 30 minutes)
def break_time : ℝ := 1   -- hour
def distance2 : ℝ := 540  -- miles
def time2 : ℝ := 8        -- hours

-- Define the total distance and time
def total_distance : ℝ := distance1 + distance2
def total_time : ℝ := time1 + break_time + time2

-- Theorem to prove
theorem emilys_average_speed :
  total_distance / total_time = 60 := by sorry

end NUMINAMATH_CALUDE_emilys_average_speed_l2841_284105


namespace NUMINAMATH_CALUDE_point_in_second_quadrant_l2841_284186

theorem point_in_second_quadrant (A B C : Real) (h1 : 0 < A ∧ A < π/2) 
  (h2 : 0 < B ∧ B < π/2) (h3 : 0 < C ∧ C < π/2) (h4 : A + B + C = π) :
  let P : ℝ × ℝ := (Real.cos B - Real.sin A, Real.sin B - Real.cos A)
  (P.1 < 0 ∧ P.2 > 0) := by sorry

end NUMINAMATH_CALUDE_point_in_second_quadrant_l2841_284186


namespace NUMINAMATH_CALUDE_cube_equation_solution_l2841_284119

theorem cube_equation_solution : ∃ (N : ℕ), N > 0 ∧ 26^3 * 65^3 = 10^3 * N^3 ∧ N = 169 := by
  sorry

end NUMINAMATH_CALUDE_cube_equation_solution_l2841_284119


namespace NUMINAMATH_CALUDE_remainder_444_pow_444_mod_13_l2841_284157

theorem remainder_444_pow_444_mod_13 : 444^444 ≡ 1 [ZMOD 13] := by
  sorry

end NUMINAMATH_CALUDE_remainder_444_pow_444_mod_13_l2841_284157


namespace NUMINAMATH_CALUDE_kylie_piggy_bank_coins_l2841_284146

/-- The number of coins Kylie got from her piggy bank -/
def piggy_bank_coins : ℕ := 15

/-- The number of coins Kylie got from her brother -/
def brother_coins : ℕ := 13

/-- The number of coins Kylie got from her father -/
def father_coins : ℕ := 8

/-- The number of coins Kylie gave to her friend Laura -/
def coins_given_away : ℕ := 21

/-- The number of coins Kylie was left with -/
def coins_left : ℕ := 15

/-- Theorem stating that the number of coins Kylie got from her piggy bank is 15 -/
theorem kylie_piggy_bank_coins :
  piggy_bank_coins = coins_left + coins_given_away - brother_coins - father_coins :=
by sorry

end NUMINAMATH_CALUDE_kylie_piggy_bank_coins_l2841_284146


namespace NUMINAMATH_CALUDE_min_alterations_for_equal_sum_l2841_284152

def initial_matrix : Matrix (Fin 3) (Fin 3) ℕ := !![1,2,3; 4,5,6; 7,8,9]

def row_sum (M : Matrix (Fin 3) (Fin 3) ℕ) (i : Fin 3) : ℕ :=
  M i 0 + M i 1 + M i 2

def col_sum (M : Matrix (Fin 3) (Fin 3) ℕ) (j : Fin 3) : ℕ :=
  M 0 j + M 1 j + M 2 j

def all_sums_different (M : Matrix (Fin 3) (Fin 3) ℕ) : Prop :=
  ∀ i j i' j', i ≠ i' ∨ j ≠ j' → row_sum M i ≠ row_sum M j ∧ col_sum M i ≠ col_sum M j'

theorem min_alterations_for_equal_sum :
  all_sums_different initial_matrix ∧
  (∃ M : Matrix (Fin 3) (Fin 3) ℕ, ∃ i j : Fin 3,
    (∀ x y, (M x y ≠ initial_matrix x y) → (x = i ∧ y = j)) ∧
    (∃ r c, row_sum M r = col_sum M c)) ∧
  ¬(∃ r c, row_sum initial_matrix r = col_sum initial_matrix c) :=
by sorry

end NUMINAMATH_CALUDE_min_alterations_for_equal_sum_l2841_284152


namespace NUMINAMATH_CALUDE_triangle_longest_side_l2841_284124

theorem triangle_longest_side (x : ℝ) : 
  let side1 := x^2 + 1
  let side2 := x + 5
  let side3 := 3*x - 1
  (side1 + side2 + side3 = 40) →
  (max side1 (max side2 side3) = 26) :=
by sorry

end NUMINAMATH_CALUDE_triangle_longest_side_l2841_284124


namespace NUMINAMATH_CALUDE_min_value_xy_l2841_284147

theorem min_value_xy (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 2/x + 8/y = 1) :
  x * y ≥ 64 ∧ ∃ (x₀ y₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ 2/x₀ + 8/y₀ = 1 ∧ x₀ * y₀ = 64 :=
by sorry

end NUMINAMATH_CALUDE_min_value_xy_l2841_284147
