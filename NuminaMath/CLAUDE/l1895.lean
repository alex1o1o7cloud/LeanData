import Mathlib

namespace NUMINAMATH_CALUDE_bakery_children_count_l1895_189592

theorem bakery_children_count (initial_count : ℕ) (girls_entered : ℕ) (boys_left : ℕ) 
  (h1 : initial_count = 85) (h2 : girls_entered = 24) (h3 : boys_left = 31) :
  initial_count + girls_entered - boys_left = 78 :=
by
  sorry

end NUMINAMATH_CALUDE_bakery_children_count_l1895_189592


namespace NUMINAMATH_CALUDE_polygon_sides_with_120_degree_interior_angles_l1895_189516

theorem polygon_sides_with_120_degree_interior_angles :
  ∀ (n : ℕ) (interior_angle exterior_angle : ℝ),
    interior_angle = 120 →
    exterior_angle = 180 - interior_angle →
    (n : ℝ) * exterior_angle = 360 →
    n = 6 := by
  sorry

end NUMINAMATH_CALUDE_polygon_sides_with_120_degree_interior_angles_l1895_189516


namespace NUMINAMATH_CALUDE_value_of_a_l1895_189596

theorem value_of_a (a b d : ℝ) 
  (h1 : a + b = d) 
  (h2 : b + d = 7) 
  (h3 : d = 4) : 
  a = 1 := by
sorry

end NUMINAMATH_CALUDE_value_of_a_l1895_189596


namespace NUMINAMATH_CALUDE_least_positive_period_is_36_l1895_189514

-- Define the property of the function
def has_property (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (x + 6) + f (x - 6) = f x

-- Define the concept of a period for a function
def is_period (f : ℝ → ℝ) (p : ℝ) : Prop :=
  ∀ x : ℝ, f (x + p) = f x

-- State the theorem
theorem least_positive_period_is_36 (f : ℝ → ℝ) (h : has_property f) :
  (∃ p : ℝ, p > 0 ∧ is_period f p) →
  (∀ q : ℝ, q > 0 → is_period f q → q ≥ 36) ∧ is_period f 36 :=
sorry

end NUMINAMATH_CALUDE_least_positive_period_is_36_l1895_189514


namespace NUMINAMATH_CALUDE_stating_circle_implies_a_eq_neg_one_l1895_189525

/-- 
A function that represents the equation of a potential circle.
-/
def potential_circle (a : ℝ) (x y : ℝ) : ℝ :=
  x^2 + (a + 2) * y^2 + 2 * a * x + a

/-- 
A predicate that determines if an equation represents a circle.
This is a simplified representation and may need to be adjusted based on the specific criteria for a circle.
-/
def is_circle (f : ℝ → ℝ → ℝ) : Prop :=
  ∃ (h k r : ℝ), ∀ (x y : ℝ), f x y = (x - h)^2 + (y - k)^2 - r^2

/-- 
Theorem stating that if the given equation represents a circle, then a = -1.
-/
theorem circle_implies_a_eq_neg_one :
  is_circle (potential_circle a) → a = -1 := by
  sorry


end NUMINAMATH_CALUDE_stating_circle_implies_a_eq_neg_one_l1895_189525


namespace NUMINAMATH_CALUDE_difference_of_squares_l1895_189550

theorem difference_of_squares (x : ℝ) : x^2 - 4 = (x + 2) * (x - 2) := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_l1895_189550


namespace NUMINAMATH_CALUDE_no_special_polyhedron_l1895_189529

/-- A convex polyhedron. -/
structure ConvexPolyhedron where
  vertices : ℕ
  edges : ℕ
  faces : ℕ
  four_faces_share_edges : Bool

/-- Theorem stating that there does not exist a convex polyhedron with the specified properties. -/
theorem no_special_polyhedron :
  ¬ ∃ (p : ConvexPolyhedron), 
    p.vertices = 8 ∧ 
    p.edges = 12 ∧ 
    p.faces = 6 ∧ 
    p.four_faces_share_edges = true :=
by
  sorry

end NUMINAMATH_CALUDE_no_special_polyhedron_l1895_189529


namespace NUMINAMATH_CALUDE_problem_statement_l1895_189555

theorem problem_statement :
  (∃ x₀ : ℝ, Real.tan x₀ = 2) ∧ ¬(∀ x : ℝ, x^2 + 2*x + 1 > 0) := by sorry

end NUMINAMATH_CALUDE_problem_statement_l1895_189555


namespace NUMINAMATH_CALUDE_opposite_of_2023_l1895_189515

theorem opposite_of_2023 : 
  ∀ x : ℤ, (x + 2023 = 0) ↔ (x = -2023) :=
by
  sorry

end NUMINAMATH_CALUDE_opposite_of_2023_l1895_189515


namespace NUMINAMATH_CALUDE_conditional_probability_B_given_A_l1895_189508

/-- Two fair six-sided dice are thrown. -/
def dice_space : Type := Fin 6 × Fin 6

/-- Event A: "the number of points on die A is greater than 4" -/
def event_A : Set dice_space :=
  {x | x.1 > 4}

/-- Event B: "the sum of the number of points on dice A and B is equal to 7" -/
def event_B : Set dice_space :=
  {x | x.1.val + x.2.val = 7}

/-- The probability measure on the dice space -/
def P : Set dice_space → ℝ :=
  sorry

/-- Theorem: The conditional probability P(B|A) = 1/6 -/
theorem conditional_probability_B_given_A :
  P event_B / P event_A = 1 / 6 := by
  sorry

end NUMINAMATH_CALUDE_conditional_probability_B_given_A_l1895_189508


namespace NUMINAMATH_CALUDE_fair_distribution_theorem_l1895_189562

/-- Represents the state of an interrupted game --/
structure GameState where
  total_coins : ℕ
  rounds_to_win : ℕ
  player_a_wins : ℕ
  player_b_wins : ℕ

/-- Calculates the probability of winning for a player --/
def winProbability (state : GameState) (player : Bool) : ℚ :=
  sorry

/-- Calculates the fair coin distribution for a player --/
def fairDistribution (state : GameState) (player : Bool) : ℕ :=
  sorry

/-- Theorem stating the fair distribution of coins in the interrupted game --/
theorem fair_distribution_theorem (state : GameState) :
  state.total_coins = 96 ∧ 
  state.rounds_to_win = 3 ∧ 
  state.player_a_wins = 2 ∧ 
  state.player_b_wins = 1 →
  fairDistribution state true = 72 ∧ 
  fairDistribution state false = 24 :=
sorry

end NUMINAMATH_CALUDE_fair_distribution_theorem_l1895_189562


namespace NUMINAMATH_CALUDE_marty_voters_l1895_189591

theorem marty_voters (total : ℕ) (biff_percent : ℚ) (undecided_percent : ℚ) 
  (h1 : total = 200)
  (h2 : biff_percent = 45 / 100)
  (h3 : undecided_percent = 8 / 100) :
  ⌊(1 - biff_percent - undecided_percent) * total⌋ = 94 := by
  sorry

end NUMINAMATH_CALUDE_marty_voters_l1895_189591


namespace NUMINAMATH_CALUDE_tables_required_l1895_189566

-- Define the base-5 number
def base5_seating : ℕ := 3 * 5^2 + 2 * 5^1 + 1 * 5^0

-- Define the number of people per table
def people_per_table : ℕ := 3

-- Theorem to prove
theorem tables_required :
  (base5_seating + people_per_table - 1) / people_per_table = 29 := by
  sorry

end NUMINAMATH_CALUDE_tables_required_l1895_189566


namespace NUMINAMATH_CALUDE_cost_ratio_when_b_tripled_x_halved_l1895_189549

/-- The cost ratio when b is tripled and x is halved in the formula C = at(bx)^6 -/
theorem cost_ratio_when_b_tripled_x_halved (a t b x : ℝ) :
  let original_cost := a * t * (b * x)^6
  let new_cost := a * t * (3 * b * (x / 2))^6
  (new_cost / original_cost) * 100 = 1139.0625 := by
sorry

end NUMINAMATH_CALUDE_cost_ratio_when_b_tripled_x_halved_l1895_189549


namespace NUMINAMATH_CALUDE_probability_in_range_l1895_189546

/-- 
Given a random variable ξ with probability distribution:
P(ξ=k) = 1/(2^(k-1)) for k = 2, 3, ..., n
P(ξ=1) = a
Prove that P(2 < ξ ≤ 5) = 7/16
-/
theorem probability_in_range (n : ℕ) (a : ℝ) (ξ : ℕ → ℝ) 
  (h1 : ∀ k ∈ Finset.range (n - 1) \ {0}, ξ (k + 2) = (1 : ℝ) / 2^k)
  (h2 : ξ 1 = a) :
  (ξ 3 + ξ 4 + ξ 5) = 7/16 := by
sorry

end NUMINAMATH_CALUDE_probability_in_range_l1895_189546


namespace NUMINAMATH_CALUDE_parallelepiped_vector_sum_l1895_189504

/-- In a parallelepiped ABCD-A₁B₁C₁D₁, if AC₁ = x⋅AB + 2y⋅BC + 3z⋅CC₁, then x + y + z = 11/6 -/
theorem parallelepiped_vector_sum (ABCD_A₁B₁C₁D₁ : Set (EuclideanSpace ℝ (Fin 3)))
  (AB BC CC₁ AC₁ : EuclideanSpace ℝ (Fin 3)) (x y z : ℝ) :
  AC₁ = x • AB + (2 * y) • BC + (3 * z) • CC₁ →
  AC₁ = AB + BC + CC₁ →
  x + y + z = 11 / 6 := by
  sorry

end NUMINAMATH_CALUDE_parallelepiped_vector_sum_l1895_189504


namespace NUMINAMATH_CALUDE_binary_digit_difference_l1895_189535

theorem binary_digit_difference (n m : ℕ) (hn : n = 950) (hm : m = 150) :
  (Nat.log 2 n + 1) - (Nat.log 2 m + 1) = 2 := by
  sorry

end NUMINAMATH_CALUDE_binary_digit_difference_l1895_189535


namespace NUMINAMATH_CALUDE_min_value_f_l1895_189590

theorem min_value_f (x : ℝ) (h : x > 1) : x + 1 / (x - 1) ≥ 3 := by
  sorry

end NUMINAMATH_CALUDE_min_value_f_l1895_189590


namespace NUMINAMATH_CALUDE_sufficient_condition_for_reciprocal_inequality_l1895_189579

theorem sufficient_condition_for_reciprocal_inequality (a b : ℝ) :
  b < a ∧ a < 0 → (1 : ℝ) / a < (1 : ℝ) / b :=
by sorry

end NUMINAMATH_CALUDE_sufficient_condition_for_reciprocal_inequality_l1895_189579


namespace NUMINAMATH_CALUDE_drevlandia_roads_l1895_189519

-- Define the number of cities
def num_cities : ℕ := 101

-- Define the function to calculate the number of roads
def num_roads (n : ℕ) : ℕ := n * (n - 1) / 2

-- Theorem statement
theorem drevlandia_roads : num_roads num_cities = 5050 := by
  sorry

end NUMINAMATH_CALUDE_drevlandia_roads_l1895_189519


namespace NUMINAMATH_CALUDE_min_value_expression_l1895_189512

theorem min_value_expression (x : ℝ) (hx : x > 0) : 
  (x + 5) / Real.sqrt (x + 1) ≥ 4 ∧ 
  ∃ y : ℝ, y > 0 ∧ (y + 5) / Real.sqrt (y + 1) = 4 := by
sorry

end NUMINAMATH_CALUDE_min_value_expression_l1895_189512


namespace NUMINAMATH_CALUDE_pet_store_siamese_cats_l1895_189518

/-- The number of Siamese cats initially in the pet store -/
def initial_siamese_cats : ℕ := 13

/-- The number of house cats initially in the pet store -/
def initial_house_cats : ℕ := 5

/-- The total number of cats sold during the sale -/
def cats_sold : ℕ := 10

/-- The number of cats remaining after the sale -/
def cats_remaining : ℕ := 8

/-- Theorem stating that the initial number of Siamese cats is 13 -/
theorem pet_store_siamese_cats : 
  initial_siamese_cats = 13 ∧ 
  initial_siamese_cats + initial_house_cats = cats_sold + cats_remaining :=
by sorry

end NUMINAMATH_CALUDE_pet_store_siamese_cats_l1895_189518


namespace NUMINAMATH_CALUDE_g_negative_in_range_l1895_189500

def f (a x : ℝ) : ℝ := x^3 + 3*a*x - 1

def g (a x : ℝ) : ℝ := (3*x^2 + 3*a) - a*x - 5

theorem g_negative_in_range :
  ∀ x : ℝ, -2/3 < x → x < 1 →
    ∀ a : ℝ, -1 ≤ a → a ≤ 1 →
      g a x < 0 :=
by sorry

end NUMINAMATH_CALUDE_g_negative_in_range_l1895_189500


namespace NUMINAMATH_CALUDE_min_product_of_geometric_sequence_l1895_189567

theorem min_product_of_geometric_sequence (x y : ℝ) 
  (hx : x > 1) (hy : y > 1) 
  (h_seq : (Real.log x) * (Real.log y) = (1/2)^2) : 
  x * y ≥ Real.exp 1 ∧ ∃ x y, x > 1 ∧ y > 1 ∧ (Real.log x) * (Real.log y) = (1/2)^2 ∧ x * y = Real.exp 1 :=
sorry

end NUMINAMATH_CALUDE_min_product_of_geometric_sequence_l1895_189567


namespace NUMINAMATH_CALUDE_expression_simplification_and_evaluation_l1895_189580

theorem expression_simplification_and_evaluation (x y : ℝ) 
  (hx : x = (1/2)⁻¹) 
  (hy : y = (-2023)^0) : 
  (((2*x - y) / (x + y) - ((x^2 - 2*x*y + y^2) / (x^2 - y^2))) / ((x - y) / (x + y))) = 2 :=
by sorry

end NUMINAMATH_CALUDE_expression_simplification_and_evaluation_l1895_189580


namespace NUMINAMATH_CALUDE_inequalities_equivalence_l1895_189551

theorem inequalities_equivalence (x : ℝ) :
  (2 * (x + 1) - 1 < 3 * x + 2 ↔ x > -1) ∧
  ((x + 3) / 2 - 1 ≥ (2 * x - 3) / 3 ↔ x ≤ 9) :=
by sorry

end NUMINAMATH_CALUDE_inequalities_equivalence_l1895_189551


namespace NUMINAMATH_CALUDE_complex_power_of_four_l1895_189509

theorem complex_power_of_four :
  (3 * Complex.cos (30 * Real.pi / 180) + 3 * Complex.I * Complex.sin (30 * Real.pi / 180)) ^ 4 =
  -40.5 + 40.5 * Complex.I * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_complex_power_of_four_l1895_189509


namespace NUMINAMATH_CALUDE_pizza_varieties_count_l1895_189513

/-- The number of base pizza flavors -/
def base_flavors : ℕ := 4

/-- The number of topping combinations (including no additional toppings) -/
def topping_combinations : ℕ := 4

/-- Calculates the total number of pizza varieties -/
def total_varieties : ℕ := base_flavors * topping_combinations

theorem pizza_varieties_count :
  total_varieties = 16 := by
  sorry

end NUMINAMATH_CALUDE_pizza_varieties_count_l1895_189513


namespace NUMINAMATH_CALUDE_distinct_determinants_count_l1895_189544

-- Define a type for third-order determinants
def ThirdOrderDeterminant := Matrix (Fin 3) (Fin 3) ℝ

-- Define a function to calculate the number of distinct determinants
def distinctDeterminants (n : ℕ) : ℕ :=
  if n = 9 then Nat.factorial 9 / 36 else 0

theorem distinct_determinants_count :
  distinctDeterminants 9 = 10080 := by
  sorry

#eval distinctDeterminants 9

end NUMINAMATH_CALUDE_distinct_determinants_count_l1895_189544


namespace NUMINAMATH_CALUDE_smallest_block_with_399_hidden_cubes_l1895_189586

/-- A rectangular block made of identical cubes -/
structure RectangularBlock where
  length : ℕ
  width : ℕ
  height : ℕ

/-- The number of cubes in a rectangular block -/
def RectangularBlock.volume (b : RectangularBlock) : ℕ :=
  b.length * b.width * b.height

/-- The number of hidden cubes when three faces are visible -/
def RectangularBlock.hiddenCubes (b : RectangularBlock) : ℕ :=
  (b.length - 1) * (b.width - 1) * (b.height - 1)

/-- The theorem stating the smallest possible value of N -/
theorem smallest_block_with_399_hidden_cubes :
  ∀ b : RectangularBlock,
    b.hiddenCubes = 399 →
    b.volume ≥ 640 ∧
    ∃ b' : RectangularBlock, b'.hiddenCubes = 399 ∧ b'.volume = 640 :=
by sorry

end NUMINAMATH_CALUDE_smallest_block_with_399_hidden_cubes_l1895_189586


namespace NUMINAMATH_CALUDE_product_inequality_l1895_189556

theorem product_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a + b) * (b + c) * (c + a) ≥ 8 * a * b * c :=
by sorry

end NUMINAMATH_CALUDE_product_inequality_l1895_189556


namespace NUMINAMATH_CALUDE_completing_square_equivalence_l1895_189545

theorem completing_square_equivalence (x : ℝ) :
  (x^2 - 2*x - 5 = 0) ↔ ((x - 1)^2 = 6) :=
by sorry

end NUMINAMATH_CALUDE_completing_square_equivalence_l1895_189545


namespace NUMINAMATH_CALUDE_min_sum_of_internally_tangent_circles_l1895_189572

/-- Given two circles C₁ and C₂ with equations x² + y² + 2ax + a² - 4 = 0 and x² + y² - 2by - 1 + b² = 0 respectively, 
    where a, b ∈ ℝ, and C₁ and C₂ have only one common tangent line, 
    the minimum value of a + b is -√2. -/
theorem min_sum_of_internally_tangent_circles (a b : ℝ) : 
  (∀ x y : ℝ, x^2 + y^2 + 2*a*x + a^2 - 4 = 0 → x^2 + y^2 - 2*b*y - 1 + b^2 = 0 → False) ∧ 
  (∃ x y : ℝ, x^2 + y^2 + 2*a*x + a^2 - 4 = 0 ∧ x^2 + y^2 - 2*b*y - 1 + b^2 = 0) →
  (a + b ≥ -Real.sqrt 2) ∧ (∃ a₀ b₀ : ℝ, a₀ + b₀ = -Real.sqrt 2) :=
by sorry

end NUMINAMATH_CALUDE_min_sum_of_internally_tangent_circles_l1895_189572


namespace NUMINAMATH_CALUDE_bus_driver_max_regular_hours_l1895_189527

/-- Proves that the maximum number of regular hours is 40 given the conditions --/
theorem bus_driver_max_regular_hours : 
  let regular_rate : ℚ := 16
  let overtime_rate : ℚ := regular_rate * (1 + 3/4)
  let total_compensation : ℚ := 1340
  let total_hours : ℕ := 65
  let max_regular_hours : ℕ := 40
  regular_rate * max_regular_hours + 
  overtime_rate * (total_hours - max_regular_hours) = total_compensation := by
sorry


end NUMINAMATH_CALUDE_bus_driver_max_regular_hours_l1895_189527


namespace NUMINAMATH_CALUDE_rearrangement_methods_l1895_189599

theorem rearrangement_methods (n m k : ℕ) (hn : n = 8) (hm : m = 4) (hk : k = 2) :
  Nat.choose n k = 28 := by
  sorry

end NUMINAMATH_CALUDE_rearrangement_methods_l1895_189599


namespace NUMINAMATH_CALUDE_gardener_hours_per_day_l1895_189506

/-- Calculates the number of hours a gardener works each day given the project details --/
theorem gardener_hours_per_day
  (total_cost : ℕ)
  (num_rose_bushes : ℕ)
  (cost_per_rose_bush : ℕ)
  (gardener_hourly_rate : ℕ)
  (num_work_days : ℕ)
  (soil_volume : ℕ)
  (soil_cost_per_unit : ℕ)
  (h_total_cost : total_cost = 4100)
  (h_num_rose_bushes : num_rose_bushes = 20)
  (h_cost_per_rose_bush : cost_per_rose_bush = 150)
  (h_gardener_hourly_rate : gardener_hourly_rate = 30)
  (h_num_work_days : num_work_days = 4)
  (h_soil_volume : soil_volume = 100)
  (h_soil_cost_per_unit : soil_cost_per_unit = 5) :
  (total_cost - (num_rose_bushes * cost_per_rose_bush + soil_volume * soil_cost_per_unit)) / gardener_hourly_rate / num_work_days = 5 := by
  sorry

end NUMINAMATH_CALUDE_gardener_hours_per_day_l1895_189506


namespace NUMINAMATH_CALUDE_hyperbola_equation_l1895_189539

/-- A hyperbola with specific properties -/
structure Hyperbola where
  conjugate_axis_length : ℝ
  eccentricity : ℝ
  focal_length : ℝ
  point_m : ℝ × ℝ
  point_p : ℝ × ℝ
  point_q : ℝ × ℝ

/-- The standard equation of a hyperbola -/
def standard_equation (h : Hyperbola) (x y : ℝ) : Prop :=
  y^2 / 25 - x^2 / 75 = 1

/-- Theorem stating the standard equation of the specific hyperbola -/
theorem hyperbola_equation (h : Hyperbola)
  (h_conjugate : h.conjugate_axis_length = 12)
  (h_eccentricity : h.eccentricity = 5/4)
  (h_focal : h.focal_length = 26)
  (h_point_m : h.point_m = (0, 12))
  (h_point_p : h.point_p = (-3, 2 * Real.sqrt 7))
  (h_point_q : h.point_q = (-6 * Real.sqrt 2, -7)) :
  ∀ x y, standard_equation h x y ↔ 
    (x = h.point_m.1 ∧ y = h.point_m.2) ∨
    (x = h.point_p.1 ∧ y = h.point_p.2) ∨
    (x = h.point_q.1 ∧ y = h.point_q.2) :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l1895_189539


namespace NUMINAMATH_CALUDE_negative_three_times_five_l1895_189542

theorem negative_three_times_five : (-3 : ℤ) * 5 = -15 := by
  sorry

end NUMINAMATH_CALUDE_negative_three_times_five_l1895_189542


namespace NUMINAMATH_CALUDE_water_balloon_packs_l1895_189593

/-- Represents the number of packs of water balloons --/
def num_own_packs : ℕ := 3

/-- Represents the number of balloons in each pack --/
def balloons_per_pack : ℕ := 6

/-- Represents the number of neighbor's packs used --/
def num_neighbor_packs : ℕ := 2

/-- Represents the extra balloons Milly takes --/
def extra_balloons : ℕ := 7

/-- Represents the number of balloons Floretta is left with --/
def floretta_balloons : ℕ := 8

theorem water_balloon_packs :
  num_own_packs * balloons_per_pack + num_neighbor_packs * balloons_per_pack =
  2 * (floretta_balloons + extra_balloons) :=
sorry

end NUMINAMATH_CALUDE_water_balloon_packs_l1895_189593


namespace NUMINAMATH_CALUDE_corridor_lights_l1895_189502

/-- The number of ways to choose k non-adjacent items from n consecutive items -/
def nonAdjacentChoices (n : ℕ) (k : ℕ) : ℕ :=
  Nat.choose (n - k + 1) k

/-- Theorem: There are 20 ways to choose 3 non-adjacent positions from 8 consecutive positions -/
theorem corridor_lights : nonAdjacentChoices 8 3 = 20 := by
  sorry

#eval nonAdjacentChoices 8 3

end NUMINAMATH_CALUDE_corridor_lights_l1895_189502


namespace NUMINAMATH_CALUDE_central_number_is_14_l1895_189520

/-- Represents a 5x5 grid of natural numbers -/
def Grid := Fin 5 → Fin 5 → ℕ

/-- Checks if two positions in the grid are adjacent -/
def adjacent (p q : Fin 5 × Fin 5) : Prop :=
  (p.1 = q.1 ∧ (p.2 = q.2 + 1 ∨ p.2 + 1 = q.2)) ∨
  (p.2 = q.2 ∧ (p.1 = q.1 + 1 ∨ p.1 + 1 = q.1)) ∨
  ((p.1 = q.1 + 1 ∨ p.1 + 1 = q.1) ∧ (p.2 = q.2 + 1 ∨ p.2 + 1 = q.2))

/-- The main theorem to prove -/
theorem central_number_is_14 (g : Grid) : 
  (∀ i j, g i j ∈ Finset.range 26) →
  (∀ i j k l, (i, j) ≠ (k, l) → g i j ≠ g k l) →
  (∀ i j, g i j ≠ 1 → g i j ≠ 2 → 
    ∃ p q, adjacent (i, j) p ∧ adjacent (i, j) q ∧ g i j = g p.1 p.2 + g q.1 q.2) →
  g 0 0 = 1 →
  g 1 1 = 16 →
  g 1 3 = 18 →
  g 2 0 = 17 →
  g 2 4 = 21 →
  g 3 1 = 23 →
  g 3 3 = 25 →
  g 1 2 = 14 := by
sorry

end NUMINAMATH_CALUDE_central_number_is_14_l1895_189520


namespace NUMINAMATH_CALUDE_candy_distribution_l1895_189533

theorem candy_distribution (total_candies : ℕ) (candies_per_student : ℕ) (num_students : ℕ) :
  total_candies = 81 →
  candies_per_student = 9 →
  total_candies = candies_per_student * num_students →
  num_students = 9 := by
sorry

end NUMINAMATH_CALUDE_candy_distribution_l1895_189533


namespace NUMINAMATH_CALUDE_expand_expression_l1895_189531

theorem expand_expression (x y : ℝ) : 12 * (3 * x + 4 * y + 6) = 36 * x + 48 * y + 72 := by
  sorry

end NUMINAMATH_CALUDE_expand_expression_l1895_189531


namespace NUMINAMATH_CALUDE_equation_solution_l1895_189578

theorem equation_solution : ∃ x : ℝ, (x + 6) / (x - 3) = 4 ∧ x = 6 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1895_189578


namespace NUMINAMATH_CALUDE_arithmetic_sequence_problem_l1895_189536

def arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n, a (n + 1) = a n + d

theorem arithmetic_sequence_problem (a : ℕ → ℝ) (d : ℝ) (m : ℕ) :
  d ≠ 0 →
  arithmetic_sequence a d →
  a 3 + a 6 + a 10 + a 13 = 32 →
  a 8 = 8 →
  a m = 8 →
  m = 8 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_problem_l1895_189536


namespace NUMINAMATH_CALUDE_power_function_through_point_value_l1895_189597

-- Define a power function
def isPowerFunction (f : ℝ → ℝ) : Prop :=
  ∃ α : ℝ, ∀ x : ℝ, f x = x ^ α

-- Define the theorem
theorem power_function_through_point_value :
  ∀ f : ℝ → ℝ,
  isPowerFunction f →
  f 2 = 8 →
  f 3 = 27 := by
sorry

end NUMINAMATH_CALUDE_power_function_through_point_value_l1895_189597


namespace NUMINAMATH_CALUDE_total_students_is_240_l1895_189541

/-- The number of students from Know It All High School -/
def know_it_all_students : ℕ := 50

/-- The number of students from Karen High School -/
def karen_high_students : ℕ := (3 * know_it_all_students) / 5

/-- The combined number of students from Know It All High School and Karen High School -/
def combined_students : ℕ := know_it_all_students + karen_high_students

/-- The number of students from Novel Corona High School -/
def novel_corona_students : ℕ := 2 * combined_students

/-- The total number of students at the competition -/
def total_students : ℕ := know_it_all_students + karen_high_students + novel_corona_students

/-- Theorem stating that the total number of students at the competition is 240 -/
theorem total_students_is_240 : total_students = 240 := by
  sorry

end NUMINAMATH_CALUDE_total_students_is_240_l1895_189541


namespace NUMINAMATH_CALUDE_max_value_quadratic_l1895_189517

theorem max_value_quadratic (x : ℝ) (h : 0 < x ∧ x < 6) :
  (∀ y, 0 < y ∧ y < 6 → (6 - y) * y ≤ (6 - x) * x) → (6 - x) * x = 9 :=
by sorry

end NUMINAMATH_CALUDE_max_value_quadratic_l1895_189517


namespace NUMINAMATH_CALUDE_goldies_hourly_rate_l1895_189583

/-- Goldie's pet-sitting earnings problem -/
theorem goldies_hourly_rate (hours_last_week hours_this_week total_earnings : ℚ) 
  (h1 : hours_last_week = 20)
  (h2 : hours_this_week = 30)
  (h3 : total_earnings = 250) :
  total_earnings / (hours_last_week + hours_this_week) = 5 := by
  sorry

end NUMINAMATH_CALUDE_goldies_hourly_rate_l1895_189583


namespace NUMINAMATH_CALUDE_sufficient_but_not_necessary_l1895_189571

theorem sufficient_but_not_necessary :
  (∀ x : ℝ, x - 1 = 0 → (x - 1) * (x + 2) = 0) ∧
  (∃ x : ℝ, (x - 1) * (x + 2) = 0 ∧ x - 1 ≠ 0) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_but_not_necessary_l1895_189571


namespace NUMINAMATH_CALUDE_prob_at_least_as_many_females_l1895_189524

/-- The probability of selecting at least as many females as males when randomly choosing 2 students from a group of 5 students with 2 females and 3 males is 7/10. -/
theorem prob_at_least_as_many_females (total : ℕ) (females : ℕ) (males : ℕ) :
  total = 5 →
  females = 2 →
  males = 3 →
  females + males = total →
  (Nat.choose total 2 : ℚ) ≠ 0 →
  (Nat.choose females 2 + Nat.choose females 1 * Nat.choose males 1 : ℚ) / Nat.choose total 2 = 7 / 10 := by
  sorry

#check prob_at_least_as_many_females

end NUMINAMATH_CALUDE_prob_at_least_as_many_females_l1895_189524


namespace NUMINAMATH_CALUDE_inequality_and_floor_function_l1895_189548

theorem inequality_and_floor_function (n : ℕ) : 
  (Real.sqrt (n + 1) + 2 * Real.sqrt n < Real.sqrt (9 * n + 3)) ∧ 
  ¬∃ n : ℕ, ⌊Real.sqrt (n + 1) + 2 * Real.sqrt n⌋ < ⌊Real.sqrt (9 * n + 3)⌋ :=
by sorry

end NUMINAMATH_CALUDE_inequality_and_floor_function_l1895_189548


namespace NUMINAMATH_CALUDE_binders_per_student_is_one_l1895_189530

/-- Calculates the number of binders per student given the class size, costs of supplies, and total spent -/
def bindersPerStudent (
  classSize : ℕ) 
  (penCost notebookCost binderCost highlighterCost : ℚ)
  (pensPerStudent notebooksPerStudent highlightersPerStudent : ℕ)
  (teacherDiscount totalSpent : ℚ) : ℚ :=
  let totalPenCost := classSize * pensPerStudent * penCost
  let totalNotebookCost := classSize * notebooksPerStudent * notebookCost
  let totalHighlighterCost := classSize * highlightersPerStudent * highlighterCost
  let effectiveAmount := totalSpent + teacherDiscount
  let binderSpend := effectiveAmount - (totalPenCost + totalNotebookCost + totalHighlighterCost)
  let totalBinders := binderSpend / binderCost
  totalBinders / classSize

theorem binders_per_student_is_one :
  bindersPerStudent 30 0.5 1.25 4.25 0.75 5 3 2 100 260 = 1 := by
  sorry

end NUMINAMATH_CALUDE_binders_per_student_is_one_l1895_189530


namespace NUMINAMATH_CALUDE_smallest_prime_with_digit_sum_23_l1895_189568

/-- A function that returns the sum of digits of a natural number -/
def digit_sum (n : ℕ) : ℕ :=
  if n < 10 then n else n % 10 + digit_sum (n / 10)

/-- A function that checks if a natural number is prime -/
def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, 1 < m → m < n → ¬(n % m = 0)

theorem smallest_prime_with_digit_sum_23 :
  ∃ (p : ℕ), is_prime p ∧ digit_sum p = 23 ∧
  ∀ (q : ℕ), is_prime q → digit_sum q = 23 → p ≤ q :=
by sorry

end NUMINAMATH_CALUDE_smallest_prime_with_digit_sum_23_l1895_189568


namespace NUMINAMATH_CALUDE_katy_summer_reading_l1895_189547

/-- The number of books Katy read during the summer -/
def summer_reading (june july august : ℕ) : ℕ := june + july + august

theorem katy_summer_reading :
  ∀ (june july august : ℕ),
  june = 8 →
  july = 2 * june →
  august = july - 3 →
  summer_reading june july august = 37 :=
by
  sorry

end NUMINAMATH_CALUDE_katy_summer_reading_l1895_189547


namespace NUMINAMATH_CALUDE_al_return_probability_l1895_189560

/-- Represents the cities in Prime Land -/
inductive City : Type
| C : Fin 7 → City

/-- The next city based on the current city and coin flip result -/
def nextCity (current : City) (heads : Bool) : City :=
  match current with
  | City.C k => 
    if heads then
      City.C (2 * k)
    else
      City.C (2 * k + 1)

/-- The probability of being in a specific city after a certain number of moves -/
def probInCity (moves : Nat) (city : City) : ℚ :=
  sorry

/-- The main theorem to prove -/
theorem al_return_probability : 
  probInCity 10 (City.C 0) = 147 / 1024 := by
  sorry

end NUMINAMATH_CALUDE_al_return_probability_l1895_189560


namespace NUMINAMATH_CALUDE_probability_of_valid_pair_l1895_189594

def ball_numbers : Finset ℕ := {1, 2, 3, 4, 5, 6, 7}

def is_valid_pair (x y : ℕ) : Bool :=
  x ∈ ball_numbers ∧ y ∈ ball_numbers ∧ Even (x * y) ∧ x * y > 14

def valid_pairs : Finset (ℕ × ℕ) :=
  Finset.filter (λ p => is_valid_pair p.1 p.2) (ball_numbers.product ball_numbers)

theorem probability_of_valid_pair :
  (valid_pairs.card : ℚ) / (ball_numbers.card * ball_numbers.card : ℚ) = 16 / 49 := by
  sorry

end NUMINAMATH_CALUDE_probability_of_valid_pair_l1895_189594


namespace NUMINAMATH_CALUDE_remainder_theorem_l1895_189595

theorem remainder_theorem (A B : ℕ) (h : A = B * 9 + 13) : A % 9 = 4 := by
  sorry

end NUMINAMATH_CALUDE_remainder_theorem_l1895_189595


namespace NUMINAMATH_CALUDE_cross_out_all_stars_star_remains_uncrossed_l1895_189552

/-- Represents a 2n × 2n table with stars -/
structure StarTable (n : ℕ) where
  stars : Finset (Fin (2*n) × Fin (2*n))

/-- Theorem for part (a) -/
theorem cross_out_all_stars (n : ℕ) (table : StarTable n) 
  (h : table.stars.card = 3*n) :
  ∃ (rows columns : Finset (Fin (2*n))),
    rows.card = n ∧ 
    columns.card = n ∧
    (∀ star ∈ table.stars, star.1 ∈ rows ∨ star.2 ∈ columns) :=
sorry

/-- Theorem for part (b) -/
theorem star_remains_uncrossed (n : ℕ) (table : StarTable n)
  (h : table.stars.card = 3*n + 1) :
  ∀ (rows columns : Finset (Fin (2*n))),
    rows.card = n →
    columns.card = n →
    ∃ star ∈ table.stars, star.1 ∉ rows ∧ star.2 ∉ columns :=
sorry

end NUMINAMATH_CALUDE_cross_out_all_stars_star_remains_uncrossed_l1895_189552


namespace NUMINAMATH_CALUDE_negation_of_absolute_value_statement_l1895_189532

theorem negation_of_absolute_value_statement (S : Set ℝ) :
  (¬ ∀ x ∈ S, |x| ≥ 3) ↔ (∃ x ∈ S, |x| < 3) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_absolute_value_statement_l1895_189532


namespace NUMINAMATH_CALUDE_triangle_side_range_l1895_189540

theorem triangle_side_range (a : ℝ) : 
  (∃ (x y z : ℝ), x = 3 ∧ y = 2*a - 1 ∧ z = 4 ∧ 
    x + y > z ∧ x + z > y ∧ y + z > x) ↔ 
  (1 < a ∧ a < 4) := by
sorry

end NUMINAMATH_CALUDE_triangle_side_range_l1895_189540


namespace NUMINAMATH_CALUDE_fifth_term_of_sequence_l1895_189561

/-- An arithmetic sequence with specific first four terms -/
def arithmetic_sequence (x y : ℚ) : ℕ → ℚ
  | 0 => x + 2*y
  | 1 => x - 2*y
  | 2 => x + 2*y^2
  | 3 => x / (2*y)
  | n+4 => arithmetic_sequence x y 3 + (n+1) * (arithmetic_sequence x y 1 - arithmetic_sequence x y 0)

/-- The fifth term of the specific arithmetic sequence -/
def fifth_term (x y : ℚ) : ℚ := -x/6 - 12

theorem fifth_term_of_sequence (x y : ℚ) (h : y ≠ 0) :
  arithmetic_sequence x y 4 = fifth_term x y := by
  sorry

end NUMINAMATH_CALUDE_fifth_term_of_sequence_l1895_189561


namespace NUMINAMATH_CALUDE_eight_digit_divisibility_l1895_189557

theorem eight_digit_divisibility (a b : Nat) (h1 : a ≤ 9) (h2 : b ≤ 9) (h3 : a ≠ 0) : 
  ∃ k : Nat, (a * 10 + b) * 1010101 = 101 * k := by
  sorry

end NUMINAMATH_CALUDE_eight_digit_divisibility_l1895_189557


namespace NUMINAMATH_CALUDE_factorial_ones_divisibility_l1895_189584

/-- Definition of [n]! -/
def factorial_ones (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | k + 1 => (factorial_ones k) * (Nat.ofDigits 2 (List.replicate (k + 1) 1))

/-- Theorem stating that [n+m]! is divisible by [n]! · [m]! -/
theorem factorial_ones_divisibility (n m : ℕ) :
  ∃ k : ℕ, factorial_ones (n + m) = k * (factorial_ones n * factorial_ones m) := by
  sorry


end NUMINAMATH_CALUDE_factorial_ones_divisibility_l1895_189584


namespace NUMINAMATH_CALUDE_quadratic_polynomial_property_l1895_189534

/-- A quadratic polynomial of the form x^2 - (p+q)x + pq -/
def QuadraticPolynomial (p q : ℝ) : ℝ → ℝ := fun x ↦ x^2 - (p+q)*x + p*q

/-- The composite function p(p(x)) -/
def CompositePolynomial (p q : ℝ) : ℝ → ℝ :=
  fun x ↦ let px := QuadraticPolynomial p q x
          (QuadraticPolynomial p q) px

/-- Predicate that checks if a polynomial has exactly four distinct real roots -/
def HasFourDistinctRealRoots (f : ℝ → ℝ) : Prop :=
  ∃ (a b c d : ℝ), (a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d) ∧
    (∀ x, f x = 0 ↔ x = a ∨ x = b ∨ x = c ∨ x = d)

/-- The theorem to be proved -/
theorem quadratic_polynomial_property :
  ∃ (p q : ℝ),
    HasFourDistinctRealRoots (CompositePolynomial p q) ∧
    (∀ (p' q' : ℝ),
      HasFourDistinctRealRoots (CompositePolynomial p' q') →
      (let f := QuadraticPolynomial p q
       let f' := QuadraticPolynomial p' q'
       ∀ (a b c d : ℝ),
         f a = a → f b = b → f c = c → f d = d →
         ∀ (a' b' c' d' : ℝ),
           f' a' = a' → f' b' = b' → f' c' = c' → f' d' = d' →
           a * b * c * d ≥ a' * b' * c' * d')) →
    QuadraticPolynomial p q 0 = 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_polynomial_property_l1895_189534


namespace NUMINAMATH_CALUDE_cistern_wet_surface_area_l1895_189581

/-- Calculates the total wet surface area of a rectangular cistern -/
def wetSurfaceArea (length width depth : ℝ) : ℝ :=
  length * width + 2 * (length * depth + width * depth)

theorem cistern_wet_surface_area :
  let length : ℝ := 8
  let width : ℝ := 4
  let depth : ℝ := 1.25
  wetSurfaceArea length width depth = 62 := by
sorry

end NUMINAMATH_CALUDE_cistern_wet_surface_area_l1895_189581


namespace NUMINAMATH_CALUDE_theater_tickets_proof_l1895_189585

theorem theater_tickets_proof (reduced_first_week : ℕ) 
  (h1 : reduced_first_week > 0)
  (h2 : 5 * reduced_first_week = 16500)
  (h3 : reduced_first_week + 16500 = 25200) : 
  reduced_first_week = 8700 := by
  sorry

end NUMINAMATH_CALUDE_theater_tickets_proof_l1895_189585


namespace NUMINAMATH_CALUDE_monic_quadratic_with_complex_root_l1895_189521

theorem monic_quadratic_with_complex_root :
  ∃! (a b : ℝ), (∀ x : ℂ, x^2 + a*x + b = 0 ↔ x = 3 - 2*I ∨ x = 3 + 2*I) ∧
                (∀ x : ℂ, x^2 + a*x + b = (x - (3 - 2*I)) * (x - (3 + 2*I))) :=
by sorry

end NUMINAMATH_CALUDE_monic_quadratic_with_complex_root_l1895_189521


namespace NUMINAMATH_CALUDE_fraction_problem_l1895_189526

theorem fraction_problem (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (h1 : (a + b + c) / (a + b - c) = 7)
  (h2 : (a + b + c) / (a + c - b) = 1.75) :
  (a + b + c) / (b + c - a) = 3.5 := by
sorry

end NUMINAMATH_CALUDE_fraction_problem_l1895_189526


namespace NUMINAMATH_CALUDE_james_cryptocurrency_investment_l1895_189503

theorem james_cryptocurrency_investment (C : ℕ) : 
  (C * 15 = 12 * (15 + 15 * (2/3))) → C = 20 := by
  sorry

end NUMINAMATH_CALUDE_james_cryptocurrency_investment_l1895_189503


namespace NUMINAMATH_CALUDE_initial_men_correct_l1895_189577

/-- The number of men initially working -/
def initial_men : ℕ := 72

/-- The depth dug by the initial group in meters -/
def initial_depth : ℕ := 30

/-- The hours worked by the initial group per day -/
def initial_hours : ℕ := 8

/-- The new depth to be dug in meters -/
def new_depth : ℕ := 50

/-- The new hours to be worked per day -/
def new_hours : ℕ := 6

/-- The number of extra men needed for the new task -/
def extra_men : ℕ := 88

/-- Theorem stating that the initial number of men is correct -/
theorem initial_men_correct : 
  (initial_depth : ℚ) / (initial_hours * initial_men) = 
  (new_depth : ℚ) / (new_hours * (initial_men + extra_men)) :=
sorry

end NUMINAMATH_CALUDE_initial_men_correct_l1895_189577


namespace NUMINAMATH_CALUDE_hyperbola_distance_l1895_189573

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x^2 / 16 - y^2 / 20 = 1

-- Define the foci
def F1 : ℝ × ℝ := sorry
def F2 : ℝ × ℝ := sorry

-- Define a point on the hyperbola
def P : ℝ × ℝ := sorry

-- Distance function
def distance (p1 p2 : ℝ × ℝ) : ℝ := sorry

-- Theorem statement
theorem hyperbola_distance :
  hyperbola P.1 P.2 → distance P F1 = 9 → distance P F2 = 17 := by sorry

end NUMINAMATH_CALUDE_hyperbola_distance_l1895_189573


namespace NUMINAMATH_CALUDE_infinitely_many_integer_pairs_l1895_189543

theorem infinitely_many_integer_pairs : 
  ∃ (S : Set (ℤ × ℤ)), Set.Infinite S ∧ 
    ∀ (pair : ℤ × ℤ), pair ∈ S → 
      ∃ (k : ℤ), (pair.1 + 1) / pair.2 + (pair.2 + 1) / pair.1 = k :=
by sorry

end NUMINAMATH_CALUDE_infinitely_many_integer_pairs_l1895_189543


namespace NUMINAMATH_CALUDE_oranges_minus_apples_difference_l1895_189554

/-- The number of apples Leif has -/
def num_apples : ℕ := 14

/-- The number of dozens of oranges Leif has -/
def dozens_oranges : ℕ := 2

/-- The number of fruits in a dozen -/
def fruits_per_dozen : ℕ := 12

/-- Calculates the total number of oranges -/
def total_oranges : ℕ := dozens_oranges * fruits_per_dozen

/-- Theorem stating the difference between oranges and apples -/
theorem oranges_minus_apples_difference : 
  total_oranges - num_apples = 10 := by sorry

end NUMINAMATH_CALUDE_oranges_minus_apples_difference_l1895_189554


namespace NUMINAMATH_CALUDE_multiply_sum_equality_l1895_189563

theorem multiply_sum_equality : 15 * 36 + 15 * 24 = 900 := by
  sorry

end NUMINAMATH_CALUDE_multiply_sum_equality_l1895_189563


namespace NUMINAMATH_CALUDE_no_right_triangle_with_sqrt_2016_side_l1895_189505

theorem no_right_triangle_with_sqrt_2016_side : ¬ ∃ (a b : ℕ) (c : ℝ), 
  c = Real.sqrt 2016 ∧ (a * a + b * b = c * c ∨ a * a + c * c = b * b ∨ c * c + b * b = a * a) :=
sorry

end NUMINAMATH_CALUDE_no_right_triangle_with_sqrt_2016_side_l1895_189505


namespace NUMINAMATH_CALUDE_complex_equation_sum_l1895_189538

theorem complex_equation_sum (x y : ℝ) : 
  Complex.mk (2 * x - 1) 1 = Complex.mk y (y - 3) → x + y = 13 / 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_sum_l1895_189538


namespace NUMINAMATH_CALUDE_cubic_equation_solution_l1895_189553

theorem cubic_equation_solution :
  ∃ x : ℝ, x^3 + (x + 2)^3 + (x + 4)^3 = (x + 6)^3 ∧ x = 6 := by
  sorry

end NUMINAMATH_CALUDE_cubic_equation_solution_l1895_189553


namespace NUMINAMATH_CALUDE_chicken_nugget_ratio_l1895_189507

theorem chicken_nugget_ratio : 
  ∀ (keely kendall : ℕ),
  keely + kendall + 20 = 100 →
  (keely + kendall) / 20 = 4 := by
sorry

end NUMINAMATH_CALUDE_chicken_nugget_ratio_l1895_189507


namespace NUMINAMATH_CALUDE_sqrt_squared_equals_original_sqrt_529441_squared_l1895_189570

theorem sqrt_squared_equals_original (n : ℝ) (h : 0 ≤ n) : (Real.sqrt n)^2 = n := by
  sorry

theorem sqrt_529441_squared :
  (Real.sqrt 529441)^2 = 529441 := by
  apply sqrt_squared_equals_original
  norm_num

end NUMINAMATH_CALUDE_sqrt_squared_equals_original_sqrt_529441_squared_l1895_189570


namespace NUMINAMATH_CALUDE_coin_division_problem_l1895_189558

theorem coin_division_problem : 
  ∃ (n : ℕ), n > 0 ∧ 
  n % 8 = 6 ∧ 
  n % 7 = 5 ∧ 
  n % 9 = 0 ∧ 
  (∀ m : ℕ, m > 0 → m % 8 = 6 → m % 7 = 5 → m ≥ n) := by
  sorry

end NUMINAMATH_CALUDE_coin_division_problem_l1895_189558


namespace NUMINAMATH_CALUDE_number_equation_solution_l1895_189511

theorem number_equation_solution : ∃ x : ℝ, 
  x^(5/4) * 12^(1/4) * 60^(3/4) = 300 ∧ 
  ∀ ε > 0, |x - 6| < ε :=
by sorry

end NUMINAMATH_CALUDE_number_equation_solution_l1895_189511


namespace NUMINAMATH_CALUDE_relationship_significance_l1895_189528

/-- The critical value for a 2x2 contingency table at 0.05 significance level -/
def critical_value : ℝ := 3.841

/-- The observed K^2 value from a 2x2 contingency table -/
def observed_value : ℝ := 4.013

/-- The maximum probability of making a mistake -/
def max_error_probability : ℝ := 0.05

/-- Theorem stating the relationship between the observed value, critical value, and maximum error probability -/
theorem relationship_significance (h : observed_value > critical_value) :
  max_error_probability = 0.05 := by
  sorry

end NUMINAMATH_CALUDE_relationship_significance_l1895_189528


namespace NUMINAMATH_CALUDE_even_function_symmetric_and_f_not_odd_l1895_189575

-- Define even function
def IsEven (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

-- Define odd function
def IsOdd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

-- Define symmetry about y-axis
def SymmetricAboutYAxis (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

-- Define the function f(x) = x³ + 1
def f (x : ℝ) : ℝ := x^3 + 1

theorem even_function_symmetric_and_f_not_odd :
  (∀ f : ℝ → ℝ, IsEven f → SymmetricAboutYAxis f) ∧
  ¬IsOdd f :=
by sorry

end NUMINAMATH_CALUDE_even_function_symmetric_and_f_not_odd_l1895_189575


namespace NUMINAMATH_CALUDE_parabola_line_intersection_l1895_189576

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a parabola y² = 2px -/
structure Parabola where
  p : ℝ
  h : p > 0

/-- Represents a line y = mx + b -/
structure Line where
  m : ℝ
  b : ℝ

/-- Check if two points are perpendicular with respect to the origin -/
def perpendicular (a b : Point) : Prop :=
  a.x * b.x + a.y * b.y = 0

/-- Check if a point lies on a parabola -/
def onParabola (point : Point) (parabola : Parabola) : Prop :=
  point.y^2 = 2 * parabola.p * point.x

/-- Check if a point lies on a line -/
def onLine (point : Point) (line : Line) : Prop :=
  point.y = line.m * point.x + line.b

/-- The main theorem -/
theorem parabola_line_intersection 
  (C : Parabola) 
  (F : Point)
  (l : Line)
  (A B : Point)
  (h1 : F.x = 1/2 ∧ F.y = 0)
  (h2 : l.m = 2)
  (h3 : onParabola A C ∧ onParabola B C)
  (h4 : onLine A l ∧ onLine B l)
  (h5 : A ≠ ⟨0, 0⟩ ∧ B ≠ ⟨0, 0⟩)
  (h6 : perpendicular A B) :
  C.p = 1 ∧ l.b = -4 :=
sorry

end NUMINAMATH_CALUDE_parabola_line_intersection_l1895_189576


namespace NUMINAMATH_CALUDE_A_equiv_B_l1895_189537

-- Define a function to calculate the sum of digits
def sumOfDigits (n : ℕ) : ℕ := sorry

-- Define set A
def A : Set ℕ := {n : ℕ | 10000 ≤ n ∧ n < 100000 ∧
  (∃ k : ℤ, (sumOfDigits n + 1 = 5 * k ∨ sumOfDigits n - 1 = 5 * k))}

-- Define set B
def B : Set ℕ := {n : ℕ | 10000 ≤ n ∧ n < 100000 ∧
  (∃ k : ℤ, sumOfDigits n = 5 * k ∨ sumOfDigits n - 2 = 5 * k)}

-- Theorem statement
theorem A_equiv_B : Equiv A B := by sorry

end NUMINAMATH_CALUDE_A_equiv_B_l1895_189537


namespace NUMINAMATH_CALUDE_base_two_representation_of_125_l1895_189598

/-- Represents a natural number in base 2 as a list of bits (least significant bit first) -/
def BaseTwoRepresentation := List Bool

/-- Converts a natural number to its base 2 representation -/
def toBaseTwoRepresentation (n : ℕ) : BaseTwoRepresentation :=
  sorry

/-- Converts a base 2 representation to its decimal (base 10) value -/
def fromBaseTwoRepresentation (bits : BaseTwoRepresentation) : ℕ :=
  sorry

theorem base_two_representation_of_125 :
  toBaseTwoRepresentation 125 = [true, false, true, true, true, true, true] := by
  sorry

end NUMINAMATH_CALUDE_base_two_representation_of_125_l1895_189598


namespace NUMINAMATH_CALUDE_limit_of_sequence_l1895_189582

def a (n : ℕ) : ℚ := (4 * n - 1) / (2 * n + 1)

theorem limit_of_sequence :
  ∀ ε > 0, ∃ N : ℕ, ∀ n ≥ N, |a n - 2| < ε :=
by
  sorry

end NUMINAMATH_CALUDE_limit_of_sequence_l1895_189582


namespace NUMINAMATH_CALUDE_trig_values_150_degrees_l1895_189523

/-- Given a point P on the unit circle corresponding to an angle of 150°, 
    prove that tan(150°) = -√3 and sin(150°) = √3/2 -/
theorem trig_values_150_degrees : 
  ∀ (P : ℝ × ℝ), 
  (P.1^2 + P.2^2 = 1) →  -- P is on the unit circle
  (P.1 = -1/2 ∧ P.2 = Real.sqrt 3 / 2) →  -- P corresponds to 150°
  (Real.tan (150 * π / 180) = -Real.sqrt 3 ∧ 
   Real.sin (150 * π / 180) = Real.sqrt 3 / 2) := by
  sorry

end NUMINAMATH_CALUDE_trig_values_150_degrees_l1895_189523


namespace NUMINAMATH_CALUDE_treasure_chest_age_conversion_l1895_189589

/-- Converts a number from base 8 to base 10 --/
def base8ToBase10 (hundreds : Nat) (tens : Nat) (ones : Nat) : Nat :=
  hundreds * 8^2 + tens * 8^1 + ones * 8^0

/-- The age of the treasure chest in base 8 --/
def treasureChestAgeBase8 : Nat × Nat × Nat := (3, 4, 7)

theorem treasure_chest_age_conversion :
  let (h, t, o) := treasureChestAgeBase8
  base8ToBase10 h t o = 231 := by
  sorry

end NUMINAMATH_CALUDE_treasure_chest_age_conversion_l1895_189589


namespace NUMINAMATH_CALUDE_larger_number_proof_l1895_189588

theorem larger_number_proof (a b : ℕ) (ha : a > 0) (hb : b > 0) : 
  (Nat.gcd a b = 23) → 
  (Nat.lcm a b = 23 * 12 * 13) → 
  (max a b = 299) := by
sorry

end NUMINAMATH_CALUDE_larger_number_proof_l1895_189588


namespace NUMINAMATH_CALUDE_cupcake_ratio_l1895_189501

theorem cupcake_ratio (total : ℕ) (gluten_free : ℕ) (vegan : ℕ) (non_vegan_gluten : ℕ) :
  total = 80 →
  gluten_free = total / 2 →
  vegan = 24 →
  non_vegan_gluten = 28 →
  (gluten_free - non_vegan_gluten) / vegan = 1 / 2 := by
sorry

end NUMINAMATH_CALUDE_cupcake_ratio_l1895_189501


namespace NUMINAMATH_CALUDE_multiple_problem_l1895_189565

theorem multiple_problem (x m : ℝ) : 
  x = 69 → x - 18 = m * (86 - x) → m = 3 := by
  sorry

end NUMINAMATH_CALUDE_multiple_problem_l1895_189565


namespace NUMINAMATH_CALUDE_value_added_to_number_l1895_189522

theorem value_added_to_number (n v : ℤ) : n = 9 → 3 * (n + 2) = v + n → v = 24 := by
  sorry

end NUMINAMATH_CALUDE_value_added_to_number_l1895_189522


namespace NUMINAMATH_CALUDE_largest_package_size_and_cost_l1895_189587

def lucas_notebooks : ℕ := 36
def maria_notebooks : ℕ := 60
def package_cost : ℕ := 3

theorem largest_package_size_and_cost :
  let max_package_size := Nat.gcd lucas_notebooks maria_notebooks
  (max_package_size = 12) ∧ (package_cost = 3) := by
  sorry

end NUMINAMATH_CALUDE_largest_package_size_and_cost_l1895_189587


namespace NUMINAMATH_CALUDE_triangle_solutions_l1895_189564

/-- Represents a triangle with side lengths and angles -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- Checks if the given triangle satisfies the law of sines -/
def satisfiesLawOfSines (t : Triangle) : Prop :=
  t.a / Real.sin t.A = t.b / Real.sin t.B ∧
  t.b / Real.sin t.B = t.c / Real.sin t.C

/-- Checks if the given triangle satisfies the law of cosines -/
def satisfiesLawOfCosines (t : Triangle) : Prop :=
  t.c^2 = t.a^2 + t.b^2 - 2 * t.a * t.b * Real.cos t.C

/-- Checks if the angles of the triangle sum to 180 degrees -/
def anglesSum180 (t : Triangle) : Prop :=
  t.A + t.B + t.C = Real.pi

/-- Theorem stating the two possible solutions for the given triangle -/
theorem triangle_solutions :
  ∃ (t1 t2 : Triangle),
    t1.a = 6 ∧ t1.b = 6 * Real.sqrt 3 ∧ t1.A = Real.pi / 6 ∧
    t2.a = 6 ∧ t2.b = 6 * Real.sqrt 3 ∧ t2.A = Real.pi / 6 ∧
    satisfiesLawOfSines t1 ∧ satisfiesLawOfCosines t1 ∧ anglesSum180 t1 ∧
    satisfiesLawOfSines t2 ∧ satisfiesLawOfCosines t2 ∧ anglesSum180 t2 ∧
    t1.c = 12 ∧ t1.B = Real.pi / 3 ∧ t1.C = Real.pi / 2 ∧
    t2.c = 6 ∧ t2.B = 2 * Real.pi / 3 ∧ t2.C = Real.pi / 6 :=
  sorry

end NUMINAMATH_CALUDE_triangle_solutions_l1895_189564


namespace NUMINAMATH_CALUDE_coffee_package_size_l1895_189574

theorem coffee_package_size
  (total_coffee : ℕ)
  (num_larger_packages : ℕ)
  (num_small_packages : ℕ)
  (small_package_size : ℕ)
  (h1 : total_coffee = 55)
  (h2 : num_larger_packages = 3)
  (h3 : num_small_packages = num_larger_packages + 2)
  (h4 : small_package_size = 5)
  : ∃ (larger_package_size : ℕ),
    larger_package_size * num_larger_packages +
    small_package_size * num_small_packages = total_coffee ∧
    larger_package_size = 10 := by
  sorry

end NUMINAMATH_CALUDE_coffee_package_size_l1895_189574


namespace NUMINAMATH_CALUDE_division_problem_l1895_189559

theorem division_problem (dividend : ℕ) (divisor : ℕ) (remainder : ℕ) (quotient : ℕ) : 
  dividend = 760 → 
  divisor = 36 → 
  remainder = 4 → 
  dividend = divisor * quotient + remainder → 
  quotient = 21 := by
sorry

end NUMINAMATH_CALUDE_division_problem_l1895_189559


namespace NUMINAMATH_CALUDE_problem_solution_l1895_189569

theorem problem_solution :
  (999 * (-13) = -12987) ∧
  (999 * 118 * (4/5) + 333 * (-3/5) - 999 * 18 * (3/5) = 99900) ∧
  (6 / (-1/2 + 1/3) = -36) := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l1895_189569


namespace NUMINAMATH_CALUDE_arithmetic_sequence_property_l1895_189510

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_property (a : ℕ → ℝ) :
  arithmetic_sequence a →
  (a 1 + 2 * a 8 + a 15 = 96) →
  2 * a 9 - a 10 = 24 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_property_l1895_189510
