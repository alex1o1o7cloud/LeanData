import Mathlib

namespace valid_coloring_exists_l2720_272070

/-- Represents a 9x9 grid where each cell can be colored or uncolored -/
def Grid := Fin 9 → Fin 9 → Bool

/-- Check if two cells are adjacent (by side or corner) -/
def adjacent (x1 y1 x2 y2 : Fin 9) : Bool :=
  (x1 = x2 ∧ y1.val = y2.val + 1) ∨
  (x1 = x2 ∧ y1.val + 1 = y2.val) ∨
  (x1.val = x2.val + 1 ∧ y1 = y2) ∨
  (x1.val + 1 = x2.val ∧ y1 = y2) ∨
  (x1.val = x2.val + 1 ∧ y1.val = y2.val + 1) ∨
  (x1.val + 1 = x2.val ∧ y1.val + 1 = y2.val) ∨
  (x1.val = x2.val + 1 ∧ y1.val + 1 = y2.val) ∨
  (x1.val + 1 = x2.val ∧ y1.val = y2.val + 1)

/-- Check if a grid coloring is valid -/
def valid_coloring (g : Grid) : Prop :=
  -- Center is not colored
  ¬g 4 4 ∧
  -- No adjacent cells are colored
  (∀ x1 y1 x2 y2, adjacent x1 y1 x2 y2 → ¬(g x1 y1 ∧ g x2 y2)) ∧
  -- Any ray from center intersects a colored cell
  (∀ dx dy, dx ≠ 0 ∨ dy ≠ 0 →
    ∃ t : ℚ, t > 0 ∧ g ⌊4 + t * dx⌋ ⌊4 + t * dy⌋)

/-- Theorem: There exists a valid coloring of the 9x9 grid -/
theorem valid_coloring_exists : ∃ g : Grid, valid_coloring g :=
sorry

end valid_coloring_exists_l2720_272070


namespace equivalent_proposition_l2720_272091

/-- Represents the quality of goods -/
inductive Quality
| High
| NotHigh

/-- Represents the price of goods -/
inductive Price
| Cheap
| NotCheap

/-- Translates a Chinese phrase to its logical meaning -/
def translate : String → (Quality → Price → Prop)
| "好货不便宜" => λ q p => q = Quality.High → p ≠ Price.Cheap
| "便宜没好货" => λ q p => p = Price.Cheap → q ≠ Quality.High
| _ => λ _ _ => False

theorem equivalent_proposition : 
  ∀ (q : Quality) (p : Price), 
    (translate "好货不便宜" q p) ↔ (translate "便宜没好货" q p) :=
by sorry

end equivalent_proposition_l2720_272091


namespace jill_bouncy_balls_difference_l2720_272049

/-- The number of bouncy balls in each package -/
def balls_per_pack : ℕ := 18

/-- The number of packs of red bouncy balls Jill bought -/
def red_packs : ℕ := 5

/-- The number of packs of yellow bouncy balls Jill bought -/
def yellow_packs : ℕ := 4

/-- The total number of red bouncy balls Jill bought -/
def total_red_balls : ℕ := balls_per_pack * red_packs

/-- The total number of yellow bouncy balls Jill bought -/
def total_yellow_balls : ℕ := balls_per_pack * yellow_packs

/-- The difference between the number of red and yellow bouncy balls -/
def difference : ℕ := total_red_balls - total_yellow_balls

theorem jill_bouncy_balls_difference :
  difference = 18 := by sorry

end jill_bouncy_balls_difference_l2720_272049


namespace builder_problem_l2720_272012

/-- Calculate the minimum number of packs needed given the total items and items per pack -/
def minPacks (total : ℕ) (perPack : ℕ) : ℕ :=
  (total + perPack - 1) / perPack

/-- The problem statement -/
theorem builder_problem :
  let totalBrackets := 42
  let bracketsPerPack := 5
  minPacks totalBrackets bracketsPerPack = 9 := by
  sorry

end builder_problem_l2720_272012


namespace only_135_and_144_satisfy_l2720_272092

/-- Represents a 3-digit positive integer abc --/
structure ThreeDigitInt where
  a : Nat
  b : Nat
  c : Nat
  h1 : a > 0
  h2 : a ≤ 9
  h3 : b ≤ 9
  h4 : c ≤ 9

/-- The decimal representation of abc --/
def decimal_rep (n : ThreeDigitInt) : Nat :=
  100 * n.a + 10 * n.b + n.c

/-- The product of digits multiplied by their sum --/
def digit_product_sum (n : ThreeDigitInt) : Nat :=
  n.a * n.b * n.c * (n.a + n.b + n.c)

/-- The theorem stating that only 135 and 144 satisfy the equation --/
theorem only_135_and_144_satisfy :
  ∀ n : ThreeDigitInt, decimal_rep n = digit_product_sum n ↔ decimal_rep n = 135 ∨ decimal_rep n = 144 := by
  sorry

end only_135_and_144_satisfy_l2720_272092


namespace power_23_2023_mod_29_l2720_272006

theorem power_23_2023_mod_29 : 23^2023 % 29 = 24 := by
  sorry

end power_23_2023_mod_29_l2720_272006


namespace nested_cubes_properties_l2720_272005

/-- Represents a cube with an inscribed sphere, which contains another inscribed cube. -/
structure NestedCubes where
  outer_surface_area : ℝ
  outer_side_length : ℝ
  sphere_diameter : ℝ
  inner_side_length : ℝ

/-- The surface area of a cube given its side length. -/
def cube_surface_area (side_length : ℝ) : ℝ := 6 * side_length^2

/-- The volume of a cube given its side length. -/
def cube_volume (side_length : ℝ) : ℝ := side_length^3

/-- Theorem stating the properties of the nested cubes structure. -/
theorem nested_cubes_properties (nc : NestedCubes) 
  (h1 : nc.outer_surface_area = 54)
  (h2 : nc.outer_side_length^2 = 54 / 6)
  (h3 : nc.sphere_diameter = nc.outer_side_length)
  (h4 : nc.inner_side_length * Real.sqrt 3 = nc.sphere_diameter) :
  cube_surface_area nc.inner_side_length = 18 ∧ 
  cube_volume nc.inner_side_length = 3 * Real.sqrt 3 := by
  sorry

#check nested_cubes_properties

end nested_cubes_properties_l2720_272005


namespace equation_solution_l2720_272093

theorem equation_solution : 
  ∃ y : ℝ, (4 : ℝ) * 8^3 = 4^y ∧ y = 11/2 := by
  sorry

end equation_solution_l2720_272093


namespace back_squat_increase_calculation_l2720_272000

/-- Represents the increase in John's back squat in kg -/
def back_squat_increase : ℝ := sorry

/-- John's original back squat weight in kg -/
def original_back_squat : ℝ := 200

/-- The ratio of John's front squat to his back squat -/
def front_squat_ratio : ℝ := 0.8

/-- The ratio of a triple to John's front squat -/
def triple_ratio : ℝ := 0.9

/-- The total weight moved in three triples in kg -/
def total_triple_weight : ℝ := 540

theorem back_squat_increase_calculation :
  3 * (triple_ratio * front_squat_ratio * (original_back_squat + back_squat_increase)) = total_triple_weight ∧
  back_squat_increase = 50 := by sorry

end back_squat_increase_calculation_l2720_272000


namespace seven_balls_four_boxes_l2720_272025

/-- The number of ways to distribute indistinguishable balls into distinguishable boxes -/
def distribute_balls (num_balls : ℕ) (num_boxes : ℕ) : ℕ :=
  sorry

/-- Theorem: There are 92 ways to distribute 7 indistinguishable balls into 4 distinguishable boxes -/
theorem seven_balls_four_boxes : distribute_balls 7 4 = 92 := by
  sorry

end seven_balls_four_boxes_l2720_272025


namespace parallel_vectors_x_value_l2720_272082

/-- Two vectors are parallel if their components are proportional -/
def parallel (a b : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, a.1 = k * b.1 ∧ a.2 = k * b.2

theorem parallel_vectors_x_value :
  ∀ x : ℝ, parallel (-1, x) (-2, 4) → x = 2 := by
  sorry

end parallel_vectors_x_value_l2720_272082


namespace diamonds_10th_pattern_l2720_272046

/-- The number of diamonds in the n-th pattern of the sequence -/
def diamonds (n : ℕ) : ℕ :=
  if n = 0 then 0
  else if n = 1 then 4
  else diamonds (n - 1) + 4 * (2 * n - 1)

/-- The theorem stating that the 10th pattern has 400 diamonds -/
theorem diamonds_10th_pattern : diamonds 10 = 400 := by
  sorry

end diamonds_10th_pattern_l2720_272046


namespace intersection_sum_is_eight_l2720_272087

-- Define the two parabolas
def parabola1 (x y : ℝ) : Prop := y = (x + 2)^2
def parabola2 (x y : ℝ) : Prop := x + 5 = (y - 4)^2

-- Define the set of intersection points
def intersectionPoints : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | parabola1 p.1 p.2 ∧ parabola2 p.1 p.2}

-- Theorem statement
theorem intersection_sum_is_eight :
  ∃ (points : Finset (ℝ × ℝ)), points.toSet = intersectionPoints ∧
  points.card = 4 ∧
  (points.sum (λ p => p.1) + points.sum (λ p => p.2) = 8) := by
  sorry

end intersection_sum_is_eight_l2720_272087


namespace solution_equality_l2720_272038

-- Define the set of solutions
def solution_set : Set ℝ := {x : ℝ | |x - 1| + |x + 2| < 5}

-- State the theorem
theorem solution_equality : solution_set = Set.Ioo (-3) 2 := by
  sorry

end solution_equality_l2720_272038


namespace fifteen_children_pencil_count_l2720_272039

/-- Given a number of children and pencils per child, calculates the total number of pencils -/
def total_pencils (num_children : ℕ) (pencils_per_child : ℕ) : ℕ :=
  num_children * pencils_per_child

/-- Proves that 15 children with 2 pencils each have 30 pencils in total -/
theorem fifteen_children_pencil_count :
  total_pencils 15 2 = 30 := by
  sorry

end fifteen_children_pencil_count_l2720_272039


namespace no_cyclic_prime_divisibility_l2720_272075

theorem no_cyclic_prime_divisibility : ¬∃ (p : Fin 2007 → ℕ), 
  (∀ i, Nat.Prime (p i)) ∧ 
  (∀ i : Fin 2006, (p i)^2 - 1 ∣ p (i + 1)) ∧
  ((p 2006)^2 - 1 ∣ p 0) := by
  sorry

end no_cyclic_prime_divisibility_l2720_272075


namespace equation_describes_cone_l2720_272081

-- Define cylindrical coordinates
structure CylindricalCoord where
  r : ℝ
  θ : ℝ
  z : ℝ

-- Define the equation z = kr
def coneEquation (k : ℝ) (p : CylindricalCoord) : Prop :=
  p.z = k * p.r

-- Define a cone in cylindrical coordinates
def isCone (S : Set CylindricalCoord) : Prop :=
  ∃ k : ℝ, ∀ p ∈ S, coneEquation k p

-- Theorem statement
theorem equation_describes_cone (k : ℝ) :
  isCone { p : CylindricalCoord | coneEquation k p } :=
sorry

end equation_describes_cone_l2720_272081


namespace incorrect_height_correction_l2720_272099

theorem incorrect_height_correction (n : ℕ) (initial_avg wrong_height actual_avg : ℝ) :
  n = 35 →
  initial_avg = 180 →
  wrong_height = 166 →
  actual_avg = 178 →
  (n * initial_avg - wrong_height + (n * actual_avg - n * initial_avg + wrong_height)) / n = 236 :=
by sorry

end incorrect_height_correction_l2720_272099


namespace abc_negative_root_at_four_y1_greater_y2_l2720_272076

/-- Represents a parabola y = ax² + bx + c with vertex at (1, n) and 4a - 2b + c = 0 -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ
  n : ℝ
  vertex_x : a * 1 + b = 0
  vertex_y : a * 1^2 + b * 1 + c = n
  condition : 4 * a - 2 * b + c = 0

/-- If n > 0, then abc < 0 -/
theorem abc_negative (p : Parabola) (h : p.n > 0) : p.a * p.b * p.c < 0 := by sorry

/-- The equation ax² + bx + c = 0 has a root at x = 4 -/
theorem root_at_four (p : Parabola) : p.a * 4^2 + p.b * 4 + p.c = 0 := by sorry

/-- For any two points A(x₁, y₁) and B(x₂, y₂) on the parabola with x₁ < x₂, 
    if a(x₁ + x₂ - 2) < 0, then y₁ > y₂ -/
theorem y1_greater_y2 (p : Parabola) (x₁ x₂ y₁ y₂ : ℝ) 
  (h1 : y₁ = p.a * x₁^2 + p.b * x₁ + p.c)
  (h2 : y₂ = p.a * x₂^2 + p.b * x₂ + p.c)
  (h3 : x₁ < x₂)
  (h4 : p.a * (x₁ + x₂ - 2) < 0) : 
  y₁ > y₂ := by sorry

end abc_negative_root_at_four_y1_greater_y2_l2720_272076


namespace cubic_roots_sum_squares_l2720_272066

theorem cubic_roots_sum_squares (a b c : ℝ) : 
  a^3 - 3*a - 2 = 0 ∧ 
  b^3 - 3*b - 2 = 0 ∧ 
  c^3 - 3*c - 2 = 0 → 
  a^2*(b - c)^2 + b^2*(c - a)^2 + c^2*(a - b)^2 = 9 := by sorry

end cubic_roots_sum_squares_l2720_272066


namespace min_value_xy_plus_reciprocal_l2720_272062

theorem min_value_xy_plus_reciprocal (x y : ℝ) 
  (h1 : x + y = -1) 
  (h2 : x < 0) 
  (h3 : y < 0) : 
  ∃ (min : ℝ), min = 17/4 ∧ ∀ z, z = x*y + 1/(x*y) → z ≥ min :=
sorry

end min_value_xy_plus_reciprocal_l2720_272062


namespace cooper_fence_length_l2720_272015

/-- The length of each wall in Cooper's fence --/
def wall_length : ℕ := 20

/-- The number of walls in Cooper's fence --/
def num_walls : ℕ := 4

/-- The height of each wall in bricks --/
def wall_height : ℕ := 5

/-- The depth of each wall in bricks --/
def wall_depth : ℕ := 2

/-- The total number of bricks needed for the fence --/
def total_bricks : ℕ := 800

theorem cooper_fence_length :
  wall_length * num_walls * wall_height * wall_depth = total_bricks :=
by sorry

end cooper_fence_length_l2720_272015


namespace vector_sum_proof_l2720_272017

theorem vector_sum_proof :
  let v1 : Fin 3 → ℝ := ![5, -3, 8]
  let v2 : Fin 3 → ℝ := ![-2, 4, 1]
  let v3 : Fin 3 → ℝ := ![3, -6, -9]
  v1 + v2 + v3 = ![6, -5, 0] :=
by
  sorry

end vector_sum_proof_l2720_272017


namespace circle_m_range_l2720_272096

/-- A circle equation with parameter m -/
def circle_equation (x y m : ℝ) : Prop :=
  x^2 + y^2 + 2*x + m = 0

/-- Condition for the equation to represent a circle -/
def is_circle (m : ℝ) : Prop :=
  ∃ (x₀ y₀ r : ℝ), r > 0 ∧ ∀ (x y : ℝ), circle_equation x y m ↔ (x - x₀)^2 + (y - y₀)^2 = r^2

/-- The range of m for which the equation represents a circle -/
theorem circle_m_range :
  ∀ m : ℝ, is_circle m ↔ m < 1 :=
sorry

end circle_m_range_l2720_272096


namespace oshea_large_planters_l2720_272069

/-- The number of large planters Oshea has -/
def num_large_planters (total_seeds small_planter_capacity large_planter_capacity num_small_planters : ℕ) : ℕ :=
  (total_seeds - small_planter_capacity * num_small_planters) / large_planter_capacity

/-- Proof that Oshea has 4 large planters -/
theorem oshea_large_planters :
  num_large_planters 200 4 20 30 = 4 := by
  sorry

end oshea_large_planters_l2720_272069


namespace circle_areas_in_right_triangle_l2720_272024

theorem circle_areas_in_right_triangle (a b c : Real) (r : Real) :
  a = 3 ∧ b = 4 ∧ c = 5 ∧ r = 1 →
  a^2 + b^2 = c^2 →
  let α := Real.arctan (a / b)
  let β := Real.arctan (b / a)
  let γ := π / 2
  (α + β + γ = π) →
  (α / 2 + β / 2 + γ / 2) * r^2 = π / 2 := by
  sorry

end circle_areas_in_right_triangle_l2720_272024


namespace square_sum_given_difference_and_product_l2720_272031

theorem square_sum_given_difference_and_product (x y : ℝ) 
  (h1 : x - y = 10) (h2 : x * y = 9) : x^2 + y^2 = 118 := by
  sorry

end square_sum_given_difference_and_product_l2720_272031


namespace sphere_intersection_area_ratio_l2720_272020

theorem sphere_intersection_area_ratio (R : ℝ) (h : R > 0) :
  let r := Real.sqrt ((3 / 4) * R^2)
  let circle_area := π * r^2
  let sphere_surface_area := 4 * π * R^2
  circle_area / sphere_surface_area = 3 / 16 := by
  sorry

end sphere_intersection_area_ratio_l2720_272020


namespace complex_modulus_problem_l2720_272057

theorem complex_modulus_problem (i : ℂ) (h : i^2 = -1) : 
  Complex.abs (4 * i / (1 - i)) = 2 * Real.sqrt 2 := by sorry

end complex_modulus_problem_l2720_272057


namespace single_shot_exclusivity_two_shooters_not_exclusive_hit_or_miss_exclusivity_at_least_one_not_exclusive_l2720_272086

-- Define the basic events
def hits_9_rings : Prop := sorry
def hits_8_rings : Prop := sorry
def A_hits_10_rings : Prop := sorry
def B_hits_8_rings : Prop := sorry
def A_hits_target : Prop := sorry
def B_hits_target : Prop := sorry

-- Define compound events
def both_hit_target : Prop := A_hits_target ∧ B_hits_target
def neither_hit_target : Prop := ¬A_hits_target ∧ ¬B_hits_target
def at_least_one_hits : Prop := A_hits_target ∨ B_hits_target
def A_misses_B_hits : Prop := ¬A_hits_target ∧ B_hits_target

-- Define mutual exclusivity
def mutually_exclusive (p q : Prop) : Prop := ¬(p ∧ q)

-- Theorem statements
theorem single_shot_exclusivity : 
  mutually_exclusive hits_9_rings hits_8_rings := by sorry

theorem two_shooters_not_exclusive : 
  ¬(mutually_exclusive A_hits_10_rings B_hits_8_rings) := by sorry

theorem hit_or_miss_exclusivity : 
  mutually_exclusive both_hit_target neither_hit_target := by sorry

theorem at_least_one_not_exclusive : 
  ¬(mutually_exclusive at_least_one_hits A_misses_B_hits) := by sorry

end single_shot_exclusivity_two_shooters_not_exclusive_hit_or_miss_exclusivity_at_least_one_not_exclusive_l2720_272086


namespace two_red_two_blue_probability_l2720_272044

theorem two_red_two_blue_probability (total_marbles : ℕ) (red_marbles : ℕ) (blue_marbles : ℕ) 
  (h1 : total_marbles = red_marbles + blue_marbles)
  (h2 : red_marbles = 12)
  (h3 : blue_marbles = 8) :
  (Nat.choose red_marbles 2 * Nat.choose blue_marbles 2 : ℚ) / Nat.choose total_marbles 4 = 1848 / 4845 := by
  sorry

end two_red_two_blue_probability_l2720_272044


namespace opposite_numbers_fraction_equals_one_l2720_272054

theorem opposite_numbers_fraction_equals_one (a b : ℝ) 
  (h1 : a + b = 0) 
  (h2 : |a - b| = 2) : 
  (a^2 + 2*a*b + 2*b^2 + 2*a + 2*b + 1) / (a^2 + 3*a*b + b^2 + 3) = 1 := by
  sorry

end opposite_numbers_fraction_equals_one_l2720_272054


namespace sum_of_fractions_l2720_272018

theorem sum_of_fractions : 
  (2 : ℚ) / 10 + 4 / 10 + 6 / 10 + 8 / 10 + 10 / 10 + 12 / 10 + 14 / 10 + 16 / 10 + 18 / 10 + 20 / 10 = 11 := by
  sorry

end sum_of_fractions_l2720_272018


namespace cosine_power_expansion_sum_of_squares_l2720_272098

open Real

theorem cosine_power_expansion_sum_of_squares :
  ∃ (b₁ b₂ b₃ b₄ b₅ b₆ b₇ : ℝ),
    (∀ θ : ℝ, (cos θ)^7 = b₁ * cos θ + b₂ * cos (2*θ) + b₃ * cos (3*θ) + 
                          b₄ * cos (4*θ) + b₅ * cos (5*θ) + b₆ * cos (6*θ) + 
                          b₇ * cos (7*θ)) →
    b₁^2 + b₂^2 + b₃^2 + b₄^2 + b₅^2 + b₆^2 + b₇^2 = 429 / 1024 := by
  sorry

end cosine_power_expansion_sum_of_squares_l2720_272098


namespace money_distribution_l2720_272030

/-- Given three people A, B, and C with money, prove that B and C together have 360 Rs. -/
theorem money_distribution (a b c : ℕ) : 
  a + b + c = 500 →  -- Total money between A, B, and C
  a + c = 200 →      -- Money A and C have together
  c = 60 →           -- Money C has
  b + c = 360        -- Money B and C have together
  := by sorry

end money_distribution_l2720_272030


namespace count_descending_even_digits_is_five_count_ascending_even_digits_is_one_l2720_272048

/-- A function that returns the count of four-digit numbers with all even digits in descending order -/
def count_descending_even_digits : ℕ :=
  5

/-- A function that returns the count of four-digit numbers with all even digits in ascending order -/
def count_ascending_even_digits : ℕ :=
  1

/-- Theorem stating the count of four-digit numbers with all even digits in descending order -/
theorem count_descending_even_digits_is_five :
  count_descending_even_digits = 5 := by sorry

/-- Theorem stating the count of four-digit numbers with all even digits in ascending order -/
theorem count_ascending_even_digits_is_one :
  count_ascending_even_digits = 1 := by sorry

end count_descending_even_digits_is_five_count_ascending_even_digits_is_one_l2720_272048


namespace davids_remaining_money_is_19_90_l2720_272094

/-- Calculates David's remaining money after expenses and taxes -/
def davidsRemainingMoney (rate1 rate2 rate3 : ℝ) (hours : ℝ) (shoePrice : ℝ) 
  (shoeDiscount taxRate giftFraction : ℝ) : ℝ :=
  let totalEarnings := (rate1 + rate2 + rate3) * hours
  let taxAmount := totalEarnings * taxRate
  let discountedShoePrice := shoePrice * (1 - shoeDiscount)
  let remainingAfterShoes := totalEarnings - taxAmount - discountedShoePrice
  remainingAfterShoes * (1 - giftFraction)

/-- Theorem stating that David's remaining money is $19.90 -/
theorem davids_remaining_money_is_19_90 :
  davidsRemainingMoney 14 18 20 2 75 0.15 0.1 (1/3) = 19.90 := by
  sorry

end davids_remaining_money_is_19_90_l2720_272094


namespace clock_right_angles_in_day_l2720_272079

/-- Represents a clock with an hour hand and a minute hand. -/
structure Clock :=
  (hour_hand : ℕ)
  (minute_hand : ℕ)

/-- Represents a day consisting of 24 hours. -/
def Day := 24

/-- Checks if the hands of a clock are at right angles. -/
def is_right_angle (c : Clock) : Prop :=
  (c.hour_hand * 5 - c.minute_hand) % 60 = 15 ∨ (c.minute_hand - c.hour_hand * 5) % 60 = 15

/-- Counts the number of times the clock hands are at right angles in a day. -/
def count_right_angles (d : ℕ) : ℕ :=
  -- Implementation details omitted
  sorry

/-- Theorem stating that the hands of a clock are at right angles 44 times in a day. -/
theorem clock_right_angles_in_day :
  count_right_angles Day = 44 :=
sorry

end clock_right_angles_in_day_l2720_272079


namespace room_length_calculation_l2720_272029

/-- The length of a rectangular room given its width, paving cost, and paving rate. -/
theorem room_length_calculation (width : ℝ) (paving_cost : ℝ) (paving_rate : ℝ) :
  width = 4.75 ∧ paving_cost = 34200 ∧ paving_rate = 900 →
  paving_cost / paving_rate / width = 8 := by
  sorry

end room_length_calculation_l2720_272029


namespace positive_fourth_root_of_6561_l2720_272021

theorem positive_fourth_root_of_6561 (x : ℝ) (h1 : x > 0) (h2 : x^4 = 6561) : x = 9 := by
  sorry

end positive_fourth_root_of_6561_l2720_272021


namespace question_paper_combinations_l2720_272047

def choose (n k : ℕ) : ℕ := Nat.choose n k

theorem question_paper_combinations : choose 10 8 * choose 10 5 = 11340 := by
  sorry

end question_paper_combinations_l2720_272047


namespace relationship_proof_l2720_272058

theorem relationship_proof (a b : ℝ) (h1 : a + b > 0) (h2 : b < 0) : a > -b ∧ -b > b ∧ b > -a := by
  sorry

end relationship_proof_l2720_272058


namespace five_teachers_three_classes_l2720_272073

/-- The number of ways to assign n teachers to k distinct classes, 
    with at least one teacher per class -/
def teacher_assignments (n : ℕ) (k : ℕ) : ℕ :=
  sorry

/-- Theorem stating that there are 150 ways to assign 5 teachers to 3 classes -/
theorem five_teachers_three_classes : 
  teacher_assignments 5 3 = 150 := by
  sorry

end five_teachers_three_classes_l2720_272073


namespace sqrt_difference_of_squares_l2720_272026

theorem sqrt_difference_of_squares : (Real.sqrt 5 + Real.sqrt 3) * (Real.sqrt 5 - Real.sqrt 3) = 2 := by
  sorry

end sqrt_difference_of_squares_l2720_272026


namespace max_profit_is_33000_l2720_272041

/-- Profit function for the first store -/
def L₁ (x : ℝ) : ℝ := -5 * x^2 + 900 * x - 16000

/-- Profit function for the second store -/
def L₂ (x : ℝ) : ℝ := 300 * x - 2000

/-- Total number of vehicles sold -/
def total_vehicles : ℝ := 110

/-- Total profit function -/
def S (x : ℝ) : ℝ := L₁ x + L₂ (total_vehicles - x)

theorem max_profit_is_33000 :
  ∃ x : ℝ, x ≥ 0 ∧ x ≤ total_vehicles ∧ 
  (∀ y : ℝ, y ≥ 0 → y ≤ total_vehicles → S y ≤ S x) ∧
  S x = 33000 :=
sorry

end max_profit_is_33000_l2720_272041


namespace sum_even_coefficients_l2720_272067

theorem sum_even_coefficients (a a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ a₉ a₁₀ a₁₁ a₁₂ : ℝ) :
  (∀ x : ℝ, (1 + x + x^2)^6 = a + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5 + 
    a₆*x^6 + a₇*x^7 + a₈*x^8 + a₉*x^9 + a₁₀*x^10 + a₁₁*x^11 + a₁₂*x^12) →
  a₂ + a₄ + a₆ + a₈ + a₁₀ + a₁₂ = 364 := by
sorry

end sum_even_coefficients_l2720_272067


namespace five_cubic_yards_equals_135_cubic_feet_l2720_272072

/-- Conversion from yards to feet -/
def yards_to_feet (yards : ℝ) : ℝ := 3 * yards

/-- Conversion from cubic yards to cubic feet -/
def cubic_yards_to_cubic_feet (cubic_yards : ℝ) : ℝ := 27 * cubic_yards

/-- Theorem: 5 cubic yards is equal to 135 cubic feet -/
theorem five_cubic_yards_equals_135_cubic_feet :
  cubic_yards_to_cubic_feet 5 = 135 := by
  sorry

end five_cubic_yards_equals_135_cubic_feet_l2720_272072


namespace rectangle_shading_l2720_272097

theorem rectangle_shading (length width : ℕ) (initial_shaded_fraction final_shaded_fraction : ℚ) :
  length = 15 →
  width = 20 →
  initial_shaded_fraction = 1 / 4 →
  final_shaded_fraction = 1 / 5 →
  (initial_shaded_fraction * final_shaded_fraction : ℚ) = 1 / 20 :=
by sorry

end rectangle_shading_l2720_272097


namespace coin_flip_probability_difference_l2720_272064

/-- The probability of getting exactly k heads in n flips of a fair coin -/
def prob_k_heads (n k : ℕ) : ℚ :=
  (Nat.choose n k : ℚ) * (1 / 2) ^ n

/-- The theorem statement -/
theorem coin_flip_probability_difference : 
  |prob_k_heads 6 4 - prob_k_heads 6 6| = 7 / 32 := by
  sorry

end coin_flip_probability_difference_l2720_272064


namespace k_range_for_unique_integer_solution_l2720_272077

/-- Given a real number k, this function represents the system of inequalities -/
def inequality_system (x k : ℝ) : Prop :=
  x^2 - x - 2 > 0 ∧ 2*x^2 + (5+2*k)*x + 5*k < 0

/-- This theorem states that if -2 is the only integer solution to the inequality system,
    then k is in the range [-3, 2) -/
theorem k_range_for_unique_integer_solution :
  (∀ x : ℤ, inequality_system (x : ℝ) k ↔ x = -2) →
  -3 ≤ k ∧ k < 2 :=
sorry

end k_range_for_unique_integer_solution_l2720_272077


namespace double_length_isosceles_triangle_base_length_l2720_272052

/-- A triangle is double-length if one side is twice the length of another side. -/
def is_double_length_triangle (a b c : ℝ) : Prop :=
  a = 2 * b ∨ a = 2 * c ∨ b = 2 * c

/-- An isosceles triangle has two sides of equal length. -/
def is_isosceles_triangle (a b c : ℝ) : Prop :=
  a = b ∨ a = c ∨ b = c

theorem double_length_isosceles_triangle_base_length
  (a b c : ℝ)
  (h_isosceles : is_isosceles_triangle a b c)
  (h_double_length : is_double_length_triangle a b c)
  (h_ab_length : a = 10) :
  c = 5 := by
  sorry

end double_length_isosceles_triangle_base_length_l2720_272052


namespace paul_weed_eating_earnings_l2720_272034

/-- The amount of money Paul made mowing lawns -/
def money_mowing : ℕ := 68

/-- The number of weeks Paul's money would last -/
def weeks : ℕ := 9

/-- The amount Paul would spend per week -/
def spend_per_week : ℕ := 9

/-- The total amount of money Paul had -/
def total_money : ℕ := weeks * spend_per_week

/-- The amount of money Paul made weed eating -/
def money_weed_eating : ℕ := total_money - money_mowing

theorem paul_weed_eating_earnings : money_weed_eating = 13 := by
  sorry

end paul_weed_eating_earnings_l2720_272034


namespace license_plate_count_l2720_272089

/-- The number of possible letters in each position of the license plate -/
def num_letters : ℕ := 26

/-- The number of possible odd digits for the first digit position -/
def num_odd_digits : ℕ := 5

/-- The number of possible even digits for the second digit position -/
def num_even_digits : ℕ := 5

/-- The number of possible digits for the third digit position -/
def num_all_digits : ℕ := 10

/-- The total number of possible license plates under the given conditions -/
def total_license_plates : ℕ := num_letters^3 * num_odd_digits * num_even_digits * num_all_digits

theorem license_plate_count : total_license_plates = 17576000 := by
  sorry

end license_plate_count_l2720_272089


namespace value_range_equivalence_l2720_272045

def f (a x : ℝ) : ℝ := x^2 - 2*a*x + 3

theorem value_range_equivalence (a : ℝ) : 
  (∀ x ∈ Set.Icc (-2 : ℝ) 4, f a x ∈ Set.Icc (f a a) (f a 4)) ↔ 
  a ∈ Set.Icc (-2 : ℝ) 1 :=
sorry

end value_range_equivalence_l2720_272045


namespace fred_final_cards_l2720_272016

-- Define the initial number of cards Fred has
def initial_cards : ℕ := 5

-- Define the number of cards Fred gives to Melanie
def cards_to_melanie : ℕ := 2

-- Define the number of cards Fred trades with Sam
def cards_traded_with_sam : ℕ := 1

-- Define the number of cards Sam gives to Fred
def cards_from_sam : ℕ := 4

-- Define the number of cards Lisa has
def lisa_cards : ℕ := 3

-- Theorem to prove
theorem fred_final_cards : 
  initial_cards - cards_to_melanie - cards_traded_with_sam + cards_from_sam + 2 * lisa_cards = 12 :=
by sorry

end fred_final_cards_l2720_272016


namespace valid_placement_exists_l2720_272042

/-- Represents the configuration of the circles --/
inductive Position
| TopLeft | TopMiddle | TopRight
| MiddleLeft | MiddleRight
| BottomLeft | BottomMiddle | BottomRight
| Center

/-- A function type that maps positions to numbers --/
def Placement := Position → Fin 9

/-- Checks if two numbers are adjacent in the configuration --/
def are_adjacent (p1 p2 : Position) : Bool :=
  match p1, p2 with
  | Position.TopLeft, Position.TopMiddle => true
  | Position.TopLeft, Position.MiddleLeft => true
  | Position.TopMiddle, Position.TopRight => true
  | Position.TopMiddle, Position.Center => true
  | Position.TopRight, Position.MiddleRight => true
  | Position.MiddleLeft, Position.BottomLeft => true
  | Position.MiddleLeft, Position.Center => true
  | Position.MiddleRight, Position.BottomRight => true
  | Position.MiddleRight, Position.Center => true
  | Position.BottomLeft, Position.BottomMiddle => true
  | Position.BottomMiddle, Position.BottomRight => true
  | Position.BottomMiddle, Position.Center => true
  | _, _ => false

/-- The main theorem stating the existence of a valid placement --/
theorem valid_placement_exists : ∃ (p : Placement),
  (∀ pos1 pos2, pos1 ≠ pos2 → p pos1 ≠ p pos2) ∧
  (∀ pos1 pos2, are_adjacent pos1 pos2 → Nat.gcd (p pos1).val.succ (p pos2).val.succ = 1) :=
sorry

end valid_placement_exists_l2720_272042


namespace primality_test_upper_bound_l2720_272035

theorem primality_test_upper_bound :
  ∀ n : ℕ, 1000 ≤ n ∧ n ≤ 1100 →
  (∀ p : ℕ, p.Prime ∧ p ≤ 31 → ¬(p ∣ n)) →
  n.Prime ∨ n = 1 :=
sorry

end primality_test_upper_bound_l2720_272035


namespace circle_circumference_from_area_l2720_272050

theorem circle_circumference_from_area :
  ∀ (r : ℝ), r > 0 → π * r^2 = 225 * π → 2 * π * r = 30 * π := by
  sorry

end circle_circumference_from_area_l2720_272050


namespace rachel_earnings_l2720_272001

/-- Rachel's earnings as a waitress in one hour -/
theorem rachel_earnings (hourly_wage : ℝ) (people_served : ℕ) (tip_per_person : ℝ) 
  (h1 : hourly_wage = 12)
  (h2 : people_served = 20)
  (h3 : tip_per_person = 1.25) :
  hourly_wage + (people_served : ℝ) * tip_per_person = 37 := by
  sorry

end rachel_earnings_l2720_272001


namespace sum_outside_layers_l2720_272059

/-- Represents a 3D cube with specific properties -/
structure Cube3D where
  size : Nat
  total_units : Nat
  sum_per_line : ℝ
  special_value : ℝ

/-- Theorem stating the sum of numbers outside three layers in a specific cube -/
theorem sum_outside_layers (c : Cube3D) 
  (h_size : c.size = 20)
  (h_units : c.total_units = 8000)
  (h_sum : c.sum_per_line = 1)
  (h_special : c.special_value = 10) :
  let total_sum := c.size * c.size * c.sum_per_line
  let layer_sum := 3 * c.sum_per_line - 2 * c.sum_per_line + c.special_value
  total_sum - layer_sum = 392 := by
  sorry

end sum_outside_layers_l2720_272059


namespace triangle_ABC_properties_l2720_272053

-- Define the triangle ABC
variable (A B C : ℝ)

-- Define the dot product of vectors AB and BC
def dot_product_AB_BC : ℝ := sorry

-- Define the area of triangle ABC
def area_ABC : ℝ := sorry

-- State the theorem
theorem triangle_ABC_properties 
  (h1 : dot_product_AB_BC = (3/2) * area_ABC)
  (h2 : A - C = π/4) : 
  Real.sin B = 4/5 ∧ Real.cos A = (Real.sqrt (50 + 5 * Real.sqrt 2)) / 10 := by
  sorry

end triangle_ABC_properties_l2720_272053


namespace polynomial_root_sum_l2720_272036

theorem polynomial_root_sum (p q r s : ℝ) : 
  let g : ℂ → ℂ := λ x => x^4 + p*x^3 + q*x^2 + r*x + s
  (g (-3*I) = 0 ∧ g (1 + I) = 0) → p + q + r + s = 9 := by
sorry

end polynomial_root_sum_l2720_272036


namespace isosceles_triangle_base_length_l2720_272056

/-- Represents the base length of an isosceles triangle -/
def BaseLengthIsosceles (area : ℝ) (equalSide : ℝ) : Set ℝ :=
  {x | x > 0 ∧ (x * (equalSide ^ 2 - (x / 2) ^ 2).sqrt / 2 = area)}

/-- Theorem: The base length of an isosceles triangle with area 3 cm² and equal side 25 cm is either 14 cm or 48 cm -/
theorem isosceles_triangle_base_length :
  BaseLengthIsosceles 3 25 = {14, 48} := by
  sorry

end isosceles_triangle_base_length_l2720_272056


namespace purely_imaginary_complex_number_l2720_272055

theorem purely_imaginary_complex_number (m : ℝ) : 
  (m^2 - 1 : ℂ) + (m + 1 : ℂ) * Complex.I = Complex.I * y → m = 1 :=
by
  sorry

end purely_imaginary_complex_number_l2720_272055


namespace expression_value_l2720_272051

theorem expression_value : (36 + 9)^2 - (9^2 + 36^2) = -1894224 := by
  sorry

end expression_value_l2720_272051


namespace common_chord_equation_length_AB_l2720_272007

-- Define the circles C and M
def C (x y : ℝ) : Prop := x^2 + y^2 - 10*x - 10*y = 0
def M (x y : ℝ) : Prop := x^2 + y^2 + 6*x + 2*y - 40 = 0

-- Define the intersection points A and B
def A : ℝ × ℝ := (-2, 6)
def B : ℝ × ℝ := (4, -2)

-- Theorem for the equation of the common chord
theorem common_chord_equation : 
  ∀ (x y : ℝ), C x y ∧ M x y → 4*x + 2*y - 10 = 0 :=
sorry

-- Theorem for the length of AB
theorem length_AB : 
  Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 10 :=
sorry

end common_chord_equation_length_AB_l2720_272007


namespace jill_makes_30_trips_l2720_272008

/-- Represents the water-carrying problem with Jack and Jill --/
structure WaterProblem where
  tank_capacity : ℕ
  bucket_capacity : ℕ
  jack_buckets_per_trip : ℕ
  jill_buckets_per_trip : ℕ
  jack_trips_ratio : ℕ
  jill_trips_ratio : ℕ

/-- Calculates the number of trips Jill makes to fill the tank --/
def jill_trips (wp : WaterProblem) : ℕ :=
  let jack_water_per_trip := wp.jack_buckets_per_trip * wp.bucket_capacity
  let jill_water_per_trip := wp.jill_buckets_per_trip * wp.bucket_capacity
  let water_per_cycle := jack_water_per_trip * wp.jack_trips_ratio + jill_water_per_trip * wp.jill_trips_ratio
  let cycles := wp.tank_capacity / water_per_cycle
  cycles * wp.jill_trips_ratio

/-- Theorem stating that Jill makes 30 trips to fill the tank under the given conditions --/
theorem jill_makes_30_trips :
  let wp : WaterProblem := {
    tank_capacity := 600,
    bucket_capacity := 5,
    jack_buckets_per_trip := 2,
    jill_buckets_per_trip := 1,
    jack_trips_ratio := 3,
    jill_trips_ratio := 2
  }
  jill_trips wp = 30 := by
  sorry


end jill_makes_30_trips_l2720_272008


namespace chinese_math_problem_l2720_272063

-- Define the system of equations
def equation_system (x y : ℝ) : Prop :=
  5 * x + 2 * y = 19 ∧ 2 * x + 5 * y = 16

-- Define the profit function
def profit_function (m : ℝ) : ℝ := 0.5 * m + 5

-- Theorem statement
theorem chinese_math_problem :
  (∃ (x y : ℝ), equation_system x y ∧ x = 3 ∧ y = 2) ∧
  (∀ m : ℝ, 0 ≤ m ∧ m ≤ 5 → profit_function m ≤ profit_function 5) :=
by sorry

end chinese_math_problem_l2720_272063


namespace smallest_solution_congruence_l2720_272071

theorem smallest_solution_congruence (x : ℕ) :
  (x > 0 ∧ 5 * x ≡ 17 [MOD 31]) ↔ x = 13 := by
  sorry

end smallest_solution_congruence_l2720_272071


namespace quadratic_two_zeros_l2720_272084

theorem quadratic_two_zeros (b : ℝ) : 
  ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ 
    (x₁^2 + b*x₁ - 3 = 0) ∧ 
    (x₂^2 + b*x₂ - 3 = 0) ∧ 
    (∀ x : ℝ, x^2 + b*x - 3 = 0 → (x = x₁ ∨ x = x₂)) :=
by sorry

end quadratic_two_zeros_l2720_272084


namespace product_multiple_of_three_probability_l2720_272004

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

end product_multiple_of_three_probability_l2720_272004


namespace quadratic_inequality_l2720_272010

theorem quadratic_inequality (x : ℝ) : x^2 - 6*x > 15 ↔ x < -1.5 ∨ x > 7.5 := by
  sorry

end quadratic_inequality_l2720_272010


namespace quartic_roots_sum_product_l2720_272032

theorem quartic_roots_sum_product (p q : ℝ) : 
  (p^4 - 6*p - 1 = 0) → 
  (q^4 - 6*q - 1 = 0) → 
  (p ≠ q) →
  (∀ x : ℝ, x^4 - 6*x - 1 = 0 → x = p ∨ x = q) →
  p*q + p + q = 1 := by
sorry

end quartic_roots_sum_product_l2720_272032


namespace triangle_tan_c_l2720_272013

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    if the area S satisfies 2S = (a + b)² - c², then tan C = -4/3 -/
theorem triangle_tan_c (a b c : ℝ) (h : a > 0 ∧ b > 0 ∧ c > 0) :
  let S := (1 / 2) * a * b * Real.sin C
  2 * S = (a + b)^2 - c^2 →
  Real.tan C = -4/3 :=
by sorry

end triangle_tan_c_l2720_272013


namespace smallest_fraction_between_l2720_272065

theorem smallest_fraction_between (p q : ℕ+) : 
  (3 : ℚ) / 5 < p / q ∧ p / q < (5 : ℚ) / 8 ∧ 
  (∀ (p' q' : ℕ+), (3 : ℚ) / 5 < p' / q' ∧ p' / q' < (5 : ℚ) / 8 → q' ≥ q) →
  q - p = 5 := by
sorry

end smallest_fraction_between_l2720_272065


namespace appended_number_divisible_by_seven_l2720_272014

theorem appended_number_divisible_by_seven (a b : ℕ) (ha : 100 ≤ a ∧ a < 1000) 
  (hb : 100 ≤ b ∧ b < 1000) (h_rem : a % 7 = b % 7) :
  ∃ k : ℕ, 1000 * a + b = 7 * k :=
by sorry

end appended_number_divisible_by_seven_l2720_272014


namespace grapes_filling_days_l2720_272085

/-- The number of days required to fill a certain number of drums of grapes -/
def days_to_fill_grapes (pickers : ℕ) (drums_per_day : ℕ) (total_drums : ℕ) : ℕ :=
  total_drums / drums_per_day

/-- Theorem stating that it takes 77 days to fill 17017 drums of grapes -/
theorem grapes_filling_days :
  days_to_fill_grapes 235 221 17017 = 77 := by
  sorry

end grapes_filling_days_l2720_272085


namespace always_quadratic_radical_l2720_272090

theorem always_quadratic_radical (a : ℝ) : 0 ≤ a^2 + 1 := by sorry

end always_quadratic_radical_l2720_272090


namespace infinitely_many_solutions_l2720_272061

theorem infinitely_many_solutions (b : ℝ) :
  (∀ x : ℝ, 5 * (3 * x - b) = 3 * (5 * x + 15)) ↔ b = -9 := by
  sorry

end infinitely_many_solutions_l2720_272061


namespace sequence_is_decreasing_l2720_272074

def is_decreasing (a : ℕ+ → ℝ) : Prop :=
  ∀ n : ℕ+, a n ≥ a (n + 1)

theorem sequence_is_decreasing (a : ℕ+ → ℝ) 
  (h : ∀ n : ℕ+, a n - a (n + 1) = 10) : 
  is_decreasing a :=
sorry

end sequence_is_decreasing_l2720_272074


namespace arithmetic_mean_of_fractions_l2720_272037

theorem arithmetic_mean_of_fractions :
  let a := 3 / 8
  let b := 5 / 9
  (a + b) / 2 = 67 / 144 := by
sorry

end arithmetic_mean_of_fractions_l2720_272037


namespace banana_price_theorem_l2720_272078

/-- The cost of a banana in pence -/
def banana_cost : ℚ := 1.25

/-- The number of pence in a shilling -/
def pence_per_shilling : ℕ := 12

/-- The number of shillings in a pound -/
def shillings_per_pound : ℕ := 20

/-- The number of bananas in a dozen dozen -/
def dozen_dozen : ℕ := 12 * 12

theorem banana_price_theorem :
  let pence_per_pound : ℕ := pence_per_shilling * shillings_per_pound
  let bananas_per_fiver : ℚ := (5 * pence_per_pound : ℚ) / banana_cost
  let sixpences_for_16_dozen_dozen : ℚ := (16 * dozen_dozen * banana_cost) / 6
  sixpences_for_16_dozen_dozen = bananas_per_fiver / 2 :=
by sorry


end banana_price_theorem_l2720_272078


namespace existence_of_indices_with_inequalities_l2720_272040

theorem existence_of_indices_with_inequalities 
  (a b c : ℕ → ℕ) : 
  ∃ p q : ℕ, a p ≥ a q ∧ b p ≥ b q ∧ c p ≥ c q :=
by sorry

end existence_of_indices_with_inequalities_l2720_272040


namespace min_value_greater_than_five_l2720_272003

/-- The function f(x) defined in the problem -/
def f (a : ℝ) (x : ℝ) : ℝ := x^2 + (a + 1)^2 + |x + a - 1|

/-- The theorem statement -/
theorem min_value_greater_than_five (a : ℝ) :
  (∀ x, f a x > 5) ↔ a < (1 - Real.sqrt 14) / 2 ∨ a > Real.sqrt 6 / 2 := by sorry

end min_value_greater_than_five_l2720_272003


namespace equidistant_point_y_coordinate_l2720_272095

/-- The y-coordinate of the point on the y-axis that is equidistant from A(3, 0) and B(-4, 5) -/
theorem equidistant_point_y_coordinate : 
  ∃ y : ℝ, (y = 16/5) ∧ 
  ((0 - 3)^2 + (y - 0)^2 = (0 - (-4))^2 + (y - 5)^2) := by
  sorry

end equidistant_point_y_coordinate_l2720_272095


namespace M_mod_500_l2720_272028

/-- A function that counts the number of 1s in the binary representation of a natural number -/
def countOnes (n : ℕ) : ℕ := sorry

/-- The sequence of positive integers whose binary representation has exactly 9 ones -/
def T : ℕ → ℕ := sorry

/-- M is the 500th number in the sequence T -/
def M : ℕ := T 500

theorem M_mod_500 : M % 500 = 281 := by sorry

end M_mod_500_l2720_272028


namespace smallest_N_l2720_272023

/-- Represents a point in the square array -/
structure Point where
  row : Fin 4
  col : Nat

/-- The first numbering scheme (left to right, top to bottom) -/
def x (p : Point) (N : Nat) : Nat :=
  p.row.val * N + p.col

/-- The second numbering scheme (top to bottom, left to right) -/
def y (p : Point) : Nat :=
  (p.col - 1) * 4 + p.row.val + 1

/-- The theorem stating the smallest possible value of N -/
theorem smallest_N : ∃ (N : Nat) (p₁ p₂ p₃ p₄ : Point),
  N > 0 ∧
  p₁.row = 0 ∧ p₂.row = 1 ∧ p₃.row = 2 ∧ p₄.row = 3 ∧
  p₁.col > 0 ∧ p₂.col > 0 ∧ p₃.col > 0 ∧ p₄.col > 0 ∧
  p₁.col ≤ N ∧ p₂.col ≤ N ∧ p₃.col ≤ N ∧ p₄.col ≤ N ∧
  x p₁ N = y p₃ ∧
  x p₂ N = y p₁ ∧
  x p₃ N = y p₄ ∧
  x p₄ N = y p₂ ∧
  (∀ (M : Nat) (q₁ q₂ q₃ q₄ : Point),
    M > 0 ∧
    q₁.row = 0 ∧ q₂.row = 1 ∧ q₃.row = 2 ∧ q₄.row = 3 ∧
    q₁.col > 0 ∧ q₂.col > 0 ∧ q₃.col > 0 ∧ q₄.col > 0 ∧
    q₁.col ≤ M ∧ q₂.col ≤ M ∧ q₃.col ≤ M ∧ q₄.col ≤ M ∧
    x q₁ M = y q₃ ∧
    x q₂ M = y q₁ ∧
    x q₃ M = y q₄ ∧
    x q₄ M = y q₂ →
    N ≤ M) ∧
  N = 12 :=
by sorry

end smallest_N_l2720_272023


namespace complementary_angle_adjustment_l2720_272080

/-- Proves that when two complementary angles with a ratio of 3:7 have the smaller angle
    increased by 20%, the larger angle must decrease by 8.571% to maintain complementary angles. -/
theorem complementary_angle_adjustment (smaller larger : ℝ) : 
  smaller + larger = 90 →  -- angles are complementary
  smaller / larger = 3 / 7 →  -- ratio of angles is 3:7
  let new_smaller := smaller * 1.20  -- smaller angle increased by 20%
  let new_larger := 90 - new_smaller  -- new larger angle to maintain complementary
  (larger - new_larger) / larger * 100 = 8.571 :=  -- percentage decrease of larger angle
by sorry

end complementary_angle_adjustment_l2720_272080


namespace frank_game_points_l2720_272002

theorem frank_game_points (enemies_defeated : ℕ) (points_per_enemy : ℕ) 
  (level_completion_points : ℕ) (special_challenges : ℕ) (points_per_challenge : ℕ) : 
  enemies_defeated = 15 → 
  points_per_enemy = 12 → 
  level_completion_points = 20 → 
  special_challenges = 5 → 
  points_per_challenge = 10 → 
  enemies_defeated * points_per_enemy + level_completion_points + special_challenges * points_per_challenge = 250 := by
  sorry

#check frank_game_points

end frank_game_points_l2720_272002


namespace modular_inverse_28_mod_29_l2720_272083

theorem modular_inverse_28_mod_29 : ∃ x : ℕ, x ≤ 28 ∧ (28 * x) % 29 = 1 :=
by
  -- The proof goes here
  sorry

end modular_inverse_28_mod_29_l2720_272083


namespace minutes_after_midnight_l2720_272068

/-- Represents a date and time -/
structure DateTime where
  year : ℕ
  month : ℕ
  day : ℕ
  hour : ℕ
  minute : ℕ

/-- Adds minutes to a DateTime -/
def addMinutes (dt : DateTime) (minutes : ℕ) : DateTime :=
  sorry

/-- The starting DateTime (midnight on January 1, 2013) -/
def startDateTime : DateTime :=
  { year := 2013, month := 1, day := 1, hour := 0, minute := 0 }

/-- The resulting DateTime after adding 2537 minutes -/
def resultDateTime : DateTime :=
  { year := 2013, month := 1, day := 2, hour := 18, minute := 17 }

/-- Theorem stating that adding 2537 minutes to the start time results in the correct end time -/
theorem minutes_after_midnight (startTime : DateTime) (elapsedMinutes : ℕ) :
  startTime = startDateTime → elapsedMinutes = 2537 →
  addMinutes startTime elapsedMinutes = resultDateTime :=
by
  sorry

end minutes_after_midnight_l2720_272068


namespace linear_equation_solution_l2720_272043

theorem linear_equation_solution (k : ℝ) (x : ℝ) :
  k - 2 = 0 →        -- Condition for linearity
  4 * k ≠ 0 →        -- Ensure non-trivial equation
  (k - 2) * x^2 + 4 * k * x - 5 = 0 →
  x = 5 / 8 := by
sorry

end linear_equation_solution_l2720_272043


namespace polynomial_product_equality_l2720_272027

theorem polynomial_product_equality (x a : ℝ) : (x - a) * (x^2 + a*x + a^2) = x^3 - a^3 := by
  sorry

end polynomial_product_equality_l2720_272027


namespace pizza_toppings_l2720_272009

theorem pizza_toppings (total_slices : ℕ) (pepperoni_slices : ℕ) (mushroom_slices : ℕ) 
  (h1 : total_slices = 15)
  (h2 : pepperoni_slices = 8)
  (h3 : mushroom_slices = 12)
  (h4 : ∀ slice, slice ≤ total_slices → (slice ≤ pepperoni_slices ∨ slice ≤ mushroom_slices)) :
  ∃ both_toppings : ℕ, 
    both_toppings = pepperoni_slices + mushroom_slices - total_slices ∧
    both_toppings = 5 :=
by
  sorry

end pizza_toppings_l2720_272009


namespace unique_three_digit_factorion_l2720_272019

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

def sumOfDigitFactorials (n : ℕ) : ℕ :=
  (n.digits 10).map factorial |>.sum

def isFactorion (n : ℕ) : Prop :=
  n = sumOfDigitFactorials n

theorem unique_three_digit_factorion :
  ∀ n : ℕ, 100 ≤ n → n < 1000 → isFactorion n → n = 145 :=
by sorry

end unique_three_digit_factorion_l2720_272019


namespace exponent_division_thirteen_eleven_div_thirteen_four_l2720_272033

theorem exponent_division (a : ℕ) (m n : ℕ) (h : a > 0) : a^m / a^n = a^(m - n) := by sorry

theorem thirteen_eleven_div_thirteen_four :
  (13 : ℕ)^11 / (13 : ℕ)^4 = (13 : ℕ)^7 := by sorry

end exponent_division_thirteen_eleven_div_thirteen_four_l2720_272033


namespace five_ounce_letter_cost_l2720_272011

/-- Postage fee structure -/
structure PostageFee where
  baseRate : ℚ  -- Base rate in dollars
  additionalRate : ℚ  -- Additional rate per ounce in dollars
  handlingFee : ℚ  -- Handling fee in dollars
  handlingFeeThreshold : ℕ  -- Threshold in ounces for applying handling fee

/-- Calculate the total postage fee for a given weight -/
def calculatePostageFee (fee : PostageFee) (weight : ℕ) : ℚ :=
  fee.baseRate +
  fee.additionalRate * (weight - 1) +
  if weight > fee.handlingFeeThreshold then fee.handlingFee else 0

/-- Theorem: The cost to send a 5-ounce letter is $1.45 -/
theorem five_ounce_letter_cost :
  let fee : PostageFee := {
    baseRate := 35 / 100,
    additionalRate := 25 / 100,
    handlingFee := 10 / 100,
    handlingFeeThreshold := 2
  }
  calculatePostageFee fee 5 = 145 / 100 := by
  sorry

end five_ounce_letter_cost_l2720_272011


namespace intersection_of_intervals_l2720_272060

open Set

-- Define the sets A and B
def A : Set ℝ := Ioo (-1) 2
def B : Set ℝ := Ioi 0

-- State the theorem
theorem intersection_of_intervals : A ∩ B = Ioo 0 2 := by
  sorry

end intersection_of_intervals_l2720_272060


namespace triangle_properties_l2720_272022

/-- Triangle with sides a, b, c opposite to angles A, B, C --/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- Theorem about properties of a specific triangle --/
theorem triangle_properties (t : Triangle) 
  (h_acute : t.A > 0 ∧ t.A < π/2 ∧ t.B > 0 ∧ t.B < π/2 ∧ t.C > 0 ∧ t.C < π/2)
  (h_cosine : t.a * Real.cos t.A + t.b * Real.cos t.B = t.c) :
  (t.a = t.b) ∧ 
  (∀ (circumcircle_area : ℝ), circumcircle_area = π → 
    7 < (3 * t.b^2 + t.b + 4 * t.c) / t.a ∧ 
    (3 * t.b^2 + t.b + 4 * t.c) / t.a < 7 * Real.sqrt 2 + 1) := by
  sorry

end triangle_properties_l2720_272022


namespace iron_to_steel_ratio_l2720_272088

/-- Represents the ratio of two quantities -/
structure Ratio where
  numerator : ℕ
  denominator : ℕ

/-- Represents the composition of an alloy -/
structure Alloy where
  iron : ℕ
  steel : ℕ

/-- Simplifies a ratio by dividing both numerator and denominator by their GCD -/
def simplifyRatio (r : Ratio) : Ratio :=
  let gcd := Nat.gcd r.numerator r.denominator
  { numerator := r.numerator / gcd, denominator := r.denominator / gcd }

/-- Theorem: The ratio of iron to steel in the alloy is 2:5 -/
theorem iron_to_steel_ratio (alloy : Alloy) (h : alloy = { iron := 14, steel := 35 }) :
  simplifyRatio { numerator := alloy.iron, denominator := alloy.steel } = { numerator := 2, denominator := 5 } := by
  sorry

end iron_to_steel_ratio_l2720_272088
