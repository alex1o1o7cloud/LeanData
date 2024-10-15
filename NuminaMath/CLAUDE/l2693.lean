import Mathlib

namespace NUMINAMATH_CALUDE_candy_bar_cost_is_25_l2693_269326

/-- The cost of a candy bar in cents -/
def candy_bar_cost : ℕ := sorry

/-- The cost of a piece of chocolate in cents -/
def chocolate_cost : ℕ := 75

/-- The cost of a pack of juice in cents -/
def juice_cost : ℕ := 50

/-- The number of quarters needed to buy the items -/
def quarters_needed : ℕ := 11

/-- The value of a quarter in cents -/
def quarter_value : ℕ := 25

/-- The number of candy bars purchased -/
def candy_bars_bought : ℕ := 3

/-- The number of chocolate pieces purchased -/
def chocolates_bought : ℕ := 2

/-- The number of juice packs purchased -/
def juices_bought : ℕ := 1

theorem candy_bar_cost_is_25 : 
  candy_bar_cost = 25 :=
by
  have h1 : quarters_needed * quarter_value = 
    candy_bars_bought * candy_bar_cost + 
    chocolates_bought * chocolate_cost + 
    juices_bought * juice_cost := by sorry
  sorry

end NUMINAMATH_CALUDE_candy_bar_cost_is_25_l2693_269326


namespace NUMINAMATH_CALUDE_jerry_removed_figures_l2693_269310

/-- The number of old action figures removed from Jerry's shelf. -/
def old_figures_removed (initial : ℕ) (added : ℕ) (current : ℕ) : ℕ :=
  initial + added - current

/-- Theorem stating the number of old action figures Jerry removed. -/
theorem jerry_removed_figures : old_figures_removed 7 11 8 = 10 := by
  sorry

end NUMINAMATH_CALUDE_jerry_removed_figures_l2693_269310


namespace NUMINAMATH_CALUDE_bird_count_correct_bird_count_l2693_269369

theorem bird_count (total_heads : ℕ) (total_legs : ℕ) : ℕ :=
  let birds : ℕ := total_heads - (total_legs - 2 * total_heads) / 2
  birds

theorem correct_bird_count :
  bird_count 300 980 = 110 := by
  sorry

end NUMINAMATH_CALUDE_bird_count_correct_bird_count_l2693_269369


namespace NUMINAMATH_CALUDE_sock_pair_probability_l2693_269357

def total_socks : ℕ := 40
def white_socks : ℕ := 10
def red_socks : ℕ := 12
def black_socks : ℕ := 18
def drawn_socks : ℕ := 3

theorem sock_pair_probability :
  let total_ways := Nat.choose total_socks drawn_socks
  let all_different := white_socks * red_socks * black_socks
  let at_least_one_pair := total_ways - all_different
  (at_least_one_pair : ℚ) / total_ways = 193 / 247 := by
  sorry

end NUMINAMATH_CALUDE_sock_pair_probability_l2693_269357


namespace NUMINAMATH_CALUDE_pyramid_volume_change_l2693_269390

/-- Given a pyramid with rectangular base and volume 60 cubic feet, 
    prove that tripling its length, doubling its width, and increasing its height by 20% 
    results in a new volume of 432 cubic feet. -/
theorem pyramid_volume_change (V : ℝ) (l w h : ℝ) : 
  V = 60 → 
  V = (1/3) * l * w * h → 
  (1/3) * (3*l) * (2*w) * (1.2*h) = 432 :=
by sorry

end NUMINAMATH_CALUDE_pyramid_volume_change_l2693_269390


namespace NUMINAMATH_CALUDE_triangle_problem_l2693_269379

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The theorem statement -/
theorem triangle_problem (t : Triangle) 
  (h1 : 2 * t.a * Real.cos t.C = 2 * t.b - t.c)
  (h2 : t.a = Real.sqrt 21)
  (h3 : t.b = 4) :
  t.A = π / 3 ∧ t.c = 5 := by
  sorry


end NUMINAMATH_CALUDE_triangle_problem_l2693_269379


namespace NUMINAMATH_CALUDE_monkey_climb_l2693_269350

/-- Monkey's climb on a greased pole -/
theorem monkey_climb (pole_height : ℝ) (ascent : ℝ) (total_minutes : ℕ) (slip : ℝ) : 
  pole_height = 10 →
  ascent = 2 →
  total_minutes = 17 →
  (total_minutes / 2 : ℝ) * ascent - ((total_minutes - 1) / 2 : ℝ) * slip = pole_height →
  slip = 6/7 := by
  sorry

end NUMINAMATH_CALUDE_monkey_climb_l2693_269350


namespace NUMINAMATH_CALUDE_a_minus_b_squared_l2693_269307

theorem a_minus_b_squared (a b : ℝ) (h1 : (a + b)^2 = 49) (h2 : a * b = 6) : 
  (a - b)^2 = 25 := by
  sorry

end NUMINAMATH_CALUDE_a_minus_b_squared_l2693_269307


namespace NUMINAMATH_CALUDE_fan_shooting_theorem_l2693_269331

/-- Represents a fan with four blades rotating at a given speed -/
structure Fan :=
  (revolution_speed : ℝ)
  (num_blades : ℕ)

/-- Represents a bullet trajectory -/
structure BulletTrajectory :=
  (angle : ℝ)
  (speed : ℝ)

/-- Checks if a bullet trajectory intersects all blades of a fan -/
def intersects_all_blades (f : Fan) (bt : BulletTrajectory) : Prop :=
  sorry

/-- The main theorem stating that there exists a bullet trajectory that intersects all blades -/
theorem fan_shooting_theorem (f : Fan) 
  (h1 : f.revolution_speed = 50)
  (h2 : f.num_blades = 4) : 
  ∃ (bt : BulletTrajectory), intersects_all_blades f bt :=
sorry

end NUMINAMATH_CALUDE_fan_shooting_theorem_l2693_269331


namespace NUMINAMATH_CALUDE_scientific_notation_of_44300000_l2693_269338

theorem scientific_notation_of_44300000 :
  ∃ (a : ℝ) (n : ℤ), 1 ≤ a ∧ a < 10 ∧ 44300000 = a * (10 : ℝ) ^ n ∧ a = 4.43 ∧ n = 7 := by
  sorry

end NUMINAMATH_CALUDE_scientific_notation_of_44300000_l2693_269338


namespace NUMINAMATH_CALUDE_additional_sugar_needed_l2693_269356

/-- The amount of sugar needed for a cake recipe -/
def recipe_sugar : ℕ := 14

/-- The amount of sugar already added to the cake -/
def sugar_added : ℕ := 2

/-- The additional amount of sugar needed -/
def additional_sugar : ℕ := recipe_sugar - sugar_added

theorem additional_sugar_needed : additional_sugar = 12 := by
  sorry

end NUMINAMATH_CALUDE_additional_sugar_needed_l2693_269356


namespace NUMINAMATH_CALUDE_system_solution_l2693_269314

theorem system_solution (x y z t : ℝ) : 
  (x * y * z = x + y + z ∧
   y * z * t = y + z + t ∧
   z * t * x = z + t + x ∧
   t * x * y = t + x + y) →
  ((x = 0 ∧ y = 0 ∧ z = 0 ∧ t = 0) ∨
   (x = Real.sqrt 3 ∧ y = Real.sqrt 3 ∧ z = Real.sqrt 3 ∧ t = Real.sqrt 3) ∨
   (x = -Real.sqrt 3 ∧ y = -Real.sqrt 3 ∧ z = -Real.sqrt 3 ∧ t = -Real.sqrt 3)) :=
by sorry

end NUMINAMATH_CALUDE_system_solution_l2693_269314


namespace NUMINAMATH_CALUDE_remainder_of_division_l2693_269328

theorem remainder_of_division (dividend : ℕ) (divisor : ℕ) (quotient : ℕ) (remainder : ℕ) 
    (h1 : dividend = 1235678)
    (h2 : divisor = 127)
    (h3 : remainder < divisor)
    (h4 : dividend = quotient * divisor + remainder) :
  remainder = 69 := by
  sorry

end NUMINAMATH_CALUDE_remainder_of_division_l2693_269328


namespace NUMINAMATH_CALUDE_cubic_root_difference_l2693_269399

theorem cubic_root_difference : ∃ (r₁ r₂ r₃ : ℝ),
  (∀ x : ℝ, x^3 - 7*x^2 + 11*x - 6 = 0 ↔ (x = r₁ ∨ x = r₂ ∨ x = r₃)) ∧
  max r₁ (max r₂ r₃) - min r₁ (min r₂ r₃) = 2 :=
sorry

end NUMINAMATH_CALUDE_cubic_root_difference_l2693_269399


namespace NUMINAMATH_CALUDE_total_vacations_and_classes_l2693_269397

/-- The number of classes Kelvin has -/
def kelvin_classes : ℕ := 90

/-- The number of vacations Grant has -/
def grant_vacations : ℕ := 4 * kelvin_classes

/-- The total number of vacations and classes Grant and Kelvin have altogether -/
def total : ℕ := grant_vacations + kelvin_classes

theorem total_vacations_and_classes : total = 450 := by
  sorry

end NUMINAMATH_CALUDE_total_vacations_and_classes_l2693_269397


namespace NUMINAMATH_CALUDE_direct_proportion_decreasing_l2693_269320

theorem direct_proportion_decreasing (k x₁ x₂ y₁ y₂ : ℝ) :
  k < 0 →
  x₁ < x₂ →
  y₁ = k * x₁ →
  y₂ = k * x₂ →
  y₁ > y₂ := by
  sorry

end NUMINAMATH_CALUDE_direct_proportion_decreasing_l2693_269320


namespace NUMINAMATH_CALUDE_find_S_value_l2693_269345

-- Define the relationship between R, S, and T
def relationship (R S T : ℝ) : Prop :=
  ∃ c : ℝ, c > 0 ∧ ∀ R S T, R = c * S^2 / T

-- Define the initial condition
def initial_condition (R S T : ℝ) : Prop :=
  R = 2 ∧ S = 1 ∧ T = 3

-- Theorem to prove
theorem find_S_value (R S T : ℝ) :
  relationship R S T →
  initial_condition R S T →
  R = 18 ∧ T = 2 →
  S = Real.sqrt 6 := by
  sorry

end NUMINAMATH_CALUDE_find_S_value_l2693_269345


namespace NUMINAMATH_CALUDE_original_number_problem_l2693_269389

theorem original_number_problem (x : ℝ) : ((x + 5 - 2) / 4 = 7) → x = 25 := by
  sorry

end NUMINAMATH_CALUDE_original_number_problem_l2693_269389


namespace NUMINAMATH_CALUDE_package_weight_ratio_l2693_269355

theorem package_weight_ratio (x y : ℝ) (h1 : x > y) (h2 : x > 0) (h3 : y > 0) 
  (h4 : x + y = 7 * (x - y)) : x / y = 4 / 3 := by
  sorry

end NUMINAMATH_CALUDE_package_weight_ratio_l2693_269355


namespace NUMINAMATH_CALUDE_cos_135_degrees_l2693_269364

theorem cos_135_degrees :
  Real.cos (135 * π / 180) = -Real.sqrt 2 / 2 := by sorry

end NUMINAMATH_CALUDE_cos_135_degrees_l2693_269364


namespace NUMINAMATH_CALUDE_triangle_side_length_l2693_269313

-- Define the triangle PQS
structure Triangle :=
  (P Q S : ℝ × ℝ)

-- Define the length function
def length (A B : ℝ × ℝ) : ℝ := sorry

-- State the theorem
theorem triangle_side_length (PQS : Triangle) :
  length PQS.P PQS.Q = 2 → length PQS.P PQS.S = 1.5 := by
  sorry

end NUMINAMATH_CALUDE_triangle_side_length_l2693_269313


namespace NUMINAMATH_CALUDE_peach_difference_l2693_269305

/-- Proves that the difference between green and red peaches is 150 --/
theorem peach_difference : 
  let total_baskets : ℕ := 20
  let odd_red : ℕ := 12
  let odd_green : ℕ := 22
  let even_red : ℕ := 15
  let even_green : ℕ := 20
  let total_odd : ℕ := total_baskets / 2
  let total_even : ℕ := total_baskets / 2
  let total_red : ℕ := odd_red * total_odd + even_red * total_even
  let total_green : ℕ := odd_green * total_odd + even_green * total_even
  total_green - total_red = 150 := by
  sorry

end NUMINAMATH_CALUDE_peach_difference_l2693_269305


namespace NUMINAMATH_CALUDE_nonreal_cubic_root_sum_l2693_269385

/-- Given ω is a nonreal cubic root of unity, 
    prove that (2 - ω + 2ω^2)^6 + (2 + ω - 2ω^2)^6 = 38908 -/
theorem nonreal_cubic_root_sum (ω : ℂ) : 
  ω^3 = 1 → ω ≠ (1 : ℂ) → (2 - ω + 2*ω^2)^6 + (2 + ω - 2*ω^2)^6 = 38908 := by
  sorry

end NUMINAMATH_CALUDE_nonreal_cubic_root_sum_l2693_269385


namespace NUMINAMATH_CALUDE_correct_divisor_l2693_269382

/-- Represents a person with their age in years -/
structure Person where
  name : String
  age : Nat

/-- The divisor that gives Gokul's age when (Arun's age - 6) is divided by it -/
def divisor (arun gokul : Person) : Nat :=
  (arun.age - 6) / gokul.age

theorem correct_divisor (arun madan gokul : Person) : 
  arun.name = "Arun" → 
  arun.age = 60 →
  madan.name = "Madan" → 
  madan.age = 5 →
  gokul.name = "Gokul" →
  gokul.age = madan.age - 2 →
  divisor arun gokul = 18 := by
  sorry

#check correct_divisor

end NUMINAMATH_CALUDE_correct_divisor_l2693_269382


namespace NUMINAMATH_CALUDE_a_plus_2b_equals_one_l2693_269398

theorem a_plus_2b_equals_one (a b : ℝ) 
  (ha : a^3 - 21*a^2 + 140*a - 120 = 0)
  (hb : 4*b^3 - 12*b^2 - 32*b + 448 = 0) :
  a + 2*b = 1 := by
  sorry

end NUMINAMATH_CALUDE_a_plus_2b_equals_one_l2693_269398


namespace NUMINAMATH_CALUDE_number_problem_l2693_269368

theorem number_problem : ∃ x : ℝ, x = 25 ∧ (2/5) * x + 22 = (80/100) * 40 := by
  sorry

end NUMINAMATH_CALUDE_number_problem_l2693_269368


namespace NUMINAMATH_CALUDE_quadratic_minimum_value_l2693_269376

/-- Given a quadratic function f(x) = ax^2 + bx + c that is always non-negative
    and a < b, prove that (3a-2b+c)/(b-a) ≥ 1 -/
theorem quadratic_minimum_value (a b c : ℝ) 
    (h1 : ∀ x : ℝ, a * x^2 + b * x + c ≥ 0)
    (h2 : a < b) : 
    (3*a - 2*b + c) / (b - a) ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_minimum_value_l2693_269376


namespace NUMINAMATH_CALUDE_binomial_prime_divisors_l2693_269302

theorem binomial_prime_divisors (k : ℕ+) :
  ∃ (N : ℕ), ∀ (n : ℕ), n ≥ N → (Nat.choose n k.val).factors.card ≥ k.val := by
  sorry

end NUMINAMATH_CALUDE_binomial_prime_divisors_l2693_269302


namespace NUMINAMATH_CALUDE_raised_bed_width_l2693_269392

theorem raised_bed_width (num_beds : ℕ) (length height : ℝ) (num_bags : ℕ) (soil_per_bag : ℝ) :
  num_beds = 2 →
  length = 8 →
  height = 1 →
  num_bags = 16 →
  soil_per_bag = 4 →
  (num_bags : ℝ) * soil_per_bag / num_beds / (length * height) = 4 :=
by sorry

end NUMINAMATH_CALUDE_raised_bed_width_l2693_269392


namespace NUMINAMATH_CALUDE_sum_of_x_solutions_l2693_269377

theorem sum_of_x_solutions (y : ℝ) (x : ℝ → Prop) : 
  y = 5 → 
  (∀ x', x x' ↔ x'^2 + y^2 + 2*x' - 4*y = 80) → 
  (∃ a b, (x a ∧ x b) ∧ (∀ c, x c → (c = a ∨ c = b)) ∧ (a + b = -2)) :=
by sorry

end NUMINAMATH_CALUDE_sum_of_x_solutions_l2693_269377


namespace NUMINAMATH_CALUDE_max_subsets_with_intersection_property_l2693_269363

/-- The maximum number of distinct subsets satisfying the intersection property -/
theorem max_subsets_with_intersection_property (n : ℕ) :
  (∃ (t : ℕ) (A : Fin t → Finset (Fin n)),
    (∀ i j k, i < j → j < k → (A i ∩ A k) ⊆ A j) ∧
    (∀ i j, i ≠ j → A i ≠ A j)) →
  (∀ (t : ℕ) (A : Fin t → Finset (Fin n)),
    (∀ i j k, i < j → j < k → (A i ∩ A k) ⊆ A j) ∧
    (∀ i j, i ≠ j → A i ≠ A j) →
    t ≤ 2 * n + 1) :=
by sorry

end NUMINAMATH_CALUDE_max_subsets_with_intersection_property_l2693_269363


namespace NUMINAMATH_CALUDE_factor_expression_l2693_269334

theorem factor_expression (y : ℝ) : 5 * y * (y - 4) + 2 * (y - 4) = (5 * y + 2) * (y - 4) := by
  sorry

end NUMINAMATH_CALUDE_factor_expression_l2693_269334


namespace NUMINAMATH_CALUDE_range_of_m_l2693_269343

theorem range_of_m (m : ℝ) : 
  (∃ α : ℝ, 0 < α ∧ α < π / 2 ∧ Real.sqrt 3 * Real.sin α + Real.cos α = m) → 
  1 < m ∧ m ≤ 2 := by
  sorry

end NUMINAMATH_CALUDE_range_of_m_l2693_269343


namespace NUMINAMATH_CALUDE_chairs_left_theorem_l2693_269374

/-- The number of chairs left to move given the total number of chairs and the number of chairs moved by each person. -/
def chairs_left_to_move (total : ℕ) (moved_by_carey : ℕ) (moved_by_pat : ℕ) : ℕ :=
  total - (moved_by_carey + moved_by_pat)

/-- Theorem stating that given 74 total chairs, with 28 moved by Carey and 29 moved by Pat, there are 17 chairs left to move. -/
theorem chairs_left_theorem : chairs_left_to_move 74 28 29 = 17 := by
  sorry

end NUMINAMATH_CALUDE_chairs_left_theorem_l2693_269374


namespace NUMINAMATH_CALUDE_quadratic_factorization_l2693_269324

theorem quadratic_factorization (x : ℝ) :
  x^2 - 6*x - 5 = 0 ↔ (x - 3)^2 = 14 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_factorization_l2693_269324


namespace NUMINAMATH_CALUDE_cube_collinear_triples_l2693_269330

/-- Represents a point in a cube -/
inductive CubePoint
  | Vertex
  | EdgeMidpoint
  | FaceCenter
  | CubeCenter

/-- Represents a set of three collinear points in a cube -/
structure CollinearTriple where
  p1 : CubePoint
  p2 : CubePoint
  p3 : CubePoint

/-- The total number of points in the cube -/
def totalPoints : Nat := 27

/-- The number of vertices in the cube -/
def numVertices : Nat := 8

/-- The number of edge midpoints in the cube -/
def numEdgeMidpoints : Nat := 12

/-- The number of face centers in the cube -/
def numFaceCenters : Nat := 6

/-- The number of cube centers (always 1) -/
def numCubeCenters : Nat := 1

/-- Function to count the number of collinear triples in the cube -/
def countCollinearTriples : List CollinearTriple → Nat :=
  List.length

/-- Theorem: The number of sets of three collinear points in the cube is 49 -/
theorem cube_collinear_triples :
  ∃ (triples : List CollinearTriple),
    countCollinearTriples triples = 49 ∧
    totalPoints = numVertices + numEdgeMidpoints + numFaceCenters + numCubeCenters :=
  sorry

end NUMINAMATH_CALUDE_cube_collinear_triples_l2693_269330


namespace NUMINAMATH_CALUDE_fraction_to_decimal_l2693_269380

theorem fraction_to_decimal : (45 : ℚ) / 72 = 0.625 := by
  sorry

end NUMINAMATH_CALUDE_fraction_to_decimal_l2693_269380


namespace NUMINAMATH_CALUDE_de_morgan_and_jenkins_birth_years_l2693_269353

def birth_year_de_morgan (x : ℕ) : Prop :=
  x^2 - x = 1806

def birth_year_jenkins (a b m n : ℕ) : Prop :=
  (a^4 + b^4) - (a^2 + b^2) = 1860 ∧
  2 * m^2 - 2 * m = 1860 ∧
  3 * n^4 - 3 * n = 1860

theorem de_morgan_and_jenkins_birth_years :
  ∃ (x a b m n : ℕ),
    birth_year_de_morgan x ∧
    birth_year_jenkins a b m n :=
sorry

end NUMINAMATH_CALUDE_de_morgan_and_jenkins_birth_years_l2693_269353


namespace NUMINAMATH_CALUDE_range_of_a_l2693_269335

theorem range_of_a (a : ℝ) : 
  (∀ x ∈ Set.Icc 1 2, (a - 1) * x - 1 > 0 → False) ∧ 
  (∀ x : ℝ, x^2 + a*x + 1 > 0) → False ∧
  (∀ x ∈ Set.Icc 1 2, (a - 1) * x - 1 > 0 → False) ∨ 
  (∀ x : ℝ, x^2 + a*x + 1 > 0) → 
  a ≤ -2 ∨ a = 2 := by sorry

end NUMINAMATH_CALUDE_range_of_a_l2693_269335


namespace NUMINAMATH_CALUDE_pat_final_sticker_count_l2693_269304

/-- Calculates the final number of stickers Pat has at the end of the week -/
def final_sticker_count (initial : ℕ) (monday : ℕ) (tuesday : ℕ) (wednesday : ℕ) (thursday_out : ℕ) (thursday_in : ℕ) (friday : ℕ) : ℕ :=
  initial + monday - tuesday + wednesday - thursday_out + thursday_in + friday

/-- Theorem stating that Pat ends up with 43 stickers at the end of the week -/
theorem pat_final_sticker_count :
  final_sticker_count 39 15 22 10 12 8 5 = 43 := by
  sorry

end NUMINAMATH_CALUDE_pat_final_sticker_count_l2693_269304


namespace NUMINAMATH_CALUDE_circle_diameter_l2693_269318

theorem circle_diameter (r : ℝ) (h : r > 0) : 
  3 * (2 * π * r) = 2 * (π * r^2) → 2 * r = 6 := by
sorry

end NUMINAMATH_CALUDE_circle_diameter_l2693_269318


namespace NUMINAMATH_CALUDE_second_player_guarantee_seven_moves_l2693_269375

/-- Represents a game on a polygon where two players mark vertices alternately --/
structure PolygonGame where
  sides : ℕ
  -- Assume sides ≥ 3 for a valid polygon

/-- Represents a strategy for the second player --/
def SecondPlayerStrategy := PolygonGame → ℕ

/-- The maximum number of moves the second player can guarantee --/
def maxGuaranteedMoves (game : PolygonGame) : ℕ :=
  -- Implementation details omitted
  sorry

/-- Theorem stating that for a 129-sided polygon, the second player can guarantee at least 7 moves --/
theorem second_player_guarantee_seven_moves :
  ∀ (game : PolygonGame),
    game.sides = 129 →
    maxGuaranteedMoves game ≥ 7 := by
  sorry

end NUMINAMATH_CALUDE_second_player_guarantee_seven_moves_l2693_269375


namespace NUMINAMATH_CALUDE_card_addition_l2693_269315

theorem card_addition (initial_cards : ℕ) (added_cards : ℕ) : 
  initial_cards = 9 → added_cards = 4 → initial_cards + added_cards = 13 := by
sorry

end NUMINAMATH_CALUDE_card_addition_l2693_269315


namespace NUMINAMATH_CALUDE_volume_ratio_cube_sphere_l2693_269342

/-- The ratio of the volume of a cube to the volume of a sphere -/
theorem volume_ratio_cube_sphere (cube_edge : Real) (other_cube_edge : Real) : 
  cube_edge = 4 → other_cube_edge = 3 →
  (cube_edge ^ 3) / ((4/3) * π * (other_cube_edge ^ 3)) = 16 / (9 * π) := by
  sorry

end NUMINAMATH_CALUDE_volume_ratio_cube_sphere_l2693_269342


namespace NUMINAMATH_CALUDE_right_triangle_common_factor_l2693_269325

theorem right_triangle_common_factor (d : ℝ) (h_pos : d > 0) : 
  (2 * d = 45 ∨ 4 * d = 45 ∨ 5 * d = 45) ∧ 
  (2 * d)^2 + (4 * d)^2 = (5 * d)^2 → 
  d = 9 := by sorry

end NUMINAMATH_CALUDE_right_triangle_common_factor_l2693_269325


namespace NUMINAMATH_CALUDE_class_trip_cost_l2693_269354

/-- Calculates the total cost of a class trip to a science museum --/
def total_cost (num_students : ℕ) (num_teachers : ℕ) (student_ticket_price : ℚ) 
  (teacher_ticket_price : ℚ) (discount_rate : ℚ) (min_group_size : ℕ) 
  (bus_fee : ℚ) (meal_price : ℚ) : ℚ :=
  let total_people := num_students + num_teachers
  let ticket_cost := num_students * student_ticket_price + num_teachers * teacher_ticket_price
  let discounted_ticket_cost := 
    if total_people ≥ min_group_size 
    then ticket_cost * (1 - discount_rate) 
    else ticket_cost
  let meal_cost := meal_price * total_people
  discounted_ticket_cost + bus_fee + meal_cost

/-- Theorem stating the total cost for the class trip --/
theorem class_trip_cost : 
  total_cost 30 4 8 12 0.2 25 150 10 = 720.4 := by
  sorry

end NUMINAMATH_CALUDE_class_trip_cost_l2693_269354


namespace NUMINAMATH_CALUDE_binomial_eight_zero_l2693_269361

theorem binomial_eight_zero : Nat.choose 8 0 = 1 := by
  sorry

end NUMINAMATH_CALUDE_binomial_eight_zero_l2693_269361


namespace NUMINAMATH_CALUDE_plant_is_red_daisy_l2693_269336

structure Plant where
  color : String
  type : String

structure Statement where
  person : String
  plant : Plant

def is_partially_correct (actual : Plant) (statement : Statement) : Prop :=
  (actual.color = statement.plant.color) ≠ (actual.type = statement.plant.type)

theorem plant_is_red_daisy (actual : Plant) 
  (anika_statement : Statement)
  (bill_statement : Statement)
  (cathy_statement : Statement)
  (h1 : anika_statement.person = "Anika" ∧ anika_statement.plant = ⟨"red", "rose"⟩)
  (h2 : bill_statement.person = "Bill" ∧ bill_statement.plant = ⟨"purple", "daisy"⟩)
  (h3 : cathy_statement.person = "Cathy" ∧ cathy_statement.plant = ⟨"red", "dahlia"⟩)
  (h4 : is_partially_correct actual anika_statement)
  (h5 : is_partially_correct actual bill_statement)
  (h6 : is_partially_correct actual cathy_statement)
  : actual = ⟨"red", "daisy"⟩ := by
  sorry

end NUMINAMATH_CALUDE_plant_is_red_daisy_l2693_269336


namespace NUMINAMATH_CALUDE_range_of_a_l2693_269333

theorem range_of_a (a : ℝ) : 
  (∃ x : ℝ, |x + 1| + |x - a| < 4) → a ∈ Set.Ioo (-5) 3 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l2693_269333


namespace NUMINAMATH_CALUDE_max_rooks_on_chessboard_l2693_269347

/-- Represents a chessboard --/
def Chessboard := Fin 8 → Fin 8 → Bool

/-- Checks if a rook at position (x, y) attacks an odd number of rooks on the board --/
def attacks_odd (board : Chessboard) (x y : Fin 8) : Bool :=
  sorry

/-- Returns the number of rooks on the board --/
def count_rooks (board : Chessboard) : Nat :=
  sorry

/-- Checks if a board configuration is valid according to the rules --/
def is_valid_configuration (board : Chessboard) : Prop :=
  sorry

theorem max_rooks_on_chessboard :
  ∃ (board : Chessboard),
    is_valid_configuration board ∧
    count_rooks board = 63 ∧
    ∀ (other_board : Chessboard),
      is_valid_configuration other_board →
      count_rooks other_board ≤ 63 :=
by sorry

end NUMINAMATH_CALUDE_max_rooks_on_chessboard_l2693_269347


namespace NUMINAMATH_CALUDE_max_fraction_bound_l2693_269312

theorem max_fraction_bound (A B : ℕ) (hA : A ≠ 0) (hB : B ≠ 0) (hAB : A ≠ B) 
  (hA1000 : A < 1000) (hB1000 : B < 1000) : 
  (A : ℚ) - B ≤ 499 * ((A : ℚ) + B) / 500 :=
sorry

end NUMINAMATH_CALUDE_max_fraction_bound_l2693_269312


namespace NUMINAMATH_CALUDE_f_negation_property_l2693_269308

theorem f_negation_property (f : ℝ → ℝ) (a : ℝ) :
  (∀ x, f x = 2 * Real.sin x + x^3 + 1) →
  f a = 3 →
  f (-a) = -1 := by
sorry

end NUMINAMATH_CALUDE_f_negation_property_l2693_269308


namespace NUMINAMATH_CALUDE_max_sum_of_integers_l2693_269311

theorem max_sum_of_integers (A B C D : ℕ) : 
  (10 ≤ A) ∧ (A < 100) ∧
  (10 ≤ B) ∧ (B < 100) ∧
  (10 ≤ C) ∧ (C < 100) ∧
  (10 ≤ D) ∧ (D < 100) ∧
  (B = 3 * C) ∧
  (D = 2 * B - C) ∧
  (A = B + D) →
  A + B + C + D ≤ 204 :=
by
  sorry

end NUMINAMATH_CALUDE_max_sum_of_integers_l2693_269311


namespace NUMINAMATH_CALUDE_crafts_club_necklaces_l2693_269348

theorem crafts_club_necklaces 
  (members : ℕ) 
  (beads_per_necklace : ℕ) 
  (total_beads : ℕ) 
  (h1 : members = 9)
  (h2 : beads_per_necklace = 50)
  (h3 : total_beads = 900) :
  total_beads / beads_per_necklace / members = 2 := by
sorry

end NUMINAMATH_CALUDE_crafts_club_necklaces_l2693_269348


namespace NUMINAMATH_CALUDE_no_twelve_parallelepipeds_l2693_269387

/-- A rectangular parallelepiped with edges parallel to coordinate axes -/
structure RectParallelepiped where
  xRange : Set ℝ
  yRange : Set ℝ
  zRange : Set ℝ

/-- Two parallelepipeds intersect if their projections on all axes intersect -/
def intersect (p q : RectParallelepiped) : Prop :=
  (p.xRange ∩ q.xRange).Nonempty ∧
  (p.yRange ∩ q.yRange).Nonempty ∧
  (p.zRange ∩ q.zRange).Nonempty

/-- The condition for intersection based on indices -/
def shouldIntersect (i j : Fin 12) : Prop :=
  i ≠ j + 1 ∧ i ≠ j - 1

/-- The main theorem stating that 12 such parallelepipeds cannot exist -/
theorem no_twelve_parallelepipeds :
  ¬ ∃ (ps : Fin 12 → RectParallelepiped),
    ∀ (i j : Fin 12), intersect (ps i) (ps j) ↔ shouldIntersect i j :=
sorry

end NUMINAMATH_CALUDE_no_twelve_parallelepipeds_l2693_269387


namespace NUMINAMATH_CALUDE_sum_of_even_and_multiples_of_seven_l2693_269362

/-- The number of five-digit even numbers -/
def X : ℕ := 45000

/-- The number of five-digit multiples of 7 -/
def Y : ℕ := 12857

/-- The sum of five-digit even numbers and five-digit multiples of 7 -/
theorem sum_of_even_and_multiples_of_seven : X + Y = 57857 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_even_and_multiples_of_seven_l2693_269362


namespace NUMINAMATH_CALUDE_stating_football_league_equation_l2693_269327

/-- 
The number of matches in a football league where each pair of classes plays a match,
given the number of class teams.
-/
def number_of_matches (x : ℕ) : ℕ := x * (x - 1) / 2

/-- 
Theorem stating that for a football league with x class teams, where each pair plays a match,
and there are 15 matches in total, the equation relating x to the number of matches is correct.
-/
theorem football_league_equation (x : ℕ) : 
  (number_of_matches x = 15) ↔ (x * (x - 1) / 2 = 15) := by
  sorry

end NUMINAMATH_CALUDE_stating_football_league_equation_l2693_269327


namespace NUMINAMATH_CALUDE_missing_number_proof_l2693_269332

theorem missing_number_proof : ∃ x : ℝ, 0.72 * x + 0.12 * 0.34 = 0.3504 :=
  by
  use 0.43
  sorry

end NUMINAMATH_CALUDE_missing_number_proof_l2693_269332


namespace NUMINAMATH_CALUDE_basketball_weight_proof_l2693_269367

/-- The weight of one basketball in pounds -/
def basketball_weight : ℝ := 20.83333

/-- The weight of one bicycle in pounds -/
def bicycle_weight : ℝ := 37.5

/-- The weight of one skateboard in pounds -/
def skateboard_weight : ℝ := 15

theorem basketball_weight_proof :
  (9 * basketball_weight = 5 * bicycle_weight) ∧
  (2 * bicycle_weight + 3 * skateboard_weight = 120) ∧
  (skateboard_weight = 15) :=
by sorry

end NUMINAMATH_CALUDE_basketball_weight_proof_l2693_269367


namespace NUMINAMATH_CALUDE_constant_grid_values_l2693_269317

theorem constant_grid_values (f : ℤ × ℤ → ℕ) 
  (h : ∀ (x y : ℤ), f (x, y) = (f (x+1, y) + f (x-1, y) + f (x, y+1) + f (x, y-1)) / 4) : 
  ∃ (c : ℕ), ∀ (x y : ℤ), f (x, y) = c :=
sorry

end NUMINAMATH_CALUDE_constant_grid_values_l2693_269317


namespace NUMINAMATH_CALUDE_even_function_sum_l2693_269303

/-- A function f is even if f(-x) = f(x) for all x in its domain -/
def IsEven (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f x

/-- The domain of f is symmetric about the origin if its endpoints are additive inverses -/
def SymmetricDomain (a : ℝ) : Prop :=
  a - 1 = -3 * a

theorem even_function_sum (a b : ℝ) (f : ℝ → ℝ) 
    (h1 : f = fun x ↦ a * x^2 + b * x)
    (h2 : IsEven f)
    (h3 : SymmetricDomain a) : 
  a + b = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_even_function_sum_l2693_269303


namespace NUMINAMATH_CALUDE_distance_AB_is_250_l2693_269388

/-- The distance between two points A and B, where two people walk towards each other and meet under specific conditions. -/
def distance_AB : ℝ :=
  let first_meeting_distance := 100 -- meters from B
  let second_meeting_distance := 50 -- meters from A
  let total_distance := first_meeting_distance + second_meeting_distance + 100
  total_distance

/-- Theorem stating that the distance between points A and B is 250 meters. -/
theorem distance_AB_is_250 : distance_AB = 250 := by
  sorry

end NUMINAMATH_CALUDE_distance_AB_is_250_l2693_269388


namespace NUMINAMATH_CALUDE_max_xy_in_all_H_l2693_269300

-- Define the set H_n recursively
def H : ℕ → Set (ℝ × ℝ)
| 0 => {p : ℝ × ℝ | 0 ≤ p.1 ∧ p.1 ≤ 1 ∧ 0 ≤ p.2 ∧ p.2 ≤ 1}
| (n+1) => {p : ℝ × ℝ | ∃ q ∈ H n, 
    (n % 2 = 0 ∧ ((p.1 = q.1 - 1/(2^(n+2)) ∧ p.2 = q.2 + 1/(2^(n+2))) ∨ 
                  (p.1 = q.1 + 1/(2^(n+2)) ∧ p.2 = q.2 - 1/(2^(n+2))))) ∨
    (n % 2 = 1 ∧ ((p.1 = q.1 + 1/(2^(n+2)) ∧ p.2 = q.2 + 1/(2^(n+2))) ∨ 
                  (p.1 = q.1 - 1/(2^(n+2)) ∧ p.2 = q.2 - 1/(2^(n+2)))))}

-- Define the set of points that lie in all H_n
def InAllH := {p : ℝ × ℝ | ∀ n : ℕ, p ∈ H n}

-- State the theorem
theorem max_xy_in_all_H : 
  ∀ p ∈ InAllH, p.1 * p.2 ≤ 11/16 :=
sorry

end NUMINAMATH_CALUDE_max_xy_in_all_H_l2693_269300


namespace NUMINAMATH_CALUDE_always_real_roots_roots_difference_condition_l2693_269394

-- Define the quadratic equation
def quadratic_equation (m x : ℝ) : ℝ := m * x^2 - (3*m - 1) * x + (2*m - 2)

-- Theorem 1: The equation always has real roots
theorem always_real_roots (m : ℝ) : 
  ∃ x : ℝ, quadratic_equation m x = 0 :=
sorry

-- Theorem 2: If the difference between the roots is 2, then m = 1 or m = -1/3
theorem roots_difference_condition (m : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ 
    quadratic_equation m x₁ = 0 ∧ 
    quadratic_equation m x₂ = 0 ∧ 
    |x₁ - x₂| = 2) →
  (m = 1 ∨ m = -1/3) :=
sorry

end NUMINAMATH_CALUDE_always_real_roots_roots_difference_condition_l2693_269394


namespace NUMINAMATH_CALUDE_radical_equality_l2693_269395

theorem radical_equality (a b c : ℤ) :
  Real.sqrt (a + b / c) = a * Real.sqrt (b / c) ↔ c = b * (a^2 - 1) / a :=
by sorry

end NUMINAMATH_CALUDE_radical_equality_l2693_269395


namespace NUMINAMATH_CALUDE_decreasing_function_a_range_l2693_269372

-- Define the piecewise function f(x)
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≥ 1 then -4 * x + 2 * a else x^2 - a * x + 4

-- Define what it means for f to be decreasing on ℝ
def is_decreasing (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f x > f y

-- Main theorem statement
theorem decreasing_function_a_range :
  ∃ a_min a_max : ℝ, a_min = 2 ∧ a_max = 3 ∧
  (∀ a : ℝ, is_decreasing (f a) ↔ a_min ≤ a ∧ a ≤ a_max) :=
sorry

end NUMINAMATH_CALUDE_decreasing_function_a_range_l2693_269372


namespace NUMINAMATH_CALUDE_average_price_23_story_building_l2693_269306

/-- The average price per square meter of a 23-story building with specific pricing conditions. -/
theorem average_price_23_story_building (a₁ a₂ a : ℝ) : 
  let floor_prices : List ℝ := 
    [a₁] ++ (List.range 21).map (λ i => a + i * (a / 100)) ++ [a₂]
  (floor_prices.sum / 23 : ℝ) = (a₁ + a₂ + 23.1 * a) / 23 := by
  sorry

end NUMINAMATH_CALUDE_average_price_23_story_building_l2693_269306


namespace NUMINAMATH_CALUDE_final_hair_length_l2693_269396

/-- Given initial hair length x, amount cut off y, and growth z,
    prove that the final hair length F is 17 inches. -/
theorem final_hair_length
  (x y z : ℝ)
  (hx : x = 16)
  (hy : y = 11)
  (hz : z = 12)
  (hF : F = (x - y) + z) :
  F = 17 :=
by sorry

end NUMINAMATH_CALUDE_final_hair_length_l2693_269396


namespace NUMINAMATH_CALUDE_total_pumpkins_sold_l2693_269366

/-- Represents the price of a jumbo pumpkin in dollars -/
def jumbo_price : ℚ := 9

/-- Represents the price of a regular pumpkin in dollars -/
def regular_price : ℚ := 4

/-- Represents the total amount collected in dollars -/
def total_collected : ℚ := 395

/-- Represents the number of regular pumpkins sold -/
def regular_sold : ℕ := 65

/-- Theorem stating that the total number of pumpkins sold is 80 -/
theorem total_pumpkins_sold : 
  ∃ (jumbo_sold : ℕ), 
    (jumbo_price * jumbo_sold + regular_price * regular_sold = total_collected) ∧
    (jumbo_sold + regular_sold = 80) := by
  sorry

end NUMINAMATH_CALUDE_total_pumpkins_sold_l2693_269366


namespace NUMINAMATH_CALUDE_max_regions_theorem_l2693_269344

/-- Represents a circular disk with chords and secant lines. -/
structure DiskWithChords where
  n : ℕ
  chord_count : ℕ := 2 * n + 1
  secant_count : ℕ := 2

/-- Calculates the maximum number of non-overlapping regions in the disk. -/
def max_regions (disk : DiskWithChords) : ℕ :=
  8 * disk.n + 8

/-- Theorem stating the maximum number of non-overlapping regions. -/
theorem max_regions_theorem (disk : DiskWithChords) (h : disk.n > 0) :
  max_regions disk = 8 * disk.n + 8 := by
  sorry

end NUMINAMATH_CALUDE_max_regions_theorem_l2693_269344


namespace NUMINAMATH_CALUDE_mary_sugar_needed_l2693_269321

/-- Given a recipe that requires a certain amount of sugar and an amount already added,
    calculate the remaining amount of sugar needed. -/
def sugar_needed (recipe_requirement : ℕ) (already_added : ℕ) : ℕ :=
  recipe_requirement - already_added

/-- Prove that Mary needs to add 3 more cups of sugar. -/
theorem mary_sugar_needed : sugar_needed 7 4 = 3 := by
  sorry

end NUMINAMATH_CALUDE_mary_sugar_needed_l2693_269321


namespace NUMINAMATH_CALUDE_five_sixths_of_thirty_l2693_269358

theorem five_sixths_of_thirty : (5 / 6 : ℚ) * 30 = 25 := by
  sorry

end NUMINAMATH_CALUDE_five_sixths_of_thirty_l2693_269358


namespace NUMINAMATH_CALUDE_greatest_divisor_with_digit_sum_l2693_269352

def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + sum_of_digits (n / 10)

theorem greatest_divisor_with_digit_sum (a b : ℕ) (ha : a = 4665) (hb : b = 6905) :
  ∃ (n : ℕ), n = 40 ∧ 
  (b - a) % n = 0 ∧
  sum_of_digits n = 4 ∧
  ∀ (m : ℕ), m > n → ((b - a) % m = 0 → sum_of_digits m ≠ 4) :=
sorry

end NUMINAMATH_CALUDE_greatest_divisor_with_digit_sum_l2693_269352


namespace NUMINAMATH_CALUDE_max_safe_destroyers_l2693_269384

/-- Represents the configuration of ships and torpedo boats --/
structure NavalSetup where
  total_ships : Nat
  destroyers : Nat
  small_boats : Nat
  torpedo_boats : Nat
  torpedoes_per_boat : Nat

/-- Represents the targeting capabilities of torpedo boats --/
inductive TargetingStrategy
  | Successive : TargetingStrategy  -- Can target 10 successive ships
  | NextByOne : TargetingStrategy   -- Can target 10 ships next by one

/-- Defines a valid naval setup based on the problem conditions --/
def valid_setup (s : NavalSetup) : Prop :=
  s.total_ships = 30 ∧
  s.destroyers = 10 ∧
  s.small_boats = 20 ∧
  s.torpedo_boats = 2 ∧
  s.torpedoes_per_boat = 10

/-- Defines the maximum number of destroyers that can be targeted --/
def max_targeted_destroyers (s : NavalSetup) : Nat :=
  7  -- Based on the solution analysis

/-- The main theorem to be proved --/
theorem max_safe_destroyers (s : NavalSetup) 
  (h_valid : valid_setup s) :
  ∃ (safe_destroyers : Nat),
    safe_destroyers = s.destroyers - max_targeted_destroyers s ∧
    safe_destroyers = 3 :=
  sorry


end NUMINAMATH_CALUDE_max_safe_destroyers_l2693_269384


namespace NUMINAMATH_CALUDE_curtis_family_children_l2693_269391

/-- Represents the Curtis family -/
structure CurtisFamily where
  mother_age : ℕ
  father_age : ℕ
  num_children : ℕ
  children_ages : Fin num_children → ℕ

/-- The average age of the family -/
def family_average_age (f : CurtisFamily) : ℚ :=
  (f.mother_age + f.father_age + (Finset.sum Finset.univ f.children_ages)) / (2 + f.num_children)

/-- The average age of the mother and children -/
def mother_children_average_age (f : CurtisFamily) : ℚ :=
  (f.mother_age + (Finset.sum Finset.univ f.children_ages)) / (1 + f.num_children)

/-- The theorem stating the number of children in the Curtis family -/
theorem curtis_family_children (f : CurtisFamily) 
  (h1 : family_average_age f = 25)
  (h2 : f.father_age = 50)
  (h3 : mother_children_average_age f = 20) : 
  f.num_children = 4 := by
  sorry


end NUMINAMATH_CALUDE_curtis_family_children_l2693_269391


namespace NUMINAMATH_CALUDE_calvin_haircut_goal_l2693_269301

/-- Calculate the percentage of progress towards a goal -/
def progressPercentage (completed : ℕ) (total : ℕ) : ℚ :=
  (completed : ℚ) / (total : ℚ) * 100

/-- Calvin's haircut goal problem -/
theorem calvin_haircut_goal :
  let total_haircuts : ℕ := 10
  let completed_haircuts : ℕ := 8
  progressPercentage completed_haircuts total_haircuts = 80 := by
  sorry

end NUMINAMATH_CALUDE_calvin_haircut_goal_l2693_269301


namespace NUMINAMATH_CALUDE_line_through_two_points_l2693_269360

-- Define a point in 2D plane
structure Point2D where
  x : ℝ
  y : ℝ

-- Define a line equation
def lineEquation (p1 p2 : Point2D) (x y : ℝ) : Prop :=
  (y - p1.y) * (p2.x - p1.x) = (x - p1.x) * (p2.y - p1.y)

-- Theorem statement
theorem line_through_two_points (p1 p2 : Point2D) :
  ∀ x y : ℝ, (x, y) ∈ {(x, y) | lineEquation p1 p2 x y} ↔ 
  ∃ t : ℝ, x = p1.x + t * (p2.x - p1.x) ∧ y = p1.y + t * (p2.y - p1.y) :=
by sorry

end NUMINAMATH_CALUDE_line_through_two_points_l2693_269360


namespace NUMINAMATH_CALUDE_sqrt_product_quotient_l2693_269337

theorem sqrt_product_quotient : Real.sqrt 3 * Real.sqrt 10 / Real.sqrt 6 = Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_product_quotient_l2693_269337


namespace NUMINAMATH_CALUDE_percentage_less_than_y_l2693_269383

theorem percentage_less_than_y (y q w z : ℝ) 
  (hw : w = 0.6 * q) 
  (hq : q = 0.6 * y) 
  (hz : z = 1.5 * w) : 
  z = 0.54 * y := by sorry

end NUMINAMATH_CALUDE_percentage_less_than_y_l2693_269383


namespace NUMINAMATH_CALUDE_dodecahedron_interior_diagonals_l2693_269309

/-- A dodecahedron is a 3D figure with 20 vertices and 3 faces meeting at each vertex. -/
structure Dodecahedron where
  vertices : ℕ
  faces_per_vertex : ℕ
  h_vertices : vertices = 20
  h_faces_per_vertex : faces_per_vertex = 3

/-- The number of interior diagonals in a dodecahedron. -/
def interior_diagonals (d : Dodecahedron) : ℕ :=
  (d.vertices * (d.vertices - 1 - 2 * d.faces_per_vertex)) / 2

/-- Theorem: The number of interior diagonals in a dodecahedron is 160. -/
theorem dodecahedron_interior_diagonals (d : Dodecahedron) :
  interior_diagonals d = 160 := by
  sorry

end NUMINAMATH_CALUDE_dodecahedron_interior_diagonals_l2693_269309


namespace NUMINAMATH_CALUDE_num_positive_divisors_180_l2693_269378

/-- The number of positive divisors of a natural number -/
def numPositiveDivisors (n : ℕ) : ℕ := sorry

/-- The prime factorization of 180 -/
def primeFactorization180 : List (ℕ × ℕ) := [(2, 2), (3, 2), (5, 1)]

/-- Theorem: The number of positive divisors of 180 is 18 -/
theorem num_positive_divisors_180 : numPositiveDivisors 180 = 18 := by sorry

end NUMINAMATH_CALUDE_num_positive_divisors_180_l2693_269378


namespace NUMINAMATH_CALUDE_min_value_theorem_l2693_269316

theorem min_value_theorem (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h_sum : 1/a + 1/b + 1/c = 9) :
  ∀ x y z : ℝ, x > 0 → y > 0 → z > 0 → 1/x + 1/y + 1/z = 9 →
  a^4 * b^3 * c^2 ≤ x^4 * y^3 * z^2 ∧
  a^4 * b^3 * c^2 = 1/1152 := by
  sorry

end NUMINAMATH_CALUDE_min_value_theorem_l2693_269316


namespace NUMINAMATH_CALUDE_point_with_distance_6_l2693_269340

def distance_from_origin (x : ℝ) : ℝ := |x|

theorem point_with_distance_6 (A : ℝ) :
  distance_from_origin A = 6 ↔ A = 6 ∨ A = -6 := by
  sorry

end NUMINAMATH_CALUDE_point_with_distance_6_l2693_269340


namespace NUMINAMATH_CALUDE_expand_and_simplify_l2693_269323

theorem expand_and_simplify (x : ℝ) : (1 + x^3) * (1 - x^4) = 1 + x^3 - x^4 - x^7 := by
  sorry

end NUMINAMATH_CALUDE_expand_and_simplify_l2693_269323


namespace NUMINAMATH_CALUDE_lottery_expected_correct_guesses_l2693_269386

/-- Represents the number of matches in the lottery -/
def num_matches : ℕ := 12

/-- Represents the number of possible outcomes for each match -/
def num_outcomes : ℕ := 3

/-- Probability of guessing correctly for a single match -/
def p_correct : ℚ := 1 / num_outcomes

/-- Probability of guessing incorrectly for a single match -/
def p_incorrect : ℚ := 1 - p_correct

/-- Expected number of correct guesses in the lottery -/
def expected_correct_guesses : ℚ := num_matches * p_correct

theorem lottery_expected_correct_guesses :
  expected_correct_guesses = 4 := by sorry

end NUMINAMATH_CALUDE_lottery_expected_correct_guesses_l2693_269386


namespace NUMINAMATH_CALUDE_triangle_area_expression_range_l2693_269359

-- Define the triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the conditions
def arithmeticSequence (t : Triangle) : Prop :=
  ∃ d : ℝ, t.B = t.A + d ∧ t.C = t.B + d

def triangleConditions (t : Triangle) : Prop :=
  arithmeticSequence t ∧ t.b = 7 ∧ t.a + t.c = 13

-- Theorem for the area of the triangle
theorem triangle_area (t : Triangle) (h : triangleConditions t) :
  (1/2) * t.a * t.c * Real.sin t.B = 10 * Real.sqrt 3 := by sorry

-- Theorem for the range of the expression
theorem expression_range (t : Triangle) (h : triangleConditions t) :
  ∃ x : ℝ, x ∈ Set.Ioo 1 2 ∧ 
  x = Real.sqrt 3 * Real.sin t.A + Real.sin (t.C - π/6) := by sorry

end NUMINAMATH_CALUDE_triangle_area_expression_range_l2693_269359


namespace NUMINAMATH_CALUDE_tinas_career_difference_l2693_269329

/-- Represents the career of a boxer --/
structure BoxerCareer where
  initial_wins : ℕ
  additional_wins_before_first_loss : ℕ
  losses : ℕ

/-- Calculates the total wins for a boxer's career --/
def total_wins (career : BoxerCareer) : ℕ :=
  career.initial_wins + career.additional_wins_before_first_loss + 
  (career.initial_wins + career.additional_wins_before_first_loss)

/-- Theorem stating the difference between wins and losses for Tina's career --/
theorem tinas_career_difference : 
  ∀ (career : BoxerCareer), 
  career.initial_wins = 10 → 
  career.additional_wins_before_first_loss = 5 → 
  career.losses = 2 → 
  total_wins career - career.losses = 43 :=
by sorry

end NUMINAMATH_CALUDE_tinas_career_difference_l2693_269329


namespace NUMINAMATH_CALUDE_y_min_at_a_or_b_l2693_269351

/-- The function y in terms of x, a, and b -/
def y (x a b : ℝ) : ℝ := (x - a)^2 * (x - b)^2

/-- Theorem stating that the minimum of y occurs at x = a or x = b -/
theorem y_min_at_a_or_b (a b : ℝ) :
  ∃ (x : ℝ), ∀ (z : ℝ), y z a b ≥ y x a b ∧ (x = a ∨ x = b) :=
sorry

end NUMINAMATH_CALUDE_y_min_at_a_or_b_l2693_269351


namespace NUMINAMATH_CALUDE_expression_simplification_l2693_269346

theorem expression_simplification (a b c : ℝ) (ha : a = 12) (hb : b = 14) (hc : c = 18) :
  (a^2 * (1/b - 1/c) + b^2 * (1/c - 1/a) + c^2 * (1/a - 1/b)) / 
  (a * (1/b - 1/c) + b * (1/c - 1/a) + c * (1/a - 1/b)) = a + b + c := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l2693_269346


namespace NUMINAMATH_CALUDE_marble_problem_l2693_269341

theorem marble_problem (r b : ℕ) : 
  (r > 0) →
  (b > 0) →
  ((r - 1 : ℚ) / (r + b - 1) = 1 / 7) →
  (r / (r + b - 2 : ℚ) = 1 / 5) →
  (r + b = 22) :=
by sorry

end NUMINAMATH_CALUDE_marble_problem_l2693_269341


namespace NUMINAMATH_CALUDE_molecular_weight_N2O5_is_108_l2693_269365

/-- The molecular weight of N2O5 in grams per mole. -/
def molecular_weight_N2O5 : ℝ := 108

/-- The number of moles used in the given condition. -/
def given_moles : ℝ := 10

/-- The total weight of the given number of moles in grams. -/
def given_total_weight : ℝ := 1080

/-- Theorem stating that the molecular weight of N2O5 is 108 grams/mole,
    given that 10 moles of N2O5 weigh 1080 grams. -/
theorem molecular_weight_N2O5_is_108 :
  molecular_weight_N2O5 = given_total_weight / given_moles :=
by sorry

end NUMINAMATH_CALUDE_molecular_weight_N2O5_is_108_l2693_269365


namespace NUMINAMATH_CALUDE_store_clearance_sale_profit_store_profit_is_3000_l2693_269339

/-- Calculates the money left after a store's clearance sale and paying creditors -/
theorem store_clearance_sale_profit (total_items : ℕ) (original_price : ℝ) 
  (discount_percent : ℝ) (sold_percent : ℝ) (owed_to_creditors : ℝ) : ℝ :=
  let sale_price := original_price * (1 - discount_percent)
  let items_sold := total_items * sold_percent
  let total_revenue := items_sold * sale_price
  let money_left := total_revenue - owed_to_creditors
  money_left

/-- Proves that the store has $3000 left after the clearance sale and paying creditors -/
theorem store_profit_is_3000 :
  store_clearance_sale_profit 2000 50 0.8 0.9 15000 = 3000 := by
  sorry

end NUMINAMATH_CALUDE_store_clearance_sale_profit_store_profit_is_3000_l2693_269339


namespace NUMINAMATH_CALUDE_complement_of_at_least_two_defective_l2693_269322

/-- The number of products sampled -/
def n : ℕ := 10

/-- The event of having at least two defective products -/
def event_A (defective : ℕ) : Prop := defective ≥ 2

/-- The complementary event of event_A -/
def complement_A (defective : ℕ) : Prop := defective ≤ 1

/-- Theorem stating that the complement of event_A is having at most one defective product -/
theorem complement_of_at_least_two_defective :
  ∀ defective : ℕ, defective ≤ n → (¬ event_A defective ↔ complement_A defective) :=
by sorry

end NUMINAMATH_CALUDE_complement_of_at_least_two_defective_l2693_269322


namespace NUMINAMATH_CALUDE_fraction_identity_condition_l2693_269381

theorem fraction_identity_condition (a b c d : ℝ) :
  (∀ x : ℝ, (a * x + c) / (b * x + d) = (a + c * x) / (b + d * x)) →
  a / b = c / d :=
by sorry

end NUMINAMATH_CALUDE_fraction_identity_condition_l2693_269381


namespace NUMINAMATH_CALUDE_intersection_complement_theorem_l2693_269373

def M : Set ℝ := {x | x^2 > 4}
def N : Set ℝ := {x | 1 < x ∧ x ≤ 3}

theorem intersection_complement_theorem :
  N ∩ (Set.univ \ M) = {x | 1 < x ∧ x ≤ 2} := by sorry

end NUMINAMATH_CALUDE_intersection_complement_theorem_l2693_269373


namespace NUMINAMATH_CALUDE_equation_solutions_l2693_269349

theorem equation_solutions :
  ∀ a b : ℤ, 3 * a^2 * b^2 + b^2 = 517 + 30 * a^2 ↔ 
  ((a = 2 ∧ b = 7) ∨ (a = -2 ∧ b = 7) ∨ (a = 2 ∧ b = -7) ∨ (a = -2 ∧ b = -7)) :=
by sorry

end NUMINAMATH_CALUDE_equation_solutions_l2693_269349


namespace NUMINAMATH_CALUDE_total_contribution_l2693_269319

def contribution_problem (niraj brittany angela : ℕ) : Prop :=
  brittany = 3 * niraj ∧
  angela = 3 * brittany ∧
  niraj = 80

theorem total_contribution :
  ∀ niraj brittany angela : ℕ,
  contribution_problem niraj brittany angela →
  niraj + brittany + angela = 1040 :=
by
  sorry

end NUMINAMATH_CALUDE_total_contribution_l2693_269319


namespace NUMINAMATH_CALUDE_astronomers_use_analogical_reasoning_l2693_269370

/-- Represents a celestial body in the solar system -/
structure CelestialBody where
  name : String
  hasLife : Bool

/-- Represents a type of reasoning -/
inductive ReasoningType
  | Inductive
  | Analogical
  | Deductive
  | ProofByContradiction

/-- Determines if two celestial bodies are similar -/
def areSimilar (a b : CelestialBody) : Bool := sorry

/-- Represents the astronomers' reasoning process -/
def astronomersReasoning (earth mars : CelestialBody) : ReasoningType :=
  if areSimilar earth mars ∧ earth.hasLife then
    ReasoningType.Analogical
  else
    sorry

/-- Theorem stating that the astronomers' reasoning is analogical -/
theorem astronomers_use_analogical_reasoning (earth mars : CelestialBody) 
  (h1 : areSimilar earth mars = true)
  (h2 : earth.hasLife = true) :
  astronomersReasoning earth mars = ReasoningType.Analogical := by
  sorry

end NUMINAMATH_CALUDE_astronomers_use_analogical_reasoning_l2693_269370


namespace NUMINAMATH_CALUDE_inequality_holds_iff_a_in_range_l2693_269371

theorem inequality_holds_iff_a_in_range :
  ∀ a : ℝ, (∀ x : ℝ, x ≤ 1 → (1 + 2^x + 4^x * a) / (a^2 - a + 1) > 0) ↔ a > -3/4 :=
by sorry

end NUMINAMATH_CALUDE_inequality_holds_iff_a_in_range_l2693_269371


namespace NUMINAMATH_CALUDE_log_equation_solution_l2693_269393

theorem log_equation_solution (p q : ℝ) (hp : p > 0) (hq : q > 0) (hq1 : q ≠ 1) :
  Real.log p + Real.log (q^2) = Real.log (p + q^2) ↔ p = q^2 / (q^2 - 1) :=
by sorry

end NUMINAMATH_CALUDE_log_equation_solution_l2693_269393
