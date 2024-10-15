import Mathlib

namespace NUMINAMATH_CALUDE_expression_evaluation_l481_48197

theorem expression_evaluation :
  let f (x : ℚ) := (2*x - 3) / (x + 2)
  let g (x : ℚ) := (2*(f x) - 3) / (f x + 2)
  g 2 = -10/9 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l481_48197


namespace NUMINAMATH_CALUDE_singing_competition_ratio_l481_48131

/-- Proves that the ratio of female contestants to the total number of contestants is 1/3 -/
theorem singing_competition_ratio :
  let total_contestants : ℕ := 18
  let male_contestants : ℕ := 12
  let female_contestants : ℕ := total_contestants - male_contestants
  (female_contestants : ℚ) / total_contestants = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_singing_competition_ratio_l481_48131


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_find_S_5_l481_48157

/-- Given an arithmetic sequence {aₙ}, Sₙ represents the sum of its first n terms -/
def S (n : ℕ) : ℝ := sorry

/-- aₙ represents the nth term of the arithmetic sequence -/
def a (n : ℕ) : ℝ := sorry

/-- d represents the common difference of the arithmetic sequence -/
def d : ℝ := sorry

theorem arithmetic_sequence_sum (n : ℕ) :
  S n = n * a 1 + (n * (n - 1) / 2) * d := sorry

axiom sum_condition : S 3 + S 6 = 18

theorem find_S_5 : S 5 = 10 := by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_find_S_5_l481_48157


namespace NUMINAMATH_CALUDE_cube_sum_equals_one_l481_48193

theorem cube_sum_equals_one (x y : ℝ) 
  (h1 : x * (x^4 + y^4) = y^5) 
  (h2 : x^2 * (x + y) ≠ y^3) : 
  x^3 + y^3 = 1 := by
sorry

end NUMINAMATH_CALUDE_cube_sum_equals_one_l481_48193


namespace NUMINAMATH_CALUDE_least_value_of_x_l481_48153

theorem least_value_of_x (x p q : ℕ) : 
  x > 0 →
  Nat.Prime p →
  Nat.Prime q →
  p < q →
  x / (12 * p * q) = 2 →
  2 * p - q = 3 →
  ∀ y, y > 0 ∧ 
       ∃ p' q', Nat.Prime p' ∧ Nat.Prime q' ∧ p' < q' ∧
                y / (12 * p' * q') = 2 ∧
                2 * p' - q' = 3 →
       x ≤ y →
  x = 840 := by
sorry


end NUMINAMATH_CALUDE_least_value_of_x_l481_48153


namespace NUMINAMATH_CALUDE_extra_large_posters_count_l481_48185

def total_posters : ℕ := 200

def small_posters : ℕ := total_posters / 4
def medium_posters : ℕ := total_posters / 3
def large_posters : ℕ := total_posters / 5

def extra_large_posters : ℕ := total_posters - (small_posters + medium_posters + large_posters)

theorem extra_large_posters_count : extra_large_posters = 44 := by
  sorry

end NUMINAMATH_CALUDE_extra_large_posters_count_l481_48185


namespace NUMINAMATH_CALUDE_shirt_price_l481_48108

/-- Given the sales of shoes and shirts, prove the price of each shirt -/
theorem shirt_price (num_shoes : ℕ) (shoe_price : ℚ) (num_shirts : ℕ) (total_earnings_per_person : ℚ) :
  num_shoes = 6 →
  shoe_price = 3 →
  num_shirts = 18 →
  total_earnings_per_person = 27 →
  ∃ (shirt_price : ℚ), 
    (↑num_shoes * shoe_price + ↑num_shirts * shirt_price) / 2 = total_earnings_per_person ∧
    shirt_price = 2 :=
by sorry

end NUMINAMATH_CALUDE_shirt_price_l481_48108


namespace NUMINAMATH_CALUDE_planned_goats_addition_l481_48118

/-- Represents the number of animals on the farm -/
structure FarmAnimals where
  cows : ℕ
  pigs : ℕ
  goats : ℕ

/-- Calculates the total number of animals -/
def totalAnimals (farm : FarmAnimals) : ℕ :=
  farm.cows + farm.pigs + farm.goats

/-- The initial number of animals on the farm -/
def initialFarm : FarmAnimals :=
  { cows := 2, pigs := 3, goats := 6 }

/-- The planned additions to the farm -/
def plannedAdditions : FarmAnimals :=
  { cows := 3, pigs := 5, goats := 0 }

/-- The final desired number of animals -/
def finalTotal : ℕ := 21

/-- Theorem: The number of goats the farmer plans to add is 2 -/
theorem planned_goats_addition :
  finalTotal = totalAnimals initialFarm + totalAnimals plannedAdditions + 2 := by
  sorry

end NUMINAMATH_CALUDE_planned_goats_addition_l481_48118


namespace NUMINAMATH_CALUDE_quadratic_completion_l481_48147

theorem quadratic_completion (y : ℝ) : ∃ (k : ℤ) (a : ℝ), y^2 + 12*y + 40 = (y + a)^2 + k := by
  sorry

end NUMINAMATH_CALUDE_quadratic_completion_l481_48147


namespace NUMINAMATH_CALUDE_min_value_theorem_l481_48168

theorem min_value_theorem (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : 3 * a + 2 * b = 1) :
  (∀ x y : ℝ, x > 0 → y > 0 → 3 * x + 2 * y = 1 → 2 / x + 3 / y ≥ 2 / a + 3 / b) →
  2 / a + 3 / b = 24 :=
by sorry

end NUMINAMATH_CALUDE_min_value_theorem_l481_48168


namespace NUMINAMATH_CALUDE_cells_reach_1540_in_9_hours_l481_48141

/-- The number of cells after n hours -/
def cell_count (n : ℕ) : ℕ :=
  3 * 2^(n-1) + 4

/-- The theorem stating that it takes 9 hours to reach 1540 cells -/
theorem cells_reach_1540_in_9_hours :
  cell_count 9 = 1540 ∧
  ∀ k : ℕ, k < 9 → cell_count k < 1540 :=
by sorry

end NUMINAMATH_CALUDE_cells_reach_1540_in_9_hours_l481_48141


namespace NUMINAMATH_CALUDE_pizza_problem_l481_48143

/-- Represents a pizza with a given number of slices and topping distribution. -/
structure Pizza where
  total_slices : ℕ
  pepperoni_slices : ℕ
  mushroom_slices : ℕ
  both_toppings_slices : ℕ

/-- The pizza satisfies the given conditions. -/
def valid_pizza (p : Pizza) : Prop :=
  p.total_slices = 15 ∧
  p.pepperoni_slices = 8 ∧
  p.mushroom_slices = 12 ∧
  p.both_toppings_slices ≤ p.pepperoni_slices ∧
  p.both_toppings_slices ≤ p.mushroom_slices ∧
  p.pepperoni_slices + p.mushroom_slices - p.both_toppings_slices = p.total_slices

theorem pizza_problem (p : Pizza) (h : valid_pizza p) : p.both_toppings_slices = 5 := by
  sorry

end NUMINAMATH_CALUDE_pizza_problem_l481_48143


namespace NUMINAMATH_CALUDE_smallest_multiple_35_with_digit_product_35_l481_48122

def is_multiple_of_35 (n : ℕ) : Prop := ∃ k : ℕ, n = 35 * k

def digit_product (n : ℕ) : ℕ :=
  if n = 0 then 1 else
  let digits := n.digits 10
  digits.prod

theorem smallest_multiple_35_with_digit_product_35 :
  ∀ n : ℕ, n > 0 → is_multiple_of_35 n → is_multiple_of_35 (digit_product n) →
  n ≥ 735 := by sorry

end NUMINAMATH_CALUDE_smallest_multiple_35_with_digit_product_35_l481_48122


namespace NUMINAMATH_CALUDE_sum_of_powers_l481_48116

theorem sum_of_powers (a b : ℝ) : 
  (a + b = 1) → 
  (a^2 + b^2 = 3) → 
  (a^3 + b^3 = 4) → 
  (a^4 + b^4 = 7) → 
  (a^5 + b^5 = 11) → 
  (∀ n ≥ 3, a^n + b^n = (a^(n-1) + b^(n-1)) + (a^(n-2) + b^(n-2))) →
  a^6 + b^6 = 18 := by
sorry

end NUMINAMATH_CALUDE_sum_of_powers_l481_48116


namespace NUMINAMATH_CALUDE_square_difference_305_301_l481_48167

theorem square_difference_305_301 : 305^2 - 301^2 = 2424 := by sorry

end NUMINAMATH_CALUDE_square_difference_305_301_l481_48167


namespace NUMINAMATH_CALUDE_functional_equation_zero_function_l481_48105

theorem functional_equation_zero_function 
  (f : ℝ → ℝ) 
  (h : ∀ x y : ℝ, f (x + y) = x * f x + y * f y) : 
  ∀ x : ℝ, f x = 0 := by
sorry

end NUMINAMATH_CALUDE_functional_equation_zero_function_l481_48105


namespace NUMINAMATH_CALUDE_factorization_problem_1_factorization_problem_2_l481_48163

-- Problem 1
theorem factorization_problem_1 (x : ℝ) : 2*x^2 - 4*x = 2*x*(x - 2) := by sorry

-- Problem 2
theorem factorization_problem_2 (x y : ℝ) : x*y^2 - 2*x*y + x = x*(y - 1)^2 := by sorry

end NUMINAMATH_CALUDE_factorization_problem_1_factorization_problem_2_l481_48163


namespace NUMINAMATH_CALUDE_fraction_decomposition_l481_48151

theorem fraction_decomposition (x : ℚ) (A B : ℚ) 
  (h : x ≠ -5 ∧ x ≠ 2/3) : 
  (7 * x - 13) / (3 * x^2 + 13 * x - 10) = A / (x + 5) + B / (3 * x - 2) → 
  A = 48/17 ∧ B = -25/17 :=
by sorry

end NUMINAMATH_CALUDE_fraction_decomposition_l481_48151


namespace NUMINAMATH_CALUDE_square_last_digit_l481_48171

theorem square_last_digit (n : ℕ) 
  (h : (n^2 / 10) % 10 = 7) : 
  n^2 % 10 = 6 := by
sorry

end NUMINAMATH_CALUDE_square_last_digit_l481_48171


namespace NUMINAMATH_CALUDE_group_payment_l481_48130

/-- Calculates the total amount paid by a group of moviegoers -/
def total_amount_paid (adult_price child_price : ℚ) (total_people adults : ℕ) : ℚ :=
  adult_price * adults + child_price * (total_people - adults)

/-- Theorem: The group paid $54.50 in total -/
theorem group_payment : total_amount_paid 9.5 6.5 7 3 = 54.5 := by
  sorry

end NUMINAMATH_CALUDE_group_payment_l481_48130


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l481_48176

theorem quadratic_inequality_solution (x : ℝ) :
  -x^2 - 2*x + 3 < 0 ↔ x < -3 ∨ x > 1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l481_48176


namespace NUMINAMATH_CALUDE_complementary_sets_count_l481_48120

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

end NUMINAMATH_CALUDE_complementary_sets_count_l481_48120


namespace NUMINAMATH_CALUDE_kyle_fish_count_l481_48106

/-- Given that Carla, Kyle, and Tasha caught a total of 36 fish, 
    Carla caught 8 fish, and Kyle and Tasha caught the same number of fish,
    prove that Kyle caught 14 fish. -/
theorem kyle_fish_count (total : ℕ) (carla : ℕ) (kyle : ℕ) (tasha : ℕ)
  (h1 : total = 36)
  (h2 : carla = 8)
  (h3 : kyle = tasha)
  (h4 : total = carla + kyle + tasha) :
  kyle = 14 := by
  sorry

end NUMINAMATH_CALUDE_kyle_fish_count_l481_48106


namespace NUMINAMATH_CALUDE_complex_equality_implies_modulus_l481_48187

theorem complex_equality_implies_modulus (x y : ℝ) :
  (Complex.I + 1) * Complex.mk x y = 2 →
  Complex.abs (Complex.mk (2*x) y) = Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_complex_equality_implies_modulus_l481_48187


namespace NUMINAMATH_CALUDE_trigonometric_expression_equals_one_l481_48113

theorem trigonometric_expression_equals_one :
  (Real.tan (45 * π / 180))^2 - (Real.sin (45 * π / 180))^2 = 
  (Real.tan (45 * π / 180))^2 * (Real.sin (45 * π / 180))^2 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_expression_equals_one_l481_48113


namespace NUMINAMATH_CALUDE_min_value_weighted_sum_l481_48128

theorem min_value_weighted_sum (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h_sum : 2*x + 3*y + 4*z = 1) :
  (4/x) + (9/y) + (8/z) ≥ 81 ∧ 
  ∃ (x₀ y₀ z₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ z₀ > 0 ∧ 
    2*x₀ + 3*y₀ + 4*z₀ = 1 ∧ (4/x₀) + (9/y₀) + (8/z₀) = 81 := by
  sorry

end NUMINAMATH_CALUDE_min_value_weighted_sum_l481_48128


namespace NUMINAMATH_CALUDE_pump_fill_time_l481_48186

theorem pump_fill_time (P : ℝ) (h1 : P > 0) (h2 : 14 > 0) :
  1 / P - 1 / 14 = 1 / (7 / 3) → P = 2 := by
  sorry

end NUMINAMATH_CALUDE_pump_fill_time_l481_48186


namespace NUMINAMATH_CALUDE_quadratic_value_theorem_l481_48138

theorem quadratic_value_theorem (x : ℝ) : 
  x^2 - 2*x - 3 = 0 → 2*x^2 - 4*x + 12 = 18 := by
sorry

end NUMINAMATH_CALUDE_quadratic_value_theorem_l481_48138


namespace NUMINAMATH_CALUDE_spoiled_milk_percentage_l481_48159

theorem spoiled_milk_percentage
  (egg_rotten_percentage : ℝ)
  (flour_weevil_percentage : ℝ)
  (all_good_probability : ℝ)
  (h1 : egg_rotten_percentage = 60)
  (h2 : flour_weevil_percentage = 25)
  (h3 : all_good_probability = 24) :
  ∃ (spoiled_milk_percentage : ℝ),
    spoiled_milk_percentage = 20 ∧
    (100 - spoiled_milk_percentage) / 100 * (100 - egg_rotten_percentage) / 100 * (100 - flour_weevil_percentage) / 100 = all_good_probability / 100 :=
by sorry

end NUMINAMATH_CALUDE_spoiled_milk_percentage_l481_48159


namespace NUMINAMATH_CALUDE_snowman_volume_l481_48144

theorem snowman_volume (π : ℝ) (h : π > 0) : 
  let sphere_volume (r : ℝ) := (4 / 3) * π * r^3
  sphere_volume 4 + sphere_volume 6 + sphere_volume 8 + sphere_volume 10 = (7168 / 3) * π :=
by sorry

end NUMINAMATH_CALUDE_snowman_volume_l481_48144


namespace NUMINAMATH_CALUDE_sector_area_l481_48139

/-- Given a sector with perimeter 10 and central angle 2 radians, its area is 25/4 -/
theorem sector_area (r : ℝ) (l : ℝ) (h1 : l + 2*r = 10) (h2 : l = 2*r) : 
  (1/2) * r * l = 25/4 := by
  sorry

end NUMINAMATH_CALUDE_sector_area_l481_48139


namespace NUMINAMATH_CALUDE_negation_of_forall_positive_square_plus_x_l481_48160

theorem negation_of_forall_positive_square_plus_x :
  (¬ ∀ x : ℝ, x^2 + x > 0) ↔ (∃ x : ℝ, x^2 + x ≤ 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_forall_positive_square_plus_x_l481_48160


namespace NUMINAMATH_CALUDE_x_convergence_bound_l481_48133

def x : ℕ → ℚ
  | 0 => 3
  | n + 1 => (x n ^ 2 + 6 * x n + 8) / (x n + 7)

theorem x_convergence_bound :
  ∃ m : ℕ, 31 ≤ m ∧ m ≤ 90 ∧ 
    x m ≤ 5 + 1 / (2^15) ∧
    ∀ k : ℕ, 0 < k → k < m → x k > 5 + 1 / (2^15) := by
  sorry

end NUMINAMATH_CALUDE_x_convergence_bound_l481_48133


namespace NUMINAMATH_CALUDE_field_trip_arrangements_l481_48135

/-- The number of grades --/
def num_grades : ℕ := 6

/-- The number of museums --/
def num_museums : ℕ := 6

/-- The number of grades that choose Museum A --/
def grades_choosing_a : ℕ := 2

/-- The number of ways to choose exactly two grades to visit Museum A --/
def ways_to_choose_a : ℕ := Nat.choose num_grades grades_choosing_a

/-- The number of museums excluding Museum A --/
def remaining_museums : ℕ := num_museums - 1

/-- The number of grades not choosing Museum A --/
def grades_not_choosing_a : ℕ := num_grades - grades_choosing_a

/-- The total number of ways to arrange the field trip --/
def total_arrangements : ℕ := ways_to_choose_a * (remaining_museums ^ grades_not_choosing_a)

theorem field_trip_arrangements :
  total_arrangements = Nat.choose num_grades grades_choosing_a * (remaining_museums ^ grades_not_choosing_a) :=
by sorry

end NUMINAMATH_CALUDE_field_trip_arrangements_l481_48135


namespace NUMINAMATH_CALUDE_basketball_team_selection_l481_48127

-- Define the total number of players
def total_players : ℕ := 16

-- Define the number of quadruplets
def num_quadruplets : ℕ := 4

-- Define the number of starters to choose
def num_starters : ℕ := 7

-- Define the number of quadruplets that must be in the starting lineup
def quadruplets_in_lineup : ℕ := 3

-- Theorem statement
theorem basketball_team_selection :
  (Nat.choose num_quadruplets quadruplets_in_lineup) *
  (Nat.choose (total_players - num_quadruplets) (num_starters - quadruplets_in_lineup)) = 1980 := by
  sorry

end NUMINAMATH_CALUDE_basketball_team_selection_l481_48127


namespace NUMINAMATH_CALUDE_wheel_probability_l481_48119

theorem wheel_probability (p_D p_E p_FG : ℚ) : 
  p_D = 1/4 → p_E = 1/3 → p_D + p_E + p_FG = 1 → p_FG = 5/12 := by
  sorry

end NUMINAMATH_CALUDE_wheel_probability_l481_48119


namespace NUMINAMATH_CALUDE_ratio_problem_l481_48134

theorem ratio_problem (q r s t : ℚ) 
  (h1 : q / r = 12)
  (h2 : s / r = 8)
  (h3 : s / t = 1 / 3) :
  t / q = 2 := by sorry

end NUMINAMATH_CALUDE_ratio_problem_l481_48134


namespace NUMINAMATH_CALUDE_tangent_fraction_equality_l481_48196

theorem tangent_fraction_equality (α : Real) (h : Real.tan α = 3) :
  (4 * Real.sin α - 2 * Real.cos α) / (5 * Real.cos α + 3 * Real.sin α) = 5 / 7 := by
  sorry

end NUMINAMATH_CALUDE_tangent_fraction_equality_l481_48196


namespace NUMINAMATH_CALUDE_particle_position_after_3000_minutes_l481_48103

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

end NUMINAMATH_CALUDE_particle_position_after_3000_minutes_l481_48103


namespace NUMINAMATH_CALUDE_equilateral_triangle_perimeter_area_ratio_l481_48124

/-- The ratio of the perimeter to the area of an equilateral triangle with side length 6 -/
theorem equilateral_triangle_perimeter_area_ratio :
  let side_length : ℝ := 6
  let perimeter : ℝ := 3 * side_length
  let area : ℝ := (side_length^2 * Real.sqrt 3) / 4
  perimeter / area = 2 * Real.sqrt 3 / 3 := by sorry

end NUMINAMATH_CALUDE_equilateral_triangle_perimeter_area_ratio_l481_48124


namespace NUMINAMATH_CALUDE_circle_center_and_radius_l481_48150

/-- A circle in the 2D plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- The equation of the circle: x^2 + y^2 - 4x = 0 -/
def circle_equation (x y : ℝ) : Prop :=
  x^2 + y^2 - 4*x = 0

/-- The circle defined by the equation -/
def given_circle : Circle :=
  { center := (2, 0),
    radius := 2 }

/-- Theorem: The given equation defines a circle with center (2, 0) and radius 2 -/
theorem circle_center_and_radius :
  ∀ x y : ℝ, circle_equation x y ↔ 
    (x - given_circle.center.1)^2 + (y - given_circle.center.2)^2 = given_circle.radius^2 :=
by sorry

end NUMINAMATH_CALUDE_circle_center_and_radius_l481_48150


namespace NUMINAMATH_CALUDE_sufficient_but_not_necessary_l481_48154

theorem sufficient_but_not_necessary :
  (∃ p q : Prop, (p ∨ q = False) → (¬p = True)) ∧
  (∃ p q : Prop, (¬p = True) ∧ ¬(p ∨ q = False)) := by
  sorry

end NUMINAMATH_CALUDE_sufficient_but_not_necessary_l481_48154


namespace NUMINAMATH_CALUDE_stacys_farm_goats_l481_48177

/-- Calculates the number of goats on Stacy's farm given the conditions --/
theorem stacys_farm_goats (chickens : ℕ) (piglets : ℕ) (sick_animals : ℕ) :
  chickens = 26 →
  piglets = 40 →
  sick_animals = 50 →
  (chickens + piglets + (34 : ℕ)) / 2 = sick_animals →
  34 = (2 * sick_animals) - chickens - piglets :=
by
  sorry

end NUMINAMATH_CALUDE_stacys_farm_goats_l481_48177


namespace NUMINAMATH_CALUDE_min_sum_squares_l481_48180

def S : Finset Int := {-9, -4, -3, 0, 1, 5, 8, 10}

theorem min_sum_squares (p q r s t u v w : Int) 
  (h_distinct : p ≠ q ∧ p ≠ r ∧ p ≠ s ∧ p ≠ t ∧ p ≠ u ∧ p ≠ v ∧ p ≠ w ∧
                q ≠ r ∧ q ≠ s ∧ q ≠ t ∧ q ≠ u ∧ q ≠ v ∧ q ≠ w ∧
                r ≠ s ∧ r ≠ t ∧ r ≠ u ∧ r ≠ v ∧ r ≠ w ∧
                s ≠ t ∧ s ≠ u ∧ s ≠ v ∧ s ≠ w ∧
                t ≠ u ∧ t ≠ v ∧ t ≠ w ∧
                u ≠ v ∧ u ≠ w ∧
                v ≠ w)
  (h_in_S : p ∈ S ∧ q ∈ S ∧ r ∈ S ∧ s ∈ S ∧ t ∈ S ∧ u ∈ S ∧ v ∈ S ∧ w ∈ S) :
  32 ≤ (p + q + r + s)^2 + (t + u + v + w)^2 :=
by
  sorry

#check min_sum_squares

end NUMINAMATH_CALUDE_min_sum_squares_l481_48180


namespace NUMINAMATH_CALUDE_inequality_system_solution_range_l481_48114

theorem inequality_system_solution_range (m : ℝ) : 
  (∃ x₁ x₂ : ℤ, x₁ ≠ x₂ ∧ 
    (2 * ↑x₁ - 1 ≤ 5 ∧ ↑x₁ - 1 ≥ m) ∧ 
    (2 * ↑x₂ - 1 ≤ 5 ∧ ↑x₂ - 1 ≥ m) ∧
    (∀ x : ℤ, (2 * ↑x - 1 ≤ 5 ∧ ↑x - 1 ≥ m) → (x = x₁ ∨ x = x₂))) ↔
  (-1 < m ∧ m ≤ 0) :=
by sorry

end NUMINAMATH_CALUDE_inequality_system_solution_range_l481_48114


namespace NUMINAMATH_CALUDE_average_age_proof_l481_48169

/-- Given three people a, b, and c, this theorem proves that if their average age is 28 years
    and the age of b is 26 years, then the average age of a and c is 29 years. -/
theorem average_age_proof (a b c : ℕ) : 
  (a + b + c) / 3 = 28 → b = 26 → (a + c) / 2 = 29 := by
  sorry

end NUMINAMATH_CALUDE_average_age_proof_l481_48169


namespace NUMINAMATH_CALUDE_roja_work_time_l481_48104

/-- Given that Malar and Roja combined complete a task in 35 days,
    and Malar alone completes the same work in 60 days,
    prove that Roja alone can complete the work in 210 days. -/
theorem roja_work_time (combined_time malar_time : ℝ)
  (h_combined : combined_time = 35)
  (h_malar : malar_time = 60) :
  let roja_time := (combined_time * malar_time) / (malar_time - combined_time)
  roja_time = 210 := by
sorry

end NUMINAMATH_CALUDE_roja_work_time_l481_48104


namespace NUMINAMATH_CALUDE_unique_B_for_divisibility_l481_48164

/-- Represents a four-digit number in the form 4BB2 -/
def fourDigitNumber (B : ℕ) : ℕ := 4000 + 100 * B + 10 * B + 2

/-- Checks if a number is divisible by 11 -/
def divisibleBy11 (n : ℕ) : Prop := n % 11 = 0

/-- B is a single digit -/
def isSingleDigit (B : ℕ) : Prop := B ≥ 0 ∧ B ≤ 9

theorem unique_B_for_divisibility : 
  ∃! B : ℕ, isSingleDigit B ∧ divisibleBy11 (fourDigitNumber B) ∧ B = 3 :=
sorry

end NUMINAMATH_CALUDE_unique_B_for_divisibility_l481_48164


namespace NUMINAMATH_CALUDE_time_to_cook_one_potato_l481_48109

theorem time_to_cook_one_potato 
  (total_potatoes : ℕ) 
  (cooked_potatoes : ℕ) 
  (time_for_remaining : ℕ) : 
  total_potatoes = 13 → 
  cooked_potatoes = 5 → 
  time_for_remaining = 48 → 
  (time_for_remaining / (total_potatoes - cooked_potatoes) : ℚ) = 6 := by
  sorry

end NUMINAMATH_CALUDE_time_to_cook_one_potato_l481_48109


namespace NUMINAMATH_CALUDE_sum_of_three_numbers_l481_48194

theorem sum_of_three_numbers (a b c : ℝ) 
  (h1 : a^2 + b^2 + c^2 = 62) 
  (h2 : a*b + b*c + c*a = 131) : 
  a + b + c = 18 := by
sorry

end NUMINAMATH_CALUDE_sum_of_three_numbers_l481_48194


namespace NUMINAMATH_CALUDE_marilyn_bananas_count_l481_48199

/-- The number of boxes Marilyn has for her bananas. -/
def num_boxes : ℕ := 8

/-- The number of bananas required in each box. -/
def bananas_per_box : ℕ := 5

/-- Theorem stating that Marilyn has 40 bananas in total. -/
theorem marilyn_bananas_count :
  num_boxes * bananas_per_box = 40 := by
  sorry

end NUMINAMATH_CALUDE_marilyn_bananas_count_l481_48199


namespace NUMINAMATH_CALUDE_product_equals_fraction_l481_48172

def product_term (n : ℕ) : ℚ :=
  (2 * (n^4 - 1)) / (2 * (n^4 + 1))

def product_result : ℚ :=
  (product_term 2) * (product_term 3) * (product_term 4) * 
  (product_term 5) * (product_term 6) * (product_term 7)

theorem product_equals_fraction : product_result = 4400 / 135 := by
  sorry

end NUMINAMATH_CALUDE_product_equals_fraction_l481_48172


namespace NUMINAMATH_CALUDE_flower_bed_length_l481_48174

/-- A rectangular flower bed with given area and width has a specific length -/
theorem flower_bed_length (area width : ℝ) (h1 : area = 35) (h2 : width = 5) :
  area / width = 7 := by
  sorry

end NUMINAMATH_CALUDE_flower_bed_length_l481_48174


namespace NUMINAMATH_CALUDE_olivias_score_l481_48121

theorem olivias_score (n : ℕ) (avg_without : ℚ) (avg_with : ℚ) :
  n = 20 →
  avg_without = 85 →
  avg_with = 86 →
  (n * avg_without + x) / (n + 1) = avg_with →
  x = 106 :=
by sorry

end NUMINAMATH_CALUDE_olivias_score_l481_48121


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l481_48161

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_sum (a : ℕ → ℝ) (h : arithmetic_sequence a) :
  (a 2 + a 11 = 3) → (a 5 + a 8 = 3) := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l481_48161


namespace NUMINAMATH_CALUDE_dmitry_black_socks_l481_48110

/-- Proves that Dmitry bought 22 pairs of black socks -/
theorem dmitry_black_socks :
  let initial_blue : ℕ := 10
  let initial_black : ℕ := 22
  let initial_white : ℕ := 12
  let bought_black : ℕ := x
  let total_initial : ℕ := initial_blue + initial_black + initial_white
  let total_after : ℕ := total_initial + bought_black
  let black_after : ℕ := initial_black + bought_black
  (black_after : ℚ) / (total_after : ℚ) = 2 / 3 →
  x = 22 := by
sorry

end NUMINAMATH_CALUDE_dmitry_black_socks_l481_48110


namespace NUMINAMATH_CALUDE_boat_speed_l481_48102

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

end NUMINAMATH_CALUDE_boat_speed_l481_48102


namespace NUMINAMATH_CALUDE_cubic_factorization_l481_48198

theorem cubic_factorization (t : ℝ) : t^3 - 144 = (t - 12) * (t^2 + 12*t + 144) := by
  sorry

end NUMINAMATH_CALUDE_cubic_factorization_l481_48198


namespace NUMINAMATH_CALUDE_surtido_criterion_l481_48166

def sum_of_digits (A : ℕ) : ℕ := sorry

def is_sum_of_digits (A n : ℕ) : Prop := sorry

def is_surtido (A : ℕ) : Prop :=
  ∀ k : ℕ, 1 ≤ k ∧ k ≤ sum_of_digits A → is_sum_of_digits A k

theorem surtido_criterion (A : ℕ) :
  (∀ k : ℕ, 1 ≤ k ∧ k ≤ 8 → is_sum_of_digits A k) → is_surtido A := by sorry

end NUMINAMATH_CALUDE_surtido_criterion_l481_48166


namespace NUMINAMATH_CALUDE_production_line_b_l481_48149

def total_production : ℕ := 5000

def sampling_ratio : List ℕ := [1, 2, 2]

theorem production_line_b (a b c : ℕ) : 
  a + b + c = total_production →
  [a, b, c] = sampling_ratio.map (λ x => x * (total_production / sampling_ratio.sum)) →
  b = 2000 := by
  sorry

end NUMINAMATH_CALUDE_production_line_b_l481_48149


namespace NUMINAMATH_CALUDE_henrys_socks_l481_48173

theorem henrys_socks (a b c : ℕ) : 
  a + b + c = 15 →
  2 * a + 3 * b + 5 * c = 36 →
  a ≥ 1 →
  b ≥ 1 →
  c ≥ 1 →
  a = 11 :=
by sorry

end NUMINAMATH_CALUDE_henrys_socks_l481_48173


namespace NUMINAMATH_CALUDE_min_value_xy_expression_l481_48142

theorem min_value_xy_expression :
  (∀ x y : ℝ, (x*y - 2)^2 + (x - y)^2 ≥ 0) ∧
  (∃ x y : ℝ, (x*y - 2)^2 + (x - y)^2 = 0) :=
by sorry

end NUMINAMATH_CALUDE_min_value_xy_expression_l481_48142


namespace NUMINAMATH_CALUDE_no_three_squares_sum_2015_l481_48125

theorem no_three_squares_sum_2015 : ¬ ∃ (a b c : ℤ), a^2 + b^2 + c^2 = 2015 := by
  sorry

end NUMINAMATH_CALUDE_no_three_squares_sum_2015_l481_48125


namespace NUMINAMATH_CALUDE_triangle_angle_value_max_side_sum_l481_48115

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The given condition in the problem -/
def triangle_condition (t : Triangle) : Prop :=
  2 * t.a * Real.sin t.A = (2 * t.b + t.c) * Real.sin t.B + (2 * t.c + t.b) * Real.sin t.C

theorem triangle_angle_value (t : Triangle) (h : triangle_condition t) : t.A = 2 * Real.pi / 3 := by
  sorry

theorem max_side_sum (t : Triangle) (h1 : triangle_condition t) (h2 : t.a = 4) :
  ∃ (b c : ℝ), t.b = b ∧ t.c = c ∧ b + c ≤ 8 * Real.sqrt 3 / 3 ∧
  ∀ (b' c' : ℝ), t.b = b' ∧ t.c = c' → b' + c' ≤ 8 * Real.sqrt 3 / 3 := by
  sorry

end NUMINAMATH_CALUDE_triangle_angle_value_max_side_sum_l481_48115


namespace NUMINAMATH_CALUDE_football_team_size_l481_48123

theorem football_team_size :
  ∀ (P : ℕ),
  (49 : ℕ) ≤ P →  -- There are at least 49 throwers
  (63 : ℕ) ≤ P →  -- There are at least 63 right-handed players
  (P - 49) % 3 = 0 →  -- The non-throwers can be divided into thirds
  63 = 49 + (2 * (P - 49) / 3) →  -- Right-handed players equation
  P = 70 :=
by
  sorry

end NUMINAMATH_CALUDE_football_team_size_l481_48123


namespace NUMINAMATH_CALUDE_square_difference_of_integers_l481_48117

theorem square_difference_of_integers (a b : ℕ) (h1 : a + b = 60) (h2 : a - b = 20) :
  a^2 - b^2 = 1200 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_of_integers_l481_48117


namespace NUMINAMATH_CALUDE_hyperbola_equation_theorem_l481_48170

/-- A hyperbola with given asymptotes and a point it passes through -/
structure Hyperbola where
  /-- The slope of the asymptotes -/
  asymptote_slope : ℝ
  /-- A point that the hyperbola passes through -/
  point : ℝ × ℝ

/-- The equation of a hyperbola given its asymptotes and a point it passes through -/
def hyperbola_equation (h : Hyperbola) : ℝ → ℝ → Prop :=
  fun x y => x^2 - (y^2 / (4 * h.asymptote_slope^2)) = 1

/-- Theorem stating that a hyperbola with asymptotes y = ±2x passing through (√2, 2) has the equation x² - y²/4 = 1 -/
theorem hyperbola_equation_theorem (h : Hyperbola) 
    (h_asymptote : h.asymptote_slope = 2)
    (h_point : h.point = (Real.sqrt 2, 2)) :
    hyperbola_equation h = fun x y => x^2 - y^2/4 = 1 := by
  sorry

#check hyperbola_equation_theorem

end NUMINAMATH_CALUDE_hyperbola_equation_theorem_l481_48170


namespace NUMINAMATH_CALUDE_sin_double_angle_l481_48132

theorem sin_double_angle (α : ℝ) (h : Real.sin (α - π/4) = 3/5) : 
  Real.sin (2 * α) = 7/25 := by
sorry

end NUMINAMATH_CALUDE_sin_double_angle_l481_48132


namespace NUMINAMATH_CALUDE_cuboid_volume_transformation_l481_48101

theorem cuboid_volume_transformation (V : ℝ) (h : V = 343) : 
  let s := V^(1/3)
  let L := 3 * s
  let W := 1.5 * s
  let H := 2.5 * s
  L * W * H = 38587.5 := by
  sorry

end NUMINAMATH_CALUDE_cuboid_volume_transformation_l481_48101


namespace NUMINAMATH_CALUDE_fifteenth_triangular_number_l481_48148

def triangular_number (n : ℕ) : ℕ := n * (n + 1) / 2

theorem fifteenth_triangular_number : triangular_number 15 = 120 := by
  sorry

end NUMINAMATH_CALUDE_fifteenth_triangular_number_l481_48148


namespace NUMINAMATH_CALUDE_subtraction_problem_l481_48175

theorem subtraction_problem (n : ℝ) (h : n = 5) : ∃! x : ℝ, 7 * n - x = 2 * n + 10 ∧ x = 15 := by
  sorry

end NUMINAMATH_CALUDE_subtraction_problem_l481_48175


namespace NUMINAMATH_CALUDE_packing_peanuts_theorem_l481_48162

/-- Calculates the amount of packing peanuts needed for each small order -/
def packing_peanuts_per_small_order (total_peanuts : ℕ) (large_orders : ℕ) (small_orders : ℕ) (peanuts_per_large_order : ℕ) : ℕ :=
  (total_peanuts - large_orders * peanuts_per_large_order) / small_orders

/-- Theorem: Given the conditions, the amount of packing peanuts needed for each small order is 50g -/
theorem packing_peanuts_theorem :
  packing_peanuts_per_small_order 800 3 4 200 = 50 := by
  sorry

end NUMINAMATH_CALUDE_packing_peanuts_theorem_l481_48162


namespace NUMINAMATH_CALUDE_vasyas_birthday_l481_48178

-- Define the days of the week
inductive DayOfWeek
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday

def next_day (d : DayOfWeek) : DayOfWeek :=
  match d with
  | DayOfWeek.Sunday => DayOfWeek.Monday
  | DayOfWeek.Monday => DayOfWeek.Tuesday
  | DayOfWeek.Tuesday => DayOfWeek.Wednesday
  | DayOfWeek.Wednesday => DayOfWeek.Thursday
  | DayOfWeek.Thursday => DayOfWeek.Friday
  | DayOfWeek.Friday => DayOfWeek.Saturday
  | DayOfWeek.Saturday => DayOfWeek.Sunday

def two_days_after (d : DayOfWeek) : DayOfWeek :=
  next_day (next_day d)

theorem vasyas_birthday (birthday : DayOfWeek) 
  (h1 : next_day birthday ≠ DayOfWeek.Sunday)
  (h2 : two_days_after (next_day birthday) = DayOfWeek.Sunday) :
  birthday = DayOfWeek.Thursday := by
  sorry

end NUMINAMATH_CALUDE_vasyas_birthday_l481_48178


namespace NUMINAMATH_CALUDE_complex_magnitude_squared_l481_48112

theorem complex_magnitude_squared (z : ℂ) (h : z^2 + Complex.abs z ^ 2 = 2 - 3*I) : 
  Complex.abs z ^ 2 = 13/4 := by
sorry

end NUMINAMATH_CALUDE_complex_magnitude_squared_l481_48112


namespace NUMINAMATH_CALUDE_lcm_consecutive_sum_l481_48192

theorem lcm_consecutive_sum (a b c : ℕ) : 
  (a + 1 = b) → (b + 1 = c) → (Nat.lcm a (Nat.lcm b c) = 168) → (a + b + c = 21) := by
  sorry

end NUMINAMATH_CALUDE_lcm_consecutive_sum_l481_48192


namespace NUMINAMATH_CALUDE_shelter_cat_count_l481_48129

/-- Proves that the total number of cats and kittens in the shelter is 280 --/
theorem shelter_cat_count : ∀ (adult_cats female_cats litters kittens_per_litter : ℕ),
  adult_cats = 120 →
  female_cats = 2 * adult_cats / 3 →
  litters = 2 * female_cats / 5 →
  kittens_per_litter = 5 →
  adult_cats + litters * kittens_per_litter = 280 :=
by
  sorry

end NUMINAMATH_CALUDE_shelter_cat_count_l481_48129


namespace NUMINAMATH_CALUDE_sequence_property_l481_48111

/-- Represents the number of items in the nth row of the sequence -/
def num_items (n : ℕ) : ℕ := 2 * n - 1

/-- Represents the sum of items in the nth row of the sequence -/
def sum_items (n : ℕ) : ℕ := n * (2 * n - 1)

/-- The row number we're interested in -/
def target_row : ℕ := 1005

/-- The target value we're trying to match -/
def target_value : ℕ := 2009^2

theorem sequence_property :
  num_items target_row = 2009 ∧ sum_items target_row = target_value := by
  sorry

end NUMINAMATH_CALUDE_sequence_property_l481_48111


namespace NUMINAMATH_CALUDE_ratio_equality_l481_48137

theorem ratio_equality (x y z : ℝ) 
  (h_pos : x > 0 ∧ y > 0 ∧ z > 0) 
  (h_eq : y / (x + z) = (x - y) / z ∧ y / (x + z) = x / (y + 2*z)) : 
  x / y = 2 / 3 := by
sorry

end NUMINAMATH_CALUDE_ratio_equality_l481_48137


namespace NUMINAMATH_CALUDE_chord_length_range_l481_48136

/-- The chord length intercepted by the line y = x + t on the circle x + y² = 8 -/
def chordLength (t : ℝ) : ℝ := sorry

theorem chord_length_range (t : ℝ) :
  (∀ x y : ℝ, y = x + t ∧ x + y^2 = 8 → chordLength t ≥ 4 * Real.sqrt 2 / 3) →
  t ∈ Set.Icc (-(8 * Real.sqrt 2 / 3)) (8 * Real.sqrt 2 / 3) :=
sorry

end NUMINAMATH_CALUDE_chord_length_range_l481_48136


namespace NUMINAMATH_CALUDE_sandys_savings_difference_l481_48181

/-- Calculates the difference in savings between two years given the initial salary,
    savings percentages, and salary increase. -/
def savings_difference (initial_salary : ℝ) (savings_percent_year1 : ℝ) 
                       (savings_percent_year2 : ℝ) (salary_increase_percent : ℝ) : ℝ :=
  let salary_year2 := initial_salary * (1 + salary_increase_percent)
  let savings_year1 := initial_salary * savings_percent_year1
  let savings_year2 := salary_year2 * savings_percent_year2
  savings_year1 - savings_year2

/-- The difference in Sandy's savings between two years is $925.20 -/
theorem sandys_savings_difference :
  savings_difference 45000 0.083 0.056 0.115 = 925.20 := by
  sorry

end NUMINAMATH_CALUDE_sandys_savings_difference_l481_48181


namespace NUMINAMATH_CALUDE_rectangles_in_grid_l481_48179

/-- The number of different rectangles in a 3x5 grid -/
def num_rectangles : ℕ := 30

/-- The number of rows in the grid -/
def num_rows : ℕ := 3

/-- The number of columns in the grid -/
def num_cols : ℕ := 5

/-- Theorem stating that the number of rectangles in a 3x5 grid is 30 -/
theorem rectangles_in_grid :
  num_rectangles = (num_rows.choose 2) * (num_cols.choose 2) := by
  sorry

#eval num_rectangles -- This should output 30

end NUMINAMATH_CALUDE_rectangles_in_grid_l481_48179


namespace NUMINAMATH_CALUDE_gaeun_wins_l481_48146

/-- Conversion factor from meters to centimeters -/
def meters_to_cm : ℝ := 100

/-- Nana's flight distance in meters -/
def nana_distance_m : ℝ := 1.618

/-- Gaeun's flight distance in centimeters -/
def gaeun_distance_cm : ℝ := 162.3

/-- Theorem stating that Gaeun's flight distance is greater than Nana's by 0.5 cm -/
theorem gaeun_wins :
  gaeun_distance_cm - (nana_distance_m * meters_to_cm) = 0.5 := by
  sorry

end NUMINAMATH_CALUDE_gaeun_wins_l481_48146


namespace NUMINAMATH_CALUDE_cone_generatrix_property_cylinder_generatrix_parallel_l481_48182

-- Define the necessary geometric objects
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

structure Cone where
  vertex : Point3D
  base_center : Point3D
  base_radius : ℝ

structure Cylinder where
  base_center : Point3D
  height : ℝ
  radius : ℝ

-- Define what a generatrix is for a cone and a cylinder
def is_generatrix_of_cone (l : Set Point3D) (c : Cone) : Prop :=
  ∃ p : Point3D, p ∈ l ∧ 
    (p.x - c.base_center.x)^2 + (p.y - c.base_center.y)^2 = c.base_radius^2 ∧
    p.z = c.base_center.z ∧
    c.vertex ∈ l

def are_parallel (l1 l2 : Set Point3D) : Prop :=
  ∃ v : Point3D, ∀ p q : Point3D, p ∈ l1 ∧ q ∈ l2 → 
    ∃ t : ℝ, q.x - p.x = t * v.x ∧ q.y - p.y = t * v.y ∧ q.z - p.z = t * v.z

def is_generatrix_of_cylinder (l : Set Point3D) (c : Cylinder) : Prop :=
  ∃ p q : Point3D, p ∈ l ∧ q ∈ l ∧
    (p.x - c.base_center.x)^2 + (p.y - c.base_center.y)^2 = c.radius^2 ∧
    p.z = c.base_center.z ∧
    (q.x - c.base_center.x)^2 + (q.y - c.base_center.y)^2 = c.radius^2 ∧
    q.z = c.base_center.z + c.height

-- State the theorems to be proved
theorem cone_generatrix_property (c : Cone) (p : Point3D) :
  (p.x - c.base_center.x)^2 + (p.y - c.base_center.y)^2 = c.base_radius^2 ∧ 
  p.z = c.base_center.z →
  is_generatrix_of_cone {q : Point3D | ∃ t : ℝ, q = Point3D.mk 
    (c.vertex.x + t * (p.x - c.vertex.x))
    (c.vertex.y + t * (p.y - c.vertex.y))
    (c.vertex.z + t * (p.z - c.vertex.z))} c :=
sorry

theorem cylinder_generatrix_parallel (c : Cylinder) (l1 l2 : Set Point3D) :
  is_generatrix_of_cylinder l1 c ∧ is_generatrix_of_cylinder l2 c →
  are_parallel l1 l2 :=
sorry

end NUMINAMATH_CALUDE_cone_generatrix_property_cylinder_generatrix_parallel_l481_48182


namespace NUMINAMATH_CALUDE_rhombus_area_l481_48189

/-- The area of a rhombus with side length 4 cm and an interior angle of 30 degrees is 8 cm² -/
theorem rhombus_area (s : ℝ) (θ : ℝ) (h1 : s = 4) (h2 : θ = π / 6) :
  s * s * Real.sin θ = 8 :=
sorry

end NUMINAMATH_CALUDE_rhombus_area_l481_48189


namespace NUMINAMATH_CALUDE_minimum_draws_for_divisible_by_3_or_5_l481_48152

theorem minimum_draws_for_divisible_by_3_or_5 (n : ℕ) (hn : n = 90) :
  let divisible_by_3_or_5 (k : ℕ) := k % 3 = 0 ∨ k % 5 = 0
  let count_divisible := (Finset.range n).filter divisible_by_3_or_5 |>.card
  49 = n - count_divisible + 1 :=
by sorry

end NUMINAMATH_CALUDE_minimum_draws_for_divisible_by_3_or_5_l481_48152


namespace NUMINAMATH_CALUDE_complex_conjugate_roots_imply_zero_coefficients_l481_48156

theorem complex_conjugate_roots_imply_zero_coefficients 
  (c d : ℝ) 
  (h : ∃ (u v : ℝ), (Complex.I * v + u)^2 + (15 + Complex.I * c) * (Complex.I * v + u) + (35 + Complex.I * d) = 0 ∧ 
                     (Complex.I * -v + u)^2 + (15 + Complex.I * c) * (Complex.I * -v + u) + (35 + Complex.I * d) = 0) : 
  c = 0 ∧ d = 0 := by
sorry

end NUMINAMATH_CALUDE_complex_conjugate_roots_imply_zero_coefficients_l481_48156


namespace NUMINAMATH_CALUDE_fraction_zero_implies_x_one_third_l481_48107

theorem fraction_zero_implies_x_one_third (x : ℝ) :
  (3*x - 1) / (x^2 + 1) = 0 → x = 1/3 :=
by
  sorry

end NUMINAMATH_CALUDE_fraction_zero_implies_x_one_third_l481_48107


namespace NUMINAMATH_CALUDE_bridge_support_cans_l481_48155

/-- The weight of a full can of soda in ounces -/
def full_can_weight : ℕ := 12 + 2

/-- The weight of an empty can in ounces -/
def empty_can_weight : ℕ := 2

/-- The total weight the bridge must support in ounces -/
def total_bridge_weight : ℕ := 88

/-- The number of additional empty cans -/
def additional_empty_cans : ℕ := 2

/-- The number of full cans of soda the bridge needs to support -/
def num_full_cans : ℕ := (total_bridge_weight - additional_empty_cans * empty_can_weight) / full_can_weight

theorem bridge_support_cans : num_full_cans = 6 := by
  sorry

end NUMINAMATH_CALUDE_bridge_support_cans_l481_48155


namespace NUMINAMATH_CALUDE_product_in_base9_l481_48195

/-- Converts a base-9 number to its decimal (base-10) equivalent -/
def base9ToDecimal (n : ℕ) : ℕ := sorry

/-- Converts a decimal (base-10) number to its base-9 equivalent -/
def decimalToBase9 (n : ℕ) : ℕ := sorry

theorem product_in_base9 :
  decimalToBase9 (base9ToDecimal 327 * base9ToDecimal 6) = 2406 := by sorry

end NUMINAMATH_CALUDE_product_in_base9_l481_48195


namespace NUMINAMATH_CALUDE_bus_passengers_count_l481_48145

theorem bus_passengers_count :
  let men_count : ℕ := 18
  let women_count : ℕ := 26
  let children_count : ℕ := 10
  let total_passengers : ℕ := men_count + women_count + children_count
  total_passengers = 54 := by sorry

end NUMINAMATH_CALUDE_bus_passengers_count_l481_48145


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_l481_48183

theorem imaginary_part_of_z (i : ℂ) (h : i^2 = -1) : 
  (i^2 * (1 + i)).im = -1 := by sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_l481_48183


namespace NUMINAMATH_CALUDE_braden_winnings_l481_48140

/-- Calculates the total amount in Braden's money box after winning a bet -/
def total_amount_after_bet (initial_amount : ℕ) (bet_multiplier : ℕ) : ℕ :=
  initial_amount + bet_multiplier * initial_amount

/-- Theorem stating that given the initial conditions, Braden's final amount is $1200 -/
theorem braden_winnings :
  let initial_amount := 400
  let bet_multiplier := 2
  total_amount_after_bet initial_amount bet_multiplier = 1200 := by
  sorry

end NUMINAMATH_CALUDE_braden_winnings_l481_48140


namespace NUMINAMATH_CALUDE_fraction_value_l481_48191

theorem fraction_value : (3020 - 2931)^2 / 121 = 64 := by
  sorry

end NUMINAMATH_CALUDE_fraction_value_l481_48191


namespace NUMINAMATH_CALUDE_inequality_range_l481_48190

theorem inequality_range (a : ℝ) : 
  (∀ x : ℝ, (a - 2) * x^2 + 2 * (a - 2) * x - 4 < 0) → 
  -2 < a ∧ a ≤ 2 :=
by sorry

end NUMINAMATH_CALUDE_inequality_range_l481_48190


namespace NUMINAMATH_CALUDE_cos_two_theta_value_l481_48100

theorem cos_two_theta_value (θ : Real) 
  (h : Real.sin (θ / 2) + Real.cos (θ / 2) = 1 / 2) : 
  Real.cos (2 * θ) = -1 / 8 := by
  sorry

end NUMINAMATH_CALUDE_cos_two_theta_value_l481_48100


namespace NUMINAMATH_CALUDE_complex_fraction_value_l481_48158

theorem complex_fraction_value : Complex.I / (1 - Complex.I)^2 = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_value_l481_48158


namespace NUMINAMATH_CALUDE_cloth_cost_price_l481_48126

/-- Given a trader sells cloth with the following conditions:
  * Sells 45 meters of cloth
  * Total selling price is 4500 Rs
  * Profit per meter is 14 Rs
  Prove that the cost price of one meter of cloth is 86 Rs -/
theorem cloth_cost_price 
  (total_meters : ℕ) 
  (selling_price : ℕ) 
  (profit_per_meter : ℕ) 
  (h1 : total_meters = 45)
  (h2 : selling_price = 4500)
  (h3 : profit_per_meter = 14) :
  (selling_price - total_meters * profit_per_meter) / total_meters = 86 := by
sorry

end NUMINAMATH_CALUDE_cloth_cost_price_l481_48126


namespace NUMINAMATH_CALUDE_principal_amount_l481_48188

/-- Proves that given the conditions of the problem, the principal amount is 3000 --/
theorem principal_amount (P : ℝ) : 
  (P * 4 * 5) / 100 = P - 2400 → P = 3000 := by
  sorry

end NUMINAMATH_CALUDE_principal_amount_l481_48188


namespace NUMINAMATH_CALUDE_fifth_month_sale_l481_48184

/-- Proves that the sale in the 5th month is 6029, given the conditions of the problem -/
theorem fifth_month_sale (
  average_sale : ℕ)
  (first_month_sale : ℕ)
  (second_month_sale : ℕ)
  (third_month_sale : ℕ)
  (fourth_month_sale : ℕ)
  (sixth_month_sale : ℕ)
  (h1 : average_sale = 5600)
  (h2 : first_month_sale = 5266)
  (h3 : second_month_sale = 5768)
  (h4 : third_month_sale = 5922)
  (h5 : fourth_month_sale = 5678)
  (h6 : sixth_month_sale = 4937) :
  first_month_sale + second_month_sale + third_month_sale + fourth_month_sale + 6029 + sixth_month_sale = 6 * average_sale :=
by sorry

#eval 5266 + 5768 + 5922 + 5678 + 6029 + 4937
#eval 6 * 5600

end NUMINAMATH_CALUDE_fifth_month_sale_l481_48184


namespace NUMINAMATH_CALUDE_complex_product_sum_l481_48165

theorem complex_product_sum (a b : ℝ) : (1 + Complex.I) * (1 - Complex.I) = a + b * Complex.I → a + b = 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_product_sum_l481_48165
