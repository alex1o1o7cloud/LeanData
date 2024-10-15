import Mathlib

namespace NUMINAMATH_CALUDE_quadratic_completing_square_l734_73483

theorem quadratic_completing_square : ∀ x : ℝ, x^2 - 8*x + 6 = 0 ↔ (x - 4)^2 = 10 := by sorry

end NUMINAMATH_CALUDE_quadratic_completing_square_l734_73483


namespace NUMINAMATH_CALUDE_birdseed_theorem_l734_73459

/-- Calculates the amount of birdseed Peter needs to buy for a week -/
def birdseed_for_week : ℕ :=
  let parakeet_daily_consumption : ℕ := 2
  let parrot_daily_consumption : ℕ := 14
  let finch_daily_consumption : ℕ := parakeet_daily_consumption / 2
  let num_parakeets : ℕ := 3
  let num_parrots : ℕ := 2
  let num_finches : ℕ := 4
  let days_in_week : ℕ := 7
  
  let total_daily_consumption : ℕ := 
    num_parakeets * parakeet_daily_consumption +
    num_parrots * parrot_daily_consumption +
    num_finches * finch_daily_consumption

  total_daily_consumption * days_in_week

/-- Theorem stating that the amount of birdseed Peter needs to buy for a week is 266 grams -/
theorem birdseed_theorem : birdseed_for_week = 266 := by
  sorry

end NUMINAMATH_CALUDE_birdseed_theorem_l734_73459


namespace NUMINAMATH_CALUDE_find_number_l734_73467

theorem find_number (A B : ℕ+) : 
  Nat.gcd A B = 14 →
  Nat.lcm A B = 312 →
  B = 182 →
  A = 24 := by
sorry

end NUMINAMATH_CALUDE_find_number_l734_73467


namespace NUMINAMATH_CALUDE_sequence_problem_solution_l734_73437

def sequence_problem (a : ℕ → ℝ) : Prop :=
  (∀ n : ℕ, a n * a (n + 1) = 1 - a (n + 1)) ∧ 
  (a 2010 = 2) ∧
  (a 2008 = -3)

theorem sequence_problem_solution :
  ∃ a : ℕ → ℝ, sequence_problem a :=
sorry

end NUMINAMATH_CALUDE_sequence_problem_solution_l734_73437


namespace NUMINAMATH_CALUDE_zoo_animal_count_l734_73465

/-- The number of animals Brinley counted at the San Diego Zoo --/
theorem zoo_animal_count :
  let snakes : ℕ := 100
  let arctic_foxes : ℕ := 80
  let leopards : ℕ := 20
  let bee_eaters : ℕ := 10 * leopards
  let cheetahs : ℕ := snakes / 2
  let alligators : ℕ := 2 * (arctic_foxes + leopards)
  snakes + arctic_foxes + leopards + bee_eaters + cheetahs + alligators = 650 :=
by sorry


end NUMINAMATH_CALUDE_zoo_animal_count_l734_73465


namespace NUMINAMATH_CALUDE_smallest_with_digit_sum_41_plus_2021_l734_73492

def digit_sum (n : ℕ) : ℕ := sorry

def is_smallest_with_digit_sum (N : ℕ) (sum : ℕ) : Prop :=
  digit_sum N = sum ∧ ∀ m < N, digit_sum m ≠ sum

theorem smallest_with_digit_sum_41_plus_2021 :
  ∃ N : ℕ, is_smallest_with_digit_sum N 41 ∧ digit_sum (N + 2021) = 10 := by
  sorry

end NUMINAMATH_CALUDE_smallest_with_digit_sum_41_plus_2021_l734_73492


namespace NUMINAMATH_CALUDE_sqrt_product_plus_one_equals_271_l734_73468

theorem sqrt_product_plus_one_equals_271 : 
  Real.sqrt ((18 : ℝ) * 17 * 16 * 15 + 1) = 271 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_product_plus_one_equals_271_l734_73468


namespace NUMINAMATH_CALUDE_intersection_forms_ellipse_l734_73461

theorem intersection_forms_ellipse (a b : ℝ) (hab : a * b ≠ 0) :
  ∃ (h k r s : ℝ), ∀ (x y : ℝ),
    (a * x - y + b = 0) ∧ (b * x^2 + a * y^2 = a * b) →
    ((x - h) / r)^2 + ((y - k) / s)^2 = 1 :=
by sorry

end NUMINAMATH_CALUDE_intersection_forms_ellipse_l734_73461


namespace NUMINAMATH_CALUDE_circle_C_equation_l734_73412

/-- A circle C with the following properties:
  - The center is on the positive x-axis
  - The radius is √2
  - The circle is tangent to the line x + y = 0
-/
structure CircleC where
  center : ℝ × ℝ
  radius : ℝ
  center_on_x_axis : center.2 = 0
  center_positive : center.1 > 0
  radius_is_sqrt2 : radius = Real.sqrt 2
  tangent_to_line : ∃ (p : ℝ × ℝ), p.1 + p.2 = 0 ∧ 
    (center.1 - p.1)^2 + (center.2 - p.2)^2 = radius^2

/-- The standard equation of circle C is (x-2)² + y² = 2 -/
theorem circle_C_equation (c : CircleC) : 
  ∀ (x y : ℝ), (x - 2)^2 + y^2 = 2 ↔ (x - c.center.1)^2 + (y - c.center.2)^2 = c.radius^2 :=
by sorry

end NUMINAMATH_CALUDE_circle_C_equation_l734_73412


namespace NUMINAMATH_CALUDE_tan_105_degrees_l734_73442

theorem tan_105_degrees :
  Real.tan (105 * π / 180) = -2 - Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_tan_105_degrees_l734_73442


namespace NUMINAMATH_CALUDE_student_count_l734_73404

theorem student_count (n : ℕ) (ella : ℕ) : 
  (ella = 60) → -- Ella's position from best
  (n + 1 - ella = 60) → -- Ella's position from worst (n is total students minus 1)
  (n + 1 = 119) := by
sorry

end NUMINAMATH_CALUDE_student_count_l734_73404


namespace NUMINAMATH_CALUDE_lincoln_county_houses_l734_73422

/-- The number of houses in Lincoln County after a housing boom -/
def houses_after_boom (original : ℕ) (new_built : ℕ) : ℕ :=
  original + new_built

/-- Theorem stating the total number of houses after the housing boom -/
theorem lincoln_county_houses :
  houses_after_boom 20817 97741 = 118558 := by
  sorry

end NUMINAMATH_CALUDE_lincoln_county_houses_l734_73422


namespace NUMINAMATH_CALUDE_solution_set_of_inequality_l734_73484

theorem solution_set_of_inequality (x : ℝ) :
  (((3 * x + 1) / (1 - 2 * x) ≥ 0) ↔ (-1/3 ≤ x ∧ x < 1/2)) :=
by sorry

end NUMINAMATH_CALUDE_solution_set_of_inequality_l734_73484


namespace NUMINAMATH_CALUDE_river_length_l734_73462

/-- The length of a river given Karen's paddling speed, river current speed, and time taken to paddle up the river -/
theorem river_length
  (karen_speed : ℝ)  -- Karen's paddling speed on still water
  (river_speed : ℝ)  -- River's current speed
  (time_taken : ℝ)   -- Time taken to paddle up the river
  (h1 : karen_speed = 10)  -- Karen's speed is 10 miles per hour
  (h2 : river_speed = 4)   -- River flows at 4 miles per hour
  (h3 : time_taken = 2)    -- It takes 2 hours to paddle up the river
  : (karen_speed - river_speed) * time_taken = 12 :=
by sorry

end NUMINAMATH_CALUDE_river_length_l734_73462


namespace NUMINAMATH_CALUDE_complex_number_fourth_quadrant_l734_73474

theorem complex_number_fourth_quadrant (z : ℂ) : 
  (z.re > 0) →  -- z is in the fourth quadrant (real part positive)
  (z.im < 0) →  -- z is in the fourth quadrant (imaginary part negative)
  (z.re + z.im = 7) →  -- sum of real and imaginary parts is 7
  (Complex.abs z = 13) →  -- magnitude of z is 13
  z = Complex.mk 12 (-5) :=  -- z equals 12 - 5i
by sorry

end NUMINAMATH_CALUDE_complex_number_fourth_quadrant_l734_73474


namespace NUMINAMATH_CALUDE_fraction_equivalence_l734_73487

theorem fraction_equivalence (a b : ℚ) : 
  (a ≠ 0) → (b ≠ 0) → ((1 / (a / b)) * (5 / 6) = 1 / (5 / 2)) → (a / b = 25 / 12) := by
  sorry

end NUMINAMATH_CALUDE_fraction_equivalence_l734_73487


namespace NUMINAMATH_CALUDE_pentagonal_prism_faces_l734_73440

/-- A polyhedron with pentagonal bases and lateral faces -/
structure PentagonalPrism where
  base_edges : ℕ
  base_count : ℕ
  lateral_faces : ℕ

/-- The total number of faces in a pentagonal prism -/
def total_faces (p : PentagonalPrism) : ℕ :=
  p.base_count + p.lateral_faces

/-- Theorem: A pentagonal prism has 7 faces in total -/
theorem pentagonal_prism_faces :
  ∀ (p : PentagonalPrism), 
    p.base_edges = 5 → 
    p.base_count = 2 → 
    p.lateral_faces = 5 → 
    total_faces p = 7 := by
  sorry

#check pentagonal_prism_faces

end NUMINAMATH_CALUDE_pentagonal_prism_faces_l734_73440


namespace NUMINAMATH_CALUDE_product_of_binary_and_ternary_l734_73414

/-- Converts a list of digits in a given base to its decimal representation -/
def to_decimal (digits : List Nat) (base : Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * base ^ i) 0

/-- The problem statement -/
theorem product_of_binary_and_ternary : 
  let binary := [1, 0, 1, 1]  -- 1101 in binary, least significant digit first
  let ternary := [2, 0, 2]    -- 202 in ternary, least significant digit first
  (to_decimal binary 2) * (to_decimal ternary 3) = 260 := by
  sorry

end NUMINAMATH_CALUDE_product_of_binary_and_ternary_l734_73414


namespace NUMINAMATH_CALUDE_solution_set_when_a_is_2_range_of_a_l734_73458

-- Define the function f
def f (x a : ℝ) : ℝ := |x - a^2| + |x - 2*a + 1|

-- Part 1: Solution set when a = 2
theorem solution_set_when_a_is_2 :
  {x : ℝ | f x 2 ≥ 4} = {x : ℝ | x ≤ 3/2 ∨ x ≥ 11/2} :=
sorry

-- Part 2: Range of values for a
theorem range_of_a (h : ∀ x : ℝ, f x a ≥ 4) :
  a ≤ -1 ∨ a ≥ 3 :=
sorry

end NUMINAMATH_CALUDE_solution_set_when_a_is_2_range_of_a_l734_73458


namespace NUMINAMATH_CALUDE_quadratic_roots_problem_l734_73420

theorem quadratic_roots_problem (a b m p r : ℝ) : 
  (a^2 - m*a + 3 = 0) →
  (b^2 - m*b + 3 = 0) →
  ((a + 2/b)^2 - p*(a + 2/b) + r = 0) →
  ((b + 2/a)^2 - p*(b + 2/a) + r = 0) →
  r = 25/3 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_problem_l734_73420


namespace NUMINAMATH_CALUDE_matching_pair_probability_is_0_5226_l734_73424

/-- Represents the types of shoes in the warehouse -/
inductive ShoeType
  | Sneaker
  | Boot
  | DressShoe

/-- Represents the shoe warehouse inventory -/
structure ShoeWarehouse where
  sneakers : ℕ
  boots : ℕ
  dressShoes : ℕ
  sneakerProb : ℝ
  bootProb : ℝ
  dressShoeProb : ℝ

/-- Calculates the probability of selecting a matching pair of shoes -/
def matchingPairProbability (warehouse : ShoeWarehouse) : ℝ :=
  let sneakerProb := warehouse.sneakers * warehouse.sneakerProb * (warehouse.sneakers - 1) * warehouse.sneakerProb
  let bootProb := warehouse.boots * warehouse.bootProb * (warehouse.boots - 1) * warehouse.bootProb
  let dressShoeProb := warehouse.dressShoes * warehouse.dressShoeProb * (warehouse.dressShoes - 1) * warehouse.dressShoeProb
  sneakerProb + bootProb + dressShoeProb

/-- Theorem stating the probability of selecting a matching pair of shoes -/
theorem matching_pair_probability_is_0_5226 :
  let warehouse : ShoeWarehouse := {
    sneakers := 12,
    boots := 15,
    dressShoes := 18,
    sneakerProb := 0.04,
    bootProb := 0.03,
    dressShoeProb := 0.02
  }
  matchingPairProbability warehouse = 0.5226 := by sorry

end NUMINAMATH_CALUDE_matching_pair_probability_is_0_5226_l734_73424


namespace NUMINAMATH_CALUDE_range_of_a_l734_73456

theorem range_of_a (a : ℝ) : 2 * a ≠ a^2 ↔ a ≠ 0 ∧ a ≠ 2 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l734_73456


namespace NUMINAMATH_CALUDE_expression_simplification_and_evaluation_l734_73409

theorem expression_simplification_and_evaluation (a : ℕ) 
  (h1 : 2 * a + 1 < 3 * a + 3) 
  (h2 : 2 / 3 * (a - 1) ≤ 1 / 2 * (a + 1 / 3)) 
  (h3 : a ≠ 0) 
  (h4 : a ≠ 1) 
  (h5 : a ≠ 2) : 
  ∃ (result : ℕ), 
    ((a + 1 - (4 * a - 5) / (a - 1)) / (1 / a - 1 / (a^2 - a)) = a * (a - 2)) ∧ 
    (result = a * (a - 2)) ∧ 
    (result = 3 ∨ result = 8 ∨ result = 15) :=
sorry

end NUMINAMATH_CALUDE_expression_simplification_and_evaluation_l734_73409


namespace NUMINAMATH_CALUDE_fraction_calculation_l734_73480

theorem fraction_calculation : 
  (((1 : ℚ) / 2 + (1 : ℚ) / 5) / ((3 : ℚ) / 7 - (1 : ℚ) / 14)) * (2 : ℚ) / 3 = 98 / 75 := by
  sorry

end NUMINAMATH_CALUDE_fraction_calculation_l734_73480


namespace NUMINAMATH_CALUDE_problem_solution_l734_73496

theorem problem_solution (r s : ℝ) 
  (h1 : 1 < r) 
  (h2 : r < s) 
  (h3 : 1 / r + 1 / s = 1) 
  (h4 : r * s = 15 / 4) : 
  s = (15 + Real.sqrt 15) / 8 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l734_73496


namespace NUMINAMATH_CALUDE_failed_implies_no_perfect_essay_l734_73457

-- Define the universe of students
variable (Student : Type)

-- Define predicates
variable (wrote_perfect_essay : Student → Prop)
variable (passed_course : Student → Prop)

-- Define the given condition
axiom perfect_essay_implies_pass :
  ∀ (s : Student), wrote_perfect_essay s → passed_course s

-- The statement to prove
theorem failed_implies_no_perfect_essay :
  ∀ (s : Student), ¬(passed_course s) → ¬(wrote_perfect_essay s) :=
sorry

end NUMINAMATH_CALUDE_failed_implies_no_perfect_essay_l734_73457


namespace NUMINAMATH_CALUDE_inverse_81_mod_103_l734_73406

theorem inverse_81_mod_103 (h : (9⁻¹ : ZMod 103) = 65) : (81⁻¹ : ZMod 103) = 2 := by
  sorry

end NUMINAMATH_CALUDE_inverse_81_mod_103_l734_73406


namespace NUMINAMATH_CALUDE_dinner_price_problem_l734_73415

theorem dinner_price_problem (original_price : ℝ) : 
  -- John's payment (after discount and tip)
  (0.90 * original_price + 0.15 * original_price) -
  -- Jane's payment (after discount and tip)
  (0.90 * original_price + 0.15 * (0.90 * original_price)) = 0.54 →
  original_price = 36 := by
sorry

end NUMINAMATH_CALUDE_dinner_price_problem_l734_73415


namespace NUMINAMATH_CALUDE_removed_ball_number_l734_73452

theorem removed_ball_number (n : ℕ) (h1 : n > 0) :
  (n * (n + 1)) / 2 - 5048 = 2 :=
by
  sorry

end NUMINAMATH_CALUDE_removed_ball_number_l734_73452


namespace NUMINAMATH_CALUDE_prob_non_blue_specific_cube_l734_73451

/-- A cube with colored faces -/
structure ColoredCube where
  green_faces : ℕ
  yellow_faces : ℕ
  blue_faces : ℕ

/-- The probability of rolling a non-blue face on a colored cube -/
def prob_non_blue (cube : ColoredCube) : ℚ :=
  (cube.green_faces + cube.yellow_faces) / (cube.green_faces + cube.yellow_faces + cube.blue_faces)

/-- Theorem: The probability of rolling a non-blue face on a cube with 3 green faces, 2 yellow faces, and 1 blue face is 5/6 -/
theorem prob_non_blue_specific_cube :
  prob_non_blue ⟨3, 2, 1⟩ = 5/6 := by sorry

end NUMINAMATH_CALUDE_prob_non_blue_specific_cube_l734_73451


namespace NUMINAMATH_CALUDE_factorization_problem_l734_73497

theorem factorization_problem (a m n b : ℝ) : 
  (∀ x, x^2 + a*x + m = (x + 2) * (x + 4)) →
  (∀ x, x^2 + n*x + b = (x + 1) * (x + 9)) →
  (∀ x, x^2 + a*x + b = (x + 3)^2) :=
by sorry

end NUMINAMATH_CALUDE_factorization_problem_l734_73497


namespace NUMINAMATH_CALUDE_hexagon_centers_square_area_ratio_l734_73425

/-- Square represents a square in 2D space -/
structure Square where
  side : ℝ
  center : ℝ × ℝ

/-- RegularHexagon represents a regular hexagon in 2D space -/
structure RegularHexagon where
  side : ℝ
  center : ℝ × ℝ

/-- Configuration represents the problem setup -/
structure Configuration where
  square : Square
  hexagons : Fin 4 → RegularHexagon

/-- Defines the specific configuration described in the problem -/
def problem_configuration : Configuration :=
  sorry

/-- Calculate the area of a square given its side length -/
def square_area (s : Square) : ℝ :=
  s.side * s.side

/-- Calculate the area of the square formed by the centers of the hexagons -/
def hexagon_centers_square_area (c : Configuration) : ℝ :=
  sorry

/-- The main theorem to be proved -/
theorem hexagon_centers_square_area_ratio (c : Configuration) :
  hexagon_centers_square_area c / square_area c.square = 4.5 := by
  sorry

end NUMINAMATH_CALUDE_hexagon_centers_square_area_ratio_l734_73425


namespace NUMINAMATH_CALUDE_convex_lattice_nonagon_centroid_l734_73488

/-- A lattice point in 2D space -/
structure LatticePoint where
  x : Int
  y : Int

/-- A convex nonagon represented by 9 lattice points -/
structure ConvexLatticeNonagon where
  vertices : Fin 9 → LatticePoint
  is_convex : Bool  -- We assume this property without defining it explicitly

/-- The centroid of three points -/
def centroid (p1 p2 p3 : LatticePoint) : (Rat × Rat) :=
  ((p1.x + p2.x + p3.x) / 3, (p1.y + p2.y + p3.y) / 3)

/-- Check if a point with rational coordinates is a lattice point -/
def isLatticePoint (p : Rat × Rat) : Prop :=
  ∃ (x y : Int), p.1 = x ∧ p.2 = y

/-- Main theorem: Any convex lattice nonagon has three vertices whose centroid is a lattice point -/
theorem convex_lattice_nonagon_centroid (n : ConvexLatticeNonagon) :
  ∃ (i j k : Fin 9), i ≠ j ∧ j ≠ k ∧ i ≠ k ∧
    isLatticePoint (centroid (n.vertices i) (n.vertices j) (n.vertices k)) :=
  sorry

end NUMINAMATH_CALUDE_convex_lattice_nonagon_centroid_l734_73488


namespace NUMINAMATH_CALUDE_isosceles_trapezoid_angles_l734_73427

/-- An isosceles trapezoid with inscribed and circumscribed circles -/
structure IsoscelesTrapezoid where
  height : ℝ
  circum_radius : ℝ
  height_ratio : height / circum_radius = Real.sqrt (2/3)

/-- The angles of an isosceles trapezoid -/
def trapezoid_angles (t : IsoscelesTrapezoid) : ℝ × ℝ := sorry

theorem isosceles_trapezoid_angles (t : IsoscelesTrapezoid) :
  trapezoid_angles t = (45, 135) := by sorry

end NUMINAMATH_CALUDE_isosceles_trapezoid_angles_l734_73427


namespace NUMINAMATH_CALUDE_kelly_games_theorem_l734_73402

/-- The number of games Kelly gives away -/
def games_given_away : ℕ := 15

/-- The number of games Kelly has left after giving some away -/
def games_left : ℕ := 35

/-- The initial number of games Kelly has -/
def initial_games : ℕ := games_given_away + games_left

theorem kelly_games_theorem : initial_games = 50 := by
  sorry

end NUMINAMATH_CALUDE_kelly_games_theorem_l734_73402


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_is_zero_l734_73430

theorem imaginary_part_of_z_is_zero (z : ℂ) (h : z / (1 + 2 * I) = 1 - 2 * I) : 
  z.im = 0 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_is_zero_l734_73430


namespace NUMINAMATH_CALUDE_fractional_equation_solution_l734_73449

theorem fractional_equation_solution : 
  ∃ x : ℝ, (2 / (x - 1) = 1 / x) ∧ (x = -1) := by
  sorry

end NUMINAMATH_CALUDE_fractional_equation_solution_l734_73449


namespace NUMINAMATH_CALUDE_billy_restaurant_bill_l734_73411

/-- The total bill at Billy's Restaurant for a group with given characteristics -/
def total_bill (num_adults num_children : ℕ) (adult_meal_cost child_meal_cost : ℚ) 
  (num_fries_baskets : ℕ) (fries_basket_cost : ℚ) (drink_cost : ℚ) : ℚ :=
  num_adults * adult_meal_cost + 
  num_children * child_meal_cost + 
  num_fries_baskets * fries_basket_cost + 
  drink_cost

/-- Theorem stating that the total bill for the given group is $89 -/
theorem billy_restaurant_bill : 
  total_bill 4 3 12 7 2 5 10 = 89 := by
  sorry

end NUMINAMATH_CALUDE_billy_restaurant_bill_l734_73411


namespace NUMINAMATH_CALUDE_distance_to_school_l734_73444

/-- The distance to school given travel conditions -/
theorem distance_to_school : 
  ∀ (total_time speed_to speed_from : ℝ),
  total_time = 1 →
  speed_to = 5 →
  speed_from = 25 →
  ∃ (distance : ℝ),
    distance / speed_to + distance / speed_from = total_time ∧
    distance = 25 / 6 :=
by sorry

end NUMINAMATH_CALUDE_distance_to_school_l734_73444


namespace NUMINAMATH_CALUDE_jewelry_ensemble_orders_l734_73431

theorem jewelry_ensemble_orders (necklace_price bracelet_price earring_price ensemble_price : ℚ)
  (necklaces_sold bracelets_sold earrings_sold : ℕ)
  (total_amount : ℚ)
  (h1 : necklace_price = 25)
  (h2 : bracelet_price = 15)
  (h3 : earring_price = 10)
  (h4 : ensemble_price = 45)
  (h5 : necklaces_sold = 5)
  (h6 : bracelets_sold = 10)
  (h7 : earrings_sold = 20)
  (h8 : total_amount = 565) :
  (total_amount - (necklace_price * necklaces_sold + bracelet_price * bracelets_sold + earring_price * earrings_sold)) / ensemble_price = 2 := by
  sorry

end NUMINAMATH_CALUDE_jewelry_ensemble_orders_l734_73431


namespace NUMINAMATH_CALUDE_intersecting_subset_exists_l734_73478

theorem intersecting_subset_exists (X : Finset ℕ) (A : Fin 100 → Finset ℕ) 
  (h_size : X.card ≥ 4) 
  (h_subsets : ∀ i, A i ⊆ X) 
  (h_large : ∀ i, (A i).card > 3/4 * X.card) :
  ∃ Y : Finset ℕ, Y ⊆ X ∧ Y.card ≤ 4 ∧ ∀ i, (Y ∩ A i).Nonempty := by
  sorry


end NUMINAMATH_CALUDE_intersecting_subset_exists_l734_73478


namespace NUMINAMATH_CALUDE_jamie_lost_balls_jamie_lost_six_balls_l734_73490

theorem jamie_lost_balls (initial_red : ℕ) (blue_multiplier : ℕ) (yellow_bought : ℕ) (final_total : ℕ) : ℕ :=
  let initial_blue := blue_multiplier * initial_red
  let initial_total := initial_red + initial_blue + yellow_bought
  initial_total - final_total

theorem jamie_lost_six_balls : jamie_lost_balls 16 2 32 74 = 6 := by
  sorry

end NUMINAMATH_CALUDE_jamie_lost_balls_jamie_lost_six_balls_l734_73490


namespace NUMINAMATH_CALUDE_smallest_nonnegative_value_l734_73473

theorem smallest_nonnegative_value (p : ℕ) (hp : Nat.Prime p) (hp_odd : p % 2 = 1) :
  ∃ (x y : ℕ), 
    x > 0 ∧ y > 0 ∧ x ≤ y ∧
    Real.sqrt (2 * p : ℝ) - Real.sqrt x - Real.sqrt y ≥ 0 ∧
    ∀ (a b : ℕ), a > 0 → b > 0 → a ≤ b →
      Real.sqrt (2 * p : ℝ) - Real.sqrt a - Real.sqrt b ≥ 0 →
      Real.sqrt (2 * p : ℝ) - Real.sqrt a - Real.sqrt b ≥ 
      Real.sqrt (2 * p : ℝ) - Real.sqrt x - Real.sqrt y ∧
    x = (p - 1) / 2 ∧ y = (p + 1) / 2 :=
by sorry

end NUMINAMATH_CALUDE_smallest_nonnegative_value_l734_73473


namespace NUMINAMATH_CALUDE_stack_height_is_3_meters_l734_73417

/-- The number of packages in a stack -/
def packages_per_stack : ℕ := 60

/-- The number of sheets in a package -/
def sheets_per_package : ℕ := 500

/-- The thickness of a single sheet in millimeters -/
def sheet_thickness : ℚ := 1/10

/-- The height of a stack in meters -/
def stack_height : ℚ := 3

/-- Theorem stating that the height of a stack of packages is 3 meters -/
theorem stack_height_is_3_meters :
  (packages_per_stack : ℚ) * sheets_per_package * sheet_thickness / 1000 = stack_height :=
by sorry

end NUMINAMATH_CALUDE_stack_height_is_3_meters_l734_73417


namespace NUMINAMATH_CALUDE_max_notebooks_proof_l734_73450

/-- The maximum number of notebooks that can be bought given the constraints -/
def max_notebooks : ℕ := 5

/-- The total budget in yuan -/
def total_budget : ℚ := 30

/-- The total number of books -/
def total_books : ℕ := 30

/-- The cost of each notebook in yuan -/
def notebook_cost : ℚ := 4

/-- The cost of each exercise book in yuan -/
def exercise_book_cost : ℚ := 0.4

theorem max_notebooks_proof :
  (∀ n : ℕ, n ≤ total_books →
    n * notebook_cost + (total_books - n) * exercise_book_cost ≤ total_budget) →
  (max_notebooks * notebook_cost + (total_books - max_notebooks) * exercise_book_cost ≤ total_budget) ∧
  (∀ m : ℕ, m > max_notebooks →
    m * notebook_cost + (total_books - m) * exercise_book_cost > total_budget) :=
by sorry

end NUMINAMATH_CALUDE_max_notebooks_proof_l734_73450


namespace NUMINAMATH_CALUDE_unique_solution_equation_l734_73408

theorem unique_solution_equation : ∃! x : ℝ, (28 + 48 / x) * x = 1980 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_equation_l734_73408


namespace NUMINAMATH_CALUDE_triangle_ABC_properties_l734_73498

/-- Given a triangle ABC with vertices A(4, 4), B(-4, 2), and C(2, 0) -/
def triangle_ABC : Set (ℝ × ℝ) := {(4, 4), (-4, 2), (2, 0)}

/-- The equation of a line ax + by + c = 0 is represented by the triple (a, b, c) -/
def Line := ℝ × ℝ × ℝ

/-- The median CD of triangle ABC -/
def median_CD : Line := sorry

/-- The altitude from C to AB -/
def altitude_C : Line := sorry

/-- The centroid G of triangle ABC -/
def centroid_G : ℝ × ℝ := sorry

theorem triangle_ABC_properties :
  (median_CD = (3, 2, -6)) ∧
  (altitude_C = (4, 1, -8)) ∧
  (centroid_G = (2/3, 2)) := by sorry

end NUMINAMATH_CALUDE_triangle_ABC_properties_l734_73498


namespace NUMINAMATH_CALUDE_symmetric_points_sum_power_l734_73441

theorem symmetric_points_sum_power (m n : ℤ) : 
  (2*n - m = -14) → (m = 4) → (m + n)^2023 = -1 := by
  sorry

end NUMINAMATH_CALUDE_symmetric_points_sum_power_l734_73441


namespace NUMINAMATH_CALUDE_marta_book_count_l734_73499

/-- The number of books on Marta's shelf after all changes -/
def final_book_count (initial_books added_books removed_books birthday_multiplier : ℕ) : ℕ :=
  initial_books + added_books - removed_books + birthday_multiplier * initial_books

/-- Theorem stating the final number of books on Marta's shelf -/
theorem marta_book_count : final_book_count 38 10 5 3 = 157 := by
  sorry

#eval final_book_count 38 10 5 3

end NUMINAMATH_CALUDE_marta_book_count_l734_73499


namespace NUMINAMATH_CALUDE_shortest_distance_is_one_l734_73453

/-- Curve C₁ parameterized by θ -/
noncomputable def C₁ (θ : ℝ) : ℝ × ℝ × ℝ := sorry

/-- Curve C₂ parameterized by t -/
noncomputable def C₂ (t : ℝ) : ℝ × ℝ × ℝ := sorry

/-- Distance function between points on C₁ and C₂ -/
noncomputable def D (θ t : ℝ) : ℝ :=
  let (x₁, y₁, z₁) := C₁ θ
  let (x₂, y₂, z₂) := C₂ t
  Real.sqrt ((x₁ - x₂)^2 + (y₁ - y₂)^2 + (z₁ - z₂)^2)

/-- The shortest distance between C₁ and C₂ is 1 -/
theorem shortest_distance_is_one : ∃ θ₀ t₀, ∀ θ t, D θ₀ t₀ ≤ D θ t ∧ D θ₀ t₀ = 1 := by
  sorry

end NUMINAMATH_CALUDE_shortest_distance_is_one_l734_73453


namespace NUMINAMATH_CALUDE_cube_surface_area_l734_73419

theorem cube_surface_area (volume : ℝ) (side : ℝ) (surface_area : ℝ) : 
  volume = 729 → 
  volume = side^3 → 
  surface_area = 6 * side^2 → 
  surface_area = 486 :=
by sorry

end NUMINAMATH_CALUDE_cube_surface_area_l734_73419


namespace NUMINAMATH_CALUDE_camel_cost_l734_73410

/-- The cost of animals in rupees -/
structure AnimalCosts where
  camel : ℚ
  horse : ℚ
  ox : ℚ
  elephant : ℚ

/-- The conditions given in the problem -/
def problem_conditions (costs : AnimalCosts) : Prop :=
  10 * costs.camel = 24 * costs.horse ∧
  16 * costs.horse = 4 * costs.ox ∧
  6 * costs.ox = 4 * costs.elephant ∧
  10 * costs.elephant = 130000

/-- The theorem stating that given the problem conditions, the cost of a camel is 5200 rupees -/
theorem camel_cost (costs : AnimalCosts) :
  problem_conditions costs → costs.camel = 5200 := by
  sorry

end NUMINAMATH_CALUDE_camel_cost_l734_73410


namespace NUMINAMATH_CALUDE_prob_at_least_one_male_l734_73438

/-- The number of male students -/
def num_male : ℕ := 3

/-- The number of female students -/
def num_female : ℕ := 2

/-- The total number of students -/
def total_students : ℕ := num_male + num_female

/-- The number of students to be chosen -/
def num_chosen : ℕ := 2

/-- The probability of choosing at least one male student -/
theorem prob_at_least_one_male :
  (1 : ℚ) - (Nat.choose num_female num_chosen : ℚ) / (Nat.choose total_students num_chosen : ℚ) = 9/10 :=
sorry

end NUMINAMATH_CALUDE_prob_at_least_one_male_l734_73438


namespace NUMINAMATH_CALUDE_ellipse_axis_endpoint_distance_l734_73455

/-- Given an ellipse defined by the equation 16(x+2)^2 + 4(y-3)^2 = 64,
    prove that the distance between an endpoint of its major axis
    and an endpoint of its minor axis is 2√5. -/
theorem ellipse_axis_endpoint_distance :
  ∀ (C D : ℝ × ℝ),
  (∀ (x y : ℝ), 16 * (x + 2)^2 + 4 * (y - 3)^2 = 64 →
    (C.1 + 2)^2 / 4 + (C.2 - 3)^2 / 16 = 1 ∧
    (D.1 + 2)^2 / 4 + (D.2 - 3)^2 / 16 = 1 ∧
    ((C.1 + 2)^2 / 4 = 1 ∨ (C.2 - 3)^2 / 16 = 1) ∧
    ((D.1 + 2)^2 / 4 = 1 ∨ (D.2 - 3)^2 / 16 = 1) ∧
    (C.1 + 2)^2 / 4 ≠ (D.1 + 2)^2 / 4) →
  Real.sqrt ((C.1 - D.1)^2 + (C.2 - D.2)^2) = 2 * Real.sqrt 5 := by
sorry

end NUMINAMATH_CALUDE_ellipse_axis_endpoint_distance_l734_73455


namespace NUMINAMATH_CALUDE_value_of_c_l734_73407

theorem value_of_c (k a b c : ℝ) (hk : k ≠ 0) (ha : a ≠ 0) (hb : b ≠ 0) 
  (h : 1 / (k * a) - 1 / (k * b) = 1 / c) : c = k * a * b / (b - a) := by
  sorry

end NUMINAMATH_CALUDE_value_of_c_l734_73407


namespace NUMINAMATH_CALUDE_ricky_rose_distribution_l734_73491

/-- Calculates the number of roses each person receives when Ricky distributes his roses. -/
def roses_per_person (initial_roses : ℕ) (stolen_roses : ℕ) (num_people : ℕ) : ℕ :=
  (initial_roses - stolen_roses) / num_people

/-- Theorem: Given the problem conditions, each person will receive 4 roses. -/
theorem ricky_rose_distribution : roses_per_person 40 4 9 = 4 := by
  sorry

end NUMINAMATH_CALUDE_ricky_rose_distribution_l734_73491


namespace NUMINAMATH_CALUDE_exists_hole_free_square_meter_l734_73464

/-- Represents a point on the carpet -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents the carpet with its dimensions and holes -/
structure Carpet where
  side_length : ℝ
  holes : Finset Point

/-- Represents a square piece that could be cut from the carpet -/
structure SquarePiece where
  bottom_left : Point
  side_length : ℝ

/-- Checks if a point is inside a square piece -/
def point_in_square (p : Point) (s : SquarePiece) : Prop :=
  s.bottom_left.x ≤ p.x ∧ p.x < s.bottom_left.x + s.side_length ∧
  s.bottom_left.y ≤ p.y ∧ p.y < s.bottom_left.y + s.side_length

/-- The main theorem to be proved -/
theorem exists_hole_free_square_meter (c : Carpet) 
    (h_side : c.side_length = 275)
    (h_holes : c.holes.card = 4) :
    ∃ (s : SquarePiece), s.side_length = 100 ∧ 
    s.bottom_left.x + s.side_length ≤ c.side_length ∧
    s.bottom_left.y + s.side_length ≤ c.side_length ∧
    ∀ (p : Point), p ∈ c.holes → ¬point_in_square p s :=
  sorry

end NUMINAMATH_CALUDE_exists_hole_free_square_meter_l734_73464


namespace NUMINAMATH_CALUDE_unique_solution_mn_l734_73469

theorem unique_solution_mn : 
  ∃! (m n : ℕ+), 18 * (m : ℕ) * (n : ℕ) = 72 - 9 * (m : ℕ) - 4 * (n : ℕ) ∧ m = 8 ∧ n = 36 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_mn_l734_73469


namespace NUMINAMATH_CALUDE_f_5_equals_56_l734_73454

def f (x : ℝ) : ℝ := 2*x^7 - 9*x^6 + 5*x^5 - 49*x^4 - 5*x^3 + 2*x^2 + x + 1

def horner_eval (coeffs : List ℝ) (x : ℝ) : ℝ :=
  coeffs.foldl (fun acc a => acc * x + a) 0

theorem f_5_equals_56 :
  f 5 = horner_eval [2, -9, 5, -49, -5, 2, 1, 1] 5 ∧
  horner_eval [2, -9, 5, -49, -5, 2, 1, 1] 5 = 56 := by
  sorry

end NUMINAMATH_CALUDE_f_5_equals_56_l734_73454


namespace NUMINAMATH_CALUDE_fidos_yard_area_fraction_l734_73421

theorem fidos_yard_area_fraction :
  ∀ (s : ℝ), s > 0 →
  (π * s^2) / (4 * s^2) = π / 4 := by
  sorry

end NUMINAMATH_CALUDE_fidos_yard_area_fraction_l734_73421


namespace NUMINAMATH_CALUDE_set_A_properties_l734_73403

def A : Set ℝ := {x | x^2 - 1 = 0}

theorem set_A_properties : 
  (1 ∈ A) ∧ (∅ ⊆ A) ∧ ({1, -1} ⊆ A) := by sorry

end NUMINAMATH_CALUDE_set_A_properties_l734_73403


namespace NUMINAMATH_CALUDE_anns_skating_speed_l734_73418

/-- Proves that Ann's skating speed is 6 miles per hour given the problem conditions. -/
theorem anns_skating_speed :
  ∀ (ann_speed : ℝ),
  let glenda_speed : ℝ := 8
  let time : ℝ := 3
  let total_distance : ℝ := 42
  (ann_speed * time + glenda_speed * time = total_distance) →
  ann_speed = 6 := by
sorry

end NUMINAMATH_CALUDE_anns_skating_speed_l734_73418


namespace NUMINAMATH_CALUDE_joan_has_ten_books_l734_73481

/-- The number of books Tom has -/
def tom_books : ℕ := 38

/-- The total number of books Joan and Tom have together -/
def total_books : ℕ := 48

/-- The number of books Joan has -/
def joan_books : ℕ := total_books - tom_books

theorem joan_has_ten_books : joan_books = 10 := by
  sorry

end NUMINAMATH_CALUDE_joan_has_ten_books_l734_73481


namespace NUMINAMATH_CALUDE_cube_edge_ratio_l734_73476

theorem cube_edge_ratio (v₁ v₂ v₃ v₄ : ℝ) (h : v₁ / v₂ = 216 / 64 ∧ v₂ / v₃ = 64 / 27 ∧ v₃ / v₄ = 27 / 1) :
  ∃ (e₁ e₂ e₃ e₄ : ℝ), v₁ = e₁^3 ∧ v₂ = e₂^3 ∧ v₃ = e₃^3 ∧ v₄ = e₄^3 ∧ 
  e₁ / e₂ = 6 / 4 ∧ e₂ / e₃ = 4 / 3 ∧ e₃ / e₄ = 3 / 1 :=
by sorry

end NUMINAMATH_CALUDE_cube_edge_ratio_l734_73476


namespace NUMINAMATH_CALUDE_birds_meeting_point_l734_73470

/-- The distance between West-town and East-town in kilometers -/
def total_distance : ℝ := 20

/-- The speed of the first bird in kilometers per minute -/
def speed_bird1 : ℝ := 4

/-- The speed of the second bird in kilometers per minute -/
def speed_bird2 : ℝ := 1

/-- The distance traveled by the first bird before meeting -/
def distance_bird1 : ℝ := 16

/-- The distance traveled by the second bird before meeting -/
def distance_bird2 : ℝ := 4

theorem birds_meeting_point :
  distance_bird1 + distance_bird2 = total_distance ∧
  distance_bird1 / speed_bird1 = distance_bird2 / speed_bird2 ∧
  distance_bird1 = 16 := by sorry

end NUMINAMATH_CALUDE_birds_meeting_point_l734_73470


namespace NUMINAMATH_CALUDE_base_b_square_l734_73446

theorem base_b_square (b : ℕ) (hb : b > 1) : 
  (∃ n : ℕ, b^2 + 4*b + 4 = n^2) ↔ b > 4 := by
  sorry

end NUMINAMATH_CALUDE_base_b_square_l734_73446


namespace NUMINAMATH_CALUDE_root_expression_equality_l734_73432

/-- Given a cubic polynomial f(t) with roots p, q, r, and expressions for x, y, z,
    prove that xyz - qrx - rpy - pqz = -674 -/
theorem root_expression_equality (p q r : ℝ) : 
  let f : ℝ → ℝ := fun t ↦ t^3 - 2022*t^2 + 2022*t - 337
  let x := (q-1)*((2022 - q)/(r-1) + (2022 - r)/(p-1))
  let y := (r-1)*((2022 - r)/(p-1) + (2022 - p)/(q-1))
  let z := (p-1)*((2022 - p)/(q-1) + (2022 - q)/(r-1))
  f p = 0 ∧ f q = 0 ∧ f r = 0 →
  x*y*z - q*r*x - r*p*y - p*q*z = -674 := by
sorry

end NUMINAMATH_CALUDE_root_expression_equality_l734_73432


namespace NUMINAMATH_CALUDE_gym_member_ratio_l734_73485

theorem gym_member_ratio (f m : ℕ) (hf : f > 0) (hm : m > 0) :
  (35 : ℝ) * f + 30 * m = 32 * (f + m) →
  (f : ℝ) / m = 2 / 3 := by
sorry

end NUMINAMATH_CALUDE_gym_member_ratio_l734_73485


namespace NUMINAMATH_CALUDE_factorial_sum_equality_l734_73423

theorem factorial_sum_equality : 7 * Nat.factorial 7 + 5 * Nat.factorial 5 - 3 * Nat.factorial 3 + 2 * Nat.factorial 2 = 35866 := by
  sorry

end NUMINAMATH_CALUDE_factorial_sum_equality_l734_73423


namespace NUMINAMATH_CALUDE_hyperbola_sum_l734_73479

theorem hyperbola_sum (h k a b c : ℝ) : 
  h = 1 ∧ 
  k = 2 ∧ 
  c = Real.sqrt 50 ∧ 
  a = 4 ∧ 
  b * b = c * c - a * a → 
  h + k + a + b = 7 + Real.sqrt 34 := by
sorry

end NUMINAMATH_CALUDE_hyperbola_sum_l734_73479


namespace NUMINAMATH_CALUDE_quadratic_root_implies_k_l734_73428

theorem quadratic_root_implies_k (p k : ℝ) : 
  (∃ x : ℂ, 3 * x^2 + p * x + k = 0 ∧ x = 4 + 3*I) → k = 75 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_root_implies_k_l734_73428


namespace NUMINAMATH_CALUDE_line_arrangement_count_l734_73489

def number_of_students : ℕ := 5
def number_of_boys : ℕ := 2
def number_of_girls : ℕ := 3

theorem line_arrangement_count : 
  (number_of_students = number_of_boys + number_of_girls) →
  (number_of_boys = 2) →
  (number_of_girls = 3) →
  (∃ (arrangement_count : ℕ), 
    arrangement_count = (Nat.factorial number_of_boys) * (Nat.factorial (number_of_girls + 1)) ∧
    arrangement_count = 48) :=
by sorry

end NUMINAMATH_CALUDE_line_arrangement_count_l734_73489


namespace NUMINAMATH_CALUDE_equilateral_triangle_on_parabola_l734_73494

/-- An equilateral triangle with one vertex at the origin and the other two on the parabola x^2 = 2y has side length 4√3. -/
theorem equilateral_triangle_on_parabola :
  ∃ (a : ℝ) (v1 v2 : ℝ × ℝ),
    a > 0 ∧
    v1.1^2 = 2 * v1.2 ∧
    v2.1^2 = 2 * v2.2 ∧
    (v1.1 - 0)^2 + (v1.2 - 0)^2 = a^2 ∧
    (v2.1 - 0)^2 + (v2.2 - 0)^2 = a^2 ∧
    (v2.1 - v1.1)^2 + (v2.2 - v1.2)^2 = a^2 ∧
    a = 4 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_equilateral_triangle_on_parabola_l734_73494


namespace NUMINAMATH_CALUDE_octagon_interior_angles_sum_l734_73486

/-- The sum of interior angles of a polygon with n sides is (n - 2) * 180 degrees -/
def sum_interior_angles (n : ℕ) : ℕ := (n - 2) * 180

/-- An octagon is a polygon with 8 sides -/
def octagon_sides : ℕ := 8

/-- The sum of the interior angles of an octagon is 1080 degrees -/
theorem octagon_interior_angles_sum :
  sum_interior_angles octagon_sides = 1080 := by
  sorry

end NUMINAMATH_CALUDE_octagon_interior_angles_sum_l734_73486


namespace NUMINAMATH_CALUDE_y_axis_intersection_l734_73435

/-- The line equation 4y + 3x = 24 -/
def line_equation (x y : ℝ) : Prop := 4 * y + 3 * x = 24

/-- The y-axis is defined as the set of points with x-coordinate equal to 0 -/
def on_y_axis (x y : ℝ) : Prop := x = 0

theorem y_axis_intersection :
  ∃ (x y : ℝ), line_equation x y ∧ on_y_axis x y ∧ x = 0 ∧ y = 6 := by
  sorry

end NUMINAMATH_CALUDE_y_axis_intersection_l734_73435


namespace NUMINAMATH_CALUDE_square_divisibility_l734_73447

theorem square_divisibility (k : ℕ) (n : ℕ) : 
  (∃ m : ℕ, k ^ 2 = n * m) →  -- k^2 is divisible by n
  (∀ j : ℕ, j < k → ¬(∃ m : ℕ, j ^ 2 = n * m)) →  -- k is the least possible value
  k = 60 →  -- the least possible value of k is 60
  n = 3600 :=  -- the number that k^2 is divisible by is 3600
by
  sorry

end NUMINAMATH_CALUDE_square_divisibility_l734_73447


namespace NUMINAMATH_CALUDE_coefficient_x2y2_in_expansion_l734_73401

def binomial_coefficient (n k : ℕ) : ℕ := Nat.choose n k

theorem coefficient_x2y2_in_expansion :
  let n : ℕ := 4
  let k : ℕ := 2
  let coefficient : ℤ := binomial_coefficient n k * (-2)^k
  coefficient = 24 := by sorry

end NUMINAMATH_CALUDE_coefficient_x2y2_in_expansion_l734_73401


namespace NUMINAMATH_CALUDE_cone_sphere_ratio_l734_73416

/-- Proof that the ratio of cone height to base radius is 4/3 when cone volume is 1/3 of sphere volume --/
theorem cone_sphere_ratio (r h : ℝ) (hr : r > 0) :
  (1 / 3) * ((4 / 3) * Real.pi * r^3) = (1 / 3) * Real.pi * r^2 * h → h / r = 4 / 3 := by
  sorry

end NUMINAMATH_CALUDE_cone_sphere_ratio_l734_73416


namespace NUMINAMATH_CALUDE_complex_equation_solution_l734_73475

theorem complex_equation_solution (z : ℂ) (i : ℂ) :
  i * i = -1 →
  i * z = 2 + 4 * i →
  z = 4 - 2 * i :=
by sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l734_73475


namespace NUMINAMATH_CALUDE_second_floor_cost_l734_73477

/-- Represents the cost of rooms on each floor of an apartment building --/
structure ApartmentCosts where
  first_floor : ℕ
  second_floor : ℕ
  third_floor : ℕ

/-- Calculates the total monthly income from all rooms --/
def total_income (costs : ApartmentCosts) : ℕ :=
  3 * (costs.first_floor + costs.second_floor + costs.third_floor)

/-- Theorem stating the cost of rooms on the second floor --/
theorem second_floor_cost (costs : ApartmentCosts) :
  costs.first_floor = 15 →
  costs.third_floor = 2 * costs.first_floor →
  total_income costs = 165 →
  costs.second_floor = 10 := by
  sorry

#check second_floor_cost

end NUMINAMATH_CALUDE_second_floor_cost_l734_73477


namespace NUMINAMATH_CALUDE_evaluate_expression_l734_73434

theorem evaluate_expression (a : ℝ) : 
  let x : ℝ := a + 9
  (x - a + 5) = 14 := by sorry

end NUMINAMATH_CALUDE_evaluate_expression_l734_73434


namespace NUMINAMATH_CALUDE_sum_of_coefficients_l734_73493

theorem sum_of_coefficients (a b c d e : ℝ) : 
  (∀ x, 512 * x^3 + 27 = (a * x + b) * (c * x^2 + d * x + e)) →
  a + b + c + d + e = 60 := by
sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_l734_73493


namespace NUMINAMATH_CALUDE_inequality_proof_l734_73460

theorem inequality_proof (x y z : ℝ) (hx : x > -1) (hy : y > -1) (hz : z > -1) :
  (1 + x^2) / (1 + y + z^2) + (1 + y^2) / (1 + z + x^2) + (1 + z^2) / (1 + x + y^2) ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l734_73460


namespace NUMINAMATH_CALUDE_jerry_added_two_figures_l734_73439

/-- Represents the number of action figures Jerry added to the shelf. -/
def added_figures : ℕ := sorry

/-- The initial number of books on the shelf. -/
def initial_books : ℕ := 7

/-- The initial number of action figures on the shelf. -/
def initial_figures : ℕ := 3

/-- The difference between the number of books and action figures after adding. -/
def book_figure_difference : ℕ := 2

theorem jerry_added_two_figures : 
  added_figures = 2 ∧ 
  initial_books = (initial_figures + added_figures) + book_figure_difference :=
sorry

end NUMINAMATH_CALUDE_jerry_added_two_figures_l734_73439


namespace NUMINAMATH_CALUDE_train_distance_l734_73448

/-- Proves that a train traveling at a rate of 1 mile per 2 minutes will cover 15 miles in 30 minutes. -/
theorem train_distance (rate : ℚ) (time : ℚ) (distance : ℚ) : 
  rate = 1 / 2 →  -- The train travels 1 mile in 2 minutes
  time = 30 →     -- We want to know the distance traveled in 30 minutes
  distance = rate * time →  -- Distance is calculated as rate times time
  distance = 15 :=  -- The train will travel 15 miles
by
  sorry

end NUMINAMATH_CALUDE_train_distance_l734_73448


namespace NUMINAMATH_CALUDE_point_coordinates_wrt_origin_l734_73463

/-- Given a point P with coordinates (2, -3), prove that its coordinates with respect to the origin are (2, -3) -/
theorem point_coordinates_wrt_origin : 
  let P : ℝ × ℝ := (2, -3)
  P = (2, -3) := by sorry

end NUMINAMATH_CALUDE_point_coordinates_wrt_origin_l734_73463


namespace NUMINAMATH_CALUDE_triangle_areas_l734_73413

-- Define the triangle ABC
structure Triangle :=
  (BC : ℝ)
  (AC : ℝ)
  (AB : ℝ)

-- Define the areas of the triangles formed by altitude and median
def AreaTriangles (t : Triangle) : (ℝ × ℝ × ℝ) :=
  sorry

-- Theorem statement
theorem triangle_areas (t : Triangle) 
  (h1 : t.BC = 3)
  (h2 : t.AC = 4)
  (h3 : t.AB = 5) :
  AreaTriangles t = (3, 0.84, 2.16) :=
sorry

end NUMINAMATH_CALUDE_triangle_areas_l734_73413


namespace NUMINAMATH_CALUDE_simplify_and_evaluate_l734_73433

theorem simplify_and_evaluate (a : ℚ) (h : a = -2) :
  (2 / (a - 1) - 1 / a) / ((a^2 + a) / (a^2 - 2*a + 1)) = -3/4 := by
  sorry

end NUMINAMATH_CALUDE_simplify_and_evaluate_l734_73433


namespace NUMINAMATH_CALUDE_orange_bin_calculation_l734_73471

/-- Calculates the final number of oranges in a bin after a series of transactions -/
theorem orange_bin_calculation (initial : ℕ) (sold : ℕ) (new_shipment : ℕ) : 
  initial = 124 → sold = 46 → new_shipment = 250 → 
  (initial - sold - (initial - sold) / 2 + new_shipment) = 289 := by
  sorry

end NUMINAMATH_CALUDE_orange_bin_calculation_l734_73471


namespace NUMINAMATH_CALUDE_solution_set_characterization_l734_73436

def is_even (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

def monotone_decreasing_on (f : ℝ → ℝ) (s : Set ℝ) : Prop :=
  ∀ {x y}, x ∈ s → y ∈ s → x ≤ y → f y ≤ f x

theorem solution_set_characterization
  (f : ℝ → ℝ)
  (h_even : is_even f)
  (h_decreasing : monotone_decreasing_on f (Set.Ici 0))
  (h_f1 : f 1 = 0) :
  {x | f x > 0} = {x | -1 < x ∧ x < 1} := by sorry

end NUMINAMATH_CALUDE_solution_set_characterization_l734_73436


namespace NUMINAMATH_CALUDE_area_is_192_l734_73472

/-- A right triangle with a circle tangent to its legs -/
structure RightTriangleWithTangentCircle where
  /-- The circle cuts the hypotenuse into segments of lengths 1, 24, and 3 -/
  hypotenuse_segments : ℝ × ℝ × ℝ
  /-- The middle segment (of length 24) is a chord of the circle -/
  middle_segment_is_chord : hypotenuse_segments.2.1 = 24

/-- The area of a right triangle with a tangent circle satisfying specific conditions -/
def area (t : RightTriangleWithTangentCircle) : ℝ := sorry

/-- Theorem: The area of the triangle is 192 -/
theorem area_is_192 (t : RightTriangleWithTangentCircle) 
  (h1 : t.hypotenuse_segments.1 = 1)
  (h2 : t.hypotenuse_segments.2.2 = 3) : 
  area t = 192 := by sorry

end NUMINAMATH_CALUDE_area_is_192_l734_73472


namespace NUMINAMATH_CALUDE_max_power_under_500_l734_73400

theorem max_power_under_500 :
  ∃ (a b : ℕ), 
    a > 0 ∧ b > 1 ∧ 
    a^b < 500 ∧
    (∀ (c d : ℕ), c > 0 → d > 1 → c^d < 500 → c^d ≤ a^b) ∧
    a = 22 ∧ b = 2 ∧ 
    a + b = 24 := by
  sorry

end NUMINAMATH_CALUDE_max_power_under_500_l734_73400


namespace NUMINAMATH_CALUDE_unique_solution_value_l734_73466

theorem unique_solution_value (k : ℝ) : 
  (∃! x : ℝ, (x + 3) / (k * x + 2) = x) ↔ k = -1/12 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_value_l734_73466


namespace NUMINAMATH_CALUDE_bus_trip_speed_l734_73445

theorem bus_trip_speed (distance : ℝ) (speed_increase : ℝ) (time_decrease : ℝ) :
  distance = 210 ∧ speed_increase = 5 ∧ time_decrease = 1 →
  ∃ (original_speed : ℝ),
    distance / original_speed - time_decrease = distance / (original_speed + speed_increase) ∧
    original_speed = 30 :=
by sorry

end NUMINAMATH_CALUDE_bus_trip_speed_l734_73445


namespace NUMINAMATH_CALUDE_seats_needed_l734_73429

/-- Given 58 children and 2 children per seat, prove that 29 seats are needed. -/
theorem seats_needed (total_children : ℕ) (children_per_seat : ℕ) (h1 : total_children = 58) (h2 : children_per_seat = 2) :
  total_children / children_per_seat = 29 := by
  sorry

end NUMINAMATH_CALUDE_seats_needed_l734_73429


namespace NUMINAMATH_CALUDE_deal_or_no_deal_probability_l734_73495

theorem deal_or_no_deal_probability (total : Nat) (desired : Nat) (chosen : Nat) 
  (h1 : total = 26)
  (h2 : desired = 9)
  (h3 : chosen = 1) :
  ∃ (removed : Nat), 
    (1 : ℚ) * desired / (total - removed - chosen) ≥ (1 : ℚ) / 2 ∧ 
    ∀ (r : Nat), r < removed → (1 : ℚ) * desired / (total - r - chosen) < (1 : ℚ) / 2 :=
by sorry

end NUMINAMATH_CALUDE_deal_or_no_deal_probability_l734_73495


namespace NUMINAMATH_CALUDE_triangle_theorem_l734_73443

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively --/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The given condition a/(√3 * cos A) = c/sin C --/
def condition (t : Triangle) : Prop :=
  t.a / (Real.sqrt 3 * Real.cos t.A) = t.c / Real.sin t.C

/-- The theorem to be proved --/
theorem triangle_theorem (t : Triangle) (h : condition t) (ha : t.a = 6) :
  t.A = π / 3 ∧ 6 < t.b + t.c ∧ t.b + t.c ≤ 12 := by
  sorry

end NUMINAMATH_CALUDE_triangle_theorem_l734_73443


namespace NUMINAMATH_CALUDE_gasoline_spending_increase_l734_73426

theorem gasoline_spending_increase (P Q : ℝ) (P_increase : ℝ) (Q_decrease : ℝ) :
  P > 0 ∧ Q > 0 ∧ P_increase = 0.25 ∧ Q_decrease = 0.16 →
  (1 + 0.05) * (P * Q) = (P * (1 + P_increase)) * (Q * (1 - Q_decrease)) :=
by sorry

end NUMINAMATH_CALUDE_gasoline_spending_increase_l734_73426


namespace NUMINAMATH_CALUDE_permutations_of_six_distinct_objects_l734_73405

theorem permutations_of_six_distinct_objects : Nat.factorial 6 = 720 := by
  sorry

end NUMINAMATH_CALUDE_permutations_of_six_distinct_objects_l734_73405


namespace NUMINAMATH_CALUDE_function_inequality_l734_73482

open Real

theorem function_inequality (f : ℝ → ℝ) (hf : Differentiable ℝ f) :
  (∀ x ∈ (Set.Ioo 0 (π / 2)), deriv f x * sin x < f x * cos x) →
  Real.sqrt 3 * f (π / 4) > Real.sqrt 2 * f (π / 3) := by
  sorry

end NUMINAMATH_CALUDE_function_inequality_l734_73482
