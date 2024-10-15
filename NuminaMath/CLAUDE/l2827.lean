import Mathlib

namespace NUMINAMATH_CALUDE_rhombus_diagonal_l2827_282701

/-- Given a rhombus with one diagonal of length 60 meters and an area of 1950 square meters,
    prove that the length of the other diagonal is 65 meters. -/
theorem rhombus_diagonal (d₁ : ℝ) (d₂ : ℝ) (area : ℝ) 
    (h₁ : d₁ = 60)
    (h₂ : area = 1950)
    (h₃ : area = (d₁ * d₂) / 2) : 
  d₂ = 65 := by
  sorry

end NUMINAMATH_CALUDE_rhombus_diagonal_l2827_282701


namespace NUMINAMATH_CALUDE_smallest_cube_volume_for_pyramid_l2827_282756

/-- Represents the dimensions of a rectangular pyramid -/
structure PyramidDimensions where
  height : ℝ
  baseLength : ℝ
  baseWidth : ℝ

/-- Calculates the volume of a cube given its side length -/
def cubeVolume (side : ℝ) : ℝ := side^3

/-- Theorem: The volume of the smallest cube-shaped box that can house a given rectangular pyramid uprightly -/
theorem smallest_cube_volume_for_pyramid (p : PyramidDimensions) 
  (h_height : p.height = 15)
  (h_length : p.baseLength = 8)
  (h_width : p.baseWidth = 12) :
  cubeVolume (max p.height (max p.baseLength p.baseWidth)) = 3375 := by
  sorry

#check smallest_cube_volume_for_pyramid

end NUMINAMATH_CALUDE_smallest_cube_volume_for_pyramid_l2827_282756


namespace NUMINAMATH_CALUDE_halloween_candy_weight_l2827_282737

/-- The combined weight of candy for Frank and Gwen -/
theorem halloween_candy_weight (frank_candy : ℕ) (gwen_candy : ℕ) 
  (h1 : frank_candy = 10) (h2 : gwen_candy = 7) : 
  frank_candy + gwen_candy = 17 := by
  sorry

end NUMINAMATH_CALUDE_halloween_candy_weight_l2827_282737


namespace NUMINAMATH_CALUDE_complex_absolute_value_l2827_282757

theorem complex_absolute_value (z : ℂ) : z = 10 + 3*I → Complex.abs (z^2 + 8*z + 85) = 4 * Real.sqrt 3922 := by
  sorry

end NUMINAMATH_CALUDE_complex_absolute_value_l2827_282757


namespace NUMINAMATH_CALUDE_set_operation_result_l2827_282707

-- Define the sets A, B, and C
def A : Set ℕ := {0, 1, 2, 4, 5, 7}
def B : Set ℕ := {1, 3, 6, 8, 9}
def C : Set ℕ := {3, 7, 8}

-- State the theorem
theorem set_operation_result : (A ∪ B) ∩ C = {3, 7, 8} := by
  sorry

end NUMINAMATH_CALUDE_set_operation_result_l2827_282707


namespace NUMINAMATH_CALUDE_multiple_calculation_l2827_282762

theorem multiple_calculation (n a : ℕ) (m : ℚ) 
  (h1 : n = 16) 
  (h2 : a = 12) 
  (h3 : m * n - a = 20) : 
  m = 2 := by
  sorry

end NUMINAMATH_CALUDE_multiple_calculation_l2827_282762


namespace NUMINAMATH_CALUDE_negative_fraction_comparison_l2827_282731

theorem negative_fraction_comparison : -5/4 > -4/3 := by
  sorry

end NUMINAMATH_CALUDE_negative_fraction_comparison_l2827_282731


namespace NUMINAMATH_CALUDE_smallest_n_congruence_l2827_282729

theorem smallest_n_congruence (n : ℕ) : 
  (n > 0 ∧ 29 * n ≡ 5678 [ZMOD 11] ∧ ∀ m : ℕ, (0 < m ∧ m < n) → ¬(29 * m ≡ 5678 [ZMOD 11])) ↔ 
  n = 9 := by
sorry

end NUMINAMATH_CALUDE_smallest_n_congruence_l2827_282729


namespace NUMINAMATH_CALUDE_hyperbola_vertex_distance_l2827_282785

/-- The distance between the vertices of the hyperbola x^2/144 - y^2/49 = 1 is 24 -/
theorem hyperbola_vertex_distance : 
  let a : ℝ := Real.sqrt 144
  let hyperbola := fun (x y : ℝ) ↦ x^2 / 144 - y^2 / 49 = 1
  2 * a = 24 := by sorry

end NUMINAMATH_CALUDE_hyperbola_vertex_distance_l2827_282785


namespace NUMINAMATH_CALUDE_cubic_difference_l2827_282709

theorem cubic_difference (a b : ℝ) (h1 : a - b = 6) (h2 : a^2 + b^2 = 50) :
  a^3 - b^3 = 342 := by
  sorry

end NUMINAMATH_CALUDE_cubic_difference_l2827_282709


namespace NUMINAMATH_CALUDE_sum_distinct_prime_divisors_1800_l2827_282793

def sum_of_distinct_prime_divisors (n : Nat) : Nat :=
  (Nat.factors n).toFinset.sum id

theorem sum_distinct_prime_divisors_1800 :
  sum_of_distinct_prime_divisors 1800 = 10 := by
  sorry

end NUMINAMATH_CALUDE_sum_distinct_prime_divisors_1800_l2827_282793


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l2827_282735

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_sum (a : ℕ → ℝ) :
  arithmetic_sequence a →
  (a 3 + a 4 + a 5 + a 6 + a 7 = 450) →
  (a 2 + a 8 = 180) := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l2827_282735


namespace NUMINAMATH_CALUDE_max_polyline_length_6x10_l2827_282704

/-- Represents a checkered field with rows and columns -/
structure CheckeredField where
  rows : Nat
  columns : Nat

/-- Represents a polyline on a checkered field -/
structure Polyline where
  field : CheckeredField
  length : Nat
  closed : Bool
  nonSelfIntersecting : Bool

/-- The maximum length of a closed, non-self-intersecting polyline on a given field -/
def maxPolylineLength (field : CheckeredField) : Nat :=
  sorry

/-- Theorem: The maximum length of a closed, non-self-intersecting polyline
    on a 6 × 10 checkered field is 76 -/
theorem max_polyline_length_6x10 :
  let field := CheckeredField.mk 6 10
  maxPolylineLength field = 76 := by
  sorry

end NUMINAMATH_CALUDE_max_polyline_length_6x10_l2827_282704


namespace NUMINAMATH_CALUDE_extreme_value_implies_a_range_l2827_282727

/-- The function f(x) defined in terms of a parameter a -/
def f (a : ℝ) (x : ℝ) : ℝ := x^3 - 3*a*x^2 + 3*x + 1

/-- The derivative of f(x) with respect to x -/
def f_deriv (a : ℝ) (x : ℝ) : ℝ := 3*x^2 - 6*a*x + 3

/-- Theorem: If f(x) has at least one extreme value point in (2, 3), then 5/4 < a < 5/3 -/
theorem extreme_value_implies_a_range (a : ℝ) :
  (∃ x : ℝ, 2 < x ∧ x < 3 ∧ f_deriv a x = 0) →
  (5/4 : ℝ) < a ∧ a < (5/3 : ℝ) :=
by
  sorry


end NUMINAMATH_CALUDE_extreme_value_implies_a_range_l2827_282727


namespace NUMINAMATH_CALUDE_completing_square_equivalence_l2827_282726

theorem completing_square_equivalence :
  ∀ x : ℝ, x^2 + 8*x + 9 = 0 ↔ (x + 4)^2 = 7 :=
by sorry

end NUMINAMATH_CALUDE_completing_square_equivalence_l2827_282726


namespace NUMINAMATH_CALUDE_water_storage_solution_l2827_282741

/-- Represents the water storage problem with barrels and casks. -/
def WaterStorage (cask_capacity : ℕ) (barrel_count : ℕ) : Prop :=
  let barrel_capacity := 2 * cask_capacity + 3
  barrel_count * barrel_capacity = 172

/-- Theorem stating that given the problem conditions, the total water storage is 172 gallons. -/
theorem water_storage_solution :
  WaterStorage 20 4 := by
  sorry

end NUMINAMATH_CALUDE_water_storage_solution_l2827_282741


namespace NUMINAMATH_CALUDE_slower_train_speed_l2827_282744

/-- Proves that the speed of the slower train is 36 km/hr given the problem conditions -/
theorem slower_train_speed (faster_speed : ℝ) (passing_time : ℝ) (train_length : ℝ)
  (h1 : faster_speed = 46)
  (h2 : passing_time = 54)
  (h3 : train_length = 75) :
  ∃ (slower_speed : ℝ),
    slower_speed = 36 ∧
    (faster_speed - slower_speed) * passing_time * (1000 / 3600) = 2 * train_length :=
by
  sorry

#check slower_train_speed

end NUMINAMATH_CALUDE_slower_train_speed_l2827_282744


namespace NUMINAMATH_CALUDE_division_subtraction_equality_l2827_282740

theorem division_subtraction_equality : 144 / (12 / 3) - 5 = 31 := by
  sorry

end NUMINAMATH_CALUDE_division_subtraction_equality_l2827_282740


namespace NUMINAMATH_CALUDE_solve_trailer_problem_l2827_282719

/-- Represents the trailer home problem --/
def trailer_problem (initial_count : ℕ) (initial_avg_age : ℕ) (current_avg_age : ℕ) (time_elapsed : ℕ) : Prop :=
  ∃ (new_count : ℕ),
    (initial_count * (initial_avg_age + time_elapsed) + new_count * time_elapsed) / (initial_count + new_count) = current_avg_age

/-- The theorem statement for the trailer home problem --/
theorem solve_trailer_problem :
  trailer_problem 30 15 10 3 → ∃ (new_count : ℕ), new_count = 34 :=
by
  sorry


end NUMINAMATH_CALUDE_solve_trailer_problem_l2827_282719


namespace NUMINAMATH_CALUDE_larger_number_problem_l2827_282771

theorem larger_number_problem (L S : ℕ) (h1 : L > S) (h2 : L - S = 1355) (h3 : L = 6 * S + 15) : L = 1623 := by
  sorry

end NUMINAMATH_CALUDE_larger_number_problem_l2827_282771


namespace NUMINAMATH_CALUDE_max_garden_area_l2827_282734

/-- Represents a rectangular garden -/
structure Garden where
  length : ℝ
  width : ℝ

/-- The perimeter of the garden -/
def Garden.perimeter (g : Garden) : ℝ := 2 * (g.length + g.width)

/-- The area of the garden -/
def Garden.area (g : Garden) : ℝ := g.length * g.width

/-- Theorem stating the maximum area of a rectangular garden with given constraints -/
theorem max_garden_area (g : Garden) 
  (h_perimeter : g.perimeter = 400) 
  (h_min_length : g.length ≥ 100) : 
  g.area ≤ 10000 ∧ (g.area = 10000 ↔ g.length = 100 ∧ g.width = 100) := by
  sorry

#check max_garden_area

end NUMINAMATH_CALUDE_max_garden_area_l2827_282734


namespace NUMINAMATH_CALUDE_identify_real_coins_l2827_282792

/-- Represents the result of weighing two coins -/
inductive WeighResult
| Equal : WeighResult
| LeftHeavier : WeighResult
| RightHeavier : WeighResult

/-- Represents a coin -/
structure Coin :=
  (id : Nat)
  (isReal : Bool)

/-- Represents the balance scale that always shows an incorrect result -/
def incorrectBalance (left right : Coin) : WeighResult :=
  sorry

/-- The main theorem to prove -/
theorem identify_real_coins 
  (coins : Finset Coin) 
  (h_count : coins.card = 100) 
  (h_real : ∃ (fake : Coin), fake ∈ coins ∧ 
    (∀ c ∈ coins, c ≠ fake → c.isReal) ∧ 
    (¬fake.isReal)) : 
  ∃ (realCoins : Finset Coin), realCoins ⊆ coins ∧ realCoins.card = 98 ∧ 
    (∀ c ∈ realCoins, c.isReal) :=
  sorry

end NUMINAMATH_CALUDE_identify_real_coins_l2827_282792


namespace NUMINAMATH_CALUDE_sequence_properties_l2827_282789

theorem sequence_properties :
  (∀ n m : ℕ, (2 * n)^2 + 1 ≠ 3 * m^2) ∧
  (∀ p q : ℕ, p^2 + 1 ≠ 7 * q^2) := by
  sorry

end NUMINAMATH_CALUDE_sequence_properties_l2827_282789


namespace NUMINAMATH_CALUDE_truck_distance_l2827_282710

/-- Proves the distance traveled by a truck in yards over 5 minutes -/
theorem truck_distance (b t : ℝ) (h1 : t > 0) : 
  let feet_per_t_seconds : ℝ := b / 4
  let feet_in_yard : ℝ := 2
  let minutes : ℝ := 5
  let seconds_in_minute : ℝ := 60
  let yards_traveled : ℝ := (feet_per_t_seconds * (minutes * seconds_in_minute) / t) / feet_in_yard
  yards_traveled = 37.5 * b / t := by
sorry

end NUMINAMATH_CALUDE_truck_distance_l2827_282710


namespace NUMINAMATH_CALUDE_students_playing_neither_sport_l2827_282788

theorem students_playing_neither_sport
  (total : ℕ)
  (football : ℕ)
  (tennis : ℕ)
  (both : ℕ)
  (h1 : total = 50)
  (h2 : football = 32)
  (h3 : tennis = 28)
  (h4 : both = 24) :
  total - (football + tennis - both) = 14 :=
by sorry

end NUMINAMATH_CALUDE_students_playing_neither_sport_l2827_282788


namespace NUMINAMATH_CALUDE_fruit_crate_total_l2827_282738

theorem fruit_crate_total (strawberry_count : ℕ) (kiwi_fraction : ℚ) 
  (h1 : kiwi_fraction = 1/3)
  (h2 : strawberry_count = 52) :
  ∃ (total : ℕ), total = 78 ∧ 
    strawberry_count = (1 - kiwi_fraction) * total ∧
    kiwi_fraction * total + strawberry_count = total := by
  sorry

end NUMINAMATH_CALUDE_fruit_crate_total_l2827_282738


namespace NUMINAMATH_CALUDE_polygon_sides_l2827_282758

theorem polygon_sides (n : ℕ) : n ≥ 3 →
  (n - 2) * 180 = 3 * 360 - 180 → n = 7 := by
  sorry

end NUMINAMATH_CALUDE_polygon_sides_l2827_282758


namespace NUMINAMATH_CALUDE_diagonal_passes_900_cubes_l2827_282736

/-- The number of cubes an internal diagonal passes through in a rectangular solid -/
def cubes_passed (x y z : ℕ) : ℕ :=
  x + y + z - (Nat.gcd x y + Nat.gcd y z + Nat.gcd z x) + Nat.gcd x (Nat.gcd y z)

/-- Theorem: An internal diagonal of a 200 × 400 × 500 rectangular solid
    passes through 900 cubes -/
theorem diagonal_passes_900_cubes :
  cubes_passed 200 400 500 = 900 := by
  sorry

end NUMINAMATH_CALUDE_diagonal_passes_900_cubes_l2827_282736


namespace NUMINAMATH_CALUDE_absolute_value_inequality_solution_set_l2827_282764

theorem absolute_value_inequality_solution_set :
  {x : ℝ | |x + 1| > 1} = {x : ℝ | x < -2 ∨ x > 0} := by sorry

end NUMINAMATH_CALUDE_absolute_value_inequality_solution_set_l2827_282764


namespace NUMINAMATH_CALUDE_inequality_proof_l2827_282778

theorem inequality_proof (a b c d e f : ℝ) 
  (h_pos : a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧ e > 0 ∧ f > 0) 
  (h_cond : |Real.sqrt (a * b) - Real.sqrt (c * d)| ≤ 2) : 
  (e / a + b / e) * (e / c + d / e) ≥ (f / a - b) * (d - f / c) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2827_282778


namespace NUMINAMATH_CALUDE_inverse_f_at_120_l2827_282799

-- Define the function f
def f (x : ℝ) : ℝ := 3 * x^3 + 9

-- State the theorem
theorem inverse_f_at_120 :
  ∃ (y : ℝ), f y = 120 ∧ y = (37 : ℝ)^(1/3) :=
sorry

end NUMINAMATH_CALUDE_inverse_f_at_120_l2827_282799


namespace NUMINAMATH_CALUDE_paul_sold_94_books_l2827_282721

/-- Calculates the number of books Paul sold given his initial, purchased, and final book counts. -/
def books_sold (initial : ℕ) (purchased : ℕ) (final : ℕ) : ℕ :=
  initial + purchased - final

theorem paul_sold_94_books : books_sold 2 150 58 = 94 := by
  sorry

end NUMINAMATH_CALUDE_paul_sold_94_books_l2827_282721


namespace NUMINAMATH_CALUDE_sum_of_p_x_coordinates_l2827_282761

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Calculates the area of a triangle given three points -/
def triangleArea (a b c : Point) : ℝ := sorry

/-- Theorem: Sum of possible x-coordinates of P -/
theorem sum_of_p_x_coordinates : ∃ (P₁ P₂ P₃ P₄ : Point),
  let Q : Point := ⟨0, 0⟩
  let R : Point := ⟨368, 0⟩
  let S₁ : Point := ⟨901, 501⟩
  let S₂ : Point := ⟨912, 514⟩
  triangleArea P₁ Q R = 4128 ∧
  triangleArea P₂ Q R = 4128 ∧
  triangleArea P₃ Q R = 4128 ∧
  triangleArea P₄ Q R = 4128 ∧
  (triangleArea P₁ R S₁ = 12384 ∨ triangleArea P₁ R S₂ = 12384) ∧
  (triangleArea P₂ R S₁ = 12384 ∨ triangleArea P₂ R S₂ = 12384) ∧
  (triangleArea P₃ R S₁ = 12384 ∨ triangleArea P₃ R S₂ = 12384) ∧
  (triangleArea P₄ R S₁ = 12384 ∨ triangleArea P₄ R S₂ = 12384) ∧
  P₁.x + P₂.x + P₃.x + P₄.x = 4000 :=
sorry

end NUMINAMATH_CALUDE_sum_of_p_x_coordinates_l2827_282761


namespace NUMINAMATH_CALUDE_gcd_count_equals_fourteen_l2827_282754

theorem gcd_count_equals_fourteen : 
  (Finset.filter (fun n : ℕ => Nat.gcd 21 n = 7) (Finset.range 150)).card = 14 := by
  sorry

end NUMINAMATH_CALUDE_gcd_count_equals_fourteen_l2827_282754


namespace NUMINAMATH_CALUDE_range_of_a_l2827_282711

def P : Set ℝ := {x : ℝ | x^2 ≤ 4}
def M (a : ℝ) : Set ℝ := {a}

theorem range_of_a (a : ℝ) : P ∪ M a = P → a ∈ Set.Icc (-2) 2 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l2827_282711


namespace NUMINAMATH_CALUDE_birthday_cake_division_l2827_282765

/-- Calculates the weight of cake each of Juelz's sisters received after the birthday party -/
theorem birthday_cake_division (total_pieces : ℕ) (square_pieces : ℕ) (triangle_pieces : ℕ)
  (square_weight : ℕ) (triangle_weight : ℕ) (square_eaten_percent : ℚ) 
  (triangle_eaten_percent : ℚ) (forest_family_percent : ℚ) (friends_percent : ℚ) 
  (num_sisters : ℕ) :
  total_pieces = square_pieces + triangle_pieces →
  square_pieces = 160 →
  triangle_pieces = 80 →
  square_weight = 25 →
  triangle_weight = 20 →
  square_eaten_percent = 60 / 100 →
  triangle_eaten_percent = 40 / 100 →
  forest_family_percent = 30 / 100 →
  friends_percent = 25 / 100 →
  num_sisters = 3 →
  ∃ (sisters_share : ℕ), sisters_share = 448 ∧
    sisters_share = 
      ((1 - friends_percent) * 
       ((1 - forest_family_percent) * 
        ((square_pieces * (1 - square_eaten_percent) * square_weight) + 
         (triangle_pieces * (1 - triangle_eaten_percent) * triangle_weight)))) / num_sisters :=
by sorry

end NUMINAMATH_CALUDE_birthday_cake_division_l2827_282765


namespace NUMINAMATH_CALUDE_circle_properties_l2827_282767

/-- The equation of a circle in the xy-plane -/
def circle_equation (x y : ℝ) : Prop := x^2 + y^2 - 4*x = 0

/-- The center of the circle -/
def circle_center : ℝ × ℝ := (2, 0)

/-- The radius of the circle -/
def circle_radius : ℝ := 2

/-- Theorem stating that the given equation describes a circle with the specified center and radius -/
theorem circle_properties :
  ∀ (x y : ℝ), circle_equation x y ↔ (x - circle_center.1)^2 + (y - circle_center.2)^2 = circle_radius^2 :=
by sorry

end NUMINAMATH_CALUDE_circle_properties_l2827_282767


namespace NUMINAMATH_CALUDE_fraction_calculation_l2827_282700

theorem fraction_calculation : 
  (1/4 + 1/5) / (3/7 - 1/8) = 42/25 := by sorry

end NUMINAMATH_CALUDE_fraction_calculation_l2827_282700


namespace NUMINAMATH_CALUDE_equation_solution_l2827_282747

theorem equation_solution : ∃ x : ℝ, (x + 1 ≠ 0 ∧ x^2 - 1 ≠ 0) ∧ 
  (2 * x / (x + 1) - 2 = 3 / (x^2 - 1)) ∧ x = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2827_282747


namespace NUMINAMATH_CALUDE_orange_apple_cost_l2827_282795

/-- The cost of oranges and apples given specific quantities and prices -/
theorem orange_apple_cost (orange_price apple_price : ℕ) 
  (h1 : 6 * orange_price + 5 * apple_price = 419)
  (h2 : orange_price = 29)
  (h3 : apple_price = 29) :
  5 * orange_price + 7 * apple_price = 348 := by
  sorry

#check orange_apple_cost

end NUMINAMATH_CALUDE_orange_apple_cost_l2827_282795


namespace NUMINAMATH_CALUDE_james_toys_l2827_282755

theorem james_toys (toy_cars : ℕ) (toy_soldiers : ℕ) : 
  toy_cars = 20 → 
  toy_soldiers = 2 * toy_cars → 
  toy_cars + toy_soldiers = 60 := by
  sorry

end NUMINAMATH_CALUDE_james_toys_l2827_282755


namespace NUMINAMATH_CALUDE_last_two_digits_sum_l2827_282748

theorem last_two_digits_sum : (13^27 + 17^27) % 100 = 90 := by
  sorry

end NUMINAMATH_CALUDE_last_two_digits_sum_l2827_282748


namespace NUMINAMATH_CALUDE_distance_between_trees_l2827_282783

theorem distance_between_trees (yard_length : ℝ) (num_trees : ℕ) 
  (h1 : yard_length = 180)
  (h2 : num_trees = 11)
  (h3 : num_trees ≥ 2) :
  let distance := yard_length / (num_trees - 1)
  distance = 18 := by
  sorry

end NUMINAMATH_CALUDE_distance_between_trees_l2827_282783


namespace NUMINAMATH_CALUDE_chocolate_distribution_l2827_282746

/-- Represents the number of bags of chocolates initially bought by Robie -/
def initial_bags : ℕ := 3

/-- Represents the number of pieces of chocolate in each bag -/
def pieces_per_bag : ℕ := 30

/-- Represents the number of bags given to siblings -/
def bags_to_siblings : ℕ := 2

/-- Represents the number of Robie's siblings -/
def num_siblings : ℕ := 4

/-- Represents the percentage of chocolates received by the oldest sibling -/
def oldest_sibling_share : ℚ := 40 / 100

/-- Represents the percentage of chocolates received by the second oldest sibling -/
def second_oldest_sibling_share : ℚ := 30 / 100

/-- Represents the percentage of chocolates shared by the last two siblings -/
def youngest_siblings_share : ℚ := 30 / 100

/-- Represents the number of additional bags bought by Robie -/
def additional_bags : ℕ := 3

/-- Represents the discount percentage on the third additional bag -/
def discount_percentage : ℚ := 50 / 100

/-- Represents the cost of each non-discounted bag in dollars -/
def cost_per_bag : ℕ := 12

/-- Theorem stating the total amount spent, Robie's remaining chocolates, and siblings' remaining chocolates -/
theorem chocolate_distribution :
  let total_spent := initial_bags * cost_per_bag + 
                     (additional_bags - 1) * cost_per_bag + 
                     (1 - discount_percentage) * cost_per_bag
  let robie_remaining := (initial_bags - bags_to_siblings) * pieces_per_bag + 
                         additional_bags * pieces_per_bag
  let sibling_remaining := 0
  (total_spent = 66 ∧ robie_remaining = 90 ∧ sibling_remaining = 0) := by sorry

end NUMINAMATH_CALUDE_chocolate_distribution_l2827_282746


namespace NUMINAMATH_CALUDE_total_dogs_count_l2827_282730

/-- The number of boxes containing stuffed toy dogs -/
def num_boxes : ℕ := 7

/-- The number of dogs in each box -/
def dogs_per_box : ℕ := 4

/-- The total number of dogs -/
def total_dogs : ℕ := num_boxes * dogs_per_box

theorem total_dogs_count : total_dogs = 28 := by
  sorry

end NUMINAMATH_CALUDE_total_dogs_count_l2827_282730


namespace NUMINAMATH_CALUDE_planes_perp_to_line_are_parallel_lines_perp_to_plane_are_parallel_lines_perp_to_line_not_always_parallel_planes_perp_to_plane_not_always_parallel_l2827_282780

-- Define basic geometric objects
variable (Point Line Plane : Type)

-- Define perpendicular and parallel relations
variable (perpendicular_line_plane : Line → Plane → Prop)
variable (perpendicular_plane_line : Plane → Line → Prop)
variable (perpendicular_line_line : Line → Line → Prop)
variable (perpendicular_plane_plane : Plane → Plane → Prop)
variable (parallel_line : Line → Line → Prop)
variable (parallel_plane : Plane → Plane → Prop)

-- Theorem for proposition ②
theorem planes_perp_to_line_are_parallel 
  (l : Line) (p1 p2 : Plane) 
  (h1 : perpendicular_plane_line p1 l) 
  (h2 : perpendicular_plane_line p2 l) : 
  parallel_plane p1 p2 :=
sorry

-- Theorem for proposition ③
theorem lines_perp_to_plane_are_parallel 
  (p : Plane) (l1 l2 : Line) 
  (h1 : perpendicular_line_plane l1 p) 
  (h2 : perpendicular_line_plane l2 p) : 
  parallel_line l1 l2 :=
sorry

-- Theorem for proposition ① (to be proven false)
theorem lines_perp_to_line_not_always_parallel 
  (l : Line) (l1 l2 : Line) 
  (h1 : perpendicular_line_line l1 l) 
  (h2 : perpendicular_line_line l2 l) : 
  ¬(parallel_line l1 l2) :=
sorry

-- Theorem for proposition ④ (to be proven false)
theorem planes_perp_to_plane_not_always_parallel 
  (p : Plane) (p1 p2 : Plane) 
  (h1 : perpendicular_plane_plane p1 p) 
  (h2 : perpendicular_plane_plane p2 p) : 
  ¬(parallel_plane p1 p2) :=
sorry

end NUMINAMATH_CALUDE_planes_perp_to_line_are_parallel_lines_perp_to_plane_are_parallel_lines_perp_to_line_not_always_parallel_planes_perp_to_plane_not_always_parallel_l2827_282780


namespace NUMINAMATH_CALUDE_bobs_deli_cost_l2827_282752

/-- The total cost for a customer at Bob's Deli -/
def total_cost (sandwich_price soda_price : ℕ) (sandwich_quantity soda_quantity : ℕ) (discount_threshold discount_amount : ℕ) : ℕ :=
  let initial_total := sandwich_price * sandwich_quantity + soda_price * soda_quantity
  if initial_total > discount_threshold then
    initial_total - discount_amount
  else
    initial_total

/-- The theorem stating that the customer will pay $55 in total -/
theorem bobs_deli_cost : total_cost 5 3 7 10 50 10 = 55 := by
  sorry

end NUMINAMATH_CALUDE_bobs_deli_cost_l2827_282752


namespace NUMINAMATH_CALUDE_three_distinct_roots_reciprocal_l2827_282786

theorem three_distinct_roots_reciprocal (a b c : ℝ) :
  (∃ x y z : ℝ, x ≠ y ∧ y ≠ z ∧ x ≠ z ∧
    a * x^5 + b * x^4 + c = 0 ∧
    a * y^5 + b * y^4 + c = 0 ∧
    a * z^5 + b * z^4 + c = 0) →
  (∃ u v w : ℝ, u ≠ v ∧ v ≠ w ∧ u ≠ w ∧
    c * u^5 + b * u + a = 0 ∧
    c * v^5 + b * v + a = 0 ∧
    c * w^5 + b * w + a = 0) :=
by sorry

end NUMINAMATH_CALUDE_three_distinct_roots_reciprocal_l2827_282786


namespace NUMINAMATH_CALUDE_basin_fill_time_l2827_282723

def right_eye_rate : ℚ := 1 / 48
def left_eye_rate : ℚ := 1 / 72
def right_foot_rate : ℚ := 1 / 96
def throat_rate : ℚ := 1 / 6

def combined_rate : ℚ := right_eye_rate + left_eye_rate + right_foot_rate + throat_rate

theorem basin_fill_time :
  (1 : ℚ) / combined_rate = 288 / 61 := by sorry

end NUMINAMATH_CALUDE_basin_fill_time_l2827_282723


namespace NUMINAMATH_CALUDE_total_fruit_cost_l2827_282751

def grapes_cost : ℚ := 12.08
def cherries_cost : ℚ := 9.85

theorem total_fruit_cost : grapes_cost + cherries_cost = 21.93 := by
  sorry

end NUMINAMATH_CALUDE_total_fruit_cost_l2827_282751


namespace NUMINAMATH_CALUDE_absolute_value_inequality_solution_l2827_282769

theorem absolute_value_inequality_solution (x : ℝ) :
  (|x - 2| < 1) ↔ (1 < x ∧ x < 3) :=
sorry

end NUMINAMATH_CALUDE_absolute_value_inequality_solution_l2827_282769


namespace NUMINAMATH_CALUDE_root_preservation_l2827_282712

-- Define the polynomial p(x) = x³ - 5x + 3
def p (x : ℚ) : ℚ := x^3 - 5*x + 3

-- Define a type for polynomials with rational coefficients
def RationalPolynomial := ℚ → ℚ

-- Theorem statement
theorem root_preservation 
  (α : ℚ) 
  (f : RationalPolynomial) 
  (h1 : p α = 0) 
  (h2 : p (f α) = 0) : 
  p (f (f α)) = 0 := by
  sorry

end NUMINAMATH_CALUDE_root_preservation_l2827_282712


namespace NUMINAMATH_CALUDE_systematic_sampling_5_from_100_correct_sequence_l2827_282779

/-- Systematic sampling function that returns the nth selected individual -/
def systematicSample (totalPopulation : ℕ) (sampleSize : ℕ) (n : ℕ) : ℕ :=
  n * (totalPopulation / sampleSize)

/-- Theorem stating that systematic sampling of 5 from 100 yields the correct sequence -/
theorem systematic_sampling_5_from_100 :
  let totalPopulation : ℕ := 100
  let sampleSize : ℕ := 5
  (systematicSample totalPopulation sampleSize 1 = 20) ∧
  (systematicSample totalPopulation sampleSize 2 = 40) ∧
  (systematicSample totalPopulation sampleSize 3 = 60) ∧
  (systematicSample totalPopulation sampleSize 4 = 80) ∧
  (systematicSample totalPopulation sampleSize 5 = 100) :=
by
  sorry

/-- Theorem stating that the correct sequence is 10, 30, 50, 70, 90 -/
theorem correct_sequence :
  let totalPopulation : ℕ := 100
  let sampleSize : ℕ := 5
  (systematicSample totalPopulation sampleSize 1 - 10 = 10) ∧
  (systematicSample totalPopulation sampleSize 2 - 10 = 30) ∧
  (systematicSample totalPopulation sampleSize 3 - 10 = 50) ∧
  (systematicSample totalPopulation sampleSize 4 - 10 = 70) ∧
  (systematicSample totalPopulation sampleSize 5 - 10 = 90) :=
by
  sorry

end NUMINAMATH_CALUDE_systematic_sampling_5_from_100_correct_sequence_l2827_282779


namespace NUMINAMATH_CALUDE_parabola_intercepts_sum_l2827_282790

-- Define the parabola
def parabola (x : ℝ) : ℝ := 3 * x^2 - 9 * x + 4

-- Define the y-intercept
def y_intercept (d : ℝ) : Prop := parabola 0 = d

-- Define the x-intercepts
def x_intercepts (e f : ℝ) : Prop := parabola e = 0 ∧ parabola f = 0 ∧ e ≠ f

theorem parabola_intercepts_sum (d e f : ℝ) :
  y_intercept d → x_intercepts e f → d + e + f = 7 := by
  sorry

end NUMINAMATH_CALUDE_parabola_intercepts_sum_l2827_282790


namespace NUMINAMATH_CALUDE_max_servings_is_56_l2827_282728

/-- Represents the ingredients required for one serving of salad -/
structure ServingRequirement where
  cucumbers : ℕ
  tomatoes : ℕ
  brynza : ℕ  -- in grams
  peppers : ℕ

/-- Represents the available ingredients in the restaurant's warehouse -/
structure AvailableIngredients where
  cucumbers : ℕ
  tomatoes : ℕ
  brynza : ℕ  -- in grams
  peppers : ℕ

/-- Calculates the maximum number of servings that can be made -/
def maxServings (req : ServingRequirement) (avail : AvailableIngredients) : ℕ :=
  min (avail.cucumbers / req.cucumbers)
      (min (avail.tomatoes / req.tomatoes)
           (min (avail.brynza / req.brynza)
                (avail.peppers / req.peppers)))

/-- Theorem stating that the maximum number of servings is 56 -/
theorem max_servings_is_56 :
  let req := ServingRequirement.mk 2 2 75 1
  let avail := AvailableIngredients.mk 117 116 4200 60
  maxServings req avail = 56 := by sorry

end NUMINAMATH_CALUDE_max_servings_is_56_l2827_282728


namespace NUMINAMATH_CALUDE_last_four_digits_of_5_pow_2016_l2827_282739

def last_four_digits (n : ℕ) : ℕ := n % 10000

def power_five_pattern : List ℕ := [3125, 5625, 8125, 0625]

theorem last_four_digits_of_5_pow_2016 :
  last_four_digits (5^2016) = 0625 :=
sorry

end NUMINAMATH_CALUDE_last_four_digits_of_5_pow_2016_l2827_282739


namespace NUMINAMATH_CALUDE_part_one_part_two_l2827_282705

-- Define sets A and B
def A : Set ℝ := {x | x^2 - 2*x - 3 ≤ 0}
def B (a : ℝ) : Set ℝ := {x | a - 1 ≤ x ∧ x ≤ 2*a + 1}

-- Part 1
theorem part_one :
  A ∪ B 1 = Set.Icc (-1) 3 ∧
  (Set.univ \ B 1) = {x | x < 0 ∨ x > 3} :=
sorry

-- Part 2
theorem part_two (a : ℝ) :
  (Set.univ \ A) ∩ B a = ∅ ↔ (0 ≤ a ∧ a ≤ 1) ∨ a < -2 :=
sorry

end NUMINAMATH_CALUDE_part_one_part_two_l2827_282705


namespace NUMINAMATH_CALUDE_pentagon_cannot_tessellate_l2827_282784

/-- A regular polygon can tessellate a plane if its internal angle divides 360° evenly -/
def can_tessellate (internal_angle : ℝ) : Prop :=
  ∃ n : ℕ, n * internal_angle = 360

/-- The internal angle of a regular pentagon is 108° -/
def pentagon_internal_angle : ℝ := 108

/-- Theorem: A regular pentagon cannot tessellate a plane by itself -/
theorem pentagon_cannot_tessellate :
  ¬(can_tessellate pentagon_internal_angle) :=
sorry

end NUMINAMATH_CALUDE_pentagon_cannot_tessellate_l2827_282784


namespace NUMINAMATH_CALUDE_julie_reading_ratio_l2827_282722

theorem julie_reading_ratio : 
  ∀ (total_pages pages_yesterday pages_tomorrow : ℕ) (pages_today : ℕ),
    total_pages = 120 →
    pages_yesterday = 12 →
    pages_tomorrow = 42 →
    2 * pages_tomorrow = total_pages - pages_yesterday - pages_today →
    pages_today / pages_yesterday = 2 := by
  sorry

end NUMINAMATH_CALUDE_julie_reading_ratio_l2827_282722


namespace NUMINAMATH_CALUDE_f_decreasing_and_k_maximum_l2827_282798

noncomputable def f (x : ℝ) : ℝ := (1 + Real.log (x + 1)) / x

theorem f_decreasing_and_k_maximum :
  (∀ x > 0, (deriv f) x < 0) ∧
  (∀ x > 0, f x > 3 / (x + 1)) ∧
  (¬ ∃ k : ℕ, k > 3 ∧ ∀ x > 0, f x > (k : ℝ) / (x + 1)) :=
by sorry

end NUMINAMATH_CALUDE_f_decreasing_and_k_maximum_l2827_282798


namespace NUMINAMATH_CALUDE_clock_angle_at_8_30_l2827_282732

/-- The angle between the hour and minute hands at 8:30 on a standard 12-hour clock -/
def clock_angle : ℝ :=
  let numbers_on_clock : ℕ := 12
  let angle_between_numbers : ℝ := 30
  let hour_hand_position : ℝ := 8.5  -- Between 8 and 9
  let minute_hand_position : ℝ := 6
  angle_between_numbers * (minute_hand_position - hour_hand_position)

theorem clock_angle_at_8_30 : clock_angle = 75 := by
  sorry

end NUMINAMATH_CALUDE_clock_angle_at_8_30_l2827_282732


namespace NUMINAMATH_CALUDE_inequality_proof_l2827_282725

theorem inequality_proof (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 4) :
  1 / (a^2 + b^2) ≤ 1/8 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2827_282725


namespace NUMINAMATH_CALUDE_cell_growth_after_12_days_l2827_282759

/-- The number of cells after a given number of periods, where each cell triples every period. -/
def cell_count (initial_cells : ℕ) (periods : ℕ) : ℕ :=
  initial_cells * 3^periods

/-- The problem statement -/
theorem cell_growth_after_12_days :
  let initial_cells := 5
  let days := 12
  let period := 3
  let periods := days / period
  cell_count initial_cells periods = 135 := by
  sorry

end NUMINAMATH_CALUDE_cell_growth_after_12_days_l2827_282759


namespace NUMINAMATH_CALUDE_E_equality_condition_l2827_282797

/-- Definition of the function E --/
def E (a b c : ℚ) : ℚ := a * b^2 + b * c + c

/-- Theorem stating the equality condition for E(a,3,2) and E(a,5,3) --/
theorem E_equality_condition :
  ∀ a : ℚ, E a 3 2 = E a 5 3 ↔ a = -5/8 := by sorry

end NUMINAMATH_CALUDE_E_equality_condition_l2827_282797


namespace NUMINAMATH_CALUDE_gcd_power_two_minus_one_l2827_282776

theorem gcd_power_two_minus_one :
  Nat.gcd (2^1510 - 1) (2^1500 - 1) = 2^10 - 1 := by
  sorry

end NUMINAMATH_CALUDE_gcd_power_two_minus_one_l2827_282776


namespace NUMINAMATH_CALUDE_conic_eccentricity_l2827_282706

-- Define the geometric sequence
def is_geometric_sequence (a : ℝ) : Prop := a * a = 81

-- Define the conic section
def conic_section (a : ℝ) (x y : ℝ) : Prop := x^2 + y^2 / a = 1

-- Define the eccentricity
def eccentricity (e : ℝ) (a : ℝ) : Prop :=
  (e = Real.sqrt 10 ∧ a = -9) ∨ (e = 2 * Real.sqrt 2 / 3 ∧ a = 9)

-- Theorem statement
theorem conic_eccentricity (a : ℝ) (e : ℝ) :
  is_geometric_sequence a →
  (∃ x y, conic_section a x y) →
  eccentricity e a :=
sorry

end NUMINAMATH_CALUDE_conic_eccentricity_l2827_282706


namespace NUMINAMATH_CALUDE_hulk_seventh_jump_exceeds_1500_l2827_282750

def hulk_jump (n : ℕ) : ℝ :=
  3 * (3 ^ (n - 1))

theorem hulk_seventh_jump_exceeds_1500 :
  (∀ k < 7, hulk_jump k ≤ 1500) ∧ hulk_jump 7 > 1500 :=
sorry

end NUMINAMATH_CALUDE_hulk_seventh_jump_exceeds_1500_l2827_282750


namespace NUMINAMATH_CALUDE_tangent_line_x_intercept_l2827_282773

noncomputable def f (x : ℝ) : ℝ := Real.exp x

theorem tangent_line_x_intercept :
  let point : ℝ × ℝ := (2, Real.exp 2)
  let slope : ℝ := (deriv f) point.1
  let tangent_line (x : ℝ) : ℝ := slope * (x - point.1) + point.2
  (tangent_line 1 = 0) ∧ (∀ x : ℝ, x ≠ 1 → tangent_line x ≠ 0) :=
by
  sorry

#check tangent_line_x_intercept

end NUMINAMATH_CALUDE_tangent_line_x_intercept_l2827_282773


namespace NUMINAMATH_CALUDE_arithmetic_sequence_fourth_term_l2827_282716

/-- 
Given an arithmetic sequence {a_n} with S_n as the sum of its first n terms,
prove that if S_6 = 24 and S_9 = 63, then a_4 = 5.
-/
theorem arithmetic_sequence_fourth_term 
  (a : ℕ → ℝ) 
  (S : ℕ → ℝ) 
  (h_arith : ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)) 
  (h_sum : ∀ n, S n = (n / 2) * (2 * a 1 + (n - 1) * (a 2 - a 1))) 
  (h_S6 : S 6 = 24) 
  (h_S9 : S 9 = 63) : 
  a 4 = 5 := by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_fourth_term_l2827_282716


namespace NUMINAMATH_CALUDE_angle_measure_l2827_282733

theorem angle_measure (θ φ : ℝ) : 
  (90 - θ) = 0.4 * (180 - θ) →  -- complement is 40% of supplement
  φ = 180 - θ →                 -- θ and φ form a linear pair
  φ = 2 * θ →                   -- φ is twice the size of θ
  θ = 30 := by
sorry

end NUMINAMATH_CALUDE_angle_measure_l2827_282733


namespace NUMINAMATH_CALUDE_rover_spots_l2827_282782

theorem rover_spots (granger cisco rover : ℕ) : 
  granger = 5 * cisco →
  cisco = rover / 2 - 5 →
  granger + cisco = 108 →
  rover = 46 := by
sorry

end NUMINAMATH_CALUDE_rover_spots_l2827_282782


namespace NUMINAMATH_CALUDE_rectangle_k_value_l2827_282794

-- Define the rectangle ABCD
structure Rectangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  D : ℝ × ℝ

-- Define the properties of the rectangle
def isValidRectangle (rect : Rectangle) : Prop :=
  rect.A.1 = -3 ∧ 
  rect.A.2 = 1 ∧
  rect.B.1 = 4 ∧
  rect.D.2 = rect.A.2 + (rect.B.1 - rect.A.1)

-- Define the area of the rectangle
def rectangleArea (rect : Rectangle) : ℝ :=
  (rect.B.1 - rect.A.1) * (rect.D.2 - rect.A.2)

-- Theorem statement
theorem rectangle_k_value (rect : Rectangle) (k : ℝ) :
  isValidRectangle rect →
  rectangleArea rect = 70 →
  k > 0 →
  rect.D.2 = k →
  k = 11 := by
  sorry


end NUMINAMATH_CALUDE_rectangle_k_value_l2827_282794


namespace NUMINAMATH_CALUDE_worker_speed_ratio_l2827_282713

/-- Given two workers a and b, where a is k times as fast as b, prove that k = 3 
    under the given conditions. -/
theorem worker_speed_ratio (k : ℝ) : 
  (∃ (rate_b : ℝ), 
    (k * rate_b + rate_b = 1 / 30) ∧ 
    (k * rate_b = 1 / 40)) → 
  k = 3 := by
  sorry

end NUMINAMATH_CALUDE_worker_speed_ratio_l2827_282713


namespace NUMINAMATH_CALUDE_roots_on_circle_l2827_282743

theorem roots_on_circle (a : ℝ) : 
  (∃ (z₁ z₂ z₃ z₄ : ℂ), z₁ ≠ z₂ ∧ z₁ ≠ z₃ ∧ z₁ ≠ z₄ ∧ z₂ ≠ z₃ ∧ z₂ ≠ z₄ ∧ z₃ ≠ z₄ ∧
    (z₁^2 - 2*z₁ + 5)*(z₁^2 + 2*a*z₁ + 1) = 0 ∧
    (z₂^2 - 2*z₂ + 5)*(z₂^2 + 2*a*z₂ + 1) = 0 ∧
    (z₃^2 - 2*z₃ + 5)*(z₃^2 + 2*a*z₃ + 1) = 0 ∧
    (z₄^2 - 2*z₄ + 5)*(z₄^2 + 2*a*z₄ + 1) = 0 ∧
    ∃ (c : ℂ) (r : ℝ), r > 0 ∧ 
      Complex.abs (z₁ - c) = r ∧
      Complex.abs (z₂ - c) = r ∧
      Complex.abs (z₃ - c) = r ∧
      Complex.abs (z₄ - c) = r) ↔
  (a > -1 ∧ a < 1) ∨ a = -3 :=
by sorry

end NUMINAMATH_CALUDE_roots_on_circle_l2827_282743


namespace NUMINAMATH_CALUDE_car_expense_difference_l2827_282770

/-- The difference in car expenses between Alberto and Samara -/
def expense_difference (alberto_expense : ℕ) (samara_oil : ℕ) (samara_tires : ℕ) (samara_detailing : ℕ) : ℕ :=
  alberto_expense - (samara_oil + samara_tires + samara_detailing)

/-- Theorem stating the difference in car expenses between Alberto and Samara -/
theorem car_expense_difference :
  expense_difference 2457 25 467 79 = 1886 := by
  sorry

end NUMINAMATH_CALUDE_car_expense_difference_l2827_282770


namespace NUMINAMATH_CALUDE_johns_money_l2827_282772

/-- Given three people with a total of $67, where one has $5 less than the second,
    and the third has 4 times more than the second, prove that the third person has $48. -/
theorem johns_money (total : ℕ) (alis_money nadas_money johns_money : ℕ) : 
  total = 67 →
  alis_money = nadas_money - 5 →
  johns_money = 4 * nadas_money →
  alis_money + nadas_money + johns_money = total →
  johns_money = 48 := by
  sorry

end NUMINAMATH_CALUDE_johns_money_l2827_282772


namespace NUMINAMATH_CALUDE_area_outside_inscribed_square_l2827_282763

def square_side_length : ℝ := 2

theorem area_outside_inscribed_square (square_side : ℝ) (h : square_side = square_side_length) :
  let circle_radius : ℝ := square_side * Real.sqrt 2 / 2
  let circle_area : ℝ := π * circle_radius ^ 2
  let square_area : ℝ := square_side ^ 2
  circle_area - square_area = 2 * π - 4 := by
sorry

end NUMINAMATH_CALUDE_area_outside_inscribed_square_l2827_282763


namespace NUMINAMATH_CALUDE_absolute_value_inequality_solution_set_l2827_282708

theorem absolute_value_inequality_solution_set :
  {x : ℝ | |x - 500| ≤ 5} = {x : ℝ | 495 ≤ x ∧ x ≤ 505} := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_inequality_solution_set_l2827_282708


namespace NUMINAMATH_CALUDE_square_sum_from_difference_and_sum_squares_l2827_282703

theorem square_sum_from_difference_and_sum_squares 
  (m n : ℝ) 
  (h1 : (m - n)^2 = 8) 
  (h2 : (m + n)^2 = 2) : 
  m^2 + n^2 = 5 := by
sorry

end NUMINAMATH_CALUDE_square_sum_from_difference_and_sum_squares_l2827_282703


namespace NUMINAMATH_CALUDE_cost_per_set_is_20_verify_profit_equation_l2827_282720

/-- Represents the manufacturing and sales scenario of horseshoe sets -/
structure HorseshoeManufacturing where
  initialOutlay : ℝ
  sellingPrice : ℝ
  setsSold : ℕ
  profit : ℝ
  costPerSet : ℝ

/-- The cost per set is $20 given the specified conditions -/
theorem cost_per_set_is_20 (h : HorseshoeManufacturing) 
    (h_initialOutlay : h.initialOutlay = 10000)
    (h_sellingPrice : h.sellingPrice = 50)
    (h_setsSold : h.setsSold = 500)
    (h_profit : h.profit = 5000) :
    h.costPerSet = 20 := by
  sorry

/-- Verifies that the calculated cost per set satisfies the profit equation -/
theorem verify_profit_equation (h : HorseshoeManufacturing) 
    (h_initialOutlay : h.initialOutlay = 10000)
    (h_sellingPrice : h.sellingPrice = 50)
    (h_setsSold : h.setsSold = 500)
    (h_profit : h.profit = 5000)
    (h_costPerSet : h.costPerSet = 20) :
    h.profit = h.sellingPrice * h.setsSold - (h.initialOutlay + h.costPerSet * h.setsSold) := by
  sorry

end NUMINAMATH_CALUDE_cost_per_set_is_20_verify_profit_equation_l2827_282720


namespace NUMINAMATH_CALUDE_total_charcoal_needed_l2827_282768

-- Define the ratios and water amounts for each batch
def batch1_ratio : ℚ := 2 / 30
def batch1_water : ℚ := 900

def batch2_ratio : ℚ := 3 / 50
def batch2_water : ℚ := 1150

def batch3_ratio : ℚ := 4 / 80
def batch3_water : ℚ := 1615

def batch4_ratio : ℚ := 2.3 / 25
def batch4_water : ℚ := 675

def batch5_ratio : ℚ := 5.5 / 115
def batch5_water : ℚ := 1930

-- Function to calculate charcoal needed for a batch
def charcoal_needed (ratio : ℚ) (water : ℚ) : ℚ :=
  ratio * water

-- Theorem stating the total charcoal needed is 363.28 grams
theorem total_charcoal_needed :
  (charcoal_needed batch1_ratio batch1_water +
   charcoal_needed batch2_ratio batch2_water +
   charcoal_needed batch3_ratio batch3_water +
   charcoal_needed batch4_ratio batch4_water +
   charcoal_needed batch5_ratio batch5_water) = 363.28 := by
  sorry

end NUMINAMATH_CALUDE_total_charcoal_needed_l2827_282768


namespace NUMINAMATH_CALUDE_percentage_increase_l2827_282718

theorem percentage_increase (original : ℝ) (new : ℝ) : 
  original = 80 → new = 96 → (new - original) / original * 100 = 20 := by
  sorry

end NUMINAMATH_CALUDE_percentage_increase_l2827_282718


namespace NUMINAMATH_CALUDE_zinc_weight_in_mixture_l2827_282766

/-- Given a mixture of zinc and copper in the ratio 9:11 with a total weight of 78 kg,
    the weight of zinc in the mixture is 35.1 kg. -/
theorem zinc_weight_in_mixture (zinc_ratio : ℚ) (copper_ratio : ℚ) (total_weight : ℚ) :
  zinc_ratio = 9 →
  copper_ratio = 11 →
  total_weight = 78 →
  (zinc_ratio / (zinc_ratio + copper_ratio)) * total_weight = 35.1 := by
  sorry

#check zinc_weight_in_mixture

end NUMINAMATH_CALUDE_zinc_weight_in_mixture_l2827_282766


namespace NUMINAMATH_CALUDE_converse_proposition_l2827_282745

theorem converse_proposition :
  let P : Prop := x ≥ 2 ∧ y ≥ 3
  let Q : Prop := x + y ≥ 5
  let original : Prop := P → Q
  let converse : Prop := Q → P
  converse = (x + y ≥ 5 → x ≥ 2 ∧ y ≥ 3) := by
sorry

end NUMINAMATH_CALUDE_converse_proposition_l2827_282745


namespace NUMINAMATH_CALUDE_difference_of_squares_65_35_l2827_282749

theorem difference_of_squares_65_35 : 65^2 - 35^2 = 3000 := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_65_35_l2827_282749


namespace NUMINAMATH_CALUDE_park_route_length_l2827_282760

/-- A bike route in a park -/
structure BikeRoute where
  horizontal_segments : List Float
  vertical_segments : List Float

/-- The total length of a bike route -/
def total_length (route : BikeRoute) : Float :=
  2 * (route.horizontal_segments.sum + route.vertical_segments.sum)

/-- The specific bike route described in the problem -/
def park_route : BikeRoute :=
  { horizontal_segments := [4, 7, 2],
    vertical_segments := [6, 7] }

theorem park_route_length :
  total_length park_route = 52 := by
  sorry

#eval total_length park_route

end NUMINAMATH_CALUDE_park_route_length_l2827_282760


namespace NUMINAMATH_CALUDE_landscape_length_is_240_l2827_282774

/-- Represents a rectangular landscape with a playground -/
structure Landscape where
  breadth : ℝ
  length : ℝ
  playgroundArea : ℝ
  totalArea : ℝ

/-- The length of the landscape is 8 times its breadth -/
def lengthIsTotalRule (l : Landscape) : Prop :=
  l.length = 8 * l.breadth

/-- The playground occupies 1/6 of the total landscape area -/
def playgroundRule (l : Landscape) : Prop :=
  l.playgroundArea = l.totalArea / 6

/-- The playground has an area of 1200 square meters -/
def playgroundAreaRule (l : Landscape) : Prop :=
  l.playgroundArea = 1200

/-- The total area of the landscape is the product of its length and breadth -/
def totalAreaRule (l : Landscape) : Prop :=
  l.totalArea = l.length * l.breadth

/-- Theorem: Given the conditions, the length of the landscape is 240 meters -/
theorem landscape_length_is_240 (l : Landscape) 
  (h1 : lengthIsTotalRule l) 
  (h2 : playgroundRule l) 
  (h3 : playgroundAreaRule l) 
  (h4 : totalAreaRule l) : 
  l.length = 240 := by
  sorry

end NUMINAMATH_CALUDE_landscape_length_is_240_l2827_282774


namespace NUMINAMATH_CALUDE_simplify_trig_expression_l2827_282715

theorem simplify_trig_expression (a : Real) (h : 0 < a ∧ a < π / 2) :
  Real.sqrt (1 + Real.sin a) + Real.sqrt (1 - Real.sin a) - Real.sqrt (2 + 2 * Real.cos a) = 0 := by
  sorry

end NUMINAMATH_CALUDE_simplify_trig_expression_l2827_282715


namespace NUMINAMATH_CALUDE_base_four_representation_of_256_l2827_282753

theorem base_four_representation_of_256 :
  (256 : ℕ).digits 4 = [0, 0, 0, 0, 1] :=
sorry

end NUMINAMATH_CALUDE_base_four_representation_of_256_l2827_282753


namespace NUMINAMATH_CALUDE_whitney_fish_books_l2827_282714

/-- The number of books about fish Whitney bought -/
def fish_books : ℕ := 7

/-- The number of books about whales Whitney bought -/
def whale_books : ℕ := 9

/-- The number of magazines Whitney bought -/
def magazines : ℕ := 3

/-- The cost of each book in dollars -/
def book_cost : ℕ := 11

/-- The cost of each magazine in dollars -/
def magazine_cost : ℕ := 1

/-- The total amount Whitney spent in dollars -/
def total_spent : ℕ := 179

theorem whitney_fish_books :
  whale_books * book_cost + fish_books * book_cost + magazines * magazine_cost = total_spent :=
sorry

end NUMINAMATH_CALUDE_whitney_fish_books_l2827_282714


namespace NUMINAMATH_CALUDE_boat_speed_in_still_water_l2827_282742

/-- 
Given a boat that travels at different speeds with and against a stream,
this theorem proves that its speed in still water is 6 km/hr.
-/
theorem boat_speed_in_still_water 
  (speed_with_stream : ℝ) 
  (speed_against_stream : ℝ) 
  (h1 : speed_with_stream = 7) 
  (h2 : speed_against_stream = 5) : 
  (speed_with_stream + speed_against_stream) / 2 = 6 := by
sorry


end NUMINAMATH_CALUDE_boat_speed_in_still_water_l2827_282742


namespace NUMINAMATH_CALUDE_six_chairs_three_people_l2827_282717

/-- The number of ways to arrange n people among m chairs in a row, with no two people adjacent -/
def nonadjacentArrangements (m n : ℕ) : ℕ :=
  if m ≤ n then 0
  else Nat.descFactorial (m - n + 1) n

theorem six_chairs_three_people :
  nonadjacentArrangements 6 3 = 24 := by
  sorry

end NUMINAMATH_CALUDE_six_chairs_three_people_l2827_282717


namespace NUMINAMATH_CALUDE_expression_simplification_l2827_282796

theorem expression_simplification (x y : ℝ) (h : x = -3) :
  x * (x - 4) * (x + 4) - (x + 3) * (x^2 - 6*x + 9) + 5*x^3*y^2 / (x^2*y^2) = -66 :=
by sorry

end NUMINAMATH_CALUDE_expression_simplification_l2827_282796


namespace NUMINAMATH_CALUDE_linear_function_composition_l2827_282781

def is_linear (f : ℝ → ℝ) : Prop := ∃ a b : ℝ, a ≠ 0 ∧ ∀ x, f x = a * x + b

theorem linear_function_composition (f : ℝ → ℝ) :
  is_linear f → (∀ x, f (f x) = 9 * x + 4) → 
  (∀ x, f x = 3 * x + 1) ∨ (∀ x, f x = -3 * x - 2) :=
by sorry

end NUMINAMATH_CALUDE_linear_function_composition_l2827_282781


namespace NUMINAMATH_CALUDE_abs_eq_neg_implies_nonpositive_l2827_282791

theorem abs_eq_neg_implies_nonpositive (a : ℝ) : |a| = -a → a ≤ 0 := by
  sorry

end NUMINAMATH_CALUDE_abs_eq_neg_implies_nonpositive_l2827_282791


namespace NUMINAMATH_CALUDE_sequence_sum_2000_l2827_282702

def sequence_sum (n : ℕ) : ℤ :=
  let group_sum := -1
  let num_groups := n / 6
  num_groups * group_sum

theorem sequence_sum_2000 :
  sequence_sum 2000 = -334 :=
sorry

end NUMINAMATH_CALUDE_sequence_sum_2000_l2827_282702


namespace NUMINAMATH_CALUDE_unique_solution_l2827_282787

theorem unique_solution (a b c : ℝ) : 
  a > 2 ∧ b > 2 ∧ c > 2 →
  (a + 3)^2 / (b + c - 3) + (b + 5)^2 / (c + a - 5) + (c + 7)^2 / (a + b - 7) = 48 →
  a = 7 ∧ b = 5 ∧ c = 3 :=
by sorry

end NUMINAMATH_CALUDE_unique_solution_l2827_282787


namespace NUMINAMATH_CALUDE_sqrt_a_squared_plus_one_is_quadratic_radical_l2827_282777

/-- A function is a quadratic radical if it can be expressed as the square root of a non-negative real-valued expression. -/
def is_quadratic_radical (f : ℝ → ℝ) : Prop :=
  ∃ g : ℝ → ℝ, (∀ x, g x ≥ 0) ∧ (∀ x, f x = Real.sqrt (g x))

/-- The function f(a) = √(a² + 1) is a quadratic radical. -/
theorem sqrt_a_squared_plus_one_is_quadratic_radical :
  is_quadratic_radical (fun a => Real.sqrt (a^2 + 1)) :=
sorry

end NUMINAMATH_CALUDE_sqrt_a_squared_plus_one_is_quadratic_radical_l2827_282777


namespace NUMINAMATH_CALUDE_problem_statement_l2827_282724

theorem problem_statement (m n : ℝ) (h1 : m ≠ n) (h2 : m^2 = n + 2) (h3 : n^2 = m + 2) :
  4 * m * n - m^3 - n^3 = 0 := by
sorry

end NUMINAMATH_CALUDE_problem_statement_l2827_282724


namespace NUMINAMATH_CALUDE_polynomial_remainder_l2827_282775

def polynomial (x : ℝ) : ℝ := 3*x^8 - x^7 - 7*x^5 + 3*x^3 + 4*x^2 - 12*x - 1

def divisor (x : ℝ) : ℝ := 3*x - 9

theorem polynomial_remainder :
  ∃ (q : ℝ → ℝ), ∀ (x : ℝ),
    polynomial x = (divisor x) * (q x) + 15951 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_remainder_l2827_282775
