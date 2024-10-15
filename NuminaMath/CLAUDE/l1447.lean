import Mathlib

namespace NUMINAMATH_CALUDE_southbound_cyclist_speed_l1447_144709

/-- 
Given two cyclists starting from the same point and traveling in opposite directions,
with one cyclist traveling north at 10 km/h, prove that the speed of the southbound
cyclist is 15 km/h if they are 50 km apart after 2 hours.
-/
theorem southbound_cyclist_speed 
  (north_speed : ℝ) 
  (time : ℝ) 
  (distance : ℝ) 
  (h1 : north_speed = 10) 
  (h2 : time = 2) 
  (h3 : distance = 50) : 
  ∃ south_speed : ℝ, south_speed = 15 ∧ (north_speed + south_speed) * time = distance :=
sorry

end NUMINAMATH_CALUDE_southbound_cyclist_speed_l1447_144709


namespace NUMINAMATH_CALUDE_special_quadrilateral_not_necessarily_square_l1447_144775

/-- A quadrilateral with perpendicular and bisecting diagonals -/
structure SpecialQuadrilateral where
  /-- The quadrilateral has perpendicular diagonals -/
  perpendicular_diagonals : Bool
  /-- The quadrilateral has bisecting diagonals -/
  bisecting_diagonals : Bool

/-- Definition of a square -/
structure Square where
  /-- All sides of the square are equal -/
  equal_sides : Bool
  /-- All angles of the square are right angles -/
  right_angles : Bool

/-- Theorem stating that a quadrilateral with perpendicular and bisecting diagonals is not necessarily a square -/
theorem special_quadrilateral_not_necessarily_square :
  ∃ (q : SpecialQuadrilateral), q.perpendicular_diagonals ∧ q.bisecting_diagonals ∧
  ∃ (s : Square), ¬(q.perpendicular_diagonals → s.equal_sides ∧ s.right_angles) :=
sorry

end NUMINAMATH_CALUDE_special_quadrilateral_not_necessarily_square_l1447_144775


namespace NUMINAMATH_CALUDE_num_persons_is_nine_l1447_144732

/-- The number of persons who went to the hotel -/
def num_persons : ℕ := 9

/-- The amount spent by each of the first 8 persons -/
def amount_per_person : ℕ := 12

/-- The additional amount spent by the 9th person above the average -/
def additional_amount : ℕ := 8

/-- The total expenditure of all persons -/
def total_expenditure : ℕ := 117

/-- Theorem stating that the number of persons who went to the hotel is 9 -/
theorem num_persons_is_nine :
  (num_persons - 1) * amount_per_person + 
  ((num_persons - 1) * amount_per_person + additional_amount) / num_persons + additional_amount = 
  total_expenditure :=
sorry

end NUMINAMATH_CALUDE_num_persons_is_nine_l1447_144732


namespace NUMINAMATH_CALUDE_x4_coefficient_zero_l1447_144779

theorem x4_coefficient_zero (a : ℝ) : 
  (∃ f : ℝ → ℝ, ∀ x, (x^2 + a*x + 1) * (-6*x^3) = -6*x^5 + f x * x^4 + -6*x^3) ↔ a = 0 := by
  sorry

end NUMINAMATH_CALUDE_x4_coefficient_zero_l1447_144779


namespace NUMINAMATH_CALUDE_pizza_cost_is_seven_l1447_144790

def pizza_problem (box_cost : ℚ) : Prop :=
  let num_boxes : ℕ := 5
  let tip_ratio : ℚ := 1 / 7
  let total_paid : ℚ := 40
  let pizza_cost : ℚ := box_cost * num_boxes
  let tip : ℚ := pizza_cost * tip_ratio
  pizza_cost + tip = total_paid

theorem pizza_cost_is_seven :
  ∃ (box_cost : ℚ), pizza_problem box_cost ∧ box_cost = 7 :=
by sorry

end NUMINAMATH_CALUDE_pizza_cost_is_seven_l1447_144790


namespace NUMINAMATH_CALUDE_bicycle_trip_time_l1447_144755

theorem bicycle_trip_time (distance : Real) (outbound_speed return_speed : Real) :
  distance = 28.8 ∧ outbound_speed = 16 ∧ return_speed = 24 →
  distance / outbound_speed + distance / return_speed = 3 := by
  sorry

end NUMINAMATH_CALUDE_bicycle_trip_time_l1447_144755


namespace NUMINAMATH_CALUDE_quadratic_condition_l1447_144799

/-- Represents a quadratic equation in the form ax^2 + bx + c = 0 --/
structure QuadraticEquation where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Checks if a given equation is quadratic --/
def isQuadratic (eq : QuadraticEquation) : Prop :=
  eq.a ≠ 0

/-- The equation mx^2 + 3x - 4 = 3x^2 rearranged to standard form --/
def equationOfInterest (m : ℝ) : QuadraticEquation :=
  ⟨m - 3, 3, -4⟩

/-- Theorem stating that for the equation to be quadratic, m must not equal 3 --/
theorem quadratic_condition (m : ℝ) :
  isQuadratic (equationOfInterest m) ↔ m ≠ 3 := by sorry

end NUMINAMATH_CALUDE_quadratic_condition_l1447_144799


namespace NUMINAMATH_CALUDE_cost_of_traveling_specific_roads_l1447_144782

/-- Calculates the cost of traveling two intersecting roads on a rectangular lawn. -/
def cost_of_traveling_roads (lawn_length lawn_width road_width cost_per_sqm : ℕ) : ℕ :=
  let road1_area := road_width * lawn_width
  let road2_area := road_width * lawn_length
  let intersection_area := road_width * road_width
  let total_road_area := road1_area + road2_area - intersection_area
  total_road_area * cost_per_sqm

/-- Proves that the cost of traveling two intersecting roads on a specific rectangular lawn is 6500. -/
theorem cost_of_traveling_specific_roads :
  cost_of_traveling_roads 80 60 10 5 = 6500 := by
  sorry

end NUMINAMATH_CALUDE_cost_of_traveling_specific_roads_l1447_144782


namespace NUMINAMATH_CALUDE_mappings_count_l1447_144729

/-- Set A with elements from 1 to 15 -/
def A : Finset ℕ := Finset.range 15

/-- Set B with elements 0 and 1 -/
def B : Finset ℕ := {0, 1}

/-- The number of mappings from A to B where 1 is the image of at least two elements of A -/
def num_mappings : ℕ := 2^15 - (1 + 15)

/-- Theorem stating that the number of mappings from A to B where 1 is the image of at least two elements of A is 32752 -/
theorem mappings_count : num_mappings = 32752 := by
  sorry

#eval num_mappings

end NUMINAMATH_CALUDE_mappings_count_l1447_144729


namespace NUMINAMATH_CALUDE_square_brush_ratio_l1447_144765

theorem square_brush_ratio (s w : ℝ) (h : s > 0) (h' : w > 0) : 
  w^2 + ((s - w)^2) / 2 = s^2 / 3 → s / w = 3 := by
  sorry

end NUMINAMATH_CALUDE_square_brush_ratio_l1447_144765


namespace NUMINAMATH_CALUDE_problem_solution_l1447_144753

-- Define the function f
def f (x m : ℝ) : ℝ := |x + m| + 2 * |x - 1|

-- Define the function g
def g (x m : ℝ) : ℝ := |x + 1 + m| + 2 * |x|

theorem problem_solution :
  (∀ x : ℝ, m > 0 → 
    (m = 1 → (f x m ≤ 10 ↔ -3 ≤ x ∧ x ≤ 11/3))) ∧
  (∀ m : ℝ, (∀ x : ℝ, g x m ≥ 3) ↔ m ≥ 2) :=
sorry

end NUMINAMATH_CALUDE_problem_solution_l1447_144753


namespace NUMINAMATH_CALUDE_original_stations_count_l1447_144725

def number_of_ticket_types (k : ℕ) : ℕ := k * (k - 1) / 2

theorem original_stations_count 
  (m n : ℕ) 
  (h1 : n > 1) 
  (h2 : number_of_ticket_types (m + n) - number_of_ticket_types m = 58) : 
  m = 14 := by
sorry

end NUMINAMATH_CALUDE_original_stations_count_l1447_144725


namespace NUMINAMATH_CALUDE_sqrt_2_irrational_l1447_144727

theorem sqrt_2_irrational : Irrational (Real.sqrt 2) := by
  sorry

end NUMINAMATH_CALUDE_sqrt_2_irrational_l1447_144727


namespace NUMINAMATH_CALUDE_songs_in_playlists_l1447_144762

theorem songs_in_playlists (n : ℕ) :
  ∃ (k : ℕ), n = 12 + 9 * k ↔ ∃ (m : ℕ), n = 9 * m + 3 :=
by sorry

end NUMINAMATH_CALUDE_songs_in_playlists_l1447_144762


namespace NUMINAMATH_CALUDE_quadratic_equation_coefficients_l1447_144743

theorem quadratic_equation_coefficients :
  ∀ (a b c : ℝ), 
    (∀ x, 3 * x^2 = 2 * x - 3 ↔ a * x^2 + b * x + c = 0) →
    a = 3 ∧ b = -2 ∧ c = 3 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_coefficients_l1447_144743


namespace NUMINAMATH_CALUDE_coin_tosses_properties_l1447_144715

/-- Two independent coin tosses where A is "first coin is heads" and B is "second coin is tails" -/
structure CoinTosses where
  /-- Probability of event A (first coin is heads) -/
  prob_A : ℝ
  /-- Probability of event B (second coin is tails) -/
  prob_B : ℝ
  /-- A and B are independent events -/
  independent : Prop
  /-- Both coins are fair -/
  fair_coins : prob_A = 1/2 ∧ prob_B = 1/2

/-- Properties of the coin tosses -/
theorem coin_tosses_properties (ct : CoinTosses) :
  ct.independent ∧ 
  (1 - (1 - ct.prob_A) * (1 - ct.prob_B) = 3/4) ∧
  ct.prob_A = ct.prob_B :=
sorry

end NUMINAMATH_CALUDE_coin_tosses_properties_l1447_144715


namespace NUMINAMATH_CALUDE_corner_square_probability_l1447_144718

-- Define the grid size
def gridSize : Nat := 4

-- Define the number of squares to be selected
def squaresSelected : Nat := 3

-- Define the number of corner squares
def cornerSquares : Nat := 4

-- Define the total number of squares
def totalSquares : Nat := gridSize * gridSize

-- Define the probability of selecting at least one corner square
def probabilityAtLeastOneCorner : Rat := 17 / 28

theorem corner_square_probability :
  (1 : Rat) - (Nat.choose (totalSquares - cornerSquares) squaresSelected : Rat) / 
  (Nat.choose totalSquares squaresSelected) = probabilityAtLeastOneCorner := by
  sorry

end NUMINAMATH_CALUDE_corner_square_probability_l1447_144718


namespace NUMINAMATH_CALUDE_max_operation_value_l1447_144711

/-- Represents a digit (0-9) -/
def Digit := Fin 10

/-- The operation to be maximized -/
def operation (X Y Z : Digit) : ℕ := 99 * X.val + 9 * Y.val - 9 * Z.val

/-- The theorem statement -/
theorem max_operation_value :
  ∃ (X Y Z : Digit), X ≠ Y ∧ Y ≠ Z ∧ X ≠ Z ∧
    operation X Y Z = 900 ∧
    ∀ (A B C : Digit), A ≠ B → B ≠ C → A ≠ C →
      operation A B C ≤ 900 :=
sorry

end NUMINAMATH_CALUDE_max_operation_value_l1447_144711


namespace NUMINAMATH_CALUDE_inscribed_quadrilateral_incenters_form_rectangle_l1447_144720

/-- A point in the Euclidean plane -/
structure Point :=
  (x : ℝ) (y : ℝ)

/-- A circle in the Euclidean plane -/
structure Circle :=
  (center : Point) (radius : ℝ)

/-- A quadrilateral in the Euclidean plane -/
structure Quadrilateral :=
  (A B C D : Point)

/-- The incenter of a triangle -/
def incenter (A B C : Point) : Point := sorry

/-- Predicate to check if a quadrilateral is inscribed in a circle -/
def is_inscribed (q : Quadrilateral) (c : Circle) : Prop := sorry

/-- Predicate to check if a quadrilateral is a rectangle -/
def is_rectangle (q : Quadrilateral) : Prop := sorry

theorem inscribed_quadrilateral_incenters_form_rectangle 
  (ABCD : Quadrilateral) (c : Circle) :
  is_inscribed ABCD c →
  let I_A := incenter ABCD.B ABCD.C ABCD.D
  let I_B := incenter ABCD.C ABCD.D ABCD.A
  let I_C := incenter ABCD.D ABCD.A ABCD.B
  let I_D := incenter ABCD.A ABCD.B ABCD.C
  is_rectangle (Quadrilateral.mk I_A I_B I_C I_D) :=
sorry

end NUMINAMATH_CALUDE_inscribed_quadrilateral_incenters_form_rectangle_l1447_144720


namespace NUMINAMATH_CALUDE_weight_increase_percentage_l1447_144766

/-- The percentage increase in total weight of two people given their initial weight ratio and individual weight increases -/
theorem weight_increase_percentage 
  (ram_ratio : ℝ) 
  (shyam_ratio : ℝ) 
  (ram_increase : ℝ) 
  (shyam_increase : ℝ) 
  (new_total_weight : ℝ) 
  (h1 : ram_ratio = 2) 
  (h2 : shyam_ratio = 5) 
  (h3 : ram_increase = 0.1) 
  (h4 : shyam_increase = 0.17) 
  (h5 : new_total_weight = 82.8) : 
  ∃ (percentage_increase : ℝ), 
    abs (percentage_increase - 15.06) < 0.01 ∧ 
    percentage_increase = 
      (new_total_weight - (ram_ratio + shyam_ratio) * 
        (new_total_weight / (ram_ratio * (1 + ram_increase) + shyam_ratio * (1 + shyam_increase)))) / 
      ((ram_ratio + shyam_ratio) * 
        (new_total_weight / (ram_ratio * (1 + ram_increase) + shyam_ratio * (1 + shyam_increase)))) 
      * 100 := by
  sorry


end NUMINAMATH_CALUDE_weight_increase_percentage_l1447_144766


namespace NUMINAMATH_CALUDE_fruit_cost_difference_l1447_144776

theorem fruit_cost_difference : 
  let grapes_kg : ℝ := 7
  let grapes_price : ℝ := 70
  let grapes_discount : ℝ := 0.10
  let grapes_tax : ℝ := 0.05

  let mangoes_kg : ℝ := 9
  let mangoes_price : ℝ := 55
  let mangoes_discount : ℝ := 0.05
  let mangoes_tax : ℝ := 0.07

  let apples_kg : ℝ := 5
  let apples_price : ℝ := 40
  let apples_discount : ℝ := 0.08
  let apples_tax : ℝ := 0.03

  let oranges_kg : ℝ := 3
  let oranges_price : ℝ := 30
  let oranges_discount : ℝ := 0.15
  let oranges_tax : ℝ := 0.06

  let mangoes_cost := mangoes_kg * mangoes_price * (1 - mangoes_discount) * (1 + mangoes_tax)
  let apples_cost := apples_kg * apples_price * (1 - apples_discount) * (1 + apples_tax)

  mangoes_cost - apples_cost = 313.6475 := by sorry

end NUMINAMATH_CALUDE_fruit_cost_difference_l1447_144776


namespace NUMINAMATH_CALUDE_complex_number_theorem_l1447_144717

theorem complex_number_theorem (z : ℂ) : 
  (∃ (k : ℝ), z / (1 + Complex.I) = k * Complex.I) ∧ 
  Complex.abs (z / (1 + Complex.I)) = 1 → 
  z = -1 + Complex.I ∨ z = 1 - Complex.I :=
by sorry

end NUMINAMATH_CALUDE_complex_number_theorem_l1447_144717


namespace NUMINAMATH_CALUDE_solution_set_f_shifted_empty_solution_set_l1447_144748

-- Define the function f
def f (x : ℝ) : ℝ := |x - 1| + |x - 2|

-- Theorem for part 1
theorem solution_set_f_shifted (x : ℝ) :
  f (x + 2) ≥ 2 ↔ x ≤ -3/2 ∨ x ≥ 1/2 :=
sorry

-- Theorem for part 2
theorem empty_solution_set (a : ℝ) :
  (∀ x, f x ≥ a) ↔ a ≤ 1 :=
sorry

end NUMINAMATH_CALUDE_solution_set_f_shifted_empty_solution_set_l1447_144748


namespace NUMINAMATH_CALUDE_ink_cartridge_cost_l1447_144792

/-- Given that 13 ink cartridges cost 182 dollars in total, 
    prove that the cost of one ink cartridge is 14 dollars. -/
theorem ink_cartridge_cost : ℕ → Prop :=
  fun x => (13 * x = 182) → (x = 14)

/-- Proof of the theorem -/
example : ink_cartridge_cost 14 := by
  sorry

end NUMINAMATH_CALUDE_ink_cartridge_cost_l1447_144792


namespace NUMINAMATH_CALUDE_division_multiplication_equality_l1447_144798

theorem division_multiplication_equality (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) :
  1 / (b / a) * (a / b) = a^2 / b^2 := by
  sorry

end NUMINAMATH_CALUDE_division_multiplication_equality_l1447_144798


namespace NUMINAMATH_CALUDE_correct_package_cost_l1447_144787

def packageCost (P : ℕ) : ℕ :=
  15 + 5 * (P - 1) - 8 * (if P ≥ 5 then 1 else 0)

theorem correct_package_cost (P : ℕ) (h : P ≥ 1) :
  packageCost P = 15 + 5 * (P - 1) - 8 * (if P ≥ 5 then 1 else 0) :=
by sorry

end NUMINAMATH_CALUDE_correct_package_cost_l1447_144787


namespace NUMINAMATH_CALUDE_mike_is_18_l1447_144731

-- Define Mike's age and his uncle's age
def mike_age : ℕ := sorry
def uncle_age : ℕ := sorry

-- Define the conditions
axiom age_difference : mike_age = uncle_age - 18
axiom sum_of_ages : mike_age + uncle_age = 54

-- Theorem to prove
theorem mike_is_18 : mike_age = 18 := by sorry

end NUMINAMATH_CALUDE_mike_is_18_l1447_144731


namespace NUMINAMATH_CALUDE_octagon_perimeter_96cm_l1447_144797

/-- A regular octagon is a polygon with 8 equal sides -/
structure RegularOctagon where
  side_length : ℝ
  
/-- The perimeter of a regular octagon -/
def perimeter (octagon : RegularOctagon) : ℝ :=
  8 * octagon.side_length

theorem octagon_perimeter_96cm :
  ∀ (octagon : RegularOctagon),
    octagon.side_length = 12 →
    perimeter octagon = 96 := by
  sorry

end NUMINAMATH_CALUDE_octagon_perimeter_96cm_l1447_144797


namespace NUMINAMATH_CALUDE_interior_angles_sum_plus_three_l1447_144716

/-- The sum of interior angles of a convex polygon with n sides -/
def sum_interior_angles (n : ℕ) : ℝ := 180 * (n - 2)

/-- Theorem: Given a convex polygon with n sides whose interior angles sum to 2340 degrees,
    the sum of interior angles of a convex polygon with n + 3 sides is 2880 degrees. -/
theorem interior_angles_sum_plus_three (n : ℕ) 
  (h : sum_interior_angles n = 2340) : 
  sum_interior_angles (n + 3) = 2880 := by
  sorry


end NUMINAMATH_CALUDE_interior_angles_sum_plus_three_l1447_144716


namespace NUMINAMATH_CALUDE_largest_certain_divisor_l1447_144728

/-- An eight-sided die with numbers 1 through 8 -/
def Die : Finset ℕ := Finset.range 8 

/-- The product of 7 visible numbers on the die -/
def Q (visible : Finset ℕ) : ℕ := 
  Finset.prod visible id

/-- The theorem stating that 192 is the largest number that always divides Q -/
theorem largest_certain_divisor : 
  ∀ visible : Finset ℕ, visible ⊆ Die → visible.card = 7 → 
    (∀ n : ℕ, n > 192 → ∃ visible : Finset ℕ, visible ⊆ Die ∧ visible.card = 7 ∧ ¬(n ∣ Q visible)) ∧
    (∀ visible : Finset ℕ, visible ⊆ Die → visible.card = 7 → 192 ∣ Q visible) :=
by sorry

end NUMINAMATH_CALUDE_largest_certain_divisor_l1447_144728


namespace NUMINAMATH_CALUDE_part_one_part_two_part_three_l1447_144741

-- Define the function f
def f (a b x : ℝ) : ℝ := a * x^2 + (b - 1) * x + 3

-- Part 1
theorem part_one (a b : ℝ) (ha : a ≠ 0) 
  (h_solution_set : ∀ x, f a b x > 0 ↔ -1 < x ∧ x < 3) :
  2 * a + b = -3 := by sorry

-- Part 2
theorem part_two (a b : ℝ) (ha : a ≠ 0) (hf1 : f a b 1 = 5) (hb : b > -1) :
  (∀ a' b', a' ≠ 0 → b' > -1 → f a' b' 1 = 5 → 
    1 / |a| + 4 * |a| / (b + 1) ≤ 1 / |a'| + 4 * |a'| / (b' + 1)) ∧
  1 / |a| + 4 * |a| / (b + 1) = 2 := by sorry

-- Part 3
theorem part_three (a : ℝ) (ha : a ≠ 0) :
  let b := -a - 3
  let solution_set := {x : ℝ | f a b x < -2 * x + 1}
  (a < 0 → solution_set = {x | x < 2/a ∨ x > 1}) ∧
  (0 < a ∧ a < 2 → solution_set = {x | 1 < x ∧ x < 2/a}) ∧
  (a = 2 → solution_set = ∅) ∧
  (a > 2 → solution_set = {x | 2/a < x ∧ x < 1}) := by sorry

end NUMINAMATH_CALUDE_part_one_part_two_part_three_l1447_144741


namespace NUMINAMATH_CALUDE_basketball_highlight_film_l1447_144737

theorem basketball_highlight_film (point_guard : ℕ) (shooting_guard : ℕ) (small_forward : ℕ) (power_forward : ℕ) :
  point_guard = 130 →
  shooting_guard = 145 →
  small_forward = 85 →
  power_forward = 60 →
  ∃ (center : ℕ),
    center = 180 ∧
    (point_guard + shooting_guard + small_forward + power_forward + center) / 5 = 120 :=
by sorry

end NUMINAMATH_CALUDE_basketball_highlight_film_l1447_144737


namespace NUMINAMATH_CALUDE_product_of_sum_and_sum_of_cubes_l1447_144785

theorem product_of_sum_and_sum_of_cubes (a b : ℝ) : 
  a + b = 5 → a^3 + b^3 = 35 → a * b = 6 := by
  sorry

end NUMINAMATH_CALUDE_product_of_sum_and_sum_of_cubes_l1447_144785


namespace NUMINAMATH_CALUDE_subtracted_amount_l1447_144791

theorem subtracted_amount (N : ℝ) (A : ℝ) : 
  N = 200 → 0.4 * N - A = 50 → A = 30 := by
  sorry

end NUMINAMATH_CALUDE_subtracted_amount_l1447_144791


namespace NUMINAMATH_CALUDE_balloon_count_theorem_l1447_144754

/-- Represents the number of balloons each person has --/
structure BalloonCount where
  allan : ℕ
  jake : ℕ
  sarah : ℕ

/-- Initial balloon count --/
def initial : BalloonCount :=
  { allan := 5, jake := 4, sarah := 0 }

/-- Sarah buys balloons at the park --/
def sarah_buys (bc : BalloonCount) (n : ℕ) : BalloonCount :=
  { bc with sarah := bc.sarah + n }

/-- Allan buys balloons at the park --/
def allan_buys (bc : BalloonCount) (n : ℕ) : BalloonCount :=
  { bc with allan := bc.allan + n }

/-- Allan gives balloons to Jake --/
def allan_gives_to_jake (bc : BalloonCount) (n : ℕ) : BalloonCount :=
  { bc with allan := bc.allan - n, jake := bc.jake + n }

/-- The final balloon count after all actions --/
def final : BalloonCount :=
  allan_gives_to_jake (allan_buys (sarah_buys initial 7) 3) 2

theorem balloon_count_theorem :
  final = { allan := 6, jake := 6, sarah := 7 } := by
  sorry

end NUMINAMATH_CALUDE_balloon_count_theorem_l1447_144754


namespace NUMINAMATH_CALUDE_zoe_pictures_before_dolphin_show_l1447_144757

/-- The number of pictures Zoe took at the dolphin show -/
def pictures_at_dolphin_show : ℕ := 16

/-- The total number of pictures Zoe has taken -/
def total_pictures : ℕ := 44

/-- The number of pictures Zoe took before the dolphin show -/
def pictures_before_dolphin_show : ℕ := total_pictures - pictures_at_dolphin_show

theorem zoe_pictures_before_dolphin_show :
  pictures_before_dolphin_show = 28 :=
by sorry

end NUMINAMATH_CALUDE_zoe_pictures_before_dolphin_show_l1447_144757


namespace NUMINAMATH_CALUDE_milkshakes_bought_l1447_144770

def initial_amount : ℕ := 120
def hamburger_cost : ℕ := 4
def milkshake_cost : ℕ := 3
def hamburgers_bought : ℕ := 8
def final_amount : ℕ := 70

theorem milkshakes_bought :
  ∃ (m : ℕ), 
    initial_amount - (hamburger_cost * hamburgers_bought + milkshake_cost * m) = final_amount ∧
    m = 6 := by
  sorry

end NUMINAMATH_CALUDE_milkshakes_bought_l1447_144770


namespace NUMINAMATH_CALUDE_cube_root_of_64_l1447_144784

theorem cube_root_of_64 (n : ℕ) (t : ℕ) : t = n * (n - 1) * (n + 1) + n → t = 64 → n = 4 := by
  sorry

end NUMINAMATH_CALUDE_cube_root_of_64_l1447_144784


namespace NUMINAMATH_CALUDE_stratified_sampling_theorem_l1447_144773

/-- Represents the sample size for each category of students -/
structure SampleSizes where
  junior : ℕ
  undergraduate : ℕ
  graduate : ℕ

/-- Calculates the stratified sample sizes given the total population, category populations, and total sample size -/
def calculateSampleSizes (totalPopulation : ℕ) (juniorPopulation : ℕ) (undergradPopulation : ℕ) (sampleSize : ℕ) : SampleSizes :=
  let juniorSample := (juniorPopulation * sampleSize) / totalPopulation
  let undergradSample := (undergradPopulation * sampleSize) / totalPopulation
  let gradSample := sampleSize - juniorSample - undergradSample
  { junior := juniorSample,
    undergraduate := undergradSample,
    graduate := gradSample }

theorem stratified_sampling_theorem (totalPopulation : ℕ) (juniorPopulation : ℕ) (undergradPopulation : ℕ) (sampleSize : ℕ)
    (h1 : totalPopulation = 5600)
    (h2 : juniorPopulation = 1300)
    (h3 : undergradPopulation = 3000)
    (h4 : sampleSize = 280) :
    calculateSampleSizes totalPopulation juniorPopulation undergradPopulation sampleSize =
    { junior := 65, undergraduate := 150, graduate := 65 } := by
  sorry

#check stratified_sampling_theorem

end NUMINAMATH_CALUDE_stratified_sampling_theorem_l1447_144773


namespace NUMINAMATH_CALUDE_original_number_proof_l1447_144707

theorem original_number_proof (x : ℝ) : x * 1.5 = 120 → x = 80 := by
  sorry

end NUMINAMATH_CALUDE_original_number_proof_l1447_144707


namespace NUMINAMATH_CALUDE_reflection_line_sum_l1447_144722

/-- Given a line y = mx + b, if the reflection of point (1, -2) across this line is (7, 4), then m + b = 4 -/
theorem reflection_line_sum (m b : ℝ) : 
  (∃ (x y : ℝ), 
    -- The midpoint of the segment is on the line
    y = m * x + b ∧ 
    -- The midpoint coordinates
    x = (1 + 7) / 2 ∧ 
    y = (-2 + 4) / 2 ∧ 
    -- The line is perpendicular to the segment
    m * ((7 - 1) / (4 - (-2))) = -1) → 
  m + b = 4 := by
sorry

end NUMINAMATH_CALUDE_reflection_line_sum_l1447_144722


namespace NUMINAMATH_CALUDE_cubic_function_unique_negative_zero_l1447_144726

/-- Given a cubic function f(x) = ax³ - 3x² + 1 with a unique zero point x₀ < 0, prove that a > 2 -/
theorem cubic_function_unique_negative_zero (a : ℝ) :
  (∃! x₀ : ℝ, a * x₀^3 - 3 * x₀^2 + 1 = 0) →
  (∀ x₀ : ℝ, a * x₀^3 - 3 * x₀^2 + 1 = 0 → x₀ < 0) →
  a > 2 := by
  sorry

end NUMINAMATH_CALUDE_cubic_function_unique_negative_zero_l1447_144726


namespace NUMINAMATH_CALUDE_systematic_sampling_l1447_144759

/-- Systematic sampling problem -/
theorem systematic_sampling
  (population_size : ℕ)
  (sample_size : ℕ)
  (last_sampled : ℕ)
  (h1 : population_size = 8000)
  (h2 : sample_size = 50)
  (h3 : last_sampled = 7894)
  (h4 : last_sampled < population_size) :
  let segment_size := population_size / sample_size
  let first_sampled := last_sampled - (segment_size - 1)
  first_sampled = 735 :=
by sorry

end NUMINAMATH_CALUDE_systematic_sampling_l1447_144759


namespace NUMINAMATH_CALUDE_second_artifact_time_multiple_l1447_144700

/-- Represents the time spent on artifact collection in months -/
structure ArtifactTime where
  research : ℕ
  expedition : ℕ

/-- The total time spent on both artifacts in months -/
def total_time : ℕ := 10 * 12

/-- Time spent on the first artifact -/
def first_artifact : ArtifactTime := { research := 6, expedition := 2 * 12 }

/-- Calculate the total time spent on an artifact -/
def total_artifact_time (a : ArtifactTime) : ℕ := a.research + a.expedition

/-- The multiple of time taken for the second artifact compared to the first -/
def time_multiple : ℚ :=
  (total_time - total_artifact_time first_artifact) / total_artifact_time first_artifact

theorem second_artifact_time_multiple :
  time_multiple = 3 := by sorry

end NUMINAMATH_CALUDE_second_artifact_time_multiple_l1447_144700


namespace NUMINAMATH_CALUDE_b_current_age_l1447_144751

/-- Given two people A and B, prove B's current age is 38 years old. -/
theorem b_current_age (a b : ℕ) : 
  (a + 10 = 2 * (b - 10)) → -- A's age in 10 years = 2 * (B's age 10 years ago)
  (a = b + 8) →             -- A is currently 8 years older than B
  b = 38 :=                 -- B's current age is 38
by sorry

end NUMINAMATH_CALUDE_b_current_age_l1447_144751


namespace NUMINAMATH_CALUDE_base3_to_base10_conversion_l1447_144742

/-- Converts a base 3 number represented as a list of digits to its base 10 equivalent -/
def base3ToBase10 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (3 ^ i)) 0

/-- The base 3 representation of the number 102012₃ -/
def base3Number : List Nat := [2, 1, 0, 2, 0, 1]

theorem base3_to_base10_conversion :
  base3ToBase10 base3Number = 302 := by
  sorry

end NUMINAMATH_CALUDE_base3_to_base10_conversion_l1447_144742


namespace NUMINAMATH_CALUDE_midpoint_of_specific_segment_l1447_144745

/-- Represents a point in polar coordinates -/
structure PolarPoint where
  r : ℝ
  θ : ℝ

/-- The midpoint of two polar points -/
def polarMidpoint (p1 p2 : PolarPoint) : PolarPoint :=
  sorry

theorem midpoint_of_specific_segment :
  let p1 : PolarPoint := ⟨6, π/6⟩
  let p2 : PolarPoint := ⟨2, -π/6⟩
  let m := polarMidpoint p1 p2
  0 ≤ m.θ ∧ m.θ < 2*π ∧ m.r > 0 ∧ m = ⟨Real.sqrt 13, π/6⟩ := by
  sorry

end NUMINAMATH_CALUDE_midpoint_of_specific_segment_l1447_144745


namespace NUMINAMATH_CALUDE_rohans_age_is_25_l1447_144714

/-- Rohan's current age in years -/
def rohans_current_age : ℕ := 25

/-- Rohan's age 15 years ago -/
def rohans_past_age : ℕ := rohans_current_age - 15

/-- Rohan's age 15 years from now -/
def rohans_future_age : ℕ := rohans_current_age + 15

/-- Theorem stating that Rohan's current age is 25, given the condition -/
theorem rohans_age_is_25 :
  rohans_current_age = 25 ∧
  rohans_future_age = 4 * rohans_past_age :=
by sorry

end NUMINAMATH_CALUDE_rohans_age_is_25_l1447_144714


namespace NUMINAMATH_CALUDE_largest_prime_divisor_of_39_squared_plus_52_squared_l1447_144706

theorem largest_prime_divisor_of_39_squared_plus_52_squared : 
  (Nat.factors (39^2 + 52^2)).maximum? = some 13 := by
  sorry

end NUMINAMATH_CALUDE_largest_prime_divisor_of_39_squared_plus_52_squared_l1447_144706


namespace NUMINAMATH_CALUDE_butanoic_acid_nine_moles_weight_l1447_144793

/-- The atomic weight of Carbon in g/mol -/
def carbon_weight : ℝ := 12.01

/-- The atomic weight of Hydrogen in g/mol -/
def hydrogen_weight : ℝ := 1.008

/-- The atomic weight of Oxygen in g/mol -/
def oxygen_weight : ℝ := 16.00

/-- The number of Carbon atoms in Butanoic acid -/
def carbon_count : ℕ := 4

/-- The number of Hydrogen atoms in Butanoic acid -/
def hydrogen_count : ℕ := 8

/-- The number of Oxygen atoms in Butanoic acid -/
def oxygen_count : ℕ := 2

/-- The number of moles of Butanoic acid -/
def moles : ℝ := 9

/-- The molecular weight of Butanoic acid in g/mol -/
def butanoic_acid_weight : ℝ := 
  carbon_weight * carbon_count + 
  hydrogen_weight * hydrogen_count + 
  oxygen_weight * oxygen_count

/-- Theorem: The molecular weight of 9 moles of Butanoic acid is 792.936 grams -/
theorem butanoic_acid_nine_moles_weight : 
  butanoic_acid_weight * moles = 792.936 := by sorry

end NUMINAMATH_CALUDE_butanoic_acid_nine_moles_weight_l1447_144793


namespace NUMINAMATH_CALUDE_cosine_sum_zero_l1447_144736

theorem cosine_sum_zero (n : ℤ) (h : n % 7 = 1 ∨ n % 7 = 3 ∨ n % 7 = 4) :
  Real.cos (n * π / 7 - 13 * π / 14) + 
  Real.cos (3 * n * π / 7 - 3 * π / 14) + 
  Real.cos (5 * n * π / 7 - 3 * π / 14) = 0 := by
  sorry

end NUMINAMATH_CALUDE_cosine_sum_zero_l1447_144736


namespace NUMINAMATH_CALUDE_min_tiles_for_2014_area_l1447_144733

/-- Represents the side length of a square tile in centimeters -/
inductive TileSize
  | Small : TileSize  -- 3 cm
  | Large : TileSize  -- 5 cm

/-- Calculates the area of a square tile given its size -/
def tileArea (size : TileSize) : ℕ :=
  match size with
  | TileSize.Small => 9   -- 3² = 9
  | TileSize.Large => 25  -- 5² = 25

/-- Represents a collection of tiles -/
structure TileCollection where
  smallCount : ℕ
  largeCount : ℕ

/-- Calculates the total area covered by a collection of tiles -/
def totalArea (tiles : TileCollection) : ℕ :=
  tiles.smallCount * tileArea TileSize.Small + tiles.largeCount * tileArea TileSize.Large

/-- Calculates the total number of tiles in a collection -/
def totalTiles (tiles : TileCollection) : ℕ :=
  tiles.smallCount + tiles.largeCount

theorem min_tiles_for_2014_area :
  ∃ (tiles : TileCollection),
    totalArea tiles = 2014 ∧
    (∀ (other : TileCollection), totalArea other = 2014 → totalTiles tiles ≤ totalTiles other) ∧
    totalTiles tiles = 94 :=
  sorry

end NUMINAMATH_CALUDE_min_tiles_for_2014_area_l1447_144733


namespace NUMINAMATH_CALUDE_area_TURS_l1447_144758

/-- Rectangle PQRS with trapezoid TURS inside -/
structure Geometry where
  /-- Width of rectangle PQRS -/
  width : ℝ
  /-- Height of rectangle PQRS -/
  height : ℝ
  /-- Area of rectangle PQRS -/
  area_PQRS : ℝ
  /-- Distance of T from S -/
  ST_distance : ℝ
  /-- Distance of U from R -/
  UR_distance : ℝ
  /-- Width is 6 units -/
  width_eq : width = 6
  /-- Height is 4 units -/
  height_eq : height = 4
  /-- Area of PQRS is 24 square units -/
  area_eq : area_PQRS = 24
  /-- ST distance is 1 unit -/
  ST_eq : ST_distance = 1
  /-- UR distance is 1 unit -/
  UR_eq : UR_distance = 1

/-- The area of trapezoid TURS is 20 square units -/
theorem area_TURS (g : Geometry) : Real.sqrt ((g.width - 2 * g.ST_distance) * g.height + g.ST_distance * g.height) = 20 := by
  sorry

end NUMINAMATH_CALUDE_area_TURS_l1447_144758


namespace NUMINAMATH_CALUDE_largest_divisor_of_difference_of_squares_l1447_144760

theorem largest_divisor_of_difference_of_squares (m n : ℤ) : 
  Odd m → Odd n → n < m → 
  (∃ k : ℤ, m ^ 2 - n ^ 2 = 8 * k) ∧ 
  (∀ d : ℤ, d > 8 → ∃ m' n' : ℤ, Odd m' ∧ Odd n' ∧ n' < m' ∧ ¬(d ∣ (m' ^ 2 - n' ^ 2))) :=
by sorry

end NUMINAMATH_CALUDE_largest_divisor_of_difference_of_squares_l1447_144760


namespace NUMINAMATH_CALUDE_pencils_across_diameter_l1447_144712

-- Define the radius of the circle in feet
def radius : ℝ := 14

-- Define the length of a pencil in feet
def pencil_length : ℝ := 0.5

-- Theorem: The number of pencils that can be placed end-to-end across the diameter is 56
theorem pencils_across_diameter : 
  ⌊(2 * radius) / pencil_length⌋ = 56 := by sorry

end NUMINAMATH_CALUDE_pencils_across_diameter_l1447_144712


namespace NUMINAMATH_CALUDE_floor_greater_than_x_minus_one_l1447_144739

theorem floor_greater_than_x_minus_one (x : ℝ) : ⌊x⌋ > x - 1 := by sorry

end NUMINAMATH_CALUDE_floor_greater_than_x_minus_one_l1447_144739


namespace NUMINAMATH_CALUDE_largest_product_sum_of_digits_l1447_144738

def is_single_digit (n : ℕ) : Prop := n ≥ 1 ∧ n ≤ 9

def is_prime (p : ℕ) : Prop := p > 1 ∧ ∀ m : ℕ, m > 1 → m < p → ¬(p % m = 0)

def is_odd (n : ℕ) : Prop := n % 2 = 1

def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + sum_of_digits (n / 10)

theorem largest_product_sum_of_digits :
  ∃ (n d e : ℕ),
    is_single_digit d ∧
    is_prime d ∧
    is_single_digit e ∧
    is_odd e ∧
    ¬is_prime e ∧
    n = d * e * (d^2 + e) ∧
    (∀ (m : ℕ), m = d' * e' * (d'^2 + e') →
      is_single_digit d' →
      is_prime d' →
      is_single_digit e' →
      is_odd e' →
      ¬is_prime e' →
      m ≤ n) ∧
    sum_of_digits n = 9 :=
  sorry

end NUMINAMATH_CALUDE_largest_product_sum_of_digits_l1447_144738


namespace NUMINAMATH_CALUDE_freshman_sophomore_percentage_l1447_144703

theorem freshman_sophomore_percentage
  (total_students : ℕ)
  (pet_ownership_ratio : ℚ)
  (non_pet_owners : ℕ)
  (h1 : total_students = 400)
  (h2 : pet_ownership_ratio = 1/5)
  (h3 : non_pet_owners = 160) :
  (↑(total_students - non_pet_owners) / (1 - pet_ownership_ratio)) / total_students = 1/2 :=
sorry

end NUMINAMATH_CALUDE_freshman_sophomore_percentage_l1447_144703


namespace NUMINAMATH_CALUDE_smallest_advantageous_discount_l1447_144767

theorem smallest_advantageous_discount : ∃ n : ℕ,
  (n : ℝ) > 0 ∧
  (∀ m : ℕ, m < n →
    (1 - m / 100 : ℝ) > (1 - 0.20)^2 ∨
    (1 - m / 100 : ℝ) > (1 - 0.13)^3 ∨
    (1 - m / 100 : ℝ) > (1 - 0.30) * (1 - 0.10)) ∧
  (1 - n / 100 : ℝ) ≤ (1 - 0.20)^2 ∧
  (1 - n / 100 : ℝ) ≤ (1 - 0.13)^3 ∧
  (1 - n / 100 : ℝ) ≤ (1 - 0.30) * (1 - 0.10) :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_advantageous_discount_l1447_144767


namespace NUMINAMATH_CALUDE_cooler_capacity_ratio_l1447_144777

/-- Given three coolers with specific capacities, prove the ratio of the third to the second is 1/2. -/
theorem cooler_capacity_ratio :
  ∀ (c₁ c₂ c₃ : ℝ),
  c₁ = 100 →
  c₂ = c₁ + 0.5 * c₁ →
  c₁ + c₂ + c₃ = 325 →
  c₃ / c₂ = 1 / 2 := by
sorry

end NUMINAMATH_CALUDE_cooler_capacity_ratio_l1447_144777


namespace NUMINAMATH_CALUDE_football_banquet_food_consumption_l1447_144778

theorem football_banquet_food_consumption 
  (max_food_per_guest : ℝ) 
  (min_guests : ℕ) 
  (h1 : max_food_per_guest = 2) 
  (h2 : min_guests = 160) : 
  ∃ (total_food : ℝ), total_food = max_food_per_guest * min_guests ∧ total_food = 320 := by
  sorry

end NUMINAMATH_CALUDE_football_banquet_food_consumption_l1447_144778


namespace NUMINAMATH_CALUDE_equal_numbers_exist_l1447_144786

/-- A 10x10 grid of integers -/
def Grid := Fin 10 → Fin 10 → ℤ

/-- Two cells are adjacent if they differ by 1 in exactly one coordinate -/
def adjacent (i j i' j' : Fin 10) : Prop :=
  (i = i' ∧ j.val + 1 = j'.val) ∨
  (i = i' ∧ j'.val + 1 = j.val) ∨
  (j = j' ∧ i.val + 1 = i'.val) ∨
  (j = j' ∧ i'.val + 1 = i.val)

/-- The property that adjacent cells differ by at most 5 -/
def valid_grid (g : Grid) : Prop :=
  ∀ i j i' j', adjacent i j i' j' → |g i j - g i' j'| ≤ 5

theorem equal_numbers_exist (g : Grid) (h : valid_grid g) :
  ∃ i j i' j', (i ≠ i' ∨ j ≠ j') ∧ g i j = g i' j' :=
sorry

end NUMINAMATH_CALUDE_equal_numbers_exist_l1447_144786


namespace NUMINAMATH_CALUDE_circle_center_proof_l1447_144749

theorem circle_center_proof (C : Set (ℝ × ℝ)) (center : ℝ × ℝ) :
  -- The circle passes through (1,0)
  (1, 0) ∈ C →
  -- The circle is tangent to y = x^2 at (2,4)
  (2, 4) ∈ C →
  (∀ (x y : ℝ), (x, y) ∈ C → y ≠ x^2 ∨ (x = 2 ∧ y = 4)) →
  -- The circle is tangent to the x-axis
  (∃ (x : ℝ), (x, 0) ∈ C ∧ ∀ (y : ℝ), y ≠ 0 → (x, y) ∉ C) →
  -- C is a circle with center 'center'
  (∀ (p : ℝ × ℝ), p ∈ C ↔ (p.1 - center.1)^2 + (p.2 - center.2)^2 = (1 - center.1)^2 + center.2^2) →
  -- The center is (178/15, 53/15)
  center = (178/15, 53/15) := by
sorry

end NUMINAMATH_CALUDE_circle_center_proof_l1447_144749


namespace NUMINAMATH_CALUDE_sum_squares_3005_odd_integers_units_digit_l1447_144705

def first_n_odd_integers (n : ℕ) : List ℕ :=
  List.range n |> List.map (fun i => 2 * i + 1)

def square (n : ℕ) : ℕ := n * n

def units_digit (n : ℕ) : ℕ := n % 10

theorem sum_squares_3005_odd_integers_units_digit :
  units_digit (List.sum (List.map square (first_n_odd_integers 3005))) = 3 := by
  sorry

end NUMINAMATH_CALUDE_sum_squares_3005_odd_integers_units_digit_l1447_144705


namespace NUMINAMATH_CALUDE_inequality_proof_l1447_144719

theorem inequality_proof (a b c : ℝ) 
  (h_pos_a : a > 0) (h_pos_b : b > 0) (h_pos_c : c > 0)
  (h_eq : a^2 + b^2 + 4*c^2 = 3) : 
  (a + b + 2*c ≤ 3) ∧ 
  (b = 2*c → 1/a + 1/c ≥ 3) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1447_144719


namespace NUMINAMATH_CALUDE_product_modulo_l1447_144763

theorem product_modulo : (2345 * 1554) % 700 = 630 := by
  sorry

end NUMINAMATH_CALUDE_product_modulo_l1447_144763


namespace NUMINAMATH_CALUDE_range_of_f_l1447_144750

def f (x : ℤ) : ℤ := x^2 - 1

def domain : Set ℤ := {-1, 0, 1}

theorem range_of_f : 
  {y | ∃ x ∈ domain, f x = y} = {-1, 0} := by sorry

end NUMINAMATH_CALUDE_range_of_f_l1447_144750


namespace NUMINAMATH_CALUDE_julie_delivered_600_newspapers_l1447_144710

/-- Represents Julie's earnings and expenses --/
structure JulieFinances where
  saved : ℕ
  bikeCost : ℕ
  lawnsMowed : ℕ
  lawnRate : ℕ
  dogsWalked : ℕ
  dogRate : ℕ
  newspaperRate : ℕ
  leftover : ℕ

/-- Calculates the number of newspapers Julie delivered --/
def newspapersDelivered (j : JulieFinances) : ℕ :=
  ((j.bikeCost + j.leftover) - (j.saved + j.lawnsMowed * j.lawnRate + j.dogsWalked * j.dogRate)) / j.newspaperRate

/-- Theorem stating that Julie delivered 600 newspapers --/
theorem julie_delivered_600_newspapers :
  let j : JulieFinances := {
    saved := 1500,
    bikeCost := 2345,
    lawnsMowed := 20,
    lawnRate := 20,
    dogsWalked := 24,
    dogRate := 15,
    newspaperRate := 40,  -- in cents
    leftover := 155
  }
  newspapersDelivered j = 600 := by sorry


end NUMINAMATH_CALUDE_julie_delivered_600_newspapers_l1447_144710


namespace NUMINAMATH_CALUDE_sequence_2007th_term_l1447_144783

theorem sequence_2007th_term (a : ℕ → ℝ) :
  (∀ n, a n > 0) →
  a 0 = 1 →
  (∀ n, a (n + 2) = 6 * a n - a (n + 1)) →
  a 2007 = 2^2007 := by
sorry

end NUMINAMATH_CALUDE_sequence_2007th_term_l1447_144783


namespace NUMINAMATH_CALUDE_relationship_holds_l1447_144708

def x : Fin 5 → ℕ
  | ⟨0, _⟩ => 1
  | ⟨1, _⟩ => 2
  | ⟨2, _⟩ => 3
  | ⟨3, _⟩ => 4
  | ⟨4, _⟩ => 5

def y : Fin 5 → ℕ
  | ⟨0, _⟩ => 4
  | ⟨1, _⟩ => 15
  | ⟨2, _⟩ => 40
  | ⟨3, _⟩ => 85
  | ⟨4, _⟩ => 156

theorem relationship_holds : ∀ i : Fin 5, y i = (x i)^3 + 2*(x i) + 1 := by
  sorry

end NUMINAMATH_CALUDE_relationship_holds_l1447_144708


namespace NUMINAMATH_CALUDE_sum_of_real_roots_l1447_144794

theorem sum_of_real_roots (x : ℝ) : 
  let f : ℝ → ℝ := fun x => x^4 - 8*x + 4
  ∃ (r₁ r₂ : ℝ), (f r₁ = 0 ∧ f r₂ = 0 ∧ (∀ r : ℝ, f r = 0 → r = r₁ ∨ r = r₂)) ∧ 
  r₁ + r₂ = -2 * Real.sqrt 2 :=
sorry

end NUMINAMATH_CALUDE_sum_of_real_roots_l1447_144794


namespace NUMINAMATH_CALUDE_binary_multiplication_theorem_l1447_144769

/-- Converts a list of binary digits to its decimal representation -/
def binaryToDecimal (bits : List Bool) : Nat :=
  bits.foldr (fun b acc => 2 * acc + if b then 1 else 0) 0

/-- Converts a natural number to its binary representation -/
def decimalToBinary (n : Nat) : List Bool :=
  if n = 0 then [false] else
  let rec toBinary (m : Nat) : List Bool :=
    if m = 0 then [] else (m % 2 = 1) :: toBinary (m / 2)
  toBinary n

theorem binary_multiplication_theorem :
  let a := [true, false, true, true]  -- 1101₂
  let b := [true, true, true]         -- 111₂
  let product := [true, true, true, false, false, true, true]  -- 1100111₂
  binaryToDecimal a * binaryToDecimal b = binaryToDecimal product := by
  sorry

end NUMINAMATH_CALUDE_binary_multiplication_theorem_l1447_144769


namespace NUMINAMATH_CALUDE_sally_picked_42_peaches_l1447_144768

/-- The number of peaches Sally picked from the orchard -/
def peaches_picked (initial current : ℕ) : ℕ :=
  current - initial

/-- Proof that Sally picked 42 peaches from the orchard -/
theorem sally_picked_42_peaches (initial current : ℕ) 
  (h1 : initial = 13) 
  (h2 : current = 55) : 
  peaches_picked initial current = 42 := by
  sorry

end NUMINAMATH_CALUDE_sally_picked_42_peaches_l1447_144768


namespace NUMINAMATH_CALUDE_work_completion_time_l1447_144734

/-- The number of days y needs to finish the work alone -/
def y_days : ℝ := 15

/-- The number of days y worked before leaving -/
def y_worked : ℝ := 9

/-- The number of days x needs to finish the remaining work after y left -/
def x_remaining : ℝ := 8

/-- The number of days x needs to finish the work alone -/
def x_days : ℝ := 20

theorem work_completion_time :
  x_days = 20 :=
sorry

end NUMINAMATH_CALUDE_work_completion_time_l1447_144734


namespace NUMINAMATH_CALUDE_jerry_always_escapes_l1447_144704

/-- Represents the square pool -/
structure Pool :=
  (side : ℝ)
  (is_positive : side > 0)

/-- Represents the speeds of Tom and Jerry -/
structure Speeds :=
  (jerry_swim : ℝ)
  (tom_run : ℝ)
  (speed_ratio : tom_run = 4 * jerry_swim)
  (positive_speeds : jerry_swim > 0 ∧ tom_run > 0)

/-- Represents a point in the pool -/
structure Point :=
  (x : ℝ)
  (y : ℝ)

/-- Defines whether a point is inside or on the edge of the pool -/
def in_pool (p : Point) (pool : Pool) : Prop :=
  0 ≤ p.x ∧ p.x ≤ pool.side ∧ 0 ≤ p.y ∧ p.y ≤ pool.side

/-- Defines whether Jerry can escape from Tom -/
def can_escape (pool : Pool) (speeds : Speeds) : Prop :=
  ∀ (jerry_start tom_start : Point),
    in_pool jerry_start pool →
    ¬in_pool tom_start pool →
    ∃ (escape_point : Point),
      in_pool escape_point pool ∧
      (escape_point.x = 0 ∨ escape_point.x = pool.side ∨
       escape_point.y = 0 ∨ escape_point.y = pool.side) ∧
      (escape_point.x - jerry_start.x) ^ 2 + (escape_point.y - jerry_start.y) ^ 2 <
      ((escape_point.x - tom_start.x) ^ 2 + (escape_point.y - tom_start.y) ^ 2) * (speeds.jerry_swim / speeds.tom_run) ^ 2

theorem jerry_always_escapes (pool : Pool) (speeds : Speeds) :
  can_escape pool speeds :=
sorry

end NUMINAMATH_CALUDE_jerry_always_escapes_l1447_144704


namespace NUMINAMATH_CALUDE_circle_to_ellipse_l1447_144740

/-- If z is a complex number tracing a circle centered at the origin with radius 3,
    then z + 1/z traces an ellipse. -/
theorem circle_to_ellipse (z : ℂ) (h : ∀ θ : ℝ, z = 3 * Complex.exp (Complex.I * θ)) :
  ∃ a b : ℝ, a > 0 ∧ b > 0 ∧ 
  ∀ θ : ℝ, ∃ x y : ℝ, 
    z + 1/z = Complex.mk x y ∧ 
    (x^2 / a^2) + (y^2 / b^2) = 1 :=
sorry

end NUMINAMATH_CALUDE_circle_to_ellipse_l1447_144740


namespace NUMINAMATH_CALUDE_julians_comic_book_pages_l1447_144735

theorem julians_comic_book_pages 
  (frames_per_page : ℝ) 
  (total_frames : ℕ) 
  (h1 : frames_per_page = 143.0) 
  (h2 : total_frames = 1573) : 
  ⌊(total_frames : ℝ) / frames_per_page⌋ = 11 := by
sorry

end NUMINAMATH_CALUDE_julians_comic_book_pages_l1447_144735


namespace NUMINAMATH_CALUDE_smallest_integer_with_remainder_one_l1447_144756

theorem smallest_integer_with_remainder_one : ∃ k : ℕ, 
  k > 1 ∧ 
  k % 10 = 1 ∧ 
  k % 15 = 1 ∧ 
  k % 9 = 1 ∧
  (∀ m : ℕ, m > 1 ∧ m % 10 = 1 ∧ m % 15 = 1 ∧ m % 9 = 1 → k ≤ m) ∧
  k = 91 :=
by sorry

end NUMINAMATH_CALUDE_smallest_integer_with_remainder_one_l1447_144756


namespace NUMINAMATH_CALUDE_problem_statement_l1447_144724

theorem problem_statement (x n : ℕ) : 
  x = 5^n - 1 →
  (∃ p q : ℕ, Prime p ∧ Prime q ∧ p ≠ q ∧ p ≠ 11 ∧ q ≠ 11 ∧ x = 2^(Nat.log2 x) * p * q * 11) →
  x = 3124 := by
sorry

end NUMINAMATH_CALUDE_problem_statement_l1447_144724


namespace NUMINAMATH_CALUDE_notebook_duration_example_l1447_144788

/-- The number of days notebooks last given the number of notebooks, pages per notebook, and pages used per day. -/
def notebook_duration (num_notebooks : ℕ) (pages_per_notebook : ℕ) (pages_per_day : ℕ) : ℕ :=
  (num_notebooks * pages_per_notebook) / pages_per_day

/-- Theorem stating that 5 notebooks with 40 pages each, using 4 pages per day, last for 50 days. -/
theorem notebook_duration_example : notebook_duration 5 40 4 = 50 := by
  sorry

end NUMINAMATH_CALUDE_notebook_duration_example_l1447_144788


namespace NUMINAMATH_CALUDE_factor_problem_l1447_144744

theorem factor_problem (x : ℝ) (f : ℝ) : 
  x = 6 → (2 * x + 9) * f = 63 → f = 3 := by
  sorry

end NUMINAMATH_CALUDE_factor_problem_l1447_144744


namespace NUMINAMATH_CALUDE_simple_interest_calculation_l1447_144764

def simple_interest (principal : ℚ) (rate : ℚ) (time : ℚ) : ℚ :=
  principal * rate * time / 100

theorem simple_interest_calculation :
  let principal : ℚ := 80325
  let rate : ℚ := 1
  let time : ℚ := 5
  simple_interest principal rate time = 4016.25 := by sorry

end NUMINAMATH_CALUDE_simple_interest_calculation_l1447_144764


namespace NUMINAMATH_CALUDE_arithmetic_geometric_sequence_product_l1447_144752

/-- An arithmetic-geometric sequence -/
def ArithmeticGeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = a n * q

/-- The main theorem -/
theorem arithmetic_geometric_sequence_product (a : ℕ → ℝ) :
  ArithmeticGeometricSequence a →
  a 1 = 3 →
  a 1 + a 3 + a 5 = 21 →
  a 2 * a 4 = 36 := by
  sorry


end NUMINAMATH_CALUDE_arithmetic_geometric_sequence_product_l1447_144752


namespace NUMINAMATH_CALUDE_game_probability_difference_l1447_144789

def coin_prob_heads : ℚ := 2/3
def coin_prob_tails : ℚ := 1/3

def game_x_win_prob : ℚ :=
  3 * (coin_prob_heads^2 * coin_prob_tails) + coin_prob_heads^3

def game_y_win_prob : ℚ :=
  4 * (coin_prob_heads^3 * coin_prob_tails + coin_prob_tails^3 * coin_prob_heads) +
  coin_prob_heads^4 + coin_prob_tails^4

theorem game_probability_difference :
  game_x_win_prob - game_y_win_prob = 11/81 :=
sorry

end NUMINAMATH_CALUDE_game_probability_difference_l1447_144789


namespace NUMINAMATH_CALUDE_root_analysis_uses_classification_and_discussion_l1447_144795

/-- A mathematical thinking method -/
inductive MathThinkingMethod
| Transformation
| Equation
| ClassificationAndDiscussion
| NumberAndShapeCombination

/-- A number category for root analysis -/
inductive NumberCategory
| Positive
| Zero
| Negative

/-- Represents the analysis of roots -/
structure RootAnalysis where
  categories : List NumberCategory
  method : MathThinkingMethod

/-- The specific root analysis we're considering -/
def squareAndCubeRootAnalysis : RootAnalysis :=
  { categories := [NumberCategory.Positive, NumberCategory.Zero, NumberCategory.Negative],
    method := MathThinkingMethod.ClassificationAndDiscussion }

/-- Theorem stating that the given root analysis uses classification and discussion thinking -/
theorem root_analysis_uses_classification_and_discussion :
  squareAndCubeRootAnalysis.method = MathThinkingMethod.ClassificationAndDiscussion :=
by sorry

end NUMINAMATH_CALUDE_root_analysis_uses_classification_and_discussion_l1447_144795


namespace NUMINAMATH_CALUDE_age_multiple_problem_l1447_144771

theorem age_multiple_problem (a b : ℕ) (m : ℚ) : 
  a = b + 5 →
  a + b = 13 →
  m * (a + 7 : ℚ) = 4 * (b + 7 : ℚ) →
  m = 2.75 := by
  sorry

end NUMINAMATH_CALUDE_age_multiple_problem_l1447_144771


namespace NUMINAMATH_CALUDE_plant_mass_problem_l1447_144781

theorem plant_mass_problem (initial_mass : ℝ) : 
  (((initial_mass * 3 + 4) * 3 + 4) * 3 + 4 = 133) → initial_mass = 3 := by
sorry

end NUMINAMATH_CALUDE_plant_mass_problem_l1447_144781


namespace NUMINAMATH_CALUDE_round_trip_average_speed_l1447_144774

/-- Calculates the average speed of a round trip given the following conditions:
  * The total distance of the round trip is 4 miles
  * The outbound journey of 2 miles takes 1 hour
  * The return journey of 2 miles is completed at a speed of 6.000000000000002 miles/hour
-/
theorem round_trip_average_speed : 
  let total_distance : ℝ := 4
  let outbound_distance : ℝ := 2
  let outbound_time : ℝ := 1
  let return_speed : ℝ := 6.000000000000002
  let return_time : ℝ := outbound_distance / return_speed
  let total_time : ℝ := outbound_time + return_time
  total_distance / total_time = 3 := by
sorry

end NUMINAMATH_CALUDE_round_trip_average_speed_l1447_144774


namespace NUMINAMATH_CALUDE_right_triangle_area_l1447_144796

-- Define the right triangle
def RightTriangle (a b c : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ a^2 + b^2 = c^2

-- Define the incircle radius formula for a right triangle
def IncircleRadius (a b c r : ℝ) : Prop :=
  r = (a + b - c) / 2

-- Theorem statement
theorem right_triangle_area (a b c r : ℝ) :
  RightTriangle a b c →
  IncircleRadius a b c r →
  a = 3 →
  r = 3/8 →
  (1/2) * a * b = 21/16 :=
by sorry

end NUMINAMATH_CALUDE_right_triangle_area_l1447_144796


namespace NUMINAMATH_CALUDE_david_presents_l1447_144723

/-- The total number of presents David received -/
def total_presents (christmas_presents : ℕ) (birthday_presents : ℕ) : ℕ :=
  christmas_presents + birthday_presents

/-- Theorem: Given the conditions, David received 90 presents in total -/
theorem david_presents : 
  ∀ (christmas_presents birthday_presents : ℕ),
  christmas_presents = 60 →
  christmas_presents = 2 * birthday_presents →
  total_presents christmas_presents birthday_presents = 90 := by
  sorry

end NUMINAMATH_CALUDE_david_presents_l1447_144723


namespace NUMINAMATH_CALUDE_circles_configuration_l1447_144713

-- Define the centers of the circles as points in a metric space
variable (X : Type) [MetricSpace X]
variable (P Q R : X)

-- Define the radii of the circles
variable (p q r : ℝ)

-- Define the distance between P and Q
variable (d : ℝ)

-- State the theorem
theorem circles_configuration (h1 : p > q) (h2 : q > r) 
  (h3 : dist R P < p) (h4 : dist R Q < q) (h5 : d = dist P Q) :
  ¬(p + r = d) := by
  sorry

end NUMINAMATH_CALUDE_circles_configuration_l1447_144713


namespace NUMINAMATH_CALUDE_dishonest_dealer_profit_l1447_144780

/-- Calculates the profit percentage for a dishonest dealer --/
theorem dishonest_dealer_profit (real_weight : ℝ) (cost_price : ℝ) 
  (h1 : real_weight > 0) (h2 : cost_price > 0) : 
  let counterfeit_weight := 0.8 * real_weight
  let impure_weight := counterfeit_weight * 1.15
  let selling_price := cost_price * (real_weight / impure_weight)
  let profit := selling_price - cost_price
  profit / cost_price = 0.25 := by sorry

end NUMINAMATH_CALUDE_dishonest_dealer_profit_l1447_144780


namespace NUMINAMATH_CALUDE_part_one_part_two_l1447_144730

-- Define the propositions p and q
def p (a : ℝ) : Prop := ∀ x ∈ Set.Icc 1 2, x^2 - a ≥ 0

def q (a : ℝ) : Prop := ∃ x : ℝ, x^2 + 2*a*x + 2 - a = 0

-- Theorem for part (I)
theorem part_one (a : ℝ) : p a → a ≤ 1 := by
  sorry

-- Theorem for part (II)
theorem part_two (a : ℝ) : ¬(p a ∧ q a) → a ∈ Set.union (Set.Ioo (-2) 1) (Set.Ioi 1) := by
  sorry

end NUMINAMATH_CALUDE_part_one_part_two_l1447_144730


namespace NUMINAMATH_CALUDE_max_sock_pairs_john_sock_problem_l1447_144772

theorem max_sock_pairs (initial_pairs : ℕ) (lost_socks : ℕ) : ℕ :=
  let total_socks := 2 * initial_pairs
  let remaining_socks := total_socks - lost_socks
  let guaranteed_pairs := initial_pairs - lost_socks
  let possible_new_pairs := (remaining_socks - 2 * guaranteed_pairs) / 2
  guaranteed_pairs + possible_new_pairs

theorem john_sock_problem :
  max_sock_pairs 10 5 = 7 := by
  sorry

end NUMINAMATH_CALUDE_max_sock_pairs_john_sock_problem_l1447_144772


namespace NUMINAMATH_CALUDE_complex_number_in_second_quadrant_l1447_144701

theorem complex_number_in_second_quadrant :
  let z : ℂ := (3 + 4 * Complex.I) * Complex.I
  (z.re < 0) ∧ (z.im > 0) := by
  sorry

end NUMINAMATH_CALUDE_complex_number_in_second_quadrant_l1447_144701


namespace NUMINAMATH_CALUDE_folded_rectangle_perimeter_l1447_144746

/-- Given a rectangle with length 20 cm and width 12 cm, when folded along its diagonal,
    the perimeter of the resulting shaded region is 64 cm. -/
theorem folded_rectangle_perimeter :
  ∀ (length width : ℝ),
    length = 20 →
    width = 12 →
    let perimeter := (length + width) * 2
    perimeter = 64 := by
  sorry

end NUMINAMATH_CALUDE_folded_rectangle_perimeter_l1447_144746


namespace NUMINAMATH_CALUDE_min_production_volume_correct_l1447_144747

/-- The minimum production volume to avoid a loss -/
def min_production_volume : ℕ := 150

/-- The total cost function -/
def total_cost (x : ℕ) : ℝ := 3000 + 20 * x - 0.1 * x^2

/-- The selling price per unit -/
def selling_price : ℝ := 25

/-- Theorem stating the minimum production volume to avoid a loss -/
theorem min_production_volume_correct :
  ∀ x : ℕ, 0 < x → x < 240 →
  (selling_price * x ≥ total_cost x ↔ x ≥ min_production_volume) := by
  sorry

end NUMINAMATH_CALUDE_min_production_volume_correct_l1447_144747


namespace NUMINAMATH_CALUDE_bus_stop_problem_l1447_144761

/-- The number of children who got on the bus at the bus stop -/
def children_got_on : ℕ := sorry

/-- The initial number of children on the bus -/
def initial_children : ℕ := 22

/-- The number of children who got off the bus at the bus stop -/
def children_got_off : ℕ := 60

/-- The final number of children on the bus after the bus stop -/
def final_children : ℕ := 2

theorem bus_stop_problem :
  initial_children - children_got_off + children_got_on = final_children ∧
  children_got_on = 40 := by sorry

end NUMINAMATH_CALUDE_bus_stop_problem_l1447_144761


namespace NUMINAMATH_CALUDE_quadratic_distinct_roots_l1447_144721

theorem quadratic_distinct_roots (m : ℝ) :
  (∀ x : ℝ, x^2 - 2*(m-2)*x + m^2 = 0 → (∃ y : ℝ, x ≠ y ∧ y^2 - 2*(m-2)*y + m^2 = 0)) →
  m < 1 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_distinct_roots_l1447_144721


namespace NUMINAMATH_CALUDE_tommy_nickels_l1447_144702

/-- Represents Tommy's coin collection --/
structure CoinCollection where
  quarters : ℕ
  dimes : ℕ
  nickels : ℕ
  pennies : ℕ

/-- Conditions for Tommy's coin collection --/
def tommy_collection (c : CoinCollection) : Prop :=
  c.quarters = 4 ∧
  c.dimes = c.pennies + 10 ∧
  c.nickels = 2 * c.dimes ∧
  c.pennies = 10 * c.quarters

theorem tommy_nickels (c : CoinCollection) : 
  tommy_collection c → c.nickels = 100 := by
  sorry

end NUMINAMATH_CALUDE_tommy_nickels_l1447_144702
