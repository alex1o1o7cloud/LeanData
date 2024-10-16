import Mathlib

namespace NUMINAMATH_CALUDE_correct_boat_equation_l3478_347821

/-- Represents the scenario of boats and students during the Qingming Festival outing. -/
structure BoatScenario where
  totalBoats : ℕ
  largeboatCapacity : ℕ
  smallboatCapacity : ℕ
  totalStudents : ℕ

/-- The equation representing the boat scenario. -/
def boatEquation (scenario : BoatScenario) (x : ℕ) : Prop :=
  scenario.largeboatCapacity * (scenario.totalBoats - x) + scenario.smallboatCapacity * x = scenario.totalStudents

/-- Theorem stating that the given equation correctly represents the boat scenario. -/
theorem correct_boat_equation (scenario : BoatScenario) (h1 : scenario.totalBoats = 8) 
    (h2 : scenario.largeboatCapacity = 6) (h3 : scenario.smallboatCapacity = 4) 
    (h4 : scenario.totalStudents = 38) : 
  boatEquation scenario = fun x => 6 * (8 - x) + 4 * x = 38 := by
  sorry


end NUMINAMATH_CALUDE_correct_boat_equation_l3478_347821


namespace NUMINAMATH_CALUDE_journey_distance_l3478_347814

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Calculates the final position after a series of movements -/
def finalPosition (initialDistance : ℝ) : Point :=
  let southWalk := Point.mk 0 (-initialDistance)
  let eastWalk := Point.mk initialDistance (-initialDistance)
  let northWalk := Point.mk initialDistance 0
  let finalEastWalk := Point.mk (initialDistance + 10) 0
  finalEastWalk

/-- Theorem stating that the initial distance must be 30 to end up 30 meters north -/
theorem journey_distance (initialDistance : ℝ) :
  (finalPosition initialDistance).y = 30 ↔ initialDistance = 30 :=
by sorry

end NUMINAMATH_CALUDE_journey_distance_l3478_347814


namespace NUMINAMATH_CALUDE_money_ratio_proof_l3478_347882

def ram_money : ℝ := 490
def krishan_money : ℝ := 2890

/-- The ratio of money between two people -/
structure MoneyRatio where
  person1 : ℝ
  person2 : ℝ

/-- The condition that two money ratios are equal -/
def equal_ratios (r1 r2 : MoneyRatio) : Prop :=
  r1.person1 / r1.person2 = r2.person1 / r2.person2

theorem money_ratio_proof 
  (ram_gopal : MoneyRatio) 
  (gopal_krishan : MoneyRatio) 
  (h1 : ram_gopal.person1 = ram_money)
  (h2 : gopal_krishan.person2 = krishan_money)
  (h3 : equal_ratios ram_gopal gopal_krishan) :
  ∃ (n : ℕ), 
    ram_gopal.person1 / ram_gopal.person2 = 49 / 119 ∧ 
    n * ram_gopal.person1 = 49 ∧ 
    n * ram_gopal.person2 = 119 := by
  sorry

end NUMINAMATH_CALUDE_money_ratio_proof_l3478_347882


namespace NUMINAMATH_CALUDE_vases_to_arrange_l3478_347877

/-- Proves that the number of vases of flowers to be arranged is 256,
    given that Jane can arrange 16 vases per day and needs 16 days to finish all arrangements. -/
theorem vases_to_arrange (vases_per_day : ℕ) (days_needed : ℕ) 
  (h1 : vases_per_day = 16) 
  (h2 : days_needed = 16) : 
  vases_per_day * days_needed = 256 := by
  sorry

end NUMINAMATH_CALUDE_vases_to_arrange_l3478_347877


namespace NUMINAMATH_CALUDE_triangle_area_proof_l3478_347809

-- Define the slopes of the two lines
def slope1 : ℝ := 3
def slope2 : ℝ := -1

-- Define the intersection point
def intersection_point : ℝ × ℝ := (5, 3)

-- Define the equation of the third line
def third_line (x y : ℝ) : Prop := x + y = 4

-- Define the area of the triangle
def triangle_area : ℝ := 4

-- Theorem statement
theorem triangle_area_proof :
  ∃ (A B C : ℝ × ℝ),
    -- A is on the line with slope1 and passes through intersection_point
    (A.2 - intersection_point.2 = slope1 * (A.1 - intersection_point.1)) ∧
    -- B is on the line with slope2 and passes through intersection_point
    (B.2 - intersection_point.2 = slope2 * (B.1 - intersection_point.1)) ∧
    -- C is on the third line
    third_line C.1 C.2 ∧
    -- The area of the triangle formed by A, B, and C is equal to triangle_area
    abs ((A.1 * (B.2 - C.2) + B.1 * (C.2 - A.2) + C.1 * (A.2 - B.2)) / 2) = triangle_area :=
sorry

end NUMINAMATH_CALUDE_triangle_area_proof_l3478_347809


namespace NUMINAMATH_CALUDE_manufacturing_expenses_calculation_l3478_347896

/-- Calculates the monthly manufacturing expenses for a textile firm. -/
def monthly_manufacturing_expenses (
  total_looms : ℕ)
  (aggregate_sales : ℕ)
  (establishment_charges : ℕ)
  (profit_decrease : ℕ) : ℕ :=
  let sales_per_loom := aggregate_sales / total_looms
  let expenses_per_loom := sales_per_loom - profit_decrease
  expenses_per_loom * total_looms

/-- Proves that the monthly manufacturing expenses are 150000 given the specified conditions. -/
theorem manufacturing_expenses_calculation :
  monthly_manufacturing_expenses 80 500000 75000 4375 = 150000 := by
  sorry

end NUMINAMATH_CALUDE_manufacturing_expenses_calculation_l3478_347896


namespace NUMINAMATH_CALUDE_x_minus_y_equals_twenty_l3478_347874

theorem x_minus_y_equals_twenty (x y : ℝ) 
  (h1 : x * (y + 2) = 100) 
  (h2 : y * (x + 2) = 60) : 
  x - y = 20 := by
sorry

end NUMINAMATH_CALUDE_x_minus_y_equals_twenty_l3478_347874


namespace NUMINAMATH_CALUDE_increasing_magnitude_l3478_347860

theorem increasing_magnitude (x : ℝ) (h : 1 < x ∧ x < 1.1) : x < x^x ∧ x^x < x^(x^x) := by
  sorry

end NUMINAMATH_CALUDE_increasing_magnitude_l3478_347860


namespace NUMINAMATH_CALUDE_disjunction_implies_conjunction_false_l3478_347812

theorem disjunction_implies_conjunction_false : 
  ¬(∀ (p q : Prop), (p ∨ q) → (p ∧ q)) := by
  sorry

end NUMINAMATH_CALUDE_disjunction_implies_conjunction_false_l3478_347812


namespace NUMINAMATH_CALUDE_evenBlueFaceCubesFor642Block_l3478_347840

/-- Represents a rectangular block with given dimensions -/
structure Block where
  length : Nat
  width : Nat
  height : Nat

/-- Counts the number of cubes with an even number of blue faces in a painted block -/
def evenBlueFaceCubes (b : Block) : Nat :=
  -- Implementation details are omitted
  sorry

/-- Theorem stating that a 6x4x2 inch block has 20 cubes with an even number of blue faces -/
theorem evenBlueFaceCubesFor642Block :
  evenBlueFaceCubes { length := 6, width := 4, height := 2 } = 20 := by
  sorry

end NUMINAMATH_CALUDE_evenBlueFaceCubesFor642Block_l3478_347840


namespace NUMINAMATH_CALUDE_circle_area_from_polar_equation_l3478_347863

-- Define the polar equation
def polar_equation (r θ : ℝ) : Prop :=
  r = 4 * Real.cos θ - 3 * Real.sin θ

-- State the theorem
theorem circle_area_from_polar_equation :
  ∃ (center : ℝ × ℝ) (radius : ℝ),
    (∀ (r θ : ℝ), polar_equation r θ ↔ 
      (r * Real.cos θ - center.1)^2 + (r * Real.sin θ - center.2)^2 = radius^2) ∧
    (π * radius^2 = 25 * π / 4) :=
sorry

end NUMINAMATH_CALUDE_circle_area_from_polar_equation_l3478_347863


namespace NUMINAMATH_CALUDE_estimated_area_is_10_l3478_347803

/-- The function representing the lower bound of the area -/
def f (x : ℝ) : ℝ := x^2 + 1

/-- The upper bound of the area -/
def upper_bound : ℝ := 5

/-- The total square area -/
def total_area : ℝ := 16

/-- The total number of experiments -/
def total_experiments : ℕ := 1000

/-- The number of points that fall within the desired area -/
def points_within : ℕ := 625

/-- Theorem stating that the estimated area is 10 -/
theorem estimated_area_is_10 : 
  (total_area * (points_within : ℝ) / total_experiments) = 10 := by
  sorry

end NUMINAMATH_CALUDE_estimated_area_is_10_l3478_347803


namespace NUMINAMATH_CALUDE_factorization_equality_l3478_347800

theorem factorization_equality (a b : ℝ) : a * b^2 - 3 * a = a * (b + Real.sqrt 3) * (b - Real.sqrt 3) := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l3478_347800


namespace NUMINAMATH_CALUDE_no_natural_divisible_by_49_l3478_347857

theorem no_natural_divisible_by_49 : ∀ n : ℕ, ¬(49 ∣ (n^2 + 5*n + 1)) := by sorry

end NUMINAMATH_CALUDE_no_natural_divisible_by_49_l3478_347857


namespace NUMINAMATH_CALUDE_area_of_square_II_l3478_347841

/-- Given a square I with diagonal √3ab, prove that the area of a square II 
    with three times the area of square I is 9(ab)²/2 -/
theorem area_of_square_II (a b : ℝ) (h : a > 0 ∧ b > 0) : 
  let diagonal_I := Real.sqrt 3 * a * b
  let area_I := (diagonal_I ^ 2) / 2
  let area_II := 3 * area_I
  area_II = 9 * (a * b) ^ 2 / 2 := by
sorry

end NUMINAMATH_CALUDE_area_of_square_II_l3478_347841


namespace NUMINAMATH_CALUDE_fraction_sum_cube_l3478_347822

theorem fraction_sum_cube (a b : ℝ) (h : (a + b) / (a - b) + (a - b) / (a + b) = 4) :
  (a^3 + b^3) / (a^3 - b^3) + (a^3 - b^3) / (a^3 + b^3) = 5/2 := by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_cube_l3478_347822


namespace NUMINAMATH_CALUDE_polygon_with_1080_degrees_has_8_sides_l3478_347813

/-- A polygon is a shape with a certain number of sides. -/
structure Polygon where
  sides : ℕ

/-- The sum of interior angles of a polygon. -/
def sumOfInteriorAngles (p : Polygon) : ℝ :=
  180 * (p.sides - 2)

/-- Theorem: A polygon with a sum of interior angles equal to 1080° has 8 sides. -/
theorem polygon_with_1080_degrees_has_8_sides :
  ∃ (p : Polygon), sumOfInteriorAngles p = 1080 → p.sides = 8 := by
  sorry

end NUMINAMATH_CALUDE_polygon_with_1080_degrees_has_8_sides_l3478_347813


namespace NUMINAMATH_CALUDE_number_thought_of_l3478_347864

theorem number_thought_of (x : ℚ) : (6 * x) / 2 - 5 = 25 → x = 10 := by
  sorry

end NUMINAMATH_CALUDE_number_thought_of_l3478_347864


namespace NUMINAMATH_CALUDE_hall_length_width_difference_l3478_347842

/-- Represents a rectangular hall with given properties -/
structure RectangularHall where
  length : ℝ
  width : ℝ
  area : ℝ
  width_half_length : width = length / 2
  area_eq : area = length * width

/-- Theorem stating the difference between length and width of the hall -/
theorem hall_length_width_difference (hall : RectangularHall) 
  (h_area : hall.area = 578) : hall.length - hall.width = 17 := by
  sorry

end NUMINAMATH_CALUDE_hall_length_width_difference_l3478_347842


namespace NUMINAMATH_CALUDE_race_distance_l3478_347836

theorem race_distance (t1 t2 combined_time : ℝ) (h1 : t1 = 21) (h2 : t2 = 24) 
  (h3 : combined_time = 75) : 
  let d := (5 * t1 + 5 * t2) / combined_time
  d = 3 := by sorry

end NUMINAMATH_CALUDE_race_distance_l3478_347836


namespace NUMINAMATH_CALUDE_square_sum_equals_twice_square_l3478_347888

theorem square_sum_equals_twice_square (a : ℝ) : a^2 + a^2 = 2 * a^2 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_equals_twice_square_l3478_347888


namespace NUMINAMATH_CALUDE_unique_base_number_exists_l3478_347815

/-- A number in base 2022 with digits 1 or 2 -/
def BaseNumber (n : ℕ) := {x : ℕ // ∀ d, d ∈ x.digits 2022 → d = 1 ∨ d = 2}

/-- The theorem statement -/
theorem unique_base_number_exists :
  ∃! (N : BaseNumber 1000), (N.val : ℤ) % (2^1000) = 0 := by
  sorry

end NUMINAMATH_CALUDE_unique_base_number_exists_l3478_347815


namespace NUMINAMATH_CALUDE_gcd_repeated_digits_l3478_347853

def repeat_digit (n : ℕ) : ℕ := n + 1000 * n + 1000000 * n

theorem gcd_repeated_digits :
  ∃ (g : ℕ), g > 0 ∧ 
  (∀ (n : ℕ), 100 ≤ n → n < 1000 → g ∣ repeat_digit n) ∧
  (∀ (d : ℕ), d > 0 → 
    (∀ (n : ℕ), 100 ≤ n → n < 1000 → d ∣ repeat_digit n) → 
    d ∣ g) ∧
  g = 1001001 := by
sorry

#eval 1001001

end NUMINAMATH_CALUDE_gcd_repeated_digits_l3478_347853


namespace NUMINAMATH_CALUDE_smallest_n_with_conditions_n_satisfies_conditions_l3478_347829

def has_digit (n : ℕ) (d : ℕ) : Prop :=
  ∃ k m : ℕ, n = 10 * k + d + 10 * m

def is_terminating_decimal (n : ℕ) : Prop :=
  ∃ a b : ℕ, n = 2^a * 5^b

theorem smallest_n_with_conditions :
  ∀ n : ℕ, n > 0 →
    (is_terminating_decimal n ∧ has_digit n 9 ∧ has_digit n 2) →
    n ≥ 524288 :=
by sorry

theorem n_satisfies_conditions :
  is_terminating_decimal 524288 ∧ has_digit 524288 9 ∧ has_digit 524288 2 :=
by sorry

end NUMINAMATH_CALUDE_smallest_n_with_conditions_n_satisfies_conditions_l3478_347829


namespace NUMINAMATH_CALUDE_inner_rectangle_side_length_l3478_347876

/-- Given a square with side length a and four congruent right triangles removed from its corners,
    this theorem proves the relationship between the original square's side length,
    the area removed, and the resulting inner rectangle's side length. -/
theorem inner_rectangle_side_length
  (a : ℝ)
  (h1 : a ≥ 24 * Real.sqrt 3)
  (h2 : 6 * (4 * Real.sqrt 3)^2 = 288) :
  a - 24 * Real.sqrt 3 = a - 6 * (4 * Real.sqrt 3) :=
by sorry

end NUMINAMATH_CALUDE_inner_rectangle_side_length_l3478_347876


namespace NUMINAMATH_CALUDE_total_players_on_ground_l3478_347824

theorem total_players_on_ground (cricket hockey football softball basketball volleyball netball rugby : ℕ) 
  (h1 : cricket = 35)
  (h2 : hockey = 28)
  (h3 : football = 33)
  (h4 : softball = 35)
  (h5 : basketball = 29)
  (h6 : volleyball = 32)
  (h7 : netball = 34)
  (h8 : rugby = 37) :
  cricket + hockey + football + softball + basketball + volleyball + netball + rugby = 263 := by
  sorry

end NUMINAMATH_CALUDE_total_players_on_ground_l3478_347824


namespace NUMINAMATH_CALUDE_marble_difference_l3478_347866

/-- The number of bags Mara has -/
def mara_bags : ℕ := 12

/-- The number of marbles in each of Mara's bags -/
def mara_marbles_per_bag : ℕ := 2

/-- The number of bags Markus has -/
def markus_bags : ℕ := 2

/-- The number of marbles in each of Markus's bags -/
def markus_marbles_per_bag : ℕ := 13

/-- The difference in the total number of marbles between Markus and Mara -/
theorem marble_difference : 
  markus_bags * markus_marbles_per_bag - mara_bags * mara_marbles_per_bag = 2 := by
  sorry

end NUMINAMATH_CALUDE_marble_difference_l3478_347866


namespace NUMINAMATH_CALUDE_inequality_proof_l3478_347801

theorem inequality_proof (b a : ℝ) : 
  (4 * b^2 * (b^3 - 1) - 3 * (1 - 2 * b^2) > 4 * (b^5 - 1)) ∧ 
  (a - a * |(-a^2 - 1)| < 1 - a^2 * (a - 1)) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3478_347801


namespace NUMINAMATH_CALUDE_miniou_circuit_nodes_l3478_347849

/-- Definition of a Miniou circuit -/
structure MiniouCircuit where
  nodes : ℕ
  wires : ℕ
  wire_connects_two_nodes : True
  at_most_one_wire_between_nodes : True
  three_wires_per_node : True

/-- Theorem: A Miniou circuit with 13788 wires has 9192 nodes -/
theorem miniou_circuit_nodes (c : MiniouCircuit) (h : c.wires = 13788) : c.nodes = 9192 := by
  sorry

end NUMINAMATH_CALUDE_miniou_circuit_nodes_l3478_347849


namespace NUMINAMATH_CALUDE_fibSeriesSum_l3478_347823

-- Define the Fibonacci sequence
def fib : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | (n + 2) => fib (n + 1) + fib n

-- Define the series sum
noncomputable def fibSeries : ℝ := ∑' n, (fib n : ℝ) / 7^n

-- Theorem statement
theorem fibSeriesSum : fibSeries = 49 / 287 := by sorry

end NUMINAMATH_CALUDE_fibSeriesSum_l3478_347823


namespace NUMINAMATH_CALUDE_number_ratio_l3478_347883

theorem number_ratio (x : ℚ) (h : 3 * (2 * x + 9) = 75) : x / (2 * x) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_number_ratio_l3478_347883


namespace NUMINAMATH_CALUDE_point_D_coordinates_l3478_347884

/-- A point in 2D space -/
structure Point where
  x : ℚ
  y : ℚ

/-- The distance between two points -/
def distance (p1 p2 : Point) : ℚ :=
  ((p1.x - p2.x)^2 + (p1.y - p2.y)^2).sqrt

/-- Checks if a point is on a line segment -/
def isOnSegment (p : Point) (a b : Point) : Prop :=
  distance a p + distance p b = distance a b

theorem point_D_coordinates :
  let X : Point := ⟨-2, 1⟩
  let Y : Point := ⟨4, 9⟩
  ∀ D : Point,
    isOnSegment D X Y →
    distance X D = 2 * distance Y D →
    D.x = 2 ∧ D.y = 19 / 3 := by
  sorry

end NUMINAMATH_CALUDE_point_D_coordinates_l3478_347884


namespace NUMINAMATH_CALUDE_smallest_divisible_by_18_30_50_l3478_347805

theorem smallest_divisible_by_18_30_50 : 
  ∀ n : ℕ, n > 0 ∧ 18 ∣ n ∧ 30 ∣ n ∧ 50 ∣ n → n ≥ 450 :=
by
  sorry

#check smallest_divisible_by_18_30_50

end NUMINAMATH_CALUDE_smallest_divisible_by_18_30_50_l3478_347805


namespace NUMINAMATH_CALUDE_jason_total_games_l3478_347889

/-- The number of games Jason attended this month -/
def games_this_month : ℕ := 11

/-- The number of games Jason attended last month -/
def games_last_month : ℕ := 17

/-- The number of games Jason plans to attend next month -/
def games_next_month : ℕ := 16

/-- The total number of games Jason will attend -/
def total_games : ℕ := games_this_month + games_last_month + games_next_month

theorem jason_total_games : total_games = 44 := by sorry

end NUMINAMATH_CALUDE_jason_total_games_l3478_347889


namespace NUMINAMATH_CALUDE_placement_count_no_restriction_placement_count_with_restriction_l3478_347871

/-- The number of booths in the exhibition room -/
def total_booths : ℕ := 9

/-- The number of exhibits to be displayed -/
def num_exhibits : ℕ := 3

/-- Calculates the number of ways to place exhibits under the given conditions -/
def calculate_placements (max_distance : ℕ) : ℕ :=
  sorry

/-- Theorem stating the number of placement options without distance restriction -/
theorem placement_count_no_restriction : calculate_placements total_booths = 60 :=
  sorry

/-- Theorem stating the number of placement options with distance restriction -/
theorem placement_count_with_restriction : calculate_placements 2 = 48 :=
  sorry

end NUMINAMATH_CALUDE_placement_count_no_restriction_placement_count_with_restriction_l3478_347871


namespace NUMINAMATH_CALUDE_wendi_chicken_count_l3478_347828

def chicken_count (initial : ℕ) : ℕ :=
  let step1 := initial + (initial / 2) ^ 2
  let step2 := step1 - 3
  let step3 := step2 + ((4 * step2 - 28) / 7)
  step3 + 3

theorem wendi_chicken_count : chicken_count 15 = 105 := by
  sorry

end NUMINAMATH_CALUDE_wendi_chicken_count_l3478_347828


namespace NUMINAMATH_CALUDE_dime_count_l3478_347848

/-- Given a collection of coins consisting of dimes and nickels, 
    this theorem proves the number of dimes given the total number 
    of coins and their total value. -/
theorem dime_count 
  (total_coins : ℕ) 
  (total_value : ℚ) 
  (h_total : total_coins = 36) 
  (h_value : total_value = 31/10) : 
  ∃ (dimes nickels : ℕ),
    dimes + nickels = total_coins ∧ 
    (dimes : ℚ) / 10 + (nickels : ℚ) / 20 = total_value ∧
    dimes = 26 := by
  sorry

end NUMINAMATH_CALUDE_dime_count_l3478_347848


namespace NUMINAMATH_CALUDE_image_of_3_4_preimage_of_1_neg6_l3478_347886

-- Define the mapping f
def f (p : ℝ × ℝ) : ℝ × ℝ := (p.1 + p.2, p.1 * p.2)

-- Theorem for the image of (3, 4)
theorem image_of_3_4 : f (3, 4) = (7, 12) := by sorry

-- Definition of preimage
def preimage (f : ℝ × ℝ → ℝ × ℝ) (y : ℝ × ℝ) : Set (ℝ × ℝ) :=
  {x | f x = y}

-- Theorem for the preimage of (1, -6)
theorem preimage_of_1_neg6 : preimage f (1, -6) = {(-2, 3), (3, -2)} := by sorry

end NUMINAMATH_CALUDE_image_of_3_4_preimage_of_1_neg6_l3478_347886


namespace NUMINAMATH_CALUDE_consecutive_product_sum_l3478_347897

theorem consecutive_product_sum (a b c : ℤ) : 
  b = a + 1 → c = b + 1 → a * b * c = 210 → a + b = 11 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_product_sum_l3478_347897


namespace NUMINAMATH_CALUDE_inverse_15_mod_1003_l3478_347870

theorem inverse_15_mod_1003 : ∃ x : ℤ, 0 ≤ x ∧ x < 1003 ∧ (15 * x) % 1003 = 1 :=
by
  use 937
  sorry

end NUMINAMATH_CALUDE_inverse_15_mod_1003_l3478_347870


namespace NUMINAMATH_CALUDE_correct_quotient_proof_l3478_347818

theorem correct_quotient_proof (dividend : ℕ) (mistaken_divisor correct_divisor mistaken_quotient : ℕ) 
  (h1 : mistaken_divisor = 12)
  (h2 : correct_divisor = 21)
  (h3 : mistaken_quotient = 42)
  (h4 : dividend = mistaken_divisor * mistaken_quotient)
  (h5 : dividend % correct_divisor = 0) :
  dividend / correct_divisor = 24 := by
sorry

end NUMINAMATH_CALUDE_correct_quotient_proof_l3478_347818


namespace NUMINAMATH_CALUDE_random_course_selection_probability_l3478_347854

theorem random_course_selection_probability 
  (courses : Finset String) 
  (h1 : courses.card = 4) 
  (h2 : courses.Nonempty) 
  (selected_course : String) 
  (h3 : selected_course ∈ courses) :
  (Finset.filter (· = selected_course) courses).card / courses.card = (1 : ℚ) / 4 :=
sorry

end NUMINAMATH_CALUDE_random_course_selection_probability_l3478_347854


namespace NUMINAMATH_CALUDE_cross_number_puzzle_l3478_347869

theorem cross_number_puzzle :
  ∃! (a1 a3 d1 d2 : ℕ),
    (100 ≤ a1 ∧ a1 < 1000) ∧
    (100 ≤ a3 ∧ a3 < 1000) ∧
    (100 ≤ d1 ∧ d1 < 1000) ∧
    (100 ≤ d2 ∧ d2 < 1000) ∧
    (∃ n : ℕ, a1 = n^2) ∧
    (∃ n : ℕ, a3 = n^4) ∧
    (∃ n : ℕ, d1 = 2 * n^5) ∧
    (∃ n : ℕ, d2 = n^3) ∧
    (a1 / 100 = 4) :=
by
  sorry

end NUMINAMATH_CALUDE_cross_number_puzzle_l3478_347869


namespace NUMINAMATH_CALUDE_power_between_n_and_2n_smallest_m_s_for_2_and_3_l3478_347832

theorem power_between_n_and_2n (s : ℕ) (hs : s > 1) :
  ∃ (m_s : ℕ), ∀ (n : ℕ), n ≥ m_s → ∃ (k : ℕ), n < k^s ∧ k^s < 2*n :=
by sorry

theorem smallest_m_s_for_2_and_3 :
  (∃ (m_2 : ℕ), ∀ (n : ℕ), n ≥ m_2 → ∃ (k : ℕ), n < k^2 ∧ k^2 < 2*n) ∧
  (∃ (m_3 : ℕ), ∀ (n : ℕ), n ≥ m_3 → ∃ (k : ℕ), n < k^3 ∧ k^3 < 2*n) ∧
  (∀ (m_2' : ℕ), m_2' < 5 → ∃ (n : ℕ), n ≥ m_2' ∧ ∀ (k : ℕ), n ≥ k^2 ∨ k^2 ≥ 2*n) ∧
  (∀ (m_3' : ℕ), m_3' < 33 → ∃ (n : ℕ), n ≥ m_3' ∧ ∀ (k : ℕ), n ≥ k^3 ∨ k^3 ≥ 2*n) :=
by sorry

end NUMINAMATH_CALUDE_power_between_n_and_2n_smallest_m_s_for_2_and_3_l3478_347832


namespace NUMINAMATH_CALUDE_parallel_lines_k_value_l3478_347861

/-- Two lines are parallel if and only if their slopes are equal -/
axiom parallel_lines_equal_slopes {m1 m2 b1 b2 : ℝ} : 
  (∀ x y : ℝ, y = m1 * x + b1 ↔ y = m2 * x + b2) ↔ m1 = m2

/-- The problem statement -/
theorem parallel_lines_k_value :
  (∀ x y : ℝ, y = 15 * x + 5 ↔ y = (5 * k) * x - 7) → k = 3 := by
  sorry

end NUMINAMATH_CALUDE_parallel_lines_k_value_l3478_347861


namespace NUMINAMATH_CALUDE_problem_1_l3478_347837

theorem problem_1 : 6 - (-12) / (-3) = 2 := by
  sorry

end NUMINAMATH_CALUDE_problem_1_l3478_347837


namespace NUMINAMATH_CALUDE_room_height_from_curtain_l3478_347808

/-- The height of a room from the curtain rod to the floor, given curtain length and pooling material. -/
theorem room_height_from_curtain (curtain_length : ℕ) (pooling_material : ℕ) : 
  curtain_length = 101 ∧ pooling_material = 5 → curtain_length - pooling_material = 96 := by
  sorry

#check room_height_from_curtain

end NUMINAMATH_CALUDE_room_height_from_curtain_l3478_347808


namespace NUMINAMATH_CALUDE_valid_colorings_6x6_grid_l3478_347881

/-- Recursive function to calculate the number of valid colorings for an n×n grid -/
def f : ℕ → ℕ
| 0 => 0
| 1 => 0
| 2 => 1
| n + 2 => n * (n + 1) * f (n + 1) + (n * (n + 1)^2) / 2 * f n

/-- The size of the grid -/
def gridSize : ℕ := 6

/-- The number of red squares required in each row and column -/
def redSquaresPerLine : ℕ := 2

/-- Theorem stating the number of valid colorings for a 6×6 grid -/
theorem valid_colorings_6x6_grid :
  f gridSize = 67950 :=
sorry

end NUMINAMATH_CALUDE_valid_colorings_6x6_grid_l3478_347881


namespace NUMINAMATH_CALUDE_max_withdrawal_theorem_l3478_347868

/-- Represents the possible transactions -/
inductive Transaction
| withdraw : Transaction
| deposit : Transaction

/-- Represents the bank account -/
structure BankAccount where
  balance : ℕ

/-- Applies a transaction to the bank account -/
def applyTransaction (account : BankAccount) (t : Transaction) : BankAccount :=
  match t with
  | Transaction.withdraw => ⟨if account.balance ≥ 300 then account.balance - 300 else account.balance⟩
  | Transaction.deposit => ⟨account.balance + 198⟩

/-- Checks if a sequence of transactions is valid -/
def isValidSequence (initial : ℕ) (transactions : List Transaction) : Prop :=
  let finalAccount := transactions.foldl applyTransaction ⟨initial⟩
  finalAccount.balance ≥ 0

/-- The maximum amount that can be withdrawn -/
def maxWithdrawal (initial : ℕ) : ℕ :=
  initial - (initial % 6)

/-- Theorem stating the maximum withdrawal amount -/
theorem max_withdrawal_theorem (initial : ℕ) :
  initial = 500 →
  maxWithdrawal initial = 300 ∧
  ∃ (transactions : List Transaction), isValidSequence initial transactions ∧
    (initial - (transactions.foldl applyTransaction ⟨initial⟩).balance = maxWithdrawal initial) :=
by sorry

#check max_withdrawal_theorem

end NUMINAMATH_CALUDE_max_withdrawal_theorem_l3478_347868


namespace NUMINAMATH_CALUDE_infinitely_many_planes_through_collinear_points_l3478_347867

/-- Three distinct points on a line -/
structure ThreeCollinearPoints (V : Type*) [NormedAddCommGroup V] [NormedSpace ℝ V] :=
  (p₁ p₂ p₃ : V)
  (distinct : p₁ ≠ p₂ ∧ p₁ ≠ p₃ ∧ p₂ ≠ p₃)
  (collinear : ∃ (t₁ t₂ : ℝ), p₃ - p₁ = t₁ • (p₂ - p₁) ∧ p₂ - p₁ = t₂ • (p₃ - p₁))

/-- A plane passing through three points -/
def Plane (V : Type*) [NormedAddCommGroup V] [NormedSpace ℝ V] (p₁ p₂ p₃ : V) :=
  {x : V | ∃ (a b c : ℝ), x = a • p₁ + b • p₂ + c • p₃ ∧ a + b + c = 1}

/-- Theorem: There are infinitely many planes passing through three collinear points -/
theorem infinitely_many_planes_through_collinear_points
  {V : Type*} [NormedAddCommGroup V] [NormedSpace ℝ V]
  (points : ThreeCollinearPoints V) :
  ∃ (planes : Set (Plane V points.p₁ points.p₂ points.p₃)), Infinite planes :=
sorry

end NUMINAMATH_CALUDE_infinitely_many_planes_through_collinear_points_l3478_347867


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l3478_347825

/-- An arithmetic sequence with sum of first n terms S_n -/
structure ArithmeticSequence where
  S : ℕ → ℝ

/-- Theorem: For an arithmetic sequence with S_5 = 10 and S_10 = 30, S_15 = 60 -/
theorem arithmetic_sequence_sum (a : ArithmeticSequence) 
  (h1 : a.S 5 = 10) 
  (h2 : a.S 10 = 30) : 
  a.S 15 = 60 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l3478_347825


namespace NUMINAMATH_CALUDE_equation_solution_l3478_347806

theorem equation_solution : ∃ y : ℝ, (2 / y + 3 / y / (6 / y) = 1.5) ∧ y = 2 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3478_347806


namespace NUMINAMATH_CALUDE_log_4_64_sqrt_4_l3478_347835

-- Define the logarithm function
noncomputable def log (base : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log base

-- State the theorem
theorem log_4_64_sqrt_4 : log 4 (64 * Real.sqrt 4) = 7/2 := by
  sorry

end NUMINAMATH_CALUDE_log_4_64_sqrt_4_l3478_347835


namespace NUMINAMATH_CALUDE_min_value_theorem_l3478_347844

theorem min_value_theorem (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + y = 20) :
  (2 / x) + (3 / y) ≥ 1 ∧ ((2 / x) + (3 / y) = 1 ↔ x = 12 ∧ y = 8) := by
  sorry

end NUMINAMATH_CALUDE_min_value_theorem_l3478_347844


namespace NUMINAMATH_CALUDE_min_value_expression_l3478_347895

theorem min_value_expression (p q r : ℝ) (hp : p > 0) (hq : q > 0) (hr : r > 0) :
  (5 * r) / (3 * p + 2 * q) + (5 * p) / (2 * q + 3 * r) + (2 * q) / (p + r) ≥ 19 / 4 :=
sorry

end NUMINAMATH_CALUDE_min_value_expression_l3478_347895


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l3478_347807

def set_A : Set ℝ := {x | |x - 1| < 3}
def set_B : Set ℝ := {x | (x - 1) / (x - 5) < 0}

theorem intersection_of_A_and_B : 
  set_A ∩ set_B = {x : ℝ | 1 < x ∧ x < 4} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l3478_347807


namespace NUMINAMATH_CALUDE_sin_cos_cube_sum_l3478_347847

theorem sin_cos_cube_sum (θ : ℝ) (h : 4 * Real.sin θ * Real.cos θ - 5 * Real.sin θ - 5 * Real.cos θ - 1 = 0) :
  Real.sin θ ^ 3 + Real.cos θ ^ 3 = -11/16 := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_cube_sum_l3478_347847


namespace NUMINAMATH_CALUDE_mike_marbles_l3478_347890

/-- Calculates the number of marbles Mike has after giving some to Sam. -/
def marblesLeft (initial : ℕ) (given : ℕ) : ℕ :=
  initial - given

/-- Proves that Mike has 4 marbles left after giving 4 out of his initial 8 marbles to Sam. -/
theorem mike_marbles : marblesLeft 8 4 = 4 := by
  sorry

end NUMINAMATH_CALUDE_mike_marbles_l3478_347890


namespace NUMINAMATH_CALUDE_cube_sum_ge_mixed_product_cube_sum_ge_weighted_square_sum_product_l3478_347827

-- Problem 1
theorem cube_sum_ge_mixed_product (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  x^3 + y^3 ≥ x^2*y + x*y^2 := by sorry

-- Problem 2
theorem cube_sum_ge_weighted_square_sum_product (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  a^3 + b^3 + c^3 ≥ (1/3) * (a^2 + b^2 + c^2) * (a + b + c) := by sorry

end NUMINAMATH_CALUDE_cube_sum_ge_mixed_product_cube_sum_ge_weighted_square_sum_product_l3478_347827


namespace NUMINAMATH_CALUDE_min_length_roots_l3478_347851

/-- Given a quadratic function f(x) = a x^2 + (16 - a^3) x - 16 a^2, where a > 0,
    the minimum length of the line segment connecting its roots is 12. -/
theorem min_length_roots (a : ℝ) (ha : a > 0) :
  let f := fun x : ℝ => a * x^2 + (16 - a^3) * x - 16 * a^2
  let roots := {x : ℝ | f x = 0}
  let length := fun (r₁ r₂ : ℝ) => |r₁ - r₂|
  ∃ (r₁ r₂ : ℝ), r₁ ∈ roots ∧ r₂ ∈ roots ∧ r₁ ≠ r₂ ∧
    ∀ (s₁ s₂ : ℝ), s₁ ∈ roots → s₂ ∈ roots → s₁ ≠ s₂ →
      length r₁ r₂ ≤ length s₁ s₂ ∧ length r₁ r₂ = 12 :=
by sorry


end NUMINAMATH_CALUDE_min_length_roots_l3478_347851


namespace NUMINAMATH_CALUDE_fraction_zero_implies_x_equals_two_l3478_347859

theorem fraction_zero_implies_x_equals_two (x : ℝ) : 
  (x^2 - 4)/(x + 2) = 0 → x = 2 :=
by sorry

end NUMINAMATH_CALUDE_fraction_zero_implies_x_equals_two_l3478_347859


namespace NUMINAMATH_CALUDE_group_purchase_equations_l3478_347873

/-- Represents a group purchase scenario -/
structure GroupPurchase where
  x : ℕ  -- number of people
  y : ℕ  -- price of the item

/-- Defines the conditions of the group purchase -/
def validGroupPurchase (gp : GroupPurchase) : Prop :=
  (9 * gp.x - gp.y = 4) ∧ (gp.y - 6 * gp.x = 5)

/-- Theorem stating that the given system of equations correctly represents the group purchase scenario -/
theorem group_purchase_equations (gp : GroupPurchase) : 
  validGroupPurchase gp ↔ (9 * gp.x - gp.y = 4 ∧ gp.y - 6 * gp.x = 5) :=
sorry

end NUMINAMATH_CALUDE_group_purchase_equations_l3478_347873


namespace NUMINAMATH_CALUDE_ab_value_l3478_347834

theorem ab_value (a b : ℝ) (h1 : a - b = 3) (h2 : a^2 + b^2 = 29) : a * b = 10 := by
  sorry

end NUMINAMATH_CALUDE_ab_value_l3478_347834


namespace NUMINAMATH_CALUDE_infinite_sum_theorem_l3478_347894

theorem infinite_sum_theorem (s : ℝ) (hs : s > 0) (heq : s^3 - 3/4 * s + 2 = 0) :
  ∑' n, (n + 1) * s^(2*n + 2) = 16/9 := by
sorry

end NUMINAMATH_CALUDE_infinite_sum_theorem_l3478_347894


namespace NUMINAMATH_CALUDE_smaller_two_digit_factor_l3478_347878

theorem smaller_two_digit_factor (a b : ℕ) : 
  10 ≤ a ∧ a < 100 ∧ 
  10 ≤ b ∧ b < 100 ∧ 
  a * b = 5488 → 
  min a b = 56 := by
sorry

end NUMINAMATH_CALUDE_smaller_two_digit_factor_l3478_347878


namespace NUMINAMATH_CALUDE_sum_upper_bound_l3478_347826

theorem sum_upper_bound (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (h : a^3 + b^3 = 2) : 
  a + b ≤ 2 := by
sorry

end NUMINAMATH_CALUDE_sum_upper_bound_l3478_347826


namespace NUMINAMATH_CALUDE_parabola_no_y_intercepts_l3478_347839

theorem parabola_no_y_intercepts :
  ¬∃ y : ℝ, 0 = 3 * y^2 - 4 * y + 5 := by
sorry

end NUMINAMATH_CALUDE_parabola_no_y_intercepts_l3478_347839


namespace NUMINAMATH_CALUDE_tricycle_wheels_l3478_347856

theorem tricycle_wheels (num_bicycles num_tricycles bicycle_wheels total_wheels : ℕ) 
  (h1 : num_bicycles = 24)
  (h2 : num_tricycles = 14)
  (h3 : bicycle_wheels = 2)
  (h4 : total_wheels = 90)
  : (total_wheels - num_bicycles * bicycle_wheels) / num_tricycles = 3 := by
  sorry

end NUMINAMATH_CALUDE_tricycle_wheels_l3478_347856


namespace NUMINAMATH_CALUDE_donut_calculation_l3478_347885

def total_donuts (initial_friends : ℕ) (additional_friends : ℕ) (donuts_per_friend : ℕ) (extra_donuts : ℕ) : ℕ :=
  let total_friends := initial_friends + additional_friends
  let donuts_for_friends := total_friends * (donuts_per_friend + extra_donuts)
  let donuts_for_andrew := donuts_per_friend + extra_donuts
  donuts_for_friends + donuts_for_andrew

theorem donut_calculation :
  total_donuts 2 2 3 1 = 20 := by
  sorry

end NUMINAMATH_CALUDE_donut_calculation_l3478_347885


namespace NUMINAMATH_CALUDE_ashleys_friends_ages_sum_l3478_347893

theorem ashleys_friends_ages_sum :
  ∀ (a b c d : ℕ),
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d →
    0 < a ∧ a < 10 →
    0 < b ∧ b < 10 →
    0 < c ∧ c < 10 →
    0 < d ∧ d < 10 →
    (a * b = 36 ∧ c * d = 30) ∨ (a * c = 36 ∧ b * d = 30) ∨ (a * d = 36 ∧ b * c = 30) →
    a + b + c + d = 24 :=
by sorry

end NUMINAMATH_CALUDE_ashleys_friends_ages_sum_l3478_347893


namespace NUMINAMATH_CALUDE_glove_at_midpoint_l3478_347830

/-- Represents the escalator system and Semyon's movement -/
structure EscalatorSystem where
  /-- The speed of both escalators -/
  escalator_speed : ℝ
  /-- Semyon's walking speed -/
  semyon_speed : ℝ
  /-- The total height of the escalators -/
  total_height : ℝ

/-- Theorem stating that the glove will be at the midpoint when Semyon reaches the top -/
theorem glove_at_midpoint (system : EscalatorSystem)
  (h1 : system.escalator_speed > 0)
  (h2 : system.semyon_speed = system.escalator_speed)
  (h3 : system.total_height > 0) :
  let time_to_top := system.total_height / (2 * system.escalator_speed)
  let glove_position := system.escalator_speed * time_to_top
  glove_position = system.total_height / 2 := by
  sorry


end NUMINAMATH_CALUDE_glove_at_midpoint_l3478_347830


namespace NUMINAMATH_CALUDE_greatest_solution_of_equation_l3478_347855

theorem greatest_solution_of_equation (x : ℝ) : 
  (((5*x - 20)/(4*x - 5))^2 + ((5*x - 20)/(4*x - 5)) = 20) → x ≤ 9/5 :=
by sorry

end NUMINAMATH_CALUDE_greatest_solution_of_equation_l3478_347855


namespace NUMINAMATH_CALUDE_min_value_of_expression_equality_condition_l3478_347862

theorem min_value_of_expression (x : ℝ) : 
  Real.sqrt (x^2 + (2 - x)^2) + Real.sqrt ((2 - x)^2 + (2 + x)^2) ≥ 2 * Real.sqrt 5 :=
by sorry

theorem equality_condition :
  Real.sqrt ((4/3)^2 + (2 - 4/3)^2) + Real.sqrt ((2 - 4/3)^2 + (2 + 4/3)^2) = 2 * Real.sqrt 5 :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_expression_equality_condition_l3478_347862


namespace NUMINAMATH_CALUDE_set_operations_l3478_347872

def A : Set ℝ := {x | 2 < x ∧ x < 6}
def B (m : ℝ) : Set ℝ := {x | m + 1 < x ∧ x < 2*m}

theorem set_operations (m : ℝ) :
  (m = 2 → A ∪ B m = A) ∧
  (B m ⊆ A ↔ m ≤ 3) ∧
  (B m ≠ ∅ ∧ A ∩ B m = ∅ ↔ m ≥ 5) :=
by sorry

end NUMINAMATH_CALUDE_set_operations_l3478_347872


namespace NUMINAMATH_CALUDE_intersection_union_theorem_l3478_347804

-- Define the sets A, B, and C
def A (a : ℝ) : Set ℝ := {x | x^2 + a*x + 12 = 0}
def B (b : ℝ) : Set ℝ := {x | x^2 + 3*x + 2*b = 0}
def C : Set ℝ := {2, -3}

-- State the theorem
theorem intersection_union_theorem (a b : ℝ) :
  (A a ∩ B b = {2}) →
  (A a = {2, 6}) →
  (B b = {-5, 2}) →
  ((A a ∪ B b) ∩ C = {2}) :=
by sorry

end NUMINAMATH_CALUDE_intersection_union_theorem_l3478_347804


namespace NUMINAMATH_CALUDE_two_sixty_million_scientific_notation_l3478_347880

/-- Scientific notation representation of a number -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  valid : 1 ≤ coefficient ∧ coefficient < 10

/-- Definition of 260 million -/
def two_hundred_sixty_million : ℕ := 260000000

/-- Theorem stating that 260 million is equal to 2.6 × 10^8 in scientific notation -/
theorem two_sixty_million_scientific_notation :
  ∃ (sn : ScientificNotation), (sn.coefficient * (10 : ℝ) ^ sn.exponent) = two_hundred_sixty_million :=
sorry

end NUMINAMATH_CALUDE_two_sixty_million_scientific_notation_l3478_347880


namespace NUMINAMATH_CALUDE_least_common_denominator_sum_l3478_347879

theorem least_common_denominator_sum (a b c d e : ℕ) 
  (ha : a = 4) (hb : b = 5) (hc : c = 6) (hd : d = 7) (he : e = 8) :
  Nat.lcm a (Nat.lcm b (Nat.lcm c (Nat.lcm d e))) = 840 := by
  sorry

end NUMINAMATH_CALUDE_least_common_denominator_sum_l3478_347879


namespace NUMINAMATH_CALUDE_complex_sum_equals_one_l3478_347810

-- Define the complex number i
def i : ℂ := Complex.I

-- State the theorem
theorem complex_sum_equals_one :
  (i + i^3)^100 + (i + i^2 + i^3 + i^4 + i^5)^120 = 1 :=
by
  sorry

end NUMINAMATH_CALUDE_complex_sum_equals_one_l3478_347810


namespace NUMINAMATH_CALUDE_base_7_to_base_10_l3478_347850

-- Define the base-7 number 435₇
def base_7_435 : ℕ := 4 * 7^2 + 3 * 7 + 5

-- Define the function to convert a three-digit base-10 number to its digits
def to_digits (n : ℕ) : ℕ × ℕ × ℕ :=
  let hundreds := n / 100
  let tens := (n / 10) % 10
  let ones := n % 10
  (hundreds, tens, ones)

-- Theorem statement
theorem base_7_to_base_10 :
  ∃ (c d : ℕ), c < 10 ∧ d < 10 ∧ base_7_435 = 300 + 10 * c + d →
  (c * d : ℚ) / 18 = 2 / 9 := by sorry

end NUMINAMATH_CALUDE_base_7_to_base_10_l3478_347850


namespace NUMINAMATH_CALUDE_fabric_needed_calculation_l3478_347817

/-- Calculates the additional fabric needed for dresses -/
def additional_fabric_needed (yards_per_dress : Float) (dresses : Nat) (available : Float) : Float :=
  yards_per_dress * dresses.toFloat * 3 - available

theorem fabric_needed_calculation (floral_yards_per_dress : Float) 
                                  (striped_yards_per_dress : Float)
                                  (polka_dot_yards_per_dress : Float)
                                  (floral_available : Float)
                                  (striped_available : Float)
                                  (polka_dot_available : Float) :
  floral_yards_per_dress = 5.25 →
  striped_yards_per_dress = 6.75 →
  polka_dot_yards_per_dress = 7.15 →
  floral_available = 12 →
  striped_available = 6 →
  polka_dot_available = 15 →
  additional_fabric_needed floral_yards_per_dress 2 floral_available = 19.5 ∧
  additional_fabric_needed striped_yards_per_dress 2 striped_available = 34.5 ∧
  additional_fabric_needed polka_dot_yards_per_dress 2 polka_dot_available = 27.9 :=
by sorry

end NUMINAMATH_CALUDE_fabric_needed_calculation_l3478_347817


namespace NUMINAMATH_CALUDE_sqrt_difference_equality_l3478_347899

theorem sqrt_difference_equality : Real.sqrt (49 + 121) - Real.sqrt (36 - 9) = Real.sqrt 170 - Real.sqrt 27 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_difference_equality_l3478_347899


namespace NUMINAMATH_CALUDE_parallel_vectors_x_value_l3478_347838

def vector_a : ℝ × ℝ := (1, 2)
def vector_b (x : ℝ) : ℝ × ℝ := (x, -4)

theorem parallel_vectors_x_value :
  (∃ (k : ℝ), k ≠ 0 ∧ vector_b x = k • vector_a) → x = -2 :=
by sorry

end NUMINAMATH_CALUDE_parallel_vectors_x_value_l3478_347838


namespace NUMINAMATH_CALUDE_value_of_b_l3478_347875

theorem value_of_b (a b : ℝ) (eq1 : 3 * a + 1 = 1) (eq2 : b - a = 2) : b = 2 := by
  sorry

end NUMINAMATH_CALUDE_value_of_b_l3478_347875


namespace NUMINAMATH_CALUDE_twenty_fifth_triangular_number_l3478_347887

/-- Definition of triangular number -/
def triangular_number (n : ℕ) : ℕ := n * (n + 1) / 2

/-- The 25th triangular number is 325 -/
theorem twenty_fifth_triangular_number : triangular_number 25 = 325 := by
  sorry

end NUMINAMATH_CALUDE_twenty_fifth_triangular_number_l3478_347887


namespace NUMINAMATH_CALUDE_mary_baseball_cards_l3478_347843

theorem mary_baseball_cards 
  (initial_cards : ℕ) 
  (torn_cards : ℕ) 
  (cards_from_fred : ℕ) 
  (total_cards : ℕ) 
  (h1 : initial_cards = 18) 
  (h2 : torn_cards = 8) 
  (h3 : cards_from_fred = 26) 
  (h4 : total_cards = 84) : 
  total_cards - (initial_cards - torn_cards + cards_from_fred) = 48 := by
  sorry

end NUMINAMATH_CALUDE_mary_baseball_cards_l3478_347843


namespace NUMINAMATH_CALUDE_shopping_mall_problem_l3478_347820

/-- Shopping mall goods purchasing and selling problem -/
theorem shopping_mall_problem 
  (cost_A_1_B_2 : ℝ) 
  (cost_A_3_B_2 : ℝ) 
  (sell_price_A : ℝ) 
  (sell_price_B : ℝ) 
  (total_units : ℕ) 
  (profit_lower : ℝ) 
  (profit_upper : ℝ) 
  (planned_units : ℕ) 
  (actual_profit : ℝ)
  (h1 : cost_A_1_B_2 = 320)
  (h2 : cost_A_3_B_2 = 520)
  (h3 : sell_price_A = 120)
  (h4 : sell_price_B = 140)
  (h5 : total_units = 50)
  (h6 : profit_lower = 1350)
  (h7 : profit_upper = 1375)
  (h8 : planned_units = 46)
  (h9 : actual_profit = 1220) :
  ∃ (cost_A cost_B : ℝ) (m : ℕ) (b : ℕ),
    cost_A = 100 ∧ 
    cost_B = 110 ∧ 
    13 ≤ m ∧ m ≤ 15 ∧
    b ≥ 32 := by sorry

end NUMINAMATH_CALUDE_shopping_mall_problem_l3478_347820


namespace NUMINAMATH_CALUDE_algorithm_output_l3478_347831

def algorithm (x y : Int) : (Int × Int) :=
  let x' := if x < 0 then y + 3 else x
  let y' := if x < 0 then y else y - 3
  (x' - y', y' + x')

theorem algorithm_output : algorithm (-5) 15 = (3, 33) := by
  sorry

end NUMINAMATH_CALUDE_algorithm_output_l3478_347831


namespace NUMINAMATH_CALUDE_sum_of_digits_of_sum_f_equals_8064_l3478_347802

-- Define the function f
def f (k : ℕ) : ℕ :=
  -- The smallest positive integer not written on the blackboard
  -- after the process described in the problem
  sorry

-- Define the sum of f(2k) from k=1 to 1008
def sum_f : ℕ :=
  (Finset.range 1008).sum (λ k => f (2 * (k + 1)))

-- Define a function to calculate the sum of digits of a number
def sum_of_digits (n : ℕ) : ℕ :=
  sorry

-- The main theorem
theorem sum_of_digits_of_sum_f_equals_8064 :
  sum_of_digits sum_f = 8064 :=
sorry

end NUMINAMATH_CALUDE_sum_of_digits_of_sum_f_equals_8064_l3478_347802


namespace NUMINAMATH_CALUDE_circle_sum_l3478_347852

theorem circle_sum (square circle : ℚ) 
  (eq1 : 2 * square + 3 * circle = 26)
  (eq2 : 3 * square + 2 * circle = 23) :
  4 * circle = 128 / 5 := by
sorry

end NUMINAMATH_CALUDE_circle_sum_l3478_347852


namespace NUMINAMATH_CALUDE_profit_percentage_calculation_l3478_347858

/-- Calculate the profit percentage given the selling price and profit -/
theorem profit_percentage_calculation (selling_price profit : ℝ) :
  selling_price = 850 ∧ profit = 205 →
  abs ((profit / (selling_price - profit)) * 100 - 31.78) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_profit_percentage_calculation_l3478_347858


namespace NUMINAMATH_CALUDE_cube_number_sum_l3478_347816

theorem cube_number_sum :
  ∀ (a b c d e f : ℕ),
  -- The numbers are consecutive whole numbers between 15 and 20
  15 ≤ a ∧ a < b ∧ b < c ∧ c < d ∧ d < e ∧ e < f ∧ f ≤ 20 →
  -- The sum of opposite faces is equal
  a + f = b + e ∧ b + e = c + d →
  -- The middle number in the range is the largest on one face
  (d = 18 ∨ c = 18) →
  -- The sum of all numbers is 105
  a + b + c + d + e + f = 105 :=
by sorry

end NUMINAMATH_CALUDE_cube_number_sum_l3478_347816


namespace NUMINAMATH_CALUDE_bryden_receives_45_dollars_l3478_347891

/-- The face value of a quarter in dollars -/
def quarter_value : ℚ := 1/4

/-- The number of quarters Bryden has -/
def bryden_quarters : ℕ := 6

/-- The collector's offer as a percentage of face value -/
def collector_offer_percent : ℕ := 3000

/-- Calculate the amount Bryden will receive for his quarters -/
def bryden_received : ℚ :=
  (quarter_value * bryden_quarters) * (collector_offer_percent / 100)

theorem bryden_receives_45_dollars :
  bryden_received = 45 := by sorry

end NUMINAMATH_CALUDE_bryden_receives_45_dollars_l3478_347891


namespace NUMINAMATH_CALUDE_min_cakes_to_recover_investment_l3478_347892

def investment : ℕ := 8000
def revenue_per_cake : ℕ := 15
def expense_per_cake : ℕ := 5

theorem min_cakes_to_recover_investment :
  ∀ n : ℕ, n * (revenue_per_cake - expense_per_cake) ≥ investment → n ≥ 800 :=
by sorry

end NUMINAMATH_CALUDE_min_cakes_to_recover_investment_l3478_347892


namespace NUMINAMATH_CALUDE_triangle_perimeter_l3478_347845

theorem triangle_perimeter (r : ℝ) (A : ℝ) (p : ℝ) :
  r = 3.5 →
  A = 56 →
  A = r * (p / 2) →
  p = 32 := by
sorry

end NUMINAMATH_CALUDE_triangle_perimeter_l3478_347845


namespace NUMINAMATH_CALUDE_book_pages_l3478_347819

/-- The number of pages Mrs. Hilt read -/
def pages_read : ℕ := 11

/-- The number of pages Mrs. Hilt has left to read -/
def pages_left : ℕ := 6

/-- The total number of pages in the book -/
def total_pages : ℕ := pages_read + pages_left

theorem book_pages : total_pages = 17 := by sorry

end NUMINAMATH_CALUDE_book_pages_l3478_347819


namespace NUMINAMATH_CALUDE_max_sequence_length_l3478_347846

theorem max_sequence_length (x : ℕ → ℕ) (n : ℕ) : 
  (∀ k, k < n - 1 → x k < x (k + 1)) →
  (∀ k, k ≤ n - 2 → x k ∣ x (k + 2)) →
  x n = 1000 →
  n ≤ 13 :=
sorry

end NUMINAMATH_CALUDE_max_sequence_length_l3478_347846


namespace NUMINAMATH_CALUDE_a_range_l3478_347811

-- Define propositions p and q as functions of a
def p (a : ℝ) : Prop := 0 < a ∧ a < 6

def q (a : ℝ) : Prop := a ≥ 5 ∨ a ≤ 1

-- Define the range of a
def range_a (a : ℝ) : Prop := a ≥ 6 ∨ a ≤ 0 ∨ (1 < a ∧ a < 5)

theorem a_range :
  (∀ a : ℝ, (p a ∨ q a)) ∧ (∀ a : ℝ, ¬(p a ∧ q a)) →
  ∀ a : ℝ, range_a a ↔ (p a ∨ q a) :=
by sorry

end NUMINAMATH_CALUDE_a_range_l3478_347811


namespace NUMINAMATH_CALUDE_total_cars_theorem_l3478_347833

/-- Calculates the total number of non-defective cars produced by two factories over a week --/
def total_non_defective_cars (factory_a_monday : ℕ) : ℕ :=
  let factory_a_production := [
    factory_a_monday,
    (factory_a_monday * 2 * 95) / 100,  -- Tuesday with 5% defect rate
    factory_a_monday * 4,
    factory_a_monday * 8,
    factory_a_monday * 16
  ]
  let factory_b_production := [
    factory_a_monday * 2,
    factory_a_monday * 4,
    factory_a_monday * 8,
    (factory_a_monday * 16 * 97) / 100,  -- Thursday with 3% defect rate
    factory_a_monday * 32
  ]
  (factory_a_production.sum + factory_b_production.sum)

/-- Theorem stating the total number of non-defective cars produced --/
theorem total_cars_theorem : total_non_defective_cars 60 = 5545 := by
  sorry


end NUMINAMATH_CALUDE_total_cars_theorem_l3478_347833


namespace NUMINAMATH_CALUDE_quadratic_inequality_equivalence_l3478_347865

theorem quadratic_inequality_equivalence :
  ∀ x : ℝ, x * (2 * x + 3) < 2 ↔ x ∈ Set.Ioo (-2 : ℝ) (1/2 : ℝ) := by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_equivalence_l3478_347865


namespace NUMINAMATH_CALUDE_regular_soda_count_l3478_347898

/-- The number of bottles of diet soda -/
def diet_soda : ℕ := 60

/-- The number of bottles of lite soda -/
def lite_soda : ℕ := 60

/-- The difference between regular and diet soda bottles -/
def regular_diet_difference : ℕ := 21

/-- The number of bottles of regular soda -/
def regular_soda : ℕ := diet_soda + regular_diet_difference

theorem regular_soda_count : regular_soda = 81 := by
  sorry

end NUMINAMATH_CALUDE_regular_soda_count_l3478_347898
