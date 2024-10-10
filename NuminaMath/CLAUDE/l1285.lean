import Mathlib

namespace valid_arrangements_count_l1285_128537

/-- Represents a person in the arrangement -/
inductive Person
| Man (n : Fin 4)
| Woman (n : Fin 4)

/-- A circular arrangement of people -/
def CircularArrangement := List Person

/-- Checks if two people can be adjacent in the arrangement -/
def canBeAdjacent (p1 p2 : Person) : Bool :=
  match p1, p2 with
  | Person.Man _, Person.Woman _ => true
  | Person.Woman _, Person.Man _ => true
  | _, _ => false

/-- Checks if a circular arrangement is valid -/
def isValidArrangement (arr : CircularArrangement) : Bool :=
  arr.length = 8 ∧
  arr.all (fun p => match p with
    | Person.Man n => n.val < 4
    | Person.Woman n => n.val < 4) ∧
  (List.zip arr (arr.rotateLeft 1)).all (fun (p1, p2) => canBeAdjacent p1 p2) ∧
  (List.zip arr (arr.rotateLeft 1)).all (fun (p1, p2) =>
    match p1, p2 with
    | Person.Man n1, Person.Woman n2 => n1 ≠ n2
    | Person.Woman n1, Person.Man n2 => n1 ≠ n2
    | _, _ => true)

/-- Counts the number of valid circular arrangements -/
def countValidArrangements : Nat :=
  (List.filter isValidArrangement (List.permutations (List.map Person.Man (List.range 4) ++ List.map Person.Woman (List.range 4)))).length / 8

theorem valid_arrangements_count :
  countValidArrangements = 12 := by
  sorry

end valid_arrangements_count_l1285_128537


namespace rectangles_in_5x5_grid_l1285_128561

/-- The number of dots in each row and column of the square array -/
def gridSize : ℕ := 5

/-- The number of different rectangles that can be formed in the grid -/
def numberOfRectangles : ℕ := (gridSize.choose 2) * (gridSize.choose 2)

/-- Theorem stating that the number of rectangles in a 5x5 grid is 100 -/
theorem rectangles_in_5x5_grid : numberOfRectangles = 100 := by sorry

end rectangles_in_5x5_grid_l1285_128561


namespace total_miles_theorem_l1285_128534

/-- The total miles run by Bill and Julia on Saturday and Sunday -/
def total_miles (bill_sunday : ℕ) : ℕ :=
  let bill_saturday := bill_sunday - 4
  let julia_sunday := 2 * bill_sunday
  bill_saturday + bill_sunday + julia_sunday

/-- Theorem: Given the conditions, Bill and Julia ran 36 miles in total -/
theorem total_miles_theorem (bill_sunday : ℕ) 
  (h1 : bill_sunday = 10) : total_miles bill_sunday = 36 := by
  sorry

end total_miles_theorem_l1285_128534


namespace product_ab_equals_six_l1285_128592

def A : Set ℝ := {-1, 3}
def B (a b : ℝ) : Set ℝ := {x | x^2 + a*x + b = 0}

theorem product_ab_equals_six (a b : ℝ) (h : A = B a b) : a * b = 6 := by
  sorry

end product_ab_equals_six_l1285_128592


namespace quadratic_polynomial_satisfies_conditions_l1285_128547

-- Define the quadratic polynomial q(x)
def q (x : ℚ) : ℚ := -15/14 * x^2 - 75/14 * x + 180/7

-- Theorem stating that q(x) satisfies the given conditions
theorem quadratic_polynomial_satisfies_conditions :
  q (-8) = 0 ∧ q 3 = 0 ∧ q 6 = -45 := by
  sorry

end quadratic_polynomial_satisfies_conditions_l1285_128547


namespace money_exchange_problem_l1285_128510

/-- Proves that given 100 one-hundred-yuan bills exchanged for twenty-yuan and fifty-yuan bills
    totaling 260 bills, the number of twenty-yuan bills is 100 and the number of fifty-yuan bills is 160. -/
theorem money_exchange_problem (x y : ℕ) 
  (h1 : x + y = 260)
  (h2 : 20 * x + 50 * y = 100 * 100) :
  x = 100 ∧ y = 160 := by
  sorry

end money_exchange_problem_l1285_128510


namespace andrew_stickers_distribution_l1285_128532

/-- The number of stickers Andrew bought -/
def total_stickers : ℕ := 750

/-- The number of stickers Andrew kept -/
def kept_stickers : ℕ := 130

/-- The number of additional stickers Fred received compared to Daniel -/
def extra_stickers : ℕ := 120

/-- The number of stickers Daniel received -/
def daniel_stickers : ℕ := 250

theorem andrew_stickers_distribution :
  daniel_stickers + (daniel_stickers + extra_stickers) + kept_stickers = total_stickers :=
by sorry

end andrew_stickers_distribution_l1285_128532


namespace inequality_proof_l1285_128509

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h : 1/a + 1/(2*b) + 1/(3*c) = 1) : a + 2*b + 3*c ≥ 9 := by
  sorry

end inequality_proof_l1285_128509


namespace probability_yellow_ball_l1285_128544

theorem probability_yellow_ball (total_balls yellow_balls : ℕ) 
  (h1 : total_balls = 8)
  (h2 : yellow_balls = 5) :
  (yellow_balls : ℚ) / total_balls = 5 / 8 := by
  sorry

end probability_yellow_ball_l1285_128544


namespace problem_l_shape_surface_area_l1285_128588

/-- Represents a 3D L-shaped structure made of unit cubes -/
structure LShape where
  verticalHeight : ℕ
  verticalWidth : ℕ
  horizontalLength : ℕ
  totalCubes : ℕ

/-- Calculates the surface area of an L-shaped structure -/
def surfaceArea (l : LShape) : ℕ :=
  sorry

/-- The specific L-shape described in the problem -/
def problemLShape : LShape :=
  { verticalHeight := 3
  , verticalWidth := 2
  , horizontalLength := 3
  , totalCubes := 15 }

/-- Theorem stating that the surface area of the problem's L-shape is 34 square units -/
theorem problem_l_shape_surface_area :
  surfaceArea problemLShape = 34 :=
sorry

end problem_l_shape_surface_area_l1285_128588


namespace total_mowings_is_30_l1285_128528

/-- Represents the number of times Ned mowed a lawn in each season -/
structure SeasonalMowing where
  spring : Nat
  summer : Nat
  fall : Nat

/-- Calculates the total number of mowings for a lawn across all seasons -/
def totalMowings (s : SeasonalMowing) : Nat :=
  s.spring + s.summer + s.fall

/-- The number of times Ned mowed his front lawn in each season -/
def frontLawnMowings : SeasonalMowing :=
  { spring := 6, summer := 5, fall := 4 }

/-- The number of times Ned mowed his backyard lawn in each season -/
def backyardLawnMowings : SeasonalMowing :=
  { spring := 5, summer := 7, fall := 3 }

/-- Theorem: The total number of times Ned mowed his lawns is 30 -/
theorem total_mowings_is_30 :
  totalMowings frontLawnMowings + totalMowings backyardLawnMowings = 30 := by
  sorry


end total_mowings_is_30_l1285_128528


namespace max_handshakes_networking_event_l1285_128549

/-- Calculate the number of handshakes in a group -/
def handshakesInGroup (n : ℕ) : ℕ := n * (n - 1) / 2

/-- Calculate the total number of handshakes -/
def totalHandshakes (total : ℕ) (groupA : ℕ) (groupB : ℕ) (groupC : ℕ) : ℕ :=
  handshakesInGroup total - (handshakesInGroup groupA + handshakesInGroup groupB + handshakesInGroup groupC)

/-- Theorem stating the maximum number of handshakes under given conditions -/
theorem max_handshakes_networking_event :
  let total := 100
  let groupA := 30
  let groupB := 35
  let groupC := 35
  totalHandshakes total groupA groupB groupC = 3325 := by
  sorry

end max_handshakes_networking_event_l1285_128549


namespace alex_more_pens_than_jane_l1285_128517

def alex_pens (week : Nat) : Nat :=
  4 * 2^(week - 1)

def jane_pens : Nat := 16

theorem alex_more_pens_than_jane :
  alex_pens 4 - jane_pens = 16 := by
  sorry

end alex_more_pens_than_jane_l1285_128517


namespace truck_speed_problem_l1285_128539

/-- Proves that the speed of Truck Y is 53 miles per hour given the problem conditions -/
theorem truck_speed_problem (initial_distance : ℝ) (truck_x_speed : ℝ) (overtake_time : ℝ) (final_lead : ℝ) :
  initial_distance = 13 →
  truck_x_speed = 47 →
  overtake_time = 3 →
  final_lead = 5 →
  (initial_distance + truck_x_speed * overtake_time + final_lead) / overtake_time = 53 := by
  sorry

#check truck_speed_problem

end truck_speed_problem_l1285_128539


namespace f_2_equals_13_l1285_128550

def a (k n : ℕ) : ℕ := 10^(k+1) + n^3

def b (k n : ℕ) : ℕ := (a k n) / (10^k)

def f (k : ℕ) : ℕ := sorry

theorem f_2_equals_13 : f 2 = 13 := by sorry

end f_2_equals_13_l1285_128550


namespace parallelogram_area_parallelogram_area_proof_l1285_128573

/-- The area of a parallelogram with base 28 cm and height 32 cm is 896 square centimeters. -/
theorem parallelogram_area : ℝ → ℝ → ℝ → Prop :=
  fun base height area =>
    base = 28 ∧ height = 32 → area = base * height ∧ area = 896

/-- Proof of the theorem -/
theorem parallelogram_area_proof : parallelogram_area 28 32 896 := by
  sorry

end parallelogram_area_parallelogram_area_proof_l1285_128573


namespace johns_age_is_20_l1285_128552

-- Define John's age and his dad's age
def johns_age : ℕ := sorry
def dads_age : ℕ := sorry

-- State the theorem
theorem johns_age_is_20 :
  (johns_age + 30 = dads_age) →
  (johns_age + dads_age = 70) →
  johns_age = 20 := by
sorry

end johns_age_is_20_l1285_128552


namespace largest_angle_cosine_in_triangle_l1285_128560

theorem largest_angle_cosine_in_triangle (a b c : ℝ) (h1 : a = 3) (h2 : b = 5) (h3 : c = 7) :
  let cos_largest_angle := min (min ((a^2 + b^2 - c^2) / (2*a*b)) ((b^2 + c^2 - a^2) / (2*b*c))) ((c^2 + a^2 - b^2) / (2*c*a))
  cos_largest_angle = -1/2 := by
sorry

end largest_angle_cosine_in_triangle_l1285_128560


namespace negative_fractions_comparison_l1285_128554

theorem negative_fractions_comparison : -3/4 > -4/5 := by
  sorry

end negative_fractions_comparison_l1285_128554


namespace custodian_jugs_l1285_128505

/-- The number of cups a full jug can hold -/
def jug_capacity : ℕ := 40

/-- The number of students -/
def num_students : ℕ := 200

/-- The number of cups each student drinks per day -/
def cups_per_student : ℕ := 10

/-- Calculates the number of jugs needed to provide water for all students -/
def jugs_needed : ℕ := (num_students * cups_per_student) / jug_capacity

theorem custodian_jugs : jugs_needed = 50 := by
  sorry

end custodian_jugs_l1285_128505


namespace problem_statement_l1285_128575

theorem problem_statement (a : ℝ) : 
  let A : Set ℝ := {1, 2, a + 3}
  let B : Set ℝ := {a, 5}
  A ∪ B = A → a = 2 := by
sorry

end problem_statement_l1285_128575


namespace max_handshakes_l1285_128570

theorem max_handshakes (n : ℕ) (h : n = 25) : 
  (n * (n - 1)) / 2 = 300 :=
by
  sorry

end max_handshakes_l1285_128570


namespace overlap_area_bound_l1285_128543

open Set

-- Define the type for rectangles
structure Rectangle where
  area : ℝ

-- Define the large rectangle
def largeRectangle : Rectangle :=
  { area := 5 }

-- Define the set of smaller rectangles
def smallRectangles : Set Rectangle :=
  { r : Rectangle | r.area = 1 }

-- State the theorem
theorem overlap_area_bound (n : ℕ) (h : n = 9) :
  ∃ (r₁ r₂ : Rectangle),
    r₁ ∈ smallRectangles ∧
    r₂ ∈ smallRectangles ∧
    r₁ ≠ r₂ ∧
    (∃ (overlap : Rectangle), overlap.area ≥ 1/9) :=
sorry

end overlap_area_bound_l1285_128543


namespace R_value_at_seven_l1285_128527

-- Define the function R in terms of g and S
def R (g : ℝ) (S : ℝ) : ℝ := 2 * g * S + 3

-- State the theorem
theorem R_value_at_seven :
  ∃ (g : ℝ), (R g 5 = 23) → (R g 7 = 31) := by
  sorry

end R_value_at_seven_l1285_128527


namespace arithmetic_sequence_cosine_l1285_128535

def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_cosine (a : ℕ → ℝ) :
  is_arithmetic_sequence a →
  a 1 + a 5 + a 9 = 8 * Real.pi →
  Real.cos (a 2 + a 8) = -1/2 :=
by
  sorry

end arithmetic_sequence_cosine_l1285_128535


namespace fencemaker_problem_l1285_128525

theorem fencemaker_problem (length width : ℝ) : 
  width = 40 → 
  length * width = 200 → 
  2 * length + width = 50 := by
sorry

end fencemaker_problem_l1285_128525


namespace middle_number_proof_l1285_128518

theorem middle_number_proof (x y z : ℕ) 
  (h1 : x < y) (h2 : y < z) 
  (h3 : x + y = 14) (h4 : x + z = 20) (h5 : y + z = 22) : 
  y = 8 := by
sorry

end middle_number_proof_l1285_128518


namespace only_solution_is_five_l1285_128529

theorem only_solution_is_five (n : ℤ) : 
  (⌊(n^2 : ℚ) / 5⌋ - ⌊(n : ℚ) / 2⌋^2 = 1) ↔ n = 5 := by
  sorry

end only_solution_is_five_l1285_128529


namespace linear_function_common_quadrants_l1285_128574

/-- A linear function is represented by its slope and y-intercept -/
structure LinearFunction where
  k : ℝ  -- slope
  b : ℝ  -- y-intercept

/-- Quadrants of the Cartesian plane -/
inductive Quadrant
  | first
  | second
  | third
  | fourth

/-- The set of quadrants that a linear function passes through -/
def quadrants_passed (f : LinearFunction) : Set Quadrant :=
  sorry

/-- The common quadrants for all linear functions satisfying kb < 0 -/
def common_quadrants : Set Quadrant :=
  sorry

theorem linear_function_common_quadrants (f : LinearFunction) 
  (h : f.k * f.b < 0) : 
  quadrants_passed f ∩ common_quadrants = {Quadrant.first, Quadrant.fourth} :=
sorry

end linear_function_common_quadrants_l1285_128574


namespace matrix_inverse_and_solution_l1285_128577

theorem matrix_inverse_and_solution (A B M : Matrix (Fin 2) (Fin 2) ℝ) : 
  A = ![![2, 0], ![-1, 1]] →
  B = ![![2, 4], ![3, 5]] →
  A * M = B →
  A⁻¹ = ![![1/2, 0], ![1/2, 1]] ∧
  M = ![![1, 2], ![4, 7]] := by
  sorry

end matrix_inverse_and_solution_l1285_128577


namespace power_difference_l1285_128512

theorem power_difference (a m n : ℝ) (hm : a ^ m = 3) (hn : a ^ n = 5) : 
  a ^ (m - n) = 3 / 5 := by
  sorry

end power_difference_l1285_128512


namespace athleteHitsBullseyeUncertain_l1285_128530

-- Define the type for different kinds of events
inductive EventType
  | Certain
  | Impossible
  | Uncertain

-- Define the event
def athleteHitsBullseye : EventType := EventType.Uncertain

-- Theorem statement
theorem athleteHitsBullseyeUncertain : athleteHitsBullseye = EventType.Uncertain := by
  sorry

end athleteHitsBullseyeUncertain_l1285_128530


namespace min_value_expression_l1285_128540

theorem min_value_expression (a b c k : ℝ) 
  (h1 : a < c) (h2 : c < b) (h3 : b = k * c) (h4 : k > 1) :
  ∃ (min : ℝ), min = k^2/3 + 2 ∧ 
  ∀ (x : ℝ), ((k*c - a)^2 + (a + c)^2 + (c - a)^2) / c^2 ≥ min :=
sorry

end min_value_expression_l1285_128540


namespace initial_water_is_11_l1285_128567

/-- Represents the hiking scenario with given conditions -/
structure HikingScenario where
  totalDistance : ℝ
  totalTime : ℝ
  waterRemaining : ℝ
  leakRate : ℝ
  lastMileConsumption : ℝ
  firstSixMilesRate : ℝ

/-- Calculates the initial amount of water in the canteen -/
def initialWater (scenario : HikingScenario) : ℝ :=
  scenario.waterRemaining +
  scenario.leakRate * scenario.totalTime +
  scenario.lastMileConsumption +
  scenario.firstSixMilesRate * (scenario.totalDistance - 1)

/-- Theorem stating that the initial amount of water is 11 cups -/
theorem initial_water_is_11 (scenario : HikingScenario)
  (h1 : scenario.totalDistance = 7)
  (h2 : scenario.totalTime = 2)
  (h3 : scenario.waterRemaining = 3)
  (h4 : scenario.leakRate = 1)
  (h5 : scenario.lastMileConsumption = 2)
  (h6 : scenario.firstSixMilesRate = 0.6666666666666666) :
  initialWater scenario = 11 := by
  sorry

end initial_water_is_11_l1285_128567


namespace largest_integer_less_than_150_with_remainder_5_mod_8_l1285_128557

theorem largest_integer_less_than_150_with_remainder_5_mod_8 :
  ∃ n : ℕ, n < 150 ∧ n % 8 = 5 ∧ ∀ m : ℕ, m < 150 ∧ m % 8 = 5 → m ≤ n :=
by sorry

end largest_integer_less_than_150_with_remainder_5_mod_8_l1285_128557


namespace amanda_coffee_blend_typeA_quantity_l1285_128596

/-- Represents the cost and quantity of coffee in Amanda's Coffee Shop blend --/
structure CoffeeBlend where
  typeA_cost : ℝ
  typeB_cost : ℝ
  typeA_quantity : ℝ
  typeB_quantity : ℝ
  total_cost : ℝ

/-- Theorem stating the quantity of type A coffee in the blend --/
theorem amanda_coffee_blend_typeA_quantity (blend : CoffeeBlend) 
  (h1 : blend.typeA_cost = 4.60)
  (h2 : blend.typeB_cost = 5.95)
  (h3 : blend.typeB_quantity = 2 * blend.typeA_quantity)
  (h4 : blend.total_cost = 511.50)
  (h5 : blend.total_cost = blend.typeA_cost * blend.typeA_quantity + blend.typeB_cost * blend.typeB_quantity) :
  blend.typeA_quantity = 31 := by
  sorry


end amanda_coffee_blend_typeA_quantity_l1285_128596


namespace simplify_fraction_l1285_128598

theorem simplify_fraction (a : ℝ) (h : a = 2) : 15 * a^5 / (75 * a^3) = 4/5 := by
  sorry

end simplify_fraction_l1285_128598


namespace train_passing_time_l1285_128541

/-- Theorem: Time taken for slower train to pass faster train's driver -/
theorem train_passing_time
  (train_length : ℝ)
  (fast_train_speed slow_train_speed : ℝ)
  (h1 : train_length = 500)
  (h2 : fast_train_speed = 45)
  (h3 : slow_train_speed = 15) :
  (train_length / ((fast_train_speed + slow_train_speed) * (1000 / 3600))) = 300 := by
  sorry

end train_passing_time_l1285_128541


namespace midpoint_one_sixth_to_five_sixths_l1285_128516

theorem midpoint_one_sixth_to_five_sixths :
  let a := (1 : ℚ) / 6
  let b := (5 : ℚ) / 6
  (a + b) / 2 = (1 : ℚ) / 2 := by
sorry

end midpoint_one_sixth_to_five_sixths_l1285_128516


namespace function_inequality_l1285_128504

open Real

theorem function_inequality (f : ℝ → ℝ) (hf : Differentiable ℝ f) 
  (h : ∀ x, x * (deriv f x) > -f x) (a b : ℝ) (hab : a > b) : 
  a * f a > b * f b := by
  sorry

end function_inequality_l1285_128504


namespace sequence_bound_l1285_128546

theorem sequence_bound (a : ℕ → ℝ) (c : ℝ) 
  (h1 : ∀ i : ℕ, i > 0 → 0 ≤ a i ∧ a i ≤ c)
  (h2 : ∀ i j : ℕ, i > 0 → j > 0 → i ≠ j → |a i - a j| ≥ 1 / (i + j)) :
  c ≥ 1 := by
sorry

end sequence_bound_l1285_128546


namespace joans_books_l1285_128572

/-- Given that Sam has 110 books and the total number of books Sam and Joan have together is 212,
    prove that Joan has 102 books. -/
theorem joans_books (sam_books : ℕ) (total_books : ℕ) (h1 : sam_books = 110) (h2 : total_books = 212) :
  total_books - sam_books = 102 := by
  sorry

end joans_books_l1285_128572


namespace system_solution_l1285_128559

theorem system_solution :
  ∀ x y z : ℚ,
  (x * y + 1 = 2 * z ∧
   y * z + 1 = 2 * x ∧
   z * x + 1 = 2 * y) →
  ((x = 1 ∧ y = 1 ∧ z = 1) ∨
   (x = -2 ∧ y = -2 ∧ z = 5/2) ∨
   (x = -2 ∧ y = 5/2 ∧ z = -2) ∨
   (x = 5/2 ∧ y = -2 ∧ z = -2)) :=
by sorry

end system_solution_l1285_128559


namespace alcohol_remaining_l1285_128542

/-- The amount of alcohol remaining after a series of pours and refills -/
def remaining_alcohol (initial_volume : ℚ) (pour_out1 : ℚ) (refill1 : ℚ) 
  (pour_out2 : ℚ) (refill2 : ℚ) (pour_out3 : ℚ) (refill3 : ℚ) : ℚ :=
  initial_volume * (1 - pour_out1) * (1 - pour_out2) * (1 - pour_out3)

/-- Theorem stating the final amount of alcohol in the bottle -/
theorem alcohol_remaining :
  remaining_alcohol 1 (1/3) (1/3) (1/3) (1/3) 1 1 = 8/27 := by
  sorry


end alcohol_remaining_l1285_128542


namespace binomial_expansion_sum_l1285_128553

theorem binomial_expansion_sum (a₀ a₁ a₂ a₃ a₄ a₅ : ℝ) : 
  (∀ x, (2 - x)^5 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5) →
  a₁ + a₂ + a₄ + a₅ = -31 := by
sorry

end binomial_expansion_sum_l1285_128553


namespace max_sides_with_four_obtuse_angles_l1285_128566

-- Define a convex polygon type
structure ConvexPolygon where
  sides : ℕ
  interior_angles : Fin sides → Real
  is_convex : Bool
  obtuse_count : ℕ

-- Define the theorem
theorem max_sides_with_four_obtuse_angles 
  (p : ConvexPolygon) 
  (h1 : p.is_convex = true) 
  (h2 : p.obtuse_count = 4) 
  (h3 : ∀ i, 0 < p.interior_angles i ∧ p.interior_angles i < 180) 
  (h4 : (Finset.sum Finset.univ p.interior_angles) = (p.sides - 2) * 180) :
  p.sides ≤ 7 := by
  sorry


end max_sides_with_four_obtuse_angles_l1285_128566


namespace min_value_theorem_l1285_128562

theorem min_value_theorem (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 2*x + y - 2 = 0) :
  2/x + 9/y ≥ 25/2 := by sorry

end min_value_theorem_l1285_128562


namespace charles_housesitting_hours_l1285_128500

/-- Proves that Charles housesat for 10 hours given the conditions of the problem -/
theorem charles_housesitting_hours : 
  let housesitting_rate : ℝ := 15
  let dog_walking_rate : ℝ := 22
  let num_dogs : ℕ := 3
  let total_earnings : ℝ := 216
  ∃ (h : ℝ), h * housesitting_rate + (num_dogs : ℝ) * dog_walking_rate = total_earnings ∧ h = 10 := by
  sorry

end charles_housesitting_hours_l1285_128500


namespace quadratic_equation_m_value_l1285_128580

theorem quadratic_equation_m_value :
  ∀ m : ℝ,
  (∀ x : ℝ, (m + 1) * x^(m * (m - 2) - 1) + 2 * m * x - 1 = 0 → ∃ a b c : ℝ, a ≠ 0 ∧ (m + 1) * x^(m * (m - 2) - 1) + 2 * m * x - 1 = a * x^2 + b * x + c) →
  m = 3 :=
by sorry

end quadratic_equation_m_value_l1285_128580


namespace derivative_sin_2x_minus_1_l1285_128564

theorem derivative_sin_2x_minus_1 (x : ℝ) :
  deriv (λ x => Real.sin (2 * x - 1)) x = 2 * Real.cos (2 * x - 1) := by
  sorry

end derivative_sin_2x_minus_1_l1285_128564


namespace spade_calculation_l1285_128526

-- Define the ♠ operation
def spade (x y : ℝ) : ℝ := (x + 2*y)^2 * (x - y)

-- State the theorem
theorem spade_calculation : spade 3 (spade 2 3) = 1046875 := by
  sorry

end spade_calculation_l1285_128526


namespace problem_one_problem_two_l1285_128590

-- Problem 1
theorem problem_one : -2^2 - |2 - 5| + (-1) * 2 = -1 := by sorry

-- Problem 2
theorem problem_two : ∃! x : ℝ, 5 * x - 2 = 3 * x + 18 ∧ x = 10 := by sorry

end problem_one_problem_two_l1285_128590


namespace min_cuts_for_ten_pieces_l1285_128533

/-- The number of pieces resulting from n vertical cuts on a cylindrical cake -/
def num_pieces (n : ℕ) : ℕ := n * (n + 1) / 2

/-- The minimum number of vertical cuts needed to divide a cylindrical cake into at least 10 pieces -/
theorem min_cuts_for_ten_pieces : ∃ (n : ℕ), n ≥ 4 ∧ num_pieces n ≥ 10 ∧ ∀ (m : ℕ), m < n → num_pieces m < 10 :=
sorry

end min_cuts_for_ten_pieces_l1285_128533


namespace increase_by_percentage_l1285_128563

theorem increase_by_percentage (initial : ℝ) (percentage : ℝ) (final : ℝ) :
  initial = 3680 ∧ percentage = 84.3 ∧ final = initial * (1 + percentage / 100) →
  final = 6782.64 := by
  sorry

end increase_by_percentage_l1285_128563


namespace fifth_term_of_geometric_sequence_l1285_128531

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

theorem fifth_term_of_geometric_sequence
  (a : ℕ → ℝ)
  (h_geo : geometric_sequence a)
  (h_a1 : a 1 = 2)
  (h_a3 : a 3 = 6) :
  a 5 = 18 :=
sorry

end fifth_term_of_geometric_sequence_l1285_128531


namespace third_car_year_l1285_128538

def first_car_year : ℕ := 1970
def years_between_first_and_second : ℕ := 10
def years_between_second_and_third : ℕ := 20

theorem third_car_year :
  first_car_year + years_between_first_and_second + years_between_second_and_third = 2000 :=
by sorry

end third_car_year_l1285_128538


namespace pure_imaginary_condition_l1285_128524

theorem pure_imaginary_condition (m : ℝ) : 
  (∀ z : ℂ, z = (m^2 - m : ℝ) + (m^2 - 3*m + 2 : ℝ) * I → z.re = 0 ∧ z.im ≠ 0) → m = 0 := by
  sorry

end pure_imaginary_condition_l1285_128524


namespace iris_pants_purchase_l1285_128503

/-- Represents the number of pairs of pants Iris bought -/
def num_pants : ℕ := sorry

/-- The cost of each jacket -/
def jacket_cost : ℕ := 10

/-- The number of jackets bought -/
def num_jackets : ℕ := 3

/-- The cost of each pair of shorts -/
def shorts_cost : ℕ := 6

/-- The number of pairs of shorts bought -/
def num_shorts : ℕ := 2

/-- The cost of each pair of pants -/
def pants_cost : ℕ := 12

/-- The total amount spent -/
def total_spent : ℕ := 90

theorem iris_pants_purchase :
  num_pants = 4 ∧
  num_pants * pants_cost + num_jackets * jacket_cost + num_shorts * shorts_cost = total_spent :=
by sorry

end iris_pants_purchase_l1285_128503


namespace stick_marking_underdetermined_l1285_128523

/-- Represents the length of a portion of the stick -/
structure Portion where
  length : ℚ
  isValid : 0 < length ∧ length ≤ 1

/-- Represents the configuration of markings on the stick -/
structure StickMarkings where
  fifthPortions : ℕ
  xPortions : ℕ
  xLength : ℚ
  totalLength : ℚ
  validTotal : fifthPortions + xPortions = 8
  validLength : fifthPortions * (1/5) + xPortions * xLength = totalLength

/-- Theorem stating that the problem is underdetermined -/
theorem stick_marking_underdetermined :
  ∀ (m : StickMarkings),
    m.totalLength = 1 →
    ∃ (m' : StickMarkings),
      m'.totalLength = 1 ∧
      m'.fifthPortions ≠ m.fifthPortions ∧
      m'.xLength ≠ m.xLength :=
sorry

end stick_marking_underdetermined_l1285_128523


namespace calculation_proof_l1285_128571

theorem calculation_proof : 101 * 102^2 - 101 * 98^2 = 80800 := by
  sorry

end calculation_proof_l1285_128571


namespace negative_m_exponent_division_l1285_128591

theorem negative_m_exponent_division (m : ℝ) :
  ((-m)^7) / ((-m)^2) = -m^5 := by sorry

end negative_m_exponent_division_l1285_128591


namespace complex_on_negative_diagonal_l1285_128514

/-- A complex number z = a - ai corresponds to a point on the line y = -x in the complex plane. -/
theorem complex_on_negative_diagonal (a : ℝ) : 
  let z : ℂ := a - a * I
  (z.re, z.im) ∈ {p : ℝ × ℝ | p.2 = -p.1} :=
by
  sorry

end complex_on_negative_diagonal_l1285_128514


namespace pencils_sold_is_24_l1285_128551

/-- The total number of pencils sold in a school store sale -/
def total_pencils_sold : ℕ :=
  let first_group := 2  -- number of students in the first group
  let second_group := 6 -- number of students in the second group
  let third_group := 2  -- number of students in the third group
  let pencils_first := 2  -- pencils bought by each student in the first group
  let pencils_second := 3 -- pencils bought by each student in the second group
  let pencils_third := 1  -- pencils bought by each student in the third group
  first_group * pencils_first + second_group * pencils_second + third_group * pencils_third

/-- Theorem stating that the total number of pencils sold is 24 -/
theorem pencils_sold_is_24 : total_pencils_sold = 24 := by
  sorry

end pencils_sold_is_24_l1285_128551


namespace square_diagonal_perimeter_l1285_128583

theorem square_diagonal_perimeter (d : ℝ) (h : d = 20) :
  let side := d / Real.sqrt 2
  4 * side = 40 * Real.sqrt 2 := by
  sorry

end square_diagonal_perimeter_l1285_128583


namespace perpendicular_vectors_l1285_128507

def a : Fin 2 → ℝ := ![2, -1]
def b : Fin 2 → ℝ := ![1, 3]

theorem perpendicular_vectors (m : ℝ) :
  (∀ i : Fin 2, (a i) * (a i + m * (b i)) = 0) →
  m = 5 := by sorry

end perpendicular_vectors_l1285_128507


namespace three_part_division_l1285_128545

theorem three_part_division (A B C : ℝ) (h1 : A > 0) (h2 : B > 0) (h3 : C > 0) 
  (h4 : A + B + C = 782) (h5 : C = 306) : 
  ∃ (k : ℝ), k > 0 ∧ A = k * A ∧ B = k * B ∧ C = k * 306 ∧ A + B = 476 := by
  sorry

end three_part_division_l1285_128545


namespace maries_age_l1285_128597

theorem maries_age (marie_age marco_age : ℕ) : 
  marco_age = 2 * marie_age + 1 →
  marie_age + marco_age = 37 →
  marie_age = 12 := by
sorry

end maries_age_l1285_128597


namespace problem_solution_l1285_128501

theorem problem_solution (p q : ℝ) (h1 : p > 1) (h2 : q > 1)
  (h3 : 1/p + 1/q = 1) (h4 : p * q = 9) : q = (9 + 3 * Real.sqrt 5) / 2 := by
  sorry

end problem_solution_l1285_128501


namespace fermat_point_theorem_l1285_128520

/-- Represents a line in a plane --/
structure Line where
  -- Define a line using two points it passes through
  point1 : ℝ × ℝ
  point2 : ℝ × ℝ

/-- Represents a triangle --/
structure Triangle where
  -- Define a triangle using its three vertices
  vertex1 : ℝ × ℝ
  vertex2 : ℝ × ℝ
  vertex3 : ℝ × ℝ

/-- Get the orthocenter of a triangle --/
def orthocenter (t : Triangle) : ℝ × ℝ := sorry

/-- Get the circumcenter of a triangle --/
def circumcenter (t : Triangle) : ℝ × ℝ := sorry

/-- Get the perpendicular bisector of a line segment --/
def perpendicularBisector (p1 p2 : ℝ × ℝ) : Line := sorry

/-- Get the triangles formed by four lines --/
def getTriangles (l1 l2 l3 l4 : Line) : List Triangle := sorry

/-- Check if a point lies on a line --/
def pointOnLine (p : ℝ × ℝ) (l : Line) : Prop := sorry

/-- The Fermat point theorem --/
theorem fermat_point_theorem (l1 l2 l3 l4 : Line) : 
  ∃! fermatPoint : ℝ × ℝ, 
    ∀ t ∈ getTriangles l1 l2 l3 l4, 
      pointOnLine fermatPoint (perpendicularBisector (orthocenter t) (circumcenter t)) := by
  sorry

end fermat_point_theorem_l1285_128520


namespace plot_length_is_64_l1285_128519

/-- Proves that the length of a rectangular plot is 64 meters given the specified conditions -/
theorem plot_length_is_64 (breadth : ℝ) (length : ℝ) (perimeter : ℝ) (cost_per_meter : ℝ) (total_cost : ℝ) :
  length = breadth + 28 →
  perimeter = 2 * (length + breadth) →
  cost_per_meter = 26.50 →
  total_cost = 5300 →
  cost_per_meter * perimeter = total_cost →
  length = 64 := by
sorry

end plot_length_is_64_l1285_128519


namespace cat_mouse_problem_l1285_128578

theorem cat_mouse_problem (n : ℕ+) (h1 : n * (n + 18) = 999919) : n = 991 := by
  sorry

end cat_mouse_problem_l1285_128578


namespace max_value_sqrt_sum_l1285_128576

theorem max_value_sqrt_sum (x : ℝ) (h : -49 ≤ x ∧ x ≤ 49) :
  Real.sqrt (49 + x) + Real.sqrt (49 - x) ≤ 14 ∧
  ∃ y : ℝ, -49 ≤ y ∧ y ≤ 49 ∧ Real.sqrt (49 + y) + Real.sqrt (49 - y) = 14 :=
by sorry

end max_value_sqrt_sum_l1285_128576


namespace ratio_problem_l1285_128502

theorem ratio_problem (a b c d e f : ℝ) 
  (h1 : a * b * c / (d * e * f) = 1.875)
  (h2 : a / b = 5 / 2)
  (h3 : b / c = 1 / 2)
  (h4 : d / e = 3 / 2)
  (h5 : e / f = 4 / 3) :
  c / d = 0.375 := by sorry

end ratio_problem_l1285_128502


namespace karcsi_travels_further_l1285_128586

def karcsi_speed : ℝ := 6
def joska_speed : ℝ := 4
def bus_speed : ℝ := 60

theorem karcsi_travels_further (x y : ℝ) 
  (hx : x > 0) (hy : y > 0) 
  (h : x / karcsi_speed + (x + y) / bus_speed = y / joska_speed) : x > y := by
  sorry

end karcsi_travels_further_l1285_128586


namespace dance_girls_fraction_l1285_128522

theorem dance_girls_fraction (colfax_total : ℕ) (winthrop_total : ℕ)
  (colfax_boy_ratio colfax_girl_ratio : ℕ)
  (winthrop_boy_ratio winthrop_girl_ratio : ℕ)
  (h1 : colfax_total = 270)
  (h2 : winthrop_total = 180)
  (h3 : colfax_boy_ratio = 5 ∧ colfax_girl_ratio = 4)
  (h4 : winthrop_boy_ratio = 4 ∧ winthrop_girl_ratio = 5) :
  let colfax_girls := colfax_total * colfax_girl_ratio / (colfax_boy_ratio + colfax_girl_ratio)
  let winthrop_girls := winthrop_total * winthrop_girl_ratio / (winthrop_boy_ratio + winthrop_girl_ratio)
  let total_girls := colfax_girls + winthrop_girls
  let total_students := colfax_total + winthrop_total
  (total_girls : ℚ) / total_students = 22 / 45 := by
sorry

end dance_girls_fraction_l1285_128522


namespace coconut_to_mango_ratio_l1285_128521

/-- Proves that the ratio of coconut trees to mango trees is 1:2 given the conditions --/
theorem coconut_to_mango_ratio :
  ∀ (mango_trees coconut_trees total_trees : ℕ) (ratio : ℚ),
    mango_trees = 60 →
    total_trees = 85 →
    coconut_trees = mango_trees * ratio - 5 →
    total_trees = mango_trees + coconut_trees →
    coconut_trees * 2 = mango_trees := by
  sorry

end coconut_to_mango_ratio_l1285_128521


namespace statue_weight_theorem_l1285_128506

/-- The weight of a statue after three weeks of carving --/
def final_statue_weight (initial_weight : ℝ) : ℝ :=
  initial_weight * (1 - 0.3) * (1 - 0.2) * (1 - 0.25)

/-- Theorem stating the weight of the final statue --/
theorem statue_weight_theorem :
  final_statue_weight 250 = 105 := by
  sorry

end statue_weight_theorem_l1285_128506


namespace thirtieth_term_is_59_l1285_128569

def arithmetic_sequence (n : ℕ) : ℕ := 2 * n - 1

theorem thirtieth_term_is_59 : arithmetic_sequence 30 = 59 := by
  sorry

end thirtieth_term_is_59_l1285_128569


namespace muffin_price_is_four_l1285_128515

/-- Represents the number of muffins made by each person and the total contribution --/
structure MuffinSale where
  sasha : ℕ
  melissa : ℕ
  tiffany : ℕ
  contribution : ℕ

/-- Calculates the price per muffin given the sale information --/
def price_per_muffin (sale : MuffinSale) : ℚ :=
  sale.contribution / (sale.sasha + sale.melissa + sale.tiffany)

/-- Theorem stating that the price per muffin is $4 given the conditions --/
theorem muffin_price_is_four :
  ∀ (sale : MuffinSale),
    sale.sasha = 30 →
    sale.melissa = 4 * sale.sasha →
    sale.tiffany = (sale.sasha + sale.melissa) / 2 →
    sale.contribution = 900 →
    price_per_muffin sale = 4 := by
  sorry

end muffin_price_is_four_l1285_128515


namespace quadratic_inequality_solution_l1285_128508

theorem quadratic_inequality_solution (a b : ℝ) :
  (∀ x, a * x^2 - 3 * x + 2 > 0 ↔ x < 1 ∨ x > b) →
  (a = 1 ∧ b = 2) ∧
  (∀ c : ℝ, 
    (∀ x, x^2 - (c + 2) * x + 2 * c ≤ 0 ↔ 
      (c < 2 ∧ c ≤ x ∧ x ≤ 2) ∨
      (c = 2 ∧ x = 2) ∨
      (c > 2 ∧ 2 ≤ x ∧ x ≤ c))) :=
by sorry

end quadratic_inequality_solution_l1285_128508


namespace survivor_quitters_probability_l1285_128595

def total_people : ℕ := 18
def num_tribes : ℕ := 3
def people_per_tribe : ℕ := 6
def num_quitters : ℕ := 3

theorem survivor_quitters_probability :
  let total_ways := Nat.choose total_people num_quitters
  let same_tribe_ways := num_tribes * Nat.choose people_per_tribe num_quitters
  (same_tribe_ways : ℚ) / total_ways = 5 / 68 := by
    sorry

end survivor_quitters_probability_l1285_128595


namespace unique_solution_inequality_l1285_128589

theorem unique_solution_inequality (x : ℝ) :
  x > 0 →
  16 - x ≥ 0 →
  16 * x - x^3 ≥ 0 →
  x * Real.sqrt (16 - x) + Real.sqrt (16 * x - x^3) ≥ 16 →
  x = 4 := by
sorry

end unique_solution_inequality_l1285_128589


namespace inequality_preservation_l1285_128568

theorem inequality_preservation (a b : ℝ) (h : a > b) : a - 5 > b - 5 := by
  sorry

end inequality_preservation_l1285_128568


namespace problem_l1285_128565

def p : Prop := ∀ x : ℝ, (x > 3 ↔ x^2 > 9)
def q : Prop := ∀ a b : ℝ, (a^2 > b^2 ↔ a > b)

theorem problem : ¬(p ∨ q) := by
  sorry

end problem_l1285_128565


namespace kevin_cards_l1285_128556

theorem kevin_cards (initial : ℕ) (found : ℕ) (total : ℕ) : 
  initial = 7 → found = 47 → total = initial + found → total = 54 := by
  sorry

end kevin_cards_l1285_128556


namespace fourth_root_equation_solutions_l1285_128582

theorem fourth_root_equation_solutions :
  {x : ℝ | x > 0 ∧ (x^(1/4) = 20 / (9 - x^(1/4)))} = {256, 625} := by
  sorry

end fourth_root_equation_solutions_l1285_128582


namespace money_sharing_l1285_128548

theorem money_sharing (jessica kevin laura total : ℕ) : 
  jessica + kevin + laura = total →
  jessica = 45 →
  3 * kevin = 4 * jessica →
  3 * laura = 9 * jessica →
  total = 240 :=
by
  sorry

end money_sharing_l1285_128548


namespace count_no_adjacent_same_digits_eq_597880_l1285_128581

/-- Counts the number of integers from 0 to 999999 with no two adjacent digits being the same. -/
def count_no_adjacent_same_digits : ℕ :=
  10 + (9^2 + 9^3 + 9^4 + 9^5 + 9^6)

/-- Theorem stating that the count of integers from 0 to 999999 with no two adjacent digits 
    being the same is equal to 597880. -/
theorem count_no_adjacent_same_digits_eq_597880 : 
  count_no_adjacent_same_digits = 597880 := by
  sorry

end count_no_adjacent_same_digits_eq_597880_l1285_128581


namespace median_unchanged_after_removing_extremes_l1285_128599

theorem median_unchanged_after_removing_extremes 
  (x : Fin 10 → ℝ) 
  (h_ordered : ∀ i j : Fin 10, i ≤ j → x i ≤ x j) :
  (x 4 + x 5) / 2 = (x 5 + x 6) / 2 := by
  sorry

end median_unchanged_after_removing_extremes_l1285_128599


namespace smaller_cuboid_width_l1285_128593

/-- Represents the dimensions of a cuboid -/
structure Cuboid where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Calculates the volume of a cuboid -/
def volume (c : Cuboid) : ℝ := c.length * c.width * c.height

theorem smaller_cuboid_width 
  (large : Cuboid)
  (small_length : ℝ)
  (small_height : ℝ)
  (num_small : ℕ)
  (h1 : large.length = 18)
  (h2 : large.width = 15)
  (h3 : large.height = 2)
  (h4 : small_length = 5)
  (h5 : small_height = 3)
  (h6 : num_small = 18)
  (h7 : volume large = num_small * volume { length := small_length, width := 2, height := small_height }) :
  ∃ (small : Cuboid), small.length = small_length ∧ small.height = small_height ∧ small.width = 2 :=
sorry

end smaller_cuboid_width_l1285_128593


namespace vector_magnitude_l1285_128536

/-- The magnitude of a 2D vector (1, 2) is √5 -/
theorem vector_magnitude : ∀ (a : ℝ × ℝ), a = (1, 2) → ‖a‖ = Real.sqrt 5 := by
  sorry

end vector_magnitude_l1285_128536


namespace board_symbols_l1285_128587

theorem board_symbols (total : Nat) (plus minus : Nat → Prop) : 
  total = 23 →
  (∀ n : Nat, n ≤ total → plus n ∨ minus n) →
  (∀ s : Finset Nat, s.card = 10 → ∃ i ∈ s, plus i) →
  (∀ s : Finset Nat, s.card = 15 → ∃ i ∈ s, minus i) →
  (∃! p m : Nat, p + m = total ∧ plus = λ i => i < p ∧ minus = λ i => p ≤ i ∧ i < total ∧ p = 14 ∧ m = 9) :=
by sorry

end board_symbols_l1285_128587


namespace no_arithmetic_sequence_with_sum_n_cubed_l1285_128558

theorem no_arithmetic_sequence_with_sum_n_cubed :
  ¬ ∃ (a₁ d : ℝ), ∀ (n : ℕ), n > 0 →
    (n : ℝ) / 2 * (2 * a₁ + (n - 1) * d) = (n : ℝ)^3 :=
sorry

end no_arithmetic_sequence_with_sum_n_cubed_l1285_128558


namespace c_investment_value_l1285_128511

/-- Represents the investment and profit distribution in a partnership business --/
structure Partnership where
  a_investment : ℕ
  b_investment : ℕ
  c_investment : ℕ
  total_profit : ℕ
  c_profit : ℕ

/-- Theorem stating that given the conditions of the problem, c's investment is $72,000 --/
theorem c_investment_value (p : Partnership)
  (h1 : p.a_investment = 45000)
  (h2 : p.b_investment = 63000)
  (h3 : p.total_profit = 60000)
  (h4 : p.c_profit = 24000)
  (h5 : p.c_profit * (p.a_investment + p.b_investment + p.c_investment) = p.total_profit * p.c_investment) :
  p.c_investment = 72000 := by
  sorry


end c_investment_value_l1285_128511


namespace olivias_remaining_money_l1285_128555

-- Define the given values
def initial_amount : ℝ := 500
def groceries_cost : ℝ := 125
def shoe_original_price : ℝ := 150
def shoe_discount : ℝ := 0.2
def belt_price : ℝ := 35
def jacket_price : ℝ := 85
def exchange_rate : ℝ := 1.2

-- Define the calculation steps
def discounted_shoe_price : ℝ := shoe_original_price * (1 - shoe_discount)
def total_clothing_cost : ℝ := (discounted_shoe_price + belt_price + jacket_price) * exchange_rate
def total_spent : ℝ := groceries_cost + total_clothing_cost
def remaining_amount : ℝ := initial_amount - total_spent

-- Theorem statement
theorem olivias_remaining_money :
  remaining_amount = 87 :=
by sorry

end olivias_remaining_money_l1285_128555


namespace sum_of_solutions_l1285_128579

theorem sum_of_solutions (N : ℝ) : (N * (N + 4) = 8) → (∃ x y : ℝ, x + y = -4 ∧ x * (x + 4) = 8 ∧ y * (y + 4) = 8) := by
  sorry

end sum_of_solutions_l1285_128579


namespace log_sum_equality_l1285_128585

theorem log_sum_equality : Real.log 4 / Real.log 10 + 2 * (Real.log 5 / Real.log 10) = 2 := by
  sorry

end log_sum_equality_l1285_128585


namespace triangle_properties_l1285_128594

/-- Triangle with vertices A(4, 0), B(8, 10), and C(0, 6) -/
structure Triangle where
  A : Prod ℝ ℝ := (4, 0)
  B : Prod ℝ ℝ := (8, 10)
  C : Prod ℝ ℝ := (0, 6)

/-- Equation of a line in the form ax + by + c = 0 -/
structure LineEquation where
  a : ℝ
  b : ℝ
  c : ℝ

def Triangle.altitudeFromAtoBC (t : Triangle) : LineEquation :=
  { a := 2, b := -3, c := 14 }

def Triangle.lineParallelToBCThroughA (t : Triangle) : LineEquation :=
  { a := 1, b := -2, c := -4 }

def Triangle.altitudeFromBtoAC (t : Triangle) : LineEquation :=
  { a := 2, b := 1, c := -8 }

theorem triangle_properties (t : Triangle) : 
  (t.altitudeFromAtoBC = { a := 2, b := -3, c := 14 }) ∧ 
  (t.lineParallelToBCThroughA = { a := 1, b := -2, c := -4 }) ∧
  (t.altitudeFromBtoAC = { a := 2, b := 1, c := -8 }) := by
  sorry


end triangle_properties_l1285_128594


namespace kaylee_biscuits_l1285_128513

def biscuit_problem (total_needed : ℕ) (lemon_sold : ℕ) (chocolate_sold : ℕ) (oatmeal_sold : ℕ) : ℕ :=
  total_needed - (lemon_sold + chocolate_sold + oatmeal_sold)

theorem kaylee_biscuits :
  biscuit_problem 33 12 5 4 = 12 := by
  sorry

end kaylee_biscuits_l1285_128513


namespace highway_mileage_l1285_128584

/-- Proves that the highway mileage is 37 mpg given the problem conditions -/
theorem highway_mileage (city_mpg : ℝ) (total_miles : ℝ) (total_gallons : ℝ) (highway_city_diff : ℝ) :
  city_mpg = 30 →
  total_miles = 365 →
  total_gallons = 11 →
  highway_city_diff = 5 →
  ∃ (city_miles highway_miles : ℝ),
    city_miles + highway_miles = total_miles ∧
    highway_miles = city_miles + highway_city_diff ∧
    city_miles / city_mpg + highway_miles / 37 = total_gallons :=
by sorry

end highway_mileage_l1285_128584
