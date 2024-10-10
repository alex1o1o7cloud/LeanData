import Mathlib

namespace mandy_toys_count_l2208_220895

theorem mandy_toys_count (mandy anna amanda : ℕ) 
  (h1 : anna = 3 * mandy)
  (h2 : amanda = anna + 2)
  (h3 : mandy + anna + amanda = 142) :
  mandy = 20 := by
sorry

end mandy_toys_count_l2208_220895


namespace min_value_expression_min_value_attained_l2208_220887

theorem min_value_expression (x : ℝ) : 
  (15 - x) * (8 - x) * (15 + x) * (8 + x) ≥ -6480.25 :=
by sorry

theorem min_value_attained : 
  ∃ x : ℝ, (15 - x) * (8 - x) * (15 + x) * (8 + x) = -6480.25 :=
by sorry

end min_value_expression_min_value_attained_l2208_220887


namespace stratified_sample_size_l2208_220880

/-- Represents the composition of a population --/
structure Population where
  elderly : Nat
  middleAged : Nat
  young : Nat

/-- Represents a stratified sample --/
structure StratifiedSample where
  population : Population
  sampleSize : Nat
  youngInSample : Nat

/-- Theorem stating the relationship between the sample size and the number of young people in the sample --/
theorem stratified_sample_size 
  (sample : StratifiedSample) 
  (h1 : sample.population = { elderly := 20, middleAged := 120, young := 100 })
  (h2 : sample.youngInSample = 10) : 
  sample.sampleSize = 24 := by
  sorry

end stratified_sample_size_l2208_220880


namespace abs_greater_necessary_not_sufficient_l2208_220840

theorem abs_greater_necessary_not_sufficient (a b : ℝ) :
  (∀ a b : ℝ, a > b → |a| > b) ∧
  (∃ a b : ℝ, |a| > b ∧ ¬(a > b)) :=
by sorry

end abs_greater_necessary_not_sufficient_l2208_220840


namespace complement_A_intersect_B_range_of_a_for_A_intersect_C_eq_C_l2208_220896

-- Define the sets A, B, and C
def A : Set ℝ := {x | x^2 - x - 6 ≤ 0}
def B : Set ℝ := {x | (x-4)/(x+1) < 0}
def C (a : ℝ) : Set ℝ := {x | 2-a < x ∧ x < 2+a}

-- Statement for (∁_R A) ∩ B = (3, 4)
theorem complement_A_intersect_B : (Set.univ \ A) ∩ B = Set.Ioo 3 4 := by sorry

-- Statement for the range of a when A ∩ C = C
theorem range_of_a_for_A_intersect_C_eq_C :
  ∀ a : ℝ, (A ∩ C a = C a) ↔ a ≤ 1 := by sorry

end complement_A_intersect_B_range_of_a_for_A_intersect_C_eq_C_l2208_220896


namespace prime_and_multiple_of_5_probability_l2208_220884

/-- A function that checks if a natural number is prime -/
def isPrime (n : ℕ) : Prop := sorry

/-- A function that checks if a natural number is a multiple of 5 -/
def isMultipleOf5 (n : ℕ) : Prop := sorry

/-- The set of cards numbered from 1 to 75 -/
def cardSet : Finset ℕ := sorry

/-- The probability of an event occurring when selecting from the card set -/
def probability (event : ℕ → Prop) : ℚ := sorry

theorem prime_and_multiple_of_5_probability :
  probability (fun n => n ∈ cardSet ∧ isPrime n ∧ isMultipleOf5 n) = 1 / 75 := by sorry

end prime_and_multiple_of_5_probability_l2208_220884


namespace tranquility_essence_l2208_220863

/-- Represents the philosophical concepts in the problem --/
structure PhilosophicalConcept where
  opposingAndUnified : Bool  -- The sides of a contradiction are both opposing and unified
  struggleWithinUnity : Bool -- The nature of struggle is embedded within unity
  differencesBasedOnUnity : Bool -- Differences and opposition are based on unity
  motionCharacteristic : Bool -- Motion is the only characteristic of matter

/-- Represents a painting with its elements --/
structure Painting where
  hasWaterfall : Bool
  hasTree : Bool
  hasBirdNest : Bool
  hasSleepingBird : Bool

/-- Defines the essence of tranquility based on philosophical concepts --/
def essenceOfTranquility (p : Painting) (c : PhilosophicalConcept) : Prop :=
  p.hasWaterfall ∧ p.hasTree ∧ p.hasBirdNest ∧ p.hasSleepingBird ∧
  c.opposingAndUnified ∧ c.struggleWithinUnity ∧
  ¬c.differencesBasedOnUnity ∧ ¬c.motionCharacteristic

/-- The theorem to be proved --/
theorem tranquility_essence (p : Painting) (c : PhilosophicalConcept) :
  p.hasWaterfall ∧ p.hasTree ∧ p.hasBirdNest ∧ p.hasSleepingBird →
  c.opposingAndUnified ∧ c.struggleWithinUnity →
  essenceOfTranquility p c := by
  sorry


end tranquility_essence_l2208_220863


namespace triangle_division_theorem_l2208_220843

/-- Represents a triangle with two angles α and β --/
structure Triangle where
  α : ℝ
  β : ℝ
  angle_sum : α + β < π / 2

/-- Predicate to check if two triangles are similar --/
def similar (t1 t2 : Triangle) : Prop := sorry

/-- Predicate to check if a triangle can be divided into a list of triangles --/
def can_be_divided_into (t : Triangle) (ts : List Triangle) : Prop := sorry

/-- The main theorem --/
theorem triangle_division_theorem (n : ℕ) (h : n ≥ 2) :
  ∃ (ts : List Triangle),
    ts.length = n ∧
    (∀ i j, i ≠ j → ¬similar (ts.get i) (ts.get j)) ∧
    (∀ t ∈ ts, ∃ (subts : List Triangle),
      subts.length = n ∧
      can_be_divided_into t subts ∧
      (∀ i j, i ≠ j → ¬similar (subts.get i) (subts.get j)) ∧
      (∀ subt ∈ subts, ∃ t' ∈ ts, similar subt t')) :=
sorry

end triangle_division_theorem_l2208_220843


namespace camper_difference_is_nine_l2208_220892

/-- The number of campers who went rowing in the morning -/
def morning_campers : ℕ := 52

/-- The number of campers who went rowing in the afternoon -/
def afternoon_campers : ℕ := 61

/-- The difference in the number of campers rowing in the afternoon compared to the morning -/
def camper_difference : ℕ := afternoon_campers - morning_campers

/-- Theorem stating that the difference in campers is 9 -/
theorem camper_difference_is_nine : camper_difference = 9 := by
  sorry

end camper_difference_is_nine_l2208_220892


namespace mrs_hilt_total_miles_l2208_220869

/-- Mrs. Hilt's fitness schedule for a week --/
structure FitnessSchedule where
  monday_run : ℕ
  monday_swim : ℕ
  wednesday_run : ℕ
  wednesday_bike : ℕ
  friday_run : ℕ
  friday_swim : ℕ
  friday_bike : ℕ
  sunday_bike : ℕ

/-- Calculate the total miles for a given fitness schedule --/
def total_miles (schedule : FitnessSchedule) : ℕ :=
  schedule.monday_run + schedule.monday_swim +
  schedule.wednesday_run + schedule.wednesday_bike +
  schedule.friday_run + schedule.friday_swim + schedule.friday_bike +
  schedule.sunday_bike

/-- Mrs. Hilt's actual fitness schedule --/
def mrs_hilt_schedule : FitnessSchedule := {
  monday_run := 3
  monday_swim := 1
  wednesday_run := 2
  wednesday_bike := 6
  friday_run := 7
  friday_swim := 2
  friday_bike := 3
  sunday_bike := 10
}

/-- Theorem: Mrs. Hilt's total miles for the week is 34 --/
theorem mrs_hilt_total_miles :
  total_miles mrs_hilt_schedule = 34 := by
  sorry

end mrs_hilt_total_miles_l2208_220869


namespace april_flower_sale_l2208_220830

/-- April's flower sale problem -/
theorem april_flower_sale (initial_roses : ℕ) (remaining_roses : ℕ) (price_per_rose : ℕ) :
  initial_roses = 13 →
  remaining_roses = 4 →
  price_per_rose = 4 →
  (initial_roses - remaining_roses) * price_per_rose = 36 := by
sorry

end april_flower_sale_l2208_220830


namespace square_root_fourth_power_l2208_220889

theorem square_root_fourth_power (x : ℝ) : (Real.sqrt x)^4 = 256 → x = 16 := by
  sorry

end square_root_fourth_power_l2208_220889


namespace peach_multiple_l2208_220861

theorem peach_multiple (martine_peaches benjy_peaches gabrielle_peaches m : ℕ) : 
  martine_peaches = m * benjy_peaches + 6 →
  benjy_peaches = gabrielle_peaches / 3 →
  martine_peaches = 16 →
  gabrielle_peaches = 15 →
  m = 2 := by sorry

end peach_multiple_l2208_220861


namespace part1_part2_l2208_220848

-- Part 1
def f (x : ℝ) : ℝ := |2*x - 2| + 2

theorem part1 : {x : ℝ | f x ≤ 6} = {x : ℝ | -1 ≤ x ∧ x ≤ 3} := by sorry

-- Part 2
def g (x : ℝ) : ℝ := |2*x - 1|

def h (a : ℝ) (x : ℝ) : ℝ := |2*x - a| + a

theorem part2 : {a : ℝ | ∀ x : ℝ, h a x + g x ≥ 3} = {a : ℝ | 2 ≤ a} := by sorry

end part1_part2_l2208_220848


namespace cube_collinear_points_l2208_220891

/-- Represents a point in a cube -/
inductive CubePoint
  | Vertex
  | EdgeMidpoint
  | FaceCenter
  | CubeCenter

/-- Represents a line in a cube -/
structure CubeLine where
  points : Finset CubePoint
  collinear : points.card = 3

/-- The set of all points in the cube -/
def cubePoints : Finset CubePoint := sorry

/-- The set of all lines in the cube -/
def cubeLines : Finset CubeLine := sorry

/-- The number of vertices in a cube -/
def numVertices : Nat := 8

/-- The number of edge midpoints in a cube -/
def numEdgeMidpoints : Nat := 12

/-- The number of face centers in a cube -/
def numFaceCenters : Nat := 6

/-- The number of cube centers in a cube -/
def numCubeCenters : Nat := 1

theorem cube_collinear_points :
  cubePoints.card = numVertices + numEdgeMidpoints + numFaceCenters + numCubeCenters ∧
  cubeLines.card = 49 := by sorry

end cube_collinear_points_l2208_220891


namespace quadratic_roots_sum_l2208_220839

theorem quadratic_roots_sum (α β : ℝ) : 
  (α^2 + 2*α - 2005 = 0) → 
  (β^2 + 2*β - 2005 = 0) → 
  (α^2 + 3*α + β = 2003) := by
sorry

end quadratic_roots_sum_l2208_220839


namespace rectangular_prism_volume_l2208_220803

/-- A rectangular prism with given face areas has a volume of 24 cubic centimeters. -/
theorem rectangular_prism_volume (w h d : ℝ) 
  (front_area : w * h = 12)
  (side_area : d * h = 6)
  (top_area : d * w = 8) :
  w * h * d = 24 := by
  sorry

end rectangular_prism_volume_l2208_220803


namespace chess_piece_probability_l2208_220859

/-- The probability of drawing a red piece first and a green piece second from a bag of chess pieces -/
theorem chess_piece_probability (total : ℕ) (red : ℕ) (green : ℕ) 
  (h1 : total = 32) 
  (h2 : red = 16) 
  (h3 : green = 16) 
  (h4 : red + green = total) : 
  (red / total) * (green / (total - 1)) = 8 / 31 := by
  sorry

end chess_piece_probability_l2208_220859


namespace construct_3x3x3_cube_l2208_220829

/-- Represents a 3D piece with given dimensions -/
structure Piece where
  length : ℕ
  width : ℕ
  height : ℕ

/-- Represents the collection of pieces available for construction -/
structure PieceSet where
  large_pieces : List Piece
  small_pieces : List Piece

/-- Represents a 3D cube -/
structure Cube where
  side_length : ℕ

/-- Checks if a set of pieces can construct the given cube -/
def can_construct_cube (pieces : PieceSet) (cube : Cube) : Prop :=
  -- The actual implementation would involve complex logic to check if the pieces can form the cube
  sorry

/-- The main theorem stating that the given set of pieces can construct a 3x3x3 cube -/
theorem construct_3x3x3_cube : 
  let pieces : PieceSet := {
    large_pieces := List.replicate 6 { length := 1, width := 2, height := 2 },
    small_pieces := List.replicate 3 { length := 1, width := 1, height := 1 }
  }
  let target_cube : Cube := { side_length := 3 }
  can_construct_cube pieces target_cube := by
  sorry


end construct_3x3x3_cube_l2208_220829


namespace system_solution_l2208_220809

theorem system_solution (u v w : ℝ) (hu : u ≠ 0) (hv : v ≠ 0) (hw : w ≠ 0)
  (eq1 : 3 / (u * v) + 15 / (v * w) = 2)
  (eq2 : 15 / (v * w) + 5 / (w * u) = 2)
  (eq3 : 5 / (w * u) + 3 / (u * v) = 2) :
  (u = 1 ∧ v = 3 ∧ w = 5) ∨ (u = -1 ∧ v = -3 ∧ w = -5) := by
  sorry

end system_solution_l2208_220809


namespace power_of_729_two_thirds_l2208_220821

theorem power_of_729_two_thirds : (729 : ℝ) ^ (2/3) = 81 := by
  sorry

end power_of_729_two_thirds_l2208_220821


namespace equation_solutions_l2208_220882

theorem equation_solutions : 
  (∃ x₁ x₂ : ℝ, x₁ = 2 + Real.sqrt 5 ∧ x₂ = 2 - Real.sqrt 5 ∧ 
    x₁^2 - 4*x₁ - 1 = 0 ∧ x₂^2 - 4*x₂ - 1 = 0) ∧
  (∃ x₁ x₂ : ℝ, x₁ = -3 ∧ x₂ = 6 ∧ 
    (x₁ + 3)*(x₁ - 3) = 3*(x₁ + 3) ∧ (x₂ + 3)*(x₂ - 3) = 3*(x₂ + 3)) :=
by sorry

end equation_solutions_l2208_220882


namespace marriage_age_proof_l2208_220800

/-- The average age of a husband and wife at the time of their marriage -/
def average_age_at_marriage : ℝ := 23

/-- The number of years passed since the marriage -/
def years_passed : ℕ := 5

/-- The age of the child -/
def child_age : ℕ := 1

/-- The current average age of the family -/
def current_family_average_age : ℝ := 19

/-- The number of people in the family -/
def family_size : ℕ := 3

theorem marriage_age_proof :
  average_age_at_marriage = 23 :=
by
  sorry

#check marriage_age_proof

end marriage_age_proof_l2208_220800


namespace celias_rent_l2208_220898

/-- Celia's monthly budget -/
structure MonthlyBudget where
  food : ℕ
  streaming : ℕ
  cellPhone : ℕ
  rent : ℕ
  savings : ℕ

/-- Celia's budget satisfies the given conditions -/
def validBudget (b : MonthlyBudget) : Prop :=
  b.food = 400 ∧
  b.streaming = 30 ∧
  b.cellPhone = 50 ∧
  b.savings = 198 ∧
  b.savings * 10 = b.food + b.streaming + b.cellPhone + b.rent

/-- Theorem: Celia's rent is $1500 -/
theorem celias_rent (b : MonthlyBudget) (h : validBudget b) : b.rent = 1500 := by
  sorry


end celias_rent_l2208_220898


namespace circle_radius_l2208_220818

theorem circle_radius (x y : ℝ) : 
  (∃ r : ℝ, r > 0 ∧ ∀ x y : ℝ, x^2 + y^2 + 4*x - 4*y - 1 = 0 ↔ ∃ h k : ℝ, (x - h)^2 + (y - k)^2 = r^2) →
  (∃ r : ℝ, r > 0 ∧ ∀ x y : ℝ, x^2 + y^2 + 4*x - 4*y - 1 = 0 ↔ (x + 2)^2 + (y - 2)^2 = r^2) →
  ∃ r : ℝ, r = 3 ∧ ∀ x y : ℝ, x^2 + y^2 + 4*x - 4*y - 1 = 0 ↔ (x + 2)^2 + (y - 2)^2 = r^2 :=
by sorry

end circle_radius_l2208_220818


namespace consecutive_primes_square_sum_prime_l2208_220849

/-- Definition of consecutive primes -/
def ConsecutivePrimes (p q r : Nat) : Prop :=
  Nat.Prime p ∧ Nat.Prime q ∧ Nat.Prime r ∧
  ∃ (x y : Nat), p < x ∧ ¬Nat.Prime x ∧ x < q ∧
                 q < y ∧ ¬Nat.Prime y ∧ y < r

theorem consecutive_primes_square_sum_prime :
  ∀ p q r : Nat,
    ConsecutivePrimes p q r →
    Nat.Prime (p^2 + q^2 + r^2) →
    p = 3 ∧ q = 5 ∧ r = 7 :=
by sorry

end consecutive_primes_square_sum_prime_l2208_220849


namespace sand_loss_l2208_220868

theorem sand_loss (initial_sand final_sand : ℝ) 
  (h_initial : initial_sand = 4.1)
  (h_final : final_sand = 1.7) : 
  initial_sand - final_sand = 2.4 := by
  sorry

end sand_loss_l2208_220868


namespace work_completion_time_l2208_220808

-- Define the work rates for A, B, and C
def work_rate_A : ℚ := 1 / 30
def work_rate_B : ℚ := 1 / 55
def work_rate_C : ℚ := 1 / 45

-- Define the combined work rate
def combined_work_rate : ℚ := work_rate_A + work_rate_B + work_rate_C

-- Define the time taken to complete the work together
def time_to_complete : ℚ := 1 / combined_work_rate

-- Theorem statement
theorem work_completion_time :
  time_to_complete = 55 / 4 := by sorry

end work_completion_time_l2208_220808


namespace max_cylinder_radius_in_crate_l2208_220820

/-- A rectangular crate with given dimensions. -/
structure Crate where
  length : ℝ
  width : ℝ
  height : ℝ

/-- A right circular cylinder. -/
structure Cylinder where
  radius : ℝ
  height : ℝ

/-- Check if a cylinder fits in a crate when placed upright. -/
def cylinderFitsInCrate (cyl : Cylinder) (crate : Crate) : Prop :=
  cyl.radius * 2 ≤ min crate.length crate.width ∧
  cyl.height ≤ max crate.length (max crate.width crate.height)

/-- The theorem stating the maximum radius of a cylinder that fits in the given crate. -/
theorem max_cylinder_radius_in_crate :
  let crate := Crate.mk 5 8 12
  ∃ (max_radius : ℝ),
    max_radius = 2.5 ∧
    (∀ (r : ℝ), r > max_radius → ∃ (h : ℝ),
      ¬cylinderFitsInCrate (Cylinder.mk r h) crate) ∧
    (∀ (r : ℝ), r ≤ max_radius → ∃ (h : ℝ),
      cylinderFitsInCrate (Cylinder.mk r h) crate) :=
by sorry

end max_cylinder_radius_in_crate_l2208_220820


namespace fraction_value_l2208_220842

theorem fraction_value (a b : ℝ) (h1 : 1/a + 2/b = 1) (h2 : a ≠ -b) : (a*b - a) / (a + b) = 1 := by
  sorry

end fraction_value_l2208_220842


namespace equation_solution_l2208_220826

theorem equation_solution : ∃ x : ℝ, 0.6 * x + (0.2 * 0.4) = 0.56 ∧ x = 0.8 := by
  sorry

end equation_solution_l2208_220826


namespace condition_for_squared_inequality_l2208_220833

theorem condition_for_squared_inequality (a b : ℝ) :
  (∀ a b : ℝ, a > b ∧ b > 0 → a^2 > b^2) ∧
  (∃ a b : ℝ, a^2 > b^2 ∧ ¬(a > b ∧ b > 0)) :=
by sorry

end condition_for_squared_inequality_l2208_220833


namespace factor_implies_d_value_l2208_220877

theorem factor_implies_d_value (d : ℝ) : 
  (∀ x : ℝ, (2 * x + 5) ∣ (8 * x^3 + 27 * x^2 + d * x + 55)) → d = 39.5 := by
  sorry

end factor_implies_d_value_l2208_220877


namespace min_edges_after_operations_l2208_220816

/-- A complete graph with n vertices. -/
structure CompleteGraph (n : ℕ) where
  vertices : Finset (Fin n)
  edges : Finset (Fin n × Fin n)
  complete : ∀ i j : Fin n, i ≠ j → (i, j) ∈ edges

/-- An elementary operation on a graph. -/
def elementaryOperation (G : CompleteGraph n) : CompleteGraph n → Prop :=
  sorry

/-- The result of applying any number of elementary operations. -/
def resultGraph (G : CompleteGraph n) : CompleteGraph n → Prop :=
  sorry

/-- The number of edges in a graph. -/
def numEdges (G : CompleteGraph n) : ℕ :=
  G.edges.card

theorem min_edges_after_operations (n : ℕ) (G : CompleteGraph n) (H : CompleteGraph n) :
  resultGraph G H → numEdges H ≥ n :=
  sorry

end min_edges_after_operations_l2208_220816


namespace fill_time_with_both_pumps_l2208_220836

-- Define the fill rates for the old and new pumps
def old_pump_rate : ℚ := 1 / 600
def new_pump_rate : ℚ := 1 / 200

-- Define the combined fill rate
def combined_rate : ℚ := old_pump_rate + new_pump_rate

-- Theorem to prove
theorem fill_time_with_both_pumps :
  (1 : ℚ) / combined_rate = 150 := by sorry

end fill_time_with_both_pumps_l2208_220836


namespace A_in_second_quadrant_implies_x_gt_5_l2208_220867

/-- A point in the second quadrant of the rectangular coordinate system -/
structure SecondQuadrantPoint where
  x : ℝ
  y : ℝ
  x_neg : x < 0
  y_pos : y > 0

/-- The point A with coordinates (6-2x, x-5) -/
def A (x : ℝ) : ℝ × ℝ := (6 - 2*x, x - 5)

/-- Theorem: If A(6-2x, x-5) is in the second quadrant, then x > 5 -/
theorem A_in_second_quadrant_implies_x_gt_5 :
  ∀ x : ℝ, (∃ p : SecondQuadrantPoint, A x = (p.x, p.y)) → x > 5 := by
  sorry

end A_in_second_quadrant_implies_x_gt_5_l2208_220867


namespace one_true_proposition_l2208_220894

-- Define propositions p and q
def p : Prop := ∀ a b : ℝ, a > b → (1 / a < 1 / b)
def q : Prop := ∀ a b : ℝ, (1 / (a * b) < 0) → (a * b < 0)

-- State the theorem
theorem one_true_proposition (h1 : ¬p) (h2 : q) :
  (p ∧ q) = false ∧ (p ∨ q) = true ∧ ((¬p) ∧ (¬q)) = false :=
sorry

end one_true_proposition_l2208_220894


namespace inscribed_circumscribed_ratio_l2208_220874

/-- Given a right-angled triangle with perpendicular sides of 6 and 8,
    prove that the ratio of the radius of the inscribed circle
    to the radius of the circumscribed circle is 2:5 -/
theorem inscribed_circumscribed_ratio (a b c r R : ℝ) : 
  a = 6 → b = 8 → c^2 = a^2 + b^2 → 
  r = (a + b - c) / 2 → R = c / 2 → 
  r / R = 2 / 5 := by
  sorry

end inscribed_circumscribed_ratio_l2208_220874


namespace sandys_sum_attempt_l2208_220812

/-- Sandy's sum attempt problem -/
theorem sandys_sum_attempt :
  ∀ (correct_marks incorrect_marks total_marks correct_sums : ℕ),
    correct_marks = 3 →
    incorrect_marks = 2 →
    total_marks = 45 →
    correct_sums = 21 →
    ∃ (total_sums : ℕ),
      total_sums = correct_sums + (total_marks - correct_marks * correct_sums) / incorrect_marks ∧
      total_sums = 30 :=
by sorry

end sandys_sum_attempt_l2208_220812


namespace trapezoid_area_sum_l2208_220870

/-- Represents a trapezoid with side lengths a, b, c, and d. -/
structure Trapezoid where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ

/-- Calculates the possible areas of a trapezoid. -/
def possibleAreas (t : Trapezoid) : Set ℝ :=
  sorry

/-- Checks if a number is not divisible by the square of any prime. -/
def notDivisibleBySquareOfPrime (n : ℕ) : Prop :=
  sorry

/-- The main theorem about the trapezoid areas. -/
theorem trapezoid_area_sum (t : Trapezoid) 
    (h1 : t.a = 4 ∧ t.b = 6 ∧ t.c = 8 ∧ t.d = 10) :
    ∃ (r₁ r₂ r₃ : ℚ) (n₁ n₂ : ℕ),
      (∀ A ∈ possibleAreas t, ∃ k, A = k * (r₁ * Real.sqrt n₁ + r₂ * Real.sqrt n₂ + r₃)) ∧
      notDivisibleBySquareOfPrime n₁ ∧
      notDivisibleBySquareOfPrime n₂ ∧
      ⌊r₁ + r₂ + r₃ + n₁ + n₂⌋ = 26 :=
by
  sorry

end trapezoid_area_sum_l2208_220870


namespace turnip_bag_options_l2208_220876

def bag_weights : List ℕ := [13, 15, 16, 17, 21, 24]

def is_valid_turnip_weight (t : ℕ) : Prop :=
  t ∈ bag_weights ∧
  ∃ (o c : ℕ),
    o + c = (bag_weights.sum - t) ∧
    c = 2 * o

theorem turnip_bag_options :
  ∀ t : ℕ, is_valid_turnip_weight t ↔ t = 13 ∨ t = 16 :=
by sorry

end turnip_bag_options_l2208_220876


namespace total_insects_count_l2208_220828

/-- The number of ladybugs with spots -/
def ladybugs_with_spots : ℕ := 12170

/-- The number of ladybugs without spots -/
def ladybugs_without_spots : ℕ := 54912

/-- The number of lacewings -/
def lacewings : ℕ := 23250

/-- The total number of insects on the fields -/
def total_insects : ℕ := ladybugs_with_spots + ladybugs_without_spots + lacewings

theorem total_insects_count : total_insects = 90332 := by
  sorry

end total_insects_count_l2208_220828


namespace square_area_error_l2208_220883

theorem square_area_error (x : ℝ) (h : x > 0) :
  let measured_side := 1.19 * x
  let actual_area := x^2
  let calculated_area := measured_side^2
  let area_error := (calculated_area - actual_area) / actual_area
  area_error = 0.4161 := by
sorry

end square_area_error_l2208_220883


namespace tower_height_l2208_220878

/-- The height of a tower given specific angles and distance -/
theorem tower_height (distance : ℝ) (elevation_angle depression_angle : ℝ) 
  (h1 : distance = 20)
  (h2 : elevation_angle = 30 * π / 180)
  (h3 : depression_angle = 45 * π / 180) :
  ∃ (height : ℝ), height = 20 * (1 + Real.sqrt 3 / 3) :=
by sorry

end tower_height_l2208_220878


namespace hall_length_l2208_220881

/-- A rectangular hall with breadth two-thirds of its length and area 2400 sq meters has a length of 60 meters. -/
theorem hall_length (length breadth : ℝ) : 
  breadth = (2 / 3) * length →
  length * breadth = 2400 →
  length = 60 := by
sorry

end hall_length_l2208_220881


namespace symmetry_implies_sum_power_l2208_220838

/-- Two points are symmetric with respect to the y-axis if their x-coordinates are opposites
    and their y-coordinates are equal. -/
def symmetricYAxis (A B : ℝ × ℝ) : Prop :=
  A.1 = -B.1 ∧ A.2 = B.2

theorem symmetry_implies_sum_power (a b : ℝ) :
  symmetricYAxis (a, -2) (-1, b) → (a + b)^2023 = -1 := by
  sorry

end symmetry_implies_sum_power_l2208_220838


namespace triangle_inequality_l2208_220886

theorem triangle_inequality (x y z : ℝ) (A B C : ℝ) :
  x > 0 → y > 0 → z > 0 →
  0 < A ∧ A < π →
  0 < B ∧ B < π →
  0 < C ∧ C < π →
  A + B + C = π →
  x * Real.sin A + y * Real.sin B + z * Real.sin C ≤ 
  (1/2) * (x*y + y*z + z*x) * Real.sqrt ((x + y + z)/(x*y*z)) := by
sorry

end triangle_inequality_l2208_220886


namespace train_crossing_time_l2208_220807

/-- Represents the problem of a train crossing a stationary train -/
theorem train_crossing_time
  (train_speed : Real)
  (pole_passing_time : Real)
  (stationary_train_length : Real)
  (h1 : train_speed = 72 * 1000 / 3600) -- 72 km/h converted to m/s
  (h2 : pole_passing_time = 10)
  (h3 : stationary_train_length = 500)
  : (train_speed * pole_passing_time + stationary_train_length) / train_speed = 35 := by
  sorry


end train_crossing_time_l2208_220807


namespace gcd_lcm_triples_count_l2208_220841

theorem gcd_lcm_triples_count : 
  (Finset.filter 
    (fun (triple : ℕ × ℕ × ℕ) => 
      Nat.gcd (Nat.gcd triple.1 triple.2.1) triple.2.2 = 15 ∧ 
      Nat.lcm (Nat.lcm triple.1 triple.2.1) triple.2.2 = 3^15 * 5^18)
    (Finset.product (Finset.range (3^15 * 5^18 + 1)) 
      (Finset.product (Finset.range (3^15 * 5^18 + 1)) (Finset.range (3^15 * 5^18 + 1))))).card = 8568 := by
  sorry

end gcd_lcm_triples_count_l2208_220841


namespace length_width_difference_l2208_220865

/-- The length of the basketball court in meters -/
def court_length : ℝ := 31

/-- The width of the basketball court in meters -/
def court_width : ℝ := 17

/-- The perimeter of the basketball court in meters -/
def court_perimeter : ℝ := 96

theorem length_width_difference : court_length - court_width = 14 := by
  sorry

end length_width_difference_l2208_220865


namespace total_gum_pieces_l2208_220871

theorem total_gum_pieces (packages : ℕ) (pieces_per_package : ℕ) 
  (h1 : packages = 27) (h2 : pieces_per_package = 18) : 
  packages * pieces_per_package = 486 := by
  sorry

end total_gum_pieces_l2208_220871


namespace expression_equals_one_tenth_l2208_220879

-- Define the ceiling function
def ceiling (x : ℚ) : ℤ := ⌈x⌉

-- Define the expression
def expression : ℚ := 
  (ceiling ((25 : ℚ) / 11 - ceiling ((35 : ℚ) / 19))) / 
  (ceiling ((35 : ℚ) / 11 + ceiling ((11 * 19 : ℚ) / 35)))

-- Theorem statement
theorem expression_equals_one_tenth : expression = 1 / 10 := by
  sorry

end expression_equals_one_tenth_l2208_220879


namespace test_questions_l2208_220819

theorem test_questions (score : ℕ) (correct : ℕ) (incorrect : ℕ) :
  score = correct - 2 * incorrect →
  score = 73 →
  correct = 91 →
  correct + incorrect = 100 :=
by sorry

end test_questions_l2208_220819


namespace circle_bounded_area_l2208_220897

/-- A circle in a 2D plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- The area of the region bound by two circles and the x-axis -/
def boundedArea (c1 c2 : Circle) : ℝ :=
  sorry

theorem circle_bounded_area :
  let c1 : Circle := { center := (5, 5), radius := 5 }
  let c2 : Circle := { center := (15, 5), radius := 5 }
  boundedArea c1 c2 = 50 - 12.5 * Real.pi := by
  sorry

end circle_bounded_area_l2208_220897


namespace calculate_expression_l2208_220845

theorem calculate_expression : 500 * 4020 * 0.0402 * 20 = 1616064000 := by
  sorry

end calculate_expression_l2208_220845


namespace square_sum_diff_l2208_220835

theorem square_sum_diff (a b : ℝ) 
  (h1 : (a + b)^2 = 8) 
  (h2 : (a - b)^2 = 12) : 
  a^2 + b^2 = 10 := by
sorry

end square_sum_diff_l2208_220835


namespace min_value_expression_l2208_220850

theorem min_value_expression (m n : ℝ) (h : m - n^2 = 1) :
  ∀ x y : ℝ, x - y^2 = 1 → m^2 + 2*n^2 + 4*m - 1 ≤ x^2 + 2*y^2 + 4*x - 1 ∧
  ∃ a b : ℝ, a - b^2 = 1 ∧ a^2 + 2*b^2 + 4*a - 1 = 4 :=
by sorry

end min_value_expression_l2208_220850


namespace optimal_arrangement_l2208_220824

/-- Represents the arrangement of workers in a factory --/
structure WorkerArrangement where
  total_workers : ℕ
  type_a_workers : ℕ
  type_b_workers : ℕ
  type_a_production : ℕ
  type_b_production : ℕ
  set_a_units : ℕ
  set_b_units : ℕ

/-- Checks if the arrangement produces exact sets --/
def produces_exact_sets (arrangement : WorkerArrangement) : Prop :=
  arrangement.total_workers = arrangement.type_a_workers + arrangement.type_b_workers ∧
  arrangement.type_a_workers * arrangement.type_a_production / arrangement.set_a_units =
  arrangement.type_b_workers * arrangement.type_b_production / arrangement.set_b_units

/-- Theorem stating that the given arrangement produces exact sets --/
theorem optimal_arrangement :
  produces_exact_sets {
    total_workers := 104,
    type_a_workers := 72,
    type_b_workers := 32,
    type_a_production := 8,
    type_b_production := 12,
    set_a_units := 3,
    set_b_units := 2
  } := by sorry

end optimal_arrangement_l2208_220824


namespace quadratic_function_theorem_l2208_220852

/-- A quadratic function f(x) = ax^2 + bx satisfying certain conditions -/
def QuadraticFunction (a b : ℝ) (f : ℝ → ℝ) : Prop :=
  a ≠ 0 ∧
  (∀ x, f x = a * x^2 + b * x) ∧
  (∀ x, f (-x + 5) = f (x - 3)) ∧
  (∃! x, f x = x)

/-- The domain and range conditions for the quadratic function -/
def DomainRangeCondition (f : ℝ → ℝ) (m n : ℝ) : Prop :=
  m < n ∧
  (∀ x, f x ∈ Set.Icc (3*m) (3*n) ↔ x ∈ Set.Icc m n)

theorem quadratic_function_theorem :
  ∀ a b : ℝ, ∀ f : ℝ → ℝ,
  QuadraticFunction a b f →
  ∃ m n : ℝ,
    (∀ x, f x = -1/2 * x^2 + x) ∧
    m = -4 ∧ n = 0 ∧
    DomainRangeCondition f m n :=
sorry

end quadratic_function_theorem_l2208_220852


namespace sixDigitPermutations_eq_60_l2208_220811

/-- The number of different positive, six-digit integers that can be formed using the digits 1, 1, 3, 3, 3, and 9 -/
def sixDigitPermutations : ℕ :=
  Nat.factorial 6 / (Nat.factorial 2 * Nat.factorial 3 * Nat.factorial 1)

/-- Theorem stating that the number of different positive, six-digit integers
    that can be formed using the digits 1, 1, 3, 3, 3, and 9 is equal to 60 -/
theorem sixDigitPermutations_eq_60 : sixDigitPermutations = 60 := by
  sorry

end sixDigitPermutations_eq_60_l2208_220811


namespace optimal_purchasing_plan_l2208_220802

theorem optimal_purchasing_plan :
  let total_price : ℝ := 12
  let bulb_cost : ℝ := 30
  let motor_cost : ℝ := 45
  let total_items : ℕ := 90
  let bulb_price : ℝ := 3
  let motor_price : ℝ := 9
  let optimal_bulbs : ℕ := 30
  let optimal_motors : ℕ := 60
  let optimal_cost : ℝ := 630

  (∀ x y : ℕ, 
    x = 2 * y → 
    x * bulb_price = bulb_cost ∧ 
    y * motor_price = motor_cost) ∧
  
  (∀ m : ℕ,
    m ≤ total_items ∧
    m ≤ (total_items - m) / 2 →
    3 * m + 9 * (total_items - m) ≥ optimal_cost) ∧
  
  optimal_bulbs * bulb_price + optimal_motors * motor_price = optimal_cost :=
by sorry

end optimal_purchasing_plan_l2208_220802


namespace max_removable_correct_l2208_220837

/-- Represents the number of marbles of each color in the bag -/
structure MarbleBag where
  yellow : Nat
  red : Nat
  black : Nat

/-- Checks if the remaining marbles satisfy the condition -/
def satisfiesCondition (bag : MarbleBag) : Prop :=
  (bag.yellow ≥ 4 ∧ (bag.red ≥ 3 ∨ bag.black ≥ 3)) ∨
  (bag.red ≥ 4 ∧ (bag.yellow ≥ 3 ∨ bag.black ≥ 3)) ∨
  (bag.black ≥ 4 ∧ (bag.yellow ≥ 3 ∨ bag.red ≥ 3))

/-- The initial bag of marbles -/
def initialBag : MarbleBag := ⟨8, 7, 5⟩

/-- The maximum number of marbles that can be removed -/
def maxRemovable : Nat := 7

theorem max_removable_correct :
  (∀ (removed : MarbleBag), 
    removed.yellow + removed.red + removed.black ≤ maxRemovable →
    satisfiesCondition ⟨initialBag.yellow - removed.yellow, 
                        initialBag.red - removed.red, 
                        initialBag.black - removed.black⟩) ∧
  (∃ (removed : MarbleBag), 
    removed.yellow + removed.red + removed.black = maxRemovable + 1 ∧
    ¬satisfiesCondition ⟨initialBag.yellow - removed.yellow, 
                         initialBag.red - removed.red, 
                         initialBag.black - removed.black⟩) :=
by sorry

end max_removable_correct_l2208_220837


namespace piston_experiment_l2208_220875

variable (l d P q π : ℝ)
variable (x y : ℝ)

-- Conditions
variable (h1 : l > 0)
variable (h2 : d > 0)
variable (h3 : P > 0)
variable (h4 : q > 0)
variable (h5 : π > 0)

-- Theorem statement
theorem piston_experiment :
  -- First experiment
  (P * x^2 + 2*q*l*π*x - P*l^2 = 0) ∧
  -- Pressure in AC region
  (l*π / (l + x) = P * (l - x) / q) ∧
  -- Second experiment
  (y = l*P / (q*π - P)) ∧
  -- Condition for piston not falling to bottom
  (P < q*π/2) :=
by sorry

end piston_experiment_l2208_220875


namespace complex_number_in_second_quadrant_l2208_220853

-- Define the complex function f(x) = x^2
def f (x : ℂ) : ℂ := x^2

-- Define the imaginary unit i
def i : ℂ := Complex.I

-- Theorem statement
theorem complex_number_in_second_quadrant :
  let z := f (1 + i) / (3 + i)
  (z.re < 0) ∧ (z.im > 0) := by sorry

end complex_number_in_second_quadrant_l2208_220853


namespace quadratic_roots_expression_l2208_220801

theorem quadratic_roots_expression (x₁ x₂ : ℝ) : 
  x₁^2 - x₁ - 2 = 0 → x₂^2 - x₂ - 2 = 0 → (1 + x₁) + x₂ * (1 - x₁) = 4 := by
  sorry

end quadratic_roots_expression_l2208_220801


namespace polynomial_identity_sum_l2208_220847

theorem polynomial_identity_sum (a₁ a₂ a₃ d₁ d₂ d₃ : ℝ) :
  (∀ x : ℝ, x^8 - x^6 + x^4 - x^2 + 1 = (x^2 + a₁*x + d₁) * (x^2 + a₂*x + d₂) * (x^2 + a₃*x + d₃) * (x^2 + 1)) →
  a₁*d₁ + a₂*d₂ + a₃*d₃ = -1 := by
  sorry

end polynomial_identity_sum_l2208_220847


namespace parallel_segments_and_midpoint_l2208_220822

/-- Given four points on a Cartesian plane, if two line segments formed by these points are parallel,
    then we can determine the y-coordinate of one point and the midpoint of one segment. -/
theorem parallel_segments_and_midpoint
  (A B X Y : ℝ × ℝ)
  (hA : A = (-6, 2))
  (hB : B = (2, -6))
  (hX : X = (4, 16))
  (hY : Y = (20, k))
  (h_parallel : (B.2 - A.2) / (B.1 - A.1) = (Y.2 - X.2) / (Y.1 - X.1)) :
  k = 0 ∧ ((X.1 + Y.1) / 2, (X.2 + Y.2) / 2) = (12, 8) :=
by sorry

end parallel_segments_and_midpoint_l2208_220822


namespace largest_multiple_of_18_with_6_and_9_l2208_220866

/-- A function that checks if a natural number consists only of digits 6 and 9 -/
def only_six_and_nine (n : ℕ) : Prop :=
  ∀ d : ℕ, d ∈ n.digits 10 → d = 6 ∨ d = 9

/-- The largest number consisting of only 6 and 9 digits that is divisible by 18 -/
def m : ℕ := 969696

theorem largest_multiple_of_18_with_6_and_9 :
  (∀ k : ℕ, k > m → ¬(only_six_and_nine k ∧ 18 ∣ k)) ∧
  only_six_and_nine m ∧
  18 ∣ m ∧
  m / 18 = 53872 := by sorry

end largest_multiple_of_18_with_6_and_9_l2208_220866


namespace retirement_savings_l2208_220810

/-- Calculates the final amount using simple interest -/
def simpleInterest (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * (1 + rate * time)

/-- Proves that the final amount after 15 years is 1,640,000 rubles -/
theorem retirement_savings : 
  let principal : ℝ := 800000
  let rate : ℝ := 0.07
  let time : ℝ := 15
  simpleInterest principal rate time = 1640000 := by
  sorry

end retirement_savings_l2208_220810


namespace solve_for_y_l2208_220873

theorem solve_for_y (x y : ℝ) (hx : x = 99) (heq : x^3*y - 2*x^2*y + x*y = 970200) : y = 1 := by
  sorry

end solve_for_y_l2208_220873


namespace sara_bird_count_l2208_220817

/-- The number of dozens of birds Sara saw -/
def dozens_of_birds : ℕ := 8

/-- The number of birds in one dozen -/
def birds_per_dozen : ℕ := 12

/-- The total number of birds Sara saw -/
def total_birds : ℕ := dozens_of_birds * birds_per_dozen

theorem sara_bird_count : total_birds = 96 := by
  sorry

end sara_bird_count_l2208_220817


namespace graphs_intersect_once_l2208_220872

/-- The value of b for which the graphs of y = bx^2 + 5x + 3 and y = -2x - 3 intersect at exactly one point -/
def b : ℚ := 49 / 24

/-- The first equation -/
def f (x : ℝ) : ℝ := b * x^2 + 5 * x + 3

/-- The second equation -/
def g (x : ℝ) : ℝ := -2 * x - 3

/-- Theorem stating that the graphs intersect at exactly one point when b = 49/24 -/
theorem graphs_intersect_once :
  ∃! x : ℝ, f x = g x :=
sorry

end graphs_intersect_once_l2208_220872


namespace projection_matrix_values_l2208_220885

/-- A projection matrix is idempotent (P² = P) -/
def IsProjectionMatrix (P : Matrix (Fin 2) (Fin 2) ℚ) : Prop :=
  P * P = P

/-- The specific form of our projection matrix -/
def P (a c : ℚ) : Matrix (Fin 2) (Fin 2) ℚ :=
  ![![a, 21/76],
    ![c, 55/76]]

theorem projection_matrix_values :
  ∃ (a c : ℚ), IsProjectionMatrix (P a c) ∧ a = 7/19 ∧ c = 21/76 := by
  sorry

end projection_matrix_values_l2208_220885


namespace unit_distance_preservation_implies_all_distance_preservation_l2208_220827

/-- A function that maps points on a plane to other points on the same plane -/
def PlaneMap (Plane : Type*) := Plane → Plane

/-- Distance function between two points on a plane -/
def distance (Plane : Type*) := Plane → Plane → ℝ

/-- A function preserves unit distances if the distance between the images of any two points
    that are one unit apart is also one unit -/
def preserves_unit_distances (Plane : Type*) (f : PlaneMap Plane) (d : distance Plane) :=
  ∀ (P Q : Plane), d P Q = 1 → d (f P) (f Q) = 1

/-- A function preserves all distances if the distance between the images of any two points
    is equal to the distance between the original points -/
def preserves_all_distances (Plane : Type*) (f : PlaneMap Plane) (d : distance Plane) :=
  ∀ (P Q : Plane), d (f P) (f Q) = d P Q

/-- Main theorem: if a plane map preserves unit distances, it preserves all distances -/
theorem unit_distance_preservation_implies_all_distance_preservation
  (Plane : Type*) (f : PlaneMap Plane) (d : distance Plane) :
  preserves_unit_distances Plane f d → preserves_all_distances Plane f d :=
by
  sorry

end unit_distance_preservation_implies_all_distance_preservation_l2208_220827


namespace tan_theta_value_l2208_220860

theorem tan_theta_value (θ : Real) 
  (h : 2 * Real.sin (θ + π/3) = 3 * Real.sin (π/3 - θ)) : 
  Real.tan θ = Real.sqrt 3 / 5 := by
sorry

end tan_theta_value_l2208_220860


namespace system_solution_l2208_220815

theorem system_solution (x y : ℝ) (eq1 : 3 * x + y = 21) (eq2 : x + 3 * y = 1) : 2 * x + 2 * y = 11 := by
  sorry

end system_solution_l2208_220815


namespace polynomial_identity_l2208_220890

theorem polynomial_identity : 
  102^5 - 5 * 102^4 + 10 * 102^3 - 10 * 102^2 + 5 * 102 - 1 = 1051012301 := by
  sorry

end polynomial_identity_l2208_220890


namespace pink_tie_probability_l2208_220832

-- Define the number of ties of each color
def black_ties : ℕ := 5
def gold_ties : ℕ := 7
def pink_ties : ℕ := 8

-- Define the total number of ties
def total_ties : ℕ := black_ties + gold_ties + pink_ties

-- Define the probability of choosing a pink tie
def prob_pink_tie : ℚ := pink_ties / total_ties

-- Theorem statement
theorem pink_tie_probability :
  prob_pink_tie = 2 / 5 := by sorry

end pink_tie_probability_l2208_220832


namespace cookie_is_circle_with_radius_nine_l2208_220854

/-- The cookie's boundary equation -/
def cookie_boundary (x y : ℝ) : Prop :=
  x^2 + y^2 + 28 = 6*x + 20*y

/-- The circle equation with center (3, 10) and radius 9 -/
def circle_equation (x y : ℝ) : Prop :=
  (x - 3)^2 + (y - 10)^2 = 81

/-- Theorem stating that the cookie boundary is equivalent to a circle with radius 9 -/
theorem cookie_is_circle_with_radius_nine :
  ∀ x y : ℝ, cookie_boundary x y ↔ circle_equation x y :=
by sorry

end cookie_is_circle_with_radius_nine_l2208_220854


namespace periodic_function_l2208_220856

/-- A function f is periodic with period 2c if it satisfies the given functional equation. -/
theorem periodic_function (f : ℝ → ℝ) (c : ℝ) :
  (∀ x, f (x + c) = 2 / (1 + f x) - 1) →
  (∀ x, f (x + 2*c) = f x) :=
by sorry

end periodic_function_l2208_220856


namespace min_side_triangle_l2208_220862

theorem min_side_triangle (a b c : ℝ) (A B C : ℝ) : 
  a + b = 2 → C = 2 * π / 3 → c ≥ Real.sqrt 3 := by
  sorry

end min_side_triangle_l2208_220862


namespace fraction_equality_l2208_220825

theorem fraction_equality (x y : ℝ) (h : y / x = 3 / 4) : (x + y) / x = 7 / 4 := by
  sorry

end fraction_equality_l2208_220825


namespace no_roots_of_equation_l2208_220804

theorem no_roots_of_equation (x : ℝ) (h : x ≠ 4) :
  ¬∃x, x - 9 / (x - 4) = 4 - 9 / (x - 4) :=
sorry

end no_roots_of_equation_l2208_220804


namespace ski_price_calculation_l2208_220858

theorem ski_price_calculation (initial_price : ℝ) 
  (morning_discount : ℝ) (noon_increase : ℝ) (afternoon_discount : ℝ) : 
  initial_price = 200 →
  morning_discount = 0.4 →
  noon_increase = 0.25 →
  afternoon_discount = 0.2 →
  (initial_price * (1 - morning_discount) * (1 + noon_increase) * (1 - afternoon_discount)) = 120 := by
  sorry

end ski_price_calculation_l2208_220858


namespace atomic_weight_X_is_13_l2208_220844

/-- The atomic weight of element X in the compound H3XCOOH -/
def atomic_weight_X : ℝ :=
  let atomic_weight_H : ℝ := 1
  let atomic_weight_C : ℝ := 12
  let atomic_weight_O : ℝ := 16
  let molecular_weight : ℝ := 60
  molecular_weight - (3 * atomic_weight_H + atomic_weight_C + 3 * atomic_weight_O)

/-- Theorem stating that the atomic weight of X is 13 -/
theorem atomic_weight_X_is_13 : atomic_weight_X = 13 := by
  sorry

end atomic_weight_X_is_13_l2208_220844


namespace scientific_notation_of_small_number_l2208_220857

theorem scientific_notation_of_small_number :
  ∃ (a : ℝ) (n : ℤ), 0.0000003 = a * (10 : ℝ) ^ n ∧ 1 ≤ a ∧ a < 10 ∧ a = 3 ∧ n = -7 := by
  sorry

end scientific_notation_of_small_number_l2208_220857


namespace range_of_a_l2208_220814

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, |x - 1| < 1 → x ≥ a) ∧ 
  (∃ x : ℝ, x ≥ a ∧ |x - 1| ≥ 1) → 
  a ≤ 0 := by
sorry

end range_of_a_l2208_220814


namespace special_function_inequality_l2208_220864

/-- A function f: ℝ → ℝ satisfying f(x) + f''(x) > 1 for all x -/
def SpecialFunction (f : ℝ → ℝ) : Prop :=
  ∀ x, f x + (deriv^[2] f) x > 1

/-- Theorem stating the relationship between f(2) - 1 and e^(f(3) - 1) -/
theorem special_function_inequality (f : ℝ → ℝ) (hf : SpecialFunction f) :
  f 2 - 1 < Real.exp (f 3 - 1) := by
  sorry

end special_function_inequality_l2208_220864


namespace arithmetic_computation_l2208_220855

theorem arithmetic_computation : 2 + 8 * 3 - 4 + 6 * 5 / 2 - 3^2 = 28 := by
  sorry

end arithmetic_computation_l2208_220855


namespace third_bounce_height_l2208_220834

/-- Given an initial height and a bounce ratio, calculates the height of the nth bounce -/
def bounce_height (initial_height : ℝ) (bounce_ratio : ℝ) (n : ℕ) : ℝ :=
  initial_height * (bounce_ratio ^ n)

/-- Converts meters to centimeters -/
def meters_to_cm (meters : ℝ) : ℝ :=
  meters * 100

theorem third_bounce_height :
  let initial_height : ℝ := 12.8
  let bounce_ratio : ℝ := 1/4
  let third_bounce_m := bounce_height initial_height bounce_ratio 3
  meters_to_cm third_bounce_m = 20 := by
  sorry

end third_bounce_height_l2208_220834


namespace pet_store_combinations_l2208_220888

def num_puppies : ℕ := 15
def num_kittens : ℕ := 6
def num_hamsters : ℕ := 8
def num_friends : ℕ := 3

theorem pet_store_combinations : 
  num_puppies * num_kittens * num_hamsters * Nat.factorial num_friends = 4320 := by
  sorry

end pet_store_combinations_l2208_220888


namespace acute_triangle_angle_sine_inequality_l2208_220823

theorem acute_triangle_angle_sine_inequality (A B C : Real) 
  (h1 : 0 < A ∧ A < π/2) 
  (h2 : 0 < B ∧ B < π/2) 
  (h3 : 0 < C ∧ C < π/2) 
  (h4 : A + B + C = π) 
  (h5 : A < B) 
  (h6 : B < C) : 
  Real.sin (2*A) > Real.sin (2*B) ∧ Real.sin (2*B) > Real.sin (2*C) := by
  sorry

end acute_triangle_angle_sine_inequality_l2208_220823


namespace inequality_proof_l2208_220806

theorem inequality_proof (x y z : ℝ) 
  (pos_x : 0 < x) (pos_y : 0 < y) (pos_z : 0 < z)
  (h : x * y + y * z + z * x = 1) :
  x * y * z * (x + y) * (y + z) * (x + z) ≥ (1 - x^2) * (1 - y^2) * (1 - z^2) := by
  sorry

end inequality_proof_l2208_220806


namespace ice_cream_sales_l2208_220813

theorem ice_cream_sales (tuesday_sales : ℕ) (wednesday_sales : ℕ) : 
  wednesday_sales = 2 * tuesday_sales →
  tuesday_sales + wednesday_sales = 36000 →
  tuesday_sales = 12000 :=
by
  sorry

end ice_cream_sales_l2208_220813


namespace adult_ticket_cost_l2208_220846

/-- Proves that the cost of an adult ticket is $12 given the conditions of the problem -/
theorem adult_ticket_cost (total_tickets : ℕ) (total_receipts : ℕ) (adult_tickets : ℕ) (child_ticket_cost : ℕ) :
  total_tickets = 130 →
  total_receipts = 840 →
  adult_tickets = 40 →
  child_ticket_cost = 4 →
  ∃ (adult_ticket_cost : ℕ),
    adult_ticket_cost * adult_tickets + child_ticket_cost * (total_tickets - adult_tickets) = total_receipts ∧
    adult_ticket_cost = 12 := by
  sorry

end adult_ticket_cost_l2208_220846


namespace range_of_a_l2208_220805

/-- The range of real number a satisfying the given inequality -/
theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, x > 0 → (a * x + 1) * (Real.exp x - a * Real.exp 1 * x) ≥ 0) ↔ 
  (0 ≤ a ∧ a ≤ 1) :=
by sorry

end range_of_a_l2208_220805


namespace eight_digit_divisibility_l2208_220899

theorem eight_digit_divisibility (n : ℕ) (h : 1000 ≤ n ∧ n < 10000) :
  ∃ k : ℕ, 10001 * n = k * (10000 * n + n) :=
sorry

end eight_digit_divisibility_l2208_220899


namespace tangent_sum_l2208_220893

theorem tangent_sum (x y : ℝ) 
  (h1 : (Real.sin x / Real.cos y) + (Real.sin y / Real.cos x) = 2)
  (h2 : (Real.cos x / Real.sin y) + (Real.cos y / Real.sin x) = 4) :
  (Real.tan x / Real.tan y) + (Real.tan y / Real.tan x) = 4 := by
  sorry

end tangent_sum_l2208_220893


namespace A_ends_with_14_zeros_l2208_220831

theorem A_ends_with_14_zeros :
  let A := 2^7 * (7^14 + 1) + 2^6 * 7^11 * 10^2 + 2^6 * 7^7 * 10^4 + 2^4 * 7^3 * 10^6
  A = 10^14 := by sorry

end A_ends_with_14_zeros_l2208_220831


namespace rounding_317500_equals_31_8_ten_thousand_l2208_220851

/-- Rounds a natural number to the nearest thousand -/
def round_to_nearest_thousand (n : ℕ) : ℕ :=
  ((n + 500) / 1000) * 1000

/-- Converts a natural number to ten thousands -/
def to_ten_thousands (n : ℕ) : ℚ :=
  (n : ℚ) / 10000

theorem rounding_317500_equals_31_8_ten_thousand :
  to_ten_thousands (round_to_nearest_thousand 317500) = 31.8 := by
  sorry

end rounding_317500_equals_31_8_ten_thousand_l2208_220851
