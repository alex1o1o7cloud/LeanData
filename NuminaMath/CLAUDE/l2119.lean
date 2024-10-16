import Mathlib

namespace NUMINAMATH_CALUDE_two_blue_probability_l2119_211987

def total_balls : ℕ := 15
def blue_balls : ℕ := 5
def red_balls : ℕ := 10
def drawn_balls : ℕ := 6
def target_blue : ℕ := 2

def probability_two_blue : ℚ := 2100 / 5005

theorem two_blue_probability :
  (Nat.choose blue_balls target_blue * Nat.choose red_balls (drawn_balls - target_blue)) /
  Nat.choose total_balls drawn_balls = probability_two_blue := by
  sorry

end NUMINAMATH_CALUDE_two_blue_probability_l2119_211987


namespace NUMINAMATH_CALUDE_stratified_sampling_seniors_l2119_211994

theorem stratified_sampling_seniors (total_population : ℕ) (senior_population : ℕ) (sample_size : ℕ) : 
  total_population = 2100 → 
  senior_population = 680 → 
  sample_size = 105 → 
  (sample_size * senior_population) / total_population = 34 := by
sorry

end NUMINAMATH_CALUDE_stratified_sampling_seniors_l2119_211994


namespace NUMINAMATH_CALUDE_quadratic_root_triple_l2119_211944

/-- 
For a quadratic equation ax^2 + bx + c = 0, if one root is triple the other, 
then 3b^2 = 16ac.
-/
theorem quadratic_root_triple (a b c : ℝ) (x₁ x₂ : ℝ) : 
  a ≠ 0 →
  a * x₁^2 + b * x₁ + c = 0 →
  a * x₂^2 + b * x₂ + c = 0 →
  x₂ = 3 * x₁ →
  3 * b^2 = 16 * a * c :=
by sorry


end NUMINAMATH_CALUDE_quadratic_root_triple_l2119_211944


namespace NUMINAMATH_CALUDE_physics_class_size_l2119_211961

theorem physics_class_size (total_students : ℕ) 
  (math_only : ℚ) (physics_only : ℚ) (both : ℕ) :
  total_students = 100 →
  both = 10 →
  physics_only + both = 2 * (math_only + both) →
  math_only + physics_only + both = total_students →
  physics_only + both = 220 / 3 := by
  sorry

end NUMINAMATH_CALUDE_physics_class_size_l2119_211961


namespace NUMINAMATH_CALUDE_product_sequence_value_l2119_211932

theorem product_sequence_value : 
  (1 / 3) * (9 / 1) * (1 / 27) * (81 / 1) * (1 / 243) * (729 / 1) * (1 / 729) * (2187 / 1) = 729 := by
  sorry

end NUMINAMATH_CALUDE_product_sequence_value_l2119_211932


namespace NUMINAMATH_CALUDE_flat_transactions_gain_l2119_211966

/-- Calculates the overall gain from purchasing and selling three flats with given prices and taxes -/
def overall_gain (
  purchase1 sale1 purchase2 sale2 purchase3 sale3 : ℝ
) : ℝ :=
  let purchase_tax := 0.02
  let sale_tax := 0.01
  let gain1 := sale1 * (1 - sale_tax) - purchase1 * (1 + purchase_tax)
  let gain2 := sale2 * (1 - sale_tax) - purchase2 * (1 + purchase_tax)
  let gain3 := sale3 * (1 - sale_tax) - purchase3 * (1 + purchase_tax)
  gain1 + gain2 + gain3

/-- The overall gain from the three flat transactions is $87,762 -/
theorem flat_transactions_gain :
  overall_gain 675958 725000 848592 921500 940600 982000 = 87762 := by
  sorry

end NUMINAMATH_CALUDE_flat_transactions_gain_l2119_211966


namespace NUMINAMATH_CALUDE_cube_sum_fraction_l2119_211903

theorem cube_sum_fraction (x y : ℝ) (hx : x = 3) (hy : y = 4) :
  (x^3 + 3*y^3) / 8 = 219/8 := by
  sorry

end NUMINAMATH_CALUDE_cube_sum_fraction_l2119_211903


namespace NUMINAMATH_CALUDE_simplify_sqrt_500_l2119_211905

theorem simplify_sqrt_500 : Real.sqrt 500 = 10 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_simplify_sqrt_500_l2119_211905


namespace NUMINAMATH_CALUDE_min_value_expression_min_value_achievable_l2119_211934

theorem min_value_expression (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h_sum : x + y + z = 9) (h_rel : y = 2 * x) :
  (x^2 + y^2) / (x + y) + (x^2 + z^2) / (x + z) + (y^2 + z^2) / (y + z) ≥ 6 :=
by sorry

theorem min_value_achievable (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h_sum : x + y + z = 9) (h_rel : y = 2 * x) :
  ∃ (x₀ y₀ z₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ z₀ > 0 ∧ x₀ + y₀ + z₀ = 9 ∧ y₀ = 2 * x₀ ∧
    (x₀^2 + y₀^2) / (x₀ + y₀) + (x₀^2 + z₀^2) / (x₀ + z₀) + (y₀^2 + z₀^2) / (y₀ + z₀) = 6 :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_min_value_achievable_l2119_211934


namespace NUMINAMATH_CALUDE_jamal_storage_solution_l2119_211999

/-- Represents the storage problem with given file sizes and constraints -/
structure StorageProblem where
  total_files : ℕ
  disk_capacity : ℚ
  files_085 : ℕ
  files_075 : ℕ
  files_045 : ℕ
  no_mix_constraint : Bool

/-- Calculates the minimum number of disks needed for the given storage problem -/
def min_disks_needed (p : StorageProblem) : ℕ :=
  sorry

/-- The specific storage problem instance -/
def jamal_storage : StorageProblem :=
  { total_files := 36
  , disk_capacity := 1.44
  , files_085 := 5
  , files_075 := 15
  , files_045 := 16
  , no_mix_constraint := true }

/-- Theorem stating that the minimum number of disks needed for Jamal's storage problem is 24 -/
theorem jamal_storage_solution :
  min_disks_needed jamal_storage = 24 :=
  sorry

end NUMINAMATH_CALUDE_jamal_storage_solution_l2119_211999


namespace NUMINAMATH_CALUDE_ceiling_negative_sqrt_fraction_l2119_211926

theorem ceiling_negative_sqrt_fraction : ⌈-Real.sqrt (81 / 9)⌉ = -3 := by sorry

end NUMINAMATH_CALUDE_ceiling_negative_sqrt_fraction_l2119_211926


namespace NUMINAMATH_CALUDE_complement_of_M_relative_to_U_l2119_211967

def U : Set Nat := {1, 2, 3, 4, 5, 6}
def M : Set Nat := {2, 4, 6}

theorem complement_of_M_relative_to_U :
  U \ M = {1, 3, 5} := by sorry

end NUMINAMATH_CALUDE_complement_of_M_relative_to_U_l2119_211967


namespace NUMINAMATH_CALUDE_range_of_m_l2119_211988

-- Define the propositions p and q
def p (x : ℝ) : Prop := |1 - (x - 1) / 3| ≤ 2
def q (x m : ℝ) : Prop := x^2 - 2*x + 1 - m^2 < 0

-- Define the condition that ¬p is a necessary but not sufficient condition for ¬q
def not_p_necessary_not_sufficient_for_not_q (m : ℝ) : Prop :=
  ∀ x, ¬(q x m) → ¬(p x) ∧ ∃ x, ¬(p x) ∧ (q x m)

-- State the theorem
theorem range_of_m :
  ∀ m : ℝ, (∀ x, p x → q x m) ∧ not_p_necessary_not_sufficient_for_not_q m
  ↔ m ≥ 9 ∨ m ≤ -9 :=
sorry

end NUMINAMATH_CALUDE_range_of_m_l2119_211988


namespace NUMINAMATH_CALUDE_expression_evaluation_l2119_211933

theorem expression_evaluation (a b : ℝ) 
  (h : |a + 2| + (b - 1)^2 = 0) : 
  (a + 3*b) * (2*a - b) - 2*(a - b)^2 = -23 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2119_211933


namespace NUMINAMATH_CALUDE_clownfish_display_count_l2119_211960

/-- Represents the number of fish in the aquarium -/
def total_fish : ℕ := 100

/-- Represents the number of blowfish that stay in their own tank -/
def blowfish_in_own_tank : ℕ := 26

/-- Calculates the number of clownfish in the display tank -/
def clownfish_in_display (total_fish : ℕ) (blowfish_in_own_tank : ℕ) : ℕ :=
  let total_per_species := total_fish / 2
  let blowfish_in_display := total_per_species - blowfish_in_own_tank
  let initial_clownfish_in_display := blowfish_in_display
  initial_clownfish_in_display - (initial_clownfish_in_display / 3)

theorem clownfish_display_count :
  clownfish_in_display total_fish blowfish_in_own_tank = 16 := by
  sorry

end NUMINAMATH_CALUDE_clownfish_display_count_l2119_211960


namespace NUMINAMATH_CALUDE_min_winning_set_size_l2119_211907

/-- The set of allowed digits -/
def AllowedDigits : Finset Nat := {1, 2, 3, 4}

/-- A type representing a three-digit number using only allowed digits -/
structure ThreeDigitNumber where
  d1 : Nat
  d2 : Nat
  d3 : Nat
  h1 : d1 ∈ AllowedDigits
  h2 : d2 ∈ AllowedDigits
  h3 : d3 ∈ AllowedDigits

/-- Function to count how many digits differ between two ThreeDigitNumbers -/
def diffCount (n1 n2 : ThreeDigitNumber) : Nat :=
  (if n1.d1 ≠ n2.d1 then 1 else 0) +
  (if n1.d2 ≠ n2.d2 then 1 else 0) +
  (if n1.d3 ≠ n2.d3 then 1 else 0)

/-- A set of ThreeDigitNumbers is winning if for any other ThreeDigitNumber,
    at least one number in the set differs from it by at most one digit -/
def isWinningSet (s : Finset ThreeDigitNumber) : Prop :=
  ∀ n : ThreeDigitNumber, ∃ m ∈ s, diffCount n m ≤ 1

/-- The main theorem: The minimum size of a winning set is 8 -/
theorem min_winning_set_size :
  (∃ s : Finset ThreeDigitNumber, isWinningSet s ∧ s.card = 8) ∧
  (∀ s : Finset ThreeDigitNumber, isWinningSet s → s.card ≥ 8) :=
sorry

end NUMINAMATH_CALUDE_min_winning_set_size_l2119_211907


namespace NUMINAMATH_CALUDE_debby_water_bottles_l2119_211990

theorem debby_water_bottles (initial_bottles : ℕ) (days : ℕ) (remaining_bottles : ℕ) 
  (h1 : initial_bottles = 264)
  (h2 : days = 11)
  (h3 : remaining_bottles = 99) :
  (initial_bottles - remaining_bottles) / days = 15 := by
  sorry

end NUMINAMATH_CALUDE_debby_water_bottles_l2119_211990


namespace NUMINAMATH_CALUDE_square_equation_solution_l2119_211910

theorem square_equation_solution : 
  ∃! y : ℤ, (2010 + y)^2 = y^2 :=
by
  -- The unique solution is y = -1005
  use -1005
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_square_equation_solution_l2119_211910


namespace NUMINAMATH_CALUDE_unique_solution_exists_l2119_211904

-- Define the system of equations
def system (a b c x y z : ℝ) : Prop :=
  Real.sqrt (y - a) + Real.sqrt (z - a) = 1 ∧
  Real.sqrt (z - b) + Real.sqrt (x - b) = 1 ∧
  Real.sqrt (x - c) + Real.sqrt (y - c) = 1

-- Theorem statement
theorem unique_solution_exists (a b c : ℝ) 
  (h_positive : a > 0 ∧ b > 0 ∧ c > 0) 
  (h_sum : Real.sqrt a + Real.sqrt b + Real.sqrt c = Real.sqrt 3 / 2) :
  ∃! x y z : ℝ, system a b c x y z :=
sorry

end NUMINAMATH_CALUDE_unique_solution_exists_l2119_211904


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l2119_211996

theorem sufficient_not_necessary (a b : ℝ) :
  (∀ a b : ℝ, a > 1 ∧ b > 1 → a + b > 2 ∧ a * b > 1) ∧
  (∃ a b : ℝ, a + b > 2 ∧ a * b > 1 ∧ ¬(a > 1 ∧ b > 1)) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l2119_211996


namespace NUMINAMATH_CALUDE_root_value_theorem_l2119_211901

theorem root_value_theorem (m : ℝ) (h : 2 * m^2 - 7 * m + 1 = 0) :
  m * (2 * m - 7) + 5 = 4 := by
  sorry

end NUMINAMATH_CALUDE_root_value_theorem_l2119_211901


namespace NUMINAMATH_CALUDE_at_operation_example_l2119_211921

def at_operation (x y : ℤ) : ℤ := x * y - 2 * x + 3 * y

theorem at_operation_example : (at_operation 8 5) - (at_operation 5 8) = -15 := by
  sorry

end NUMINAMATH_CALUDE_at_operation_example_l2119_211921


namespace NUMINAMATH_CALUDE_probability_two_red_balls_l2119_211979

def total_balls : ℕ := 7 + 5 + 4

def red_balls : ℕ := 7

theorem probability_two_red_balls :
  (Nat.choose red_balls 2 : ℚ) / (Nat.choose total_balls 2) = 7 / 40 :=
by sorry

end NUMINAMATH_CALUDE_probability_two_red_balls_l2119_211979


namespace NUMINAMATH_CALUDE_garden_area_increase_l2119_211900

theorem garden_area_increase : 
  let rectangle_length : ℝ := 60
  let rectangle_width : ℝ := 20
  let rectangle_perimeter : ℝ := 2 * (rectangle_length + rectangle_width)
  let square_side : ℝ := rectangle_perimeter / 4
  let rectangle_area : ℝ := rectangle_length * rectangle_width
  let square_area : ℝ := square_side * square_side
  square_area - rectangle_area = 400 := by
sorry

end NUMINAMATH_CALUDE_garden_area_increase_l2119_211900


namespace NUMINAMATH_CALUDE_all_good_numbers_less_than_1000_l2119_211939

def isGood (n : ℕ) : Prop :=
  ∀ k p : ℕ, (10^p * k + n) % n = 0

def goodNumbersLessThan1000 : List ℕ := [1, 2, 5, 10, 20, 25, 50, 100, 125, 200]

theorem all_good_numbers_less_than_1000 :
  ∀ n ∈ goodNumbersLessThan1000, isGood n ∧ n < 1000 := by
  sorry

#check all_good_numbers_less_than_1000

end NUMINAMATH_CALUDE_all_good_numbers_less_than_1000_l2119_211939


namespace NUMINAMATH_CALUDE_greg_original_seat_l2119_211913

/-- Represents a seat in the theater --/
inductive Seat
| one
| two
| three
| four
| five

/-- Represents a friend --/
inductive Friend
| Greg
| Iris
| Jamal
| Kim
| Leo

/-- Represents the seating arrangement --/
def Arrangement := Friend → Seat

/-- Represents a movement of a friend --/
def Movement := Friend → Int

theorem greg_original_seat 
  (initial_arrangement : Arrangement)
  (final_arrangement : Arrangement)
  (movements : Movement) :
  (movements Friend.Iris = 1) →
  (movements Friend.Jamal = -2) →
  (movements Friend.Kim + movements Friend.Leo = 0) →
  (final_arrangement Friend.Greg = Seat.one) →
  (initial_arrangement Friend.Greg = Seat.two) :=
sorry

end NUMINAMATH_CALUDE_greg_original_seat_l2119_211913


namespace NUMINAMATH_CALUDE_lead_is_seventeen_l2119_211968

-- Define the scores of both teams
def chucks_team_score : ℕ := 72
def yellow_team_score : ℕ := 55

-- Define the lead as the difference between the scores
def lead : ℕ := chucks_team_score - yellow_team_score

-- Theorem stating that the lead is 17 points
theorem lead_is_seventeen : lead = 17 := by
  sorry

end NUMINAMATH_CALUDE_lead_is_seventeen_l2119_211968


namespace NUMINAMATH_CALUDE_taxi_charge_proof_l2119_211986

/-- Calculates the total charge for a taxi trip -/
def totalCharge (initialFee : ℚ) (additionalChargePerIncrement : ℚ) (incrementDistance : ℚ) (tripDistance : ℚ) : ℚ :=
  initialFee + (tripDistance / incrementDistance).floor * additionalChargePerIncrement

/-- Proves that the total charge for a 3.6-mile trip is $4.50 -/
theorem taxi_charge_proof :
  let initialFee : ℚ := 9/4  -- $2.25
  let additionalChargePerIncrement : ℚ := 1/4  -- $0.25
  let incrementDistance : ℚ := 2/5  -- 2/5 mile
  let tripDistance : ℚ := 18/5  -- 3.6 miles
  totalCharge initialFee additionalChargePerIncrement incrementDistance tripDistance = 9/2  -- $4.50
:= by sorry

end NUMINAMATH_CALUDE_taxi_charge_proof_l2119_211986


namespace NUMINAMATH_CALUDE_jane_has_66_robots_l2119_211918

/-- The number of car robots each person has -/
structure CarRobots where
  tom : ℕ
  michael : ℕ
  bob : ℕ
  sarah : ℕ
  jane : ℕ

/-- The conditions of the car robot collections -/
def satisfiesConditions (c : CarRobots) : Prop :=
  c.tom = 15 ∧
  c.michael = 3 * c.tom - 5 ∧
  c.bob = 8 * (c.tom + c.michael) ∧
  c.sarah = c.bob / 2 - 7 ∧
  c.jane = (c.sarah - c.tom) / 3

/-- Theorem stating that Jane has 66 car robots -/
theorem jane_has_66_robots (c : CarRobots) (h : satisfiesConditions c) : c.jane = 66 := by
  sorry

end NUMINAMATH_CALUDE_jane_has_66_robots_l2119_211918


namespace NUMINAMATH_CALUDE_min_a_value_l2119_211963

theorem min_a_value (a : ℝ) : 
  (∀ x : ℝ, 0 < x ∧ x < 4 → |x - 1| < a) → a ≥ 3 :=
by sorry

end NUMINAMATH_CALUDE_min_a_value_l2119_211963


namespace NUMINAMATH_CALUDE_cos_240_deg_l2119_211978

/-- Cosine of 240 degrees is equal to -1/2 -/
theorem cos_240_deg : Real.cos (240 * π / 180) = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_cos_240_deg_l2119_211978


namespace NUMINAMATH_CALUDE_fourth_competitor_jump_distance_l2119_211928

theorem fourth_competitor_jump_distance (first_jump second_jump third_jump fourth_jump : ℕ) :
  first_jump = 22 ∧
  second_jump = first_jump + 1 ∧
  third_jump = second_jump - 2 ∧
  fourth_jump = third_jump + 3 →
  fourth_jump = 24 := by
  sorry

end NUMINAMATH_CALUDE_fourth_competitor_jump_distance_l2119_211928


namespace NUMINAMATH_CALUDE_wedding_decoration_cost_l2119_211924

/-- The cost of decorations for a wedding reception --/
def decorationCost (numTables : ℕ) (tableclothCost : ℕ) (placeSettingsPerTable : ℕ) 
  (placeSettingCost : ℕ) (rosesPerCenterpiece : ℕ) (roseCost : ℕ) 
  (liliesPerCenterpiece : ℕ) (lilyCost : ℕ) : ℕ :=
  numTables * tableclothCost + 
  numTables * placeSettingsPerTable * placeSettingCost +
  numTables * rosesPerCenterpiece * roseCost +
  numTables * liliesPerCenterpiece * lilyCost

/-- The total cost of decorations for Nathan's wedding reception is $3500 --/
theorem wedding_decoration_cost : 
  decorationCost 20 25 4 10 10 5 15 4 = 3500 := by
  sorry

end NUMINAMATH_CALUDE_wedding_decoration_cost_l2119_211924


namespace NUMINAMATH_CALUDE_train_passengers_proof_l2119_211925

/-- Calculates the number of passengers on each return trip given the total number of round trips, 
    passengers on each one-way trip, and total passengers transported. -/
def return_trip_passengers (round_trips : ℕ) (one_way_passengers : ℕ) (total_passengers : ℕ) : ℕ :=
  (total_passengers - round_trips * one_way_passengers) / round_trips

/-- Proves that given the specified conditions, the number of passengers on each return trip is 60. -/
theorem train_passengers_proof :
  let round_trips : ℕ := 4
  let one_way_passengers : ℕ := 100
  let total_passengers : ℕ := 640
  return_trip_passengers round_trips one_way_passengers total_passengers = 60 := by
  sorry

#eval return_trip_passengers 4 100 640

end NUMINAMATH_CALUDE_train_passengers_proof_l2119_211925


namespace NUMINAMATH_CALUDE_special_ellipse_eccentricity_l2119_211950

/-- An ellipse with a special point P -/
structure SpecialEllipse where
  a : ℝ
  b : ℝ
  h_ab : a > b
  h_b_pos : b > 0
  P : ℝ × ℝ
  h_on_ellipse : (P.1^2 / a^2) + (P.2^2 / b^2) = 1
  h_PF1_perpendicular : P.1 = -((a^2 - b^2).sqrt)
  h_PF2_parallel : P.2 / (P.1 + ((a^2 - b^2).sqrt)) = -b / a

/-- The eccentricity of an ellipse with a special point P is √5/5 -/
theorem special_ellipse_eccentricity (E : SpecialEllipse) :
  ((E.a^2 - E.b^2) / E.a^2).sqrt = (5 : ℝ).sqrt / 5 := by
  sorry

end NUMINAMATH_CALUDE_special_ellipse_eccentricity_l2119_211950


namespace NUMINAMATH_CALUDE_raghu_investment_l2119_211930

/-- Represents the investment amounts of Raghu, Trishul, and Vishal -/
structure Investments where
  raghu : ℝ
  trishul : ℝ
  vishal : ℝ

/-- The conditions of the investment problem -/
def investment_conditions (inv : Investments) : Prop :=
  inv.trishul = 0.9 * inv.raghu ∧
  inv.vishal = 1.1 * inv.trishul ∧
  inv.raghu + inv.trishul + inv.vishal = 6069

/-- Theorem stating that under the given conditions, Raghu's investment is 2100 -/
theorem raghu_investment (inv : Investments) 
  (h : investment_conditions inv) : inv.raghu = 2100 := by
  sorry

end NUMINAMATH_CALUDE_raghu_investment_l2119_211930


namespace NUMINAMATH_CALUDE_toby_breakfast_calories_l2119_211957

/-- Calculates the total calories of a breakfast with bread and peanut butter. -/
def breakfast_calories (bread_calories : ℕ) (pb_calories : ℕ) (pb_servings : ℕ) : ℕ :=
  bread_calories + pb_calories * pb_servings

/-- Proves that a breakfast with one piece of bread (100 calories) and two servings of peanut butter (200 calories each) has 500 calories. -/
theorem toby_breakfast_calories :
  breakfast_calories 100 200 2 = 500 := by
  sorry

#eval breakfast_calories 100 200 2

end NUMINAMATH_CALUDE_toby_breakfast_calories_l2119_211957


namespace NUMINAMATH_CALUDE_roof_dimensions_l2119_211995

theorem roof_dimensions (width : ℝ) (length : ℝ) :
  length = 4 * width →
  width * length = 576 →
  length - width = 36 := by
sorry

end NUMINAMATH_CALUDE_roof_dimensions_l2119_211995


namespace NUMINAMATH_CALUDE_fraction_equality_l2119_211954

theorem fraction_equality (x y : ℝ) (h : (1/x + 1/y) / (1/x - 1/y) = 1001) :
  (x + y) / (x - y) = -1001 := by sorry

end NUMINAMATH_CALUDE_fraction_equality_l2119_211954


namespace NUMINAMATH_CALUDE_x_varies_with_z_l2119_211971

theorem x_varies_with_z (x y z : ℝ) (k j : ℝ) (h1 : x = k * y^2) (h2 : y = j * z^(1/3)) :
  ∃ m : ℝ, x = m * z^(2/3) :=
sorry

end NUMINAMATH_CALUDE_x_varies_with_z_l2119_211971


namespace NUMINAMATH_CALUDE_ten_zeros_in_expansion_l2119_211977

/-- The number of trailing zeros in a natural number -/
def trailingZeros (n : ℕ) : ℕ := sorry

/-- The main theorem: The number of trailing zeros in (10^11 - 2)^2 is 10 -/
theorem ten_zeros_in_expansion : trailingZeros ((10^11 - 2)^2) = 10 := by
  sorry

end NUMINAMATH_CALUDE_ten_zeros_in_expansion_l2119_211977


namespace NUMINAMATH_CALUDE_election_winner_votes_l2119_211940

theorem election_winner_votes (total_votes : ℕ) (winner_percentage : ℚ) (vote_difference : ℕ) :
  winner_percentage = 60 / 100 →
  vote_difference = 288 →
  winner_percentage * total_votes - (1 - winner_percentage) * total_votes = vote_difference →
  winner_percentage * total_votes = 864 :=
by
  sorry

end NUMINAMATH_CALUDE_election_winner_votes_l2119_211940


namespace NUMINAMATH_CALUDE_garden_length_l2119_211915

theorem garden_length (width : ℝ) (length : ℝ) : 
  length = 2 * width →  -- The length is twice the width
  2 * length + 2 * width = 150 →  -- The perimeter is 150 yards
  length = 50 := by  -- The length is 50 yards
sorry

end NUMINAMATH_CALUDE_garden_length_l2119_211915


namespace NUMINAMATH_CALUDE_system1_solution_system2_solution_l2119_211919

-- System 1
theorem system1_solution :
  ∃ (x y : ℝ), 2*x + 3*y = -1 ∧ y = 4*x - 5 ∧ x = 1 ∧ y = -1 := by
  sorry

-- System 2
theorem system2_solution :
  ∃ (x y : ℝ), 3*x + 2*y = 20 ∧ 4*x - 5*y = 19 ∧ x = 6 ∧ y = 1 := by
  sorry

end NUMINAMATH_CALUDE_system1_solution_system2_solution_l2119_211919


namespace NUMINAMATH_CALUDE_square_plus_four_equals_54_l2119_211973

theorem square_plus_four_equals_54 (x : ℝ) (h : x = 5) : 2 * x^2 + 4 = 54 := by
  sorry

end NUMINAMATH_CALUDE_square_plus_four_equals_54_l2119_211973


namespace NUMINAMATH_CALUDE_debt_payment_calculation_l2119_211955

theorem debt_payment_calculation (total_installments : Nat) 
  (first_payments : Nat) (remaining_payments : Nat) (average_payment : ℚ) :
  total_installments = 52 →
  first_payments = 25 →
  remaining_payments = 27 →
  average_payment = 551.9230769230769 →
  ∃ (x : ℚ), 
    (x * first_payments + (x + 100) * remaining_payments) / total_installments = average_payment ∧
    x = 500 := by
  sorry

end NUMINAMATH_CALUDE_debt_payment_calculation_l2119_211955


namespace NUMINAMATH_CALUDE_negative_and_absolute_value_l2119_211937

theorem negative_and_absolute_value : 
  (-(-4) = 4) ∧ (-|(-4)| = -4) := by sorry

end NUMINAMATH_CALUDE_negative_and_absolute_value_l2119_211937


namespace NUMINAMATH_CALUDE_equation_solution_l2119_211982

theorem equation_solution :
  ∃ x : ℚ, x ≠ 1 ∧ 2*x ≠ 2 ∧ x / (x - 1) = 3 / (2*x - 2) - 2 ∧ x = 7/6 :=
by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2119_211982


namespace NUMINAMATH_CALUDE_robotics_club_enrollment_l2119_211917

theorem robotics_club_enrollment (total : ℕ) (cs : ℕ) (electronics : ℕ) (both : ℕ) 
  (h1 : total = 60)
  (h2 : cs = 45)
  (h3 : electronics = 33)
  (h4 : both = 25) :
  total - (cs + electronics - both) = 7 := by
  sorry

end NUMINAMATH_CALUDE_robotics_club_enrollment_l2119_211917


namespace NUMINAMATH_CALUDE_odd_periodic_function_value_l2119_211976

-- Define an odd function f on ℝ
def isOdd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

-- Define the periodic property of f
def hasPeriod (f : ℝ → ℝ) : Prop := ∀ x, f (x + 3/2) = -f x

-- Theorem statement
theorem odd_periodic_function_value (f : ℝ → ℝ) 
  (h_odd : isOdd f) (h_period : hasPeriod f) : f (-3/2) = 0 := by
  sorry

end NUMINAMATH_CALUDE_odd_periodic_function_value_l2119_211976


namespace NUMINAMATH_CALUDE_paint_cost_decrease_l2119_211991

theorem paint_cost_decrease (canvas_original : ℝ) (paint_original : ℝ) 
  (h1 : paint_original = 4 * canvas_original)
  (h2 : canvas_original > 0)
  (h3 : paint_original > 0) :
  let canvas_new := 0.6 * canvas_original
  let total_original := paint_original + canvas_original
  let total_new := 0.4400000000000001 * total_original
  ∃ (paint_new : ℝ), paint_new = 0.4 * paint_original ∧ total_new = paint_new + canvas_new :=
by sorry

end NUMINAMATH_CALUDE_paint_cost_decrease_l2119_211991


namespace NUMINAMATH_CALUDE_four_solutions_three_solutions_l2119_211935

/-- The equation x^2 - 4|x| + k = 0 with integer k and x -/
def equation (k : ℤ) (x : ℤ) : Prop := x^2 - 4 * x.natAbs + k = 0

/-- The set of integer solutions to the equation -/
def solution_set (k : ℤ) : Set ℤ := {x : ℤ | equation k x}

theorem four_solutions :
  solution_set 3 = {1, -1, 3, -3} :=
sorry

theorem three_solutions :
  solution_set 0 = {0, 4, -4} :=
sorry

end NUMINAMATH_CALUDE_four_solutions_three_solutions_l2119_211935


namespace NUMINAMATH_CALUDE_candy_distribution_l2119_211949

theorem candy_distribution (initial_candies : ℕ) (additional_candies : ℕ) (num_friends : ℕ) 
  (h1 : initial_candies = 10)
  (h2 : additional_candies = 4)
  (h3 : num_friends = 7)
  (h4 : num_friends > 0) :
  (initial_candies + additional_candies) / num_friends = 2 :=
by
  sorry

end NUMINAMATH_CALUDE_candy_distribution_l2119_211949


namespace NUMINAMATH_CALUDE_min_value_theorem_min_value_achievable_l2119_211993

theorem min_value_theorem (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) 
  (h : x/y + y/z + z/x + y/x + z/y + x/z = 10) :
  (x/y + y/z + z/x) * (y/x + z/y + x/z) + (x/y * y/z * z/x) ≥ 44 := by
  sorry

-- Optionally, we can add a statement to show that the lower bound is tight
theorem min_value_achievable : 
  ∃ (x y z : ℝ), x > 0 ∧ y > 0 ∧ z > 0 ∧ 
    x/y + y/z + z/x + y/x + z/y + x/z = 10 ∧
    (x/y + y/z + z/x) * (y/x + z/y + x/z) + (x/y * y/z * z/x) = 44 := by
  sorry

end NUMINAMATH_CALUDE_min_value_theorem_min_value_achievable_l2119_211993


namespace NUMINAMATH_CALUDE_pythagorean_numbers_l2119_211956

def is_pythagorean_triple (a b c : ℕ) : Prop :=
  a * a + b * b = c * c

theorem pythagorean_numbers : 
  (is_pythagorean_triple 9 12 15) ∧ 
  (¬ is_pythagorean_triple 3 4 5) ∧ 
  (¬ is_pythagorean_triple 1 1 2) :=
by
  sorry

end NUMINAMATH_CALUDE_pythagorean_numbers_l2119_211956


namespace NUMINAMATH_CALUDE_customers_added_during_lunch_rush_l2119_211958

theorem customers_added_during_lunch_rush 
  (initial_customers : ℕ) 
  (no_tip_customers : ℕ) 
  (tip_customers : ℕ) 
  (h1 : initial_customers = 29)
  (h2 : no_tip_customers = 34)
  (h3 : tip_customers = 15)
  (h4 : no_tip_customers + tip_customers = initial_customers + (customers_added : ℕ)) :
  customers_added = 20 := by
sorry

end NUMINAMATH_CALUDE_customers_added_during_lunch_rush_l2119_211958


namespace NUMINAMATH_CALUDE_simplify_expression_l2119_211927

theorem simplify_expression (x : ℝ) : 
  Real.sqrt (x^2 - 4*x + 4) + Real.sqrt (x^2 + 4*x + 4) = |x - 2| + |x + 2| := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l2119_211927


namespace NUMINAMATH_CALUDE_exists_iceberg_with_properties_l2119_211951

/-- Represents a convex polyhedron floating in water --/
structure FloatingPolyhedron where
  totalVolume : ℝ
  submergedVolume : ℝ
  totalSurfaceArea : ℝ
  submergedSurfaceArea : ℝ
  volume_nonneg : 0 < totalVolume
  submerged_volume_le_total : submergedVolume ≤ totalVolume
  surface_area_nonneg : 0 < totalSurfaceArea
  submerged_surface_le_total : submergedSurfaceArea ≤ totalSurfaceArea

/-- Theorem stating the existence of a floating polyhedron with the required properties --/
theorem exists_iceberg_with_properties :
  ∃ (iceberg : FloatingPolyhedron),
    iceberg.submergedVolume ≥ 0.9 * iceberg.totalVolume ∧
    iceberg.submergedSurfaceArea ≤ 0.5 * iceberg.totalSurfaceArea :=
sorry

end NUMINAMATH_CALUDE_exists_iceberg_with_properties_l2119_211951


namespace NUMINAMATH_CALUDE_sum_product_solution_l2119_211965

theorem sum_product_solution (S P : ℝ) (x y : ℝ) (h1 : x + y = S) (h2 : x * y = P) :
  ((x = (S + Real.sqrt (S^2 - 4*P)) / 2 ∧ y = (S - Real.sqrt (S^2 - 4*P)) / 2) ∨
   (x = (S - Real.sqrt (S^2 - 4*P)) / 2 ∧ y = (S + Real.sqrt (S^2 - 4*P)) / 2)) :=
by sorry

end NUMINAMATH_CALUDE_sum_product_solution_l2119_211965


namespace NUMINAMATH_CALUDE_total_skips_is_33_l2119_211997

/-- Represents the number of skips for each throw -/
structure ThrowSkips :=
  (first : ℕ)
  (second : ℕ)
  (third : ℕ)
  (fourth : ℕ)
  (fifth : ℕ)

/-- Conditions for the stone skipping problem -/
def SkipConditions (t : ThrowSkips) : Prop :=
  t.second = t.first + 2 ∧
  t.third = 2 * t.second ∧
  t.fourth = t.third - 3 ∧
  t.fifth = t.fourth + 1 ∧
  t.fifth = 8

/-- The total number of skips across all throws -/
def TotalSkips (t : ThrowSkips) : ℕ :=
  t.first + t.second + t.third + t.fourth + t.fifth

/-- Theorem stating that the total number of skips is 33 -/
theorem total_skips_is_33 (t : ThrowSkips) (h : SkipConditions t) :
  TotalSkips t = 33 := by
  sorry

end NUMINAMATH_CALUDE_total_skips_is_33_l2119_211997


namespace NUMINAMATH_CALUDE_half_circle_roll_center_path_length_l2119_211902

/-- The length of the path traveled by the center of a half-circle when rolled along a straight line -/
def half_circle_center_path_length (radius : ℝ) : ℝ := 2 * radius

/-- Theorem: The length of the path traveled by the center of a half-circle with radius 1 cm, 
    when rolled along a straight line until it completes a half rotation, is equal to 2 cm -/
theorem half_circle_roll_center_path_length :
  half_circle_center_path_length 1 = 2 := by sorry

end NUMINAMATH_CALUDE_half_circle_roll_center_path_length_l2119_211902


namespace NUMINAMATH_CALUDE_fraction_equality_implies_x_equals_three_l2119_211985

theorem fraction_equality_implies_x_equals_three (x : ℝ) :
  (5 / (2 * x - 1) = 3 / x) → x = 3 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_implies_x_equals_three_l2119_211985


namespace NUMINAMATH_CALUDE_sphere_radius_in_cone_l2119_211942

/-- A right circular cone with four congruent spheres inside -/
structure ConeSpheres where
  base_radius : ℝ
  height : ℝ
  sphere_radius : ℝ
  is_right_circular : base_radius > 0 ∧ height > 0
  spheres_tangent : sphere_radius > 0
  spheres_fit : sphere_radius < base_radius ∧ sphere_radius < height

/-- The theorem stating the radius of each sphere in the specific configuration -/
theorem sphere_radius_in_cone (cs : ConeSpheres) 
  (h1 : cs.base_radius = 8)
  (h2 : cs.height = 15) :
  cs.sphere_radius = 8 * Real.sqrt 3 / 17 := by
  sorry


end NUMINAMATH_CALUDE_sphere_radius_in_cone_l2119_211942


namespace NUMINAMATH_CALUDE_exam_percentage_l2119_211929

theorem exam_percentage (total_items : ℕ) (correct_A correct_B incorrect_B : ℕ) :
  total_items = 60 →
  correct_B = correct_A + 2 →
  incorrect_B = 4 →
  correct_B + incorrect_B = total_items →
  (correct_A : ℚ) / total_items * 100 = 90 := by
  sorry

end NUMINAMATH_CALUDE_exam_percentage_l2119_211929


namespace NUMINAMATH_CALUDE_dave_book_cost_l2119_211931

/-- Calculates the total cost of Dave's books including discounts and taxes -/
def total_cost (animal_books_count : ℕ) (animal_book_price : ℚ)
                (space_books_count : ℕ) (space_book_price : ℚ)
                (train_books_count : ℕ) (train_book_price : ℚ)
                (history_books_count : ℕ) (history_book_price : ℚ)
                (science_books_count : ℕ) (science_book_price : ℚ)
                (animal_discount_rate : ℚ) (science_tax_rate : ℚ) : ℚ :=
  let animal_cost := animal_books_count * animal_book_price * (1 - animal_discount_rate)
  let space_cost := space_books_count * space_book_price
  let train_cost := train_books_count * train_book_price
  let history_cost := history_books_count * history_book_price
  let science_cost := science_books_count * science_book_price * (1 + science_tax_rate)
  animal_cost + space_cost + train_cost + history_cost + science_cost

/-- Theorem stating that the total cost of Dave's books is $379.5 -/
theorem dave_book_cost :
  total_cost 8 10 6 12 9 8 4 15 5 18 (1/10) (15/100) = 379.5 := by
  sorry

end NUMINAMATH_CALUDE_dave_book_cost_l2119_211931


namespace NUMINAMATH_CALUDE_margaret_mean_score_l2119_211962

def scores : List ℕ := [84, 86, 90, 92, 93, 95, 97, 96, 99]

def cyprian_count : ℕ := 5
def margaret_count : ℕ := 4
def cyprian_mean : ℕ := 92

theorem margaret_mean_score :
  let total_sum := scores.sum
  let cyprian_sum := cyprian_count * cyprian_mean
  let margaret_sum := total_sum - cyprian_sum
  (margaret_sum : ℚ) / margaret_count = 93 := by sorry

end NUMINAMATH_CALUDE_margaret_mean_score_l2119_211962


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l2119_211952

def A : Set ℕ := {x | ∃ n : ℕ, x = 3 * n + 2}
def B : Set ℕ := {6, 8, 10, 12, 14}

theorem intersection_of_A_and_B : A ∩ B = {8, 14} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l2119_211952


namespace NUMINAMATH_CALUDE_remainder_theorem_l2119_211911

theorem remainder_theorem (n : ℤ) (k : ℤ) (h : n = 50 * k - 49) :
  (n^2 + 4*n + 5) % 50 = 10 := by
sorry

end NUMINAMATH_CALUDE_remainder_theorem_l2119_211911


namespace NUMINAMATH_CALUDE_school_average_age_l2119_211974

theorem school_average_age 
  (total_students : ℕ) 
  (boys_avg_age girls_avg_age : ℚ) 
  (num_girls : ℕ) :
  total_students = 640 →
  boys_avg_age = 12 →
  girls_avg_age = 11 →
  num_girls = 160 →
  let num_boys := total_students - num_girls
  let total_age := boys_avg_age * num_boys + girls_avg_age * num_girls
  total_age / total_students = 11.75 := by
sorry

end NUMINAMATH_CALUDE_school_average_age_l2119_211974


namespace NUMINAMATH_CALUDE_square_brush_ratio_l2119_211914

theorem square_brush_ratio (s w : ℝ) (h_positive : s > 0 ∧ w > 0) 
  (h_half_painted : w^2 + 2 * ((s - w) / 2)^2 = s^2 / 2) : 
  s / w = 2 * Real.sqrt 2 + 2 := by
sorry

end NUMINAMATH_CALUDE_square_brush_ratio_l2119_211914


namespace NUMINAMATH_CALUDE_right_triangle_set_l2119_211948

theorem right_triangle_set : ∃! (a b c : ℝ), 
  ((a = 3 ∧ b = 4 ∧ c = 5) ∨ 
   (a = 2 ∧ b = 3 ∧ c = 4) ∨ 
   (a = 6 ∧ b = 8 ∧ c = 12) ∨ 
   (a = Real.sqrt 3 ∧ b = Real.sqrt 4 ∧ c = Real.sqrt 5)) ∧
  a^2 + b^2 = c^2 := by
sorry

end NUMINAMATH_CALUDE_right_triangle_set_l2119_211948


namespace NUMINAMATH_CALUDE_even_perfect_square_factors_count_l2119_211972

/-- The number of even perfect square factors of 2^6 * 5^4 * 7^3 -/
def num_even_perfect_square_factors : ℕ :=
  sorry

/-- The given number -/
def given_number : ℕ :=
  2^6 * 5^4 * 7^3

theorem even_perfect_square_factors_count :
  num_even_perfect_square_factors = 18 :=
by sorry

end NUMINAMATH_CALUDE_even_perfect_square_factors_count_l2119_211972


namespace NUMINAMATH_CALUDE_solve_for_p_l2119_211980

theorem solve_for_p (n m p : ℚ) 
  (h1 : 5/6 = n/90)
  (h2 : 5/6 = (m + n)/105)
  (h3 : 5/6 = (p - m)/150) : 
  p = 137.5 := by sorry

end NUMINAMATH_CALUDE_solve_for_p_l2119_211980


namespace NUMINAMATH_CALUDE_marble_draw_probability_l2119_211938

/-- The probability of drawing a white marble first and a red marble second from a bag 
    containing 5 red marbles and 7 white marbles, without replacement. -/
theorem marble_draw_probability :
  let total_marbles : ℕ := 5 + 7
  let red_marbles : ℕ := 5
  let white_marbles : ℕ := 7
  let prob_white_first : ℚ := white_marbles / total_marbles
  let prob_red_second : ℚ := red_marbles / (total_marbles - 1)
  prob_white_first * prob_red_second = 35 / 132 :=
by sorry

end NUMINAMATH_CALUDE_marble_draw_probability_l2119_211938


namespace NUMINAMATH_CALUDE_number_divided_by_ten_l2119_211989

theorem number_divided_by_ten : (120 : ℚ) / 10 = 12 := by
  sorry

end NUMINAMATH_CALUDE_number_divided_by_ten_l2119_211989


namespace NUMINAMATH_CALUDE_sum_inequality_l2119_211981

theorem sum_inequality (a b c d : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) (h_prod : a * b * c * d = 1) :
  1 / Real.sqrt (1/2 + a + a*b + a*b*c) +
  1 / Real.sqrt (1/2 + b + b*c + b*c*d) +
  1 / Real.sqrt (1/2 + c + c*d + c*d*a) +
  1 / Real.sqrt (1/2 + d + d*a + d*a*b) ≥ Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_sum_inequality_l2119_211981


namespace NUMINAMATH_CALUDE_journey_time_proof_l2119_211975

-- Define the journey segments
inductive Segment
| Uphill
| Flat
| Downhill

-- Define the journey parameters
def total_distance : ℝ := 50
def uphill_speed : ℝ := 3

-- Define the ratios
def length_ratio (s : Segment) : ℝ :=
  match s with
  | .Uphill => 1
  | .Flat => 2
  | .Downhill => 3

def time_ratio (s : Segment) : ℝ :=
  match s with
  | .Uphill => 4
  | .Flat => 5
  | .Downhill => 6

-- Define the theorem
theorem journey_time_proof :
  let total_ratio : ℝ := (length_ratio .Uphill) + (length_ratio .Flat) + (length_ratio .Downhill)
  let uphill_distance : ℝ := total_distance * (length_ratio .Uphill) / total_ratio
  let uphill_time : ℝ := uphill_distance / uphill_speed
  let time_ratio_sum : ℝ := (time_ratio .Uphill) + (time_ratio .Flat) + (time_ratio .Downhill)
  let total_time : ℝ := uphill_time * time_ratio_sum / (time_ratio .Uphill)
  total_time = 10 + 5 / 12 :=
by sorry


end NUMINAMATH_CALUDE_journey_time_proof_l2119_211975


namespace NUMINAMATH_CALUDE_rectangular_garden_width_l2119_211964

theorem rectangular_garden_width (width : ℝ) (length : ℝ) (area : ℝ) : 
  length = 3 * width →
  area = length * width →
  area = 432 →
  width = 12 := by
sorry

end NUMINAMATH_CALUDE_rectangular_garden_width_l2119_211964


namespace NUMINAMATH_CALUDE_limit_of_sequence_l2119_211906

/-- The sum of the first n multiples of 3 -/
def S (n : ℕ) : ℕ := 3 * n * (n + 1) / 2

/-- The sequence we're interested in -/
def a (n : ℕ) : ℚ := (S n : ℚ) / (n^2 + 4 : ℚ)

theorem limit_of_sequence :
  ∀ ε > 0, ∃ N, ∀ n ≥ N, |a n - 3/2| < ε :=
sorry

end NUMINAMATH_CALUDE_limit_of_sequence_l2119_211906


namespace NUMINAMATH_CALUDE_specific_right_triangle_perimeter_l2119_211936

/-- A right triangle with integer side lengths, one of which is 11. -/
structure RightTriangle where
  a : ℕ
  b : ℕ
  c : ℕ
  is_right : a^2 + b^2 = c^2
  has_eleven : a = 11 ∨ b = 11

/-- The perimeter of a right triangle. -/
def perimeter (t : RightTriangle) : ℕ := t.a + t.b + t.c

/-- Theorem stating that the perimeter of the specific right triangle is 132. -/
theorem specific_right_triangle_perimeter :
  ∃ t : RightTriangle, perimeter t = 132 :=
sorry

end NUMINAMATH_CALUDE_specific_right_triangle_perimeter_l2119_211936


namespace NUMINAMATH_CALUDE_point_on_y_axis_has_x_zero_l2119_211998

/-- A point on a 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Definition of a point lying on the y-axis -/
def lies_on_y_axis (p : Point) : Prop := p.x = 0

/-- Theorem: If a point lies on the y-axis, its x-coordinate is 0 -/
theorem point_on_y_axis_has_x_zero (M : Point) (h : lies_on_y_axis M) : M.x = 0 := by
  sorry

end NUMINAMATH_CALUDE_point_on_y_axis_has_x_zero_l2119_211998


namespace NUMINAMATH_CALUDE_divisibility_3_power_l2119_211983

theorem divisibility_3_power (n : ℕ) : 
  (∃ k : ℤ, 3^n + 1 = 10 * k) → (∃ m : ℤ, 3^(n+4) + 1 = 10 * m) := by
sorry

end NUMINAMATH_CALUDE_divisibility_3_power_l2119_211983


namespace NUMINAMATH_CALUDE_axis_of_symmetry_l2119_211923

/-- For a quadratic function y = ax^2 - 4ax + 1 where a ≠ 0, the axis of symmetry is x = 2 -/
theorem axis_of_symmetry (a : ℝ) (ha : a ≠ 0) :
  let f : ℝ → ℝ := λ x ↦ a * x^2 - 4 * a * x + 1
  (∀ x : ℝ, f (2 + x) = f (2 - x)) := by
sorry


end NUMINAMATH_CALUDE_axis_of_symmetry_l2119_211923


namespace NUMINAMATH_CALUDE_purely_imaginary_implies_m_is_one_l2119_211969

/-- A complex number z is purely imaginary if its real part is zero and its imaginary part is non-zero. -/
def isPurelyImaginary (z : ℂ) : Prop := z.re = 0 ∧ z.im ≠ 0

/-- Given that z = (m^2 - m) + mi is purely imaginary and m is real, prove that m = 1. -/
theorem purely_imaginary_implies_m_is_one (m : ℝ) :
  isPurelyImaginary ((m^2 - m : ℝ) + m * I) → m = 1 := by
  sorry


end NUMINAMATH_CALUDE_purely_imaginary_implies_m_is_one_l2119_211969


namespace NUMINAMATH_CALUDE_h_of_negative_one_l2119_211953

-- Define the functions
def f (x : ℝ) : ℝ := 3 * x + 7
def g (x : ℝ) : ℝ := (f x)^2 - 3
def h (x : ℝ) : ℝ := f (g x)

-- State the theorem
theorem h_of_negative_one : h (-1) = 298 := by
  sorry

end NUMINAMATH_CALUDE_h_of_negative_one_l2119_211953


namespace NUMINAMATH_CALUDE_frank_candy_count_l2119_211909

/-- Given a number of bags, pieces per bag, and leftover pieces, 
    calculates the total number of candy pieces. -/
def total_candy (bags : ℕ) (pieces_per_bag : ℕ) (leftover : ℕ) : ℕ :=
  bags * pieces_per_bag + leftover

/-- Proves that with 37 bags of 46 pieces each and 5 leftover pieces, 
    the total number of candy pieces is 1707. -/
theorem frank_candy_count : total_candy 37 46 5 = 1707 := by
  sorry

end NUMINAMATH_CALUDE_frank_candy_count_l2119_211909


namespace NUMINAMATH_CALUDE_exactly_five_numbers_l2119_211992

/-- A function that returns the number of ways a positive integer can be written as the sum of consecutive positive odd integers -/
def numConsecutiveOddSums (n : ℕ) : ℕ := sorry

/-- A function that checks if a positive integer is less than 100 and can be written as the sum of consecutive positive odd integers in exactly 3 different ways -/
def isValidNumber (n : ℕ) : Prop :=
  n < 100 ∧ numConsecutiveOddSums n = 3

/-- The main theorem stating that there are exactly 5 numbers satisfying the conditions -/
theorem exactly_five_numbers :
  ∃ (S : Finset ℕ), S.card = 5 ∧ ∀ n, n ∈ S ↔ isValidNumber n :=
sorry

end NUMINAMATH_CALUDE_exactly_five_numbers_l2119_211992


namespace NUMINAMATH_CALUDE_optimal_strategy_l2119_211947

/-- Represents the clothing types --/
inductive ClothingType
| A
| B

/-- Represents the cost and selling price of each clothing type --/
def clothingInfo : ClothingType → (ℕ × ℕ)
| ClothingType.A => (80, 120)
| ClothingType.B => (60, 90)

/-- The total number of clothing items --/
def totalClothing : ℕ := 100

/-- The maximum total cost allowed --/
def maxTotalCost : ℕ := 7500

/-- The minimum number of type A clothing --/
def minTypeA : ℕ := 65

/-- The maximum number of type A clothing --/
def maxTypeA : ℕ := 75

/-- Calculates the total profit given the number of type A clothing and the discount --/
def totalProfit (x : ℕ) (a : ℚ) : ℚ :=
  (10 - a) * x + 3000

/-- Represents the optimal purchase strategy --/
structure OptimalStrategy where
  typeACount : ℕ
  typeBCount : ℕ

/-- Theorem stating the optimal purchase strategy based on the discount --/
theorem optimal_strategy (a : ℚ) (h1 : 0 < a) (h2 : a < 20) :
  (∃ (strategy : OptimalStrategy),
    (0 < a ∧ a < 10 → strategy.typeACount = maxTypeA ∧ strategy.typeBCount = totalClothing - maxTypeA) ∧
    (a = 10 → strategy.typeACount ≥ minTypeA ∧ strategy.typeACount ≤ maxTypeA) ∧
    (10 < a ∧ a < 20 → strategy.typeACount = minTypeA ∧ strategy.typeBCount = totalClothing - minTypeA) ∧
    (∀ (x : ℕ), minTypeA ≤ x → x ≤ maxTypeA → totalProfit strategy.typeACount a ≥ totalProfit x a)) :=
sorry

end NUMINAMATH_CALUDE_optimal_strategy_l2119_211947


namespace NUMINAMATH_CALUDE_adi_change_l2119_211970

/-- The change Adi receives when buying a pencil -/
theorem adi_change (pencil_cost : ℕ) (payment : ℕ) (change : ℕ) : 
  pencil_cost = 35 →
  payment = 100 →
  change = payment - pencil_cost →
  change = 65 :=
by
  sorry

end NUMINAMATH_CALUDE_adi_change_l2119_211970


namespace NUMINAMATH_CALUDE_box_surface_area_l2119_211916

/-- Represents the dimensions of a rectangle --/
structure Rectangle where
  length : ℕ
  width : ℕ

/-- Calculates the area of a rectangle --/
def rectangleArea (r : Rectangle) : ℕ := r.length * r.width

/-- Represents the cardboard sheet with cut corners --/
structure CutCardboard where
  sheet : Rectangle
  smallCutSize : ℕ
  largeCutSize : ℕ

/-- Calculates the surface area of the interior of the box formed from the cut cardboard --/
def interiorSurfaceArea (c : CutCardboard) : ℕ :=
  rectangleArea c.sheet -
  (2 * rectangleArea ⟨c.smallCutSize, c.smallCutSize⟩) -
  (2 * rectangleArea ⟨c.largeCutSize, c.largeCutSize⟩)

theorem box_surface_area :
  let cardboard : CutCardboard := {
    sheet := { length := 35, width := 25 },
    smallCutSize := 3,
    largeCutSize := 4
  }
  interiorSurfaceArea cardboard = 825 := by
  sorry

end NUMINAMATH_CALUDE_box_surface_area_l2119_211916


namespace NUMINAMATH_CALUDE_periodic_function_value_l2119_211946

def is_periodic (f : ℝ → ℝ) (p : ℝ) : Prop :=
  ∀ x, f (x + p) = f x

theorem periodic_function_value (f : ℝ → ℝ) (h1 : is_periodic f 1.5) (h2 : f 1 = 20) :
  f 13 = 20 := by
  sorry

end NUMINAMATH_CALUDE_periodic_function_value_l2119_211946


namespace NUMINAMATH_CALUDE_contrapositive_equivalence_l2119_211908

theorem contrapositive_equivalence (a b : ℝ) :
  (¬(a = 0 ∧ b = 0) → a^2 + b^2 ≠ 0) ↔ (a^2 + b^2 = 0 → a = 0 ∧ b = 0) :=
by sorry

end NUMINAMATH_CALUDE_contrapositive_equivalence_l2119_211908


namespace NUMINAMATH_CALUDE_least_divisible_by_second_smallest_consecutive_primes_l2119_211984

def second_smallest_consecutive_primes : List Nat := [11, 13, 17, 19]

theorem least_divisible_by_second_smallest_consecutive_primes :
  (∀ n : Nat, n > 0 ∧ (∀ p ∈ second_smallest_consecutive_primes, p ∣ n) → n ≥ 46189) ∧
  (∀ p ∈ second_smallest_consecutive_primes, p ∣ 46189) :=
sorry

end NUMINAMATH_CALUDE_least_divisible_by_second_smallest_consecutive_primes_l2119_211984


namespace NUMINAMATH_CALUDE_largest_four_digit_divisible_by_six_l2119_211941

theorem largest_four_digit_divisible_by_six : 
  ∀ n : ℕ, n ≤ 9999 ∧ n ≥ 1000 ∧ n % 6 = 0 → n ≤ 9996 :=
by
  sorry

end NUMINAMATH_CALUDE_largest_four_digit_divisible_by_six_l2119_211941


namespace NUMINAMATH_CALUDE_unique_solution_system_l2119_211959

theorem unique_solution_system (x y z : ℝ) : 
  x * (1 + y * z) = 9 ∧ 
  y * (1 + x * z) = 12 ∧ 
  z * (1 + x * y) = 10 ↔ 
  x = 1 ∧ y = 4 ∧ z = 2 :=
by sorry

end NUMINAMATH_CALUDE_unique_solution_system_l2119_211959


namespace NUMINAMATH_CALUDE_odd_swaps_change_perm_l2119_211945

/-- Represents a permutation of three elements -/
inductive Perm3
  | abc
  | acb
  | bac
  | bca
  | cab
  | cba

/-- Represents whether a permutation is "correct" or "incorrect" -/
def isCorrect (p : Perm3) : Bool :=
  match p with
  | Perm3.abc => true
  | Perm3.bca => true
  | Perm3.cab => true
  | _ => false

/-- Represents a single adjacent swap -/
def swap (p : Perm3) : Perm3 :=
  match p with
  | Perm3.abc => Perm3.acb
  | Perm3.acb => Perm3.abc
  | Perm3.bac => Perm3.bca
  | Perm3.bca => Perm3.bac
  | Perm3.cab => Perm3.cba
  | Perm3.cba => Perm3.cab

/-- Theorem: After an odd number of swaps, the permutation cannot be the same as the initial one -/
theorem odd_swaps_change_perm (n : Nat) (h : Odd n) (p : Perm3) :
  (n.iterate swap p) ≠ p :=
  sorry

#check odd_swaps_change_perm

end NUMINAMATH_CALUDE_odd_swaps_change_perm_l2119_211945


namespace NUMINAMATH_CALUDE_min_max_quadratic_form_l2119_211920

theorem min_max_quadratic_form (x y : ℝ) (h : 9*x^2 + 12*x*y + 4*y^2 = 1) :
  let f := fun (x y : ℝ) => 3*x^2 + 4*x*y + 2*y^2
  ∃ (m M : ℝ), (∀ a b : ℝ, m ≤ f a b ∧ f a b ≤ M) ∧ m = 1 ∧ M = 1 := by
  sorry

end NUMINAMATH_CALUDE_min_max_quadratic_form_l2119_211920


namespace NUMINAMATH_CALUDE_biathlon_run_distance_l2119_211922

/-- A biathlon consisting of a bicycle race and a running race. -/
structure Biathlon where
  total_distance : ℝ
  bicycle_distance : ℝ
  run_velocity : ℝ
  bicycle_velocity : ℝ

/-- The theorem stating that for a specific biathlon, the running distance is 10 miles. -/
theorem biathlon_run_distance (b : Biathlon) 
  (h1 : b.total_distance = 155) 
  (h2 : b.bicycle_distance = 145) 
  (h3 : b.run_velocity = 10)
  (h4 : b.bicycle_velocity = 29) : 
  b.total_distance - b.bicycle_distance = 10 := by
  sorry

end NUMINAMATH_CALUDE_biathlon_run_distance_l2119_211922


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_l2119_211943

theorem imaginary_part_of_z (z : ℂ) (h : Complex.I * (z + 1) = -3 + 2 * Complex.I) : 
  Complex.im z = 3 := by
sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_l2119_211943


namespace NUMINAMATH_CALUDE_cubic_real_root_l2119_211912

/-- Given a cubic polynomial ax³ + 3x² + bx - 125 = 0 where a and b are real numbers,
    if -3 - 4i is a root of this polynomial, then 5 is the real root of the polynomial. -/
theorem cubic_real_root (a b : ℝ) :
  (∃ (z : ℂ), z = -3 - 4*I ∧ a * z^3 + 3 * z^2 + b * z - 125 = 0) →
  (∃ (x : ℝ), x = 5 ∧ a * x^3 + 3 * x^2 + b * x - 125 = 0) :=
by sorry

end NUMINAMATH_CALUDE_cubic_real_root_l2119_211912
