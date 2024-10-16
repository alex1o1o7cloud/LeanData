import Mathlib

namespace NUMINAMATH_CALUDE_max_roses_for_budget_l204_20410

/-- Represents the different rose purchasing options --/
inductive RoseOption
  | Individual
  | OneDozen
  | TwoDozen
  | Bulk

/-- Returns the cost of a given rose option --/
def cost (option : RoseOption) : Rat :=
  match option with
  | RoseOption.Individual => 730/100
  | RoseOption.OneDozen => 36
  | RoseOption.TwoDozen => 50
  | RoseOption.Bulk => 200

/-- Returns the number of roses for a given option --/
def roses (option : RoseOption) : Nat :=
  match option with
  | RoseOption.Individual => 1
  | RoseOption.OneDozen => 12
  | RoseOption.TwoDozen => 24
  | RoseOption.Bulk => 100

/-- Represents a purchase of roses --/
structure Purchase where
  individual : Nat
  oneDozen : Nat
  twoDozen : Nat
  bulk : Nat

/-- Calculates the total cost of a purchase --/
def totalCost (p : Purchase) : Rat :=
  p.individual * cost RoseOption.Individual +
  p.oneDozen * cost RoseOption.OneDozen +
  p.twoDozen * cost RoseOption.TwoDozen +
  p.bulk * cost RoseOption.Bulk

/-- Calculates the total number of roses in a purchase --/
def totalRoses (p : Purchase) : Nat :=
  p.individual * roses RoseOption.Individual +
  p.oneDozen * roses RoseOption.OneDozen +
  p.twoDozen * roses RoseOption.TwoDozen +
  p.bulk * roses RoseOption.Bulk

/-- The budget constraint --/
def budget : Rat := 680

/-- Theorem: The maximum number of roses that can be purchased for $680 is 328 --/
theorem max_roses_for_budget :
  ∃ (p : Purchase),
    totalCost p ≤ budget ∧
    totalRoses p = 328 ∧
    ∀ (q : Purchase), totalCost q ≤ budget → totalRoses q ≤ totalRoses p :=
by sorry


end NUMINAMATH_CALUDE_max_roses_for_budget_l204_20410


namespace NUMINAMATH_CALUDE_rectangle_area_l204_20446

theorem rectangle_area (r : ℝ) (ratio : ℝ) : r = 6 ∧ ratio = 3 →
  ∃ (length width : ℝ),
    width = 2 * r ∧
    length = ratio * width ∧
    length * width = 432 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_l204_20446


namespace NUMINAMATH_CALUDE_product_of_numbers_with_given_sum_and_difference_l204_20402

theorem product_of_numbers_with_given_sum_and_difference :
  ∀ x y : ℝ, x + y = 60 ∧ x - y = 10 → x * y = 875 := by
  sorry

end NUMINAMATH_CALUDE_product_of_numbers_with_given_sum_and_difference_l204_20402


namespace NUMINAMATH_CALUDE_vector_sum_magnitude_l204_20408

/-- The angle between two vectors in radians -/
def angle_between (a b : ℝ × ℝ) : ℝ := sorry

/-- The magnitude (length) of a vector -/
def magnitude (v : ℝ × ℝ) : ℝ := sorry

theorem vector_sum_magnitude (a b : ℝ × ℝ) 
  (h1 : angle_between a b = π / 3)  -- 60° in radians
  (h2 : a = (2, 0))
  (h3 : magnitude b = 1) :
  magnitude (a + 2 • b) = 2 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_vector_sum_magnitude_l204_20408


namespace NUMINAMATH_CALUDE_least_number_of_trees_l204_20440

theorem least_number_of_trees : ∃ n : ℕ, 
  (∀ m : ℕ, m < n → (m % 7 ≠ 0 ∨ m % 6 ≠ 0 ∨ m % 4 ≠ 0)) ∧
  n % 7 = 0 ∧ n % 6 = 0 ∧ n % 4 = 0 ∧
  n = 84 := by
sorry

end NUMINAMATH_CALUDE_least_number_of_trees_l204_20440


namespace NUMINAMATH_CALUDE_statement_is_proposition_l204_20443

def is_proposition (statement : Prop) : Prop :=
  statement ∨ ¬statement

theorem statement_is_proposition : is_proposition (20 - 5 * 3 = 10) := by
  sorry

end NUMINAMATH_CALUDE_statement_is_proposition_l204_20443


namespace NUMINAMATH_CALUDE_geometric_sequence_problem_l204_20404

/-- Given a geometric sequence {a_n} with sum S_n = 3^n + t, prove a_2 = 6 and t = -1 -/
theorem geometric_sequence_problem (a : ℕ → ℝ) (S : ℕ → ℝ) (t : ℝ) 
  (h_geom : ∀ n : ℕ, a (n + 1) / a n = a 2 / a 1) -- geometric sequence condition
  (h_sum : ∀ n : ℕ, S n = 3^n + t) -- sum condition
  : a 2 = 6 ∧ t = -1 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_problem_l204_20404


namespace NUMINAMATH_CALUDE_expand_polynomial_product_l204_20467

theorem expand_polynomial_product : ∀ x : ℝ,
  (3 * x^2 - 2 * x + 4) * (-4 * x^2 + 3 * x - 6) =
  -12 * x^4 + 17 * x^3 - 40 * x^2 + 24 * x - 24 :=
by
  sorry

end NUMINAMATH_CALUDE_expand_polynomial_product_l204_20467


namespace NUMINAMATH_CALUDE_least_possible_c_l204_20421

theorem least_possible_c (a b c : ℕ+) : 
  (a + b + c : ℚ) / 3 = 20 →
  a ≤ b →
  b ≤ c →
  b = a + 13 →
  c ≥ 45 ∧ ∃ (a₀ b₀ c₀ : ℕ+), 
    (a₀ + b₀ + c₀ : ℚ) / 3 = 20 ∧
    a₀ ≤ b₀ ∧
    b₀ ≤ c₀ ∧
    b₀ = a₀ + 13 ∧
    c₀ = 45 :=
by sorry

end NUMINAMATH_CALUDE_least_possible_c_l204_20421


namespace NUMINAMATH_CALUDE_difference_positive_inequality_l204_20436

theorem difference_positive_inequality (x : ℝ) :
  (1 / 3 * x - x > 0) ↔ (-2 / 3 * x > 0) := by sorry

end NUMINAMATH_CALUDE_difference_positive_inequality_l204_20436


namespace NUMINAMATH_CALUDE_cylinder_volume_ratio_l204_20445

/-- Given two cylinders with the following properties:
  * S₁ and S₂ are their base areas
  * υ₁ and υ₂ are their volumes
  * They have equal lateral areas
  * S₁/S₂ = 16/9
Then υ₁/υ₂ = 4/3 -/
theorem cylinder_volume_ratio (S₁ S₂ υ₁ υ₂ : ℝ) (h_positive : S₁ > 0 ∧ S₂ > 0 ∧ υ₁ > 0 ∧ υ₂ > 0)
    (h_base_ratio : S₁ / S₂ = 16 / 9) (h_equal_lateral : ∃ (r₁ r₂ h₁ h₂ : ℝ), 
    S₁ = π * r₁^2 ∧ S₂ = π * r₂^2 ∧ υ₁ = S₁ * h₁ ∧ υ₂ = S₂ * h₂ ∧ 2 * π * r₁ * h₁ = 2 * π * r₂ * h₂) :
  υ₁ / υ₂ = 4 / 3 := by
  sorry

end NUMINAMATH_CALUDE_cylinder_volume_ratio_l204_20445


namespace NUMINAMATH_CALUDE_money_split_proof_l204_20449

/-- The total amount of money found by Donna and her friends -/
def total_money : ℝ := 97.50

/-- Donna's share of the money as a percentage -/
def donna_share : ℝ := 0.40

/-- The amount Donna received in dollars -/
def donna_amount : ℝ := 39

/-- Theorem stating that if Donna received 40% of the total money and her share was $39, 
    then the total amount of money found was $97.50 -/
theorem money_split_proof : 
  donna_share * total_money = donna_amount → total_money = 97.50 := by
  sorry

end NUMINAMATH_CALUDE_money_split_proof_l204_20449


namespace NUMINAMATH_CALUDE_resultant_polyhedron_edges_l204_20425

-- Define the convex polyhedron S
structure ConvexPolyhedron :=
  (vertices : ℕ)
  (edges : ℕ)

-- Define the operation of intersecting S with planes
def intersect_with_planes (S : ConvexPolyhedron) (num_planes : ℕ) : ℕ :=
  S.edges * 2 + S.edges

-- Theorem statement
theorem resultant_polyhedron_edges 
  (S : ConvexPolyhedron) 
  (h1 : S.vertices = S.vertices) 
  (h2 : S.edges = 150) :
  intersect_with_planes S S.vertices = 450 := by
  sorry

end NUMINAMATH_CALUDE_resultant_polyhedron_edges_l204_20425


namespace NUMINAMATH_CALUDE_total_pencils_l204_20455

/-- Given the ages and pencil counts of Asaf and Alexander, prove their total pencil count -/
theorem total_pencils (asaf_age alexander_age asaf_pencils alexander_pencils : ℕ) : 
  asaf_age = 50 →
  asaf_age + alexander_age = 140 →
  alexander_age - asaf_age = asaf_pencils / 2 →
  alexander_pencils = asaf_pencils + 60 →
  asaf_pencils + alexander_pencils = 220 := by
sorry


end NUMINAMATH_CALUDE_total_pencils_l204_20455


namespace NUMINAMATH_CALUDE_kids_from_unnamed_school_l204_20434

theorem kids_from_unnamed_school (riverside_total : ℕ) (mountaintop_total : ℕ) (total_admitted : ℕ)
  (riverside_denied_percent : ℚ) (mountaintop_denied_percent : ℚ) (unnamed_denied_percent : ℚ)
  (h1 : riverside_total = 120)
  (h2 : mountaintop_total = 50)
  (h3 : total_admitted = 148)
  (h4 : riverside_denied_percent = 1/5)
  (h5 : mountaintop_denied_percent = 1/2)
  (h6 : unnamed_denied_percent = 7/10) :
  ∃ (unnamed_total : ℕ),
    unnamed_total = 90 ∧
    total_admitted = 
      (riverside_total - riverside_total * riverside_denied_percent) +
      (mountaintop_total - mountaintop_total * mountaintop_denied_percent) +
      (unnamed_total - unnamed_total * unnamed_denied_percent) :=
by sorry

end NUMINAMATH_CALUDE_kids_from_unnamed_school_l204_20434


namespace NUMINAMATH_CALUDE_smith_children_age_problem_l204_20414

theorem smith_children_age_problem :
  ∀ (age1 age2 age3 : ℕ),
  age1 = 6 →
  age2 = 8 →
  (age1 + age2 + age3) / 3 = 9 →
  age3 = 13 := by
sorry

end NUMINAMATH_CALUDE_smith_children_age_problem_l204_20414


namespace NUMINAMATH_CALUDE_min_processed_area_l204_20490

/-- Represents the dimensions of a rectangular sheet -/
structure SheetDimensions where
  length : ℝ
  width : ℝ

/-- Calculates the area of a rectangular sheet -/
def area (d : SheetDimensions) : ℝ := d.length * d.width

/-- Applies the processing shrinkage to the sheet dimensions -/
def applyProcessing (d : SheetDimensions) (shrinkFactor : ℝ) : SheetDimensions :=
  { length := d.length * shrinkFactor,
    width := d.width * shrinkFactor }

/-- Theorem: The minimum possible area of the processed sheet is 12.15 square inches -/
theorem min_processed_area (reportedLength reportedWidth errorMargin shrinkFactor : ℝ) 
    (hLength : reportedLength = 6)
    (hWidth : reportedWidth = 4)
    (hError : errorMargin = 1)
    (hShrink : shrinkFactor = 0.9)
    : ∃ (d : SheetDimensions),
      d.length ≥ reportedLength - errorMargin ∧
      d.length ≤ reportedLength + errorMargin ∧
      d.width ≥ reportedWidth - errorMargin ∧
      d.width ≤ reportedWidth + errorMargin ∧
      area (applyProcessing d shrinkFactor) ≥ 12.15 ∧
      ∀ (d' : SheetDimensions),
        d'.length ≥ reportedLength - errorMargin →
        d'.length ≤ reportedLength + errorMargin →
        d'.width ≥ reportedWidth - errorMargin →
        d'.width ≤ reportedWidth + errorMargin →
        area (applyProcessing d' shrinkFactor) ≥ area (applyProcessing d shrinkFactor) :=
by sorry

end NUMINAMATH_CALUDE_min_processed_area_l204_20490


namespace NUMINAMATH_CALUDE_prime_between_squares_l204_20415

theorem prime_between_squares : ∃ p : ℕ, 
  Prime p ∧ 
  (∃ n : ℕ, p = n^2 + 9) ∧ 
  (∃ m : ℕ, p = (m+1)^2 - 8) ∧ 
  p = 73 := by
sorry

end NUMINAMATH_CALUDE_prime_between_squares_l204_20415


namespace NUMINAMATH_CALUDE_subset_complement_implies_a_negative_l204_20496

theorem subset_complement_implies_a_negative 
  (I : Set ℝ) 
  (A B : Set ℝ) 
  (a : ℝ) 
  (h_I : I = Set.univ) 
  (h_A : A = {x : ℝ | x ≤ a + 1}) 
  (h_B : B = {x : ℝ | x ≥ 1}) 
  (h_subset : A ⊆ (I \ B)) : 
  a < 0 := by
sorry

end NUMINAMATH_CALUDE_subset_complement_implies_a_negative_l204_20496


namespace NUMINAMATH_CALUDE_hunter_journey_l204_20462

theorem hunter_journey (swamp_speed forest_speed highway_speed : ℝ)
  (total_time total_distance : ℝ) (swamp_time forest_time highway_time : ℝ) :
  swamp_speed = 2 →
  forest_speed = 4 →
  highway_speed = 6 →
  total_time = 4 →
  total_distance = 15 →
  swamp_time + forest_time + highway_time = total_time →
  swamp_speed * swamp_time + forest_speed * forest_time + highway_speed * highway_time = total_distance →
  swamp_time > highway_time := by
  sorry

end NUMINAMATH_CALUDE_hunter_journey_l204_20462


namespace NUMINAMATH_CALUDE_fence_birds_count_l204_20477

/-- The number of birds on a fence after new birds land -/
def total_birds (initial_pairs : ℕ) (birds_per_pair : ℕ) (new_birds : ℕ) : ℕ :=
  initial_pairs * birds_per_pair + new_birds

/-- Theorem stating the total number of birds on the fence -/
theorem fence_birds_count :
  let initial_pairs : ℕ := 12
  let birds_per_pair : ℕ := 2
  let new_birds : ℕ := 8
  total_birds initial_pairs birds_per_pair new_birds = 32 := by
  sorry

end NUMINAMATH_CALUDE_fence_birds_count_l204_20477


namespace NUMINAMATH_CALUDE_equation_solutions_l204_20451

theorem equation_solutions :
  (∀ x : ℝ, (x + 1) / (x - 1) - 4 / (x^2 - 1) ≠ 1) ∧
  (∀ x : ℝ, x^2 + 3*x - 2 = 0 ↔ x = -3/2 - Real.sqrt 17/2 ∨ x = -3/2 + Real.sqrt 17/2) :=
by sorry

end NUMINAMATH_CALUDE_equation_solutions_l204_20451


namespace NUMINAMATH_CALUDE_barbara_paper_weight_l204_20405

/-- Calculates the total weight of paper removed from a chest of drawers -/
def total_weight_of_paper (
  colored_bundles : ℕ)
  (white_bunches : ℕ)
  (scrap_heaps : ℕ)
  (sheets_per_bunch : ℕ)
  (sheets_per_bundle : ℕ)
  (sheets_per_heap : ℕ)
  (colored_sheet_weight : ℚ)
  (white_sheet_weight : ℚ)
  (scrap_sheet_weight : ℚ) : ℚ :=
  let colored_sheets := colored_bundles * sheets_per_bundle
  let white_sheets := white_bunches * sheets_per_bunch
  let scrap_sheets := scrap_heaps * sheets_per_heap
  let colored_weight := colored_sheets * colored_sheet_weight
  let white_weight := white_sheets * white_sheet_weight
  let scrap_weight := scrap_sheets * scrap_sheet_weight
  colored_weight + white_weight + scrap_weight

theorem barbara_paper_weight :
  total_weight_of_paper 3 2 5 4 2 20 (3/100) (1/20) (1/25) = 458/100 := by
  sorry

end NUMINAMATH_CALUDE_barbara_paper_weight_l204_20405


namespace NUMINAMATH_CALUDE_inequality_solution_set_l204_20495

theorem inequality_solution_set (a : ℝ) (h : 0 < a ∧ a < 1) :
  {x : ℝ | (x - a) * (x - 1/a) > 0} = {x : ℝ | a < x ∧ x < 1/a} := by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l204_20495


namespace NUMINAMATH_CALUDE_max_xy_on_line_segment_l204_20488

/-- The maximum value of xy for a point P(x,y) on the line segment AB, where A(3,0) and B(0,4) -/
theorem max_xy_on_line_segment : ∀ x y : ℝ, 
  (x / 3 + y / 4 = 1) → -- Point P(x,y) is on the line segment AB
  (x ≥ 0 ∧ y ≥ 0) →    -- P is between A and B (non-negative coordinates)
  (x ≤ 3 ∧ y ≤ 4) →    -- P is between A and B (upper bounds)
  x * y ≤ 3 :=         -- The maximum value of xy is 3
by
  sorry

#check max_xy_on_line_segment

end NUMINAMATH_CALUDE_max_xy_on_line_segment_l204_20488


namespace NUMINAMATH_CALUDE_third_restaurant_meals_l204_20444

/-- The number of meals served by Gordon's third restaurant per day -/
def third_restaurant_meals_per_day (
  total_restaurants : ℕ)
  (first_restaurant_meals_per_day : ℕ)
  (second_restaurant_meals_per_day : ℕ)
  (total_meals_per_week : ℕ) : ℕ :=
  (total_meals_per_week - 7 * (first_restaurant_meals_per_day + second_restaurant_meals_per_day)) / 7

theorem third_restaurant_meals (
  total_restaurants : ℕ)
  (first_restaurant_meals_per_day : ℕ)
  (second_restaurant_meals_per_day : ℕ)
  (total_meals_per_week : ℕ)
  (h1 : total_restaurants = 3)
  (h2 : first_restaurant_meals_per_day = 20)
  (h3 : second_restaurant_meals_per_day = 40)
  (h4 : total_meals_per_week = 770) :
  third_restaurant_meals_per_day total_restaurants first_restaurant_meals_per_day second_restaurant_meals_per_day total_meals_per_week = 50 :=
by
  sorry

end NUMINAMATH_CALUDE_third_restaurant_meals_l204_20444


namespace NUMINAMATH_CALUDE_f_one_zero_implies_a_gt_one_l204_20460

/-- A function f(x) = 2ax^2 - x - 1 with parameter a -/
def f (a : ℝ) (x : ℝ) : ℝ := 2*a*x^2 - x - 1

/-- The theorem stating that if f has exactly one zero in (0,1), then a > 1 -/
theorem f_one_zero_implies_a_gt_one (a : ℝ) :
  (∃! x, x ∈ (Set.Ioo 0 1) ∧ f a x = 0) → a > 1 := by
  sorry

#check f_one_zero_implies_a_gt_one

end NUMINAMATH_CALUDE_f_one_zero_implies_a_gt_one_l204_20460


namespace NUMINAMATH_CALUDE_divisor_of_power_of_four_l204_20432

theorem divisor_of_power_of_four (a : ℕ) (d : ℕ) (h1 : a > 0) (h2 : 2 ∣ a) :
  let p := 4^a
  (p % d = 6) → d = 22 := by
sorry

end NUMINAMATH_CALUDE_divisor_of_power_of_four_l204_20432


namespace NUMINAMATH_CALUDE_krystiana_monthly_earnings_l204_20452

/-- Represents the apartment building owned by Krystiana -/
structure ApartmentBuilding where
  firstFloorRate : ℕ
  secondFloorRate : ℕ
  thirdFloorRate : ℕ
  roomsPerFloor : ℕ
  occupiedThirdFloorRooms : ℕ

/-- Calculates the monthly earnings from Krystiana's apartment building -/
def calculateMonthlyEarnings (building : ApartmentBuilding) : ℕ :=
  building.firstFloorRate * building.roomsPerFloor +
  building.secondFloorRate * building.roomsPerFloor +
  building.thirdFloorRate * building.occupiedThirdFloorRooms

/-- Theorem stating that Krystiana's monthly earnings are $165 -/
theorem krystiana_monthly_earnings :
  ∀ (building : ApartmentBuilding),
    building.firstFloorRate = 15 →
    building.secondFloorRate = 20 →
    building.thirdFloorRate = 2 * building.firstFloorRate →
    building.roomsPerFloor = 3 →
    building.occupiedThirdFloorRooms = 2 →
    calculateMonthlyEarnings building = 165 := by
  sorry

#eval calculateMonthlyEarnings {
  firstFloorRate := 15,
  secondFloorRate := 20,
  thirdFloorRate := 30,
  roomsPerFloor := 3,
  occupiedThirdFloorRooms := 2
}

end NUMINAMATH_CALUDE_krystiana_monthly_earnings_l204_20452


namespace NUMINAMATH_CALUDE_replaced_person_weight_l204_20497

/-- Proves that the weight of the replaced person is 55 kg given the conditions of the problem. -/
theorem replaced_person_weight
  (initial_count : Nat)
  (weight_increase : ℝ)
  (new_person_weight : ℝ)
  (h1 : initial_count = 8)
  (h2 : weight_increase = 2.5)
  (h3 : new_person_weight = 75) :
  (new_person_weight - initial_count * weight_increase) = 55 := by
  sorry

end NUMINAMATH_CALUDE_replaced_person_weight_l204_20497


namespace NUMINAMATH_CALUDE_inequality_proof_l204_20400

theorem inequality_proof (x y : ℝ) (n : ℕ) (h : x^2 + y^2 ≤ 1) :
  (x^n + y)^2 + y^2 ≥ (1 / (n + 2 : ℝ)) * (x^2 + y^2)^n :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_l204_20400


namespace NUMINAMATH_CALUDE_opposite_numbers_not_just_opposite_signs_l204_20478

theorem opposite_numbers_not_just_opposite_signs : ¬ (∀ a b : ℝ, (a > 0 ∧ b < 0) → (a = -b)) := by
  sorry

end NUMINAMATH_CALUDE_opposite_numbers_not_just_opposite_signs_l204_20478


namespace NUMINAMATH_CALUDE_taco_truck_beef_per_taco_l204_20469

theorem taco_truck_beef_per_taco 
  (total_beef : ℝ) 
  (selling_price : ℝ) 
  (cost_per_taco : ℝ) 
  (total_profit : ℝ) 
  (h1 : total_beef = 100)
  (h2 : selling_price = 2)
  (h3 : cost_per_taco = 1.5)
  (h4 : total_profit = 200) :
  ∃ (beef_per_taco : ℝ), 
    beef_per_taco = 1/4 ∧ 
    (total_beef / beef_per_taco) * (selling_price - cost_per_taco) = total_profit := by
  sorry

end NUMINAMATH_CALUDE_taco_truck_beef_per_taco_l204_20469


namespace NUMINAMATH_CALUDE_apollo_chariot_wheels_cost_l204_20479

/-- Represents the cost in golden apples for chariot wheels over a year -/
structure ChariotWheelsCost where
  total : ℕ  -- Total cost for the year
  second_half_multiplier : ℕ  -- Multiplier for the second half of the year

/-- 
Calculates the cost for the first half of the year given the total cost
and the multiplier for the second half of the year.
-/
def first_half_cost (c : ChariotWheelsCost) : ℕ :=
  c.total / (1 + c.second_half_multiplier)

/-- 
Theorem: If the total cost for a year is 54 golden apples, and the cost for the 
second half of the year is double the cost for the first half, then the cost 
for the first half of the year is 18 golden apples.
-/
theorem apollo_chariot_wheels_cost : 
  let c := ChariotWheelsCost.mk 54 2
  first_half_cost c = 18 := by
  sorry

end NUMINAMATH_CALUDE_apollo_chariot_wheels_cost_l204_20479


namespace NUMINAMATH_CALUDE_tangent_triangle_area_l204_20458

/-- The area of the triangle formed by the tangent line at (1, e^(-1)) on y = e^(-x) and the axes is 2/e -/
theorem tangent_triangle_area :
  let f : ℝ → ℝ := fun x ↦ Real.exp (-x)
  let M : ℝ × ℝ := (1, Real.exp (-1))
  let tangent_line (x : ℝ) : ℝ := -Real.exp (-1) * (x - 1) + Real.exp (-1)
  let x_intercept : ℝ := 2
  let y_intercept : ℝ := 2 * Real.exp (-1)
  let triangle_area : ℝ := (1/2) * x_intercept * y_intercept
  triangle_area = 2 / Real.exp 1 :=
by sorry


end NUMINAMATH_CALUDE_tangent_triangle_area_l204_20458


namespace NUMINAMATH_CALUDE_lcm_problem_l204_20484

theorem lcm_problem (a b : ℕ+) (h1 : b = 852) (h2 : Nat.lcm a b = 5964) : a = 852 := by
  sorry

end NUMINAMATH_CALUDE_lcm_problem_l204_20484


namespace NUMINAMATH_CALUDE_shortest_distance_ln_to_line_l204_20468

open Real

theorem shortest_distance_ln_to_line (x : ℝ) : 
  let g (x : ℝ) := log x
  let P : ℝ × ℝ := (x, g x)
  let d (p : ℝ × ℝ) := |p.1 - p.2| / sqrt 2
  ∃ (x₀ : ℝ), x₀ > 0 ∧ ∀ (x : ℝ), x > 0 → d P ≥ d (x₀, g x₀) ∧ d (x₀, g x₀) = 1 / sqrt 2 :=
sorry

end NUMINAMATH_CALUDE_shortest_distance_ln_to_line_l204_20468


namespace NUMINAMATH_CALUDE_knocks_to_knicks_conversion_l204_20417

/-- Conversion rate between knicks and knacks -/
def knicks_to_knacks : ℚ := 3 / 5

/-- Conversion rate between knacks and knocks -/
def knacks_to_knocks : ℚ := 7 / 4

/-- The number of knocks we want to convert -/
def target_knocks : ℕ := 28

/-- Theorem stating the equivalence between knocks and knicks -/
theorem knocks_to_knicks_conversion :
  (target_knocks : ℚ) * knacks_to_knocks⁻¹ * knicks_to_knacks⁻¹ = 80 / 3 := by
  sorry

end NUMINAMATH_CALUDE_knocks_to_knicks_conversion_l204_20417


namespace NUMINAMATH_CALUDE_triangle_problem_l204_20464

theorem triangle_problem (a b c A B C : ℝ) 
  (h1 : Real.sqrt 3 * c * Real.sin A = a * Real.cos C)
  (h2 : c = 2 * a)
  (h3 : b = 2 * Real.sqrt 3)
  (h4 : 0 < a ∧ 0 < b ∧ 0 < c)
  (h5 : 0 < A ∧ A < π)
  (h6 : 0 < B ∧ B < π)
  (h7 : 0 < C ∧ C < π)
  (h8 : A + B + C = π) :
  C = π / 6 ∧ 
  (1/2 * a * b * Real.sin C = (Real.sqrt 15 - Real.sqrt 3) / 2) := by
  sorry

end NUMINAMATH_CALUDE_triangle_problem_l204_20464


namespace NUMINAMATH_CALUDE_sixth_root_square_equation_solution_l204_20431

theorem sixth_root_square_equation_solution :
  ∃ (x : ℝ), x > 0 ∧ (((x * (x^4)^(1/2))^(1/6))^2 = 4) ∧ x = 4 := by
sorry

end NUMINAMATH_CALUDE_sixth_root_square_equation_solution_l204_20431


namespace NUMINAMATH_CALUDE_product_from_gcd_lcm_l204_20499

theorem product_from_gcd_lcm (a b : ℤ) : 
  Int.gcd a b = 8 → Int.lcm a b = 24 → a * b = 192 := by
  sorry

end NUMINAMATH_CALUDE_product_from_gcd_lcm_l204_20499


namespace NUMINAMATH_CALUDE_trig_expression_equals_one_l204_20439

theorem trig_expression_equals_one : 
  (Real.tan (30 * π / 180))^2 - (Real.sin (30 * π / 180))^2 = 
  (Real.tan (30 * π / 180))^2 * (Real.sin (30 * π / 180))^2 := by
  sorry

end NUMINAMATH_CALUDE_trig_expression_equals_one_l204_20439


namespace NUMINAMATH_CALUDE_fermatville_temperature_range_l204_20453

/-- The temperature range in Fermatville on Monday -/
def temperature_range (min_temp max_temp : Int) : Int :=
  max_temp - min_temp

/-- Theorem: The temperature range in Fermatville on Monday was 25°C -/
theorem fermatville_temperature_range :
  let min_temp : Int := -11
  let max_temp : Int := 14
  temperature_range min_temp max_temp = 25 := by
  sorry

end NUMINAMATH_CALUDE_fermatville_temperature_range_l204_20453


namespace NUMINAMATH_CALUDE_johns_age_l204_20427

theorem johns_age (john dad brother : ℕ) 
  (h1 : john + 28 = dad)
  (h2 : john + dad = 76)
  (h3 : john + 5 = 2 * (brother + 5)) : 
  john = 24 := by
  sorry

end NUMINAMATH_CALUDE_johns_age_l204_20427


namespace NUMINAMATH_CALUDE_sum_in_hex_l204_20485

-- Define the binary numbers
def binary_num1 : ℕ := 11111111111
def binary_num2 : ℕ := 11111111

-- Define the base of the binary system
def binary_base : ℕ := 2

-- Define the base of the target system
def target_base : ℕ := 16

-- Function to convert from binary to decimal
def binary_to_decimal (n : ℕ) : ℕ := sorry

-- Function to convert from decimal to hexadecimal
def decimal_to_hex (n : ℕ) : String := sorry

-- Theorem statement
theorem sum_in_hex : 
  decimal_to_hex (binary_to_decimal binary_num1 + binary_to_decimal binary_num2) = "8FE" := by sorry

end NUMINAMATH_CALUDE_sum_in_hex_l204_20485


namespace NUMINAMATH_CALUDE_quadratic_equation_relation_l204_20476

theorem quadratic_equation_relation (x : ℝ) : 
  x^2 + 3*x + 5 = 7 → x^2 + 3*x - 2 = 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_relation_l204_20476


namespace NUMINAMATH_CALUDE_one_more_stork_than_birds_l204_20419

/-- Given the initial number of storks and birds on a fence, and additional birds that join,
    prove that there is one more stork than the total number of birds. -/
theorem one_more_stork_than_birds 
  (initial_storks : ℕ) 
  (initial_birds : ℕ) 
  (new_birds : ℕ) 
  (h1 : initial_storks = 6) 
  (h2 : initial_birds = 2) 
  (h3 : new_birds = 3) :
  initial_storks - (initial_birds + new_birds) = 1 := by
  sorry

end NUMINAMATH_CALUDE_one_more_stork_than_birds_l204_20419


namespace NUMINAMATH_CALUDE_condition_relationship_l204_20498

/-- Given propositions p, q, and r, we define what it means for one proposition
    to be a sufficient but not necessary condition for another. -/
def sufficient_not_necessary (a b : Prop) : Prop :=
  (a → b) ∧ ¬(b → a)

/-- Given propositions p, q, and r, we define what it means for one proposition
    to be a necessary but not sufficient condition for another. -/
def necessary_not_sufficient (a b : Prop) : Prop :=
  (b → a) ∧ ¬(a → b)

/-- Theorem stating the relationship between p, q, and r based on their conditional properties. -/
theorem condition_relationship (p q r : Prop) 
  (h1 : sufficient_not_necessary p q) 
  (h2 : sufficient_not_necessary q r) : 
  necessary_not_sufficient r p :=
sorry

end NUMINAMATH_CALUDE_condition_relationship_l204_20498


namespace NUMINAMATH_CALUDE_geometric_sequence_minimum_value_l204_20418

/-- A geometric sequence with positive terms -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, q > 0 ∧ ∀ n : ℕ, a (n + 1) = q * a n

theorem geometric_sequence_minimum_value
  (a : ℕ → ℝ)
  (h_geo : GeometricSequence a)
  (h_positive : ∀ n, a n > 0)
  (h_sqrt : ∃ m n : ℕ, Real.sqrt (a m * a n) = 4 * a 1)
  (h_sum : a 6 = a 5 + 2 * a 4) :
  ∃ m n : ℕ, (1 : ℝ) / m + 4 / n ≥ 3 / 2 ∧
    (∀ k l : ℕ, (1 : ℝ) / k + 4 / l ≥ 3 / 2) :=
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_minimum_value_l204_20418


namespace NUMINAMATH_CALUDE_power_of_two_equality_l204_20482

theorem power_of_two_equality (x : ℕ) : (1 / 16 : ℚ) * 2^50 = 2^x → x = 46 := by
  sorry

end NUMINAMATH_CALUDE_power_of_two_equality_l204_20482


namespace NUMINAMATH_CALUDE_calculation_proof_l204_20420

theorem calculation_proof : 
  |Real.sqrt 3 - 1| - (-Real.sqrt 3)^2 - 12 * (-1/3) = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_calculation_proof_l204_20420


namespace NUMINAMATH_CALUDE_ice_cube_volume_l204_20428

theorem ice_cube_volume (V : ℝ) : 
  V > 0 → -- Assume the original volume is positive
  (1/4 * (1/4 * V)) = 0.2 → -- After two hours, the volume is 0.2 cubic inches
  V = 3.2 := by
sorry

end NUMINAMATH_CALUDE_ice_cube_volume_l204_20428


namespace NUMINAMATH_CALUDE_solution_set_is_open_interval_l204_20473

open Set Real

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the conditions
variable (h1 : Differentiable ℝ f)
variable (h2 : ∀ x, (x - 1) * (deriv f x - f x) > 0)
variable (h3 : ∀ x, f (2 - x) = f x * exp (2 - 2*x))

-- Define the solution set
def solution_set := {x : ℝ | exp 2 * f (log x) < x * f 2}

-- State the theorem
theorem solution_set_is_open_interval :
  solution_set f = Ioo 1 (exp 2) :=
sorry

end NUMINAMATH_CALUDE_solution_set_is_open_interval_l204_20473


namespace NUMINAMATH_CALUDE_inequality_proof_l204_20480

theorem inequality_proof (x y z t : ℝ) 
  (non_neg_x : x ≥ 0) (non_neg_y : y ≥ 0) (non_neg_z : z ≥ 0) (non_neg_t : t ≥ 0)
  (sum_condition : x + y + z + t = 7) : 
  Real.sqrt (x^2 + y^2) + Real.sqrt (x^2 + 1) + Real.sqrt (z^2 + y^2) + 
  Real.sqrt (t^2 + 64) + Real.sqrt (z^2 + t^2) ≥ 17 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l204_20480


namespace NUMINAMATH_CALUDE_perpendicular_transitivity_l204_20456

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the perpendicular relation between lines and planes
variable (perp : Line → Plane → Prop)

-- Define the statement
theorem perpendicular_transitivity
  (m n : Line) (α β : Plane)
  (diff_lines : m ≠ n)
  (diff_planes : α ≠ β)
  (n_perp_α : perp n α)
  (n_perp_β : perp n β)
  (m_perp_β : perp m β) :
  perp m α :=
sorry

end NUMINAMATH_CALUDE_perpendicular_transitivity_l204_20456


namespace NUMINAMATH_CALUDE_exponential_simplification_l204_20403

theorem exponential_simplification :
  (10 ^ 1.4) * (10 ^ 0.5) / ((10 ^ 0.4) * (10 ^ 0.1)) = 10 ^ 1.4 := by
  sorry

end NUMINAMATH_CALUDE_exponential_simplification_l204_20403


namespace NUMINAMATH_CALUDE_green_space_equation_l204_20433

/-- Represents a rectangular green space -/
structure GreenSpace where
  length : ℝ
  width : ℝ
  area : ℝ

/-- Theorem stating the properties of the green space and the resulting equation -/
theorem green_space_equation (g : GreenSpace) 
  (h1 : g.area = 1000)
  (h2 : g.length = g.width + 30)
  (h3 : g.area = g.length * g.width) :
  g.length * (g.length - 30) = 1000 := by
  sorry

#check green_space_equation

end NUMINAMATH_CALUDE_green_space_equation_l204_20433


namespace NUMINAMATH_CALUDE_ashley_stair_climbing_time_l204_20474

def arithmetic_sequence_sum (a₁ : ℕ) (d : ℕ) (n : ℕ) : ℕ :=
  n * (2 * a₁ + (n - 1) * d) / 2

theorem ashley_stair_climbing_time :
  arithmetic_sequence_sum 30 10 4 = 180 := by
  sorry

end NUMINAMATH_CALUDE_ashley_stair_climbing_time_l204_20474


namespace NUMINAMATH_CALUDE_two_roots_implies_c_values_l204_20437

def f (c : ℝ) (x : ℝ) : ℝ := x^3 - 3*x + c

theorem two_roots_implies_c_values (c : ℝ) :
  (∃ x y : ℝ, x ≠ y ∧ f c x = 0 ∧ f c y = 0 ∧ ∀ z : ℝ, f c z = 0 → z = x ∨ z = y) →
  c = -2 ∨ c = 2 := by
  sorry

end NUMINAMATH_CALUDE_two_roots_implies_c_values_l204_20437


namespace NUMINAMATH_CALUDE_constant_term_expansion_l204_20407

theorem constant_term_expansion (n : ℕ+) 
  (h : (2 : ℝ)^(n : ℝ) = 32) : 
  Nat.choose n.val 3 = 10 := by
  sorry

end NUMINAMATH_CALUDE_constant_term_expansion_l204_20407


namespace NUMINAMATH_CALUDE_solution_range_l204_20429

theorem solution_range (x : ℝ) :
  x ≥ 2 →
  (Real.sqrt (x + 5 - 6 * Real.sqrt (x - 2)) + Real.sqrt (x + 12 - 8 * Real.sqrt (x - 2)) = 2) →
  x ∈ Set.Icc 11 18 :=
by sorry

end NUMINAMATH_CALUDE_solution_range_l204_20429


namespace NUMINAMATH_CALUDE_polynomial_evaluation_l204_20424

def P (x : ℤ) : ℤ :=
  x^15 - 2008*x^14 + 2008*x^13 - 2008*x^12 + 2008*x^11 - 2008*x^10 + 2008*x^9 - 2008*x^8 + 2008*x^7 - 2008*x^6 + 2008*x^5 - 2008*x^4 + 2008*x^3 - 2008*x^2 + 2008*x

theorem polynomial_evaluation : P 2007 = 2007 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_evaluation_l204_20424


namespace NUMINAMATH_CALUDE_quadratic_function_theorem_l204_20442

/-- The quadratic function f(x) = ax^2 + bx -/
def f (a b : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x

/-- The function g(x) = f(x) - x -/
def g (a b : ℝ) (x : ℝ) : ℝ := f a b x - x

theorem quadratic_function_theorem (a b : ℝ) (ha : a ≠ 0) :
  (∀ x, f a b (1 - x) = f a b (1 + x)) →
  (∃! x, g a b x = 0) →
  (f a b = fun x ↦ -1/2 * x^2 + x) ∧
  (∀ x ∈ Set.Icc (-4 : ℝ) 0, f a b x ∈ Set.Icc (-12 : ℝ) 0) ∧
  (∀ y ∈ Set.Icc (-12 : ℝ) 0, ∃ x ∈ Set.Icc (-4 : ℝ) 0, f a b x = y) :=
sorry

end NUMINAMATH_CALUDE_quadratic_function_theorem_l204_20442


namespace NUMINAMATH_CALUDE_perfect_squares_mod_seven_l204_20475

theorem perfect_squares_mod_seven :
  ∃! (S : Finset ℕ), (∀ n ∈ S, ∃ m : ℤ, (m ^ 2 : ℤ) % 7 = n) ∧
                     (∀ k : ℤ, ∃ n ∈ S, (k ^ 2 : ℤ) % 7 = n) ∧
                     S.card = 4 :=
by sorry

end NUMINAMATH_CALUDE_perfect_squares_mod_seven_l204_20475


namespace NUMINAMATH_CALUDE_line_translation_l204_20491

/-- A line in the xy-plane. -/
structure Line where
  slope : ℝ
  intercept : ℝ

/-- Vertical translation of a line. -/
def vertical_translate (l : Line) (d : ℝ) : Line :=
  { slope := l.slope, intercept := l.intercept - d }

theorem line_translation (l : Line) :
  l.slope = 3 ∧ l.intercept = 2 →
  vertical_translate l 3 = { slope := 3, intercept := -1 } := by
  sorry

end NUMINAMATH_CALUDE_line_translation_l204_20491


namespace NUMINAMATH_CALUDE_inequality_solution_condition_l204_20406

theorem inequality_solution_condition (a : ℝ) : 
  (∀ x : ℝ, (a + 1) * x > a + 1 ↔ x < 1) → a < -1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_condition_l204_20406


namespace NUMINAMATH_CALUDE_intercept_sum_zero_line_equation_l204_20441

/-- A line passing through a point with sum of intercepts equal to zero -/
structure InterceptSumZeroLine where
  /-- The slope of the line -/
  slope : ℝ
  /-- The y-intercept of the line -/
  y_intercept : ℝ
  /-- The line passes through the point (1,4) -/
  passes_through_point : slope + y_intercept = 4
  /-- The sum of x and y intercepts is zero -/
  sum_of_intercepts_zero : (- y_intercept / slope) + y_intercept = 0

/-- The equation of the line is either 4x-y=0 or x-y+3=0 -/
theorem intercept_sum_zero_line_equation (line : InterceptSumZeroLine) :
  (line.slope = 4 ∧ line.y_intercept = 0) ∨ (line.slope = 1 ∧ line.y_intercept = 3) :=
sorry

end NUMINAMATH_CALUDE_intercept_sum_zero_line_equation_l204_20441


namespace NUMINAMATH_CALUDE_grid_paths_path_count_l204_20481

theorem grid_paths (n m : ℕ) : (n + m).choose n = (n + m).choose m := by sorry

theorem path_count : Nat.choose 9 4 = 126 := by sorry

end NUMINAMATH_CALUDE_grid_paths_path_count_l204_20481


namespace NUMINAMATH_CALUDE_expression_evaluation_l204_20409

-- Define the expression
def expression : ℚ := -(2^3) + 6/5 * (2/5)

-- Theorem stating the equality
theorem expression_evaluation : expression = -7 - 13/25 := by sorry

end NUMINAMATH_CALUDE_expression_evaluation_l204_20409


namespace NUMINAMATH_CALUDE_arabella_dance_time_l204_20435

/-- The time Arabella spends learning three dance steps -/
def dance_learning_time (first_step : ℕ) : ℕ :=
  let second_step := first_step / 2
  let third_step := first_step + second_step
  first_step + second_step + third_step

/-- Proof that Arabella spends 90 minutes learning three dance steps -/
theorem arabella_dance_time :
  dance_learning_time 30 = 90 := by
  sorry

end NUMINAMATH_CALUDE_arabella_dance_time_l204_20435


namespace NUMINAMATH_CALUDE_ant_spider_minimum_distance_l204_20423

/-- The minimum distance between an ant and a spider under specific conditions -/
theorem ant_spider_minimum_distance :
  let ant_position (θ : ℝ) : ℝ × ℝ := (Real.cos θ, Real.sin θ)
  let spider_position (x : ℝ) : ℝ × ℝ := (2 * x - 1, 0)
  let distance (θ x : ℝ) : ℝ := Real.sqrt ((ant_position θ).1 - (spider_position x).1)^2 + ((ant_position θ).2 - (spider_position x).2)^2
  ∃ (θ : ℝ), ∀ (φ : ℝ), distance θ θ ≤ distance φ φ ∧ distance θ θ = Real.sqrt 14 / 4 :=
by sorry

end NUMINAMATH_CALUDE_ant_spider_minimum_distance_l204_20423


namespace NUMINAMATH_CALUDE_continuous_function_on_T_has_fixed_point_l204_20493

-- Define the set T
def T : Set (ℝ × ℝ) :=
  {p | ∃ (t : ℝ) (q : ℚ), t ∈ Set.Icc 0 1 ∧ p = (t * q, 1 - t)}

-- State the theorem
theorem continuous_function_on_T_has_fixed_point
  (f : T → T) (hf : Continuous f) :
  ∃ x : T, f x = x := by
  sorry

end NUMINAMATH_CALUDE_continuous_function_on_T_has_fixed_point_l204_20493


namespace NUMINAMATH_CALUDE_parallel_lines_c_value_l204_20413

/-- Two lines are parallel if and only if their slopes are equal -/
axiom parallel_lines_equal_slopes {m₁ m₂ b₁ b₂ : ℝ} :
  (∀ x y : ℝ, y = m₁ * x + b₁ ↔ y = m₂ * x + b₂) ↔ m₁ = m₂

/-- The value of c for which the lines y = 7x + 3 and y = (3c)x + 5 are parallel -/
theorem parallel_lines_c_value :
  (∀ x y : ℝ, y = 7 * x + 3 ↔ y = (3 * c) * x + 5) → c = 7 / 3 :=
by sorry

end NUMINAMATH_CALUDE_parallel_lines_c_value_l204_20413


namespace NUMINAMATH_CALUDE_sin_2alpha_minus_pi_6_l204_20457

theorem sin_2alpha_minus_pi_6 (α : Real) 
  (h : 2 * Real.sin α = 1 + 2 * Real.sqrt 3 * Real.cos α) : 
  Real.sin (2 * α - Real.pi / 6) = 7 / 8 := by
  sorry

end NUMINAMATH_CALUDE_sin_2alpha_minus_pi_6_l204_20457


namespace NUMINAMATH_CALUDE_right_triangle_height_properties_l204_20486

/-- Properties of a right-angled triangle with height to hypotenuse --/
theorem right_triangle_height_properties
  (a b c h p q : ℝ)
  (right_triangle : a^2 + b^2 = c^2)
  (height_divides_hypotenuse : p + q = c)
  (height_forms_similar_triangles : h^2 = a * b) :
  h^2 = p * q ∧ a^2 = p * c ∧ b^2 = q * c ∧ p / q = (a / b)^2 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_height_properties_l204_20486


namespace NUMINAMATH_CALUDE_sum_vertices_penta_hexa_prism_is_22_l204_20461

/-- The number of vertices in a polygon -/
def vertices_in_polygon (sides : ℕ) : ℕ := sides

/-- The number of vertices in a prism with polygonal bases -/
def vertices_in_prism (base_vertices : ℕ) : ℕ := 2 * base_vertices

/-- The sum of vertices in a pentagonal prism and a hexagonal prism -/
def sum_vertices_penta_hexa_prism : ℕ :=
  vertices_in_prism (vertices_in_polygon 5) + vertices_in_prism (vertices_in_polygon 6)

theorem sum_vertices_penta_hexa_prism_is_22 :
  sum_vertices_penta_hexa_prism = 22 := by
  sorry

end NUMINAMATH_CALUDE_sum_vertices_penta_hexa_prism_is_22_l204_20461


namespace NUMINAMATH_CALUDE_sandy_molly_age_difference_l204_20448

theorem sandy_molly_age_difference :
  ∀ (sandy_age molly_age : ℕ),
  sandy_age = 56 →
  sandy_age * 9 = molly_age * 7 →
  molly_age - sandy_age = 16 :=
by
  sorry

end NUMINAMATH_CALUDE_sandy_molly_age_difference_l204_20448


namespace NUMINAMATH_CALUDE_smallest_n_congruence_l204_20412

theorem smallest_n_congruence (n : ℕ+) : 
  (∀ m : ℕ+, m < n → ¬(5 * m : ℤ) ≡ 1978 [ZMOD 26]) ∧ 
  (5 * n : ℤ) ≡ 1978 [ZMOD 26] ↔ 
  n = 16 := by sorry

end NUMINAMATH_CALUDE_smallest_n_congruence_l204_20412


namespace NUMINAMATH_CALUDE_remainder_97_pow_25_mod_50_l204_20472

theorem remainder_97_pow_25_mod_50 : 97^25 % 50 = 7 := by
  sorry

end NUMINAMATH_CALUDE_remainder_97_pow_25_mod_50_l204_20472


namespace NUMINAMATH_CALUDE_chosen_number_l204_20483

theorem chosen_number (x : ℝ) : x / 5 - 154 = 6 → x = 800 := by
  sorry

end NUMINAMATH_CALUDE_chosen_number_l204_20483


namespace NUMINAMATH_CALUDE_interlaced_roots_l204_20470

/-- A quadratic function -/
def QuadraticFunction := ℝ → ℝ

/-- Predicate to check if two real numbers are distinct roots of a quadratic function -/
def are_distinct_roots (f : QuadraticFunction) (x y : ℝ) : Prop :=
  x ≠ y ∧ f x = 0 ∧ f y = 0

/-- Predicate to check if four real numbers are interlaced -/
def are_interlaced (x₁ x₂ x₃ x₄ : ℝ) : Prop :=
  (x₁ < x₃ ∧ x₃ < x₂ ∧ x₂ < x₄) ∨ (x₃ < x₁ ∧ x₁ < x₄ ∧ x₄ < x₂)

theorem interlaced_roots 
  (f g : QuadraticFunction) (x₁ x₂ x₃ x₄ : ℝ) 
  (h₁ : are_distinct_roots f x₁ x₂)
  (h₂ : are_distinct_roots g x₃ x₄)
  (h₃ : g x₁ * g x₂ < 0) :
  are_interlaced x₁ x₂ x₃ x₄ :=
sorry

end NUMINAMATH_CALUDE_interlaced_roots_l204_20470


namespace NUMINAMATH_CALUDE_marys_baking_problem_l204_20489

/-- Mary's baking problem -/
theorem marys_baking_problem (sugar_recipe : ℕ) (flour_recipe : ℕ) (flour_needed : ℕ) 
  (h1 : sugar_recipe = 3)
  (h2 : flour_recipe = 10)
  (h3 : flour_needed = sugar_recipe + 5) :
  flour_recipe - flour_needed = 5 :=
by
  sorry

#check marys_baking_problem

end NUMINAMATH_CALUDE_marys_baking_problem_l204_20489


namespace NUMINAMATH_CALUDE_total_money_sam_and_billy_l204_20492

/-- Given Sam has $75 and Billy has $25 less than twice the money Sam has, 
    their total money together is $200. -/
theorem total_money_sam_and_billy : 
  ∀ (sam_money billy_money : ℕ),
  sam_money = 75 →
  billy_money = 2 * sam_money - 25 →
  sam_money + billy_money = 200 := by
  sorry

end NUMINAMATH_CALUDE_total_money_sam_and_billy_l204_20492


namespace NUMINAMATH_CALUDE_complex_product_polar_form_l204_20411

-- Define the cis function
noncomputable def cis (θ : ℝ) : ℂ := Complex.exp (θ * Complex.I)

-- Define the problem
theorem complex_product_polar_form :
  ∃ (r θ : ℝ), 
    r > 0 ∧ 
    0 ≤ θ ∧ 
    θ < 2 * Real.pi ∧
    (4 * cis (30 * Real.pi / 180)) * (-3 * cis (45 * Real.pi / 180)) = r * cis θ ∧
    r = 12 ∧
    θ = 255 * Real.pi / 180 := by
  sorry

end NUMINAMATH_CALUDE_complex_product_polar_form_l204_20411


namespace NUMINAMATH_CALUDE_circle_radius_from_area_l204_20426

theorem circle_radius_from_area (A : ℝ) (r : ℝ) (h : A = 100 * Real.pi) :
  A = Real.pi * r^2 → r = 10 := by
  sorry

end NUMINAMATH_CALUDE_circle_radius_from_area_l204_20426


namespace NUMINAMATH_CALUDE_dagger_example_l204_20494

-- Define the † operation
def dagger (m n p q : ℚ) : ℚ := m * p * (q / n)

-- Theorem statement
theorem dagger_example : dagger (7/12) (8/3) = 14 := by
  sorry

end NUMINAMATH_CALUDE_dagger_example_l204_20494


namespace NUMINAMATH_CALUDE_wrong_mark_calculation_l204_20487

theorem wrong_mark_calculation (n : ℕ) (correct_mark : ℝ) (average_increase : ℝ) : 
  n = 56 → 
  correct_mark = 45 → 
  average_increase = 1/2 → 
  ∃ x : ℝ, x - correct_mark = n * average_increase ∧ x = 73 := by
sorry

end NUMINAMATH_CALUDE_wrong_mark_calculation_l204_20487


namespace NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l204_20450

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_common_difference
  (a : ℕ → ℝ)
  (h1 : arithmetic_sequence a)
  (h2 : a 3 + a 6 = 11)
  (h3 : a 5 + a 8 = 39) :
  ∃ d : ℝ, d = 7 ∧ ∀ n : ℕ, a (n + 1) = a n + d :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l204_20450


namespace NUMINAMATH_CALUDE_unique_n_exists_l204_20463

theorem unique_n_exists : ∃! n : ℤ,
  0 ≤ n ∧ n < 17 ∧
  -150 ≡ n [ZMOD 17] ∧
  102 % n = 0 ∧
  n = 3 := by
  sorry

end NUMINAMATH_CALUDE_unique_n_exists_l204_20463


namespace NUMINAMATH_CALUDE_max_value_of_f_l204_20465

-- Define the function f(x)
def f (x : ℝ) : ℝ := 3 * x - 4 * x^3

-- State the theorem
theorem max_value_of_f :
  ∃ (c : ℝ), c ∈ Set.Icc 0 1 ∧ ∀ x, x ∈ Set.Icc 0 1 → f x ≤ f c ∧ f c = 1 :=
sorry

end NUMINAMATH_CALUDE_max_value_of_f_l204_20465


namespace NUMINAMATH_CALUDE_orange_purchase_ratio_l204_20422

/-- Proves the ratio of weekly orange purchases --/
theorem orange_purchase_ratio 
  (initial_purchase : ℕ) 
  (additional_purchase : ℕ) 
  (total_after_three_weeks : ℕ) 
  (h1 : initial_purchase = 10)
  (h2 : additional_purchase = 5)
  (h3 : total_after_three_weeks = 75) :
  (total_after_three_weeks - (initial_purchase + additional_purchase)) / 2 = 
  2 * (initial_purchase + additional_purchase) :=
by
  sorry

#check orange_purchase_ratio

end NUMINAMATH_CALUDE_orange_purchase_ratio_l204_20422


namespace NUMINAMATH_CALUDE_complex_magnitude_equation_l204_20466

theorem complex_magnitude_equation (t : ℝ) (h : t > 0) :
  Complex.abs (-6 + t * Complex.I) = 3 * Real.sqrt 10 → t = 3 * Real.sqrt 6 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_equation_l204_20466


namespace NUMINAMATH_CALUDE_parabola_directrix_l204_20454

/-- The directrix of the parabola y = -1/4 * x^2 is y = 1 -/
theorem parabola_directrix (x y : ℝ) : 
  y = -1/4 * x^2 → 
  (∀ (x₀ y₀ : ℝ), y₀ = -1/4 * x₀^2 → 
    (x₀ - x)^2 + (y₀ - y)^2 = (y₀ - 1)^2) → 
  y = 1 := by sorry

end NUMINAMATH_CALUDE_parabola_directrix_l204_20454


namespace NUMINAMATH_CALUDE_train_cross_platform_time_l204_20438

def train_length : ℝ := 300
def platform_length : ℝ := 300
def time_cross_pole : ℝ := 18

theorem train_cross_platform_time :
  let train_speed := train_length / time_cross_pole
  let total_distance := train_length + platform_length
  total_distance / train_speed = 36 := by
  sorry

end NUMINAMATH_CALUDE_train_cross_platform_time_l204_20438


namespace NUMINAMATH_CALUDE_divisor_sum_product_2016_l204_20471

def sum_odd_divisors (n : ℕ) : ℕ := sorry

def sum_even_divisors (n : ℕ) : ℕ := sorry

theorem divisor_sum_product_2016 (n : ℕ) :
  n % 2 = 0 →
  (sum_odd_divisors n) * (sum_even_divisors n) = 2016 ↔ n = 192 ∨ n = 88 := by
  sorry

end NUMINAMATH_CALUDE_divisor_sum_product_2016_l204_20471


namespace NUMINAMATH_CALUDE_shell_division_impossibility_l204_20416

theorem shell_division_impossibility : ¬ ∃ (n : ℕ), 
  (637 - n) % 3 = 0 ∧ (n + 1 : ℕ) = (637 - n) / 3 := by
  sorry

end NUMINAMATH_CALUDE_shell_division_impossibility_l204_20416


namespace NUMINAMATH_CALUDE_partner_p_investment_time_l204_20447

/-- Represents the investment and profit data for two partners -/
structure PartnershipData where
  investment_ratio_p : ℚ
  investment_ratio_q : ℚ
  profit_ratio_p : ℚ
  profit_ratio_q : ℚ
  investment_time_q : ℚ

/-- Calculates the investment time for partner p given the partnership data -/
def calculate_investment_time_p (data : PartnershipData) : ℚ :=
  (data.investment_ratio_q * data.profit_ratio_p * data.investment_time_q) /
  (data.investment_ratio_p * data.profit_ratio_q)

/-- Theorem stating that given the specific partnership data, partner p's investment time is 5 months -/
theorem partner_p_investment_time :
  let data : PartnershipData := {
    investment_ratio_p := 7,
    investment_ratio_q := 5,
    profit_ratio_p := 7,
    profit_ratio_q := 12,
    investment_time_q := 12
  }
  calculate_investment_time_p data = 5 := by sorry

end NUMINAMATH_CALUDE_partner_p_investment_time_l204_20447


namespace NUMINAMATH_CALUDE_price_to_relatives_is_correct_l204_20459

-- Define the given quantities
def total_peaches : ℕ := 15
def peaches_sold_to_friends : ℕ := 10
def peaches_sold_to_relatives : ℕ := 4
def peaches_kept : ℕ := 1
def price_per_peach_to_friends : ℚ := 2
def total_earnings : ℚ := 25

-- Define the function to calculate the price per peach sold to relatives
def price_per_peach_to_relatives : ℚ :=
  (total_earnings - price_per_peach_to_friends * peaches_sold_to_friends) / peaches_sold_to_relatives

-- Theorem statement
theorem price_to_relatives_is_correct : price_per_peach_to_relatives = 1.25 := by
  sorry

end NUMINAMATH_CALUDE_price_to_relatives_is_correct_l204_20459


namespace NUMINAMATH_CALUDE_equation_proof_l204_20401

theorem equation_proof : 289 + 2 * 17 * 5 + 25 = 484 := by
  sorry

end NUMINAMATH_CALUDE_equation_proof_l204_20401


namespace NUMINAMATH_CALUDE_distance_between_foci_l204_20430

-- Define the ellipse equation
def ellipse_equation (x y : ℝ) : Prop :=
  Real.sqrt ((x - 4)^2 + (y - 5)^2) + Real.sqrt ((x + 6)^2 + (y - 9)^2) = 25

-- Define the foci
def focus1 : ℝ × ℝ := (4, 5)
def focus2 : ℝ × ℝ := (-6, 9)

-- Theorem: The distance between the foci of the ellipse is 2√29
theorem distance_between_foci :
  let (x1, y1) := focus1
  let (x2, y2) := focus2
  Real.sqrt ((x1 - x2)^2 + (y1 - y2)^2) = 2 * Real.sqrt 29 :=
by sorry

end NUMINAMATH_CALUDE_distance_between_foci_l204_20430
