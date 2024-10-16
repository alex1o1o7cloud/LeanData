import Mathlib

namespace NUMINAMATH_CALUDE_sqrt_sum_inequality_l71_7104

theorem sqrt_sum_inequality (a b c x y z : ℝ) 
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) 
  (hx : 0 < x) (hy : 0 < y) (hz : 0 < z)
  (hxa : x ≤ a) (hyb : y ≤ b) (hzc : z ≤ c) : 
  Real.sqrt x + Real.sqrt y + Real.sqrt z < Real.sqrt a + Real.sqrt b + Real.sqrt c := by
  sorry

end NUMINAMATH_CALUDE_sqrt_sum_inequality_l71_7104


namespace NUMINAMATH_CALUDE_incenter_distance_in_right_triangle_l71_7139

/-- A right triangle with given side lengths -/
structure RightTriangle where
  pq : ℝ
  pr : ℝ
  qr : ℝ
  right_angle : pq^2 + pr^2 = qr^2

/-- The incenter of a triangle -/
def incenter (t : RightTriangle) : ℝ × ℝ := sorry

/-- The distance between two points -/
def distance (p1 p2 : ℝ × ℝ) : ℝ := sorry

theorem incenter_distance_in_right_triangle (t : RightTriangle) 
  (h1 : t.pq = 15) (h2 : t.pr = 20) (h3 : t.qr = 25) :
  distance (0, 0) (incenter t) = 5 := by sorry

end NUMINAMATH_CALUDE_incenter_distance_in_right_triangle_l71_7139


namespace NUMINAMATH_CALUDE_smallest_number_l71_7168

theorem smallest_number (a b c d : ℝ) 
  (ha : a = 1) 
  (hb : b = -3) 
  (hc : c = -Real.sqrt 2) 
  (hd : d = -Real.pi) : 
  d ≤ a ∧ d ≤ b ∧ d ≤ c := by
sorry

end NUMINAMATH_CALUDE_smallest_number_l71_7168


namespace NUMINAMATH_CALUDE_geometric_sequence_a5_l71_7182

def geometric_sequence (a : ℕ → ℝ) := ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a n * r

theorem geometric_sequence_a5 (a : ℕ → ℝ) :
  geometric_sequence a →
  a 3 = 6 →
  a 3 + a 5 + a 7 = 78 →
  a 5 = 18 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_a5_l71_7182


namespace NUMINAMATH_CALUDE_negation_equivalence_l71_7165

theorem negation_equivalence :
  (¬ ∀ x : ℝ, x^2 - 2*x + 4 ≤ 0) ↔ (∃ x : ℝ, x^2 - 2*x + 4 > 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_equivalence_l71_7165


namespace NUMINAMATH_CALUDE_chocolate_milk_amount_l71_7193

/-- Represents the ingredients for making chocolate milk -/
structure Ingredients where
  milk : ℕ
  chocolate_syrup : ℕ
  whipped_cream : ℕ

/-- Represents the recipe for one glass of chocolate milk -/
structure Recipe where
  milk : ℕ
  chocolate_syrup : ℕ
  whipped_cream : ℕ
  total : ℕ

/-- Calculates the number of full glasses that can be made with given ingredients and recipe -/
def fullGlasses (i : Ingredients) (r : Recipe) : ℕ :=
  min (i.milk / r.milk) (min (i.chocolate_syrup / r.chocolate_syrup) (i.whipped_cream / r.whipped_cream))

/-- Theorem: Charles will drink 96 ounces of chocolate milk -/
theorem chocolate_milk_amount (i : Ingredients) (r : Recipe) :
  i.milk = 130 ∧ i.chocolate_syrup = 60 ∧ i.whipped_cream = 25 ∧
  r.milk = 4 ∧ r.chocolate_syrup = 2 ∧ r.whipped_cream = 2 ∧ r.total = 8 →
  fullGlasses i r * r.total = 96 := by
  sorry

end NUMINAMATH_CALUDE_chocolate_milk_amount_l71_7193


namespace NUMINAMATH_CALUDE_min_value_reciprocal_sum_l71_7117

theorem min_value_reciprocal_sum (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 2 * x + y = 2) :
  (∀ x' y' : ℝ, x' > 0 → y' > 0 → 2 * x' + y' = 2 → 1 / x' + 1 / y' ≥ 1 / x + 1 / y) →
  1 / x + 1 / y = 3 / 2 + Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_min_value_reciprocal_sum_l71_7117


namespace NUMINAMATH_CALUDE_container_weight_sum_l71_7174

theorem container_weight_sum (x y z : ℝ) 
  (h1 : x + y = 162) 
  (h2 : y + z = 168) 
  (h3 : z + x = 174) : 
  x + y + z = 252 := by
sorry

end NUMINAMATH_CALUDE_container_weight_sum_l71_7174


namespace NUMINAMATH_CALUDE_cover_room_with_tiles_l71_7110

/-- The width of the room -/
def room_width : ℝ := 8

/-- The length of the room -/
def room_length : ℝ := 12

/-- The width of a tile -/
def tile_width : ℝ := 1.5

/-- The length of a tile -/
def tile_length : ℝ := 2

/-- The number of tiles needed to cover the room -/
def tiles_needed : ℕ := 32

theorem cover_room_with_tiles :
  (room_width * room_length) / (tile_width * tile_length) = tiles_needed := by
  sorry

end NUMINAMATH_CALUDE_cover_room_with_tiles_l71_7110


namespace NUMINAMATH_CALUDE_simple_interest_principal_l71_7158

/-- Simple interest calculation -/
theorem simple_interest_principal (rate : ℝ) (time : ℝ) (interest : ℝ) (principal : ℝ) :
  rate = 4.5 →
  time = 4 →
  interest = 144 →
  principal * rate * time / 100 = interest →
  principal = 800 :=
by
  sorry

end NUMINAMATH_CALUDE_simple_interest_principal_l71_7158


namespace NUMINAMATH_CALUDE_sum_of_odd_powers_zero_l71_7157

theorem sum_of_odd_powers_zero (a b : ℝ) (n : ℕ) (h1 : a + b = 0) (h2 : a ≠ 0) :
  a^(2*n+1) + b^(2*n+1) = 0 := by
sorry

end NUMINAMATH_CALUDE_sum_of_odd_powers_zero_l71_7157


namespace NUMINAMATH_CALUDE_quadrilaterals_equal_area_l71_7169

/-- Represents a quadrilateral on a geoboard -/
structure Quadrilateral where
  area : ℝ

/-- Quadrilateral I can be rearranged to form a 3x1 rectangle -/
def quadrilateral_I : Quadrilateral :=
  { area := 3 * 1 }

/-- Quadrilateral II can be rearranged to form two 1x1.5 rectangles -/
def quadrilateral_II : Quadrilateral :=
  { area := 2 * (1 * 1.5) }

/-- Theorem: Quadrilateral I and Quadrilateral II have the same area -/
theorem quadrilaterals_equal_area : quadrilateral_I.area = quadrilateral_II.area := by
  sorry

#check quadrilaterals_equal_area

end NUMINAMATH_CALUDE_quadrilaterals_equal_area_l71_7169


namespace NUMINAMATH_CALUDE_no_simple_condition_for_equality_l71_7118

/-- There is no simple general condition for when a + b + c² = (a+b)(a+c) for all real numbers a, b, and c. -/
theorem no_simple_condition_for_equality (a b c : ℝ) : 
  ¬ ∃ (simple_condition : Prop), simple_condition ↔ (a + b + c^2 = (a+b)*(a+c)) :=
sorry

end NUMINAMATH_CALUDE_no_simple_condition_for_equality_l71_7118


namespace NUMINAMATH_CALUDE_smallest_solution_floor_equation_l71_7164

theorem smallest_solution_floor_equation :
  ∃ (x : ℝ), x > 0 ∧ x = 89/9 ∧
  (∀ y : ℝ, y > 0 → ⌊y^2⌋ - y * ⌊y⌋ = 8 → x ≤ y) ∧
  ⌊x^2⌋ - x * ⌊x⌋ = 8 :=
sorry

end NUMINAMATH_CALUDE_smallest_solution_floor_equation_l71_7164


namespace NUMINAMATH_CALUDE_tree_spacing_l71_7114

theorem tree_spacing (yard_length : ℝ) (num_trees : ℕ) :
  yard_length = 1530 →
  num_trees = 37 →
  let num_gaps := num_trees - 1
  let tree_spacing := yard_length / num_gaps
  tree_spacing = 42.5 := by
  sorry

end NUMINAMATH_CALUDE_tree_spacing_l71_7114


namespace NUMINAMATH_CALUDE_initial_observations_l71_7175

theorem initial_observations (initial_average : ℝ) (new_observation : ℝ) (average_decrease : ℝ) :
  initial_average = 12 →
  new_observation = 5 →
  average_decrease = 1 →
  ∃ n : ℕ, 
    (n : ℝ) * initial_average + new_observation = (n + 1) * (initial_average - average_decrease) ∧
    n = 6 :=
by sorry

end NUMINAMATH_CALUDE_initial_observations_l71_7175


namespace NUMINAMATH_CALUDE_max_distinct_pairs_l71_7150

/-- Given a set of integers from 1 to 3000, we can choose at most 1199 pairs
    such that each pair sum is distinct and no greater than 3000. -/
theorem max_distinct_pairs : ∀ (k : ℕ) (a b : ℕ → ℕ),
  (∀ i, i < k → 1 ≤ a i ∧ a i < b i ∧ b i ≤ 3000) →
  (∀ i j, i < k → j < k → i ≠ j → a i ≠ a j ∧ a i ≠ b j ∧ b i ≠ b j) →
  (∀ i j, i < k → j < k → i ≠ j → a i + b i ≠ a j + b j) →
  (∀ i, i < k → a i + b i ≤ 3000) →
  k ≤ 1199 :=
by sorry

end NUMINAMATH_CALUDE_max_distinct_pairs_l71_7150


namespace NUMINAMATH_CALUDE_bakery_start_time_l71_7144

/-- Represents time in hours and minutes -/
structure Time where
  hours : ℕ
  minutes : ℕ
  valid : minutes < 60 := by sorry

/-- Represents the bakery schedule -/
structure BakerySchedule where
  openingTime : Time
  roomTempTime : ℕ
  shapeTime : ℕ
  proofTime : ℕ
  bakeTime : ℕ
  coolTime : ℕ

/-- Calculates the total preparation time in hours -/
def totalPrepTime (schedule : BakerySchedule) : ℕ :=
  schedule.roomTempTime + 
  (schedule.shapeTime + schedule.proofTime * 60 + schedule.bakeTime + schedule.coolTime) / 60

/-- Calculates the latest start time given the opening time and total prep time -/
def latestStartTime (openingTime : Time) (prepTime : ℕ) : Time :=
  { hours := (openingTime.hours - prepTime + 24) % 24,
    minutes := openingTime.minutes }

/-- Theorem statement for the bakery problem -/
theorem bakery_start_time (schedule : BakerySchedule) 
    (h1 : schedule.openingTime = { hours := 6, minutes := 0 })
    (h2 : schedule.roomTempTime = 1)
    (h3 : schedule.shapeTime = 15)
    (h4 : schedule.proofTime = 2)
    (h5 : schedule.bakeTime = 30)
    (h6 : schedule.coolTime = 15) :
    latestStartTime schedule.openingTime (totalPrepTime schedule) = { hours := 2, minutes := 0 } := by
  sorry

end NUMINAMATH_CALUDE_bakery_start_time_l71_7144


namespace NUMINAMATH_CALUDE_min_value_expression_l71_7133

theorem min_value_expression (x : ℝ) :
  ∃ (min : ℝ), min = -6480.25 ∧
  ∀ y : ℝ, (15 - y) * (8 - y) * (15 + y) * (8 + y) ≥ min :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_l71_7133


namespace NUMINAMATH_CALUDE_ratio_problem_l71_7173

theorem ratio_problem (a b : ℚ) (h : (12 * a - 5 * b) / (14 * a - 3 * b) = 4 / 7) :
  a / b = 23 / 28 := by
sorry

end NUMINAMATH_CALUDE_ratio_problem_l71_7173


namespace NUMINAMATH_CALUDE_intransitive_dice_exist_l71_7120

/-- Represents a die with 6 faces --/
def Die := Fin 6 → Nat

/-- The probability that one die shows a higher number than another --/
def winProbability (d1 d2 : Die) : ℚ :=
  (Finset.sum Finset.univ (λ i => Finset.sum Finset.univ (λ j => 
    if d1 i > d2 j then 1 else 0))) / 36

/-- Theorem stating the existence of three dice with the desired properties --/
theorem intransitive_dice_exist : 
  ∃ (A B C : Die),
    winProbability B A > 1/2 ∧ 
    winProbability C B > 1/2 ∧ 
    winProbability A C > 1/2 := by
  sorry

end NUMINAMATH_CALUDE_intransitive_dice_exist_l71_7120


namespace NUMINAMATH_CALUDE_loan_years_is_eight_l71_7155

/-- Given a loan scenario, calculate the number of years for the first part. -/
def calculate_years (total_sum interest_rate1 interest_rate2 second_part_sum second_part_years : ℚ) : ℚ :=
  let first_part_sum := total_sum - second_part_sum
  let second_part_interest := second_part_sum * interest_rate2 * second_part_years / 100
  second_part_interest * 100 / (first_part_sum * interest_rate1)

/-- Prove that the number of years for the first part of the loan is 8. -/
theorem loan_years_is_eight :
  let total_sum : ℚ := 2769
  let interest_rate1 : ℚ := 3
  let interest_rate2 : ℚ := 5
  let second_part_sum : ℚ := 1704
  let second_part_years : ℚ := 3
  calculate_years total_sum interest_rate1 interest_rate2 second_part_sum second_part_years = 8 := by
  sorry


end NUMINAMATH_CALUDE_loan_years_is_eight_l71_7155


namespace NUMINAMATH_CALUDE_intersection_range_l71_7107

def M : Set (ℝ × ℝ) := {p | p.2 = Real.sqrt (9 - p.1^2) ∧ p.2 ≠ 0}

def N (b : ℝ) : Set (ℝ × ℝ) := {p | p.2 = p.1 + b}

theorem intersection_range (b : ℝ) (h : (M ∩ N b).Nonempty) : b ∈ Set.Ioo (-3) (3 * Real.sqrt 2) := by
  sorry

end NUMINAMATH_CALUDE_intersection_range_l71_7107


namespace NUMINAMATH_CALUDE_geoffrey_initial_wallet_l71_7197

/-- The amount of money Geoffrey had initially in his wallet --/
def initial_wallet_amount : ℕ := 50

/-- The amount Geoffrey received from his grandmother --/
def grandmother_gift : ℕ := 20

/-- The amount Geoffrey received from his aunt --/
def aunt_gift : ℕ := 25

/-- The amount Geoffrey received from his uncle --/
def uncle_gift : ℕ := 30

/-- The cost of each game --/
def game_cost : ℕ := 35

/-- The number of games Geoffrey bought --/
def num_games : ℕ := 3

/-- The amount left after the purchase --/
def amount_left : ℕ := 20

theorem geoffrey_initial_wallet :
  initial_wallet_amount = 
    (amount_left + num_games * game_cost) - (grandmother_gift + aunt_gift + uncle_gift) :=
by sorry

end NUMINAMATH_CALUDE_geoffrey_initial_wallet_l71_7197


namespace NUMINAMATH_CALUDE_complement_union_complement_equals_intersection_l71_7177

theorem complement_union_complement_equals_intersection (P Q : Set α) :
  (Pᶜᶜ ∪ Qᶜ)ᶜ = P ∩ Q := by
  sorry

end NUMINAMATH_CALUDE_complement_union_complement_equals_intersection_l71_7177


namespace NUMINAMATH_CALUDE_three_planes_intersection_count_l71_7159

structure Plane

/-- Three planes that intersect pairwise -/
structure ThreePlanesIntersectingPairwise where
  plane1 : Plane
  plane2 : Plane
  plane3 : Plane
  intersect12 : plane1 ≠ plane2
  intersect23 : plane2 ≠ plane3
  intersect13 : plane1 ≠ plane3

/-- A line of intersection between two planes -/
def LineOfIntersection (p1 p2 : Plane) : Type := Unit

/-- Count the number of distinct lines of intersection -/
def CountLinesOfIntersection (t : ThreePlanesIntersectingPairwise) : Nat :=
  sorry

theorem three_planes_intersection_count
  (t : ThreePlanesIntersectingPairwise) :
  CountLinesOfIntersection t = 1 ∨ CountLinesOfIntersection t = 3 :=
sorry

end NUMINAMATH_CALUDE_three_planes_intersection_count_l71_7159


namespace NUMINAMATH_CALUDE_candy_bar_cost_l71_7126

theorem candy_bar_cost (num_members : ℕ) (avg_sold_per_member : ℕ) (total_earnings : ℚ) :
  num_members = 20 →
  avg_sold_per_member = 8 →
  total_earnings = 80 →
  (total_earnings / (num_members * avg_sold_per_member : ℚ)) = 0.5 := by
  sorry

end NUMINAMATH_CALUDE_candy_bar_cost_l71_7126


namespace NUMINAMATH_CALUDE_odd_periodic_sum_zero_l71_7116

def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

def is_periodic (f : ℝ → ℝ) (p : ℝ) : Prop := ∀ x, f (x + p) = f x

theorem odd_periodic_sum_zero (f : ℝ → ℝ) (h_odd : is_odd f) (h_periodic : is_periodic f 2) :
  f 1 + f 4 + f 7 = 0 := by
  sorry

end NUMINAMATH_CALUDE_odd_periodic_sum_zero_l71_7116


namespace NUMINAMATH_CALUDE_cone_volume_from_cylinder_volume_l71_7103

/-- Given a cylinder with volume 72π cm³ and height twice its radius,
    prove that a cone with the same radius and height has a volume of 144π cm³. -/
theorem cone_volume_from_cylinder_volume (r h : ℝ) : 
  (π * r^2 * h = 72 * π) → 
  (h = 2 * r) → 
  ((1/3) * π * r^2 * h = 144 * π) := by
  sorry

end NUMINAMATH_CALUDE_cone_volume_from_cylinder_volume_l71_7103


namespace NUMINAMATH_CALUDE_project_completion_time_l71_7185

/-- The number of days A takes to complete the project alone -/
def A_days : ℝ := 20

/-- The number of days it takes A and B together to complete the project -/
def total_days : ℝ := 15

/-- The number of days before completion that A quits -/
def A_quit_days : ℝ := 5

/-- The number of days B takes to complete the project alone -/
def B_days : ℝ := 30

/-- Theorem stating that given the conditions, B can complete the project alone in 30 days -/
theorem project_completion_time :
  A_days = 20 ∧ total_days = 15 ∧ A_quit_days = 5 →
  (total_days - A_quit_days) * (1 / A_days + 1 / B_days) + A_quit_days * (1 / B_days) = 1 :=
by sorry

end NUMINAMATH_CALUDE_project_completion_time_l71_7185


namespace NUMINAMATH_CALUDE_point_2023_coordinates_l71_7127

/-- The y-coordinate of the nth point in the sequence -/
def y_coord (n : ℕ) : ℤ :=
  match n % 4 with
  | 0 => 0
  | 1 => 1
  | 2 => 0
  | 3 => -1
  | _ => 0  -- This case is technically unreachable, but Lean requires it

/-- The sequence of points as described in the problem -/
def point_sequence (n : ℕ) : ℕ × ℤ :=
  (n, y_coord (n + 1))

theorem point_2023_coordinates :
  point_sequence 2022 = (2022, 0) := by sorry

end NUMINAMATH_CALUDE_point_2023_coordinates_l71_7127


namespace NUMINAMATH_CALUDE_cos_derivative_at_pi_sixth_l71_7170

theorem cos_derivative_at_pi_sixth (f : ℝ → ℝ) :
  (∀ x, f x = Real.cos x) → HasDerivAt f (-1/2) (π/6) := by
  sorry

end NUMINAMATH_CALUDE_cos_derivative_at_pi_sixth_l71_7170


namespace NUMINAMATH_CALUDE_frederick_tyson_age_ratio_l71_7186

/-- Represents the ages and relationships between Kyle, Julian, Frederick, and Tyson -/
structure AgeRelationships where
  kyle_age : ℕ
  tyson_age : ℕ
  kyle_julian_diff : ℕ
  julian_frederick_diff : ℕ
  kyle_age_is_25 : kyle_age = 25
  tyson_age_is_20 : tyson_age = 20
  kyle_older_than_julian : kyle_age = kyle_julian_diff + (kyle_age - kyle_julian_diff)
  julian_younger_than_frederick : kyle_age - kyle_julian_diff = (kyle_age - kyle_julian_diff + julian_frederick_diff) - julian_frederick_diff

/-- The ratio of Frederick's age to Tyson's age is 2:1 -/
theorem frederick_tyson_age_ratio (ar : AgeRelationships) :
  (ar.kyle_age - ar.kyle_julian_diff + ar.julian_frederick_diff) / ar.tyson_age = 2 := by
  sorry

end NUMINAMATH_CALUDE_frederick_tyson_age_ratio_l71_7186


namespace NUMINAMATH_CALUDE_relay_team_arrangements_l71_7149

/-- The number of ways to arrange 4 people in a line with one fixed in the second position -/
def fixed_second_arrangements : ℕ := 6

/-- The total number of team members -/
def team_size : ℕ := 4

/-- The position where Jordan is fixed -/
def jordans_position : ℕ := 2

theorem relay_team_arrangements :
  (team_size = 4) →
  (jordans_position = 2) →
  (fixed_second_arrangements = 6) := by
sorry

end NUMINAMATH_CALUDE_relay_team_arrangements_l71_7149


namespace NUMINAMATH_CALUDE_uncool_parents_count_l71_7111

theorem uncool_parents_count (total : ℕ) (cool_dads : ℕ) (cool_moms : ℕ) (cool_both : ℕ) : 
  total = 30 → cool_dads = 12 → cool_moms = 15 → cool_both = 9 →
  total - (cool_dads + cool_moms - cool_both) = 12 :=
by sorry

end NUMINAMATH_CALUDE_uncool_parents_count_l71_7111


namespace NUMINAMATH_CALUDE_final_salary_matches_expected_l71_7142

/-- Calculates the final take-home salary after a raise, pay cut, and tax --/
def finalSalary (initialSalary : ℝ) (raisePercent : ℝ) (cutPercent : ℝ) (taxPercent : ℝ) : ℝ :=
  let salaryAfterRaise := initialSalary * (1 + raisePercent)
  let salaryAfterCut := salaryAfterRaise * (1 - cutPercent)
  salaryAfterCut * (1 - taxPercent)

/-- Theorem stating that the final salary matches the expected value --/
theorem final_salary_matches_expected :
  finalSalary 2500 0.25 0.15 0.10 = 2390.63 := by
  sorry

#eval finalSalary 2500 0.25 0.15 0.10

end NUMINAMATH_CALUDE_final_salary_matches_expected_l71_7142


namespace NUMINAMATH_CALUDE_middle_box_statement_l71_7160

/-- Represents the two possible statements on a box. -/
inductive BoxStatement
  | NoPrizeHere
  | PrizeInNeighbor

/-- Represents a configuration of boxes with their statements. -/
def BoxConfiguration := Fin 23 → BoxStatement

/-- Checks if the given configuration is valid according to the problem rules. -/
def isValidConfiguration (config : BoxConfiguration) (prizeBox : Fin 23) : Prop :=
  -- Exactly one statement is true
  (∃! i, (config i = BoxStatement.NoPrizeHere ∧ i = prizeBox) ∨
         (config i = BoxStatement.PrizeInNeighbor ∧ (i + 1 = prizeBox ∨ i - 1 = prizeBox))) ∧
  -- The prize box exists
  (∃ i, i = prizeBox)

/-- The middle box index (0-based). -/
def middleBoxIndex : Fin 23 := ⟨11, by norm_num⟩

/-- The main theorem stating that the middle box must be labeled "The prize is in the neighboring box." -/
theorem middle_box_statement (config : BoxConfiguration) (prizeBox : Fin 23) 
    (h : isValidConfiguration config prizeBox) :
    config middleBoxIndex = BoxStatement.PrizeInNeighbor := by
  sorry


end NUMINAMATH_CALUDE_middle_box_statement_l71_7160


namespace NUMINAMATH_CALUDE_choir_arrangements_choir_arrangement_count_l71_7179

theorem choir_arrangements (total_boys : Nat) (total_girls : Nat) 
  (selected_boys : Nat) (selected_girls : Nat) : Nat :=
  let boy_selections := Nat.choose total_boys selected_boys
  let girl_selections := Nat.choose total_girls selected_girls
  let boy_arrangements := Nat.factorial selected_boys
  let girl_positions := Nat.factorial (selected_boys + 1) / Nat.factorial (selected_boys + 1 - selected_girls)
  boy_selections * girl_selections * boy_arrangements * girl_positions

theorem choir_arrangement_count : 
  choir_arrangements 4 3 2 2 = 216 := by sorry

end NUMINAMATH_CALUDE_choir_arrangements_choir_arrangement_count_l71_7179


namespace NUMINAMATH_CALUDE_track_length_l71_7187

/-- The length of a circular track given race conditions -/
theorem track_length (s t a : ℝ) (h₁ : s > 0) (h₂ : t > 0) (h₃ : a > 0) : 
  ∃ x : ℝ, x > 0 ∧ x = (s / (120 * t)) * (Real.sqrt (a^2 + 240 * a * t) - a) :=
by sorry

end NUMINAMATH_CALUDE_track_length_l71_7187


namespace NUMINAMATH_CALUDE_square_of_sum_l71_7130

theorem square_of_sum (a b : ℝ) : a^2 + b^2 + 2*a*b = (a + b)^2 := by
  sorry

end NUMINAMATH_CALUDE_square_of_sum_l71_7130


namespace NUMINAMATH_CALUDE_jennifers_money_l71_7112

theorem jennifers_money (initial_amount : ℚ) : 
  initial_amount > 0 →
  initial_amount - (initial_amount / 5 + initial_amount / 6 + initial_amount / 2) = 12 →
  initial_amount = 90 := by
sorry

end NUMINAMATH_CALUDE_jennifers_money_l71_7112


namespace NUMINAMATH_CALUDE_living_room_count_l71_7134

/-- The number of people in a house. -/
def total_people : ℕ := 15

/-- The number of people in the bedroom. -/
def bedroom_people : ℕ := 7

/-- The number of people in the living room. -/
def living_room_people : ℕ := total_people - bedroom_people

/-- Theorem stating that the number of people in the living room is 8. -/
theorem living_room_count : living_room_people = 8 := by
  sorry

end NUMINAMATH_CALUDE_living_room_count_l71_7134


namespace NUMINAMATH_CALUDE_curve_is_hyperbola_iff_l71_7188

/-- A curve in the xy-plane parameterized by k -/
def curve (k : ℝ) : Set (ℝ × ℝ) :=
  {(x, y) | x^2 / (4 + k) + y^2 / (1 - k) = 1}

/-- The condition for the curve to be a hyperbola -/
def is_hyperbola (k : ℝ) : Prop :=
  (4 + k) * (1 - k) < 0

/-- The range of k for which the curve is a hyperbola -/
def hyperbola_range : Set ℝ :=
  {k | k < -4 ∨ k > 1}

/-- Theorem stating that the curve is a hyperbola if and only if k is in the hyperbola_range -/
theorem curve_is_hyperbola_iff (k : ℝ) :
  is_hyperbola k ↔ k ∈ hyperbola_range := by sorry

end NUMINAMATH_CALUDE_curve_is_hyperbola_iff_l71_7188


namespace NUMINAMATH_CALUDE_g_neg_one_value_l71_7132

-- Define f as a function from ℝ to ℝ
variable (f : ℝ → ℝ)

-- Define the property of f being an odd function when combined with x^2
def isOddWithSquare (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) + (-x)^2 = -(f x + x^2)

-- Define g in terms of f
def g (f : ℝ → ℝ) (x : ℝ) : ℝ := f x + 2

-- State the theorem
theorem g_neg_one_value (f : ℝ → ℝ) 
  (h1 : isOddWithSquare f) 
  (h2 : f 1 = 1) : 
  g f (-1) = -1 := by
  sorry

end NUMINAMATH_CALUDE_g_neg_one_value_l71_7132


namespace NUMINAMATH_CALUDE_magazine_cost_l71_7156

theorem magazine_cost (m : ℝ) 
  (h1 : 8 * m < 12) 
  (h2 : 11 * m > 16.5) : 
  m = 1.5 := by
sorry

end NUMINAMATH_CALUDE_magazine_cost_l71_7156


namespace NUMINAMATH_CALUDE_geometric_progression_squared_sum_l71_7119

theorem geometric_progression_squared_sum 
  (q : ℝ) 
  (S : ℝ) 
  (h1 : abs q < 1) 
  (h2 : S = 1 / (1 - q)) : 
  1 / (1 - q^2) = S^2 / (2*S - 1) := by
  sorry

end NUMINAMATH_CALUDE_geometric_progression_squared_sum_l71_7119


namespace NUMINAMATH_CALUDE_triathlon_speed_l71_7198

/-- Triathlon problem -/
theorem triathlon_speed (total_time : ℝ) (swim_dist swim_speed : ℝ) (run_dist run_speed : ℝ) (bike_dist : ℝ) :
  total_time = 2 →
  swim_dist = 0.5 →
  swim_speed = 3 →
  run_dist = 4 →
  run_speed = 8 →
  bike_dist = 20 →
  (swim_dist / swim_speed + run_dist / run_speed + bike_dist / (bike_dist / (total_time - (swim_dist / swim_speed + run_dist / run_speed))) = total_time) →
  bike_dist / (total_time - (swim_dist / swim_speed + run_dist / run_speed)) = 15 := by
sorry


end NUMINAMATH_CALUDE_triathlon_speed_l71_7198


namespace NUMINAMATH_CALUDE_original_group_size_l71_7195

theorem original_group_size (initial_days : ℕ) (absent_men : ℕ) (final_days : ℕ) :
  initial_days = 8 ∧ absent_men = 3 ∧ final_days = 10 →
  ∃ original_size : ℕ, 
    original_size > absent_men ∧
    (original_size : ℚ) / initial_days = (original_size - absent_men) / final_days ∧
    original_size = 15 := by
  sorry

end NUMINAMATH_CALUDE_original_group_size_l71_7195


namespace NUMINAMATH_CALUDE_cubic_root_sum_cubes_l71_7109

theorem cubic_root_sum_cubes (a b c : ℝ) : 
  (a^3 - 3*a^2 + 4*a - 5 = 0) → 
  (b^3 - 3*b^2 + 4*b - 5 = 0) → 
  (c^3 - 3*c^2 + 4*c - 5 = 0) → 
  a^3 + b^3 + c^3 = 6 := by
  sorry

end NUMINAMATH_CALUDE_cubic_root_sum_cubes_l71_7109


namespace NUMINAMATH_CALUDE_fair_haired_women_percentage_l71_7125

/-- Given that 32% of employees are women with fair hair and 80% of employees have fair hair,
    prove that 40% of fair-haired employees are women. -/
theorem fair_haired_women_percentage 
  (total_employees : ℝ) 
  (h1 : total_employees > 0)
  (women_fair_hair : ℝ) 
  (h2 : women_fair_hair = 0.32 * total_employees)
  (fair_haired : ℝ) 
  (h3 : fair_haired = 0.80 * total_employees) :
  women_fair_hair / fair_haired = 0.40 := by
sorry

end NUMINAMATH_CALUDE_fair_haired_women_percentage_l71_7125


namespace NUMINAMATH_CALUDE_sum_of_squares_l71_7190

theorem sum_of_squares (x y : ℝ) (h1 : x + y = 10) (h2 : x^3 + y^3 = 400) : x^2 + y^2 = 60 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_l71_7190


namespace NUMINAMATH_CALUDE_distance_to_reflection_distance_z_to_z_reflected_l71_7180

/-- The distance between a point and its reflection over the x-axis --/
theorem distance_to_reflection (x y : ℝ) : 
  let z : ℝ × ℝ := (x, y)
  let z_reflected : ℝ × ℝ := (x, -y)
  Real.sqrt ((z.1 - z_reflected.1)^2 + (z.2 - z_reflected.2)^2) = 2 * |y| :=
by sorry

/-- The specific case for the point Z(5, 2) --/
theorem distance_z_to_z_reflected : 
  let z : ℝ × ℝ := (5, 2)
  let z_reflected : ℝ × ℝ := (5, -2)
  Real.sqrt ((z.1 - z_reflected.1)^2 + (z.2 - z_reflected.2)^2) = 4 :=
by sorry

end NUMINAMATH_CALUDE_distance_to_reflection_distance_z_to_z_reflected_l71_7180


namespace NUMINAMATH_CALUDE_ampersand_composition_l71_7194

-- Define the operations
def ampersand (y : ℤ) : ℤ := 2 * (7 - y)
def ampersandbar (y : ℤ) : ℤ := 2 * (y - 7)

-- State the theorem
theorem ampersand_composition : ampersandbar (ampersand (-13)) = 66 := by
  sorry

end NUMINAMATH_CALUDE_ampersand_composition_l71_7194


namespace NUMINAMATH_CALUDE_meetings_percentage_is_24_l71_7124

/-- Represents the duration of a work day in minutes -/
def work_day_minutes : ℕ := 10 * 60

/-- Represents the duration of a break in minutes -/
def break_minutes : ℕ := 30

/-- Represents the duration of the first meeting in minutes -/
def first_meeting_minutes : ℕ := 60

/-- Represents the duration of the second meeting in minutes -/
def second_meeting_minutes : ℕ := 75

/-- Calculates the effective work minutes (excluding break) -/
def effective_work_minutes : ℕ := work_day_minutes - break_minutes

/-- Calculates the total meeting minutes -/
def total_meeting_minutes : ℕ := first_meeting_minutes + second_meeting_minutes

/-- Theorem stating that the percentage of effective work day spent in meetings is 24% -/
theorem meetings_percentage_is_24 : 
  (total_meeting_minutes : ℚ) / (effective_work_minutes : ℚ) * 100 = 24 := by
  sorry

end NUMINAMATH_CALUDE_meetings_percentage_is_24_l71_7124


namespace NUMINAMATH_CALUDE_total_amount_is_468_l71_7108

/-- Calculates the total amount paid including service charge -/
def totalAmountPaid (originalAmount : ℝ) (serviceChargeRate : ℝ) : ℝ :=
  originalAmount * (1 + serviceChargeRate)

/-- Proves that the total amount paid is 468 given the conditions -/
theorem total_amount_is_468 :
  let originalAmount : ℝ := 450
  let serviceChargeRate : ℝ := 0.04
  totalAmountPaid originalAmount serviceChargeRate = 468 := by
  sorry

end NUMINAMATH_CALUDE_total_amount_is_468_l71_7108


namespace NUMINAMATH_CALUDE_sum_of_squares_l71_7105

theorem sum_of_squares (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0)
  (sum_zero : a + b + c = 0) (cube_eq_seventh : a^3 + b^3 + c^3 = a^7 + b^7 + c^7) :
  a^2 + b^2 + c^2 = -6/7 := by sorry

end NUMINAMATH_CALUDE_sum_of_squares_l71_7105


namespace NUMINAMATH_CALUDE_sum_squares_lengths_eq_k_squared_l71_7147

/-- A regular k-gon inscribed in a unit circle -/
structure RegularKGon (k : ℕ) where
  (k_pos : k > 0)

/-- The sum of squares of lengths of all sides and diagonals of a regular k-gon -/
def sum_squares_lengths (k : ℕ) (P : RegularKGon k) : ℝ :=
  sorry

/-- Theorem: The sum of squares of lengths of all sides and diagonals of a regular k-gon
    inscribed in a unit circle is equal to k^2 -/
theorem sum_squares_lengths_eq_k_squared (k : ℕ) (P : RegularKGon k) :
  sum_squares_lengths k P = k^2 :=
sorry

end NUMINAMATH_CALUDE_sum_squares_lengths_eq_k_squared_l71_7147


namespace NUMINAMATH_CALUDE_janes_flower_bed_area_l71_7123

/-- A rectangular flower bed with fence posts -/
structure FlowerBed where
  total_posts : ℕ
  post_spacing : ℝ
  long_side_post_ratio : ℕ

/-- Calculate the area of a flower bed given its specifications -/
def flowerBedArea (fb : FlowerBed) : ℝ :=
  let short_side_posts := (fb.total_posts + 4) / (2 * (fb.long_side_post_ratio + 1))
  let long_side_posts := short_side_posts * fb.long_side_post_ratio
  let short_side_length := (short_side_posts - 1) * fb.post_spacing
  let long_side_length := (long_side_posts - 1) * fb.post_spacing
  short_side_length * long_side_length

/-- Theorem: The area of Jane's flower bed is 144 square feet -/
theorem janes_flower_bed_area :
  let fb : FlowerBed := {
    total_posts := 24,
    post_spacing := 3,
    long_side_post_ratio := 3
  }
  flowerBedArea fb = 144 := by sorry

end NUMINAMATH_CALUDE_janes_flower_bed_area_l71_7123


namespace NUMINAMATH_CALUDE_batsman_score_difference_l71_7135

/-- Given a batsman's statistics, prove the difference between highest and lowest scores -/
theorem batsman_score_difference
  (total_innings : ℕ)
  (total_runs : ℕ)
  (excluded_innings : ℕ)
  (excluded_runs : ℕ)
  (highest_score : ℕ)
  (h_total_innings : total_innings = 46)
  (h_excluded_innings : excluded_innings = 44)
  (h_total_runs : total_runs = 60 * total_innings)
  (h_excluded_runs : excluded_runs = 58 * excluded_innings)
  (h_highest_score : highest_score = 174) :
  highest_score - (total_runs - excluded_runs - highest_score) = 140 :=
by sorry

end NUMINAMATH_CALUDE_batsman_score_difference_l71_7135


namespace NUMINAMATH_CALUDE_digit_symmetrical_equation_l71_7140

theorem digit_symmetrical_equation (a b : ℤ) (h : 2 ≤ a + b ∧ a + b ≤ 9) :
  (10*a + b) * (100*b + 10*(a + b) + a) = (100*a + 10*(a + b) + b) * (10*b + a) := by
  sorry

end NUMINAMATH_CALUDE_digit_symmetrical_equation_l71_7140


namespace NUMINAMATH_CALUDE_yu_chan_walking_distance_l71_7100

def step_length : ℝ := 0.75
def walking_time : ℝ := 13
def steps_per_minute : ℝ := 70

theorem yu_chan_walking_distance : 
  step_length * walking_time * steps_per_minute = 682.5 := by
  sorry

end NUMINAMATH_CALUDE_yu_chan_walking_distance_l71_7100


namespace NUMINAMATH_CALUDE_system_of_equations_solution_l71_7199

theorem system_of_equations_solution (a₁ a₂ b₁ b₂ c₁ c₂ : ℝ) :
  (∃ (x y : ℝ), a₁ * x - b₁ * y = c₁ ∧ a₂ * x + b₂ * y = c₂ ∧ x = 2 ∧ y = -1) →
  (∃ (x y : ℝ), a₁ * (x + 3) - b₁ * (y - 2) = c₁ ∧ a₂ * (x + 3) + b₂ * (y - 2) = c₂ ∧ x = -1 ∧ y = 1) :=
by sorry

end NUMINAMATH_CALUDE_system_of_equations_solution_l71_7199


namespace NUMINAMATH_CALUDE_officer_jawan_groups_count_l71_7171

/-- The number of combinations of n items taken k at a time -/
def binomial (n k : ℕ) : ℕ := sorry

/-- The number of ways to choose 2 officers from 7 and 4 jawans from 12 -/
def officer_jawan_groups : ℕ :=
  binomial 7 2 * binomial 12 4

theorem officer_jawan_groups_count :
  officer_jawan_groups = 20790 := by sorry

end NUMINAMATH_CALUDE_officer_jawan_groups_count_l71_7171


namespace NUMINAMATH_CALUDE_original_number_is_six_l71_7113

/-- Represents a person in the circle with their chosen number and announced average -/
structure Person where
  chosen : ℝ
  announced : ℝ

/-- The circle of 12 people -/
def Circle := Fin 12 → Person

theorem original_number_is_six
  (circle : Circle)
  (h_average : ∀ i : Fin 12, (circle i).announced = ((circle (i - 1)).chosen + (circle (i + 1)).chosen) / 2)
  (h_person : ∃ i : Fin 12, (circle i).announced = 8 ∧
    (circle (i - 1)).announced = 5 ∧ (circle (i + 1)).announced = 11) :
  ∃ i : Fin 12, (circle i).announced = 8 ∧ (circle i).chosen = 6 := by
  sorry

end NUMINAMATH_CALUDE_original_number_is_six_l71_7113


namespace NUMINAMATH_CALUDE_fixed_point_on_graph_l71_7128

theorem fixed_point_on_graph (m : ℝ) : 
  let f : ℝ → ℝ := fun x ↦ 9 * x^2 + m * x - 5 * m
  f 5 = 225 := by
  sorry

end NUMINAMATH_CALUDE_fixed_point_on_graph_l71_7128


namespace NUMINAMATH_CALUDE_dina_dolls_count_l71_7153

/-- The number of dolls Ivy has -/
def ivy_dolls : ℕ := 30

/-- The number of collector's edition dolls Ivy has -/
def ivy_collector_dolls : ℕ := 20

/-- The ratio of collector's edition dolls to total dolls for Ivy -/
def collector_ratio : ℚ := 2/3

/-- The number of dolls Dina has -/
def dina_dolls : ℕ := 2 * ivy_dolls

theorem dina_dolls_count : dina_dolls = 60 := by
  sorry

end NUMINAMATH_CALUDE_dina_dolls_count_l71_7153


namespace NUMINAMATH_CALUDE_problem_solution_l71_7192

theorem problem_solution (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : 2*a + b = 1) :
  (∀ a b, a*b ≥ 1/8) ∧
  (∀ a b, 1/a + 2/b ≥ 8) ∧
  (∀ a b, Real.sqrt (2*a) + Real.sqrt b ≤ Real.sqrt 2) ∧
  (∀ a b, (a+1)*(b+1) < 2) :=
by sorry

end NUMINAMATH_CALUDE_problem_solution_l71_7192


namespace NUMINAMATH_CALUDE_fraction_equality_l71_7163

theorem fraction_equality (a b c d : ℚ) 
  (h1 : a / b = 20)
  (h2 : c / b = 5)
  (h3 : c / d = 1 / 15) :
  a / d = 4 / 15 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l71_7163


namespace NUMINAMATH_CALUDE_palm_meadows_rooms_l71_7181

theorem palm_meadows_rooms (two_bed_rooms three_bed_rooms : ℕ) : 
  two_bed_rooms = 8 →
  two_bed_rooms * 2 + three_bed_rooms * 3 = 31 →
  two_bed_rooms + three_bed_rooms = 13 := by
sorry

end NUMINAMATH_CALUDE_palm_meadows_rooms_l71_7181


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l71_7137

def A : Set ℝ := {-2, -1, 0, 1}
def B : Set ℝ := {x : ℝ | x^2 - 1 ≤ 0}

theorem intersection_of_A_and_B : A ∩ B = {-1, 0, 1} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l71_7137


namespace NUMINAMATH_CALUDE_equation_solutions_l71_7196

theorem equation_solutions : 
  {(x, y) : ℕ × ℕ | x^2 + 6*x*y - 7*y^2 = 2009 ∧ x > 0 ∧ y > 0} = 
  {(252, 251), (42, 35), (42, 1)} := by sorry

end NUMINAMATH_CALUDE_equation_solutions_l71_7196


namespace NUMINAMATH_CALUDE_right_triangles_2012_characterization_l71_7136

/-- A right triangle with natural number side lengths where one leg is 2012 -/
structure RightTriangle2012 where
  other_leg : ℕ
  hypotenuse : ℕ
  is_right_triangle : other_leg ^ 2 + 2012 ^ 2 = hypotenuse ^ 2

/-- The set of all valid RightTriangle2012 -/
def all_right_triangles_2012 : Set RightTriangle2012 :=
  { t | t.other_leg > 0 ∧ t.hypotenuse > 0 }

/-- The four specific triangles mentioned in the problem -/
def specific_triangles : Set RightTriangle2012 :=
  { ⟨253005, 253013, by sorry⟩,
    ⟨506016, 506020, by sorry⟩,
    ⟨1012035, 1012037, by sorry⟩,
    ⟨1509, 2515, by sorry⟩ }

/-- The main theorem stating that the set of all valid right triangles with one leg 2012
    is equal to the set of four specific triangles -/
theorem right_triangles_2012_characterization :
  all_right_triangles_2012 = specific_triangles :=
sorry

end NUMINAMATH_CALUDE_right_triangles_2012_characterization_l71_7136


namespace NUMINAMATH_CALUDE_least_xy_value_l71_7161

theorem least_xy_value (x y : ℕ+) (h : (1 : ℚ) / x + (1 : ℚ) / (3 * y) = (1 : ℚ) / 6) :
  (∀ a b : ℕ+, (1 : ℚ) / a + (1 : ℚ) / (3 * b) = (1 : ℚ) / 6 → x * y ≤ a * b) ∧ x * y = 64 := by
  sorry

end NUMINAMATH_CALUDE_least_xy_value_l71_7161


namespace NUMINAMATH_CALUDE_edward_lives_left_l71_7183

theorem edward_lives_left (initial_lives : ℕ) (lives_lost : ℕ) : 
  initial_lives = 15 → lives_lost = 8 → initial_lives - lives_lost = 7 := by
  sorry

end NUMINAMATH_CALUDE_edward_lives_left_l71_7183


namespace NUMINAMATH_CALUDE_fraction_simplification_l71_7138

theorem fraction_simplification 
  (a b c k : ℝ) 
  (h : a * b = c * k) 
  (h_nonzero : c * k ≠ 0) : 
  (a - b - c + k) / (a + b + c + k) = (a - c) / (a + c) := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l71_7138


namespace NUMINAMATH_CALUDE_cost_minimized_at_35_l71_7154

/-- Represents the cost function for ordering hand sanitizers -/
def cost_function (x : ℝ) : ℝ := -2 * x^2 + 102 * x + 5000

/-- Represents the constraint on the number of boxes of type A sanitizer -/
def constraint (x : ℝ) : Prop := 15 ≤ x ∧ x ≤ 35

/-- Theorem stating that the cost function is minimized at x = 35 within the given constraints -/
theorem cost_minimized_at_35 :
  ∀ x : ℝ, constraint x → cost_function x ≥ cost_function 35 :=
sorry

end NUMINAMATH_CALUDE_cost_minimized_at_35_l71_7154


namespace NUMINAMATH_CALUDE_clock_tower_rings_per_year_l71_7121

/-- The number of times a clock tower bell rings in a year -/
def bell_rings_per_year (rings_per_hour : ℕ) (hours_per_day : ℕ) (days_per_year : ℕ) : ℕ :=
  rings_per_hour * hours_per_day * days_per_year

/-- Theorem: The clock tower bell rings 8760 times in a year -/
theorem clock_tower_rings_per_year :
  bell_rings_per_year 1 24 365 = 8760 := by
  sorry

end NUMINAMATH_CALUDE_clock_tower_rings_per_year_l71_7121


namespace NUMINAMATH_CALUDE_tom_seashells_l71_7106

/-- The number of seashells Tom has after giving some away -/
def remaining_seashells (initial : ℕ) (given_away : ℕ) : ℕ :=
  initial - given_away

/-- Theorem: Tom has 3 seashells after initially finding 5 and giving away 2 -/
theorem tom_seashells : remaining_seashells 5 2 = 3 := by
  sorry

end NUMINAMATH_CALUDE_tom_seashells_l71_7106


namespace NUMINAMATH_CALUDE_exp_two_log_five_equals_twentyfive_l71_7145

theorem exp_two_log_five_equals_twentyfive : 
  Real.exp (2 * Real.log 5) = 25 := by sorry

end NUMINAMATH_CALUDE_exp_two_log_five_equals_twentyfive_l71_7145


namespace NUMINAMATH_CALUDE_M_inter_N_eq_N_l71_7162

-- Define the sets M and N
def M : Set ℝ := {x : ℝ | x ≥ 0}
def N : Set ℝ := {x : ℝ | x^2 ≤ x}

-- Theorem statement
theorem M_inter_N_eq_N : M ∩ N = N := by
  sorry

end NUMINAMATH_CALUDE_M_inter_N_eq_N_l71_7162


namespace NUMINAMATH_CALUDE_apple_pile_count_l71_7146

theorem apple_pile_count : ∃! n : ℕ,
  50 < n ∧ n < 70 ∧
  n % 2 = 0 ∧ n % 3 = 0 ∧ n % 5 = 0 ∧
  n % 1 = 0 ∧ n % 2 = 0 ∧ n % 4 = 0 ∧ n % 6 = 0 ∧ n % 10 = 0 ∧
  n % 12 = 0 ∧ n % 15 = 0 ∧ n % 20 = 0 ∧ n % 30 = 0 ∧ n % 60 = 0 ∧
  n = 60 := by
sorry

end NUMINAMATH_CALUDE_apple_pile_count_l71_7146


namespace NUMINAMATH_CALUDE_valid_arrangement_probability_l71_7141

/-- Represents the color of a bead -/
inductive BeadColor
  | Green
  | Yellow
  | Purple

/-- Represents an arrangement of beads -/
def BeadArrangement := List BeadColor

/-- Checks if an arrangement is valid according to the given conditions -/
def isValidArrangement (arr : BeadArrangement) : Bool :=
  sorry

/-- Counts the number of valid arrangements -/
def countValidArrangements (green yellow purple : Nat) : Nat :=
  sorry

/-- Calculates the total number of possible arrangements -/
def totalArrangements (green yellow purple : Nat) : Nat :=
  sorry

/-- Theorem stating the probability of a valid arrangement -/
theorem valid_arrangement_probability :
  let green := 4
  let yellow := 3
  let purple := 2
  (countValidArrangements green yellow purple : Rat) / (totalArrangements green yellow purple) = 7 / 315 :=
sorry

end NUMINAMATH_CALUDE_valid_arrangement_probability_l71_7141


namespace NUMINAMATH_CALUDE_right_triangle_sin_value_l71_7129

theorem right_triangle_sin_value (A B C : ℝ) (h1 : 0 < A ∧ A < π/2) (h2 : 0 < B ∧ B < π/2) :
  B = π/2 → 2 * Real.sin A = 3 * Real.cos A → Real.sin A = 3 * Real.sqrt 13 / 13 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_sin_value_l71_7129


namespace NUMINAMATH_CALUDE_equation_solution_l71_7102

theorem equation_solution : ∃! x : ℝ, (4 : ℝ) ^ (x + 6) = 64 ^ x :=
  have h : (4 : ℝ) ^ (3 + 6) = 64 ^ 3 := by sorry
  ⟨3, h, λ y hy => by sorry⟩

#check equation_solution

end NUMINAMATH_CALUDE_equation_solution_l71_7102


namespace NUMINAMATH_CALUDE_inequality_equivalence_l71_7131

theorem inequality_equivalence (x : ℝ) : (x + 2) * (x - 9) < 0 ↔ -2 < x ∧ x < 9 := by
  sorry

end NUMINAMATH_CALUDE_inequality_equivalence_l71_7131


namespace NUMINAMATH_CALUDE_coin_flip_probability_l71_7122

/-- The number of coins being flipped -/
def num_coins : ℕ := 5

/-- The number of coins that need to match -/
def num_matching : ℕ := 3

/-- The number of possible outcomes for each coin -/
def outcomes_per_coin : ℕ := 2

/-- The total number of possible outcomes when flipping the coins -/
def total_outcomes : ℕ := outcomes_per_coin ^ num_coins

/-- The number of successful outcomes where the specified coins match -/
def successful_outcomes : ℕ := outcomes_per_coin * outcomes_per_coin ^ (num_coins - num_matching)

/-- The probability of the specified coins matching -/
def probability : ℚ := successful_outcomes / total_outcomes

theorem coin_flip_probability : probability = 1 / 4 := by sorry

end NUMINAMATH_CALUDE_coin_flip_probability_l71_7122


namespace NUMINAMATH_CALUDE_not_all_isosceles_30_similar_l71_7115

-- Define an isosceles triangle with one 30° angle
structure IsoscelesTriangle30 :=
  (base : ℝ)
  (leg : ℝ)
  (base_positive : base > 0)
  (leg_positive : leg > 0)
  (angle30 : ℝ)
  (angle30_is_30_degrees : angle30 = 30 * π / 180)
  (is_isosceles : leg = leg)

-- Define similarity for triangles
def are_similar (t1 t2 : IsoscelesTriangle30) : Prop :=
  ∃ (k : ℝ), k > 0 ∧ t1.base = k * t2.base ∧ t1.leg = k * t2.leg

-- Theorem statement
theorem not_all_isosceles_30_similar :
  ∃ (t1 t2 : IsoscelesTriangle30), ¬(are_similar t1 t2) :=
sorry

end NUMINAMATH_CALUDE_not_all_isosceles_30_similar_l71_7115


namespace NUMINAMATH_CALUDE_parallelogram_base_formula_l71_7166

/-- Given two right-angled parallelograms with bases x and z, and heights y and u,
    this theorem proves the formula for x given certain conditions. -/
theorem parallelogram_base_formula 
  (x z y u S p s s' : ℝ) 
  (h1 : x * y + z * u = S) 
  (h2 : x + z = p) 
  (h3 : z * y = s) 
  (h4 : x * u = s') : 
  x = (p * (2 * s' + S) + Real.sqrt (p^2 * (2 * s' + S)^2 - 4 * p^2 * s' * (s + s' + S))) / (2 * (s + s' + S)) ∨ 
  x = (p * (2 * s' + S) - Real.sqrt (p^2 * (2 * s' + S)^2 - 4 * p^2 * s' * (s + s' + S))) / (2 * (s + s' + S)) :=
sorry

end NUMINAMATH_CALUDE_parallelogram_base_formula_l71_7166


namespace NUMINAMATH_CALUDE_two_A_minus_three_B_two_A_minus_three_B_equals_seven_l71_7178

-- Define A and B as functions of x and y
def A (x y : ℝ) : ℝ := 3 * x^2 + 2 * y^2 - 2 * x * y
def B (x y : ℝ) : ℝ := y^2 - x * y + 2 * x^2

-- Theorem 1: 2A - 3B = y² - xy
theorem two_A_minus_three_B (x y : ℝ) : 2 * A x y - 3 * B x y = y^2 - x * y := by
  sorry

-- Theorem 2: 2A - 3B = 7 under the given condition
theorem two_A_minus_three_B_equals_seven (x y : ℝ) 
  (h : |2 * x - 3| + (y + 2)^2 = 0) : 2 * A x y - 3 * B x y = 7 := by
  sorry

end NUMINAMATH_CALUDE_two_A_minus_three_B_two_A_minus_three_B_equals_seven_l71_7178


namespace NUMINAMATH_CALUDE_cos_alpha_plus_20_eq_neg_alpha_l71_7143

theorem cos_alpha_plus_20_eq_neg_alpha (α : ℝ) (h : Real.sin (α - 70 * π / 180) = α) :
  Real.cos (α + 20 * π / 180) = -α := by sorry

end NUMINAMATH_CALUDE_cos_alpha_plus_20_eq_neg_alpha_l71_7143


namespace NUMINAMATH_CALUDE_three_aligned_probability_l71_7101

-- Define the grid
def Grid := Fin 3 × Fin 3

-- Define the number of markers
def num_markers : ℕ := 4

-- Define the total number of cells in the grid
def total_cells : ℕ := 9

-- Define the function to calculate combinations
def combination (n k : ℕ) : ℕ := (Nat.factorial n) / (Nat.factorial k * Nat.factorial (n - k))

-- Define the total number of ways to place markers
def total_arrangements : ℕ := combination total_cells num_markers

-- Define the number of ways to align 3 markers in a row, column, or diagonal
def aligned_arrangements : ℕ := 48

-- The main theorem
theorem three_aligned_probability :
  (aligned_arrangements : ℚ) / total_arrangements = 8 / 21 := by
  sorry

end NUMINAMATH_CALUDE_three_aligned_probability_l71_7101


namespace NUMINAMATH_CALUDE_problem_statement_l71_7148

theorem problem_statement : 
  (∀ x : ℝ, ∀ a : ℝ, x^2 + a*x + a^2 ≥ 0) ∨ (∃ x₀ : ℕ+, 2 * (x₀.val)^2 - 1 ≤ 0) :=
by sorry

end NUMINAMATH_CALUDE_problem_statement_l71_7148


namespace NUMINAMATH_CALUDE_cake_sugar_amount_l71_7189

theorem cake_sugar_amount (total_sugar frosting_sugar : ℚ)
  (h1 : total_sugar = 0.8)
  (h2 : frosting_sugar = 0.6) :
  total_sugar - frosting_sugar = 0.2 := by
sorry

end NUMINAMATH_CALUDE_cake_sugar_amount_l71_7189


namespace NUMINAMATH_CALUDE_prob_same_gender_is_one_third_l71_7191

/-- Represents the gender of a student -/
inductive Gender
| Male
| Female

/-- Represents a group of students -/
structure StudentGroup where
  males : Finset Gender
  females : Finset Gender
  male_count : males.card = 2
  female_count : females.card = 2

/-- Represents a selection of two students -/
structure Selection where
  first : Gender
  second : Gender

/-- The probability of selecting two students of the same gender -/
def prob_same_gender (group : StudentGroup) : ℚ :=
  (2 : ℚ) / 6

theorem prob_same_gender_is_one_third (group : StudentGroup) :
  prob_same_gender group = (1 : ℚ) / 3 := by
  sorry

end NUMINAMATH_CALUDE_prob_same_gender_is_one_third_l71_7191


namespace NUMINAMATH_CALUDE_shaded_area_octagon_with_sectors_l71_7176

/-- The area of the shaded region in a regular octagon with circular sectors --/
theorem shaded_area_octagon_with_sectors (side_length : Real) (sector_radius : Real) : 
  side_length = 5 → sector_radius = 3 → 
  ∃ (shaded_area : Real), shaded_area = 100 - 9 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_shaded_area_octagon_with_sectors_l71_7176


namespace NUMINAMATH_CALUDE_no_rain_probability_l71_7151

theorem no_rain_probability (pMonday pTuesday pBoth : ℝ) 
  (hMonday : pMonday = 0.6)
  (hTuesday : pTuesday = 0.55)
  (hBoth : pBoth = 0.4) :
  1 - (pMonday + pTuesday - pBoth) = 0.25 := by
sorry

end NUMINAMATH_CALUDE_no_rain_probability_l71_7151


namespace NUMINAMATH_CALUDE_absolute_value_trig_expression_l71_7184

theorem absolute_value_trig_expression : 
  |(-3 : ℝ)| + Real.sqrt 3 * Real.sin (60 * π / 180) - (1 / 2) = 4 := by
sorry

end NUMINAMATH_CALUDE_absolute_value_trig_expression_l71_7184


namespace NUMINAMATH_CALUDE_hyperbola_perimeter_l71_7172

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x^2 - y^2/4 = 1

-- Define the properties of the hyperbola and points
def hyperbola_properties (F₁ F₂ P Q : ℝ × ℝ) : Prop :=
  ∃ (l : Set (ℝ × ℝ)),
    hyperbola P.1 P.2 ∧ 
    hyperbola Q.1 Q.2 ∧
    P ∈ l ∧ Q ∈ l ∧
    F₁.1 < P.1 ∧ F₁.1 < Q.1 ∧
    F₂.1 > F₁.1 ∧
    ‖P - Q‖ = 4

-- Theorem statement
theorem hyperbola_perimeter (F₁ F₂ P Q : ℝ × ℝ) 
  (h : hyperbola_properties F₁ F₂ P Q) :
  ‖P - F₂‖ + ‖Q - F₂‖ + ‖P - Q‖ = 12 :=
sorry

end NUMINAMATH_CALUDE_hyperbola_perimeter_l71_7172


namespace NUMINAMATH_CALUDE_sum_of_factors_360_l71_7167

/-- The sum of positive factors of a natural number n -/
def sum_of_factors (n : ℕ) : ℕ := sorry

/-- Theorem: The sum of positive factors of 360 is 1170 -/
theorem sum_of_factors_360 : sum_of_factors 360 = 1170 := by sorry

end NUMINAMATH_CALUDE_sum_of_factors_360_l71_7167


namespace NUMINAMATH_CALUDE_trigonometric_identity_l71_7152

theorem trigonometric_identity : 
  Real.sin (45 * π / 180) * Real.cos (15 * π / 180) + 
  Real.cos (225 * π / 180) * Real.sin (165 * π / 180) = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_identity_l71_7152
