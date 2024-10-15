import Mathlib

namespace NUMINAMATH_CALUDE_quadratic_equation_equivalence_l2476_247692

theorem quadratic_equation_equivalence :
  ∀ x : ℝ, x^2 - 8*x + 10 = 0 ↔ (x - 4)^2 = 6 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_equivalence_l2476_247692


namespace NUMINAMATH_CALUDE_infinite_fixed_points_l2476_247695

def is_special_function (f : ℕ → ℕ) : Prop :=
  ∀ n : ℕ, (f n - n < 2021) ∧ (f^[f n] n = n)

theorem infinite_fixed_points (f : ℕ → ℕ) (hf : is_special_function f) :
  Set.Infinite {n : ℕ | f n = n} :=
sorry

end NUMINAMATH_CALUDE_infinite_fixed_points_l2476_247695


namespace NUMINAMATH_CALUDE_log_equation_implies_x_greater_than_two_l2476_247667

theorem log_equation_implies_x_greater_than_two (x : ℝ) :
  Real.log (x^2 + 5*x + 6) = Real.log ((x+1)*(x+4)) + Real.log (x-2) →
  x > 2 :=
by sorry

end NUMINAMATH_CALUDE_log_equation_implies_x_greater_than_two_l2476_247667


namespace NUMINAMATH_CALUDE_exponent_rule_l2476_247629

theorem exponent_rule (a : ℝ) : a^2 * a^3 = a^5 := by
  sorry

end NUMINAMATH_CALUDE_exponent_rule_l2476_247629


namespace NUMINAMATH_CALUDE_power_fraction_simplification_l2476_247673

theorem power_fraction_simplification :
  (2^2023 + 2^2019) / (2^2023 - 2^2019) = 17 / 15 := by
  sorry

end NUMINAMATH_CALUDE_power_fraction_simplification_l2476_247673


namespace NUMINAMATH_CALUDE_sqrt5_parts_sqrt2_plus_1_parts_sqrt3_plus_2_parts_l2476_247609

-- Define the irrational numbers
axiom sqrt2 : ℝ
axiom sqrt3 : ℝ
axiom sqrt5 : ℝ

-- Define the properties of these irrational numbers
axiom sqrt2_irrational : Irrational sqrt2
axiom sqrt3_irrational : Irrational sqrt3
axiom sqrt5_irrational : Irrational sqrt5

axiom sqrt2_bounds : 1 < sqrt2 ∧ sqrt2 < 2
axiom sqrt3_bounds : 1 < sqrt3 ∧ sqrt3 < 2
axiom sqrt5_bounds : 2 < sqrt5 ∧ sqrt5 < 3

-- Define the integer and decimal part functions
def intPart (x : ℝ) : ℤ := sorry
def decPart (x : ℝ) : ℝ := sorry

-- Theorem statements
theorem sqrt5_parts : intPart sqrt5 = 2 ∧ decPart sqrt5 = sqrt5 - 2 := by sorry

theorem sqrt2_plus_1_parts : intPart (1 + sqrt2) = 2 ∧ decPart (1 + sqrt2) = sqrt2 - 1 := by sorry

theorem sqrt3_plus_2_parts :
  let x := intPart (2 + sqrt3)
  let y := decPart (2 + sqrt3)
  x - sqrt3 * y = sqrt3 := by sorry

end NUMINAMATH_CALUDE_sqrt5_parts_sqrt2_plus_1_parts_sqrt3_plus_2_parts_l2476_247609


namespace NUMINAMATH_CALUDE_random_sampling_appropriate_for_air_quality_l2476_247618

/-- Represents a survey method -/
inductive SurveyMethod
| Comprehensive
| RandomSampling

/-- Represents a scenario for which a survey method is chosen -/
inductive Scenario
| LightBulbLifespan
| FoodPreservatives
| SpaceEquipmentQuality
| AirQuality

/-- Determines if a survey method is appropriate for a given scenario -/
def isAppropriate (method : SurveyMethod) (scenario : Scenario) : Prop :=
  match scenario with
  | Scenario.LightBulbLifespan => method = SurveyMethod.RandomSampling
  | Scenario.FoodPreservatives => method = SurveyMethod.RandomSampling
  | Scenario.SpaceEquipmentQuality => method = SurveyMethod.Comprehensive
  | Scenario.AirQuality => method = SurveyMethod.RandomSampling

/-- Theorem stating that random sampling is appropriate for air quality measurement -/
theorem random_sampling_appropriate_for_air_quality :
  isAppropriate SurveyMethod.RandomSampling Scenario.AirQuality :=
by
  sorry

#check random_sampling_appropriate_for_air_quality

end NUMINAMATH_CALUDE_random_sampling_appropriate_for_air_quality_l2476_247618


namespace NUMINAMATH_CALUDE_ceiling_floor_sum_l2476_247633

theorem ceiling_floor_sum : ⌈(7 : ℝ) / 3⌉ + ⌊-(7 : ℝ) / 3⌋ = 0 := by
  sorry

end NUMINAMATH_CALUDE_ceiling_floor_sum_l2476_247633


namespace NUMINAMATH_CALUDE_base10_to_base8_2357_l2476_247686

-- Define a function to convert a base 10 number to base 8
def toBase8 (n : ℕ) : List ℕ :=
  sorry

-- Theorem stating that 2357 in base 10 is equal to 4445 in base 8
theorem base10_to_base8_2357 :
  toBase8 2357 = [4, 4, 4, 5] :=
sorry

end NUMINAMATH_CALUDE_base10_to_base8_2357_l2476_247686


namespace NUMINAMATH_CALUDE_jacob_jogging_distance_l2476_247657

/-- Calculates the total distance jogged given a constant speed and total jogging time -/
def total_distance_jogged (speed : ℝ) (time : ℝ) : ℝ := speed * time

/-- Theorem stating that jogging at 4 miles per hour for 3 hours results in a total distance of 12 miles -/
theorem jacob_jogging_distance :
  let speed : ℝ := 4
  let time : ℝ := 3
  total_distance_jogged speed time = 12 := by
  sorry

end NUMINAMATH_CALUDE_jacob_jogging_distance_l2476_247657


namespace NUMINAMATH_CALUDE_apple_cost_is_21_cents_l2476_247624

/-- The cost of an apple and an orange satisfy the given conditions -/
def apple_orange_cost (apple_cost orange_cost : ℚ) : Prop :=
  6 * apple_cost + 3 * orange_cost = 177/100 ∧
  2 * apple_cost + 5 * orange_cost = 127/100

/-- The cost of an apple is 0.21 dollars -/
theorem apple_cost_is_21_cents :
  ∃ (orange_cost : ℚ), apple_orange_cost (21/100) orange_cost := by
  sorry

end NUMINAMATH_CALUDE_apple_cost_is_21_cents_l2476_247624


namespace NUMINAMATH_CALUDE_triangle_DEF_circles_l2476_247632

/-- Triangle DEF with side lengths -/
structure Triangle where
  DE : ℝ
  DF : ℝ
  EF : ℝ

/-- The inscribed circle of a triangle -/
def inscribedCircleDiameter (t : Triangle) : ℝ := sorry

/-- The circumscribed circle of a triangle -/
def circumscribedCircleRadius (t : Triangle) : ℝ := sorry

/-- Main theorem about triangle DEF -/
theorem triangle_DEF_circles :
  let t : Triangle := { DE := 13, DF := 8, EF := 9 }
  inscribedCircleDiameter t = 2 * Real.sqrt 14 ∧
  circumscribedCircleRadius t = (39 * Real.sqrt 14) / 35 := by sorry

end NUMINAMATH_CALUDE_triangle_DEF_circles_l2476_247632


namespace NUMINAMATH_CALUDE_basketball_game_scores_l2476_247616

/-- Represents the quarterly scores of a team --/
structure QuarterlyScores :=
  (q1 q2 q3 q4 : ℕ)

/-- Check if a sequence of four numbers is arithmetic --/
def isArithmetic (s : QuarterlyScores) : Prop :=
  s.q2 - s.q1 = s.q3 - s.q2 ∧ s.q3 - s.q2 = s.q4 - s.q3

/-- Check if a sequence of four numbers is geometric --/
def isGeometric (s : QuarterlyScores) : Prop :=
  s.q1 > 0 ∧ s.q2 % s.q1 = 0 ∧ s.q3 % s.q2 = 0 ∧ s.q4 % s.q3 = 0 ∧
  s.q2 / s.q1 = s.q3 / s.q2 ∧ s.q3 / s.q2 = s.q4 / s.q3

/-- Sum of all quarterly scores --/
def totalScore (s : QuarterlyScores) : ℕ :=
  s.q1 + s.q2 + s.q3 + s.q4

theorem basketball_game_scores :
  ∃ (raiders wildcats : QuarterlyScores),
    -- Tied at halftime
    raiders.q1 + raiders.q2 = wildcats.q1 + wildcats.q2 ∧
    -- Raiders' scores form an arithmetic sequence
    isArithmetic raiders ∧
    -- Wildcats' scores form a geometric sequence
    isGeometric wildcats ∧
    -- Fourth quarter combined score is half of total combined score
    raiders.q4 + wildcats.q4 = (totalScore raiders + totalScore wildcats) / 2 ∧
    -- Neither team scored more than 100 points
    totalScore raiders ≤ 100 ∧ totalScore wildcats ≤ 100 ∧
    -- First quarter total is one of the given options
    (raiders.q1 + wildcats.q1 = 10 ∨
     raiders.q1 + wildcats.q1 = 15 ∨
     raiders.q1 + wildcats.q1 = 20 ∨
     raiders.q1 + wildcats.q1 = 9 ∨
     raiders.q1 + wildcats.q1 = 12) :=
by
  sorry

end NUMINAMATH_CALUDE_basketball_game_scores_l2476_247616


namespace NUMINAMATH_CALUDE_lemon_cost_lemon_cost_is_fifty_cents_l2476_247696

/-- The cost of the lemon in Hannah's apple pie recipe -/
theorem lemon_cost (servings : ℕ) (apple_pounds : ℝ) (apple_price : ℝ) 
  (crust_price : ℝ) (butter_price : ℝ) (serving_price : ℝ) : ℝ :=
  let total_cost := servings * serving_price
  let ingredients_cost := apple_pounds * apple_price + crust_price + butter_price
  total_cost - ingredients_cost

/-- The lemon in Hannah's apple pie recipe costs $0.50 -/
theorem lemon_cost_is_fifty_cents : 
  lemon_cost 8 2 2 2 1.5 1 = 0.5 := by
  sorry

end NUMINAMATH_CALUDE_lemon_cost_lemon_cost_is_fifty_cents_l2476_247696


namespace NUMINAMATH_CALUDE_eight_digit_integers_count_l2476_247634

/-- The number of choices for the first digit -/
def first_digit_choices : ℕ := 9

/-- The number of choices for each of the remaining seven digits -/
def remaining_digit_choices : ℕ := 5

/-- The number of remaining digits -/
def remaining_digits : ℕ := 7

/-- The total number of different 8-digit positive integers under the given conditions -/
def total_combinations : ℕ := first_digit_choices * remaining_digit_choices ^ remaining_digits

theorem eight_digit_integers_count : total_combinations = 703125 := by
  sorry

end NUMINAMATH_CALUDE_eight_digit_integers_count_l2476_247634


namespace NUMINAMATH_CALUDE_initial_student_count_l2476_247683

theorem initial_student_count (initial_avg : ℝ) (new_avg : ℝ) (new_student_weight : ℝ) :
  initial_avg = 15 →
  new_avg = 14.4 →
  new_student_weight = 3 →
  ∃ n : ℕ, n * initial_avg + new_student_weight = (n + 1) * new_avg ∧ n = 19 :=
by
  sorry

end NUMINAMATH_CALUDE_initial_student_count_l2476_247683


namespace NUMINAMATH_CALUDE_seashells_count_l2476_247664

theorem seashells_count (joan_shells jessica_shells : ℕ) 
  (h1 : joan_shells = 6) 
  (h2 : jessica_shells = 8) : 
  joan_shells + jessica_shells = 14 := by
  sorry

end NUMINAMATH_CALUDE_seashells_count_l2476_247664


namespace NUMINAMATH_CALUDE_fraction_multiplication_equality_l2476_247621

theorem fraction_multiplication_equality : 
  (5 / 8 : ℚ)^2 * (3 / 4 : ℚ)^2 * (2 / 3 : ℚ) = 75 / 512 := by
  sorry

end NUMINAMATH_CALUDE_fraction_multiplication_equality_l2476_247621


namespace NUMINAMATH_CALUDE_negation_of_existence_is_forall_l2476_247614

theorem negation_of_existence_is_forall :
  (¬ ∃ x : ℝ, x^2 + 1 < 0) ↔ (∀ x : ℝ, x^2 + 1 ≥ 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_existence_is_forall_l2476_247614


namespace NUMINAMATH_CALUDE_exists_increasing_sequence_with_gcd_property_l2476_247661

theorem exists_increasing_sequence_with_gcd_property :
  ∃ (a : ℕ → ℕ), 
    (∀ n : ℕ, a n < a (n + 1)) ∧ 
    (∀ i j : ℕ, i ≠ j → Nat.gcd (i * a j) (j * a i) = Nat.gcd i j) := by
  sorry

end NUMINAMATH_CALUDE_exists_increasing_sequence_with_gcd_property_l2476_247661


namespace NUMINAMATH_CALUDE_max_candies_ben_l2476_247631

/-- The maximum number of candies Ben can eat -/
theorem max_candies_ben (total : ℕ) (h_total : total = 30) : ∃ (b : ℕ), b ≤ 6 ∧ 
  ∀ (k : ℕ+) (b' : ℕ), b' + 2 * b' + k * b' = total → b' ≤ b :=
sorry

end NUMINAMATH_CALUDE_max_candies_ben_l2476_247631


namespace NUMINAMATH_CALUDE_special_rectangle_dimensions_l2476_247693

/-- A rectangle with specific properties -/
structure SpecialRectangle where
  width : ℝ
  length : ℝ
  width_pos : 0 < width
  length_pos : 0 < length
  length_twice_width : length = 2 * width
  perimeter_three_times_area : 2 * (length + width) = 3 * (length * width)

/-- The width and length of the special rectangle are 1 and 2, respectively -/
theorem special_rectangle_dimensions (r : SpecialRectangle) : r.width = 1 ∧ r.length = 2 := by
  sorry


end NUMINAMATH_CALUDE_special_rectangle_dimensions_l2476_247693


namespace NUMINAMATH_CALUDE_square_garden_perimeter_l2476_247690

theorem square_garden_perimeter (area : ℝ) (perimeter : ℝ) : 
  area = 520 → perimeter = 40 * Real.sqrt 26 := by
  sorry

end NUMINAMATH_CALUDE_square_garden_perimeter_l2476_247690


namespace NUMINAMATH_CALUDE_right_triangle_circle_ratio_l2476_247604

theorem right_triangle_circle_ratio (a b : ℝ) (ha : a = 6) (hb : b = 8) :
  let c := Real.sqrt (a^2 + b^2)
  let R := c / 2
  let r := (a + b - c) / 2
  R / r = 5 / 2 := by sorry

end NUMINAMATH_CALUDE_right_triangle_circle_ratio_l2476_247604


namespace NUMINAMATH_CALUDE_triangle_angle_ratio_l2476_247676

theorem triangle_angle_ratio (A B C : ℝ) : 
  A + B + C = 180 →  -- Sum of angles in a triangle
  A = 20 →           -- Smallest angle
  B = 3 * A →        -- Middle angle is 3 times the smallest
  A ≤ B →            -- B is larger than or equal to A
  B ≤ C →            -- C is the largest angle
  C / A = 5 :=       -- Ratio of largest to smallest is 5:1
by sorry

end NUMINAMATH_CALUDE_triangle_angle_ratio_l2476_247676


namespace NUMINAMATH_CALUDE_revenue_decrease_percentage_l2476_247606

def old_revenue : ℝ := 85.0
def new_revenue : ℝ := 48.0

theorem revenue_decrease_percentage :
  abs (((old_revenue - new_revenue) / old_revenue) * 100 - 43.53) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_revenue_decrease_percentage_l2476_247606


namespace NUMINAMATH_CALUDE_grandmother_doll_count_l2476_247691

/-- Represents the number of dolls each person has -/
structure DollCounts where
  grandmother : ℕ
  sister : ℕ
  rene : ℕ

/-- Defines the conditions of the doll distribution problem -/
def validDollDistribution (d : DollCounts) : Prop :=
  d.rene = 3 * d.sister ∧
  d.sister = d.grandmother + 2 ∧
  d.rene + d.sister + d.grandmother = 258

/-- Theorem stating that the grandmother has 50 dolls -/
theorem grandmother_doll_count :
  ∀ d : DollCounts, validDollDistribution d → d.grandmother = 50 := by
  sorry

end NUMINAMATH_CALUDE_grandmother_doll_count_l2476_247691


namespace NUMINAMATH_CALUDE_fruit_drink_composition_l2476_247630

-- Define the composition of the fruit drink
def orange_percent : ℝ := 25
def watermelon_percent : ℝ := 40
def grape_ounces : ℝ := 70

-- Define the total volume of the drink
def total_volume : ℝ := 200

-- Theorem statement
theorem fruit_drink_composition :
  orange_percent + watermelon_percent + (grape_ounces / total_volume * 100) = 100 ∧
  grape_ounces / (grape_ounces / total_volume * 100) * 100 = total_volume :=
by sorry

end NUMINAMATH_CALUDE_fruit_drink_composition_l2476_247630


namespace NUMINAMATH_CALUDE_oscar_swag_bag_value_l2476_247617

/-- The total value of a swag bag with specified items -/
def swag_bag_value (earring_cost : ℕ) (iphone_cost : ℕ) (scarf_cost : ℕ) : ℕ :=
  2 * earring_cost + iphone_cost + 4 * scarf_cost

/-- Theorem: The total value of the Oscar swag bag is $20,000 -/
theorem oscar_swag_bag_value :
  swag_bag_value 6000 2000 1500 = 20000 := by
  sorry

end NUMINAMATH_CALUDE_oscar_swag_bag_value_l2476_247617


namespace NUMINAMATH_CALUDE_pills_in_week_l2476_247620

/-- Calculates the number of pills taken in a week given the interval between pills in hours -/
def pills_per_week (hours_between_pills : ℕ) : ℕ :=
  let hours_per_day : ℕ := 24
  let days_per_week : ℕ := 7
  (hours_per_day / hours_between_pills) * days_per_week

/-- Theorem: A person who takes a pill every 6 hours will take 28 pills in a week -/
theorem pills_in_week : pills_per_week 6 = 28 := by
  sorry

end NUMINAMATH_CALUDE_pills_in_week_l2476_247620


namespace NUMINAMATH_CALUDE_matching_allocation_theorem_l2476_247641

/-- Represents the allocation of workers to produce parts A and B -/
structure WorkerAllocation where
  partA : ℕ
  partB : ℕ

/-- Checks if the given allocation produces matching sets of parts A and B -/
def isMatchingAllocation (totalWorkers : ℕ) (prodRateA : ℕ) (prodRateB : ℕ) (allocation : WorkerAllocation) : Prop :=
  allocation.partA + allocation.partB = totalWorkers ∧
  prodRateB * allocation.partB = 2 * (prodRateA * allocation.partA)

/-- Theorem stating that the given allocation produces matching sets -/
theorem matching_allocation_theorem :
  let totalWorkers : ℕ := 50
  let prodRateA : ℕ := 40
  let prodRateB : ℕ := 120
  let allocation : WorkerAllocation := ⟨30, 20⟩
  isMatchingAllocation totalWorkers prodRateA prodRateB allocation := by
  sorry

#check matching_allocation_theorem

end NUMINAMATH_CALUDE_matching_allocation_theorem_l2476_247641


namespace NUMINAMATH_CALUDE_lcm_of_45_and_200_l2476_247627

theorem lcm_of_45_and_200 : Nat.lcm 45 200 = 1800 := by
  sorry

end NUMINAMATH_CALUDE_lcm_of_45_and_200_l2476_247627


namespace NUMINAMATH_CALUDE_four_similar_triangle_solutions_l2476_247602

-- Define a triangle
structure Triangle :=
  (A B C : ℝ × ℝ)

-- Define a point
def Point := ℝ × ℝ

-- Define a line
structure Line :=
  (m b : ℝ)

-- Function to check if a point is on a side of a triangle
def isPointOnSide (T : Triangle) (P : Point) : Prop :=
  sorry

-- Function to check if two triangles are similar
def areSimilarTriangles (T1 T2 : Triangle) : Prop :=
  sorry

-- Function to check if a line intersects a triangle
def lineIntersectsTriangle (L : Line) (T : Triangle) : Prop :=
  sorry

-- Function to get the triangle cut off by a line
def getCutOffTriangle (T : Triangle) (L : Line) : Triangle :=
  sorry

-- The main theorem
theorem four_similar_triangle_solutions 
  (T : Triangle) (P : Point) (h : isPointOnSide T P) :
  ∃ (L1 L2 L3 L4 : Line),
    (L1 ≠ L2 ∧ L1 ≠ L3 ∧ L1 ≠ L4 ∧ L2 ≠ L3 ∧ L2 ≠ L4 ∧ L3 ≠ L4) ∧
    (∀ (L : Line), 
      (lineIntersectsTriangle L T ∧ areSimilarTriangles (getCutOffTriangle T L) T) →
      (L = L1 ∨ L = L2 ∨ L = L3 ∨ L = L4)) :=
sorry

end NUMINAMATH_CALUDE_four_similar_triangle_solutions_l2476_247602


namespace NUMINAMATH_CALUDE_abs_eq_iff_eq_l2476_247601

theorem abs_eq_iff_eq (x y : ℝ) : 
  (|x| = |y| → x = y) ↔ False ∧ 
  (x = y → |x| = |y|) :=
sorry

end NUMINAMATH_CALUDE_abs_eq_iff_eq_l2476_247601


namespace NUMINAMATH_CALUDE_peaches_sold_to_relatives_l2476_247669

theorem peaches_sold_to_relatives (total_peaches : ℕ) 
                                  (peaches_to_friends : ℕ) 
                                  (price_to_friends : ℚ)
                                  (price_to_relatives : ℚ)
                                  (peaches_kept : ℕ)
                                  (total_sold : ℕ)
                                  (total_earnings : ℚ) :
  total_peaches = 15 →
  peaches_to_friends = 10 →
  price_to_friends = 2 →
  price_to_relatives = 5/4 →
  peaches_kept = 1 →
  total_sold = 14 →
  total_earnings = 25 →
  total_peaches = peaches_to_friends + (total_sold - peaches_to_friends) + peaches_kept →
  total_earnings = peaches_to_friends * price_to_friends + 
                   (total_sold - peaches_to_friends) * price_to_relatives →
  (total_sold - peaches_to_friends) = 4 := by
sorry

end NUMINAMATH_CALUDE_peaches_sold_to_relatives_l2476_247669


namespace NUMINAMATH_CALUDE_rice_distribution_difference_l2476_247636

/-- Given a total amount of rice and the fraction kept by Mr. Llesis,
    calculate how much more rice Mr. Llesis keeps compared to Mr. Everest. -/
def rice_difference (total : ℚ) (llesis_fraction : ℚ) : ℚ :=
  let llesis_amount := total * llesis_fraction
  let everest_amount := total - llesis_amount
  llesis_amount - everest_amount

/-- Theorem stating that given 50 kg of rice, if Mr. Llesis keeps 7/10 of it,
    he will have 20 kg more than Mr. Everest. -/
theorem rice_distribution_difference :
  rice_difference 50 (7/10) = 20 := by
  sorry

end NUMINAMATH_CALUDE_rice_distribution_difference_l2476_247636


namespace NUMINAMATH_CALUDE_simplify_polynomial_l2476_247651

theorem simplify_polynomial (b : ℝ) : (1 : ℝ) * (3 * b) * (4 * b^2) * (5 * b^3) * (6 * b^4) = 360 * b^10 := by
  sorry

end NUMINAMATH_CALUDE_simplify_polynomial_l2476_247651


namespace NUMINAMATH_CALUDE_max_value_theorem_l2476_247622

open Real

noncomputable def e : ℝ := Real.exp 1

theorem max_value_theorem (a b : ℝ) :
  (∀ x : ℝ, (e - a) * (Real.exp x) + x + b + 1 ≤ 0) →
  (b + 1) / a ≤ 1 / e :=
by sorry

end NUMINAMATH_CALUDE_max_value_theorem_l2476_247622


namespace NUMINAMATH_CALUDE_triangle_perimeter_l2476_247612

theorem triangle_perimeter (AB AC : ℝ) (h_right_angle : AB ^ 2 + AC ^ 2 = (AB + AC + Real.sqrt (AB ^ 2 + AC ^ 2)) ^ 2 - 2 * AB * AC) (h_AB : AB = 8) (h_AC : AC = 15) :
  AB + AC + Real.sqrt (AB ^ 2 + AC ^ 2) = 40 := by
  sorry

end NUMINAMATH_CALUDE_triangle_perimeter_l2476_247612


namespace NUMINAMATH_CALUDE_f_monotone_increasing_l2476_247682

theorem f_monotone_increasing (k : ℝ) (h_k : k ≥ 0) :
  ∀ x ≥ Real.sqrt (2 * k + 1), HasDerivAt (λ x => x + (2 * k + 1) / x) ((x^2 - (2 * k + 1)) / x^2) x ∧
  (x^2 - (2 * k + 1)) / x^2 ≥ 0 := by
  sorry

end NUMINAMATH_CALUDE_f_monotone_increasing_l2476_247682


namespace NUMINAMATH_CALUDE_mincheol_midterm_average_l2476_247656

/-- Calculates the average of three exam scores -/
def midterm_average (math_score korean_score english_score : ℕ) : ℚ :=
  (math_score + korean_score + english_score : ℚ) / 3

/-- Theorem: Mincheol's midterm average is 80 points -/
theorem mincheol_midterm_average : 
  midterm_average 70 80 90 = 80 := by
  sorry

end NUMINAMATH_CALUDE_mincheol_midterm_average_l2476_247656


namespace NUMINAMATH_CALUDE_rationalize_denominator_sqrt_5_l2476_247698

theorem rationalize_denominator_sqrt_5 :
  let x := 2 + Real.sqrt 5
  let y := 1 - Real.sqrt 5
  x / y = -7/4 - (3 * Real.sqrt 5) / 4 := by
  sorry

end NUMINAMATH_CALUDE_rationalize_denominator_sqrt_5_l2476_247698


namespace NUMINAMATH_CALUDE_area_scaled_and_shifted_l2476_247640

-- Define a function g: ℝ → ℝ
variable (g : ℝ → ℝ)

-- Define the area between a function and the x-axis
def area_between_curve_and_axis (f : ℝ → ℝ) : ℝ := sorry

-- Theorem statement
theorem area_scaled_and_shifted (h : area_between_curve_and_axis g = 15) :
  area_between_curve_and_axis (fun x ↦ 4 * g (x + 3)) = 60 := by
  sorry

end NUMINAMATH_CALUDE_area_scaled_and_shifted_l2476_247640


namespace NUMINAMATH_CALUDE_product_inequality_l2476_247615

theorem product_inequality (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + y = 1) :
  (1 + 1/x) * (1 + 1/y) ≥ 9 := by
  sorry

end NUMINAMATH_CALUDE_product_inequality_l2476_247615


namespace NUMINAMATH_CALUDE_alcohol_mixture_proof_l2476_247611

/-- Proves that mixing 200 mL of 10% alcohol solution with 50 mL of 30% alcohol solution 
    results in a 14% alcohol solution -/
theorem alcohol_mixture_proof (x_vol : ℝ) (y_vol : ℝ) (x_conc : ℝ) (y_conc : ℝ) 
    (mix_conc : ℝ) (h1 : x_vol = 200) (h2 : y_vol = 50) (h3 : x_conc = 0.1) 
    (h4 : y_conc = 0.3) (h5 : mix_conc = 0.14) : 
    (x_vol * x_conc + y_vol * y_conc) / (x_vol + y_vol) = mix_conc := by
  sorry

#check alcohol_mixture_proof

end NUMINAMATH_CALUDE_alcohol_mixture_proof_l2476_247611


namespace NUMINAMATH_CALUDE_max_area_rectangular_play_area_l2476_247668

/-- 
Given a rectangular area with perimeter P (excluding one side) and length l and width w,
prove that the maximum area A is achieved when l = P/2 and w = P/6, resulting in A = (P^2)/48.
-/
theorem max_area_rectangular_play_area (P : ℝ) (h : P > 0) :
  ∃ (l w : ℝ), l > 0 ∧ w > 0 ∧ l + 2*w = P ∧
  ∀ (l' w' : ℝ), l' > 0 → w' > 0 → l' + 2*w' = P →
  l * w ≥ l' * w' ∧
  l = P/2 ∧ w = P/6 ∧ l * w = (P^2)/48 :=
by sorry

end NUMINAMATH_CALUDE_max_area_rectangular_play_area_l2476_247668


namespace NUMINAMATH_CALUDE_complex_fraction_evaluation_l2476_247688

theorem complex_fraction_evaluation : 
  (1 : ℚ) / (1 - 1 / (3 + 1 / 4)) = 13 / 9 := by sorry

end NUMINAMATH_CALUDE_complex_fraction_evaluation_l2476_247688


namespace NUMINAMATH_CALUDE_zoo_population_after_changes_l2476_247645

/-- Represents the population of animals in a zoo --/
structure ZooPopulation where
  foxes : ℕ
  rabbits : ℕ

/-- Calculates the ratio of foxes to rabbits --/
def ratio (pop : ZooPopulation) : ℚ :=
  pop.foxes / pop.rabbits

theorem zoo_population_after_changes 
  (initial : ZooPopulation)
  (h1 : ratio initial = 2 / 3)
  (h2 : ratio { foxes := initial.foxes - 10, rabbits := initial.rabbits / 2 } = 13 / 10) :
  initial.foxes - 10 + initial.rabbits / 2 = 690 := by
  sorry


end NUMINAMATH_CALUDE_zoo_population_after_changes_l2476_247645


namespace NUMINAMATH_CALUDE_square_area_equal_perimeter_triangle_l2476_247663

theorem square_area_equal_perimeter_triangle (a b c : Real) (h1 : a = 7.5) (h2 : b = 5.3) (h3 : c = 11.2) :
  let triangle_perimeter := a + b + c
  let square_side := triangle_perimeter / 4
  square_side ^ 2 = 36 := by sorry

end NUMINAMATH_CALUDE_square_area_equal_perimeter_triangle_l2476_247663


namespace NUMINAMATH_CALUDE_inequality_solution_l2476_247674

theorem inequality_solution (x y : ℝ) :
  x + y^2 + Real.sqrt (x - y^2 - 1) ≤ 1 ∧
  x - y^2 - 1 ≥ 0 →
  x = 1 ∧ y = 0 := by
sorry

end NUMINAMATH_CALUDE_inequality_solution_l2476_247674


namespace NUMINAMATH_CALUDE_plane_equation_proof_l2476_247605

/-- A plane in 3D space represented by its equation coefficients -/
structure Plane where
  A : ℤ
  B : ℤ
  C : ℤ
  D : ℤ
  A_pos : A > 0
  coeff_coprime : Nat.gcd (Nat.gcd (Int.natAbs A) (Int.natAbs B)) (Nat.gcd (Int.natAbs C) (Int.natAbs D)) = 1

/-- A point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Check if a point lies on a plane -/
def point_on_plane (p : Point3D) (plane : Plane) : Prop :=
  plane.A * p.x + plane.B * p.y + plane.C * p.z + plane.D = 0

/-- Check if two planes are parallel -/
def planes_parallel (p1 p2 : Plane) : Prop :=
  ∃ (k : ℚ), k ≠ 0 ∧ p1.A = k * p2.A ∧ p1.B = k * p2.B ∧ p1.C = k * p2.C

theorem plane_equation_proof (given_plane : Plane) (point : Point3D) :
  given_plane.A = 4 ∧ given_plane.B = -2 ∧ given_plane.C = 6 ∧ given_plane.D = 14 →
  point.x = 2 ∧ point.y = -1 ∧ point.z = 3 →
  ∃ (result_plane : Plane),
    point_on_plane point result_plane ∧
    planes_parallel result_plane given_plane ∧
    result_plane.A = 2 ∧ result_plane.B = -1 ∧ result_plane.C = 3 ∧ result_plane.D = -14 :=
by sorry

end NUMINAMATH_CALUDE_plane_equation_proof_l2476_247605


namespace NUMINAMATH_CALUDE_f_inequality_l2476_247660

noncomputable def f (x : ℝ) : ℝ := x * Real.sin x

theorem f_inequality : f (π/3) > f 1 ∧ f 1 > f (-π/4) := by
  sorry

end NUMINAMATH_CALUDE_f_inequality_l2476_247660


namespace NUMINAMATH_CALUDE_pond_length_l2476_247626

/-- Given a rectangular pond with width 10 m, depth 5 m, and volume of extracted soil 1000 cubic meters, the length of the pond is 20 m. -/
theorem pond_length (width : ℝ) (depth : ℝ) (volume : ℝ) (length : ℝ) : 
  width = 10 → depth = 5 → volume = 1000 → volume = length * width * depth → length = 20 := by
  sorry

end NUMINAMATH_CALUDE_pond_length_l2476_247626


namespace NUMINAMATH_CALUDE_triangle_example_1_triangle_example_2_l2476_247619

-- Define the new operation ▲
def triangle (m n : ℤ) : ℤ := m - n + m * n

-- Theorem statements
theorem triangle_example_1 : triangle 3 (-4) = -5 := by sorry

theorem triangle_example_2 : triangle (-6) (triangle 2 (-3)) = 1 := by sorry

end NUMINAMATH_CALUDE_triangle_example_1_triangle_example_2_l2476_247619


namespace NUMINAMATH_CALUDE_hiker_catchup_time_l2476_247685

/-- Proves that a hiker catches up to a motorcyclist in 48 minutes under given conditions -/
theorem hiker_catchup_time (hiker_speed : ℝ) (motorcyclist_speed : ℝ) (stop_time : ℝ) : 
  hiker_speed = 6 →
  motorcyclist_speed = 30 →
  stop_time = 12 / 60 →
  (motorcyclist_speed * stop_time - hiker_speed * stop_time) / hiker_speed * 60 = 48 := by
sorry

end NUMINAMATH_CALUDE_hiker_catchup_time_l2476_247685


namespace NUMINAMATH_CALUDE_investment_interest_rate_l2476_247637

theorem investment_interest_rate (total_investment : ℝ) (first_part : ℝ) (first_rate : ℝ) (total_interest : ℝ) : 
  total_investment = 3600 →
  first_part = 1800 →
  first_rate = 3 →
  total_interest = 144 →
  (first_part * first_rate / 100 + (total_investment - first_part) * 5 / 100 = total_interest) :=
by sorry

end NUMINAMATH_CALUDE_investment_interest_rate_l2476_247637


namespace NUMINAMATH_CALUDE_age_difference_proof_l2476_247697

theorem age_difference_proof (son_age : ℕ) (man_age : ℕ) : son_age = 26 →
  man_age + 2 = 2 * (son_age + 2) →
  man_age - son_age = 28 := by
sorry

end NUMINAMATH_CALUDE_age_difference_proof_l2476_247697


namespace NUMINAMATH_CALUDE_floyd_books_theorem_l2476_247699

def total_books : ℕ := 89
def mcgregor_books : ℕ := 34
def unread_books : ℕ := 23

theorem floyd_books_theorem : 
  total_books - mcgregor_books - unread_books = 32 := by
  sorry

end NUMINAMATH_CALUDE_floyd_books_theorem_l2476_247699


namespace NUMINAMATH_CALUDE_seashells_given_correct_l2476_247662

/-- Calculates the number of seashells given away -/
def seashells_given (initial_seashells current_seashells : ℕ) : ℕ :=
  initial_seashells - current_seashells

/-- Proves that the number of seashells given away is correct -/
theorem seashells_given_correct (initial_seashells current_seashells : ℕ) 
  (h : initial_seashells ≥ current_seashells) :
  seashells_given initial_seashells current_seashells = initial_seashells - current_seashells :=
by
  sorry

#eval seashells_given 49 36

end NUMINAMATH_CALUDE_seashells_given_correct_l2476_247662


namespace NUMINAMATH_CALUDE_trig_inequality_l2476_247628

theorem trig_inequality (θ : Real) (h : π < θ ∧ θ < 5 * π / 4) :
  Real.cos θ < Real.sin θ ∧ Real.sin θ < Real.tan θ := by
  sorry

end NUMINAMATH_CALUDE_trig_inequality_l2476_247628


namespace NUMINAMATH_CALUDE_third_month_sale_is_10389_l2476_247644

/-- Calculates the sale in the third month given the sales for other months and the average -/
def third_month_sale (sale1 sale2 sale4 sale5 sale6 average : ℕ) : ℕ :=
  6 * average - (sale1 + sale2 + sale4 + sale5 + sale6)

/-- Proves that the sale in the third month is 10389 given the conditions -/
theorem third_month_sale_is_10389 :
  third_month_sale 4000 6524 7230 6000 12557 7000 = 10389 := by
  sorry

end NUMINAMATH_CALUDE_third_month_sale_is_10389_l2476_247644


namespace NUMINAMATH_CALUDE_days_to_complete_paper_l2476_247659

-- Define the paper length and writing rate
def paper_length : ℕ := 63
def pages_per_day : ℕ := 21

-- Theorem statement
theorem days_to_complete_paper : 
  (paper_length / pages_per_day : ℕ) = 3 := by
  sorry

end NUMINAMATH_CALUDE_days_to_complete_paper_l2476_247659


namespace NUMINAMATH_CALUDE_intersection_points_of_cubic_l2476_247672

-- Define the function f(x) = x^3 - 3x
def f (x : ℝ) : ℝ := x^3 - 3*x

-- State the theorem
theorem intersection_points_of_cubic (c : ℝ) :
  (∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ f x₁ + c = 0 ∧ f x₂ + c = 0 ∧
    ∀ x, f x + c = 0 → x = x₁ ∨ x = x₂) ↔ c = -2 ∨ c = 2 :=
sorry

end NUMINAMATH_CALUDE_intersection_points_of_cubic_l2476_247672


namespace NUMINAMATH_CALUDE_square_diagonal_ratio_l2476_247613

theorem square_diagonal_ratio (a b : ℝ) (h : a > 0) (k : b > 0) :
  (4 * a) / (4 * b) = 3 / 2 → (a * Real.sqrt 2) / (b * Real.sqrt 2) = 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_square_diagonal_ratio_l2476_247613


namespace NUMINAMATH_CALUDE_olympiad_problem_distribution_l2476_247600

theorem olympiad_problem_distribution (n : ℕ) (m : ℕ) (k : ℕ) 
  (h1 : n = 30) 
  (h2 : m = 40) 
  (h3 : k = 5) 
  (h4 : ∃ (x y z q r : ℕ), 
    x + y + z + q + r = n ∧ 
    x + 2*y + 3*z + 4*q + 5*r = m ∧ 
    x > 0 ∧ y > 0 ∧ z > 0 ∧ q > 0 ∧ r > 0) :
  ∃ (x : ℕ), x = 26 ∧ 
    ∃ (y z q r : ℕ), 
      x + y + z + q + r = n ∧ 
      x + 2*y + 3*z + 4*q + 5*r = m ∧
      y = 1 ∧ z = 1 ∧ q = 1 ∧ r = 1 := by
  sorry

end NUMINAMATH_CALUDE_olympiad_problem_distribution_l2476_247600


namespace NUMINAMATH_CALUDE_star_diameter_scientific_notation_l2476_247689

/-- Represents the diameter of the star in meters -/
def star_diameter : ℝ := 16600000000

/-- Represents the coefficient in scientific notation -/
def coefficient : ℝ := 1.66

/-- Represents the exponent in scientific notation -/
def exponent : ℕ := 10

/-- Theorem stating that the star's diameter is correctly expressed in scientific notation -/
theorem star_diameter_scientific_notation : 
  star_diameter = coefficient * (10 : ℝ) ^ exponent := by
  sorry

end NUMINAMATH_CALUDE_star_diameter_scientific_notation_l2476_247689


namespace NUMINAMATH_CALUDE_equation_solution_l2476_247625

theorem equation_solution : 
  ∃! x : ℚ, (x^2 + 3*x + 4) / (x + 5) = x + 6 :=
by
  use -13/4
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2476_247625


namespace NUMINAMATH_CALUDE_workshop_workers_l2476_247694

/-- The total number of workers in a workshop with given salary conditions -/
theorem workshop_workers (average_salary : ℕ) (technician_count : ℕ) (technician_salary : ℕ) (non_technician_salary : ℕ) 
  (h1 : average_salary = 8000)
  (h2 : technician_count = 7)
  (h3 : technician_salary = 10000)
  (h4 : non_technician_salary = 6000) :
  ∃ (total_workers : ℕ), 
    total_workers * average_salary = 
      technician_count * technician_salary + (total_workers - technician_count) * non_technician_salary ∧
    total_workers = 14 := by
  sorry

end NUMINAMATH_CALUDE_workshop_workers_l2476_247694


namespace NUMINAMATH_CALUDE_ratio_equality_l2476_247655

theorem ratio_equality (x y z w : ℝ) 
  (h : (x - y) * (z - w) / ((y - z) * (w - x)) = 3 / 7) : 
  (x - z) * (y - w) / ((x - y) * (z - w)) = -4 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ratio_equality_l2476_247655


namespace NUMINAMATH_CALUDE_tv_discount_percentage_l2476_247639

def original_price : ℚ := 480
def first_installment : ℚ := 150
def num_monthly_installments : ℕ := 3
def monthly_installment : ℚ := 102

def total_payment : ℚ := first_installment + (monthly_installment * num_monthly_installments)
def discount : ℚ := original_price - total_payment
def discount_percentage : ℚ := (discount / original_price) * 100

theorem tv_discount_percentage :
  discount_percentage = 5 := by sorry

end NUMINAMATH_CALUDE_tv_discount_percentage_l2476_247639


namespace NUMINAMATH_CALUDE_fraction_sum_squared_l2476_247665

theorem fraction_sum_squared : 
  (2/10 + 3/100 + 5/1000 + 7/10000)^2 = 0.05555649 := by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_squared_l2476_247665


namespace NUMINAMATH_CALUDE_x_value_l2476_247670

def A : Set ℝ := {1, 2, 3}
def B (x : ℝ) : Set ℝ := {1, x}

theorem x_value (x : ℝ) : A ∪ B x = A → x = 2 ∨ x = 3 := by
  sorry

end NUMINAMATH_CALUDE_x_value_l2476_247670


namespace NUMINAMATH_CALUDE_median_equal_mean_l2476_247642

def set_elements (n : ℝ) : List ℝ := [n, n+4, n+7, n+10, n+14]

theorem median_equal_mean (n : ℝ) (h : n + 7 = 14) : 
  (List.sum (set_elements n)) / (List.length (set_elements n)) = 14 := by
  sorry

end NUMINAMATH_CALUDE_median_equal_mean_l2476_247642


namespace NUMINAMATH_CALUDE_sunflower_seed_distribution_l2476_247671

theorem sunflower_seed_distribution (total_seeds : ℕ) (num_cans : ℕ) (seeds_per_can : ℕ) :
  total_seeds = 54 →
  num_cans = 9 →
  total_seeds = num_cans * seeds_per_can →
  seeds_per_can = 6 :=
by
  sorry

end NUMINAMATH_CALUDE_sunflower_seed_distribution_l2476_247671


namespace NUMINAMATH_CALUDE_geometric_sequence_statements_l2476_247684

def geometricSequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = q * a n

def increasingSequence (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, a n ≤ a (n + 1)

theorem geometric_sequence_statements (a : ℕ → ℝ) (q : ℝ) 
  (h : geometricSequence a q) : 
  (¬(q > 1 → increasingSequence a) ∧
   ¬(increasingSequence a → q > 1) ∧
   ¬(q ≤ 1 → ¬increasingSequence a) ∧
   ¬(¬increasingSequence a → q ≤ 1)) := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_statements_l2476_247684


namespace NUMINAMATH_CALUDE_eliza_ironing_time_l2476_247680

theorem eliza_ironing_time :
  ∀ (blouse_time : ℝ),
    (blouse_time > 0) →
    (120 / blouse_time + 180 / 20 = 17) →
    blouse_time = 15 := by
  sorry

end NUMINAMATH_CALUDE_eliza_ironing_time_l2476_247680


namespace NUMINAMATH_CALUDE_proposition_truth_l2476_247648

theorem proposition_truth (x y : ℝ) : x + y ≥ 5 → x ≥ 3 ∨ y ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_proposition_truth_l2476_247648


namespace NUMINAMATH_CALUDE_expand_product_l2476_247681

theorem expand_product (x : ℝ) : 2 * (x + 3) * (x + 6) = 2 * x^2 + 18 * x + 36 := by
  sorry

end NUMINAMATH_CALUDE_expand_product_l2476_247681


namespace NUMINAMATH_CALUDE_balloons_in_park_l2476_247610

/-- The number of balloons Allan and Jake had in the park -/
def total_balloons (allan_initial : ℕ) (jake_balloons : ℕ) (allan_bought : ℕ) : ℕ :=
  (allan_initial + allan_bought) + jake_balloons

/-- Theorem stating the total number of balloons Allan and Jake had in the park -/
theorem balloons_in_park : total_balloons 3 5 2 = 10 := by
  sorry

end NUMINAMATH_CALUDE_balloons_in_park_l2476_247610


namespace NUMINAMATH_CALUDE_max_performances_l2476_247652

theorem max_performances (n : ℕ) : 
  (∃ (performances : Fin n → Finset (Fin 12)),
    (∀ i : Fin n, (performances i).card = 6) ∧ 
    (∀ i j : Fin n, i ≠ j → (performances i ∩ performances j).card ≤ 2)) →
  n ≤ 4 :=
by sorry

end NUMINAMATH_CALUDE_max_performances_l2476_247652


namespace NUMINAMATH_CALUDE_exists_same_color_rectangle_l2476_247608

/-- A color represented as an enumeration -/
inductive Color
  | Red
  | Green
  | Blue

/-- A grid coloring is a function from grid coordinates to colors -/
def GridColoring := (Fin 4 × Fin 82) → Color

/-- A rectangle is represented by four points in the grid -/
structure Rectangle :=
  (p1 p2 p3 p4 : Fin 4 × Fin 82)

/-- Predicate to check if all vertices of a rectangle have the same color -/
def SameColorRectangle (coloring : GridColoring) (rect : Rectangle) : Prop :=
  coloring rect.p1 = coloring rect.p2 ∧
  coloring rect.p1 = coloring rect.p3 ∧
  coloring rect.p1 = coloring rect.p4

/-- Main theorem: There exists a rectangle with vertices of the same color in any 4x82 grid coloring -/
theorem exists_same_color_rectangle (coloring : GridColoring) :
  ∃ (rect : Rectangle), SameColorRectangle coloring rect := by
  sorry


end NUMINAMATH_CALUDE_exists_same_color_rectangle_l2476_247608


namespace NUMINAMATH_CALUDE_kennel_dogs_l2476_247603

theorem kennel_dogs (cats dogs : ℕ) : 
  (cats : ℚ) / dogs = 3 / 4 →
  cats = dogs - 8 →
  dogs = 32 := by
sorry

end NUMINAMATH_CALUDE_kennel_dogs_l2476_247603


namespace NUMINAMATH_CALUDE_cubic_one_real_root_l2476_247677

theorem cubic_one_real_root (a b : ℝ) : 
  (∃! x : ℝ, x^3 - a*x + b = 0) ↔ 
  ((a = 0 ∧ b = 2) ∨ (a = -3 ∧ b = 2) ∨ (a = 3 ∧ b = -3)) :=
sorry

end NUMINAMATH_CALUDE_cubic_one_real_root_l2476_247677


namespace NUMINAMATH_CALUDE_veridux_female_managers_l2476_247635

/-- Calculates the number of female managers given the total number of employees,
    female employees, total managers, and male associates. -/
def female_managers (total_employees : ℕ) (female_employees : ℕ) (total_managers : ℕ) (male_associates : ℕ) : ℕ :=
  total_managers - (total_employees - female_employees - male_associates)

/-- Theorem stating that given the conditions from the problem, 
    the number of female managers is 40. -/
theorem veridux_female_managers :
  female_managers 250 90 40 160 = 40 := by
  sorry

#eval female_managers 250 90 40 160

end NUMINAMATH_CALUDE_veridux_female_managers_l2476_247635


namespace NUMINAMATH_CALUDE_distance_formula_l2476_247678

/-- The distance between two points on a real number line -/
def distance (a b : ℝ) : ℝ := |b - a|

/-- Theorem: The distance between two points A and B with coordinates a and b is |b - a| -/
theorem distance_formula (a b : ℝ) : distance a b = |b - a| := by sorry

end NUMINAMATH_CALUDE_distance_formula_l2476_247678


namespace NUMINAMATH_CALUDE_netGainDifference_l2476_247658

/-- Represents a job candidate with their associated costs and revenue --/
structure Candidate where
  salary : ℕ
  revenue : ℕ
  trainingMonths : ℕ
  trainingCostPerMonth : ℕ
  hiringBonusPercent : ℕ

/-- Calculates the net gain for a candidate --/
def netGain (c : Candidate) : ℕ :=
  c.revenue - c.salary - (c.trainingMonths * c.trainingCostPerMonth) - (c.salary * c.hiringBonusPercent / 100)

/-- The two candidates as described in the problem --/
def candidate1 : Candidate :=
  { salary := 42000
    revenue := 93000
    trainingMonths := 3
    trainingCostPerMonth := 1200
    hiringBonusPercent := 0 }

def candidate2 : Candidate :=
  { salary := 45000
    revenue := 92000
    trainingMonths := 0
    trainingCostPerMonth := 0
    hiringBonusPercent := 1 }

/-- Theorem stating the difference in net gain between the two candidates --/
theorem netGainDifference : netGain candidate1 - netGain candidate2 = 850 := by
  sorry

end NUMINAMATH_CALUDE_netGainDifference_l2476_247658


namespace NUMINAMATH_CALUDE_third_term_coefficient_binomial_expansion_l2476_247646

theorem third_term_coefficient_binomial_expansion :
  let a := x
  let b := -1 / (2 * x)
  let n := 6
  let k := 2  -- Third term corresponds to k = 2
  (Nat.choose n k : ℚ) * a^(n - k) * b^k = 15/4 := by
  sorry

end NUMINAMATH_CALUDE_third_term_coefficient_binomial_expansion_l2476_247646


namespace NUMINAMATH_CALUDE_square_sum_zero_implies_both_zero_l2476_247649

theorem square_sum_zero_implies_both_zero (a b : ℝ) (h : a^2 + b^2 = 0) : a = 0 ∧ b = 0 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_zero_implies_both_zero_l2476_247649


namespace NUMINAMATH_CALUDE_train_length_l2476_247623

/-- Given a train that crosses an electric pole in 2.5 seconds at a speed of 144 km/hr,
    prove that its length is 100 meters. -/
theorem train_length (crossing_time : Real) (speed_kmh : Real) (length : Real) : 
  crossing_time = 2.5 →
  speed_kmh = 144 →
  length = speed_kmh * (1000 / 3600) * crossing_time →
  length = 100 := by
  sorry

#check train_length

end NUMINAMATH_CALUDE_train_length_l2476_247623


namespace NUMINAMATH_CALUDE_tea_party_wait_time_l2476_247647

/-- Mad Hatter's clock speed relative to real time -/
def mad_hatter_clock_speed : ℚ := 5/4

/-- March Hare's clock speed relative to real time -/
def march_hare_clock_speed : ℚ := 5/6

/-- The agreed meeting time on their clocks (in hours after noon) -/
def meeting_time : ℚ := 5

/-- Calculate the real time when someone arrives based on their clock speed -/
def real_arrival_time (clock_speed : ℚ) : ℚ :=
  meeting_time / clock_speed

theorem tea_party_wait_time :
  real_arrival_time march_hare_clock_speed - real_arrival_time mad_hatter_clock_speed = 2 := by
  sorry

end NUMINAMATH_CALUDE_tea_party_wait_time_l2476_247647


namespace NUMINAMATH_CALUDE_cinematic_academy_members_l2476_247675

/-- The minimum fraction of top-10 lists a film must appear on to be considered for "movie of the year" -/
def min_fraction : ℚ := 1 / 4

/-- The smallest number of top-10 lists a film can appear on and still be considered -/
def min_lists : ℚ := 198.75

/-- The number of members in the Cinematic Academy -/
def academy_members : ℕ := 795

/-- Theorem stating that the number of members in the Cinematic Academy is 795 -/
theorem cinematic_academy_members :
  academy_members = ⌈(min_lists / min_fraction : ℚ)⌉ := by
  sorry

end NUMINAMATH_CALUDE_cinematic_academy_members_l2476_247675


namespace NUMINAMATH_CALUDE_collinear_vectors_t_value_l2476_247666

variable {V : Type*} [AddCommGroup V] [Module ℝ V]
variable (a b : V)

theorem collinear_vectors_t_value 
  (h_non_collinear : ¬ ∃ (k : ℝ), a = k • b) 
  (h_collinear : ∃ (k : ℝ), a - t • b = k • (2 • a + b)) : 
  t = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_collinear_vectors_t_value_l2476_247666


namespace NUMINAMATH_CALUDE_triangle_side_length_l2476_247643

/-- Given a triangle ABC with side lengths a, b, c opposite to angles A, B, C respectively.
    Prove that if c = 10, A = 45°, and C = 30°, then b = 5(√6 + √2). -/
theorem triangle_side_length (a b c : ℝ) (A B C : ℝ) : 
  c = 10 → A = π/4 → C = π/6 → b = 5 * (Real.sqrt 6 + Real.sqrt 2) :=
by sorry

end NUMINAMATH_CALUDE_triangle_side_length_l2476_247643


namespace NUMINAMATH_CALUDE_two_digit_number_property_l2476_247687

/-- Represents a two-digit number -/
structure TwoDigitNumber where
  tens : Nat
  units : Nat
  tens_valid : 1 ≤ tens ∧ tens ≤ 9
  units_valid : units ≤ 9

/-- The value of a two-digit number -/
def value (n : TwoDigitNumber) : Nat :=
  10 * n.tens + n.units

/-- The reverse of a two-digit number -/
def reverse (n : TwoDigitNumber) : Nat :=
  10 * n.units + n.tens

/-- The sum of digits of a two-digit number -/
def digitSum (n : TwoDigitNumber) : Nat :=
  n.tens + n.units

theorem two_digit_number_property (n : TwoDigitNumber) :
  value n - reverse n = 7 * digitSum n →
  value n + reverse n = 99 := by
  sorry

end NUMINAMATH_CALUDE_two_digit_number_property_l2476_247687


namespace NUMINAMATH_CALUDE_trigonometric_equation_solution_l2476_247653

theorem trigonometric_equation_solution (x : ℝ) : 
  (Real.cos (x/2) * Real.cos (3*x/2) - Real.sin x * Real.sin (3*x) - Real.sin (2*x) * Real.sin (3*x) = 0) →
  ∃ k : ℤ, x = π/9 * (2*k + 1) := by
sorry

end NUMINAMATH_CALUDE_trigonometric_equation_solution_l2476_247653


namespace NUMINAMATH_CALUDE_missing_digits_sum_l2476_247607

/-- Represents a single digit (0-9) -/
def Digit := Fin 10

/-- The addition problem structure -/
structure AdditionProblem where
  d1 : Digit  -- First missing digit
  d2 : Digit  -- Second missing digit

/-- The addition problem is valid -/
def isValidAddition (p : AdditionProblem) : Prop :=
  708 + 10 * p.d1.val + 2182 = 86301 + 100 * p.d2.val

/-- The theorem to be proved -/
theorem missing_digits_sum (p : AdditionProblem) 
  (h : isValidAddition p) : p.d1.val + p.d2.val = 7 := by
  sorry

#check missing_digits_sum

end NUMINAMATH_CALUDE_missing_digits_sum_l2476_247607


namespace NUMINAMATH_CALUDE_subtraction_of_decimals_l2476_247650

theorem subtraction_of_decimals : (3.156 : ℝ) - (1.029 : ℝ) = 2.127 := by sorry

end NUMINAMATH_CALUDE_subtraction_of_decimals_l2476_247650


namespace NUMINAMATH_CALUDE_roses_handed_out_l2476_247638

theorem roses_handed_out (total : ℕ) (left : ℕ) (handed_out : ℕ) : 
  total = 29 → left = 12 → handed_out = total - left → handed_out = 17 := by
  sorry

end NUMINAMATH_CALUDE_roses_handed_out_l2476_247638


namespace NUMINAMATH_CALUDE_fraction_subtraction_proof_l2476_247654

theorem fraction_subtraction_proof : 
  (4 : ℚ) / 5 - (1 : ℚ) / 5 = (3 : ℚ) / 5 := by sorry

end NUMINAMATH_CALUDE_fraction_subtraction_proof_l2476_247654


namespace NUMINAMATH_CALUDE_statements_correctness_l2476_247679

theorem statements_correctness :
  (∃ a b : ℝ, a > b ∧ 1/a > 1/b ∧ a*b ≤ 0) ∧
  (∀ a b c : ℝ, a > b ∧ b > 0 ∧ c < 0 → c/a > c/b) ∧
  (∃ a b : ℝ, a < b ∧ b < 0 ∧ a^2 ≥ b^2) ∧
  (∀ a b c d : ℝ, a > b ∧ b > 0 ∧ c < d ∧ d < 0 → a*c < b*d) :=
by sorry

end NUMINAMATH_CALUDE_statements_correctness_l2476_247679
