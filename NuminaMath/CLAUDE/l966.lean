import Mathlib

namespace cryptarithm_solution_l966_96610

def is_valid_cryptarithm (n : ℕ) : Prop :=
  10 ≤ n ∧ n < 100 ∧ 
  ∃ (a b c d : ℕ), n * n = 1000 * c + 100 * d + n ∧ 
  c ≠ 0

theorem cryptarithm_solution :
  ∃! (S : Set ℕ), 
    (∀ n ∈ S, is_valid_cryptarithm n ∧ Odd n) ∧ 
    (∀ n, is_valid_cryptarithm n → Odd n → n ∈ S) ∧
    (∃ m, m ∈ S) ∧
    (∀ T : Set ℕ, (∀ n ∈ T, is_valid_cryptarithm n ∧ Even n) → T.Nonempty → ¬(∃! x, x ∈ T)) :=
by sorry

end cryptarithm_solution_l966_96610


namespace group_size_l966_96674

theorem group_size (average_increase : ℝ) (old_weight : ℝ) (new_weight : ℝ) :
  average_increase = 3 ∧ old_weight = 65 ∧ new_weight = 89 →
  (new_weight - old_weight) / average_increase = 8 := by
  sorry

end group_size_l966_96674


namespace midpoint_property_implies_linear_l966_96660

/-- A function satisfying the midpoint property -/
def HasMidpointProperty (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f ((x + y) / 2) = (f x + f y) / 2

/-- The main theorem: continuous functions with the midpoint property are linear -/
theorem midpoint_property_implies_linear
  (f : ℝ → ℝ) (hf : Continuous f) (hm : HasMidpointProperty f) :
  ∃ a b : ℝ, ∀ x : ℝ, f x = a * x + b :=
sorry

end midpoint_property_implies_linear_l966_96660


namespace fibonacci_sum_l966_96615

/-- Fibonacci sequence -/
def fib : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | (n + 2) => fib (n + 1) + fib n

/-- Sum of F_n / 10^n from n = 0 to infinity -/
noncomputable def fibSum : ℝ := ∑' n, (fib n : ℝ) / (10 : ℝ) ^ n

/-- Theorem: The sum of F_n / 10^n from n = 0 to infinity equals 10/89 -/
theorem fibonacci_sum : fibSum = 10 / 89 := by sorry

end fibonacci_sum_l966_96615


namespace complex_modulus_product_range_l966_96692

theorem complex_modulus_product_range (z₁ z₂ : ℂ) 
  (h₁ : Complex.abs (z₁ + z₂) = 4)
  (h₂ : Complex.abs (z₁ - z₂) = 3) :
  7/4 ≤ Complex.abs (z₁ * z₂) ∧ Complex.abs (z₁ * z₂) ≤ 25/4 :=
by sorry

end complex_modulus_product_range_l966_96692


namespace solution_set_equals_interval_l966_96619

-- Define the ⊕ operation
def circle_plus (a b : ℝ) : ℝ := a * b + 2 * a + b

-- Define the set of x satisfying the inequality
def solution_set : Set ℝ := {x | circle_plus x (x - 2) < 0}

-- Theorem statement
theorem solution_set_equals_interval :
  solution_set = Set.Ioo (-2) 1 := by sorry

end solution_set_equals_interval_l966_96619


namespace scientific_notation_43000000_l966_96682

theorem scientific_notation_43000000 :
  (43000000 : ℝ) = 4.3 * (10 : ℝ)^7 := by
  sorry

end scientific_notation_43000000_l966_96682


namespace cat_finishes_food_on_day_l966_96648

/-- Represents the days of the week -/
inductive Day : Type
| monday : Day
| tuesday : Day
| wednesday : Day
| thursday : Day
| friday : Day
| saturday : Day
| sunday : Day

/-- Calculates the number of days since Monday -/
def daysSinceMonday (d : Day) : ℕ :=
  match d with
  | Day.monday => 0
  | Day.tuesday => 1
  | Day.wednesday => 2
  | Day.thursday => 3
  | Day.friday => 4
  | Day.saturday => 5
  | Day.sunday => 6

/-- The amount of food the cat eats in the morning (in cans) -/
def morningMeal : ℚ := 2/5

/-- The amount of food the cat eats in the evening (in cans) -/
def eveningMeal : ℚ := 1/6

/-- The total number of cans in the box -/
def totalCans : ℕ := 10

/-- The day on which the cat finishes all the food -/
def finishDay : Day := Day.saturday

/-- Theorem stating that the cat finishes all the food on the specified day -/
theorem cat_finishes_food_on_day :
  (morningMeal + eveningMeal) * (daysSinceMonday finishDay + 1 : ℚ) > totalCans ∧
  (morningMeal + eveningMeal) * (daysSinceMonday finishDay : ℚ) ≤ totalCans :=
by sorry


end cat_finishes_food_on_day_l966_96648


namespace transport_cost_calculation_l966_96626

/-- The transport cost for Ramesh's refrigerator purchase --/
def transport_cost : ℕ := by sorry

/-- The labelled price of the refrigerator before discount --/
def labelled_price : ℕ := by sorry

/-- The discounted price Ramesh paid for the refrigerator --/
def discounted_price : ℕ := 17500

/-- The installation cost --/
def installation_cost : ℕ := 250

/-- The selling price to earn 10% profit without discount --/
def selling_price : ℕ := 24475

/-- The discount rate applied to the labelled price --/
def discount_rate : ℚ := 1/5

/-- The profit rate desired if no discount was offered --/
def profit_rate : ℚ := 1/10

theorem transport_cost_calculation :
  discounted_price = labelled_price * (1 - discount_rate) ∧
  selling_price = labelled_price * (1 + profit_rate) ∧
  transport_cost + discounted_price + installation_cost = selling_price ∧
  transport_cost = 6725 := by sorry

end transport_cost_calculation_l966_96626


namespace max_grain_mass_l966_96636

/-- The maximum mass of grain that can be loaded onto a rectangular platform -/
theorem max_grain_mass (length width : Real) (max_angle : Real) (density : Real) :
  length = 10 ∧ 
  width = 5 ∧ 
  max_angle = π / 4 ∧ 
  density = 1200 →
  ∃ (mass : Real),
    mass = 175000 ∧ 
    mass = density * (length * width * (width / 2) / 2 + length * width * (width / 4))
    := by sorry

end max_grain_mass_l966_96636


namespace bus_related_time_trip_time_breakdown_l966_96684

/-- Represents the duration of Luke's trip to London in minutes -/
def total_trip_time : ℕ := 525

/-- Represents the wait time for the first bus in minutes -/
def first_bus_wait : ℕ := 25

/-- Represents the duration of the first bus ride in minutes -/
def first_bus_ride : ℕ := 40

/-- Represents the wait time for the second bus in minutes -/
def second_bus_wait : ℕ := 15

/-- Represents the duration of the second bus ride in minutes -/
def second_bus_ride : ℕ := 10

/-- Represents the walk time to the train station in minutes -/
def walk_time : ℕ := 15

/-- Represents the wait time for the train in minutes -/
def train_wait : ℕ := 2 * walk_time

/-- Represents the duration of the train ride in minutes -/
def train_ride : ℕ := 360

/-- Proves that the total bus-related time is 90 minutes -/
theorem bus_related_time :
  first_bus_wait + first_bus_ride + second_bus_wait + second_bus_ride = 90 :=
by sorry

/-- Proves that the sum of all components equals the total trip time -/
theorem trip_time_breakdown :
  first_bus_wait + first_bus_ride + second_bus_wait + second_bus_ride +
  walk_time + train_wait + train_ride = total_trip_time :=
by sorry

end bus_related_time_trip_time_breakdown_l966_96684


namespace smallest_share_is_five_l966_96680

/-- Represents the distribution of coins among three children --/
structure CoinDistribution where
  one_franc : ℕ
  five_franc : ℕ
  fifty_cent : ℕ

/-- Checks if the distribution satisfies the problem conditions --/
def valid_distribution (d : CoinDistribution) : Prop :=
  d.one_franc + 5 * d.five_franc + (d.fifty_cent : ℚ) / 2 = 100 ∧
  d.fifty_cent = d.one_franc / 9

/-- Calculates the smallest share among the three children --/
def smallest_share (d : CoinDistribution) : ℚ :=
  min (min (d.one_franc : ℚ) (5 * d.five_franc : ℚ)) ((d.fifty_cent : ℚ) / 2)

/-- Theorem stating the smallest possible share is 5 francs --/
theorem smallest_share_is_five :
  ∀ d : CoinDistribution, valid_distribution d → smallest_share d = 5 :=
sorry

end smallest_share_is_five_l966_96680


namespace bennys_working_hours_l966_96656

/-- Calculates the total working hours given hours per day and number of days worked -/
def totalWorkingHours (hoursPerDay : ℕ) (daysWorked : ℕ) : ℕ :=
  hoursPerDay * daysWorked

/-- Proves that Benny's total working hours is 18 given the conditions -/
theorem bennys_working_hours :
  let hoursPerDay : ℕ := 3
  let daysWorked : ℕ := 6
  totalWorkingHours hoursPerDay daysWorked = 18 := by
  sorry

end bennys_working_hours_l966_96656


namespace equilateral_triangle_perimeter_l966_96638

/-- 
Given an equilateral triangle with area 100√3 cm², 
prove that its perimeter is 60 cm.
-/
theorem equilateral_triangle_perimeter (A : ℝ) (p : ℝ) : 
  A = 100 * Real.sqrt 3 → p = 60 := by
  sorry

end equilateral_triangle_perimeter_l966_96638


namespace stating_systematic_sampling_theorem_l966_96691

/-- Represents a systematic sampling scheme -/
structure SystematicSampling where
  population_size : ℕ
  sample_size : ℕ
  first_sample : ℕ

/-- 
  Given a systematic sampling scheme and a group number,
  returns the number drawn from that group
-/
def number_in_group (s : SystematicSampling) (group : ℕ) : ℕ :=
  s.first_sample + (group - 1) * (s.population_size / s.sample_size)

/-- 
  Theorem stating that if the number drawn from the 13th group is 101
  in a systematic sampling of 20 from 160, then the number drawn from
  the 3rd group is 21
-/
theorem systematic_sampling_theorem :
  ∀ (s : SystematicSampling),
    s.population_size = 160 →
    s.sample_size = 20 →
    number_in_group s 13 = 101 →
    number_in_group s 3 = 21 := by
  sorry

end stating_systematic_sampling_theorem_l966_96691


namespace cos_negative_1500_degrees_l966_96679

theorem cos_negative_1500_degrees : Real.cos ((-1500 : ℝ) * π / 180) = 1 / 2 := by
  sorry

end cos_negative_1500_degrees_l966_96679


namespace rats_to_chihuahuas_ratio_l966_96625

theorem rats_to_chihuahuas_ratio : 
  ∀ (total : ℕ) (rats : ℕ) (chihuahuas : ℕ),
  total = 70 →
  rats = 60 →
  chihuahuas = total - rats →
  ∃ (k : ℕ), rats = k * chihuahuas →
  (rats : ℚ) / chihuahuas = 6 / 1 := by
sorry

end rats_to_chihuahuas_ratio_l966_96625


namespace jenny_max_earnings_l966_96628

def neighborhood_A_homes : ℕ := 10
def neighborhood_A_boxes_per_home : ℕ := 2
def neighborhood_B_homes : ℕ := 5
def neighborhood_B_boxes_per_home : ℕ := 5
def price_per_box : ℕ := 2

def total_boxes_A : ℕ := neighborhood_A_homes * neighborhood_A_boxes_per_home
def total_boxes_B : ℕ := neighborhood_B_homes * neighborhood_B_boxes_per_home

def max_earnings : ℕ := max total_boxes_A total_boxes_B * price_per_box

theorem jenny_max_earnings :
  max_earnings = 50 := by
  sorry

end jenny_max_earnings_l966_96628


namespace stratified_sampling_most_reasonable_l966_96613

/-- Represents the different grade levels in the study -/
inductive GradeLevel
  | Three
  | Six
  | Nine

/-- Represents different sampling methods -/
inductive SamplingMethod
  | LotDrawing
  | Systematic
  | Stratified
  | RandomNumber

/-- Represents the study of visual acuity across different grade levels -/
structure VisualAcuityStudy where
  gradeLevels : List GradeLevel
  sampleProportion : ℝ
  samplingMethod : SamplingMethod

/-- Checks if a sampling method is the most reasonable for a given study -/
def isMostReasonable (study : VisualAcuityStudy) (method : SamplingMethod) : Prop :=
  method = study.samplingMethod ∧
  ∀ otherMethod : SamplingMethod, otherMethod ≠ method → 
    (study.samplingMethod = otherMethod → False)

/-- The main theorem stating that stratified sampling is the most reasonable method for the visual acuity study -/
theorem stratified_sampling_most_reasonable (study : VisualAcuityStudy) :
  study.gradeLevels = [GradeLevel.Three, GradeLevel.Six, GradeLevel.Nine] →
  0 < study.sampleProportion ∧ study.sampleProportion ≤ 1 →
  isMostReasonable study SamplingMethod.Stratified :=
sorry

end stratified_sampling_most_reasonable_l966_96613


namespace sqrt_sum_equals_eight_l966_96655

theorem sqrt_sum_equals_eight : 
  Real.sqrt (18 - 8 * Real.sqrt 2) + Real.sqrt (18 + 8 * Real.sqrt 2) = 8 := by
  sorry

end sqrt_sum_equals_eight_l966_96655


namespace fraction_comparison_l966_96618

theorem fraction_comparison (a b c d : ℝ) 
  (h1 : a / b < c / d) 
  (h2 : b > d) 
  (h3 : d > 0) : 
  (a + c) / (b + d) < (1 / 2) * (a / b + c / d) := by
  sorry

end fraction_comparison_l966_96618


namespace min_intersection_points_l966_96664

/-- Represents a circle in a plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Represents the configuration of circles -/
structure CircleConfiguration where
  n : ℕ+
  circles : Fin (4 * n) → Circle
  same_radius : ∀ i j, (circles i).radius = (circles j).radius
  no_tangent : ∀ i j, i ≠ j → (circles i).center ≠ (circles j).center ∨ 
               dist (circles i).center (circles j).center ≠ (circles i).radius + (circles j).radius
  intersect_at_least_three : ∀ i, ∃ j k l, i ≠ j ∧ i ≠ k ∧ i ≠ l ∧ j ≠ k ∧ j ≠ l ∧ k ≠ l ∧
                             dist (circles i).center (circles j).center < (circles i).radius + (circles j).radius ∧
                             dist (circles i).center (circles k).center < (circles i).radius + (circles k).radius ∧
                             dist (circles i).center (circles l).center < (circles i).radius + (circles l).radius

/-- The number of intersection points in a circle configuration -/
def num_intersection_points (config : CircleConfiguration) : ℕ :=
  sorry

/-- The main theorem: the minimum number of intersection points is 4n -/
theorem min_intersection_points (config : CircleConfiguration) :
  num_intersection_points config ≥ 4 * config.n :=
sorry

end min_intersection_points_l966_96664


namespace zero_sequence_arithmetic_not_geometric_l966_96666

-- Define the sequence
def a : ℕ → ℝ
  | _ => 0

-- Theorem statement
theorem zero_sequence_arithmetic_not_geometric :
  (∀ n m : ℕ, a (n + 1) - a n = a (m + 1) - a m) ∧ 
  ¬(∀ n m : ℕ, a n ≠ 0 → a (n + 1) / a n = a (m + 1) / a m) :=
by sorry

end zero_sequence_arithmetic_not_geometric_l966_96666


namespace parallelogram_iff_midpoints_l966_96685

-- Define the points
variable (A B C D P Q E F : ℝ × ℝ)

-- Define the conditions
def is_quadrilateral (A B C D : ℝ × ℝ) : Prop := sorry

def on_diagonal (P Q B D : ℝ × ℝ) : Prop := sorry

def point_order (B P Q D : ℝ × ℝ) : Prop := sorry

def equal_segments (B P Q D : ℝ × ℝ) : Prop := sorry

def line_intersection (A P B C E : ℝ × ℝ) : Prop := sorry

def line_intersection' (A Q C D F : ℝ × ℝ) : Prop := sorry

def is_parallelogram (A B C D : ℝ × ℝ) : Prop := sorry

def is_midpoint (E B C : ℝ × ℝ) : Prop := sorry

-- State the theorem
theorem parallelogram_iff_midpoints
  (h1 : is_quadrilateral A B C D)
  (h2 : on_diagonal P Q B D)
  (h3 : point_order B P Q D)
  (h4 : equal_segments B P Q D)
  (h5 : line_intersection A P B C E)
  (h6 : line_intersection' A Q C D F) :
  is_parallelogram A B C D ↔ (is_midpoint E B C ∧ is_midpoint F C D) :=
sorry

end parallelogram_iff_midpoints_l966_96685


namespace isabel_photo_distribution_l966_96651

/-- Given a total number of pictures and a number of albums, 
    calculate the number of pictures in each album assuming equal distribution. -/
def picturesPerAlbum (totalPictures : ℕ) (numAlbums : ℕ) : ℕ :=
  totalPictures / numAlbums

/-- Theorem stating that given 6 pictures divided into 3 albums, 
    each album contains 2 pictures. -/
theorem isabel_photo_distribution :
  let phonePhotos := 2
  let cameraPhotos := 4
  let totalPhotos := phonePhotos + cameraPhotos
  let numAlbums := 3
  picturesPerAlbum totalPhotos numAlbums = 2 := by
  sorry

end isabel_photo_distribution_l966_96651


namespace quadratic_equation_has_two_distinct_real_roots_l966_96665

/-- The quadratic equation x^2 - 3x - 1 = 0 has two distinct real roots -/
theorem quadratic_equation_has_two_distinct_real_roots :
  ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ x₁^2 - 3*x₁ - 1 = 0 ∧ x₂^2 - 3*x₂ - 1 = 0 :=
by sorry

end quadratic_equation_has_two_distinct_real_roots_l966_96665


namespace number_relationship_l966_96645

theorem number_relationship : 
  let a : ℝ := -0.3
  let b : ℝ := (0.3:ℝ)^2
  let c : ℝ := 2^(0.3:ℝ)
  b < a ∧ a < c := by sorry

end number_relationship_l966_96645


namespace gym_class_group_sizes_l966_96663

/-- Given a gym class with two groups of students, prove that if the total number of students is 71 and one group has 37 students, then the other group must have 34 students. -/
theorem gym_class_group_sizes (total_students : ℕ) (group1_size : ℕ) (group2_size : ℕ) 
  (h1 : total_students = 71)
  (h2 : group2_size = 37)
  (h3 : total_students = group1_size + group2_size) :
  group1_size = 34 := by
  sorry

end gym_class_group_sizes_l966_96663


namespace unknown_number_is_six_l966_96697

theorem unknown_number_is_six : ∃ x : ℚ, (2 / 3) * x + 6 = 10 ∧ x = 6 := by
  sorry

end unknown_number_is_six_l966_96697


namespace product_equals_eight_l966_96614

theorem product_equals_eight :
  (1 + 1/1) * (1 + 1/2) * (1 + 1/3) * (1 + 1/4) * (1 + 1/5) * (1 + 1/6) * (1 + 1/7) = 8 := by
  sorry

end product_equals_eight_l966_96614


namespace julio_bonus_l966_96647

/-- Calculates Julio's bonus given his commission rate, customer numbers, salary, and total earnings -/
def calculate_bonus (commission_rate : ℕ) (customers_week1 : ℕ) (salary : ℕ) (total_earnings : ℕ) : ℕ :=
  let customers_week2 := 2 * customers_week1
  let customers_week3 := 3 * customers_week1
  let total_customers := customers_week1 + customers_week2 + customers_week3
  let total_commission := commission_rate * total_customers
  total_earnings - salary - total_commission

/-- Theorem stating that Julio's bonus is $50 given the problem conditions -/
theorem julio_bonus :
  calculate_bonus 1 35 500 760 = 50 := by
  sorry

end julio_bonus_l966_96647


namespace y_greater_than_one_l966_96659

theorem y_greater_than_one (x y : ℝ) (h1 : x^3 > y^2) (h2 : y^3 > x^2) : y > 1 := by
  sorry

end y_greater_than_one_l966_96659


namespace geometric_sequence_general_term_l966_96670

/-- Given a geometric sequence {a_n} where a₁ = x, a₂ = x-1, and a₃ = 2x-2,
    prove that the general term is a_n = -2^(n-1) -/
theorem geometric_sequence_general_term (x : ℝ) (a : ℕ → ℝ) (h1 : a 1 = x) (h2 : a 2 = x - 1) (h3 : a 3 = 2*x - 2) :
  ∀ n : ℕ, a n = -2^(n-1) := by
sorry

end geometric_sequence_general_term_l966_96670


namespace x_equals_two_l966_96601

theorem x_equals_two (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : 6 * x^3 + 12 * x * y^2 = 3 * x^2 * y + 3 * x^4) : x = 2 := by
  sorry

end x_equals_two_l966_96601


namespace complement_union_A_B_complement_A_inter_B_l966_96678

-- Define the sets A and B
def A : Set ℝ := {x | 3 ≤ x ∧ x < 7}
def B : Set ℝ := {x | 2 < x ∧ x < 10}

-- State the theorems to be proved
theorem complement_union_A_B : 
  (Set.univ : Set ℝ) \ (A ∪ B) = {x | x ≤ 2 ∨ x ≥ 10} := by sorry

theorem complement_A_inter_B : 
  ((Set.univ : Set ℝ) \ A) ∩ B = {x | (2 < x ∧ x < 3) ∨ (7 ≤ x ∧ x < 10)} := by sorry

end complement_union_A_B_complement_A_inter_B_l966_96678


namespace andy_lateness_l966_96653

structure TravelDelay where
  normalTime : Nat
  redLights : Nat
  redLightDelay : Nat
  constructionDelay : Nat
  detourDelay : Nat
  storeDelay : Nat
  trafficDelay : Nat
  departureTime : Nat
  schoolStartTime : Nat

def calculateLateness (delay : TravelDelay) : Nat :=
  let totalDelay := delay.normalTime +
                    delay.redLights * delay.redLightDelay +
                    delay.constructionDelay +
                    delay.detourDelay +
                    delay.storeDelay +
                    delay.trafficDelay
  let arrivalTime := delay.departureTime + totalDelay
  if arrivalTime > delay.schoolStartTime then
    arrivalTime - delay.schoolStartTime
  else
    0

theorem andy_lateness (delay : TravelDelay)
  (h1 : delay.normalTime = 30)
  (h2 : delay.redLights = 4)
  (h3 : delay.redLightDelay = 3)
  (h4 : delay.constructionDelay = 10)
  (h5 : delay.detourDelay = 7)
  (h6 : delay.storeDelay = 5)
  (h7 : delay.trafficDelay = 15)
  (h8 : delay.departureTime = 435)  -- 7:15 AM in minutes since midnight
  (h9 : delay.schoolStartTime = 480)  -- 8:00 AM in minutes since midnight
  : calculateLateness delay = 34 := by
  sorry


end andy_lateness_l966_96653


namespace range_of_k_l966_96616

-- Define the equation
def equation (x y k : ℝ) : Prop := x + y - 6 * Real.sqrt (x + y) + 3 * k = 0

-- Define the condition that the equation represents only one line
def represents_one_line (k : ℝ) : Prop :=
  ∀ x y : ℝ, equation x y k → ∃! (x' y' : ℝ), equation x' y' k ∧ x' = x ∧ y' = y

-- Theorem statement
theorem range_of_k (k : ℝ) :
  represents_one_line k ↔ k = 3 ∨ k < 0 := by sorry

end range_of_k_l966_96616


namespace equation_solution_l966_96623

theorem equation_solution (x : ℝ) : 
  (x / 6) / 3 = 9 / (x / 3) → x = 9 * Real.sqrt 6 ∨ x = -9 * Real.sqrt 6 :=
by sorry

end equation_solution_l966_96623


namespace min_max_abs_quadratic_on_unit_interval_l966_96650

/-- The minimum value of the maximum absolute value of a quadratic function on [-1, 1] -/
theorem min_max_abs_quadratic_on_unit_interval :
  ∃ (F : ℝ), F = 1/2 ∧ 
  (∀ (a b : ℝ) (f : ℝ → ℝ), 
    (∀ x, f x = x^2 + a*x + b) → 
    (∀ x, |x| ≤ 1 → |f x| ≤ F) ∧
    (∃ a b : ℝ, ∃ x, |x| ≤ 1 ∧ |f x| = F)) :=
sorry

end min_max_abs_quadratic_on_unit_interval_l966_96650


namespace circle_center_sum_l966_96641

/-- Given a circle with equation x^2 + y^2 - 10x + 4y = -40, 
    the sum of the x and y coordinates of its center is 3. -/
theorem circle_center_sum (x y : ℝ) : 
  x^2 + y^2 - 10*x + 4*y = -40 → x + y = 3 := by
sorry

end circle_center_sum_l966_96641


namespace escalator_length_is_160_l966_96608

/-- The length of an escalator given its speed, a person's walking speed on it, and the time taken to cover the entire length. -/
def escalatorLength (escalatorSpeed personSpeed : ℝ) (timeTaken : ℝ) : ℝ :=
  (escalatorSpeed + personSpeed) * timeTaken

/-- Theorem stating that the length of the escalator is 160 feet under the given conditions. -/
theorem escalator_length_is_160 :
  escalatorLength 12 8 8 = 160 := by
  sorry

end escalator_length_is_160_l966_96608


namespace jane_inspection_fraction_l966_96694

theorem jane_inspection_fraction :
  ∀ (P : ℝ) (J : ℝ),
    P > 0 →
    J > 0 →
    J < 1 →
    0.005 * (1 - J) * P + 0.008 * J * P = 0.0075 * P →
    J = 5 / 6 := by
  sorry

end jane_inspection_fraction_l966_96694


namespace original_number_proof_l966_96646

theorem original_number_proof (x : ℝ) (h : 1 + 1/x = 7/3) : x = 3/4 := by
  sorry

end original_number_proof_l966_96646


namespace fraction_order_l966_96603

theorem fraction_order : 
  (22 : ℚ) / 19 < (18 : ℚ) / 15 ∧ 
  (18 : ℚ) / 15 < (21 : ℚ) / 17 ∧ 
  (21 : ℚ) / 17 < (20 : ℚ) / 16 := by
sorry

end fraction_order_l966_96603


namespace original_function_derivation_l966_96672

/-- A linear function represented by its slope and y-intercept -/
structure LinearFunction where
  slope : ℝ
  intercept : ℝ

/-- Rotates a linear function 180° around the origin -/
def rotate180 (f : LinearFunction) : LinearFunction :=
  { slope := -f.slope, intercept := -f.intercept }

/-- Translates a linear function horizontally -/
def translateLeft (f : LinearFunction) (units : ℝ) : LinearFunction :=
  { slope := f.slope, intercept := f.intercept + f.slope * units }

/-- Checks if a linear function passes through two points -/
def passesThrough (f : LinearFunction) (x1 y1 x2 y2 : ℝ) : Prop :=
  f.slope * x1 + f.intercept = y1 ∧ f.slope * x2 + f.intercept = y2

theorem original_function_derivation (k b : ℝ) :
  let f := LinearFunction.mk k b
  let rotated := rotate180 f
  let translated := translateLeft rotated 2
  passesThrough translated (-4) 0 0 2 →
  k = 1/2 ∧ b = -1 := by sorry

end original_function_derivation_l966_96672


namespace remainder_relationship_l966_96667

theorem remainder_relationship (M M' N D S S' s s' : ℕ) : 
  M > M' →
  M % D = S →
  M' % D = S' →
  (M^2 * M') % D = s →
  N^2 % D = s' →
  (∃ M M' N D S S' s s' : ℕ, s = s') ∧
  (∃ M M' N D S S' s s' : ℕ, s < s') :=
by sorry

end remainder_relationship_l966_96667


namespace circle_properties_l966_96673

/-- A circle with center on the y-axis, radius 1, and passing through (1, 2) -/
def circle_equation (x y : ℝ) : Prop :=
  x^2 + (y - 2)^2 = 1

theorem circle_properties :
  ∃ (b : ℝ), 
    (∀ x y : ℝ, circle_equation x y ↔ x^2 + (y - b)^2 = 1) ∧ 
    (0, b) = (0, 2) ∧
    (∀ x y : ℝ, circle_equation x y → (x - 0)^2 + (y - b)^2 = 1) ∧
    circle_equation 1 2 :=
by sorry

end circle_properties_l966_96673


namespace g_composition_sqrt3_l966_96602

noncomputable def g (b c : ℝ) (x : ℝ) : ℝ := b * x + c * x^3 - Real.sqrt 3

theorem g_composition_sqrt3 (b c : ℝ) (hb : b > 0) (hc : c > 0) :
  g b c (g b c (Real.sqrt 3)) = -Real.sqrt 3 → b = 0 ∧ c = 1/3 := by
  sorry

end g_composition_sqrt3_l966_96602


namespace union_A_B_complement_A_l966_96642

-- Define the sets A and B
def A : Set ℝ := {x | x^2 - x - 2 < 0}
def B : Set ℝ := {x | 1 < x ∧ x < 3}

-- Theorem for A ∪ B
theorem union_A_B : A ∪ B = {x | -1 < x ∧ x < 3} := by sorry

-- Theorem for ∁ℝA
theorem complement_A : (Set.univ : Set ℝ) \ A = {x | x ≤ -1 ∨ x ≥ 2} := by sorry

end union_A_B_complement_A_l966_96642


namespace modulus_of_complex_quotient_l966_96631

theorem modulus_of_complex_quotient : 
  ∀ (z₁ z₂ : ℂ), 
    z₁ = Complex.mk 0 2 → 
    z₂ = Complex.mk 1 (-1) → 
    Complex.abs (z₁ / z₂) = Real.sqrt 2 := by
sorry

end modulus_of_complex_quotient_l966_96631


namespace goats_in_field_l966_96605

theorem goats_in_field (total : Nat) (cows : Nat) (sheep : Nat) (chickens : Nat) 
  (h1 : total = 900)
  (h2 : cows = 250)
  (h3 : sheep = 310)
  (h4 : chickens = 180) :
  total - (cows + sheep + chickens) = 160 := by
  sorry

end goats_in_field_l966_96605


namespace positive_root_of_cubic_l966_96635

theorem positive_root_of_cubic (x : ℝ) :
  x = 2 + Real.sqrt 2 →
  x^3 - 4*x^2 + x - 2*Real.sqrt 2 = 0 :=
by sorry

end positive_root_of_cubic_l966_96635


namespace rotation_result_l966_96621

-- Define the shapes
inductive Shape
  | Triangle
  | SmallCircle
  | Square
  | InvertedTriangle

-- Define the initial configuration
def initial_config : List Shape :=
  [Shape.Triangle, Shape.SmallCircle, Shape.Square, Shape.InvertedTriangle]

-- Define the rotation function
def rotate (angle : ℕ) (config : List Shape) : List Shape :=
  let shift := angle / 30  -- 150° = 5 * 30°
  config.rotateLeft shift

-- Theorem statement
theorem rotation_result :
  rotate 150 initial_config = [Shape.Square, Shape.InvertedTriangle, Shape.Triangle, Shape.SmallCircle] :=
by sorry

end rotation_result_l966_96621


namespace product_base_8_units_digit_l966_96696

def base_10_to_8_units_digit (n : ℕ) : ℕ :=
  n % 8

theorem product_base_8_units_digit :
  base_10_to_8_units_digit (348 * 27) = 4 := by
sorry

end product_base_8_units_digit_l966_96696


namespace ratio_of_percentages_l966_96632

theorem ratio_of_percentages (P Q R M N : ℝ) 
  (hM : M = 0.4 * Q)
  (hQ : Q = 0.25 * P)
  (hR : R = 0.6 * P)
  (hN : N = 0.5 * R) :
  M / N = 1 / 3 := by
  sorry

end ratio_of_percentages_l966_96632


namespace tom_climbing_time_l966_96609

/-- Tom and Elizabeth's hill climbing competition -/
theorem tom_climbing_time (elizabeth_time : ℕ) (tom_factor : ℕ) : elizabeth_time = 30 → tom_factor = 4 → (elizabeth_time * tom_factor) / 60 = 2 := by
  sorry

end tom_climbing_time_l966_96609


namespace sum_of_digits_0_to_99_l966_96630

/-- Sum of digits of a natural number -/
def sumOfDigits (n : ℕ) : ℕ := sorry

/-- Sum of digits for a range of natural numbers -/
def sumOfDigitsRange (a b : ℕ) : ℕ := 
  (Finset.range (b - a + 1)).sum (fun i => sumOfDigits (a + i))

theorem sum_of_digits_0_to_99 :
  sumOfDigitsRange 0 99 = 900 :=
by
  sorry

/-- Given condition -/
axiom sum_of_digits_18_to_21 : sumOfDigitsRange 18 21 = 24

#check sum_of_digits_0_to_99
#check sum_of_digits_18_to_21

end sum_of_digits_0_to_99_l966_96630


namespace full_price_revenue_l966_96690

def total_tickets : ℕ := 180
def total_revenue : ℕ := 2400

def ticket_revenue (full_price : ℕ) (num_full_price : ℕ) : Prop :=
  ∃ (half_price : ℕ),
    half_price = full_price / 2 ∧
    num_full_price + (total_tickets - num_full_price) = total_tickets ∧
    num_full_price * full_price + (total_tickets - num_full_price) * half_price = total_revenue

theorem full_price_revenue : 
  ∃ (full_price : ℕ) (num_full_price : ℕ), 
    ticket_revenue full_price num_full_price ∧ 
    full_price * num_full_price = 300 :=
by sorry

end full_price_revenue_l966_96690


namespace evaluate_expression_l966_96624

theorem evaluate_expression : (10^8 / (2.5 * 10^5)) * 3 = 1200 := by
  sorry

end evaluate_expression_l966_96624


namespace calculation_proof_l966_96600

theorem calculation_proof : 
  |(-1/2 : ℝ)| + (2023 - Real.pi)^0 - (27 : ℝ)^(1/3) = -3/2 := by
  sorry

end calculation_proof_l966_96600


namespace marias_painting_price_l966_96627

/-- The selling price of Maria's painting --/
def selling_price (brush_cost canvas_cost paint_cost_per_liter paint_liters earnings : ℕ) : ℕ :=
  brush_cost + canvas_cost + paint_cost_per_liter * paint_liters + earnings

/-- Theorem stating the selling price of Maria's painting --/
theorem marias_painting_price :
  selling_price 20 (3 * 20) 8 5 80 = 200 := by
  sorry

end marias_painting_price_l966_96627


namespace mean_of_four_numbers_l966_96671

theorem mean_of_four_numbers (a b c d : ℚ) (h : a + b + c + d = 3/4) : 
  (a + b + c + d) / 4 = 3/16 := by
sorry

end mean_of_four_numbers_l966_96671


namespace complex_number_properties_l966_96662

/-- Given a complex number z where z + 1/z is real, this theorem proves:
    1. The value of z that minimizes |z + 2 - i|
    2. The minimum value of |z + 2 - i|
    3. u = (1 - z) / (1 + z) is purely imaginary -/
theorem complex_number_properties (z : ℂ) 
    (h : (z + z⁻¹).im = 0) : 
    ∃ (min_z : ℂ) (min_val : ℝ),
    (min_z = -2 * Real.sqrt 5 / 5 + (Real.sqrt 5 / 5) * Complex.I) ∧
    (min_val = Real.sqrt 5 - 1) ∧
    (∀ w : ℂ, Complex.abs (w + 2 - Complex.I) ≥ min_val) ∧
    (Complex.abs (min_z + 2 - Complex.I) = min_val) ∧
    ((1 - z) / (1 + z)).re = 0 := by
  sorry

end complex_number_properties_l966_96662


namespace fraction_equals_zero_l966_96607

theorem fraction_equals_zero (x : ℝ) : (x + 2) / (x - 3) = 0 → x = -2 := by
  sorry

end fraction_equals_zero_l966_96607


namespace mean_of_combined_sets_l966_96693

theorem mean_of_combined_sets :
  ∀ (set1 set2 : List ℝ),
    set1.length = 7 →
    set2.length = 8 →
    (set1.sum / set1.length : ℝ) = 15 →
    (set2.sum / set2.length : ℝ) = 20 →
    ((set1 ++ set2).sum / (set1 ++ set2).length : ℝ) = 17.67 := by
  sorry

end mean_of_combined_sets_l966_96693


namespace monotone_increasing_implies_m_leq_half_m_leq_half_not_implies_monotone_increasing_l966_96620

-- Define the function f(x)
def f (m : ℝ) (x : ℝ) : ℝ := x^2 - m*x

-- Define the property of f being monotonically increasing on (0, +∞)
def is_monotone_increasing_on_positive (m : ℝ) : Prop :=
  ∀ x y, 0 < x → 0 < y → x < y → f m x < f m y

-- Theorem stating that if f is monotonically increasing on (0, +∞), then m ≤ 1/2
theorem monotone_increasing_implies_m_leq_half (m : ℝ) :
  is_monotone_increasing_on_positive m → m ≤ 1/2 :=
sorry

-- Theorem stating that m ≤ 1/2 does not necessarily imply f is monotonically increasing on (0, +∞)
theorem m_leq_half_not_implies_monotone_increasing :
  ∃ m, m ≤ 1/2 ∧ ¬is_monotone_increasing_on_positive m :=
sorry

end monotone_increasing_implies_m_leq_half_m_leq_half_not_implies_monotone_increasing_l966_96620


namespace quadratic_real_roots_l966_96654

theorem quadratic_real_roots (k : ℝ) :
  (∃ x : ℝ, k * x^2 - 6 * x + 9 = 0) ↔ (k ≤ 1 ∧ k ≠ 0) := by
  sorry

end quadratic_real_roots_l966_96654


namespace greg_bike_rotations_l966_96677

/-- Calculates the additional wheel rotations needed to reach a goal distance -/
def additional_rotations_needed (rotations_per_block : ℕ) (goal_blocks : ℕ) (current_rotations : ℕ) : ℕ :=
  rotations_per_block * goal_blocks - current_rotations

theorem greg_bike_rotations :
  let rotations_per_block : ℕ := 200
  let goal_blocks : ℕ := 8
  let current_rotations : ℕ := 600
  additional_rotations_needed rotations_per_block goal_blocks current_rotations = 1000 := by
  sorry

end greg_bike_rotations_l966_96677


namespace easter_egg_hunt_l966_96634

theorem easter_egg_hunt (total_eggs : ℕ) (club_house_eggs : ℕ) (town_hall_eggs : ℕ) 
  (h1 : total_eggs = 80)
  (h2 : club_house_eggs = 40)
  (h3 : town_hall_eggs = 15) :
  ∃ park_eggs : ℕ, 
    park_eggs = total_eggs - club_house_eggs - town_hall_eggs ∧ 
    park_eggs = 25 := by
  sorry

end easter_egg_hunt_l966_96634


namespace last_two_digits_sum_factorials_14_l966_96683

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

def last_two_digits (n : ℕ) : ℕ := n % 100

def sum_factorials (n : ℕ) : ℕ := (List.range n).map factorial |>.sum

theorem last_two_digits_sum_factorials_14 :
  last_two_digits (sum_factorials 14) = last_two_digits 409113 := by
  sorry

end last_two_digits_sum_factorials_14_l966_96683


namespace cube_surface_area_l966_96611

theorem cube_surface_area (volume : ℝ) (side : ℝ) (surface_area : ℝ) : 
  volume = 1728 → 
  volume = side^3 → 
  surface_area = 6 * side^2 → 
  surface_area = 864 :=
by
  sorry

end cube_surface_area_l966_96611


namespace cricket_bat_profit_percentage_l966_96688

/-- Calculates the weighted average profit percentage for cricket bat sales -/
theorem cricket_bat_profit_percentage :
  let bat_a_quantity : ℕ := 5
  let bat_a_cost : ℚ := 900
  let bat_a_profit : ℚ := 300
  let bat_b_quantity : ℕ := 8
  let bat_b_cost : ℚ := 1200
  let bat_b_profit : ℚ := 400
  let bat_c_quantity : ℕ := 3
  let bat_c_cost : ℚ := 1500
  let bat_c_profit : ℚ := 500

  let total_cost : ℚ := bat_a_quantity * bat_a_cost + bat_b_quantity * bat_b_cost + bat_c_quantity * bat_c_cost
  let total_profit : ℚ := bat_a_quantity * bat_a_profit + bat_b_quantity * bat_b_profit + bat_c_quantity * bat_c_profit

  let weighted_avg_profit_percentage : ℚ := (total_profit / total_cost) * 100

  weighted_avg_profit_percentage = 100/3 := by sorry

end cricket_bat_profit_percentage_l966_96688


namespace expression_evaluation_l966_96644

theorem expression_evaluation (x y : ℝ) (h1 : x > y) (h2 : y > 0) :
  (x^(y+1) * y^x) / (y^(x+1) * x^y) = x / y := by
  sorry

end expression_evaluation_l966_96644


namespace y_intercept_of_parallel_line_through_point_l966_96639

/-- A line in 2D space represented by its slope and y-intercept -/
structure Line where
  slope : ℝ
  y_intercept : ℝ

/-- The y-coordinate of a point on a line given its x-coordinate -/
def Line.y_at (l : Line) (x : ℝ) : ℝ :=
  l.slope * x + l.y_intercept

/-- Two lines are parallel if they have the same slope -/
def parallel (l1 l2 : Line) : Prop :=
  l1.slope = l2.slope

theorem y_intercept_of_parallel_line_through_point 
  (l : Line) (x₀ y₀ : ℝ) : 
  parallel l { slope := -3, y_intercept := 6 } →
  l.y_at x₀ = y₀ →
  x₀ = 3 →
  y₀ = -2 →
  l.y_intercept = 7 := by
  sorry

end y_intercept_of_parallel_line_through_point_l966_96639


namespace minimum_eccentricity_sum_l966_96622

/-- Given two points F₁ and F₂ that are common foci of an ellipse and a hyperbola,
    and P is their common point. -/
structure CommonFociConfig where
  F₁ : ℝ × ℝ
  F₂ : ℝ × ℝ
  P : ℝ × ℝ

/-- The eccentricity of the ellipse -/
def eccentricity_ellipse (config : CommonFociConfig) : ℝ := sorry

/-- The eccentricity of the hyperbola -/
def eccentricity_hyperbola (config : CommonFociConfig) : ℝ := sorry

/-- Distance between two points -/
def distance (p q : ℝ × ℝ) : ℝ := sorry

theorem minimum_eccentricity_sum (config : CommonFociConfig) 
  (h1 : distance config.P config.F₂ > distance config.P config.F₁)
  (h2 : distance config.P config.F₁ = distance config.F₁ config.F₂) :
  (∀ e₁ e₂ : ℝ, e₁ = eccentricity_ellipse config → e₂ = eccentricity_hyperbola config →
    3 / e₁ + e₂ / 3 ≥ 8) ∧ 
  (∃ e₁ e₂ : ℝ, e₁ = eccentricity_ellipse config ∧ e₂ = eccentricity_hyperbola config ∧
    3 / e₁ + e₂ / 3 = 8) :=
sorry

end minimum_eccentricity_sum_l966_96622


namespace percentage_problem_l966_96686

theorem percentage_problem (x : ℝ) (h : 160 = 320 / 100 * x) : x = 50 := by
  sorry

end percentage_problem_l966_96686


namespace original_number_proof_l966_96657

theorem original_number_proof (x : ℝ) : 
  x - 25 = 0.75 * x + 25 → x = 200 := by
  sorry

end original_number_proof_l966_96657


namespace average_age_when_youngest_born_l966_96617

/-- Given a group of people, their average age, and the age of the youngest person,
    calculate the average age of the group when the youngest was born. -/
theorem average_age_when_youngest_born 
  (n : ℕ) -- Total number of people
  (avg : ℝ) -- Current average age
  (youngest : ℝ) -- Age of the youngest person
  (h1 : n = 7) -- There are 7 people
  (h2 : avg = 30) -- The current average age is 30
  (h3 : youngest = 3) -- The youngest person is 3 years old
  : (n * avg - youngest) / (n - 1) = 34.5 := by
  sorry

end average_age_when_youngest_born_l966_96617


namespace problem_statement_l966_96669

theorem problem_statement (a b : ℝ) (h1 : a * b = -3) (h2 : a + b = 2) :
  a^2 * b + a * b^2 = -6 := by
  sorry

end problem_statement_l966_96669


namespace ticket_distribution_proof_l966_96629

theorem ticket_distribution_proof (total_tickets : ℕ) (total_amount : ℚ) 
  (price_15 price_10 price_5_5 : ℚ) :
  total_tickets = 22 →
  total_amount = 229 →
  price_15 = 15 →
  price_10 = 10 →
  price_5_5 = (11 : ℚ) / 2 →
  ∃! (x y z : ℕ), 
    x + y + z = total_tickets ∧ 
    price_15 * x + price_10 * y + price_5_5 * z = total_amount ∧
    x = 9 ∧ y = 5 ∧ z = 8 := by
  sorry

end ticket_distribution_proof_l966_96629


namespace charity_race_fundraising_l966_96698

theorem charity_race_fundraising (total_students : ℕ) (group1_students : ℕ) (group1_amount : ℕ) (group2_amount : ℕ) :
  total_students = 30 →
  group1_students = 10 →
  group1_amount = 20 →
  group2_amount = 30 →
  (group1_students * group1_amount) + ((total_students - group1_students) * group2_amount) = 800 :=
by sorry

end charity_race_fundraising_l966_96698


namespace line_perp_plane_sufficient_condition_l966_96681

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the perpendicular and parallel relations
variable (perp : Line → Plane → Prop)
variable (para : Line → Line → Prop)

-- Theorem statement
theorem line_perp_plane_sufficient_condition 
  (m n : Line) (α : Plane) :
  para m n → perp n α → perp m α :=
sorry

end line_perp_plane_sufficient_condition_l966_96681


namespace equilateral_triangle_exists_l966_96658

-- Define the plane S parallel to x₁,₂ axis
structure Plane :=
  (s₁ : ℝ)
  (s₂ : ℝ)

-- Define a point in 3D space
structure Point3D :=
  (x : ℝ)
  (y : ℝ)
  (z : ℝ)

-- Define the trace lines of the plane
def traceLine1 (S : Plane) : Set Point3D :=
  {p : Point3D | p.y = S.s₁}

def traceLine2 (S : Plane) : Set Point3D :=
  {p : Point3D | p.z = S.s₂}

-- Define an equilateral triangle
structure EquilateralTriangle :=
  (A : Point3D)
  (B : Point3D)
  (C : Point3D)

-- State the theorem
theorem equilateral_triangle_exists (S : Plane) (A : Point3D) 
  (h : A.y = S.s₁ ∧ A.z = S.s₂) : 
  ∃ (t : EquilateralTriangle), 
    t.A = A ∧ 
    t.B ∈ traceLine1 S ∧ 
    t.C ∈ traceLine2 S ∧
    (t.A.x - t.B.x)^2 + (t.A.y - t.B.y)^2 + (t.A.z - t.B.z)^2 = 
    (t.B.x - t.C.x)^2 + (t.B.y - t.C.y)^2 + (t.B.z - t.C.z)^2 ∧
    (t.A.x - t.B.x)^2 + (t.A.y - t.B.y)^2 + (t.A.z - t.B.z)^2 = 
    (t.A.x - t.C.x)^2 + (t.A.y - t.C.y)^2 + (t.A.z - t.C.z)^2 := by
  sorry


end equilateral_triangle_exists_l966_96658


namespace cube_root_27_minus_2_l966_96640

theorem cube_root_27_minus_2 : (27 : ℝ) ^ (1/3) - 2 = 1 := by
  sorry

end cube_root_27_minus_2_l966_96640


namespace parallel_vectors_x_value_l966_96676

theorem parallel_vectors_x_value (x : ℝ) : 
  let a : Fin 2 → ℝ := ![1, 2]
  let b : Fin 2 → ℝ := ![x, -3]
  (∃ (k : ℝ), k ≠ 0 ∧ a = k • b) → x = -3/2 := by
  sorry

end parallel_vectors_x_value_l966_96676


namespace convergence_implies_cluster_sets_l966_96695

open Set
open Filter
open Topology

/-- A sequence converges to a limit -/
def SequenceConvergesTo (x : ℕ → ℝ) (a : ℝ) :=
  Tendsto x atTop (𝓝 a)

/-- An interval is a cluster set for a sequence if it contains infinitely many terms of the sequence -/
def IsClusterSet (x : ℕ → ℝ) (s : Set ℝ) :=
  ∀ n : ℕ, ∃ m ≥ n, x m ∈ s

theorem convergence_implies_cluster_sets (x : ℕ → ℝ) (a : ℝ) :
  SequenceConvergesTo x a →
  (∀ ε > 0, IsClusterSet x (Ioo (a - ε) (a + ε))) ∧
  (∀ s : Set ℝ, IsOpen s → a ∉ s → ¬IsClusterSet x s) :=
sorry

end convergence_implies_cluster_sets_l966_96695


namespace average_income_calculation_l966_96633

theorem average_income_calculation (total_customers : ℕ) 
  (wealthy_customers : ℕ) (other_customers : ℕ) 
  (wealthy_avg_income : ℝ) (other_avg_income : ℝ) :
  total_customers = wealthy_customers + other_customers →
  wealthy_customers = 10 →
  other_customers = 40 →
  wealthy_avg_income = 55000 →
  other_avg_income = 42500 →
  (wealthy_customers * wealthy_avg_income + other_customers * other_avg_income) / total_customers = 45000 :=
by sorry

end average_income_calculation_l966_96633


namespace weight_of_b_l966_96637

/-- Given three weights a, b, and c, prove that b = 33 under the given conditions -/
theorem weight_of_b (a b c : ℝ) : 
  (a + b + c) / 3 = 45 →
  (a + b) / 2 = 40 →
  (b + c) / 2 = 44 →
  b = 33 := by
sorry

end weight_of_b_l966_96637


namespace david_work_rate_l966_96612

/-- The number of days it takes John to complete the work -/
def john_days : ℝ := 9

/-- The number of days it takes David and John together to complete the work -/
def combined_days : ℝ := 3.2142857142857144

/-- The number of days it takes David to complete the work alone -/
def david_days : ℝ := 5

/-- Theorem stating that given John's work rate and the combined work rate of David and John,
    David's individual work rate can be determined -/
theorem david_work_rate (ε : ℝ) (h_ε : ε > 0) :
  ∃ (d : ℝ), abs (d - david_days) < ε ∧
  1 / d + 1 / john_days = 1 / combined_days :=
sorry

end david_work_rate_l966_96612


namespace mean_equality_implies_z_value_l966_96606

theorem mean_equality_implies_z_value :
  let mean1 := (8 + 12 + 24) / 3
  let mean2 := (16 + z) / 2
  mean1 = mean2 → z = 40 / 3 :=
by
  sorry

end mean_equality_implies_z_value_l966_96606


namespace partial_fraction_decomposition_sum_l966_96668

theorem partial_fraction_decomposition_sum (p q r A B C : ℝ) : 
  (p ≠ q ∧ q ≠ r ∧ p ≠ r) →
  (∀ (x : ℝ), x ≠ p ∧ x ≠ q ∧ x ≠ r → 
    1 / (x^3 - 15*x^2 + 50*x - 56) = A / (x - p) + B / (x - q) + C / (x - r)) →
  (x^3 - 15*x^2 + 50*x - 56 = (x - p) * (x - q) * (x - r)) →
  1 / A + 1 / B + 1 / C = 225 := by
sorry

end partial_fraction_decomposition_sum_l966_96668


namespace inequality_and_minimum_value_l966_96661

theorem inequality_and_minimum_value 
  (m n : ℝ) 
  (h_diff : m ≠ n) 
  (h_pos_m : m > 0) 
  (h_pos_n : n > 0) 
  (x y : ℝ) 
  (h_pos_x : x > 0) 
  (h_pos_y : y > 0) :
  (m^2 / x + n^2 / y > (m + n)^2 / (x + y)) ∧
  (∃ (min_val : ℝ) (min_x : ℝ), 
    min_val = 64 ∧ 
    min_x = 1/8 ∧ 
    (∀ x, 0 < x ∧ x < 1/5 → 5/x + 9/(1-5*x) ≥ min_val) ∧
    (5/min_x + 9/(1-5*min_x) = min_val)) := by
  sorry

end inequality_and_minimum_value_l966_96661


namespace right_triangle_minimum_side_l966_96649

theorem right_triangle_minimum_side : ∃ (s : ℕ), 
  (s ≥ 25) ∧ 
  (∀ (t : ℕ), t < 25 → ¬(7^2 + 24^2 = t^2)) ∧
  (7^2 + 24^2 = s^2) ∧
  (7 + 24 > s) ∧ (24 + s > 7) ∧ (7 + s > 24) := by
  sorry

end right_triangle_minimum_side_l966_96649


namespace book_selection_l966_96699

theorem book_selection (n m k : ℕ) (h1 : n = 7) (h2 : m = 5) (h3 : k = 3) :
  (Nat.choose (n - 2) k) = (Nat.choose m k) :=
by sorry

end book_selection_l966_96699


namespace least_months_to_triple_l966_96604

def interest_rate : ℝ := 1.05

theorem least_months_to_triple (n : ℕ) : (∀ m : ℕ, m < n → interest_rate ^ m ≤ 3) ∧ interest_rate ^ n > 3 ↔ n = 23 := by
  sorry

end least_months_to_triple_l966_96604


namespace error_percentage_squared_vs_multiplied_by_eight_l966_96675

theorem error_percentage_squared_vs_multiplied_by_eight (x : ℝ) (h : x > 0) :
  let correct_result := 8 * x
  let incorrect_result := x ^ 2
  let error := |incorrect_result - correct_result|
  let error_percentage := error / correct_result * 100
  error_percentage = |x - 8| / 8 * 100 := by sorry

end error_percentage_squared_vs_multiplied_by_eight_l966_96675


namespace orchids_cut_correct_l966_96689

/-- The number of red orchids Sally cut from her garden -/
def orchids_cut (initial_red : ℕ) (final_red : ℕ) : ℕ :=
  final_red - initial_red

/-- Theorem stating that the number of orchids Sally cut is the difference between final and initial red orchids -/
theorem orchids_cut_correct (initial_red initial_white final_red : ℕ) 
  (h1 : initial_red = 9)
  (h2 : initial_white = 3)
  (h3 : final_red = 15) :
  orchids_cut initial_red final_red = 6 := by
  sorry

#eval orchids_cut 9 15

end orchids_cut_correct_l966_96689


namespace range_of_c_l966_96687

theorem range_of_c (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (h1 : 1 / a + 1 / b = 1) (h2 : 1 / (a + b) + 1 / c = 1) :
  1 < c ∧ c ≤ 4 / 3 := by
  sorry

end range_of_c_l966_96687


namespace triangle_problem_l966_96652

theorem triangle_problem (a b c A B C : ℝ) :
  a > 0 ∧ b > 0 ∧ c > 0 ∧
  0 < A ∧ A < π ∧ 0 < B ∧ B < π ∧ 0 < C ∧ C < π ∧
  A + B + C = π ∧
  Real.sqrt 3 * b * Real.sin A - a * Real.cos B - 2 * a = 0 ∧
  b = Real.sqrt 7 ∧
  1/2 * a * b * Real.sin C = Real.sqrt 3 / 2 →
  B = 2 * π / 3 ∧ a + b + c = 3 + Real.sqrt 7 :=
by sorry

end triangle_problem_l966_96652


namespace canoe_current_speed_l966_96643

/-- Represents the speed of a canoe in still water and the speed of the current. -/
structure CanoeSpeedData where
  canoe_speed : ℝ
  current_speed : ℝ

/-- Calculates the effective speed of a canoe given the canoe's speed in still water and the current speed. -/
def effective_speed (upstream : Bool) (data : CanoeSpeedData) : ℝ :=
  if upstream then data.canoe_speed - data.current_speed else data.canoe_speed + data.current_speed

/-- Theorem stating that given the conditions of the canoe problem, the speed of the current is 7 miles per hour. -/
theorem canoe_current_speed : 
  ∀ (data : CanoeSpeedData),
    (effective_speed true data) * 6 = 12 →
    (effective_speed false data) * 0.75 = 12 →
    data.current_speed = 7 := by
  sorry


end canoe_current_speed_l966_96643
