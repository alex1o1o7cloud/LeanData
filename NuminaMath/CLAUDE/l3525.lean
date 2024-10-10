import Mathlib

namespace fruit_store_problem_l3525_352587

/-- The number of watermelons in a fruit store. -/
def num_watermelons : ℕ := by sorry

theorem fruit_store_problem :
  let apples : ℕ := 82
  let pears : ℕ := 90
  let tangerines : ℕ := 88
  let melons : ℕ := 84
  let total_fruits : ℕ := apples + pears + tangerines + melons + num_watermelons
  (total_fruits % 88 = 0) ∧ (total_fruits / 88 = 5) →
  num_watermelons = 96 := by sorry

end fruit_store_problem_l3525_352587


namespace max_value_trig_expression_l3525_352510

theorem max_value_trig_expression (a b φ : ℝ) :
  (∀ θ : ℝ, a * Real.cos (θ + φ) + b * Real.sin (θ + φ) ≤ Real.sqrt (a^2 + b^2)) ∧
  (∃ θ : ℝ, a * Real.cos (θ + φ) + b * Real.sin (θ + φ) = Real.sqrt (a^2 + b^2)) :=
by sorry

end max_value_trig_expression_l3525_352510


namespace airport_distance_l3525_352586

/-- Represents the problem of calculating the distance to the airport --/
def airport_distance_problem (initial_speed : ℝ) (speed_increase : ℝ) (late_time : ℝ) : Prop :=
  ∃ (distance : ℝ) (initial_time : ℝ),
    -- If he continued at initial speed, he'd be 1 hour late
    distance = initial_speed * (initial_time + 1) ∧
    -- The remaining distance at increased speed
    (distance - initial_speed) = (initial_speed + speed_increase) * (initial_time - late_time) ∧
    -- The total distance is 70 miles
    distance = 70

/-- The theorem stating that the airport is 70 miles away --/
theorem airport_distance :
  airport_distance_problem 40 20 (1/4) :=
sorry


end airport_distance_l3525_352586


namespace number_problem_l3525_352531

theorem number_problem : ∃ x : ℝ, x > 0 ∧ 0.9 * x = (4/5 * 25) + 16 := by
  sorry

end number_problem_l3525_352531


namespace f_is_power_and_increasing_l3525_352560

-- Define a power function
def isPowerFunction (f : ℝ → ℝ) : Prop :=
  ∃ a : ℝ, ∀ x > 0, f x = x^a

-- Define an increasing function on (0, +∞)
def isIncreasing (f : ℝ → ℝ) : Prop :=
  ∀ x y, 0 < x ∧ x < y → f x < f y

-- Define the function f(x) = x^(1/2)
def f (x : ℝ) : ℝ := x^(1/2)

-- Theorem statement
theorem f_is_power_and_increasing :
  isPowerFunction f ∧ isIncreasing f :=
sorry

end f_is_power_and_increasing_l3525_352560


namespace product_of_x_values_l3525_352550

theorem product_of_x_values (x : ℝ) : 
  (|15 / x - 2| = 3) → (∃ y : ℝ, (|15 / y - 2| = 3) ∧ x * y = -45) :=
by sorry

end product_of_x_values_l3525_352550


namespace total_pears_is_five_l3525_352593

/-- The number of pears Keith picked -/
def keith_pears : ℕ := 3

/-- The number of pears Jason picked -/
def jason_pears : ℕ := 2

/-- The total number of pears picked -/
def total_pears : ℕ := keith_pears + jason_pears

/-- Theorem: The total number of pears picked is 5 -/
theorem total_pears_is_five : total_pears = 5 := by
  sorry

end total_pears_is_five_l3525_352593


namespace orthogonal_vectors_m_l3525_352500

def a : ℝ × ℝ := (3, 4)
def b : ℝ × ℝ := (2, -1)

theorem orthogonal_vectors_m (m : ℝ) : 
  (a.1 + m * b.1, a.2 + m * b.2) • (a.1 - b.1, a.2 - b.2) = 0 → m = 23 / 3 :=
by sorry

end orthogonal_vectors_m_l3525_352500


namespace cube_root_rationality_l3525_352570

theorem cube_root_rationality (a b : ℚ) (ha : 0 < a) (hb : 0 < b) 
  (h : ∃ (s : ℚ), s = (a^(1/3) + b^(1/3))) : 
  ∃ (r₁ r₂ : ℚ), r₁ = a^(1/3) ∧ r₂ = b^(1/3) := by
sorry

end cube_root_rationality_l3525_352570


namespace max_b_in_box_l3525_352574

theorem max_b_in_box (a b c : ℕ) : 
  (a * b * c = 360) →
  (1 < c) →
  (c < b) →
  (b < a) →
  (∀ a' b' c' : ℕ, (a' * b' * c' = 360) → (1 < c') → (c' < b') → (b' < a') → b' ≤ b) →
  b = 10 := by
sorry

end max_b_in_box_l3525_352574


namespace larger_number_proof_l3525_352591

theorem larger_number_proof (x y : ℝ) : 
  y > x → 4 * y = 3 * x → y - x = 12 → y = -36 := by
  sorry

end larger_number_proof_l3525_352591


namespace smallest_four_digit_mod_9_l3525_352514

theorem smallest_four_digit_mod_9 :
  (∀ n : ℕ, 1000 ≤ n ∧ n < 10000 ∧ n % 9 = 8 → 1007 ≤ n) ∧
  1000 ≤ 1007 ∧ 1007 < 10000 ∧ 1007 % 9 = 8 := by
sorry

end smallest_four_digit_mod_9_l3525_352514


namespace zoo_trip_short_amount_l3525_352563

/-- Represents the zoo trip expenses and budget for two people -/
structure ZooTrip where
  total_budget : ℕ
  zoo_entry_cost : ℕ
  aquarium_entry_cost : ℕ
  animal_show_cost : ℕ
  bus_fare : ℕ
  num_transfers : ℕ
  souvenir_budget : ℕ
  noah_lunch_cost : ℕ
  ava_lunch_cost : ℕ
  beverage_cost : ℕ
  num_people : ℕ

/-- Calculates the amount short for lunch and snacks -/
def amount_short (trip : ZooTrip) : ℕ :=
  let total_entry_cost := (trip.zoo_entry_cost + trip.aquarium_entry_cost + trip.animal_show_cost) * trip.num_people
  let total_bus_fare := trip.bus_fare * trip.num_transfers * trip.num_people
  let total_lunch_cost := trip.noah_lunch_cost + trip.ava_lunch_cost
  let total_beverage_cost := trip.beverage_cost * trip.num_people
  let total_expenses := total_entry_cost + total_bus_fare + trip.souvenir_budget + total_lunch_cost + total_beverage_cost
  total_expenses - trip.total_budget

/-- Theorem stating that the amount short for lunch and snacks is $12 -/
theorem zoo_trip_short_amount (trip : ZooTrip) 
  (h1 : trip.total_budget = 100)
  (h2 : trip.zoo_entry_cost = 5)
  (h3 : trip.aquarium_entry_cost = 7)
  (h4 : trip.animal_show_cost = 4)
  (h5 : trip.bus_fare = 150) -- Using cents for precise integer arithmetic
  (h6 : trip.num_transfers = 4)
  (h7 : trip.souvenir_budget = 20)
  (h8 : trip.noah_lunch_cost = 10)
  (h9 : trip.ava_lunch_cost = 8)
  (h10 : trip.beverage_cost = 3)
  (h11 : trip.num_people = 2) :
  amount_short trip = 12 := by
  sorry


end zoo_trip_short_amount_l3525_352563


namespace min_value_of_sum_l3525_352578

theorem min_value_of_sum (x y z : ℝ) (h : x^2 + 2*y^2 + 5*z^2 = 22) :
  ∃ (m : ℝ), m = xy - yz - zx ∧ m ≥ (-55 - 11*Real.sqrt 5) / 10 ∧
  (∃ (x' y' z' : ℝ), x'^2 + 2*y'^2 + 5*z'^2 = 22 ∧
    x'*y' - y'*z' - z'*x' = (-55 - 11*Real.sqrt 5) / 10) :=
sorry

end min_value_of_sum_l3525_352578


namespace arithmetic_mean_of_numbers_l3525_352594

def numbers : List ℕ := [18, 24, 42]

theorem arithmetic_mean_of_numbers :
  (numbers.sum / numbers.length : ℚ) = 28 := by sorry

end arithmetic_mean_of_numbers_l3525_352594


namespace smallest_n_congruence_l3525_352569

theorem smallest_n_congruence (n : ℕ) : 
  (∀ k < n, ¬(5^k ≡ k^5 [ZMOD 3])) ∧ (5^n ≡ n^5 [ZMOD 3]) ↔ n = 4 :=
sorry

end smallest_n_congruence_l3525_352569


namespace fencing_required_l3525_352558

theorem fencing_required (area : ℝ) (uncovered_side : ℝ) : 
  area = 560 ∧ uncovered_side = 20 → 
  ∃ (width : ℝ), 
    area = uncovered_side * width ∧ 
    2 * width + uncovered_side = 76 := by
  sorry

end fencing_required_l3525_352558


namespace roberto_outfits_l3525_352543

/-- The number of trousers Roberto has -/
def num_trousers : ℕ := 5

/-- The number of shirts Roberto has -/
def num_shirts : ℕ := 6

/-- The number of jackets Roberto has -/
def num_jackets : ℕ := 4

/-- The number of restricted outfits (combinations of the specific shirt and jacket that can't be worn together) -/
def num_restricted : ℕ := 1 * 1 * num_trousers

/-- The total number of possible outfits without restrictions -/
def total_outfits : ℕ := num_trousers * num_shirts * num_jackets

/-- The number of permissible outfits Roberto can put together -/
def permissible_outfits : ℕ := total_outfits - num_restricted

theorem roberto_outfits : permissible_outfits = 115 := by
  sorry

end roberto_outfits_l3525_352543


namespace parallel_vectors_x_value_l3525_352562

/-- Given two vectors a and b in ℝ², where a = (1,2) and b = (2x,-3),
    if a is parallel to b, then x = -3/4 -/
theorem parallel_vectors_x_value :
  let a : ℝ × ℝ := (1, 2)
  let b : ℝ × ℝ := (2*x, -3)
  (∃ (k : ℝ), k ≠ 0 ∧ a.1 * k = b.1 ∧ a.2 * k = b.2) →
  x = -3/4 := by
  sorry

end parallel_vectors_x_value_l3525_352562


namespace journey_speed_l3525_352523

/-- Proves that given a journey where 75% is traveled at 50 mph and 25% at S mph,
    if the average speed for the entire journey is 50 mph, then S must equal 50 mph. -/
theorem journey_speed (D : ℝ) (S : ℝ) (h1 : D > 0) :
  (D / ((0.75 * D / 50) + (0.25 * D / S)) = 50) → S = 50 := by
  sorry

end journey_speed_l3525_352523


namespace square_transformation_2007_l3525_352592

-- Define the vertex order as a list of characters
def VertexOrder := List Char

-- Define the transformation operations
def rotate90Clockwise (order : VertexOrder) : VertexOrder :=
  match order with
  | [a, b, c, d] => [d, a, b, c]
  | _ => order

def reflectVertical (order : VertexOrder) : VertexOrder :=
  match order with
  | [a, b, c, d] => [d, c, b, a]
  | _ => order

def reflectHorizontal (order : VertexOrder) : VertexOrder :=
  match order with
  | [a, b, c, d] => [c, b, a, d]
  | _ => order

-- Define the complete transformation sequence
def transformSequence (order : VertexOrder) : VertexOrder :=
  reflectHorizontal (reflectVertical (rotate90Clockwise order))

-- Define a function to apply the transformation sequence n times
def applyTransformSequence (order : VertexOrder) (n : Nat) : VertexOrder :=
  match n with
  | 0 => order
  | n + 1 => applyTransformSequence (transformSequence order) n

-- Theorem statement
theorem square_transformation_2007 :
  applyTransformSequence ['A', 'B', 'C', 'D'] 2007 = ['D', 'C', 'B', 'A'] := by
  sorry


end square_transformation_2007_l3525_352592


namespace circle_equation_from_diameter_endpoints_l3525_352561

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- The equation of a circle -/
def CircleEquation (center : Point) (radius : ℝ) (x y : ℝ) : Prop :=
  (x - center.x)^2 + (y - center.y)^2 = radius^2

/-- Theorem: The equation of the circle with diameter endpoints A(1,4) and B(3,-2) -/
theorem circle_equation_from_diameter_endpoints :
  let A : Point := ⟨1, 4⟩
  let B : Point := ⟨3, -2⟩
  let center : Point := ⟨(A.x + B.x) / 2, (A.y + B.y) / 2⟩
  let radius : ℝ := Real.sqrt ((B.x - center.x)^2 + (B.y - center.y)^2)
  CircleEquation center radius = fun x y ↦ (x - 2)^2 + (y - 1)^2 = 10 := by
  sorry

end circle_equation_from_diameter_endpoints_l3525_352561


namespace solution_set_inequality_empty_solution_set_l3525_352519

-- Part 1
theorem solution_set_inequality (x : ℝ) :
  (x - 3) / (x + 7) < 0 ↔ -7 < x ∧ x < 3 := by sorry

-- Part 2
theorem empty_solution_set (a : ℝ) :
  (∀ x : ℝ, x^2 - 4*a*x + 4*a^2 + a > 0) ↔ a > 0 := by sorry

end solution_set_inequality_empty_solution_set_l3525_352519


namespace laundry_loads_required_l3525_352511

def num_families : ℕ := 3
def people_per_family : ℕ := 4
def vacation_days : ℕ := 7
def towels_per_person_per_day : ℕ := 1
def washing_machine_capacity : ℕ := 14

def total_people : ℕ := num_families * people_per_family
def total_towels : ℕ := total_people * vacation_days * towels_per_person_per_day

theorem laundry_loads_required :
  (total_towels + washing_machine_capacity - 1) / washing_machine_capacity = 6 := by
  sorry

end laundry_loads_required_l3525_352511


namespace second_meeting_day_correct_l3525_352528

/-- Represents the number of days between visits for each schoolchild -/
def VisitSchedule : Fin 4 → ℕ
  | 0 => 4
  | 1 => 5
  | 2 => 6
  | 3 => 9

/-- The day when all schoolchildren meet for the second time -/
def SecondMeetingDay : ℕ := 360

theorem second_meeting_day_correct :
  SecondMeetingDay = 2 * Nat.lcm (VisitSchedule 0) (Nat.lcm (VisitSchedule 1) (Nat.lcm (VisitSchedule 2) (VisitSchedule 3))) :=
by sorry

#check second_meeting_day_correct

end second_meeting_day_correct_l3525_352528


namespace odd_sum_15_to_55_l3525_352598

theorem odd_sum_15_to_55 : 
  let a₁ : ℕ := 15  -- First term
  let d : ℕ := 4    -- Common difference
  let n : ℕ := (55 - 15) / d + 1  -- Number of terms
  let aₙ : ℕ := a₁ + (n - 1) * d  -- Last term
  (n : ℝ) / 2 * (a₁ + aₙ) = 385 :=
by sorry

end odd_sum_15_to_55_l3525_352598


namespace rectangle_midpoint_distances_l3525_352541

theorem rectangle_midpoint_distances (a b : ℝ) (ha : a = 3) (hb : b = 4) :
  let midpoint_distance (x y : ℝ) := Real.sqrt (x^2 + y^2)
  (midpoint_distance (a/2) 0) + (midpoint_distance a (b/2)) +
  (midpoint_distance (a/2) b) + (midpoint_distance 0 (b/2)) =
  3.5 + Real.sqrt 13 + Real.sqrt 18.25 := by
  sorry

end rectangle_midpoint_distances_l3525_352541


namespace ram_gopal_ratio_l3525_352530

theorem ram_gopal_ratio (ram_money : ℕ) (krishan_money : ℕ) (gopal_krishan_ratio : Rat) :
  ram_money = 735 →
  krishan_money = 4335 →
  gopal_krishan_ratio = 7 / 17 →
  (ram_money : Rat) / ((gopal_krishan_ratio * krishan_money) : Rat) = 7 / 17 := by
  sorry

end ram_gopal_ratio_l3525_352530


namespace correct_sampling_methods_l3525_352532

/-- Represents different sampling methods -/
inductive SamplingMethod
| SimpleRandom
| Systematic
| Stratified

/-- Represents a population with different income levels -/
structure Population :=
  (total : ℕ)
  (high_income : ℕ)
  (middle_income : ℕ)
  (low_income : ℕ)

/-- Represents a sampling problem -/
structure SamplingProblem :=
  (population : Population)
  (sample_size : ℕ)

/-- Determines the best sampling method for a given problem -/
def best_sampling_method (problem : SamplingProblem) : SamplingMethod :=
  sorry

/-- The community population for problem 1 -/
def community : Population :=
  { total := 600
  , high_income := 100
  , middle_income := 380
  , low_income := 120 }

/-- Problem 1: Family income study -/
def problem1 : SamplingProblem :=
  { population := community
  , sample_size := 100 }

/-- Problem 2: Student seminar selection -/
def problem2 : SamplingProblem :=
  { population := { total := 15, high_income := 0, middle_income := 0, low_income := 0 }
  , sample_size := 3 }

theorem correct_sampling_methods :
  (best_sampling_method problem1 = SamplingMethod.Stratified) ∧
  (best_sampling_method problem2 = SamplingMethod.SimpleRandom) :=
sorry

end correct_sampling_methods_l3525_352532


namespace sum_of_cubes_remainder_l3525_352557

def sum_of_cubes (n : ℕ) : ℕ := (n * (n + 1) / 2) ^ 2

def b : ℕ := 2

theorem sum_of_cubes_remainder (n : ℕ) (h : n = 2010) : 
  sum_of_cubes n % (b ^ 2) = 1 := by
  sorry

end sum_of_cubes_remainder_l3525_352557


namespace wire_length_between_poles_l3525_352585

/-- Given two vertical poles on flat ground with a distance of 20 feet between their bases
    and a height difference of 10 feet, the length of a wire stretched between their tops
    is 10√5 feet. -/
theorem wire_length_between_poles (distance : ℝ) (height_diff : ℝ) :
  distance = 20 → height_diff = 10 → 
  ∃ (wire_length : ℝ), wire_length = 10 * Real.sqrt 5 ∧ 
  wire_length ^ 2 = distance ^ 2 + height_diff ^ 2 :=
by sorry

end wire_length_between_poles_l3525_352585


namespace expression_simplification_l3525_352559

theorem expression_simplification (a : ℝ) (h1 : a ≠ 0) (h2 : a ≠ 2) (h3 : a ≠ -2) :
  ((a / (a - 2) - a / (a^2 - 2*a)) / (a + 2) * a) = (a^2 - a) / (a^2 - 4) :=
by sorry

end expression_simplification_l3525_352559


namespace bookArrangements_eq_48_l3525_352597

/-- The number of ways to arrange 3 different math books and 2 different Chinese books in a row,
    with the Chinese books placed next to each other. -/
def bookArrangements : ℕ :=
  (Nat.factorial 4) * (Nat.factorial 2)

/-- The total number of arrangements is 48. -/
theorem bookArrangements_eq_48 : bookArrangements = 48 := by
  sorry

end bookArrangements_eq_48_l3525_352597


namespace repeating_decimal_sum_l3525_352527

/-- The sum of 0.222... and 0.0202... equals 8/33 -/
theorem repeating_decimal_sum : 
  let a : ℚ := 2/9  -- represents 0.222...
  let b : ℚ := 2/99 -- represents 0.0202...
  a + b = 8/33 := by sorry

end repeating_decimal_sum_l3525_352527


namespace percentage_decrease_l3525_352545

theorem percentage_decrease (x y z : ℝ) 
  (h1 : x = 1.2 * y) 
  (h2 : x = 0.6 * z) : 
  y = 0.5 * z := by
  sorry

end percentage_decrease_l3525_352545


namespace cloth_selling_price_l3525_352596

/-- Represents the selling price calculation for cloth --/
def total_selling_price (metres : ℕ) (cost_price : ℕ) (loss : ℕ) : ℕ :=
  metres * (cost_price - loss)

/-- Theorem stating the total selling price for the given conditions --/
theorem cloth_selling_price :
  total_selling_price 300 65 5 = 18000 := by
  sorry

end cloth_selling_price_l3525_352596


namespace factor_to_increase_average_l3525_352567

theorem factor_to_increase_average (numbers : Finset ℝ) (factor : ℝ) : 
  Finset.card numbers = 5 →
  6 ∈ numbers →
  (Finset.sum numbers id) / 5 = 6.8 →
  ((Finset.sum numbers id) - 6 + 6 * factor) / 5 = 9.2 →
  factor = 3 := by
  sorry

end factor_to_increase_average_l3525_352567


namespace friends_recycled_23_pounds_l3525_352544

/-- Represents the recycling scenario with Zoe and her friends -/
structure RecyclingScenario where
  pointsPerEightPounds : Nat
  zoeRecycled : Nat
  totalPoints : Nat

/-- Calculates the number of pounds Zoe's friends recycled -/
def friendsRecycled (scenario : RecyclingScenario) : Nat :=
  scenario.totalPoints * 8 - scenario.zoeRecycled

/-- Theorem stating that Zoe's friends recycled 23 pounds -/
theorem friends_recycled_23_pounds (scenario : RecyclingScenario)
  (h1 : scenario.pointsPerEightPounds = 1)
  (h2 : scenario.zoeRecycled = 25)
  (h3 : scenario.totalPoints = 6) :
  friendsRecycled scenario = 23 := by
  sorry

#eval friendsRecycled ⟨1, 25, 6⟩

end friends_recycled_23_pounds_l3525_352544


namespace jones_elementary_population_l3525_352539

theorem jones_elementary_population :
  ∀ (total_students : ℕ) (boys_percentage : ℚ),
    (90 : ℚ) = boys_percentage * ((20 : ℚ) / 100) * total_students →
    total_students = 450 := by
  sorry

end jones_elementary_population_l3525_352539


namespace mod_equivalence_unique_l3525_352501

theorem mod_equivalence_unique : ∃! n : ℕ, 0 ≤ n ∧ n ≤ 5 ∧ n ≡ -1723 [ZMOD 6] := by
  sorry

end mod_equivalence_unique_l3525_352501


namespace largest_number_l3525_352549

-- Define the numbers as real numbers
def A : ℝ := 8.03456
def B : ℝ := 8.034666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666
def C : ℝ := 8.034545454545454545454545454545454545454545454545454545454545454545454545454545454545454545454545454545
def D : ℝ := 8.034563456345634563456345634563456345634563456345634563456345634563456345634563456345634563456345634563456
def E : ℝ := 8.034560345603456034560345603456034560345603456034560345603456034560345603456034560345603456034560345603456

-- Theorem statement
theorem largest_number : B > A ∧ B > C ∧ B > D ∧ B > E := by sorry

end largest_number_l3525_352549


namespace parallelogram_above_x_axis_ratio_l3525_352524

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a parallelogram defined by four points -/
structure Parallelogram where
  P : Point
  Q : Point
  R : Point
  S : Point

/-- Calculates the area of a parallelogram -/
def parallelogramArea (p : Parallelogram) : ℝ := sorry

/-- Calculates the area of the part of the parallelogram above the x-axis -/
def areaAboveXAxis (p : Parallelogram) : ℝ := sorry

/-- The main theorem to be proved -/
theorem parallelogram_above_x_axis_ratio 
  (p : Parallelogram) 
  (h1 : p.P = ⟨-1, 1⟩) 
  (h2 : p.Q = ⟨3, -5⟩) 
  (h3 : p.R = ⟨1, -3⟩) 
  (h4 : p.S = ⟨-3, 3⟩) : 
  areaAboveXAxis p / parallelogramArea p = 1/4 := by sorry

end parallelogram_above_x_axis_ratio_l3525_352524


namespace factorization_identity_l3525_352573

theorem factorization_identity (x y : ℝ) : x^2 - 2*x*y + y^2 - 1 = (x - y + 1) * (x - y - 1) := by
  sorry

end factorization_identity_l3525_352573


namespace certain_number_proof_l3525_352583

theorem certain_number_proof (x : ℝ) : 
  (0.15 * x > 0.25 * 16 + 2) → (0.15 * x = 6) := by
  sorry

end certain_number_proof_l3525_352583


namespace dehydrated_men_fraction_l3525_352505

theorem dehydrated_men_fraction (total_men : ℕ) (finished_men : ℕ) 
  (h1 : total_men = 80)
  (h2 : finished_men = 52)
  (h3 : (1 : ℚ) / 4 * total_men = total_men - (total_men - (1 : ℚ) / 4 * total_men))
  (h4 : ∃ x : ℚ, x * (total_men - (1 : ℚ) / 4 * total_men) * (1 : ℚ) / 5 = 
    total_men - finished_men - (1 : ℚ) / 4 * total_men) :
  ∃ x : ℚ, x = 2 / 3 ∧ 
    x * (total_men - (1 : ℚ) / 4 * total_men) * (1 : ℚ) / 5 = 
    total_men - finished_men - (1 : ℚ) / 4 * total_men :=
by sorry


end dehydrated_men_fraction_l3525_352505


namespace log_inequality_condition_l3525_352516

theorem log_inequality_condition (a b : ℝ) : 
  (∀ a b, Real.log a > Real.log b → a > b) ∧ 
  (∃ a b, a > b ∧ ¬(Real.log a > Real.log b)) := by
  sorry

end log_inequality_condition_l3525_352516


namespace binary_sum_to_octal_to_decimal_l3525_352537

/-- Converts a binary number represented as a list of bits to its decimal equivalent -/
def binary_to_decimal (bits : List Bool) : ℕ :=
  bits.enum.foldl (fun acc (i, b) => acc + if b then 2^i else 0) 0

/-- Converts a decimal number to its octal representation -/
def decimal_to_octal (n : ℕ) : List ℕ :=
  if n = 0 then [0] else
    let rec aux (m : ℕ) (acc : List ℕ) :=
      if m = 0 then acc else aux (m / 8) ((m % 8) :: acc)
    aux n []

/-- Converts an octal number represented as a list of digits to its decimal equivalent -/
def octal_to_decimal (digits : List ℕ) : ℕ :=
  digits.enum.foldl (fun acc (i, d) => acc + d * 8^(digits.length - 1 - i)) 0

/-- The main theorem to be proved -/
theorem binary_sum_to_octal_to_decimal : 
  let a := binary_to_decimal [true, true, true, true, true, true, true, true]
  let b := binary_to_decimal [true, true, true, true, true]
  let sum := a + b
  let octal := decimal_to_octal sum
  octal_to_decimal octal = 286 := by
  sorry

end binary_sum_to_octal_to_decimal_l3525_352537


namespace scale_tower_height_l3525_352513

/-- Given a cylindrical tower and its scaled-down model, calculates the height of the model. -/
theorem scale_tower_height (actual_height : ℝ) (actual_volume : ℝ) (model_volume : ℝ) 
  (h1 : actual_height = 60) 
  (h2 : actual_volume = 80000)
  (h3 : model_volume = 0.5) :
  actual_height / Real.sqrt (actual_volume / model_volume) = 0.15 := by
  sorry

end scale_tower_height_l3525_352513


namespace paper_piles_problem_l3525_352502

theorem paper_piles_problem :
  ∃! N : ℕ,
    1000 < N ∧ N < 2000 ∧
    N % 2 = 1 ∧
    N % 3 = 1 ∧
    N % 4 = 1 ∧
    N % 5 = 1 ∧
    N % 6 = 1 ∧
    N % 7 = 1 ∧
    N % 8 = 1 ∧
    N % 41 = 0 :=
by sorry

end paper_piles_problem_l3525_352502


namespace polynomial_remainder_theorem_l3525_352504

def f (x : ℝ) : ℝ := 6*x^3 - 15*x^2 + 21*x - 23

theorem polynomial_remainder_theorem :
  ∃ (q : ℝ → ℝ), f = λ x => (3*x - 6) * q x + 7 :=
sorry

end polynomial_remainder_theorem_l3525_352504


namespace f_derivative_at_one_l3525_352595

/-- The function f(x) = x^3 - 2x^2 + 5x - 1 -/
def f (x : ℝ) : ℝ := x^3 - 2*x^2 + 5*x - 1

/-- The derivative of f -/
def f' (x : ℝ) : ℝ := 3*x^2 - 4*x + 5

theorem f_derivative_at_one : f' 1 = 4 := by sorry

end f_derivative_at_one_l3525_352595


namespace dividend_calculation_l3525_352508

theorem dividend_calculation (divisor quotient remainder : ℕ) 
  (h1 : divisor = 10 * quotient)
  (h2 : divisor = 5 * remainder)
  (h3 : remainder = 46) :
  divisor * quotient + remainder = 5336 := by
sorry

end dividend_calculation_l3525_352508


namespace intersection_complement_theorem_l3525_352529

def M : Set ℝ := {x | 1 ≤ x ∧ x ≤ 3}
def N : Set ℝ := {x | x > 2}

theorem intersection_complement_theorem :
  M ∩ (Nᶜ) = {x : ℝ | 1 ≤ x ∧ x ≤ 2} := by sorry

end intersection_complement_theorem_l3525_352529


namespace not_square_p_cubed_plus_p_plus_one_l3525_352599

theorem not_square_p_cubed_plus_p_plus_one (p : ℕ) (hp : Prime p) :
  ¬ ∃ (n : ℕ), n^2 = p^3 + p + 1 := by
  sorry

end not_square_p_cubed_plus_p_plus_one_l3525_352599


namespace league_teams_count_l3525_352571

/-- The number of games in a league where each team plays every other team once -/
def numGames (n : ℕ) : ℕ := n * (n - 1) / 2

/-- Theorem: In a league where each team plays every other team exactly once, 
    if the total number of games played is 36, then the number of teams in the league is 9 -/
theorem league_teams_count : ∃ (n : ℕ), n > 0 ∧ numGames n = 36 → n = 9 := by
  sorry

end league_teams_count_l3525_352571


namespace lent_amount_proof_l3525_352552

/-- The amount of money (in Rs.) that A lends to B -/
def lent_amount : ℝ := 1500

/-- The interest rate difference (in decimal form) between B's lending and borrowing rates -/
def interest_rate_diff : ℝ := 0.015

/-- The number of years for which the loan is considered -/
def years : ℝ := 3

/-- B's total gain (in Rs.) over the loan period -/
def total_gain : ℝ := 67.5

theorem lent_amount_proof :
  lent_amount * interest_rate_diff * years = total_gain :=
by sorry

end lent_amount_proof_l3525_352552


namespace sixth_term_of_geometric_sequence_l3525_352542

/-- A geometric sequence of positive integers -/
def GeometricSequence (a : ℕ+) (r : ℕ+) : ℕ → ℕ+
  | 0 => a
  | n + 1 => r * GeometricSequence a r n

theorem sixth_term_of_geometric_sequence 
  (a : ℕ+) (r : ℕ+) :
  a = 3 →
  GeometricSequence a r 4 = 243 →
  GeometricSequence a r 5 = 729 := by
sorry

end sixth_term_of_geometric_sequence_l3525_352542


namespace sandy_puppies_l3525_352507

def puppies_problem (initial_puppies : ℕ) (initial_spotted : ℕ) (new_puppies : ℕ) (new_spotted : ℕ) (given_away : ℕ) : Prop :=
  let initial_non_spotted := initial_puppies - initial_spotted
  let total_spotted := initial_spotted + new_spotted
  let total_non_spotted := initial_non_spotted + (new_puppies - new_spotted) - given_away
  let final_puppies := total_spotted + total_non_spotted
  final_puppies = 9

theorem sandy_puppies : puppies_problem 8 3 4 2 3 :=
by sorry

end sandy_puppies_l3525_352507


namespace digit_sum_at_positions_l3525_352533

def sequence_generator (n : ℕ) : ℕ :=
  (n - 1) % 6 + 1

def remove_nth (n : ℕ) (seq : ℕ → ℕ) : ℕ → ℕ :=
  Function.comp seq (λ m => m + m / (n - 1))

def final_sequence : ℕ → ℕ :=
  remove_nth 7 (remove_nth 5 sequence_generator)

theorem digit_sum_at_positions : 
  final_sequence 3031 + final_sequence 3032 + final_sequence 3033 = 9 := by
  sorry

end digit_sum_at_positions_l3525_352533


namespace cos_sum_of_complex_exponentials_l3525_352521

theorem cos_sum_of_complex_exponentials (θ φ : ℝ) :
  Complex.exp (θ * I) = 4/5 + 3/5 * I →
  Complex.exp (φ * I) = -5/13 + 12/13 * I →
  Real.cos (θ + φ) = -1/13 := by
  sorry

end cos_sum_of_complex_exponentials_l3525_352521


namespace student_arrangement_theorem_l3525_352518

/-- The number of ways to arrange n students in a row -/
def arrange (n : ℕ) : ℕ := Nat.factorial n

/-- The number of ways to arrange n students in a row with 2 specific students not at the ends -/
def arrangeNotAtEnds (n : ℕ) : ℕ :=
  arrange (n - 2) * (arrange (n - 3))

/-- The number of ways to arrange n students in a row with 2 specific students adjacent -/
def arrangeAdjacent (n : ℕ) : ℕ :=
  2 * arrange (n - 1)

/-- The number of ways to arrange n students in a row with 2 specific students not adjacent -/
def arrangeNotAdjacent (n : ℕ) : ℕ :=
  arrange n - arrangeAdjacent n

/-- The main theorem -/
theorem student_arrangement_theorem :
  (arrangeNotAtEnds 5 = 36) ∧
  (arrangeAdjacent 5 * arrangeNotAdjacent 3 = 24) :=
by sorry

end student_arrangement_theorem_l3525_352518


namespace probability_through_C_and_D_l3525_352525

/-- Represents the number of eastward and southward moves between two intersections -/
structure Moves where
  east : Nat
  south : Nat

/-- Calculates the number of possible paths given a number of eastward and southward moves -/
def pathCount (m : Moves) : Nat :=
  Nat.choose (m.east + m.south) m.east

/-- The moves from A to C -/
def movesAC : Moves := ⟨3, 2⟩

/-- The moves from C to D -/
def movesCD : Moves := ⟨2, 1⟩

/-- The moves from D to B -/
def movesDB : Moves := ⟨1, 2⟩

/-- The total moves from A to B -/
def movesAB : Moves := ⟨movesAC.east + movesCD.east + movesDB.east, movesAC.south + movesCD.south + movesDB.south⟩

/-- The probability of choosing a specific path at each intersection -/
def pathProbability (m : Moves) : Rat :=
  1 / (2 ^ (m.east + m.south))

theorem probability_through_C_and_D :
  (pathCount movesAC * pathCount movesCD * pathCount movesDB : Rat) /
  (pathCount movesAB : Rat) = 15 / 77 := by sorry

end probability_through_C_and_D_l3525_352525


namespace special_numbers_count_l3525_352512

def count_multiples (n : ℕ) (max : ℕ) : ℕ :=
  (max / n : ℕ)

def count_special_numbers (max : ℕ) : ℕ :=
  count_multiples 4 max + count_multiples 5 max - count_multiples 20 max - count_multiples 25 max

theorem special_numbers_count :
  count_special_numbers 3000 = 1080 := by sorry

end special_numbers_count_l3525_352512


namespace binary_10101_equals_21_l3525_352580

def binary_to_decimal (binary : List Bool) : Nat :=
  binary.enum.foldl (fun acc (i, b) => acc + if b then 2^i else 0) 0

theorem binary_10101_equals_21 :
  binary_to_decimal [true, false, true, false, true] = 21 := by
  sorry

end binary_10101_equals_21_l3525_352580


namespace nested_square_root_value_l3525_352546

theorem nested_square_root_value :
  ∃ x : ℝ, x = Real.sqrt (3 - x) ∧ x = (-1 + Real.sqrt 13) / 2 := by
  sorry

end nested_square_root_value_l3525_352546


namespace sequence_length_l3525_352538

-- Define the arithmetic sequence
def arithmetic_sequence (a₁ : ℝ) (d : ℝ) (n : ℕ) : ℝ := a₁ + (n - 1 : ℝ) * d

-- Theorem statement
theorem sequence_length :
  let a₁ : ℝ := 2.5
  let d : ℝ := 5
  let aₙ : ℝ := 62.5
  ∃ n : ℕ, n > 0 ∧ arithmetic_sequence a₁ d n = aₙ ∧ n = 13 := by
  sorry

end sequence_length_l3525_352538


namespace square_area_problem_l3525_352576

theorem square_area_problem (s₁ s₂ s₃ s₄ s₅ : ℝ) (h₁ : s₁ = 3) (h₂ : s₂ = 7) (h₃ : s₃ = 22) :
  s₁ + s₂ + s₃ + s₄ + s₅ = s₃ + s₅ → s₄ = 18 :=
by sorry

end square_area_problem_l3525_352576


namespace quadratic_inequality_solutions_l3525_352564

/-- The quadratic inequality kx^2 - 2x + 6k < 0 -/
def quadratic_inequality (k : ℝ) (x : ℝ) : Prop := k * x^2 - 2 * x + 6 * k < 0

/-- The solution set for case 1: x < -3 or x > -2 -/
def solution_set_1 (x : ℝ) : Prop := x < -3 ∨ x > -2

/-- The solution set for case 2: all real numbers -/
def solution_set_2 (x : ℝ) : Prop := True

/-- The solution set for case 3: empty set -/
def solution_set_3 (x : ℝ) : Prop := False

theorem quadratic_inequality_solutions (k : ℝ) (h : k ≠ 0) :
  (∀ x, quadratic_inequality k x ↔ solution_set_1 x) → k = -2/5 ∧
  (∀ x, quadratic_inequality k x ↔ solution_set_2 x) → k < -Real.sqrt 6 / 6 ∧
  (∀ x, quadratic_inequality k x ↔ solution_set_3 x) → k ≥ Real.sqrt 6 / 6 :=
sorry

end quadratic_inequality_solutions_l3525_352564


namespace three_integer_chords_l3525_352553

/-- Represents a circle with a given radius and a point inside it. -/
structure CircleWithPoint where
  radius : ℝ
  distanceFromCenter : ℝ

/-- Counts the number of chords with integer lengths that contain the given point. -/
def countIntegerChords (c : CircleWithPoint) : ℕ :=
  sorry

/-- The main theorem stating that for a circle with radius 13 and a point 5 units from the center,
    there are exactly 3 chords with integer lengths containing the point. -/
theorem three_integer_chords :
  let c := CircleWithPoint.mk 13 5
  countIntegerChords c = 3 := by
  sorry

end three_integer_chords_l3525_352553


namespace marble_jar_problem_l3525_352506

theorem marble_jar_problem (num_marbles : ℕ) : 
  (∀ (x : ℚ), x = num_marbles / 20 → 
    x - 1 = num_marbles / 22) → 
  num_marbles = 220 := by
  sorry

end marble_jar_problem_l3525_352506


namespace chain_breaking_theorem_l3525_352536

/-- Represents a chain with n links -/
structure Chain (n : ℕ) where
  links : Fin n → ℕ
  all_links_one : ∀ i, links i = 1

/-- Represents a set of chain segments after breaking -/
structure Segments (n : ℕ) where
  pieces : List ℕ
  sum_pieces : pieces.sum = n

/-- Function to break a chain into segments -/
def break_chain (n : ℕ) (k : ℕ) (break_points : Fin (k-1) → ℕ) : Segments n :=
  sorry

/-- Function to check if a weight can be measured using given segments -/
def can_measure (segments : List ℕ) (weight : ℕ) : Prop :=
  sorry

theorem chain_breaking_theorem (k : ℕ) :
  let n := k * 2^k - 1
  ∃ (break_points : Fin (k-1) → ℕ),
    let segments := (break_chain n k break_points).pieces
    ∀ w : ℕ, w ≤ n → can_measure segments w :=
  sorry

end chain_breaking_theorem_l3525_352536


namespace perpendicular_transitivity_l3525_352547

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the perpendicular relation between lines and planes
variable (perp : Line → Plane → Prop)

-- Define the parallel relation between lines
variable (parallel : Line → Line → Prop)

-- Theorem statement
theorem perpendicular_transitivity 
  (m n : Line) (α β : Plane) 
  (h1 : perp m β) 
  (h2 : perp n β) 
  (h3 : perp n α) : 
  perp m α :=
sorry

end perpendicular_transitivity_l3525_352547


namespace cat_shortest_distance_to_origin_l3525_352515

theorem cat_shortest_distance_to_origin :
  let center : ℝ × ℝ := (5, -2)
  let radius : ℝ := 8
  let origin : ℝ × ℝ := (0, 0)
  let distance_center_to_origin : ℝ := Real.sqrt ((center.1 - origin.1)^2 + (center.2 - origin.2)^2)
  ∀ p : ℝ × ℝ, (p.1 - center.1)^2 + (p.2 - center.2)^2 = radius^2 →
    Real.sqrt ((p.1 - origin.1)^2 + (p.2 - origin.2)^2) ≥ |distance_center_to_origin - radius| :=
by sorry

end cat_shortest_distance_to_origin_l3525_352515


namespace diamond_equation_solution_l3525_352590

-- Define the diamond operation
noncomputable def diamond (a b : ℝ) : ℝ := a^2 + Real.sqrt (b + Real.sqrt (b + Real.sqrt b))

-- Theorem statement
theorem diamond_equation_solution :
  ∃ h : ℝ, diamond 3 h = 12 ∧ h = 6 := by sorry

end diamond_equation_solution_l3525_352590


namespace hiker_rate_ratio_l3525_352535

/-- Proves that the ratio of the rate down to the rate up is 1.5 given the hiking conditions --/
theorem hiker_rate_ratio 
  (rate_up : ℝ) 
  (time_up : ℝ) 
  (distance_down : ℝ) 
  (h1 : rate_up = 3) 
  (h2 : time_up = 2) 
  (h3 : distance_down = 9) 
  (h4 : time_up = distance_down / (distance_down / time_up)) : 
  (distance_down / time_up) / rate_up = 1.5 := by
  sorry

end hiker_rate_ratio_l3525_352535


namespace sequence_theorem_l3525_352551

/-- A positive sequence satisfying the given condition -/
def PositiveSequence (a : ℕ+ → ℝ) : Prop :=
  ∀ n : ℕ+, a n > 0

/-- The sum of the first n terms of the sequence -/
def S (a : ℕ+ → ℝ) (n : ℕ+) : ℝ :=
  (Finset.range n).sum (fun i => a ⟨i + 1, Nat.succ_pos i⟩)

/-- The main theorem -/
theorem sequence_theorem (a : ℕ+ → ℝ) (h_pos : PositiveSequence a)
    (h_cond : ∀ n : ℕ+, 2 * S a n = a n ^ 2 + a n) :
    ∀ n : ℕ+, a n = n := by
  sorry

end sequence_theorem_l3525_352551


namespace right_triangle_angles_l3525_352522

theorem right_triangle_angles (α β : Real) : 
  α > 0 → β > 0 → α + β = π / 2 →
  Real.tan α + Real.tan β + (Real.tan α)^2 + (Real.tan β)^2 + (Real.tan α)^3 + (Real.tan β)^3 = 70 →
  α = π / 2.4 ∧ β = π / 12 := by
sorry

end right_triangle_angles_l3525_352522


namespace cupboard_cost_price_l3525_352565

theorem cupboard_cost_price (C : ℝ) : C = 7450 :=
  let selling_price := C * 0.86
  let profitable_price := C * 1.14
  have h1 : profitable_price = selling_price + 2086 := by sorry
  sorry

end cupboard_cost_price_l3525_352565


namespace similar_triangle_perimeter_similar_triangle_perimeter_proof_l3525_352575

/-- Given two similar triangles where the smaller triangle has sides 15, 15, and 24,
    and the larger triangle has its longest side measuring 72,
    the perimeter of the larger triangle is 162. -/
theorem similar_triangle_perimeter : ℝ → ℝ → ℝ → ℝ → ℝ → Prop :=
  fun a b c d p =>
    (a = 15 ∧ b = 15 ∧ c = 24) →  -- Dimensions of smaller triangle
    (d = 72) →                    -- Longest side of larger triangle
    (d / c = b / a) →             -- Triangles are similar
    (p = 3 * a + d) →             -- Perimeter of larger triangle
    p = 162

theorem similar_triangle_perimeter_proof : similar_triangle_perimeter 15 15 24 72 162 := by
  sorry

end similar_triangle_perimeter_similar_triangle_perimeter_proof_l3525_352575


namespace prime_pairs_divisibility_l3525_352526

theorem prime_pairs_divisibility (p q : ℕ) : 
  Prime p ∧ Prime q ∧ 
  (p^2 ∣ q^3 + 1) ∧ 
  (q^2 ∣ p^6 - 1) ↔ 
  ((p = 3 ∧ q = 2) ∨ (p = 2 ∧ q = 3)) :=
by sorry

end prime_pairs_divisibility_l3525_352526


namespace ab_multiplier_l3525_352581

theorem ab_multiplier (a b m : ℝ) : 
  4 * a = 30 ∧ 5 * b = 30 ∧ m * (a * b) = 1800 → m = 40 := by
  sorry

end ab_multiplier_l3525_352581


namespace repeating_decimal_sum_l3525_352556

/-- Represents a repeating decimal with a single digit repeating -/
def SingleDigitRepeatingDecimal (whole : ℚ) (repeating : ℕ) : ℚ :=
  whole + repeating / 9

/-- Represents a repeating decimal with two digits repeating -/
def TwoDigitRepeatingDecimal (whole : ℚ) (repeating : ℕ) : ℚ :=
  whole + repeating / 99

theorem repeating_decimal_sum :
  SingleDigitRepeatingDecimal 0 3 + TwoDigitRepeatingDecimal 0 6 = 13 / 33 := by
  sorry

end repeating_decimal_sum_l3525_352556


namespace equation_represents_two_lines_l3525_352509

/-- The equation represents two lines -/
theorem equation_represents_two_lines :
  ∃ (a b c d : ℝ), a ≠ c ∧ b ≠ d ∧
  ∀ (x y : ℝ), x^2 - 50*y^2 - 10*x + 25 = 0 ↔ 
  ((x = a*y + b) ∨ (x = c*y + d)) :=
sorry

end equation_represents_two_lines_l3525_352509


namespace solid_is_cone_l3525_352572

-- Define the properties of the solid
structure Solid where
  front_view_isosceles : Bool
  left_view_isosceles : Bool
  top_view_circle_with_center : Bool

-- Define what it means for a solid to be a cone
def is_cone (s : Solid) : Prop :=
  s.front_view_isosceles ∧ s.left_view_isosceles ∧ s.top_view_circle_with_center

-- Theorem statement
theorem solid_is_cone (s : Solid) 
  (h1 : s.front_view_isosceles = true) 
  (h2 : s.left_view_isosceles = true) 
  (h3 : s.top_view_circle_with_center = true) : 
  is_cone s := by sorry

end solid_is_cone_l3525_352572


namespace simplify_sqrt_plus_x_l3525_352548

theorem simplify_sqrt_plus_x (x : ℝ) (h : 1 < x ∧ x < 2) : 
  Real.sqrt ((x - 2)^2) + x = 2 := by sorry

end simplify_sqrt_plus_x_l3525_352548


namespace necklace_length_theorem_l3525_352568

/-- The total length of a necklace made of overlapping paper pieces -/
def necklaceLength (n : ℕ) (pieceLength : ℝ) (overlap : ℝ) : ℝ :=
  n * (pieceLength - overlap)

/-- Theorem: The total length of a necklace made of 16 pieces of colored paper,
    each 10.4 cm long and overlapping by 3.5 cm, is equal to 110.4 cm -/
theorem necklace_length_theorem :
  necklaceLength 16 10.4 3.5 = 110.4 := by
  sorry

end necklace_length_theorem_l3525_352568


namespace correct_outfit_assignment_l3525_352555

-- Define the colors
inductive Color
  | White
  | Red
  | Blue

-- Define a person's outfit
structure Outfit :=
  (dress : Color)
  (shoes : Color)

-- Define the friends
inductive Friend
  | Nadya
  | Valya
  | Masha

def outfit_assignment : Friend → Outfit
  | Friend.Nadya => { dress := Color.Blue, shoes := Color.Blue }
  | Friend.Valya => { dress := Color.Red, shoes := Color.White }
  | Friend.Masha => { dress := Color.White, shoes := Color.Red }

theorem correct_outfit_assignment :
  -- Nadya's shoes match her dress
  (outfit_assignment Friend.Nadya).dress = (outfit_assignment Friend.Nadya).shoes ∧
  -- Valya's dress and shoes are not blue
  (outfit_assignment Friend.Valya).dress ≠ Color.Blue ∧
  (outfit_assignment Friend.Valya).shoes ≠ Color.Blue ∧
  -- Masha wears red shoes
  (outfit_assignment Friend.Masha).shoes = Color.Red ∧
  -- All dresses are different colors
  (outfit_assignment Friend.Nadya).dress ≠ (outfit_assignment Friend.Valya).dress ∧
  (outfit_assignment Friend.Nadya).dress ≠ (outfit_assignment Friend.Masha).dress ∧
  (outfit_assignment Friend.Valya).dress ≠ (outfit_assignment Friend.Masha).dress ∧
  -- All shoes are different colors
  (outfit_assignment Friend.Nadya).shoes ≠ (outfit_assignment Friend.Valya).shoes ∧
  (outfit_assignment Friend.Nadya).shoes ≠ (outfit_assignment Friend.Masha).shoes ∧
  (outfit_assignment Friend.Valya).shoes ≠ (outfit_assignment Friend.Masha).shoes := by
  sorry

end correct_outfit_assignment_l3525_352555


namespace egg_count_theorem_l3525_352566

/-- Represents a carton of eggs -/
structure EggCarton where
  total_yolks : ℕ
  double_yolk_eggs : ℕ

/-- Calculate the number of eggs in a carton -/
def count_eggs (carton : EggCarton) : ℕ :=
  carton.double_yolk_eggs + (carton.total_yolks - 2 * carton.double_yolk_eggs)

/-- Theorem: A carton with 17 yolks and 5 double-yolk eggs contains 12 eggs -/
theorem egg_count_theorem (carton : EggCarton) 
  (h1 : carton.total_yolks = 17) 
  (h2 : carton.double_yolk_eggs = 5) : 
  count_eggs carton = 12 := by
  sorry

#eval count_eggs { total_yolks := 17, double_yolk_eggs := 5 }

end egg_count_theorem_l3525_352566


namespace equation_solution_l3525_352520

theorem equation_solution : 
  ∃ y : ℚ, y + 2/3 = 1/4 - 2/5 * 2 ∧ y = -511/420 := by
  sorry

end equation_solution_l3525_352520


namespace small_circle_radius_l3525_352584

/-- Given two circles where the radius of the larger circle is 80 cm and 4 times
    the radius of the smaller circle, prove that the radius of the smaller circle is 20 cm. -/
theorem small_circle_radius (r : ℝ) : 
  r > 0 → 4 * r = 80 → r = 20 := by
  sorry

end small_circle_radius_l3525_352584


namespace monika_total_expense_l3525_352534

def mall_expense : ℝ := 250
def movie_cost : ℝ := 24
def movie_count : ℕ := 3
def bean_bag_cost : ℝ := 1.25
def bean_bag_count : ℕ := 20

theorem monika_total_expense : 
  mall_expense + movie_cost * movie_count + bean_bag_cost * bean_bag_count = 347 := by
  sorry

end monika_total_expense_l3525_352534


namespace tangent_sum_equality_l3525_352540

theorem tangent_sum_equality (α β : Real) (h_acute_α : 0 < α ∧ α < π / 2) (h_acute_β : 0 < β ∧ β < π / 2)
  (h_equality : Real.tan (α - β) = Real.sin (2 * β)) :
  Real.tan α + Real.tan β = 2 * Real.tan (2 * β) := by
  sorry

end tangent_sum_equality_l3525_352540


namespace system_solution_l3525_352577

theorem system_solution :
  ∃ (x y : ℝ), (3 * x - 5 * y = -1.5) ∧ (7 * x + 2 * y = 4.7) ∧ (x = 0.5) ∧ (y = 0.6) := by
  sorry

end system_solution_l3525_352577


namespace addition_puzzle_l3525_352588

theorem addition_puzzle (P Q R : ℕ) : 
  P < 10 ∧ Q < 10 ∧ R < 10 →
  1000 * P + 100 * Q + 10 * P + R * 1000 + Q * 100 + Q * 10 + Q = 2009 →
  P + Q + R = 10 := by
  sorry

end addition_puzzle_l3525_352588


namespace large_block_length_multiple_l3525_352582

/-- Represents the dimensions of a block of cheese -/
structure CheeseDimensions where
  width : ℝ
  depth : ℝ
  length : ℝ

/-- Calculates the volume of a block of cheese given its dimensions -/
def volume (d : CheeseDimensions) : ℝ :=
  d.width * d.depth * d.length

theorem large_block_length_multiple (normal : CheeseDimensions) (large : CheeseDimensions) :
  volume normal = 3 →
  large.width = 2 * normal.width →
  large.depth = 2 * normal.depth →
  volume large = 36 →
  large.length = 3 * normal.length := by
  sorry

#check large_block_length_multiple

end large_block_length_multiple_l3525_352582


namespace complex_ratio_l3525_352503

theorem complex_ratio (a b : ℝ) (h1 : a * b ≠ 0) :
  let z : ℂ := Complex.mk a b
  (∃ (k : ℝ), z * Complex.mk 1 (-2) = k) → a / b = 1 / 2 := by
sorry

end complex_ratio_l3525_352503


namespace alice_baking_cake_l3525_352554

theorem alice_baking_cake (total_flour : ℕ) (cup_capacity : ℕ) (h1 : total_flour = 750) (h2 : cup_capacity = 125) :
  total_flour / cup_capacity = 6 := by
  sorry

end alice_baking_cake_l3525_352554


namespace one_more_tile_possible_exists_blocking_configuration_l3525_352589

/-- Represents a 4 × 6 grid -/
def Grid := Fin 4 → Fin 6 → Bool

/-- Represents an L-shaped tile -/
structure LTile :=
  (pos : Fin 4 × Fin 6)

/-- Checks if a tile placement is valid -/
def is_valid_placement (g : Grid) (t : LTile) : Prop :=
  sorry

/-- Places a tile on the grid -/
def place_tile (g : Grid) (t : LTile) : Grid :=
  sorry

/-- Theorem: After placing two tiles, one more can always be placed -/
theorem one_more_tile_possible (g : Grid) (t1 t2 : LTile) 
  (h1 : is_valid_placement g t1)
  (h2 : is_valid_placement (place_tile g t1) t2) :
  ∃ t3 : LTile, is_valid_placement (place_tile (place_tile g t1) t2) t3 :=
sorry

/-- Theorem: There exists a configuration of three tiles that blocks further placement -/
theorem exists_blocking_configuration :
  ∃ g : Grid, ∃ t1 t2 t3 : LTile,
    is_valid_placement g t1 ∧
    is_valid_placement (place_tile g t1) t2 ∧
    is_valid_placement (place_tile (place_tile g t1) t2) t3 ∧
    ∀ t4 : LTile, ¬is_valid_placement (place_tile (place_tile (place_tile g t1) t2) t3) t4 :=
sorry

end one_more_tile_possible_exists_blocking_configuration_l3525_352589


namespace circular_road_width_l3525_352517

theorem circular_road_width (r R : ℝ) (h1 : r = R / 3) (h2 : 2 * π * r + 2 * π * R = 88) :
  R - r = 22 / π := by
  sorry

end circular_road_width_l3525_352517


namespace average_cost_theorem_l3525_352579

def iPhone_quantity : ℕ := 100
def iPhone_price : ℝ := 1000
def iPhone_tax_rate : ℝ := 0.1

def iPad_quantity : ℕ := 20
def iPad_price : ℝ := 900
def iPad_discount_rate : ℝ := 0.05

def AppleTV_quantity : ℕ := 80
def AppleTV_price : ℝ := 200
def AppleTV_tax_rate : ℝ := 0.08

def MacBook_quantity : ℕ := 50
def MacBook_price : ℝ := 1500
def MacBook_discount_rate : ℝ := 0.15

def total_quantity : ℕ := iPhone_quantity + iPad_quantity + AppleTV_quantity + MacBook_quantity

def total_cost : ℝ :=
  iPhone_quantity * iPhone_price * (1 + iPhone_tax_rate) +
  iPad_quantity * iPad_price * (1 - iPad_discount_rate) +
  AppleTV_quantity * AppleTV_price * (1 + AppleTV_tax_rate) +
  MacBook_quantity * MacBook_price * (1 - MacBook_discount_rate)

theorem average_cost_theorem :
  total_cost / total_quantity = 832.52 := by sorry

end average_cost_theorem_l3525_352579
