import Mathlib

namespace NUMINAMATH_CALUDE_inequality_proof_l2559_255925

theorem inequality_proof (x a : ℝ) (h : x < a ∧ a < 0) : x^2 > a*x ∧ a*x > a^2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2559_255925


namespace NUMINAMATH_CALUDE_vector_computation_l2559_255971

theorem vector_computation : 
  5 • !![3, -9] - 4 • !![2, -6] + !![1, 3] = !![8, -18] := by
  sorry

end NUMINAMATH_CALUDE_vector_computation_l2559_255971


namespace NUMINAMATH_CALUDE_apartment_counts_equation_l2559_255949

/-- Represents the number of apartments of each type in a building -/
structure ApartmentCounts where
  studio : ℝ
  twoPerson : ℝ
  threePerson : ℝ
  fourPerson : ℝ
  fivePerson : ℝ

/-- The apartment complex configuration -/
structure ApartmentComplex where
  buildingCount : ℕ
  maxOccupancy : ℕ
  occupancyRate : ℝ
  studioCapacity : ℝ
  twoPersonCapacity : ℝ
  threePersonCapacity : ℝ
  fourPersonCapacity : ℝ
  fivePersonCapacity : ℝ

/-- Theorem stating the equation for apartment counts given the complex configuration -/
theorem apartment_counts_equation (complex : ApartmentComplex) 
    (counts : ApartmentCounts) : 
    complex.buildingCount = 8 ∧ 
    complex.maxOccupancy = 3000 ∧ 
    complex.occupancyRate = 0.9 ∧
    complex.studioCapacity = 0.95 ∧
    complex.twoPersonCapacity = 0.85 ∧
    complex.threePersonCapacity = 0.8 ∧
    complex.fourPersonCapacity = 0.75 ∧
    complex.fivePersonCapacity = 0.65 →
    0.11875 * counts.studio + 0.2125 * counts.twoPerson + 
    0.3 * counts.threePerson + 0.375 * counts.fourPerson + 
    0.40625 * counts.fivePerson = 337.5 := by
  sorry

end NUMINAMATH_CALUDE_apartment_counts_equation_l2559_255949


namespace NUMINAMATH_CALUDE_min_value_of_function_min_value_achievable_l2559_255985

theorem min_value_of_function (x : ℝ) (h : x > 0) : 2 * x + 3 / x ≥ 2 * Real.sqrt 6 := by
  sorry

theorem min_value_achievable : ∃ x : ℝ, x > 0 ∧ 2 * x + 3 / x = 2 * Real.sqrt 6 := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_function_min_value_achievable_l2559_255985


namespace NUMINAMATH_CALUDE_max_volume_cylinder_in_sphere_l2559_255909

noncomputable section

theorem max_volume_cylinder_in_sphere (R : ℝ) (h r : ℝ → ℝ) :
  (∀ t, 4 * R^2 = 4 * (r t)^2 + (h t)^2) →
  (∀ t, (r t) ≥ 0 ∧ (h t) ≥ 0) →
  (∃ t₀, ∀ t, π * (r t)^2 * (h t) ≤ π * (r t₀)^2 * (h t₀)) →
  h t₀ = 2 * R / Real.sqrt 3 ∧ r t₀ = R * Real.sqrt (2/3) :=
by sorry

end

end NUMINAMATH_CALUDE_max_volume_cylinder_in_sphere_l2559_255909


namespace NUMINAMATH_CALUDE_circle_equation_l2559_255942

/-- A circle with center on the line y = x passing through (-1, 1) and (1, 3) has the equation (x-1)^2 + (y-1)^2 = 4 -/
theorem circle_equation : ∀ (a : ℝ),
  (∀ (x y : ℝ), (x - a)^2 + (y - a)^2 = (a + 1)^2 + (a - 1)^2) →
  (∀ (x y : ℝ), (x - a)^2 + (y - a)^2 = (a - 1)^2 + (a - 3)^2) →
  (∀ (x y : ℝ), (x - 1)^2 + (y - 1)^2 = 4) :=
by sorry

end NUMINAMATH_CALUDE_circle_equation_l2559_255942


namespace NUMINAMATH_CALUDE_midpoint_of_complex_line_segment_l2559_255944

theorem midpoint_of_complex_line_segment :
  let z₁ : ℂ := -7 + 5*I
  let z₂ : ℂ := 5 - 9*I
  let midpoint := (z₁ + z₂) / 2
  midpoint = -1 - 2*I := by sorry

end NUMINAMATH_CALUDE_midpoint_of_complex_line_segment_l2559_255944


namespace NUMINAMATH_CALUDE_f_has_max_and_min_l2559_255951

/-- A cubic function with parameter a -/
def f (a : ℝ) (x : ℝ) : ℝ := x^3 + a*x^2 + (a + 6)*x + 1

/-- The derivative of f with respect to x -/
def f_deriv (a : ℝ) (x : ℝ) : ℝ := 3*x^2 + 2*a*x + (a + 6)

/-- Theorem stating the condition for f to have both maximum and minimum values -/
theorem f_has_max_and_min (a : ℝ) : 
  (∃ (max min : ℝ), ∀ x, f a x ≤ max ∧ f a x ≥ min) ↔ (a < -3 ∨ a > 6) :=
sorry

end NUMINAMATH_CALUDE_f_has_max_and_min_l2559_255951


namespace NUMINAMATH_CALUDE_quadratic_inequality_l2559_255941

theorem quadratic_inequality (x : ℝ) : x^2 - 4*x > 44 ↔ x < -4 ∨ x > 11 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_l2559_255941


namespace NUMINAMATH_CALUDE_fraction_equality_l2559_255982

theorem fraction_equality : 
  (2 + 4 - 8 + 16 + 32 - 64 + 128 - 256) / (4 + 8 - 16 + 32 + 64 - 128 + 256 - 512) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l2559_255982


namespace NUMINAMATH_CALUDE_parallelogram_area_l2559_255972

/-- The area of a parallelogram with one angle of 135 degrees and two consecutive sides of lengths 10 and 17 is equal to 85√2. -/
theorem parallelogram_area (a b : ℝ) (θ : ℝ) (h1 : a = 10) (h2 : b = 17) (h3 : θ = 135 * π / 180) :
  a * b * Real.sin θ = 85 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_area_l2559_255972


namespace NUMINAMATH_CALUDE_new_person_weight_l2559_255932

/-- Given a group of 8 people, prove that when one person weighing 65 kg is replaced
    by a new person, and the average weight increases by 2.5 kg,
    the weight of the new person is 85 kg. -/
theorem new_person_weight (initial_average : ℝ) : 
  let num_people : ℕ := 8
  let weight_increase : ℝ := 2.5
  let old_person_weight : ℝ := 65
  let new_average : ℝ := initial_average + weight_increase
  let new_person_weight : ℝ := old_person_weight + (num_people * weight_increase)
  new_person_weight = 85 := by
sorry

end NUMINAMATH_CALUDE_new_person_weight_l2559_255932


namespace NUMINAMATH_CALUDE_cylinder_volume_change_l2559_255965

/-- Theorem: Cylinder Volume Change
  Given a cylinder with an original volume of 20 cubic feet,
  if its radius is tripled and its height is quadrupled,
  then its new volume will be 720 cubic feet.
-/
theorem cylinder_volume_change (r h : ℝ) :
  (π * r^2 * h = 20) →  -- Original volume is 20 cubic feet
  (π * (3*r)^2 * (4*h) = 720) :=  -- New volume is 720 cubic feet
by sorry

end NUMINAMATH_CALUDE_cylinder_volume_change_l2559_255965


namespace NUMINAMATH_CALUDE_distance_between_circle_centers_l2559_255918

/-- Given a triangle DEF with side lengths, prove the distance between incircle and excircle centers --/
theorem distance_between_circle_centers (DE DF EF : ℝ) (h_DE : DE = 16) (h_DF : DF = 17) (h_EF : EF = 15) :
  let s := (DE + DF + EF) / 2
  let K := Real.sqrt (s * (s - DE) * (s - DF) * (s - EF))
  let r := K / s
  let DI := Real.sqrt (((s - DE) ^ 2) + (r ^ 2))
  let DE' := 3 * DI
  DE' - DI = 10 * Real.sqrt 30 := by
  sorry

end NUMINAMATH_CALUDE_distance_between_circle_centers_l2559_255918


namespace NUMINAMATH_CALUDE_kylie_and_nelly_stamps_l2559_255930

/-- Given that Kylie has 34 stamps and Nelly has 44 more stamps than Kylie,
    prove that they have 112 stamps together. -/
theorem kylie_and_nelly_stamps :
  let kylie_stamps : ℕ := 34
  let nelly_stamps : ℕ := kylie_stamps + 44
  kylie_stamps + nelly_stamps = 112 := by sorry

end NUMINAMATH_CALUDE_kylie_and_nelly_stamps_l2559_255930


namespace NUMINAMATH_CALUDE_michelle_needs_three_more_racks_l2559_255900

/-- The number of additional drying racks Michelle needs -/
def additional_racks_needed : ℕ :=
  let total_flour : ℕ := 6 * 12 -- 6 bags * 12 cups per bag
  let flour_per_type : ℕ := total_flour / 2 -- equal amounts for both types
  let pasta_type1 : ℕ := flour_per_type / 3 -- 3 cups per pound for type 1
  let pasta_type2 : ℕ := flour_per_type / 4 -- 4 cups per pound for type 2
  let total_pasta : ℕ := pasta_type1 + pasta_type2
  let total_racks_needed : ℕ := (total_pasta + 4) / 5 -- Ceiling division by 5
  total_racks_needed - 2 -- Subtract the 2 racks she already owns

theorem michelle_needs_three_more_racks :
  additional_racks_needed = 3 := by
  sorry

end NUMINAMATH_CALUDE_michelle_needs_three_more_racks_l2559_255900


namespace NUMINAMATH_CALUDE_garden_perimeter_is_60_l2559_255994

/-- A rectangular garden with given diagonal and area -/
structure RectangularGarden where
  width : ℝ
  height : ℝ
  diagonal_sq : width^2 + height^2 = 26^2
  area : width * height = 120

/-- The perimeter of a rectangular garden -/
def perimeter (g : RectangularGarden) : ℝ := 2 * (g.width + g.height)

/-- Theorem: The perimeter of the given rectangular garden is 60 meters -/
theorem garden_perimeter_is_60 (g : RectangularGarden) : perimeter g = 60 := by
  sorry

end NUMINAMATH_CALUDE_garden_perimeter_is_60_l2559_255994


namespace NUMINAMATH_CALUDE_sara_marbles_l2559_255943

theorem sara_marbles (initial_marbles additional_marbles : ℝ) 
  (h1 : initial_marbles = 4892.5)
  (h2 : additional_marbles = 2337.8) :
  initial_marbles + additional_marbles = 7230.3 := by
sorry

end NUMINAMATH_CALUDE_sara_marbles_l2559_255943


namespace NUMINAMATH_CALUDE_cube_number_placement_impossible_l2559_255945

/-- Represents a cube with 8 vertices -/
structure Cube :=
  (vertices : Fin 8 → ℕ)

/-- Predicate to check if two vertices are adjacent on a cube -/
def adjacent (i j : Fin 8) : Prop := sorry

/-- The theorem stating the impossibility of the number placement on a cube -/
theorem cube_number_placement_impossible :
  ¬ ∃ (c : Cube),
    (∀ i : Fin 8, 1 ≤ c.vertices i ∧ c.vertices i ≤ 220) ∧
    (∀ i j : Fin 8, i ≠ j → c.vertices i ≠ c.vertices j) ∧
    (∀ i j : Fin 8, adjacent i j → ∃ (d : ℕ), d > 1 ∧ d ∣ c.vertices i ∧ d ∣ c.vertices j) ∧
    (∀ i j : Fin 8, ¬adjacent i j → ∀ (d : ℕ), d > 1 → ¬(d ∣ c.vertices i ∧ d ∣ c.vertices j)) :=
sorry

end NUMINAMATH_CALUDE_cube_number_placement_impossible_l2559_255945


namespace NUMINAMATH_CALUDE_factor_theorem_cubic_l2559_255980

theorem factor_theorem_cubic (a : ℚ) :
  (∀ x, x^3 + 2*x^2 + a*x + 20 = 0 → x = 3) →
  a = -65/3 := by
sorry

end NUMINAMATH_CALUDE_factor_theorem_cubic_l2559_255980


namespace NUMINAMATH_CALUDE_two_digit_multiples_of_6_and_9_l2559_255902

theorem two_digit_multiples_of_6_and_9 : 
  (Finset.filter (fun n => n % 6 = 0 ∧ n % 9 = 0) (Finset.range 90 \ Finset.range 10)).card = 5 := by
  sorry

end NUMINAMATH_CALUDE_two_digit_multiples_of_6_and_9_l2559_255902


namespace NUMINAMATH_CALUDE_right_triangle_perimeter_l2559_255927

/-- A right triangle with one leg of prime length n and other sides of natural number lengths has perimeter n + n^2 -/
theorem right_triangle_perimeter (n : ℕ) (h_prime : Nat.Prime n) :
  ∃ (x y : ℕ), x^2 + n^2 = y^2 ∧ x + y + n = n + n^2 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_perimeter_l2559_255927


namespace NUMINAMATH_CALUDE_complex_equation_solution_l2559_255989

theorem complex_equation_solution (z : ℂ) : (z - 2*Complex.I) * (2 - Complex.I) = 5 → z = 2 + 3*Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l2559_255989


namespace NUMINAMATH_CALUDE_perpendicular_line_equation_l2559_255991

/-- A line in 2D space represented by its slope and y-intercept -/
structure Line where
  slope : ℚ
  intercept : ℚ

/-- A point in 2D space -/
structure Point where
  x : ℚ
  y : ℚ

/-- Check if a point lies on a line -/
def pointOnLine (p : Point) (l : Line) : Prop :=
  p.y = l.slope * p.x + l.intercept

/-- Check if two lines are perpendicular -/
def perpendicular (l1 l2 : Line) : Prop :=
  l1.slope * l2.slope = -1

theorem perpendicular_line_equation (given_line : Line) (point : Point) : 
  given_line.slope = 2/3 ∧ given_line.intercept = -2 ∧ point.x = 4 ∧ point.y = 2 →
  ∃ (result_line : Line), 
    result_line.slope = -3/2 ∧ 
    result_line.intercept = 8 ∧
    pointOnLine point result_line ∧
    perpendicular given_line result_line :=
by sorry

end NUMINAMATH_CALUDE_perpendicular_line_equation_l2559_255991


namespace NUMINAMATH_CALUDE_problem_pyramid_volume_l2559_255997

/-- Represents a truncated triangular pyramid -/
structure TruncatedPyramid where
  height : ℝ
  base1_sides : Fin 3 → ℝ
  base2_perimeter : ℝ

/-- Calculates the volume of a truncated triangular pyramid -/
def volume (p : TruncatedPyramid) : ℝ :=
  sorry

/-- The specific truncated pyramid from the problem -/
def problem_pyramid : TruncatedPyramid :=
  { height := 10
  , base1_sides := ![27, 29, 52]
  , base2_perimeter := 72 }

/-- Theorem stating that the volume of the problem pyramid is 1900 -/
theorem problem_pyramid_volume :
  volume problem_pyramid = 1900 := by sorry

end NUMINAMATH_CALUDE_problem_pyramid_volume_l2559_255997


namespace NUMINAMATH_CALUDE_running_is_experimental_l2559_255956

/-- Represents an investigation method -/
inductive InvestigationMethod
  | Experimental
  | NonExperimental

/-- Represents the characteristics of an investigation -/
structure Investigation where
  description : String
  quantitative : Bool
  directlyMeasurable : Bool
  controlledSetting : Bool

/-- Determines if an investigation is suitable for the experimental method -/
def isSuitableForExperiment (i : Investigation) : InvestigationMethod :=
  if i.quantitative && i.directlyMeasurable && i.controlledSetting then
    InvestigationMethod.Experimental
  else
    InvestigationMethod.NonExperimental

/-- The investigation of running distance in 10 seconds -/
def runningInvestigation : Investigation where
  description := "How many meters you can run in 10 seconds"
  quantitative := true
  directlyMeasurable := true
  controlledSetting := true

/-- Theorem stating that the running investigation is suitable for the experimental method -/
theorem running_is_experimental :
  isSuitableForExperiment runningInvestigation = InvestigationMethod.Experimental := by
  sorry


end NUMINAMATH_CALUDE_running_is_experimental_l2559_255956


namespace NUMINAMATH_CALUDE_least_sum_of_four_primes_l2559_255981

-- Define a function that checks if a number is prime
def isPrime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(n % m = 0)

-- Define a function that represents the sum of 4 different primes greater than 10
def sumOfFourPrimes (a b c d : ℕ) : Prop :=
  isPrime a ∧ isPrime b ∧ isPrime c ∧ isPrime d ∧
  a > 10 ∧ b > 10 ∧ c > 10 ∧ d > 10 ∧
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d

-- Theorem statement
theorem least_sum_of_four_primes :
  ∀ n : ℕ, (∃ a b c d : ℕ, sumOfFourPrimes a b c d ∧ a + b + c + d = n) →
  n ≥ 60 :=
sorry

end NUMINAMATH_CALUDE_least_sum_of_four_primes_l2559_255981


namespace NUMINAMATH_CALUDE_ant_path_problem_l2559_255990

/-- Represents the ant's path in the rectangle -/
structure AntPath where
  rectangle_width : ℝ
  rectangle_height : ℝ
  start_point : ℝ
  path_angle : ℝ

/-- The problem statement -/
theorem ant_path_problem (path : AntPath) :
  path.rectangle_width = 150 ∧
  path.rectangle_height = 18 ∧
  path.path_angle = π / 4 ∧
  path.start_point ≥ 0 ∧
  path.start_point ≤ path.rectangle_height ∧
  (∃ (n : ℕ), 
    path.start_point + n * path.rectangle_height - 2 * n * path.start_point = path.rectangle_width / 2) →
  min path.start_point (path.rectangle_height - path.start_point) = 3 := by
  sorry

end NUMINAMATH_CALUDE_ant_path_problem_l2559_255990


namespace NUMINAMATH_CALUDE_journey_speed_calculation_l2559_255916

/-- Proves that given a journey of 1.5 km, if a person arrives 7 minutes late when traveling
    at speed v km/hr, and arrives 8 minutes early when traveling at 6 km/hr, then v = 10 km/hr. -/
theorem journey_speed_calculation (v : ℝ) : 
  (∃ t : ℝ, 
    1.5 = v * (t - 7/60) ∧ 
    1.5 = 6 * (t - 8/60)) → 
  v = 10 := by
sorry

end NUMINAMATH_CALUDE_journey_speed_calculation_l2559_255916


namespace NUMINAMATH_CALUDE_roots_ratio_implies_k_value_l2559_255946

theorem roots_ratio_implies_k_value :
  ∀ (k : ℝ) (r s : ℝ),
    r ≠ 0 → s ≠ 0 →
    r^2 + 8*r + k = 0 →
    s^2 + 8*s + k = 0 →
    r / s = 3 →
    k = 12 := by
sorry

end NUMINAMATH_CALUDE_roots_ratio_implies_k_value_l2559_255946


namespace NUMINAMATH_CALUDE_white_ball_count_l2559_255979

theorem white_ball_count : ∃ (x y : ℕ), 
  x < y ∧ 
  y < 2 * x ∧ 
  2 * x + 3 * y = 60 ∧ 
  x = 9 ∧ 
  y = 14 := by
  sorry

end NUMINAMATH_CALUDE_white_ball_count_l2559_255979


namespace NUMINAMATH_CALUDE_special_quadrilateral_is_kite_l2559_255984

/-- A quadrilateral with specific properties -/
structure SpecialQuadrilateral where
  /-- The diagonals of the quadrilateral bisect each other -/
  diagonals_bisect : Bool
  /-- The diagonals of the quadrilateral are perpendicular -/
  diagonals_perpendicular : Bool
  /-- Two adjacent sides of the quadrilateral are equal -/
  two_adjacent_sides_equal : Bool

/-- Definition of a kite -/
def is_kite (q : SpecialQuadrilateral) : Prop :=
  q.diagonals_bisect ∧ q.diagonals_perpendicular ∧ q.two_adjacent_sides_equal

/-- The main theorem stating that a quadrilateral with the given properties is most likely a kite -/
theorem special_quadrilateral_is_kite (q : SpecialQuadrilateral) 
  (h1 : q.diagonals_bisect = true) 
  (h2 : q.diagonals_perpendicular = true) 
  (h3 : q.two_adjacent_sides_equal = true) : 
  is_kite q :=
sorry

end NUMINAMATH_CALUDE_special_quadrilateral_is_kite_l2559_255984


namespace NUMINAMATH_CALUDE_truncated_cone_angle_l2559_255958

theorem truncated_cone_angle (R : ℝ) (h : ℝ) (r : ℝ) : 
  h = R → 
  (12 * r) / Real.sqrt 3 = 3 * R * Real.sqrt 3 → 
  Real.arctan (h / (R - r)) = Real.arctan 4 := by
  sorry

end NUMINAMATH_CALUDE_truncated_cone_angle_l2559_255958


namespace NUMINAMATH_CALUDE_bd_squared_equals_25_l2559_255964

theorem bd_squared_equals_25 
  (h1 : a - b - c + d = 13)
  (h2 : a + b - c - d = 3)
  (h3 : 2*a - 3*b + c + 4*d = 17)
  : (b - d)^2 = 25 := by
  sorry

end NUMINAMATH_CALUDE_bd_squared_equals_25_l2559_255964


namespace NUMINAMATH_CALUDE_fixed_point_of_exponential_function_l2559_255983

/-- For all a > 0 and a ≠ 1, the function f(x) = a^(x-3) - 3 passes through the point (3, -2) -/
theorem fixed_point_of_exponential_function (a : ℝ) (ha : a > 0) (hna : a ≠ 1) :
  let f : ℝ → ℝ := λ x ↦ a^(x - 3) - 3
  f 3 = -2 := by
sorry

end NUMINAMATH_CALUDE_fixed_point_of_exponential_function_l2559_255983


namespace NUMINAMATH_CALUDE_room_length_calculation_l2559_255924

theorem room_length_calculation (breadth height pole_length : ℝ) 
  (h1 : breadth = 8)
  (h2 : height = 9)
  (h3 : pole_length = 17) : 
  ∃ length : ℝ, length^2 + breadth^2 + height^2 = pole_length^2 ∧ length = 12 := by
  sorry

end NUMINAMATH_CALUDE_room_length_calculation_l2559_255924


namespace NUMINAMATH_CALUDE_election_win_margin_l2559_255952

theorem election_win_margin (total_votes : ℕ) (winner_votes : ℕ) :
  (winner_votes : ℚ) / total_votes = 62 / 100 →
  winner_votes = 868 →
  winner_votes - (total_votes - winner_votes) = 336 :=
by sorry

end NUMINAMATH_CALUDE_election_win_margin_l2559_255952


namespace NUMINAMATH_CALUDE_truck_rental_example_l2559_255950

/-- Calculates the total cost of renting a truck given the daily rate, per-mile rate, number of days, and miles driven. -/
def truck_rental_cost (daily_rate : ℚ) (mile_rate : ℚ) (days : ℕ) (miles : ℕ) : ℚ :=
  daily_rate * days + mile_rate * miles

/-- Proves that renting a truck for $35 per day and $0.25 per mile for 3 days and 300 miles costs $180 in total. -/
theorem truck_rental_example : truck_rental_cost 35 (1/4) 3 300 = 180 := by
  sorry

end NUMINAMATH_CALUDE_truck_rental_example_l2559_255950


namespace NUMINAMATH_CALUDE_thermostat_problem_l2559_255955

theorem thermostat_problem (initial_temp : ℝ) (final_temp : ℝ) (x : ℝ) 
  (h1 : initial_temp = 40)
  (h2 : final_temp = 59) : 
  (((initial_temp * 2 - 30) * 0.7) + x = final_temp) → x = 24 := by
  sorry

end NUMINAMATH_CALUDE_thermostat_problem_l2559_255955


namespace NUMINAMATH_CALUDE_even_function_implies_a_equals_one_l2559_255947

/-- A function f : ℝ → ℝ is even if f(-x) = f(x) for all x ∈ ℝ -/
def IsEven (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f (-x) = f x

/-- The function f(x) = (x+1)(x-a) -/
def f (a : ℝ) : ℝ → ℝ := fun x ↦ (x + 1) * (x - a)

/-- If f(x) = (x+1)(x-a) is an even function, then a = 1 -/
theorem even_function_implies_a_equals_one :
  ∀ a : ℝ, IsEven (f a) → a = 1 := by
  sorry


end NUMINAMATH_CALUDE_even_function_implies_a_equals_one_l2559_255947


namespace NUMINAMATH_CALUDE_rational_equation_sum_l2559_255954

theorem rational_equation_sum (A B : ℝ) :
  (∀ x : ℝ, x ≠ 2 ∧ x ≠ 5 →
    (B * x - 11) / (x^2 - 7*x + 10) = A / (x - 2) + 3 / (x - 5)) →
  A + B = 5 := by
sorry

end NUMINAMATH_CALUDE_rational_equation_sum_l2559_255954


namespace NUMINAMATH_CALUDE_first_day_sale_l2559_255936

theorem first_day_sale (total_days : ℕ) (average_sale : ℕ) (known_days_sales : List ℕ) :
  total_days = 5 →
  average_sale = 625 →
  known_days_sales = [927, 855, 230, 562] →
  (total_days * average_sale) - known_days_sales.sum = 551 := by
  sorry

end NUMINAMATH_CALUDE_first_day_sale_l2559_255936


namespace NUMINAMATH_CALUDE_race_distance_l2559_255973

/-- The race problem -/
theorem race_distance (a_time b_time : ℕ) (beat_distance : ℕ) (total_distance : ℕ) : 
  a_time = 36 →
  b_time = 45 →
  beat_distance = 20 →
  (total_distance : ℚ) / a_time * b_time = total_distance + beat_distance →
  total_distance = 80 := by
  sorry

end NUMINAMATH_CALUDE_race_distance_l2559_255973


namespace NUMINAMATH_CALUDE_exists_number_not_exceeding_kr_l2559_255926

/-- The operation that replaces a number with two new numbers -/
def replace_operation (x : ℝ) : ℝ × ℝ :=
  sorry

/-- Perform the operation k^2 - 1 times -/
def perform_operations (r : ℝ) (k : ℕ) : List ℝ :=
  sorry

theorem exists_number_not_exceeding_kr (r : ℝ) (k : ℕ) (h_r : r > 0) :
  ∃ x ∈ perform_operations r k, x ≤ k * r :=
sorry

end NUMINAMATH_CALUDE_exists_number_not_exceeding_kr_l2559_255926


namespace NUMINAMATH_CALUDE_drama_club_organization_l2559_255976

theorem drama_club_organization (participants : ℕ) (girls : ℕ) (boys : ℕ) : 
  participants = girls + boys →
  girls > (85 * participants) / 100 →
  boys ≥ 2 →
  participants ≥ 14 :=
by
  sorry

end NUMINAMATH_CALUDE_drama_club_organization_l2559_255976


namespace NUMINAMATH_CALUDE_intercept_ratio_l2559_255959

theorem intercept_ratio (b s t : ℝ) (hb : b ≠ 0) : 
  0 = 8 * s + b ∧ 0 = 4 * t + b → s / t = 1 / 2 := by
sorry

end NUMINAMATH_CALUDE_intercept_ratio_l2559_255959


namespace NUMINAMATH_CALUDE_distribution_problem_l2559_255961

/-- Represents the number of ways to distribute n distinct objects into k non-empty groups -/
def distribute (n k : ℕ) : ℕ := sorry

/-- Represents the number of ways to distribute n distinct objects into k non-empty groups,
    where two specific objects cannot be in the same group -/
def distributeWithRestriction (n k : ℕ) : ℕ := sorry

/-- The main theorem stating that there are 114 ways to distribute 5 distinct objects
    into 3 non-empty groups, where two specific objects cannot be in the same group -/
theorem distribution_problem : distributeWithRestriction 5 3 = 114 := by sorry

end NUMINAMATH_CALUDE_distribution_problem_l2559_255961


namespace NUMINAMATH_CALUDE_prime_pair_divisibility_l2559_255962

theorem prime_pair_divisibility (p q : ℕ) : 
  Prime p ∧ Prime q → (p * q ∣ p^p + q^q + 1) ↔ ((p = 2 ∧ q = 5) ∨ (p = 5 ∧ q = 2)) := by
  sorry

end NUMINAMATH_CALUDE_prime_pair_divisibility_l2559_255962


namespace NUMINAMATH_CALUDE_probability_of_convex_pentagon_l2559_255978

def num_points : ℕ := 7
def num_chords_selected : ℕ := 5

def total_chords (n : ℕ) : ℕ := n.choose 2

def ways_to_select_chords (total : ℕ) (k : ℕ) : ℕ := total.choose k

def convex_pentagons (n : ℕ) : ℕ := n.choose 5

theorem probability_of_convex_pentagon :
  (convex_pentagons num_points : ℚ) / (ways_to_select_chords (total_chords num_points) num_chords_selected) = 1 / 969 :=
sorry

end NUMINAMATH_CALUDE_probability_of_convex_pentagon_l2559_255978


namespace NUMINAMATH_CALUDE_tshirt_price_correct_l2559_255928

/-- The regular price of a T-shirt -/
def regular_price : ℝ := 14.5

/-- The total number of T-shirts purchased -/
def total_shirts : ℕ := 12

/-- The total cost of the purchase -/
def total_cost : ℝ := 120

/-- The cost of a group of three T-shirts (two at regular price, one at $1) -/
def group_cost (price : ℝ) : ℝ := 2 * price + 1

/-- The number of groups of three T-shirts -/
def num_groups : ℕ := total_shirts / 3

theorem tshirt_price_correct :
  group_cost regular_price * num_groups = total_cost ∧
  regular_price > 0 := by
  sorry

end NUMINAMATH_CALUDE_tshirt_price_correct_l2559_255928


namespace NUMINAMATH_CALUDE_arithmetic_mean_of_special_set_l2559_255974

theorem arithmetic_mean_of_special_set (n : ℕ) (h : n > 1) :
  let set := List.replicate (n - 3) 1 ++ [1 + 1/n, 1 + 1/n, 1 - 1/n]
  (set.sum / n : ℚ) = 1 + 1/n^2 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_mean_of_special_set_l2559_255974


namespace NUMINAMATH_CALUDE_prism_volume_l2559_255957

-- Define the prism dimensions
variable (a b c : ℝ)

-- Define the conditions
axiom face_area_1 : a * b = 30
axiom face_area_2 : b * c = 72
axiom face_area_3 : c * a = 45

-- State the theorem
theorem prism_volume : a * b * c = 180 * Real.sqrt 3 := by sorry

end NUMINAMATH_CALUDE_prism_volume_l2559_255957


namespace NUMINAMATH_CALUDE_problem_statement_l2559_255910

theorem problem_statement (x y : ℝ) (m n : ℤ) 
  (h : x > 0) (h' : y > 0) 
  (eq : x^m * y * 4*y^n / (4*x^6*y^4) = 1) : 
  m - n = 3 := by
sorry

end NUMINAMATH_CALUDE_problem_statement_l2559_255910


namespace NUMINAMATH_CALUDE_complex_sum_equals_i_l2559_255939

theorem complex_sum_equals_i : Complex.I ^ 2 = -1 → (1 : ℂ) + Complex.I + Complex.I ^ 2 = Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_sum_equals_i_l2559_255939


namespace NUMINAMATH_CALUDE_land_development_profit_l2559_255908

theorem land_development_profit (cost_per_acre : ℝ) (sale_price_per_acre : ℝ) (profit : ℝ) (acres : ℝ) : 
  cost_per_acre = 70 →
  sale_price_per_acre = 200 →
  profit = 6000 →
  sale_price_per_acre * (acres / 2) - cost_per_acre * acres = profit →
  acres = 200 := by
sorry

end NUMINAMATH_CALUDE_land_development_profit_l2559_255908


namespace NUMINAMATH_CALUDE_area_enclosed_by_curve_l2559_255904

/-- The area enclosed by a curve composed of 12 congruent circular arcs -/
theorem area_enclosed_by_curve (arc_length : Real) (hexagon_side : Real) : 
  arc_length = 5 * Real.pi / 6 →
  hexagon_side = 4 →
  ∃ (area : Real), 
    area = 48 * Real.sqrt 3 + 125 * Real.pi / 2 ∧
    area = (3 * Real.sqrt 3 / 2 * hexagon_side ^ 2) + 
           (12 * (arc_length / (2 * Real.pi)) * Real.pi * (arc_length / Real.pi) ^ 2) :=
by sorry


end NUMINAMATH_CALUDE_area_enclosed_by_curve_l2559_255904


namespace NUMINAMATH_CALUDE_star_symmetric_zero_l2559_255919

/-- Define the binary operation ⋆ for real numbers -/
def star (a b : ℝ) : ℝ := (a^2 - b^2)^2

/-- Theorem: For any real numbers x and y, (x-y)² ⋆ (y-x)² = 0 -/
theorem star_symmetric_zero (x y : ℝ) : star ((x - y)^2) ((y - x)^2) = 0 := by
  sorry

end NUMINAMATH_CALUDE_star_symmetric_zero_l2559_255919


namespace NUMINAMATH_CALUDE_calculator_sum_theorem_l2559_255975

/-- The number of participants in the game --/
def num_participants : ℕ := 47

/-- The initial value of calculator A --/
def initial_A : ℤ := 2

/-- The initial value of calculator B --/
def initial_B : ℕ := 0

/-- The initial value of calculator C --/
def initial_C : ℤ := -1

/-- The initial value of calculator D --/
def initial_D : ℕ := 3

/-- The final value of calculator A after all participants have processed it --/
def final_A : ℤ := -initial_A

/-- The final value of calculator B after all participants have processed it --/
def final_B : ℕ := initial_B

/-- The final value of calculator C after all participants have processed it --/
def final_C : ℤ := -initial_C

/-- The final value of calculator D after all participants have processed it --/
noncomputable def final_D : ℕ := initial_D ^ (3 ^ num_participants)

/-- The theorem stating that the sum of the final calculator values equals 3^(3^47) - 3 --/
theorem calculator_sum_theorem :
  final_A + final_B + final_C + final_D = 3^(3^47) - 3 := by
  sorry


end NUMINAMATH_CALUDE_calculator_sum_theorem_l2559_255975


namespace NUMINAMATH_CALUDE_prob_X_equals_three_l2559_255913

/-- X is a random variable following a binomial distribution B(6, 1/2) -/
def X : Real → Real := sorry

/-- The probability mass function of X -/
def pmf (k : ℕ) : Real := sorry

/-- Theorem: The probability of X = 3 is 5/16 -/
theorem prob_X_equals_three : pmf 3 = 5/16 := by sorry

end NUMINAMATH_CALUDE_prob_X_equals_three_l2559_255913


namespace NUMINAMATH_CALUDE_solution_set_quadratic_inequality_l2559_255986

theorem solution_set_quadratic_inequality :
  {x : ℝ | x^2 - 5*x + 6 ≤ 0} = {x : ℝ | 2 ≤ x ∧ x ≤ 3} := by sorry

end NUMINAMATH_CALUDE_solution_set_quadratic_inequality_l2559_255986


namespace NUMINAMATH_CALUDE_logarithmic_equation_solution_l2559_255914

noncomputable def log_base (b x : ℝ) : ℝ := Real.log x / Real.log b

theorem logarithmic_equation_solution :
  ∃ (x : ℝ), x > 0 ∧ 
  log_base 3 (x - 1) + log_base (Real.sqrt 3) (x^2 - 1) + log_base (1/3) (x - 1) = 3 ∧
  x = Real.sqrt (1 + 3 * Real.sqrt 3) :=
sorry

end NUMINAMATH_CALUDE_logarithmic_equation_solution_l2559_255914


namespace NUMINAMATH_CALUDE_g_2022_l2559_255938

/-- Given a function g: ℝ → ℝ satisfying the functional equation
    g(x - y) = 2022 * (g x + g y) - 2021 * x * y for all real x and y,
    prove that g(2022) = 2043231 -/
theorem g_2022 (g : ℝ → ℝ) 
    (h : ∀ x y : ℝ, g (x - y) = 2022 * (g x + g y) - 2021 * x * y) : 
  g 2022 = 2043231 := by
  sorry

end NUMINAMATH_CALUDE_g_2022_l2559_255938


namespace NUMINAMATH_CALUDE_geometric_sequence_a5_l2559_255996

/-- A geometric sequence with positive terms -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r ≠ 0 ∧ ∀ n : ℕ, a (n + 1) = a n * r

theorem geometric_sequence_a5 (a : ℕ → ℝ) :
  GeometricSequence a →
  (∀ n : ℕ, a n > 0) →
  a 1 * a 9 = 10 →
  a 5 = Real.sqrt 10 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_a5_l2559_255996


namespace NUMINAMATH_CALUDE_no_integer_solution_l2559_255966

theorem no_integer_solution (n : ℕ+) : ¬ (∃ k : ℤ, (n.val^2 + 1 : ℤ) = k * ((Int.floor (Real.sqrt n.val))^2 + 2)) := by
  sorry

end NUMINAMATH_CALUDE_no_integer_solution_l2559_255966


namespace NUMINAMATH_CALUDE_item_distribution_l2559_255907

theorem item_distribution (n₁ n₂ n₃ k : ℕ) (h₁ : n₁ = 5) (h₂ : n₂ = 3) (h₃ : n₃ = 2) (h₄ : k = 3) :
  (Nat.choose (n₁ + k - 1) (k - 1)) * (Nat.choose (n₂ + k - 1) (k - 1)) * (Nat.choose (n₃ + k - 1) (k - 1)) = 1260 :=
by sorry

end NUMINAMATH_CALUDE_item_distribution_l2559_255907


namespace NUMINAMATH_CALUDE_sample_size_l2559_255960

theorem sample_size (n : ℕ) (f₁ f₂ f₃ f₄ f₅ f₆ : ℕ) : 
  f₁ + f₂ + f₃ + f₄ + f₅ + f₆ = n →
  f₁ = 2 * (f₆) →
  f₂ = 3 * (f₆) →
  f₃ = 4 * (f₆) →
  f₄ = 6 * (f₆) →
  f₅ = 4 * (f₆) →
  f₁ + f₂ + f₃ = 27 →
  n = 60 := by
sorry

end NUMINAMATH_CALUDE_sample_size_l2559_255960


namespace NUMINAMATH_CALUDE_diagonal_length_from_offsets_and_area_l2559_255969

/-- The length of a diagonal of a quadrilateral, given its offsets and area -/
theorem diagonal_length_from_offsets_and_area 
  (offset1 : ℝ) (offset2 : ℝ) (area : ℝ) :
  offset1 = 7 →
  offset2 = 3 →
  area = 50 →
  ∃ (d : ℝ), d = 10 ∧ area = (1/2) * d * (offset1 + offset2) :=
by sorry

end NUMINAMATH_CALUDE_diagonal_length_from_offsets_and_area_l2559_255969


namespace NUMINAMATH_CALUDE_rectangle_area_breadth_ratio_l2559_255921

/-- Proves that for a rectangular plot with breadth 11 metres and length 10 metres more than its breadth, 
    the area of the plot divided by its breadth equals 21. -/
theorem rectangle_area_breadth_ratio : 
  ∀ (length breadth area : ℝ),
    breadth = 11 →
    length = breadth + 10 →
    area = length * breadth →
    area / breadth = 21 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_breadth_ratio_l2559_255921


namespace NUMINAMATH_CALUDE_no_solution_implies_a_zero_l2559_255934

/-- A system of equations with no solutions implies a = 0 -/
theorem no_solution_implies_a_zero 
  (h : ∀ (x y : ℝ), (y^2 = x^2 + a*x + b ∧ x^2 = y^2 + a*y + b) → False) :
  a = 0 :=
by sorry

end NUMINAMATH_CALUDE_no_solution_implies_a_zero_l2559_255934


namespace NUMINAMATH_CALUDE_cot_thirty_degrees_l2559_255988

theorem cot_thirty_degrees : Real.cos (π / 6) / Real.sin (π / 6) = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_cot_thirty_degrees_l2559_255988


namespace NUMINAMATH_CALUDE_pizza_toppings_l2559_255948

theorem pizza_toppings (total_slices : ℕ) (pepperoni_slices : ℕ) (mushroom_slices : ℕ)
  (h_total : total_slices = 24)
  (h_pep : pepperoni_slices = 15)
  (h_mush : mushroom_slices = 16)
  (h_at_least_one : ∀ slice, slice ≤ total_slices → (slice ≤ pepperoni_slices ∨ slice ≤ mushroom_slices)) :
  ∃ both_toppings : ℕ, 
    both_toppings = pepperoni_slices + mushroom_slices - total_slices ∧
    both_toppings = 7 := by
  sorry

end NUMINAMATH_CALUDE_pizza_toppings_l2559_255948


namespace NUMINAMATH_CALUDE_complement_of_B_in_U_l2559_255999

-- Define the universal set U
def U : Set Nat := {1, 2, 3, 4, 5}

-- Define set A
def A : Set Nat := {2, 3, 5}

-- Define set B
def B : Set Nat := {2, 5}

-- Theorem statement
theorem complement_of_B_in_U :
  U \ B = {1, 3, 4} := by sorry

end NUMINAMATH_CALUDE_complement_of_B_in_U_l2559_255999


namespace NUMINAMATH_CALUDE_percentage_relation_l2559_255993

/-- Given three real numbers A, B, and C, where A is 6% of C and 20% of B,
    prove that B is 30% of C. -/
theorem percentage_relation (A B C : ℝ) 
  (h1 : A = 0.06 * C) 
  (h2 : A = 0.20 * B) : 
  B = 0.30 * C := by
  sorry

end NUMINAMATH_CALUDE_percentage_relation_l2559_255993


namespace NUMINAMATH_CALUDE_polynomial_division_remainder_l2559_255929

def p (x : ℝ) : ℝ := 2 * x^3 - 5 * x^2 - 12 * x + 7
def d (x : ℝ) : ℝ := 2 * x + 3
def q (x : ℝ) : ℝ := x^2 - 4 * x + 2
def r (x : ℝ) : ℝ := -4 * x + 1

theorem polynomial_division_remainder :
  ∀ x : ℝ, p x = d x * q x + r x :=
sorry

end NUMINAMATH_CALUDE_polynomial_division_remainder_l2559_255929


namespace NUMINAMATH_CALUDE_floor_inequality_iff_equal_l2559_255933

theorem floor_inequality_iff_equal (m n : ℕ+) :
  (∀ α β : ℝ, ⌊(m + n : ℝ) * α⌋ + ⌊(m + n : ℝ) * β⌋ ≥ ⌊(m : ℝ) * α⌋ + ⌊(m : ℝ) * β⌋ + ⌊(n : ℝ) * (α + β)⌋) ↔
  m = n :=
by sorry

end NUMINAMATH_CALUDE_floor_inequality_iff_equal_l2559_255933


namespace NUMINAMATH_CALUDE_cookie_shop_problem_l2559_255922

def num_cookie_flavors : ℕ := 7
def num_milk_types : ℕ := 4
def total_products : ℕ := 4

def ways_charlie_buys (k : ℕ) : ℕ := Nat.choose (num_cookie_flavors + num_milk_types) k

def ways_delta_buys_distinct (k : ℕ) : ℕ := Nat.choose num_cookie_flavors k

def ways_delta_buys_with_repeats (k : ℕ) : ℕ :=
  if k = 1 then num_cookie_flavors
  else if k = 2 then ways_delta_buys_distinct 2 + num_cookie_flavors
  else if k = 3 then ways_delta_buys_distinct 3 + num_cookie_flavors * (num_cookie_flavors - 1) + num_cookie_flavors
  else if k = 4 then ways_delta_buys_distinct 4 + num_cookie_flavors * (num_cookie_flavors - 1) + 
                     (num_cookie_flavors * (num_cookie_flavors - 1)) / 2 + num_cookie_flavors
  else 0

def total_ways : ℕ :=
  (ways_charlie_buys 4) +
  (ways_charlie_buys 3 * ways_delta_buys_with_repeats 1) +
  (ways_charlie_buys 2 * ways_delta_buys_with_repeats 2) +
  (ways_charlie_buys 1 * ways_delta_buys_with_repeats 3) +
  (ways_delta_buys_with_repeats 4)

theorem cookie_shop_problem : total_ways = 4054 := by sorry

end NUMINAMATH_CALUDE_cookie_shop_problem_l2559_255922


namespace NUMINAMATH_CALUDE_carA_distance_at_2016th_meeting_l2559_255935

/-- Represents a car with its current speed and direction -/
structure Car where
  speed : ℝ
  direction : Bool

/-- Represents the state of the system at any given time -/
structure State where
  carA : Car
  carB : Car
  positionA : ℝ
  positionB : ℝ
  meetingCount : ℕ
  distanceTraveledA : ℝ

/-- The distance between points A and B -/
def distance : ℝ := 900

/-- Function to update the state after each meeting -/
def updateState (s : State) : State :=
  -- Implementation details omitted
  sorry

/-- Theorem stating the total distance traveled by Car A at the 2016th meeting -/
theorem carA_distance_at_2016th_meeting :
  ∃ (finalState : State),
    finalState.meetingCount = 2016 ∧
    finalState.distanceTraveledA = 1813900 :=
by
  sorry

end NUMINAMATH_CALUDE_carA_distance_at_2016th_meeting_l2559_255935


namespace NUMINAMATH_CALUDE_gary_initial_stickers_l2559_255967

/-- The number of stickers Gary gave to Lucy -/
def stickers_to_lucy : ℕ := 42

/-- The number of stickers Gary gave to Alex -/
def stickers_to_alex : ℕ := 26

/-- The number of stickers Gary had left -/
def stickers_left : ℕ := 31

/-- The initial number of stickers Gary had -/
def initial_stickers : ℕ := stickers_to_lucy + stickers_to_alex + stickers_left

theorem gary_initial_stickers :
  initial_stickers = 99 :=
by sorry

end NUMINAMATH_CALUDE_gary_initial_stickers_l2559_255967


namespace NUMINAMATH_CALUDE_b_invests_after_six_months_l2559_255931

/-- A partnership model with three partners --/
structure Partnership where
  x : ℝ  -- A's investment
  m : ℝ  -- Months after which B invests
  total_gain : ℝ  -- Total annual gain
  a_share : ℝ  -- A's share of the gain

/-- The investment-time products for each partner --/
def investment_time (p : Partnership) : ℝ × ℝ × ℝ :=
  (p.x * 12, 2 * p.x * (12 - p.m), 3 * p.x * 4)

/-- The total investment-time product --/
def total_investment_time (p : Partnership) : ℝ :=
  let (a, b, c) := investment_time p
  a + b + c

/-- Theorem stating that B invests after 6 months --/
theorem b_invests_after_six_months (p : Partnership) 
  (h1 : p.total_gain = 12000)
  (h2 : p.a_share = 4000)
  (h3 : p.a_share / p.total_gain = 1 / 3)
  (h4 : p.x * 12 = (1 / 3) * total_investment_time p) :
  p.m = 6 := by
  sorry


end NUMINAMATH_CALUDE_b_invests_after_six_months_l2559_255931


namespace NUMINAMATH_CALUDE_employment_percentage_l2559_255970

theorem employment_percentage (total_population : ℝ) 
  (employed_males_percentage : ℝ) (employed_females_ratio : ℝ) :
  employed_males_percentage = 36 →
  employed_females_ratio = 50 →
  (employed_males_percentage / employed_females_ratio) * 100 = 72 :=
by
  sorry

end NUMINAMATH_CALUDE_employment_percentage_l2559_255970


namespace NUMINAMATH_CALUDE_sum_of_squares_and_products_l2559_255937

theorem sum_of_squares_and_products (a b c : ℝ) (h1 : 0 ≤ a) (h2 : 0 ≤ b) (h3 : 0 ≤ c)
  (h4 : a^2 + b^2 + c^2 = 58) (h5 : a*b + b*c + c*a = 32) :
  a + b + c = Real.sqrt 122 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_and_products_l2559_255937


namespace NUMINAMATH_CALUDE_count_integers_satisfying_inequality_l2559_255920

/-- The number of integers satisfying (x - 2)^2 ≤ 4 is 5 -/
theorem count_integers_satisfying_inequality : 
  (Finset.filter (fun x => (x - 2)^2 ≤ 4) (Finset.range 100)).card = 5 := by
  sorry

end NUMINAMATH_CALUDE_count_integers_satisfying_inequality_l2559_255920


namespace NUMINAMATH_CALUDE_arithmetic_sequence_seventh_term_l2559_255906

/-- An arithmetic sequence with the given properties has its 7th term equal to 19 -/
theorem arithmetic_sequence_seventh_term (n : ℕ) (a d : ℚ) 
  (h1 : n > 7)
  (h2 : 5 * a + 10 * d = 34)
  (h3 : 5 * a + 5 * (n - 1) * d = 146)
  (h4 : n * (2 * a + (n - 1) * d) / 2 = 234) :
  a + 6 * d = 19 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_seventh_term_l2559_255906


namespace NUMINAMATH_CALUDE_range_of_x_l2559_255903

theorem range_of_x (a b c x : ℝ) (h : a^2 + 2*b^2 + 3*c^2 = 6) 
  (h2 : a + 2*b + 3*c > |x + 1|) : -7 < x ∧ x < 5 := by
  sorry

end NUMINAMATH_CALUDE_range_of_x_l2559_255903


namespace NUMINAMATH_CALUDE_factor_expression_l2559_255992

theorem factor_expression (y : ℝ) : 64 - 16 * y^2 = 16 * (2 - y) * (2 + y) := by
  sorry

end NUMINAMATH_CALUDE_factor_expression_l2559_255992


namespace NUMINAMATH_CALUDE_third_root_of_cubic_l2559_255963

theorem third_root_of_cubic (c d : ℚ) :
  (∀ x : ℚ, c * x^3 + (c + 3*d) * x^2 + (2*d - 4*c) * x + (10 - c) = 0 ↔ x = -1 ∨ x = 4 ∨ x = 76/11) :=
by sorry

end NUMINAMATH_CALUDE_third_root_of_cubic_l2559_255963


namespace NUMINAMATH_CALUDE_georges_socks_l2559_255987

/-- The number of socks George's dad gave him -/
def socks_from_dad (initial_socks bought_socks total_socks : ℝ) : ℝ :=
  total_socks - (initial_socks + bought_socks)

/-- Proof that George's dad gave him 4 socks -/
theorem georges_socks : socks_from_dad 28 36 68 = 4 := by
  sorry

end NUMINAMATH_CALUDE_georges_socks_l2559_255987


namespace NUMINAMATH_CALUDE_sequence_bounded_l2559_255940

/-- A sequence of non-negative real numbers satisfying certain conditions is bounded -/
theorem sequence_bounded (c : ℝ) (a : ℕ → ℝ) 
  (hc : c > 2)
  (ha_nonneg : ∀ n, a n ≥ 0)
  (h1 : ∀ m n : ℕ, a (m + n) ≤ 2 * a m + 2 * a n)
  (h2 : ∀ k : ℕ, a (2^k) ≤ 1 / ((k + 1 : ℝ)^c)) :
  ∃ M : ℝ, ∀ n : ℕ, a n ≤ M :=
sorry

end NUMINAMATH_CALUDE_sequence_bounded_l2559_255940


namespace NUMINAMATH_CALUDE_sin_double_angle_tangent_two_l2559_255915

theorem sin_double_angle_tangent_two (α : Real) (h : Real.tan α = 2) : 
  Real.sin (2 * α) = 4/5 := by
  sorry

end NUMINAMATH_CALUDE_sin_double_angle_tangent_two_l2559_255915


namespace NUMINAMATH_CALUDE_binomial_coefficient_congruence_l2559_255968

theorem binomial_coefficient_congruence (n : ℕ+) :
  ∃ σ : Fin (2^(n.val-1)) ≃ Fin (2^(n.val-1)),
    ∀ k : Fin (2^(n.val-1)),
      (Nat.choose (2^n.val - 1) k) ≡ (2 * σ k + 1) [MOD 2^n.val] := by
sorry

end NUMINAMATH_CALUDE_binomial_coefficient_congruence_l2559_255968


namespace NUMINAMATH_CALUDE_hyperbola_parabola_intersection_l2559_255998

/-- The value of p for which the left focus of the hyperbola 
    x²/3 - 16y²/p² = 1 (p > 0) lies on the latus rectum of 
    the parabola y² = 2px -/
theorem hyperbola_parabola_intersection (p : ℝ) : 
  p > 0 → 
  (∃ x y : ℝ, x^2/3 - 16*y^2/p^2 = 1) → 
  (∃ x y : ℝ, y^2 = 2*p*x) → 
  (∃ x : ℝ, x^2/3 - 16*0^2/p^2 = 1 ∧ 0^2 = 2*p*x) → 
  p = 4 := by sorry

end NUMINAMATH_CALUDE_hyperbola_parabola_intersection_l2559_255998


namespace NUMINAMATH_CALUDE_negation_false_implies_proposition_true_l2559_255911

theorem negation_false_implies_proposition_true (P : Prop) : 
  ¬(¬P) → P :=
sorry

end NUMINAMATH_CALUDE_negation_false_implies_proposition_true_l2559_255911


namespace NUMINAMATH_CALUDE_canoe_water_removal_rate_l2559_255995

theorem canoe_water_removal_rate 
  (distance : ℝ) 
  (paddling_speed : ℝ) 
  (water_intake_rate : ℝ) 
  (sinking_threshold : ℝ) 
  (h1 : distance = 2) 
  (h2 : paddling_speed = 3) 
  (h3 : water_intake_rate = 8) 
  (h4 : sinking_threshold = 40) : 
  ∃ (min_removal_rate : ℝ), 
    min_removal_rate = 7 ∧ 
    ∀ (removal_rate : ℝ), 
      removal_rate ≥ min_removal_rate → 
      (water_intake_rate - removal_rate) * (distance / paddling_speed * 60) ≤ sinking_threshold :=
by sorry

end NUMINAMATH_CALUDE_canoe_water_removal_rate_l2559_255995


namespace NUMINAMATH_CALUDE_janet_oranges_l2559_255923

theorem janet_oranges (sharon_oranges : ℕ) (total_oranges : ℕ) (h1 : sharon_oranges = 7) (h2 : total_oranges = 16) :
  total_oranges - sharon_oranges = 9 :=
by sorry

end NUMINAMATH_CALUDE_janet_oranges_l2559_255923


namespace NUMINAMATH_CALUDE_maggies_age_l2559_255917

theorem maggies_age (kate_age sue_age maggie_age : ℕ) 
  (total_age : kate_age + sue_age + maggie_age = 48)
  (kate : kate_age = 19)
  (sue : sue_age = 12) :
  maggie_age = 17 := by
  sorry

end NUMINAMATH_CALUDE_maggies_age_l2559_255917


namespace NUMINAMATH_CALUDE_arithmetic_sequence_ratio_l2559_255953

/-- An arithmetic sequence with a_5 = 5a_3 has S_9/S_5 = 9 -/
theorem arithmetic_sequence_ratio (a : ℕ → ℝ) (S : ℕ → ℝ) :
  (∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)) →  -- arithmetic sequence condition
  (∀ n, S n = (n / 2) * (2 * a 1 + (n - 1) * (a 2 - a 1))) →  -- sum formula
  a 5 = 5 * a 3 →  -- given condition
  S 9 / S 5 = 9 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_ratio_l2559_255953


namespace NUMINAMATH_CALUDE_find_t_l2559_255901

def A (t : ℝ) : Set ℝ := {-4, t^2}
def B (t : ℝ) : Set ℝ := {t-5, 9, 1-t}

theorem find_t : ∀ t : ℝ, 9 ∈ A t ∩ B t → t = -3 := by sorry

end NUMINAMATH_CALUDE_find_t_l2559_255901


namespace NUMINAMATH_CALUDE_max_min_x_values_l2559_255912

theorem max_min_x_values (x y z : ℝ) 
  (sum_zero : x + y + z = 0)
  (inequality : (x - y)^2 + (y - z)^2 + (z - x)^2 ≤ 2) :
  (∀ w, w = x → w ≤ 2/3) ∧ 
  (∃ v, v = x ∧ v = 2/3) ∧
  (∀ u, u = x → u ≥ -2/3) ∧
  (∃ t, t = x ∧ t = -2/3) :=
sorry

end NUMINAMATH_CALUDE_max_min_x_values_l2559_255912


namespace NUMINAMATH_CALUDE_stamp_problem_l2559_255977

theorem stamp_problem (x y : ℕ) : 
  (x + y > 400) →
  (∃ k : ℕ, x - k = (13 : ℚ) / 19 * (y + k)) →
  (∃ k : ℕ, y - k = (11 : ℚ) / 17 * (x + k)) →
  x = 227 ∧ y = 221 :=
by sorry

end NUMINAMATH_CALUDE_stamp_problem_l2559_255977


namespace NUMINAMATH_CALUDE_years_of_writing_comics_l2559_255905

/-- Represents the number of comics written in a year -/
def comics_per_year : ℕ := 182

/-- Represents the total number of comics written -/
def total_comics : ℕ := 730

/-- Theorem: Given the conditions, the number of years of writing comics is 4 -/
theorem years_of_writing_comics : 
  (total_comics / comics_per_year : ℕ) = 4 := by sorry

end NUMINAMATH_CALUDE_years_of_writing_comics_l2559_255905
