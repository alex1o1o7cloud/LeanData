import Mathlib

namespace cosine_of_angle_between_vectors_l1088_108806

def vector_a : Fin 3 → ℝ := ![1, 1, 2]
def vector_b : Fin 3 → ℝ := ![2, -1, 2]

theorem cosine_of_angle_between_vectors :
  let a := vector_a
  let b := vector_b
  let dot_product := (a 0) * (b 0) + (a 1) * (b 1) + (a 2) * (b 2)
  let magnitude_a := Real.sqrt ((a 0)^2 + (a 1)^2 + (a 2)^2)
  let magnitude_b := Real.sqrt ((b 0)^2 + (b 1)^2 + (b 2)^2)
  dot_product / (magnitude_a * magnitude_b) = (5 * Real.sqrt 6) / 18 :=
by sorry

end cosine_of_angle_between_vectors_l1088_108806


namespace boys_in_line_l1088_108815

/-- If a boy in a single line is 19th from both ends, then the total number of boys is 37 -/
theorem boys_in_line (n : ℕ) (h : n > 0) : 
  (∃ k : ℕ, k > 0 ∧ k ≤ n ∧ k = 19 ∧ n - k + 1 = 19) → n = 37 := by
  sorry

end boys_in_line_l1088_108815


namespace no_solutions_absolute_value_equation_l1088_108832

theorem no_solutions_absolute_value_equation :
  ¬ ∃ x : ℝ, |x - 2| = |x - 1| + |x - 4| := by
sorry

end no_solutions_absolute_value_equation_l1088_108832


namespace water_consumption_proof_l1088_108862

/-- Represents the daily dosage of a medication -/
structure MedicationSchedule where
  name : String
  timesPerDay : ℕ

/-- Represents the adherence to medication schedule -/
structure Adherence where
  medication : MedicationSchedule
  missedDoses : ℕ

def waterPerDose : ℕ := 4

def daysInWeek : ℕ := 7

def numWeeks : ℕ := 2

def medicationA : MedicationSchedule := ⟨"A", 3⟩
def medicationB : MedicationSchedule := ⟨"B", 4⟩
def medicationC : MedicationSchedule := ⟨"C", 2⟩

def adherenceA : Adherence := ⟨medicationA, 1⟩
def adherenceB : Adherence := ⟨medicationB, 2⟩
def adherenceC : Adherence := ⟨medicationC, 2⟩

def totalWaterConsumed : ℕ := 484

/-- Theorem stating that the total water consumed with medications over two weeks is 484 ounces -/
theorem water_consumption_proof :
  (medicationA.timesPerDay + medicationB.timesPerDay + medicationC.timesPerDay) * daysInWeek * numWeeks * waterPerDose -
  (adherenceA.missedDoses + adherenceB.missedDoses + adherenceC.missedDoses) * waterPerDose = totalWaterConsumed := by
  sorry

end water_consumption_proof_l1088_108862


namespace gondor_wednesday_laptops_l1088_108874

/-- Represents the earnings and repair data for Gondor --/
structure RepairData where
  phone_repair_fee : ℕ
  laptop_repair_fee : ℕ
  monday_phones : ℕ
  tuesday_phones : ℕ
  thursday_laptops : ℕ
  total_earnings : ℕ

/-- Calculates the number of laptops repaired on Wednesday --/
def laptops_repaired_wednesday (data : RepairData) : ℕ :=
  let phone_earnings := data.phone_repair_fee * (data.monday_phones + data.tuesday_phones)
  let thursday_laptop_earnings := data.laptop_repair_fee * data.thursday_laptops
  let wednesday_laptop_earnings := data.total_earnings - phone_earnings - thursday_laptop_earnings
  wednesday_laptop_earnings / data.laptop_repair_fee

/-- Theorem stating that Gondor repaired 2 laptops on Wednesday --/
theorem gondor_wednesday_laptops :
  laptops_repaired_wednesday {
    phone_repair_fee := 10,
    laptop_repair_fee := 20,
    monday_phones := 3,
    tuesday_phones := 5,
    thursday_laptops := 4,
    total_earnings := 200
  } = 2 := by
  sorry

end gondor_wednesday_laptops_l1088_108874


namespace right_triangle_leg_lengths_l1088_108805

/-- 
Given a right triangle where the height from the right angle vertex 
divides the hypotenuse into segments of lengths a and b, 
the lengths of the legs are √(a(a+b)) and √(b(a+b)).
-/
theorem right_triangle_leg_lengths 
  (a b : ℝ) 
  (ha : 0 < a) 
  (hb : 0 < b) : 
  ∃ (leg1 leg2 : ℝ), 
    leg1 = Real.sqrt (a * (a + b)) ∧ 
    leg2 = Real.sqrt (b * (a + b)) ∧
    leg1^2 + leg2^2 = (a + b)^2 := by
  sorry

end right_triangle_leg_lengths_l1088_108805


namespace harmonic_mean_of_1_and_5040_l1088_108898

def harmonic_mean (a b : ℚ) : ℚ := 2 * a * b / (a + b)

theorem harmonic_mean_of_1_and_5040 :
  harmonic_mean 1 5040 = 10080 / 5041 := by
  sorry

end harmonic_mean_of_1_and_5040_l1088_108898


namespace pizza_toppings_combinations_l1088_108834

theorem pizza_toppings_combinations :
  Nat.choose 9 3 = 84 := by
  sorry

end pizza_toppings_combinations_l1088_108834


namespace parabola_symmetry_l1088_108811

/-- Given a parabola y = x^2 + 3x + m in the Cartesian coordinate system,
    prove that when translated 5 units to the right,
    the original and translated parabolas are symmetric about the line x = 1 -/
theorem parabola_symmetry (m : ℝ) :
  let f (x : ℝ) := x^2 + 3*x + m
  let g (x : ℝ) := f (x - 5)
  ∀ (x y : ℝ), f (1 - (x - 1)) = g (1 + (x - 1)) := by sorry

end parabola_symmetry_l1088_108811


namespace perpendicular_lines_slope_l1088_108877

/-- Given two lines l and l₁ in a 2D coordinate system:
    - l has a slope of -1
    - l₁ passes through points (3,2) and (a,-1)
    - l and l₁ are perpendicular
    Then a = 6 -/
theorem perpendicular_lines_slope (a : ℝ) : 
  let slope_l : ℝ := -1
  let point_A : ℝ × ℝ := (3, 2)
  let point_B : ℝ × ℝ := (a, -1)
  let slope_l₁ : ℝ := (point_B.2 - point_A.2) / (point_B.1 - point_A.1)
  slope_l * slope_l₁ = -1 → a = 6 := by
  sorry

end perpendicular_lines_slope_l1088_108877


namespace floor_negative_seven_halves_l1088_108844

theorem floor_negative_seven_halves : 
  ⌊(-7 : ℚ) / 2⌋ = -4 := by sorry

end floor_negative_seven_halves_l1088_108844


namespace inverse_variation_problem_l1088_108871

/-- Given that x varies inversely with y³ and x = 8 when y = 1, prove that x = 1 when y = 2 -/
theorem inverse_variation_problem (x y : ℝ) (k : ℝ) : 
  (∀ y, x * y^3 = k) →  -- x varies inversely with y³
  (8 * 1^3 = k) →       -- x = 8 when y = 1
  (1 * 2^3 = k) →       -- x = 1 when y = 2
  True := by sorry

end inverse_variation_problem_l1088_108871


namespace is_reflection_l1088_108863

def reflection_matrix (a b : ℚ) : Matrix (Fin 2) (Fin 2) ℚ :=
  ![![a, b],
    ![3/7, -4/7]]

theorem is_reflection : 
  let R := reflection_matrix (4/7) (-16/21)
  R * R = 1 :=
sorry

end is_reflection_l1088_108863


namespace y_satisfies_equation_l1088_108882

open Real

/-- The function y(x) -/
noncomputable def y (x : ℝ) : ℝ := 2 * (sin x / x) + cos x

/-- The differential equation -/
def differential_equation (y : ℝ → ℝ) (x : ℝ) : Prop :=
  x * sin x * deriv y x + (sin x - x * cos x) * y x = sin x * cos x - x

/-- Theorem stating that y satisfies the differential equation -/
theorem y_satisfies_equation : ∀ x : ℝ, x ≠ 0 → differential_equation y x := by
  sorry

end y_satisfies_equation_l1088_108882


namespace cloth_cost_l1088_108836

/-- Given a piece of cloth with the following properties:
  1. The original length is 10 meters.
  2. Increasing the length by 4 meters and decreasing the cost per meter by 1 rupee
     leaves the total cost unchanged.
  This theorem proves that the total cost of the original piece is 35 rupees. -/
theorem cloth_cost (original_length : ℝ) (cost_per_meter : ℝ) 
  (h1 : original_length = 10)
  (h2 : original_length * cost_per_meter = (original_length + 4) * (cost_per_meter - 1)) :
  original_length * cost_per_meter = 35 := by
sorry

end cloth_cost_l1088_108836


namespace angle_opposite_geometric_mean_side_at_most_60_degrees_l1088_108860

/-- 
If in a triangle ABC, side a is the geometric mean of sides b and c,
then the angle A opposite to side a is less than or equal to 60°.
-/
theorem angle_opposite_geometric_mean_side_at_most_60_degrees 
  (a b c : ℝ) (h_positive : 0 < a ∧ 0 < b ∧ 0 < c) (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b)
  (h_geometric_mean : a^2 = b*c) : 
  let A := Real.arccos ((b^2 + c^2 - a^2) / (2*b*c))
  A ≤ π/3 := by sorry

end angle_opposite_geometric_mean_side_at_most_60_degrees_l1088_108860


namespace sixth_term_of_geometric_sequence_l1088_108878

def geometric_sequence (a : ℕ → ℚ) : Prop :=
  ∃ r : ℚ, ∀ n : ℕ, a (n + 1) = a n * r

theorem sixth_term_of_geometric_sequence
  (a : ℕ → ℚ)
  (h_geom : geometric_sequence a)
  (h_first : a 1 = 27)
  (h_fourth : a 4 = a 3 * a 5) :
  a 6 = 1 / 9 := by
sorry

end sixth_term_of_geometric_sequence_l1088_108878


namespace fraction_of_25_l1088_108812

theorem fraction_of_25 : 
  ∃ x : ℚ, x * 25 = (80 / 100) * 40 - 22 ∧ x = 2 / 5 := by
  sorry

end fraction_of_25_l1088_108812


namespace roots_on_circle_l1088_108843

theorem roots_on_circle : ∃ (r : ℝ), r = 2/3 ∧
  ∀ (z : ℂ), (z - 2)^6 = 64*z^6 → Complex.abs (z - 2/3) = r := by
  sorry

end roots_on_circle_l1088_108843


namespace rhombus_area_l1088_108803

/-- The area of a rhombus with diagonals of 15 cm and 21 cm is 157.5 cm². -/
theorem rhombus_area (d1 d2 area : ℝ) (h1 : d1 = 15) (h2 : d2 = 21) 
    (h3 : area = (d1 * d2) / 2) : area = 157.5 := by
  sorry

end rhombus_area_l1088_108803


namespace wire_cut_ratio_l1088_108828

/-- Given a wire cut into two pieces of lengths x and y, where x forms a square and y forms a regular pentagon with equal perimeters, prove that x/y = 1 -/
theorem wire_cut_ratio (x y : ℝ) (h : x > 0 ∧ y > 0) : 
  (4 * (x / 4) = 5 * (y / 5)) → x / y = 1 := by
  sorry

end wire_cut_ratio_l1088_108828


namespace player_jump_height_to_dunk_l1088_108853

/-- Represents the height of a basketball player in feet -/
def player_height : ℝ := 6

/-- Represents the additional reach of the player above their head in inches -/
def player_reach : ℝ := 22

/-- Represents the height of the basketball rim in feet -/
def rim_height : ℝ := 10

/-- Represents the additional height above the rim needed to dunk in inches -/
def dunk_clearance : ℝ := 6

/-- Converts feet to inches -/
def feet_to_inches (feet : ℝ) : ℝ := feet * 12

/-- Calculates the jump height required for the player to dunk -/
def required_jump_height : ℝ :=
  feet_to_inches rim_height + dunk_clearance - (feet_to_inches player_height + player_reach)

theorem player_jump_height_to_dunk :
  required_jump_height = 32 := by sorry

end player_jump_height_to_dunk_l1088_108853


namespace number_problem_l1088_108801

theorem number_problem : ∃ x : ℚ, x - (3/5) * x = 58 ∧ x = 145 := by
  sorry

end number_problem_l1088_108801


namespace intersection_of_A_and_B_l1088_108822

def A : Set ℝ := {x | x - 1 ≤ 0}
def B : Set ℝ := {0, 1, 2}

theorem intersection_of_A_and_B : A ∩ B = {0, 1} := by sorry

end intersection_of_A_and_B_l1088_108822


namespace ayen_extra_minutes_friday_l1088_108869

/-- Represents Ayen's jogging routine for a week --/
structure JoggingRoutine where
  regularMinutes : ℕ  -- Regular minutes jogged per weekday
  weekdays : ℕ        -- Number of weekdays
  tuesdayExtra : ℕ    -- Extra minutes jogged on Tuesday
  totalHours : ℕ      -- Total hours jogged in the week

/-- Calculates the extra minutes jogged on Friday --/
def extraMinutesFriday (routine : JoggingRoutine) : ℕ :=
  routine.totalHours * 60 - (routine.regularMinutes * routine.weekdays + routine.tuesdayExtra)

/-- Theorem stating that Ayen jogged an extra 25 minutes on Friday --/
theorem ayen_extra_minutes_friday :
  ∀ (routine : JoggingRoutine),
    routine.regularMinutes = 30 ∧
    routine.weekdays = 5 ∧
    routine.tuesdayExtra = 5 ∧
    routine.totalHours = 3 →
    extraMinutesFriday routine = 25 := by
  sorry

end ayen_extra_minutes_friday_l1088_108869


namespace equation_solution_l1088_108895

theorem equation_solution : 
  let f (x : ℝ) := (x^2 + 3*x - 4)^2 + (2*x^2 - 7*x + 6)^2 - (3*x^2 - 4*x + 2)^2
  ∀ x : ℝ, f x = 0 ↔ x = -4 ∨ x = 1 ∨ x = 3/2 ∨ x = 2 := by sorry

end equation_solution_l1088_108895


namespace line_passes_through_point_l1088_108880

/-- A line in the form y + 2 = k(x + 1) always passes through the point (-1, -2) -/
theorem line_passes_through_point (k : ℝ) : 
  let f : ℝ → ℝ := λ x ↦ k * (x + 1) - 2
  f (-1) = -2 := by
  sorry


end line_passes_through_point_l1088_108880


namespace area_calculation_l1088_108859

/-- The circle equation -/
def circle_equation (x y : ℝ) : Prop := x^2 - 16*x + y^2 = 60

/-- The line equation -/
def line_equation (x y : ℝ) : Prop := y = 4 - x

/-- The region of interest -/
def region_of_interest (x y : ℝ) : Prop :=
  circle_equation x y ∧ y ≥ 0 ∧ y > 4 - x

/-- The area of the region of interest -/
noncomputable def area_of_region : ℝ := sorry

theorem area_calculation : area_of_region = 77.5 * Real.pi := by sorry

end area_calculation_l1088_108859


namespace odd_functions_max_min_l1088_108890

-- Define the property of being an odd function
def IsOdd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

-- Define the function F
def F (f g : ℝ → ℝ) (x : ℝ) : ℝ := f x + g x + 2

-- State the theorem
theorem odd_functions_max_min (f g : ℝ → ℝ) (hf : IsOdd f) (hg : IsOdd g) 
  (hmax : ∃ M, M = 8 ∧ ∀ x > 0, F f g x ≤ M) :
  ∃ m, m = -4 ∧ ∀ x < 0, F f g x ≥ m :=
sorry

end odd_functions_max_min_l1088_108890


namespace division_of_A_by_1001_l1088_108852

/-- A number consisting of 1001 sevens -/
def A : ℕ := (10 ^ 1001 - 1) / 9 * 7

/-- The expected quotient when A is divided by 1001 -/
def expected_quotient : ℕ := (10 ^ 1001 - 1) / (9 * 1001) * 777

/-- The expected remainder when A is divided by 1001 -/
def expected_remainder : ℕ := 700

theorem division_of_A_by_1001 :
  (A / 1001 = expected_quotient) ∧ (A % 1001 = expected_remainder) :=
sorry

end division_of_A_by_1001_l1088_108852


namespace team_a_champion_probability_l1088_108830

/-- The probability of a team winning a single game -/
def win_prob : ℚ := 1/2

/-- The probability of Team A becoming the champion -/
def champion_prob : ℚ := win_prob + win_prob * win_prob

theorem team_a_champion_probability :
  champion_prob = 3/4 :=
by sorry

end team_a_champion_probability_l1088_108830


namespace drivers_distance_difference_l1088_108884

/-- Proves that the difference in distance traveled by two drivers meeting on a highway is 140 km -/
theorem drivers_distance_difference (initial_distance : ℝ) (speed_a : ℝ) (speed_b : ℝ) (delay : ℝ) : 
  initial_distance = 940 →
  speed_a = 90 →
  speed_b = 80 →
  delay = 1 →
  let remaining_distance := initial_distance - speed_a * delay
  let meeting_time := remaining_distance / (speed_a + speed_b)
  let distance_a := speed_a * (meeting_time + delay)
  let distance_b := speed_b * meeting_time
  distance_a - distance_b = 140 := by
  sorry

end drivers_distance_difference_l1088_108884


namespace closest_years_with_property_l1088_108849

def has_property (year : ℕ) : Prop :=
  let a := year / 1000
  let b := (year / 100) % 10
  let c := (year / 10) % 10
  let d := year % 10
  10 * a + b + 10 * c + d = 10 * b + c

theorem closest_years_with_property : 
  (∀ y : ℕ, 1868 < y ∧ y < 1978 → ¬(has_property y)) ∧ 
  (∀ y : ℕ, 1978 < y ∧ y < 2307 → ¬(has_property y)) ∧
  has_property 1868 ∧ 
  has_property 2307 :=
sorry

end closest_years_with_property_l1088_108849


namespace quarter_count_in_collection_l1088_108867

/-- Represents the types of coins --/
inductive CoinType
  | Penny
  | Nickel
  | Dime
  | Quarter
  | HalfDollar

/-- The value of each coin type in cents --/
def coinValue : CoinType → Nat
  | CoinType.Penny => 1
  | CoinType.Nickel => 5
  | CoinType.Dime => 10
  | CoinType.Quarter => 25
  | CoinType.HalfDollar => 50

/-- A collection of coins --/
structure CoinCollection where
  pennies : Nat
  nickels : Nat
  dimes : Nat
  quarters : Nat
  halfDollars : Nat

/-- The total value of a coin collection in cents --/
def totalValue (c : CoinCollection) : Nat :=
  c.pennies * coinValue CoinType.Penny +
  c.nickels * coinValue CoinType.Nickel +
  c.dimes * coinValue CoinType.Dime +
  c.quarters * coinValue CoinType.Quarter +
  c.halfDollars * coinValue CoinType.HalfDollar

/-- The total number of coins in a collection --/
def totalCoins (c : CoinCollection) : Nat :=
  c.pennies + c.nickels + c.dimes + c.quarters + c.halfDollars

theorem quarter_count_in_collection :
  ∀ c : CoinCollection,
    totalCoins c = 11 ∧
    totalValue c = 163 ∧
    c.pennies ≥ 1 ∧
    c.nickels ≥ 1 ∧
    c.dimes ≥ 1 ∧
    c.quarters ≥ 1 ∧
    c.halfDollars ≥ 1
    →
    c.quarters = 2 :=
by sorry

end quarter_count_in_collection_l1088_108867


namespace least_addition_for_divisibility_l1088_108824

theorem least_addition_for_divisibility : ∃! x : ℕ, 
  (x ≤ 22) ∧ (∃ k : ℤ, 1077 + x = 23 * k) ∧ 
  (∀ y : ℕ, y < x → ¬∃ k : ℤ, 1077 + y = 23 * k) :=
by sorry

end least_addition_for_divisibility_l1088_108824


namespace sports_club_overlap_l1088_108845

theorem sports_club_overlap (N B T X : ℕ) (h1 : N = 40) (h2 : B = 20) (h3 : T = 18) (h4 : X = 5) :
  B + T - (N - X) = 3 :=
by
  sorry

end sports_club_overlap_l1088_108845


namespace line_passes_through_fixed_point_l1088_108816

/-- The line mx + y - m - 1 = 0 passes through the point (1, 1) for all real m -/
theorem line_passes_through_fixed_point :
  ∀ (m : ℝ), (m * 1 + 1 - m - 1 = 0) := by sorry

end line_passes_through_fixed_point_l1088_108816


namespace messaging_packages_theorem_l1088_108858

/-- Represents a messaging package --/
structure Package where
  cost : ℕ
  people : ℕ

/-- Calculates the number of ways to connect n people using given packages --/
def countConnections (n : ℕ) (packages : List Package) : ℕ :=
  sorry

/-- Calculates the minimum cost to connect n people using given packages --/
def minCost (n : ℕ) (packages : List Package) : ℕ :=
  sorry

theorem messaging_packages_theorem :
  let n := 4  -- number of friends
  let packageA := Package.mk 10 3
  let packageB := Package.mk 5 2
  let packages := [packageA, packageB]
  (minCost n packages = 15) ∧
  (countConnections n packages = 28) := by
  sorry

end messaging_packages_theorem_l1088_108858


namespace final_value_is_four_l1088_108813

def program_execution (M : Nat) : Nat :=
  let M1 := M + 1
  let M2 := M1 + 2
  M2

theorem final_value_is_four : program_execution 1 = 4 := by
  sorry

end final_value_is_four_l1088_108813


namespace equivalence_condition_l1088_108821

theorem equivalence_condition (a : ℝ) : 
  (∀ x : ℝ, (5 - x) / (x - 2) ≥ 0 ↔ -3 < x ∧ x < a) ↔ a > 5 :=
by sorry

end equivalence_condition_l1088_108821


namespace cricket_team_size_l1088_108802

theorem cricket_team_size :
  ∀ (n : ℕ),
  (n : ℝ) * 23 = (n - 2 : ℝ) * 22 + 55 →
  n = 11 := by
sorry

end cricket_team_size_l1088_108802


namespace gcd_13n_plus_4_8n_plus_3_max_9_l1088_108870

theorem gcd_13n_plus_4_8n_plus_3_max_9 :
  (∃ (n : ℕ+), Nat.gcd (13 * n + 4) (8 * n + 3) = 9) ∧
  (∀ (n : ℕ+), Nat.gcd (13 * n + 4) (8 * n + 3) ≤ 9) := by
  sorry

end gcd_13n_plus_4_8n_plus_3_max_9_l1088_108870


namespace triangle_area_l1088_108861

/-- The area of a triangle with sides 5, 5, and 6 units is 12 square units. -/
theorem triangle_area : ∀ (a b c : ℝ), a = 5 ∧ b = 5 ∧ c = 6 →
  ∃ (s : ℝ), s = (a + b + c) / 2 ∧ Real.sqrt (s * (s - a) * (s - b) * (s - c)) = 12 := by
  sorry

end triangle_area_l1088_108861


namespace remaining_work_days_l1088_108826

/-- Given workers x and y, where x can finish a job in 24 days and y in 16 days,
    prove that x needs 9 days to finish the remaining work after y works for 10 days. -/
theorem remaining_work_days (x_days y_days y_worked_days : ℕ) 
  (hx : x_days = 24)
  (hy : y_days = 16)
  (hw : y_worked_days = 10) :
  (x_days : ℚ) * (1 - y_worked_days / y_days) = 9 := by
  sorry

end remaining_work_days_l1088_108826


namespace investment_split_l1088_108804

theorem investment_split (initial_investment : ℝ) (rate1 rate2 : ℝ) (years : ℕ) (final_amount : ℝ) :
  initial_investment = 2000 ∧
  rate1 = 0.04 ∧
  rate2 = 0.06 ∧
  years = 3 ∧
  final_amount = 2436.29 →
  ∃ (x : ℝ),
    x * (1 + rate1) ^ years + (initial_investment - x) * (1 + rate2) ^ years = final_amount ∧
    x = 820 := by
  sorry

end investment_split_l1088_108804


namespace equation_solution_l1088_108873

theorem equation_solution :
  let f (x : ℝ) := x^4 / (2*x + 1) + x^2 - 6*(2*x + 1)
  ∀ x : ℝ, f x = 0 ↔ x = -3 - Real.sqrt 6 ∨ x = -3 + Real.sqrt 6 ∨ x = 2 - Real.sqrt 6 ∨ x = 2 + Real.sqrt 6 :=
by sorry

end equation_solution_l1088_108873


namespace geometric_sequence_problem_l1088_108800

theorem geometric_sequence_problem (a : ℕ → ℝ) (q : ℝ) :
  (∀ n : ℕ, a (n + 1) = a n * q) →  -- geometric sequence definition
  q > 0 →  -- positive common ratio
  a 3 * a 9 = 2 * (a 5)^2 →  -- given condition
  a 2 = 2 →  -- given condition
  a 1 = Real.sqrt 2 := by  -- conclusion to prove
sorry


end geometric_sequence_problem_l1088_108800


namespace perfect_square_sum_l1088_108887

theorem perfect_square_sum (n : ℝ) (h : n > 2) :
  ∃ m : ℝ, ∃ k : ℝ, n^2 + m^2 = k^2 := by
  sorry

end perfect_square_sum_l1088_108887


namespace complex_equation_result_l1088_108868

theorem complex_equation_result (x y : ℂ) (hx : x ≠ 0) (hy : y ≠ 0) 
  (h : x^2 + x*y + y^2 = 0) : 
  (x / (x + y))^1990 + (y / (x + y))^1990 = -1 := by
  sorry

end complex_equation_result_l1088_108868


namespace johnson_farm_wheat_acreage_l1088_108846

/-- Proves that given the conditions of Johnson Farm, the number of acres of wheat planted is 200 -/
theorem johnson_farm_wheat_acreage :
  let total_land : ℝ := 500
  let corn_cost : ℝ := 42
  let wheat_cost : ℝ := 30
  let total_budget : ℝ := 18600
  let wheat_acres : ℝ := (total_land * corn_cost - total_budget) / (corn_cost - wheat_cost)
  wheat_acres = 200 ∧
  wheat_acres > 0 ∧
  wheat_acres < total_land ∧
  wheat_acres * wheat_cost + (total_land - wheat_acres) * corn_cost = total_budget :=
by sorry

end johnson_farm_wheat_acreage_l1088_108846


namespace broken_line_circle_cover_l1088_108881

/-- A closed broken line in a metric space -/
structure ClosedBrokenLine (X : Type*) [MetricSpace X] where
  points : Set X
  is_closed : IsClosed points
  perimeter : ℝ

/-- Theorem: Any closed broken line with perimeter 1 can be covered by a circle of radius 1/4 -/
theorem broken_line_circle_cover {X : Type*} [MetricSpace X] (L : ClosedBrokenLine X) 
  (h_perimeter : L.perimeter = 1) :
  ∃ (center : X), ∀ (p : X), p ∈ L.points → dist center p ≤ 1/4 := by
  sorry

end broken_line_circle_cover_l1088_108881


namespace anca_rest_time_l1088_108817

-- Define the constants
def bruce_speed : ℝ := 50
def anca_speed : ℝ := 60
def total_distance : ℝ := 200

-- Define the theorem
theorem anca_rest_time :
  let bruce_time := total_distance / bruce_speed
  let anca_drive_time := total_distance / anca_speed
  let rest_time := bruce_time - anca_drive_time
  rest_time * 60 = 40 := by
  sorry

end anca_rest_time_l1088_108817


namespace even_function_extension_l1088_108888

/-- A function f : ℝ → ℝ is even if f(-x) = f(x) for all x -/
def IsEven (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

theorem even_function_extension (f : ℝ → ℝ) (h_even : IsEven f) 
  (h_def : ∀ x > 0, f x = x * (1 + x)) :
  ∀ x < 0, f x = x * (x - 1) := by
  sorry

end even_function_extension_l1088_108888


namespace dividend_calculation_l1088_108809

theorem dividend_calculation (divisor quotient remainder : ℕ) 
  (h1 : divisor = 17) 
  (h2 : quotient = 9) 
  (h3 : remainder = 10) : 
  divisor * quotient + remainder = 163 := by
  sorry

end dividend_calculation_l1088_108809


namespace paco_initial_cookies_l1088_108865

/-- Proves that Paco had 40 cookies initially given the problem conditions -/
theorem paco_initial_cookies :
  ∀ x : ℕ,
  x - 2 + 37 = 75 →
  x = 40 :=
by
  sorry

end paco_initial_cookies_l1088_108865


namespace square_side_ratio_l1088_108891

theorem square_side_ratio (area_ratio : ℚ) :
  area_ratio = 45 / 64 →
  ∃ (a b c : ℕ), (a * Real.sqrt b) / c = Real.sqrt (area_ratio) ∧
                  a = 3 ∧ b = 5 ∧ c = 8 := by
  sorry

end square_side_ratio_l1088_108891


namespace arithmetic_sequence_properties_l1088_108886

/-- An arithmetic sequence with given properties -/
structure ArithmeticSequence where
  first : ℕ
  last : ℕ
  diff : ℕ
  h_first : first = 17
  h_last : last = 95
  h_diff : diff = 4

/-- The number of terms in the sequence -/
def numTerms (seq : ArithmeticSequence) : ℕ :=
  (seq.last - seq.first) / seq.diff + 1

/-- The sum of all terms in the sequence -/
def sumTerms (seq : ArithmeticSequence) : ℕ :=
  (numTerms seq * (seq.first + seq.last)) / 2

theorem arithmetic_sequence_properties (seq : ArithmeticSequence) :
  numTerms seq = 20 ∧ sumTerms seq = 1100 := by
  sorry

end arithmetic_sequence_properties_l1088_108886


namespace tangent_line_intersection_l1088_108892

/-- The function f(x) = x³ - x -/
def f (x : ℝ) : ℝ := x^3 - x

/-- The function g(x) = x² + a -/
def g (a : ℝ) (x : ℝ) : ℝ := x^2 + a

/-- The derivative of f -/
def f' (x : ℝ) : ℝ := 3 * x^2 - 1

/-- The tangent line of f at x = -1 -/
def tangent_line (x : ℝ) : ℝ := 2 * x + 2

theorem tangent_line_intersection (a : ℝ) : 
  (∀ x, tangent_line x = g a x) → a = 3 := by
  sorry

end tangent_line_intersection_l1088_108892


namespace geometric_series_sum_l1088_108838

/-- Given a geometric series with first term a and common ratio r,
    this function calculates the sum of the first n terms. -/
def geometricSum (a : ℚ) (r : ℚ) (n : ℕ) : ℚ :=
  a * (1 - r^n) / (1 - r)

/-- Theorem: For the geometric series with first term 7/8 and common ratio -1/2,
    the sum of the first four terms is 35/64. -/
theorem geometric_series_sum :
  let a : ℚ := 7/8
  let r : ℚ := -1/2
  let n : ℕ := 4
  geometricSum a r n = 35/64 := by
sorry

end geometric_series_sum_l1088_108838


namespace recurring_decimal_equals_fraction_l1088_108831

/-- Represents the decimal expansion 7.836836836... -/
def recurring_decimal : ℚ := 7 + 836 / 999

/-- The fraction representation of the recurring decimal -/
def fraction : ℚ := 7829 / 999

theorem recurring_decimal_equals_fraction :
  recurring_decimal = fraction := by sorry

end recurring_decimal_equals_fraction_l1088_108831


namespace box_two_three_l1088_108808

/-- Define the box operation -/
def box (a b : ℝ) : ℝ := a * (b^2 + 3) - b + 1

/-- Theorem: The value of (2) □ (3) is 22 -/
theorem box_two_three : box 2 3 = 22 := by
  sorry

end box_two_three_l1088_108808


namespace expression_simplification_l1088_108851

theorem expression_simplification (x : ℝ) (h : x = -3) :
  (x - 3) * (x + 4) - (x - x^2) = 6 := by
  sorry

end expression_simplification_l1088_108851


namespace solve_grocery_cost_l1088_108893

def grocery_cost_problem (initial_amount : ℝ) (sister_fraction : ℝ) (remaining_amount : ℝ) : Prop :=
  let amount_to_sister := initial_amount * sister_fraction
  let amount_after_giving := initial_amount - amount_to_sister
  let grocery_cost := amount_after_giving - remaining_amount
  grocery_cost = 40

theorem solve_grocery_cost :
  grocery_cost_problem 100 (1/4) 35 := by
  sorry

end solve_grocery_cost_l1088_108893


namespace rationalization_factor_l1088_108848

theorem rationalization_factor (a b : ℝ) (ha : 0 ≤ a) (hb : 0 ≤ b) :
  (Real.sqrt a + Real.sqrt b) * (Real.sqrt a - Real.sqrt b) = a - b ∧
  (Real.sqrt a + Real.sqrt b) * (Real.sqrt b - Real.sqrt a) = b - a :=
by sorry

end rationalization_factor_l1088_108848


namespace fibonacci_like_sequence_l1088_108829

theorem fibonacci_like_sequence (α β : ℝ) (h1 : α^2 - α - 1 = 0) (h2 : β^2 - β - 1 = 0) (h3 : α > β) :
  let s : ℕ → ℝ := λ n => α^n + β^n
  ∀ n ≥ 3, s n = s (n-1) + s (n-2) :=
by sorry

end fibonacci_like_sequence_l1088_108829


namespace power_equation_l1088_108818

theorem power_equation (k m n : ℕ) 
  (h1 : 3^(k - 1) = 81) 
  (h2 : 4^(m + 2) = 256) 
  (h3 : 5^(n - 3) = 625) : 
  2^(4*k - 3*m + 5*n) = 2^49 := by
sorry

end power_equation_l1088_108818


namespace triangle_max_side_sum_l1088_108833

theorem triangle_max_side_sum (A B C : ℝ) (a b c : ℝ) :
  0 < A ∧ 0 < B ∧ 0 < C ∧
  A + B + C = π ∧
  B = π / 3 ∧
  b = Real.sqrt 3 ∧
  0 ≤ a ∧ 0 ≤ b ∧ 0 ≤ c ∧
  a / Real.sin A = b / Real.sin B ∧
  b / Real.sin B = c / Real.sin C →
  (2 * a + c) ≤ 2 * Real.sqrt 7 :=
by sorry

end triangle_max_side_sum_l1088_108833


namespace expression_value_at_three_l1088_108866

theorem expression_value_at_three :
  let x : ℝ := 3
  (x^8 + 16*x^4 + 64) / (x^4 + 8) = 89 := by
sorry

end expression_value_at_three_l1088_108866


namespace congruence_theorem_l1088_108872

theorem congruence_theorem (n : ℕ+) :
  (122 ^ n.val - 102 ^ n.val - 21 ^ n.val) % 2020 = 2019 := by
  sorry

end congruence_theorem_l1088_108872


namespace smallest_three_digit_palindrome_times_111_not_five_digit_palindrome_l1088_108897

/-- Checks if a number is a three-digit palindrome -/
def isThreeDigitPalindrome (n : ℕ) : Prop :=
  100 ≤ n ∧ n ≤ 999 ∧ (n / 100 = n % 10)

/-- Checks if a number is a five-digit palindrome -/
def isFiveDigitPalindrome (n : ℕ) : Prop :=
  10000 ≤ n ∧ n ≤ 99999 ∧ (n / 10000 = n % 10) ∧ ((n / 1000) % 10 = (n / 10) % 10)

/-- The main theorem -/
theorem smallest_three_digit_palindrome_times_111_not_five_digit_palindrome :
  ∀ n : ℕ, isThreeDigitPalindrome n →
    (n < 111 ∨ isFiveDigitPalindrome (n * 111)) →
    ¬isThreeDigitPalindrome 111 ∨ isFiveDigitPalindrome (111 * 111) :=
by
  sorry

#check smallest_three_digit_palindrome_times_111_not_five_digit_palindrome

end smallest_three_digit_palindrome_times_111_not_five_digit_palindrome_l1088_108897


namespace class_assignment_arrangements_l1088_108875

theorem class_assignment_arrangements :
  let num_teachers : ℕ := 3
  let num_classes : ℕ := 6
  let classes_per_teacher : ℕ := 2
  let total_arrangements : ℕ := (Nat.choose num_classes classes_per_teacher) *
                                (Nat.choose (num_classes - classes_per_teacher) classes_per_teacher) *
                                (Nat.choose (num_classes - 2 * classes_per_teacher) classes_per_teacher) /
                                (Nat.factorial num_teachers)
  total_arrangements = 90 := by
  sorry

end class_assignment_arrangements_l1088_108875


namespace part1_part2_l1088_108856

/-- Represents a hotel accommodation scenario for a tour group -/
structure HotelAccommodation where
  totalPeople : ℕ
  singleRooms : ℕ
  tripleRooms : ℕ
  singleRoomPrice : ℕ
  tripleRoomPrice : ℕ
  menCount : ℕ

/-- Calculates the total cost for one night -/
def totalCost (h : HotelAccommodation) : ℕ :=
  h.singleRooms * h.singleRoomPrice + h.tripleRooms * h.tripleRoomPrice

/-- Part 1: Proves that given a total cost of 1530 yuan, the number of single rooms rented is 1 -/
theorem part1 (h : HotelAccommodation) 
    (hTotal : h.totalPeople = 33)
    (hSinglePrice : h.singleRoomPrice = 100)
    (hTriplePrice : h.tripleRoomPrice = 130)
    (hCost : totalCost h = 1530)
    (hSingleAvailable : h.singleRooms ≤ 4) :
  h.singleRooms = 1 := by
  sorry

/-- Part 2: Proves that given 3 single rooms and 19 men, the minimum cost is 1600 yuan -/
theorem part2 (h : HotelAccommodation) 
    (hTotal : h.totalPeople = 33)
    (hSinglePrice : h.singleRoomPrice = 100)
    (hTriplePrice : h.tripleRoomPrice = 130)
    (hSingleRooms : h.singleRooms = 3)
    (hMenCount : h.menCount = 19) :
  ∃ (minCost : ℕ), minCost = 1600 ∧ ∀ (cost : ℕ), totalCost h ≥ minCost := by
  sorry

end part1_part2_l1088_108856


namespace remainder_theorem_l1088_108807

theorem remainder_theorem : 2^9 * 3^10 + 14 ≡ 2 [ZMOD 25] := by
  sorry

end remainder_theorem_l1088_108807


namespace right_triangle_area_l1088_108841

theorem right_triangle_area (a b c : ℝ) (h1 : a^2 = 64) (h2 : b^2 = 49) (h3 : c^2 = 225) 
  (h4 : a^2 + b^2 = c^2) : (1/2) * a * b = 28 := by
  sorry

end right_triangle_area_l1088_108841


namespace new_refrigerator_cost_l1088_108896

/-- The daily cost of electricity for Kurt's old refrigerator in dollars -/
def old_cost : ℚ := 85/100

/-- The number of days in a month -/
def days_in_month : ℕ := 30

/-- The amount Kurt saves in a month with his new refrigerator in dollars -/
def monthly_savings : ℚ := 12

/-- The daily cost of electricity for Kurt's new refrigerator in dollars -/
def new_cost : ℚ := 45/100

theorem new_refrigerator_cost :
  (days_in_month : ℚ) * old_cost - (days_in_month : ℚ) * new_cost = monthly_savings :=
by sorry

end new_refrigerator_cost_l1088_108896


namespace rectangle_max_area_l1088_108847

theorem rectangle_max_area :
  ∀ l w : ℕ,
  l + w = 22 →
  l * w ≤ 121 :=
by
  sorry

end rectangle_max_area_l1088_108847


namespace smallest_satisfying_number_l1088_108850

/-- Returns the last four digits of a number in base 10 -/
def lastFourDigits (n : ℕ) : ℕ := n % 10000

/-- Checks if a number satisfies the conditions of the problem -/
def satisfiesConditions (n : ℕ) : Prop :=
  (n > 0) ∧ (lastFourDigits n = lastFourDigits (n^2)) ∧ ((n - 2) % 7 = 0)

theorem smallest_satisfying_number :
  satisfiesConditions 625 ∧ ∀ n < 625, ¬(satisfiesConditions n) := by
  sorry

end smallest_satisfying_number_l1088_108850


namespace no_negative_roots_l1088_108823

theorem no_negative_roots :
  ∀ x : ℝ, x < 0 → x^4 - 4*x^3 - 6*x^2 - 3*x + 9 ≠ 0 := by
  sorry

end no_negative_roots_l1088_108823


namespace min_distance_to_2i_l1088_108894

theorem min_distance_to_2i (z : ℂ) (h : Complex.abs (z^2 - 1) = Complex.abs (z * (z - Complex.I))) :
  ∃ (w : ℂ), Complex.abs (w - 2 * Complex.I) = 1 ∧ 
  ∀ (z : ℂ), Complex.abs (z - 2 * Complex.I) ≥ 1 := by
sorry

end min_distance_to_2i_l1088_108894


namespace fib_50_mod_5_l1088_108889

def fib : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | (n + 2) => fib (n + 1) + fib n

theorem fib_50_mod_5 : fib 50 % 5 = 0 := by
  sorry

end fib_50_mod_5_l1088_108889


namespace tank_capacity_correct_l1088_108899

/-- The capacity of a tank in litres. -/
def tank_capacity : ℝ := 1592

/-- The time in hours it takes for the leak to empty the full tank. -/
def leak_empty_time : ℝ := 7

/-- The rate at which the inlet pipe fills the tank in litres per minute. -/
def inlet_rate : ℝ := 6

/-- The time in hours it takes to empty the tank when both inlet and leak are open. -/
def combined_empty_time : ℝ := 12

/-- Theorem stating that the tank capacity is correct given the conditions. -/
theorem tank_capacity_correct : 
  tank_capacity = 
    (inlet_rate * 60 * combined_empty_time * leak_empty_time) / 
    (leak_empty_time - combined_empty_time) :=
by sorry

end tank_capacity_correct_l1088_108899


namespace distance_traveled_l1088_108864

/-- Given a speed of 25 km/hr and a time of 5 hr, the distance traveled is 125 km. -/
theorem distance_traveled (speed : ℝ) (time : ℝ) (distance : ℝ) 
  (h1 : speed = 25) 
  (h2 : time = 5) 
  (h3 : distance = speed * time) : 
  distance = 125 := by
  sorry

end distance_traveled_l1088_108864


namespace remainder_problem_l1088_108855

theorem remainder_problem (x : ℤ) : x % 63 = 11 → x % 9 = 2 := by
  sorry

end remainder_problem_l1088_108855


namespace train_passengers_l1088_108840

theorem train_passengers (adults_first : ℕ) (children_first : ℕ) 
  (adults_second : ℕ) (children_second : ℕ) (got_off : ℕ) (total : ℕ) : 
  children_first = adults_first - 17 →
  adults_second = 57 →
  children_second = 18 →
  got_off = 44 →
  total = 502 →
  adults_first + children_first + adults_second + children_second - got_off = total →
  adults_first = 244 := by
sorry

end train_passengers_l1088_108840


namespace modulo_congruence_unique_solution_l1088_108814

theorem modulo_congruence_unique_solution : ∃! n : ℤ, 0 ≤ n ∧ n < 25 ∧ -250 ≡ n [ZMOD 23] ∧ n = 3 := by
  sorry

end modulo_congruence_unique_solution_l1088_108814


namespace hours_difference_l1088_108857

/-- Represents the project and candidates' information -/
structure Project where
  total_pay : ℕ
  p_wage : ℕ
  q_wage : ℕ
  p_hours : ℕ
  q_hours : ℕ

/-- Conditions of the project -/
def project_conditions (proj : Project) : Prop :=
  proj.total_pay = 360 ∧
  proj.p_wage = proj.q_wage + proj.q_wage / 2 ∧
  proj.p_wage = proj.q_wage + 6 ∧
  proj.total_pay = proj.p_wage * proj.p_hours ∧
  proj.total_pay = proj.q_wage * proj.q_hours

/-- Theorem stating the difference in hours between candidates q and p -/
theorem hours_difference (proj : Project) 
  (h : project_conditions proj) : proj.q_hours - proj.p_hours = 10 := by
  sorry


end hours_difference_l1088_108857


namespace total_animals_l1088_108827

def farm_animals (pigs cows goats : ℕ) : Prop :=
  (pigs = 10) ∧
  (cows = 2 * pigs - 3) ∧
  (goats = cows + 6)

theorem total_animals :
  ∀ pigs cows goats : ℕ,
  farm_animals pigs cows goats →
  pigs + cows + goats = 50 :=
by
  sorry

end total_animals_l1088_108827


namespace M_intersect_complement_N_l1088_108842

def R := ℝ

def M : Set ℝ := {x : ℝ | x^2 - 2*x < 0}

def N : Set ℝ := {x : ℝ | x ≥ 1}

theorem M_intersect_complement_N :
  M ∩ (Set.univ \ N) = Set.Ioo 0 1 := by
  sorry

end M_intersect_complement_N_l1088_108842


namespace tangent_parallel_points_tangent_points_on_curve_unique_tangent_points_l1088_108825

-- Define the function f(x) = x^3 + x - 2
def f (x : ℝ) : ℝ := x^3 + x - 2

-- Define the derivative of f
def f' (x : ℝ) : ℝ := 3 * x^2 + 1

theorem tangent_parallel_points :
  ∀ x : ℝ, (f' x = 4) ↔ (x = 1 ∨ x = -1) :=
sorry

theorem tangent_points_on_curve :
  f 1 = 0 ∧ f (-1) = -4 :=
sorry

theorem unique_tangent_points :
  ∀ x : ℝ, f' x = 4 → (x = 1 ∨ x = -1) :=
sorry

end tangent_parallel_points_tangent_points_on_curve_unique_tangent_points_l1088_108825


namespace complement_of_union_equals_singleton_l1088_108810

-- Define the universal set U
def U : Set Nat := {1, 2, 3, 4}

-- Define set A
def A : Set Nat := {1, 2}

-- Define set B
def B : Set Nat := {2, 3}

-- Theorem statement
theorem complement_of_union_equals_singleton :
  (U \ (A ∪ B)) = {4} := by
  sorry

end complement_of_union_equals_singleton_l1088_108810


namespace firefighter_pay_theorem_l1088_108819

/-- Represents the firefighter's hourly pay in dollars -/
def hourly_pay : ℝ := 30

/-- Represents the number of work hours per week -/
def work_hours_per_week : ℝ := 48

/-- Represents the number of weeks in a month -/
def weeks_per_month : ℝ := 4

/-- Represents the monthly food expense in dollars -/
def food_expense : ℝ := 500

/-- Represents the monthly tax expense in dollars -/
def tax_expense : ℝ := 1000

/-- Represents the remaining money after expenses in dollars -/
def remaining_money : ℝ := 2340

theorem firefighter_pay_theorem :
  let monthly_pay := hourly_pay * work_hours_per_week * weeks_per_month
  let rent_expense := (1 / 3) * monthly_pay
  monthly_pay - rent_expense - food_expense - tax_expense = remaining_money :=
by sorry

end firefighter_pay_theorem_l1088_108819


namespace no_real_solutions_for_arithmetic_progression_l1088_108820

-- Define the property of being an arithmetic progression
def is_arithmetic_progression (x y z w : ℝ) : Prop :=
  y - x = z - y ∧ z - y = w - z

-- Theorem statement
theorem no_real_solutions_for_arithmetic_progression :
  ¬ ∃ (a b : ℝ), is_arithmetic_progression 15 a b (a * b) := by
  sorry

end no_real_solutions_for_arithmetic_progression_l1088_108820


namespace inequality_proof_l1088_108883

theorem inequality_proof (a b c : ℝ) 
  (h_pos : a > 0 ∧ b > 0 ∧ c > 0) 
  (h_sum : a + b + c = 1) : 
  (a^2 + b^2 + c^2 ≥ 1/3) ∧ 
  (a^2/b + b^2/c + c^2/a ≥ 1) := by
sorry

end inequality_proof_l1088_108883


namespace no_rain_probability_l1088_108839

/-- The probability of an event occurring on Monday -/
def prob_monday : ℝ := 0.7

/-- The probability of an event occurring on Tuesday -/
def prob_tuesday : ℝ := 0.5

/-- The probability of an event occurring on both Monday and Tuesday -/
def prob_both_days : ℝ := 0.4

/-- The probability of an event not occurring on either Monday or Tuesday -/
def prob_no_rain : ℝ := 1 - (prob_monday + prob_tuesday - prob_both_days)

theorem no_rain_probability :
  prob_no_rain = 0.2 := by sorry

end no_rain_probability_l1088_108839


namespace base_eight_addition_sum_l1088_108854

/-- Given distinct non-zero digits S, H, and E less than 8 that satisfy the base-8 addition
    SEH₈ + EHS₈ = SHE₈, prove that their sum in base 10 is 6. -/
theorem base_eight_addition_sum (S H E : ℕ) : 
  S ≠ 0 → H ≠ 0 → E ≠ 0 →
  S < 8 → H < 8 → E < 8 →
  S ≠ H → S ≠ E → H ≠ E →
  S * 64 + E * 8 + H + E * 64 + H * 8 + S = S * 64 + H * 8 + E →
  S + H + E = 6 := by
  sorry

end base_eight_addition_sum_l1088_108854


namespace cubic_inequality_solution_l1088_108835

theorem cubic_inequality_solution (x : ℝ) : 
  x^3 - x^2 + 11*x - 30 < 12 ↔ (x > -2 ∧ x < 3) ∨ (x > 3 ∧ x < 7) :=
sorry

end cubic_inequality_solution_l1088_108835


namespace system_equation_range_l1088_108876

theorem system_equation_range (x y m : ℝ) : 
  x + 2*y = 1 + m → 
  2*x + y = -3 → 
  x + y > 0 → 
  m > 2 := by
sorry

end system_equation_range_l1088_108876


namespace chess_team_selection_l1088_108885

theorem chess_team_selection (total_players : Nat) (quadruplets : Nat) (team_size : Nat) :
  total_players = 18 →
  quadruplets = 4 →
  team_size = 8 →
  Nat.choose (total_players - quadruplets) (team_size - quadruplets) = 1001 :=
by sorry

end chess_team_selection_l1088_108885


namespace desired_circle_properties_l1088_108837

/-- The equation of the first given circle -/
def circle1 (x y : ℝ) : Prop := x^2 + y^2 - x + y - 2 = 0

/-- The equation of the second given circle -/
def circle2 (x y : ℝ) : Prop := x^2 + y^2 = 5

/-- The equation of the line on which the center of the desired circle lies -/
def centerLine (x y : ℝ) : Prop := 3*x + 4*y - 1 = 0

/-- The equation of the desired circle -/
def desiredCircle (x y : ℝ) : Prop := x^2 + y^2 + 2*x - 2*y - 11 = 0

/-- Theorem stating that the desired circle passes through the intersection points of circle1 and circle2,
    and its center lies on the centerLine -/
theorem desired_circle_properties :
  ∀ (x y : ℝ),
    (circle1 x y ∧ circle2 x y → desiredCircle x y) ∧
    (∃ (h k : ℝ), desiredCircle h k ∧ centerLine h k) :=
by sorry

end desired_circle_properties_l1088_108837


namespace sqrt_equation_solution_l1088_108879

theorem sqrt_equation_solution :
  let x : ℝ := 49
  Real.sqrt x + Real.sqrt (x + 3) = 12 - Real.sqrt (x - 4) := by
  sorry

end sqrt_equation_solution_l1088_108879
