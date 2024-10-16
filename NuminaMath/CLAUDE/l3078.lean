import Mathlib

namespace NUMINAMATH_CALUDE_max_of_min_is_sqrt_two_l3078_307883

theorem max_of_min_is_sqrt_two (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  (⨅ z ∈ ({x, 1/y, y + 1/x} : Set ℝ), z) ≤ Real.sqrt 2 ∧
  ∃ x y : ℝ, x > 0 ∧ y > 0 ∧ (⨅ z ∈ ({x, 1/y, y + 1/x} : Set ℝ), z) = Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_max_of_min_is_sqrt_two_l3078_307883


namespace NUMINAMATH_CALUDE_min_P_over_Q_l3078_307845

theorem min_P_over_Q (x P Q : ℝ) (hx : x > 0) (hP : P > 0) (hQ : Q > 0)
  (hP_def : x^2 + 1/x^2 = P) (hQ_def : x^3 - 1/x^3 = Q) :
  ∀ y : ℝ, y > 0 → y^2 + 1/y^2 = P → y^3 - 1/y^3 = Q → P / Q ≥ 1 / Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_min_P_over_Q_l3078_307845


namespace NUMINAMATH_CALUDE_garden_fencing_needed_l3078_307836

/-- Calculates the perimeter of a rectangular garden with given length and width. -/
def garden_perimeter (length width : ℝ) : ℝ := 2 * (length + width)

/-- Proves that a rectangular garden with length 300 yards and width half its length
    requires 900 yards of fencing. -/
theorem garden_fencing_needed :
  let length : ℝ := 300
  let width : ℝ := length / 2
  garden_perimeter length width = 900 := by
sorry

#eval garden_perimeter 300 150

end NUMINAMATH_CALUDE_garden_fencing_needed_l3078_307836


namespace NUMINAMATH_CALUDE_T_equals_one_l3078_307810

theorem T_equals_one (S : ℝ) : 
  let T := Real.sin (50 * π / 180) * (S + Real.sqrt 3 * Real.tan (10 * π / 180))
  T = 1 :=
by sorry

end NUMINAMATH_CALUDE_T_equals_one_l3078_307810


namespace NUMINAMATH_CALUDE_system_solution_l3078_307856

theorem system_solution : 
  ∀ x y z : ℝ, 
  (x^2 + y^2 + 25*z^2 = 6*x*z + 8*y*z) ∧ 
  (3*x^2 + 2*y^2 + z^2 = 240) → 
  ((x = 6 ∧ y = 8 ∧ z = 2) ∨ (x = -6 ∧ y = -8 ∧ z = -2)) := by
sorry

end NUMINAMATH_CALUDE_system_solution_l3078_307856


namespace NUMINAMATH_CALUDE_systematic_sampling_fourth_student_l3078_307852

/-- Represents a systematic sampling of students. -/
structure SystematicSample where
  totalStudents : ℕ
  sampleSize : ℕ
  sampleInterval : ℕ
  firstStudent : ℕ

/-- Checks if a student number is in the sample. -/
def isInSample (s : SystematicSample) (studentNumber : ℕ) : Prop :=
  ∃ k : ℕ, studentNumber = s.firstStudent + k * s.sampleInterval ∧ 
           studentNumber ≤ s.totalStudents

theorem systematic_sampling_fourth_student 
  (s : SystematicSample)
  (h1 : s.totalStudents = 60)
  (h2 : s.sampleSize = 4)
  (h3 : s.firstStudent = 3)
  (h4 : isInSample s 33)
  (h5 : isInSample s 48) :
  isInSample s 18 := by
  sorry

#check systematic_sampling_fourth_student

end NUMINAMATH_CALUDE_systematic_sampling_fourth_student_l3078_307852


namespace NUMINAMATH_CALUDE_single_transmission_prob_triple_transmission_better_for_zero_l3078_307867

/-- Represents a binary communication channel with error probabilities α and β -/
structure BinaryChannel where
  α : ℝ
  β : ℝ
  α_pos : 0 < α
  α_lt_one : α < 1
  β_pos : 0 < β
  β_lt_one : β < 1

/-- Probability of receiving 1,0,1 when sending 1,0,1 in single transmission -/
def singleTransmissionProb (c : BinaryChannel) : ℝ :=
  (1 - c.α) * (1 - c.β)^2

/-- Probability of decoding 0 when sending 0 in single transmission -/
def singleTransmission0Prob (c : BinaryChannel) : ℝ :=
  1 - c.α

/-- Probability of decoding 0 when sending 0 in triple transmission -/
def tripleTransmission0Prob (c : BinaryChannel) : ℝ :=
  (1 - c.α)^3 + 3 * c.α * (1 - c.α)^2

theorem single_transmission_prob (c : BinaryChannel) :
  singleTransmissionProb c = (1 - c.α) * (1 - c.β)^2 := by sorry

theorem triple_transmission_better_for_zero (c : BinaryChannel) (h : c.α < 0.5) :
  singleTransmission0Prob c < tripleTransmission0Prob c := by sorry

end NUMINAMATH_CALUDE_single_transmission_prob_triple_transmission_better_for_zero_l3078_307867


namespace NUMINAMATH_CALUDE_divisibility_by_twelve_l3078_307860

theorem divisibility_by_twelve (m : Nat) : m ≤ 9 → (915 * 10 + m) % 12 = 0 ↔ m = 6 := by
  sorry

end NUMINAMATH_CALUDE_divisibility_by_twelve_l3078_307860


namespace NUMINAMATH_CALUDE_fraction_sum_equality_l3078_307818

theorem fraction_sum_equality : (1 : ℚ) / 3 + 5 / 9 - 2 / 9 = 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_equality_l3078_307818


namespace NUMINAMATH_CALUDE_function_value_at_three_l3078_307878

theorem function_value_at_three (f : ℝ → ℝ) (h : ∀ x, f (x + 1) = 2 * x + 3) : f 3 = 7 := by
  sorry

end NUMINAMATH_CALUDE_function_value_at_three_l3078_307878


namespace NUMINAMATH_CALUDE_range_of_a_l3078_307821

def A := {x : ℝ | -1 < x ∧ x < 6}
def B (a : ℝ) := {x : ℝ | x^2 - 2*x + 1 - a^2 ≥ 0}

theorem range_of_a (a : ℝ) (h_a : a > 0) :
  (∀ x : ℝ, x ∉ A → x ∈ B a) ∧ (∃ x : ℝ, x ∈ A ∧ x ∈ B a) → 
  0 < a ∧ a ≤ 2 :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l3078_307821


namespace NUMINAMATH_CALUDE_max_min_values_l3078_307887

def constraint (x y : ℝ) : Prop :=
  Real.sqrt (x - 1) + Real.sqrt (y - 4) = 2

def objective (x y : ℝ) : ℝ :=
  2 * x + y

theorem max_min_values :
  (∃ x y : ℝ, constraint x y ∧ objective x y = 14 ∧ x = 5 ∧ y = 4) ∧
  (∃ x y : ℝ, constraint x y ∧ objective x y = 26/3 ∧ x = 13/9 ∧ y = 52/9) ∧
  (∀ x y : ℝ, constraint x y → objective x y ≤ 14 ∧ objective x y ≥ 26/3) :=
by sorry

end NUMINAMATH_CALUDE_max_min_values_l3078_307887


namespace NUMINAMATH_CALUDE_gcd_15012_34765_l3078_307832

theorem gcd_15012_34765 : Nat.gcd 15012 34765 = 3 := by
  sorry

end NUMINAMATH_CALUDE_gcd_15012_34765_l3078_307832


namespace NUMINAMATH_CALUDE_triangle_angle_measure_l3078_307863

theorem triangle_angle_measure (D E F : ℝ) : 
  D = 70 → 
  E = 2 * F + 18 → 
  D + E + F = 180 → 
  F = 92 / 3 :=
by sorry

end NUMINAMATH_CALUDE_triangle_angle_measure_l3078_307863


namespace NUMINAMATH_CALUDE_intersection_distance_and_difference_l3078_307824

theorem intersection_distance_and_difference : ∃ (x₁ x₂ : ℝ),
  (4 * x₁^2 + x₁ - 1 = 5) ∧
  (4 * x₂^2 + x₂ - 1 = 5) ∧
  (x₁ ≠ x₂) ∧
  (|x₁ - x₂| = Real.sqrt 97 / 4) ∧
  (97 - 4 = 93) := by
  sorry

end NUMINAMATH_CALUDE_intersection_distance_and_difference_l3078_307824


namespace NUMINAMATH_CALUDE_cone_sphere_volume_ratio_l3078_307834

theorem cone_sphere_volume_ratio (r : ℝ) (h : r > 0) :
  let cone_volume := (1 / 3) * π * r^3
  let sphere_volume := (4 / 3) * π * r^3
  cone_volume / sphere_volume = 1 / 4 := by sorry

end NUMINAMATH_CALUDE_cone_sphere_volume_ratio_l3078_307834


namespace NUMINAMATH_CALUDE_inequality_system_solution_set_l3078_307813

theorem inequality_system_solution_set :
  let S := {x : ℝ | 2 * x - 1 ≥ x + 1 ∧ x + 8 ≤ 4 * x - 1}
  S = {x : ℝ | x ≥ 3} := by
  sorry

end NUMINAMATH_CALUDE_inequality_system_solution_set_l3078_307813


namespace NUMINAMATH_CALUDE_first_storm_rate_l3078_307877

/-- Represents the rainfall data for a week with two rainstorms -/
structure RainfallData where
  firstStormRate : ℝ
  secondStormRate : ℝ
  totalRainTime : ℝ
  totalRainfall : ℝ
  firstStormDuration : ℝ

/-- Theorem stating that given the rainfall conditions, the first storm's rate was 30 mm/hour -/
theorem first_storm_rate (data : RainfallData)
    (h1 : data.secondStormRate = 15)
    (h2 : data.totalRainTime = 45)
    (h3 : data.totalRainfall = 975)
    (h4 : data.firstStormDuration = 20) :
    data.firstStormRate = 30 := by
  sorry

#check first_storm_rate

end NUMINAMATH_CALUDE_first_storm_rate_l3078_307877


namespace NUMINAMATH_CALUDE_mixed_groups_count_l3078_307820

theorem mixed_groups_count (total_children : ℕ) (total_groups : ℕ) (group_size : ℕ)
  (boy_boy_photos : ℕ) (girl_girl_photos : ℕ) :
  total_children = 300 →
  total_groups = 100 →
  group_size = 3 →
  total_children = total_groups * group_size →
  boy_boy_photos = 100 →
  girl_girl_photos = 56 →
  ∃ (mixed_groups : ℕ),
    mixed_groups = 72 ∧
    mixed_groups * 2 + boy_boy_photos + girl_girl_photos = total_groups * group_size :=
by sorry

end NUMINAMATH_CALUDE_mixed_groups_count_l3078_307820


namespace NUMINAMATH_CALUDE_river_width_calculation_l3078_307876

/-- The width of a river given the length of an existing bridge and the additional length needed to cross it. -/
def river_width (existing_bridge_length additional_length : ℕ) : ℕ :=
  existing_bridge_length + additional_length

/-- Theorem: The width of the river is equal to the sum of the existing bridge length and the additional length needed. -/
theorem river_width_calculation (existing_bridge_length additional_length : ℕ) :
  river_width existing_bridge_length additional_length = existing_bridge_length + additional_length :=
by
  sorry

/-- The width of the specific river in the problem. -/
def specific_river_width : ℕ := river_width 295 192

#eval specific_river_width

end NUMINAMATH_CALUDE_river_width_calculation_l3078_307876


namespace NUMINAMATH_CALUDE_marla_nightly_cost_l3078_307801

/-- Represents the exchange rates and Marla's scavenging situation in the post-apocalyptic wasteland -/
structure WastelandEconomy where
  lizard_to_caps : ℕ → ℕ
  lizards_to_water : ℕ → ℕ
  horse_to_water : ℕ
  daily_scavenge : ℕ
  days_to_horse : ℕ

/-- Calculates the number of bottle caps Marla needs to pay per night for food and shelter -/
def nightly_cost (we : WastelandEconomy) : ℕ :=
  -- The actual calculation goes here, but we'll use sorry to skip the proof
  sorry

/-- Theorem stating that in the given wasteland economy, Marla needs to pay 4 bottle caps per night -/
theorem marla_nightly_cost :
  let we : WastelandEconomy := {
    lizard_to_caps := λ n => 8 * n,
    lizards_to_water := λ n => (5 * n) / 3,
    horse_to_water := 80,
    daily_scavenge := 20,
    days_to_horse := 24
  }
  nightly_cost we = 4 := by
  sorry

end NUMINAMATH_CALUDE_marla_nightly_cost_l3078_307801


namespace NUMINAMATH_CALUDE_tank_water_supply_l3078_307839

theorem tank_water_supply (C V : ℝ) 
  (h1 : C = 75 * (V + 10))
  (h2 : C = 60 * (V + 20)) :
  C / V = 100 := by
  sorry

end NUMINAMATH_CALUDE_tank_water_supply_l3078_307839


namespace NUMINAMATH_CALUDE_pie_eating_contest_l3078_307807

theorem pie_eating_contest (first_student second_student : ℚ) 
  (h1 : first_student = 7/8)
  (h2 : second_student = 5/6) : 
  first_student - second_student = 1/24 := by
  sorry

end NUMINAMATH_CALUDE_pie_eating_contest_l3078_307807


namespace NUMINAMATH_CALUDE_factor_implies_c_value_l3078_307874

theorem factor_implies_c_value (c : ℝ) : 
  (∀ x : ℝ, (x - 5) * ((c/100) * x^2 + (23/100) * x - (c/20) + 11/20) = 
             c * x^3 + 23 * x^2 - 5 * c * x + 55) → 
  c = -6.3 := by sorry

end NUMINAMATH_CALUDE_factor_implies_c_value_l3078_307874


namespace NUMINAMATH_CALUDE_quadratic_two_roots_l3078_307828

theorem quadratic_two_roots (c : ℝ) (h : c < 4) :
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ x₁^2 - 4*x₁ + c = 0 ∧ x₂^2 - 4*x₂ + c = 0 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_two_roots_l3078_307828


namespace NUMINAMATH_CALUDE_train_crossing_time_l3078_307847

/-- Proves that a train with given length and speed takes the calculated time to cross a pole -/
theorem train_crossing_time (train_length : Real) (train_speed_kmh : Real) :
  train_length = 50 ∧ train_speed_kmh = 60 →
  (train_length / (train_speed_kmh * 1000 / 3600)) = 3 := by
  sorry

end NUMINAMATH_CALUDE_train_crossing_time_l3078_307847


namespace NUMINAMATH_CALUDE_problem_solution_l3078_307825

theorem problem_solution : 
  (27 / 8) ^ (-1/3 : ℝ) + Real.log 3 / Real.log 2 * Real.log 4 / Real.log 3 + 
  Real.log 2 / Real.log 10 + Real.log 50 / Real.log 10 = 14/3 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l3078_307825


namespace NUMINAMATH_CALUDE_arithmetic_and_geometric_sequences_l3078_307805

-- Define the arithmetic sequence a_n
def a (n : ℕ) : ℝ := 2 * n - 12

-- Define the geometric sequence b_n
def b (n : ℕ) : ℝ := -8

-- Define the sum of the first n terms of b_n
def S (n : ℕ) : ℝ := -8 * n

theorem arithmetic_and_geometric_sequences :
  (a 3 = -6) ∧ 
  (a 6 = 0) ∧ 
  (b 1 = -8) ∧ 
  (b 2 = a 1 + a 2 + a 3) ∧
  (∀ n : ℕ, a (n + 1) - a n = a 2 - a 1) ∧  -- arithmetic sequence property
  (∀ n : ℕ, b (n + 1) / b n = b 2 / b 1) ∧  -- geometric sequence property
  (∀ n : ℕ, S n = (1 - (b 2 / b 1)^n) / (1 - b 2 / b 1) * b 1) -- sum formula for geometric sequence
  :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_and_geometric_sequences_l3078_307805


namespace NUMINAMATH_CALUDE_sandy_saturday_hours_l3078_307897

/-- Sandy's hourly rate in dollars -/
def hourly_rate : ℚ := 15

/-- Hours Sandy worked on Friday -/
def friday_hours : ℚ := 10

/-- Hours Sandy worked on Sunday -/
def sunday_hours : ℚ := 14

/-- Total earnings for Friday, Saturday, and Sunday in dollars -/
def total_earnings : ℚ := 450

/-- Calculates the number of hours Sandy worked on Saturday -/
def saturday_hours : ℚ :=
  (total_earnings - hourly_rate * (friday_hours + sunday_hours)) / hourly_rate

theorem sandy_saturday_hours :
  saturday_hours = 6 := by sorry

end NUMINAMATH_CALUDE_sandy_saturday_hours_l3078_307897


namespace NUMINAMATH_CALUDE_question_selection_ways_eq_13838400_l3078_307849

/-- The number of ways to select questions from a question paper with three parts -/
def questionSelectionWays : ℕ :=
  let partA := Nat.choose 12 8
  let partB := Nat.choose 10 5
  let partC := Nat.choose 8 3
  partA * partB * partC

/-- Theorem stating the correct number of ways to select questions -/
theorem question_selection_ways_eq_13838400 : questionSelectionWays = 13838400 := by
  sorry

end NUMINAMATH_CALUDE_question_selection_ways_eq_13838400_l3078_307849


namespace NUMINAMATH_CALUDE_mystic_aquarium_fish_duration_l3078_307873

/-- The number of weeks that a given number of fish buckets will last at the Mystic Aquarium -/
def weeks_of_fish (total_buckets : ℕ) : ℕ :=
  let sharks_daily := 4
  let dolphins_daily := sharks_daily / 2
  let others_daily := sharks_daily * 5
  let daily_consumption := sharks_daily + dolphins_daily + others_daily
  let weekly_consumption := daily_consumption * 7
  total_buckets / weekly_consumption

/-- Theorem stating that 546 buckets of fish will last for 3 weeks -/
theorem mystic_aquarium_fish_duration : weeks_of_fish 546 = 3 := by
  sorry

end NUMINAMATH_CALUDE_mystic_aquarium_fish_duration_l3078_307873


namespace NUMINAMATH_CALUDE_quadratic_roots_equivalence_l3078_307858

/-- A quadratic function f(x) = ax^2 + bx + c where a > 0 -/
def QuadraticFunction (a b c : ℝ) (h : a > 0) : ℝ → ℝ := fun x ↦ a * x^2 + b * x + c

theorem quadratic_roots_equivalence (a b c : ℝ) (h : a > 0) :
  let f := QuadraticFunction a b c h
  (f (f (-b / (2 * a))) < 0) ↔
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ f x₁ = 0 ∧ f x₂ = 0) ∧
  (∃ y₁ y₂ : ℝ, y₁ ≠ y₂ ∧ f (f y₁) = 0 ∧ f (f y₂) = 0) :=
sorry

end NUMINAMATH_CALUDE_quadratic_roots_equivalence_l3078_307858


namespace NUMINAMATH_CALUDE_max_value_of_inverse_sum_l3078_307890

open Real

-- Define the quadratic equation and its roots
def quadratic (t q : ℝ) (x : ℝ) : ℝ := x^2 - t*x + q

-- Define the condition for the roots
def roots_condition (α β : ℝ) : Prop :=
  α + β = α^2 + β^2 ∧ α + β = α^3 + β^3 ∧ α + β = α^4 + β^4 ∧ α + β = α^5 + β^5

-- Theorem statement
theorem max_value_of_inverse_sum (t q α β : ℝ) :
  (∀ x, quadratic t q x = 0 ↔ x = α ∨ x = β) →
  roots_condition α β →
  (∀ γ δ : ℝ, roots_condition γ δ → 1/γ^6 + 1/δ^6 ≤ 2) :=
sorry

end NUMINAMATH_CALUDE_max_value_of_inverse_sum_l3078_307890


namespace NUMINAMATH_CALUDE_rearrangement_theorem_l3078_307875

/-- Represents the number of people in the line -/
def n : ℕ := 8

/-- Represents the number of people moved to the front -/
def k : ℕ := 3

/-- Calculates the number of ways to rearrange people in a line
    under the given conditions -/
def rearrangement_count (n k : ℕ) : ℕ :=
  (n - k - 1) * (n - k) * (n - k + 1)

/-- The theorem stating that the number of rearrangements is 210 -/
theorem rearrangement_theorem :
  rearrangement_count n k = 210 := by sorry

end NUMINAMATH_CALUDE_rearrangement_theorem_l3078_307875


namespace NUMINAMATH_CALUDE_range_of_a_l3078_307869

open Real

theorem range_of_a (a : ℝ) : 
  let P := ∀ x : ℝ, x^2 + 2*a*x + 4 > 0
  let Q := ∀ x y : ℝ, x < y → -(5-2*a)^x > -(5-2*a)^y
  (P ∧ ¬Q) ∨ (¬P ∧ Q) → a ≤ -2 :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l3078_307869


namespace NUMINAMATH_CALUDE_hendecagon_diagonals_l3078_307800

/-- The number of diagonals in a polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- A hendecagon is an 11-sided polygon -/
def hendecagon_sides : ℕ := 11

/-- The number of diagonals in a hendecagon is 44 -/
theorem hendecagon_diagonals : num_diagonals hendecagon_sides = 44 := by
  sorry

end NUMINAMATH_CALUDE_hendecagon_diagonals_l3078_307800


namespace NUMINAMATH_CALUDE_smallest_n_congruence_l3078_307879

theorem smallest_n_congruence : ∃ (n : ℕ), n > 0 ∧ (3 * n) % 30 = 2412 % 30 ∧ ∀ (m : ℕ), m > 0 → (3 * m) % 30 = 2412 % 30 → n ≤ m :=
by sorry

end NUMINAMATH_CALUDE_smallest_n_congruence_l3078_307879


namespace NUMINAMATH_CALUDE_projection_a_onto_b_is_sqrt5_l3078_307871

/-- The projection of vector a onto the direction of vector b is √5 -/
theorem projection_a_onto_b_is_sqrt5 (a b : ℝ × ℝ) : 
  a = (1, 3) → a + b = (-1, 7) → 
  (a.1 * b.1 + a.2 * b.2) / Real.sqrt (b.1^2 + b.2^2) = Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_projection_a_onto_b_is_sqrt5_l3078_307871


namespace NUMINAMATH_CALUDE_voronovich_inequality_l3078_307814

theorem voronovich_inequality {a b c : ℝ} (ha : 0 < a) (hab : a < b) (hbc : b < c) :
  a^20 * b^12 + b^20 * c^12 + c^20 * a^12 < b^20 * a^12 + a^20 * c^12 + c^20 * b^12 :=
by sorry

end NUMINAMATH_CALUDE_voronovich_inequality_l3078_307814


namespace NUMINAMATH_CALUDE_quadratic_minimum_value_l3078_307882

def f (x : ℝ) : ℝ := 2 * x^2 + 6 * x + 5

theorem quadratic_minimum_value :
  (f 1 = 13) →
  ∃ (x_min : ℝ), ∀ (x : ℝ), f x_min ≤ f x ∧ f x_min = 0.5 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_minimum_value_l3078_307882


namespace NUMINAMATH_CALUDE_odd_expressions_l3078_307823

theorem odd_expressions (m n p : ℕ) 
  (hm : m % 2 = 1) 
  (hn : n % 2 = 1) 
  (hp : p % 2 = 0) 
  (hm_pos : 0 < m) 
  (hn_pos : 0 < n) 
  (hp_pos : 0 < p) : 
  ((2 * m * n + 5)^2) % 2 = 1 ∧ (5 * m * n + p) % 2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_odd_expressions_l3078_307823


namespace NUMINAMATH_CALUDE_reciprocal_of_negative_three_l3078_307840

theorem reciprocal_of_negative_three :
  ∀ x : ℚ, x * (-3) = 1 → x = -1/3 := by
  sorry

end NUMINAMATH_CALUDE_reciprocal_of_negative_three_l3078_307840


namespace NUMINAMATH_CALUDE_resistance_value_l3078_307831

/-- Given two identical resistors connected in series to a DC voltage source,
    prove that the resistance of each resistor is 2 Ω based on voltmeter and ammeter readings. -/
theorem resistance_value (R U Uv IA : ℝ) : 
  Uv = 10 →  -- Voltmeter reading
  IA = 10 →  -- Ammeter reading
  U = 2 * Uv →  -- Total voltage
  U = R * IA →  -- Ohm's law for the circuit with ammeter
  R = 2 := by
  sorry

end NUMINAMATH_CALUDE_resistance_value_l3078_307831


namespace NUMINAMATH_CALUDE_michaels_brother_money_l3078_307822

/-- Given that Michael has $42 and his brother has $17, Michael gives half his money to his brother,
    and his brother then buys $3 worth of candy, prove that his brother ends up with $35. -/
theorem michaels_brother_money (michael_initial : ℕ) (brother_initial : ℕ) 
    (candy_cost : ℕ) (h1 : michael_initial = 42) (h2 : brother_initial = 17) 
    (h3 : candy_cost = 3) : 
    brother_initial + michael_initial / 2 - candy_cost = 35 := by
  sorry

end NUMINAMATH_CALUDE_michaels_brother_money_l3078_307822


namespace NUMINAMATH_CALUDE_meter_to_skips_l3078_307898

/-- Given the relationships between hops, skips, jumps, and meters, 
    prove that one meter equals (90bde)/(56acf) skips -/
theorem meter_to_skips 
  (hop_skip : (2 : ℝ) * a * 1 = (3 : ℝ) * b * 1)
  (jump_hop : (4 : ℝ) * c * 1 = (5 : ℝ) * d * 1)
  (jump_meter : (6 : ℝ) * e * 1 = (7 : ℝ) * f * 1)
  (a b c d e f : ℝ) 
  (h_nonzero : a ≠ 0 ∧ c ≠ 0 ∧ f ≠ 0) : 
  (1 : ℝ) = (90 * b * d * e) / (56 * a * c * f) * 1 :=
sorry

end NUMINAMATH_CALUDE_meter_to_skips_l3078_307898


namespace NUMINAMATH_CALUDE_quadratic_real_roots_m_range_l3078_307811

/-- 
Given a quadratic equation x^2 - 2x + m = 0, if it has real roots, 
then m ≤ 1.
-/
theorem quadratic_real_roots_m_range (m : ℝ) : 
  (∃ x : ℝ, x^2 - 2*x + m = 0) → m ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_real_roots_m_range_l3078_307811


namespace NUMINAMATH_CALUDE_shaded_fraction_of_square_l3078_307881

/-- Given a square with side length x, where P is at a corner and Q is at the midpoint of an adjacent side,
    the fraction of the square's interior that is shaded is 3/4. -/
theorem shaded_fraction_of_square (x : ℝ) (h : x > 0) : 
  let square_area := x^2
  let triangle_area := (1/2) * x * (x/2)
  let shaded_area := square_area - triangle_area
  shaded_area / square_area = 3/4 := by
sorry

end NUMINAMATH_CALUDE_shaded_fraction_of_square_l3078_307881


namespace NUMINAMATH_CALUDE_alipay_growth_rate_l3078_307848

theorem alipay_growth_rate (initial : ℕ) (final : ℕ) (years : ℕ) (rate : ℝ) : 
  initial = 45000 →
  final = 64800 →
  years = 2 →
  (initial : ℝ) * (1 + rate) ^ years = final →
  rate = 0.2 := by
  sorry

end NUMINAMATH_CALUDE_alipay_growth_rate_l3078_307848


namespace NUMINAMATH_CALUDE_quadratic_function_minimum_l3078_307862

/-- A quadratic function that takes specific values for consecutive integers -/
def QuadraticFunction (f : ℝ → ℝ) : Prop :=
  ∃ n : ℕ, f n = 13 ∧ f (n + 1) = 13 ∧ f (n + 2) = 35

/-- The theorem stating the minimum value of the quadratic function -/
theorem quadratic_function_minimum (f : ℝ → ℝ) (h : QuadraticFunction f) :
  ∃ x : ℝ, ∀ y : ℝ, f x ≤ f y ∧ f x = 41 / 4 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_function_minimum_l3078_307862


namespace NUMINAMATH_CALUDE_events_mutually_exclusive_not_complementary_l3078_307838

/-- A bag containing red and white balls -/
structure Bag where
  red : Nat
  white : Nat

/-- The event of drawing exactly one white ball -/
def exactlyOneWhite (b : Bag) : Prop := sorry

/-- The event of drawing exactly two white balls -/
def exactlyTwoWhite (b : Bag) : Prop := sorry

/-- Two events are mutually exclusive -/
def mutuallyExclusive (e1 e2 : Prop) : Prop := ¬(e1 ∧ e2)

/-- Two events are complementary -/
def complementary (e1 e2 : Prop) : Prop := (e1 ∨ e2) ∧ mutuallyExclusive e1 e2

/-- The main theorem -/
theorem events_mutually_exclusive_not_complementary (b : Bag) 
  (h : b.red = 2 ∧ b.white = 2) : 
  mutuallyExclusive (exactlyOneWhite b) (exactlyTwoWhite b) ∧ 
  ¬complementary (exactlyOneWhite b) (exactlyTwoWhite b) := by
  sorry

end NUMINAMATH_CALUDE_events_mutually_exclusive_not_complementary_l3078_307838


namespace NUMINAMATH_CALUDE_wool_price_calculation_l3078_307865

theorem wool_price_calculation (num_sheep : ℕ) (shearing_cost : ℕ) (wool_per_sheep : ℕ) (profit : ℕ) : 
  num_sheep = 200 → 
  shearing_cost = 2000 → 
  wool_per_sheep = 10 → 
  profit = 38000 → 
  (profit + shearing_cost) / (num_sheep * wool_per_sheep) = 20 := by
sorry

end NUMINAMATH_CALUDE_wool_price_calculation_l3078_307865


namespace NUMINAMATH_CALUDE_truth_values_of_p_and_q_l3078_307880

theorem truth_values_of_p_and_q (hp_and_q : ¬(p ∧ q)) (hnot_p_or_q : ¬p ∨ q) :
  ¬p ∧ (q ∨ ¬q) :=
by sorry

end NUMINAMATH_CALUDE_truth_values_of_p_and_q_l3078_307880


namespace NUMINAMATH_CALUDE_sum_remainder_eleven_l3078_307895

theorem sum_remainder_eleven (n : ℤ) : ((11 - n) + (n + 5)) % 11 = 5 := by
  sorry

end NUMINAMATH_CALUDE_sum_remainder_eleven_l3078_307895


namespace NUMINAMATH_CALUDE_quadratic_function_k_value_l3078_307804

theorem quadratic_function_k_value (a b c : ℤ) (k : ℤ) : 
  let f : ℝ → ℝ := λ x => (a * x^2 + b * x + c : ℝ)
  (f 1 = 0) →
  (60 < f 9 ∧ f 9 < 70) →
  (90 < f 10 ∧ f 10 < 100) →
  (10000 * k < f 100 ∧ f 100 < 10000 * (k + 1)) →
  k = 2 := by
sorry

end NUMINAMATH_CALUDE_quadratic_function_k_value_l3078_307804


namespace NUMINAMATH_CALUDE_tangent_line_to_exponential_curve_l3078_307859

/-- The line y = kx is tangent to the curve y = 2e^x if and only if k = 2e -/
theorem tangent_line_to_exponential_curve (k : ℝ) :
  (∃ x₀ : ℝ, k * x₀ = 2 * Real.exp x₀ ∧
             k = 2 * Real.exp x₀) ↔ k = 2 * Real.exp 1 :=
by sorry

end NUMINAMATH_CALUDE_tangent_line_to_exponential_curve_l3078_307859


namespace NUMINAMATH_CALUDE_lemonade_stand_revenue_calculation_l3078_307802

/-- Calculates the gross revenue of a lemonade stand given total profit, babysitting income, and lemonade stand expenses. -/
def lemonade_stand_revenue (total_profit babysitting_income lemonade_expenses : ℤ) : ℤ :=
  (total_profit - babysitting_income) + lemonade_expenses

theorem lemonade_stand_revenue_calculation :
  lemonade_stand_revenue 44 31 34 = 47 := by
  sorry

end NUMINAMATH_CALUDE_lemonade_stand_revenue_calculation_l3078_307802


namespace NUMINAMATH_CALUDE_factor_expression_l3078_307842

theorem factor_expression (x : ℝ) : 3 * x^2 - 75 = 3 * (x + 5) * (x - 5) := by
  sorry

end NUMINAMATH_CALUDE_factor_expression_l3078_307842


namespace NUMINAMATH_CALUDE_line_equation_conversion_l3078_307850

/-- Given a line in the form (3, 7) · ((x, y) - (-2, 4)) = 0, 
    prove that its slope-intercept form y = mx + b 
    has m = -3/7 and b = 22/7 -/
theorem line_equation_conversion :
  ∀ (x y : ℝ), 
  (3 : ℝ) * (x + 2) + (7 : ℝ) * (y - 4) = 0 →
  ∃ (m b : ℝ), y = m * x + b ∧ m = -3/7 ∧ b = 22/7 := by
  sorry

end NUMINAMATH_CALUDE_line_equation_conversion_l3078_307850


namespace NUMINAMATH_CALUDE_inequality_proof_l3078_307872

theorem inequality_proof (f : ℝ → ℝ) (a m n : ℝ) :
  (∀ x, f x = |x - a| + 1) →
  (Set.Icc 0 2 = {x | f x ≤ 2}) →
  m > 0 →
  n > 0 →
  1/m + 1/n = a →
  m + 2*n ≥ 3 + 2*Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_l3078_307872


namespace NUMINAMATH_CALUDE_square_division_and_triangle_area_l3078_307829

/-- The area of the remaining part after cutting off squares from a unit square -/
def S (n : ℕ) : ℚ :=
  (n + 1 : ℚ) / (2 * n)

/-- The area of triangle ABP formed by the intersection of y = (1/2)x and y = 1/(2x) -/
def triangle_area (n : ℕ) : ℚ :=
  1 / 2 + (1 / 2) * (1 / 2)

theorem square_division_and_triangle_area (n : ℕ) (h : n ≥ 2) :
  S n = (n + 1 : ℚ) / (2 * n) ∧ 
  triangle_area n = 1 ∧
  ∀ ε > 0, ∃ N, ∀ n ≥ N, |triangle_area n - 1| < ε :=
sorry

end NUMINAMATH_CALUDE_square_division_and_triangle_area_l3078_307829


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_ratio_l3078_307885

theorem quadratic_equation_roots_ratio (k : ℝ) : 
  (∃ x y : ℝ, x ≠ 0 ∧ y ≠ 0 ∧ 
   x^2 + 10*x + k = 0 ∧ 
   y^2 + 10*y + k = 0 ∧ 
   x / y = 3 / 2) → 
  k = 24 := by
sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_ratio_l3078_307885


namespace NUMINAMATH_CALUDE_sector_area_l3078_307835

theorem sector_area (α l : Real) (h1 : α = π / 6) (h2 : l = π / 3) :
  let r := l / α
  let s := (1 / 2) * l * r
  s = π / 3 := by
  sorry

end NUMINAMATH_CALUDE_sector_area_l3078_307835


namespace NUMINAMATH_CALUDE_smallest_with_24_factors_div_by_18_and_30_is_360_l3078_307815

/-- The smallest integer with 24 positive factors that is divisible by both 18 and 30 -/
def smallest_with_24_factors_div_by_18_and_30 : ℕ := 360

/-- Proposition: The smallest integer with 24 positive factors that is divisible by both 18 and 30 is 360 -/
theorem smallest_with_24_factors_div_by_18_and_30_is_360 :
  ∀ y : ℕ, 
    (Finset.card (Nat.divisors y) = 24) → 
    (18 ∣ y) → 
    (30 ∣ y) → 
    y ≥ smallest_with_24_factors_div_by_18_and_30 :=
sorry

end NUMINAMATH_CALUDE_smallest_with_24_factors_div_by_18_and_30_is_360_l3078_307815


namespace NUMINAMATH_CALUDE_middle_digit_zero_l3078_307886

/-- Represents a three-digit number in base 8 -/
structure Base8Number where
  hundreds : Nat
  tens : Nat
  ones : Nat
  valid : hundreds < 8 ∧ tens < 8 ∧ ones < 8

/-- Represents a three-digit number in base 10 -/
structure Base10Number where
  hundreds : Nat
  tens : Nat
  ones : Nat
  valid : hundreds < 10 ∧ tens < 10 ∧ ones < 10

/-- Converts a Base8Number to its decimal representation -/
def toDecimal (n : Base8Number) : Nat :=
  512 * n.hundreds + 64 * n.tens + 8 * n.ones

/-- Converts a Base10Number to its decimal representation -/
def fromBase10 (n : Base10Number) : Nat :=
  100 * n.hundreds + 10 * n.tens + n.ones

/-- Checks if the digits of a Base10Number are a right rotation of a Base8Number -/
def isRightRotation (n8 : Base8Number) (n10 : Base10Number) : Prop :=
  n10.hundreds = n8.tens ∧ n10.tens = n8.ones ∧ n10.ones = n8.hundreds

theorem middle_digit_zero (n8 : Base8Number) (n10 : Base10Number) 
  (h : toDecimal n8 = fromBase10 n10) 
  (rot : isRightRotation n8 n10) : 
  n10.tens = 0 := by
  sorry

end NUMINAMATH_CALUDE_middle_digit_zero_l3078_307886


namespace NUMINAMATH_CALUDE_pure_imaginary_product_l3078_307837

theorem pure_imaginary_product (a b c d : ℝ) :
  (∃ k : ℝ, (a + b * Complex.I) * (c + d * Complex.I) = k * Complex.I) →
  (a * c - b * d = 0 ∧ a * d + b * c ≠ 0) :=
by sorry

end NUMINAMATH_CALUDE_pure_imaginary_product_l3078_307837


namespace NUMINAMATH_CALUDE_norris_remaining_money_l3078_307870

/-- Calculates the remaining money for Norris after savings and spending --/
def remaining_money (september_savings october_savings november_savings spent : ℕ) : ℕ :=
  september_savings + october_savings + november_savings - spent

/-- Theorem stating that Norris has $10 left after his savings and spending --/
theorem norris_remaining_money :
  remaining_money 29 25 31 75 = 10 := by
  sorry

end NUMINAMATH_CALUDE_norris_remaining_money_l3078_307870


namespace NUMINAMATH_CALUDE_max_sum_of_factors_l3078_307888

theorem max_sum_of_factors (A B C : ℕ+) : 
  A ≠ B ∧ B ≠ C ∧ A ≠ C →
  A * B * C = 2023 →
  A + B + C ≤ 297 :=
by sorry

end NUMINAMATH_CALUDE_max_sum_of_factors_l3078_307888


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l3078_307894

theorem geometric_sequence_sum (a : ℕ → ℚ) (q : ℚ) :
  (∀ n : ℕ, a (n + 1) = a n * q) →  -- a is a geometric sequence with common ratio q
  a 1 + a 2 + a 3 + a 4 = 15/8 →    -- sum of first four terms
  a 2 * a 3 = -9/8 →                -- product of second and third terms
  1 / a 1 + 1 / a 2 + 1 / a 3 + 1 / a 4 = -5/3 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l3078_307894


namespace NUMINAMATH_CALUDE_total_legs_in_room_l3078_307827

/-- Represents the count of furniture items with their respective leg counts -/
structure FurnitureCount where
  four_leg_tables : ℕ
  four_leg_sofas : ℕ
  four_leg_chairs : ℕ
  three_leg_tables : ℕ
  one_leg_tables : ℕ
  two_leg_rocking_chairs : ℕ

/-- Calculates the total number of legs in the room -/
def total_legs (fc : FurnitureCount) : ℕ :=
  4 * fc.four_leg_tables +
  4 * fc.four_leg_sofas +
  4 * fc.four_leg_chairs +
  3 * fc.three_leg_tables +
  1 * fc.one_leg_tables +
  2 * fc.two_leg_rocking_chairs

/-- The given furniture configuration in the room -/
def room_furniture : FurnitureCount :=
  { four_leg_tables := 4
  , four_leg_sofas := 1
  , four_leg_chairs := 2
  , three_leg_tables := 3
  , one_leg_tables := 1
  , two_leg_rocking_chairs := 1 }

/-- Theorem stating that the total number of legs in the room is 40 -/
theorem total_legs_in_room : total_legs room_furniture = 40 := by
  sorry

end NUMINAMATH_CALUDE_total_legs_in_room_l3078_307827


namespace NUMINAMATH_CALUDE_cricket_team_size_is_eleven_l3078_307806

/-- Represents the number of members in a cricket team satisfying specific age conditions. -/
def cricket_team_size : ℕ :=
  let captain_age : ℕ := 28
  let wicket_keeper_age : ℕ := captain_age + 3
  let team_average_age : ℕ := 25
  let n : ℕ := 11  -- The number we want to prove

  have h1 : n * team_average_age = (n - 2) * (team_average_age - 1) + captain_age + wicket_keeper_age :=
    by sorry

  n

theorem cricket_team_size_is_eleven : cricket_team_size = 11 := by
  unfold cricket_team_size
  sorry

end NUMINAMATH_CALUDE_cricket_team_size_is_eleven_l3078_307806


namespace NUMINAMATH_CALUDE_smallest_odd_integer_triangle_perimeter_l3078_307864

/-- A function that generates the nth odd integer -/
def nthOddInteger (n : ℕ) : ℕ := 2 * n + 1

/-- A predicate that checks if three numbers form a valid triangle -/
def isValidTriangle (a b c : ℕ) : Prop := a + b > c ∧ a + c > b ∧ b + c > a

/-- The theorem stating the smallest possible perimeter of a triangle with consecutive odd integer sides -/
theorem smallest_odd_integer_triangle_perimeter :
  ∃ (n : ℕ), 
    isValidTriangle (nthOddInteger n) (nthOddInteger (n + 1)) (nthOddInteger (n + 2)) ∧
    (∀ (m : ℕ), m < n → ¬isValidTriangle (nthOddInteger m) (nthOddInteger (m + 1)) (nthOddInteger (m + 2))) ∧
    nthOddInteger n + nthOddInteger (n + 1) + nthOddInteger (n + 2) = 15 :=
sorry

end NUMINAMATH_CALUDE_smallest_odd_integer_triangle_perimeter_l3078_307864


namespace NUMINAMATH_CALUDE_events_A_D_independent_l3078_307826

-- Define the sample space
def Ω : Type := Fin 6 × Fin 6

-- Define the probability measure
def P : Set Ω → ℝ := sorry

-- Define events A and D
def A : Set Ω := {ω | Odd ω.1}
def D : Set Ω := {ω | ω.1 + ω.2 = 7}

-- State the theorem
theorem events_A_D_independent : 
  P (A ∩ D) = P A * P D := by sorry

end NUMINAMATH_CALUDE_events_A_D_independent_l3078_307826


namespace NUMINAMATH_CALUDE_triangle_problem_l3078_307833

-- Define the triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real

-- Define the theorem
theorem triangle_problem (t : Triangle) 
  (h1 : t.a + t.b = 5)
  (h2 : t.c = Real.sqrt 7)
  (h3 : Real.cos (2 * t.C) + 2 * Real.cos (t.A + t.B) = -1) :
  t.C = π / 3 ∧ 
  (1/2 : Real) * t.a * t.b * Real.sin t.C = (3 * Real.sqrt 3) / 2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_problem_l3078_307833


namespace NUMINAMATH_CALUDE_expression_simplification_l3078_307846

theorem expression_simplification :
  3 + Real.sqrt 3 + (1 / (3 + Real.sqrt 3)) + (1 / (Real.sqrt 3 - 3)) = 3 + (2 * Real.sqrt 3) / 3 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l3078_307846


namespace NUMINAMATH_CALUDE_coin_problem_l3078_307843

theorem coin_problem :
  let total_coins : ℕ := 56
  let total_value : ℕ := 440
  let coins_of_one_type : ℕ := 24
  let x : ℕ := total_coins - coins_of_one_type  -- number of 10-peso coins
  let y : ℕ := coins_of_one_type  -- number of 5-peso coins
  (x + y = total_coins) ∧ (10 * x + 5 * y = total_value) → y = 24 :=
by
  sorry

end NUMINAMATH_CALUDE_coin_problem_l3078_307843


namespace NUMINAMATH_CALUDE_impossible_equal_sums_l3078_307854

/-- A configuration of numbers on a triangle with medians -/
structure TriangleConfig where
  vertices : Fin 3 → ℕ
  midpoints : Fin 3 → ℕ
  center : ℕ

/-- The sum of numbers on a side of the triangle -/
def side_sum (config : TriangleConfig) (i : Fin 3) : ℕ :=
  config.vertices i + config.midpoints i + config.vertices ((i + 1) % 3)

/-- The sum of numbers on a median of the triangle -/
def median_sum (config : TriangleConfig) (i : Fin 3) : ℕ :=
  config.vertices i + config.midpoints ((i + 1) % 3) + config.center

/-- Predicate to check if a configuration is valid -/
def is_valid_config (config : TriangleConfig) : Prop :=
  (∀ i : Fin 3, config.vertices i ≤ 7) ∧
  (∀ i : Fin 3, config.midpoints i ≤ 7) ∧
  (config.center ≤ 7) ∧
  (config.vertices 0 + config.vertices 1 + config.vertices 2 +
   config.midpoints 0 + config.midpoints 1 + config.midpoints 2 +
   config.center = 28)

/-- Predicate to check if a configuration has equal sums -/
def has_equal_sums (config : TriangleConfig) : Prop :=
  ∃ x : ℕ, (∀ i : Fin 3, side_sum config i = x) ∧
            (∀ i : Fin 3, median_sum config i = x)

theorem impossible_equal_sums : ¬∃ config : TriangleConfig, 
  is_valid_config config ∧ has_equal_sums config := by
  sorry

end NUMINAMATH_CALUDE_impossible_equal_sums_l3078_307854


namespace NUMINAMATH_CALUDE_gwen_total_books_l3078_307853

/-- The number of books on each shelf -/
def books_per_shelf : ℕ := 4

/-- The number of shelves for mystery books -/
def mystery_shelves : ℕ := 5

/-- The number of shelves for picture books -/
def picture_shelves : ℕ := 3

/-- The total number of books Gwen has -/
def total_books : ℕ := books_per_shelf * (mystery_shelves + picture_shelves)

theorem gwen_total_books : total_books = 32 := by sorry

end NUMINAMATH_CALUDE_gwen_total_books_l3078_307853


namespace NUMINAMATH_CALUDE_valid_outfit_choices_l3078_307816

/-- The number of shirts available -/
def num_shirts : ℕ := 8

/-- The number of pants available -/
def num_pants : ℕ := 5

/-- The number of hats available -/
def num_hats : ℕ := 7

/-- The number of colors available for each item -/
def num_colors : ℕ := 5

/-- The total number of possible outfits -/
def total_outfits : ℕ := num_shirts * num_pants * num_hats

/-- The number of outfits where pants and hat are the same color -/
def matching_pants_hat_outfits : ℕ := num_colors * num_shirts

/-- The number of valid outfit choices -/
def valid_outfits : ℕ := total_outfits - matching_pants_hat_outfits

theorem valid_outfit_choices :
  valid_outfits = 240 :=
by sorry

end NUMINAMATH_CALUDE_valid_outfit_choices_l3078_307816


namespace NUMINAMATH_CALUDE_ellipse_C_equation_constant_ratio_l3078_307889

/-- An ellipse C with semi-major axis a and semi-minor axis b -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_pos : 0 < b ∧ b < a

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- A line with slope k passing through point (x₀, y₀) -/
structure Line where
  k : ℝ
  x₀ : ℝ
  y₀ : ℝ

/-- Definition of the ellipse C -/
def ellipse_C : Ellipse := {
  a := sorry,
  b := sorry,
  h_pos := sorry
}

/-- The ellipse C passes through (0, -1) -/
axiom passes_through : 0^2 / ellipse_C.a^2 + (-1)^2 / ellipse_C.b^2 = 1

/-- The eccentricity of C is √2/2 -/
axiom eccentricity : Real.sqrt ((ellipse_C.a^2 - ellipse_C.b^2) / ellipse_C.a^2) = Real.sqrt 2 / 2

/-- The equation of ellipse C -/
def ellipse_equation (p : Point) : Prop :=
  p.x^2 / 2 + p.y^2 = 1

/-- Point F -/
def F : Point := { x := 1, y := 0 }

/-- Theorem: The equation of ellipse C is x²/2 + y² = 1 -/
theorem ellipse_C_equation :
  ∀ p : Point, p.x^2 / ellipse_C.a^2 + p.y^2 / ellipse_C.b^2 = 1 ↔ ellipse_equation p :=
sorry

/-- Theorem: The ratio |MN|/|PF| is constant for any non-zero slope k -/
theorem constant_ratio (k : ℝ) (hk : k ≠ 0) :
  ∃ M N P : Point,
    (∃ l : Line, l.k = k ∧ l.x₀ = F.x ∧ l.y₀ = F.y) ∧
    ellipse_equation M ∧
    ellipse_equation N ∧
    (P.y = 0) ∧
    (N.y - M.y) * (P.x - M.x) = (M.x - N.x) * (P.y - M.y) →
    (N.x - M.x)^2 + (N.y - M.y)^2 = 8 * ((P.x - F.x)^2 + (P.y - F.y)^2) :=
sorry

end NUMINAMATH_CALUDE_ellipse_C_equation_constant_ratio_l3078_307889


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l3078_307812

theorem quadratic_equation_solution : 
  ∀ x : ℝ, x^2 - 2*x = 0 ↔ x = 0 ∨ x = 2 := by sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l3078_307812


namespace NUMINAMATH_CALUDE_four_distinct_roots_implies_c_magnitude_l3078_307861

/-- The polynomial Q(x) -/
def Q (c x : ℂ) : ℂ := (x^2 - 2*x + 3) * (x^2 - c*x + 6) * (x^2 - 4*x + 12)

/-- The theorem statement -/
theorem four_distinct_roots_implies_c_magnitude (c : ℂ) :
  (∃ (s : Finset ℂ), s.card = 4 ∧ (∀ x ∈ s, Q c x = 0) ∧
   (∀ x : ℂ, Q c x = 0 → x ∈ s)) →
  Complex.abs c = Real.sqrt 11 := by
  sorry

end NUMINAMATH_CALUDE_four_distinct_roots_implies_c_magnitude_l3078_307861


namespace NUMINAMATH_CALUDE_complex_division_result_l3078_307857

theorem complex_division_result : (1 + 3*Complex.I) / (1 - Complex.I) = -1 + 2*Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_division_result_l3078_307857


namespace NUMINAMATH_CALUDE_coffee_mixture_cost_l3078_307899

/-- Proves that the cost of the second coffee brand is $116.67 per kg -/
theorem coffee_mixture_cost (brand1_cost brand2_cost mixture_price profit_rate : ℝ)
  (h1 : brand1_cost = 200)
  (h2 : mixture_price = 177)
  (h3 : profit_rate = 0.18)
  (h4 : (2 * brand1_cost + 3 * brand2_cost) / 5 * (1 + profit_rate) = mixture_price) :
  brand2_cost = 116.67 := by
  sorry

end NUMINAMATH_CALUDE_coffee_mixture_cost_l3078_307899


namespace NUMINAMATH_CALUDE_consecutive_even_product_l3078_307809

theorem consecutive_even_product : 442 * 444 * 446 = 87526608 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_even_product_l3078_307809


namespace NUMINAMATH_CALUDE_lunks_needed_for_bananas_l3078_307808

/-- Exchange rate of lunks to kunks -/
def lunk_to_kunk_rate : ℚ := 2 / 3

/-- Exchange rate of kunks to bananas -/
def kunk_to_banana_rate : ℚ := 5 / 6

/-- Number of bananas to purchase -/
def bananas_to_buy : ℕ := 20

/-- Theorem stating the number of lunks needed to buy the specified number of bananas -/
theorem lunks_needed_for_bananas :
  ⌈(bananas_to_buy : ℚ) / (kunk_to_banana_rate * lunk_to_kunk_rate)⌉ = 36 := by
  sorry

end NUMINAMATH_CALUDE_lunks_needed_for_bananas_l3078_307808


namespace NUMINAMATH_CALUDE_solve_equation_l3078_307851

theorem solve_equation : (45 : ℚ) / (8 - 3/4) = 180/29 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l3078_307851


namespace NUMINAMATH_CALUDE_nancy_deleted_files_l3078_307855

theorem nancy_deleted_files (initial_files : ℕ) (folders : ℕ) (files_per_folder : ℕ) : 
  initial_files = 43 →
  folders = 2 →
  files_per_folder = 6 →
  initial_files - (folders * files_per_folder) = 31 := by
sorry

end NUMINAMATH_CALUDE_nancy_deleted_files_l3078_307855


namespace NUMINAMATH_CALUDE_algebraic_expression_value_l3078_307896

theorem algebraic_expression_value (x : ℝ) (h : x^2 - 2*x = 3) : 2*x^2 - 4*x + 3 = 9 := by
  sorry

end NUMINAMATH_CALUDE_algebraic_expression_value_l3078_307896


namespace NUMINAMATH_CALUDE_sum_of_three_numbers_l3078_307841

theorem sum_of_three_numbers (a b c : ℝ) 
  (sum_ab : a + b = 35)
  (sum_bc : b + c = 48)
  (sum_ca : c + a = 60) :
  a + b + c = 71.5 := by
sorry

end NUMINAMATH_CALUDE_sum_of_three_numbers_l3078_307841


namespace NUMINAMATH_CALUDE_babysitting_earnings_l3078_307868

theorem babysitting_earnings (total : ℚ) 
  (h1 : total / 4 + total / 2 + 50 = total) : total = 200 := by
  sorry

end NUMINAMATH_CALUDE_babysitting_earnings_l3078_307868


namespace NUMINAMATH_CALUDE_journey_distance_l3078_307817

theorem journey_distance (total_time : ℝ) (speed1 : ℝ) (speed2 : ℝ) 
  (h1 : total_time = 10)
  (h2 : speed1 = 21)
  (h3 : speed2 = 24) :
  ∃ (distance : ℝ), 
    distance / (2 * speed1) + distance / (2 * speed2) = total_time ∧ 
    distance = 224 := by
  sorry

end NUMINAMATH_CALUDE_journey_distance_l3078_307817


namespace NUMINAMATH_CALUDE_train_speed_calculation_l3078_307891

-- Define the distance in meters
def distance : ℝ := 200

-- Define the time in seconds (as a variable)
variable (p : ℝ)

-- Define the speed conversion factor from m/s to km/h
def conversion_factor : ℝ := 3.6

-- Theorem statement
theorem train_speed_calculation (p : ℝ) (h : p > 0) :
  (distance / p) * conversion_factor = 720 / p := by
  sorry

#check train_speed_calculation

end NUMINAMATH_CALUDE_train_speed_calculation_l3078_307891


namespace NUMINAMATH_CALUDE_three_number_average_l3078_307803

theorem three_number_average (a b c : ℝ) 
  (h1 : a = 2 * b) 
  (h2 : a = 3 * c) 
  (h3 : a - c = 96) : 
  (a + b + c) / 3 = 88 := by
  sorry

end NUMINAMATH_CALUDE_three_number_average_l3078_307803


namespace NUMINAMATH_CALUDE_hexagon_area_l3078_307844

theorem hexagon_area (s : ℝ) (t : ℝ) : 
  s > 0 → t > 0 →
  (4 * s = 6 * t) →  -- Equal perimeters
  (s^2 = 16) →       -- Area of square is 16
  (6 * (t^2 * Real.sqrt 3) / 4) = (64 * Real.sqrt 3) / 3 := by
  sorry

end NUMINAMATH_CALUDE_hexagon_area_l3078_307844


namespace NUMINAMATH_CALUDE_absolute_value_simplification_l3078_307884

theorem absolute_value_simplification : |(-5^2 + 6 * 2)| = 13 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_simplification_l3078_307884


namespace NUMINAMATH_CALUDE_product_cost_price_l3078_307866

theorem product_cost_price (original_price : ℝ) (cost_price : ℝ) : 
  (0.8 * original_price - cost_price = 120) →
  (0.6 * original_price - cost_price = -20) →
  cost_price = 440 := by
  sorry

end NUMINAMATH_CALUDE_product_cost_price_l3078_307866


namespace NUMINAMATH_CALUDE_distance_between_points_l3078_307892

theorem distance_between_points : 
  let p1 : ℝ × ℝ := (3.5, -2)
  let p2 : ℝ × ℝ := (7.5, 5)
  Real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2) = Real.sqrt 65 := by
sorry

end NUMINAMATH_CALUDE_distance_between_points_l3078_307892


namespace NUMINAMATH_CALUDE_sum_of_fractions_l3078_307830

theorem sum_of_fractions : (3 : ℚ) / 4 + (6 : ℚ) / 9 = (17 : ℚ) / 12 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_fractions_l3078_307830


namespace NUMINAMATH_CALUDE_haley_trees_after_typhoon_l3078_307893

/-- The number of trees Haley has left after a typhoon -/
def trees_left (initial_trees dead_trees : ℕ) : ℕ :=
  initial_trees - dead_trees

/-- Theorem: Haley has 10 trees left after the typhoon -/
theorem haley_trees_after_typhoon :
  trees_left 12 2 = 10 := by
  sorry

end NUMINAMATH_CALUDE_haley_trees_after_typhoon_l3078_307893


namespace NUMINAMATH_CALUDE_inheritance_calculation_l3078_307819

theorem inheritance_calculation (inheritance : ℝ) : 
  (0.25 * inheritance + 0.15 * (inheritance - 0.25 * inheritance) = 20000) → 
  inheritance = 55172.41 := by
  sorry

end NUMINAMATH_CALUDE_inheritance_calculation_l3078_307819
