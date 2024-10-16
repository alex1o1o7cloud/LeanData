import Mathlib

namespace NUMINAMATH_CALUDE_factoring_quadratic_l492_49246

theorem factoring_quadratic (a : ℝ) : a^2 - 4*a + 3 = (a - 1) * (a - 3) := by
  sorry

#check factoring_quadratic

end NUMINAMATH_CALUDE_factoring_quadratic_l492_49246


namespace NUMINAMATH_CALUDE_inequality_proof_l492_49275

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  a^2 / (b + c) + b^2 / (c + a) + c^2 / (a + b) ≥ (1/2) * (a + b + c) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l492_49275


namespace NUMINAMATH_CALUDE_golf_cost_l492_49212

/-- If 5 rounds of golf cost $400, then one round of golf costs $80 -/
theorem golf_cost (total_cost : ℝ) (num_rounds : ℕ) (cost_per_round : ℝ) 
  (h1 : total_cost = 400)
  (h2 : num_rounds = 5)
  (h3 : total_cost = num_rounds * cost_per_round) : 
  cost_per_round = 80 := by
  sorry

end NUMINAMATH_CALUDE_golf_cost_l492_49212


namespace NUMINAMATH_CALUDE_collinear_probability_5x4_l492_49250

/-- A rectangular array of dots -/
structure DotArray :=
  (rows : ℕ)
  (cols : ℕ)

/-- The number of ways to choose k items from n items -/
def choose (n k : ℕ) : ℕ := sorry

/-- The number of collinear sets of 4 dots in a DotArray -/
def collinearSets (arr : DotArray) : ℕ := sorry

/-- The probability of choosing 4 collinear dots from a DotArray -/
def collinearProbability (arr : DotArray) : ℚ :=
  (collinearSets arr : ℚ) / choose (arr.rows * arr.cols) 4

/-- The main theorem -/
theorem collinear_probability_5x4 :
  collinearProbability ⟨5, 4⟩ = 2 / 4845 := by sorry

end NUMINAMATH_CALUDE_collinear_probability_5x4_l492_49250


namespace NUMINAMATH_CALUDE_unique_base_ten_l492_49280

/-- Converts a list of digits in base b to its decimal representation -/
def toDecimal (digits : List Nat) (b : Nat) : Nat :=
  digits.foldr (fun d acc => d + b * acc) 0

/-- Checks if the equation is valid in base b -/
def isValidEquation (b : Nat) : Prop :=
  toDecimal [8, 7, 3, 6, 4] b + toDecimal [9, 2, 4, 1, 7] b = toDecimal [1, 8, 5, 8, 7, 1] b

theorem unique_base_ten :
  ∃! b, isValidEquation b ∧ b = 10 := by sorry

end NUMINAMATH_CALUDE_unique_base_ten_l492_49280


namespace NUMINAMATH_CALUDE_algebraic_expression_value_l492_49218

theorem algebraic_expression_value : 
  ∀ (a b : ℝ), (2 * a * 1^2 - b * 1 = -3) → (a * 2^2 - b * 2 = -6) := by
  sorry

end NUMINAMATH_CALUDE_algebraic_expression_value_l492_49218


namespace NUMINAMATH_CALUDE_division_of_decimals_l492_49214

theorem division_of_decimals : (0.05 : ℝ) / 0.0025 = 20 := by sorry

end NUMINAMATH_CALUDE_division_of_decimals_l492_49214


namespace NUMINAMATH_CALUDE_smallest_divisible_by_15_20_18_l492_49266

theorem smallest_divisible_by_15_20_18 :
  ∃ n : ℕ, n > 0 ∧ 15 ∣ n ∧ 20 ∣ n ∧ 18 ∣ n ∧ ∀ m : ℕ, m > 0 → 15 ∣ m → 20 ∣ m → 18 ∣ m → n ≤ m :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_divisible_by_15_20_18_l492_49266


namespace NUMINAMATH_CALUDE_toothpicks_300th_stage_l492_49274

/-- The number of toothpicks in the nth stage of the pattern -/
def toothpicks (n : ℕ) : ℕ := 5 + (n - 1) * 4

/-- Theorem: The number of toothpicks in the 300th stage is 1201 -/
theorem toothpicks_300th_stage :
  toothpicks 300 = 1201 := by
  sorry

end NUMINAMATH_CALUDE_toothpicks_300th_stage_l492_49274


namespace NUMINAMATH_CALUDE_fraction_inequality_l492_49208

theorem fraction_inequality (a : ℝ) (h1 : 0 < a) (h2 : a < 1) :
  1 / a + 4 / (1 - a) ≥ 9 := by
  sorry

end NUMINAMATH_CALUDE_fraction_inequality_l492_49208


namespace NUMINAMATH_CALUDE_age_problem_l492_49222

/-- Given a group of 7 people, if adding a person of age x increases the average age by 2,
    and adding a person aged 15 decreases the average age by 1, then x = 39. -/
theorem age_problem (T : ℝ) (A : ℝ) (x : ℝ) : 
  T = 7 * A →
  T + x = 8 * (A + 2) →
  T + 15 = 8 * (A - 1) →
  x = 39 := by
  sorry

end NUMINAMATH_CALUDE_age_problem_l492_49222


namespace NUMINAMATH_CALUDE_circle_plus_five_two_l492_49271

-- Define the ⊕ operation
def circle_plus (a b : ℝ) : ℝ := 5 * a + 2 * b

-- State the theorem
theorem circle_plus_five_two : circle_plus 5 2 = 29 := by
  sorry

end NUMINAMATH_CALUDE_circle_plus_five_two_l492_49271


namespace NUMINAMATH_CALUDE_f_composition_negative_two_l492_49236

noncomputable def f (x : ℝ) : ℝ :=
  if x < 0 then x + 2 else 3^(x + 1)

theorem f_composition_negative_two : f (f (-2)) = 3 := by
  sorry

end NUMINAMATH_CALUDE_f_composition_negative_two_l492_49236


namespace NUMINAMATH_CALUDE_part_one_part_two_l492_49278

/-- The quadratic equation -/
def quadratic (k x : ℝ) : ℝ := k * x^2 + 4 * x + 1

/-- Part 1: Prove that if x = -1 is a solution, then k = 3 -/
theorem part_one (k : ℝ) :
  quadratic k (-1) = 0 → k = 3 := by sorry

/-- Part 2: Prove that if the equation has two real roots and k ≠ 0, then k ≤ 4 and k ≠ 0 -/
theorem part_two (k : ℝ) :
  (∃ x y : ℝ, x ≠ y ∧ quadratic k x = 0 ∧ quadratic k y = 0) →
  k ≠ 0 →
  k ≤ 4 ∧ k ≠ 0 := by sorry

end NUMINAMATH_CALUDE_part_one_part_two_l492_49278


namespace NUMINAMATH_CALUDE_perfect_square_expression_l492_49289

theorem perfect_square_expression (x : ℤ) :
  ∃ d : ℤ, (4 * x + 1 - Real.sqrt (8 * x + 1 : ℝ)) / 2 = d →
  ∃ k : ℤ, d = k^2 := by
sorry

end NUMINAMATH_CALUDE_perfect_square_expression_l492_49289


namespace NUMINAMATH_CALUDE_flour_bags_theorem_l492_49276

def measurements : List Int := [3, 1, 0, 2, 6, -1, 2, 1, -4, 1]

def standard_weight : Int := 100

theorem flour_bags_theorem (measurements : List Int) (standard_weight : Int) :
  measurements = [3, 1, 0, 2, 6, -1, 2, 1, -4, 1] →
  standard_weight = 100 →
  (∀ m ∈ measurements, |0| ≤ |m|) ∧
  (measurements.sum = 11) ∧
  (measurements.length * standard_weight + measurements.sum = 1011) :=
by sorry

end NUMINAMATH_CALUDE_flour_bags_theorem_l492_49276


namespace NUMINAMATH_CALUDE_square_area_problem_l492_49224

theorem square_area_problem (x : ℝ) : 
  (5 * x - 18 = 27 - 4 * x) →
  (5 * x - 18)^2 = 49 := by
  sorry

end NUMINAMATH_CALUDE_square_area_problem_l492_49224


namespace NUMINAMATH_CALUDE_x_gt_one_sufficient_not_necessary_for_abs_x_gt_one_l492_49237

theorem x_gt_one_sufficient_not_necessary_for_abs_x_gt_one :
  (∀ x : ℝ, x > 1 → |x| > 1) ∧
  (∃ x : ℝ, |x| > 1 ∧ x ≤ 1) := by
  sorry

end NUMINAMATH_CALUDE_x_gt_one_sufficient_not_necessary_for_abs_x_gt_one_l492_49237


namespace NUMINAMATH_CALUDE_complex_fraction_simplification_l492_49273

theorem complex_fraction_simplification :
  let z : ℂ := Complex.mk 3 8 / Complex.mk 1 (-4)
  (z.re = -29/17) ∧ (z.im = 20/17) := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_simplification_l492_49273


namespace NUMINAMATH_CALUDE_rose_difference_l492_49258

theorem rose_difference (santiago_roses garrett_roses : ℕ) 
  (h1 : santiago_roses = 58) 
  (h2 : garrett_roses = 24) : 
  santiago_roses - garrett_roses = 34 := by
sorry

end NUMINAMATH_CALUDE_rose_difference_l492_49258


namespace NUMINAMATH_CALUDE_planes_parallel_from_perpendicular_lines_l492_49259

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the relations
variable (parallel : Plane → Plane → Prop)
variable (perpendicular : Line → Plane → Prop)
variable (line_parallel : Line → Line → Prop)

-- State the theorem
theorem planes_parallel_from_perpendicular_lines 
  (α β : Plane) (m n : Line) :
  perpendicular m α → 
  perpendicular n β → 
  line_parallel m n → 
  parallel α β :=
by sorry

end NUMINAMATH_CALUDE_planes_parallel_from_perpendicular_lines_l492_49259


namespace NUMINAMATH_CALUDE_kabadi_kho_kho_players_l492_49220

theorem kabadi_kho_kho_players (kabadi_players : ℕ) (kho_kho_only : ℕ) (total_players : ℕ)
  (h1 : kabadi_players = 10)
  (h2 : kho_kho_only = 20)
  (h3 : total_players = 30) :
  kabadi_players + kho_kho_only - total_players = 0 :=
by sorry

end NUMINAMATH_CALUDE_kabadi_kho_kho_players_l492_49220


namespace NUMINAMATH_CALUDE_circle_equation_from_diameter_l492_49240

/-- Given a circle with diameter endpoints (2, 0) and (2, -2), its equation is (x - 2)² + (y + 1)² = 1 -/
theorem circle_equation_from_diameter (x y : ℝ) :
  let endpoint1 : ℝ × ℝ := (2, 0)
  let endpoint2 : ℝ × ℝ := (2, -2)
  (x - 2)^2 + (y + 1)^2 = 1 := by sorry

end NUMINAMATH_CALUDE_circle_equation_from_diameter_l492_49240


namespace NUMINAMATH_CALUDE_square_sum_fraction_difference_l492_49252

theorem square_sum_fraction_difference : 
  (2^2 + 4^2 + 6^2) / (1^2 + 3^2 + 5^2) - (1^2 + 3^2 + 5^2) / (2^2 + 4^2 + 6^2) = 1911 / 1960 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_fraction_difference_l492_49252


namespace NUMINAMATH_CALUDE_stating_third_number_formula_l492_49211

/-- 
Given a triangular array of positive odd numbers arranged as follows:
1
3  5
7  9  11
13 15 17 19
...
This function returns the third number from the left in the nth row.
-/
def thirdNumberInRow (n : ℕ) : ℕ :=
  n^2 - n + 5

/-- 
Theorem stating that for n ≥ 3, the third number from the left 
in the nth row of the described triangular array is n^2 - n + 5.
-/
theorem third_number_formula (n : ℕ) (h : n ≥ 3) : 
  thirdNumberInRow n = n^2 - n + 5 := by
  sorry

end NUMINAMATH_CALUDE_stating_third_number_formula_l492_49211


namespace NUMINAMATH_CALUDE_four_divided_by_p_l492_49288

theorem four_divided_by_p (p q : ℝ) (h1 : 4 / q = 18) (h2 : p - q = 0.2777777777777778) :
  4 / p = 8 := by
  sorry

end NUMINAMATH_CALUDE_four_divided_by_p_l492_49288


namespace NUMINAMATH_CALUDE_events_B_C_complementary_l492_49255

-- Define the sample space (faces of the cube)
def S : Set Nat := {1, 2, 3, 4, 5, 6}

-- Define events A, B, and C
def A : Set Nat := {n ∈ S | n % 2 = 1}
def B : Set Nat := {n ∈ S | n ≤ 3}
def C : Set Nat := {n ∈ S | n ≥ 4}

-- Theorem statement
theorem events_B_C_complementary : B ∪ C = S ∧ B ∩ C = ∅ :=
sorry

end NUMINAMATH_CALUDE_events_B_C_complementary_l492_49255


namespace NUMINAMATH_CALUDE_three_tenths_of_number_l492_49294

theorem three_tenths_of_number (n : ℚ) (h : (1/3) * (1/4) * n = 15) : (3/10) * n = 54 := by
  sorry

end NUMINAMATH_CALUDE_three_tenths_of_number_l492_49294


namespace NUMINAMATH_CALUDE_events_mutually_exclusive_not_complementary_l492_49231

-- Define the set of cards
inductive Card : Type
| Hearts : Card
| Spades : Card
| Diamonds : Card
| Clubs : Card

-- Define the set of people
inductive Person : Type
| A : Person
| B : Person
| C : Person
| D : Person

-- Define a distribution of cards
def Distribution := Person → Card

-- Define the event "A receives a club"
def A_receives_club (d : Distribution) : Prop := d Person.A = Card.Clubs

-- Define the event "B receives a club"
def B_receives_club (d : Distribution) : Prop := d Person.B = Card.Clubs

-- Theorem statement
theorem events_mutually_exclusive_not_complementary :
  (∀ d : Distribution, ¬(A_receives_club d ∧ B_receives_club d)) ∧
  (∃ d : Distribution, ¬(A_receives_club d ∨ B_receives_club d)) :=
sorry

end NUMINAMATH_CALUDE_events_mutually_exclusive_not_complementary_l492_49231


namespace NUMINAMATH_CALUDE_mikes_tire_changes_l492_49238

/-- Calculates the number of sets of tires Mike changed given his work schedule and task durations. -/
theorem mikes_tire_changes (wash_time min_per_car_wash : ℕ)
                            (oil_change_time min_per_oil_change : ℕ)
                            (tire_change_time min_per_tire_set : ℕ)
                            (cars_washed num_cars_washed : ℕ)
                            (oil_changes num_oil_changes : ℕ)
                            (total_work_time total_minutes : ℕ) :
  wash_time = 10 →
  oil_change_time = 15 →
  tire_change_time = 30 →
  cars_washed = 9 →
  oil_changes = 6 →
  total_work_time = 4 * 60 →
  (total_work_time - (cars_washed * wash_time + oil_changes * oil_change_time)) / tire_change_time = 2 :=
by sorry

end NUMINAMATH_CALUDE_mikes_tire_changes_l492_49238


namespace NUMINAMATH_CALUDE_race_time_calculation_l492_49256

/-- Represents a runner in the race -/
structure Runner where
  speed : ℝ

/-- Represents the race scenario -/
structure Race where
  distance : ℝ
  runner_a : Runner
  runner_b : Runner
  time_difference : ℝ
  distance_difference : ℝ

/-- The theorem to prove -/
theorem race_time_calculation (race : Race) 
  (h1 : race.distance = 1000)
  (h2 : race.time_difference = 10)
  (h3 : race.distance_difference = 20) :
  ∃ (t : ℝ), t = 490 ∧ t * race.runner_a.speed = race.distance :=
sorry

end NUMINAMATH_CALUDE_race_time_calculation_l492_49256


namespace NUMINAMATH_CALUDE_cone_volume_over_pi_l492_49230

-- Define the given parameters
def sector_angle : ℝ := 240
def circle_radius : ℝ := 15

-- Define the theorem
theorem cone_volume_over_pi (sector_angle : ℝ) (circle_radius : ℝ) :
  sector_angle = 240 ∧ circle_radius = 15 →
  ∃ (cone_volume : ℝ),
    cone_volume / π = 500 * Real.sqrt 5 / 3 := by
  sorry


end NUMINAMATH_CALUDE_cone_volume_over_pi_l492_49230


namespace NUMINAMATH_CALUDE_average_of_remaining_numbers_l492_49295

theorem average_of_remaining_numbers
  (total_count : Nat)
  (total_average : ℝ)
  (first_pair_average : ℝ)
  (second_pair_average : ℝ)
  (h_total_count : total_count = 6)
  (h_total_average : total_average = 4.60)
  (h_first_pair_average : first_pair_average = 3.4)
  (h_second_pair_average : second_pair_average = 3.8) :
  (total_count : ℝ) * total_average - 2 * first_pair_average - 2 * second_pair_average = 2 * 6.6 := by
sorry

end NUMINAMATH_CALUDE_average_of_remaining_numbers_l492_49295


namespace NUMINAMATH_CALUDE_existence_of_special_number_l492_49297

theorem existence_of_special_number (P : Finset Nat) (h_prime : ∀ p ∈ P, Nat.Prime p) :
  ∃ x : Nat,
    (∀ p ∈ P, ∃ a b : Nat, a > 0 ∧ b > 0 ∧ x = a^p + b^p) ∧
    (∀ p : Nat, p ∉ P → Nat.Prime p → ¬∃ a b : Nat, a > 0 ∧ b > 0 ∧ x = a^p + b^p) :=
by sorry

end NUMINAMATH_CALUDE_existence_of_special_number_l492_49297


namespace NUMINAMATH_CALUDE_tom_program_duration_l492_49281

/-- Represents the duration of a combined BS and Ph.D. program -/
structure ProgramDuration where
  bs : ℕ
  phd : ℕ

/-- Calculates the time taken to complete a program given the standard duration and a completion factor -/
def completionTime (d : ProgramDuration) (factor : ℚ) : ℚ :=
  factor * (d.bs + d.phd)

theorem tom_program_duration :
  let standard_duration : ProgramDuration := { bs := 3, phd := 5 }
  let completion_factor : ℚ := 3/4
  completionTime standard_duration completion_factor = 6 := by sorry

end NUMINAMATH_CALUDE_tom_program_duration_l492_49281


namespace NUMINAMATH_CALUDE_quadratic_distinct_roots_l492_49293

theorem quadratic_distinct_roots (c : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ x₁^2 + 2*x₁ + 4*c = 0 ∧ x₂^2 + 2*x₂ + 4*c = 0) →
  c < 1/4 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_distinct_roots_l492_49293


namespace NUMINAMATH_CALUDE_expression_simplification_l492_49282

theorem expression_simplification (x y : ℝ) (n : Nat) (h1 : x > 0) (h2 : y > 0) (h3 : x ≠ y) (h4 : n = 2 ∨ n = 3 ∨ n = 4) :
  let r := (x^2 + y^2) / (2*x*y)
  (((Real.sqrt (r + 1) - Real.sqrt (r - 1))^n - (Real.sqrt (r + 1) + Real.sqrt (r - 1))^n) /
   ((Real.sqrt (r + 1) - Real.sqrt (r - 1))^n + (Real.sqrt (r + 1) + Real.sqrt (r - 1))^n)) =
  (y^n - x^n) / (y^n + x^n) := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l492_49282


namespace NUMINAMATH_CALUDE_train_bridge_crossing_time_l492_49234

/-- The time taken for a train to cross a bridge -/
theorem train_bridge_crossing_time 
  (train_length : ℝ) 
  (train_speed_kmh : ℝ) 
  (bridge_length : ℝ) 
  (h1 : train_length = 120)
  (h2 : train_speed_kmh = 45)
  (h3 : bridge_length = 255.03) : 
  (train_length + bridge_length) / (train_speed_kmh * 1000 / 3600) = 30.0024 :=
by sorry

end NUMINAMATH_CALUDE_train_bridge_crossing_time_l492_49234


namespace NUMINAMATH_CALUDE_log_equation_solution_l492_49254

theorem log_equation_solution (a : ℝ) (h1 : a > 1) 
  (h2 : Real.log a / Real.log 5 + Real.log a / Real.log 3 = 
        (Real.log a / Real.log 5) * (Real.log a / Real.log 3)) : 
  a = 15 := by
  sorry

end NUMINAMATH_CALUDE_log_equation_solution_l492_49254


namespace NUMINAMATH_CALUDE_complex_fraction_equality_l492_49265

theorem complex_fraction_equality : (1 + Complex.I * Real.sqrt 3) ^ 2 / (Complex.I * Real.sqrt 3 - 1) = -2 - 2 * Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_equality_l492_49265


namespace NUMINAMATH_CALUDE_order_of_abc_l492_49229

theorem order_of_abc : 
  let a : ℝ := Real.rpow 0.9 (1/3)
  let b : ℝ := Real.rpow (1/3) 0.9
  let c : ℝ := (1/2) * (Real.log 9 / Real.log 27)
  c < b ∧ b < a := by sorry

end NUMINAMATH_CALUDE_order_of_abc_l492_49229


namespace NUMINAMATH_CALUDE_equation_solution_l492_49227

theorem equation_solution : ∃! x : ℝ, (1 / (x - 1) = 3 / (x - 3)) ∧ x = 0 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l492_49227


namespace NUMINAMATH_CALUDE_fourth_power_sum_l492_49279

theorem fourth_power_sum (a b c : ℝ) 
  (sum_eq : a + b + c = 3)
  (sum_squares_eq : a^2 + b^2 + c^2 = 5)
  (sum_cubes_eq : a^3 + b^3 + c^3 = 7) :
  a^4 + b^4 + c^4 = 3 := by
  sorry

end NUMINAMATH_CALUDE_fourth_power_sum_l492_49279


namespace NUMINAMATH_CALUDE_original_workers_is_seven_l492_49217

/-- Represents the work scenario described in the problem -/
structure WorkScenario where
  planned_days : ℕ
  absent_workers : ℕ
  actual_days : ℕ

/-- Calculates the original number of workers given a work scenario -/
def original_workers (scenario : WorkScenario) : ℕ :=
  (scenario.absent_workers * scenario.actual_days) / (scenario.actual_days - scenario.planned_days)

/-- The specific work scenario from the problem -/
def problem_scenario : WorkScenario :=
  { planned_days := 8
  , absent_workers := 3
  , actual_days := 14 }

/-- Theorem stating that the original number of workers in the problem scenario is 7 -/
theorem original_workers_is_seven :
  original_workers problem_scenario = 7 := by
  sorry

end NUMINAMATH_CALUDE_original_workers_is_seven_l492_49217


namespace NUMINAMATH_CALUDE_sharon_in_middle_l492_49269

-- Define the set of people
inductive Person : Type
  | Maren : Person
  | Aaron : Person
  | Sharon : Person
  | Darren : Person
  | Karen : Person

-- Define the seating arrangement as a function from position (1 to 5) to Person
def Seating := Fin 5 → Person

-- Define the constraints
def satisfies_constraints (s : Seating) : Prop :=
  -- Maren sat in the last car
  s 5 = Person.Maren ∧
  -- Aaron sat directly behind Sharon
  (∃ i : Fin 4, s i = Person.Sharon ∧ s (i.succ) = Person.Aaron) ∧
  -- Darren sat directly behind Karen
  (∃ i : Fin 4, s i = Person.Karen ∧ s (i.succ) = Person.Darren) ∧
  -- At least one person sat between Aaron and Maren
  (∃ i j : Fin 5, i < j ∧ j < 5 ∧ s i = Person.Aaron ∧ s j ≠ Person.Maren ∧ s (j+1) = Person.Maren)

-- Theorem: Given the constraints, Sharon must be in the middle car
theorem sharon_in_middle (s : Seating) (h : satisfies_constraints s) : s 3 = Person.Sharon :=
sorry

end NUMINAMATH_CALUDE_sharon_in_middle_l492_49269


namespace NUMINAMATH_CALUDE_example_linear_equation_l492_49284

/-- Represents a linear equation with two variables -/
structure LinearEquationTwoVars where
  a : ℝ
  b : ℝ
  c : ℝ
  eq : ℝ → ℝ → Prop
  h_eq : ∀ x y, eq x y ↔ a * x + b * y = c

/-- The equation 5x + y = 2 is a linear equation with two variables -/
theorem example_linear_equation : ∃ e : LinearEquationTwoVars, e.a = 5 ∧ e.b = 1 ∧ e.c = 2 := by
  sorry

end NUMINAMATH_CALUDE_example_linear_equation_l492_49284


namespace NUMINAMATH_CALUDE_min_value_2a_plus_b_plus_c_l492_49299

theorem min_value_2a_plus_b_plus_c (a b c : ℝ) 
  (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) 
  (h4 : a * (a + b + c) + b * c = 4) : 
  ∀ x y z, x > 0 ∧ y > 0 ∧ z > 0 ∧ x * (x + y + z) + y * z = 4 → 2*a + b + c ≤ 2*x + y + z :=
by sorry

end NUMINAMATH_CALUDE_min_value_2a_plus_b_plus_c_l492_49299


namespace NUMINAMATH_CALUDE_fraction_simplification_l492_49264

theorem fraction_simplification (d : ℝ) : (5 + 4 * d) / 9 + 3 = (32 + 4 * d) / 9 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l492_49264


namespace NUMINAMATH_CALUDE_triangle_base_length_l492_49244

/-- Proves that a triangle with height 8 cm and area 24 cm² has a base length of 6 cm -/
theorem triangle_base_length (height : ℝ) (area : ℝ) (base : ℝ) : 
  height = 8 → area = 24 → area = (base * height) / 2 → base = 6 := by
  sorry

end NUMINAMATH_CALUDE_triangle_base_length_l492_49244


namespace NUMINAMATH_CALUDE_triangle_side_length_l492_49263

theorem triangle_side_length (b : ℝ) (B : ℝ) (A : ℝ) (a : ℝ) :
  b = 5 → B = π / 4 → Real.sin A = 1 / 3 → a = 5 * Real.sqrt 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_triangle_side_length_l492_49263


namespace NUMINAMATH_CALUDE_jean_jail_time_l492_49245

/-- Calculates the total jail time for Jean based on his charges --/
def total_jail_time (arson_counts : ℕ) (burglary_charges : ℕ) (arson_sentence : ℕ) (burglary_sentence : ℕ) : ℕ :=
  let petty_larceny_charges := 6 * burglary_charges
  let petty_larceny_sentence := burglary_sentence / 3
  arson_counts * arson_sentence + 
  burglary_charges * burglary_sentence + 
  petty_larceny_charges * petty_larceny_sentence

/-- Theorem stating that Jean's total jail time is 216 months --/
theorem jean_jail_time :
  total_jail_time 3 2 36 18 = 216 := by
  sorry

#eval total_jail_time 3 2 36 18

end NUMINAMATH_CALUDE_jean_jail_time_l492_49245


namespace NUMINAMATH_CALUDE_unique_a_with_prime_roots_l492_49298

theorem unique_a_with_prime_roots : ∃! a : ℕ+, 
  ∃ p q : ℕ, Prime p ∧ Prime q ∧ p ≠ q ∧ 
  (2 : ℝ) * p^2 - 30 * p + (a : ℝ) = 0 ∧
  (2 : ℝ) * q^2 - 30 * q + (a : ℝ) = 0 ∧
  a = 52 := by
sorry

end NUMINAMATH_CALUDE_unique_a_with_prime_roots_l492_49298


namespace NUMINAMATH_CALUDE_pitcher_juice_distribution_l492_49226

theorem pitcher_juice_distribution (C : ℝ) (h : C > 0) :
  let juice_amount : ℝ := (3 / 4) * C
  let cups : ℕ := 8
  let juice_per_cup : ℝ := juice_amount / cups
  let percent_per_cup : ℝ := (juice_per_cup / C) * 100
  percent_per_cup = 9.375 := by
  sorry

end NUMINAMATH_CALUDE_pitcher_juice_distribution_l492_49226


namespace NUMINAMATH_CALUDE_equal_commission_l492_49251

/-- The list price of the item -/
def list_price : ℝ := 34

/-- Alice's selling price -/
def alice_price (x : ℝ) : ℝ := x - 15

/-- Bob's selling price -/
def bob_price (x : ℝ) : ℝ := x - 25

/-- Alice's commission rate -/
def alice_rate : ℝ := 0.12

/-- Bob's commission rate -/
def bob_rate : ℝ := 0.25

/-- Alice's commission -/
def alice_commission (x : ℝ) : ℝ := alice_rate * alice_price x

/-- Bob's commission -/
def bob_commission (x : ℝ) : ℝ := bob_rate * bob_price x

theorem equal_commission :
  alice_commission list_price = bob_commission list_price :=
sorry

end NUMINAMATH_CALUDE_equal_commission_l492_49251


namespace NUMINAMATH_CALUDE_hexagon_diagonal_intersection_probability_l492_49223

/-- A convex hexagon -/
structure ConvexHexagon where
  -- Add any necessary properties of a convex hexagon

/-- A diagonal of a convex hexagon -/
structure Diagonal (H : ConvexHexagon) where
  -- Add any necessary properties of a diagonal

/-- Two diagonals intersect inside the hexagon (not at a vertex) -/
def intersect_inside (H : ConvexHexagon) (d1 d2 : Diagonal H) : Prop :=
  sorry

/-- The probability of two randomly chosen diagonals intersecting inside the hexagon -/
def intersection_probability (H : ConvexHexagon) : ℚ :=
  sorry

/-- Theorem: The probability of two randomly chosen diagonals intersecting inside a convex hexagon is 5/12 -/
theorem hexagon_diagonal_intersection_probability (H : ConvexHexagon) :
  intersection_probability H = 5 / 12 :=
sorry

end NUMINAMATH_CALUDE_hexagon_diagonal_intersection_probability_l492_49223


namespace NUMINAMATH_CALUDE_largest_divisor_consecutive_odd_squares_l492_49261

theorem largest_divisor_consecutive_odd_squares (m n : ℤ) : 
  (∃ k : ℤ, n = 2*k + 1 ∧ m = 2*k + 3) →  -- m and n are consecutive odd integers
  n < m →                                 -- n is less than m
  (∀ d : ℤ, d ∣ (m^2 - n^2) → d ≤ 8) ∧    -- 8 is an upper bound for divisors
  8 ∣ (m^2 - n^2)                         -- 8 divides m^2 - n^2
  := by sorry

end NUMINAMATH_CALUDE_largest_divisor_consecutive_odd_squares_l492_49261


namespace NUMINAMATH_CALUDE_john_chocolate_gain_l492_49228

/-- Represents the chocolate types --/
inductive ChocolateType
  | A
  | B
  | C

/-- Represents a purchase of chocolates --/
structure Purchase where
  chocolateType : ChocolateType
  quantity : ℕ
  costPrice : ℚ

/-- Represents a sale of chocolates --/
structure Sale where
  chocolateType : ChocolateType
  quantity : ℕ
  sellingPrice : ℚ

def purchases : List Purchase := [
  ⟨ChocolateType.A, 100, 2⟩,
  ⟨ChocolateType.B, 150, 3⟩,
  ⟨ChocolateType.C, 200, 4⟩
]

def sales : List Sale := [
  ⟨ChocolateType.A, 90, 5/2⟩,
  ⟨ChocolateType.A, 60, 3⟩,
  ⟨ChocolateType.B, 140, 7/2⟩,
  ⟨ChocolateType.B, 10, 4⟩,
  ⟨ChocolateType.B, 50, 5⟩,
  ⟨ChocolateType.C, 180, 9/2⟩,
  ⟨ChocolateType.C, 20, 5⟩
]

def totalCostPrice : ℚ :=
  purchases.foldr (fun p acc => acc + p.quantity * p.costPrice) 0

def totalSellingPrice : ℚ :=
  sales.foldr (fun s acc => acc + s.quantity * s.sellingPrice) 0

def gainPercentage : ℚ :=
  ((totalSellingPrice - totalCostPrice) / totalCostPrice) * 100

theorem john_chocolate_gain :
  gainPercentage = 89/2 := by sorry

end NUMINAMATH_CALUDE_john_chocolate_gain_l492_49228


namespace NUMINAMATH_CALUDE_polygon_sides_from_angle_sum_l492_49239

theorem polygon_sides_from_angle_sum (n : ℕ) (angle_sum : ℝ) : 
  angle_sum = 720 → (n - 2) * 180 = angle_sum → n = 6 := by
  sorry

end NUMINAMATH_CALUDE_polygon_sides_from_angle_sum_l492_49239


namespace NUMINAMATH_CALUDE_expression_simplification_l492_49225

theorem expression_simplification (q : ℝ) : 
  ((6 * q + 2) - 3 * q * 5) * 4 + (5 - 2 / 4) * (7 * q - 14) = -4.5 * q - 55 := by
sorry

end NUMINAMATH_CALUDE_expression_simplification_l492_49225


namespace NUMINAMATH_CALUDE_system_inequality_solution_l492_49233

theorem system_inequality_solution (a : ℝ) : 
  (∃ x : ℝ, x > 2 ∧ x > -1 ∧ x > a) → ∀ b : ℝ, ∃ x : ℝ, x > 2 ∧ x > -1 ∧ x > b :=
by sorry

end NUMINAMATH_CALUDE_system_inequality_solution_l492_49233


namespace NUMINAMATH_CALUDE_cayley_hamilton_for_A_l492_49210

def A : Matrix (Fin 3) (Fin 3) ℤ :=
  ![![1, 2, 3],
    ![2, 1, 2],
    ![3, 2, 1]]

theorem cayley_hamilton_for_A :
  A^3 + (-8 : ℤ) • A^2 + (-2 : ℤ) • A + (-8 : ℤ) • (1 : Matrix (Fin 3) (Fin 3) ℤ) = 0 := by
  sorry

end NUMINAMATH_CALUDE_cayley_hamilton_for_A_l492_49210


namespace NUMINAMATH_CALUDE_sum_of_parts_l492_49272

theorem sum_of_parts (x y : ℝ) (h1 : x + y = 60) (h2 : y = 45) (h3 : x ≥ 0) (h4 : y ≥ 0) :
  10 * x + 22 * y = 1140 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_parts_l492_49272


namespace NUMINAMATH_CALUDE_symmetric_lines_symmetric_line_equation_l492_49206

/-- Given two lines in the 2D plane and a point, this theorem states that these lines are symmetric with respect to the given point. -/
theorem symmetric_lines (x y : ℝ) : 
  (2 * x + 3 * y - 6 = 0) ↔ (2 * (2 - x) + 3 * (-2 - y) - 6 = 0) := by
  sorry

/-- The equation of the line symmetric to 2x + 3y - 6 = 0 with respect to the point (1, -1) is 2x + 3y + 8 = 0. -/
theorem symmetric_line_equation : 
  ∀ x y : ℝ, (2 * x + 3 * y - 6 = 0) ↔ (2 * ((2 - x) - 1) + 3 * ((-2 - y) - (-1)) + 8 = 0) := by
  sorry

end NUMINAMATH_CALUDE_symmetric_lines_symmetric_line_equation_l492_49206


namespace NUMINAMATH_CALUDE_geometric_sequence_general_term_l492_49267

def geometric_sequence (a : ℕ → ℚ) : Prop :=
  ∃ r : ℚ, ∀ n : ℕ, a (n + 1) = r * a n

theorem geometric_sequence_general_term 
  (a : ℕ → ℚ) 
  (h_geo : geometric_sequence a) 
  (h_a1 : a 1 = 3/2) 
  (h_S3 : (a 1) + (a 2) + (a 3) = 9/2) :
  (∃ n : ℕ, a n = 3/2 * (-2)^(n-1)) ∨ (∀ n : ℕ, a n = 3/2) :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_general_term_l492_49267


namespace NUMINAMATH_CALUDE_no_function_satisfies_inequality_l492_49260

/-- There does not exist a function satisfying the given inequality for all real numbers. -/
theorem no_function_satisfies_inequality :
  ¬ ∃ f : ℝ → ℝ, ∀ x y : ℝ, (f x + f y) / 2 ≥ f ((x + y) / 2) + |x - y| := by
  sorry

end NUMINAMATH_CALUDE_no_function_satisfies_inequality_l492_49260


namespace NUMINAMATH_CALUDE_isaac_number_problem_l492_49205

theorem isaac_number_problem (a b : ℤ) : 
  (2 * a + 3 * b = 100) → 
  ((a = 28 ∨ b = 28) → (a = 8 ∨ b = 8)) :=
by sorry

end NUMINAMATH_CALUDE_isaac_number_problem_l492_49205


namespace NUMINAMATH_CALUDE_systematic_sample_fourth_number_l492_49241

def class_size : ℕ := 48
def sample_size : ℕ := 4
def interval : ℕ := class_size / sample_size

def is_valid_sample (s : Finset ℕ) : Prop :=
  s.card = sample_size ∧ 
  ∀ x ∈ s, 1 ≤ x ∧ x ≤ class_size ∧
  ∃ k : ℕ, x = 1 + k * interval

theorem systematic_sample_fourth_number :
  ∀ s : Finset ℕ, is_valid_sample s →
  (5 ∈ s ∧ 29 ∈ s ∧ 41 ∈ s) →
  17 ∈ s :=
by sorry

end NUMINAMATH_CALUDE_systematic_sample_fourth_number_l492_49241


namespace NUMINAMATH_CALUDE_lemonade_proportion_lemons_for_lemonade_l492_49207

theorem lemonade_proportion (lemons_initial : ℝ) (gallons_initial : ℝ) (gallons_target : ℝ) :
  lemons_initial > 0 ∧ gallons_initial > 0 ∧ gallons_target > 0 →
  let lemons_target := (lemons_initial * gallons_target) / gallons_initial
  lemons_initial / gallons_initial = lemons_target / gallons_target :=
by
  sorry

theorem lemons_for_lemonade :
  let lemons_initial : ℝ := 36
  let gallons_initial : ℝ := 48
  let gallons_target : ℝ := 10
  (lemons_initial * gallons_target) / gallons_initial = 7.5 :=
by
  sorry

end NUMINAMATH_CALUDE_lemonade_proportion_lemons_for_lemonade_l492_49207


namespace NUMINAMATH_CALUDE_sandwich_non_condiment_percentage_l492_49209

theorem sandwich_non_condiment_percentage
  (total_weight : ℝ)
  (condiment_weight : ℝ)
  (h1 : total_weight = 150)
  (h2 : condiment_weight = 45) :
  (total_weight - condiment_weight) / total_weight * 100 = 70 := by
  sorry

end NUMINAMATH_CALUDE_sandwich_non_condiment_percentage_l492_49209


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l492_49286

theorem sufficient_not_necessary_condition (m : ℝ) : m = 1/2 →
  m > 0 ∧
  (∀ x : ℝ, 0 < x ∧ x < m → x * (x - 1) < 0) ∧
  (∃ x : ℝ, x * (x - 1) < 0 ∧ ¬(0 < x ∧ x < m)) := by
  sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l492_49286


namespace NUMINAMATH_CALUDE_sticker_collection_probability_l492_49235

theorem sticker_collection_probability : 
  let total_stickers : ℕ := 18
  let selected_stickers : ℕ := 10
  let uncollected_stickers : ℕ := 6
  let collected_stickers : ℕ := 12
  (Nat.choose uncollected_stickers uncollected_stickers * Nat.choose collected_stickers (selected_stickers - uncollected_stickers)) / 
  Nat.choose total_stickers selected_stickers = 5 / 442 := by
sorry

end NUMINAMATH_CALUDE_sticker_collection_probability_l492_49235


namespace NUMINAMATH_CALUDE_cube_difference_identity_l492_49277

theorem cube_difference_identity (a b : ℝ) : 
  (a^3 + b^3 = (a + b) * (a^2 - a*b + b^2)) → 
  (a^3 - b^3 = (a - b) * (a^2 + a*b + b^2)) := by
sorry

end NUMINAMATH_CALUDE_cube_difference_identity_l492_49277


namespace NUMINAMATH_CALUDE_school_population_theorem_l492_49219

theorem school_population_theorem (total : ℕ) (boys : ℕ) (teachers : ℕ) (girls : ℕ) : 
  total = 1396 → boys = 309 → teachers = 772 → girls = total - boys - teachers → girls = 315 :=
by sorry

end NUMINAMATH_CALUDE_school_population_theorem_l492_49219


namespace NUMINAMATH_CALUDE_power_product_equals_4410000_l492_49257

theorem power_product_equals_4410000 : 2^4 * 3^2 * 5^4 * 7^2 = 4410000 := by
  sorry

end NUMINAMATH_CALUDE_power_product_equals_4410000_l492_49257


namespace NUMINAMATH_CALUDE_scientific_notation_of_small_number_l492_49213

/-- Represents a number in scientific notation -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  property : 1 ≤ coefficient ∧ coefficient < 10

/-- Converts a real number to scientific notation -/
def toScientificNotation (x : ℝ) : ScientificNotation :=
  sorry

theorem scientific_notation_of_small_number :
  toScientificNotation 0.000000032 = ScientificNotation.mk 3.2 (-8) sorry := by
  sorry

end NUMINAMATH_CALUDE_scientific_notation_of_small_number_l492_49213


namespace NUMINAMATH_CALUDE_team_win_percentage_l492_49290

theorem team_win_percentage (total_games : ℕ) (win_rate : ℚ) : 
  total_games = 75 → win_rate = 65/100 → 
  (win_rate * total_games) / total_games = 65/100 := by
  sorry

end NUMINAMATH_CALUDE_team_win_percentage_l492_49290


namespace NUMINAMATH_CALUDE_expression_simplification_and_evaluation_l492_49291

theorem expression_simplification_and_evaluation :
  ∀ x : ℝ, x ≠ 1 → x ≠ 2 →
  (x + 1 - 3 / (x - 1)) / ((x^2 - 4*x + 4) / (x - 1)) = (x + 2) / (x - 2) ∧
  (3 + 1 - 3 / (3 - 1)) / ((3^2 - 4*3 + 4) / (3 - 1)) = 5 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_and_evaluation_l492_49291


namespace NUMINAMATH_CALUDE_xyz_sum_max_min_l492_49283

theorem xyz_sum_max_min (x y z : ℝ) (h : 4 * (x + y + z) = x^2 + y^2 + z^2) :
  let f := fun (a b c : ℝ) => a * b + a * c + b * c
  ∃ (M m : ℝ), (∀ (a b c : ℝ), 4 * (a + b + c) = a^2 + b^2 + c^2 → f a b c ≤ M) ∧
               (∀ (a b c : ℝ), 4 * (a + b + c) = a^2 + b^2 + c^2 → m ≤ f a b c) ∧
               M + 10 * m = 28 :=
by sorry

end NUMINAMATH_CALUDE_xyz_sum_max_min_l492_49283


namespace NUMINAMATH_CALUDE_complex_multiplication_l492_49296

theorem complex_multiplication (i : ℂ) : i * i = -1 → i * (1 + i) = -1 + i := by
  sorry

end NUMINAMATH_CALUDE_complex_multiplication_l492_49296


namespace NUMINAMATH_CALUDE_six_by_six_grid_shaded_half_l492_49216

/-- Represents a square grid -/
structure Grid :=
  (size : ℕ)
  (shaded_per_row : ℕ)

/-- Calculates the percentage of shaded area in a grid -/
def shaded_percentage (g : Grid) : ℚ :=
  (g.size * g.shaded_per_row : ℚ) / (g.size * g.size)

/-- The main theorem: for a 6x6 grid with 3 shaded squares per row,
    the shaded percentage is 50% -/
theorem six_by_six_grid_shaded_half :
  let g : Grid := { size := 6, shaded_per_row := 3 }
  shaded_percentage g = 1/2 := by sorry

end NUMINAMATH_CALUDE_six_by_six_grid_shaded_half_l492_49216


namespace NUMINAMATH_CALUDE_board_coverage_five_by_five_uncoverable_four_by_four_removed_uncoverable_four_by_five_coverable_six_by_three_coverable_l492_49248

/-- Represents a checkerboard -/
structure Checkerboard where
  rows : Nat
  cols : Nat
  removed_squares : Nat

/-- Checks if a board can be completely covered by non-overlapping dominoes -/
def can_be_covered (board : Checkerboard) : Prop :=
  (board.rows * board.cols - board.removed_squares) % 2 = 0

/-- Theorem: A board can be covered iff the number of squares is even -/
theorem board_coverage (board : Checkerboard) :
  can_be_covered board ↔ (board.rows * board.cols - board.removed_squares) % 2 = 0 := by
  sorry

/-- 5x5 board cannot be covered -/
theorem five_by_five_uncoverable :
  ¬(can_be_covered { rows := 5, cols := 5, removed_squares := 0 }) := by
  sorry

/-- 4x4 board with one square removed cannot be covered -/
theorem four_by_four_removed_uncoverable :
  ¬(can_be_covered { rows := 4, cols := 4, removed_squares := 1 }) := by
  sorry

/-- 4x5 board can be covered -/
theorem four_by_five_coverable :
  can_be_covered { rows := 4, cols := 5, removed_squares := 0 } := by
  sorry

/-- 6x3 board can be covered -/
theorem six_by_three_coverable :
  can_be_covered { rows := 6, cols := 3, removed_squares := 0 } := by
  sorry

end NUMINAMATH_CALUDE_board_coverage_five_by_five_uncoverable_four_by_four_removed_uncoverable_four_by_five_coverable_six_by_three_coverable_l492_49248


namespace NUMINAMATH_CALUDE_quadratic_form_j_value_l492_49215

theorem quadratic_form_j_value 
  (a b c : ℝ) 
  (h1 : ∀ x, a * x^2 + b * x + c = 5 * (x - 3)^2 + 15) 
  (m n j : ℝ) 
  (h2 : ∀ x, 4 * (a * x^2 + b * x + c) = m * (x - j)^2 + n) : 
  j = 3 := by
sorry

end NUMINAMATH_CALUDE_quadratic_form_j_value_l492_49215


namespace NUMINAMATH_CALUDE_radical_product_simplification_l492_49201

theorem radical_product_simplification (q : ℝ) (hq : q > 0) :
  2 * Real.sqrt (20 * q) * Real.sqrt (10 * q) * Real.sqrt (15 * q) = 60 * q * Real.sqrt (30 * q) := by
  sorry

end NUMINAMATH_CALUDE_radical_product_simplification_l492_49201


namespace NUMINAMATH_CALUDE_equal_angles_l492_49253

-- Define the basic structures
variable (Circle₁ Circle₂ : Set (ℝ × ℝ))
variable (K M A B C D : ℝ × ℝ)

-- Define the conditions
variable (h1 : K ∈ Circle₁ ∩ Circle₂)
variable (h2 : M ∈ Circle₁ ∩ Circle₂)
variable (h3 : A ∈ Circle₁)
variable (h4 : B ∈ Circle₂)
variable (h5 : C ∈ Circle₁)
variable (h6 : D ∈ Circle₂)
variable (h7 : ∃ ray₁ : Set (ℝ × ℝ), K ∈ ray₁ ∧ A ∈ ray₁ ∧ B ∈ ray₁)
variable (h8 : ∃ ray₂ : Set (ℝ × ℝ), K ∈ ray₂ ∧ C ∈ ray₂ ∧ D ∈ ray₂)

-- Define the angle function
noncomputable def angle (p q r : ℝ × ℝ) : ℝ := sorry

-- State the theorem
theorem equal_angles : angle M A B = angle M C D := sorry

end NUMINAMATH_CALUDE_equal_angles_l492_49253


namespace NUMINAMATH_CALUDE_average_sleep_time_l492_49221

def sleep_times : List ℝ := [10, 9, 10, 8, 8]

theorem average_sleep_time : (sleep_times.sum / sleep_times.length) = 9 := by
  sorry

end NUMINAMATH_CALUDE_average_sleep_time_l492_49221


namespace NUMINAMATH_CALUDE_infinitely_many_non_prime_n4_plus_k_l492_49249

theorem infinitely_many_non_prime_n4_plus_k :
  ∀ N : ℕ, ∃ k : ℕ, k > N ∧ ∀ n : ℕ, ¬ Prime (n^4 + k) := by
  sorry

end NUMINAMATH_CALUDE_infinitely_many_non_prime_n4_plus_k_l492_49249


namespace NUMINAMATH_CALUDE_eighth_term_of_sequence_l492_49243

/-- Given a geometric sequence with first term a and common ratio r,
    the nth term of the sequence is given by a * r^(n-1) -/
def geometric_sequence (a : ℚ) (r : ℚ) (n : ℕ) : ℚ :=
  a * r^(n - 1)

/-- The 8th term of a geometric sequence with first term 5 and common ratio 2/3
    is equal to 640/2187 -/
theorem eighth_term_of_sequence :
  geometric_sequence 5 (2/3) 8 = 640/2187 := by
  sorry

end NUMINAMATH_CALUDE_eighth_term_of_sequence_l492_49243


namespace NUMINAMATH_CALUDE_modulo_equivalence_l492_49262

theorem modulo_equivalence : ∃! n : ℤ, 0 ≤ n ∧ n < 23 ∧ 54126 ≡ n [ZMOD 23] ∧ n = 13 := by
  sorry

end NUMINAMATH_CALUDE_modulo_equivalence_l492_49262


namespace NUMINAMATH_CALUDE_molecular_weight_CCl4_is_152_l492_49285

/-- The molecular weight of Carbon tetrachloride -/
def molecular_weight_CCl4 : ℝ := 152

/-- The number of moles in the given sample -/
def num_moles : ℝ := 9

/-- The total molecular weight of the given sample -/
def total_weight : ℝ := 1368

/-- Theorem stating that the molecular weight of Carbon tetrachloride is 152 g/mol -/
theorem molecular_weight_CCl4_is_152 :
  molecular_weight_CCl4 = total_weight / num_moles :=
by
  sorry

end NUMINAMATH_CALUDE_molecular_weight_CCl4_is_152_l492_49285


namespace NUMINAMATH_CALUDE_arithmetic_sequence_general_term_l492_49270

/-- An increasing arithmetic sequence with specific initial conditions -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  (∀ n, a (n + 1) > a n) ∧  -- Increasing
  (∃ d, ∀ n, a (n + 1) = a n + d) ∧  -- Arithmetic
  (a 1 = 1) ∧  -- Initial condition
  (a 3 = (a 2)^2 - 4)  -- Given relation

/-- The theorem stating the general term of the sequence -/
theorem arithmetic_sequence_general_term (a : ℕ → ℝ) 
  (h : ArithmeticSequence a) : 
  ∀ n : ℕ, a n = 2 * n - 1 :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_general_term_l492_49270


namespace NUMINAMATH_CALUDE_largest_even_not_sum_of_odd_composites_l492_49200

/-- A number is composite if it has a factor other than 1 and itself -/
def IsComposite (n : ℕ) : Prop :=
  ∃ k m : ℕ, k > 1 ∧ m > 1 ∧ n = k * m

/-- A number is odd if it leaves a remainder of 1 when divided by 2 -/
def IsOdd (n : ℕ) : Prop :=
  n % 2 = 1

/-- The property of being expressible as the sum of two odd composite numbers -/
def IsSumOfTwoOddComposites (n : ℕ) : Prop :=
  ∃ a b : ℕ, IsOdd a ∧ IsOdd b ∧ IsComposite a ∧ IsComposite b ∧ n = a + b

/-- 38 is the largest even integer that cannot be written as the sum of two odd composite numbers -/
theorem largest_even_not_sum_of_odd_composites :
  (∀ n : ℕ, n % 2 = 0 → n > 38 → IsSumOfTwoOddComposites n) ∧
  ¬IsSumOfTwoOddComposites 38 :=
sorry

end NUMINAMATH_CALUDE_largest_even_not_sum_of_odd_composites_l492_49200


namespace NUMINAMATH_CALUDE_rectangle_area_ratio_l492_49287

theorem rectangle_area_ratio (x y l : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : l > 0)
  (h4 : 2 * (x + 3 * y) = 2 * (l + y)) -- Equal perimeters
  (h5 : 2 * x + l = 3 * y) -- Square property
  : (x * (3 * y)) / (l * y) = 3 / 7 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_ratio_l492_49287


namespace NUMINAMATH_CALUDE_convex_n_gon_interior_angles_ratio_l492_49202

theorem convex_n_gon_interior_angles_ratio (n : ℕ) : 
  n ≥ 3 →
  ∃ x : ℝ, x > 0 ∧
    (∀ k : ℕ, k ≤ n → k * x < 180) ∧
    n * (n + 1) / 2 * x = (n - 2) * 180 →
  n = 3 ∨ n = 4 :=
sorry

end NUMINAMATH_CALUDE_convex_n_gon_interior_angles_ratio_l492_49202


namespace NUMINAMATH_CALUDE_different_tens_digit_probability_l492_49203

/-- The number of integers to be chosen -/
def n : ℕ := 6

/-- The lower bound of the range (inclusive) -/
def lower_bound : ℕ := 10

/-- The upper bound of the range (inclusive) -/
def upper_bound : ℕ := 79

/-- The total number of integers in the range -/
def total_numbers : ℕ := upper_bound - lower_bound + 1

/-- The number of different tens digits in the range -/
def tens_digits : ℕ := 7

/-- The probability of choosing n different integers from the range
    such that they each have a different tens digit -/
def probability : ℚ := 1750 / 2980131

theorem different_tens_digit_probability :
  probability = (tens_digits.choose n * (10 ^ n : ℕ)) / total_numbers.choose n :=
sorry

end NUMINAMATH_CALUDE_different_tens_digit_probability_l492_49203


namespace NUMINAMATH_CALUDE_polynomial_remainder_theorem_l492_49268

theorem polynomial_remainder_theorem (a b : ℚ) : 
  let f : ℚ → ℚ := λ x ↦ a * x^4 + 3 * x^3 - 5 * x^2 + b * x - 7
  (f 2 = 9 ∧ f (-1) = -4) → (a = 7/9 ∧ b = -2/9) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_remainder_theorem_l492_49268


namespace NUMINAMATH_CALUDE_annika_three_times_hans_age_l492_49242

/-- The number of years in the future when Annika will be three times as old as Hans -/
def future_years : ℕ := 4

/-- Hans's current age -/
def hans_current_age : ℕ := 8

/-- Annika's current age -/
def annika_current_age : ℕ := 32

theorem annika_three_times_hans_age :
  annika_current_age + future_years = 3 * (hans_current_age + future_years) :=
by sorry

end NUMINAMATH_CALUDE_annika_three_times_hans_age_l492_49242


namespace NUMINAMATH_CALUDE_house_rent_percentage_l492_49204

def total_income : ℝ := 1000
def petrol_percentage : ℝ := 0.3
def petrol_expenditure : ℝ := 300
def house_rent : ℝ := 210

theorem house_rent_percentage : 
  (house_rent / (total_income * (1 - petrol_percentage))) * 100 = 30 := by
  sorry

end NUMINAMATH_CALUDE_house_rent_percentage_l492_49204


namespace NUMINAMATH_CALUDE_min_people_in_group_l492_49247

/-- Represents the number of people who like a specific fruit or combination of fruits. -/
structure FruitPreferences where
  apples : Nat
  blueberries : Nat
  cantaloupe : Nat
  dates : Nat
  blueberriesAndApples : Nat
  blueberriesAndCantaloupe : Nat
  cantaloupeAndDates : Nat

/-- The conditions given in the problem. -/
def problemConditions : FruitPreferences where
  apples := 13
  blueberries := 9
  cantaloupe := 15
  dates := 6
  blueberriesAndApples := 0  -- Derived from the solution
  blueberriesAndCantaloupe := 9  -- Derived from the solution
  cantaloupeAndDates := 6  -- Derived from the solution

/-- Theorem stating the minimum number of people in the group. -/
theorem min_people_in_group (prefs : FruitPreferences) 
  (h1 : prefs.blueberries = prefs.blueberriesAndApples + prefs.blueberriesAndCantaloupe)
  (h2 : prefs.cantaloupe = prefs.blueberriesAndCantaloupe + prefs.cantaloupeAndDates)
  (h3 : prefs = problemConditions) :
  prefs.apples + prefs.blueberriesAndCantaloupe + prefs.cantaloupeAndDates = 22 := by
  sorry

end NUMINAMATH_CALUDE_min_people_in_group_l492_49247


namespace NUMINAMATH_CALUDE_outlet_pipe_emptying_time_l492_49292

theorem outlet_pipe_emptying_time 
  (fill_time_1 : ℝ) 
  (fill_time_2 : ℝ) 
  (combined_fill_time : ℝ) 
  (h1 : fill_time_1 = 18) 
  (h2 : fill_time_2 = 30) 
  (h3 : combined_fill_time = 0.06666666666666665) :
  let fill_rate_1 := 1 / fill_time_1
  let fill_rate_2 := 1 / fill_time_2
  let combined_fill_rate := 1 / combined_fill_time
  ∃ (empty_time : ℝ), 
    fill_rate_1 + fill_rate_2 - (1 / empty_time) = combined_fill_rate ∧ 
    empty_time = 45 :=
by sorry

end NUMINAMATH_CALUDE_outlet_pipe_emptying_time_l492_49292


namespace NUMINAMATH_CALUDE_train_crossing_time_l492_49232

/-- The time taken for a train to cross a pole -/
theorem train_crossing_time (train_speed_kmh : ℝ) (train_length_m : ℝ) : 
  train_speed_kmh = 72 → train_length_m = 180 → 
  (train_length_m / (train_speed_kmh * 1000 / 3600)) = 9 := by
  sorry

end NUMINAMATH_CALUDE_train_crossing_time_l492_49232
