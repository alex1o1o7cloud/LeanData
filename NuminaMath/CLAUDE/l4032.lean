import Mathlib

namespace NUMINAMATH_CALUDE_inequality_proof_l4032_403292

theorem inequality_proof (a b c : ℝ) 
  (h_pos : a > 0 ∧ b > 0 ∧ c > 0) 
  (h_sum : a + b + c = 1) : 
  (a^2 + b^2 + c^2 ≥ 1/3) ∧ 
  (a^2/b + b^2/c + c^2/a ≥ 1) := by
sorry

end NUMINAMATH_CALUDE_inequality_proof_l4032_403292


namespace NUMINAMATH_CALUDE_harmonic_mean_of_1_and_5040_l4032_403259

def harmonic_mean (a b : ℚ) : ℚ := 2 * a * b / (a + b)

theorem harmonic_mean_of_1_and_5040 :
  harmonic_mean 1 5040 = 10080 / 5041 := by
  sorry

end NUMINAMATH_CALUDE_harmonic_mean_of_1_and_5040_l4032_403259


namespace NUMINAMATH_CALUDE_expand_polynomial_l4032_403271

theorem expand_polynomial (x : ℝ) : (1 + x^3) * (1 - x^4 + x^5) = 1 + x^3 - x^4 + x^5 - x^7 + x^8 := by
  sorry

end NUMINAMATH_CALUDE_expand_polynomial_l4032_403271


namespace NUMINAMATH_CALUDE_desired_circle_properties_l4032_403297

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

end NUMINAMATH_CALUDE_desired_circle_properties_l4032_403297


namespace NUMINAMATH_CALUDE_chess_team_selection_l4032_403211

theorem chess_team_selection (total_players : Nat) (quadruplets : Nat) (team_size : Nat) :
  total_players = 18 →
  quadruplets = 4 →
  team_size = 8 →
  Nat.choose (total_players - quadruplets) (team_size - quadruplets) = 1001 :=
by sorry

end NUMINAMATH_CALUDE_chess_team_selection_l4032_403211


namespace NUMINAMATH_CALUDE_parabola_symmetry_l4032_403261

/-- Given a parabola y = x^2 + 3x + m in the Cartesian coordinate system,
    prove that when translated 5 units to the right,
    the original and translated parabolas are symmetric about the line x = 1 -/
theorem parabola_symmetry (m : ℝ) :
  let f (x : ℝ) := x^2 + 3*x + m
  let g (x : ℝ) := f (x - 5)
  ∀ (x y : ℝ), f (1 - (x - 1)) = g (1 + (x - 1)) := by sorry

end NUMINAMATH_CALUDE_parabola_symmetry_l4032_403261


namespace NUMINAMATH_CALUDE_M_intersect_complement_N_l4032_403275

def R := ℝ

def M : Set ℝ := {x : ℝ | x^2 - 2*x < 0}

def N : Set ℝ := {x : ℝ | x ≥ 1}

theorem M_intersect_complement_N :
  M ∩ (Set.univ \ N) = Set.Ioo 0 1 := by
  sorry

end NUMINAMATH_CALUDE_M_intersect_complement_N_l4032_403275


namespace NUMINAMATH_CALUDE_player_jump_height_to_dunk_l4032_403233

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

end NUMINAMATH_CALUDE_player_jump_height_to_dunk_l4032_403233


namespace NUMINAMATH_CALUDE_contest_paths_count_l4032_403205

/-- Represents the grid structure for the word "CONTEST" --/
inductive ContestGrid
| C : ContestGrid
| O : ContestGrid → ContestGrid
| N : ContestGrid → ContestGrid
| T : ContestGrid → ContestGrid
| E : ContestGrid → ContestGrid
| S : ContestGrid → ContestGrid

/-- Counts the number of paths to form "CONTEST" in the given grid --/
def countContestPaths (grid : ContestGrid) : ℕ :=
  match grid with
  | ContestGrid.C => 1
  | ContestGrid.O g => 2 * countContestPaths g
  | ContestGrid.N g => 2 * countContestPaths g
  | ContestGrid.T g => 2 * countContestPaths g
  | ContestGrid.E g => 2 * countContestPaths g
  | ContestGrid.S g => 2 * countContestPaths g

/-- The contest grid structure --/
def contestGrid : ContestGrid :=
  ContestGrid.S (ContestGrid.E (ContestGrid.T (ContestGrid.N (ContestGrid.O (ContestGrid.C)))))

theorem contest_paths_count :
  countContestPaths contestGrid = 127 :=
sorry

end NUMINAMATH_CALUDE_contest_paths_count_l4032_403205


namespace NUMINAMATH_CALUDE_litter_patrol_problem_l4032_403285

/-- The Litter Patrol problem -/
theorem litter_patrol_problem (total_litter aluminum_cans : ℕ) 
  (h1 : total_litter = 18)
  (h2 : aluminum_cans = 8)
  (h3 : total_litter = aluminum_cans + glass_bottles) :
  glass_bottles = 10 := by
  sorry

end NUMINAMATH_CALUDE_litter_patrol_problem_l4032_403285


namespace NUMINAMATH_CALUDE_johnson_farm_wheat_acreage_l4032_403264

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

end NUMINAMATH_CALUDE_johnson_farm_wheat_acreage_l4032_403264


namespace NUMINAMATH_CALUDE_perfect_square_sum_l4032_403236

theorem perfect_square_sum (n : ℝ) (h : n > 2) :
  ∃ m : ℝ, ∃ k : ℝ, n^2 + m^2 = k^2 := by
  sorry

end NUMINAMATH_CALUDE_perfect_square_sum_l4032_403236


namespace NUMINAMATH_CALUDE_cloth_cost_l4032_403238

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

end NUMINAMATH_CALUDE_cloth_cost_l4032_403238


namespace NUMINAMATH_CALUDE_max_value_reciprocal_sum_l4032_403284

theorem max_value_reciprocal_sum (x y a b : ℝ) 
  (ha : a > 1) (hb : b > 1) 
  (hx : a^x = 3) (hy : b^y = 3) 
  (hab : a + b = 2 * Real.sqrt 3) : 
  (∀ x' y' : ℝ, a^x' = 3 → b^y' = 3 → 1/x' + 1/y' ≤ 1) ∧ 
  (∃ x'' y'' : ℝ, a^x'' = 3 ∧ b^y'' = 3 ∧ 1/x'' + 1/y'' = 1) :=
sorry

end NUMINAMATH_CALUDE_max_value_reciprocal_sum_l4032_403284


namespace NUMINAMATH_CALUDE_triangle_area_l4032_403247

/-- The area of a triangle with sides 5, 5, and 6 units is 12 square units. -/
theorem triangle_area : ∀ (a b c : ℝ), a = 5 ∧ b = 5 ∧ c = 6 →
  ∃ (s : ℝ), s = (a + b + c) / 2 ∧ Real.sqrt (s * (s - a) * (s - b) * (s - c)) = 12 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_l4032_403247


namespace NUMINAMATH_CALUDE_cricket_team_size_l4032_403225

theorem cricket_team_size :
  ∀ (n : ℕ),
  (n : ℝ) * 23 = (n - 2 : ℝ) * 22 + 55 →
  n = 11 := by
sorry

end NUMINAMATH_CALUDE_cricket_team_size_l4032_403225


namespace NUMINAMATH_CALUDE_firefighter_pay_theorem_l4032_403272

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

end NUMINAMATH_CALUDE_firefighter_pay_theorem_l4032_403272


namespace NUMINAMATH_CALUDE_union_equals_universal_l4032_403200

-- Define the universal set U
def U : Finset Nat := {2, 3, 4, 5, 6}

-- Define set M
def M : Finset Nat := {3, 4, 5}

-- Define set N
def N : Finset Nat := {2, 4, 6}

-- Theorem statement
theorem union_equals_universal : M ∪ N = U := by sorry

end NUMINAMATH_CALUDE_union_equals_universal_l4032_403200


namespace NUMINAMATH_CALUDE_fib_50_mod_5_l4032_403290

def fib : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | (n + 2) => fib (n + 1) + fib n

theorem fib_50_mod_5 : fib 50 % 5 = 0 := by
  sorry

end NUMINAMATH_CALUDE_fib_50_mod_5_l4032_403290


namespace NUMINAMATH_CALUDE_geometric_series_sum_l4032_403298

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

end NUMINAMATH_CALUDE_geometric_series_sum_l4032_403298


namespace NUMINAMATH_CALUDE_cosine_of_angle_between_vectors_l4032_403274

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

end NUMINAMATH_CALUDE_cosine_of_angle_between_vectors_l4032_403274


namespace NUMINAMATH_CALUDE_sixth_term_of_geometric_sequence_l4032_403256

def geometric_sequence (a : ℕ → ℚ) : Prop :=
  ∃ r : ℚ, ∀ n : ℕ, a (n + 1) = a n * r

theorem sixth_term_of_geometric_sequence
  (a : ℕ → ℚ)
  (h_geom : geometric_sequence a)
  (h_first : a 1 = 27)
  (h_fourth : a 4 = a 3 * a 5) :
  a 6 = 1 / 9 := by
sorry

end NUMINAMATH_CALUDE_sixth_term_of_geometric_sequence_l4032_403256


namespace NUMINAMATH_CALUDE_least_addition_for_divisibility_l4032_403266

theorem least_addition_for_divisibility : ∃! x : ℕ, 
  (x ≤ 22) ∧ (∃ k : ℤ, 1077 + x = 23 * k) ∧ 
  (∀ y : ℕ, y < x → ¬∃ k : ℤ, 1077 + y = 23 * k) :=
by sorry

end NUMINAMATH_CALUDE_least_addition_for_divisibility_l4032_403266


namespace NUMINAMATH_CALUDE_square_side_ratio_l4032_403248

theorem square_side_ratio (area_ratio : ℚ) :
  area_ratio = 45 / 64 →
  ∃ (a b c : ℕ), (a * Real.sqrt b) / c = Real.sqrt (area_ratio) ∧
                  a = 3 ∧ b = 5 ∧ c = 8 := by
  sorry

end NUMINAMATH_CALUDE_square_side_ratio_l4032_403248


namespace NUMINAMATH_CALUDE_cubic_inequality_solution_l4032_403237

theorem cubic_inequality_solution (x : ℝ) : 
  x^3 - x^2 + 11*x - 30 < 12 ↔ (x > -2 ∧ x < 3) ∨ (x > 3 ∧ x < 7) :=
sorry

end NUMINAMATH_CALUDE_cubic_inequality_solution_l4032_403237


namespace NUMINAMATH_CALUDE_repeating_decimal_to_fraction_l4032_403270

theorem repeating_decimal_to_fraction : 
  ∃ (x : ℚ), x = 57 / 99 ∧ (∀ n : ℕ, (x * 10^(2*n+2) - ⌊x * 10^(2*n+2)⌋ : ℚ) = 57 / 100) := by
  sorry

end NUMINAMATH_CALUDE_repeating_decimal_to_fraction_l4032_403270


namespace NUMINAMATH_CALUDE_final_value_is_four_l4032_403212

def program_execution (M : Nat) : Nat :=
  let M1 := M + 1
  let M2 := M1 + 2
  M2

theorem final_value_is_four : program_execution 1 = 4 := by
  sorry

end NUMINAMATH_CALUDE_final_value_is_four_l4032_403212


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l4032_403283

def A : Set ℝ := {x | x - 1 ≤ 0}
def B : Set ℝ := {0, 1, 2}

theorem intersection_of_A_and_B : A ∩ B = {0, 1} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l4032_403283


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l4032_403251

theorem sqrt_equation_solution :
  let x : ℝ := 49
  Real.sqrt x + Real.sqrt (x + 3) = 12 - Real.sqrt (x - 4) := by
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l4032_403251


namespace NUMINAMATH_CALUDE_number_problem_l4032_403224

theorem number_problem : ∃ x : ℚ, x - (3/5) * x = 58 ∧ x = 145 := by
  sorry

end NUMINAMATH_CALUDE_number_problem_l4032_403224


namespace NUMINAMATH_CALUDE_rectangle_max_area_l4032_403265

theorem rectangle_max_area :
  ∀ l w : ℕ,
  l + w = 22 →
  l * w ≤ 121 :=
by
  sorry

end NUMINAMATH_CALUDE_rectangle_max_area_l4032_403265


namespace NUMINAMATH_CALUDE_water_consumption_proof_l4032_403277

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

end NUMINAMATH_CALUDE_water_consumption_proof_l4032_403277


namespace NUMINAMATH_CALUDE_rationalization_factor_l4032_403249

theorem rationalization_factor (a b : ℝ) (ha : 0 ≤ a) (hb : 0 ≤ b) :
  (Real.sqrt a + Real.sqrt b) * (Real.sqrt a - Real.sqrt b) = a - b ∧
  (Real.sqrt a + Real.sqrt b) * (Real.sqrt b - Real.sqrt a) = b - a :=
by sorry

end NUMINAMATH_CALUDE_rationalization_factor_l4032_403249


namespace NUMINAMATH_CALUDE_pizza_toppings_combinations_l4032_403296

theorem pizza_toppings_combinations :
  Nat.choose 9 3 = 84 := by
  sorry

end NUMINAMATH_CALUDE_pizza_toppings_combinations_l4032_403296


namespace NUMINAMATH_CALUDE_not_always_geometric_b_l4032_403229

/-- A sequence is geometric if there exists a common ratio q such that a(n+1) = q * a(n) for all n -/
def IsGeometric (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = q * a n

/-- Definition of the sequence b_n in terms of a_n -/
def B (a : ℕ → ℝ) (n : ℕ) : ℝ := a (2*n - 1) + a (2*n)

theorem not_always_geometric_b (a : ℕ → ℝ) :
  IsGeometric a → ¬ (∀ a : ℕ → ℝ, IsGeometric a → IsGeometric (B a)) :=
by
  sorry

end NUMINAMATH_CALUDE_not_always_geometric_b_l4032_403229


namespace NUMINAMATH_CALUDE_complement_of_union_equals_singleton_l4032_403260

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

end NUMINAMATH_CALUDE_complement_of_union_equals_singleton_l4032_403260


namespace NUMINAMATH_CALUDE_line_passes_through_point_l4032_403252

/-- A line in the form y + 2 = k(x + 1) always passes through the point (-1, -2) -/
theorem line_passes_through_point (k : ℝ) : 
  let f : ℝ → ℝ := λ x ↦ k * (x + 1) - 2
  f (-1) = -2 := by
  sorry


end NUMINAMATH_CALUDE_line_passes_through_point_l4032_403252


namespace NUMINAMATH_CALUDE_part_one_part_two_l4032_403214

-- Define the function f
def f (x : ℝ) (a : ℝ) : ℝ := |x + 1| - a * |x - 1|

-- Part I
theorem part_one : 
  ∀ x : ℝ, (|x + 1| + 2 * |x - 1| > 5) ↔ (x < -4/3 ∨ x > 2) :=
sorry

-- Part II
theorem part_two :
  (∃ a : ℝ, ∀ x : ℝ, f x a ≤ a * |x + 3|) ∧
  (∀ b : ℝ, (∀ x : ℝ, f x b ≤ b * |x + 3|) → b ≥ 1/2) :=
sorry

end NUMINAMATH_CALUDE_part_one_part_two_l4032_403214


namespace NUMINAMATH_CALUDE_sequence_property_l4032_403278

def sequence_sum (a : ℕ+ → ℚ) (n : ℕ+) : ℚ :=
  (Finset.range n).sum (fun i => a ⟨i + 1, Nat.succ_pos i⟩)

theorem sequence_property (a : ℕ+ → ℚ) 
    (h : ∀ n : ℕ+, a n + sequence_sum a n = 2 * n.val + 1) :
  (∀ n : ℕ+, a n = 2 - 1 / (2 ^ n.val)) ∧
  (∀ n : ℕ+, (Finset.range n).sum (fun i => 1 / (2 ^ (i + 1) * a ⟨i + 1, Nat.succ_pos i⟩ * a ⟨i + 2, Nat.succ_pos (i + 1)⟩)) < 1 / 3) :=
by sorry

end NUMINAMATH_CALUDE_sequence_property_l4032_403278


namespace NUMINAMATH_CALUDE_closest_years_with_property_l4032_403250

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

end NUMINAMATH_CALUDE_closest_years_with_property_l4032_403250


namespace NUMINAMATH_CALUDE_congruence_theorem_l4032_403221

theorem congruence_theorem (n : ℕ+) :
  (122 ^ n.val - 102 ^ n.val - 21 ^ n.val) % 2020 = 2019 := by
  sorry

end NUMINAMATH_CALUDE_congruence_theorem_l4032_403221


namespace NUMINAMATH_CALUDE_goals_tied_in_june_l4032_403217

def ronaldo_goals : List Nat := [2, 9, 14, 8, 7, 11, 12]
def messi_goals : List Nat := [5, 8, 18, 6, 10, 9, 9]

def cumulative_sum (xs : List Nat) : List Nat :=
  List.scanl (·+·) 0 xs

def first_equal_index (xs ys : List Nat) : Option Nat :=
  (List.zip xs ys).findIdx (fun (x, y) => x = y)

def months : List String := ["January", "February", "March", "April", "May", "June", "July"]

theorem goals_tied_in_june :
  first_equal_index (cumulative_sum ronaldo_goals) (cumulative_sum messi_goals) = some 5 :=
by sorry

end NUMINAMATH_CALUDE_goals_tied_in_june_l4032_403217


namespace NUMINAMATH_CALUDE_equation_solution_l4032_403222

theorem equation_solution :
  let f (x : ℝ) := x^4 / (2*x + 1) + x^2 - 6*(2*x + 1)
  ∀ x : ℝ, f x = 0 ↔ x = -3 - Real.sqrt 6 ∨ x = -3 + Real.sqrt 6 ∨ x = 2 - Real.sqrt 6 ∨ x = 2 + Real.sqrt 6 :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l4032_403222


namespace NUMINAMATH_CALUDE_team_a_champion_probability_l4032_403240

/-- The probability of a team winning a single game -/
def win_prob : ℚ := 1/2

/-- The probability of Team A becoming the champion -/
def champion_prob : ℚ := win_prob + win_prob * win_prob

theorem team_a_champion_probability :
  champion_prob = 3/4 :=
by sorry

end NUMINAMATH_CALUDE_team_a_champion_probability_l4032_403240


namespace NUMINAMATH_CALUDE_geometric_sequence_problem_l4032_403216

theorem geometric_sequence_problem (a : ℕ → ℝ) (q : ℝ) :
  (∀ n : ℕ, a (n + 1) = a n * q) →  -- geometric sequence definition
  q > 0 →  -- positive common ratio
  a 3 * a 9 = 2 * (a 5)^2 →  -- given condition
  a 2 = 2 →  -- given condition
  a 1 = Real.sqrt 2 := by  -- conclusion to prove
sorry


end NUMINAMATH_CALUDE_geometric_sequence_problem_l4032_403216


namespace NUMINAMATH_CALUDE_is_reflection_l4032_403219

def reflection_matrix (a b : ℚ) : Matrix (Fin 2) (Fin 2) ℚ :=
  ![![a, b],
    ![3/7, -4/7]]

theorem is_reflection : 
  let R := reflection_matrix (4/7) (-16/21)
  R * R = 1 :=
sorry

end NUMINAMATH_CALUDE_is_reflection_l4032_403219


namespace NUMINAMATH_CALUDE_calories_burned_l4032_403228

/-- The number of times players run up and down the bleachers -/
def num_runs : ℕ := 40

/-- The number of stairs in one direction -/
def stairs_one_way : ℕ := 32

/-- The number of calories burned per stair -/
def calories_per_stair : ℕ := 2

/-- The total number of calories burned during the exercise -/
def total_calories : ℕ := num_runs * (2 * stairs_one_way) * calories_per_stair

theorem calories_burned :
  total_calories = 5120 :=
by sorry

end NUMINAMATH_CALUDE_calories_burned_l4032_403228


namespace NUMINAMATH_CALUDE_only_height_weight_correlated_l4032_403201

-- Define the concept of a variable pair
structure VariablePair where
  var1 : String
  var2 : String

-- Define the concept of a functional relationship
def functionalRelationship (pair : VariablePair) : Prop := sorry

-- Define the concept of correlation
def correlated (pair : VariablePair) : Prop := sorry

-- Define the given variable pairs
def taxiFareDistance : VariablePair := ⟨"taxi fare", "distance traveled"⟩
def houseSizePrice : VariablePair := ⟨"house size", "house price"⟩
def heightWeight : VariablePair := ⟨"human height", "human weight"⟩
def ironSizeMass : VariablePair := ⟨"iron block size", "iron block mass"⟩

-- State the theorem
theorem only_height_weight_correlated :
  functionalRelationship taxiFareDistance →
  functionalRelationship houseSizePrice →
  (correlated heightWeight ∧ ¬functionalRelationship heightWeight) →
  functionalRelationship ironSizeMass →
  (correlated heightWeight ∧
   ¬correlated taxiFareDistance ∧
   ¬correlated houseSizePrice ∧
   ¬correlated ironSizeMass) := by
  sorry

end NUMINAMATH_CALUDE_only_height_weight_correlated_l4032_403201


namespace NUMINAMATH_CALUDE_company_employees_l4032_403206

theorem company_employees (december_employees : ℕ) (percentage_increase : ℚ) 
  (h1 : december_employees = 987)
  (h2 : percentage_increase = 127 / 1000) : 
  ∃ january_employees : ℕ, 
    (january_employees : ℚ) * (1 + percentage_increase) = december_employees ∧ 
    january_employees = 875 := by
  sorry

end NUMINAMATH_CALUDE_company_employees_l4032_403206


namespace NUMINAMATH_CALUDE_y_satisfies_equation_l4032_403291

open Real

/-- The function y(x) -/
noncomputable def y (x : ℝ) : ℝ := 2 * (sin x / x) + cos x

/-- The differential equation -/
def differential_equation (y : ℝ → ℝ) (x : ℝ) : Prop :=
  x * sin x * deriv y x + (sin x - x * cos x) * y x = sin x * cos x - x

/-- Theorem stating that y satisfies the differential equation -/
theorem y_satisfies_equation : ∀ x : ℝ, x ≠ 0 → differential_equation y x := by
  sorry

end NUMINAMATH_CALUDE_y_satisfies_equation_l4032_403291


namespace NUMINAMATH_CALUDE_angle_opposite_geometric_mean_side_at_most_60_degrees_l4032_403246

/-- 
If in a triangle ABC, side a is the geometric mean of sides b and c,
then the angle A opposite to side a is less than or equal to 60°.
-/
theorem angle_opposite_geometric_mean_side_at_most_60_degrees 
  (a b c : ℝ) (h_positive : 0 < a ∧ 0 < b ∧ 0 < c) (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b)
  (h_geometric_mean : a^2 = b*c) : 
  let A := Real.arccos ((b^2 + c^2 - a^2) / (2*b*c))
  A ≤ π/3 := by sorry

end NUMINAMATH_CALUDE_angle_opposite_geometric_mean_side_at_most_60_degrees_l4032_403246


namespace NUMINAMATH_CALUDE_positive_numbers_inequality_l4032_403279

theorem positive_numbers_inequality (a b c : ℝ) 
  (h_pos_a : a > 0) (h_pos_b : b > 0) (h_pos_c : c > 0)
  (h_eq : 4 * a^2 + b^2 + 16 * c^2 = 1) : 
  (0 < a * b ∧ a * b < 1/4) ∧ 
  (1/a^2 + 1/b^2 + 1/(4*a*b*c^2) > 49) := by
  sorry

end NUMINAMATH_CALUDE_positive_numbers_inequality_l4032_403279


namespace NUMINAMATH_CALUDE_class_assignment_arrangements_l4032_403280

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

end NUMINAMATH_CALUDE_class_assignment_arrangements_l4032_403280


namespace NUMINAMATH_CALUDE_sports_club_overlap_l4032_403263

theorem sports_club_overlap (N B T X : ℕ) (h1 : N = 40) (h2 : B = 20) (h3 : T = 18) (h4 : X = 5) :
  B + T - (N - X) = 3 :=
by
  sorry

end NUMINAMATH_CALUDE_sports_club_overlap_l4032_403263


namespace NUMINAMATH_CALUDE_least_integer_square_condition_l4032_403208

theorem least_integer_square_condition : ∃ x : ℤ, x^2 = 3*x + 12 ∧ ∀ y : ℤ, y^2 = 3*y + 12 → x ≤ y :=
by sorry

end NUMINAMATH_CALUDE_least_integer_square_condition_l4032_403208


namespace NUMINAMATH_CALUDE_charles_chocolate_syrup_l4032_403276

/-- The amount of chocolate milk in each glass (in ounces) -/
def glass_size : ℝ := 8

/-- The amount of milk in each glass (in ounces) -/
def milk_per_glass : ℝ := 6.5

/-- The amount of chocolate syrup in each glass (in ounces) -/
def syrup_per_glass : ℝ := 1.5

/-- The total amount of milk Charles has (in ounces) -/
def total_milk : ℝ := 130

/-- The total amount of chocolate milk Charles will drink (in ounces) -/
def total_drink : ℝ := 160

/-- The theorem stating the amount of chocolate syrup Charles has -/
theorem charles_chocolate_syrup : 
  ∃ (syrup : ℝ), 
    (total_drink / glass_size) * syrup_per_glass = syrup ∧ 
    syrup = 30 := by sorry

end NUMINAMATH_CALUDE_charles_chocolate_syrup_l4032_403276


namespace NUMINAMATH_CALUDE_circles_intersect_l4032_403203

-- Define the circles
def C1 (x y : ℝ) : Prop := (x + 2)^2 + (y - 2)^2 = 4
def C2 (x y : ℝ) : Prop := (x - 2)^2 + (y - 5)^2 = 16

-- Define the centers and radii
def center1 : ℝ × ℝ := (-2, 2)
def center2 : ℝ × ℝ := (2, 5)
def radius1 : ℝ := 2
def radius2 : ℝ := 4

-- Theorem statement
theorem circles_intersect :
  let d := Real.sqrt ((center1.1 - center2.1)^2 + (center1.2 - center2.2)^2)
  radius1 + radius2 > d ∧ d > abs (radius1 - radius2) := by sorry

end NUMINAMATH_CALUDE_circles_intersect_l4032_403203


namespace NUMINAMATH_CALUDE_distance_traveled_l4032_403220

/-- Given a speed of 25 km/hr and a time of 5 hr, the distance traveled is 125 km. -/
theorem distance_traveled (speed : ℝ) (time : ℝ) (distance : ℝ) 
  (h1 : speed = 25) 
  (h2 : time = 5) 
  (h3 : distance = speed * time) : 
  distance = 125 := by
  sorry

end NUMINAMATH_CALUDE_distance_traveled_l4032_403220


namespace NUMINAMATH_CALUDE_quarter_count_in_collection_l4032_403288

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

end NUMINAMATH_CALUDE_quarter_count_in_collection_l4032_403288


namespace NUMINAMATH_CALUDE_solve_grocery_cost_l4032_403242

def grocery_cost_problem (initial_amount : ℝ) (sister_fraction : ℝ) (remaining_amount : ℝ) : Prop :=
  let amount_to_sister := initial_amount * sister_fraction
  let amount_after_giving := initial_amount - amount_to_sister
  let grocery_cost := amount_after_giving - remaining_amount
  grocery_cost = 40

theorem solve_grocery_cost :
  grocery_cost_problem 100 (1/4) 35 := by
  sorry

end NUMINAMATH_CALUDE_solve_grocery_cost_l4032_403242


namespace NUMINAMATH_CALUDE_equivalence_condition_l4032_403282

theorem equivalence_condition (a : ℝ) : 
  (∀ x : ℝ, (5 - x) / (x - 2) ≥ 0 ↔ -3 < x ∧ x < a) ↔ a > 5 :=
by sorry

end NUMINAMATH_CALUDE_equivalence_condition_l4032_403282


namespace NUMINAMATH_CALUDE_investment_split_l4032_403258

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

end NUMINAMATH_CALUDE_investment_split_l4032_403258


namespace NUMINAMATH_CALUDE_gondor_wednesday_laptops_l4032_403223

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

end NUMINAMATH_CALUDE_gondor_wednesday_laptops_l4032_403223


namespace NUMINAMATH_CALUDE_min_lines_is_seven_l4032_403230

/-- A line in a 2D Cartesian coordinate system -/
structure Line where
  k : ℝ
  b : ℝ
  k_nonzero : k ≠ 0

/-- The set of quadrants a line passes through -/
def quadrants (l : Line) : Set (Fin 4) :=
  sorry

/-- The minimum number of lines needed to guarantee two lines pass through the same quadrants -/
def min_lines_same_quadrants : ℕ :=
  sorry

/-- Theorem stating that the minimum number of lines is 7 -/
theorem min_lines_is_seven : min_lines_same_quadrants = 7 := by
  sorry

end NUMINAMATH_CALUDE_min_lines_is_seven_l4032_403230


namespace NUMINAMATH_CALUDE_hours_difference_l4032_403243

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


end NUMINAMATH_CALUDE_hours_difference_l4032_403243


namespace NUMINAMATH_CALUDE_xiao_ming_score_l4032_403226

/-- Calculates the comprehensive score based on individual scores and weights -/
def comprehensive_score (written_score practical_score publicity_score : ℝ)
  (written_weight practical_weight publicity_weight : ℝ) : ℝ :=
  written_score * written_weight + practical_score * practical_weight + publicity_score * publicity_weight

/-- Theorem stating that Xiao Ming's comprehensive score is 97 -/
theorem xiao_ming_score :
  let written_score : ℝ := 96
  let practical_score : ℝ := 98
  let publicity_score : ℝ := 96
  let written_weight : ℝ := 0.30
  let practical_weight : ℝ := 0.50
  let publicity_weight : ℝ := 0.20
  comprehensive_score written_score practical_score publicity_score
    written_weight practical_weight publicity_weight = 97 :=
by sorry


end NUMINAMATH_CALUDE_xiao_ming_score_l4032_403226


namespace NUMINAMATH_CALUDE_max_value_symmetric_function_l4032_403286

def f (a b x : ℝ) : ℝ := (1 + 2*x) * (x^2 + a*x + b)

theorem max_value_symmetric_function (a b : ℝ) :
  (∀ x : ℝ, f a b (1 - x) = f a b (1 + x)) →
  (∃ x₀ : ℝ, x₀ ∈ Set.Icc (-1 : ℝ) 1 ∧
    ∀ x : ℝ, x ∈ Set.Icc (-1 : ℝ) 1 → f a b x ≤ f a b x₀) →
  (∃ x₀ : ℝ, x₀ ∈ Set.Icc (-1 : ℝ) 1 ∧
    f a b x₀ = 3 * Real.sqrt 3 / 2 ∧
    ∀ x : ℝ, x ∈ Set.Icc (-1 : ℝ) 1 → f a b x ≤ f a b x₀) :=
by sorry

end NUMINAMATH_CALUDE_max_value_symmetric_function_l4032_403286


namespace NUMINAMATH_CALUDE_base_eight_addition_sum_l4032_403234

/-- Given distinct non-zero digits S, H, and E less than 8 that satisfy the base-8 addition
    SEH₈ + EHS₈ = SHE₈, prove that their sum in base 10 is 6. -/
theorem base_eight_addition_sum (S H E : ℕ) : 
  S ≠ 0 → H ≠ 0 → E ≠ 0 →
  S < 8 → H < 8 → E < 8 →
  S ≠ H → S ≠ E → H ≠ E →
  S * 64 + E * 8 + H + E * 64 + H * 8 + S = S * 64 + H * 8 + E →
  S + H + E = 6 := by
  sorry

end NUMINAMATH_CALUDE_base_eight_addition_sum_l4032_403234


namespace NUMINAMATH_CALUDE_modulo_congruence_unique_solution_l4032_403213

theorem modulo_congruence_unique_solution : ∃! n : ℤ, 0 ≤ n ∧ n < 25 ∧ -250 ≡ n [ZMOD 23] ∧ n = 3 := by
  sorry

end NUMINAMATH_CALUDE_modulo_congruence_unique_solution_l4032_403213


namespace NUMINAMATH_CALUDE_remainder_problem_l4032_403209

theorem remainder_problem (x : ℤ) : x % 63 = 11 → x % 9 = 2 := by
  sorry

end NUMINAMATH_CALUDE_remainder_problem_l4032_403209


namespace NUMINAMATH_CALUDE_right_triangle_leg_lengths_l4032_403273

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

end NUMINAMATH_CALUDE_right_triangle_leg_lengths_l4032_403273


namespace NUMINAMATH_CALUDE_drivers_distance_difference_l4032_403210

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

end NUMINAMATH_CALUDE_drivers_distance_difference_l4032_403210


namespace NUMINAMATH_CALUDE_no_rain_probability_l4032_403299

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

end NUMINAMATH_CALUDE_no_rain_probability_l4032_403299


namespace NUMINAMATH_CALUDE_complex_equation_result_l4032_403269

theorem complex_equation_result (x y : ℂ) (hx : x ≠ 0) (hy : y ≠ 0) 
  (h : x^2 + x*y + y^2 = 0) : 
  (x / (x + y))^1990 + (y / (x + y))^1990 = -1 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_result_l4032_403269


namespace NUMINAMATH_CALUDE_unique_420_sequence_l4032_403207

-- Define the sum of consecutive integers
def sum_consecutive (n : ℕ) (k : ℕ) : ℕ := k * n + k * (k - 1) / 2

-- Define a predicate for valid sequences
def valid_sequence (n : ℕ) (k : ℕ) : Prop :=
  k ≥ 2 ∧ (k % 2 = 0 ∨ k = 3) ∧ sum_consecutive n k = 420

-- The main theorem
theorem unique_420_sequence :
  ∃! (n k : ℕ), valid_sequence n k :=
sorry

end NUMINAMATH_CALUDE_unique_420_sequence_l4032_403207


namespace NUMINAMATH_CALUDE_arithmetic_sequence_properties_l4032_403218

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

end NUMINAMATH_CALUDE_arithmetic_sequence_properties_l4032_403218


namespace NUMINAMATH_CALUDE_no_real_solutions_for_arithmetic_progression_l4032_403281

-- Define the property of being an arithmetic progression
def is_arithmetic_progression (x y z w : ℝ) : Prop :=
  y - x = z - y ∧ z - y = w - z

-- Theorem statement
theorem no_real_solutions_for_arithmetic_progression :
  ¬ ∃ (a b : ℝ), is_arithmetic_progression 15 a b (a * b) := by
  sorry

end NUMINAMATH_CALUDE_no_real_solutions_for_arithmetic_progression_l4032_403281


namespace NUMINAMATH_CALUDE_train_crossing_cars_l4032_403215

/-- Represents the properties of a train passing through a crossing -/
structure TrainCrossing where
  cars_in_sample : ℕ
  sample_time : ℕ
  total_time : ℕ

/-- Calculates the number of cars in the train, rounded to the nearest multiple of 10 -/
def cars_in_train (tc : TrainCrossing) : ℕ :=
  let rate := tc.cars_in_sample / tc.sample_time
  let total_cars := rate * tc.total_time
  ((total_cars + 5) / 10) * 10

/-- Theorem stating that for the given train crossing scenario, the number of cars is 120 -/
theorem train_crossing_cars :
  let tc : TrainCrossing := { cars_in_sample := 9, sample_time := 15, total_time := 210 }
  cars_in_train tc = 120 := by
  sorry

end NUMINAMATH_CALUDE_train_crossing_cars_l4032_403215


namespace NUMINAMATH_CALUDE_tangent_line_intersection_l4032_403241

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

end NUMINAMATH_CALUDE_tangent_line_intersection_l4032_403241


namespace NUMINAMATH_CALUDE_smallest_four_digit_congruence_l4032_403204

theorem smallest_four_digit_congruence :
  ∃ (n : ℕ), 
    (1000 ≤ n ∧ n < 10000) ∧ 
    (75 * n) % 450 = 225 ∧
    (∀ m, (1000 ≤ m ∧ m < 10000) → (75 * m) % 450 = 225 → n ≤ m) ∧
    n = 1005 :=
by sorry

end NUMINAMATH_CALUDE_smallest_four_digit_congruence_l4032_403204


namespace NUMINAMATH_CALUDE_remaining_work_days_l4032_403268

/-- Given workers x and y, where x can finish a job in 24 days and y in 16 days,
    prove that x needs 9 days to finish the remaining work after y works for 10 days. -/
theorem remaining_work_days (x_days y_days y_worked_days : ℕ) 
  (hx : x_days = 24)
  (hy : y_days = 16)
  (hw : y_worked_days = 10) :
  (x_days : ℚ) * (1 - y_worked_days / y_days) = 9 := by
  sorry

end NUMINAMATH_CALUDE_remaining_work_days_l4032_403268


namespace NUMINAMATH_CALUDE_no_solutions_absolute_value_equation_l4032_403294

theorem no_solutions_absolute_value_equation :
  ¬ ∃ x : ℝ, |x - 2| = |x - 1| + |x - 4| := by
sorry

end NUMINAMATH_CALUDE_no_solutions_absolute_value_equation_l4032_403294


namespace NUMINAMATH_CALUDE_fibonacci_like_sequence_l4032_403239

theorem fibonacci_like_sequence (α β : ℝ) (h1 : α^2 - α - 1 = 0) (h2 : β^2 - β - 1 = 0) (h3 : α > β) :
  let s : ℕ → ℝ := λ n => α^n + β^n
  ∀ n ≥ 3, s n = s (n-1) + s (n-2) :=
by sorry

end NUMINAMATH_CALUDE_fibonacci_like_sequence_l4032_403239


namespace NUMINAMATH_CALUDE_x_value_proof_l4032_403202

theorem x_value_proof (y : ℝ) (x : ℝ) (h1 : y = -2) (h2 : (x - 2*y)^y = 0.001) :
  x = -4 + 10 * Real.sqrt 10 :=
sorry

end NUMINAMATH_CALUDE_x_value_proof_l4032_403202


namespace NUMINAMATH_CALUDE_triangle_max_side_sum_l4032_403295

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

end NUMINAMATH_CALUDE_triangle_max_side_sum_l4032_403295


namespace NUMINAMATH_CALUDE_odd_functions_max_min_l4032_403231

-- Define the property of being an odd function
def IsOdd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

-- Define the function F
def F (f g : ℝ → ℝ) (x : ℝ) : ℝ := f x + g x + 2

-- State the theorem
theorem odd_functions_max_min (f g : ℝ → ℝ) (hf : IsOdd f) (hg : IsOdd g) 
  (hmax : ∃ M, M = 8 ∧ ∀ x > 0, F f g x ≤ M) :
  ∃ m, m = -4 ∧ ∀ x < 0, F f g x ≥ m :=
sorry

end NUMINAMATH_CALUDE_odd_functions_max_min_l4032_403231


namespace NUMINAMATH_CALUDE_tangent_parallel_points_tangent_points_on_curve_unique_tangent_points_l4032_403267

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

end NUMINAMATH_CALUDE_tangent_parallel_points_tangent_points_on_curve_unique_tangent_points_l4032_403267


namespace NUMINAMATH_CALUDE_broken_line_circle_cover_l4032_403253

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

end NUMINAMATH_CALUDE_broken_line_circle_cover_l4032_403253


namespace NUMINAMATH_CALUDE_division_of_A_by_1001_l4032_403232

/-- A number consisting of 1001 sevens -/
def A : ℕ := (10 ^ 1001 - 1) / 9 * 7

/-- The expected quotient when A is divided by 1001 -/
def expected_quotient : ℕ := (10 ^ 1001 - 1) / (9 * 1001) * 777

/-- The expected remainder when A is divided by 1001 -/
def expected_remainder : ℕ := 700

theorem division_of_A_by_1001 :
  (A / 1001 = expected_quotient) ∧ (A % 1001 = expected_remainder) :=
sorry

end NUMINAMATH_CALUDE_division_of_A_by_1001_l4032_403232


namespace NUMINAMATH_CALUDE_perpendicular_lines_slope_l4032_403255

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

end NUMINAMATH_CALUDE_perpendicular_lines_slope_l4032_403255


namespace NUMINAMATH_CALUDE_even_function_extension_l4032_403289

/-- A function f : ℝ → ℝ is even if f(-x) = f(x) for all x -/
def IsEven (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

theorem even_function_extension (f : ℝ → ℝ) (h_even : IsEven f) 
  (h_def : ∀ x > 0, f x = x * (1 + x)) :
  ∀ x < 0, f x = x * (x - 1) := by
  sorry

end NUMINAMATH_CALUDE_even_function_extension_l4032_403289


namespace NUMINAMATH_CALUDE_least_product_of_distinct_primes_l4032_403227

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d ∣ n → d = 1 ∨ d = n

def distinct_primes_product (p q : ℕ) : Prop :=
  is_prime p ∧ is_prime q ∧ p ≠ q ∧ 50 < p ∧ p < 100 ∧ 50 < q ∧ q < 100

theorem least_product_of_distinct_primes :
  ∃ p q : ℕ, distinct_primes_product p q ∧
    p * q = 3127 ∧
    ∀ r s : ℕ, distinct_primes_product r s → p * q ≤ r * s :=
sorry

end NUMINAMATH_CALUDE_least_product_of_distinct_primes_l4032_403227


namespace NUMINAMATH_CALUDE_total_animals_l4032_403287

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

end NUMINAMATH_CALUDE_total_animals_l4032_403287


namespace NUMINAMATH_CALUDE_fraction_of_25_l4032_403262

theorem fraction_of_25 : 
  ∃ x : ℚ, x * 25 = (80 / 100) * 40 - 22 ∧ x = 2 / 5 := by
  sorry

end NUMINAMATH_CALUDE_fraction_of_25_l4032_403262


namespace NUMINAMATH_CALUDE_benny_eggs_dozens_l4032_403235

/-- The number of eggs in a dozen -/
def eggs_per_dozen : ℕ := 12

/-- The total number of eggs Benny bought -/
def total_eggs : ℕ := 84

/-- The number of dozens of eggs Benny bought -/
def dozens_bought : ℕ := total_eggs / eggs_per_dozen

theorem benny_eggs_dozens : dozens_bought = 7 := by
  sorry

end NUMINAMATH_CALUDE_benny_eggs_dozens_l4032_403235


namespace NUMINAMATH_CALUDE_ayen_extra_minutes_friday_l4032_403293

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

end NUMINAMATH_CALUDE_ayen_extra_minutes_friday_l4032_403293


namespace NUMINAMATH_CALUDE_system_equation_range_l4032_403254

theorem system_equation_range (x y m : ℝ) : 
  x + 2*y = 1 + m → 
  2*x + y = -3 → 
  x + y > 0 → 
  m > 2 := by
sorry

end NUMINAMATH_CALUDE_system_equation_range_l4032_403254


namespace NUMINAMATH_CALUDE_area_calculation_l4032_403245

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

end NUMINAMATH_CALUDE_area_calculation_l4032_403245


namespace NUMINAMATH_CALUDE_messaging_packages_theorem_l4032_403244

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

end NUMINAMATH_CALUDE_messaging_packages_theorem_l4032_403244


namespace NUMINAMATH_CALUDE_rhombus_area_l4032_403257

/-- The area of a rhombus with diagonals of 15 cm and 21 cm is 157.5 cm². -/
theorem rhombus_area (d1 d2 area : ℝ) (h1 : d1 = 15) (h2 : d2 = 21) 
    (h3 : area = (d1 * d2) / 2) : area = 157.5 := by
  sorry

end NUMINAMATH_CALUDE_rhombus_area_l4032_403257
