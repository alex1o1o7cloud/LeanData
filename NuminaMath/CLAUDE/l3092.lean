import Mathlib

namespace NUMINAMATH_CALUDE_set_of_possible_a_l3092_309258

def M : Set ℝ := {x | x^2 - 3*x + 2 = 0}
def N (a : ℝ) : Set ℝ := {x | x^2 - a*x + 3*a - 5 = 0}

theorem set_of_possible_a (a : ℝ) : M ∪ N a = M → 2 ≤ a ∧ a < 10 := by
  sorry

end NUMINAMATH_CALUDE_set_of_possible_a_l3092_309258


namespace NUMINAMATH_CALUDE_measure_8_liters_possible_min_operations_is_30_l3092_309297

/-- Represents the state of the two vessels --/
structure VesselState :=
  (vessel15 : ℕ)
  (vessel16 : ℕ)

/-- Represents an operation on the vessels --/
inductive Operation
  | Fill15
  | Fill16
  | Empty15
  | Empty16
  | Pour15To16
  | Pour16To15

/-- Applies an operation to a vessel state --/
def applyOperation (state : VesselState) (op : Operation) : VesselState :=
  match op with
  | Operation.Fill15 => ⟨15, state.vessel16⟩
  | Operation.Fill16 => ⟨state.vessel15, 16⟩
  | Operation.Empty15 => ⟨0, state.vessel16⟩
  | Operation.Empty16 => ⟨state.vessel15, 0⟩
  | Operation.Pour15To16 => 
      let amount := min state.vessel15 (16 - state.vessel16)
      ⟨state.vessel15 - amount, state.vessel16 + amount⟩
  | Operation.Pour16To15 => 
      let amount := min state.vessel16 (15 - state.vessel15)
      ⟨state.vessel15 + amount, state.vessel16 - amount⟩

/-- Checks if a sequence of operations results in 8 liters in either vessel --/
def achieves8Liters (ops : List Operation) : Prop :=
  let finalState := ops.foldl applyOperation ⟨0, 0⟩
  finalState.vessel15 = 8 ∨ finalState.vessel16 = 8

/-- The main theorem stating that it's possible to measure 8 liters --/
theorem measure_8_liters_possible : ∃ (ops : List Operation), achieves8Liters ops :=
  sorry

/-- The theorem stating that the minimum number of operations is 30 --/
theorem min_operations_is_30 : 
  (∃ (ops : List Operation), achieves8Liters ops ∧ ops.length = 30) ∧
  (∀ (ops : List Operation), achieves8Liters ops → ops.length ≥ 30) :=
  sorry

end NUMINAMATH_CALUDE_measure_8_liters_possible_min_operations_is_30_l3092_309297


namespace NUMINAMATH_CALUDE_total_donation_is_65_inches_l3092_309251

/-- Represents the hair donation of a person -/
structure HairDonation where
  initialLength : ℕ
  keptLength : ℕ
  donatedLength : ℕ
  donation_calculation : donatedLength = initialLength - keptLength

/-- The total hair donation of the five friends -/
def totalDonation (isabella damien ella toby lisa : HairDonation) : ℕ :=
  isabella.donatedLength + damien.donatedLength + ella.donatedLength + toby.donatedLength + lisa.donatedLength

/-- Theorem stating the total hair donation is 65 inches -/
theorem total_donation_is_65_inches : 
  ∃ (isabella damien ella toby lisa : HairDonation),
    isabella.initialLength = 18 ∧ isabella.keptLength = 9 ∧
    damien.initialLength = 24 ∧ damien.keptLength = 12 ∧
    ella.initialLength = 30 ∧ ella.keptLength = 10 ∧
    toby.initialLength = 16 ∧ toby.keptLength = 0 ∧
    lisa.initialLength = 28 ∧ lisa.donatedLength = 8 ∧
    totalDonation isabella damien ella toby lisa = 65 := by
  sorry

end NUMINAMATH_CALUDE_total_donation_is_65_inches_l3092_309251


namespace NUMINAMATH_CALUDE_intersection_distance_and_difference_l3092_309296

def f (x : ℝ) := 5 * x^2 + 3 * x - 2

theorem intersection_distance_and_difference :
  ∃ (C D : ℝ × ℝ),
    (f C.1 = 4 ∧ C.2 = 4) ∧
    (f D.1 = 4 ∧ D.2 = 4) ∧
    C ≠ D ∧
    ∃ (p q : ℕ),
      p = 129 ∧
      q = 5 ∧
      (C.1 - D.1)^2 + (C.2 - D.2)^2 = (Real.sqrt p / q)^2 ∧
      p - q = 124 := by
  sorry

end NUMINAMATH_CALUDE_intersection_distance_and_difference_l3092_309296


namespace NUMINAMATH_CALUDE_sum_of_bases_equals_1135_l3092_309290

/-- Converts a number from base 9 to base 10 -/
def base9ToBase10 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (9 ^ i)) 0

/-- Converts a number from base 13 to base 10 -/
def base13ToBase10 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (13 ^ i)) 0

/-- The value of digit C in base 13 -/
def C : Nat := 12

/-- The theorem to prove -/
theorem sum_of_bases_equals_1135 :
  base9ToBase10 [1, 6, 3] + base13ToBase10 [5, C, 4] = 1135 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_bases_equals_1135_l3092_309290


namespace NUMINAMATH_CALUDE_rectangle_thirteen_squares_l3092_309277

/-- A rectangle can be divided into 13 equal squares if and only if its side length ratio is 13:1 -/
theorem rectangle_thirteen_squares (a b : ℕ) (h : a > 0 ∧ b > 0) :
  (∃ (s : ℕ), s > 0 ∧ a * b = 13 * s * s) ↔ (a = 13 * b ∨ b = 13 * a) :=
sorry

end NUMINAMATH_CALUDE_rectangle_thirteen_squares_l3092_309277


namespace NUMINAMATH_CALUDE_two_solutions_l3092_309287

/-- A solution to the system of equations is a triple of positive integers (x, y, z) 
    satisfying the given conditions. -/
def IsSolution (x y z : ℕ+) : Prop :=
  x * y + y * z = 63 ∧ x * z + y * z = 23

/-- The theorem states that there are exactly two solutions to the system of equations. -/
theorem two_solutions : 
  ∃! (s : Finset (ℕ+ × ℕ+ × ℕ+)), 
    (∀ (x y z : ℕ+), (x, y, z) ∈ s ↔ IsSolution x y z) ∧ 
    Finset.card s = 2 := by
  sorry

end NUMINAMATH_CALUDE_two_solutions_l3092_309287


namespace NUMINAMATH_CALUDE_solve_potatoes_problem_l3092_309237

def potatoes_problem (total : ℕ) (gina : ℕ) : Prop :=
  let tom := 2 * gina
  let anne := tom / 3
  let remaining := total - (gina + tom + anne)
  remaining = 47

theorem solve_potatoes_problem :
  potatoes_problem 300 69 := by
  sorry

end NUMINAMATH_CALUDE_solve_potatoes_problem_l3092_309237


namespace NUMINAMATH_CALUDE_chemistry_physics_difference_l3092_309202

/-- Represents the scores of a student in three subjects -/
structure Scores where
  math : ℕ
  physics : ℕ
  chemistry : ℕ

/-- The conditions of the problem -/
def satisfies_conditions (s : Scores) : Prop :=
  s.math + s.physics = 30 ∧
  s.chemistry > s.physics ∧
  (s.math + s.chemistry) / 2 = 25

/-- The theorem to be proved -/
theorem chemistry_physics_difference (s : Scores) :
  satisfies_conditions s → s.chemistry - s.physics = 20 := by
  sorry


end NUMINAMATH_CALUDE_chemistry_physics_difference_l3092_309202


namespace NUMINAMATH_CALUDE_max_abs_z_plus_4_l3092_309274

theorem max_abs_z_plus_4 (z : ℂ) (h : Complex.abs (z + 3 * Complex.I) = 5) :
  ∃ (max_val : ℝ), max_val = 10 ∧ ∀ (w : ℂ), Complex.abs (w + 3 * Complex.I) = 5 → Complex.abs (w + 4) ≤ max_val :=
sorry

end NUMINAMATH_CALUDE_max_abs_z_plus_4_l3092_309274


namespace NUMINAMATH_CALUDE_zoes_overall_accuracy_l3092_309245

/-- Represents the problem of calculating Zoe's overall accuracy rate -/
theorem zoes_overall_accuracy 
  (x : ℝ) -- Total number of problems
  (h_positive : x > 0) -- Ensure x is positive
  (h_chloe_indep : ℝ) -- Chloe's independent accuracy rate
  (h_chloe_indep_val : h_chloe_indep = 0.8) -- Chloe's independent accuracy is 80%
  (h_overall : ℝ) -- Overall accuracy rate for all problems
  (h_overall_val : h_overall = 0.88) -- Overall accuracy is 88%
  (h_zoe_indep : ℝ) -- Zoe's independent accuracy rate
  (h_zoe_indep_val : h_zoe_indep = 0.9) -- Zoe's independent accuracy is 90%
  : ∃ (y : ℝ), -- y represents the accuracy rate of problems solved together
    (0.5 * x * h_chloe_indep + 0.5 * x * y) / x = h_overall ∧ 
    (0.5 * x * h_zoe_indep + 0.5 * x * y) / x = 0.93 := by
  sorry

end NUMINAMATH_CALUDE_zoes_overall_accuracy_l3092_309245


namespace NUMINAMATH_CALUDE_sum_distinct_prime_factors_156000_l3092_309284

def sum_of_distinct_prime_factors (n : ℕ) : ℕ :=
  (Nat.factors n).toFinset.sum id

theorem sum_distinct_prime_factors_156000 :
  sum_of_distinct_prime_factors 156000 = 23 := by
  sorry

end NUMINAMATH_CALUDE_sum_distinct_prime_factors_156000_l3092_309284


namespace NUMINAMATH_CALUDE_reciprocal_and_square_properties_l3092_309208

theorem reciprocal_and_square_properties : 
  (∀ x : ℝ, x ≠ 0 → (x = 1/x ↔ x = 1 ∨ x = -1)) ∧ 
  (∀ x : ℝ, x = x^2 ↔ x = 0 ∨ x = 1) := by
  sorry

end NUMINAMATH_CALUDE_reciprocal_and_square_properties_l3092_309208


namespace NUMINAMATH_CALUDE_no_square_divisible_by_six_in_range_l3092_309256

theorem no_square_divisible_by_six_in_range : ¬∃ y : ℕ, 
  (∃ n : ℕ, y = n^2) ∧ 
  (y % 6 = 0) ∧ 
  (50 ≤ y) ∧ 
  (y ≤ 120) := by
  sorry

end NUMINAMATH_CALUDE_no_square_divisible_by_six_in_range_l3092_309256


namespace NUMINAMATH_CALUDE_impossible_arrangement_l3092_309224

theorem impossible_arrangement : ¬ ∃ (A B C : ℕ), 
  (A + B = 45) ∧ 
  (3 * A + B = 6 * C) ∧ 
  (A ≥ 0) ∧ (B ≥ 0) ∧ (C > 0) :=
by sorry

end NUMINAMATH_CALUDE_impossible_arrangement_l3092_309224


namespace NUMINAMATH_CALUDE_decagon_diagonal_intersections_l3092_309259

/-- The number of vertices in a regular decagon -/
def n : ℕ := 10

/-- The number of distinct interior intersection points of diagonals in a regular decagon -/
def intersection_points : ℕ := Nat.choose n 4

theorem decagon_diagonal_intersections :
  intersection_points = 210 :=
sorry

end NUMINAMATH_CALUDE_decagon_diagonal_intersections_l3092_309259


namespace NUMINAMATH_CALUDE_equation_solutions_l3092_309225

theorem equation_solutions :
  (∀ x : ℝ, (x + 1)^2 = 4 ↔ x = 1 ∨ x = -3) ∧
  (∀ x : ℝ, 3 * x^2 - 1 = 2 * x ↔ x = 1 ∨ x = -1/3) := by
  sorry

end NUMINAMATH_CALUDE_equation_solutions_l3092_309225


namespace NUMINAMATH_CALUDE_unique_multiple_of_6_l3092_309244

def is_multiple_of_6 (n : ℕ) : Prop := ∃ k : ℕ, n = 6 * k

def last_digit (n : ℕ) : ℕ := n % 10

def sum_of_digits (n : ℕ) : ℕ :=
  let digits := n.digits 10
  digits.sum

theorem unique_multiple_of_6 :
  ∀ n : ℕ, 63470 ≤ n ∧ n ≤ 63479 →
    (is_multiple_of_6 n ↔ n = 63474) :=
by sorry

end NUMINAMATH_CALUDE_unique_multiple_of_6_l3092_309244


namespace NUMINAMATH_CALUDE_domain_log_range_exp_intersection_empty_l3092_309260

-- Define the sets A and B
def A : Set ℝ := {x : ℝ | x < 0}
def B : Set ℝ := {y : ℝ | y > 0}

-- State the theorem
theorem domain_log_range_exp_intersection_empty : A ∩ B = ∅ := by
  sorry

end NUMINAMATH_CALUDE_domain_log_range_exp_intersection_empty_l3092_309260


namespace NUMINAMATH_CALUDE_laundry_time_ratio_l3092_309214

/-- Proves that the ratio of time to wash towels to time to wash clothes is 2:1 --/
theorem laundry_time_ratio :
  ∀ (towel_time sheet_time clothes_time : ℕ),
    clothes_time = 30 →
    sheet_time = towel_time - 15 →
    towel_time + sheet_time + clothes_time = 135 →
    towel_time / clothes_time = 2 :=
by
  sorry


end NUMINAMATH_CALUDE_laundry_time_ratio_l3092_309214


namespace NUMINAMATH_CALUDE_calories_in_pound_of_fat_l3092_309215

/-- Represents the number of calories in a pound of body fat -/
def calories_per_pound : ℝ := 3500

/-- Represents the number of calories burned per day through light exercise -/
def calories_burned_per_day : ℝ := 2500

/-- Represents the number of calories consumed per day -/
def calories_consumed_per_day : ℝ := 2000

/-- Represents the number of days it takes to lose the weight -/
def days_to_lose_weight : ℝ := 35

/-- Represents the number of pounds lost -/
def pounds_lost : ℝ := 5

theorem calories_in_pound_of_fat :
  calories_per_pound = 
    ((calories_burned_per_day - calories_consumed_per_day) * days_to_lose_weight) / pounds_lost :=
by sorry

end NUMINAMATH_CALUDE_calories_in_pound_of_fat_l3092_309215


namespace NUMINAMATH_CALUDE_eccentricity_relation_l3092_309293

/-- Given an ellipse with eccentricity e₁ and a hyperbola with eccentricity e₂,
    both sharing common foci F₁ and F₂, and a common point P such that
    the vectors PF₁ and PF₂ are perpendicular, prove that
    (e₁² + e₂²) / (e₁e₂)² = 2 -/
theorem eccentricity_relation (e₁ e₂ : ℝ) (F₁ F₂ P : ℝ × ℝ) :
  e₁ > 0 ∧ e₁ < 1 →  -- Condition for ellipse eccentricity
  e₂ > 1 →  -- Condition for hyperbola eccentricity
  (P.1 - F₁.1) * (P.1 - F₂.1) + (P.2 - F₁.2) * (P.2 - F₂.2) = 0 →  -- Perpendicularity condition
  (e₁^2 + e₂^2) / (e₁ * e₂)^2 = 2 := by
  sorry

end NUMINAMATH_CALUDE_eccentricity_relation_l3092_309293


namespace NUMINAMATH_CALUDE_unique_triple_divisibility_l3092_309204

theorem unique_triple_divisibility (a b c : ℕ) : 
  (∃ k : ℕ, (a * b + 1) = k * (2 * c)) ∧
  (∃ m : ℕ, (b * c + 1) = m * (2 * a)) ∧
  (∃ n : ℕ, (c * a + 1) = n * (2 * b)) →
  a = 1 ∧ b = 1 ∧ c = 1 := by
  sorry

end NUMINAMATH_CALUDE_unique_triple_divisibility_l3092_309204


namespace NUMINAMATH_CALUDE_multiplication_addition_equality_l3092_309235

theorem multiplication_addition_equality : 25 * 13 * 2 + 15 * 13 * 7 = 2015 := by
  sorry

end NUMINAMATH_CALUDE_multiplication_addition_equality_l3092_309235


namespace NUMINAMATH_CALUDE_correct_propositions_l3092_309241

theorem correct_propositions (a b c d : ℝ) : 
  (∀ (a b c d : ℝ), a > b → c > d → a + c > b + d) ∧ 
  (∃ (a b c d : ℝ), a > b ∧ c > d ∧ ¬(a - c > b - d)) ∧
  (∃ (a b c d : ℝ), a > b ∧ c > d ∧ ¬(a * c > b * d)) ∧
  (∀ (a b c : ℝ), a > b → c > 0 → a * c > b * c) := by
  sorry

end NUMINAMATH_CALUDE_correct_propositions_l3092_309241


namespace NUMINAMATH_CALUDE_correct_calculation_l3092_309253

theorem correct_calculation (x : ℤ) (h : x + 392 = 541) : x + 293 = 442 := by
  sorry

end NUMINAMATH_CALUDE_correct_calculation_l3092_309253


namespace NUMINAMATH_CALUDE_sum_of_squares_squared_l3092_309281

theorem sum_of_squares_squared (n a b c : ℕ) (h : n = a^2 + b^2 + c^2) :
  ∃ x y z : ℕ, n^2 = x^2 + y^2 + z^2 := by
sorry

end NUMINAMATH_CALUDE_sum_of_squares_squared_l3092_309281


namespace NUMINAMATH_CALUDE_sum_max_min_cubes_l3092_309265

/-- Represents a view of the geometric figure -/
structure View where
  (front : Set (ℕ × ℕ))
  (left : Set (ℕ × ℕ))
  (top : Set (ℕ × ℕ))

/-- Counts the number of cubes in a valid configuration -/
def count_cubes (v : View) : ℕ → Bool := sorry

/-- The maximum number of cubes that can form the figure -/
def max_cubes (v : View) : ℕ := sorry

/-- The minimum number of cubes that can form the figure -/
def min_cubes (v : View) : ℕ := sorry

/-- The theorem stating that the sum of max and min cubes is 20 -/
theorem sum_max_min_cubes (v : View) : max_cubes v + min_cubes v = 20 := by sorry

end NUMINAMATH_CALUDE_sum_max_min_cubes_l3092_309265


namespace NUMINAMATH_CALUDE_bakery_storage_ratio_l3092_309227

/-- Proves that the ratio of flour to baking soda is 10:1 given the conditions in the bakery storage room. -/
theorem bakery_storage_ratio : 
  ∀ (sugar flour baking_soda : ℕ),
  -- Ratio of sugar to flour is 3:8
  8 * sugar = 3 * flour →
  -- There are 900 pounds of sugar
  sugar = 900 →
  -- If 60 more pounds of baking soda were added, the ratio of flour to baking soda would be 8:1
  8 * (baking_soda + 60) = flour →
  -- The ratio of flour to baking soda is 10:1
  10 * baking_soda = flour :=
by
  sorry


end NUMINAMATH_CALUDE_bakery_storage_ratio_l3092_309227


namespace NUMINAMATH_CALUDE_not_proposition_example_l3092_309217

-- Define what a proposition is in this context
def is_proposition (s : String) : Prop :=
  ∀ (interpretation : Type), (∃ (truth_value : Bool), true)

-- State the theorem
theorem not_proposition_example : ¬ (is_proposition "x^2 + 2x - 3 < 0") :=
sorry

end NUMINAMATH_CALUDE_not_proposition_example_l3092_309217


namespace NUMINAMATH_CALUDE_baseball_league_games_l3092_309240

/-- The number of teams in the baseball league -/
def num_teams : ℕ := 9

/-- The number of games each team plays with every other team -/
def games_per_pair : ℕ := 4

/-- The total number of games played in the season -/
def total_games : ℕ := (num_teams * (num_teams - 1) / 2) * games_per_pair

theorem baseball_league_games :
  total_games = 144 :=
sorry

end NUMINAMATH_CALUDE_baseball_league_games_l3092_309240


namespace NUMINAMATH_CALUDE_right_triangle_area_l3092_309206

theorem right_triangle_area (a b c : ℝ) (h1 : a^2 + b^2 = c^2) (h2 : c = 13) (h3 : a = 5) :
  (1/2) * a * b = 30 := by
sorry

end NUMINAMATH_CALUDE_right_triangle_area_l3092_309206


namespace NUMINAMATH_CALUDE_fruit_drink_total_amount_l3092_309223

/-- Represents the composition of a fruit drink -/
structure FruitDrink where
  orange_percent : Real
  watermelon_percent : Real
  grape_percent : Real
  pineapple_percent : Real
  grape_ounces : Real
  total_ounces : Real

/-- The theorem stating the total amount of the drink given its composition -/
theorem fruit_drink_total_amount (drink : FruitDrink) 
  (h1 : drink.orange_percent = 0.1)
  (h2 : drink.watermelon_percent = 0.55)
  (h3 : drink.grape_percent = 0.2)
  (h4 : drink.pineapple_percent = 1 - (drink.orange_percent + drink.watermelon_percent + drink.grape_percent))
  (h5 : drink.grape_ounces = 40)
  (h6 : drink.total_ounces * drink.grape_percent = drink.grape_ounces) :
  drink.total_ounces = 200 := by
  sorry

end NUMINAMATH_CALUDE_fruit_drink_total_amount_l3092_309223


namespace NUMINAMATH_CALUDE_min_value_theorem_l3092_309218

theorem min_value_theorem (a m n : ℝ) (ha : a > 0) (ha_neq : a ≠ 1) 
  (hm : m > 0) (hn : n > 0) : 
  let f := fun x => a^(x + 3) - 2
  let A := (-3, -1)
  (A.1 / m + A.2 / n = -1) → 
  (∀ k l, k > 0 → l > 0 → k / m + l / n = -1 → 3*m + n ≤ 3*k + l) →
  3*m + n ≥ 16 :=
by sorry

end NUMINAMATH_CALUDE_min_value_theorem_l3092_309218


namespace NUMINAMATH_CALUDE_min_blocks_needed_l3092_309262

/-- Represents a three-dimensional structure made of cube blocks -/
structure CubeStructure where
  blocks : ℕ → ℕ → ℕ → Bool

/-- The front view of the structure shows a 2x2 grid -/
def front_view_valid (s : CubeStructure) : Prop :=
  ∃ (i j : Fin 2), s.blocks i.val j.val 0 = true

/-- The left side view of the structure shows a 2x2 grid -/
def left_view_valid (s : CubeStructure) : Prop :=
  ∃ (i k : Fin 2), s.blocks 0 i.val k.val = true

/-- Count the number of blocks in the structure -/
def block_count (s : CubeStructure) : ℕ :=
  (Finset.range 2).sum fun i =>
    (Finset.range 2).sum fun j =>
      (Finset.range 2).sum fun k =>
        if s.blocks i j k then 1 else 0

/-- The main theorem: minimum number of blocks needed is 4 -/
theorem min_blocks_needed (s : CubeStructure) 
  (h_front : front_view_valid s) (h_left : left_view_valid s) :
  block_count s ≥ 4 := by
  sorry


end NUMINAMATH_CALUDE_min_blocks_needed_l3092_309262


namespace NUMINAMATH_CALUDE_iphone_savings_l3092_309209

/-- Represents the cost of an iPhone X in dollars -/
def iphone_cost : ℝ := 600

/-- Represents the discount percentage for buying multiple smartphones -/
def discount_percentage : ℝ := 5

/-- Represents the number of iPhones being purchased -/
def num_iphones : ℕ := 3

/-- Theorem stating that the savings from buying 3 iPhones X together with a 5% discount,
    compared to buying them individually without a discount, is $90 -/
theorem iphone_savings :
  (num_iphones * iphone_cost) * (discount_percentage / 100) = 90 := by
  sorry

end NUMINAMATH_CALUDE_iphone_savings_l3092_309209


namespace NUMINAMATH_CALUDE_negation_of_existence_ln_positive_l3092_309221

theorem negation_of_existence_ln_positive :
  (¬ ∃ x : ℝ, x > 0 ∧ Real.log x > 0) ↔ (∀ x : ℝ, x > 0 → Real.log x ≤ 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_existence_ln_positive_l3092_309221


namespace NUMINAMATH_CALUDE_simplify_expression_l3092_309228

theorem simplify_expression : 
  (625 : ℝ) ^ (1/4 : ℝ) * (343 : ℝ) ^ (1/3 : ℝ) = 35 := by
  sorry

#check simplify_expression

end NUMINAMATH_CALUDE_simplify_expression_l3092_309228


namespace NUMINAMATH_CALUDE_xy_square_value_l3092_309238

theorem xy_square_value (x y : ℝ) 
  (h1 : x * (x + y) = 22)
  (h2 : y * (x + y) = 78 - y) : 
  (x + y)^2 = 100 := by
sorry

end NUMINAMATH_CALUDE_xy_square_value_l3092_309238


namespace NUMINAMATH_CALUDE_min_jumps_to_blue_l3092_309230

/-- Represents a 4x4 grid where each cell can be either red or blue -/
def Grid := Fin 4 → Fin 4 → Bool

/-- A position on the 4x4 grid -/
structure Position where
  row : Fin 4
  col : Fin 4

/-- Checks if two positions are adjacent (share a side) -/
def adjacent (p1 p2 : Position) : Bool :=
  (p1.row = p2.row ∧ (p1.col = p2.col + 1 ∨ p1.col + 1 = p2.col)) ∨
  (p1.col = p2.col ∧ (p1.row = p2.row + 1 ∨ p1.row + 1 = p2.row))

/-- The effect of jumping on a position, changing it and adjacent positions to blue -/
def jump (g : Grid) (p : Position) : Grid :=
  fun r c => if (r = p.row ∧ c = p.col) ∨ adjacent p ⟨r, c⟩ then true else g r c

/-- A sequence of jumps -/
def JumpSequence := List Position

/-- Apply a sequence of jumps to a grid -/
def applyJumps (g : Grid) : JumpSequence → Grid
  | [] => g
  | p::ps => applyJumps (jump g p) ps

/-- Check if all squares in the grid are blue -/
def allBlue (g : Grid) : Prop :=
  ∀ r c, g r c = true

/-- The initial all-red grid -/
def initialGrid : Grid :=
  fun _ _ => false

/-- Theorem: There exists a sequence of 4 jumps that turns the entire grid blue -/
theorem min_jumps_to_blue :
  ∃ (js : JumpSequence), js.length = 4 ∧ allBlue (applyJumps initialGrid js) :=
sorry


end NUMINAMATH_CALUDE_min_jumps_to_blue_l3092_309230


namespace NUMINAMATH_CALUDE_students_with_glasses_l3092_309201

theorem students_with_glasses (total : ℕ) (difference : ℕ) : total = 36 → difference = 24 → 
  ∃ (with_glasses : ℕ) (without_glasses : ℕ), 
    with_glasses + without_glasses = total ∧ 
    with_glasses + difference = without_glasses ∧ 
    with_glasses = 6 := by
  sorry

end NUMINAMATH_CALUDE_students_with_glasses_l3092_309201


namespace NUMINAMATH_CALUDE_right_rectangular_prism_volume_l3092_309222

theorem right_rectangular_prism_volume 
  (face_area1 face_area2 face_area3 : ℝ) 
  (h1 : face_area1 = 30)
  (h2 : face_area2 = 45)
  (h3 : face_area3 = 75) :
  ∃ (x y z : ℝ), 
    x * y = face_area1 ∧ 
    x * z = face_area2 ∧ 
    y * z = face_area3 ∧ 
    x * y * z = 150 := by
  sorry

end NUMINAMATH_CALUDE_right_rectangular_prism_volume_l3092_309222


namespace NUMINAMATH_CALUDE_season_games_l3092_309298

/-- The number of teams in the league -/
def num_teams : ℕ := 20

/-- The number of times each team faces another team -/
def games_per_matchup : ℕ := 10

/-- Calculate the number of unique matchups in the league -/
def unique_matchups (n : ℕ) : ℕ := n * (n - 1) / 2

/-- Calculate the total number of games in the season -/
def total_games (n : ℕ) (g : ℕ) : ℕ := unique_matchups n * g

theorem season_games :
  total_games num_teams games_per_matchup = 1900 := by sorry

end NUMINAMATH_CALUDE_season_games_l3092_309298


namespace NUMINAMATH_CALUDE_units_digit_product_minus_power_l3092_309285

def units_digit (n : ℤ) : ℕ :=
  (n % 10).toNat

theorem units_digit_product_minus_power : units_digit (8 * 18 * 1988 - 8^4) = 6 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_product_minus_power_l3092_309285


namespace NUMINAMATH_CALUDE_complex_modulus_l3092_309226

theorem complex_modulus (z : ℂ) (h : z * (2 + Complex.I) = Complex.I ^ 10) :
  Complex.abs z = Real.sqrt 5 / 5 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_l3092_309226


namespace NUMINAMATH_CALUDE_bianca_carrots_l3092_309292

def carrot_problem (initial : ℕ) (thrown_out : ℕ) (additional : ℕ) : ℕ :=
  initial - thrown_out + additional

theorem bianca_carrots : carrot_problem 23 10 47 = 60 := by
  sorry

end NUMINAMATH_CALUDE_bianca_carrots_l3092_309292


namespace NUMINAMATH_CALUDE_stating_all_magpies_fly_away_l3092_309276

/-- Represents the number of magpies remaining on a tree after a hunting incident -/
def magpies_remaining (initial : ℕ) (killed : ℕ) : ℕ :=
  0

/-- 
Theorem stating that regardless of the initial number of magpies and the number killed,
no magpies remain on the tree after the incident.
-/
theorem all_magpies_fly_away (initial : ℕ) (killed : ℕ) :
  magpies_remaining initial killed = 0 := by
  sorry

end NUMINAMATH_CALUDE_stating_all_magpies_fly_away_l3092_309276


namespace NUMINAMATH_CALUDE_largest_c_for_one_in_range_l3092_309266

-- Define the function f(x)
def f (c : ℝ) (x : ℝ) : ℝ := x^2 - 6*x + c

-- State the theorem
theorem largest_c_for_one_in_range : 
  (∃ (c : ℝ), ∀ (d : ℝ), (∃ (x : ℝ), f d x = 1) → d ≤ c) ∧ 
  (∃ (x : ℝ), f 10 x = 1) :=
by sorry

end NUMINAMATH_CALUDE_largest_c_for_one_in_range_l3092_309266


namespace NUMINAMATH_CALUDE_exhibits_permutation_l3092_309250

theorem exhibits_permutation : Nat.factorial 5 = 120 := by
  sorry

end NUMINAMATH_CALUDE_exhibits_permutation_l3092_309250


namespace NUMINAMATH_CALUDE_distribution_ratio_l3092_309263

def num_balls : ℕ := 20
def num_bins : ℕ := 5

def distribution_A : List ℕ := [3, 5, 4, 4, 4]
def distribution_B : List ℕ := [4, 4, 4, 4, 4]

def count_distributions (n : ℕ) (k : ℕ) (dist : List ℕ) : ℕ :=
  sorry

theorem distribution_ratio :
  (count_distributions num_balls num_bins distribution_A) /
  (count_distributions num_balls num_bins distribution_B) = 4 :=
by sorry

end NUMINAMATH_CALUDE_distribution_ratio_l3092_309263


namespace NUMINAMATH_CALUDE_square_1600_product_l3092_309252

theorem square_1600_product (x : ℤ) (h : x^2 = 1600) : (x + 2) * (x - 2) = 1596 := by
  sorry

end NUMINAMATH_CALUDE_square_1600_product_l3092_309252


namespace NUMINAMATH_CALUDE_number_division_problem_l3092_309282

theorem number_division_problem (x y : ℚ) : 
  (x - 5) / y = 7 → (x - 2) / 13 = 4 → y = 7 := by
  sorry

end NUMINAMATH_CALUDE_number_division_problem_l3092_309282


namespace NUMINAMATH_CALUDE_equal_distribution_of_treats_l3092_309232

theorem equal_distribution_of_treats (cookies cupcakes brownies students : ℕ) 
  (h1 : cookies = 20)
  (h2 : cupcakes = 25)
  (h3 : brownies = 35)
  (h4 : students = 20) :
  (cookies + cupcakes + brownies) / students = 4 :=
by sorry

end NUMINAMATH_CALUDE_equal_distribution_of_treats_l3092_309232


namespace NUMINAMATH_CALUDE_average_first_15_even_numbers_l3092_309207

theorem average_first_15_even_numbers : 
  let first_15_even : List ℕ := List.range 15 |>.map (fun n => 2 * (n + 1))
  (first_15_even.sum : ℚ) / 15 = 16 := by
  sorry

end NUMINAMATH_CALUDE_average_first_15_even_numbers_l3092_309207


namespace NUMINAMATH_CALUDE_track_meet_boys_count_l3092_309205

theorem track_meet_boys_count :
  ∀ (total girls boys : ℕ),
  total = 55 →
  total = girls + boys →
  (3 * girls : ℚ) / 5 + (2 * girls : ℚ) / 5 = girls →
  (2 * girls : ℚ) / 5 = 10 →
  boys = 30 :=
by
  sorry

end NUMINAMATH_CALUDE_track_meet_boys_count_l3092_309205


namespace NUMINAMATH_CALUDE_abs_sum_eq_sum_abs_iff_product_pos_l3092_309219

theorem abs_sum_eq_sum_abs_iff_product_pos (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) :
  |a + b| = |a| + |b| ↔ a * b > 0 := by sorry

end NUMINAMATH_CALUDE_abs_sum_eq_sum_abs_iff_product_pos_l3092_309219


namespace NUMINAMATH_CALUDE_camp_cedar_counselors_l3092_309272

def camp_cedar (num_boys : ℕ) (girl_ratio : ℕ) (children_per_counselor : ℕ) : ℕ :=
  let num_girls := num_boys * girl_ratio
  let total_children := num_boys + num_girls
  total_children / children_per_counselor

theorem camp_cedar_counselors :
  camp_cedar 40 3 8 = 20 := by
  sorry

end NUMINAMATH_CALUDE_camp_cedar_counselors_l3092_309272


namespace NUMINAMATH_CALUDE_horizontal_line_slope_line_2023_slope_l3092_309239

/-- The slope of a horizontal line y = k is 0 -/
theorem horizontal_line_slope (k : ℝ) : 
  let f : ℝ → ℝ := λ x => k
  (∀ x : ℝ, (f x) = k) → 
  (∀ x₁ x₂ : ℝ, x₁ ≠ x₂ → (f x₂ - f x₁) / (x₂ - x₁) = 0) :=
by
  sorry

/-- The slope of the line y = 2023 is 0 -/
theorem line_2023_slope : 
  let f : ℝ → ℝ := λ x => 2023
  (∀ x : ℝ, (f x) = 2023) → 
  (∀ x₁ x₂ : ℝ, x₁ ≠ x₂ → (f x₂ - f x₁) / (x₂ - x₁) = 0) :=
by
  sorry

end NUMINAMATH_CALUDE_horizontal_line_slope_line_2023_slope_l3092_309239


namespace NUMINAMATH_CALUDE_fuel_consumption_analysis_l3092_309289

/-- Represents the fuel consumption data for a sedan --/
structure FuelData where
  initial_fuel : ℝ
  distance : ℝ
  remaining_fuel : ℝ

/-- Theorem about fuel consumption of a sedan --/
theorem fuel_consumption_analysis 
  (data : List FuelData)
  (h1 : data.length ≥ 2)
  (h2 : data[0].distance = 0 ∧ data[0].remaining_fuel = 50)
  (h3 : data[1].distance = 100 ∧ data[1].remaining_fuel = 42)
  (h4 : ∀ d ∈ data, d.initial_fuel = 50)
  (h5 : ∀ d ∈ data, d.remaining_fuel = d.initial_fuel - 0.08 * d.distance) :
  (∀ d ∈ data, d.initial_fuel = 50) ∧ 
  (∀ d ∈ data, d.remaining_fuel = -0.08 * d.distance + 50) := by
  sorry


end NUMINAMATH_CALUDE_fuel_consumption_analysis_l3092_309289


namespace NUMINAMATH_CALUDE_taxi_fare_for_80_miles_l3092_309248

/-- Represents the fare structure of a taxi company -/
structure TaxiFare where
  fixedFare : ℝ
  costPerMile : ℝ

/-- Calculates the total fare for a given distance -/
def totalFare (tf : TaxiFare) (distance : ℝ) : ℝ :=
  tf.fixedFare + tf.costPerMile * distance

theorem taxi_fare_for_80_miles :
  ∃ (tf : TaxiFare),
    tf.fixedFare = 15 ∧
    totalFare tf 60 = 135 ∧
    totalFare tf 80 = 175 := by
  sorry

end NUMINAMATH_CALUDE_taxi_fare_for_80_miles_l3092_309248


namespace NUMINAMATH_CALUDE_find_a_value_l3092_309211

-- Define an odd function
def OddFunction (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

-- Define the specific function for x < 0
def SpecificFunction (f : ℝ → ℝ) (a : ℝ) : Prop :=
  ∀ x, x < 0 → f x = x^2 + a*x

theorem find_a_value (f : ℝ → ℝ) (a : ℝ) 
  (h_odd : OddFunction f)
  (h_spec : SpecificFunction f a)
  (h_f3 : f 3 = 6) :
  a = 5 := by
sorry

end NUMINAMATH_CALUDE_find_a_value_l3092_309211


namespace NUMINAMATH_CALUDE_range_of_m_for_P_and_not_Q_l3092_309216

/-- The function f(x) parameterized by m -/
def f (m : ℝ) (x : ℝ) : ℝ := x^3 + m*x^2 + (m+6)*x + 1

/-- Predicate P: 2 ≤ m ≤ 8 -/
def P (m : ℝ) : Prop := 2 ≤ m ∧ m ≤ 8

/-- Predicate Q: f has both a maximum and a minimum value -/
def Q (m : ℝ) : Prop := ∃ (a b : ℝ), ∀ (x : ℝ), f m a ≤ f m x ∧ f m x ≤ f m b

/-- The range of m for which P ∩ ¬Q is true is [2, 6] -/
theorem range_of_m_for_P_and_not_Q :
  {m : ℝ | P m ∧ ¬Q m} = {m : ℝ | 2 ≤ m ∧ m ≤ 6} := by sorry

end NUMINAMATH_CALUDE_range_of_m_for_P_and_not_Q_l3092_309216


namespace NUMINAMATH_CALUDE_complex_magnitude_fourth_power_l3092_309249

theorem complex_magnitude_fourth_power : 
  Complex.abs ((4 + 2 * Real.sqrt 2 * Complex.I) ^ 4) = 576 := by sorry

end NUMINAMATH_CALUDE_complex_magnitude_fourth_power_l3092_309249


namespace NUMINAMATH_CALUDE_garden_ratio_l3092_309254

def garden_problem (table_price bench_price : ℕ) : Prop :=
  table_price + bench_price = 450 ∧
  ∃ k : ℕ, table_price = k * bench_price ∧
  bench_price = 150

theorem garden_ratio :
  ∀ table_price bench_price : ℕ,
  garden_problem table_price bench_price →
  table_price / bench_price = 2 :=
by
  sorry

end NUMINAMATH_CALUDE_garden_ratio_l3092_309254


namespace NUMINAMATH_CALUDE_fish_per_multicolor_duck_l3092_309261

theorem fish_per_multicolor_duck 
  (white_fish_ratio : ℕ) 
  (black_fish_ratio : ℕ) 
  (white_ducks : ℕ) 
  (black_ducks : ℕ) 
  (multicolor_ducks : ℕ) 
  (total_fish : ℕ) 
  (h1 : white_fish_ratio = 5)
  (h2 : black_fish_ratio = 10)
  (h3 : white_ducks = 3)
  (h4 : black_ducks = 7)
  (h5 : multicolor_ducks = 6)
  (h6 : total_fish = 157) :
  (total_fish - (white_fish_ratio * white_ducks + black_fish_ratio * black_ducks)) / multicolor_ducks = 12 := by
sorry

end NUMINAMATH_CALUDE_fish_per_multicolor_duck_l3092_309261


namespace NUMINAMATH_CALUDE_infinite_sum_equals_three_l3092_309229

open BigOperators

theorem infinite_sum_equals_three :
  ∑' k, (5^k) / ((4^k - 3^k) * (4^(k+1) - 3^(k+1))) = 3 := by
  sorry

end NUMINAMATH_CALUDE_infinite_sum_equals_three_l3092_309229


namespace NUMINAMATH_CALUDE_product_evaluation_l3092_309291

theorem product_evaluation (n : ℕ) (h : n = 4) : n * (n + 1) * (n + 2) = 120 := by
  sorry

end NUMINAMATH_CALUDE_product_evaluation_l3092_309291


namespace NUMINAMATH_CALUDE_max_value_abc_expression_l3092_309203

theorem max_value_abc_expression (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a * b * c * (a + b + c)) / ((a + b)^2 * (b + c)^2) ≤ (1 : ℝ) / 4 := by
  sorry

end NUMINAMATH_CALUDE_max_value_abc_expression_l3092_309203


namespace NUMINAMATH_CALUDE_quadratic_equal_roots_l3092_309288

theorem quadratic_equal_roots (m : ℝ) :
  (∃ x : ℝ, x^2 - 4*x + m = 0 ∧ 
   ∀ y : ℝ, y^2 - 4*y + m = 0 → y = x) ↔ m = 4 := by
sorry

end NUMINAMATH_CALUDE_quadratic_equal_roots_l3092_309288


namespace NUMINAMATH_CALUDE_geometric_sequence_ratio_l3092_309257

theorem geometric_sequence_ratio (a : ℕ → ℝ) (q : ℝ) :
  (∀ n, a n > 0) →
  (∀ n, a (n + 1) = q * a n) →
  (a 1 + a 1 * q + a 1 * q^2 + a 1 * q^3 = 10 * (a 1 + a 1 * q)) →
  q = 3 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_ratio_l3092_309257


namespace NUMINAMATH_CALUDE_product_of_elements_is_zero_l3092_309200

theorem product_of_elements_is_zero
  (n : ℕ)
  (M : Finset ℝ)
  (h_odd : Odd n)
  (h_gt_one : n > 1)
  (h_card : M.card = n)
  (h_sum_invariant : ∀ x ∈ M, M.sum id = (M.erase x).sum id + (M.sum id - x)) :
  M.prod id = 0 := by
sorry

end NUMINAMATH_CALUDE_product_of_elements_is_zero_l3092_309200


namespace NUMINAMATH_CALUDE_car_profit_percentage_l3092_309246

theorem car_profit_percentage (original_price : ℝ) (h1 : original_price > 0) : 
  let discount_rate : ℝ := 0.2
  let increase_rate : ℝ := 1
  let buying_price : ℝ := original_price * (1 - discount_rate)
  let selling_price : ℝ := buying_price * (1 + increase_rate)
  let profit : ℝ := selling_price - original_price
  let profit_percentage : ℝ := (profit / original_price) * 100
  profit_percentage = 60 := by sorry

end NUMINAMATH_CALUDE_car_profit_percentage_l3092_309246


namespace NUMINAMATH_CALUDE_square_sum_given_sum_square_and_product_l3092_309231

theorem square_sum_given_sum_square_and_product (x y : ℝ) 
  (h1 : (x + y)^2 = 4) 
  (h2 : x * y = -9) : 
  x^2 + y^2 = 22 := by
sorry

end NUMINAMATH_CALUDE_square_sum_given_sum_square_and_product_l3092_309231


namespace NUMINAMATH_CALUDE_smallest_k_with_multiple_sequences_l3092_309234

/-- A sequence of positive integers satisfying the given conditions -/
def ValidSequence (a : ℕ → ℕ) : Prop :=
  (∀ n, a n > 0) ∧
  (∀ n, a (n + 1) ≥ a n) ∧
  (∀ n > 2, a n = a (n - 1) + a (n - 2))

/-- The existence of at least two distinct valid sequences with a₉ = k -/
def HasMultipleSequences (k : ℕ) : Prop :=
  ∃ a b : ℕ → ℕ, ValidSequence a ∧ ValidSequence b ∧ a ≠ b ∧ a 9 = k ∧ b 9 = k

/-- 748 is the smallest k for which multiple valid sequences exist -/
theorem smallest_k_with_multiple_sequences :
  HasMultipleSequences 748 ∧ ∀ k < 748, ¬HasMultipleSequences k :=
sorry

end NUMINAMATH_CALUDE_smallest_k_with_multiple_sequences_l3092_309234


namespace NUMINAMATH_CALUDE_min_difference_l3092_309286

noncomputable def f (x : ℝ) : ℝ := Real.exp (2 * x - 3)
noncomputable def g (x : ℝ) : ℝ := 1 / 4 + Real.log (x / 2)

theorem min_difference (m n : ℝ) (h : f m = g n) :
  ∃ (d : ℝ), d = 1 / 2 + Real.log 2 ∧ n - m ≥ d ∧ ∃ (m₀ n₀ : ℝ), f m₀ = g n₀ ∧ n₀ - m₀ = d :=
sorry

end NUMINAMATH_CALUDE_min_difference_l3092_309286


namespace NUMINAMATH_CALUDE_least_multiple_squared_l3092_309267

theorem least_multiple_squared (X Y : ℕ) : 
  (∃ Y, 3456^2 * X = 6789^2 * Y) ∧ 
  (∀ Z, Z < X → ¬∃ W, 3456^2 * Z = 6789^2 * W) →
  X = 290521 := by
sorry

end NUMINAMATH_CALUDE_least_multiple_squared_l3092_309267


namespace NUMINAMATH_CALUDE_two_valid_colorings_l3092_309247

/-- Represents the three possible colors for a hexagon -/
inductive Color
| Red
| Yellow
| Green

/-- Represents a column of hexagons -/
structure Column where
  hexagons : List Color
  size : Nat
  size_eq : hexagons.length = size

/-- Represents the entire figure of hexagons -/
structure HexagonFigure where
  column1 : Column
  column2 : Column
  column3 : Column
  column4 : Column
  col1_size : column1.size = 3
  col2_size : column2.size = 4
  col3_size : column3.size = 4
  col4_size : column4.size = 3
  bottom_red : column1.hexagons.head? = some Color.Red

/-- Predicate to check if two colors are different -/
def differentColors (c1 c2 : Color) : Prop :=
  c1 ≠ c2

/-- Predicate to check if a coloring is valid (no adjacent hexagons have the same color) -/
def validColoring (figure : HexagonFigure) : Prop :=
  -- Add conditions to check adjacent hexagons in each column and between columns
  sorry

/-- The number of valid colorings for the hexagon figure -/
def numValidColorings : Nat :=
  -- Count the number of valid colorings
  sorry

/-- Theorem stating that there are exactly 2 valid colorings -/
theorem two_valid_colorings : numValidColorings = 2 := by
  sorry

end NUMINAMATH_CALUDE_two_valid_colorings_l3092_309247


namespace NUMINAMATH_CALUDE_apollonius_circle_tangency_locus_l3092_309269

/-- Apollonius circle associated with segment AB -/
structure ApolloniusCircle (A B : ℝ × ℝ) where
  center : ℝ × ℝ
  radius : ℝ
  divides_ratio : ℝ → ℝ → Prop

/-- Point of tangency from A to the Apollonius circle -/
def tangency_point (A B : ℝ × ℝ) (circle : ApolloniusCircle A B) : ℝ × ℝ := sorry

/-- Line perpendicular to AB at point B -/
def perpendicular_line_at_B (A B : ℝ × ℝ) : Set (ℝ × ℝ) := sorry

theorem apollonius_circle_tangency_locus 
  (A B : ℝ × ℝ) 
  (p : ℝ) 
  (h_p : p > 1) 
  (circle : ApolloniusCircle A B) 
  (h_circle : circle.divides_ratio p 1) :
  tangency_point A B circle ∈ perpendicular_line_at_B A B :=
sorry

end NUMINAMATH_CALUDE_apollonius_circle_tangency_locus_l3092_309269


namespace NUMINAMATH_CALUDE_helen_hand_wash_frequency_l3092_309279

/-- The frequency of Helen's hand washing her pillowcases in weeks -/
def hand_wash_frequency (time_per_wash : ℕ) (total_time_per_year : ℕ) (weeks_per_year : ℕ) : ℕ :=
  weeks_per_year / (total_time_per_year / time_per_wash)

/-- Theorem stating that Helen hand washes her pillowcases every 4 weeks -/
theorem helen_hand_wash_frequency :
  hand_wash_frequency 30 390 52 = 4 := by
  sorry

end NUMINAMATH_CALUDE_helen_hand_wash_frequency_l3092_309279


namespace NUMINAMATH_CALUDE_cube_volume_l3092_309268

theorem cube_volume (a : ℤ) : 
  (∃ (x y : ℤ), x = a + 2 ∧ y = a - 2 ∧ 
    x * a * y = a^3 - 16 ∧
    2 * (x + a) = 2 * (a + a) + 4) →
  a^3 = 216 := by
sorry

end NUMINAMATH_CALUDE_cube_volume_l3092_309268


namespace NUMINAMATH_CALUDE_divisors_equidistant_from_third_l3092_309210

theorem divisors_equidistant_from_third (n : ℕ) : 
  (∃ (a b : ℕ), a ≠ b ∧ a ∣ n ∧ b ∣ n ∧ 
   (n : ℚ) / 3 - (a : ℚ) = (b : ℚ) - (n : ℚ) / 3) → 
  ∃ (k : ℕ), n = 6 * k :=
sorry

end NUMINAMATH_CALUDE_divisors_equidistant_from_third_l3092_309210


namespace NUMINAMATH_CALUDE_binomial_expansion_problem_l3092_309236

theorem binomial_expansion_problem (a₀ a₁ a₂ a₃ a₄ a₅ : ℝ) :
  (∀ x : ℝ, (3 - 2*x)^5 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5) →
  (a₀ + a₂ + a₄)^2 - (a₁ + a₃ + a₅)^2 = 3125 := by
  sorry

end NUMINAMATH_CALUDE_binomial_expansion_problem_l3092_309236


namespace NUMINAMATH_CALUDE_corrected_mean_calculation_l3092_309278

def correct_mean (n : ℕ) (original_mean : ℚ) (incorrect_value : ℚ) (correct_value : ℚ) : ℚ :=
  (n : ℚ) * original_mean - incorrect_value + correct_value

theorem corrected_mean_calculation (n : ℕ) (original_mean incorrect_value correct_value : ℚ) :
  n = 50 →
  original_mean = 36 →
  incorrect_value = 23 →
  correct_value = 45 →
  (correct_mean n original_mean incorrect_value correct_value) / n = 36.44 :=
by
  sorry

#eval (correct_mean 50 36 23 45) / 50

end NUMINAMATH_CALUDE_corrected_mean_calculation_l3092_309278


namespace NUMINAMATH_CALUDE_tan_fraction_equals_two_l3092_309213

theorem tan_fraction_equals_two (α : Real) (h : Real.tan α = 3) :
  (Real.sin (α - Real.pi) + Real.cos (Real.pi - α)) /
  (Real.sin (Real.pi / 2 - α) + Real.cos (Real.pi / 2 + α)) = 2 := by
  sorry

end NUMINAMATH_CALUDE_tan_fraction_equals_two_l3092_309213


namespace NUMINAMATH_CALUDE_max_books_borrowed_l3092_309295

theorem max_books_borrowed (total_students : Nat) (zero_books : Nat) (one_book : Nat) 
  (two_books : Nat) (min_three_books : Nat) (avg_books : Nat) :
  total_students = 20 →
  zero_books = 2 →
  one_book = 10 →
  two_books = 5 →
  min_three_books = total_students - zero_books - one_book - two_books →
  avg_books = 2 →
  ∃ (max_books : Nat), 
    max_books = (total_students * avg_books) - 
      (one_book * 1 + two_books * 2 + (min_three_books - 1) * 3) ∧
    max_books ≤ 14 :=
by sorry

end NUMINAMATH_CALUDE_max_books_borrowed_l3092_309295


namespace NUMINAMATH_CALUDE_function_value_at_negative_m_l3092_309299

-- Define the function f
def f (a b x : ℝ) : ℝ := a * x^3 + b * x + 1

-- State the theorem
theorem function_value_at_negative_m (a b m : ℝ) :
  f a b m = 6 → f a b (-m) = -4 := by
  sorry

end NUMINAMATH_CALUDE_function_value_at_negative_m_l3092_309299


namespace NUMINAMATH_CALUDE_mean_median_difference_l3092_309283

/-- Represents the frequency of students for each number of days missed -/
def frequency : List (ℕ × ℕ) := [(0, 4), (1, 2), (2, 5), (3, 3), (4, 2), (5, 4)]

/-- Total number of students -/
def total_students : ℕ := 20

/-- Calculates the median of the dataset -/
def median (freq : List (ℕ × ℕ)) (total : ℕ) : ℚ := sorry

/-- Calculates the mean of the dataset -/
def mean (freq : List (ℕ × ℕ)) (total : ℕ) : ℚ := sorry

/-- The main theorem stating the difference between mean and median -/
theorem mean_median_difference :
  mean frequency total_students - median frequency total_students = 9 / 20 := by sorry

end NUMINAMATH_CALUDE_mean_median_difference_l3092_309283


namespace NUMINAMATH_CALUDE_taxes_paid_equals_135_l3092_309212

/-- Calculate taxes paid given gross pay and net pay -/
def calculate_taxes (gross_pay : ℝ) (net_pay : ℝ) : ℝ :=
  gross_pay - net_pay

/-- Theorem: Taxes paid are 135 dollars given the conditions -/
theorem taxes_paid_equals_135 :
  let gross_pay : ℝ := 450
  let net_pay : ℝ := 315
  calculate_taxes gross_pay net_pay = 135 := by
  sorry

end NUMINAMATH_CALUDE_taxes_paid_equals_135_l3092_309212


namespace NUMINAMATH_CALUDE_sarah_age_l3092_309243

/-- Given the ages of Billy, Joe, and Sarah, prove that Sarah is 10 years old -/
theorem sarah_age (B J S : ℕ) 
  (h1 : B = 2 * J)           -- Billy's age is twice Joe's age
  (h2 : B + J = 60)          -- The sum of Billy's and Joe's ages is 60 years
  (h3 : S = J - 10)          -- Sarah's age is 10 years less than Joe's age
  : S = 10 := by             -- Prove that Sarah is 10 years old
  sorry

end NUMINAMATH_CALUDE_sarah_age_l3092_309243


namespace NUMINAMATH_CALUDE_doctor_lawyer_engineer_ratio_l3092_309273

-- Define the number of doctors, lawyers, and engineers
variable (d l e : ℕ)

-- Define the average ages
def avg_all : ℚ := 45
def avg_doctors : ℕ := 40
def avg_lawyers : ℕ := 55
def avg_engineers : ℕ := 35

-- State the theorem
theorem doctor_lawyer_engineer_ratio :
  (avg_all : ℚ) * (d + l + e : ℚ) = avg_doctors * d + avg_lawyers * l + avg_engineers * e →
  l = d + 2 * e :=
by sorry

end NUMINAMATH_CALUDE_doctor_lawyer_engineer_ratio_l3092_309273


namespace NUMINAMATH_CALUDE_chess_tournament_players_l3092_309220

/-- Represents a chess tournament with the given conditions -/
structure ChessTournament where
  n : ℕ  -- number of players
  winner_wins : ℕ  -- number of wins by the winner
  winner_draws : ℕ  -- number of draws by the winner

/-- The conditions of the tournament are satisfied -/
def valid_tournament (t : ChessTournament) : Prop :=
  t.n > 1 ∧  -- more than one player
  t.winner_wins = t.winner_draws ∧  -- winner won half and drew half
  t.winner_wins + t.winner_draws = t.n - 1 ∧  -- winner played against every other player once
  (t.winner_wins : ℚ) + (t.winner_draws : ℚ) / 2 = (t.n * (t.n - 1) : ℚ) / 20  -- winner's points are 9 times less than others'

theorem chess_tournament_players (t : ChessTournament) :
  valid_tournament t → t.n = 15 := by
  sorry

#check chess_tournament_players

end NUMINAMATH_CALUDE_chess_tournament_players_l3092_309220


namespace NUMINAMATH_CALUDE_phase_shift_of_sine_function_l3092_309270

/-- The phase shift of the function y = 2 sin(2x + π/3) is -π/6 -/
theorem phase_shift_of_sine_function :
  let f : ℝ → ℝ := λ x => 2 * Real.sin (2 * x + π / 3)
  ∃ (A B C D : ℝ), A ≠ 0 ∧ B ≠ 0 ∧
    (∀ x, f x = A * Real.sin (B * (x - C)) + D) ∧
    C = -π / 6 :=
by sorry

end NUMINAMATH_CALUDE_phase_shift_of_sine_function_l3092_309270


namespace NUMINAMATH_CALUDE_unit_conversions_l3092_309275

-- Define conversion factors
def meters_to_decimeters : ℝ → ℝ := (· * 10)
def minutes_to_seconds : ℝ → ℝ := (· * 60)

-- Theorem to prove the conversions
theorem unit_conversions :
  (meters_to_decimeters 2 = 20) ∧
  (minutes_to_seconds 2 = 120) ∧
  (minutes_to_seconds (600 / 60) = 10) := by
  sorry

end NUMINAMATH_CALUDE_unit_conversions_l3092_309275


namespace NUMINAMATH_CALUDE_candy_distribution_l3092_309233

theorem candy_distribution (initial_candies : ℕ) (additional_candies : ℕ) (friends : ℕ) 
  (h1 : initial_candies = 20)
  (h2 : additional_candies = 4)
  (h3 : friends = 6)
  (h4 : friends > 0) :
  (initial_candies + additional_candies) / friends = 4 := by
  sorry

end NUMINAMATH_CALUDE_candy_distribution_l3092_309233


namespace NUMINAMATH_CALUDE_infinitely_many_composites_in_sequence_l3092_309280

theorem infinitely_many_composites_in_sequence :
  ∀ N : ℕ, ∃ k : ℕ, k > N ∧ 
    (∃ m : ℕ, (10^(16*k+8) - 1) / 3 = 17 * m) :=
sorry

end NUMINAMATH_CALUDE_infinitely_many_composites_in_sequence_l3092_309280


namespace NUMINAMATH_CALUDE_fruit_salad_composition_l3092_309255

theorem fruit_salad_composition (total : ℕ) (red_grapes : ℕ) (green_grapes : ℕ) (raspberries : ℕ) :
  total = 102 →
  red_grapes = 67 →
  raspberries = green_grapes - 5 →
  red_grapes = 3 * green_grapes + (red_grapes - 3 * green_grapes) →
  red_grapes - 3 * green_grapes = 7 :=
by
  sorry

end NUMINAMATH_CALUDE_fruit_salad_composition_l3092_309255


namespace NUMINAMATH_CALUDE_sheila_saves_for_four_years_l3092_309294

/-- Calculates the number of years Sheila plans to save. -/
def sheilas_savings_years (initial_savings : ℕ) (monthly_savings : ℕ) (family_addition : ℕ) (final_amount : ℕ) : ℕ :=
  ((final_amount - family_addition - initial_savings) / monthly_savings) / 12

/-- Theorem stating that Sheila plans to save for 4 years. -/
theorem sheila_saves_for_four_years :
  sheilas_savings_years 3000 276 7000 23248 = 4 := by
  sorry

end NUMINAMATH_CALUDE_sheila_saves_for_four_years_l3092_309294


namespace NUMINAMATH_CALUDE_expression_evaluation_l3092_309242

theorem expression_evaluation : 
  (4 * 6) / (12 * 18) * (9 * 12 * 18) / (4 * 6 * 9^2) = 1 / 9 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l3092_309242


namespace NUMINAMATH_CALUDE_absolute_value_equality_l3092_309271

theorem absolute_value_equality (x : ℝ) (h : x > 0) :
  |x + Real.sqrt ((x + 1)^2)| = 2*x + 1 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_equality_l3092_309271


namespace NUMINAMATH_CALUDE_sum_of_digits_in_19_minutes_l3092_309264

def sum_of_digits (n : Nat) : Nat :=
  if n < 10 then n else (n % 10) + sum_of_digits (n / 10)

def time_to_minutes (hours minutes : Nat) : Nat :=
  (hours % 12) * 60 + minutes

def minutes_to_time (total_minutes : Nat) : (Nat × Nat) :=
  ((total_minutes / 60) % 12, total_minutes % 60)

theorem sum_of_digits_in_19_minutes 
  (current_hours current_minutes : Nat) 
  (h_valid_time : current_hours < 12 ∧ current_minutes < 60) 
  (h_sum_condition : 
    let (prev_hours, prev_minutes) := minutes_to_time (time_to_minutes current_hours current_minutes - 19)
    sum_of_digits prev_hours + sum_of_digits prev_minutes = 
      sum_of_digits current_hours + sum_of_digits current_minutes - 2) :
  let (future_hours, future_minutes) := minutes_to_time (time_to_minutes current_hours current_minutes + 19)
  sum_of_digits future_hours + sum_of_digits future_minutes = 11 := by
sorry

end NUMINAMATH_CALUDE_sum_of_digits_in_19_minutes_l3092_309264
