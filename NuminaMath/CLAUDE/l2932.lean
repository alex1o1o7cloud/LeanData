import Mathlib

namespace NUMINAMATH_CALUDE_streetlight_purchase_l2932_293241

theorem streetlight_purchase (squares : Nat) (lights_per_square : Nat) (repair_lights : Nat) (bought_lights : Nat) : 
  squares = 15 → 
  lights_per_square = 12 → 
  repair_lights = 35 → 
  bought_lights = 200 → 
  squares * lights_per_square + repair_lights - bought_lights = 15 := by
  sorry

end NUMINAMATH_CALUDE_streetlight_purchase_l2932_293241


namespace NUMINAMATH_CALUDE_quadratic_inequality_sum_l2932_293217

/-- Given a quadratic inequality ax^2 - 3ax - 6 < 0 with solution set {x | x < 1 or x > b}, 
    prove that a + b = -1 -/
theorem quadratic_inequality_sum (a b : ℝ) : 
  (∀ x, ax^2 - 3*a*x - 6 < 0 ↔ x < 1 ∨ x > b) → 
  a + b = -1 := by
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_sum_l2932_293217


namespace NUMINAMATH_CALUDE_factor_polynomial_l2932_293273

theorem factor_polynomial (x : ℝ) : 18 * x^3 + 9 * x^2 + 3 * x = 3 * x * (6 * x^2 + 3 * x + 1) := by
  sorry

end NUMINAMATH_CALUDE_factor_polynomial_l2932_293273


namespace NUMINAMATH_CALUDE_floor_equation_solution_l2932_293209

theorem floor_equation_solution :
  ∀ m n : ℕ+,
  (⌊(m^2 : ℚ) / n⌋ + ⌊(n^2 : ℚ) / m⌋ = ⌊(m : ℚ) / n + (n : ℚ) / m⌋ + m * n) ↔ (m = 2 ∧ n = 1) :=
by sorry

end NUMINAMATH_CALUDE_floor_equation_solution_l2932_293209


namespace NUMINAMATH_CALUDE_equation_equivalence_l2932_293269

theorem equation_equivalence : ∀ x : ℝ, x * (x + 2) = 5 ↔ x^2 + 2*x - 5 = 0 := by sorry

end NUMINAMATH_CALUDE_equation_equivalence_l2932_293269


namespace NUMINAMATH_CALUDE_hat_shoppe_pricing_l2932_293252

theorem hat_shoppe_pricing (x : ℝ) (h : x > 0) : 
  0.75 * (1.3 * x) = 0.975 * x := by sorry

end NUMINAMATH_CALUDE_hat_shoppe_pricing_l2932_293252


namespace NUMINAMATH_CALUDE_temperature_difference_l2932_293224

theorem temperature_difference 
  (highest_temp lowest_temp : ℝ) 
  (h_highest : highest_temp = 27) 
  (h_lowest : lowest_temp = 17) : 
  highest_temp - lowest_temp = 10 := by
sorry

end NUMINAMATH_CALUDE_temperature_difference_l2932_293224


namespace NUMINAMATH_CALUDE_base4_subtraction_l2932_293253

/-- Converts a natural number to its base 4 representation -/
def toBase4 (n : ℕ) : List ℕ :=
  sorry

/-- Subtracts two lists of digits in base 4 -/
def subtractBase4 (a b : List ℕ) : List ℕ :=
  sorry

/-- Converts a list of digits in base 4 to a natural number -/
def fromBase4 (digits : List ℕ) : ℕ :=
  sorry

theorem base4_subtraction :
  let a := 207
  let b := 85
  let a_base4 := toBase4 a
  let b_base4 := toBase4 b
  let diff_base4 := subtractBase4 a_base4 b_base4
  fromBase4 diff_base4 = fromBase4 [1, 2, 3, 2] :=
by sorry

end NUMINAMATH_CALUDE_base4_subtraction_l2932_293253


namespace NUMINAMATH_CALUDE_moneybox_fills_in_60_weeks_l2932_293278

/-- The number of weeks it takes for Monica's moneybox to get full -/
def weeks_to_fill : ℕ := sorry

/-- The amount Monica puts into her moneybox each week -/
def weekly_savings : ℕ := 15

/-- The number of times Monica repeats the saving process -/
def repetitions : ℕ := 5

/-- The total amount Monica takes to the bank -/
def total_savings : ℕ := 4500

/-- Theorem stating that the moneybox gets full in 60 weeks -/
theorem moneybox_fills_in_60_weeks : 
  weeks_to_fill = 60 ∧ 
  weekly_savings * weeks_to_fill * repetitions = total_savings :=
sorry

end NUMINAMATH_CALUDE_moneybox_fills_in_60_weeks_l2932_293278


namespace NUMINAMATH_CALUDE_divisibility_problem_l2932_293239

theorem divisibility_problem (a b c d : ℕ+) 
  (h1 : Nat.gcd a b = 18)
  (h2 : Nat.gcd b c = 45)
  (h3 : Nat.gcd c d = 75)
  (h4 : 80 < Nat.gcd d a)
  (h5 : Nat.gcd d a < 120) :
  7 ∣ a.val := by
  sorry

end NUMINAMATH_CALUDE_divisibility_problem_l2932_293239


namespace NUMINAMATH_CALUDE_quadrilateral_inequality_l2932_293247

theorem quadrilateral_inequality (A B P : ℝ) (θ₁ θ₂ : ℝ) 
  (hA : A > 0) (hB : B > 0) (hP : P > 0) 
  (hP_bound : P ≤ A + B)
  (h_cos : A * Real.cos θ₁ + B * Real.cos θ₂ = P) :
  A * Real.sin θ₁ + B * Real.sin θ₂ ≤ Real.sqrt ((A + B - P) * (A + B + P)) := by
  sorry

end NUMINAMATH_CALUDE_quadrilateral_inequality_l2932_293247


namespace NUMINAMATH_CALUDE_incorrect_conclusion_l2932_293201

theorem incorrect_conclusion (a b : ℝ) (h1 : a > b) (h2 : b > a + b) : 
  ¬(a * b > (a + b)^2) := by
sorry

end NUMINAMATH_CALUDE_incorrect_conclusion_l2932_293201


namespace NUMINAMATH_CALUDE_original_decimal_l2932_293254

theorem original_decimal (x : ℝ) : (1000 * x) / 100 = 12.5 → x = 1.25 := by
  sorry

end NUMINAMATH_CALUDE_original_decimal_l2932_293254


namespace NUMINAMATH_CALUDE_factorable_implies_even_l2932_293250

-- Define the quadratic expression
def quadratic (a : ℤ) (x : ℝ) : ℝ := 21 * x^2 + a * x + 21

-- Define what it means for the quadratic to be factorable into linear binomials with integer coefficients
def is_factorable (a : ℤ) : Prop :=
  ∃ (m n p q : ℤ), 
    ∀ (x : ℝ), quadratic a x = (m * x + n) * (p * x + q)

-- The theorem to prove
theorem factorable_implies_even (a : ℤ) : 
  is_factorable a → ∃ k : ℤ, a = 2 * k :=
sorry

end NUMINAMATH_CALUDE_factorable_implies_even_l2932_293250


namespace NUMINAMATH_CALUDE_expand_expression_l2932_293290

theorem expand_expression (x y : ℝ) :
  5 * (3 * x^3 - 4 * x * y + x^2 - y^2) = 15 * x^3 - 20 * x * y + 5 * x^2 - 5 * y^2 := by
  sorry

end NUMINAMATH_CALUDE_expand_expression_l2932_293290


namespace NUMINAMATH_CALUDE_two_books_different_genres_l2932_293289

/-- Represents the number of books in each genre -/
def books_per_genre : ℕ := 3

/-- Represents the number of genres -/
def num_genres : ℕ := 3

/-- Represents the total number of books -/
def total_books : ℕ := books_per_genre * num_genres

/-- Calculates the number of ways to choose two books of different genres -/
def choose_two_different_genres : ℕ := 
  (total_books * (total_books - books_per_genre)) / 2

theorem two_books_different_genres : 
  choose_two_different_genres = 27 := by sorry

end NUMINAMATH_CALUDE_two_books_different_genres_l2932_293289


namespace NUMINAMATH_CALUDE_total_jokes_after_four_weeks_l2932_293265

def total_jokes (initial_jessy initial_alan initial_tom initial_emily : ℕ)
                (rate_jessy rate_alan rate_tom rate_emily : ℕ)
                (weeks : ℕ) : ℕ :=
  let jessy := initial_jessy * (rate_jessy ^ weeks - 1) / (rate_jessy - 1)
  let alan := initial_alan * (rate_alan ^ weeks - 1) / (rate_alan - 1)
  let tom := initial_tom * (rate_tom ^ weeks - 1) / (rate_tom - 1)
  let emily := initial_emily * (rate_emily ^ weeks - 1) / (rate_emily - 1)
  jessy + alan + tom + emily

theorem total_jokes_after_four_weeks :
  total_jokes 11 7 5 3 3 2 4 4 4 = 1225 := by
  sorry

end NUMINAMATH_CALUDE_total_jokes_after_four_weeks_l2932_293265


namespace NUMINAMATH_CALUDE_composition_f_one_ninth_l2932_293210

noncomputable def f (x : ℝ) : ℝ :=
  if x > 0 then Real.log x / Real.log 3 else 2^x

theorem composition_f_one_ninth :
  f (f (1/9)) = 1/4 := by sorry

end NUMINAMATH_CALUDE_composition_f_one_ninth_l2932_293210


namespace NUMINAMATH_CALUDE_angle_trigonometry_l2932_293220

theorem angle_trigonometry (a : ℝ) (θ : ℝ) (h : a < 0) :
  let P : ℝ × ℝ := (4*a, 3*a)
  (∃ (r : ℝ), r > 0 ∧ P.1 = r * Real.cos θ ∧ P.2 = r * Real.sin θ) →
  (Real.sin θ = -3/5 ∧ Real.cos θ = -4/5) ∧
  ((1 + 2 * Real.sin (π + θ) * Real.cos (2023 * π - θ)) / 
   (Real.sin (π/2 + θ)^2 - Real.cos (5*π/2 - θ)^2) = 7) :=
by sorry

end NUMINAMATH_CALUDE_angle_trigonometry_l2932_293220


namespace NUMINAMATH_CALUDE_x_range_for_quadratic_inequality_l2932_293267

theorem x_range_for_quadratic_inequality :
  ∀ x : ℝ, (∀ a : ℝ, a ∈ Set.Icc (-3) 3 → x^2 - a*x + 1 ≥ 1) →
  x ≥ (3 + Real.sqrt 5) / 2 ∨ x ≤ (-3 - Real.sqrt 5) / 2 := by
sorry

end NUMINAMATH_CALUDE_x_range_for_quadratic_inequality_l2932_293267


namespace NUMINAMATH_CALUDE_scout_troop_profit_l2932_293222

/-- The profit calculation for a scout troop selling candy bars -/
theorem scout_troop_profit : 
  let total_bars : ℕ := 1500
  let cost_price : ℚ := 1 / 3  -- price per bar when buying more than 800
  let selling_price : ℚ := 1 / 2  -- price per bar when selling
  let total_cost : ℚ := total_bars * cost_price
  let total_revenue : ℚ := total_bars * selling_price
  let profit : ℚ := total_revenue - total_cost
  profit = 250 := by sorry

end NUMINAMATH_CALUDE_scout_troop_profit_l2932_293222


namespace NUMINAMATH_CALUDE_partition_set_exists_l2932_293272

theorem partition_set_exists (n : ℕ) (h : n ≥ 3) :
  ∃ (S : Finset ℕ), 
    (Finset.card S = 2 * n) ∧ 
    (∀ m : ℕ, 2 ≤ m ∧ m ≤ n → 
      ∃ (A : Finset ℕ), 
        A ⊆ S ∧ 
        Finset.card A = m ∧ 
        (∃ (B : Finset ℕ), B = S \ A ∧ Finset.sum A id = Finset.sum B id)) :=
by sorry

end NUMINAMATH_CALUDE_partition_set_exists_l2932_293272


namespace NUMINAMATH_CALUDE_eggs_per_group_l2932_293260

theorem eggs_per_group (total_eggs : ℕ) (num_groups : ℕ) (h1 : total_eggs = 16) (h2 : num_groups = 8) :
  total_eggs / num_groups = 2 := by
  sorry

end NUMINAMATH_CALUDE_eggs_per_group_l2932_293260


namespace NUMINAMATH_CALUDE_arithmetic_geometric_sequences_l2932_293248

/-- Arithmetic sequence {a_n} -/
def arithmetic_sequence (n : ℕ) : ℝ := 2 * n - 2

/-- Geometric sequence {b_n} -/
def geometric_sequence (n : ℕ) : ℝ := 2^(n - 1)

/-- Sum of first n terms of geometric sequence -/
def geometric_sum (n : ℕ) : ℝ := 2^n - 1

theorem arithmetic_geometric_sequences :
  (∀ n : ℕ, arithmetic_sequence n = 2 * n - 2) ∧
  arithmetic_sequence 2 = 2 ∧
  arithmetic_sequence 5 = 8 ∧
  (∀ n : ℕ, geometric_sequence n > 0) ∧
  geometric_sequence 1 = 1 ∧
  geometric_sequence 2 + geometric_sequence 3 = arithmetic_sequence 4 ∧
  (∀ n : ℕ, geometric_sum n = 2^n - 1) :=
sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_sequences_l2932_293248


namespace NUMINAMATH_CALUDE_zoo_population_after_changes_l2932_293294

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


end NUMINAMATH_CALUDE_zoo_population_after_changes_l2932_293294


namespace NUMINAMATH_CALUDE_system_solution_iff_conditions_l2932_293282

-- Define the system of equations
def has_solution (n p x y z : ℕ) : Prop :=
  x + p * y = n ∧ x + y = p^z

-- Define the conditions
def conditions (n p : ℕ) : Prop :=
  p > 1 ∧ (n - 1) % (p - 1) = 0 ∧ ∀ k : ℕ, n ≠ p^k

-- Theorem statement
theorem system_solution_iff_conditions (n p : ℕ) :
  (∃! x y z : ℕ, has_solution n p x y z) ↔ conditions n p :=
sorry

end NUMINAMATH_CALUDE_system_solution_iff_conditions_l2932_293282


namespace NUMINAMATH_CALUDE_garden_area_increase_l2932_293251

/-- Proves that changing a rectangular garden to a square garden with the same perimeter increases the area by 100 square feet. -/
theorem garden_area_increase (length width : ℝ) (h1 : length = 40) (h2 : width = 20) :
  let rectangle_area := length * width
  let perimeter := 2 * (length + width)
  let square_side := perimeter / 4
  let square_area := square_side * square_side
  square_area - rectangle_area = 100 := by
  sorry

end NUMINAMATH_CALUDE_garden_area_increase_l2932_293251


namespace NUMINAMATH_CALUDE_michaels_dogs_l2932_293221

theorem michaels_dogs (num_cats : ℕ) (cost_per_animal : ℕ) (total_cost : ℕ) :
  num_cats = 2 →
  cost_per_animal = 13 →
  total_cost = 65 →
  ∃ num_dogs : ℕ, num_dogs = 3 ∧ total_cost = cost_per_animal * (num_cats + num_dogs) :=
by sorry

end NUMINAMATH_CALUDE_michaels_dogs_l2932_293221


namespace NUMINAMATH_CALUDE_polynomial_coefficient_value_l2932_293227

/-- Given a polynomial equation, prove the value of a specific coefficient -/
theorem polynomial_coefficient_value 
  (a₀ a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ a₉ a₁₀ : ℝ) :
  (∀ x : ℝ, x^2 + x^10 = a₀ + a₁*(x+1) + a₂*(x+1)^2 + a₃*(x+1)^3 + a₄*(x+1)^4 + 
    a₅*(x+1)^5 + a₆*(x+1)^6 + a₇*(x+1)^7 + a₈*(x+1)^8 + a₉*(x+1)^9 + a₁₀*(x+1)^10) →
  a₉ = -10 := by
sorry

end NUMINAMATH_CALUDE_polynomial_coefficient_value_l2932_293227


namespace NUMINAMATH_CALUDE_birch_count_l2932_293207

/-- Represents the number of trees of each species in the forest -/
structure ForestComposition where
  oak : ℕ
  pine : ℕ
  spruce : ℕ
  birch : ℕ

/-- The total number of trees in the forest -/
def total_trees : ℕ := 4000

/-- The forest composition satisfies the given conditions -/
def is_valid_composition (fc : ForestComposition) : Prop :=
  fc.oak + fc.pine + fc.spruce + fc.birch = total_trees ∧
  fc.spruce = total_trees / 10 ∧
  fc.pine = total_trees * 13 / 100 ∧
  fc.oak = fc.spruce + fc.pine

theorem birch_count (fc : ForestComposition) (h : is_valid_composition fc) : fc.birch = 2160 := by
  sorry

#check birch_count

end NUMINAMATH_CALUDE_birch_count_l2932_293207


namespace NUMINAMATH_CALUDE_multiplication_subtraction_equality_l2932_293234

theorem multiplication_subtraction_equality : 120 * 2400 - 20 * 2400 - 100 * 2400 = 0 := by
  sorry

end NUMINAMATH_CALUDE_multiplication_subtraction_equality_l2932_293234


namespace NUMINAMATH_CALUDE_man_walked_five_minutes_l2932_293263

/-- Represents the scenario of a man walking home and being picked up by his wife --/
structure WalkingScenario where
  usual_travel_time : ℕ  -- Time it usually takes to drive from station to home
  early_arrival : ℕ      -- How early the man arrived at the station (in minutes)
  time_saved : ℕ         -- How much earlier they arrived home than usual

/-- Calculates the time the man spent walking given a WalkingScenario --/
def time_spent_walking (scenario : WalkingScenario) : ℕ :=
  sorry

/-- Theorem stating that given the specific scenario, the man spent 5 minutes walking --/
theorem man_walked_five_minutes :
  let scenario : WalkingScenario := {
    usual_travel_time := 10,
    early_arrival := 60,
    time_saved := 10
  }
  time_spent_walking scenario = 5 := by
  sorry

end NUMINAMATH_CALUDE_man_walked_five_minutes_l2932_293263


namespace NUMINAMATH_CALUDE_solution_set_for_a_equals_2_range_of_a_l2932_293299

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |x - 3| - |x - a|

-- Theorem for the first part of the problem
theorem solution_set_for_a_equals_2 :
  {x : ℝ | f 2 x ≤ -1/2} = {x : ℝ | x ≥ 11/4} := by sorry

-- Theorem for the second part of the problem
theorem range_of_a :
  {a : ℝ | ∀ x, f a x ≥ a} = Set.Iic (3/2) := by sorry

end NUMINAMATH_CALUDE_solution_set_for_a_equals_2_range_of_a_l2932_293299


namespace NUMINAMATH_CALUDE_decimal_111_to_base5_l2932_293271

/-- Converts a natural number to its base-5 representation -/
def toBase5 (n : ℕ) : List ℕ :=
  if n < 5 then [n]
  else (n % 5) :: toBase5 (n / 5)

/-- Checks if a list of digits is a valid base-5 representation -/
def isValidBase5 (digits : List ℕ) : Prop :=
  digits.all (· < 5)

theorem decimal_111_to_base5 :
  let base5_repr := toBase5 111
  isValidBase5 base5_repr ∧ base5_repr = [1, 2, 4] :=
by sorry

end NUMINAMATH_CALUDE_decimal_111_to_base5_l2932_293271


namespace NUMINAMATH_CALUDE_min_distance_line_parabola_l2932_293225

/-- The minimum distance between a point on the line y = (15/8)x - 4 and a point on the parabola y = x^2 is 47/32 -/
theorem min_distance_line_parabola :
  let line := fun (x : ℝ) => (15/8) * x - 4
  let parabola := fun (x : ℝ) => x^2
  ∃ (x₁ x₂ : ℝ),
    (∀ (y₁ y₂ : ℝ),
      (line y₁ = (15/8) * y₁ - 4) →
      (parabola y₂ = y₂^2) →
      ((x₂ - x₁)^2 + (parabola x₂ - line x₁)^2)^(1/2) ≤ ((y₂ - y₁)^2 + (parabola y₂ - line y₁)^2)^(1/2)) ∧
    ((x₂ - x₁)^2 + (parabola x₂ - line x₁)^2)^(1/2) = 47/32 :=
by sorry

end NUMINAMATH_CALUDE_min_distance_line_parabola_l2932_293225


namespace NUMINAMATH_CALUDE_quadratic_roots_less_than_one_l2932_293243

theorem quadratic_roots_less_than_one (a b : ℝ) 
  (h1 : abs a + abs b < 1) 
  (h2 : a^2 - 4*b ≥ 0) : 
  ∀ x, x^2 + a*x + b = 0 → abs x < 1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_roots_less_than_one_l2932_293243


namespace NUMINAMATH_CALUDE_minimum_cubes_for_valid_assembly_l2932_293240

/-- Represents a cube with either one or two snaps -/
inductive Cube
  | SingleSnap
  | DoubleSnap

/-- An assembly of cubes -/
def Assembly := List Cube

/-- Checks if an assembly is valid (all snaps covered, only receptacles exposed) -/
def isValidAssembly : Assembly → Bool := sorry

/-- Counts the number of cubes in an assembly -/
def countCubes : Assembly → Nat := sorry

theorem minimum_cubes_for_valid_assembly :
  ∃ (a : Assembly), isValidAssembly a ∧ countCubes a = 6 ∧
  ∀ (b : Assembly), isValidAssembly b → countCubes b ≥ 6 := by
  sorry

end NUMINAMATH_CALUDE_minimum_cubes_for_valid_assembly_l2932_293240


namespace NUMINAMATH_CALUDE_point_in_third_quadrant_l2932_293237

theorem point_in_third_quadrant (a b : ℝ) : a + b < 0 → a * b > 0 → a < 0 ∧ b < 0 := by
  sorry

end NUMINAMATH_CALUDE_point_in_third_quadrant_l2932_293237


namespace NUMINAMATH_CALUDE_base_conversion_problem_l2932_293230

theorem base_conversion_problem :
  ∃! (x y z b : ℕ),
    x * b^2 + y * b + z = 1987 ∧
    x + y + z = 25 ∧
    x < b ∧ y < b ∧ z < b ∧
    b > 10 ∧
    x = 5 ∧ y = 9 ∧ z = 11 ∧ b = 19 :=
by sorry

end NUMINAMATH_CALUDE_base_conversion_problem_l2932_293230


namespace NUMINAMATH_CALUDE_employee_pay_l2932_293211

/-- Given two employees A and B with a total weekly pay of 450 and A's pay being 150% of B's,
    prove that B's weekly pay is 180. -/
theorem employee_pay (total_pay : ℝ) (a_pay : ℝ) (b_pay : ℝ) 
  (h1 : total_pay = 450)
  (h2 : a_pay = 1.5 * b_pay)
  (h3 : total_pay = a_pay + b_pay) :
  b_pay = 180 := by
  sorry

end NUMINAMATH_CALUDE_employee_pay_l2932_293211


namespace NUMINAMATH_CALUDE_room002_is_selected_l2932_293279

/-- Represents a room number in the range [1, 60] -/
def RoomNumber := Fin 60

/-- The total number of examination rooms -/
def totalRooms : Nat := 60

/-- The number of rooms to be selected for inspection -/
def selectedRooms : Nat := 12

/-- The sample interval for systematic sampling -/
def sampleInterval : Nat := totalRooms / selectedRooms

/-- Predicate to check if a room is selected in the systematic sampling -/
def isSelected (room : RoomNumber) : Prop :=
  ∃ k : Nat, (room.val + 1) = k * sampleInterval + 2

/-- Theorem stating that room 002 is selected given the conditions -/
theorem room002_is_selected :
  isSelected ⟨1, by norm_num⟩ ∧ isSelected ⟨6, by norm_num⟩ := by
  sorry


end NUMINAMATH_CALUDE_room002_is_selected_l2932_293279


namespace NUMINAMATH_CALUDE_third_term_coefficient_binomial_expansion_l2932_293295

theorem third_term_coefficient_binomial_expansion :
  let a := x
  let b := -1 / (2 * x)
  let n := 6
  let k := 2  -- Third term corresponds to k = 2
  (Nat.choose n k : ℚ) * a^(n - k) * b^k = 15/4 := by
  sorry

end NUMINAMATH_CALUDE_third_term_coefficient_binomial_expansion_l2932_293295


namespace NUMINAMATH_CALUDE_simple_interest_rate_percent_l2932_293202

/-- Given simple interest conditions, prove the rate percent -/
theorem simple_interest_rate_percent 
  (P : ℝ) (SI : ℝ) (T : ℝ) 
  (h_P : P = 900) 
  (h_SI : SI = 160) 
  (h_T : T = 4) 
  (h_formula : SI = (P * R * T) / 100) : 
  R = 400 / 90 := by
  sorry

end NUMINAMATH_CALUDE_simple_interest_rate_percent_l2932_293202


namespace NUMINAMATH_CALUDE_xiao_yun_age_l2932_293293

theorem xiao_yun_age :
  ∀ (x : ℕ),
  (∃ (f : ℕ), f = x + 25) →
  (x + 25 + 5 = 2 * (x + 5) - 10) →
  x = 30 := by
sorry

end NUMINAMATH_CALUDE_xiao_yun_age_l2932_293293


namespace NUMINAMATH_CALUDE_quadratic_equation_root_l2932_293242

theorem quadratic_equation_root (k : ℚ) : 
  (∃ x : ℚ, x - x^2 = k*x^2 + 1) → 
  (2 - 2^2 = k*2^2 + 1) → 
  k = -3/4 := by
sorry

end NUMINAMATH_CALUDE_quadratic_equation_root_l2932_293242


namespace NUMINAMATH_CALUDE_golf_cart_capacity_l2932_293285

/-- The number of patrons that can fit in a golf cart -/
def patrons_per_cart (patrons_from_cars : ℕ) (patrons_from_bus : ℕ) (total_carts : ℕ) : ℕ :=
  (patrons_from_cars + patrons_from_bus) / total_carts

/-- Theorem: Given the conditions from the problem, prove that 3 patrons can fit in a golf cart -/
theorem golf_cart_capacity :
  patrons_per_cart 12 27 13 = 3 := by
  sorry

end NUMINAMATH_CALUDE_golf_cart_capacity_l2932_293285


namespace NUMINAMATH_CALUDE_angle_A_measure_l2932_293228

-- Define the triangle ABC
structure Triangle :=
  (A B C : ℝ)
  (sum_angles : A + B + C = 180)
  (positive_angles : 0 < A ∧ 0 < B ∧ 0 < C)

-- Theorem statement
theorem angle_A_measure (abc : Triangle) (h1 : abc.C = 3 * abc.B) (h2 : abc.B = 15) : 
  abc.A = 120 := by
sorry

end NUMINAMATH_CALUDE_angle_A_measure_l2932_293228


namespace NUMINAMATH_CALUDE_average_age_increase_l2932_293257

theorem average_age_increase (initial_count : ℕ) (replaced_age1 replaced_age2 : ℕ) 
  (new_average : ℕ) (h1 : initial_count = 8) (h2 : replaced_age1 = 21) 
  (h3 : replaced_age2 = 23) (h4 : new_average = 30) : 
  let initial_total := initial_count * (initial_count * A - replaced_age1 - replaced_age2) / initial_count
  let new_total := initial_total - replaced_age1 - replaced_age2 + 2 * new_average
  let new_average := new_total / initial_count
  new_average - (initial_total / initial_count) = 2 := by
  sorry

end NUMINAMATH_CALUDE_average_age_increase_l2932_293257


namespace NUMINAMATH_CALUDE_lcm_45_75_180_l2932_293292

theorem lcm_45_75_180 : Nat.lcm 45 (Nat.lcm 75 180) = 900 := by
  sorry

end NUMINAMATH_CALUDE_lcm_45_75_180_l2932_293292


namespace NUMINAMATH_CALUDE_repair_cost_is_5000_l2932_293255

/-- Calculates the repair cost for a machine given its purchase price, transportation cost, selling price, and profit percentage. -/
def repair_cost (purchase_price transportation_cost selling_price profit_percentage : ℕ) : ℕ :=
  let total_cost := purchase_price + transportation_cost + (selling_price * 100 / (100 + profit_percentage) - purchase_price - transportation_cost)
  selling_price * 100 / (100 + profit_percentage) - purchase_price - transportation_cost

/-- Theorem stating that for the given conditions, the repair cost is 5000. -/
theorem repair_cost_is_5000 :
  repair_cost 10000 1000 24000 50 = 5000 := by
  sorry

end NUMINAMATH_CALUDE_repair_cost_is_5000_l2932_293255


namespace NUMINAMATH_CALUDE_function_properties_l2932_293214

-- Define the function f
def f (a c : ℕ) (x : ℝ) : ℝ := a * x^2 + 2 * x + c

-- State the theorem
theorem function_properties (a c : ℕ) (h1 : a ≠ 0) (h2 : c ≠ 0) :
  (f a c 1 = 5) →
  (6 < f a c 2 ∧ f a c 2 < 11) →
  (a = 1 ∧ c = 2) ∧
  (∀ m : ℝ, (∀ x : ℝ, f a c x - 2 * m * x ≤ 1) → m ≥ 1) :=
by sorry

end NUMINAMATH_CALUDE_function_properties_l2932_293214


namespace NUMINAMATH_CALUDE_train_length_l2932_293277

/-- Given a train that crosses a platform in 39 seconds, crosses a signal pole in 20 seconds,
    and the platform length is 285 meters, the length of the train is 300 meters. -/
theorem train_length (crossing_time_platform : ℝ) (crossing_time_pole : ℝ) (platform_length : ℝ)
    (h1 : crossing_time_platform = 39)
    (h2 : crossing_time_pole = 20)
    (h3 : platform_length = 285) :
    ∃ train_length : ℝ, train_length = 300 := by
  sorry

end NUMINAMATH_CALUDE_train_length_l2932_293277


namespace NUMINAMATH_CALUDE_min_width_for_rectangular_area_l2932_293223

theorem min_width_for_rectangular_area :
  ∀ w : ℝ,
  w > 0 →
  w * (w + 18) ≥ 150 →
  (∀ x : ℝ, x > 0 ∧ x * (x + 18) ≥ 150 → x ≥ w) →
  w = 6 :=
by sorry

end NUMINAMATH_CALUDE_min_width_for_rectangular_area_l2932_293223


namespace NUMINAMATH_CALUDE_matthew_cake_division_l2932_293249

/-- Given that Matthew has 30 cakes and 2 friends, prove that each friend receives 15 cakes when the cakes are divided equally. -/
theorem matthew_cake_division (total_cakes : ℕ) (num_friends : ℕ) (cakes_per_friend : ℕ) :
  total_cakes = 30 →
  num_friends = 2 →
  cakes_per_friend = total_cakes / num_friends →
  cakes_per_friend = 15 := by
  sorry

end NUMINAMATH_CALUDE_matthew_cake_division_l2932_293249


namespace NUMINAMATH_CALUDE_parabolas_same_vertex_l2932_293219

/-- 
Two parabolas have the same vertex if and only if their coefficients satisfy specific relations.
-/
theorem parabolas_same_vertex (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hab : a ≠ b) :
  (∃ (x y : ℝ), 
    (x = -b / (2 * a) ∧ y = a * x^2 + b * x + c) ∧
    (x = -c / (2 * b) ∧ y = b * x^2 + c * x + a)) ↔
  (b = -2 * a ∧ c = 4 * a) :=
sorry

end NUMINAMATH_CALUDE_parabolas_same_vertex_l2932_293219


namespace NUMINAMATH_CALUDE_probability_neither_prime_nor_composite_l2932_293297

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → n % m ≠ 0

def is_composite (n : ℕ) : Prop := n > 1 ∧ ∃ m : ℕ, m > 1 ∧ m < n ∧ n % m = 0

theorem probability_neither_prime_nor_composite :
  let S := Finset.range 97
  let E := {1}
  (Finset.card E : ℚ) / (Finset.card S : ℚ) = 1 / 97 :=
by sorry

end NUMINAMATH_CALUDE_probability_neither_prime_nor_composite_l2932_293297


namespace NUMINAMATH_CALUDE_opposite_numbers_abs_l2932_293216

theorem opposite_numbers_abs (m n : ℝ) : m + n = 0 → |m + n - 1| = 1 := by sorry

end NUMINAMATH_CALUDE_opposite_numbers_abs_l2932_293216


namespace NUMINAMATH_CALUDE_f_intersects_x_axis_iff_l2932_293231

/-- A function that represents (k-3)x^2+2x+1 --/
def f (k : ℝ) (x : ℝ) : ℝ := (k - 3) * x^2 + 2 * x + 1

/-- Predicate to check if a function intersects the x-axis --/
def intersects_x_axis (g : ℝ → ℝ) : Prop :=
  ∃ x, g x = 0

/-- Theorem stating that f intersects the x-axis iff k ≤ 4 --/
theorem f_intersects_x_axis_iff (k : ℝ) :
  intersects_x_axis (f k) ↔ k ≤ 4 := by
  sorry

end NUMINAMATH_CALUDE_f_intersects_x_axis_iff_l2932_293231


namespace NUMINAMATH_CALUDE_quadratic_root_condition_l2932_293298

theorem quadratic_root_condition (m : ℝ) : 
  (∃ x₁ x₂ : ℝ, x₁ < 1 ∧ x₂ > 1 ∧ 
   (∀ x : ℝ, x^2 + 2*(m-1)*x - 5*m - 2 = 0 ↔ (x = x₁ ∨ x = x₂))) 
  ↔ m > 1 :=
sorry

end NUMINAMATH_CALUDE_quadratic_root_condition_l2932_293298


namespace NUMINAMATH_CALUDE_coinciding_rest_days_l2932_293232

/-- Chris's schedule cycle length -/
def chris_cycle : ℕ := 6

/-- Dana's schedule cycle length -/
def dana_cycle : ℕ := 7

/-- Total number of days -/
def total_days : ℕ := 1000

/-- Chris's rest days within his cycle -/
def chris_rest_days : List ℕ := [5, 6]

/-- Dana's rest days within her cycle -/
def dana_rest_days : List ℕ := [6, 7]

/-- The number of coinciding rest days for Chris and Dana in the first 1000 days -/
theorem coinciding_rest_days : 
  (List.filter (λ d : ℕ => 
    (d % chris_cycle ∈ chris_rest_days) ∧ 
    (d % dana_cycle ∈ dana_rest_days)) 
    (List.range total_days)).length = 23 := by
  sorry

end NUMINAMATH_CALUDE_coinciding_rest_days_l2932_293232


namespace NUMINAMATH_CALUDE_christen_peeled_21_potatoes_l2932_293226

/-- Calculates the number of potatoes Christen peeled --/
def christenPotatoesCount (totalPotatoes : ℕ) (homerRate : ℕ) (christenRate : ℕ) (timeBeforeJoin : ℕ) : ℕ :=
  let homerInitialPotatoes := homerRate * timeBeforeJoin
  let remainingPotatoes := totalPotatoes - homerInitialPotatoes
  let combinedRate := homerRate + christenRate
  let timeAfterJoin := remainingPotatoes / combinedRate
  christenRate * timeAfterJoin

theorem christen_peeled_21_potatoes :
  christenPotatoesCount 60 4 6 6 = 21 := by
  sorry

end NUMINAMATH_CALUDE_christen_peeled_21_potatoes_l2932_293226


namespace NUMINAMATH_CALUDE_exponential_function_fixed_point_l2932_293206

theorem exponential_function_fixed_point (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  let f : ℝ → ℝ := λ x ↦ a^x + 1
  f 0 = 2 := by
sorry

end NUMINAMATH_CALUDE_exponential_function_fixed_point_l2932_293206


namespace NUMINAMATH_CALUDE_unique_solution_l2932_293276

def equation (x y : ℤ) : Prop := 3 * x + y = 10

theorem unique_solution :
  (equation 2 4) ∧
  ¬(equation 1 6) ∧
  ¬(equation (-2) 12) ∧
  ¬(equation (-1) 11) :=
sorry

end NUMINAMATH_CALUDE_unique_solution_l2932_293276


namespace NUMINAMATH_CALUDE_total_paths_is_14_l2932_293261

/- Define the number of paths between points -/
def paths_AB : Nat := 2
def paths_BC : Nat := 2
def paths_CD : Nat := 2
def direct_AC : Nat := 1
def direct_BD : Nat := 2

/- Define the total number of paths from A to D -/
def total_paths : Nat :=
  paths_AB * paths_BC * paths_CD +  -- Paths through B and C
  direct_AC * paths_CD +            -- Direct path to C, then to D
  paths_AB * direct_BD              -- Path to B, then direct to D

/- Theorem statement -/
theorem total_paths_is_14 : total_paths = 14 := by
  sorry

end NUMINAMATH_CALUDE_total_paths_is_14_l2932_293261


namespace NUMINAMATH_CALUDE_toy_price_after_discounts_l2932_293236

theorem toy_price_after_discounts (initial_price : ℝ) (discount : ℝ) : 
  initial_price = 200 → discount = 0.1 → 
  initial_price * (1 - discount)^2 = 162 := by
  sorry

#eval (200 : ℝ) * (1 - 0.1)^2

end NUMINAMATH_CALUDE_toy_price_after_discounts_l2932_293236


namespace NUMINAMATH_CALUDE_c_5_value_l2932_293281

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ (r : ℝ), ∀ n : ℕ, a (n + 1) = r * a n

theorem c_5_value (c : ℕ → ℝ) :
  geometric_sequence (λ n => Real.sqrt (c n)) ∧
  Real.sqrt (c 1) = 1 ∧
  Real.sqrt (c 2) = 2 * Real.sqrt (c 1) →
  c 5 = 256 := by
sorry

end NUMINAMATH_CALUDE_c_5_value_l2932_293281


namespace NUMINAMATH_CALUDE_max_blocks_fit_l2932_293262

/-- Represents the dimensions of a rectangular box -/
structure BoxDimensions where
  length : ℕ
  width : ℕ
  height : ℕ

/-- Calculates the volume of a box given its dimensions -/
def volume (b : BoxDimensions) : ℕ :=
  b.length * b.width * b.height

/-- The dimensions of the larger box -/
def largeBox : BoxDimensions :=
  { length := 4, width := 3, height := 3 }

/-- The dimensions of the smaller block -/
def smallBlock : BoxDimensions :=
  { length := 3, width := 2, height := 1 }

/-- Theorem: The maximum number of small blocks that can fit in the large box is 6 -/
theorem max_blocks_fit : 
  (volume largeBox) / (volume smallBlock) = 6 ∧ 
  (2 * smallBlock.length ≤ largeBox.length) ∧
  (smallBlock.width ≤ largeBox.width) ∧
  (2 * smallBlock.height ≤ largeBox.height) := by
  sorry

end NUMINAMATH_CALUDE_max_blocks_fit_l2932_293262


namespace NUMINAMATH_CALUDE_isosceles_trapezoid_base_ratio_l2932_293205

structure IsoscelesTrapezoid where
  shorter_base : ℝ
  longer_base : ℝ
  height : ℝ
  diagonal : ℝ
  is_positive : shorter_base > 0 ∧ longer_base > 0 ∧ height > 0 ∧ diagonal > 0
  shorter_base_eq_height : shorter_base = height
  longer_base_eq_diagonal : longer_base = diagonal

theorem isosceles_trapezoid_base_ratio 
  (t : IsoscelesTrapezoid) : t.shorter_base / t.longer_base = 3 / 5 := by
  sorry

end NUMINAMATH_CALUDE_isosceles_trapezoid_base_ratio_l2932_293205


namespace NUMINAMATH_CALUDE_sqrt_product_plus_factorial_equals_1114_l2932_293264

theorem sqrt_product_plus_factorial_equals_1114 : 
  Real.sqrt ((35 * 34 * 33 * 32) + 24) = 1114 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_product_plus_factorial_equals_1114_l2932_293264


namespace NUMINAMATH_CALUDE_students_in_no_subjects_l2932_293229

theorem students_in_no_subjects (total : ℕ) (math chem bio : ℕ) (math_chem chem_bio math_bio : ℕ) (all_three : ℕ) : 
  total = 120 →
  math = 70 →
  chem = 50 →
  bio = 40 →
  math_chem = 30 →
  chem_bio = 20 →
  math_bio = 10 →
  all_three = 5 →
  total - (math + chem + bio - math_chem - chem_bio - math_bio + all_three) = 20 :=
by sorry

end NUMINAMATH_CALUDE_students_in_no_subjects_l2932_293229


namespace NUMINAMATH_CALUDE_factor_expression_l2932_293256

theorem factor_expression (x : ℝ) : x * (x + 2) + (x + 2) = (x + 1) * (x + 2) := by
  sorry

end NUMINAMATH_CALUDE_factor_expression_l2932_293256


namespace NUMINAMATH_CALUDE_fifth_term_is_negative_three_l2932_293215

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℤ) : Prop :=
  ∃ d : ℤ, ∀ n : ℕ, a (n + 1) = a n + d

/-- The 5th term of the sequence is -3 -/
theorem fifth_term_is_negative_three
  (a : ℕ → ℤ)
  (h_arith : arithmetic_sequence a)
  (h_third : a 3 = -5)
  (h_ninth : a 9 = 1) :
  a 5 = -3 := by
sorry

end NUMINAMATH_CALUDE_fifth_term_is_negative_three_l2932_293215


namespace NUMINAMATH_CALUDE_can_capacity_l2932_293284

/-- Represents the contents of a can with milk and water -/
structure CanContents where
  milk : ℝ
  water : ℝ

/-- Represents a can with its contents and capacity -/
structure Can where
  contents : CanContents
  capacity : ℝ

/-- The theorem stating the capacity of the can given the conditions -/
theorem can_capacity (initial : CanContents) (final : CanContents) : 
  (initial.milk / initial.water = 5 / 3) →
  (final.milk / final.water = 2 / 1) →
  (final.milk = initial.milk + 8) →
  (final.water = initial.water) →
  (∃ (can : Can), can.contents = final ∧ can.capacity = 72) :=
by sorry

end NUMINAMATH_CALUDE_can_capacity_l2932_293284


namespace NUMINAMATH_CALUDE_barbara_typing_speed_reduction_l2932_293270

/-- Calculates the reduction in typing speed given the original speed, document length, and typing time. -/
def typing_speed_reduction (original_speed : ℕ) (document_length : ℕ) (typing_time : ℕ) : ℕ :=
  original_speed - (document_length / typing_time)

/-- Theorem stating that Barbara's typing speed has reduced by 40 words per minute. -/
theorem barbara_typing_speed_reduction :
  typing_speed_reduction 212 3440 20 = 40 := by
  sorry

#eval typing_speed_reduction 212 3440 20

end NUMINAMATH_CALUDE_barbara_typing_speed_reduction_l2932_293270


namespace NUMINAMATH_CALUDE_oatmeal_cookies_count_l2932_293246

/-- Represents the number of cookies in each baggie -/
def cookies_per_baggie : ℕ := 3

/-- Represents the number of chocolate chip cookies Maria had -/
def chocolate_chip_cookies : ℕ := 2

/-- Represents the number of baggies Maria could make -/
def total_baggies : ℕ := 6

/-- Theorem stating the number of oatmeal cookies Maria had -/
theorem oatmeal_cookies_count : 
  (total_baggies * cookies_per_baggie) - chocolate_chip_cookies = 16 := by
  sorry

end NUMINAMATH_CALUDE_oatmeal_cookies_count_l2932_293246


namespace NUMINAMATH_CALUDE_simplify_expression_l2932_293245

theorem simplify_expression (b : ℝ) (h : b ≠ 1) :
  1 - (1 / (1 + b / (1 - b))) = b := by sorry

end NUMINAMATH_CALUDE_simplify_expression_l2932_293245


namespace NUMINAMATH_CALUDE_sum_difference_equals_eight_ninths_l2932_293259

open BigOperators

-- Define the harmonic series
def harmonic_sum (n : ℕ) : ℚ := ∑ y in Finset.range n, (1 : ℚ) / (y + 1)

-- State the theorem
theorem sum_difference_equals_eight_ninths :
  (∑ y in Finset.range 8, (1 : ℚ) / (y + 1)) - (∑ y in Finset.range 8, (1 : ℚ) / (y + 2)) = 8 / 9 := by
  sorry

end NUMINAMATH_CALUDE_sum_difference_equals_eight_ninths_l2932_293259


namespace NUMINAMATH_CALUDE_value_of_D_l2932_293288

theorem value_of_D (A B C D : ℝ) 
  (h1 : A + A = 6)
  (h2 : B - A = 4)
  (h3 : C + B = 9)
  (h4 : D - C = 7)
  (h5 : A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D) : 
  D = 9 := by
sorry

end NUMINAMATH_CALUDE_value_of_D_l2932_293288


namespace NUMINAMATH_CALUDE_monthly_compound_greater_than_annual_l2932_293275

def annual_rate : ℝ := 0.03

theorem monthly_compound_greater_than_annual (t : ℝ) (h : t > 0) :
  (1 + annual_rate)^t < (1 + annual_rate / 12)^(12 * t) := by
  sorry


end NUMINAMATH_CALUDE_monthly_compound_greater_than_annual_l2932_293275


namespace NUMINAMATH_CALUDE_d_is_nonzero_l2932_293283

/-- A polynomial of degree 5 with six distinct x-intercepts, including 0 and -1 -/
def Q (a b c d : ℝ) (x : ℝ) : ℝ := x^5 + a*x^4 + b*x^3 + c*x^2 + d*x

/-- The property that Q has six distinct x-intercepts, including 0 and -1 -/
def has_six_distinct_intercepts (a b c d : ℝ) : Prop :=
  ∃ p q r s : ℝ, p ≠ q ∧ p ≠ r ∧ p ≠ s ∧ q ≠ r ∧ q ≠ s ∧ r ≠ s ∧
              p ≠ 0 ∧ q ≠ 0 ∧ r ≠ 0 ∧ s ≠ 0 ∧
              p ≠ -1 ∧ q ≠ -1 ∧ r ≠ -1 ∧ s ≠ -1 ∧
              ∀ x : ℝ, Q a b c d x = 0 ↔ x = 0 ∨ x = -1 ∨ x = p ∨ x = q ∨ x = r ∨ x = s

theorem d_is_nonzero (a b c d : ℝ) (h : has_six_distinct_intercepts a b c d) : d ≠ 0 :=
sorry

end NUMINAMATH_CALUDE_d_is_nonzero_l2932_293283


namespace NUMINAMATH_CALUDE_sophie_laundry_loads_l2932_293280

/-- Represents the cost of a box of dryer sheets in dollars -/
def box_cost : ℚ := 5.5

/-- Represents the number of dryer sheets in a box -/
def sheets_per_box : ℕ := 104

/-- Represents the amount saved in a year by not buying dryer sheets, in dollars -/
def yearly_savings : ℚ := 11

/-- Represents the number of weeks in a year -/
def weeks_per_year : ℕ := 52

/-- Represents the number of dryer sheets used per load of laundry -/
def sheets_per_load : ℕ := 1

/-- Theorem stating that Sophie does 4 loads of laundry per week -/
theorem sophie_laundry_loads : 
  ∃ (loads_per_week : ℕ), 
    loads_per_week = 4 ∧ 
    (yearly_savings / box_cost : ℚ) * sheets_per_box = loads_per_week * weeks_per_year :=
sorry

end NUMINAMATH_CALUDE_sophie_laundry_loads_l2932_293280


namespace NUMINAMATH_CALUDE_remainder_of_3_pow_500_mod_7_l2932_293287

theorem remainder_of_3_pow_500_mod_7 : 3^500 % 7 = 2 := by sorry

end NUMINAMATH_CALUDE_remainder_of_3_pow_500_mod_7_l2932_293287


namespace NUMINAMATH_CALUDE_bridge_length_l2932_293274

/-- Given a train of length 150 meters traveling at 45 km/hr that crosses a bridge in 30 seconds,
    the length of the bridge is 225 meters. -/
theorem bridge_length (train_length : ℝ) (train_speed_kmh : ℝ) (crossing_time : ℝ) :
  train_length = 150 ∧ 
  train_speed_kmh = 45 ∧ 
  crossing_time = 30 →
  (train_speed_kmh * 1000 / 3600) * crossing_time - train_length = 225 := by
  sorry

#check bridge_length

end NUMINAMATH_CALUDE_bridge_length_l2932_293274


namespace NUMINAMATH_CALUDE_snooker_tournament_ticket_sales_l2932_293238

/-- Proves that the total number of tickets sold is 336 given the specified conditions -/
theorem snooker_tournament_ticket_sales
  (vip_cost : ℕ)
  (general_cost : ℕ)
  (total_revenue : ℕ)
  (ticket_difference : ℕ)
  (h1 : vip_cost = 45)
  (h2 : general_cost = 20)
  (h3 : total_revenue = 7500)
  (h4 : ticket_difference = 276)
  (h5 : ∃ (vip general : ℕ),
    vip_cost * vip + general_cost * general = total_revenue ∧
    vip + ticket_difference = general) :
  ∃ (vip general : ℕ), vip + general = 336 := by
  sorry

end NUMINAMATH_CALUDE_snooker_tournament_ticket_sales_l2932_293238


namespace NUMINAMATH_CALUDE_max_red_socks_l2932_293244

theorem max_red_socks (r b : ℕ) : 
  let t := r + b
  r + b ≤ 2000 →
  (r * (r - 1) + b * (b - 1)) / (t * (t - 1)) = 5 / 12 →
  r ≤ 109 :=
by sorry

end NUMINAMATH_CALUDE_max_red_socks_l2932_293244


namespace NUMINAMATH_CALUDE_quadratic_root_relation_l2932_293235

theorem quadratic_root_relation (a b c : ℝ) (h : a ≠ 0) :
  (∃ x y : ℝ, x ≠ y ∧ a * x^2 + b * x + c = 0 ∧ a * y^2 + b * y + c = 0 ∧ y = 3 * x) →
  3 * b^2 = 16 * a * c :=
by sorry

end NUMINAMATH_CALUDE_quadratic_root_relation_l2932_293235


namespace NUMINAMATH_CALUDE_texas_maine_plate_difference_l2932_293218

/-- The number of possible choices for a letter in a license plate. -/
def letter_choices : ℕ := 26

/-- The number of possible choices for a number in a license plate. -/
def number_choices : ℕ := 10

/-- The number of possible license plates in Texas format (LLNNNNL). -/
def texas_plates : ℕ := letter_choices^3 * number_choices^4

/-- The number of possible license plates in Maine format (LLLNNN). -/
def maine_plates : ℕ := letter_choices^3 * number_choices^3

/-- The difference in the number of possible license plates between Texas and Maine. -/
def plate_difference : ℕ := texas_plates - maine_plates

theorem texas_maine_plate_difference :
  plate_difference = 158184000 := by
  sorry

end NUMINAMATH_CALUDE_texas_maine_plate_difference_l2932_293218


namespace NUMINAMATH_CALUDE_sphere_volume_from_surface_area_l2932_293233

theorem sphere_volume_from_surface_area :
  ∀ (r : ℝ), 
    r > 0 → 
    4 * Real.pi * r^2 = 36 * Real.pi → 
    (4 / 3) * Real.pi * r^3 = 36 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_sphere_volume_from_surface_area_l2932_293233


namespace NUMINAMATH_CALUDE_complex_modulus_problem_l2932_293268

theorem complex_modulus_problem (z : ℂ) : 
  z * (1 + Complex.I) = 4 - 2 * Complex.I → Complex.abs z = Real.sqrt 10 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_problem_l2932_293268


namespace NUMINAMATH_CALUDE_store_profit_theorem_l2932_293200

/-- Represents the store's sales and profit model -/
structure Store where
  initial_profit_per_unit : ℝ
  initial_daily_sales : ℝ
  sales_increase_per_yuan : ℝ

/-- Calculates the new daily sales after a price reduction -/
def new_daily_sales (s : Store) (price_reduction : ℝ) : ℝ :=
  s.initial_daily_sales + s.sales_increase_per_yuan * price_reduction

/-- Calculates the daily profit after a price reduction -/
def daily_profit (s : Store) (price_reduction : ℝ) : ℝ :=
  (s.initial_profit_per_unit - price_reduction) * (new_daily_sales s price_reduction)

/-- The store model based on the given conditions -/
def my_store : Store := {
  initial_profit_per_unit := 60,
  initial_daily_sales := 40,
  sales_increase_per_yuan := 2
}

theorem store_profit_theorem (s : Store) :
  (new_daily_sales s 10 = 60) ∧
  (∃ x : ℝ, x = 30 ∧ daily_profit s x = 3000) ∧
  (¬ ∃ y : ℝ, daily_profit s y = 3300) := by
  sorry

#check store_profit_theorem my_store

end NUMINAMATH_CALUDE_store_profit_theorem_l2932_293200


namespace NUMINAMATH_CALUDE_ellipse_equation_correct_l2932_293291

/-- An ellipse with foci (-1,0) and (1,0), passing through (2,0) -/
def Ellipse : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 / 4 + p.2^2 / 3 = 1}

/-- The foci of the ellipse -/
def foci : (ℝ × ℝ) × (ℝ × ℝ) := ((-1, 0), (1, 0))

/-- A point on the ellipse -/
def P : ℝ × ℝ := (2, 0)

theorem ellipse_equation_correct :
  (∀ p ∈ Ellipse, 
    (Real.sqrt ((p.1 - foci.1.1)^2 + (p.2 - foci.1.2)^2) + 
     Real.sqrt ((p.1 - foci.2.1)^2 + (p.2 - foci.2.2)^2)) = 
    (Real.sqrt ((P.1 - foci.1.1)^2 + (P.2 - foci.1.2)^2) + 
     Real.sqrt ((P.1 - foci.2.1)^2 + (P.2 - foci.2.2)^2))) ∧
  P ∈ Ellipse := by
  sorry

#check ellipse_equation_correct

end NUMINAMATH_CALUDE_ellipse_equation_correct_l2932_293291


namespace NUMINAMATH_CALUDE_power_fraction_simplification_l2932_293203

theorem power_fraction_simplification :
  (3^2020 - 3^2018) / (3^2020 + 3^2018) = 4/5 := by
  sorry

end NUMINAMATH_CALUDE_power_fraction_simplification_l2932_293203


namespace NUMINAMATH_CALUDE_show_attendance_l2932_293258

theorem show_attendance (adult_price children_price total_receipts : ℚ)
  (h1 : adult_price = 5.5)
  (h2 : children_price = 2.5)
  (h3 : total_receipts = 1026) :
  ∃ (adults children : ℕ),
    adults = 2 * children ∧
    adult_price * adults + children_price * children = total_receipts ∧
    adults = 152 :=
by sorry

end NUMINAMATH_CALUDE_show_attendance_l2932_293258


namespace NUMINAMATH_CALUDE_hours_per_week_calculation_l2932_293204

/-- Calculates the number of hours per week needed to earn a target amount given initial work conditions and duration --/
def hours_per_week (initial_hours : ℕ) (initial_weeks : ℕ) (initial_earnings : ℕ) 
                   (target_earnings : ℕ) (target_weeks : ℕ) : ℚ :=
  let hourly_wage : ℚ := initial_earnings / (initial_hours * initial_weeks)
  let total_hours_needed : ℚ := target_earnings / hourly_wage
  total_hours_needed / target_weeks

theorem hours_per_week_calculation :
  hours_per_week 40 10 4000 8000 50 = 16 := by sorry

end NUMINAMATH_CALUDE_hours_per_week_calculation_l2932_293204


namespace NUMINAMATH_CALUDE_oranges_left_l2932_293266

/-- The number of oranges originally in the basket -/
def original_oranges : ℕ := 8

/-- The number of oranges taken from the basket -/
def oranges_taken : ℕ := 5

/-- Theorem: The number of oranges left in the basket is 3 -/
theorem oranges_left : original_oranges - oranges_taken = 3 := by
  sorry

end NUMINAMATH_CALUDE_oranges_left_l2932_293266


namespace NUMINAMATH_CALUDE_circle_inequality_l2932_293286

theorem circle_inequality (a b c d : ℝ) (x₁ y₁ x₂ y₂ x₃ y₃ x₄ y₄ : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0)
  (hab : a * b + c * d = 1)
  (h1 : x₁^2 + y₁^2 = 1) (h2 : x₂^2 + y₂^2 = 1) 
  (h3 : x₃^2 + y₃^2 = 1) (h4 : x₄^2 + y₄^2 = 1) :
  (a*y₁ + b*y₂ + c*y₃ + d*y₄)^2 + (a*x₄ + b*x₃ + c*x₂ + d*x₁)^2 
    ≤ 2*((a^2 + b^2)/(a*b) + (c^2 + d^2)/(c*d)) := by
  sorry

end NUMINAMATH_CALUDE_circle_inequality_l2932_293286


namespace NUMINAMATH_CALUDE_maxwell_brad_meeting_time_l2932_293208

theorem maxwell_brad_meeting_time 
  (distance : ℝ) 
  (maxwell_speed : ℝ) 
  (brad_speed : ℝ) 
  (maxwell_head_start : ℝ) :
  distance = 34 →
  maxwell_speed = 4 →
  brad_speed = 6 →
  maxwell_head_start = 1 →
  ∃ (total_time : ℝ), 
    total_time = 4 ∧
    maxwell_speed * total_time + brad_speed * (total_time - maxwell_head_start) = distance :=
by
  sorry

end NUMINAMATH_CALUDE_maxwell_brad_meeting_time_l2932_293208


namespace NUMINAMATH_CALUDE_largest_value_with_brackets_l2932_293296

def original_expression : List Int := [2, 0, 1, 9]

def insert_brackets (expr : List Int) (pos : Nat) : Int :=
  match pos with
  | 0 => (expr[0]! - expr[1]!) - expr[2]! - expr[3]!
  | 1 => expr[0]! - (expr[1]! - expr[2]!) - expr[3]!
  | 2 => expr[0]! - expr[1]! - (expr[2]! - expr[3]!)
  | _ => 0

def all_possible_values (expr : List Int) : List Int :=
  [insert_brackets expr 0, insert_brackets expr 1, insert_brackets expr 2]

theorem largest_value_with_brackets :
  (all_possible_values original_expression).maximum? = some 10 := by
  sorry

end NUMINAMATH_CALUDE_largest_value_with_brackets_l2932_293296


namespace NUMINAMATH_CALUDE_max_value_fraction_l2932_293213

theorem max_value_fraction (x : ℝ) : 
  (3 * x^2 + 9 * x + 17) / (3 * x^2 + 9 * x + 7) ≤ 41 ∧ 
  ∃ y : ℝ, (3 * y^2 + 9 * y + 17) / (3 * y^2 + 9 * y + 7) = 41 := by
  sorry

end NUMINAMATH_CALUDE_max_value_fraction_l2932_293213


namespace NUMINAMATH_CALUDE_divisor_problem_l2932_293212

theorem divisor_problem (n : ℕ) (d : ℕ) (h1 : n > 0) (h2 : ∃ k : ℕ, n + 1 = k * d + 4) (h3 : ∃ m : ℕ, n = 2 * m + 1) : d = 6 := by
  sorry

end NUMINAMATH_CALUDE_divisor_problem_l2932_293212
