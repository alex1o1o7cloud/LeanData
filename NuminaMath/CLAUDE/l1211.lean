import Mathlib

namespace NUMINAMATH_CALUDE_triangle_angle_c_value_l1211_121112

theorem triangle_angle_c_value (A B C x : ℝ) : 
  A = 45 ∧ B = 3 * x ∧ C = (1 / 2) * B ∧ A + B + C = 180 → C = 45 := by
  sorry

end NUMINAMATH_CALUDE_triangle_angle_c_value_l1211_121112


namespace NUMINAMATH_CALUDE_quadratic_real_roots_condition_l1211_121180

/-- Defines the quadratic equation kx^2 - x - 1 = 0 -/
def quadratic_equation (k x : ℝ) : Prop :=
  k * x^2 - x - 1 = 0

/-- Defines when a quadratic equation has real roots -/
def has_real_roots (k : ℝ) : Prop :=
  ∃ x : ℝ, quadratic_equation k x

/-- Theorem stating the condition for the quadratic equation to have real roots -/
theorem quadratic_real_roots_condition (k : ℝ) :
  has_real_roots k ↔ k ≥ -1/4 ∧ k ≠ 0 :=
sorry

end NUMINAMATH_CALUDE_quadratic_real_roots_condition_l1211_121180


namespace NUMINAMATH_CALUDE_kati_age_l1211_121116

/-- Represents a person's age and birthday information -/
structure Person where
  age : ℕ
  birthdays : ℕ

/-- Represents the family members -/
structure Family where
  kati : Person
  brother : Person
  grandfather : Person

/-- The conditions of the problem -/
def problem_conditions (f : Family) : Prop :=
  f.kati.age = f.grandfather.birthdays ∧
  f.kati.age + f.brother.age + f.grandfather.age = 111 ∧
  f.kati.age > f.brother.age ∧
  f.kati.age - f.brother.age < 4 ∧
  f.grandfather.age = 4 * f.grandfather.birthdays + (f.grandfather.age % 4)

/-- The theorem to prove -/
theorem kati_age (f : Family) : 
  problem_conditions f → f.kati.age = 18 :=
by
  sorry

end NUMINAMATH_CALUDE_kati_age_l1211_121116


namespace NUMINAMATH_CALUDE_base_conversion_1234_to_base_4_l1211_121168

theorem base_conversion_1234_to_base_4 :
  (3 * 4^4 + 4 * 4^3 + 1 * 4^2 + 0 * 4^1 + 2 * 4^0) = 1234 := by
  sorry

end NUMINAMATH_CALUDE_base_conversion_1234_to_base_4_l1211_121168


namespace NUMINAMATH_CALUDE_joggers_regain_sight_main_proof_l1211_121101

/-- The time it takes for two joggers to regain sight of each other after being obscured by a circular stadium --/
theorem joggers_regain_sight (steven_speed linda_speed : ℝ) 
  (path_distance stadium_diameter : ℝ) (initial_distance : ℝ) : ℝ :=
  let t : ℝ := 225
  sorry

/-- The main theorem that proves the time is 225 seconds --/
theorem main_proof : joggers_regain_sight 4 2 300 200 300 = 225 := by
  sorry

end NUMINAMATH_CALUDE_joggers_regain_sight_main_proof_l1211_121101


namespace NUMINAMATH_CALUDE_coin_collection_value_l1211_121144

theorem coin_collection_value (total_coins : ℕ) (sample_coins : ℕ) (sample_value : ℚ) :
  total_coins = 15 →
  sample_coins = 5 →
  sample_value = 12 →
  (total_coins : ℚ) * (sample_value / sample_coins) = 36 :=
by sorry

end NUMINAMATH_CALUDE_coin_collection_value_l1211_121144


namespace NUMINAMATH_CALUDE_problem_solution_l1211_121161

theorem problem_solution (x y : ℝ) (hx : x = 3) (hy : y = 4) :
  (x^5 + 3*y^3) / 8 = 54.375 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l1211_121161


namespace NUMINAMATH_CALUDE_intersection_parallel_line_l1211_121115

/-- The equation of a line passing through the intersection of two lines and parallel to a third line -/
theorem intersection_parallel_line (a b c d e f g h i j : ℝ) :
  (∃ x y, a * x + b * y + c = 0 ∧ d * x + e * y + f = 0) →  -- Intersection exists
  (g * x + h * y + i = 0) →                                -- Third line
  (∃ k l m : ℝ, k ≠ 0 ∧
    (∀ x y, (a * x + b * y + c = 0 ∧ d * x + e * y + f = 0) →
      k * (g * x + h * y) + l * x + m * y + j = 0)) →      -- Parallel condition
  (∃ p q r : ℝ, p ≠ 0 ∧
    (∀ x y, (a * x + b * y + c = 0 ∧ d * x + e * y + f = 0) →
      p * x + q * y + r = 0) ∧                             -- Line through intersection
    ∃ s, p = s * g ∧ q = s * h) →                          -- Parallel to third line
  ∃ t, 2 * x + y = t                                       -- Resulting equation
  := by sorry

end NUMINAMATH_CALUDE_intersection_parallel_line_l1211_121115


namespace NUMINAMATH_CALUDE_vector_operation_proof_l1211_121138

def vector_a : Fin 2 → ℝ := ![2, 4]
def vector_b : Fin 2 → ℝ := ![-1, 1]

theorem vector_operation_proof :
  (2 • vector_a - vector_b) = ![5, 7] := by sorry

end NUMINAMATH_CALUDE_vector_operation_proof_l1211_121138


namespace NUMINAMATH_CALUDE_first_train_length_l1211_121173

/-- Given two trains with specific speeds, lengths, and crossing time, prove the length of the first train. -/
theorem first_train_length
  (v1 : ℝ) -- Speed of first train
  (v2 : ℝ) -- Speed of second train
  (l2 : ℝ) -- Length of second train
  (d : ℝ)  -- Distance between trains
  (t : ℝ)  -- Time for second train to cross first train
  (h1 : v1 = 10)
  (h2 : v2 = 15)
  (h3 : l2 = 150)
  (h4 : d = 50)
  (h5 : t = 60) :
  ∃ l1 : ℝ, l1 = 100 ∧ l1 + l2 + d = (v2 - v1) * t := by
  sorry

#check first_train_length

end NUMINAMATH_CALUDE_first_train_length_l1211_121173


namespace NUMINAMATH_CALUDE_square_sum_geq_product_l1211_121198

theorem square_sum_geq_product (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (h : a + b + c ≥ a * b * c) : a^2 + b^2 + c^2 ≥ a * b * c := by
  sorry

end NUMINAMATH_CALUDE_square_sum_geq_product_l1211_121198


namespace NUMINAMATH_CALUDE_ruble_payment_combinations_l1211_121119

theorem ruble_payment_combinations : 
  ∃! n : ℕ, n = (Finset.filter (λ (x : ℕ × ℕ) => 3 * x.1 + 5 * x.2 = 78) (Finset.product (Finset.range 79) (Finset.range 79))).card ∧ n = 6 :=
by sorry

end NUMINAMATH_CALUDE_ruble_payment_combinations_l1211_121119


namespace NUMINAMATH_CALUDE_round_to_scientific_notation_l1211_121151

/-- Rounds a real number to a specified number of significant figures -/
def roundToSigFigs (x : ℝ) (n : ℕ) : ℝ := sorry

/-- Converts a real number to scientific notation (a * 10^b form) -/
def toScientificNotation (x : ℝ) : ℝ × ℤ := sorry

theorem round_to_scientific_notation :
  let x := -29800000
  let sigFigs := 3
  let (a, b) := toScientificNotation (roundToSigFigs x sigFigs)
  a = -2.98 ∧ b = 7 := by sorry

end NUMINAMATH_CALUDE_round_to_scientific_notation_l1211_121151


namespace NUMINAMATH_CALUDE_inequality_always_positive_l1211_121111

theorem inequality_always_positive (a : ℝ) :
  (∀ x : ℝ, x^2 - x - a^2 + a + 1 > 0) →
  -1/2 < a ∧ a < 3/2 :=
by sorry

end NUMINAMATH_CALUDE_inequality_always_positive_l1211_121111


namespace NUMINAMATH_CALUDE_total_cost_calculation_l1211_121188

def shirt_price : ℝ := 50
def pants_price : ℝ := 40
def shoes_price : ℝ := 60
def shirt_discount : ℝ := 0.2
def shoes_discount : ℝ := 0.5
def sales_tax : ℝ := 0.08

def total_cost : ℝ :=
  let shirt_cost := 6 * shirt_price * (1 - shirt_discount)
  let pants_cost := 2 * pants_price
  let shoes_cost := 2 * shoes_price + shoes_price * (1 - shoes_discount)
  let subtotal := shirt_cost + pants_cost + shoes_cost
  subtotal * (1 + sales_tax)

theorem total_cost_calculation :
  total_cost = 507.60 := by sorry

end NUMINAMATH_CALUDE_total_cost_calculation_l1211_121188


namespace NUMINAMATH_CALUDE_inequality_solution_set_l1211_121104

theorem inequality_solution_set (a : ℝ) (ha : a > 0) :
  {x : ℝ | x^2 - (a + 1/a + 1)*x + a + 1/a < 0} = {x : ℝ | 1 < x ∧ x < a + 1/a} := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l1211_121104


namespace NUMINAMATH_CALUDE_pencil_box_theorems_l1211_121166

/-- Represents the number of pencils of each color in the box -/
structure PencilBox where
  blue : Nat
  red : Nat
  green : Nat
  yellow : Nat

/-- The initial state of the pencil box -/
def initialBox : PencilBox := {
  blue := 5,
  red := 9,
  green := 6,
  yellow := 4
}

/-- The minimum number of pencils to ensure at least one of each color -/
def minPencilsForAllColors (box : PencilBox) : Nat :=
  box.blue + box.red + box.green + box.yellow - 3

/-- The maximum number of pencils to ensure at least one of each color remains -/
def maxPencilsLeaveAllColors (box : PencilBox) : Nat :=
  min box.blue box.red |> min box.green |> min box.yellow |> (· - 1)

/-- The maximum number of pencils to ensure at least five red pencils remain -/
def maxPencilsLeaveFiveRed (box : PencilBox) : Nat :=
  max (box.red - 5) 0

theorem pencil_box_theorems (box : PencilBox := initialBox) :
  (minPencilsForAllColors box = 21) ∧
  (maxPencilsLeaveAllColors box = 3) ∧
  (maxPencilsLeaveFiveRed box = 4) := by
  sorry

end NUMINAMATH_CALUDE_pencil_box_theorems_l1211_121166


namespace NUMINAMATH_CALUDE_cake_slices_proof_l1211_121187

/-- The number of calories in each slice of cake -/
def calories_per_cake_slice : ℕ := 347

/-- The number of brownies in a pan -/
def brownies_per_pan : ℕ := 6

/-- The number of calories in each brownie -/
def calories_per_brownie : ℕ := 375

/-- The difference in calories between the cake and the pan of brownies -/
def calorie_difference : ℕ := 526

/-- The number of slices in the cake -/
def cake_slices : ℕ := 8

theorem cake_slices_proof :
  cake_slices * calories_per_cake_slice = 
  brownies_per_pan * calories_per_brownie + calorie_difference := by
  sorry

end NUMINAMATH_CALUDE_cake_slices_proof_l1211_121187


namespace NUMINAMATH_CALUDE_points_on_circle_l1211_121165

-- Define the points
variable (A B C X Y A' : EuclideanSpace ℝ (Fin 2))

-- Define the conditions
variable (acute_triangle : IsAcute A B C)
(X_side : DifferentSide X C (Line.throughPoints A B))
(Y_side : DifferentSide Y B (Line.throughPoints A C))
(BX_eq_AC : dist B X = dist A C)
(CY_eq_AB : dist C Y = dist A B)
(AX_eq_AY : dist A X = dist A Y)
(A'_reflection : IsReflection A A' (Perp.bisector B C))
(XY_diff_sides : DifferentSide X Y (Line.throughPoints A A'))

-- State the theorem
theorem points_on_circle :
  ∃ (O : EuclideanSpace ℝ (Fin 2)) (r : ℝ), 
    dist O A = r ∧ dist O A' = r ∧ dist O X = r ∧ dist O Y = r :=
sorry

end NUMINAMATH_CALUDE_points_on_circle_l1211_121165


namespace NUMINAMATH_CALUDE_negation_equivalence_l1211_121147

def exactly_one_even (a b c : ℕ) : Prop :=
  (Even a ∧ Odd b ∧ Odd c) ∨ (Odd a ∧ Even b ∧ Odd c) ∨ (Odd a ∧ Odd b ∧ Even c)

def negation_statement (a b c : ℕ) : Prop :=
  (Even a ∧ Even b) ∨ (Even a ∧ Even c) ∨ (Even b ∧ Even c) ∨ (Odd a ∧ Odd b ∧ Odd c)

theorem negation_equivalence (a b c : ℕ) :
  ¬(exactly_one_even a b c) ↔ negation_statement a b c :=
sorry

end NUMINAMATH_CALUDE_negation_equivalence_l1211_121147


namespace NUMINAMATH_CALUDE_total_cost_calculation_l1211_121174

def dog_cost : ℕ := 60
def cat_cost : ℕ := 40
def num_dogs : ℕ := 20
def num_cats : ℕ := 60

theorem total_cost_calculation : 
  dog_cost * num_dogs + cat_cost * num_cats = 3600 := by
  sorry

end NUMINAMATH_CALUDE_total_cost_calculation_l1211_121174


namespace NUMINAMATH_CALUDE_three_digit_sum_l1211_121137

theorem three_digit_sum (a b c : Nat) : 
  a ≠ b ∧ b ≠ c ∧ a ≠ c →
  a < 10 ∧ b < 10 ∧ c < 10 →
  (1730 + a) % 9 = 0 →
  (1730 + b) % 11 = 0 →
  (1730 + c) % 6 = 0 →
  a + b + c = 19 := by
sorry

end NUMINAMATH_CALUDE_three_digit_sum_l1211_121137


namespace NUMINAMATH_CALUDE_floor_equation_solution_l1211_121108

def solution_set : Set ℚ := {16/23, 17/23, 18/23, 19/23, 20/23, 21/23, 22/23, 1}

theorem floor_equation_solution (x : ℚ) :
  (⌊(20 : ℚ) * x + 23⌋ = 20 + 23 * x) ↔ x ∈ solution_set := by
  sorry

end NUMINAMATH_CALUDE_floor_equation_solution_l1211_121108


namespace NUMINAMATH_CALUDE_unicorns_total_games_l1211_121199

theorem unicorns_total_games : 
  ∀ (initial_games initial_wins district_wins district_losses : ℕ),
    initial_wins = initial_games / 2 →
    district_wins = 8 →
    district_losses = 3 →
    (initial_wins + district_wins) * 100 = 55 * (initial_games + district_wins + district_losses) →
    initial_games + district_wins + district_losses = 50 := by
  sorry

end NUMINAMATH_CALUDE_unicorns_total_games_l1211_121199


namespace NUMINAMATH_CALUDE_natural_number_pairs_l1211_121182

theorem natural_number_pairs : ∀ (a b : ℕ+), 
  (∃ (k l : ℕ+), (a + 1 : ℕ) = k * b ∧ (b + 1 : ℕ) = l * a) →
  ((a = 1 ∧ b = 1) ∨ 
   (a = 1 ∧ b = 2) ∨ 
   (a = 2 ∧ b = 3) ∨ 
   (a = 2 ∧ b = 1) ∨ 
   (a = 3 ∧ b = 2)) :=
by sorry


end NUMINAMATH_CALUDE_natural_number_pairs_l1211_121182


namespace NUMINAMATH_CALUDE_vertical_shift_equation_line_shift_theorem_l1211_121193

/-- Represents a linear function of the form y = mx + b -/
structure LinearFunction where
  slope : ℝ
  intercept : ℝ

/-- Applies a vertical shift to a linear function -/
def verticalShift (f : LinearFunction) (shift : ℝ) : LinearFunction :=
  { slope := f.slope, intercept := f.intercept + shift }

theorem vertical_shift_equation (m : ℝ) (shift : ℝ) :
  let original := LinearFunction.mk m 0
  let shifted := verticalShift original shift
  shifted = LinearFunction.mk m shift := by sorry

/-- The main theorem proving that shifting y = -5x upwards by 2 units results in y = -5x + 2 -/
theorem line_shift_theorem :
  let original := LinearFunction.mk (-5) 0
  let shifted := verticalShift original 2
  shifted = LinearFunction.mk (-5) 2 := by sorry

end NUMINAMATH_CALUDE_vertical_shift_equation_line_shift_theorem_l1211_121193


namespace NUMINAMATH_CALUDE_new_average_production_l1211_121195

theorem new_average_production (n : ℕ) (old_average : ℝ) (today_production : ℝ) 
  (h1 : n = 5)
  (h2 : old_average = 60)
  (h3 : today_production = 90) :
  let total_production := n * old_average
  let new_total_production := total_production + today_production
  let new_days := n + 1
  (new_total_production / new_days : ℝ) = 65 := by sorry

end NUMINAMATH_CALUDE_new_average_production_l1211_121195


namespace NUMINAMATH_CALUDE_class_representation_ratio_l1211_121128

theorem class_representation_ratio (boys girls : ℕ) 
  (h1 : boys + girls > 0)  -- ensure non-empty class
  (h2 : (boys : ℚ) / (boys + girls : ℚ) = 3/4 * (girls : ℚ) / (boys + girls : ℚ)) :
  (boys : ℚ) / (boys + girls : ℚ) = 3/7 := by
sorry

end NUMINAMATH_CALUDE_class_representation_ratio_l1211_121128


namespace NUMINAMATH_CALUDE_ball_draw_probability_l1211_121118

theorem ball_draw_probability (n : ℕ) : 
  (200 ≤ n) ∧ (n ≤ 1000) ∧ 
  (∃ k : ℕ, n = k^2) ∧
  (∃ x y : ℕ, x + y = n ∧ (x - y)^2 = n) →
  (∃ l : List ℕ, l.length = 17 ∧ n ∈ l) :=
sorry

end NUMINAMATH_CALUDE_ball_draw_probability_l1211_121118


namespace NUMINAMATH_CALUDE_perimeter_values_finite_l1211_121133

-- Define the quadrilateral ABCD
structure Quadrilateral :=
  (AB BC CD AD : ℕ+)

-- Define the conditions for our specific quadrilateral
def ValidQuadrilateral (q : Quadrilateral) : Prop :=
  q.AB = 3 ∧ q.CD = 2 * q.AD

-- Define the perimeter
def Perimeter (q : Quadrilateral) : ℕ :=
  q.AB + q.BC + q.CD + q.AD

-- Define the right angle condition using Pythagorean theorem
def RightAngles (q : Quadrilateral) : Prop :=
  q.BC ^ 2 + (q.CD - q.AB) ^ 2 = q.AD ^ 2

-- Main theorem
theorem perimeter_values_finite :
  {p : ℕ | p < 3025 ∧ ∃ q : Quadrilateral, ValidQuadrilateral q ∧ RightAngles q ∧ Perimeter q = p}.Finite :=
sorry

end NUMINAMATH_CALUDE_perimeter_values_finite_l1211_121133


namespace NUMINAMATH_CALUDE_boat_speed_l1211_121179

/-- The speed of a boat in still water, given its downstream and upstream speeds -/
theorem boat_speed (downstream upstream : ℝ) (h1 : downstream = 15) (h2 : upstream = 7) :
  (downstream + upstream) / 2 = 11 :=
by
  sorry

#check boat_speed

end NUMINAMATH_CALUDE_boat_speed_l1211_121179


namespace NUMINAMATH_CALUDE_binary_addition_multiplication_l1211_121117

def binary_to_decimal (b : List Bool) : ℕ :=
  b.enum.foldl (λ acc (i, bit) => acc + if bit then 2^i else 0) 0

theorem binary_addition_multiplication : 
  let b1 := [true, true, true, true, true]
  let b2 := [true, true, true, true, true, true, true, true]
  let b3 := [false, true]
  (binary_to_decimal b1 + binary_to_decimal b2) * binary_to_decimal b3 = 572 := by
  sorry

end NUMINAMATH_CALUDE_binary_addition_multiplication_l1211_121117


namespace NUMINAMATH_CALUDE_range_of_m_l1211_121160

def A : Set ℝ := {y | ∃ x ∈ Set.Icc (3/4 : ℝ) 2, y = x^2 - (3/2)*x + 1}

def B (m : ℝ) : Set ℝ := {x | x + m^2 ≥ 1}

theorem range_of_m :
  {m : ℝ | A ⊆ B m} = Set.Iic (-3/4) ∪ Set.Ici (3/4) :=
sorry

end NUMINAMATH_CALUDE_range_of_m_l1211_121160


namespace NUMINAMATH_CALUDE_andrew_fruit_purchase_l1211_121113

/-- Calculates the total amount paid for fruits given the quantities and prices -/
def totalAmountPaid (grapeQuantity mangoQuantity grapePrice mangoPrice : ℕ) : ℕ :=
  grapeQuantity * grapePrice + mangoQuantity * mangoPrice

/-- Theorem stating that Andrew paid 975 for his fruit purchase -/
theorem andrew_fruit_purchase : 
  totalAmountPaid 6 9 74 59 = 975 := by
  sorry

end NUMINAMATH_CALUDE_andrew_fruit_purchase_l1211_121113


namespace NUMINAMATH_CALUDE_complex_number_ratio_l1211_121129

theorem complex_number_ratio (x y z : ℂ) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0)
  (h_sum : x + y + z = 30)
  (h_eq : ((x - y)^2 + (x - z)^2 + (y - z)^2) * (x + y + z) = x * y * z) :
  (x^3 + y^3 + z^3) / (x * y * z) = (3.5 : ℂ) := by
sorry

end NUMINAMATH_CALUDE_complex_number_ratio_l1211_121129


namespace NUMINAMATH_CALUDE_zoo_elephants_l1211_121178

theorem zoo_elephants (giraffes : ℕ) (penguins : ℕ) (total : ℕ) (elephants : ℕ) : 
  giraffes = 5 →
  penguins = 2 * giraffes →
  penguins = (20 : ℚ) / 100 * total →
  elephants = (4 : ℚ) / 100 * total →
  elephants = 2 :=
by sorry

end NUMINAMATH_CALUDE_zoo_elephants_l1211_121178


namespace NUMINAMATH_CALUDE_ceiling_floor_difference_l1211_121106

theorem ceiling_floor_difference : ⌈(15 / 8) * (-34 / 4)⌉ - ⌊(15 / 8) * ⌊-34 / 4⌋⌋ = 2 := by sorry

end NUMINAMATH_CALUDE_ceiling_floor_difference_l1211_121106


namespace NUMINAMATH_CALUDE_cone_hemisphere_relation_cone_base_radius_is_10_5_l1211_121150

/-- Represents a cone with a hemisphere resting on its base --/
structure ConeWithHemisphere where
  cone_height : ℝ
  hemisphere_radius : ℝ
  cone_base_radius : ℝ

/-- Checks if the configuration is valid --/
def is_valid_configuration (c : ConeWithHemisphere) : Prop :=
  c.cone_height > 0 ∧ c.hemisphere_radius > 0 ∧ c.cone_base_radius > c.hemisphere_radius

/-- Theorem stating the relationship between cone dimensions and hemisphere --/
theorem cone_hemisphere_relation (c : ConeWithHemisphere) 
  (h_valid : is_valid_configuration c)
  (h_height : c.cone_height = 9)
  (h_radius : c.hemisphere_radius = 3) :
  c.cone_base_radius = 10.5 := by
  sorry

/-- Main theorem proving the base radius of the cone --/
theorem cone_base_radius_is_10_5 :
  ∃ c : ConeWithHemisphere, 
    is_valid_configuration c ∧ 
    c.cone_height = 9 ∧ 
    c.hemisphere_radius = 3 ∧ 
    c.cone_base_radius = 10.5 := by
  sorry

end NUMINAMATH_CALUDE_cone_hemisphere_relation_cone_base_radius_is_10_5_l1211_121150


namespace NUMINAMATH_CALUDE_chemist_problem_solution_l1211_121131

/-- Represents the purity of a salt solution as a real number between 0 and 1 -/
def Purity := { p : ℝ // 0 ≤ p ∧ p ≤ 1 }

/-- The chemist's problem setup -/
structure ChemistProblem where
  solution1 : Purity
  solution2 : Purity
  total_amount : ℝ
  final_purity : Purity
  amount_solution1 : ℝ
  h1 : solution1.val = 0.3
  h2 : total_amount = 60
  h3 : final_purity.val = 0.5
  h4 : amount_solution1 = 40

theorem chemist_problem_solution (p : ChemistProblem) : p.solution2.val = 0.9 := by
  sorry

end NUMINAMATH_CALUDE_chemist_problem_solution_l1211_121131


namespace NUMINAMATH_CALUDE_train_length_l1211_121176

theorem train_length (platform_time : ℝ) (pole_time : ℝ) (platform_length : ℝ)
  (h1 : platform_time = 39)
  (h2 : pole_time = 18)
  (h3 : platform_length = 350) :
  ∃ (train_length : ℝ), train_length = 300 ∧
    train_length / pole_time = (train_length + platform_length) / platform_time :=
by sorry

end NUMINAMATH_CALUDE_train_length_l1211_121176


namespace NUMINAMATH_CALUDE_power_equality_l1211_121130

theorem power_equality (x : ℝ) : (1/8 : ℝ) * 2^36 = 8^x → x = 11 := by
  sorry

end NUMINAMATH_CALUDE_power_equality_l1211_121130


namespace NUMINAMATH_CALUDE_triangle_geometric_sequence_l1211_121145

/-- In a triangle ABC where sides a, b, c form a geometric sequence and satisfy a² - c² = ac - bc, 
    the ratio (b * sin B) / c is equal to √3/2. -/
theorem triangle_geometric_sequence (a b c : ℝ) (A B C : ℝ) :
  a > 0 → b > 0 → c > 0 →
  (b ^ 2 = a * c) →  -- geometric sequence condition
  (a ^ 2 - c ^ 2 = a * c - b * c) →
  (a ^ 2 = b ^ 2 + c ^ 2 - 2 * b * c * Real.cos A) →  -- cosine rule
  (b * Real.sin B = a * Real.sin A) →  -- sine rule
  (b * Real.sin B) / c = Real.sqrt 3 / 2 := by
sorry

end NUMINAMATH_CALUDE_triangle_geometric_sequence_l1211_121145


namespace NUMINAMATH_CALUDE_problem_statement_l1211_121167

theorem problem_statement (x y : ℚ) : 
  x = 3/4 → y = 4/3 → (3/5 : ℚ) * x^5 * y^8 = 897/1000 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l1211_121167


namespace NUMINAMATH_CALUDE_imon_entanglement_reduction_l1211_121171

/-- Represents a graph of imons and their entanglements -/
structure ImonGraph where
  vertices : Set ℕ
  edges : Set (ℕ × ℕ)

/-- Operation 1: Remove a vertex with odd degree -/
def removeOddDegreeVertex (G : ImonGraph) (v : ℕ) : ImonGraph :=
  sorry

/-- Operation 2: Duplicate the graph and connect each vertex to its duplicate -/
def duplicateGraph (G : ImonGraph) : ImonGraph :=
  sorry

/-- Predicate to check if a graph has no edges -/
def hasNoEdges (G : ImonGraph) : Prop :=
  G.edges = ∅

/-- Main theorem: There exists a sequence of operations to reduce any ImonGraph to one with no edges -/
theorem imon_entanglement_reduction (G : ImonGraph) :
  ∃ (seq : List (ImonGraph → ImonGraph)), hasNoEdges (seq.foldl (λ g f => f g) G) :=
  sorry

end NUMINAMATH_CALUDE_imon_entanglement_reduction_l1211_121171


namespace NUMINAMATH_CALUDE_rabbit_speed_problem_l1211_121125

theorem rabbit_speed_problem (rabbit_speed : ℕ) (x : ℕ) : 
  rabbit_speed = 45 →
  2 * (2 * rabbit_speed + x) = 188 →
  x = 4 := by
sorry

end NUMINAMATH_CALUDE_rabbit_speed_problem_l1211_121125


namespace NUMINAMATH_CALUDE_x_squared_gt_1_sufficient_not_necessary_for_reciprocal_lt_1_l1211_121148

theorem x_squared_gt_1_sufficient_not_necessary_for_reciprocal_lt_1 :
  (∀ x : ℝ, x^2 > 1 → 1/x < 1) ∧
  (∃ x : ℝ, 1/x < 1 ∧ ¬(x^2 > 1)) := by
  sorry

end NUMINAMATH_CALUDE_x_squared_gt_1_sufficient_not_necessary_for_reciprocal_lt_1_l1211_121148


namespace NUMINAMATH_CALUDE_bella_earrings_l1211_121123

/-- Given three friends Bella, Monica, and Rachel, with the following conditions:
    1. Bella has 25% of Monica's earrings
    2. Monica has twice as many earrings as Rachel
    3. The total number of earrings among the three friends is 70
    Prove that Bella has 10 earrings. -/
theorem bella_earrings (bella monica rachel : ℕ) : 
  bella = (25 : ℕ) * monica / 100 →
  monica = 2 * rachel →
  bella + monica + rachel = 70 →
  bella = 10 := by
sorry

end NUMINAMATH_CALUDE_bella_earrings_l1211_121123


namespace NUMINAMATH_CALUDE_base5_123_to_base10_l1211_121105

/-- Converts a base-5 number to base-10 --/
def base5ToBase10 (digits : List Nat) : Nat :=
  digits.foldr (fun d acc => 5 * acc + d) 0

/-- The base-5 representation of 123 --/
def base5_123 : List Nat := [1, 2, 3]

theorem base5_123_to_base10 :
  base5ToBase10 base5_123 = 38 := by
  sorry

end NUMINAMATH_CALUDE_base5_123_to_base10_l1211_121105


namespace NUMINAMATH_CALUDE_equilateral_side_length_l1211_121154

/-- Given a diagram with an equilateral triangle and a right-angled triangle,
    where the right-angled triangle has a side length of 6 and both triangles
    have a 45-degree angle, the side length y of the equilateral triangle is 6√2. -/
theorem equilateral_side_length (y : ℝ) : y = 6 * Real.sqrt 2 := by sorry

end NUMINAMATH_CALUDE_equilateral_side_length_l1211_121154


namespace NUMINAMATH_CALUDE_ellipse_sum_coordinates_and_axes_specific_ellipse_sum_l1211_121169

/-- Definition of an ellipse with given center and axis lengths -/
structure Ellipse where
  center : ℝ × ℝ
  semi_major_axis : ℝ
  semi_minor_axis : ℝ

/-- Theorem: Sum of center coordinates and axis lengths for a specific ellipse -/
theorem ellipse_sum_coordinates_and_axes (e : Ellipse) 
  (h1 : e.center = (-3, 4)) 
  (h2 : e.semi_major_axis = 7) 
  (h3 : e.semi_minor_axis = 2) : 
  e.center.1 + e.center.2 + e.semi_major_axis + e.semi_minor_axis = 10 := by
  sorry

/-- Main theorem to be proved -/
theorem specific_ellipse_sum : 
  ∃ (e : Ellipse), e.center = (-3, 4) ∧ e.semi_major_axis = 7 ∧ e.semi_minor_axis = 2 ∧
  e.center.1 + e.center.2 + e.semi_major_axis + e.semi_minor_axis = 10 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_sum_coordinates_and_axes_specific_ellipse_sum_l1211_121169


namespace NUMINAMATH_CALUDE_range_of_a_l1211_121135

-- Define propositions p and q
def p (a : ℝ) : Prop := -3 < a ∧ a < 1
def q (a : ℝ) : Prop := a > 0

-- Define the theorem
theorem range_of_a (a : ℝ) :
  (¬(p a ∧ q a) ∧ (p a ∨ q a)) → ((-3 < a ∧ a ≤ 0) ∨ a ≥ 1) :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l1211_121135


namespace NUMINAMATH_CALUDE_total_goals_is_sixteen_l1211_121177

def bruce_goals : ℕ := 4
def michael_goals_multiplier : ℕ := 3

def total_goals : ℕ := bruce_goals + michael_goals_multiplier * bruce_goals

theorem total_goals_is_sixteen : total_goals = 16 := by
  sorry

end NUMINAMATH_CALUDE_total_goals_is_sixteen_l1211_121177


namespace NUMINAMATH_CALUDE_uncovered_area_of_box_l1211_121185

/-- Given a rectangular box with dimensions 4 inches by 6 inches and a square block with side length 4 inches placed inside, the uncovered area of the box is 8 square inches. -/
theorem uncovered_area_of_box (box_length : ℕ) (box_width : ℕ) (block_side : ℕ) : 
  box_length = 4 → box_width = 6 → block_side = 4 → 
  (box_length * box_width) - (block_side * block_side) = 8 := by
sorry

end NUMINAMATH_CALUDE_uncovered_area_of_box_l1211_121185


namespace NUMINAMATH_CALUDE_rotated_line_slope_l1211_121102

theorem rotated_line_slope (m : ℝ) (θ : ℝ) :
  m = -Real.sqrt 3 →
  θ = π / 3 →
  (m * Real.cos θ + Real.sin θ) / (Real.cos θ - m * Real.sin θ) = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_rotated_line_slope_l1211_121102


namespace NUMINAMATH_CALUDE_triangle_problem_l1211_121189

theorem triangle_problem (A B C : ℝ) (m n : ℝ × ℝ) (AC : ℝ) :
  0 < A ∧ A < π / 2 →
  0 < B ∧ B < π / 2 →
  0 < C ∧ C < π / 2 →
  m = (Real.cos (A + π / 3), Real.sin (A + π / 3)) →
  n = (Real.cos B, Real.sin B) →
  m.1 * n.1 + m.2 * n.2 = 0 →
  Real.cos B = 3 / 5 →
  AC = 8 →
  A - B = π / 6 ∧ Real.sqrt ((4 * Real.sqrt 3 + 3) ^ 2) = 4 * Real.sqrt 3 + 3 := by
  sorry

end NUMINAMATH_CALUDE_triangle_problem_l1211_121189


namespace NUMINAMATH_CALUDE_specific_quadrilateral_perimeter_l1211_121157

/-- A convex quadrilateral with a point inside it -/
structure ConvexQuadrilateral where
  W : ℝ × ℝ
  X : ℝ × ℝ
  Y : ℝ × ℝ
  Z : ℝ × ℝ
  Q : ℝ × ℝ
  area : ℝ
  wq : ℝ
  xq : ℝ
  yq : ℝ
  zq : ℝ
  convex : Bool
  inside : Bool

/-- The perimeter of a quadrilateral -/
def perimeter (quad : ConvexQuadrilateral) : ℝ :=
  sorry

/-- Theorem stating the perimeter of the specific quadrilateral -/
theorem specific_quadrilateral_perimeter :
  ∀ (quad : ConvexQuadrilateral),
    quad.area = 2500 ∧
    quad.wq = 30 ∧
    quad.xq = 40 ∧
    quad.yq = 35 ∧
    quad.zq = 50 ∧
    quad.convex = true ∧
    quad.inside = true →
    perimeter quad = 155 + 10 * Real.sqrt 34 + 5 * Real.sqrt 113 :=
  sorry

end NUMINAMATH_CALUDE_specific_quadrilateral_perimeter_l1211_121157


namespace NUMINAMATH_CALUDE_min_sum_squares_l1211_121109

theorem min_sum_squares (x y z : ℝ) (h : x^3 + y^3 + z^3 - 3*x*y*z = 8) :
  ∃ (m : ℝ), m = 8 ∧ ∀ (a b c : ℝ), a^3 + b^3 + c^3 - 3*a*b*c = 8 → x^2 + y^2 + z^2 ≤ a^2 + b^2 + c^2 := by
sorry

end NUMINAMATH_CALUDE_min_sum_squares_l1211_121109


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l1211_121153

theorem quadratic_inequality_solution_set :
  {x : ℝ | 2*x + 3 - x^2 > 0} = {x : ℝ | -1 < x ∧ x < 3} := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l1211_121153


namespace NUMINAMATH_CALUDE_plot_length_is_sixty_l1211_121136

/-- Proves that the length of a rectangular plot is 60 metres given the specified conditions -/
theorem plot_length_is_sixty (breadth : ℝ) (length : ℝ) (perimeter : ℝ) (cost_per_metre : ℝ) (total_cost : ℝ) :
  length = breadth + 20 →
  perimeter = 2 * length + 2 * breadth →
  cost_per_metre = 26.50 →
  total_cost = 5300 →
  perimeter = total_cost / cost_per_metre →
  length = 60 := by
  sorry

end NUMINAMATH_CALUDE_plot_length_is_sixty_l1211_121136


namespace NUMINAMATH_CALUDE_identify_roles_l1211_121141

-- Define the possible roles
inductive Role
  | Knight
  | Liar
  | Normal

-- Define the statements made by each individual
def statement_A : Prop := ∃ x, x = Role.Normal
def statement_B : Prop := statement_A
def statement_C : Prop := ¬∃ x, x = Role.Normal

-- Define the properties of each role
def always_true (r : Role) : Prop := r = Role.Knight
def always_false (r : Role) : Prop := r = Role.Liar
def can_be_either (r : Role) : Prop := r = Role.Normal

-- The main theorem
theorem identify_roles :
  ∃! (role_A role_B role_C : Role),
    -- Each person has a unique role
    role_A ≠ role_B ∧ role_B ≠ role_C ∧ role_A ≠ role_C ∧
    -- One of each role exists
    (always_true role_A ∨ always_true role_B ∨ always_true role_C) ∧
    (always_false role_A ∨ always_false role_B ∨ always_false role_C) ∧
    (can_be_either role_A ∨ can_be_either role_B ∨ can_be_either role_C) ∧
    -- Statements are consistent with roles
    ((always_true role_A → statement_A) ∧ (always_false role_A → ¬statement_A) ∧ (can_be_either role_A → True)) ∧
    ((always_true role_B → statement_B) ∧ (always_false role_B → ¬statement_B) ∧ (can_be_either role_B → True)) ∧
    ((always_true role_C → statement_C) ∧ (always_false role_C → ¬statement_C) ∧ (can_be_either role_C → True)) ∧
    -- The solution
    always_false role_A ∧ always_true role_B ∧ can_be_either role_C :=
by sorry


end NUMINAMATH_CALUDE_identify_roles_l1211_121141


namespace NUMINAMATH_CALUDE_countable_planar_graph_coloring_l1211_121172

-- Define a type for colors
inductive Color
| blue
| red
| green

-- Define a type for graphs
structure Graph (α : Type) where
  vertices : Set α
  edges : Set (α × α)

-- Define what it means for a graph to be planar
def isPlanar {α : Type} (G : Graph α) : Prop := sorry

-- Define what it means for a graph to be countable
def isCountable {α : Type} (G : Graph α) : Prop := sorry

-- Define what it means for a cycle to be odd
def isOddCycle {α : Type} (G : Graph α) (cycle : List α) : Prop := sorry

-- Define what it means for a coloring to be valid (no odd monochromatic cycles)
def isValidColoring {α : Type} (G : Graph α) (coloring : α → Color) : Prop :=
  ∀ cycle, isOddCycle G cycle → ∃ v ∈ cycle, ∃ w ∈ cycle, coloring v ≠ coloring w

-- The main theorem
theorem countable_planar_graph_coloring 
  {α : Type} (G : Graph α) 
  (h_planar : isPlanar G) 
  (h_countable : isCountable G) 
  (h_finite : ∀ (H : Graph α), isPlanar H → (Finite H.vertices) → 
    ∃ coloring : α → Color, isValidColoring H coloring) :
  ∃ coloring : α → Color, isValidColoring G coloring := by
  sorry

end NUMINAMATH_CALUDE_countable_planar_graph_coloring_l1211_121172


namespace NUMINAMATH_CALUDE_function_decreasing_interval_l1211_121126

def f (a b x : ℝ) : ℝ := x^3 - 3*a*x + b

theorem function_decreasing_interval
  (a b : ℝ)
  (h1 : a > 0)
  (h2 : ∃ x1, f a b x1 = 6 ∧ ∀ x, f a b x ≤ 6)
  (h3 : ∃ x2, f a b x2 = 2 ∧ ∀ x, f a b x ≥ 2) :
  ∀ x ∈ Set.Ioo (-1 : ℝ) 1, ∀ y ∈ Set.Ioo (-1 : ℝ) 1,
    x < y → f a b x > f a b y :=
by sorry

end NUMINAMATH_CALUDE_function_decreasing_interval_l1211_121126


namespace NUMINAMATH_CALUDE_sugar_profit_problem_l1211_121120

/-- A merchant sells sugar with two different profit percentages --/
theorem sugar_profit_problem (total_sugar : ℝ) (sugar_at_known_profit : ℝ) (sugar_at_unknown_profit : ℝ)
  (known_profit_percentage : ℝ) (overall_profit_percentage : ℝ) (unknown_profit_percentage : ℝ)
  (h1 : total_sugar = 1000)
  (h2 : sugar_at_known_profit = 400)
  (h3 : sugar_at_unknown_profit = 600)
  (h4 : known_profit_percentage = 8)
  (h5 : overall_profit_percentage = 14)
  (h6 : total_sugar = sugar_at_known_profit + sugar_at_unknown_profit)
  (h7 : sugar_at_known_profit * (known_profit_percentage / 100) +
        sugar_at_unknown_profit * (unknown_profit_percentage / 100) =
        total_sugar * (overall_profit_percentage / 100)) :
  unknown_profit_percentage = 18 := by
sorry

end NUMINAMATH_CALUDE_sugar_profit_problem_l1211_121120


namespace NUMINAMATH_CALUDE_complex_equation_solution_l1211_121197

theorem complex_equation_solution (z : ℂ) : z = Complex.I * (2 - z) → z = 1 + Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l1211_121197


namespace NUMINAMATH_CALUDE_trajectory_of_moving_circle_center_l1211_121149

/-- A moving circle passes through a fixed point F(2, 0) and is tangent to the line x = -2.
    The trajectory of its center C is a parabola. -/
theorem trajectory_of_moving_circle_center (C : ℝ × ℝ) : 
  (∃ (r : ℝ), (C.1 - 2)^2 + C.2^2 = r^2 ∧ (C.1 + 2)^2 + C.2^2 = r^2) →
  C.2^2 = 8 * C.1 := by
  sorry


end NUMINAMATH_CALUDE_trajectory_of_moving_circle_center_l1211_121149


namespace NUMINAMATH_CALUDE_unique_a_for_set_equality_l1211_121163

theorem unique_a_for_set_equality : ∃! a : ℝ, ({1, 3, a^2} ∪ {1, a+2} : Set ℝ) = {1, 3, a^2} := by
  sorry

end NUMINAMATH_CALUDE_unique_a_for_set_equality_l1211_121163


namespace NUMINAMATH_CALUDE_article_original_price_l1211_121181

/-- Given an article with a discounted price after a 24% decrease, 
    prove that its original price was Rs. 1400. -/
theorem article_original_price (discounted_price : ℝ) : 
  discounted_price = 1064 → 
  ∃ (original_price : ℝ), 
    original_price * (1 - 0.24) = discounted_price ∧ 
    original_price = 1400 := by
  sorry

end NUMINAMATH_CALUDE_article_original_price_l1211_121181


namespace NUMINAMATH_CALUDE_track_circumference_l1211_121162

/-- Represents the circular track and the movement of A and B -/
structure CircularTrack where
  /-- The circumference of the track in yards -/
  circumference : ℝ
  /-- The distance B has traveled when they first meet -/
  first_meeting : ℝ
  /-- The distance A is shy of completing a full lap at the second meeting -/
  second_meeting : ℝ

/-- Theorem stating that under the given conditions, the track's circumference is 720 yards -/
theorem track_circumference (track : CircularTrack) 
  (h1 : track.first_meeting = 150)
  (h2 : track.second_meeting = 90)
  (h3 : track.circumference > 0) :
  track.circumference = 720 := by
  sorry

#check track_circumference

end NUMINAMATH_CALUDE_track_circumference_l1211_121162


namespace NUMINAMATH_CALUDE_angel_score_is_11_l1211_121183

-- Define the scores for each player
def beth_score : ℕ := 12
def jan_score : ℕ := 10
def judy_score : ℕ := 8

-- Define the total score of the first team
def first_team_score : ℕ := beth_score + jan_score

-- Define the difference between the first and second team scores
def score_difference : ℕ := 3

-- Define Angel's score as a variable
def angel_score : ℕ := sorry

-- Theorem to prove
theorem angel_score_is_11 :
  angel_score = 11 :=
by
  sorry

end NUMINAMATH_CALUDE_angel_score_is_11_l1211_121183


namespace NUMINAMATH_CALUDE_probability_at_least_one_one_l1211_121132

-- Define the number of sides on each die
def num_sides : ℕ := 8

-- Define the probability of at least one die showing 1
def prob_at_least_one_one : ℚ := 15 / 64

-- Theorem statement
theorem probability_at_least_one_one :
  let total_outcomes := num_sides * num_sides
  let outcomes_without_one := (num_sides - 1) * (num_sides - 1)
  let favorable_outcomes := total_outcomes - outcomes_without_one
  (favorable_outcomes : ℚ) / total_outcomes = prob_at_least_one_one := by
  sorry

end NUMINAMATH_CALUDE_probability_at_least_one_one_l1211_121132


namespace NUMINAMATH_CALUDE_first_four_eq_last_four_l1211_121103

/-- A sequence of 0s and 1s -/
def BinarySeq := List Bool

/-- Checks if two segments of 5 terms are different -/
def differentSegments (s : BinarySeq) (i j : Nat) : Prop :=
  i < j ∧ j + 4 < s.length ∧
  (List.take 5 (List.drop i s) ≠ List.take 5 (List.drop j s))

/-- The condition that any two consecutive segments of 5 terms are different -/
def validSequence (s : BinarySeq) : Prop :=
  ∀ i j, i < j → j + 4 < s.length → differentSegments s i j

/-- S is the longest sequence satisfying the condition -/
def longestValidSequence (S : BinarySeq) : Prop :=
  validSequence S ∧ ∀ s, validSequence s → s.length ≤ S.length

theorem first_four_eq_last_four (S : BinarySeq) (h : longestValidSequence S) :
  S.take 4 = (S.reverse.take 4).reverse :=
sorry

end NUMINAMATH_CALUDE_first_four_eq_last_four_l1211_121103


namespace NUMINAMATH_CALUDE_isosceles_right_triangle_l1211_121110

-- Define a triangle ABC
structure Triangle :=
  (A B C : ℝ × ℝ)

-- Define the length of a line segment
def length (p q : ℝ × ℝ) : ℝ := sorry

-- Define the altitude from a point to a line segment
def altitude (p : ℝ × ℝ) (q r : ℝ × ℝ) : ℝ := sorry

-- Define the angle at a vertex
def angle (p q r : ℝ × ℝ) : ℝ := sorry

-- Theorem statement
theorem isosceles_right_triangle (t : Triangle) :
  altitude t.A t.B t.C ≥ length t.B t.C →
  altitude t.B t.A t.C ≥ length t.A t.C →
  angle t.B t.A t.C = 90 ∧ angle t.A t.B t.C = 45 ∧ angle t.A t.C t.B = 45 :=
by sorry

end NUMINAMATH_CALUDE_isosceles_right_triangle_l1211_121110


namespace NUMINAMATH_CALUDE_problem_1_problem_2_l1211_121175

-- Problem 1
theorem problem_1 : Real.sqrt 3 * Real.sqrt 6 - (Real.sqrt (1/2) - Real.sqrt 8) = (9 * Real.sqrt 2) / 2 := by sorry

-- Problem 2
theorem problem_2 (x : ℝ) (h : x = Real.sqrt 5) : 
  (1 + 1/x) / ((x^2 + x) / x) = Real.sqrt 5 / 5 := by sorry

end NUMINAMATH_CALUDE_problem_1_problem_2_l1211_121175


namespace NUMINAMATH_CALUDE_geometric_sequence_ratio_l1211_121142

-- Define a geometric sequence
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ (r : ℝ), ∀ (n : ℕ), a (n + 1) = r * a n

-- Define the conditions and theorem
theorem geometric_sequence_ratio 
  (a : ℕ → ℝ) 
  (h_geom : geometric_sequence a) 
  (h_a3 : a 3 = 2) 
  (h_a4a6 : a 4 * a 6 = 16) :
  (a 9 - a 11) / (a 5 - a 7) = 4 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_ratio_l1211_121142


namespace NUMINAMATH_CALUDE_max_cos_a_l1211_121143

theorem max_cos_a (a b : Real) (h : Real.cos (a + b) = Real.cos a - Real.cos b) :
  ∃ (max_cos_a : Real), max_cos_a = 1 ∧ ∀ x, Real.cos x ≤ max_cos_a :=
by sorry

end NUMINAMATH_CALUDE_max_cos_a_l1211_121143


namespace NUMINAMATH_CALUDE_fraction_equals_eight_over_twentyseven_l1211_121190

def numerator : ℕ := 1*2*4 + 2*4*8 + 3*6*12 + 4*8*16
def denominator : ℕ := 1*3*9 + 2*6*18 + 3*9*27 + 4*12*36

theorem fraction_equals_eight_over_twentyseven :
  (numerator : ℚ) / denominator = 8 / 27 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equals_eight_over_twentyseven_l1211_121190


namespace NUMINAMATH_CALUDE_least_three_digit_multiple_l1211_121170

def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

theorem least_three_digit_multiple : 
  (∀ k : ℕ, is_three_digit k ∧ 3 ∣ k ∧ 7 ∣ k ∧ 11 ∣ k → 231 ≤ k) ∧ 
  is_three_digit 231 ∧ 3 ∣ 231 ∧ 7 ∣ 231 ∧ 11 ∣ 231 :=
sorry

end NUMINAMATH_CALUDE_least_three_digit_multiple_l1211_121170


namespace NUMINAMATH_CALUDE_hotdog_eating_record_l1211_121191

/-- The hotdog eating record problem -/
theorem hotdog_eating_record 
  (total_time : ℕ) 
  (halfway_time : ℕ) 
  (halfway_hotdogs : ℕ) 
  (required_rate : ℕ) 
  (h1 : total_time = 10) 
  (h2 : halfway_time = total_time / 2) 
  (h3 : halfway_hotdogs = 20) 
  (h4 : required_rate = 11) : 
  halfway_hotdogs + required_rate * (total_time - halfway_time) = 75 := by
sorry


end NUMINAMATH_CALUDE_hotdog_eating_record_l1211_121191


namespace NUMINAMATH_CALUDE_chess_tournament_participants_perfect_square_l1211_121124

theorem chess_tournament_participants_perfect_square 
  (B : ℕ) -- number of boys
  (G : ℕ) -- number of girls
  (total_points : ℕ → ℕ → ℕ) -- function that calculates total points given boys and girls
  (h1 : ∀ x y, total_points x y = x * y) -- each participant plays once with every other
  (h2 : ∀ x y, 2 * (x * y) = x * (x - 1) + y * (y - 1)) -- half points from boys
  : ∃ k : ℕ, B + G = k^2 :=
sorry

end NUMINAMATH_CALUDE_chess_tournament_participants_perfect_square_l1211_121124


namespace NUMINAMATH_CALUDE_complex_fraction_calculation_l1211_121114

theorem complex_fraction_calculation : 
  (1 / (7 / 9) - (3 / 5) / 7) * (11 / (6 + 3 / 5)) / (4 / 13) - 2.4 = 4.1 := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_calculation_l1211_121114


namespace NUMINAMATH_CALUDE_total_cards_l1211_121121

theorem total_cards (mao_cards : ℕ) (li_cards : ℕ) 
  (h1 : mao_cards = 23) (h2 : li_cards = 20) : 
  mao_cards + li_cards = 43 := by
  sorry

end NUMINAMATH_CALUDE_total_cards_l1211_121121


namespace NUMINAMATH_CALUDE_clown_balloons_l1211_121186

theorem clown_balloons (initial_balloons : ℕ) (additional_balloons : ℕ) 
  (h1 : initial_balloons = 47) 
  (h2 : additional_balloons = 13) : 
  initial_balloons + additional_balloons = 60 := by
  sorry

end NUMINAMATH_CALUDE_clown_balloons_l1211_121186


namespace NUMINAMATH_CALUDE_digit_sum_congruence_part1_digit_sum_congruence_part2_l1211_121196

/-- S_r(n) is the sum of the digits of n in base r -/
def S_r (r : ℕ) (n : ℕ) : ℕ :=
  sorry

theorem digit_sum_congruence_part1 :
  ∀ r : ℕ, r > 2 → ∃ p : ℕ, Nat.Prime p ∧ ∀ n : ℕ, n > 0 → S_r r n ≡ n [MOD p] :=
sorry

theorem digit_sum_congruence_part2 :
  ∀ r : ℕ, r > 1 → ∀ p : ℕ, Nat.Prime p →
  ∃ f : ℕ → ℕ, Function.Injective f ∧ ∀ k : ℕ, S_r r (f k) ≡ f k [MOD p] :=
sorry

end NUMINAMATH_CALUDE_digit_sum_congruence_part1_digit_sum_congruence_part2_l1211_121196


namespace NUMINAMATH_CALUDE_green_shirts_count_l1211_121158

-- Define the total number of shirts
def total_shirts : ℕ := 23

-- Define the number of blue shirts
def blue_shirts : ℕ := 6

-- Theorem: The number of green shirts is 17
theorem green_shirts_count : total_shirts - blue_shirts = 17 := by
  sorry

end NUMINAMATH_CALUDE_green_shirts_count_l1211_121158


namespace NUMINAMATH_CALUDE_nancy_carrots_l1211_121122

/-- The number of carrots Nancy picked the next day -/
def carrots_picked_next_day (initial_carrots : ℕ) (thrown_out : ℕ) (total_carrots : ℕ) : ℕ :=
  total_carrots - (initial_carrots - thrown_out)

/-- Proof that Nancy picked 21 carrots the next day -/
theorem nancy_carrots :
  carrots_picked_next_day 12 2 31 = 21 := by
  sorry

end NUMINAMATH_CALUDE_nancy_carrots_l1211_121122


namespace NUMINAMATH_CALUDE_baseball_league_games_played_l1211_121146

/-- Represents a baseball league with the given parameters -/
structure BaseballLeague where
  num_teams : ℕ
  games_per_week : ℕ
  season_length_months : ℕ
  
/-- Calculates the total number of games played in a season -/
def total_games_played (league : BaseballLeague) : ℕ :=
  (league.num_teams * league.games_per_week * league.season_length_months * 4) / 2

/-- Theorem stating the total number of games played in the given league configuration -/
theorem baseball_league_games_played :
  ∃ (league : BaseballLeague),
    league.num_teams = 10 ∧
    league.games_per_week = 5 ∧
    league.season_length_months = 6 ∧
    total_games_played league = 600 := by
  sorry

end NUMINAMATH_CALUDE_baseball_league_games_played_l1211_121146


namespace NUMINAMATH_CALUDE_min_races_correct_l1211_121140

/-- Represents a race strategy to find the top 3 fastest horses -/
structure RaceStrategy where
  numRaces : ℕ
  ensuresTop3 : Bool

/-- The minimum number of races needed to find the top 3 fastest horses -/
def minRaces : ℕ := 7

/-- The total number of horses -/
def totalHorses : ℕ := 25

/-- The maximum number of horses that can race together -/
def maxHorsesPerRace : ℕ := 5

/-- Predicate to check if a race strategy is valid -/
def isValidStrategy (s : RaceStrategy) : Prop :=
  s.numRaces ≥ minRaces ∧ s.ensuresTop3

/-- Theorem stating that the minimum number of races is correct -/
theorem min_races_correct :
  ∀ s : RaceStrategy,
    isValidStrategy s →
    s.numRaces ≥ minRaces :=
sorry

end NUMINAMATH_CALUDE_min_races_correct_l1211_121140


namespace NUMINAMATH_CALUDE_expression_evaluation_l1211_121107

theorem expression_evaluation :
  let x : ℝ := 0
  let expr := (1 + 1 / (x - 2)) / ((x^2 - 2*x + 1) / (x - 2))
  expr = -1 := by sorry

end NUMINAMATH_CALUDE_expression_evaluation_l1211_121107


namespace NUMINAMATH_CALUDE_cupcakes_eaten_equals_packaged_l1211_121127

/-- Proves that the number of cupcakes Todd ate is equal to the number of cupcakes used for packaging -/
theorem cupcakes_eaten_equals_packaged (initial_cupcakes : ℕ) (num_packages : ℕ) (cupcakes_per_package : ℕ)
  (h1 : initial_cupcakes = 71)
  (h2 : num_packages = 4)
  (h3 : cupcakes_per_package = 7) :
  initial_cupcakes - (initial_cupcakes - num_packages * cupcakes_per_package) = num_packages * cupcakes_per_package :=
by sorry

end NUMINAMATH_CALUDE_cupcakes_eaten_equals_packaged_l1211_121127


namespace NUMINAMATH_CALUDE_parallel_vectors_k_value_l1211_121139

/-- Two vectors are parallel if one is a scalar multiple of the other -/
def are_parallel (v w : ℝ × ℝ) : Prop :=
  ∃ (c : ℝ), v.1 * w.2 = c * v.2 * w.1

theorem parallel_vectors_k_value :
  let a : ℝ × ℝ := (1, -2)
  let b : ℝ × ℝ := (3, 4)
  ∀ k : ℝ, are_parallel (a.1 - b.1, a.2 - b.2) (2 * a.1 + k * b.1, 2 * a.2 + k * b.2) →
    k = -2 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_k_value_l1211_121139


namespace NUMINAMATH_CALUDE_first_discount_percentage_l1211_121192

/-- Proves that the first discount is 15% given the initial price, final price, and second discount rate -/
theorem first_discount_percentage (initial_price final_price : ℝ) (second_discount : ℝ) :
  initial_price = 400 →
  final_price = 323 →
  second_discount = 0.05 →
  ∃ (first_discount : ℝ),
    first_discount = 0.15 ∧
    final_price = initial_price * (1 - first_discount) * (1 - second_discount) :=
by sorry

end NUMINAMATH_CALUDE_first_discount_percentage_l1211_121192


namespace NUMINAMATH_CALUDE_inequality_solution_set_l1211_121164

theorem inequality_solution_set (x : ℝ) : (x^2 + x) / (2*x - 1) ≤ 1 ↔ x < 1/2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l1211_121164


namespace NUMINAMATH_CALUDE_problem_statement_l1211_121100

theorem problem_statement (x y : ℝ) : 
  |x - 2| + (y + 3)^2 = 0 → (x + y)^2020 = 1 := by sorry

end NUMINAMATH_CALUDE_problem_statement_l1211_121100


namespace NUMINAMATH_CALUDE_inequality_sum_l1211_121184

theorem inequality_sum (a b c d : ℝ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : a > b) (h4 : c > d) : 
  a + c > b + d := by
  sorry

end NUMINAMATH_CALUDE_inequality_sum_l1211_121184


namespace NUMINAMATH_CALUDE_tangent_point_on_reciprocal_curve_l1211_121155

/-- Prove that the point of tangency on y = 1/x, where the tangent line passes through (0,2), is (1,1) -/
theorem tangent_point_on_reciprocal_curve :
  ∀ m n : ℝ,
  (n = 1 / m) →                         -- Point (m,n) is on the curve y = 1/x
  (2 - n) / m = -1 / (m^2) →            -- Tangent line passes through (0,2) with slope -1/m^2
  (m = 1 ∧ n = 1) :=                    -- The point of tangency is (1,1)
by sorry

end NUMINAMATH_CALUDE_tangent_point_on_reciprocal_curve_l1211_121155


namespace NUMINAMATH_CALUDE_linear_function_not_in_first_quadrant_implies_negative_k_and_b_l1211_121152

/-- A linear function that does not pass through the first quadrant -/
structure LinearFunctionNotInFirstQuadrant where
  k : ℝ
  b : ℝ
  not_in_first_quadrant : ∀ x y : ℝ, y = k * x + b → ¬(x > 0 ∧ y > 0)

/-- Theorem: If a linear function y = kx + b does not pass through the first quadrant, then k < 0 and b < 0 -/
theorem linear_function_not_in_first_quadrant_implies_negative_k_and_b 
  (f : LinearFunctionNotInFirstQuadrant) : f.k < 0 ∧ f.b < 0 := by
  sorry


end NUMINAMATH_CALUDE_linear_function_not_in_first_quadrant_implies_negative_k_and_b_l1211_121152


namespace NUMINAMATH_CALUDE_greatest_integer_less_than_neg_nineteen_fifths_l1211_121156

theorem greatest_integer_less_than_neg_nineteen_fifths :
  Int.floor (-19 / 5 : ℚ) = -4 := by sorry

end NUMINAMATH_CALUDE_greatest_integer_less_than_neg_nineteen_fifths_l1211_121156


namespace NUMINAMATH_CALUDE_find_divisor_l1211_121194

theorem find_divisor (dividend quotient remainder : ℕ) 
  (h1 : dividend = 12401)
  (h2 : quotient = 76)
  (h3 : remainder = 13)
  (h4 : dividend = quotient * (dividend / quotient) + remainder) : 
  dividend / quotient = 163 := by
sorry

end NUMINAMATH_CALUDE_find_divisor_l1211_121194


namespace NUMINAMATH_CALUDE_geometric_sequence_first_term_l1211_121159

/-- A geometric sequence is a sequence where each term after the first is found by multiplying the previous term by a fixed, non-zero number called the common ratio. -/
def IsGeometricSequence (a : ℕ → ℚ) : Prop :=
  ∃ r : ℚ, r ≠ 0 ∧ ∀ n : ℕ, a (n + 1) = a n * r

theorem geometric_sequence_first_term
  (a : ℕ → ℚ)
  (h_geometric : IsGeometricSequence a)
  (h_term2 : a 2 = 12)
  (h_term3 : a 3 = 24)
  (h_term6 : a 6 = 384) :
  a 1 = 3/4 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_first_term_l1211_121159


namespace NUMINAMATH_CALUDE_smallest_positive_largest_negative_expression_l1211_121134

theorem smallest_positive_largest_negative_expression :
  ∃ (a b c d : ℚ),
    (∀ n : ℚ, n > 0 → a ≤ n) ∧
    (∀ n : ℚ, n < 0 → n ≤ b) ∧
    (∀ n : ℚ, n ≠ 0 → |c| ≤ |n|) ∧
    (d⁻¹ = d) ∧
    (a - b + c^2 - |d| = 1) := by
  sorry

end NUMINAMATH_CALUDE_smallest_positive_largest_negative_expression_l1211_121134
