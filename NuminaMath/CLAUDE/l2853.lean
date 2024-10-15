import Mathlib

namespace NUMINAMATH_CALUDE_f_properties_l2853_285371

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (a * x) / (x + 1) - a

theorem f_properties (a : ℝ) (h : a ≠ 0) :
  (∀ x, f a x = f a x → x ∈ {x : ℝ | x < -1 ∨ x > -1}) ∧
  (∀ x, f a x = -f a (-x - 2)) :=
by sorry

end NUMINAMATH_CALUDE_f_properties_l2853_285371


namespace NUMINAMATH_CALUDE_polynomial_remainder_l2853_285346

/-- Given a polynomial p(x) such that p(2) = 7 and p(5) = 11,
    prove that the remainder when p(x) is divided by (x-2)(x-5) is (4/3)x + (13/3) -/
theorem polynomial_remainder (p : ℝ → ℝ) (h1 : p 2 = 7) (h2 : p 5 = 11) :
  ∃ q : ℝ → ℝ, ∀ x, p x = q x * (x - 2) * (x - 5) + (4/3 * x + 13/3) :=
sorry

end NUMINAMATH_CALUDE_polynomial_remainder_l2853_285346


namespace NUMINAMATH_CALUDE_class_size_proof_l2853_285356

theorem class_size_proof (boys_ratio : Nat) (girls_ratio : Nat) (num_girls : Nat) :
  boys_ratio = 5 →
  girls_ratio = 8 →
  num_girls = 160 →
  (boys_ratio + girls_ratio : Rat) * (num_girls / girls_ratio : Rat) = 260 :=
by sorry

end NUMINAMATH_CALUDE_class_size_proof_l2853_285356


namespace NUMINAMATH_CALUDE_lineup_combinations_l2853_285338

/-- The number of ways to choose a starting lineup for a football team -/
def choose_lineup (total_players : ℕ) (offensive_linemen : ℕ) : ℕ :=
  offensive_linemen * (total_players - 1) * (total_players - 2) * (total_players - 3)

/-- Theorem stating the number of ways to choose a starting lineup for the given team composition -/
theorem lineup_combinations : choose_lineup 12 4 = 3960 := by
  sorry

end NUMINAMATH_CALUDE_lineup_combinations_l2853_285338


namespace NUMINAMATH_CALUDE_opposite_sqrt_81_l2853_285362

theorem opposite_sqrt_81 : -(Real.sqrt 81) = -9 := by
  sorry

end NUMINAMATH_CALUDE_opposite_sqrt_81_l2853_285362


namespace NUMINAMATH_CALUDE_symmetry_properties_l2853_285326

-- Define the shapes
inductive Shape
  | Parallelogram
  | Rectangle
  | Square
  | Rhombus
  | IsoscelesTrapezoid

-- Define the symmetry properties
def isAxisymmetric (s : Shape) : Prop :=
  match s with
  | Shape.Rectangle => true
  | Shape.Square => true
  | Shape.Rhombus => true
  | Shape.IsoscelesTrapezoid => true
  | _ => false

def isCentrallySymmetric (s : Shape) : Prop :=
  match s with
  | Shape.Parallelogram => true
  | Shape.Rectangle => true
  | Shape.Square => true
  | Shape.Rhombus => true
  | _ => false

-- Theorem statement
theorem symmetry_properties :
  ∀ s : Shape,
    (isAxisymmetric s ∧ isCentrallySymmetric s) ↔
    (s = Shape.Rectangle ∨ s = Shape.Square ∨ s = Shape.Rhombus) :=
by sorry

end NUMINAMATH_CALUDE_symmetry_properties_l2853_285326


namespace NUMINAMATH_CALUDE_sequence_is_constant_l2853_285370

/-- A sequence of positive integers -/
def Sequence := ℕ → ℕ

/-- The divisibility condition for the sequence -/
def DivisibilityCondition (a : Sequence) : Prop :=
  ∀ i j : ℕ, i > j → (i - j)^(2*(i - j)) + 1 ∣ a i - a j

/-- The theorem stating that a sequence satisfying the divisibility condition is constant -/
theorem sequence_is_constant (a : Sequence) (h : DivisibilityCondition a) :
  ∀ n m : ℕ, a n = a m :=
sorry

end NUMINAMATH_CALUDE_sequence_is_constant_l2853_285370


namespace NUMINAMATH_CALUDE_prove_trip_length_l2853_285343

def trip_length : ℚ := 360 / 7

theorem prove_trip_length :
  let first_part : ℚ := 1 / 4
  let second_part : ℚ := 30
  let third_part : ℚ := 1 / 6
  (first_part + third_part + second_part / trip_length = 1) →
  trip_length = 360 / 7 := by
sorry

end NUMINAMATH_CALUDE_prove_trip_length_l2853_285343


namespace NUMINAMATH_CALUDE_max_regions_formula_l2853_285381

/-- The maximum number of regions formed by n lines in a plane -/
def max_regions (n : ℕ) : ℚ :=
  (n^2 + n + 2) / 2

/-- Conditions for the lines in the plane -/
structure PlaneLines where
  n : ℕ
  n_ge_3 : n ≥ 3
  no_parallel : True  -- represents the condition that no two lines are parallel
  no_triple_intersection : True  -- represents the condition that no three lines intersect at the same point

theorem max_regions_formula (p : PlaneLines) :
  max_regions p.n = (p.n^2 + p.n + 2) / 2 :=
sorry

end NUMINAMATH_CALUDE_max_regions_formula_l2853_285381


namespace NUMINAMATH_CALUDE_total_cost_is_correct_l2853_285388

/-- Represents the quantity of an ingredient on a given day -/
structure IngredientQuantity where
  day1 : Float
  day7 : Float

/-- Represents the price of an ingredient -/
structure IngredientPrice where
  value : Float
  unit : String

/-- Represents an ingredient with its quantities and price -/
structure Ingredient where
  name : String
  quantity : IngredientQuantity
  price : IngredientPrice
  unit : String

def ingredients : List Ingredient := [
  { name := "Baking powder",
    quantity := { day1 := 12, day7 := 6 },
    price := { value := 3, unit := "per pound" },
    unit := "lbs" },
  { name := "Flour",
    quantity := { day1 := 6, day7 := 3.5 },
    price := { value := 1.5, unit := "per pound" },
    unit := "kg" },
  { name := "Sugar",
    quantity := { day1 := 20, day7 := 15 },
    price := { value := 0.5, unit := "per pound" },
    unit := "lbs" },
  { name := "Chocolate chips",
    quantity := { day1 := 5000, day7 := 1500 },
    price := { value := 0.015, unit := "per gram" },
    unit := "g" }
]

def kgToPounds : Float := 2.20462
def gToPounds : Float := 0.00220462

def calculateTotalCost (ingredients : List Ingredient) : Float :=
  sorry

theorem total_cost_is_correct :
  calculateTotalCost ingredients = 81.27 := by
  sorry

end NUMINAMATH_CALUDE_total_cost_is_correct_l2853_285388


namespace NUMINAMATH_CALUDE_batsman_average_after_12th_innings_l2853_285398

/-- Calculates the new average of a batsman after 12 innings -/
def new_average (previous_total : ℕ) (new_score : ℕ) : ℚ :=
  (previous_total + new_score) / 12

/-- Represents the increase in average after the 12th innings -/
def average_increase (previous_average : ℚ) (new_average : ℚ) : ℚ :=
  new_average - previous_average

theorem batsman_average_after_12th_innings 
  (previous_total : ℕ) 
  (previous_average : ℚ) 
  (new_score : ℕ) :
  previous_total = previous_average * 11 →
  new_score = 115 →
  average_increase previous_average (new_average previous_total new_score) = 3 →
  new_average previous_total new_score = 82 :=
by sorry

end NUMINAMATH_CALUDE_batsman_average_after_12th_innings_l2853_285398


namespace NUMINAMATH_CALUDE_hyperbola_equation_l2853_285373

/-- Given two hyperbolas with the same asymptotes and a specific focus, prove the equation of one hyperbola -/
theorem hyperbola_equation (a b : ℝ) (ha : a > 0) (hb : b > 0) : 
  (∀ x y : ℝ, x^2/a^2 - y^2/b^2 = 1) →  -- Equation of C₁
  (∀ x y : ℝ, x^2/4 - y^2/16 = 1) →    -- Equation of C₂
  (b/a = 2) →                          -- Same asymptotes condition
  (a^2 + b^2 = 5) →                    -- Right focus condition
  (∀ x y : ℝ, x^2 - y^2/4 = 1) :=      -- Conclusion: Equation of C₁
by sorry


end NUMINAMATH_CALUDE_hyperbola_equation_l2853_285373


namespace NUMINAMATH_CALUDE_base_c_problem_l2853_285386

/-- Representation of a number in base c -/
def baseC (n : ℕ) (c : ℕ) : ℕ → ℕ
| 0 => n % c
| i + 1 => baseC (n / c) c i

/-- Given that in base c, 33_c squared equals 1201_c, prove that c = 10 -/
theorem base_c_problem (c : ℕ) (h : c > 1) :
  (baseC 33 c 1 * c + baseC 33 c 0)^2 = 
  baseC 1201 c 3 * c^3 + baseC 1201 c 2 * c^2 + baseC 1201 c 1 * c + baseC 1201 c 0 →
  c = 10 := by
sorry

end NUMINAMATH_CALUDE_base_c_problem_l2853_285386


namespace NUMINAMATH_CALUDE_gcd_102_238_l2853_285395

theorem gcd_102_238 : Nat.gcd 102 238 = 34 := by
  sorry

end NUMINAMATH_CALUDE_gcd_102_238_l2853_285395


namespace NUMINAMATH_CALUDE_EF_is_one_eighth_of_GH_l2853_285379

-- Define the line segment GH and points E and F on it
variable (G H E F : Real)

-- Define the condition that E and F lie on GH
axiom E_on_GH : G ≤ E ∧ E ≤ H
axiom F_on_GH : G ≤ F ∧ F ≤ H

-- Define the length ratios
axiom GE_ratio : E - G = 3 * (H - E)
axiom GF_ratio : F - G = 7 * (H - F)

-- State the theorem to be proved
theorem EF_is_one_eighth_of_GH : (F - E) = (1/8) * (H - G) := by
  sorry

end NUMINAMATH_CALUDE_EF_is_one_eighth_of_GH_l2853_285379


namespace NUMINAMATH_CALUDE_geometric_sequence_ratio_l2853_285383

/-- Given a geometric sequence with first term a₁ and common ratio q,
    S₃ represents the sum of the first 3 terms -/
def S₃ (a₁ q : ℝ) : ℝ := a₁ + a₁ * q + a₁ * q^2

/-- Theorem: For a geometric sequence with common ratio q,
    if S₃ = 7a₁, then q = 2 or q = -3 -/
theorem geometric_sequence_ratio (a₁ q : ℝ) (h : a₁ ≠ 0) :
  S₃ a₁ q = 7 * a₁ → q = 2 ∨ q = -3 := by
  sorry


end NUMINAMATH_CALUDE_geometric_sequence_ratio_l2853_285383


namespace NUMINAMATH_CALUDE_pencil_box_puzzle_l2853_285316

structure Box where
  blue : ℕ
  green : ℕ

def vasya_statement (box : Box) : Prop :=
  box.blue ≥ 4

def kolya_statement (box : Box) : Prop :=
  box.green ≥ 5

def petya_statement (box : Box) : Prop :=
  box.blue ≥ 3 ∧ box.green ≥ 4

def misha_statement (box : Box) : Prop :=
  box.blue ≥ 4 ∧ box.green ≥ 4

theorem pencil_box_puzzle (box : Box) :
  (vasya_statement box ∧ ¬kolya_statement box ∧ petya_statement box ∧ misha_statement box) ↔
  (box.blue ≥ 4 ∧ box.green = 4) :=
by sorry

end NUMINAMATH_CALUDE_pencil_box_puzzle_l2853_285316


namespace NUMINAMATH_CALUDE_parallel_to_same_line_implies_parallel_l2853_285300

-- Define a type for lines in a plane
variable {Line : Type}

-- Define a relation for parallel lines
variable (parallel : Line → Line → Prop)

-- Axiom: Parallel relation is symmetric
axiom parallel_symmetric {l1 l2 : Line} : parallel l1 l2 → parallel l2 l1

-- Axiom: Parallel relation is transitive
axiom parallel_transitive {l1 l2 l3 : Line} : parallel l1 l2 → parallel l2 l3 → parallel l1 l3

-- Theorem: If two lines are parallel to a third line, they are parallel to each other
theorem parallel_to_same_line_implies_parallel (l1 l2 l3 : Line) :
  parallel l1 l3 → parallel l2 l3 → parallel l1 l2 :=
by sorry

end NUMINAMATH_CALUDE_parallel_to_same_line_implies_parallel_l2853_285300


namespace NUMINAMATH_CALUDE_system_solution_range_l2853_285337

theorem system_solution_range (a x y : ℝ) : 
  x - y = a + 3 →
  2 * x + y = 5 * a →
  x < y →
  a < -3 := by
sorry

end NUMINAMATH_CALUDE_system_solution_range_l2853_285337


namespace NUMINAMATH_CALUDE_twelfth_term_of_specific_sequence_l2853_285322

/-- The nth term of a geometric sequence with first term a and common ratio r -/
def geometric_sequence (a : ℚ) (r : ℚ) (n : ℕ) : ℚ := a * r ^ (n - 1)

/-- The 12th term of the specific geometric sequence is 1/6561 -/
theorem twelfth_term_of_specific_sequence :
  geometric_sequence 27 (1/3) 12 = 1/6561 := by sorry

end NUMINAMATH_CALUDE_twelfth_term_of_specific_sequence_l2853_285322


namespace NUMINAMATH_CALUDE_coordinate_points_count_l2853_285367

theorem coordinate_points_count (S : Finset ℕ) (h : S = {1, 2, 3, 4, 5}) :
  Finset.card (Finset.product S S) = 25 := by
  sorry

end NUMINAMATH_CALUDE_coordinate_points_count_l2853_285367


namespace NUMINAMATH_CALUDE_weight_difference_is_35_l2853_285339

def labrador_start : ℝ := 40
def dachshund_start : ℝ := 12
def weight_gain_percentage : ℝ := 0.25

def weight_difference : ℝ :=
  (labrador_start + labrador_start * weight_gain_percentage) -
  (dachshund_start + dachshund_start * weight_gain_percentage)

theorem weight_difference_is_35 : weight_difference = 35 := by
  sorry

end NUMINAMATH_CALUDE_weight_difference_is_35_l2853_285339


namespace NUMINAMATH_CALUDE_fraction_of_girls_l2853_285384

theorem fraction_of_girls (total : ℕ) (boys : ℕ) (h1 : total = 45) (h2 : boys = 30) :
  (total - boys : ℚ) / total = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_fraction_of_girls_l2853_285384


namespace NUMINAMATH_CALUDE_quadrilateral_area_is_0125_l2853_285336

/-- The area of the quadrilateral formed by the intersection of four lines -/
def quadrilateral_area (line1 line2 : ℝ → ℝ → Prop) (x_line y_line : ℝ → Prop) : ℝ := sorry

/-- The first line: 3x + 4y - 12 = 0 -/
def line1 (x y : ℝ) : Prop := 3 * x + 4 * y - 12 = 0

/-- The second line: 5x - 4y - 10 = 0 -/
def line2 (x y : ℝ) : Prop := 5 * x - 4 * y - 10 = 0

/-- The vertical line: x = 3 -/
def x_line (x : ℝ) : Prop := x = 3

/-- The horizontal line: y = 1 -/
def y_line (y : ℝ) : Prop := y = 1

theorem quadrilateral_area_is_0125 : 
  quadrilateral_area line1 line2 x_line y_line = 0.125 := by sorry

end NUMINAMATH_CALUDE_quadrilateral_area_is_0125_l2853_285336


namespace NUMINAMATH_CALUDE_oleg_can_win_l2853_285306

/-- The nth prime number -/
def nthPrime (n : ℕ) : ℕ := sorry

/-- A list of n positive integers, all smaller than the nth prime -/
def validList (n : ℕ) (list : List ℕ) : Prop :=
  list.length = n ∧ 
  ∀ x ∈ list, 0 < x ∧ x < nthPrime n

/-- The operation of replacing one number with the product of two numbers -/
def replaceWithProduct (list : List ℕ) (i j k : ℕ) : List ℕ :=
  sorry

/-- Predicate to check if a list contains at least two equal elements -/
def hasEqualElements (list : List ℕ) : Prop :=
  ∃ i j, i ≠ j ∧ list.get! i = list.get! j

/-- The main theorem: Oleg can always win for n > 1 -/
theorem oleg_can_win (n : ℕ) (list : List ℕ) (h : n > 1) (hlist : validList n list) :
  ∃ (steps : List (ℕ × ℕ × ℕ)), 
    let finalList := steps.foldl (fun acc step => replaceWithProduct acc step.1 step.2.1 step.2.2) list
    hasEqualElements finalList :=
  sorry

end NUMINAMATH_CALUDE_oleg_can_win_l2853_285306


namespace NUMINAMATH_CALUDE_min_sum_bound_min_sum_achievable_l2853_285335

theorem min_sum_bound (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  a / (3 * b) + b / (5 * c) + c / (6 * a) ≥ 3 / Real.rpow 90 (1/3) :=
sorry

theorem min_sum_achievable :
  ∃ (a b c : ℝ), a > 0 ∧ b > 0 ∧ c > 0 ∧
    a / (3 * b) + b / (5 * c) + c / (6 * a) = 3 / Real.rpow 90 (1/3) :=
sorry

end NUMINAMATH_CALUDE_min_sum_bound_min_sum_achievable_l2853_285335


namespace NUMINAMATH_CALUDE_three_cakes_cooking_time_l2853_285315

/-- Represents the cooking process for cakes -/
structure CookingProcess where
  pot_capacity : ℕ
  cooking_time_per_cake : ℕ
  num_cakes : ℕ

/-- The minimum time required to cook the given number of cakes -/
def min_cooking_time (process : CookingProcess) : ℕ :=
  sorry

/-- Theorem stating the minimum time to cook three cakes under given conditions -/
theorem three_cakes_cooking_time :
  ∀ (process : CookingProcess),
    process.pot_capacity = 2 →
    process.cooking_time_per_cake = 5 →
    process.num_cakes = 3 →
    min_cooking_time process = 15 :=
by sorry

end NUMINAMATH_CALUDE_three_cakes_cooking_time_l2853_285315


namespace NUMINAMATH_CALUDE_min_triangle_perimeter_l2853_285368

theorem min_triangle_perimeter (a b x : ℕ) (ha : a = 24) (hb : b = 51) : 
  (a + b + x > a + b ∧ a + x > b ∧ b + x > a) → (∀ y : ℕ, (a + b + y > a + b ∧ a + y > b ∧ b + y > a) → x ≤ y) 
  → a + b + x = 103 :=
sorry

end NUMINAMATH_CALUDE_min_triangle_perimeter_l2853_285368


namespace NUMINAMATH_CALUDE_spring_ice_cream_percentage_spring_ice_cream_percentage_proof_l2853_285341

theorem spring_ice_cream_percentage : ℝ → Prop :=
  fun spring_percentage =>
    (spring_percentage + 30 + 25 + 20 = 100) →
    spring_percentage = 25

-- The proof is omitted
theorem spring_ice_cream_percentage_proof : spring_ice_cream_percentage 25 := by
  sorry

end NUMINAMATH_CALUDE_spring_ice_cream_percentage_spring_ice_cream_percentage_proof_l2853_285341


namespace NUMINAMATH_CALUDE_unique_function_theorem_l2853_285310

def is_surjective (f : ℕ → ℕ) : Prop :=
  ∀ y : ℕ, ∃ x : ℕ, f x = y

def satisfies_condition (f : ℕ → ℕ) : Prop :=
  ∀ (m n : ℕ) (p : ℕ), Nat.Prime p →
    (p ∣ f (m + n) ↔ p ∣ (f m + f n))

theorem unique_function_theorem :
  ∃! f : ℕ → ℕ, is_surjective f ∧ satisfies_condition f :=
by
  sorry

end NUMINAMATH_CALUDE_unique_function_theorem_l2853_285310


namespace NUMINAMATH_CALUDE_certain_number_proof_l2853_285369

theorem certain_number_proof (x : ℝ) : 144 / x = 14.4 / 0.0144 → x = 0.144 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_proof_l2853_285369


namespace NUMINAMATH_CALUDE_inequality_solution_l2853_285348

theorem inequality_solution (x : ℝ) : (x^2 - 49) / (x + 7) < 0 ↔ -7 < x ∧ x < 7 := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_l2853_285348


namespace NUMINAMATH_CALUDE_susannah_swims_24_times_l2853_285345

/-- The number of times Camden went swimming in March -/
def camden_swims : ℕ := 16

/-- The number of weeks in March -/
def weeks_in_march : ℕ := 4

/-- The number of times Camden swam per week -/
def camden_swims_per_week : ℕ := camden_swims / weeks_in_march

/-- The number of additional times Susannah swam per week compared to Camden -/
def susannah_additional_swims : ℕ := 2

/-- The number of times Susannah swam per week -/
def susannah_swims_per_week : ℕ := camden_swims_per_week + susannah_additional_swims

/-- The total number of times Susannah went swimming in March -/
def susannah_total_swims : ℕ := susannah_swims_per_week * weeks_in_march

theorem susannah_swims_24_times : susannah_total_swims = 24 := by
  sorry

end NUMINAMATH_CALUDE_susannah_swims_24_times_l2853_285345


namespace NUMINAMATH_CALUDE_fraction_simplification_l2853_285342

theorem fraction_simplification (x y : ℝ) (hx : -x ≥ 0) (hy : -y ≥ 0) :
  (Real.sqrt (-x) - Real.sqrt (-3 * y)) / (x + 3 * y + 2 * Real.sqrt (3 * x * y)) =
  1 / (Real.sqrt (-3 * y) - Real.sqrt (-x)) :=
by sorry

end NUMINAMATH_CALUDE_fraction_simplification_l2853_285342


namespace NUMINAMATH_CALUDE_cos_120_degrees_l2853_285361

theorem cos_120_degrees : Real.cos (120 * π / 180) = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_cos_120_degrees_l2853_285361


namespace NUMINAMATH_CALUDE_right_triangle_perimeter_l2853_285303

/-- Given a right triangle with area 180 square units and one leg of length 18 units,
    its perimeter is 38 + 2√181 units. -/
theorem right_triangle_perimeter : ∀ a b c : ℝ,
  a > 0 → b > 0 → c > 0 →
  a * b / 2 = 180 →
  a = 18 →
  a^2 + b^2 = c^2 →
  a + b + c = 38 + 2 * Real.sqrt 181 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_perimeter_l2853_285303


namespace NUMINAMATH_CALUDE_hash_seven_two_l2853_285377

-- Define the # operation
def hash (a b : ℤ) : ℤ := 4 * a - 2 * b

-- State the theorem
theorem hash_seven_two : hash 7 2 = 24 := by
  sorry

end NUMINAMATH_CALUDE_hash_seven_two_l2853_285377


namespace NUMINAMATH_CALUDE_storks_on_fence_l2853_285397

/-- The number of storks that joined the birds on the fence -/
def storks_joined : ℕ := 6

/-- The initial number of birds on the fence -/
def initial_birds : ℕ := 3

/-- The number of additional birds that joined -/
def additional_birds : ℕ := 2

theorem storks_on_fence :
  storks_joined = initial_birds + additional_birds + 1 :=
by sorry

end NUMINAMATH_CALUDE_storks_on_fence_l2853_285397


namespace NUMINAMATH_CALUDE_vacation_expense_sharing_l2853_285380

/-- The vacation expense sharing problem -/
theorem vacation_expense_sharing 
  (alex kim lee nina : ℝ)
  (h_alex : alex = 130)
  (h_kim : kim = 150)
  (h_lee : lee = 170)
  (h_nina : nina = 200)
  (h_total : alex + kim + lee + nina = 650)
  (h_equal_share : (alex + kim + lee + nina) / 4 = 162.5)
  (a k : ℝ)
  (h_a : a = 162.5 - alex)
  (h_k : k = 162.5 - kim) :
  a - k = 20 := by
sorry

end NUMINAMATH_CALUDE_vacation_expense_sharing_l2853_285380


namespace NUMINAMATH_CALUDE_cylinder_height_in_hemisphere_l2853_285393

theorem cylinder_height_in_hemisphere (r_cylinder r_hemisphere : ℝ) (h_cylinder_radius : r_cylinder = 3)
  (h_hemisphere_radius : r_hemisphere = 7) (h_inscribed : r_cylinder ≤ r_hemisphere) :
  let height := Real.sqrt (r_hemisphere^2 - r_cylinder^2)
  height = 2 * Real.sqrt 10 :=
by sorry

end NUMINAMATH_CALUDE_cylinder_height_in_hemisphere_l2853_285393


namespace NUMINAMATH_CALUDE_books_read_in_week_l2853_285321

/-- The number of books Mrs. Hilt reads per day -/
def books_per_day : ℕ := 2

/-- The number of days in a week -/
def days_in_week : ℕ := 7

/-- Theorem: Mrs. Hilt reads 14 books in one week -/
theorem books_read_in_week : books_per_day * days_in_week = 14 := by
  sorry

end NUMINAMATH_CALUDE_books_read_in_week_l2853_285321


namespace NUMINAMATH_CALUDE_season_games_count_l2853_285314

/-- The number of teams in the sports conference -/
def total_teams : ℕ := 16

/-- The number of divisions in the sports conference -/
def num_divisions : ℕ := 2

/-- The number of teams in each division -/
def teams_per_division : ℕ := 8

/-- The number of times each team plays other teams in its own division -/
def intra_division_games : ℕ := 3

/-- The number of times each team plays teams in the other division -/
def inter_division_games : ℕ := 2

/-- The total number of games in a complete season -/
def total_games : ℕ := 296

theorem season_games_count :
  total_teams = num_divisions * teams_per_division ∧
  (teams_per_division * (teams_per_division - 1) / 2) * intra_division_games * num_divisions +
  (teams_per_division * teams_per_division * inter_division_games) = total_games := by
  sorry

end NUMINAMATH_CALUDE_season_games_count_l2853_285314


namespace NUMINAMATH_CALUDE_avg_days_before_trial_is_four_l2853_285389

/-- The average number of days spent in jail before trial -/
def avg_days_before_trial (num_cities num_days arrests_per_day total_weeks : ℕ) : ℚ :=
  let total_arrests := num_cities * num_days * arrests_per_day
  let total_jail_days := total_weeks * 7
  let days_after_trial := 7
  (total_jail_days / total_arrests : ℚ) - days_after_trial

theorem avg_days_before_trial_is_four :
  avg_days_before_trial 21 30 10 9900 = 4 := by
  sorry

end NUMINAMATH_CALUDE_avg_days_before_trial_is_four_l2853_285389


namespace NUMINAMATH_CALUDE_greatest_integer_for_all_real_domain_greatest_integer_value_l2853_285308

theorem greatest_integer_for_all_real_domain (b : ℤ) : 
  (∀ x : ℝ, (x^2 + b*x + 5 ≠ 0)) ↔ b^2 < 20 :=
by sorry

theorem greatest_integer_value : 
  ∃ b : ℤ, b = 4 ∧ (∀ x : ℝ, (x^2 + b*x + 5 ≠ 0)) ∧ 
  (∀ c : ℤ, c > b → ∃ x : ℝ, (x^2 + c*x + 5 = 0)) :=
by sorry

end NUMINAMATH_CALUDE_greatest_integer_for_all_real_domain_greatest_integer_value_l2853_285308


namespace NUMINAMATH_CALUDE_min_value_of_expression_l2853_285365

theorem min_value_of_expression : 
  ∃ (min : ℝ), min = Real.sqrt 2 * Real.sqrt 5 ∧ 
  ∀ (x : ℝ), Real.sqrt (x^2 + (1 + 2*x)^2) + Real.sqrt ((x - 1)^2 + (x - 1)^2) ≥ min :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_expression_l2853_285365


namespace NUMINAMATH_CALUDE_binomial_55_3_l2853_285319

theorem binomial_55_3 : Nat.choose 55 3 = 26235 := by
  sorry

end NUMINAMATH_CALUDE_binomial_55_3_l2853_285319


namespace NUMINAMATH_CALUDE_rectangle_ratio_l2853_285325

theorem rectangle_ratio (l w : ℝ) (hl : l = 10) (hp : 2 * l + 2 * w = 36) :
  w / l = 4 / 5 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_ratio_l2853_285325


namespace NUMINAMATH_CALUDE_constant_discount_increase_l2853_285333

/-- Represents the discount percentage for a given number of pizzas -/
def discount (n : ℕ) : ℚ :=
  match n with
  | 1 => 0
  | 2 => 4/100
  | 3 => 8/100
  | _ => 0  -- Default case, not used in this problem

/-- The theorem states that the discount increase is constant -/
theorem constant_discount_increase :
  ∃ (r : ℚ), (discount 2 - discount 1 = r) ∧ (discount 3 - discount 2 = r) ∧ (r = 4/100) := by
  sorry

#check constant_discount_increase

end NUMINAMATH_CALUDE_constant_discount_increase_l2853_285333


namespace NUMINAMATH_CALUDE_sin_three_pi_fourth_minus_alpha_l2853_285331

theorem sin_three_pi_fourth_minus_alpha (α : ℝ) 
  (h : Real.sin (π / 4 + α) = Real.sqrt 3 / 2) : 
  Real.sin (3 * π / 4 - α) = Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_three_pi_fourth_minus_alpha_l2853_285331


namespace NUMINAMATH_CALUDE_f_properties_l2853_285390

def f (x : ℝ) : ℝ := x^2 + x - 6

theorem f_properties :
  (f 0 = -6) ∧ (∀ x : ℝ, f x = 0 → x = -3 ∨ x = 2) := by
  sorry

end NUMINAMATH_CALUDE_f_properties_l2853_285390


namespace NUMINAMATH_CALUDE_sin_70_cos_20_plus_cos_70_sin_20_l2853_285344

theorem sin_70_cos_20_plus_cos_70_sin_20 : 
  Real.sin (70 * π / 180) * Real.cos (20 * π / 180) + 
  Real.cos (70 * π / 180) * Real.sin (20 * π / 180) = 1 := by
  sorry

end NUMINAMATH_CALUDE_sin_70_cos_20_plus_cos_70_sin_20_l2853_285344


namespace NUMINAMATH_CALUDE_trapezoid_angles_l2853_285357

-- Define a trapezoid
structure Trapezoid where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ
  sum_360 : a + b + c + d = 360
  sum_ab_180 : a + b = 180
  sum_cd_180 : c + d = 180

-- Theorem statement
theorem trapezoid_angles (t : Trapezoid) (h1 : t.a = 60) (h2 : t.b = 130) :
  t.c = 50 ∧ t.d = 120 := by
  sorry

end NUMINAMATH_CALUDE_trapezoid_angles_l2853_285357


namespace NUMINAMATH_CALUDE_sphere_radius_from_surface_area_l2853_285330

theorem sphere_radius_from_surface_area :
  ∀ (r : ℝ), (4 : ℝ) * Real.pi * r^2 = (4 : ℝ) * Real.pi → r = 1 := by
  sorry

end NUMINAMATH_CALUDE_sphere_radius_from_surface_area_l2853_285330


namespace NUMINAMATH_CALUDE_collin_cans_at_home_l2853_285349

/-- The number of cans Collin found at home -/
def cans_at_home : ℕ := sorry

/-- The amount earned per can in cents -/
def cents_per_can : ℕ := 25

/-- The number of cans from the neighbor -/
def cans_from_neighbor : ℕ := 46

/-- The number of cans from dad's office -/
def cans_from_office : ℕ := 250

/-- The amount Collin has to put into savings in cents -/
def savings_amount : ℕ := 4300

theorem collin_cans_at_home :
  cans_at_home = 12 ∧
  cents_per_can * (cans_at_home + 3 * cans_at_home + cans_from_neighbor + cans_from_office) = 2 * savings_amount :=
by sorry

end NUMINAMATH_CALUDE_collin_cans_at_home_l2853_285349


namespace NUMINAMATH_CALUDE_greatest_c_for_no_real_solutions_l2853_285387

theorem greatest_c_for_no_real_solutions : 
  (∃ c : ℤ, c = (Nat.floor (Real.sqrt 116)) ∧ 
   ∀ x : ℝ, x^2 + c*x + 29 ≠ 0 ∧
   ∀ d : ℤ, d > c → ∃ x : ℝ, x^2 + d*x + 29 = 0) ∧
  (Nat.floor (Real.sqrt 116) = 10) :=
by sorry

end NUMINAMATH_CALUDE_greatest_c_for_no_real_solutions_l2853_285387


namespace NUMINAMATH_CALUDE_probability_sum_seven_l2853_285307

/-- Represents the faces of the first die -/
def die1 : Finset ℕ := {1, 3, 5}

/-- Represents the faces of the second die -/
def die2 : Finset ℕ := {2, 4, 6}

/-- The total number of possible outcomes when rolling both dice -/
def total_outcomes : ℕ := 36

/-- The number of favorable outcomes (sum of 7) -/
def favorable_outcomes : ℕ := 12

/-- Theorem stating that the probability of rolling a sum of 7 is 1/3 -/
theorem probability_sum_seven :
  (favorable_outcomes : ℚ) / total_outcomes = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_probability_sum_seven_l2853_285307


namespace NUMINAMATH_CALUDE_marys_unique_score_l2853_285327

/-- Represents the score in a mathematics competition. -/
structure Score where
  total : ℕ
  correct : ℕ
  wrong : ℕ
  h_total : total = 35 + 5 * correct - 2 * wrong

/-- Determines if a score is uniquely determinable. -/
def isUniqueDeterminable (s : Score) : Prop :=
  ∀ s' : Score, s'.total = s.total → s'.correct = s.correct ∧ s'.wrong = s.wrong

/-- The theorem stating Mary's unique score. -/
theorem marys_unique_score :
  ∃! s : Score,
    s.total > 90 ∧
    isUniqueDeterminable s ∧
    ∀ s' : Score, s'.total > 90 ∧ s'.total < s.total → ¬isUniqueDeterminable s' :=
by sorry

end NUMINAMATH_CALUDE_marys_unique_score_l2853_285327


namespace NUMINAMATH_CALUDE_banana_price_is_60_cents_l2853_285329

def apple_price : ℚ := 0.70
def total_cost : ℚ := 5.60
def total_fruits : ℕ := 9

theorem banana_price_is_60_cents :
  ∃ (num_apples num_bananas : ℕ) (banana_price : ℚ),
    num_apples + num_bananas = total_fruits ∧
    num_apples * apple_price + num_bananas * banana_price = total_cost ∧
    banana_price = 0.60 := by
  sorry

end NUMINAMATH_CALUDE_banana_price_is_60_cents_l2853_285329


namespace NUMINAMATH_CALUDE_inverse_uniqueness_non_commutative_non_unique_inverse_l2853_285313

/-- A binary operation on a type α -/
def BinaryOp (α : Type) := α → α → α

/-- The inverse of a binary operation -/
def InverseOp (α : Type) (op : BinaryOp α) := BinaryOp α

/-- Property of an inverse operation -/
def IsInverse {α : Type} (op : BinaryOp α) (inv : InverseOp α op) : Prop :=
  ∀ a b c : α, op a b = c → inv c b = a ∧ op (inv c b) b = c

/-- Uniqueness of inverse operation -/
theorem inverse_uniqueness {α : Type} (op : BinaryOp α) :
  ∃! inv : InverseOp α op, IsInverse op inv :=
sorry

/-- Non-uniqueness for non-commutative operations -/
theorem non_commutative_non_unique_inverse {α : Type} (op : BinaryOp α) :
  (∃ a b : α, op a b ≠ op b a) →
  ¬∃! inv : InverseOp α op, IsInverse op inv :=
sorry

end NUMINAMATH_CALUDE_inverse_uniqueness_non_commutative_non_unique_inverse_l2853_285313


namespace NUMINAMATH_CALUDE_ink_cost_per_ml_l2853_285323

/-- Proves that the cost of ink per milliliter is 50 cents given the specified conditions -/
theorem ink_cost_per_ml (num_classes : ℕ) (boards_per_class : ℕ) (ink_per_board : ℕ) (total_cost : ℕ) : 
  num_classes = 5 → 
  boards_per_class = 2 → 
  ink_per_board = 20 → 
  total_cost = 100 → 
  (total_cost * 100) / (num_classes * boards_per_class * ink_per_board) = 50 := by
  sorry

#check ink_cost_per_ml

end NUMINAMATH_CALUDE_ink_cost_per_ml_l2853_285323


namespace NUMINAMATH_CALUDE_donut_selection_problem_l2853_285396

theorem donut_selection_problem :
  let n : ℕ := 6  -- number of donuts to select
  let k : ℕ := 4  -- number of donut types
  Nat.choose (n + k - 1) (k - 1) = 84 :=
by sorry

end NUMINAMATH_CALUDE_donut_selection_problem_l2853_285396


namespace NUMINAMATH_CALUDE_parallelogram_height_base_difference_l2853_285302

theorem parallelogram_height_base_difference 
  (area : ℝ) (base : ℝ) (height : ℝ) 
  (h_area : area = 24) 
  (h_base : base = 4) 
  (h_parallelogram : area = base * height) : 
  height - base = 2 := by
sorry

end NUMINAMATH_CALUDE_parallelogram_height_base_difference_l2853_285302


namespace NUMINAMATH_CALUDE_tangent_circle_circumference_is_36_l2853_285340

/-- Represents a geometric setup with two circular arcs and a tangent circle -/
structure GeometricSetup where
  -- The length of arc BC
  arc_length : ℝ
  -- Predicate that the arcs subtend 90° angles
  subtend_right_angle : Prop
  -- Predicate that the circle is tangent to both arcs and line segment AB
  circle_tangent : Prop

/-- The circumference of the tangent circle in the given geometric setup -/
def tangent_circle_circumference (setup : GeometricSetup) : ℝ :=
  sorry

/-- Theorem stating that the circumference of the tangent circle is 36 -/
theorem tangent_circle_circumference_is_36 (setup : GeometricSetup) 
  (h1 : setup.arc_length = 18)
  (h2 : setup.subtend_right_angle)
  (h3 : setup.circle_tangent) :
  tangent_circle_circumference setup = 36 :=
sorry

end NUMINAMATH_CALUDE_tangent_circle_circumference_is_36_l2853_285340


namespace NUMINAMATH_CALUDE_cone_height_ratio_l2853_285374

/-- Proves the ratio of new height to original height for a cone with reduced height -/
theorem cone_height_ratio (base_circumference : ℝ) (original_height : ℝ) (new_volume : ℝ) :
  base_circumference = 20 * Real.pi →
  original_height = 40 →
  new_volume = 400 * Real.pi →
  ∃ (new_height : ℝ),
    (1 / 3) * Real.pi * ((base_circumference / (2 * Real.pi)) ^ 2) * new_height = new_volume ∧
    new_height / original_height = 3 / 10 := by
  sorry


end NUMINAMATH_CALUDE_cone_height_ratio_l2853_285374


namespace NUMINAMATH_CALUDE_binomial_expansion_properties_l2853_285375

theorem binomial_expansion_properties :
  ∀ (a₀ a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ a₉ a₁₀ : ℝ),
  (∀ x : ℝ, (2 * x - Real.sqrt 3) ^ 10 = a₀ + a₁ * x + a₂ * x^2 + a₃ * x^3 + a₄ * x^4 + a₅ * x^5 + 
                                         a₆ * x^6 + a₇ * x^7 + a₈ * x^8 + a₉ * x^9 + a₁₀ * x^10) →
  (a₀ = 243 ∧ 
   (a₀ + a₁ + a₂ + a₃ + a₄ + a₅ + a₆ + a₇ + a₈ + a₉ + a₁₀) * 
   (a₀ - a₁ + a₂ - a₃ + a₄ - a₅ + a₆ - a₇ + a₈ - a₉ + a₁₀) = 1) := by
  sorry

end NUMINAMATH_CALUDE_binomial_expansion_properties_l2853_285375


namespace NUMINAMATH_CALUDE_purely_imaginary_complex_number_l2853_285351

/-- If the complex number lg(m^2-2m-2) + (m^2+3m+2)i is purely imaginary and m is real, then m = 3 -/
theorem purely_imaginary_complex_number (m : ℝ) : 
  (Complex.log (m^2 - 2*m - 2) + Complex.I * (m^2 + 3*m + 2)).im ≠ 0 ∧ 
  (Complex.log (m^2 - 2*m - 2) + Complex.I * (m^2 + 3*m + 2)).re = 0 → 
  m = 3 := by sorry

end NUMINAMATH_CALUDE_purely_imaginary_complex_number_l2853_285351


namespace NUMINAMATH_CALUDE_trigonometric_system_solution_l2853_285354

theorem trigonometric_system_solution (x y : ℝ) (k : ℤ) :
  (Real.cos x)^2 + (Real.cos y)^2 = 0.25 →
  x + y = 5 * Real.pi / 6 →
  ((x = Real.pi / 2 * (2 * ↑k + 1) ∧ y = Real.pi / 3 * (1 - 3 * ↑k)) ∨
   (x = Real.pi / 3 * (3 * ↑k + 1) ∧ y = Real.pi / 2 * (1 - 2 * ↑k))) :=
by sorry

end NUMINAMATH_CALUDE_trigonometric_system_solution_l2853_285354


namespace NUMINAMATH_CALUDE_marbles_selection_with_red_l2853_285318

def total_marbles : ℕ := 10
def marbles_to_choose : ℕ := 5

theorem marbles_selection_with_red (total : ℕ) (choose : ℕ) 
  (h1 : total = total_marbles) 
  (h2 : choose = marbles_to_choose) 
  (h3 : total > 0) 
  (h4 : choose > 0) 
  (h5 : total ≥ choose) :
  Nat.choose total choose - Nat.choose (total - 1) choose = 126 := by
  sorry

end NUMINAMATH_CALUDE_marbles_selection_with_red_l2853_285318


namespace NUMINAMATH_CALUDE_triangle_rectangle_perimeter_l2853_285372

theorem triangle_rectangle_perimeter (d : ℕ) : 
  ∀ (t w : ℝ),
  t > 0 ∧ w > 0 →  -- positive sides
  3 * t - (6 * w) = 2016 →  -- perimeter difference
  t = 2 * w + d →  -- side length difference
  d = 672 ∧ ∀ (x : ℕ), x ≠ 672 → x ≠ d :=
by sorry

end NUMINAMATH_CALUDE_triangle_rectangle_perimeter_l2853_285372


namespace NUMINAMATH_CALUDE_expression_result_l2853_285394

theorem expression_result : (3.242 * 16) / 100 = 0.51872 := by
  sorry

end NUMINAMATH_CALUDE_expression_result_l2853_285394


namespace NUMINAMATH_CALUDE_accident_rate_calculation_l2853_285324

theorem accident_rate_calculation (total_vehicles : ℕ) (accident_vehicles : ℕ) 
  (rate_vehicles : ℕ) (rate_accidents : ℕ) :
  total_vehicles = 3000000000 →
  accident_vehicles = 2880 →
  rate_accidents = 96 →
  (rate_accidents : ℚ) / rate_vehicles = (accident_vehicles : ℚ) / total_vehicles →
  rate_vehicles = 100000000 := by
sorry

end NUMINAMATH_CALUDE_accident_rate_calculation_l2853_285324


namespace NUMINAMATH_CALUDE_simultaneous_equations_solution_l2853_285364

theorem simultaneous_equations_solution (m : ℝ) : 
  ∃ (x y : ℝ), y = 3 * m * x + 5 ∧ y = (3 * m - 2) * x + 7 :=
by sorry

end NUMINAMATH_CALUDE_simultaneous_equations_solution_l2853_285364


namespace NUMINAMATH_CALUDE_intersection_line_ellipse_l2853_285334

/-- Prove that if a line y = kx intersects the ellipse x^2/4 + y^2/3 = 1 at points A and B, 
    and the perpendiculars from A and B to the x-axis have their feet at ±1 
    (which are the foci of the ellipse), then k = ± 3/2. -/
theorem intersection_line_ellipse (k : ℝ) : 
  (∀ x y : ℝ, y = k * x → x^2 / 4 + y^2 / 3 = 1 → 
    (x = 1 ∨ x = -1) → k = 3/2 ∨ k = -3/2) := by
  sorry


end NUMINAMATH_CALUDE_intersection_line_ellipse_l2853_285334


namespace NUMINAMATH_CALUDE_travelers_checks_average_l2853_285301

theorem travelers_checks_average (x y : ℕ) : 
  x + y = 30 →
  50 * x + 100 * y = 1800 →
  let remaining_50 := x - 6
  let remaining_100 := y
  let total_remaining := remaining_50 + remaining_100
  let total_value := 50 * remaining_50 + 100 * remaining_100
  (total_value : ℚ) / total_remaining = 125/2 := by sorry

end NUMINAMATH_CALUDE_travelers_checks_average_l2853_285301


namespace NUMINAMATH_CALUDE_expression_evaluation_l2853_285378

theorem expression_evaluation :
  let a : ℝ := Real.sqrt 2 - 1
  (1 - 1 / (a + 1)) * ((a^2 + 2*a + 1) / a) = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2853_285378


namespace NUMINAMATH_CALUDE_negate_all_guitarists_proficient_l2853_285385

-- Define the universe of discourse
variable (U : Type)

-- Define the predicates
variable (Musician : U → Prop)
variable (Guitarist : U → Prop)
variable (ProficientViolinist : U → Prop)

-- Theorem statement
theorem negate_all_guitarists_proficient :
  (∃ x, Guitarist x ∧ ¬ProficientViolinist x) ↔ 
  ¬(∀ x, Guitarist x → ProficientViolinist x) :=
by sorry

end NUMINAMATH_CALUDE_negate_all_guitarists_proficient_l2853_285385


namespace NUMINAMATH_CALUDE_circle_op_eq_power_l2853_285363

noncomputable def circle_op (a : ℚ) (n : ℕ) : ℚ :=
  if n = 0 then 1 else if n = 1 then a else a / (circle_op a (n - 1))

theorem circle_op_eq_power (a : ℚ) (n : ℕ) (h : a ≠ 0) :
  circle_op a n = (1 / a) ^ (n - 2) :=
sorry

end NUMINAMATH_CALUDE_circle_op_eq_power_l2853_285363


namespace NUMINAMATH_CALUDE_octopus_gloves_bracelets_arrangements_l2853_285332

/-- The number of arms an octopus has -/
def num_arms : ℕ := 8

/-- The total number of items (gloves and bracelets) -/
def total_items : ℕ := 2 * num_arms

/-- The number of valid arrangements for putting on gloves and bracelets -/
def valid_arrangements : ℕ := Nat.factorial total_items / (2^num_arms)

/-- Theorem stating the correct number of valid arrangements -/
theorem octopus_gloves_bracelets_arrangements :
  valid_arrangements = Nat.factorial total_items / (2^num_arms) :=
by sorry

end NUMINAMATH_CALUDE_octopus_gloves_bracelets_arrangements_l2853_285332


namespace NUMINAMATH_CALUDE_cameron_chase_speed_ratio_l2853_285366

/-- Proves that the ratio of Cameron's speed to Chase's speed is 2:1 given the conditions -/
theorem cameron_chase_speed_ratio 
  (cameron_speed chase_speed danielle_speed : ℝ)
  (danielle_time chase_time : ℝ)
  (h1 : danielle_speed = 3 * cameron_speed)
  (h2 : danielle_time = 30)
  (h3 : chase_time = 180)
  (h4 : danielle_speed * danielle_time = chase_speed * chase_time) :
  cameron_speed / chase_speed = 2 := by
  sorry

end NUMINAMATH_CALUDE_cameron_chase_speed_ratio_l2853_285366


namespace NUMINAMATH_CALUDE_expression_evaluation_l2853_285392

theorem expression_evaluation (a : ℝ) (h : a = 3) : 
  (3 * a⁻¹ + (2 * a⁻¹) / 3) / (2 * a) = 11 / 54 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2853_285392


namespace NUMINAMATH_CALUDE_train_length_l2853_285358

/-- The length of a train given its speed and time to cross a point -/
theorem train_length (speed : ℝ) (time : ℝ) (h1 : speed = 56.8) (h2 : time = 18) :
  speed * time = 1022.4 := by
  sorry

end NUMINAMATH_CALUDE_train_length_l2853_285358


namespace NUMINAMATH_CALUDE_quartic_roots_l2853_285359

theorem quartic_roots (x : ℝ) :
  (7 * x^4 - 50 * x^3 + 94 * x^2 - 50 * x + 7 = 0) ↔
  (x + 1/x = (50 + Real.sqrt 260)/14 ∨ x + 1/x = (50 - Real.sqrt 260)/14) :=
by sorry

end NUMINAMATH_CALUDE_quartic_roots_l2853_285359


namespace NUMINAMATH_CALUDE_intersections_divisible_by_three_l2853_285360

/-- The number of intersections between segments connecting points on parallel lines -/
def num_intersections (n : ℕ) : ℕ :=
  n * (n - 1) * (n + 1) * n / 4

/-- Theorem stating that the number of intersections is divisible by 3 -/
theorem intersections_divisible_by_three (n : ℕ) :
  ∃ k : ℕ, num_intersections n = 3 * k :=
sorry

end NUMINAMATH_CALUDE_intersections_divisible_by_three_l2853_285360


namespace NUMINAMATH_CALUDE_smallest_prime_perimeter_scalene_triangle_l2853_285382

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d ∣ n → d = 1 ∨ d = n

def is_scalene_triangle (a b c : ℕ) : Prop :=
  a ≠ b ∧ b ≠ c ∧ a ≠ c

def triangle_inequality (a b c : ℕ) : Prop :=
  a + b > c ∧ b + c > a ∧ a + c > b

theorem smallest_prime_perimeter_scalene_triangle :
  ∀ a b c : ℕ,
    is_scalene_triangle a b c →
    is_prime a ∧ is_prime b ∧ is_prime c →
    is_prime (a + b + c) →
    a ≥ 5 ∧ b ≥ 5 ∧ c ≥ 5 →
    triangle_inequality a b c →
    (a + b + c ≥ 23) ∧ (∃ x y z : ℕ, x + y + z = 23 ∧ 
      is_scalene_triangle x y z ∧
      is_prime x ∧ is_prime y ∧ is_prime z ∧
      is_prime (x + y + z) ∧
      x ≥ 5 ∧ y ≥ 5 ∧ z ≥ 5 ∧
      triangle_inequality x y z) :=
by sorry

end NUMINAMATH_CALUDE_smallest_prime_perimeter_scalene_triangle_l2853_285382


namespace NUMINAMATH_CALUDE_cos_five_pi_sixths_l2853_285305

theorem cos_five_pi_sixths : Real.cos (5 * π / 6) = -Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_five_pi_sixths_l2853_285305


namespace NUMINAMATH_CALUDE_julio_salary_julio_salary_is_500_l2853_285353

/-- Calculates Julio's salary for 3 weeks based on given conditions --/
theorem julio_salary (commission_per_customer : ℕ) (first_week_customers : ℕ) 
  (bonus : ℕ) (total_earnings : ℕ) : ℕ :=
  let second_week_customers := 2 * first_week_customers
  let third_week_customers := 3 * first_week_customers
  let total_customers := first_week_customers + second_week_customers + third_week_customers
  let total_commission := total_customers * commission_per_customer
  let salary := total_earnings - (total_commission + bonus)
  salary

/-- Proves that Julio's salary for 3 weeks is $500 --/
theorem julio_salary_is_500 : 
  julio_salary 1 35 50 760 = 500 := by
  sorry

end NUMINAMATH_CALUDE_julio_salary_julio_salary_is_500_l2853_285353


namespace NUMINAMATH_CALUDE_smallest_seating_arrangement_three_satisfies_seating_arrangement_smallest_M_is_three_l2853_285317

theorem smallest_seating_arrangement (M : ℕ+) : (∃ (x y : ℕ+), 8 * M = 12 * x ∧ 12 * M = 8 * y ∧ x = y) → M ≥ 3 :=
by sorry

theorem three_satisfies_seating_arrangement : ∃ (x y : ℕ+), 8 * 3 = 12 * x ∧ 12 * 3 = 8 * y ∧ x = y :=
by sorry

theorem smallest_M_is_three : (∀ M : ℕ+, M < 3 → ¬(∃ (x y : ℕ+), 8 * M = 12 * x ∧ 12 * M = 8 * y ∧ x = y)) ∧
                              (∃ (x y : ℕ+), 8 * 3 = 12 * x ∧ 12 * 3 = 8 * y ∧ x = y) :=
by sorry

end NUMINAMATH_CALUDE_smallest_seating_arrangement_three_satisfies_seating_arrangement_smallest_M_is_three_l2853_285317


namespace NUMINAMATH_CALUDE_cube_volume_surface_area_l2853_285311

theorem cube_volume_surface_area (V : ℝ) : 
  (∃ (x : ℝ), V = x^3 ∧ 2*V = 6*x^2) → V = 27 := by
  sorry

end NUMINAMATH_CALUDE_cube_volume_surface_area_l2853_285311


namespace NUMINAMATH_CALUDE_students_registered_l2853_285391

theorem students_registered (students_yesterday : ℕ) (students_today : ℕ) : ℕ :=
  let students_registered := 156
  let students_absent := 30
  have h1 : students_today + students_absent = students_registered := by sorry
  have h2 : students_today = (2 * students_yesterday * 9) / 10 := by sorry
  students_registered

#check students_registered

end NUMINAMATH_CALUDE_students_registered_l2853_285391


namespace NUMINAMATH_CALUDE_diamond_ratio_l2853_285352

def diamond (n m : ℝ) : ℝ := n^4 * m^3

theorem diamond_ratio : (diamond 3 2) / (diamond 2 3) = 3/2 := by
  sorry

end NUMINAMATH_CALUDE_diamond_ratio_l2853_285352


namespace NUMINAMATH_CALUDE_power_product_equals_l2853_285312

theorem power_product_equals : (3 : ℕ)^4 * (6 : ℕ)^4 = 104976 := by
  sorry

end NUMINAMATH_CALUDE_power_product_equals_l2853_285312


namespace NUMINAMATH_CALUDE_sum_of_possible_x_minus_y_values_l2853_285304

theorem sum_of_possible_x_minus_y_values (x y : ℝ) 
  (eq1 : x^2 - x*y + x = 2018)
  (eq2 : y^2 - x*y - y = 52) : 
  ∃ (z₁ z₂ : ℝ), (z₁ = x - y ∨ z₂ = x - y) ∧ z₁ + z₂ = -1 := by
sorry

end NUMINAMATH_CALUDE_sum_of_possible_x_minus_y_values_l2853_285304


namespace NUMINAMATH_CALUDE_playground_boys_count_l2853_285347

theorem playground_boys_count (total : ℕ) (girls : ℕ) (boys : ℕ) : 
  total = 63 → girls = 28 → boys = total - girls → boys = 35 := by
  sorry

end NUMINAMATH_CALUDE_playground_boys_count_l2853_285347


namespace NUMINAMATH_CALUDE_cycle_loss_percentage_l2853_285355

/-- Calculate the percentage of loss given the cost price and selling price -/
def percentageLoss (costPrice sellingPrice : ℚ) : ℚ :=
  (costPrice - sellingPrice) / costPrice * 100

theorem cycle_loss_percentage :
  let costPrice : ℚ := 1400
  let sellingPrice : ℚ := 1330
  percentageLoss costPrice sellingPrice = 5 := by
  sorry

end NUMINAMATH_CALUDE_cycle_loss_percentage_l2853_285355


namespace NUMINAMATH_CALUDE_possible_values_of_a_l2853_285350

/-- Given sets A and B, where A ⊆ B, prove that a can only be 0, 1, or 1/2 -/
theorem possible_values_of_a (a : ℝ) :
  let A := {x : ℝ | a * x - 1 = 0}
  let B := {x : ℝ | x^2 - 3*x + 2 = 0}
  A ⊆ B → (a = 0 ∨ a = 1 ∨ a = 1/2) := by
  sorry


end NUMINAMATH_CALUDE_possible_values_of_a_l2853_285350


namespace NUMINAMATH_CALUDE_ellipse_focus_k_value_l2853_285399

/-- Given an ellipse 3kx^2 + y^2 = 1 with focus F(2,0), prove that k = 1/15 -/
theorem ellipse_focus_k_value (k : ℝ) : 
  (∃ (x y : ℝ), 3 * k * x^2 + y^2 = 1) →  -- Ellipse equation
  (2 : ℝ)^2 = (1 / (3 * k)) - 1 →  -- Focus condition (c^2 = a^2 - b^2)
  k = 1 / 15 := by
sorry

end NUMINAMATH_CALUDE_ellipse_focus_k_value_l2853_285399


namespace NUMINAMATH_CALUDE_regular_tire_usage_l2853_285309

theorem regular_tire_usage
  (total_miles : ℕ)
  (spare_miles : ℕ)
  (regular_tires : ℕ)
  (h1 : total_miles = 50000)
  (h2 : spare_miles = 2000)
  (h3 : regular_tires = 4) :
  (total_miles - spare_miles) / regular_tires = 12000 :=
by sorry

end NUMINAMATH_CALUDE_regular_tire_usage_l2853_285309


namespace NUMINAMATH_CALUDE_range_of_f_l2853_285320

/-- The function f(x) = x^2 - 1 --/
def f (x : ℝ) : ℝ := x^2 - 1

/-- The range of f is [-1, +∞) --/
theorem range_of_f :
  Set.range f = Set.Ici (-1) := by sorry

end NUMINAMATH_CALUDE_range_of_f_l2853_285320


namespace NUMINAMATH_CALUDE_clarence_spent_12_96_l2853_285328

/-- The cost of Clarence's amusement park visit -/
def clarence_total_cost (cost_per_ride : ℚ) (water_slide_rides : ℕ) (roller_coaster_rides : ℕ) : ℚ :=
  cost_per_ride * (water_slide_rides + roller_coaster_rides)

/-- Theorem stating that Clarence's total cost at the amusement park was $12.96 -/
theorem clarence_spent_12_96 :
  clarence_total_cost 2.16 3 3 = 12.96 := by
  sorry

end NUMINAMATH_CALUDE_clarence_spent_12_96_l2853_285328


namespace NUMINAMATH_CALUDE_count_eight_digit_integers_l2853_285376

/-- The number of different 8-digit positive integers where the first digit is not 0
    and the last digit is neither 0 nor 1 -/
def eight_digit_integers : ℕ :=
  9 * (10 ^ 6) * 8

theorem count_eight_digit_integers :
  eight_digit_integers = 72000000 := by
  sorry

end NUMINAMATH_CALUDE_count_eight_digit_integers_l2853_285376
