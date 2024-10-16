import Mathlib

namespace NUMINAMATH_CALUDE_fence_cost_l3702_370293

/-- The cost of building a fence around a square plot -/
theorem fence_cost (area : ℝ) (price_per_foot : ℝ) (h1 : area = 289) (h2 : price_per_foot = 56) :
  4 * Real.sqrt area * price_per_foot = 3808 := by
  sorry

#check fence_cost

end NUMINAMATH_CALUDE_fence_cost_l3702_370293


namespace NUMINAMATH_CALUDE_colored_balls_probabilities_l3702_370255

/-- Represents a bag of colored balls -/
structure ColoredBalls where
  total : ℕ
  red : ℕ
  yellow : ℕ

/-- Calculate the probability of drawing a ball of a specific color -/
def probability (bag : ColoredBalls) (color : ℕ) : ℚ :=
  color / bag.total

/-- Calculate the number of red balls to add to achieve a target probability -/
def addRedBalls (bag : ColoredBalls) (targetProb : ℚ) : ℕ :=
  let x := (targetProb * bag.total - bag.red) / (1 - targetProb)
  x.ceil.toNat

theorem colored_balls_probabilities (bag : ColoredBalls) :
  bag.total = 10 ∧ bag.red = 4 ∧ bag.yellow = 6 →
  (probability bag bag.yellow = 3/5) ∧
  (addRedBalls bag (2/3) = 8) := by
  sorry

end NUMINAMATH_CALUDE_colored_balls_probabilities_l3702_370255


namespace NUMINAMATH_CALUDE_tangent_lines_to_parabola_l3702_370261

/-- The curve function f(x) = x^2 -/
def f (x : ℝ) : ℝ := x^2

/-- Point B -/
def B : ℝ × ℝ := (3, 5)

/-- Tangent line equation type -/
structure TangentLine where
  a : ℝ
  b : ℝ
  c : ℝ
  equation : ℝ → ℝ → Prop := fun x y => a * x + b * y + c = 0

/-- Theorem: The equations of the lines that pass through B and are tangent to f are 2x - y - 1 = 0 and 10x - y - 25 = 0 -/
theorem tangent_lines_to_parabola :
  ∃ (l₁ l₂ : TangentLine),
    (l₁.equation 3 5 ∧ l₂.equation 3 5) ∧
    (∀ x y, y = f x → (l₁.equation x y ∨ l₂.equation x y) → 
      ∃ ε > 0, ∀ h ∈ Set.Ioo (x - ε) (x + ε), h ≠ x → f h > (l₁.a * h + l₁.c) ∧ f h > (l₂.a * h + l₂.c)) ∧
    l₁.equation = fun x y => 2 * x - y - 1 = 0 ∧
    l₂.equation = fun x y => 10 * x - y - 25 = 0 :=
by sorry

end NUMINAMATH_CALUDE_tangent_lines_to_parabola_l3702_370261


namespace NUMINAMATH_CALUDE_circumcircle_area_of_triangle_l3702_370251

/-- Given a triangle ABC with side lengths a, b, c opposite to angles A, B, C respectively,
    prove that the area of its circumcircle is π/2 under certain conditions. -/
theorem circumcircle_area_of_triangle (a b c : Real) (S : Real) :
  a = 1 →
  4 * S = b^2 + c^2 - 1 →
  (∃ A B C : Real, 
    0 < A ∧ A < π ∧
    0 < B ∧ B < π ∧
    0 < C ∧ C < π ∧
    A + B + C = π ∧
    S = (1/2) * b * c * Real.sin A) →
  (∃ R : Real, R > 0 ∧ π * R^2 = π/2) :=
by sorry

end NUMINAMATH_CALUDE_circumcircle_area_of_triangle_l3702_370251


namespace NUMINAMATH_CALUDE_difference_of_squares_2x_3_l3702_370266

theorem difference_of_squares_2x_3 (x : ℝ) : (2*x + 3) * (2*x - 3) = 4*x^2 - 9 := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_2x_3_l3702_370266


namespace NUMINAMATH_CALUDE_min_odd_in_A_P_l3702_370204

/-- A polynomial of degree 8 -/
def Polynomial8 : Type := ℝ → ℝ

/-- The set A_P for a polynomial P -/
def A_P (P : Polynomial8) (c : ℝ) : Set ℝ := {x | P x = c}

/-- Theorem: If 8 is in A_P, then A_P contains at least one odd number -/
theorem min_odd_in_A_P (P : Polynomial8) (c : ℝ) (h : 8 ∈ A_P P c) :
  ∃ (x : ℝ), x ∈ A_P P c ∧ ∃ (n : ℤ), x = 2 * n + 1 :=
sorry

end NUMINAMATH_CALUDE_min_odd_in_A_P_l3702_370204


namespace NUMINAMATH_CALUDE_weight_of_oranges_l3702_370249

/-- Proves that the weight of oranges is 1 kilogram, given the total weight of fruits
    and the weights of apples, grapes, and strawberries. -/
theorem weight_of_oranges (total_weight apple_weight grape_weight strawberry_weight : ℕ)
  (h_total : total_weight = 10)
  (h_apple : apple_weight = 3)
  (h_grape : grape_weight = 3)
  (h_strawberry : strawberry_weight = 3) :
  total_weight - (apple_weight + grape_weight + strawberry_weight) = 1 := by
  sorry

#check weight_of_oranges

end NUMINAMATH_CALUDE_weight_of_oranges_l3702_370249


namespace NUMINAMATH_CALUDE_certain_number_calculation_l3702_370297

theorem certain_number_calculation : ∃ (n : ℕ), 9823 + 3377 = n := by
  sorry

end NUMINAMATH_CALUDE_certain_number_calculation_l3702_370297


namespace NUMINAMATH_CALUDE_line_angle_slope_relation_l3702_370211

/-- Given two lines L₁ and L₂ in the xy-plane, prove that mn = 1/3 under specific conditions. -/
theorem line_angle_slope_relation (m n : ℝ) : 
  -- L₁ has equation y = 3mx
  -- L₂ has equation y = nx
  -- L₁ makes three times as large of an angle with the horizontal as L₂
  -- L₁ has 3 times the slope of L₂
  (∃ (θ₁ θ₂ : ℝ), θ₁ = 3 * θ₂ ∧ Real.tan θ₁ = 3 * m ∧ Real.tan θ₂ = n) →
  -- L₁ has 3 times the slope of L₂
  3 * m = n →
  -- L₁ is not vertical
  m ≠ 0 →
  -- Conclusion: mn = 1/3
  m * n = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_line_angle_slope_relation_l3702_370211


namespace NUMINAMATH_CALUDE_sets_A_B_characterization_l3702_370206

theorem sets_A_B_characterization (A B : Set ℤ) :
  (A ∪ B = Set.univ) ∧
  (∀ x, x ∈ A → x - 1 ∈ B) ∧
  (∀ x y, x ∈ B ∧ y ∈ B → x + y ∈ A) →
  ((A = {x | ∃ k, x = 2 * k} ∧ B = {x | ∃ k, x = 2 * k + 1}) ∨
   (A = Set.univ ∧ B = Set.univ)) :=
by sorry

end NUMINAMATH_CALUDE_sets_A_B_characterization_l3702_370206


namespace NUMINAMATH_CALUDE_triangle_perimeter_l3702_370271

/-- Given a triangle with two sides of lengths 4 and 6, and the third side length
    being a root of (x-6)(x-10)=0, prove that the perimeter of the triangle is 16. -/
theorem triangle_perimeter : ∀ x : ℝ, 
  (x - 6) * (x - 10) = 0 → 
  (4 < x ∧ x < 10) →  -- Triangle inequality
  4 + 6 + x = 16 := by
sorry

end NUMINAMATH_CALUDE_triangle_perimeter_l3702_370271


namespace NUMINAMATH_CALUDE_system_sampling_theorem_l3702_370207

/-- Represents a system sampling method -/
structure SystemSampling where
  total_students : ℕ
  sample_size : ℕ
  common_difference : ℕ

/-- Checks if a list of numbers forms a valid system sample -/
def is_valid_sample (s : SystemSampling) (sample : List ℕ) : Prop :=
  sample.length = s.sample_size ∧
  ∀ i j, i < j → j < s.sample_size →
    sample[j]! - sample[i]! = s.common_difference * (j - i)

theorem system_sampling_theorem (s : SystemSampling)
  (h_total : s.total_students = 160)
  (h_size : s.sample_size = 5)
  (h_diff : s.common_difference = 32)
  (h_known : ∃ (sample : List ℕ), is_valid_sample s sample ∧ 
    40 ∈ sample ∧ 72 ∈ sample ∧ 136 ∈ sample) :
  ∃ (full_sample : List ℕ), is_valid_sample s full_sample ∧
    40 ∈ full_sample ∧ 72 ∈ full_sample ∧ 136 ∈ full_sample ∧
    8 ∈ full_sample ∧ 104 ∈ full_sample :=
sorry

end NUMINAMATH_CALUDE_system_sampling_theorem_l3702_370207


namespace NUMINAMATH_CALUDE_expression_lower_bound_l3702_370260

theorem expression_lower_bound (a b c : ℝ) (h1 : a > b) (h2 : b > c) (h3 : b ≠ 0) :
  ((a + b)^2 + (b - c)^2 + (a - c)^2) / b^2 ≥ 2.5 := by
  sorry

end NUMINAMATH_CALUDE_expression_lower_bound_l3702_370260


namespace NUMINAMATH_CALUDE_perfect_square_condition_l3702_370270

theorem perfect_square_condition (x y k : ℝ) :
  (∃ (z : ℝ), x^2 + k*x*y + 49*y^2 = z^2) → (k = 14 ∨ k = -14) := by
  sorry

end NUMINAMATH_CALUDE_perfect_square_condition_l3702_370270


namespace NUMINAMATH_CALUDE_specific_triangle_area_l3702_370236

/-- Represents a triangle with given properties -/
structure Triangle where
  base : ℝ
  side : ℝ
  median : ℝ

/-- Calculates the area of a triangle given its properties -/
def triangleArea (t : Triangle) : ℝ :=
  sorry

/-- Theorem stating that a triangle with base 30, side 14, and median 13 has an area of 168 -/
theorem specific_triangle_area :
  let t : Triangle := { base := 30, side := 14, median := 13 }
  triangleArea t = 168 := by
  sorry

end NUMINAMATH_CALUDE_specific_triangle_area_l3702_370236


namespace NUMINAMATH_CALUDE_cubic_roots_sum_of_cubes_l3702_370291

theorem cubic_roots_sum_of_cubes (a b c : ℂ) : 
  (a^3 - 2*a^2 + 3*a - 4 = 0) → 
  (b^3 - 2*b^2 + 3*b - 4 = 0) → 
  (c^3 - 2*c^2 + 3*c - 4 = 0) → 
  a^3 + b^3 + c^3 = 2 := by
sorry

end NUMINAMATH_CALUDE_cubic_roots_sum_of_cubes_l3702_370291


namespace NUMINAMATH_CALUDE_equation_solutions_l3702_370294

theorem equation_solutions : 
  let f (x : ℝ) := 1/((x-1)*(x-2)) + 1/((x-2)*(x-3)) + 1/((x-3)*(x-4)) + 1/((x-4)*(x-5))
  ∀ x : ℝ, f x = 1/12 ↔ x = 12 ∨ x = -4.5 :=
by sorry

end NUMINAMATH_CALUDE_equation_solutions_l3702_370294


namespace NUMINAMATH_CALUDE_exists_excursion_with_frequent_participants_l3702_370222

/-- Represents an excursion --/
structure Excursion where
  participants : Finset Nat
  deriving Inhabited

/-- The problem statement --/
theorem exists_excursion_with_frequent_participants
  (n : Nat) -- number of excursions
  (excursions : Finset Excursion)
  (h1 : excursions.card = n) -- there are n excursions
  (h2 : ∀ e ∈ excursions, e.participants.card ≥ 4) -- each excursion has at least 4 participants
  (h3 : ∀ e ∈ excursions, e.participants.card ≤ 20) -- each excursion has at most 20 participants
  : ∃ e ∈ excursions, ∀ s ∈ e.participants,
    (excursions.filter (λ ex : Excursion => s ∈ ex.participants)).card ≥ n / 17 :=
sorry

end NUMINAMATH_CALUDE_exists_excursion_with_frequent_participants_l3702_370222


namespace NUMINAMATH_CALUDE_x_value_l3702_370229

theorem x_value : 
  let x := 98 * (1 + 20 / 100)
  x = 117.6 := by sorry

end NUMINAMATH_CALUDE_x_value_l3702_370229


namespace NUMINAMATH_CALUDE_roots_depend_on_k_l3702_370285

theorem roots_depend_on_k : 
  ∀ (k : ℝ), 
  ∃ (δ : ℝ), 
  δ = 1 + 4*k ∧ 
  (δ > 0 → ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ (x₁ - 1)*(x₁ - 2) = k ∧ (x₂ - 1)*(x₂ - 2) = k) ∧
  (δ = 0 → ∃ (x : ℝ), (x - 1)*(x - 2) = k) ∧
  (δ < 0 → ¬∃ (x : ℝ), (x - 1)*(x - 2) = k) :=
by sorry


end NUMINAMATH_CALUDE_roots_depend_on_k_l3702_370285


namespace NUMINAMATH_CALUDE_triangle_side_length_l3702_370282

/-- Proves that in a triangle ABC with given conditions, the length of side c is 20 -/
theorem triangle_side_length (a b c : ℝ) (A B C : ℝ) (S : ℝ) : 
  a = 4 → B = π / 3 → S = 20 * Real.sqrt 3 → c = 20 := by
  sorry

end NUMINAMATH_CALUDE_triangle_side_length_l3702_370282


namespace NUMINAMATH_CALUDE_troy_beef_purchase_l3702_370299

/-- Represents the problem of determining the amount of beef Troy buys -/
theorem troy_beef_purchase 
  (veg_pounds : ℝ) 
  (veg_price : ℝ) 
  (beef_price_multiplier : ℝ) 
  (total_cost : ℝ) 
  (h1 : veg_pounds = 6)
  (h2 : veg_price = 2)
  (h3 : beef_price_multiplier = 3)
  (h4 : total_cost = 36) :
  ∃ (beef_pounds : ℝ), 
    beef_pounds * (veg_price * beef_price_multiplier) + veg_pounds * veg_price = total_cost ∧ 
    beef_pounds = 4 := by
  sorry

end NUMINAMATH_CALUDE_troy_beef_purchase_l3702_370299


namespace NUMINAMATH_CALUDE_max_sequence_length_l3702_370240

/-- Represents a quadratic equation in the sequence -/
structure QuadraticEquation where
  p : ℝ
  q : ℝ
  h : p < q

/-- Constructs the next quadratic equation in the sequence -/
def nextEquation (eq : QuadraticEquation) : QuadraticEquation :=
  { p := eq.q, q := -eq.p - eq.q, h := sorry }

/-- The sequence of quadratic equations -/
def quadraticSequence (initial : QuadraticEquation) : ℕ → QuadraticEquation
  | 0 => initial
  | n + 1 => nextEquation (quadraticSequence initial n)

/-- The main theorem: the maximum length of the sequence is 5 -/
theorem max_sequence_length (initial : QuadraticEquation) :
  ∃ n : ℕ, n ≤ 5 ∧ ∀ m : ℕ, m > n → ¬ (quadraticSequence initial m).p < (quadraticSequence initial m).q :=
sorry

end NUMINAMATH_CALUDE_max_sequence_length_l3702_370240


namespace NUMINAMATH_CALUDE_reaction_compound_is_chloramine_l3702_370200

/-- Represents a chemical compound --/
structure Compound where
  formula : String

/-- Represents a chemical reaction --/
structure Reaction where
  reactant : Compound
  water_amount : ℝ
  hcl_product : ℝ
  nh4oh_product : ℝ

/-- The molecular weight of water in g/mol --/
def water_molecular_weight : ℝ := 18

/-- Checks if a compound is chloramine --/
def is_chloramine (c : Compound) : Prop :=
  c.formula = "NH2Cl"

/-- Theorem stating that the compound in the reaction is chloramine --/
theorem reaction_compound_is_chloramine (r : Reaction) : 
  r.water_amount = water_molecular_weight ∧ 
  r.hcl_product = 1 ∧ 
  r.nh4oh_product = 1 → 
  is_chloramine r.reactant :=
by
  sorry


end NUMINAMATH_CALUDE_reaction_compound_is_chloramine_l3702_370200


namespace NUMINAMATH_CALUDE_small_circle_radius_l3702_370268

/-- Given a large circle with radius 10 meters and seven congruent smaller circles
    arranged so that four of their diameters align with the diameter of the large circle,
    the radius of each smaller circle is 2.5 meters. -/
theorem small_circle_radius (large_radius : ℝ) (small_radius : ℝ) : 
  large_radius = 10 → 
  4 * (2 * small_radius) = 2 * large_radius → 
  small_radius = 2.5 := by
  sorry

end NUMINAMATH_CALUDE_small_circle_radius_l3702_370268


namespace NUMINAMATH_CALUDE_pure_imaginary_modulus_l3702_370278

theorem pure_imaginary_modulus (b : ℝ) : 
  (∃ y : ℝ, (1 + b * Complex.I) * (2 - Complex.I) = y * Complex.I) → 
  Complex.abs (1 + b * Complex.I) = Real.sqrt 5 := by
sorry

end NUMINAMATH_CALUDE_pure_imaginary_modulus_l3702_370278


namespace NUMINAMATH_CALUDE_max_value_of_g_l3702_370228

/-- Definition of the function f --/
def f (n : ℕ+) : ℕ := 70 + n^2

/-- Definition of the function g --/
def g (n : ℕ+) : ℕ := Nat.gcd (f n) (f (n + 1))

/-- Theorem stating the maximum value of g(n) --/
theorem max_value_of_g :
  ∃ (m : ℕ+), ∀ (n : ℕ+), g n ≤ g m ∧ g m = 281 :=
sorry

end NUMINAMATH_CALUDE_max_value_of_g_l3702_370228


namespace NUMINAMATH_CALUDE_certain_number_problem_l3702_370217

theorem certain_number_problem (x : ℤ) : 
  ((7 * (x + 5)) / 5 : ℚ) - 5 = 33 ↔ x = 22 :=
by sorry

end NUMINAMATH_CALUDE_certain_number_problem_l3702_370217


namespace NUMINAMATH_CALUDE_sum_remainder_mod_11_l3702_370219

theorem sum_remainder_mod_11 : (99001 + 99002 + 99003 + 99004 + 99005 + 99006) % 11 = 5 := by
  sorry

end NUMINAMATH_CALUDE_sum_remainder_mod_11_l3702_370219


namespace NUMINAMATH_CALUDE_nathan_tomato_harvest_l3702_370203

/-- Represents the harvest and sales data for Nathan's garden --/
structure GardenData where
  strawberry_plants : ℕ
  tomato_plants : ℕ
  strawberries_per_plant : ℕ
  fruits_per_basket : ℕ
  strawberry_basket_price : ℕ
  tomato_basket_price : ℕ
  total_revenue : ℕ

/-- Calculates the number of tomatoes harvested per plant --/
def tomatoes_per_plant (data : GardenData) : ℕ :=
  let strawberry_baskets := (data.strawberry_plants * data.strawberries_per_plant) / data.fruits_per_basket
  let strawberry_revenue := strawberry_baskets * data.strawberry_basket_price
  let tomato_revenue := data.total_revenue - strawberry_revenue
  let tomato_baskets := tomato_revenue / data.tomato_basket_price
  let total_tomatoes := tomato_baskets * data.fruits_per_basket
  total_tomatoes / data.tomato_plants

/-- Theorem stating that given Nathan's garden data, he harvested 16 tomatoes per plant --/
theorem nathan_tomato_harvest :
  let data : GardenData := {
    strawberry_plants := 5,
    tomato_plants := 7,
    strawberries_per_plant := 14,
    fruits_per_basket := 7,
    strawberry_basket_price := 9,
    tomato_basket_price := 6,
    total_revenue := 186
  }
  tomatoes_per_plant data = 16 := by
  sorry

end NUMINAMATH_CALUDE_nathan_tomato_harvest_l3702_370203


namespace NUMINAMATH_CALUDE_wire_cutting_l3702_370231

theorem wire_cutting (total_length : ℝ) (difference : ℝ) (shorter_piece : ℝ) :
  total_length = 30 →
  difference = 2 →
  total_length = shorter_piece + (shorter_piece + difference) →
  shorter_piece = 14 := by
sorry

end NUMINAMATH_CALUDE_wire_cutting_l3702_370231


namespace NUMINAMATH_CALUDE_intersection_equality_l3702_370213

def A : Set ℝ := {x | x^2 - 2*x - 3 = 0}

def B (a : ℝ) : Set ℝ := {x | a*x - 1 = 0}

theorem intersection_equality (a : ℝ) : A ∩ B a = B a → a = 0 ∨ a = -1 ∨ a = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_intersection_equality_l3702_370213


namespace NUMINAMATH_CALUDE_max_equal_covering_is_three_l3702_370232

/-- Represents a square covering on a cube face -/
structure SquareCovering where
  position : Fin 6 × Fin 6
  folded : Bool

/-- Represents the cube and its covering -/
structure CubeCovering where
  squares : List SquareCovering

/-- Check if a cell is covered by a square -/
def covers (s : SquareCovering) (cell : Fin 6 × Fin 6) : Bool :=
  sorry

/-- Count how many squares cover a given cell -/
def coverCount (cc : CubeCovering) (cell : Fin 6 × Fin 6 × Fin 3) : Nat :=
  sorry

/-- Check if the covering is valid (no overlaps, all 2x2) -/
def isValidCovering (cc : CubeCovering) : Bool :=
  sorry

/-- Check if all cells are covered equally -/
def isEqualCovering (cc : CubeCovering) : Bool :=
  sorry

/-- The main theorem -/
theorem max_equal_covering_is_three :
  ∀ (cc : CubeCovering),
    isValidCovering cc →
    isEqualCovering cc →
    ∃ (n : Nat), (∀ (cell : Fin 6 × Fin 6 × Fin 3), coverCount cc cell = n) ∧ n ≤ 3 :=
  sorry

end NUMINAMATH_CALUDE_max_equal_covering_is_three_l3702_370232


namespace NUMINAMATH_CALUDE_average_speed_uphill_downhill_l3702_370248

/-- Theorem: Average speed of a car traveling uphill and downhill -/
theorem average_speed_uphill_downhill 
  (uphill_speed : ℝ) 
  (downhill_speed : ℝ) 
  (uphill_distance : ℝ) 
  (downhill_distance : ℝ) 
  (h1 : uphill_speed = 30) 
  (h2 : downhill_speed = 40) 
  (h3 : uphill_distance = 100) 
  (h4 : downhill_distance = 50) : 
  (uphill_distance + downhill_distance) / 
  (uphill_distance / uphill_speed + downhill_distance / downhill_speed) = 1800 / 55 := by
  sorry

end NUMINAMATH_CALUDE_average_speed_uphill_downhill_l3702_370248


namespace NUMINAMATH_CALUDE_unique_solution_l3702_370242

theorem unique_solution (a m n : ℕ+) (h : Real.sqrt (a^2 - 4 * Real.sqrt 2) = Real.sqrt m - Real.sqrt n) :
  m = 8 ∧ n = 1 ∧ a = 3 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_l3702_370242


namespace NUMINAMATH_CALUDE_fast_area_scientific_notation_l3702_370250

/-- The area of the reflecting surface of the FAST radio telescope in square meters -/
def fast_area : ℝ := 250000

/-- Scientific notation representation of the FAST area -/
def fast_area_scientific : ℝ := 2.5 * (10 ^ 5)

/-- Theorem stating that the FAST area is equal to its scientific notation representation -/
theorem fast_area_scientific_notation : fast_area = fast_area_scientific := by
  sorry

end NUMINAMATH_CALUDE_fast_area_scientific_notation_l3702_370250


namespace NUMINAMATH_CALUDE_emmas_age_l3702_370264

/-- Given the ages and relationships between Jose, Zack, Inez, and Emma, prove Emma's age --/
theorem emmas_age (jose_age : ℕ) (zack_age : ℕ) (inez_age : ℕ) (emma_age : ℕ)
  (h1 : jose_age = 20)
  (h2 : zack_age = jose_age + 4)
  (h3 : inez_age = zack_age - 12)
  (h4 : emma_age = jose_age + 5) :
  emma_age = 25 := by
  sorry


end NUMINAMATH_CALUDE_emmas_age_l3702_370264


namespace NUMINAMATH_CALUDE_pear_weighs_130_l3702_370272

/-- The weight of an apple in grams -/
def apple_weight : ℝ := sorry

/-- The weight of a pear in grams -/
def pear_weight : ℝ := sorry

/-- The weight of a banana in grams -/
def banana_weight : ℝ := sorry

/-- The first condition: one apple, three pears, and two bananas weigh 920 grams -/
axiom condition1 : apple_weight + 3 * pear_weight + 2 * banana_weight = 920

/-- The second condition: two apples, four bananas, and five pears weigh 1,710 grams -/
axiom condition2 : 2 * apple_weight + 4 * banana_weight + 5 * pear_weight = 1710

/-- Theorem stating that a pear weighs 130 grams -/
theorem pear_weighs_130 : pear_weight = 130 := by sorry

end NUMINAMATH_CALUDE_pear_weighs_130_l3702_370272


namespace NUMINAMATH_CALUDE_product_has_34_digits_l3702_370253

/-- The number of digits in a positive integer -/
def num_digits (n : ℕ) : ℕ := sorry

/-- The product of two large numbers -/
def n : ℕ := 3659893456789325678 * 342973489379256

/-- Theorem stating that the product has 34 digits -/
theorem product_has_34_digits : num_digits n = 34 := by sorry

end NUMINAMATH_CALUDE_product_has_34_digits_l3702_370253


namespace NUMINAMATH_CALUDE_job_completion_time_l3702_370256

theorem job_completion_time (x : ℝ) : 
  x > 0 → 
  (1 / x + 1 / 30 = 1 / 10) → 
  x = 15 := by
sorry

end NUMINAMATH_CALUDE_job_completion_time_l3702_370256


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l3702_370246

theorem sqrt_equation_solution (x : ℝ) :
  Real.sqrt (3 + Real.sqrt (4 * x - 5)) = Real.sqrt 10 → x = 13.5 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l3702_370246


namespace NUMINAMATH_CALUDE_rectangle_area_perimeter_ratio_max_l3702_370210

theorem rectangle_area_perimeter_ratio_max (A P : ℝ) (h1 : A > 0) (h2 : P > 0) : 
  A / P^2 ≤ 1 / 16 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_perimeter_ratio_max_l3702_370210


namespace NUMINAMATH_CALUDE_sam_total_dimes_l3702_370208

def initial_dimes : ℕ := 9
def received_dimes : ℕ := 7

theorem sam_total_dimes : initial_dimes + received_dimes = 16 := by
  sorry

end NUMINAMATH_CALUDE_sam_total_dimes_l3702_370208


namespace NUMINAMATH_CALUDE_neg_f_is_reflection_about_x_axis_l3702_370243

/-- A function representing the original graph -/
def f : ℝ → ℝ := sorry

/-- The negation of function f -/
def neg_f (x : ℝ) : ℝ := -f x

/-- Theorem stating that neg_f is a reflection of f about the x-axis -/
theorem neg_f_is_reflection_about_x_axis :
  ∀ x y : ℝ, f x = y ↔ neg_f x = -y :=
sorry

end NUMINAMATH_CALUDE_neg_f_is_reflection_about_x_axis_l3702_370243


namespace NUMINAMATH_CALUDE_range_of_c_l3702_370244

def A (c : ℝ) := {x : ℝ | |x - 1| < c}
def B := {x : ℝ | |x - 3| > 4}

theorem range_of_c :
  ∀ c : ℝ, (A c ∩ B = ∅) ↔ c ≤ 2 :=
sorry

end NUMINAMATH_CALUDE_range_of_c_l3702_370244


namespace NUMINAMATH_CALUDE_nancy_money_total_l3702_370267

/-- Given that Nancy has 9 5-dollar bills, prove that she has $45 in total. -/
theorem nancy_money_total :
  let num_bills : ℕ := 9
  let bill_value : ℕ := 5
  num_bills * bill_value = 45 := by
sorry

end NUMINAMATH_CALUDE_nancy_money_total_l3702_370267


namespace NUMINAMATH_CALUDE_sheet_area_difference_l3702_370290

/-- The combined area (front and back) of a rectangular sheet of paper -/
def combinedArea (length width : ℝ) : ℝ := 2 * length * width

/-- The difference in combined area between two rectangular sheets of paper -/
def areaDifference (length1 width1 length2 width2 : ℝ) : ℝ :=
  combinedArea length1 width1 - combinedArea length2 width2

theorem sheet_area_difference :
  areaDifference 11 9 4.5 11 = 99 := by
  sorry

end NUMINAMATH_CALUDE_sheet_area_difference_l3702_370290


namespace NUMINAMATH_CALUDE_no_solution_for_specific_a_l3702_370281

/-- The equation 7|x-4a|+|x-a²|+6x-2a=0 has no solution when a ∈ (-∞, -18) ∪ (0, +∞) -/
theorem no_solution_for_specific_a (a : ℝ) : 
  (a < -18 ∨ a > 0) → ¬∃ x : ℝ, 7*|x - 4*a| + |x - a^2| + 6*x - 2*a = 0 := by
  sorry

end NUMINAMATH_CALUDE_no_solution_for_specific_a_l3702_370281


namespace NUMINAMATH_CALUDE_polygon_sides_l3702_370295

theorem polygon_sides (sum_interior_angles : ℝ) :
  sum_interior_angles = 1620 →
  ∃ n : ℕ, n = 11 ∧ sum_interior_angles = 180 * (n - 2) :=
by
  sorry

end NUMINAMATH_CALUDE_polygon_sides_l3702_370295


namespace NUMINAMATH_CALUDE_items_can_fit_in_containers_l3702_370258

/-- Represents an item with a weight -/
structure Item where
  weight : ℝ
  weight_bound : weight ≤ 1/2

/-- Represents a set of items -/
def ItemSet := List Item

/-- Calculate the total weight of a set of items -/
def totalWeight (items : ItemSet) : ℝ :=
  items.foldl (fun acc item => acc + item.weight) 0

/-- Theorem: Given a set of items, each weighing at most 1/2 unit, 
    with a total weight W > 1/3, these items can be placed into 
    ⌈(3W - 1)/2⌉ or fewer containers, each with a capacity of 1 unit. -/
theorem items_can_fit_in_containers (items : ItemSet) 
    (h_total_weight : totalWeight items > 1/3) :
    ∃ (num_containers : ℕ), 
      num_containers ≤ Int.ceil ((3 * totalWeight items - 1) / 2) ∧ 
      (∃ (partition : List (List Item)), 
        partition.length = num_containers ∧
        partition.all (fun container => totalWeight container ≤ 1) ∧
        partition.join = items) := by
  sorry

end NUMINAMATH_CALUDE_items_can_fit_in_containers_l3702_370258


namespace NUMINAMATH_CALUDE_cubic_function_properties_l3702_370224

/-- A cubic function with a maximum at x = 1 -/
def f (a b : ℝ) (x : ℝ) : ℝ := a * x^3 + b * x^2

/-- The derivative of f -/
def f_deriv (a b : ℝ) (x : ℝ) : ℝ := 3 * a * x^2 + 2 * b * x

theorem cubic_function_properties (a b : ℝ) :
  f a b 1 = 3 ∧ f_deriv a b 1 = 0 →
  a = -6 ∧ b = 9 ∧ ∀ x, f a b x ≥ 0 ∧ (∃ x₀, f a b x₀ = 0) := by
  sorry

end NUMINAMATH_CALUDE_cubic_function_properties_l3702_370224


namespace NUMINAMATH_CALUDE_toy_truck_cost_l3702_370237

/-- The amount spent on toy trucks, given the total spent on toys and the costs of toy cars and skateboard. -/
theorem toy_truck_cost (total_toys : ℚ) (toy_cars : ℚ) (skateboard : ℚ) 
  (h1 : total_toys = 25.62)
  (h2 : toy_cars = 14.88)
  (h3 : skateboard = 4.88) :
  total_toys - (toy_cars + skateboard) = 5.86 := by
  sorry

end NUMINAMATH_CALUDE_toy_truck_cost_l3702_370237


namespace NUMINAMATH_CALUDE_distance_moonbase_to_skyhaven_l3702_370252

theorem distance_moonbase_to_skyhaven :
  let moonbase : ℂ := 0
  let skyhaven : ℂ := 900 + 1200 * I
  Complex.abs (skyhaven - moonbase) = 1500 := by
  sorry

end NUMINAMATH_CALUDE_distance_moonbase_to_skyhaven_l3702_370252


namespace NUMINAMATH_CALUDE_largest_non_sum_30_composite_l3702_370202

def is_composite (n : ℕ) : Prop := ∃ k m : ℕ, k > 1 ∧ m > 1 ∧ n = k * m

def is_sum_of_multiple_30_and_composite (n : ℕ) : Prop :=
  ∃ k m : ℕ, k > 0 ∧ is_composite m ∧ n = 30 * k + m

theorem largest_non_sum_30_composite : 
  (∀ n : ℕ, n > 217 → is_sum_of_multiple_30_and_composite n) ∧
  ¬is_sum_of_multiple_30_and_composite 217 :=
sorry

end NUMINAMATH_CALUDE_largest_non_sum_30_composite_l3702_370202


namespace NUMINAMATH_CALUDE_class_size_problem_l3702_370216

theorem class_size_problem (average_age : ℝ) (teacher_age : ℝ) (new_average : ℝ) :
  average_age = 10 →
  teacher_age = 26 →
  new_average = average_age + 1 →
  ∃ n : ℕ, (n : ℝ) * average_age + teacher_age = (n + 1 : ℝ) * new_average ∧ n = 15 :=
by sorry

end NUMINAMATH_CALUDE_class_size_problem_l3702_370216


namespace NUMINAMATH_CALUDE_min_value_x_plus_4y_min_value_is_3_plus_2sqrt2_min_value_achieved_l3702_370262

theorem min_value_x_plus_4y (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : 1/x + 1/(2*y) = 1) : 
  ∀ a b : ℝ, a > 0 → b > 0 → 1/a + 1/(2*b) = 1 → x + 4*y ≤ a + 4*b :=
by sorry

theorem min_value_is_3_plus_2sqrt2 (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : 1/x + 1/(2*y) = 1) : 
  x + 4*y ≥ 3 + 2*Real.sqrt 2 :=
by sorry

theorem min_value_achieved (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : 1/x + 1/(2*y) = 1) : 
  ∃ a b : ℝ, a > 0 ∧ b > 0 ∧ 1/a + 1/(2*b) = 1 ∧ a + 4*b = 3 + 2*Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_min_value_x_plus_4y_min_value_is_3_plus_2sqrt2_min_value_achieved_l3702_370262


namespace NUMINAMATH_CALUDE_smallest_integer_power_l3702_370280

theorem smallest_integer_power (x : ℕ) : (∀ y : ℕ, y < x → 27^y ≤ 3^24) ∧ 27^x > 3^24 ↔ x = 9 := by
  sorry

end NUMINAMATH_CALUDE_smallest_integer_power_l3702_370280


namespace NUMINAMATH_CALUDE_sales_job_base_salary_l3702_370218

/-- The base salary of a sales job, given the following conditions:
  - The original salary was $75,000 per year
  - The new job pays a base salary plus 15% commission
  - Each sale is worth $750
  - 266.67 sales per year are needed to not lose money
-/
theorem sales_job_base_salary :
  ∀ (original_salary : ℝ) (commission_rate : ℝ) (sale_value : ℝ) (sales_needed : ℝ),
    original_salary = 75000 →
    commission_rate = 0.15 →
    sale_value = 750 →
    sales_needed = 266.67 →
    ∃ (base_salary : ℝ),
      base_salary + sales_needed * commission_rate * sale_value = original_salary ∧
      base_salary = 45000 :=
by sorry

end NUMINAMATH_CALUDE_sales_job_base_salary_l3702_370218


namespace NUMINAMATH_CALUDE_train_length_l3702_370226

/-- The length of a train given its speed and time to cross a pole -/
theorem train_length (speed : ℝ) (time : ℝ) : 
  speed = 52 → time = 9 → ∃ length : ℝ, 
  (abs (length - 129.96) < 0.01) ∧ (length = speed * 1000 / 3600 * time) := by
  sorry

#check train_length

end NUMINAMATH_CALUDE_train_length_l3702_370226


namespace NUMINAMATH_CALUDE_wax_left_after_detailing_l3702_370241

/-- The amount of wax needed to detail Kellan's car in ounces -/
def car_wax : ℕ := 3

/-- The amount of wax needed to detail Kellan's SUV in ounces -/
def suv_wax : ℕ := 4

/-- The amount of wax in the bottle Kellan bought in ounces -/
def bottle_wax : ℕ := 11

/-- The amount of wax Kellan spilled in ounces -/
def spilled_wax : ℕ := 2

/-- Theorem stating the amount of wax left after detailing both vehicles -/
theorem wax_left_after_detailing : 
  bottle_wax - spilled_wax - (car_wax + suv_wax) = 2 := by
  sorry

end NUMINAMATH_CALUDE_wax_left_after_detailing_l3702_370241


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l3702_370284

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_sum (a : ℕ → ℝ) :
  arithmetic_sequence a →
  (a 3) ^ 2 - 6 * (a 3) - 1 = 0 →
  (a 15) ^ 2 - 6 * (a 15) - 1 = 0 →
  (a 7) + (a 8) + (a 9) + (a 10) + (a 11) = 15 :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l3702_370284


namespace NUMINAMATH_CALUDE_solve_equation_l3702_370286

theorem solve_equation (X : ℝ) : 
  (X^3).sqrt = 81 * (81^(1/12)) → X = 3^(14/9) := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l3702_370286


namespace NUMINAMATH_CALUDE_least_addition_for_divisibility_l3702_370259

theorem least_addition_for_divisibility :
  ∃ (n : ℕ), n = 8 ∧
  (∀ (m : ℕ), m < n → ¬((821562 + m) % 5 = 0 ∧ (821562 + m) % 13 = 0)) ∧
  (821562 + n) % 5 = 0 ∧ (821562 + n) % 13 = 0 := by
  sorry

end NUMINAMATH_CALUDE_least_addition_for_divisibility_l3702_370259


namespace NUMINAMATH_CALUDE_cos_120_degrees_l3702_370273

theorem cos_120_degrees : Real.cos (120 * π / 180) = -1/2 := by sorry

end NUMINAMATH_CALUDE_cos_120_degrees_l3702_370273


namespace NUMINAMATH_CALUDE_quadratic_translation_l3702_370238

/-- Represents a quadratic function of the form ax^2 + bx + c -/
structure QuadraticFunction where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Applies a horizontal translation to a quadratic function -/
def horizontalTranslation (f : QuadraticFunction) (h : ℝ) : QuadraticFunction :=
  { a := f.a
  , b := 2 * f.a * h + f.b
  , c := f.a * h^2 + f.b * h + f.c }

/-- Applies a vertical translation to a quadratic function -/
def verticalTranslation (f : QuadraticFunction) (v : ℝ) : QuadraticFunction :=
  { a := f.a
  , b := f.b
  , c := f.c + v }

/-- The original quadratic function y = x^2 + 1 -/
def originalFunction : QuadraticFunction :=
  { a := 1, b := 0, c := 1 }

theorem quadratic_translation :
  let f := originalFunction
  let g := verticalTranslation (horizontalTranslation f (-2)) (-3)
  g = { a := 1, b := 4, c := 2 } := by sorry

end NUMINAMATH_CALUDE_quadratic_translation_l3702_370238


namespace NUMINAMATH_CALUDE_parallelogram_condition_inscribed_quadrilateral_condition_l3702_370276

-- Define a quadrilateral
structure Quadrilateral :=
  (A B C D : Point)

-- Define a parallelogram
def is_parallelogram (q : Quadrilateral) : Prop := sorry

-- Define parallel sides
def parallel_sides (q : Quadrilateral) (side1 side2 : Segment) : Prop := sorry

-- Define equal sides
def equal_sides (side1 side2 : Segment) : Prop := sorry

-- Define supplementary angles
def supplementary_angles (a1 a2 : Angle) : Prop := sorry

-- Define inscribed in a circle
def inscribed_in_circle (q : Quadrilateral) : Prop := sorry

-- Theorem 1
theorem parallelogram_condition (q : Quadrilateral) 
  (side1 side2 : Segment) :
  parallel_sides q side1 side2 → 
  equal_sides side1 side2 → 
  is_parallelogram q :=
sorry

-- Theorem 2
theorem inscribed_quadrilateral_condition (q : Quadrilateral) 
  (a1 a2 a3 a4 : Angle) :
  supplementary_angles a1 a3 → 
  supplementary_angles a2 a4 → 
  inscribed_in_circle q :=
sorry

end NUMINAMATH_CALUDE_parallelogram_condition_inscribed_quadrilateral_condition_l3702_370276


namespace NUMINAMATH_CALUDE_calculation_problem_l3702_370220

theorem calculation_problem (n : ℝ) : n = -6.4 ↔ 10 * 1.8 - (n * 1.5 / 0.3) = 50 := by
  sorry

end NUMINAMATH_CALUDE_calculation_problem_l3702_370220


namespace NUMINAMATH_CALUDE_trigonometric_identities_l3702_370247

theorem trigonometric_identities :
  (Real.sin (15 * π / 180) = (Real.sqrt 6 - Real.sqrt 2) / 4) ∧
  (Real.cos (15 * π / 180) = (Real.sqrt 6 + Real.sqrt 2) / 4) ∧
  (Real.sin (18 * π / 180) = (-1 + Real.sqrt 5) / 4) ∧
  (Real.cos (18 * π / 180) = Real.sqrt (10 + 2 * Real.sqrt 5) / 4) := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_identities_l3702_370247


namespace NUMINAMATH_CALUDE_book_pages_and_reading_schedule_l3702_370292

-- Define the total number of pages in the book
variable (P : ℕ)

-- Define the number of pages read on the 4th day
variable (x : ℕ)

-- Theorem statement
theorem book_pages_and_reading_schedule :
  -- Conditions
  (2 / 3 : ℚ) * P = ((2 / 3 : ℚ) * P - (1 / 3 : ℚ) * P) + 90 ∧
  (1 / 3 : ℚ) * P = x + (x - 10) ∧
  x > 10 →
  -- Conclusions
  P = 270 ∧ x = 50 ∧ x - 10 = 40 := by
sorry

end NUMINAMATH_CALUDE_book_pages_and_reading_schedule_l3702_370292


namespace NUMINAMATH_CALUDE_runner_speed_increase_l3702_370205

/-- Represents a runner's speed and time improvement factors -/
structure Runner where
  initialSpeed : ℝ
  speedIncrease1 : ℝ
  speedIncrease2 : ℝ
  timeFactor1 : ℝ

/-- Theorem: If increasing speed by speedIncrease1 makes the runner timeFactor1 times faster,
    then increasing speed by speedIncrease2 will make them speedRatio times faster -/
theorem runner_speed_increase (runner : Runner)
  (h1 : runner.speedIncrease1 = 2)
  (h2 : runner.timeFactor1 = 2.5)
  (h3 : runner.speedIncrease2 = 4)
  (h4 : runner.initialSpeed > 0)
  : (runner.initialSpeed + runner.speedIncrease2) / runner.initialSpeed = 4 := by
  sorry

end NUMINAMATH_CALUDE_runner_speed_increase_l3702_370205


namespace NUMINAMATH_CALUDE_smallest_number_with_properties_l3702_370215

def digit_sum (n : ℕ) : ℕ := sorry

def is_smallest_with_properties (n : ℕ) : Prop :=
  (n % 5 = 0) ∧
  (digit_sum n = 100) ∧
  (∀ m : ℕ, m < n → (m % 5 ≠ 0 ∨ digit_sum m ≠ 100))

theorem smallest_number_with_properties :
  is_smallest_with_properties 599999999995 := by sorry

end NUMINAMATH_CALUDE_smallest_number_with_properties_l3702_370215


namespace NUMINAMATH_CALUDE_deck_size_problem_l3702_370254

theorem deck_size_problem (r b : ℕ) : 
  -- Initial probability of selecting a red card
  (r : ℚ) / (r + b) = 2 / 5 →
  -- Probability after adding 6 black cards
  (r : ℚ) / (r + b + 6) = 1 / 3 →
  -- Total number of cards initially
  r + b = 5 := by
sorry

end NUMINAMATH_CALUDE_deck_size_problem_l3702_370254


namespace NUMINAMATH_CALUDE_missing_number_proof_l3702_370288

theorem missing_number_proof (x : ℝ) : 
  (x + 42 + 78 + 104) / 4 = 62 ∧ 
  (128 + 255 + 511 + 1023 + x) / 5 = 398.2 →
  x = 74 := by
sorry

end NUMINAMATH_CALUDE_missing_number_proof_l3702_370288


namespace NUMINAMATH_CALUDE_cookies_in_box_l3702_370214

/-- The number of ounces in a pound -/
def ounces_per_pound : ℕ := 16

/-- The weight capacity of the box in pounds -/
def box_capacity : ℕ := 40

/-- The weight of each cookie in ounces -/
def cookie_weight : ℕ := 2

/-- Proves that the number of cookies that can fit in the box is 320 -/
theorem cookies_in_box : 
  (box_capacity * ounces_per_pound) / cookie_weight = 320 := by
  sorry

end NUMINAMATH_CALUDE_cookies_in_box_l3702_370214


namespace NUMINAMATH_CALUDE_min_value_expression_lower_bound_achievable_l3702_370279

theorem min_value_expression (x y : ℝ) : 4 * x^2 + 4 * x * Real.sin y - Real.cos y^2 ≥ -1 := by
  sorry

theorem lower_bound_achievable : ∃ x y : ℝ, 4 * x^2 + 4 * x * Real.sin y - Real.cos y^2 = -1 := by
  sorry

end NUMINAMATH_CALUDE_min_value_expression_lower_bound_achievable_l3702_370279


namespace NUMINAMATH_CALUDE_roses_cut_l3702_370245

theorem roses_cut (initial_roses vase_roses garden_roses : ℕ) 
  (h1 : initial_roses = 7)
  (h2 : vase_roses = 20)
  (h3 : garden_roses = 59) :
  vase_roses - initial_roses = 13 := by
  sorry

end NUMINAMATH_CALUDE_roses_cut_l3702_370245


namespace NUMINAMATH_CALUDE_rectangle_length_l3702_370274

/-- Represents a rectangle with length, width, diagonal, and perimeter -/
structure Rectangle where
  length : ℝ
  width : ℝ
  diagonal : ℝ
  perimeter : ℝ

/-- Theorem: A rectangle with diagonal 17 cm and perimeter 46 cm has a length of 15 cm -/
theorem rectangle_length (r : Rectangle) 
  (h_diagonal : r.diagonal = 17)
  (h_perimeter : r.perimeter = 46)
  (h_perimeter_def : r.perimeter = 2 * (r.length + r.width))
  (h_diagonal_def : r.diagonal ^ 2 = r.length ^ 2 + r.width ^ 2) :
  r.length = 15 := by
  sorry


end NUMINAMATH_CALUDE_rectangle_length_l3702_370274


namespace NUMINAMATH_CALUDE_pascal_row8_sum_and_difference_l3702_370212

/-- Pascal's Triangle sum for a given row -/
def pascal_sum (n : ℕ) : ℕ := 2^n

theorem pascal_row8_sum_and_difference :
  (pascal_sum 8 = 256) ∧
  (pascal_sum 8 - pascal_sum 7 = 128) := by
  sorry

end NUMINAMATH_CALUDE_pascal_row8_sum_and_difference_l3702_370212


namespace NUMINAMATH_CALUDE_solution_bijection_l3702_370225

def equation_x (x : Fin 10 → ℕ+) : Prop :=
  (x 0) + 2^3 * (x 1) + 3^3 * (x 2) + 4^3 * (x 3) + 5^3 * (x 4) + 
  6^3 * (x 5) + 7^3 * (x 6) + 8^3 * (x 7) + 9^3 * (x 8) + 10^3 * (x 9) = 3025

def equation_y (y : Fin 10 → ℕ) : Prop :=
  (y 0) + 2^3 * (y 1) + 3^3 * (y 2) + 4^3 * (y 3) + 5^3 * (y 4) + 
  6^3 * (y 5) + 7^3 * (y 6) + 8^3 * (y 7) + 9^3 * (y 8) + 10^3 * (y 9) = 0

theorem solution_bijection :
  ∃ (f : {x : Fin 10 → ℕ+ // equation_x x} → {y : Fin 10 → ℕ // equation_y y}),
    Function.Bijective f ∧
    f ⟨λ _ => 1, sorry⟩ = ⟨λ _ => 0, sorry⟩ :=
sorry

end NUMINAMATH_CALUDE_solution_bijection_l3702_370225


namespace NUMINAMATH_CALUDE_inequality_proof_l3702_370257

theorem inequality_proof (x y z : ℝ) (hx : x ≥ 0) (hy : y ≥ 0) (hz : z ≥ 0) (hsum : x + y + z = 1) :
  x^2 * y^2 + y^2 * z^2 + z^2 * x^2 + x^2 * y^2 * z^2 ≤ 1/16 := by
sorry

end NUMINAMATH_CALUDE_inequality_proof_l3702_370257


namespace NUMINAMATH_CALUDE_arctan_equation_solution_l3702_370233

theorem arctan_equation_solution (y : ℝ) : 
  2 * Real.arctan (1/3) - Real.arctan (1/5) + Real.arctan (1/y) = π/4 → y = 31/9 := by
sorry

end NUMINAMATH_CALUDE_arctan_equation_solution_l3702_370233


namespace NUMINAMATH_CALUDE_james_chore_time_l3702_370201

/-- The time James spends vacuuming, in hours -/
def vacuum_time : ℝ := 3

/-- The factor by which the time spent on other chores exceeds vacuuming time -/
def other_chores_factor : ℝ := 3

/-- The total time James spends on his chores, in hours -/
def total_chore_time : ℝ := vacuum_time + other_chores_factor * vacuum_time

theorem james_chore_time : total_chore_time = 12 := by
  sorry

end NUMINAMATH_CALUDE_james_chore_time_l3702_370201


namespace NUMINAMATH_CALUDE_linear_function_property_l3702_370283

theorem linear_function_property (a : ℝ) :
  (∃ y : ℝ, y = a * 3 + (1 - a) ∧ y = 7) →
  (∃ y : ℝ, y = a * 8 + (1 - a) ∧ y = 22) :=
by sorry

end NUMINAMATH_CALUDE_linear_function_property_l3702_370283


namespace NUMINAMATH_CALUDE_meaningful_square_root_range_l3702_370277

theorem meaningful_square_root_range (x : ℝ) :
  (∃ y : ℝ, y ^ 2 = x + 1) ↔ x ≥ -1 := by
  sorry

end NUMINAMATH_CALUDE_meaningful_square_root_range_l3702_370277


namespace NUMINAMATH_CALUDE_max_value_of_f_on_interval_l3702_370265

-- Define the function f(x)
def f (x : ℝ) : ℝ := x^3 - 3*x^2 + 2

-- Define the interval
def interval : Set ℝ := Set.Icc (-1) 1

-- Theorem statement
theorem max_value_of_f_on_interval :
  ∃ (c : ℝ), c ∈ interval ∧ f c = 2 ∧ ∀ x ∈ interval, f x ≤ f c :=
sorry

end NUMINAMATH_CALUDE_max_value_of_f_on_interval_l3702_370265


namespace NUMINAMATH_CALUDE_probability_one_red_ball_l3702_370298

def total_balls : ℕ := 12
def red_balls : ℕ := 3
def black_balls : ℕ := 4
def white_balls : ℕ := 5
def drawn_balls : ℕ := 2

theorem probability_one_red_ball :
  (red_balls * (black_balls + white_balls)) / (total_balls.choose drawn_balls) = 9 / 22 := by
  sorry

end NUMINAMATH_CALUDE_probability_one_red_ball_l3702_370298


namespace NUMINAMATH_CALUDE_final_expression_l3702_370263

/-- Given a real number b, prove that doubling b, adding 4, subtracting 4b, and dividing by 2 results in -b + 2 -/
theorem final_expression (b : ℝ) : ((2 * b + 4) - 4 * b) / 2 = -b + 2 := by
  sorry

end NUMINAMATH_CALUDE_final_expression_l3702_370263


namespace NUMINAMATH_CALUDE_sqrt_sum_difference_l3702_370239

theorem sqrt_sum_difference : Real.sqrt 18 - Real.sqrt 8 + Real.sqrt 2 = 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_sum_difference_l3702_370239


namespace NUMINAMATH_CALUDE_circle_area_equality_l3702_370275

theorem circle_area_equality (r₁ r₂ r₃ : ℝ) (h₁ : r₁ = 12) (h₂ : r₂ = 25) (h₃ : r₃ = Real.sqrt 481) :
  π * r₃^2 = π * r₂^2 - π * r₁^2 :=
by sorry

end NUMINAMATH_CALUDE_circle_area_equality_l3702_370275


namespace NUMINAMATH_CALUDE_platform_length_l3702_370227

/-- Given a train of length 450 m that crosses a platform in 56 sec and a signal pole in 24 sec,
    the length of the platform is 600 m. -/
theorem platform_length (train_length : ℝ) (platform_time : ℝ) (pole_time : ℝ) 
    (h1 : train_length = 450)
    (h2 : platform_time = 56)
    (h3 : pole_time = 24) : 
  train_length * (platform_time / pole_time - 1) = 600 := by
  sorry

end NUMINAMATH_CALUDE_platform_length_l3702_370227


namespace NUMINAMATH_CALUDE_square_difference_greater_than_polynomial_l3702_370209

theorem square_difference_greater_than_polynomial :
  ∀ x : ℝ, (x - 3)^2 > x^2 - 6*x + 8 := by
sorry

end NUMINAMATH_CALUDE_square_difference_greater_than_polynomial_l3702_370209


namespace NUMINAMATH_CALUDE_polynomial_difference_independent_of_x_l3702_370230

theorem polynomial_difference_independent_of_x (m n : ℝ) :
  (∀ x y : ℝ, ∃ k : ℝ, (x^2 + m*x - 2*y + n) - (n*x^2 - 3*x + 4*y - 7) = k) →
  n - m = 4 := by
sorry

end NUMINAMATH_CALUDE_polynomial_difference_independent_of_x_l3702_370230


namespace NUMINAMATH_CALUDE_max_xy_value_l3702_370269

theorem max_xy_value (x y c : ℝ) (h : x + y = c - 195) : 
  ∃ d : ℝ, d = 4 ∧ ∀ x' y' : ℝ, x' + y' = c - 195 → x' * y' ≤ d :=
sorry

end NUMINAMATH_CALUDE_max_xy_value_l3702_370269


namespace NUMINAMATH_CALUDE_binary_conversion_l3702_370223

-- Define the binary number
def binary_num : List Bool := [true, true, false, false, true, true]

-- Function to convert binary to decimal
def binary_to_decimal (b : List Bool) : ℕ :=
  b.enum.foldl (fun acc (i, bit) => acc + if bit then 2^i else 0) 0

-- Function to convert decimal to base 5
def decimal_to_base5 (n : ℕ) : List ℕ :=
  if n = 0 then [0] else
    let rec aux (m : ℕ) (acc : List ℕ) :=
      if m = 0 then acc
      else aux (m / 5) ((m % 5) :: acc)
    aux n []

-- Theorem statement
theorem binary_conversion :
  binary_to_decimal binary_num = 51 ∧
  decimal_to_base5 (binary_to_decimal binary_num) = [2, 0, 1] := by
  sorry

end NUMINAMATH_CALUDE_binary_conversion_l3702_370223


namespace NUMINAMATH_CALUDE_devin_teaching_years_l3702_370235

/-- Represents the number of years Devin taught each subject -/
structure TeachingYears where
  calculus : ℕ
  algebra : ℕ
  statistics : ℕ
  geometry : ℕ
  discrete_math : ℕ

/-- Calculates the total number of years taught -/
def total_years (years : TeachingYears) : ℕ :=
  years.calculus + years.algebra + years.statistics + years.geometry + years.discrete_math

/-- Theorem stating the total number of years Devin taught -/
theorem devin_teaching_years :
  ∃ (years : TeachingYears),
    years.calculus = 4 ∧
    years.algebra = 2 * years.calculus ∧
    years.statistics = 5 * years.algebra ∧
    years.geometry = 3 * years.statistics ∧
    years.discrete_math = years.geometry / 2 ∧
    total_years years = 232 :=
by sorry

end NUMINAMATH_CALUDE_devin_teaching_years_l3702_370235


namespace NUMINAMATH_CALUDE_initially_calculated_average_weight_l3702_370234

theorem initially_calculated_average_weight
  (n : ℕ)
  (correct_avg : ℝ)
  (misread_weight : ℝ)
  (correct_weight : ℝ)
  (h1 : n = 20)
  (h2 : correct_avg = 58.9)
  (h3 : misread_weight = 56)
  (h4 : correct_weight = 66)
  : ∃ (initial_avg : ℝ), initial_avg = 58.4 :=
by
  sorry

end NUMINAMATH_CALUDE_initially_calculated_average_weight_l3702_370234


namespace NUMINAMATH_CALUDE_complement_of_A_in_U_l3702_370296

-- Define the universal set U
def U : Set ℝ := {x | -3 < x ∧ x < 3}

-- Define set A
def A : Set ℝ := {x | -2 < x ∧ x ≤ 1}

-- State the theorem
theorem complement_of_A_in_U : 
  (U \ A) = {x : ℝ | (-3 < x ∧ x ≤ -2) ∨ (1 < x ∧ x < 3)} := by sorry

end NUMINAMATH_CALUDE_complement_of_A_in_U_l3702_370296


namespace NUMINAMATH_CALUDE_three_samples_in_interval_l3702_370289

/-- Represents a systematic sampling scenario -/
structure SystematicSample where
  population : ℕ
  sample_size : ℕ
  interval_start : ℕ
  interval_end : ℕ

/-- Calculates the sampling interval -/
def sampling_interval (s : SystematicSample) : ℕ :=
  s.population / s.sample_size

/-- Counts the number of sampled elements within the given interval -/
def count_sampled_in_interval (s : SystematicSample) : ℕ :=
  let k := sampling_interval s
  let first_sample := k * ((s.interval_start - 1) / k + 1)
  (s.interval_end - first_sample) / k + 1

/-- Theorem stating that for the given systematic sample, 
    exactly 3 sampled numbers fall within the interval [61, 120] -/
theorem three_samples_in_interval : 
  let s : SystematicSample := {
    population := 840
    sample_size := 42
    interval_start := 61
    interval_end := 120
  }
  count_sampled_in_interval s = 3 := by
  sorry

end NUMINAMATH_CALUDE_three_samples_in_interval_l3702_370289


namespace NUMINAMATH_CALUDE_social_media_ratio_l3702_370287

/-- Represents the daily phone usage in hours -/
def daily_phone_usage : ℝ := 16

/-- Represents the weekly social media usage in hours -/
def weekly_social_media_usage : ℝ := 56

/-- Represents the number of days in a week -/
def days_in_week : ℝ := 7

/-- Theorem: The ratio of daily time spent on social media to total daily time spent on phone is 1:2 -/
theorem social_media_ratio : 
  (weekly_social_media_usage / days_in_week) / daily_phone_usage = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_social_media_ratio_l3702_370287


namespace NUMINAMATH_CALUDE_max_value_of_expression_l3702_370221

theorem max_value_of_expression (x : ℝ) : (2*x^2 + 8*x + 16) / (2*x^2 + 8*x + 6) ≤ 6 := by
  sorry

end NUMINAMATH_CALUDE_max_value_of_expression_l3702_370221
