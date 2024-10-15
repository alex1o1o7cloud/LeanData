import Mathlib

namespace NUMINAMATH_CALUDE_quadratic_properties_l4002_400264

-- Define the quadratic function
def f (a : ℝ) (x : ℝ) : ℝ := x^2 - 4*x + a

-- Theorem statement
theorem quadratic_properties (a : ℝ) :
  (∀ x y, x < 1 ∧ y < 1 ∧ x < y → f a x > f a y) ∧
  (∃ x, f a x = 0 → a ≤ 4) ∧
  (¬(a = 3 → ∀ x, f a x > 0 ↔ 1 < x ∧ x < 3)) ∧
  (∀ b, f a 2013 = b → f a (-2009) = b) :=
sorry

end NUMINAMATH_CALUDE_quadratic_properties_l4002_400264


namespace NUMINAMATH_CALUDE_xyz_value_l4002_400213

theorem xyz_value (a b c x y z : ℂ) 
  (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0)
  (eq1 : a = (b + c) / (x - 3))
  (eq2 : b = (a + c) / (y - 3))
  (eq3 : c = (a + b) / (z - 3))
  (eq4 : x * y + x * z + y * z = 10)
  (eq5 : x + y + z = 6) :
  x * y * z = 15 := by
  sorry

end NUMINAMATH_CALUDE_xyz_value_l4002_400213


namespace NUMINAMATH_CALUDE_polynomial_identity_proof_l4002_400286

theorem polynomial_identity_proof :
  ∀ (a a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ a₉ a₁₀ a₁₁ a₁₂ : ℝ),
  (∀ x : ℝ, (x^2 - x + 1)^6 = a + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5 + 
                              a₆*x^6 + a₇*x^7 + a₈*x^8 + a₉*x^9 + a₁₀*x^10 + a₁₁*x^11 + a₁₂*x^12) →
  (a + a₂ + a₄ + a₆ + a₈ + a₁₀ + a₁₂)^2 - (a₁ + a₃ + a₅ + a₇ + a₉ + a₁₁)^2 = 729 := by
sorry

end NUMINAMATH_CALUDE_polynomial_identity_proof_l4002_400286


namespace NUMINAMATH_CALUDE_inverse_variation_cube_l4002_400297

/-- Given that x and y are positive real numbers, x^3 and y vary inversely,
    and y = 8 when x = 2, prove that x = 1 when y = 64. -/
theorem inverse_variation_cube (x y : ℝ) (hx : x > 0) (hy : y > 0)
  (h_inverse : ∃ k : ℝ, ∀ x y, x^3 * y = k)
  (h_initial : 2^3 * 8 = (Classical.choose h_inverse))
  (h_y : y = 64) : x = 1 := by
  sorry

end NUMINAMATH_CALUDE_inverse_variation_cube_l4002_400297


namespace NUMINAMATH_CALUDE_original_fraction_proof_l4002_400250

theorem original_fraction_proof (N D : ℚ) :
  (N > 0) →
  (D > 0) →
  ((1.15 * N) / (0.92 * D) = 15 / 16) →
  (N / D = 4 / 3) :=
by
  sorry

end NUMINAMATH_CALUDE_original_fraction_proof_l4002_400250


namespace NUMINAMATH_CALUDE_tea_mixture_price_l4002_400208

theorem tea_mixture_price (price1 price2 : ℝ) (ratio : ℝ) (mixture_price : ℝ) : 
  price1 = 64 →
  price2 = 74 →
  ratio = 1 →
  mixture_price = (price1 + price2) / (2 * ratio) →
  mixture_price = 69 := by
sorry

end NUMINAMATH_CALUDE_tea_mixture_price_l4002_400208


namespace NUMINAMATH_CALUDE_unique_solution_for_equation_l4002_400242

theorem unique_solution_for_equation (m n : ℕ+) : 
  (m : ℤ)^(n : ℕ) - (n : ℤ)^(m : ℕ) = 3 ↔ m = 4 ∧ n = 1 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_for_equation_l4002_400242


namespace NUMINAMATH_CALUDE_product_equals_sum_solutions_l4002_400221

theorem product_equals_sum_solutions (x y : ℤ) :
  x * y = x + y ↔ (x = 2 ∧ y = 2) ∨ (x = 0 ∧ y = 0) := by
  sorry

end NUMINAMATH_CALUDE_product_equals_sum_solutions_l4002_400221


namespace NUMINAMATH_CALUDE_unbounded_expression_l4002_400224

theorem unbounded_expression (x y : ℝ) (hx : x ≥ 0) (hy : y ≥ 0) :
  ∀ M : ℝ, ∃ x y : ℝ, x ≥ 0 ∧ y ≥ 0 ∧ (x*y + 1)^2 + (x - y)^2 > M :=
sorry

end NUMINAMATH_CALUDE_unbounded_expression_l4002_400224


namespace NUMINAMATH_CALUDE_complement_of_A_in_U_l4002_400265

universe u

def U : Set ℕ := {1, 2, 3, 4, 5}
def A : Set ℕ := {1, 3}

theorem complement_of_A_in_U :
  (U \ A) = {2, 4, 5} := by sorry

end NUMINAMATH_CALUDE_complement_of_A_in_U_l4002_400265


namespace NUMINAMATH_CALUDE_diplomats_conference_l4002_400202

/-- The number of diplomats who attended the conference -/
def D : ℕ := 120

/-- The number of diplomats who spoke Japanese -/
def J : ℕ := 20

/-- The number of diplomats who did not speak Russian -/
def not_R : ℕ := 32

/-- The percentage of diplomats who spoke neither Japanese nor Russian -/
def neither_percent : ℚ := 20 / 100

/-- The percentage of diplomats who spoke both Japanese and Russian -/
def both_percent : ℚ := 10 / 100

theorem diplomats_conference :
  D = 120 ∧
  J = 20 ∧
  not_R = 32 ∧
  neither_percent = 20 / 100 ∧
  both_percent = 10 / 100 ∧
  (D : ℚ) * neither_percent = (D - (J + (D - not_R) - (D : ℚ) * both_percent) : ℚ) :=
by sorry

end NUMINAMATH_CALUDE_diplomats_conference_l4002_400202


namespace NUMINAMATH_CALUDE_intersection_property_l4002_400283

-- Define the curve
def f (x : ℝ) : ℝ := x^3 - 6*x^2 + 13*x - 8

-- Define the line
def line (k m x : ℝ) : ℝ := k*x + m

-- Theorem statement
theorem intersection_property (k m : ℝ) 
  (hA : ∃ xA, f xA = line k m xA)  -- A exists
  (hB : ∃ xB, f xB = line k m xB)  -- B exists
  (hC : ∃ xC, f xC = line k m xC)  -- C exists
  (h_distinct : ∀ x y, x ≠ y → (f x = line k m x ∧ f y = line k m y) → 
                 ∃ z, f z = line k m z ∧ z ≠ x ∧ z ≠ y)  -- A, B, C are distinct
  (h_midpoint : ∃ xA xB xC, f xA = line k m xA ∧ f xB = line k m xB ∧ f xC = line k m xC ∧
                 xB = (xA + xC) / 2)  -- B is the midpoint of AC
  : 2*k + m = 2 := by
  sorry

end NUMINAMATH_CALUDE_intersection_property_l4002_400283


namespace NUMINAMATH_CALUDE_magic_square_vector_sum_l4002_400214

/-- Represents a magic square of size n × n -/
structure MagicSquare (n : ℕ) where
  grid : Matrix (Fin n) (Fin n) ℕ
  elements : ∀ i j, grid i j ∈ Finset.range (n^2 + 1) \ {0}
  row_sum : ∀ i, (Finset.univ.sum fun j => grid i j) = n * (n^2 + 1) / 2
  col_sum : ∀ j, (Finset.univ.sum fun i => grid i j) = n * (n^2 + 1) / 2
  diag_sum : (Finset.univ.sum fun i => grid i i) = n * (n^2 + 1) / 2

/-- Vector connecting centers of two cells -/
def cellVector (n : ℕ) (i j k l : Fin n) : ℝ × ℝ :=
  (↑k - ↑i, ↑l - ↑j)

/-- The theorem to be proved -/
theorem magic_square_vector_sum (n : ℕ) (ms : MagicSquare n) :
  (Finset.univ.sum fun i =>
    (Finset.univ.sum fun j =>
      (Finset.univ.sum fun k =>
        (Finset.univ.sum fun l =>
          if ms.grid i j > ms.grid k l
          then cellVector n i j k l
          else (0, 0))))) = (0, 0) :=
sorry

end NUMINAMATH_CALUDE_magic_square_vector_sum_l4002_400214


namespace NUMINAMATH_CALUDE_hexagonal_grid_toothpicks_l4002_400255

/-- The number of sides in a hexagon -/
def hexagon_sides : ℕ := 6

/-- The number of toothpicks in each side of the hexagon -/
def toothpicks_per_side : ℕ := 6

/-- The total number of toothpicks used to build the hexagonal grid -/
def total_toothpicks : ℕ := hexagon_sides * toothpicks_per_side

/-- Theorem: The total number of toothpicks used to build the hexagonal grid is 36 -/
theorem hexagonal_grid_toothpicks :
  total_toothpicks = 36 := by
  sorry

end NUMINAMATH_CALUDE_hexagonal_grid_toothpicks_l4002_400255


namespace NUMINAMATH_CALUDE_danys_farm_bushels_l4002_400236

/-- Represents the farm animals and their food consumption -/
structure Farm where
  cows : Nat
  sheep : Nat
  chickens : Nat
  cow_sheep_consumption : Nat
  chicken_consumption : Nat

/-- Calculates the total bushels needed for a day -/
def total_bushels (farm : Farm) : Nat :=
  (farm.cows + farm.sheep) * farm.cow_sheep_consumption + 
  farm.chickens * farm.chicken_consumption

/-- Dany's farm -/
def danys_farm : Farm := {
  cows := 4,
  sheep := 3,
  chickens := 7,
  cow_sheep_consumption := 2,
  chicken_consumption := 3
}

/-- Theorem stating that Dany needs 35 bushels for a day -/
theorem danys_farm_bushels : total_bushels danys_farm = 35 := by
  sorry

end NUMINAMATH_CALUDE_danys_farm_bushels_l4002_400236


namespace NUMINAMATH_CALUDE_saras_sister_notebooks_l4002_400231

theorem saras_sister_notebooks (initial final ordered lost : ℕ) : 
  final = 8 → ordered = 6 → lost = 2 → initial + ordered - lost = final → initial = 4 := by
  sorry

end NUMINAMATH_CALUDE_saras_sister_notebooks_l4002_400231


namespace NUMINAMATH_CALUDE_dans_candy_bars_l4002_400223

theorem dans_candy_bars (total_spent : ℝ) (cost_per_bar : ℝ) (h1 : total_spent = 4) (h2 : cost_per_bar = 2) :
  total_spent / cost_per_bar = 2 := by
  sorry

end NUMINAMATH_CALUDE_dans_candy_bars_l4002_400223


namespace NUMINAMATH_CALUDE_f_of_5_l4002_400227

def f (x : ℝ) : ℝ := x^2 - x

theorem f_of_5 : f 5 = 20 := by sorry

end NUMINAMATH_CALUDE_f_of_5_l4002_400227


namespace NUMINAMATH_CALUDE_line_tangent_to_parabola_l4002_400217

theorem line_tangent_to_parabola :
  ∃! d : ℝ, ∀ x y : ℝ,
    (y = 3 * x + d) ∧ (y^2 = 12 * x) →
    (∃! t : ℝ, y = 3 * t + d ∧ y^2 = 12 * t) →
    d = 1 := by
  sorry

end NUMINAMATH_CALUDE_line_tangent_to_parabola_l4002_400217


namespace NUMINAMATH_CALUDE_derivative_product_at_4_and_neg1_l4002_400241

noncomputable def f (x : ℝ) : ℝ :=
  if x > 0 then -1 / Real.sqrt x else 1 + x^2

theorem derivative_product_at_4_and_neg1 :
  (deriv f 4) * (deriv f (-1)) = -1/8 := by sorry

end NUMINAMATH_CALUDE_derivative_product_at_4_and_neg1_l4002_400241


namespace NUMINAMATH_CALUDE_airplane_seats_total_l4002_400296

theorem airplane_seats_total (first_class : ℕ) (coach : ℕ) : 
  first_class = 77 → 
  coach = 4 * first_class + 2 → 
  first_class + coach = 387 := by
sorry

end NUMINAMATH_CALUDE_airplane_seats_total_l4002_400296


namespace NUMINAMATH_CALUDE_right_triangle_sine_l4002_400205

theorem right_triangle_sine (X Y Z : ℝ) : 
  -- XYZ is a right triangle with Y as the right angle
  (X + Y + Z = π) →
  (Y = π / 2) →
  -- sin X = 8/17
  (Real.sin X = 8 / 17) →
  -- Conclusion: sin Z = 15/17
  (Real.sin Z = 15 / 17) := by
sorry

end NUMINAMATH_CALUDE_right_triangle_sine_l4002_400205


namespace NUMINAMATH_CALUDE_no_matching_product_and_sum_l4002_400289

theorem no_matching_product_and_sum : 
  ¬ ∃ (a b : ℕ), 1 ≤ a ∧ a < b ∧ b ≤ 15 ∧ 
  a * b = (List.range 16).sum - a - b :=
by sorry

end NUMINAMATH_CALUDE_no_matching_product_and_sum_l4002_400289


namespace NUMINAMATH_CALUDE_max_guaranteed_amount_l4002_400252

/-- Represents a set of bank cards with values from 1 to n rubles -/
def BankCards (n : ℕ) := Finset (Fin n)

/-- The strategy function takes a number of cards and returns the optimal request amount -/
def strategy (n : ℕ) : ℕ := n / 2

/-- Calculates the guaranteed amount for a given strategy on a set of cards -/
def guaranteedAmount (cards : BankCards 100) (s : ℕ → ℕ) : ℕ :=
  (cards.filter (λ i => i.val + 1 ≥ s 100)).card * s 100

theorem max_guaranteed_amount :
  ∀ (cards : BankCards 100),
    ∀ (s : ℕ → ℕ),
      guaranteedAmount cards s ≤ guaranteedAmount cards strategy ∧
      guaranteedAmount cards strategy = 2550 := by
  sorry

#eval strategy 100  -- Should output 50

end NUMINAMATH_CALUDE_max_guaranteed_amount_l4002_400252


namespace NUMINAMATH_CALUDE_power_mod_seven_l4002_400292

theorem power_mod_seven : 76^77 % 7 = 6 := by
  sorry

end NUMINAMATH_CALUDE_power_mod_seven_l4002_400292


namespace NUMINAMATH_CALUDE_zephyr_in_top_three_l4002_400235

-- Define the propositions
variable (X : Prop) -- Xenon is in the top three
variable (Y : Prop) -- Yenofa is in the top three
variable (Z : Prop) -- Zephyr is in the top three

-- Define the conditions
axiom condition1 : Z → X
axiom condition2 : (X ∨ Y) → ¬Z
axiom condition3 : ¬((X ∨ Y) → ¬Z)

-- Theorem to prove
theorem zephyr_in_top_three : Z ∧ ¬X ∧ ¬Y := by
  sorry

end NUMINAMATH_CALUDE_zephyr_in_top_three_l4002_400235


namespace NUMINAMATH_CALUDE_sqrt_multiplication_equality_l4002_400244

theorem sqrt_multiplication_equality : 3 * Real.sqrt 2 * Real.sqrt 6 = 6 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_multiplication_equality_l4002_400244


namespace NUMINAMATH_CALUDE_fundraising_contribution_l4002_400215

theorem fundraising_contribution
  (total_goal : ℕ)
  (num_participants : ℕ)
  (admin_fee : ℕ)
  (h1 : total_goal = 2400)
  (h2 : num_participants = 8)
  (h3 : admin_fee = 20) :
  (total_goal / num_participants) + admin_fee = 320 := by
sorry

end NUMINAMATH_CALUDE_fundraising_contribution_l4002_400215


namespace NUMINAMATH_CALUDE_balanced_scale_l4002_400277

/-- The weight of a children's book in kilograms. -/
def book_weight : ℝ := 1.1

/-- The weight of a doll in kilograms. -/
def doll_weight : ℝ := 0.3

/-- The weight of a toy car in kilograms. -/
def toy_car_weight : ℝ := 0.5

/-- The number of dolls on the scale. -/
def num_dolls : ℕ := 2

/-- The number of toy cars on the scale. -/
def num_toy_cars : ℕ := 1

theorem balanced_scale : 
  book_weight = num_dolls * doll_weight + num_toy_cars * toy_car_weight :=
by sorry

end NUMINAMATH_CALUDE_balanced_scale_l4002_400277


namespace NUMINAMATH_CALUDE_number_equality_l4002_400253

theorem number_equality (y : ℚ) : 
  (30 / 100 : ℚ) * y = (25 / 100 : ℚ) * 40 → y = 100 / 3 := by
  sorry

end NUMINAMATH_CALUDE_number_equality_l4002_400253


namespace NUMINAMATH_CALUDE_consecutive_triples_divisible_by_1001_l4002_400274

def is_valid_triple (a b c : ℕ) : Prop :=
  a < 101 ∧ b < 101 ∧ c < 101 ∧
  b = a + 1 ∧ c = b + 1 ∧
  (a * b * c) % 1001 = 0

theorem consecutive_triples_divisible_by_1001 :
  ∀ a b c : ℕ,
    is_valid_triple a b c ↔ (a = 76 ∧ b = 77 ∧ c = 78) ∨ (a = 77 ∧ b = 78 ∧ c = 79) :=
by sorry

end NUMINAMATH_CALUDE_consecutive_triples_divisible_by_1001_l4002_400274


namespace NUMINAMATH_CALUDE_koschei_coins_count_l4002_400294

theorem koschei_coins_count :
  ∃! n : ℕ, 300 ≤ n ∧ n ≤ 400 ∧ n % 10 = 7 ∧ n % 12 = 9 ∧ n = 357 := by
  sorry

end NUMINAMATH_CALUDE_koschei_coins_count_l4002_400294


namespace NUMINAMATH_CALUDE_fourth_child_receives_24_l4002_400218

/-- Represents the distribution of sweets among a mother and her children -/
structure SweetDistribution where
  total : ℕ
  mother_fraction : ℚ
  num_children : ℕ
  eldest_youngest_ratio : ℕ
  second_third_diff : ℕ
  third_fourth_diff : ℕ
  youngest_second_ratio : ℚ

/-- Calculates the number of sweets the fourth child receives -/
def fourth_child_sweets (d : SweetDistribution) : ℕ :=
  sorry

/-- Theorem stating that given the problem conditions, the fourth child receives 24 sweets -/
theorem fourth_child_receives_24 (d : SweetDistribution) 
  (h1 : d.total = 120)
  (h2 : d.mother_fraction = 1/4)
  (h3 : d.num_children = 5)
  (h4 : d.eldest_youngest_ratio = 2)
  (h5 : d.second_third_diff = 6)
  (h6 : d.third_fourth_diff = 8)
  (h7 : d.youngest_second_ratio = 4/5) : 
  fourth_child_sweets d = 24 :=
sorry

end NUMINAMATH_CALUDE_fourth_child_receives_24_l4002_400218


namespace NUMINAMATH_CALUDE_lindas_nickels_l4002_400295

/-- The number of nickels Linda initially has -/
def initial_nickels : ℕ := 5

/-- The total number of coins Linda has after receiving additional coins -/
def total_coins : ℕ := 35

/-- The number of initial dimes -/
def initial_dimes : ℕ := 2

/-- The number of initial quarters -/
def initial_quarters : ℕ := 6

/-- The number of additional dimes given by her mother -/
def additional_dimes : ℕ := 2

/-- The number of additional quarters given by her mother -/
def additional_quarters : ℕ := 10

theorem lindas_nickels :
  initial_dimes + initial_quarters + initial_nickels +
  additional_dimes + additional_quarters + 2 * initial_nickels = total_coins :=
by sorry

end NUMINAMATH_CALUDE_lindas_nickels_l4002_400295


namespace NUMINAMATH_CALUDE_light_reflection_theorem_l4002_400287

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a quadrilateral -/
structure Quadrilateral where
  A : Point
  B : Point
  C : Point
  D : Point

/-- Checks if a point is on a line segment -/
def isOnSegment (P : Point) (A : Point) (B : Point) : Prop := sorry

/-- Checks if a quadrilateral is cyclic -/
def isCyclic (q : Quadrilateral) : Prop := sorry

/-- Calculates the perimeter of a quadrilateral -/
def perimeter (q : Quadrilateral) : ℝ := sorry

/-- Represents the light ray path -/
structure LightPath (q : Quadrilateral) where
  P : Point
  Q : Point
  R : Point
  S : Point
  pOnAB : isOnSegment P q.A q.B
  qOnBC : isOnSegment Q q.B q.C
  rOnCD : isOnSegment R q.C q.D
  sOnDA : isOnSegment S q.D q.A

theorem light_reflection_theorem (q : Quadrilateral) :
  (∀ (path : LightPath q), isCyclic q) ∧
  (∃ (c : ℝ), ∀ (path : LightPath q), perimeter ⟨path.P, path.Q, path.R, path.S⟩ = c) :=
sorry

end NUMINAMATH_CALUDE_light_reflection_theorem_l4002_400287


namespace NUMINAMATH_CALUDE_special_function_unique_l4002_400270

/-- A function f: ℤ × ℤ → ℝ satisfying specific conditions -/
def special_function (f : ℤ × ℤ → ℝ) : Prop :=
  (∀ x y z : ℤ, f (x, y) * f (y, z) * f (z, x) = 1) ∧
  (∀ x : ℤ, f (x + 1, x) = 2)

/-- Theorem stating that any function satisfying the special_function conditions 
    must be of the form f(x,y) = 2^(x-y) -/
theorem special_function_unique (f : ℤ × ℤ → ℝ) 
  (hf : special_function f) : 
  ∀ x y : ℤ, f (x, y) = 2^(x - y) := by
  sorry

end NUMINAMATH_CALUDE_special_function_unique_l4002_400270


namespace NUMINAMATH_CALUDE_triangle_tangent_product_l4002_400210

-- Define a triangle ABC
structure Triangle where
  A : ℝ
  B : ℝ
  C : ℝ
  a : ℝ
  b : ℝ
  c : ℝ

-- Define the theorem
theorem triangle_tangent_product (t : Triangle) 
  (h : t.a + t.c = 2 * t.b) : 
  Real.tan (t.A / 2) * Real.tan (t.C / 2) = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_triangle_tangent_product_l4002_400210


namespace NUMINAMATH_CALUDE_original_number_l4002_400263

theorem original_number (N : ℤ) : 
  (∃ k : ℤ, N + 4 = 25 * k) ∧ 
  (∀ m : ℤ, m < 4 → ¬(∃ j : ℤ, N + m = 25 * j)) →
  N = 21 := by
sorry

end NUMINAMATH_CALUDE_original_number_l4002_400263


namespace NUMINAMATH_CALUDE_min_perimeter_cross_section_min_perimeter_problem_l4002_400248

/-- Regular triangular pyramid with given dimensions -/
structure RegularTriangularPyramid where
  base_side_length : ℝ
  lateral_edge_length : ℝ

/-- Intersection plane for the pyramid -/
structure IntersectionPlane where
  base_point : Point
  intersection_point1 : Point
  intersection_point2 : Point

/-- Theorem stating the minimum perimeter of the cross-sectional triangle -/
theorem min_perimeter_cross_section 
  (pyramid : RegularTriangularPyramid) 
  (plane : IntersectionPlane) : ℝ :=
  sorry

/-- Main theorem proving the minimum perimeter for the given problem -/
theorem min_perimeter_problem : 
  ∀ (pyramid : RegularTriangularPyramid) 
    (plane : IntersectionPlane),
  pyramid.base_side_length = 4 ∧ 
  pyramid.lateral_edge_length = 8 →
  min_perimeter_cross_section pyramid plane = 11 :=
sorry

end NUMINAMATH_CALUDE_min_perimeter_cross_section_min_perimeter_problem_l4002_400248


namespace NUMINAMATH_CALUDE_possible_values_of_a_l4002_400261

theorem possible_values_of_a (a b c d : ℕ+) 
  (h1 : a > b ∧ b > c ∧ c > d)
  (h2 : a + b + c + d = 2020)
  (h3 : a^2 - b^2 + c^2 - d^2 = 2020) :
  ∃! s : Finset ℕ+, s.card = 501 ∧ ∀ x, x ∈ s ↔ ∃ b' c' d' : ℕ+, 
    x > b' ∧ b' > c' ∧ c' > d' ∧
    x + b' + c' + d' = 2020 ∧
    x^2 - b'^2 + c'^2 - d'^2 = 2020 :=
by sorry

end NUMINAMATH_CALUDE_possible_values_of_a_l4002_400261


namespace NUMINAMATH_CALUDE_intersection_and_conditions_l4002_400298

-- Define the intersection point M
def M : ℝ × ℝ := (-1, 2)

-- Define the given lines
def line1 (x y : ℝ) : Prop := 3 * x + 4 * y - 5 = 0
def line2 (x y : ℝ) : Prop := 2 * x - 3 * y + 8 = 0
def line3 (x y : ℝ) : Prop := 2 * x + y + 5 = 0

-- Define the resulting lines
def result_line1 (x y : ℝ) : Prop := x = -1
def result_line2 (x y : ℝ) : Prop := x - 2 * y + 5 = 0

theorem intersection_and_conditions :
  -- M is the intersection point of line1 and line2
  (line1 M.1 M.2 ∧ line2 M.1 M.2) ∧
  -- result_line1 passes through M and (-1, 0)
  (result_line1 M.1 M.2 ∧ result_line1 (-1) 0) ∧
  -- result_line2 passes through M
  result_line2 M.1 M.2 ∧
  -- result_line2 is perpendicular to line3
  (∃ (k : ℝ), k ≠ 0 ∧ 1 * 2 + (-2) * 1 = -k * k) :=
by sorry

end NUMINAMATH_CALUDE_intersection_and_conditions_l4002_400298


namespace NUMINAMATH_CALUDE_haley_recycling_cans_l4002_400216

theorem haley_recycling_cans (collected : ℕ) (in_bag : ℕ) 
  (h1 : collected = 9) (h2 : in_bag = 7) : 
  collected - in_bag = 2 := by
  sorry

end NUMINAMATH_CALUDE_haley_recycling_cans_l4002_400216


namespace NUMINAMATH_CALUDE_fraction_sum_l4002_400285

theorem fraction_sum : (3 : ℚ) / 4 + 9 / 12 = 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_l4002_400285


namespace NUMINAMATH_CALUDE_diagonal_intersection_y_value_l4002_400262

/-- A square in the coordinate plane with specific properties -/
structure Square where
  vertex : ℝ × ℝ
  diagonal_intersection_x : ℝ
  area : ℝ

/-- The y-coordinate of the diagonal intersection point of the square -/
def diagonal_intersection_y (s : Square) : ℝ :=
  s.vertex.2 + (s.diagonal_intersection_x - s.vertex.1)

/-- Theorem stating the y-coordinate of the diagonal intersection point -/
theorem diagonal_intersection_y_value (s : Square) 
  (h1 : s.vertex = (-6, -4))
  (h2 : s.diagonal_intersection_x = 3)
  (h3 : s.area = 324) :
  diagonal_intersection_y s = 5 := by
  sorry


end NUMINAMATH_CALUDE_diagonal_intersection_y_value_l4002_400262


namespace NUMINAMATH_CALUDE_problem_solution_l4002_400201

theorem problem_solution (m n : ℕ+) 
  (h1 : m.val + 5 < n.val)
  (h2 : (m.val + (m.val + 3) + (m.val + 5) + n.val + (n.val + 1) + (2 * n.val - 1)) / 6 = n.val)
  (h3 : (m.val + 5 + n.val) / 2 = n.val) : 
  m.val + n.val = 11 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l4002_400201


namespace NUMINAMATH_CALUDE_bicycle_cost_price_l4002_400290

theorem bicycle_cost_price 
  (profit_A_to_B : ℝ) 
  (profit_B_to_C : ℝ) 
  (final_price : ℝ) 
  (h1 : profit_A_to_B = 0.35)
  (h2 : profit_B_to_C = 0.45)
  (h3 : final_price = 225) :
  final_price / ((1 + profit_A_to_B) * (1 + profit_B_to_C)) = 
    final_price / (1.35 * 1.45) := by
  sorry

end NUMINAMATH_CALUDE_bicycle_cost_price_l4002_400290


namespace NUMINAMATH_CALUDE_sin_theta_plus_pi_fourth_l4002_400234

theorem sin_theta_plus_pi_fourth (θ : Real) 
  (h1 : θ > π/2 ∧ θ < π) 
  (h2 : Real.tan (θ - π/4) = -4/3) : 
  Real.sin (θ + π/4) = -3/5 := by
  sorry

end NUMINAMATH_CALUDE_sin_theta_plus_pi_fourth_l4002_400234


namespace NUMINAMATH_CALUDE_circle_equation_and_max_ratio_l4002_400278

-- Define the circle C
def circle_C (x y : ℝ) : Prop :=
  (x + 1)^2 + y^2 = 2

-- Define the given line equations
def line1 (x y : ℝ) : Prop := x - y + 1 = 0
def line2 (x y : ℝ) : Prop := x + y + 3 = 0

-- Define the second circle
def circle2 (x y : ℝ) : Prop :=
  x^2 + y^2 - 4*y + 3 = 0

theorem circle_equation_and_max_ratio :
  (∃ (x₀ y₀ : ℝ), line1 x₀ y₀ ∧ y₀ = 0 ∧
    ∀ (x y : ℝ), circle_C x y ↔ (x - x₀)^2 + (y - y₀)^2 = 2 ∧
    ∃ (x₁ y₁ : ℝ), line2 x₁ y₁ ∧ (x₁ - x₀)^2 + (y₁ - y₀)^2 = 2) ∧
  (∀ (x y : ℝ), circle2 x y → y / x ≤ Real.sqrt 3 / 3) ∧
  (∃ (x y : ℝ), circle2 x y ∧ y / x = Real.sqrt 3 / 3) :=
sorry

end NUMINAMATH_CALUDE_circle_equation_and_max_ratio_l4002_400278


namespace NUMINAMATH_CALUDE_birthday_cake_problem_l4002_400279

/-- Represents a cube cake with icing -/
structure CakeCube where
  size : Nat
  has_icing : Bool

/-- Counts the number of small cubes with icing on exactly two sides -/
def count_two_sided_icing (cake : CakeCube) : Nat :=
  sorry

/-- The main theorem about the birthday cake problem -/
theorem birthday_cake_problem (cake : CakeCube) :
  cake.size = 5 ∧ cake.has_icing = true → count_two_sided_icing cake = 96 := by
  sorry

end NUMINAMATH_CALUDE_birthday_cake_problem_l4002_400279


namespace NUMINAMATH_CALUDE_may_cookie_cost_l4002_400288

/-- The total amount spent on cookies in May -/
def total_cookie_cost (weekday_count : ℕ) (weekend_count : ℕ) 
  (weekday_cookie_count : ℕ) (weekend_cookie_count : ℕ)
  (weekday_cookie1_price : ℕ) (weekday_cookie2_price : ℕ)
  (weekend_cookie1_price : ℕ) (weekend_cookie2_price : ℕ) : ℕ :=
  (weekday_count * (2 * weekday_cookie1_price + 2 * weekday_cookie2_price)) +
  (weekend_count * (3 * weekend_cookie1_price + 2 * weekend_cookie2_price))

/-- Theorem stating the total amount spent on cookies in May -/
theorem may_cookie_cost : 
  total_cookie_cost 22 9 4 5 15 18 12 20 = 2136 := by
  sorry

end NUMINAMATH_CALUDE_may_cookie_cost_l4002_400288


namespace NUMINAMATH_CALUDE_connie_marbles_l4002_400219

/-- The number of marbles Connie has after giving some away -/
def remaining_marbles (initial : ℝ) (given_away : ℝ) : ℝ :=
  initial - given_away

/-- Theorem stating that Connie has 3.2 marbles after giving some away -/
theorem connie_marbles : remaining_marbles 73.5 70.3 = 3.2 := by
  sorry

end NUMINAMATH_CALUDE_connie_marbles_l4002_400219


namespace NUMINAMATH_CALUDE_min_distance_point_to_circle_l4002_400256

/-- The minimum distance between the point (3,4) and any point on the circle x^2 + y^2 = 1 is 4 -/
theorem min_distance_point_to_circle : ∃ (d : ℝ),
  d = 4 ∧
  ∀ (x y : ℝ), x^2 + y^2 = 1 → 
  d ≤ Real.sqrt ((x - 3)^2 + (y - 4)^2) :=
by sorry

end NUMINAMATH_CALUDE_min_distance_point_to_circle_l4002_400256


namespace NUMINAMATH_CALUDE_jiaki_calculation_final_result_l4002_400240

-- Define A as a function of x
def A (x : ℤ) : ℤ := 3 * x^2 - x + 1

-- Define B as a function of x
def B (x : ℤ) : ℤ := -x^2 - 2*x - 3

-- State the theorem
theorem jiaki_calculation (x : ℤ) :
  A x - B x = 2 * x^2 - 3*x - 2 ∧
  (x = -1 → A x - B x = 3) :=
by sorry

-- Define the largest negative integer
def largest_negative_integer : ℤ := -1

-- State the final result
theorem final_result :
  A largest_negative_integer - B largest_negative_integer = 3 :=
by sorry

end NUMINAMATH_CALUDE_jiaki_calculation_final_result_l4002_400240


namespace NUMINAMATH_CALUDE_power_product_cube_l4002_400254

theorem power_product_cube (a b : ℝ) : (a * b^2)^3 = a^3 * b^6 := by
  sorry

end NUMINAMATH_CALUDE_power_product_cube_l4002_400254


namespace NUMINAMATH_CALUDE_max_sundays_in_84_days_l4002_400228

theorem max_sundays_in_84_days : ℕ :=
  let days_in_period : ℕ := 84
  let days_in_week : ℕ := 7
  let sundays_per_week : ℕ := 1

  have h1 : days_in_period % days_in_week = 0 := by sorry
  have h2 : days_in_period / days_in_week * sundays_per_week = 12 := by sorry

  12

/- Proof
sorry
-/

end NUMINAMATH_CALUDE_max_sundays_in_84_days_l4002_400228


namespace NUMINAMATH_CALUDE_complex_square_i_positive_l4002_400238

theorem complex_square_i_positive (a : ℝ) : 
  (((a + Complex.I) ^ 2) * Complex.I).re > 0 → a = -1 := by
  sorry

end NUMINAMATH_CALUDE_complex_square_i_positive_l4002_400238


namespace NUMINAMATH_CALUDE_sin_cos_sum_special_angle_l4002_400259

theorem sin_cos_sum_special_angle : 
  Real.sin (5 * π / 180) * Real.cos (55 * π / 180) + 
  Real.cos (5 * π / 180) * Real.sin (55 * π / 180) = 
  Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_sum_special_angle_l4002_400259


namespace NUMINAMATH_CALUDE_cube_root_8000_l4002_400271

theorem cube_root_8000 : ∃ (a b : ℕ+), (a : ℝ) * (b : ℝ)^(1/3 : ℝ) = 8000^(1/3 : ℝ) ∧ b = 1 := by
  sorry

end NUMINAMATH_CALUDE_cube_root_8000_l4002_400271


namespace NUMINAMATH_CALUDE_reciprocal_sum_of_roots_l4002_400203

theorem reciprocal_sum_of_roots (α β : ℝ) : 
  (∃ x y : ℝ, 7 * x^2 - 6 * x + 8 = 0 ∧ 
               7 * y^2 - 6 * y + 8 = 0 ∧ 
               x ≠ y ∧ 
               α = 1 / x ∧ 
               β = 1 / y) → 
  α + β = 3/4 := by
sorry


end NUMINAMATH_CALUDE_reciprocal_sum_of_roots_l4002_400203


namespace NUMINAMATH_CALUDE_libby_igloo_top_bricks_l4002_400246

/-- Represents the structure of an igloo --/
structure Igloo where
  total_rows : ℕ
  bottom_rows : ℕ
  top_rows : ℕ
  bottom_bricks_per_row : ℕ
  total_bricks : ℕ

/-- Calculates the number of bricks in each row of the top half of the igloo --/
def top_bricks_per_row (i : Igloo) : ℕ :=
  (i.total_bricks - i.bottom_rows * i.bottom_bricks_per_row) / i.top_rows

/-- Theorem stating the number of bricks in each row of the top half of Libby's igloo --/
theorem libby_igloo_top_bricks :
  let i : Igloo := {
    total_rows := 10,
    bottom_rows := 5,
    top_rows := 5,
    bottom_bricks_per_row := 12,
    total_bricks := 100
  }
  top_bricks_per_row i = 8 := by
  sorry

end NUMINAMATH_CALUDE_libby_igloo_top_bricks_l4002_400246


namespace NUMINAMATH_CALUDE_circle_center_transformation_l4002_400232

/-- Reflects a point across the x-axis -/
def reflect_x (p : ℝ × ℝ) : ℝ × ℝ :=
  (p.1, -p.2)

/-- Reflects a point across the y-axis -/
def reflect_y (p : ℝ × ℝ) : ℝ × ℝ :=
  (-p.1, p.2)

/-- Translates a point vertically by a given amount -/
def translate_vertical (p : ℝ × ℝ) (d : ℝ) : ℝ × ℝ :=
  (p.1, p.2 + d)

/-- The initial center of circle U -/
def initial_center : ℝ × ℝ := (3, -4)

/-- The transformation applied to the center of circle U -/
def transform (p : ℝ × ℝ) : ℝ × ℝ :=
  translate_vertical (reflect_y (reflect_x p)) (-10)

theorem circle_center_transformation :
  transform initial_center = (-3, -6) := by sorry

end NUMINAMATH_CALUDE_circle_center_transformation_l4002_400232


namespace NUMINAMATH_CALUDE_event_ratio_l4002_400275

theorem event_ratio (total : ℕ) (children : ℕ) (adults : ℕ) : 
  total = 42 → children = 28 → adults = total - children → 
  (children : ℚ) / (adults : ℚ) = 2 / 1 := by
  sorry

end NUMINAMATH_CALUDE_event_ratio_l4002_400275


namespace NUMINAMATH_CALUDE_balloon_blowup_ratio_l4002_400291

theorem balloon_blowup_ratio (total : ℕ) (intact : ℕ) : 
  total = 200 → 
  intact = 80 → 
  (total - intact) / (total / 5) = 2 := by
sorry

end NUMINAMATH_CALUDE_balloon_blowup_ratio_l4002_400291


namespace NUMINAMATH_CALUDE_optimal_arrangement_l4002_400230

/-- Represents the capacity and cost of a truck type -/
structure TruckType where
  water_capacity : ℕ
  vegetable_capacity : ℕ
  cost : ℕ

/-- Represents the donation quantities and truck types -/
structure DonationProblem where
  total_donation : ℕ
  water_vegetable_diff : ℕ
  type_a : TruckType
  type_b : TruckType
  total_trucks : ℕ

def problem : DonationProblem :=
  { total_donation := 120
  , water_vegetable_diff := 12
  , type_a := { water_capacity := 5, vegetable_capacity := 8, cost := 400 }
  , type_b := { water_capacity := 6, vegetable_capacity := 6, cost := 360 }
  , total_trucks := 10
  }

def water_amount (p : DonationProblem) : ℕ :=
  (p.total_donation - p.water_vegetable_diff) / 2

def vegetable_amount (p : DonationProblem) : ℕ :=
  p.total_donation - water_amount p

def is_valid_arrangement (p : DonationProblem) (type_a_count : ℕ) : Prop :=
  let type_b_count := p.total_trucks - type_a_count
  type_a_count * p.type_a.water_capacity + type_b_count * p.type_b.water_capacity ≥ water_amount p ∧
  type_a_count * p.type_a.vegetable_capacity + type_b_count * p.type_b.vegetable_capacity ≥ vegetable_amount p

def transportation_cost (p : DonationProblem) (type_a_count : ℕ) : ℕ :=
  type_a_count * p.type_a.cost + (p.total_trucks - type_a_count) * p.type_b.cost

theorem optimal_arrangement (p : DonationProblem) :
  ∃ (type_a_count : ℕ),
    type_a_count = 3 ∧
    is_valid_arrangement p type_a_count ∧
    ∀ (other_count : ℕ),
      is_valid_arrangement p other_count →
      transportation_cost p type_a_count ≤ transportation_cost p other_count :=
sorry

#eval transportation_cost problem 3  -- Should evaluate to 3720

end NUMINAMATH_CALUDE_optimal_arrangement_l4002_400230


namespace NUMINAMATH_CALUDE_g_comp_three_roots_l4002_400200

/-- A quadratic function g(x) = x^2 + 4x + d where d is a real parameter -/
def g (d : ℝ) (x : ℝ) : ℝ := x^2 + 4*x + d

/-- The composition of g with itself -/
def g_comp (d : ℝ) (x : ℝ) : ℝ := g d (g d x)

/-- Theorem stating that g(g(x)) has exactly 3 distinct real roots iff d = 0 -/
theorem g_comp_three_roots (d : ℝ) : 
  (∃! (s : Finset ℝ), s.card = 3 ∧ ∀ x ∈ s, g_comp d x = 0) ↔ d = 0 := by
  sorry

end NUMINAMATH_CALUDE_g_comp_three_roots_l4002_400200


namespace NUMINAMATH_CALUDE_james_monthly_earnings_l4002_400266

/-- Calculates the monthly earnings of a Twitch streamer --/
def monthly_earnings (initial_subscribers : ℕ) (gifted_subscribers : ℕ) (earnings_per_subscriber : ℕ) : ℕ :=
  (initial_subscribers + gifted_subscribers) * earnings_per_subscriber

theorem james_monthly_earnings :
  monthly_earnings 150 50 9 = 1800 := by
  sorry

end NUMINAMATH_CALUDE_james_monthly_earnings_l4002_400266


namespace NUMINAMATH_CALUDE_range_of_a_l4002_400243

/-- The piecewise function f(x) as defined in the problem -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x > 2 then 2^x + a else x + a^2

/-- The theorem stating the range of a given the conditions -/
theorem range_of_a (a : ℝ) : 
  (∀ y : ℝ, ∃ x : ℝ, f a x = y) → 
  (a ≤ -1 ∨ a ≥ 2) :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l4002_400243


namespace NUMINAMATH_CALUDE_paint_bought_l4002_400267

theorem paint_bought (total_needed paint_existing paint_still_needed : ℕ) 
  (h1 : total_needed = 70)
  (h2 : paint_existing = 36)
  (h3 : paint_still_needed = 11) :
  total_needed - paint_existing - paint_still_needed = 23 :=
by sorry

end NUMINAMATH_CALUDE_paint_bought_l4002_400267


namespace NUMINAMATH_CALUDE_equilateral_triangle_division_l4002_400207

theorem equilateral_triangle_division (side_length : ℕ) (h : side_length = 1536) :
  ∃ (n : ℕ), side_length^2 = 3 * n := by
  sorry

end NUMINAMATH_CALUDE_equilateral_triangle_division_l4002_400207


namespace NUMINAMATH_CALUDE_fourth_grade_students_l4002_400225

/-- Calculates the final number of students in fourth grade -/
def final_student_count (initial : ℕ) (left : ℕ) (new : ℕ) : ℕ :=
  initial - left + new

/-- Theorem stating that the final number of students is 43 -/
theorem fourth_grade_students : final_student_count 4 3 42 = 43 := by
  sorry

end NUMINAMATH_CALUDE_fourth_grade_students_l4002_400225


namespace NUMINAMATH_CALUDE_unique_triangle_arrangement_l4002_400269

-- Define the structure of the triangle
structure Triangle :=
  (A B C D : ℕ)
  (side1 side2 side3 : ℕ)

-- Define the conditions of the problem
def validTriangle (t : Triangle) : Prop :=
  t.A ∈ ({6, 7, 8, 9} : Set ℕ) ∧
  t.B ∈ ({6, 7, 8, 9} : Set ℕ) ∧
  t.C ∈ ({6, 7, 8, 9} : Set ℕ) ∧
  t.D ∈ ({6, 7, 8, 9} : Set ℕ) ∧
  t.A ≠ t.B ∧ t.A ≠ t.C ∧ t.A ≠ t.D ∧
  t.B ≠ t.C ∧ t.B ≠ t.D ∧
  t.C ≠ t.D ∧
  t.side1 = 1 + t.B + 5 ∧
  t.side2 = 3 + 4 + t.D ∧
  t.side3 = 2 + t.A + 4 ∧
  t.side1 = t.side2 ∧ t.side2 = t.side3

-- Theorem statement
theorem unique_triangle_arrangement :
  ∃! t : Triangle, validTriangle t ∧ t.A = 6 ∧ t.B = 8 ∧ t.C = 7 ∧ t.D = 9 :=
sorry

end NUMINAMATH_CALUDE_unique_triangle_arrangement_l4002_400269


namespace NUMINAMATH_CALUDE_ratio_x_to_y_l4002_400247

theorem ratio_x_to_y (x y : ℚ) (h : (15 * x - 4 * y) / (18 * x - 3 * y) = 2 / 3) :
  x / y = 2 / 3 := by sorry

end NUMINAMATH_CALUDE_ratio_x_to_y_l4002_400247


namespace NUMINAMATH_CALUDE_jenny_cans_collected_l4002_400284

/-- Represents the number of cans Jenny collects -/
def num_cans : ℕ := 20

/-- Represents the number of bottles Jenny collects -/
def num_bottles : ℕ := (100 - 2 * num_cans) / 6

/-- The weight of a bottle in ounces -/
def bottle_weight : ℕ := 6

/-- The weight of a can in ounces -/
def can_weight : ℕ := 2

/-- The payment for a bottle in cents -/
def bottle_payment : ℕ := 10

/-- The payment for a can in cents -/
def can_payment : ℕ := 3

/-- The total weight Jenny can carry in ounces -/
def total_weight : ℕ := 100

/-- The total payment Jenny receives in cents -/
def total_payment : ℕ := 160

theorem jenny_cans_collected :
  (num_bottles * bottle_weight + num_cans * can_weight = total_weight) ∧
  (num_bottles * bottle_payment + num_cans * can_payment = total_payment) :=
sorry

end NUMINAMATH_CALUDE_jenny_cans_collected_l4002_400284


namespace NUMINAMATH_CALUDE_same_terminal_side_angle_l4002_400204

theorem same_terminal_side_angle : ∃ k : ℤ, k * 360 - 70 = 290 := by
  sorry

end NUMINAMATH_CALUDE_same_terminal_side_angle_l4002_400204


namespace NUMINAMATH_CALUDE_book_cost_is_15_l4002_400237

def total_books : ℕ := 10
def num_magazines : ℕ := 10
def magazine_cost : ℚ := 2
def total_spent : ℚ := 170

theorem book_cost_is_15 :
  ∃ (book_cost : ℚ),
    book_cost * total_books + magazine_cost * num_magazines = total_spent ∧
    book_cost = 15 :=
by sorry

end NUMINAMATH_CALUDE_book_cost_is_15_l4002_400237


namespace NUMINAMATH_CALUDE_arithmetic_square_root_of_81_l4002_400273

theorem arithmetic_square_root_of_81 : Real.sqrt 81 = 9 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_square_root_of_81_l4002_400273


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l4002_400229

theorem sufficient_not_necessary (a b c d : ℝ) :
  (a > b ∧ c > d → a * c + b * d > b * c + a * d) ∧
  ∃ a b c d : ℝ, a * c + b * d > b * c + a * d ∧ ¬(a > b ∧ c > d) :=
sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l4002_400229


namespace NUMINAMATH_CALUDE_managers_salary_l4002_400249

theorem managers_salary (num_employees : ℕ) (avg_salary : ℚ) (avg_increase : ℚ) :
  num_employees = 20 ∧ 
  avg_salary = 1600 ∧ 
  avg_increase = 100 →
  (num_employees * avg_salary + (avg_salary + avg_increase) * (num_employees + 1) - num_employees * avg_salary) = 3700 :=
by sorry

end NUMINAMATH_CALUDE_managers_salary_l4002_400249


namespace NUMINAMATH_CALUDE_greatest_integer_gcd_18_is_9_l4002_400251

theorem greatest_integer_gcd_18_is_9 :
  ∃ n : ℕ, n < 200 ∧ n.gcd 18 = 9 ∧ ∀ m : ℕ, m < 200 ∧ m.gcd 18 = 9 → m ≤ n :=
by sorry

end NUMINAMATH_CALUDE_greatest_integer_gcd_18_is_9_l4002_400251


namespace NUMINAMATH_CALUDE_systematic_sampling_result_l4002_400281

/-- Systematic sampling function -/
def systematic_sample (total : ℕ) (sample_size : ℕ) (last_sampled : ℕ) : List ℕ :=
  sorry

/-- Theorem for systematic sampling results -/
theorem systematic_sampling_result 
  (total : ℕ) 
  (sample_size : ℕ) 
  (last_sampled : ℕ) 
  (h1 : total = 8000) 
  (h2 : sample_size = 50) 
  (h3 : last_sampled = 7894) :
  let segment_size := total / sample_size
  let last_segment_start := total - segment_size
  let samples := systematic_sample total sample_size last_sampled
  (last_segment_start = 7840 ∧ 
   samples.take 5 = [54, 214, 374, 534, 694]) :=
sorry

end NUMINAMATH_CALUDE_systematic_sampling_result_l4002_400281


namespace NUMINAMATH_CALUDE_negation_existential_l4002_400233

theorem negation_existential (f : ℝ → Prop) :
  (¬ ∃ x₀ > -1, x₀^2 + x₀ - 2018 > 0) ↔ (∀ x > -1, x^2 + x - 2018 ≤ 0) :=
sorry

end NUMINAMATH_CALUDE_negation_existential_l4002_400233


namespace NUMINAMATH_CALUDE_cooking_cleaning_arrangements_l4002_400280

theorem cooking_cleaning_arrangements (n : ℕ) (k : ℕ) (h1 : n = 5) (h2 : k = 3) :
  Nat.choose n k = 10 := by
  sorry

end NUMINAMATH_CALUDE_cooking_cleaning_arrangements_l4002_400280


namespace NUMINAMATH_CALUDE_factorization_proof_l4002_400206

theorem factorization_proof (a : ℝ) : 3 * a^2 - 27 = 3 * (a + 3) * (a - 3) := by
  sorry

end NUMINAMATH_CALUDE_factorization_proof_l4002_400206


namespace NUMINAMATH_CALUDE_share_ratio_l4002_400272

theorem share_ratio (total c b a : ℕ) (h1 : total = 406) (h2 : total = a + b + c) 
  (h3 : b = c / 2) (h4 : c = 232) : a / b = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_share_ratio_l4002_400272


namespace NUMINAMATH_CALUDE_probability_negative_product_l4002_400299

def dice_faces : Finset Int := {-3, -2, -1, 0, 1, 2}

def is_negative_product (x y : Int) : Bool :=
  x * y < 0

def count_negative_products : Nat :=
  (dice_faces.filter (λ x => x < 0)).card * (dice_faces.filter (λ x => x > 0)).card * 2

theorem probability_negative_product :
  (count_negative_products : ℚ) / (dice_faces.card * dice_faces.card) = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_probability_negative_product_l4002_400299


namespace NUMINAMATH_CALUDE_max_min_product_l4002_400258

theorem max_min_product (a b : ℕ+) (h : a + b = 100) :
  (∀ x y : ℕ+, x + y = 100 → x * y ≤ a * b) → a * b = 2500 ∧
  (∀ x y : ℕ+, x + y = 100 → a * b ≤ x * y) → a * b = 99 :=
by sorry

end NUMINAMATH_CALUDE_max_min_product_l4002_400258


namespace NUMINAMATH_CALUDE_isosceles_triangle_other_side_l4002_400257

structure IsoscelesTriangle where
  side1 : ℝ
  side2 : ℝ
  base : ℝ
  is_isosceles : side1 = side2
  perimeter : side1 + side2 + base = 15

def has_side_6 (t : IsoscelesTriangle) : Prop :=
  t.side1 = 6 ∨ t.side2 = 6 ∨ t.base = 6

theorem isosceles_triangle_other_side (t : IsoscelesTriangle) 
  (h : has_side_6 t) : 
  (t.side1 = 3 ∧ t.side2 = 3) ∨ (t.side1 = 4.5 ∧ t.side2 = 4.5) ∨ t.base = 3 := by
  sorry

end NUMINAMATH_CALUDE_isosceles_triangle_other_side_l4002_400257


namespace NUMINAMATH_CALUDE_intersection_of_specific_sets_l4002_400212

theorem intersection_of_specific_sets :
  let A : Set ℕ := {1, 2, 3}
  let B : Set ℕ := {3, 4, 5}
  A ∩ B = {3} := by
sorry

end NUMINAMATH_CALUDE_intersection_of_specific_sets_l4002_400212


namespace NUMINAMATH_CALUDE_triangle_side_sum_range_l4002_400293

/-- Given a triangle ABC with sides a, b, and c opposite to angles A, B, and C respectively,
    if b^2 + c^2 - a^2 = bc, AB · BC > 0, and a = √3/2,
    then √3/2 < b + c < 3/2. -/
theorem triangle_side_sum_range (a b c : ℝ) (A B C : ℝ) :
  0 < a ∧ 0 < b ∧ 0 < c →  -- positive side lengths
  0 < A ∧ 0 < B ∧ 0 < C →  -- positive angles
  A + B + C = π →  -- angles sum to π
  b^2 + c^2 - a^2 = b * c →  -- given condition
  (b * c * Real.cos A) > 0 →  -- AB · BC > 0
  a = Real.sqrt 3 / 2 →  -- given condition
  Real.sqrt 3 / 2 < b + c ∧ b + c < 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_side_sum_range_l4002_400293


namespace NUMINAMATH_CALUDE_circle_center_point_distance_l4002_400260

/-- The distance between the center of a circle and a point --/
theorem circle_center_point_distance (x y : ℝ) : 
  (x^2 + y^2 = 6*x - 2*y - 15) → 
  Real.sqrt ((3 - (-2))^2 + ((-1) - 5)^2) = Real.sqrt 61 := by
  sorry

end NUMINAMATH_CALUDE_circle_center_point_distance_l4002_400260


namespace NUMINAMATH_CALUDE_solution_value_l4002_400282

theorem solution_value (m : ℝ) : 
  (∃ x y : ℝ, m * x + 2 * y = 6 ∧ x = 1 ∧ y = 2) → m = 2 := by
  sorry

end NUMINAMATH_CALUDE_solution_value_l4002_400282


namespace NUMINAMATH_CALUDE_lines_properties_l4002_400220

-- Define the lines l₁ and l₂
def l₁ (a : ℝ) (x y : ℝ) : Prop := a * x - 3 * y + 1 = 0
def l₂ (b : ℝ) (x y : ℝ) : Prop := x - b * y + 2 = 0

-- Define parallelism for two lines
def parallel (f g : ℝ → ℝ → Prop) : Prop :=
  ∃ (k : ℝ), ∀ (x y : ℝ), f x y ↔ g (k * x) (k * y)

-- Define the first quadrant
def first_quadrant (x y : ℝ) : Prop := x > 0 ∧ y > 0

-- Theorem statement
theorem lines_properties (a b : ℝ) :
  (parallel (l₁ a) (l₂ b) → a * b = 3) ∧
  (b < 0 → ¬∃ (x y : ℝ), l₂ b x y ∧ first_quadrant x y) :=
sorry

end NUMINAMATH_CALUDE_lines_properties_l4002_400220


namespace NUMINAMATH_CALUDE_jason_pokemon_cards_l4002_400276

theorem jason_pokemon_cards (initial_cards given_away_cards : ℕ) :
  initial_cards = 9 →
  given_away_cards = 4 →
  initial_cards - given_away_cards = 5 :=
by sorry

end NUMINAMATH_CALUDE_jason_pokemon_cards_l4002_400276


namespace NUMINAMATH_CALUDE_union_of_sets_l4002_400222

theorem union_of_sets : 
  let M : Set ℕ := {0, 3}
  let N : Set ℕ := {1, 2, 3}
  M ∪ N = {0, 1, 2, 3} := by
sorry

end NUMINAMATH_CALUDE_union_of_sets_l4002_400222


namespace NUMINAMATH_CALUDE_odd_function_negative_domain_l4002_400268

-- Define an odd function f on ℝ
def isOddFunction (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

-- Define the condition for f when x ≥ 0
def fPositive (f : ℝ → ℝ) : Prop :=
  ∀ x, x ≥ 0 → f x = x * (1 + x)

-- Theorem statement
theorem odd_function_negative_domain
  (f : ℝ → ℝ) (odd : isOddFunction f) (pos : fPositive f) :
  ∀ x, x < 0 → f x = x * (1 - x) := by
  sorry

end NUMINAMATH_CALUDE_odd_function_negative_domain_l4002_400268


namespace NUMINAMATH_CALUDE_beta_function_integral_l4002_400239

theorem beta_function_integral (p q : ℕ) :
  ∫ x in (0:ℝ)..1, x^p * (1-x)^q = (p.factorial * q.factorial) / (p+q+1).factorial :=
sorry

end NUMINAMATH_CALUDE_beta_function_integral_l4002_400239


namespace NUMINAMATH_CALUDE_parallel_vectors_x_value_l4002_400209

/-- Two vectors are parallel if one is a scalar multiple of the other -/
def parallel (a b : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, a = (k * b.1, k * b.2)

theorem parallel_vectors_x_value :
  ∀ x : ℝ,
  let a : ℝ × ℝ := (x, 2)
  let b : ℝ × ℝ := (2, 1)
  parallel a b → x = 4 := by
sorry

end NUMINAMATH_CALUDE_parallel_vectors_x_value_l4002_400209


namespace NUMINAMATH_CALUDE_binary_110011_eq_51_l4002_400245

/-- Converts a list of binary digits to a decimal number -/
def binary_to_decimal (bits : List Bool) : ℕ :=
  bits.enum.foldl (fun acc (i, b) => acc + if b then 2^i else 0) 0

/-- The binary representation of 110011 -/
def binary_110011 : List Bool := [true, true, false, false, true, true]

/-- Theorem stating that 110011₂ is equal to 51 in decimal -/
theorem binary_110011_eq_51 : binary_to_decimal binary_110011 = 51 := by
  sorry

end NUMINAMATH_CALUDE_binary_110011_eq_51_l4002_400245


namespace NUMINAMATH_CALUDE_mismatching_socks_l4002_400226

theorem mismatching_socks (total_socks : ℕ) (matching_pairs : ℕ) 
  (h1 : total_socks = 25) (h2 : matching_pairs = 4) :
  total_socks - 2 * matching_pairs = 17 := by
  sorry

end NUMINAMATH_CALUDE_mismatching_socks_l4002_400226


namespace NUMINAMATH_CALUDE_irrational_among_given_numbers_l4002_400211

theorem irrational_among_given_numbers : 
  (¬ (∃ (a b : ℤ), (22 : ℚ) / 7 = a / b)) ∧ 
  (¬ (∃ (a b : ℤ), (0.303003 : ℚ) = a / b)) ∧ 
  (¬ (∃ (a b : ℤ), Real.sqrt 27 = a / b)) ∧ 
  (¬ (∃ (a b : ℤ), ((-64 : ℝ) ^ (1/3 : ℝ)) = a / b)) ↔ 
  (∃ (a b : ℤ), (22 : ℚ) / 7 = a / b) ∧ 
  (∃ (a b : ℤ), (0.303003 : ℚ) = a / b) ∧ 
  (¬ (∃ (a b : ℤ), Real.sqrt 27 = a / b)) ∧ 
  (∃ (a b : ℤ), ((-64 : ℝ) ^ (1/3 : ℝ)) = a / b) :=
sorry

end NUMINAMATH_CALUDE_irrational_among_given_numbers_l4002_400211
