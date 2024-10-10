import Mathlib

namespace simplify_logarithmic_expression_l4001_400123

-- Define the base-10 logarithm
noncomputable def lg (x : ℝ) : ℝ := Real.log x / Real.log 10

-- State the theorem
theorem simplify_logarithmic_expression :
  lg 5 * lg 20 - lg 2 * lg 50 - lg 25 = -lg 5 :=
by sorry

end simplify_logarithmic_expression_l4001_400123


namespace strawberry_price_difference_l4001_400148

theorem strawberry_price_difference (sale_price regular_price : ℚ) : 
  (54 * sale_price = 216) →
  (54 * regular_price = 216 + 108) →
  regular_price - sale_price = 2 := by
sorry

end strawberry_price_difference_l4001_400148


namespace sequences_not_periodic_l4001_400152

/-- Sequence A constructed by writing slices of increasing lengths from 1,0,0,0,... -/
def sequence_A : ℕ → ℕ := sorry

/-- Sequence B constructed by writing slices of two, four, six, etc., elements from 1,2,3,1,2,3,... -/
def sequence_B : ℕ → ℕ := sorry

/-- Sequence C formed by adding the corresponding elements of A and B -/
def sequence_C (n : ℕ) : ℕ := sequence_A n + sequence_B n

/-- A sequence is periodic if there exists a positive integer k such that
    for all n ≥ some fixed N, a(n+k) = a(n) -/
def is_periodic (a : ℕ → ℕ) : Prop :=
  ∃ (k : ℕ) (h : k > 0), ∃ (N : ℕ), ∀ (n : ℕ), n ≥ N → a (n + k) = a n

theorem sequences_not_periodic :
  ¬(is_periodic sequence_A) ∧ ¬(is_periodic sequence_B) ∧ ¬(is_periodic sequence_C) := by sorry

end sequences_not_periodic_l4001_400152


namespace problem_statement_l4001_400174

theorem problem_statement : 
  (∀ x : ℝ, x^2 - x + 1 > 0) ∨ ¬(∃ x : ℝ, x > 0 ∧ Real.sin x > 1) := by
  sorry

end problem_statement_l4001_400174


namespace andrew_bought_62_eggs_l4001_400103

/-- Represents the number of eggs Andrew has at different points -/
structure EggCount where
  initial : Nat
  final : Nat

/-- Calculates the number of eggs bought -/
def eggsBought (e : EggCount) : Nat :=
  e.final - e.initial

/-- Theorem stating that Andrew bought 62 eggs -/
theorem andrew_bought_62_eggs :
  let e : EggCount := { initial := 8, final := 70 }
  eggsBought e = 62 := by
  sorry

end andrew_bought_62_eggs_l4001_400103


namespace pen_distribution_l4001_400177

theorem pen_distribution (num_pencils : ℕ) (num_students : ℕ) (num_pens : ℕ) : 
  num_pencils = 828 →
  num_students = 4 →
  num_pencils % num_students = 0 →
  num_pens % num_students = 0 →
  ∃ k : ℕ, num_pens = 4 * k :=
by sorry

end pen_distribution_l4001_400177


namespace cube_collinear_groups_l4001_400157

/-- Represents a point in a cube structure -/
inductive CubePoint
  | Vertex
  | EdgeMidpoint
  | FaceCenter
  | Center

/-- Represents a group of three collinear points in the cube -/
structure CollinearGroup :=
  (points : Fin 3 → CubePoint)

/-- The cube structure with its points -/
structure Cube :=
  (vertices : Fin 8 → CubePoint)
  (edgeMidpoints : Fin 12 → CubePoint)
  (faceCenters : Fin 6 → CubePoint)
  (center : CubePoint)

/-- Function to count collinear groups in the cube -/
def countCollinearGroups (c : Cube) : Nat :=
  sorry

theorem cube_collinear_groups :
  ∀ c : Cube, countCollinearGroups c = 49 :=
sorry

end cube_collinear_groups_l4001_400157


namespace fraction_addition_and_simplification_l4001_400130

theorem fraction_addition_and_simplification :
  ∃ (n d : ℤ), (8 : ℚ) / 19 + (5 : ℚ) / 57 = n / d ∧ 
  n / d = (29 : ℚ) / 57 ∧
  (∀ k : ℤ, k ∣ n ∧ k ∣ d → k = 1 ∨ k = -1) :=
by sorry

end fraction_addition_and_simplification_l4001_400130


namespace orange_seller_loss_percentage_l4001_400134

theorem orange_seller_loss_percentage :
  ∀ (cost_price selling_price_10 selling_price_6 : ℚ),
    cost_price > 0 →
    selling_price_10 = 1 / 10 →
    selling_price_6 = 1 / 6 →
    selling_price_6 = 3/2 * cost_price →
    (cost_price - selling_price_10) / cost_price * 100 = 10/9 := by
  sorry

end orange_seller_loss_percentage_l4001_400134


namespace unique_f_exists_and_power_of_two_property_l4001_400109

def is_valid_f (f : ℕ+ → ℕ+) : Prop :=
  f 1 = 1 ∧ f 2 = 1 ∧ ∀ n ≥ 3, f n = f (f (n-1)) + f (n - f (n-1))

theorem unique_f_exists_and_power_of_two_property :
  ∃! f : ℕ+ → ℕ+, is_valid_f f ∧ ∀ m : ℕ, m ≥ 1 → f (2^m) = 2^(m-1) :=
sorry

end unique_f_exists_and_power_of_two_property_l4001_400109


namespace f_is_quadratic_l4001_400106

/-- Definition of a quadratic equation in one variable -/
def is_quadratic_equation (f : ℝ → ℝ) : Prop :=
  ∃ (a b c : ℝ), a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

/-- The function representing the equation x^2 - 2 = x -/
def f (x : ℝ) : ℝ := x^2 - x - 2

/-- Theorem stating that f is a quadratic equation -/
theorem f_is_quadratic : is_quadratic_equation f := by
  sorry

end f_is_quadratic_l4001_400106


namespace sum_product_equality_l4001_400179

theorem sum_product_equality (x y z : ℝ) 
  (hx : |x| ≠ 1/Real.sqrt 3) 
  (hy : |y| ≠ 1/Real.sqrt 3) 
  (hz : |z| ≠ 1/Real.sqrt 3) 
  (h : x + y + z = x * y * z) : 
  (3*x - x^3)/(1-3*x^2) + (3*y - y^3)/(1-3*y^2) + (3*z - z^3)/(1-3*z^2) = 
  (3*x - x^3)/(1-3*x^2) * (3*y - y^3)/(1-3*y^2) * (3*z - z^3)/(1-3*z^2) := by
  sorry

end sum_product_equality_l4001_400179


namespace badge_exchange_l4001_400140

theorem badge_exchange (x : ℝ) : 
  (x + 5) - (24/100) * (x + 5) + (20/100) * x = x - (20/100) * x + (24/100) * (x + 5) - 1 → 
  x = 45 := by
sorry

end badge_exchange_l4001_400140


namespace quadratic_inequality_min_value_l4001_400196

theorem quadratic_inequality_min_value (a b : ℝ) (h1 : a > b)
  (h2 : ∀ x : ℝ, a * x^2 + 2 * x + b ≥ 0)
  (h3 : ∃ x0 : ℝ, a * x0^2 + 2 * x0 + b = 0) :
  (∀ x : ℝ, (a^2 + b^2) / (a - b) ≥ 2 * Real.sqrt 2) ∧
  (∃ x : ℝ, (a^2 + b^2) / (a - b) = 2 * Real.sqrt 2) :=
by sorry

end quadratic_inequality_min_value_l4001_400196


namespace recipe_reduction_l4001_400119

/-- Represents a mixed number as a pair of integers (whole, fractional) -/
def MixedNumber := ℤ × ℚ

/-- Converts a mixed number to a rational number -/
def mixedToRational (m : MixedNumber) : ℚ :=
  m.1 + m.2

/-- The amount of flour in the original recipe -/
def originalFlour : MixedNumber := (5, 3/4)

/-- The amount of sugar in the original recipe -/
def originalSugar : MixedNumber := (2, 1/2)

/-- The fraction of the recipe we want to make -/
def recipeFraction : ℚ := 1/3

theorem recipe_reduction :
  (mixedToRational originalFlour * recipeFraction = 23/12) ∧
  (mixedToRational originalSugar * recipeFraction = 5/6) :=
sorry

end recipe_reduction_l4001_400119


namespace runt_pig_revenue_l4001_400146

/-- Calculates the revenue from selling bacon from a pig -/
def bacon_revenue (average_yield : ℝ) (price_per_pound : ℝ) (size_ratio : ℝ) : ℝ :=
  average_yield * size_ratio * price_per_pound

/-- Proves that the farmer will make $60 from the runt pig's bacon -/
theorem runt_pig_revenue :
  let average_yield : ℝ := 20
  let price_per_pound : ℝ := 6
  let size_ratio : ℝ := 0.5
  bacon_revenue average_yield price_per_pound size_ratio = 60 := by
sorry

end runt_pig_revenue_l4001_400146


namespace exists_silver_division_l4001_400160

/-- Represents the relationship between the number of people and the amount of silver in the problem. -/
def silver_division (x y : ℕ) : Prop :=
  (6 * x - 6 = y) ∧ (5 * x + 5 = y)

/-- The theorem states that the silver_division relationship holds for some positive integers x and y. -/
theorem exists_silver_division : ∃ (x y : ℕ), x > 0 ∧ y > 0 ∧ silver_division x y := by
  sorry

end exists_silver_division_l4001_400160


namespace unrolled_value_is_four_fifty_l4001_400143

/-- The number of quarters -/
def total_quarters : ℕ := 100

/-- The number of dimes -/
def total_dimes : ℕ := 185

/-- The capacity of a roll of quarters -/
def quarters_per_roll : ℕ := 45

/-- The capacity of a roll of dimes -/
def dimes_per_roll : ℕ := 55

/-- The value of a quarter in dollars -/
def quarter_value : ℚ := 1/4

/-- The value of a dime in dollars -/
def dime_value : ℚ := 1/10

/-- The total dollar value of coins that cannot be rolled -/
def unrolled_value : ℚ :=
  (total_quarters % quarters_per_roll) * quarter_value +
  (total_dimes % dimes_per_roll) * dime_value

theorem unrolled_value_is_four_fifty :
  unrolled_value = 9/2 := by sorry

end unrolled_value_is_four_fifty_l4001_400143


namespace sqrt_product_simplification_l4001_400153

theorem sqrt_product_simplification (x : ℝ) :
  Real.sqrt (96 * x^2) * Real.sqrt (50 * x) * Real.sqrt (28 * x^3) = 1260 * x^3 :=
by sorry

end sqrt_product_simplification_l4001_400153


namespace locus_of_Q_l4001_400151

-- Define the circle
def Circle (O : ℝ × ℝ) (r : ℝ) := {P : ℝ × ℝ | (P.1 - O.1)^2 + (P.2 - O.2)^2 = r^2}

-- Define the points and their properties
def SymmetricalPoints (O A B : ℝ × ℝ) := A.1 + B.1 = 2 * O.1 ∧ A.2 + B.2 = 2 * O.2

-- Define the perpendicular chord
def PerpendicularChord (P P' A : ℝ × ℝ) := 
  (P'.1 - P.1) * (A.1 - P.1) + (P'.2 - P.2) * (A.2 - P.2) = 0

-- Define the symmetric point C
def SymmetricPoint (B C PP' : ℝ × ℝ) := 
  (C.1 - PP'.1) = (PP'.1 - B.1) ∧ (C.2 - PP'.2) = (PP'.2 - B.2)

-- Define the intersection point Q
def IntersectionPoint (Q PP' A C : ℝ × ℝ) := 
  (Q.1 - PP'.1) * (C.2 - A.2) = (Q.2 - PP'.2) * (C.1 - A.1) ∧
  (Q.1 - A.1) * (C.2 - A.2) = (Q.2 - A.2) * (C.1 - A.1)

-- Theorem statement
theorem locus_of_Q (O : ℝ × ℝ) (r : ℝ) (A B P P' C Q : ℝ × ℝ) :
  P ∈ Circle O r →
  SymmetricalPoints O A B →
  PerpendicularChord P P' A →
  SymmetricPoint B C P →
  IntersectionPoint Q P A C →
  ∃ (a b : ℝ), 
    (Q.1 / a)^2 + (Q.2 / b)^2 = 1 ∧
    a^2 - b^2 = (A.1 - B.1)^2 + (A.2 - B.2)^2 ∧
    a = r :=
by sorry

end locus_of_Q_l4001_400151


namespace decimal_to_fraction_l4001_400110

theorem decimal_to_fraction :
  (0.36 : ℚ) = 9 / 25 := by sorry

end decimal_to_fraction_l4001_400110


namespace intersection_range_intersection_length_l4001_400126

-- Define the hyperbola and line
def hyperbola (x y : ℝ) : Prop := x^2 - y^2 = 1
def line (k : ℝ) (x y : ℝ) : Prop := y = k * x + 1

-- Define the intersection condition
def intersects_at_two_points (k : ℝ) : Prop :=
  ∃ x₁ y₁ x₂ y₂ : ℝ, x₁ ≠ x₂ ∧ hyperbola x₁ y₁ ∧ hyperbola x₂ y₂ ∧ line k x₁ y₁ ∧ line k x₂ y₂

-- Define the range of k
def k_range (k : ℝ) : Prop :=
  (k > -Real.sqrt 2 ∧ k < -1) ∨ (k > -1 ∧ k < 1) ∨ (k > 1 ∧ k < Real.sqrt 2)

-- Define the midpoint condition
def midpoint_condition (k : ℝ) : Prop :=
  ∃ x₁ y₁ x₂ y₂ : ℝ, hyperbola x₁ y₁ ∧ hyperbola x₂ y₂ ∧ line k x₁ y₁ ∧ line k x₂ y₂ ∧
    (x₁ + x₂) / 2 = Real.sqrt 2

-- Theorem 1: Range of k for two distinct intersections
theorem intersection_range :
  ∀ k : ℝ, intersects_at_two_points k ↔ k_range k := by sorry

-- Theorem 2: Length of AB when midpoint x-coordinate is √2
theorem intersection_length :
  ∀ k : ℝ, midpoint_condition k → 
    ∃ x₁ y₁ x₂ y₂ : ℝ, hyperbola x₁ y₁ ∧ hyperbola x₂ y₂ ∧ line k x₁ y₁ ∧ line k x₂ y₂ ∧
      Real.sqrt ((x₂ - x₁)^2 + (y₂ - y₁)^2) = 6 := by sorry

end intersection_range_intersection_length_l4001_400126


namespace solution_set_inequality_l4001_400117

theorem solution_set_inequality (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  {x : ℝ | -b < 1/x ∧ 1/x < a} = {x : ℝ | x < -1/b ∨ x > 1/a} := by sorry

end solution_set_inequality_l4001_400117


namespace mary_performance_l4001_400104

theorem mary_performance (total_days : ℕ) (adequate_rate : ℕ) (outstanding_rate : ℕ) (total_amount : ℕ) :
  total_days = 15 ∧ 
  adequate_rate = 4 ∧ 
  outstanding_rate = 7 ∧ 
  total_amount = 85 →
  ∃ (adequate_days outstanding_days : ℕ),
    adequate_days + outstanding_days = total_days ∧
    adequate_days * adequate_rate + outstanding_days * outstanding_rate = total_amount ∧
    outstanding_days = 8 := by
  sorry

end mary_performance_l4001_400104


namespace apples_given_to_larry_l4001_400125

/-- Given that Joyce starts with 75 apples and ends up with 23 apples,
    prove that she gave 52 apples to Larry. -/
theorem apples_given_to_larry (initial : ℕ) (final : ℕ) (given : ℕ) :
  initial = 75 →
  final = 23 →
  given = initial - final →
  given = 52 := by sorry

end apples_given_to_larry_l4001_400125


namespace vector_simplification_l4001_400105

variable {V : Type*} [AddCommGroup V]

theorem vector_simplification 
  (A B C D : V) : 
  ((B - A) - (D - C)) - ((C - A) - (D - B)) = (0 : V) := by
  sorry

end vector_simplification_l4001_400105


namespace cycle_price_calculation_l4001_400131

/-- Proves that given a cycle sold at a loss of 18% with a selling price of Rs. 1558, the original price of the cycle is Rs. 1900. -/
theorem cycle_price_calculation (selling_price : ℝ) (loss_percentage : ℝ) 
  (h1 : selling_price = 1558)
  (h2 : loss_percentage = 18) : 
  ∃ (original_price : ℝ), 
    original_price = 1900 ∧ 
    selling_price = original_price * (1 - loss_percentage / 100) := by
  sorry

end cycle_price_calculation_l4001_400131


namespace ages_sum_l4001_400168

theorem ages_sum (kiana_age twin_age : ℕ) : 
  kiana_age > twin_age →
  kiana_age * twin_age * twin_age = 72 →
  kiana_age + twin_age + twin_age = 14 :=
by sorry

end ages_sum_l4001_400168


namespace jude_current_age_l4001_400173

/-- Heath's age today -/
def heath_age_today : ℕ := 16

/-- Heath's age in 5 years -/
def heath_age_future : ℕ := heath_age_today + 5

/-- Jude's age in 5 years -/
def jude_age_future : ℕ := heath_age_future / 3

/-- Jude's age today -/
def jude_age_today : ℕ := jude_age_future - 5

/-- Theorem stating Jude's age today -/
theorem jude_current_age : jude_age_today = 2 := by
  sorry

end jude_current_age_l4001_400173


namespace spatial_diagonals_count_l4001_400182

/-- A convex polyhedron with specified properties -/
structure ConvexPolyhedron where
  vertices : ℕ
  edges : ℕ
  faces : ℕ
  triangular_faces : ℕ
  quadrilateral_faces : ℕ

/-- Calculate the number of spatial diagonals in a convex polyhedron -/
def spatial_diagonals (P : ConvexPolyhedron) : ℕ :=
  Nat.choose P.vertices 2 - P.edges - 2 * P.quadrilateral_faces

/-- Theorem stating the number of spatial diagonals in the given polyhedron -/
theorem spatial_diagonals_count (P : ConvexPolyhedron) 
  (h1 : P.vertices = 26)
  (h2 : P.edges = 60)
  (h3 : P.faces = 36)
  (h4 : P.triangular_faces = 24)
  (h5 : P.quadrilateral_faces = 12)
  (h6 : P.triangular_faces + P.quadrilateral_faces = P.faces) :
  spatial_diagonals P = 241 := by
  sorry

#eval spatial_diagonals ⟨26, 60, 36, 24, 12⟩

end spatial_diagonals_count_l4001_400182


namespace ratio_problem_l4001_400188

theorem ratio_problem (w x y z : ℚ) 
  (h1 : w / x = 4 / 3)
  (h2 : y / z = 3 / 2)
  (h3 : z / x = 1 / 6) :
  w / y = 16 / 3 := by
  sorry

end ratio_problem_l4001_400188


namespace function_minimum_and_inequality_l4001_400158

-- Define the function f
def f (a b x : ℝ) : ℝ := |x + a| + |2*x - b|

-- State the theorem
theorem function_minimum_and_inequality (a b : ℝ) 
  (ha : a > 0) (hb : b > 0) 
  (hmin : ∀ x, f a b x ≥ 1) 
  (hequal : ∃ x, f a b x = 1) : 
  (2*a + b = 2) ∧ 
  (∀ t : ℝ, a + 2*b ≥ t*a*b → t ≤ 9/2) ∧
  (∃ t : ℝ, t = 9/2 ∧ a + 2*b = t*a*b) :=
by sorry

end function_minimum_and_inequality_l4001_400158


namespace not_both_bidirectional_l4001_400118

-- Define the proof methods
inductive ProofMethod
| Synthetic
| Analytic

-- Define the reasoning directions
inductive ReasoningDirection
| CauseToEffect
| EffectToCause

-- Define the properties of the proof methods
def methodProperties (m : ProofMethod) : ReasoningDirection :=
  match m with
  | ProofMethod.Synthetic => ReasoningDirection.CauseToEffect
  | ProofMethod.Analytic => ReasoningDirection.EffectToCause

-- Theorem statement
theorem not_both_bidirectional : 
  ¬ (∀ (m : ProofMethod), 
      methodProperties m = ReasoningDirection.CauseToEffect ∧ 
      methodProperties m = ReasoningDirection.EffectToCause) :=
by sorry

end not_both_bidirectional_l4001_400118


namespace y_divisibility_l4001_400166

def y : ℕ := 96 + 144 + 200 + 320 + 480 + 512 + 4096

theorem y_divisibility :
  (∃ k : ℕ, y = 4 * k) ∧
  (∃ k : ℕ, y = 8 * k) ∧
  (∃ k : ℕ, y = 16 * k) ∧
  ¬(∃ k : ℕ, y = 32 * k) := by
  sorry

end y_divisibility_l4001_400166


namespace last_three_digits_of_3_to_8000_l4001_400135

theorem last_three_digits_of_3_to_8000 (h : 3^400 ≡ 1 [ZMOD 800]) :
  3^8000 ≡ 1 [ZMOD 1000] := by sorry

end last_three_digits_of_3_to_8000_l4001_400135


namespace sum_of_digit_differences_eq_495_l4001_400170

/-- The sum of the differences between the first and last digits of all natural numbers from 1 to 999 -/
def sum_of_digit_differences : ℕ :=
  (List.range 999).foldl (λ sum n =>
    let first_digit := (n + 1) / 100
    let last_digit := (n + 1) % 10
    sum + (first_digit - last_digit)) 0

/-- Theorem stating that the sum of the differences between the first and last digits
    of all natural numbers from 1 to 999 is equal to 495 -/
theorem sum_of_digit_differences_eq_495 :
  sum_of_digit_differences = 495 := by sorry

end sum_of_digit_differences_eq_495_l4001_400170


namespace train_length_calculation_l4001_400189

/-- Calculates the length of two trains given their speeds and overtaking time -/
theorem train_length_calculation (v_fast v_slow : ℝ) (t : ℝ) (h1 : v_fast = 46) (h2 : v_slow = 36) (h3 : t = 72) :
  let v_rel := v_fast - v_slow
  let d := v_rel * t * (1000 / 3600)
  let L := d / 2
  L = 100 := by sorry

end train_length_calculation_l4001_400189


namespace tangent_circle_radius_l4001_400193

/-- A circle tangent to coordinate axes and the hypotenuse of a 45-45-90 triangle --/
structure TangentCircle where
  O : ℝ × ℝ  -- Center of the circle
  r : ℝ      -- Radius of the circle
  h : r > 0  -- Radius is positive

/-- A 45-45-90 triangle with side length 2 --/
structure RightIsoscelesTriangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  h1 : B.1 - A.1 = 2  -- AB has length 2
  h2 : C.2 - A.2 = 2  -- AC has length 2 in y-direction
  h3 : B.2 = A.2      -- AB is horizontal

/-- The main theorem --/
theorem tangent_circle_radius
  (t : TangentCircle)
  (tri : RightIsoscelesTriangle)
  (h_tangent_x : t.O.2 = t.r)
  (h_tangent_y : t.O.1 = t.r)
  (h_tangent_hyp : t.O.2 + t.r = tri.C.2) :
  t.r = 2 := by
  sorry

end tangent_circle_radius_l4001_400193


namespace b_geq_0_sufficient_not_necessary_for_a_squared_plus_b_geq_0_l4001_400183

theorem b_geq_0_sufficient_not_necessary_for_a_squared_plus_b_geq_0 :
  ∃ (a b : ℝ), (b ≥ 0 → a^2 + b ≥ 0) ∧ ¬(a^2 + b ≥ 0 → b ≥ 0) :=
by sorry

end b_geq_0_sufficient_not_necessary_for_a_squared_plus_b_geq_0_l4001_400183


namespace daves_total_expense_l4001_400137

/-- The amount Dave spent on books -/
def daves_book_expense (animal_books outer_space_books train_books book_price : ℕ) : ℕ :=
  (animal_books + outer_space_books + train_books) * book_price

/-- Theorem stating the total amount Dave spent on books -/
theorem daves_total_expense : 
  daves_book_expense 8 6 3 6 = 102 := by
  sorry

end daves_total_expense_l4001_400137


namespace base_3_of_121_l4001_400161

def base_3_representation (n : ℕ) : List ℕ :=
  sorry

theorem base_3_of_121 :
  base_3_representation 121 = [1, 1, 1, 1, 1] :=
sorry

end base_3_of_121_l4001_400161


namespace calculate_rates_l4001_400113

/-- Represents the rates and quantities in the problem -/
structure Rates where
  p : ℕ  -- number of pears
  b : ℕ  -- number of bananas
  d : ℕ  -- number of dishes
  tp : ℕ -- time spent picking pears (in hours)
  tb : ℕ -- time spent cooking bananas (in hours)
  tw : ℕ -- time spent washing dishes (in hours)
  rp : ℚ -- rate of picking pears (pears per hour)
  rb : ℚ -- rate of cooking bananas (bananas per hour)
  rw : ℚ -- rate of washing dishes (dishes per hour)

/-- The main theorem that proves the rates given the conditions -/
theorem calculate_rates (r : Rates) 
    (h1 : r.d = r.b + 10)
    (h2 : r.b = 3 * r.p)
    (h3 : r.p = 50)
    (h4 : r.tp = 4)
    (h5 : r.tb = 2)
    (h6 : r.tw = 5)
    : r.rp = 25/2 ∧ r.rb = 75 ∧ r.rw = 32 := by
  sorry


end calculate_rates_l4001_400113


namespace vector_equation_solution_l4001_400107

/-- Given vectors e₁ and e₂, and real numbers x and y satisfying the equation,
    prove that x - y = -3 -/
theorem vector_equation_solution (e₁ e₂ : ℝ × ℝ) (x y : ℝ) 
    (h₁ : e₁ = (1, 2))
    (h₂ : e₂ = (3, 4))
    (h₃ : x • e₁ + y • e₂ = (5, 6)) :
  x - y = -3 := by
  sorry

end vector_equation_solution_l4001_400107


namespace wheel_spinner_probability_wheel_spinner_probability_proof_l4001_400195

theorem wheel_spinner_probability : Real → Real → Real → Real → Prop :=
  fun prob_E prob_F prob_G prob_H =>
    prob_E = 1/2 →
    prob_F = 1/4 →
    prob_G = 2 * prob_H →
    prob_E + prob_F + prob_G + prob_H = 1 →
    prob_G = 1/6

-- The proof is omitted
theorem wheel_spinner_probability_proof : wheel_spinner_probability (1/2) (1/4) (1/6) (1/12) := by
  sorry

end wheel_spinner_probability_wheel_spinner_probability_proof_l4001_400195


namespace all_reals_satisfy_property_l4001_400144

theorem all_reals_satisfy_property :
  ∀ (α : ℝ), ∀ (n : ℕ), n > 0 → ∃ (m : ℤ), |α - (m : ℝ) / n| < 1 / (3 * n) :=
by sorry

end all_reals_satisfy_property_l4001_400144


namespace linda_furniture_fraction_l4001_400159

/-- Proves that the fraction of Linda's savings spent on furniture is 3/5 -/
theorem linda_furniture_fraction (original_savings : ℚ) (tv_cost : ℚ) : 
  original_savings = 1000 →
  tv_cost = 400 →
  (original_savings - tv_cost) / original_savings = 3/5 := by
sorry

end linda_furniture_fraction_l4001_400159


namespace sqrt_5184_div_18_eq_4_l4001_400141

theorem sqrt_5184_div_18_eq_4 : Real.sqrt 5184 / 18 = 4 := by
  sorry

end sqrt_5184_div_18_eq_4_l4001_400141


namespace false_statement_l4001_400142

-- Define the types for planes and lines
variable {α β : Plane} {m n : Line}

-- Define the relations
def perpendicular (l : Line) (p : Plane) : Prop := sorry
def parallel (p q : Plane) : Prop := sorry
def parallel_line_plane (l : Line) (p : Plane) : Prop := sorry
def parallel_lines (l1 l2 : Line) : Prop := sorry
def contained_in (l : Line) (p : Plane) : Prop := sorry

-- State the theorem
theorem false_statement :
  ¬(∀ (α β : Plane) (m n : Line),
    (¬parallel_line_plane m α ∧ parallel α β ∧ contained_in n β) →
    parallel_lines m n) :=
sorry

end false_statement_l4001_400142


namespace existence_of_odd_fifth_powers_sum_l4001_400198

theorem existence_of_odd_fifth_powers_sum (m : ℤ) :
  ∃ (a b : ℤ) (k : ℕ+), 
    Odd a ∧ Odd b ∧ (2 * m = a^5 + b^5 + k * 2^100) := by
  sorry

end existence_of_odd_fifth_powers_sum_l4001_400198


namespace original_number_proof_l4001_400139

theorem original_number_proof (x : ℝ) (h1 : x > 0) (h2 : 1000 * x = 3 * (1 / x)) :
  x = Real.sqrt 30 / 100 := by
  sorry

end original_number_proof_l4001_400139


namespace monica_savings_l4001_400121

/-- Calculates the total amount saved given the weekly savings, number of weeks, and number of repetitions. -/
def total_savings (weekly_savings : ℕ) (weeks : ℕ) (repetitions : ℕ) : ℕ :=
  weekly_savings * weeks * repetitions

/-- Proves that saving $15 per week for 60 weeks, repeated 5 times, results in a total savings of $4500. -/
theorem monica_savings : total_savings 15 60 5 = 4500 := by
  sorry

end monica_savings_l4001_400121


namespace xiao_wang_processes_60_parts_l4001_400145

/-- Represents the number of parts processed by a worker in a given time period -/
def ProcessedParts (rate : ℕ) (workTime : ℕ) : ℕ := rate * workTime

/-- Represents the total time taken by Xiao Wang to process a given number of parts -/
def XiaoWangTotalTime (parts : ℕ) : ℚ :=
  let workHours := parts / 15
  let breaks := workHours / 2
  (workHours + breaks : ℚ)

/-- Represents the total time taken by Xiao Li to process a given number of parts -/
def XiaoLiTotalTime (parts : ℕ) : ℚ := parts / 12

/-- Theorem stating that Xiao Wang processes 60 parts when both finish at the same time -/
theorem xiao_wang_processes_60_parts :
  ∃ (parts : ℕ), parts = 60 ∧ XiaoWangTotalTime parts = XiaoLiTotalTime parts :=
sorry

end xiao_wang_processes_60_parts_l4001_400145


namespace tank_capacity_comparison_l4001_400163

theorem tank_capacity_comparison :
  let tank_a_height : ℝ := 10
  let tank_a_circumference : ℝ := 7
  let tank_b_height : ℝ := 7
  let tank_b_circumference : ℝ := 10
  let tank_a_volume := π * (tank_a_circumference / (2 * π))^2 * tank_a_height
  let tank_b_volume := π * (tank_b_circumference / (2 * π))^2 * tank_b_height
  (tank_a_volume / tank_b_volume) * 100 = 70 :=
by
  sorry

end tank_capacity_comparison_l4001_400163


namespace correlation_coefficient_comparison_l4001_400116

def X : List Float := [10, 11.3, 11.8, 12.5, 13]
def Y : List Float := [1, 2, 3, 4, 5]
def U : List Float := [10, 11.3, 11.8, 12.5, 13]
def V : List Float := [5, 4, 3, 2, 1]

def linear_correlation_coefficient (x : List Float) (y : List Float) : Float :=
  sorry

def r₁ : Float := linear_correlation_coefficient X Y
def r₂ : Float := linear_correlation_coefficient U V

theorem correlation_coefficient_comparison : r₂ < r₁ := by
  sorry

end correlation_coefficient_comparison_l4001_400116


namespace additional_week_cost_is_eleven_l4001_400184

/-- The cost per day for additional weeks in a student youth hostel -/
def additional_week_cost (first_week_daily_rate : ℚ) (total_days : ℕ) (total_cost : ℚ) : ℚ :=
  let first_week_cost := 7 * first_week_daily_rate
  let additional_days := total_days - 7
  let additional_cost := total_cost - first_week_cost
  additional_cost / additional_days

theorem additional_week_cost_is_eleven :
  additional_week_cost 18 23 302 = 11 := by
sorry

end additional_week_cost_is_eleven_l4001_400184


namespace prism_volume_in_cubic_yards_l4001_400199

/-- Conversion factor from cubic feet to cubic yards -/
def cubic_feet_per_cubic_yard : ℝ := 27

/-- Volume of the rectangular prism in cubic feet -/
def prism_volume_cubic_feet : ℝ := 216

/-- Theorem stating that the volume of the prism in cubic yards is 8 -/
theorem prism_volume_in_cubic_yards :
  prism_volume_cubic_feet / cubic_feet_per_cubic_yard = 8 := by
  sorry

end prism_volume_in_cubic_yards_l4001_400199


namespace total_musicians_is_98_l4001_400187

/-- The total number of musicians in the orchestra, band, and choir -/
def total_musicians (orchestra_males orchestra_females band_multiplier choir_males choir_females : ℕ) : ℕ :=
  let orchestra_total := orchestra_males + orchestra_females
  let band_total := band_multiplier * orchestra_total
  let choir_total := choir_males + choir_females
  orchestra_total + band_total + choir_total

/-- Theorem stating that the total number of musicians is 98 given the specific conditions -/
theorem total_musicians_is_98 :
  total_musicians 11 12 2 12 17 = 98 := by
  sorry

end total_musicians_is_98_l4001_400187


namespace joe_video_game_spending_l4001_400190

/-- Joe's video game spending problem -/
theorem joe_video_game_spending
  (initial_money : ℕ)
  (selling_price : ℕ)
  (months : ℕ)
  (h1 : initial_money = 240)
  (h2 : selling_price = 30)
  (h3 : months = 12)
  : ∃ (monthly_spending : ℕ),
    monthly_spending = 50 ∧
    initial_money = months * monthly_spending - months * selling_price :=
by sorry

end joe_video_game_spending_l4001_400190


namespace handshakes_count_l4001_400156

/-- Represents the social event with given conditions -/
structure SocialEvent where
  total_people : ℕ
  group_a_size : ℕ
  group_b_size : ℕ
  group_a_knows_all : group_a_size = 25
  group_b_knows_one : group_b_size = 15
  total_is_sum : total_people = group_a_size + group_b_size

/-- Calculates the number of handshakes in the social event -/
def count_handshakes (event : SocialEvent) : ℕ :=
  let group_b_internal_handshakes := (event.group_b_size * (event.group_b_size - 1)) / 2
  let group_a_b_handshakes := event.group_b_size * (event.group_a_size - 1)
  group_b_internal_handshakes + group_a_b_handshakes

/-- Theorem stating that the number of handshakes in the given social event is 465 -/
theorem handshakes_count (event : SocialEvent) : count_handshakes event = 465 := by
  sorry

end handshakes_count_l4001_400156


namespace michelangelo_painting_l4001_400108

theorem michelangelo_painting (total : ℕ) (left : ℕ) (this_week : ℕ) : 
  total = 28 → 
  left = 13 → 
  total - left = this_week + this_week / 4 →
  this_week = 12 := by
  sorry

end michelangelo_painting_l4001_400108


namespace all_N_composite_l4001_400192

def N (n : ℕ) : ℕ := 200 * 10^n + 88 * ((10^n - 1) / 9) + 21

theorem all_N_composite (n : ℕ) : ∃ (a b : ℕ), a > 1 ∧ b > 1 ∧ N n = a * b := by
  sorry

end all_N_composite_l4001_400192


namespace sinusoidal_function_properties_l4001_400115

/-- Given a sinusoidal function y = a * sin(b * x + c) with a > 0 and b > 0,
    if the maximum occurs at x = π/6 and the amplitude is 3,
    then a = 3 and c = (3 - b) * π/6 -/
theorem sinusoidal_function_properties (a b c : ℝ) (h1 : a > 0) (h2 : b > 0)
  (h3 : ∀ x, a * Real.sin (b * x + c) ≤ a * Real.sin (b * (π/6) + c))
  (h4 : a = 3) :
  a = 3 ∧ c = (3 - b) * π/6 := by
sorry

end sinusoidal_function_properties_l4001_400115


namespace cos_75_degrees_l4001_400114

theorem cos_75_degrees :
  Real.cos (75 * π / 180) = (Real.sqrt 6 - Real.sqrt 2) / 4 := by
  sorry

end cos_75_degrees_l4001_400114


namespace simplify_radical_product_l4001_400132

theorem simplify_radical_product (x : ℝ) (h : x > 0) :
  Real.sqrt (48 * x) * Real.sqrt (3 * x) * (81 * x^2)^(1/3) = 36 * x * (3 * x^2)^(1/3) := by
  sorry

end simplify_radical_product_l4001_400132


namespace inequality_solution_l4001_400129

theorem inequality_solution (x : ℝ) : 1 - 1 / (3 * x + 4) < 3 ↔ x < -5/3 ∨ x > -4/3 := by
  sorry

end inequality_solution_l4001_400129


namespace boat_speed_in_still_water_l4001_400100

-- Define the speed of the stream
def stream_speed : ℝ := 5

-- Define the distance traveled downstream
def downstream_distance : ℝ := 81

-- Define the time taken to travel downstream
def downstream_time : ℝ := 3

-- Define the speed of the boat in still water
def boat_speed : ℝ := 22

-- Theorem statement
theorem boat_speed_in_still_water :
  boat_speed = downstream_distance / downstream_time - stream_speed :=
by
  sorry

end boat_speed_in_still_water_l4001_400100


namespace original_number_l4001_400133

theorem original_number (x : ℝ) : 3 * (2 * x + 5) = 111 → x = 16 := by
  sorry

end original_number_l4001_400133


namespace rectangular_prism_volume_l4001_400136

theorem rectangular_prism_volume
  (face_area1 face_area2 face_area3 : ℝ)
  (h1 : face_area1 = 15)
  (h2 : face_area2 = 20)
  (h3 : face_area3 = 30)
  (h4 : ∃ l w h : ℝ, l * w = face_area1 ∧ w * h = face_area2 ∧ l * h = face_area3) :
  ∃ volume : ℝ, volume = 30 * Real.sqrt 10 ∧
    (∀ l w h : ℝ, l * w = face_area1 → w * h = face_area2 → l * h = face_area3 →
      volume = l * w * h) :=
by sorry

end rectangular_prism_volume_l4001_400136


namespace spencer_jump_rope_session_length_l4001_400164

/-- Proves that Spencer's jump rope session length is 10 minutes -/
theorem spencer_jump_rope_session_length :
  ∀ (jumps_per_minute : ℕ) 
    (sessions_per_day : ℕ) 
    (total_jumps : ℕ) 
    (total_days : ℕ),
  jumps_per_minute = 4 →
  sessions_per_day = 2 →
  total_jumps = 400 →
  total_days = 5 →
  (total_jumps / total_days / sessions_per_day) / jumps_per_minute = 10 :=
by
  sorry

#check spencer_jump_rope_session_length

end spencer_jump_rope_session_length_l4001_400164


namespace fourth_grade_students_l4001_400171

theorem fourth_grade_students (initial_students leaving_students new_students : ℕ) :
  initial_students = 35 →
  leaving_students = 10 →
  new_students = 10 →
  initial_students - leaving_students + new_students = 35 := by
sorry

end fourth_grade_students_l4001_400171


namespace triangle_tan_A_l4001_400180

theorem triangle_tan_A (A B C : ℝ) (AB BC : ℝ) 
  (h_angle : A = π/3)
  (h_AB : AB = 20)
  (h_BC : BC = 21) : 
  Real.tan A = (21 * Real.sqrt 3) / (2 * Real.sqrt (421 - 1323/4)) := by
  sorry

end triangle_tan_A_l4001_400180


namespace equal_distribution_of_drawings_l4001_400154

theorem equal_distribution_of_drawings (total_drawings : ℕ) (num_neighbors : ℕ) 
  (h1 : total_drawings = 54) (h2 : num_neighbors = 6) :
  total_drawings / num_neighbors = 9 := by
  sorry

end equal_distribution_of_drawings_l4001_400154


namespace shirt_price_change_l4001_400175

theorem shirt_price_change (P : ℝ) (P_pos : P > 0) :
  P * (1 + 0.15) * (1 - 0.15) = P * 0.9775 := by
  sorry

#check shirt_price_change

end shirt_price_change_l4001_400175


namespace winter_olympics_volunteer_allocation_l4001_400112

theorem winter_olympics_volunteer_allocation :
  let n_volunteers : ℕ := 5
  let n_projects : ℕ := 4
  let allocation_schemes : ℕ := (n_volunteers.choose 2) * n_projects.factorial
  allocation_schemes = 240 :=
by sorry

end winter_olympics_volunteer_allocation_l4001_400112


namespace reciprocal_root_property_l4001_400181

theorem reciprocal_root_property (c : ℝ) : 
  c^3 - c + 1 = 0 → (1/c)^5 + (1/c) + 1 = 0 := by
sorry

end reciprocal_root_property_l4001_400181


namespace sqrt_product_equals_six_l4001_400197

theorem sqrt_product_equals_six : Real.sqrt 8 * Real.sqrt (9/2) = 6 := by
  sorry

end sqrt_product_equals_six_l4001_400197


namespace nonagon_diagonal_intersection_probability_l4001_400167

/-- A regular nonagon is a 9-sided regular polygon -/
def RegularNonagon : Type := Unit

/-- The number of vertices in a regular nonagon -/
def num_vertices : ℕ := 9

/-- The number of diagonals in a regular nonagon -/
def num_diagonals (n : RegularNonagon) : ℕ := (num_vertices * (num_vertices - 3)) / 2

/-- The number of pairs of intersecting diagonals in a regular nonagon -/
def num_intersecting_diagonals (n : RegularNonagon) : ℕ := Nat.choose num_vertices 4

/-- The total number of pairs of diagonals in a regular nonagon -/
def total_diagonal_pairs (n : RegularNonagon) : ℕ := Nat.choose (num_diagonals n) 2

/-- The probability that two randomly chosen diagonals intersect inside the nonagon -/
def intersection_probability (n : RegularNonagon) : ℚ :=
  (num_intersecting_diagonals n : ℚ) / (total_diagonal_pairs n : ℚ)

theorem nonagon_diagonal_intersection_probability (n : RegularNonagon) :
  intersection_probability n = 14 / 39 := by
  sorry

end nonagon_diagonal_intersection_probability_l4001_400167


namespace zoo_trip_money_left_l4001_400128

/-- The amount of money left for lunch and snacks after a zoo trip -/
def money_left_for_lunch_and_snacks (
  zoo_ticket_price : ℚ)
  (bus_fare_one_way : ℚ)
  (total_money : ℚ)
  (num_people : ℕ) : ℚ :=
  total_money - (num_people * zoo_ticket_price + 2 * num_people * bus_fare_one_way)

/-- Theorem: Noah and Ava have $24 left for lunch and snacks after their zoo trip -/
theorem zoo_trip_money_left :
  money_left_for_lunch_and_snacks 5 (3/2) 40 2 = 24 := by
  sorry

end zoo_trip_money_left_l4001_400128


namespace base_to_lateral_area_ratio_l4001_400172

/-- A cone whose lateral surface unfolds into a semicircle -/
structure SemicircleCone where
  r : ℝ  -- radius of the base
  l : ℝ  -- slant height
  h : 2 * π * r = π * l  -- condition for unfolding into a semicircle

theorem base_to_lateral_area_ratio (cone : SemicircleCone) :
  (π * cone.r^2) / ((1/2) * π * cone.l^2) = 1/2 := by
  sorry

end base_to_lateral_area_ratio_l4001_400172


namespace spade_example_l4001_400124

-- Define the spade operation
def spade (a b : ℝ) : ℝ := |a - b|

-- Theorem statement
theorem spade_example : spade 3 (spade 5 8) = 0 := by
  sorry

end spade_example_l4001_400124


namespace factorial_base_700_a4_l4001_400185

/-- Factorial function -/
def factorial (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

/-- Coefficient in factorial base representation -/
def factorial_base_coeff (n k : ℕ) : ℕ :=
  (n / factorial k) % (k + 1)

/-- Theorem: The coefficient a₄ in the factorial base representation of 700 is 4 -/
theorem factorial_base_700_a4 : factorial_base_coeff 700 4 = 4 := by
  sorry

end factorial_base_700_a4_l4001_400185


namespace enrique_commission_l4001_400176

/-- Calculates the commission earned by Enrique based on his sales --/
theorem enrique_commission :
  let commission_rate : ℚ := 15 / 100
  let suit_price : ℚ := 700
  let suit_quantity : ℕ := 2
  let shirt_price : ℚ := 50
  let shirt_quantity : ℕ := 6
  let loafer_price : ℚ := 150
  let loafer_quantity : ℕ := 2
  let total_sales : ℚ := suit_price * suit_quantity + shirt_price * shirt_quantity + loafer_price * loafer_quantity
  let commission : ℚ := commission_rate * total_sales
  commission = 300
  := by sorry

end enrique_commission_l4001_400176


namespace median_mode_difference_l4001_400120

def data : List ℕ := [42, 44, 44, 45, 45, 45, 51, 51, 51, 53, 53, 53, 62, 64, 66, 66, 67, 68, 70, 74, 74, 75, 75, 76, 81, 82, 85, 88, 89, 89]

def mode (l : List ℕ) : ℕ := sorry

def median (l : List ℕ) : ℚ := sorry

theorem median_mode_difference : 
  |median data - (mode data : ℚ)| = 23 := by sorry

end median_mode_difference_l4001_400120


namespace sufficient_not_necessary_l4001_400186

def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f x = f (-x)

def phi_condition (φ : ℝ) : Prop :=
  ∃ k : ℤ, φ = 2 * k * Real.pi + Real.pi / 2

theorem sufficient_not_necessary :
  (∀ φ : ℝ, phi_condition φ → is_even_function (λ x => Real.sin (x + φ))) ∧
  (∃ φ : ℝ, is_even_function (λ x => Real.sin (x + φ)) ∧ ¬phi_condition φ) := by
  sorry

end sufficient_not_necessary_l4001_400186


namespace cube_volume_from_surface_area_l4001_400162

def cube_surface_area (s : ℝ) : ℝ := 6 * s^2

def cube_volume (s : ℝ) : ℝ := s^3

theorem cube_volume_from_surface_area (surface_area : ℝ) (h : surface_area = 150) :
  ∃ s : ℝ, cube_surface_area s = surface_area ∧ cube_volume s = 125 :=
by sorry

end cube_volume_from_surface_area_l4001_400162


namespace car_cost_car_cost_proof_l4001_400191

/-- The cost of Alex's car, given his savings and earnings from grocery deliveries -/
theorem car_cost (initial_savings : ℝ) (trip_charge : ℝ) (grocery_percentage : ℝ) 
  (num_trips : ℕ) (grocery_value : ℝ) : ℝ :=
  let earnings_from_trips := num_trips * trip_charge
  let earnings_from_groceries := grocery_percentage * grocery_value
  let total_earnings := earnings_from_trips + earnings_from_groceries
  let total_savings := initial_savings + total_earnings
  total_savings

/-- Proof that the car costs $14,600 -/
theorem car_cost_proof : 
  car_cost 14500 1.5 0.05 40 800 = 14600 := by
  sorry

end car_cost_car_cost_proof_l4001_400191


namespace tangent_line_triangle_area_l4001_400155

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := a * x^3 + 3 * x

-- Define the derivative of f(x)
def f' (a : ℝ) (x : ℝ) : ℝ := 3 * a * x^2 + 3

-- State the theorem
theorem tangent_line_triangle_area (a : ℝ) : 
  (f' a 1 = -6) →  -- Condition for perpendicularity
  (∃ b c : ℝ, 
    (∀ x : ℝ, -6 * x + b = c * x + f a 1) ∧  -- Equation of tangent line
    (b = 6) ∧  -- y-intercept of tangent line
    (c = -6)) →  -- Slope of tangent line
  (1/2 * 1 * 6 = 3) :=  -- Area of triangle
by sorry

end tangent_line_triangle_area_l4001_400155


namespace complex_expression_simplification_l4001_400147

theorem complex_expression_simplification :
  (-3 : ℂ) + 7 * Complex.I - 3 * (2 - 5 * Complex.I) + 4 * Complex.I = -9 + 26 * Complex.I :=
by sorry

end complex_expression_simplification_l4001_400147


namespace selection_theorem_1_selection_theorem_2_selection_theorem_3_l4001_400165

/-- The number of female students -/
def num_female : ℕ := 5

/-- The number of male students -/
def num_male : ℕ := 4

/-- The number of students to be selected -/
def num_selected : ℕ := 4

/-- The number of ways to select exactly 2 male and 2 female students -/
def selection_method_1 : ℕ := 1440

/-- The number of ways to select at least 1 male and 1 female student -/
def selection_method_2 : ℕ := 2880

/-- The number of ways to select at least 1 male and 1 female student, 
    but male student A and female student B cannot be selected together -/
def selection_method_3 : ℕ := 2376

/-- Theorem for the first selection method -/
theorem selection_theorem_1 : 
  (Nat.choose num_male 2 * Nat.choose num_female 2) * (Nat.factorial num_selected) = selection_method_1 := by
  sorry

/-- Theorem for the second selection method -/
theorem selection_theorem_2 : 
  ((Nat.choose num_male 1 * Nat.choose num_female 3) + 
   (Nat.choose num_male 2 * Nat.choose num_female 2) + 
   (Nat.choose num_male 3 * Nat.choose num_female 1)) * 
  (Nat.factorial num_selected) = selection_method_2 := by
  sorry

/-- Theorem for the third selection method -/
theorem selection_theorem_3 : 
  selection_method_2 - 
  ((Nat.choose (num_male - 1) 2 + Nat.choose (num_female - 1) 1 * Nat.choose (num_male - 1) 1 + 
    Nat.choose (num_female - 1) 2) * Nat.factorial num_selected) = selection_method_3 := by
  sorry

end selection_theorem_1_selection_theorem_2_selection_theorem_3_l4001_400165


namespace range_of_a_l4001_400169

-- Define propositions p and q
def p (a : ℝ) : Prop := -2 < a ∧ a ≤ 2
def q (a : ℝ) : Prop := 0 < a ∧ a < 1

-- Define the set of valid a values
def valid_a_set : Set ℝ := {a | (1 ≤ a ∧ a ≤ 2) ∨ (-2 < a ∧ a ≤ 0)}

-- Theorem statement
theorem range_of_a (a : ℝ) :
  (p a ∨ q a) ∧ ¬(p a ∧ q a) ↔ a ∈ valid_a_set :=
sorry

end range_of_a_l4001_400169


namespace teacher_problem_l4001_400127

theorem teacher_problem (x : ℤ) : 4 * (3 * (x + 3) - 2) = 4 * (3 * x + 9 - 2) := by
  sorry

#check teacher_problem

end teacher_problem_l4001_400127


namespace simple_interest_time_period_l4001_400178

/-- Proves that given a principal of 6000, if increasing the interest rate by 2%
    results in 360 more interest over the same time period, then the time period is 3 years. -/
theorem simple_interest_time_period 
  (principal : ℝ) 
  (rate : ℝ) 
  (time : ℝ) 
  (h1 : principal = 6000)
  (h2 : principal * (rate + 2) / 100 * time = principal * rate / 100 * time + 360) :
  time = 3 :=
by sorry

end simple_interest_time_period_l4001_400178


namespace fraction_equivalence_l4001_400102

theorem fraction_equivalence : (10 : ℝ) / (8 * 60) = 0.1 / (0.8 * 60) := by sorry

end fraction_equivalence_l4001_400102


namespace binary_operation_theorem_l4001_400111

def binary_to_decimal (b : List Bool) : Nat :=
  b.reverse.enum.foldl (fun acc (i, bit) => acc + if bit then 2^i else 0) 0

def decimal_to_binary (n : Nat) : List Bool :=
  if n = 0 then [false] else
  let rec aux (m : Nat) (acc : List Bool) : List Bool :=
    if m = 0 then acc
    else aux (m / 2) ((m % 2 = 1) :: acc)
  aux n []

def binary_add_subtract (a b c d : List Bool) : List Bool :=
  let sum := binary_to_decimal a + binary_to_decimal b - binary_to_decimal c + binary_to_decimal d
  decimal_to_binary sum

theorem binary_operation_theorem :
  binary_add_subtract [true, true, false, true] [true, true, true] [true, false, true, false] [true, false, false, true] =
  [true, false, false, true, true] := by sorry

end binary_operation_theorem_l4001_400111


namespace min_value_of_f_l4001_400122

-- Define the function f
def f (x : ℝ) : ℝ := x^2 - 4*x + 4

-- State the theorem
theorem min_value_of_f :
  ∃ (x_min : ℝ), ∀ (x : ℝ), f x_min ≤ f x ∧ f x_min = 0 := by
  sorry

end min_value_of_f_l4001_400122


namespace arithmetic_sequence_condition_l4001_400194

def is_arithmetic_sequence (a : ℕ+ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ+, a (n + 1) = a n + d

theorem arithmetic_sequence_condition (a : ℕ+ → ℝ) :
  (∀ n : ℕ+, a n = 2 * (n : ℝ) + 1) → is_arithmetic_sequence a ∧
  ∃ b : ℕ+ → ℝ, is_arithmetic_sequence b ∧ ∃ m : ℕ+, b m ≠ 2 * (m : ℝ) + 1 :=
by sorry

end arithmetic_sequence_condition_l4001_400194


namespace sequence_general_term_formula_l4001_400101

/-- Given a quadratic equation with real roots, prove the general term formula for a sequence defined by a recurrence relation. -/
theorem sequence_general_term_formula 
  (p q : ℝ) 
  (hq : q ≠ 0) 
  (α β : ℝ) 
  (hroots : α^2 - p*α + q = 0 ∧ β^2 - p*β + q = 0) 
  (a : ℕ → ℝ) 
  (ha1 : a 1 = p) 
  (ha2 : a 2 = p^2 - q) 
  (han : ∀ n : ℕ, n ≥ 3 → a n = p * a (n-1) - q * a (n-2)) :
  ∀ n : ℕ, n ≥ 1 → a n = (α^(n+1) - β^(n+1)) / (α - β) :=
sorry

end sequence_general_term_formula_l4001_400101


namespace max_value_of_f_l4001_400138

open Real

noncomputable def f (x : ℝ) := Real.log (3 * x) - 3 * x

theorem max_value_of_f :
  ∃ (c : ℝ), c ∈ Set.Ioo 0 (Real.exp 1) ∧
  (∀ x, x ∈ Set.Ioo 0 (Real.exp 1) → f x ≤ f c) ∧
  f c = -Real.log 3 - 1 :=
sorry

end max_value_of_f_l4001_400138


namespace prob_three_odd_in_six_rolls_prob_three_odd_in_six_rolls_correct_l4001_400149

/-- The probability of getting exactly 3 odd numbers when rolling a fair 6-sided die 6 times -/
theorem prob_three_odd_in_six_rolls : ℚ :=
  5/16

/-- Proves that the probability of getting exactly 3 odd numbers when rolling a fair 6-sided die 6 times is 5/16 -/
theorem prob_three_odd_in_six_rolls_correct : prob_three_odd_in_six_rolls = 5/16 := by
  sorry

end prob_three_odd_in_six_rolls_prob_three_odd_in_six_rolls_correct_l4001_400149


namespace train_length_l4001_400150

/-- The length of a train given its speed, the speed of a man walking in the opposite direction, and the time it takes for the train to pass the man. -/
theorem train_length (train_speed : ℝ) (man_speed : ℝ) (crossing_time : ℝ) :
  train_speed = 174.98560115190784 →
  man_speed = 5 →
  crossing_time = 10 →
  ∃ (length : ℝ), abs (length - 499.96) < 0.01 ∧
    length = (train_speed + man_speed) * (1000 / 3600) * crossing_time :=
sorry

end train_length_l4001_400150
