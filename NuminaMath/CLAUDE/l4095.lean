import Mathlib

namespace NUMINAMATH_CALUDE_centroid_coincidence_l4095_409518

/-- Represents a point in 2D space -/
structure Point :=
  (x : ℝ)
  (y : ℝ)

/-- Represents a triangle defined by three points -/
structure Triangle :=
  (A : Point)
  (B : Point)
  (C : Point)

/-- Calculates the area of a triangle -/
def triangleArea (t : Triangle) : ℝ := sorry

/-- Calculates the centroid of a triangle -/
def triangleCentroid (t : Triangle) : Point := sorry

/-- Theorem: The centroid of a triangle coincides with the centroid of its subtriangles -/
theorem centroid_coincidence (ABC : Triangle) : 
  let D : Point := sorry -- D is the foot of the altitude from C to AB
  let ACD : Triangle := ⟨ABC.A, ABC.C, D⟩
  let BCD : Triangle := ⟨ABC.B, ABC.C, D⟩
  let M1 : Point := triangleCentroid ACD
  let M2 : Point := triangleCentroid BCD
  let Z : Point := triangleCentroid ABC
  ∃ (t : ℝ), 0 < t ∧ t < 1 ∧
    Z.x = t * M1.x + (1 - t) * M2.x ∧
    Z.y = t * M1.y + (1 - t) * M2.y ∧
    t = (triangleArea BCD) / (triangleArea ACD + triangleArea BCD) :=
by sorry

end NUMINAMATH_CALUDE_centroid_coincidence_l4095_409518


namespace NUMINAMATH_CALUDE_perfect_cube_units_digits_l4095_409536

theorem perfect_cube_units_digits : 
  ∃! (S : Finset ℕ), 
    (∀ n : ℕ, n ∈ S ↔ ∃ m : ℕ, n = m^3 % 10) ∧ 
    Finset.card S = 10 :=
sorry

end NUMINAMATH_CALUDE_perfect_cube_units_digits_l4095_409536


namespace NUMINAMATH_CALUDE_triangle_properties_l4095_409503

theorem triangle_properties (a b c : ℝ) (A B C : ℝ) :
  a > 0 ∧ b > 0 ∧ c > 0 →
  A > 0 ∧ B > 0 ∧ C > 0 →
  A + B + C = π →
  a * Real.sin C = Real.sqrt 3 * c * Real.cos A →
  (A = π / 3) ∧
  (a = 2 → (1 / 2) * b * c * Real.sin A = Real.sqrt 3 → b = 2 ∧ c = 2) := by
  sorry

end NUMINAMATH_CALUDE_triangle_properties_l4095_409503


namespace NUMINAMATH_CALUDE_triangle_problem_l4095_409589

open Real

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively --/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The given conditions and the statements to prove --/
theorem triangle_problem (t : Triangle) 
  (h1 : 2 * t.b * cos t.C + t.c = 2 * t.a) 
  (h2 : cos t.A = 1 / 7) : 
  t.B = π / 3 ∧ t.c / t.a = 5 / 8 := by
  sorry

end NUMINAMATH_CALUDE_triangle_problem_l4095_409589


namespace NUMINAMATH_CALUDE_subset_implies_M_M_implies_subset_M_iff_subset_l4095_409531

-- Define the sets A and B
def A : Set ℝ := {x | x^2 - 4 = 0}
def B (a : ℝ) : Set ℝ := {x | a*x - 6 = 0}

-- Define the set M
def M : Set ℝ := {0, 3, -3}

-- Theorem statement
theorem subset_implies_M (a : ℝ) : B a ⊆ A → a ∈ M := by sorry

-- Theorem for the converse
theorem M_implies_subset (a : ℝ) : a ∈ M → B a ⊆ A := by sorry

-- Theorem for the equivalence
theorem M_iff_subset (a : ℝ) : a ∈ M ↔ B a ⊆ A := by sorry

end NUMINAMATH_CALUDE_subset_implies_M_M_implies_subset_M_iff_subset_l4095_409531


namespace NUMINAMATH_CALUDE_raise_calculation_l4095_409580

-- Define the original weekly earnings
def original_earnings : ℚ := 60

-- Define the percentage increase
def percentage_increase : ℚ := 33.33 / 100

-- Define the new weekly earnings
def new_earnings : ℚ := original_earnings * (1 + percentage_increase)

-- Theorem to prove
theorem raise_calculation :
  new_earnings = 80 := by sorry

end NUMINAMATH_CALUDE_raise_calculation_l4095_409580


namespace NUMINAMATH_CALUDE_decimal_23_equals_binary_10111_l4095_409514

/-- Converts a natural number to its binary representation as a list of bits -/
def to_binary (n : ℕ) : List Bool :=
  if n = 0 then [false] else
  let rec aux (m : ℕ) : List Bool :=
    if m = 0 then [] else (m % 2 = 1) :: aux (m / 2)
  aux n

/-- Converts a list of bits to its decimal representation -/
def from_binary (bits : List Bool) : ℕ :=
  bits.foldl (fun acc b => 2 * acc + if b then 1 else 0) 0

theorem decimal_23_equals_binary_10111 :
  to_binary 23 = [true, true, true, false, true] ∧
  from_binary [true, true, true, false, true] = 23 := by
  sorry

end NUMINAMATH_CALUDE_decimal_23_equals_binary_10111_l4095_409514


namespace NUMINAMATH_CALUDE_prob_same_color_l4095_409501

/-- Represents the contents of a bag of colored balls -/
structure BagContents where
  white : ℕ
  red : ℕ
  black : ℕ

/-- Calculates the total number of balls in a bag -/
def BagContents.total (bag : BagContents) : ℕ :=
  bag.white + bag.red + bag.black

/-- Represents the two bags in the problem -/
def bagA : BagContents := { white := 1, red := 2, black := 3 }
def bagB : BagContents := { white := 2, red := 3, black := 1 }

/-- Calculates the probability of drawing a specific color from a bag -/
def probColor (bag : BagContents) (color : ℕ) : ℚ :=
  color / bag.total

/-- The main theorem: probability of drawing same color from both bags -/
theorem prob_same_color :
  (probColor bagA bagA.white * probColor bagB bagB.white) +
  (probColor bagA bagA.red * probColor bagB bagB.red) +
  (probColor bagA bagA.black * probColor bagB bagB.black) = 11 / 36 := by
  sorry

end NUMINAMATH_CALUDE_prob_same_color_l4095_409501


namespace NUMINAMATH_CALUDE_tire_cost_theorem_l4095_409554

/-- Calculates the total cost of tires with given prices, discounts, and taxes -/
def totalTireCost (allTerrainPrice : ℝ) (allTerrainDiscount : ℝ) (allTerrainTax : ℝ)
                  (sparePrice : ℝ) (spareDiscount : ℝ) (spareTax : ℝ) : ℝ :=
  let allTerrainDiscountedPrice := allTerrainPrice * (1 - allTerrainDiscount)
  let allTerrainFinalPrice := allTerrainDiscountedPrice * (1 + allTerrainTax)
  let allTerrainTotal := 4 * allTerrainFinalPrice

  let spareDiscountedPrice := sparePrice * (1 - spareDiscount)
  let spareFinalPrice := spareDiscountedPrice * (1 + spareTax)

  allTerrainTotal + spareFinalPrice

/-- The total cost of tires is $291.20 -/
theorem tire_cost_theorem :
  totalTireCost 60 0.15 0.08 75 0.10 0.05 = 291.20 := by
  sorry

end NUMINAMATH_CALUDE_tire_cost_theorem_l4095_409554


namespace NUMINAMATH_CALUDE_nancy_soap_packs_l4095_409508

/-- Proves that Nancy bought 6 packs of soap given the conditions -/
theorem nancy_soap_packs : 
  ∀ (bars_per_pack total_bars : ℕ),
    bars_per_pack = 5 →
    total_bars = 30 →
    total_bars / bars_per_pack = 6 := by
  sorry

end NUMINAMATH_CALUDE_nancy_soap_packs_l4095_409508


namespace NUMINAMATH_CALUDE_no_multiple_of_five_2C4_l4095_409517

theorem no_multiple_of_five_2C4 : 
  ¬ ∃ (C : ℕ), 
    (100 ≤ 200 + 10 * C + 4) ∧ 
    (200 + 10 * C + 4 < 1000) ∧ 
    (C < 10) ∧ 
    ((200 + 10 * C + 4) % 5 = 0) :=
by sorry

end NUMINAMATH_CALUDE_no_multiple_of_five_2C4_l4095_409517


namespace NUMINAMATH_CALUDE_parabola_latus_rectum_p_l4095_409561

/-- Given a parabola with equation x^2 = 2py (p > 0) and latus rectum equation y = -3,
    prove that the value of p is 6. -/
theorem parabola_latus_rectum_p (p : ℝ) (h1 : p > 0) : 
  (∀ x y : ℝ, x^2 = 2*p*y) → (∃ x : ℝ, x^2 = 2*p*(-3)) → p = 6 := by
  sorry

end NUMINAMATH_CALUDE_parabola_latus_rectum_p_l4095_409561


namespace NUMINAMATH_CALUDE_a_eq_b_sufficient_not_necessary_l4095_409599

/-- The line y = x + 2 is tangent to the circle (x - a)² + (y - b)² = 2 -/
def is_tangent (a b : ℝ) : Prop :=
  ∃ (x y : ℝ), y = x + 2 ∧ (x - a)^2 + (y - b)^2 = 2 ∧
  ∀ (x' y' : ℝ), y' = x' + 2 → (x' - a)^2 + (y' - b)^2 ≥ 2

/-- The condition a = b is sufficient but not necessary for the tangency -/
theorem a_eq_b_sufficient_not_necessary :
  (∀ a b : ℝ, a = b → is_tangent a b) ∧
  ¬(∀ a b : ℝ, is_tangent a b → a = b) :=
sorry

end NUMINAMATH_CALUDE_a_eq_b_sufficient_not_necessary_l4095_409599


namespace NUMINAMATH_CALUDE_article_price_decrease_l4095_409532

theorem article_price_decrease (decreased_price : ℝ) (decrease_percentage : ℝ) (original_price : ℝ) : 
  decreased_price = 836 →
  decrease_percentage = 24 →
  decreased_price = original_price * (1 - decrease_percentage / 100) →
  original_price = 1100 := by
  sorry

end NUMINAMATH_CALUDE_article_price_decrease_l4095_409532


namespace NUMINAMATH_CALUDE_problem_statement_l4095_409598

theorem problem_statement (a b : ℝ) : 
  |a - 2| + (b + 1/2)^2 = 0 → a^2022 * b^2023 = -1/2 := by sorry

end NUMINAMATH_CALUDE_problem_statement_l4095_409598


namespace NUMINAMATH_CALUDE_smallest_k_for_sum_4n_plus_1_l4095_409558

/-- Given a positive integer n, M is the set of integers from 1 to 2n -/
def M (n : ℕ+) : Finset ℕ := Finset.range (2 * n) \ {0}

/-- A function that checks if a subset of M contains 4 distinct elements summing to 4n + 1 -/
def has_sum_4n_plus_1 (n : ℕ+) (S : Finset ℕ) : Prop :=
  ∃ (a b c d : ℕ), a ∈ S ∧ b ∈ S ∧ c ∈ S ∧ d ∈ S ∧
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
    a + b + c + d = 4 * n + 1

theorem smallest_k_for_sum_4n_plus_1 (n : ℕ+) :
  (∀ (S : Finset ℕ), S ⊆ M n → S.card = n + 3 → has_sum_4n_plus_1 n S) ∧
  (∃ (T : Finset ℕ), T ⊆ M n ∧ T.card = n + 2 ∧ ¬has_sum_4n_plus_1 n T) :=
sorry

end NUMINAMATH_CALUDE_smallest_k_for_sum_4n_plus_1_l4095_409558


namespace NUMINAMATH_CALUDE_geometric_configurations_l4095_409563

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the relations
variable (passes_through : Plane → Line → Prop)
variable (skew : Line → Line → Prop)
variable (perpendicular : Line → Line → Prop)
variable (parallel : Plane → Plane → Prop)
variable (perpendicular_planes : Plane → Plane → Prop)
variable (parallel_line_plane : Line → Plane → Prop)
variable (perpendicular_line_plane : Line → Plane → Prop)

-- State the theorem
theorem geometric_configurations 
  (α β : Plane) (l m : Line) 
  (h1 : passes_through α l)
  (h2 : passes_through α m)
  (h3 : passes_through β l)
  (h4 : passes_through β m)
  (h5 : skew l m)
  (h6 : perpendicular l m) :
  (∃ (α' β' : Plane) (l' m' : Line), 
    passes_through α' l' ∧ 
    passes_through α' m' ∧ 
    passes_through β' l' ∧ 
    passes_through β' m' ∧ 
    skew l' m' ∧ 
    perpendicular l' m' ∧
    ((parallel α' β') ∨ 
     (perpendicular_planes α' β') ∨ 
     (parallel_line_plane l' β') ∨ 
     (perpendicular_line_plane m' α'))) :=
by sorry

end NUMINAMATH_CALUDE_geometric_configurations_l4095_409563


namespace NUMINAMATH_CALUDE_sum_of_specific_terms_l4095_409578

theorem sum_of_specific_terms (a : ℕ → ℝ) (S : ℕ → ℝ) 
    (h : ∀ n : ℕ, S n = n^2 + 2*n) :
  a 3 + a 4 + a 5 + a 6 = 40 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_specific_terms_l4095_409578


namespace NUMINAMATH_CALUDE_tangent_sum_simplification_l4095_409535

theorem tangent_sum_simplification :
  (Real.tan (20 * π / 180) + Real.tan (50 * π / 180) + 
   Real.tan (70 * π / 180) + Real.tan (80 * π / 180)) / 
  Real.sin (30 * π / 180) = 
  2 * (1 / Real.cos (50 * π / 180) + 
       1 / (2 * Real.cos (70 * π / 180) * Real.cos (80 * π / 180))) := by
  sorry

end NUMINAMATH_CALUDE_tangent_sum_simplification_l4095_409535


namespace NUMINAMATH_CALUDE_largest_divisor_of_n_squared_div_7200_l4095_409526

theorem largest_divisor_of_n_squared_div_7200 (n : ℕ) (h1 : n > 0) (h2 : 7200 ∣ n^2) :
  (60 ∣ n) ∧ ∀ k : ℕ, k ∣ n → k ≤ 60 :=
by sorry

end NUMINAMATH_CALUDE_largest_divisor_of_n_squared_div_7200_l4095_409526


namespace NUMINAMATH_CALUDE_bagel_cut_theorem_l4095_409566

/-- The number of pieces formed when cutting a bagel into sectors -/
def bagelPieces (n : ℕ) : ℕ := n + 1

/-- Theorem: Cutting a bagel into sectors with 10 cuts results in 11 pieces -/
theorem bagel_cut_theorem : bagelPieces 10 = 11 := by
  sorry

end NUMINAMATH_CALUDE_bagel_cut_theorem_l4095_409566


namespace NUMINAMATH_CALUDE_soap_box_length_l4095_409560

/-- Represents the dimensions of a box -/
structure BoxDimensions where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Calculates the volume of a box given its dimensions -/
def boxVolume (box : BoxDimensions) : ℝ :=
  box.length * box.width * box.height

/-- Theorem: Given the carton and soap box dimensions, if 360 soap boxes fit exactly in the carton,
    then the length of a soap box is 7 inches -/
theorem soap_box_length
  (carton : BoxDimensions)
  (soap : BoxDimensions)
  (h1 : carton.length = 30 ∧ carton.width = 42 ∧ carton.height = 60)
  (h2 : soap.width = 6 ∧ soap.height = 5)
  (h3 : boxVolume carton = 360 * boxVolume soap) :
  soap.length = 7 := by
  sorry

end NUMINAMATH_CALUDE_soap_box_length_l4095_409560


namespace NUMINAMATH_CALUDE_hat_number_sum_l4095_409511

theorem hat_number_sum : ∀ (alice_num bob_num : ℕ),
  alice_num ∈ Finset.range 51 →
  bob_num ∈ Finset.range 51 →
  alice_num ≠ bob_num →
  (∃ (x : ℕ), alice_num < x ∧ x ≤ 50) →
  (∃ (y : ℕ), y < bob_num ∧ y ≤ 50) →
  bob_num % 3 = 0 →
  ∃ (k : ℕ), 2 * bob_num + alice_num = k^2 →
  alice_num + bob_num = 22 :=
by sorry

end NUMINAMATH_CALUDE_hat_number_sum_l4095_409511


namespace NUMINAMATH_CALUDE_ryan_english_study_time_l4095_409519

/-- The number of hours Ryan spends on learning Chinese daily -/
def chinese_hours : ℕ := 5

/-- The number of additional hours Ryan spends on learning English compared to Chinese -/
def additional_english_hours : ℕ := 2

/-- The number of hours Ryan spends on learning English daily -/
def english_hours : ℕ := chinese_hours + additional_english_hours

theorem ryan_english_study_time : english_hours = 7 := by
  sorry

end NUMINAMATH_CALUDE_ryan_english_study_time_l4095_409519


namespace NUMINAMATH_CALUDE_sides_divisible_by_three_l4095_409597

/-- A convex polygon divided into triangles by non-intersecting diagonals. -/
structure TriangulatedPolygon where
  /-- The number of sides of the polygon. -/
  sides : ℕ
  /-- The number of triangles in the triangulation. -/
  triangles : ℕ
  /-- The property that each vertex is a vertex of an odd number of triangles. -/
  odd_vertex_property : Bool

/-- 
Theorem: If a convex polygon is divided into triangles by non-intersecting diagonals,
and each vertex of the polygon is a vertex of an odd number of these triangles,
then the number of sides of the polygon is divisible by 3.
-/
theorem sides_divisible_by_three (p : TriangulatedPolygon) 
  (h : p.odd_vertex_property = true) : 
  ∃ k : ℕ, p.sides = 3 * k :=
sorry

end NUMINAMATH_CALUDE_sides_divisible_by_three_l4095_409597


namespace NUMINAMATH_CALUDE_distance_from_center_to_point_l4095_409506

-- Define the circle equation
def circle_equation (x y : ℝ) : Prop := x^2 + y^2 = 6*x - 8*y + 18

-- Define the center of the circle
def circle_center : ℝ × ℝ := (3, -4)

-- Define the point
def point : ℝ × ℝ := (3, -2)

-- Theorem statement
theorem distance_from_center_to_point : 
  let (cx, cy) := circle_center
  let (px, py) := point
  Real.sqrt ((cx - px)^2 + (cy - py)^2) = 2 :=
by sorry

end NUMINAMATH_CALUDE_distance_from_center_to_point_l4095_409506


namespace NUMINAMATH_CALUDE_pipe_b_rate_is_50_l4095_409528

/-- Represents the water tank system with three pipes -/
structure WaterTankSystem where
  tank_capacity : ℕ
  pipe_a_rate : ℕ
  pipe_b_rate : ℕ
  pipe_c_rate : ℕ
  cycle_time : ℕ
  total_time : ℕ

/-- Calculates the volume filled in one cycle -/
def volume_per_cycle (system : WaterTankSystem) : ℤ :=
  system.pipe_a_rate * 1 + system.pipe_b_rate * 2 - system.pipe_c_rate * 2

/-- Theorem stating that the rate of Pipe B must be 50 L/min -/
theorem pipe_b_rate_is_50 (system : WaterTankSystem) 
  (h1 : system.tank_capacity = 2000)
  (h2 : system.pipe_a_rate = 200)
  (h3 : system.pipe_c_rate = 25)
  (h4 : system.cycle_time = 5)
  (h5 : system.total_time = 40)
  (h6 : (system.total_time / system.cycle_time : ℤ) * volume_per_cycle system = system.tank_capacity) :
  system.pipe_b_rate = 50 := by
  sorry

end NUMINAMATH_CALUDE_pipe_b_rate_is_50_l4095_409528


namespace NUMINAMATH_CALUDE_incorrect_calculation_l4095_409590

theorem incorrect_calculation (x y : ℝ) : 
  (-2 * x^2 * y^2)^3 / (-x * y)^3 ≠ -2 * x^3 * y^3 := by
  sorry

end NUMINAMATH_CALUDE_incorrect_calculation_l4095_409590


namespace NUMINAMATH_CALUDE_money_distribution_l4095_409546

theorem money_distribution (raquel sam nataly tom : ℚ) : 
  raquel = 40 →
  nataly = 3 * raquel →
  nataly = (5/3) * sam →
  tom = (1/4) * nataly →
  tom + raquel + nataly + sam = 262 := by
sorry

end NUMINAMATH_CALUDE_money_distribution_l4095_409546


namespace NUMINAMATH_CALUDE_basketball_score_l4095_409539

/-- Calculates the total points scored in a basketball game given the number of 2-point and 3-point shots made. -/
def totalPoints (twoPointShots threePointShots : ℕ) : ℕ :=
  2 * twoPointShots + 3 * threePointShots

/-- Proves that 7 two-point shots and 3 three-point shots result in a total of 23 points. -/
theorem basketball_score : totalPoints 7 3 = 23 := by
  sorry

end NUMINAMATH_CALUDE_basketball_score_l4095_409539


namespace NUMINAMATH_CALUDE_no_nontrivial_integer_solutions_l4095_409550

theorem no_nontrivial_integer_solutions :
  ∀ (x y z : ℤ), x^3 + 2*y^3 + 4*z^3 - 6*x*y*z = 0 → x = 0 ∧ y = 0 ∧ z = 0 :=
by sorry

end NUMINAMATH_CALUDE_no_nontrivial_integer_solutions_l4095_409550


namespace NUMINAMATH_CALUDE_jogger_train_distance_l4095_409530

/-- Represents the problem of calculating the distance a jogger is ahead of a train. -/
theorem jogger_train_distance
  (jogger_speed : ℝ)
  (train_speed : ℝ)
  (train_length : ℝ)
  (passing_time : ℝ)
  (h1 : jogger_speed = 9 * (5 / 18))  -- 9 km/hr in m/s
  (h2 : train_speed = 45 * (5 / 18))  -- 45 km/hr in m/s
  (h3 : train_length = 150)           -- 150 meters
  (h4 : passing_time = 39)            -- 39 seconds
  : (train_speed - jogger_speed) * passing_time = train_length + 240 :=
by sorry

end NUMINAMATH_CALUDE_jogger_train_distance_l4095_409530


namespace NUMINAMATH_CALUDE_a_range_theorem_l4095_409572

theorem a_range_theorem (a : ℝ) : 
  (∀ x : ℝ, a^2 * x - 2*(a - x - 4) < 0) ↔ -2 < a ∧ a ≤ 2 :=
by sorry

end NUMINAMATH_CALUDE_a_range_theorem_l4095_409572


namespace NUMINAMATH_CALUDE_complex_multiplication_l4095_409588

theorem complex_multiplication (i : ℂ) (h : i^2 = -1) : (1 - i) * i = 1 + i := by
  sorry

end NUMINAMATH_CALUDE_complex_multiplication_l4095_409588


namespace NUMINAMATH_CALUDE_line_inclination_angle_l4095_409529

theorem line_inclination_angle (a : ℝ) : 
  (∃ y : ℝ → ℝ, ∀ x, a * x - y x - 1 = 0) →  -- line equation
  (Real.tan (π / 3) = a) →                   -- angle of inclination
  a = Real.sqrt 3 :=                         -- conclusion
by sorry

end NUMINAMATH_CALUDE_line_inclination_angle_l4095_409529


namespace NUMINAMATH_CALUDE_florist_roses_count_l4095_409533

/-- Calculates the final number of roses a florist has after selling and picking more. -/
def final_roses (initial : ℕ) (sold : ℕ) (picked : ℕ) : ℕ :=
  initial - sold + picked

/-- Proves that given the initial conditions, the florist ends up with 56 roses. -/
theorem florist_roses_count : final_roses 50 15 21 = 56 := by
  sorry

end NUMINAMATH_CALUDE_florist_roses_count_l4095_409533


namespace NUMINAMATH_CALUDE_rectangle_dimension_l4095_409586

theorem rectangle_dimension (x : ℝ) : 
  (3*x - 5 > 0) ∧ (x + 7 > 0) ∧ ((3*x - 5) * (x + 7) = 15*x - 14) → x = 3 :=
by sorry

end NUMINAMATH_CALUDE_rectangle_dimension_l4095_409586


namespace NUMINAMATH_CALUDE_accounting_course_count_l4095_409575

/-- Represents the number of employees who took an accounting course -/
def accounting_course : ℕ := sorry

/-- Represents the number of employees who took a finance course -/
def finance_course : ℕ := 14

/-- Represents the number of employees who took a marketing course -/
def marketing_course : ℕ := 15

/-- Represents the number of employees who took exactly two courses -/
def two_courses : ℕ := 10

/-- Represents the number of employees who took all three courses -/
def all_courses : ℕ := 1

/-- Represents the number of employees who took none of the courses -/
def no_courses : ℕ := 11

/-- The total number of employees -/
def total_employees : ℕ := 50

theorem accounting_course_count : accounting_course = 19 := by
  sorry

end NUMINAMATH_CALUDE_accounting_course_count_l4095_409575


namespace NUMINAMATH_CALUDE_sam_pages_sam_read_100_pages_l4095_409565

def minimum_assigned : ℕ := 25

def harrison_extra : ℕ := 10

def pam_extra : ℕ := 15

def sam_multiplier : ℕ := 2

theorem sam_pages : ℕ :=
  let harrison_pages := minimum_assigned + harrison_extra
  let pam_pages := harrison_pages + pam_extra
  sam_multiplier * pam_pages

theorem sam_read_100_pages : sam_pages = 100 := by
  sorry

end NUMINAMATH_CALUDE_sam_pages_sam_read_100_pages_l4095_409565


namespace NUMINAMATH_CALUDE_square_side_length_l4095_409500

theorem square_side_length (s : ℝ) (h : s > 0) :
  s^2 = 3 * (4 * s) → s = 12 := by
  sorry

end NUMINAMATH_CALUDE_square_side_length_l4095_409500


namespace NUMINAMATH_CALUDE_susan_age_l4095_409582

/-- Proves that Susan's age is 25 given the conditions in the problem -/
theorem susan_age (susan joe billy : ℕ) : 
  susan = 2 * joe →           -- Susan is twice as old as Joe
  susan + joe + billy = 60 → -- The sum of their ages is 60
  billy = joe + 10 →         -- Billy is 10 years older than Joe
  susan = 25 := by
sorry

end NUMINAMATH_CALUDE_susan_age_l4095_409582


namespace NUMINAMATH_CALUDE_final_S_value_l4095_409581

def S : ℕ → ℕ
  | 0 => 1
  | n + 1 => S n + 2

theorem final_S_value : S 4 = 9 := by
  sorry

end NUMINAMATH_CALUDE_final_S_value_l4095_409581


namespace NUMINAMATH_CALUDE_pascal_triangle_51st_row_third_number_l4095_409534

theorem pascal_triangle_51st_row_third_number : 
  let n : ℕ := 51
  let k : ℕ := 2
  Nat.choose n k = 1275 := by sorry

end NUMINAMATH_CALUDE_pascal_triangle_51st_row_third_number_l4095_409534


namespace NUMINAMATH_CALUDE_hyperbola_equation_l4095_409557

/-- A hyperbola with right focus at (5, 0) and an asymptote with equation 2x - y = 0 
    has the standard equation x²/5 - y²/20 = 1 -/
theorem hyperbola_equation (x y : ℝ) : 
  let right_focus : ℝ × ℝ := (5, 0)
  let asymptote (x y : ℝ) : Prop := 2 * x - y = 0
  x^2 / 5 - y^2 / 20 = 1 :=
by sorry


end NUMINAMATH_CALUDE_hyperbola_equation_l4095_409557


namespace NUMINAMATH_CALUDE_quentavious_gum_pieces_l4095_409584

/-- Given the initial number of nickels, the number of nickels left, and the number of gum pieces per nickel,
    calculate the total number of gum pieces received. -/
def gumPiecesReceived (initialNickels : ℕ) (nickelsLeft : ℕ) (gumPiecesPerNickel : ℕ) : ℕ :=
  (initialNickels - nickelsLeft) * gumPiecesPerNickel

/-- Theorem: The number of gum pieces Quentavious received is 6, given the problem conditions. -/
theorem quentavious_gum_pieces :
  gumPiecesReceived 5 2 2 = 6 := by
  sorry

end NUMINAMATH_CALUDE_quentavious_gum_pieces_l4095_409584


namespace NUMINAMATH_CALUDE_inequality_proof_l4095_409507

theorem inequality_proof (α β γ : ℝ) 
  (h1 : β * γ ≠ 0) 
  (h2 : (1 - γ^2) / (β * γ) ≥ 0) : 
  10 * (α^2 + β^2 + γ^2 - β * γ^2) ≥ 2 * α * β + 5 * α * γ := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l4095_409507


namespace NUMINAMATH_CALUDE_simplified_expression_equality_l4095_409523

theorem simplified_expression_equality (a b : ℝ) : 
  (∀ x : ℝ, x^2 - 6*x + b = (x - a)^2 - 1) → b - a = 5 := by
  sorry

end NUMINAMATH_CALUDE_simplified_expression_equality_l4095_409523


namespace NUMINAMATH_CALUDE_smallest_k_value_l4095_409576

theorem smallest_k_value (p q r s k : ℕ+) : 
  (p + 2*q + 3*r + 4*s = k) →
  (4*p = 3*q) →
  (4*p = 2*r) →
  (4*p = s) →
  (∀ p' q' r' s' k' : ℕ+, 
    (p' + 2*q' + 3*r' + 4*s' = k') →
    (4*p' = 3*q') →
    (4*p' = 2*r') →
    (4*p' = s') →
    k ≤ k') →
  k = 77 := by
sorry

end NUMINAMATH_CALUDE_smallest_k_value_l4095_409576


namespace NUMINAMATH_CALUDE_sally_picked_42_peaches_l4095_409577

/-- The number of peaches Sally picked at the orchard -/
def peaches_picked (initial final : ℕ) : ℕ := final - initial

/-- Theorem: Sally picked 42 peaches at the orchard -/
theorem sally_picked_42_peaches (initial final : ℕ) 
  (h1 : initial = 13) 
  (h2 : final = 55) : 
  peaches_picked initial final = 42 := by
  sorry

end NUMINAMATH_CALUDE_sally_picked_42_peaches_l4095_409577


namespace NUMINAMATH_CALUDE_sum_of_positive_reals_l4095_409542

theorem sum_of_positive_reals (x y z w : ℝ) 
  (pos_x : 0 < x) (pos_y : 0 < y) (pos_z : 0 < z) (pos_w : 0 < w)
  (sum_sq_xy : x^2 + y^2 = 2500)
  (sum_sq_zw : z^2 + w^2 = 2500)
  (prod_xz : x * z = 1200)
  (prod_yw : y * w = 1200) :
  x + y + z + w = 140 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_positive_reals_l4095_409542


namespace NUMINAMATH_CALUDE_inverse_proportion_k_value_l4095_409538

/-- Given an inverse proportion function y = k/x passing through (2, -6), prove k = -12 -/
theorem inverse_proportion_k_value : ∀ k : ℝ, 
  (∃ f : ℝ → ℝ, (∀ x : ℝ, x ≠ 0 → f x = k / x) ∧ f 2 = -6) → 
  k = -12 := by
  sorry

end NUMINAMATH_CALUDE_inverse_proportion_k_value_l4095_409538


namespace NUMINAMATH_CALUDE_greatest_four_digit_divisible_by_63_and_9_l4095_409549

/-- Reverses the digits of a positive integer -/
def reverseDigits (n : ℕ) : ℕ := sorry

/-- Checks if a number is a four-digit integer -/
def isFourDigit (n : ℕ) : Prop := 1000 ≤ n ∧ n ≤ 9999

theorem greatest_four_digit_divisible_by_63_and_9 :
  ∃ (p : ℕ), 
    isFourDigit p ∧ 
    isFourDigit (reverseDigits p) ∧ 
    p % 63 = 0 ∧ 
    (reverseDigits p) % 63 = 0 ∧ 
    p % 9 = 0 ∧
    ∀ (x : ℕ), 
      isFourDigit x ∧ 
      isFourDigit (reverseDigits x) ∧ 
      x % 63 = 0 ∧ 
      (reverseDigits x) % 63 = 0 ∧ 
      x % 9 = 0 → 
      x ≤ p ∧
    p = 9507 := by
  sorry

end NUMINAMATH_CALUDE_greatest_four_digit_divisible_by_63_and_9_l4095_409549


namespace NUMINAMATH_CALUDE_average_with_five_thousandths_l4095_409585

theorem average_with_five_thousandths (x : ℝ) : 
  (x + 0.005) / 2 = 0.2025 → x = 0.400 := by
  sorry

end NUMINAMATH_CALUDE_average_with_five_thousandths_l4095_409585


namespace NUMINAMATH_CALUDE_distance_from_origin_to_point_l4095_409587

/-- The distance from the origin to the point (12, -5) in a rectangular coordinate system is 13 units. -/
theorem distance_from_origin_to_point : Real.sqrt (12^2 + (-5)^2) = 13 := by
  sorry

end NUMINAMATH_CALUDE_distance_from_origin_to_point_l4095_409587


namespace NUMINAMATH_CALUDE_y_value_l4095_409524

theorem y_value (y : ℚ) (h : (2/3 : ℚ) - (3/5 : ℚ) = 5/y) : y = 75 := by
  sorry

end NUMINAMATH_CALUDE_y_value_l4095_409524


namespace NUMINAMATH_CALUDE_proposition_q_undetermined_l4095_409551

theorem proposition_q_undetermined (p q : Prop) 
  (h1 : ¬(p ∧ q)) 
  (h2 : ¬p) : 
  (q ∨ ¬q) ∧ ¬(q ∧ ¬q) :=
by sorry

end NUMINAMATH_CALUDE_proposition_q_undetermined_l4095_409551


namespace NUMINAMATH_CALUDE_bus_journey_distance_l4095_409571

/-- Represents the distance traveled by a bus after k hours, given a total journey of 100 km -/
def distance_traveled (k : ℕ) : ℚ :=
  (100 * k) / (k + 1)

/-- Theorem stating that after 6 hours, the distance traveled is 600/7 km -/
theorem bus_journey_distance :
  distance_traveled 6 = 600 / 7 := by
  sorry

end NUMINAMATH_CALUDE_bus_journey_distance_l4095_409571


namespace NUMINAMATH_CALUDE_hyperbola_equation_l4095_409515

/-- The standard equation of a hyperbola with the same foci as a given ellipse and passing through a specific point -/
theorem hyperbola_equation (e : Real → Real → Prop) (p : Real × Real) :
  (∀ x y, e x y ↔ x^2 / 9 + y^2 / 5 = 1) →
  p = (Real.sqrt 2, Real.sqrt 3) →
  ∃ h : Real → Real → Prop,
    (∀ x y, h x y ↔ x^2 - y^2 / 3 = 1) ∧
    (∀ c : Real, (∃ x, e x 0 ∧ x^2 = c^2) ↔ (∃ x, h x 0 ∧ x^2 = c^2)) ∧
    h p.1 p.2 :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l4095_409515


namespace NUMINAMATH_CALUDE_wall_passing_skill_l4095_409556

theorem wall_passing_skill (n : ℕ) (h : 8 * Real.sqrt (8 / n) = Real.sqrt (8 * (8 / n))) :
  n = 63 := by
  sorry

end NUMINAMATH_CALUDE_wall_passing_skill_l4095_409556


namespace NUMINAMATH_CALUDE_parallelogram_diagonals_contain_conjugate_diameters_l4095_409502

-- Define an ellipse
structure Ellipse where
  center : ℝ × ℝ
  a : ℝ  -- semi-major axis
  b : ℝ  -- semi-minor axis

-- Define a parallelogram
structure Parallelogram where
  vertices : Fin 4 → ℝ × ℝ

-- Define conjugate diameters of an ellipse
def conjugate_diameters (e : Ellipse) : Set (ℝ × ℝ) := sorry

-- Define the diagonals of a parallelogram
def diagonals (p : Parallelogram) : Set (ℝ × ℝ) := sorry

-- Define what it means for a parallelogram to be inscribed around an ellipse
def is_inscribed (p : Parallelogram) (e : Ellipse) : Prop := sorry

-- Theorem statement
theorem parallelogram_diagonals_contain_conjugate_diameters 
  (e : Ellipse) (p : Parallelogram) (h : is_inscribed p e) :
  diagonals p ⊆ conjugate_diameters e := by sorry

end NUMINAMATH_CALUDE_parallelogram_diagonals_contain_conjugate_diameters_l4095_409502


namespace NUMINAMATH_CALUDE_min_value_complex_expression_l4095_409570

theorem min_value_complex_expression (z : ℂ) (h : Complex.abs z = 2) :
  ∃ (min : ℝ), min = 0 ∧ ∀ w : ℂ, Complex.abs w = 2 → Complex.abs ((w - 2)^2 * (w + 2)) ≥ min :=
by sorry

end NUMINAMATH_CALUDE_min_value_complex_expression_l4095_409570


namespace NUMINAMATH_CALUDE_essay_competition_probability_l4095_409513

theorem essay_competition_probability (n : ℕ) (h : n = 6) :
  let total_outcomes := n * n
  let favorable_outcomes := n * (n - 1)
  (favorable_outcomes : ℚ) / total_outcomes = 5 / 6 :=
by sorry

end NUMINAMATH_CALUDE_essay_competition_probability_l4095_409513


namespace NUMINAMATH_CALUDE_f_of_two_equals_negative_twenty_six_l4095_409516

/-- Given a function f(x) = ax^5 + bx^3 + sin(x) - 8 where f(-2) = 10, prove that f(2) = -26 -/
theorem f_of_two_equals_negative_twenty_six 
  (a b : ℝ) 
  (f : ℝ → ℝ) 
  (h1 : ∀ x, f x = a * x^5 + b * x^3 + Real.sin x - 8) 
  (h2 : f (-2) = 10) : 
  f 2 = -26 := by
sorry

end NUMINAMATH_CALUDE_f_of_two_equals_negative_twenty_six_l4095_409516


namespace NUMINAMATH_CALUDE_shyne_garden_theorem_l4095_409573

/-- Represents the number of plants that can be grown from one packet of seeds for each type of plant. -/
structure PlantsPerPacket where
  eggplants : ℕ
  sunflowers : ℕ
  tomatoes : ℕ
  peas : ℕ
  cucumbers : ℕ

/-- Represents the number of seed packets bought for each type of plant. -/
structure PacketsBought where
  eggplants : ℕ
  sunflowers : ℕ
  tomatoes : ℕ
  peas : ℕ
  cucumbers : ℕ

/-- Represents the percentage of plants that can be grown in each season. -/
structure PlantingPercentages where
  spring_eggplants_peas : ℚ
  summer_sunflowers_cucumbers : ℚ
  both_seasons_tomatoes : ℚ

/-- Calculates the total number of plants Shyne can potentially grow across spring and summer. -/
def totalPlants (plantsPerPacket : PlantsPerPacket) (packetsBought : PacketsBought) (percentages : PlantingPercentages) : ℕ :=
  sorry

/-- Theorem stating that Shyne can potentially grow 366 plants across spring and summer. -/
theorem shyne_garden_theorem (plantsPerPacket : PlantsPerPacket) (packetsBought : PacketsBought) (percentages : PlantingPercentages) :
  plantsPerPacket.eggplants = 14 ∧
  plantsPerPacket.sunflowers = 10 ∧
  plantsPerPacket.tomatoes = 16 ∧
  plantsPerPacket.peas = 20 ∧
  plantsPerPacket.cucumbers = 18 ∧
  packetsBought.eggplants = 6 ∧
  packetsBought.sunflowers = 8 ∧
  packetsBought.tomatoes = 7 ∧
  packetsBought.peas = 9 ∧
  packetsBought.cucumbers = 5 ∧
  percentages.spring_eggplants_peas = 3/5 ∧
  percentages.summer_sunflowers_cucumbers = 7/10 ∧
  percentages.both_seasons_tomatoes = 4/5 →
  totalPlants plantsPerPacket packetsBought percentages = 366 :=
by
  sorry

end NUMINAMATH_CALUDE_shyne_garden_theorem_l4095_409573


namespace NUMINAMATH_CALUDE_system_solution_l4095_409544

-- Define the system of equations
def equation1 (x y a b : ℝ) : Prop := x / (x - a) + y / (y - b) = 2
def equation2 (x y a b : ℝ) : Prop := a * x + b * y = 2 * a * b

-- State the theorem
theorem system_solution (a b : ℝ) (ha : a ≠ b) (hab : a + b ≠ 0) :
  ∃ x y : ℝ, equation1 x y a b ∧ equation2 x y a b ∧ x = 2 * a * b / (a + b) ∧ y = 2 * a * b / (a + b) :=
by sorry

end NUMINAMATH_CALUDE_system_solution_l4095_409544


namespace NUMINAMATH_CALUDE_ball_box_arrangements_l4095_409537

/-- The number of different arrangements of 4 balls in 4 boxes -/
def arrangements (n : ℕ) : ℕ := sorry

/-- The number of arrangements where exactly one box contains 2 balls -/
def one_box_two_balls : ℕ := arrangements 1

/-- The number of arrangements where exactly two boxes are left empty -/
def two_boxes_empty : ℕ := arrangements 2

theorem ball_box_arrangements :
  (one_box_two_balls = 144) ∧ (two_boxes_empty = 84) := by sorry

end NUMINAMATH_CALUDE_ball_box_arrangements_l4095_409537


namespace NUMINAMATH_CALUDE_train_length_calculation_train_B_length_l4095_409545

/-- Given two trains running in opposite directions, calculate the length of the second train. -/
theorem train_length_calculation (length_A : ℝ) (speed_A : ℝ) (speed_B : ℝ) (time : ℝ) : ℝ :=
  let relative_speed := (speed_A + speed_B) * 1000 / 3600
  let total_distance := relative_speed * time
  total_distance - length_A

/-- Prove that the length of Train B is approximately 219.95 meters. -/
theorem train_B_length :
  ∃ (length_B : ℝ), abs (length_B - train_length_calculation 280 120 80 9) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_train_length_calculation_train_B_length_l4095_409545


namespace NUMINAMATH_CALUDE_quadratic_equal_roots_equation_C_has_equal_roots_l4095_409591

theorem quadratic_equal_roots (a b c : ℝ) (h : a ≠ 0) :
  (b^2 - 4*a*c = 0) ↔ ∃! x, a*x^2 + b*x + c = 0 :=
sorry

theorem equation_C_has_equal_roots :
  ∃! x, x^2 + 12*x + 36 = 0 :=
sorry

end NUMINAMATH_CALUDE_quadratic_equal_roots_equation_C_has_equal_roots_l4095_409591


namespace NUMINAMATH_CALUDE_fraction_power_simplification_l4095_409548

theorem fraction_power_simplification :
  (66666 : ℕ) = 3 * 22222 →
  (66666 : ℚ)^4 / (22222 : ℚ)^4 = 81 :=
by
  sorry

end NUMINAMATH_CALUDE_fraction_power_simplification_l4095_409548


namespace NUMINAMATH_CALUDE_diagonal_crosses_24_tiles_l4095_409505

/-- The number of tiles crossed by a diagonal line on a rectangular grid --/
def tiles_crossed (width length : ℕ) : ℕ :=
  width + length - Nat.gcd width length

/-- Proof that a diagonal on a 12x15 rectangle crosses 24 tiles --/
theorem diagonal_crosses_24_tiles :
  tiles_crossed 12 15 = 24 := by
  sorry

end NUMINAMATH_CALUDE_diagonal_crosses_24_tiles_l4095_409505


namespace NUMINAMATH_CALUDE_ariel_current_age_l4095_409569

/-- Represents a person with birth year and fencing information -/
structure Person where
  birth_year : Nat
  fencing_start_year : Nat
  years_fencing : Nat

/-- Calculate the current year based on fencing information -/
def current_year (p : Person) : Nat :=
  p.fencing_start_year + p.years_fencing

/-- Calculate the age of a person in a given year -/
def age_in_year (p : Person) (year : Nat) : Nat :=
  year - p.birth_year

/-- Ariel's information -/
def ariel : Person :=
  { birth_year := 1992
  , fencing_start_year := 2006
  , years_fencing := 16 }

/-- Theorem: Ariel's current age is 30 years old -/
theorem ariel_current_age :
  age_in_year ariel (current_year ariel) = 30 := by
  sorry

end NUMINAMATH_CALUDE_ariel_current_age_l4095_409569


namespace NUMINAMATH_CALUDE_range_of_a_l4095_409521

-- Define propositions p and q
def p (a : ℝ) : Prop := ∀ x, ∃ y, y = Real.log (a * x^2 - x + (1/4) * a)
def q (a : ℝ) : Prop := ∀ x > 0, 3^x - 9^x < a

-- Define the theorem
theorem range_of_a :
  ∀ a : ℝ, (p a ∨ q a) ∧ ¬(p a ∧ q a) → 0 ≤ a ∧ a ≤ 1 :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l4095_409521


namespace NUMINAMATH_CALUDE_expand_product_l4095_409579

theorem expand_product (x : ℝ) : (x + 2) * (x^2 - 4*x + 1) = x^3 - 2*x^2 - 7*x + 2 := by
  sorry

end NUMINAMATH_CALUDE_expand_product_l4095_409579


namespace NUMINAMATH_CALUDE_placards_per_person_l4095_409592

def total_placards : ℕ := 5682
def people_entered : ℕ := 2841

theorem placards_per_person :
  total_placards / people_entered = 2 := by
  sorry

end NUMINAMATH_CALUDE_placards_per_person_l4095_409592


namespace NUMINAMATH_CALUDE_ellipse_a_value_l4095_409593

-- Define the ellipse equation
def ellipse_equation (a x y : ℝ) : Prop := x^2 / a^2 + y^2 / 2 = 1

-- Define the parabola equation
def parabola_equation (x y : ℝ) : Prop := y^2 = 8 * x

-- Define the focus of the parabola
def parabola_focus : ℝ × ℝ := (2, 0)

-- Theorem statement
theorem ellipse_a_value :
  ∃ (a : ℝ), 
    (∀ (x y : ℝ), ellipse_equation a x y → 
      ∃ (c : ℝ), c = 2 ∧ a^2 = 2 + c^2) ∧ 
    (∀ (x y : ℝ), parabola_equation x y → 
      ∃ (f : ℝ × ℝ), f = parabola_focus) →
  a = Real.sqrt 6 ∨ a = -Real.sqrt 6 :=
sorry

end NUMINAMATH_CALUDE_ellipse_a_value_l4095_409593


namespace NUMINAMATH_CALUDE_coin_selection_valid_l4095_409525

def available_coins : List ℕ := [1, 3, 5, 10, 20, 50]
def selected_coins : List ℕ := [1, 1, 3, 5, 10, 10, 20, 50]

def can_make_amount (coins : List ℕ) (amount : ℕ) : Prop :=
  ∃ (counts : List ℕ), 
    coins.length = counts.length ∧ 
    (List.sum (List.zipWith (· * ·) coins counts) = amount)

theorem coin_selection_valid :
  (∀ amount : ℕ, 1 ≤ amount ∧ amount ≤ 100 → can_make_amount selected_coins amount) ∧
  (∀ coin : ℕ, coin ∈ selected_coins → coin ∈ available_coins) ∧
  selected_coins.length = 8 :=
sorry

end NUMINAMATH_CALUDE_coin_selection_valid_l4095_409525


namespace NUMINAMATH_CALUDE_snack_cost_theorem_l4095_409543

/-- The total cost of snacks bought by Robert and Teddy -/
def total_cost (pizza_price : ℕ) (pizza_quantity : ℕ) (drink_price : ℕ) (robert_drink_quantity : ℕ) (hamburger_price : ℕ) (hamburger_quantity : ℕ) (teddy_drink_quantity : ℕ) : ℕ :=
  pizza_price * pizza_quantity + 
  drink_price * robert_drink_quantity + 
  hamburger_price * hamburger_quantity + 
  drink_price * teddy_drink_quantity

theorem snack_cost_theorem : 
  total_cost 10 5 2 10 3 6 10 = 108 := by
  sorry

end NUMINAMATH_CALUDE_snack_cost_theorem_l4095_409543


namespace NUMINAMATH_CALUDE_correct_calculation_l4095_409520

theorem correct_calculation (x : ℝ) : 3 * x - 10 = 50 → 3 * x + 10 = 70 := by
  sorry

end NUMINAMATH_CALUDE_correct_calculation_l4095_409520


namespace NUMINAMATH_CALUDE_complement_A_B_when_a_is_one_A_intersection_B_equals_A_l4095_409564

-- Define sets A and B
def A (a : ℝ) : Set ℝ := {x | 0 < a * x + 1 ∧ a * x + 1 ≤ 3}
def B : Set ℝ := {x | -1/2 < x ∧ x < 2}

-- Theorem for part (1)
theorem complement_A_B_when_a_is_one :
  (Set.univ \ A 1) ∩ B = {x | -1 < x ∧ x ≤ -1/2} ∪ {2} := by sorry

-- Theorem for part (2)
theorem A_intersection_B_equals_A (a : ℝ) :
  A a ∩ B = A a ↔ a < -4 ∨ a ≥ 2 := by sorry

end NUMINAMATH_CALUDE_complement_A_B_when_a_is_one_A_intersection_B_equals_A_l4095_409564


namespace NUMINAMATH_CALUDE_divisibility_criterion_a_divisibility_criterion_b_l4095_409540

-- Part a
theorem divisibility_criterion_a (n : ℕ) : 
  (∃ q : Polynomial ℚ, X^(2*n) + X^n + 1 = (X^2 + X + 1) * q) ↔ 
  (n % 3 = 1 ∨ n % 3 = 2) :=
sorry

-- Part b
theorem divisibility_criterion_b (n : ℕ) : 
  (∃ q : Polynomial ℚ, X^(2*n) - X^n + 1 = (X^2 - X + 1) * q) ↔ 
  (n % 6 = 1 ∨ n % 6 = 5) :=
sorry

end NUMINAMATH_CALUDE_divisibility_criterion_a_divisibility_criterion_b_l4095_409540


namespace NUMINAMATH_CALUDE_max_food_per_guest_l4095_409547

theorem max_food_per_guest (total_food : ℕ) (min_guests : ℕ) (max_food : ℕ) : 
  total_food = 406 → 
  min_guests = 163 → 
  max_food = 2 → 
  (total_food : ℚ) / min_guests ≤ max_food :=
by sorry

end NUMINAMATH_CALUDE_max_food_per_guest_l4095_409547


namespace NUMINAMATH_CALUDE_cos_pi_fourth_plus_alpha_l4095_409574

theorem cos_pi_fourth_plus_alpha (α : ℝ) (h : Real.sin (π/4 - α) = 1/3) : 
  Real.cos (π/4 + α) = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_cos_pi_fourth_plus_alpha_l4095_409574


namespace NUMINAMATH_CALUDE_bruce_grape_purchase_l4095_409553

/-- The price of grapes per kg -/
def grape_price : ℕ := 70

/-- The quantity of mangoes purchased in kg -/
def mango_quantity : ℕ := 9

/-- The price of mangoes per kg -/
def mango_price : ℕ := 55

/-- The total amount paid -/
def total_paid : ℕ := 985

/-- The quantity of grapes purchased in kg -/
def grape_quantity : ℕ := (total_paid - mango_quantity * mango_price) / grape_price

theorem bruce_grape_purchase :
  grape_quantity * grape_price + mango_quantity * mango_price = total_paid ∧ grape_quantity = 7 := by
  sorry

end NUMINAMATH_CALUDE_bruce_grape_purchase_l4095_409553


namespace NUMINAMATH_CALUDE_correct_paintball_spending_l4095_409512

/-- Represents the paintball spending calculation for John --/
def paintball_spending (regular_plays_per_month : ℕ) 
                       (boxes_per_play : ℕ) 
                       (price_1_5 : ℚ) 
                       (price_6_11 : ℚ) 
                       (price_12_plus : ℚ) 
                       (discount_12_plus : ℚ) 
                       (regular_maintenance : ℚ) 
                       (peak_maintenance : ℚ) 
                       (travel_week1 : ℚ) 
                       (travel_week2 : ℚ) 
                       (travel_week3 : ℚ) 
                       (travel_week4 : ℚ) : ℚ × ℚ :=
  let regular_boxes := regular_plays_per_month * boxes_per_play
  let peak_boxes := 2 * regular_boxes
  let travel_cost := travel_week1 + travel_week2 + travel_week3 + travel_week4
  
  let regular_paintball_cost := 
    if regular_boxes ≤ 5 then regular_boxes * price_1_5
    else if regular_boxes ≤ 11 then regular_boxes * price_6_11
    else let cost := regular_boxes * price_12_plus
         cost - (cost * discount_12_plus)
  
  let peak_paintball_cost := 
    let cost := peak_boxes * price_12_plus
    cost - (cost * discount_12_plus)
  
  let regular_total := regular_paintball_cost + regular_maintenance + travel_cost
  let peak_total := peak_paintball_cost + peak_maintenance + travel_cost
  
  (regular_total, peak_total)

/-- Theorem stating the correct paintball spending for John --/
theorem correct_paintball_spending :
  paintball_spending 3 3 25 23 22 (1/10) 40 60 10 15 12 8 = (292, 461.4) :=
sorry

end NUMINAMATH_CALUDE_correct_paintball_spending_l4095_409512


namespace NUMINAMATH_CALUDE_odd_integer_divides_power_factorial_minus_one_l4095_409595

theorem odd_integer_divides_power_factorial_minus_one (n : ℕ) (h_odd : Odd n) (h_ge_one : n ≥ 1) :
  n ∣ 2^(n!) - 1 := by
  sorry

end NUMINAMATH_CALUDE_odd_integer_divides_power_factorial_minus_one_l4095_409595


namespace NUMINAMATH_CALUDE_least_number_with_remainder_one_l4095_409559

theorem least_number_with_remainder_one (n : ℕ) : 
  (∀ m : ℕ, m > 0 → m < 386 → (m % 35 ≠ 1 ∨ m % 11 ≠ 1)) ∧ 
  386 % 35 = 1 ∧ 
  386 % 11 = 1 := by
sorry

end NUMINAMATH_CALUDE_least_number_with_remainder_one_l4095_409559


namespace NUMINAMATH_CALUDE_percentage_male_students_l4095_409562

theorem percentage_male_students 
  (T : ℝ) -- Total number of students
  (M : ℝ) -- Number of male students
  (F : ℝ) -- Number of female students
  (h1 : M + F = T) -- Total students equation
  (h2 : (2/7) * M + (1/3) * F = 0.3 * T) -- Married students equation
  : M / T = 0.7 := by sorry

end NUMINAMATH_CALUDE_percentage_male_students_l4095_409562


namespace NUMINAMATH_CALUDE_circle_line_distance_difference_l4095_409504

/-- Given a circle with equation x² + (y-1)² = 1 and a line x - y - 2 = 0,
    the difference between the maximum and minimum distances from points
    on the circle to the line is (√2)/2 + 1. -/
theorem circle_line_distance_difference :
  let circle := {p : ℝ × ℝ | p.1^2 + (p.2 - 1)^2 = 1}
  let line := {p : ℝ × ℝ | p.1 - p.2 - 2 = 0}
  let max_distance := Real.sqrt 8
  let min_distance := (3 * Real.sqrt 2) / 2 - 1
  max_distance - min_distance = Real.sqrt 2 / 2 + 1 := by
  sorry

end NUMINAMATH_CALUDE_circle_line_distance_difference_l4095_409504


namespace NUMINAMATH_CALUDE_cube_root_fifth_power_sixth_l4095_409527

theorem cube_root_fifth_power_sixth : (((5 ^ (1/2)) ^ 4) ^ (1/3)) ^ 6 = 625 := by
  sorry

end NUMINAMATH_CALUDE_cube_root_fifth_power_sixth_l4095_409527


namespace NUMINAMATH_CALUDE_trapezoid_xy_length_l4095_409583

/-- Represents a trapezoid WXYZ with specific properties -/
structure Trapezoid where
  -- Points W, X, Y, Z
  W : ℝ × ℝ
  X : ℝ × ℝ
  Y : ℝ × ℝ
  Z : ℝ × ℝ
  -- WX is parallel to ZY
  parallel_WX_ZY : (X.1 - W.1) * (Y.2 - Z.2) = (X.2 - W.2) * (Y.1 - Z.1)
  -- WY is perpendicular to ZY
  perpendicular_WY_ZY : (Y.1 - W.1) * (Y.1 - Z.1) + (Y.2 - W.2) * (Y.2 - Z.2) = 0
  -- YZ = 20
  yz_length : Real.sqrt ((Y.1 - Z.1)^2 + (Y.2 - Z.2)^2) = 20
  -- tan Z = 2
  tan_Z : (Y.2 - Z.2) / (Y.1 - Z.1) = 2
  -- tan X = 2.5
  tan_X : (Y.2 - X.2) / (X.1 - Y.1) = 2.5

/-- The length of XY in the trapezoid is 4√116 -/
theorem trapezoid_xy_length (t : Trapezoid) : 
  Real.sqrt ((t.X.1 - t.Y.1)^2 + (t.X.2 - t.Y.2)^2) = 4 * Real.sqrt 116 := by
  sorry


end NUMINAMATH_CALUDE_trapezoid_xy_length_l4095_409583


namespace NUMINAMATH_CALUDE_power_sixteen_divided_by_eight_l4095_409510

theorem power_sixteen_divided_by_eight (m : ℕ) : m = 16^2023 → m / 8 = 2^8089 := by
  sorry

end NUMINAMATH_CALUDE_power_sixteen_divided_by_eight_l4095_409510


namespace NUMINAMATH_CALUDE_expected_pine_saplings_in_sample_l4095_409522

/-- Given a forestry farm with the following characteristics:
  * total_saplings: The total number of saplings
  * pine_saplings: The number of pine saplings
  * sample_size: The size of the sample to be drawn
  
  This theorem proves that the expected number of pine saplings in the sample
  is equal to (pine_saplings / total_saplings) * sample_size. -/
theorem expected_pine_saplings_in_sample
  (total_saplings : ℕ)
  (pine_saplings : ℕ)
  (sample_size : ℕ)
  (h1 : total_saplings = 3000)
  (h2 : pine_saplings = 400)
  (h3 : sample_size = 150)
  : (pine_saplings : ℚ) / total_saplings * sample_size = 20 := by
  sorry

end NUMINAMATH_CALUDE_expected_pine_saplings_in_sample_l4095_409522


namespace NUMINAMATH_CALUDE_only_height_weight_correlated_l4095_409509

/-- Represents the relationship between two variables -/
inductive Relationship
  | Functional
  | Correlated
  | Unrelated

/-- Defines the relationship between a cube's volume and its edge length -/
def cube_volume_edge_relationship : Relationship := Relationship.Functional

/-- Defines the relationship between distance traveled and time for constant speed motion -/
def distance_time_relationship : Relationship := Relationship.Functional

/-- Defines the relationship between a person's height and eyesight -/
def height_eyesight_relationship : Relationship := Relationship.Unrelated

/-- Defines the relationship between a person's height and weight -/
def height_weight_relationship : Relationship := Relationship.Correlated

/-- Theorem stating that only height and weight have a correlation among the given pairs -/
theorem only_height_weight_correlated :
  (cube_volume_edge_relationship ≠ Relationship.Correlated) ∧
  (distance_time_relationship ≠ Relationship.Correlated) ∧
  (height_eyesight_relationship ≠ Relationship.Correlated) ∧
  (height_weight_relationship = Relationship.Correlated) :=
sorry

end NUMINAMATH_CALUDE_only_height_weight_correlated_l4095_409509


namespace NUMINAMATH_CALUDE_volume_ratio_of_cubes_l4095_409567

/-- The ratio of volumes of two cubes -/
theorem volume_ratio_of_cubes (inches_per_foot : ℚ) (small_edge : ℚ) (large_edge : ℚ) :
  inches_per_foot = 12 →
  small_edge = 3 →
  large_edge = 3/2 →
  (small_edge^3) / ((large_edge * inches_per_foot)^3) = 1/216 := by sorry

end NUMINAMATH_CALUDE_volume_ratio_of_cubes_l4095_409567


namespace NUMINAMATH_CALUDE_fraction_sum_equality_l4095_409568

theorem fraction_sum_equality (p q r : ℝ) (h_distinct : p ≠ q ∧ q ≠ r ∧ r ≠ p)
  (h_sum : p / (q - r) + q / (r - p) + r / (p - q) = 1) :
  p / (q - r)^2 + q / (r - p)^2 + r / (p - q)^2 = 
  1 / (q - r) + 1 / (r - p) + 1 / (p - q) - 1 := by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_equality_l4095_409568


namespace NUMINAMATH_CALUDE_hash_three_two_l4095_409555

-- Define the # operation
def hash (a b : ℕ) : ℕ := a * (b + 1) + a * b + b^2

-- Theorem statement
theorem hash_three_two : hash 3 2 = 19 := by
  sorry

end NUMINAMATH_CALUDE_hash_three_two_l4095_409555


namespace NUMINAMATH_CALUDE_system_solution_l4095_409541

theorem system_solution (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  (y - 2 * Real.sqrt (x * y) - Real.sqrt (y / x) + 2 = 0 ∧
   3 * x^2 * y^2 + y^4 = 84) ↔
  ((x = 1/3 ∧ y = 3) ∨ (x = (21/76)^(1/4) ∧ y = 2 * (84/19)^(1/4))) :=
sorry

end NUMINAMATH_CALUDE_system_solution_l4095_409541


namespace NUMINAMATH_CALUDE_parabola_coefficients_l4095_409596

/-- A parabola with given properties -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ
  vertex_x : ℝ
  vertex_y : ℝ
  point_x : ℝ
  point_y : ℝ
  vertex_property : vertex_y = a * vertex_x^2 + b * vertex_x + c
  point_property : point_y = a * point_x^2 + b * point_x + c
  symmetry_property : b = -2 * a * vertex_x

/-- The theorem stating the values of a, b, and c for the given parabola -/
theorem parabola_coefficients (p : Parabola)
  (h_vertex : p.vertex_x = 2 ∧ p.vertex_y = 4)
  (h_point : p.point_x = 0 ∧ p.point_y = 5) :
  p.a = 1/4 ∧ p.b = -1 ∧ p.c = 5 := by
  sorry

end NUMINAMATH_CALUDE_parabola_coefficients_l4095_409596


namespace NUMINAMATH_CALUDE_min_value_expression_l4095_409594

theorem min_value_expression (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (9 * b) / (4 * a) + (a + b) / b ≥ 4 := by
  sorry

end NUMINAMATH_CALUDE_min_value_expression_l4095_409594


namespace NUMINAMATH_CALUDE_min_value_of_expression_existence_of_minimum_l4095_409552

theorem min_value_of_expression (a : ℝ) (h1 : 1 < a) (h2 : a < 4) :
  (a / (4 - a)) + (1 / (a - 1)) ≥ 2 :=
sorry

theorem existence_of_minimum (a : ℝ) (h1 : 1 < a) (h2 : a < 4) :
  ∃ a, (a / (4 - a)) + (1 / (a - 1)) = 2 :=
sorry

end NUMINAMATH_CALUDE_min_value_of_expression_existence_of_minimum_l4095_409552
