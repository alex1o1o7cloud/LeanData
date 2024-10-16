import Mathlib

namespace NUMINAMATH_CALUDE_committee_selection_l1943_194314

theorem committee_selection (boys girls : ℕ) (h1 : boys = 21) (h2 : girls = 14) :
  (Nat.choose (boys + girls) 4) - (Nat.choose boys 4 + Nat.choose girls 4) = 45374 :=
sorry

end NUMINAMATH_CALUDE_committee_selection_l1943_194314


namespace NUMINAMATH_CALUDE_complex_fraction_sum_l1943_194380

theorem complex_fraction_sum (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hab : a * b ≠ a^3) :
  let sum := (a^2 - b^2) / (a * b) + (a * b + b^2) / (a * b - a^3)
  sum ≠ 1 ∧ sum ≠ (b^2 + b) / (b - a^2) ∧ sum ≠ 0 ∧ sum ≠ (a^2 + b) / (a^2 - b) :=
by sorry

end NUMINAMATH_CALUDE_complex_fraction_sum_l1943_194380


namespace NUMINAMATH_CALUDE_john_journey_distance_l1943_194347

/-- Calculates the total distance traveled given two journey segments -/
def total_distance (speed1 speed2 : ℝ) (time1 time2 : ℝ) : ℝ :=
  speed1 * time1 + speed2 * time2

/-- Theorem stating that the total distance of John's journey is 240 miles -/
theorem john_journey_distance :
  total_distance 45 50 2 3 = 240 := by
  sorry

end NUMINAMATH_CALUDE_john_journey_distance_l1943_194347


namespace NUMINAMATH_CALUDE_family_suitcases_l1943_194311

theorem family_suitcases (num_siblings : ℕ) (suitcases_per_sibling : ℕ) (total_suitcases : ℕ) : 
  num_siblings = 4 →
  suitcases_per_sibling = 2 →
  total_suitcases = 14 →
  ∃ (parents_suitcases : ℕ), 
    parents_suitcases = total_suitcases - (num_siblings * suitcases_per_sibling) ∧
    parents_suitcases % 2 = 0 ∧
    parents_suitcases / 2 = 3 :=
by sorry

end NUMINAMATH_CALUDE_family_suitcases_l1943_194311


namespace NUMINAMATH_CALUDE_nitrogen_electron_count_hydrazine_N2O4_reaction_hydrazine_combustion_l1943_194385

-- Define the chemical reactions and their enthalpies
def reaction1_enthalpy : ℝ := -19.5
def reaction2_enthalpy : ℝ := -534.2
def reaction3_enthalpy : ℝ := 44.0

-- Define the number of electrons in the L shell of a nitrogen atom
def nitrogen_L_shell_electrons : ℕ := 5

-- Define the enthalpy of the reaction between hydrazine and N₂O₄
def hydrazine_N2O4_reaction_enthalpy : ℝ := -1048.9

-- Define the combustion heat of hydrazine
def hydrazine_combustion_heat : ℝ := -622.2

-- Theorem statements
theorem nitrogen_electron_count :
  nitrogen_L_shell_electrons = 5 := by sorry

theorem hydrazine_N2O4_reaction :
  hydrazine_N2O4_reaction_enthalpy = 2 * reaction2_enthalpy - reaction1_enthalpy := by sorry

theorem hydrazine_combustion :
  hydrazine_combustion_heat = reaction2_enthalpy - 2 * reaction3_enthalpy := by sorry

end NUMINAMATH_CALUDE_nitrogen_electron_count_hydrazine_N2O4_reaction_hydrazine_combustion_l1943_194385


namespace NUMINAMATH_CALUDE_car_speed_problem_l1943_194351

/-- Given two cars traveling in opposite directions, prove that the speed of one car is 52 mph -/
theorem car_speed_problem (v : ℝ) : 
  v > 0 →  -- Assuming positive speed
  3.5 * v + 3.5 * 58 = 385 → 
  v = 52 := by
sorry

end NUMINAMATH_CALUDE_car_speed_problem_l1943_194351


namespace NUMINAMATH_CALUDE_triangle_centroid_distance_sum_l1943_194359

/-- Given a triangle ABC with centroid G, prove that if GA^2 + GB^2 + GC^2 = 58, 
    then AB^2 + AC^2 + BC^2 = 174. -/
theorem triangle_centroid_distance_sum (A B C G : ℝ × ℝ) : 
  (G = ((A.1 + B.1 + C.1) / 3, (A.2 + B.2 + C.2) / 3)) →  -- G is the centroid
  (dist G A)^2 + (dist G B)^2 + (dist G C)^2 = 58 →       -- Given condition
  (dist A B)^2 + (dist A C)^2 + (dist B C)^2 = 174 :=     -- Conclusion to prove
by
  sorry

#check triangle_centroid_distance_sum

end NUMINAMATH_CALUDE_triangle_centroid_distance_sum_l1943_194359


namespace NUMINAMATH_CALUDE_min_S_independent_of_P_l1943_194310

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a parabola of the form y = x² + c -/
structure Parabola where
  c : ℝ

/-- Represents the area bounded by a line and a parabola -/
def boundedArea (p₁ p₂ : Point) (C : Parabola) : ℝ := sorry

/-- The sum of areas S as described in the problem -/
def S (P : Point) (C₁ C₂ : Parabola) (m : ℕ) : ℝ := sorry

/-- The minimum value of S -/
def minS (m : ℕ) : ℝ := sorry

theorem min_S_independent_of_P (m : ℕ) :
  ∀ P : Point, P.y = P.x^2 + m^2 → minS m = m^3 / 3 := by sorry

end NUMINAMATH_CALUDE_min_S_independent_of_P_l1943_194310


namespace NUMINAMATH_CALUDE_computer_price_2004_l1943_194350

/-- The yearly decrease rate of the computer price -/
def yearly_decrease_rate : ℚ := 1 / 3

/-- The initial price of the computer in 2000 -/
def initial_price : ℚ := 8100

/-- The number of years between 2000 and 2004 -/
def years : ℕ := 4

/-- The price of the computer in 2004 -/
def price_2004 : ℚ := initial_price * (1 - yearly_decrease_rate) ^ years

theorem computer_price_2004 : price_2004 = 1600 := by
  sorry

end NUMINAMATH_CALUDE_computer_price_2004_l1943_194350


namespace NUMINAMATH_CALUDE_polygon_properties_l1943_194357

/-- Represents a convex polygon with properties as described in the problem -/
structure ConvexPolygon where
  n : ℕ                             -- number of sides
  interior_angle_sum : ℝ             -- sum of interior angles minus one unknown angle
  triangle_area : ℝ                  -- area of triangle formed by three adjacent vertices
  triangle_side : ℝ                  -- length of one side of the triangle
  triangle_opposite_angle : ℝ        -- angle opposite to the known side in the triangle

/-- The theorem to be proved -/
theorem polygon_properties (p : ConvexPolygon) 
  (h1 : p.interior_angle_sum = 3240)
  (h2 : p.triangle_area = 150)
  (h3 : p.triangle_side = 15)
  (h4 : p.triangle_opposite_angle = 60) :
  p.n = 20 ∧ (180 * (p.n - 2) - p.interior_angle_sum = 0) := by
  sorry


end NUMINAMATH_CALUDE_polygon_properties_l1943_194357


namespace NUMINAMATH_CALUDE_miranda_monthly_savings_l1943_194323

/-- Calculates the monthly savings given total cost, sister's contribution, and number of months saved. -/
def monthlySavings (totalCost : ℚ) (sisterContribution : ℚ) (monthsSaved : ℕ) : ℚ :=
  (totalCost - sisterContribution) / monthsSaved

/-- Proves that Miranda's monthly savings for the heels is $70. -/
theorem miranda_monthly_savings :
  let totalCost : ℚ := 260
  let sisterContribution : ℚ := 50
  let monthsSaved : ℕ := 3
  monthlySavings totalCost sisterContribution monthsSaved = 70 := by
sorry

end NUMINAMATH_CALUDE_miranda_monthly_savings_l1943_194323


namespace NUMINAMATH_CALUDE_hypotenuse_product_squared_l1943_194363

/-- Right triangle with given area and side lengths -/
structure RightTriangle where
  area : ℝ
  side1 : ℝ
  side2 : ℝ
  hypotenuse : ℝ
  area_eq : area = (side1 * side2) / 2
  pythagorean : side1^2 + side2^2 = hypotenuse^2

/-- The problem statement -/
theorem hypotenuse_product_squared
  (T₁ T₂ : RightTriangle)
  (h_area₁ : T₁.area = 2)
  (h_area₂ : T₂.area = 3)
  (h_side_congruent : T₁.side1 = T₂.side1)
  (h_side_double : T₁.side2 = 2 * T₂.side2) :
  (T₁.hypotenuse * T₂.hypotenuse)^2 = 325 := by
  sorry

end NUMINAMATH_CALUDE_hypotenuse_product_squared_l1943_194363


namespace NUMINAMATH_CALUDE_union_of_intervals_l1943_194382

open Set

theorem union_of_intervals (A B : Set ℝ) :
  A = Ioc (-1) 1 → B = Ioo 0 2 → A ∪ B = Ioo (-1) 2 := by
  sorry

end NUMINAMATH_CALUDE_union_of_intervals_l1943_194382


namespace NUMINAMATH_CALUDE_complement_of_A_in_U_l1943_194309

def U : Set ℕ := {1, 2, 3, 4, 5}
def A : Set ℕ := {2, 4}

theorem complement_of_A_in_U :
  (U \ A) = {1, 3, 5} := by sorry

end NUMINAMATH_CALUDE_complement_of_A_in_U_l1943_194309


namespace NUMINAMATH_CALUDE_complex_power_four_l1943_194301

theorem complex_power_four (i : ℂ) : i^2 = -1 → (1 - i)^4 = -4 := by
  sorry

end NUMINAMATH_CALUDE_complex_power_four_l1943_194301


namespace NUMINAMATH_CALUDE_expression_evaluation_expression_simplification_l1943_194379

-- Part 1
theorem expression_evaluation :
  Real.sqrt 2 + (1 : ℝ)^2014 + 2 * Real.cos (45 * π / 180) + Real.sqrt 16 = 2 * Real.sqrt 2 + 5 := by
  sorry

-- Part 2
theorem expression_simplification (x y : ℝ) (hx : x = Real.sqrt 2 + 1) (hy : y = Real.sqrt 2 - 1) :
  (x^2 + y^2 - 2*x*y) / (x - y) / ((x / y) - (y / x)) = Real.sqrt 2 / 4 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_expression_simplification_l1943_194379


namespace NUMINAMATH_CALUDE_cost_of_graveling_specific_lawn_l1943_194313

/-- Calculates the cost of graveling two intersecting roads on a rectangular lawn. -/
def cost_of_graveling (lawn_length lawn_width road_width gravel_cost : ℝ) : ℝ :=
  let road_length_area := lawn_length * road_width
  let road_width_area := (lawn_width - road_width) * road_width
  let total_area := road_length_area + road_width_area
  total_area * gravel_cost

/-- The cost of graveling two intersecting roads on a 70m × 60m lawn with 10m wide roads at Rs. 3 per sq m is Rs. 3600. -/
theorem cost_of_graveling_specific_lawn :
  cost_of_graveling 70 60 10 3 = 3600 := by
  sorry

end NUMINAMATH_CALUDE_cost_of_graveling_specific_lawn_l1943_194313


namespace NUMINAMATH_CALUDE_geometry_propositions_l1943_194339

-- Define the types for planes and lines
def Plane : Type := Unit
def Line : Type := Unit

-- Define the operations
def perpendicular (l : Line) (p : Plane) : Prop := sorry
def parallel_planes (p1 p2 : Plane) : Prop := sorry
def parallel_lines (l1 l2 : Line) : Prop := sorry
def parallel_line_plane (l : Line) (p : Plane) : Prop := sorry
def intersect (p1 p2 : Plane) : Line := sorry
def contained_in (l : Line) (p : Plane) : Prop := sorry

-- The theorem
theorem geometry_propositions (α β : Plane) (m n : Line) :
  (∀ α β m, perpendicular m α → perpendicular m β → parallel_planes α β) ∧
  ¬(∀ α β m n, parallel_line_plane m α → intersect α β = n → parallel_lines m n) ∧
  (∀ α m n, parallel_lines m n → perpendicular m α → perpendicular n α) ∧
  (∀ α β m n, perpendicular m α → parallel_lines m n → contained_in n β → perpendicular α β) :=
by sorry

end NUMINAMATH_CALUDE_geometry_propositions_l1943_194339


namespace NUMINAMATH_CALUDE_sin_cos_sum_equals_half_l1943_194324

theorem sin_cos_sum_equals_half : 
  Real.sin (21 * π / 180) * Real.cos (9 * π / 180) + 
  Real.sin (69 * π / 180) * Real.sin (9 * π / 180) = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_sum_equals_half_l1943_194324


namespace NUMINAMATH_CALUDE_intersection_distance_squared_l1943_194390

-- Define the curve C
def curve_C (x y : ℝ) : Prop :=
  x^2 + y^2 - 2*x - 4*y = 0

-- Define the line l
def line_l (x y : ℝ) : Prop :=
  x - Real.sqrt 3 * y + Real.sqrt 3 = 0

-- Define the intersection points
def intersection_points (A B M : ℝ × ℝ) : Prop :=
  curve_C A.1 A.2 ∧ curve_C B.1 B.2 ∧
  line_l A.1 A.2 ∧ line_l B.1 B.2 ∧
  line_l M.1 M.2 ∧ M.1 = 0

-- State the theorem
theorem intersection_distance_squared 
  (A B M : ℝ × ℝ) 
  (h : intersection_points A B M) : 
  (Real.sqrt ((A.1 - M.1)^2 + (A.2 - M.2)^2) + 
   Real.sqrt ((B.1 - M.1)^2 + (B.2 - M.2)^2))^2 = 16 + 2 * Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_intersection_distance_squared_l1943_194390


namespace NUMINAMATH_CALUDE_intersection_M_N_l1943_194320

def M : Set ℝ := {x | |x| ≤ 1}
def N : Set ℝ := {x | x^2 - x < 0}

theorem intersection_M_N : M ∩ N = {x : ℝ | 0 < x ∧ x < 1} := by sorry

end NUMINAMATH_CALUDE_intersection_M_N_l1943_194320


namespace NUMINAMATH_CALUDE_c4h1o_molecular_weight_l1943_194372

/-- The atomic weight of carbon in atomic mass units (amu) -/
def carbon_weight : ℝ := 12.01

/-- The atomic weight of hydrogen in atomic mass units (amu) -/
def hydrogen_weight : ℝ := 1.008

/-- The atomic weight of oxygen in atomic mass units (amu) -/
def oxygen_weight : ℝ := 16.00

/-- The number of carbon atoms in the compound -/
def carbon_count : ℕ := 4

/-- The number of hydrogen atoms in the compound -/
def hydrogen_count : ℕ := 1

/-- The number of oxygen atoms in the compound -/
def oxygen_count : ℕ := 1

/-- The molecular weight of C4H1O -/
def molecular_weight : ℝ :=
  carbon_weight * carbon_count + hydrogen_weight * hydrogen_count + oxygen_weight * oxygen_count

theorem c4h1o_molecular_weight :
  molecular_weight = 65.048 := by sorry

end NUMINAMATH_CALUDE_c4h1o_molecular_weight_l1943_194372


namespace NUMINAMATH_CALUDE_sqrt_x4_minus_x2_l1943_194368

theorem sqrt_x4_minus_x2 (x : ℝ) : Real.sqrt (x^4 - x^2) = |x| * Real.sqrt (x^2 - 1) := by
  sorry

end NUMINAMATH_CALUDE_sqrt_x4_minus_x2_l1943_194368


namespace NUMINAMATH_CALUDE_probability_all_genuine_proof_l1943_194370

/-- The total number of coins -/
def total_coins : ℕ := 15

/-- The number of genuine coins -/
def genuine_coins : ℕ := 12

/-- The number of counterfeit coins -/
def counterfeit_coins : ℕ := 3

/-- The number of pairs selected -/
def pairs_selected : ℕ := 3

/-- The number of coins in each pair -/
def coins_per_pair : ℕ := 2

/-- Predicate that the weight of counterfeit coins is different from genuine coins -/
axiom counterfeit_weight_different : True

/-- Predicate that the combined weight of all three pairs is the same -/
axiom combined_weight_same : True

/-- The probability of selecting all genuine coins given the conditions -/
def probability_all_genuine : ℚ := 264 / 443

/-- Theorem stating that the probability of selecting all genuine coins
    given the conditions is equal to 264/443 -/
theorem probability_all_genuine_proof :
  probability_all_genuine = 264 / 443 :=
by sorry

end NUMINAMATH_CALUDE_probability_all_genuine_proof_l1943_194370


namespace NUMINAMATH_CALUDE_percentage_calculation_l1943_194300

theorem percentage_calculation (x : ℝ) : 
  (70 / 100 * 600 : ℝ) = (x / 100 * 1050 : ℝ) → x = 40 := by
  sorry

end NUMINAMATH_CALUDE_percentage_calculation_l1943_194300


namespace NUMINAMATH_CALUDE_repeating_decimal_56_l1943_194312

/-- Represents a repeating decimal with a two-digit repetend -/
def RepeatingDecimal (a b : ℕ) : ℚ :=
  (10 * a + b : ℚ) / 99

theorem repeating_decimal_56 : RepeatingDecimal 5 6 = 56 / 99 := by
  sorry

end NUMINAMATH_CALUDE_repeating_decimal_56_l1943_194312


namespace NUMINAMATH_CALUDE_girls_points_in_checkers_tournament_l1943_194331

theorem girls_points_in_checkers_tournament (x : ℕ) : 
  x > 0 →  -- number of girls is positive
  2 * x * (10 * x - 1) = 18 →  -- derived equation for girls' points
  ∃ (total_games : ℕ) (total_points : ℕ),
    -- total number of games
    total_games = (10 * x) * (10 * x - 1) / 2 ∧
    -- total points distributed
    total_points = 2 * total_games ∧
    -- boys' points are 4 times girls' points
    4 * (2 * x * (10 * x - 1)) = total_points - (2 * x * (10 * x - 1)) :=
by
  sorry

#check girls_points_in_checkers_tournament

end NUMINAMATH_CALUDE_girls_points_in_checkers_tournament_l1943_194331


namespace NUMINAMATH_CALUDE_f_even_implies_specific_points_l1943_194348

/-- A function f on the real numbers. -/
def f (a b : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + 2 * a - b

/-- The domain of f is [2a-1, a^2+1] -/
def domain (a : ℝ) : Set ℝ := Set.Icc (2 * a - 1) (a^2 + 1)

/-- f is an even function -/
def is_even (a b : ℝ) : Prop := ∀ x ∈ domain a, f a b x = f a b (-x)

/-- The theorem stating that given the conditions, (a, b) can only be (0, 0) or (-2, 0) -/
theorem f_even_implies_specific_points :
  ∀ a b : ℝ, is_even a b → (a = 0 ∧ b = 0) ∨ (a = -2 ∧ b = 0) :=
sorry

end NUMINAMATH_CALUDE_f_even_implies_specific_points_l1943_194348


namespace NUMINAMATH_CALUDE_science_project_percentage_l1943_194344

theorem science_project_percentage (total_pages math_pages remaining_pages : ℕ) 
  (h1 : total_pages = 120)
  (h2 : math_pages = 10)
  (h3 : remaining_pages = 80) :
  (total_pages - math_pages - remaining_pages) / total_pages * 100 = 25 := by
  sorry

end NUMINAMATH_CALUDE_science_project_percentage_l1943_194344


namespace NUMINAMATH_CALUDE_log_inequality_implies_greater_l1943_194337

theorem log_inequality_implies_greater (a b : ℝ) (ha : a > 0) (hb : b > 0) : 
  Real.log a > Real.log b → a > b := by
  sorry

end NUMINAMATH_CALUDE_log_inequality_implies_greater_l1943_194337


namespace NUMINAMATH_CALUDE_unique_integer_satisfying_equation_l1943_194303

theorem unique_integer_satisfying_equation :
  ∃! n : ℕ, 1 ≤ n ∧ n ≤ 20200 ∧
  1 + ⌊(200 * n : ℚ) / 201⌋ = ⌈(198 * n : ℚ) / 200⌉ := by
  sorry

end NUMINAMATH_CALUDE_unique_integer_satisfying_equation_l1943_194303


namespace NUMINAMATH_CALUDE_probability_theorem_l1943_194315

/-- The probability of selecting three distinct integers between 1 and 100 (inclusive) 
    such that their product is odd and a multiple of 5 -/
def probability_odd_multiple_of_five : ℚ := 3 / 125

/-- The set of integers from 1 to 100, inclusive -/
def integer_set : Set ℕ := {n : ℕ | 1 ≤ n ∧ n ≤ 100}

/-- A function that determines if a natural number is odd -/
def is_odd (n : ℕ) : Prop := n % 2 = 1

/-- A function that determines if a natural number is a multiple of 5 -/
def is_multiple_of_five (n : ℕ) : Prop := n % 5 = 0

/-- The main theorem stating that the probability of selecting three distinct integers 
    between 1 and 100 (inclusive) such that their product is odd and a multiple of 5 
    is equal to 3/125 -/
theorem probability_theorem : 
  ∀ (a b c : ℕ), a ∈ integer_set → b ∈ integer_set → c ∈ integer_set → 
  a ≠ b → b ≠ c → a ≠ c →
  (is_odd a ∧ is_odd b ∧ is_odd c ∧ (is_multiple_of_five a ∨ is_multiple_of_five b ∨ is_multiple_of_five c)) →
  probability_odd_multiple_of_five = 3 / 125 := by
  sorry

end NUMINAMATH_CALUDE_probability_theorem_l1943_194315


namespace NUMINAMATH_CALUDE_x_squared_mod_20_l1943_194383

theorem x_squared_mod_20 (x : ℤ) 
  (h1 : 4 * x ≡ 8 [ZMOD 20]) 
  (h2 : 3 * x ≡ 16 [ZMOD 20]) : 
  x^2 ≡ 4 [ZMOD 20] := by
sorry

end NUMINAMATH_CALUDE_x_squared_mod_20_l1943_194383


namespace NUMINAMATH_CALUDE_practice_time_difference_l1943_194332

/-- Represents the practice schedule for Carlo's music recital --/
structure PracticeSchedule where
  monday : ℕ
  tuesday : ℕ
  wednesday : ℕ
  thursday : ℕ
  friday : ℕ

/-- Theorem stating the difference in practice time between Wednesday and Thursday --/
theorem practice_time_difference (schedule : PracticeSchedule) : 
  schedule.monday = 2 * schedule.tuesday →
  schedule.tuesday = schedule.wednesday - 10 →
  schedule.wednesday > schedule.thursday →
  schedule.thursday = 50 →
  schedule.friday = 60 →
  schedule.monday + schedule.tuesday + schedule.wednesday + schedule.thursday + schedule.friday = 300 →
  schedule.wednesday - schedule.thursday = 5 := by
  sorry

end NUMINAMATH_CALUDE_practice_time_difference_l1943_194332


namespace NUMINAMATH_CALUDE_function_and_cosine_value_l1943_194321

noncomputable def f (ω : ℝ) (m : ℝ) (x : ℝ) : ℝ := 
  2 * (Real.cos (ω * x))^2 + 2 * Real.sqrt 3 * Real.sin (ω * x) * Real.cos (ω * x) + m

theorem function_and_cosine_value 
  (ω : ℝ) (m : ℝ) (x₀ : ℝ) 
  (h_ω : ω > 0)
  (h_highest : f ω m (π / 6) = f ω m x → x ≤ π / 6)
  (h_passes : f ω m 0 = 2)
  (h_x₀_value : f ω m x₀ = 11 / 5)
  (h_x₀_range : π / 4 ≤ x₀ ∧ x₀ ≤ π / 2) :
  (∀ x, f ω m x = 2 * Real.sin (2 * x + π / 6) + 1) ∧
  Real.cos (2 * x₀) = (3 - 4 * Real.sqrt 3) / 10 := by
sorry

end NUMINAMATH_CALUDE_function_and_cosine_value_l1943_194321


namespace NUMINAMATH_CALUDE_quadrilateral_reconstruction_l1943_194364

/-- Given a quadrilateral ABCD with extended sides, prove that A can be expressed
    as a linear combination of A'', B'', C'', D'' with specific coefficients. -/
theorem quadrilateral_reconstruction
  (A B C D A'' B'' C'' D'' : ℝ × ℝ) -- Points in 2D space
  (h1 : A'' - A = 2 * (B - A))      -- AA'' = 2AB
  (h2 : B'' - B = 3 * (C - B))      -- BB'' = 3BC
  (h3 : C'' - C = 2 * (D - C))      -- CC'' = 2CD
  (h4 : D'' - D = 2 * (A - D)) :    -- DD'' = 2DA
  A = (1/6 : ℝ) • A'' + (1/9 : ℝ) • B'' + (1/9 : ℝ) • C'' + (1/18 : ℝ) • D'' := by
  sorry


end NUMINAMATH_CALUDE_quadrilateral_reconstruction_l1943_194364


namespace NUMINAMATH_CALUDE_chord_addition_theorem_sum_of_squares_theorem_l1943_194399

/-- Represents a circle with chords --/
structure ChordedCircle where
  num_chords : ℕ
  num_regions : ℕ

/-- The result of adding a chord to a circle --/
structure ChordAdditionResult where
  min_regions : ℕ
  max_regions : ℕ

/-- Function to add a chord to a circle --/
def add_chord (circle : ChordedCircle) : ChordAdditionResult :=
  { min_regions := circle.num_regions + 1,
    max_regions := circle.num_regions + circle.num_chords + 1 }

/-- Theorem statement --/
theorem chord_addition_theorem (initial_circle : ChordedCircle) 
  (h1 : initial_circle.num_chords = 4) 
  (h2 : initial_circle.num_regions = 9) : 
  let result := add_chord initial_circle
  result.min_regions = 10 ∧ result.max_regions = 14 := by
  sorry

/-- Corollary: The sum of squares of max and min regions --/
theorem sum_of_squares_theorem (initial_circle : ChordedCircle) 
  (h1 : initial_circle.num_chords = 4) 
  (h2 : initial_circle.num_regions = 9) : 
  let result := add_chord initial_circle
  result.max_regions ^ 2 + result.min_regions ^ 2 = 296 := by
  sorry

end NUMINAMATH_CALUDE_chord_addition_theorem_sum_of_squares_theorem_l1943_194399


namespace NUMINAMATH_CALUDE_electrocardiogram_is_line_chart_l1943_194374

/-- Represents different types of charts --/
inductive ChartType
  | BarChart
  | LineChart
  | PieChart

/-- Represents a chart that can display data --/
structure Chart where
  type : ChartType
  representsChangesOverTime : Bool

/-- Defines an electrocardiogram as a chart --/
def Electrocardiogram : Chart :=
  { type := ChartType.LineChart,
    representsChangesOverTime := true }

/-- Theorem stating that an electrocardiogram is a line chart --/
theorem electrocardiogram_is_line_chart : 
  Electrocardiogram.type = ChartType.LineChart :=
by
  sorry


end NUMINAMATH_CALUDE_electrocardiogram_is_line_chart_l1943_194374


namespace NUMINAMATH_CALUDE_overall_discount_rate_l1943_194358

def bag_marked : ℕ := 200
def shirt_marked : ℕ := 80
def shoes_marked : ℕ := 150
def hat_marked : ℕ := 50
def jacket_marked : ℕ := 220

def bag_sold : ℕ := 120
def shirt_sold : ℕ := 60
def shoes_sold : ℕ := 105
def hat_sold : ℕ := 40
def jacket_sold : ℕ := 165

def total_marked : ℕ := bag_marked + shirt_marked + shoes_marked + hat_marked + jacket_marked
def total_sold : ℕ := bag_sold + shirt_sold + shoes_sold + hat_sold + jacket_sold

theorem overall_discount_rate :
  (1 - (total_sold : ℚ) / total_marked) * 100 = 30 := by sorry

end NUMINAMATH_CALUDE_overall_discount_rate_l1943_194358


namespace NUMINAMATH_CALUDE_range_of_a_l1943_194349

-- Define the conditions
def p (x : ℝ) : Prop := x > 1 ∨ x < -3
def q (x a : ℝ) : Prop := x > a

-- State the theorem
theorem range_of_a (h : ∀ x a : ℝ, q x a → p x) :
  ∀ a : ℝ, (∀ x : ℝ, q x a → p x) ↔ a ≥ 1 := by sorry

end NUMINAMATH_CALUDE_range_of_a_l1943_194349


namespace NUMINAMATH_CALUDE_weight_of_replaced_person_l1943_194307

/-- Proves that given 8 persons, if replacing one person with a new person weighing 93 kg
    increases the average weight by 3.5 kg, then the weight of the replaced person was 65 kg. -/
theorem weight_of_replaced_person
  (initial_count : Nat)
  (weight_increase : ℝ)
  (new_person_weight : ℝ)
  (h1 : initial_count = 8)
  (h2 : weight_increase = 3.5)
  (h3 : new_person_weight = 93)
  : ℝ :=
by
  sorry

#check weight_of_replaced_person

end NUMINAMATH_CALUDE_weight_of_replaced_person_l1943_194307


namespace NUMINAMATH_CALUDE_hat_problem_probabilities_q_div_p_undefined_l1943_194395

/-- The number of slips in the hat -/
def total_slips : ℕ := 42

/-- The number of distinct numbers on the slips -/
def distinct_numbers : ℕ := 14

/-- The number of slips for each number -/
def slips_per_number : ℕ := 3

/-- The number of slips drawn -/
def drawn_slips : ℕ := 4

/-- The probability of drawing four slips with the same number -/
def p : ℚ := 0

/-- The number of ways to choose two distinct numbers and two slips for each -/
def favorable_outcomes : ℕ := Nat.choose distinct_numbers 2 * Nat.choose slips_per_number 2 * Nat.choose slips_per_number 2

/-- The probability of drawing two pairs of slips with different numbers -/
def q : ℚ := favorable_outcomes / Nat.choose total_slips drawn_slips

theorem hat_problem_probabilities :
  p = 0 ∧ q = 819 / Nat.choose total_slips drawn_slips :=
sorry

theorem q_div_p_undefined : ¬∃ (x : ℚ), q / p = x :=
sorry

end NUMINAMATH_CALUDE_hat_problem_probabilities_q_div_p_undefined_l1943_194395


namespace NUMINAMATH_CALUDE_planes_perpendicular_l1943_194330

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (subset : Line → Plane → Prop)
variable (parallel : Line → Line → Prop)
variable (perpendicular : Line → Plane → Prop)
variable (perp_planes : Plane → Plane → Prop)

-- State the theorem
theorem planes_perpendicular 
  (l m : Line) (α β : Plane)
  (h1 : subset l α)
  (h2 : parallel l m)
  (h3 : perpendicular m β) :
  perp_planes α β :=
sorry

end NUMINAMATH_CALUDE_planes_perpendicular_l1943_194330


namespace NUMINAMATH_CALUDE_max_regions_four_lines_l1943_194353

/-- The maximum number of regions into which a plane can be divided using n straight lines -/
def L (n : ℕ) : ℕ :=
  n * (n + 1) / 2 + 1

/-- The theorem stating that 4 straight lines can divide a plane into at most 11 regions -/
theorem max_regions_four_lines : L 4 = 11 := by
  sorry

end NUMINAMATH_CALUDE_max_regions_four_lines_l1943_194353


namespace NUMINAMATH_CALUDE_smallest_solution_congruence_l1943_194326

theorem smallest_solution_congruence :
  ∃ (x : ℕ), x > 0 ∧ (5 * x) % 31 = 22 % 31 ∧
  ∀ (y : ℕ), y > 0 → (5 * y) % 31 = 22 % 31 → x ≤ y :=
by sorry

end NUMINAMATH_CALUDE_smallest_solution_congruence_l1943_194326


namespace NUMINAMATH_CALUDE_number_2008_in_45th_group_l1943_194345

/-- The sequence of arrays where the nth group has n numbers and the last number of the nth group is n(n+1) -/
def sequence_group (n : ℕ) : ℕ := n * (n + 1)

/-- The proposition that 2008 is in the 45th group of the sequence -/
theorem number_2008_in_45th_group :
  ∃ k : ℕ, k ≤ 45 ∧ 
  sequence_group 44 < 2008 ∧ 
  2008 ≤ sequence_group 45 :=
by sorry

end NUMINAMATH_CALUDE_number_2008_in_45th_group_l1943_194345


namespace NUMINAMATH_CALUDE_unique_quadratic_solution_l1943_194336

theorem unique_quadratic_solution (b : ℝ) (hb : b ≠ 0) :
  (∃! x, 2 * b * x^2 + 16 * x + 5 = 0) →
  (∃ x, 2 * b * x^2 + 16 * x + 5 = 0 ∧ x = -5/8) :=
by sorry

end NUMINAMATH_CALUDE_unique_quadratic_solution_l1943_194336


namespace NUMINAMATH_CALUDE_all_heads_or_tails_probability_l1943_194397

def num_coins : ℕ := 8

def total_outcomes : ℕ := 2^num_coins

def favorable_outcomes : ℕ := 2

def probability : ℚ := favorable_outcomes / total_outcomes

theorem all_heads_or_tails_probability :
  probability = 1 / 128 := by sorry

end NUMINAMATH_CALUDE_all_heads_or_tails_probability_l1943_194397


namespace NUMINAMATH_CALUDE_one_in_set_zero_one_l1943_194316

theorem one_in_set_zero_one : 1 ∈ ({0, 1} : Set ℕ) := by
  sorry

end NUMINAMATH_CALUDE_one_in_set_zero_one_l1943_194316


namespace NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l1943_194306

/-- Given an arithmetic sequence {a_n} with a_3 = 5 and a_15 = 41, 
    prove that the common difference d is equal to 3. -/
theorem arithmetic_sequence_common_difference 
  (a : ℕ → ℝ) 
  (h_arithmetic : ∀ n : ℕ, a (n + 1) - a n = a (n + 2) - a (n + 1)) 
  (h_a3 : a 3 = 5) 
  (h_a15 : a 15 = 41) : 
  ∃ d : ℝ, d = 3 ∧ ∀ n : ℕ, a (n + 1) - a n = d :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l1943_194306


namespace NUMINAMATH_CALUDE_number_division_problem_l1943_194341

theorem number_division_problem : ∃ x : ℚ, x / 5 = 75 + x / 6 ∧ x = 2250 := by
  sorry

end NUMINAMATH_CALUDE_number_division_problem_l1943_194341


namespace NUMINAMATH_CALUDE_trig_expression_equals_sqrt_three_l1943_194362

theorem trig_expression_equals_sqrt_three : 
  (2 * Real.cos (10 * π / 180) - Real.sin (20 * π / 180)) / Real.sin (70 * π / 180) = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_trig_expression_equals_sqrt_three_l1943_194362


namespace NUMINAMATH_CALUDE_chocolate_difference_l1943_194394

/-- The number of chocolates Robert ate -/
def robert_chocolates : ℚ := 3 / 7 * 70

/-- The number of chocolates Nickel ate -/
def nickel_chocolates : ℚ := 120 / 100 * 40

/-- The number of chocolates Penny ate -/
def penny_chocolates : ℚ := 3 / 8 * 80

/-- The number of chocolates Dime ate -/
def dime_chocolates : ℚ := 1 / 2 * 90

/-- The difference between the number of chocolates eaten by Robert and Nickel combined
    and the number of chocolates eaten by Penny and Dime combined -/
theorem chocolate_difference :
  (robert_chocolates + nickel_chocolates) - (penny_chocolates + dime_chocolates) = -3 := by
  sorry

end NUMINAMATH_CALUDE_chocolate_difference_l1943_194394


namespace NUMINAMATH_CALUDE_binomial_sum_one_l1943_194346

theorem binomial_sum_one (a : ℝ) (a₁ a₂ a₃ a₄ a₅ : ℝ) :
  (∀ x : ℝ, (a*x - 1)^5 = a + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5) →
  a₃ = 80 →
  a + a₁ + a₂ + a₃ + a₄ + a₅ = 1 := by
sorry

end NUMINAMATH_CALUDE_binomial_sum_one_l1943_194346


namespace NUMINAMATH_CALUDE_problem_solution_l1943_194381

noncomputable def f (x : ℝ) := |Real.log x|

theorem problem_solution (a b : ℝ) (h1 : 0 < a) (h2 : a < b) (h3 : f a = f b) :
  (a * b = 1) ∧ 
  ((a + b) / 2 > 1) ∧ 
  (∃ b₀ : ℝ, 3 < b₀ ∧ b₀ < 4 ∧ 1 / b₀^2 + b₀^2 + 2 - 4 * b₀ = 0) := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l1943_194381


namespace NUMINAMATH_CALUDE_correct_number_placement_l1943_194302

-- Define the grid
inductive Square
| A | B | C | D | E | F | G | One | Nine

-- Define the arrow directions
inductive Direction
| Right | Down | Left | Up

-- Function to get the number in a square
def number_in_square (s : Square) : ℕ :=
  match s with
  | Square.A => 6
  | Square.B => 2
  | Square.C => 4
  | Square.D => 5
  | Square.E => 3
  | Square.F => 8
  | Square.G => 7
  | Square.One => 1
  | Square.Nine => 9

-- Function to get the directions of arrows in a square
def arrows_in_square (s : Square) : List Direction :=
  match s with
  | Square.One => [Direction.Right, Direction.Down]
  | Square.B => [Direction.Right, Direction.Down]
  | Square.C => [Direction.Right, Direction.Down]
  | Square.D => [Direction.Up]
  | Square.E => [Direction.Left]
  | Square.F => [Direction.Left]
  | Square.G => [Direction.Up, Direction.Right]
  | _ => []

-- Function to get the next square in a given direction
def next_square (s : Square) (d : Direction) : Option Square :=
  match s, d with
  | Square.One, Direction.Right => some Square.B
  | Square.One, Direction.Down => some Square.D
  | Square.B, Direction.Right => some Square.C
  | Square.B, Direction.Down => some Square.E
  | Square.C, Direction.Right => some Square.Nine
  | Square.C, Direction.Down => some Square.F
  | Square.D, Direction.Up => some Square.A
  | Square.E, Direction.Left => some Square.D
  | Square.F, Direction.Left => some Square.E
  | Square.G, Direction.Up => some Square.D
  | Square.G, Direction.Right => some Square.F
  | _, _ => none

-- Theorem statement
theorem correct_number_placement :
  (∀ s : Square, number_in_square s ∈ Set.range (fun i => i + 1) ∩ Set.Icc 1 9) ∧
  (∀ s : Square, s ≠ Square.Nine → 
    ∃ d ∈ arrows_in_square s, 
      ∃ next : Square, 
        next_square s d = some next ∧ 
        number_in_square next = number_in_square s + 1) :=
sorry

end NUMINAMATH_CALUDE_correct_number_placement_l1943_194302


namespace NUMINAMATH_CALUDE_least_multiple_with_digit_product_multiple_l1943_194392

/-- Returns the product of digits of a natural number -/
def digitProduct (n : ℕ) : ℕ := sorry

/-- Returns true if n is a multiple of m -/
def isMultipleOf (n m : ℕ) : Prop := sorry

/-- Returns true if n is the least number satisfying the given property -/
def isLeast (n : ℕ) (property : ℕ → Prop) : Prop := sorry

theorem least_multiple_with_digit_product_multiple :
  isLeast 315 (λ n : ℕ => isMultipleOf n 15 ∧ 
                          n > 0 ∧ 
                          isMultipleOf (digitProduct n) 15 ∧ 
                          digitProduct n > 0) := by
  sorry

end NUMINAMATH_CALUDE_least_multiple_with_digit_product_multiple_l1943_194392


namespace NUMINAMATH_CALUDE_sequence_a_formula_l1943_194373

def sequence_a (n : ℕ) : ℚ :=
  if n = 1 then -1
  else 2 / (n * (n + 1))

def S (n : ℕ) : ℚ := -2 / (n + 1)

theorem sequence_a_formula (n : ℕ) :
  (n = 1 ∧ sequence_a n = -1) ∨
  (n ≥ 2 ∧ sequence_a n = 2 / (n * (n + 1))) ∧
  (∀ k ≥ 2, (S k)^2 - (sequence_a k) * (S k) = 2 * (sequence_a k)) :=
sorry

end NUMINAMATH_CALUDE_sequence_a_formula_l1943_194373


namespace NUMINAMATH_CALUDE_triangle_area_theorem_l1943_194387

-- Define the triangle ABC and point P
structure Triangle :=
  (A B C P : ℝ × ℝ)

-- Define the areas of the smaller triangles
def small_triangle_areas (T : Triangle) : ℕ × ℕ × ℕ := (16, 25, 64)

-- Define the theorem
theorem triangle_area_theorem (T : Triangle) : 
  let (a1, a2, a3) := small_triangle_areas T
  (a1 : ℝ) + a2 + a3 > 0 →
  (∃ (l1 l2 l3 : ℝ), l1 > 0 ∧ l2 > 0 ∧ l3 > 0 ∧ 
    l1^2 = a1 ∧ l2^2 = a2 ∧ l3^2 = a3) →
  (∃ (A : ℝ), A = (l1 + l2 + l3)^2 * (a1 + a2 + a3) / (l1^2 + l2^2 + l3^2)) →
  A = 30345 :=
sorry

end NUMINAMATH_CALUDE_triangle_area_theorem_l1943_194387


namespace NUMINAMATH_CALUDE_team_selection_count_l1943_194352

/-- The number of ways to select a team of 3 people from 3 male and 3 female teachers,
    with both genders included -/
def select_team (male_teachers female_teachers team_size : ℕ) : ℕ :=
  (male_teachers.choose 2 * female_teachers.choose 1) +
  (male_teachers.choose 1 * female_teachers.choose 2)

/-- Theorem: There are 18 ways to select a team of 3 from 3 male and 3 female teachers,
    with both genders included -/
theorem team_selection_count :
  select_team 3 3 3 = 18 := by
  sorry

end NUMINAMATH_CALUDE_team_selection_count_l1943_194352


namespace NUMINAMATH_CALUDE_geometric_sequence_formula_l1943_194322

def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

theorem geometric_sequence_formula 
  (a : ℕ → ℝ) 
  (h_pos : ∀ n, a n > 0)
  (h_geometric : is_geometric_sequence a)
  (h_a4_a2 : a 4 = (a 2)^2)
  (h_sum : a 2 + a 4 = 5/16) :
  ∀ n : ℕ, a n = (1/2)^n :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_formula_l1943_194322


namespace NUMINAMATH_CALUDE_complement_union_theorem_l1943_194376

def U : Set ℕ := {x : ℕ | x > 0 ∧ x < 9}
def A : Set ℕ := {1, 2, 3, 4, 5, 6}
def B : Set ℕ := {4, 5, 6}

theorem complement_union_theorem :
  (U \ A) ∪ B = {4, 5, 6, 7, 8} := by sorry

end NUMINAMATH_CALUDE_complement_union_theorem_l1943_194376


namespace NUMINAMATH_CALUDE_smallest_divisible_fraction_l1943_194333

def fraction1 : Rat := 6 / 7
def fraction2 : Rat := 5 / 14
def fraction3 : Rat := 10 / 21

def smallest_fraction : Rat := 1 / 42

theorem smallest_divisible_fraction :
  (∀ r : Rat, (fraction1 ∣ r ∧ fraction2 ∣ r ∧ fraction3 ∣ r) → smallest_fraction ≤ r) ∧
  (fraction1 ∣ smallest_fraction ∧ fraction2 ∣ smallest_fraction ∧ fraction3 ∣ smallest_fraction) :=
sorry

end NUMINAMATH_CALUDE_smallest_divisible_fraction_l1943_194333


namespace NUMINAMATH_CALUDE_matrix_sum_theorem_l1943_194369

theorem matrix_sum_theorem (a b c : ℝ) : 
  let M : Matrix (Fin 3) (Fin 3) ℝ := !![a^2, b^2, c^2; b^2, c^2, a^2; c^2, a^2, b^2]
  ¬(IsUnit (M.det)) →
  (a^2 / (b^2 + c^2) + b^2 / (a^2 + c^2) + c^2 / (a^2 + b^2) = 3/2) ∨
  (a^2 / (b^2 + c^2) + b^2 / (a^2 + c^2) + c^2 / (a^2 + b^2) = -3) :=
by sorry


end NUMINAMATH_CALUDE_matrix_sum_theorem_l1943_194369


namespace NUMINAMATH_CALUDE_evaluate_expression_l1943_194342

theorem evaluate_expression : 3 * (-5) ^ (2 ^ (3/4)) = -15 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l1943_194342


namespace NUMINAMATH_CALUDE_lauras_apartment_number_l1943_194340

theorem lauras_apartment_number :
  ∃! n : ℕ,
    100 ≤ n ∧ n < 1000 ∧
    ∃ m : ℕ, n = m^2 ∧
    Even n ∧
    ¬(n % 11 = 0) ∧
    (n / 100 + (n / 10) % 10 + n % 10 = 12) ∧
    n % 10 = 4 :=
by sorry

end NUMINAMATH_CALUDE_lauras_apartment_number_l1943_194340


namespace NUMINAMATH_CALUDE_triangle_angle_solution_l1943_194361

theorem triangle_angle_solution (a b c : ℝ) (h1 : a = 40)
  (h2 : b = 3 * y) (h3 : c = y + 10) (h4 : a + b + c = 180) : y = 32.5 := by
  sorry

end NUMINAMATH_CALUDE_triangle_angle_solution_l1943_194361


namespace NUMINAMATH_CALUDE_triangle_cosine_inequality_l1943_194393

theorem triangle_cosine_inequality (A B C : Real) 
  (h_acute : A > 0 ∧ B > 0 ∧ C > 0)
  (h_triangle : A + B + C = Real.pi) : 
  (Real.cos A)^2 / (Real.cos B)^2 + 
  (Real.cos B)^2 / (Real.cos C)^2 + 
  (Real.cos C)^2 / (Real.cos A)^2 ≥ 
  4 * ((Real.cos A)^2 + (Real.cos B)^2 + (Real.cos C)^2) := by
  sorry

end NUMINAMATH_CALUDE_triangle_cosine_inequality_l1943_194393


namespace NUMINAMATH_CALUDE_cos_4theta_l1943_194338

theorem cos_4theta (θ : ℝ) (h : Complex.exp (θ * Complex.I) = (1 + Complex.I * Real.sqrt 7) / 4) : 
  Real.cos (4 * θ) = 1 / 32 := by
sorry

end NUMINAMATH_CALUDE_cos_4theta_l1943_194338


namespace NUMINAMATH_CALUDE_lottery_probability_l1943_194335

theorem lottery_probability : 
  let mega_balls : ℕ := 30
  let winner_balls : ℕ := 50
  let picked_winner_balls : ℕ := 5
  let mega_prob : ℚ := 1 / mega_balls
  let winner_prob : ℚ := 1 / (winner_balls.choose picked_winner_balls)
  mega_prob * winner_prob = 1 / 63562800 :=
by sorry

end NUMINAMATH_CALUDE_lottery_probability_l1943_194335


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l1943_194355

theorem quadratic_equation_solution (c d : ℝ) (hc : c ≠ 0) (hd : d ≠ 0) 
  (h1 : c^2 + c*c + d = 0) (h2 : d^2 + c*d + d = 0) : c = 1 ∧ d = -2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l1943_194355


namespace NUMINAMATH_CALUDE_average_speed_is_25_l1943_194354

def initial_reading : ℕ := 45654
def final_reading : ℕ := 45854
def total_time : ℕ := 8

def distance : ℕ := final_reading - initial_reading
def average_speed : ℚ := distance / total_time

theorem average_speed_is_25 : average_speed = 25 := by
  sorry

end NUMINAMATH_CALUDE_average_speed_is_25_l1943_194354


namespace NUMINAMATH_CALUDE_no_right_triangle_perimeter_twice_hypotenuse_l1943_194367

theorem no_right_triangle_perimeter_twice_hypotenuse :
  ¬∃ (a b c : ℝ), 
    a > 0 ∧ b > 0 ∧ c > 0 ∧  -- positive sides
    a^2 + b^2 = c^2 ∧        -- right triangle (Pythagorean theorem)
    a + b + c = 2*c          -- perimeter equals twice the hypotenuse
    := by sorry

end NUMINAMATH_CALUDE_no_right_triangle_perimeter_twice_hypotenuse_l1943_194367


namespace NUMINAMATH_CALUDE_solution_set_of_inequality_l1943_194305

theorem solution_set_of_inequality (x : ℝ) :
  (x * (x + 2) / (x - 3) < 0) ↔ (x < -2 ∨ (0 < x ∧ x < 3)) :=
by sorry

end NUMINAMATH_CALUDE_solution_set_of_inequality_l1943_194305


namespace NUMINAMATH_CALUDE_sum_of_squared_coefficients_l1943_194365

def polynomial (x : ℝ) : ℝ := 5 * (x^4 + 2*x^3 + 4*x^2 + 3)

theorem sum_of_squared_coefficients : 
  (5^2) + (10^2) + (20^2) + (0^2) + (15^2) = 750 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_squared_coefficients_l1943_194365


namespace NUMINAMATH_CALUDE_drews_age_l1943_194391

theorem drews_age (sam_current_age : ℕ) (h1 : sam_current_age = 46) :
  ∃ drew_current_age : ℕ,
    drew_current_age = 12 ∧
    sam_current_age + 5 = 3 * (drew_current_age + 5) :=
by sorry

end NUMINAMATH_CALUDE_drews_age_l1943_194391


namespace NUMINAMATH_CALUDE_square_exterior_points_diagonal_l1943_194356

-- Define the square ABCD
def square_side_length : ℝ := 15

-- Define the lengths BG, DH, AG, and CH
def BG : ℝ := 7
def DH : ℝ := 7
def AG : ℝ := 17
def CH : ℝ := 17

-- Define the theorem
theorem square_exterior_points_diagonal (A B C D G H : ℝ × ℝ) :
  let AB := square_side_length
  let AD := square_side_length
  (B.1 - G.1)^2 + (B.2 - G.2)^2 = BG^2 →
  (D.1 - H.1)^2 + (D.2 - H.2)^2 = DH^2 →
  (A.1 - G.1)^2 + (A.2 - G.2)^2 = AG^2 →
  (C.1 - H.1)^2 + (C.2 - H.2)^2 = CH^2 →
  (G.1 - H.1)^2 + (G.2 - H.2)^2 = 98 :=
by sorry


end NUMINAMATH_CALUDE_square_exterior_points_diagonal_l1943_194356


namespace NUMINAMATH_CALUDE_parabola_coefficient_sum_l1943_194371

/-- Given a parabola x = ay² + by + c passing through (6, -5) and (2, -1), prove a + b + c = -3.25 -/
theorem parabola_coefficient_sum (a b c : ℝ) : 
  (6 = a * (-5)^2 + b * (-5) + c) →
  (2 = a * (-1)^2 + b * (-1) + c) →
  a + b + c = -3.25 := by
sorry

end NUMINAMATH_CALUDE_parabola_coefficient_sum_l1943_194371


namespace NUMINAMATH_CALUDE_inequality_proof_l1943_194334

theorem inequality_proof (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) 
  (hsum : x + y + z = 1) : 
  (1/x^2 + x) * (1/y^2 + y) * (1/z^2 + z) ≥ (28/3)^3 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1943_194334


namespace NUMINAMATH_CALUDE_smaller_circle_circumference_l1943_194375

theorem smaller_circle_circumference :
  ∀ (r R s d : ℝ),
  s^2 = 784 →
  s = 2 * R →
  d = r + R →
  R = (7/3) * r →
  2 * π * r = 12 * π :=
by
  sorry

end NUMINAMATH_CALUDE_smaller_circle_circumference_l1943_194375


namespace NUMINAMATH_CALUDE_inscribed_semicircle_radius_l1943_194377

/-- Given a right triangle ABC with AC = 12 and BC = 5, the radius of the inscribed semicircle is 10/3 -/
theorem inscribed_semicircle_radius (A B C : ℝ × ℝ) : 
  let d := λ p q : ℝ × ℝ => Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)
  (d A C = 12) →
  (d B C = 5) →
  (d A B)^2 = (d A C)^2 + (d B C)^2 →
  (∃ r : ℝ, r = 10/3 ∧ 
    ∃ O : ℝ × ℝ, 
      d O A + d O B = d A B ∧
      d O C = r ∧
      ∀ P : ℝ × ℝ, d O P = r → 
        (P.1 - O.1)^2 + (P.2 - O.2)^2 = r^2 ∧
        (P.1 - A.1) * (B.2 - A.2) = (P.2 - A.2) * (B.1 - A.1)) := by
  sorry

end NUMINAMATH_CALUDE_inscribed_semicircle_radius_l1943_194377


namespace NUMINAMATH_CALUDE_sector_max_area_l1943_194388

theorem sector_max_area (R c : ℝ) (h : c > 0) :
  let perimeter := 2 * R + R * (c / R - 2)
  let area := (1 / 2) * R * (c / R - 2) * R
  ∀ R > 0, perimeter = c → area ≤ c^2 / 16 :=
sorry

end NUMINAMATH_CALUDE_sector_max_area_l1943_194388


namespace NUMINAMATH_CALUDE_optimal_plan_maximizes_profit_l1943_194389

/-- Represents the production plan for transformers --/
structure ProductionPlan where
  typeA : ℕ
  typeB : ℕ

/-- Calculates the profit for a given production plan --/
def profit (plan : ProductionPlan) : ℕ :=
  12 * plan.typeA + 10 * plan.typeB

/-- Checks if a production plan is feasible given the resource constraints --/
def isFeasible (plan : ProductionPlan) : Prop :=
  5 * plan.typeA + 3 * plan.typeB ≤ 481 ∧
  3 * plan.typeA + 2 * plan.typeB ≤ 301

/-- The optimal production plan --/
def optimalPlan : ProductionPlan :=
  { typeA := 1, typeB := 149 }

/-- Theorem stating that the optimal plan achieves the maximum profit --/
theorem optimal_plan_maximizes_profit :
  isFeasible optimalPlan ∧
  ∀ plan, isFeasible plan → profit plan ≤ profit optimalPlan :=
by sorry

#eval profit optimalPlan  -- Should output 1502

end NUMINAMATH_CALUDE_optimal_plan_maximizes_profit_l1943_194389


namespace NUMINAMATH_CALUDE_min_value_inequality_l1943_194343

theorem min_value_inequality (a b c : ℝ) 
  (h1 : 1 ≤ a) (h2 : a ≤ b) (h3 : b ≤ c) (h4 : c ≤ 5) : 
  (a - 1)^2 + (b/a - 1)^2 + (c/b - 1)^2 + (5/c - 1)^2 ≥ 4 * (Real.sqrt 10 - 1)^2 := by
  sorry

end NUMINAMATH_CALUDE_min_value_inequality_l1943_194343


namespace NUMINAMATH_CALUDE_determinant_equality_l1943_194384

theorem determinant_equality (a x y : ℝ) : 
  Matrix.det ![![1, x^2, y], ![1, a*x + y, y^2], ![1, x^2, a*x + y]] = 
    a^2*x^2 + 2*a*x*y + y^2 - a*x^3 - x*y^2 := by sorry

end NUMINAMATH_CALUDE_determinant_equality_l1943_194384


namespace NUMINAMATH_CALUDE_audrey_sleep_time_l1943_194325

/-- Given that Audrey dreamed for 2/5 of her sleep time and was not dreaming for 6 hours,
    prove that she was asleep for 10 hours. -/
theorem audrey_sleep_time :
  ∀ (total_sleep : ℝ),
  (2 / 5 : ℝ) * total_sleep + 6 = total_sleep →
  total_sleep = 10 :=
by
  sorry

end NUMINAMATH_CALUDE_audrey_sleep_time_l1943_194325


namespace NUMINAMATH_CALUDE_improper_fraction_decomposition_l1943_194317

theorem improper_fraction_decomposition (x : ℝ) :
  (6 * x^3 + 5 * x^2 + 3 * x - 4) / (x^2 + 4) = 6 * x + 5 - (21 * x + 24) / (x^2 + 4) := by
  sorry

end NUMINAMATH_CALUDE_improper_fraction_decomposition_l1943_194317


namespace NUMINAMATH_CALUDE_rectangle_perimeter_l1943_194366

/-- Given a square with side z containing a smaller square with side w,
    prove that the perimeter of a rectangle formed by the remaining area is 2z. -/
theorem rectangle_perimeter (z w : ℝ) (hz : z > 0) (hw : w > 0) (hw_lt_z : w < z) :
  2 * w + 2 * (z - w) = 2 * z := by
  sorry

end NUMINAMATH_CALUDE_rectangle_perimeter_l1943_194366


namespace NUMINAMATH_CALUDE_maze_paths_count_l1943_194308

/-- Represents a maze with specific branching structure -/
structure Maze where
  initial_branches : Nat
  subsequent_branches : Nat
  final_paths : Nat

/-- Calculates the number of unique paths through the maze -/
def count_paths (m : Maze) : Nat :=
  m.initial_branches * m.subsequent_branches.pow m.final_paths

/-- Theorem stating that a maze with given properties has 16 unique paths -/
theorem maze_paths_count :
  ∀ (m : Maze), m.initial_branches = 2 ∧ m.subsequent_branches = 2 ∧ m.final_paths = 3 →
  count_paths m = 16 := by
  sorry

#eval count_paths ⟨2, 2, 3⟩  -- Should output 16

end NUMINAMATH_CALUDE_maze_paths_count_l1943_194308


namespace NUMINAMATH_CALUDE_intersection_sum_l1943_194360

theorem intersection_sum (c d : ℝ) : 
  (3 = (1/3) * 3 + c) ∧ (3 = (1/3) * 3 + d) → c + d = 4 := by
  sorry

end NUMINAMATH_CALUDE_intersection_sum_l1943_194360


namespace NUMINAMATH_CALUDE_function_properties_l1943_194378

-- Define the function f
def f (a b c : ℝ) (x : ℝ) := a * x^2 + b * x + c

-- State the theorem
theorem function_properties
  (a b c : ℝ)
  (h_min : ∀ x : ℝ, f a b c x ≥ 0 ∧ ∃ y : ℝ, f a b c y = 0)
  (h_sym : ∀ x : ℝ, f a b c (x - 1) = f a b c (-x - 1))
  (h_bound : ∀ x ∈ Set.Ioo 0 5, x ≤ f a b c x ∧ f a b c x ≤ 2 * |x - 1| + 1) :
  (f a b c 1 = 1) ∧
  (∀ x : ℝ, f a b c x = (1/4) * (x + 1)^2) ∧
  (∃ m : ℝ, m > 1 ∧ 
    (∃ t : ℝ, ∀ x ∈ Set.Icc 1 m, f a b c (x + t) ≤ x) ∧
    (∀ m' > m, ¬∃ t : ℝ, ∀ x ∈ Set.Icc 1 m', f a b c (x + t) ≤ x) ∧
    m = 9) :=
by sorry

end NUMINAMATH_CALUDE_function_properties_l1943_194378


namespace NUMINAMATH_CALUDE_sugar_water_experiment_l1943_194386

theorem sugar_water_experiment (a b m : ℝ) (h1 : b > a) (h2 : a > 0) (h3 : m > 0) :
  let initial_concentration := a / b
  let water_added_concentration := a / (b + m)
  let sugar_added_concentration := (a + m) / b
  (water_added_concentration = a / (b + m)) ∧
  (initial_concentration > water_added_concentration) ∧
  (initial_concentration < sugar_added_concentration) := by
  sorry

end NUMINAMATH_CALUDE_sugar_water_experiment_l1943_194386


namespace NUMINAMATH_CALUDE_sum_palindromic_primes_l1943_194398

def isPrime (n : Nat) : Bool := sorry

def reverseDigits (n : Nat) : Nat := sorry

def isPalindromicPrime (n : Nat) : Bool :=
  isPrime n ∧ isPrime (reverseDigits n)

def palindromicPrimes : List Nat :=
  (List.range 90).filter (fun n => n ≥ 10 ∧ isPalindromicPrime n)

theorem sum_palindromic_primes :
  palindromicPrimes.sum = 429 := by sorry

end NUMINAMATH_CALUDE_sum_palindromic_primes_l1943_194398


namespace NUMINAMATH_CALUDE_ball_max_height_l1943_194319

/-- The height of the ball as a function of time -/
def h (t : ℝ) : ℝ := -16 * t^2 + 80 * t + 35

/-- The maximum height reached by the ball -/
theorem ball_max_height : ∃ (t : ℝ), ∀ (s : ℝ), h s ≤ h t ∧ h t = 135 := by
  sorry

end NUMINAMATH_CALUDE_ball_max_height_l1943_194319


namespace NUMINAMATH_CALUDE_weighted_inequality_l1943_194318

theorem weighted_inequality (a b c : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) (sum_one : a + b + c = 1) :
  (a + 2*a*b + 2*a*c + b*c)^a * (b + 2*b*c + 2*b*a + c*a)^b * (c + 2*c*a + 2*c*b + a*b)^c ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_weighted_inequality_l1943_194318


namespace NUMINAMATH_CALUDE_band_size_correct_l1943_194396

/-- The number of flutes that tried out -/
def num_flutes : ℕ := 20

/-- The percentage of flutes that got in -/
def flute_acceptance_rate : ℚ := 4/5

/-- The number of clarinets that tried out -/
def num_clarinets : ℕ := 30

/-- The percentage of clarinets that got in -/
def clarinet_acceptance_rate : ℚ := 1/2

/-- The number of trumpets that tried out -/
def num_trumpets : ℕ := 60

/-- The percentage of trumpets that got in -/
def trumpet_acceptance_rate : ℚ := 1/3

/-- The number of pianists that tried out -/
def num_pianists : ℕ := 20

/-- The percentage of pianists that got in -/
def pianist_acceptance_rate : ℚ := 1/10

/-- The total number of people in the band -/
def total_in_band : ℕ := 53

theorem band_size_correct : 
  (num_flutes : ℚ) * flute_acceptance_rate +
  (num_clarinets : ℚ) * clarinet_acceptance_rate +
  (num_trumpets : ℚ) * trumpet_acceptance_rate +
  (num_pianists : ℚ) * pianist_acceptance_rate = total_in_band := by sorry

end NUMINAMATH_CALUDE_band_size_correct_l1943_194396


namespace NUMINAMATH_CALUDE_cheese_division_possible_l1943_194328

/-- Represents the state of the cheese pieces -/
structure CheeseState where
  piece1 : ℕ
  piece2 : ℕ
  piece3 : ℕ

/-- Represents a single cut operation -/
inductive Cut
  | cut12 : Cut  -- Cut 1g from piece1 and piece2
  | cut13 : Cut  -- Cut 1g from piece1 and piece3
  | cut23 : Cut  -- Cut 1g from piece2 and piece3

/-- Applies a single cut to a CheeseState -/
def applyCut (state : CheeseState) (cut : Cut) : CheeseState :=
  match cut with
  | Cut.cut12 => ⟨state.piece1 - 1, state.piece2 - 1, state.piece3⟩
  | Cut.cut13 => ⟨state.piece1 - 1, state.piece2, state.piece3 - 1⟩
  | Cut.cut23 => ⟨state.piece1, state.piece2 - 1, state.piece3 - 1⟩

/-- Checks if all pieces in a CheeseState are equal -/
def allEqual (state : CheeseState) : Prop :=
  state.piece1 = state.piece2 ∧ state.piece2 = state.piece3

/-- The theorem to be proved -/
theorem cheese_division_possible : ∃ (cuts : List Cut), 
  let finalState := cuts.foldl applyCut ⟨5, 8, 11⟩
  allEqual finalState ∧ finalState.piece1 ≥ 0 := by
  sorry


end NUMINAMATH_CALUDE_cheese_division_possible_l1943_194328


namespace NUMINAMATH_CALUDE_max_sum_cubes_l1943_194304

theorem max_sum_cubes (a b c d e : ℝ) (h : a^2 + b^2 + c^2 + d^2 + e^2 = 5) :
  ∃ (max : ℝ), max = 5 * Real.sqrt 5 ∧ 
  a^3 + b^3 + c^3 + d^3 + e^3 ≤ max ∧
  ∃ (a' b' c' d' e' : ℝ), a'^2 + b'^2 + c'^2 + d'^2 + e'^2 = 5 ∧
                           a'^3 + b'^3 + c'^3 + d'^3 + e'^3 = max :=
by sorry

end NUMINAMATH_CALUDE_max_sum_cubes_l1943_194304


namespace NUMINAMATH_CALUDE_sqrt_product_simplification_l1943_194327

theorem sqrt_product_simplification (p : ℝ) : 
  Real.sqrt (12 * p) * Real.sqrt (20 * p) * Real.sqrt (15 * p^2) = 60 * p^2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_product_simplification_l1943_194327


namespace NUMINAMATH_CALUDE_bucket_capacity_l1943_194329

theorem bucket_capacity : ∀ (x : ℚ), 
  (13 * x = 91 * 6) → x = 42 := by
  sorry

end NUMINAMATH_CALUDE_bucket_capacity_l1943_194329
