import Mathlib

namespace NUMINAMATH_CALUDE_complex_absolute_value_l159_15988

/-- Given a complex number z such that (1 + 2i) / z = 1 + i,
    prove that the absolute value of z is equal to √10 / 2. -/
theorem complex_absolute_value (z : ℂ) (h : (1 + 2 * Complex.I) / z = 1 + Complex.I) :
  Complex.abs z = Real.sqrt 10 / 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_absolute_value_l159_15988


namespace NUMINAMATH_CALUDE_continuity_definition_relation_l159_15955

-- Define a real-valued function
variable (f : ℝ → ℝ)
-- Define a point x₀
variable (x₀ : ℝ)

-- Define what it means for f to be defined at x₀
def is_defined_at (f : ℝ → ℝ) (x₀ : ℝ) : Prop :=
  ∃ y : ℝ, f x₀ = y

-- State the theorem
theorem continuity_definition_relation :
  (ContinuousAt f x₀ → is_defined_at f x₀) ∧
  ¬(is_defined_at f x₀ → ContinuousAt f x₀) :=
sorry

end NUMINAMATH_CALUDE_continuity_definition_relation_l159_15955


namespace NUMINAMATH_CALUDE_vector_operation_l159_15909

def a : Fin 2 → ℝ := ![3, 2]
def b : Fin 2 → ℝ := ![0, -1]

theorem vector_operation :
  (3 • b - a) = ![(-3 : ℝ), -5] := by sorry

end NUMINAMATH_CALUDE_vector_operation_l159_15909


namespace NUMINAMATH_CALUDE_rectangleEnclosures_eq_100_l159_15957

/-- The number of ways to choose 4 lines (2 horizontal and 2 vertical) from 5 horizontal and 5 vertical lines to enclose a rectangular region. -/
def rectangleEnclosures : ℕ :=
  let horizontalLines := 5
  let verticalLines := 5
  let horizontalChoices := Nat.choose horizontalLines 2
  let verticalChoices := Nat.choose verticalLines 2
  horizontalChoices * verticalChoices

/-- Theorem stating that the number of ways to choose 4 lines to enclose a rectangular region is 100. -/
theorem rectangleEnclosures_eq_100 : rectangleEnclosures = 100 := by
  sorry

end NUMINAMATH_CALUDE_rectangleEnclosures_eq_100_l159_15957


namespace NUMINAMATH_CALUDE_ny_striploin_cost_l159_15995

theorem ny_striploin_cost (total_bill : ℝ) (tax_rate : ℝ) (wine_cost : ℝ) (gratuities : ℝ) :
  total_bill = 140 →
  tax_rate = 0.1 →
  wine_cost = 10 →
  gratuities = 41 →
  ∃ (striploin_cost : ℝ), abs (striploin_cost - 71.82) < 0.01 :=
by
  sorry

end NUMINAMATH_CALUDE_ny_striploin_cost_l159_15995


namespace NUMINAMATH_CALUDE_emily_calculation_l159_15948

theorem emily_calculation (x y z : ℝ) 
  (h1 : 2*x - 3*y + z = 14) 
  (h2 : 2*x - 3*y - z = 6) : 
  2*x - 3*y = 10 := by
sorry

end NUMINAMATH_CALUDE_emily_calculation_l159_15948


namespace NUMINAMATH_CALUDE_remainder_theorem_l159_15946

theorem remainder_theorem : ∃ q : ℕ, 2^222 + 222 = q * (2^111 + 2^56 + 1) + 218 := by
  sorry

end NUMINAMATH_CALUDE_remainder_theorem_l159_15946


namespace NUMINAMATH_CALUDE_a_values_l159_15928

-- Define the sets M and N
def M : Set ℝ := {x | x^2 + x - 6 = 0}
def N (a : ℝ) : Set ℝ := {x | a * x + 2 = 0}

-- Define the set of possible values for a
def possible_a : Set ℝ := {-1, 0, 2/3}

-- Statement to prove
theorem a_values (a : ℝ) : (N a ⊆ M) ↔ a ∈ possible_a := by sorry

end NUMINAMATH_CALUDE_a_values_l159_15928


namespace NUMINAMATH_CALUDE_divisibility_implication_l159_15960

theorem divisibility_implication (a b : ℤ) :
  (∃ k : ℤ, a^2 + 9*a*b + b^2 = 11*k) → (∃ m : ℤ, a^2 - b^2 = 11*m) := by
  sorry

end NUMINAMATH_CALUDE_divisibility_implication_l159_15960


namespace NUMINAMATH_CALUDE_triangle_similarity_fc_value_l159_15975

/-- Given a triangle ADE with point C on AD and point B on AC, prove that FC = 13.875 -/
theorem triangle_similarity_fc_value (DC CB AD AB ED : ℝ) : 
  DC = 10 →
  CB = 9 →
  AB = (1/3) * AD →
  ED = (3/4) * AD →
  ∃ (FC : ℝ), FC = 13.875 := by
  sorry

end NUMINAMATH_CALUDE_triangle_similarity_fc_value_l159_15975


namespace NUMINAMATH_CALUDE_ball_placement_count_l159_15938

-- Define the number of balls
def num_balls : ℕ := 3

-- Define the number of available boxes (excluding box 1)
def num_boxes : ℕ := 3

-- Theorem statement
theorem ball_placement_count : (num_boxes ^ num_balls) = 27 := by
  sorry

end NUMINAMATH_CALUDE_ball_placement_count_l159_15938


namespace NUMINAMATH_CALUDE_crayons_count_l159_15968

/-- The number of rows of crayons --/
def num_rows : ℕ := 7

/-- The number of crayons in each row --/
def crayons_per_row : ℕ := 30

/-- The total number of crayons --/
def total_crayons : ℕ := num_rows * crayons_per_row

theorem crayons_count : total_crayons = 210 := by
  sorry

end NUMINAMATH_CALUDE_crayons_count_l159_15968


namespace NUMINAMATH_CALUDE_sqrt_123454321_l159_15959

theorem sqrt_123454321 : Int.sqrt 123454321 = 11111 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_123454321_l159_15959


namespace NUMINAMATH_CALUDE_b_share_is_3000_l159_15980

/-- Proves that B's share is 3000 when money is distributed in the proportion 6:3:5:4 and C gets 1000 more than D -/
theorem b_share_is_3000 (total : ℕ) (a b c d : ℕ) : 
  a + b + c + d = total →  -- Sum of all shares equals the total
  6 * b = 3 * a →          -- A:B proportion is 6:3
  5 * b = 5 * a →          -- B:C proportion is 3:5
  4 * b = 3 * d →          -- B:D proportion is 3:4
  c = d + 1000 →           -- C gets 1000 more than D
  b = 3000 := by
sorry

end NUMINAMATH_CALUDE_b_share_is_3000_l159_15980


namespace NUMINAMATH_CALUDE_angle_C_is_pi_over_3_side_c_is_sqrt_6_l159_15910

-- Define the triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the conditions
def satisfiesConditions (t : Triangle) : Prop :=
  t.a + t.b = Real.sqrt 3 * t.c ∧
  2 * (Real.sin t.C)^2 = 3 * Real.sin t.A * Real.sin t.B

-- Define the area condition
def hasAreaSqrt3 (t : Triangle) : Prop :=
  1/2 * t.a * t.b * Real.sin t.C = Real.sqrt 3

-- Theorem 1
theorem angle_C_is_pi_over_3 (t : Triangle) 
  (h : satisfiesConditions t) : t.C = π/3 :=
sorry

-- Theorem 2
theorem side_c_is_sqrt_6 (t : Triangle) 
  (h1 : satisfiesConditions t) 
  (h2 : hasAreaSqrt3 t) : t.c = Real.sqrt 6 :=
sorry

end NUMINAMATH_CALUDE_angle_C_is_pi_over_3_side_c_is_sqrt_6_l159_15910


namespace NUMINAMATH_CALUDE_farm_legs_count_l159_15926

/-- The number of legs for a given animal type -/
def legs_per_animal (animal : String) : ℕ :=
  match animal with
  | "cow" => 4
  | "duck" => 2
  | _ => 0

/-- The total number of animals in the farm -/
def total_animals : ℕ := 15

/-- The number of cows in the farm -/
def num_cows : ℕ := 6

/-- The number of ducks in the farm -/
def num_ducks : ℕ := total_animals - num_cows

theorem farm_legs_count : 
  legs_per_animal "cow" * num_cows + legs_per_animal "duck" * num_ducks = 42 := by
sorry

end NUMINAMATH_CALUDE_farm_legs_count_l159_15926


namespace NUMINAMATH_CALUDE_sufficient_material_for_box_l159_15970

/-- A rectangular box with integer dimensions -/
structure Box where
  length : ℕ
  width : ℕ
  height : ℕ

/-- Calculate the volume of a box -/
def volume (b : Box) : ℕ :=
  b.length * b.width * b.height

/-- Calculate the surface area of a box -/
def surface_area (b : Box) : ℕ :=
  2 * (b.length * b.width + b.length * b.height + b.width * b.height)

/-- Theorem: There exists a box with volume at least 1995 and surface area exactly 958 -/
theorem sufficient_material_for_box : 
  ∃ (b : Box), volume b ≥ 1995 ∧ surface_area b = 958 :=
by
  sorry

end NUMINAMATH_CALUDE_sufficient_material_for_box_l159_15970


namespace NUMINAMATH_CALUDE_problem_solution_l159_15964

theorem problem_solution (a b c d : ℝ) :
  a^2 + b^2 + c^2 + 2 = d + Real.sqrt (a + b + c - 2*d) →
  d = -1/8 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l159_15964


namespace NUMINAMATH_CALUDE_four_monotonic_intervals_condition_l159_15913

/-- A function f(x) defined by a quadratic expression inside an absolute value. -/
def f (m : ℝ) (x : ℝ) : ℝ := |m * x^2 - (2*m + 1) * x + (m + 2)|

/-- The property of having exactly four monotonic intervals. -/
def has_four_monotonic_intervals (g : ℝ → ℝ) : Prop := sorry

/-- The main theorem stating the conditions on m for f to have exactly four monotonic intervals. -/
theorem four_monotonic_intervals_condition (m : ℝ) :
  has_four_monotonic_intervals (f m) ↔ m < (1/4 : ℝ) ∧ m ≠ 0 :=
sorry

end NUMINAMATH_CALUDE_four_monotonic_intervals_condition_l159_15913


namespace NUMINAMATH_CALUDE_first_account_interest_rate_l159_15998

/-- Proves that the interest rate of the first account is 0.02 given the problem conditions --/
theorem first_account_interest_rate :
  ∀ (r : ℝ),
    r > 0 →
    r < 1 →
    1000 * r + 1800 * 0.04 = 92 →
    r = 0.02 := by
  sorry

end NUMINAMATH_CALUDE_first_account_interest_rate_l159_15998


namespace NUMINAMATH_CALUDE_triangle_property_l159_15986

-- Define the triangle ABC
structure Triangle :=
  (A B C : ℝ)  -- Angles
  (a b c : ℝ)  -- Sides opposite to A, B, C respectively
  (h1 : A + B + C = π)  -- Sum of angles in a triangle
  (h2 : a > 0 ∧ b > 0 ∧ c > 0)  -- Positive side lengths
  (h3 : b < c)  -- Given condition

-- Define the existence of points E and F
def points_exist (t : Triangle) : Prop :=
  ∃ E F : ℝ, 
    E > 0 ∧ F > 0 ∧
    E ≤ t.c ∧ F ≤ t.b ∧
    E = F ∧
    ∃ D : ℝ, D > 0 ∧ D < t.a ∧
    (t.A / 2 = Real.arctan (D / E) + Real.arctan (D / F))

-- Theorem statement
theorem triangle_property (t : Triangle) (h : points_exist t) :
  t.A / 2 ≤ t.B ∧ (t.a * t.c) / (t.b + t.c) = t.c * (t.a / (t.b + t.c)) :=
sorry

end NUMINAMATH_CALUDE_triangle_property_l159_15986


namespace NUMINAMATH_CALUDE_functional_equation_solution_l159_15952

/-- A function satisfying the given functional equation -/
def FunctionalEquation (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (x + y) = f x + f y + 2 * x * y

theorem functional_equation_solution :
  ∀ f : ℝ → ℝ, FunctionalEquation f → f 1 = 2 → f (-2) = 2 := by
  sorry

end NUMINAMATH_CALUDE_functional_equation_solution_l159_15952


namespace NUMINAMATH_CALUDE_second_group_women_l159_15911

/-- The work rate of one man -/
def man_rate : ℝ := sorry

/-- The work rate of one woman -/
def woman_rate : ℝ := sorry

/-- The number of women in the second group -/
def x : ℕ := sorry

/-- The work rate of 3 men and 8 women equals the work rate of 6 men and x women -/
axiom work_rate_equality : 3 * man_rate + 8 * woman_rate = 6 * man_rate + x * woman_rate

/-- The work rate of 4 men and 5 women is 0.9285714285714286 times the work rate of 3 men and 8 women -/
axiom work_rate_fraction : 4 * man_rate + 5 * woman_rate = 0.9285714285714286 * (3 * man_rate + 8 * woman_rate)

/-- The number of women in the second group is 14 -/
theorem second_group_women : x = 14 := by sorry

end NUMINAMATH_CALUDE_second_group_women_l159_15911


namespace NUMINAMATH_CALUDE_sons_age_l159_15951

theorem sons_age (son_age father_age : ℕ) : 
  father_age = son_age + 24 →
  father_age + 2 = 2 * (son_age + 2) →
  son_age = 22 := by
sorry

end NUMINAMATH_CALUDE_sons_age_l159_15951


namespace NUMINAMATH_CALUDE_max_a_for_monotonic_f_l159_15940

/-- Given a function f(x) = x^3 - ax that is monotonically increasing on [1, +∞),
    the maximum value of a is 3. -/
theorem max_a_for_monotonic_f (a : ℝ) : 
  (∀ x ≥ 1, ∀ y ≥ x, (x^3 - a*x) ≤ (y^3 - a*y)) → a ≤ 3 :=
sorry

end NUMINAMATH_CALUDE_max_a_for_monotonic_f_l159_15940


namespace NUMINAMATH_CALUDE_grade12_population_l159_15900

/-- Represents the number of students in each grade (10, 11, 12) -/
structure GradePopulation where
  grade10 : ℕ
  grade11 : ℕ
  grade12 : ℕ

/-- The ratio of students in grades 10, 11, and 12 -/
def gradeRatio : GradePopulation := ⟨10, 8, 7⟩

/-- The number of students sampled -/
def sampleSize : ℕ := 200

/-- The sampling probability for each student -/
def samplingProbability : ℚ := 1/5

theorem grade12_population (pop : GradePopulation) :
  pop.grade10 / gradeRatio.grade10 = pop.grade11 / gradeRatio.grade11 ∧
  pop.grade11 / gradeRatio.grade11 = pop.grade12 / gradeRatio.grade12 ∧
  pop.grade10 + pop.grade11 + pop.grade12 = sampleSize / samplingProbability →
  pop.grade12 = 280 := by
sorry

end NUMINAMATH_CALUDE_grade12_population_l159_15900


namespace NUMINAMATH_CALUDE_value_range_of_f_l159_15914

def f (x : ℝ) : ℝ := x^2 - 4*x + 3

theorem value_range_of_f :
  ∀ y ∈ Set.Icc (-1 : ℝ) 3, ∃ x ∈ Set.Icc 0 3, f x = y ∧
  ∀ x ∈ Set.Icc 0 3, f x ∈ Set.Icc (-1 : ℝ) 3 :=
sorry

end NUMINAMATH_CALUDE_value_range_of_f_l159_15914


namespace NUMINAMATH_CALUDE_sqrt_expression_equality_l159_15979

theorem sqrt_expression_equality (t : ℝ) : 
  Real.sqrt (t^6 + t^4 + t^2) = |t| * Real.sqrt (t^4 + t^2 + 1) := by
  sorry

end NUMINAMATH_CALUDE_sqrt_expression_equality_l159_15979


namespace NUMINAMATH_CALUDE_sample_size_is_sampled_athletes_l159_15942

/-- Represents the total number of athletes in the sports meeting -/
def total_athletes : ℕ := 1000

/-- Represents the number of athletes sampled -/
def sampled_athletes : ℕ := 100

/-- Theorem stating that the sample size is equal to the number of sampled athletes -/
theorem sample_size_is_sampled_athletes :
  sampled_athletes = 100 :=
by sorry

end NUMINAMATH_CALUDE_sample_size_is_sampled_athletes_l159_15942


namespace NUMINAMATH_CALUDE_square_minus_self_divisible_by_two_l159_15969

theorem square_minus_self_divisible_by_two (n : ℕ) : 
  2 ∣ (n^2 - n) := by sorry

end NUMINAMATH_CALUDE_square_minus_self_divisible_by_two_l159_15969


namespace NUMINAMATH_CALUDE_profitable_after_three_years_l159_15939

/-- Represents the financial data for the communication equipment --/
structure EquipmentData where
  initialInvestment : ℕ
  firstYearExpenses : ℕ
  annualExpenseIncrease : ℕ
  annualProfit : ℕ

/-- Calculates the cumulative profit after a given number of years --/
def cumulativeProfit (data : EquipmentData) (years : ℕ) : ℤ :=
  (data.annualProfit * years : ℤ) - 
  (data.initialInvestment + data.firstYearExpenses * years + 
   data.annualExpenseIncrease * (years * (years - 1) / 2) : ℤ)

/-- Theorem stating that the equipment becomes profitable after 3 years --/
theorem profitable_after_three_years (data : EquipmentData) 
  (h1 : data.initialInvestment = 980000)
  (h2 : data.firstYearExpenses = 120000)
  (h3 : data.annualExpenseIncrease = 40000)
  (h4 : data.annualProfit = 500000) :
  cumulativeProfit data 3 > 0 ∧ cumulativeProfit data 2 ≤ 0 := by
  sorry

#check profitable_after_three_years

end NUMINAMATH_CALUDE_profitable_after_three_years_l159_15939


namespace NUMINAMATH_CALUDE_mia_wall_paint_area_l159_15925

/-- The area to be painted on Mia's wall --/
def areaToBePainted (wallHeight wallLength unPaintedWidth unPaintedHeight : ℝ) : ℝ :=
  wallHeight * wallLength - unPaintedWidth * unPaintedHeight

/-- Theorem stating the area Mia needs to paint --/
theorem mia_wall_paint_area :
  areaToBePainted 10 15 3 5 = 135 := by
  sorry

end NUMINAMATH_CALUDE_mia_wall_paint_area_l159_15925


namespace NUMINAMATH_CALUDE_inconsistent_equations_l159_15901

theorem inconsistent_equations : ¬∃ (x y S : ℝ), (x + y = S) ∧ (x + 3*y = 1) ∧ (x + 2*y = 10) := by
  sorry

end NUMINAMATH_CALUDE_inconsistent_equations_l159_15901


namespace NUMINAMATH_CALUDE_ratio_sum_problem_l159_15974

theorem ratio_sum_problem (a b c d : ℝ) (h_positive : a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0) 
  (h_ratio : b = 2*a ∧ c = 4*a ∧ d = 5*a) 
  (h_sum_squares : a^2 + b^2 + c^2 + d^2 = 2540) : 
  a + b + c + d = 12 * Real.sqrt 55 := by
  sorry

end NUMINAMATH_CALUDE_ratio_sum_problem_l159_15974


namespace NUMINAMATH_CALUDE_opposite_face_of_A_is_B_l159_15936

/-- Represents the letters on the cube faces -/
inductive CubeLetter
  | A | B | V | G | D | E

/-- Represents a face of the cube -/
structure CubeFace where
  letter : CubeLetter

/-- Represents the cube -/
structure Cube where
  faces : Finset CubeFace
  face_count : faces.card = 6

/-- Represents a perspective of the cube showing three visible faces -/
structure CubePerspective where
  visible_faces : Finset CubeFace
  visible_count : visible_faces.card = 3

/-- Defines the opposite face relation -/
def opposite_face (c : Cube) (f1 f2 : CubeFace) : Prop :=
  f1 ∈ c.faces ∧ f2 ∈ c.faces ∧ f1 ≠ f2 ∧ 
  ∀ (p : CubePerspective), ¬(f1 ∈ p.visible_faces ∧ f2 ∈ p.visible_faces)

theorem opposite_face_of_A_is_B 
  (c : Cube) 
  (p1 p2 p3 : CubePerspective) 
  (hA : ∃ (fA : CubeFace), fA ∈ c.faces ∧ fA.letter = CubeLetter.A)
  (hB : ∃ (fB : CubeFace), fB ∈ c.faces ∧ fB.letter = CubeLetter.B)
  (h_perspectives : 
    (∃ (f1 f2 : CubeFace), f1 ∈ p1.visible_faces ∧ f2 ∈ p1.visible_faces ∧ 
      f1.letter = CubeLetter.A ∧ f2.letter = CubeLetter.B) ∧
    (∃ (f1 f2 : CubeFace), f1 ∈ p2.visible_faces ∧ f2 ∈ p2.visible_faces ∧ 
      f1.letter = CubeLetter.B) ∧
    (∃ (f1 f2 : CubeFace), f1 ∈ p3.visible_faces ∧ f2 ∈ p3.visible_faces ∧ 
      f1.letter = CubeLetter.A)) :
  ∃ (fA fB : CubeFace), 
    fA.letter = CubeLetter.A ∧ 
    fB.letter = CubeLetter.B ∧ 
    opposite_face c fA fB :=
  sorry

end NUMINAMATH_CALUDE_opposite_face_of_A_is_B_l159_15936


namespace NUMINAMATH_CALUDE_fraction_simplification_l159_15905

theorem fraction_simplification (a : ℝ) (h : a ≠ 2) :
  (3 - a) / (a - 2) + 1 = 1 / (a - 2) :=
by sorry

end NUMINAMATH_CALUDE_fraction_simplification_l159_15905


namespace NUMINAMATH_CALUDE_sean_bedroom_bulbs_l159_15917

/-- The number of light bulbs Sean needs to replace in his bedroom. -/
def bedroom_bulbs : ℕ := 2

/-- The number of light bulbs Sean needs to replace in the bathroom. -/
def bathroom_bulbs : ℕ := 1

/-- The number of light bulbs Sean needs to replace in the kitchen. -/
def kitchen_bulbs : ℕ := 1

/-- The number of light bulbs Sean needs to replace in the basement. -/
def basement_bulbs : ℕ := 4

/-- The number of light bulbs per pack. -/
def bulbs_per_pack : ℕ := 2

/-- The number of packs Sean needs. -/
def packs_needed : ℕ := 6

/-- The total number of light bulbs Sean needs. -/
def total_bulbs : ℕ := packs_needed * bulbs_per_pack

theorem sean_bedroom_bulbs :
  bedroom_bulbs + bathroom_bulbs + kitchen_bulbs + basement_bulbs +
  (bedroom_bulbs + bathroom_bulbs + kitchen_bulbs + basement_bulbs) / 2 = total_bulbs :=
by sorry

end NUMINAMATH_CALUDE_sean_bedroom_bulbs_l159_15917


namespace NUMINAMATH_CALUDE_divisibility_implication_l159_15906

theorem divisibility_implication (u v : ℤ) : 
  (9 ∣ u^2 + u*v + v^2) → (3 ∣ u) ∧ (3 ∣ v) := by
  sorry

end NUMINAMATH_CALUDE_divisibility_implication_l159_15906


namespace NUMINAMATH_CALUDE_unique_solution_for_equation_l159_15923

/-- Sum of digits function -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- Main theorem -/
theorem unique_solution_for_equation :
  ∀ (m n : ℕ), n * (n + 1) = 3^m + sum_of_digits n + 1182 → m = 0 ∧ n = 34 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_for_equation_l159_15923


namespace NUMINAMATH_CALUDE_fifth_term_of_special_sequence_l159_15904

/-- A sequence where each term after the first is 1/4 of the sum of the term before it and the term after it -/
def SpecialSequence (a : ℕ → ℚ) : Prop :=
  ∀ n : ℕ, n > 0 → a (n + 1) = (1 : ℚ) / 4 * (a n + a (n + 2))

theorem fifth_term_of_special_sequence
  (a : ℕ → ℚ)
  (h_seq : SpecialSequence a)
  (h_first : a 1 = 2)
  (h_fourth : a 4 = 50) :
  a 5 = 2798 / 15 := by
sorry

end NUMINAMATH_CALUDE_fifth_term_of_special_sequence_l159_15904


namespace NUMINAMATH_CALUDE_select_four_from_eighteen_l159_15954

theorem select_four_from_eighteen (n m : ℕ) : n = 18 ∧ m = 4 → Nat.choose n m = 3060 := by
  sorry

end NUMINAMATH_CALUDE_select_four_from_eighteen_l159_15954


namespace NUMINAMATH_CALUDE_q_div_p_equals_225_l159_15943

/-- The number of cards in the box -/
def total_cards : ℕ := 50

/-- The range of numbers on the cards -/
def number_range : ℕ := 10

/-- The number of cards for each number -/
def cards_per_number : ℕ := 5

/-- The number of cards drawn -/
def drawn_cards : ℕ := 5

/-- The probability that all drawn cards have the same number -/
def p : ℚ := (number_range : ℚ) / Nat.choose total_cards drawn_cards

/-- The probability that 4 drawn cards have number a and 1 card has number b (b ≠ a) -/
def q : ℚ := (2250 : ℚ) / Nat.choose total_cards drawn_cards

/-- The main theorem stating that q/p = 225 -/
theorem q_div_p_equals_225 : q / p = 225 := by sorry

end NUMINAMATH_CALUDE_q_div_p_equals_225_l159_15943


namespace NUMINAMATH_CALUDE_wage_payment_theorem_l159_15916

/-- Represents the daily wage of a worker -/
structure DailyWage where
  amount : ℝ
  amount_pos : amount > 0

/-- Represents a sum of money -/
def SumOfMoney : Type := ℝ

/-- Given two workers a and b, and a sum of money S, 
    prove that if S can pay a's wages for 21 days and 
    both a and b's wages for 12 days, then S can pay 
    b's wages for 28 days -/
theorem wage_payment_theorem 
  (a b : DailyWage) 
  (S : SumOfMoney) 
  (h1 : S = 21 * a.amount)
  (h2 : S = 12 * (a.amount + b.amount)) :
  S = 28 * b.amount := by
  sorry


end NUMINAMATH_CALUDE_wage_payment_theorem_l159_15916


namespace NUMINAMATH_CALUDE_parabola_focus_directrix_l159_15919

/-- For a parabola y = ax^2 where a > 0, if the distance from the focus to the directrix is 2, then a = 1/4 -/
theorem parabola_focus_directrix (a : ℝ) (h1 : a > 0) : 
  (∃ (f d : ℝ), ∀ (x y : ℝ), y = a * x^2 ∧ |f - d| = 2) → a = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_parabola_focus_directrix_l159_15919


namespace NUMINAMATH_CALUDE_consecutive_numbers_sum_l159_15963

theorem consecutive_numbers_sum (n : ℕ) : 
  (n + (n + 1) + (n + 2) + (n + 3) + (n + 4) = 60) → n = 10 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_numbers_sum_l159_15963


namespace NUMINAMATH_CALUDE_system_solution_l159_15990

-- Define the system of equations
def equation1 (x y : ℚ) : Prop := 2 * x - 3 * y = 5
def equation2 (x y : ℚ) : Prop := 4 * x + y = 9

-- Define the solution
def solution : ℚ × ℚ := (16/7, -1/7)

-- Theorem statement
theorem system_solution :
  let (x, y) := solution
  equation1 x y ∧ equation2 x y := by sorry

end NUMINAMATH_CALUDE_system_solution_l159_15990


namespace NUMINAMATH_CALUDE_vegetable_planting_methods_l159_15984

theorem vegetable_planting_methods (n : Nat) (k : Nat) (m : Nat) : 
  n = 4 ∧ k = 3 ∧ m = 3 → 
  (Nat.choose (n - 1) (k - 1)) * (Nat.factorial k) = 18 :=
by
  sorry

end NUMINAMATH_CALUDE_vegetable_planting_methods_l159_15984


namespace NUMINAMATH_CALUDE_interest_difference_l159_15945

/-- Calculate the difference between the principal and simple interest -/
theorem interest_difference (principal : ℝ) (rate : ℝ) (time : ℝ) :
  principal = 9200 ∧ rate = 12 ∧ time = 3 →
  principal - (principal * rate * time / 100) = 5888 := by
sorry

end NUMINAMATH_CALUDE_interest_difference_l159_15945


namespace NUMINAMATH_CALUDE_cost_price_equation_l159_15903

/-- The cost price of a watch satisfying the given conditions -/
def cost_price : ℝ := 
  let C : ℝ := 2070.31
  C

/-- Theorem stating the equation that the cost price must satisfy -/
theorem cost_price_equation : 
  3 * (0.925 * cost_price + 265) = 3 * cost_price * 1.053 := by
  sorry

end NUMINAMATH_CALUDE_cost_price_equation_l159_15903


namespace NUMINAMATH_CALUDE_arrangement_count_l159_15922

/-- Represents the number of different seed types -/
def num_seed_types : ℕ := 5

/-- Represents the number of experimental fields -/
def num_fields : ℕ := 5

/-- Represents the number of seed types that can be placed at the ends -/
def num_end_seeds : ℕ := 3

/-- Represents the number of positions for the A-B pair -/
def num_ab_positions : ℕ := 3

/-- Calculates the number of ways to arrange seeds under the given conditions -/
def calculate_arrangements : ℕ :=
  (num_end_seeds * (num_end_seeds - 1)) * -- Arrangements for the ends
  (num_ab_positions * 2)                  -- Arrangements for A-B pair and remaining seed

/-- Theorem stating that the number of arrangement methods is 24 -/
theorem arrangement_count : calculate_arrangements = 24 := by
  sorry

end NUMINAMATH_CALUDE_arrangement_count_l159_15922


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l159_15933

theorem sqrt_equation_solution (x : ℝ) : Real.sqrt (5 + Real.sqrt x) = 4 → x = 121 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l159_15933


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_algebraic_expression_value_l159_15966

-- Part 1
theorem quadratic_equation_roots (x : ℝ) :
  x^2 - 4*x - 3 = 0 ↔ x = 2 + Real.sqrt 7 ∨ x = 2 - Real.sqrt 7 := by
  sorry

-- Part 2
theorem algebraic_expression_value (a : ℝ) :
  a^2 = 3*a + 10 → (a + 4) * (a - 4) - 3 * (a - 1) = -3 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_algebraic_expression_value_l159_15966


namespace NUMINAMATH_CALUDE_complex_number_quadrant_l159_15987

theorem complex_number_quadrant : 
  let z : ℂ := (2 + 3*I) / (1 + 2*I)
  (z.re > 0 ∧ z.im < 0) := by sorry

end NUMINAMATH_CALUDE_complex_number_quadrant_l159_15987


namespace NUMINAMATH_CALUDE_greater_number_proof_l159_15902

theorem greater_number_proof (x y : ℝ) 
  (sum_eq : x + y = 30)
  (diff_eq : x - y = 6)
  (prod_eq : x * y = 216) :
  max x y = 18 := by
sorry

end NUMINAMATH_CALUDE_greater_number_proof_l159_15902


namespace NUMINAMATH_CALUDE_insurance_cost_decade_l159_15971

/-- Benjamin's yearly car insurance cost in dollars -/
def yearly_cost : ℕ := 3000

/-- Number of years in a decade -/
def decade : ℕ := 10

/-- Theorem: Benjamin's car insurance cost over a decade -/
theorem insurance_cost_decade : yearly_cost * decade = 30000 := by
  sorry

end NUMINAMATH_CALUDE_insurance_cost_decade_l159_15971


namespace NUMINAMATH_CALUDE_stability_comparison_l159_15937

/-- Represents a student's test performance -/
structure StudentPerformance where
  average_score : ℝ
  variance : ℝ

/-- Determines if the first student's performance is more stable than the second -/
def more_stable (student1 student2 : StudentPerformance) : Prop :=
  student1.variance < student2.variance

theorem stability_comparison 
  (student_A student_B : StudentPerformance)
  (h_same_average : student_A.average_score = student_B.average_score)
  (h_A_variance : student_A.variance = 51)
  (h_B_variance : student_B.variance = 12) :
  more_stable student_B student_A :=
sorry

end NUMINAMATH_CALUDE_stability_comparison_l159_15937


namespace NUMINAMATH_CALUDE_dog_toy_discount_l159_15915

/-- Proves that the discount on the second toy in each pair is $6.00 given the conditions --/
theorem dog_toy_discount (toy_price : ℝ) (num_toys : ℕ) (total_spent : ℝ) 
  (h1 : toy_price = 12)
  (h2 : num_toys = 4)
  (h3 : total_spent = 36) :
  (toy_price * num_toys - total_spent) / 2 = 6 := by
sorry

end NUMINAMATH_CALUDE_dog_toy_discount_l159_15915


namespace NUMINAMATH_CALUDE_possible_values_of_c_l159_15950

theorem possible_values_of_c (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a ≠ b)
  (h : a^3 - b^3 = a^2 - b^2) :
  {c : ℤ | ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ a ≠ b ∧ a^3 - b^3 = a^2 - b^2 ∧ c = ⌊9 * a * b⌋} = {1, 2, 3} :=
by sorry

end NUMINAMATH_CALUDE_possible_values_of_c_l159_15950


namespace NUMINAMATH_CALUDE_max_value_of_objective_function_l159_15999

def objective_function (x₁ x₂ : ℝ) : ℝ := 4 * x₁ + 6 * x₂

def feasible_region (x₁ x₂ : ℝ) : Prop :=
  x₁ ≥ 0 ∧ x₂ ≥ 0 ∧ x₁ + x₂ ≤ 18 ∧ 0.5 * x₁ + x₂ ≤ 12 ∧ 2 * x₁ ≤ 24 ∧ 2 * x₂ ≤ 18

theorem max_value_of_objective_function :
  ∃ (x₁ x₂ : ℝ), feasible_region x₁ x₂ ∧
    ∀ (y₁ y₂ : ℝ), feasible_region y₁ y₂ →
      objective_function x₁ x₂ ≥ objective_function y₁ y₂ ∧
      objective_function x₁ x₂ = 84 :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_objective_function_l159_15999


namespace NUMINAMATH_CALUDE_train_speed_with_stoppages_train_problem_l159_15976

/-- Calculates the speed of a train including stoppages -/
theorem train_speed_with_stoppages 
  (speed_without_stoppages : ℝ) 
  (stoppage_time : ℝ) 
  (total_time : ℝ) :
  speed_without_stoppages * (total_time - stoppage_time) / total_time = 
  speed_without_stoppages * (1 - stoppage_time / total_time) := by
  sorry

/-- The speed of a train including stoppages, given its speed without stoppages and stoppage time -/
theorem train_problem 
  (speed_without_stoppages : ℝ) 
  (stoppage_time : ℝ) :
  speed_without_stoppages = 45 →
  stoppage_time = 1/3 →
  speed_without_stoppages * (1 - stoppage_time) = 30 := by
  sorry

end NUMINAMATH_CALUDE_train_speed_with_stoppages_train_problem_l159_15976


namespace NUMINAMATH_CALUDE_terms_before_negative_23_l159_15944

def arithmetic_sequence (a₁ : ℤ) (d : ℤ) (n : ℕ) : ℤ := a₁ + (n - 1) * d

theorem terms_before_negative_23 :
  let a₁ := 101
  let d := -4
  ∃ n : ℕ, 
    (arithmetic_sequence a₁ d n = -23) ∧ 
    (∀ k : ℕ, k < n → arithmetic_sequence a₁ d k > -23) ∧
    n - 1 = 31 :=
by sorry

end NUMINAMATH_CALUDE_terms_before_negative_23_l159_15944


namespace NUMINAMATH_CALUDE_isosceles_obtuse_triangle_smallest_angle_l159_15973

/-- An isosceles, obtuse triangle with one angle 75% larger than a right angle has smallest angles of 11.25°. -/
theorem isosceles_obtuse_triangle_smallest_angle :
  ∀ (a b c : ℝ),
  0 < a ∧ 0 < b ∧ 0 < c →  -- angles are positive
  a + b + c = 180 →  -- sum of angles in a triangle
  a = b →  -- isosceles condition
  c = 90 + 0.75 * 90 →  -- largest angle is 75% larger than right angle
  a = 11.25 :=
by
  sorry

#check isosceles_obtuse_triangle_smallest_angle

end NUMINAMATH_CALUDE_isosceles_obtuse_triangle_smallest_angle_l159_15973


namespace NUMINAMATH_CALUDE_range_of_a_l159_15961

-- Define the inequality as a function of x and a
def inequality (x a : ℝ) : Prop := 2 * x^2 + a * x - a^2 > 0

-- Define the theorem
theorem range_of_a : 
  (∃ a : ℝ, inequality 2 a) → 
  (∀ a : ℝ, inequality 2 a ↔ -2 < a ∧ a < 4) :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l159_15961


namespace NUMINAMATH_CALUDE_lines_are_skew_l159_15962

-- Define the concept of lines in 3D space
variable (Line : Type)

-- Define the properties of lines
variable (parallel : Line → Line → Prop)
variable (intersect : Line → Line → Prop)
variable (skew : Line → Line → Prop)

-- Theorem statement
theorem lines_are_skew (a b : Line) : 
  (¬ parallel a b) → (¬ intersect a b) → skew a b :=
by sorry

end NUMINAMATH_CALUDE_lines_are_skew_l159_15962


namespace NUMINAMATH_CALUDE_amount_of_b_l159_15985

theorem amount_of_b (a b : ℚ) : 
  a + b = 1210 → 
  (4 / 15 : ℚ) * a = (2 / 5 : ℚ) * b → 
  b = 484 := by
sorry

end NUMINAMATH_CALUDE_amount_of_b_l159_15985


namespace NUMINAMATH_CALUDE_initial_profit_percentage_l159_15981

/-- Proves that given an article with a cost price of Rs. 50, if reducing the cost price by 20% 
    and the selling price by Rs. 10.50 results in a 30% profit, then the initial profit percentage is 25%. -/
theorem initial_profit_percentage 
  (cost : ℝ) 
  (reduced_cost_percentage : ℝ) 
  (reduced_selling_price : ℝ) 
  (new_profit_percentage : ℝ) :
  cost = 50 →
  reduced_cost_percentage = 0.8 →
  reduced_selling_price = 10.5 →
  new_profit_percentage = 0.3 →
  (reduced_cost_percentage * cost * (1 + new_profit_percentage) - reduced_selling_price) / cost * 100 = 25 := by
sorry

end NUMINAMATH_CALUDE_initial_profit_percentage_l159_15981


namespace NUMINAMATH_CALUDE_free_fall_time_l159_15929

/-- The time taken for an object to fall from a height of 490m, given the relationship h = 4.9t² -/
theorem free_fall_time : ∃ (t : ℝ), t > 0 ∧ 490 = 4.9 * t^2 ∧ t = 10 := by
  sorry

end NUMINAMATH_CALUDE_free_fall_time_l159_15929


namespace NUMINAMATH_CALUDE_sequence_problem_l159_15978

/-- Given S_n = n^2 - 1 for all natural numbers n, prove that a_2016 = 4031 where a_n = S_n - S_(n-1) for n ≥ 2 -/
theorem sequence_problem (n : ℕ) (S : ℕ → ℤ) (a : ℕ → ℤ) 
    (h1 : ∀ n, S n = n^2 - 1)
    (h2 : ∀ n ≥ 2, a n = S n - S (n-1)) :
  a 2016 = 4031 := by
  sorry

end NUMINAMATH_CALUDE_sequence_problem_l159_15978


namespace NUMINAMATH_CALUDE_inequalities_not_equivalent_l159_15983

-- Define the two inequalities
def inequality1 (x : ℝ) : Prop := x + 3 - 1 / (x - 1) > -x + 2 - 1 / (x - 1)
def inequality2 (x : ℝ) : Prop := x + 3 > -x + 2

-- Theorem stating that the inequalities are not equivalent
theorem inequalities_not_equivalent : ¬(∀ x : ℝ, inequality1 x ↔ inequality2 x) := by
  sorry

end NUMINAMATH_CALUDE_inequalities_not_equivalent_l159_15983


namespace NUMINAMATH_CALUDE_largest_integer_problem_l159_15930

theorem largest_integer_problem :
  ∃ (m : ℕ), m < 150 ∧ m > 50 ∧ 
  (∃ (a : ℕ), m = 9 * a - 2) ∧
  (∃ (b : ℕ), m = 6 * b - 4) ∧
  (∀ (n : ℕ), n < 150 ∧ n > 50 ∧ 
    (∃ (c : ℕ), n = 9 * c - 2) ∧
    (∃ (d : ℕ), n = 6 * d - 4) → n ≤ m) ∧
  m = 106 :=
sorry

end NUMINAMATH_CALUDE_largest_integer_problem_l159_15930


namespace NUMINAMATH_CALUDE_expand_product_l159_15967

theorem expand_product (x : ℝ) (hx : x ≠ 0) :
  (3 / 7) * ((7 / x^3) + 14 * x^5) = 3 / x^3 + 6 * x^5 := by
  sorry

end NUMINAMATH_CALUDE_expand_product_l159_15967


namespace NUMINAMATH_CALUDE_range_of_function_l159_15997

theorem range_of_function (f : ℝ → ℝ) (h : ∀ x, f x ∈ Set.Icc (3/8) (4/9)) :
  ∀ x, f x + Real.sqrt (1 - 2 * f x) ∈ Set.Icc (7/9) (7/8) := by
  sorry

end NUMINAMATH_CALUDE_range_of_function_l159_15997


namespace NUMINAMATH_CALUDE_point_in_fourth_quadrant_l159_15947

theorem point_in_fourth_quadrant (A B C : Real) (h_acute : A > 0 ∧ B > 0 ∧ C > 0 ∧ A < π/2 ∧ B < π/2 ∧ C < π/2) (h_triangle : A + B + C = π) :
  let P : Real × Real := (Real.sin A - Real.cos B, Real.cos A - Real.sin C)
  P.1 > 0 ∧ P.2 < 0 :=
by sorry

end NUMINAMATH_CALUDE_point_in_fourth_quadrant_l159_15947


namespace NUMINAMATH_CALUDE_line_chart_best_for_fever_temperature_l159_15924

/- Define the types of charts -/
inductive ChartType
| Bar
| Line
| Pie

/- Define the properties of data we want to visualize -/
structure TemperatureData where
  showsQuantity : Bool
  showsChanges : Bool
  showsRelationship : Bool

/- Define the characteristics of fever temperature data -/
def feverTemperatureData : TemperatureData :=
  { showsQuantity := true
  , showsChanges := true
  , showsRelationship := false }

/- Define which chart types are suitable for different data properties -/
def suitableChartType (data : TemperatureData) : ChartType :=
  if data.showsChanges then ChartType.Line
  else if data.showsQuantity then ChartType.Bar
  else ChartType.Pie

/- Theorem: Line chart is the best for tracking fever temperature changes -/
theorem line_chart_best_for_fever_temperature : 
  suitableChartType feverTemperatureData = ChartType.Line := by
  sorry

end NUMINAMATH_CALUDE_line_chart_best_for_fever_temperature_l159_15924


namespace NUMINAMATH_CALUDE_crate_stacking_ways_l159_15907

/-- Represents the dimensions of a crate -/
structure CrateDimensions where
  length : ℕ
  width : ℕ
  height : ℕ

/-- Calculates the number of ways to stack crates to achieve a specific height -/
def countStackingWays (dimensions : CrateDimensions) (numCrates : ℕ) (targetHeight : ℕ) : ℕ :=
  sorry

/-- The main theorem stating the number of ways to stack 11 crates to 47ft -/
theorem crate_stacking_ways :
  let dimensions : CrateDimensions := { length := 3, width := 4, height := 5 }
  countStackingWays dimensions 11 47 = 2277 := by
  sorry

end NUMINAMATH_CALUDE_crate_stacking_ways_l159_15907


namespace NUMINAMATH_CALUDE_coeff_x_squared_expansion_l159_15918

open Polynomial

/-- The coefficient of x^2 in the expansion of (1-2x)^5(1+3x)^4 is -26 -/
theorem coeff_x_squared_expansion : 
  (coeff ((1 - 2 * X) ^ 5 * (1 + 3 * X) ^ 4) 2) = -26 := by
  sorry

end NUMINAMATH_CALUDE_coeff_x_squared_expansion_l159_15918


namespace NUMINAMATH_CALUDE_darla_books_count_l159_15994

/-- Proves that Darla has 6 books given the conditions of the problem -/
theorem darla_books_count :
  ∀ (d k g : ℕ),
  k = d / 2 →
  g = 5 * (d + k) →
  d + k + g = 54 →
  d = 6 :=
by sorry

end NUMINAMATH_CALUDE_darla_books_count_l159_15994


namespace NUMINAMATH_CALUDE_office_officers_count_l159_15931

/-- Represents the number of officers in an office. -/
def num_officers : ℕ := 15

/-- Represents the number of non-officers in the office. -/
def num_non_officers : ℕ := 525

/-- Represents the average salary of all employees in rupees per month. -/
def avg_salary_all : ℕ := 120

/-- Represents the average salary of officers in rupees per month. -/
def avg_salary_officers : ℕ := 470

/-- Represents the average salary of non-officers in rupees per month. -/
def avg_salary_non_officers : ℕ := 110

/-- Theorem stating that the number of officers is 15, given the conditions. -/
theorem office_officers_count :
  (num_officers * avg_salary_officers + num_non_officers * avg_salary_non_officers) / (num_officers + num_non_officers) = avg_salary_all ∧
  num_officers = 15 :=
sorry

end NUMINAMATH_CALUDE_office_officers_count_l159_15931


namespace NUMINAMATH_CALUDE_zero_exponent_is_one_l159_15958

theorem zero_exponent_is_one (x : ℝ) (h : x ≠ 0) : x^0 = 1 := by
  sorry

end NUMINAMATH_CALUDE_zero_exponent_is_one_l159_15958


namespace NUMINAMATH_CALUDE_min_value_exponential_sum_l159_15949

theorem min_value_exponential_sum (x y : ℝ) (hx : x > 0) (hy : y > 0) (h_sum : 2 * x + y = 6) :
  ∃ (m : ℝ), m = 54 ∧ ∀ (z : ℝ), 9^x + 3^y ≥ z → z ≤ m :=
by sorry

end NUMINAMATH_CALUDE_min_value_exponential_sum_l159_15949


namespace NUMINAMATH_CALUDE_dividend_percentage_calculation_l159_15932

theorem dividend_percentage_calculation (face_value : ℝ) (purchase_price : ℝ) (return_on_investment : ℝ) :
  face_value = 40 →
  purchase_price = 20 →
  return_on_investment = 0.25 →
  (purchase_price * return_on_investment) / face_value = 0.125 :=
by sorry

end NUMINAMATH_CALUDE_dividend_percentage_calculation_l159_15932


namespace NUMINAMATH_CALUDE_restaurant_sales_tax_rate_l159_15993

theorem restaurant_sales_tax_rate 
  (total_bill : ℝ) 
  (striploin_cost : ℝ) 
  (wine_cost : ℝ) 
  (gratuities : ℝ) 
  (h1 : total_bill = 140)
  (h2 : striploin_cost = 80)
  (h3 : wine_cost = 10)
  (h4 : gratuities = 41) :
  (total_bill - striploin_cost - wine_cost - gratuities) / (striploin_cost + wine_cost) = 0.1 := by
  sorry

end NUMINAMATH_CALUDE_restaurant_sales_tax_rate_l159_15993


namespace NUMINAMATH_CALUDE_banana_problem_l159_15921

/-- Represents the number of bananas eaten on a given day -/
def bananas_eaten (day : ℕ) (first_day : ℕ) : ℕ :=
  first_day + 6 * (day - 1)

/-- The total number of bananas eaten over 5 days -/
def total_bananas (first_day : ℕ) : ℕ :=
  (bananas_eaten 1 first_day) + (bananas_eaten 2 first_day) + 
  (bananas_eaten 3 first_day) + (bananas_eaten 4 first_day) + 
  (bananas_eaten 5 first_day)

theorem banana_problem : 
  ∃ (first_day : ℕ), total_bananas first_day = 100 ∧ first_day = 8 :=
by
  sorry

end NUMINAMATH_CALUDE_banana_problem_l159_15921


namespace NUMINAMATH_CALUDE_ratio_odd_even_divisors_N_l159_15992

/-- The number N as defined in the problem -/
def N : ℕ := 46 * 46 * 81 * 450

/-- Sum of odd divisors of a natural number -/
def sum_odd_divisors (n : ℕ) : ℕ := sorry

/-- Sum of even divisors of a natural number -/
def sum_even_divisors (n : ℕ) : ℕ := sorry

/-- Theorem stating the ratio of sum of odd divisors to sum of even divisors of N -/
theorem ratio_odd_even_divisors_N :
  (sum_odd_divisors N : ℚ) / (sum_even_divisors N : ℚ) = 1 / 14 := by sorry

end NUMINAMATH_CALUDE_ratio_odd_even_divisors_N_l159_15992


namespace NUMINAMATH_CALUDE_find_number_l159_15996

theorem find_number : ∃ x : ℝ, (0.8 * x - 20 = 60) ∧ x = 100 := by
  sorry

end NUMINAMATH_CALUDE_find_number_l159_15996


namespace NUMINAMATH_CALUDE_museum_discount_percentage_l159_15935

/-- Represents the discount percentage for people 18 years old or younger -/
def discount_percentage : ℝ := 30

/-- Represents the regular ticket cost -/
def regular_ticket_cost : ℝ := 10

/-- Represents Dorothy's initial amount of money -/
def dorothy_initial_money : ℝ := 70

/-- Represents Dorothy's remaining money after the trip -/
def dorothy_remaining_money : ℝ := 26

/-- Represents the number of people in Dorothy's family -/
def family_size : ℕ := 5

/-- Represents the number of adults (paying full price) in Dorothy's family -/
def num_adults : ℕ := 3

/-- Represents the number of children (eligible for discount) in Dorothy's family -/
def num_children : ℕ := 2

theorem museum_discount_percentage :
  let total_spent := dorothy_initial_money - dorothy_remaining_money
  let adult_cost := num_adults * regular_ticket_cost
  let children_cost := total_spent - adult_cost
  let discounted_ticket_cost := regular_ticket_cost * (1 - discount_percentage / 100)
  children_cost = num_children * discounted_ticket_cost :=
by sorry

#check museum_discount_percentage

end NUMINAMATH_CALUDE_museum_discount_percentage_l159_15935


namespace NUMINAMATH_CALUDE_candy_purchase_calculation_l159_15977

/-- Calculates the change and discounted price per pack for a candy purchase. -/
theorem candy_purchase_calculation (packs : ℕ) (regular_price discount payment : ℚ) 
  (h_packs : packs = 3)
  (h_regular_price : regular_price = 12)
  (h_discount : discount = 15 / 100)
  (h_payment : payment = 20) :
  let discounted_total := regular_price * (1 - discount)
  let change := payment - discounted_total
  let price_per_pack := discounted_total / packs
  change = 980 / 100 ∧ price_per_pack = 340 / 100 := by
  sorry

end NUMINAMATH_CALUDE_candy_purchase_calculation_l159_15977


namespace NUMINAMATH_CALUDE_razorback_tshirt_sales_l159_15953

/-- The Razorback T-shirt Shop problem -/
theorem razorback_tshirt_sales (profit_per_shirt : ℕ) (total_profit : ℕ) 
    (h1 : profit_per_shirt = 9)
    (h2 : total_profit = 2205) :
  total_profit / profit_per_shirt = 245 := by
  sorry

#check razorback_tshirt_sales

end NUMINAMATH_CALUDE_razorback_tshirt_sales_l159_15953


namespace NUMINAMATH_CALUDE_adults_average_age_l159_15956

def robotics_camp_problem (total_members : ℕ) (overall_average_age : ℝ)
  (num_girls num_boys num_adults : ℕ) (girls_average_age boys_average_age : ℝ) : Prop :=
  total_members = 50 ∧
  overall_average_age = 20 ∧
  num_girls = 25 ∧
  num_boys = 18 ∧
  num_adults = 7 ∧
  girls_average_age = 18 ∧
  boys_average_age = 19 ∧
  (total_members : ℝ) * overall_average_age =
    (num_girls : ℝ) * girls_average_age +
    (num_boys : ℝ) * boys_average_age +
    (num_adults : ℝ) * ((1000 - 450 - 342) / 7)

theorem adults_average_age
  (total_members : ℕ) (overall_average_age : ℝ)
  (num_girls num_boys num_adults : ℕ) (girls_average_age boys_average_age : ℝ)
  (h : robotics_camp_problem total_members overall_average_age
    num_girls num_boys num_adults girls_average_age boys_average_age) :
  (1000 - 450 - 342) / 7 = (total_members * overall_average_age -
    num_girls * girls_average_age - num_boys * boys_average_age) / num_adults :=
by sorry

end NUMINAMATH_CALUDE_adults_average_age_l159_15956


namespace NUMINAMATH_CALUDE_tom_video_game_spending_l159_15920

/-- The total amount Tom spent on new video games --/
def total_spent (batman_price superman_price discount_rate tax_rate : ℚ) : ℚ :=
  let discounted_batman := batman_price * (1 - discount_rate)
  let discounted_superman := superman_price * (1 - discount_rate)
  let total_before_tax := discounted_batman + discounted_superman
  total_before_tax * (1 + tax_rate)

/-- Theorem stating the total amount Tom spent on new video games --/
theorem tom_video_game_spending :
  total_spent 13.60 5.06 0.20 0.08 = 16.12 := by
  sorry

#eval total_spent 13.60 5.06 0.20 0.08

end NUMINAMATH_CALUDE_tom_video_game_spending_l159_15920


namespace NUMINAMATH_CALUDE_sequence_properties_l159_15941

/-- Definition of an arithmetic sequence -/
def is_arithmetic_sequence (a : ℕ+ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ+, a (n + 1) = a n + d

/-- Definition of a geometric sequence -/
def is_geometric_sequence (a : ℕ+ → ℝ) : Prop :=
  ∃ q : ℝ, q ≠ 0 ∧ ∀ n : ℕ+, a (n + 1) = a n * q

/-- Definition of the sum of the first n terms -/
def S (a : ℕ+ → ℝ) (n : ℕ+) : ℝ :=
  (Finset.range n).sum (λ i => a ⟨i + 1, Nat.succ_pos i⟩)

/-- Main theorem -/
theorem sequence_properties (a : ℕ+ → ℝ) :
  (∀ n : ℕ+, (is_arithmetic_sequence a ∧ is_geometric_sequence a) → a n = a (n + 1)) ∧
  (∃ α β : ℝ, ∀ n : ℕ+, S a n = α * n^2 + β * n) → is_arithmetic_sequence a ∧
  (∀ n : ℕ+, S a n = 1 - (-1)^(n : ℕ)) → is_geometric_sequence a :=
sorry

end NUMINAMATH_CALUDE_sequence_properties_l159_15941


namespace NUMINAMATH_CALUDE_increase_and_subtract_l159_15934

theorem increase_and_subtract (initial : ℝ) (increase_percent : ℝ) (subtract_amount : ℝ) : 
  initial = 837 → 
  increase_percent = 135 → 
  subtract_amount = 250 → 
  (initial * (1 + increase_percent / 100) - subtract_amount) = 1717.95 := by
  sorry

end NUMINAMATH_CALUDE_increase_and_subtract_l159_15934


namespace NUMINAMATH_CALUDE_total_profit_calculation_l159_15991

theorem total_profit_calculation (x_investment y_investment z_investment : ℕ)
  (x_months y_months z_months : ℕ) (z_profit : ℕ) :
  x_investment = 36000 →
  y_investment = 42000 →
  z_investment = 48000 →
  x_months = 12 →
  y_months = 12 →
  z_months = 8 →
  z_profit = 4096 →
  (z_investment * z_months * 14080 = z_profit * (x_investment * x_months + y_investment * y_months + z_investment * z_months)) :=
by
  sorry

#check total_profit_calculation

end NUMINAMATH_CALUDE_total_profit_calculation_l159_15991


namespace NUMINAMATH_CALUDE_mean_proportional_problem_l159_15912

theorem mean_proportional_problem (x : ℝ) : 
  (156 : ℝ)^2 = x * 104 → x = 234 := by
  sorry

end NUMINAMATH_CALUDE_mean_proportional_problem_l159_15912


namespace NUMINAMATH_CALUDE_complex_exponential_form_l159_15908

/-- Given a complex number z = e^a(cos b + i sin b), its exponential form is e^(a + ib) -/
theorem complex_exponential_form (a b : ℝ) :
  let z : ℂ := Complex.exp a * (Complex.cos b + Complex.I * Complex.sin b)
  z = Complex.exp (a + Complex.I * b) := by
  sorry

end NUMINAMATH_CALUDE_complex_exponential_form_l159_15908


namespace NUMINAMATH_CALUDE_domain_range_sum_l159_15972

-- Define the function f
def f (x : ℝ) : ℝ := -x^2 + x

-- Define the theorem
theorem domain_range_sum (m n : ℝ) : 
  (∀ x, m ≤ x ∧ x ≤ n → 2*m ≤ f x ∧ f x ≤ 2*n) →
  (∀ y, 2*m ≤ y ∧ y ≤ 2*n → ∃ x, m ≤ x ∧ x ≤ n ∧ f x = y) →
  m + n = -2 := by
sorry

end NUMINAMATH_CALUDE_domain_range_sum_l159_15972


namespace NUMINAMATH_CALUDE_range_of_t_l159_15982

/-- Set A definition -/
def A : Set ℝ := {x : ℝ | (x + 8) / (x - 5) ≤ 0}

/-- Set B definition -/
def B (t : ℝ) : Set ℝ := {x : ℝ | t + 1 ≤ x ∧ x ≤ 2*t - 1}

/-- Theorem stating the range of t -/
theorem range_of_t (t : ℝ) : 
  (∃ x, x ∈ B t) → -- B is non-empty
  (A ∩ B t = ∅) → -- A and B have no intersection
  t ≥ 4 := by sorry

end NUMINAMATH_CALUDE_range_of_t_l159_15982


namespace NUMINAMATH_CALUDE_prob_red_ball_one_third_l159_15989

/-- A bag containing red and yellow balls -/
structure Bag where
  red_balls : ℕ
  yellow_balls : ℕ

/-- The probability of drawing a red ball from the bag -/
def prob_red_ball (bag : Bag) : ℚ :=
  bag.red_balls / (bag.red_balls + bag.yellow_balls)

/-- The theorem stating the probability of drawing a red ball -/
theorem prob_red_ball_one_third (bag : Bag) 
  (h1 : bag.red_balls = 1) 
  (h2 : bag.yellow_balls = 2) : 
  prob_red_ball bag = 1/3 := by
  sorry

#check prob_red_ball_one_third

end NUMINAMATH_CALUDE_prob_red_ball_one_third_l159_15989


namespace NUMINAMATH_CALUDE_book_distribution_theorem_l159_15927

/-- The number of ways to choose 3 books from 5 different books for 3 students -/
def choose_books : ℕ := 60

/-- The number of ways to buy 3 books from 5 different books for 3 students -/
def buy_books : ℕ := 125

/-- The number of different books available -/
def num_books : ℕ := 5

/-- The number of students receiving books -/
def num_students : ℕ := 3

theorem book_distribution_theorem :
  (choose_books = num_books * (num_books - 1) * (num_books - 2)) ∧
  (buy_books = num_books * num_books * num_books) := by
  sorry

end NUMINAMATH_CALUDE_book_distribution_theorem_l159_15927


namespace NUMINAMATH_CALUDE_proper_subset_implies_a_geq_two_l159_15965

def A : Set ℝ := {x | 1 < x ∧ x < 2}
def B (a : ℝ) : Set ℝ := {x | x - a < 0}

theorem proper_subset_implies_a_geq_two (a : ℝ) :
  A ⊂ B a → a ≥ 2 := by sorry

end NUMINAMATH_CALUDE_proper_subset_implies_a_geq_two_l159_15965
