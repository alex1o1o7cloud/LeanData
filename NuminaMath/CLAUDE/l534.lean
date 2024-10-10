import Mathlib

namespace employed_females_percentage_l534_53430

theorem employed_females_percentage
  (total_employed : ℝ)
  (employable_population : ℝ)
  (h1 : total_employed = 1.2 * employable_population)
  (h2 : 0.8 * employable_population = total_employed * (80 / 100)) :
  (total_employed - 0.8 * employable_population) / total_employed = 1 / 3 := by
  sorry

end employed_females_percentage_l534_53430


namespace quadratic_properties_l534_53417

/-- A quadratic function passing through (-3, 0) with axis of symmetry x = -1 -/
structure QuadraticFunction where
  a : ℝ
  b : ℝ
  c : ℝ
  a_nonzero : a ≠ 0
  passes_through_minus_three : a * (-3)^2 + b * (-3) + c = 0
  axis_of_symmetry : -b / (2 * a) = -1

/-- Properties of the quadratic function -/
theorem quadratic_properties (f : QuadraticFunction) :
  (f.a + f.b + f.c = 0) ∧
  (2 * f.c + 3 * f.b = 0) ∧
  (∀ k : ℝ, k > 0 → ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧
    f.a * x₁^2 + f.b * x₁ + f.c = k * (x₁ + 1) ∧
    f.a * x₂^2 + f.b * x₂ + f.c = k * (x₂ + 1)) :=
by sorry

end quadratic_properties_l534_53417


namespace repeating_decimal_sum_l534_53412

theorem repeating_decimal_sum (c d : ℕ) : 
  (5 : ℚ) / 13 = 0.1 * (10 * c + d : ℚ) / 99 → c + d = 11 := by
sorry

end repeating_decimal_sum_l534_53412


namespace fraction_equality_l534_53486

def fraction_pairs : Set (ℤ × ℤ) :=
  {(0, 6), (1, -1), (6, -6), (13, -7), (-2, -22), (-3, -15), (-8, -10), (-15, -9)}

theorem fraction_equality (k l : ℤ) :
  (7 * k - 5) / (5 * k - 3) = (6 * l - 1) / (4 * l - 3) ↔ (k, l) ∈ fraction_pairs := by
  sorry

#check fraction_equality

end fraction_equality_l534_53486


namespace clothing_discount_l534_53482

theorem clothing_discount (P : ℝ) (f : ℝ) (h1 : P > 0) (h2 : f > 0) (h3 : f < 1) :
  (f * P - (1/2) * P = 0.4 * (f * P)) → f = 5/6 := by
  sorry

end clothing_discount_l534_53482


namespace wire_ratio_l534_53405

theorem wire_ratio (x y : ℝ) : 
  x > 0 → y > 0 → 
  (4 * (x / 4) = 5 * (y / 5)) → 
  x / y = 1 := by
  sorry

end wire_ratio_l534_53405


namespace least_total_cost_equal_quantity_l534_53425

def strawberry_pack_size : ℕ := 6
def strawberry_pack_price : ℕ := 2
def blueberry_pack_size : ℕ := 5
def blueberry_pack_price : ℕ := 3
def cherry_pack_size : ℕ := 8
def cherry_pack_price : ℕ := 4

def least_common_multiple (a b c : ℕ) : ℕ := Nat.lcm a (Nat.lcm b c)

def total_cost (lcm : ℕ) : ℕ :=
  (lcm / strawberry_pack_size) * strawberry_pack_price +
  (lcm / blueberry_pack_size) * blueberry_pack_price +
  (lcm / cherry_pack_size) * cherry_pack_price

theorem least_total_cost_equal_quantity :
  total_cost (least_common_multiple strawberry_pack_size blueberry_pack_size cherry_pack_size) = 172 := by
  sorry

end least_total_cost_equal_quantity_l534_53425


namespace kate_lives_on_15_dollars_per_month_kate_has_frugal_lifestyle_l534_53408

/-- Represents a person living in New York --/
structure NYResident where
  name : String
  monthly_expenses : ℕ
  uses_dumpster_diving : Bool
  has_frugal_habits : Bool

/-- Represents Kate Hashimoto --/
def kate : NYResident :=
  { name := "Kate Hashimoto"
  , monthly_expenses := 15
  , uses_dumpster_diving := true
  , has_frugal_habits := true }

/-- Theorem stating that Kate can live on $15 a month in New York --/
theorem kate_lives_on_15_dollars_per_month :
  kate.monthly_expenses = 15 ∧ kate.uses_dumpster_diving ∧ kate.has_frugal_habits :=
by sorry

/-- Definition of a frugal lifestyle in New York --/
def is_frugal_lifestyle (r : NYResident) : Prop :=
  r.monthly_expenses ≤ 15 ∧ r.uses_dumpster_diving ∧ r.has_frugal_habits

/-- Theorem stating that Kate has a frugal lifestyle --/
theorem kate_has_frugal_lifestyle : is_frugal_lifestyle kate :=
by sorry

end kate_lives_on_15_dollars_per_month_kate_has_frugal_lifestyle_l534_53408


namespace number_difference_l534_53499

theorem number_difference (x y : ℝ) (h1 : x + y = 10) (h2 : x^2 - y^2 = 40) : |x - y| = 4 := by
  sorry

end number_difference_l534_53499


namespace opposite_of_sin_60_degrees_l534_53400

theorem opposite_of_sin_60_degrees :
  -(Real.sin (π / 3)) = -(Real.sqrt 3 / 2) := by
  sorry

end opposite_of_sin_60_degrees_l534_53400


namespace volunteer_count_l534_53427

/-- Represents the number of volunteers selected from each school -/
structure Volunteers where
  schoolA : ℕ
  schoolB : ℕ
  schoolC : ℕ

/-- The ratio of students in schools A, B, and C -/
def schoolRatio : Fin 3 → ℕ
  | 0 => 2  -- School A
  | 1 => 3  -- School B
  | 2 => 5  -- School C

/-- The total ratio sum -/
def totalRatio : ℕ := (schoolRatio 0) + (schoolRatio 1) + (schoolRatio 2)

/-- Stratified sampling condition -/
def isStratifiedSample (v : Volunteers) : Prop :=
  (v.schoolA * schoolRatio 1 = v.schoolB * schoolRatio 0) ∧
  (v.schoolA * schoolRatio 2 = v.schoolC * schoolRatio 0)

/-- The main theorem -/
theorem volunteer_count (v : Volunteers) 
  (h_stratified : isStratifiedSample v) 
  (h_schoolA : v.schoolA = 6) : 
  v.schoolA + v.schoolB + v.schoolC = 30 := by
  sorry

end volunteer_count_l534_53427


namespace intersection_and_union_when_m_eq_5_subset_condition_l534_53473

-- Define the sets A and B
def A : Set ℝ := {x | -3 ≤ x ∧ x ≤ 5}
def B (m : ℝ) : Set ℝ := {x | x < 2*m - 3}

-- Theorem for part (1)
theorem intersection_and_union_when_m_eq_5 :
  (A ∩ B 5 = A) ∧ (Aᶜ ∪ B 5 = Set.univ) := by sorry

-- Theorem for part (2)
theorem subset_condition :
  ∀ m, A ⊆ B m ↔ m > 4 := by sorry

end intersection_and_union_when_m_eq_5_subset_condition_l534_53473


namespace max_distance_circle_to_line_l534_53444

/-- The maximum distance from a point on the circle ρ = 8sinθ to the line θ = π/3 is 6 -/
theorem max_distance_circle_to_line :
  let circle := {p : ℝ × ℝ | p.1^2 + (p.2 - 4)^2 = 16}
  let line := {p : ℝ × ℝ | p.2 = Real.sqrt 3 * p.1}
  ∃ (max_dist : ℝ), max_dist = 6 ∧
    ∀ (p : ℝ × ℝ), p ∈ circle →
      ∀ (q : ℝ × ℝ), q ∈ line →
        Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2) ≤ max_dist :=
by sorry

end max_distance_circle_to_line_l534_53444


namespace snake_eating_time_l534_53469

/-- Represents the number of weeks it takes for a snake to eat one mouse. -/
def weeks_per_mouse (mice_per_decade : ℕ) : ℚ :=
  (10 * 52) / mice_per_decade

/-- Proves that a snake eating 130 mice in a decade takes 4 weeks to eat one mouse. -/
theorem snake_eating_time : weeks_per_mouse 130 = 4 := by
  sorry

end snake_eating_time_l534_53469


namespace math_pass_count_l534_53413

/-- Represents the number of students in various categories -/
structure StudentCounts where
  english : ℕ
  math : ℕ
  bothSubjects : ℕ
  onlyEnglish : ℕ
  onlyMath : ℕ

/-- Theorem stating the number of students who pass in Math -/
theorem math_pass_count (s : StudentCounts) 
  (h1 : s.english = 30)
  (h2 : s.english = s.onlyEnglish + s.bothSubjects)
  (h3 : s.onlyEnglish = s.onlyMath + 10)
  (h4 : s.math = s.onlyMath + s.bothSubjects) :
  s.math = 20 := by
  sorry

end math_pass_count_l534_53413


namespace systematic_sampling_first_number_l534_53463

/-- Systematic sampling function that returns the number drawn from the nth group -/
def systematicSample (firstNumber : ℕ) (groupNumber : ℕ) (interval : ℕ) : ℕ :=
  firstNumber + interval * (groupNumber - 1)

theorem systematic_sampling_first_number :
  ∀ (totalStudents : ℕ) (numGroups : ℕ) (firstNumber : ℕ),
    totalStudents = 160 →
    numGroups = 20 →
    systematicSample firstNumber 15 8 = 116 →
    firstNumber = 4 := by
  sorry

end systematic_sampling_first_number_l534_53463


namespace negative_five_plus_eight_equals_three_l534_53446

theorem negative_five_plus_eight_equals_three : -5 + 8 = 3 := by
  sorry

end negative_five_plus_eight_equals_three_l534_53446


namespace sum_of_fractions_equals_one_l534_53404

theorem sum_of_fractions_equals_one (x y z : ℝ) (h : x * y * z = 1) :
  (1 / (1 + x + x * y)) + (1 / (1 + y + y * z)) + (1 / (1 + z + z * x)) = 1 := by
  sorry

end sum_of_fractions_equals_one_l534_53404


namespace parabola_kite_sum_l534_53450

/-- Represents a parabola of the form y = ax^2 + b -/
structure Parabola where
  a : ℝ
  b : ℝ

/-- The kite formed by the intersections of two parabolas with the coordinate axes -/
structure Kite where
  p1 : Parabola
  p2 : Parabola

/-- The area of a kite -/
def kite_area (k : Kite) : ℝ := sorry

/-- The theorem to be proved -/
theorem parabola_kite_sum (c d : ℝ) :
  let p1 : Parabola := ⟨c, 3⟩
  let p2 : Parabola := ⟨-d, 7⟩
  let k : Kite := ⟨p1, p2⟩
  kite_area k = 20 → c + d = 18/25 := by sorry

end parabola_kite_sum_l534_53450


namespace total_cost_train_and_bus_l534_53423

/-- The cost of a train ride from town P to town Q -/
def train_cost : ℝ := 8.35

/-- The cost of a bus ride from town P to town Q -/
def bus_cost : ℝ := 1.50

/-- The difference in cost between a train ride and a bus ride -/
def cost_difference : ℝ := 6.85

theorem total_cost_train_and_bus :
  train_cost + bus_cost = 9.85 ∧
  train_cost = bus_cost + cost_difference :=
sorry

end total_cost_train_and_bus_l534_53423


namespace original_number_l534_53447

theorem original_number : ∃ x : ℕ, 100 * x - x = 1980 ∧ x = 20 := by
  sorry

end original_number_l534_53447


namespace total_distance_two_wheels_l534_53402

/-- The total distance covered by two wheels with different radii -/
theorem total_distance_two_wheels 
  (r1 r2 N : ℝ) 
  (h_positive : r1 > 0 ∧ r2 > 0 ∧ N > 0) : 
  let wheel1_revolutions : ℝ := 1500
  let wheel2_revolutions : ℝ := N * wheel1_revolutions
  let distance_wheel1 : ℝ := 2 * Real.pi * r1 * wheel1_revolutions
  let distance_wheel2 : ℝ := 2 * Real.pi * r2 * wheel2_revolutions
  let total_distance : ℝ := distance_wheel1 + distance_wheel2
  total_distance = 3000 * Real.pi * (r1 + N * r2) :=
by sorry

end total_distance_two_wheels_l534_53402


namespace square_diff_equals_four_l534_53495

theorem square_diff_equals_four (a b : ℝ) (h : a = b + 2) : a^2 - 2*a*b + b^2 = 4 := by
  sorry

end square_diff_equals_four_l534_53495


namespace tennis_balls_first_set_l534_53401

theorem tennis_balls_first_set :
  ∀ (total_balls first_set second_set : ℕ),
    total_balls = 175 →
    second_set = 75 →
    first_set + second_set = total_balls →
    (2 : ℚ) / 5 * first_set + (1 : ℚ) / 3 * second_set + 110 = total_balls →
    first_set = 100 := by
  sorry

end tennis_balls_first_set_l534_53401


namespace complex_fraction_simplification_l534_53487

theorem complex_fraction_simplification :
  (3 - 2 * Complex.I) / (1 + 4 * Complex.I) = -5/17 - 14/17 * Complex.I :=
by sorry

end complex_fraction_simplification_l534_53487


namespace worksheets_graded_l534_53428

/-- Represents the problem of determining the number of worksheets graded before new ones were turned in. -/
theorem worksheets_graded (initial : ℕ) (new_turned_in : ℕ) (final : ℕ) : 
  initial = 34 → new_turned_in = 36 → final = 63 → 
  ∃ (graded : ℕ), graded = 7 ∧ initial - graded + new_turned_in = final := by
  sorry

end worksheets_graded_l534_53428


namespace opposite_of_2024_l534_53472

theorem opposite_of_2024 : -(2024 : ℤ) = -2024 := by
  sorry

end opposite_of_2024_l534_53472


namespace function_properties_l534_53411

-- Define the function f(x) with parameter a
def f (a : ℝ) (x : ℝ) : ℝ := x^3 + (4-a)*x^2 - 15*x + a

-- Define the derivative of f(x) with respect to x
def f' (a : ℝ) (x : ℝ) : ℝ := 3*x^2 + 2*(4-a)*x - 15

theorem function_properties :
  -- Part 1: When f(0) = -2, a = -2
  (∀ a : ℝ, f a 0 = -2 → a = -2) ∧
  
  -- Part 2: The minimum value of f(x) when a = -2 is -10
  (∃ x : ℝ, f (-2) x = -10 ∧ ∀ y : ℝ, f (-2) y ≥ -10) ∧
  
  -- Part 3: The maximum value of a for which f'(x) ≤ 0 on (-1, 1) is 10
  (∀ a : ℝ, (∀ x : ℝ, -1 < x ∧ x < 1 → f' a x ≤ 0) → a ≤ 10) ∧
  (∃ a : ℝ, a = 10 ∧ ∀ x : ℝ, -1 < x ∧ x < 1 → f' a x ≤ 0) :=
by sorry

end function_properties_l534_53411


namespace min_weighings_to_identify_defective_l534_53454

/-- Represents a piece that can be either standard or defective -/
inductive Piece
| Standard
| Defective

/-- Represents the result of a weighing -/
inductive WeighingResult
| Equal
| LeftHeavier
| RightHeavier

/-- A function that simulates a weighing on a balance scale -/
def weigh (left right : List Piece) : WeighingResult := sorry

/-- The set of all possible pieces -/
def allPieces : Finset Piece := sorry

/-- The number of pieces -/
def numPieces : Nat := 5

/-- The number of standard pieces -/
def numStandard : Nat := 4

/-- The number of defective pieces -/
def numDefective : Nat := 1

/-- A strategy for identifying the defective piece -/
def identifyDefective : Nat → Option Piece := sorry

theorem min_weighings_to_identify_defective :
  ∃ (strategy : Nat → Option Piece),
    (∀ defective : Piece, 
      defective ∈ allPieces → 
      ∃ n : Nat, n ≤ 3 ∧ strategy n = some defective) ∧
    (∀ m : Nat, 
      (∀ defective : Piece, 
        defective ∈ allPieces → 
        (∃ n : Nat, n ≤ m ∧ strategy n = some defective)) → 
      m ≥ 3) :=
sorry

end min_weighings_to_identify_defective_l534_53454


namespace cos_a3_plus_a5_l534_53484

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem cos_a3_plus_a5 (a : ℕ → ℝ) (h : arithmetic_sequence a) 
  (h1 : a 1 + a 4 + a 7 = 5 * Real.pi / 4) : 
  Real.cos (a 3 + a 5) = -Real.sqrt 3 / 2 := by
  sorry

end cos_a3_plus_a5_l534_53484


namespace payment_divisible_by_25_l534_53497

theorem payment_divisible_by_25 (B : ℕ) (h : B ≤ 9) : 
  ∃ k : ℕ, 2000 + 100 * B + 5 = 25 * k := by
  sorry

end payment_divisible_by_25_l534_53497


namespace inverse_of_B_squared_l534_53445

theorem inverse_of_B_squared (B : Matrix (Fin 2) (Fin 2) ℝ) :
  B⁻¹ = ![![3, -2], ![0, 1]] →
  (B^2)⁻¹ = ![![9, -6], ![0, 1]] := by
  sorry

end inverse_of_B_squared_l534_53445


namespace sum_of_numbers_l534_53470

theorem sum_of_numbers (a b : ℝ) (h1 : a - b = 5) (h2 : max a b = 25) : a + b = 45 := by
  sorry

end sum_of_numbers_l534_53470


namespace unique_solution_equation_l534_53485

theorem unique_solution_equation :
  ∃! x : ℝ, x ≠ 2 ∧ x > 0 ∧ (3 * x^2 - 12 * x) / (x^2 - 4 * x) = x - 2 :=
by sorry

end unique_solution_equation_l534_53485


namespace equation_solutions_l534_53419

theorem equation_solutions :
  ∀ x y : ℕ, 1 + 3^x = 2^y ↔ (x = 0 ∧ y = 1) ∨ (x = 1 ∧ y = 2) :=
by sorry

end equation_solutions_l534_53419


namespace school_supplies_cost_l534_53415

theorem school_supplies_cost :
  let pencil_cartons : ℕ := 20
  let pencil_boxes_per_carton : ℕ := 10
  let pencil_box_cost : ℕ := 2
  let marker_cartons : ℕ := 10
  let marker_boxes_per_carton : ℕ := 5
  let marker_box_cost : ℕ := 4
  
  pencil_cartons * pencil_boxes_per_carton * pencil_box_cost +
  marker_cartons * marker_boxes_per_carton * marker_box_cost = 600 :=
by
  sorry

end school_supplies_cost_l534_53415


namespace solution_set_f_greater_than_one_range_of_m_l534_53442

-- Define the function f
def f (x : ℝ) : ℝ := |x + 2| - |x - 1|

-- Theorem for the solution set of f(x) > 1
theorem solution_set_f_greater_than_one :
  {x : ℝ | f x > 1} = {x : ℝ | x > 0} := by sorry

-- Theorem for the range of m
theorem range_of_m :
  {m : ℝ | ∃ x, f x + 4 ≥ |1 - 2*m|} = {m : ℝ | -6 ≤ m ∧ m ≤ 8} := by sorry

end solution_set_f_greater_than_one_range_of_m_l534_53442


namespace farm_ratio_l534_53480

/-- Given a farm with horses and cows, prove that the initial ratio of horses to cows is 4:1 --/
theorem farm_ratio (initial_horses initial_cows : ℕ) : 
  (initial_horses - 15 : ℚ) / (initial_cows + 15 : ℚ) = 7 / 3 →
  (initial_horses - 15) = (initial_cows + 15 + 60) →
  (initial_horses : ℚ) / initial_cows = 4 / 1 := by
  sorry

end farm_ratio_l534_53480


namespace unique_three_digit_number_l534_53429

theorem unique_three_digit_number : 
  ∃! (m g u : ℕ), 
    m ≠ g ∧ m ≠ u ∧ g ≠ u ∧
    m ∈ Finset.range 10 ∧ g ∈ Finset.range 10 ∧ u ∈ Finset.range 10 ∧
    100 * m + 10 * g + u ≥ 100 ∧ 100 * m + 10 * g + u < 1000 ∧
    100 * m + 10 * g + u = (m + g + u) * (m + g + u - 2) ∧
    100 * m + 10 * g + u = 195 :=
by sorry

end unique_three_digit_number_l534_53429


namespace g_at_3_equals_20_l534_53479

noncomputable def f (x : ℝ) : ℝ := 30 / (x + 5)

noncomputable def g (x : ℝ) : ℝ := 4 * (f⁻¹ x)

theorem g_at_3_equals_20 : g 3 = 20 := by
  sorry

end g_at_3_equals_20_l534_53479


namespace journey_distance_l534_53491

/-- Prove that the total distance traveled is 300 km given the specified conditions -/
theorem journey_distance (total_time : ℝ) (speed1 : ℝ) (speed2 : ℝ)
  (h_total_time : total_time = 11)
  (h_speed1 : speed1 = 30)
  (h_speed2 : speed2 = 25)
  (h_half_distance : ∀ d : ℝ, d / speed1 + d / speed2 = total_time → d = 300) :
  ∃ d : ℝ, d = 300 ∧ d / (2 * speed1) + d / (2 * speed2) = total_time :=
by sorry

end journey_distance_l534_53491


namespace min_value_expression_l534_53434

theorem min_value_expression (x y : ℝ) (hx : x > 0) (hy : y > 0) (h_sum : x + 2*y = 1) :
  (y/x) + (1/y) ≥ 4 ∧ ((y/x) + (1/y) = 4 ↔ x = 1/3 ∧ y = 1/3) :=
sorry

end min_value_expression_l534_53434


namespace binary_to_octal_conversion_l534_53488

-- Define the binary number
def binary_number : List Bool := [true, false, true, true, true, false]

-- Define the octal number
def octal_number : Nat := 56

-- Theorem statement
theorem binary_to_octal_conversion :
  (binary_number.foldl (fun acc b => 2 * acc + if b then 1 else 0) 0) = octal_number := by
  sorry

end binary_to_octal_conversion_l534_53488


namespace determinant_inequality_l534_53452

def det2 (a b c d : ℝ) : ℝ := a * d - b * c

theorem determinant_inequality (a : ℝ) : 
  det2 (a^2) 1 3 2 < det2 a 0 4 1 ↔ -1 < a ∧ a < 3/2 := by sorry

end determinant_inequality_l534_53452


namespace faye_team_size_l534_53475

def team_size (total_points : ℕ) (faye_points : ℕ) (others_points : ℕ) : ℕ :=
  (total_points - faye_points) / others_points + 1

theorem faye_team_size :
  team_size 68 28 8 = 6 := by
  sorry

end faye_team_size_l534_53475


namespace four_digit_addition_l534_53466

theorem four_digit_addition (A B C D : ℕ) : 
  4000 * A + 500 * B + 100 * C + 20 * D + 7 = 8070 → C = 3 := by
  sorry

end four_digit_addition_l534_53466


namespace solution_set_x_one_minus_x_l534_53435

theorem solution_set_x_one_minus_x (x : ℝ) : x * (1 - x) > 0 ↔ 0 < x ∧ x < 1 := by
  sorry

end solution_set_x_one_minus_x_l534_53435


namespace hyperbola_sum_l534_53451

/-- Represents a hyperbola with center (h, k) and parameters a and b -/
structure Hyperbola where
  h : ℝ
  k : ℝ
  a : ℝ
  b : ℝ

/-- The equation of the hyperbola -/
def hyperbola_equation (hyp : Hyperbola) (x y : ℝ) : Prop :=
  (x - hyp.h)^2 / hyp.a^2 - (y - hyp.k)^2 / hyp.b^2 = 1

theorem hyperbola_sum (hyp : Hyperbola) 
  (center : hyp.h = -3 ∧ hyp.k = 1)
  (vertex_distance : 2 * hyp.a = 8)
  (foci_distance : Real.sqrt (hyp.a^2 + hyp.b^2) = 5) :
  hyp.h + hyp.k + hyp.a + hyp.b = 5 := by
  sorry

end hyperbola_sum_l534_53451


namespace bran_remaining_payment_l534_53458

def tuition_fee : ℝ := 90
def monthly_earnings : ℝ := 15
def scholarship_percentage : ℝ := 0.30
def payment_period : ℕ := 3

theorem bran_remaining_payment :
  tuition_fee * (1 - scholarship_percentage) - monthly_earnings * payment_period = 18 := by
  sorry

end bran_remaining_payment_l534_53458


namespace twentieth_term_da_yan_l534_53449

def da_yan_sequence (n : ℕ) : ℕ :=
  if n % 2 = 0 then
    n^2 / 2
  else
    (n^2 - 1) / 2

theorem twentieth_term_da_yan (n : ℕ) : da_yan_sequence 20 = 200 := by
  sorry

end twentieth_term_da_yan_l534_53449


namespace inequality_problem_l534_53416

theorem inequality_problem (x y : ℝ) 
  (h1 : 2 * x - 3 * y > 2 * x) 
  (h2 : 2 * x + 3 * y < 3 * y) : 
  x < 0 ∧ y < 0 := by
sorry

end inequality_problem_l534_53416


namespace furniture_fraction_l534_53433

/-- Prove that the fraction of savings spent on furniture is 3/4, given that
the original savings were $800, the TV cost $200, and the rest was spent on furniture. -/
theorem furniture_fraction (savings : ℚ) (tv_cost : ℚ) (furniture_cost : ℚ) 
  (h1 : savings = 800)
  (h2 : tv_cost = 200)
  (h3 : furniture_cost + tv_cost = savings) :
  furniture_cost / savings = 3 / 4 := by
  sorry

end furniture_fraction_l534_53433


namespace sqrt_of_sum_of_cubes_l534_53494

theorem sqrt_of_sum_of_cubes : Real.sqrt (5 * (4^3 + 4^3 + 4^3 + 4^3)) = 8 * Real.sqrt 5 := by
  sorry

end sqrt_of_sum_of_cubes_l534_53494


namespace value_of_expression_l534_53407

theorem value_of_expression (a b : ℝ) (h : a - 2*b = -2) : 4 - 2*a + 4*b = 8 := by
  sorry

end value_of_expression_l534_53407


namespace postcard_perimeter_l534_53496

/-- The perimeter of a rectangle with width 6 inches and height 4 inches is 20 inches. -/
theorem postcard_perimeter : 
  let width : ℝ := 6
  let height : ℝ := 4
  let perimeter := 2 * (width + height)
  perimeter = 20 :=
by sorry

end postcard_perimeter_l534_53496


namespace eggs_eaten_in_morning_l534_53465

theorem eggs_eaten_in_morning (initial_eggs : ℕ) (afternoon_eggs : ℕ) (remaining_eggs : ℕ) :
  initial_eggs = 20 →
  afternoon_eggs = 3 →
  remaining_eggs = 13 →
  initial_eggs - remaining_eggs - afternoon_eggs = 4 :=
by
  sorry

end eggs_eaten_in_morning_l534_53465


namespace base_product_sum_theorem_l534_53448

/-- Represents a number in a given base --/
structure BaseNumber (base : ℕ) where
  value : ℕ

/-- Converts a BaseNumber to its decimal representation --/
def toDecimal {base : ℕ} (n : BaseNumber base) : ℕ := sorry

/-- Converts a decimal number to a BaseNumber --/
def fromDecimal (base : ℕ) (n : ℕ) : BaseNumber base := sorry

/-- Multiplies two BaseNumbers --/
def mult {base : ℕ} (a b : BaseNumber base) : BaseNumber base := sorry

/-- Adds two BaseNumbers --/
def add {base : ℕ} (a b : BaseNumber base) : BaseNumber base := sorry

theorem base_product_sum_theorem :
  ∀ c : ℕ,
    c > 1 →
    let thirteen := fromDecimal c 13
    let seventeen := fromDecimal c 17
    let nineteen := fromDecimal c 19
    let product := mult thirteen (mult seventeen nineteen)
    let sum := add thirteen (add seventeen nineteen)
    toDecimal product = toDecimal (fromDecimal c 4375) →
    toDecimal sum = toDecimal (fromDecimal 8 53) := by
  sorry

end base_product_sum_theorem_l534_53448


namespace quadratic_equation_m_value_l534_53476

theorem quadratic_equation_m_value (m : ℝ) : 
  (∀ x, ∃ a b c : ℝ, a ≠ 0 ∧ (m - 2) * x^(m^2 - 2) - 3*x + 1 = a*x^2 + b*x + c) → 
  m = -2 :=
sorry

end quadratic_equation_m_value_l534_53476


namespace simplify_fraction_product_l534_53455

theorem simplify_fraction_product : 
  (4 : ℚ) * (18 / 5) * (35 / -63) * (8 / 14) = -32 / 7 := by sorry

end simplify_fraction_product_l534_53455


namespace z_power_2017_l534_53410

theorem z_power_2017 (z : ℂ) (h : z * (1 - Complex.I) = 1 + Complex.I) : 
  z^2017 = Complex.I := by
sorry

end z_power_2017_l534_53410


namespace sara_quarters_theorem_l534_53498

/-- The number of quarters Sara had initially -/
def initial_quarters : ℕ := 783

/-- The number of quarters Sara's dad gave her -/
def quarters_from_dad : ℕ := 271

/-- The total number of quarters Sara has now -/
def total_quarters : ℕ := 1054

/-- Theorem stating that the initial number of quarters plus the quarters from dad equals the total quarters -/
theorem sara_quarters_theorem : initial_quarters + quarters_from_dad = total_quarters := by
  sorry

end sara_quarters_theorem_l534_53498


namespace largest_multiple_18_with_9_0_m_div_18_eq_555_l534_53421

/-- A function that checks if a natural number consists only of digits 9 and 0 -/
def onlyNineAndZero (n : ℕ) : Prop := sorry

/-- The largest positive multiple of 18 consisting only of digits 9 and 0 -/
def m : ℕ := sorry

theorem largest_multiple_18_with_9_0 :
  m > 0 ∧
  m % 18 = 0 ∧
  onlyNineAndZero m ∧
  ∀ k : ℕ, k > m → (k % 18 = 0 → ¬onlyNineAndZero k) :=
sorry

theorem m_div_18_eq_555 : m / 18 = 555 := sorry

end largest_multiple_18_with_9_0_m_div_18_eq_555_l534_53421


namespace solution_set_a_eq_1_min_value_range_l534_53474

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := |3*x - 1| + a*x + 3

-- Theorem 1: Solution set for a = 1
theorem solution_set_a_eq_1 :
  {x : ℝ | f 1 x ≤ 4} = Set.Icc 0 (1/2) := by sorry

-- Theorem 2: Range of a for which f(x) has a minimum value
theorem min_value_range :
  ∀ a : ℝ, (∃ x : ℝ, ∀ y : ℝ, f a x ≤ f a y) ↔ a ∈ Set.Icc (-3) 3 := by sorry

end solution_set_a_eq_1_min_value_range_l534_53474


namespace distance_is_7920_meters_l534_53414

/-- The distance traveled by a man driving at constant speed from the site of a blast -/
def distance_traveled (speed_of_sound : ℝ) (time_between_blasts : ℝ) (time_heard_second_blast : ℝ) : ℝ :=
  speed_of_sound * (time_heard_second_blast - time_between_blasts)

/-- Theorem stating that the distance traveled is 7920 meters -/
theorem distance_is_7920_meters :
  let speed_of_sound : ℝ := 330
  let time_between_blasts : ℝ := 30 * 60  -- 30 minutes in seconds
  let time_heard_second_blast : ℝ := 30 * 60 + 24  -- 30 minutes and 24 seconds in seconds
  distance_traveled speed_of_sound time_between_blasts time_heard_second_blast = 7920 := by
  sorry


end distance_is_7920_meters_l534_53414


namespace count_non_negative_l534_53492

theorem count_non_negative : 
  let numbers := [-(-4), |-1|, -|0|, (-2)^3]
  (numbers.filter (λ x => x ≥ 0)).length = 3 := by sorry

end count_non_negative_l534_53492


namespace fraction_transformation_l534_53438

theorem fraction_transformation (a b c d x : ℤ) 
  (hb : b ≠ 0) 
  (hcd : c - d ≠ 0) 
  (h_simplest : ∀ k : ℤ, k ∣ c ∧ k ∣ d → k = 1 ∨ k = -1) 
  (h_eq : (2 * a + x) * d = (b - x) * c) : 
  x = (b * c - 2 * a * d) / (d + c) := by
sorry

end fraction_transformation_l534_53438


namespace cylinder_surface_area_l534_53478

/-- The surface area of a cylinder with height 2 and base circumference 2π is 6π -/
theorem cylinder_surface_area :
  ∀ (h : ℝ) (c : ℝ),
  h = 2 →
  c = 2 * Real.pi →
  2 * Real.pi * (c / (2 * Real.pi))^2 + c * h = 6 * Real.pi :=
by sorry

end cylinder_surface_area_l534_53478


namespace vector_on_line_iff_k_eq_half_l534_53481

variable {n : Type*} [NormedAddCommGroup n] [InnerProductSpace ℝ n]

/-- A line passing through points represented by vectors p and q -/
def line (p q : n) : Set n :=
  {x | ∃ t : ℝ, x = p + t • (q - p)}

/-- The vector that should lie on the line -/
def vector_on_line (p q : n) (k : ℝ) : n :=
  k • p + (1/2) • q

/-- Theorem stating that the vector lies on the line if and only if k = 1/2 -/
theorem vector_on_line_iff_k_eq_half (p q : n) :
  ∀ k : ℝ, vector_on_line p q k ∈ line p q ↔ k = 1/2 := by
  sorry

end vector_on_line_iff_k_eq_half_l534_53481


namespace zoo_field_trip_zoo_field_trip_result_l534_53457

/-- Calculates the number of individuals left at the zoo after a field trip -/
theorem zoo_field_trip (students_per_class : ℕ) (num_classes : ℕ) (parent_chaperones : ℕ) 
  (teachers : ℕ) (students_left : ℕ) (chaperones_left : ℕ) : ℕ :=
  let initial_total := students_per_class * num_classes + parent_chaperones + teachers
  let left_total := students_left + chaperones_left
  initial_total - left_total

/-- Proves that the number of individuals left at the zoo is 15 -/
theorem zoo_field_trip_result : 
  zoo_field_trip 10 2 5 2 10 2 = 15 := by
  sorry

end zoo_field_trip_zoo_field_trip_result_l534_53457


namespace enclosing_polygon_sides_l534_53431

/-- The number of sides of the regular polygon that exactly encloses a regular decagon --/
def n : ℕ := 5

/-- The number of sides of the regular polygon being enclosed (decagon) --/
def m : ℕ := 10

theorem enclosing_polygon_sides :
  (∀ (k : ℕ), k > 2 → (360 : ℝ) / k = (720 : ℝ) / m) → n = 5 := by sorry

end enclosing_polygon_sides_l534_53431


namespace speed_to_arrive_on_time_l534_53464

/-- The speed required to arrive on time given late and early arrival conditions -/
theorem speed_to_arrive_on_time (d : ℝ) (t : ℝ) (h1 : d = 50 * (t + 1/12)) (h2 : d = 70 * (t - 1/12)) : 
  d / t = 58 := by
  sorry

end speed_to_arrive_on_time_l534_53464


namespace yogurt_production_cost_l534_53437

/-- The cost of producing three batches of yogurt given the following conditions:
  - Milk costs $1.5 per liter
  - Fruit costs $2 per kilogram
  - One batch of yogurt requires 10 liters of milk and 3 kilograms of fruit
-/
theorem yogurt_production_cost :
  let milk_cost_per_liter : ℚ := 3/2
  let fruit_cost_per_kg : ℚ := 2
  let milk_per_batch : ℚ := 10
  let fruit_per_batch : ℚ := 3
  let num_batches : ℕ := 3
  (milk_cost_per_liter * milk_per_batch + fruit_cost_per_kg * fruit_per_batch) * num_batches = 63 := by
  sorry

end yogurt_production_cost_l534_53437


namespace february_greatest_difference_l534_53440

-- Define the sales data for drummers and bugle players
def drummer_sales : Fin 5 → ℕ
  | 0 => 4  -- January
  | 1 => 5  -- February
  | 2 => 4  -- March
  | 3 => 3  -- April
  | 4 => 2  -- May

def bugle_sales : Fin 5 → ℕ
  | 0 => 3  -- January
  | 1 => 3  -- February
  | 2 => 4  -- March
  | 3 => 4  -- April
  | 4 => 3  -- May

-- Define the percentage difference function
def percentage_difference (a b : ℕ) : ℚ :=
  (max a b - min a b : ℚ) / (min a b : ℚ) * 100

-- Define a function to calculate the percentage difference for each month
def month_percentage_difference (i : Fin 5) : ℚ :=
  percentage_difference (drummer_sales i) (bugle_sales i)

-- Theorem: February has the greatest percentage difference
theorem february_greatest_difference :
  ∀ i : Fin 5, i ≠ 1 → month_percentage_difference 1 ≥ month_percentage_difference i :=
by sorry

end february_greatest_difference_l534_53440


namespace square_difference_204_202_l534_53426

theorem square_difference_204_202 : 204^2 - 202^2 = 812 := by
  sorry

end square_difference_204_202_l534_53426


namespace unknown_number_proof_l534_53443

theorem unknown_number_proof (x : ℝ) : x - (1002 / 200.4) = 3029 → x = 3034 := by
  sorry

end unknown_number_proof_l534_53443


namespace factor_polynomial_l534_53424

theorem factor_polynomial (x : ℝ) : 72 * x^5 - 90 * x^9 = -18 * x^5 * (5 * x^4 - 4) := by
  sorry

end factor_polynomial_l534_53424


namespace hospital_staff_count_l534_53489

theorem hospital_staff_count (total : ℕ) (doc_ratio nurse_ratio : ℕ) (nurse_count : ℕ) : 
  total = 280 → 
  doc_ratio = 5 →
  nurse_ratio = 9 →
  doc_ratio + nurse_ratio = (total / nurse_count) →
  nurse_count = 180 := by
sorry

end hospital_staff_count_l534_53489


namespace ellipse_equation_l534_53422

theorem ellipse_equation (a b c : ℝ) (h1 : 2 * a = 10) (h2 : c / a = 3 / 5) (h3 : b^2 = a^2 - c^2) :
  ∀ (x y : ℝ), (x^2 / 16 + y^2 / 25 = 1) ↔ (x^2 / b^2 + y^2 / a^2 = 1) :=
sorry

end ellipse_equation_l534_53422


namespace orange_count_l534_53406

/-- The number of oranges in a bin after some changes -/
def final_oranges (initial : ℕ) (removed : ℕ) (added : ℕ) : ℕ :=
  initial - removed + added

/-- Proof that given 40 initial oranges, removing 25 and adding 21 results in 36 oranges -/
theorem orange_count : final_oranges 40 25 21 = 36 := by
  sorry

end orange_count_l534_53406


namespace quadratic_intercept_distance_l534_53439

/-- A quadratic function -/
structure QuadraticFunction where
  f : ℝ → ℝ
  is_quadratic : ∃ a b c : ℝ, ∀ x, f x = a * x^2 + b * x + c

/-- The x-coordinate of the vertex of a quadratic function -/
def vertex (qf : QuadraticFunction) : ℝ := sorry

theorem quadratic_intercept_distance 
  (f g : QuadraticFunction)
  (h1 : ∀ x, g.f x = -f.f (120 - x))
  (h2 : ∃ v, g.f (vertex f) = v)
  (x₁ x₂ x₃ x₄ : ℝ)
  (h3 : x₁ < x₂ ∧ x₂ < x₃ ∧ x₃ < x₄)
  (h4 : f.f x₁ = 0 ∨ g.f x₁ = 0)
  (h5 : f.f x₂ = 0 ∨ g.f x₂ = 0)
  (h6 : f.f x₃ = 0 ∨ g.f x₃ = 0)
  (h7 : f.f x₄ = 0 ∨ g.f x₄ = 0)
  (h8 : x₃ - x₂ = 120) :
  x₄ - x₁ = 360 + 240 * Real.sqrt 2 := by sorry

end quadratic_intercept_distance_l534_53439


namespace sum_of_xyz_l534_53462

theorem sum_of_xyz (x y z : ℝ) 
  (h1 : x^2 + y^2 + x*y = 1)
  (h2 : y^2 + z^2 + y*z = 2)
  (h3 : x^2 + z^2 + x*z = 3) :
  x + y + z = Real.sqrt (3 + Real.sqrt 6) :=
sorry

end sum_of_xyz_l534_53462


namespace sin_equal_implies_isosceles_exists_isosceles_with_unequal_sines_l534_53477

/-- A triangle ABC is isosceles if at least two of its sides are equal. -/
def IsIsosceles (A B C : ℝ × ℝ) : Prop :=
  let a := dist B C
  let b := dist A C
  let c := dist A B
  a = b ∨ b = c ∨ a = c

/-- The sine of an angle in a triangle. -/
noncomputable def sinAngle (A B C : ℝ × ℝ) (vertex : ℝ × ℝ) : ℝ :=
  sorry -- Definition of sine for an angle in a triangle

theorem sin_equal_implies_isosceles (A B C : ℝ × ℝ) :
  sinAngle A B C A = sinAngle A B C B → IsIsosceles A B C :=
sorry

theorem exists_isosceles_with_unequal_sines :
  ∃ (A B C : ℝ × ℝ), IsIsosceles A B C ∧ sinAngle A B C A ≠ sinAngle A B C B :=
sorry

end sin_equal_implies_isosceles_exists_isosceles_with_unequal_sines_l534_53477


namespace product_equality_l534_53459

theorem product_equality : 469111111 * 99999999 = 46911111053088889 := by
  sorry

end product_equality_l534_53459


namespace cart_distance_theorem_l534_53441

def cart_distance (initial_distance : ℕ) (first_increment : ℕ) (second_increment : ℕ) (total_time : ℕ) : ℕ :=
  let first_section := (total_time / 2) * (2 * initial_distance + (total_time / 2 - 1) * first_increment) / 2
  let final_first_speed := initial_distance + (total_time / 2 - 1) * first_increment
  let second_section := (total_time / 2) * (2 * final_first_speed + (total_time / 2 - 1) * second_increment) / 2
  first_section + second_section

theorem cart_distance_theorem :
  cart_distance 8 10 6 30 = 4020 := by
  sorry

end cart_distance_theorem_l534_53441


namespace smallest_numbers_with_percentage_property_l534_53490

theorem smallest_numbers_with_percentage_property :
  ∃ (a b : ℕ), a = 21 ∧ b = 19 ∧
  (∀ (x y : ℕ), (95 * x = 105 * y) → (x ≥ a ∨ y ≥ b)) ∧
  (95 * a = 105 * b) := by
  sorry

end smallest_numbers_with_percentage_property_l534_53490


namespace f_properties_l534_53403

noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 1 then x^2 else x + 6/x - 6

theorem f_properties :
  (f (f (-2)) = -1/2) ∧
  (∀ x, f x ≥ 2 * Real.sqrt 6 - 6) ∧
  (∃ x, f x = 2 * Real.sqrt 6 - 6) := by
  sorry

end f_properties_l534_53403


namespace geometric_ratio_from_arithmetic_l534_53493

/-- An arithmetic sequence with a non-zero common difference -/
def arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
  d ≠ 0 ∧ ∀ n, a (n + 1) = a n + d

/-- A geometric sequence -/
def geometric_sequence (b : ℕ → ℝ) (r : ℝ) : Prop :=
  ∀ n, b (n + 1) = r * b n

/-- The theorem statement -/
theorem geometric_ratio_from_arithmetic (a : ℕ → ℝ) (d : ℝ) (b : ℕ → ℝ) :
  arithmetic_sequence a d →
  (∃ k, b k = a 1 ∧ b (k + 1) = a 3 ∧ b (k + 2) = a 7) →
  ∃ r, geometric_sequence b r ∧ r = 2 :=
sorry

end geometric_ratio_from_arithmetic_l534_53493


namespace min_score_jack_l534_53467

-- Define the parameters of the normal distribution
def mean : ℝ := 60
def std_dev : ℝ := 10

-- Define the z-score for the 90th percentile (top 10%)
def z_score_90th_percentile : ℝ := 1.28

-- Define the function to calculate the score from z-score
def score_from_z (z : ℝ) : ℝ := z * std_dev + mean

-- Define the 90th percentile score
def score_90th_percentile : ℝ := score_from_z z_score_90th_percentile

-- Define the upper bound of 2 standard deviations above the mean
def two_std_dev_above_mean : ℝ := mean + 2 * std_dev

-- State the theorem
theorem min_score_jack : 
  ∀ (score : ℕ), 
    (score ≥ ⌈score_90th_percentile⌉) ∧ 
    (↑score ≤ two_std_dev_above_mean) → 
    score ≥ 73 :=
sorry

end min_score_jack_l534_53467


namespace only_two_random_events_l534_53418

-- Define the events
inductive Event
| SameChargesRepel
| SunnyTomorrow
| FreeFallStraightLine
| ExponentialIncreasing

-- Define a predicate for random events
def IsRandomEvent : Event → Prop :=
  fun e => match e with
  | Event.SunnyTomorrow => True
  | Event.ExponentialIncreasing => True
  | _ => False

-- Theorem statement
theorem only_two_random_events :
  (∀ e : Event, IsRandomEvent e ↔ (e = Event.SunnyTomorrow ∨ e = Event.ExponentialIncreasing)) :=
by sorry

end only_two_random_events_l534_53418


namespace specific_garage_full_spots_l534_53468

/-- Represents a parking garage with given specifications -/
structure ParkingGarage where
  stories : Nat
  spotsPerLevel : Nat
  openSpotsFirstLevel : Nat
  openSpotsSecondLevel : Nat
  openSpotsThirdLevel : Nat
  openSpotsFourthLevel : Nat

/-- Calculates the number of full parking spots in the garage -/
def fullParkingSpots (garage : ParkingGarage) : Nat :=
  garage.stories * garage.spotsPerLevel - 
  (garage.openSpotsFirstLevel + garage.openSpotsSecondLevel + 
   garage.openSpotsThirdLevel + garage.openSpotsFourthLevel)

/-- Theorem stating the number of full parking spots in the specific garage -/
theorem specific_garage_full_spots :
  ∃ (garage : ParkingGarage),
    garage.stories = 4 ∧
    garage.spotsPerLevel = 100 ∧
    garage.openSpotsFirstLevel = 58 ∧
    garage.openSpotsSecondLevel = garage.openSpotsFirstLevel + 2 ∧
    garage.openSpotsThirdLevel = garage.openSpotsSecondLevel + 5 ∧
    garage.openSpotsFourthLevel = 31 ∧
    fullParkingSpots garage = 186 := by
  sorry

end specific_garage_full_spots_l534_53468


namespace dorokhov_vacation_cost_l534_53420

/-- Represents a travel agency with its pricing structure -/
structure TravelAgency where
  name : String
  price_young : ℕ
  price_old : ℕ
  age_threshold : ℕ
  discount_rate : ℚ
  is_commission : Bool

/-- Calculates the total cost for a family's vacation package -/
def calculate_total_cost (agency : TravelAgency) (num_adults num_children : ℕ) (child_age : ℕ) : ℚ :=
  sorry

/-- The Dorokhov family's vacation cost theorem -/
theorem dorokhov_vacation_cost :
  let globus : TravelAgency := {
    name := "Globus",
    price_young := 11200,
    price_old := 25400,
    age_threshold := 5,
    discount_rate := -2/100,
    is_commission := false
  }
  let around_the_world : TravelAgency := {
    name := "Around the World",
    price_young := 11400,
    price_old := 23500,
    age_threshold := 6,
    discount_rate := 1/100,
    is_commission := true
  }
  let num_adults : ℕ := 2
  let num_children : ℕ := 1
  let child_age : ℕ := 5
  
  min (calculate_total_cost globus num_adults num_children child_age)
      (calculate_total_cost around_the_world num_adults num_children child_age) = 58984 := by
  sorry

end dorokhov_vacation_cost_l534_53420


namespace black_length_is_two_l534_53409

def pencil_length : ℝ := 6
def purple_length : ℝ := 3
def blue_length : ℝ := 1

theorem black_length_is_two :
  pencil_length - purple_length - blue_length = 2 := by
  sorry

end black_length_is_two_l534_53409


namespace exists_valid_square_forming_strategy_l534_53483

/-- Represents a geometric shape on a graph paper --/
structure Shape :=
  (area : ℝ)
  (is_square : Bool)

/-- Represents a cutting strategy for a shape --/
structure CuttingStrategy :=
  (num_parts : ℕ)
  (all_triangles : Bool)

/-- The original figure given in the problem --/
def original_figure : Shape :=
  { area := 1, is_square := false }

/-- Checks if a cutting strategy is valid for the given conditions --/
def is_valid_strategy (s : CuttingStrategy) : Bool :=
  (s.num_parts ≤ 4) ∨ (s.num_parts ≤ 5 ∧ s.all_triangles)

/-- Theorem stating that there exists a valid cutting strategy to form a square --/
theorem exists_valid_square_forming_strategy :
  ∃ (s : CuttingStrategy) (result : Shape),
    is_valid_strategy s ∧
    result.is_square ∧
    result.area = original_figure.area :=
  sorry

end exists_valid_square_forming_strategy_l534_53483


namespace cubic_increasing_iff_a_positive_l534_53461

/-- A cubic function f(x) = ax³ + x is increasing on ℝ if and only if a > 0 -/
theorem cubic_increasing_iff_a_positive (a : ℝ) :
  (∀ x : ℝ, StrictMono (fun x => a * x^3 + x)) ↔ a > 0 := by
  sorry

end cubic_increasing_iff_a_positive_l534_53461


namespace job_completion_time_l534_53453

/-- 
Given two people who can complete a job independently in 10 and 15 days respectively,
this theorem proves that they can complete the job together in 6 days.
-/
theorem job_completion_time 
  (ram_time : ℝ) 
  (gohul_time : ℝ) 
  (h1 : ram_time = 10) 
  (h2 : gohul_time = 15) : 
  (ram_time * gohul_time) / (ram_time + gohul_time) = 6 := by
  sorry

end job_completion_time_l534_53453


namespace mikes_file_space_l534_53432

/-- The amount of space Mike's files take up on his disk drive. -/
def space_taken_by_files (total_space : ℕ) (space_left : ℕ) : ℕ :=
  total_space - space_left

/-- Proof that Mike's files take up 26 GB of space. -/
theorem mikes_file_space :
  space_taken_by_files 28 2 = 26 := by
  sorry

end mikes_file_space_l534_53432


namespace smallest_sector_angle_l534_53471

def circle_sectors (n : ℕ) (a : ℕ → ℕ) : Prop :=
  (∀ i, i ∈ Finset.range n → a i > 0) ∧
  (∀ i j k, i < j ∧ j < k → a j - a i = a k - a j) ∧
  (Finset.sum (Finset.range n) a = 360)

theorem smallest_sector_angle :
  ∀ a : ℕ → ℕ, circle_sectors 16 a → ∃ i, a i = 15 ∧ ∀ j, a j ≥ a i := by
  sorry

end smallest_sector_angle_l534_53471


namespace convention_handshakes_specific_l534_53456

/-- The number of handshakes in a convention with specified conditions -/
def convention_handshakes (num_companies : ℕ) (reps_per_company : ℕ) : ℕ :=
  let total_participants := num_companies * reps_per_company
  let handshakes_per_person := total_participants - reps_per_company
  (total_participants * handshakes_per_person) / 2

theorem convention_handshakes_specific : convention_handshakes 5 5 = 250 := by
  sorry

#eval convention_handshakes 5 5

end convention_handshakes_specific_l534_53456


namespace imaginary_part_of_one_minus_i_squared_l534_53460

theorem imaginary_part_of_one_minus_i_squared (i : ℂ) : Complex.im ((1 - i)^2) = -2 :=
by
  sorry

end imaginary_part_of_one_minus_i_squared_l534_53460


namespace solution_set_part1_range_of_a_part2_l534_53436

-- Define the function f(x) with parameter a
def f (a : ℝ) (x : ℝ) : ℝ := |3 * x - 1| + a * x + 3

-- Part 1: Prove the solution set for f(x) ≤ 5 when a = 1
theorem solution_set_part1 : 
  {x : ℝ | f 1 x ≤ 5} = {x : ℝ | -1/2 ≤ x ∧ x ≤ 3/4} := by sorry

-- Part 2: Prove the range of a for which f(x) has a minimum value
theorem range_of_a_part2 :
  ∀ a : ℝ, (∃ x : ℝ, ∀ y : ℝ, f a x ≤ f a y) → -3 ≤ a ∧ a ≤ 3 := by sorry

end solution_set_part1_range_of_a_part2_l534_53436
