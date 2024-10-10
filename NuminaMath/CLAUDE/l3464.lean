import Mathlib

namespace intersection_locus_is_hyperbola_l3464_346479

/-- The locus of points (x, y) satisfying the given system of equations forms a hyperbola -/
theorem intersection_locus_is_hyperbola :
  ∀ (x y u : ℝ), 
  (2 * u * x - 3 * y - 4 * u = 0) →
  (x - 3 * u * y + 4 = 0) →
  ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ x^2 / a^2 - y^2 / b^2 = 1 :=
by sorry

end intersection_locus_is_hyperbola_l3464_346479


namespace sine_tangent_relation_l3464_346447

theorem sine_tangent_relation (α : Real) (h : 0 < α ∧ α < Real.pi) :
  (∃ β, (Real.sqrt 2 / 2 < Real.sin β ∧ Real.sin β < 1) ∧ ¬(Real.tan β > 1)) ∧
  (∀ γ, Real.tan γ > 1 → Real.sqrt 2 / 2 < Real.sin γ ∧ Real.sin γ < 1) :=
by sorry

end sine_tangent_relation_l3464_346447


namespace magnitude_of_OP_l3464_346431

/-- Given vectors OA and OB, and the relation between AP and AB, prove the magnitude of OP --/
theorem magnitude_of_OP (OA OB OP : ℝ × ℝ) : 
  OA = (1, 2) → 
  OB = (-2, -1) → 
  2 * (OP - OA) = OB - OA → 
  Real.sqrt ((OP.1)^2 + (OP.2)^2) = Real.sqrt 2 / 2 := by
  sorry

end magnitude_of_OP_l3464_346431


namespace fence_length_not_eighteen_l3464_346461

theorem fence_length_not_eighteen (length width : ℝ) : 
  length = 6 → width = 3 → 
  ¬(length + 2 * width = 18 ∨ 2 * length + width = 18) :=
by
  sorry

end fence_length_not_eighteen_l3464_346461


namespace cloud_computing_analysis_l3464_346449

/-- Cloud computing market data --/
structure MarketData :=
  (year : ℕ)
  (market_scale : ℝ)

/-- Regression equation coefficients --/
structure RegressionCoefficients :=
  (b : ℝ)
  (a : ℝ)

/-- Cloud computing market analysis --/
theorem cloud_computing_analysis 
  (data : List MarketData)
  (sum_ln_y : ℝ)
  (sum_x_ln_y : ℝ)
  (initial_error_variance : ℝ → ℝ)
  (initial_probability : ℝ)
  (new_error_variance : ℝ → ℝ) :
  ∃ (coef : RegressionCoefficients) 
    (new_probability : ℝ) 
    (cost_decrease : ℝ),
  (coef.b = 0.386 ∧ coef.a = 6.108) ∧
  (new_probability = 0.9545) ∧
  (cost_decrease = 3) :=
by
  sorry

end cloud_computing_analysis_l3464_346449


namespace problem_statement_l3464_346464

theorem problem_statement : 4 * Real.sqrt (1/2) + 3 * Real.sqrt (1/3) - Real.sqrt 8 = Real.sqrt 3 := by
  sorry

end problem_statement_l3464_346464


namespace g_neg_two_eq_eleven_l3464_346488

/-- The function g(x) = x^2 - 2x + 3 -/
def g (x : ℝ) : ℝ := x^2 - 2*x + 3

/-- Theorem: g(-2) = 11 -/
theorem g_neg_two_eq_eleven : g (-2) = 11 := by
  sorry

end g_neg_two_eq_eleven_l3464_346488


namespace smallest_square_containing_circle_l3464_346481

theorem smallest_square_containing_circle (r : ℝ) (h : r = 4) :
  (2 * r) ^ 2 = 64 := by
  sorry

end smallest_square_containing_circle_l3464_346481


namespace quilt_remaining_squares_l3464_346407

/-- Given a quilt with 16 squares on each side and 25% of it already sewn,
    prove that the number of remaining squares to sew is 24. -/
theorem quilt_remaining_squares (squares_per_side : ℕ) (percent_sewn : ℚ) : 
  squares_per_side = 16 →
  percent_sewn = 1/4 →
  (2 * squares_per_side : ℕ) - (percent_sewn * (2 * squares_per_side : ℕ) : ℚ).num = 24 := by
  sorry

end quilt_remaining_squares_l3464_346407


namespace line_parallel_perpendicular_implies_planes_perpendicular_l3464_346482

-- Define the types for lines and planes
variable (L : Type) [LinearOrder L]
variable (P : Type)

-- Define the relationships
variable (parallel : L → P → Prop)
variable (perpendicular : L → P → Prop)
variable (plane_perpendicular : P → P → Prop)

-- State the theorem
theorem line_parallel_perpendicular_implies_planes_perpendicular
  (ι : L) (α β : P) (h1 : parallel ι α) (h2 : perpendicular ι β) :
  plane_perpendicular α β :=
sorry

end line_parallel_perpendicular_implies_planes_perpendicular_l3464_346482


namespace perpendicular_lines_m_value_l3464_346462

/-- 
Given two lines in the xy-plane:
  Line1: x - y - 2 = 0
  Line2: mx + y = 0
If Line1 is perpendicular to Line2, then m = 1
-/
theorem perpendicular_lines_m_value (m : ℝ) : 
  (∀ x y : ℝ, x - y - 2 = 0 → mx + y = 0 → (1 : ℝ) * m = -1) → 
  m = 1 := by sorry

end perpendicular_lines_m_value_l3464_346462


namespace function_lower_bound_l3464_346499

theorem function_lower_bound
  (f : ℝ → ℝ)
  (h_cont : ContinuousOn f (Set.Ioi 0))
  (h_ineq : ∀ x > 0, f (x^2) ≥ f x)
  (h_f1 : f 1 = 5) :
  ∀ x > 0, f x ≥ 5 :=
by sorry

end function_lower_bound_l3464_346499


namespace middle_card_is_four_l3464_346476

/-- Represents a valid triple of card numbers -/
def ValidTriple (a b c : ℕ) : Prop :=
  0 < a ∧ a < b ∧ b < c ∧ a + b + c = 15

/-- Predicate for uncertainty about other numbers given the left card -/
def LeftUncertain (a : ℕ) : Prop :=
  ∃ b₁ c₁ b₂ c₂, b₁ ≠ b₂ ∧ ValidTriple a b₁ c₁ ∧ ValidTriple a b₂ c₂

/-- Predicate for uncertainty about other numbers given the right card -/
def RightUncertain (c : ℕ) : Prop :=
  ∃ a₁ b₁ a₂ b₂, a₁ ≠ a₂ ∧ ValidTriple a₁ b₁ c ∧ ValidTriple a₂ b₂ c

/-- Predicate for uncertainty about other numbers given the middle card -/
def MiddleUncertain (b : ℕ) : Prop :=
  ∃ a₁ c₁ a₂ c₂, a₁ ≠ a₂ ∧ ValidTriple a₁ b c₁ ∧ ValidTriple a₂ b c₂

theorem middle_card_is_four :
  ∀ a b c : ℕ,
    ValidTriple a b c →
    (∀ x, ValidTriple x b c → LeftUncertain x) →
    (∀ z, ValidTriple a b z → RightUncertain z) →
    MiddleUncertain b →
    b = 4 := by
  sorry

end middle_card_is_four_l3464_346476


namespace expression_value_l3464_346485

theorem expression_value (x y : ℚ) (hx : x ≠ 0) (hy : y ≠ 0) :
  x / |x| + |y| / y = 2 ∨ x / |x| + |y| / y = 0 ∨ x / |x| + |y| / y = -2 := by
  sorry

end expression_value_l3464_346485


namespace invalid_reasoning_l3464_346457

-- Define the types of reasoning
inductive ReasoningType
  | Analogy
  | Inductive
  | Deductive

-- Define the concept of valid reasoning
def isValidReasoning (r : ReasoningType) : Prop :=
  match r with
  | ReasoningType.Analogy => true
  | ReasoningType.Inductive => true
  | ReasoningType.Deductive => true

-- Define the reasoning options
def optionA : ReasoningType := ReasoningType.Analogy
def optionB : ReasoningType := ReasoningType.Inductive
def optionC : ReasoningType := ReasoningType.Inductive
def optionD : ReasoningType := ReasoningType.Inductive

-- Theorem to prove
theorem invalid_reasoning :
  isValidReasoning optionA ∧
  isValidReasoning optionB ∧
  ¬(isValidReasoning optionC) ∧
  isValidReasoning optionD :=
by sorry

end invalid_reasoning_l3464_346457


namespace square_root_625_divided_by_5_l3464_346424

theorem square_root_625_divided_by_5 : Real.sqrt 625 / 5 = 5 := by
  sorry

end square_root_625_divided_by_5_l3464_346424


namespace opposite_of_negative_2023_l3464_346456

theorem opposite_of_negative_2023 : -((-2023) : ℤ) = (2023 : ℤ) := by
  sorry

end opposite_of_negative_2023_l3464_346456


namespace square_sum_reciprocal_l3464_346445

theorem square_sum_reciprocal (x : ℝ) (h : x + 1/x = 3) : x^2 + 1/x^2 = 7 := by
  sorry

end square_sum_reciprocal_l3464_346445


namespace eating_contest_l3464_346492

/-- Eating contest problem -/
theorem eating_contest (hot_dog_weight burger_weight pie_weight : ℕ)
  (mason_hotdog_multiplier : ℕ) (noah_burger_count : ℕ) (mason_hotdog_total_weight : ℕ)
  (h1 : hot_dog_weight = 2)
  (h2 : burger_weight = 5)
  (h3 : pie_weight = 10)
  (h4 : mason_hotdog_multiplier = 3)
  (h5 : noah_burger_count = 8)
  (h6 : mason_hotdog_total_weight = 30) :
  ∃ (jacob_pie_count : ℕ),
    jacob_pie_count = 5 ∧
    mason_hotdog_total_weight = jacob_pie_count * mason_hotdog_multiplier * hot_dog_weight :=
by
  sorry


end eating_contest_l3464_346492


namespace jesse_room_area_l3464_346451

/-- Calculates the area of a rectangle given its length and width -/
def rectangleArea (length width : ℝ) : ℝ := length * width

/-- Represents an L-shaped room with two rectangular parts -/
structure LShapedRoom where
  length1 : ℝ
  width1 : ℝ
  length2 : ℝ
  width2 : ℝ

/-- Calculates the total area of an L-shaped room -/
def totalArea (room : LShapedRoom) : ℝ :=
  rectangleArea room.length1 room.width1 + rectangleArea room.length2 room.width2

/-- Theorem: The total area of Jesse's L-shaped room is 120 square feet -/
theorem jesse_room_area :
  let room : LShapedRoom := { length1 := 12, width1 := 8, length2 := 6, width2 := 4 }
  totalArea room = 120 := by
  sorry

end jesse_room_area_l3464_346451


namespace hazel_fish_count_l3464_346487

theorem hazel_fish_count (total : ℕ) (father : ℕ) (hazel : ℕ) : 
  total = 94 → father = 46 → total = father + hazel → hazel = 48 := by
  sorry

end hazel_fish_count_l3464_346487


namespace crayon_selection_theorem_l3464_346439

/-- The number of ways to select k items from n items -/
def choose (n k : ℕ) : ℕ := Nat.choose n k

/-- The total number of crayons in the box -/
def total_crayons : ℕ := 15

/-- The number of red crayons in the box -/
def red_crayons : ℕ := 3

/-- The number of crayons to be selected -/
def selected_crayons : ℕ := 5

/-- The number of red crayons that must be selected -/
def selected_red : ℕ := 2

/-- The number of ways to select crayons under the given conditions -/
def ways_to_select : ℕ := choose red_crayons selected_red * choose (total_crayons - red_crayons) (selected_crayons - selected_red)

theorem crayon_selection_theorem : ways_to_select = 660 := by
  sorry

end crayon_selection_theorem_l3464_346439


namespace award_distribution_probability_l3464_346444

def num_classes : ℕ := 4
def num_awards : ℕ := 8

def distribute_awards (n : ℕ) (k : ℕ) : ℕ := Nat.choose (n + k - 1) (k - 1)

theorem award_distribution_probability :
  let total_distributions := distribute_awards (num_awards - num_classes) num_classes
  let favorable_distributions := distribute_awards ((num_awards - num_classes) - 1) (num_classes - 1)
  (favorable_distributions : ℚ) / total_distributions = 2 / 7 := by
  sorry

end award_distribution_probability_l3464_346444


namespace cube_product_theorem_l3464_346493

theorem cube_product_theorem : 
  let f (n : ℕ) := (n^3 - 1) / (n^3 + 1)
  (f 3) * (f 4) * (f 5) * (f 6) * (f 7) * (f 8) = 73 / 256 := by
  sorry

end cube_product_theorem_l3464_346493


namespace distance_to_specific_line_l3464_346484

/-- Polar coordinates of a point -/
structure PolarPoint where
  r : ℝ
  θ : ℝ

/-- Polar equation of a line -/
structure PolarLine where
  equation : ℝ → ℝ → Prop

/-- Distance from a point to a line -/
def distanceToLine (p : PolarPoint) (l : PolarLine) : ℝ := sorry

theorem distance_to_specific_line :
  let A : PolarPoint := ⟨2, 7 * π / 4⟩
  let L : PolarLine := ⟨fun ρ θ ↦ ρ * Real.sin (θ + π / 4) = Real.sqrt 2 / 2⟩
  distanceToLine A L = Real.sqrt 2 / 2 := by
  sorry

end distance_to_specific_line_l3464_346484


namespace tangent_line_circle_m_value_l3464_346468

/-- A circle in the xy-plane -/
structure Circle where
  equation : ℝ → ℝ → ℝ → Prop

/-- A line in the xy-plane -/
structure Line where
  equation : ℝ → ℝ → ℝ → Prop

/-- Predicate to check if a line is tangent to a circle -/
def IsTangent (l : Line) (c : Circle) : Prop := sorry

/-- The main theorem -/
theorem tangent_line_circle_m_value (m : ℝ) :
  let c : Circle := ⟨λ x y m => x^2 + y^2 = m⟩
  let l : Line := ⟨λ x y m => x + y + m = 0⟩
  IsTangent l c → m = 2 := by sorry

end tangent_line_circle_m_value_l3464_346468


namespace no_solution_exponential_equation_l3464_346417

theorem no_solution_exponential_equation :
  ¬ ∃ y : ℝ, (16 : ℝ) ^ (3 * y) = (64 : ℝ) ^ (2 * y + 1) :=
by sorry

end no_solution_exponential_equation_l3464_346417


namespace roots_sum_and_product_l3464_346405

theorem roots_sum_and_product (a b : ℝ) : 
  (a^4 - 6*a^2 - 4*a + 1 = 0) → 
  (b^4 - 6*b^2 - 4*b + 1 = 0) → 
  (a ≠ b) →
  (∀ x : ℝ, x^4 - 6*x^2 - 4*x + 1 = 0 → x = a ∨ x = b) →
  a * b + a + b = -1 := by
sorry

end roots_sum_and_product_l3464_346405


namespace triangulation_reconstruction_l3464_346435

/-- A convex polygon represented by its vertices -/
structure ConvexPolygon where
  vertices : List ℝ × ℝ
  is_convex : sorry

/-- A triangulation of a convex polygon -/
structure Triangulation (P : ConvexPolygon) where
  diagonals : List (ℕ × ℕ)
  is_valid : sorry

/-- The number of triangles adjacent to each vertex in a triangulation -/
def adjacentTriangles (P : ConvexPolygon) (T : Triangulation P) : List ℕ :=
  sorry

/-- Theorem stating that a triangulation can be uniquely reconstructed from adjacent triangle counts -/
theorem triangulation_reconstruction
  (P : ConvexPolygon)
  (T1 T2 : Triangulation P)
  (h : adjacentTriangles P T1 = adjacentTriangles P T2) :
  T1 = T2 :=
sorry

end triangulation_reconstruction_l3464_346435


namespace four_propositions_l3464_346454

-- Define the propositions
def opposite_numbers (x y : ℝ) : Prop := x = -y

def has_real_roots (a b c : ℝ) : Prop :=
  ∃ x : ℝ, a * x^2 + b * x + c = 0

def congruent_triangles (t1 t2 : Set ℝ × Set ℝ) : Prop :=
  sorry  -- Definition of congruent triangles

def equal_areas (t1 t2 : Set ℝ × Set ℝ) : Prop :=
  sorry  -- Definition of equal areas for triangles

def right_triangle (t : Set ℝ × Set ℝ) : Prop :=
  sorry  -- Definition of right triangle

def has_two_acute_angles (t : Set ℝ × Set ℝ) : Prop :=
  sorry  -- Definition of triangle with two acute angles

-- Theorem to prove
theorem four_propositions :
  (∀ x y : ℝ, opposite_numbers x y → x + y = 0) ∧
  (∀ q : ℝ, ¬(has_real_roots 1 2 q) → q > 1) ∧
  ¬(∀ t1 t2 : Set ℝ × Set ℝ, ¬(congruent_triangles t1 t2) → ¬(equal_areas t1 t2)) ∧
  ¬(∀ t : Set ℝ × Set ℝ, has_two_acute_angles t → right_triangle t) :=
by
  sorry

end four_propositions_l3464_346454


namespace matrix_inverse_equality_l3464_346437

/-- Given a 3x3 matrix B with a variable d in the (2,3) position, prove that if B^(-1) = k * B, then d = 13/9 and k = -329/52 -/
theorem matrix_inverse_equality (d k : ℚ) : 
  let B : Matrix (Fin 3) (Fin 3) ℚ := !![1, 2, 3; 4, 5, d; 6, 7, 8]
  (B⁻¹ = k • B) → (d = 13/9 ∧ k = -329/52) := by
  sorry

end matrix_inverse_equality_l3464_346437


namespace arithmetic_sequence_problem_l3464_346406

/-- An arithmetic sequence with common difference -2 and S_5 = 10 has a_100 = -192 -/
theorem arithmetic_sequence_problem (a : ℕ → ℤ) (S : ℕ → ℤ) : 
  (∀ n, a (n + 1) - a n = -2) →  -- arithmetic sequence with common difference -2
  (S 5 = 10) →                   -- sum of first 5 terms is 10
  (∀ n, S n = n * a 1 + n * (n - 1) * (-1)) →  -- formula for sum of arithmetic sequence
  (a 100 = -192) :=              -- a_100 = -192
by sorry

end arithmetic_sequence_problem_l3464_346406


namespace distance_ratio_bound_l3464_346413

/-- Given n points on a plane with maximum distance D and minimum distance d between any two points,
    the ratio of maximum to minimum distance is greater than (√(nπ)/2) - 1. -/
theorem distance_ratio_bound (n : ℕ) (D d : ℝ) (h_pos : 0 < d) (h_max : d ≤ D) :
  D / d > Real.sqrt (n * Real.pi) / 2 - 1 := by
  sorry

end distance_ratio_bound_l3464_346413


namespace line_chart_most_appropriate_l3464_346433

/-- Represents a chart type -/
inductive ChartType
| LineChart
| BarChart
| PieChart
| ScatterPlot

/-- Represents the requirements for a temperature chart -/
structure TemperatureChartRequirements where
  showsChangeOverTime : Bool
  reflectsAmountAndChanges : Bool
  showsIncreasesAndDecreases : Bool

/-- Defines the properties of a line chart -/
def lineChartProperties : TemperatureChartRequirements :=
  { showsChangeOverTime := true
  , reflectsAmountAndChanges := true
  , showsIncreasesAndDecreases := true }

/-- Determines if a chart type is appropriate for the given requirements -/
def isAppropriateChart (c : ChartType) (r : TemperatureChartRequirements) : Bool :=
  match c with
  | ChartType.LineChart => r.showsChangeOverTime ∧ r.reflectsAmountAndChanges ∧ r.showsIncreasesAndDecreases
  | _ => false

/-- Theorem: A line chart is the most appropriate for recording temperature changes of a feverish patient -/
theorem line_chart_most_appropriate :
  isAppropriateChart ChartType.LineChart lineChartProperties = true :=
sorry

end line_chart_most_appropriate_l3464_346433


namespace mrs_sheridan_fish_count_l3464_346416

/-- The number of fish Mrs. Sheridan initially had -/
def initial_fish : ℕ := 22

/-- The number of fish Mrs. Sheridan's sister gave her -/
def additional_fish : ℕ := 47

/-- The total number of fish Mrs. Sheridan has now -/
def total_fish : ℕ := initial_fish + additional_fish

theorem mrs_sheridan_fish_count : total_fish = 69 := by
  sorry

end mrs_sheridan_fish_count_l3464_346416


namespace bike_price_proof_l3464_346440

theorem bike_price_proof (upfront_payment : ℝ) (upfront_percentage : ℝ) (total_price : ℝ) : 
  upfront_payment = 150 ∧ 
  upfront_percentage = 0.1 ∧ 
  upfront_payment = upfront_percentage * total_price →
  total_price = 1500 :=
by sorry

end bike_price_proof_l3464_346440


namespace expand_product_l3464_346458

theorem expand_product (x : ℝ) : (2*x + 3) * (x + 5) = 2*x^2 + 13*x + 15 := by
  sorry

end expand_product_l3464_346458


namespace tan70_cos10_sqrt3tan20_minus1_eq_neg1_l3464_346441

theorem tan70_cos10_sqrt3tan20_minus1_eq_neg1 :
  Real.tan (70 * π / 180) * Real.cos (10 * π / 180) * (Real.sqrt 3 * Real.tan (20 * π / 180) - 1) = -1 := by
  sorry

end tan70_cos10_sqrt3tan20_minus1_eq_neg1_l3464_346441


namespace ben_initial_eggs_l3464_346414

/-- The number of eggs Ben had initially -/
def initial_eggs : ℕ := sorry

/-- The number of eggs Ben ate in the morning -/
def morning_eggs : ℕ := 4

/-- The number of eggs Ben ate in the afternoon -/
def afternoon_eggs : ℕ := 3

/-- The number of eggs Ben has left -/
def remaining_eggs : ℕ := 13

/-- Theorem stating that Ben initially had 20 eggs -/
theorem ben_initial_eggs : initial_eggs = 20 := by
  sorry

end ben_initial_eggs_l3464_346414


namespace triangle_circumcircle_l3464_346421

-- Define the triangle ABC
def A : ℝ × ℝ := (1, 3)

-- Define the line BC
def line_BC (x y : ℝ) : Prop := y - 1 = 0

-- Define the median from A to BC
def median_A (x y : ℝ) : Prop := x - 3*y + 4 = 0

-- Define the circumcircle equation
def circumcircle (x y : ℝ) : Prop := (x - 1)^2 + (y - 1)^2 = 4

-- Theorem statement
theorem triangle_circumcircle : 
  ∀ (B C : ℝ × ℝ),
  line_BC B.1 B.2 ∧ line_BC C.1 C.2 ∧
  median_A ((B.1 + C.1)/2) ((B.2 + C.2)/2) →
  circumcircle B.1 B.2 ∧ circumcircle C.1 C.2 ∧ circumcircle A.1 A.2 :=
by sorry

end triangle_circumcircle_l3464_346421


namespace minkyung_height_calculation_l3464_346401

def haeun_height : ℝ := 1.56
def nayeon_height : ℝ := haeun_height - 0.14
def minkyung_height : ℝ := nayeon_height + 0.27

theorem minkyung_height_calculation : minkyung_height = 1.69 := by
  sorry

end minkyung_height_calculation_l3464_346401


namespace fencemaker_problem_l3464_346427

/-- Given a rectangular yard with one side of 40 feet and an area of 480 square feet,
    the perimeter minus one side is equal to 64 feet. -/
theorem fencemaker_problem (length width : ℝ) : 
  length = 40 ∧ 
  length * width = 480 ∧ 
  width > 0 → 
  2 * width + length = 64 := by
  sorry

end fencemaker_problem_l3464_346427


namespace square_sum_equals_three_l3464_346459

theorem square_sum_equals_three (x y z : ℝ) 
  (h1 : x - y - z = 3) 
  (h2 : y * z - x * y - x * z = 3) : 
  x^2 + y^2 + z^2 = 3 := by
sorry

end square_sum_equals_three_l3464_346459


namespace sum_of_digits_l3464_346432

theorem sum_of_digits (a b c d e : ℕ) : 
  (10 ≤ 10*a + b) ∧ (10*a + b ≤ 99) ∧
  (100 ≤ 100*c + 10*d + e) ∧ (100*c + 10*d + e ≤ 999) ∧
  (10*a + b + 100*c + 10*d + e = 1079) →
  a + b + c + d + e = 35 := by
sorry

end sum_of_digits_l3464_346432


namespace smallest_divisible_by_15_16_18_l3464_346463

theorem smallest_divisible_by_15_16_18 : 
  ∃ n : ℕ+, (∀ m : ℕ+, (15 ∣ m) ∧ (16 ∣ m) ∧ (18 ∣ m) → n ≤ m) ∧ 
             (15 ∣ n) ∧ (16 ∣ n) ∧ (18 ∣ n) :=
by
  use 720
  sorry

end smallest_divisible_by_15_16_18_l3464_346463


namespace inequality_proof_l3464_346404

theorem inequality_proof (a b c d : ℝ) (h1 : a > b) (h2 : c > d) : d - a < c - b := by
  sorry

end inequality_proof_l3464_346404


namespace octal_arithmetic_sum_1_to_30_l3464_346442

/-- Represents a number in base 8 -/
def OctalNum := Nat

/-- Convert a decimal number to its octal representation -/
def toOctal (n : Nat) : OctalNum := sorry

/-- Convert an octal number to its decimal representation -/
def fromOctal (n : OctalNum) : Nat := sorry

/-- Sum of arithmetic series in base 8 -/
def octalArithmeticSum (first last : OctalNum) : OctalNum := sorry

theorem octal_arithmetic_sum_1_to_30 :
  octalArithmeticSum (toOctal 1) (toOctal 24) = toOctal 300 := by
  sorry

end octal_arithmetic_sum_1_to_30_l3464_346442


namespace jelly_bean_match_probability_l3464_346489

/-- Represents the distribution of jelly beans for a person -/
structure JellyBeanDistribution where
  green : ℕ
  blue : ℕ
  red : ℕ

/-- Calculates the total number of jelly beans -/
def JellyBeanDistribution.total (d : JellyBeanDistribution) : ℕ :=
  d.green + d.blue + d.red

/-- Lila's jelly bean distribution -/
def lila_beans : JellyBeanDistribution :=
  { green := 1, blue := 1, red := 1 }

/-- Max's jelly bean distribution -/
def max_beans : JellyBeanDistribution :=
  { green := 2, blue := 1, red := 3 }

/-- Calculates the probability of picking a specific color -/
def pick_probability (d : JellyBeanDistribution) (color : ℕ) : ℚ :=
  color / d.total

/-- Calculates the probability of both people picking the same color -/
def match_probability (d1 d2 : JellyBeanDistribution) : ℚ :=
  pick_probability d1 d1.green * pick_probability d2 d2.green +
  pick_probability d1 d1.blue * pick_probability d2 d2.blue +
  pick_probability d1 d1.red * pick_probability d2 d2.red

theorem jelly_bean_match_probability :
  match_probability lila_beans max_beans = 1/3 := by
  sorry

end jelly_bean_match_probability_l3464_346489


namespace translate_upward_5_units_l3464_346496

/-- Represents a linear function of the form y = mx + b -/
structure LinearFunction where
  m : ℝ  -- slope
  b : ℝ  -- y-intercept

/-- Translates a linear function vertically by a given amount -/
def translateVertically (f : LinearFunction) (δ : ℝ) : LinearFunction :=
  { m := f.m, b := f.b + δ }

/-- The theorem to prove -/
theorem translate_upward_5_units :
  let f : LinearFunction := { m := 2, b := -3 }
  let g : LinearFunction := translateVertically f 5
  g = { m := 2, b := 2 } := by sorry

end translate_upward_5_units_l3464_346496


namespace circle_point_marking_l3464_346415

/-- The number of points on the circle -/
def n : ℕ := 2021

/-- 
Given n points on a circle, prove that the smallest positive integer b 
such that b(b+1)/2 is divisible by n is 67.
-/
theorem circle_point_marking (b : ℕ) : 
  (∀ k < b, ¬(2 ∣ k * (k + 1) ∧ n ∣ k * (k + 1))) ∧ 
  (2 ∣ b * (b + 1) ∧ n ∣ b * (b + 1)) → 
  b = 67 := by sorry

end circle_point_marking_l3464_346415


namespace min_value_reciprocal_sum_min_value_product_l3464_346420

-- Part 1
theorem min_value_reciprocal_sum (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 2 * x + y = 1) :
  1 / x + 1 / y ≥ 3 + 2 * Real.sqrt 2 := by sorry

-- Part 2
theorem min_value_product (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x + 8 * y - x * y = 0) :
  x * y ≥ 32 := by sorry

end min_value_reciprocal_sum_min_value_product_l3464_346420


namespace max_profit_optimal_plan_model_b_units_l3464_346418

/-- Represents the profit function for tablet sales -/
def profit_function (x : ℕ) : ℝ := -100 * x + 10000

/-- Represents the total cost function for tablet purchases -/
def total_cost (x : ℕ) : ℝ := 1600 * x + 2500 * (20 - x)

/-- Theorem stating the maximum profit and optimal purchasing plan -/
theorem max_profit_optimal_plan :
  ∃ (x : ℕ),
    x ≤ 20 ∧
    total_cost x ≤ 39200 ∧
    profit_function x ≥ 8500 ∧
    (∀ (y : ℕ), y ≤ 20 → total_cost y ≤ 39200 → profit_function y ≥ 8500 →
      profit_function x ≥ profit_function y) ∧
    x = 12 ∧
    profit_function x = 8800 :=
by sorry

/-- Corollary stating the number of units for model B tablets -/
theorem model_b_units (x : ℕ) (h : x = 12) : 20 - x = 8 :=
by sorry

end max_profit_optimal_plan_model_b_units_l3464_346418


namespace quadratic_factorization_l3464_346478

theorem quadratic_factorization (E F : ℤ) :
  (∀ y : ℝ, 15 * y^2 - 82 * y + 48 = (E * y - 16) * (F * y - 3)) →
  E * F + E = 20 := by
sorry

end quadratic_factorization_l3464_346478


namespace equation_solution_l3464_346480

theorem equation_solution : 
  ∃! x : ℚ, (53 - 3*x)^(1/4) + (39 + 3*x)^(1/4) = 5 :=
by
  -- The unique solution is x = -23/3
  use -23/3
  sorry

end equation_solution_l3464_346480


namespace projectile_max_height_l3464_346429

/-- The height function of the projectile -/
def h (t : ℝ) : ℝ := -20 * t^2 + 100 * t + 50

/-- The maximum height reached by the projectile -/
def max_height : ℝ := 175

/-- Theorem stating that the maximum height reached by the projectile is 175 meters -/
theorem projectile_max_height :
  ∀ t : ℝ, h t ≤ max_height :=
by sorry

end projectile_max_height_l3464_346429


namespace f_inequality_range_l3464_346450

-- Define the function f
def f (x : ℝ) : ℝ := |x - 1| + |x + 1|

-- State the theorem
theorem f_inequality_range (a : ℝ) :
  (∀ x : ℝ, f x - a ≤ 0) ↔ a ≥ 4 := by
  sorry

end f_inequality_range_l3464_346450


namespace two_numbers_sum_667_lcm_gcd_120_l3464_346465

theorem two_numbers_sum_667_lcm_gcd_120 :
  ∀ a b : ℕ,
  a + b = 667 →
  (Nat.lcm a b) / (Nat.gcd a b) = 120 →
  ((a = 552 ∧ b = 115) ∨ (a = 115 ∧ b = 552) ∨ (a = 435 ∧ b = 232) ∨ (a = 232 ∧ b = 435)) :=
by sorry

end two_numbers_sum_667_lcm_gcd_120_l3464_346465


namespace water_depth_l3464_346411

/-- The depth of water given heights of two people -/
theorem water_depth (ron_height dean_height water_depth : ℕ) : 
  ron_height = 13 →
  dean_height = ron_height + 4 →
  water_depth = 15 * dean_height →
  water_depth = 255 :=
by
  sorry

#check water_depth

end water_depth_l3464_346411


namespace tangent_line_to_parabola_l3464_346472

theorem tangent_line_to_parabola (b : ℝ) :
  (∀ x y : ℝ, y = -2*x + b → y^2 = 8*x → (∀ ε > 0, ∃ δ > 0, ∀ x' y', 
    ((x' - x)^2 + (y' - y)^2 < δ^2) → (y' + 2*x' - b)^2 > 0 ∨ 
    ((y')^2 - 8*x')^2 > 0)) →
  b = -1 :=
sorry

end tangent_line_to_parabola_l3464_346472


namespace min_value_theorem_l3464_346473

theorem min_value_theorem (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x + 3 * y = 5 * x * y) :
  (∀ a b : ℝ, a > 0 → b > 0 → a + 3 * b = 5 * a * b → 3 * a + 4 * b ≥ 3 * x + 4 * y) →
  3 * x + 4 * y = 5 ∧ x + 4 * y = 3 := by sorry

end min_value_theorem_l3464_346473


namespace segment_existence_l3464_346455

theorem segment_existence (pencil_length eraser_length : ℝ) 
  (h_pencil : pencil_length > 0) (h_eraser : eraser_length > 0) : 
  ∃ (x y : ℝ), x > 0 ∧ y > 0 ∧ x + y = pencil_length ∧ Real.sqrt (x * y) = eraser_length :=
sorry

end segment_existence_l3464_346455


namespace triangle_intersection_area_l3464_346497

/-- Given a triangle PQR with vertices P(0, 10), Q(3, 0), R(9, 0),
    and a horizontal line y=s intersecting PQ at V and PR at W,
    if the area of triangle PVW is 18, then s = 10 - 2√15. -/
theorem triangle_intersection_area (s : ℝ) : 
  let P : ℝ × ℝ := (0, 10)
  let Q : ℝ × ℝ := (3, 0)
  let R : ℝ × ℝ := (9, 0)
  let V : ℝ × ℝ := ((3/10) * (10 - s), s)
  let W : ℝ × ℝ := ((9/10) * (10 - s), s)
  let area_PVW : ℝ := (1/2) * ((W.1 - V.1) * (P.2 - V.2))
  area_PVW = 18 → s = 10 - 2 * Real.sqrt 15 :=
by sorry

end triangle_intersection_area_l3464_346497


namespace differential_equation_solution_l3464_346490

open Real

theorem differential_equation_solution (x : ℝ) (C : ℝ) :
  let y : ℝ → ℝ := λ x => cos x * (sin x + C)
  (deriv y) x + y x * tan x = cos x ^ 2 := by
  sorry

end differential_equation_solution_l3464_346490


namespace expected_sum_of_two_marbles_l3464_346434

def marbleSet : Finset ℕ := Finset.range 6

def marblePairs : Finset (ℕ × ℕ) :=
  (marbleSet.product marbleSet).filter (fun p => p.1 < p.2)

def pairSum (p : ℕ × ℕ) : ℕ := p.1 + p.2 + 2

theorem expected_sum_of_two_marbles :
  (marblePairs.sum pairSum) / marblePairs.card = 7 := by
  sorry

end expected_sum_of_two_marbles_l3464_346434


namespace total_soccer_balls_donated_l3464_346436

-- Define the given conditions
def soccer_balls_per_class : ℕ := 5
def number_of_schools : ℕ := 2
def elementary_classes_per_school : ℕ := 4
def middle_classes_per_school : ℕ := 5

-- Define the theorem
theorem total_soccer_balls_donated : 
  soccer_balls_per_class * number_of_schools * (elementary_classes_per_school + middle_classes_per_school) = 90 := by
  sorry


end total_soccer_balls_donated_l3464_346436


namespace no_solution_equation_l3464_346452

theorem no_solution_equation : ∀ x : ℝ, 
  4 * x * (10 * x - (-10 - (3 * x - 8 * (x + 1)))) + 
  5 * (12 - (4 * (x + 1) - 3 * x)) ≠ 
  18 * x^2 - (6 * x^2 - (7 * x + 4 * (2 * x^2 - x + 11))) := by
  sorry

end no_solution_equation_l3464_346452


namespace composite_sum_l3464_346483

theorem composite_sum (a b c d m n : ℕ) 
  (ha : a > b) (hb : b > c) (hc : c > d) 
  (hdiv : (a + b - c + d) ∣ (a * c + b * d))
  (hm : m > 0) (hn : Odd n) : 
  ∃ k > 1, k ∣ (a^n * b^m + c^m * d^n) :=
sorry

end composite_sum_l3464_346483


namespace cubic_roots_sum_of_cubes_l3464_346422

theorem cubic_roots_sum_of_cubes (a b c : ℂ) : 
  (a^3 - 2*a^2 + 3*a - 5 = 0) → 
  (b^3 - 2*b^2 + 3*b - 5 = 0) → 
  (c^3 - 2*c^2 + 3*c - 5 = 0) → 
  a^3 + b^3 + c^3 = 5 := by sorry

end cubic_roots_sum_of_cubes_l3464_346422


namespace kindergarten_total_l3464_346426

/-- Represents the number of children in a kindergarten with different pet ownership patterns -/
structure KindergartenPets where
  dogs_only : ℕ
  both : ℕ
  cats_total : ℕ

/-- Calculates the total number of children in the kindergarten -/
def total_children (k : KindergartenPets) : ℕ :=
  k.dogs_only + k.both + (k.cats_total - k.both)

/-- Theorem stating the total number of children in the kindergarten -/
theorem kindergarten_total (k : KindergartenPets) 
  (h1 : k.dogs_only = 18)
  (h2 : k.both = 6)
  (h3 : k.cats_total = 12) :
  total_children k = 30 := by
  sorry

#check kindergarten_total

end kindergarten_total_l3464_346426


namespace josh_pencils_left_josh_pencils_left_proof_l3464_346410

/-- Given that Josh initially had 142 pencils and gave away 31 pencils,
    prove that he has 111 pencils left. -/
theorem josh_pencils_left : ℕ → ℕ → ℕ → Prop :=
  fun initial_pencils pencils_given_away pencils_left =>
    initial_pencils = 142 →
    pencils_given_away = 31 →
    pencils_left = initial_pencils - pencils_given_away →
    pencils_left = 111

/-- Proof of the theorem -/
theorem josh_pencils_left_proof : josh_pencils_left 142 31 111 := by
  sorry

end josh_pencils_left_josh_pencils_left_proof_l3464_346410


namespace perpendicular_to_parallel_planes_l3464_346491

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the perpendicular and parallel relations
variable (perp : Line → Plane → Prop)
variable (para : Plane → Plane → Prop)

-- State the theorem
theorem perpendicular_to_parallel_planes 
  (α β : Plane) (l : Line) 
  (h1 : perp l α) (h2 : para α β) : 
  perp l β := by
  sorry

end perpendicular_to_parallel_planes_l3464_346491


namespace inequality_proof_l3464_346470

theorem inequality_proof (a b c : ℝ) 
  (h_pos_a : a > 0) (h_pos_b : b > 0) (h_pos_c : c > 0)
  (h_eq : a^2 + b^2 + 4*c^2 = 3) : 
  (a + b + 2*c ≤ 3) ∧ (b = 2*c → 1/a + 1/c ≥ 3) := by
  sorry

end inequality_proof_l3464_346470


namespace custom_operation_results_l3464_346467

-- Define the custom operation
def customOp (a b : ℤ) : ℤ := a^2 - (a + b) + a*b

-- State the theorem
theorem custom_operation_results :
  (customOp 2 (-3) = -1) ∧ (customOp 4 (customOp 2 (-3)) = 7) := by
  sorry

end custom_operation_results_l3464_346467


namespace no_infinite_sequence_sqrt_difference_l3464_346446

theorem no_infinite_sequence_sqrt_difference :
  ¬ (∃ (x : ℕ → ℝ), (∀ n, 0 < x n) ∧ 
    (∀ n, x (n + 2) = Real.sqrt (x (n + 1)) - Real.sqrt (x n))) := by
  sorry

end no_infinite_sequence_sqrt_difference_l3464_346446


namespace coin_flip_probability_l3464_346409

theorem coin_flip_probability (p : ℝ) (n : ℕ) (h_p : p = 1 / 2) (h_n : n = 5) :
  p ^ 4 * (1 - p) = 1 / 32 := by
  sorry

end coin_flip_probability_l3464_346409


namespace regular_18gon_relation_l3464_346495

/-- A regular 18-gon inscribed in a circle -/
structure Regular18Gon where
  /-- The radius of the circumscribed circle -/
  r : ℝ
  /-- The side length of the 18-gon -/
  a : ℝ
  /-- The radius is positive -/
  r_pos : 0 < r

/-- Theorem: For a regular 18-gon inscribed in a circle, a^3 + r^3 = 3ar^2 -/
theorem regular_18gon_relation (polygon : Regular18Gon) : 
  polygon.a^3 + polygon.r^3 = 3 * polygon.a * polygon.r^2 := by
  sorry

end regular_18gon_relation_l3464_346495


namespace fools_gold_ounces_l3464_346423

def earnings_per_ounce : ℝ := 9
def fine : ℝ := 50
def remaining_money : ℝ := 22

theorem fools_gold_ounces :
  ∃ (x : ℝ), x * earnings_per_ounce - fine = remaining_money ∧ x = 8 := by
  sorry

end fools_gold_ounces_l3464_346423


namespace rahim_pillows_l3464_346474

/-- The number of pillows Rahim bought initially -/
def initial_pillows : ℕ := 4

/-- The initial average cost of pillows -/
def initial_avg_cost : ℚ := 5

/-- The price of the fifth pillow -/
def fifth_pillow_price : ℚ := 10

/-- The new average price of 5 pillows -/
def new_avg_price : ℚ := 6

/-- Proof that the number of pillows Rahim bought initially is 4 -/
theorem rahim_pillows :
  (initial_avg_cost * initial_pillows + fifth_pillow_price) / (initial_pillows + 1) = new_avg_price :=
by sorry

end rahim_pillows_l3464_346474


namespace repeating_decimal_equals_fraction_l3464_346430

-- Define the repeating decimal 0.4̅36̅
def repeating_decimal : ℚ := 0.4 + (36 / 990)

-- Theorem statement
theorem repeating_decimal_equals_fraction : repeating_decimal = 24 / 55 := by
  sorry

end repeating_decimal_equals_fraction_l3464_346430


namespace candidate_vote_difference_l3464_346408

theorem candidate_vote_difference (total_votes : ℕ) (candidate_percentage : ℚ) : 
  total_votes = 8000 → 
  candidate_percentage = 35 / 100 → 
  (total_votes : ℚ) * candidate_percentage + 
  (total_votes : ℚ) * (1 - candidate_percentage) = total_votes → 
  (total_votes : ℚ) * (1 - candidate_percentage) - 
  (total_votes : ℚ) * candidate_percentage = 2400 := by
sorry

end candidate_vote_difference_l3464_346408


namespace smallest_y_for_perfect_cube_l3464_346428

/-- Given x = 4 * 21 * 63, the smallest positive integer y such that xy is a perfect cube is 14 -/
theorem smallest_y_for_perfect_cube (x : ℕ) (hx : x = 4 * 21 * 63) :
  ∃ y : ℕ, y > 0 ∧ 
    (∃ z : ℕ, x * y = z^3) ∧
    (∀ w : ℕ, w > 0 ∧ w < y → ¬∃ z : ℕ, x * w = z^3) ∧
    y = 14 := by
  sorry

end smallest_y_for_perfect_cube_l3464_346428


namespace fraction_between_one_quarter_between_one_seventh_and_one_fourth_l3464_346486

theorem fraction_between (a b : ℚ) (t : ℚ) (h : 0 ≤ t ∧ t ≤ 1) :
  a + t * (b - a) = (1 - t) * a + t * b :=
by sorry

theorem one_quarter_between_one_seventh_and_one_fourth :
  (1 : ℚ)/7 + (1/4) * ((1/4) - (1/7)) = 23/112 :=
by sorry

end fraction_between_one_quarter_between_one_seventh_and_one_fourth_l3464_346486


namespace dinner_bill_ratio_l3464_346453

/-- Given a dinner bill split between three people, this theorem proves
    the ratio of two people's payments given certain conditions. -/
theorem dinner_bill_ratio (total bill : ℚ) (daniel clarence matthew : ℚ) :
  bill = 20.20 →
  daniel = 6.06 →
  daniel = (1 / 2) * clarence →
  bill = daniel + clarence + matthew →
  clarence / matthew = 6 / 1 := by
  sorry

end dinner_bill_ratio_l3464_346453


namespace match_processes_count_l3464_346419

def number_of_match_processes : ℕ := 2 * Nat.choose 13 6

theorem match_processes_count :
  number_of_match_processes = 3432 :=
by sorry

end match_processes_count_l3464_346419


namespace ferris_wheel_seat_count_l3464_346494

/-- The number of seats on a Ferris wheel -/
def ferris_wheel_seats (total_people : ℕ) (people_per_seat : ℕ) : ℕ :=
  (total_people + people_per_seat - 1) / people_per_seat

/-- Theorem: The Ferris wheel has 3 seats -/
theorem ferris_wheel_seat_count : ferris_wheel_seats 8 3 = 3 := by
  sorry

end ferris_wheel_seat_count_l3464_346494


namespace typing_time_proof_l3464_346475

/-- Calculates the time in hours required to type a research paper given the typing speed and number of words. -/
def time_to_type (typing_speed : ℕ) (total_words : ℕ) : ℚ :=
  (total_words : ℚ) / (typing_speed : ℚ) / 60

/-- Proves that given a typing speed of 38 words per minute and a research paper with 4560 words, the time required to type the paper is 2 hours. -/
theorem typing_time_proof :
  time_to_type 38 4560 = 2 := by
  sorry

end typing_time_proof_l3464_346475


namespace optimal_price_l3464_346466

/-- Represents the daily sales volume as a function of price -/
def sales (x : ℝ) : ℝ := 400 - 20 * x

/-- Represents the daily profit as a function of price -/
def profit (x : ℝ) : ℝ := (x - 8) * sales x

theorem optimal_price :
  ∃ (x : ℝ), 8 ≤ x ∧ x ≤ 15 ∧ profit x = 640 :=
by sorry

end optimal_price_l3464_346466


namespace average_difference_l3464_346403

theorem average_difference (a b c : ℝ) 
  (h1 : (a + b) / 2 = 115) 
  (h2 : (b + c) / 2 = 160) : 
  a - c = -90 := by
sorry

end average_difference_l3464_346403


namespace largest_prime_for_prime_check_l3464_346400

theorem largest_prime_for_prime_check : ∀ n : ℕ, 1000 ≤ n ∧ n ≤ 1050 →
  ∀ p : ℕ, Prime p ∧ p ≤ Real.sqrt n → p ≤ 31 := by
  sorry

end largest_prime_for_prime_check_l3464_346400


namespace expression_arrangements_l3464_346460

/-- Given three distinct real numbers, there are 96 possible ways to arrange
    the eight expressions ±x ±y ±z in increasing order. -/
theorem expression_arrangements (x y z : ℝ) (hxy : x ≠ y) (hyz : y ≠ z) (hxz : x ≠ z) :
  (Set.ncard {l : List ℝ | 
    l.length = 8 ∧ 
    l.Nodup ∧
    (∀ a ∈ l, ∃ (s₁ s₂ s₃ : Bool), a = (if s₁ then x else -x) + (if s₂ then y else -y) + (if s₃ then z else -z)) ∧
    l.Sorted (· < ·)}) = 96 :=
by sorry

end expression_arrangements_l3464_346460


namespace factorization_of_75x_plus_45_l3464_346498

theorem factorization_of_75x_plus_45 (x : ℝ) : 75 * x + 45 = 15 * (5 * x + 3) := by
  sorry

end factorization_of_75x_plus_45_l3464_346498


namespace cookie_bringers_l3464_346469

theorem cookie_bringers (num_brownie_students : ℕ) (brownies_per_student : ℕ)
                        (num_donut_students : ℕ) (donuts_per_student : ℕ)
                        (cookies_per_student : ℕ) (price_per_item : ℚ)
                        (total_raised : ℚ) :
  num_brownie_students = 30 →
  brownies_per_student = 12 →
  num_donut_students = 15 →
  donuts_per_student = 12 →
  cookies_per_student = 24 →
  price_per_item = 2 →
  total_raised = 2040 →
  ∃ (num_cookie_students : ℕ),
    num_cookie_students = 20 ∧
    total_raised = price_per_item * (num_brownie_students * brownies_per_student +
                                     num_cookie_students * cookies_per_student +
                                     num_donut_students * donuts_per_student) :=
by sorry

end cookie_bringers_l3464_346469


namespace number_puzzle_l3464_346477

theorem number_puzzle : ∃ x : ℝ, 3 * (2 * x + 9) = 81 := by
  sorry

end number_puzzle_l3464_346477


namespace company_employees_ratio_salary_increase_impact_l3464_346402

theorem company_employees_ratio (M F N : ℕ) : 
  (M : ℚ) / F = 7 / 8 ∧ 
  (N : ℚ) / F = 6 / 8 ∧ 
  ((M + 5 : ℚ) / F = 8 / 9 ∧ (N + 3 : ℚ) / F = 7 / 9) → 
  M = 315 ∧ F = 360 ∧ N = 270 := by
  sorry

theorem salary_increase_impact (T : ℚ) :
  T > 0 → T * (110 / 100) - T = T / 10 := by
  sorry

end company_employees_ratio_salary_increase_impact_l3464_346402


namespace fifth_power_sum_l3464_346425

theorem fifth_power_sum (a b x y : ℝ) 
  (eq1 : a * x + b * y = 3)
  (eq2 : a * x^2 + b * y^2 = 7)
  (eq3 : a * x^3 + b * y^3 = 16)
  (eq4 : a * x^4 + b * y^4 = 42) :
  a * x^5 + b * y^5 = 99 := by
sorry

end fifth_power_sum_l3464_346425


namespace sequence_problem_l3464_346412

def arithmetic_sequence (a b c d : ℝ) : Prop :=
  (b - a = c - b) ∧ (c - b = d - c)

def geometric_sequence (a b c d e : ℝ) : Prop :=
  (b / a = c / b) ∧ (c / b = d / c) ∧ (d / c = e / d)

theorem sequence_problem (a₁ a₂ b₁ b₂ b₃ : ℝ) :
  arithmetic_sequence (-7) a₁ a₂ (-1) →
  geometric_sequence (-4) b₁ b₂ b₃ (-1) →
  (a₂ - a₁) / b₂ = -1 := by
  sorry

end sequence_problem_l3464_346412


namespace batsman_average_l3464_346448

theorem batsman_average (x : ℕ) : 
  (40 * x + 30 * 10) / (x + 10) = 35 → x = 10 := by
  sorry

end batsman_average_l3464_346448


namespace novel_pages_count_l3464_346443

theorem novel_pages_count : 
  ∀ (vacation_days : ℕ) 
    (first_two_days_avg : ℕ) 
    (next_three_days_avg : ℕ) 
    (last_day_pages : ℕ),
  vacation_days = 6 →
  first_two_days_avg = 42 →
  next_three_days_avg = 35 →
  last_day_pages = 15 →
  (2 * first_two_days_avg + 3 * next_three_days_avg + last_day_pages) = 204 := by
sorry

end novel_pages_count_l3464_346443


namespace factorization_equality_l3464_346471

theorem factorization_equality (x y : ℝ) :
  (1 - x^2) * (1 - y^2) - 4*x*y = (x*y - 1 + x + y) * (x*y - 1 - x - y) := by
  sorry

end factorization_equality_l3464_346471


namespace asparagus_cost_l3464_346438

def initial_amount : ℕ := 55
def banana_pack_cost : ℕ := 4
def banana_packs : ℕ := 2
def pear_cost : ℕ := 2
def chicken_cost : ℕ := 11
def remaining_amount : ℕ := 28

theorem asparagus_cost :
  ∃ (asparagus_cost : ℕ),
    initial_amount - (banana_pack_cost * banana_packs + pear_cost + chicken_cost + asparagus_cost) = remaining_amount ∧
    asparagus_cost = 6 := by
  sorry

end asparagus_cost_l3464_346438
