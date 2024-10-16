import Mathlib

namespace NUMINAMATH_CALUDE_amanda_summer_work_hours_l704_70456

/-- Calculates the required weekly work hours for Amanda during summer -/
theorem amanda_summer_work_hours 
  (winter_weekly_hours : ℝ) 
  (winter_weeks : ℝ) 
  (winter_earnings : ℝ) 
  (summer_weeks : ℝ) 
  (summer_earnings : ℝ) 
  (h1 : winter_weekly_hours = 45) 
  (h2 : winter_weeks = 8) 
  (h3 : winter_earnings = 3600) 
  (h4 : summer_weeks = 20) 
  (h5 : summer_earnings = 4500) :
  (summer_earnings / (winter_earnings / (winter_weekly_hours * winter_weeks))) / summer_weeks = 22.5 := by
  sorry

#check amanda_summer_work_hours

end NUMINAMATH_CALUDE_amanda_summer_work_hours_l704_70456


namespace NUMINAMATH_CALUDE_problem_pyramid_volume_l704_70470

/-- Triangular pyramid with given side lengths -/
structure TriangularPyramid where
  base_side : ℝ
  pa : ℝ
  pb : ℝ
  pc : ℝ

/-- Volume of a triangular pyramid -/
noncomputable def volume (p : TriangularPyramid) : ℝ :=
  sorry

/-- The specific triangular pyramid from the problem -/
def problem_pyramid : TriangularPyramid :=
  { base_side := 3
  , pa := 3
  , pb := 4
  , pc := 5 }

/-- Theorem stating that the volume of the problem pyramid is √11 -/
theorem problem_pyramid_volume :
  volume problem_pyramid = Real.sqrt 11 := by
  sorry

end NUMINAMATH_CALUDE_problem_pyramid_volume_l704_70470


namespace NUMINAMATH_CALUDE_sum_is_34_l704_70498

/-- Represents a 4x4 grid filled with integers from 1 to 16 -/
def Grid : Type := Fin 4 → Fin 4 → Fin 16

/-- Fills the grid sequentially from 1 to 16 -/
def fillGrid : Grid :=
  fun i j => ⟨i.val * 4 + j.val + 1, by sorry⟩

/-- Represents a selection of 4 positions in the grid, each from a different row and column -/
structure Selection :=
  (pos : Fin 4 → Fin 4 × Fin 4)
  (different_rows : ∀ i j, i ≠ j → (pos i).1 ≠ (pos j).1)
  (different_cols : ∀ i j, i ≠ j → (pos i).2 ≠ (pos j).2)

/-- The main theorem to be proved -/
theorem sum_is_34 (s : Selection) : 
  (Finset.univ.sum fun i => (fillGrid (s.pos i).1 (s.pos i).2).val) = 34 := by
  sorry

end NUMINAMATH_CALUDE_sum_is_34_l704_70498


namespace NUMINAMATH_CALUDE_lcm_of_9_and_14_l704_70414

theorem lcm_of_9_and_14 : Nat.lcm 9 14 = 126 := by
  sorry

end NUMINAMATH_CALUDE_lcm_of_9_and_14_l704_70414


namespace NUMINAMATH_CALUDE_invalid_statement_d_l704_70406

/-- Represents a mathematical statement --/
structure MathStatement where
  content : String

/-- Represents a mathematical proof --/
structure Proof where
  premises : List MathStatement
  conclusion : MathStatement

/-- Checks if a statement is true --/
def isTrue (s : MathStatement) : Bool :=
  sorry

/-- Checks if a proof is valid --/
def isValidProof (p : Proof) : Bool :=
  sorry

/-- Checks if a statement is used in deriving the conclusion of a proof --/
def isUsedInConclusion (s : MathStatement) (p : Proof) : Bool :=
  sorry

theorem invalid_statement_d :
  ¬∀ (p : Proof) (s : MathStatement),
    (s ∈ p.premises ∧ ¬isTrue s ∧ ¬isUsedInConclusion s p) →
    (isValidProof p → isTrue p.conclusion) :=
  sorry

end NUMINAMATH_CALUDE_invalid_statement_d_l704_70406


namespace NUMINAMATH_CALUDE_complex_equation_solution_l704_70490

theorem complex_equation_solution (a : ℝ) :
  (2 + a * Complex.I) / (1 + Complex.I) = (3 : ℂ) + Complex.I → a = 4 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l704_70490


namespace NUMINAMATH_CALUDE_rice_consumption_l704_70441

theorem rice_consumption (initial_rice : ℕ) (daily_consumption : ℕ) (days : ℕ) 
  (h1 : initial_rice = 52)
  (h2 : daily_consumption = 9)
  (h3 : days = 3) :
  initial_rice - (daily_consumption * days) = 25 := by
  sorry

end NUMINAMATH_CALUDE_rice_consumption_l704_70441


namespace NUMINAMATH_CALUDE_greatest_integer_x_l704_70405

theorem greatest_integer_x (x : ℕ) : x^4 / x^2 < 18 → x ≤ 4 :=
sorry

end NUMINAMATH_CALUDE_greatest_integer_x_l704_70405


namespace NUMINAMATH_CALUDE_polynomial_simplification_l704_70433

theorem polynomial_simplification (x : ℝ) :
  (5 * x^10 + 8 * x^9 + 3 * x^8) + (2 * x^12 + 3 * x^10 + x^9 + 4 * x^8 + 6 * x^4 + 7 * x^2 + 9) =
  2 * x^12 + 8 * x^10 + 9 * x^9 + 7 * x^8 + 6 * x^4 + 7 * x^2 + 9 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_simplification_l704_70433


namespace NUMINAMATH_CALUDE_pedoes_inequality_pedoes_inequality_equality_condition_l704_70460

/-- Pedoe's inequality for triangles -/
theorem pedoes_inequality (a b c a₁ b₁ c₁ Δ Δ₁ : ℝ) 
  (h_abc : 0 < a ∧ 0 < b ∧ 0 < c)
  (h_a₁b₁c₁ : 0 < a₁ ∧ 0 < b₁ ∧ 0 < c₁)
  (h_Δ : 0 < Δ)
  (h_Δ₁ : 0 < Δ₁)
  (h_abc_triangle : a + b > c ∧ b + c > a ∧ c + a > b)
  (h_a₁b₁c₁_triangle : a₁ + b₁ > c₁ ∧ b₁ + c₁ > a₁ ∧ c₁ + a₁ > b₁)
  (h_Δ_def : Δ = Real.sqrt ((a + b + c) * (b + c - a) * (c + a - b) * (a + b - c)) / 4)
  (h_Δ₁_def : Δ₁ = Real.sqrt ((a₁ + b₁ + c₁) * (b₁ + c₁ - a₁) * (c₁ + a₁ - b₁) * (a₁ + b₁ - c₁)) / 4) :
  a^2 * (b₁^2 + c₁^2 - a₁^2) + b^2 * (c₁^2 + a₁^2 - b₁^2) + c^2 * (a₁^2 + b₁^2 - c₁^2) ≥ 16 * Δ * Δ₁ :=
by sorry

/-- Condition for equality in Pedoe's inequality -/
theorem pedoes_inequality_equality_condition (a b c a₁ b₁ c₁ Δ Δ₁ : ℝ) 
  (h_abc : 0 < a ∧ 0 < b ∧ 0 < c)
  (h_a₁b₁c₁ : 0 < a₁ ∧ 0 < b₁ ∧ 0 < c₁)
  (h_Δ : 0 < Δ)
  (h_Δ₁ : 0 < Δ₁)
  (h_abc_triangle : a + b > c ∧ b + c > a ∧ c + a > b)
  (h_a₁b₁c₁_triangle : a₁ + b₁ > c₁ ∧ b₁ + c₁ > a₁ ∧ c₁ + a₁ > b₁)
  (h_Δ_def : Δ = Real.sqrt ((a + b + c) * (b + c - a) * (c + a - b) * (a + b - c)) / 4)
  (h_Δ₁_def : Δ₁ = Real.sqrt ((a₁ + b₁ + c₁) * (b₁ + c₁ - a₁) * (c₁ + a₁ - b₁) * (a₁ + b₁ - c₁)) / 4) :
  (a^2 * (b₁^2 + c₁^2 - a₁^2) + b^2 * (c₁^2 + a₁^2 - b₁^2) + c^2 * (a₁^2 + b₁^2 - c₁^2) = 16 * Δ * Δ₁) ↔
  (∃ (k : ℝ), k > 0 ∧ a = k * a₁ ∧ b = k * b₁ ∧ c = k * c₁) :=
by sorry

end NUMINAMATH_CALUDE_pedoes_inequality_pedoes_inequality_equality_condition_l704_70460


namespace NUMINAMATH_CALUDE_total_cost_is_1100_l704_70445

def piano_cost : ℝ := 500
def num_lessons : ℕ := 20
def lesson_cost : ℝ := 40
def discount_rate : ℝ := 0.25

def total_cost : ℝ := 
  piano_cost + (1 - discount_rate) * (num_lessons : ℝ) * lesson_cost

theorem total_cost_is_1100 : total_cost = 1100 := by sorry

end NUMINAMATH_CALUDE_total_cost_is_1100_l704_70445


namespace NUMINAMATH_CALUDE_trigonometric_equation_solution_l704_70491

open Real

theorem trigonometric_equation_solution (x : ℝ) :
  sin x + sin (2*x) + sin (3*x) = 1 + cos x + cos (2*x) ↔
  (∃ k : ℤ, x = π/2 + k * π) ∨
  (∃ k : ℤ, x = 2*π/3 + k * 2*π) ∨
  (∃ k : ℤ, x = 4*π/3 + k * 2*π) ∨
  (∃ k : ℤ, x = π/6 + k * 2*π) ∨
  (∃ k : ℤ, x = 5*π/6 + k * 2*π) :=
by sorry

end NUMINAMATH_CALUDE_trigonometric_equation_solution_l704_70491


namespace NUMINAMATH_CALUDE_factorization_a_squared_minus_six_l704_70430

theorem factorization_a_squared_minus_six (a : ℝ) :
  a^2 - 6 = (a + Real.sqrt 6) * (a - Real.sqrt 6) := by
  sorry

end NUMINAMATH_CALUDE_factorization_a_squared_minus_six_l704_70430


namespace NUMINAMATH_CALUDE_increasing_f_range_of_a_l704_70489

-- Define the piecewise function f(x)
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 1 then (6 - a) * x - 4 * a
  else Real.log x / Real.log a

-- Theorem statement
theorem increasing_f_range_of_a :
  ∀ a : ℝ, (∀ x y : ℝ, x < y → f a x < f a y) →
  a ∈ Set.Icc (6/5) 6 ∧ a ≠ 6 :=
sorry

end NUMINAMATH_CALUDE_increasing_f_range_of_a_l704_70489


namespace NUMINAMATH_CALUDE_prob_sum_three_eq_one_over_216_l704_70422

/-- The probability of rolling a specific number on a standard die -/
def prob_single_die : ℚ := 1 / 6

/-- The number of dice rolled -/
def num_dice : ℕ := 3

/-- The target sum we're looking for -/
def target_sum : ℕ := 3

/-- The probability of rolling a sum of 3 with three standard dice -/
def prob_sum_three : ℚ := (prob_single_die) ^ num_dice

theorem prob_sum_three_eq_one_over_216 : 
  prob_sum_three = 1 / 216 := by sorry

end NUMINAMATH_CALUDE_prob_sum_three_eq_one_over_216_l704_70422


namespace NUMINAMATH_CALUDE_even_function_implies_a_equals_four_l704_70426

/-- A function f is even if f(-x) = f(x) for all x -/
def IsEven (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

/-- The function f(x) = (x+a)(x-4) -/
def f (a : ℝ) (x : ℝ) : ℝ := (x + a) * (x - 4)

/-- If f(x) = (x+a)(x-4) is an even function, then a = 4 -/
theorem even_function_implies_a_equals_four (a : ℝ) :
  IsEven (f a) → a = 4 := by
  sorry

end NUMINAMATH_CALUDE_even_function_implies_a_equals_four_l704_70426


namespace NUMINAMATH_CALUDE_granola_discounted_price_l704_70449

/-- Calculates the discounted price per bag of granola given the following conditions:
    - Cost of ingredients per bag
    - Total number of bags made
    - Original selling price per bag
    - Number of bags sold at original price
    - Total net profit -/
def discounted_price (cost_per_bag : ℚ) (total_bags : ℕ) (original_price : ℚ)
                     (bags_sold_full_price : ℕ) (net_profit : ℚ) : ℚ :=
  let total_cost := cost_per_bag * total_bags
  let full_price_revenue := original_price * bags_sold_full_price
  let total_revenue := net_profit + total_cost
  let discounted_revenue := total_revenue - full_price_revenue
  let discounted_bags := total_bags - bags_sold_full_price
  discounted_revenue / discounted_bags

theorem granola_discounted_price :
  discounted_price 3 20 6 15 50 = 4 := by
  sorry

end NUMINAMATH_CALUDE_granola_discounted_price_l704_70449


namespace NUMINAMATH_CALUDE_largest_hexagon_angle_l704_70457

/-- Represents the angles of a hexagon -/
structure HexagonAngles where
  a₁ : ℝ
  a₂ : ℝ
  a₃ : ℝ
  a₄ : ℝ
  a₅ : ℝ
  a₆ : ℝ

/-- The sum of angles in a hexagon is 720 degrees -/
axiom hexagon_angle_sum (h : HexagonAngles) : h.a₁ + h.a₂ + h.a₃ + h.a₄ + h.a₅ + h.a₆ = 720

/-- The angles of the hexagon are in the ratio 3:3:3:4:5:6 -/
def hexagon_angle_ratio (h : HexagonAngles) : Prop :=
  ∃ x : ℝ, h.a₁ = 3*x ∧ h.a₂ = 3*x ∧ h.a₃ = 3*x ∧ h.a₄ = 4*x ∧ h.a₅ = 5*x ∧ h.a₆ = 6*x

/-- The largest angle in the hexagon is 180 degrees -/
theorem largest_hexagon_angle (h : HexagonAngles) 
  (ratio : hexagon_angle_ratio h) : h.a₆ = 180 := by
  sorry

end NUMINAMATH_CALUDE_largest_hexagon_angle_l704_70457


namespace NUMINAMATH_CALUDE_concert_ticket_sales_l704_70494

theorem concert_ticket_sales
  (student_price : ℕ)
  (non_student_price : ℕ)
  (total_revenue : ℕ)
  (student_tickets : ℕ)
  (h1 : student_price = 9)
  (h2 : non_student_price = 11)
  (h3 : total_revenue = 20960)
  (h4 : student_tickets = 520) :
  ∃ (non_student_tickets : ℕ),
    student_tickets * student_price + non_student_tickets * non_student_price = total_revenue ∧
    student_tickets + non_student_tickets = 2000 :=
by
  sorry

#check concert_ticket_sales

end NUMINAMATH_CALUDE_concert_ticket_sales_l704_70494


namespace NUMINAMATH_CALUDE_intersection_property_l704_70450

-- Define the circle
def Circle (a : ℝ) := {p : ℝ × ℝ | p.1^2 + p.2^2 = a}

-- Define the line
def Line := {p : ℝ × ℝ | p.1 + p.2 = 1}

-- Define the origin
def O : ℝ × ℝ := (0, 0)

theorem intersection_property (a : ℝ) 
  (A B : ℝ × ℝ) 
  (hA : A ∈ Circle a ∩ Line) 
  (hB : B ∈ Circle a ∩ Line) 
  (C : ℝ × ℝ) 
  (hC : C ∈ Circle a) 
  (h_vec : (A.1 - O.1, A.2 - O.2) + (B.1 - O.1, B.2 - O.2) = (C.1 - O.1, C.2 - O.2)) :
  a = 2 :=
sorry

end NUMINAMATH_CALUDE_intersection_property_l704_70450


namespace NUMINAMATH_CALUDE_proper_subset_condition_l704_70455

def M : Set ℝ := {x : ℝ | 2 * x^2 - 3 * x - 2 = 0}

def N (a : ℝ) : Set ℝ := {x : ℝ | a * x = 1}

theorem proper_subset_condition (a : ℝ) :
  N a ⊂ M → a = 0 ∨ a = -2 ∨ a = 1/2 := by sorry

end NUMINAMATH_CALUDE_proper_subset_condition_l704_70455


namespace NUMINAMATH_CALUDE_common_tangents_exist_l704_70454

/-- Represents a circle in a 2D plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Represents a line in a 2D plane -/
structure Line where
  point1 : ℝ × ℝ
  point2 : ℝ × ℝ

/-- Checks if a line is tangent to a circle -/
def isTangent (l : Line) (c : Circle) : Prop := sorry

/-- Theorem: Common tangents exist for two given circles -/
theorem common_tangents_exist (c1 c2 : Circle) 
  (h : c1.radius > c2.radius) : 
  ∃ (l : Line), isTangent l c1 ∧ isTangent l c2 := by
  sorry

/-- Function to construct common tangents -/
noncomputable def construct_common_tangents (c1 c2 : Circle) 
  (h : c1.radius > c2.radius) : 
  List Line := sorry

end NUMINAMATH_CALUDE_common_tangents_exist_l704_70454


namespace NUMINAMATH_CALUDE_regular_pentagon_diagonal_intersection_angle_l704_70464

/-- A regular pentagon is a polygon with 5 equal sides and 5 equal angles. -/
structure RegularPentagon where
  vertices : Fin 5 → ℝ × ℝ
  is_regular : sorry

/-- The diagonals of a pentagon are line segments connecting non-adjacent vertices. -/
def diagonal (p : RegularPentagon) (i j : Fin 5) : sorry := sorry

/-- The intersection point of two diagonals in a pentagon. -/
def intersectionPoint (p : RegularPentagon) (d1 d2 : sorry) : ℝ × ℝ := sorry

/-- The angle between two line segments at their intersection point. -/
def angleBetween (p1 p2 p3 : ℝ × ℝ) : ℝ := sorry

theorem regular_pentagon_diagonal_intersection_angle (p : RegularPentagon) :
  let s := intersectionPoint p (diagonal p 0 2) (diagonal p 1 3)
  angleBetween (p.vertices 2) s (p.vertices 3) = 72 := by sorry

end NUMINAMATH_CALUDE_regular_pentagon_diagonal_intersection_angle_l704_70464


namespace NUMINAMATH_CALUDE_simplify_expression_l704_70458

theorem simplify_expression (a b : ℝ) : a - 4*(2*a - b) - 2*(a + 2*b) = -9*a := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l704_70458


namespace NUMINAMATH_CALUDE_factorization_proof_l704_70492

/-- Prove the factorization of two polynomial expressions -/
theorem factorization_proof (x y : ℝ) : 
  (2 * x^2 * y - 4 * x * y + 2 * y = 2 * y * (x - 1)^2) ∧ 
  (x^4 - 9 * x^2 = x^2 * (x + 3) * (x - 3)) := by
  sorry

end NUMINAMATH_CALUDE_factorization_proof_l704_70492


namespace NUMINAMATH_CALUDE_complex_arithmetic_equation_l704_70493

theorem complex_arithmetic_equation : 
  ((4501 * 2350) - (7125 / 9)) + (3250 ^ 2) * 4167 = 44045164058.33 := by
  sorry

end NUMINAMATH_CALUDE_complex_arithmetic_equation_l704_70493


namespace NUMINAMATH_CALUDE_range_of_a_l704_70459

theorem range_of_a (x : ℝ) (h1 : x > 0) (h2 : 2^x * (x - a) < 1) : a > -1 :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l704_70459


namespace NUMINAMATH_CALUDE_weight_of_a_l704_70471

theorem weight_of_a (a b c d e : ℝ) : 
  (a + b + c) / 3 = 84 →
  (a + b + c + d) / 4 = 80 →
  e = d + 3 →
  (b + c + d + e) / 4 = 79 →
  a = 75 := by
sorry

end NUMINAMATH_CALUDE_weight_of_a_l704_70471


namespace NUMINAMATH_CALUDE_disco_vote_participants_l704_70475

theorem disco_vote_participants :
  ∀ (initial_voters : ℕ) 
    (initial_oct22_percent initial_oct29_percent : ℚ)
    (additional_voters : ℕ)
    (final_oct29_percent : ℚ),
  initial_oct22_percent + initial_oct29_percent = 1 →
  initial_oct22_percent = 35 / 100 →
  initial_oct29_percent = 65 / 100 →
  additional_voters = 80 →
  final_oct29_percent = 45 / 100 →
  initial_oct29_percent * initial_voters = 
    final_oct29_percent * (initial_voters + additional_voters) →
  initial_voters + additional_voters = 260 := by
sorry


end NUMINAMATH_CALUDE_disco_vote_participants_l704_70475


namespace NUMINAMATH_CALUDE_complex_magnitude_equation_l704_70463

theorem complex_magnitude_equation (t : ℝ) : 
  (t > 0 ∧ Complex.abs (t + 2 * Complex.I * Real.sqrt 3) * Complex.abs (6 - 4 * Complex.I) = 26) → t = 1 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_equation_l704_70463


namespace NUMINAMATH_CALUDE_f_range_l704_70482

-- Define the closest multiple function
def closestMultiple (k : ℤ) (n : ℕ) : ℤ :=
  let m := (2 * n + 1 : ℤ)
  m * ((k + m / 2) / m)

-- Define the function f
def f (k : ℤ) : ℤ :=
  closestMultiple k 1 + closestMultiple (2 * k) 2 + closestMultiple (3 * k) 3 - 6 * k

-- State the theorem
theorem f_range :
  ∀ k : ℤ, -6 ≤ f k ∧ f k ≤ 6 :=
sorry

end NUMINAMATH_CALUDE_f_range_l704_70482


namespace NUMINAMATH_CALUDE_min_value_theorem_l704_70439

theorem min_value_theorem (x : ℝ) (h : x > 4) :
  (x + 15) / Real.sqrt (x - 4) ≥ 2 * Real.sqrt 19 ∧
  ((x + 15) / Real.sqrt (x - 4) = 2 * Real.sqrt 19 ↔ x = 23) := by
sorry

end NUMINAMATH_CALUDE_min_value_theorem_l704_70439


namespace NUMINAMATH_CALUDE_statistics_collection_count_l704_70469

/-- Represents the multiset of letters in "STATISTICS" --/
def statistics : Multiset Char := {'S', 'T', 'A', 'T', 'I', 'S', 'T', 'I', 'C', 'S'}

/-- Represents the vowels in "STATISTICS" --/
def vowels : Multiset Char := {'A', 'I', 'I'}

/-- Represents the consonants in "STATISTICS", with S and T treated as indistinguishable --/
def consonants : Multiset Char := {'C', 'S', 'S', 'S'}

/-- The number of distinct collections of 7 letters (3 vowels and 4 consonants) from "STATISTICS" --/
def distinct_collections : ℕ := 30

theorem statistics_collection_count :
  (Multiset.card statistics = 10) →
  (Multiset.card vowels = 3) →
  (Multiset.card consonants = 4) →
  (∀ x ∈ vowels, x ∈ statistics) →
  (∀ x ∈ consonants, x ∈ statistics ∨ x = 'S') →
  (distinct_collections = 30) := by
  sorry

end NUMINAMATH_CALUDE_statistics_collection_count_l704_70469


namespace NUMINAMATH_CALUDE_tiles_per_row_l704_70421

-- Define the area of the room
def room_area : ℝ := 144

-- Define the side length of a tile in meters
def tile_side : ℝ := 0.3

-- Theorem statement
theorem tiles_per_row (room_area : ℝ) (tile_side : ℝ) :
  room_area = 144 ∧ tile_side = 0.3 →
  (Real.sqrt room_area / tile_side : ℝ) = 40 := by
  sorry

end NUMINAMATH_CALUDE_tiles_per_row_l704_70421


namespace NUMINAMATH_CALUDE_g_at_zero_l704_70472

-- Define polynomials f, g, and h
variable (f g h : ℝ[X])

-- Define the relationship between h, f, and g
axiom h_eq_f_mul_g : h = f * g

-- Define the constant term of f
axiom f_constant_term : f.coeff 0 = 5

-- Define the constant term of h
axiom h_constant_term : h.coeff 0 = -10

-- Theorem to prove
theorem g_at_zero : g.eval 0 = -2 := by sorry

end NUMINAMATH_CALUDE_g_at_zero_l704_70472


namespace NUMINAMATH_CALUDE_cleanup_drive_total_l704_70473

theorem cleanup_drive_total (lizzie_group : ℕ) (other_group_difference : ℕ) 
  (h1 : lizzie_group = 387)
  (h2 : other_group_difference = 39) :
  lizzie_group + (lizzie_group - other_group_difference) = 735 :=
by sorry

end NUMINAMATH_CALUDE_cleanup_drive_total_l704_70473


namespace NUMINAMATH_CALUDE_cereal_eating_time_l704_70452

/-- The time it takes for two people to eat a certain amount of cereal together -/
def eating_time (rate1 rate2 amount : ℚ) : ℚ :=
  amount / (rate1 + rate2)

/-- The proposition that Mr. Fat and Mr. Thin can eat 4 pounds of cereal in 37.5 minutes -/
theorem cereal_eating_time :
  let fat_rate : ℚ := 1 / 15  -- Mr. Fat's eating rate in pounds per minute
  let thin_rate : ℚ := 1 / 25 -- Mr. Thin's eating rate in pounds per minute
  let total_amount : ℚ := 4   -- Total amount of cereal in pounds
  eating_time fat_rate thin_rate total_amount = 75 / 2 := by
  sorry

#eval eating_time (1/15 : ℚ) (1/25 : ℚ) 4

end NUMINAMATH_CALUDE_cereal_eating_time_l704_70452


namespace NUMINAMATH_CALUDE_consecutive_numbers_sum_l704_70465

theorem consecutive_numbers_sum (a : ℤ) :
  let consecutive_numbers := List.range 9
  let sum_of_eight := (consecutive_numbers.map (λ i => a + i - 4)).sum - (a + b - 4)
  ∃ b : ℤ, -4 ≤ b ∧ b ≤ 4 ∧ sum_of_eight = 1703 → a + b - 4 = 214 :=
by sorry

end NUMINAMATH_CALUDE_consecutive_numbers_sum_l704_70465


namespace NUMINAMATH_CALUDE_simplify_expression_l704_70451

theorem simplify_expression (x : ℝ) : 3*x + 6*x + 9*x + 12*x + 15*x + 18 = 45*x + 18 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l704_70451


namespace NUMINAMATH_CALUDE_anniversary_sale_cost_l704_70427

/-- The cost of the purchase during the anniversary sale -/
def total_cost (original_ice_cream_price sale_discount juice_price_per_5 ice_cream_tubs juice_cans : ℚ) : ℚ :=
  (ice_cream_tubs * (original_ice_cream_price - sale_discount)) + 
  (juice_cans / 5 * juice_price_per_5)

/-- Theorem stating that the total cost of the purchase is $24 -/
theorem anniversary_sale_cost : 
  total_cost 12 2 2 2 10 = 24 := by sorry

end NUMINAMATH_CALUDE_anniversary_sale_cost_l704_70427


namespace NUMINAMATH_CALUDE_jessicas_balloons_l704_70429

theorem jessicas_balloons (joan_balloons sally_balloons total_balloons : ℕ) 
  (h1 : joan_balloons = 9)
  (h2 : sally_balloons = 5)
  (h3 : total_balloons = 16) :
  total_balloons - (joan_balloons + sally_balloons) = 2 := by
  sorry

end NUMINAMATH_CALUDE_jessicas_balloons_l704_70429


namespace NUMINAMATH_CALUDE_ladder_problem_l704_70417

theorem ladder_problem (ladder_length height : ℝ) 
  (h1 : ladder_length = 13)
  (h2 : height = 12) :
  ∃ (base : ℝ), base^2 + height^2 = ladder_length^2 ∧ base = 5 := by
  sorry

end NUMINAMATH_CALUDE_ladder_problem_l704_70417


namespace NUMINAMATH_CALUDE_system_of_equations_solution_l704_70499

theorem system_of_equations_solution :
  ∃! (x y : ℝ), x + 2*y = 4 ∧ x + 3*y = 5 :=
by
  sorry

end NUMINAMATH_CALUDE_system_of_equations_solution_l704_70499


namespace NUMINAMATH_CALUDE_f_not_in_third_quadrant_l704_70495

/-- The quadratic function under consideration -/
def f (x : ℝ) : ℝ := x^2 - 2*x

/-- A point is in the third quadrant if both its x and y coordinates are negative -/
def in_third_quadrant (x y : ℝ) : Prop := x < 0 ∧ y < 0

/-- Theorem stating that the graph of f does not pass through the third quadrant -/
theorem f_not_in_third_quadrant :
  ∀ x : ℝ, ¬(in_third_quadrant x (f x)) :=
sorry

end NUMINAMATH_CALUDE_f_not_in_third_quadrant_l704_70495


namespace NUMINAMATH_CALUDE_work_completion_time_l704_70407

/-- The time it takes for A, B, and C to complete a work together -/
def time_together (time_A time_B time_C : ℚ) : ℚ :=
  1 / (1 / time_A + 1 / time_B + 1 / time_C)

/-- Theorem stating that A, B, and C can complete the work in 2 days -/
theorem work_completion_time :
  time_together 4 10 (20 / 3) = 2 := by sorry

end NUMINAMATH_CALUDE_work_completion_time_l704_70407


namespace NUMINAMATH_CALUDE_distribution_ways_eq_1080_l704_70403

/-- The number of ways to distribute 6 distinct items among 4 groups,
    where two groups receive 2 items each and two groups receive 1 item each -/
def distribution_ways : ℕ :=
  (Nat.choose 6 2 * Nat.choose 4 2) / 2 * 24

/-- Theorem stating that the number of distribution ways is 1080 -/
theorem distribution_ways_eq_1080 : distribution_ways = 1080 := by
  sorry

end NUMINAMATH_CALUDE_distribution_ways_eq_1080_l704_70403


namespace NUMINAMATH_CALUDE_quadratic_always_positive_a_squared_plus_a_zero_not_equivalent_a_zero_a_plus_b_greater_than_two_ab_greater_than_one_not_equivalent_a_greater_than_four_iff_positive_roots_l704_70480

-- Proposition A
theorem quadratic_always_positive : ∀ x : ℝ, x^2 - x + 1 > 0 := by sorry

-- Proposition B
theorem a_squared_plus_a_zero_not_equivalent_a_zero : ∃ a : ℝ, a^2 + a = 0 ∧ a ≠ 0 := by sorry

-- Proposition C
theorem a_plus_b_greater_than_two_ab_greater_than_one_not_equivalent :
  ∃ a b : ℝ, a + b > 2 ∧ a * b > 1 ∧ ¬(a > 1 ∧ b > 1) := by sorry

-- Proposition D
theorem a_greater_than_four_iff_positive_roots (a : ℝ) :
  (∀ x : ℝ, x^2 - a*x + a = 0 → x > 0) ↔ a > 4 := by sorry

end NUMINAMATH_CALUDE_quadratic_always_positive_a_squared_plus_a_zero_not_equivalent_a_zero_a_plus_b_greater_than_two_ab_greater_than_one_not_equivalent_a_greater_than_four_iff_positive_roots_l704_70480


namespace NUMINAMATH_CALUDE_red_stamp_price_proof_l704_70474

/-- The number of red stamps Simon has -/
def simon_stamps : ℕ := 30

/-- The number of white stamps Peter has -/
def peter_stamps : ℕ := 80

/-- The price of each white stamp in dollars -/
def white_stamp_price : ℚ := 1/5

/-- The difference in the amount of money they make in dollars -/
def money_difference : ℚ := 1

/-- The price of each red stamp in dollars -/
def red_stamp_price : ℚ := 17/30

theorem red_stamp_price_proof :
  simon_stamps * red_stamp_price - peter_stamps * white_stamp_price = money_difference :=
sorry

end NUMINAMATH_CALUDE_red_stamp_price_proof_l704_70474


namespace NUMINAMATH_CALUDE_jeff_total_distance_l704_70418

/-- Represents a segment of Jeff's journey --/
structure Segment where
  speed : ℝ
  duration : ℝ

/-- Calculates the distance traveled in a segment --/
def distanceInSegment (s : Segment) : ℝ := s.speed * s.duration

/-- Jeff's journey segments --/
def jeffJourney : List Segment := [
  ⟨80, 3⟩, ⟨50, 2⟩, ⟨70, 1⟩, ⟨60, 1.5⟩, ⟨45, 1.5⟩,
  ⟨60, 1.5⟩, ⟨35, 2⟩, ⟨40, 2⟩, ⟨30, 2.5⟩, ⟨25, 1⟩
]

/-- Theorem: The total distance Jeff traveled is 907.5 miles --/
theorem jeff_total_distance :
  (jeffJourney.map distanceInSegment).sum = 907.5 := by
  sorry


end NUMINAMATH_CALUDE_jeff_total_distance_l704_70418


namespace NUMINAMATH_CALUDE_final_shell_count_l704_70402

def shell_collection (initial_shells : ℕ) (shells_per_day : ℕ) (days : ℕ) (extra_shells : ℕ) : ℕ :=
  initial_shells + shells_per_day * days + extra_shells

theorem final_shell_count :
  shell_collection 20 5 3 6 = 41 := by
  sorry

end NUMINAMATH_CALUDE_final_shell_count_l704_70402


namespace NUMINAMATH_CALUDE_smallest_integer_satisfying_inequality_ten_satisfies_inequality_ten_is_smallest_satisfying_integer_l704_70466

theorem smallest_integer_satisfying_inequality :
  ∀ n : ℤ, n^2 - 14*n + 45 > 0 → n ≥ 10 :=
by sorry

theorem ten_satisfies_inequality :
  10^2 - 14*10 + 45 > 0 :=
by sorry

theorem ten_is_smallest_satisfying_integer :
  ∀ n : ℤ, n < 10 → n^2 - 14*n + 45 ≤ 0 :=
by sorry

end NUMINAMATH_CALUDE_smallest_integer_satisfying_inequality_ten_satisfies_inequality_ten_is_smallest_satisfying_integer_l704_70466


namespace NUMINAMATH_CALUDE_area_of_EFGH_l704_70423

/-- Parallelogram with vertices E, F, G, H in 2D space -/
structure Parallelogram where
  E : ℝ × ℝ
  F : ℝ × ℝ
  G : ℝ × ℝ
  H : ℝ × ℝ

/-- Calculate the area of a parallelogram -/
def parallelogramArea (p : Parallelogram) : ℝ :=
  let base := |p.F.2 - p.E.2|
  let height := |p.G.1 - p.E.1|
  base * height

/-- The specific parallelogram EFGH from the problem -/
def EFGH : Parallelogram :=
  { E := (2, -3)
    F := (2, 2)
    G := (7, 9)
    H := (7, 2) }

theorem area_of_EFGH : parallelogramArea EFGH = 25 := by
  sorry

end NUMINAMATH_CALUDE_area_of_EFGH_l704_70423


namespace NUMINAMATH_CALUDE_johnson_family_reunion_ratio_l704_70428

theorem johnson_family_reunion_ratio :
  ∀ (total_adults : ℕ) (total_children : ℕ),
  total_children = 45 →
  (total_adults / 3 : ℚ) + 10 = total_adults →
  (total_adults : ℚ) / total_children = 1 / 3 :=
by
  sorry

end NUMINAMATH_CALUDE_johnson_family_reunion_ratio_l704_70428


namespace NUMINAMATH_CALUDE_empty_intersection_l704_70497

def S : Set ℚ := {x : ℚ | x ≠ -1 ∧ x ≠ 0 ∧ x ≠ 1}

def f (x : ℚ) : ℚ := x - 1/x

def f_iter (n : ℕ) : Set ℚ → Set ℚ :=
  match n with
  | 0 => id
  | n + 1 => f_iter n ∘ (λ s => f '' s)

theorem empty_intersection :
  (⋂ n : ℕ, f_iter n S) = ∅ := by sorry

end NUMINAMATH_CALUDE_empty_intersection_l704_70497


namespace NUMINAMATH_CALUDE_consecutive_even_integers_sum_l704_70435

theorem consecutive_even_integers_sum (a : ℤ) : 
  (a + (a + 4) = 144) → 
  (a + (a + 2) + (a + 4) + (a + 6) + (a + 8) = 370) := by
  sorry

end NUMINAMATH_CALUDE_consecutive_even_integers_sum_l704_70435


namespace NUMINAMATH_CALUDE_max_plus_min_of_f_l704_70408

def f (x : ℝ) : ℝ := x^3 - 3*x + 1

theorem max_plus_min_of_f : 
  ∃ (m n : ℝ), (∀ x, f x ≤ m) ∧ (∃ x₁, f x₁ = m) ∧ 
               (∀ x, n ≤ f x) ∧ (∃ x₂, f x₂ = n) ∧ 
               m + n = 2 := by
  sorry

end NUMINAMATH_CALUDE_max_plus_min_of_f_l704_70408


namespace NUMINAMATH_CALUDE_exponent_equality_comparison_l704_70411

theorem exponent_equality_comparison : 
  (4^3 ≠ 3^4) ∧ 
  (-5^3 = (-5)^3) ∧ 
  ((-6)^2 ≠ -6^2) ∧ 
  (((-5/2)^2 : ℚ) ≠ ((-2/5)^2 : ℚ)) :=
by sorry

end NUMINAMATH_CALUDE_exponent_equality_comparison_l704_70411


namespace NUMINAMATH_CALUDE_ryan_leaf_collection_l704_70401

theorem ryan_leaf_collection (lost broken left initial : ℕ) : 
  lost = 24 → broken = 43 → left = 22 → initial = lost + broken + left :=
by sorry

end NUMINAMATH_CALUDE_ryan_leaf_collection_l704_70401


namespace NUMINAMATH_CALUDE_fish_market_problem_l704_70424

theorem fish_market_problem (mackerel croaker tuna : ℕ) : 
  mackerel = 48 →
  mackerel * 11 = croaker * 6 →
  croaker * 8 = tuna →
  tuna = 704 := by
sorry

end NUMINAMATH_CALUDE_fish_market_problem_l704_70424


namespace NUMINAMATH_CALUDE_floor_ceiling_sum_l704_70436

theorem floor_ceiling_sum : ⌊(-2.54 : ℝ)⌋ + ⌈(25.4 : ℝ)⌉ = 23 := by sorry

end NUMINAMATH_CALUDE_floor_ceiling_sum_l704_70436


namespace NUMINAMATH_CALUDE_system_solvability_l704_70442

/-- The system of equations and inequalities -/
def system (a b x y : ℝ) : Prop :=
  x * Real.cos a - y * Real.sin a - 3 ≤ 0 ∧
  x^2 + y^2 - 8*x + 2*y - b^2 - 6*b + 8 = 0

/-- The set of valid b values -/
def valid_b_set : Set ℝ :=
  {b | b ≤ -Real.sqrt 17 ∨ b ≥ Real.sqrt 17 - 6}

/-- Theorem stating the equivalence between the system having a solution
    for any a and b being in the valid set -/
theorem system_solvability (b : ℝ) :
  (∀ a, ∃ x y, system a b x y) ↔ b ∈ valid_b_set :=
sorry

end NUMINAMATH_CALUDE_system_solvability_l704_70442


namespace NUMINAMATH_CALUDE_difference_of_squares_l704_70488

theorem difference_of_squares (x y : ℝ) (h1 : x + y = 10) (h2 : x - y = 8) : x^2 - y^2 = 80 := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_l704_70488


namespace NUMINAMATH_CALUDE_smallest_non_phd_count_l704_70447

/-- The tournament structure -/
structure Tournament where
  total_participants : ℕ
  phd_participants : ℕ
  non_phd_participants : ℕ
  total_points : ℕ
  phd_points : ℕ
  non_phd_points : ℕ

/-- The theorem to prove -/
theorem smallest_non_phd_count (t : Tournament) : 
  199 ≤ t.total_participants ∧ 
  t.total_participants ≤ 229 ∧
  t.total_participants = t.phd_participants + t.non_phd_participants ∧
  t.total_points = t.total_participants * (t.total_participants - 1) / 2 ∧
  t.phd_points = t.phd_participants * (t.phd_participants - 1) / 2 ∧
  t.non_phd_points = t.non_phd_participants * (t.non_phd_participants - 1) / 2 ∧
  2 * (t.phd_points + t.non_phd_points) = t.total_points →
  t.non_phd_participants ≥ 105 ∧ 
  ∃ (t' : Tournament), t'.non_phd_participants = 105 ∧ 
    199 ≤ t'.total_participants ∧ 
    t'.total_participants ≤ 229 ∧
    t'.total_participants = t'.phd_participants + t'.non_phd_participants ∧
    t'.total_points = t'.total_participants * (t'.total_participants - 1) / 2 ∧
    t'.phd_points = t'.phd_participants * (t'.phd_participants - 1) / 2 ∧
    t'.non_phd_points = t'.non_phd_participants * (t'.non_phd_participants - 1) / 2 ∧
    2 * (t'.phd_points + t'.non_phd_points) = t'.total_points :=
by sorry

end NUMINAMATH_CALUDE_smallest_non_phd_count_l704_70447


namespace NUMINAMATH_CALUDE_problem_1_l704_70416

theorem problem_1 : |-3| - 2 - (-6) / (-2) = -2 := by
  sorry

end NUMINAMATH_CALUDE_problem_1_l704_70416


namespace NUMINAMATH_CALUDE_steven_needs_three_more_seeds_l704_70415

/-- The number of seeds needed for the assignment -/
def assignment_seeds : ℕ := 60

/-- The average number of seeds in an apple -/
def apple_seeds : ℕ := 6

/-- The average number of seeds in a pear -/
def pear_seeds : ℕ := 2

/-- The average number of seeds in a grape -/
def grape_seeds : ℕ := 3

/-- The number of apples Steven has -/
def steven_apples : ℕ := 4

/-- The number of pears Steven has -/
def steven_pears : ℕ := 3

/-- The number of grapes Steven has -/
def steven_grapes : ℕ := 9

/-- The number of additional seeds Steven needs -/
def additional_seeds_needed : ℕ := 3

theorem steven_needs_three_more_seeds :
  assignment_seeds - (steven_apples * apple_seeds + steven_pears * pear_seeds + steven_grapes * grape_seeds) = additional_seeds_needed := by
  sorry

end NUMINAMATH_CALUDE_steven_needs_three_more_seeds_l704_70415


namespace NUMINAMATH_CALUDE_arithmetic_sequence_formula_l704_70467

/-- An arithmetic sequence with first term a-1 and common difference 2 has the general formula a_n = a + 2n - 3 -/
theorem arithmetic_sequence_formula (a : ℝ) :
  let a_n := fun (n : ℕ) => a - 1 + 2 * (n - 1)
  ∀ n : ℕ, a_n n = a + 2 * n - 3 :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_formula_l704_70467


namespace NUMINAMATH_CALUDE_polynomial_divisibility_l704_70487

theorem polynomial_divisibility (p q : ℤ) : 
  (∀ x : ℤ, (x - 2) * (x + 1) ∣ (x^5 - x^4 + x^3 - p*x^2 + q*x - 8)) →
  p = -1 ∧ q = -10 := by
sorry

end NUMINAMATH_CALUDE_polynomial_divisibility_l704_70487


namespace NUMINAMATH_CALUDE_twelve_sided_polygon_area_l704_70448

/-- A 12-sided polygon composed of squares and triangles on a grid --/
structure TwelveSidedPolygon where
  center_square : ℝ  -- Area of the center square
  corner_triangles : ℝ  -- Number of corner triangles
  side_triangles : ℝ  -- Number of effective side triangles
  unit_square_area : ℝ  -- Area of a unit square
  unit_triangle_area : ℝ  -- Area of a unit right triangle

/-- The area of the 12-sided polygon --/
def polygon_area (p : TwelveSidedPolygon) : ℝ :=
  p.center_square * p.unit_square_area +
  p.corner_triangles * p.unit_triangle_area +
  p.side_triangles * p.unit_square_area

/-- Theorem stating that the area of the specific 12-sided polygon is 13 square units --/
theorem twelve_sided_polygon_area :
  ∀ (p : TwelveSidedPolygon),
  p.center_square = 9 ∧
  p.corner_triangles = 4 ∧
  p.side_triangles = 4 ∧
  p.unit_square_area = 1 ∧
  p.unit_triangle_area = 1/2 →
  polygon_area p = 13 := by
  sorry

end NUMINAMATH_CALUDE_twelve_sided_polygon_area_l704_70448


namespace NUMINAMATH_CALUDE_probability_of_double_is_two_ninths_l704_70446

/-- Represents a domino tile with two squares -/
structure Domino :=
  (first : Nat)
  (second : Nat)

/-- The set of all possible dominos with integers from 0 to 7 -/
def dominoSet : Finset Domino :=
  sorry

/-- Predicate to check if a domino is a double -/
def isDouble (d : Domino) : Bool :=
  d.first = d.second

/-- The probability of selecting a double from the domino set -/
def probabilityOfDouble : ℚ :=
  sorry

/-- Theorem stating that the probability of selecting a double is 2/9 -/
theorem probability_of_double_is_two_ninths :
  probabilityOfDouble = 2 / 9 := by
  sorry

end NUMINAMATH_CALUDE_probability_of_double_is_two_ninths_l704_70446


namespace NUMINAMATH_CALUDE_quadratic_root_relation_l704_70485

/-- Given two quadratic equations where the roots of one are three times the roots of the other,
    prove that the ratio of certain coefficients is 27. -/
theorem quadratic_root_relation (m n p : ℝ) (hm : m ≠ 0) (hn : n ≠ 0) (hp : p ≠ 0) :
  (∃ (s₁ s₂ : ℝ),
    (s₁ + s₂ = -p ∧ s₁ * s₂ = m) ∧
    (3*s₁ + 3*s₂ = -m ∧ 9*s₁ * s₂ = n)) →
  n / p = 27 := by
sorry

end NUMINAMATH_CALUDE_quadratic_root_relation_l704_70485


namespace NUMINAMATH_CALUDE_max_k_value_l704_70496

theorem max_k_value (m : ℝ) (h1 : 0 < m) (h2 : m < 1/2) :
  (∀ k : ℝ, (1/m + 2/(1-2*m) ≥ k) → k ≤ 8) ∧ 
  ∃ k : ℝ, k = 8 ∧ 1/m + 2/(1-2*m) ≥ k :=
sorry

end NUMINAMATH_CALUDE_max_k_value_l704_70496


namespace NUMINAMATH_CALUDE_three_digit_sum_problem_l704_70419

/-- Represents a three-digit number in the form abc -/
structure ThreeDigitNumber where
  hundreds : Nat
  tens : Nat
  ones : Nat
  is_valid : hundreds ≥ 1 ∧ hundreds ≤ 9 ∧ tens ≥ 0 ∧ tens ≤ 9 ∧ ones ≥ 0 ∧ ones ≤ 9

/-- Converts a ThreeDigitNumber to a natural number -/
def ThreeDigitNumber.toNat (n : ThreeDigitNumber) : Nat :=
  100 * n.hundreds + 10 * n.tens + n.ones

theorem three_digit_sum_problem (a c : Nat) :
  let num1 := ThreeDigitNumber.mk 3 a 7 (by sorry)
  let num2 := ThreeDigitNumber.mk 2 1 4 (by sorry)
  let sum := ThreeDigitNumber.mk 5 c 1 (by sorry)
  (num1.toNat + num2.toNat = sum.toNat) →
  (sum.toNat % 3 = 0) →
  a + c = 4 := by
  sorry

#check three_digit_sum_problem

end NUMINAMATH_CALUDE_three_digit_sum_problem_l704_70419


namespace NUMINAMATH_CALUDE_circle_ratio_l704_70461

theorem circle_ratio (r₁ r₂ : ℝ) (h : r₁ > 0 ∧ r₂ > 0) 
  (h_area : π * r₂^2 - π * r₁^2 = 4 * π * r₁^2) : 
  r₁ / r₂ = 1 / Real.sqrt 5 := by
sorry

end NUMINAMATH_CALUDE_circle_ratio_l704_70461


namespace NUMINAMATH_CALUDE_product_expansion_l704_70437

theorem product_expansion (x y : ℝ) :
  (3 * x^4 - 7 * y^3) * (9 * x^8 + 21 * x^4 * y^3 + 49 * y^6) = 27 * x^12 - 343 * y^9 := by
  sorry

end NUMINAMATH_CALUDE_product_expansion_l704_70437


namespace NUMINAMATH_CALUDE_dog_age_difference_l704_70484

/-- The ratio of dog years to human years -/
def dogYearRatio : ℕ := 7

/-- The age of Max (the human) in years -/
def maxAge : ℕ := 3

/-- The age of Max's dog in human years -/
def dogAgeHuman : ℕ := 3

/-- Calculates the age of a dog in dog years given its age in human years -/
def dogAgeInDogYears (humanYears : ℕ) : ℕ := humanYears * dogYearRatio

/-- The difference in years between a dog's age in dog years and its owner's age in human years -/
def ageDifference (humanAge : ℕ) (dogAgeHuman : ℕ) : ℕ :=
  dogAgeInDogYears dogAgeHuman - humanAge

theorem dog_age_difference :
  ageDifference maxAge dogAgeHuman = 18 := by
  sorry

end NUMINAMATH_CALUDE_dog_age_difference_l704_70484


namespace NUMINAMATH_CALUDE_sequence_equality_l704_70462

/-- Sequence a_n defined recursively -/
def a : ℕ → ℚ
  | 0 => 0
  | 1 => 1
  | (n + 2) => 2018 / (n + 1) * a (n + 1) + a n

/-- Sequence b_n defined recursively -/
def b : ℕ → ℚ
  | 0 => 0
  | 1 => 1
  | (n + 2) => 2020 / (n + 1) * b (n + 1) + b n

/-- The main theorem to prove -/
theorem sequence_equality : a 1010 / 1010 = b 1009 / 1009 := by
  sorry

end NUMINAMATH_CALUDE_sequence_equality_l704_70462


namespace NUMINAMATH_CALUDE_quadratic_equation_condition_l704_70478

theorem quadratic_equation_condition (m : ℝ) : 
  (|m - 2| = 2 ∧ m - 4 ≠ 0) ↔ m = 0 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_condition_l704_70478


namespace NUMINAMATH_CALUDE_original_mean_calculation_l704_70400

theorem original_mean_calculation (n : ℕ) (decrease : ℝ) (new_mean : ℝ) (h1 : n = 50) (h2 : decrease = 15) (h3 : new_mean = 185) : 
  (n : ℝ) * new_mean + n * decrease = n * 200 := by
  sorry

end NUMINAMATH_CALUDE_original_mean_calculation_l704_70400


namespace NUMINAMATH_CALUDE_max_boxes_arrangement_l704_70425

/-- A Box represents a rectangle in the plane with sides parallel to coordinate axes. -/
structure Box where
  x₁ : ℝ
  y₁ : ℝ
  x₂ : ℝ
  y₂ : ℝ
  h_positive : x₁ < x₂ ∧ y₁ < y₂

/-- Two boxes intersect if they have a common point. -/
def intersect (b₁ b₂ : Box) : Prop :=
  ¬(b₁.x₂ ≤ b₂.x₁ ∨ b₂.x₂ ≤ b₁.x₁ ∨ b₁.y₂ ≤ b₂.y₁ ∨ b₂.y₂ ≤ b₁.y₁)

/-- A valid arrangement of n boxes satisfies the intersection condition. -/
def valid_arrangement (n : ℕ) (boxes : Fin n → Box) : Prop :=
  ∀ i j : Fin n, intersect (boxes i) (boxes j) ↔ (i.val + 1) % n ≠ j.val ∧ (i.val + n - 1) % n ≠ j.val

/-- The main theorem: The maximum number of boxes in a valid arrangement is 6. -/
theorem max_boxes_arrangement :
  (∃ (boxes : Fin 6 → Box), valid_arrangement 6 boxes) ∧
  (∀ n : ℕ, n > 6 → ¬∃ (boxes : Fin n → Box), valid_arrangement n boxes) :=
sorry

end NUMINAMATH_CALUDE_max_boxes_arrangement_l704_70425


namespace NUMINAMATH_CALUDE_lineup_constraint_ways_l704_70404

/-- The number of ways to arrange 5 people in a line with constraints -/
def lineupWays : ℕ :=
  let totalPeople : ℕ := 5
  let firstPositionOptions : ℕ := totalPeople - 1
  let lastPositionOptions : ℕ := totalPeople - 2
  let middlePositionsOptions : ℕ := 3 * 2 * 1
  firstPositionOptions * lastPositionOptions * middlePositionsOptions

theorem lineup_constraint_ways :
  lineupWays = 216 := by
  sorry

end NUMINAMATH_CALUDE_lineup_constraint_ways_l704_70404


namespace NUMINAMATH_CALUDE_product_from_hcf_lcm_l704_70443

theorem product_from_hcf_lcm (a b : ℕ+) (h1 : Nat.gcd a b = 55) (h2 : Nat.lcm a b = 1500) :
  a * b = 82500 := by
  sorry

end NUMINAMATH_CALUDE_product_from_hcf_lcm_l704_70443


namespace NUMINAMATH_CALUDE_apps_added_l704_70410

theorem apps_added (initial_apps final_apps : ℕ) (h1 : initial_apps = 17) (h2 : final_apps = 18) :
  final_apps - initial_apps = 1 := by
  sorry

end NUMINAMATH_CALUDE_apps_added_l704_70410


namespace NUMINAMATH_CALUDE_sum_of_square_areas_l704_70413

/-- The sum of areas of an infinite sequence of squares -/
theorem sum_of_square_areas (first_side : ℝ) (h : first_side = 4) : 
  let area_ratio : ℝ := (0.5 * Real.sqrt 2)^2
  let first_area : ℝ := first_side^2
  let sum_areas : ℝ := first_area / (1 - area_ratio)
  sum_areas = 32 := by sorry

end NUMINAMATH_CALUDE_sum_of_square_areas_l704_70413


namespace NUMINAMATH_CALUDE_expression_evaluation_l704_70420

theorem expression_evaluation : -25 + 12 * (8 / (2 + 2)) = -1 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l704_70420


namespace NUMINAMATH_CALUDE_min_lines_for_100_squares_l704_70444

/-- The number of squares formed by n lines when n is odd -/
def squares_odd (n : ℕ) : ℕ := ((n - 3) * (n - 1) * (n + 1)) / 24

/-- The number of squares formed by n lines when n is even -/
def squares_even (n : ℕ) : ℕ := ((n - 2) * n * (n - 1)) / 24

/-- The maximum number of squares that can be formed by n lines -/
def max_squares (n : ℕ) : ℕ :=
  if n % 2 = 0 then squares_even n else squares_odd n

/-- Predicate indicating whether it's possible to form exactly k squares with n lines -/
def can_form_squares (n k : ℕ) : Prop :=
  k ≤ max_squares n ∧ k > max_squares (n - 1)

theorem min_lines_for_100_squares :
  ∃ n : ℕ, can_form_squares n 100 ∧ ∀ m : ℕ, m < n → ¬can_form_squares m 100 :=
sorry

end NUMINAMATH_CALUDE_min_lines_for_100_squares_l704_70444


namespace NUMINAMATH_CALUDE_doughnuts_per_person_l704_70434

def samuel_doughnuts : ℕ := 2 * 12
def cathy_doughnuts : ℕ := 3 * 12
def total_friends : ℕ := 8
def total_people : ℕ := total_friends + 2

theorem doughnuts_per_person :
  (samuel_doughnuts + cathy_doughnuts) / total_people = 6 :=
sorry

end NUMINAMATH_CALUDE_doughnuts_per_person_l704_70434


namespace NUMINAMATH_CALUDE_complex_fraction_equality_l704_70438

theorem complex_fraction_equality : (5 : ℂ) / (2 - I) = 2 + I := by sorry

end NUMINAMATH_CALUDE_complex_fraction_equality_l704_70438


namespace NUMINAMATH_CALUDE_prob_red_white_blue_eq_two_fifty_five_l704_70453

/-- The number of red marbles initially in the bag -/
def red_marbles : ℕ := 4

/-- The number of white marbles initially in the bag -/
def white_marbles : ℕ := 6

/-- The number of blue marbles initially in the bag -/
def blue_marbles : ℕ := 2

/-- The total number of marbles initially in the bag -/
def total_marbles : ℕ := red_marbles + white_marbles + blue_marbles

/-- The probability of drawing a red marble first, then a white marble, then a blue marble -/
def prob_red_white_blue : ℚ :=
  (red_marbles : ℚ) / total_marbles *
  (white_marbles : ℚ) / (total_marbles - 1) *
  (blue_marbles : ℚ) / (total_marbles - 2)

theorem prob_red_white_blue_eq_two_fifty_five :
  prob_red_white_blue = 2 / 55 := by sorry

end NUMINAMATH_CALUDE_prob_red_white_blue_eq_two_fifty_five_l704_70453


namespace NUMINAMATH_CALUDE_divided_number_l704_70476

theorem divided_number (x y n : ℤ) 
  (h_x_pos : x > 0)
  (h_div1 : x = n * y + 4)
  (h_div2 : 2 * x = 8 * (3 * y) + 3)
  (h_eq : 13 * y - x = 1) :
  n = 11 := by
sorry

end NUMINAMATH_CALUDE_divided_number_l704_70476


namespace NUMINAMATH_CALUDE_cubic_inequality_l704_70486

theorem cubic_inequality (x : ℝ) : x^3 - 4*x^2 + 4*x < 0 ↔ x < 0 := by
  sorry

end NUMINAMATH_CALUDE_cubic_inequality_l704_70486


namespace NUMINAMATH_CALUDE_football_throw_distance_l704_70483

theorem football_throw_distance (parker_distance : ℝ) :
  let grant_distance := parker_distance * 1.25
  let kyle_distance := grant_distance * 2
  kyle_distance - parker_distance = 24 →
  parker_distance = 16 := by
sorry

end NUMINAMATH_CALUDE_football_throw_distance_l704_70483


namespace NUMINAMATH_CALUDE_difference_of_difference_eq_intersection_l704_70432

-- Define the difference of two sets
def set_difference (A B : Set α) : Set α := {x | x ∈ A ∧ x ∉ B}

-- Theorem statement
theorem difference_of_difference_eq_intersection
  {α : Type*} (A B : Set α) (hA : A.Nonempty) (hB : B.Nonempty) :
  A \ (A \ B) = A ∩ B :=
sorry

end NUMINAMATH_CALUDE_difference_of_difference_eq_intersection_l704_70432


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l704_70431

theorem sufficient_not_necessary_condition (a b : ℝ) : 
  (∀ a b : ℝ, b ≥ 0 → (a + 1)^2 + b ≥ 0) ∧ 
  (∃ a b : ℝ, (a + 1)^2 + b ≥ 0 ∧ b < 0) := by
  sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l704_70431


namespace NUMINAMATH_CALUDE_initial_students_per_class_l704_70477

theorem initial_students_per_class 
  (initial_classes : ℕ) 
  (added_classes : ℕ) 
  (total_students : ℕ) 
  (h1 : initial_classes = 15)
  (h2 : added_classes = 5)
  (h3 : total_students = 400) :
  (total_students / (initial_classes + added_classes) : ℚ) = 20 := by
sorry

end NUMINAMATH_CALUDE_initial_students_per_class_l704_70477


namespace NUMINAMATH_CALUDE_puzzle_solvable_l704_70412

/-- Represents a polygonal piece --/
structure Piece where
  vertices : List (ℝ × ℝ)
  is_valid : List.length vertices ≥ 3

/-- Represents a shape formed by arranging pieces --/
structure Shape where
  pieces : List Piece
  arrangement : List (ℝ × ℝ) -- positions of pieces

/-- The original rectangle --/
def original_rectangle : Piece :=
  { vertices := [(0, 0), (4, 0), (4, 5), (0, 5)],
    is_valid := by sorry }

/-- The set of seven pieces cut from the original rectangle --/
def puzzle_pieces : List Piece :=
  sorry -- Define the seven pieces here

/-- The set of target shapes to be formed --/
def target_shapes : List Shape :=
  sorry -- Define the target shapes here

/-- Checks if a given arrangement of pieces forms a valid shape --/
def is_valid_arrangement (pieces : List Piece) (arrangement : List (ℝ × ℝ)) : Prop :=
  sorry -- Define the conditions for a valid arrangement

/-- The main theorem stating that the puzzle pieces can form the target shapes --/
theorem puzzle_solvable :
  ∀ shape ∈ target_shapes,
  ∃ arrangement : List (ℝ × ℝ),
  is_valid_arrangement puzzle_pieces arrangement ∧
  Shape.pieces shape = puzzle_pieces ∧
  Shape.arrangement shape = arrangement :=
sorry

end NUMINAMATH_CALUDE_puzzle_solvable_l704_70412


namespace NUMINAMATH_CALUDE_fraction_value_l704_70481

theorem fraction_value (N : ℝ) (h : 0.4 * N = 168) : (1/4) * (1/3) * (2/5) * N = 14 := by
  sorry

end NUMINAMATH_CALUDE_fraction_value_l704_70481


namespace NUMINAMATH_CALUDE_western_olympiad_2004_l704_70468

theorem western_olympiad_2004 (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  1 < (a / Real.sqrt (a^2 + b^2)) + (b / Real.sqrt (b^2 + c^2)) + (c / Real.sqrt (c^2 + a^2)) ∧
  (a / Real.sqrt (a^2 + b^2)) + (b / Real.sqrt (b^2 + c^2)) + (c / Real.sqrt (c^2 + a^2)) ≤ 3 * Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_western_olympiad_2004_l704_70468


namespace NUMINAMATH_CALUDE_q_is_false_l704_70409

theorem q_is_false (h1 : ¬(p ∧ q)) (h2 : ¬¬p) : ¬q := by
  sorry

end NUMINAMATH_CALUDE_q_is_false_l704_70409


namespace NUMINAMATH_CALUDE_chalkboard_area_l704_70479

/-- The area of a rectangle with width 3.5 feet and length 2.3 times its width is 28.175 square feet. -/
theorem chalkboard_area : 
  let width : ℝ := 3.5
  let length : ℝ := 2.3 * width
  width * length = 28.175 := by sorry

end NUMINAMATH_CALUDE_chalkboard_area_l704_70479


namespace NUMINAMATH_CALUDE_ship_passengers_with_round_trip_tickets_l704_70440

theorem ship_passengers_with_round_trip_tickets 
  (total_passengers : ℝ) 
  (h1 : total_passengers > 0) 
  (round_trip_with_car : ℝ) 
  (h2 : round_trip_with_car = 0.15 * total_passengers) 
  (h3 : round_trip_with_car > 0) 
  (round_trip_without_car_ratio : ℝ) 
  (h4 : round_trip_without_car_ratio = 0.6) :
  (round_trip_with_car / (1 - round_trip_without_car_ratio)) / total_passengers = 0.375 := by
sorry

end NUMINAMATH_CALUDE_ship_passengers_with_round_trip_tickets_l704_70440
