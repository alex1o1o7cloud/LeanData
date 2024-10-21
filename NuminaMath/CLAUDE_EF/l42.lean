import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solve_exponential_equation_l42_4206

theorem solve_exponential_equation (m : ℝ) : (5 : ℝ)^m * (5 : ℝ)^m * (5 : ℝ)^m = (125 : ℝ)^3 → m = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solve_exponential_equation_l42_4206


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chess_match_probability_l42_4265

noncomputable def probability_a_wins (p_win : ℝ) (a_needed : ℕ) (b_needed : ℕ) : ℝ :=
  let p_2_games := p_win * p_win
  let p_3_games := 3 * (1 - p_win) * p_win * p_win
  let p_4_games := 3 * (1 - p_win) * (1 - p_win) * p_win * p_win
  p_2_games + p_3_games + p_4_games

theorem chess_match_probability (p_win : ℝ) (a_needed : ℕ) (b_needed : ℕ) : 
  p_win = 1/2 → a_needed = 2 → b_needed = 3 → 
  (probability_a_wins p_win a_needed b_needed) = 11/16 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_chess_match_probability_l42_4265


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_roots_min_degree_l42_4293

-- Define the polynomial G(x)
def G : ℝ → ℝ := sorry

-- Define the five points where G(x) = 2022
def x₁ : ℝ := sorry
def x₂ : ℝ := sorry
def x₃ : ℝ := sorry
def x₄ : ℝ := sorry
def x₅ : ℝ := sorry

-- Axioms based on the problem conditions
axiom distinct_points : x₁ < x₂ ∧ x₂ < x₃ ∧ x₃ < x₄ ∧ x₄ < x₅
axiom G_values : G x₁ = 2022 ∧ G x₂ = 2022 ∧ G x₃ = 2022 ∧ G x₄ = 2022 ∧ G x₅ = 2022
axiom G_symmetry : ∀ x : ℝ, G x = G (-12 - x)

-- Theorem statements
theorem sum_of_roots : x₁ + x₃ + x₅ = -18 := by sorry

theorem min_degree : ∃ n : ℕ, (∀ m : ℕ, (∀ x : ℝ, G x = 0 → (x - x₁) * (x - x₂) * (x - x₃) * (x - x₄) * (x - x₅) = 0) → m ≥ n) ∧ n = 6 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_roots_min_degree_l42_4293


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_omega_value_l42_4200

noncomputable section

open Real

/-- The function f(x) = sin(ωx - π/6) -/
def f (ω : ℝ) (x : ℝ) : ℝ := sin (ω * x - π / 6)

/-- The theorem stating the conditions and the result to be proved -/
theorem omega_value (ω : ℝ) (h1 : ω > 0) 
  (h2 : f ω 0 = - f ω (π / 2))
  (h3 : ∃! x, x ∈ Set.Ioo 0 (π / 2) ∧ f ω x = 0) :
  ω = 14 / 3 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_omega_value_l42_4200


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_shift_for_symmetry_l42_4299

/-- The original function f(x) = √3 * sin(x) + 3 * cos(x) -/
noncomputable def f (x : ℝ) : ℝ := Real.sqrt 3 * Real.sin x + 3 * Real.cos x

/-- The shifted function g(x) = f(x - m) -/
noncomputable def g (m : ℝ) (x : ℝ) : ℝ := f (x - m)

/-- A function is symmetric about the y-axis if f(x) = f(-x) for all x -/
def symmetricAboutYAxis (h : ℝ → ℝ) : Prop := ∀ x, h x = h (-x)

theorem min_shift_for_symmetry :
  ∃ m : ℝ, m > 0 ∧ symmetricAboutYAxis (g m) ∧ 
  ∀ m' : ℝ, m' > 0 ∧ symmetricAboutYAxis (g m') → m ≤ m' := by
  sorry

#check min_shift_for_symmetry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_shift_for_symmetry_l42_4299


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_properties_l42_4270

/-- An arithmetic sequence with its first term and the sum of its first three terms. -/
structure ArithmeticSequence where
  a1 : ℝ
  s3 : ℝ
  h1 : a1 = -9
  h2 : s3 / 3 - a1 = 1

/-- The common difference of the arithmetic sequence. -/
noncomputable def commonDifference (seq : ArithmeticSequence) : ℝ :=
  (seq.s3 - 3 * seq.a1) / 3

/-- The sum of the first n terms of the arithmetic sequence. -/
noncomputable def sumFirstNTerms (seq : ArithmeticSequence) (n : ℕ) : ℝ :=
  n * (2 * seq.a1 + (n - 1) * commonDifference seq) / 2

/-- The theorem stating the common difference and minimum value of the sum. -/
theorem arithmetic_sequence_properties (seq : ArithmeticSequence) :
  commonDifference seq = 1 ∧
  ∃ n : ℕ, ∀ m : ℕ, sumFirstNTerms seq n ≤ sumFirstNTerms seq m ∧
  sumFirstNTerms seq n = -45 := by
  sorry

#check arithmetic_sequence_properties

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_properties_l42_4270


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solution_l42_4288

theorem equation_solution : 
  ∀ x : ℝ, x ≠ 2 → (x + 2 = 2 / (x - 2) ↔ x = Real.sqrt 6 ∨ x = -Real.sqrt 6) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solution_l42_4288


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_upper_bound_of_expression_l42_4292

theorem upper_bound_of_expression (n : ℕ) (U : ℕ) : 
  (∀ k : ℕ, (4 * k + 7 > 1 ∧ 4 * k + 7 < U) → k < 20) →
  (∀ k : ℕ, k < 20 → (4 * k + 7 > 1 ∧ 4 * k + 7 < U)) →
  (∀ V : ℕ, V < U → ∃ k : ℕ, k < 20 ∧ 4 * k + 7 ≥ V) →
  U = 84 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_upper_bound_of_expression_l42_4292


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mans_rate_is_11_l42_4290

/-- Given a man's rowing speeds with and against a stream, calculate his rate in still water -/
noncomputable def mans_rate (speed_with_stream speed_against_stream : ℝ) : ℝ :=
  (speed_with_stream + speed_against_stream) / 2

/-- Theorem: The man's rate in still water is 11 km/h -/
theorem mans_rate_is_11 :
  mans_rate 18 4 = 11 := by
  -- Unfold the definition of mans_rate
  unfold mans_rate
  -- Simplify the arithmetic
  simp [add_div]
  -- Check that 22 / 2 = 11
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_mans_rate_is_11_l42_4290


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_newspaper_sales_l42_4246

/-- Represents the daily newspaper sales scenario -/
structure NewspaperSales where
  buyPrice : ℚ
  sellPrice : ℚ
  returnPrice : ℚ
  daysHighSales : ℕ
  daysLowSales : ℕ
  highSalesAmount : ℕ
  lowSalesAmount : ℕ

/-- Calculates the monthly profit for a given number of daily purchases -/
def monthlyProfit (s : NewspaperSales) (dailyPurchase : ℕ) : ℚ :=
  let highSalesProfit := s.daysHighSales * (s.sellPrice - s.buyPrice) * dailyPurchase
  let lowSalesProfit := s.daysLowSales * (
    s.lowSalesAmount * (s.sellPrice - s.buyPrice) -
    (dailyPurchase - s.lowSalesAmount) * (s.buyPrice - s.returnPrice)
  )
  highSalesProfit + lowSalesProfit

/-- The main theorem stating the optimal daily purchase and maximum profit -/
theorem optimal_newspaper_sales (s : NewspaperSales) 
  (h1 : s.buyPrice = 60/100)
  (h2 : s.sellPrice = 80/100)
  (h3 : s.returnPrice = 40/100)
  (h4 : s.daysHighSales = 20)
  (h5 : s.daysLowSales = 10)
  (h6 : s.highSalesAmount = 100)
  (h7 : s.lowSalesAmount = 70) :
  ∃ (optimalPurchase : ℕ) (maxProfit : ℚ),
    optimalPurchase = 100 ∧
    maxProfit = 480 ∧
    ∀ (x : ℕ), monthlyProfit s x ≤ maxProfit :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_newspaper_sales_l42_4246


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bounded_index_element_product_l42_4272

/-- Given a finite decreasing sequence of positive integers and an upper bound for their pairwise LCMs,
    the product of any index and the corresponding element is at most the upper bound. -/
theorem bounded_index_element_product
  (k : ℕ+)
  (a : Fin k → ℕ)
  (n : ℕ)
  (h_pos : ∀ i, a i > 0)
  (h_decr : ∀ i j, i < j → a i > a j)
  (h_lcm_bound : ∀ i j, i ≠ j → Nat.lcm (a i) (a j) ≤ n)
  (h_upper : a 0 ≤ n) :
  ∀ i : Fin k, (i.val + 1) * a i ≤ n :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_bounded_index_element_product_l42_4272


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_general_term_l42_4275

noncomputable def a : ℕ → ℝ
| 0 => 9
| n + 1 => (a n) ^ 2

theorem a_general_term : ∀ n : ℕ, a n = 3 ^ (2 ^ n) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_general_term_l42_4275


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_h_of_two_equals_ten_l42_4231

/-- Given a function h : ℝ → ℝ such that h(3x - 7) = 2x + 4 for all real x, prove that h(2) = 10 -/
theorem h_of_two_equals_ten (h : ℝ → ℝ) (h_def : ∀ x : ℝ, h (3*x - 7) = 2*x + 4) : h 2 = 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_h_of_two_equals_ten_l42_4231


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_only_last_set_not_pythagorean_l42_4277

/-- A set of three numbers representing the lengths of a triangle's sides. -/
structure TriangleSides where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Checks if the given sides satisfy the Pythagorean theorem. -/
def isPythagoreanTriple (sides : TriangleSides) : Prop :=
  sides.a^2 + sides.b^2 = sides.c^2 ∨ sides.a^2 + sides.c^2 = sides.b^2 ∨ sides.b^2 + sides.c^2 = sides.a^2

/-- The given sets of triangle sides. -/
noncomputable def givenSets : List TriangleSides := [
  ⟨1, Real.sqrt 3, 2⟩,
  ⟨5, 4, 3⟩,
  ⟨13, 12, 5⟩,
  ⟨2, 2, 3⟩
]

theorem only_last_set_not_pythagorean :
  ∀ sides ∈ givenSets, isPythagoreanTriple sides ↔ sides ≠ ⟨2, 2, 3⟩ := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_only_last_set_not_pythagorean_l42_4277


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dress_assignment_l42_4234

-- Define the girls and colors
inductive Girl : Type where
  | Katya : Girl
  | Olya : Girl
  | Liza : Girl
  | Rita : Girl

inductive Color : Type where
  | Pink : Color
  | Green : Color
  | Yellow : Color
  | Blue : Color

-- Define the dress assignment function
def dress : Girl → Color := sorry

-- Define the standing order
def next_girl : Girl → Girl := sorry

-- Conditions
axiom katya_not_pink_or_blue :
  dress Girl.Katya ≠ Color.Pink ∧ dress Girl.Katya ≠ Color.Blue

axiom green_between_liza_and_yellow :
  (dress Girl.Liza = Color.Pink ∧ dress (next_girl Girl.Liza) = Color.Green ∧
   dress (next_girl (next_girl Girl.Liza)) = Color.Yellow) ∨
  (dress Girl.Liza = Color.Yellow ∧ dress (next_girl Girl.Liza) = Color.Green ∧
   dress (next_girl (next_girl Girl.Liza)) = Color.Pink)

axiom rita_not_green_or_blue :
  dress Girl.Rita ≠ Color.Green ∧ dress Girl.Rita ≠ Color.Blue

axiom olya_between_rita_and_pink :
  (dress Girl.Rita = Color.Yellow ∧ dress (next_girl Girl.Rita) = Color.Blue ∧
   dress (next_girl (next_girl Girl.Rita)) = Color.Pink) ∨
  (dress Girl.Rita = Color.Yellow ∧ dress (next_girl Girl.Rita) = Color.Pink ∧
   dress (next_girl (next_girl Girl.Rita)) = Color.Blue)

-- Theorem to prove
theorem dress_assignment :
  dress Girl.Katya = Color.Green ∧
  dress Girl.Olya = Color.Blue ∧
  dress Girl.Liza = Color.Pink ∧
  dress Girl.Rita = Color.Yellow := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_dress_assignment_l42_4234


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complement_union_equals_set_l42_4297

open Set

def U : Set Nat := {1, 2, 3, 4, 5, 6, 7, 8}
def M : Set Nat := {2, 3, 4}
def P : Set Nat := {1, 3, 6}

theorem complement_union_equals_set : (U \ (M ∪ P)) = {5, 7, 8} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complement_union_equals_set_l42_4297


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_radius_with_parabolas_is_quarter_l42_4250

/-- The radius of a circle with eight congruent parabolas arranged around it -/
noncomputable def circle_radius_with_parabolas : ℝ :=
  let parabola := fun (x : ℝ) => x^2
  let num_parabolas := 8
  let tangent_angle := Real.pi / 4  -- 45° in radians
  1/4

/-- Theorem: The radius of the circle with eight congruent parabolas arranged around it is 1/4 -/
theorem circle_radius_with_parabolas_is_quarter : circle_radius_with_parabolas = 1/4 := by
  -- Unfold the definition of circle_radius_with_parabolas
  unfold circle_radius_with_parabolas
  -- The proof goes here, but we'll use sorry for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_radius_with_parabolas_is_quarter_l42_4250


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_l42_4218

-- Define the function f(x) = 2√x - 3x
noncomputable def f (x : ℝ) : ℝ := 2 * Real.sqrt x - 3 * x

-- Theorem statement
theorem max_value_of_f :
  ∃ (M : ℝ), M = 1/3 ∧ ∀ (x : ℝ), x ≥ 0 → f x ≤ M := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_l42_4218


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_matrix_power_result_l42_4289

theorem matrix_power_result (B : Matrix (Fin 2) (Fin 2) ℝ) 
  (h : B.vecMul ![3, -1] = ![12, -4]) :
  (B^4).vecMul ![3, -1] = ![768, -256] := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_matrix_power_result_l42_4289


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l42_4242

-- Define the function f as noncomputable
noncomputable def f (x : ℝ) : ℝ := Real.log (abs (Real.sin x))

-- State the theorem
theorem f_properties : 
  (∀ x, f (-x) = f x) ∧ 
  (∃ p > 0, ∀ x, f (x + p) = f x) ∧
  (∀ q, 0 < q → (∀ x, f (x + q) = f x) → q ≥ π) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l42_4242


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rect_to_polar_correct_l42_4237

noncomputable def rect_to_polar (x y : ℝ) : ℝ × ℝ :=
  let r := Real.sqrt (x^2 + y^2)
  let θ := if x > 0 then Real.arctan (y / x)
           else if x < 0 && y ≥ 0 then Real.arctan (y / x) + Real.pi
           else if x < 0 && y < 0 then Real.arctan (y / x) - Real.pi
           else if x = 0 && y > 0 then Real.pi / 2
           else if x = 0 && y < 0 then -Real.pi / 2
           else 0  -- undefined for (0, 0)
  (r, θ)

theorem rect_to_polar_correct (x y : ℝ) :
  let (r, θ) := rect_to_polar x y
  x = -3 ∧ y = 3 * Real.sqrt 3 →
  r > 0 ∧ 0 ≤ θ ∧ θ < 2 * Real.pi →
  r = 6 ∧ θ = 2 * Real.pi / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rect_to_polar_correct_l42_4237


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_segments_equal_l42_4211

-- Define the circles and points
variable (O O₁ : Set (EuclideanSpace ℝ (Fin 2)))
variable (A B E F C D : EuclideanSpace ℝ (Fin 2))

-- Define the conditions
variable (h1 : A ∈ O ∩ O₁)
variable (h2 : B ∈ O ∩ O₁)
variable (h3 : E ∈ O₁)
variable (h4 : F ∈ O)
variable (h5 : C ∈ O)
variable (h6 : D ∈ O₁)
variable (h7 : A ≠ B)
variable (h8 : A ≠ E)
variable (h9 : A ≠ F)
variable (h10 : B ≠ E)
variable (h11 : B ≠ F)
variable (h12 : E ≠ C)
variable (h13 : F ≠ D)

-- Define the equal angles condition
noncomputable def angle (P Q R : EuclideanSpace ℝ (Fin 2)) : ℝ := sorry

variable (h14 : angle B A E = angle B A F)

-- State the theorem
theorem chord_segments_equal :
  ‖E - C‖ = ‖F - D‖ :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_segments_equal_l42_4211


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_trajectory_and_tangent_lines_l42_4204

-- Define the curve C
def C : Set (ℝ × ℝ) := {p | p.1^2 = -4 * p.2}

-- Define the line l
def l : Set (ℝ × ℝ) := {p | p.1 - p.2 + 2 = 0}

-- Define the fixed point F
def F : ℝ × ℝ := (0, -1)

-- Define a point P on line l
def P (x₀ y₀ : ℝ) : Prop := (x₀, y₀) ∈ l

-- Define the function for |AF|⋅|BF|
def AF_BF_product (x₀ y₀ : ℝ) : ℝ := 2 * y₀^2 - 2 * y₀ + 5

theorem circle_trajectory_and_tangent_lines :
  ∃ (M : Set (ℝ × ℝ)) (A B : ℝ × ℝ),
    (∀ m ∈ M, (m.1 - F.1)^2 + (m.2 - F.2)^2 = (m.2 - 1)^2) ∧
    (∀ m ∈ M, m.2 ≤ 1) ∧
    (C = {p : ℝ × ℝ | ∃ m ∈ M, p = m}) ∧
    (∀ x₀ y₀ : ℝ, P x₀ y₀ → 
      (∃ A B : ℝ × ℝ, A ∈ C ∧ B ∈ C ∧
        (x₀ * A.1 + 2 * A.2 + 2 * y₀ = 0) ∧
        (x₀ * B.1 + 2 * B.2 + 2 * y₀ = 0))) ∧
    (∀ x₀ y₀ : ℝ, P x₀ y₀ → AF_BF_product x₀ y₀ ≥ 9/2) ∧
    (∃ x₀ y₀ : ℝ, P x₀ y₀ ∧ AF_BF_product x₀ y₀ = 9/2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_trajectory_and_tangent_lines_l42_4204


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_characterization_l42_4273

def PositiveInt := { n : ℕ // n > 0 }

def DivisibilityCondition (f : PositiveInt → PositiveInt) :=
  ∀ m n : PositiveInt, (n.val + (f m).val) ∣ ((f n).val + n.val * (f m).val)

theorem function_characterization (f : PositiveInt → PositiveInt) 
  (h : DivisibilityCondition f) :
  (∀ n : PositiveInt, (f n).val = 1) ∨ 
  (∀ n : PositiveInt, (f n).val = n.val ^ 2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_characterization_l42_4273


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_root_interval_of_f_l42_4227

noncomputable def f (x : ℝ) : ℝ := Real.log x / Real.log 10 + x - 3

theorem root_interval_of_f : ∃! k : ℤ, ∃ x : ℝ, k < x ∧ x < k + 1 ∧ f x = 0 ∧ k = 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_root_interval_of_f_l42_4227


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_monotonicity_l42_4238

-- Define strict monotonicity
def StrictlyMonotonic (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f x < f y

-- State the theorem
theorem polynomial_monotonicity (P : ℝ → ℝ) 
  (h1 : StrictlyMonotonic (λ x ↦ P (P x)))
  (h2 : StrictlyMonotonic (λ x ↦ P (P (P x)))) :
  StrictlyMonotonic P := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_monotonicity_l42_4238


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_count_l42_4249

theorem triangle_side_count (a b : ℕ) (ha : a = 8) (hb : b = 5) : 
  (Finset.range 13 \ Finset.range 4).card = 9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_count_l42_4249


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_less_than_2b_l42_4202

-- Define the function f(x) = 3^x + log_3(x)
noncomputable def f (x : ℝ) : ℝ := 3^x + Real.log x / Real.log 3

-- State the theorem
theorem a_less_than_2b (a b : ℝ) (h : f a = 3^(2*b) + Real.log b / Real.log 3) : a < 2*b := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_less_than_2b_l42_4202


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_width_decrease_l42_4203

theorem rectangle_width_decrease (L W : ℝ) (h : L > 0 ∧ W > 0) : 
  let new_length := 1.4 * L
  let new_width := W * L / new_length
  abs ((W - new_width) / W - 0.2857) < 0.0001 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_width_decrease_l42_4203


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_scalar_multiple_l42_4260

theorem vector_scalar_multiple (m : ℝ) :
  let a : Fin 2 → ℝ := ![4, 2]
  let b : Fin 2 → ℝ := ![m, 3]
  (∃ k : ℝ, a = k • b) → m = 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_scalar_multiple_l42_4260


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_side_ratio_in_triangle_l42_4226

theorem max_side_ratio_in_triangle (a b c : ℝ) (A B C : ℝ) :
  a > 0 ∧ b > 0 ∧ c > 0 →
  0 < A ∧ A ≤ π / 2 →
  0 < B ∧ B < π →
  0 < C ∧ C < π →
  A + B + C = π →
  a = b * Real.sin C →
  a = c * Real.sin B →
  b = c * Real.sin A →
  a / 2 = b * c * Real.sin A / 2 →
  ∃ (max_ratio : ℝ), max_ratio = Real.sqrt 2 + 1 ∧
    ∀ (ratio : ℝ), ratio = c / b → ratio ≤ max_ratio :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_side_ratio_in_triangle_l42_4226


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_f_equals_sqrt_three_l42_4254

noncomputable def f (n : ℕ+) : ℝ := Real.tan (n * Real.pi / 3)

theorem sum_of_f_equals_sqrt_three :
  (Finset.range 100).sum (λ i => f ⟨i + 1, Nat.succ_pos i⟩) = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_f_equals_sqrt_three_l42_4254


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_variance_invariant_under_translation_l42_4282

noncomputable def variance (s : List ℝ) : ℝ := 
  let mean := s.sum / s.length
  (s.map (λ x => (x - mean) ^ 2)).sum / s.length

theorem variance_invariant_under_translation (s : List ℝ) (c : ℝ) :
  variance s = variance (s.map (λ x => x + c)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_variance_invariant_under_translation_l42_4282


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_sum_equivalence_l42_4215

theorem triangle_angle_sum_equivalence (A B C : ℝ) :
  (A + B + C = π) →
  (Real.sin A + Real.sin B = Real.cos A + Real.cos B ↔ C = π / 2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_sum_equivalence_l42_4215


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_second_race_remaining_distance_l42_4291

/-- Represents a runner in the race -/
structure Runner where
  speed : ℝ

/-- Represents the race conditions -/
structure RaceConditions where
  distance : ℝ
  runner_a : Runner
  runner_b : Runner
  b_remaining_distance : ℝ

/-- Theorem stating the distance B will be from the finish line when A finishes the second race -/
theorem second_race_remaining_distance 
  (race1 : RaceConditions)
  (h1 : race1.distance = 10000)
  (h2 : race1.b_remaining_distance = 500)
  (h3 : race1.runner_a.speed * 50 = race1.distance)
  (race2 : RaceConditions)
  (h4 : race2.distance = race1.distance + 500)
  (h5 : race2.runner_a = race1.runner_a)
  (h6 : race2.runner_b = race1.runner_b) :
  race2.distance - (race2.runner_b.speed * (race2.distance / race2.runner_a.speed)) = 25 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_second_race_remaining_distance_l42_4291


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lines_parallel_and_perpendicular_l42_4210

/-- Represents a line in the form ax + by = c --/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Calculates the slope of a line --/
noncomputable def Line.slope (l : Line) : ℝ := -l.a / l.b

/-- Checks if two lines are parallel --/
def are_parallel (l1 l2 : Line) : Prop := l1.slope = l2.slope

/-- Checks if two lines are perpendicular --/
def are_perpendicular (l1 l2 : Line) : Prop := l1.slope * l2.slope = -1

/-- The first line l1: x + (1+m)y = 2-m --/
def l1 (m : ℝ) : Line := { a := 1, b := 1 + m, c := 2 - m }

/-- The second line l2: 2mx + 4y = -16 --/
def l2 (m : ℝ) : Line := { a := 2 * m, b := 4, c := -16 }

theorem lines_parallel_and_perpendicular :
  (are_parallel (l1 1) (l2 1)) ∧
  (are_perpendicular (l1 (-2/3)) (l2 (-2/3))) := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_lines_parallel_and_perpendicular_l42_4210


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_alice_journey_distance_l42_4258

/-- Represents a hybrid car journey --/
structure HybridCarJourney where
  electricDistance : ℝ
  gasolineConsumptionRate : ℝ
  averageFuelEfficiency : ℝ

/-- Calculates the total distance of a hybrid car journey --/
noncomputable def totalDistance (journey : HybridCarJourney) : ℝ :=
  journey.electricDistance + (journey.averageFuelEfficiency * journey.gasolineConsumptionRate * journey.electricDistance) / (1 - journey.averageFuelEfficiency * journey.gasolineConsumptionRate)

/-- Theorem: The total distance of Alice's journey is 90 miles --/
theorem alice_journey_distance :
  let journey : HybridCarJourney := {
    electricDistance := 30,
    gasolineConsumptionRate := 0.03,
    averageFuelEfficiency := 50
  }
  totalDistance journey = 90 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_alice_journey_distance_l42_4258


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_spherical_coordinates_reflection_l42_4296

/-- Given a point with rectangular coordinates (x, y, z) corresponding to 
    spherical coordinates (4, 3π/7, π/8), prove that the point with rectangular 
    coordinates (-x, y, z) has spherical coordinates (4, 4π/7, π/8). -/
theorem spherical_coordinates_reflection (x y z : ℝ) :
  let ρ : ℝ := 4
  let θ : ℝ := 3 * π / 7
  let φ : ℝ := π / 8
  (x = ρ * Real.sin φ * Real.cos θ) ∧ 
  (y = ρ * Real.sin φ * Real.sin θ) ∧ 
  (z = ρ * Real.cos φ) →
  ∃ (ρ' θ' φ' : ℝ), 
    (ρ' = 4) ∧ 
    (θ' = 4 * π / 7) ∧ 
    (φ' = π / 8) ∧
    (-x = ρ' * Real.sin φ' * Real.cos θ') ∧ 
    (y = ρ' * Real.sin φ' * Real.sin θ') ∧ 
    (z = ρ' * Real.cos φ') ∧
    (ρ' > 0) ∧ 
    (0 ≤ θ' ∧ θ' < 2 * π) ∧ 
    (0 ≤ φ' ∧ φ' ≤ π) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_spherical_coordinates_reflection_l42_4296


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_field_perimeter_l42_4259

/-- Calculates the perimeter of a rectangle given its area and length. -/
noncomputable def rectangle_perimeter (area : ℝ) (length : ℝ) : ℝ :=
  let width := area / length
  2 * (length + width)

/-- Theorem: A rectangular field with an area of 300 square meters and a length of 15 meters has a perimeter of 70 meters. -/
theorem rectangle_field_perimeter :
  rectangle_perimeter 300 15 = 70 := by
  -- Unfold the definition of rectangle_perimeter
  unfold rectangle_perimeter
  -- Simplify the expression
  simp
  -- The proof is complete
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_field_perimeter_l42_4259


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_values_l42_4233

theorem triangle_side_values (a b c : ℝ) (A B C : ℝ) : 
  -- Triangle ABC exists
  (0 < a) → (0 < b) → (0 < c) →
  -- a, b, c are sides opposite to angles A, B, C
  (a^2 = b^2 + c^2 - 2*b*c*Real.cos A) →
  (b^2 = a^2 + c^2 - 2*a*c*Real.cos B) →
  (c^2 = a^2 + b^2 - 2*a*b*Real.cos C) →
  -- Given conditions
  (a*Real.sin A - b*Real.sin B = c*Real.sin C - b*Real.sin C) →
  (b + c = 4) →
  -- Conclusion
  (a = 2 ∨ a = 3) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_values_l42_4233


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monic_polynomial_cube_sum_l42_4281

open Polynomial

theorem monic_polynomial_cube_sum {R : Type*} [Field R] (P Q : R[X]) : 
  Monic P → Monic Q → P^3 + Q^3 = X^12 + 1 → 
  ((P = 1 ∧ Q = X^4) ∨ (P = X^4 ∧ Q = 1)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_monic_polynomial_cube_sum_l42_4281


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_range_with_ten_perfect_squares_l42_4220

-- Define is_perfect_square as a computable function
def is_perfect_square (n : ℕ) : Bool :=
  match Nat.sqrt n with
  | m => m * m = n

theorem smallest_range_with_ten_perfect_squares : 
  ∃! n : ℕ, (n ≥ 100 ∧ (∀ m : ℕ, m < n → (Finset.filter (fun x => is_perfect_square x) (Finset.range m)).card < 10)) ∧
             (Finset.filter (fun x => is_perfect_square x) (Finset.range (n + 1))).card = 10 :=
by
  -- The proof goes here
  sorry

#eval (Finset.filter (fun x => is_perfect_square x) (Finset.range 101)).card

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_range_with_ten_perfect_squares_l42_4220


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_half_equator_latitude_l42_4214

theorem half_equator_latitude (R : ℝ) (φ : ℝ) : 
  (2 * π * (R * Real.cos φ) = π * R) ↔ (φ = π / 3 ∨ φ = -π / 3) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_half_equator_latitude_l42_4214


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_distance_l42_4295

noncomputable section

/-- The parabola y = x^2 - 4x + 10 -/
def parabola (x : ℝ) : ℝ := x^2 - 4*x + 10

/-- The line y = 3x - 5 -/
def line (x : ℝ) : ℝ := 3*x - 5

/-- The distance between a point (x, parabola x) and the line -/
def distance (x : ℝ) : ℝ := 
  |3*x - (parabola x) - 5| / Real.sqrt 10

theorem shortest_distance : 
  ∃ (min_dist : ℝ), min_dist = Real.sqrt 10 / 20 ∧ 
    ∀ (x : ℝ), distance x ≥ min_dist := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_distance_l42_4295


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_round_table_seating_arrangements_l42_4269

def number_of_people : ℕ := 10

theorem round_table_seating_arrangements :
  Nat.factorial (number_of_people - 1) = 362880 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_round_table_seating_arrangements_l42_4269


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_function_of_unit_circle_mean_value_property_l42_4253

open Set
open Metric
open MeasureTheory
open Topology
open ContinuousMap
open Real

/-- A function satisfying the mean value property on unit circles -/
noncomputable def HasUnitCircleMeanValueProperty (f : ℝ × ℝ → ℝ) : Prop :=
  ∀ x : ℝ × ℝ, (∫ θ in (0:ℝ)..2*π, f (x.1 + Real.cos θ, x.2 + Real.sin θ)) / (2*π) = f x

/-- The main theorem: if a continuous function f: ℝ² → [0,1] satisfies the mean value property on unit circles, then it is constant -/
theorem constant_function_of_unit_circle_mean_value_property
  (f : ℝ × ℝ → ℝ)
  (hf_continuous : Continuous f)
  (hf_range : ∀ x, 0 ≤ f x ∧ f x ≤ 1)
  (hf_mean_value : HasUnitCircleMeanValueProperty f) :
  ∃ c, ∀ x, f x = c := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_function_of_unit_circle_mean_value_property_l42_4253


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_work_time_solution_l42_4271

/-- Represents the time (in days) it takes for a worker to complete the job alone -/
structure WorkTime where
  days : ℚ
  days_pos : days > 0

/-- Represents the rate at which a worker completes the job per day -/
def work_rate (w : WorkTime) : ℚ := 1 / w.days

theorem work_time_solution (a b c : WorkTime)
  (ha : a.days = 4)
  (hc : c.days = 12)
  (habc : work_rate a + work_rate b + work_rate c = 1 / 2) :
  b.days = 6 := by
  sorry

#check work_time_solution

end NUMINAMATH_CALUDE_ERRORFEEDBACK_work_time_solution_l42_4271


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_y_coordinate_range_l42_4264

noncomputable section

-- Define the ellipse C
def ellipse_C (x y : ℝ) : Prop := x^2 / 4 + y^2 / 3 = 1

-- Define the right focus F
def right_focus : ℝ × ℝ := (1, 0)

-- Define a line passing through the right focus
def line_through_focus (k : ℝ) (x y : ℝ) : Prop := y = k * (x - 1)

-- Define the intersection points M and N
def intersection_points (k : ℝ) : Set (ℝ × ℝ) :=
  {p | ∃ x y, p = (x, y) ∧ ellipse_C x y ∧ line_through_focus k x y}

-- Define the perpendicular bisector of MN
def perpendicular_bisector (k : ℝ) (x y : ℝ) : Prop :=
  y + (3*k)/(3+4*k^2) = -(1/k) * (x - (4*k^2)/(3+4*k^2))

-- Define the y-coordinate of point P
noncomputable def y_coordinate (k : ℝ) : ℝ := k / (3 + 4*k^2)

-- Theorem statement
theorem y_coordinate_range :
  ∀ k : ℝ, k ≠ 0 → -Real.sqrt 3 / 12 ≤ y_coordinate k ∧ y_coordinate k ≤ Real.sqrt 3 / 12 :=
by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_y_coordinate_range_l42_4264


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_terms_a_10_and_a_11_l42_4257

noncomputable def a (n : ℕ+) : ℝ := (2/3)^(n.val - 1) * (n.val - 8)

theorem max_terms_a_10_and_a_11 :
  ∀ k : ℕ+, (k ≠ 10 ∧ k ≠ 11) → a k ≤ a 10 ∧ a k ≤ a 11 :=
by
  sorry

#check max_terms_a_10_and_a_11

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_terms_a_10_and_a_11_l42_4257


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_elective_subjects_probability_l42_4239

/-- The number of subjects in the first group (Physics, History) -/
def n₁ : ℕ := 2

/-- The number of subjects in the second group (Chemistry, Biology, Geography, Politics) -/
def n₂ : ℕ := 4

/-- The number of subjects to be chosen from the first group -/
def k₁ : ℕ := 1

/-- The number of subjects to be chosen from the second group -/
def k₂ : ℕ := 2

/-- The total number of ways two students can choose their elective subjects -/
def total_combinations : ℕ := (n₁.choose k₁ * n₂.choose k₂) ^ 2

/-- The number of ways two students can have exactly two subjects in common -/
def common_two_subjects : ℕ := 
  n₂.choose 2 * n₁.choose 2 + n₂.choose 1 * (n₂ - 1).choose 2 * n₁.choose 1

theorem elective_subjects_probability : 
  (common_two_subjects : ℚ) / total_combinations = 5 / 12 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_elective_subjects_probability_l42_4239


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_m_range_of_t_l42_4241

-- Define the function f(x) as noncomputable
noncomputable def f (x : ℝ) : ℝ := (1 + Real.log x) / x

-- Theorem for the range of m
theorem range_of_m :
  ∀ m : ℝ, m > 0 →
  (∃ x₁ x₂ : ℝ, m < x₁ ∧ x₂ < m + 1/2 ∧ 
   (∀ x : ℝ, m < x ∧ x < m + 1/2 → f x ≤ f x₁ ∧ f x ≤ f x₂)) ↔
  1/2 < m ∧ m < 1 :=
by
  sorry

-- Theorem for the range of t
theorem range_of_t :
  ∀ t : ℝ, (∀ x : ℝ, x ≥ 1 → f x ≥ t / (x + 1)) ↔ t ≤ 2 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_m_range_of_t_l42_4241


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_simplify_P_l42_4256

-- Define P(x, y)
noncomputable def P (x y : ℝ) : ℝ := (x^2 - y^2) / (x * y) - (x * y - y^2) / (x * y - x^2)

-- Theorem statement
theorem simplify_P (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hxy : x ≠ y) :
  P x y = x / y := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_simplify_P_l42_4256


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_factorial_square_root_squared_l42_4278

theorem factorial_square_root_squared : (Real.sqrt (Nat.factorial 5 * Nat.factorial 4))^2 = 2880 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_factorial_square_root_squared_l42_4278


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_import_tax_is_seven_percent_l42_4217

/-- The import tax percentage for a given item -/
noncomputable def import_tax_percentage (total_value : ℝ) (tax_paid : ℝ) : ℝ :=
  let taxable_amount := total_value - 1000
  (tax_paid / taxable_amount) * 100

/-- Theorem: The import tax percentage is 7% given the problem conditions -/
theorem import_tax_is_seven_percent :
  -- Unfold the definition of import_tax_percentage
  unfold import_tax_percentage
  -- Simplify the expression
  simp
  -- The proof steps would go here, but we'll use sorry for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_import_tax_is_seven_percent_l42_4217


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_value_bound_l42_4287

/-- For any polynomial of degree n, there exists a value in [1, n+1] where the absolute value of the polynomial is at least n!/(2^n) -/
theorem polynomial_value_bound (n : ℕ) (f : Polynomial ℝ) : 
  Polynomial.degree f = n → ∃ k : ℕ, k ∈ Finset.range (n + 2) \ {0} ∧ |f.eval (k : ℝ)| ≥ (n.factorial : ℝ) / 2^n := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_value_bound_l42_4287


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_product_l42_4263

/-- The curve C in the xy-plane -/
def C (x y : ℝ) : Prop := (x - 2)^2 + (y - 1)^2 = 5

/-- The line passing through P(1,2) with inclination angle π/6 -/
noncomputable def line (t : ℝ) : ℝ × ℝ := (1 + (Real.sqrt 3 / 2) * t, 2 + (1 / 2) * t)

/-- Point P -/
def P : ℝ × ℝ := (1, 2)

/-- The theorem stating that the product of distances from P to intersection points is 3 -/
theorem intersection_distance_product : 
  ∃ t₁ t₂ : ℝ, 
    C (line t₁).1 (line t₁).2 ∧ 
    C (line t₂).1 (line t₂).2 ∧ 
    t₁ ≠ t₂ ∧
    (((line t₁).1 - P.1)^2 + ((line t₁).2 - P.2)^2) *
    (((line t₂).1 - P.1)^2 + ((line t₂).2 - P.2)^2) = 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_product_l42_4263


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_nine_pointed_star_sum_tip_angles_l42_4274

/-- A star formed by connecting evenly spaced points on a circle -/
structure StarPolygon where
  n : ℕ  -- number of points
  -- Other properties of the star can be added here if needed

/-- The sum of tip angles in a star polygon -/
def sumTipAngles (s : StarPolygon) : ℝ := sorry

/-- Theorem: The sum of tip angles in a 9-pointed star is 540° -/
theorem nine_pointed_star_sum_tip_angles :
  ∀ (s : StarPolygon), s.n = 9 → sumTipAngles s = 540 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_nine_pointed_star_sum_tip_angles_l42_4274


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_cbf_is_15_degrees_l42_4261

/-- An isosceles right triangle with additional points on one side -/
structure IsoscelesRightTriangleWithPoints where
  -- The triangle
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  -- The additional points
  E : ℝ × ℝ
  F : ℝ × ℝ
  -- Triangle ABC is isosceles right with right angle at A
  isIsoscelesRight : (A.1 = B.1 ∧ A.2 = C.2) ∧ (B.1 - A.1)^2 + (B.2 - A.2)^2 = (C.1 - A.1)^2 + (C.2 - A.2)^2
  -- E and F are on ray AC
  EFOnAC : ∃ t s : ℝ, t ≥ 0 ∧ s ≥ 0 ∧ 
    E = (A.1 + t * (C.1 - A.1), A.2 + t * (C.2 - A.2)) ∧
    F = (A.1 + s * (C.1 - A.1), A.2 + s * (C.2 - A.2))
  -- Angle ABE is 15 degrees
  angleABE : Real.cos (15 * π / 180) = 
    ((E.1 - A.1) * (B.1 - A.1) + (E.2 - A.2) * (B.2 - A.2)) / 
    (((E.1 - A.1)^2 + (E.2 - A.2)^2)^(1/2) * ((B.1 - A.1)^2 + (B.2 - A.2)^2)^(1/2))
  -- CE = CF
  CEEqualsCF : (E.1 - C.1)^2 + (E.2 - C.2)^2 = (F.1 - C.1)^2 + (F.2 - C.2)^2

/-- The theorem to be proved -/
theorem angle_cbf_is_15_degrees (t : IsoscelesRightTriangleWithPoints) : 
  Real.cos (15 * π / 180) = 
    ((t.F.1 - t.B.1) * (t.C.1 - t.B.1) + (t.F.2 - t.B.2) * (t.C.2 - t.B.2)) / 
    (((t.F.1 - t.B.1)^2 + (t.F.2 - t.B.2)^2)^(1/2) * ((t.C.1 - t.B.1)^2 + (t.C.2 - t.B.2)^2)^(1/2)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_cbf_is_15_degrees_l42_4261


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solve_exponential_equation_l42_4298

theorem solve_exponential_equation (x : ℝ) : (100 : ℝ)^3 = 10^x → x = 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solve_exponential_equation_l42_4298


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_eccentricity_bounds_l42_4201

/-- The eccentricity of a hyperbola with equation x²/a² - y²/(a+1)² = 1 where a > 2 -/
noncomputable def eccentricity (a : ℝ) : ℝ := Real.sqrt (1 + (a + 1)^2 / a^2)

/-- Theorem stating the bounds of the eccentricity for a > 2 -/
theorem eccentricity_bounds (a : ℝ) (h : a > 2) :
  Real.sqrt 2 < eccentricity a ∧ eccentricity a < Real.sqrt 13 / 2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_eccentricity_bounds_l42_4201


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l42_4228

noncomputable def f (x : ℝ) : ℝ := Real.sqrt (x - 1) / (x + 1)

def IsValidInput (f : ℝ → ℝ) (x : ℝ) : Prop := ∃ y, f x = y

theorem domain_of_f : 
  {x : ℝ | IsValidInput f x} = {x : ℝ | x ≥ 1} :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l42_4228


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_board_division_l42_4248

/-- A board can be divided into shapes of six unit squares if and only if
    one dimension is divisible by 3 and the other by 4 -/
theorem board_division (m n : ℕ) (hm : m > 5) (hn : n > 5) :
  (∃ (shapes : List (List (ℕ × ℕ))),
    (∀ shape : List (ℕ × ℕ), shape ∈ shapes → shape.length = 6) ∧
    (∀ (i j : ℕ), i < m ∧ j < n → ∃ (shape : List (ℕ × ℕ)), shape ∈ shapes ∧ (i, j) ∈ shape) ∧
    (∀ (shape1 shape2 : List (ℕ × ℕ)), shape1 ∈ shapes → shape2 ∈ shapes → shape1 ≠ shape2 → 
      ∀ (i j : ℕ), (i, j) ∈ shape1 → (i, j) ∉ shape2)) ↔
  ((m % 3 = 0 ∧ n % 4 = 0) ∨ (m % 4 = 0 ∧ n % 3 = 0)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_board_division_l42_4248


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circles_intersect_circles_not_tangent_or_disjoint_l42_4232

-- Define the circles
def circle_M (x y : ℝ) : Prop := x^2 + y^2 - 4*y = 0
def circle_N (x y : ℝ) : Prop := x^2 + y^2 - 2*x - 2*y + 1 = 0

-- Define the center and radius of each circle
def center_M : ℝ × ℝ := (0, 2)
def radius_M : ℝ := 2

def center_N : ℝ × ℝ := (1, 1)
def radius_N : ℝ := 1

-- Define the distance between the centers
noncomputable def distance_centers : ℝ := Real.sqrt 2

-- Theorem stating that the circles are intersecting
theorem circles_intersect :
  distance_centers > abs (radius_M - radius_N) ∧
  distance_centers < radius_M + radius_N := by
  sorry

-- Additional theorem to show the circles are not tangent or disjoint
theorem circles_not_tangent_or_disjoint :
  distance_centers ≠ abs (radius_M - radius_N) ∧
  distance_centers ≠ radius_M + radius_N := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circles_intersect_circles_not_tangent_or_disjoint_l42_4232


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_class_size_calculation_class_size_calculation_proof_l42_4208

theorem class_size_calculation (smoking_percentage : ℝ) 
                               (hospitalized_percentage : ℝ) 
                               (non_hospitalized_count : ℕ) 
                               (class_size : ℕ) : Prop :=
  smoking_percentage = 0.4 ∧
  hospitalized_percentage = 0.7 ∧
  non_hospitalized_count = 36 ∧
  (1 - hospitalized_percentage) * smoking_percentage * (class_size : ℝ) = non_hospitalized_count ∧
  class_size = 300

theorem class_size_calculation_proof : 
  ∃ (smoking_percentage hospitalized_percentage : ℝ) (non_hospitalized_count class_size : ℕ),
    class_size_calculation smoking_percentage hospitalized_percentage non_hospitalized_count class_size :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_class_size_calculation_class_size_calculation_proof_l42_4208


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_limit_a_eq_exp_minus_one_l42_4236

/-- Definition of the sequence a_{i,j} -/
noncomputable def a (x : ℝ) : ℕ → ℕ → ℝ
  | i, 0 => x / (2^i)
  | i, j + 1 => (a x i j)^2 + 2 * (a x i j)

/-- The limit of a_{n,n} as n approaches infinity -/
noncomputable def limit_a (x : ℝ) : ℝ := Real.exp x - 1

/-- Theorem stating that the limit of a_{n,n} is e^x - 1 -/
theorem limit_a_eq_exp_minus_one (x : ℝ) :
  ∀ ε > 0, ∃ N, ∀ n ≥ N, |a x n n - limit_a x| < ε := by
  sorry

#check limit_a_eq_exp_minus_one

end NUMINAMATH_CALUDE_ERRORFEEDBACK_limit_a_eq_exp_minus_one_l42_4236


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_pyramid_volume_l42_4213

/-- Definition of a pyramid with an equilateral triangular base -/
structure PyramidWithEquilateralBase where
  base_side : ℝ
  apex_angle : ℝ
  apex_angles_equal : Prop
  volume : ℝ

/-- The minimum volume of a pyramid with specific conditions -/
theorem min_pyramid_volume :
  let base_side : ℝ := 6
  let max_apex_angle : ℝ := 2 * Real.arcsin (1 / 3)
  ∀ (volume : ℝ),
    (∃ (pyramid : PyramidWithEquilateralBase),
      pyramid.base_side = base_side ∧
      pyramid.apex_angles_equal ∧
      pyramid.apex_angle ≤ max_apex_angle ∧
      pyramid.volume = volume) →
    volume ≥ 5 * Real.sqrt 23 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_pyramid_volume_l42_4213


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_point_E_coordinates_l42_4243

/-- Given points A, B, C, D, and E on a plane, prove that E has specific coordinates -/
theorem point_E_coordinates :
  let A : ℝ × ℝ := (-2, 1)
  let B : ℝ × ℝ := (1, 4)
  let C : ℝ × ℝ := (4, -3)
  let D : ℝ × ℝ := (-1, 2)  -- Calculated from AD/BD = 1/2
  let E : ℝ × ℝ := (17/3, -14/3)
  (D.1 - A.1) / (B.1 - D.1) = 1/2 →  -- AD/BD = 1/2
  (D.2 - A.2) / (B.2 - D.2) = 1/2 →  -- AD/BD = 1/2 (y-coordinate)
  (E.1 - C.1) / (C.1 - D.1) = 4 →    -- |CE|/|ED| = 1/4
  (E.2 - C.2) / (C.2 - D.2) = 4 →    -- |CE|/|ED| = 1/4 (y-coordinate)
  E = (17/3, -14/3) := by
  sorry

#check point_E_coordinates

end NUMINAMATH_CALUDE_ERRORFEEDBACK_point_E_coordinates_l42_4243


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_youseff_commute_time_l42_4222

/-- The number of blocks between Youseff's home and office -/
noncomputable def x : ℚ := 12

/-- Walking time in minutes -/
def walkingTime (blocks : ℚ) : ℚ := blocks

/-- Biking time in minutes -/
def bikingTime (blocks : ℚ) : ℚ := (1/3) * blocks

/-- Theorem stating that x satisfies the given conditions -/
theorem youseff_commute_time :
  walkingTime x = bikingTime x + 8 ∧ x = 12 := by
  constructor
  · -- Prove walkingTime x = bikingTime x + 8
    simp [walkingTime, bikingTime, x]
    norm_num
  · -- Prove x = 12
    rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_youseff_commute_time_l42_4222


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_roots_cube_relation_l42_4209

/-- Given a polynomial h(x) = x³ - 2x² + 3x - 4 with three distinct roots,
    j(x) = x³ - 8x² + 108x - 64 has roots that are the cubes of the roots of h(x). -/
theorem roots_cube_relation (h j : ℝ → ℝ) :
  (h = λ x => x^3 - 2*x^2 + 3*x - 4) →
  (∃ a b c : ℝ, a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ h a = 0 ∧ h b = 0 ∧ h c = 0) →
  (j = λ x => x^3 - 8*x^2 + 108*x - 64) →
  (∀ r : ℝ, h r = 0 → j (r^3) = 0) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_roots_cube_relation_l42_4209


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_lower_bound_l42_4229

/-- An odd function defined on ℝ satisfying given conditions -/
def f (a : ℝ) : ℝ → ℝ := sorry

/-- The function f is odd -/
axiom f_odd (a : ℝ) : ∀ x, f a x = -f a (-x)

/-- Definition of f for negative x -/
axiom f_neg (a : ℝ) : ∀ x, x < 0 → f a x = x + a / x + 7

/-- Lower bound of f for non-negative x -/
axiom f_nonneg_bound (a : ℝ) : ∀ x, x ≥ 0 → f a x ≥ 1 - a

theorem a_lower_bound (a : ℝ) (ha : a > 0) : a ≥ 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_lower_bound_l42_4229


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetric_point_to_origin_l42_4252

/-- 
Given a line with equation x - 2y + 2 = 0, this theorem proves that the point (-4/5, 8/5) 
is symmetric to the origin (0, 0) with respect to this line.
-/
theorem symmetric_point_to_origin (x y : ℝ) : 
  (x - 2*y + 2 = 0) →  -- Line equation
  let symmetric_point := (-4/5, 8/5)
  let origin := (0, 0)
  let midpoint := ((symmetric_point.1 + origin.1)/2, (symmetric_point.2 + origin.2)/2)
  (midpoint.1 - 2*midpoint.2 + 2 = 0) ∧  -- Midpoint lies on the line
  ((symmetric_point.2 - origin.2)/(symmetric_point.1 - origin.1) * (x - y) = -1)  -- Perpendicular slopes
  := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetric_point_to_origin_l42_4252


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_function_point_l42_4247

-- Define the function f
noncomputable def f (b : ℝ) : ℝ → ℝ := λ x ↦ Real.exp (x * Real.log 2) + b

-- Define the inverse function property
def is_inverse (f g : ℝ → ℝ) : Prop :=
  ∀ x, g (f x) = x ∧ f (g x) = x

-- Theorem statement
theorem inverse_function_point (b : ℝ) :
  (∃ f_inv : ℝ → ℝ, is_inverse (f b) f_inv ∧ f_inv 5 = 2) → b = 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_function_point_l42_4247


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_add_base7_example_l42_4207

/-- Represents a number in base 7 --/
structure Base7 where
  value : Nat
  isValid : value < 7^64 := by sorry

/-- Converts a base 7 number to its decimal (base 10) representation --/
def toDecimal (n : Base7) : Nat := sorry

/-- Converts a decimal (base 10) number to its base 7 representation --/
def toBase7 (n : Nat) : Base7 := ⟨n % (7^64), by sorry⟩

/-- Addition operation for base 7 numbers --/
def addBase7 (a b : Base7) : Base7 := toBase7 (toDecimal a + toDecimal b)

/-- Helper function to create a Base7 number from a Nat --/
def mkBase7 (n : Nat) : Base7 := toBase7 n

theorem add_base7_example : addBase7 (mkBase7 10) (mkBase7 163) = mkBase7 203 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_add_base7_example_l42_4207


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angles_proof_l42_4235

/-- Triangle structure with three angles -/
structure Triangle where
  angle1 : ℝ
  angle2 : ℝ
  angle3 : ℝ
  sum_180 : angle1 + angle2 + angle3 = 180

/-- Given a geometric configuration with two triangles as described, prove that α = 120° and β = 85° -/
theorem triangle_angles_proof (small_triangle large_triangle : Triangle) 
  (h1 : small_triangle.angle1 = 70)
  (h2 : small_triangle.angle2 = 50)
  (h3 : large_triangle.angle1 = 45)
  (h4 : large_triangle.angle2 = 50)
  (α β : ℝ)
  (h5 : α = 180 - small_triangle.angle3)
  (h6 : β = large_triangle.angle3) : 
  α = 120 ∧ β = 85 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angles_proof_l42_4235


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_values_l42_4266

theorem sine_values (α β : Real) 
  (h1 : α ∈ Set.Ioo 0 (π/2))
  (h2 : β ∈ Set.Ioo (π/2) π)
  (h3 : Real.cos β = -1/3)
  (h4 : Real.sin (α + β) = 7/9) :
  (Real.sin α = 1/3) ∧ (Real.sin (2*α + β) = 10*Real.sqrt 2/27) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_values_l42_4266


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_necessary_sufficient_condition_l42_4268

theorem necessary_sufficient_condition (a b : ℝ) : a > b ↔ (2 : ℝ)^a > (2 : ℝ)^b := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_necessary_sufficient_condition_l42_4268


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_number_of_proper_subsets_l42_4284

def f (x : ℤ) : ℤ := -x^2 + 6

def S : Finset ℤ := {2, 5, 6}

theorem number_of_proper_subsets : Finset.card (Finset.powerset S \ {S}) = 7 := by
  -- Proof goes here
  sorry

#eval Finset.card (Finset.powerset S \ {S})

end NUMINAMATH_CALUDE_ERRORFEEDBACK_number_of_proper_subsets_l42_4284


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l42_4279

noncomputable def f (x : ℝ) : ℝ := 2^x / (4^x + 1)

theorem f_properties :
  let log_sqrt2_3 := Real.log 3 / Real.log (Real.sqrt 2)
  (f log_sqrt2_3 = 9/82) ∧
  (∀ x₁ x₂ : ℝ, 0 < x₁ → x₁ < x₂ → f x₁ > f x₂) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l42_4279


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_billiard_return_theorem_l42_4205

/-- Represents a point in a 2D plane -/
structure Point where
  x : ℚ
  y : ℚ

/-- Represents a rectangle -/
structure Rectangle where
  width : ℕ
  height : ℕ

/-- Represents the path of a billiard ball -/
def BilliardPath (rect : Rectangle) (start : Point) : ℕ → Point → Prop :=
  sorry

/-- The number of bounces before returning to the starting point -/
def NumBounces (rect : Rectangle) (start : Point) : ℕ :=
  sorry

/-- Theorem stating that the ball returns to its starting point after 294 bounces -/
theorem billiard_return_theorem (rect : Rectangle) (start : Point) :
  rect.width = 2021 → rect.height = 4300 →
  BilliardPath rect start (NumBounces rect start) start ∧
  NumBounces rect start = 294 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_billiard_return_theorem_l42_4205


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_team_transfer_equation_l42_4221

/-- Represents the number of members in the traffic safety team -/
def traffic_team : ℕ := 8

/-- Represents the number of members in the fire safety team -/
def fire_team : ℕ := 7

/-- Theorem stating the correct equation for the team transfer scenario -/
theorem team_transfer_equation (x : ℤ) : 
  (traffic_team : ℤ) + x = 2 * ((fire_team : ℤ) - x) ↔ 
  (traffic_team : ℤ) + x = 2 * (fire_team : ℤ) - 2 * x :=
by
  -- The proof goes here
  sorry

#check team_transfer_equation

end NUMINAMATH_CALUDE_ERRORFEEDBACK_team_transfer_equation_l42_4221


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_n_for_M_not_exceeding_2022_l42_4219

def S (n : ℕ) := (n^2 + 3*n) / 2

def b (n : ℕ) : ℕ := 2^(n-1)

def T (n : ℕ) := 2*b n - 1

def a (n : ℕ) : ℕ := n + 1

def M (n : ℕ) : ℕ := n + 2^n - 1

theorem largest_n_for_M_not_exceeding_2022 :
  (∀ n : ℕ, n ≤ 10 → M n ≤ 2022) ∧ M 11 > 2022 :=
by
  constructor
  · intro n hn
    sorry -- Proof that M n ≤ 2022 for n ≤ 10
  · sorry -- Proof that M 11 > 2022

#eval M 10
#eval M 11

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_n_for_M_not_exceeding_2022_l42_4219


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_slant_height_angle_formula_l42_4276

/-- A cone with an inscribed cube where one face of the cube lies in the base plane of the cone -/
structure InscribedCubeCone where
  /-- The ratio of the cone's height to the cube's edge -/
  k : ℝ
  /-- k is positive -/
  k_pos : k > 0

/-- The angle between the slant height and the height of the cone -/
noncomputable def slant_height_angle (c : InscribedCubeCone) : ℝ :=
  Real.arctan (1 / (Real.sqrt 2 * (c.k - 1)))

/-- Theorem stating the relationship between the cone's dimensions and the angle -/
theorem slant_height_angle_formula (c : InscribedCubeCone) :
  slant_height_angle c = Real.arctan (1 / (Real.sqrt 2 * (c.k - 1))) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_slant_height_angle_formula_l42_4276


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_oblique_projection_length_preservation_l42_4230

/-- Represents a line segment in 3D space -/
structure LineSegment where
  start : ℝ × ℝ × ℝ
  end_ : ℝ × ℝ × ℝ

/-- Represents an oblique projection -/
structure ObliqueProjection where
  angle : ℝ  -- Angle of obliquity
  scale : ℝ  -- Scale factor

/-- Returns true if the line segment is perpendicular to the coordinate axes -/
def isPerpendicular (l : LineSegment) : Prop :=
  sorry

/-- Returns the length of a line segment -/
noncomputable def length (l : LineSegment) : ℝ :=
  sorry

/-- Projects a line segment using oblique projection -/
def project (o : ObliqueProjection) (l : LineSegment) : LineSegment :=
  sorry

/-- Theorem: In oblique projection, there exists a line segment not perpendicular 
    to the coordinate axes whose projected length remains the same as its true length -/
theorem oblique_projection_length_preservation :
  ∃ (o : ObliqueProjection) (l : LineSegment), 
    ¬isPerpendicular l ∧ length (project o l) = length l := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_oblique_projection_length_preservation_l42_4230


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_term_eq_160_l42_4251

/-- The constant term in the expansion of (1/x + 2x)^6 -/
noncomputable def constant_term (x : ℝ) : ℝ :=
  (Finset.range 7).sum (λ k => Nat.choose 6 k * (1/x)^(6-k) * (2*x)^k)

/-- Theorem: The constant term in the expansion of (1/x + 2x)^6 is equal to 160 -/
theorem constant_term_eq_160 :
  ∀ x : ℝ, x ≠ 0 → constant_term x = 160 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_term_eq_160_l42_4251


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_for_negative_one_range_of_a_for_nonnegative_f_l42_4212

-- Define the function f
def f (a x : ℝ) : ℝ := |x - a| + 5 * x

-- Part I
theorem solution_set_for_negative_one (x : ℝ) :
  f (-1) x ≤ 5 * x + 3 ↔ x ∈ Set.Icc (-4) 2 :=
sorry

-- Part II
theorem range_of_a_for_nonnegative_f :
  {a : ℝ | ∀ x ≥ -1, f a x ≥ 0} = Set.Ioi 4 ∪ Set.Iic (-6) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_for_negative_one_range_of_a_for_nonnegative_f_l42_4212


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_point_on_line_with_sum_of_distances_l42_4283

-- Define the necessary types and structures
structure Point where
  x : ℝ
  y : ℝ

structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

-- Define the distance function between two points
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

-- Define a function to check if a point is on a line
def isOnLine (p : Point) (l : Line) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

-- Define a function to check if two points are on the same side of a line
def sameSideOfLine (p1 p2 : Point) (l : Line) : Prop :=
  (l.a * p1.x + l.b * p1.y + l.c) * (l.a * p2.x + l.b * p2.y + l.c) > 0

-- State the theorem
theorem point_on_line_with_sum_of_distances
  (l : Line) (A B : Point) (a : ℝ)
  (h1 : sameSideOfLine A B l)
  (h2 : a > 0) :
  ∃ X : Point, isOnLine X l ∧ distance A X + distance B X = a := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_point_on_line_with_sum_of_distances_l42_4283


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_squared_minus_two_sin_zero_solution_set_l42_4244

theorem sin_squared_minus_two_sin_zero_solution_set :
  {x : ℝ | Real.sin x ^ 2 - 2 * Real.sin x = 0} = {x : ℝ | ∃ k : ℤ, x = k * Real.pi} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_squared_minus_two_sin_zero_solution_set_l42_4244


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_product_l42_4216

-- Define the line l
noncomputable def line_l (t : ℝ) : ℝ × ℝ := (5 + (Real.sqrt 3 / 2) * t, Real.sqrt 3 + (1 / 2) * t)

-- Define the curve C in rectangular coordinates
def curve_C (x y : ℝ) : Prop := x^2 + y^2 - 2*x = 0

-- Define point M
noncomputable def point_M : ℝ × ℝ := (5, Real.sqrt 3)

-- Theorem statement
theorem intersection_distance_product :
  ∃ (t₁ t₂ : ℝ),
    curve_C (line_l t₁).1 (line_l t₁).2 ∧
    curve_C (line_l t₂).1 (line_l t₂).2 ∧
    t₁ ≠ t₂ ∧
    (t₁ - 0) * (t₂ - 0) = 18 := by
  sorry

#check intersection_distance_product

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_product_l42_4216


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_undefined_sum_is_three_l42_4262

noncomputable def f (x : ℝ) : ℝ := 5 * x / (3 * x^2 - 9 * x + 6)

theorem undefined_sum_is_three :
  ∃ (a b : ℝ), a ≠ b ∧ 
  (∀ x : ℝ, (3 * x^2 - 9 * x + 6 = 0) → x = a ∨ x = b) ∧
  a + b = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_undefined_sum_is_three_l42_4262


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_13_parts_squared_difference_l42_4223

theorem sqrt_13_parts_squared_difference (m n : ℝ) : 
  (m = ⌊Real.sqrt 13⌋) → 
  (n = Real.sqrt 13 - ⌊Real.sqrt 13⌋) → 
  (m - n)^2 = 49 - 12 * Real.sqrt 13 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_13_parts_squared_difference_l42_4223


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_series_sum_l42_4225

theorem geometric_series_sum : 
  let a : ℚ := 1/4
  let r : ℚ := -1/4
  let n : ℕ := 6
  let series := (Finset.range n).sum (λ i => a * r^i)
  series = 4095/81920 := by
  -- Proof goes here
  sorry

#eval (Finset.range 6).sum (λ i => (1/4) * (-1/4)^i)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_series_sum_l42_4225


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rounding_0_055049_l42_4286

noncomputable def round_to_nearest (x : ℝ) (precision : ℝ) : ℝ :=
  ⌊(x / precision + 0.5)⌋ * precision

theorem rounding_0_055049 :
  let x := 0.055049
  (round_to_nearest x 0.1 = 0.1) ∧
  (round_to_nearest x 0.01 = 0.06) ∧
  (round_to_nearest x 0.001 = 0.055) ∧
  (round_to_nearest x 0.0001 ≠ 0.055) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rounding_0_055049_l42_4286


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_progression_complete_iff_divides_l42_4267

/-- A sequence is complete if it contains only non-zero natural terms and 
    for any nonzero integer, there exists a term in the sequence that is divisible by it. -/
def IsCompleteSeq (s : ℕ → ℤ) : Prop :=
  (∀ n, s n ≠ 0) ∧ (∀ m : ℤ, m ≠ 0 → ∃ n k : ℕ, s n = m * k)

/-- An arithmetic progression with first term a and common difference r. -/
def ArithmeticProgression (a r : ℤ) (n : ℕ) : ℤ :=
  a + (n - 1) * r

theorem arithmetic_progression_complete_iff_divides (a r : ℤ) (hr : r ≠ 0) :
  IsCompleteSeq (ArithmeticProgression a r) ↔ r ∣ a := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_progression_complete_iff_divides_l42_4267


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ball_never_returns_to_start_l42_4255

-- Define a polygon type
structure Polygon where
  angles : List Int
  has_one_degree_angle : 1 ∈ angles

-- Define a ball
structure Ball where
  position : ℝ × ℝ
  direction : ℝ

-- Define the reflection function (simplified)
def reflect (b : Ball) (side : ℝ × ℝ) : Ball :=
  sorry

-- Define the theorem
theorem ball_never_returns_to_start (p : Polygon) (initial_ball : Ball) (side : ℝ × ℝ) :
  ∀ n : ℕ, (Nat.iterate (λ b => reflect b side) n initial_ball).position ≠ initial_ball.position :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ball_never_returns_to_start_l42_4255


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_effective_profit_percentage_l42_4245

theorem effective_profit_percentage (SP : ℝ) (SP_pos : SP > 0) : 
  let CP := 0.92 * SP
  let total_received := SP + 0.08 * SP
  let effective_profit := total_received - CP
  let effective_profit_percentage := (effective_profit / CP) * 100
  ∃ ε > 0, |effective_profit_percentage - 17.39| < ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_effective_profit_percentage_l42_4245


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_laura_running_speed_l42_4240

/-- Represents Laura's workout parameters -/
structure WorkoutParams where
  walkTime : ℝ
  walkSpeed : ℝ
  bikeDistance : ℝ
  bikeSpeed : ℝ → ℝ
  transitionTime : ℝ
  runDistance : ℝ
  totalTime : ℝ

/-- Laura's workout satisfies the given conditions -/
noncomputable def lauraWorkout : WorkoutParams where
  walkTime := 15 / 60  -- 15 minutes in hours
  walkSpeed := 3
  bikeDistance := 30
  bikeSpeed := fun x => 3 * x + 2
  transitionTime := 10 / 60  -- 10 minutes in hours
  runDistance := 10
  totalTime := 205 / 60  -- 205 minutes in hours

/-- Theorem stating that Laura's running speed satisfies the quadratic equation -/
theorem laura_running_speed (x : ℝ) :
  let w := lauraWorkout
  let bikeTime := w.bikeDistance / (w.bikeSpeed x)
  let runTime := w.runDistance / x
  w.walkTime + bikeTime + w.transitionTime + runTime = w.totalTime →
  9 * x^2 - 54 * x - 20 = 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_laura_running_speed_l42_4240


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_semicircles_area_ratio_l42_4280

theorem semicircles_area_ratio (r : ℝ) (hr : r > 0) : 
  (((1/2) * π * r^2 + (1/2) * π * (2*r)^2) / (π * (2*r)^2)) = 5/8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_semicircles_area_ratio_l42_4280


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_number_of_students_l42_4285

theorem number_of_students (n : ℝ) : 
  let student_avg : ℝ := 21
  let teacher_age : ℝ := 44
  let new_avg : ℝ := 22
  (student_avg * n + teacher_age) / (n + 1) = new_avg →
  n = 22 := by
  intros
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_number_of_students_l42_4285


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_origin_l42_4224

noncomputable def TangentLine (f : ℝ → ℝ) (a : ℝ) : ℝ → ℝ :=
  fun x ↦ (deriv f a) * (x - a) + f a

theorem tangent_line_at_origin (a : ℝ) (f : ℝ → ℝ) (f' : ℝ → ℝ) :
  (∀ x, f x = x^3 + a*x^2 + (a-3)*x) →
  (∀ x, HasDerivAt f (f' x) x) →
  (∀ x, f' (-x) = f' x) →
  (∃ m b, ∀ x, TangentLine f 0 x = m * x + b) ∧
  TangentLine f 0 = (fun x ↦ -3 * x) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_origin_l42_4224


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_one_extreme_values_range_of_c_minus_a_l42_4294

-- Define the function f(x)
def f (a b c x : ℝ) : ℝ := (x - a) * (x - b) * (x - c)

-- Theorem for part 1
theorem tangent_line_at_one (a b c : ℝ) (h1 : 2 * b = a + c) (h2 : b = 0) (h3 : c = 1) :
  ∃ m k : ℝ, m = 2 ∧ k = -2 ∧ 
  (∀ x y : ℝ, y = m * x + k ↔ y = (deriv (f a b c)) 1 * (x - 1) + f a b c 1) :=
sorry

-- Theorem for part 2
theorem extreme_values (a b c : ℝ) (h1 : 2 * b = a + c) (h2 : b - a = 3) :
  ∃ max_val min_val : ℝ, max_val = 6 * Real.sqrt 3 ∧ min_val = -6 * Real.sqrt 3 ∧
  (∀ x : ℝ, f a b c x ≤ max_val ∧ f a b c x ≥ min_val) ∧
  (∃ x1 x2 : ℝ, f a b c x1 = max_val ∧ f a b c x2 = min_val) :=
sorry

-- Theorem for part 3
theorem range_of_c_minus_a (a b c : ℝ) (h1 : 2 * b = a + c) 
  (h2 : ∃ x1 x2 x3 : ℝ, x1 ≠ x2 ∧ x1 ≠ x3 ∧ x2 ≠ x3 ∧
    f a b c x1 = -(x1 - b) - 2 ∧
    f a b c x2 = -(x2 - b) - 2 ∧
    f a b c x3 = -(x3 - b) - 2) :
  c - a < -4 ∨ c - a > 4 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_one_extreme_values_range_of_c_minus_a_l42_4294
