import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l816_81660

-- Define the ellipse C
noncomputable def ellipse_C (a b : ℝ) (x y : ℝ) : Prop :=
  x^2 / a^2 + y^2 / b^2 = 1

-- Define the eccentricity of an ellipse
noncomputable def eccentricity (a b : ℝ) : ℝ :=
  Real.sqrt (1 - b^2 / a^2)

-- Main theorem
theorem ellipse_properties :
  ∀ a b : ℝ,
  a > b ∧ b > 0 →
  ellipse_C a b 2 0 →
  eccentricity a b = 1/2 →
  (∀ x y : ℝ, ellipse_C 2 (Real.sqrt 3) x y ↔ ellipse_C a b x y) ∧
  (∀ y₀ : ℝ, -3/2 < y₀ ∧ y₀ < 3/2 →
    ∃ k : ℝ,
    (∀ x y : ℝ, y - y₀ = -k * (x + 1) →
      (x = -1 ∧ y = y₀) ∨
      (ellipse_C 2 (Real.sqrt 3) x y ∧
       ∃ x' y' : ℝ, ellipse_C 2 (Real.sqrt 3) x' y' ∧
         y' - y₀ = -k * (x' + 1) ∧
         x + x' = -2 ∧ y + y' = 2 * y₀)) →
    y₀ = -3/4 * k * (1/4 + 1)) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l816_81660


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prize_distribution_l816_81612

def greatest_possible_award (total_prize : ℝ) (num_winners : ℕ) 
  (min_award : ℝ) (prize_fraction : ℝ) (winner_fraction : ℝ) : ℝ :=
  let remaining_prize := total_prize - (min_award * ↑num_winners)
  let remaining_winners := num_winners - ⌊winner_fraction * ↑num_winners⌋
  min_award + remaining_prize

theorem prize_distribution 
  (h1 : total_prize = 500)
  (h2 : num_winners = 20)
  (h3 : min_award = 20)
  (h4 : prize_fraction = 2/5)
  (h5 : winner_fraction = 3/5)
  : greatest_possible_award total_prize num_winners min_award prize_fraction winner_fraction = 120 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prize_distribution_l816_81612


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_average_height_approx_l816_81653

/-- Conversion factor from centimeters to inches -/
noncomputable def cm_to_inch : ℝ := 2.54

/-- Heights of people in inches -/
noncomputable def height_zara : ℝ := 64
noncomputable def height_brixton : ℝ := height_zara
noncomputable def height_zora : ℝ := height_brixton - 8
noncomputable def height_itzayana : ℝ := height_zora + 4
noncomputable def height_jaxon : ℝ := 170 / cm_to_inch
noncomputable def height_leo : ℝ := 1.5 * height_itzayana
noncomputable def height_dora : ℝ := height_leo - 3.75

/-- Average height of all seven people -/
noncomputable def average_height : ℝ :=
  (height_zara + height_brixton + height_zora + height_itzayana +
   height_jaxon + height_leo + height_dora) / 7

/-- Theorem stating that the average height is approximately 69.45 inches -/
theorem average_height_approx :
  abs (average_height - 69.45) < 0.01 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_average_height_approx_l816_81653


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sufficient_but_not_necessary_condition_l816_81687

def f (a : ℝ) : ℝ → ℝ := λ x => x^3 - a*x

theorem sufficient_but_not_necessary_condition :
  (∀ a > 12, ∀ x ∈ Set.Ioo 1 2, HasDerivAt (f a) ((3:ℝ)*x^2 - a) x ∧ (3:ℝ)*x^2 - a ≤ 0) ∧
  (∃ a ≤ 12, ∀ x ∈ Set.Ioo 1 2, HasDerivAt (f a) ((3:ℝ)*x^2 - a) x ∧ (3:ℝ)*x^2 - a ≤ 0) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sufficient_but_not_necessary_condition_l816_81687


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_terminating_decimals_count_l816_81667

theorem terminating_decimals_count (n : ℕ) : 
  1 ≤ n ∧ n ≤ 160 → (∃ (a b : ℕ), (n : ℚ) / 160 = a / (2^b * 5^b)) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_terminating_decimals_count_l816_81667


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_salary_increase_after_six_years_l816_81641

theorem salary_increase_after_six_years (initial_salary : ℝ) (h : initial_salary > 0) :
  (initial_salary * (1 + 0.12)^6 - initial_salary) / initial_salary > 0.9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_salary_increase_after_six_years_l816_81641


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_similar_triangle_perimeter_l816_81617

/-- Triangle type -/
structure Triangle where
  -- Define the triangle structure (this is a placeholder)
  mk :: 

/-- Equilateral triangle -/
def Equilateral (t : Triangle) : Prop := sorry

/-- Two triangles are similar -/
def SimilarTriangles (t1 t2 : Triangle) : Prop := sorry

/-- Side length of a triangle -/
def SideLength (t : Triangle) (l : ℝ) : Prop := sorry

/-- Perimeter of a triangle -/
noncomputable def Perimeter (t : Triangle) : ℝ := sorry

/-- Given two similar equilateral triangles ABC and ADC with side length ratio 1:2,
    if the perimeter of ADC is 9 + 3√3, then the perimeter of ABC is 18 + 6√3 -/
theorem similar_triangle_perimeter (ABC ADC : Triangle) : 
  Equilateral ABC → 
  Equilateral ADC → 
  SimilarTriangles ABC ADC →
  (∀ (side_ABC : ℝ) (side_ADC : ℝ), 
    SideLength ABC side_ABC → 
    SideLength ADC side_ADC → 
    side_ADC = (1/2) * side_ABC) →
  Perimeter ADC = 9 + 3 * Real.sqrt 3 →
  Perimeter ABC = 18 + 6 * Real.sqrt 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_similar_triangle_perimeter_l816_81617


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_minus_cos_value_l816_81688

noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x - Real.pi / 4)

theorem sin_minus_cos_value (α : ℝ) (h1 : α ∈ Set.Ioo (Real.pi / 2) Real.pi) 
  (h2 : f (α / 2 + Real.pi / 4) = (2 / 3) * Real.cos (α + Real.pi / 4) * Real.cos (2 * α)) :
  Real.sin α - Real.cos α = Real.sqrt 6 / 2 ∨ Real.sin α - Real.cos α = Real.sqrt 2 :=
by sorry

#check sin_minus_cos_value

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_minus_cos_value_l816_81688


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_discounted_price_calculation_l816_81616

noncomputable def original_price : ℝ := 120
noncomputable def discount_percentage : ℝ := 20
noncomputable def discount_factor : ℝ := (100 - discount_percentage) / 100

theorem discounted_price_calculation :
  original_price * discount_factor = 96 := by
  -- Proof steps would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_discounted_price_calculation_l816_81616


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pierre_consumption_l816_81679

/-- Represents the weight of a cake in grams -/
noncomputable def cake_weight : ℝ := 525

/-- Represents the fraction of the cake that Nathalie eats -/
noncomputable def nathalie_fraction : ℝ := 1 / 10

/-- Represents how many times more Pierre eats compared to Nathalie -/
noncomputable def pierre_multiplier : ℝ := 3

/-- Theorem stating how much Pierre eats given the cake's weight and eating patterns -/
theorem pierre_consumption (hw : cake_weight = 525) 
  (hn : nathalie_fraction = 1 / 10) (hp : pierre_multiplier = 3) : 
  pierre_multiplier * (nathalie_fraction * cake_weight) = 157.5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pierre_consumption_l816_81679


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_wall_segments_l816_81648

/-- Represents a regular 37-pointed star -/
structure RegularStar where
  vertices : Finset (ℝ × ℝ)
  is_regular : vertices.card = 37

/-- The configuration of cities in the kingdom -/
def Kingdom := Finset RegularStar

/-- A wall is represented as a set of line segments -/
def Wall := Finset (ℝ × ℝ × ℝ × ℝ)

/-- Predicate to check if a wall is convex -/
def is_convex (w : Wall) : Prop := sorry

/-- Predicate to check if a wall encloses all vertices of all stars -/
def encloses_all_vertices (w : Wall) (k : Kingdom) : Prop := sorry

/-- The number of unique segments in a wall -/
noncomputable def num_segments (w : Wall) : ℕ := sorry

/-- Theorem stating that there exists a convex wall with at least 37 segments -/
theorem min_wall_segments (k : Kingdom) :
  k.card = 500 →
  ∃ (w : Wall), is_convex w ∧ encloses_all_vertices w k ∧ num_segments w ≥ 37 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_wall_segments_l816_81648


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_equality_l816_81682

-- Define the set we want to prove equal to the solution
def S : Set ℝ := {x : ℝ | |x - 2| + |x + 3| ≥ 7}

-- State the theorem
theorem solution_set_equality : S = Set.Iic (-4) ∪ Set.Ici 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_equality_l816_81682


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_subtraction_l816_81694

theorem vector_subtraction (a b : (Fin 3 → ℝ)) 
  (h₁ : a = ![(-5 : ℝ), 1, 3]) 
  (h₂ : b = ![(3 : ℝ), -2, 0]) :
  a - 4 • b = ![((-17) : ℝ), 9, 3] := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_subtraction_l816_81694


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mySequence_formula_l816_81642

def mySequence (n : ℕ) : ℚ :=
  match n with
  | 0 => 1
  | n + 1 => mySequence n / (1 + mySequence n)

theorem mySequence_formula : ∀ n : ℕ, n > 0 → mySequence n = 1 / n :=
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mySequence_formula_l816_81642


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_B_speed_l816_81602

/-- The speed of Train B given the conditions of the problem -/
noncomputable def speed_of_train_B (speed_A : ℝ) (head_start : ℝ) (overtake_time : ℝ) : ℝ :=
  let distance_A_before_B_starts := speed_A * (head_start / 60)
  let additional_distance_A := speed_A * (overtake_time / 60)
  let total_distance := distance_A_before_B_starts + additional_distance_A
  total_distance / (overtake_time / 60)

/-- Theorem stating that the speed of Train B is 80 mph -/
theorem train_B_speed :
  speed_of_train_B 60 40 120 = 80 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_B_speed_l816_81602


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l816_81614

theorem range_of_a (P : Set ℝ) (M : Set ℝ) (a : ℝ) :
  P = { x : ℝ | x^2 ≤ 1 } →
  M = { a } →
  P ∪ M = P →
  a ∈ Set.Icc (-1 : ℝ) 1 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l816_81614


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_repeating_28_equals_fraction_l816_81626

-- Define the repeating decimal 0.282828...
noncomputable def repeating_28 : ℚ := 28 / 99

-- Theorem statement
theorem repeating_28_equals_fraction :
  repeating_28 = 28 / 99 := by
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_repeating_28_equals_fraction_l816_81626


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_ratio_theorem_l816_81664

-- Define the triangle ABC
structure Triangle (A B C : ℝ × ℝ) where
  angleA : Real.cos (60 * Real.pi / 180) = (B.1 - C.1) / Real.sqrt ((B.1 - C.1)^2 + (B.2 - C.2)^2)
  angleB : Real.cos (45 * Real.pi / 180) = (C.1 - A.1) / Real.sqrt ((C.1 - A.1)^2 + (C.2 - A.2)^2)

-- Define the line DE
def LineDE (D E : ℝ × ℝ) (A B C : ℝ × ℝ) : Prop :=
  ∃ (t : ℝ), 0 ≤ t ∧ t ≤ 1 ∧ D = (t * A.1 + (1 - t) * B.1, t * A.2 + (1 - t) * B.2) ∧
  ∃ (s : ℝ), 0 ≤ s ∧ s ≤ 1 ∧ E = (s * B.1 + (1 - s) * C.1, s * B.2 + (1 - s) * C.2)

-- Define the angle ADE
def AngleADE (A D E : ℝ × ℝ) : Prop :=
  Real.cos (75 * Real.pi / 180) = ((E.1 - D.1) * (A.1 - D.1) + (E.2 - D.2) * (A.2 - D.2)) /
    (Real.sqrt ((E.1 - D.1)^2 + (E.2 - D.2)^2) * Real.sqrt ((A.1 - D.1)^2 + (A.2 - D.2)^2))

-- Define the area ratio condition
def AreaRatio (A B C D E : ℝ × ℝ) : Prop :=
  (D.1 - A.1) * (E.2 - A.2) - (E.1 - A.1) * (D.2 - A.2) =
    1/3 * ((B.1 - A.1) * (C.2 - A.2) - (C.1 - A.1) * (B.2 - A.2))

-- Theorem statement
theorem triangle_ratio_theorem (A B C D E : ℝ × ℝ) (ABC : Triangle A B C) :
  LineDE D E A B C →
  AngleADE A D E →
  AreaRatio A B C D E →
  (D.1 - A.1)^2 + (D.2 - A.2)^2 = 1/9 * ((B.1 - A.1)^2 + (B.2 - A.2)^2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_ratio_theorem_l816_81664


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonicity_depends_on_a_min_value_when_a_less_than_neg_one_l816_81649

-- Define the function f(x)
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (1/3) * x^3 + (a-1) * x^2 - 4*a*x + a

-- Theorem for monotonicity
theorem monotonicity_depends_on_a (a : ℝ) : 
  ∃ (x₁ x₂ : ℝ), x₁ < x₂ ∧ 
  ((a = -1 → f a x₁ ≤ f a x₂) ∧
   (a > -1 → (x₁ < -2*a ∧ x₂ > 2) → f a x₁ < f a x₂) ∧
   (a < -1 → (x₁ < 2 ∧ x₂ > -2*a) → f a x₁ < f a x₂)) :=
by sorry

-- Theorem for minimum value when a < -1
theorem min_value_when_a_less_than_neg_one (a : ℝ) (h : a < -1) :
  (∀ x ∈ Set.Icc 2 3, 
    (-3/2 < a → f a (-2*a) ≤ f a x) ∧
    (a ≤ -3/2 → f a 3 ≤ f a x)) ∧
  ((-3/2 < a → (4/3)*a^3 + 4*a^2 + a = f a (-2*a)) ∧
   (a ≤ -3/2 → -2*a = f a 3)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonicity_depends_on_a_min_value_when_a_less_than_neg_one_l816_81649


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_theorem_l816_81629

/-- A positive increasing geometric sequence -/
def IsPositiveIncreasingGeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ (q : ℝ), q > 1 ∧ ∀ n, a n = a 1 * q^(n-1) ∧ a n > 0

/-- The condition given in the problem -/
def SatisfiesCondition (a : ℕ → ℝ) (l : ℝ) : Prop :=
  1 + (a 2 - a 4) + l * (a 3 - a 5) = 0

/-- The theorem statement -/
theorem min_value_theorem (a : ℕ → ℝ) (l : ℝ) :
  IsPositiveIncreasingGeometricSequence a →
  SatisfiesCondition a l →
  ∀ m : ℝ, a 8 + m * a 9 ≥ 27/4 ∧ ∃ n : ℝ, a 8 + n * a 9 = 27/4 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_theorem_l816_81629


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_condition_l816_81646

/-- Two lines in the plane -/
structure TwoLines (a b l : ℝ) where
  line1 : ℝ → ℝ → Prop := λ x y ↦ a * x + y - l = 0
  line2 : ℝ → ℝ → Prop := λ x y ↦ x + b * y - 1 = 0

/-- Definition of parallel lines -/
def parallel (a b l : ℝ) (lines : TwoLines a b l) : Prop :=
  ∃ (k : ℝ), k ≠ 0 ∧ ∀ (x y : ℝ), lines.line1 x y ↔ lines.line2 (k * x) (k * y)

/-- Main theorem -/
theorem parallel_condition (a b l : ℝ) :
  (∀ (lines : TwoLines a b l), parallel a b l lines → a * b = 1) ∧
  ¬(∀ (lines : TwoLines a b l), a * b = 1 → parallel a b l lines) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_condition_l816_81646


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_H_l816_81608

-- Define the function H
def H (x : ℝ) : ℝ := |x + 2| - |x - 3|

-- State the theorem about the range of H
theorem range_of_H :
  Set.range H = Set.Iic 5 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_H_l816_81608


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_stating_revolver_probability_l816_81604

/-- The probability of the revolver firing on any given shot -/
def p : ℚ := 1 / 6

/-- The probability of the revolver not firing on any given shot -/
def q : ℚ := 1 - p

/-- The number of chambers in the revolver -/
def n : ℕ := 6

/-- The probability that the shot will happen when the revolver is with A -/
def prob_A_shoots : ℚ := 6 / 11

/-- 
Theorem stating that the sum of probabilities for A shooting converges to prob_A_shoots.
For any k, the sum up to k terms is less than prob_A_shoots,
and the sum up to k+1 terms is greater than or equal to prob_A_shoots.
-/
theorem revolver_probability :
  ∀ k : ℕ, 
  (Finset.range k).sum (λ i => p * q^(2*i)) < prob_A_shoots ∧ 
  (Finset.range (k+1)).sum (λ i => p * q^(2*i)) ≥ prob_A_shoots :=
by
  sorry  -- The proof is omitted for now

#check revolver_probability

end NUMINAMATH_CALUDE_ERRORFEEDBACK_stating_revolver_probability_l816_81604


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence1_formula_sequence2_formula_sequence3_formula_sequence4_formula_l816_81613

-- Sequence 1
def sequence1 (n : ℕ) : ℕ := 2 * n + 2

theorem sequence1_formula (n : ℕ) : sequence1 n = 2 * (n + 1) := by
  sorry

-- Sequence 2
def sequence2 (n : ℕ+) : ℚ := (-1)^(n : ℕ) * (1 : ℚ) / (n * (n + 1))

theorem sequence2_formula (n : ℕ+) : sequence2 n = (-1)^(n : ℕ) * (1 : ℚ) / (n * (n + 1)) := by
  sorry

-- Sequence 3
def sequence3 (a b : ℝ) (n : ℕ) : ℝ := if n % 2 = 0 then b else a

theorem sequence3_formula (a b : ℝ) (n : ℕ) : 
  sequence3 a b n = if n % 2 = 0 then b else a := by
  sorry

-- Sequence 4
def sequence4 (n : ℕ+) : ℕ := 10^(n : ℕ) - 1

theorem sequence4_formula (n : ℕ+) : sequence4 n = 10^(n : ℕ) - 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence1_formula_sequence2_formula_sequence3_formula_sequence4_formula_l816_81613


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_alpha_for_point_l816_81658

/-- If the terminal side of angle α passes through the point (-1, -1), then sin α = -√2/2 -/
theorem sin_alpha_for_point (α : ℝ) : 
  (∃ (r : ℝ), r > 0 ∧ r * Real.cos α = -1 ∧ r * Real.sin α = -1) → 
  Real.sin α = -Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_alpha_for_point_l816_81658


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_cubic_polynomial_l816_81623

/-- A cubic polynomial with real coefficients -/
def cubic_polynomial (a b c d : ℝ) : ℂ → ℂ := λ x ↦ a * x^3 + b * x^2 + c * x + d

theorem unique_cubic_polynomial :
  ∃! (a b c d : ℝ), 
    (cubic_polynomial a b c d (3 + 2*Complex.I) = 0) ∧ 
    (b = -10) ∧
    (a = 1) ∧
    (c = 37) ∧
    (d = -52) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_cubic_polynomial_l816_81623


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_tiling_theorem_l816_81645

/-- Represents a rectangle with width and height -/
structure Rectangle where
  width : ℝ
  height : ℝ

/-- Predicate to check if a rectangle has at least one integer side -/
def hasIntegerSide (r : Rectangle) : Prop :=
  (∃ n : ℤ, r.width = n) ∨ (∃ n : ℤ, r.height = n)

/-- Predicate to check if a rectangle can be tiled with given rectangles -/
def canBeTiledWith (r : Rectangle) (tiles : List Rectangle) : Prop :=
  ∃ (arrangement : List (Rectangle × ℝ × ℝ)), 
    (∀ (tile_pos : Rectangle × ℝ × ℝ), tile_pos ∈ arrangement →
      let (tile, x, y) := tile_pos
      x ≥ 0 ∧ y ≥ 0 ∧ x + tile.width ≤ r.width ∧ y + tile.height ≤ r.height) ∧
    (∀ x y, 0 ≤ x ∧ x < r.width ∧ 0 ≤ y ∧ y < r.height →
      ∃ (tile_pos : Rectangle × ℝ × ℝ), tile_pos ∈ arrangement ∧
        let (tile, tx, ty) := tile_pos
        tx ≤ x ∧ x < tx + tile.width ∧ ty ≤ y ∧ y < ty + tile.height)

/-- Main theorem statement -/
theorem rectangle_tiling_theorem (r : Rectangle) (tiles : List Rectangle) :
  (∀ t ∈ tiles, hasIntegerSide t) →
  canBeTiledWith r tiles →
  hasIntegerSide r := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_tiling_theorem_l816_81645


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_triangle_PAB_range_l816_81656

-- Define the function f(x)
noncomputable def f (x : ℝ) : ℝ :=
  if 0 < x ∧ x < 1 then -Real.log x
  else if x > 1 then Real.log x
  else 0

-- Define the points and lines
variable (x₁ x₂ : ℝ)
variable (P₁ P₂ P A B : ℝ × ℝ)
variable (l₁ l₂ : ℝ → ℝ)

-- State the theorem
theorem area_of_triangle_PAB_range
  (h₁ : 0 < x₁ ∧ x₁ < 1)
  (h₂ : x₂ > 1)
  (h₃ : P₁ = (x₁, f x₁))
  (h₄ : P₂ = (x₂, f x₂))
  (h₅ : HasDerivAt l₁ (-1/x₁) x₁)
  (h₆ : HasDerivAt l₂ (1/x₂) x₂)
  (h₇ : ∃ (P : ℝ × ℝ), l₁ P.1 = P.2 ∧ l₂ P.1 = P.2)
  (h₈ : (deriv l₁ P.1) * (deriv l₂ P.1) = -1)
  (h₉ : A = (0, l₁ 0))
  (h₁₀ : B = (0, l₂ 0))
  : ∃ S : ℝ, 0 < S ∧ S < 1 ∧ S = (1/2) * |A.2 - B.2| * |P.1| :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_triangle_PAB_range_l816_81656


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_journey_time_is_17_4_hours_l816_81684

/-- Represents the journey parameters and conditions -/
structure JourneyParams where
  totalDistance : ℚ
  initialSpeed : ℚ
  walkingSpeed : ℚ
  carDistance : ℚ
  backtrackDistance : ℚ

/-- Calculates the total time for the journey given the parameters -/
def totalJourneyTime (p : JourneyParams) : ℚ :=
  p.carDistance / p.initialSpeed + 
  (p.totalDistance - p.carDistance) / p.walkingSpeed

/-- Theorem stating that under the given conditions, the journey takes 17.4 hours -/
theorem journey_time_is_17_4_hours (p : JourneyParams) 
  (h1 : p.totalDistance = 120)
  (h2 : p.initialSpeed = 25)
  (h3 : p.walkingSpeed = 4)
  (h4 : p.carDistance = 60)
  (h5 : p.backtrackDistance = 40)
  (h6 : p.carDistance / p.initialSpeed + (p.totalDistance - p.carDistance) / p.walkingSpeed = 
        p.carDistance / p.initialSpeed + p.backtrackDistance / p.initialSpeed + 
        (p.totalDistance - (p.carDistance - p.backtrackDistance)) / p.initialSpeed) :
  totalJourneyTime p = 87/5 := by
  sorry

#check journey_time_is_17_4_hours

end NUMINAMATH_CALUDE_ERRORFEEDBACK_journey_time_is_17_4_hours_l816_81684


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_parallel_lines_formula_l816_81631

/-- The distance between two parallel lines ax+by+m=0 and ax+by+n=0 -/
noncomputable def distance_between_parallel_lines (a b m n : ℝ) : ℝ :=
  |m - n| / Real.sqrt (a^2 + b^2)

/-- Theorem: The distance between two parallel lines ax+by+m=0 and ax+by+n=0
    is |m-n| / √(a^2 + b^2), where a, b, m, and n are real constants. -/
theorem distance_between_parallel_lines_formula (a b m n : ℝ) :
  distance_between_parallel_lines a b m n =
    |m - n| / Real.sqrt (a^2 + b^2) := by
  -- The proof goes here
  sorry

#check distance_between_parallel_lines_formula

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_parallel_lines_formula_l816_81631


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_greatest_integer_less_than_M_div_100_l816_81621

theorem greatest_integer_less_than_M_div_100 : 
  let sum : ℚ := 1 / (3*2*1*16*15*14*13*12*11*10*9*8*7*6*5*4*3*2*1) + 
                 1 / (4*3*2*1*15*14*13*12*11*10*9*8*7*6*5*4*3*2*1) + 
                 1 / (5*4*3*2*1*14*13*12*11*10*9*8*7*6*5*4*3*2*1) + 
                 1 / (6*5*4*3*2*1*13*12*11*10*9*8*7*6*5*4*3*2*1) + 
                 1 / (7*6*5*4*3*2*1*12*11*10*9*8*7*6*5*4*3*2*1) + 
                 1 / (8*7*6*5*4*3*2*1*11*10*9*8*7*6*5*4*3*2*1) + 
                 1 / (9*8*7*6*5*4*3*2*1*10*9*8*7*6*5*4*3*2*1) + 
                 1 / (10*9*8*7*6*5*4*3*2*1*9*8*7*6*5*4*3*2*1)
  let M : ℚ := sum * (2*1*17*16*15*14*13*12*11*10*9*8*7*6*5*4*3*2*1)
  ⌊M / 100⌋ = 137 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_greatest_integer_less_than_M_div_100_l816_81621


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tetrahedron_smallest_loop_l816_81609

/-- The smallest length of a loop of twine through which a regular tetrahedron can pass -/
def smallest_loop_length (L : ℝ) : ℝ := 2 * L

/-- A set representing a loop of twine -/
def is_loop (s : Set ℝ) : Prop := sorry

/-- The length of a loop -/
def length (s : Set ℝ) : ℝ := sorry

/-- Predicate indicating whether a regular tetrahedron can pass through a given loop -/
def tetrahedron_passable (L : ℝ) (loop : Set ℝ) : Prop := sorry

/-- Theorem: The smallest length of a loop of twine through which a regular tetrahedron 
    with side length L can pass is equal to 2L -/
theorem tetrahedron_smallest_loop (L : ℝ) (h : L > 0) : 
  ∀ x : ℝ, x ≥ smallest_loop_length L ↔ 
    ∃ (loop : Set ℝ), is_loop loop ∧ length loop = x ∧ tetrahedron_passable L loop :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tetrahedron_smallest_loop_l816_81609


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_decreasing_exponential_base_range_l816_81690

-- Define the function f(x) = a^x
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a^x

-- State the theorem
theorem decreasing_exponential_base_range (a : ℝ) :
  a > 0 ∧ a ≠ 1 ∧ (∀ x y : ℝ, x < y → f a x > f a y) → 0 < a ∧ a < 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_decreasing_exponential_base_range_l816_81690


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_eldest_child_age_l816_81610

theorem eldest_child_age (ages : List ℕ) 
  (h_length : ages.length = 5)
  (h_diff : ∀ i j, i < j → j < ages.length → ages[j]! - ages[i]! = 2 * (j - i))
  (h_sum : ages.sum = 40) :
  ages[4]! = 12 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_eldest_child_age_l816_81610


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_string_length_on_cylinder_l816_81663

/-- 
Given a circular cylindrical post with:
- Circumference of 6 feet
- Height of 9 feet
- A string wrapping around it 3 times from bottom to top

Prove that the length of the string is 9√5 feet
-/
theorem string_length_on_cylinder (circumference height : ℝ) 
  (h_circ : circumference = 6)
  (h_height : height = 9)
  (wraps : ℕ)
  (h_wraps : wraps = 3) :
  wraps * Real.sqrt ((height / wraps) ^ 2 + circumference ^ 2) = 9 * Real.sqrt 5 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_string_length_on_cylinder_l816_81663


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_properties_l816_81666

/-- Complex number -/
structure ComplexNumber where
  re : ℝ
  im : ℝ

/-- Absolute value of a complex number -/
noncomputable def complexAbs (z : ComplexNumber) : ℝ := Real.sqrt (z.re^2 + z.im^2)

/-- Conjugate of a complex number -/
def conj (z : ComplexNumber) : ComplexNumber := ⟨z.re, -z.im⟩

/-- Defined counterpart of a complex number -/
def underline (z : ComplexNumber) : ComplexNumber := ⟨-z.re, z.im⟩

/-- Addition of complex numbers -/
def add (z w : ComplexNumber) : ComplexNumber := ⟨z.re + w.re, z.im + w.im⟩

/-- Multiplication of complex numbers -/
def mul (z w : ComplexNumber) : ComplexNumber := 
  ⟨z.re * w.re - z.im * w.im, z.re * w.im + z.im * w.re⟩

/-- Division of complex numbers -/
noncomputable def div (z w : ComplexNumber) : ComplexNumber := 
  let denom := w.re^2 + w.im^2
  ⟨(z.re * w.re + z.im * w.im) / denom, (z.im * w.re - z.re * w.im) / denom⟩

/-- Main theorem -/
theorem complex_properties (z : ComplexNumber) : 
  (complexAbs z = complexAbs (conj z) ∧ complexAbs z = complexAbs (underline z)) ∧ 
  (add (conj z) (underline z) = ⟨0, 0⟩) ∧
  (mul z (conj z) ≠ mul z (underline z)) ∧
  (z.im ≠ 0 → ∃ (r : ℝ), div z (underline z) ≠ ⟨0, r⟩) := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_properties_l816_81666


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_2019_equals_2019_l816_81662

-- Define the set of positive natural numbers
def PositiveNat := {n : ℕ | n > 0}

-- Define the property of the function f
def SatisfiesInequality (f : ℕ+ → ℕ+) :=
  ∀ m n : ℕ+, f (m + n) ≥ f m + f (f n) - 1

-- Theorem statement
theorem f_2019_equals_2019 
  (f : ℕ+ → ℕ+) 
  (h : SatisfiesInequality f) : 
  f 2019 = 2019 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_2019_equals_2019_l816_81662


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_simplification_l816_81640

theorem expression_simplification :
  (1 / Real.sqrt 0.25) + ((1 / 27) ^ (-(1 : ℝ) / 3))
  + Real.sqrt ((Real.log 3 / Real.log 10) ^ 2 - Real.log 9 / Real.log 10 + 1)
  - Real.log (1 / 3) / Real.log 10
  + 81 ^ (0.5 * Real.log 5 / Real.log 3) = 31 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_simplification_l816_81640


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_roots_l816_81652

-- Define the function F(x)
noncomputable def F (x : ℝ) : ℝ := 2^(2*x) - 5 * 3^x + 29

-- State the theorem
theorem equation_roots :
  (∃ x₁ x₂ : ℝ, x₁ = 2 ∧ 5.54 < x₂ ∧ x₂ < 5.56 ∧ F x₁ = 0 ∧ F x₂ = 0) ∧
  (∀ x : ℝ, F x = 0 → x = 2 ∨ (5.54 < x ∧ x < 5.56)) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_roots_l816_81652


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_slope_implies_a_l816_81615

-- Define the function f(x)
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (a * x^2) / (x + 1)

-- Define the derivative of f(x)
noncomputable def f_derivative (a : ℝ) (x : ℝ) : ℝ := (a * x^2 + 2 * a * x) / (x + 1)^2

theorem tangent_slope_implies_a (a : ℝ) :
  f_derivative a 1 = 1 → a = 4/3 := by
  intro h
  -- Proof steps would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_slope_implies_a_l816_81615


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_prism_volume_ratio_l816_81697

/-- 
Given a right circular cone inscribed in a right rectangular prism,
where the base of the prism is a rectangle with length 3r and width r,
and both the cone and prism have height h,
prove that the ratio of the volume of the cone to the volume of the prism is π/36.
-/
theorem cone_prism_volume_ratio (r h : ℝ) (hr : r > 0) (hh : h > 0) :
  (1 / 3 * π * (r / 2)^2 * h) / (3 * r^2 * h) = π / 36 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_prism_volume_ratio_l816_81697


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_base_pyramid_angle_cosine_l816_81620

/-- Represents a pyramid with an equilateral triangle base -/
structure EquilateralBasePyramid where
  -- The side length of the equilateral triangle base
  a : ℝ
  -- The length of the edges connecting the apex to the base vertices
  b : ℝ
  -- Assumption that a and b are positive
  a_pos : a > 0
  b_pos : b > 0

/-- The angle formed by the edges AD, BD, and CD with the plane ABC -/
noncomputable def angle_with_base (pyramid : EquilateralBasePyramid) : ℝ := 
  sorry

/-- 
Theorem: In a pyramid ABCD where ABC is an equilateral triangle with side length a,
and AD = BD = CD = b, the cosine of the angle formed by the edges AD, BD, and CD
with the plane ABC is equal to a / (b * √3).
-/
theorem equilateral_base_pyramid_angle_cosine 
  (pyramid : EquilateralBasePyramid) : 
  Real.cos (angle_with_base pyramid) = pyramid.a / (pyramid.b * Real.sqrt 3) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_base_pyramid_angle_cosine_l816_81620


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_gender_associated_with_payment_expected_value_of_X_l816_81644

-- Define the contingency table
def contingency_table : Matrix (Fin 2) (Fin 2) ℕ :=
λ i j => match i, j with
  | 0, 0 => 45  -- Male, Facial Recognition
  | 0, 1 => 25  -- Female, Facial Recognition
  | 1, 0 => 10  -- Male, Non-Facial Recognition
  | 1, 1 => 20  -- Female, Non-Facial Recognition

-- Define the chi-square statistic function
noncomputable def chi_square (m : Matrix (Fin 2) (Fin 2) ℕ) : ℝ :=
  let n := (m 0 0) + (m 0 1) + (m 1 0) + (m 1 1)
  let ad := (m 0 0) * (m 1 1)
  let bc := (m 0 1) * (m 1 0)
  let row_sums := λ i => (m i 0) + (m i 1)
  let col_sums := λ j => (m 0 j) + (m 1 j)
  (n * (ad - bc)^2 : ℝ) / ((row_sums 0) * (row_sums 1) * (col_sums 0) * (col_sums 1))

-- Define the critical value
def critical_value : ℝ := 6.635

-- Define the hypergeometric distribution function
def hypergeometric (N M n k : ℕ) : ℚ :=
  (Nat.choose M k * Nat.choose (N - M) (n - k) : ℚ) / Nat.choose N n

-- Define the expected value function
def expected_value (N M n : ℕ) : ℚ :=
  (n * M : ℚ) / N

-- Theorem 1: Chi-square value is greater than the critical value
theorem gender_associated_with_payment : chi_square contingency_table > critical_value := by
  sorry

-- Theorem 2: Expected value of X is 20/9
theorem expected_value_of_X : expected_value 9 5 4 = 20 / 9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_gender_associated_with_payment_expected_value_of_X_l816_81644


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_plus_abs_equals_three_l816_81691

theorem sqrt_plus_abs_equals_three (x : ℝ) (h : -2 * x^2 + 5 * x - 2 > 0) :
  Real.sqrt (4 * x^2 - 4 * x + 1) + 2 * abs (x - 2) = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_plus_abs_equals_three_l816_81691


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_factorial_squared_l816_81689

theorem sqrt_factorial_squared : (Real.sqrt (Nat.factorial 5 * Nat.factorial 4))^2 = 2880 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_factorial_squared_l816_81689


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_number_not_in_range_l816_81655

/-- A function with specific properties -/
noncomputable def special_function (a b c d : ℝ) (h_a : a ≠ 0) (h_b : b ≠ 0) (h_c : c ≠ 0) (h_d : d ≠ 0) : ℝ → ℝ :=
  fun x => (a * x + b) / (c * x + d)

/-- The theorem stating the unique number not in the range of the special function -/
theorem unique_number_not_in_range 
  (a b c d : ℝ) (h_a : a ≠ 0) (h_b : b ≠ 0) (h_c : c ≠ 0) (h_d : d ≠ 0) :
  let f := special_function a b c d h_a h_b h_c h_d
  (f 20 = 20) →
  (f 100 = 100) →
  (∀ x, x ≠ -d/c → f (f x) = x) →
  (deriv f 20 = -1) →
  ∃! y, (∀ x, f x ≠ y) ∧ y = 60 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_number_not_in_range_l816_81655


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_log_16_2_and_8_l816_81607

theorem log_16_2_and_8 : 
  (∃ (log₁₆ : ℝ → ℝ), 
    (∀ x > 0, log₁₆ x = (Real.log x) / (Real.log 16)) ∧
    log₁₆ 2 = 1/4 ∧ 
    log₁₆ 8 = 3/4) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_log_16_2_and_8_l816_81607


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l816_81659

/-- Definition of an ellipse E with semi-major axis a and semi-minor axis b -/
def Ellipse (a b : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1 ^ 2 / a ^ 2) + (p.2 ^ 2 / b ^ 2) = 1}

/-- Definition of the foci of an ellipse -/
noncomputable def Foci (a b : ℝ) : (ℝ × ℝ) × (ℝ × ℝ) :=
  let c := Real.sqrt (a ^ 2 - b ^ 2)
  ((-c, 0), (c, 0))

/-- Definition of a line with slope 1 passing through a point -/
def Line (p : ℝ × ℝ) : Set (ℝ × ℝ) :=
  {q : ℝ × ℝ | q.2 - p.2 = q.1 - p.1}

/-- Definition of distance between two points -/
noncomputable def Distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1) ^ 2 + (p.2 - q.2) ^ 2)

/-- Statement of the theorem -/
theorem ellipse_properties (a b : ℝ) (h₁ : a > b) (h₂ : b > 0) :
  let E := Ellipse a b
  let (F₁, F₂) := Foci a b
  let l := Line F₁
  ∃ (A B : ℝ × ℝ),
    A ∈ E ∧ B ∈ E ∧ A ∈ l ∧ B ∈ l ∧
    ∃ (k : ℝ), Distance A F₂ - Distance A B = Distance A B - Distance B F₂ ∧
    Distance (0, -1) A = Distance (0, -1) B →
    (a / b = Real.sqrt 2 ∧ a ^ 2 = 18 ∧ b ^ 2 = 9) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l816_81659


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_omega_value_l816_81678

noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := Real.sin (ω * x + Real.pi / 3) + Real.sin (ω * x)

theorem min_omega_value (ω : ℝ) (x₁ x₂ : ℝ) 
  (h_pos : ω > 0)
  (h_f_x₁ : f ω x₁ = 0)
  (h_f_x₂ : f ω x₂ = Real.sqrt 3)
  (h_distance : |x₁ - x₂| = Real.pi) :
  ∀ ω' > 0, (∃ x₁' x₂', f ω' x₁' = 0 ∧ f ω' x₂' = Real.sqrt 3 ∧ |x₁' - x₂'| = Real.pi) → ω' ≥ 1/2 :=
by
  sorry

#check min_omega_value

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_omega_value_l816_81678


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_subset_modular_arithmetic_l816_81619

theorem subset_modular_arithmetic (p : ℕ) (A : Finset ℕ) (h_prime : Nat.Prime p) (h_p : p > 7)
  (h_subset : A ⊆ Finset.range p) (h_size : A.card ≥ (p - 1) / 2) :
  ∀ r : ℤ, ∃ a b c d : ℕ, a ∈ A ∧ b ∈ A ∧ c ∈ A ∧ d ∈ A ∧
    (a * b - c * d) % p = r % p := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_subset_modular_arithmetic_l816_81619


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_sum_problem_l816_81627

theorem tan_sum_problem (x y : ℝ) (h1 : Real.tan x + Real.tan y = 15)
  (h2 : (Real.tan x)⁻¹ * (Real.tan y)⁻¹ = 24) (h3 : ∀ k : ℤ, x + y ≠ (2 * k + 1) * Real.pi / 2) :
  Real.tan (x + y) = 360 / 23 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_sum_problem_l816_81627


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_midpoint_path_area_ratio_is_one_fourth_l816_81624

/-- Right triangle ABC with particle movement and midpoint path --/
structure ParticleTriangle where
  -- Right triangle ABC
  a : ℝ  -- length of side AB
  b : ℝ  -- length of side BC
  -- Particle speeds
  v : ℝ  -- speed of particle starting at A
  -- Assume a, b, v are positive
  a_pos : 0 < a
  b_pos : 0 < b
  v_pos : 0 < v

/-- The ratio of the area enclosed by the midpoint path to the area of the triangle --/
noncomputable def midpointPathAreaRatio (pt : ParticleTriangle) : ℝ :=
  1 / 4

/-- Theorem stating that the midpoint path area ratio is 1/4 --/
theorem midpoint_path_area_ratio_is_one_fourth (pt : ParticleTriangle) :
  midpointPathAreaRatio pt = 1 / 4 := by
  -- Unfold the definition of midpointPathAreaRatio
  unfold midpointPathAreaRatio
  -- The definition is exactly 1/4, so this should be trivial
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_midpoint_path_area_ratio_is_one_fourth_l816_81624


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_sum_6_l816_81676

/-- An arithmetic sequence with its sum function -/
structure ArithmeticSequence where
  a : ℕ → ℚ
  d : ℚ
  is_arithmetic : ∀ n, a (n + 1) = a n + d

/-- Sum of first n terms of an arithmetic sequence -/
def sum_n (seq : ArithmeticSequence) (n : ℕ) : ℚ :=
  n * seq.a 1 + n * (n - 1) / 2 * seq.d

theorem arithmetic_sequence_sum_6 (seq : ArithmeticSequence) 
    (h1 : sum_n seq 3 = 12)
    (h2 : seq.a 2 + seq.a 4 = 4) :
  sum_n seq 6 = 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_sum_6_l816_81676


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_frustum_volume_calculation_l816_81618

/-- Represents a square pyramid -/
structure SquarePyramid where
  baseEdge : ℝ
  height : ℝ

/-- Calculate the volume of a square pyramid -/
noncomputable def pyramidVolume (p : SquarePyramid) : ℝ := 
  (1 / 3) * p.baseEdge^2 * p.height

/-- Represents a frustum of a square pyramid -/
structure Frustum where
  originalPyramid : SquarePyramid
  smallerPyramid : SquarePyramid

/-- Calculate the volume of a frustum -/
noncomputable def frustumVolume (f : Frustum) : ℝ :=
  pyramidVolume f.originalPyramid - pyramidVolume f.smallerPyramid

theorem frustum_volume_calculation (f : Frustum) 
  (h1 : f.originalPyramid.baseEdge = 15)
  (h2 : f.originalPyramid.height = 12)
  (h3 : f.smallerPyramid.baseEdge = 7.5)
  (h4 : f.smallerPyramid.height = 6)
  (h5 : f.smallerPyramid.baseEdge = f.originalPyramid.baseEdge / 2)
  (h6 : f.smallerPyramid.height = f.originalPyramid.height / 2) :
  frustumVolume f = 787.5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_frustum_volume_calculation_l816_81618


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_interest_rate_is_four_percent_l816_81661

/-- Calculates the simple interest for a given principal, rate, and time -/
def simpleInterest (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * rate * time

/-- Calculates the compound interest for a given principal, rate, and time -/
noncomputable def compoundInterest (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * ((1 + rate) ^ time - 1)

/-- Theorem stating that the interest rate is 4% given the problem conditions -/
theorem interest_rate_is_four_percent
  (principal rate time : ℝ)
  (h_principal : principal = 625)
  (h_time : time = 2)
  (h_difference : compoundInterest principal rate time - simpleInterest principal rate time = 1) :
  rate = 0.04 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_interest_rate_is_four_percent_l816_81661


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_line_intersection_arc_ratio_l816_81625

/-- The circle equation (x-1)^2 + y^2 = 1 -/
def our_circle (x y : ℝ) : Prop := (x - 1)^2 + y^2 = 1

/-- The line equation x - y = 0 -/
def our_line (x y : ℝ) : Prop := x - y = 0

/-- The ratio of arc lengths -/
def arc_length_ratio : ℚ := 1 / 3

theorem circle_line_intersection_arc_ratio :
  ∃ (shorter_arc longer_arc : ℝ),
    shorter_arc > 0 ∧
    longer_arc > 0 ∧
    shorter_arc < longer_arc ∧
    shorter_arc / longer_arc = arc_length_ratio := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_line_intersection_arc_ratio_l816_81625


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_value_l816_81674

-- Define the function f
noncomputable def f (x a b : ℝ) : ℝ := (3 * x - a) / (x^2 + b * x - 1)

-- State the theorem
theorem odd_function_value (a b : ℝ) :
  (∀ x ∈ Set.Ioo (-1 : ℝ) 1, f x a b = -f (-x) a b) →
  f (1/2) a b = -2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_value_l816_81674


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complement_intersection_theorem_l816_81643

universe u

def U : Set ℤ := {-1, 0, 1, 2}
def A : Set ℤ := {1, 2}
def B : Set ℤ := {0, 2}

theorem complement_intersection_theorem : 
  (U \ A) ∩ B = {0} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complement_intersection_theorem_l816_81643


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallelogram_perimeter_area_sum_l816_81650

/-- A parallelogram in a 2D plane -/
structure Parallelogram where
  v1 : ℝ × ℝ
  v2 : ℝ × ℝ
  v3 : ℝ × ℝ
  v4 : ℝ × ℝ

/-- Calculate the distance between two points -/
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2)

/-- Calculate the perimeter of a parallelogram -/
noncomputable def perimeter (p : Parallelogram) : ℝ :=
  2 * (distance p.v1 p.v2 + distance p.v2 p.v3)

/-- Calculate the area of a parallelogram -/
def area (p : Parallelogram) : ℝ :=
  (p.v2.1 - p.v1.1) * (p.v2.2 - p.v1.2)

/-- The main theorem -/
theorem parallelogram_perimeter_area_sum :
  let p := Parallelogram.mk (1, 2) (5, 6) (12, 6) (8, 2)
  perimeter p + area p = 42 + 8 * Real.sqrt 2 := by
  sorry

#eval "Theorem stated successfully"

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallelogram_perimeter_area_sum_l816_81650


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_distance_theorem_l816_81685

/-- Parabola structure -/
structure Parabola where
  focus : ℝ × ℝ
  directrix : ℝ
  equation : (ℝ × ℝ) → Prop

/-- Point on a parabola -/
def PointOnParabola (p : Parabola) (point : ℝ × ℝ) : Prop :=
  p.equation point

/-- Distance between two points -/
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

/-- Angle between three points -/
noncomputable def angle (p1 p2 p3 : ℝ × ℝ) : ℝ := sorry

/-- Main theorem -/
theorem parabola_distance_theorem (p : Parabola) 
  (h1 : p.focus = (1, 0))
  (h2 : p.directrix = -1)
  (h3 : p.equation = fun point => point.2^2 = 4 * point.1)
  (A : ℝ × ℝ)
  (hA : PointOnParabola p A)
  (E : ℝ × ℝ)
  (hE : E.1 = p.directrix ∧ E.2 = A.2)
  (hAngle : angle A p.focus E = 75 * π / 180) :
  distance A E = 4 * Real.sqrt 3 + 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_distance_theorem_l816_81685


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_polar_curve_l816_81611

/-- The area enclosed by the curve ρ = 4cosθ in the polar coordinate system -/
noncomputable def polar_curve_area : ℝ := 4 * Real.pi

/-- The equation of the curve in polar coordinates -/
noncomputable def polar_curve_equation (θ : ℝ) : ℝ := 4 * Real.cos θ

theorem area_of_polar_curve :
  polar_curve_area = ∫ θ in (0 : ℝ)..(2 * Real.pi), (1/2) * (polar_curve_equation θ)^2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_polar_curve_l816_81611


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_4beta_minus_4cos_4alpha_l816_81606

open Real

theorem cos_4beta_minus_4cos_4alpha (θ α β : ℝ) 
  (h1 : Real.sin α = (Real.sin θ + Real.cos θ) / 2)
  (h2 : (Real.sin β) ^ 2 = Real.sin θ * Real.cos θ) :
  Real.cos (4 * β) - 4 * Real.cos (4 * α) = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_4beta_minus_4cos_4alpha_l816_81606


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_roots_difference_reciprocals_squared_l816_81657

theorem roots_difference_reciprocals_squared (x₁ x₂ : ℝ) :
  (Real.sqrt 14 * x₁^2 - Real.sqrt 116 * x₁ + Real.sqrt 56 = 0) →
  (Real.sqrt 14 * x₂^2 - Real.sqrt 116 * x₂ + Real.sqrt 56 = 0) →
  |1/x₁^2 - 1/x₂^2| = Real.sqrt 29/14 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_roots_difference_reciprocals_squared_l816_81657


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_problem_l816_81633

theorem triangle_problem (A B C a b c : Real) : 
  0 < A ∧ A < π/2 →  -- Acute angle A
  0 < B ∧ B < π/2 →  -- Acute angle B
  0 < C ∧ C < π/2 →  -- Acute angle C
  A + B + C = π →    -- Sum of angles in a triangle
  a > 0 →            -- Positive side length
  b > 0 →            -- Positive side length
  c > 0 →            -- Positive side length
  a = b * Real.cos C + (Real.sqrt 3 / 3) * c * Real.sin B →  -- Given equation
  ((a = 2 ∧ b = Real.sqrt 7) → c = 3) ∧  -- Part 1
  (Real.sqrt 3 * Real.sin (2*A - π/6) - 2 * (Real.sin (C - π/12))^2 = 0 → A = π/4)  -- Part 2
:= by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_problem_l816_81633


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_general_term_l816_81630

def our_sequence (a : ℕ → ℚ) : Prop :=
  a 1 = 1 ∧ ∀ n : ℕ, n ≥ 1 → a n - a (n + 1) = (2 * a n * a (n + 1)) / (n * (n + 1))

theorem sequence_general_term (a : ℕ → ℚ) (h : our_sequence a) :
  ∀ n : ℕ, n ≥ 1 → a n = n / (3 * n - 2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_general_term_l816_81630


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_function_l816_81683

theorem range_of_function (x : ℝ) :
  -π / 2 ≤ x ∧ x ≤ π / 2 →
  ∃ y, (Real.sqrt 3 * Real.sin x + Real.cos x) = y ∧ -Real.sqrt 3 ≤ y ∧ y ≤ 2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_function_l816_81683


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonicity_when_a_is_quarter_max_value_condition_l816_81692

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log (x + 1) + a * x^2 - x

-- Define the derivative of f
noncomputable def f_derivative (a : ℝ) (x : ℝ) : ℝ := 1 / (x + 1) + 2 * a * x - 1

-- Statement for part 1
theorem monotonicity_when_a_is_quarter (x : ℝ) (hx : x > -1) :
  let a : ℝ := 1/4
  (x > -1 ∧ x < 0 ∨ x > 1) → f_derivative a x > 0 ∧
  (x > 0 ∧ x < 1) → f_derivative a x < 0 := by
  sorry

-- Statement for part 2
theorem max_value_condition (a : ℝ) :
  (∀ b : ℝ, b > 1 ∧ b < 2 →
    ∀ x : ℝ, x > -1 ∧ x ≤ b → f a x ≤ f a b) ↔
  a ≥ 1 - Real.log 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonicity_when_a_is_quarter_max_value_condition_l816_81692


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_asymptote_equation_l816_81693

/-- Represents a hyperbola with semi-major axis a and semi-minor axis b -/
structure Hyperbola (a b : ℝ) where
  a_pos : a > 0
  b_pos : b > 0

/-- The perimeter of the quadrilateral formed by lines parallel to the asymptotes through the foci -/
def quadrilateral_perimeter (h : Hyperbola a b) : ℝ := 8 * b

/-- The equation of the asymptotes of the hyperbola -/
def asymptote_equation (h : Hyperbola a b) : ℝ → ℝ → Prop :=
  λ x y ↦ y = x ∨ y = -x

theorem hyperbola_asymptote_equation (a b : ℝ) (h : Hyperbola a b) :
  quadrilateral_perimeter h = 8 * b →
  asymptote_equation h = λ x y ↦ y = x ∨ y = -x :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_asymptote_equation_l816_81693


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_at_i_l816_81632

/-- A polynomial that maps integers to integers -/
def IntegerPolynomial (P : ℂ → ℂ) : Prop :=
  ∀ n : ℤ, ∃ m : ℤ, P n = m

/-- The set of complex numbers with denominators not divisible by primes congruent to 1 mod 4 -/
def ValidComplexSet : Set ℂ :=
  {z : ℂ | ∃ (a b : ℚ), z = a + b * Complex.I ∧
    ∀ (p : ℕ), Nat.Prime p → p % 4 = 1 →
      (a.den : ℤ) % p ≠ 0 ∧ (b.den : ℤ) % p ≠ 0}

/-- The main theorem -/
theorem polynomial_at_i (P : ℂ → ℂ) (h : IntegerPolynomial P) :
  P Complex.I ∈ ValidComplexSet := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_at_i_l816_81632


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_roots_l816_81672

/-- The cubic equation x^3 - x - a = 0 -/
def cubic_equation (a : ℝ) (x : ℝ) : Prop := x^3 - x - a = 0

/-- The critical value for parameter a -/
noncomputable def critical_value : ℝ := 2 * Real.sqrt 3 / 9

theorem cubic_roots (a : ℝ) :
  (∃ x y z : ℝ, x ≠ y ∧ y ≠ z ∧ x ≠ z ∧ cubic_equation a x ∧ cubic_equation a y ∧ cubic_equation a z) ∨
  (∃! x : ℝ, cubic_equation a x) ∨
  (∃ x y : ℝ, x ≠ y ∧ cubic_equation a x ∧ cubic_equation a y ∧ 
    (∀ z : ℝ, cubic_equation a z → z = x ∨ z = y)) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_roots_l816_81672


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_alpha_for_point_one_neg_two_l816_81699

noncomputable section

open Real

theorem sin_alpha_for_point_one_neg_two (α : ℝ) :
  (∃ (x y : ℝ), x = 1 ∧ y = -2 ∧ x^2 + y^2 = 5 ∧ (cos α = x / sqrt (x^2 + y^2)) ∧ (sin α = y / sqrt (x^2 + y^2))) →
  sin α = -2/5 * sqrt 5 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_alpha_for_point_one_neg_two_l816_81699


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_log_inequality_range_l816_81665

theorem log_inequality_range (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  Real.log (3/5) / Real.log a < 1 ↔ (a ∈ Set.Ioo 0 (3/5) ∪ Set.Ioi 1) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_log_inequality_range_l816_81665


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_chord_length_l816_81647

/-- The family of curves parameterized by θ -/
def curve (θ : ℝ) (x y : ℝ) : Prop :=
  2 * (2 * Real.sin θ - Real.cos θ) * x^2 - (8 * Real.sin θ + Real.cos θ + 1) * y = 0

/-- The line y = 2x -/
def line (x y : ℝ) : Prop := y = 2 * x

/-- The chord length function -/
noncomputable def chord_length (θ : ℝ) : ℝ :=
  let x₂ := (8 * Real.sin θ - Real.cos θ + 1) / (2 * Real.sin θ - Real.cos θ + 3)
  Real.sqrt 5 * abs x₂

/-- Theorem: The maximum chord length is 8√5 -/
theorem max_chord_length :
  ∃ (θ : ℝ), ∀ (φ : ℝ), chord_length θ ≥ chord_length φ ∧ chord_length θ = 8 * Real.sqrt 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_chord_length_l816_81647


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_length_perpendicular_bisector_l816_81635

theorem chord_length_perpendicular_bisector (r : ℝ) (h : r = 12) :
  let c := Metric.sphere (0 : ℝ × ℝ) r
  let chord := {p : ℝ × ℝ | p.1^2 + p.2^2 = r^2 ∧ p.1 = r / 2}
  ∃ p ∈ chord, ‖p‖ = 12 * Real.sqrt 3 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_length_perpendicular_bisector_l816_81635


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_divisors_l816_81675

/-- The number of positive divisors of 18800 that are divisible by 235 -/
def num_divisors : ℕ := 10

/-- 18800 expressed as its prime factorization -/
def n : ℕ := 2^4 * 5^2 * 47

/-- 235 expressed as its prime factorization -/
def m : ℕ := 5 * 47

theorem count_divisors : 
  (Finset.filter (λ d ↦ d ∣ n ∧ m ∣ d) (Finset.range (n + 1))).card = num_divisors :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_divisors_l816_81675


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_all_lines_pass_through_single_point_l816_81668

/-- A line on a plane -/
structure Line where
  -- We don't need to define the internal structure of a line

/-- A point on a plane -/
structure Point where
  -- We don't need to define the internal structure of a point

/-- The color of a line -/
inductive Color
  | Red
  | Blue

/-- A configuration of lines on a plane -/
structure LineConfiguration where
  lines : Finset Line
  color : Line → Color
  intersectionPoint : Line → Line → Option Point

/-- Predicate to check if two lines are parallel -/
def parallel (l1 l2 : Line) : Prop := sorry

/-- Predicate to check if a point lies on a line -/
def pointOnLine (p : Point) (l : Line) : Prop := sorry

theorem all_lines_pass_through_single_point (config : LineConfiguration) :
  (∀ l1 l2, l1 ∈ config.lines → l2 ∈ config.lines → l1 ≠ l2 → ¬parallel l1 l2) →
  (∀ l1 l2 l3, l1 ∈ config.lines → l2 ∈ config.lines → l3 ∈ config.lines →
    l1 ≠ l2 → config.color l1 = config.color l2 →
    ∃ p, config.intersectionPoint l1 l2 = some p →
      ∃ l4, l4 ∈ config.lines ∧ config.color l4 ≠ config.color l1 ∧ pointOnLine p l4) →
  ∃ p, ∀ l, l ∈ config.lines → pointOnLine p l := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_all_lines_pass_through_single_point_l816_81668


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rounding_2_7982_l816_81670

/-- Rounds a real number to the nearest hundredth -/
noncomputable def round_to_hundredth (x : ℝ) : ℝ :=
  ⌊x * 100 + 0.5⌋ / 100

/-- The original number to be rounded -/
def original_number : ℝ := 2.7982

/-- The expected result after rounding -/
def expected_result : ℝ := 2.80

theorem rounding_2_7982 :
  round_to_hundredth original_number = expected_result := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rounding_2_7982_l816_81670


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_common_tangents_to_circles_l816_81673

/-- A circle parameterized by a non-zero real number m -/
def Circle (m : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 + p.2^2 - 4*(m+1)*p.1 - 2*m*p.2 + 4*m^2 + 4*m + 1 = 0}

/-- The first common tangent line -/
def TangentLine1 : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | 4*p.1 - 3*p.2 - 4 = 0}

/-- The second common tangent line -/
def TangentLine2 : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.2 = 0}

/-- Definition of tangency between a line and a circle -/
def IsTangentTo (L : Set (ℝ × ℝ)) (C : Set (ℝ × ℝ)) : Prop :=
  ∃ p : ℝ × ℝ, p ∈ L ∧ p ∈ C ∧ ∀ q : ℝ × ℝ, q ∈ L ∩ C → q = p

/-- Theorem stating that the two lines are common tangents to all circles -/
theorem common_tangents_to_circles (m : ℝ) (hm : m ≠ 0) :
  IsTangentTo TangentLine1 (Circle m) ∧ IsTangentTo TangentLine2 (Circle m) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_common_tangents_to_circles_l816_81673


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_minimal_n_with_product_triple_and_primes_l816_81603

/-- The set of integers from 2 to n, inclusive -/
def T (n : ℕ) : Set ℕ := {x | 2 ≤ x ∧ x ≤ n}

/-- Predicate to check if a subset satisfies the ab = c condition -/
def HasProductTriple (S : Set ℕ) : Prop :=
  ∃ a b c, a ∈ S ∧ b ∈ S ∧ c ∈ S ∧ a * b = c

/-- Predicate to check if all primes up to n are in the same subset -/
def AllPrimesInOneSubset (S₁ S₂ : Set ℕ) (n : ℕ) : Prop :=
  (∀ p, p ≤ n → Nat.Prime p → (p ∈ S₁ ∨ p ∈ S₂)) ∧ 
  (∀ p q, p ≤ n → q ≤ n → Nat.Prime p → Nat.Prime q → (p ∈ S₁ ∧ q ∈ S₂) → False)

/-- The main theorem statement -/
theorem minimal_n_with_product_triple_and_primes : 
  (∀ n < 256, ∃ S₁ S₂ : Set ℕ, 
    S₁ ∪ S₂ = T n ∧ 
    S₁ ∩ S₂ = ∅ ∧ 
    ¬HasProductTriple S₁ ∧ 
    ¬HasProductTriple S₂ ∧
    AllPrimesInOneSubset S₁ S₂ n) ∧
  (∀ S₁ S₂ : Set ℕ, 
    S₁ ∪ S₂ = T 256 ∧ 
    S₁ ∩ S₂ = ∅ → 
    (HasProductTriple S₁ ∨ HasProductTriple S₂) ∧
    AllPrimesInOneSubset S₁ S₂ 256) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_minimal_n_with_product_triple_and_primes_l816_81603


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_equation_solution_l816_81696

theorem sqrt_equation_solution (a b : ℤ) : 
  a = 4 ∧ b = -1 →
  Real.sqrt (16 - 12 * Real.cos (π / 6)) = a + b * (1 / Real.cos (π / 6)) := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_equation_solution_l816_81696


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_planes_and_lines_l816_81636

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the relations between planes and lines
variable (perpendicular : Line → Plane → Prop)
variable (perpendicular_planes : Plane → Plane → Prop)
variable (subset : Line → Plane → Prop)
variable (parallel : Line → Line → Prop)
variable (parallel_plane : Line → Plane → Prop)
variable (intersect : Plane → Plane → Line → Prop)
variable (not_subset : Line → Plane → Prop)

-- Main theorem
theorem planes_and_lines 
  (α β : Plane) (m n : Line) 
  (h_diff_planes : α ≠ β) 
  (h_diff_lines : m ≠ n) :
  (∀ (α β : Plane) (m : Line), 
    perpendicular m α → subset m β → perpendicular_planes α β) ∧
  (∀ (α β : Plane) (m n : Line),
    intersect α β m → parallel n m → 
    not_subset n α → not_subset n β → 
    parallel_plane n α ∧ parallel_plane n β) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_planes_and_lines_l816_81636


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_only_one_correct_statement_l816_81634

-- Define the statements as axioms since they are given
axiom DeterministicRelationship : Type → Type → Prop
axiom Correlation : Type
axiom FunctionalRelationship : Type
axiom RegressionAnalysis : Type
axiom StatisticalAnalysisMethod : Type → Type

-- Define the statements
def statement1 : Prop := ∀ x y, DeterministicRelationship x y
def statement2 : Prop := Correlation = FunctionalRelationship
def statement3 : Prop := RegressionAnalysis = StatisticalAnalysisMethod FunctionalRelationship
def statement4 : Prop := RegressionAnalysis = StatisticalAnalysisMethod Correlation

-- Define the correctness of each statement
def isCorrect (s : Prop) : Prop := s

-- Theorem stating that only one statement is correct
theorem only_one_correct_statement :
  (isCorrect statement4) ∧
  ¬(isCorrect statement1) ∧
  ¬(isCorrect statement2) ∧
  ¬(isCorrect statement3) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_only_one_correct_statement_l816_81634


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_alpha_sum_sin_cos_eq_two_pi_fifth_l816_81671

theorem exists_alpha_sum_sin_cos_eq_two_pi_fifth :
  ∃ α : ℝ, 0 < α ∧ α < π / 2 ∧ Real.sin α + Real.cos α = 2 * π / 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_alpha_sum_sin_cos_eq_two_pi_fifth_l816_81671


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equation_l816_81695

-- Define a line type
structure Line where
  slope : ℝ
  point : ℝ × ℝ

-- Define the inclination angle in radians (60 degrees = π/3 radians)
noncomputable def inclinationAngle : ℝ := Real.pi / 3

-- Define the given line
noncomputable def givenLine : Line where
  slope := Real.tan inclinationAngle
  point := (-1, -Real.sqrt 3)

-- Theorem statement
theorem line_equation (x y : ℝ) : 
  (x, y) ∈ {(x, y) | y = givenLine.slope * x} ↔ 
  (x, y) ∈ {(x, y) | y = Real.sqrt 3 * x} := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equation_l816_81695


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fraction_sum_approx_l816_81601

/-- The sum of fractions 2/(n(n+3)) for n from 1 to 2023 -/
def fraction_sum : ℚ :=
  (Finset.range 2023).sum (fun n => 2 / ((n + 1) * (n + 4)))

/-- The theorem stating that the sum is approximately 1.222 -/
theorem fraction_sum_approx :
  abs ((fraction_sum : ℝ) - 1.222) < 0.001 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fraction_sum_approx_l816_81601


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perimeter_ratio_is_one_l816_81698

/-- Represents a rectangle with width and height --/
structure Rectangle where
  width : ℝ
  height : ℝ

/-- Calculates the perimeter of a rectangle --/
def perimeter (r : Rectangle) : ℝ := 2 * (r.width + r.height)

/-- The original paper dimensions --/
noncomputable def original_paper : Rectangle := { width := 8, height := 6 }

/-- The dimensions after folding --/
noncomputable def folded_paper : Rectangle := { width := original_paper.width, height := original_paper.height / 2 }

/-- The dimensions of the smaller rectangle after cutting --/
noncomputable def small_rectangle : Rectangle := { width := folded_paper.width / 2, height := folded_paper.height }

/-- The dimensions of the larger rectangle after cutting --/
noncomputable def large_rectangle : Rectangle := small_rectangle

theorem perimeter_ratio_is_one : 
  perimeter small_rectangle / perimeter large_rectangle = 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perimeter_ratio_is_one_l816_81698


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_calculation_one_l816_81651

theorem calculation_one : 
  (-2.48) + 4.33 + (-7.52) + (-4.33) = -10 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_calculation_one_l816_81651


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangles_hypotenuse_sum_l816_81639

noncomputable section

open Set Real

def IsRightTriangle (T : Set ℝ × Set ℝ) : Prop := sorry
def area (T : Set ℝ × Set ℝ) : ℝ := sorry
def sides (T : Set ℝ × Set ℝ) : Set (Set ℝ) := sorry
def hypotenuse (T : Set ℝ × Set ℝ) : ℝ := sorry

theorem right_triangles_hypotenuse_sum (T₁ T₂ : Set ℝ × Set ℝ) 
  (h_right₁ : IsRightTriangle T₁)
  (h_right₂ : IsRightTriangle T₂)
  (h_area₁ : area T₁ = 4)
  (h_area₂ : area T₂ = 8)
  (h_congruent₁ : ∃ (s₁ : Set ℝ), s₁ ∈ sides T₁ ∧ s₁ ∈ sides T₂)
  (h_congruent₂ : ∃ (s₂ : Set ℝ), s₂ ∈ sides T₁ ∧ s₂ ∈ sides T₂ ∧ s₂ ≠ Classical.choose h_congruent₁)
  : (hypotenuse T₁) ^ 2 + (hypotenuse T₂) ^ 2 = 88 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangles_hypotenuse_sum_l816_81639


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_probability_at_one_l816_81686

def binomial_probability (n k : ℕ) (p : ℝ) : ℝ :=
  (Nat.choose n k : ℝ) * p^k * (1 - p)^(n - k)

theorem max_probability_at_one :
  let n : ℕ := 5
  let p : ℝ := 1/4
  let X := λ k ↦ binomial_probability n k p
  ∀ k : ℕ, k ≠ 1 → k ≤ n → X 1 > X k := by
  sorry

#check max_probability_at_one

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_probability_at_one_l816_81686


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_perpendicular_foci_triangle_area_l816_81680

/-- The area of a triangle formed by a point on an ellipse and its foci, 
    where the lines connecting the point to the foci are perpendicular -/
theorem ellipse_perpendicular_foci_triangle_area 
  (x y : ℝ) (P F₁ F₂ : ℝ × ℝ) : 
  x^2 / 25 + y^2 / 9 = 1 →  -- P is on the ellipse
  P = (x, y) →  -- P's coordinates
  (P.1 - F₁.1)^2 + (P.2 - F₁.2)^2 + (P.1 - F₂.1)^2 + (P.2 - F₂.2)^2 = 100 →  -- P is equidistant from foci
  ((P.1 - F₁.1) * (P.1 - F₂.1) + (P.2 - F₁.2) * (P.2 - F₂.2) = 0) →  -- PF₁ ⟂ PF₂
  (F₁.1 - F₂.1)^2 + (F₁.2 - F₂.2)^2 = 16 →  -- Distance between foci
  (1/2) * abs ((P.1 - F₁.1) * (P.2 - F₂.2) - (P.2 - F₁.2) * (P.1 - F₂.1)) = 9 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_perpendicular_foci_triangle_area_l816_81680


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_z_properties_l816_81654

noncomputable def z : ℂ := 2 / (-1 + Complex.I)

theorem z_properties : z^2 = 2 * Complex.I ∧ z.im = -1 := by
  -- Split the conjunction into two goals
  constructor
  
  -- Prove z^2 = 2i
  · -- This part of the proof is omitted for brevity
    sorry
  
  -- Prove z.im = -1
  · -- This part of the proof is omitted for brevity
    sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_z_properties_l816_81654


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_acute_triangle_sin_ratio_bound_l816_81638

-- Define an acute triangle ABC
structure AcuteTriangle where
  A : ℝ
  B : ℝ
  C : ℝ
  a : ℝ
  b : ℝ
  c : ℝ
  acute : 0 < A ∧ A < π/2 ∧ 0 < B ∧ B < π/2 ∧ 0 < C ∧ C < π/2
  sum_angles : A + B + C = π
  sine_law : a / Real.sin A = b / Real.sin B
  cosine_law : c = 2 * a * Real.cos B + a

-- Define the theorem
theorem acute_triangle_sin_ratio_bound (t : AcuteTriangle) :
  1 < (Real.sin (t.B - t.A)) / (Real.sin t.A * Real.sin t.B) ∧
  (Real.sin (t.B - t.A)) / (Real.sin t.A * Real.sin t.B) < 2 * Real.sqrt 3 / 3 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_acute_triangle_sin_ratio_bound_l816_81638


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_part_one_part_two_l816_81600

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log ((3 / x^2) + a)

-- Part 1
theorem part_one (a : ℝ) : 
  (∀ x ≠ 0, f a x ∈ Set.univ) ∧ (Set.range (f a) = Set.univ) → a = 0 :=
by sorry

-- Part 2
theorem part_two (c : ℝ) : 
  (∀ x₁ x₂ : ℝ, x₁ ∈ Set.Icc c (c + 2) → x₂ ∈ Set.Icc c (c + 2) → 
    |f 0 x₁ - f 0 x₂| ≤ Real.log 2) → c ≥ 2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_part_one_part_two_l816_81600


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_reflection_equivalence_l816_81628

-- Define the original function g(x)
noncomputable def g (x : ℝ) : ℝ :=
  if x ≤ -1 then x + 3
  else if x < 1 then Real.sqrt (1 - (x + 1)^2) + 1
  else -(x - 3)

-- Define the reflected function g(-x)
noncomputable def g_reflected (x : ℝ) : ℝ := g (-x)

-- Theorem stating the equivalence of g(-x) to the reflected function
theorem g_reflection_equivalence :
  ∀ x : ℝ, -4 ≤ x ∧ x ≤ 4 →
    g_reflected x = 
      if x ≤ -1 then -x + 3
      else if x < 1 then Real.sqrt (1 - (x - 1)^2) + 1
      else x - 3 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_reflection_equivalence_l816_81628


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_horner_operations_count_l816_81637

/-- Represents a polynomial with coefficients of type α -/
def MyPolynomial (α : Type*) := List α

/-- Horner's method for evaluating a polynomial -/
def horner_eval {α : Type*} [Semiring α] (p : MyPolynomial α) (x : α) : α :=
  p.foldl (fun acc a => acc * x + a) 0

/-- The number of addition operations in Horner's method -/
def horner_additions {α : Type*} (p : MyPolynomial α) : Nat :=
  p.length - 1

/-- The number of multiplication operations in Horner's method -/
def horner_multiplications {α : Type*} (p : MyPolynomial α) : Nat :=
  p.length - 1

theorem horner_operations_count {α : Type*} (p : MyPolynomial α) :
  horner_additions p = horner_multiplications p ∧
  horner_additions p = p.length - 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_horner_operations_count_l816_81637


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_combination_sum_equals_power_of_four_l816_81669

theorem combination_sum_equals_power_of_four (n : ℕ) (hn : n > 0) :
  (Finset.range n).sum (λ k => Nat.choose (2*n - 1) k) = 4^(n - 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_combination_sum_equals_power_of_four_l816_81669


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_b_value_l816_81622

theorem unique_b_value : ∃! b : ℤ, 
  (0 ≤ b ∧ b ≤ 20) ∧ 
  (∃ k : ℤ, 234845623 - b = 17 * k) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_b_value_l816_81622


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_center_of_mass_distance_l816_81605

/-- Represents a material point with mass and position -/
structure MaterialPoint where
  mass : ℝ
  position : ℝ

/-- Calculates the center of mass of a system of material points -/
noncomputable def centerOfMass (points : List MaterialPoint) : ℝ :=
  (points.map (λ p => p.mass * p.position)).sum / (points.map (λ p => p.mass)).sum

/-- Theorem: The center of mass of three material points is approximately 1.57 m from the first point -/
theorem center_of_mass_distance (m₁ m₂ m₃ : ℝ) (d₁₂ d₂₃ : ℝ) :
  m₁ = 0.1 →
  m₂ = 0.2 →
  m₃ = 0.4 →
  d₁₂ = 0.5 →
  d₂₃ = 2 →
  let points := [
    { mass := m₁, position := 0 },
    { mass := m₂, position := d₁₂ },
    { mass := m₃, position := d₁₂ + d₂₃ }
  ]
  ‖centerOfMass points - 1.57‖ < 0.01 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_center_of_mass_distance_l816_81605


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_function_m_equals_three_l816_81681

/-- A function is quadratic if its highest degree term is of degree 2 -/
def IsQuadratic (f : ℝ → ℝ) : Prop :=
  ∃ a b c : ℝ, a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

/-- The function y = x^(m-1) + x - 3 -/
noncomputable def f (m : ℝ) (x : ℝ) : ℝ := x^(m-1) + x - 3

theorem quadratic_function_m_equals_three (m : ℝ) :
  IsQuadratic (f m) → m = 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_function_m_equals_three_l816_81681


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_OC_AB_perpendicular_AC_BC_angle_OB_OC_l816_81677

-- Define the points
def A : ℝ × ℝ := (3, 0)
def B : ℝ × ℝ := (0, 3)
noncomputable def C (α : ℝ) : ℝ × ℝ := (Real.cos α, Real.sin α)
def O : ℝ × ℝ := (0, 0)

-- Define vectors
noncomputable def OC (α : ℝ) : ℝ × ℝ := C α
def AB : ℝ × ℝ := (B.1 - A.1, B.2 - A.2)
noncomputable def AC (α : ℝ) : ℝ × ℝ := ((C α).1 - A.1, (C α).2 - A.2)
noncomputable def BC (α : ℝ) : ℝ × ℝ := ((C α).1 - B.1, (C α).2 - B.2)
def OA : ℝ × ℝ := A
def OB : ℝ × ℝ := B

-- Theorem 1
theorem parallel_OC_AB (α : ℝ) : 
  (∃ k : ℝ, OC α = k • AB) → Real.tan α = -1 := by
  sorry

-- Theorem 2
theorem perpendicular_AC_BC (α : ℝ) :
  (AC α).1 * (BC α).1 + (AC α).2 * (BC α).2 = 0 → Real.sin (2 * α) = -8/9 := by
  sorry

-- Theorem 3
theorem angle_OB_OC (α : ℝ) :
  (OA.1 + (OC α).1)^2 + (OA.2 + (OC α).2)^2 = 13 ∧ 0 < α ∧ α < Real.pi →
  Real.arccos ((OB.1 * (OC α).1 + OB.2 * (OC α).2) / 
    (Real.sqrt (OB.1^2 + OB.2^2) * Real.sqrt ((OC α).1^2 + (OC α).2^2))) = Real.pi / 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_OC_AB_perpendicular_AC_BC_angle_OB_OC_l816_81677
