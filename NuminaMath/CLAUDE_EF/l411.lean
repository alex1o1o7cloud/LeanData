import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_cos_value_l411_41169

theorem sin_cos_value (x : ℝ) (h : Real.sin x = 5 * Real.cos x) : Real.sin x * Real.cos x = 5 / 26 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_cos_value_l411_41169


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_beta_value_l411_41113

theorem sin_beta_value (α β : Real) 
  (h1 : Real.tan α = 1/3) 
  (h2 : Real.tan (α + β) = 1/2) : 
  Real.sin β = Real.sqrt 2 / 10 ∨ Real.sin β = -Real.sqrt 2 / 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_beta_value_l411_41113


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_fixed_point_intersection_l411_41167

/-- Definition of the ellipse C -/
noncomputable def ellipse (x y a b : ℝ) : Prop :=
  x^2 / a^2 + y^2 / b^2 = 1

/-- Definition of eccentricity -/
noncomputable def eccentricity (a b : ℝ) : ℝ :=
  Real.sqrt (1 - b^2 / a^2)

/-- Theorem: Any line intersecting the ellipse at two points (excluding vertices) 
    passes through a fixed point -/
theorem ellipse_fixed_point_intersection 
  (a b : ℝ) 
  (h1 : a > b) 
  (h2 : b > 0) 
  (h3 : eccentricity a b = Real.sqrt 3 / 2) 
  (h4 : ellipse (-Real.sqrt 3) (1/2) a b) :
  ∃ (x₀ y₀ : ℝ), ∀ (P Q : ℝ × ℝ),
    P ≠ Q ∧ 
    ellipse P.1 P.2 a b ∧ 
    ellipse Q.1 Q.2 a b ∧ 
    P ≠ (0, b) ∧ P ≠ (0, -b) ∧ 
    Q ≠ (0, b) ∧ Q ≠ (0, -b) →
    ∃ (t : ℝ), t ≠ 0 ∧ 
      (y₀ - P.2) * (Q.1 - P.1) = (Q.2 - P.2) * (x₀ - P.1) ∧
      x₀ = 0 ∧ y₀ = 1/2 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_fixed_point_intersection_l411_41167


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_erdos_hajnal_theorem_l411_41114

open Set

-- Define a simple graph structure
structure Graph (V : Type) where
  adj : V → V → Prop

variable {V : Type}
variable (G : Graph V)

-- Define what it means for a vertex to be joined to infinitely many elements of a set
def joinedToInfinitelyMany (G : Graph V) (v : V) (A : Set V) : Prop :=
  ¬Finite {a ∈ A | G.adj v a}

-- The main theorem
theorem erdos_hajnal_theorem (G : Graph V) 
    (h : ∀ (A : Set V), Countable A → ∃ p, p ∉ A ∧ joinedToInfinitelyMany G p A) :
    ∃ (A : Set V), Countable A ∧ ¬Countable {p | p ∉ A ∧ joinedToInfinitelyMany G p A} := by
  sorry  -- The proof would go here


end NUMINAMATH_CALUDE_ERRORFEEDBACK_erdos_hajnal_theorem_l411_41114


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_value_at_negative_one_l411_41181

def is_monic (P : ℝ → ℝ) : Prop :=
  ∃ Q : ℝ → ℝ, P = λ x ↦ x^2023 + Q x

def satisfies_functional_equation (P : ℝ → ℝ) : Prop :=
  ∀ k : ℕ, 1 ≤ k ∧ k ≤ 2023 → P k = k^2023 * P (1 - 1/k)

theorem polynomial_value_at_negative_one
  (P : ℝ → ℝ)
  (h_monic : is_monic P)
  (h_func_eq : satisfies_functional_equation P) :
  P (-1) = 0 := by
  sorry

#check polynomial_value_at_negative_one

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_value_at_negative_one_l411_41181


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_distance_point_B_l411_41192

-- Define the polar coordinate system
structure PolarPoint where
  r : ℝ
  θ : ℝ

-- Define the line in polar form
def polarLine (ρ θ : ℝ) : Prop := ρ * Real.cos θ + ρ * Real.sin θ = 0

-- Define the distance between two points in polar coordinates
noncomputable def polarDistance (p1 p2 : PolarPoint) : ℝ :=
  Real.sqrt (p1.r^2 + p2.r^2 - 2 * p1.r * p2.r * Real.cos (p2.θ - p1.θ))

theorem shortest_distance_point_B (A B : PolarPoint) (h1 : A.r = 2) (h2 : A.θ = Real.pi / 2)
    (h3 : polarLine B.r B.θ) (h4 : 0 ≤ B.θ ∧ B.θ ≤ 2 * Real.pi) :
  (∀ C : PolarPoint, polarLine C.r C.θ → polarDistance A C ≥ polarDistance A B) →
  B.r = Real.sqrt 2 ∧ B.θ = 3 * Real.pi / 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_distance_point_B_l411_41192


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_sum_positive_max_l411_41162

/-- An arithmetic sequence with a positive first term -/
structure ArithmeticSequence where
  a : ℕ → ℝ
  a₁_pos : a 1 > 0
  is_arithmetic : ∀ n : ℕ, a (n + 1) - a n = a 2 - a 1

/-- Sum of first n terms of an arithmetic sequence -/
noncomputable def S_n (seq : ArithmeticSequence) (n : ℕ) : ℝ :=
  (n : ℝ) * (seq.a 1 + seq.a n) / 2

/-- Statement of the problem -/
theorem arithmetic_sequence_sum_positive_max (seq : ArithmeticSequence) 
  (h₁ : seq.a 1006 + seq.a 1007 = 2012)
  (h₂ : seq.a 1006 * seq.a 1007 = -2011) :
  (∃ n : ℕ, S_n seq n > 0 ∧ ∀ m : ℕ, m > n → S_n seq m ≤ 0) →
  ∃ n : ℕ, S_n seq n > 0 ∧ (∀ m : ℕ, m > n → S_n seq m ≤ 0) ∧ n = 2011 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_sum_positive_max_l411_41162


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_special_parallelogram_area_l411_41125

/-- A parallelogram with two special circles -/
structure SpecialParallelogram where
  -- The inscribed circle
  R : ℝ
  -- The second circle
  r : ℝ
  -- Distance between tangency points
  d : ℝ
  -- Conditions
  inscribed : R = 3
  tangent_distance : d = 3
  circles_touch : r > 0

/-- The area of the special parallelogram -/
noncomputable def area (p : SpecialParallelogram) : ℝ :=
  75 / 2

/-- Theorem: The area of the special parallelogram is 75/2 -/
theorem special_parallelogram_area (p : SpecialParallelogram) :
  area p = 75 / 2 := by
  -- Unfold the definition of area
  unfold area
  -- The definition directly gives us the result
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_special_parallelogram_area_l411_41125


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_digit_number_divisibility_l411_41166

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

theorem two_digit_number_divisibility (N : ℕ) 
  (h_two_digit : is_two_digit N) 
  (h_four_false : (if N % 3 = 0 then 0 else 1) + (if N % 4 = 0 then 0 else 1) + 
                  (if N % 5 = 0 then 0 else 1) + (if N % 9 = 0 then 0 else 1) + 
                  (if N % 10 = 0 then 0 else 1) + (if N % 15 = 0 then 0 else 1) + 
                  (if N % 18 = 0 then 0 else 1) + (if N % 30 = 0 then 0 else 1) = 4) : 
  N = 36 ∨ N = 45 ∨ N = 72 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_digit_number_divisibility_l411_41166


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_real_part_product_complex_exp_l411_41101

theorem real_part_product_complex_exp (α β : ℝ) : 
  (Complex.exp (α * I) * Complex.exp (β * I)).re = Real.cos (α + β) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_real_part_product_complex_exp_l411_41101


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_variance_of_data_l411_41120

noncomputable def data : List ℝ := [8, 10, 9, 12, 11]

noncomputable def mean (xs : List ℝ) : ℝ := (xs.sum) / xs.length

noncomputable def variance (xs : List ℝ) : ℝ :=
  (xs.map (fun x => (x - mean xs) ^ 2)).sum / xs.length

theorem variance_of_data : variance data = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_variance_of_data_l411_41120


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_PAB_value_l411_41122

/-- Line l with parametric equations x = 1 + t and y = 1 - t -/
noncomputable def line_l (t : ℝ) : ℝ × ℝ := (1 + t, 1 - t)

/-- Curve C with polar equation ρ²(5 - 4cos(2θ)) = 9 -/
noncomputable def curve_C (θ : ℝ) : ℝ := Real.sqrt (9 / (5 - 4 * Real.cos (2 * θ)))

/-- Point P with polar coordinates (√2, 3π/4) -/
noncomputable def point_P : ℝ × ℝ := (Real.sqrt 2, 3 * Real.pi / 4)

/-- The area of triangle PAB, where A and B are intersection points of line l and curve C -/
noncomputable def area_PAB : ℝ := 3 * Real.sqrt 6 / 5

/-- Theorem stating that the area of triangle PAB is 3√6/5 -/
theorem area_PAB_value : area_PAB = 3 * Real.sqrt 6 / 5 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_PAB_value_l411_41122


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_proposition_l411_41144

theorem negation_of_proposition :
  (¬(∀ x : ℝ, (3 : ℝ)^x > 0)) ↔ (∃ x : ℝ, (3 : ℝ)^x ≤ 0) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_proposition_l411_41144


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_functional_equation_solution_l411_41176

theorem functional_equation_solution :
  ∀ (a : ℝ), (∃ (f : ℝ → ℝ), ∀ (x y : ℝ), f (x + f y) = f x + a * ⌊y⌋) → a = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_functional_equation_solution_l411_41176


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_selling_price_is_10_l411_41112

/-- The minimum selling price per kilogram to avoid losses -/
noncomputable def min_selling_price (total_cost : ℝ) (total_weight : ℝ) (spoilage_rate : ℝ) : ℝ :=
  total_cost / (total_weight * (1 - spoilage_rate))

/-- Proof that the minimum selling price is 10 yuan per kilogram -/
theorem min_selling_price_is_10 :
  let total_cost : ℝ := 760
  let total_weight : ℝ := 80
  let spoilage_rate : ℝ := 0.05
  min_selling_price total_cost total_weight spoilage_rate = 10 := by
  sorry

-- Remove the #eval statement as it's not compatible with noncomputable definitions

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_selling_price_is_10_l411_41112


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_throws_for_repeat_is_22_l411_41172

/-- Represents a fair six-sided die -/
def Die : Type := Fin 6

/-- Represents a throw of four dice -/
def ThrowResult := Fin 4 → Die

/-- The sum of a throw result -/
def sum_throw (t : ThrowResult) : ℕ :=
  (t 0).val + (t 1).val + (t 2).val + (t 3).val + 4

/-- The set of all possible sums from throwing four dice -/
def possible_sums : Finset ℕ :=
  Finset.range 21 |>.image (· + 4)

/-- The minimum number of throws needed to guarantee a repeated sum -/
def min_throws_for_repeat : ℕ := possible_sums.card + 1

theorem min_throws_for_repeat_is_22 :
  min_throws_for_repeat = 22 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_throws_for_repeat_is_22_l411_41172


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_three_integers_divisible_by_three_l411_41165

theorem three_integers_divisible_by_three (a b c d e : ℤ) :
  ∃ (x y z : ℤ), x ∈ ({a, b, c, d, e} : Set ℤ) ∧ 
                  y ∈ ({a, b, c, d, e} : Set ℤ) ∧ 
                  z ∈ ({a, b, c, d, e} : Set ℤ) ∧
                  x ≠ y ∧ y ≠ z ∧ x ≠ z ∧
                  (x + y + z) % 3 = 0 :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_three_integers_divisible_by_three_l411_41165


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_dot_product_on_ellipse_l411_41151

/-- The ellipse equation -/
def is_on_ellipse (x y : ℝ) : Prop := x^2/4 + y^2/3 = 1

/-- The center of the ellipse -/
def O : ℝ × ℝ := (0, 0)

/-- The left focus of the ellipse -/
def F : ℝ × ℝ := (-1, 0)

/-- Any point on the ellipse -/
def P (x y : ℝ) : ℝ × ℝ := (x, y)

/-- The dot product of OP and FP -/
def dot_product (x y : ℝ) : ℝ := x^2 + x + y^2

theorem max_dot_product_on_ellipse :
  ∀ x y : ℝ, is_on_ellipse x y →
  dot_product x y ≤ 6 :=
by
  sorry

#check max_dot_product_on_ellipse

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_dot_product_on_ellipse_l411_41151


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_difference_comparison_l411_41138

theorem sqrt_difference_comparison : Real.sqrt 3 - Real.sqrt 2 > Real.sqrt 6 - Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_difference_comparison_l411_41138


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_equation_l411_41163

-- Define the hyperbola
noncomputable def hyperbola (a b : ℝ) (x y : ℝ) : Prop := x^2 / a^2 - y^2 / b^2 = 1

-- Define the ellipse
noncomputable def ellipse (x y : ℝ) : Prop := x^2 / 16 + y^2 / 9 = 1

-- Define the eccentricity of an ellipse
noncomputable def ellipse_eccentricity : ℝ := Real.sqrt 7 / 4

-- Define the eccentricity of a hyperbola
noncomputable def hyperbola_eccentricity (a c : ℝ) : ℝ := c / a

-- Theorem statement
theorem hyperbola_equation (a b : ℝ) (h1 : a > 0) (h2 : b > 0) :
  (∃ c : ℝ, c > 0 ∧ 
    (∀ x y : ℝ, hyperbola a b x y ↔ x^2 - y^2 = c^2) ∧
    (∀ x y : ℝ, ellipse x y ↔ x^2 + y^2 = c^2) ∧
    hyperbola_eccentricity a c = 2 * ellipse_eccentricity) →
  (∀ x y : ℝ, hyperbola a b x y ↔ x^2 / 4 - y^2 / 3 = 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_equation_l411_41163


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_x_l411_41198

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := (4 : ℝ)^x - 3 * (2 : ℝ)^x + 3

-- State the theorem
theorem domain_of_x (h : Set.range f = Set.Icc 1 7) :
  {x : ℝ | f x ∈ Set.Icc 1 7} = Set.Iic 0 ∪ Set.Icc 1 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_x_l411_41198


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cubes_equal_prime_power_l411_41106

theorem cubes_equal_prime_power (a b p n : ℕ) (h_pos_a : a > 0) (h_pos_b : b > 0) (h_pos_p : p > 0) (h_pos_n : n > 0) (h_prime : Nat.Prime p) :
  a^3 + b^3 = p^n ↔
  (∃ k : ℕ, (a = 2^k ∧ b = 2^k ∧ p = 2 ∧ n = 3*k + 1) ∨
            (a = 3^k ∧ b = 2 * 3^k ∧ p = 3 ∧ n = 3*k + 2) ∨
            (a = 2 * 3^k ∧ b = 3^k ∧ p = 3 ∧ n = 3*k + 2)) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cubes_equal_prime_power_l411_41106


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_distance_graph_l411_41196

-- Define a trapezoid
structure Trapezoid :=
  (A B C D : ℝ × ℝ)
  (parallel_AB_CD : (A.2 - B.2) / (A.1 - B.1) = (C.2 - D.2) / (C.1 - D.1))
  (AB_less_than_CD : ((A.1 - B.1)^2 + (A.2 - B.2)^2).sqrt < ((C.1 - D.1)^2 + (C.2 - D.2)^2).sqrt)
  (AD_not_equal_BC : ((A.1 - D.1)^2 + (A.2 - D.2)^2).sqrt ≠ ((B.1 - C.1)^2 + (B.2 - C.2)^2).sqrt)

-- Define a function to represent the distance from A to a point on the trapezoid
noncomputable def distance_from_A (t : Trapezoid) (θ : ℝ) : ℝ :=
  sorry -- This would be the actual distance function

-- State the theorem
theorem trapezoid_distance_graph (t : Trapezoid) :
  ∃ θ_peak : ℝ, ∀ θ : ℝ, θ ≠ θ_peak →
    (distance_from_A t θ < distance_from_A t θ_peak ∨
     (θ < θ_peak ∧ distance_from_A t θ < distance_from_A t θ_peak) ∨
     (θ > θ_peak ∧ distance_from_A t θ < distance_from_A t θ_peak)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_distance_graph_l411_41196


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_line_distance_bound_l411_41133

/-- The hyperbola equation -/
def hyperbola (x y : ℝ) : Prop := x^2 - y^2/4 = 1

/-- The line equation -/
def line (x y : ℝ) : Prop := y = 2*x + 5

/-- Distance from a point to a line -/
noncomputable def distance_to_line (x y : ℝ) : ℝ :=
  |2*x - y + 5| / Real.sqrt 5

theorem hyperbola_line_distance_bound :
  ∃ (m : ℝ), ∀ (x y : ℝ),
    hyperbola x y → x > 0 →
    distance_to_line x y > m ∧
    ∀ (m' : ℝ), (∀ (x' y' : ℝ), hyperbola x' y' → x' > 0 → 
      distance_to_line x' y' > m') → m' ≤ m ∧ m = Real.sqrt 5 := by
  sorry

#check hyperbola_line_distance_bound

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_line_distance_bound_l411_41133


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_insufficient_funds_l411_41103

/-- The cost of one шалтай in kopecks -/
noncomputable def cost_shaltay : ℝ := sorry

/-- The cost of one болтай in kopecks -/
noncomputable def cost_boltay : ℝ := sorry

/-- The relationship between the costs of шалтаев and болтаев -/
axiom cost_relation : 126 * cost_boltay > 175 * cost_shaltay ∧ 175 * cost_shaltay > 125 * cost_boltay

/-- Theorem stating that 80 kopecks is not sufficient to buy 3 шалтаев and 1 болтая -/
theorem insufficient_funds : 3 * cost_shaltay + cost_boltay > 80 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_insufficient_funds_l411_41103


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_to_triangle_area_l411_41180

/-- Represents a part of the square's outline -/
structure OutlinePart where
  length : ℝ

/-- Represents a triangle formed from the square's outline parts -/
structure Triangle where
  side1 : ℝ
  side2 : ℝ
  side3 : ℝ

/-- Function to cut the square's outline into parts -/
noncomputable def cutSquareOutline (sideLength : ℝ) : List OutlinePart :=
  sorry

/-- Function to arrange outline parts into a triangle -/
noncomputable def arrangeTriangle (parts : List OutlinePart) : Triangle :=
  sorry

/-- Calculate the area of a triangle given its sides -/
noncomputable def triangleArea (t : Triangle) : ℝ :=
  sorry

/-- Main theorem: It's possible to form a triangle with area 2/3 from a unit square's outline -/
theorem square_to_triangle_area (sideLength : ℝ) (h : sideLength = 1) :
  ∃ (t : Triangle), triangleArea t = 2/3 ∧ t = arrangeTriangle (cutSquareOutline sideLength) := by
  sorry

#check square_to_triangle_area

end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_to_triangle_area_l411_41180


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_70_cents_is_three_fifths_l411_41130

/-- Represents the types of coins in John's pocket -/
inductive Coin
  | twenty  : Coin
  | fifty   : Coin
deriving DecidableEq

/-- The set of coins in John's pocket -/
def pocket : Multiset Coin :=
  Multiset.replicate 2 Coin.twenty + Multiset.replicate 3 Coin.fifty

/-- The value of a coin in cents -/
def coin_value : Coin → ℕ
  | Coin.twenty => 20
  | Coin.fifty  => 50

/-- The probability of selecting two coins with a total value of 70 cents -/
noncomputable def prob_70_cents : ℚ :=
  let total_ways := (pocket.card.choose 2 : ℚ)
  let favorable_ways := (pocket.count Coin.twenty * pocket.count Coin.fifty : ℚ)
  favorable_ways / total_ways

/-- The main theorem to be proved -/
theorem prob_70_cents_is_three_fifths : prob_70_cents = 3/5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_70_cents_is_three_fifths_l411_41130


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_octagon_extended_sides_angle_l411_41199

-- Define a regular octagon
structure RegularOctagon where
  vertices : Fin 8 → ℝ × ℝ
  is_regular : ∀ i j : Fin 8, 
    dist (vertices i) (vertices ((i + 1) % 8)) = dist (vertices j) (vertices ((j + 1) % 8))

-- Define the extension of two sides
noncomputable def extend_sides (o : RegularOctagon) : ℝ × ℝ := sorry

-- Define angle function
noncomputable def angle (a b c : ℝ × ℝ) : ℝ := sorry

-- Theorem statement
theorem regular_octagon_extended_sides_angle (o : RegularOctagon) :
  let q := extend_sides o
  let a := o.vertices 0
  let h := o.vertices 6
  angle a q h = 90 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_octagon_extended_sides_angle_l411_41199


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_clock_angle_at_3_45_l411_41115

/-- The number of divisions on a clock face -/
def clock_divisions : ℕ := 12

/-- The number of degrees in a full circle -/
def full_circle_degrees : ℝ := 360

/-- The number of minutes in one clock division -/
def minutes_per_division : ℕ := 60

/-- The hour on the clock (3 for 3:45) -/
def hour : ℕ := 3

/-- The minute on the clock (45 for 3:45) -/
def minute : ℕ := 45

/-- Calculates the acute angle between the hour and minute hands at a given time -/
noncomputable def clock_angle (h : ℕ) (m : ℕ) : ℝ :=
  let degrees_per_division := full_circle_degrees / clock_divisions
  let hour_angle := (h % 12 : ℝ) * degrees_per_division + (m : ℝ) * degrees_per_division / minutes_per_division
  let minute_angle := (m : ℝ) * degrees_per_division / (minutes_per_division / 4)
  min (abs (hour_angle - minute_angle)) (full_circle_degrees - abs (hour_angle - minute_angle))

theorem clock_angle_at_3_45 :
  clock_angle hour minute = 22.5 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_clock_angle_at_3_45_l411_41115


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infinitely_many_composites_l411_41117

/-- Represents a digit from 0 to 8 -/
def Digit := Fin 9

/-- An infinite sequence of digits -/
def DigitSequence := ℕ → Digit

/-- Converts a Digit to a natural number -/
def digitToNat (d : Digit) : ℕ := d.val

/-- Constructs a number from the first n digits of a sequence -/
def constructNumber (s : DigitSequence) (n : ℕ) : ℕ :=
  (List.range n).foldl (fun acc i => 10 * acc + digitToNat (s i.succ)) (digitToNat (s 0))

/-- A number is composite if it's not prime and greater than 1 -/
def isComposite (n : ℕ) : Prop := n > 1 ∧ ¬ Nat.Prime n

theorem infinitely_many_composites (s : DigitSequence) :
  {n : ℕ | isComposite (constructNumber s n)}.Infinite :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_infinitely_many_composites_l411_41117


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l411_41121

noncomputable section

/-- Definition of the ellipse C -/
def ellipse_C (x y a : ℝ) : Prop := x^2 / a^2 + y^2 / 3 = 1

/-- Definition of eccentricity -/
def eccentricity (a c : ℝ) : ℝ := c / a

theorem ellipse_properties (a : ℝ) (h1 : a > Real.sqrt 3) 
  (h2 : eccentricity a (Real.sqrt (a^2 - 3)) = Real.sqrt 2 / 2) :
  ∃ (max_area : ℝ) (l_eq : ℝ → ℝ),
    /- 1. The value of a is √6 -/
    a = Real.sqrt 6 ∧
    /- 2. The maximum area of triangle ABD -/
    max_area = 3 + 3 * Real.sqrt 2 / 2 ∧
    /- 3. The equation of line l that produces the maximum area -/
    l_eq = λ x ↦ -Real.sqrt 3 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l411_41121


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l411_41194

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
def Triangle (a b c : ℝ) (A B C : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ 
  0 < A ∧ A < Real.pi ∧ 
  0 < B ∧ B < Real.pi ∧ 
  0 < C ∧ C < Real.pi ∧
  A + B + C = Real.pi

theorem triangle_properties 
  (a b c : ℝ) (A B C : ℝ) 
  (h_triangle : Triangle a b c A B C)
  (h_eq1 : (a^2 + c^2 - b^2) / Real.cos B = 4)
  (h_eq2 : (2*b*Real.cos C - 2*c*Real.cos B) / (b*Real.cos C + c*Real.cos B) - c/a = 2) :
  a * c = 2 ∧ 
  (1/2) * a * c * Real.sin B = Real.sqrt 15 / 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l411_41194


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_has_vertical_asymptotes_l411_41149

/-- The function f(x) = (3x + 5) / (x^2 - 5x + 6) -/
noncomputable def f (x : ℝ) : ℝ := (3 * x + 5) / (x^2 - 5 * x + 6)

/-- The set of x-values where f(x) has vertical asymptotes -/
def vertical_asymptotes : Set ℝ := {2, 3}

/-- Theorem stating that f has vertical asymptotes at x = 2 and x = 3 -/
theorem f_has_vertical_asymptotes :
  ∀ x ∈ vertical_asymptotes, 
    (∀ ε > 0, ∃ δ > 0, ∀ y ∈ Set.Ioo (x - δ) (x + δ) \ {x}, |f y| > 1/ε) ∧
    (∀ y ≠ x, f y ≠ 0) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_has_vertical_asymptotes_l411_41149


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_eliot_balance_l411_41129

/-- Represents the current account balance of a person -/
structure AccountBalance where
  balance : ℝ

/-- Al's current account balance -/
noncomputable def Al : AccountBalance := ⟨0⟩

/-- Eliot's current account balance -/
noncomputable def Eliot : AccountBalance := ⟨0⟩

/-- Zoe's current account balance -/
noncomputable def Zoe : AccountBalance := ⟨0⟩

/-- Al has more money than Eliot -/
axiom al_more_than_eliot : Al.balance > Eliot.balance

/-- Zoe has more money than Al -/
axiom zoe_more_than_al : Zoe.balance > Al.balance

/-- The difference between Al's and Eliot's accounts is 1/12 of the sum of their two accounts -/
axiom al_eliot_difference : Al.balance - Eliot.balance = (1 / 12) * (Al.balance + Eliot.balance)

/-- The difference between Zoe's and Al's accounts is 1/10 of the sum of their two accounts -/
axiom zoe_al_difference : Zoe.balance - Al.balance = (1 / 10) * (Zoe.balance + Al.balance)

/-- After increases, Al would have $20 more than Eliot -/
axiom al_20_more_than_eliot : 1.1 * Al.balance = 1.2 * Eliot.balance + 20

/-- After increases, Al would have $30 less than Zoe -/
axiom al_30_less_than_zoe : 1.1 * Al.balance + 30 = 1.15 * Zoe.balance

/-- Eliot's current account balance is $86.96 -/
theorem eliot_balance : Eliot.balance = 86.96 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_eliot_balance_l411_41129


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_abs_v_equals_five_l411_41156

-- Use the built-in complex number type
open Complex

-- Define the absolute value (magnitude) for complex numbers
noncomputable def abs_complex (z : ℂ) : ℝ :=
  Complex.abs z

-- Define multiplication for complex numbers
def complex_mul (z w : ℂ) : ℂ :=
  z * w

-- State the theorem
theorem abs_v_equals_five (u v : ℂ) 
  (h1 : complex_mul u v = Complex.mk 20 (-15))
  (h2 : abs_complex u = 5) : 
  abs_complex v = 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_abs_v_equals_five_l411_41156


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monochromatic_triangle_exists_l411_41116

/-- A color of a segment -/
inductive Color
| Blue
| Red

/-- A point in the plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- A segment between two points -/
structure Segment where
  p1 : Point
  p2 : Point
  color : Color

/-- A configuration of 6 points and their segments -/
structure Configuration where
  points : Fin 6 → Point
  segments : (i j : Fin 6) → i < j → Segment

/-- A triangle in the configuration -/
structure Triangle where
  p1 : Fin 6
  p2 : Fin 6
  p3 : Fin 6
  h12 : p1 < p2
  h23 : p2 < p3
  h13 : p1 < p3

/-- The theorem statement -/
theorem monochromatic_triangle_exists (config : Configuration) :
  ∃ (t : Triangle), 
    (config.segments t.p1 t.p2 t.h12).color = 
    (config.segments t.p2 t.p3 t.h23).color ∧
    (config.segments t.p2 t.p3 t.h23).color = 
    (config.segments t.p1 t.p3 t.h13).color := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_monochromatic_triangle_exists_l411_41116


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_k_is_perfect_square_l411_41147

theorem k_is_perfect_square (m n : ℕ) (k : ℕ) 
  (h : k = (m+n)^2 / (4*m*(m-n)^2 + 4)) : 
  ∃ (a : ℕ), k = a^2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_k_is_perfect_square_l411_41147


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l411_41164

noncomputable def f (a k : ℝ) (x : ℝ) : ℝ := k * a^x - a^(-x)

noncomputable def g (a m : ℝ) (x : ℝ) : ℝ := a^(2*x) + a^(-2*x) - 2*m*(f a 1 x)

theorem problem_solution (a : ℝ) (h_a_pos : a > 0) (h_a_neq_1 : a ≠ 1) :
  (∀ x : ℝ, f a 1 (-x) = -(f a 1 x)) →
  (f a 1 1 > 0) →
  {x : ℝ | f a 1 (x^2 + 2*x) + f a 1 (x - 4) > 0} = {x : ℝ | x > 1 ∨ x < -4} ∧
  (f a 1 1 = 0) →
  (∃ m : ℝ, (∀ x : ℝ, x ≥ 1 → g a m x ≥ -2) ∧ 
   (∃ x₀ : ℝ, x₀ ≥ 1 ∧ g a m x₀ = -2)) →
  m = 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l411_41164


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_space_geometry_relationships_l411_41175

-- Define the types for lines and planes
structure Line : Type
structure Plane : Type

-- Define the relationships between lines and planes
axiom perpendicular : Line → Plane → Prop
axiom parallel : Line → Plane → Prop
axiom contains : Plane → Line → Prop
axiom perpendicular_lines : Line → Line → Prop
axiom parallel_planes : Plane → Plane → Prop
axiom perpendicular_planes : Plane → Plane → Prop

-- State the theorem
theorem space_geometry_relationships 
  (m n : Line) (α β : Plane) 
  (h_distinct_lines : m ≠ n) 
  (h_distinct_planes : α ≠ β) :
  (perpendicular m α ∧ perpendicular n β ∧ perpendicular_lines m n → perpendicular_planes α β) ∧
  (contains β m ∧ parallel_planes α β → parallel m α) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_space_geometry_relationships_l411_41175


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_movie_ratio_change_l411_41126

/-- Represents the ratio of DVD to Blu-ray movies -/
structure MovieRatio where
  dvd : ℕ
  bluray : ℕ
deriving Repr

/-- Calculates the new ratio after returning Blu-ray movies -/
def new_ratio (initial_ratio : MovieRatio) (total_movies : ℕ) (returned : ℕ) : MovieRatio :=
  let x : ℕ := total_movies / (initial_ratio.dvd + initial_ratio.bluray)
  let dvd_count : ℕ := initial_ratio.dvd * x
  let bluray_count : ℕ := initial_ratio.bluray * x - returned
  ⟨dvd_count, bluray_count⟩

/-- Theorem stating the new ratio after returning Blu-ray movies -/
theorem movie_ratio_change 
  (initial_ratio : MovieRatio) 
  (total_movies : ℕ) 
  (returned : ℕ) 
  (h1 : initial_ratio = ⟨17, 4⟩) 
  (h2 : total_movies = 378) 
  (h3 : returned = 4) : 
  new_ratio initial_ratio total_movies returned = ⟨306, 68⟩ := by
  sorry

-- Use #eval to display the result
#eval new_ratio ⟨17, 4⟩ 378 4

end NUMINAMATH_CALUDE_ERRORFEEDBACK_movie_ratio_change_l411_41126


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_charts_brought_by_associate_l411_41143

/-- Represents the number of associate professors -/
def associate_professors : ℕ := sorry

/-- Represents the number of assistant professors -/
def assistant_professors : ℕ := sorry

/-- Represents the number of charts each associate professor brings -/
def charts_per_associate : ℕ := sorry

/-- The total number of people present is 5 -/
axiom total_people : associate_professors + assistant_professors = 5

/-- The total number of pencils brought is 10 -/
axiom total_pencils : 2 * associate_professors + assistant_professors = 10

/-- The total number of charts brought is 5 -/
axiom total_charts : charts_per_associate * associate_professors + 2 * assistant_professors = 5

theorem charts_brought_by_associate : charts_per_associate = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_charts_brought_by_associate_l411_41143


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_right_angle_perpendicular_l411_41174

/-- Two lines in a plane -/
structure Line where

/-- An angle formed by two intersecting lines -/
def Angle (l1 l2 : Line) : Type := sorry

/-- Perpendicularity of two lines -/
def Perpendicular (l1 l2 : Line) : Prop := sorry

/-- The measure of an angle in degrees -/
def AngleMeasure {l1 l2 : Line} (a : Angle l1 l2) : ℝ := sorry

/-- If one of the angles formed by the intersection of two lines is 90°, then the lines are perpendicular -/
theorem intersection_right_angle_perpendicular (l1 l2 : Line) :
  (∃ a : Angle l1 l2, AngleMeasure a = 90) →
  Perpendicular l1 l2 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_right_angle_perpendicular_l411_41174


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_focus_distance_l411_41140

structure Hyperbola where
  a : ℝ
  b : ℝ
  equation : (x y : ℝ) → Prop := λ x y => x^2 / a^2 - y^2 / b^2 = 1

noncomputable def right_focus (h : Hyperbola) : ℝ × ℝ := (Real.sqrt (h.a^2 + h.b^2), 0)
noncomputable def left_focus (h : Hyperbola) : ℝ × ℝ := (-Real.sqrt (h.a^2 + h.b^2), 0)

noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

theorem hyperbola_focus_distance (h : Hyperbola) (p : ℝ × ℝ) :
  h.equation p.1 p.2 →
  distance p (right_focus h) = 8 →
  (distance p (left_focus h) = 4 ∨ distance p (left_focus h) = 12) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_focus_distance_l411_41140


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_tangent_to_line_l411_41104

-- Define the circle
def myCircle (x y : ℝ) : Prop := (x - 2)^2 + y^2 = 2

-- Define the line y = x
def myLine (x y : ℝ) : Prop := y = x

-- Theorem statement
theorem circle_tangent_to_line :
  -- The circle's center is on the x-axis (implied by the equation)
  -- The circle is tangent to the line y=x at the point (1,1)
  (myCircle 1 1 ∧ myLine 1 1) →
  -- For any point (x,y) on the circle
  ∀ x y : ℝ, myCircle x y →
  -- The distance from (x,y) to (1,1) is greater than or equal to
  -- the distance from any point on the line to (1,1),
  -- with equality only when (x,y) = (1,1)
  ∀ a b : ℝ, myLine a b →
  (x - 1)^2 + (y - 1)^2 ≥ (a - 1)^2 + (b - 1)^2 ∧
  ((x - 1)^2 + (y - 1)^2 = (a - 1)^2 + (b - 1)^2 → x = 1 ∧ y = 1) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_tangent_to_line_l411_41104


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integer_points_collinearity_l411_41107

/-- A point with integer coordinates -/
structure IntPoint where
  x : ℤ
  y : ℤ

/-- A convex polygon on a plane -/
structure ConvexPolygon where
  points : Finset IntPoint
  is_convex : sorry -- We don't need to define the convexity property for this statement

/-- Definition of collinearity for integer points -/
def collinear (p q r : IntPoint) : Prop :=
  (q.x - p.x) * (r.y - p.y) = (r.x - p.x) * (q.y - p.y)

theorem integer_points_collinearity (m : ℕ) (polygon : ConvexPolygon) :
  (polygon.points.card ≥ m^2 + 1) →
  ∃ (line : Finset IntPoint), (line.card = m + 1) ∧
    (line ⊆ polygon.points) ∧
    (∀ (p q r : IntPoint), p ∈ line → q ∈ line → r ∈ line → collinear p q r) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_integer_points_collinearity_l411_41107


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_area_and_equation_l411_41142

/-- A line that passes through (3,2) and intersects positive x and y axes -/
structure IntersectingLine where
  /-- x-intercept -/
  a : ℝ
  /-- y-intercept -/
  b : ℝ
  /-- a and b are positive -/
  a_pos : a > 0
  b_pos : b > 0
  /-- Line passes through (3,2) -/
  passes_through : 3 / a + 2 / b = 1

/-- Area of triangle AOB -/
noncomputable def triangleArea (l : IntersectingLine) : ℝ :=
  l.a * l.b / 2

/-- Theorem stating the minimum area and corresponding equation -/
theorem min_area_and_equation :
  ∃ (l : IntersectingLine),
    (∀ (l' : IntersectingLine), triangleArea l ≤ triangleArea l') ∧
    triangleArea l = 12 ∧
    (4 * 3 + 6 * 2 - 24 = 0) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_area_and_equation_l411_41142


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_with_18_divisors_l411_41173

/-- The number of positive divisors of a positive integer -/
def num_divisors (n : ℕ+) : ℕ :=
  (Finset.filter (· ∣ n.val) (Finset.range n.val.succ)).card

/-- A positive integer has exactly 18 positive divisors -/
def has_18_divisors (n : ℕ+) : Prop :=
  num_divisors n = 18

/-- 180 has exactly 18 positive divisors -/
axiom divisors_of_180 : has_18_divisors 180

/-- For any positive integer less than 180, it does not have exactly 18 positive divisors -/
axiom smaller_than_180 (m : ℕ+) : m < 180 → ¬(has_18_divisors m)

theorem smallest_with_18_divisors :
  ∀ n : ℕ+, has_18_divisors n → n ≥ 180 := by
  sorry

#check smallest_with_18_divisors

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_with_18_divisors_l411_41173


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_james_toy_profit_l411_41184

/-- James' toy selling problem -/
theorem james_toy_profit :
  let initial_toys : ℕ := 200
  let sell_percentage : ℚ := 80 / 100
  let buy_price : ℕ := 20
  let sell_price : ℕ := 30
  let sold_toys : ℚ := (initial_toys : ℚ) * sell_percentage
  let revenue : ℚ := sold_toys * (sell_price : ℚ)
  let cost : ℚ := sold_toys * (buy_price : ℚ)
  ⌊revenue - cost⌋ = 1600 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_james_toy_profit_l411_41184


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l411_41193

noncomputable def f (x : ℝ) := 2 * (Real.cos (x / 2))^2 + Real.sin x - 1

theorem f_properties :
  ∃ (period : ℝ) (decr_interval : Set ℝ) (sym_center : ℝ × ℝ) (min_val : ℝ) (min_x : ℝ),
    (∀ x, f (x + period) = f x) ∧ 
    period = 2 * Real.pi ∧
    decr_interval = {x : ℝ | ∃ k : ℤ, π/4 + 2*↑k*π ≤ x ∧ x ≤ 5*π/4 + 2*↑k*π} ∧
    (∃ k : ℤ, sym_center = (↑k*π - π/4, 0)) ∧
    (∀ x ∈ Set.Icc (-π) 0, f x ≥ min_val) ∧
    min_val = -Real.sqrt 2 ∧
    min_x = -3*π/4 ∧
    f min_x = min_val := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l411_41193


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_genetic_material_distribution_l411_41128

-- Define the basic components
structure Cell where
  chromosomes : ℕ
  genetic_material : ℚ
deriving Repr

-- Define meiosis
def meiosis (parent_cell : Cell) : Cell :=
  { chromosomes := parent_cell.chromosomes / 2,
    genetic_material := parent_cell.genetic_material / 2 }

-- Define fertilization
def fertilization (sperm : Cell) (egg : Cell) : Cell :=
  { chromosomes := sperm.chromosomes + egg.chromosomes,
    genetic_material := sperm.genetic_material + egg.genetic_material }

-- Define the theorem
theorem genetic_material_distribution 
  (sperm : Cell) (egg : Cell) (fertilized_egg : Cell) 
  (h1 : fertilized_egg = fertilization sperm egg) :
  ¬(fertilized_egg.genetic_material / 2 = egg.genetic_material) := by
  sorry

#eval meiosis { chromosomes := 46, genetic_material := 1 }
#eval fertilization { chromosomes := 23, genetic_material := 1/2 } { chromosomes := 23, genetic_material := 1/2 }

end NUMINAMATH_CALUDE_ERRORFEEDBACK_genetic_material_distribution_l411_41128


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_blue_on_middle_red_off_second_last_l411_41155

def total_lamps : ℕ := 9
def red_lamps : ℕ := 4
def blue_lamps : ℕ := 5
def lamps_on : ℕ := 4

def probability_specific_arrangement : ℚ :=
  (Nat.choose (total_lamps - 1) (blue_lamps - 1) * Nat.choose (total_lamps - 1) (lamps_on - 1)) /
  (Nat.choose total_lamps red_lamps * Nat.choose total_lamps lamps_on)

theorem probability_blue_on_middle_red_off_second_last :
  probability_specific_arrangement = 35 / 143 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_blue_on_middle_red_off_second_last_l411_41155


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_n_constant_term_l411_41178

/-- The minimum positive integer n for which ((3x^2 - 2/(3x))^n has a constant term in its expansion -/
def min_n : ℕ := 7

/-- The binomial expression ((3x^2 - 2/(3x))^n -/
noncomputable def binomial_expr (x : ℝ) (n : ℕ) : ℝ := (3 * x^2 - 2 / (3 * x))^n

/-- Predicate to check if the expansion has a constant term -/
def has_constant_term (n : ℕ) : Prop :=
  ∃ (c : ℝ), c ≠ 0 ∧ ∀ (x : ℝ), x ≠ 0 → binomial_expr x n = c + x * (binomial_expr x n - c) / x

theorem min_n_constant_term :
  (∀ (k : ℕ), k < min_n → ¬(has_constant_term k)) ∧
  has_constant_term min_n :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_n_constant_term_l411_41178


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_field_level_increase_l411_41108

/-- Proves that digging a rectangular tank in a field and spreading the earth evenly raises the field level by a specific amount. -/
theorem field_level_increase (tank_length tank_width tank_height field_length field_width : ℝ) 
  (h1 : tank_length = 5)
  (h2 : tank_width = 4.5)
  (h3 : tank_height = 2.1)
  (h4 : field_length = 13.5)
  (h5 : field_width = 2.5) : 
  (tank_length * tank_width * tank_height) / 
  (field_length * field_width - tank_length * tank_width) = 4.2 := by
  sorry

#check field_level_increase

end NUMINAMATH_CALUDE_ERRORFEEDBACK_field_level_increase_l411_41108


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_vectors_component_relation_l411_41123

/-- Given two parallel vectors in a plane, prove that their components satisfy a specific relationship. -/
theorem parallel_vectors_component_relation (x : ℝ) : 
  (∃ (k : ℝ), k ≠ 0 ∧ (∀ i : Fin 2, ![-3, x] i = k * ![1, 2] i)) → x = -6 :=
by
  intro h
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_vectors_component_relation_l411_41123


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_minimum_amount_spent_on_boxes_l411_41170

/-- The minimum amount spent on boxes for packaging a fine arts collection -/
theorem minimum_amount_spent_on_boxes 
  (box_length box_width box_height total_volume cost_per_box : ℝ) : 
  box_length = 20 ∧ box_width = 20 ∧ box_height = 12 ∧ 
  total_volume = 2400000 ∧ cost_per_box = 0.5 → 
  (total_volume / (box_length * box_width * box_height)) * cost_per_box = 250 := by
  sorry

#check minimum_amount_spent_on_boxes

end NUMINAMATH_CALUDE_ERRORFEEDBACK_minimum_amount_spent_on_boxes_l411_41170


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_visitor_equation_correct_l411_41152

/-- Represents the monthly average growth rate of visitors -/
def x : ℝ := sorry

/-- Represents the initial number of visitors in the first month -/
def initial_visitors : ℕ := 600

/-- Represents the total number of visitors after three months -/
def total_visitors : ℕ := 2850

/-- Theorem stating that the given equation correctly represents the total visitors over three months -/
theorem visitor_equation_correct :
  (initial_visitors : ℝ) + initial_visitors * (1 + x) + initial_visitors * (1 + x)^2 = total_visitors := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_visitor_equation_correct_l411_41152


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_center_connecting_segment_l411_41158

/-- A rectangle in a 2D plane -/
structure Rectangle where
  -- Define rectangle properties (e.g., corners, dimensions)
  mk :: -- Constructor

/-- A partition of a square into rectangles -/
structure SquarePartition where
  rectangles : Finset Rectangle
  -- Define partition properties

/-- A point in a 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- A line segment between two points -/
structure LineSegment where
  start : Point
  finish : Point

/-- The center of a rectangle -/
def Rectangle.center (r : Rectangle) : Point :=
  sorry

/-- Check if a line segment intersects a rectangle -/
def LineSegment.intersects (l : LineSegment) (r : Rectangle) : Prop :=
  sorry

/-- Main theorem: There always exists a line segment connecting centers of two rectangles
    without intersecting any other rectangle in the partition -/
theorem exists_center_connecting_segment (p : SquarePartition) :
  ∃ (r1 r2 : Rectangle) (l : LineSegment),
    r1 ∈ p.rectangles ∧
    r2 ∈ p.rectangles ∧
    r1 ≠ r2 ∧
    l.start = r1.center ∧
    l.finish = r2.center ∧
    ∀ (r : Rectangle), r ∈ p.rectangles → r ≠ r1 → r ≠ r2 → ¬l.intersects r :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_center_connecting_segment_l411_41158


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_num_paths_6_l411_41187

/-- Number of paths in a triangular grid -/
def num_paths : ℕ → ℕ
  | 0 => 2  -- Base case for n = 1 (we use 0 here due to Lean's natural number representation)
  | n + 1 => 2 * (n + 2) * num_paths n

/-- Theorem stating the number of paths for a triangular grid with side length 6 -/
theorem num_paths_6 : num_paths 5 = 46080 := by
  -- We use 5 here because num_paths 5 corresponds to the 6th triangle (due to 0-based indexing)
  rw [num_paths]
  rw [num_paths]
  rw [num_paths]
  rw [num_paths]
  rw [num_paths]
  norm_num
  rfl

#eval num_paths 5  -- This will evaluate and print the result

end NUMINAMATH_CALUDE_ERRORFEEDBACK_num_paths_6_l411_41187


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solve_3a_plus_2b_l411_41124

-- Define the function g
noncomputable def g (x : ℝ) : ℝ := 3 * x - 7

-- Define the function f
noncomputable def f (a b x : ℝ) : ℝ := a * x + b

-- Define the inverse function of f
noncomputable def f_inv (a b x : ℝ) : ℝ := (x - b) / a

-- Theorem statement
theorem solve_3a_plus_2b (a b : ℝ) : 
  (∀ x, g x = f_inv a b x - 5) → 
  (∀ x, f a b (f_inv a b x) = x) → 
  (∀ x, f_inv a b (f a b x) = x) → 
  3 * a + 2 * b = 7/3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_solve_3a_plus_2b_l411_41124


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ferris_wheel_end_time_l411_41102

/-- Represents time as hours and minutes past noon -/
structure Time where
  hours : ℕ
  minutes : ℕ
  hMinutes : minutes < 60

def addMinutes (t : Time) (m : ℕ) : Time :=
  let totalMinutes := t.hours * 60 + t.minutes + m
  ⟨totalMinutes / 60, totalMinutes % 60, by sorry⟩

structure FerrisWheel where
  capacity : ℕ
  rideDuration : ℕ
  startTime : Time
  totalRiders : ℕ

theorem ferris_wheel_end_time (fw : FerrisWheel) 
  (hCapacity : fw.capacity = 70)
  (hDuration : fw.rideDuration = 20)
  (hStart : fw.startTime = ⟨13, 0, by sorry⟩)
  (hRiders : fw.totalRiders = 1260) :
  addMinutes fw.startTime ((fw.totalRiders / fw.capacity) * fw.rideDuration) = ⟨19, 0, by sorry⟩ :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ferris_wheel_end_time_l411_41102


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_digits_of_2_pow_2015_l411_41182

-- Define the logarithm base 10 of 2
def lg2 : ℝ := 0.3010

-- Define the function to calculate the number of digits in a positive real number
noncomputable def numDigits (x : ℝ) : ℕ := ⌊Real.log x / Real.log 10⌋.toNat + 1

-- Theorem statement
theorem digits_of_2_pow_2015 (h : lg2 = 0.3010) : numDigits (2^2015) = 607 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_digits_of_2_pow_2015_l411_41182


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_length_of_AE_l411_41188

-- Define the points
def A : ℝ × ℝ := (0, 5)
def B : ℝ × ℝ := (5, 0)
def C : ℝ × ℝ := (5, 3)
def D : ℝ × ℝ := (1, 0)

-- Define the grid size
def grid_size : ℕ := 6

-- Define the intersection point E (we don't know its exact coordinates)
def E : ℝ × ℝ := (0, 0)  -- Placeholder coordinates

-- Define that E is on both AB and CD
axiom E_on_AB : ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ E = (t * B.1 + (1 - t) * A.1, t * B.2 + (1 - t) * A.2)
axiom E_on_CD : ∃ s : ℝ, 0 ≤ s ∧ s ≤ 1 ∧ E = (s * C.1 + (1 - s) * D.1, s * C.2 + (1 - s) * D.2)

-- Define the distance function
noncomputable def distance (p q : ℝ × ℝ) : ℝ := Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

-- State the theorem
theorem length_of_AE : distance A E = (25 * Real.sqrt 2) / 6 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_length_of_AE_l411_41188


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_focus_coordinates_l411_41137

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents an ellipse -/
structure Ellipse where
  center : Point
  majorAxisLength : ℝ
  minorAxisLength : ℝ
  isHorizontal : Bool

/-- Computes the x-coordinate of the focus with greater x-coordinate -/
noncomputable def focusX (e : Ellipse) : ℝ :=
  e.center.x + Real.sqrt ((e.majorAxisLength / 2) ^ 2 - (e.minorAxisLength / 2) ^ 2)

/-- Theorem: The focus with greater x-coordinate of the given ellipse has coordinates (4+√7, -2) -/
theorem focus_coordinates (e : Ellipse) 
    (h1 : e.center = ⟨4, -2⟩) 
    (h2 : e.majorAxisLength = 8) 
    (h3 : e.minorAxisLength = 6) 
    (h4 : e.isHorizontal = true) : 
  (focusX e, e.center.y) = (4 + Real.sqrt 7, -2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_focus_coordinates_l411_41137


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l411_41159

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := 2 * (Real.cos x)^2 * Real.tan x + Real.cos (2 * x)

-- State the theorem
theorem f_properties :
  (∃ (p : ℝ), p > 0 ∧ (∀ (x : ℝ), f (x + p) = f x) ∧
    (∀ (q : ℝ), q > 0 ∧ (∀ (x : ℝ), f (x + q) = f x) → p ≤ q)) ∧
  (∃ (M : ℝ), M = Real.sqrt 2 ∧ (∀ (x : ℝ), f x ≤ M) ∧ (∃ (x : ℝ), f x = M)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l411_41159


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_sum_l411_41150

noncomputable def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

noncomputable def is_arithmetic_sequence (b : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, b (n + 1) = b n + d

noncomputable def sum_of_arithmetic_sequence (b : ℕ → ℝ) (n : ℕ) : ℝ :=
  (n : ℝ) * (b 1 + b n) / 2

theorem arithmetic_sequence_sum
  (a b : ℕ → ℝ)
  (h_geom : is_geometric_sequence a)
  (h_arith : is_arithmetic_sequence b)
  (h_relation : a 2 * a 14 = 4 * a 8)
  (h_equal : b 8 = a 8) :
  sum_of_arithmetic_sequence b 15 = 60 := by
  sorry

#check arithmetic_sequence_sum

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_sum_l411_41150


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circumcircle_radius_l411_41127

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := 2 * (Real.cos x)^2 + Real.sqrt 3 * Real.sin (2 * x)

-- Define the triangle ABC
structure Triangle where
  A : ℝ
  B : ℝ
  C : ℝ
  a : ℝ
  b : ℝ
  c : ℝ

-- Define the theorem
theorem circumcircle_radius (ABC : Triangle) (h1 : f ABC.A = 2) (h2 : ABC.b = 1) 
  (h3 : (1/2) * ABC.b * ABC.c * Real.sin ABC.A = Real.sqrt 3 / 2) : 
  (1/2) * ABC.c = 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circumcircle_radius_l411_41127


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_unusual_numbers_l411_41131

/-- A number is unusual if its cube ends with itself but its square doesn't -/
def IsUnusual (n : Nat) : Prop :=
  n % (10^100) = (n^3) % (10^100) ∧ n % (10^100) ≠ (n^2) % (10^100)

/-- There are at least two distinct hundred-digit unusual numbers -/
theorem two_unusual_numbers : ∃ n₁ n₂ : Nat,
  n₁ ≠ n₂ ∧
  10^99 ≤ n₁ ∧ n₁ < 10^100 ∧
  10^99 ≤ n₂ ∧ n₂ < 10^100 ∧
  IsUnusual n₁ ∧ IsUnusual n₂ := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_unusual_numbers_l411_41131


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_k_value_l411_41189

-- Define the arithmetic sequence and its sum
def a : ℕ → ℝ := sorry
def S : ℕ → ℝ := sorry

-- Define k
def k : ℕ := sorry

-- State the given conditions
axiom a_1 : a 1 = 1
axiom a_3 : a 3 = 5
axiom S_diff : ∀ n, S (n + 2) - S n = 36

-- Theorem to prove
theorem k_value : k = 8 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_k_value_l411_41189


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_locus_of_P_l411_41105

-- Define the equilateral triangle ABC
structure EquilateralTriangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  is_equilateral : Prop

-- Define point D on BC
def D (t : ℝ) (ABC : EquilateralTriangle) : ℝ × ℝ := sorry

-- Define circumcenter and incenter
noncomputable def circumcenter (p q r : ℝ × ℝ) : ℝ × ℝ := sorry
noncomputable def incenter (p q r : ℝ × ℝ) : ℝ × ℝ := sorry

-- Define the intersection of two lines
noncomputable def line_intersection (p1 p2 q1 q2 : ℝ × ℝ) : ℝ × ℝ := sorry

-- Define point P
noncomputable def P (t : ℝ) (ABC : EquilateralTriangle) : ℝ × ℝ :=
  let D := D t ABC
  let O1 := circumcenter ABC.A ABC.B D
  let I1 := incenter ABC.A ABC.B D
  let O2 := circumcenter ABC.A D ABC.C
  let I2 := incenter ABC.A D ABC.C
  line_intersection O1 I1 O2 I2

-- Theorem statement
theorem locus_of_P (ABC : EquilateralTriangle) :
  ∃ (x y : ℝ), ∀ t : ℝ, -1 ≤ t ∧ t ≤ 1 →
    let (px, py) := P t ABC
    px = x ∧ py = y ∧ y^2 - x^2/3 = 1 ∧ -1 < x ∧ x < 1 ∧ y < 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_locus_of_P_l411_41105


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_curvature_x_cubed_plus_one_curvature_range_x_cubed_plus_two_l411_41134

-- Define the curvature function
noncomputable def curvature (f : ℝ → ℝ) (x₁ x₂ : ℝ) : ℝ :=
  let y₁ := f x₁
  let y₂ := f x₂
  let k₁ := (deriv f) x₁
  let k₂ := (deriv f) x₂
  |k₁ - k₂| / Real.sqrt ((x₂ - x₁)^2 + (y₂ - y₁)^2)

-- Part 1
theorem curvature_x_cubed_plus_one :
  let f := λ x : ℝ => x^3 + 1
  curvature f 1 2 = (9 * Real.sqrt 2) / 10 := by sorry

-- Part 2
theorem curvature_range_x_cubed_plus_two :
  let f := λ x : ℝ => x^3 + 2
  ∀ x₁ x₂ : ℝ, x₁ ≠ x₂ → x₁ * x₂ ≠ 1 →
    0 < curvature f x₁ x₂ ∧ curvature f x₁ x₂ < (3 * Real.sqrt 10) / 5 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_curvature_x_cubed_plus_one_curvature_range_x_cubed_plus_two_l411_41134


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_max_value_l411_41110

-- Define the function f as noncomputable
noncomputable def f (x : ℝ) : ℝ := Real.sqrt 3 * Real.sin x * Real.cos x - Real.cos x ^ 2

-- Theorem statement
theorem f_max_value :
  (∃ (k : ℤ), ∀ (x : ℝ), f x ≤ f (k * Real.pi + Real.pi / 3)) ∧
  f (Real.pi / 3) = 1 / 2 := by
  sorry

-- Note: The theorem states two things:
-- 1. There exists an integer k such that f(x) ≤ f(kπ + π/3) for all real x
-- 2. The value of f at π/3 (which corresponds to k = 0) is 1/2

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_max_value_l411_41110


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_directrix_l411_41145

/-- The equation of a parabola -/
noncomputable def parabola (x : ℝ) : ℝ := -3 * x^2 + 6 * x - 5

/-- The y-coordinate of the directrix -/
def directrix : ℚ := -23/12

/-- Theorem: The directrix of the given parabola is y = -23/12 -/
theorem parabola_directrix : 
  ∀ x : ℝ, ∃ y : ℝ, y = parabola x → 
  (∃ h k a : ℝ, a < 0 ∧ 
    y = a * (x - h)^2 + k ∧ 
    (directrix : ℝ) = k - 1 / (4 * a)) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_directrix_l411_41145


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_is_60_l411_41118

/-- Calculates the speed of a train in km/hr given its length in meters and time in seconds to cross a pole. -/
noncomputable def train_speed (length : ℝ) (time : ℝ) : ℝ :=
  (length / 1000) / (time / 3600)

/-- Theorem stating that a train with length 250 meters crossing a pole in 15 seconds has a speed of 60 km/hr. -/
theorem train_speed_is_60 :
  train_speed 250 15 = 60 := by
  -- Unfold the definition of train_speed
  unfold train_speed
  -- Simplify the expression
  simp [div_div]
  -- Evaluate the numerical expression
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_is_60_l411_41118


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sunzi_carriage_problem_l411_41139

/-- The Sunzi carriage problem: Find the number of people and carriages -/
theorem sunzi_carriage_problem (x y : ℕ) :
  (3 * (x - 2) = y ∧ 2 * x + 9 = y) ↔ (x = 11 ∧ y = 27) :=
sorry

#check sunzi_carriage_problem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sunzi_carriage_problem_l411_41139


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_centers_triangle_area_l411_41141

theorem square_centers_triangle_area : 
  let square_areas : List ℝ := [36, 64, 100]
  let diagonal_square_area : ℝ := (square_areas.maximum?).getD 0
  let other_squares_areas : List ℝ := square_areas.filter (· ≠ diagonal_square_area)
  let side_lengths : List ℝ := square_areas.map Real.sqrt
  -- Condition: One square's area equals the length of the diagonal formed by the other two squares
  diagonal_square_area = (other_squares_areas.map Real.sqrt).sum ^ 2 →
  -- Prove: The area of the interior triangle is 24
  (side_lengths.prod / 4 : ℝ) = 24 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_centers_triangle_area_l411_41141


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_soda_consumption_l411_41197

/-- Proves that given a person who buys 18 bottles of soda and has 4 bottles left after 28 days, 
    their daily soda consumption is 0.5 bottles. -/
theorem soda_consumption (total_bottles : ℚ) (bottles_left : ℚ) (days : ℚ) 
    (h1 : total_bottles = 18) 
    (h2 : bottles_left = 4) 
    (h3 : days = 28) : 
    (total_bottles - bottles_left) / days = 0.5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_soda_consumption_l411_41197


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shorter_can_radius_l411_41119

/-- Given two cylindrical cans with equal volume, where one can's height is twice the other's,
    and the radius of the taller can is 8 units, the radius of the shorter can is 8√2 units. -/
theorem shorter_can_radius (h : ℝ) (v : ℝ) (r : ℝ) : 
  v = π * 8^2 * (2*h) → -- Volume of taller can
  v = π * r^2 * h →    -- Volume of shorter can
  r = 8 * Real.sqrt 2  -- Radius of shorter can
:= by
  intro h1 h2
  -- The proof steps would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_shorter_can_radius_l411_41119


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_distinct_symmetric_configurations_l411_41185

/-- A domino is a 2×1 rectangle -/
structure Domino :=
  (length : Nat)
  (width : Nat)
  (is_valid : length = 2 ∧ width = 1)

/-- A square configuration is a 4×4 arrangement of dominoes -/
structure SquareConfiguration :=
  (size : Nat)
  (dominoes : List Domino)
  (is_valid : size = 4 ∧ dominoes.length = 8)

/-- Determines if a square configuration is symmetric along one diagonal -/
def is_diagonally_symmetric (config : SquareConfiguration) : Prop :=
  sorry

/-- Determines if two square configurations are equivalent under rotation -/
def are_rotationally_equivalent (config1 config2 : SquareConfiguration) : Prop :=
  sorry

/-- The set of all valid square configurations -/
def all_configurations : Set SquareConfiguration :=
  sorry

/-- The set of diagonally symmetric configurations -/
def symmetric_configurations : Set SquareConfiguration :=
  {config ∈ all_configurations | is_diagonally_symmetric config}

/-- The set of distinct diagonally symmetric configurations under rotation -/
noncomputable def distinct_symmetric_configurations : Finset SquareConfiguration :=
  sorry

/-- The number of distinct diagonally symmetric configurations under rotation is 8 -/
theorem count_distinct_symmetric_configurations :
  Finset.card distinct_symmetric_configurations = 8 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_distinct_symmetric_configurations_l411_41185


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_10_equals_420_l411_41135

def f : ℕ → ℕ
  | 0 => 0  -- Add a case for 0
  | 1 => 1
  | 2 => 2
  | n+3 => f (n+2) + f (n+1) + (n+3)

theorem f_10_equals_420 : f 10 = 420 := by
  -- We'll use 'sorry' to skip the proof for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_10_equals_420_l411_41135


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_not_perfect_power_l411_41132

def a : ℕ → ℕ
  | 0 => 18
  | n + 1 => (a n)^2 + 6 * (a n)

def is_perfect_power (n : ℕ) : Prop :=
  ∃ (b k : ℕ), k > 1 ∧ n = b^k

theorem sequence_not_perfect_power :
  ∀ n : ℕ, ¬(is_perfect_power (a n)) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_not_perfect_power_l411_41132


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_three_digit_product_four_zeros_l411_41111

def digits (n : ℕ) : List ℕ :=
  if n = 0 then [0] else
    let rec aux (m : ℕ) (acc : List ℕ) : List ℕ :=
      if m = 0 then acc else aux (m / 10) ((m % 10) :: acc)
    aux n []

theorem three_digit_product_four_zeros : ∃ (a b c : ℕ), 
  (100 ≤ a ∧ a < 1000) ∧ 
  (100 ≤ b ∧ b < 1000) ∧ 
  (100 ≤ c ∧ c < 1000) ∧ 
  (a = 625 ∧ b = 384 ∧ c = 971) ∧
  (∀ d : ℕ, d ∈ [1, 2, 3, 4, 5, 6, 7, 8, 9] → 
    (d ∈ (digits a) ∨ d ∈ (digits b) ∨ d ∈ (digits c))) ∧
  (∀ d : ℕ, d ∈ (digits a) ∪ (digits b) ∪ (digits c) → 
    d ∈ [1, 2, 3, 4, 5, 6, 7, 8, 9]) ∧
  (a * b * c) % 10000 = 0 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_three_digit_product_four_zeros_l411_41111


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_points_on_ellipse_l411_41171

/-- Represents a point on the ellipse -/
structure EllipsePoint where
  x : ℝ
  y : ℝ
  on_ellipse : x^2 / 4 + y^2 / 3 = 1

/-- Represents the right focus of the ellipse -/
def RightFocus : ℝ × ℝ := (1, 0)

/-- Distance between a point on the ellipse and the right focus -/
noncomputable def distToFocus (p : EllipsePoint) : ℝ :=
  Real.sqrt ((p.x - 1)^2 + p.y^2)

/-- Theorem stating the maximum number of points on the ellipse forming an arithmetic sequence -/
theorem max_points_on_ellipse (n : ℕ) 
  (points : Fin n → EllipsePoint) 
  (is_arithmetic_seq : ∀ i j k : Fin n, 
    distToFocus (points i) - distToFocus (points j) = 
    distToFocus (points j) - distToFocus (points k))
  (min_diff : ∀ i j : Fin n, i ≠ j → 
    |distToFocus (points i) - distToFocus (points j)| ≥ 1/100) :
  n ≤ 201 := by
  sorry

#check max_points_on_ellipse

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_points_on_ellipse_l411_41171


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_oblique_axonometric_preserves_parallel_equal_segments_l411_41161

/-- Represents a line segment in 3D space -/
structure LineSegment where
  start : ℝ × ℝ × ℝ
  end_ : ℝ × ℝ × ℝ

/-- Represents an oblique axonometric projection -/
structure ObliqueAxonometricProjection where
  projection_matrix : Matrix (Fin 3) (Fin 3) ℝ

/-- Checks if two line segments are parallel and equal -/
def are_parallel_and_equal (l1 l2 : LineSegment) : Prop :=
  sorry

/-- Projects a line segment using oblique axonometric projection -/
def project_line_segment (proj : ObliqueAxonometricProjection) (l : LineSegment) : LineSegment :=
  sorry

/-- Theorem: Under oblique axonometric projection, parallel and equal line segments
    remain parallel and equal in the intuitive diagram -/
theorem oblique_axonometric_preserves_parallel_equal_segments
  (proj : ObliqueAxonometricProjection) (l1 l2 : LineSegment) :
  are_parallel_and_equal l1 l2 →
  are_parallel_and_equal (project_line_segment proj l1) (project_line_segment proj l2) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_oblique_axonometric_preserves_parallel_equal_segments_l411_41161


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_periodic_smallest_period_l411_41153

noncomputable def f (x : ℝ) : ℝ := |Real.sin (x + Real.pi/3)|

theorem f_periodic_smallest_period :
  (∃ (p : ℝ), p > 0 ∧ (∀ x, f (x + p) = f x)) ∧
  (∀ q : ℝ, q > 0 → (∀ x, f (x + q) = f x) → q ≥ Real.pi) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_periodic_smallest_period_l411_41153


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_positive_distance_implies_parallel_l411_41190

/-- Directed distance from a point to a line -/
noncomputable def directed_distance (a b c x y : ℝ) : ℝ :=
  (a * x + b * y + c) / Real.sqrt (a^2 + b^2)

/-- Two points are on the same side of a line if their directed distances have the same sign -/
def same_side (d₁ d₂ : ℝ) : Prop := d₁ * d₂ > 0

/-- A line is defined by its normal vector (a, b) and a point (x, y) on the line -/
structure Line where
  a : ℝ
  b : ℝ
  x : ℝ
  y : ℝ
  norm_nonzero : a^2 + b^2 ≠ 0

/-- Two lines are parallel if their normal vectors are scalar multiples of each other -/
def parallel (l₁ l₂ : Line) : Prop :=
  ∃ (k : ℝ), k ≠ 0 ∧ l₁.a = k * l₂.a ∧ l₁.b = k * l₂.b

theorem equal_positive_distance_implies_parallel (l : Line) (x₁ y₁ x₂ y₂ : ℝ) :
  let d₁ := directed_distance l.a l.b (-l.a * l.x - l.b * l.y) x₁ y₁
  let d₂ := directed_distance l.a l.b (-l.a * l.x - l.b * l.y) x₂ y₂
  d₁ = 1 ∧ d₂ = 1 →
  parallel l (Line.mk (y₂ - y₁) (x₁ - x₂) x₁ y₁ (by sorry)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_positive_distance_implies_parallel_l411_41190


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_radius_from_distances_l411_41100

/-- A circle in a plane. -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- The distance between two points in a plane. -/
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

/-- The shortest distance from a point to a circle. -/
noncomputable def shortestDistance (p : ℝ × ℝ) (c : Circle) : ℝ :=
  (distance p c.center) - c.radius

/-- The longest distance from a point to a circle. -/
noncomputable def longestDistance (p : ℝ × ℝ) (c : Circle) : ℝ :=
  (distance p c.center) + c.radius

/-- Theorem: If the shortest distance from a point P to a circle O is 2
    and the longest distance is 6, then the radius of O is 2. -/
theorem circle_radius_from_distances (P : ℝ × ℝ) (O : Circle)
    (h_shortest : shortestDistance P O = 2)
    (h_longest : longestDistance P O = 6) :
    O.radius = 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_radius_from_distances_l411_41100


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_liar_statements_l411_41186

/-- Represents a person in the row -/
inductive Person
| Knight
| Liar

/-- Represents a statement about the distribution of a type of person -/
inductive Statement
| MoreRight
| MoreLeft

/-- A row of people -/
def Row := List Person

/-- A function that returns the statement made by a person about knights -/
def statementAboutKnights : Person → Statement → Bool
| Person.Knight, Statement.MoreRight => true
| Person.Knight, Statement.MoreLeft => true
| Person.Liar, Statement.MoreRight => false
| Person.Liar, Statement.MoreLeft => false

/-- A function that returns the statement made by a person about liars -/
def statementAboutLiars : Person → Statement → Bool
| Person.Knight, Statement.MoreRight => true
| Person.Knight, Statement.MoreLeft => true
| Person.Liar, Statement.MoreRight => false
| Person.Liar, Statement.MoreLeft => false

/-- The number of people making each statement about knights is equal -/
def equalKnightStatements (row : Row) : Prop :=
  (row.filter (λ p => statementAboutKnights p Statement.MoreRight)).length =
  (row.filter (λ p => statementAboutKnights p Statement.MoreLeft)).length

theorem equal_liar_statements (row : Row) (h : equalKnightStatements row) :
  (row.filter (λ p => statementAboutLiars p Statement.MoreRight)).length =
  (row.filter (λ p => statementAboutLiars p Statement.MoreLeft)).length :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_liar_statements_l411_41186


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_book_sale_loss_percentage_l411_41154

/-- Given a book with selling prices resulting in loss and gain, calculate the loss percentage -/
theorem book_sale_loss_percentage 
  (loss_price : ℝ) 
  (gain_price : ℝ) 
  (gain_percentage : ℝ) 
  (h1 : loss_price = 810) 
  (h2 : gain_price = 990) 
  (h3 : gain_percentage = 10) :
  (gain_price / (1 + gain_percentage / 100) - loss_price) / (gain_price / (1 + gain_percentage / 100)) * 100 = 10 :=
by
  -- Define cost_price
  let cost_price := gain_price / (1 + gain_percentage / 100)
  
  -- Define loss_amount
  let loss_amount := cost_price - loss_price
  
  -- Define loss_percentage
  let loss_percentage := (loss_amount / cost_price) * 100
  
  -- Proof steps would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_book_sale_loss_percentage_l411_41154


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_divisors_iff_even_l411_41148

/-- The number of positive divisors of a positive integer -/
def num_divisors (n : ℕ+) : ℕ := sorry

/-- The property that k is the number of divisors for a, b, and 2a + 3b -/
def has_equal_divisors (k a b : ℕ+) : Prop :=
  (num_divisors a = k) ∧ (num_divisors b = k) ∧ (num_divisors (2 * a + 3 * b) = k)

/-- The main theorem: k has the equal divisors property if and only if k is even -/
theorem equal_divisors_iff_even (k : ℕ+) :
  (∃ (a b : ℕ+), has_equal_divisors k a b) ↔ Even k.val := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_divisors_iff_even_l411_41148


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_value_at_sum_l411_41146

-- Define the function f
noncomputable def f (x : ℝ) (φ : ℝ) := 4 * Real.cos (3 * x + φ)

-- State the theorem
theorem function_value_at_sum (φ : ℝ) (x₁ x₂ : ℝ) :
  |φ| < π / 2 →
  (∀ x, f x φ = f (11 * π / 6 - x) φ) →
  x₁ ∈ Set.Ioo (-7 * π / 12) (-π / 12) →
  x₂ ∈ Set.Ioo (-7 * π / 12) (-π / 12) →
  x₁ ≠ x₂ →
  f x₁ φ = f x₂ φ →
  f (x₁ + x₂) φ = 2 * Real.sqrt 2 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_value_at_sum_l411_41146


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_and_g_l411_41157

-- Define the functions
noncomputable def f (x : ℝ) : ℝ := 2 * Real.cos x - 1
noncomputable def g (x : ℝ) : ℝ := (2 * Real.sin x - 1)^2 + 3

-- State the theorem
theorem range_of_f_and_g :
  (∀ y ∈ Set.range f, -3 ≤ y ∧ y ≤ 1) ∧
  (∀ z ∈ Set.range f, ∃ x, f x = z) ∧
  (∀ y ∈ Set.range g, 3 ≤ y ∧ y ≤ 12) ∧
  (∀ z ∈ Set.range g, ∃ x, g x = z) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_and_g_l411_41157


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_focal_chord_l411_41109

/-- Represents a parabola y^2 = 2px where p > 0 -/
structure Parabola where
  p : ℝ
  h_p_pos : p > 0

/-- Represents a line passing through the focus of a parabola at an angle of π/3 -/
structure FocalLine (par : Parabola) where
  angle : ℝ
  h_angle : angle = π / 3

/-- Represents two points A and B where the focal line intersects the parabola -/
structure IntersectionPoints (par : Parabola) (line : FocalLine par) where
  A : ℝ × ℝ
  B : ℝ × ℝ
  h_on_parabola : (A.2)^2 = 2 * par.p * A.1 ∧ (B.2)^2 = 2 * par.p * B.1
  h_on_line : A.2 = Real.sqrt 3 * (A.1 - par.p / 2) ∧ B.2 = Real.sqrt 3 * (B.1 - par.p / 2)
  h_distance : Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 6

/-- Represents a focal chord of the parabola -/
def FocalChord (par : Parabola) := ℝ

/-- The main theorem stating that there exists exactly one focal chord of length 9/2 -/
theorem unique_focal_chord
  (par : Parabola)
  (line : FocalLine par)
  (points : IntersectionPoints par line) :
  ∃! (chord : FocalChord par), chord = (9 : ℝ) / 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_focal_chord_l411_41109


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_y_coordinate_of_P_l411_41160

/-- The ellipse equation -/
def is_on_ellipse (x y : ℝ) : Prop := x^2/4 + y^2/3 = 1

/-- The foci of the ellipse -/
def F₁ : ℝ × ℝ := (-1, 0)
def F₂ : ℝ × ℝ := (1, 0)

/-- Points A and B are on the upper half-ellipse -/
noncomputable def A : ℝ × ℝ := sorry
noncomputable def B : ℝ × ℝ := sorry

/-- F₁A is parallel to F₂B -/
axiom parallel_lines : (A.1 - F₁.1) / (A.2 - F₁.2) = (B.1 - F₂.1) / (B.2 - F₂.2)

/-- P is the intersection of BF₁ and AF₂ -/
noncomputable def P : ℝ × ℝ := sorry

/-- A and B are on the ellipse -/
axiom A_on_ellipse : is_on_ellipse A.1 A.2
axiom B_on_ellipse : is_on_ellipse B.1 B.2

/-- A and B are on the upper half of the ellipse -/
axiom A_upper : A.2 > 0
axiom B_upper : B.2 > 0

/-- The maximum y-coordinate of P is 3/4 -/
theorem max_y_coordinate_of_P : 
  ∀ θ : ℝ, 0 ≤ θ ∧ θ ≤ π → P.2 ≤ 3/4 ∧ ∃ θ₀ : ℝ, P.2 = 3/4 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_y_coordinate_of_P_l411_41160


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_pole_time_l411_41195

/-- Calculates the time (in seconds) it takes for a train to cross a pole
    given its speed (in km/hr) and length (in meters). -/
noncomputable def train_crossing_time (speed_km_hr : ℝ) (length_m : ℝ) : ℝ :=
  length_m / (speed_km_hr * 1000 / 3600)

/-- Theorem: A train with a speed of 180 km/hr and a length of 450 meters
    takes 9 seconds to cross a pole. -/
theorem train_crossing_pole_time :
  train_crossing_time 180 450 = 9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_pole_time_l411_41195


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_nested_fraction_sum_l411_41168

def nested_fraction : ℚ := 1 + 1 / (1 + 1 / (1 + 1 / 1))

theorem nested_fraction_sum : ∃ (a b : ℕ), 
  nested_fraction = a / b ∧ 
  Nat.Coprime a b ∧ 
  a + b = 21 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_nested_fraction_sum_l411_41168


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pi_minus_three_power_zero_minus_one_third_power_minus_one_l411_41136

theorem pi_minus_three_power_zero_minus_one_third_power_minus_one : 
  (π - 3 : ℝ) ^ (0 : ℝ) - (1 / 3 : ℝ) ^ (-1 : ℝ) = -2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pi_minus_three_power_zero_minus_one_third_power_minus_one_l411_41136


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_sine_cotangent_inequality_side_cosine_product_inequality_l411_41191

-- Define an acute-angled triangle
structure AcuteTriangle where
  A : ℝ
  B : ℝ
  C : ℝ
  a : ℝ
  b : ℝ
  c : ℝ
  acute : 0 < A ∧ A < π/2 ∧ 0 < B ∧ B < π/2 ∧ 0 < C ∧ C < π/2
  sum_angles : A + B + C = π
  side_angle_relation : a / (Real.sin A) = b / (Real.sin B) ∧ b / (Real.sin B) = c / (Real.sin C)

-- State the theorems
theorem cosine_sine_cotangent_inequality (t : AcuteTriangle) :
  Real.cos t.A + Real.cos t.B + Real.cos t.C ≤ 
  1/3 * (Real.sin t.A + Real.sin t.B + Real.sin t.C) * (1 / Real.tan t.A + 1 / Real.tan t.B + 1 / Real.tan t.C) :=
by sorry

theorem side_cosine_product_inequality (t : AcuteTriangle) :
  (t.a + Real.cos t.A) * (t.b + Real.cos t.B) * (t.c + Real.cos t.C) ≥ 
  (t.a + Real.cos t.B) * (t.b + Real.cos t.C) * (t.c + Real.cos t.A) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_sine_cotangent_inequality_side_cosine_product_inequality_l411_41191


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_F_unique_zero_l411_41177

-- Define the function f implicitly
noncomputable def f : ℝ → ℝ := sorry

-- Define the equation that f satisfies
axiom f_eq : ∀ x : ℝ, x^2 / 4 + f x * |f x| = 1

-- Theorem for the range of f
theorem f_range : Set.range f = Set.Iic 1 := by sorry

-- Define F in terms of f
noncomputable def F (x : ℝ) : ℝ := f x + x

-- Theorem for the number of zero points of F
theorem F_unique_zero : ∃! x : ℝ, F x = 0 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_F_unique_zero_l411_41177


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_diff_intersection_y_coords_is_zero_l411_41183

-- Define the two functions
def f (x : ℝ) : ℝ := 5 - 2 * x^2 + x^4
def g (x : ℝ) : ℝ := 3 + 2 * x^2 + x^4

-- Define the intersection points
def intersection_points : Set ℝ := {x : ℝ | f x = g x}

-- Define the y-coordinates of intersection points
def intersection_y_coords : Set ℝ := {y : ℝ | ∃ x, x ∈ intersection_points ∧ y = f x}

-- Statement: The maximum difference between y-coordinates of intersection points is 0
theorem max_diff_intersection_y_coords_is_zero :
  ∀ y₁ y₂, y₁ ∈ intersection_y_coords → y₂ ∈ intersection_y_coords → |y₁ - y₂| = 0 :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_diff_intersection_y_coords_is_zero_l411_41183


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_numbers_count_and_largest_l411_41179

def is_valid_number (n : ℕ) : Prop :=
  ∃ (a b : ℕ),
    a ≠ 0 ∧ b ≠ 0 ∧
    n = 700000 + a * 10000 + 6000 + 300 + 10 + b

def cyclic_sum (n : ℕ) : ℕ :=
  let digits := [7, n / 10000 % 10, 6, 3, 1, n % 10]
  (digits.sum * 111111)

theorem valid_numbers_count_and_largest :
  (∃ (valid_numbers : Finset ℕ),
    valid_numbers.card = 7 ∧
    (∀ n ∈ valid_numbers, is_valid_number n ∧ cyclic_sum n % 121 = 0) ∧
    (∀ n, is_valid_number n → cyclic_sum n % 121 = 0 → n ∈ valid_numbers)) ∧
  (∃ (largest : ℕ),
    is_valid_number largest ∧
    cyclic_sum largest % 121 = 0 ∧
    largest = 796317 ∧
    (∀ n, is_valid_number n → cyclic_sum n % 121 = 0 → n ≤ largest)) := by
  sorry

#check valid_numbers_count_and_largest

end NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_numbers_count_and_largest_l411_41179
