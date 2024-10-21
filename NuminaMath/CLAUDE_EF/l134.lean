import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circumscribing_sphere_volume_l134_13479

/-- A hexagonal prism with a regular hexagonal base -/
structure HexagonalPrism where
  /-- The height of the prism -/
  height : ℝ
  /-- The circumference of the base -/
  base_circumference : ℝ

/-- A sphere circumscribing a hexagonal prism -/
structure CircumscribingSphere (prism : HexagonalPrism) where
  /-- All vertices of the prism lie on this sphere -/
  contains_vertices : Prop

/-- The volume of a sphere -/
noncomputable def sphere_volume (radius : ℝ) : ℝ := (4 / 3) * Real.pi * radius ^ 3

/-- Theorem: The volume of a sphere circumscribing a hexagonal prism with given properties -/
theorem circumscribing_sphere_volume
  (prism : HexagonalPrism)
  (sphere : CircumscribingSphere prism)
  (h1 : prism.height = Real.sqrt 3)
  (h2 : prism.base_circumference = 3) :
  sphere_volume 1 = (4 / 3) * Real.pi :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circumscribing_sphere_volume_l134_13479


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_phi_for_even_f_l134_13404

noncomputable def f (x φ : ℝ) : ℝ := Real.cos (Real.sqrt 3 * x + φ) - Real.sqrt 3 * Real.sin (Real.sqrt 3 * x + φ)

theorem smallest_positive_phi_for_even_f :
  ∀ x : ℝ, f x (2 * Real.pi / 3) = f (-x) (2 * Real.pi / 3) ∧
  ∀ ψ : ℝ, 0 < ψ ∧ ψ < 2 * Real.pi / 3 →
    ∃ y : ℝ, f y ψ ≠ f (-y) ψ :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_phi_for_even_f_l134_13404


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complement_M_l134_13454

-- Define the function f
def f : ℝ → ℝ := sorry

-- Define the properties of f
axiom f_increasing : ∀ x y : ℝ, x < y → f x < f y
axiom f_0 : f 0 = -1
axiom f_3 : f 3 = 1

-- Define the set M
def M : Set ℝ := {x : ℝ | |f (x + 1)| < 1}

-- State the theorem
theorem complement_M : 
  (Set.univ : Set ℝ) \ M = Set.Iic (-1) ∪ Set.Ici 2 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complement_M_l134_13454


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_m_value_when_z_purely_imaginary_max_abs_w_given_distance_from_z_max_abs_w_achieved_l134_13405

/-- The complex number z as a function of m -/
noncomputable def z (m : ℝ) : ℂ := ((m^2 - m - 2) + (m^2 + m)*Complex.I) / (1 + Complex.I)

/-- Theorem stating that m = 1 when z is purely imaginary -/
theorem m_value_when_z_purely_imaginary :
  ∃ m : ℝ, z m = Complex.I * Complex.im (z m) ↔ m = 1 := by sorry

/-- The maximum value of |w| when |w - 2i| = 1 -/
theorem max_abs_w_given_distance_from_z (w : ℂ) :
  Complex.abs (w - 2*Complex.I) = 1 →
  Complex.abs w ≤ 3 := by sorry

/-- The maximum value of |w| is achieved -/
theorem max_abs_w_achieved :
  ∃ w : ℂ, Complex.abs (w - 2*Complex.I) = 1 ∧ Complex.abs w = 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_m_value_when_z_purely_imaginary_max_abs_w_given_distance_from_z_max_abs_w_achieved_l134_13405


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_triangle_area_l134_13422

-- Define the necessary structures
structure Triangle where
  vertices : Fin 3 → ℝ × ℝ

structure Square where
  vertices : Fin 4 → ℝ × ℝ

-- Define the necessary functions
def Triangle.isEquilateral (t : Triangle) : Prop := sorry
def Triangle.sideLength (t : Triangle) : ℝ := sorry
def Square.sideLength (s : Square) : ℝ := sorry
def Square.contains (s : Square) (t : Triangle) : Prop := sorry
def Triangle.contains (t : Triangle) (s : Square) : Prop := sorry
def Triangle.area (t : Triangle) : ℝ := sorry

theorem smallest_triangle_area (ABC : Triangle) (ABDE BCFG CAHI : Square) :
  Triangle.isEquilateral ABC →
  Triangle.sideLength ABC = 2 →
  Square.sideLength ABDE = 2 →
  Square.sideLength BCFG = 2 →
  Square.sideLength CAHI = 2 →
  Square.contains ABDE ABC →
  Square.contains BCFG ABC →
  Square.contains CAHI ABC →
  ∃ (JKL : Triangle),
    Triangle.contains JKL ABDE ∧
    Triangle.contains JKL BCFG ∧
    Triangle.contains JKL CAHI ∧
    (∀ (T : Triangle),
      Triangle.contains T ABDE →
      Triangle.contains T BCFG →
      Triangle.contains T CAHI →
      Triangle.area T ≥ Triangle.area JKL) →
    Triangle.area JKL = 13 * Real.sqrt 3 - 12 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_triangle_area_l134_13422


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exist_diametric_red_points_l134_13443

/-- Representation of a point on the circle -/
structure Point where
  is_red : Bool

/-- Representation of an arc on the circle -/
structure Arc where
  length : Nat

/-- A circle with red points and arcs -/
structure Circle where
  points : Finset Point
  arcs : Finset Arc
  red_point_count : Nat
  arc_count : Nat
  arc_length_1_count : Nat
  arc_length_2_count : Nat
  arc_length_3_count : Nat

/-- Predicate to check if two points are diametrically opposite -/
def DiametricallyOpposite (p1 p2 : Point) : Prop := sorry

/-- The main theorem -/
theorem exist_diametric_red_points (c : Circle)
  (h1 : c.red_point_count = 2019)
  (h2 : c.arc_count = 2019)
  (h3 : c.arc_length_1_count = 673)
  (h4 : c.arc_length_2_count = 673)
  (h5 : c.arc_length_3_count = 673) :
  ∃ p1 p2 : Point, p1 ∈ c.points ∧ p2 ∈ c.points ∧ p1.is_red ∧ p2.is_red ∧ p1 ≠ p2 ∧ DiametricallyOpposite p1 p2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exist_diametric_red_points_l134_13443


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_cost_theorem_l134_13407

-- Define the cost of seedlings
def cost_A : ℕ → ℕ := sorry
def cost_B : ℕ → ℕ := sorry

-- Define the total number of seedlings
def total_seedlings : ℕ := 150

-- Define the constraints
axiom cost_difference : ∀ n, cost_A n = cost_B n + 5
axiom equal_quantity : 400 / cost_A 1 = 300 / cost_B 1
axiom min_A_seedlings (m : ℕ) : m ≥ (total_seedlings - m) / 2 → m ≥ 50

-- Define the total cost function
def total_cost (m : ℕ) : ℕ := m * cost_A 1 + (total_seedlings - m) * cost_B 1

-- Theorem to prove
theorem min_cost_theorem :
  ∃ m, m = 50 ∧ 
    total_cost m = 2500 ∧ 
    (∀ n, n ≥ (total_seedlings - n) / 2 → total_cost m ≤ total_cost n) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_cost_theorem_l134_13407


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_relationship_abc_l134_13474

theorem relationship_abc (a b c : ℝ) 
  (ha : (7 : ℝ)^a = 5)
  (hb : (8 : ℝ)^b = 6)
  (hc : Real.exp (2/c) = 2 + Real.exp 2) :
  c > b ∧ b > a := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_relationship_abc_l134_13474


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_compound_interest_approximation_l134_13427

/-- Calculates the compound interest given principal, rate, time, and compounding frequency -/
noncomputable def compound_interest (principal : ℝ) (rate : ℝ) (time : ℝ) (frequency : ℝ) : ℝ :=
  principal * (1 + rate / frequency) ^ (frequency * time)

/-- The compound interest calculation is approximately correct -/
theorem compound_interest_approximation :
  let principal : ℝ := 8000
  let rate : ℝ := 0.07
  let time : ℝ := 5
  let frequency : ℝ := 2
  let result := compound_interest principal rate time frequency
  abs (result - 11284.8) < 0.1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_compound_interest_approximation_l134_13427


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_common_difference_range_l134_13494

/-- An arithmetic sequence with common difference d -/
noncomputable def arithmetic_sequence (d : ℝ) : ℕ → ℝ
  | 0 => 8 - d
  | n + 1 => arithmetic_sequence d n + d

/-- Sum of the first n terms of the arithmetic sequence -/
noncomputable def sum_arithmetic (d : ℝ) (n : ℕ) : ℝ :=
  n * (8 - d) + n * (n - 1) * d / 2

theorem common_difference_range :
  ∀ d : ℝ, (∀ n : ℕ, sum_arithmetic d n ≤ sum_arithmetic d 7) ↔ 
  -8/5 ≤ d ∧ d ≤ -4/3 := by
  sorry

#check common_difference_range

end NUMINAMATH_CALUDE_ERRORFEEDBACK_common_difference_range_l134_13494


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_point_B_and_expression_value_l134_13464

def unit_circle (x y : ℝ) : Prop := x^2 + y^2 = 1

theorem point_B_and_expression_value 
  (θ : ℝ) 
  (h_sin : Real.sin θ = 4/5) 
  (h_second_quadrant : ∃ (x y : ℝ), x < 0 ∧ y > 0 ∧ unit_circle x y ∧ θ = Real.arccos (-x)) :
  (∃ (x y : ℝ), x = -3/5 ∧ y = 4/5 ∧ unit_circle x y) ∧ 
  (Real.sin (π + θ) + 2 * Real.sin (π/2 - θ)) / (2 * Real.cos (π - θ)) = -5/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_point_B_and_expression_value_l134_13464


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometry_propositions_proposition_descriptions_l134_13409

-- Define the four propositions as axioms
axiom proposition1 : Prop
axiom proposition2 : Prop
axiom proposition3 : Prop
axiom proposition4 : Prop

-- Provide informal descriptions of the propositions
def description1 : String := "Three points determine a plane"
def description2 : String := "A rectangle is a plane figure"
def description3 : String := "If three lines intersect pairwise, they determine a plane"
def description4 : String := "Two intersecting planes divide space into four regions"

-- Theorem stating which propositions are correct and incorrect
theorem geometry_propositions :
  (¬ proposition1) ∧
  proposition2 ∧
  (¬ proposition3) ∧
  proposition4 := by
  sorry

-- Additional theorem to connect descriptions to propositions
theorem proposition_descriptions :
  (proposition1 ↔ description1 = "Three points determine a plane") ∧
  (proposition2 ↔ description2 = "A rectangle is a plane figure") ∧
  (proposition3 ↔ description3 = "If three lines intersect pairwise, they determine a plane") ∧
  (proposition4 ↔ description4 = "Two intersecting planes divide space into four regions") := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometry_propositions_proposition_descriptions_l134_13409


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_min_distance_l134_13431

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x^2 / 4 - y^2 = 1

-- Define the distance between two points
noncomputable def distance (x1 y1 x2 y2 : ℝ) : ℝ := 
  Real.sqrt ((x1 - x2)^2 + (y1 - y2)^2)

-- Define the theorem
theorem hyperbola_min_distance (a : ℝ) :
  (∃ (x y : ℝ), x ≥ 2 ∧ hyperbola x y ∧
    (∀ (x' y' : ℝ), x' ≥ 2 → hyperbola x' y' →
      distance x y a 0 ≤ distance x' y' a 0) ∧
    distance x y a 0 = 3) →
  a = -1 ∨ a = 2 * Real.sqrt 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_min_distance_l134_13431


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_departure_difference_l134_13433

/-- The time difference between the departure of two trains -/
noncomputable def time_difference (v_goods v_express : ℝ) (catch_up_time : ℝ) : ℝ :=
  (v_express * catch_up_time - v_goods * catch_up_time) / v_goods

/-- Theorem stating the time difference between the departure of two trains
    given their speeds and the time it takes for the express train to catch up -/
theorem train_departure_difference 
  (v_goods : ℝ) 
  (v_express : ℝ) 
  (catch_up_time : ℝ) 
  (h_v_goods : v_goods = 36) 
  (h_v_express : v_express = 90) 
  (h_catch_up : catch_up_time = 4) :
  time_difference v_goods v_express catch_up_time = 6 := by
  sorry

-- Remove the #eval statement as it's not necessary for the proof
-- and may cause issues with noncomputable definitions

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_departure_difference_l134_13433


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rational_greater_than_three_l134_13466

theorem rational_greater_than_three : ∃ x : ℚ, (x > 3) ∧ 
  (x = 11/3 ∨ x = |(-3)| ∨ x = Real.pi ∨ x = Real.sqrt 10) ∧
  (∀ y : ℝ, (y = |(-3)| ∨ y = Real.pi ∨ y = Real.sqrt 10) → 
    (¬ (∃ (a b : ℤ), y = ↑a / ↑b) ∨ y ≤ 3)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rational_greater_than_three_l134_13466


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_last_digit_of_sum_l134_13411

theorem last_digit_of_sum (n : ℕ) (h1 : Even n) 
    (h2 : (n * (n + 1) / 2) % 10 = 8) : 
    (((2 * n + 1 + 4 * n) * n / 2) % 10 = 2) := by
  sorry

#check last_digit_of_sum

end NUMINAMATH_CALUDE_ERRORFEEDBACK_last_digit_of_sum_l134_13411


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_relative_errors_equal_l134_13476

-- Define the lengths and errors
noncomputable def line1_length : ℝ := 10
noncomputable def line1_error : ℝ := 0.02
noncomputable def line2_length : ℝ := 100
noncomputable def line2_error : ℝ := 0.2

-- Define relative error
noncomputable def relative_error (error : ℝ) (length : ℝ) : ℝ := (error / length) * 100

-- Theorem statement
theorem relative_errors_equal :
  relative_error line1_error line1_length = relative_error line2_error line2_length :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_relative_errors_equal_l134_13476


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_point_ratio_l134_13434

/-- Hyperbola type representing x²/8 - y²/12 = 1 --/
structure Hyperbola where
  a : ℝ
  b : ℝ
  c : ℝ
  eq : a^2 = 8 ∧ b^2 = 12 ∧ c^2 = a^2 + b^2

/-- Point in 2D space --/
structure Point where
  x : ℝ
  y : ℝ

/-- Vector in 2D space --/
structure Vec2D where
  x : ℝ
  y : ℝ

/-- Dot product of two vectors --/
def dot_product (v1 v2 : Vec2D) : ℝ := v1.x * v2.x + v1.y * v2.y

/-- Length of a vector --/
noncomputable def vector_length (v : Vec2D) : ℝ := Real.sqrt (v.x^2 + v.y^2)

/-- Vector from one point to another --/
def vector_between (p1 p2 : Point) : Vec2D :=
  { x := p2.x - p1.x, y := p2.y - p1.y }

/-- Vector addition --/
def vector_add (v1 v2 : Vec2D) : Vec2D :=
  { x := v1.x + v2.x, y := v1.y + v2.y }

/-- Main theorem --/
theorem hyperbola_point_ratio (h : Hyperbola) (o f1 f2 m : Point) (t : ℝ) :
  let om := vector_between o m
  let of2 := vector_between o f2
  let f2m := vector_between f2 m
  let f1m := vector_between f1 m
  (m.x^2 / 8 - m.y^2 / 12 = 1) →  -- M is on the hyperbola
  (dot_product (vector_add om of2) f2m = 0) →  -- (OM + OF2) ⋅ F2M = 0
  (vector_length f1m = t * vector_length f2m) →  -- |F1M| = t|F2M|
  (t = 3) := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_point_ratio_l134_13434


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cricket_run_rate_theorem_l134_13448

/-- Represents a cricket game scenario -/
structure CricketGame where
  total_overs : ℕ
  first_half_overs : ℕ
  first_half_run_rate : ℚ
  target_runs : ℕ

/-- Calculates the required run rate for the second half of the game -/
noncomputable def required_run_rate (game : CricketGame) : ℚ :=
  let runs_scored := game.first_half_run_rate * game.first_half_overs
  let remaining_runs := game.target_runs - runs_scored.floor
  let remaining_overs := game.total_overs - game.first_half_overs
  remaining_runs / remaining_overs

/-- Theorem stating the required run rate for the given scenario -/
theorem cricket_run_rate_theorem (game : CricketGame) 
  (h1 : game.total_overs = 20)
  (h2 : game.first_half_overs = 10)
  (h3 : game.first_half_run_rate = 16/5)
  (h4 : game.target_runs = 282) :
  required_run_rate game = 25 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cricket_run_rate_theorem_l134_13448


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_peach_tree_space_is_15_l134_13401

/-- Represents the space taken by fruit trees in Quinton's backyard -/
structure FruitTreeSpace where
  apple_width : ℕ
  apple_space : ℕ
  peach_width : ℕ
  total_space : ℕ

/-- Calculates the space between peach trees -/
def space_between_peach_trees (s : FruitTreeSpace) : ℕ :=
  s.total_space - (2 * s.apple_width + s.apple_space + 2 * s.peach_width)

/-- Theorem: The space between peach trees is 15 feet -/
theorem peach_tree_space_is_15 (s : FruitTreeSpace)
  (h1 : s.apple_width = 10)
  (h2 : s.apple_space = 12)
  (h3 : s.peach_width = 12)
  (h4 : s.total_space = 71) :
  space_between_peach_trees s = 15 := by
  sorry

#eval space_between_peach_trees { apple_width := 10, apple_space := 12, peach_width := 12, total_space := 71 }

end NUMINAMATH_CALUDE_ERRORFEEDBACK_peach_tree_space_is_15_l134_13401


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integer_count_satisfying_conditions_l134_13468

theorem integer_count_satisfying_conditions : 
  ∃! (S : Finset ℤ), (∀ x ∈ S, (-4:ℤ)*x ≥ x + 9 ∧ 
                                (-3:ℤ)*x ≤ 15 ∧ 
                                (-5:ℤ)*x ≥ 3*x + 20 ∧ 
                                x ≥ -8) ∧ 
                     Finset.card S = 3 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_integer_count_satisfying_conditions_l134_13468


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_divisible_by_nine_l134_13477

theorem divisible_by_nine (n : ℕ) : ∃ k : ℤ, (10 ^ n : ℕ) * (9 * n - 1) + 1 = 9 * k := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_divisible_by_nine_l134_13477


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_integer_pairs_l134_13475

theorem count_integer_pairs : 
  (Finset.filter (fun p => p.1 > 0 ∧ p.2 > 0 ∧ p.1^2 - p.2 < 28) (Finset.range 6 ×ˢ Finset.range 26)).card = 55 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_integer_pairs_l134_13475


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_u_2008_equals_4015_l134_13441

/-- Represents the nth term of the sequence described in the problem. -/
def u (n : ℕ) : ℕ := sorry

/-- Represents the sum of the first n natural numbers. -/
def triangular_number (n : ℕ) : ℕ := n * (n + 1) / 2

/-- Represents the last term of the nth group in the sequence. -/
def group_end (n : ℕ) : ℕ := n^2 - n + 2

/-- The main theorem stating that the 2008th term of the sequence is 4015. -/
theorem u_2008_equals_4015 : u 2008 = 4015 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_u_2008_equals_4015_l134_13441


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distinct_word_permutations_l134_13490

/-- 
Given a word with n letters, where some letters may be repeated,
the number of distinct permutations is equal to n! divided by
the product of factorials of the counts of each repeated letter.
-/
theorem distinct_word_permutations (n : ℕ) (letter_counts : List ℕ) 
  (h : letter_counts.sum = n) : 
  ∃ (num_distinct_perms : ℕ),
  num_distinct_perms = (Nat.factorial n) / (letter_counts.map Nat.factorial).prod := by
  sorry

#check distinct_word_permutations

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distinct_word_permutations_l134_13490


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_factors_of_2550_with_more_than_three_l134_13451

noncomputable def num_factors_with_more_than_three (n : ℕ) : ℕ :=
  (Finset.filter (fun d => (Nat.divisors d).card > 3) (Nat.divisors n)).card

theorem factors_of_2550_with_more_than_three :
  num_factors_with_more_than_three 2550 = 9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_factors_of_2550_with_more_than_three_l134_13451


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_length_l134_13430

-- Define the circle
def circle_eq (x y : ℝ) : Prop := (x - 2)^2 + (y - 1)^2 = 2

-- Define the line passing through (1, 1) with slope angle 45°
def line_eq (x y : ℝ) : Prop := x - y = 0

-- Define the point P
def P : ℝ × ℝ := (1, 1)

-- Define the center of the circle
def C : ℝ × ℝ := (2, 1)

-- Define the radius of the circle
noncomputable def r : ℝ := Real.sqrt 2

-- Theorem statement
theorem chord_length : 
  ∃ A B : ℝ × ℝ, 
    circle_eq A.1 A.2 ∧ 
    circle_eq B.1 B.2 ∧ 
    line_eq A.1 A.2 ∧ 
    line_eq B.1 B.2 ∧ 
    Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = Real.sqrt 6 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_length_l134_13430


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sector_angle_answer_is_c_l134_13460

theorem sector_angle (circumference : ℝ) (area : ℝ) (angle : ℝ) :
  circumference = 8 ∧ area = 4 → angle = 2 := by
  intro h
  have h1 : circumference = 8 := h.left
  have h2 : area = 4 := h.right
  
  -- Let l be the arc length and r be the radius
  let l : ℝ := 4  -- We'll prove this later
  let r : ℝ := 2  -- We'll prove this later
  
  -- Prove that 2r + l = 8
  have h3 : 2 * r + l = 8 := by
    -- Proof steps here
    sorry
  
  -- Prove that (1/2) * l * r = 4
  have h4 : (1/2) * l * r = 4 := by
    -- Proof steps here
    sorry
  
  -- Prove that l = 4 and r = 2
  have h5 : l = 4 ∧ r = 2 := by
    -- Proof steps here
    sorry
  
  -- The angle in radians is l / r
  have h6 : angle = l / r := by
    -- Proof steps here
    sorry
  
  -- Therefore, angle = 4 / 2 = 2
  calc
    angle = l / r := h6
    _ = 4 / 2 := by rw [h5.left, h5.right]
    _ = 2 := by norm_num

theorem answer_is_c : 2 = 2 := by rfl

#check sector_angle
#check answer_is_c

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sector_angle_answer_is_c_l134_13460


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_alice_arrives_before_bob_l134_13435

/-- The minimum speed Alice needs to exceed to arrive before Bob -/
noncomputable def alices_min_speed (distance : ℝ) (bobs_speed : ℝ) (alice_delay : ℝ) : ℝ :=
  distance / (distance / bobs_speed - alice_delay)

theorem alice_arrives_before_bob (distance : ℝ) (bobs_speed : ℝ) (alice_delay : ℝ)
    (h1 : distance = 180)
    (h2 : bobs_speed = 40)
    (h3 : alice_delay = 0.5) :
    alices_min_speed distance bobs_speed alice_delay = 45 := by
  sorry

-- Remove the #eval statement as it's not computable
-- #eval alices_min_speed 180 40 0.5

end NUMINAMATH_CALUDE_ERRORFEEDBACK_alice_arrives_before_bob_l134_13435


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_robot_purchase_plan_l134_13488

/-- Represents the daily transport capacity and cost of robots -/
structure Robot where
  transport : ℕ  -- Daily transport capacity in tons
  cost : ℕ       -- Cost in thousands of yuan

/-- Represents the company's robot purchase plan -/
structure PurchasePlan where
  a_robots : ℕ   -- Number of A robots
  b_robots : ℕ   -- Number of B robots

/-- The problem statement and conditions -/
def robot_problem (a : Robot) (b : Robot) : Prop :=
  -- Condition 1: B robots transport 10 tons more than A robots
  b.transport = a.transport + 10 ∧
  -- Condition 2: Equal number of robots needed for 540 and 600 tons
  (540 / a.transport : ℚ) = (600 / b.transport : ℚ) ∧
  -- Condition 3: Costs of robots
  a.cost = 12 ∧ b.cost = 20

/-- Checks if a purchase plan is valid according to the problem constraints -/
def is_valid_plan (a : Robot) (b : Robot) (plan : PurchasePlan) : Prop :=
  -- Condition 4: Total of 30 robots
  plan.a_robots + plan.b_robots = 30 ∧
  -- Condition 5: Daily transport capacity of at least 2830 tons
  plan.a_robots * a.transport + plan.b_robots * b.transport ≥ 2830 ∧
  -- Condition 6: Total cost not exceeding 480,000 yuan
  plan.a_robots * a.cost + plan.b_robots * b.cost ≤ 480

/-- The total cost of a purchase plan in thousands of yuan -/
def total_cost (a : Robot) (b : Robot) (plan : PurchasePlan) : ℕ :=
  plan.a_robots * a.cost + plan.b_robots * b.cost

/-- The theorem to be proved -/
theorem optimal_robot_purchase_plan (a b : Robot) 
  (h : robot_problem a b) :
  ∃ (optimal_plan : PurchasePlan),
    is_valid_plan a b optimal_plan ∧
    optimal_plan.a_robots = 17 ∧
    optimal_plan.b_robots = 13 ∧
    total_cost a b optimal_plan = 464 ∧
    ∀ (plan : PurchasePlan), 
      is_valid_plan a b plan → 
      total_cost a b plan ≥ total_cost a b optimal_plan :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_robot_purchase_plan_l134_13488


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_coordinates_is_74_l134_13463

/- Define the distance from a point to a line -/
def distanceToLine (y : ℝ) : ℝ := |y - 13|

/- Define the distance between two points -/
noncomputable def distanceBetweenPoints (x1 y1 x2 y2 : ℝ) : ℝ :=
  Real.sqrt ((x1 - x2)^2 + (y1 - y2)^2)

/- Main theorem -/
theorem sum_of_coordinates_is_74 :
  ∃ (x1 y1 x2 y2 x3 y3 x4 y4 : ℝ),
    (distanceToLine y1 = 7 ∧ distanceBetweenPoints x1 y1 7 13 = 15) ∧
    (distanceToLine y2 = 7 ∧ distanceBetweenPoints x2 y2 7 13 = 15) ∧
    (distanceToLine y3 = 7 ∧ distanceBetweenPoints x3 y3 7 13 = 15) ∧
    (distanceToLine y4 = 7 ∧ distanceBetweenPoints x4 y4 7 13 = 15) ∧
    x1 + y1 + x2 + y2 + x3 + y3 + x4 + y4 = 74 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_coordinates_is_74_l134_13463


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_volume_with_semicircle_lateral_surface_l134_13482

/-- A cone with base radius 1 and lateral surface that unfolds to a semicircle has volume (√3/3)π -/
theorem cone_volume_with_semicircle_lateral_surface :
  ∀ (cone : Real → Real → Real),
    (∀ r h, cone r h = (1/3) * Real.pi * r^2 * h) →  -- Standard cone volume formula
    (1 : Real) > 0 →  -- Base radius is positive
    (∃ l : Real, l > 0 ∧ l^2 = 1^2 + (Real.sqrt 3)^2) →  -- Pythagorean theorem for slant height
    (2 * Real.pi * 1 = Real.pi * l) →  -- Lateral surface area equation
    cone 1 (Real.sqrt 3) = (Real.sqrt 3 / 3) * Real.pi := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_volume_with_semicircle_lateral_surface_l134_13482


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_theorem_1_theorem_2_l134_13458

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x^2 - a*x + 3

-- Define the function g
noncomputable def g (m : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log 2 + m

-- Theorem 1: If f(1-x) = f(1+x) for all x, then a = 2
theorem theorem_1 (a : ℝ) : (∀ x : ℝ, f a (1-x) = f a (1+x)) → a = 2 := by
  sorry

-- Theorem 2: If f(x_1) > g(x_2) for all x_1, x_2 in [1,4], then m < 2
theorem theorem_2 (m : ℝ) : 
  (∀ x₁ x₂ : ℝ, x₁ ∈ Set.Icc 1 4 → x₂ ∈ Set.Icc 1 4 → f 2 x₁ > g m x₂) → m < 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_theorem_1_theorem_2_l134_13458


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_longest_arithmetic_progression_of_primes_l134_13473

/-- An arithmetic progression of primes with common difference 6 -/
def ArithmeticProgressionOfPrimes (a : ℕ) (n : ℕ) : List ℕ :=
  List.range n |>.map (fun i => a + 6 * i)

/-- Predicate to check if a list is an arithmetic progression of primes with common difference 6 -/
def IsValidProgressionOfPrimes (l : List ℕ) : Prop :=
  l.length > 0 ∧
  (∀ x ∈ l, Nat.Prime x) ∧
  (∀ i : ℕ, i + 1 < l.length → l.get! (i+1) - l.get! i = 6)

/-- The length of the longest arithmetic progression of primes with common difference 6 -/
def LongestProgressionLength : ℕ := 5

theorem longest_arithmetic_progression_of_primes :
  ∀ l : List ℕ, IsValidProgressionOfPrimes l → l.length ≤ LongestProgressionLength :=
by
  sorry

#check longest_arithmetic_progression_of_primes

end NUMINAMATH_CALUDE_ERRORFEEDBACK_longest_arithmetic_progression_of_primes_l134_13473


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallelepiped_parallelogram_analogy_l134_13429

/- A parallelepiped in 3D space has parallel faces that are parallelograms -/
def Parallelepiped : Type := Unit

/- A parallelogram in 2D space -/
def Parallelogram : Type := Unit

/- Analogy between parallelepiped and parallelogram -/
theorem parallelepiped_parallelogram_analogy :
  ∃ (analogy : Parallelepiped → Parallelogram), True :=
by
  -- The existence of an analogy is asserted without proving the details
  exact ⟨λ _ => (), trivial⟩

#check parallelepiped_parallelogram_analogy

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallelepiped_parallelogram_analogy_l134_13429


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_plus_cos_equals_four_thirds_l134_13453

theorem sin_plus_cos_equals_four_thirds :
  ∃ θ : ℝ, 0 < θ ∧ θ < π / 2 ∧ Real.sin θ + Real.cos θ = 4 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_plus_cos_equals_four_thirds_l134_13453


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ramola_rank_l134_13446

theorem ramola_rank (total_students : ℕ) (rank_from_last : ℕ) (rank_from_first : ℕ) :
  total_students = 26 →
  rank_from_last = 13 →
  rank_from_first = total_students - (rank_from_last - 1) →
  rank_from_first = 14 := by
  intros h1 h2 h3
  rw [h1, h2] at h3
  norm_num at h3
  exact h3

#check ramola_rank

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ramola_rank_l134_13446


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_product_one_implies_max_fraction_ge_two_l134_13483

theorem product_one_implies_max_fraction_ge_two
  (a b c d : ℝ)
  (ha : 0 < a)
  (hb : 0 < b)
  (hc : 0 < c)
  (hd : 0 < d)
  (h_prod : a * b * c * d = 1) :
  2 ≤ max ((a^2 + 1) / b^2) (max ((b^2 + 1) / c^2) (max ((c^2 + 1) / d^2) ((d^2 + 1) / a^2))) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_product_one_implies_max_fraction_ge_two_l134_13483


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_b_formula_l134_13470

/-- The sequence a_n satisfying the given recurrence relation -/
def a : ℕ → ℚ
  | 0 => 1  -- Define a₀ to be 1 (same as a₁)
  | 1 => 1
  | n + 2 => 5/2 - 1 / a (n + 1)

/-- The sequence b_n defined in terms of a_n -/
def b (n : ℕ) : ℚ := 1 / (a n - 2)

/-- The theorem stating the general term formula for b_n -/
theorem b_formula (n : ℕ) : b n = 4^(n-1) / 3 - 2/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_b_formula_l134_13470


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_54880000_l134_13408

theorem cube_root_54880000 : (54880000 : ℝ)^(1/3) = 140 * 5^(1/3) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_54880000_l134_13408


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_specific_pentagon_area_l134_13497

/-- Represents a pentagon that can be divided into a right triangle and a trapezoid -/
structure DivisiblePentagon where
  side1 : ℝ
  side2 : ℝ
  side3 : ℝ
  side4 : ℝ
  side5 : ℝ
  triangle_base : ℝ
  triangle_height : ℝ
  trapezoid_side1 : ℝ
  trapezoid_side2 : ℝ
  trapezoid_height : ℝ

/-- Calculates the area of a right triangle -/
noncomputable def triangleArea (base height : ℝ) : ℝ :=
  (1/2) * base * height

/-- Calculates the area of a trapezoid -/
noncomputable def trapezoidArea (side1 side2 height : ℝ) : ℝ :=
  (1/2) * (side1 + side2) * height

/-- Calculates the total area of a DivisiblePentagon -/
noncomputable def pentagonArea (p : DivisiblePentagon) : ℝ :=
  triangleArea p.triangle_base p.triangle_height + trapezoidArea p.trapezoid_side1 p.trapezoid_side2 p.trapezoid_height

/-- Theorem: The area of the specific pentagon is 803 square units -/
theorem specific_pentagon_area :
  let p : DivisiblePentagon := {
    side1 := 17, side2 := 22, side3 := 30, side4 := 26, side5 := 22,
    triangle_base := 22, triangle_height := 17,
    trapezoid_side1 := 26, trapezoid_side2 := 30, trapezoid_height := 22
  }
  pentagonArea p = 803 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_specific_pentagon_area_l134_13497


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_triangle_OPA_l134_13440

noncomputable section

-- Define the points and the line
def A : ℝ × ℝ := (4, 0)
def O : ℝ × ℝ := (0, 0)
def P (x : ℝ) : ℝ × ℝ := (x, -x + 6)

-- Define the conditions
def is_in_first_quadrant (p : ℝ × ℝ) : Prop :=
  p.1 > 0 ∧ p.2 > 0

-- Define the area of the triangle
def triangle_area (a b c : ℝ × ℝ) : ℝ :=
  (1/2) * abs ((b.1 - a.1) * (c.2 - a.2) - (c.1 - a.1) * (b.2 - a.2))

-- State the theorem
theorem area_of_triangle_OPA (x : ℝ) 
  (h1 : is_in_first_quadrant (P x)) 
  (h2 : 0 < x) 
  (h3 : x < 6) :
  triangle_area O (P x) A = 12 - 2*x := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_triangle_OPA_l134_13440


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_base_diameter_l134_13424

noncomputable section

/-- The diameter of a sphere -/
def sphere_diameter : ℝ := 6

/-- The height of the cone -/
def cone_height : ℝ := 3

/-- The volume of a sphere given its diameter -/
noncomputable def sphere_volume (d : ℝ) : ℝ := (1/6) * Real.pi * d^3

/-- The volume of a cone given its base radius and height -/
noncomputable def cone_volume (r : ℝ) (h : ℝ) : ℝ := (1/3) * Real.pi * r^2 * h

/-- The theorem stating that the diameter of the base of the cone is 12 cm -/
theorem cone_base_diameter : 
  ∃ (r : ℝ), 
    sphere_volume sphere_diameter = cone_volume r cone_height ∧ 
    2 * r = 12 := by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_base_diameter_l134_13424


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_l134_13487

noncomputable def f (x : ℝ) := Real.cos x - Real.sqrt 3 * Real.sin x

theorem max_value_of_f :
  ∃ (max : ℝ), max = 1 ∧
  ∀ x, x ∈ Set.Icc 0 (Real.pi / 2) → f x ≤ max ∧
  ∃ x₀, x₀ ∈ Set.Icc 0 (Real.pi / 2) ∧ f x₀ = max := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_l134_13487


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_of_two_fours_l134_13423

-- Define the number of dice
def num_dice : ℕ := 8

-- Define the number of sides on each die
def num_sides : ℕ := 6

-- Define the target number we're looking for
def target_number : ℕ := 4

-- Define the number of target occurrences we want
def target_occurrences : ℕ := 2

-- Define the probability of rolling the target number on a single die
def single_prob : ℚ := 1 / num_sides

-- Define the probability of not rolling the target number on a single die
def single_prob_complement : ℚ := 1 - single_prob

-- Define the function to calculate the probability
def calc_probability : ℚ := 
  (Nat.choose num_dice target_occurrences : ℚ) * 
  single_prob ^ target_occurrences * 
  single_prob_complement ^ (num_dice - target_occurrences)

-- Theorem statement
theorem probability_of_two_fours : 
  (calc_probability * 1000).floor / 1000 = 94 / 1000 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_of_two_fours_l134_13423


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_log_equation_solution_l134_13456

theorem log_equation_solution :
  ∃ y : ℝ, y > 0 ∧ (Real.log 16 / Real.log y = Real.log 4 / Real.log 64) ∧ y = 4096 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_log_equation_solution_l134_13456


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sixteen_classes_120_matches_unique_solution_l134_13437

/-- The number of basketball matches played when x classes participate -/
def num_matches (x : ℕ) : ℕ := x * (x - 1) / 2

/-- Theorem stating that 16 classes result in 120 matches -/
theorem sixteen_classes_120_matches : num_matches 16 = 120 := by
  unfold num_matches
  norm_num

/-- Theorem proving the uniqueness of the solution -/
theorem unique_solution (x : ℕ) (h : x > 1) : num_matches x = 120 ↔ x = 16 := by
  sorry -- Proof omitted for brevity

#eval num_matches 16  -- Should output 120

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sixteen_classes_120_matches_unique_solution_l134_13437


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_sin_C_l134_13459

theorem right_triangle_sin_C (A B C : Real) (h1 : Real.cos A = 0)
  (h2 : Real.tan C = 4 / 3) : Real.sin C = 3 / 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_sin_C_l134_13459


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_increasing_l134_13455

-- Define the function f(x)
noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x) - Real.cos (2 * x)

-- State the theorem
theorem f_monotone_increasing :
  MonotoneOn f (Set.Icc (-Real.pi/8) (3*Real.pi/8)) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_increasing_l134_13455


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mixture_problem_l134_13498

/-- Represents the mixture of petrol and kerosene in a container -/
structure Mixture where
  petrol : ℝ
  kerosene : ℝ

/-- Calculates the ratio of petrol to kerosene in a mixture -/
noncomputable def ratio (m : Mixture) : ℝ := m.petrol / m.kerosene

/-- Removes a certain amount from the mixture, maintaining the same ratio -/
noncomputable def remove (m : Mixture) (amount : ℝ) : Mixture :=
  let total := m.petrol + m.kerosene
  let factor := (total - amount) / total
  { petrol := m.petrol * factor, kerosene := m.kerosene * factor }

/-- Adds a certain amount of kerosene to the mixture -/
def addKerosene (m : Mixture) (amount : ℝ) : Mixture :=
  { petrol := m.petrol, kerosene := m.kerosene + amount }

theorem mixture_problem (initialMixture : Mixture) :
  ratio initialMixture = 3/2 →
  initialMixture.petrol + initialMixture.kerosene = 30 →
  let removedMixture := remove initialMixture 10
  let finalMixture := addKerosene removedMixture 10
  ratio finalMixture = 2/3 →
  finalMixture.petrol + finalMixture.kerosene = 30 := by
  sorry

#check mixture_problem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mixture_problem_l134_13498


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_n_for_power_relation_f_satisfies_power_relation_l134_13416

def f (m : ℕ) : ℕ := m^(m+2)

theorem smallest_n_for_power_relation (m : ℕ) (h_m : m ≥ 2) :
  ∀ n : ℕ, n > m →
    (∀ A B : Set ℕ, A ∪ B = Finset.range (n - m + 1) →
      (∃ a b c, a ∈ A ∧ b ∈ A ∧ c ∈ A ∧ a^b = c) ∨ 
      (∃ a b c, a ∈ B ∧ b ∈ B ∧ c ∈ B ∧ a^b = c)) →
    n ≥ f m :=
by sorry

theorem f_satisfies_power_relation (m : ℕ) (h_m : m ≥ 2) :
  ∀ A B : Set ℕ, A ∪ B = Finset.range (f m - m + 1) →
    (∃ a b c, a ∈ A ∧ b ∈ A ∧ c ∈ A ∧ a^b = c) ∨ 
    (∃ a b c, a ∈ B ∧ b ∈ B ∧ c ∈ B ∧ a^b = c) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_n_for_power_relation_f_satisfies_power_relation_l134_13416


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_a_value_l134_13418

theorem smallest_a_value (a b : ℤ) :
  (∃ x y z : ℕ+, (x : ℤ) * y * z = 2730 ∧ (x : ℤ) + y + z = a ∧ (x : ℤ) * y + y * z + z * x = b) →
  a ≥ 54 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_a_value_l134_13418


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expected_games_played_is_266_over_81_l134_13414

/-- Represents the outcome of a single game -/
inductive GameOutcome
  | PlayerAWins
  | PlayerBWins

/-- Represents the state of the match -/
structure MatchState :=
  (gamesPlayed : ℕ)
  (scoreA : ℕ)
  (scoreB : ℕ)

/-- The probability of Player A winning a single game -/
noncomputable def probPlayerAWins : ℝ := 2/3

/-- The probability of Player B winning a single game -/
noncomputable def probPlayerBWins : ℝ := 1/3

/-- The maximum number of games that can be played -/
def maxGames : ℕ := 6

/-- Checks if the match should end based on the current state -/
def isMatchOver (state : MatchState) : Bool :=
  state.gamesPlayed = maxGames ∨ state.scoreA - state.scoreB = 2 ∨ state.scoreB - state.scoreA = 2

/-- Calculates the expected number of games played -/
noncomputable def expectedGamesPlayed : ℝ := 266/81

/-- Theorem stating that the expected number of games played is 266/81 -/
theorem expected_games_played_is_266_over_81 :
  expectedGamesPlayed = 266/81 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_expected_games_played_is_266_over_81_l134_13414


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_greater_than_10_b_decreases_as_a_increases_l134_13445

/-- Represents the money of person A in dollars -/
def a : ℝ := sorry

/-- Represents the money of person B in dollars -/
def b : ℝ := sorry

/-- The first condition: 8 times A's money plus B's money is more than 160 -/
axiom condition1 : 8 * a + b > 160

/-- The second condition: 4 times A's money plus B's money equals 120 -/
axiom condition2 : 4 * a + b = 120

/-- Theorem stating that A's money is greater than 10 dollars -/
theorem a_greater_than_10 : a > 10 := by sorry

/-- Theorem stating that B's money decreases as A's money increases -/
theorem b_decreases_as_a_increases : 
  ∀ (a1 a2 : ℝ), a1 < a2 → (120 - 4 * a1) > (120 - 4 * a2) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_greater_than_10_b_decreases_as_a_increases_l134_13445


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_double_angle_on_unit_circle_l134_13400

theorem cos_double_angle_on_unit_circle (x : ℝ) (α : ℝ) :
  x^2 + (Real.sqrt 3/2)^2 = 1 → Real.cos (2*α) = -1/2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_double_angle_on_unit_circle_l134_13400


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_boxes_arrangement_l134_13493

/-- A box is a rectangle in the plane with sides parallel to the coordinate axes. -/
structure Box where
  x_min : ℝ
  x_max : ℝ
  y_min : ℝ
  y_max : ℝ
  h_x : x_min < x_max
  h_y : y_min < y_max

/-- Two boxes intersect if they have a common point. -/
def intersect (b1 b2 : Box) : Prop :=
  ¬(b1.x_max ≤ b2.x_min ∨ b2.x_max ≤ b1.x_min ∨ b1.y_max ≤ b2.y_min ∨ b2.y_max ≤ b1.y_min)

/-- A valid arrangement of n boxes satisfies the intersection condition. -/
def valid_arrangement (n : ℕ) (boxes : Fin n → Box) : Prop :=
  ∀ i j : Fin n, intersect (boxes i) (boxes j) ↔ 
    ¬(i.val + 1 = j.val ∨ j.val + 1 = i.val ∨ 
      (i.val = 0 ∧ j.val = n - 1) ∨ (j.val = 0 ∧ i.val = n - 1))

/-- The maximum number of boxes that can be arranged satisfying the intersection condition is 6. -/
theorem max_boxes_arrangement :
  (∃ (n : ℕ) (boxes : Fin n → Box), valid_arrangement n boxes) ∧
  (∀ n : ℕ, n > 6 → ¬∃ (boxes : Fin n → Box), valid_arrangement n boxes) := by
  sorry

#check max_boxes_arrangement

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_boxes_arrangement_l134_13493


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_degree_for_horizontal_asymptote_l134_13478

-- Define the denominator polynomial
def q (x : ℝ) : ℝ := 3 * x^6 - 2 * x^3 + x - 4

-- Define a general polynomial p of degree n
noncomputable def p (n : ℕ) (x : ℝ) : ℝ := sorry

-- Define the rational function
noncomputable def f (n : ℕ) (x : ℝ) : ℝ := p n x / q x

-- Theorem stating that the maximum degree of p for a horizontal asymptote is 6
theorem max_degree_for_horizontal_asymptote :
  ∀ n : ℕ, (∃ (h : ℝ), ∀ ε > 0, ∃ M, ∀ x > M, |f n x - h| < ε) ↔ n ≤ 6 :=
by sorry

#check max_degree_for_horizontal_asymptote

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_degree_for_horizontal_asymptote_l134_13478


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_one_prime_in_set_l134_13461

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

theorem one_prime_in_set : 
  ∃! x, x ∈ ({1, 11, 111, 1111} : Set ℕ) ∧ is_prime x :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_one_prime_in_set_l134_13461


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_intersection_theorem_l134_13415

/-- Represents a parabola that opens upwards -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ
  opens_upward : a > 0

/-- The x-coordinate of the vertex of a parabola -/
noncomputable def vertex_x (p : Parabola) : ℝ := -p.b / (2 * p.a)

/-- The y-coordinate of a point on the parabola given its x-coordinate -/
noncomputable def parabola_y (p : Parabola) (x : ℝ) : ℝ := p.a * x^2 + p.b * x + p.c

theorem parabola_intersection_theorem (π₁ π₂ : Parabola) :
  parabola_y π₁ 10 = 0 →
  parabola_y π₁ 13 = 0 →
  parabola_y π₂ 13 = 0 →
  vertex_x π₁ = (0 + vertex_x π₂) / 2 →
  ∃ t : ℝ, t = 33 ∧ parabola_y π₂ t = 0 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_intersection_theorem_l134_13415


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_evans_books_l134_13419

/-- The number of books Evan owns now -/
def current_books : ℕ := 160

/-- The number of books Evan will have in 5 years -/
def future_books : ℕ := 860

/-- The number of books Evan had 2 years ago -/
def past_books : ℕ := current_books + 40

theorem evans_books : 
  5 * current_books = future_books - 60 ∧
  future_books = 860 ∧
  past_books = current_books + 40 →
  past_books = 200 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_evans_books_l134_13419


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_conic_eccentricity_l134_13439

noncomputable def geometric_mean (a b : ℝ) : ℝ := Real.sqrt (a * b)

noncomputable def ellipse_eccentricity (a b : ℝ) : ℝ := Real.sqrt (1 - b^2 / a^2)

noncomputable def hyperbola_eccentricity (a b : ℝ) : ℝ := Real.sqrt (1 + b^2 / a^2)

theorem conic_eccentricity (m : ℝ) :
  m = geometric_mean 2 8 →
  (∃ (e : ℝ), (e = ellipse_eccentricity 2 1 ∨ e = hyperbola_eccentricity 1 2) ∧
              (∀ (x y : ℝ), x^2 + y^2/m = 1 → e = ellipse_eccentricity 2 1 ∨ e = hyperbola_eccentricity 1 2)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_conic_eccentricity_l134_13439


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_property_l134_13495

/-- Represents a hyperbola -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

def dot_product (v1 v2 : Point) : ℝ :=
  v1.x * v2.x + v1.y * v2.y

theorem hyperbola_property (M : Hyperbola) (F₁ F₂ P : Point) :
  (∀ x y : ℝ, y = (Real.sqrt 7 / 3) * x → (x^2 / M.a^2 - y^2 / M.b^2 = 1)) →  -- asymptote condition
  F₁.x = -4 →  -- directrix passes through a focus
  F₁.y = 0 ∧ F₂.y = 0 →  -- foci on x-axis
  (P.x^2 / M.a^2 - P.y^2 / M.b^2 = 1) →  -- P is on the hyperbola
  dot_product (Point.mk (P.x - F₁.x) (P.y - F₁.y)) (Point.mk (P.x - F₂.x) (P.y - F₂.y)) = 0 →
  distance P F₁ * distance P F₂ = 14 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_property_l134_13495


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_visible_area_one_triangle_visible_area_both_triangles_l134_13469

/-- Represents the figure with a square and two isosceles triangles -/
structure PaperFigure where
  squareSideLength : ℝ
  triangleSideLength : ℝ
  triangleSidesEqual : triangleSideLength = squareSideLength

/-- Calculates the area of the square -/
noncomputable def squareArea (p : PaperFigure) : ℝ :=
  p.squareSideLength * p.squareSideLength

/-- Calculates the area of one isosceles triangle -/
noncomputable def triangleArea (p : PaperFigure) : ℝ :=
  1 / 2 * p.triangleSideLength * p.triangleSideLength

/-- Theorem: The visible area when one triangle is folded over is 50 cm² -/
theorem visible_area_one_triangle (p : PaperFigure) 
  (h : p.squareSideLength = 10) : 
  squareArea p - triangleArea p = 50 := by
  sorry

/-- Theorem: The visible area when both triangles are folded over is 25 cm² -/
theorem visible_area_both_triangles (p : PaperFigure) 
  (h : p.squareSideLength = 10) : 
  squareArea p / 4 = 25 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_visible_area_one_triangle_visible_area_both_triangles_l134_13469


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_properties_l134_13450

-- Define the sequence a_n
def a (c : ℕ+) : ℕ → ℚ
| 0 => c / (c + 1)
| n + 1 => let prev := a c n; (prev.num + 2) / (prev.den + 3)

-- Function to check if a fraction needs simplification
def needs_simplification (q : ℚ) : Bool :=
  q.num.gcd q.den ≠ 1

-- Function to check if both numerator and denominator are divisible by 5
def both_div_by_5 (q : ℚ) : Bool :=
  q.num % 5 = 0 ∧ q.den % 5 = 0

theorem sequence_properties :
  (∃ n : ℕ, n = 7 ∧ needs_simplification (a 10 n) ∧ ∀ m > n, ¬needs_simplification (a 10 m)) ∧
  (∀ n : ℕ, ¬needs_simplification (a 99 n)) ∧
  both_div_by_5 ((a 7 4).num + 2 / (a 7 4).den + 3) ∧
  both_div_by_5 ((a 27 4).num + 2 / (a 27 4).den + 3) :=
by sorry

#eval a 10 0
#eval a 10 1
#eval a 10 2
#eval a 10 3

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_properties_l134_13450


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_even_sin_shifted_sin_sum_squares_range_l134_13442

-- Define the function f
noncomputable def f (x : ℝ) := Real.sin x

-- Part I
theorem even_sin_shifted (θ : ℝ) (h1 : 0 ≤ θ) (h2 : θ < 2 * Real.pi) :
  (∀ x, f (x + θ) = f (-x - θ)) → θ = Real.pi / 2 ∨ θ = 3 * Real.pi / 2 := by
  sorry

-- Part II
theorem sin_sum_squares_range :
  Set.range (λ x => (f (x + Real.pi / 12))^2 + (f (x + Real.pi / 4))^2) =
  Set.Icc (1 - Real.sqrt 3 / 2) (1 + Real.sqrt 3 / 2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_even_sin_shifted_sin_sum_squares_range_l134_13442


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_lambda_value_l134_13480

-- Define the inequality function
def inequality (a b c lambda : ℝ) : Prop :=
  a * b + b^2 + c^2 ≥ lambda * (a + b) * c

-- State the theorem
theorem max_lambda_value :
  (∃ (lambda_max : ℝ), 
    (∀ (a b c : ℝ), a > 0 → b > 0 → c > 0 → b + c ≥ a → inequality a b c lambda_max) ∧ 
    (∀ (lambda' : ℝ), lambda' > lambda_max → 
      ∃ (a b c : ℝ), a > 0 ∧ b > 0 ∧ c > 0 ∧ b + c ≥ a ∧ ¬(inequality a b c lambda'))) ∧
  (∀ (lambda_max : ℝ), 
    ((∀ (a b c : ℝ), a > 0 → b > 0 → c > 0 → b + c ≥ a → inequality a b c lambda_max) ∧ 
     (∀ (lambda' : ℝ), lambda' > lambda_max → 
       ∃ (a b c : ℝ), a > 0 ∧ b > 0 ∧ c > 0 ∧ b + c ≥ a ∧ ¬(inequality a b c lambda')))
    → lambda_max = Real.sqrt 2 - 1/2) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_lambda_value_l134_13480


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_four_digit_number_l134_13499

def digits : List Nat := [9, 4, 1, 5]

def is_valid_number (n : Nat) : Prop :=
  n ≥ 1000 ∧ n < 10000 ∧ 
  (List.toFinset (List.map (λ d => d % 10) (List.reverse (Nat.digits 10 n)))) = List.toFinset digits

theorem largest_four_digit_number : 
  ∀ n : Nat, is_valid_number n → n ≤ 9541 :=
by sorry

#check largest_four_digit_number

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_four_digit_number_l134_13499


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_students_in_line_l134_13403

def students_in_line (front_person : ℕ) (people_between_front_and_namjoon : ℕ) (people_behind_namjoon : ℕ) : ℕ :=
  front_person + people_between_front_and_namjoon + 1 + people_behind_namjoon

theorem total_students_in_line :
  students_in_line 1 3 8 = 13 := by
  rfl

#eval students_in_line 1 3 8

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_students_in_line_l134_13403


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_regression_line_correct_l134_13425

/-- Data point representing an (x, y) coordinate --/
structure DataPoint where
  x : ℝ
  y : ℝ

/-- Calculate the mean of a list of real numbers --/
noncomputable def mean (list : List ℝ) : ℝ :=
  list.sum / list.length

/-- Check if a line equation passes through a point --/
def line_passes_through (m a : ℝ) (point : DataPoint) : Prop :=
  point.y = m * point.x + a

/-- The regression line for a set of data points --/
def regression_line (data : List DataPoint) (m a : ℝ) : Prop :=
  let x_mean := mean (data.map (·.x))
  let y_mean := mean (data.map (·.y))
  y_mean = m * x_mean + a

theorem regression_line_correct (data : List DataPoint) : 
  data = [⟨1, 3⟩, ⟨2, 3.8⟩, ⟨3, 5.2⟩, ⟨4, 6⟩] →
  regression_line data 1.04 1.9 ∧ 
  (∀ point ∈ data, line_passes_through 1.04 1.9 point) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_regression_line_correct_l134_13425


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_magnitude_a_l134_13436

open Complex Real

-- Define a primitive cube root of unity
noncomputable def ω : ℂ := exp (2 * Real.pi * I / 3)

theorem max_magnitude_a (a b : ℂ) 
  (eq1 : a^3 - 3*a*b^2 = 36)
  (eq2 : b^3 - 3*b*a^2 = 28*I) :
  (∃ (M : ℝ), ∀ (a' : ℂ), (a'^3 - 3*a'*b^2 = 36 ∧ b^3 - 3*b*a'^2 = 28*I) → abs a' ≤ M) ∧
  (abs a = 3 ↔ a ∈ ({3, 3*ω, 3*ω^2} : Set ℂ)) := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_magnitude_a_l134_13436


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_carpet_needed_per_section_l134_13449

/-- Represents the dimensions of a rectangular room in feet -/
structure RoomDimensions where
  length : ℝ
  width : ℝ

/-- Calculates the area of a room in square feet -/
noncomputable def roomArea (d : RoomDimensions) : ℝ :=
  d.length * d.width

/-- Converts square feet to square yards -/
noncomputable def sqFeetToSqYards (sqFeet : ℝ) : ℝ :=
  sqFeet / 9

/-- The number of sections the room is divided into -/
def numSections : ℕ := 2

theorem carpet_needed_per_section (room : RoomDimensions)
    (h1 : room.length = 15)
    (h2 : room.width = 9) :
    sqFeetToSqYards (roomArea room / numSections) = 7.5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_carpet_needed_per_section_l134_13449


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_angles_E_and_G_l134_13484

-- Define the basic types
def Point : Type := ℝ × ℝ

-- Define the triangle structure
structure Triangle :=
  (A B C : Point)

-- Define the angle function (noncomputable as it involves trigonometry)
noncomputable def angle (p q r : Point) : ℝ := sorry

-- Define the main theorem
theorem sum_of_angles_E_and_G (ABC : Triangle) (E F G : Point) :
  angle ABC.A ABC.B ABC.C = 30 →
  angle ABC.B ABC.A ABC.C = angle ABC.B ABC.C ABC.A →
  angle E F G = angle E G F →
  angle E F ABC.A = angle E ABC.A ABC.B →
  angle G F ABC.B = angle G ABC.B ABC.A →
  angle E ABC.A ABC.B + angle G ABC.B ABC.A = 30 := by
  sorry

#check sum_of_angles_E_and_G

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_angles_E_and_G_l134_13484


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_max_min_on_interval_l134_13420

-- Define the function f(x) = ln(1+x) - (1/4)x²
noncomputable def f (x : ℝ) : ℝ := Real.log (1 + x) - (1/4) * x^2

-- Theorem statement
theorem f_max_min_on_interval :
  let a := 0
  let b := 2
  ∃ (x_max x_min : ℝ), x_max ∈ Set.Icc a b ∧ x_min ∈ Set.Icc a b ∧
    (∀ x ∈ Set.Icc a b, f x ≤ f x_max) ∧
    (∀ x ∈ Set.Icc a b, f x_min ≤ f x) ∧
    f x_max = Real.log 2 - 1/4 ∧
    f x_min = 0 := by
  sorry

#check f_max_min_on_interval

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_max_min_on_interval_l134_13420


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_f_equals_neg_345_l134_13402

-- Define the functions f and g
def f : ℝ → ℝ := sorry
def g : ℝ → ℝ := sorry

-- State the given conditions
axiom cond1 : ∀ x, f x - g (2 - x) = 4
axiom cond2 : ∀ x, g x + f (x - 4) = 6
axiom cond3 : ∀ x, g (3 - x) + g (x + 1) = 0

-- Define the sum of f(n) from 1 to 30
def sum_f : ℕ → ℝ
  | 0 => 0
  | n + 1 => sum_f n + f (n + 1 : ℝ)

-- State the theorem to be proved
theorem sum_f_equals_neg_345 : sum_f 30 = -345 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_f_equals_neg_345_l134_13402


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_nonnegative_num_zeros_correct_l134_13457

noncomputable section

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 - 1 - 2 * Real.log x

-- Theorem 1: f(x) ≥ 0 for all x > 0 when a ≥ 1
theorem f_nonnegative (a : ℝ) (h : a ≥ 1) : ∀ x > 0, f a x ≥ 0 := by
  sorry

-- Define the number of zeros
def num_zeros (a : ℝ) : Nat :=
  if 0 < a ∧ a < 1 then 2
  else if a = 1 then 1
  else 0

-- Theorem 2: Correctness of num_zeros
theorem num_zeros_correct (a : ℝ) : 
  ((∃ x y : ℝ, x ≠ y ∧ x > 0 ∧ y > 0 ∧ f a x = 0 ∧ f a y = 0) ↔ num_zeros a = 2) ∧
  ((∃! x : ℝ, x > 0 ∧ f a x = 0) ↔ num_zeros a = 1) ∧
  ((∀ x > 0, f a x ≠ 0) ↔ num_zeros a = 0) := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_nonnegative_num_zeros_correct_l134_13457


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_A_measure_l134_13438

theorem angle_A_measure (A B C : ℝ) (a b c : ℝ) : 
  C = 60 * Real.pi / 180 →
  b = Real.sqrt 6 →
  c = 3 →
  Real.sin C = b / c →
  A + B + C = Real.pi →
  A = 75 * Real.pi / 180 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_A_measure_l134_13438


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_new_poly_roots_new_poly_is_correct_l134_13465

-- Define the original polynomial
def original_poly (x : ℝ) : ℝ := x^3 - 5*x^2 + 16

-- Define the roots of the original polynomial as variables
variable (r₁ r₂ r₃ : ℝ)

-- State that r₁, r₂, r₃ are roots of the original polynomial
axiom root_def₁ : original_poly r₁ = 0
axiom root_def₂ : original_poly r₂ = 0
axiom root_def₃ : original_poly r₃ = 0

-- Define the new polynomial
def new_poly (x : ℝ) : ℝ := x^3 - 15*x^2 + 432

-- Theorem statement
theorem new_poly_roots : 
  new_poly (3*r₁) = 0 ∧ new_poly (3*r₂) = 0 ∧ new_poly (3*r₃) = 0 := by
  sorry

-- Additional theorem to show that new_poly is indeed the required polynomial
theorem new_poly_is_correct : 
  ∀ x, new_poly x = x^3 - 15*x^2 + 432 := by
  intro x
  rfl

end NUMINAMATH_CALUDE_ERRORFEEDBACK_new_poly_roots_new_poly_is_correct_l134_13465


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallelogram_H_coordinates_l134_13432

def E : ℝ × ℝ × ℝ := (2, 3, -1)
def F : ℝ × ℝ × ℝ := (0, 5, 3)
def G : ℝ × ℝ × ℝ := (-4, 0, 4)

def is_parallelogram (A B C D : ℝ × ℝ × ℝ) : Prop :=
  (A.1 + C.1 = B.1 + D.1) ∧ 
  (A.2.1 + C.2.1 = B.2.1 + D.2.1) ∧ 
  (A.2.2 + C.2.2 = B.2.2 + D.2.2)

theorem parallelogram_H_coordinates :
  ∃ H : ℝ × ℝ × ℝ, is_parallelogram E F G H ∧ H = (-2, -2, 0) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallelogram_H_coordinates_l134_13432


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_three_punches_sufficient_l134_13412

/-- A punch removes points with irrational distance from its center -/
def Punch := ℝ × ℝ

/-- A point is removed if its distance from the center of a punch is irrational -/
def isRemoved (p : ℝ × ℝ) (punch : Punch) : Prop :=
  Irrational (Real.sqrt ((p.1 - punch.1)^2 + (p.2 - punch.2)^2))

/-- Three punches at (0,0), (1,0), and (π,0) -/
noncomputable def threePunches : List Punch := [(0, 0), (1, 0), (Real.pi, 0)]

theorem three_punches_sufficient :
  ∀ p : ℝ × ℝ, ∃ punch ∈ threePunches, isRemoved p punch :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_three_punches_sufficient_l134_13412


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_point_M_coordinates_one_intersection_point_two_intersection_points_no_intersection_points_l134_13485

-- Define the parabola C
def C (x y : ℝ) : Prop := y^2 = -4*x

-- Define the focus of the parabola
def focus : ℝ × ℝ := (1, 0)

-- Define the distance function
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- Define the line l
def l (k : ℝ) (x : ℝ) : ℝ := k*x - k + 2

-- Theorem for part (I)
theorem point_M_coordinates :
  ∀ x y : ℝ, C x y → distance (x, y) focus = 5 →
    ((x = -4 ∧ y = 4) ∨ (x = -4 ∧ y = -4)) := by sorry

-- Theorems for part (II)
theorem one_intersection_point :
  ∀ k : ℝ, (k = 0 ∨ k = 1 + Real.sqrt 2 ∨ k = 1 - Real.sqrt 2) ↔
    ∃! x y : ℝ, C x y ∧ y = l k x := by sorry

theorem two_intersection_points :
  ∀ k : ℝ, (1 - Real.sqrt 2 < k ∧ k < 1 + Real.sqrt 2) ↔
    ∃ x₁ y₁ x₂ y₂ : ℝ, x₁ ≠ x₂ ∧ C x₁ y₁ ∧ C x₂ y₂ ∧ y₁ = l k x₁ ∧ y₂ = l k x₂ := by sorry

theorem no_intersection_points :
  ∀ k : ℝ, (k > 1 + Real.sqrt 2 ∨ k < 1 - Real.sqrt 2) ↔
    ¬∃ x y : ℝ, C x y ∧ y = l k x := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_point_M_coordinates_one_intersection_point_two_intersection_points_no_intersection_points_l134_13485


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_sine_sum_l134_13492

theorem cosine_sine_sum (θ : Real) (h1 : Real.tan θ = 5/12) (h2 : π ≤ θ ∧ θ ≤ 3*π/2) : 
  Real.cos θ + Real.sin θ = -17/13 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_sine_sum_l134_13492


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_l134_13481

noncomputable def f (x : ℝ) : ℝ := Real.sin x ^ 4 - Real.sin x * Real.cos x + Real.cos x ^ 4

theorem f_range :
  (∀ x : ℝ, 0 ≤ f x ∧ f x ≤ 1) ∧
  (∃ y : ℝ, f y = 0) ∧
  (∃ z : ℝ, f z = 1) := by
  sorry

#check f_range

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_l134_13481


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_area_covered_l134_13471

/-- Represents a rectangular strip with length and width -/
structure Strip where
  length : ℕ
  width : ℕ

/-- Calculates the area of a strip -/
def stripArea (s : Strip) : ℕ := s.length * s.width

/-- Calculates the overlap area between two perpendicular strips -/
def overlapArea (s1 s2 : Strip) : ℕ := min s1.width s2.width * min s1.length s2.length

/-- The set of strips on the table -/
def strips : List Strip := [
  ⟨12, 1⟩, ⟨12, 1⟩, ⟨8, 2⟩, ⟨8, 2⟩
]

/-- The theorem to prove -/
theorem total_area_covered :
  (List.sum (List.map stripArea strips)) - (List.sum (List.map (fun p => overlapArea p.1 p.2) (List.product strips strips))) / 2 = 48 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_area_covered_l134_13471


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_difference_l134_13462

-- Define a fair coin
def fair_coin_prob : ℚ := 1/2

-- Define the number of flips
def num_flips : ℕ := 5

-- Define the probability of exactly k heads in n flips
def prob_k_heads (n k : ℕ) : ℚ :=
  (Nat.choose n k : ℚ) * fair_coin_prob^k * (1 - fair_coin_prob)^(n-k)

-- Define the probability of exactly 3 heads in 5 flips
def prob_3_heads : ℚ := prob_k_heads num_flips 3

-- Define the probability of exactly 5 heads in 5 flips
def prob_5_heads : ℚ := prob_k_heads num_flips 5

-- Theorem statement
theorem prob_difference : prob_3_heads - prob_5_heads = 9/32 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_difference_l134_13462


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_sum_l134_13428

-- Define the parabola
def is_on_parabola (x y : ℝ) : Prop := y^2 = 8*x

-- Define the focus of the parabola
def focus : ℝ × ℝ := (2, 0)

-- Define the circle
def is_on_circle (x y : ℝ) : Prop := (x - 3)^2 + (y + 1)^2 = 1

-- Define the distance function
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- State the theorem
theorem min_distance_sum :
  ∃ (min_val : ℝ), min_val = 4 ∧
  ∀ (mx my ax ay : ℝ),
    is_on_parabola mx my →
    is_on_circle ax ay →
    distance (mx, my) (ax, ay) + distance (mx, my) focus ≥ min_val :=
by
  -- Proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_sum_l134_13428


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_downstream_distance_is_81_l134_13426

-- Define the constants given in the problem
def upstream_distance : ℚ := 36
def time_per_direction : ℚ := 9
def current_speed : ℚ := 5/2

-- Define the function to calculate the downstream distance
noncomputable def downstream_distance (v : ℚ) : ℚ :=
  time_per_direction * (v + current_speed)

-- Define the function to calculate the upstream time
noncomputable def upstream_time (v : ℚ) : ℚ :=
  upstream_distance / (v - current_speed)

-- Theorem statement
theorem downstream_distance_is_81 :
  ∃ v : ℚ, v > current_speed ∧ 
    upstream_time v = time_per_direction ∧
    downstream_distance v = 81 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_downstream_distance_is_81_l134_13426


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_iff_isosceles_120_l134_13452

-- Define the triangle ABC
structure Triangle :=
  (A B C : ℂ)

-- Define the similarity condition
def similar_triangles (T : Triangle) (A₁ B₁ C₁ : ℂ) : Prop :=
  ∃ z : ℂ,
    A₁ = T.B + (T.C - T.B) * z ∧
    B₁ = T.C + (T.A - T.C) * z ∧
    C₁ = T.A + (T.B - T.A) * z

-- Define the equilateral condition for A₁B₁C₁
def is_equilateral (A₁ B₁ C₁ : ℂ) : Prop :=
  Complex.abs (A₁ - B₁) = Complex.abs (B₁ - C₁) ∧ Complex.abs (B₁ - C₁) = Complex.abs (C₁ - A₁)

-- Define the isosceles condition with 120° angle
def is_isosceles_120 (A B C : ℂ) : Prop :=
  Complex.abs (A - B) = Complex.abs (A - C) ∧ 
  (B - A).re * (C - A).re + (B - A).im * (C - A).im = 
    -Complex.abs (B - A) * Complex.abs (C - A) / 2

-- Main theorem
theorem equilateral_iff_isosceles_120 (T : Triangle) (A₁ B₁ C₁ : ℂ) 
  (h_non_equilateral : ¬ is_equilateral T.A T.B T.C)
  (h_similar : similar_triangles T A₁ B₁ C₁) :
  is_equilateral A₁ B₁ C₁ ↔ 
    is_isosceles_120 T.B A₁ T.C ∧ 
    is_isosceles_120 T.C B₁ T.A ∧ 
    is_isosceles_120 T.A C₁ T.B :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_iff_isosceles_120_l134_13452


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_from_point_to_line_l134_13444

-- Define the point in polar coordinates
noncomputable def polar_point : ℝ × ℝ := (2, Real.pi/6)

-- Define the line equation in polar coordinates
def polar_line (ρ θ : ℝ) : Prop := ρ * Real.sin θ = 2

-- Define the distance function
noncomputable def distance_to_line (p : ℝ × ℝ) (line : ℝ → ℝ → Prop) : ℝ :=
  sorry -- Actual implementation would go here

-- Theorem statement
theorem distance_from_point_to_line :
  distance_to_line polar_point polar_line = 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_from_point_to_line_l134_13444


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tower_surface_area_l134_13496

def cube_volumes : List ℕ := [1, 27, 125, 343, 512, 729, 1000, 1331]

def is_decreasing (l : List ℕ) : Prop :=
  ∀ i j, i < j → j < l.length → l.get! i > l.get! j

def rotated_stack (l : List ℕ) : Prop :=
  l.length > 1 ∧ ∀ i, 1 ≤ i → i < l.length → true  -- Simplified rotation condition

theorem tower_surface_area (h_decreasing : is_decreasing cube_volumes)
                           (h_rotated : rotated_stack cube_volumes) :
  (cube_volumes.map (λ v => 6 * (v^(1/3) : ℝ)^2 - (v^(1/3) : ℝ)^2)).sum = 2250 := by
  sorry

#eval (cube_volumes.map (λ v => 6 * (v^(1/3) : ℝ)^2 - (v^(1/3) : ℝ)^2)).sum

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tower_surface_area_l134_13496


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l134_13410

-- Define the function f(x) = -cos(2x)
noncomputable def f (x : ℝ) : ℝ := -Real.cos (2 * x)

-- State the theorem
theorem f_properties :
  -- 1. f is an even function
  (∀ x, f x = f (-x)) ∧
  -- 2. f has a period of π
  (∀ x, f (x + π) = f x) ∧
  -- 3. f is monotonically increasing in (0, π/4)
  (∀ x y, 0 < x ∧ x < y ∧ y < π/4 → f x < f y) :=
by
  -- We use 'sorry' to skip the proof for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l134_13410


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_strictly_decreasing_f_implies_a_range_l134_13486

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 1 then (3*a - 2)*x + 6*a - 1 else a^x

theorem strictly_decreasing_f_implies_a_range (a : ℝ) :
  (∀ x y : ℝ, x < y → f a x > f a y) →
  a ≥ 3/8 ∧ a < 2/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_strictly_decreasing_f_implies_a_range_l134_13486


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_alpha_value_P_on_terminal_side_l134_13421

-- Define the point P
def P : ℝ × ℝ := (-3, -4)

-- Define the angle α
noncomputable def α : ℝ := Real.arctan (P.2 / P.1) + Real.pi

-- Theorem statement
theorem sin_alpha_value : 
  Real.sin α = -4/5 := by
  sorry

-- Helper theorem to show that P is on the terminal side of α
theorem P_on_terminal_side : 
  ∃ (t : ℝ), t > 0 ∧ (t * P.1, t * P.2) = (Real.cos α, Real.sin α) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_alpha_value_P_on_terminal_side_l134_13421


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_solutions_l134_13472

theorem infinite_solutions (k : ℕ) (hk : k ≥ 8) :
  (∃ x y : ℕ+, (x ∣ y^2 - 3) ∧ (y ∣ x^2 - 2) ∧
    Nat.gcd ((3*x : ℕ) + 2*(y^2-3)/x) ((2*y : ℕ) + 3*(x^2-2)/y) = k) →
  ∃ (S : Set (ℕ+ × ℕ+)), Set.Infinite S ∧
    ∀ (p : ℕ+ × ℕ+), p ∈ S →
      let (x, y) := p
      (x ∣ y^2 - 3) ∧ (y ∣ x^2 - 2) ∧
      Nat.gcd ((3*x : ℕ) + 2*(y^2-3)/x) ((2*y : ℕ) + 3*(x^2-2)/y) = k :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_solutions_l134_13472


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_at_least_three_hits_l134_13413

-- Define the probability of hitting the target
def hit_probability : ℝ := 0.8

-- Define the number of shots
def total_shots : ℕ := 4

-- Define the minimum number of hits we're interested in
def min_hits : ℕ := 3

-- Theorem statement
theorem probability_at_least_three_hits : 
  Finset.sum (Finset.range (total_shots - min_hits + 1)) (fun k => 
    (total_shots.choose (total_shots - k)) * 
    hit_probability^(total_shots - k) * 
    (1 - hit_probability)^k) = 0.8192 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_at_least_three_hits_l134_13413


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_reflection_symmetry_l134_13467

-- Define the necessary types and structures
structure Point where
  x : ℝ
  y : ℝ

structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

-- Define the functions we need
def symmetric (A A' : Point) (l : Line) : Prop := sorry

def on_line (P : Point) (l : Line) : Prop := sorry

-- Change the angle function to take three Points
def angle (P Q R : Point) : ℝ := sorry

-- State the theorem
theorem reflection_symmetry 
  (PQ : Line) (A B L A' : Point) :
  on_line L PQ →
  symmetric A A' PQ →
  angle A L B = angle A' L B →  -- Changed to use B instead of PQ
  on_line A' (Line.mk (B.y - L.y) (L.x - B.x) (B.x * L.y - L.x * B.y)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_reflection_symmetry_l134_13467


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_cylinder_ratio_theorem_l134_13491

noncomputable def cone_cylinder_volume_ratio (α β : Real) : Real :=
  (Real.cos α)^3 * (Real.cos β)^3 / (3 * Real.sin α * Real.sin β * (Real.cos (α + β))^2)

theorem cone_cylinder_ratio_theorem (α β R : Real) :
  let cone_volume := Real.pi * R^3 * (Real.cos β / Real.sin β) / 3
  let cylinder_volume := Real.pi * R^3 * (Real.cos (α + β))^2 * (Real.sin α / Real.cos α) / ((Real.cos α)^2 * (Real.cos β)^2)
  cone_volume / cylinder_volume = cone_cylinder_volume_ratio α β :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_cylinder_ratio_theorem_l134_13491


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_symmetric_point_on_ellipse_l134_13489

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents an ellipse -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_pos : 0 < b ∧ b < a

/-- Checks if a point lies on the ellipse -/
def onEllipse (p : Point) (e : Ellipse) : Prop :=
  p.x^2 / e.a^2 + p.y^2 / e.b^2 = 1

/-- Calculates the focal length of an ellipse -/
noncomputable def focalLength (e : Ellipse) : ℝ :=
  2 * Real.sqrt (e.a^2 - e.b^2)

/-- Finds the symmetric point of A with respect to the line y = kx + 1 -/
noncomputable def symmetricPoint (A : Point) (k : ℝ) : Point :=
  let m := -1/k
  let c := 1 - m * (A.x + (A.y - 1) / k) / 2
  { x := (A.x + (A.y - 1) / k), y := m * (A.x + (A.y - 1) / k) + c }

/-- The main theorem -/
theorem no_symmetric_point_on_ellipse :
  ∀ (e : Ellipse),
    e.a^2 = 3 ∧ e.b^2 = 1 →
    focalLength e = 2 * Real.sqrt 2 →
    let A : Point := { x := 3/2, y := -1/2 }
    onEllipse A e →
    ∀ (k : ℝ),
      let B := symmetricPoint A k
      B ≠ A →
      ¬ onEllipse B e :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_symmetric_point_on_ellipse_l134_13489


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_squares_theorem_l134_13406

/-- Represents a chess board -/
structure ChessBoard where
  size : Nat
  valid_move : Nat → Nat → Bool
  start_anywhere : Bool

/-- Represents the specific chess piece movement -/
def specific_move (n : Nat) : Bool :=
  n = 8 ∨ n = 9

/-- The chess board for our problem -/
def our_board : ChessBoard :=
  { size := 15
    valid_move := λ x y ↦ specific_move x ∨ specific_move y
    start_anywhere := true }

/-- The maximum number of visitable squares -/
def max_visitable_squares (board : ChessBoard) : Nat :=
  board.size * board.size - (2 * board.size - 1)

/-- Theorem stating the maximum number of visitable squares for our specific board -/
theorem max_squares_theorem :
  max_visitable_squares our_board = 196 := by
  sorry

#eval max_visitable_squares our_board

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_squares_theorem_l134_13406


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_guacamole_theorem_l134_13447

/-- Represents the problem of calculating guacamole servings and cost --/
theorem guacamole_theorem (x : ℝ) :
  let avocados_per_serving : ℕ := 3
  let initial_avocados : ℕ := 5
  let bought_avocados : ℕ := 4
  let total_avocados : ℕ := initial_avocados + bought_avocados
  let servings : ℕ := total_avocados / avocados_per_serving
  let total_cost : ℝ := x * (bought_avocados : ℝ)
  (servings = 3) ∧ (total_cost = 4 * x) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_guacamole_theorem_l134_13447


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cyclic_sum_simplification_l134_13417

/-- Given distinct real numbers a, b, and c, the function
    f(x) = ∑ (a^2 * (x-b) * (x-c)) / ((a-b) * (a-c))
    is equal to x^2. -/
theorem cyclic_sum_simplification 
  (a b c : ℝ) 
  (hab : a ≠ b) (hbc : b ≠ c) (hca : c ≠ a) :
  let f : ℝ → ℝ := λ x ↦ 
    (a^2 * (x - b) * (x - c)) / ((a - b) * (a - c)) +
    (b^2 * (x - c) * (x - a)) / ((b - c) * (b - a)) +
    (c^2 * (x - a) * (x - b)) / ((c - a) * (c - b))
  ∀ x, f x = x^2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cyclic_sum_simplification_l134_13417
