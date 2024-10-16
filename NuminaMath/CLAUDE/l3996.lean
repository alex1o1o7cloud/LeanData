import Mathlib

namespace NUMINAMATH_CALUDE_second_hole_depth_l3996_399631

/-- Calculates the depth of a second hole given the conditions of two digging scenarios -/
theorem second_hole_depth
  (workers₁ : ℕ) (hours₁ : ℕ) (depth₁ : ℝ)
  (workers₂ : ℕ) (hours₂ : ℕ) :
  workers₁ = 45 →
  hours₁ = 8 →
  depth₁ = 30 →
  workers₂ = workers₁ + 45 →
  hours₂ = 6 →
  (workers₂ * hours₂ : ℝ) * (depth₁ / (workers₁ * hours₁ : ℝ)) = 45 :=
by sorry

end NUMINAMATH_CALUDE_second_hole_depth_l3996_399631


namespace NUMINAMATH_CALUDE_smaller_angle_is_45_degrees_l3996_399619

/-- A parallelogram with a specific angle ratio -/
structure AngleRatioParallelogram where
  -- The measure of the smaller interior angle
  small_angle : ℝ
  -- The measure of the larger interior angle
  large_angle : ℝ
  -- The ratio of the angles is 1:3
  angle_ratio : small_angle * 3 = large_angle
  -- The angles are supplementary (add up to 180°)
  supplementary : small_angle + large_angle = 180

/-- The theorem stating that the smaller angle in the parallelogram is 45° -/
theorem smaller_angle_is_45_degrees (p : AngleRatioParallelogram) : p.small_angle = 45 := by
  sorry


end NUMINAMATH_CALUDE_smaller_angle_is_45_degrees_l3996_399619


namespace NUMINAMATH_CALUDE_is_fractional_expression_example_l3996_399652

/-- A fractional expression is an expression where the denominator contains a variable. -/
def IsFractionalExpression (n d : ℝ → ℝ) : Prop :=
  ∃ x, d x ≠ 0 ∧ (∀ y, d y ≠ d x)

/-- The expression (x + 3) / x is a fractional expression. -/
theorem is_fractional_expression_example :
  IsFractionalExpression (λ x => x + 3) (λ x => x) := by
  sorry

end NUMINAMATH_CALUDE_is_fractional_expression_example_l3996_399652


namespace NUMINAMATH_CALUDE_committee_selection_probability_l3996_399687

theorem committee_selection_probability :
  let total_members : ℕ := 9
  let english_teachers : ℕ := 3
  let select_count : ℕ := 2

  let total_combinations := total_members.choose select_count
  let english_combinations := english_teachers.choose select_count

  (english_combinations : ℚ) / total_combinations = 1 / 12 :=
by sorry

end NUMINAMATH_CALUDE_committee_selection_probability_l3996_399687


namespace NUMINAMATH_CALUDE_alfred_maize_storage_l3996_399601

/-- Proves that Alfred stores 1 tonne of maize per month given the conditions -/
theorem alfred_maize_storage (x : ℝ) : 
  24 * x - 5 + 8 = 27 → x = 1 := by
  sorry

end NUMINAMATH_CALUDE_alfred_maize_storage_l3996_399601


namespace NUMINAMATH_CALUDE_opposite_of_negative_two_cubed_l3996_399679

theorem opposite_of_negative_two_cubed : -((-2)^3) = 8 := by
  sorry

end NUMINAMATH_CALUDE_opposite_of_negative_two_cubed_l3996_399679


namespace NUMINAMATH_CALUDE_arithmetic_mean_neg6_to_8_l3996_399654

def arithmetic_mean (a b : Int) : ℚ :=
  let n := b - a + 1
  let sum := (n * (a + b)) / 2
  sum / n

theorem arithmetic_mean_neg6_to_8 :
  arithmetic_mean (-6) 8 = 1 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_mean_neg6_to_8_l3996_399654


namespace NUMINAMATH_CALUDE_absolute_value_inequality_l3996_399621

theorem absolute_value_inequality (x y : ℝ) : 
  |y - 3*x| < 2*x ↔ x > 0 ∧ x < y ∧ y < 5*x :=
sorry

end NUMINAMATH_CALUDE_absolute_value_inequality_l3996_399621


namespace NUMINAMATH_CALUDE_rotation_volumes_equal_l3996_399604

/-- The volume obtained by rotating a region about the y-axis -/
noncomputable def rotationVolume (region : Set (ℝ × ℝ)) : ℝ := sorry

/-- The region enclosed by x^2 = 4y, x^2 = -4y, x = 4, and x = -4 -/
def region1 : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1^2 = 4*p.2 ∨ p.1^2 = -4*p.2) ∧ (p.1 = 4 ∨ p.1 = -4)}

/-- The region defined by x^2 + y^2 ≤ 16, x^2 + (y-2)^2 ≥ 4, and x^2 + (y+2)^2 ≥ 4 -/
def region2 : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 + p.2^2 ≤ 16 ∧ p.1^2 + (p.2-2)^2 ≥ 4 ∧ p.1^2 + (p.2+2)^2 ≥ 4}

/-- The theorem stating that the volumes of rotation are equal -/
theorem rotation_volumes_equal : rotationVolume region1 = rotationVolume region2 := by
  sorry

end NUMINAMATH_CALUDE_rotation_volumes_equal_l3996_399604


namespace NUMINAMATH_CALUDE_simplify_expression_l3996_399695

theorem simplify_expression : (((3 + 4 + 6 + 7) / 3) + ((3 * 6 + 9) / 4)) = 161 / 12 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l3996_399695


namespace NUMINAMATH_CALUDE_max_value_theorem_l3996_399634

theorem max_value_theorem (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + y = 2) :
  x^2 + y^2 + 4 * Real.sqrt (x * y) ≤ 6 ∧
  (x^2 + y^2 + 4 * Real.sqrt (x * y) = 6 ↔ x = 1 ∧ y = 1) :=
sorry

end NUMINAMATH_CALUDE_max_value_theorem_l3996_399634


namespace NUMINAMATH_CALUDE_number_difference_l3996_399680

theorem number_difference (x y : ℤ) (h1 : x > y) (h2 : x + y = 64) (h3 : y = 26) : x - y = 12 := by
  sorry

end NUMINAMATH_CALUDE_number_difference_l3996_399680


namespace NUMINAMATH_CALUDE_arithmetic_equalities_l3996_399618

theorem arithmetic_equalities :
  (96 * 98 * 189 = 81 * 343 * 2^6) ∧
  (12^18 = 27^6 * 16^9) ∧
  (25^28 * 0.008^19 ≠ 0.25) := by
sorry

end NUMINAMATH_CALUDE_arithmetic_equalities_l3996_399618


namespace NUMINAMATH_CALUDE_average_of_averages_l3996_399633

theorem average_of_averages (x y : ℝ) (x_positive : 0 < x) (y_positive : 0 < y) : 
  let total_sum := x * y + y * x
  let total_count := x + y
  (x * y) / x = y ∧ (y * x) / y = x → total_sum / total_count = (2 * x * y) / (x + y) := by
sorry

end NUMINAMATH_CALUDE_average_of_averages_l3996_399633


namespace NUMINAMATH_CALUDE_punch_bowl_capacity_l3996_399665

/-- Proves that the total capacity of a punch bowl is 72 cups given the specified conditions -/
theorem punch_bowl_capacity 
  (lemonade : ℕ) 
  (cranberry : ℕ) 
  (h1 : lemonade * 5 = cranberry * 3) 
  (h2 : cranberry = lemonade + 18) : 
  lemonade + cranberry = 72 := by
  sorry

#check punch_bowl_capacity

end NUMINAMATH_CALUDE_punch_bowl_capacity_l3996_399665


namespace NUMINAMATH_CALUDE_shortest_chord_through_focus_of_ellipse_l3996_399663

/-- 
Given an ellipse with equation x²/16 + y²/9 = 1, 
prove that the length of the shortest chord passing through a focus is 9/2.
-/
theorem shortest_chord_through_focus_of_ellipse :
  let ellipse := fun (x y : ℝ) => x^2/16 + y^2/9 = 1
  ∃ (f : ℝ × ℝ), 
    (ellipse f.1 f.2) ∧ 
    (∀ (p q : ℝ × ℝ), ellipse p.1 p.2 ∧ ellipse q.1 q.2 ∧ 
      (∃ (t : ℝ), (1 - t) • f + t • p = q) →
      Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2) ≥ 9/2) ∧
    (∃ (p q : ℝ × ℝ), ellipse p.1 p.2 ∧ ellipse q.1 q.2 ∧
      (∃ (t : ℝ), (1 - t) • f + t • p = q) ∧
      Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2) = 9/2) :=
by sorry

end NUMINAMATH_CALUDE_shortest_chord_through_focus_of_ellipse_l3996_399663


namespace NUMINAMATH_CALUDE_saber_toothed_frog_tails_l3996_399678

/-- Represents the number of tadpoles of each type -/
structure TadpoleCount where
  triassic : ℕ
  saber : ℕ

/-- Represents the characteristics of each tadpole type -/
structure TadpoleType where
  legs : ℕ
  tails : ℕ

/-- The main theorem to prove -/
theorem saber_toothed_frog_tails 
  (triassic : TadpoleType)
  (saber : TadpoleType)
  (count : TadpoleCount)
  (h1 : triassic.legs = 5)
  (h2 : triassic.tails = 1)
  (h3 : saber.legs = 4)
  (h4 : count.triassic * triassic.legs + count.saber * saber.legs = 100)
  (h5 : count.triassic * triassic.tails + count.saber * saber.tails = 64) :
  saber.tails = 3 := by
  sorry

end NUMINAMATH_CALUDE_saber_toothed_frog_tails_l3996_399678


namespace NUMINAMATH_CALUDE_cube_plus_reciprocal_cube_l3996_399639

theorem cube_plus_reciprocal_cube (r : ℝ) (h : (r + 1/r)^2 = 5) :
  r^3 + 1/r^3 = 2 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_cube_plus_reciprocal_cube_l3996_399639


namespace NUMINAMATH_CALUDE_logarithm_equation_a_l3996_399622

theorem logarithm_equation_a (x : ℝ) :
  1 - Real.log 5 = (1 / 3) * (Real.log (1 / 2) + Real.log x + (1 / 3) * Real.log 5) →
  x = 16 / Real.rpow 5 (1 / 3) :=
by sorry

end NUMINAMATH_CALUDE_logarithm_equation_a_l3996_399622


namespace NUMINAMATH_CALUDE_simplify_fraction_l3996_399670

theorem simplify_fraction (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) :
  (10 * x * y^2) / (5 * x * y) = 2 * y :=
sorry

end NUMINAMATH_CALUDE_simplify_fraction_l3996_399670


namespace NUMINAMATH_CALUDE_seventh_term_of_geometric_sequence_l3996_399694

/-- Given a geometric sequence with first term a and common ratio r,
    this function returns the nth term of the sequence. -/
def geometric_term (a : ℚ) (r : ℚ) (n : ℕ) : ℚ :=
  a * r^(n - 1)

/-- Theorem stating that the 7th term of a geometric sequence
    with first term 5 and second term -1 is 1/3125 -/
theorem seventh_term_of_geometric_sequence :
  let a₁ : ℚ := 5
  let a₂ : ℚ := -1
  let r : ℚ := a₂ / a₁
  geometric_term a₁ r 7 = 1 / 3125 := by
  sorry


end NUMINAMATH_CALUDE_seventh_term_of_geometric_sequence_l3996_399694


namespace NUMINAMATH_CALUDE_board_numbers_l3996_399612

theorem board_numbers (n : ℕ) (h : n = 2014) : 
  ∀ (S : Finset ℤ), 
    S.card = n → 
    (∀ (a b c : ℤ), a ∈ S → b ∈ S → c ∈ S → (a + b + c) / 3 ∈ S) → 
    ∃ (x : ℤ), ∀ (y : ℤ), y ∈ S → y = x :=
by sorry

end NUMINAMATH_CALUDE_board_numbers_l3996_399612


namespace NUMINAMATH_CALUDE_value_of_t_l3996_399675

-- Define variables
variable (p j t : ℝ)

-- Define the conditions
def condition1 : Prop := j = 0.75 * p
def condition2 : Prop := j = 0.80 * t
def condition3 : Prop := t = p * (1 - t / 100)

-- Theorem statement
theorem value_of_t (h1 : condition1 p j) (h2 : condition2 j t) (h3 : condition3 p t) : t = 6.25 := by
  sorry

end NUMINAMATH_CALUDE_value_of_t_l3996_399675


namespace NUMINAMATH_CALUDE_decreasing_prop_function_k_range_l3996_399613

/-- A proportional function y = (k-3)x where y decreases as x increases -/
def decreasing_prop_function (k : ℝ) (x y : ℝ → ℝ) : Prop :=
  ∀ x₁ x₂, x₁ < x₂ → y x₁ > y x₂ ∧ y = λ t => (k - 3) * x t

/-- Theorem: If y decreases as x increases in the function y = (k-3)x, then k < 3 -/
theorem decreasing_prop_function_k_range (k : ℝ) (x y : ℝ → ℝ) :
  decreasing_prop_function k x y → k < 3 := by
  sorry


end NUMINAMATH_CALUDE_decreasing_prop_function_k_range_l3996_399613


namespace NUMINAMATH_CALUDE_track_circumference_l3996_399632

/-- Represents the circular track and the runners' movement --/
structure TrackSystem where
  circumference : ℝ
  speed_a : ℝ
  speed_b : ℝ

/-- The conditions of the problem --/
def satisfies_conditions (s : TrackSystem) : Prop :=
  s.speed_a > 0 ∧ s.speed_b > 0 ∧
  s.speed_a ≠ s.speed_b ∧
  (s.circumference / 2) / s.speed_b = 150 / s.speed_a ∧
  (s.circumference - 90) / s.speed_a = (s.circumference / 2 + 90) / s.speed_b

/-- The theorem to be proved --/
theorem track_circumference (s : TrackSystem) :
  satisfies_conditions s → s.circumference = 540 := by
  sorry

end NUMINAMATH_CALUDE_track_circumference_l3996_399632


namespace NUMINAMATH_CALUDE_three_number_sum_l3996_399626

theorem three_number_sum : ∀ a b c : ℝ, 
  a ≤ b → b ≤ c → 
  b = 10 → 
  (a + b + c) / 3 = a + 20 → 
  (a + b + c) / 3 = c - 30 → 
  a + b + c = 60 := by
  sorry

end NUMINAMATH_CALUDE_three_number_sum_l3996_399626


namespace NUMINAMATH_CALUDE_inscribed_hexagon_area_l3996_399607

/-- The area of a regular hexagon inscribed in a circle -/
theorem inscribed_hexagon_area (circle_area : ℝ) (h : circle_area = 100 * Real.pi) :
  let r := (circle_area / Real.pi).sqrt
  let hexagon_area := 6 * (r^2 * Real.sqrt 3 / 4)
  hexagon_area = 150 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_hexagon_area_l3996_399607


namespace NUMINAMATH_CALUDE_sequence_problem_l3996_399657

theorem sequence_problem (n : ℕ+) (b : ℕ → ℝ)
  (h0 : b 0 = 40)
  (h1 : b 1 = 70)
  (hn : b n = 0)
  (hk : ∀ k : ℕ, 1 ≤ k → k < n → b (k + 1) = b (k - 1) - 2 / b k) :
  n = 1401 := by sorry

end NUMINAMATH_CALUDE_sequence_problem_l3996_399657


namespace NUMINAMATH_CALUDE_stirling_number_second_kind_formula_l3996_399647

def stirling_number_second_kind (n r : ℕ) : ℚ :=
  (1 / r.factorial) *
    (Finset.sum (Finset.range (r + 1)) (fun k => 
      ((-1 : ℚ) ^ k * (r.choose k) * ((r - k) ^ n))))

theorem stirling_number_second_kind_formula (n r : ℕ) (h : n ≥ r) (hr : r > 0) :
  stirling_number_second_kind n r =
    (1 / r.factorial : ℚ) *
      (Finset.sum (Finset.range (r + 1)) (fun k => 
        ((-1 : ℚ) ^ k * (r.choose k) * ((r - k) ^ n)))) :=
by sorry

end NUMINAMATH_CALUDE_stirling_number_second_kind_formula_l3996_399647


namespace NUMINAMATH_CALUDE_triangular_floor_area_l3996_399696

theorem triangular_floor_area (length_feet : ℝ) (width_feet : ℝ) (feet_per_yard : ℝ) : 
  length_feet = 15 → width_feet = 12 → feet_per_yard = 3 →
  (1 / 2) * (length_feet / feet_per_yard) * (width_feet / feet_per_yard) = 10 := by
sorry

end NUMINAMATH_CALUDE_triangular_floor_area_l3996_399696


namespace NUMINAMATH_CALUDE_negation_of_quadratic_equation_l3996_399685

theorem negation_of_quadratic_equation :
  (¬ ∀ x : ℝ, x^2 + 2*x - 1 = 0) ↔ (∃ x : ℝ, x^2 + 2*x - 1 ≠ 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_quadratic_equation_l3996_399685


namespace NUMINAMATH_CALUDE_circle_y_axis_intersection_sum_l3996_399649

theorem circle_y_axis_intersection_sum :
  ∀ (y₁ y₂ : ℝ),
  (0 + 3)^2 + (y₁ - 5)^2 = 8^2 →
  (0 + 3)^2 + (y₂ - 5)^2 = 8^2 →
  y₁ ≠ y₂ →
  y₁ + y₂ = 10 := by
sorry

end NUMINAMATH_CALUDE_circle_y_axis_intersection_sum_l3996_399649


namespace NUMINAMATH_CALUDE_max_value_of_vector_sum_l3996_399686

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]

/-- Given unit vectors a and b satisfying |3a + 4b| = |4a - 3b|, and a vector c with |c| = 2,
    the maximum value of |a + b - c| is √2 + 2. -/
theorem max_value_of_vector_sum (a b c : E) 
    (ha : ‖a‖ = 1) 
    (hb : ‖b‖ = 1) 
    (hab : ‖3 • a + 4 • b‖ = ‖4 • a - 3 • b‖) 
    (hc : ‖c‖ = 2) : 
  (‖a + b - c‖ : ℝ) ≤ Real.sqrt 2 + 2 := by
  sorry

end NUMINAMATH_CALUDE_max_value_of_vector_sum_l3996_399686


namespace NUMINAMATH_CALUDE_quadratic_function_properties_l3996_399609

-- Define the quadratic function f(x) = ax^2 + bx + c
def f (a b c x : ℝ) : ℝ := a * x^2 + b * x + c

-- Define the function g(x) = ax^2 - bx + c
def g (a b c x : ℝ) : ℝ := a * x^2 - b * x + c

theorem quadratic_function_properties
  (a b c : ℝ)
  (ha : a ≠ 0)
  (hf : ∀ x, f a b c x ≠ x) :
  (∀ x, f a b c (f a b c x) ≠ x) ∧
  (a > 0 → ∀ x, f a b c (f a b c x) > x) ∧
  (a + b + c = 0 → ∀ x, f a b c (f a b c x) < x) ∧
  (∀ x, g a b c x ≠ -x) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_function_properties_l3996_399609


namespace NUMINAMATH_CALUDE_tangent_to_parabola_l3996_399688

-- Define the parabola
def parabola (x : ℝ) : ℝ := 2 * x^2

-- Define the given line
def given_line (x y : ℝ) : Prop := 4 * x - y + 3 = 0

-- Define the tangent line
def tangent_line (x y : ℝ) : Prop := 4 * x - y - 2 = 0

theorem tangent_to_parabola :
  ∃ (x₀ y₀ : ℝ),
    -- The point (x₀, y₀) is on the parabola
    y₀ = parabola x₀ ∧
    -- The tangent line passes through (x₀, y₀)
    tangent_line x₀ y₀ ∧
    -- The tangent line is parallel to the given line
    (∀ (x y : ℝ), tangent_line x y ↔ ∃ (k : ℝ), y = 4 * x + k) ∧
    -- The tangent line touches the parabola at exactly one point
    (∀ (x y : ℝ), x ≠ x₀ → y = parabola x → ¬ tangent_line x y) :=
sorry

end NUMINAMATH_CALUDE_tangent_to_parabola_l3996_399688


namespace NUMINAMATH_CALUDE_max_plus_min_equals_two_l3996_399625

noncomputable def f (x : ℝ) : ℝ := (2^x + 1)^2 / (2^x * x) + 1

def interval := {x : ℝ | (x ∈ Set.Icc (-2018) 0 ∧ x ≠ 0) ∨ (x ∈ Set.Ioc 0 2018)}

theorem max_plus_min_equals_two :
  ∃ (M N : ℝ), (∀ x ∈ interval, f x ≤ M) ∧
               (∀ x ∈ interval, N ≤ f x) ∧
               (∃ x₁ ∈ interval, f x₁ = M) ∧
               (∃ x₂ ∈ interval, f x₂ = N) ∧
               M + N = 2 := by
  sorry

end NUMINAMATH_CALUDE_max_plus_min_equals_two_l3996_399625


namespace NUMINAMATH_CALUDE_arnold_jellybean_count_l3996_399677

/-- Given the following conditions about jellybean counts:
  - Tino has 24 more jellybeans than Lee
  - Arnold has half as many jellybeans as Lee
  - Tino has 34 jellybeans
Prove that Arnold has 5 jellybeans. -/
theorem arnold_jellybean_count (tino lee arnold : ℕ) : 
  tino = lee + 24 →
  arnold = lee / 2 →
  tino = 34 →
  arnold = 5 := by
  sorry

end NUMINAMATH_CALUDE_arnold_jellybean_count_l3996_399677


namespace NUMINAMATH_CALUDE_not_right_angled_triangle_l3996_399698

theorem not_right_angled_triangle : ∃! (a b c : ℝ), 
  ((a = 3 ∧ b = 4 ∧ c = 5) ∨
   (a = 6 ∧ b = 8 ∧ c = 10) ∨
   (a = 2 ∧ b = 3 ∧ c = 3) ∨
   (a = 1 ∧ b = 1 ∧ c = Real.sqrt 2)) ∧
  (a^2 + b^2 ≠ c^2) := by
sorry

end NUMINAMATH_CALUDE_not_right_angled_triangle_l3996_399698


namespace NUMINAMATH_CALUDE_other_solution_quadratic_l3996_399630

theorem other_solution_quadratic (h : 48 * (3/4)^2 + 31 = 74 * (3/4) - 16) :
  48 * (11/12)^2 + 31 = 74 * (11/12) - 16 := by
  sorry

end NUMINAMATH_CALUDE_other_solution_quadratic_l3996_399630


namespace NUMINAMATH_CALUDE_complex_number_in_fourth_quadrant_l3996_399693

theorem complex_number_in_fourth_quadrant :
  let z : ℂ := (2 - 7*I) / (4 - I)
  (z.re > 0) ∧ (z.im < 0) :=
by sorry

end NUMINAMATH_CALUDE_complex_number_in_fourth_quadrant_l3996_399693


namespace NUMINAMATH_CALUDE_consecutive_even_numbers_sum_l3996_399644

theorem consecutive_even_numbers_sum (a b c d : ℤ) : 
  (∀ n : ℤ, a = 2*n ∧ b = 2*n + 2 ∧ c = 2*n + 4 ∧ d = 2*n + 6) →  -- Consecutive even numbers
  (a + b + c + d = 140) →                                        -- Sum condition
  (d = 38) :=                                                    -- Conclusion (largest number)
by
  sorry

end NUMINAMATH_CALUDE_consecutive_even_numbers_sum_l3996_399644


namespace NUMINAMATH_CALUDE_final_value_exceeds_initial_l3996_399655

theorem final_value_exceeds_initial (p q r M : ℝ) 
  (hp : p > 0) (hq : 0 < q ∧ q < 100) (hr : 0 < r ∧ r < 100) (hM : M > 0) :
  M * (1 + p / 100) * (1 - q / 100) * (1 + r / 100) > M ↔ 
  p > (100 * (q - r + q * r / 100)) / (100 - q + r + q * r / 100) :=
by sorry

end NUMINAMATH_CALUDE_final_value_exceeds_initial_l3996_399655


namespace NUMINAMATH_CALUDE_complex_number_modulus_l3996_399627

theorem complex_number_modulus (z w : ℂ) 
  (h1 : Complex.abs (3 * z - w) = 15)
  (h2 : Complex.abs (z + 3 * w) = 3)
  (h3 : Complex.abs (z - w) = 1) :
  Complex.abs z = 4 * Real.sqrt 7 := by
  sorry

end NUMINAMATH_CALUDE_complex_number_modulus_l3996_399627


namespace NUMINAMATH_CALUDE_birds_flying_away_l3996_399682

theorem birds_flying_away (total : ℕ) (remaining : ℕ) : 
  total = 60 → remaining = 8 → 
  ∃ (F : ℚ), F = 1/3 ∧ 
  (1 - 2/3) * (1 - 2/5) * (1 - F) * total = remaining :=
by sorry

end NUMINAMATH_CALUDE_birds_flying_away_l3996_399682


namespace NUMINAMATH_CALUDE_problem_part1_problem_part2_l3996_399643

/-- Custom multiplication operation -/
def customMult (a b : ℤ) : ℤ := a^2 - b + a * b

/-- Theorem for the first part of the problem -/
theorem problem_part1 : customMult 2 (-5) = -1 := by sorry

/-- Theorem for the second part of the problem -/
theorem problem_part2 : customMult (-2) (customMult 2 (-3)) = 1 := by sorry

end NUMINAMATH_CALUDE_problem_part1_problem_part2_l3996_399643


namespace NUMINAMATH_CALUDE_two_by_two_table_sum_l3996_399650

theorem two_by_two_table_sum (a b c d : ℝ) : 
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d →
  a + b = c + d →
  a * c = b * d →
  a + b + c + d = 0 := by
sorry

end NUMINAMATH_CALUDE_two_by_two_table_sum_l3996_399650


namespace NUMINAMATH_CALUDE_rectangle_area_at_stage_4_l3996_399620

/-- Represents the stage number of the rectangle formation process -/
def Stage : ℕ := 4

/-- The side length of each square added at each stage -/
def SquareSideLength : ℝ := 5

/-- The area of the rectangle at a given stage -/
def RectangleArea (stage : ℕ) : ℝ :=
  (stage : ℝ) * SquareSideLength * SquareSideLength

/-- Theorem stating that the area of the rectangle at Stage 4 is 100 square inches -/
theorem rectangle_area_at_stage_4 : RectangleArea Stage = 100 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_at_stage_4_l3996_399620


namespace NUMINAMATH_CALUDE_medal_award_ways_l3996_399676

def total_sprinters : ℕ := 8
def american_sprinters : ℕ := 3
def medals : ℕ := 3

def ways_to_award_medals (total : ℕ) (americans : ℕ) (medals : ℕ) : ℕ :=
  -- Number of ways to award medals with at most one American getting a medal
  sorry

theorem medal_award_ways :
  ways_to_award_medals total_sprinters american_sprinters medals = 240 :=
sorry

end NUMINAMATH_CALUDE_medal_award_ways_l3996_399676


namespace NUMINAMATH_CALUDE_geometric_progression_values_l3996_399628

theorem geometric_progression_values (p : ℝ) : 
  (∃ r : ℝ, r ≠ 0 ∧ (2*p - 1) = |p - 8| * r ∧ (4*p + 5) = (2*p - 1) * r) ↔ 
  (p = -1 ∨ p = 39/8) := by
sorry

end NUMINAMATH_CALUDE_geometric_progression_values_l3996_399628


namespace NUMINAMATH_CALUDE_power_of_two_plus_one_square_l3996_399616

theorem power_of_two_plus_one_square (m n : ℕ+) :
  2^(m : ℕ) + 1 = (n : ℕ)^2 ↔ m = 3 ∧ n = 3 := by
  sorry

end NUMINAMATH_CALUDE_power_of_two_plus_one_square_l3996_399616


namespace NUMINAMATH_CALUDE_basketball_shots_l3996_399699

theorem basketball_shots (total_points : ℕ) (total_shots : ℕ) 
  (h_points : total_points = 26) (h_shots : total_shots = 11) :
  ∃ (three_pointers two_pointers : ℕ),
    three_pointers + two_pointers = total_shots ∧
    3 * three_pointers + 2 * two_pointers = total_points ∧
    three_pointers = 4 := by
  sorry

end NUMINAMATH_CALUDE_basketball_shots_l3996_399699


namespace NUMINAMATH_CALUDE_todds_initial_gum_l3996_399602

theorem todds_initial_gum (initial : ℕ) (h : initial + 16 = 54) : initial = 38 := by
  sorry

end NUMINAMATH_CALUDE_todds_initial_gum_l3996_399602


namespace NUMINAMATH_CALUDE_expression_simplification_l3996_399636

theorem expression_simplification (a b : ℝ) :
  (3 * a^5 * b^3 + a^4 * b^2) / ((-a^2 * b)^2) - (2 + a) * (2 - a) - a * (a - 5 * b) = 8 * a * b - 3 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l3996_399636


namespace NUMINAMATH_CALUDE_melted_cubes_edge_l3996_399623

def cube_volume (edge : ℝ) : ℝ := edge ^ 3

def new_cube_edge (a b c : ℝ) : ℝ :=
  (cube_volume a + cube_volume b + cube_volume c) ^ (1/3)

theorem melted_cubes_edge : new_cube_edge 3 4 5 = 6 := by
  sorry

end NUMINAMATH_CALUDE_melted_cubes_edge_l3996_399623


namespace NUMINAMATH_CALUDE_chess_tournament_players_l3996_399640

/-- Represents the possible total scores recorded by the scorers -/
def possible_scores : List ℕ := [1979, 1980, 1984, 1985]

/-- Calculates the total number of games in a tournament with n players -/
def total_games (n : ℕ) : ℕ := n * (n - 1) / 2

/-- Theorem: The number of players in the chess tournament is 45 -/
theorem chess_tournament_players : ∃ (n : ℕ), n = 45 ∧ 
  ∃ (score : ℕ), score ∈ possible_scores ∧ score = 2 * total_games n := by
  sorry


end NUMINAMATH_CALUDE_chess_tournament_players_l3996_399640


namespace NUMINAMATH_CALUDE_bus_stop_walk_time_l3996_399617

theorem bus_stop_walk_time (usual_speed : ℝ) (usual_time : ℝ) 
  (h : usual_speed > 0) 
  (h1 : usual_time > 0)
  (h2 : (4/5 * usual_speed) * (usual_time + 6) = usual_speed * usual_time) : 
  usual_time = 30 := by
sorry

end NUMINAMATH_CALUDE_bus_stop_walk_time_l3996_399617


namespace NUMINAMATH_CALUDE_polygon_sides_l3996_399610

theorem polygon_sides (n : ℕ) (sum_interior_angles : ℝ) : sum_interior_angles = 1440 → n = 10 := by
  sorry

end NUMINAMATH_CALUDE_polygon_sides_l3996_399610


namespace NUMINAMATH_CALUDE_minimum_questionnaires_to_mail_l3996_399648

theorem minimum_questionnaires_to_mail (response_rate : ℝ) (required_responses : ℕ) :
  response_rate = 0.62 →
  required_responses = 300 →
  ∃ n : ℕ, n ≥ (required_responses : ℝ) / response_rate ∧
    ∀ m : ℕ, m < n → (m : ℝ) * response_rate < required_responses := by
  sorry

#check minimum_questionnaires_to_mail

end NUMINAMATH_CALUDE_minimum_questionnaires_to_mail_l3996_399648


namespace NUMINAMATH_CALUDE_intersection_line_l3996_399642

/-- The line of intersection of two planes -/
def line_of_intersection (t : ℝ) : ℝ × ℝ × ℝ := (t, 2 - t, t + 1)

/-- First plane equation -/
def plane1 (x y z : ℝ) : Prop := 2 * x - y - 3 * z + 5 = 0

/-- Second plane equation -/
def plane2 (x y z : ℝ) : Prop := x + y - 2 = 0

theorem intersection_line (t : ℝ) :
  let (x, y, z) := line_of_intersection t
  plane1 x y z ∧ plane2 x y z := by
  sorry

end NUMINAMATH_CALUDE_intersection_line_l3996_399642


namespace NUMINAMATH_CALUDE_apple_boxes_theorem_l3996_399684

/-- Calculates the number of boxes of apples after removing rotten ones -/
def calculate_apple_boxes (apples_per_crate : ℕ) (num_crates : ℕ) (rotten_apples : ℕ) (apples_per_box : ℕ) : ℕ :=
  ((apples_per_crate * num_crates) - rotten_apples) / apples_per_box

/-- Theorem: Given the problem conditions, the number of boxes of apples is 100 -/
theorem apple_boxes_theorem :
  calculate_apple_boxes 180 12 160 20 = 100 := by
  sorry

end NUMINAMATH_CALUDE_apple_boxes_theorem_l3996_399684


namespace NUMINAMATH_CALUDE_computer_pricing_l3996_399611

/-- Given a computer's cost and selling prices, prove the relationship between different profit percentages. -/
theorem computer_pricing (C : ℝ) : 
  (1.5 * C = 2678.57) → (1.4 * C = 2500.00) := by sorry

end NUMINAMATH_CALUDE_computer_pricing_l3996_399611


namespace NUMINAMATH_CALUDE_potatoes_for_salads_correct_l3996_399608

/-- Given the total number of potatoes, the number used for mashed potatoes,
    and the number of leftover potatoes, calculate the number of potatoes
    used for salads. -/
def potatoes_for_salads (total mashed leftover : ℕ) : ℕ :=
  total - mashed - leftover

/-- Theorem stating that the number of potatoes used for salads is correct. -/
theorem potatoes_for_salads_correct
  (total mashed leftover salads : ℕ)
  (h_total : total = 52)
  (h_mashed : mashed = 24)
  (h_leftover : leftover = 13)
  (h_salads : salads = potatoes_for_salads total mashed leftover) :
  salads = 15 := by
  sorry

end NUMINAMATH_CALUDE_potatoes_for_salads_correct_l3996_399608


namespace NUMINAMATH_CALUDE_football_cost_l3996_399666

theorem football_cost (total_spent marbles_cost baseball_cost : ℚ)
  (h1 : total_spent = 20.52)
  (h2 : marbles_cost = 9.05)
  (h3 : baseball_cost = 6.52) :
  total_spent - marbles_cost - baseball_cost = 4.95 := by
  sorry

end NUMINAMATH_CALUDE_football_cost_l3996_399666


namespace NUMINAMATH_CALUDE_complex_point_in_fourth_quadrant_l3996_399606

theorem complex_point_in_fourth_quadrant (a b : ℝ) :
  (a^2 - 4*a + 5 > 0) ∧ (-b^2 + 2*b - 6 < 0) :=
by
  sorry

#check complex_point_in_fourth_quadrant

end NUMINAMATH_CALUDE_complex_point_in_fourth_quadrant_l3996_399606


namespace NUMINAMATH_CALUDE_constant_term_binomial_expansion_l3996_399668

theorem constant_term_binomial_expansion :
  ∃ (c : ℝ), ∀ (x : ℝ),
    (λ x => (4^x - 2^(-x))^6) x = c + (λ x => (4^x - 2^(-x))^6 - c) x ∧ c = 15 := by
  sorry

end NUMINAMATH_CALUDE_constant_term_binomial_expansion_l3996_399668


namespace NUMINAMATH_CALUDE_expression_simplification_l3996_399669

theorem expression_simplification (x : ℝ) (h : x = Real.sqrt 3 - 1) :
  (1 / (x + 1) + 1 / (x^2 - 1)) / (x / (x - 1)) = Real.sqrt 3 / 3 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l3996_399669


namespace NUMINAMATH_CALUDE_positive_solution_x_l3996_399656

theorem positive_solution_x (x y z : ℝ) : 
  x > 0 →
  x * y + 3 * x + 4 * y + 10 = 30 →
  y * z + 4 * y + 2 * z + 8 = 6 →
  x * z + 4 * x + 3 * z + 12 = 30 →
  x = 3 := by
sorry

end NUMINAMATH_CALUDE_positive_solution_x_l3996_399656


namespace NUMINAMATH_CALUDE_ages_of_peter_and_grace_l3996_399689

/-- Represents a person's age -/
structure Age where
  value : ℕ

/-- Represents the ages of Peter, Jacob, and Grace -/
structure AgeGroup where
  peter : Age
  jacob : Age
  grace : Age

/-- Check if the given ages satisfy the problem conditions -/
def satisfies_conditions (ages : AgeGroup) : Prop :=
  (ages.peter.value - 10 = (ages.jacob.value - 10) / 3) ∧
  (ages.jacob.value = ages.peter.value + 12) ∧
  (ages.grace.value = (ages.peter.value + ages.jacob.value) / 2)

theorem ages_of_peter_and_grace (ages : AgeGroup) 
  (h : satisfies_conditions ages) : 
  ages.peter.value = 16 ∧ ages.grace.value = 22 := by
  sorry

#check ages_of_peter_and_grace

end NUMINAMATH_CALUDE_ages_of_peter_and_grace_l3996_399689


namespace NUMINAMATH_CALUDE_intersection_nonempty_l3996_399624

def M (k : ℝ) : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.2 - 1 = k * (p.1 + 1)}

def N : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1^2 + p.2^2 - 2*p.2 = 0}

theorem intersection_nonempty (k : ℝ) : ∃ p : ℝ × ℝ, p ∈ M k ∩ N := by
  sorry

end NUMINAMATH_CALUDE_intersection_nonempty_l3996_399624


namespace NUMINAMATH_CALUDE_divisibility_property_l3996_399629

theorem divisibility_property (n : ℕ) : ∃ k : ℤ, 1 + ⌊(3 + Real.sqrt 5)^n⌋ = k * 2^n := by
  sorry

end NUMINAMATH_CALUDE_divisibility_property_l3996_399629


namespace NUMINAMATH_CALUDE_marble_remainder_l3996_399662

theorem marble_remainder (r p : ℤ) 
  (hr : r % 5 = 2) 
  (hp : p % 5 = 4) : 
  (r + p) % 5 = 1 := by
sorry

end NUMINAMATH_CALUDE_marble_remainder_l3996_399662


namespace NUMINAMATH_CALUDE_impossible_to_reach_target_l3996_399646

/-- Represents a 3x3 grid of integers -/
def Grid := Fin 3 → Fin 3 → ℕ

/-- The initial grid state with all zeros -/
def initial_grid : Grid := fun _ _ => 0

/-- Represents a 2x2 subgrid position in the 3x3 grid -/
inductive SubgridPos
| TopLeft
| TopRight
| BottomLeft
| BottomRight

/-- Applies a 2x2 increment operation to the grid at the specified position -/
def apply_operation (g : Grid) (pos : SubgridPos) : Grid :=
  fun i j =>
    match pos with
    | SubgridPos.TopLeft => if i < 2 && j < 2 then g i j + 1 else g i j
    | SubgridPos.TopRight => if i < 2 && j > 0 then g i j + 1 else g i j
    | SubgridPos.BottomLeft => if i > 0 && j < 2 then g i j + 1 else g i j
    | SubgridPos.BottomRight => if i > 0 && j > 0 then g i j + 1 else g i j

/-- The target grid state we want to prove is impossible to reach -/
def target_grid : Grid :=
  fun i j => if i = 1 && j = 1 then 4 else 1

/-- Theorem stating that it's impossible to reach the target grid from the initial grid
    using any sequence of 2x2 increment operations -/
theorem impossible_to_reach_target :
  ∀ (ops : List SubgridPos),
    (ops.foldl apply_operation initial_grid) ≠ target_grid :=
sorry

end NUMINAMATH_CALUDE_impossible_to_reach_target_l3996_399646


namespace NUMINAMATH_CALUDE_sum_lent_is_1000_l3996_399691

/-- Proves that given the conditions of the problem, the sum lent is 1000 -/
theorem sum_lent_is_1000 (P : ℝ) (I : ℝ) : 
  (I = P * 5 * 5 / 100) →  -- Simple interest formula for 5% over 5 years
  (I = P - 750) →          -- Interest is 750 less than principal
  P = 1000 := by 
sorry

end NUMINAMATH_CALUDE_sum_lent_is_1000_l3996_399691


namespace NUMINAMATH_CALUDE_count_numbers_with_seven_is_152_l3996_399681

/-- A function that checks if a natural number contains the digit 7 -/
def contains_seven (n : ℕ) : Bool :=
  sorry

/-- The count of natural numbers from 1 to 800 containing the digit 7 -/
def count_numbers_with_seven : ℕ :=
  (List.range 800).filter (λ n => contains_seven (n + 1)) |>.length

/-- Theorem stating that the count of numbers with seven is 152 -/
theorem count_numbers_with_seven_is_152 :
  count_numbers_with_seven = 152 :=
by sorry

end NUMINAMATH_CALUDE_count_numbers_with_seven_is_152_l3996_399681


namespace NUMINAMATH_CALUDE_total_athletes_count_l3996_399638

/-- Represents the number of athletes in the sports meeting -/
structure AthleteCount where
  male : ℕ
  female : ℕ

/-- The ratio of male to female athletes at different stages -/
def initial_ratio : Rat := 19 / 12
def after_gymnastics_ratio : Rat := 20 / 13
def final_ratio : Rat := 30 / 19

/-- The difference between added male chess players and female gymnasts -/
def extra_male_players : ℕ := 30

theorem total_athletes_count (initial : AthleteCount) 
  (h1 : initial.male / initial.female = initial_ratio)
  (h2 : (initial.male / (initial.female + extra_male_players)) = after_gymnastics_ratio)
  (h3 : ((initial.male + extra_male_players) / (initial.female + extra_male_players)) = final_ratio)
  : initial.male + initial.female + 2 * extra_male_players = 6370 := by
  sorry

#check total_athletes_count

end NUMINAMATH_CALUDE_total_athletes_count_l3996_399638


namespace NUMINAMATH_CALUDE_correct_propositions_count_l3996_399645

-- Define the types of events
inductive EventType
  | Certain
  | Impossible
  | Random

-- Define the propositions
def proposition1 : EventType := EventType.Certain
def proposition2 : EventType := EventType.Impossible
def proposition3 : EventType := EventType.Certain
def proposition4 : EventType := EventType.Random

-- Define a function to check if a proposition is correct
def is_correct (prop : EventType) : Bool :=
  match prop with
  | EventType.Certain => true
  | EventType.Impossible => true
  | EventType.Random => true

-- Theorem: The number of correct propositions is 3
theorem correct_propositions_count :
  (is_correct proposition1).toNat +
  (is_correct proposition2).toNat +
  (is_correct proposition3).toNat +
  (is_correct proposition4).toNat = 3 := by
  sorry


end NUMINAMATH_CALUDE_correct_propositions_count_l3996_399645


namespace NUMINAMATH_CALUDE_floor_inequality_l3996_399600

theorem floor_inequality (x : ℝ) : 
  ⌊5*x⌋ ≥ ⌊x⌋ + ⌊2*x⌋/2 + ⌊3*x⌋/3 + ⌊4*x⌋/4 + ⌊5*x⌋/5 := by
  sorry

end NUMINAMATH_CALUDE_floor_inequality_l3996_399600


namespace NUMINAMATH_CALUDE_factorization_3m_squared_minus_12m_l3996_399615

theorem factorization_3m_squared_minus_12m (m : ℝ) : 3 * m^2 - 12 * m = 3 * m * (m - 4) := by
  sorry

end NUMINAMATH_CALUDE_factorization_3m_squared_minus_12m_l3996_399615


namespace NUMINAMATH_CALUDE_negation_equivalence_l3996_399659

theorem negation_equivalence :
  (¬ ∃ x : ℝ, x > 1) ↔ (∀ x : ℝ, x ≤ 1) := by sorry

end NUMINAMATH_CALUDE_negation_equivalence_l3996_399659


namespace NUMINAMATH_CALUDE_quadratic_equation_has_real_root_l3996_399673

theorem quadratic_equation_has_real_root (a b : ℝ) : 
  ∃ x : ℝ, x^2 + a*x + b = 0 := by
sorry

end NUMINAMATH_CALUDE_quadratic_equation_has_real_root_l3996_399673


namespace NUMINAMATH_CALUDE_parallel_line_through_point_l3996_399605

/-- Two lines are parallel if they do not intersect -/
def Parallel (l1 l2 : Set Point) : Prop := l1 ∩ l2 = ∅

/-- A point lies on a line if it is a member of the line's point set -/
def PointOnLine (p : Point) (l : Set Point) : Prop := p ∈ l

theorem parallel_line_through_point 
  (l l₁ : Set Point) (M : Point) 
  (h_parallel : Parallel l l₁)
  (h_M_not_on_l : ¬ PointOnLine M l)
  (h_M_not_on_l₁ : ¬ PointOnLine M l₁) :
  ∃ l₂ : Set Point, Parallel l₂ l ∧ Parallel l₂ l₁ ∧ PointOnLine M l₂ :=
sorry

end NUMINAMATH_CALUDE_parallel_line_through_point_l3996_399605


namespace NUMINAMATH_CALUDE_cos_42_cos_18_minus_cos_48_sin_18_l3996_399664

theorem cos_42_cos_18_minus_cos_48_sin_18 :
  Real.cos (42 * π / 180) * Real.cos (18 * π / 180) - 
  Real.cos (48 * π / 180) * Real.sin (18 * π / 180) = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_cos_42_cos_18_minus_cos_48_sin_18_l3996_399664


namespace NUMINAMATH_CALUDE_line_passes_through_point_l3996_399661

theorem line_passes_through_point :
  ∀ (k : ℝ), (1 + 4 * k^2) * 2 - (2 - 3 * k) * 2 + (2 - 14 * k) = 0 := by
  sorry

end NUMINAMATH_CALUDE_line_passes_through_point_l3996_399661


namespace NUMINAMATH_CALUDE_dogwood_trees_after_planting_l3996_399653

/-- The number of dogwood trees in the park after planting is equal to the sum of 
    the initial number of trees and the number of trees planted. -/
theorem dogwood_trees_after_planting (initial_trees planted_trees : ℕ) :
  initial_trees = 34 → planted_trees = 49 → initial_trees + planted_trees = 83 := by
  sorry

end NUMINAMATH_CALUDE_dogwood_trees_after_planting_l3996_399653


namespace NUMINAMATH_CALUDE_twentyseven_eighths_two_thirds_power_l3996_399667

theorem twentyseven_eighths_two_thirds_power :
  (27 / 8 : ℝ) ^ (2 / 3) = 9 / 4 := by
  sorry

end NUMINAMATH_CALUDE_twentyseven_eighths_two_thirds_power_l3996_399667


namespace NUMINAMATH_CALUDE_friendship_theorem_l3996_399690

/-- A simple graph with 17 vertices where each vertex has degree 4 -/
structure FriendshipGraph where
  vertices : Finset (Fin 17)
  edges : Finset (Fin 17 × Fin 17)
  edge_symmetric : ∀ a b, (a, b) ∈ edges → (b, a) ∈ edges
  no_self_loops : ∀ a, (a, a) ∉ edges
  degree_four : ∀ v, (edges.filter (λ e => e.1 = v ∨ e.2 = v)).card = 4

/-- Two vertices are acquainted if there's an edge between them -/
def acquainted (G : FriendshipGraph) (a b : Fin 17) : Prop :=
  (a, b) ∈ G.edges

/-- Two vertices share a common neighbor if there exists a vertex connected to both -/
def share_neighbor (G : FriendshipGraph) (a b : Fin 17) : Prop :=
  ∃ c, acquainted G a c ∧ acquainted G b c

/-- Main theorem: There exist two vertices that are not acquainted and do not share a neighbor -/
theorem friendship_theorem (G : FriendshipGraph) : 
  ∃ a b, a ≠ b ∧ ¬(acquainted G a b) ∧ ¬(share_neighbor G a b) := by
  sorry

end NUMINAMATH_CALUDE_friendship_theorem_l3996_399690


namespace NUMINAMATH_CALUDE_teachers_arrangements_count_l3996_399614

def num_students : ℕ := 5
def num_teachers : ℕ := 2

def arrangements (n_students : ℕ) (n_teachers : ℕ) : ℕ :=
  (Nat.factorial n_students) * (n_students - 1) * (Nat.factorial n_teachers)

theorem teachers_arrangements_count :
  arrangements num_students num_teachers = 960 := by
  sorry

end NUMINAMATH_CALUDE_teachers_arrangements_count_l3996_399614


namespace NUMINAMATH_CALUDE_expand_product_l3996_399658

theorem expand_product (x : ℝ) (h : x ≠ 0) :
  (3 / 7) * ((7 / x^2) + 6 * x^3 - 2) = 3 / x^2 + (18 * x^3) / 7 - 6 / 7 := by
  sorry

end NUMINAMATH_CALUDE_expand_product_l3996_399658


namespace NUMINAMATH_CALUDE_race_dead_heat_l3996_399697

theorem race_dead_heat (L : ℝ) (vA vB : ℝ) (h : vA = (17 / 15) * vB) :
  let d := (2 / 17) * L
  (L / vA) = ((L - d) / vB) :=
by sorry

end NUMINAMATH_CALUDE_race_dead_heat_l3996_399697


namespace NUMINAMATH_CALUDE_bottle_caps_remaining_l3996_399674

theorem bottle_caps_remaining (initial : Nat) (removed : Nat) (h1 : initial = 16) (h2 : removed = 6) :
  initial - removed = 10 := by
  sorry

end NUMINAMATH_CALUDE_bottle_caps_remaining_l3996_399674


namespace NUMINAMATH_CALUDE_sum_of_reciprocal_cubes_l3996_399660

theorem sum_of_reciprocal_cubes (a b c : ℝ) 
  (sum_condition : a + b + c = 6)
  (prod_sum_condition : a * b + b * c + c * a = 5)
  (prod_condition : a * b * c = 1) :
  1 / a^3 + 1 / b^3 + 1 / c^3 = 128 := by
sorry

end NUMINAMATH_CALUDE_sum_of_reciprocal_cubes_l3996_399660


namespace NUMINAMATH_CALUDE_magic_square_y_value_l3996_399672

/-- Represents a 3x3 magic square -/
structure MagicSquare :=
  (a : ℕ) (b : ℕ) (c : ℕ)
  (d : ℕ) (e : ℕ) (f : ℕ)
  (g : ℕ) (h : ℕ) (i : ℕ)

/-- Checks if a given 3x3 square is a magic square -/
def is_magic_square (s : MagicSquare) : Prop :=
  let sum := s.a + s.b + s.c
  sum = s.d + s.e + s.f ∧
  sum = s.g + s.h + s.i ∧
  sum = s.a + s.d + s.g ∧
  sum = s.b + s.e + s.h ∧
  sum = s.c + s.f + s.i ∧
  sum = s.a + s.e + s.i ∧
  sum = s.c + s.e + s.g

theorem magic_square_y_value (y : ℕ) :
  ∃ (s : MagicSquare), 
    is_magic_square s ∧ 
    s.a = y ∧ s.b = 25 ∧ s.c = 70 ∧ 
    s.d = 5 → 
    y = 90 := by sorry

end NUMINAMATH_CALUDE_magic_square_y_value_l3996_399672


namespace NUMINAMATH_CALUDE_student_group_problem_first_group_size_l3996_399603

theorem student_group_problem (x : ℕ) : 
  x * x + (x + 5) * (x + 5) = 13000 → x = 78 := by sorry

theorem first_group_size (x : ℕ) :
  x * x + (x + 5) * (x + 5) = 13000 → x + 5 = 83 := by sorry

end NUMINAMATH_CALUDE_student_group_problem_first_group_size_l3996_399603


namespace NUMINAMATH_CALUDE_average_score_five_students_l3996_399692

theorem average_score_five_students
  (initial_students : Nat)
  (initial_average : ℝ)
  (fifth_student_score : ℝ)
  (h1 : initial_students = 4)
  (h2 : initial_average = 85)
  (h3 : fifth_student_score = 90) :
  (initial_students * initial_average + fifth_student_score) / (initial_students + 1) = 86 :=
by sorry

end NUMINAMATH_CALUDE_average_score_five_students_l3996_399692


namespace NUMINAMATH_CALUDE_cyclic_quadrilateral_theorem_l3996_399637

-- Define the basic structures
structure Point := (x y : ℝ)

structure Line := (a b c : ℝ)

-- Define the quadrilateral ABCD
def A : Point := sorry
def B : Point := sorry
def C : Point := sorry
def D : Point := sorry

-- Define points E and F on CD
def E : Point := sorry
def F : Point := sorry

-- Define the circumcenters G and H
def G : Point := sorry
def H : Point := sorry

-- Define the lines AB, CD, and GH
def AB : Line := sorry
def CD : Line := sorry
def GH : Line := sorry

-- Define the property of being cyclic
def is_cyclic (p q r s : Point) : Prop := sorry

-- Define the property of lines being concurrent or parallel
def lines_concurrent_or_parallel (l m n : Line) : Prop := sorry

-- Define the property of a point lying on a line
def point_on_line (p : Point) (l : Line) : Prop := sorry

-- Define the property of being a circumcenter
def is_circumcenter (p : Point) (a b c : Point) : Prop := sorry

-- Main theorem
theorem cyclic_quadrilateral_theorem :
  (∀ (X : Point), is_cyclic A B C D) →  -- ABCD is cyclic
  (¬ (AB.a * CD.b = AB.b * CD.a)) →  -- AD is not parallel to BC
  point_on_line E CD →  -- E lies on CD
  point_on_line F CD →  -- F lies on CD
  is_circumcenter G B C E →  -- G is circumcenter of BCE
  is_circumcenter H A D F →  -- H is circumcenter of ADF
  (lines_concurrent_or_parallel AB CD GH ↔ is_cyclic A B E F) := by
  sorry

end NUMINAMATH_CALUDE_cyclic_quadrilateral_theorem_l3996_399637


namespace NUMINAMATH_CALUDE_equation_solution_exists_l3996_399635

theorem equation_solution_exists : ∃ (x y z t : ℕ+), x + y + z + t = 10 ∧ z = 7 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_exists_l3996_399635


namespace NUMINAMATH_CALUDE_points_collinear_l3996_399641

/-- Three points are collinear if the slope between any two pairs of points is the same. -/
def collinear (p1 p2 p3 : ℝ × ℝ) : Prop :=
  let (x1, y1) := p1
  let (x2, y2) := p2
  let (x3, y3) := p3
  (y2 - y1) * (x3 - x2) = (y3 - y2) * (x2 - x1)

theorem points_collinear : collinear (1, 2) (3, -2) (4, -4) := by
  sorry

end NUMINAMATH_CALUDE_points_collinear_l3996_399641


namespace NUMINAMATH_CALUDE_smaller_number_in_ratio_l3996_399683

theorem smaller_number_in_ratio (x y a b c : ℝ) 
  (h_pos_x : x > 0) 
  (h_pos_y : y > 0) 
  (h_ratio : x / y = a / b) 
  (h_a_lt_b : 0 < a ∧ a < b) 
  (h_sum : x + y = c) : 
  min x y = a * c / (a + b) := by
  sorry

end NUMINAMATH_CALUDE_smaller_number_in_ratio_l3996_399683


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l3996_399651

-- Define an arithmetic sequence
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

-- State the theorem
theorem arithmetic_sequence_sum
  (a : ℕ → ℝ)
  (h_arith : arithmetic_sequence a)
  (h_a5 : a 5 = 8) :
  a 2 + a 4 + a 5 + a 9 = 32 :=
by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l3996_399651


namespace NUMINAMATH_CALUDE_legs_multiple_of_heads_l3996_399671

/-- Represents the number of legs for each animal type -/
def legs_per_animal : Fin 3 → ℕ
| 0 => 2  -- Ducks
| 1 => 4  -- Cows
| 2 => 4  -- Buffaloes

/-- Represents the number of animals of each type -/
structure AnimalCounts where
  ducks : ℕ
  cows : ℕ
  buffaloes : ℕ
  buffalo_count_eq : buffaloes = 6

/-- Calculates the total number of legs -/
def total_legs (counts : AnimalCounts) : ℕ :=
  counts.ducks * legs_per_animal 0 +
  counts.cows * legs_per_animal 1 +
  counts.buffaloes * legs_per_animal 2

/-- Calculates the total number of heads -/
def total_heads (counts : AnimalCounts) : ℕ :=
  counts.ducks + counts.cows + counts.buffaloes

/-- The theorem to be proved -/
theorem legs_multiple_of_heads (counts : AnimalCounts) :
  ∃ m : ℕ, m ≥ 2 ∧ total_legs counts = m * total_heads counts + 12 ∧
  ∀ k : ℕ, k < m → ¬(total_legs counts = k * total_heads counts + 12) :=
sorry

end NUMINAMATH_CALUDE_legs_multiple_of_heads_l3996_399671
