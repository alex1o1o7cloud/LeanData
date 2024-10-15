import Mathlib

namespace NUMINAMATH_CALUDE_max_k_value_l176_17664

noncomputable def f (x : ℝ) : ℝ := x * Real.exp x - 1

noncomputable def g (k : ℝ) (x : ℝ) : ℝ := Real.log x + k * x

theorem max_k_value :
  (∃ k : ℝ, ∀ x : ℝ, x > 0 → f x ≥ g k x) →
  (∀ k : ℝ, (∀ x : ℝ, x > 0 → f x ≥ g k x) → k ≤ 1) ∧
  (∀ x : ℝ, x > 0 → f x ≥ g 1 x) :=
sorry

end NUMINAMATH_CALUDE_max_k_value_l176_17664


namespace NUMINAMATH_CALUDE_johns_number_proof_l176_17631

theorem johns_number_proof : 
  ∃! x : ℕ, 
    10 ≤ x ∧ x < 100 ∧ 
    (∃ a b : ℕ, 
      4 * x + 17 = 10 * a + b ∧
      10 * b + a ≥ 91 ∧ 
      10 * b + a ≤ 95) ∧
    x = 8 := by
  sorry

end NUMINAMATH_CALUDE_johns_number_proof_l176_17631


namespace NUMINAMATH_CALUDE_rhombus_longer_diagonal_l176_17616

/-- A rhombus with side length 37 units and shorter diagonal 40 units has a longer diagonal of 62 units. -/
theorem rhombus_longer_diagonal (side : ℝ) (shorter_diagonal : ℝ) (longer_diagonal : ℝ) : 
  side = 37 → shorter_diagonal = 40 → longer_diagonal = 62 → 
  side^2 = (shorter_diagonal / 2)^2 + (longer_diagonal / 2)^2 :=
by sorry

end NUMINAMATH_CALUDE_rhombus_longer_diagonal_l176_17616


namespace NUMINAMATH_CALUDE_smallest_angle_for_complete_circle_l176_17625

theorem smallest_angle_for_complete_circle : 
  ∃ (t : ℝ), t > 0 ∧ 
  (∀ (θ : ℝ), 0 ≤ θ ∧ θ ≤ t → ∃ (r : ℝ), r = Real.sin θ) ∧
  (∀ (s : ℝ), s > 0 ∧ s < t → ¬(∀ (θ : ℝ), 0 ≤ θ ∧ θ ≤ s → ∃ (r : ℝ), r = Real.sin θ)) ∧
  t = π :=
sorry

end NUMINAMATH_CALUDE_smallest_angle_for_complete_circle_l176_17625


namespace NUMINAMATH_CALUDE_hyperbola_asymptotes_l176_17667

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop :=
  (x - 3)^2 / 49 - (y + 2)^2 / 36 = 1

-- Define the asymptote function
def asymptote (m c x : ℝ) (y : ℝ) : Prop :=
  y = m * x + c

-- Theorem statement
theorem hyperbola_asymptotes :
  ∃ (m₁ m₂ c : ℝ),
    m₁ = 6/7 ∧ m₂ = -6/7 ∧ c = -32/7 ∧
    (∀ (x y : ℝ), hyperbola x y →
      (asymptote m₁ c x y ∨ asymptote m₂ c x y)) :=
sorry

end NUMINAMATH_CALUDE_hyperbola_asymptotes_l176_17667


namespace NUMINAMATH_CALUDE_factor_expression_l176_17695

theorem factor_expression (x : ℝ) : 3*x*(x+1) + 7*(x+1) = (3*x+7)*(x+1) := by
  sorry

end NUMINAMATH_CALUDE_factor_expression_l176_17695


namespace NUMINAMATH_CALUDE_sum_product_equality_l176_17655

theorem sum_product_equality : 1.25 * 67.875 + 125 * 6.7875 + 1250 * 0.053375 = 1000 := by
  sorry

end NUMINAMATH_CALUDE_sum_product_equality_l176_17655


namespace NUMINAMATH_CALUDE_max_students_distribution_l176_17675

def max_students (pens pencils : ℕ) : ℕ :=
  Nat.gcd pens pencils

theorem max_students_distribution (pens pencils : ℕ) :
  pens = 100 → pencils = 50 → max_students pens pencils = 50 := by
  sorry

end NUMINAMATH_CALUDE_max_students_distribution_l176_17675


namespace NUMINAMATH_CALUDE_spoons_multiple_of_groups_l176_17607

/-- Represents the number of commemorative plates Daniel has -/
def num_plates : ℕ := 44

/-- Represents the number of groups Daniel can form -/
def num_groups : ℕ := 11

/-- Represents the number of commemorative spoons Daniel has -/
def num_spoons : ℕ := sorry

/-- Theorem stating that the number of spoons is a multiple of the number of groups -/
theorem spoons_multiple_of_groups :
  ∃ k : ℕ, num_spoons = k * num_groups :=
sorry

end NUMINAMATH_CALUDE_spoons_multiple_of_groups_l176_17607


namespace NUMINAMATH_CALUDE_min_distance_to_midpoint_l176_17673

/-- Given a line segment AB with length 4 and a point P satisfying |PA| - |PB| = 3,
    where O is the midpoint of AB, the minimum value of |OP| is 3/2. -/
theorem min_distance_to_midpoint (A B P O : EuclideanSpace ℝ (Fin 2)) :
  dist A B = 4 →
  O = midpoint ℝ A B →
  dist P A - dist P B = 3 →
  ∃ (min_dist : ℝ), min_dist = 3/2 ∧ ∀ Q, dist P A - dist Q B = 3 → dist O Q ≥ min_dist :=
by sorry

end NUMINAMATH_CALUDE_min_distance_to_midpoint_l176_17673


namespace NUMINAMATH_CALUDE_unique_number_division_multiplication_l176_17604

theorem unique_number_division_multiplication : ∃! x : ℚ, (x / 6) * 12 = 12 := by
  sorry

end NUMINAMATH_CALUDE_unique_number_division_multiplication_l176_17604


namespace NUMINAMATH_CALUDE_sally_orange_balloons_l176_17634

/-- The number of orange balloons Sally has after losing some -/
def remaining_orange_balloons (initial : ℕ) (lost : ℕ) : ℕ :=
  initial - lost

/-- Theorem stating that Sally has 7 orange balloons after losing 2 -/
theorem sally_orange_balloons :
  remaining_orange_balloons 9 2 = 7 := by
  sorry

end NUMINAMATH_CALUDE_sally_orange_balloons_l176_17634


namespace NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l176_17691

theorem arithmetic_sequence_common_difference 
  (a : ℕ → ℝ) 
  (h_arithmetic : ∀ n : ℕ, a (n + 1) - a n = a (n + 2) - a (n + 1)) 
  (h_a2 : a 2 = 3) 
  (h_a7 : a 7 = 13) : 
  ∃ d : ℝ, d = 2 ∧ ∀ n : ℕ, a (n + 1) - a n = d :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l176_17691


namespace NUMINAMATH_CALUDE_greatest_integer_fraction_inequality_l176_17643

theorem greatest_integer_fraction_inequality :
  ∀ x : ℤ, (7 : ℚ) / 9 > (x : ℚ) / 13 ↔ x ≤ 10 :=
sorry

end NUMINAMATH_CALUDE_greatest_integer_fraction_inequality_l176_17643


namespace NUMINAMATH_CALUDE_quadratic_factor_evaluation_l176_17618

theorem quadratic_factor_evaluation (b c : ℤ) : 
  let p : ℝ → ℝ := λ x => x^2 + b*x + c
  (∃ q : ℝ → ℝ, (∀ x, x^4 + 8*x^2 + 36 = p x * q x)) →
  (∃ r : ℝ → ℝ, (∀ x, 2*x^4 + 9*x^2 + 37*x + 18 = p x * r x)) →
  p (-1) = 12 := by
sorry

end NUMINAMATH_CALUDE_quadratic_factor_evaluation_l176_17618


namespace NUMINAMATH_CALUDE_reciprocal_of_negative_two_l176_17614

theorem reciprocal_of_negative_two :
  ∃ (x : ℚ), x * (-2) = 1 ∧ x = -1/2 := by sorry

end NUMINAMATH_CALUDE_reciprocal_of_negative_two_l176_17614


namespace NUMINAMATH_CALUDE_x_squared_mod_26_l176_17615

theorem x_squared_mod_26 (x : ℤ) (h1 : 6 * x ≡ 14 [ZMOD 26]) (h2 : 4 * x ≡ 20 [ZMOD 26]) :
  x^2 ≡ 12 [ZMOD 26] := by
  sorry

end NUMINAMATH_CALUDE_x_squared_mod_26_l176_17615


namespace NUMINAMATH_CALUDE_at_least_eight_empty_columns_at_least_eight_people_in_one_column_l176_17601

/-- Represents the state of people on columns -/
structure ColumnState where
  num_people : Nat
  num_columns : Nat
  initial_column : Nat

/-- Proves that at least 8 columns are empty after any number of steps -/
theorem at_least_eight_empty_columns (state : ColumnState) 
  (h1 : state.num_people = 65)
  (h2 : state.num_columns = 17)
  (h3 : state.initial_column = 9) :
  ∀ (steps : Nat), ∃ (empty_columns : Nat), empty_columns ≥ 8 := by
  sorry

/-- Proves that there is always at least one column with at least 8 people -/
theorem at_least_eight_people_in_one_column (state : ColumnState) 
  (h1 : state.num_people = 65)
  (h2 : state.num_columns = 17)
  (h3 : state.initial_column = 9) :
  ∀ (steps : Nat), ∃ (column : Nat), ∃ (people_in_column : Nat), 
    people_in_column ≥ 8 ∧ column ≤ state.num_columns := by
  sorry

end NUMINAMATH_CALUDE_at_least_eight_empty_columns_at_least_eight_people_in_one_column_l176_17601


namespace NUMINAMATH_CALUDE_emilees_earnings_l176_17612

/-- Given the earnings and work conditions of Jermaine, Terrence, and Emilee, prove Emilee's earnings. -/
theorem emilees_earnings 
  (total_earnings : ℝ)
  (j_hours r_j : ℝ)
  (t_hours r_t : ℝ)
  (e_hours r_e : ℝ)
  (h1 : total_earnings = 90)
  (h2 : r_j * j_hours = r_t * t_hours + 5)
  (h3 : r_t * t_hours = 30)
  (h4 : total_earnings = r_j * j_hours + r_t * t_hours + r_e * e_hours) :
  r_e * e_hours = 25 := by
  sorry

end NUMINAMATH_CALUDE_emilees_earnings_l176_17612


namespace NUMINAMATH_CALUDE_inequality_proof_l176_17654

theorem inequality_proof (x a : ℝ) (h : x < a ∧ a < 0) : x^3 > a*x ∧ a*x < 0 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l176_17654


namespace NUMINAMATH_CALUDE_nathan_blankets_l176_17644

-- Define the warmth provided by each blanket
def warmth_per_blanket : ℕ := 3

-- Define the total warmth provided by the blankets Nathan used
def total_warmth : ℕ := 21

-- Define the number of blankets Nathan used (half of the total)
def blankets_used : ℕ := total_warmth / warmth_per_blanket

-- Define the total number of blankets in Nathan's closet
def total_blankets : ℕ := 2 * blankets_used

-- Theorem stating that the total number of blankets is 14
theorem nathan_blankets : total_blankets = 14 := by
  sorry

end NUMINAMATH_CALUDE_nathan_blankets_l176_17644


namespace NUMINAMATH_CALUDE_cubic_roots_negative_real_parts_l176_17613

theorem cubic_roots_negative_real_parts
  (a₀ a₁ a₂ a₃ : ℝ) :
  (∀ x : ℂ, a₀ * x^3 + a₁ * x^2 + a₂ * x + a₃ = 0 → (x.re < 0)) ↔
  ((a₀ > 0 ∧ a₁ > 0 ∧ a₂ > 0 ∧ a₃ > 0) ∨ (a₀ < 0 ∧ a₁ < 0 ∧ a₂ < 0 ∧ a₃ < 0)) ∧
  a₁ * a₂ - a₀ * a₃ > 0 :=
by sorry

end NUMINAMATH_CALUDE_cubic_roots_negative_real_parts_l176_17613


namespace NUMINAMATH_CALUDE_subsets_with_three_adjacent_chairs_l176_17671

/-- The number of chairs arranged in a circle -/
def n : ℕ := 12

/-- A function that calculates the number of subsets of n chairs
    arranged in a circle that contain at least three adjacent chairs -/
def subsets_with_adjacent_chairs (n : ℕ) : ℕ :=
  sorry

/-- Theorem stating that the number of subsets of 12 chairs arranged
    in a circle that contain at least three adjacent chairs is 1634 -/
theorem subsets_with_three_adjacent_chairs :
  subsets_with_adjacent_chairs n = 1634 := by
  sorry

end NUMINAMATH_CALUDE_subsets_with_three_adjacent_chairs_l176_17671


namespace NUMINAMATH_CALUDE_no_rational_solutions_for_positive_k_l176_17637

theorem no_rational_solutions_for_positive_k : 
  ¬ ∃ (k : ℕ+) (x : ℚ), k * x^2 + 30 * x + k = 0 := by
  sorry

end NUMINAMATH_CALUDE_no_rational_solutions_for_positive_k_l176_17637


namespace NUMINAMATH_CALUDE_multiply_divide_example_l176_17617

theorem multiply_divide_example : (3.242 * 15) / 100 = 0.4863 := by
  sorry

end NUMINAMATH_CALUDE_multiply_divide_example_l176_17617


namespace NUMINAMATH_CALUDE_smallest_n_divisible_by_23_l176_17611

theorem smallest_n_divisible_by_23 :
  ∃ (n : ℕ), (n^3 + 12*n^2 + 15*n + 180) % 23 = 0 ∧
  ∀ (m : ℕ), m < n → (m^3 + 12*m^2 + 15*m + 180) % 23 ≠ 0 :=
by
  use 10
  sorry

end NUMINAMATH_CALUDE_smallest_n_divisible_by_23_l176_17611


namespace NUMINAMATH_CALUDE_target_hit_probability_l176_17666

theorem target_hit_probability (prob_A prob_B : ℝ) 
  (h_prob_A : prob_A = 0.6) 
  (h_prob_B : prob_B = 0.5) : 
  let prob_hit_atleast_once := prob_A * (1 - prob_B) + (1 - prob_A) * prob_B + prob_A * prob_B
  prob_A * (1 - prob_B) / prob_hit_atleast_once + prob_A * prob_B / prob_hit_atleast_once = 3/4 := by
  sorry

end NUMINAMATH_CALUDE_target_hit_probability_l176_17666


namespace NUMINAMATH_CALUDE_cubic_and_sixth_degree_polynomial_roots_l176_17651

theorem cubic_and_sixth_degree_polynomial_roots : ∀ s : ℂ,
  s^3 - 2*s^2 + s - 1 = 0 → s^6 - 16*s - 8 = 0 := by
  sorry

end NUMINAMATH_CALUDE_cubic_and_sixth_degree_polynomial_roots_l176_17651


namespace NUMINAMATH_CALUDE_sqrt_simplification_l176_17656

theorem sqrt_simplification :
  Real.sqrt (49 - 20 * Real.sqrt 3) = 5 - 2 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_simplification_l176_17656


namespace NUMINAMATH_CALUDE_power_function_m_value_l176_17685

theorem power_function_m_value (m : ℕ+) (f : ℝ → ℝ) : 
  (∀ x, f x = x ^ (m.val ^ 2 + m.val)) → 
  f (Real.sqrt 2) = 2 → 
  m = 1 := by sorry

end NUMINAMATH_CALUDE_power_function_m_value_l176_17685


namespace NUMINAMATH_CALUDE_limit_rational_function_l176_17688

/-- The limit of (x^3 - 3x - 2) / (x - 2) as x approaches 2 is 9 -/
theorem limit_rational_function : 
  ∀ ε > 0, ∃ δ > 0, ∀ x : ℝ, 0 < |x - 2| ∧ |x - 2| < δ → 
    |(x^3 - 3*x - 2) / (x - 2) - 9| < ε := by
  sorry

end NUMINAMATH_CALUDE_limit_rational_function_l176_17688


namespace NUMINAMATH_CALUDE_two_digit_number_50th_power_l176_17680

theorem two_digit_number_50th_power (log2 log3 log11 : ℝ) 
  (h_log2 : log2 = 0.3010)
  (h_log3 : log3 = 0.4771)
  (h_log11 : log11 = 1.0414) :
  ∃! P : ℕ, 
    10 ≤ P ∧ P < 100 ∧ 
    (10^68 : ℝ) ≤ (P^50 : ℝ) ∧ (P^50 : ℝ) < (10^69 : ℝ) ∧
    P = 23 := by
  sorry

end NUMINAMATH_CALUDE_two_digit_number_50th_power_l176_17680


namespace NUMINAMATH_CALUDE_abc_sum_reciprocal_l176_17627

theorem abc_sum_reciprocal (a b c : ℝ) (ha : a ≠ 1) (hb : b ≠ 1) (hc : c ≠ 1)
  (h1 : a * b * c = 1)
  (h2 : a^2 + b^2 + c^2 - (1/a^2 + 1/b^2 + 1/c^2) = 8*(a+b+c) - 8*(a*b+b*c+c*a)) :
  1/(a-1) + 1/(b-1) + 1/(c-1) = -3/2 := by
  sorry

end NUMINAMATH_CALUDE_abc_sum_reciprocal_l176_17627


namespace NUMINAMATH_CALUDE_equation_three_solutions_l176_17622

/-- The equation has exactly three solutions when a is 0, 5, or 9 -/
theorem equation_three_solutions (x : ℝ) (a : ℝ) : 
  (∃! (s : Finset ℝ), s.card = 3 ∧ ∀ x ∈ s, 
    (Real.sqrt (x - 1) * (|x^2 - 10*x + 16| - a)) / 
    (a*x^2 - 7*x^2 - 10*a*x + 70*x + 21*a - 147) = 0) ↔ 
  (a = 0 ∨ a = 5 ∨ a = 9) :=
sorry

end NUMINAMATH_CALUDE_equation_three_solutions_l176_17622


namespace NUMINAMATH_CALUDE_valid_arrangement_exists_l176_17682

/-- Represents a tile with a diagonal --/
inductive Tile
| TopLeftToBottomRight
| TopRightToBottomLeft

/-- Represents a position in the 6×6 grid --/
structure Position :=
  (x : Fin 6)
  (y : Fin 6)

/-- Represents an arrangement of tiles in the 6×6 grid --/
def Arrangement := Position → Option Tile

/-- Checks if two positions are adjacent --/
def adjacent (p1 p2 : Position) : Bool :=
  (p1.x = p2.x ∧ (p1.y.val + 1 = p2.y.val ∨ p2.y.val + 1 = p1.y.val)) ∨
  (p1.y = p2.y ∧ (p1.x.val + 1 = p2.x.val ∨ p2.x.val + 1 = p1.x.val))

/-- Checks if the arrangement is valid --/
def validArrangement (arr : Arrangement) : Prop :=
  (∀ p : Position, ∃ t : Tile, arr p = some t) ∧
  (∀ p1 p2 : Position, adjacent p1 p2 → arr p1 ≠ arr p2)

/-- The main theorem stating that a valid arrangement exists --/
theorem valid_arrangement_exists : ∃ arr : Arrangement, validArrangement arr :=
sorry


end NUMINAMATH_CALUDE_valid_arrangement_exists_l176_17682


namespace NUMINAMATH_CALUDE_five_fourths_of_sum_l176_17619

theorem five_fourths_of_sum : ∀ (a b c d : ℚ),
  a = 6 / 3 ∧ b = 8 / 4 → (5 / 4) * (a + b) = 5 :=
by
  sorry

end NUMINAMATH_CALUDE_five_fourths_of_sum_l176_17619


namespace NUMINAMATH_CALUDE_parabola_point_value_l176_17653

/-- Prove that for a parabola y = x^2 + (a+1)x + a passing through (-1, m), m must equal 0 -/
theorem parabola_point_value (a m : ℝ) : 
  ((-1)^2 + (a + 1)*(-1) + a = m) → m = 0 := by
  sorry

end NUMINAMATH_CALUDE_parabola_point_value_l176_17653


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_l176_17624

theorem quadratic_equation_roots (k : ℝ) : 
  (∃ x₁ x₂ : ℝ, 2 * x₁^2 - 7 * x₁ + k = 0 ∧ 2 * x₂^2 - 7 * x₂ + k = 0 ∧ x₁ = 2) →
  (∃ x₂ : ℝ, x₂ = 3/2 ∧ k = 6) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_l176_17624


namespace NUMINAMATH_CALUDE_box_length_given_cube_fill_l176_17681

/-- Represents the dimensions of a box -/
structure BoxDimensions where
  length : ℕ
  width : ℕ
  depth : ℕ

/-- Represents the properties of cubes filling the box -/
structure CubeFill where
  sideLength : ℕ
  count : ℕ

/-- Theorem stating the relationship between box dimensions and cube fill -/
theorem box_length_given_cube_fill 
  (box : BoxDimensions) 
  (cube : CubeFill) 
  (h1 : box.width = 20) 
  (h2 : box.depth = 10) 
  (h3 : cube.count = 56) 
  (h4 : box.length * box.width * box.depth = cube.count * cube.sideLength ^ 3) 
  (h5 : cube.sideLength ∣ box.width ∧ cube.sideLength ∣ box.depth) :
  box.length = 280 := by
  sorry

#check box_length_given_cube_fill

end NUMINAMATH_CALUDE_box_length_given_cube_fill_l176_17681


namespace NUMINAMATH_CALUDE_equation_solution_l176_17646

theorem equation_solution : ∃ x : ℚ, (27 / 4 : ℚ) * x - 18 = 3 * x + 27 ∧ x = 12 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l176_17646


namespace NUMINAMATH_CALUDE_plates_problem_l176_17623

theorem plates_problem (x : ℚ) : 
  (1/3 * x - 2/3) - 1/2 * ((2/3 * x) - 4/3) = 9 → x = 29 := by
  sorry

end NUMINAMATH_CALUDE_plates_problem_l176_17623


namespace NUMINAMATH_CALUDE_ellipse_k_range_l176_17694

/-- The ellipse equation -/
def ellipse (k x y : ℝ) : ℝ := k^2 * x^2 + y^2 - 4*k*x + 2*k*y + k^2 - 1

/-- Theorem stating the range of k for the given ellipse -/
theorem ellipse_k_range :
  ∀ k : ℝ, (∀ x y : ℝ, ellipse k x y = 0 → ellipse k 0 0 < 0) →
  (0 < |k| ∧ |k| < 1) :=
sorry

end NUMINAMATH_CALUDE_ellipse_k_range_l176_17694


namespace NUMINAMATH_CALUDE_min_value_of_expression_l176_17683

theorem min_value_of_expression (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : a + 2*b - 4*a*b = 0) :
  ∀ x y, x > 0 → y > 0 → x + 2*y - 4*x*y = 0 → a + 8*b ≤ x + 8*y ∧ ∃ a b, a > 0 ∧ b > 0 ∧ a + 2*b - 4*a*b = 0 ∧ a + 8*b = 9/2 :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_expression_l176_17683


namespace NUMINAMATH_CALUDE_cubic_root_coefficient_a_l176_17679

theorem cubic_root_coefficient_a (a b : ℚ) : 
  ((-1 - 4 * Real.sqrt 2)^3 + a * (-1 - 4 * Real.sqrt 2)^2 + b * (-1 - 4 * Real.sqrt 2) + 31 = 0) →
  a = 1 := by
  sorry

end NUMINAMATH_CALUDE_cubic_root_coefficient_a_l176_17679


namespace NUMINAMATH_CALUDE_stamp_theorem_l176_17603

/-- Represents the ability to form a value using given stamp denominations -/
def can_form (n : ℕ) (k : ℕ) : Prop :=
  ∃ (a b : ℕ), k = a * n + b * (n + 2)

/-- Theorem stating that for n = 3, any value k ≥ 8 can be formed using stamps of denominations 3 and 5 -/
theorem stamp_theorem :
  ∀ k : ℕ, k ≥ 8 → can_form 3 k :=
by sorry

end NUMINAMATH_CALUDE_stamp_theorem_l176_17603


namespace NUMINAMATH_CALUDE_cycling_time_problem_l176_17657

theorem cycling_time_problem (total_distance : ℝ) (total_time : ℝ) (initial_speed : ℝ) (reduced_speed : ℝ)
  (h1 : total_distance = 140)
  (h2 : total_time = 7)
  (h3 : initial_speed = 25)
  (h4 : reduced_speed = 15) :
  ∃ (energetic_time : ℝ), 
    energetic_time * initial_speed + (total_time - energetic_time) * reduced_speed = total_distance ∧
    energetic_time = 7/2 := by
  sorry

end NUMINAMATH_CALUDE_cycling_time_problem_l176_17657


namespace NUMINAMATH_CALUDE_smallest_fraction_l176_17647

theorem smallest_fraction (x : ℝ) (hx : x = 9) : 
  min ((x - 3) / 8) (min (8 / x) (min (8 / (x + 2)) (min (8 / (x - 2)) ((x + 3) / 8)))) = (x - 3) / 8 := by
  sorry

end NUMINAMATH_CALUDE_smallest_fraction_l176_17647


namespace NUMINAMATH_CALUDE_parallel_vectors_subtraction_l176_17661

/-- Given two parallel vectors a and b, prove that 2a - b equals (4, -8) -/
theorem parallel_vectors_subtraction (m : ℝ) :
  let a : ℝ × ℝ := (1, -2)
  let b : ℝ × ℝ := (m, 4)
  (a.1 * b.2 = a.2 * b.1) →  -- Condition for parallel vectors
  (2 * a.1 - b.1, 2 * a.2 - b.2) = (4, -8) := by
sorry

end NUMINAMATH_CALUDE_parallel_vectors_subtraction_l176_17661


namespace NUMINAMATH_CALUDE_sam_supplies_cost_school_supplies_cost_proof_l176_17626

/-- Represents the school supplies -/
structure Supplies :=
  (glue_sticks : ℕ)
  (pencils : ℕ)
  (erasers : ℕ)

/-- Calculates the cost of supplies given their quantities and prices -/
def calculate_cost (s : Supplies) (glue_price pencil_price eraser_price : ℚ) : ℚ :=
  s.glue_sticks * glue_price + s.pencils * pencil_price + s.erasers * eraser_price

theorem sam_supplies_cost (total : Supplies) (emily : Supplies) (sophie : Supplies)
    (glue_price pencil_price eraser_price : ℚ) : ℚ :=
  let sam : Supplies := {
    glue_sticks := total.glue_sticks - emily.glue_sticks - sophie.glue_sticks,
    pencils := total.pencils - emily.pencils - sophie.pencils,
    erasers := total.erasers - emily.erasers - sophie.erasers
  }
  calculate_cost sam glue_price pencil_price eraser_price

/-- The main theorem to prove -/
theorem school_supplies_cost_proof 
    (total : Supplies)
    (emily : Supplies)
    (sophie : Supplies)
    (glue_price pencil_price eraser_price : ℚ) :
    total.glue_sticks = 27 ∧ 
    total.pencils = 40 ∧ 
    total.erasers = 15 ∧
    glue_price = 1 ∧
    pencil_price = 1/2 ∧
    eraser_price = 3/4 ∧
    emily.glue_sticks = 9 ∧
    emily.pencils = 18 ∧
    emily.erasers = 5 ∧
    sophie.glue_sticks = 12 ∧
    sophie.pencils = 14 ∧
    sophie.erasers = 4 →
    sam_supplies_cost total emily sophie glue_price pencil_price eraser_price = 29/2 := by
  sorry

end NUMINAMATH_CALUDE_sam_supplies_cost_school_supplies_cost_proof_l176_17626


namespace NUMINAMATH_CALUDE_largest_angle_in_specific_hexagon_l176_17668

/-- Represents the ratio of angles in a hexagon -/
structure HexagonRatio :=
  (a b c d e f : ℕ)

/-- Calculates the largest angle in a hexagon given a ratio of angles -/
def largestAngleInHexagon (ratio : HexagonRatio) : ℚ :=
  let sum := ratio.a + ratio.b + ratio.c + ratio.d + ratio.e + ratio.f
  let angleUnit := 720 / sum
  angleUnit * (max ratio.a (max ratio.b (max ratio.c (max ratio.d (max ratio.e ratio.f)))))

theorem largest_angle_in_specific_hexagon :
  largestAngleInHexagon ⟨2, 3, 3, 4, 4, 5⟩ = 1200 / 7 := by
  sorry

#eval largestAngleInHexagon ⟨2, 3, 3, 4, 4, 5⟩

end NUMINAMATH_CALUDE_largest_angle_in_specific_hexagon_l176_17668


namespace NUMINAMATH_CALUDE_solution_values_l176_17610

-- Define the equations
def equation_1 (a x : ℝ) : Prop := a * x + 3 = 2 * (x - a)
def equation_2 (x : ℝ) : Prop := |x - 2| - 3 = 0

-- Theorem statement
theorem solution_values (a x : ℝ) :
  equation_1 a x ∧ equation_2 x → a = -5 ∨ a = 1 := by
  sorry

end NUMINAMATH_CALUDE_solution_values_l176_17610


namespace NUMINAMATH_CALUDE_crease_length_of_folded_equilateral_triangle_l176_17674

-- Define an equilateral triangle
structure EquilateralTriangle :=
  (side_length : ℝ)

-- Define the folded triangle
structure FoldedTriangle extends EquilateralTriangle :=
  (crease_length : ℝ)

-- Theorem statement
theorem crease_length_of_folded_equilateral_triangle 
  (triangle : EquilateralTriangle) 
  (h : triangle.side_length = 6) : 
  ∃ (folded : FoldedTriangle), 
    folded.side_length = triangle.side_length ∧ 
    folded.crease_length = 3 * Real.sqrt 3 :=
sorry

end NUMINAMATH_CALUDE_crease_length_of_folded_equilateral_triangle_l176_17674


namespace NUMINAMATH_CALUDE_millet_exceeds_half_on_fourth_day_l176_17678

/-- Represents the fraction of millet seeds in the feeder on a given day -/
def milletFraction (day : ℕ) : ℚ :=
  match day with
  | 0 => 3/10
  | n + 1 => (1/2 * milletFraction n + 3/10)

/-- Theorem stating that on the 4th day, the fraction of millet seeds exceeds 1/2 for the first time -/
theorem millet_exceeds_half_on_fourth_day :
  (milletFraction 4 > 1/2) ∧
  (∀ d : ℕ, d < 4 → milletFraction d ≤ 1/2) :=
sorry

end NUMINAMATH_CALUDE_millet_exceeds_half_on_fourth_day_l176_17678


namespace NUMINAMATH_CALUDE_fermats_little_theorem_l176_17638

theorem fermats_little_theorem (p : ℕ) (a : ℕ) (hp : Nat.Prime p) :
  a^p ≡ a [MOD p] := by
  sorry

end NUMINAMATH_CALUDE_fermats_little_theorem_l176_17638


namespace NUMINAMATH_CALUDE_power_of_product_l176_17605

theorem power_of_product (a b : ℝ) : (a^2 * b)^3 = a^6 * b^3 := by
  sorry

end NUMINAMATH_CALUDE_power_of_product_l176_17605


namespace NUMINAMATH_CALUDE_circle_center_and_chord_length_l176_17660

/-- Definition of the circle C -/
def circle_C (x y : ℝ) : Prop := x^2 + y^2 - 2*x = 0

/-- Definition of the line y = x -/
def line_y_eq_x (x y : ℝ) : Prop := y = x

theorem circle_center_and_chord_length :
  ∃ (center_x center_y : ℝ) (chord_length : ℝ),
    (∀ x y, circle_C x y ↔ (x - center_x)^2 + (y - center_y)^2 = 1) ∧
    center_x = 1 ∧
    center_y = 0 ∧
    chord_length = Real.sqrt 2 ∧
    chord_length^2 = 2 * (1 - (1 / Real.sqrt 2)^2) :=
sorry

end NUMINAMATH_CALUDE_circle_center_and_chord_length_l176_17660


namespace NUMINAMATH_CALUDE_quadratic_max_value_l176_17699

/-- The quadratic function y = -(x-m)^2 + m^2 + 1 has a maximum value of 4 when -2 ≤ x ≤ 1. -/
theorem quadratic_max_value (m : ℝ) : 
  (∀ x, -2 ≤ x ∧ x ≤ 1 → -(x-m)^2 + m^2 + 1 ≤ 4) ∧ 
  (∃ x, -2 ≤ x ∧ x ≤ 1 ∧ -(x-m)^2 + m^2 + 1 = 4) →
  m = 2 ∨ m = -Real.sqrt 3 :=
sorry

end NUMINAMATH_CALUDE_quadratic_max_value_l176_17699


namespace NUMINAMATH_CALUDE_problem_solution_l176_17690

-- Define the function f
def f (x : ℝ) := |2*x - 2| + |x + 2|

-- Define the theorem
theorem problem_solution :
  (∃ (S : Set ℝ), S = {x : ℝ | -3 ≤ x ∧ x ≤ 3/2} ∧ ∀ x, f x ≤ 6 - x ↔ x ∈ S) ∧
  (∃ (T : ℝ), T = 3 ∧ ∀ x, f x ≥ T) ∧
  (∀ a b c : ℝ, a > 0 → b > 0 → c > 0 → a + b + c = 3 → 1/a + 1/b + 4/c ≥ 16/3) :=
by sorry

end NUMINAMATH_CALUDE_problem_solution_l176_17690


namespace NUMINAMATH_CALUDE_iced_cube_theorem_l176_17629

/-- Represents a cube with icing on some faces -/
structure IcedCube :=
  (size : ℕ)
  (has_top_icing : Bool)
  (has_lateral_icing : Bool)
  (has_bottom_icing : Bool)

/-- Counts the number of subcubes with icing on exactly two sides -/
def count_two_sided_iced_subcubes (cube : IcedCube) : ℕ :=
  sorry

/-- The main theorem about the 5x5x5 iced cube -/
theorem iced_cube_theorem :
  let cake : IcedCube := {
    size := 5,
    has_top_icing := true,
    has_lateral_icing := true,
    has_bottom_icing := false
  }
  count_two_sided_iced_subcubes cake = 32 :=
sorry

end NUMINAMATH_CALUDE_iced_cube_theorem_l176_17629


namespace NUMINAMATH_CALUDE_least_x_72_implies_n_8_l176_17600

theorem least_x_72_implies_n_8 (x : ℕ+) (p : ℕ) (n : ℕ+) :
  Nat.Prime p →
  (∃ q : ℕ, Nat.Prime q ∧ q % 2 = 1 ∧ (x : ℚ) / (n * p : ℚ) = q) →
  (∀ y : ℕ+, y < x → ¬∃ q : ℕ, Nat.Prime q ∧ q % 2 = 1 ∧ (y : ℚ) / (n * p : ℚ) = q) →
  x = 72 →
  n = 8 :=
by sorry

end NUMINAMATH_CALUDE_least_x_72_implies_n_8_l176_17600


namespace NUMINAMATH_CALUDE_vincent_songs_before_camp_l176_17608

/-- The number of songs Vincent knows now -/
def total_songs : ℕ := 74

/-- The number of songs Vincent learned at summer camp -/
def learned_at_camp : ℕ := 18

/-- The number of songs Vincent knew before summer camp -/
def songs_before_camp : ℕ := total_songs - learned_at_camp

theorem vincent_songs_before_camp :
  songs_before_camp = 56 :=
sorry

end NUMINAMATH_CALUDE_vincent_songs_before_camp_l176_17608


namespace NUMINAMATH_CALUDE_expression_equality_l176_17687

theorem expression_equality : (3^1003 + 5^1003)^2 - (3^1003 - 5^1003)^2 = 4 * 15^1003 := by
  sorry

end NUMINAMATH_CALUDE_expression_equality_l176_17687


namespace NUMINAMATH_CALUDE_angle_B_is_45_degrees_l176_17677

theorem angle_B_is_45_degrees 
  (A B C : Real) 
  (a b c : Real) 
  (triangle_inequality : a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b > c ∧ b + c > a ∧ c + a > b) 
  (side_angle_correspondence : a = BC ∧ b = AC ∧ c = AB) 
  (equation : a^2 + c^2 = b^2 + Real.sqrt 2 * a * c) : 
  B = 45 * π / 180 := by
sorry

end NUMINAMATH_CALUDE_angle_B_is_45_degrees_l176_17677


namespace NUMINAMATH_CALUDE_initial_mean_calculation_l176_17696

theorem initial_mean_calculation (n : ℕ) (wrong_value correct_value : ℝ) (corrected_mean : ℝ) :
  n = 40 ∧ wrong_value = 20 ∧ correct_value = 34 ∧ corrected_mean = 36.45 →
  ∃ initial_mean : ℝ, initial_mean = 36.1 ∧
    n * corrected_mean = n * initial_mean + (correct_value - wrong_value) :=
by sorry

end NUMINAMATH_CALUDE_initial_mean_calculation_l176_17696


namespace NUMINAMATH_CALUDE_triangle_not_unique_l176_17652

/-- A triangle is defined by three side lengths -/
structure Triangle :=
  (a b c : ℝ)

/-- Predicate to check if three real numbers can form a triangle -/
def is_triangle (a b c : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b > c ∧ a + c > b ∧ b + c > a

/-- Given one side and the sum of the other two sides, 
    the triangle is not uniquely determined -/
theorem triangle_not_unique (s : ℝ) (sum : ℝ) :
  ∃ (t1 t2 : Triangle),
    t1 ≠ t2 ∧
    t1.a = s ∧
    t2.a = s ∧
    t1.b + t1.c = sum ∧
    t2.b + t2.c = sum ∧
    is_triangle t1.a t1.b t1.c ∧
    is_triangle t2.a t2.b t2.c :=
  sorry


end NUMINAMATH_CALUDE_triangle_not_unique_l176_17652


namespace NUMINAMATH_CALUDE_smallest_number_l176_17662

def jungkook_number : ℚ := 6 / 3
def yoongi_number : ℚ := 4
def yuna_number : ℚ := 5

theorem smallest_number : 
  jungkook_number ≤ yoongi_number ∧ jungkook_number ≤ yuna_number :=
by sorry

end NUMINAMATH_CALUDE_smallest_number_l176_17662


namespace NUMINAMATH_CALUDE_events_mutually_exclusive_not_complementary_l176_17658

/-- A bag containing red and white balls -/
structure Bag where
  red : Nat
  white : Nat

/-- The event of drawing exactly one white ball -/
def exactlyOneWhite (b : Bag) : Prop := sorry

/-- The event of drawing exactly two white balls -/
def exactlyTwoWhite (b : Bag) : Prop := sorry

/-- Two events are mutually exclusive -/
def mutuallyExclusive (e1 e2 : Prop) : Prop := ¬(e1 ∧ e2)

/-- Two events are complementary -/
def complementary (e1 e2 : Prop) : Prop := (e1 ∨ e2) ∧ mutuallyExclusive e1 e2

/-- The main theorem -/
theorem events_mutually_exclusive_not_complementary (b : Bag) 
  (h : b.red = 2 ∧ b.white = 2) : 
  mutuallyExclusive (exactlyOneWhite b) (exactlyTwoWhite b) ∧ 
  ¬complementary (exactlyOneWhite b) (exactlyTwoWhite b) := by
  sorry

end NUMINAMATH_CALUDE_events_mutually_exclusive_not_complementary_l176_17658


namespace NUMINAMATH_CALUDE_equilateral_triangle_division_l176_17650

/-- A type representing a polygon with a given number of sides -/
def Polygon (n : ℕ) := Unit

/-- A type representing an equilateral triangle -/
def EquilateralTriangle := Unit

/-- A function that divides an equilateral triangle into two polygons -/
def divide (t : EquilateralTriangle) : Polygon 2020 × Polygon 2021 := sorry

/-- Theorem stating that an equilateral triangle can be divided into a 2020-gon and a 2021-gon -/
theorem equilateral_triangle_division :
  ∃ (t : EquilateralTriangle), ∃ (p : Polygon 2020 × Polygon 2021), divide t = p := by sorry

end NUMINAMATH_CALUDE_equilateral_triangle_division_l176_17650


namespace NUMINAMATH_CALUDE_eulers_formula_eulers_identity_complex_exp_sum_bound_l176_17632

-- Define the complex exponential function
noncomputable def cexp (z : ℂ) : ℂ := sorry

-- Define the imaginary unit
def i : ℂ := sorry

-- Define pi
noncomputable def π : ℝ := sorry

-- Theorem 1: Euler's formula
theorem eulers_formula (x : ℝ) : cexp (i * x) = Complex.cos x + i * Complex.sin x := by sorry

-- Theorem 2: Euler's identity
theorem eulers_identity : cexp (i * π) + 1 = 0 := by sorry

-- Theorem 3: Bound on sum of complex exponentials
theorem complex_exp_sum_bound (x : ℝ) : Complex.abs (cexp (i * x) + cexp (-i * x)) ≤ 2 := by sorry

end NUMINAMATH_CALUDE_eulers_formula_eulers_identity_complex_exp_sum_bound_l176_17632


namespace NUMINAMATH_CALUDE_probability_both_defective_six_two_two_l176_17636

/-- The probability of both selected products being defective, given that one is defective -/
def probability_both_defective (total : ℕ) (defective : ℕ) (selected : ℕ) : ℚ :=
  if total ≥ defective ∧ total ≥ selected ∧ selected > 0 then
    (defective.choose (selected - 1)) / (total.choose 1 * (total - 1).choose (selected - 1))
  else
    0

theorem probability_both_defective_six_two_two :
  probability_both_defective 6 2 2 = 1 / 15 := by
  sorry

end NUMINAMATH_CALUDE_probability_both_defective_six_two_two_l176_17636


namespace NUMINAMATH_CALUDE_product_expansion_l176_17693

theorem product_expansion (x : ℝ) : 
  (2 * x^2 - 3 * x + 4) * (2 * x^2 + 3 * x + 4) = 4 * x^4 + 7 * x^2 + 16 := by
  sorry

end NUMINAMATH_CALUDE_product_expansion_l176_17693


namespace NUMINAMATH_CALUDE_smallest_n_with_properties_l176_17676

theorem smallest_n_with_properties : 
  ∃ (n : ℕ), n > 0 ∧ 
  (∃ (k : ℕ), 2 * n = k^2) ∧ 
  (∃ (l : ℕ), 3 * n = l^3) ∧ 
  (∃ (m : ℕ), 5 * n = m^5) ∧ 
  (∀ (x : ℕ), x > 0 ∧ 
    (∃ (k : ℕ), 2 * x = k^2) ∧ 
    (∃ (l : ℕ), 3 * x = l^3) ∧ 
    (∃ (m : ℕ), 5 * x = m^5) → 
    x ≥ n) ∧ 
  n = 11250 :=
sorry

end NUMINAMATH_CALUDE_smallest_n_with_properties_l176_17676


namespace NUMINAMATH_CALUDE_art_of_passing_through_walls_l176_17606

theorem art_of_passing_through_walls (n : ℕ) : 
  (8 * Real.sqrt (8 / n) = Real.sqrt (8 * 8 / n)) ↔ n = 63 :=
sorry

end NUMINAMATH_CALUDE_art_of_passing_through_walls_l176_17606


namespace NUMINAMATH_CALUDE_complex_fraction_evaluation_l176_17648

theorem complex_fraction_evaluation :
  (2 : ℂ) / (Complex.I * (3 - Complex.I)) = (1 / 5 : ℂ) - (3 / 5 : ℂ) * Complex.I :=
by sorry

end NUMINAMATH_CALUDE_complex_fraction_evaluation_l176_17648


namespace NUMINAMATH_CALUDE_calculate_expression_l176_17697

theorem calculate_expression : -2⁻¹ * (-8) + (2022)^0 - Real.sqrt 9 - abs (-4) = -3 := by
  sorry

end NUMINAMATH_CALUDE_calculate_expression_l176_17697


namespace NUMINAMATH_CALUDE_second_company_base_rate_l176_17635

/-- The base rate of United Telephone in dollars -/
def united_base_rate : ℝ := 8.00

/-- The per-minute rate of United Telephone in dollars -/
def united_per_minute : ℝ := 0.25

/-- The per-minute rate of the second company in dollars -/
def second_per_minute : ℝ := 0.20

/-- The number of minutes at which the bills are equal -/
def equal_minutes : ℝ := 80

/-- The base rate of the second company in dollars -/
def second_base_rate : ℝ := 12.00

/-- Theorem stating that the base rate of the second company is $12.00 -/
theorem second_company_base_rate :
  united_base_rate + united_per_minute * equal_minutes =
  second_base_rate + second_per_minute * equal_minutes :=
by sorry

end NUMINAMATH_CALUDE_second_company_base_rate_l176_17635


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l176_17689

/-- The eccentricity of a hyperbola with specific properties -/
theorem hyperbola_eccentricity (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) : 
  (∀ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1 → 
    ∃ B C : ℝ × ℝ, 
      B.1 = c ∧ C.1 = c ∧ 
      B.2^2 = (b^2 / a^2) * (c^2 - a^2) ∧ 
      C.2^2 = (b^2 / a^2) * (c^2 - a^2) ∧
      (B.2 - C.2)^2 = 2 * (c + a)^2) →
  c / a = 2 := by
sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l176_17689


namespace NUMINAMATH_CALUDE_tangent_line_cubic_l176_17641

/-- The equation of the tangent line to y = x³ at (1, 1) is 3x - y - 2 = 0 -/
theorem tangent_line_cubic (x y : ℝ) : 
  (y = x^3) → -- The curve equation
  (x = 1 ∧ y = 1) → -- The point (1, 1) on the curve
  (3*x - y - 2 = 0) -- The equation of the tangent line
:= by sorry

end NUMINAMATH_CALUDE_tangent_line_cubic_l176_17641


namespace NUMINAMATH_CALUDE_factor_theorem_application_l176_17684

-- Define the polynomial P(x)
def P (c : ℝ) (x : ℝ) : ℝ := x^3 + 2*x^2 + c*x + 10

-- Theorem statement
theorem factor_theorem_application (c : ℝ) :
  (∀ x, P c x = 0 ↔ x = 5) → c = -37 := by
  sorry

end NUMINAMATH_CALUDE_factor_theorem_application_l176_17684


namespace NUMINAMATH_CALUDE_gcd_459_357_l176_17640

theorem gcd_459_357 : Nat.gcd 459 357 = 51 := by
  sorry

end NUMINAMATH_CALUDE_gcd_459_357_l176_17640


namespace NUMINAMATH_CALUDE_initial_average_production_l176_17621

theorem initial_average_production (n : ℕ) (today_production : ℕ) (new_average : ℕ) 
  (h1 : n = 9)
  (h2 : today_production = 90)
  (h3 : new_average = 54) :
  ∃ initial_average : ℕ, 
    initial_average * n + today_production = new_average * (n + 1) ∧ 
    initial_average = 50 := by
  sorry

end NUMINAMATH_CALUDE_initial_average_production_l176_17621


namespace NUMINAMATH_CALUDE_optimal_meal_plan_l176_17602

/-- Represents the nutritional content of a meal -/
structure Nutrition :=
  (carbs : ℕ)
  (protein : ℕ)
  (vitaminC : ℕ)

/-- Represents the meal plan -/
structure MealPlan :=
  (lunch : ℕ)
  (dinner : ℕ)

def lunch_nutrition : Nutrition := ⟨12, 6, 6⟩
def dinner_nutrition : Nutrition := ⟨8, 6, 10⟩

def lunch_cost : ℚ := 2.5
def dinner_cost : ℚ := 4

def minimum_nutrition : Nutrition := ⟨64, 42, 54⟩

def total_nutrition (plan : MealPlan) : Nutrition :=
  ⟨plan.lunch * lunch_nutrition.carbs + plan.dinner * dinner_nutrition.carbs,
   plan.lunch * lunch_nutrition.protein + plan.dinner * dinner_nutrition.protein,
   plan.lunch * lunch_nutrition.vitaminC + plan.dinner * dinner_nutrition.vitaminC⟩

def meets_requirements (plan : MealPlan) : Prop :=
  let total := total_nutrition plan
  total.carbs ≥ minimum_nutrition.carbs ∧
  total.protein ≥ minimum_nutrition.protein ∧
  total.vitaminC ≥ minimum_nutrition.vitaminC

def total_cost (plan : MealPlan) : ℚ :=
  plan.lunch * lunch_cost + plan.dinner * dinner_cost

theorem optimal_meal_plan :
  ∃ (plan : MealPlan),
    meets_requirements plan ∧
    (∀ (other : MealPlan), meets_requirements other → total_cost plan ≤ total_cost other) ∧
    plan.lunch = 4 ∧ plan.dinner = 3 :=
sorry

end NUMINAMATH_CALUDE_optimal_meal_plan_l176_17602


namespace NUMINAMATH_CALUDE_water_distribution_solution_l176_17645

/-- Represents the water distribution problem -/
structure WaterDistribution where
  totalWater : Nat
  fiveOunceGlasses : Nat
  eightOunceGlasses : Nat
  sevenOunceGlasses : Nat

/-- Calculates the maximum number of friends and remaining 4-ounce glasses -/
def distributeWater (w : WaterDistribution) : Nat × Nat :=
  let usedWater := w.fiveOunceGlasses * 5 + w.eightOunceGlasses * 8 + w.sevenOunceGlasses * 7
  let remainingWater := w.totalWater - usedWater
  let fourOunceGlasses := remainingWater / 4
  let totalGlasses := w.fiveOunceGlasses + w.eightOunceGlasses + w.sevenOunceGlasses + fourOunceGlasses
  (totalGlasses, fourOunceGlasses)

/-- Theorem stating the solution to the water distribution problem -/
theorem water_distribution_solution (w : WaterDistribution) 
  (h1 : w.totalWater = 122)
  (h2 : w.fiveOunceGlasses = 6)
  (h3 : w.eightOunceGlasses = 4)
  (h4 : w.sevenOunceGlasses = 3) :
  distributeWater w = (22, 9) := by
  sorry

end NUMINAMATH_CALUDE_water_distribution_solution_l176_17645


namespace NUMINAMATH_CALUDE_campground_distance_l176_17649

theorem campground_distance (speed1 speed2 speed3 time1 time2 time3 : ℝ) 
  (h1 : speed1 = 60)
  (h2 : speed2 = 50)
  (h3 : speed3 = 55)
  (h4 : time1 = 2)
  (h5 : time2 = 3)
  (h6 : time3 = 4) :
  speed1 * time1 + speed2 * time2 + speed3 * time3 = 490 :=
by
  sorry

end NUMINAMATH_CALUDE_campground_distance_l176_17649


namespace NUMINAMATH_CALUDE_roots_sum_powers_l176_17633

theorem roots_sum_powers (a b : ℝ) : 
  a^2 - 4*a + 5 = 0 → b^2 - 4*b + 5 = 0 → a^3 + a^4*b^2 + a^2*b^4 + b^3 = 154 := by
  sorry

end NUMINAMATH_CALUDE_roots_sum_powers_l176_17633


namespace NUMINAMATH_CALUDE_perfect_square_trinomial_l176_17630

theorem perfect_square_trinomial (m : ℝ) : 
  (∃ a b : ℝ, ∀ x, x^2 - (m-3)*x + 16 = (a*x + b)^2) ↔ (m = -5 ∨ m = 11) :=
sorry

end NUMINAMATH_CALUDE_perfect_square_trinomial_l176_17630


namespace NUMINAMATH_CALUDE_XAXAXA_divisible_by_seven_l176_17665

/-- Given two digits X and A, XAXAXA is the six-digit number formed by repeating XA three times -/
def XAXAXA (X A : ℕ) : ℕ :=
  100000 * X + 10000 * A + 1000 * X + 100 * A + 10 * X + A

/-- Theorem: For any two digits X and A, XAXAXA is divisible by 7 -/
theorem XAXAXA_divisible_by_seven (X A : ℕ) (hX : X < 10) (hA : A < 10) :
  ∃ k, XAXAXA X A = 7 * k :=
sorry

end NUMINAMATH_CALUDE_XAXAXA_divisible_by_seven_l176_17665


namespace NUMINAMATH_CALUDE_special_permutations_count_l176_17670

/-- The number of permutations of n distinct elements where a₁ is not in the 1st position,
    a₂ is not in the 2nd position, and a₃ is not in the 3rd position. -/
def special_permutations (n : ℕ) : ℕ :=
  (n^3 - 6*n^2 + 14*n - 13) * Nat.factorial (n - 3)

/-- Theorem stating that for n ≥ 3, the number of permutations of n distinct elements
    where a₁ is not in the 1st position, a₂ is not in the 2nd position, and a₃ is not
    in the 3rd position is equal to (n³ - 6n² + 14n - 13) * (n-3)! -/
theorem special_permutations_count (n : ℕ) (h : n ≥ 3) :
  special_permutations n = (n^3 - 6*n^2 + 14*n - 13) * Nat.factorial (n - 3) := by
  sorry

end NUMINAMATH_CALUDE_special_permutations_count_l176_17670


namespace NUMINAMATH_CALUDE_mia_has_110_dollars_l176_17692

def darwins_money : ℕ := 45

def mias_money : ℕ := 2 * darwins_money + 20

theorem mia_has_110_dollars : mias_money = 110 := by
  sorry

end NUMINAMATH_CALUDE_mia_has_110_dollars_l176_17692


namespace NUMINAMATH_CALUDE_max_distinct_angles_for_ten_points_l176_17609

/-- The number of points on the circle -/
def n : ℕ := 10

/-- The maximum number of distinct inscribed angle values -/
def max_distinct_angles : ℕ := 80

/-- A function that calculates the number of distinct inscribed angle values
    given the number of points on a circle -/
noncomputable def distinct_angles (points : ℕ) : ℕ := sorry

/-- Theorem stating that the maximum number of distinct inscribed angle values
    for 10 points on a circle is 80 -/
theorem max_distinct_angles_for_ten_points :
  distinct_angles n = max_distinct_angles :=
sorry

end NUMINAMATH_CALUDE_max_distinct_angles_for_ten_points_l176_17609


namespace NUMINAMATH_CALUDE_tank_water_supply_l176_17659

theorem tank_water_supply (C V : ℝ) 
  (h1 : C = 75 * (V + 10))
  (h2 : C = 60 * (V + 20)) :
  C / V = 100 := by
  sorry

end NUMINAMATH_CALUDE_tank_water_supply_l176_17659


namespace NUMINAMATH_CALUDE_inequality_of_distinct_reals_l176_17639

theorem inequality_of_distinct_reals (a b c : ℝ) 
  (hab : a ≠ b) (hbc : b ≠ c) (hca : c ≠ a) : 
  |a / (b - c)| + |b / (c - a)| + |c / (a - b)| ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_of_distinct_reals_l176_17639


namespace NUMINAMATH_CALUDE_rounding_estimate_larger_l176_17620

theorem rounding_estimate_larger (a b c d a' b' c' d' : ℕ) : 
  a > 0 → b > 0 → c > 0 → d > 0 →
  a' ≥ a → b' ≤ b → c' ≤ c → d' ≤ d →
  (a' : ℚ) / b' - c' - d' > (a : ℚ) / b - c - d :=
by sorry

end NUMINAMATH_CALUDE_rounding_estimate_larger_l176_17620


namespace NUMINAMATH_CALUDE_abs_ratio_eq_sqrt_seven_halves_l176_17663

theorem abs_ratio_eq_sqrt_seven_halves (a b : ℝ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : a^2 + b^2 = 5*a*b) :
  |((a + b) / (a - b))| = Real.sqrt (7/2) := by
  sorry

end NUMINAMATH_CALUDE_abs_ratio_eq_sqrt_seven_halves_l176_17663


namespace NUMINAMATH_CALUDE_investment_sum_l176_17669

/-- Proves that if a sum P invested at 18% p.a. simple interest for two years yields Rs. 600 more interest than if invested at 12% p.a. simple interest for the same period, then P = 5000. -/
theorem investment_sum (P : ℝ) : 
  P * (18 / 100) * 2 - P * (12 / 100) * 2 = 600 → P = 5000 := by
  sorry

end NUMINAMATH_CALUDE_investment_sum_l176_17669


namespace NUMINAMATH_CALUDE_inequalities_satisfied_l176_17698

theorem inequalities_satisfied (a b c x y z : ℝ) 
  (h1 : x ≤ a) (h2 : y ≤ b) (h3 : z ≤ c) : 
  (x*y + y*z + z*x ≤ a*b + b*c + c*a + 3) ∧ 
  (x^2 + y^2 + z^2 ≤ a^2 + b^2 + c^2 + 3) ∧ 
  (x*y*z ≤ a*b*c + 1) := by
  sorry

end NUMINAMATH_CALUDE_inequalities_satisfied_l176_17698


namespace NUMINAMATH_CALUDE_paco_cookie_consumption_l176_17672

/-- Represents the number of sweet cookies Paco ate -/
def sweet_cookies_eaten : ℕ := 15

/-- Represents the initial number of sweet cookies Paco had -/
def initial_sweet_cookies : ℕ := 40

/-- Represents the initial number of salty cookies Paco had -/
def initial_salty_cookies : ℕ := 25

/-- Represents the number of salty cookies Paco ate -/
def salty_cookies_eaten : ℕ := 28

theorem paco_cookie_consumption :
  sweet_cookies_eaten = 15 ∧
  initial_sweet_cookies = 40 ∧
  initial_salty_cookies = 25 ∧
  salty_cookies_eaten = 28 ∧
  salty_cookies_eaten = sweet_cookies_eaten + 13 :=
by sorry

end NUMINAMATH_CALUDE_paco_cookie_consumption_l176_17672


namespace NUMINAMATH_CALUDE_dave_spent_102_l176_17628

/-- The amount Dave spent on books -/
def dave_spent (animal_books outer_space_books train_books cost_per_book : ℕ) : ℕ :=
  (animal_books + outer_space_books + train_books) * cost_per_book

/-- Theorem stating that Dave spent $102 on books -/
theorem dave_spent_102 :
  dave_spent 8 6 3 6 = 102 := by
  sorry

end NUMINAMATH_CALUDE_dave_spent_102_l176_17628


namespace NUMINAMATH_CALUDE_expression_simplification_l176_17686

theorem expression_simplification (x : ℚ) (h : x = -3) :
  (1 - 1 / (x - 1)) / ((x^2 - 4*x + 4) / (x^2 - 1)) = 2/5 :=
by sorry

end NUMINAMATH_CALUDE_expression_simplification_l176_17686


namespace NUMINAMATH_CALUDE_profit_percentage_l176_17642

/-- Given that the cost price of 25 articles equals the selling price of 18 articles,
    prove that the profit percentage is 700/18. -/
theorem profit_percentage (C S : ℝ) (h : 25 * C = 18 * S) :
  (S - C) / C * 100 = 700 / 18 := by
  sorry

end NUMINAMATH_CALUDE_profit_percentage_l176_17642
