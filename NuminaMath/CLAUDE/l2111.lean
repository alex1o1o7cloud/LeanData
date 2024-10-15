import Mathlib

namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l2111_211124

theorem quadratic_inequality_solution_set (a : ℝ) :
  (∀ x : ℝ, a * x^2 + x - 1 ≤ 0) → a ≤ -1/4 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l2111_211124


namespace NUMINAMATH_CALUDE_flour_recipe_reduction_reduced_recipe_as_mixed_number_l2111_211106

theorem flour_recipe_reduction :
  let original_recipe : ℚ := 19/4  -- 4 3/4 as an improper fraction
  let reduced_recipe : ℚ := original_recipe / 3
  reduced_recipe = 19/12 := by sorry

theorem reduced_recipe_as_mixed_number :
  (19 : ℚ) / 12 = 1 + 7/12 := by sorry

end NUMINAMATH_CALUDE_flour_recipe_reduction_reduced_recipe_as_mixed_number_l2111_211106


namespace NUMINAMATH_CALUDE_kenzo_round_tables_l2111_211174

/-- The number of round tables Kenzo initially had -/
def num_round_tables : ℕ := 20

/-- The number of office chairs Kenzo initially had -/
def initial_chairs : ℕ := 80

/-- The number of legs each office chair has -/
def legs_per_chair : ℕ := 5

/-- The number of legs each round table has -/
def legs_per_table : ℕ := 3

/-- The percentage of chairs that were damaged and disposed of -/
def damaged_chair_percentage : ℚ := 40 / 100

/-- The total number of remaining legs of furniture -/
def total_remaining_legs : ℕ := 300

theorem kenzo_round_tables :
  num_round_tables * legs_per_table = 
    total_remaining_legs - 
    (initial_chairs * (1 - damaged_chair_percentage) : ℚ).num * legs_per_chair :=
by sorry

end NUMINAMATH_CALUDE_kenzo_round_tables_l2111_211174


namespace NUMINAMATH_CALUDE_congruence_solution_l2111_211171

theorem congruence_solution (x : ℤ) : 
  x ∈ Finset.Icc 20 50 ∧ (6 * x + 5) % 10 = 19 % 10 ↔ 
  x ∈ ({24, 29, 34, 39, 44, 49} : Finset ℤ) := by
sorry

end NUMINAMATH_CALUDE_congruence_solution_l2111_211171


namespace NUMINAMATH_CALUDE_problem_statement_l2111_211181

-- Define sets A, B, and C
def A : Set ℝ := {x | |3*x - 4| > 2}
def B : Set ℝ := {x | x^2 - x - 2 > 0}
def C (a : ℝ) : Set ℝ := {x | (x - a) * (x - a - 1) ≥ 0}

-- Define predicates p, q, and r
def p : Set ℝ := {x | 2/3 ≤ x ∧ x ≤ 2}
def q : Set ℝ := {x | -1 ≤ x ∧ x ≤ 2}
def r (a : ℝ) : Set ℝ := {x | x ≤ a ∨ x ≥ a + 1}

theorem problem_statement :
  (∀ x : ℝ, x ∈ p → x ∈ q) ∧
  (∃ x : ℝ, x ∈ q ∧ x ∉ p) ∧
  (∀ a : ℝ, (∀ x : ℝ, x ∈ p → x ∈ r a) ∧ (∃ x : ℝ, x ∈ r a ∧ x ∉ p) ↔ a ≥ 2 ∨ a ≤ -1/3) :=
by sorry

end NUMINAMATH_CALUDE_problem_statement_l2111_211181


namespace NUMINAMATH_CALUDE_factor_expression_l2111_211190

theorem factor_expression (x : ℝ) : 4*x*(x+2) + 10*(x+2) + 2*(x+2) = (x+2)*(4*x+12) := by
  sorry

end NUMINAMATH_CALUDE_factor_expression_l2111_211190


namespace NUMINAMATH_CALUDE_johns_final_push_time_l2111_211102

/-- The time of John's final push in a speed walking race --/
theorem johns_final_push_time (john_initial_distance_behind : ℝ)
                               (john_speed : ℝ)
                               (steve_speed : ℝ)
                               (john_final_distance_ahead : ℝ)
                               (h1 : john_initial_distance_behind = 16)
                               (h2 : john_speed = 4.2)
                               (h3 : steve_speed = 3.7)
                               (h4 : john_final_distance_ahead = 2) :
  let t : ℝ := (john_initial_distance_behind + john_final_distance_ahead) / (john_speed - steve_speed)
  t = 36 := by
  sorry

end NUMINAMATH_CALUDE_johns_final_push_time_l2111_211102


namespace NUMINAMATH_CALUDE_sum_of_reciprocals_lower_bound_l2111_211182

theorem sum_of_reciprocals_lower_bound (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b ≤ 4) :
  1/a + 1/b ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_reciprocals_lower_bound_l2111_211182


namespace NUMINAMATH_CALUDE_max_positive_integers_l2111_211130

theorem max_positive_integers (a b c d e f : ℤ) (h : a * b + c * d * e * f < 0) :
  ∃ (pos : Finset ℤ), pos ⊆ {a, b, c, d, e, f} ∧ pos.card ≤ 5 ∧
  (∀ x ∈ pos, x > 0) ∧
  (∀ pos' : Finset ℤ, pos' ⊆ {a, b, c, d, e, f} → (∀ x ∈ pos', x > 0) → pos'.card ≤ pos.card) :=
by sorry

end NUMINAMATH_CALUDE_max_positive_integers_l2111_211130


namespace NUMINAMATH_CALUDE_tablet_diagonal_comparison_l2111_211103

theorem tablet_diagonal_comparison (d : ℝ) : 
  d > 0 →  -- d is positive (diagonal length)
  (6 / Real.sqrt 2)^2 = (d / Real.sqrt 2)^2 + 5.5 →  -- area comparison
  d = 5 := by
sorry

end NUMINAMATH_CALUDE_tablet_diagonal_comparison_l2111_211103


namespace NUMINAMATH_CALUDE_y_in_terms_of_x_l2111_211128

theorem y_in_terms_of_x (x y : ℝ) (h : x + y = -1) : y = -1 - x := by
  sorry

end NUMINAMATH_CALUDE_y_in_terms_of_x_l2111_211128


namespace NUMINAMATH_CALUDE_average_after_17th_inning_l2111_211189

def batsman_average (previous_innings : ℕ) (previous_total : ℕ) (new_score : ℕ) : ℚ :=
  (previous_total + new_score) / (previous_innings + 1)

theorem average_after_17th_inning 
  (previous_innings : ℕ) 
  (previous_total : ℕ) 
  (new_score : ℕ) 
  (average_increase : ℚ) :
  previous_innings = 16 →
  new_score = 88 →
  average_increase = 3 →
  batsman_average previous_innings previous_total new_score - 
    (previous_total / previous_innings) = average_increase →
  batsman_average previous_innings previous_total new_score = 40 :=
by
  sorry

#check average_after_17th_inning

end NUMINAMATH_CALUDE_average_after_17th_inning_l2111_211189


namespace NUMINAMATH_CALUDE_hannah_running_difference_l2111_211134

def monday_distance : ℕ := 9
def wednesday_distance : ℕ := 4816
def friday_distance : ℕ := 2095

theorem hannah_running_difference :
  (monday_distance * 1000) - (wednesday_distance + friday_distance) = 2089 := by
  sorry

end NUMINAMATH_CALUDE_hannah_running_difference_l2111_211134


namespace NUMINAMATH_CALUDE_largest_integer_with_two_digit_square_l2111_211155

theorem largest_integer_with_two_digit_square : ∃ M : ℕ, 
  (∀ n : ℕ, n^2 ≥ 10 ∧ n^2 < 100 → n ≤ M) ∧ 
  M^2 ≥ 10 ∧ M^2 < 100 ∧ 
  M = 9 := by
  sorry

end NUMINAMATH_CALUDE_largest_integer_with_two_digit_square_l2111_211155


namespace NUMINAMATH_CALUDE_min_value_theorem_l2111_211115

theorem min_value_theorem (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h_line : 2*a + b = 2) : 
  (∀ x y : ℝ, x > 0 → y > 0 → 2*x + y = 2 → 1/a + 2/b ≤ 1/x + 2/y) ∧ 
  (∃ x y : ℝ, x > 0 ∧ y > 0 ∧ 2*x + y = 2 ∧ 1/x + 2/y = 4) :=
sorry

end NUMINAMATH_CALUDE_min_value_theorem_l2111_211115


namespace NUMINAMATH_CALUDE_matrix_power_2023_l2111_211149

def A : Matrix (Fin 2) (Fin 2) ℕ := !![1, 0; 2, 1]

theorem matrix_power_2023 :
  A ^ 2023 = !![1, 0; 4046, 1] := by sorry

end NUMINAMATH_CALUDE_matrix_power_2023_l2111_211149


namespace NUMINAMATH_CALUDE_mod_17_equivalence_l2111_211105

theorem mod_17_equivalence : ∃! n : ℤ, 0 ≤ n ∧ n < 17 ∧ 42762 % 17 = n % 17 ∧ n = 7 := by
  sorry

end NUMINAMATH_CALUDE_mod_17_equivalence_l2111_211105


namespace NUMINAMATH_CALUDE_smallest_x_absolute_value_equation_l2111_211180

theorem smallest_x_absolute_value_equation : 
  ∃ x : ℝ, x = -8.6 ∧ ∀ y : ℝ, |5 * y + 9| = 34 → y ≥ x := by
  sorry

end NUMINAMATH_CALUDE_smallest_x_absolute_value_equation_l2111_211180


namespace NUMINAMATH_CALUDE_lcm_1188_924_l2111_211137

theorem lcm_1188_924 : Nat.lcm 1188 924 = 8316 := by
  sorry

end NUMINAMATH_CALUDE_lcm_1188_924_l2111_211137


namespace NUMINAMATH_CALUDE_max_value_fraction_l2111_211156

theorem max_value_fraction (x y : ℝ) (hx : -5 ≤ x ∧ x ≤ -3) (hy : 1 ≤ y ∧ y ≤ 3) :
  (x + y) / (x - 1) ≤ 2/3 := by
  sorry

end NUMINAMATH_CALUDE_max_value_fraction_l2111_211156


namespace NUMINAMATH_CALUDE_inequality_of_four_positive_reals_l2111_211135

theorem inequality_of_four_positive_reals (a b c d : ℝ) 
  (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c) (h_pos_d : 0 < d)
  (h_sum : a + b + c + d = 3) :
  1 / a^2 + 1 / b^2 + 1 / c^2 + 1 / d^2 ≤ 1 / (a * b * c * d)^2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_of_four_positive_reals_l2111_211135


namespace NUMINAMATH_CALUDE_clownfish_ratio_l2111_211120

/-- The aquarium scenario -/
structure Aquarium where
  total_fish : ℕ
  clownfish : ℕ
  blowfish : ℕ
  blowfish_in_own_tank : ℕ
  clownfish_in_display : ℕ
  (equal_fish : clownfish = blowfish)
  (total_sum : clownfish + blowfish = total_fish)
  (blowfish_display : blowfish - blowfish_in_own_tank = clownfish - clownfish_in_display)

/-- The theorem to prove -/
theorem clownfish_ratio (aq : Aquarium) 
  (h1 : aq.total_fish = 100)
  (h2 : aq.blowfish_in_own_tank = 26)
  (h3 : aq.clownfish_in_display = 16) :
  (aq.clownfish - aq.clownfish_in_display) / (aq.clownfish - aq.blowfish_in_own_tank) = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_clownfish_ratio_l2111_211120


namespace NUMINAMATH_CALUDE_larger_integer_value_l2111_211153

theorem larger_integer_value (a b : ℕ+) : 
  (a : ℚ) / (b : ℚ) = 3 / 2 → 
  (a : ℕ) * b = 216 → 
  max a b = 18 := by
sorry

end NUMINAMATH_CALUDE_larger_integer_value_l2111_211153


namespace NUMINAMATH_CALUDE_balanced_digraph_has_valid_coloring_l2111_211177

/-- A directed graph where each vertex has in-degree 2 and out-degree 2 -/
structure BalancedDigraph (V : Type) :=
  (edge : V → V → Prop)
  (in_degree_two : ∀ v, (∃ u w, u ≠ w ∧ edge u v ∧ edge w v) ∧ 
                        (∀ x y z, edge x v → edge y v → edge z v → (x = y ∨ x = z ∨ y = z)))
  (out_degree_two : ∀ v, (∃ u w, u ≠ w ∧ edge v u ∧ edge v w) ∧ 
                         (∀ x y z, edge v x → edge v y → edge v z → (x = y ∨ x = z ∨ y = z)))

/-- A valid coloring of edges in a balanced digraph -/
def ValidColoring (V : Type) (G : BalancedDigraph V) (color : V → V → Bool) : Prop :=
  ∀ v, (∃! u, G.edge v u ∧ color v u = true) ∧
       (∃! u, G.edge v u ∧ color v u = false) ∧
       (∃! u, G.edge u v ∧ color u v = true) ∧
       (∃! u, G.edge u v ∧ color u v = false)

/-- The main theorem: every balanced digraph has a valid coloring -/
theorem balanced_digraph_has_valid_coloring (V : Type) (G : BalancedDigraph V) :
  ∃ color : V → V → Bool, ValidColoring V G color := by
  sorry

end NUMINAMATH_CALUDE_balanced_digraph_has_valid_coloring_l2111_211177


namespace NUMINAMATH_CALUDE_largest_integer_with_remainder_l2111_211150

theorem largest_integer_with_remainder : ∃ n : ℕ, n < 100 ∧ n % 7 = 4 ∧ ∀ m : ℕ, m < 100 → m % 7 = 4 → m ≤ n := by
  sorry

end NUMINAMATH_CALUDE_largest_integer_with_remainder_l2111_211150


namespace NUMINAMATH_CALUDE_seven_eighths_of_64_l2111_211198

theorem seven_eighths_of_64 : (7 / 8 : ℚ) * 64 = 56 := by
  sorry

end NUMINAMATH_CALUDE_seven_eighths_of_64_l2111_211198


namespace NUMINAMATH_CALUDE_unique_solution_l2111_211196

/-- Represents the intersection point of two lines --/
structure IntersectionPoint where
  x : ℤ
  y : ℤ

/-- Checks if a given point satisfies both line equations --/
def is_valid_intersection (m : ℕ) (p : IntersectionPoint) : Prop :=
  13 * p.x + 11 * p.y = 700 ∧ p.y = m * p.x - 1

/-- Main theorem: m = 6 is the only solution --/
theorem unique_solution : 
  ∃! (m : ℕ), ∃ (p : IntersectionPoint), is_valid_intersection m p :=
sorry

end NUMINAMATH_CALUDE_unique_solution_l2111_211196


namespace NUMINAMATH_CALUDE_one_fifth_greater_than_decimal_l2111_211159

theorem one_fifth_greater_than_decimal : 1/5 = 0.20000001 + 1/(5*10^8) := by
  sorry

end NUMINAMATH_CALUDE_one_fifth_greater_than_decimal_l2111_211159


namespace NUMINAMATH_CALUDE_new_sequence_common_difference_l2111_211179

theorem new_sequence_common_difference 
  (a : ℕ → ℝ) 
  (d : ℝ) 
  (h : ∀ n : ℕ, a (n + 1) = a n + d) :
  let b : ℕ → ℝ := λ n => a n + a (n + 3)
  ∀ n : ℕ, b (n + 1) = b n + 2 * d :=
by sorry

end NUMINAMATH_CALUDE_new_sequence_common_difference_l2111_211179


namespace NUMINAMATH_CALUDE_min_perimeter_of_cross_sectional_triangle_l2111_211100

/-- Regular triangular pyramid with given dimensions -/
structure RegularTriangularPyramid where
  baseEdgeLength : ℝ
  lateralEdgeLength : ℝ

/-- Cross-sectional triangle in the pyramid -/
structure CrossSectionalTriangle (p : RegularTriangularPyramid) where
  intersectsLateralEdges : Bool

/-- The minimum perimeter of the cross-sectional triangle -/
def minPerimeter (p : RegularTriangularPyramid) (t : CrossSectionalTriangle p) : ℝ :=
  sorry

/-- Theorem: Minimum perimeter of cross-sectional triangle in given pyramid -/
theorem min_perimeter_of_cross_sectional_triangle 
  (p : RegularTriangularPyramid) 
  (t : CrossSectionalTriangle p)
  (h1 : p.baseEdgeLength = 4)
  (h2 : p.lateralEdgeLength = 8)
  (h3 : t.intersectsLateralEdges = true) :
  minPerimeter p t = 11 :=
sorry

end NUMINAMATH_CALUDE_min_perimeter_of_cross_sectional_triangle_l2111_211100


namespace NUMINAMATH_CALUDE_p_squared_plus_eight_composite_l2111_211164

theorem p_squared_plus_eight_composite (p : ℕ) (h_prime : Nat.Prime p) (h_not_three : p ≠ 3) :
  ¬(Nat.Prime (p^2 + 8)) := by
  sorry

end NUMINAMATH_CALUDE_p_squared_plus_eight_composite_l2111_211164


namespace NUMINAMATH_CALUDE_revenue_decrease_l2111_211138

theorem revenue_decrease (projected_increase : ℝ) (actual_vs_projected : ℝ) 
  (h1 : projected_increase = 0.20)
  (h2 : actual_vs_projected = 0.625) : 
  (1 - actual_vs_projected * (1 + projected_increase)) * 100 = 25 := by
  sorry

end NUMINAMATH_CALUDE_revenue_decrease_l2111_211138


namespace NUMINAMATH_CALUDE_tan_negative_3900_degrees_l2111_211184

theorem tan_negative_3900_degrees : Real.tan ((-3900 : ℝ) * π / 180) = Real.sqrt 3 := by sorry

end NUMINAMATH_CALUDE_tan_negative_3900_degrees_l2111_211184


namespace NUMINAMATH_CALUDE_perpendicular_vectors_l2111_211178

/-- Given vectors a and b in ℝ², prove that if (a + kb) ⊥ (a - kb), then k = ±√5 -/
theorem perpendicular_vectors (a b : ℝ × ℝ) (k : ℝ) 
  (h1 : a = (2, -1))
  (h2 : b = (-Real.sqrt 3 / 2, -1 / 2))
  (h3 : (a.1 + k * b.1, a.2 + k * b.2) • (a.1 - k * b.1, a.2 - k * b.2) = 0) :
  k = Real.sqrt 5 ∨ k = -Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_vectors_l2111_211178


namespace NUMINAMATH_CALUDE_power_of_two_equation_l2111_211117

theorem power_of_two_equation (N : ℕ) : (32^5 * 16^4) / 8^7 = 2^N → N = 20 := by
  sorry

end NUMINAMATH_CALUDE_power_of_two_equation_l2111_211117


namespace NUMINAMATH_CALUDE_f_properties_l2111_211109

noncomputable def f (x : ℝ) : ℝ := (Real.log x) / x

theorem f_properties :
  let max_value : ℝ := 1 / Real.exp 1
  ∀ (x₁ x₂ x₀ m : ℝ),
  (∀ x > 0, f x = (Real.log x) / x) →
  (∀ x > 0, f x ≤ max_value) →
  (f (Real.exp 1) = max_value) →
  (∀ x ∈ Set.Ioo 0 (Real.exp 1), f (Real.exp 1 + x) > f (Real.exp 1 - x)) →
  (f x₁ = m) →
  (f x₂ = m) →
  (x₀ = (x₁ + x₂) / 2) →
  (deriv f x₀ < 0) :=
by sorry

end NUMINAMATH_CALUDE_f_properties_l2111_211109


namespace NUMINAMATH_CALUDE_cat_cleaner_amount_l2111_211175

/-- The amount of cleaner used for a dog stain in ounces -/
def dog_cleaner : ℝ := 6

/-- The amount of cleaner used for a rabbit stain in ounces -/
def rabbit_cleaner : ℝ := 1

/-- The total amount of cleaner used for all stains in ounces -/
def total_cleaner : ℝ := 49

/-- The number of dogs -/
def num_dogs : ℕ := 6

/-- The number of cats -/
def num_cats : ℕ := 3

/-- The number of rabbits -/
def num_rabbits : ℕ := 1

/-- The amount of cleaner used for a cat stain in ounces -/
def cat_cleaner : ℝ := 4

theorem cat_cleaner_amount :
  dog_cleaner * num_dogs + cat_cleaner * num_cats + rabbit_cleaner * num_rabbits = total_cleaner :=
by sorry

end NUMINAMATH_CALUDE_cat_cleaner_amount_l2111_211175


namespace NUMINAMATH_CALUDE_average_weight_of_four_friends_l2111_211148

/-- The average weight of four friends given their relative weights -/
theorem average_weight_of_four_friends 
  (jalen_weight : ℝ)
  (ponce_weight : ℝ)
  (ishmael_weight : ℝ)
  (mike_weight : ℝ)
  (h1 : jalen_weight = 160)
  (h2 : ponce_weight = jalen_weight - 10)
  (h3 : ishmael_weight = ponce_weight + 20)
  (h4 : mike_weight = ishmael_weight + ponce_weight + jalen_weight - 15) :
  (jalen_weight + ponce_weight + ishmael_weight + mike_weight) / 4 = 236.25 := by
  sorry

end NUMINAMATH_CALUDE_average_weight_of_four_friends_l2111_211148


namespace NUMINAMATH_CALUDE_total_savings_calculation_l2111_211113

def chlorine_price : ℝ := 10
def soap_price : ℝ := 16
def wipes_price : ℝ := 8

def chlorine_discount1 : ℝ := 0.20
def chlorine_discount2 : ℝ := 0.10
def chlorine_discount3 : ℝ := 0.05

def soap_discount1 : ℝ := 0.25
def soap_discount2 : ℝ := 0.05

def wipes_discount1 : ℝ := 0.30
def wipes_discount2 : ℝ := 0.15
def wipes_discount3 : ℝ := 0.20

def chlorine_quantity : ℕ := 4
def soap_quantity : ℕ := 6
def wipes_quantity : ℕ := 8

theorem total_savings_calculation :
  let chlorine_final_price := chlorine_price * (1 - chlorine_discount1) * (1 - chlorine_discount2) * (1 - chlorine_discount3)
  let soap_final_price := soap_price * (1 - soap_discount1) * (1 - soap_discount2)
  let wipes_final_price := wipes_price * (1 - wipes_discount1) * (1 - wipes_discount2) * (1 - wipes_discount3)
  let chlorine_savings := (chlorine_price - chlorine_final_price) * chlorine_quantity
  let soap_savings := (soap_price - soap_final_price) * soap_quantity
  let wipes_savings := (wipes_price - wipes_final_price) * wipes_quantity
  chlorine_savings + soap_savings + wipes_savings = 73.776 := by sorry

end NUMINAMATH_CALUDE_total_savings_calculation_l2111_211113


namespace NUMINAMATH_CALUDE_min_value_neg_half_l2111_211110

/-- A function f with specific properties -/
def f (a b : ℝ) (x : ℝ) : ℝ := a * x^3 + b * x^9 + 2

/-- The maximum value of f on (0, +∞) -/
def max_value : ℝ := 5

/-- Theorem: The minimum value of f on (-∞, 0) is -1 -/
theorem min_value_neg_half (a b : ℝ) :
  (∀ x > 0, f a b x ≤ max_value) →
  (∃ x > 0, f a b x = max_value) →
  (∀ x < 0, f a b x ≥ -1) ∧
  (∃ x < 0, f a b x = -1) :=
sorry

end NUMINAMATH_CALUDE_min_value_neg_half_l2111_211110


namespace NUMINAMATH_CALUDE_base4_representation_has_four_digits_l2111_211136

/-- Converts a natural number from decimal to base 4 representation -/
def toBase4 (n : ℕ) : List ℕ :=
  if n = 0 then [0] else
  let rec aux (m : ℕ) (acc : List ℕ) : List ℕ :=
    if m = 0 then acc else aux (m / 4) ((m % 4) :: acc)
  aux n []

/-- The decimal number to be converted -/
def decimalNumber : ℕ := 75

/-- Theorem stating that the base 4 representation of 75 has four digits -/
theorem base4_representation_has_four_digits :
  (toBase4 decimalNumber).length = 4 := by
  sorry

end NUMINAMATH_CALUDE_base4_representation_has_four_digits_l2111_211136


namespace NUMINAMATH_CALUDE_balls_after_2010_steps_l2111_211101

/-- Converts a natural number to its base-6 representation --/
def toBase6 (n : ℕ) : List ℕ :=
  if n = 0 then [0]
  else
    let rec aux (m : ℕ) (acc : List ℕ) : List ℕ :=
      if m = 0 then acc
      else aux (m / 6) ((m % 6) :: acc)
    aux n []

/-- Sums the digits in a list --/
def sumDigits (digits : List ℕ) : ℕ :=
  digits.foldl (· + ·) 0

/-- Represents the ball-and-box process --/
def ballBoxProcess (steps : ℕ) : ℕ :=
  sumDigits (toBase6 steps)

/-- Theorem stating that the number of balls after 2010 steps
    is equal to the sum of digits in the base-6 representation of 2010 --/
theorem balls_after_2010_steps :
  ballBoxProcess 2010 = 11 := by sorry

end NUMINAMATH_CALUDE_balls_after_2010_steps_l2111_211101


namespace NUMINAMATH_CALUDE_kenny_trumpet_practice_l2111_211111

def basketball_hours : ℕ := 10

def running_hours (b : ℕ) : ℕ := 2 * b

def trumpet_hours (r : ℕ) : ℕ := 2 * r

def total_practice_hours (b r t : ℕ) : ℕ := b + r + t

theorem kenny_trumpet_practice (x y : ℕ) :
  let b := basketball_hours
  let r := running_hours b
  let t := trumpet_hours r
  total_practice_hours b r t = x + y →
  t = 40 := by
sorry

end NUMINAMATH_CALUDE_kenny_trumpet_practice_l2111_211111


namespace NUMINAMATH_CALUDE_recipe_flour_amount_l2111_211166

/-- Represents the recipe and Mary's baking progress -/
structure Recipe :=
  (total_sugar : ℕ)
  (flour_added : ℕ)
  (sugar_added : ℕ)
  (sugar_to_add : ℕ)

/-- The amount of flour required is independent of the amount of sugar -/
axiom flour_independent_of_sugar (r : Recipe) : 
  r.flour_added = r.flour_added

/-- Theorem: The recipe calls for 10 cups of flour -/
theorem recipe_flour_amount (r : Recipe) 
  (h1 : r.total_sugar = 14)
  (h2 : r.flour_added = 10)
  (h3 : r.sugar_added = 2)
  (h4 : r.sugar_to_add = 12) :
  r.flour_added = 10 := by
  sorry

end NUMINAMATH_CALUDE_recipe_flour_amount_l2111_211166


namespace NUMINAMATH_CALUDE_height_comparison_l2111_211168

theorem height_comparison (p q : ℝ) (h : p = 0.6 * q) :
  (q - p) / p = 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_height_comparison_l2111_211168


namespace NUMINAMATH_CALUDE_three_heads_in_eight_tosses_l2111_211144

/-- The probability of getting exactly k heads in n tosses of a fair coin -/
def probability_k_heads (n k : ℕ) : ℚ :=
  (Nat.choose n k : ℚ) / 2^n

/-- The probability of getting exactly 3 heads in 8 tosses of a fair coin is 7/32 -/
theorem three_heads_in_eight_tosses :
  probability_k_heads 8 3 = 7 / 32 := by
  sorry

end NUMINAMATH_CALUDE_three_heads_in_eight_tosses_l2111_211144


namespace NUMINAMATH_CALUDE_sandys_savings_ratio_l2111_211129

/-- The ratio of Sandy's savings this year to last year -/
theorem sandys_savings_ratio (S1 D1 : ℝ) (S1_pos : 0 < S1) (D1_pos : 0 < D1) :
  let Y := 0.06 * S1 + 0.08 * D1
  let X := 0.099 * S1 + 0.126 * D1
  X / Y = (0.099 + 0.126) / (0.06 + 0.08) := by
  sorry

end NUMINAMATH_CALUDE_sandys_savings_ratio_l2111_211129


namespace NUMINAMATH_CALUDE_f_composition_equals_one_fourth_l2111_211126

noncomputable def f (x : ℝ) : ℝ :=
  if x > 0 then Real.log x / Real.log 3
  else 2^x

theorem f_composition_equals_one_fourth :
  f (f (1/9)) = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_f_composition_equals_one_fourth_l2111_211126


namespace NUMINAMATH_CALUDE_cPass_max_entries_aPass_cost_effective_l2111_211176

-- Define the ticketing options
structure TicketOption where
  initialCost : ℕ
  entryCost : ℕ

-- Define the budget
def budget : ℕ := 80

-- Define the ticketing options
def noPass : TicketOption := ⟨0, 10⟩
def aPass : TicketOption := ⟨120, 0⟩
def bPass : TicketOption := ⟨60, 2⟩
def cPass : TicketOption := ⟨40, 3⟩

-- Function to calculate the number of entries for a given option and budget
def numEntries (option : TicketOption) (budget : ℕ) : ℕ :=
  if option.initialCost > budget then 0
  else (budget - option.initialCost) / option.entryCost

-- Theorem 1: C pass allows for the maximum number of entries with 80 yuan budget
theorem cPass_max_entries :
  ∀ option : TicketOption, numEntries cPass budget ≥ numEntries option budget :=
sorry

-- Theorem 2: A pass becomes more cost-effective when entering more than 30 times
theorem aPass_cost_effective (n : ℕ) (h : n > 30) :
  ∀ option : TicketOption, option.initialCost + n * option.entryCost > aPass.initialCost :=
sorry

end NUMINAMATH_CALUDE_cPass_max_entries_aPass_cost_effective_l2111_211176


namespace NUMINAMATH_CALUDE_max_sum_on_circle_l2111_211186

theorem max_sum_on_circle (x y : ℤ) (h : x^2 + y^2 = 169) : x + y ≤ 17 :=
sorry

end NUMINAMATH_CALUDE_max_sum_on_circle_l2111_211186


namespace NUMINAMATH_CALUDE_john_walked_four_miles_l2111_211195

/-- Represents the distance John traveled in miles -/
structure JohnTravel where
  initial_skate : ℝ
  total_skate : ℝ
  walk : ℝ

/-- The conditions of John's travel -/
def travel_conditions (j : JohnTravel) : Prop :=
  j.initial_skate = 10 ∧ 
  j.total_skate = 24 ∧
  j.total_skate = 2 * j.initial_skate + j.walk

/-- Theorem stating that John walked 4 miles to the park -/
theorem john_walked_four_miles (j : JohnTravel) 
  (h : travel_conditions j) : j.walk = 4 := by
  sorry

end NUMINAMATH_CALUDE_john_walked_four_miles_l2111_211195


namespace NUMINAMATH_CALUDE_acrobats_count_l2111_211158

/-- The number of acrobats in a parade group -/
def num_acrobats : ℕ := 10

/-- The number of elephants in a parade group -/
def num_elephants : ℕ := 20 - num_acrobats

/-- The total number of legs in the parade group -/
def total_legs : ℕ := 60

/-- The total number of heads in the parade group -/
def total_heads : ℕ := 20

/-- Theorem stating that the number of acrobats is 10 given the conditions -/
theorem acrobats_count :
  (2 * num_acrobats + 4 * num_elephants = total_legs) ∧
  (num_acrobats + num_elephants = total_heads) ∧
  (num_acrobats = 10) := by
  sorry

end NUMINAMATH_CALUDE_acrobats_count_l2111_211158


namespace NUMINAMATH_CALUDE_probability_D_given_E_l2111_211133

-- Define the regions D and E
def region_D (x y : ℝ) : Prop :=
  0 ≤ x ∧ x ≤ 1 ∧ x + 1 ≤ y ∧ y ≤ 2

def region_E (x y : ℝ) : Prop :=
  -1 ≤ x ∧ x ≤ 1 ∧ 0 ≤ y ∧ y ≤ 2

-- Define the area function
def area (region : ℝ → ℝ → Prop) : ℝ := sorry

-- Theorem statement
theorem probability_D_given_E :
  (area region_D) / (area region_E) = 3/8 := by sorry

end NUMINAMATH_CALUDE_probability_D_given_E_l2111_211133


namespace NUMINAMATH_CALUDE_canteen_leak_rate_l2111_211191

/-- Proves that the canteen leak rate is 1 cup per hour given the hiking conditions -/
theorem canteen_leak_rate
  (total_distance : ℝ)
  (initial_water : ℝ)
  (hike_duration : ℝ)
  (remaining_water : ℝ)
  (last_mile_consumption : ℝ)
  (first_three_miles_rate : ℝ)
  (h1 : total_distance = 4)
  (h2 : initial_water = 6)
  (h3 : hike_duration = 2)
  (h4 : remaining_water = 1)
  (h5 : last_mile_consumption = 1)
  (h6 : first_three_miles_rate = 0.6666666666666666)
  : (initial_water - remaining_water - (first_three_miles_rate * 3 + last_mile_consumption)) / hike_duration = 1 := by
  sorry

#check canteen_leak_rate

end NUMINAMATH_CALUDE_canteen_leak_rate_l2111_211191


namespace NUMINAMATH_CALUDE_paths_via_checkpoint_count_l2111_211132

/-- Number of paths in a grid from (0,0) to (a,b) -/
def gridPaths (a b : ℕ) : ℕ := Nat.choose (a + b) a

/-- The coordinates of point A -/
def A : ℕ × ℕ := (0, 0)

/-- The coordinates of point B -/
def B : ℕ × ℕ := (5, 4)

/-- The coordinates of checkpoint C -/
def C : ℕ × ℕ := (3, 2)

/-- The number of paths from A to B via C -/
def pathsViaCPoint : ℕ := 
  (gridPaths (C.1 - A.1) (C.2 - A.2)) * (gridPaths (B.1 - C.1) (B.2 - C.2))

theorem paths_via_checkpoint_count : pathsViaCPoint = 60 := by sorry

end NUMINAMATH_CALUDE_paths_via_checkpoint_count_l2111_211132


namespace NUMINAMATH_CALUDE_five_digit_permutations_l2111_211161

/-- The number of permutations of a multiset with repeated elements -/
def multiset_permutations (total : ℕ) (repetitions : List ℕ) : ℕ :=
  Nat.factorial total / (repetitions.map Nat.factorial).prod

/-- The number of different five-digit integers formed using 1, 1, 1, 8, and 8 -/
theorem five_digit_permutations : multiset_permutations 5 [3, 2] = 10 := by
  sorry

end NUMINAMATH_CALUDE_five_digit_permutations_l2111_211161


namespace NUMINAMATH_CALUDE_x_gt_5_sufficient_not_necessary_for_x_sq_gt_25_l2111_211170

theorem x_gt_5_sufficient_not_necessary_for_x_sq_gt_25 :
  (∀ x : ℝ, x > 5 → x^2 > 25) ∧
  (∃ x : ℝ, x^2 > 25 ∧ x ≤ 5) := by
  sorry

end NUMINAMATH_CALUDE_x_gt_5_sufficient_not_necessary_for_x_sq_gt_25_l2111_211170


namespace NUMINAMATH_CALUDE_train_speed_l2111_211140

/-- The speed of a train crossing a bridge -/
theorem train_speed (train_length bridge_length : ℝ) (crossing_time : ℝ) :
  train_length = 170 →
  bridge_length = 205 →
  crossing_time = 30 →
  (train_length + bridge_length) / crossing_time * 3.6 = 45 := by
  sorry


end NUMINAMATH_CALUDE_train_speed_l2111_211140


namespace NUMINAMATH_CALUDE_count_rectangles_3x3_grid_l2111_211151

/-- The number of rectangles that can be formed on a 3x3 grid -/
def rectangles_on_3x3_grid : ℕ := 9

/-- Theorem stating that the number of rectangles on a 3x3 grid is 9 -/
theorem count_rectangles_3x3_grid : 
  rectangles_on_3x3_grid = 9 := by sorry

end NUMINAMATH_CALUDE_count_rectangles_3x3_grid_l2111_211151


namespace NUMINAMATH_CALUDE_bob_total_candies_l2111_211152

/-- Calculates Bob's share of candies given the total amount and the ratio --/
def bobShare (total : ℕ) (samRatio : ℕ) (bobRatio : ℕ) : ℕ :=
  (total * bobRatio) / (samRatio + bobRatio)

/-- Theorem: Bob receives 64 candies in total --/
theorem bob_total_candies : 
  let chewingGums := bobShare 45 2 3
  let chocolateBars := bobShare 60 3 1
  let assortedCandies := 45 / 2
  chewingGums + chocolateBars + assortedCandies = 64 := by
  sorry

#eval bobShare 45 2 3 -- Should output 27
#eval bobShare 60 3 1 -- Should output 15
#eval 45 / 2          -- Should output 22

end NUMINAMATH_CALUDE_bob_total_candies_l2111_211152


namespace NUMINAMATH_CALUDE_quadratic_roots_sum_product_l2111_211131

theorem quadratic_roots_sum_product (p q : ℝ) (k : ℕ+) :
  (∃ x y : ℝ, x^2 + p*x + q = 0 ∧ y^2 + p*y + q = 0) →
  (∃ x y : ℝ, x^2 + p*x + q = 0 ∧ y^2 + p*y + q = 0 ∧ x + y = 2) →
  (∃ x y : ℝ, x^2 + p*x + q = 0 ∧ y^2 + p*y + q = 0 ∧ x * y = k) →
  p = -2 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_roots_sum_product_l2111_211131


namespace NUMINAMATH_CALUDE_pages_to_read_in_third_week_l2111_211145

theorem pages_to_read_in_third_week 
  (total_pages : ℕ) 
  (first_week_fraction : ℚ) 
  (second_week_percent : ℚ) 
  (h1 : total_pages = 600)
  (h2 : first_week_fraction = 1/2)
  (h3 : second_week_percent = 30/100) :
  total_pages - 
  (first_week_fraction * total_pages).floor - 
  (second_week_percent * (total_pages - (first_week_fraction * total_pages).floor)).floor = 210 :=
by
  sorry

end NUMINAMATH_CALUDE_pages_to_read_in_third_week_l2111_211145


namespace NUMINAMATH_CALUDE_sum_of_four_numbers_l2111_211193

theorem sum_of_four_numbers : 1357 + 3571 + 5713 + 7135 = 17776 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_four_numbers_l2111_211193


namespace NUMINAMATH_CALUDE_inequality_proof_l2111_211162

theorem inequality_proof (x y : ℝ) : 5 * x^2 + y^2 + 4 - 4 * x - 4 * x * y ≥ 0 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2111_211162


namespace NUMINAMATH_CALUDE_curve_properties_l2111_211154

/-- The curve function -/
def curve (c : ℝ) (x : ℝ) : ℝ := c * x^4 + x^2 - c

theorem curve_properties :
  ∀ (c : ℝ),
  -- The points (1, 1) and (-1, 1) lie on the curve for all values of c
  curve c 1 = 1 ∧ curve c (-1) = 1 ∧
  -- When c = -1/4, the curve is tangent to the line y = x at the point (1, 1)
  (let c := -1/4
   curve c 1 = 1 ∧ (deriv (curve c)) 1 = 1) ∧
  -- The curve intersects the line y = x at the points (1, 1) and (-1 + √2, -1 + √2)
  (∃ (x : ℝ), x ≠ 1 ∧ curve (-1/4) x = x ∧ x = -1 + Real.sqrt 2) :=
by sorry

end NUMINAMATH_CALUDE_curve_properties_l2111_211154


namespace NUMINAMATH_CALUDE_equation_solution_l2111_211197

/-- The overall substitution method for solving quadratic equations -/
def overall_substitution_method (a b c : ℝ) : Set ℝ :=
  { x | ∃ y, y^2 + b*y + c = 0 ∧ a*x + b = y }

/-- The equation (2x-5)^2 - 2(2x-5) - 3 = 0 has solutions x₁ = 2 and x₂ = 4 -/
theorem equation_solution : 
  overall_substitution_method 2 (-5) (-3) = {2, 4} := by
  sorry

#check equation_solution

end NUMINAMATH_CALUDE_equation_solution_l2111_211197


namespace NUMINAMATH_CALUDE_line_l_equation_line_l_prime_equation_l2111_211160

-- Define the lines l₁ and l₂
def l₁ (x y : ℝ) : Prop := 3 * x + 4 * y - 5 = 0
def l₂ (x y : ℝ) : Prop := 2 * x - 3 * y + 8 = 0

-- Define the perpendicular line
def perp_line (x y : ℝ) : Prop := 2 * x + y + 2 = 0

-- Define the point of symmetry
def sym_point : ℝ × ℝ := (1, -1)

-- Theorem for the equation of line l
theorem line_l_equation :
  ∃ (m : ℝ), ∀ (x y : ℝ),
    (∃ (x₀ y₀ : ℝ), l₁ x₀ y₀ ∧ l₂ x₀ y₀ ∧ x - 2 * y + m = 0) ∧
    (∀ (x₁ y₁ : ℝ), perp_line x₁ y₁ → (x - x₁) * 2 + (y - y₁) * 1 = 0) →
    x - 2 * y + 5 = 0 :=
sorry

-- Theorem for the equation of line l'
theorem line_l_prime_equation :
  ∀ (x y : ℝ),
    (∃ (x₀ y₀ : ℝ), l₁ x₀ y₀ ∧ 
      x₀ = 2 * sym_point.1 - x ∧
      y₀ = 2 * sym_point.2 - y) →
    3 * x + 4 * y + 7 = 0 :=
sorry

end NUMINAMATH_CALUDE_line_l_equation_line_l_prime_equation_l2111_211160


namespace NUMINAMATH_CALUDE_sqrt_eight_same_type_as_sqrt_two_l2111_211108

/-- Two real numbers are of the same type if one is a rational multiple of the other -/
def same_type (a b : ℝ) : Prop := ∃ q : ℚ, a = q * b

/-- √2 is a real number -/
axiom sqrt_two : ℝ

/-- √8 is a real number -/
axiom sqrt_eight : ℝ

/-- The statement to be proved -/
theorem sqrt_eight_same_type_as_sqrt_two : same_type sqrt_eight sqrt_two := by sorry

end NUMINAMATH_CALUDE_sqrt_eight_same_type_as_sqrt_two_l2111_211108


namespace NUMINAMATH_CALUDE_water_transfer_equilibrium_l2111_211119

theorem water_transfer_equilibrium (total : ℕ) (a b : ℕ) : 
  total = 48 →
  a = 30 →
  b = 18 →
  a + b = total →
  let a' := a - 2 * a
  let b' := b + 2 * a
  let a'' := a' + 2 * a'
  let b'' := b' - 2 * a'
  a'' = b'' := by sorry

end NUMINAMATH_CALUDE_water_transfer_equilibrium_l2111_211119


namespace NUMINAMATH_CALUDE_no_integer_roots_l2111_211146

theorem no_integer_roots (a b : ℤ) : ¬ ∃ x : ℤ, x^2 + 3*a*x + 3*(2 - b^2) = 0 := by
  sorry

end NUMINAMATH_CALUDE_no_integer_roots_l2111_211146


namespace NUMINAMATH_CALUDE_square_difference_equality_l2111_211123

theorem square_difference_equality : 1013^2 - 1009^2 - 1011^2 + 997^2 = -19924 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_equality_l2111_211123


namespace NUMINAMATH_CALUDE_largest_mersenne_prime_under_500_l2111_211114

def is_mersenne_prime (m : ℕ) : Prop :=
  ∃ n : ℕ, Prime n ∧ m = 2^n - 1 ∧ Prime m

theorem largest_mersenne_prime_under_500 :
  (∀ m : ℕ, is_mersenne_prime m ∧ m < 500 → m ≤ 127) ∧
  is_mersenne_prime 127 ∧
  127 < 500 :=
sorry

end NUMINAMATH_CALUDE_largest_mersenne_prime_under_500_l2111_211114


namespace NUMINAMATH_CALUDE_fence_cost_for_square_plot_l2111_211185

theorem fence_cost_for_square_plot (area : ℝ) (price_per_foot : ℝ) :
  area = 289 →
  price_per_foot = 58 →
  (4 * Real.sqrt area) * price_per_foot = 3944 := by
  sorry

end NUMINAMATH_CALUDE_fence_cost_for_square_plot_l2111_211185


namespace NUMINAMATH_CALUDE_perfect_square_condition_l2111_211112

theorem perfect_square_condition (a b : ℤ) :
  (∀ m n : ℕ, ∃ k : ℤ, a * m^2 + b * n^2 = k^2) →
  a * b = 0 := by
sorry

end NUMINAMATH_CALUDE_perfect_square_condition_l2111_211112


namespace NUMINAMATH_CALUDE_subset_implies_range_l2111_211125

-- Define the sets N and M
def N (a : ℝ) : Set ℝ := {x | a + 1 ≤ x ∧ x < 2 * a - 1}
def M : Set ℝ := {x | -2 ≤ x ∧ x ≤ 5}

-- State the theorem
theorem subset_implies_range (a : ℝ) : N a ⊆ M → a ≤ 3 := by
  sorry

end NUMINAMATH_CALUDE_subset_implies_range_l2111_211125


namespace NUMINAMATH_CALUDE_mary_payment_l2111_211157

def apple_cost : ℕ := 1
def orange_cost : ℕ := 2
def banana_cost : ℕ := 3
def discount_threshold : ℕ := 5
def discount_amount : ℕ := 1

def mary_apples : ℕ := 5
def mary_oranges : ℕ := 3
def mary_bananas : ℕ := 2

def total_fruits : ℕ := mary_apples + mary_oranges + mary_bananas

def fruit_cost : ℕ := mary_apples * apple_cost + mary_oranges * orange_cost + mary_bananas * banana_cost

def discount_sets : ℕ := total_fruits / discount_threshold

def total_discount : ℕ := discount_sets * discount_amount

def final_cost : ℕ := fruit_cost - total_discount

theorem mary_payment : final_cost = 15 := by
  sorry

end NUMINAMATH_CALUDE_mary_payment_l2111_211157


namespace NUMINAMATH_CALUDE_median_length_half_side_l2111_211139

/-- Prove that the length of a median in a triangle is half the length of its corresponding side. -/
theorem median_length_half_side {A B C : ℝ × ℝ} : 
  let M := ((B.1 + C.1) / 2, (B.2 + C.2) / 2)  -- Midpoint of BC
  dist A M = (1/2) * dist B C := by
  sorry

end NUMINAMATH_CALUDE_median_length_half_side_l2111_211139


namespace NUMINAMATH_CALUDE_rowing_distance_problem_l2111_211199

/-- Proves that given a man who can row 7.5 km/hr in still water, in a river flowing at 1.5 km/hr,
    if it takes him 50 minutes to row to a place and back, the distance to that place is 3 km. -/
theorem rowing_distance_problem (man_speed : ℝ) (river_speed : ℝ) (total_time : ℝ) :
  man_speed = 7.5 →
  river_speed = 1.5 →
  total_time = 50 / 60 →
  ∃ (distance : ℝ),
    distance / (man_speed - river_speed) + distance / (man_speed + river_speed) = total_time ∧
    distance = 3 :=
by sorry

end NUMINAMATH_CALUDE_rowing_distance_problem_l2111_211199


namespace NUMINAMATH_CALUDE_problem_1_problem_2_l2111_211107

-- Problem 1
theorem problem_1 (a : ℝ) (h : a = -1) : 
  (1 : ℝ) * (a + 3)^2 + (3 + a) * (3 - a) = 12 := by sorry

-- Problem 2
theorem problem_2 (x y : ℝ) (hx : x = 2) (hy : y = 3) : 
  (x - 2*y) * (x + 2*y) - (x + 2*y)^2 + 8*y^2 = -24 := by sorry

end NUMINAMATH_CALUDE_problem_1_problem_2_l2111_211107


namespace NUMINAMATH_CALUDE_point_line_distance_constraint_l2111_211122

/-- Given a point P(4, a) and a line 4x - 3y - 1 = 0, if the distance from P to the line
    is no greater than 3, then a is in the range [0, 10]. -/
theorem point_line_distance_constraint (a : ℝ) : 
  let P : ℝ × ℝ := (4, a)
  let line (x y : ℝ) : Prop := 4 * x - 3 * y - 1 = 0
  let distance := |4 * 4 - 3 * a - 1| / 5
  distance ≤ 3 → 0 ≤ a ∧ a ≤ 10 := by
sorry

end NUMINAMATH_CALUDE_point_line_distance_constraint_l2111_211122


namespace NUMINAMATH_CALUDE_solution_range_l2111_211172

theorem solution_range (x : ℝ) :
  (5 * x - 8 > 12 - 2 * x) ∧ (|x - 1| ≤ 3) → (20 / 7 < x ∧ x ≤ 4) :=
by sorry

end NUMINAMATH_CALUDE_solution_range_l2111_211172


namespace NUMINAMATH_CALUDE_percentage_problem_l2111_211121

theorem percentage_problem (P : ℝ) : 
  (P / 100) * 16 = 40 → P = 250 := by
  sorry

end NUMINAMATH_CALUDE_percentage_problem_l2111_211121


namespace NUMINAMATH_CALUDE_largest_prime_less_than_5000_l2111_211194

def is_prime (p : Nat) : Prop :=
  p > 1 ∧ ∀ d : Nat, d > 1 → d < p → ¬(p % d = 0)

def is_of_form (p : Nat) : Prop :=
  ∃ (a n : Nat), a > 0 ∧ n > 1 ∧ p = a^n - 1

theorem largest_prime_less_than_5000 :
  ∀ p : Nat, p < 5000 → is_prime p → is_of_form p →
  p ≤ 127 :=
sorry

end NUMINAMATH_CALUDE_largest_prime_less_than_5000_l2111_211194


namespace NUMINAMATH_CALUDE_cube_root_cube_identity_l2111_211187

theorem cube_root_cube_identity (x : ℝ) : (x^3)^(1/3) = x := by
  sorry

end NUMINAMATH_CALUDE_cube_root_cube_identity_l2111_211187


namespace NUMINAMATH_CALUDE_not_center_of_symmetry_l2111_211147

/-- The tangent function -/
noncomputable def tan (x : ℝ) : ℝ := sorry

/-- The function y = tan(2x - π/4) -/
noncomputable def f (x : ℝ) : ℝ := tan (2 * x - Real.pi / 4)

/-- A point is a center of symmetry if it has the form (kπ/4 + π/8, 0) for some integer k -/
def is_center_of_symmetry (p : ℝ × ℝ) : Prop :=
  ∃ k : ℤ, p.1 = k * Real.pi / 4 + Real.pi / 8 ∧ p.2 = 0

/-- The statement to be proved -/
theorem not_center_of_symmetry :
  ¬ is_center_of_symmetry (Real.pi / 4, 0) :=
sorry

end NUMINAMATH_CALUDE_not_center_of_symmetry_l2111_211147


namespace NUMINAMATH_CALUDE_inequality_and_equality_condition_l2111_211142

theorem inequality_and_equality_condition (a b c d : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) :
  let expr := (a-b)*(a-c)/(a+b+c) + (b-c)*(b-d)/(b+c+d) + 
               (c-d)*(c-a)/(c+d+a) + (d-a)*(d-b)/(d+a+b)
  (expr ≥ 0) ∧ 
  (expr = 0 ↔ a = c ∧ b = d) :=
by sorry

end NUMINAMATH_CALUDE_inequality_and_equality_condition_l2111_211142


namespace NUMINAMATH_CALUDE_magnitude_n_equals_five_l2111_211141

/-- Given two vectors m and n in ℝ², prove that |n| = 5 -/
theorem magnitude_n_equals_five (m n : ℝ × ℝ) 
  (h1 : m.1 * n.1 + m.2 * n.2 = 0)  -- m is perpendicular to n
  (h2 : (m.1 - 2 * n.1, m.2 - 2 * n.2) = (11, -2))  -- m - 2n = (11, -2)
  (h3 : Real.sqrt (m.1^2 + m.2^2) = 5)  -- |m| = 5
  : Real.sqrt (n.1^2 + n.2^2) = 5 := by
  sorry

end NUMINAMATH_CALUDE_magnitude_n_equals_five_l2111_211141


namespace NUMINAMATH_CALUDE_inequality_proof_l2111_211188

theorem inequality_proof (a b c : ℝ) 
  (h1 : a ≥ 0) (h2 : b ≥ 0) (h3 : c ≥ 0)
  (h4 : a + b + c = 2 * Real.sqrt (a * b * c)) : 
  b * c ≥ b + c := by
sorry

end NUMINAMATH_CALUDE_inequality_proof_l2111_211188


namespace NUMINAMATH_CALUDE_solution_m_l2111_211127

theorem solution_m (x y m : ℝ) 
  (hx : x = 1) 
  (hy : y = 3) 
  (heq : 3 * m * x - 2 * y = 9) : m = 5 := by
  sorry

end NUMINAMATH_CALUDE_solution_m_l2111_211127


namespace NUMINAMATH_CALUDE_min_value_of_a_l2111_211173

theorem min_value_of_a (a x y : ℤ) : 
  x ≠ y →
  x - y^2 = a →
  y - x^2 = a →
  |x| ≤ 10 →
  (∀ b : ℤ, (∃ x' y' : ℤ, x' ≠ y' ∧ x' - y'^2 = b ∧ y' - x'^2 = b ∧ |x'| ≤ 10) → b ≥ a) →
  a = -111 :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_a_l2111_211173


namespace NUMINAMATH_CALUDE_problem_statement_l2111_211118

theorem problem_statement (x y : ℤ) (hx : x = 3) (hy : y = 2) : 3 * x - 4 * y = 1 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l2111_211118


namespace NUMINAMATH_CALUDE_exponent_calculations_l2111_211183

theorem exponent_calculations :
  (16 ^ (1/2 : ℝ) + (1/81 : ℝ) ^ (-1/4 : ℝ) - (-1/2 : ℝ) ^ (0 : ℝ) = 10/3) ∧
  (∀ (a b : ℝ), a ≠ 0 → b ≠ 0 → 
    ((2 * a ^ (1/4 : ℝ) * b ^ (-1/3 : ℝ)) * (-3 * a ^ (-1/2 : ℝ) * b ^ (2/3 : ℝ))) / 
    (-1/4 * a ^ (-1/4 : ℝ) * b ^ (-2/3 : ℝ)) = 24 * b) := by
  sorry

end NUMINAMATH_CALUDE_exponent_calculations_l2111_211183


namespace NUMINAMATH_CALUDE_min_reciprocal_sum_l2111_211163

theorem min_reciprocal_sum (x y z : ℝ) (hpos : x > 0 ∧ y > 0 ∧ z > 0) (hsum : x + y + z = 2) :
  (1/x + 1/y + 1/z) ≥ 4.5 ∧ ∃ (x' y' z' : ℝ), x' > 0 ∧ y' > 0 ∧ z' > 0 ∧ x' + y' + z' = 2 ∧ 1/x' + 1/y' + 1/z' = 4.5 := by
  sorry

end NUMINAMATH_CALUDE_min_reciprocal_sum_l2111_211163


namespace NUMINAMATH_CALUDE_train_speed_problem_l2111_211192

/-- The speed of train B given the conditions of the problem -/
theorem train_speed_problem (length_A length_B : ℝ) (speed_A : ℝ) (crossing_time : ℝ) :
  length_A = 150 ∧ 
  length_B = 150 ∧ 
  speed_A = 54 ∧ 
  crossing_time = 12 →
  (length_A + length_B) / crossing_time * 3.6 - speed_A = 36 := by
  sorry

#check train_speed_problem

end NUMINAMATH_CALUDE_train_speed_problem_l2111_211192


namespace NUMINAMATH_CALUDE_stating_probability_reroll_two_dice_l2111_211169

/-- Represents a fair six-sided die -/
def Die := Fin 6

/-- The sum we're aiming for -/
def targetSum : ℕ := 9

/-- The total number of possible outcomes when rolling three dice -/
def totalOutcomes : ℕ := 216

/-- 
Represents the optimal strategy for rerolling dice to achieve the target sum.
d1, d2, d3 are the values of the three dice.
Returns the number of dice to reroll (0, 1, 2, or 3).
-/
def optimalReroll (d1 d2 d3 : Die) : Fin 4 :=
  sorry

/-- 
The number of outcomes where rerolling exactly two dice is optimal.
-/
def twoRerollOutcomes : ℕ := 84

/-- 
Theorem stating that the probability of choosing to reroll exactly two dice
to optimize the chances of getting a sum of 9 is 7/18.
-/
theorem probability_reroll_two_dice :
  (twoRerollOutcomes : ℚ) / totalOutcomes = 7 / 18 := by
  sorry

end NUMINAMATH_CALUDE_stating_probability_reroll_two_dice_l2111_211169


namespace NUMINAMATH_CALUDE_triangle_properties_l2111_211116

-- Define a triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the theorem
theorem triangle_properties (t : Triangle) 
  (h1 : t.a^2 + t.c^2 - t.b^2 = t.a * t.c) 
  (h2 : t.c = 3 * t.a) : 
  t.B = π/3 ∧ Real.sin t.A = Real.sqrt 21 / 14 := by
  sorry

end NUMINAMATH_CALUDE_triangle_properties_l2111_211116


namespace NUMINAMATH_CALUDE_diagonal_length_is_13_l2111_211167

/-- Represents an isosceles trapezoid with given dimensions -/
structure IsoscelesTrapezoid where
  AB : ℝ  -- Length of the longer parallel side
  CD : ℝ  -- Length of the shorter parallel side
  AD : ℝ  -- Length of a leg (equal to BC in an isosceles trapezoid)

/-- The diagonal length of the isosceles trapezoid -/
def diagonal_length (t : IsoscelesTrapezoid) : ℝ := 
  sorry

/-- Theorem stating that for the given trapezoid dimensions, the diagonal length is 13 -/
theorem diagonal_length_is_13 :
  let t : IsoscelesTrapezoid := { AB := 24, CD := 10, AD := 13 }
  diagonal_length t = 13 := by
  sorry

end NUMINAMATH_CALUDE_diagonal_length_is_13_l2111_211167


namespace NUMINAMATH_CALUDE_nth_derivative_reciprocal_polynomial_l2111_211143

theorem nth_derivative_reciprocal_polynomial (k n : ℕ) (h : k > 0) :
  let f : ℝ → ℝ := λ x => 1 / (x^k - 1)
  let nth_derivative := (deriv^[n] f)
  ∃ P : ℝ → ℝ, (∀ x, nth_derivative x = P x / (x^k - 1)^(n + 1)) ∧
                P 1 = (-1)^n * n.factorial * k^n :=
by
  sorry

end NUMINAMATH_CALUDE_nth_derivative_reciprocal_polynomial_l2111_211143


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l2111_211165

-- Define set A
def A : Set ℝ := {x | x^2 - 2*x - 3 ≤ 0}

-- Define set B
def B : Set ℝ := {x | ∃ y, y = Real.log (2 - x)}

-- Theorem statement
theorem intersection_of_A_and_B : A ∩ B = {x | -1 ≤ x ∧ x < 2} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l2111_211165


namespace NUMINAMATH_CALUDE_charity_share_is_75_l2111_211104

-- Define the quantities of each baked good (in dozens)
def cookie_dozens : ℕ := 6
def brownie_dozens : ℕ := 4
def muffin_dozens : ℕ := 3

-- Define the selling prices (in dollars)
def cookie_price : ℚ := 3/2
def brownie_price : ℚ := 2
def muffin_price : ℚ := 5/2

-- Define the costs to make each item (in dollars)
def cookie_cost : ℚ := 1/4
def brownie_cost : ℚ := 1/2
def muffin_cost : ℚ := 3/4

-- Define the number of charities
def num_charities : ℕ := 3

-- Define a function to calculate the profit for each type of baked good
def profit_per_type (dozens : ℕ) (price : ℚ) (cost : ℚ) : ℚ :=
  (dozens * 12 : ℚ) * (price - cost)

-- Define the total profit
def total_profit : ℚ :=
  profit_per_type cookie_dozens cookie_price cookie_cost +
  profit_per_type brownie_dozens brownie_price brownie_cost +
  profit_per_type muffin_dozens muffin_price muffin_cost

-- Theorem to prove
theorem charity_share_is_75 :
  total_profit / num_charities = 75 := by sorry

end NUMINAMATH_CALUDE_charity_share_is_75_l2111_211104
