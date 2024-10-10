import Mathlib

namespace basketball_passes_l3135_313574

/-- Represents the number of ways the ball can be with player A after n moves -/
def ball_with_A (n : ℕ) : ℕ :=
  if n = 0 then 0
  else if n = 1 then 1
  else 2 * ball_with_A (n - 1) + 3 * ball_with_A (n - 2)

/-- The problem statement -/
theorem basketball_passes :
  ball_with_A 7 = 1094 := by
  sorry


end basketball_passes_l3135_313574


namespace largest_n_satisfying_conditions_l3135_313500

theorem largest_n_satisfying_conditions : ∃ (n : ℕ), n = 50 ∧ 
  (∀ m : ℕ, n^2 = (m+1)^3 - m^3 → m ≤ 50) ∧
  (∃ k : ℕ, 2*n + 99 = k^2) ∧
  (∀ n' : ℕ, n' > n → 
    (¬∃ m : ℕ, n'^2 = (m+1)^3 - m^3) ∨ 
    (¬∃ k : ℕ, 2*n' + 99 = k^2)) :=
by sorry

end largest_n_satisfying_conditions_l3135_313500


namespace first_rectangle_width_first_rectangle_width_proof_l3135_313547

/-- Given two rectangles, where the second has width 3 and height 6,
    and the first has height 5 and area 2 square inches more than the second,
    prove that the width of the first rectangle is 4 inches. -/
theorem first_rectangle_width : ℝ → Prop :=
  fun w : ℝ =>
    let first_height : ℝ := 5
    let second_width : ℝ := 3
    let second_height : ℝ := 6
    let first_area : ℝ := w * first_height
    let second_area : ℝ := second_width * second_height
    first_area = second_area + 2 → w = 4

/-- Proof of the theorem -/
theorem first_rectangle_width_proof : first_rectangle_width 4 := by
  sorry

end first_rectangle_width_first_rectangle_width_proof_l3135_313547


namespace geometric_sequence_common_ratio_l3135_313515

theorem geometric_sequence_common_ratio 
  (a : ℕ → ℝ) 
  (h_geometric : ∃ (q : ℝ) (a₁ : ℝ), ∀ n, a n = a₁ * q^(n-1))
  (h_condition : 2 * a 4 = a 6 - a 5) :
  ∃ (q : ℝ), (q = -1 ∨ q = 2) ∧ 
    (∃ (a₁ : ℝ), ∀ n, a n = a₁ * q^(n-1)) := by
sorry

end geometric_sequence_common_ratio_l3135_313515


namespace point_quadrant_relation_l3135_313594

-- Define a point in 2D space
def Point := ℝ × ℝ

-- Define the quadrants
def first_quadrant (p : Point) : Prop := p.1 > 0 ∧ p.2 > 0
def second_quadrant (p : Point) : Prop := p.1 < 0 ∧ p.2 > 0
def third_quadrant (p : Point) : Prop := p.1 < 0 ∧ p.2 < 0
def fourth_quadrant (p : Point) : Prop := p.1 > 0 ∧ p.2 < 0

-- Define the points P and Q
def P (b : ℝ) : Point := (2, b)
def Q (b : ℝ) : Point := (b, -2)

-- State the theorem
theorem point_quadrant_relation (b : ℝ) :
  fourth_quadrant (P b) → third_quadrant (Q b) :=
by
  sorry

end point_quadrant_relation_l3135_313594


namespace jessica_quarters_l3135_313514

/-- The number of quarters Jessica has after receiving quarters from her sister and friend. -/
def total_quarters (initial : ℕ) (from_sister : ℕ) (from_friend : ℕ) : ℕ :=
  initial + from_sister + from_friend

/-- Theorem stating that Jessica's total quarters is 16 given the initial amount and gifts. -/
theorem jessica_quarters : total_quarters 8 3 5 = 16 := by
  sorry

end jessica_quarters_l3135_313514


namespace projection_magnitude_l3135_313531

def a : ℝ × ℝ := (1, 3)
def b : ℝ × ℝ := (-2, 1)

theorem projection_magnitude :
  let proj_magnitude := abs ((a.1 * b.1 + a.2 * b.2) / (b.1^2 + b.2^2)) * Real.sqrt (b.1^2 + b.2^2)
  proj_magnitude = Real.sqrt 5 / 5 := by sorry

end projection_magnitude_l3135_313531


namespace equidistant_point_l3135_313589

theorem equidistant_point : ∃ x : ℝ, |x - (-2)| = |x - 4| ∧ x = 1 := by
  sorry

end equidistant_point_l3135_313589


namespace inequality_equivalence_l3135_313542

def set_A (x y : ℝ) : Prop := abs x ≤ 1 ∧ abs y ≤ 1 ∧ x * y ≤ 0

def set_B (x y : ℝ) : Prop := abs x ≤ 1 ∧ abs y ≤ 1 ∧ x^2 + y^2 ≤ 1

theorem inequality_equivalence (x y : ℝ) :
  Real.sqrt (1 - x^2) * Real.sqrt (1 - y^2) ≥ x * y ↔ set_A x y ∨ set_B x y :=
by sorry

end inequality_equivalence_l3135_313542


namespace proposition_a_is_true_l3135_313567

theorem proposition_a_is_true : ∀ x : ℝ, x = 2 → (x - 2) * (x - 1) = 0 := by
  sorry

#check proposition_a_is_true

end proposition_a_is_true_l3135_313567


namespace right_triangle_altitude_segment_length_l3135_313576

/-- A right triangle with specific altitude properties -/
structure RightTriangleWithAltitudes where
  -- The lengths of the segments on the hypotenuse
  hypotenuse_segment1 : ℝ
  hypotenuse_segment2 : ℝ
  -- The length of one segment on a leg
  leg_segment : ℝ
  -- Ensure the hypotenuse segments are positive
  hyp_seg1_pos : 0 < hypotenuse_segment1
  hyp_seg2_pos : 0 < hypotenuse_segment2
  -- Ensure the leg segment is positive
  leg_seg_pos : 0 < leg_segment

/-- The theorem stating the length of the unknown segment -/
theorem right_triangle_altitude_segment_length 
  (triangle : RightTriangleWithAltitudes) 
  (h1 : triangle.hypotenuse_segment1 = 4)
  (h2 : triangle.hypotenuse_segment2 = 6)
  (h3 : triangle.leg_segment = 3) :
  ∃ y : ℝ, y = 4.5 ∧ 
    (triangle.leg_segment / triangle.hypotenuse_segment1 = 
     (triangle.leg_segment + y) / (triangle.hypotenuse_segment1 + triangle.hypotenuse_segment2)) := by
  sorry

end right_triangle_altitude_segment_length_l3135_313576


namespace age_ratio_after_years_l3135_313544

def suzy_current_age : ℕ := 20
def mary_current_age : ℕ := 8
def years_elapsed : ℕ := 4

theorem age_ratio_after_years : 
  (suzy_current_age + years_elapsed) / (mary_current_age + years_elapsed) = 2 := by
  sorry

end age_ratio_after_years_l3135_313544


namespace bakers_cakes_l3135_313598

/-- Baker's cake problem -/
theorem bakers_cakes (total_cakes : ℕ) (cakes_left : ℕ) (cakes_sold : ℕ) : 
  total_cakes = 217 → cakes_left = 72 → cakes_sold = total_cakes - cakes_left → cakes_sold = 145 := by
  sorry

end bakers_cakes_l3135_313598


namespace total_weight_jack_and_sam_l3135_313582

theorem total_weight_jack_and_sam : 
  ∀ (jack_weight sam_weight : ℕ),
  jack_weight = 52 →
  jack_weight = sam_weight + 8 →
  jack_weight + sam_weight = 96 :=
by
  sorry

end total_weight_jack_and_sam_l3135_313582


namespace translation_preserves_shape_and_size_l3135_313557

-- Define a geometric figure
structure GeometricFigure where
  -- We don't need to specify the exact properties of a geometric figure for this statement
  dummy : Unit

-- Define a translation
def Translation := ℝ → ℝ → ℝ → ℝ

-- Define the concept of shape preservation
def PreservesShape (f : Translation) (fig : GeometricFigure) : Prop :=
  -- The exact definition is not provided in the problem, so we leave it abstract
  sorry

-- Define the concept of size preservation
def PreservesSize (f : Translation) (fig : GeometricFigure) : Prop :=
  -- The exact definition is not provided in the problem, so we leave it abstract
  sorry

-- Theorem statement
theorem translation_preserves_shape_and_size (f : Translation) (fig : GeometricFigure) :
  PreservesShape f fig ∧ PreservesSize f fig :=
by
  sorry

end translation_preserves_shape_and_size_l3135_313557


namespace chloe_picked_42_carrots_l3135_313551

/-- Represents the number of carrots Chloe picked on the second day -/
def carrots_picked_next_day (initial_carrots : ℕ) (carrots_thrown : ℕ) (final_total : ℕ) : ℕ :=
  final_total - (initial_carrots - carrots_thrown)

/-- Theorem stating that Chloe picked 42 carrots the next day -/
theorem chloe_picked_42_carrots : 
  carrots_picked_next_day 48 45 45 = 42 := by
  sorry

#eval carrots_picked_next_day 48 45 45

end chloe_picked_42_carrots_l3135_313551


namespace right_triangle_leg_square_l3135_313536

theorem right_triangle_leg_square (a c : ℝ) (h1 : c = a + 2) :
  ∃ b : ℝ, b^2 = 4*a + 4 ∧ a^2 + b^2 = c^2 := by
sorry

end right_triangle_leg_square_l3135_313536


namespace pearl_distribution_l3135_313555

theorem pearl_distribution (n : ℕ) : 
  (∀ m : ℕ, m > n → (m % 8 ≠ 6 ∨ m % 7 ≠ 5)) → 
  n % 8 = 6 → 
  n % 7 = 5 → 
  n % 9 = 0 → 
  n = 54 := by
sorry

end pearl_distribution_l3135_313555


namespace derivative_f_at_zero_l3135_313519

def f (x : ℝ) := x + x^2

theorem derivative_f_at_zero : 
  deriv f 0 = 1 := by sorry

end derivative_f_at_zero_l3135_313519


namespace success_permutations_l3135_313526

/-- The number of unique arrangements of letters in "SUCCESS" -/
def success_arrangements : ℕ := 420

/-- The total number of letters in "SUCCESS" -/
def total_letters : ℕ := 7

/-- The number of S's in "SUCCESS" -/
def num_s : ℕ := 3

/-- The number of C's in "SUCCESS" -/
def num_c : ℕ := 2

/-- The number of U's in "SUCCESS" -/
def num_u : ℕ := 1

/-- The number of E's in "SUCCESS" -/
def num_e : ℕ := 1

theorem success_permutations :
  success_arrangements = (Nat.factorial total_letters) / ((Nat.factorial num_s) * (Nat.factorial num_c)) :=
by sorry

end success_permutations_l3135_313526


namespace checkerboard_probability_l3135_313565

/-- The size of one side of the square checkerboard -/
def board_size : ℕ := 10

/-- The total number of squares on the checkerboard -/
def total_squares : ℕ := board_size * board_size

/-- The number of squares on the perimeter of the checkerboard -/
def perimeter_squares : ℕ := 4 * board_size - 4

/-- The number of squares not on the perimeter of the checkerboard -/
def inner_squares : ℕ := total_squares - perimeter_squares

/-- The probability of choosing a square not on the perimeter -/
def inner_square_probability : ℚ := inner_squares / total_squares

theorem checkerboard_probability :
  inner_square_probability = 16 / 25 := by sorry

end checkerboard_probability_l3135_313565


namespace sum_of_large_numbers_l3135_313534

theorem sum_of_large_numbers : 800000000000 + 299999999999 = 1099999999999 := by
  sorry

end sum_of_large_numbers_l3135_313534


namespace average_b_c_l3135_313508

theorem average_b_c (a b c : ℝ) 
  (h1 : (a + b) / 2 = 35) 
  (h2 : c - a = 90) : 
  (b + c) / 2 = 80 := by
sorry

end average_b_c_l3135_313508


namespace total_paper_weight_is_2074_l3135_313512

/-- Calculates the total weight of paper Barbara removed from the chest of drawers. -/
def total_paper_weight : ℕ :=
  let bundle_size : ℕ := 2
  let bunch_size : ℕ := 4
  let heap_size : ℕ := 20
  let pile_size : ℕ := 10
  let stack_size : ℕ := 5

  let colored_bundles : ℕ := 3
  let white_bunches : ℕ := 2
  let scrap_heaps : ℕ := 5
  let glossy_piles : ℕ := 4
  let cardstock_stacks : ℕ := 3

  let colored_weight : ℕ := 8
  let white_weight : ℕ := 12
  let scrap_weight : ℕ := 10
  let glossy_weight : ℕ := 15
  let cardstock_weight : ℕ := 22

  let colored_total := colored_bundles * bundle_size * colored_weight
  let white_total := white_bunches * bunch_size * white_weight
  let scrap_total := scrap_heaps * heap_size * scrap_weight
  let glossy_total := glossy_piles * pile_size * glossy_weight
  let cardstock_total := cardstock_stacks * stack_size * cardstock_weight

  colored_total + white_total + scrap_total + glossy_total + cardstock_total

theorem total_paper_weight_is_2074 : total_paper_weight = 2074 := by
  sorry

end total_paper_weight_is_2074_l3135_313512


namespace largest_valid_selection_l3135_313585

/-- Represents a selection of squares on an n × n grid -/
def Selection (n : ℕ) := Fin n → Fin n → Bool

/-- Checks if a rectangle contains a selected square -/
def containsSelected (s : Selection n) (x y w h : ℕ) : Prop :=
  ∃ (i j : Fin n), s i j ∧ x ≤ i.val ∧ i.val < x + w ∧ y ≤ j.val ∧ j.val < y + h

/-- Checks if a selection satisfies the condition for all rectangles -/
def validSelection (n : ℕ) (s : Selection n) : Prop :=
  ∀ (x y w h : ℕ), x + w ≤ n → y + h ≤ n → w * h ≥ n → containsSelected s x y w h

/-- The main theorem stating that 7 is the largest n satisfying the condition -/
theorem largest_valid_selection :
  (∀ n : ℕ, n ≤ 7 → ∃ s : Selection n, (∀ i : Fin n, ∃! j : Fin n, s i j) ∧ validSelection n s) ∧
  (∀ n : ℕ, n > 7 → ¬∃ s : Selection n, (∀ i : Fin n, ∃! j : Fin n, s i j) ∧ validSelection n s) :=
sorry

end largest_valid_selection_l3135_313585


namespace quadratic_intersection_and_sum_of_y_l3135_313522

/-- Quadratic function -/
def f (a x : ℝ) : ℝ := a * x^2 - (2*a - 2) * x - 3*a - 1

theorem quadratic_intersection_and_sum_of_y (a : ℝ) (h1 : a > 0) :
  (∃! x, f a x = -3*a - 2) →
  a^2 + 1/a^2 = 7 ∧
  ∀ m n y1 y2 : ℝ, m ≠ n → m + n = -2 → 
    f a m = y1 → f a n = y2 → y1 + y2 > -6 :=
by sorry

end quadratic_intersection_and_sum_of_y_l3135_313522


namespace cone_lateral_surface_area_l3135_313510

theorem cone_lateral_surface_area (radius : ℝ) (slant_height : ℝ) :
  radius = 3 → slant_height = 5 → π * radius * slant_height = 15 * π := by
  sorry

end cone_lateral_surface_area_l3135_313510


namespace find_k_value_l3135_313525

/-- Given two functions f and g, prove that k = 27/25 when f(5) - g(5) = 45 -/
theorem find_k_value (f g : ℝ → ℝ) (k : ℝ) 
    (hf : ∀ x, f x = 2*x^3 - 5*x^2 + 3*x + 7)
    (hg : ∀ x, g x = 3*x^3 - k*x^2 + 4)
    (h_diff : f 5 - g 5 = 45) : 
  k = 27/25 := by
sorry

end find_k_value_l3135_313525


namespace evaluate_expression_l3135_313558

theorem evaluate_expression : 5 - 9 * (8 - 3 * 2) / 2 = -4 := by
  sorry

end evaluate_expression_l3135_313558


namespace roots_sum_minus_product_l3135_313579

theorem roots_sum_minus_product (x₁ x₂ : ℝ) : 
  x₁^2 - 2*x₁ - 1 = 0 → 
  x₂^2 - 2*x₂ - 1 = 0 → 
  x₁ + x₂ - x₁*x₂ = 3 := by
sorry

end roots_sum_minus_product_l3135_313579


namespace accounting_class_average_score_l3135_313563

/-- The average score for an accounting class --/
def average_score (total_students : ℕ) 
  (day1_percent day2_percent day3_percent : ℚ)
  (day1_score day2_score day3_score : ℚ) : ℚ :=
  (day1_percent * day1_score + day2_percent * day2_score + day3_percent * day3_score) / 1

theorem accounting_class_average_score :
  let total_students : ℕ := 200
  let day1_percent : ℚ := 60 / 100
  let day2_percent : ℚ := 30 / 100
  let day3_percent : ℚ := 10 / 100
  let day1_score : ℚ := 65 / 100
  let day2_score : ℚ := 75 / 100
  let day3_score : ℚ := 95 / 100
  average_score total_students day1_percent day2_percent day3_percent day1_score day2_score day3_score = 71 / 100 := by
  sorry

end accounting_class_average_score_l3135_313563


namespace river_depth_difference_l3135_313568

/-- River depth problem -/
theorem river_depth_difference (depth_may depth_june depth_july : ℝ) : 
  depth_may = 5 →
  depth_july = 45 →
  depth_july = 3 * depth_june →
  depth_june - depth_may = 10 := by
  sorry

end river_depth_difference_l3135_313568


namespace triangle_properties_l3135_313513

/-- Given a triangle ABC with circumradius R and satisfying the given equation,
    prove that angle C is π/3 and the maximum area is 3√3/2 -/
theorem triangle_properties (A B C : Real) (a b c : Real) (R : Real) :
  R = Real.sqrt 2 →
  2 * Real.sqrt 2 * (Real.sin A ^ 2 - Real.sin C ^ 2) = (a - b) * Real.sin B →
  (C = Real.pi / 3 ∧ 
   ∃ (S : Real), S = 3 * Real.sqrt 3 / 2 ∧ 
   ∀ (S' : Real), S' = 1/2 * a * b * Real.sin C → S' ≤ S) :=
by sorry

end triangle_properties_l3135_313513


namespace intersection_of_M_and_N_l3135_313509

open Set

def M : Set ℝ := {x | (x + 2) * (x - 1) < 0}
def N : Set ℝ := {x | x + 1 < 0}

theorem intersection_of_M_and_N : M ∩ N = {x : ℝ | -2 < x ∧ x < -1} := by sorry

end intersection_of_M_and_N_l3135_313509


namespace replacement_concentration_l3135_313588

/-- Represents a salt solution with a given concentration -/
structure SaltSolution where
  concentration : ℝ
  concentration_nonneg : 0 ≤ concentration
  concentration_le_one : concentration ≤ 1

/-- The result of mixing two salt solutions -/
def mix_solutions (s1 s2 : SaltSolution) (ratio : ℝ) : SaltSolution where
  concentration := s1.concentration * (1 - ratio) + s2.concentration * ratio
  concentration_nonneg := sorry
  concentration_le_one := sorry

theorem replacement_concentration 
  (original second : SaltSolution)
  (h1 : original.concentration = 0.14)
  (h2 : (mix_solutions original second 0.25).concentration = 0.16) :
  second.concentration = 0.22 := by
  sorry

end replacement_concentration_l3135_313588


namespace sqrt_of_nine_equals_three_l3135_313523

theorem sqrt_of_nine_equals_three : Real.sqrt 9 = 3 := by
  sorry

end sqrt_of_nine_equals_three_l3135_313523


namespace percentage_equality_l3135_313590

theorem percentage_equality (x : ℝ) (h : x = 130) : 
  (65 / 100 * x) / 422.50 * 100 = 20 := by
  sorry

end percentage_equality_l3135_313590


namespace two_digit_swap_difference_divisible_by_nine_l3135_313528

theorem two_digit_swap_difference_divisible_by_nine 
  (a b : ℕ) 
  (h1 : a ≤ 9) 
  (h2 : b ≤ 9) 
  (h3 : a ≠ b) : 
  ∃ k : ℤ, (|(10 * a + b) - (10 * b + a)| : ℤ) = 9 * k := by
  sorry

end two_digit_swap_difference_divisible_by_nine_l3135_313528


namespace sequoia_maple_height_difference_l3135_313505

/-- Represents the height of a tree in feet and quarters of a foot -/
structure TreeHeight where
  feet : ℕ
  quarters : Fin 4

/-- Converts a TreeHeight to a rational number -/
def treeHeightToRational (h : TreeHeight) : ℚ :=
  h.feet + h.quarters.val / 4

/-- The height of the maple tree -/
def mapleHeight : TreeHeight := ⟨13, 3⟩

/-- The height of the sequoia -/
def sequoiaHeight : TreeHeight := ⟨20, 2⟩

theorem sequoia_maple_height_difference :
  treeHeightToRational sequoiaHeight - treeHeightToRational mapleHeight = 27 / 4 := by
  sorry

#eval treeHeightToRational sequoiaHeight - treeHeightToRational mapleHeight

end sequoia_maple_height_difference_l3135_313505


namespace complex_multiplication_l3135_313580

theorem complex_multiplication (i : ℂ) : i * i = -1 → (1 + i) * (1 - i) = 2 := by
  sorry

end complex_multiplication_l3135_313580


namespace sixth_term_of_geometric_sequence_l3135_313577

/-- A geometric sequence is a sequence where the ratio of successive terms is constant. -/
def IsGeometric (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = a n * q

/-- The sixth term of a geometric sequence satisfying given conditions. -/
theorem sixth_term_of_geometric_sequence
  (a : ℕ → ℝ)
  (h_geometric : IsGeometric a)
  (h_sum : a 1 + a 2 = -1)
  (h_diff : a 1 - a 3 = -3) :
  a 6 = -32 := by
sorry

end sixth_term_of_geometric_sequence_l3135_313577


namespace triangle_perimeter_bound_l3135_313559

theorem triangle_perimeter_bound : ∀ s : ℝ,
  s > 0 →
  s + 7 > 21 →
  s + 21 > 7 →
  7 + 21 > s →
  (∃ n : ℕ, n = 57 ∧ ∀ m : ℕ, m > (s + 7 + 21) → m ≥ n) :=
by sorry

end triangle_perimeter_bound_l3135_313559


namespace tomato_price_equality_l3135_313527

/-- Prove that the original price per pound of tomatoes equals the selling price of remaining tomatoes --/
theorem tomato_price_equality (original_price : ℝ) 
  (ruined_percentage : ℝ) (profit_percentage : ℝ) (selling_price : ℝ) : 
  ruined_percentage = 0.2 →
  profit_percentage = 0.08 →
  selling_price = 1.08 →
  (1 - ruined_percentage) * selling_price = (1 + profit_percentage) * original_price :=
by sorry

end tomato_price_equality_l3135_313527


namespace nancy_vacation_pictures_l3135_313554

/-- The number of pictures Nancy took at the zoo -/
def zoo_pictures : ℕ := 49

/-- The number of pictures Nancy took at the museum -/
def museum_pictures : ℕ := 8

/-- The number of pictures Nancy deleted -/
def deleted_pictures : ℕ := 38

/-- The total number of pictures Nancy took during her vacation -/
def total_pictures : ℕ := zoo_pictures + museum_pictures

/-- The number of pictures Nancy has after deleting some -/
def remaining_pictures : ℕ := total_pictures - deleted_pictures

theorem nancy_vacation_pictures : remaining_pictures = 19 := by
  sorry

end nancy_vacation_pictures_l3135_313554


namespace tangent_line_equation_l3135_313539

/-- The curve function -/
def f (x : ℝ) : ℝ := 2 * x^2 + 1

/-- The derivative of the curve function -/
def f' (x : ℝ) : ℝ := 4 * x

/-- The point of tangency -/
def P : ℝ × ℝ := (-1, 3)

/-- The slope of the tangent line at P -/
def k : ℝ := f' P.1

/-- The equation of the tangent line -/
def tangent_line (x y : ℝ) : Prop := 4 * x + y + 1 = 0

theorem tangent_line_equation :
  ∀ x y : ℝ, (x, y) ∈ {(x, y) | tangent_line x y} ↔
  y - P.2 = k * (x - P.1) ∧ y = f x := by sorry

end tangent_line_equation_l3135_313539


namespace shanes_bread_packages_l3135_313562

theorem shanes_bread_packages :
  ∀ (slices_per_bread_package : ℕ) 
    (ham_packages : ℕ) 
    (slices_per_ham_package : ℕ) 
    (bread_slices_per_sandwich : ℕ) 
    (leftover_bread_slices : ℕ),
  slices_per_bread_package = 20 →
  ham_packages = 2 →
  slices_per_ham_package = 8 →
  bread_slices_per_sandwich = 2 →
  leftover_bread_slices = 8 →
  (ham_packages * slices_per_ham_package * bread_slices_per_sandwich + leftover_bread_slices) / slices_per_bread_package = 2 :=
by
  sorry

end shanes_bread_packages_l3135_313562


namespace belt_length_sufficient_l3135_313511

/-- Given three pulleys with parallel axes and identical radii, prove that 
    a 54 cm cord is sufficient for the belt connecting them. -/
theorem belt_length_sufficient 
  (r : ℝ) 
  (O₁O₂ O₁O₃ O₃_to_plane : ℝ) 
  (h_r : r = 2)
  (h_O₁O₂ : O₁O₂ = 12)
  (h_O₁O₃ : O₁O₃ = 10)
  (h_O₃_to_plane : O₃_to_plane = 8) :
  ∃ (belt_length : ℝ), 
    belt_length < 54 ∧ 
    belt_length = 
      O₁O₂ + O₁O₃ + Real.sqrt (O₁O₂^2 + O₁O₃^2 - 2 * O₁O₂ * O₁O₃ * (O₃_to_plane / O₁O₃)) + 
      2 * π * r :=
by sorry

end belt_length_sufficient_l3135_313511


namespace sequence_a_equals_fibonacci_6n_l3135_313573

def fibonacci : ℕ → ℕ
  | 0 => 1
  | 1 => 1
  | (n + 2) => fibonacci (n + 1) + fibonacci n

def sequence_a : ℕ → ℕ
  | 0 => 1
  | (n + 1) => (18 * sequence_a n + 8 * (Nat.sqrt (5 * (sequence_a n)^2 - 4))) / 2

theorem sequence_a_equals_fibonacci_6n :
  ∀ n : ℕ, sequence_a n = fibonacci (6 * n) := by
  sorry

end sequence_a_equals_fibonacci_6n_l3135_313573


namespace cube_inequality_equivalence_l3135_313504

theorem cube_inequality_equivalence (a b : ℝ) : a > b ↔ a^3 > b^3 := by sorry

end cube_inequality_equivalence_l3135_313504


namespace journey_time_calculation_l3135_313592

/-- Given a constant speed, if a 200-mile journey takes 5 hours, 
    then a 120-mile journey will take 3 hours. -/
theorem journey_time_calculation (speed : ℝ) 
  (h1 : speed > 0)
  (h2 : 200 = speed * 5) : 
  120 = speed * 3 := by
  sorry

end journey_time_calculation_l3135_313592


namespace beavers_still_working_l3135_313545

theorem beavers_still_working (total : ℕ) (wood : ℕ) (dam : ℕ) (lodge : ℕ)
  (wood_break : ℕ) (dam_break : ℕ) (lodge_break : ℕ)
  (h1 : total = 12)
  (h2 : wood = 5)
  (h3 : dam = 4)
  (h4 : lodge = 3)
  (h5 : wood_break = 3)
  (h6 : dam_break = 2)
  (h7 : lodge_break = 1) :
  (wood - wood_break) + (dam - dam_break) + (lodge - lodge_break) = 6 :=
by
  sorry

end beavers_still_working_l3135_313545


namespace complex_modulus_power_eight_l3135_313583

theorem complex_modulus_power_eight : Complex.abs ((1 - Complex.I) ^ 8) = 16 := by
  sorry

end complex_modulus_power_eight_l3135_313583


namespace election_ratio_l3135_313543

theorem election_ratio (X Y : ℝ) 
  (h1 : 0.64 * X + 0.46 * Y = 0.58 * (X + Y)) 
  (h2 : X > 0) 
  (h3 : Y > 0) : 
  X / Y = 2 := by
sorry

end election_ratio_l3135_313543


namespace unique_star_solution_l3135_313597

/-- Definition of the ★ operation -/
def star (x y : ℝ) : ℝ := 5*x - 4*y + 2*x*y

/-- Theorem stating that there exists a unique real number y such that 4 ★ y = 10 -/
theorem unique_star_solution : ∃! y : ℝ, star 4 y = 10 := by
  sorry

end unique_star_solution_l3135_313597


namespace smallest_total_score_l3135_313569

theorem smallest_total_score : 
  ∃ (T : ℕ), T > 0 ∧ 
  (∃ (n m : ℕ), 2 * n + 5 * m = T ∧ (n ≥ m + 3 ∨ m ≥ n + 3)) ∧ 
  (∀ (S : ℕ), S > 0 → S < T → 
    ¬(∃ (n m : ℕ), 2 * n + 5 * m = S ∧ (n ≥ m + 3 ∨ m ≥ n + 3))) ∧
  T = 20 :=
by sorry

end smallest_total_score_l3135_313569


namespace complex_root_equation_l3135_313591

/-- Given a quadratic equation with complex coefficients and a real parameter,
    prove that if it has a real root, then the complex number formed by the
    parameter and the root has a specific value. -/
theorem complex_root_equation (a : ℝ) (b : ℝ) :
  (∃ x : ℝ, x^2 + (4 + Complex.I) * x + (4 : ℂ) + a * Complex.I = 0) →
  (b^2 + (4 + Complex.I) * b + (4 : ℂ) + a * Complex.I = 0) →
  (a + b * Complex.I = 2 - 2 * Complex.I) := by
  sorry

end complex_root_equation_l3135_313591


namespace cookies_division_l3135_313532

/-- The number of cookies each person received -/
def cookies_per_person : ℕ := 30

/-- The total number of cookies prepared -/
def total_cookies : ℕ := 420

/-- The number of people Brenda's mother made cookies for -/
def number_of_people : ℕ := total_cookies / cookies_per_person

theorem cookies_division (cookies_per_person : ℕ) (total_cookies : ℕ) :
  cookies_per_person > 0 →
  total_cookies % cookies_per_person = 0 →
  number_of_people = 14 := by
  sorry

end cookies_division_l3135_313532


namespace least_x_divisible_by_three_l3135_313550

theorem least_x_divisible_by_three : 
  ∃ (x : ℕ), x < 10 ∧ 
  (∀ (y : ℕ), y < x → ¬(23 * 100 + y * 10 + 57) % 3 = 0) ∧
  (23 * 100 + x * 10 + 57) % 3 = 0 :=
by
  -- The proof goes here
  sorry

end least_x_divisible_by_three_l3135_313550


namespace trigonometric_equality_l3135_313540

theorem trigonometric_equality (θ : Real) (h : Real.sin (3 * Real.pi + θ) = 1/2) :
  (Real.cos (3 * Real.pi + θ)) / (Real.cos θ * (Real.cos (Real.pi + θ) - 1)) +
  (Real.cos (θ - 4 * Real.pi)) / (Real.cos (θ + 2 * Real.pi) * Real.cos (3 * Real.pi + θ) + Real.cos (-θ)) = 8 := by
  sorry

end trigonometric_equality_l3135_313540


namespace original_flock_size_l3135_313541

/-- Represents the flock size and its changes over time -/
structure FlockDynamics where
  initialSize : ℕ
  yearlyKilled : ℕ
  yearlyBorn : ℕ
  years : ℕ
  joinedFlockSize : ℕ
  finalCombinedSize : ℕ

/-- Theorem stating the original flock size given the conditions -/
theorem original_flock_size (fd : FlockDynamics)
  (h1 : fd.yearlyKilled = 20)
  (h2 : fd.yearlyBorn = 30)
  (h3 : fd.years = 5)
  (h4 : fd.joinedFlockSize = 150)
  (h5 : fd.finalCombinedSize = 300)
  : fd.initialSize = 100 := by
  sorry

#check original_flock_size

end original_flock_size_l3135_313541


namespace david_window_washing_time_l3135_313553

/-- Represents the time taken to wash windows -/
def wash_time (windows_per_unit : ℕ) (minutes_per_unit : ℕ) (total_windows : ℕ) : ℕ :=
  (total_windows / windows_per_unit) * minutes_per_unit

/-- Proves that it takes David 160 minutes to wash all windows in his house -/
theorem david_window_washing_time :
  wash_time 4 10 64 = 160 := by
  sorry

#eval wash_time 4 10 64

end david_window_washing_time_l3135_313553


namespace imaginary_complex_implies_m_condition_l3135_313520

theorem imaginary_complex_implies_m_condition (m : ℝ) : 
  (Complex.I * (m^2 - 5*m - 6) ≠ 0) → (m ≠ -1 ∧ m ≠ 6) := by
  sorry

end imaginary_complex_implies_m_condition_l3135_313520


namespace sum_of_two_numbers_l3135_313552

theorem sum_of_two_numbers (s l : ℕ) : s = 9 → l = 4 * s → s + l = 45 := by
  sorry

end sum_of_two_numbers_l3135_313552


namespace generalized_spatial_apollonian_problems_l3135_313571

/-- The number of types of objects (sphere, point, plane) --/
def n : ℕ := 3

/-- The number of objects to be chosen --/
def k : ℕ := 4

/-- Combinations with repetition --/
def combinations_with_repetition (n k : ℕ) : ℕ := Nat.choose (n + k - 1) k

/-- The number of generalized spatial Apollonian problems --/
theorem generalized_spatial_apollonian_problems :
  combinations_with_repetition n k = 15 := by
  sorry

end generalized_spatial_apollonian_problems_l3135_313571


namespace ellen_initial_legos_l3135_313518

/-- The number of Legos Ellen lost -/
def lost_legos : ℕ := 17

/-- The number of Legos Ellen currently has -/
def current_legos : ℕ := 2063

/-- The initial number of Legos Ellen had -/
def initial_legos : ℕ := current_legos + lost_legos

theorem ellen_initial_legos : initial_legos = 2080 := by
  sorry

end ellen_initial_legos_l3135_313518


namespace k_eval_at_one_l3135_313521

-- Define the polynomials h and k
def h (p : ℝ) (x : ℝ) : ℝ := x^3 + p*x^2 + 2*x + 15
def k (q r : ℝ) (x : ℝ) : ℝ := x^4 + x^3 + q*x^2 + 150*x + r

-- State the theorem
theorem k_eval_at_one (p q r : ℝ) : 
  (∃ x y z : ℝ, x ≠ y ∧ y ≠ z ∧ x ≠ z ∧ h p x = 0 ∧ h p y = 0 ∧ h p z = 0) →  -- h has three distinct roots
  (∀ x : ℝ, h p x = 0 → k q r x = 0) →  -- each root of h is a root of k
  k q r 1 = -3322.25 := by
sorry

end k_eval_at_one_l3135_313521


namespace equal_roots_quadratic_l3135_313549

theorem equal_roots_quadratic (m : ℝ) : 
  (∃ x : ℝ, 2 * x^2 - x + m = 0 ∧ 
   ∀ y : ℝ, 2 * y^2 - y + m = 0 → y = x) → 
  m = 1/8 := by
sorry

end equal_roots_quadratic_l3135_313549


namespace last_three_digits_of_11_pow_30_l3135_313546

theorem last_three_digits_of_11_pow_30 : 11^30 ≡ 801 [ZMOD 1000] := by
  sorry

end last_three_digits_of_11_pow_30_l3135_313546


namespace plate_difference_l3135_313501

/- Define the number of kitchen supplies for Angela and Sharon -/
def angela_pots : ℕ := 20
def angela_plates : ℕ := 3 * angela_pots + 6
def angela_cutlery : ℕ := angela_plates / 2

def sharon_pots : ℕ := angela_pots / 2
def sharon_cutlery : ℕ := angela_cutlery * 2
def sharon_total : ℕ := 254

/- Define Sharon's plates as the remaining items after subtracting pots and cutlery from the total -/
def sharon_plates : ℕ := sharon_total - (sharon_pots + sharon_cutlery)

/- Theorem stating the difference between Sharon's plates and three times Angela's plates -/
theorem plate_difference : 
  3 * angela_plates - sharon_plates = 20 := by sorry

end plate_difference_l3135_313501


namespace two_solutions_for_equation_l3135_313584

theorem two_solutions_for_equation : 
  ∃! (n : ℕ), n = (Finset.filter 
    (fun p : ℕ × ℕ => 
      let a := p.1
      let b := p.2
      a > 0 ∧ b > 0 ∧ (a + b + 3)^2 = 4*(a^2 + b^2))
    (Finset.product (Finset.range 1000) (Finset.range 1000))).card 
  ∧ n = 2 := by sorry

end two_solutions_for_equation_l3135_313584


namespace sqrt_of_four_equals_two_l3135_313530

theorem sqrt_of_four_equals_two : Real.sqrt 4 = 2 := by
  sorry

end sqrt_of_four_equals_two_l3135_313530


namespace inequality_proof_l3135_313537

theorem inequality_proof (a b c : ℝ) (ha : a ≥ c) (hb : b ≥ c) (hc : c > 0) :
  Real.sqrt (c * (a - c)) + Real.sqrt (c * (b - c)) ≤ Real.sqrt (a * b) ∧
  (Real.sqrt (c * (a - c)) + Real.sqrt (c * (b - c)) = Real.sqrt (a * b) ↔ a * b = c * (a + b)) :=
by sorry

end inequality_proof_l3135_313537


namespace expression_equals_expected_result_l3135_313561

-- Define the expression
def expression : ℤ := 8 - (-3) + (-5) + (-7)

-- Define the expected result
def expected_result : ℤ := 3 + 8 - 7 - 5

-- Theorem statement
theorem expression_equals_expected_result :
  expression = expected_result :=
by sorry

end expression_equals_expected_result_l3135_313561


namespace salary_change_percentage_l3135_313596

theorem salary_change_percentage (x : ℝ) : 
  (1 - x / 100) * (1 + x / 100) = 0.51 → x = 70 := by sorry

end salary_change_percentage_l3135_313596


namespace range_of_m_l3135_313507

-- Define a monotonically decreasing function
def MonoDecreasing (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f x > f y

-- State the theorem
theorem range_of_m (f : ℝ → ℝ) (m : ℝ) 
  (h1 : MonoDecreasing f) (h2 : f (2 * m) > f (1 + m)) : 
  m < 1 := by
  sorry

end range_of_m_l3135_313507


namespace triangle_covering_polygon_l3135_313506

-- Define the types for points and polygons
variable (Point : Type) [NormedAddCommGroup Point] [InnerProductSpace ℝ Point]
variable (Polygon : Type)
variable (Triangle : Type)

-- Define the properties and relations
variable (covers : Triangle → Polygon → Prop)
variable (congruent : Triangle → Triangle → Prop)
variable (has_parallel_side : Triangle → Polygon → Prop)

-- State the theorem
theorem triangle_covering_polygon
  (ABC : Triangle) (M : Polygon) 
  (h_covers : covers ABC M) :
  ∃ (DEF : Triangle), 
    congruent DEF ABC ∧ 
    covers DEF M ∧ 
    has_parallel_side DEF M :=
sorry

end triangle_covering_polygon_l3135_313506


namespace matching_shoes_probability_l3135_313502

theorem matching_shoes_probability (total_shoes : ℕ) (total_pairs : ℕ) (h1 : total_shoes = 12) (h2 : total_pairs = 6) :
  let total_selections := total_shoes.choose 2
  let matching_selections := total_pairs
  (matching_selections : ℚ) / total_selections = 1 / 11 := by sorry

end matching_shoes_probability_l3135_313502


namespace mari_buttons_l3135_313535

theorem mari_buttons (sue_buttons : ℕ) (kendra_buttons : ℕ) (mari_buttons : ℕ) : 
  sue_buttons = 6 →
  sue_buttons = kendra_buttons / 2 →
  mari_buttons = 4 + 5 * kendra_buttons →
  mari_buttons = 64 := by
sorry

end mari_buttons_l3135_313535


namespace vector_equation_l3135_313517

def a : ℝ × ℝ := (1, -1)
def b : ℝ × ℝ := (2, 1)
def c : ℝ × ℝ := (-2, 1)

theorem vector_equation (x y : ℝ) (h : c = x • a + y • b) : x - y = -1 := by
  sorry

end vector_equation_l3135_313517


namespace sticker_distribution_l3135_313570

theorem sticker_distribution (total_stickers : ℕ) (num_friends : ℕ) 
  (h1 : total_stickers = 72) (h2 : num_friends = 9) :
  total_stickers / num_friends = 8 := by
sorry

end sticker_distribution_l3135_313570


namespace age_ratio_is_two_to_one_l3135_313564

def age_ratio (a_current : ℕ) (b_current : ℕ) : ℚ :=
  (a_current + 20) / (b_current - 20)

theorem age_ratio_is_two_to_one :
  ∀ (a_current b_current : ℕ),
    b_current = 70 →
    a_current = b_current + 10 →
    age_ratio a_current b_current = 2 := by
  sorry

end age_ratio_is_two_to_one_l3135_313564


namespace factor_polynomial_l3135_313578

theorem factor_polynomial (x : ℝ) : 72 * x^5 - 162 * x^9 = -18 * x^5 * (9 * x^4 - 4) := by
  sorry

end factor_polynomial_l3135_313578


namespace original_number_l3135_313560

theorem original_number : ∃ x : ℕ, x - (x / 3) = 36 ∧ x = 54 := by
  sorry

end original_number_l3135_313560


namespace solution_difference_l3135_313538

theorem solution_difference (r s : ℝ) : 
  ((r - 5) * (r + 5) = 24 * r - 120) →
  ((s - 5) * (s + 5) = 24 * s - 120) →
  r ≠ s →
  r > s →
  r - s = 14 := by
sorry

end solution_difference_l3135_313538


namespace infinite_solutions_l3135_313516

theorem infinite_solutions (b : ℝ) :
  (∀ x, 5 * (4 * x - b) = 3 * (5 * x + 15)) ↔ b = -9 := by
  sorry

end infinite_solutions_l3135_313516


namespace desk_rearrangement_combinations_l3135_313593

theorem desk_rearrangement_combinations : 
  let day1_choices : ℕ := 1
  let day2_choices : ℕ := 2
  let day3_choices : ℕ := 3
  let day4_choices : ℕ := 2
  let day5_choices : ℕ := 1
  day1_choices * day2_choices * day3_choices * day4_choices * day5_choices = 12 := by
sorry

end desk_rearrangement_combinations_l3135_313593


namespace contradiction_assumption_l3135_313529

theorem contradiction_assumption (a b : ℝ) : ¬(a > b) ↔ (a ≤ b) := by sorry

end contradiction_assumption_l3135_313529


namespace andrew_donuts_problem_l3135_313586

theorem andrew_donuts_problem (monday tuesday wednesday : ℕ) : 
  tuesday = monday / 2 →
  wednesday = 4 * monday →
  monday + tuesday + wednesday = 49 →
  monday = 9 :=
by
  sorry

end andrew_donuts_problem_l3135_313586


namespace school_growth_difference_l3135_313587

theorem school_growth_difference
  (total_last_year : ℕ)
  (school_yy_last_year : ℕ)
  (xx_growth_rate : ℚ)
  (yy_growth_rate : ℚ)
  (h1 : total_last_year = 4000)
  (h2 : school_yy_last_year = 2400)
  (h3 : xx_growth_rate = 7 / 100)
  (h4 : yy_growth_rate = 3 / 100) :
  let school_xx_last_year := total_last_year - school_yy_last_year
  let xx_growth := (school_xx_last_year : ℚ) * xx_growth_rate
  let yy_growth := (school_yy_last_year : ℚ) * yy_growth_rate
  ⌊xx_growth - yy_growth⌋ = 40 := by
  sorry

end school_growth_difference_l3135_313587


namespace quadratic_equation_properties_l3135_313503

theorem quadratic_equation_properties (m : ℝ) :
  let f : ℝ → ℝ := λ x => x^2 - m*x + m - 2
  (∃ x, f x = 0 ∧ x = -1) → m = 1/2 ∧
  ∃ x y, x ≠ y ∧ f x = 0 ∧ f y = 0 := by
sorry

end quadratic_equation_properties_l3135_313503


namespace sin_2y_plus_x_l3135_313575

theorem sin_2y_plus_x (x y : Real) 
  (h1 : Real.sin x = 1/3) 
  (h2 : Real.sin (x + y) = 1) : 
  Real.sin (2*y + x) = 1/3 := by
sorry

end sin_2y_plus_x_l3135_313575


namespace x_plus_p_equals_2p_plus_3_l3135_313599

theorem x_plus_p_equals_2p_plus_3 (x p : ℝ) (h1 : |x - 3| = p) (h2 : x > 3) :
  x + p = 2*p + 3 := by
  sorry

end x_plus_p_equals_2p_plus_3_l3135_313599


namespace y_squared_value_l3135_313548

theorem y_squared_value (y : ℝ) (h : (y + 16) ^ (1/4) - (y - 16) ^ (1/4) = 2) : y^2 = 272 := by
  sorry

end y_squared_value_l3135_313548


namespace union_of_sets_l3135_313581

/-- Given sets A and B, prove that their union is [-1, +∞) -/
theorem union_of_sets (A B : Set ℝ) : 
  (A = {x : ℝ | -3 ≤ 2*x - 1 ∧ 2*x - 1 ≤ 3}) →
  (B = {x : ℝ | x > 1}) →
  A ∪ B = Set.Ici (-1) := by
  sorry

end union_of_sets_l3135_313581


namespace daniels_cats_l3135_313566

theorem daniels_cats (horses dogs turtles goats cats : ℕ) : 
  horses = 2 → 
  dogs = 5 → 
  turtles = 3 → 
  goats = 1 → 
  4 * (horses + dogs + cats + turtles + goats) = 72 → 
  cats = 7 := by
sorry

end daniels_cats_l3135_313566


namespace todds_time_is_correct_l3135_313572

/-- Todd's running time around the track -/
def todds_time : ℕ := 88

/-- Brian's running time around the track -/
def brians_time : ℕ := 96

/-- The difference in running time between Brian and Todd -/
def time_difference : ℕ := 8

/-- Theorem stating that Todd's time is correct given the conditions -/
theorem todds_time_is_correct : todds_time = brians_time - time_difference := by
  sorry

end todds_time_is_correct_l3135_313572


namespace angle_from_terminal_point_l3135_313524

theorem angle_from_terminal_point (α : Real) :
  (∃ (x y : Real), x = Real.sin (π / 5) ∧ y = -Real.cos (π / 5) ∧ 
   x = Real.sin α ∧ y = Real.cos α) →
  ∃ (k : ℤ), α = -3 * π / 10 + 2 * π * (k : Real) :=
by sorry

end angle_from_terminal_point_l3135_313524


namespace log_equation_l3135_313556

-- Define the logarithm base 10 function
noncomputable def log10 (x : ℝ) : ℝ := Real.log x / Real.log 10

-- State the theorem
theorem log_equation : log10 2 * log10 50 + log10 25 - log10 5 * log10 20 = 1 := by
  sorry

end log_equation_l3135_313556


namespace sin_360_minus_alpha_eq_sin_alpha_l3135_313595

theorem sin_360_minus_alpha_eq_sin_alpha (α : ℝ) : 
  Real.sin (2 * Real.pi - α) = Real.sin α := by sorry

end sin_360_minus_alpha_eq_sin_alpha_l3135_313595


namespace jenny_easter_eggs_l3135_313533

theorem jenny_easter_eggs (red_eggs : ℕ) (orange_eggs : ℕ) (eggs_per_basket : ℕ) 
  (h1 : red_eggs = 21)
  (h2 : orange_eggs = 28)
  (h3 : eggs_per_basket ≥ 5)
  (h4 : red_eggs % eggs_per_basket = 0)
  (h5 : orange_eggs % eggs_per_basket = 0) :
  eggs_per_basket = 7 := by
sorry

end jenny_easter_eggs_l3135_313533
