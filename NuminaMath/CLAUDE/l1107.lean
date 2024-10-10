import Mathlib

namespace ajax_initial_weight_ajax_initial_weight_is_80_l1107_110799

/-- Proves that Ajax's initial weight is 80 kg given the exercise and weight conditions --/
theorem ajax_initial_weight : ℝ → Prop :=
  fun (initial_weight : ℝ) =>
    let pounds_per_kg : ℝ := 2.2
    let weight_loss_per_hour : ℝ := 1.5
    let hours_per_day : ℝ := 2
    let days : ℝ := 14
    let final_weight_pounds : ℝ := 134
    
    let total_weight_loss : ℝ := weight_loss_per_hour * hours_per_day * days
    let initial_weight_pounds : ℝ := final_weight_pounds + total_weight_loss
    
    initial_weight = initial_weight_pounds / pounds_per_kg ∧ initial_weight = 80

theorem ajax_initial_weight_is_80 : ajax_initial_weight 80 := by
  sorry

end ajax_initial_weight_ajax_initial_weight_is_80_l1107_110799


namespace function_inequality_l1107_110782

theorem function_inequality (f : ℝ → ℝ) (hf : Differentiable ℝ f) 
  (h : ∀ x, f x + 2 * (deriv f x) > 0) : 
  f 1 > f 0 / Real.sqrt (Real.exp 1) := by
  sorry

end function_inequality_l1107_110782


namespace attractions_permutations_l1107_110703

theorem attractions_permutations : Nat.factorial 5 = 120 := by
  sorry

end attractions_permutations_l1107_110703


namespace log_expression_equality_l1107_110701

theorem log_expression_equality : 
  (Real.log 3 / Real.log 2 + Real.log 3 / Real.log 8) / (Real.log 9 / Real.log 4) = 4 / 3 := by
  sorry

end log_expression_equality_l1107_110701


namespace bug_meeting_point_l1107_110734

theorem bug_meeting_point (PQ QR RP : ℝ) (h1 : PQ = 7) (h2 : QR = 8) (h3 : RP = 9) :
  let perimeter := PQ + QR + RP
  let distance_traveled := 10
  let QS := distance_traveled - PQ
  QS = 3 := by sorry

end bug_meeting_point_l1107_110734


namespace ab_in_terms_of_m_and_n_l1107_110756

theorem ab_in_terms_of_m_and_n (a b m n : ℝ) 
  (h1 : (a + b)^2 = m) 
  (h2 : (a - b)^2 = n) : 
  a * b = (m - n) / 4 := by
sorry

end ab_in_terms_of_m_and_n_l1107_110756


namespace total_flowers_l1107_110729

theorem total_flowers (num_pots : ℕ) (flowers_per_pot : ℕ) 
  (h1 : num_pots = 2150) 
  (h2 : flowers_per_pot = 128) : 
  num_pots * flowers_per_pot = 275200 := by
  sorry

end total_flowers_l1107_110729


namespace intimate_functions_range_l1107_110727

theorem intimate_functions_range (f g : ℝ → ℝ) (m : ℝ) :
  (∀ x ∈ Set.Icc 2 3, f x = x^3 - 2*x + 7) →
  (∀ x ∈ Set.Icc 2 3, g x = x + m) →
  (∀ x ∈ Set.Icc 2 3, |f x - g x| ≤ 10) →
  15 ≤ m ∧ m ≤ 19 := by
sorry

end intimate_functions_range_l1107_110727


namespace largest_area_cross_section_passes_through_center_exists_larger_radius_non_center_cross_section_l1107_110718

-- Define a convex centrally symmetric polyhedron
structure ConvexCentrallySymmetricPolyhedron where
  -- Add necessary fields and properties
  is_convex : Bool
  is_centrally_symmetric : Bool

-- Define a cross-section of the polyhedron
structure CrossSection where
  polyhedron : ConvexCentrallySymmetricPolyhedron
  plane : Plane
  passes_through_center : Bool

-- Define the area of a cross-section
def area (cs : CrossSection) : ℝ := sorry

-- Define the radius of the smallest enclosing circle of a cross-section
def smallest_enclosing_circle_radius (cs : CrossSection) : ℝ := sorry

-- Theorem 1: The cross-section with the largest area passes through the center
theorem largest_area_cross_section_passes_through_center 
  (p : ConvexCentrallySymmetricPolyhedron) :
  ∀ (cs : CrossSection), cs.polyhedron = p → 
    ∃ (center_cs : CrossSection), 
      center_cs.polyhedron = p ∧ 
      center_cs.passes_through_center = true ∧
      area center_cs ≥ area cs :=
sorry

-- Theorem 2: There exists a cross-section not passing through the center with a larger 
-- radius of the smallest enclosing circle than the cross-section passing through the center
theorem exists_larger_radius_non_center_cross_section 
  (p : ConvexCentrallySymmetricPolyhedron) :
  ∃ (cs_non_center cs_center : CrossSection), 
    cs_non_center.polyhedron = p ∧ 
    cs_center.polyhedron = p ∧
    cs_non_center.passes_through_center = false ∧
    cs_center.passes_through_center = true ∧
    smallest_enclosing_circle_radius cs_non_center > smallest_enclosing_circle_radius cs_center :=
sorry

end largest_area_cross_section_passes_through_center_exists_larger_radius_non_center_cross_section_l1107_110718


namespace binomial_unique_parameters_l1107_110741

/-- A random variable following a binomial distribution -/
structure BinomialRV where
  n : ℕ
  p : ℝ
  h1 : 0 ≤ p ∧ p ≤ 1

/-- The expected value of a binomial random variable -/
def expectation (ξ : BinomialRV) : ℝ := ξ.n * ξ.p

/-- The variance of a binomial random variable -/
def variance (ξ : BinomialRV) : ℝ := ξ.n * ξ.p * (1 - ξ.p)

theorem binomial_unique_parameters :
  ∀ ξ : BinomialRV, expectation ξ = 12 → variance ξ = 2.4 → ξ.n = 15 ∧ ξ.p = 0.8 := by
  sorry

end binomial_unique_parameters_l1107_110741


namespace sine_cosine_sum_equals_one_l1107_110762

theorem sine_cosine_sum_equals_one :
  Real.sin (15 * π / 180) * Real.cos (75 * π / 180) + 
  Real.cos (15 * π / 180) * Real.sin (105 * π / 180) = 1 := by
  sorry

end sine_cosine_sum_equals_one_l1107_110762


namespace incorrect_statement_l1107_110759

theorem incorrect_statement : ¬ (∀ x : ℝ, |x| = x ↔ x = 0 ∨ x = 1) := by
  sorry

end incorrect_statement_l1107_110759


namespace friends_receiving_pebbles_l1107_110790

def pebbles_per_dozen : ℕ := 12

theorem friends_receiving_pebbles 
  (total_dozens : ℕ) 
  (pebbles_per_friend : ℕ) 
  (h1 : total_dozens = 3) 
  (h2 : pebbles_per_friend = 4) : 
  (total_dozens * pebbles_per_dozen) / pebbles_per_friend = 9 := by
  sorry

end friends_receiving_pebbles_l1107_110790


namespace remainder_theorem_polynomial_remainder_l1107_110784

def f (x : ℝ) : ℝ := x^5 - 8*x^4 + 20*x^3 + x^2 - 47*x + 15

theorem remainder_theorem (f : ℝ → ℝ) (a : ℝ) :
  ∃ q : ℝ → ℝ, ∀ x, f x = (x - a) * q x + f a :=
sorry

theorem polynomial_remainder : 
  ∃ q : ℝ → ℝ, ∀ x, f x = (x - 2) * q x + (-11) :=
sorry

end remainder_theorem_polynomial_remainder_l1107_110784


namespace sand_weight_formula_l1107_110764

/-- Given a number of bags n, where each full bag contains 65 pounds of sand,
    and one bag is not full containing 42 pounds of sand,
    the total weight of sand W is (n-1) * 65 + 42 pounds. -/
theorem sand_weight_formula (n : ℕ) (W : ℕ) : W = (n - 1) * 65 + 42 :=
by sorry

end sand_weight_formula_l1107_110764


namespace equation_has_four_solutions_l1107_110772

/-- The number of integer solutions to the equation 6y² + 3xy + x + 2y - 72 = 0 -/
def num_solutions : ℕ := 4

/-- The equation 6y² + 3xy + x + 2y - 72 = 0 -/
def equation (x y : ℤ) : Prop :=
  6 * y^2 + 3 * x * y + x + 2 * y - 72 = 0

theorem equation_has_four_solutions :
  ∃! (s : Finset (ℤ × ℤ)), s.card = num_solutions ∧ 
  ∀ (p : ℤ × ℤ), p ∈ s ↔ equation p.1 p.2 := by
  sorry

end equation_has_four_solutions_l1107_110772


namespace linda_total_sales_eq_366_9_l1107_110769

/-- Calculates the total sales for Linda's store given the following conditions:
  * Jeans are sold at $22 each
  * Tees are sold at $15 each
  * Jackets are sold at $37 each
  * 10% discount on jackets during the first half of the day
  * 7 tees sold
  * 4 jeans sold
  * 5 jackets sold in total
  * 3 jackets sold during the discount period
-/
def lindaTotalSales : ℝ :=
  let jeanPrice : ℝ := 22
  let teePrice : ℝ := 15
  let jacketPrice : ℝ := 37
  let jacketDiscount : ℝ := 0.1
  let teesSold : ℕ := 7
  let jeansSold : ℕ := 4
  let jacketsSold : ℕ := 5
  let discountedJackets : ℕ := 3
  let fullPriceJackets : ℕ := jacketsSold - discountedJackets
  let discountedJacketPrice : ℝ := jacketPrice * (1 - jacketDiscount)
  
  jeanPrice * jeansSold +
  teePrice * teesSold +
  jacketPrice * fullPriceJackets +
  discountedJacketPrice * discountedJackets

/-- Theorem stating that Linda's total sales at the end of the day equal $366.9 -/
theorem linda_total_sales_eq_366_9 : lindaTotalSales = 366.9 := by
  sorry

end linda_total_sales_eq_366_9_l1107_110769


namespace syrup_dilution_l1107_110788

theorem syrup_dilution (x : ℝ) : 
  (0 < x) ∧ 
  (x < 1000) ∧ 
  ((1000 - 2*x) * (1000 - x) = 120000) → 
  x = 400 :=
by sorry

end syrup_dilution_l1107_110788


namespace no_isosceles_triangle_36_degree_l1107_110743

theorem no_isosceles_triangle_36_degree (a b : ℕ+) : ¬ ∃ θ : ℝ,
  θ = 36 * π / 180 ∧
  (a : ℝ) * ((5 : ℝ).sqrt - 1) / 2 = b :=
sorry

end no_isosceles_triangle_36_degree_l1107_110743


namespace hash_property_l1107_110755

/-- Operation # for non-negative integers -/
def hash (a b : ℕ) : ℕ := 4 * a^2 + 4 * b^2 + 8 * a * b

/-- Theorem stating that if a # b = 100, then (a + b) + 6 = 11 -/
theorem hash_property (a b : ℕ) (h : hash a b = 100) : (a + b) + 6 = 11 := by
  sorry

end hash_property_l1107_110755


namespace advanced_math_group_arrangements_l1107_110717

/-- The number of students in the advanced mathematics study group -/
def total_students : ℕ := 5

/-- The number of boys in the group -/
def num_boys : ℕ := 3

/-- The number of girls in the group -/
def num_girls : ℕ := 2

/-- Student A -/
def student_A : ℕ := 1

/-- Student B -/
def student_B : ℕ := 2

/-- The number of arrangements where A and B must stand next to each other -/
def arrangements_adjacent : ℕ := 48

/-- The number of arrangements where A and B must not stand next to each other -/
def arrangements_not_adjacent : ℕ := 72

/-- The number of arrangements where A cannot stand at the far left and B cannot stand at the far right -/
def arrangements_restricted : ℕ := 78

theorem advanced_math_group_arrangements :
  (total_students = num_boys + num_girls) ∧
  (arrangements_adjacent = 48) ∧
  (arrangements_not_adjacent = 72) ∧
  (arrangements_restricted = 78) := by
  sorry

end advanced_math_group_arrangements_l1107_110717


namespace triangle_construction_with_two_angles_and_perimeter_l1107_110707

-- Define a triangle type
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  angleA : ℝ
  angleB : ℝ
  angleC : ℝ

-- Define perimeter
def perimeter (t : Triangle) : ℝ := t.a + t.b + t.c

-- Theorem statement
theorem triangle_construction_with_two_angles_and_perimeter 
  (A B P : ℝ) 
  (h_angles : 0 < A ∧ 0 < B ∧ A + B < π) 
  (h_perimeter : P > 0) :
  ∃ (t : Triangle), 
    t.angleA = A ∧ 
    t.angleB = B ∧ 
    perimeter t = P := by
  sorry

end triangle_construction_with_two_angles_and_perimeter_l1107_110707


namespace cylinder_prism_pyramid_elements_l1107_110705

/-- Represents a cylinder unwrapped into a prism with a pyramid attached -/
structure CylinderPrismPyramid where
  /-- Number of faces in the original prism -/
  prism_faces : ℕ
  /-- Number of edges in the original prism -/
  prism_edges : ℕ
  /-- Number of vertices in the original prism -/
  prism_vertices : ℕ
  /-- Number of faces added by the pyramid -/
  pyramid_faces : ℕ
  /-- Number of edges added by the pyramid -/
  pyramid_edges : ℕ
  /-- Number of vertices added by the pyramid -/
  pyramid_vertices : ℕ

/-- The total number of geometric elements in the CylinderPrismPyramid -/
def total_elements (cpp : CylinderPrismPyramid) : ℕ :=
  cpp.prism_faces + cpp.prism_edges + cpp.prism_vertices +
  cpp.pyramid_faces + cpp.pyramid_edges + cpp.pyramid_vertices

/-- Theorem stating that the total number of elements is 31 -/
theorem cylinder_prism_pyramid_elements :
  ∀ cpp : CylinderPrismPyramid,
  cpp.prism_faces = 5 ∧ 
  cpp.prism_edges = 10 ∧ 
  cpp.prism_vertices = 8 ∧
  cpp.pyramid_faces = 3 ∧
  cpp.pyramid_edges = 4 ∧
  cpp.pyramid_vertices = 1 →
  total_elements cpp = 31 := by
  sorry

#check cylinder_prism_pyramid_elements

end cylinder_prism_pyramid_elements_l1107_110705


namespace ice_cream_flavors_l1107_110738

theorem ice_cream_flavors (cone_types : ℕ) (total_combinations : ℕ) (h1 : cone_types = 2) (h2 : total_combinations = 8) :
  total_combinations / cone_types = 4 := by
  sorry

end ice_cream_flavors_l1107_110738


namespace sandbox_side_length_l1107_110767

/-- Represents the properties of a square sandbox. -/
structure Sandbox where
  sandPerArea : Real  -- Pounds of sand per square inch
  totalSand : Real    -- Total pounds of sand needed
  sideLength : Real   -- Length of each side in inches

/-- 
Theorem: Given a square sandbox where 30 pounds of sand fills 80 square inches,
and 600 pounds of sand fills the entire sandbox, the length of each side is 40 inches.
-/
theorem sandbox_side_length (sb : Sandbox)
  (h1 : sb.sandPerArea = 30 / 80)
  (h2 : sb.totalSand = 600) :
  sb.sideLength = 40 := by
  sorry


end sandbox_side_length_l1107_110767


namespace max_female_to_male_ratio_l1107_110796

/-- Proves that the maximum ratio of female to male students is 4:1 given the problem conditions -/
theorem max_female_to_male_ratio :
  ∀ (female_count male_count bench_count : ℕ),
  male_count = 29 →
  bench_count = 29 →
  ∃ (x : ℕ), female_count = x * male_count →
  female_count + male_count ≤ bench_count * 5 →
  x ≤ 4 :=
by sorry

end max_female_to_male_ratio_l1107_110796


namespace product_xyz_l1107_110714

theorem product_xyz (x y z : ℝ) 
  (h1 : x + 1/y = 2) 
  (h2 : y + 1/z = 2) : 
  x * y * z = -1 := by sorry

end product_xyz_l1107_110714


namespace tree_planting_problem_l1107_110771

theorem tree_planting_problem (total_trees : ℕ) 
  (h1 : 205000 ≤ total_trees ∧ total_trees ≤ 205300) 
  (h2 : ∃ (x y : ℕ), x > 0 ∧ y > 0 ∧ 7 * (x - 1) = total_trees ∧ 13 * (y - 1) = total_trees) : 
  ∃ (students : ℕ), students = 62 ∧ 
    ∃ (x y : ℕ), x > 0 ∧ y > 0 ∧ x + y = students ∧ 
      7 * (x - 1) = total_trees ∧ 13 * (y - 1) = total_trees :=
by sorry

end tree_planting_problem_l1107_110771


namespace function_expression_l1107_110700

noncomputable def f (x : ℝ) (ω : ℝ) (φ : ℝ) : ℝ := Real.sqrt 3 * Real.sin (ω * x + φ)

theorem function_expression 
  (ω : ℝ) 
  (φ : ℝ) 
  (h_ω : ω > 0) 
  (h_φ : -π/2 < φ ∧ φ < π/2) 
  (h_symmetry : f (1/3) ω φ = 0) 
  (h_amplitude : ∃ (x y : ℝ), f x ω φ - f y ω φ = 4) :
  ∀ x, f x ω φ = Real.sqrt 3 * Real.sin (π/2 * x - π/6) :=
sorry

end function_expression_l1107_110700


namespace smallest_sum_of_coefficients_l1107_110795

theorem smallest_sum_of_coefficients (a b : ℝ) : 
  a > 0 → b > 0 → 
  (∃ x : ℝ, x^2 + a*x + 2*b = 0) → 
  (∃ x : ℝ, x^2 + 2*b*x + a = 0) → 
  (∀ a' b' : ℝ, a' > 0 → b' > 0 → 
    (∃ x : ℝ, x^2 + a'*x + 2*b' = 0) → 
    (∃ x : ℝ, x^2 + 2*b'*x + a' = 0) → 
    a' + b' ≥ a + b) → 
  a + b = 6 :=
by sorry

end smallest_sum_of_coefficients_l1107_110795


namespace complex_expression_value_l1107_110732

theorem complex_expression_value : 
  let expr := (4.7 * 13.26 + 4.7 * 9.43 + 4.7 * 77.31) * Real.exp 3.5 + Real.log (Real.sin 0.785)
  ∃ ε > 0, |expr - 15563.91492641| < ε :=
by sorry

end complex_expression_value_l1107_110732


namespace rhombus_60_min_rotation_l1107_110776

/-- A rhombus with a 60° angle -/
structure Rhombus60 where
  /-- The rhombus has a 60° angle -/
  angle_60 : ∃ θ, θ = 60

/-- Minimum rotation for a Rhombus60 to coincide with its original position -/
def min_rotation (r : Rhombus60) : ℝ :=
  180

/-- Theorem: The minimum rotation for a Rhombus60 to coincide with its original position is 180° -/
theorem rhombus_60_min_rotation (r : Rhombus60) :
  min_rotation r = 180 := by
  sorry

end rhombus_60_min_rotation_l1107_110776


namespace xiaohong_mother_age_l1107_110798

/-- Xiaohong's age when her mother was her current age -/
def xiaohong_age_then : ℕ := 3

/-- Xiaohong's mother's future age when Xiaohong will be her mother's current age -/
def mother_age_future : ℕ := 78

/-- The age difference between Xiaohong and her mother -/
def age_difference : ℕ := mother_age_future - xiaohong_age_then

/-- Xiaohong's current age -/
def xiaohong_age_now : ℕ := age_difference + xiaohong_age_then

/-- Xiaohong's mother's current age -/
def mother_age_now : ℕ := mother_age_future - age_difference

theorem xiaohong_mother_age : mother_age_now = 53 := by
  sorry

#eval mother_age_now

end xiaohong_mother_age_l1107_110798


namespace food_drive_problem_l1107_110728

/-- Represents the food drive problem in Ms. Perez's class -/
theorem food_drive_problem (total_students : ℕ) (total_cans : ℕ) 
  (students_with_four_cans : ℕ) (students_with_zero_cans : ℕ) :
  total_students = 30 →
  total_cans = 232 →
  students_with_four_cans = 13 →
  students_with_zero_cans = 2 →
  2 * (total_students - students_with_four_cans - students_with_zero_cans) = total_students →
  (total_cans - 4 * students_with_four_cans) / 
    (total_students - students_with_four_cans - students_with_zero_cans) = 12 :=
by sorry

end food_drive_problem_l1107_110728


namespace solution_exists_for_a_in_range_l1107_110751

/-- The system of equations has a solution for a given 'a' -/
def has_solution (a : ℝ) : Prop :=
  ∃ b x y : ℝ, x^2 + y^2 + 2*a*(a + y - x) = 49 ∧ y = 8 / ((x - b)^2 + 1)

/-- The theorem stating the range of 'a' for which the system has a solution -/
theorem solution_exists_for_a_in_range :
  ∀ a : ℝ, -15 ≤ a ∧ a < 7 → has_solution a :=
sorry

end solution_exists_for_a_in_range_l1107_110751


namespace billie_baking_days_l1107_110757

/-- The number of days Billie bakes pumpkin pies -/
def days_baking : ℕ := 11

/-- The number of pies Billie bakes per day -/
def pies_per_day : ℕ := 3

/-- The number of cans of whipped cream needed to cover one pie -/
def cans_per_pie : ℕ := 2

/-- The number of pies eaten -/
def pies_eaten : ℕ := 4

/-- The number of cans of whipped cream needed for the remaining pies -/
def cans_needed : ℕ := 58

theorem billie_baking_days :
  days_baking * pies_per_day - pies_eaten = cans_needed / cans_per_pie := by sorry

end billie_baking_days_l1107_110757


namespace x_equals_two_l1107_110733

-- Define the * operation
def star (a b : ℕ) : ℕ := 
  Finset.sum (Finset.range b) (λ i => a + i)

-- State the theorem
theorem x_equals_two : 
  ∃ x : ℕ, star x 10 = 65 ∧ x = 2 :=
by sorry

end x_equals_two_l1107_110733


namespace min_distance_curve_to_line_l1107_110785

/-- The minimum distance from a point on y = e^(2x) to the line 2x - y - 4 = 0 -/
theorem min_distance_curve_to_line : 
  let f : ℝ → ℝ := fun x ↦ Real.exp (2 * x)
  let l : ℝ → ℝ → ℝ := fun x y ↦ 2 * x - y - 4
  let d : ℝ → ℝ := fun x ↦ |l x (f x)| / Real.sqrt 5
  ∃ (x_min : ℝ), ∀ (x : ℝ), d x_min ≤ d x ∧ d x_min = 4 * Real.sqrt 5 / 5 := by
sorry

end min_distance_curve_to_line_l1107_110785


namespace article_cost_proof_l1107_110712

theorem article_cost_proof (sp1 sp2 : ℝ) (gain_percentage : ℝ) :
  sp1 = 348 ∧ sp2 = 350 ∧ gain_percentage = 0.05 →
  ∃ (cost gain : ℝ),
    sp1 = cost + gain ∧
    sp2 = cost + gain + gain_percentage * gain ∧
    cost = 308 :=
by sorry

end article_cost_proof_l1107_110712


namespace roots_log_sum_l1107_110758

-- Define the equation
def equation (x : ℝ) : Prop := (Real.log x)^2 - Real.log (x^2) = 2

-- Define α and β as the roots of the equation
axiom α : ℝ
axiom β : ℝ
axiom α_pos : α > 0
axiom β_pos : β > 0
axiom α_root : equation α
axiom β_root : equation β

-- State the theorem
theorem roots_log_sum : Real.log β / Real.log α + Real.log α / Real.log β = -4 := by
  sorry

end roots_log_sum_l1107_110758


namespace integer_division_problem_l1107_110719

theorem integer_division_problem (D d q r : ℤ) 
  (h1 : D = q * d + r) 
  (h2 : D + 65 = q * (d + 5) + r) : q = 13 := by
  sorry

end integer_division_problem_l1107_110719


namespace max_digit_sum_three_digit_number_l1107_110731

theorem max_digit_sum_three_digit_number (a b c : ℕ) : 
  a < 10 → b < 10 → c < 10 →
  (100 * a + 10 * b + c) + (100 * a + 10 * c + b) = 1732 →
  a + b + c ≤ 20 := by
sorry

end max_digit_sum_three_digit_number_l1107_110731


namespace unique_base_for_special_palindrome_l1107_110745

def is_palindrome (n : ℕ) (base : ℕ) : Prop :=
  ∃ (digits : List ℕ), digits.length > 0 ∧ 
    (digits.reverse = digits) ∧ 
    (n = digits.foldl (λ acc d => acc * base + d) 0)

theorem unique_base_for_special_palindrome : 
  ∃! (r : ℕ), 
    r % 2 = 0 ∧ 
    r ≥ 18 ∧ 
    (∃ (x : ℕ), 
      x = 5 * r^3 + 5 * r^2 + 5 * r + 5 ∧
      is_palindrome (x^2) r ∧
      (∃ (a b c d : ℕ), 
        x^2 = a * r^7 + b * r^6 + c * r^5 + d * r^4 + 
              d * r^3 + c * r^2 + b * r + a ∧
        d - c = 2)) ∧
    r = 24 :=
sorry

end unique_base_for_special_palindrome_l1107_110745


namespace journey_distance_l1107_110774

/-- Given a journey that takes 3 hours and can be completed in half the time
    at a speed of 293.3333333333333 kmph, prove that the distance traveled is 440 km. -/
theorem journey_distance (original_time : ℝ) (new_speed : ℝ) (distance : ℝ) : 
  original_time = 3 →
  new_speed = 293.3333333333333 →
  distance = new_speed * (original_time / 2) →
  distance = 440 := by
  sorry

end journey_distance_l1107_110774


namespace linear_function_relationship_l1107_110749

/-- A linear function f(x) = 2x + 1 -/
def f (x : ℝ) : ℝ := 2 * x + 1

theorem linear_function_relationship (y₁ y₂ : ℝ) 
  (h1 : f (-3) = y₁) 
  (h2 : f 4 = y₂) : 
  y₁ < y₂ := by
sorry

end linear_function_relationship_l1107_110749


namespace number_problem_l1107_110737

theorem number_problem (x y : ℝ) : 
  (x^2)/2 + 5*y = 15 ∧ x + y = 10 → x = 5 ∧ y = 5 := by
  sorry

end number_problem_l1107_110737


namespace passengers_from_other_continents_l1107_110748

theorem passengers_from_other_continents : 
  ∀ (total : ℕ) (north_america europe africa asia other : ℚ),
    total = 96 →
    north_america = 1/4 →
    europe = 1/8 →
    africa = 1/12 →
    asia = 1/6 →
    other = 1 - (north_america + europe + africa + asia) →
    (other * total : ℚ) = 36 := by
  sorry

end passengers_from_other_continents_l1107_110748


namespace unique_A_for_club_equation_l1107_110702

-- Define the ♣ operation
def club (A B : ℝ) : ℝ := 4 * A + 2 * B + 6

-- Theorem statement
theorem unique_A_for_club_equation : ∃! A : ℝ, club A 6 = 70 ∧ A = 13 := by
  sorry

end unique_A_for_club_equation_l1107_110702


namespace point_in_fourth_quadrant_l1107_110780

theorem point_in_fourth_quadrant :
  let P : ℝ × ℝ := (Real.tan (2015 * π / 180), Real.cos (2015 * π / 180))
  (0 < P.1) ∧ (P.2 < 0) :=
by sorry

end point_in_fourth_quadrant_l1107_110780


namespace train_crossing_time_l1107_110778

/-- Proves that a train with the given length and speed takes the calculated time to cross an electric pole -/
theorem train_crossing_time (train_length : Real) (train_speed_kmh : Real) : 
  train_length = 120 ∧ train_speed_kmh = 72 → 
  (train_length / (train_speed_kmh * 1000 / 3600)) = 6 := by
  sorry

#check train_crossing_time

end train_crossing_time_l1107_110778


namespace total_saltwater_animals_l1107_110775

/-- The number of aquariums -/
def num_aquariums : ℕ := 20

/-- The number of animals per aquarium -/
def animals_per_aquarium : ℕ := 2

/-- Theorem stating the total number of saltwater animals -/
theorem total_saltwater_animals : 
  num_aquariums * animals_per_aquarium = 40 := by
  sorry

end total_saltwater_animals_l1107_110775


namespace sets_A_and_B_proof_l1107_110765

def U : Set Nat := {x | x ≤ 20 ∧ Nat.Prime x}

theorem sets_A_and_B_proof (A B : Set Nat) 
  (h1 : A ∩ (U \ B) = {3, 5})
  (h2 : B ∩ (U \ A) = {7, 19})
  (h3 : (U \ A) ∩ (U \ B) = {2, 17}) :
  A = {3, 5, 11, 13} ∧ B = {7, 11, 13, 19} := by
sorry

end sets_A_and_B_proof_l1107_110765


namespace trigonometric_equality_l1107_110726

theorem trigonometric_equality (α β : ℝ) 
  (h : (Real.cos α)^6 / (Real.cos β)^3 + (Real.sin α)^6 / (Real.sin β)^3 = 1) :
  (Real.sin β)^6 / (Real.sin α)^3 + (Real.cos β)^6 / (Real.cos α)^3 = 1 := by
  sorry

end trigonometric_equality_l1107_110726


namespace mango_rate_proof_l1107_110761

def grape_quantity : ℕ := 10
def grape_rate : ℕ := 70
def mango_quantity : ℕ := 9
def total_paid : ℕ := 1195

theorem mango_rate_proof :
  (total_paid - grape_quantity * grape_rate) / mango_quantity = 55 := by
  sorry

end mango_rate_proof_l1107_110761


namespace sqrt_3_times_sqrt_12_l1107_110781

theorem sqrt_3_times_sqrt_12 : Real.sqrt 3 * Real.sqrt 12 = 6 := by
  sorry

end sqrt_3_times_sqrt_12_l1107_110781


namespace dog_treat_expenditure_l1107_110777

/-- Calculate John's total expenditure on dog treats for a month --/
theorem dog_treat_expenditure :
  let treats_first_half : ℕ := 3 * 15
  let treats_second_half : ℕ := 4 * 15
  let total_treats : ℕ := treats_first_half + treats_second_half
  let original_price : ℚ := 0.1
  let discount_threshold : ℕ := 50
  let discount_rate : ℚ := 0.1
  let discounted_price : ℚ := original_price * (1 - discount_rate)
  total_treats > discount_threshold →
  (total_treats : ℚ) * discounted_price = 9.45 :=
by sorry

end dog_treat_expenditure_l1107_110777


namespace rectangle_square_division_l1107_110742

theorem rectangle_square_division (n : ℕ) : 
  (∃ (a b : ℚ), a > 0 ∧ b > 0 ∧ 
    (∃ (p q : ℕ), p > q ∧ 
      (a * b / n).sqrt / (a * b / (n + 76)).sqrt = p / q)) → 
  n = 324 :=
by sorry

end rectangle_square_division_l1107_110742


namespace ab_value_l1107_110709

theorem ab_value (a b : ℝ) (h : |a + 3| + (b - 2)^2 = 0) : a^b = 9 := by
  sorry

end ab_value_l1107_110709


namespace percentage_problem_l1107_110715

theorem percentage_problem (P : ℝ) : 
  (0.15 * (P / 100) * 0.5 * 5200 = 117) → P = 30 := by
sorry

end percentage_problem_l1107_110715


namespace solution_set_m_zero_solution_set_all_reals_l1107_110725

-- Define the inequality function
def f (m : ℝ) (x : ℝ) : ℝ := (m - 1) * x^2 + (m - 1) * x + 2

-- Part 1: Solution set when m = 0
theorem solution_set_m_zero :
  {x : ℝ | f 0 x > 0} = {x : ℝ | -2 < x ∧ x < 1} :=
sorry

-- Part 2: Range of m when solution set is ℝ
theorem solution_set_all_reals (m : ℝ) :
  ({x : ℝ | f m x > 0} = Set.univ) ↔ (1 < m ∧ m < 9) :=
sorry

end solution_set_m_zero_solution_set_all_reals_l1107_110725


namespace bicycle_selling_price_l1107_110730

/-- The final selling price of a bicycle given initial cost and profit percentages -/
theorem bicycle_selling_price 
  (initial_cost : ℝ) 
  (profit_a_percent : ℝ) 
  (profit_b_percent : ℝ) 
  (h1 : initial_cost = 150)
  (h2 : profit_a_percent = 20)
  (h3 : profit_b_percent = 25) : 
  initial_cost * (1 + profit_a_percent / 100) * (1 + profit_b_percent / 100) = 225 := by
  sorry

end bicycle_selling_price_l1107_110730


namespace days_to_complete_correct_l1107_110708

/-- The number of days required for a given number of men to complete a work,
    given that 12 men can do it in 80 days and 16 men can do it in 60 days. -/
def days_to_complete (num_men : ℕ) : ℚ :=
  960 / num_men

/-- Theorem stating that the number of days required for any number of men
    to complete the work is correctly given by the days_to_complete function,
    based on the given conditions. -/
theorem days_to_complete_correct (num_men : ℕ) (num_men_pos : 0 < num_men) :
  days_to_complete num_men * num_men = 960 ∧
  days_to_complete 12 = 80 ∧
  days_to_complete 16 = 60 :=
by sorry

end days_to_complete_correct_l1107_110708


namespace m_minus_n_eq_neg_reals_l1107_110768

-- Define the sets M and N
def M : Set ℝ := {x | ∃ y, y = Real.sqrt (1 - x) ∧ -1 ≤ x ∧ x ≤ 1}
def N : Set ℝ := {y | ∃ x, y = x^2 ∧ -1 ≤ x ∧ x ≤ 1}

-- Define the set difference operation
def setDifference (A B : Set ℝ) : Set ℝ := {x | x ∈ A ∧ x ∉ B}

-- State the theorem
theorem m_minus_n_eq_neg_reals : 
  setDifference M N = {x : ℝ | x < 0} := by sorry

end m_minus_n_eq_neg_reals_l1107_110768


namespace weekly_earnings_is_1454000_l1107_110752

/-- Represents the days of the week -/
inductive Day
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday
  | Sunday

/-- Returns the number of computers produced on a given day -/
def production_rate (d : Day) : ℕ :=
  match d with
  | Day.Monday    => 1200
  | Day.Tuesday   => 1500
  | Day.Wednesday => 1800
  | Day.Thursday  => 1600
  | Day.Friday    => 1400
  | Day.Saturday  => 1000
  | Day.Sunday    => 800

/-- Returns the selling price per computer on a given day -/
def selling_price (d : Day) : ℕ :=
  match d with
  | Day.Monday    => 150
  | Day.Tuesday   => 160
  | Day.Wednesday => 170
  | Day.Thursday  => 155
  | Day.Friday    => 145
  | Day.Saturday  => 165
  | Day.Sunday    => 140

/-- Calculates the earnings for a given day -/
def daily_earnings (d : Day) : ℕ :=
  production_rate d * selling_price d

/-- Calculates the total earnings for the week -/
def total_weekly_earnings : ℕ :=
  daily_earnings Day.Monday +
  daily_earnings Day.Tuesday +
  daily_earnings Day.Wednesday +
  daily_earnings Day.Thursday +
  daily_earnings Day.Friday +
  daily_earnings Day.Saturday +
  daily_earnings Day.Sunday

/-- Theorem stating that the total weekly earnings is $1,454,000 -/
theorem weekly_earnings_is_1454000 :
  total_weekly_earnings = 1454000 := by
  sorry

end weekly_earnings_is_1454000_l1107_110752


namespace twenty_fives_sum_1000_l1107_110793

/-- A list of integers representing a grouping of fives -/
def Grouping : Type := List Nat

/-- The number of fives in a grouping -/
def count_fives : Grouping → Nat
  | [] => 0
  | (x::xs) => (x.digits 10).length + count_fives xs

/-- The sum of a grouping -/
def sum_grouping : Grouping → Nat
  | [] => 0
  | (x::xs) => x + sum_grouping xs

/-- A valid grouping of 20 fives that sums to 1000 -/
theorem twenty_fives_sum_1000 : ∃ (g : Grouping), 
  (count_fives g = 20) ∧ (sum_grouping g = 1000) := by
  sorry

end twenty_fives_sum_1000_l1107_110793


namespace school_teachers_count_l1107_110766

theorem school_teachers_count (total : ℕ) (sample_size : ℕ) (sampled_students : ℕ) :
  total = 2400 →
  sample_size = 160 →
  sampled_students = 150 →
  total - (total * sampled_students / sample_size) = 150 :=
by sorry

end school_teachers_count_l1107_110766


namespace eric_park_time_ratio_l1107_110754

/-- The ratio of Eric's return time to his time to reach the park is 3:1 -/
theorem eric_park_time_ratio :
  let time_to_park : ℕ := 20 + 10  -- Time to reach the park (running + jogging)
  let time_to_return : ℕ := 90     -- Time to return home
  (time_to_return : ℚ) / time_to_park = 3 / 1 := by
  sorry

end eric_park_time_ratio_l1107_110754


namespace workshop_pairing_probability_l1107_110787

/-- The probability of a specific pairing in a group of participants. -/
def specific_pairing_probability (total_participants : ℕ) : ℚ :=
  if total_participants ≤ 1 then 0
  else 1 / (total_participants - 1 : ℚ)

/-- Theorem: In a workshop with 24 participants, the probability of John pairing with Alice is 1/23. -/
theorem workshop_pairing_probability :
  specific_pairing_probability 24 = 1 / 23 := by
  sorry


end workshop_pairing_probability_l1107_110787


namespace people_on_boats_l1107_110704

theorem people_on_boats (total_boats : Nat) (boats_with_four : Nat) (boats_with_five : Nat)
  (h1 : total_boats = 7)
  (h2 : boats_with_four = 4)
  (h3 : boats_with_five = 3)
  (h4 : total_boats = boats_with_four + boats_with_five) :
  boats_with_four * 4 + boats_with_five * 5 = 31 := by
  sorry

end people_on_boats_l1107_110704


namespace beta_value_l1107_110760

open Real

def operation (a b c d : ℝ) : ℝ := a * d - b * c

theorem beta_value (α β : ℝ) : 
  cos α = 1/7 →
  operation (sin α) (sin β) (cos α) (cos β) = 3 * Real.sqrt 3 / 14 →
  0 < β →
  β < α →
  α < π/2 →
  β = π/3 := by sorry

end beta_value_l1107_110760


namespace series_duration_l1107_110783

theorem series_duration (episode1 episode2 episode3 episode4 : ℕ) 
  (h1 : episode1 = 58)
  (h2 : episode2 = 62)
  (h3 : episode3 = 65)
  (h4 : episode4 = 55) :
  (episode1 + episode2 + episode3 + episode4) / 60 = 4 := by
  sorry

end series_duration_l1107_110783


namespace roundness_of_eight_million_l1107_110735

def roundness (n : ℕ) : ℕ := sorry

theorem roundness_of_eight_million : roundness 8000000 = 15 := by sorry

end roundness_of_eight_million_l1107_110735


namespace linear_function_decreasing_l1107_110713

theorem linear_function_decreasing (k : ℝ) :
  (∀ x y : ℝ, x < y → ((k + 2) * x + 1) > ((k + 2) * y + 1)) ↔ k < -2 := by
  sorry

end linear_function_decreasing_l1107_110713


namespace volume_maximized_at_height_1_2_l1107_110797

/-- Represents the dimensions of a rectangular container frame -/
structure ContainerFrame where
  shortSide : ℝ
  longSide : ℝ
  height : ℝ

/-- Calculates the volume of a container given its dimensions -/
def volume (frame : ContainerFrame) : ℝ :=
  frame.shortSide * frame.longSide * frame.height

/-- Calculates the perimeter of a container given its dimensions -/
def perimeter (frame : ContainerFrame) : ℝ :=
  2 * (frame.shortSide + frame.longSide + frame.height)

/-- Theorem: The volume of the container is maximized when the height is 1.2 m -/
theorem volume_maximized_at_height_1_2 :
  ∃ (frame : ContainerFrame),
    frame.longSide = frame.shortSide + 0.5 ∧
    perimeter frame = 14.8 ∧
    ∀ (other : ContainerFrame),
      other.longSide = other.shortSide + 0.5 →
      perimeter other = 14.8 →
      volume other ≤ volume frame ∧
      frame.height = 1.2 := by
  sorry

end volume_maximized_at_height_1_2_l1107_110797


namespace two_digit_number_problem_l1107_110786

/-- Represents a two-digit number as a pair of natural numbers -/
def TwoDigitNumber := Nat × Nat

/-- Converts a two-digit number to its decimal representation -/
def toDecimal (n : TwoDigitNumber) : ℚ :=
  (n.1 : ℚ) / 10 + (n.2 : ℚ) / 100

/-- Converts a two-digit number to its repeating decimal representation -/
def toRepeatingDecimal (n : TwoDigitNumber) : ℚ :=
  1 + (n.1 : ℚ) / 10 + (n.2 : ℚ) / 100 + (n.1 : ℚ) / 1000 + (n.2 : ℚ) / 10000

theorem two_digit_number_problem (cd : TwoDigitNumber) :
  72 * toRepeatingDecimal cd - 72 * (1 + toDecimal cd) = 0.8 → cd = (1, 1) := by
  sorry

end two_digit_number_problem_l1107_110786


namespace greatest_power_of_two_factor_l1107_110723

theorem greatest_power_of_two_factor (n : ℕ) : 
  ∃ (k : ℕ), (2^459 : ℕ) ∣ (9^456 - 3^684) ∧ 
  ∀ m > 459, ¬((2^m : ℕ) ∣ (9^456 - 3^684)) := by
  sorry

end greatest_power_of_two_factor_l1107_110723


namespace bell_pepper_ratio_l1107_110753

/-- Represents the number of bell peppers --/
def num_peppers : ℕ := 5

/-- Represents the number of large slices per bell pepper --/
def slices_per_pepper : ℕ := 20

/-- Represents the total number of slices and pieces in the meal --/
def total_pieces : ℕ := 200

/-- Represents the number of smaller pieces each large slice is cut into --/
def pieces_per_slice : ℕ := 3

/-- Calculates the total number of large slices --/
def total_large_slices : ℕ := num_peppers * slices_per_pepper

/-- Theorem stating the ratio of large slices cut into smaller pieces to total large slices --/
theorem bell_pepper_ratio : 
  ∃ (x : ℕ), x * pieces_per_slice + (total_large_slices - x) = total_pieces ∧ 
             x = 33 ∧
             (x : ℚ) / (total_large_slices : ℚ) = 33 / 100 := by
  sorry

end bell_pepper_ratio_l1107_110753


namespace right_triangle_area_l1107_110740

/-- A line passing through two points -/
structure Line where
  point1 : ℝ × ℝ
  point2 : ℝ × ℝ

/-- A right triangle formed by x-axis, y-axis, and a line -/
structure RightTriangle where
  line : Line

/-- Calculate the area of a right triangle -/
def area (t : RightTriangle) : ℝ :=
  sorry

theorem right_triangle_area :
  let l := Line.mk (-4, 8) (-8, 4)
  let t := RightTriangle.mk l
  area t = 72 := by
  sorry

end right_triangle_area_l1107_110740


namespace ice_cream_sales_for_video_games_l1107_110750

theorem ice_cream_sales_for_video_games :
  let game_cost : ℕ := 60
  let ice_cream_price : ℕ := 5
  let num_games : ℕ := 2
  let total_cost : ℕ := game_cost * num_games
  let ice_creams_needed : ℕ := total_cost / ice_cream_price
  ice_creams_needed = 24 := by
sorry

end ice_cream_sales_for_video_games_l1107_110750


namespace simultaneous_equations_solution_l1107_110789

theorem simultaneous_equations_solution :
  ∃ (x y : ℚ), 3 * x - 4 * y = -2 ∧ 4 * x + 5 * y = 23 ∧ x = 82/31 ∧ y = 77/31 := by
  sorry

end simultaneous_equations_solution_l1107_110789


namespace geometric_ratio_is_four_l1107_110739

/-- An arithmetic sequence with a_1 = 2 and non-zero common difference -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  a 1 = 2 ∧ ∃ d ≠ 0, ∀ n, a (n + 1) = a n + d

/-- Three terms of an arithmetic sequence form a geometric sequence -/
def forms_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, q ≠ 0 ∧ a 3 = a 1 * q ∧ a 11 = a 1 * q^2

/-- The common ratio of the geometric sequence formed by a_1, a_3, and a_11 is 4 -/
theorem geometric_ratio_is_four
  (a : ℕ → ℝ)
  (h_arith : arithmetic_sequence a)
  (h_geom : forms_geometric_sequence a) :
  ∃ q : ℝ, q = 4 ∧ a 3 = a 1 * q ∧ a 11 = a 1 * q^2 := by
  sorry

end geometric_ratio_is_four_l1107_110739


namespace f_derivative_at_2_l1107_110721

/-- The function f(x) = x^3 + 2 -/
def f (x : ℝ) : ℝ := x^3 + 2

/-- Theorem: The derivative of f at x = 2 is equal to 12 -/
theorem f_derivative_at_2 : 
  deriv f 2 = 12 := by sorry

end f_derivative_at_2_l1107_110721


namespace clara_three_times_anna_age_l1107_110736

/-- Proves that Clara was three times Anna's age 41 years ago -/
theorem clara_three_times_anna_age : ∃ x : ℕ, x = 41 ∧ 
  (80 : ℝ) - x = 3 * ((54 : ℝ) - x) := by
  sorry

end clara_three_times_anna_age_l1107_110736


namespace indeterminate_equation_solutions_l1107_110746

theorem indeterminate_equation_solutions :
  ∀ x y : ℤ, 2 * (x + y) = x * y + 7 ↔ 
    (x = 3 ∧ y = -1) ∨ (x = 5 ∧ y = 1) ∨ (x = 1 ∧ y = 5) ∨ (x = -1 ∧ y = 3) := by
  sorry

end indeterminate_equation_solutions_l1107_110746


namespace age_inconsistency_l1107_110716

theorem age_inconsistency (a b c d : ℝ) : 
  (a + c + d) / 3 = 30 →
  (a + c) / 2 = 32 →
  (b + d) / 2 = 34 →
  ¬(0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d) :=
by
  sorry

#check age_inconsistency

end age_inconsistency_l1107_110716


namespace t_shape_perimeter_l1107_110722

/-- The perimeter of a T shape formed by two rectangles -/
def t_perimeter (width height overlap : ℝ) : ℝ :=
  2 * (width + height - 2 * overlap) + 2 * height

/-- Theorem: The perimeter of the T shape is 20 inches -/
theorem t_shape_perimeter :
  let width := 3
  let height := 5
  let overlap := 1.5
  t_perimeter width height overlap = 20 := by
sorry

#eval t_perimeter 3 5 1.5

end t_shape_perimeter_l1107_110722


namespace product_equality_l1107_110791

theorem product_equality : 3.6 * 0.4 * 1.5 = 2.16 := by
  sorry

end product_equality_l1107_110791


namespace gcd_sequence_a_odd_l1107_110747

def sequence_a (a₁ : ℤ) : ℕ → ℤ
  | 0 => a₁
  | n + 1 => (sequence_a a₁ n)^2 - (sequence_a a₁ n) - 1

theorem gcd_sequence_a_odd (a₁ : ℤ) (n : ℕ) :
  Nat.gcd (Int.natAbs (sequence_a a₁ (n + 1))) (2 * (n + 1) + 1) = 1 := by
  sorry

end gcd_sequence_a_odd_l1107_110747


namespace ellipse_fixed_point_intersection_l1107_110770

/-- Represents an ellipse with semi-major axis a and semi-minor axis b -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_ab : a > b
  h_b_pos : b > 0

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a line in 2D space -/
structure Line where
  m : ℝ  -- slope
  c : ℝ  -- y-intercept

theorem ellipse_fixed_point_intersection
  (C : Ellipse)
  (h_point : (1 : ℝ)^2 / C.a^2 + (Real.sqrt 6 / 3)^2 / C.b^2 = 1)
  (h_eccentricity : Real.sqrt (C.a^2 - C.b^2) / C.a = Real.sqrt 6 / 3)
  (l : Line)
  (P Q : Point)
  (h_intersect_P : P.x^2 / C.a^2 + P.y^2 / C.b^2 = 1)
  (h_intersect_Q : Q.x^2 / C.a^2 + Q.y^2 / C.b^2 = 1)
  (h_on_line_P : P.y = l.m * P.x + l.c)
  (h_on_line_Q : Q.y = l.m * Q.x + l.c)
  (h_perpendicular : P.x * Q.x + P.y * Q.y = 0)
  (h_not_vertex : l.m ≠ 0 ∨ l.c ≠ 1) :
  l.m * 0 + l.c = -1/2 := by sorry

end ellipse_fixed_point_intersection_l1107_110770


namespace inscribed_square_properties_l1107_110773

theorem inscribed_square_properties (circle_area : ℝ) (h : circle_area = 324 * Real.pi) :
  let r : ℝ := Real.sqrt (circle_area / Real.pi)
  let d : ℝ := 2 * r
  let s : ℝ := d / Real.sqrt 2
  let square_area : ℝ := s ^ 2
  let total_diagonal_length : ℝ := 2 * d
  (square_area = 648) ∧ (total_diagonal_length = 72) := by
  sorry

#check inscribed_square_properties

end inscribed_square_properties_l1107_110773


namespace house_cleaning_time_l1107_110724

theorem house_cleaning_time (sawyer_time nick_time joint_time : ℝ) 
  (h1 : sawyer_time / 2 = nick_time / 3)
  (h2 : joint_time = 3.6)
  (h3 : 1 / sawyer_time + 1 / nick_time = 1 / joint_time) :
  sawyer_time = 6 := by
sorry

end house_cleaning_time_l1107_110724


namespace sufficient_not_necessary_condition_l1107_110710

theorem sufficient_not_necessary_condition (a : ℝ) : 
  (∀ a, a > 1 → a^2 > 1) ∧ 
  (∃ a, a^2 > 1 ∧ ¬(a > 1)) :=
by sorry

end sufficient_not_necessary_condition_l1107_110710


namespace magical_green_knights_fraction_l1107_110744

theorem magical_green_knights_fraction (total : ℕ) (total_pos : 0 < total) :
  let green := total / 3
  let yellow := total - green
  let magical := total / 5
  let green_magical_fraction := magical_green / green
  let yellow_magical_fraction := magical_yellow / yellow
  green_magical_fraction = 3 * yellow_magical_fraction →
  magical_green + magical_yellow = magical →
  green_magical_fraction = 9 / 25 :=
by sorry

end magical_green_knights_fraction_l1107_110744


namespace value_of_a_l1107_110720

theorem value_of_a (a b : ℝ) (h1 : |a| = 5) (h2 : b = 4) (h3 : a < b) : a = -5 := by
  sorry

end value_of_a_l1107_110720


namespace num_divisors_5400_multiple_of_5_l1107_110792

/-- The number of positive divisors of 5400 that are multiples of 5 -/
def num_divisors_multiple_of_5 (n : ℕ) : ℕ :=
  (Finset.filter (λ d => d ∣ n ∧ 5 ∣ d) (Finset.range (n + 1))).card

/-- Theorem stating that the number of positive divisors of 5400 that are multiples of 5 is 24 -/
theorem num_divisors_5400_multiple_of_5 :
  num_divisors_multiple_of_5 5400 = 24 := by
  sorry

end num_divisors_5400_multiple_of_5_l1107_110792


namespace ellipse_focus_x_axis_l1107_110779

/-- 
Given an ellipse with equation x²/(1-k) + y²/(2+k) = 1,
if its focus lies on the x-axis, then k ∈ (-2, -1/2)
-/
theorem ellipse_focus_x_axis (k : ℝ) : 
  (∃ (x y : ℝ), x^2 / (1 - k) + y^2 / (2 + k) = 1) →
  (∃ (c : ℝ), c > 0 ∧ c^2 = (1 - k) - (2 + k)) →
  k ∈ Set.Ioo (-2 : ℝ) (-1/2 : ℝ) :=
by sorry

end ellipse_focus_x_axis_l1107_110779


namespace school_capacity_l1107_110711

theorem school_capacity (total_capacity : ℕ) (known_school_capacity : ℕ) (num_schools : ℕ) (num_known_schools : ℕ) :
  total_capacity = 1480 →
  known_school_capacity = 400 →
  num_schools = 4 →
  num_known_schools = 2 →
  (total_capacity - num_known_schools * known_school_capacity) / (num_schools - num_known_schools) = 340 := by
  sorry

end school_capacity_l1107_110711


namespace px_length_l1107_110794

-- Define the quadrilateral CDXW
structure Quadrilateral :=
  (C D W X P : ℝ × ℝ)
  (cd_parallel_wx : (D.1 - C.1) * (X.2 - W.2) = (D.2 - C.2) * (X.1 - W.1))
  (p_on_cx : ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ P = (C.1 + t * (X.1 - C.1), C.2 + t * (X.2 - C.2)))
  (p_on_dw : ∃ s : ℝ, 0 ≤ s ∧ s ≤ 1 ∧ P = (D.1 + s * (W.1 - D.1), D.2 + s * (W.2 - D.2)))
  (cx_length : Real.sqrt ((X.1 - C.1)^2 + (X.2 - C.2)^2) = 30)
  (dp_length : Real.sqrt ((P.1 - D.1)^2 + (P.2 - D.2)^2) = 15)
  (pw_length : Real.sqrt ((W.1 - P.1)^2 + (W.2 - P.2)^2) = 45)

-- Theorem statement
theorem px_length (q : Quadrilateral) : 
  Real.sqrt ((q.X.1 - q.P.1)^2 + (q.X.2 - q.P.2)^2) = 22.5 := by
  sorry

end px_length_l1107_110794


namespace books_borrowed_second_day_l1107_110706

def initial_books : ℕ := 100
def people_first_day : ℕ := 5
def books_per_person : ℕ := 2
def remaining_books : ℕ := 70

theorem books_borrowed_second_day :
  initial_books - people_first_day * books_per_person - remaining_books = 20 :=
by sorry

end books_borrowed_second_day_l1107_110706


namespace mass_of_man_on_boat_l1107_110763

/-- The mass of a man who causes a boat to sink by a certain amount -/
def mass_of_man (boat_length boat_breadth boat_sink_height water_density : ℝ) : ℝ :=
  boat_length * boat_breadth * boat_sink_height * water_density

/-- Theorem stating the mass of the man in the given problem -/
theorem mass_of_man_on_boat :
  let boat_length : ℝ := 8
  let boat_breadth : ℝ := 2
  let boat_sink_height : ℝ := 0.01  -- 1 cm in meters
  let water_density : ℝ := 1000     -- kg/m³
  mass_of_man boat_length boat_breadth boat_sink_height water_density = 160 := by
  sorry

#check mass_of_man_on_boat

end mass_of_man_on_boat_l1107_110763
