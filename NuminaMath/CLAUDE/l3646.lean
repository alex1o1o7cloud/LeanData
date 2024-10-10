import Mathlib

namespace triangle_theorem_l3646_364606

open Real

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle :=
  (a b c : ℝ)
  (A B C : ℝ)
  (h1 : 0 < a ∧ 0 < b ∧ 0 < c)
  (h2 : 0 < A ∧ 0 < B ∧ 0 < C)
  (h3 : A + B + C = π)
  (h4 : a * sin A = b * sin B)
  (h5 : b * sin B = c * sin C)
  (h6 : a * sin A - c * sin C = (a - b) * sin B)

/-- The theorem stating the angle C and maximum area of the triangle -/
theorem triangle_theorem (t : Triangle) (h : t.c = sqrt 6) :
  t.C = π / 3 ∧
  ∃ (S : ℝ), S = (3 * sqrt 3) / 2 ∧ ∀ (S' : ℝ), S' ≤ S := by
  sorry


end triangle_theorem_l3646_364606


namespace rhea_count_l3646_364633

theorem rhea_count (num_wombats : ℕ) (wombat_claws : ℕ) (rhea_claws : ℕ) (total_claws : ℕ) : 
  num_wombats = 9 →
  wombat_claws = 4 →
  rhea_claws = 1 →
  total_claws = 39 →
  total_claws = num_wombats * wombat_claws + (total_claws - num_wombats * wombat_claws) →
  (total_claws - num_wombats * wombat_claws) / rhea_claws = 3 := by
sorry

end rhea_count_l3646_364633


namespace cubic_root_property_l3646_364662

-- Define the cubic polynomial
def f (x : ℝ) : ℝ := x^3 - 3*x - 1

-- Define the roots and their properties
theorem cubic_root_property (x₁ x₂ x₃ : ℝ) 
  (h1 : f x₁ = 0) (h2 : f x₂ = 0) (h3 : f x₃ = 0)
  (h_distinct : x₁ ≠ x₂ ∧ x₂ ≠ x₃ ∧ x₁ ≠ x₃)
  (h_order : x₁ < x₂ ∧ x₂ < x₃) :
  x₃^2 - x₂^2 = x₃ - x₁ := by
sorry

end cubic_root_property_l3646_364662


namespace birth_interval_l3646_364688

/-- Given 5 children born at equal intervals, with the youngest being 6 years old
    and the sum of all ages being 60 years, the interval between births is 3.6 years. -/
theorem birth_interval (n : ℕ) (youngest_age sum_ages : ℝ) (h1 : n = 5) (h2 : youngest_age = 6)
    (h3 : sum_ages = 60) : ∃ interval : ℝ,
  interval = 3.6 ∧
  sum_ages = n * youngest_age + (interval * (n * (n - 1)) / 2) := by
  sorry

end birth_interval_l3646_364688


namespace tea_set_problem_l3646_364652

/-- Tea Set Problem -/
theorem tea_set_problem (cost_A cost_B : ℕ) 
  (h1 : cost_A + 2 * cost_B = 250)
  (h2 : 3 * cost_A + 4 * cost_B = 600)
  (h3 : ∀ a b : ℕ, a + b = 80 → 108 * a + 60 * b ≤ 6240)
  (h4 : ∀ a b : ℕ, a + b = 80 → 30 * a + 20 * b ≤ 1900)
  : ∃ a b : ℕ, a + b = 80 ∧ 30 * a + 20 * b = 1900 :=
sorry

end tea_set_problem_l3646_364652


namespace quadratic_rewrite_l3646_364660

theorem quadratic_rewrite (a b c : ℤ) :
  (∀ x : ℚ, 16 * x^2 + 40 * x + 18 = (a * x + b)^2 + c) →
  a * b = 20 := by
sorry

end quadratic_rewrite_l3646_364660


namespace sum_of_decimals_l3646_364666

theorem sum_of_decimals : (4.3 : ℝ) + 3.88 = 8.18 := by
  sorry

end sum_of_decimals_l3646_364666


namespace quadratic_point_m_value_l3646_364608

theorem quadratic_point_m_value (a m : ℝ) : 
  a > 0 → 
  m ≠ 0 → 
  3 = -a * m^2 + 2 * a * m + 3 → 
  m = 2 := by
sorry

end quadratic_point_m_value_l3646_364608


namespace triangle_angle_b_l3646_364650

theorem triangle_angle_b (a b c : ℝ) (A B C : ℝ) :
  c = 2 * b * Real.cos B →
  C = 2 * Real.pi / 3 →
  0 < a ∧ 0 < b ∧ 0 < c →
  0 < A ∧ 0 < B ∧ 0 < C →
  A + B + C = Real.pi →
  B = Real.pi / 6 := by
sorry

end triangle_angle_b_l3646_364650


namespace inscribed_circle_radius_is_one_over_sqrt_two_l3646_364675

/-- A right-angled triangle with its height and inscribed circles -/
structure RightTriangleWithInscribedCircles where
  /-- The original right-angled triangle -/
  originalTriangle : Set (ℝ × ℝ)
  /-- The two triangles formed by the height -/
  subTriangle1 : Set (ℝ × ℝ)
  subTriangle2 : Set (ℝ × ℝ)
  /-- The center of the inscribed circle of subTriangle1 -/
  center1 : ℝ × ℝ
  /-- The center of the inscribed circle of subTriangle2 -/
  center2 : ℝ × ℝ
  /-- The distance between center1 and center2 is 1 -/
  centers_distance : Real.sqrt ((center1.1 - center2.1)^2 + (center1.2 - center2.2)^2) = 1
  /-- The height divides the original triangle into subTriangle1 and subTriangle2 -/
  height_divides : originalTriangle = subTriangle1 ∪ subTriangle2
  /-- The original triangle is right-angled -/
  is_right_angled : ∃ (a b c : ℝ × ℝ), a ∈ originalTriangle ∧ b ∈ originalTriangle ∧ c ∈ originalTriangle ∧
    (b.1 - a.1) * (c.1 - a.1) + (b.2 - a.2) * (c.2 - a.2) = 0

/-- The radius of the inscribed circle of the original triangle -/
def inscribed_circle_radius (t : RightTriangleWithInscribedCircles) : ℝ :=
  sorry

/-- Theorem: The radius of the inscribed circle of the original triangle is 1/√2 -/
theorem inscribed_circle_radius_is_one_over_sqrt_two (t : RightTriangleWithInscribedCircles) :
  inscribed_circle_radius t = 1 / Real.sqrt 2 :=
sorry

end inscribed_circle_radius_is_one_over_sqrt_two_l3646_364675


namespace family_suitcases_l3646_364641

theorem family_suitcases (num_siblings : ℕ) (suitcases_per_sibling : ℕ) (total_suitcases : ℕ) : 
  num_siblings = 4 →
  suitcases_per_sibling = 2 →
  total_suitcases = 14 →
  ∃ (parents_suitcases : ℕ), 
    parents_suitcases = total_suitcases - (num_siblings * suitcases_per_sibling) ∧
    parents_suitcases % 2 = 0 ∧
    parents_suitcases / 2 = 3 :=
by sorry

end family_suitcases_l3646_364641


namespace seven_people_round_table_l3646_364665

/-- The number of unique seating arrangements for n people around a round table,
    considering rotations as the same arrangement -/
def roundTableArrangements (n : ℕ) : ℕ := Nat.factorial (n - 1)

/-- Theorem stating that the number of unique seating arrangements for 7 people
    around a round table is equal to 6! -/
theorem seven_people_round_table :
  roundTableArrangements 7 = Nat.factorial 6 := by sorry

end seven_people_round_table_l3646_364665


namespace arithmetic_calculation_l3646_364612

theorem arithmetic_calculation : 15 * 35 - 15 * 5 + 25 * 15 = 825 := by
  sorry

end arithmetic_calculation_l3646_364612


namespace quadratic_rewrite_l3646_364640

theorem quadratic_rewrite (k : ℝ) :
  let f := fun k : ℝ => 8 * k^2 - 6 * k + 16
  ∃ c r s : ℝ, (∀ k, f k = c * (k + r)^2 + s) ∧ s / r = -119 / 3 := by
  sorry

end quadratic_rewrite_l3646_364640


namespace volume_of_smaller_cube_l3646_364614

/-- Given that eight equal-sized cubes form a larger cube with a surface area of 1536 cm²,
    prove that the volume of each smaller cube is 512 cm³. -/
theorem volume_of_smaller_cube (surface_area : ℝ) (num_small_cubes : ℕ) :
  surface_area = 1536 →
  num_small_cubes = 8 →
  ∃ (side_length : ℝ),
    side_length > 0 ∧
    surface_area = 6 * side_length^2 ∧
    (side_length / 2)^3 = 512 :=
sorry

end volume_of_smaller_cube_l3646_364614


namespace stone_slab_length_l3646_364691

theorem stone_slab_length (n : ℕ) (total_area : ℝ) (h1 : n = 30) (h2 : total_area = 120) :
  ∃ (slab_length : ℝ), slab_length > 0 ∧ n * slab_length^2 = total_area ∧ slab_length = 2 := by
  sorry

end stone_slab_length_l3646_364691


namespace planet_colonization_combinations_l3646_364610

/-- Represents the number of planets of each type -/
structure PlanetCounts where
  venusLike : Nat
  jupiterLike : Nat

/-- Represents the colonization units required for each planet type -/
structure ColonizationUnits where
  venusLike : Nat
  jupiterLike : Nat

/-- Calculates the number of ways to choose planets given the constraints -/
def countPlanetCombinations (totalPlanets : PlanetCounts) (units : ColonizationUnits) (totalUnits : Nat) : Nat :=
  sorry

/-- The main theorem stating the number of combinations for the given problem -/
theorem planet_colonization_combinations :
  let totalPlanets := PlanetCounts.mk 7 5
  let units := ColonizationUnits.mk 3 1
  let totalUnits := 15
  countPlanetCombinations totalPlanets units totalUnits = 435 := by
  sorry

end planet_colonization_combinations_l3646_364610


namespace cubic_equation_solution_l3646_364604

theorem cubic_equation_solution (a b c : ℝ) : 
  (a^3 - 7*a^2 + 12*a = 18) ∧ 
  (b^3 - 7*b^2 + 12*b = 18) ∧ 
  (c^3 - 7*c^2 + 12*c = 18) →
  a*b/c + b*c/a + c*a/b = -6 := by
sorry

end cubic_equation_solution_l3646_364604


namespace apples_per_box_l3646_364643

theorem apples_per_box 
  (apples_per_crate : ℕ) 
  (crates_delivered : ℕ) 
  (rotten_apples : ℕ) 
  (boxes_used : ℕ) 
  (h1 : apples_per_crate = 42)
  (h2 : crates_delivered = 12)
  (h3 : rotten_apples = 4)
  (h4 : boxes_used = 50)
  : (apples_per_crate * crates_delivered - rotten_apples) / boxes_used = 10 :=
by
  sorry

#check apples_per_box

end apples_per_box_l3646_364643


namespace cube_of_hundred_l3646_364618

theorem cube_of_hundred : 99^3 + 3*(99^2) + 3*99 + 1 = 1000000 := by
  sorry

end cube_of_hundred_l3646_364618


namespace cos_two_thirds_pi_minus_two_alpha_l3646_364649

theorem cos_two_thirds_pi_minus_two_alpha (α : ℝ) 
  (h : Real.sin (α + π / 6) = Real.sqrt 6 / 3) : 
  Real.cos (2 * π / 3 - 2 * α) = 1 / 3 := by
  sorry

end cos_two_thirds_pi_minus_two_alpha_l3646_364649


namespace science_quiz_bowl_participation_l3646_364630

/-- The Science Quiz Bowl Participation Problem -/
theorem science_quiz_bowl_participation (participants_2018 : ℕ) : 
  participants_2018 = 150 → 
  ∃ (participants_2019 participants_2020 : ℕ),
    participants_2019 = 2 * participants_2018 + 20 ∧
    participants_2020 = participants_2019 / 2 - 40 ∧
    participants_2019 - participants_2020 = 200 := by
  sorry

end science_quiz_bowl_participation_l3646_364630


namespace smallest_n_for_sqrt_difference_l3646_364605

theorem smallest_n_for_sqrt_difference : 
  ∀ n : ℕ, n > 0 → (Real.sqrt n - Real.sqrt (n - 1) < 0.02 → n ≥ 626) ∧ 
  (Real.sqrt 626 - Real.sqrt 625 < 0.02) := by
sorry

end smallest_n_for_sqrt_difference_l3646_364605


namespace pythagorean_triple_identification_l3646_364602

def is_pythagorean_triple (a b c : ℕ) : Prop :=
  a * a + b * b = c * c

theorem pythagorean_triple_identification :
  is_pythagorean_triple 5 12 13 ∧
  ¬is_pythagorean_triple 8 12 15 ∧
  is_pythagorean_triple 8 15 17 ∧
  is_pythagorean_triple 9 40 41 :=
by sorry

end pythagorean_triple_identification_l3646_364602


namespace repeating_decimal_56_l3646_364642

/-- Represents a repeating decimal with a two-digit repetend -/
def RepeatingDecimal (a b : ℕ) : ℚ :=
  (10 * a + b : ℚ) / 99

theorem repeating_decimal_56 : RepeatingDecimal 5 6 = 56 / 99 := by
  sorry

end repeating_decimal_56_l3646_364642


namespace derivative_condition_implies_constants_l3646_364663

open Real

theorem derivative_condition_implies_constants (a b c d : ℝ) :
  let f : ℝ → ℝ := λ x => (a*x + b) * sin x + (c*x + d) * cos x
  (∀ x, deriv f x = x * cos x) →
  (a = 1 ∧ b = 0 ∧ c = 0 ∧ d = 1) := by
  sorry

end derivative_condition_implies_constants_l3646_364663


namespace intersection_point_l3646_364636

/-- The equation of the line -/
def line_equation (x y : ℝ) : Prop := y = 2 * x - 5

/-- A point is on the y-axis if its x-coordinate is 0 -/
def on_y_axis (x y : ℝ) : Prop := x = 0

/-- The intersection point of the line y = 2x - 5 and the y-axis is (0, -5) -/
theorem intersection_point : 
  ∃ (x y : ℝ), line_equation x y ∧ on_y_axis x y ∧ x = 0 ∧ y = -5 := by
  sorry

end intersection_point_l3646_364636


namespace power_multiplication_l3646_364600

theorem power_multiplication (a : ℝ) : a^2 * a^3 = a^5 := by
  sorry

end power_multiplication_l3646_364600


namespace smallest_k_no_real_roots_l3646_364638

theorem smallest_k_no_real_roots :
  ∃ k : ℤ, k = 3 ∧ 
  (∀ x : ℝ, (3*k - 2) * x^2 - 15*x + 13 ≠ 0) ∧
  (∀ k' : ℤ, k' < k → ∃ x : ℝ, (3*k' - 2) * x^2 - 15*x + 13 = 0) :=
sorry

end smallest_k_no_real_roots_l3646_364638


namespace rectangle_area_l3646_364684

theorem rectangle_area (perimeter width : ℝ) (h1 : perimeter = 52) (h2 : width = 11) :
  (width * (perimeter / 2 - width)) = 165 := by
  sorry

end rectangle_area_l3646_364684


namespace complex_addition_point_l3646_364658

/-- A complex number corresponding to a point in the complex plane -/
def complex_point (x y : ℝ) : ℂ := x + y * Complex.I

/-- The theorem stating that if z corresponds to (2,5), then 1+z corresponds to (3,5) -/
theorem complex_addition_point (z : ℂ) (h : z = complex_point 2 5) :
  1 + z = complex_point 3 5 := by
  sorry

end complex_addition_point_l3646_364658


namespace wades_food_truck_l3646_364645

/-- Wade's hot dog food truck problem -/
theorem wades_food_truck (tips_per_customer : ℚ) 
  (friday_customers sunday_customers : ℕ) (total_tips : ℚ) :
  tips_per_customer = 2 →
  friday_customers = 28 →
  sunday_customers = 36 →
  total_tips = 296 →
  let saturday_customers := (total_tips - tips_per_customer * (friday_customers + sunday_customers)) / tips_per_customer
  (saturday_customers : ℚ) / friday_customers = 3 := by
  sorry

end wades_food_truck_l3646_364645


namespace two_digit_square_sum_equals_concatenation_l3646_364634

theorem two_digit_square_sum_equals_concatenation : 
  {(x, y) : ℕ × ℕ | 10 ≤ x ∧ x ≤ 99 ∧ 10 ≤ y ∧ y ≤ 99 ∧ (x + y)^2 = 100 * x + y} = 
  {(20, 25), (30, 25)} := by
sorry

end two_digit_square_sum_equals_concatenation_l3646_364634


namespace no_solution_condition_l3646_364653

theorem no_solution_condition (a : ℝ) :
  (∀ x : ℝ, 9 * |x - 4*a| + |x - a^2| + 8*x - 2*a ≠ 0) ↔ (a < -26 ∨ a > 0) := by
  sorry

end no_solution_condition_l3646_364653


namespace cost_increase_doubles_b_l3646_364656

/-- The cost function for a given parameter b and coefficient t -/
def cost (t : ℝ) (b : ℝ) : ℝ := t * b^4

/-- Theorem stating that if the new cost is 1600% of the original cost,
    then the new value of b is 2 times the original value -/
theorem cost_increase_doubles_b (t : ℝ) (b₁ b₂ : ℝ) (h : t > 0) :
  cost t b₂ = 16 * cost t b₁ → b₂ = 2 * b₁ := by
  sorry


end cost_increase_doubles_b_l3646_364656


namespace taxi_fare_80_miles_l3646_364603

/-- Calculates the taxi fare for a given distance -/
def taxiFare (distance : ℝ) : ℝ :=
  sorry

theorem taxi_fare_80_miles : 
  -- Given conditions
  (taxiFare 60 = 150) →  -- 60-mile ride costs $150
  (∀ d, taxiFare d = 20 + (taxiFare d - 20) * d / 60) →  -- Flat rate of $20 and proportional charge
  -- Conclusion
  (taxiFare 80 = 193) :=
by
  sorry

end taxi_fare_80_miles_l3646_364603


namespace triple_integral_equality_l3646_364611

open MeasureTheory Interval Set

theorem triple_integral_equality {f : ℝ → ℝ} (hf : ContinuousOn f (Ioo 0 1)) :
  ∫ x in (Icc 0 1), ∫ y in (Icc x 1), ∫ z in (Icc x y), f x * f y * f z = 
  (1 / 6) * (∫ x in (Icc 0 1), f x) ^ 3 := by sorry

end triple_integral_equality_l3646_364611


namespace max_rectangle_area_l3646_364681

/-- Given a string of length 32 cm, the maximum area of a rectangle that can be formed is 64 cm². -/
theorem max_rectangle_area (string_length : ℝ) (h : string_length = 32) : 
  (∀ w h : ℝ, w > 0 → h > 0 → 2*w + 2*h ≤ string_length → w * h ≤ 64) ∧ 
  (∃ w h : ℝ, w > 0 ∧ h > 0 ∧ 2*w + 2*h = string_length ∧ w * h = 64) :=
by sorry

end max_rectangle_area_l3646_364681


namespace complex_sum_theorem_l3646_364682

theorem complex_sum_theorem (a b c d : ℝ) (ω : ℂ) 
  (ha : a ≠ -1) (hb : b ≠ -1) (hc : c ≠ -1) (hd : d ≠ -1)
  (hω1 : ω^4 = 1) (hω2 : ω ≠ 1)
  (h : 1 / (a + ω) + 1 / (b + ω) + 1 / (c + ω) + 1 / (d + ω) = 4 / ω^2) :
  1 / (a + 1) + 1 / (b + 1) + 1 / (c + 1) + 1 / (d + 1) = 4 := by
  sorry

end complex_sum_theorem_l3646_364682


namespace cos_pi_minus_2alpha_l3646_364646

theorem cos_pi_minus_2alpha (α : Real) (h : Real.sin α = 2/3) :
  Real.cos (π - 2*α) = -1/9 := by
  sorry

end cos_pi_minus_2alpha_l3646_364646


namespace simplify_expression_l3646_364693

theorem simplify_expression :
  ∃ (a b c : ℕ+),
    c.val = 24 ∧
    a.val = 56 ∧
    b.val = 54 ∧
    (∀ (x y z : ℕ+),
      Real.sqrt 6 + (1 / Real.sqrt 6) + Real.sqrt 8 + (1 / Real.sqrt 8) =
      (x.val * Real.sqrt 6 + y.val * Real.sqrt 8) / z.val →
      z.val ≥ c.val) ∧
    Real.sqrt 6 + (1 / Real.sqrt 6) + Real.sqrt 8 + (1 / Real.sqrt 8) =
    (a.val * Real.sqrt 6 + b.val * Real.sqrt 8) / c.val :=
by
  sorry

end simplify_expression_l3646_364693


namespace function_decreasing_range_l3646_364685

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 1 then (2 * a - 1) * x + 4 * a else a / x

theorem function_decreasing_range (a : ℝ) :
  (∀ x₁ x₂ : ℝ, x₁ ≠ x₂ → (f a x₂ - f a x₁) / (x₂ - x₁) < 0) ↔
  a ∈ Set.Icc (1 / 5 : ℝ) (1 / 2 : ℝ) ∧ a ≠ 1 / 2 :=
sorry

end function_decreasing_range_l3646_364685


namespace cone_volume_l3646_364669

/-- Given a cone with base radius √3 cm and lateral area 6π cm², its volume is 3π cm³ -/
theorem cone_volume (r h : ℝ) : 
  r = Real.sqrt 3 → 
  2 * π * r * (Real.sqrt (h^2 + r^2)) = 6 * π → 
  (1/3) * π * r^2 * h = 3 * π := by
  sorry

end cone_volume_l3646_364669


namespace circle_area_through_isosceles_triangle_vertices_l3646_364692

/-- The area of a circle passing through the vertices of an isosceles triangle -/
theorem circle_area_through_isosceles_triangle_vertices (a b c : ℝ) (h1 : a = 5) (h2 : b = 5) (h3 : c = 4) :
  let r := (a * b * c) / (4 * (1/2 * c * (a^2 - (c/2)^2).sqrt))
  π * r^2 = (13125/1764) * π := by
sorry

end circle_area_through_isosceles_triangle_vertices_l3646_364692


namespace monotonic_decreasing_cubic_function_l3646_364672

theorem monotonic_decreasing_cubic_function (a : ℝ) :
  (∀ x ∈ Set.Ioo (-1 : ℝ) 1, 
    ∀ y ∈ Set.Ioo (-1 : ℝ) 1, 
    x < y → (a * x^3 - 3*x) > (a * y^3 - 3*y)) →
  0 < a ∧ a ≤ 1 := by
sorry

end monotonic_decreasing_cubic_function_l3646_364672


namespace intersection_sum_l3646_364613

theorem intersection_sum (c d : ℝ) : 
  (3 = (1/3) * 3 + c) ∧ (3 = (1/3) * 3 + d) → c + d = 4 := by
  sorry

end intersection_sum_l3646_364613


namespace trigonometric_expression_evaluation_l3646_364689

theorem trigonometric_expression_evaluation (α : Real) (h : Real.tan α = 3) :
  (2 * Real.sin (2 * α) - 3 * Real.cos (2 * α)) / (4 * Real.sin (2 * α) + 5 * Real.cos (2 * α)) = -9/4 := by
  sorry

end trigonometric_expression_evaluation_l3646_364689


namespace remainder_problem_l3646_364616

theorem remainder_problem : (123456789012 : ℕ) % 252 = 84 := by
  sorry

end remainder_problem_l3646_364616


namespace sock_drawing_probability_l3646_364659

def total_socks : ℕ := 12
def socks_per_color_a : ℕ := 3
def colors_with_three_socks : ℕ := 3
def colors_with_one_sock : ℕ := 2
def socks_drawn : ℕ := 5

def favorable_outcomes : ℕ := 
  (colors_with_three_socks.choose 2) * 
  (colors_with_one_sock.choose 1) * 
  (socks_per_color_a.choose 2) * 
  (socks_per_color_a.choose 2) * 
  1

def total_outcomes : ℕ := total_socks.choose socks_drawn

theorem sock_drawing_probability :
  (favorable_outcomes : ℚ) / total_outcomes = 3 / 44 := by sorry

end sock_drawing_probability_l3646_364659


namespace triangle_sum_vertices_sides_l3646_364625

/-- Definition of a triangle -/
structure Triangle where
  vertices : ℕ
  sides : ℕ

/-- The sum of vertices and sides of a triangle is 6 -/
theorem triangle_sum_vertices_sides : ∀ t : Triangle, t.vertices + t.sides = 6 := by
  sorry

end triangle_sum_vertices_sides_l3646_364625


namespace prob_at_least_one_one_correct_l3646_364690

/-- The number of sides on each die -/
def num_sides : ℕ := 8

/-- The probability of at least one die showing a 1 when two fair 8-sided dice are rolled -/
def prob_at_least_one_one : ℚ := 15 / 64

/-- Theorem stating that the probability of at least one die showing a 1 
    when two fair 8-sided dice are rolled is 15/64 -/
theorem prob_at_least_one_one_correct : 
  prob_at_least_one_one = 1 - (num_sides - 1)^2 / num_sides^2 := by
  sorry

end prob_at_least_one_one_correct_l3646_364690


namespace girls_average_height_l3646_364648

theorem girls_average_height
  (num_boys : ℕ)
  (num_girls : ℕ)
  (total_students : ℕ)
  (avg_height_all : ℝ)
  (avg_height_boys : ℝ)
  (h1 : num_boys = 12)
  (h2 : num_girls = 10)
  (h3 : total_students = num_boys + num_girls)
  (h4 : avg_height_all = 103)
  (h5 : avg_height_boys = 108) :
  (total_students : ℝ) * avg_height_all - (num_boys : ℝ) * avg_height_boys = (num_girls : ℝ) * 97 :=
sorry

end girls_average_height_l3646_364648


namespace xia_shared_hundred_stickers_l3646_364632

/-- The number of stickers Xia shared with her friends -/
def shared_stickers (total : ℕ) (sheets_left : ℕ) (stickers_per_sheet : ℕ) : ℕ :=
  total - (sheets_left * stickers_per_sheet)

/-- Theorem stating that Xia shared 100 stickers with her friends -/
theorem xia_shared_hundred_stickers :
  shared_stickers 150 5 10 = 100 := by
  sorry

end xia_shared_hundred_stickers_l3646_364632


namespace union_of_S_and_T_l3646_364607

def S : Set ℕ := {1, 3, 5}
def T : Set ℕ := {3, 6}

theorem union_of_S_and_T : S ∪ T = {1, 3, 5, 6} := by sorry

end union_of_S_and_T_l3646_364607


namespace science_project_percentage_l3646_364677

theorem science_project_percentage (total_pages math_pages remaining_pages : ℕ) 
  (h1 : total_pages = 120)
  (h2 : math_pages = 10)
  (h3 : remaining_pages = 80) :
  (total_pages - math_pages - remaining_pages) / total_pages * 100 = 25 := by
  sorry

end science_project_percentage_l3646_364677


namespace initial_candies_count_l3646_364624

def candies_remaining (initial : ℕ) (day : ℕ) : ℤ :=
  match day with
  | 0 => initial
  | n + 1 => (candies_remaining initial n / 2 : ℤ) - 1

theorem initial_candies_count :
  ∃ initial : ℕ, 
    candies_remaining initial 3 = 0 ∧ 
    ∀ d : ℕ, d < 3 → candies_remaining initial d > 0 ∧ 
    initial = 14 :=
by sorry

end initial_candies_count_l3646_364624


namespace min_p_plus_q_l3646_364664

theorem min_p_plus_q (p q : ℕ+) (h : 108 * p = q^3) : 
  ∀ (p' q' : ℕ+), 108 * p' = q'^3 → p + q ≤ p' + q' :=
by sorry

end min_p_plus_q_l3646_364664


namespace production_rates_satisfy_conditions_unique_solution_l3646_364619

/-- The number of parts person A can make per day -/
def parts_per_day_A : ℕ := 60

/-- The number of parts person B can make per day -/
def parts_per_day_B : ℕ := 80

/-- The total number of machine parts -/
def total_parts : ℕ := 400

/-- Theorem stating that the given production rates satisfy the problem conditions -/
theorem production_rates_satisfy_conditions :
  (parts_per_day_A + 2 * parts_per_day_A + 2 * parts_per_day_B = total_parts - 60) ∧
  (3 * parts_per_day_A + 3 * parts_per_day_B = total_parts + 20) := by
  sorry

/-- Theorem proving the uniqueness of the solution -/
theorem unique_solution (a b : ℕ) 
  (h1 : a + 2 * a + 2 * b = total_parts - 60)
  (h2 : 3 * a + 3 * b = total_parts + 20) :
  a = parts_per_day_A ∧ b = parts_per_day_B := by
  sorry

end production_rates_satisfy_conditions_unique_solution_l3646_364619


namespace seven_factorial_mod_thirteen_l3646_364671

def factorial (n : ℕ) : ℕ := (List.range n).foldl (·*·) 1

theorem seven_factorial_mod_thirteen : factorial 7 % 13 = 11 := by sorry

end seven_factorial_mod_thirteen_l3646_364671


namespace intersection_point_median_altitude_l3646_364615

/-- Given a triangle ABC with vertices A(5,1), B(-1,-3), and C(4,3),
    the intersection point of the median CM and altitude BN
    has coordinates (5/3, -5/3). -/
theorem intersection_point_median_altitude (A B C M N : ℝ × ℝ) :
  A = (5, 1) →
  B = (-1, -3) →
  C = (4, 3) →
  M = ((A.1 + B.1) / 2, (A.2 + B.2) / 2) →
  (N.2 - B.2) * (C.1 - A.1) = (C.2 - A.2) * (N.1 - B.1) →
  (∃ t : ℝ, C + t • (M - C) = N) →
  N = (5/3, -5/3) :=
by sorry

end intersection_point_median_altitude_l3646_364615


namespace perfect_cubes_between_powers_of_three_l3646_364657

theorem perfect_cubes_between_powers_of_three : 
  let lower_bound := 3^6 + 1
  let upper_bound := 3^12 + 1
  (Finset.filter (fun n => lower_bound ≤ n^3 ∧ n^3 ≤ upper_bound) 
    (Finset.range (upper_bound + 1))).card = 72 := by
  sorry

end perfect_cubes_between_powers_of_three_l3646_364657


namespace f_properties_l3646_364601

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.sin (2 * x) + a * (Real.cos x) ^ 2

theorem f_properties (a : ℝ) (h : f a (π / 4) = 0) :
  -- The smallest positive period of f(x) is π
  (∃ (T : ℝ), T > 0 ∧ T = π ∧ ∀ (x : ℝ), f a (x + T) = f a x) ∧
  -- The maximum value of f(x) on [π/24, 11π/24] is √2 - 1
  (∀ (x : ℝ), x ∈ Set.Icc (π / 24) (11 * π / 24) → f a x ≤ Real.sqrt 2 - 1) ∧
  (∃ (x : ℝ), x ∈ Set.Icc (π / 24) (11 * π / 24) ∧ f a x = Real.sqrt 2 - 1) ∧
  -- The minimum value of f(x) on [π/24, 11π/24] is -√2/2 - 1
  (∀ (x : ℝ), x ∈ Set.Icc (π / 24) (11 * π / 24) → f a x ≥ -Real.sqrt 2 / 2 - 1) ∧
  (∃ (x : ℝ), x ∈ Set.Icc (π / 24) (11 * π / 24) ∧ f a x = -Real.sqrt 2 / 2 - 1) :=
by sorry

end f_properties_l3646_364601


namespace line_symmetry_l3646_364635

/-- A line in the 2D plane represented by its slope and y-intercept -/
structure Line where
  slope : ℝ
  intercept : ℝ

/-- The property of two lines being symmetrical about the x-axis -/
def symmetrical_about_x_axis (l1 l2 : Line) : Prop :=
  l1.slope = -l2.slope ∧ l1.intercept = -l2.intercept

/-- The given line y = 2x + 1 -/
def given_line : Line :=
  { slope := 2, intercept := 1 }

/-- The proposed symmetrical line y = -2x - 1 -/
def symmetrical_line : Line :=
  { slope := -2, intercept := -1 }

theorem line_symmetry :
  symmetrical_about_x_axis given_line symmetrical_line :=
sorry

end line_symmetry_l3646_364635


namespace worker_completion_time_l3646_364617

/-- Given two workers P and Q, this theorem proves the time taken by Q to complete a task alone,
    given the time taken by P alone and the time taken by P and Q together. -/
theorem worker_completion_time (time_p : ℝ) (time_pq : ℝ) (time_q : ℝ) : 
  time_p = 15 → time_pq = 6 → time_q = 10 → 
  1 / time_pq = 1 / time_p + 1 / time_q :=
by sorry

end worker_completion_time_l3646_364617


namespace larger_number_proof_l3646_364637

theorem larger_number_proof (L S : ℕ) (h1 : L > S) (h2 : L - S = 1365) (h3 : L = 8 * S + 15) :
  L = 1557 := by
  sorry

end larger_number_proof_l3646_364637


namespace intersection_implies_range_l3646_364626

def A : Set ℝ := {x : ℝ | 1 ≤ x ∧ x < 4}
def B (a : ℝ) : Set ℝ := {x : ℝ | x < a}

theorem intersection_implies_range (a : ℝ) : A ∩ B a = A → a ≥ 4 := by
  sorry

end intersection_implies_range_l3646_364626


namespace percentage_10_years_or_more_is_correct_l3646_364647

/-- Represents the employment distribution at Apex Innovations -/
structure EmploymentDistribution (X : ℕ) :=
  (less_than_2_years : ℕ := 7 * X)
  (two_to_4_years : ℕ := 4 * X)
  (four_to_6_years : ℕ := 3 * X)
  (six_to_8_years : ℕ := 3 * X)
  (eight_to_10_years : ℕ := 2 * X)
  (ten_to_12_years : ℕ := 2 * X)
  (twelve_to_14_years : ℕ := X)
  (fourteen_to_16_years : ℕ := X)
  (sixteen_to_18_years : ℕ := X)

/-- Calculates the percentage of employees who have worked for 10 years or more -/
def percentage_10_years_or_more (dist : EmploymentDistribution X) : ℚ :=
  let total_employees := 23 * X
  let employees_10_years_or_more := 5 * X
  (employees_10_years_or_more : ℚ) / total_employees * 100

/-- Theorem stating that the percentage of employees who have worked for 10 years or more is (5/23) * 100 -/
theorem percentage_10_years_or_more_is_correct (X : ℕ) (dist : EmploymentDistribution X) :
  percentage_10_years_or_more dist = 5 / 23 * 100 := by
  sorry

end percentage_10_years_or_more_is_correct_l3646_364647


namespace inequality_solution_set_l3646_364695

-- Define the inequality
def inequality (x : ℝ) : Prop := (3*x - 1) / (x - 2) ≤ 0

-- Define the solution set
def solution_set : Set ℝ := {x | 1/3 ≤ x ∧ x < 2}

-- Theorem statement
theorem inequality_solution_set :
  ∀ x : ℝ, x ≠ 2 → (x ∈ solution_set ↔ inequality x) :=
by sorry

end inequality_solution_set_l3646_364695


namespace number_2008_in_45th_group_l3646_364678

/-- The sequence of arrays where the nth group has n numbers and the last number of the nth group is n(n+1) -/
def sequence_group (n : ℕ) : ℕ := n * (n + 1)

/-- The proposition that 2008 is in the 45th group of the sequence -/
theorem number_2008_in_45th_group :
  ∃ k : ℕ, k ≤ 45 ∧ 
  sequence_group 44 < 2008 ∧ 
  2008 ≤ sequence_group 45 :=
by sorry

end number_2008_in_45th_group_l3646_364678


namespace continued_fraction_evaluation_l3646_364680

theorem continued_fraction_evaluation :
  2 + (3 / (4 + (5 / 6))) = 76 / 29 := by
  sorry

end continued_fraction_evaluation_l3646_364680


namespace sine_addition_formula_l3646_364622

theorem sine_addition_formula (x y z : ℝ) :
  Real.sin (x + y) * Real.cos z + Real.cos (x + y) * Real.sin z = Real.sin (x + y + z) := by
  sorry

end sine_addition_formula_l3646_364622


namespace average_speed_not_necessarily_five_l3646_364686

/-- A pedestrian's walk with varying speeds over 2.5 hours -/
structure PedestrianWalk where
  duration : ℝ
  hourly_distance : ℝ
  average_speed : ℝ

/-- Axiom: The pedestrian walks for 2.5 hours -/
axiom walk_duration : ∀ (w : PedestrianWalk), w.duration = 2.5

/-- Axiom: The pedestrian covers 5 km in any one-hour interval -/
axiom hourly_distance : ∀ (w : PedestrianWalk), w.hourly_distance = 5

/-- Theorem: The average speed for the entire journey is not necessarily 5 km per hour -/
theorem average_speed_not_necessarily_five :
  ∃ (w : PedestrianWalk), w.average_speed ≠ 5 := by
  sorry


end average_speed_not_necessarily_five_l3646_364686


namespace tourists_knowing_both_languages_l3646_364667

theorem tourists_knowing_both_languages 
  (total : ℕ) 
  (neither : ℕ) 
  (german : ℕ) 
  (french : ℕ) 
  (h1 : total = 100) 
  (h2 : neither = 10) 
  (h3 : german = 76) 
  (h4 : french = 83) : 
  total - neither = german + french - 69 := by
sorry

end tourists_knowing_both_languages_l3646_364667


namespace solve_equation_l3646_364670

theorem solve_equation (n : ℚ) : 
  (1 : ℚ) / (n + 1) + 2 / (n + 1) + (n + 1) / (n + 1) = 2 → n = 2 := by
  sorry

end solve_equation_l3646_364670


namespace square_plus_one_nonnegative_l3646_364679

theorem square_plus_one_nonnegative (m : ℝ) : m^2 + 1 ≥ 0 := by
  sorry

end square_plus_one_nonnegative_l3646_364679


namespace initial_group_size_l3646_364631

theorem initial_group_size (initial_avg : ℝ) (new_avg : ℝ) (weight1 : ℝ) (weight2 : ℝ) :
  initial_avg = 48 →
  new_avg = 51 →
  weight1 = 78 →
  weight2 = 93 →
  ∃ n : ℕ, n * initial_avg + weight1 + weight2 = (n + 2) * new_avg ∧ n = 23 :=
by sorry

end initial_group_size_l3646_364631


namespace amelia_monday_distance_l3646_364698

theorem amelia_monday_distance (total_distance tuesday_distance remaining_distance : ℕ) 
  (h1 : total_distance = 8205)
  (h2 : tuesday_distance = 582)
  (h3 : remaining_distance = 6716) :
  total_distance = tuesday_distance + remaining_distance + 907 := by
  sorry

end amelia_monday_distance_l3646_364698


namespace cost_price_per_metre_l3646_364661

def total_selling_price : ℕ := 18000
def total_length : ℕ := 300
def loss_per_metre : ℕ := 5

theorem cost_price_per_metre : 
  (total_selling_price / total_length) + loss_per_metre = 65 := by
  sorry

end cost_price_per_metre_l3646_364661


namespace expression_evaluation_l3646_364673

theorem expression_evaluation : 8 / 4 - 3^2 - 10 + 5 * 2 = -7 := by
  sorry

end expression_evaluation_l3646_364673


namespace marys_income_percentage_l3646_364676

theorem marys_income_percentage (juan tim mary : ℝ) 
  (h1 : tim = juan * 0.5) 
  (h2 : mary = tim * 1.6) : 
  mary = juan * 0.8 := by
sorry

end marys_income_percentage_l3646_364676


namespace prob_at_least_one_on_l3646_364668

/-- The probability that at least one of three independent electronic components is on,
    given that each component has a probability of 1/2 of being on. -/
theorem prob_at_least_one_on (n : Nat) (p : ℝ) (h1 : n = 3) (h2 : p = 1 / 2) :
  1 - (1 - p) ^ n = 7 / 8 := by
  sorry

end prob_at_least_one_on_l3646_364668


namespace no_right_triangle_perimeter_twice_hypotenuse_l3646_364655

theorem no_right_triangle_perimeter_twice_hypotenuse :
  ¬∃ (a b c : ℝ), 
    a > 0 ∧ b > 0 ∧ c > 0 ∧  -- positive sides
    a^2 + b^2 = c^2 ∧        -- right triangle (Pythagorean theorem)
    a + b + c = 2*c          -- perimeter equals twice the hypotenuse
    := by sorry

end no_right_triangle_perimeter_twice_hypotenuse_l3646_364655


namespace sequence_squared_l3646_364629

theorem sequence_squared (a : ℕ → ℝ) :
  a 1 = 1 ∧
  (∀ n : ℕ, n > 0 → 4 * a n * a (n + 1) = (a n + a (n + 1) - 1)^2) ∧
  (∀ n : ℕ, n > 1 → a n > a (n - 1)) →
  ∀ n : ℕ, n > 0 → a n = n^2 := by
sorry

end sequence_squared_l3646_364629


namespace table_runner_coverage_l3646_364627

theorem table_runner_coverage (total_runner_area : ℝ) (table_area : ℝ) 
  (two_layer_area : ℝ) (four_layer_area : ℝ) 
  (h1 : total_runner_area = 360)
  (h2 : table_area = 250)
  (h3 : two_layer_area = 35)
  (h4 : four_layer_area = 15)
  (h5 : 0.9 * table_area = two_layer_area + three_layer_area + four_layer_area + one_layer_area)
  (h6 : total_runner_area = one_layer_area + 2 * two_layer_area + 3 * three_layer_area + 4 * four_layer_area) :
  three_layer_area = 65 := by
  sorry


end table_runner_coverage_l3646_364627


namespace petyas_chips_l3646_364699

theorem petyas_chips (x : ℕ) (y : ℕ) : 
  y = x - 2 → -- The side of the square has 2 fewer chips than the triangle
  3 * x - 3 = 4 * y - 4 → -- Total chips are the same for both shapes
  3 * x - 3 = 24 -- The total number of chips is 24
  := by sorry

end petyas_chips_l3646_364699


namespace sergeant_travel_distance_l3646_364609

/-- Proves that given an infantry column of length 1 km, if the infantry walks 4/3 km
    during the time it takes for someone to travel from the end to the beginning of
    the column and back at twice the speed of the infantry, then the total distance
    traveled by that person is 8/3 km. -/
theorem sergeant_travel_distance
  (column_length : ℝ)
  (infantry_distance : ℝ)
  (sergeant_speed_ratio : ℝ)
  (h1 : column_length = 1)
  (h2 : infantry_distance = 4/3)
  (h3 : sergeant_speed_ratio = 2) :
  2 * infantry_distance = 8/3 := by
  sorry

#check sergeant_travel_distance

end sergeant_travel_distance_l3646_364609


namespace greatest_three_digit_multiple_of_17_l3646_364620

theorem greatest_three_digit_multiple_of_17 : 
  ∀ n : ℕ, n ≤ 999 → n ≥ 100 → n % 17 = 0 → n ≤ 986 :=
by
  sorry

end greatest_three_digit_multiple_of_17_l3646_364620


namespace jordan_weight_change_l3646_364628

def weight_change (initial_weight : ℕ) (loss_first_4_weeks : ℕ) (loss_week_5 : ℕ) 
  (loss_next_7_weeks : ℕ) (gain_week_13 : ℕ) : ℕ :=
  initial_weight - (4 * loss_first_4_weeks + loss_week_5 + 7 * loss_next_7_weeks - gain_week_13)

theorem jordan_weight_change :
  weight_change 250 3 5 2 2 = 221 :=
by sorry

end jordan_weight_change_l3646_364628


namespace peach_difference_l3646_364621

theorem peach_difference (martine_peaches benjy_peaches gabrielle_peaches : ℕ) : 
  martine_peaches > 2 * benjy_peaches →
  benjy_peaches = gabrielle_peaches / 3 →
  martine_peaches = 16 →
  gabrielle_peaches = 15 →
  martine_peaches - 2 * benjy_peaches = 6 := by
sorry

end peach_difference_l3646_364621


namespace arithmetic_sequence_21st_term_l3646_364651

/-- Given an arithmetic sequence with first term 11 and common difference -3,
    prove that the 21st term is -49. -/
theorem arithmetic_sequence_21st_term :
  let a : ℕ → ℤ := λ n => 11 + (n - 1) * (-3)
  a 21 = -49 := by sorry

end arithmetic_sequence_21st_term_l3646_364651


namespace hall_of_mirrors_wall_length_l3646_364674

/-- Given three walls in a hall of mirrors, where two walls have the same unknown length and are 12 feet high,
    and the third wall is 20 feet by 12 feet, if the total glass needed is 960 square feet,
    then the length of each of the two unknown walls is 30 feet. -/
theorem hall_of_mirrors_wall_length :
  ∀ (L : ℝ),
  (2 * L * 12 + 20 * 12 = 960) →
  L = 30 := by
sorry

end hall_of_mirrors_wall_length_l3646_364674


namespace square_sum_reciprocal_l3646_364683

theorem square_sum_reciprocal (x : ℝ) (h : x + 1/x = 3/2) : x^2 + 1/x^2 = 1/4 := by
  sorry

end square_sum_reciprocal_l3646_364683


namespace biased_coin_probability_l3646_364639

theorem biased_coin_probability (p : ℝ) : 
  p < (1 : ℝ) / 2 →
  (Nat.choose 6 3 : ℝ) * p^3 * (1 - p)^3 = (1 : ℝ) / 20 →
  p = 0.125 := by
sorry

end biased_coin_probability_l3646_364639


namespace percent_of_number_zero_point_one_percent_of_12356_l3646_364694

theorem percent_of_number (x : ℝ) : x * 0.001 = 0.001 * x := by sorry

theorem zero_point_one_percent_of_12356 : (12356 : ℝ) * 0.001 = 12.356 := by sorry

end percent_of_number_zero_point_one_percent_of_12356_l3646_364694


namespace cube_difference_equality_l3646_364697

theorem cube_difference_equality (x y : ℝ) (h : x - y = 1) :
  x^3 - 3*x*y - y^3 = 1 := by sorry

end cube_difference_equality_l3646_364697


namespace product_125_sum_31_l3646_364696

theorem product_125_sum_31 (a b c : ℕ+) : 
  a ≠ b ∧ b ≠ c ∧ a ≠ c →
  a * b * c = 125 →
  (a : ℕ) + b + c = 31 := by
sorry

end product_125_sum_31_l3646_364696


namespace doubling_function_m_range_l3646_364644

/-- A function f is a "doubling function" if there exists an interval [a,b] in its domain
    such that the range of f on [a,b] is [2a,2b] -/
def DoublingFunction (f : ℝ → ℝ) : Prop :=
  ∃ a b : ℝ, a < b ∧ (∀ x ∈ Set.Icc a b, f x ∈ Set.Icc (2*a) (2*b)) ∧
    (∀ y ∈ Set.Icc (2*a) (2*b), ∃ x ∈ Set.Icc a b, f x = y)

/-- The main theorem stating that for f(x) = ln(e^x + m) to be a doubling function,
    m must be in the range (-1/4, 0) -/
theorem doubling_function_m_range :
  ∀ m : ℝ, (DoublingFunction (fun x ↦ Real.log (Real.exp x + m))) ↔ -1/4 < m ∧ m < 0 := by
  sorry


end doubling_function_m_range_l3646_364644


namespace circle_sum_is_twenty_l3646_364623

def CircleSum (digits : Finset ℕ) (sum : ℕ) : Prop :=
  ∃ (a b c d e f x : ℕ),
    digits = {1, 2, 3, 4, 5, 6, 7, 8, 9} ∧
    5 ∈ digits ∧
    2 ∈ digits ∧
    x + a + b + 5 = sum ∧
    x + e + f + 2 = sum ∧
    5 + c + d + 2 = sum

theorem circle_sum_is_twenty :
  ∃ (digits : Finset ℕ) (sum : ℕ), CircleSum digits sum ∧ sum = 20 := by
  sorry

end circle_sum_is_twenty_l3646_364623


namespace rectangle_perimeter_l3646_364654

/-- Given a square with side z containing a smaller square with side w,
    prove that the perimeter of a rectangle formed by the remaining area is 2z. -/
theorem rectangle_perimeter (z w : ℝ) (hz : z > 0) (hw : w > 0) (hw_lt_z : w < z) :
  2 * w + 2 * (z - w) = 2 * z := by
  sorry

end rectangle_perimeter_l3646_364654


namespace condition_equiv_range_l3646_364687

/-- The set A in the real numbers -/
def A : Set ℝ := {x | -5 < x ∧ x < 4}

/-- The set B in the real numbers -/
def B : Set ℝ := {x | x < -6 ∨ x > 1}

/-- The set C in the real numbers, parameterized by m -/
def C (m : ℝ) : Set ℝ := {x | x < m}

/-- The theorem stating the equivalence of the conditions and the range of m -/
theorem condition_equiv_range :
  ∀ m : ℝ,
  (C m ⊇ (A ∩ B) ∧ C m ⊇ (Aᶜ ∩ Bᶜ)) ↔ m ∈ Set.Ici 4 := by
  sorry

end condition_equiv_range_l3646_364687
