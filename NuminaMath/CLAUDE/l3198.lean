import Mathlib

namespace NUMINAMATH_CALUDE_thomson_savings_l3198_319827

/-- Calculates the amount saved by Mrs. Thomson given her spending pattern -/
def amount_saved (X : ℝ) : ℝ :=
  let after_food := X * (1 - 0.375)
  let after_clothes := after_food * (1 - 0.22)
  let after_household := after_clothes * (1 - 0.15)
  let after_stocks := after_household * (1 - 0.30)
  let after_tuition := after_stocks * (1 - 0.40)
  after_tuition

theorem thomson_savings (X : ℝ) : amount_saved X = 0.1740375 * X := by
  sorry

end NUMINAMATH_CALUDE_thomson_savings_l3198_319827


namespace NUMINAMATH_CALUDE_triangle_angles_l3198_319841

theorem triangle_angles (α β γ : Real) : 
  (180 - α) / (180 - β) = 13 / 9 →
  (180 - α) - (180 - β) = 45 →
  α + β + γ = 180 →
  α = 33.75 ∧ β = 78.75 ∧ γ = 67.5 := by
sorry

end NUMINAMATH_CALUDE_triangle_angles_l3198_319841


namespace NUMINAMATH_CALUDE_f_has_inverse_when_x_geq_2_l3198_319896

def f (x : ℝ) : ℝ := x^2 - 4*x + 5

theorem f_has_inverse_when_x_geq_2 :
  ∀ (a b : ℝ), a ≥ 2 → b ≥ 2 → a ≠ b → f a ≠ f b :=
by
  sorry

end NUMINAMATH_CALUDE_f_has_inverse_when_x_geq_2_l3198_319896


namespace NUMINAMATH_CALUDE_inequality_proof_l3198_319846

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a^2 + b*c) / (a*(b+c)) + (b^2 + c*a) / (b*(c+a)) + (c^2 + a*b) / (c*(a+b)) ≥ 3 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3198_319846


namespace NUMINAMATH_CALUDE_vector_properties_l3198_319800

/-- Given vectors in ℝ², prove properties about their relationships -/
theorem vector_properties (a b c : ℝ × ℝ) (k : ℝ) :
  a = (1, 2) →
  b = (-3, 2) →
  ‖c‖ = 2 * Real.sqrt 5 →
  ∃ (t : ℝ), c = t • a →
  (∃ (t : ℝ), (k • a + 2 • b) = t • (2 • a - 4 • b) → k = -1) ∧
  ((k • a + 2 • b) • (2 • a - 4 • b) = 0 → k = 50/3) ∧
  (c = (2, 4) ∨ c = (-2, -4)) := by
sorry

end NUMINAMATH_CALUDE_vector_properties_l3198_319800


namespace NUMINAMATH_CALUDE_classroom_lights_l3198_319844

theorem classroom_lights (num_lamps : ℕ) (h : num_lamps = 4) : 
  (2^num_lamps : ℕ) - 1 = 15 := by
  sorry

#check classroom_lights

end NUMINAMATH_CALUDE_classroom_lights_l3198_319844


namespace NUMINAMATH_CALUDE_quadratic_radical_for_all_real_l3198_319867

theorem quadratic_radical_for_all_real (a : ℝ) : ∃ (x : ℝ), x ^ 2 = a ^ 2 + 1 :=
sorry

end NUMINAMATH_CALUDE_quadratic_radical_for_all_real_l3198_319867


namespace NUMINAMATH_CALUDE_arithmetic_sequence_13th_term_l3198_319875

def arithmetic_sequence (a : ℕ → ℤ) (d : ℤ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n + d

def geometric_sequence (a b c : ℤ) : Prop :=
  b * b = a * c

theorem arithmetic_sequence_13th_term
  (a : ℕ → ℤ)
  (d : ℤ)
  (h1 : d ≠ 0)
  (h2 : arithmetic_sequence a d)
  (h3 : geometric_sequence (a 9) (a 1) (a 5))
  (h4 : a 1 + 3 * a 5 + a 9 = 20) :
  a 13 = 28 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_13th_term_l3198_319875


namespace NUMINAMATH_CALUDE_marys_brother_height_l3198_319887

/-- The height of Mary's brother given the conditions of the roller coaster problem -/
theorem marys_brother_height (min_height : ℝ) (mary_ratio : ℝ) (mary_growth : ℝ) :
  min_height = 140 ∧ mary_ratio = 2/3 ∧ mary_growth = 20 →
  ∃ (mary_height : ℝ) (brother_height : ℝ),
    mary_height + mary_growth = min_height ∧
    mary_height = mary_ratio * brother_height ∧
    brother_height = 180 := by
  sorry

end NUMINAMATH_CALUDE_marys_brother_height_l3198_319887


namespace NUMINAMATH_CALUDE_tom_share_calculation_l3198_319822

def total_amount : ℝ := 18500

def natalie_percentage : ℝ := 0.35
def rick_percentage : ℝ := 0.30
def lucy_percentage : ℝ := 0.40

def minimum_share : ℝ := 1000

def natalie_share : ℝ := natalie_percentage * total_amount
def remaining_after_natalie : ℝ := total_amount - natalie_share

def rick_share : ℝ := rick_percentage * remaining_after_natalie
def remaining_after_rick : ℝ := remaining_after_natalie - rick_share

def lucy_share : ℝ := lucy_percentage * remaining_after_rick
def tom_share : ℝ := remaining_after_rick - lucy_share

theorem tom_share_calculation :
  tom_share = 5050.50 ∧
  natalie_share ≥ minimum_share ∧
  rick_share ≥ minimum_share ∧
  lucy_share ≥ minimum_share ∧
  tom_share ≥ minimum_share :=
by sorry

end NUMINAMATH_CALUDE_tom_share_calculation_l3198_319822


namespace NUMINAMATH_CALUDE_sum_equals_quadratic_l3198_319889

open BigOperators

/-- The sequence a_n defined as 4n - 3 -/
def a (n : ℕ) : ℤ := 4 * n - 3

/-- The sum of the first n terms of the sequence -/
def S (n : ℕ) : ℤ := ∑ i in Finset.range n, a (i + 1)

/-- The theorem stating the equality between the sum and the quadratic expression -/
theorem sum_equals_quadratic (a b c : ℤ) : 
  (∀ n : ℕ, n > 0 → S n = 2 * a * n^2 + b * n + c) →
  a - b + c = 2 := by
  sorry


end NUMINAMATH_CALUDE_sum_equals_quadratic_l3198_319889


namespace NUMINAMATH_CALUDE_log_product_equal_twelve_l3198_319838

theorem log_product_equal_twelve :
  Real.log 9 / Real.log 2 * (Real.log 5 / Real.log 3) * (Real.log 8 / Real.log (Real.sqrt 5)) = 12 := by
  sorry

end NUMINAMATH_CALUDE_log_product_equal_twelve_l3198_319838


namespace NUMINAMATH_CALUDE_rationalize_denominator_l3198_319869

theorem rationalize_denominator : (1 : ℝ) / (Real.sqrt 3 - 1) = (Real.sqrt 3 + 1) / 2 := by
  sorry

end NUMINAMATH_CALUDE_rationalize_denominator_l3198_319869


namespace NUMINAMATH_CALUDE_g_composition_of_three_l3198_319808

def g (x : ℤ) : ℤ :=
  if x % 2 = 0 then x / 2 else 3 * x - 1

theorem g_composition_of_three : g (g (g (g 3))) = 1 := by
  sorry

end NUMINAMATH_CALUDE_g_composition_of_three_l3198_319808


namespace NUMINAMATH_CALUDE_certain_number_proof_l3198_319818

theorem certain_number_proof : ∃ x : ℝ, 45 * x = 0.6 * 900 ∧ x = 12 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_proof_l3198_319818


namespace NUMINAMATH_CALUDE_binomial_expansion_coefficient_l3198_319839

def binomial_coefficient (n k : ℕ) : ℕ := sorry

theorem binomial_expansion_coefficient (k : ℚ) : 
  (binomial_coefficient 5 1) * (-k) = -10 → k = 2 := by sorry

end NUMINAMATH_CALUDE_binomial_expansion_coefficient_l3198_319839


namespace NUMINAMATH_CALUDE_father_son_age_sum_father_son_age_sum_proof_l3198_319811

/-- The sum of the present ages of a father and son is 36 years, given that 6 years ago
    the father was 3 times as old as his son, and now the father is only twice as old as his son. -/
theorem father_son_age_sum : ℕ → ℕ → Prop :=
  fun son_age father_age =>
    (father_age - 6 = 3 * (son_age - 6)) ∧  -- 6 years ago condition
    (father_age = 2 * son_age) ∧             -- current age condition
    (son_age + father_age = 36)              -- sum of ages

/-- Proof of the theorem -/
theorem father_son_age_sum_proof : ∃ (son_age father_age : ℕ), father_son_age_sum son_age father_age :=
  sorry

end NUMINAMATH_CALUDE_father_son_age_sum_father_son_age_sum_proof_l3198_319811


namespace NUMINAMATH_CALUDE_rectangle_length_proof_l3198_319804

theorem rectangle_length_proof (width area : ℝ) (h1 : width = 3 * Real.sqrt 2) (h2 : area = 18 * Real.sqrt 6) :
  area / width = 6 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_length_proof_l3198_319804


namespace NUMINAMATH_CALUDE_twelfth_team_games_l3198_319814

/-- Represents a football tournament -/
structure Tournament where
  teams : Fin 12 → ℕ
  first_team_games : teams 0 = 11
  three_teams_nine_games : ∃ i j k, i ≠ j ∧ j ≠ k ∧ i ≠ k ∧ teams i = 9 ∧ teams j = 9 ∧ teams k = 9
  one_team_five_games : ∃ i, teams i = 5
  four_teams_four_games : ∃ i j k l, i ≠ j ∧ j ≠ k ∧ k ≠ l ∧ i ≠ k ∧ i ≠ l ∧ j ≠ l ∧
                          teams i = 4 ∧ teams j = 4 ∧ teams k = 4 ∧ teams l = 4
  two_teams_one_game : ∃ i j, i ≠ j ∧ teams i = 1 ∧ teams j = 1
  no_repeat_games : ∀ i j, i ≠ j → teams i + teams j ≤ 12

theorem twelfth_team_games (t : Tournament) : 
  ∃ i, t.teams i = 5 ∧ ∀ j, j ≠ i → t.teams j ≠ 5 :=
sorry

end NUMINAMATH_CALUDE_twelfth_team_games_l3198_319814


namespace NUMINAMATH_CALUDE_sqrt_inequality_l3198_319872

theorem sqrt_inequality (x : ℝ) (h1 : x > 0) (h2 : Real.sqrt (x + 4) < 3 * x) : 
  x > (1 + Real.sqrt 145) / 18 := by
sorry

end NUMINAMATH_CALUDE_sqrt_inequality_l3198_319872


namespace NUMINAMATH_CALUDE_point_shift_theorem_l3198_319842

theorem point_shift_theorem (original_coord final_coord shift : ℤ) :
  final_coord = original_coord + shift →
  final_coord = 8 →
  shift = 13 →
  original_coord = -5 := by
sorry

end NUMINAMATH_CALUDE_point_shift_theorem_l3198_319842


namespace NUMINAMATH_CALUDE_friend_lunch_cost_l3198_319810

theorem friend_lunch_cost (total : ℝ) (difference : ℝ) (friend_cost : ℝ) : 
  total = 11 →
  difference = 3 →
  friend_cost = (total + difference) / 2 →
  friend_cost = 7 :=
by sorry

end NUMINAMATH_CALUDE_friend_lunch_cost_l3198_319810


namespace NUMINAMATH_CALUDE_twice_x_minus_three_l3198_319888

/-- The algebraic expression for "twice x minus 3" is equal to 2x - 3. -/
theorem twice_x_minus_three (x : ℝ) : 2 * x - 3 = 2 * x - 3 := by
  sorry

end NUMINAMATH_CALUDE_twice_x_minus_three_l3198_319888


namespace NUMINAMATH_CALUDE_row3_seat6_representation_l3198_319854

/-- Represents a seat in a movie theater -/
structure Seat :=
  (row : ℕ)
  (seat : ℕ)

/-- The representation format for seats in the theater -/
def represent (s : Seat) : ℕ × ℕ := (s.row, s.seat)

/-- Given condition: (5,8) represents row 5, seat 8 -/
axiom example_representation : represent { row := 5, seat := 8 } = (5, 8)

/-- Theorem: The representation of row 3, seat 6 is (3,6) -/
theorem row3_seat6_representation :
  represent { row := 3, seat := 6 } = (3, 6) := by
  sorry

end NUMINAMATH_CALUDE_row3_seat6_representation_l3198_319854


namespace NUMINAMATH_CALUDE_set_intersection_problem_l3198_319899

theorem set_intersection_problem (a : ℝ) : 
  let A : Set ℝ := {-4, 2*a-1, a^2}
  let B : Set ℝ := {a-5, 1-a, 9}
  9 ∈ (A ∩ B) → a = 5 ∨ a = -3 :=
by
  sorry


end NUMINAMATH_CALUDE_set_intersection_problem_l3198_319899


namespace NUMINAMATH_CALUDE_rectangle_width_l3198_319880

theorem rectangle_width (width length perimeter : ℝ) : 
  length = (7 / 5) * width →
  perimeter = 2 * length + 2 * width →
  perimeter = 288 →
  width = 60 := by
sorry

end NUMINAMATH_CALUDE_rectangle_width_l3198_319880


namespace NUMINAMATH_CALUDE_inequality_proof_l3198_319823

theorem inequality_proof (p : ℝ) (h1 : 18 * p < 10) (h2 : p > 0.5) : 0.5 < p ∧ p < 5/9 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3198_319823


namespace NUMINAMATH_CALUDE_parallel_lines_from_parallel_planes_l3198_319874

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the parallel relation for lines and planes
variable (parallelLines : Line → Line → Prop)
variable (parallelPlanes : Plane → Plane → Prop)

-- Define the intersection operation for planes
variable (planeIntersection : Plane → Plane → Line)

-- Define the subset relation for lines and planes
variable (lineInPlane : Line → Plane → Prop)

-- State the theorem
theorem parallel_lines_from_parallel_planes 
  (m n : Line) (α β γ : Plane)
  (nonCoincidentLines : m ≠ n)
  (nonCoincidentPlanes : α ≠ β ∧ β ≠ γ ∧ α ≠ γ)
  (planesParallel : parallelPlanes α β)
  (mIntersection : planeIntersection α γ = m)
  (nIntersection : planeIntersection β γ = n) :
  parallelLines m n :=
sorry

end NUMINAMATH_CALUDE_parallel_lines_from_parallel_planes_l3198_319874


namespace NUMINAMATH_CALUDE_taehyung_ran_160_meters_l3198_319851

/-- The perimeter of a square -/
def squarePerimeter (side : ℝ) : ℝ := 4 * side

/-- The distance Taehyung ran around the square park -/
def taehyungDistance : ℝ := squarePerimeter 40

theorem taehyung_ran_160_meters : taehyungDistance = 160 := by
  sorry

end NUMINAMATH_CALUDE_taehyung_ran_160_meters_l3198_319851


namespace NUMINAMATH_CALUDE_school_robe_cost_l3198_319876

/-- Calculates the cost of robes based on the given pricing tiers --/
def robeCost (n : ℕ) : ℚ :=
  if n ≤ 10 then 3 * n
  else if n ≤ 20 then 2.5 * n
  else 2 * n

/-- Calculates the total cost including alterations, customization, and sales tax --/
def totalCost (singers : ℕ) (existingRobes : ℕ) (alterationCost : ℚ) (customizationCost : ℚ) (salesTax : ℚ) : ℚ :=
  let neededRobes := singers - existingRobes
  let baseCost := robeCost neededRobes
  let additionalCost := (alterationCost + customizationCost) * neededRobes
  let subtotal := baseCost + additionalCost
  subtotal * (1 + salesTax)

theorem school_robe_cost :
  totalCost 30 12 1.5 0.75 0.08 = 92.34 :=
sorry

end NUMINAMATH_CALUDE_school_robe_cost_l3198_319876


namespace NUMINAMATH_CALUDE_expression_simplification_l3198_319890

theorem expression_simplification 
  (p q r x : ℝ) 
  (h_distinct : p ≠ q ∧ q ≠ r ∧ p ≠ r) : 
  ((x + p)^4) / ((p - q)*(p - r)) + 
  ((x + q)^4) / ((q - p)*(q - r)) + 
  ((x + r)^4) / ((r - p)*(r - q)) = 
  p + q + r + 4*x := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l3198_319890


namespace NUMINAMATH_CALUDE_remainder_problem_l3198_319893

theorem remainder_problem : (7 * 10^20 + 2^20) % 11 = 8 := by
  sorry

end NUMINAMATH_CALUDE_remainder_problem_l3198_319893


namespace NUMINAMATH_CALUDE_probability_consecutive_numbers_l3198_319858

/-- The number of balls -/
def n : ℕ := 6

/-- The number of balls to be drawn -/
def k : ℕ := 3

/-- The total number of ways to draw k balls from n balls -/
def total_ways : ℕ := Nat.choose n k

/-- The number of ways to draw k balls with exactly two consecutive numbers -/
def consecutive_ways : ℕ := 12

/-- The probability of drawing exactly two balls with consecutive numbers -/
def probability : ℚ := consecutive_ways / total_ways

theorem probability_consecutive_numbers : probability = 3/5 := by sorry

end NUMINAMATH_CALUDE_probability_consecutive_numbers_l3198_319858


namespace NUMINAMATH_CALUDE_incorrect_calculation_l3198_319853

theorem incorrect_calculation (m n : ℕ) (h1 : n ≤ 100) : 
  ¬ (∃ k : ℕ, ∃ B : ℕ, 
    (m : ℚ) / n = (k : ℚ) + B / (1000 * n) ∧ 
    167 ≤ B ∧ B < 168) := by
  sorry

end NUMINAMATH_CALUDE_incorrect_calculation_l3198_319853


namespace NUMINAMATH_CALUDE_min_sum_absolute_values_l3198_319826

theorem min_sum_absolute_values :
  ∃ (min : ℝ), min = 4 ∧ 
  (∀ x : ℝ, |x + 3| + |x + 5| + |x + 7| ≥ min) ∧
  (∃ x : ℝ, |x + 3| + |x + 5| + |x + 7| = min) := by
  sorry

end NUMINAMATH_CALUDE_min_sum_absolute_values_l3198_319826


namespace NUMINAMATH_CALUDE_expression_evaluation_l3198_319824

theorem expression_evaluation (a : ℚ) (h : a = 4/3) : (4*a^2 - 12*a + 9)*(3*a - 4) = 0 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l3198_319824


namespace NUMINAMATH_CALUDE_fixed_point_of_shifted_function_l3198_319825

theorem fixed_point_of_shifted_function 
  (f : ℝ → ℝ) 
  (h : f 1 = 1) :
  ∃ x : ℝ, f (x + 2) = x ∧ x = -1 := by
  sorry

end NUMINAMATH_CALUDE_fixed_point_of_shifted_function_l3198_319825


namespace NUMINAMATH_CALUDE_least_with_12_factors_l3198_319806

/-- The number of positive factors of a positive integer -/
def num_factors (n : ℕ+) : ℕ := sorry

/-- Check if a number has exactly 12 positive factors -/
def has_12_factors (n : ℕ+) : Prop := num_factors n = 12

/-- Theorem: 72 is the least positive integer with exactly 12 positive factors -/
theorem least_with_12_factors :
  (∀ m : ℕ+, m < 72 → ¬(has_12_factors m)) ∧ has_12_factors 72 := by sorry

end NUMINAMATH_CALUDE_least_with_12_factors_l3198_319806


namespace NUMINAMATH_CALUDE_empty_subset_of_intersection_l3198_319834

theorem empty_subset_of_intersection (A B : Set α) 
  (hA : A ≠ ∅) (hB : B ≠ ∅) (hAB : A ≠ B) : 
  ∅ ⊆ A ∩ B :=
sorry

end NUMINAMATH_CALUDE_empty_subset_of_intersection_l3198_319834


namespace NUMINAMATH_CALUDE_x_gt_2_sufficient_not_necessary_for_x_neq_2_l3198_319895

theorem x_gt_2_sufficient_not_necessary_for_x_neq_2 :
  (∃ x : ℝ, x ≠ 2 ∧ ¬(x > 2)) ∧
  (∀ x : ℝ, x > 2 → x ≠ 2) :=
by sorry

end NUMINAMATH_CALUDE_x_gt_2_sufficient_not_necessary_for_x_neq_2_l3198_319895


namespace NUMINAMATH_CALUDE_system_to_quadratic_l3198_319885

theorem system_to_quadratic (x y : ℝ) 
  (eq1 : 3 * x^2 + 9 * x + 4 * y + 2 = 0)
  (eq2 : 3 * x + y + 4 = 0) :
  y^2 + 11 * y - 14 = 0 := by
  sorry

end NUMINAMATH_CALUDE_system_to_quadratic_l3198_319885


namespace NUMINAMATH_CALUDE_complex_sum_magnitude_l3198_319848

theorem complex_sum_magnitude (a b c : ℂ) : 
  Complex.abs a = 1 → 
  Complex.abs b = 1 → 
  Complex.abs c = 1 → 
  a^3 / (b * c) + b^3 / (a * c) + c^3 / (a * b) = -3 → 
  Complex.abs (a + b + c) = 1 := by
  sorry

end NUMINAMATH_CALUDE_complex_sum_magnitude_l3198_319848


namespace NUMINAMATH_CALUDE_tina_homework_time_l3198_319878

/-- Tina's keyboard cleaning and homework problem -/
theorem tina_homework_time (initial_keys : ℕ) (cleaning_time_per_key : ℕ) 
  (remaining_keys : ℕ) (total_time : ℕ) : 
  initial_keys = 15 →
  cleaning_time_per_key = 3 →
  remaining_keys = 14 →
  total_time = 52 →
  total_time - (remaining_keys * cleaning_time_per_key) = 10 := by
  sorry

end NUMINAMATH_CALUDE_tina_homework_time_l3198_319878


namespace NUMINAMATH_CALUDE_cone_section_max_area_l3198_319884

/-- Given a cone whose lateral surface unfolds into a sector with radius 2 and central angle 5π/3,
    the maximum area of any section determined by two generatrices is 2. -/
theorem cone_section_max_area :
  ∀ (r : ℝ) (l : ℝ) (a : ℝ),
  r > 0 →
  l = 2 →
  2 * π * r = 10 * π / 3 →
  0 < a →
  a ≤ 10 / 3 →
  (∃ (h : ℝ), h > 0 ∧ h^2 + (a/2)^2 = l^2 ∧ 
    ∀ (S : ℝ), S = a * h / 2 → S ≤ 2) :=
by sorry

end NUMINAMATH_CALUDE_cone_section_max_area_l3198_319884


namespace NUMINAMATH_CALUDE_fish_tank_problem_l3198_319837

theorem fish_tank_problem (fish_taken_out fish_remaining : ℕ) 
  (h1 : fish_taken_out = 16) 
  (h2 : fish_remaining = 3) : 
  fish_taken_out + fish_remaining = 19 := by
  sorry

end NUMINAMATH_CALUDE_fish_tank_problem_l3198_319837


namespace NUMINAMATH_CALUDE_angle_bisector_quadrilateral_sum_l3198_319816

/-- Given a convex quadrilateral ABCD with angles α, β, γ, δ, 
    and its angle bisectors intersecting to form quadrilateral HIJE,
    the sum of opposite angles HIJ and JEH in HIJE is 180°. -/
theorem angle_bisector_quadrilateral_sum (α β γ δ : Real) 
  (h_convex : α > 0 ∧ β > 0 ∧ γ > 0 ∧ δ > 0)
  (h_sum : α + β + γ + δ = 360) : 
  (α/2 + β/2) + (γ/2 + δ/2) = 180 := by
  sorry

end NUMINAMATH_CALUDE_angle_bisector_quadrilateral_sum_l3198_319816


namespace NUMINAMATH_CALUDE_triangle_isosceles_or_right_l3198_319882

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively. -/
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real

/-- The theorem states that if a * cos(A) = b * cos(B) in a triangle,
    then the triangle is either isosceles (A = B) or right-angled (A + B = π/2). -/
theorem triangle_isosceles_or_right (t : Triangle) 
  (h : t.a * Real.cos t.A = t.b * Real.cos t.B) :
  t.A = t.B ∨ t.A + t.B = Real.pi / 2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_isosceles_or_right_l3198_319882


namespace NUMINAMATH_CALUDE_distance_AC_l3198_319852

-- Define the movement from A to C
def south_displacement : ℝ := 10
def east_displacement : ℝ := 5

-- Theorem statement
theorem distance_AC : Real.sqrt (south_displacement ^ 2 + east_displacement ^ 2) = 5 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_distance_AC_l3198_319852


namespace NUMINAMATH_CALUDE_inequality_proof_l3198_319870

theorem inequality_proof (a b c d : ℝ) 
  (pos_a : 0 < a) (pos_b : 0 < b) (pos_c : 0 < c) (pos_d : 0 < d)
  (sum_eq_one : a + b + c + d = 1) : 
  (a^2 + b^2 + c^2 + d) / (a + b + c)^3 + 
  (b^2 + c^2 + d^2 + a) / (b + c + d)^3 + 
  (c^2 + d^2 + a^2 + b) / (c + d + a)^3 + 
  (d^2 + a^2 + b^2 + c) / (d + a + b)^3 > 4 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3198_319870


namespace NUMINAMATH_CALUDE_real_part_of_z_l3198_319821

theorem real_part_of_z (z : ℂ) (h : Complex.I * z = 1 + 2 * Complex.I) : 
  (z.re : ℝ) = 2 := by sorry

end NUMINAMATH_CALUDE_real_part_of_z_l3198_319821


namespace NUMINAMATH_CALUDE_largest_number_with_nine_factors_l3198_319862

/-- A function that returns the number of positive factors of a natural number -/
def num_factors (n : ℕ) : ℕ := sorry

/-- A function that checks if a number is less than 150 -/
def less_than_150 (n : ℕ) : Prop := n < 150

/-- The theorem stating that 100 is the largest number less than 150 with exactly 9 factors -/
theorem largest_number_with_nine_factors :
  (∀ m : ℕ, less_than_150 m → num_factors m = 9 → m ≤ 100) ∧
  (less_than_150 100 ∧ num_factors 100 = 9) :=
sorry

end NUMINAMATH_CALUDE_largest_number_with_nine_factors_l3198_319862


namespace NUMINAMATH_CALUDE_quadratic_equation_negative_roots_l3198_319835

theorem quadratic_equation_negative_roots (m : ℝ) : 
  (∃ x₁ x₂ : ℝ, x₁ < 0 ∧ x₂ < 0 ∧ x₁ ≠ x₂ ∧ 
   ∀ x : ℝ, 3 * x^2 + 6 * x + m = 0 ↔ (x = x₁ ∨ x = x₂)) ↔ 
  (m = 1 ∨ m = 2 ∨ m = 3) :=
sorry

end NUMINAMATH_CALUDE_quadratic_equation_negative_roots_l3198_319835


namespace NUMINAMATH_CALUDE_officers_selection_count_l3198_319871

/-- The number of ways to select officers from a group -/
def select_officers (n : ℕ) (k : ℕ) : ℕ :=
  Nat.choose n k * Nat.factorial k

/-- Theorem: Selecting 3 officers from 4 people results in 24 ways -/
theorem officers_selection_count :
  select_officers 4 3 = 24 := by
  sorry

#eval select_officers 4 3  -- This should output 24

end NUMINAMATH_CALUDE_officers_selection_count_l3198_319871


namespace NUMINAMATH_CALUDE_sprite_volume_calculation_l3198_319883

def maaza_volume : ℕ := 50
def pepsi_volume : ℕ := 144
def total_cans : ℕ := 281

def can_volume : ℕ := Nat.gcd maaza_volume pepsi_volume

def maaza_cans : ℕ := maaza_volume / can_volume
def pepsi_cans : ℕ := pepsi_volume / can_volume

def sprite_cans : ℕ := total_cans - (maaza_cans + pepsi_cans)

def sprite_volume : ℕ := sprite_cans * can_volume

theorem sprite_volume_calculation :
  sprite_volume = 368 :=
by sorry

end NUMINAMATH_CALUDE_sprite_volume_calculation_l3198_319883


namespace NUMINAMATH_CALUDE_five_twelve_thirteen_pythagorean_triple_l3198_319813

/-- A Pythagorean triple is a set of three positive integers (a, b, c) that satisfy a² + b² = c² --/
def is_pythagorean_triple (a b c : ℕ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ a^2 + b^2 = c^2

/-- The set (5, 12, 13) is a Pythagorean triple --/
theorem five_twelve_thirteen_pythagorean_triple :
  is_pythagorean_triple 5 12 13 := by
  sorry

end NUMINAMATH_CALUDE_five_twelve_thirteen_pythagorean_triple_l3198_319813


namespace NUMINAMATH_CALUDE_defective_switch_probability_l3198_319820

/-- The probability of drawing a defective switch from a population,
    given the total number of switches, sample size, and number of defective switches in the sample. -/
def defective_probability (total : ℕ) (sample_size : ℕ) (defective_in_sample : ℕ) : ℚ :=
  defective_in_sample / sample_size

theorem defective_switch_probability :
  let total := 2000
  let sample_size := 100
  let defective_in_sample := 10
  defective_probability total sample_size defective_in_sample = 1/10 := by
sorry

end NUMINAMATH_CALUDE_defective_switch_probability_l3198_319820


namespace NUMINAMATH_CALUDE_factorization_equality_l3198_319803

theorem factorization_equality (x y : ℝ) : x * y^2 + 6 * x * y + 9 * x = x * (y + 3)^2 := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l3198_319803


namespace NUMINAMATH_CALUDE_ben_points_l3198_319840

theorem ben_points (zach_points ben_points : ℕ) 
  (h1 : zach_points = 42)
  (h2 : zach_points = ben_points + 21) : 
  ben_points = 21 := by
sorry

end NUMINAMATH_CALUDE_ben_points_l3198_319840


namespace NUMINAMATH_CALUDE_modulus_of_z_l3198_319861

theorem modulus_of_z (z : ℂ) : z = 5 / (1 - 2*I) → Complex.abs z = Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_modulus_of_z_l3198_319861


namespace NUMINAMATH_CALUDE_quadratic_inequality_l3198_319860

/-- A quadratic polynomial with no real roots -/
structure QuadraticNoRoots where
  a : ℝ
  b : ℝ
  c : ℝ
  a_pos : a > 0
  no_roots : ∀ x, a * x^2 + b * x + c > 0

/-- The quadratic function -/
def f (q : QuadraticNoRoots) (x : ℝ) : ℝ :=
  q.a * x^2 + q.b * x + q.c

/-- The main theorem -/
theorem quadratic_inequality (q : QuadraticNoRoots) :
  ∀ x, f q x + f q (x - 1) - f q (x + 1) > -4 * q.a := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_l3198_319860


namespace NUMINAMATH_CALUDE_certain_fraction_proof_l3198_319856

theorem certain_fraction_proof (n : ℚ) (x : ℚ) 
  (h1 : n = 0.5833333333333333)
  (h2 : n = x + 1/4) : 
  x = 0.3333333333333333 := by
sorry

end NUMINAMATH_CALUDE_certain_fraction_proof_l3198_319856


namespace NUMINAMATH_CALUDE_circle_equation_proof_l3198_319831

theorem circle_equation_proof :
  ∀ (x y : ℝ), (x^2 + 12*x + y^2 + 8*y + 3 = 0) ↔ 
  (∃ (h k : ℝ), (x - h)^2 + (y - k)^2 = 7^2) :=
by sorry

end NUMINAMATH_CALUDE_circle_equation_proof_l3198_319831


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l3198_319855

theorem sufficient_not_necessary : 
  (∃ x : ℝ, x < -1 → x^2 > 1) ∧ 
  (∃ x : ℝ, x^2 > 1 ∧ ¬(x < -1)) := by
  sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l3198_319855


namespace NUMINAMATH_CALUDE_frank_one_dollar_bills_l3198_319805

/-- Represents the number of bills Frank has of each denomination --/
structure Bills :=
  (ones : ℕ)
  (fives : ℕ)
  (tens : ℕ)
  (twenties : ℕ)

/-- Calculates the total value of bills --/
def totalValue (b : Bills) : ℕ :=
  b.ones + 5 * b.fives + 10 * b.tens + 20 * b.twenties

theorem frank_one_dollar_bills :
  ∃ (b : Bills),
    b.fives = 4 ∧
    b.tens = 2 ∧
    b.twenties = 1 ∧
    (∃ (peanutsPounds : ℕ),
      3 * peanutsPounds + 4 = 10 ∧  -- Cost of peanuts plus $4 change equals $10
      totalValue b + 4 = 54) →      -- Total money including change is $54
    b.ones = 4 := by
  sorry

end NUMINAMATH_CALUDE_frank_one_dollar_bills_l3198_319805


namespace NUMINAMATH_CALUDE_ribbon_used_parts_l3198_319873

-- Define the total length of the ribbon
def total_length : ℕ := 30

-- Define the number of parts the ribbon is cut into
def num_parts : ℕ := 6

-- Define the length of unused ribbon
def unused_length : ℕ := 10

-- Theorem to prove
theorem ribbon_used_parts : 
  ∃ (part_length : ℕ) (unused_parts : ℕ),
    part_length * num_parts = total_length ∧
    unused_parts * part_length = unused_length ∧
    num_parts - unused_parts = 4 :=
by sorry

end NUMINAMATH_CALUDE_ribbon_used_parts_l3198_319873


namespace NUMINAMATH_CALUDE_largest_common_term_l3198_319802

def is_common_term (a : ℕ) : Prop :=
  ∃ n m : ℕ, a = 4 + 5 * n ∧ a = 3 + 9 * m

def is_largest_common_term_under_1000 (a : ℕ) : Prop :=
  is_common_term a ∧ 
  a < 1000 ∧
  ∀ b : ℕ, is_common_term b → b < 1000 → b ≤ a

theorem largest_common_term :
  ∃ a : ℕ, is_largest_common_term_under_1000 a ∧ a = 984 :=
sorry

end NUMINAMATH_CALUDE_largest_common_term_l3198_319802


namespace NUMINAMATH_CALUDE_cubic_sum_theorem_l3198_319877

theorem cubic_sum_theorem (a b c : ℝ) 
  (h1 : a^3 + b^3 + c^3 = 3*a*b*c)
  (h2 : a^3 + b^3 + c^3 = 6)
  (h3 : a^2 + b^2 + c^2 = 8) :
  a*b/(a+b) + b*c/(b+c) + c*a/(c+a) = -8 :=
sorry

end NUMINAMATH_CALUDE_cubic_sum_theorem_l3198_319877


namespace NUMINAMATH_CALUDE_concentric_circles_diameter_l3198_319866

theorem concentric_circles_diameter (r₁ r₂ : ℝ) : 
  r₁ > 0 → r₂ > r₁ → 
  π * r₁^2 = 4 * π → 
  π * r₂^2 - π * r₁^2 = 4 * π → 
  2 * r₂ = 4 * Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_concentric_circles_diameter_l3198_319866


namespace NUMINAMATH_CALUDE_total_amount_is_152_l3198_319845

/-- Represents the share distribution among five individuals -/
structure ShareDistribution where
  p : ℝ
  q : ℝ
  r : ℝ
  s : ℝ
  t : ℝ

/-- The conditions of the problem -/
def satisfies_conditions (sd : ShareDistribution) : Prop :=
  sd.q = 35 ∧
  sd.q / sd.p = 1.75 / 2 ∧
  sd.r / sd.p = 1.5 / 2 ∧
  sd.s / sd.p = 1.25 / 2 ∧
  sd.t / sd.p = 1.1 / 2

/-- The theorem stating that the total amount is $152 -/
theorem total_amount_is_152 (sd : ShareDistribution) 
  (h : satisfies_conditions sd) : 
  sd.p + sd.q + sd.r + sd.s + sd.t = 152 := by
  sorry


end NUMINAMATH_CALUDE_total_amount_is_152_l3198_319845


namespace NUMINAMATH_CALUDE_function_property_l3198_319809

theorem function_property (α : ℝ) (hα : α > 0) :
  ∃ (b : ℝ), ∀ (f : ℕ+ → ℝ),
    (∀ (k m : ℕ+), α * m ≤ k ∧ k < (α + 1) * m → f (k + m) = f k + f m) ↔
    (∀ (n : ℕ+), f n = b * n) :=
by sorry

end NUMINAMATH_CALUDE_function_property_l3198_319809


namespace NUMINAMATH_CALUDE_probability_three_same_is_one_third_l3198_319864

/-- Represents the outcome of rolling five dice -/
structure DiceRoll :=
  (pair : Fin 6)
  (different : Fin 6)
  (reroll1 : Fin 6)
  (reroll2 : Fin 6)
  (pair_count : Nat)
  (different_from_pair : pair ≠ different)
  (rerolls_different : reroll1 ≠ reroll2)

/-- The probability of getting at least three dice with the same value after rerolling -/
def probability_three_same (roll : DiceRoll) : ℚ :=
  sorry

/-- Theorem stating that the probability is 1/3 -/
theorem probability_three_same_is_one_third :
  ∀ roll : DiceRoll, probability_three_same roll = 1/3 :=
sorry

end NUMINAMATH_CALUDE_probability_three_same_is_one_third_l3198_319864


namespace NUMINAMATH_CALUDE_ways_to_buy_three_items_l3198_319894

/-- Represents the inventory of a store selling computer peripherals -/
structure StoreInventory where
  headphones : ℕ
  mice : ℕ
  keyboards : ℕ
  keyboard_mouse_sets : ℕ
  headphone_mouse_sets : ℕ

/-- Calculates the number of ways to buy a headphone, a keyboard, and a mouse -/
def waysToButThreeItems (inventory : StoreInventory) : ℕ :=
  inventory.headphones * inventory.mice * inventory.keyboards +
  inventory.keyboard_mouse_sets * inventory.headphones +
  inventory.headphone_mouse_sets * inventory.keyboards

/-- The store's actual inventory -/
def actualInventory : StoreInventory := {
  headphones := 9
  mice := 13
  keyboards := 5
  keyboard_mouse_sets := 4
  headphone_mouse_sets := 5
}

theorem ways_to_buy_three_items :
  waysToButThreeItems actualInventory = 646 := by
  sorry

end NUMINAMATH_CALUDE_ways_to_buy_three_items_l3198_319894


namespace NUMINAMATH_CALUDE_probability_information_both_clubs_l3198_319898

def total_students : ℕ := 30
def art_club_students : ℕ := 22
def music_club_students : ℕ := 25

def probability_both_clubs : ℚ := 397 / 435

theorem probability_information_both_clubs :
  let students_in_both := art_club_students + music_club_students - total_students
  let students_only_art := art_club_students - students_in_both
  let students_only_music := music_club_students - students_in_both
  let total_combinations := total_students.choose 2
  let incompatible_combinations := students_only_art.choose 2 + students_only_music.choose 2
  (1 : ℚ) - (incompatible_combinations : ℚ) / total_combinations = probability_both_clubs :=
by sorry

end NUMINAMATH_CALUDE_probability_information_both_clubs_l3198_319898


namespace NUMINAMATH_CALUDE_tangent_line_to_ellipse_l3198_319801

/-- The equation of a line tangent to an ellipse -/
theorem tangent_line_to_ellipse (x y : ℝ) :
  let P : ℝ × ℝ := (1, Real.sqrt 3 / 2)
  let ellipse (x y : ℝ) : Prop := x^2 / 4 + y^2 = 1
  let line (x y : ℝ) : Prop := x + 2 * Real.sqrt 3 * y - 4 = 0
  (ellipse P.1 P.2) →  -- Point P is on the ellipse
  (∀ x y, line x y → (x - P.1) * P.1 / 4 + (y - P.2) * P.2 = 0) →  -- Line passes through P
  (∀ x y, ellipse x y → line x y → 
    ∀ ε > 0, ∃ δ > 0, ∀ x' y', 
      (x' - x)^2 + (y' - y)^2 < δ^2 → ¬(ellipse x' y' ∧ line x' y')) -- Line is tangent to ellipse
  := by sorry

end NUMINAMATH_CALUDE_tangent_line_to_ellipse_l3198_319801


namespace NUMINAMATH_CALUDE_berry_cobbler_cartons_l3198_319892

/-- The number of cartons of berries Maria needs for her cobbler -/
theorem berry_cobbler_cartons (strawberries blueberries additional : ℕ) 
  (h1 : strawberries = 4)
  (h2 : blueberries = 8)
  (h3 : additional = 9) :
  strawberries + blueberries + additional = 21 := by
  sorry

end NUMINAMATH_CALUDE_berry_cobbler_cartons_l3198_319892


namespace NUMINAMATH_CALUDE_pi_shape_points_for_10cm_square_l3198_319868

/-- The number of unique points on a "П" shape formed from a square --/
def unique_points_on_pi_shape (square_side : ℕ) (point_spacing : ℕ) : ℕ :=
  let points_per_side := square_side / point_spacing + 1
  let total_points := points_per_side * 3
  let corner_points := 3
  total_points - (corner_points - 1)

/-- Theorem stating that for a square with side 10 cm and points placed every 1 cm,
    the number of unique points on the "П" shape is 31 --/
theorem pi_shape_points_for_10cm_square :
  unique_points_on_pi_shape 10 1 = 31 := by
  sorry

end NUMINAMATH_CALUDE_pi_shape_points_for_10cm_square_l3198_319868


namespace NUMINAMATH_CALUDE_ticket_distribution_count_l3198_319849

/-- Represents a valid ticket distribution for a class -/
structure TicketDistribution :=
  (min : ℕ)
  (max : ℕ)

/-- The total number of tickets to be distributed -/
def total_tickets : ℕ := 18

/-- The ticket distribution constraints for each class -/
def class_constraints : List TicketDistribution := [
  ⟨1, 5⟩,  -- Class A
  ⟨1, 6⟩,  -- Class B
  ⟨2, 7⟩,  -- Class C
  ⟨4, 10⟩  -- Class D
]

/-- 
  Counts the number of ways to distribute tickets according to the given constraints
  @param total The total number of tickets to distribute
  @param constraints The list of constraints for each class
  @return The number of valid distributions
-/
def count_distributions (total : ℕ) (constraints : List TicketDistribution) : ℕ :=
  sorry  -- Proof implementation goes here

/-- The main theorem stating that there are 140 ways to distribute the tickets -/
theorem ticket_distribution_count : count_distributions total_tickets class_constraints = 140 :=
  sorry  -- Proof goes here

end NUMINAMATH_CALUDE_ticket_distribution_count_l3198_319849


namespace NUMINAMATH_CALUDE_people_who_left_l3198_319815

theorem people_who_left (initial_people : ℕ) (remaining_people : ℕ) : 
  initial_people = 11 → remaining_people = 5 → initial_people - remaining_people = 6 := by
  sorry

end NUMINAMATH_CALUDE_people_who_left_l3198_319815


namespace NUMINAMATH_CALUDE_relative_speed_object_image_l3198_319886

/-- Relative speed between an object and its image for a converging lens -/
theorem relative_speed_object_image 
  (f : ℝ) (t : ℝ) (v_object : ℝ) :
  f = 10 →
  t = 30 →
  v_object = 200 →
  let k := f * t / (t - f)
  let v_image := f^2 / (t - f)^2 * v_object
  |v_object + v_image| = 150 := by
  sorry

end NUMINAMATH_CALUDE_relative_speed_object_image_l3198_319886


namespace NUMINAMATH_CALUDE_power_of_twenty_l3198_319881

theorem power_of_twenty : (20 : ℕ) ^ ((20 : ℕ) / 2) = 102400000000000000000 := by
  sorry

end NUMINAMATH_CALUDE_power_of_twenty_l3198_319881


namespace NUMINAMATH_CALUDE_unique_remainder_in_range_l3198_319833

theorem unique_remainder_in_range : ∃! n : ℕ, n ≤ 100 ∧ n % 9 = 3 ∧ n % 13 = 5 ∧ n = 57 := by
  sorry

end NUMINAMATH_CALUDE_unique_remainder_in_range_l3198_319833


namespace NUMINAMATH_CALUDE_sum_of_jenna_and_darius_ages_l3198_319829

def sum_of_ages (jenna_age : ℕ) (darius_age : ℕ) : ℕ :=
  jenna_age + darius_age

theorem sum_of_jenna_and_darius_ages :
  ∀ (jenna_age : ℕ) (darius_age : ℕ),
    jenna_age = darius_age + 5 →
    jenna_age = 13 →
    darius_age = 8 →
    sum_of_ages jenna_age darius_age = 21 :=
by
  sorry

end NUMINAMATH_CALUDE_sum_of_jenna_and_darius_ages_l3198_319829


namespace NUMINAMATH_CALUDE_height_from_bisected_hypotenuse_l3198_319843

/-- 
Given a right-angled triangle where the bisector of the right angle divides 
the hypotenuse into segments p and q, the height m corresponding to the 
hypotenuse is equal to (p + q) * p * q / (p^2 + q^2).
-/
theorem height_from_bisected_hypotenuse (p q : ℝ) (hp : p > 0) (hq : q > 0) :
  ∃ m : ℝ, m = (p + q) * p * q / (p^2 + q^2) ∧ 
  m = Real.sqrt ((p + q)^2 * p^2 * q^2 / (p^2 + q^2)^2) :=
by sorry

end NUMINAMATH_CALUDE_height_from_bisected_hypotenuse_l3198_319843


namespace NUMINAMATH_CALUDE_range_of_m_l3198_319819

-- Define the sets P and S
def P : Set ℝ := {x | x^2 - 8*x - 20 ≤ 0}
def S (m : ℝ) : Set ℝ := {x | 1 - m ≤ x ∧ x ≤ 1 + m}

-- State the theorem
theorem range_of_m (m : ℝ) (h_nonempty : (S m).Nonempty) 
  (h_condition : ∀ x, x ∉ P → (x ∉ S m → True) ∧ (x ∉ S m → x ∉ P → False)) : 
  m ≥ 9 := by
  sorry

end NUMINAMATH_CALUDE_range_of_m_l3198_319819


namespace NUMINAMATH_CALUDE_decimal_to_fraction_sum_l3198_319865

theorem decimal_to_fraction_sum (a b : ℕ+) :
  (a : ℚ) / b = 0.31375 ∧ 
  Int.gcd a.val b.val = 1 →
  a.val + b.val = 1051 := by
  sorry

end NUMINAMATH_CALUDE_decimal_to_fraction_sum_l3198_319865


namespace NUMINAMATH_CALUDE_derivative_not_equivalent_l3198_319830

-- Define a real-valued function f
variable (f : ℝ → ℝ)

-- Assume f is differentiable
variable (hf : Differentiable ℝ f)

-- Define a point x₀
variable (x₀ : ℝ)

-- Statement: f'(x₀) is not equivalent to [f(x₀)]'
theorem derivative_not_equivalent :
  ¬(∀ (f : ℝ → ℝ) (hf : Differentiable ℝ f) (x₀ : ℝ), 
    deriv f x₀ = deriv (λ _ => f x₀) x₀) :=
sorry

end NUMINAMATH_CALUDE_derivative_not_equivalent_l3198_319830


namespace NUMINAMATH_CALUDE_kellys_grade_is_42_l3198_319832

/-- Calculates Kelly's grade based on the grades of Jenny, Jason, and Bob -/
def kellysGrade (jennysGrade : ℕ) : ℕ :=
  let jasonsGrade := jennysGrade - 25
  let bobsGrade := jasonsGrade / 2
  let kellyGradeIncrease := bobsGrade * 20 / 100
  bobsGrade + kellyGradeIncrease

/-- Theorem stating that Kelly's grade is 42 given the conditions in the problem -/
theorem kellys_grade_is_42 : kellysGrade 95 = 42 := by
  sorry

end NUMINAMATH_CALUDE_kellys_grade_is_42_l3198_319832


namespace NUMINAMATH_CALUDE_log_equation_solution_l3198_319863

theorem log_equation_solution :
  ∀ y : ℝ, (Real.log y + 3 * Real.log 5 = 1) ↔ (y = 2/25) :=
by sorry

end NUMINAMATH_CALUDE_log_equation_solution_l3198_319863


namespace NUMINAMATH_CALUDE_music_festival_group_count_l3198_319817

/-- The number of ways to select k items from n items -/
def binomial (n k : ℕ) : ℕ := sorry

/-- The number of ways to select a group for the music festival -/
def musicFestivalGroupWays : ℕ :=
  binomial 6 2 * binomial 4 2 * binomial 4 1

theorem music_festival_group_count :
  musicFestivalGroupWays = 360 := by sorry

end NUMINAMATH_CALUDE_music_festival_group_count_l3198_319817


namespace NUMINAMATH_CALUDE_complex_number_quadrant_l3198_319828

theorem complex_number_quadrant (m : ℝ) (h1 : 1 < m) (h2 : m < 3/2) :
  let z : ℂ := (3 + I) - m * (2 + I)
  (z.re > 0) ∧ (z.im < 0) :=
by sorry

end NUMINAMATH_CALUDE_complex_number_quadrant_l3198_319828


namespace NUMINAMATH_CALUDE_number_difference_l3198_319857

theorem number_difference (x y : ℝ) (h1 : x + y = 30) (h2 : x * y = 216) : 
  |x - y| = 6 := by sorry

end NUMINAMATH_CALUDE_number_difference_l3198_319857


namespace NUMINAMATH_CALUDE_sum_in_range_l3198_319891

theorem sum_in_range : 
  let sum := (5/4 : ℚ) + (13/3 : ℚ) + (73/12 : ℚ)
  11.5 < sum ∧ sum < 12 := by sorry

end NUMINAMATH_CALUDE_sum_in_range_l3198_319891


namespace NUMINAMATH_CALUDE_product_abcd_is_zero_l3198_319897

theorem product_abcd_is_zero
  (a b c d : ℤ)
  (eq1 : 3*a + 2*b + 4*c + 8*d = 40)
  (eq2 : 4*(d+c) = b)
  (eq3 : 2*b + 2*c = a)
  (eq4 : c + 1 = d) :
  a * b * c * d = 0 := by
  sorry

end NUMINAMATH_CALUDE_product_abcd_is_zero_l3198_319897


namespace NUMINAMATH_CALUDE_complement_S_union_T_eq_less_equal_one_l3198_319836

-- Define the sets S and T
def S : Set ℝ := {x | x > -2}
def T : Set ℝ := {x | x^2 + 3*x - 4 ≤ 0}

-- State the theorem
theorem complement_S_union_T_eq_less_equal_one :
  (Set.univ \ S) ∪ T = {x : ℝ | x ≤ 1} := by sorry

end NUMINAMATH_CALUDE_complement_S_union_T_eq_less_equal_one_l3198_319836


namespace NUMINAMATH_CALUDE_special_sequence_2000th_term_l3198_319812

/-- A sequence where the sum of any three consecutive terms is 20 -/
def SpecialSequence (x : ℕ → ℝ) : Prop :=
  ∀ n, x n + x (n + 1) + x (n + 2) = 20

theorem special_sequence_2000th_term
  (x : ℕ → ℝ)
  (h_special : SpecialSequence x)
  (h_x1 : x 1 = 9)
  (h_x12 : x 12 = 7) :
  x 2000 = 4 := by
sorry

end NUMINAMATH_CALUDE_special_sequence_2000th_term_l3198_319812


namespace NUMINAMATH_CALUDE_max_min_f_on_interval_l3198_319850

-- Define the function f(x) = x³ - 3x + 1
def f (x : ℝ) : ℝ := x^3 - 3*x + 1

-- Define the closed interval [-3, 0]
def interval : Set ℝ := Set.Icc (-3) 0

-- Theorem statement
theorem max_min_f_on_interval :
  ∃ (max min : ℝ), 
    (∀ x ∈ interval, f x ≤ max) ∧ 
    (∃ x ∈ interval, f x = max) ∧
    (∀ x ∈ interval, min ≤ f x) ∧ 
    (∃ x ∈ interval, f x = min) ∧
    max = 3 ∧ min = -17 := by
  sorry

end NUMINAMATH_CALUDE_max_min_f_on_interval_l3198_319850


namespace NUMINAMATH_CALUDE_least_subtraction_for_divisibility_l3198_319879

theorem least_subtraction_for_divisibility (n : ℕ) (h : n = 165826) :
  ∃ x : ℕ, x = 2 ∧ 
    (∀ y : ℕ, y < x → ¬(4 ∣ (n - y))) ∧
    (4 ∣ (n - x)) :=
by sorry

end NUMINAMATH_CALUDE_least_subtraction_for_divisibility_l3198_319879


namespace NUMINAMATH_CALUDE_double_burgers_count_l3198_319859

/-- Represents the number of single burgers -/
def S : ℕ := sorry

/-- Represents the number of double burgers -/
def D : ℕ := sorry

/-- The total number of burgers -/
def total_burgers : ℕ := 50

/-- The cost of a single burger in cents -/
def single_cost : ℕ := 100

/-- The cost of a double burger in cents -/
def double_cost : ℕ := 150

/-- The total cost of all burgers in cents -/
def total_cost : ℕ := 6650

theorem double_burgers_count :
  S + D = total_burgers ∧
  S * single_cost + D * double_cost = total_cost →
  D = 33 := by sorry

end NUMINAMATH_CALUDE_double_burgers_count_l3198_319859


namespace NUMINAMATH_CALUDE_four_digit_square_and_cube_l3198_319807

theorem four_digit_square_and_cube (a : ℕ) :
  (1000 ≤ 4 * a^2) ∧ (4 * a^2 < 10000) ∧
  (1000 ≤ (4 / 3) * a^3) ∧ ((4 / 3) * a^3 < 10000) ∧
  (∃ (n : ℕ), (4 / 3) * a^3 = n) →
  a = 18 := by sorry

end NUMINAMATH_CALUDE_four_digit_square_and_cube_l3198_319807


namespace NUMINAMATH_CALUDE_sum_of_two_numbers_l3198_319847

theorem sum_of_two_numbers (a b c : ℝ) 
  (h1 : |a - b| ≥ |c|) 
  (h2 : |b - c| ≥ |a|) 
  (h3 : |c - a| ≥ |b|) : 
  a = b + c ∨ b = a + c ∨ c = a + b := by
  sorry

end NUMINAMATH_CALUDE_sum_of_two_numbers_l3198_319847
