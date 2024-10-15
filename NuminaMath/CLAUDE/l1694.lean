import Mathlib

namespace NUMINAMATH_CALUDE_parallel_planes_k_value_l1694_169434

/-- Given two planes α and β with normal vectors n₁ and n₂ respectively,
    prove that if the planes are parallel, then k = 6. -/
theorem parallel_planes_k_value (n₁ n₂ : ℝ × ℝ × ℝ) (k : ℝ) :
  n₁ = (1, 2, -3) →
  n₂ = (-2, -4, k) →
  (∃ (c : ℝ), c ≠ 0 ∧ n₁ = c • n₂) →
  k = 6 := by sorry

end NUMINAMATH_CALUDE_parallel_planes_k_value_l1694_169434


namespace NUMINAMATH_CALUDE_kate_change_l1694_169440

/-- The amount Kate gave to the clerk in cents -/
def amount_given : ℕ := 100

/-- The cost of Kate's candy in cents -/
def candy_cost : ℕ := 54

/-- The change Kate should receive in cents -/
def change : ℕ := amount_given - candy_cost

theorem kate_change : change = 46 := by
  sorry

end NUMINAMATH_CALUDE_kate_change_l1694_169440


namespace NUMINAMATH_CALUDE_percentage_calculation_l1694_169433

theorem percentage_calculation (total : ℝ) (result : ℝ) (percentage : ℝ) :
  total = 50 →
  result = 2.125 →
  percentage = 4.25 →
  (percentage / 100) * total = result :=
by
  sorry

end NUMINAMATH_CALUDE_percentage_calculation_l1694_169433


namespace NUMINAMATH_CALUDE_fundraiser_goal_reached_l1694_169452

/-- Proves that the total amount raised by the group is equal to the total amount needed for the trip. -/
theorem fundraiser_goal_reached (
  num_students : ℕ) 
  (individual_cost : ℕ)
  (collective_expenses : ℕ)
  (day1_raised : ℕ)
  (day2_raised : ℕ)
  (day3_raised : ℕ)
  (num_half_days : ℕ)
  (h1 : num_students = 6)
  (h2 : individual_cost = 450)
  (h3 : collective_expenses = 3000)
  (h4 : day1_raised = 600)
  (h5 : day2_raised = 900)
  (h6 : day3_raised = 400)
  (h7 : num_half_days = 4) :
  (num_students * individual_cost + collective_expenses) = 
  (day1_raised + day2_raised + day3_raised + 
   num_half_days * ((day1_raised + day2_raised + day3_raised) / 2)) := by
  sorry

#eval 6 * 450 + 3000 -- Total needed
#eval 600 + 900 + 400 + 4 * ((600 + 900 + 400) / 2) -- Total raised

end NUMINAMATH_CALUDE_fundraiser_goal_reached_l1694_169452


namespace NUMINAMATH_CALUDE_circle_intersection_theorem_l1694_169405

-- Define the types for points and circles
variable (Point Circle : Type)

-- Define the radius function for circles
variable (radius : Circle → ℝ)

-- Define the distance function between two points
variable (dist : Point → Point → ℝ)

-- Define the "on_circle" predicate
variable (on_circle : Point → Circle → Prop)

-- Define the "intersect" predicate for two circles
variable (intersect : Circle → Circle → Point → Prop)

-- Define the "interior_point" predicate
variable (interior_point : Point → Circle → Prop)

-- Define the "line_intersect" predicate
variable (line_intersect : Point → Point → Circle → Point → Prop)

-- Define the "equilateral" predicate for triangles
variable (equilateral : Point → Point → Point → Prop)

-- Theorem statement
theorem circle_intersection_theorem 
  (k₁ k₂ : Circle) (O A B S T : Point) (r : ℝ) :
  radius k₁ = r →
  on_circle O k₁ →
  intersect k₁ k₂ A →
  intersect k₁ k₂ B →
  interior_point S k₁ →
  line_intersect B S k₁ T →
  equilateral A O S →
  dist T S = r :=
sorry

end NUMINAMATH_CALUDE_circle_intersection_theorem_l1694_169405


namespace NUMINAMATH_CALUDE_pencil_count_l1694_169442

theorem pencil_count (pens pencils : ℕ) 
  (h_ratio : pens * 6 = pencils * 5)
  (h_difference : pencils = pens + 4) :
  pencils = 24 := by
sorry

end NUMINAMATH_CALUDE_pencil_count_l1694_169442


namespace NUMINAMATH_CALUDE_product_calculation_l1694_169417

theorem product_calculation : 10 * 0.2 * 0.5 * 4 / 2 = 2 := by
  sorry

end NUMINAMATH_CALUDE_product_calculation_l1694_169417


namespace NUMINAMATH_CALUDE_increasing_digits_mod_1000_l1694_169428

/-- The number of 8-digit positive integers with digits in increasing order -/
def count_increasing_digits : ℕ := (Nat.choose 17 8)

/-- The theorem stating that the count of such integers is congruent to 310 modulo 1000 -/
theorem increasing_digits_mod_1000 :
  count_increasing_digits % 1000 = 310 := by sorry

end NUMINAMATH_CALUDE_increasing_digits_mod_1000_l1694_169428


namespace NUMINAMATH_CALUDE_nested_root_simplification_l1694_169423

theorem nested_root_simplification (y : ℝ) (h : y ≥ 0) :
  Real.sqrt (y * Real.sqrt (y^3 * Real.sqrt (y^5))) = (y^15)^(1/8) := by
  sorry

end NUMINAMATH_CALUDE_nested_root_simplification_l1694_169423


namespace NUMINAMATH_CALUDE_classroom_gpa_l1694_169465

theorem classroom_gpa (total_students : ℕ) (gpa1 gpa2 gpa3 : ℚ) 
  (h1 : total_students = 60)
  (h2 : gpa1 = 54)
  (h3 : gpa2 = 48)
  (h4 : gpa3 = 45)
  (h5 : (total_students : ℚ) / 3 * gpa1 + (total_students : ℚ) / 4 * gpa2 + 
        (total_students - (total_students / 3 + total_students / 4) : ℚ) * gpa3 = 
        total_students * 48.75) : 
  (((total_students : ℚ) / 3 * gpa1 + (total_students : ℚ) / 4 * gpa2 + 
    (total_students - (total_students / 3 + total_students / 4) : ℚ) * gpa3) / total_students) = 48.75 :=
by sorry

end NUMINAMATH_CALUDE_classroom_gpa_l1694_169465


namespace NUMINAMATH_CALUDE_plant_purchase_solution_l1694_169409

/-- Represents the prices and quantities of plants A and B -/
structure PlantPurchase where
  price_a : ℝ
  price_b : ℝ
  quantity_a : ℕ
  quantity_b : ℕ

/-- Calculates the total cost of a plant purchase -/
def total_cost (p : PlantPurchase) : ℝ :=
  p.price_a * p.quantity_a + p.price_b * p.quantity_b

/-- Represents the given conditions from the problem -/
structure ProblemConditions where
  first_phase : PlantPurchase
  second_phase : PlantPurchase
  total_cost_both_phases : ℝ

/-- The main theorem representing the problem and its solution -/
theorem plant_purchase_solution (conditions : ProblemConditions) 
  (h1 : conditions.first_phase.quantity_a = 30)
  (h2 : conditions.first_phase.quantity_b = 15)
  (h3 : total_cost conditions.first_phase = 675)
  (h4 : conditions.second_phase.quantity_a = 12)
  (h5 : conditions.second_phase.quantity_b = 5)
  (h6 : conditions.total_cost_both_phases = 940)
  (h7 : conditions.first_phase.price_a = conditions.second_phase.price_a)
  (h8 : conditions.first_phase.price_b = conditions.second_phase.price_b) :
  ∃ (optimal_plan : PlantPurchase),
    conditions.first_phase.price_a = 20 ∧
    conditions.first_phase.price_b = 5 ∧
    optimal_plan.quantity_a + optimal_plan.quantity_b = 31 ∧
    optimal_plan.quantity_b < 2 * optimal_plan.quantity_a ∧
    total_cost optimal_plan = 320 ∧
    ∀ (other_plan : PlantPurchase),
      other_plan.quantity_a + other_plan.quantity_b = 31 →
      other_plan.quantity_b < 2 * other_plan.quantity_a →
      total_cost other_plan ≥ total_cost optimal_plan := by
  sorry

end NUMINAMATH_CALUDE_plant_purchase_solution_l1694_169409


namespace NUMINAMATH_CALUDE_new_person_weight_l1694_169462

/-- Given a group of 8 people, if replacing a person weighing 45 kg with a new person
    increases the average weight by 2.5 kg, then the weight of the new person is 65 kg. -/
theorem new_person_weight (initial_count : ℕ) (weight_increase : ℝ) (replaced_weight : ℝ) :
  initial_count = 8 →
  weight_increase = 2.5 →
  replaced_weight = 45 →
  (initial_count : ℝ) * weight_increase + replaced_weight = 65 :=
by sorry

end NUMINAMATH_CALUDE_new_person_weight_l1694_169462


namespace NUMINAMATH_CALUDE_cube_to_sphere_surface_area_ratio_l1694_169493

theorem cube_to_sphere_surface_area_ratio :
  ∀ (a R : ℝ), a > 0 → R > 0 →
  (a^3 = (4/3) * π * R^3) →
  ((6 * a^2) / (4 * π * R^2) = 3 * (6/π)) :=
by sorry

end NUMINAMATH_CALUDE_cube_to_sphere_surface_area_ratio_l1694_169493


namespace NUMINAMATH_CALUDE_total_collected_is_4336_5_l1694_169458

/-- Represents the total amount collected by Mark during the week in US dollars -/
def total_collected : ℝ :=
  let households_per_day : ℕ := 60
  let days : ℕ := 7
  let total_households : ℕ := households_per_day * days
  let usd_20_percent : ℝ := 0.25
  let eur_15_percent : ℝ := 0.15
  let gbp_10_percent : ℝ := 0.10
  let both_percent : ℝ := 0.05
  let no_donation_percent : ℝ := 0.30
  let usd_20_amount : ℝ := 20
  let eur_15_amount : ℝ := 15
  let gbp_10_amount : ℝ := 10
  let eur_to_usd : ℝ := 1.1
  let gbp_to_usd : ℝ := 1.3

  let usd_20_donation := (usd_20_percent * total_households) * usd_20_amount
  let eur_15_donation := (eur_15_percent * total_households) * eur_15_amount * eur_to_usd
  let gbp_10_donation := (gbp_10_percent * total_households) * gbp_10_amount * gbp_to_usd
  let both_donation := (both_percent * total_households) * (usd_20_amount + eur_15_amount * eur_to_usd)

  usd_20_donation + eur_15_donation + gbp_10_donation + both_donation

theorem total_collected_is_4336_5 :
  total_collected = 4336.5 := by
  sorry

end NUMINAMATH_CALUDE_total_collected_is_4336_5_l1694_169458


namespace NUMINAMATH_CALUDE_polynomial_transformation_l1694_169475

theorem polynomial_transformation (x y : ℝ) (h : y = x + 1/x) :
  x^4 + x^3 - 4*x^2 + x + 1 = 0 ↔ x^2 * (y^2 + y - 6) = 0 :=
by sorry

end NUMINAMATH_CALUDE_polynomial_transformation_l1694_169475


namespace NUMINAMATH_CALUDE_largest_common_divisor_under_60_l1694_169418

theorem largest_common_divisor_under_60 : 
  ∃ (n : ℕ), n ∣ 456 ∧ n ∣ 108 ∧ n < 60 ∧ 
  ∀ (m : ℕ), m ∣ 456 → m ∣ 108 → m < 60 → m ≤ n :=
by
  -- The proof would go here
  sorry

end NUMINAMATH_CALUDE_largest_common_divisor_under_60_l1694_169418


namespace NUMINAMATH_CALUDE_sin_bounded_difference_l1694_169460

theorem sin_bounded_difference (a : ℝ) : 
  ∃ (x₀ : ℝ), x₀ > 0 ∧ ∀ (x : ℝ), x > 0 → |Real.sin x - a| ≤ |Real.sin x₀ - a| := by
  sorry

end NUMINAMATH_CALUDE_sin_bounded_difference_l1694_169460


namespace NUMINAMATH_CALUDE_perfect_square_condition_l1694_169436

/-- A polynomial is a perfect square trinomial if it can be written as (px + q)^2 for some real p and q. -/
def is_perfect_square_trinomial (a b c : ℝ) : Prop :=
  ∃ p q : ℝ, ∀ x, a * x^2 + b * x + c = (p * x + q)^2

/-- If 4x^2 + bx + 1 is a perfect square trinomial, then b = ±4. -/
theorem perfect_square_condition (b : ℝ) :
  is_perfect_square_trinomial 4 b 1 → b = 4 ∨ b = -4 := by
  sorry

#check perfect_square_condition

end NUMINAMATH_CALUDE_perfect_square_condition_l1694_169436


namespace NUMINAMATH_CALUDE_sector_angle_measure_l1694_169490

theorem sector_angle_measure (r : ℝ) (l : ℝ) :
  (2 * r + l = 12) →
  (1 / 2 * l * r = 8) →
  (l / r = 1 ∨ l / r = 4) :=
sorry

end NUMINAMATH_CALUDE_sector_angle_measure_l1694_169490


namespace NUMINAMATH_CALUDE_fireflies_remaining_joined_fireflies_l1694_169459

/-- The number of fireflies remaining after some join and some leave --/
def remaining_fireflies (initial : ℕ) (joined : ℕ) (left : ℕ) : ℕ :=
  initial + joined - left

/-- Proof that 9 fireflies remain given the initial conditions --/
theorem fireflies_remaining : remaining_fireflies 3 8 2 = 9 := by
  sorry

/-- The number of fireflies that joined is 4 less than a dozen --/
theorem joined_fireflies : (12 : ℕ) - 4 = 8 := by
  sorry

end NUMINAMATH_CALUDE_fireflies_remaining_joined_fireflies_l1694_169459


namespace NUMINAMATH_CALUDE_sum_of_distances_l1694_169419

/-- Given two line segments AB and A'B', with points D and D' on them respectively,
    and a point P on AB, prove that the sum of PD and P'D' is 10/3 units. -/
theorem sum_of_distances (AB A'B' AD A'D' PD : ℝ) (h1 : AB = 8)
    (h2 : A'B' = 6) (h3 : AD = 3) (h4 : A'D' = 1) (h5 : PD = 2)
    (h6 : PD / P'D' = 3 / 2) : PD + P'D' = 10 / 3 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_distances_l1694_169419


namespace NUMINAMATH_CALUDE_max_sum_with_reciprocals_l1694_169491

theorem max_sum_with_reciprocals (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : x + y + 1/x + 1/y = 5) : 
  x + y ≤ 4 ∧ ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ a + b + 1/a + 1/b = 5 ∧ a + b = 4 := by
  sorry

#check max_sum_with_reciprocals

end NUMINAMATH_CALUDE_max_sum_with_reciprocals_l1694_169491


namespace NUMINAMATH_CALUDE_fixed_point_theorem_l1694_169446

/-- Given a triangle ABC and a point M, prove that a certain line always passes through a fixed point -/
theorem fixed_point_theorem (a b c t m : ℝ) : 
  let A : ℝ × ℝ := (0, a)
  let B : ℝ × ℝ := (b, 0)
  let C : ℝ × ℝ := (c, 0)
  let M : ℝ × ℝ := (t, m)
  let D : ℝ × ℝ := ((b + c) / 3, a / 3)  -- Centroid
  let E : ℝ × ℝ := ((t + b) / 2, m / 2)  -- Midpoint of MB
  let F : ℝ × ℝ := ((t + c) / 2, m / 2)  -- Midpoint of MC
  let P : ℝ × ℝ := ((t + b) / 2, a * (1 - (t + b) / (2 * b)))  -- Intersection of AB and perpendicular through E
  let Q : ℝ × ℝ := ((t + c) / 2, a * (1 - (t + c) / (2 * c)))  -- Intersection of AC and perpendicular through F
  let slope_PQ : ℝ := (a * t) / (b * c)
  let perpendicular_slope : ℝ := -b * c / (a * t)
  True → ∃ k : ℝ, (0, m + b * c / a) = (t + k, m + k * perpendicular_slope) :=
by
  sorry


end NUMINAMATH_CALUDE_fixed_point_theorem_l1694_169446


namespace NUMINAMATH_CALUDE_point_in_second_quadrant_implies_m_range_l1694_169484

/-- A point in the Cartesian plane is in the second quadrant if and only if its x-coordinate is negative and its y-coordinate is positive. -/
def is_in_second_quadrant (x y : ℝ) : Prop := x < 0 ∧ y > 0

/-- Given a real number m, prove that if the point P(m-3, m+1) is in the second quadrant,
    then -1 < m and m < 3. -/
theorem point_in_second_quadrant_implies_m_range (m : ℝ) :
  is_in_second_quadrant (m - 3) (m + 1) → -1 < m ∧ m < 3 := by
  sorry


end NUMINAMATH_CALUDE_point_in_second_quadrant_implies_m_range_l1694_169484


namespace NUMINAMATH_CALUDE_existence_of_point_l1694_169426

theorem existence_of_point (f : ℝ → ℝ) (hf : Differentiable ℝ f) :
  ∃ x ∈ Set.Icc 0 1, (4 / Real.pi) * (f 1 - f 0) = (1 + x^2) * (deriv f x) := by
  sorry

end NUMINAMATH_CALUDE_existence_of_point_l1694_169426


namespace NUMINAMATH_CALUDE_cube_edge_length_l1694_169480

theorem cube_edge_length (V : ℝ) (h : V = 32 * Real.pi / 3) :
  ∃ s : ℝ, s = 4 * Real.sqrt 3 / 3 ∧ V = 4 * Real.pi * (s * Real.sqrt 3 / 2)^3 / 3 := by
  sorry

end NUMINAMATH_CALUDE_cube_edge_length_l1694_169480


namespace NUMINAMATH_CALUDE_gcd_of_polynomial_and_multiple_l1694_169432

theorem gcd_of_polynomial_and_multiple : ∀ x : ℤ, 
  18432 ∣ x → 
  Nat.gcd (Int.natAbs ((3*x+5)*(7*x+2)*(13*x+7)*(2*x+10))) (Int.natAbs x) = 28 :=
by sorry

end NUMINAMATH_CALUDE_gcd_of_polynomial_and_multiple_l1694_169432


namespace NUMINAMATH_CALUDE_baking_difference_l1694_169431

/-- Calculates the difference between remaining flour to be added and total sugar required -/
def flour_sugar_difference (total_flour sugar_required flour_added : ℕ) : ℤ :=
  (total_flour - flour_added : ℤ) - sugar_required

/-- Proves that the difference between remaining flour and total sugar is 1 cup -/
theorem baking_difference (total_flour sugar_required flour_added : ℕ) 
  (h1 : total_flour = 10)
  (h2 : sugar_required = 2)
  (h3 : flour_added = 7) :
  flour_sugar_difference total_flour sugar_required flour_added = 1 := by
  sorry

end NUMINAMATH_CALUDE_baking_difference_l1694_169431


namespace NUMINAMATH_CALUDE_tan_five_pi_fourths_l1694_169449

theorem tan_five_pi_fourths : Real.tan (5 * π / 4) = 1 := by
  sorry

end NUMINAMATH_CALUDE_tan_five_pi_fourths_l1694_169449


namespace NUMINAMATH_CALUDE_model2_best_fit_l1694_169408

-- Define the coefficient of determination for each model
def R2_model1 : ℝ := 0.78
def R2_model2 : ℝ := 0.85
def R2_model3 : ℝ := 0.61
def R2_model4 : ℝ := 0.31

-- Define a function to calculate the distance from 1
def distance_from_one (x : ℝ) : ℝ := |1 - x|

-- Theorem stating that Model 2 has the best fitting effect
theorem model2_best_fit :
  distance_from_one R2_model2 < distance_from_one R2_model1 ∧
  distance_from_one R2_model2 < distance_from_one R2_model3 ∧
  distance_from_one R2_model2 < distance_from_one R2_model4 :=
by sorry


end NUMINAMATH_CALUDE_model2_best_fit_l1694_169408


namespace NUMINAMATH_CALUDE_three_lines_intersection_l1694_169450

theorem three_lines_intersection (x : ℝ) : 
  (∀ (a b c d e f : ℝ), a = x ∧ b = x ∧ c = x ∧ d = x ∧ e = x ∧ f = x) →  -- opposite angles are equal
  (a + b + c + d + e + f = 360) →                                       -- sum of angles around a point is 360°
  x = 60 :=                                                            -- prove that x = 60°
by
  sorry

end NUMINAMATH_CALUDE_three_lines_intersection_l1694_169450


namespace NUMINAMATH_CALUDE_sum_of_five_cubes_l1694_169416

theorem sum_of_five_cubes (n : ℤ) : ∃ (a b c d e : ℤ), n = a^3 + b^3 + c^3 + d^3 + e^3 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_five_cubes_l1694_169416


namespace NUMINAMATH_CALUDE_decimal_93_to_binary_binary_to_decimal_93_l1694_169457

/-- Converts a natural number to its binary representation as a list of bits -/
def toBinary (n : ℕ) : List Bool :=
  if n = 0 then [false]
  else
    let rec toBinaryAux (m : ℕ) : List Bool :=
      if m = 0 then []
      else (m % 2 = 1) :: toBinaryAux (m / 2)
    toBinaryAux n

/-- Converts a list of bits to its decimal representation -/
def fromBinary (bits : List Bool) : ℕ :=
  bits.foldl (fun acc b => 2 * acc + if b then 1 else 0) 0

theorem decimal_93_to_binary :
  toBinary 93 = [true, false, true, true, true, false, true] := by
  sorry

theorem binary_to_decimal_93 :
  fromBinary [true, false, true, true, true, false, true] = 93 := by
  sorry

end NUMINAMATH_CALUDE_decimal_93_to_binary_binary_to_decimal_93_l1694_169457


namespace NUMINAMATH_CALUDE_decimal_multiplication_l1694_169414

theorem decimal_multiplication : (0.5 : ℝ) * 0.7 = 0.35 := by sorry

end NUMINAMATH_CALUDE_decimal_multiplication_l1694_169414


namespace NUMINAMATH_CALUDE_tangent_ellipse_hyperbola_l1694_169412

-- Define the ellipse equation
def ellipse (x y : ℝ) : Prop := x^2 + 9*y^2 = 9

-- Define the hyperbola equation
def hyperbola (x y m : ℝ) : Prop := x^2 - m*(y+1)^2 = 1

-- Define the tangency condition
def are_tangent (m : ℝ) : Prop :=
  ∃ x y, ellipse x y ∧ hyperbola x y m

-- Theorem statement
theorem tangent_ellipse_hyperbola :
  ∀ m, are_tangent m → m = 2 := by
  sorry

end NUMINAMATH_CALUDE_tangent_ellipse_hyperbola_l1694_169412


namespace NUMINAMATH_CALUDE_investment_plans_count_l1694_169406

theorem investment_plans_count (n_projects : ℕ) (n_cities : ℕ) (max_per_city : ℕ) : 
  n_projects = 3 → n_cities = 5 → max_per_city = 2 →
  (Nat.choose n_cities 3 * Nat.factorial 3 + 
   Nat.choose n_cities 1 * Nat.choose (n_cities - 1) 1 * 3) = 120 := by
  sorry

end NUMINAMATH_CALUDE_investment_plans_count_l1694_169406


namespace NUMINAMATH_CALUDE_M_mod_1000_l1694_169498

/-- Number of blue flags -/
def blue_flags : ℕ := 12

/-- Number of green flags -/
def green_flags : ℕ := 9

/-- Total number of flags -/
def total_flags : ℕ := blue_flags + green_flags

/-- Number of flagpoles -/
def flagpoles : ℕ := 2

/-- Function to calculate the number of distinguishable arrangements -/
noncomputable def M : ℕ := sorry

/-- Theorem stating the remainder when M is divided by 1000 -/
theorem M_mod_1000 : M % 1000 = 596 := by sorry

end NUMINAMATH_CALUDE_M_mod_1000_l1694_169498


namespace NUMINAMATH_CALUDE_original_prices_theorem_l1694_169453

def shirt_discount : Float := 0.20
def shoes_discount : Float := 0.30
def jacket_discount : Float := 0.10

def discounted_shirt_price : Float := 780
def discounted_shoes_price : Float := 2100
def discounted_jacket_price : Float := 2700

def original_shirt_price : Float := discounted_shirt_price / (1 - shirt_discount)
def original_shoes_price : Float := discounted_shoes_price / (1 - shoes_discount)
def original_jacket_price : Float := discounted_jacket_price / (1 - jacket_discount)

theorem original_prices_theorem :
  original_shirt_price = 975 ∧
  original_shoes_price = 3000 ∧
  original_jacket_price = 3000 :=
by sorry

end NUMINAMATH_CALUDE_original_prices_theorem_l1694_169453


namespace NUMINAMATH_CALUDE_contractor_problem_l1694_169444

/-- The number of days initially planned to complete the work -/
def initial_days : ℕ := 15

/-- The number of absent laborers -/
def absent_laborers : ℕ := 5

/-- The number of days taken to complete the work with reduced laborers -/
def actual_days : ℕ := 20

/-- The original number of laborers employed -/
def original_laborers : ℕ := 15

theorem contractor_problem :
  (original_laborers - absent_laborers) * initial_days = original_laborers * actual_days :=
sorry

end NUMINAMATH_CALUDE_contractor_problem_l1694_169444


namespace NUMINAMATH_CALUDE_johnny_work_days_l1694_169420

def daily_earnings : ℝ := 3 * 7 + 2 * 10 + 4 * 12

theorem johnny_work_days (x : ℝ) (h : x * daily_earnings = 445) : x = 5 := by
  sorry

end NUMINAMATH_CALUDE_johnny_work_days_l1694_169420


namespace NUMINAMATH_CALUDE_smallest_union_size_l1694_169476

theorem smallest_union_size (X Y : Finset ℕ) : 
  Finset.card X = 30 → 
  Finset.card Y = 25 → 
  Finset.card (X ∩ Y) ≥ 10 → 
  45 ≤ Finset.card (X ∪ Y) ∧ ∃ X' Y' : Finset ℕ, 
    Finset.card X' = 30 ∧ 
    Finset.card Y' = 25 ∧ 
    Finset.card (X' ∩ Y') ≥ 10 ∧ 
    Finset.card (X' ∪ Y') = 45 :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_union_size_l1694_169476


namespace NUMINAMATH_CALUDE_line_intersection_y_axis_l1694_169486

/-- The line passing through points (4, 2) and (6, 14) intersects the y-axis at (0, -22) -/
theorem line_intersection_y_axis :
  let p1 : ℝ × ℝ := (4, 2)
  let p2 : ℝ × ℝ := (6, 14)
  let m : ℝ := (p2.2 - p1.2) / (p2.1 - p1.1)
  let b : ℝ := p1.2 - m * p1.1
  let line (x : ℝ) : ℝ := m * x + b
  (0, line 0) = (0, -22) :=
by sorry

end NUMINAMATH_CALUDE_line_intersection_y_axis_l1694_169486


namespace NUMINAMATH_CALUDE_valid_configurations_count_l1694_169495

/-- Represents a configuration of lit and unlit bulbs -/
def BulbConfiguration := List Bool

/-- Checks if a configuration is valid (no adjacent lit bulbs) -/
def isValidConfiguration (config : BulbConfiguration) : Bool :=
  match config with
  | [] => true
  | [_] => true
  | true :: true :: _ => false
  | _ :: rest => isValidConfiguration rest

/-- Counts the number of lit bulbs in a configuration -/
def countLitBulbs (config : BulbConfiguration) : Nat :=
  config.filter id |>.length

/-- Generates all possible configurations for n bulbs -/
def allConfigurations (n : Nat) : List BulbConfiguration :=
  sorry

/-- Counts valid configurations with at least k lit bulbs out of n total bulbs -/
def countValidConfigurations (n k : Nat) : Nat :=
  (allConfigurations n).filter (fun config => 
    isValidConfiguration config && countLitBulbs config ≥ k
  ) |>.length

theorem valid_configurations_count : 
  countValidConfigurations 7 3 = 11 := by sorry

end NUMINAMATH_CALUDE_valid_configurations_count_l1694_169495


namespace NUMINAMATH_CALUDE_sum_of_digits_7_pow_2023_l1694_169499

theorem sum_of_digits_7_pow_2023 :
  ∃ (a b : ℕ), a < 10 ∧ b < 10 ∧ 7^2023 ≡ 10 * a + b [ZMOD 100] ∧ a + b = 16 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_digits_7_pow_2023_l1694_169499


namespace NUMINAMATH_CALUDE_gcd_204_85_l1694_169430

theorem gcd_204_85 : Nat.gcd 204 85 = 17 := by
  sorry

end NUMINAMATH_CALUDE_gcd_204_85_l1694_169430


namespace NUMINAMATH_CALUDE_ellipse_C_equation_min_OP_OQ_sum_l1694_169487

-- Define the ellipse C
def ellipse_C (a b : ℝ) (x y : ℝ) : Prop :=
  x^2 / a^2 + y^2 / b^2 = 1 ∧ a > b ∧ b > 0

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x^2 - y^2 = 1

-- Theorem for the equation of ellipse C
theorem ellipse_C_equation :
  ∀ a b : ℝ, (ellipse_C a b 1 (Real.sqrt 6 / 3)) →
  (∀ x y : ℝ, hyperbola x y ↔ hyperbola x y) →
  (∀ x y : ℝ, ellipse_C a b x y ↔ x^2 / 3 + y^2 = 1) :=
sorry

-- Define a line passing through two points on the ellipse
def line_through_ellipse (a b : ℝ) (x1 y1 x2 y2 : ℝ) : Prop :=
  ellipse_C a b x1 y1 ∧ ellipse_C a b x2 y2

-- Define points P and Q on the x-axis
def point_P (x : ℝ) : Prop := x ≠ 0
def point_Q (x : ℝ) : Prop := x ≠ 0

-- Theorem for the minimum value of |OP| + |OQ|
theorem min_OP_OQ_sum :
  ∀ a b x1 y1 x2 y2 p q : ℝ,
  line_through_ellipse a b x1 y1 x2 y2 →
  point_P p → point_Q q →
  |p| + |q| ≥ 2 * Real.sqrt 3 :=
sorry

end NUMINAMATH_CALUDE_ellipse_C_equation_min_OP_OQ_sum_l1694_169487


namespace NUMINAMATH_CALUDE_wednesday_to_tuesday_rainfall_ratio_l1694_169443

/-- Represents the rainfall data for a day -/
structure RainfallData where
  hours : ℝ
  rate : ℝ

/-- Calculates the total rainfall for a given day -/
def totalRainfall (data : RainfallData) : ℝ := data.hours * data.rate

theorem wednesday_to_tuesday_rainfall_ratio :
  let monday : RainfallData := { hours := 7, rate := 1 }
  let tuesday : RainfallData := { hours := 4, rate := 2 }
  let wednesday : RainfallData := { hours := 2, rate := ((23 : ℝ) - totalRainfall monday - totalRainfall tuesday) / 2 }
  wednesday.rate / tuesday.rate = 2 := by sorry

end NUMINAMATH_CALUDE_wednesday_to_tuesday_rainfall_ratio_l1694_169443


namespace NUMINAMATH_CALUDE_sequence_product_l1694_169497

/-- Given an arithmetic sequence and a geometric sequence with specific properties,
    prove that b₂(a₂-a₁) = -8 --/
theorem sequence_product (a₁ a₂ b₁ b₂ b₃ : ℝ) : 
  ((-9 : ℝ) < a₁ ∧ a₁ < a₂ ∧ a₂ < (-1 : ℝ)) →  -- arithmetic sequence condition
  (∃ d : ℝ, a₁ = -9 + d ∧ a₂ = a₁ + d ∧ -1 = a₂ + d) →  -- arithmetic sequence definition
  ((-9 : ℝ) < b₁ ∧ b₁ < b₂ ∧ b₂ < b₃ ∧ b₃ < (-1 : ℝ)) →  -- geometric sequence condition
  (∃ q : ℝ, b₁ = -9 * q ∧ b₂ = b₁ * q ∧ b₃ = b₂ * q ∧ -1 = b₃ * q) →  -- geometric sequence definition
  b₂ * (a₂ - a₁) = -8 := by
sorry

end NUMINAMATH_CALUDE_sequence_product_l1694_169497


namespace NUMINAMATH_CALUDE_watermelon_seeds_count_l1694_169441

/-- Represents a watermelon with its properties -/
structure Watermelon :=
  (slices : ℕ)
  (black_seeds_per_slice : ℕ)
  (white_seeds_per_slice : ℕ)

/-- Calculates the total number of seeds in a watermelon -/
def total_seeds (w : Watermelon) : ℕ :=
  w.slices * (w.black_seeds_per_slice + w.white_seeds_per_slice)

/-- Theorem stating that a watermelon with 40 slices, 20 black seeds and 20 white seeds per slice has 1600 total seeds -/
theorem watermelon_seeds_count :
  ∀ (w : Watermelon),
  w.slices = 40 →
  w.black_seeds_per_slice = 20 →
  w.white_seeds_per_slice = 20 →
  total_seeds w = 1600 :=
by
  sorry


end NUMINAMATH_CALUDE_watermelon_seeds_count_l1694_169441


namespace NUMINAMATH_CALUDE_tim_grew_44_cantaloupes_l1694_169481

/-- The number of cantaloupes Fred grew -/
def fred_cantaloupes : ℕ := 38

/-- The total number of cantaloupes Fred and Tim grew together -/
def total_cantaloupes : ℕ := 82

/-- The number of cantaloupes Tim grew -/
def tim_cantaloupes : ℕ := total_cantaloupes - fred_cantaloupes

theorem tim_grew_44_cantaloupes : tim_cantaloupes = 44 := by
  sorry

end NUMINAMATH_CALUDE_tim_grew_44_cantaloupes_l1694_169481


namespace NUMINAMATH_CALUDE_cheapest_combination_is_12_apples_3_oranges_l1694_169401

/-- Represents the price of a fruit deal -/
structure FruitDeal where
  quantity : Nat
  price : Rat

/-- Represents the fruit options available -/
structure FruitOptions where
  apple_deals : List FruitDeal
  orange_deals : List FruitDeal

/-- Represents a combination of apples and oranges -/
structure FruitCombination where
  apples : Nat
  oranges : Nat

def total_fruits (combo : FruitCombination) : Nat :=
  combo.apples + combo.oranges

def is_valid_combination (combo : FruitCombination) : Prop :=
  total_fruits combo = 15 ∧
  (combo.apples % 2 = 0 ∨ combo.apples % 3 = 0) ∧
  (combo.oranges % 2 = 0 ∨ combo.oranges % 3 = 0)

def cost_of_combination (options : FruitOptions) (combo : FruitCombination) : Rat :=
  sorry

def cheapest_combination (options : FruitOptions) : FruitCombination :=
  sorry

theorem cheapest_combination_is_12_apples_3_oranges
  (options : FruitOptions)
  (h_apple_deals : options.apple_deals = [
    ⟨2, 48/100⟩, ⟨6, 126/100⟩, ⟨12, 224/100⟩
  ])
  (h_orange_deals : options.orange_deals = [
    ⟨2, 60/100⟩, ⟨6, 164/100⟩, ⟨12, 300/100⟩
  ]) :
  cheapest_combination options = ⟨12, 3⟩ ∧
  cost_of_combination options (cheapest_combination options) = 314/100 :=
sorry

end NUMINAMATH_CALUDE_cheapest_combination_is_12_apples_3_oranges_l1694_169401


namespace NUMINAMATH_CALUDE_min_value_3x_plus_4y_l1694_169448

theorem min_value_3x_plus_4y (x y : ℝ) (h_pos_x : 0 < x) (h_pos_y : 0 < y) (h_eq : x + 3 * y = x * y) :
  25 ≤ 3 * x + 4 * y := by
  sorry

end NUMINAMATH_CALUDE_min_value_3x_plus_4y_l1694_169448


namespace NUMINAMATH_CALUDE_total_value_is_305_l1694_169477

/-- The value of a gold coin in dollars -/
def gold_coin_value : ℕ := 50

/-- The value of a silver coin in dollars -/
def silver_coin_value : ℕ := 25

/-- The number of gold coins -/
def num_gold_coins : ℕ := 3

/-- The number of silver coins -/
def num_silver_coins : ℕ := 5

/-- The amount of cash in dollars -/
def cash : ℕ := 30

/-- The total value of all coins and cash -/
def total_value : ℕ := num_gold_coins * gold_coin_value + num_silver_coins * silver_coin_value + cash

theorem total_value_is_305 : total_value = 305 := by
  sorry

end NUMINAMATH_CALUDE_total_value_is_305_l1694_169477


namespace NUMINAMATH_CALUDE_diorama_building_time_l1694_169402

/-- Represents the time spent on the diorama project -/
structure DioramaTime where
  planning : ℕ  -- Planning time in minutes
  building : ℕ  -- Building time in minutes

/-- Defines the conditions of the diorama project -/
def validDioramaTime (t : DioramaTime) : Prop :=
  t.building = 3 * t.planning - 5 ∧
  t.building + t.planning = 67

/-- Theorem stating that the building time is 49 minutes -/
theorem diorama_building_time :
  ∀ t : DioramaTime, validDioramaTime t → t.building = 49 :=
by
  sorry


end NUMINAMATH_CALUDE_diorama_building_time_l1694_169402


namespace NUMINAMATH_CALUDE_apple_value_in_cake_slices_l1694_169438

/-- Represents the value of one apple in terms of cake slices -/
def apple_value : ℚ := 15 / 4

/-- Represents the number of apples that can be traded for juice bottles -/
def apples_per_juice_trade : ℕ := 4

/-- Represents the number of juice bottles received in trade for apples -/
def juice_bottles_per_apple_trade : ℕ := 3

/-- Represents the number of cake slices that can be traded for one juice bottle -/
def cake_slices_per_juice_bottle : ℕ := 5

theorem apple_value_in_cake_slices :
  apple_value = (juice_bottles_per_apple_trade * cake_slices_per_juice_bottle : ℚ) / apples_per_juice_trade :=
sorry

#eval apple_value -- Should output 3.75

end NUMINAMATH_CALUDE_apple_value_in_cake_slices_l1694_169438


namespace NUMINAMATH_CALUDE_find_number_l1694_169479

theorem find_number : ∃! x : ℕ+, 
  (172 / x.val : ℚ) = 172 / 4 - 28 ∧ 
  172 % x.val = 7 ∧ 
  x = 11 := by
  sorry

end NUMINAMATH_CALUDE_find_number_l1694_169479


namespace NUMINAMATH_CALUDE_largest_angle_in_triangle_l1694_169445

theorem largest_angle_in_triangle : ∀ (a b c : ℝ),
  a + b + c = 180 →  -- Sum of angles in a triangle is 180°
  a + b = 120 →      -- Sum of two angles is 4/3 of right angle (90° * 4/3 = 120°)
  b = a + 36 →       -- One angle is 36° larger than the other
  max a (max b c) = 78 := by
sorry

end NUMINAMATH_CALUDE_largest_angle_in_triangle_l1694_169445


namespace NUMINAMATH_CALUDE_cruise_ship_problem_l1694_169437

/-- Cruise ship problem -/
theorem cruise_ship_problem 
  (distance : ℝ) 
  (x : ℝ) 
  (k : ℝ) 
  (h1 : distance = 5)
  (h2 : 20 ≤ x ∧ x ≤ 50)
  (h3 : 1/15 ≤ k ∧ k ≤ 1/5)
  (h4 : x/40 - k = 5/8) :
  (∃ (x_range : Set ℝ), x_range = {x | 20 ≤ x ∧ x ≤ 40} ∧ 
    ∀ y ∈ x_range, y/40 - k + 1/y ≤ 9/10) ∧
  (∀ y : ℝ, 20 ≤ y ∧ y ≤ 50 →
    (1/15 ≤ k ∧ k < 1/10 → 
      5/y * (y/40 - k + 1/y) ≥ (1 - 10*k^2) / 8) ∧
    (1/10 ≤ k ∧ k ≤ 1/5 → 
      5/y * (y/40 - k + 1/y) ≥ (11 - 20*k) / 80)) :=
sorry

end NUMINAMATH_CALUDE_cruise_ship_problem_l1694_169437


namespace NUMINAMATH_CALUDE_aquarium_height_is_three_l1694_169469

/-- Represents an aquarium with given dimensions and water filling process --/
structure Aquarium where
  length : ℝ
  width : ℝ
  height : ℝ
  initialFillFraction : ℝ
  spillFraction : ℝ
  finalMultiplier : ℝ

/-- Calculates the final volume of water in the aquarium after the described process --/
def finalVolume (a : Aquarium) : ℝ :=
  a.length * a.width * a.height * a.initialFillFraction * (1 - a.spillFraction) * a.finalMultiplier

/-- Theorem stating that an aquarium with the given properties has a height of 3 feet --/
theorem aquarium_height_is_three :
  ∀ (a : Aquarium),
    a.length = 4 →
    a.width = 6 →
    a.initialFillFraction = 1/2 →
    a.spillFraction = 1/2 →
    a.finalMultiplier = 3 →
    finalVolume a = 54 →
    a.height = 3 := by sorry

end NUMINAMATH_CALUDE_aquarium_height_is_three_l1694_169469


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l1694_169421

def f (a : ℝ) (x : ℝ) : ℝ := (x - a)^2

theorem sufficient_not_necessary_condition :
  (∀ a : ℝ, a = 1 → (∀ x y : ℝ, 1 < x → x < y → f a x < f a y)) ∧
  (∃ a : ℝ, a ≠ 1 ∧ (∀ x y : ℝ, 1 < x → x < y → f a x < f a y)) := by
  sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l1694_169421


namespace NUMINAMATH_CALUDE_tom_current_blue_tickets_l1694_169461

/-- Represents the number of tickets Tom has -/
structure TomTickets where
  yellow : ℕ
  red : ℕ
  blue : ℕ

/-- Represents the conversion rates between ticket types -/
structure TicketConversion where
  yellow_to_red : ℕ
  red_to_blue : ℕ

/-- Theorem: Given the conditions, Tom currently has 7 blue tickets -/
theorem tom_current_blue_tickets 
  (total_yellow_needed : ℕ)
  (conversion : TicketConversion)
  (tom_tickets : TomTickets)
  (additional_blue_needed : ℕ)
  (h1 : total_yellow_needed = 10)
  (h2 : conversion.yellow_to_red = 10)
  (h3 : conversion.red_to_blue = 10)
  (h4 : tom_tickets.yellow = 8)
  (h5 : tom_tickets.red = 3)
  (h6 : additional_blue_needed = 163) :
  tom_tickets.blue = 7 := by
  sorry

#check tom_current_blue_tickets

end NUMINAMATH_CALUDE_tom_current_blue_tickets_l1694_169461


namespace NUMINAMATH_CALUDE_min_balls_for_fifteen_colors_l1694_169496

/-- Represents the number of balls of each color in the box -/
structure BallCounts where
  red : Nat
  green : Nat
  yellow : Nat
  blue : Nat
  white : Nat
  black : Nat
  purple : Nat

/-- The minimum number of balls needed to guarantee at least n balls of a single color -/
def minBallsForColor (counts : BallCounts) (n : Nat) : Nat :=
  sorry

theorem min_balls_for_fifteen_colors (counts : BallCounts) 
  (h_red : counts.red = 35)
  (h_green : counts.green = 18)
  (h_yellow : counts.yellow = 15)
  (h_blue : counts.blue = 17)
  (h_white : counts.white = 12)
  (h_black : counts.black = 12)
  (h_purple : counts.purple = 8) :
  minBallsForColor counts 15 = 89 := by
  sorry

end NUMINAMATH_CALUDE_min_balls_for_fifteen_colors_l1694_169496


namespace NUMINAMATH_CALUDE_sqrt_a_minus_two_real_l1694_169400

theorem sqrt_a_minus_two_real (a : ℝ) : (∃ x : ℝ, x^2 = a - 2) → a ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_a_minus_two_real_l1694_169400


namespace NUMINAMATH_CALUDE_max_log_sum_max_log_sum_attained_l1694_169467

theorem max_log_sum (x y : ℝ) (hx : x > 0) (hy : y > 0) (h_eq : 2 * x + 3 * y = 6) :
  (Real.log x / Real.log (3/2) + Real.log y / Real.log (3/2)) ≤ 1 :=
by sorry

theorem max_log_sum_attained (x y : ℝ) (hx : x > 0) (hy : y > 0) (h_eq : 2 * x + 3 * y = 6) :
  (Real.log x / Real.log (3/2) + Real.log y / Real.log (3/2)) = 1 ↔ x = 3/2 ∧ y = 1 :=
by sorry

end NUMINAMATH_CALUDE_max_log_sum_max_log_sum_attained_l1694_169467


namespace NUMINAMATH_CALUDE_other_number_problem_l1694_169415

theorem other_number_problem (a b : ℕ) : 
  a + b = 96 → 
  (a = b + 12 ∨ b = a + 12) → 
  (a = 42 ∨ b = 42) → 
  (a = 54 ∨ b = 54) :=
by sorry

end NUMINAMATH_CALUDE_other_number_problem_l1694_169415


namespace NUMINAMATH_CALUDE_integer_partition_impossibility_l1694_169411

theorem integer_partition_impossibility : 
  ¬ (∃ (A B C : Set Int), 
    (∀ (n : Int), n ∈ A ∨ n ∈ B ∨ n ∈ C) ∧ 
    (A ∪ B ∪ C = Set.univ) ∧
    (A ∩ B = ∅) ∧ (B ∩ C = ∅) ∧ (C ∩ A = ∅) ∧
    (∀ (n : Int), 
      ((n ∈ A ∧ (n - 50) ∈ B ∧ (n + 1987) ∈ C) ∨
       (n ∈ A ∧ (n - 50) ∈ C ∧ (n + 1987) ∈ B) ∨
       (n ∈ B ∧ (n - 50) ∈ A ∧ (n + 1987) ∈ C) ∨
       (n ∈ B ∧ (n - 50) ∈ C ∧ (n + 1987) ∈ A) ∨
       (n ∈ C ∧ (n - 50) ∈ A ∧ (n + 1987) ∈ B) ∨
       (n ∈ C ∧ (n - 50) ∈ B ∧ (n + 1987) ∈ A)))) :=
by sorry

end NUMINAMATH_CALUDE_integer_partition_impossibility_l1694_169411


namespace NUMINAMATH_CALUDE_polynomial_remainder_l1694_169422

/-- Given a polynomial Q with Q(25) = 50 and Q(50) = 25, 
    the remainder when Q is divided by (x - 25)(x - 50) is -x + 75 -/
theorem polynomial_remainder (Q : ℝ → ℝ) (h1 : Q 25 = 50) (h2 : Q 50 = 25) :
  ∃ (R : ℝ → ℝ), ∀ x, Q x = (x - 25) * (x - 50) * R x + (-x + 75) :=
sorry

end NUMINAMATH_CALUDE_polynomial_remainder_l1694_169422


namespace NUMINAMATH_CALUDE_complex_number_problems_l1694_169447

open Complex

theorem complex_number_problems (z₁ z₂ z : ℂ) (b : ℝ) :
  z₁ = 1 - I ∧ z₂ = 4 + 6 * I ∧ z = 1 + b * I ∧ (z + z₁).im = 0 →
  z₂ / z₁ = -1 + 5 * I ∧ abs z = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_number_problems_l1694_169447


namespace NUMINAMATH_CALUDE_two_balls_same_box_probability_l1694_169404

theorem two_balls_same_box_probability :
  let num_balls : ℕ := 3
  let num_boxes : ℕ := 5
  let total_outcomes : ℕ := num_boxes ^ num_balls
  let favorable_outcomes : ℕ := (num_balls.choose 2) * num_boxes * (num_boxes - 1)
  favorable_outcomes / total_outcomes = 12 / 25 := by
sorry

end NUMINAMATH_CALUDE_two_balls_same_box_probability_l1694_169404


namespace NUMINAMATH_CALUDE_purple_balls_count_l1694_169425

theorem purple_balls_count (total_balls : ℕ) (white_balls : ℕ) (green_balls : ℕ) (yellow_balls : ℕ) (red_balls : ℕ) 
  (h1 : total_balls = 60)
  (h2 : white_balls = 22)
  (h3 : green_balls = 18)
  (h4 : yellow_balls = 2)
  (h5 : red_balls = 15)
  (h6 : (white_balls + green_balls + yellow_balls : ℚ) / total_balls = 7/10) :
  ∃ (purple_balls : ℕ), purple_balls = 3 ∧ total_balls = white_balls + green_balls + yellow_balls + red_balls + purple_balls :=
by
  sorry


end NUMINAMATH_CALUDE_purple_balls_count_l1694_169425


namespace NUMINAMATH_CALUDE_new_student_height_l1694_169470

def original_heights : List ℝ := [145, 139, 155, 160, 143]

def average_increase : ℝ := 1.2

theorem new_student_height :
  let original_sum := original_heights.sum
  let original_count := original_heights.length
  let original_average := original_sum / original_count
  let new_average := original_average + average_increase
  let new_count := original_count + 1
  let new_sum := new_average * new_count
  new_sum - original_sum = 155.6 := by sorry

end NUMINAMATH_CALUDE_new_student_height_l1694_169470


namespace NUMINAMATH_CALUDE_value_of_120abc_l1694_169482

theorem value_of_120abc (a b c d : ℝ) 
  (h1 : 10 * a = 20) 
  (h2 : 6 * b = 20) 
  (h3 : c^2 + d^2 = 50) : 
  120 * a * b * c = 800 * Real.sqrt (50 - d^2) := by
  sorry

end NUMINAMATH_CALUDE_value_of_120abc_l1694_169482


namespace NUMINAMATH_CALUDE_stratified_sampling_first_grade_l1694_169463

theorem stratified_sampling_first_grade (total_students : ℕ) (sample_size : ℕ) 
  (grade_1_ratio grade_2_ratio grade_3_ratio : ℕ) :
  total_students = 2400 →
  sample_size = 120 →
  grade_1_ratio = 5 →
  grade_2_ratio = 4 →
  grade_3_ratio = 3 →
  (grade_1_ratio * sample_size) / (grade_1_ratio + grade_2_ratio + grade_3_ratio) = 50 := by
  sorry

#check stratified_sampling_first_grade

end NUMINAMATH_CALUDE_stratified_sampling_first_grade_l1694_169463


namespace NUMINAMATH_CALUDE_lillian_sugar_bags_lillian_sugar_bags_proof_l1694_169427

/-- Lillian's cupcake sugar problem -/
theorem lillian_sugar_bags : ℕ :=
  let sugar_at_home : ℕ := 3
  let sugar_per_bag : ℕ := 6
  let sugar_per_dozen_batter : ℕ := 1
  let sugar_per_dozen_frosting : ℕ := 2
  let dozens_of_cupcakes : ℕ := 5

  let total_sugar_needed := dozens_of_cupcakes * (sugar_per_dozen_batter + sugar_per_dozen_frosting)
  let sugar_to_buy := total_sugar_needed - sugar_at_home
  let bags_to_buy := sugar_to_buy / sugar_per_bag

  2

theorem lillian_sugar_bags_proof : lillian_sugar_bags = 2 := by
  sorry

end NUMINAMATH_CALUDE_lillian_sugar_bags_lillian_sugar_bags_proof_l1694_169427


namespace NUMINAMATH_CALUDE_repeating_decimal_division_l1694_169435

/-- Represents a repeating decimal with a single repeating digit -/
def RepeatingDecimal (whole : ℚ) (repeating : ℕ) : ℚ :=
  whole + (repeating : ℚ) / 999

theorem repeating_decimal_division (h : RepeatingDecimal 0 4 / RepeatingDecimal 1 6 = 4 / 15) : 
  RepeatingDecimal 0 4 / RepeatingDecimal 1 6 = 4 / 15 := by
  sorry

end NUMINAMATH_CALUDE_repeating_decimal_division_l1694_169435


namespace NUMINAMATH_CALUDE_simple_interest_difference_l1694_169471

/-- Simple interest calculation and comparison with principal -/
theorem simple_interest_difference (principal : ℕ) (rate : ℚ) (time : ℕ) :
  principal = 2500 →
  rate = 4 / 100 →
  time = 5 →
  principal - (principal * rate * time) = 2000 := by
  sorry

end NUMINAMATH_CALUDE_simple_interest_difference_l1694_169471


namespace NUMINAMATH_CALUDE_morning_afternoon_difference_l1694_169468

/-- The number of campers who went rowing in the morning -/
def morning_campers : ℕ := 44

/-- The number of campers who went rowing in the afternoon -/
def afternoon_campers : ℕ := 39

/-- The number of campers who went rowing in the evening -/
def evening_campers : ℕ := 31

theorem morning_afternoon_difference :
  morning_campers - afternoon_campers = 5 := by
  sorry

end NUMINAMATH_CALUDE_morning_afternoon_difference_l1694_169468


namespace NUMINAMATH_CALUDE_max_value_theorem_l1694_169492

theorem max_value_theorem (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0)
  (h1 : a^2 * (b + c - a) = b^2 * (a + c - b))
  (h2 : b^2 * (a + c - b) = c^2 * (b + a - c)) :
  ∀ x : ℝ, (2*b + 3*c) / a ≤ 5 :=
by sorry

end NUMINAMATH_CALUDE_max_value_theorem_l1694_169492


namespace NUMINAMATH_CALUDE_pure_imaginary_a_value_l1694_169413

/-- The imaginary unit -/
def i : ℂ := Complex.I

/-- The complex number z -/
def z (a : ℝ) : ℂ := (a + i) * (3 + 2*i)

/-- A complex number is pure imaginary if its real part is zero -/
def is_pure_imaginary (c : ℂ) : Prop := c.re = 0 ∧ c.im ≠ 0

theorem pure_imaginary_a_value :
  ∃ (a : ℝ), is_pure_imaginary (z a) ∧ a = 2/3 :=
sorry

end NUMINAMATH_CALUDE_pure_imaginary_a_value_l1694_169413


namespace NUMINAMATH_CALUDE_range_of_m_l1694_169407

theorem range_of_m (m : ℝ) : 
  (∀ x : ℝ, |x + 1| + |x - 3| ≥ |m - 1|) → 
  -3 ≤ m ∧ m ≤ 5 := by
sorry

end NUMINAMATH_CALUDE_range_of_m_l1694_169407


namespace NUMINAMATH_CALUDE_increase_by_percentage_l1694_169494

theorem increase_by_percentage (initial : ℝ) (percentage : ℝ) (final : ℝ) :
  initial = 784.3 →
  percentage = 28.5 →
  final = initial * (1 + percentage / 100) →
  final = 1007.8255 := by
  sorry

end NUMINAMATH_CALUDE_increase_by_percentage_l1694_169494


namespace NUMINAMATH_CALUDE_beetle_projection_theorem_l1694_169483

/-- Represents a line in 2D space --/
structure Line where
  slope : ℝ
  intercept : ℝ

/-- Represents a beetle moving on a line --/
structure Beetle where
  line : Line
  speed : ℝ
  initialPosition : ℝ

/-- Theorem: If two beetles move on intersecting lines with constant speeds,
    and their projections on the OX axis never coincide,
    then their projections on the OY axis must either coincide or have coincided in the past --/
theorem beetle_projection_theorem (L1 L2 : Line) (b1 b2 : Beetle)
    (h_intersect : L1 ≠ L2)
    (h_b1_on_L1 : b1.line = L1)
    (h_b2_on_L2 : b2.line = L2)
    (h_constant_speed : b1.speed ≠ 0 ∧ b2.speed ≠ 0)
    (h_x_proj_never_coincide : ∀ t : ℝ, 
      b1.initialPosition + b1.speed * t ≠ b2.initialPosition + b2.speed * t) :
    ∃ t : ℝ, 
      L1.slope * (b1.initialPosition + b1.speed * t) + L1.intercept = 
      L2.slope * (b2.initialPosition + b2.speed * t) + L2.intercept :=
sorry

end NUMINAMATH_CALUDE_beetle_projection_theorem_l1694_169483


namespace NUMINAMATH_CALUDE_water_intake_calculation_l1694_169403

theorem water_intake_calculation (morning_intake : Real) 
  (h1 : morning_intake = 1.5)
  (h2 : afternoon_intake = 3 * morning_intake)
  (h3 : evening_intake = 0.5 * afternoon_intake) :
  morning_intake + afternoon_intake + evening_intake = 8.25 := by
  sorry

end NUMINAMATH_CALUDE_water_intake_calculation_l1694_169403


namespace NUMINAMATH_CALUDE_coefficient_of_x3_in_expansion_l1694_169410

-- Define the binomial coefficient
def binomial (n k : ℕ) : ℕ := sorry

-- Define the function to calculate the coefficient of x^3
def coefficientOfX3 (a b : ℝ) (n : ℕ) : ℝ :=
  (-b)^1 * binomial n 1 * a^3

-- Theorem statement
theorem coefficient_of_x3_in_expansion :
  coefficientOfX3 1 3 7 = -21 := by sorry

end NUMINAMATH_CALUDE_coefficient_of_x3_in_expansion_l1694_169410


namespace NUMINAMATH_CALUDE_wrong_mark_calculation_l1694_169455

/-- Given a class of students with an incorrect average and one student's mark wrongly noted,
    calculate the wrongly noted mark. -/
theorem wrong_mark_calculation 
  (n : ℕ) -- number of students
  (initial_avg : ℚ) -- initial (incorrect) average
  (correct_mark : ℚ) -- correct mark for the student
  (correct_avg : ℚ) -- correct average after fixing the mark
  (h1 : n = 25) -- there are 25 students
  (h2 : initial_avg = 100) -- initial average is 100
  (h3 : correct_mark = 10) -- the correct mark is 10
  (h4 : correct_avg = 98) -- the correct average is 98
  : 
  -- The wrongly noted mark
  (n : ℚ) * initial_avg - ((n : ℚ) * correct_avg - correct_mark) = 60 :=
by sorry

end NUMINAMATH_CALUDE_wrong_mark_calculation_l1694_169455


namespace NUMINAMATH_CALUDE_first_interest_rate_is_five_percent_l1694_169472

/-- Proves that the first interest rate is 5% given the problem conditions -/
theorem first_interest_rate_is_five_percent 
  (total_amount : ℝ)
  (first_amount : ℝ)
  (second_rate : ℝ)
  (total_income : ℝ)
  (h1 : total_amount = 2600)
  (h2 : first_amount = 1600)
  (h3 : second_rate = 6)
  (h4 : total_income = 140)
  (h5 : ∃ r, (r * first_amount / 100) + (second_rate * (total_amount - first_amount) / 100) = total_income) :
  ∃ r, r = 5 ∧ (r * first_amount / 100) + (second_rate * (total_amount - first_amount) / 100) = total_income :=
by sorry

end NUMINAMATH_CALUDE_first_interest_rate_is_five_percent_l1694_169472


namespace NUMINAMATH_CALUDE_tangent_line_at_zero_l1694_169478

noncomputable def f (x : ℝ) : ℝ := Real.cos x - x / 2

theorem tangent_line_at_zero (x y : ℝ) :
  (f 0 = 1) →
  (∀ x, HasDerivAt f (-Real.sin x - 1/2) x) →
  (y = -1/2 * x + 1) →
  (x + 2*y = 2) :=
sorry

end NUMINAMATH_CALUDE_tangent_line_at_zero_l1694_169478


namespace NUMINAMATH_CALUDE_nPointedStar_interiorAngleSum_l1694_169424

/-- Represents an n-pointed star formed from an n-sided convex polygon -/
structure NPointedStar where
  n : ℕ
  h_n : n ≥ 6

/-- The sum of interior angles at the vertices of an n-pointed star -/
def interiorAngleSum (star : NPointedStar) : ℝ :=
  180 * (star.n - 2)

/-- Theorem: The sum of interior angles at the vertices of an n-pointed star
    formed by extending every third side of an n-sided convex polygon (n ≥ 6)
    is equal to 180°(n-2) -/
theorem nPointedStar_interiorAngleSum (star : NPointedStar) :
  interiorAngleSum star = 180 * (star.n - 2) := by
  sorry

end NUMINAMATH_CALUDE_nPointedStar_interiorAngleSum_l1694_169424


namespace NUMINAMATH_CALUDE_double_iced_cubes_count_l1694_169474

/-- Represents a cube cake -/
structure CubeCake where
  size : Nat
  top_iced : Bool
  front_iced : Bool

/-- Counts the number of 1x1x1 subcubes with icing on exactly two sides -/
def count_double_iced_cubes (cake : CubeCake) : Nat :=
  if cake.top_iced && cake.front_iced then
    cake.size - 1
  else
    0

/-- Theorem: A 3x3x3 cake with top and front face iced has 2 subcubes with icing on two sides -/
theorem double_iced_cubes_count :
  let cake : CubeCake := { size := 3, top_iced := true, front_iced := true }
  count_double_iced_cubes cake = 2 := by
  sorry

end NUMINAMATH_CALUDE_double_iced_cubes_count_l1694_169474


namespace NUMINAMATH_CALUDE_number_of_observations_l1694_169451

theorem number_of_observations
  (initial_mean : ℝ)
  (incorrect_value : ℝ)
  (correct_value : ℝ)
  (corrected_mean : ℝ)
  (h1 : initial_mean = 41)
  (h2 : incorrect_value = 23)
  (h3 : correct_value = 48)
  (h4 : corrected_mean = 41.5) :
  ∃ n : ℕ, n * initial_mean - incorrect_value + correct_value = n * corrected_mean ∧ n = 50 := by
  sorry

end NUMINAMATH_CALUDE_number_of_observations_l1694_169451


namespace NUMINAMATH_CALUDE_find_g_of_x_l1694_169473

theorem find_g_of_x (x : ℝ) : 
  let g := fun (x : ℝ) ↦ -4*x^4 + 5*x^3 - 2*x^2 + 7*x + 2
  4*x^4 + 2*x^2 - 7*x + g x = 5*x^3 - 4*x + 2 := by sorry

end NUMINAMATH_CALUDE_find_g_of_x_l1694_169473


namespace NUMINAMATH_CALUDE_classes_per_semester_l1694_169439

/-- Given the following conditions:
  1. Maddy is in college for 8 semesters.
  2. She needs 120 credits to graduate.
  3. Each class is 3 credits.
  Prove that Maddy needs to take 5 classes per semester. -/
theorem classes_per_semester :
  let total_semesters : ℕ := 8
  let total_credits : ℕ := 120
  let credits_per_class : ℕ := 3
  let classes_per_semester : ℕ := total_credits / (credits_per_class * total_semesters)
  classes_per_semester = 5 := by sorry

end NUMINAMATH_CALUDE_classes_per_semester_l1694_169439


namespace NUMINAMATH_CALUDE_max_product_with_sum_2016_l1694_169466

theorem max_product_with_sum_2016 :
  ∀ x y : ℤ, x + y = 2016 → x * y ≤ 1016064 := by
  sorry

end NUMINAMATH_CALUDE_max_product_with_sum_2016_l1694_169466


namespace NUMINAMATH_CALUDE_unique_parallel_line_l1694_169429

-- Define the concept of a plane
variable (Plane : Type)

-- Define the concept of a point
variable (Point : Type)

-- Define the concept of a line
variable (Line : Type)

-- Define the relation of a point lying on a plane
variable (lies_on : Point → Plane → Prop)

-- Define the relation of two planes intersecting
variable (intersect : Plane → Plane → Prop)

-- Define the relation of a line being parallel to a plane
variable (parallel_to_plane : Line → Plane → Prop)

-- Define the relation of a line passing through a point
variable (passes_through : Line → Point → Prop)

-- Theorem statement
theorem unique_parallel_line 
  (α β : Plane) (A : Point) 
  (h_intersect : intersect α β)
  (h_not_on_α : ¬ lies_on A α)
  (h_not_on_β : ¬ lies_on A β) :
  ∃! l : Line, passes_through l A ∧ parallel_to_plane l α ∧ parallel_to_plane l β :=
sorry

end NUMINAMATH_CALUDE_unique_parallel_line_l1694_169429


namespace NUMINAMATH_CALUDE_cubic_expansion_coefficient_l1694_169489

theorem cubic_expansion_coefficient (a a₁ a₂ a₃ : ℝ) :
  (∀ x : ℝ, x^3 = a + a₁*(x-2) + a₂*(x-2)^2 + a₃*(x-2)^3) →
  a₂ = 6 := by
sorry

end NUMINAMATH_CALUDE_cubic_expansion_coefficient_l1694_169489


namespace NUMINAMATH_CALUDE_polynomial_divisibility_l1694_169456

theorem polynomial_divisibility (C D : ℚ) : 
  (∀ x : ℂ, x^2 - x + 1 = 0 → x^103 + C*x^2 + D = 0) → 
  C = -1 ∧ D = -1 := by
sorry

end NUMINAMATH_CALUDE_polynomial_divisibility_l1694_169456


namespace NUMINAMATH_CALUDE_five_Z_three_equals_nineteen_l1694_169464

def Z (x y : ℝ) : ℝ := x^2 - x*y + y^2

theorem five_Z_three_equals_nineteen : Z 5 3 = 19 := by
  sorry

end NUMINAMATH_CALUDE_five_Z_three_equals_nineteen_l1694_169464


namespace NUMINAMATH_CALUDE_ellipse_properties_l1694_169485

-- Define the ellipse C
def ellipse_C (x y a : ℝ) : Prop :=
  x^2 / a^2 + y^2 / (7 - a^2) = 1 ∧ a > 0

-- Define the eccentricity
def eccentricity (a : ℝ) : ℝ := 2

-- Define the standard form of the ellipse
def standard_ellipse (x y : ℝ) : Prop :=
  x^2 / 4 + y^2 / 3 = 1

-- Define a line passing through (4,0)
def line_through_R (x y k : ℝ) : Prop :=
  y = k * (x - 4)

-- Define a point on the ellipse
def point_on_ellipse (x y : ℝ) : Prop :=
  standard_ellipse x y

-- Define a perpendicular line to x-axis
def perpendicular_to_x (x y x₁ : ℝ) : Prop :=
  x = x₁

-- Define the right focus of the ellipse
def right_focus : ℝ × ℝ := (1, 0)

-- State the theorem
theorem ellipse_properties (a x y x₁ y₁ x₂ y₂ k : ℝ) :
  ellipse_C x y a →
  eccentricity a = 2 →
  line_through_R x y k →
  point_on_ellipse x₁ y₁ →
  point_on_ellipse x₂ y₂ →
  perpendicular_to_x x y x₁ →
  point_on_ellipse x₁ (-y₁) →
  (∀ x y, ellipse_C x y a ↔ standard_ellipse x y) ∧
  (∃ t, t * (x₁, -y₁) + (1 - t) * right_focus = (x₂, y₂)) :=
sorry

end NUMINAMATH_CALUDE_ellipse_properties_l1694_169485


namespace NUMINAMATH_CALUDE_special_sequence_2023_l1694_169454

/-- A sequence satisfying the given conditions -/
def special_sequence (a : ℕ → ℕ) : Prop :=
  a 1 = 3 ∧ ∀ m n : ℕ, m > 0 → n > 0 → a (m + n) = a m + a n

/-- The 2023rd term of the special sequence equals 6069 -/
theorem special_sequence_2023 (a : ℕ → ℕ) (h : special_sequence a) : a 2023 = 6069 := by
  sorry

end NUMINAMATH_CALUDE_special_sequence_2023_l1694_169454


namespace NUMINAMATH_CALUDE_ant_walk_probability_l1694_169488

/-- The probability of returning to the starting vertex after n moves on a square,
    given the probability of moving clockwise and counter-clockwise. -/
def return_probability (n : ℕ) (p_cw : ℚ) (p_ccw : ℚ) : ℚ :=
  sorry

/-- The number of ways to choose k items from n items. -/
def binomial_coefficient (n k : ℕ) : ℕ :=
  sorry

theorem ant_walk_probability :
  let n : ℕ := 6
  let p_cw : ℚ := 2/3
  let p_ccw : ℚ := 1/3
  return_probability n p_cw p_ccw = 160/729 :=
sorry

end NUMINAMATH_CALUDE_ant_walk_probability_l1694_169488
