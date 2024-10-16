import Mathlib

namespace NUMINAMATH_CALUDE_bob_tv_width_is_90_l846_84686

/-- The width of Bob's TV -/
def bob_tv_width : ℝ := 90

/-- The height of Bob's TV -/
def bob_tv_height : ℝ := 60

/-- The width of Bill's TV -/
def bill_tv_width : ℝ := 100

/-- The height of Bill's TV -/
def bill_tv_height : ℝ := 48

/-- Weight of TV per square inch in ounces -/
def tv_weight_per_sq_inch : ℝ := 4

/-- Ounces per pound -/
def oz_per_pound : ℝ := 16

/-- Weight difference between Bob's and Bill's TVs in pounds -/
def weight_difference : ℝ := 150

theorem bob_tv_width_is_90 :
  bob_tv_width = 90 :=
by
  sorry

#check bob_tv_width_is_90

end NUMINAMATH_CALUDE_bob_tv_width_is_90_l846_84686


namespace NUMINAMATH_CALUDE_arccos_neg_half_eq_two_pi_thirds_l846_84656

theorem arccos_neg_half_eq_two_pi_thirds : 
  Real.arccos (-1/2) = 2*π/3 := by
  sorry

end NUMINAMATH_CALUDE_arccos_neg_half_eq_two_pi_thirds_l846_84656


namespace NUMINAMATH_CALUDE_simplify_radical_expression_l846_84616

theorem simplify_radical_expression (y : ℝ) (h : y > 0) :
  (32 * y) ^ (1/4) * (50 * y) ^ (1/4) + (18 * y) ^ (1/4) = 
  10 * (8 * y^2) ^ (1/4) + 3 * (2 * y) ^ (1/4) :=
by sorry

end NUMINAMATH_CALUDE_simplify_radical_expression_l846_84616


namespace NUMINAMATH_CALUDE_min_value_expression_min_value_achievable_l846_84605

theorem min_value_expression (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a / b) + (b / c) + (c / a) + Real.sqrt ((a / b)^2 + (b / c)^2 + (c / a)^2) ≥ 3 + Real.sqrt 3 :=
by sorry

theorem min_value_achievable :
  ∃ (a b c : ℝ), a > 0 ∧ b > 0 ∧ c > 0 ∧
    (a / b) + (b / c) + (c / a) + Real.sqrt ((a / b)^2 + (b / c)^2 + (c / a)^2) = 3 + Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_min_value_achievable_l846_84605


namespace NUMINAMATH_CALUDE_tangent_line_intersection_f_increasing_inequality_proof_l846_84602

noncomputable section

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := (x + 1) * Real.log x - a * (x - 1)

-- Define the derivative of f
def f' (a : ℝ) (x : ℝ) : ℝ := Real.log x + 1 / x + 1 - a

theorem tangent_line_intersection (a : ℝ) :
  (f' a (Real.exp 1)) * (Real.exp 1) = f a (Real.exp 1) - (2 - Real.exp 1) →
  a = 2 := by sorry

theorem f_increasing (a : ℝ) :
  a ≤ 2 →
  ∀ x > 0, f' a x ≥ 0 := by sorry

theorem inequality_proof (x : ℝ) :
  1 < x → x < 2 →
  (2 / (x - 1)) > (1 / Real.log x - 1 / Real.log (2 - x)) := by sorry

end

end NUMINAMATH_CALUDE_tangent_line_intersection_f_increasing_inequality_proof_l846_84602


namespace NUMINAMATH_CALUDE_angle_cosine_relation_l846_84688

theorem angle_cosine_relation (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  let r := Real.sqrt (x^2 + y^2 + z^2)
  x / r = 1/4 ∧ y / r = 1/8 → z / r = Real.sqrt 59 / 8 := by
  sorry

end NUMINAMATH_CALUDE_angle_cosine_relation_l846_84688


namespace NUMINAMATH_CALUDE_exists_irrational_in_interval_l846_84617

theorem exists_irrational_in_interval :
  ∃ x : ℝ, x ∈ Set.Ioo 0.3 0.4 ∧ Irrational x ∧ x * (x + 1) * (x + 2) = 1 := by
  sorry

end NUMINAMATH_CALUDE_exists_irrational_in_interval_l846_84617


namespace NUMINAMATH_CALUDE_negation_of_existence_l846_84696

theorem negation_of_existence (p : Prop) :
  (¬ ∃ x : ℝ, x^2 - x + 1 < 0) ↔ (∀ x : ℝ, x^2 - x + 1 ≥ 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_existence_l846_84696


namespace NUMINAMATH_CALUDE_coin_array_problem_l846_84667

/-- The sum of the first n natural numbers -/
def triangular_sum (n : ℕ) : ℕ := n * (n + 1) / 2

/-- Sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + sum_of_digits (n / 10)

/-- The problem statement -/
theorem coin_array_problem : 
  ∃ (N : ℕ), triangular_sum N = 5050 ∧ sum_of_digits N = 1 :=
sorry

end NUMINAMATH_CALUDE_coin_array_problem_l846_84667


namespace NUMINAMATH_CALUDE_cookie_sales_total_l846_84647

/-- Calculate the total money made from selling chocolate and vanilla cookies -/
theorem cookie_sales_total (chocolate_count : ℕ) (chocolate_price : ℕ) 
  (vanilla_count : ℕ) (vanilla_price : ℕ) :
  chocolate_count = 220 →
  chocolate_price = 1 →
  vanilla_count = 70 →
  vanilla_price = 2 →
  chocolate_count * chocolate_price + vanilla_count * vanilla_price = 360 := by
  sorry

end NUMINAMATH_CALUDE_cookie_sales_total_l846_84647


namespace NUMINAMATH_CALUDE_modulo_seven_residue_l846_84673

theorem modulo_seven_residue : (312 + 6 * 51 + 8 * 175 + 3 * 28) % 7 = 3 := by
  sorry

end NUMINAMATH_CALUDE_modulo_seven_residue_l846_84673


namespace NUMINAMATH_CALUDE_sufficient_condition_for_integral_inequality_not_necessary_condition_sufficient_but_not_necessary_condition_l846_84644

theorem sufficient_condition_for_integral_inequality (a : ℝ) :
  a > Real.exp 1 →
  a > ∫ x in (1 : ℝ)..Real.exp 1, 1 / x :=
by sorry

theorem not_necessary_condition (a : ℝ) :
  ¬(∀ a : ℝ, a > ∫ x in (1 : ℝ)..Real.exp 1, 1 / x → a > Real.exp 1) :=
by sorry

-- The main theorem combining both conditions
theorem sufficient_but_not_necessary_condition (a : ℝ) :
  (a > Real.exp 1 → a > ∫ x in (1 : ℝ)..Real.exp 1, 1 / x) ∧
  ¬(∀ a : ℝ, a > ∫ x in (1 : ℝ)..Real.exp 1, 1 / x → a > Real.exp 1) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_condition_for_integral_inequality_not_necessary_condition_sufficient_but_not_necessary_condition_l846_84644


namespace NUMINAMATH_CALUDE_triangle_count_l846_84609

/-- The number of distinct triangles that can be formed from 10 points -/
def num_triangles : ℕ := 120

/-- The number of points on the circle -/
def num_points : ℕ := 10

/-- The number of vertices in a triangle -/
def triangle_vertices : ℕ := 3

theorem triangle_count :
  Nat.choose num_points triangle_vertices = num_triangles :=
sorry

end NUMINAMATH_CALUDE_triangle_count_l846_84609


namespace NUMINAMATH_CALUDE_toms_speed_from_r_to_b_l846_84655

/-- Represents the speed of a journey between two towns -/
structure Journey where
  distance : ℝ
  speed : ℝ

/-- Represents Tom's entire trip -/
structure TripData where
  rb : Journey
  bc : Journey
  averageSpeed : ℝ

theorem toms_speed_from_r_to_b (trip : TripData) : trip.rb.speed = 60 :=
  by
  have h1 : trip.rb.distance = 2 * trip.bc.distance := by sorry
  have h2 : trip.averageSpeed = 36 := by sorry
  have h3 : trip.bc.speed = 20 := by sorry
  have h4 : trip.averageSpeed = (trip.rb.distance + trip.bc.distance) / 
    (trip.rb.distance / trip.rb.speed + trip.bc.distance / trip.bc.speed) := by sorry
  sorry


end NUMINAMATH_CALUDE_toms_speed_from_r_to_b_l846_84655


namespace NUMINAMATH_CALUDE_slope_divides_area_in_half_l846_84658

/-- L-shaped region in the xy-plane -/
structure LShapedRegion where
  vertices : List (ℝ × ℝ)
  is_l_shaped : vertices = [(0,0), (0,4), (4,4), (4,2), (6,2), (6,0)]

/-- Line passing through the origin -/
structure LineFromOrigin where
  slope : ℝ

/-- Function to calculate the area of the L-shaped region -/
def area (r : LShapedRegion) : ℝ :=
  20 -- The total area of the L-shaped region

/-- Function to calculate the area divided by a line -/
def area_divided_by_line (r : LShapedRegion) (l : LineFromOrigin) : ℝ × ℝ :=
  sorry -- Returns a pair of areas divided by the line

/-- Theorem stating that the slope 1/2 divides the area in half -/
theorem slope_divides_area_in_half (r : LShapedRegion) :
  let l := LineFromOrigin.mk (1/2)
  let (area1, area2) := area_divided_by_line r l
  area1 = area2 ∧ area1 + area2 = area r :=
sorry

end NUMINAMATH_CALUDE_slope_divides_area_in_half_l846_84658


namespace NUMINAMATH_CALUDE_company_average_salary_l846_84657

theorem company_average_salary
  (num_managers : ℕ)
  (num_associates : ℕ)
  (avg_salary_managers : ℚ)
  (avg_salary_associates : ℚ)
  (h1 : num_managers = 15)
  (h2 : num_associates = 75)
  (h3 : avg_salary_managers = 90000)
  (h4 : avg_salary_associates = 30000) :
  (num_managers * avg_salary_managers + num_associates * avg_salary_associates) /
  (num_managers + num_associates : ℚ) = 40000 :=
by sorry

end NUMINAMATH_CALUDE_company_average_salary_l846_84657


namespace NUMINAMATH_CALUDE_intersection_A_complement_B_a_range_when_intersection_empty_l846_84641

-- Define the sets A and B
def A : Set ℝ := {x | 1 < x ∧ x < 3}
def B (a : ℝ) : Set ℝ := {x | x > a}

-- Theorem for part 1
theorem intersection_A_complement_B :
  A ∩ (Set.univ \ B 2) = {x | 1 < x ∧ x ≤ 2} := by sorry

-- Theorem for part 2
theorem a_range_when_intersection_empty :
  ∀ a : ℝ, A ∩ B a = ∅ → a ≥ 3 := by sorry

end NUMINAMATH_CALUDE_intersection_A_complement_B_a_range_when_intersection_empty_l846_84641


namespace NUMINAMATH_CALUDE_percentage_problem_l846_84639

theorem percentage_problem (x : ℝ) (h1 : x > 0) (h2 : x * (x / 100) = 4) : x = 20 := by
  sorry

end NUMINAMATH_CALUDE_percentage_problem_l846_84639


namespace NUMINAMATH_CALUDE_square_sum_given_difference_and_product_l846_84629

theorem square_sum_given_difference_and_product (x y : ℝ) 
  (h1 : x - y = 20) 
  (h2 : x * y = 9) : 
  x^2 + y^2 = 418 := by
sorry

end NUMINAMATH_CALUDE_square_sum_given_difference_and_product_l846_84629


namespace NUMINAMATH_CALUDE_triangle_area_l846_84669

theorem triangle_area (a b c : ℝ) (A B C : ℝ) :
  c^2 = (a - b)^2 + 6 →
  C = π / 3 →
  (1/2) * a * b * Real.sin C = (3 * Real.sqrt 3) / 2 :=
by sorry

end NUMINAMATH_CALUDE_triangle_area_l846_84669


namespace NUMINAMATH_CALUDE_man_son_age_difference_l846_84632

/-- The age difference between a man and his son -/
def age_difference : ℕ → ℕ → ℕ
  | father_age, son_age => father_age - son_age

/-- Theorem stating the age difference between the man and his son -/
theorem man_son_age_difference :
  ∀ (man_age son_age : ℕ),
    son_age = 44 →
    man_age + 2 = 2 * (son_age + 2) →
    age_difference man_age son_age = 46 := by
  sorry

#check man_son_age_difference

end NUMINAMATH_CALUDE_man_son_age_difference_l846_84632


namespace NUMINAMATH_CALUDE_root_value_theorem_l846_84698

theorem root_value_theorem (a : ℝ) : a^2 + 3*a + 2 = 0 → a^2 + 3*a = -2 := by
  sorry

end NUMINAMATH_CALUDE_root_value_theorem_l846_84698


namespace NUMINAMATH_CALUDE_divisible_by_236_sum_of_middle_digits_l846_84610

theorem divisible_by_236_sum_of_middle_digits :
  ∀ (a b : ℕ),
  (a < 10 ∧ b < 10) →
  (6000 + 100 * a + 10 * b + 8) % 236 = 0 →
  a + b = 6 :=
by sorry

end NUMINAMATH_CALUDE_divisible_by_236_sum_of_middle_digits_l846_84610


namespace NUMINAMATH_CALUDE_third_grade_sample_size_l846_84695

/-- Represents the number of samples to be drawn from a stratum in stratified sampling -/
def stratumSample (totalSample : ℕ) (stratumRatio : ℕ) (totalRatio : ℕ) : ℕ :=
  totalSample * stratumRatio / totalRatio

/-- Theorem: In a stratified sampling with a total sample size of 200 and a population ratio of 5:2:3
    for three strata, the number of samples to be drawn from the third stratum is 60 -/
theorem third_grade_sample_size :
  let totalSample : ℕ := 200
  let firstRatio : ℕ := 5
  let secondRatio : ℕ := 2
  let thirdRatio : ℕ := 3
  let totalRatio : ℕ := firstRatio + secondRatio + thirdRatio
  stratumSample totalSample thirdRatio totalRatio = 60 := by
  sorry


end NUMINAMATH_CALUDE_third_grade_sample_size_l846_84695


namespace NUMINAMATH_CALUDE_intersection_point_is_unique_l846_84693

/-- The intersection point of two lines in 2D space -/
def intersection_point : ℚ × ℚ := (-9/7, 20/7)

/-- First line equation: 3y = -2x + 6 -/
def line1 (x y : ℚ) : Prop := 3 * y = -2 * x + 6

/-- Second line equation: -2y = 6x + 2 -/
def line2 (x y : ℚ) : Prop := -2 * y = 6 * x + 2

theorem intersection_point_is_unique :
  ∃! p : ℚ × ℚ, line1 p.1 p.2 ∧ line2 p.1 p.2 ∧ p = intersection_point :=
sorry

end NUMINAMATH_CALUDE_intersection_point_is_unique_l846_84693


namespace NUMINAMATH_CALUDE_range_of_a_l846_84625

-- Define the function f
def f (x : ℝ) : ℝ := |x - 1| - |x - 2|

-- Define the range M of y = 2f(x)
def M : Set ℝ := Set.range (λ x => 2 * f x)

-- Theorem statement
theorem range_of_a (a : ℝ) (h : Set.Icc a (2*a - 1) ⊆ M) : 1 ≤ a ∧ a ≤ 3/2 := by
  sorry


end NUMINAMATH_CALUDE_range_of_a_l846_84625


namespace NUMINAMATH_CALUDE_grunters_win_probability_l846_84634

theorem grunters_win_probability (n : ℕ) (p : ℚ) (h : p = 3/5) :
  p^n = 243/3125 → n = 5 :=
by
  sorry

end NUMINAMATH_CALUDE_grunters_win_probability_l846_84634


namespace NUMINAMATH_CALUDE_tan_seven_pi_sixths_l846_84643

theorem tan_seven_pi_sixths : Real.tan (7 * π / 6) = 1 / Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_tan_seven_pi_sixths_l846_84643


namespace NUMINAMATH_CALUDE_distance_on_line_l846_84679

/-- The distance between two points on a line y = mx + k -/
theorem distance_on_line (m k a b c d : ℝ) 
  (h1 : b = m * a + k) 
  (h2 : d = m * c + k) : 
  Real.sqrt ((a - c)^2 + (b - d)^2) = |a - c| * Real.sqrt (1 + m^2) := by
  sorry

end NUMINAMATH_CALUDE_distance_on_line_l846_84679


namespace NUMINAMATH_CALUDE_quadratic_inequality_always_negative_l846_84660

theorem quadratic_inequality_always_negative :
  ∀ x : ℝ, -7 * x^2 + 4 * x - 6 < 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_always_negative_l846_84660


namespace NUMINAMATH_CALUDE_pens_given_to_friends_l846_84642

def initial_pens : ℕ := 56
def remaining_pens : ℕ := 34

theorem pens_given_to_friends :
  initial_pens - remaining_pens = 22 := by
  sorry

end NUMINAMATH_CALUDE_pens_given_to_friends_l846_84642


namespace NUMINAMATH_CALUDE_function_equality_implies_sum_greater_than_four_l846_84635

open Real

noncomputable def f (x : ℝ) : ℝ := log x + 2 / x

theorem function_equality_implies_sum_greater_than_four
  (x₁ x₂ : ℝ)
  (h₁ : x₁ > 0)
  (h₂ : x₂ > 0)
  (h₃ : x₁ ≠ x₂)
  (h₄ : f x₁ = f x₂) :
  x₁ + x₂ > 4 :=
by sorry

end NUMINAMATH_CALUDE_function_equality_implies_sum_greater_than_four_l846_84635


namespace NUMINAMATH_CALUDE_c_is_largest_l846_84618

/-- Given that a - 1 = b + 2 = c - 3 = d + 4, prove that c is the largest among a, b, c, and d -/
theorem c_is_largest (a b c d : ℝ) (h : a - 1 = b + 2 ∧ b + 2 = c - 3 ∧ c - 3 = d + 4) :
  c = max a (max b (max c d)) := by
  sorry

end NUMINAMATH_CALUDE_c_is_largest_l846_84618


namespace NUMINAMATH_CALUDE_mod_equivalence_2023_l846_84637

theorem mod_equivalence_2023 :
  ∃! (n : ℤ), 0 ≤ n ∧ n ≤ 7 ∧ n ≡ -2023 [ZMOD 8] ∧ n = 1 := by
  sorry

end NUMINAMATH_CALUDE_mod_equivalence_2023_l846_84637


namespace NUMINAMATH_CALUDE_potions_needed_for_owl_l846_84685

/-- The number of Knuts in a Sickle -/
def knuts_per_sickle : ℕ := 23

/-- The number of Sickles in a Galleon -/
def sickles_per_galleon : ℕ := 17

/-- The cost of the owl in Galleons, Sickles, and Knuts -/
def owl_cost : ℕ × ℕ × ℕ := (2, 1, 5)

/-- The worth of each potion in Knuts -/
def potion_worth : ℕ := 9

/-- The function to calculate the total cost in Knuts -/
def total_cost_in_knuts (cost : ℕ × ℕ × ℕ) : ℕ :=
  cost.1 * sickles_per_galleon * knuts_per_sickle + 
  cost.2.1 * knuts_per_sickle + 
  cost.2.2

/-- The theorem stating the number of potions needed -/
theorem potions_needed_for_owl : 
  (total_cost_in_knuts owl_cost) / potion_worth = 90 := by
  sorry

end NUMINAMATH_CALUDE_potions_needed_for_owl_l846_84685


namespace NUMINAMATH_CALUDE_pencil_count_l846_84633

theorem pencil_count :
  ∀ (pens pencils : ℕ),
  (pens : ℚ) / (pencils : ℚ) = 5 / 6 →
  pencils = pens + 5 →
  pencils = 30 :=
by
  sorry

end NUMINAMATH_CALUDE_pencil_count_l846_84633


namespace NUMINAMATH_CALUDE_ratio_calculation_l846_84684

theorem ratio_calculation (P Q M N : ℝ) 
  (hM : M = 0.4 * Q) 
  (hQ : Q = 0.3 * P) 
  (hN : N = 0.5 * P) : 
  M / N = 6 / 25 := by
  sorry

end NUMINAMATH_CALUDE_ratio_calculation_l846_84684


namespace NUMINAMATH_CALUDE_triangle_perimeter_l846_84674

theorem triangle_perimeter (a b c : ℕ) (h1 : a = 3) (h2 : b = 8) (h3 : Odd c) 
  (h4 : a + b > c) (h5 : a + c > b) (h6 : b + c > a) : 
  a + b + c = 18 ∨ a + b + c = 20 :=
sorry

end NUMINAMATH_CALUDE_triangle_perimeter_l846_84674


namespace NUMINAMATH_CALUDE_water_level_rise_l846_84646

/-- Calculates the rise in water level when a cube is immersed in a rectangular vessel. -/
theorem water_level_rise 
  (cube_edge : ℝ) 
  (vessel_length : ℝ) 
  (vessel_width : ℝ) 
  (h_cube_edge : cube_edge = 5) 
  (h_vessel_length : vessel_length = 10) 
  (h_vessel_width : vessel_width = 5) : 
  (cube_edge ^ 3) / (vessel_length * vessel_width) = 2.5 := by
  sorry

end NUMINAMATH_CALUDE_water_level_rise_l846_84646


namespace NUMINAMATH_CALUDE_relationship_abc_l846_84675

theorem relationship_abc : 
  let a := (1/2) * Real.cos (2 * Real.pi / 180) - (Real.sqrt 3 / 2) * Real.sin (2 * Real.pi / 180)
  let b := (2 * Real.tan (14 * Real.pi / 180)) / (1 - Real.tan (14 * Real.pi / 180)^2)
  let c := Real.sqrt ((1 - Real.cos (50 * Real.pi / 180)) / 2)
  c < a ∧ a < b :=
sorry

end NUMINAMATH_CALUDE_relationship_abc_l846_84675


namespace NUMINAMATH_CALUDE_basketball_game_score_l846_84630

theorem basketball_game_score (a r b d : ℕ) : 
  -- Raiders' scores form a geometric sequence
  0 < a ∧ 1 < r ∧ 
  -- Wildcats' scores form an arithmetic sequence
  0 < b ∧ 0 < d ∧ 
  -- Game tied at end of first quarter
  a = b ∧ 
  -- Raiders won by one point
  a * (1 + r + r^2 + r^3) = 4 * b + 6 * d + 1 ∧ 
  -- Neither team scored more than 100 points
  a * (1 + r + r^2 + r^3) ≤ 100 ∧ 
  4 * b + 6 * d ≤ 100 →
  -- Total points in first half
  a + a * r + b + (b + d) = 34 := by
sorry

end NUMINAMATH_CALUDE_basketball_game_score_l846_84630


namespace NUMINAMATH_CALUDE_inequality_solution_set_l846_84671

theorem inequality_solution_set (x : ℝ) : 
  (5 - x^2 > 4*x) ↔ (x > -5 ∧ x < 1) :=
sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l846_84671


namespace NUMINAMATH_CALUDE_smaller_circle_radius_l846_84623

theorem smaller_circle_radius (R : ℝ) (r : ℝ) : 
  R = 12 →  -- Larger circle radius is 12 meters
  4 * (2 * r) = 2 * R →  -- Four smaller circles' diameters equal larger circle's diameter
  r = 3  -- Radius of smaller circle is 3 meters
:= by sorry

end NUMINAMATH_CALUDE_smaller_circle_radius_l846_84623


namespace NUMINAMATH_CALUDE_red_probability_both_jars_l846_84601

/-- Represents a jar containing buttons -/
structure Jar where
  red : ℕ
  blue : ℕ

/-- Calculates the total number of buttons in a jar -/
def Jar.total (j : Jar) : ℕ := j.red + j.blue

/-- Calculates the probability of drawing a red button from a jar -/
def Jar.redProbability (j : Jar) : ℚ := j.red / j.total

/-- Represents the initial state of Jar A -/
def initialJarA : Jar := { red := 8, blue := 8 }

/-- Represents the transfer process -/
def transfer (j : Jar) : (Jar × Jar) :=
  let redTransfer := j.red / 3
  let blueTransfer := j.blue / 2
  let newJarA : Jar := { red := j.red - redTransfer, blue := j.blue - blueTransfer }
  let jarB : Jar := { red := redTransfer, blue := blueTransfer }
  (newJarA, jarB)

/-- The main theorem stating the probability of drawing red buttons from both jars -/
theorem red_probability_both_jars :
  let (jarA, jarB) := transfer initialJarA
  (jarA.redProbability * jarB.redProbability) = 5 / 21 := by
  sorry


end NUMINAMATH_CALUDE_red_probability_both_jars_l846_84601


namespace NUMINAMATH_CALUDE_second_account_interest_rate_l846_84664

/-- Proves that the interest rate of the second account is 4% given the problem conditions --/
theorem second_account_interest_rate :
  ∀ (first_amount second_amount first_rate second_rate total_interest : ℝ),
    first_amount = 1000 →
    second_amount = first_amount + 800 →
    first_rate = 0.02 →
    total_interest = 92 →
    total_interest = first_rate * first_amount + second_rate * second_amount →
    second_rate = 0.04 := by
  sorry

end NUMINAMATH_CALUDE_second_account_interest_rate_l846_84664


namespace NUMINAMATH_CALUDE_largest_binomial_coefficient_sum_binomial_coefficient_sum_equals_six_largest_n_is_six_l846_84692

theorem largest_binomial_coefficient_sum (n : ℕ) : 
  (Nat.choose 10 4 + Nat.choose 10 5 = Nat.choose 11 n) → n ≤ 6 :=
by sorry

theorem binomial_coefficient_sum_equals_six : 
  Nat.choose 10 4 + Nat.choose 10 5 = Nat.choose 11 6 :=
by sorry

theorem largest_n_is_six : 
  ∃ (n : ℕ), n = 6 ∧ 
    Nat.choose 10 4 + Nat.choose 10 5 = Nat.choose 11 n ∧
    ∀ (m : ℕ), Nat.choose 10 4 + Nat.choose 10 5 = Nat.choose 11 m → m ≤ n :=
by sorry

end NUMINAMATH_CALUDE_largest_binomial_coefficient_sum_binomial_coefficient_sum_equals_six_largest_n_is_six_l846_84692


namespace NUMINAMATH_CALUDE_smallest_k_for_fraction_equation_l846_84622

theorem smallest_k_for_fraction_equation : 
  (∃ k : ℕ, k > 0 ∧ 
    (∃ a b : ℕ, a > 500000 ∧ 
      1 / (a : ℚ) + 1 / ((a + k) : ℚ) = 1 / (b : ℚ))) ∧ 
  (∀ k : ℕ, k > 0 → k < 1001 → 
    ¬(∃ a b : ℕ, a > 500000 ∧ 
      1 / (a : ℚ) + 1 / ((a + k) : ℚ) = 1 / (b : ℚ))) := by
  sorry

end NUMINAMATH_CALUDE_smallest_k_for_fraction_equation_l846_84622


namespace NUMINAMATH_CALUDE_probability_at_least_one_female_l846_84697

def total_students : ℕ := 10
def male_students : ℕ := 6
def female_students : ℕ := 4
def selected_students : ℕ := 3

theorem probability_at_least_one_female :
  let total_combinations := Nat.choose total_students selected_students
  let all_male_combinations := Nat.choose male_students selected_students
  (1 : ℚ) - (all_male_combinations : ℚ) / (total_combinations : ℚ) = 5 / 6 := by
  sorry

end NUMINAMATH_CALUDE_probability_at_least_one_female_l846_84697


namespace NUMINAMATH_CALUDE_square_inequality_l846_84631

theorem square_inequality (a b : ℝ) : a > |b| → a^2 > b^2 := by sorry

end NUMINAMATH_CALUDE_square_inequality_l846_84631


namespace NUMINAMATH_CALUDE_mod_29_graph_intercepts_sum_l846_84699

theorem mod_29_graph_intercepts_sum : ∃ (x₀ y₀ : ℕ), 
  x₀ < 29 ∧ y₀ < 29 ∧
  (∀ x : ℤ, (4 * x) % 29 = (5 * 0 - 1) % 29 ↔ x % 29 = x₀) ∧
  (∀ y : ℤ, (4 * 0) % 29 = (5 * y - 1) % 29 ↔ y % 29 = y₀) ∧
  x₀ + y₀ = 30 :=
by sorry

end NUMINAMATH_CALUDE_mod_29_graph_intercepts_sum_l846_84699


namespace NUMINAMATH_CALUDE_ones_digit_of_largest_power_of_two_dividing_32_factorial_l846_84668

def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

def largest_power_of_two_dividing (n : ℕ) : ℕ :=
  -- This function is not implemented, but represents the concept
  sorry

def ones_digit (n : ℕ) : ℕ :=
  n % 10

theorem ones_digit_of_largest_power_of_two_dividing_32_factorial :
  ones_digit (largest_power_of_two_dividing (factorial 32)) = 8 := by
  sorry

end NUMINAMATH_CALUDE_ones_digit_of_largest_power_of_two_dividing_32_factorial_l846_84668


namespace NUMINAMATH_CALUDE_inequality_proofs_l846_84648

theorem inequality_proofs (x : ℝ) : 
  (6 + 3 * x > 30 → x > 8) ∧ 
  (1 - x < 3 - (x - 5) / 2 → x > -9) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proofs_l846_84648


namespace NUMINAMATH_CALUDE_distribute_5_3_l846_84640

/-- The number of ways to distribute n distinct objects into k distinct boxes,
    with each box containing at least one object. -/
def distribute (n k : ℕ) : ℕ := sorry

/-- The number of ways to distribute 5 distinct objects into 3 distinct boxes,
    with each box containing at least one object, is 150. -/
theorem distribute_5_3 : distribute 5 3 = 150 := by sorry

end NUMINAMATH_CALUDE_distribute_5_3_l846_84640


namespace NUMINAMATH_CALUDE_scores_mode_l846_84606

def scores : List ℕ := [61, 62, 71, 78, 85, 85, 92, 96]

def mode (l : List ℕ) : ℕ := sorry

theorem scores_mode : mode scores = 85 := by sorry

end NUMINAMATH_CALUDE_scores_mode_l846_84606


namespace NUMINAMATH_CALUDE_deceased_member_income_l846_84613

/-- Given a family with 4 earning members and an average monthly income,
    calculate the income of a deceased member when the average income changes. -/
theorem deceased_member_income
  (initial_members : ℕ)
  (initial_average_income : ℚ)
  (final_members : ℕ)
  (final_average_income : ℚ)
  (h1 : initial_members = 4)
  (h2 : final_members = initial_members - 1)
  (h3 : initial_average_income = 840)
  (h4 : final_average_income = 650) :
  (initial_members : ℚ) * initial_average_income - (final_members : ℚ) * final_average_income = 1410 :=
by sorry

end NUMINAMATH_CALUDE_deceased_member_income_l846_84613


namespace NUMINAMATH_CALUDE_football_league_selection_l846_84653

def division_A : ℕ := 12
def division_B : ℕ := 8
def teams_to_select : ℕ := 5

theorem football_league_selection :
  -- Part 1
  (Nat.choose (division_A + division_B - 2) (teams_to_select - 1) = 3060) ∧
  -- Part 2
  (Nat.choose (division_A + division_B) teams_to_select -
   Nat.choose division_A teams_to_select -
   Nat.choose division_B teams_to_select = 14656) := by
  sorry

end NUMINAMATH_CALUDE_football_league_selection_l846_84653


namespace NUMINAMATH_CALUDE_base_2_representation_of_84_l846_84619

theorem base_2_representation_of_84 :
  ∃ (a b c d e f g : ℕ),
    (a = 1 ∧ b = 0 ∧ c = 1 ∧ d = 0 ∧ e = 1 ∧ f = 0 ∧ g = 0) ∧
    84 = a * 2^6 + b * 2^5 + c * 2^4 + d * 2^3 + e * 2^2 + f * 2^1 + g * 2^0 :=
by sorry

end NUMINAMATH_CALUDE_base_2_representation_of_84_l846_84619


namespace NUMINAMATH_CALUDE_investment_percentage_l846_84654

theorem investment_percentage (initial_investment : ℝ) (initial_rate : ℝ) 
  (additional_investment : ℝ) (additional_rate : ℝ) :
  initial_investment = 2000 →
  initial_rate = 0.05 →
  additional_investment = 999.9999999999998 →
  additional_rate = 0.08 →
  let total_investment := initial_investment + additional_investment
  let total_income := initial_investment * initial_rate + additional_investment * additional_rate
  (total_income / total_investment) * 100 = 6 := by
  sorry

end NUMINAMATH_CALUDE_investment_percentage_l846_84654


namespace NUMINAMATH_CALUDE_combined_fuel_efficiency_l846_84608

/-- The combined fuel efficiency of three cars -/
theorem combined_fuel_efficiency 
  (m : ℝ) -- distance driven by each car
  (h1 : m > 0) -- ensure distance is positive
  (efficiency1 efficiency2 efficiency3 : ℝ) -- individual car efficiencies
  (h2 : efficiency1 = 35) -- Ray's car efficiency
  (h3 : efficiency2 = 25) -- Tom's car efficiency
  (h4 : efficiency3 = 20) -- Alice's car efficiency
  : (3 * m) / (m / efficiency1 + m / efficiency2 + m / efficiency3) = 2100 / 83 := by
  sorry

#eval (2100 : ℚ) / 83 -- To show the decimal approximation

end NUMINAMATH_CALUDE_combined_fuel_efficiency_l846_84608


namespace NUMINAMATH_CALUDE_rand_code_is_1236_l846_84650

/-- A coding system that assigns numerical codes to words -/
structure CodeSystem where
  range_code : Nat
  random_code : Nat

/-- The code for a given word in the coding system -/
def word_code (system : CodeSystem) (word : String) : Nat :=
  sorry

/-- Our specific coding system -/
def our_system : CodeSystem :=
  { range_code := 12345, random_code := 123678 }

theorem rand_code_is_1236 :
  word_code our_system "rand" = 1236 := by
  sorry

end NUMINAMATH_CALUDE_rand_code_is_1236_l846_84650


namespace NUMINAMATH_CALUDE_parabola_vertex_l846_84682

/-- The parabola equation -/
def parabola_equation (x y : ℝ) : Prop := y = 3 * (x - 1)^2 + 8

/-- The vertex of the parabola -/
def vertex : ℝ × ℝ := (1, 8)

/-- Theorem: The vertex of the parabola y = 3(x-1)^2 + 8 is (1, 8) -/
theorem parabola_vertex :
  ∀ x y : ℝ, parabola_equation x y → (x, y) = vertex :=
sorry

end NUMINAMATH_CALUDE_parabola_vertex_l846_84682


namespace NUMINAMATH_CALUDE_black_coverage_probability_theorem_l846_84689

/-- Represents the square with black regions -/
structure ColoredSquare where
  side_length : ℝ
  triangle_leg : ℝ
  diamond_side : ℝ

/-- Represents the circular coin -/
structure Coin where
  diameter : ℝ

/-- Calculates the probability of the coin covering part of the black region -/
def black_coverage_probability (square : ColoredSquare) (coin : Coin) : ℝ :=
  sorry

/-- The main theorem to prove -/
theorem black_coverage_probability_theorem (square : ColoredSquare) (coin : Coin) :
  square.side_length = 10 ∧
  square.triangle_leg = 3 ∧
  square.diamond_side = 3 * Real.sqrt 2 ∧
  coin.diameter = 2 →
  black_coverage_probability square coin = (48 + 12 * Real.sqrt 2 + 2 * Real.pi) / 100 :=
sorry

end NUMINAMATH_CALUDE_black_coverage_probability_theorem_l846_84689


namespace NUMINAMATH_CALUDE_picture_book_shelves_l846_84607

/-- Given a bookcase with the following properties:
  * Each shelf contains exactly 6 books
  * There are 5 shelves of mystery books
  * The total number of books is 54
  Prove that the number of shelves of picture books is 4 -/
theorem picture_book_shelves :
  ∀ (books_per_shelf : ℕ) (mystery_shelves : ℕ) (total_books : ℕ),
    books_per_shelf = 6 →
    mystery_shelves = 5 →
    total_books = 54 →
    (total_books - mystery_shelves * books_per_shelf) / books_per_shelf = 4 :=
by sorry

end NUMINAMATH_CALUDE_picture_book_shelves_l846_84607


namespace NUMINAMATH_CALUDE_sisters_age_ratio_l846_84683

/-- Given John's current age and the future ages of John and his sister,
    prove that the ratio of his sister's age to his age is 2:1 -/
theorem sisters_age_ratio (johns_current_age : ℕ) (johns_future_age : ℕ) (sisters_future_age : ℕ)
  (h1 : johns_current_age = 10)
  (h2 : johns_future_age = 50)
  (h3 : sisters_future_age = 60) :
  (sisters_future_age - (johns_future_age - johns_current_age)) / johns_current_age = 2 :=
by sorry

end NUMINAMATH_CALUDE_sisters_age_ratio_l846_84683


namespace NUMINAMATH_CALUDE_A_inter_B_eq_l846_84615

def A : Set ℝ := {x | x + 1 > 0}
def B : Set ℝ := {x | x - 3 < 0}

theorem A_inter_B_eq : A ∩ B = {x : ℝ | -1 < x ∧ x < 3} := by
  sorry

end NUMINAMATH_CALUDE_A_inter_B_eq_l846_84615


namespace NUMINAMATH_CALUDE_even_tower_for_odd_walls_l846_84672

/-- A standard die has opposite faces summing to 7 -/
structure StandardDie where
  faces : Fin 6 → ℕ
  sum_opposite : ∀ i : Fin 3, faces i + faces (i + 3) = 7

/-- A tower of dice -/
def DiceTower (n : ℕ) := Fin n → StandardDie

/-- The sum of visible dots on a vertical wall of the tower -/
def wall_sum (tower : DiceTower n) (wall : Fin 4) : ℕ := sorry

theorem even_tower_for_odd_walls (n : ℕ) (tower : DiceTower n) :
  (∀ wall : Fin 4, Odd (wall_sum tower wall)) → Even n := by sorry

end NUMINAMATH_CALUDE_even_tower_for_odd_walls_l846_84672


namespace NUMINAMATH_CALUDE_postcard_cost_l846_84649

theorem postcard_cost (cost : ℕ) : cost = 111 :=
  by
  have h1 : 9 * cost < 1000 := sorry
  have h2 : 10 * cost > 1100 := sorry
  have h3 : cost > 0 := sorry
  sorry

end NUMINAMATH_CALUDE_postcard_cost_l846_84649


namespace NUMINAMATH_CALUDE_missing_number_problem_l846_84624

theorem missing_number_problem (x n : ℕ) (h_pos : x > 0) :
  let numbers := [x, x + 2, x + n, x + 7, x + 17]
  let mean := (x + (x + 2) + (x + n) + (x + 7) + (x + 17)) / 5
  let median := x + n
  (mean = median + 2) → n = 4 := by
sorry

end NUMINAMATH_CALUDE_missing_number_problem_l846_84624


namespace NUMINAMATH_CALUDE_min_value_of_expression_l846_84665

theorem min_value_of_expression (x : ℝ) (h : x > 3) :
  x + 4 / (x - 3) ≥ 7 ∧ (x + 4 / (x - 3) = 7 ↔ x = 5) := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_expression_l846_84665


namespace NUMINAMATH_CALUDE_ellipse_foci_y_axis_l846_84680

/-- An ellipse with foci on the y-axis represented by the equation x²/a - y²/b = 1 -/
structure Ellipse where
  a : ℝ
  b : ℝ
  is_ellipse : a > 0 ∧ b < 0 ∧ -b > a

theorem ellipse_foci_y_axis (e : Ellipse) : Real.sqrt (-e.b) > Real.sqrt e.a := by
  sorry

end NUMINAMATH_CALUDE_ellipse_foci_y_axis_l846_84680


namespace NUMINAMATH_CALUDE_solution_interval_l846_84627

theorem solution_interval (x₀ : ℝ) : 
  (1/2:ℝ)^x₀ = x₀^(1/3) → 1/3 < x₀ ∧ x₀ < 1/2 := by
  sorry

end NUMINAMATH_CALUDE_solution_interval_l846_84627


namespace NUMINAMATH_CALUDE_train_length_proof_l846_84626

theorem train_length_proof (bridge_length : ℝ) (crossing_time : ℝ) (train_speed : ℝ) :
  bridge_length = 300 ∧ crossing_time = 36 ∧ train_speed = 40 →
  train_speed * crossing_time - bridge_length = 1140 := by
  sorry

end NUMINAMATH_CALUDE_train_length_proof_l846_84626


namespace NUMINAMATH_CALUDE_chemical_equilibrium_and_precipitate_l846_84678

-- Define the chemical reaction parameters
def initial_BaCl2_concentration : ℝ := 10
def equilibrium_constant : ℝ := 5 * 10^6
def initial_volume : ℝ := 1

-- Define the molar mass of BaSO4
def molar_mass_BaSO4 : ℝ := 233.40

-- Define the theorem
theorem chemical_equilibrium_and_precipitate :
  ∃ (equilibrium_BaSO4_concentration : ℝ) (mass_BaSO4_precipitate : ℝ),
    (abs (equilibrium_BaSO4_concentration - 10) < 0.01) ∧
    (abs (mass_BaSO4_precipitate - 2334) < 0.1) :=
sorry

end NUMINAMATH_CALUDE_chemical_equilibrium_and_precipitate_l846_84678


namespace NUMINAMATH_CALUDE_geometric_sequences_exist_l846_84676

/-- Represents a geometric sequence --/
structure GeometricSequence where
  firstTerm : ℝ
  ratio : ℝ

/-- Represents three geometric sequences --/
structure ThreeGeometricSequences where
  seq1 : GeometricSequence
  seq2 : GeometricSequence
  seq3 : GeometricSequence

/-- Checks if the first terms of three geometric sequences form a geometric sequence with ratio 2 --/
def firstTermsFormGeometricSequence (s : ThreeGeometricSequences) : Prop :=
  s.seq2.firstTerm = 2 * s.seq1.firstTerm ∧ s.seq3.firstTerm = 2 * s.seq2.firstTerm

/-- Checks if the ratios of three geometric sequences form an arithmetic sequence with difference 1 --/
def ratiosFormArithmeticSequence (s : ThreeGeometricSequences) : Prop :=
  s.seq2.ratio = s.seq1.ratio + 1 ∧ s.seq3.ratio = s.seq2.ratio + 1

/-- Calculates the sum of the second terms of three geometric sequences --/
def sumOfSecondTerms (s : ThreeGeometricSequences) : ℝ :=
  s.seq1.firstTerm * s.seq1.ratio + s.seq2.firstTerm * s.seq2.ratio + s.seq3.firstTerm * s.seq3.ratio

/-- Calculates the sum of the first three terms of a geometric sequence --/
def sumOfFirstThreeTerms (s : GeometricSequence) : ℝ :=
  s.firstTerm + s.firstTerm * s.ratio + s.firstTerm * s.ratio^2

/-- The main theorem stating the existence of two sets of three geometric sequences satisfying the given conditions --/
theorem geometric_sequences_exist : 
  ∃ (s1 s2 : ThreeGeometricSequences), 
    firstTermsFormGeometricSequence s1 ∧
    firstTermsFormGeometricSequence s2 ∧
    ratiosFormArithmeticSequence s1 ∧
    ratiosFormArithmeticSequence s2 ∧
    sumOfSecondTerms s1 = 24 ∧
    sumOfSecondTerms s2 = 24 ∧
    sumOfFirstThreeTerms s1.seq3 = 84 ∧
    sumOfFirstThreeTerms s2.seq3 = 84 ∧
    s1 ≠ s2 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequences_exist_l846_84676


namespace NUMINAMATH_CALUDE_line_plane_parallelism_l846_84670

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the parallelism relation for lines and planes
variable (parallelLine : Line → Line → Prop)
variable (parallelLinePlane : Line → Plane → Prop)

-- Define the containment relation for lines in planes
variable (containedIn : Line → Plane → Prop)

-- Define the intersection operation for planes
variable (intersect : Plane → Plane → Line)

-- State the theorem
theorem line_plane_parallelism 
  (m n : Line) (α β : Plane) 
  (h_diff_lines : m ≠ n)
  (h_diff_planes : α ≠ β)
  (h_m_parallel_α : parallelLinePlane m α)
  (h_m_in_β : containedIn m β)
  (h_intersection : intersect α β = n) :
  parallelLine m n :=
sorry

end NUMINAMATH_CALUDE_line_plane_parallelism_l846_84670


namespace NUMINAMATH_CALUDE_statement_equivalence_l846_84651

theorem statement_equivalence (triangle_red circle_large : Prop) :
  (triangle_red → ¬circle_large) ↔ 
  (circle_large → ¬triangle_red) ∧ 
  (¬triangle_red ∨ ¬circle_large) := by sorry

end NUMINAMATH_CALUDE_statement_equivalence_l846_84651


namespace NUMINAMATH_CALUDE_three_true_propositions_l846_84687

theorem three_true_propositions
  (a₁ a₂ a₃ b₁ b₂ b₃ : ℝ)
  (h_a_order : a₁ < a₂ ∧ a₂ < a₃)
  (h_b_order : b₁ < b₂ ∧ b₂ < b₃)
  (h_sum : a₁ + a₂ + a₃ = b₁ + b₂ + b₃)
  (h_sum_prod : a₁*a₂ + a₁*a₃ + a₂*a₃ = b₁*b₂ + b₁*b₃ + b₂*b₃)
  (h_a₁_b₁ : a₁ < b₁) :
  ∃! (count : ℕ), count = 3 ∧ count = (
    (if b₂ < a₂ then 1 else 0) +
    (if a₃ < b₃ then 1 else 0) +
    (if a₁*a₂*a₃ < b₁*b₂*b₃ then 1 else 0) +
    (if (1-a₁)*(1-a₂)*(1-a₃) > (1-b₁)*(1-b₂)*(1-b₃) then 1 else 0)
  ) :=
sorry

end NUMINAMATH_CALUDE_three_true_propositions_l846_84687


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l846_84628

theorem sqrt_equation_solution (x : ℝ) : 
  Real.sqrt x + Real.sqrt (x + 4) = 8 → x = 225 / 16 := by
sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l846_84628


namespace NUMINAMATH_CALUDE_piece_sequence_properties_l846_84690

/-- Represents the number of small squares in the nth piece -/
def pieceSquares (n : ℕ) : ℕ := 2 * n - 1

/-- Represents the sum of small squares in pieces 1 to n -/
def totalSquares (n : ℕ) : ℕ := n * n

/-- Represents the sum of the first n even numbers -/
def sumFirstEvenNumbers (n : ℕ) : ℕ := n * (n + 1)

/-- Represents the sum of integers from 1 to n -/
def sumIntegers (n : ℕ) : ℕ := n * (n + 1) / 2

theorem piece_sequence_properties :
  (pieceSquares 50 = 99) ∧
  (totalSquares 50 = 2500) ∧
  (sumFirstEvenNumbers 50 = 2550) ∧
  (sumIntegers 100 = 5050) := by
  sorry

end NUMINAMATH_CALUDE_piece_sequence_properties_l846_84690


namespace NUMINAMATH_CALUDE_books_on_shelf_after_changes_l846_84652

/-- The total number of books on the shelf after Marta's changes -/
def total_books_after_changes (initial_fiction : ℕ) (initial_nonfiction : ℕ) 
  (added_fiction : ℕ) (removed_nonfiction : ℕ) (added_sets : ℕ) (books_per_set : ℕ) : ℕ :=
  (initial_fiction + added_fiction) + 
  (initial_nonfiction - removed_nonfiction) + 
  (added_sets * books_per_set)

/-- Theorem stating that the total number of books after changes is 70 -/
theorem books_on_shelf_after_changes : 
  total_books_after_changes 38 15 10 5 3 4 = 70 := by
  sorry

end NUMINAMATH_CALUDE_books_on_shelf_after_changes_l846_84652


namespace NUMINAMATH_CALUDE_inequality_proof_l846_84677

theorem inequality_proof (x y z : ℝ) 
  (h1 : x ≥ 0) (h2 : y ≥ 0) (h3 : z ≥ 0) 
  (h4 : x + y + z = 1) : 
  0 ≤ x * y + y * z + z * x - 2 * x * y * z ∧ 
  x * y + y * z + z * x - 2 * x * y * z ≤ 7 / 27 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l846_84677


namespace NUMINAMATH_CALUDE_complement_of_union_M_N_l846_84661

open Set

def U : Set ℕ := {1, 2, 3, 4, 5, 6}
def M : Set ℕ := {2, 3, 4}
def N : Set ℕ := {4, 5}

theorem complement_of_union_M_N :
  (M ∪ N)ᶜ = {1, 6} := by sorry

end NUMINAMATH_CALUDE_complement_of_union_M_N_l846_84661


namespace NUMINAMATH_CALUDE_rhombus_area_l846_84663

/-- Given a rhombus with side length √113 and diagonals differing by 10 units, 
    its area is 72 square units -/
theorem rhombus_area (s : ℝ) (d₁ d₂ : ℝ) : 
  s = Real.sqrt 113 →
  d₁ - d₂ = 10 →
  d₁ * d₂ = 4 * s^2 →
  (d₁ * d₂) / 2 = 72 := by
  sorry

end NUMINAMATH_CALUDE_rhombus_area_l846_84663


namespace NUMINAMATH_CALUDE_trader_loss_percentage_l846_84638

theorem trader_loss_percentage (cost_price : ℝ) (cost_price_pos : cost_price > 0) : 
  let marked_price := cost_price * 1.1
  let selling_price := marked_price * 0.9
  let loss := cost_price - selling_price
  loss / cost_price = 0.01 := by
sorry

end NUMINAMATH_CALUDE_trader_loss_percentage_l846_84638


namespace NUMINAMATH_CALUDE_largest_two_digit_number_with_conditions_l846_84662

theorem largest_two_digit_number_with_conditions : ∃ n : ℕ,
  10 ≤ n ∧ n < 100 ∧  -- two-digit number
  n % 3 = 0 ∧         -- divisible by 3
  n % 4 = 0 ∧         -- divisible by 4
  n % 5 = 4 ∧         -- remainder 4 when divided by 5
  ∀ m : ℕ, (10 ≤ m ∧ m < 100 ∧ m % 3 = 0 ∧ m % 4 = 0 ∧ m % 5 = 4) → m ≤ n :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_largest_two_digit_number_with_conditions_l846_84662


namespace NUMINAMATH_CALUDE_hotel_flat_fee_l846_84620

/-- Given a hotel charging a flat fee for the first night and a fixed amount for additional nights,
    prove that the flat fee is $60 if a 4-night stay costs $205 and a 7-night stay costs $350. -/
theorem hotel_flat_fee (flat_fee nightly_fee : ℚ) : 
  (flat_fee + 3 * nightly_fee = 205) →
  (flat_fee + 6 * nightly_fee = 350) →
  flat_fee = 60 := by sorry

end NUMINAMATH_CALUDE_hotel_flat_fee_l846_84620


namespace NUMINAMATH_CALUDE_gcd_8321_6489_l846_84636

theorem gcd_8321_6489 : Nat.gcd 8321 6489 = 1 := by
  sorry

end NUMINAMATH_CALUDE_gcd_8321_6489_l846_84636


namespace NUMINAMATH_CALUDE_smaller_number_problem_l846_84666

theorem smaller_number_problem (x y : ℝ) : 
  y = 3 * x → x + y = 124 → x = 31 := by
sorry

end NUMINAMATH_CALUDE_smaller_number_problem_l846_84666


namespace NUMINAMATH_CALUDE_sin_cos_sum_17_43_l846_84621

theorem sin_cos_sum_17_43 :
  Real.sin (17 * π / 180) * Real.cos (43 * π / 180) +
  Real.cos (17 * π / 180) * Real.sin (43 * π / 180) =
  Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_sum_17_43_l846_84621


namespace NUMINAMATH_CALUDE_prime_power_sum_implies_power_of_three_l846_84614

theorem prime_power_sum_implies_power_of_three (n : ℕ) :
  Nat.Prime (1 + 2^n + 4^n) → ∃ k : ℕ, n = 3^k :=
by sorry

end NUMINAMATH_CALUDE_prime_power_sum_implies_power_of_three_l846_84614


namespace NUMINAMATH_CALUDE_park_problem_solution_l846_84659

/-- The problem setup -/
structure ParkProblem where
  distance_to_park : ℝ
  mother_speed_ratio : ℝ
  time_difference : ℝ
  distance_to_company : ℝ
  mother_run_speed : ℝ
  available_time : ℝ

/-- The solution to be proved -/
structure ParkSolution where
  mother_speed : ℝ
  min_run_time : ℝ

/-- The main theorem to be proved -/
theorem park_problem_solution (p : ParkProblem) 
  (h1 : p.distance_to_park = 4320)
  (h2 : p.mother_speed_ratio = 1.2)
  (h3 : p.time_difference = 12)
  (h4 : p.distance_to_company = 2940)
  (h5 : p.mother_run_speed = 150)
  (h6 : p.available_time = 30) :
  ∃ (s : ParkSolution), 
    s.mother_speed = 72 ∧ 
    s.min_run_time = 10 ∧
    (p.distance_to_park / s.mother_speed - p.distance_to_park / (s.mother_speed / p.mother_speed_ratio) = p.time_difference) ∧
    ((p.distance_to_company - p.mother_run_speed * s.min_run_time) / s.mother_speed + s.min_run_time ≤ p.available_time) := by
  sorry

end NUMINAMATH_CALUDE_park_problem_solution_l846_84659


namespace NUMINAMATH_CALUDE_simplify_fraction_l846_84691

theorem simplify_fraction (a b : ℝ) (h1 : a + b ≠ 0) (h2 : a + 2*b ≠ 0) :
  (a - b) / (a + 2*b) / ((a^2 - b^2) / (a^2 + 4*a*b + 4*b^2)) - 2 = -a / (a + b) := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_l846_84691


namespace NUMINAMATH_CALUDE_function_with_given_derivative_l846_84611

/-- Given a differentiable function f on ℝ with f'(x) = 1 + sin x,
    prove that there exists a constant C such that f(x) = x - cos x + C. -/
theorem function_with_given_derivative
  (f : ℝ → ℝ) (hf : Differentiable ℝ f) (hf' : ∀ x, deriv f x = 1 + Real.sin x) :
  ∃ C, ∀ x, f x = x - Real.cos x + C := by
  sorry

end NUMINAMATH_CALUDE_function_with_given_derivative_l846_84611


namespace NUMINAMATH_CALUDE_grandfathers_age_l846_84694

theorem grandfathers_age : ∃! n : ℕ, 
  10 ≤ n ∧ n < 100 ∧ 
  (n / 10 : ℕ) + (n % 10)^2 = n :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_grandfathers_age_l846_84694


namespace NUMINAMATH_CALUDE_anya_balloons_count_l846_84604

def total_balloons : ℕ := 672
def num_colors : ℕ := 4

theorem anya_balloons_count : 
  let balloons_per_color := total_balloons / num_colors
  let anya_balloons := balloons_per_color / 2
  anya_balloons = 84 := by sorry

end NUMINAMATH_CALUDE_anya_balloons_count_l846_84604


namespace NUMINAMATH_CALUDE_cookie_brownie_difference_l846_84600

def cookies_remaining (initial_cookies : ℕ) (daily_cookie_consumption : ℕ) (days : ℕ) : ℕ :=
  initial_cookies - daily_cookie_consumption * days

def brownies_remaining (initial_brownies : ℕ) (daily_brownie_consumption : ℕ) (days : ℕ) : ℕ :=
  initial_brownies - daily_brownie_consumption * days

theorem cookie_brownie_difference :
  let initial_cookies : ℕ := 60
  let initial_brownies : ℕ := 10
  let daily_cookie_consumption : ℕ := 3
  let daily_brownie_consumption : ℕ := 1
  let days : ℕ := 7
  cookies_remaining initial_cookies daily_cookie_consumption days -
  brownies_remaining initial_brownies daily_brownie_consumption days = 36 := by
  sorry

end NUMINAMATH_CALUDE_cookie_brownie_difference_l846_84600


namespace NUMINAMATH_CALUDE_loss_percent_calculation_l846_84612

theorem loss_percent_calculation (cost_price selling_price : ℝ) 
  (h1 : cost_price = 600)
  (h2 : selling_price = 450) :
  (cost_price - selling_price) / cost_price * 100 = 25 := by
  sorry

end NUMINAMATH_CALUDE_loss_percent_calculation_l846_84612


namespace NUMINAMATH_CALUDE_inequality_solution_l846_84603

theorem inequality_solution (x : ℝ) : 2 ≤ x / (2 * x - 4) ∧ x / (2 * x - 4) < 7 ↔ x ∈ Set.Ici 2 ∩ Set.Iio (28 / 13) :=
sorry

end NUMINAMATH_CALUDE_inequality_solution_l846_84603


namespace NUMINAMATH_CALUDE_quadruple_sum_square_l846_84645

theorem quadruple_sum_square (a b c d m n : ℕ+) : 
  a^2 + b^2 + c^2 + d^2 = 1989 →
  a + b + c + d = m^2 →
  max a (max b (max c d)) = n^2 →
  m = 9 ∧ n = 6 := by
sorry

end NUMINAMATH_CALUDE_quadruple_sum_square_l846_84645


namespace NUMINAMATH_CALUDE_tangent_line_property_l846_84681

theorem tangent_line_property (x₁ x₂ : ℝ) (h₁ : x₁ > 0) (h₂ : x₁ ≠ 1) :
  (((1 : ℝ) / x₁ = Real.exp x₂) ∧
   (Real.log x₁ - 1 = Real.exp x₂ * (1 - x₂))) →
  2 / (x₁ - 1) + x₂ = -1 := by sorry

end NUMINAMATH_CALUDE_tangent_line_property_l846_84681
