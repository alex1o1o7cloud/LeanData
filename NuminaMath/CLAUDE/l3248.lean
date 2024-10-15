import Mathlib

namespace NUMINAMATH_CALUDE_base5_to_base7_conversion_l3248_324841

/-- Converts a number from base 5 to base 10 -/
def base5_to_decimal (n : ℕ) : ℕ := sorry

/-- Converts a number from base 10 to base 7 -/
def decimal_to_base7 (n : ℕ) : ℕ := sorry

/-- Theorem: The base-7 representation of 412₅ is 212₇ -/
theorem base5_to_base7_conversion :
  decimal_to_base7 (base5_to_decimal 412) = 212 := by sorry

end NUMINAMATH_CALUDE_base5_to_base7_conversion_l3248_324841


namespace NUMINAMATH_CALUDE_cosine_inequality_solution_l3248_324857

theorem cosine_inequality_solution (y : Real) : 
  (y ∈ Set.Icc 0 Real.pi ∧ 
   ∀ x ∈ Set.Icc 0 Real.pi, Real.cos (x + y) ≥ Real.cos x + Real.cos y - 1) ↔ 
  (y = 0 ∨ y = Real.pi) := by
sorry

end NUMINAMATH_CALUDE_cosine_inequality_solution_l3248_324857


namespace NUMINAMATH_CALUDE_inequality_equivalence_l3248_324881

theorem inequality_equivalence (x : ℝ) :
  (x - 3) / (2 - x) ≥ 0 ↔ (3 - x) / (x - 2) ≥ 0 :=
by sorry

end NUMINAMATH_CALUDE_inequality_equivalence_l3248_324881


namespace NUMINAMATH_CALUDE_orange_pyramid_count_l3248_324817

def pyramid_oranges (base_length : ℕ) (base_width : ℕ) (top_oranges : ℕ) : ℕ :=
  let layers := min base_length base_width
  (layers * (base_length + base_width - layers + 1) * (2 * base_length + 2 * base_width - 3 * layers + 1)) / 6 + top_oranges

theorem orange_pyramid_count : pyramid_oranges 7 10 3 = 227 := by
  sorry

end NUMINAMATH_CALUDE_orange_pyramid_count_l3248_324817


namespace NUMINAMATH_CALUDE_polynomial_simplification_l3248_324849

theorem polynomial_simplification (x : ℝ) :
  (3 * x^2 + 9 * x - 5) - (2 * x^2 + 4 * x - 15) = x^2 + 5 * x + 10 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_simplification_l3248_324849


namespace NUMINAMATH_CALUDE_g_of_3_l3248_324827

def g (x : ℝ) : ℝ := 5 * x^4 + 4 * x^3 - 7 * x^2 + 3 * x - 2

theorem g_of_3 : g 3 = 401 := by
  sorry

end NUMINAMATH_CALUDE_g_of_3_l3248_324827


namespace NUMINAMATH_CALUDE_time_difference_per_question_l3248_324885

/-- Prove that the difference in time per question between the Math and English exams is 4 minutes -/
theorem time_difference_per_question (english_questions math_questions : ℕ) 
  (english_duration math_duration : ℚ) : 
  english_questions = 30 →
  math_questions = 15 →
  english_duration = 1 →
  math_duration = 3/2 →
  (math_duration * 60 / math_questions) - (english_duration * 60 / english_questions) = 4 := by
  sorry

end NUMINAMATH_CALUDE_time_difference_per_question_l3248_324885


namespace NUMINAMATH_CALUDE_probability_multiple_3_or_5_l3248_324808

def is_multiple_of_3_or_5 (n : ℕ) : Bool :=
  n % 3 = 0 || n % 5 = 0

def count_multiples (max : ℕ) : ℕ :=
  (List.range max).filter is_multiple_of_3_or_5 |>.length

theorem probability_multiple_3_or_5 :
  (count_multiples 20 : ℚ) / 20 = 9 / 20 := by
  sorry

end NUMINAMATH_CALUDE_probability_multiple_3_or_5_l3248_324808


namespace NUMINAMATH_CALUDE_claire_shirts_count_l3248_324828

theorem claire_shirts_count :
  ∀ (brian_shirts andrew_shirts steven_shirts claire_shirts : ℕ),
    brian_shirts = 3 →
    andrew_shirts = 6 * brian_shirts →
    steven_shirts = 4 * andrew_shirts →
    claire_shirts = 5 * steven_shirts →
    claire_shirts = 360 := by
  sorry

end NUMINAMATH_CALUDE_claire_shirts_count_l3248_324828


namespace NUMINAMATH_CALUDE_expression_simplification_l3248_324866

theorem expression_simplification (a c : ℝ) (ha : a ≠ 0) (hc : c ≠ 0) (hac : a ≠ c) :
  (c^2 - a^2) / (c * a) - (c * a - c^2) / (c * a - a^2) = (2 * c^2 - a^2) / (c * a) := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l3248_324866


namespace NUMINAMATH_CALUDE_fourth_root_over_seventh_root_of_seven_l3248_324812

theorem fourth_root_over_seventh_root_of_seven (x : ℝ) (h : x = 7) :
  (x^(1/4)) / (x^(1/7)) = x^(3/28) :=
by sorry

end NUMINAMATH_CALUDE_fourth_root_over_seventh_root_of_seven_l3248_324812


namespace NUMINAMATH_CALUDE_smallest_cube_for_cone_l3248_324887

/-- Represents a right circular cone -/
structure Cone where
  height : ℝ
  baseDiameter : ℝ

/-- Represents a cube -/
structure Cube where
  sideLength : ℝ

/-- The volume of a cube -/
def cubeVolume (c : Cube) : ℝ :=
  c.sideLength ^ 3

/-- Predicate to check if a cube can contain a cone upright -/
def canContainCone (cube : Cube) (cone : Cone) : Prop :=
  cube.sideLength ≥ cone.height ∧ cube.sideLength ≥ cone.baseDiameter

theorem smallest_cube_for_cone (cone : Cone) 
    (h1 : cone.height = 15)
    (h2 : cone.baseDiameter = 8) :
    ∃ (cube : Cube), 
      canContainCone cube cone ∧ 
      cubeVolume cube = 3375 ∧
      ∀ (c : Cube), canContainCone c cone → cubeVolume c ≥ 3375 := by
  sorry

end NUMINAMATH_CALUDE_smallest_cube_for_cone_l3248_324887


namespace NUMINAMATH_CALUDE_decagon_triangle_probability_l3248_324807

/-- A decagon is a polygon with 10 vertices -/
def Decagon : ℕ := 10

/-- The number of ways to choose 3 distinct vertices from a decagon -/
def TotalChoices : ℕ := Nat.choose Decagon 3

/-- The number of ways to choose 3 distinct vertices that form a triangle with sides as edges -/
def FavorableChoices : ℕ := Decagon

/-- The probability of choosing 3 distinct vertices that form a triangle with sides as edges -/
def ProbabilityOfTriangle : ℚ := FavorableChoices / TotalChoices

theorem decagon_triangle_probability :
  ProbabilityOfTriangle = 1 / 12 := by
  sorry

end NUMINAMATH_CALUDE_decagon_triangle_probability_l3248_324807


namespace NUMINAMATH_CALUDE_positive_solution_x_l3248_324834

theorem positive_solution_x (x y z : ℝ) 
  (eq1 : x * y = 8 - 2 * x - 3 * y)
  (eq2 : y * z = 8 - 4 * y - 2 * z)
  (eq3 : x * z = 40 - 4 * x - 3 * z)
  (pos : x > 0) :
  x = (7 * Real.sqrt 13 - 6) / 2 := by
sorry

end NUMINAMATH_CALUDE_positive_solution_x_l3248_324834


namespace NUMINAMATH_CALUDE_initial_blue_marbles_l3248_324870

/-- Proves that the initial number of blue marbles is 30 given the conditions of the problem -/
theorem initial_blue_marbles (initial_red : ℕ) (removed_red : ℕ) (total_left : ℕ)
  (h1 : initial_red = 20)
  (h2 : removed_red = 3)
  (h3 : total_left = 35) :
  initial_red + (total_left + removed_red + 4 * removed_red - (initial_red - removed_red)) = 30 := by
  sorry

end NUMINAMATH_CALUDE_initial_blue_marbles_l3248_324870


namespace NUMINAMATH_CALUDE_z_in_second_quadrant_l3248_324899

def z : ℂ := (2 + Complex.I) * Complex.I

theorem z_in_second_quadrant : 
  Real.sign (z.re) = -1 ∧ Real.sign (z.im) = 1 :=
sorry

end NUMINAMATH_CALUDE_z_in_second_quadrant_l3248_324899


namespace NUMINAMATH_CALUDE_particle_probability_l3248_324836

/-- The probability of a particle reaching (0,0) from (x,y) -/
def P (x y : ℕ) : ℚ :=
  if x = 0 ∧ y = 0 then 1
  else if x = 0 ∨ y = 0 then 0
  else (1/3) * P (x-1) y + (1/3) * P x (y-1) + (1/3) * P (x-1) (y-1)

theorem particle_probability :
  P 5 5 = 793 / 6561 :=
sorry

end NUMINAMATH_CALUDE_particle_probability_l3248_324836


namespace NUMINAMATH_CALUDE_line_slope_45_degrees_l3248_324852

theorem line_slope_45_degrees (y : ℝ) : 
  (∃ (line : Set (ℝ × ℝ)), 
    (⟨4, y⟩ ∈ line) ∧ 
    (⟨2, -3⟩ ∈ line) ∧ 
    (∀ (x₁ y₁ x₂ y₂ : ℝ), ⟨x₁, y₁⟩ ∈ line → ⟨x₂, y₂⟩ ∈ line → x₁ ≠ x₂ → (y₂ - y₁) / (x₂ - x₁) = 1)) →
  y = -1 := by
sorry

end NUMINAMATH_CALUDE_line_slope_45_degrees_l3248_324852


namespace NUMINAMATH_CALUDE_football_team_throwers_l3248_324815

theorem football_team_throwers :
  ∀ (total_players throwers right_handed : ℕ),
    total_players = 70 →
    throwers ≤ total_players →
    right_handed = 63 →
    3 * (right_handed - throwers) = 2 * (total_players - throwers) →
    throwers = 49 := by
  sorry

end NUMINAMATH_CALUDE_football_team_throwers_l3248_324815


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l3248_324868

-- Define the sets A and B
def A : Set ℝ := {x | x > 0}
def B : Set ℝ := {y | ∃ x, y = 1 - x^2}

-- State the theorem
theorem intersection_of_A_and_B : A ∩ B = {x | 0 < x ∧ x ≤ 1} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l3248_324868


namespace NUMINAMATH_CALUDE_sum_of_coefficients_equals_value_at_one_l3248_324855

/-- The polynomial for which we want to find the sum of coefficients -/
def p (x : ℝ) : ℝ := 3*(x^8 - 2*x^5 + 4*x^3 - 6) - 5*(x^4 - 3*x + 7) + 2*(x^6 - 5)

/-- The sum of coefficients of a polynomial is equal to its value at x = 1 -/
theorem sum_of_coefficients_equals_value_at_one :
  p 1 = -42 := by sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_equals_value_at_one_l3248_324855


namespace NUMINAMATH_CALUDE_total_spent_correct_l3248_324819

def regular_fee : ℝ := 150
def discount_rate : ℝ := 0.075
def tax_rate : ℝ := 0.06
def total_teachers : ℕ := 22
def special_diet_teachers : ℕ := 3
def regular_food_allowance : ℝ := 10
def special_food_allowance : ℝ := 15

def total_spent : ℝ :=
  let discounted_fee := regular_fee * (1 - discount_rate) * total_teachers
  let taxed_fee := discounted_fee * (1 + tax_rate)
  let food_allowance := regular_food_allowance * (total_teachers - special_diet_teachers) +
                        special_food_allowance * special_diet_teachers
  taxed_fee + food_allowance

theorem total_spent_correct :
  total_spent = 3470.65 := by sorry

end NUMINAMATH_CALUDE_total_spent_correct_l3248_324819


namespace NUMINAMATH_CALUDE_sweater_fraction_is_one_fourth_l3248_324863

/-- The amount Leila spent on the sweater -/
def sweater_cost : ℕ := 40

/-- The amount Leila had left after buying jewelry -/
def remaining_money : ℕ := 20

/-- The additional amount Leila spent on jewelry compared to the sweater -/
def jewelry_additional_cost : ℕ := 60

/-- Leila's total initial money -/
def total_money : ℕ := sweater_cost + remaining_money + sweater_cost + jewelry_additional_cost

/-- The fraction of total money spent on the sweater -/
def sweater_fraction : ℚ := sweater_cost / total_money

theorem sweater_fraction_is_one_fourth : sweater_fraction = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_sweater_fraction_is_one_fourth_l3248_324863


namespace NUMINAMATH_CALUDE_hair_dye_cost_salon_hair_dye_cost_l3248_324804

/-- Calculates the cost of a box of hair dye based on salon revenue and expenses --/
theorem hair_dye_cost (haircut_price perm_price dye_job_price : ℕ)
  (haircuts perms dye_jobs : ℕ) (tips final_amount : ℕ) : ℕ :=
  let total_revenue := haircut_price * haircuts + perm_price * perms + dye_job_price * dye_jobs + tips
  let dye_cost := total_revenue - final_amount
  dye_cost / dye_jobs

/-- Proves that the cost of a box of hair dye is $10 given the problem conditions --/
theorem salon_hair_dye_cost : hair_dye_cost 30 40 60 4 1 2 50 310 = 10 := by
  sorry

end NUMINAMATH_CALUDE_hair_dye_cost_salon_hair_dye_cost_l3248_324804


namespace NUMINAMATH_CALUDE_abs_x_minus_two_plus_three_min_l3248_324891

theorem abs_x_minus_two_plus_three_min (x : ℝ) : 
  ∃ (min : ℝ), (∀ x, |x - 2| + 3 ≥ min) ∧ (∃ x, |x - 2| + 3 = min) := by
  sorry

end NUMINAMATH_CALUDE_abs_x_minus_two_plus_three_min_l3248_324891


namespace NUMINAMATH_CALUDE_selection_problem_l3248_324805

theorem selection_problem (n m k l : ℕ) (h1 : n = 12) (h2 : m = 5) (h3 : k = 3) (h4 : l = 2) : 
  (Nat.choose n m) - (Nat.choose k (l + 1) * Nat.choose (n - k) (m - l - 1)) = 756 := by
  sorry

end NUMINAMATH_CALUDE_selection_problem_l3248_324805


namespace NUMINAMATH_CALUDE_second_candidate_percentage_l3248_324869

theorem second_candidate_percentage : ∀ (total_marks : ℕ) (passing_mark : ℕ),
  passing_mark = 160 →
  (0.4 : ℝ) * total_marks = passing_mark - 40 →
  let second_candidate_marks := passing_mark + 20
  ((second_candidate_marks : ℝ) / total_marks) * 100 = 60 := by
  sorry

end NUMINAMATH_CALUDE_second_candidate_percentage_l3248_324869


namespace NUMINAMATH_CALUDE_relationship_proof_l3248_324800

theorem relationship_proof (a : ℝ) (h : a^2 + a < 0) : -a > a^2 ∧ a^2 > -a^3 := by
  sorry

end NUMINAMATH_CALUDE_relationship_proof_l3248_324800


namespace NUMINAMATH_CALUDE_complex_square_equation_l3248_324880

theorem complex_square_equation (a b : ℕ+) :
  (a + b * Complex.I) ^ 2 = 7 + 24 * Complex.I →
  a + b * Complex.I = 4 + 3 * Complex.I := by
sorry

end NUMINAMATH_CALUDE_complex_square_equation_l3248_324880


namespace NUMINAMATH_CALUDE_divisibility_condition_l3248_324897

theorem divisibility_condition (x : ℤ) : (x - 1) ∣ (x - 3) ↔ x ∈ ({-1, 0, 2, 3} : Set ℤ) := by
  sorry

end NUMINAMATH_CALUDE_divisibility_condition_l3248_324897


namespace NUMINAMATH_CALUDE_gcd_of_B_is_five_l3248_324888

def B : Set ℕ := {n : ℕ | ∃ x : ℕ, n = (x - 2) + (x - 1) + x + (x + 1) + (x + 2)}

theorem gcd_of_B_is_five : 
  ∃ d : ℕ, d > 0 ∧ (∀ n ∈ B, d ∣ n) ∧ (∀ m : ℕ, (∀ n ∈ B, m ∣ n) → m ∣ d) ∧ d = 5 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_B_is_five_l3248_324888


namespace NUMINAMATH_CALUDE_scaling_transformation_for_given_points_l3248_324872

/-- A scaling transformation in 2D space -/
structure ScalingTransformation where
  sx : ℝ  -- scaling factor for x
  sy : ℝ  -- scaling factor for y

/-- Apply a scaling transformation to a point -/
def apply_scaling (t : ScalingTransformation) (p : ℝ × ℝ) : ℝ × ℝ :=
  (t.sx * p.1, t.sy * p.2)

theorem scaling_transformation_for_given_points :
  ∃ (t : ScalingTransformation),
    apply_scaling t (-2, 2) = (-6, 1) ∧
    t.sx = 3 ∧
    t.sy = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_scaling_transformation_for_given_points_l3248_324872


namespace NUMINAMATH_CALUDE_evaluate_expression_l3248_324844

theorem evaluate_expression : 3 - 5 * (2^3 + 3) * 2 = -107 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l3248_324844


namespace NUMINAMATH_CALUDE_combined_weight_theorem_l3248_324847

def leo_weight : ℝ := 80
def weight_gain : ℝ := 10

theorem combined_weight_theorem (kendra_weight : ℝ) 
  (h : leo_weight + weight_gain = 1.5 * kendra_weight) :
  leo_weight + kendra_weight = 140 := by
sorry

end NUMINAMATH_CALUDE_combined_weight_theorem_l3248_324847


namespace NUMINAMATH_CALUDE_cubic_roots_sum_l3248_324820

theorem cubic_roots_sum (a b c : ℝ) : 
  (3 * a^3 - 5 * a^2 + 90 * a - 2 = 0) →
  (3 * b^3 - 5 * b^2 + 90 * b - 2 = 0) →
  (3 * c^3 - 5 * c^2 + 90 * c - 2 = 0) →
  (a + b + 1)^3 + (b + c + 1)^3 + (c + a + 1)^3 = 259 + 1/3 := by
  sorry

end NUMINAMATH_CALUDE_cubic_roots_sum_l3248_324820


namespace NUMINAMATH_CALUDE_even_function_implies_m_equals_one_l3248_324813

/-- A function f: ℝ → ℝ is even if f(-x) = f(x) for all x ∈ ℝ -/
def IsEven (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f (-x) = f x

/-- The function f(x) = -x^2 + 2(m-1)x + 3 -/
def f (m : ℝ) (x : ℝ) : ℝ := -x^2 + 2*(m-1)*x + 3

theorem even_function_implies_m_equals_one :
  ∀ m : ℝ, IsEven (f m) → m = 1 := by
  sorry

end NUMINAMATH_CALUDE_even_function_implies_m_equals_one_l3248_324813


namespace NUMINAMATH_CALUDE_abc_inequality_l3248_324882

theorem abc_inequality (a b c : ℝ) (ha : 0 ≤ a) (hb : 0 ≤ b) (hc : 0 ≤ c)
  (h : a^2 + b^2 + c^2 + a*b*c = 4) :
  0 ≤ a*b + b*c + c*a - a*b*c ∧ a*b + b*c + c*a - a*b*c ≤ 2 := by
  sorry

end NUMINAMATH_CALUDE_abc_inequality_l3248_324882


namespace NUMINAMATH_CALUDE_largest_package_size_l3248_324829

/-- The largest possible number of markers in a package given that Lucy bought 30 markers, 
    Mia bought 45 markers, and Noah bought 75 markers. -/
theorem largest_package_size : Nat.gcd 30 (Nat.gcd 45 75) = 15 := by
  sorry

end NUMINAMATH_CALUDE_largest_package_size_l3248_324829


namespace NUMINAMATH_CALUDE_cylinder_equal_volume_increase_l3248_324830

/-- Theorem: For a cylinder with radius 6 inches and height 4 inches, 
    the value of x that satisfies the equation π(R+x)²H = πR²(H+2x) is 6 inches. -/
theorem cylinder_equal_volume_increase (π : ℝ) : 
  ∃ (x : ℝ), x = 6 ∧ π * (6 + x)^2 * 4 = π * 6^2 * (4 + 2*x) :=
by sorry

end NUMINAMATH_CALUDE_cylinder_equal_volume_increase_l3248_324830


namespace NUMINAMATH_CALUDE_students_taking_one_subject_l3248_324878

theorem students_taking_one_subject (both : ℕ) (geometry : ℕ) (history_only : ℕ)
  (h1 : both = 15)
  (h2 : geometry = 30)
  (h3 : history_only = 18) :
  geometry - both + history_only = 33 := by
sorry

end NUMINAMATH_CALUDE_students_taking_one_subject_l3248_324878


namespace NUMINAMATH_CALUDE_yoongi_has_more_points_l3248_324816

theorem yoongi_has_more_points : ∀ (yoongi_points jungkook_points : ℕ),
  yoongi_points = 4 →
  jungkook_points = 6 - 3 →
  yoongi_points > jungkook_points :=
by
  sorry

end NUMINAMATH_CALUDE_yoongi_has_more_points_l3248_324816


namespace NUMINAMATH_CALUDE_race_difference_l3248_324839

/-- In a race, given the total distance and the differences between runners,
    calculate the difference between two runners. -/
theorem race_difference (total_distance : ℕ) (a_beats_b b_beats_c a_beats_c : ℕ) :
  total_distance = 1000 →
  a_beats_b = 70 →
  a_beats_c = 163 →
  b_beats_c = 93 :=
by sorry

end NUMINAMATH_CALUDE_race_difference_l3248_324839


namespace NUMINAMATH_CALUDE_max_regions_50_lines_20_parallel_l3248_324821

/-- The maximum number of regions created by n lines in a plane -/
def max_regions (n : ℕ) : ℕ := n * (n + 1) / 2 + 1

/-- The number of additional regions created by m parallel lines intersecting n non-parallel lines -/
def parallel_regions (m n : ℕ) : ℕ := m * (n + 1)

/-- The maximum number of regions created by n lines in a plane, where m of them are parallel -/
def max_regions_with_parallel (n m : ℕ) : ℕ :=
  max_regions (n - m) + parallel_regions m (n - m)

theorem max_regions_50_lines_20_parallel :
  max_regions_with_parallel 50 20 = 1086 := by
  sorry

end NUMINAMATH_CALUDE_max_regions_50_lines_20_parallel_l3248_324821


namespace NUMINAMATH_CALUDE_sin_less_than_x_l3248_324864

theorem sin_less_than_x :
  (∀ x : ℝ, 0 < x → x < π / 2 → Real.sin x < x) ∧
  (∀ x : ℝ, x > 0 → Real.sin x < x) := by
  sorry

end NUMINAMATH_CALUDE_sin_less_than_x_l3248_324864


namespace NUMINAMATH_CALUDE_turner_tickets_l3248_324856

def rollercoaster_rides : ℕ := 3
def catapult_rides : ℕ := 2
def ferris_wheel_rides : ℕ := 1

def rollercoaster_cost : ℕ := 4
def catapult_cost : ℕ := 4
def ferris_wheel_cost : ℕ := 1

def total_tickets : ℕ := 
  rollercoaster_rides * rollercoaster_cost + 
  catapult_rides * catapult_cost + 
  ferris_wheel_rides * ferris_wheel_cost

theorem turner_tickets : total_tickets = 21 := by
  sorry

end NUMINAMATH_CALUDE_turner_tickets_l3248_324856


namespace NUMINAMATH_CALUDE_difference_twice_x_and_three_less_than_zero_l3248_324867

theorem difference_twice_x_and_three_less_than_zero (x : ℝ) :
  (2 * x - 3 < 0) ↔ (∃ y, y = 2 * x ∧ y - 3 < 0) :=
by sorry

end NUMINAMATH_CALUDE_difference_twice_x_and_three_less_than_zero_l3248_324867


namespace NUMINAMATH_CALUDE_pure_imaginary_condition_l3248_324831

/-- Given that i is the imaginary unit and (1+ai)/i is a pure imaginary number, prove that a = 0 -/
theorem pure_imaginary_condition (a : ℝ) : 
  (Complex.I : ℂ) ^ 2 = -1 →  -- i is the imaginary unit
  (↑1 + a * Complex.I) / Complex.I = b * Complex.I →  -- (1+ai)/i is a pure imaginary number
  a = 0 := by sorry

end NUMINAMATH_CALUDE_pure_imaginary_condition_l3248_324831


namespace NUMINAMATH_CALUDE_dogs_sold_l3248_324890

theorem dogs_sold (cats : ℕ) (dogs : ℕ) (ratio : ℚ) : 
  ratio = 2 / 1 → cats = 16 → dogs = 8 := by
  sorry

end NUMINAMATH_CALUDE_dogs_sold_l3248_324890


namespace NUMINAMATH_CALUDE_quadratic_sum_powers_divisibility_l3248_324886

/-- Represents a quadratic polynomial with integer coefficients -/
structure QuadraticPolynomial where
  p : ℤ
  q : ℤ

/-- Condition that the polynomial has a positive discriminant -/
def has_positive_discriminant (f : QuadraticPolynomial) : Prop :=
  f.p * f.p - 4 * f.q > 0

/-- Sum of the hundredth powers of the roots of a quadratic polynomial -/
noncomputable def sum_of_hundredth_powers (f : QuadraticPolynomial) : ℝ :=
  let α := (-f.p + Real.sqrt (f.p * f.p - 4 * f.q)) / 2
  let β := (-f.p - Real.sqrt (f.p * f.p - 4 * f.q)) / 2
  α^100 + β^100

/-- Main theorem statement -/
theorem quadratic_sum_powers_divisibility 
  (f : QuadraticPolynomial) 
  (h_disc : has_positive_discriminant f) 
  (h_p : f.p % 5 = 0) 
  (h_q : f.q % 5 = 0) : 
  ∃ (k : ℤ), sum_of_hundredth_powers f = k * (5^50 : ℝ) ∧ 
  ∀ (n : ℕ), n > 50 → ¬∃ (m : ℤ), sum_of_hundredth_powers f = m * (5^n : ℝ) :=
sorry

end NUMINAMATH_CALUDE_quadratic_sum_powers_divisibility_l3248_324886


namespace NUMINAMATH_CALUDE_exists_valid_layout_18_rectangles_l3248_324896

/-- Represents a rectangle with width and height --/
structure Rectangle where
  width : ℝ
  height : ℝ

/-- Represents a position on a 2D grid --/
structure Position where
  x : ℕ
  y : ℕ

/-- Represents a layout of rectangles on a grid --/
def Layout := Position → Option Rectangle

/-- Checks if two positions are adjacent --/
def adjacent (p1 p2 : Position) : Prop :=
  (p1.x = p2.x ∧ (p1.y + 1 = p2.y ∨ p2.y + 1 = p1.y)) ∨
  (p1.y = p2.y ∧ (p1.x + 1 = p2.x ∨ p2.x + 1 = p1.x))

/-- Checks if two rectangles form a larger rectangle when adjacent --/
def formsLargerRectangle (r1 r2 : Rectangle) : Prop :=
  r1.width = r2.width ∨ r1.height = r2.height

/-- Checks if a layout satisfies the non-adjacency condition --/
def validLayout (l : Layout) : Prop :=
  ∀ p1 p2, adjacent p1 p2 →
    match l p1, l p2 with
    | some r1, some r2 => ¬formsLargerRectangle r1 r2
    | _, _ => True

/-- The main theorem: there exists a valid layout with 18 rectangles --/
theorem exists_valid_layout_18_rectangles :
  ∃ (l : Layout) (r : Rectangle),
    validLayout l ∧
    (∃ (positions : Finset Position), positions.card = 18 ∧
      ∀ p, p ∈ positions ↔ ∃ (smallR : Rectangle), l p = some smallR) :=
sorry

end NUMINAMATH_CALUDE_exists_valid_layout_18_rectangles_l3248_324896


namespace NUMINAMATH_CALUDE_item_costs_l3248_324894

/-- The cost of items in yuan -/
structure ItemCosts where
  tableLamp : ℕ
  electricFan : ℕ
  bicycle : ℕ

/-- Theorem stating the total cost of all items and the cost of lamp and fan -/
theorem item_costs (c : ItemCosts) 
  (h1 : c.tableLamp = 86)
  (h2 : c.electricFan = 185)
  (h3 : c.bicycle = 445) :
  (c.tableLamp + c.electricFan + c.bicycle = 716) ∧
  (c.tableLamp + c.electricFan = 271) := by
  sorry

#check item_costs

end NUMINAMATH_CALUDE_item_costs_l3248_324894


namespace NUMINAMATH_CALUDE_euler_disproof_l3248_324877

theorem euler_disproof : 133^4 + 110^4 + 56^4 = 143^4 := by
  sorry

end NUMINAMATH_CALUDE_euler_disproof_l3248_324877


namespace NUMINAMATH_CALUDE_solution_in_quadrant_III_l3248_324826

/-- 
Given a system of equations x - y = 4 and cx + y = 5, where c is a constant,
this theorem states that the solution (x, y) is in Quadrant III 
(i.e., x < 0 and y < 0) if and only if c < -1.
-/
theorem solution_in_quadrant_III (c : ℝ) :
  (∃ x y : ℝ, x - y = 4 ∧ c * x + y = 5 ∧ x < 0 ∧ y < 0) ↔ c < -1 :=
by sorry

end NUMINAMATH_CALUDE_solution_in_quadrant_III_l3248_324826


namespace NUMINAMATH_CALUDE_transaction_handling_l3248_324825

/-- Problem: Transaction Handling --/
theorem transaction_handling 
  (mabel_transactions : ℕ)
  (anthony_percentage : ℚ)
  (cal_fraction : ℚ)
  (jade_additional : ℕ)
  (h1 : mabel_transactions = 90)
  (h2 : anthony_percentage = 11/10)
  (h3 : cal_fraction = 2/3)
  (h4 : jade_additional = 18) :
  let anthony_transactions := mabel_transactions * anthony_percentage
  let cal_transactions := anthony_transactions * cal_fraction
  let jade_transactions := cal_transactions + jade_additional
  jade_transactions = 84 := by
sorry

end NUMINAMATH_CALUDE_transaction_handling_l3248_324825


namespace NUMINAMATH_CALUDE_nancy_museum_pictures_l3248_324858

theorem nancy_museum_pictures :
  ∀ (zoo_pics museum_pics deleted_pics remaining_pics : ℕ),
    zoo_pics = 49 →
    deleted_pics = 38 →
    remaining_pics = 19 →
    zoo_pics + museum_pics = deleted_pics + remaining_pics →
    museum_pics = 8 :=
by sorry

end NUMINAMATH_CALUDE_nancy_museum_pictures_l3248_324858


namespace NUMINAMATH_CALUDE_base9_to_base5_conversion_l3248_324861

/-- Converts a base-9 number to its decimal (base-10) representation -/
def base9ToDecimal (n : List Nat) : Nat :=
  n.enum.foldl (fun acc (i, d) => acc + d * (9 ^ i)) 0

/-- Converts a decimal (base-10) number to its base-5 representation -/
def decimalToBase5 (n : Nat) : List Nat :=
  if n = 0 then [0] else
    let rec aux (m : Nat) (acc : List Nat) : List Nat :=
      if m = 0 then acc else aux (m / 5) ((m % 5) :: acc)
    aux n []

/-- The base-9 representation of the number to be converted -/
def number_base9 : List Nat := [4, 2, 7]

theorem base9_to_base5_conversion :
  decimalToBase5 (base9ToDecimal number_base9) = [4, 3, 2, 4] :=
sorry

end NUMINAMATH_CALUDE_base9_to_base5_conversion_l3248_324861


namespace NUMINAMATH_CALUDE_olivia_change_l3248_324838

def basketball_card_price : ℕ := 3
def baseball_card_price : ℕ := 4
def num_basketball_packs : ℕ := 2
def num_baseball_decks : ℕ := 5
def bill_value : ℕ := 50

def total_cost : ℕ := num_basketball_packs * basketball_card_price + num_baseball_decks * baseball_card_price

theorem olivia_change : bill_value - total_cost = 24 := by
  sorry

end NUMINAMATH_CALUDE_olivia_change_l3248_324838


namespace NUMINAMATH_CALUDE_inequality_solution_set_l3248_324810

theorem inequality_solution_set (x : ℝ) : 
  (2 * x^2 / (x + 2) ≥ 3 / (x - 2) + 6 / 4) ↔ 
  (x < -2 ∨ (-2 < x ∧ x < (1 - Real.sqrt 129) / 8) ∨ 
   (2 < x ∧ x < 3) ∨ 
   ((1 + Real.sqrt 129) / 8 < x)) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l3248_324810


namespace NUMINAMATH_CALUDE_decreasing_function_on_positive_reals_l3248_324879

/-- The function f(x) = -x(x+2) is decreasing on the interval (0, +∞) -/
theorem decreasing_function_on_positive_reals :
  ∀ x y : ℝ, 0 < x → 0 < y → x < y → -x * (x + 2) > -y * (y + 2) := by
  sorry

end NUMINAMATH_CALUDE_decreasing_function_on_positive_reals_l3248_324879


namespace NUMINAMATH_CALUDE_local_maximum_at_two_l3248_324823

/-- The function f(x) = x(x-c)² --/
def f (c : ℝ) (x : ℝ) : ℝ := x * (x - c)^2

/-- The derivative of f(x) --/
def f_derivative (c : ℝ) (x : ℝ) : ℝ := 3*x^2 - 4*c*x + c^2

theorem local_maximum_at_two (c : ℝ) :
  (∃ δ > 0, ∀ x ∈ Set.Ioo (2 - δ) (2 + δ), f c x ≤ f c 2) →
  (f_derivative c 2 = 0) →
  (∀ x ∈ Set.Ioo (2 - δ) 2, f_derivative c x > 0) →
  (∀ x ∈ Set.Ioo 2 (2 + δ), f_derivative c x < 0) →
  c = 6 := by
  sorry

end NUMINAMATH_CALUDE_local_maximum_at_two_l3248_324823


namespace NUMINAMATH_CALUDE_P_intersect_Q_eq_P_l3248_324824

def P : Set ℝ := {-1, 0, 1}
def Q : Set ℝ := {y | ∃ x, y = Real.cos x}

theorem P_intersect_Q_eq_P : P ∩ Q = P := by sorry

end NUMINAMATH_CALUDE_P_intersect_Q_eq_P_l3248_324824


namespace NUMINAMATH_CALUDE_halfway_between_fractions_l3248_324884

theorem halfway_between_fractions : 
  (2 : ℚ) / 9 + (1 : ℚ) / 3 = (5 : ℚ) / 9 ∧ (5 : ℚ) / 9 / 2 = (5 : ℚ) / 18 := by
  sorry

end NUMINAMATH_CALUDE_halfway_between_fractions_l3248_324884


namespace NUMINAMATH_CALUDE_peters_pizza_consumption_l3248_324802

theorem peters_pizza_consumption :
  ∀ (total_slices : ℕ) (whole_slices : ℕ) (shared_slice : ℚ),
    total_slices = 16 →
    whole_slices = 2 →
    shared_slice = 1/3 →
    (whole_slices : ℚ) / total_slices + shared_slice / total_slices = 7/48 := by
  sorry

end NUMINAMATH_CALUDE_peters_pizza_consumption_l3248_324802


namespace NUMINAMATH_CALUDE_solve_for_a_l3248_324889

theorem solve_for_a : ∀ a : ℚ, 
  (∃ x : ℚ, (2 * a * x + 3) / (a - x) = 3 / 4 ∧ x = 1) → 
  a = -3 := by
sorry

end NUMINAMATH_CALUDE_solve_for_a_l3248_324889


namespace NUMINAMATH_CALUDE_beth_coin_ratio_l3248_324860

/-- Proves that the ratio of coins Beth sold to her total coins after receiving Carl's gift is 1:2 -/
theorem beth_coin_ratio :
  let initial_coins : ℕ := 125
  let gift_coins : ℕ := 35
  let sold_coins : ℕ := 80
  let total_coins : ℕ := initial_coins + gift_coins
  (sold_coins : ℚ) / total_coins = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_beth_coin_ratio_l3248_324860


namespace NUMINAMATH_CALUDE_segments_covered_by_q_at_most_q_plus_one_l3248_324818

/-- A half-line on the real number line -/
structure HalfLine where
  endpoint : ℝ
  direction : Bool -- true for right-infinite, false for left-infinite

/-- A configuration of half-lines on the real number line -/
def Configuration := List HalfLine

/-- A segment on the real number line -/
structure Segment where
  left : ℝ
  right : ℝ

/-- Count the number of half-lines covering a given segment -/
def coverCount (config : Configuration) (seg : Segment) : ℕ :=
  sorry

/-- The segments formed by the endpoints of the half-lines -/
def segments (config : Configuration) : List Segment :=
  sorry

/-- The segments covered by exactly q half-lines -/
def segmentsCoveredByQ (config : Configuration) (q : ℕ) : List Segment :=
  sorry

/-- The main theorem -/
theorem segments_covered_by_q_at_most_q_plus_one (config : Configuration) (q : ℕ) :
  (segmentsCoveredByQ config q).length ≤ q + 1 :=
  sorry

end NUMINAMATH_CALUDE_segments_covered_by_q_at_most_q_plus_one_l3248_324818


namespace NUMINAMATH_CALUDE_problem_statement_l3248_324832

theorem problem_statement (x y z : ℝ) 
  (h_pos_x : 0 < x) (h_pos_y : 0 < y) (h_pos_z : 0 < z)
  (h_diff_xy : x ≠ y) (h_diff_xz : x ≠ z) (h_diff_yz : y ≠ z)
  (h_eq1 : (y + 1) / (x - z) = (x + y) / (z + 1))
  (h_eq2 : (y + 1) / (x - z) = x / (y + 1)) :
  x / (y + 1) = 2 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l3248_324832


namespace NUMINAMATH_CALUDE_roots_of_polynomial_l3248_324875

def p (x : ℝ) : ℝ := 3*x^4 + 16*x^3 - 36*x^2 + 8*x

theorem roots_of_polynomial :
  ∃ (a b : ℝ), a^2 = 17 ∧
  (p 0 = 0) ∧
  (p (1/3) = 0) ∧
  (p (-3 + 2*a) = 0) ∧
  (p (-3 - 2*a) = 0) ∧
  (∀ x : ℝ, p x = 0 → x = 0 ∨ x = 1/3 ∨ x = -3 + 2*a ∨ x = -3 - 2*a) :=
by sorry

end NUMINAMATH_CALUDE_roots_of_polynomial_l3248_324875


namespace NUMINAMATH_CALUDE_dannys_remaining_bottle_caps_l3248_324862

def initial_bottle_caps : ℕ := 91
def lost_bottle_caps : ℕ := 66

theorem dannys_remaining_bottle_caps :
  initial_bottle_caps - lost_bottle_caps = 25 := by
  sorry

end NUMINAMATH_CALUDE_dannys_remaining_bottle_caps_l3248_324862


namespace NUMINAMATH_CALUDE_absolute_value_eq_four_sum_of_absolute_values_min_value_of_sum_min_value_is_three_l3248_324806

-- Problem 1
theorem absolute_value_eq_four (a : ℝ) : 
  |a + 2| = 4 ↔ a = -6 ∨ a = 2 := by sorry

-- Problem 2
theorem sum_of_absolute_values (a : ℝ) :
  -4 < a ∧ a < 2 → |a + 4| + |a - 2| = 6 := by sorry

-- Problem 3
theorem min_value_of_sum (a : ℝ) :
  (∀ x : ℝ, |x - 1| + |x + 2| ≥ |a - 1| + |a + 2|) ↔ -2 ≤ a ∧ a ≤ 1 := by sorry

theorem min_value_is_three (a : ℝ) :
  -2 ≤ a ∧ a ≤ 1 → |a - 1| + |a + 2| = 3 := by sorry

end NUMINAMATH_CALUDE_absolute_value_eq_four_sum_of_absolute_values_min_value_of_sum_min_value_is_three_l3248_324806


namespace NUMINAMATH_CALUDE_geometric_sequence_properties_l3248_324865

/-- Geometric sequence with specified properties -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, q ≠ 1 ∧ ∀ n : ℕ, a (n + 1) = q * a n

theorem geometric_sequence_properties
  (a : ℕ → ℝ)
  (h_geometric : GeometricSequence a)
  (h_arithmetic_mean : 2 * a 1 = a 2 + a 3)
  (h_a1 : a 1 = 1) :
  (∃ q : ℝ, q = -2 ∧ ∀ n : ℕ, a (n + 1) = q * a n) ∧
  (∀ n : ℕ, (Finset.range n).sum (fun i => (i + 1 : ℝ) * a (i + 1)) = (1 - (1 + 3 * n) * (-2)^n) / 9) :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_properties_l3248_324865


namespace NUMINAMATH_CALUDE_expression_evaluation_l3248_324874

theorem expression_evaluation (y : ℝ) (h : y ≠ 1/2) :
  (2*y - 1)^0 / (6⁻¹ + 2⁻¹) = 3/2 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l3248_324874


namespace NUMINAMATH_CALUDE_no_solution_exists_l3248_324809

def is_digit (n : ℕ) : Prop := n ≥ 0 ∧ n ≤ 9

def number_with_B (B : ℕ) : ℕ := 12345670 + B

theorem no_solution_exists :
  ¬ ∃ B : ℕ, is_digit B ∧ 
    (number_with_B B).mod 2 = 0 ∧
    (number_with_B B).mod 5 = 0 ∧
    (number_with_B B).mod 11 = 0 :=
sorry

end NUMINAMATH_CALUDE_no_solution_exists_l3248_324809


namespace NUMINAMATH_CALUDE_work_completion_time_l3248_324895

/-- Given two workers X and Y, where X can finish a job in 21 days and Y in 15 days,
    if Y works for 5 days and then leaves, prove that X needs 14 days to finish the remaining work. -/
theorem work_completion_time (x_rate y_rate : ℚ) (y_days : ℕ) :
  x_rate = 1 / 21 →
  y_rate = 1 / 15 →
  y_days = 5 →
  (1 - y_rate * y_days) / x_rate = 14 := by
  sorry

end NUMINAMATH_CALUDE_work_completion_time_l3248_324895


namespace NUMINAMATH_CALUDE_vector_subtraction_magnitude_l3248_324835

/-- Given two 2D vectors a and b, where the angle between them is 45°,
    a = (-1, 1), and |b| = 1, prove that |a - 2b| = √2 -/
theorem vector_subtraction_magnitude (a b : ℝ × ℝ) :
  let angle := Real.pi / 4
  a.1 = -1 ∧ a.2 = 1 →
  Real.sqrt (b.1^2 + b.2^2) = 1 →
  a.1 * b.1 + a.2 * b.2 = Real.sqrt 2 / 2 →
  Real.sqrt ((a.1 - 2*b.1)^2 + (a.2 - 2*b.2)^2) = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_vector_subtraction_magnitude_l3248_324835


namespace NUMINAMATH_CALUDE_parabola_and_line_intersection_l3248_324893

-- Define the parabola and line
def parabola (p : ℝ) (x y : ℝ) : Prop := x^2 = -2*p*y
def line1 (x y : ℝ) : Prop := y = (1/2)*x - 1
def line2 (k : ℝ) (x y : ℝ) : Prop := y = k*x - 3/2

-- Define the theorem
theorem parabola_and_line_intersection 
  (p : ℝ) 
  (x_M y_M x_N y_N : ℝ) 
  (h_p : p > 0)
  (h_intersect1 : line1 x_M y_M ∧ line1 x_N y_N)
  (h_parabola1 : parabola p x_M y_M ∧ parabola p x_N y_N)
  (h_condition : (x_M + 1) * (x_N + 1) = -8)
  (k : ℝ)
  (x_A y_A x_B y_B : ℝ)
  (h_k : k ≠ 0)
  (h_intersect2 : line2 k x_A y_A ∧ line2 k x_B y_B)
  (h_parabola2 : parabola p x_A y_A ∧ parabola p x_B y_B)
  (x_A' : ℝ)
  (h_symmetric : x_A' = -x_A) :
  (∀ x y, parabola p x y ↔ x^2 = -6*y) ∧
  (∃ t : ℝ, t = (y_B - y_A) / (x_B - x_A') ∧ 
            0 = t * 0 + y_A - t * x_A' ∧
            3/2 = t * 0 + y_A - t * x_A') :=
by sorry

end NUMINAMATH_CALUDE_parabola_and_line_intersection_l3248_324893


namespace NUMINAMATH_CALUDE_complex_parts_l3248_324873

theorem complex_parts (z : ℂ) (h : z = 2 - 3*I) : 
  z.re = 2 ∧ z.im = -3 := by sorry

end NUMINAMATH_CALUDE_complex_parts_l3248_324873


namespace NUMINAMATH_CALUDE_original_price_from_decreased_price_l3248_324842

/-- Proves that if an article's price after a 24% decrease is 684, then its original price was 900. -/
theorem original_price_from_decreased_price (decreased_price : ℝ) (decrease_percentage : ℝ) :
  decreased_price = 684 ∧ decrease_percentage = 24 →
  (1 - decrease_percentage / 100) * 900 = decreased_price := by
  sorry

#check original_price_from_decreased_price

end NUMINAMATH_CALUDE_original_price_from_decreased_price_l3248_324842


namespace NUMINAMATH_CALUDE_max_value_on_ellipse_l3248_324854

theorem max_value_on_ellipse :
  ∃ (max : ℝ),
    (∀ x y : ℝ, (y^2 / 4 + x^2 / 3 = 1) → 2*x + y ≤ max) ∧
    (∃ x y : ℝ, (y^2 / 4 + x^2 / 3 = 1) ∧ 2*x + y = max) ∧
    max = 4 := by
  sorry

end NUMINAMATH_CALUDE_max_value_on_ellipse_l3248_324854


namespace NUMINAMATH_CALUDE_composite_sum_of_powers_l3248_324843

theorem composite_sum_of_powers (a b c d : ℕ) (h : a * b = c * d) :
  ∃ (x y : ℕ), x > 1 ∧ y > 1 ∧ x * y = a^2000 + b^2000 + c^2000 + d^2000 :=
by
  sorry

end NUMINAMATH_CALUDE_composite_sum_of_powers_l3248_324843


namespace NUMINAMATH_CALUDE_perpendicular_bisector_value_l3248_324803

/-- The perpendicular bisector of a line segment passing through two points. -/
structure PerpendicularBisector where
  -- The equation of the line: x + y = b
  b : ℝ
  -- The two points defining the line segment
  p1 : ℝ × ℝ := (2, 4)
  p2 : ℝ × ℝ := (6, 8)
  -- The condition that the line is a perpendicular bisector
  is_perp_bisector : b = p1.1 + p1.2 + p2.1 + p2.2

/-- The value of b for the perpendicular bisector of the line segment from (2,4) to (6,8) is 10. -/
theorem perpendicular_bisector_value : 
  ∀ (pb : PerpendicularBisector), pb.b = 10 := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_bisector_value_l3248_324803


namespace NUMINAMATH_CALUDE_tens_digit_of_2023_power_minus_2025_squared_l3248_324846

theorem tens_digit_of_2023_power_minus_2025_squared : ∃ n : ℕ, 
  2023^2024 - 2025^2 = 100 * n + 16 :=
by sorry

end NUMINAMATH_CALUDE_tens_digit_of_2023_power_minus_2025_squared_l3248_324846


namespace NUMINAMATH_CALUDE_shoes_savings_theorem_l3248_324811

/-- The number of weekends needed to save for shoes -/
def weekends_needed (shoe_cost : ℕ) (saved : ℕ) (earnings_per_lawn : ℕ) (lawns_per_weekend : ℕ) : ℕ :=
  let remaining := shoe_cost - saved
  let earnings_per_weekend := earnings_per_lawn * lawns_per_weekend
  (remaining + earnings_per_weekend - 1) / earnings_per_weekend

theorem shoes_savings_theorem (shoe_cost saved earnings_per_lawn lawns_per_weekend : ℕ) 
  (h1 : shoe_cost = 120)
  (h2 : saved = 30)
  (h3 : earnings_per_lawn = 5)
  (h4 : lawns_per_weekend = 3) :
  weekends_needed shoe_cost saved earnings_per_lawn lawns_per_weekend = 6 := by
  sorry

end NUMINAMATH_CALUDE_shoes_savings_theorem_l3248_324811


namespace NUMINAMATH_CALUDE_cylinder_max_volume_l3248_324845

/-- Given a cylinder with an axial cross-section circumference of 90 cm,
    prove that its maximum volume is 3375π cm³. -/
theorem cylinder_max_volume (d m : ℝ) (h : d + m = 45) :
  ∃ (V : ℝ), V ≤ 3375 * Real.pi ∧ ∃ (r : ℝ), V = π * r^2 * m ∧ d = 2 * r :=
sorry

end NUMINAMATH_CALUDE_cylinder_max_volume_l3248_324845


namespace NUMINAMATH_CALUDE_parametric_to_circle_equation_l3248_324898

/-- Given parametric equations for a curve and a relationship between parameters,
    prove that the resulting equation is that of a circle with specific center and radius,
    excluding two points on the x-axis. -/
theorem parametric_to_circle_equation 
  (u v : ℝ) (m : ℝ) (hm : m ≠ 0)
  (hx : ∀ u v, x = (1 - u^2 - v^2) / ((1 - u)^2 + v^2))
  (hy : ∀ u v, y = 2 * v / ((1 - u)^2 + v^2))
  (hv : v = m * u) :
  x^2 + (y - 1/m)^2 = 1 + 1/m^2 ∧ 
  (x ≠ 1 ∨ y ≠ 0) ∧ (x ≠ -1 ∨ y ≠ 0) := by
  sorry


end NUMINAMATH_CALUDE_parametric_to_circle_equation_l3248_324898


namespace NUMINAMATH_CALUDE_nancy_homework_problem_l3248_324840

theorem nancy_homework_problem (finished : ℕ) (pages_left : ℕ) (problems_per_page : ℕ) : 
  finished = 47 → pages_left = 6 → problems_per_page = 9 →
  finished + pages_left * problems_per_page = 101 := by
  sorry

end NUMINAMATH_CALUDE_nancy_homework_problem_l3248_324840


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l3248_324801

/-- Given an arithmetic sequence {aₙ} where (a₂ + a₅ = 4) and (a₆ + a₉ = 20),
    prove that (a₄ + a₇) = 12 -/
theorem arithmetic_sequence_sum (a : ℕ → ℝ) :
  (∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)) →  -- arithmetic sequence condition
  a 2 + a 5 = 4 →
  a 6 + a 9 = 20 →
  a 4 + a 7 = 12 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l3248_324801


namespace NUMINAMATH_CALUDE_existence_of_intersection_point_l3248_324848

/-- Represents a circle on a plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ
  radius_pos : radius > 0

/-- Represents a point on a plane -/
def Point : Type := ℝ × ℝ

/-- Checks if a point is outside a circle -/
def is_outside (p : Point) (c : Circle) : Prop :=
  (p.1 - c.center.1)^2 + (p.2 - c.center.2)^2 > c.radius^2

/-- Represents a line on a plane -/
structure Line where
  point : Point
  direction : ℝ × ℝ
  non_zero : direction ≠ (0, 0)

/-- Checks if a line intersects a circle -/
def intersects (l : Line) (c : Circle) : Prop :=
  ∃ t : ℝ, is_outside (l.point.1 + t * l.direction.1, l.point.2 + t * l.direction.2) c = false

/-- Main theorem: There exists a point outside both circles such that 
    any line passing through it intersects at least one of the circles -/
theorem existence_of_intersection_point (c1 c2 : Circle) 
  (h : ∀ p : Point, ¬(is_outside p c1 ∧ is_outside p c2)) : 
  ∃ p : Point, is_outside p c1 ∧ is_outside p c2 ∧ 
    ∀ l : Line, l.point = p → (intersects l c1 ∨ intersects l c2) := by
  sorry

end NUMINAMATH_CALUDE_existence_of_intersection_point_l3248_324848


namespace NUMINAMATH_CALUDE_equidistant_point_y_axis_l3248_324876

/-- The y-coordinate of the point on the y-axis that is equidistant from A(-3, -2) and B(2, 3) is 0 -/
theorem equidistant_point_y_axis : ∃ y : ℝ, 
  (y = 0) ∧ 
  ((-3 - 0)^2 + (-2 - y)^2 = (2 - 0)^2 + (3 - y)^2) := by
  sorry

end NUMINAMATH_CALUDE_equidistant_point_y_axis_l3248_324876


namespace NUMINAMATH_CALUDE_shane_sandwiches_l3248_324871

/-- The number of slices in each package of sliced ham -/
def slices_per_ham_package (
  bread_slices_per_package : ℕ)
  (num_bread_packages : ℕ)
  (num_ham_packages : ℕ)
  (leftover_bread_slices : ℕ)
  (bread_slices_per_sandwich : ℕ) : ℕ :=
  let total_bread_slices := bread_slices_per_package * num_bread_packages
  let used_bread_slices := total_bread_slices - leftover_bread_slices
  let num_sandwiches := used_bread_slices / bread_slices_per_sandwich
  num_sandwiches / num_ham_packages

theorem shane_sandwiches :
  slices_per_ham_package 20 2 2 8 2 = 8 := by
  sorry

end NUMINAMATH_CALUDE_shane_sandwiches_l3248_324871


namespace NUMINAMATH_CALUDE_min_value_of_bisecting_line_l3248_324851

/-- A line that bisects the circumference of a circle -/
structure BisectingLine where
  a : ℝ
  b : ℝ
  a_pos : a > 0
  b_pos : b > 0
  bisects : ∀ x y : ℝ, a * x + 2 * b * y - 2 = 0 → x^2 + y^2 - 4*x - 2*y - 8 = 0

/-- The minimum value of 1/a + 2/b for a bisecting line -/
theorem min_value_of_bisecting_line (l : BisectingLine) : 
  ∃ (m : ℝ), (∀ a b : ℝ, a > 0 → b > 0 → 
    (∀ x y : ℝ, a * x + 2 * b * y - 2 = 0 → x^2 + y^2 - 4*x - 2*y - 8 = 0) → 
    1/a + 2/b ≥ m) ∧ 
  1/l.a + 2/l.b = m ∧ 
  m = 3 + 2 * Real.sqrt 2 :=
sorry

end NUMINAMATH_CALUDE_min_value_of_bisecting_line_l3248_324851


namespace NUMINAMATH_CALUDE_min_perimeter_triangle_l3248_324814

theorem min_perimeter_triangle (a b x : ℕ) (ha : a = 24) (hb : b = 37) : 
  (a + b + x > a + b ∧ a + x > b ∧ b + x > a) → 
  (∀ y : ℕ, (a + b + y > a + b ∧ a + y > b ∧ b + y > a) → x ≤ y) →
  a + b + x = 75 := by sorry

end NUMINAMATH_CALUDE_min_perimeter_triangle_l3248_324814


namespace NUMINAMATH_CALUDE_min_lcm_x_z_l3248_324892

def problem (x y z : ℕ) : Prop :=
  Nat.lcm x y = 20 ∧ Nat.lcm y z = 28

theorem min_lcm_x_z (x y z : ℕ) (h : problem x y z) :
  Nat.lcm x z ≥ 35 :=
sorry

end NUMINAMATH_CALUDE_min_lcm_x_z_l3248_324892


namespace NUMINAMATH_CALUDE_emma_popsicle_production_l3248_324850

/-- Emma's popsicle production problem -/
theorem emma_popsicle_production 
  (p h : ℝ) 
  (h_positive : h > 0)
  (p_def : p = 3/2 * h) :
  p * h - (p + 2) * (h - 3) = 7/2 * h + 6 := by
  sorry

end NUMINAMATH_CALUDE_emma_popsicle_production_l3248_324850


namespace NUMINAMATH_CALUDE_intersection_when_a_is_4_range_of_a_for_subset_l3248_324853

-- Define sets A and B
def A (a : ℝ) : Set ℝ := {x | (x - 3) * (x - 3 * a - 5) < 0}
def B : Set ℝ := {x | -x^2 + 5*x + 14 > 0}

-- Part 1
theorem intersection_when_a_is_4 :
  A 4 ∩ B = {x | 3 < x ∧ x < 7} :=
sorry

-- Part 2
theorem range_of_a_for_subset :
  {a : ℝ | A a ⊆ B} = {a : ℝ | -7/3 ≤ a ∧ a ≤ 2/3} :=
sorry

end NUMINAMATH_CALUDE_intersection_when_a_is_4_range_of_a_for_subset_l3248_324853


namespace NUMINAMATH_CALUDE_aunt_gave_109_l3248_324837

/-- The amount of money Paula's aunt gave her -/
def money_from_aunt (shirt_cost shirt_count pant_cost money_left : ℕ) : ℕ :=
  shirt_cost * shirt_count + pant_cost + money_left

/-- Proof that Paula's aunt gave her $109 -/
theorem aunt_gave_109 :
  money_from_aunt 11 2 13 74 = 109 := by
  sorry

end NUMINAMATH_CALUDE_aunt_gave_109_l3248_324837


namespace NUMINAMATH_CALUDE_min_dot_product_on_locus_l3248_324883

/-- The locus of point P -/
def locus (x y : ℝ) : Prop :=
  Real.sqrt ((x - 1)^2 + y^2) - abs x = 1

/-- A line through F(1,0) with slope k -/
def line_through_F (k : ℝ) (x y : ℝ) : Prop :=
  y = k * (x - 1)

/-- Two points on the locus -/
structure LocusPoint where
  x : ℝ
  y : ℝ
  on_locus : locus x y

/-- The dot product of two vectors -/
def dot_product (x1 y1 x2 y2 : ℝ) : ℝ :=
  x1 * x2 + y1 * y2

theorem min_dot_product_on_locus :
  ∀ (k : ℝ),
  k ≠ 0 →
  ∃ (A B D E : LocusPoint),
  line_through_F k A.x A.y ∧
  line_through_F k B.x B.y ∧
  line_through_F (-1/k) D.x D.y ∧
  line_through_F (-1/k) E.x E.y →
  ∀ (AD_dot_EB : ℝ),
  AD_dot_EB = dot_product (D.x - A.x) (D.y - A.y) (B.x - E.x) (B.y - E.y) →
  AD_dot_EB ≥ 16 :=
sorry

end NUMINAMATH_CALUDE_min_dot_product_on_locus_l3248_324883


namespace NUMINAMATH_CALUDE_shaded_region_perimeter_l3248_324859

/-- The perimeter of a region consisting of two radii of length 5 and a 3/4 circular arc of a circle with radius 5 is equal to 10 + (15π/2). -/
theorem shaded_region_perimeter (r : ℝ) (h : r = 5) :
  2 * r + (3/4) * (2 * π * r) = 10 + (15 * π) / 2 :=
by sorry

end NUMINAMATH_CALUDE_shaded_region_perimeter_l3248_324859


namespace NUMINAMATH_CALUDE_dorchester_daily_pay_l3248_324833

/-- Represents Dorchester's earnings at the puppy wash -/
structure PuppyWashEarnings where
  dailyPay : ℝ
  puppyWashRate : ℝ
  puppiesWashed : ℕ
  totalEarnings : ℝ

/-- Dorchester's earnings satisfy the given conditions -/
def dorchesterEarnings : PuppyWashEarnings where
  dailyPay := 40
  puppyWashRate := 2.25
  puppiesWashed := 16
  totalEarnings := 76

/-- Theorem: Dorchester's daily pay is $40 given the conditions -/
theorem dorchester_daily_pay :
  dorchesterEarnings.dailyPay = 40 ∧
  dorchesterEarnings.totalEarnings = dorchesterEarnings.dailyPay +
    dorchesterEarnings.puppyWashRate * dorchesterEarnings.puppiesWashed :=
by sorry

end NUMINAMATH_CALUDE_dorchester_daily_pay_l3248_324833


namespace NUMINAMATH_CALUDE_inequality_proof_l3248_324822

theorem inequality_proof (x y z : ℝ) (h1 : x ≥ 0) (h2 : y ≥ 0) (h3 : z ≥ 0) 
  (h4 : x * y + y * z + z * x = 1) : 
  (1 / (x + y)) + (1 / (y + z)) + (1 / (z + x)) ≥ 5 / 2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3248_324822
