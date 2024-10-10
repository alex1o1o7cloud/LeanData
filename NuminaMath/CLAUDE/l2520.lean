import Mathlib

namespace min_value_sum_and_sqrt_l2520_252094

theorem min_value_sum_and_sqrt (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  1/a + 1/b + 2 * Real.sqrt (a * b) ≥ 4 ∧
  (1/a + 1/b + 2 * Real.sqrt (a * b) = 4 ↔ a = b) := by
  sorry

end min_value_sum_and_sqrt_l2520_252094


namespace nail_positions_symmetry_l2520_252069

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents the shape of the flag -/
structure FlagShape where
  width : ℝ
  height : ℝ
  -- Additional parameters could be added to describe the specific shape

/-- Predicate to check if a nail position allows the flag to cover the hole -/
def covers (hole : Point) (nail : Point) (flag : FlagShape) : Prop :=
  -- This would involve checking if the hole is within the bounds of the flag
  -- when placed at the nail position
  sorry

/-- The set of all valid nail positions for a given hole and flag shape -/
def validNailPositions (hole : Point) (flag : FlagShape) : Set Point :=
  {nail : Point | covers hole nail flag}

theorem nail_positions_symmetry (hole : Point) (flag : FlagShape) :
  ∃ (center : Point), ∀ (nail : Point),
    nail ∈ validNailPositions hole flag →
    ∃ (symmetricNail : Point),
      symmetricNail ∈ validNailPositions hole flag ∧
      center.x = (nail.x + symmetricNail.x) / 2 ∧
      center.y = (nail.y + symmetricNail.y) / 2 :=
  sorry

end nail_positions_symmetry_l2520_252069


namespace ellipse_properties_l2520_252027

/-- Given an ellipse with equation x²/100 + y²/36 = 1, prove that its major axis length is 20 and eccentricity is 4/5 -/
theorem ellipse_properties (x y : ℝ) :
  x^2 / 100 + y^2 / 36 = 1 →
  ∃ (a b c : ℝ),
    a = 10 ∧
    b = 6 ∧
    c^2 = a^2 - b^2 ∧
    2 * a = 20 ∧
    c / a = 4 / 5 :=
by sorry

end ellipse_properties_l2520_252027


namespace choose_four_from_twelve_l2520_252025

theorem choose_four_from_twelve : Nat.choose 12 4 = 495 := by
  sorry

end choose_four_from_twelve_l2520_252025


namespace expand_and_simplify_l2520_252060

theorem expand_and_simplify (x : ℝ) (h : x ≠ 0) :
  (3 / 7) * (7 / x^2 + 15 * x^3 - 4 * x) = 3 / x^2 + 45 * x^3 / 7 - 12 * x / 7 := by
  sorry

end expand_and_simplify_l2520_252060


namespace min_value_of_sum_of_roots_l2520_252099

theorem min_value_of_sum_of_roots (x : ℝ) : 
  Real.sqrt (x^2 + 4*x + 20) + Real.sqrt (x^2 + 2*x + 10) ≥ 5 * Real.sqrt 2 := by
  sorry

end min_value_of_sum_of_roots_l2520_252099


namespace last_three_digits_of_7_to_106_l2520_252013

theorem last_three_digits_of_7_to_106 : ∃ n : ℕ, 7^106 ≡ 321 [ZMOD 1000] :=
by sorry

end last_three_digits_of_7_to_106_l2520_252013


namespace wood_measurement_correct_l2520_252078

/-- Represents the length of the wood in feet -/
def wood_length : ℝ := sorry

/-- Represents the length of the rope in feet -/
def rope_length : ℝ := sorry

/-- The system of equations for the wood measurement problem -/
def wood_measurement_equations : Prop :=
  (rope_length - wood_length = 4.5) ∧ (wood_length - 1/2 * rope_length = 1)

/-- Theorem stating that the system of equations correctly represents the wood measurement problem -/
theorem wood_measurement_correct : wood_measurement_equations :=
sorry

end wood_measurement_correct_l2520_252078


namespace alice_number_puzzle_l2520_252096

theorem alice_number_puzzle (x : ℝ) : 
  (((x + 3) * 3 - 3) / 3 = 10) → x = 8 := by sorry

end alice_number_puzzle_l2520_252096


namespace stating_prob_no_adjacent_same_dice_rolls_l2520_252068

/-- The number of people sitting around the circular table -/
def n : ℕ := 5

/-- The number of sides on the die -/
def die_sides : ℕ := 8

/-- The probability of no two adjacent people rolling the same number -/
def prob_no_adjacent_same : ℚ := 637 / 2048

/-- 
Theorem stating that the probability of no two adjacent people
rolling the same number on an eight-sided die when 5 people
sit around a circular table is 637/2048.
-/
theorem prob_no_adjacent_same_dice_rolls 
  (h1 : n = 5)
  (h2 : die_sides = 8) :
  prob_no_adjacent_same = 637 / 2048 := by
  sorry


end stating_prob_no_adjacent_same_dice_rolls_l2520_252068


namespace sin_plus_cos_range_l2520_252052

open Real

theorem sin_plus_cos_range :
  ∃ (f : ℝ → ℝ), (∀ x, f x = sin x + cos x) ∧
  (∀ y, y ∈ Set.range f ↔ -Real.sqrt 2 ≤ y ∧ y ≤ Real.sqrt 2) := by
  sorry

end sin_plus_cos_range_l2520_252052


namespace retailer_profit_calculation_l2520_252064

theorem retailer_profit_calculation 
  (cost_price : ℝ) 
  (markup_percentage : ℝ) 
  (discount_percentage : ℝ) 
  (actual_profit_percentage : ℝ) 
  (h1 : markup_percentage = 60) 
  (h2 : discount_percentage = 25) 
  (h3 : actual_profit_percentage = 20) : 
  markup_percentage = 60 := by
sorry

end retailer_profit_calculation_l2520_252064


namespace sqrt_eight_minus_sqrt_two_equals_sqrt_two_l2520_252088

theorem sqrt_eight_minus_sqrt_two_equals_sqrt_two :
  Real.sqrt 8 - Real.sqrt 2 = Real.sqrt 2 := by
  sorry

end sqrt_eight_minus_sqrt_two_equals_sqrt_two_l2520_252088


namespace inequality_proof_l2520_252034

theorem inequality_proof (a b c d e f : ℝ) (h : b^2 ≤ a*c) :
  (a*f - c*d)^2 ≥ (a*e - b*d)*(b*f - c*e) := by
  sorry

end inequality_proof_l2520_252034


namespace vertex_distance_is_five_l2520_252029

/-- The equation of the curve -/
def curve_equation (x y : ℝ) : Prop :=
  Real.sqrt (x^2 + y^2) + |y - 1| = 5

/-- The y-coordinate of the upper vertex -/
def upper_vertex_y : ℝ := 3

/-- The y-coordinate of the lower vertex -/
def lower_vertex_y : ℝ := -2

/-- The distance between the vertices -/
def vertex_distance : ℝ := |upper_vertex_y - lower_vertex_y|

theorem vertex_distance_is_five :
  vertex_distance = 5 :=
by sorry

end vertex_distance_is_five_l2520_252029


namespace cubic_roots_problem_l2520_252012

/-- A cubic polynomial with coefficient c and constant term d -/
def cubic (c d : ℝ) (x : ℝ) : ℝ := x^3 + c*x + d

theorem cubic_roots_problem (c d : ℝ) (u v : ℝ) :
  (∃ w, cubic c d u = 0 ∧ cubic c d v = 0 ∧ cubic c d w = 0) ∧
  (∃ w', cubic c (d + 300) (u + 5) = 0 ∧ cubic c (d + 300) (v - 4) = 0 ∧ cubic c (d + 300) w' = 0) →
  d = -616 ∨ d = 1575 := by
sorry

end cubic_roots_problem_l2520_252012


namespace quadratic_root_problem_l2520_252036

theorem quadratic_root_problem (m n k : ℝ) : 
  (m^2 + 2*m + k = 0) → 
  (n^2 + 2*n + k = 0) → 
  (1/m + 1/n = 6) → 
  (k = -1/3) := by
sorry

end quadratic_root_problem_l2520_252036


namespace cube_root_5488000_l2520_252030

theorem cube_root_5488000 :
  let n : ℝ := 5488000
  ∀ (x : ℝ), x^3 = n → x = 140 * Real.rpow 2 (1/3) := by
  sorry

end cube_root_5488000_l2520_252030


namespace beef_weight_before_processing_l2520_252055

theorem beef_weight_before_processing (weight_after : ℝ) (percentage_lost : ℝ) 
  (h1 : weight_after = 546)
  (h2 : percentage_lost = 35) : 
  weight_after / (1 - percentage_lost / 100) = 840 := by
  sorry

end beef_weight_before_processing_l2520_252055


namespace power_sum_difference_l2520_252031

theorem power_sum_difference : 2^(1+2+3) - (2^1 + 2^2 + 2^3) = 50 := by
  sorry

end power_sum_difference_l2520_252031


namespace extremum_point_of_f_l2520_252035

def f (x : ℝ) := x^2 + 1

theorem extremum_point_of_f :
  ∃ (x : ℝ), ∀ (y : ℝ), f y ≥ f x :=
by
  -- The proof would go here
  sorry

end extremum_point_of_f_l2520_252035


namespace equation_solution_l2520_252015

theorem equation_solution :
  let f (x : ℝ) := x + 2 - 4 / (x - 3)
  ∃ (x₁ x₂ : ℝ), x₁ ≠ 3 ∧ x₂ ≠ 3 ∧
    x₁ = (1 + Real.sqrt 41) / 2 ∧
    x₂ = (1 - Real.sqrt 41) / 2 ∧
    f x₁ = 0 ∧ f x₂ = 0 ∧
    ∀ (x : ℝ), x ≠ 3 → f x = 0 → (x = x₁ ∨ x = x₂) := by
  sorry

end equation_solution_l2520_252015


namespace solve_for_x_l2520_252074

theorem solve_for_x (x y : ℝ) (h1 : x + 2*y = 100) (h2 : y = 25) : x = 50 := by
  sorry

end solve_for_x_l2520_252074


namespace man_crossing_bridge_l2520_252075

/-- Proves that a man walking at 6 km/hr will take 15 minutes to cross a bridge of 1500 meters in length. -/
theorem man_crossing_bridge (walking_speed : Real) (bridge_length : Real) (crossing_time : Real) : 
  walking_speed = 6 → bridge_length = 1500 → crossing_time = 15 → 
  crossing_time * (walking_speed * 1000 / 60) = bridge_length := by
  sorry

#check man_crossing_bridge

end man_crossing_bridge_l2520_252075


namespace santa_gift_combinations_l2520_252071

theorem santa_gift_combinations (n : ℤ) : 30 ∣ (n^5 - n) := by
  sorry

end santa_gift_combinations_l2520_252071


namespace sqrt_difference_inequality_l2520_252002

theorem sqrt_difference_inequality : 
  let a := Real.sqrt 2023 - Real.sqrt 2022
  let b := Real.sqrt 2022 - Real.sqrt 2021
  let c := Real.sqrt 2021 - Real.sqrt 2020
  c > b ∧ b > a := by sorry

end sqrt_difference_inequality_l2520_252002


namespace sum_series_result_l2520_252070

def double_factorial : ℕ → ℕ
  | 0 => 1
  | 1 => 1
  | n + 2 => (n + 2) * double_factorial n

def sum_series (n : ℕ) : ℚ :=
  (Finset.range n).sum (λ i => (↑(double_factorial (2*i+1)) / ↑(double_factorial (2*i+2)) + 1 / 2^(i+1)))

theorem sum_series_result : 
  ∃ (a b : ℕ), b % 2 = 1 ∧ 
    (∃ (num : ℕ), sum_series 2023 = num / (2^a * b : ℚ)) ∧
    a * b / 10 = 4039 / 10 := by
  sorry

end sum_series_result_l2520_252070


namespace two_enchiladas_five_tacos_cost_l2520_252086

/-- The price of an enchilada in dollars -/
def enchilada_price : ℝ := sorry

/-- The price of a taco in dollars -/
def taco_price : ℝ := sorry

/-- The condition that one enchilada and four tacos cost $3.50 -/
axiom condition1 : enchilada_price + 4 * taco_price = 3.50

/-- The condition that four enchiladas and one taco cost $4.20 -/
axiom condition2 : 4 * enchilada_price + taco_price = 4.20

/-- The theorem stating that two enchiladas and five tacos cost $5.04 -/
theorem two_enchiladas_five_tacos_cost : 
  2 * enchilada_price + 5 * taco_price = 5.04 := by sorry

end two_enchiladas_five_tacos_cost_l2520_252086


namespace ellipse_and_line_properties_l2520_252040

/-- An ellipse with given properties -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_ab : a > b ∧ b > 0
  h_point : (1^2 / a^2) + ((3/2)^2 / b^2) = 1
  h_ecc : (a^2 - b^2).sqrt / a = 1/2

/-- A line intersecting the ellipse -/
structure IntersectingLine (e : Ellipse) where
  k : ℝ
  m : ℝ
  h_k : k ≠ 0
  h_intersect : ∃ (x₁ y₁ x₂ y₂ : ℝ), 
    x₁ ≠ x₂ ∧
    (x₁^2 / 4) + (y₁^2 / 3) = 1 ∧
    (x₂^2 / 4) + (y₂^2 / 3) = 1 ∧
    y₁ = k * x₁ + m ∧
    y₂ = k * x₂ + m

/-- The main theorem -/
theorem ellipse_and_line_properties (e : Ellipse) (l : IntersectingLine e) :
  (∀ (x y : ℝ), (x^2 / 4) + (y^2 / 3) = 1 ↔ (x^2 / e.a^2) + (y^2 / e.b^2) = 1) ∧
  (∃ (x₁ y₁ x₂ y₂ : ℝ), 
    x₁ ≠ x₂ ∧
    (x₁^2 / 4) + (y₁^2 / 3) = 1 ∧
    (x₂^2 / 4) + (y₂^2 / 3) = 1 ∧
    y₁ = l.k * x₁ + l.m ∧
    y₂ = l.k * x₂ + l.m ∧
    (-(x₂ - x₁) * ((y₁ + y₂)/2 - 0) = (y₂ - y₁) * ((x₁ + x₂)/2 - 1/8)) →
    l.k < -Real.sqrt 5 / 10 ∨ l.k > Real.sqrt 5 / 10) := by
  sorry

end ellipse_and_line_properties_l2520_252040


namespace lottery_not_guaranteed_win_l2520_252056

/-- Represents the probability of not winning any ticket when buying n tickets -/
def prob_no_win (total : ℕ) (rate : ℚ) (n : ℕ) : ℚ :=
  (1 - rate) ^ n

theorem lottery_not_guaranteed_win (total : ℕ) (rate : ℚ) (n : ℕ) 
  (h_total : total = 100000)
  (h_rate : rate = 1 / 1000)
  (h_n : n = 2000) :
  prob_no_win total rate n > 0 := by
  sorry

#check lottery_not_guaranteed_win

end lottery_not_guaranteed_win_l2520_252056


namespace f_prime_zero_l2520_252000

theorem f_prime_zero (f : ℝ → ℝ) (h : Differentiable ℝ f) 
  (h1 : ∀ x, f x = x^2 + 2 * (deriv f 2) * x + 3) : 
  deriv f 0 = -8 := by sorry

end f_prime_zero_l2520_252000


namespace max_value_sum_ratios_l2520_252028

theorem max_value_sum_ratios (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (h1 : a ≤ b) (h2 : b ≤ c) (h3 : c ≤ 2*a) :
  b/a + c/b + a/c ≤ 7/2 :=
by sorry

end max_value_sum_ratios_l2520_252028


namespace original_pencils_count_l2520_252007

/-- The number of pencils originally in the drawer -/
def original_pencils : ℕ := sorry

/-- The number of pencils Joan added to the drawer -/
def added_pencils : ℕ := 27

/-- The total number of pencils after Joan's addition -/
def total_pencils : ℕ := 60

/-- Theorem stating that the original number of pencils was 33 -/
theorem original_pencils_count : original_pencils = 33 :=
by
  sorry

#check original_pencils_count

end original_pencils_count_l2520_252007


namespace min_disks_for_jamal_files_l2520_252091

/-- Represents the minimum number of disks needed to store files --/
def min_disks (total_files : ℕ) (disk_capacity : ℚ) 
  (file_size_a : ℚ) (count_a : ℕ) 
  (file_size_b : ℚ) (count_b : ℕ) 
  (file_size_c : ℚ) : ℕ :=
  sorry

theorem min_disks_for_jamal_files : 
  min_disks 35 2 0.95 5 0.85 15 0.5 = 14 := by sorry

end min_disks_for_jamal_files_l2520_252091


namespace largest_four_digit_perfect_square_l2520_252073

theorem largest_four_digit_perfect_square : 
  ∀ n : ℕ, 1000 ≤ n ∧ n ≤ 9999 → (∃ m : ℕ, n = m^2) → n ≤ 9261 :=
by sorry

end largest_four_digit_perfect_square_l2520_252073


namespace parabola_kite_sum_l2520_252085

/-- Given two parabolas y = ax^2 + 4 and y = 6 - bx^2 that intersect the coordinate axes
    in exactly four points forming a kite with area 18, prove that a + b = 4/45 -/
theorem parabola_kite_sum (a b : ℝ) : 
  (∃ x₁ x₂ y₁ y₂ : ℝ, 
    -- First parabola intersects x-axis
    (a * x₁^2 + 4 = 0 ∧ a * x₂^2 + 4 = 0 ∧ x₁ ≠ x₂) ∧ 
    -- Second parabola intersects x-axis
    (6 - b * x₁^2 = 0 ∧ 6 - b * x₂^2 = 0 ∧ x₁ ≠ x₂) ∧
    -- First parabola intersects y-axis
    (a * 0^2 + 4 = y₁) ∧
    -- Second parabola intersects y-axis
    (6 - b * 0^2 = y₂) ∧
    -- Area of the kite formed by these points is 18
    (1/2 * (x₂ - x₁) * (y₂ - y₁) = 18)) →
  a + b = 4/45 := by
sorry

end parabola_kite_sum_l2520_252085


namespace intersection_distance_l2520_252092

-- Define the quadratic function
def f (x : ℝ) : ℝ := 3 * x^2 + 2 * x - 5

-- Define the horizontal line
def g (x : ℝ) : ℝ := 2

-- Define the intersection points
def intersection_points : Set ℝ := {x : ℝ | f x = g x}

-- Theorem statement
theorem intersection_distance :
  ∃ (x₁ x₂ : ℝ), x₁ ∈ intersection_points ∧ x₂ ∈ intersection_points ∧ x₁ ≠ x₂ ∧
  |x₁ - x₂| = 2 * Real.sqrt 22 / 3 :=
sorry

end intersection_distance_l2520_252092


namespace imaginary_part_of_z_l2520_252066

theorem imaginary_part_of_z (z : ℂ) (h : (2 - Complex.I) * z = 5) : 
  Complex.im z = 1 := by
  sorry

end imaginary_part_of_z_l2520_252066


namespace integers_abs_lt_3_l2520_252017

theorem integers_abs_lt_3 : 
  {n : ℤ | |n| < 3} = {-2, -1, 0, 1, 2} := by sorry

end integers_abs_lt_3_l2520_252017


namespace triangle_pcd_area_l2520_252072

/-- Given points P(0, 18), D(3, 18), and C(0, q) in a Cartesian coordinate system,
    where PD and PC are perpendicular sides of triangle PCD,
    prove that the area of triangle PCD is equal to 27 - (3/2)q. -/
theorem triangle_pcd_area (q : ℝ) : 
  let P : ℝ × ℝ := (0, 18)
  let D : ℝ × ℝ := (3, 18)
  let C : ℝ × ℝ := (0, q)
  -- PD and PC are perpendicular
  (D.1 - P.1) * (C.2 - P.2) = 0 →
  -- Area of triangle PCD
  (1/2) * (D.1 - P.1) * (P.2 - C.2) = 27 - (3/2) * q := by
  sorry

end triangle_pcd_area_l2520_252072


namespace set_A_equals_one_two_l2520_252044

def A : Set ℕ := {x | x^2 - 3*x < 0 ∧ x > 0}

theorem set_A_equals_one_two : A = {1, 2} := by sorry

end set_A_equals_one_two_l2520_252044


namespace father_twice_son_age_l2520_252087

/-- Represents the age difference between father and son when the father's age becomes more than twice the son's age -/
def AgeDifference : ℕ → Prop :=
  λ x => ∃ (y : ℕ), (27 + x = 2 * (((27 - 3) / 3) + x) + y) ∧ y > 0

/-- Theorem stating that it takes 11 years for the father's age to be more than twice the son's age -/
theorem father_twice_son_age : AgeDifference 11 := by
  sorry

end father_twice_son_age_l2520_252087


namespace certain_number_problem_l2520_252082

theorem certain_number_problem (x : ℝ) : 
  (x + 40 + 60) / 3 = (10 + 70 + 16) / 3 + 8 → x = 20 := by
  sorry

end certain_number_problem_l2520_252082


namespace halloween_candy_weight_l2520_252058

/-- The combined weight of candy Frank and Gwen received for Halloween -/
def combined_candy_weight (frank_candy : ℕ) (gwen_candy : ℕ) : ℕ :=
  frank_candy + gwen_candy

/-- Theorem: The combined weight of candy Frank and Gwen received is 17 pounds -/
theorem halloween_candy_weight :
  combined_candy_weight 10 7 = 17 := by
  sorry

end halloween_candy_weight_l2520_252058


namespace a_squared_gt_b_squared_necessary_not_sufficient_l2520_252062

theorem a_squared_gt_b_squared_necessary_not_sufficient (a b : ℝ) :
  (∀ a b : ℝ, a^3 > b^3 ∧ b^3 > 0 → a^2 > b^2) ∧
  (∃ a b : ℝ, a^2 > b^2 ∧ ¬(a^3 > b^3 ∧ b^3 > 0)) :=
by sorry

end a_squared_gt_b_squared_necessary_not_sufficient_l2520_252062


namespace candies_sum_l2520_252061

/-- The number of candies Linda has -/
def linda_candies : ℕ := 34

/-- The number of candies Chloe has -/
def chloe_candies : ℕ := 28

/-- The total number of candies Linda and Chloe have together -/
def total_candies : ℕ := linda_candies + chloe_candies

theorem candies_sum : total_candies = 62 := by
  sorry

end candies_sum_l2520_252061


namespace quadratic_two_distinct_roots_l2520_252080

theorem quadratic_two_distinct_roots (k : ℝ) : 
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ x₁^2 - 3*k*x₁ - 2 = 0 ∧ x₂^2 - 3*k*x₂ - 2 = 0 :=
by sorry

end quadratic_two_distinct_roots_l2520_252080


namespace arithmetic_mean_of_fractions_l2520_252090

theorem arithmetic_mean_of_fractions :
  (1 / 2 : ℚ) * ((2 / 5 : ℚ) + (4 / 7 : ℚ)) = (17 / 35 : ℚ) := by
  sorry

end arithmetic_mean_of_fractions_l2520_252090


namespace abs_negative_2023_l2520_252059

theorem abs_negative_2023 : |(-2023 : ℤ)| = 2023 := by
  sorry

end abs_negative_2023_l2520_252059


namespace chinese_and_math_books_same_student_probability_l2520_252053

def num_books : ℕ := 4
def num_students : ℕ := 2

def has_chinese_book : Bool := true
def has_math_book : Bool := true

def books_per_student : ℕ := num_books / num_students

theorem chinese_and_math_books_same_student_probability :
  let total_distributions := (num_books.choose books_per_student)
  let favorable_distributions := 2  -- Number of ways Chinese and Math books can be together
  (favorable_distributions : ℚ) / total_distributions = 1 / 3 := by
  sorry

end chinese_and_math_books_same_student_probability_l2520_252053


namespace sum_division_problem_l2520_252043

theorem sum_division_problem (share_ratio_a share_ratio_b share_ratio_c : ℕ) 
  (second_person_share : ℚ) (total_amount : ℚ) : 
  share_ratio_a = 100 → 
  share_ratio_b = 45 → 
  share_ratio_c = 30 → 
  second_person_share = 63 → 
  total_amount = (second_person_share / share_ratio_b) * (share_ratio_a + share_ratio_b + share_ratio_c) → 
  total_amount = 245 := by
sorry

end sum_division_problem_l2520_252043


namespace triangle_area_l2520_252098

theorem triangle_area (A B C : ℝ) (a b c : ℝ) :
  B = π / 3 →
  a = 2 →
  b = Real.sqrt 3 →
  (1 / 2) * a * b * Real.sin B = Real.sqrt 3 / 2 := by
  sorry

end triangle_area_l2520_252098


namespace function_inequality_equivalence_l2520_252016

-- Define the functions f and g with domain and range ℝ
variable (f g : ℝ → ℝ)

-- State the theorem
theorem function_inequality_equivalence :
  (∀ x, f x > g x) ↔ (∀ x, x ∉ {x | f x ≤ g x}) :=
sorry

end function_inequality_equivalence_l2520_252016


namespace circle_motion_problem_l2520_252001

/-- Given a circle with two points A and B moving along its circumference, 
    this theorem proves the speeds of the points and the circumference of the circle. -/
theorem circle_motion_problem 
  (smaller_arc : ℝ) 
  (smaller_arc_time : ℝ) 
  (larger_arc_time : ℝ) 
  (b_distance : ℝ) 
  (h1 : smaller_arc = 150)
  (h2 : smaller_arc_time = 10)
  (h3 : larger_arc_time = 14)
  (h4 : b_distance = 90) :
  ∃ (va vb l : ℝ),
    va = 12 ∧ 
    vb = 3 ∧ 
    l = 360 ∧
    smaller_arc_time * (va + vb) = smaller_arc ∧
    larger_arc_time * (va + vb) = l - smaller_arc ∧
    l / va = b_distance / vb :=
by sorry

end circle_motion_problem_l2520_252001


namespace arithmetic_sequence_a10_l2520_252084

/-- An arithmetic sequence {a_n} -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_a10
  (a : ℕ → ℝ)
  (h_arithmetic : arithmetic_sequence a)
  (h_sum : a 6 + a 8 = 16)
  (h_a4 : a 4 = 1) :
  a 10 = 15 := by
sorry

end arithmetic_sequence_a10_l2520_252084


namespace hex_count_and_sum_l2520_252005

/-- Converts a positive integer to its hexadecimal representation --/
def toHex (n : ℕ+) : List (Fin 16) := sorry

/-- Checks if a hexadecimal representation uses only digits 0-9 --/
def usesOnlyDigits (hex : List (Fin 16)) : Prop := sorry

/-- Counts numbers in [1, n] whose hexadecimal representation uses only digits 0-9 --/
def countOnlyDigits (n : ℕ+) : ℕ := sorry

/-- Computes the sum of digits of a natural number --/
def sumOfDigits (n : ℕ) : ℕ := sorry

theorem hex_count_and_sum :
  let count := countOnlyDigits 500
  count = 199 ∧ sumOfDigits count = 19 := by sorry

end hex_count_and_sum_l2520_252005


namespace inequality_proof_l2520_252046

theorem inequality_proof (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h_sum : x^2 + y^2 + z^2 = 1) :
  x / (1 - x^2) + y / (1 - y^2) + z / (1 - z^2) ≥ (3 / 2) * Real.sqrt 3 :=
by sorry

end inequality_proof_l2520_252046


namespace H_constant_l2520_252032

def H (x : ℝ) : ℝ := |x + 2| - |x - 3|

theorem H_constant : ∀ x : ℝ, H x = 5 := by sorry

end H_constant_l2520_252032


namespace problem_statement_l2520_252039

theorem problem_statement (a b c : ℝ) (h1 : a < 0) (h2 : a < b) (h3 : b < 0) (h4 : 0 < c) :
  (a * b > a * c) ∧ (a * b > b * c) ∧ (a + c < b + c) := by
  sorry

end problem_statement_l2520_252039


namespace four_points_with_given_distances_l2520_252038

theorem four_points_with_given_distances : 
  ∃! (points : Finset (ℝ × ℝ)), 
    Finset.card points = 4 ∧ 
    (∀ p ∈ points, 
      (abs p.2 = 2 ∧ abs p.1 = 4)) ∧
    (∀ p : ℝ × ℝ, 
      (abs p.2 = 2 ∧ abs p.1 = 4) → p ∈ points) :=
by sorry

end four_points_with_given_distances_l2520_252038


namespace average_income_l2520_252063

/-- Given the average monthly incomes of pairs of individuals and the income of one individual,
    prove the average monthly income of a specific pair. -/
theorem average_income (P Q R : ℕ) : 
  (P + Q) / 2 = 5050 →
  (Q + R) / 2 = 6250 →
  P = 4000 →
  (P + R) / 2 = 5200 := by
  sorry

end average_income_l2520_252063


namespace simplify_expression_l2520_252047

theorem simplify_expression (x y : ℝ) :
  let P := 2 * x + 3 * y
  let Q := x - 2 * y
  (P + Q) / (P - Q) - (P - Q) / (P + Q) = (8 * x^2 - 4 * x * y - 24 * y^2) / (3 * x^2 + 16 * x * y + 5 * y^2) :=
by sorry

end simplify_expression_l2520_252047


namespace age_difference_l2520_252019

/-- Given a man and his son, prove that the man is 35 years older than his son. -/
theorem age_difference (man_age son_age : ℕ) : 
  man_age > son_age →
  son_age = 33 →
  man_age + 2 = 2 * (son_age + 2) →
  man_age - son_age = 35 := by
  sorry


end age_difference_l2520_252019


namespace derivative_f_at_specific_point_l2520_252011

-- Define the function f
def f (x : ℝ) : ℝ := x^2008

-- State the theorem
theorem derivative_f_at_specific_point :
  deriv f ((1 / 2008 : ℝ)^(1 / 2007)) = 1 := by sorry

end derivative_f_at_specific_point_l2520_252011


namespace rhombus_tiling_2n_gon_rhombus_tiling_2002_gon_l2520_252004

/-- The number of rhombuses needed to tile a regular 2n-gon -/
def num_rhombuses (n : ℕ) : ℕ := n * (n - 1) / 2

/-- Theorem stating the number of rhombuses in a tiling of a regular 2n-gon -/
theorem rhombus_tiling_2n_gon (n : ℕ) (h : n > 1) :
  num_rhombuses n = n * (n - 1) / 2 :=
by sorry

/-- Corollary for the specific case of a 2002-gon -/
theorem rhombus_tiling_2002_gon :
  num_rhombuses 1001 = 500500 :=
by sorry

end rhombus_tiling_2n_gon_rhombus_tiling_2002_gon_l2520_252004


namespace cube_side_area_l2520_252042

theorem cube_side_area (edge_sum : ℝ) (h : edge_sum = 132) : 
  let edge_length := edge_sum / 12
  (edge_length ^ 2) = 121 := by sorry

end cube_side_area_l2520_252042


namespace rectangle_area_l2520_252003

/-- Given a rectangle with perimeter 40 feet and length-to-width ratio 3:2, its area is 96 square feet -/
theorem rectangle_area (length width : ℝ) : 
  (2 * (length + width) = 40) →  -- perimeter condition
  (length = 3/2 * width) →       -- ratio condition
  (length * width = 96) :=        -- area is 96 square feet
by
  sorry

end rectangle_area_l2520_252003


namespace age_difference_value_l2520_252014

/-- Represents the ages of three individuals and their relationships -/
structure AgeRelationship where
  /-- Age of Ramesh -/
  x : ℚ
  /-- Age of Suresh -/
  y : ℚ
  /-- Ratio of Ramesh's age to Suresh's age is 2:y -/
  age_ratio : 2 * x = y
  /-- 20 years later, ratio of Ramesh's age to Suresh's age is 8:3 -/
  future_ratio : (5 * x + 20) / (y + 20) = 8 / 3

/-- The difference between Mahesh's and Suresh's present ages -/
def age_difference (ar : AgeRelationship) : ℚ :=
  5 * ar.x - ar.y

/-- Theorem stating the difference between Mahesh's and Suresh's present ages -/
theorem age_difference_value (ar : AgeRelationship) :
  age_difference ar = 125 / 8 := by
  sorry

end age_difference_value_l2520_252014


namespace portias_university_students_l2520_252079

theorem portias_university_students :
  ∀ (p l c : ℕ),
  p = 4 * l →
  c = l / 2 →
  p + l + c = 4500 →
  p = 3273 :=
by
  sorry

end portias_university_students_l2520_252079


namespace unique_triple_l2520_252081

theorem unique_triple : ∃! (x y z : ℕ+), 
  (z > 1) ∧ 
  ((y + 1 : ℕ) % x = 0) ∧ 
  ((z - 1 : ℕ) % y = 0) ∧ 
  ((x^2 + 1 : ℕ) % z = 0) ∧
  x = 1 ∧ y = 1 ∧ z = 2 := by
sorry

end unique_triple_l2520_252081


namespace tan_alpha_implies_sin_2alpha_plus_pi_half_l2520_252093

theorem tan_alpha_implies_sin_2alpha_plus_pi_half (α : Real) 
  (h : Real.tan α = -Real.cos α / (3 + Real.sin α)) : 
  Real.sin (2 * α + π / 2) = 7 / 9 := by
  sorry

end tan_alpha_implies_sin_2alpha_plus_pi_half_l2520_252093


namespace fifteen_factorial_base16_zeros_l2520_252024

-- Define factorial function
def factorial (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

-- Define a function to count trailing zeros in base 16
def trailingZerosBase16 (n : ℕ) : ℕ :=
  -- Implementation details omitted
  sorry

-- Theorem statement
theorem fifteen_factorial_base16_zeros :
  trailingZerosBase16 (factorial 15) = 4 := by
  sorry

end fifteen_factorial_base16_zeros_l2520_252024


namespace parallelogram_area_32_18_l2520_252020

/-- The area of a parallelogram with given base and height -/
def parallelogram_area (base height : ℝ) : ℝ := base * height

/-- Theorem: The area of a parallelogram with base 32 cm and height 18 cm is 576 square centimeters -/
theorem parallelogram_area_32_18 : parallelogram_area 32 18 = 576 := by
  sorry

end parallelogram_area_32_18_l2520_252020


namespace arithmetic_sequence_fifth_term_l2520_252051

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_fifth_term 
  (a : ℕ → ℝ) 
  (h_arith : arithmetic_sequence a) 
  (h_sum : a 1 + a 9 = 10) : 
  a 5 = 5 := by
sorry

end arithmetic_sequence_fifth_term_l2520_252051


namespace ellipse_and_line_theorem_l2520_252006

-- Define the ellipse E
def ellipse (a b : ℝ) (x y : ℝ) : Prop :=
  x^2 / a^2 + y^2 / b^2 = 1

-- Define the line that B and F lie on
def line_BF (x y : ℝ) : Prop :=
  x - y + 2 = 0

-- Define the midpoint condition
def is_midpoint (x₁ y₁ x₂ y₂ xₘ yₘ : ℝ) : Prop :=
  xₘ = (x₁ + x₂) / 2 ∧ yₘ = (y₁ + y₂) / 2

theorem ellipse_and_line_theorem :
  ∀ a b : ℝ,
  a > b ∧ b > 0 →
  (∃ xB yB xF yF : ℝ,
    ellipse a b xB yB ∧
    ellipse a b xF yF ∧
    line_BF xB yB ∧
    line_BF xF yF ∧
    yB > yF) →
  (∀ x y : ℝ, ellipse a b x y ↔ x^2 / 8 + y^2 / 4 = 1) ∧
  (∀ x₁ y₁ x₂ y₂ : ℝ,
    ellipse a b x₁ y₁ ∧
    ellipse a b x₂ y₂ ∧
    is_midpoint x₁ y₁ x₂ y₂ (-1) 1 →
    x₁ - 2*y₁ + 3 = 0 ∧
    x₂ - 2*y₂ + 3 = 0) :=
by sorry

end ellipse_and_line_theorem_l2520_252006


namespace reaction_result_l2520_252050

-- Define the chemical equation
structure ChemicalEquation where
  nh4cl : ℕ
  naoh : ℕ
  nh3 : ℕ
  h2o : ℕ
  nacl : ℕ

-- Define the initial reactants
def initial_reactants : ChemicalEquation :=
  { nh4cl := 2, naoh := 3, nh3 := 0, h2o := 0, nacl := 0 }

-- Define the balanced equation coefficients
def balanced_equation : ChemicalEquation :=
  { nh4cl := 1, naoh := 1, nh3 := 1, h2o := 1, nacl := 1 }

-- Define the reaction function
def react (reactants : ChemicalEquation) : ChemicalEquation :=
  let limiting_reactant := min reactants.nh4cl reactants.naoh
  { nh4cl := reactants.nh4cl - limiting_reactant,
    naoh := reactants.naoh - limiting_reactant,
    nh3 := limiting_reactant,
    h2o := limiting_reactant,
    nacl := limiting_reactant }

-- Theorem statement
theorem reaction_result :
  let result := react initial_reactants
  result.h2o = 2 ∧ result.nh3 = 2 ∧ result.nacl = 2 ∧ result.naoh = 1 :=
sorry

end reaction_result_l2520_252050


namespace drum_capacity_ratio_l2520_252021

/-- Given two drums X and Y with oil, prove that the ratio of Y's capacity to X's capacity is 2 -/
theorem drum_capacity_ratio (C_X C_Y : ℝ) : 
  C_X > 0 → C_Y > 0 → 
  (1/2 : ℝ) * C_X + (1/5 : ℝ) * C_Y = (0.45 : ℝ) * C_Y → 
  C_Y / C_X = 2 := by
sorry

end drum_capacity_ratio_l2520_252021


namespace work_completion_time_l2520_252049

/-- Given that:
  * A can finish a work in 10 days
  * When A and B work together, A's share of the work is 3/5
Prove that B can finish the work alone in 15 days -/
theorem work_completion_time (a_time : ℝ) (a_share : ℝ) (b_time : ℝ) : 
  a_time = 10 →
  a_share = 3/5 →
  b_time = (a_time * a_share) / (1 - a_share) →
  b_time = 15 := by
sorry

end work_completion_time_l2520_252049


namespace two_different_books_count_l2520_252041

/-- The number of ways to select 2 books from different subjects -/
def selectTwoDifferentBooks (chinese : ℕ) (math : ℕ) (english : ℕ) : ℕ :=
  chinese * math + chinese * english + math * english

/-- Theorem: Given 9 Chinese books, 7 math books, and 5 English books,
    there are 143 ways to select 2 books from different subjects -/
theorem two_different_books_count :
  selectTwoDifferentBooks 9 7 5 = 143 := by
  sorry

end two_different_books_count_l2520_252041


namespace greenfield_high_lockers_l2520_252009

/-- The cost in cents for each plastic digit used in labeling lockers -/
def digit_cost : ℚ := 3

/-- The total cost in dollars for labeling all lockers -/
def total_cost : ℚ := 273.39

/-- Calculates the cost of labeling lockers from 1 to n -/
def labeling_cost (n : ℕ) : ℚ :=
  let ones := min n 9
  let tens := min (n - 9) 90
  let hundreds := min (n - 99) 900
  let thousands := max (n - 999) 0
  (ones * digit_cost + 
   tens * 2 * digit_cost + 
   hundreds * 3 * digit_cost + 
   thousands * 4 * digit_cost) / 100

/-- The number of lockers at Greenfield High -/
def num_lockers : ℕ := 2555

theorem greenfield_high_lockers : 
  labeling_cost num_lockers = total_cost :=
sorry

end greenfield_high_lockers_l2520_252009


namespace mary_uber_time_l2520_252057

/-- Mary's business trip timeline --/
def business_trip_timeline (t : ℕ) : Prop :=
  let uber_to_house := t
  let uber_to_airport := 5 * t
  let check_bag := 15
  let security := 3 * 15
  let wait_boarding := 20
  let wait_takeoff := 2 * 20
  uber_to_house + uber_to_airport + check_bag + security + wait_boarding + wait_takeoff = 180

theorem mary_uber_time : ∃ t : ℕ, business_trip_timeline t ∧ t = 10 := by
  sorry

end mary_uber_time_l2520_252057


namespace greatest_divisor_remainder_l2520_252045

theorem greatest_divisor_remainder (G R1 : ℕ) : 
  G = 29 →
  1490 % G = 11 →
  (∀ d : ℕ, d > G → (1255 % d ≠ 0 ∨ 1490 % d ≠ 0)) →
  1255 % G = R1 →
  R1 = 8 := by
sorry

end greatest_divisor_remainder_l2520_252045


namespace math_competition_non_participants_l2520_252097

theorem math_competition_non_participants (total_students : ℕ) 
  (participation_ratio : ℚ) (h1 : total_students = 89) 
  (h2 : participation_ratio = 3/5) : 
  total_students - (participation_ratio * total_students).floor = 35 := by
  sorry

end math_competition_non_participants_l2520_252097


namespace count_valid_arrangements_l2520_252026

/-- The number of ways to arrange the digits 1, 1, 2, 5, 0 into a five-digit multiple of 5 -/
def arrangementCount : ℕ := 21

/-- The set of digits available for arrangement -/
def availableDigits : Finset ℕ := {1, 2, 5, 0}

/-- Predicate to check if a number is a five-digit multiple of 5 -/
def isValidNumber (n : ℕ) : Prop :=
  10000 ≤ n ∧ n < 100000 ∧ n % 5 = 0

/-- The set of all valid arrangements -/
def validArrangements : Finset ℕ :=
  sorry

theorem count_valid_arrangements :
  Finset.card validArrangements = arrangementCount := by
  sorry

end count_valid_arrangements_l2520_252026


namespace pirate_overtakes_at_six_hours_l2520_252023

/-- Represents the chase scenario between a pirate ship and a merchant vessel -/
structure ChaseScenario where
  initial_distance : ℝ
  initial_pirate_speed : ℝ
  initial_merchant_speed : ℝ
  speed_change_time : ℝ
  final_pirate_speed : ℝ
  final_merchant_speed : ℝ

/-- Calculates the time when the pirate ship overtakes the merchant vessel -/
def overtake_time (scenario : ChaseScenario) : ℝ :=
  sorry

/-- Theorem stating that the pirate ship overtakes the merchant vessel after 6 hours -/
theorem pirate_overtakes_at_six_hours (scenario : ChaseScenario) 
  (h1 : scenario.initial_distance = 15)
  (h2 : scenario.initial_pirate_speed = 14)
  (h3 : scenario.initial_merchant_speed = 10)
  (h4 : scenario.speed_change_time = 3)
  (h5 : scenario.final_pirate_speed = 12)
  (h6 : scenario.final_merchant_speed = 11) :
  overtake_time scenario = 6 :=
  sorry

end pirate_overtakes_at_six_hours_l2520_252023


namespace A_intersect_B_equals_unit_interval_l2520_252076

-- Define the sets A and B
def A : Set ℝ := {y | ∃ x, y = Real.sin x}
def B : Set ℝ := {y | ∃ x, y = Real.sqrt (-x^2 + 4*x - 3)}

-- State the theorem
theorem A_intersect_B_equals_unit_interval :
  A ∩ B = Set.Icc 0 1 := by sorry

end A_intersect_B_equals_unit_interval_l2520_252076


namespace jerry_added_figures_l2520_252037

/-- Represents the shelf of action figures -/
structure ActionFigureShelf :=
  (initial_count : Nat)
  (final_count : Nat)
  (removed_count : Nat)
  (is_arithmetic_sequence : Bool)
  (first_last_preserved : Bool)
  (common_difference_preserved : Bool)

/-- Calculates the number of action figures added to the shelf -/
def added_figures (shelf : ActionFigureShelf) : Nat :=
  shelf.final_count + shelf.removed_count - shelf.initial_count

/-- Theorem stating the number of added action figures -/
theorem jerry_added_figures (shelf : ActionFigureShelf) 
  (h1 : shelf.initial_count = 7)
  (h2 : shelf.final_count = 8)
  (h3 : shelf.removed_count = 10)
  (h4 : shelf.is_arithmetic_sequence = true)
  (h5 : shelf.first_last_preserved = true)
  (h6 : shelf.common_difference_preserved = true) :
  added_figures shelf = 18 := by
  sorry

#check jerry_added_figures

end jerry_added_figures_l2520_252037


namespace investment_principal_l2520_252083

/-- 
Given two investments with the same principal and interest rate:
1. Peter's investment yields $815 after 3 years
2. David's investment yields $850 after 4 years
3. Both use simple interest

This theorem proves that the principal invested is $710
-/
theorem investment_principal (P r : ℚ) : 
  (P + P * r * 3 = 815) →
  (P + P * r * 4 = 850) →
  P = 710 := by
sorry

end investment_principal_l2520_252083


namespace no_natural_solution_for_square_difference_2014_l2520_252048

theorem no_natural_solution_for_square_difference_2014 :
  ∀ (m n : ℕ), m^2 ≠ n^2 + 2014 := by
  sorry

end no_natural_solution_for_square_difference_2014_l2520_252048


namespace new_person_weight_l2520_252008

/-- Given a group of 9 people where one person weighing 86 kg is replaced by a new person,
    and the average weight of the group increases by 5.5 kg,
    prove that the weight of the new person is 135.5 kg. -/
theorem new_person_weight (initial_count : ℕ) (weight_increase : ℝ) (replaced_weight : ℝ) :
  initial_count = 9 →
  weight_increase = 5.5 →
  replaced_weight = 86 →
  (initial_count : ℝ) * weight_increase + replaced_weight = 135.5 :=
by sorry

end new_person_weight_l2520_252008


namespace three_right_angles_implies_rectangle_l2520_252089

/-- A quadrilateral is a polygon with four sides and four vertices. -/
structure Quadrilateral where
  vertices : Fin 4 → ℝ × ℝ

/-- An angle is right if it measures 90 degrees or π/2 radians. -/
def is_right_angle (q : Quadrilateral) (i : Fin 4) : Prop := sorry

/-- A rectangle is a quadrilateral with four right angles. -/
def is_rectangle (q : Quadrilateral) : Prop :=
  ∀ i : Fin 4, is_right_angle q i

/-- Theorem: If a quadrilateral has three right angles, it is a rectangle. -/
theorem three_right_angles_implies_rectangle (q : Quadrilateral) 
  (h : ∃ i j k : Fin 4, i ≠ j ∧ j ≠ k ∧ i ≠ k ∧ 
    is_right_angle q i ∧ is_right_angle q j ∧ is_right_angle q k) : 
  is_rectangle q :=
sorry

end three_right_angles_implies_rectangle_l2520_252089


namespace min_non_isosceles_2008gon_l2520_252077

/-- The number of ones in the binary representation of a natural number -/
def binary_ones (n : ℕ) : ℕ := sorry

/-- A regular polygon -/
structure RegularPolygon where
  sides : ℕ
  sides_pos : sides > 0

/-- A triangulation of a regular polygon -/
structure Triangulation (p : RegularPolygon) where
  diagonals : ℕ
  diag_bound : diagonals ≤ p.sides - 3

/-- The number of non-isosceles triangles in a triangulation -/
def non_isosceles_count (p : RegularPolygon) (t : Triangulation p) : ℕ := sorry

theorem min_non_isosceles_2008gon :
  ∀ (p : RegularPolygon) (t : Triangulation p),
    p.sides = 2008 →
    t.diagonals = 2005 →
    non_isosceles_count p t ≥ 5 :=
sorry

end min_non_isosceles_2008gon_l2520_252077


namespace solution_set_equality_l2520_252033

open Set

-- Define the solution set
def solutionSet : Set ℝ := {x | |2*x + 1| > 3}

-- State the theorem
theorem solution_set_equality : solutionSet = Iio (-2) ∪ Ioi 1 := by
  sorry

end solution_set_equality_l2520_252033


namespace function_inequality_implies_parameter_bound_l2520_252018

/-- Given f(x) = 2ln(x) - x^2 and g(x) = xe^x - (a-1)x^2 - x - 2ln(x) for x > 0,
    if f(x) + g(x) ≥ 0 for all x > 0, then a ≤ 1 -/
theorem function_inequality_implies_parameter_bound (a : ℝ) :
  (∀ x > 0, 2 * Real.log x - x^2 + x * Real.exp x - (a - 1) * x^2 - x - 2 * Real.log x ≥ 0) →
  a ≤ 1 := by
  sorry

end function_inequality_implies_parameter_bound_l2520_252018


namespace ten_people_round_table_l2520_252054

-- Define the number of people
def n : ℕ := 10

-- Define the function to calculate the number of distinct arrangements
def distinct_circular_arrangements (m : ℕ) : ℕ := Nat.factorial (m - 1)

-- Theorem statement
theorem ten_people_round_table : 
  distinct_circular_arrangements n = Nat.factorial 9 :=
sorry

end ten_people_round_table_l2520_252054


namespace ellipse_constant_product_l2520_252095

-- Define the ellipse C
def ellipse_C (x y : ℝ) : Prop := x^2 / 6 + y^2 / 2 = 1

-- Define the moving line
def moving_line (k x y : ℝ) : Prop := y = k * (x - 2)

-- Define the dot product of vectors
def dot_product (x1 y1 x2 y2 : ℝ) : ℝ := x1 * x2 + y1 * y2

-- Theorem statement
theorem ellipse_constant_product :
  ∃ (E : ℝ × ℝ),
    E.2 = 0 ∧
    ∀ (k : ℝ) (A B : ℝ × ℝ),
      k ≠ 0 →
      ellipse_C A.1 A.2 →
      ellipse_C B.1 B.2 →
      moving_line k A.1 A.2 →
      moving_line k B.1 B.2 →
      dot_product (A.1 - E.1) (A.2 - E.2) (B.1 - E.1) (B.2 - E.2) = -5/9 :=
sorry

end ellipse_constant_product_l2520_252095


namespace fraction_subtraction_equality_l2520_252065

theorem fraction_subtraction_equality : (3 + 6 + 9) / (2 + 5 + 8) - (2 + 5 + 8) / (3 + 6 + 9) = 11 / 30 := by
  sorry

end fraction_subtraction_equality_l2520_252065


namespace function_value_at_a_plus_one_l2520_252022

/-- Given a function f(x) = x^2 + 1, prove that f(a+1) = a^2 + 2a + 2 for any real number a. -/
theorem function_value_at_a_plus_one (a : ℝ) : 
  let f : ℝ → ℝ := λ x ↦ x^2 + 1
  f (a + 1) = a^2 + 2*a + 2 := by
sorry

end function_value_at_a_plus_one_l2520_252022


namespace washing_machines_removed_l2520_252067

/-- Represents a shipping container with crates, boxes, and washing machines -/
structure ShippingContainer where
  num_crates : ℕ
  boxes_per_crate : ℕ
  initial_machines_per_box : ℕ
  machines_removed_per_box : ℕ

/-- Calculates the total number of washing machines removed from a shipping container -/
def total_machines_removed (container : ShippingContainer) : ℕ :=
  container.num_crates * container.boxes_per_crate * container.machines_removed_per_box

/-- Theorem stating the number of washing machines removed from the specific shipping container -/
theorem washing_machines_removed : 
  let container : ShippingContainer := {
    num_crates := 10,
    boxes_per_crate := 6,
    initial_machines_per_box := 4,
    machines_removed_per_box := 1
  }
  total_machines_removed container = 60 := by
  sorry


end washing_machines_removed_l2520_252067


namespace prime_cube_plus_seven_composite_l2520_252010

theorem prime_cube_plus_seven_composite (P : ℕ) (h1 : Nat.Prime P) (h2 : Nat.Prime (P^3 + 5)) :
  ¬Nat.Prime (P^3 + 7) ∧ (P^3 + 7) > 1 := by
  sorry

end prime_cube_plus_seven_composite_l2520_252010
