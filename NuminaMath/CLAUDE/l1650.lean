import Mathlib

namespace NUMINAMATH_CALUDE_min_value_of_expression_l1650_165098

theorem min_value_of_expression (a b : ℝ) (ha : a > 0) (hb : b > 0)
  (h_geometric_mean : Real.sqrt 3 = Real.sqrt (3^a * 3^(2*b))) :
  ∃ (x y : ℝ) (hx : x > 0) (hy : y > 0),
    (∀ (u v : ℝ) (hu : u > 0) (hv : v > 0),
      Real.sqrt 3 = Real.sqrt (3^u * 3^(2*v)) →
      2/u + 1/v ≥ 2/x + 1/y) ∧
    2/x + 1/y = 8 :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_expression_l1650_165098


namespace NUMINAMATH_CALUDE_jens_son_age_l1650_165025

theorem jens_son_age :
  ∀ (sons_age : ℕ),
  (41 : ℕ) = 25 + sons_age →  -- Jen was 25 when her son was born, and she's 41 now
  (41 : ℕ) = 3 * sons_age - 7 →  -- Jen's age is 7 less than 3 times her son's age
  sons_age = 16 :=
by
  sorry

end NUMINAMATH_CALUDE_jens_son_age_l1650_165025


namespace NUMINAMATH_CALUDE_complex_magnitude_two_thirds_minus_four_fifths_i_l1650_165077

theorem complex_magnitude_two_thirds_minus_four_fifths_i :
  Complex.abs (⟨2/3, -4/5⟩ : ℂ) = Real.sqrt 244 / 15 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_two_thirds_minus_four_fifths_i_l1650_165077


namespace NUMINAMATH_CALUDE_toothpick_pattern_200th_stage_l1650_165018

/-- 
Given an arithmetic sequence where:
- a is the first term
- d is the common difference
- n is the term number
This function calculates the nth term of the sequence.
-/
def arithmeticSequenceTerm (a : ℕ) (d : ℕ) (n : ℕ) : ℕ :=
  a + (n - 1) * d

/--
Theorem: In an arithmetic sequence where the first term is 6 and the common difference is 5,
the 200th term is equal to 1001.
-/
theorem toothpick_pattern_200th_stage :
  arithmeticSequenceTerm 6 5 200 = 1001 := by
  sorry

#eval arithmeticSequenceTerm 6 5 200

end NUMINAMATH_CALUDE_toothpick_pattern_200th_stage_l1650_165018


namespace NUMINAMATH_CALUDE_one_carton_per_case_l1650_165046

/-- Given a case containing cartons, each carton containing b boxes,
    each box containing 400 paper clips, and 800 paper clips in 2 cases,
    prove that there is 1 carton in a case. -/
theorem one_carton_per_case (b : ℕ) (h1 : b ≥ 1) :
  ∃ (c : ℕ), c = 1 ∧ 2 * c * b * 400 = 800 := by
  sorry

#check one_carton_per_case

end NUMINAMATH_CALUDE_one_carton_per_case_l1650_165046


namespace NUMINAMATH_CALUDE_intersection_condition_l1650_165011

/-- The set A defined by the equation y = x^2 + mx + 2 -/
def A (m : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.2 = p.1^2 + m * p.1 + 2}

/-- The set B defined by the equation y = x + 1 -/
def B : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.2 = p.1 + 1}

/-- The theorem stating the condition for non-empty intersection of A and B -/
theorem intersection_condition (m : ℝ) :
  (A m ∩ B).Nonempty ↔ m ≤ -1 ∨ m ≥ 3 := by
  sorry

end NUMINAMATH_CALUDE_intersection_condition_l1650_165011


namespace NUMINAMATH_CALUDE_football_shoes_cost_l1650_165067

-- Define the costs and amounts
def football_cost : ℚ := 3.75
def shorts_cost : ℚ := 2.40
def zachary_has : ℚ := 10
def zachary_needs : ℚ := 8

-- Define the theorem
theorem football_shoes_cost :
  let total_cost := zachary_has + zachary_needs
  let other_items_cost := football_cost + shorts_cost
  total_cost - other_items_cost = 11.85 := by sorry

end NUMINAMATH_CALUDE_football_shoes_cost_l1650_165067


namespace NUMINAMATH_CALUDE_triangle_inequality_l1650_165062

theorem triangle_inequality (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (htriangle : a + b > c ∧ b + c > a ∧ c + a > b) :
  let S := (a + b + c) / 2
  2 * S * (Real.sqrt (S - a) + Real.sqrt (S - b) + Real.sqrt (S - c)) ≤
  3 * (Real.sqrt (b * c * (S - a)) + Real.sqrt (c * a * (S - b)) + Real.sqrt (a * b * (S - c))) := by
  sorry

end NUMINAMATH_CALUDE_triangle_inequality_l1650_165062


namespace NUMINAMATH_CALUDE_count_rectangles_with_cell_l1650_165027

/-- The number of rectangles containing a specific cell in a grid. -/
def num_rectangles (m n p q : ℕ) : ℕ :=
  p * q * (m - p + 1) * (n - q + 1)

/-- Theorem stating the number of rectangles containing a specific cell in a grid. -/
theorem count_rectangles_with_cell (m n p q : ℕ) 
  (hpm : p ≤ m) (hqn : q ≤ n) (hp : p > 0) (hq : q > 0) : 
  num_rectangles m n p q = p * q * (m - p + 1) * (n - q + 1) := by
  sorry

end NUMINAMATH_CALUDE_count_rectangles_with_cell_l1650_165027


namespace NUMINAMATH_CALUDE_five_digit_divisible_by_11_l1650_165058

/-- Represents a five-digit number in the form 53A47 -/
def number (A : ℕ) : ℕ := 53000 + A * 100 + 47

/-- Checks if a number is divisible by 11 -/
def divisible_by_11 (n : ℕ) : Prop := n % 11 = 0

theorem five_digit_divisible_by_11 :
  ∃ (A : ℕ), A < 10 ∧ divisible_by_11 (number A) ∧
  ∀ (B : ℕ), B < A → ¬divisible_by_11 (number B) :=
by sorry

end NUMINAMATH_CALUDE_five_digit_divisible_by_11_l1650_165058


namespace NUMINAMATH_CALUDE_max_value_polynomial_l1650_165094

theorem max_value_polynomial (x y : ℝ) (h : x + y = 5) :
  (∃ (z : ℝ), x^4*y + x^3*y + x^2*y + x*y + x*y^2 + x*y^3 + x*y^4 ≤ z) ∧
  (∀ (z : ℝ), x^4*y + x^3*y + x^2*y + x*y + x*y^2 + x*y^3 + x*y^4 ≤ z → 6084/17 ≤ z) :=
sorry

end NUMINAMATH_CALUDE_max_value_polynomial_l1650_165094


namespace NUMINAMATH_CALUDE_point_distance_product_l1650_165043

theorem point_distance_product : 
  ∃ (y₁ y₂ : ℝ), 
    (((5 - (-3))^2 + (2 - y₁)^2 : ℝ) = 14^2) ∧
    (((5 - (-3))^2 + (2 - y₂)^2 : ℝ) = 14^2) ∧
    (y₁ * y₂ = -128) :=
by sorry

end NUMINAMATH_CALUDE_point_distance_product_l1650_165043


namespace NUMINAMATH_CALUDE_remaining_integers_l1650_165066

/-- The number of integers remaining in a set of 1 to 80 after removing multiples of 4 and 5 -/
theorem remaining_integers (n : ℕ) (hn : n = 80) : 
  n - (n / 4 + n / 5 - n / 20) = 48 := by
  sorry

end NUMINAMATH_CALUDE_remaining_integers_l1650_165066


namespace NUMINAMATH_CALUDE_tenth_stays_tenth_probability_sum_of_numerator_and_denominator_l1650_165035

/-- Represents a sequence of distinct numbers -/
def DistinctSequence (n : ℕ) := { s : Fin n → ℕ // Function.Injective s }

/-- The probability of the 10th element staying in the 10th position after one bubble pass -/
def probabilityTenthStaysTenth (n : ℕ) : ℚ :=
  if n < 12 then 0 else 1 / (12 * 11)

theorem tenth_stays_tenth_probability :
  probabilityTenthStaysTenth 20 = 1 / 132 := by sorry

#eval Nat.gcd 1 132  -- Should output 1, confirming 1/132 is in lowest terms

theorem sum_of_numerator_and_denominator :
  let p := 1
  let q := 132
  p + q = 133 := by sorry

end NUMINAMATH_CALUDE_tenth_stays_tenth_probability_sum_of_numerator_and_denominator_l1650_165035


namespace NUMINAMATH_CALUDE_lesser_fraction_l1650_165015

theorem lesser_fraction (x y : ℚ) 
  (sum_eq : x + y = 9/10)
  (prod_eq : x * y = 1/15) :
  min x y = 1/5 := by sorry

end NUMINAMATH_CALUDE_lesser_fraction_l1650_165015


namespace NUMINAMATH_CALUDE_surface_area_bound_l1650_165059

/-- A convex broken line -/
structure ConvexBrokenLine where
  points : List (ℝ × ℝ)
  is_convex : Bool
  length : ℝ

/-- The surface area of revolution of a convex broken line -/
def surface_area_of_revolution (line : ConvexBrokenLine) : ℝ := sorry

/-- Theorem: The surface area of revolution of a convex broken line
    is less than or equal to π * d² / 2, where d is the length of the line -/
theorem surface_area_bound (line : ConvexBrokenLine) :
  surface_area_of_revolution line ≤ Real.pi * line.length^2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_surface_area_bound_l1650_165059


namespace NUMINAMATH_CALUDE_units_digit_of_k_squared_plus_two_to_k_l1650_165074

def k : ℕ := 2009^2 + 2^2009 - 3

theorem units_digit_of_k_squared_plus_two_to_k (k : ℕ := 2009^2 + 2^2009 - 3) :
  (k^2 + 2^k) % 10 = 1 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_k_squared_plus_two_to_k_l1650_165074


namespace NUMINAMATH_CALUDE_zhang_li_age_ratio_l1650_165039

theorem zhang_li_age_ratio :
  ∀ (zhang_age li_age jung_age : ℕ),
    li_age = 12 →
    jung_age = 26 →
    jung_age = zhang_age + 2 →
    zhang_age / li_age = 2 :=
by
  sorry

end NUMINAMATH_CALUDE_zhang_li_age_ratio_l1650_165039


namespace NUMINAMATH_CALUDE_always_odd_l1650_165006

theorem always_odd (a b c : ℕ+) (ha : a.val % 2 = 1) (hb : b.val % 2 = 1) :
  (3^a.val + (b.val - 1)^2 * c.val) % 2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_always_odd_l1650_165006


namespace NUMINAMATH_CALUDE_min_score_is_45_l1650_165064

/-- Represents the test scores and conditions -/
structure TestScores where
  num_tests : ℕ
  max_score : ℕ
  first_three : Fin 3 → ℕ
  target_average : ℕ

/-- Calculates the minimum score needed on one of the last two tests -/
def min_score (ts : TestScores) : ℕ :=
  let total_needed := ts.target_average * ts.num_tests
  let first_three_sum := (ts.first_three 0) + (ts.first_three 1) + (ts.first_three 2)
  let remaining_sum := total_needed - first_three_sum
  remaining_sum - ts.max_score

/-- Theorem stating the minimum score needed is 45 -/
theorem min_score_is_45 (ts : TestScores) 
  (h1 : ts.num_tests = 5)
  (h2 : ts.max_score = 120)
  (h3 : ts.first_three 0 = 86 ∧ ts.first_three 1 = 102 ∧ ts.first_three 2 = 97)
  (h4 : ts.target_average = 90) :
  min_score ts = 45 := by
  sorry

#eval min_score { num_tests := 5, max_score := 120, first_three := ![86, 102, 97], target_average := 90 }

end NUMINAMATH_CALUDE_min_score_is_45_l1650_165064


namespace NUMINAMATH_CALUDE_jogger_speed_l1650_165083

/-- The speed of a jogger given specific conditions involving a train --/
theorem jogger_speed (train_length : ℝ) (initial_distance : ℝ) (train_speed : ℝ) (passing_time : ℝ)
  (h1 : train_length = 120)
  (h2 : initial_distance = 120)
  (h3 : train_speed = 45)
  (h4 : passing_time = 24)
  : ∃ (jogger_speed : ℝ), jogger_speed = 9 := by
  sorry


end NUMINAMATH_CALUDE_jogger_speed_l1650_165083


namespace NUMINAMATH_CALUDE_rectangle_areas_sum_l1650_165002

theorem rectangle_areas_sum : 
  let widths : List ℕ := [2, 3, 4, 5, 6, 7, 8]
  let lengths : List ℕ := [5, 8, 11, 14, 17, 20, 23]
  (widths.zip lengths).map (fun (w, l) => w * l) |>.sum = 574 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_areas_sum_l1650_165002


namespace NUMINAMATH_CALUDE_cos_2x_value_l1650_165090

theorem cos_2x_value (x : ℝ) (h : 2 * Real.sin (Real.pi - x) + 1 = 0) : 
  Real.cos (2 * x) = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_cos_2x_value_l1650_165090


namespace NUMINAMATH_CALUDE_total_oranges_l1650_165093

def orange_groups : ℕ := 16
def oranges_per_group : ℕ := 24

theorem total_oranges : orange_groups * oranges_per_group = 384 := by
  sorry

end NUMINAMATH_CALUDE_total_oranges_l1650_165093


namespace NUMINAMATH_CALUDE_smallest_norm_given_condition_l1650_165022

/-- Given a vector v in ℝ², prove that the smallest possible value of its norm,
    given that ‖v + (4, 2)‖ = 10, is 10 - 2√5. -/
theorem smallest_norm_given_condition (v : ℝ × ℝ) 
    (h : ‖v + (4, 2)‖ = 10) : 
    ∃ (w : ℝ × ℝ), ‖w‖ = 10 - 2 * Real.sqrt 5 ∧ 
    ∀ (u : ℝ × ℝ), ‖u + (4, 2)‖ = 10 → ‖w‖ ≤ ‖u‖ := by
  sorry

end NUMINAMATH_CALUDE_smallest_norm_given_condition_l1650_165022


namespace NUMINAMATH_CALUDE_binomial_seven_four_l1650_165000

theorem binomial_seven_four : Nat.choose 7 4 = 35 := by
  sorry

end NUMINAMATH_CALUDE_binomial_seven_four_l1650_165000


namespace NUMINAMATH_CALUDE_average_price_is_52_cents_l1650_165095

/-- Represents the fruit selection problem --/
structure FruitSelection where
  apple_price : ℚ
  orange_price : ℚ
  total_fruits : ℕ
  initial_avg_price : ℚ
  oranges_returned : ℕ

/-- Calculates the average price of fruits kept --/
def average_price_kept (fs : FruitSelection) : ℚ :=
  let apples := fs.total_fruits - (fs.initial_avg_price * fs.total_fruits - fs.apple_price * fs.total_fruits) / (fs.orange_price - fs.apple_price)
  let oranges := fs.total_fruits - apples
  let kept_oranges := oranges - fs.oranges_returned
  let total_kept := apples + kept_oranges
  (fs.apple_price * apples + fs.orange_price * kept_oranges) / total_kept

/-- Theorem stating that the average price of fruits kept is 52 cents --/
theorem average_price_is_52_cents (fs : FruitSelection) 
    (h1 : fs.apple_price = 40/100)
    (h2 : fs.orange_price = 60/100)
    (h3 : fs.total_fruits = 30)
    (h4 : fs.initial_avg_price = 56/100)
    (h5 : fs.oranges_returned = 15) :
  average_price_kept fs = 52/100 := by
  sorry

#eval average_price_kept {
  apple_price := 40/100,
  orange_price := 60/100,
  total_fruits := 30,
  initial_avg_price := 56/100,
  oranges_returned := 15
}

end NUMINAMATH_CALUDE_average_price_is_52_cents_l1650_165095


namespace NUMINAMATH_CALUDE_nectar_water_percentage_l1650_165085

/-- Given that 1.7 kg of nectar yields 1 kg of honey, and the honey contains 15% water,
    prove that the nectar contains 50% water. -/
theorem nectar_water_percentage :
  ∀ (nectar_weight honey_weight : ℝ) 
    (honey_water_percentage nectar_water_percentage : ℝ),
  nectar_weight = 1.7 →
  honey_weight = 1 →
  honey_water_percentage = 15 →
  nectar_water_percentage = 
    (nectar_weight * honey_water_percentage / 100 + (nectar_weight - honey_weight)) / 
    nectar_weight * 100 →
  nectar_water_percentage = 50 := by
  sorry

end NUMINAMATH_CALUDE_nectar_water_percentage_l1650_165085


namespace NUMINAMATH_CALUDE_symmetric_cubic_at_1_l1650_165089

/-- A cubic function f(x) = x³ + ax² + bx + 2 -/
def f (a b : ℝ) (x : ℝ) : ℝ := x^3 + a*x^2 + b*x + 2

/-- The function f is symmetric about the point (2,0) -/
def is_symmetric_about_2_0 (a b : ℝ) : Prop :=
  ∀ x : ℝ, f a b (x + 2) = f a b (2 - x)

theorem symmetric_cubic_at_1 (a b : ℝ) : 
  is_symmetric_about_2_0 a b → f a b 1 = 4 := by
sorry

end NUMINAMATH_CALUDE_symmetric_cubic_at_1_l1650_165089


namespace NUMINAMATH_CALUDE_triangle_angle_equality_l1650_165061

open Real

theorem triangle_angle_equality (a b c : ℝ) (α β γ : ℝ) (R : ℝ) :
  a > 0 → b > 0 → c > 0 → R > 0 →
  α > 0 → β > 0 → γ > 0 →
  α + β + γ = pi →
  (a * cos α + b * cos β + c * cos γ) / (a * sin β + b * sin γ + c * sin α) = (a + b + c) / (9 * R) →
  α = pi / 3 ∧ β = pi / 3 ∧ γ = pi / 3 := by
sorry

end NUMINAMATH_CALUDE_triangle_angle_equality_l1650_165061


namespace NUMINAMATH_CALUDE_polynomial_simplification_l1650_165024

theorem polynomial_simplification (p : ℝ) : 
  (4 * p^4 + 2 * p^3 - 7 * p^2 + 3 * p - 5) + (-3 * p^4 - 2 * p^3 + 8 * p^2 - 4 * p + 6) = 
  p^4 + p^2 - p + 1 := by
sorry

end NUMINAMATH_CALUDE_polynomial_simplification_l1650_165024


namespace NUMINAMATH_CALUDE_matrix_equation_solution_l1650_165004

/-- The value of a 2x2 matrix [[a, c], [d, b]] is ab - cd -/
def matrix_value (a b c d : ℝ) : ℝ := a * b - c * d

/-- The solution to the matrix equation for a given k -/
def solution (k : ℝ) : Set ℝ :=
  {x : ℝ | x = (4 + Real.sqrt (16 + 60 * k)) / 30 ∨ x = (4 - Real.sqrt (16 + 60 * k)) / 30}

theorem matrix_equation_solution (k : ℝ) (h : k ≥ -4/15) :
  ∀ x : ℝ, matrix_value (3*x) (5*x) 2 (2*x) = k ↔ x ∈ solution k := by
  sorry

end NUMINAMATH_CALUDE_matrix_equation_solution_l1650_165004


namespace NUMINAMATH_CALUDE_eighth_square_fully_shaded_l1650_165072

/-- Represents the number of shaded squares and total squares in the nth diagram -/
def squarePattern (n : ℕ) : ℕ := n^2

/-- The fraction of shaded squares in the nth diagram -/
def shadedFraction (n : ℕ) : ℚ := squarePattern n / squarePattern n

theorem eighth_square_fully_shaded :
  shadedFraction 8 = 1 := by sorry

end NUMINAMATH_CALUDE_eighth_square_fully_shaded_l1650_165072


namespace NUMINAMATH_CALUDE_factor_polynomial_l1650_165026

theorem factor_polynomial (x : ℝ) : 66 * x^6 - 231 * x^12 = 33 * x^6 * (2 - 7 * x^6) := by
  sorry

end NUMINAMATH_CALUDE_factor_polynomial_l1650_165026


namespace NUMINAMATH_CALUDE_smallest_angle_at_vertices_l1650_165057

/-- A cube in 3D space -/
structure Cube where
  side : ℝ
  center : ℝ × ℝ × ℝ

/-- A point in 3D space -/
def Point3D := ℝ × ℝ × ℝ

/-- The angle at which a point sees the space diagonal of a cube -/
def angle_at_point (c : Cube) (p : Point3D) : ℝ := sorry

/-- The vertices of a cube -/
def cube_vertices (c : Cube) : Set Point3D := sorry

/-- The surface of a cube -/
def cube_surface (c : Cube) : Set Point3D := sorry

/-- Theorem: The vertices of a cube are the only points on its surface where 
    the space diagonal is seen at a 90-degree angle, which is the smallest possible angle -/
theorem smallest_angle_at_vertices (c : Cube) : 
  ∀ p ∈ cube_surface c, 
    angle_at_point c p = Real.pi / 2 ↔ p ∈ cube_vertices c :=
by sorry

end NUMINAMATH_CALUDE_smallest_angle_at_vertices_l1650_165057


namespace NUMINAMATH_CALUDE_toms_floor_replacement_cost_l1650_165014

/-- The total cost to replace a floor given the room dimensions, removal cost, and new floor cost per square foot. -/
def total_floor_replacement_cost (length width removal_cost cost_per_sqft : ℝ) : ℝ :=
  removal_cost + length * width * cost_per_sqft

/-- Theorem stating that the total cost to replace the floor in Tom's room is $120. -/
theorem toms_floor_replacement_cost :
  total_floor_replacement_cost 8 7 50 1.25 = 120 := by
  sorry

end NUMINAMATH_CALUDE_toms_floor_replacement_cost_l1650_165014


namespace NUMINAMATH_CALUDE_sum_remainder_l1650_165080

theorem sum_remainder (a b c d e : ℕ) 
  (ha : a % 13 = 3)
  (hb : b % 13 = 5)
  (hc : c % 13 = 7)
  (hd : d % 13 = 9)
  (he : e % 13 = 11)
  (hsquare : ∃ k : ℕ, c = k * k) :
  (a + b + c + d + e) % 13 = 9 := by
  sorry

end NUMINAMATH_CALUDE_sum_remainder_l1650_165080


namespace NUMINAMATH_CALUDE_car_speed_problem_l1650_165078

/-- Proves that car R's speed is 50 mph given the problem conditions -/
theorem car_speed_problem (distance : ℝ) (time_diff : ℝ) (speed_diff : ℝ) :
  distance = 600 →
  time_diff = 2 →
  speed_diff = 10 →
  (distance / (distance / 50 - time_diff) = 50 + speed_diff) →
  50 = distance / (distance / 50) :=
by sorry

end NUMINAMATH_CALUDE_car_speed_problem_l1650_165078


namespace NUMINAMATH_CALUDE_log_sum_problem_l1650_165040

theorem log_sum_problem (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h1 : y = 2016 * x) (h2 : x^y = y^x) : 
  Real.log x / Real.log 2016 + Real.log y / Real.log 2016 = 2017 / 2015 := by
  sorry

end NUMINAMATH_CALUDE_log_sum_problem_l1650_165040


namespace NUMINAMATH_CALUDE_triangle_rectangle_equal_area_l1650_165044

theorem triangle_rectangle_equal_area (h : ℝ) (h_pos : h > 0) :
  let triangle_base : ℝ := 24
  let triangle_area : ℝ := (1 / 2) * triangle_base * h
  let rectangle_base : ℝ := (1 / 2) * triangle_base
  let rectangle_area : ℝ := rectangle_base * h
  triangle_area = rectangle_area →
  rectangle_base = 12 := by
sorry


end NUMINAMATH_CALUDE_triangle_rectangle_equal_area_l1650_165044


namespace NUMINAMATH_CALUDE_bridget_apples_l1650_165012

/-- The number of apples Bridget originally bought -/
def original_apples : ℕ := 18

/-- The number of apples Bridget kept for herself in the end -/
def kept_apples : ℕ := 6

/-- The number of apples Bridget gave to Cassie -/
def given_apples : ℕ := 5

/-- The number of additional apples Bridget found in the bag -/
def found_apples : ℕ := 2

theorem bridget_apples : 
  original_apples / 2 - given_apples + found_apples = kept_apples := by
  sorry

#check bridget_apples

end NUMINAMATH_CALUDE_bridget_apples_l1650_165012


namespace NUMINAMATH_CALUDE_complete_square_transform_l1650_165019

theorem complete_square_transform (a : ℝ) : a^2 + 4*a - 5 = (a + 2)^2 - 9 := by
  sorry

end NUMINAMATH_CALUDE_complete_square_transform_l1650_165019


namespace NUMINAMATH_CALUDE_candy_box_total_l1650_165091

theorem candy_box_total (purple orange yellow total : ℕ) : 
  purple + orange + yellow = total →
  2 * orange = 4 * purple →
  5 * purple = 2 * yellow →
  yellow = 40 →
  total = 88 := by
sorry

end NUMINAMATH_CALUDE_candy_box_total_l1650_165091


namespace NUMINAMATH_CALUDE_greatest_prime_divisor_plus_floor_sqrt_equality_l1650_165088

def greatest_prime_divisor (n : ℕ) : ℕ := sorry

def floor_sqrt (n : ℕ) : ℕ := sorry

theorem greatest_prime_divisor_plus_floor_sqrt_equality (n : ℕ) :
  n ≥ 2 →
  (greatest_prime_divisor n + floor_sqrt n = greatest_prime_divisor (n + 1) + floor_sqrt (n + 1)) ↔
  n = 3 := by sorry

end NUMINAMATH_CALUDE_greatest_prime_divisor_plus_floor_sqrt_equality_l1650_165088


namespace NUMINAMATH_CALUDE_root_in_interval_implies_m_range_l1650_165021

-- Define the function f(x) = x^3 - 3x
def f (x : ℝ) : ℝ := x^3 - 3*x

-- Define the theorem
theorem root_in_interval_implies_m_range :
  ∀ m : ℝ, (∃ x : ℝ, x ∈ Set.Icc 0 2 ∧ f x + m = 0) → m ∈ Set.Icc (-2) 2 := by
  sorry

end NUMINAMATH_CALUDE_root_in_interval_implies_m_range_l1650_165021


namespace NUMINAMATH_CALUDE_max_sum_given_product_l1650_165041

theorem max_sum_given_product (x y : ℝ) : 
  (2015 + x^2) * (2015 + y^2) = 2^22 → x + y ≤ 2 * Real.sqrt 33 :=
by sorry

end NUMINAMATH_CALUDE_max_sum_given_product_l1650_165041


namespace NUMINAMATH_CALUDE_correct_price_reduction_equation_l1650_165016

/-- Represents the price reduction model for a sportswear item -/
def PriceReductionModel (initial_price final_price : ℝ) : Prop :=
  ∃ x : ℝ, 
    0 < x ∧ 
    x < 1 ∧ 
    initial_price * (1 - x)^2 = final_price

/-- Theorem stating that the given equation correctly models the price reduction -/
theorem correct_price_reduction_equation :
  PriceReductionModel 560 315 :=
sorry

end NUMINAMATH_CALUDE_correct_price_reduction_equation_l1650_165016


namespace NUMINAMATH_CALUDE_min_value_expression_l1650_165023

theorem min_value_expression (a b c d : ℝ) (hb : b ≠ 0) (horder : b > c ∧ c > a ∧ a > d) :
  ((2*a + b)^2 + (b - 2*c)^2 + (c - a)^2 + 3*d^2) / b^2 ≥ 49/36 :=
sorry

end NUMINAMATH_CALUDE_min_value_expression_l1650_165023


namespace NUMINAMATH_CALUDE_shaded_cubes_count_l1650_165081

/-- Represents a 4x4x4 cube with a specific shading pattern -/
structure ShadedCube where
  /-- The number of smaller cubes along each edge of the large cube -/
  size : Nat
  /-- The shading pattern on one face of the cube -/
  shading_pattern : Fin 4 → Fin 4 → Bool
  /-- Assertion that the cube is 4x4x4 -/
  size_is_four : size = 4
  /-- The shading pattern includes the entire top row -/
  top_row_shaded : ∀ j, shading_pattern 0 j = true
  /-- The shading pattern includes the entire bottom row -/
  bottom_row_shaded : ∀ j, shading_pattern 3 j = true
  /-- The shading pattern includes one cube in each corner of the second and third rows -/
  corners_shaded : (shading_pattern 1 0 = true) ∧ (shading_pattern 1 3 = true) ∧
                   (shading_pattern 2 0 = true) ∧ (shading_pattern 2 3 = true)

/-- The total number of smaller cubes with at least one face shaded -/
def count_shaded_cubes (cube : ShadedCube) : Nat :=
  sorry

/-- Theorem stating that the number of shaded cubes is 32 -/
theorem shaded_cubes_count (cube : ShadedCube) : count_shaded_cubes cube = 32 := by
  sorry

end NUMINAMATH_CALUDE_shaded_cubes_count_l1650_165081


namespace NUMINAMATH_CALUDE_five_digit_divisible_by_36_l1650_165071

theorem five_digit_divisible_by_36 (n : ℕ) : 
  (n ≥ 10000 ∧ n < 100000) ∧  -- five-digit number
  (∃ a b : ℕ, n = a * 10000 + 1000 + 200 + 30 + b) ∧  -- form ⬜123⬜
  (n % 36 = 0) →  -- divisible by 36
  (n = 11232 ∨ n = 61236) := by
sorry

end NUMINAMATH_CALUDE_five_digit_divisible_by_36_l1650_165071


namespace NUMINAMATH_CALUDE_tangent_line_parallel_l1650_165029

/-- The function f(x) = ax³ + x + 1 -/
def f (a : ℝ) (x : ℝ) : ℝ := a * x^3 + x + 1

/-- The derivative of f(x) -/
def f_derivative (a : ℝ) (x : ℝ) : ℝ := 3 * a * x^2 + 1

theorem tangent_line_parallel (a : ℝ) : 
  (f_derivative a 1 = 4) → a = 1 := by
  sorry

end NUMINAMATH_CALUDE_tangent_line_parallel_l1650_165029


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1650_165097

/-- An arithmetic sequence -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- The theorem to be proved -/
theorem arithmetic_sequence_sum (a : ℕ → ℝ) :
  ArithmeticSequence a →
  a 1 + a 4 + a 7 = 39 →
  a 2 + a 5 + a 8 = 33 →
  a 3 + a 6 + a 9 = 27 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1650_165097


namespace NUMINAMATH_CALUDE_max_odd_sum_2019_l1650_165037

/-- The maximum number of different odd natural numbers that sum to 2019 -/
def max_odd_sum_terms : ℕ := 43

/-- Predicate to check if a list of natural numbers consists of different odd numbers -/
def is_different_odd_list (l : List ℕ) : Prop :=
  l.Nodup ∧ ∀ n ∈ l, Odd n

theorem max_odd_sum_2019 :
  ∀ l : List ℕ,
    is_different_odd_list l →
    l.sum = 2019 →
    l.length ≤ max_odd_sum_terms ∧
    ∃ l' : List ℕ, is_different_odd_list l' ∧ l'.sum = 2019 ∧ l'.length = max_odd_sum_terms :=
by sorry

#check max_odd_sum_2019

end NUMINAMATH_CALUDE_max_odd_sum_2019_l1650_165037


namespace NUMINAMATH_CALUDE_solve_percentage_problem_l1650_165013

def percentage_problem (P : ℝ) (x : ℝ) : Prop :=
  (P / 100) * x = (5 / 100) * 500 - 20 ∧ x = 10

theorem solve_percentage_problem :
  ∃ P : ℝ, percentage_problem P 10 ∧ P = 50 := by
  sorry

end NUMINAMATH_CALUDE_solve_percentage_problem_l1650_165013


namespace NUMINAMATH_CALUDE_parabola_translation_l1650_165048

/-- Represents a parabola in the form y = ax^2 + bx + c -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Translates a parabola horizontally and vertically -/
def translate (p : Parabola) (h : ℝ) (v : ℝ) : Parabola :=
  { a := p.a
  , b := -2 * p.a * h + p.b
  , c := p.a * h^2 - p.b * h + p.c + v }

theorem parabola_translation :
  let original := Parabola.mk (-5) 0 1
  let translated := translate original (-1) (-2)
  translated = Parabola.mk (-5) 10 (-1) := by sorry

end NUMINAMATH_CALUDE_parabola_translation_l1650_165048


namespace NUMINAMATH_CALUDE_circle_line_intersection_l1650_165045

theorem circle_line_intersection (m : ℝ) :
  (∃! (p1 p2 p3 : ℝ × ℝ), 
    p1 ≠ p2 ∧ p1 ≠ p3 ∧ p2 ≠ p3 ∧
    (p1.1^2 + p1.2^2 = m) ∧ (p2.1^2 + p2.2^2 = m) ∧ (p3.1^2 + p3.2^2 = m) ∧
    |p1.1 - p1.2 + Real.sqrt 2| / Real.sqrt 2 = 1 ∧
    |p2.1 - p2.2 + Real.sqrt 2| / Real.sqrt 2 = 1 ∧
    |p3.1 - p3.2 + Real.sqrt 2| / Real.sqrt 2 = 1) →
  m = 4 :=
by sorry


end NUMINAMATH_CALUDE_circle_line_intersection_l1650_165045


namespace NUMINAMATH_CALUDE_quadratic_through_origin_l1650_165082

/-- A quadratic function of the form f(x) = ax² - 3x + a² - 1 -/
def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 - 3 * x + a^2 - 1

/-- Theorem: If f(0) = 0 and a > 0, then a = 1 -/
theorem quadratic_through_origin (a : ℝ) (h1 : f a 0 = 0) (h2 : a > 0) : a = 1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_through_origin_l1650_165082


namespace NUMINAMATH_CALUDE_extreme_points_cubic_l1650_165017

/-- Given a cubic function f(x) = x³ + ax² + bx with extreme points at x = -2 and x = 4,
    prove that a - b = 21. -/
theorem extreme_points_cubic (a b : ℝ) : 
  (∀ x : ℝ, (x = -2 ∨ x = 4) → (3 * x^2 + 2 * a * x + b = 0)) → 
  a - b = 21 := by
  sorry

end NUMINAMATH_CALUDE_extreme_points_cubic_l1650_165017


namespace NUMINAMATH_CALUDE_banana_arrangements_l1650_165032

def word := "BANANA"
def total_letters : ℕ := 6
def freq_B : ℕ := 1
def freq_A : ℕ := 3
def freq_N : ℕ := 2

theorem banana_arrangements : 
  (Nat.factorial total_letters) / 
  (Nat.factorial freq_B * Nat.factorial freq_A * Nat.factorial freq_N) = 60 := by
  sorry

end NUMINAMATH_CALUDE_banana_arrangements_l1650_165032


namespace NUMINAMATH_CALUDE_bc_cd_ratio_l1650_165054

-- Define the points on the line
variable (a b c d e : ℝ)

-- Define the conditions
axiom consecutive_points : a < b ∧ b < c ∧ c < d ∧ d < e
axiom de_length : e - d = 8
axiom ab_length : b - a = 5
axiom ac_length : c - a = 11
axiom ae_length : e - a = 22

-- Define the theorem
theorem bc_cd_ratio :
  (c - b) / (d - c) = 2 / 1 :=
sorry

end NUMINAMATH_CALUDE_bc_cd_ratio_l1650_165054


namespace NUMINAMATH_CALUDE_positive_range_of_even_function_l1650_165075

def evenFunction (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

theorem positive_range_of_even_function
  (f : ℝ → ℝ)
  (f' : ℝ → ℝ)
  (h_even : evenFunction f)
  (h_deriv : ∀ x ≠ 0, HasDerivAt f (f' x) x)
  (h_zero : f (-1) = 0)
  (h_ineq : ∀ x > 0, x * f' x - f x < 0) :
  {x : ℝ | f x > 0} = Set.Ioo (-1 : ℝ) 0 ∪ Set.Ioo 0 1 := by
sorry

end NUMINAMATH_CALUDE_positive_range_of_even_function_l1650_165075


namespace NUMINAMATH_CALUDE_max_value_sum_of_roots_l1650_165052

theorem max_value_sum_of_roots (x y z : ℝ) (h1 : x ≥ 0) (h2 : y ≥ 0) (h3 : z ≥ 0) (h4 : x + y + z = 5) :
  Real.sqrt (2 * x + 1) + Real.sqrt (2 * y + 1) + Real.sqrt (2 * z + 1) ≤ Real.sqrt 39 ∧
  ∃ x y z : ℝ, x ≥ 0 ∧ y ≥ 0 ∧ z ≥ 0 ∧ x + y + z = 5 ∧
    Real.sqrt (2 * x + 1) + Real.sqrt (2 * y + 1) + Real.sqrt (2 * z + 1) = Real.sqrt 39 :=
by sorry

end NUMINAMATH_CALUDE_max_value_sum_of_roots_l1650_165052


namespace NUMINAMATH_CALUDE_grasshopper_jumps_l1650_165079

-- Define a circle
def Circle := ℝ × ℝ → Prop

-- Define a point
def Point := ℝ × ℝ

-- Define a segment
def Segment := Point × Point

-- Define the property of being inside a circle
def InsideCircle (c : Circle) (p : Point) : Prop := sorry

-- Define the property of being on the boundary of a circle
def OnCircleBoundary (c : Circle) (p : Point) : Prop := sorry

-- Define the property of two segments not intersecting
def DoNotIntersect (s1 s2 : Segment) : Prop := sorry

-- Define the property of a point being reachable from another point
def Reachable (c : Circle) (points_inside : List Point) (points_boundary : List Point) (p q : Point) : Prop := sorry

theorem grasshopper_jumps 
  (c : Circle) 
  (n : ℕ) 
  (points_inside : List Point) 
  (points_boundary : List Point) 
  (h1 : points_inside.length = n) 
  (h2 : points_boundary.length = n)
  (h3 : ∀ p ∈ points_inside, InsideCircle c p)
  (h4 : ∀ p ∈ points_boundary, OnCircleBoundary c p)
  (h5 : ∀ (i j : Fin n), i ≠ j → 
    DoNotIntersect (points_inside[i], points_boundary[i]) (points_inside[j], points_boundary[j]))
  : ∀ (p q : Point), p ∈ points_inside → q ∈ points_inside → Reachable c points_inside points_boundary p q :=
sorry

end NUMINAMATH_CALUDE_grasshopper_jumps_l1650_165079


namespace NUMINAMATH_CALUDE_valid_arrangements_l1650_165036

/-- The number of ways to arrange 4 boys and 4 girls in a row with specific conditions -/
def arrangement_count : ℕ := 504

/-- The number of boys -/
def num_boys : ℕ := 4

/-- The number of girls -/
def num_girls : ℕ := 4

/-- The condition that adjacent individuals must be of opposite genders -/
def opposite_gender_adjacent : Prop := sorry

/-- The condition that a specific boy must stand next to a specific girl -/
def specific_pair_adjacent : Prop := sorry

/-- Theorem stating that the number of valid arrangements is 504 -/
theorem valid_arrangements :
  (num_boys = 4) →
  (num_girls = 4) →
  opposite_gender_adjacent →
  specific_pair_adjacent →
  arrangement_count = 504 := by
  sorry

end NUMINAMATH_CALUDE_valid_arrangements_l1650_165036


namespace NUMINAMATH_CALUDE_partnership_profit_l1650_165053

/-- Given two partners A and B in a business partnership, this theorem proves
    that the total profit is 7 times B's profit under certain conditions. -/
theorem partnership_profit
  (investment_A investment_B : ℝ)
  (period_A period_B : ℝ)
  (profit_B : ℝ)
  (h1 : investment_A = 3 * investment_B)
  (h2 : period_A = 2 * period_B)
  (h3 : profit_B = investment_B * period_B)
  : investment_A * period_A + investment_B * period_B = 7 * profit_B :=
by sorry

end NUMINAMATH_CALUDE_partnership_profit_l1650_165053


namespace NUMINAMATH_CALUDE_alicia_tax_payment_l1650_165092

/-- Calculates the total tax paid in cents per hour given an hourly wage and tax rates -/
def total_tax_cents (hourly_wage : ℝ) (local_tax_rate : ℝ) (state_tax_rate : ℝ) : ℝ :=
  hourly_wage * 100 * (local_tax_rate + state_tax_rate)

/-- Proves that Alicia's total tax paid is 62.5 cents per hour -/
theorem alicia_tax_payment :
  total_tax_cents 25 0.02 0.005 = 62.5 := by
  sorry

#eval total_tax_cents 25 0.02 0.005

end NUMINAMATH_CALUDE_alicia_tax_payment_l1650_165092


namespace NUMINAMATH_CALUDE_range_of_m_l1650_165030

theorem range_of_m (m : ℝ) : 
  (∃ x : ℝ, x^2 - 2*x - 1 < m^2 - 3*m) → m < 1 ∨ m > 2 := by
  sorry

end NUMINAMATH_CALUDE_range_of_m_l1650_165030


namespace NUMINAMATH_CALUDE_y_coord_comparison_l1650_165076

/-- Given two points on a line, prove that the y-coordinate of the left point is greater than the y-coordinate of the right point. -/
theorem y_coord_comparison (y₁ y₂ : ℝ) : 
  ((-4 : ℝ), y₁) ∈ {(x, y) | y = -1/2 * x + 2} →
  ((2 : ℝ), y₂) ∈ {(x, y) | y = -1/2 * x + 2} →
  y₁ > y₂ := by
  sorry

end NUMINAMATH_CALUDE_y_coord_comparison_l1650_165076


namespace NUMINAMATH_CALUDE_x_in_terms_of_y_and_k_l1650_165050

theorem x_in_terms_of_y_and_k (x y k : ℝ) :
  x / (x - k) = (y^2 + 3*y + 2) / (y^2 + 3*y + 1) →
  x = k*y^2 + 3*k*y + 2*k := by
  sorry

end NUMINAMATH_CALUDE_x_in_terms_of_y_and_k_l1650_165050


namespace NUMINAMATH_CALUDE_circles_intersect_l1650_165068

/-- Circle C1 with equation x^2 + y^2 - 2x - 3 = 0 -/
def C1 (x y : ℝ) : Prop := x^2 + y^2 - 2*x - 3 = 0

/-- Circle C2 with equation x^2 + y^2 - 4x + 2y + 4 = 0 -/
def C2 (x y : ℝ) : Prop := x^2 + y^2 - 4*x + 2*y + 4 = 0

/-- Two circles are intersecting if they have at least one point in common -/
def intersecting (C1 C2 : ℝ → ℝ → Prop) : Prop :=
  ∃ x y : ℝ, C1 x y ∧ C2 x y

/-- The circles C1 and C2 are intersecting -/
theorem circles_intersect : intersecting C1 C2 := by
  sorry

end NUMINAMATH_CALUDE_circles_intersect_l1650_165068


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1650_165001

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_sum (a : ℕ → ℝ) :
  arithmetic_sequence a →
  (a 2 + a 16 = 6) →
  (a 2 * a 16 = 1) →
  a 7 + a 8 + a 9 + a 10 + a 11 = 15 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1650_165001


namespace NUMINAMATH_CALUDE_other_root_of_quadratic_l1650_165060

theorem other_root_of_quadratic (c : ℝ) : 
  (∃ x : ℝ, 6 * x^2 + c * x = -3) → 
  (-1/2 : ℝ) ∈ {x : ℝ | 6 * x^2 + c * x = -3} →
  (-1 : ℝ) ∈ {x : ℝ | 6 * x^2 + c * x = -3} := by
sorry

end NUMINAMATH_CALUDE_other_root_of_quadratic_l1650_165060


namespace NUMINAMATH_CALUDE_league_teams_count_league_teams_count_proof_l1650_165003

theorem league_teams_count : ℕ → Prop :=
  fun n => (n * (n - 1) / 2 = 45) → n = 10

-- The proof is omitted
theorem league_teams_count_proof : league_teams_count 10 := by
  sorry

end NUMINAMATH_CALUDE_league_teams_count_league_teams_count_proof_l1650_165003


namespace NUMINAMATH_CALUDE_max_sum_of_squares_l1650_165031

theorem max_sum_of_squares (m n : ℕ) : 
  m ∈ Finset.range 1982 →
  n ∈ Finset.range 1982 →
  (n^2 - m*n - m^2)^2 = 1 →
  m^2 + n^2 ≤ 3524578 :=
by sorry

end NUMINAMATH_CALUDE_max_sum_of_squares_l1650_165031


namespace NUMINAMATH_CALUDE_equation_solutions_l1650_165056

def satisfies_equation (a b c : ℤ) : Prop :=
  (abs (a + 3) : ℤ) + b^2 + 4*c^2 - 14*b - 12*c + 55 = 0

def solution_set : Set (ℤ × ℤ × ℤ) :=
  {(-2, 8, 2), (-2, 6, 2), (-4, 8, 2), (-4, 6, 2), (-1, 7, 2), (-1, 7, 1), (-5, 7, 2), (-5, 7, 1)}

theorem equation_solutions :
  ∀ (a b c : ℤ), satisfies_equation a b c ↔ (a, b, c) ∈ solution_set :=
by sorry

end NUMINAMATH_CALUDE_equation_solutions_l1650_165056


namespace NUMINAMATH_CALUDE_solution_is_negative_two_l1650_165008

-- Define the equation
def fractional_equation (x : ℝ) : Prop :=
  x ≠ 0 ∧ x ≠ 2 ∧ (4 / (x - 2) = 2 / x)

-- Theorem statement
theorem solution_is_negative_two :
  ∃ (x : ℝ), fractional_equation x ∧ x = -2 :=
by sorry

end NUMINAMATH_CALUDE_solution_is_negative_two_l1650_165008


namespace NUMINAMATH_CALUDE_solution_set_implies_m_equals_two_l1650_165099

theorem solution_set_implies_m_equals_two (a : ℝ) (m : ℝ) :
  (∀ x : ℝ, a * x^2 - 6 * x + a^2 < 0 ↔ 1 < x ∧ x < m) →
  m = 2 :=
by sorry

end NUMINAMATH_CALUDE_solution_set_implies_m_equals_two_l1650_165099


namespace NUMINAMATH_CALUDE_no_infinite_sequence_positive_integers_l1650_165096

theorem no_infinite_sequence_positive_integers :
  ¬ ∃ (a : ℕ → ℕ+), ∀ (n : ℕ), (a (n-1))^2 ≥ 2 * (a n) * (a (n+2)) := by
  sorry

end NUMINAMATH_CALUDE_no_infinite_sequence_positive_integers_l1650_165096


namespace NUMINAMATH_CALUDE_min_distance_point_to_circle_l1650_165020

/-- The minimum distance from a point to a circle -/
theorem min_distance_point_to_circle (x y : ℝ) :
  (x - 1)^2 + y^2 = 4 →  -- Circle equation
  ∃ (d : ℝ), d = Real.sqrt 18 - 2 ∧ 
  ∀ (px py : ℝ), (px + 2)^2 + (py + 3)^2 ≥ d^2 := by
  sorry

end NUMINAMATH_CALUDE_min_distance_point_to_circle_l1650_165020


namespace NUMINAMATH_CALUDE_largest_difference_l1650_165028

theorem largest_difference (U V W X Y Z : ℕ) 
  (hU : U = 2 * 1002^1003)
  (hV : V = 1002^1003)
  (hW : W = 1001 * 1002^1002)
  (hX : X = 2 * 1002^1002)
  (hY : Y = 1002^1002)
  (hZ : Z = 1002^1001) :
  (U - V > V - W) ∧ 
  (U - V > W - X) ∧ 
  (U - V > X - Y) ∧ 
  (U - V > Y - Z) :=
sorry

end NUMINAMATH_CALUDE_largest_difference_l1650_165028


namespace NUMINAMATH_CALUDE_ap_sum_possible_n_values_l1650_165047

theorem ap_sum_possible_n_values :
  let S (n : ℕ) (a : ℤ) := (n : ℤ) * (2 * a + (n - 1) * 3) / 2
  (∃! k : ℕ, k > 1 ∧ (∃ a : ℤ, S k a = 180) ∧
    ∀ m : ℕ, m > 1 → (∃ b : ℤ, S m b = 180) → m ∈ Finset.range k) :=
by sorry

end NUMINAMATH_CALUDE_ap_sum_possible_n_values_l1650_165047


namespace NUMINAMATH_CALUDE_infinite_series_sum_l1650_165038

open Real

/-- The sum of the infinite series ∑(n=1 to ∞) (n + 1) / (n + 2)! is equal to e - 3 -/
theorem infinite_series_sum : 
  (∑' n : ℕ, (n + 1 : ℝ) / (n + 2).factorial) = Real.exp 1 - 3 := by
  sorry

end NUMINAMATH_CALUDE_infinite_series_sum_l1650_165038


namespace NUMINAMATH_CALUDE_abc_inequalities_l1650_165049

theorem abc_inequalities (a b c : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (sum_eq_one : a + b + c = 1) : 
  (2 * a * b + b * c + c * a + c^2 / 2 ≤ 1 / 2) ∧ 
  ((a^2 + c^2) / b + (b^2 + a^2) / c + (c^2 + b^2) / a ≥ 2) := by
  sorry

end NUMINAMATH_CALUDE_abc_inequalities_l1650_165049


namespace NUMINAMATH_CALUDE_film_rewinding_time_l1650_165010

/-- Time required to rewind a film onto a reel -/
theorem film_rewinding_time
  (a L S ω : ℝ)
  (ha : a > 0)
  (hL : L > 0)
  (hS : S > 0)
  (hω : ω > 0) :
  ∃ T : ℝ,
    T > 0 ∧
    T = (π / (S * ω)) * (Real.sqrt (a^2 + (4 * S * L / π)) - a) :=
by sorry

end NUMINAMATH_CALUDE_film_rewinding_time_l1650_165010


namespace NUMINAMATH_CALUDE_exhibition_average_l1650_165051

theorem exhibition_average : 
  let works : List ℕ := [58, 52, 58, 60]
  (works.sum / works.length : ℚ) = 57 := by sorry

end NUMINAMATH_CALUDE_exhibition_average_l1650_165051


namespace NUMINAMATH_CALUDE_seating_arrangements_with_restriction_l1650_165034

/-- The number of ways to arrange n people in a row -/
def totalArrangements (n : ℕ) : ℕ := n.factorial

/-- The number of ways to arrange n people in a row where 2 specific people must sit together -/
def restrictedArrangements (n : ℕ) : ℕ := (n - 1).factorial * 2

/-- The number of ways to arrange n people in a row where 2 specific people cannot sit together -/
def acceptableArrangements (n : ℕ) : ℕ := totalArrangements n - restrictedArrangements n

theorem seating_arrangements_with_restriction (n : ℕ) (hn : n = 9) :
  acceptableArrangements n = 282240 := by
  sorry

#eval acceptableArrangements 9

end NUMINAMATH_CALUDE_seating_arrangements_with_restriction_l1650_165034


namespace NUMINAMATH_CALUDE_multiplication_error_l1650_165084

theorem multiplication_error (a b : ℕ) (h1 : 100 ≤ a ∧ a < 1000) (h2 : 100 ≤ b ∧ b < 1000) :
  (∃ n : ℕ, 10000 * a + b = n * (a * b)) → (∃ n : ℕ, 10000 * a + b = 73 * (a * b)) :=
by sorry

end NUMINAMATH_CALUDE_multiplication_error_l1650_165084


namespace NUMINAMATH_CALUDE_equal_bills_at_20_minutes_l1650_165009

/-- Represents a telephone company with a base rate and per-minute charge. -/
structure TelephoneCompany where
  base_rate : ℝ
  per_minute_charge : ℝ

/-- Calculates the total cost for a given number of minutes. -/
def total_cost (company : TelephoneCompany) (minutes : ℝ) : ℝ :=
  company.base_rate + company.per_minute_charge * minutes

/-- The three telephone companies with their respective rates. -/
def united_telephone : TelephoneCompany := ⟨11, 0.25⟩
def atlantic_call : TelephoneCompany := ⟨12, 0.20⟩
def global_connect : TelephoneCompany := ⟨13, 0.15⟩

theorem equal_bills_at_20_minutes :
  ∃ (m : ℝ),
    m = 20 ∧
    total_cost united_telephone m = total_cost atlantic_call m ∧
    total_cost atlantic_call m = total_cost global_connect m :=
  sorry

end NUMINAMATH_CALUDE_equal_bills_at_20_minutes_l1650_165009


namespace NUMINAMATH_CALUDE_sum_of_three_consecutive_multiples_of_three_l1650_165073

theorem sum_of_three_consecutive_multiples_of_three (a b c : ℕ) : 
  a % 3 = 0 ∧ b % 3 = 0 ∧ c % 3 = 0 ∧  -- a, b, c are multiples of 3
  b = a + 3 ∧ c = b + 3 ∧               -- a, b, c are consecutive
  c = 27 →                              -- the largest number is 27
  a + b + c = 72 :=                     -- the sum is 72
by sorry

end NUMINAMATH_CALUDE_sum_of_three_consecutive_multiples_of_three_l1650_165073


namespace NUMINAMATH_CALUDE_toy_selection_proof_l1650_165033

def factorial (n : ℕ) : ℕ := sorry

def combinations (n r : ℕ) : ℕ := 
  factorial n / (factorial r * factorial (n - r))

theorem toy_selection_proof : 
  combinations 10 3 = 120 := by sorry

end NUMINAMATH_CALUDE_toy_selection_proof_l1650_165033


namespace NUMINAMATH_CALUDE_park_warden_citations_l1650_165007

theorem park_warden_citations :
  ∀ (littering off_leash parking : ℕ),
    littering = off_leash →
    parking = 2 * (littering + off_leash) →
    littering + off_leash + parking = 24 →
    littering = 4 := by
  sorry

end NUMINAMATH_CALUDE_park_warden_citations_l1650_165007


namespace NUMINAMATH_CALUDE_max_daily_profit_l1650_165065

/-- The daily profit function for a cake factory -/
def daily_profit (x : ℝ) : ℝ := 2000 * (-4 * x^2 + 3 * x + 10)

/-- The theorem stating the value of x that maximizes the daily profit -/
theorem max_daily_profit :
  ∃ (x : ℝ), 0 < x ∧ x < 1 ∧
  ∀ (y : ℝ), 0 < y ∧ y < 1 → daily_profit x ≥ daily_profit y ∧
  x = 0.375 := by
  sorry

end NUMINAMATH_CALUDE_max_daily_profit_l1650_165065


namespace NUMINAMATH_CALUDE_cubic_sum_inequality_l1650_165087

theorem cubic_sum_inequality (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  a^3 * b + b^3 * c + c^3 * a ≥ a * b * c * (a + b + c) := by
  sorry

end NUMINAMATH_CALUDE_cubic_sum_inequality_l1650_165087


namespace NUMINAMATH_CALUDE_prank_combinations_l1650_165042

theorem prank_combinations (monday tuesday wednesday thursday friday : ℕ) :
  monday = 3 →
  tuesday = 1 →
  wednesday = 6 →
  thursday = 4 →
  friday = 2 →
  monday * tuesday * wednesday * thursday * friday = 144 := by
  sorry

end NUMINAMATH_CALUDE_prank_combinations_l1650_165042


namespace NUMINAMATH_CALUDE_tan_x_equals_sqrt_three_l1650_165055

theorem tan_x_equals_sqrt_three (x : ℝ) 
  (h : Real.sin (x + Real.pi / 9) = Real.cos (x + Real.pi / 18) + Real.cos (x - Real.pi / 18)) : 
  Real.tan x = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_tan_x_equals_sqrt_three_l1650_165055


namespace NUMINAMATH_CALUDE_fiftieth_parentheses_sum_l1650_165005

/-- Represents the sum of numbers in a set of parentheses at a given position -/
def parenthesesSum (n : ℕ) : ℕ :=
  match n % 4 with
  | 1 => 1
  | 2 => 2 + 2
  | 3 => 3 + 3 + 3
  | 0 => 4 + 4 + 4 + 4
  | _ => 0  -- This case should never occur

/-- The sum of numbers in the 50th set of parentheses is 4 -/
theorem fiftieth_parentheses_sum : parenthesesSum 50 = 4 := by
  sorry

end NUMINAMATH_CALUDE_fiftieth_parentheses_sum_l1650_165005


namespace NUMINAMATH_CALUDE_cylinder_height_calculation_l1650_165086

/-- Given a cylinder with radius 3 units, if increasing the radius by 4 units
    and increasing the height by 10 units both result in the same volume increase,
    then the original height of the cylinder is 2.25 units. -/
theorem cylinder_height_calculation (h : ℝ) : 
  let r := 3
  let new_r := r + 4
  let new_h := h + 10
  let volume := π * r^2 * h
  let volume_after_radius_increase := π * new_r^2 * h
  let volume_after_height_increase := π * r^2 * new_h
  (volume_after_radius_increase - volume = volume_after_height_increase - volume) →
  h = 2.25 :=
by sorry

end NUMINAMATH_CALUDE_cylinder_height_calculation_l1650_165086


namespace NUMINAMATH_CALUDE_price_comparison_l1650_165070

theorem price_comparison (a : ℝ) (h : a > 0) : a * (1.1^5) * (0.9^5) < a := by
  sorry

end NUMINAMATH_CALUDE_price_comparison_l1650_165070


namespace NUMINAMATH_CALUDE_hiking_distance_sum_l1650_165063

theorem hiking_distance_sum : 
  let leg1 : ℝ := 3.8
  let leg2 : ℝ := 1.75
  let leg3 : ℝ := 2.3
  let leg4 : ℝ := 0.45
  let leg5 : ℝ := 1.92
  leg1 + leg2 + leg3 + leg4 + leg5 = 10.22 := by
  sorry

end NUMINAMATH_CALUDE_hiking_distance_sum_l1650_165063


namespace NUMINAMATH_CALUDE_square_area_side_ratio_l1650_165069

theorem square_area_side_ratio (a b : ℝ) (h : b ^ 2 = 16 * a ^ 2) : b = 4 * a := by
  sorry

end NUMINAMATH_CALUDE_square_area_side_ratio_l1650_165069
