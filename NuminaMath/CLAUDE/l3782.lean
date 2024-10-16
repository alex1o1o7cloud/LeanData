import Mathlib

namespace NUMINAMATH_CALUDE_arithmetic_sequence_problem_l3782_378218

theorem arithmetic_sequence_problem (a : ℕ → ℝ) :
  (∀ n m : ℕ, n < m → a n < a m) →  -- increasing sequence
  (a 4)^2 - 10 * (a 4) + 24 = 0 →   -- a_4 is a root
  (a 6)^2 - 10 * (a 6) + 24 = 0 →   -- a_6 is a root
  (∀ n : ℕ, a (n + 1) - a n = a (n + 2) - a (n + 1)) →  -- arithmetic sequence
  a 20 = 20 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_problem_l3782_378218


namespace NUMINAMATH_CALUDE_yoongi_total_carrots_l3782_378284

/-- The number of carrots Yoongi has -/
def yoongi_carrots (initial : ℕ) (from_sister : ℕ) : ℕ :=
  initial + from_sister

theorem yoongi_total_carrots :
  yoongi_carrots 3 2 = 5 := by
  sorry

end NUMINAMATH_CALUDE_yoongi_total_carrots_l3782_378284


namespace NUMINAMATH_CALUDE_independence_day_banana_distribution_l3782_378213

theorem independence_day_banana_distribution :
  ∀ (total_children : ℕ) (total_bananas : ℕ),
    (2 * total_children = total_bananas) →
    (4 * (total_children - 390) = total_bananas) →
    total_children = 780 := by
  sorry

end NUMINAMATH_CALUDE_independence_day_banana_distribution_l3782_378213


namespace NUMINAMATH_CALUDE_no_half_parallel_diagonals_l3782_378292

/-- A regular polygon with n sides -/
structure RegularPolygon where
  n : ℕ
  h : n > 2

/-- The number of diagonals in a polygon -/
def numDiagonals (p : RegularPolygon) : ℕ :=
  p.n * (p.n - 3) / 2

/-- The number of diagonals parallel to sides in a polygon -/
def numParallelDiagonals (p : RegularPolygon) : ℕ :=
  if p.n % 2 = 1 then numDiagonals p else (p.n / 2) - 1

/-- Theorem: No regular polygon has exactly half of its diagonals parallel to its sides -/
theorem no_half_parallel_diagonals (p : RegularPolygon) :
  2 * numParallelDiagonals p ≠ numDiagonals p :=
sorry

end NUMINAMATH_CALUDE_no_half_parallel_diagonals_l3782_378292


namespace NUMINAMATH_CALUDE_rectangle_circle_area_ratio_l3782_378211

/-- Given a rectangle with perimeter equal to the circumference of a circle,
    and the length of the rectangle is twice its width,
    prove that the ratio of the area of the rectangle to the area of the circle is 2π/9. -/
theorem rectangle_circle_area_ratio (w : ℝ) (r : ℝ) (h1 : w > 0) (h2 : r > 0) :
  let l := 2 * w
  let rectangle_perimeter := 2 * l + 2 * w
  let circle_circumference := 2 * Real.pi * r
  let rectangle_area := l * w
  let circle_area := Real.pi * r^2
  rectangle_perimeter = circle_circumference →
  rectangle_area / circle_area = 2 * Real.pi / 9 := by
sorry

end NUMINAMATH_CALUDE_rectangle_circle_area_ratio_l3782_378211


namespace NUMINAMATH_CALUDE_range_of_m_l3782_378259

open Real

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := -a * log x + x + (1 - a) / x

noncomputable def g (m : ℝ) (x : ℝ) : ℝ := exp x + m * x^2 - 2 * exp 2 - 3

theorem range_of_m :
  ∀ m : ℝ, (∃ x₂ : ℝ, x₂ ≥ 1 ∧ ∀ x₁ : ℝ, x₁ ≥ 1 → g m x₂ ≤ f (exp 2 + 1) x₁) ↔ m ≤ exp 2 - exp 1 :=
by sorry

end NUMINAMATH_CALUDE_range_of_m_l3782_378259


namespace NUMINAMATH_CALUDE_min_value_expression_l3782_378268

theorem min_value_expression (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  (4 * z) / (2 * x + y) + (4 * x) / (y + 2 * z) + y / (x + z) ≥ 3 ∧
  ((4 * z) / (2 * x + y) + (4 * x) / (y + 2 * z) + y / (x + z) = 3 ↔ 2 * x = y ∧ y = 2 * z) := by
sorry

end NUMINAMATH_CALUDE_min_value_expression_l3782_378268


namespace NUMINAMATH_CALUDE_sequence_inequality_l3782_378254

theorem sequence_inequality (n : ℕ) (a : ℕ → ℝ) 
  (h0 : a 0 = 0) 
  (hn : a (n + 1) = 0)
  (h : ∀ k : ℕ, k ≥ 1 → k ≤ n → |a (k - 1) - 2 * a k + a (k + 1)| ≤ 1) :
  ∀ k : ℕ, k ≤ n + 1 → |a k| ≤ k * (n + 1 - k) / 2 :=
by sorry

end NUMINAMATH_CALUDE_sequence_inequality_l3782_378254


namespace NUMINAMATH_CALUDE_milk_water_ratio_problem_l3782_378220

/-- Given two vessels with volumes in ratio 3:5, where the first vessel has a milk to water ratio
    of 1:2, and when mixed the overall milk to water ratio is 1:1, prove that the milk to water
    ratio in the second vessel must be 3:2. -/
theorem milk_water_ratio_problem (v : ℝ) (x y : ℝ) (h_x_pos : x > 0) (h_y_pos : y > 0) : 
  (1 : ℝ) + (5 * x) / (x + y) = (2 : ℝ) + (5 * y) / (x + y) → x / y = 3 / 2 :=
by sorry

end NUMINAMATH_CALUDE_milk_water_ratio_problem_l3782_378220


namespace NUMINAMATH_CALUDE_sam_distance_l3782_378224

/-- Given that Marguerite drove 150 miles in 3 hours, and Sam drove for 4 hours at the same average rate as Marguerite, prove that Sam drove 200 miles. -/
theorem sam_distance (marguerite_distance : ℝ) (marguerite_time : ℝ) (sam_time : ℝ)
  (h1 : marguerite_distance = 150)
  (h2 : marguerite_time = 3)
  (h3 : sam_time = 4) :
  (marguerite_distance / marguerite_time) * sam_time = 200 :=
by sorry

end NUMINAMATH_CALUDE_sam_distance_l3782_378224


namespace NUMINAMATH_CALUDE_equation_solutions_l3782_378245

theorem equation_solutions :
  (∀ x : ℝ, x^2 - 2*x = 1 ↔ x = 1 + Real.sqrt 2 ∨ x = 1 - Real.sqrt 2) ∧
  (∀ x : ℝ, x^2 + 5*x + 6 = 0 ↔ x = -2 ∨ x = -3) := by
  sorry

end NUMINAMATH_CALUDE_equation_solutions_l3782_378245


namespace NUMINAMATH_CALUDE_garden_length_l3782_378215

/-- Given a rectangular garden with perimeter 1200 m and breadth 240 m, prove its length is 360 m -/
theorem garden_length (perimeter : ℝ) (breadth : ℝ) (length : ℝ)
  (h1 : perimeter = 1200)
  (h2 : breadth = 240)
  (h3 : perimeter = 2 * length + 2 * breadth) :
  length = 360 :=
by sorry

end NUMINAMATH_CALUDE_garden_length_l3782_378215


namespace NUMINAMATH_CALUDE_horner_method_f_2_l3782_378246

def f (x : ℝ) : ℝ := 2 * x^4 + 3 * x^3 + 5 * x - 4

theorem horner_method_f_2 : f 2 = 62 := by
  sorry

end NUMINAMATH_CALUDE_horner_method_f_2_l3782_378246


namespace NUMINAMATH_CALUDE_file_app_difference_l3782_378289

/-- Given initial and final counts of apps and files on a phone, 
    prove the difference between final files and apps --/
theorem file_app_difference 
  (initial_apps : ℕ) 
  (initial_files : ℕ) 
  (final_apps : ℕ) 
  (final_files : ℕ) 
  (h1 : initial_apps = 11) 
  (h2 : initial_files = 3) 
  (h3 : final_apps = 2) 
  (h4 : final_files = 24) : 
  final_files - final_apps = 22 := by
  sorry

end NUMINAMATH_CALUDE_file_app_difference_l3782_378289


namespace NUMINAMATH_CALUDE_cuboid_dimensions_l3782_378294

theorem cuboid_dimensions (x y v : ℕ) 
  (h1 : x * y * v - v = 602)
  (h2 : x * y * v - x = 605)
  (h3 : v = x + 3)
  (hx : x > 0)
  (hy : y > 0)
  (hv : v > 0) :
  x = 11 ∧ y = 4 ∧ v = 14 := by
sorry

end NUMINAMATH_CALUDE_cuboid_dimensions_l3782_378294


namespace NUMINAMATH_CALUDE_min_victory_points_l3782_378237

/-- Represents the point system for a football competition --/
structure PointSystem where
  victory_points : ℕ
  draw_points : ℕ
  defeat_points : ℕ

/-- Represents the state of a team's performance --/
structure TeamPerformance where
  total_matches : ℕ
  played_matches : ℕ
  current_points : ℕ
  target_points : ℕ
  min_victories_needed : ℕ

/-- The theorem to prove --/
theorem min_victory_points (ps : PointSystem) (tp : TeamPerformance) : 
  ps.draw_points = 1 ∧ 
  ps.defeat_points = 0 ∧
  tp.total_matches = 20 ∧ 
  tp.played_matches = 5 ∧
  tp.current_points = 14 ∧
  tp.target_points = 40 ∧
  tp.min_victories_needed = 6 →
  ps.victory_points ≥ 3 :=
by sorry

end NUMINAMATH_CALUDE_min_victory_points_l3782_378237


namespace NUMINAMATH_CALUDE_book_price_proof_l3782_378261

theorem book_price_proof (selling_price : ℝ) (profit_percentage : ℝ) 
  (h1 : selling_price = 63)
  (h2 : profit_percentage = 5) :
  ∃ original_price : ℝ, 
    original_price * (1 + profit_percentage / 100) = selling_price ∧ 
    original_price = 60 := by
  sorry

end NUMINAMATH_CALUDE_book_price_proof_l3782_378261


namespace NUMINAMATH_CALUDE_necklace_diamond_count_l3782_378285

theorem necklace_diamond_count (total_necklaces : ℕ) (total_diamonds : ℕ) : 
  total_necklaces = 20 →
  total_diamonds = 79 →
  ∃ (two_diamond_necklaces five_diamond_necklaces : ℕ),
    two_diamond_necklaces + five_diamond_necklaces = total_necklaces ∧
    2 * two_diamond_necklaces + 5 * five_diamond_necklaces = total_diamonds ∧
    five_diamond_necklaces = 13 := by
  sorry

end NUMINAMATH_CALUDE_necklace_diamond_count_l3782_378285


namespace NUMINAMATH_CALUDE_arithmetic_sequence_a6_l3782_378217

/-- An arithmetic sequence with positive terms -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, q > 0 ∧ ∀ n, a n > 0 ∧ a (n + 1) = a n * q

theorem arithmetic_sequence_a6 (a : ℕ → ℝ) 
    (h_arithmetic : ArithmeticSequence a)
    (h_a1 : a 1 = 1)
    (h_a2a4 : a 2 * a 4 = 16) :
    a 6 = 32 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_a6_l3782_378217


namespace NUMINAMATH_CALUDE_complex_number_properties_l3782_378247

theorem complex_number_properties (z : ℂ) (h : z * (2 + Complex.I) = Complex.I ^ 10) :
  (Complex.abs z = Real.sqrt 5 / 5) ∧
  (Complex.re z < 0 ∧ Complex.im z > 0) :=
sorry

end NUMINAMATH_CALUDE_complex_number_properties_l3782_378247


namespace NUMINAMATH_CALUDE_sine_amplitude_negative_a_l3782_378270

theorem sine_amplitude_negative_a (a b : ℝ) (h1 : a < 0) (h2 : b > 0) :
  (∀ x, ∃ y, y = a * Real.sin (b * x)) →
  (∀ x, a * Real.sin (b * x) ≥ -2 ∧ a * Real.sin (b * x) ≤ 0) →
  (∃ x, a * Real.sin (b * x) = -2) →
  a = -2 := by
sorry

end NUMINAMATH_CALUDE_sine_amplitude_negative_a_l3782_378270


namespace NUMINAMATH_CALUDE_glass_bowl_selling_price_l3782_378256

theorem glass_bowl_selling_price 
  (total_bowls : ℕ) 
  (cost_per_bowl : ℚ) 
  (bowls_sold : ℕ) 
  (percentage_gain : ℚ) 
  (h1 : total_bowls = 118) 
  (h2 : cost_per_bowl = 12) 
  (h3 : bowls_sold = 102) 
  (h4 : percentage_gain = 8050847457627118 / 100000000000000000) : 
  ∃ (selling_price : ℚ), selling_price = 15 ∧ 
  (total_bowls * cost_per_bowl * (1 + percentage_gain) / bowls_sold).floor = selling_price := by
  sorry

#check glass_bowl_selling_price

end NUMINAMATH_CALUDE_glass_bowl_selling_price_l3782_378256


namespace NUMINAMATH_CALUDE_triangle_shape_l3782_378278

theorem triangle_shape (A B C : ℝ) (hA : 0 < A) (hB : 0 < B) (hC : 0 < C) 
  (hSum : A + B + C = π) (hSine : 2 * Real.sin B * Real.cos C = Real.sin A) :
  ∃ (a b c : ℝ), a > 0 ∧ b > 0 ∧ c > 0 ∧ 
  a / Real.sin A = b / Real.sin B ∧
  b / Real.sin B = c / Real.sin C ∧
  b = c :=
sorry

end NUMINAMATH_CALUDE_triangle_shape_l3782_378278


namespace NUMINAMATH_CALUDE_origin_inside_ellipse_iff_k_range_l3782_378232

/-- The ellipse equation -/
def ellipse (k x y : ℝ) : Prop :=
  k^2 * x^2 + y^2 - 4*k*x + 2*k*y + k^2 - 1 = 0

/-- A point (x,y) is inside the ellipse if the left side of the equation is negative -/
def inside_ellipse (k x y : ℝ) : Prop :=
  k^2 * x^2 + y^2 - 4*k*x + 2*k*y + k^2 - 1 < 0

theorem origin_inside_ellipse_iff_k_range :
  ∀ k : ℝ, inside_ellipse k 0 0 ↔ (0 < |k| ∧ |k| < 1) :=
by sorry

end NUMINAMATH_CALUDE_origin_inside_ellipse_iff_k_range_l3782_378232


namespace NUMINAMATH_CALUDE_sum_of_reciprocal_equations_l3782_378214

theorem sum_of_reciprocal_equations (x y : ℚ) 
  (h1 : 1/x + 1/y = 4)
  (h2 : 1/x - 1/y = -3) :
  x + y = 16/7 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_reciprocal_equations_l3782_378214


namespace NUMINAMATH_CALUDE_second_share_interest_rate_l3782_378249

theorem second_share_interest_rate 
  (total_investment : ℝ) 
  (first_share_yield : ℝ) 
  (total_interest_rate : ℝ) 
  (second_share_amount : ℝ) :
  total_investment = 100000 →
  first_share_yield = 0.09 →
  total_interest_rate = 0.0925 →
  second_share_amount = 12500 →
  let first_share_amount := total_investment - second_share_amount
  let total_interest := total_interest_rate * total_investment
  let first_share_interest := first_share_yield * first_share_amount
  let second_share_interest := total_interest - first_share_interest
  second_share_interest / second_share_amount = 0.11 := by
  sorry

end NUMINAMATH_CALUDE_second_share_interest_rate_l3782_378249


namespace NUMINAMATH_CALUDE_multiply_by_fraction_l3782_378295

theorem multiply_by_fraction (a b c : ℝ) (h : a * b = c) :
  (b / 10) * a = c / 10 := by
  sorry

end NUMINAMATH_CALUDE_multiply_by_fraction_l3782_378295


namespace NUMINAMATH_CALUDE_ceiling_floor_product_l3782_378203

theorem ceiling_floor_product (y : ℝ) : 
  y < 0 → ⌈y⌉ * ⌊y⌋ = 132 → -12 < y ∧ y < -11 := by
  sorry

end NUMINAMATH_CALUDE_ceiling_floor_product_l3782_378203


namespace NUMINAMATH_CALUDE_fourth_person_height_l3782_378258

/-- Given four people with heights in increasing order, prove the height of the fourth person. -/
theorem fourth_person_height (h₁ h₂ h₃ h₄ : ℝ) : 
  h₁ < h₂ ∧ h₂ < h₃ ∧ h₃ < h₄ →  -- Heights in increasing order
  h₂ - h₁ = 2 →  -- Difference between 1st and 2nd
  h₃ - h₂ = 2 →  -- Difference between 2nd and 3rd
  h₄ - h₃ = 6 →  -- Difference between 3rd and 4th
  (h₁ + h₂ + h₃ + h₄) / 4 = 79 →  -- Average height
  h₄ = 85 :=
by sorry

end NUMINAMATH_CALUDE_fourth_person_height_l3782_378258


namespace NUMINAMATH_CALUDE_total_shells_count_l3782_378282

def purple_shells : ℕ := 13
def pink_shells : ℕ := 8
def yellow_shells : ℕ := 18
def blue_shells : ℕ := 12
def orange_shells : ℕ := 14

theorem total_shells_count :
  purple_shells + pink_shells + yellow_shells + blue_shells + orange_shells = 65 := by
  sorry

end NUMINAMATH_CALUDE_total_shells_count_l3782_378282


namespace NUMINAMATH_CALUDE_smallest_number_above_50_with_conditions_fifty_one_satisfies_conditions_fifty_one_is_answer_l3782_378253

def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, n = m * m

def count_factors (n : ℕ) : ℕ := (Finset.filter (· ∣ n) (Finset.range (n + 1))).card

theorem smallest_number_above_50_with_conditions : 
  ∀ n : ℕ, n > 50 → n < 51 → 
  (¬ is_perfect_square n ∨ count_factors n % 2 = 1 ∨ n % 3 ≠ 0) :=
by sorry

theorem fifty_one_satisfies_conditions : 
  ¬ is_perfect_square 51 ∧ count_factors 51 % 2 = 0 ∧ 51 % 3 = 0 :=
by sorry

theorem fifty_one_is_answer : 
  ∀ n : ℕ, n > 50 → n < 51 → 
  (¬ is_perfect_square n ∨ count_factors n % 2 = 1 ∨ n % 3 ≠ 0) ∧
  (¬ is_perfect_square 51 ∧ count_factors 51 % 2 = 0 ∧ 51 % 3 = 0) :=
by sorry

end NUMINAMATH_CALUDE_smallest_number_above_50_with_conditions_fifty_one_satisfies_conditions_fifty_one_is_answer_l3782_378253


namespace NUMINAMATH_CALUDE_periodic_function_value_l3782_378248

/-- Given a function f(x) = a*sin(π*x + α) + b*cos(π*x + β) + 4,
    where a, b, α, β are non-zero real numbers, and f(2007) = 5,
    prove that f(2008) = 3 -/
theorem periodic_function_value (a b α β : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hα : α ≠ 0) (hβ : β ≠ 0) :
  let f : ℝ → ℝ := λ x ↦ a * Real.sin (π * x + α) + b * Real.cos (π * x + β) + 4
  f 2007 = 5 → f 2008 = 3 := by
  sorry

end NUMINAMATH_CALUDE_periodic_function_value_l3782_378248


namespace NUMINAMATH_CALUDE_cyclist_problem_l3782_378226

theorem cyclist_problem (v t : ℝ) 
  (h1 : (v + 3) * (t - 1) = v * t)
  (h2 : (v - 2) * (t + 1) = v * t) :
  v * t = 60 ∧ v = 12 ∧ t = 5 :=
by sorry

end NUMINAMATH_CALUDE_cyclist_problem_l3782_378226


namespace NUMINAMATH_CALUDE_spade_nested_operation_l3782_378263

def spade (a b : ℝ) : ℝ := |a - b|

theorem spade_nested_operation : spade 5 (spade 3 9) = 1 := by
  sorry

end NUMINAMATH_CALUDE_spade_nested_operation_l3782_378263


namespace NUMINAMATH_CALUDE_parabola_translation_l3782_378275

/-- The equation of a parabola translated upwards by 1 unit from y = x^2 -/
theorem parabola_translation (x y : ℝ) : 
  (y = x^2) → (∃ y', y' = y + 1 ∧ y' = x^2 + 1) :=
by sorry

end NUMINAMATH_CALUDE_parabola_translation_l3782_378275


namespace NUMINAMATH_CALUDE_pentagon_rectangle_ratio_l3782_378274

/-- Given a regular pentagon and a rectangle with perimeters of 75 inches,
    where the rectangle's length is twice its width, prove that the ratio of
    the side length of the pentagon to the width of the rectangle is 6/5. -/
theorem pentagon_rectangle_ratio :
  ∀ (pentagon_side : ℚ) (rect_width : ℚ),
    -- Pentagon perimeter is 75 inches
    5 * pentagon_side = 75 →
    -- Rectangle perimeter is 75 inches, and length is twice the width
    2 * (rect_width + 2 * rect_width) = 75 →
    -- The ratio of pentagon side to rectangle width is 6/5
    pentagon_side / rect_width = 6 / 5 := by
  sorry

end NUMINAMATH_CALUDE_pentagon_rectangle_ratio_l3782_378274


namespace NUMINAMATH_CALUDE_f_ratio_range_l3782_378267

-- Define the function f
noncomputable def f : ℝ → ℝ := sorry

-- Define the derivative of f
noncomputable def f' : ℝ → ℝ := sorry

-- State the theorem
theorem f_ratio_range :
  (∀ x : ℝ, f' x - f x = 2 * x * Real.exp x) →
  f 0 = 1 →
  ∀ x : ℝ, x > 0 → 1 < (f' x) / (f x) ∧ (f' x) / (f x) ≤ 2 :=
by sorry

end NUMINAMATH_CALUDE_f_ratio_range_l3782_378267


namespace NUMINAMATH_CALUDE_sum_in_base4_l3782_378229

-- Define a function to convert from base 10 to base 4
def toBase4 (n : ℕ) : List ℕ :=
  sorry

-- Define a function to convert from base 4 to base 10
def fromBase4 (l : List ℕ) : ℕ :=
  sorry

theorem sum_in_base4 : 
  toBase4 (195 + 61) = [1, 0, 0, 0, 0] :=
sorry

end NUMINAMATH_CALUDE_sum_in_base4_l3782_378229


namespace NUMINAMATH_CALUDE_class_average_problem_l3782_378286

/-- Given a class of 50 students with an overall average of 92 and the first 30 students
    having an average of 90, the average of the remaining 20 students is 95. -/
theorem class_average_problem :
  ∀ (total_score first_group_score last_group_score : ℝ),
  (50 : ℝ) * 92 = total_score →
  (30 : ℝ) * 90 = first_group_score →
  total_score = first_group_score + last_group_score →
  last_group_score / (20 : ℝ) = 95 := by
sorry

end NUMINAMATH_CALUDE_class_average_problem_l3782_378286


namespace NUMINAMATH_CALUDE_website_earnings_l3782_378281

/-- Calculates daily earnings for a website given monthly visits, days in a month, and earnings per visit -/
def daily_earnings (monthly_visits : ℕ) (days_in_month : ℕ) (earnings_per_visit : ℚ) : ℚ :=
  (monthly_visits : ℚ) / (days_in_month : ℚ) * earnings_per_visit

/-- Proves that given 30000 monthly visits in a 30-day month with $0.01 earnings per visit, daily earnings are $10 -/
theorem website_earnings : daily_earnings 30000 30 (1/100) = 10 := by
  sorry

end NUMINAMATH_CALUDE_website_earnings_l3782_378281


namespace NUMINAMATH_CALUDE_sqrt_equation_sum_l3782_378280

theorem sqrt_equation_sum (a t : ℝ) (ha : a > 0) (ht : t > 0) :
  Real.sqrt (6 + a / t) = 6 * Real.sqrt (a / t) → t + a = 41 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_sum_l3782_378280


namespace NUMINAMATH_CALUDE_P_symmetric_l3782_378231

/-- Definition of the polynomial sequence P_m -/
def P : ℕ → (ℚ → ℚ → ℚ → ℚ)
| 0 => λ _ _ _ => 1
| (m + 1) => λ x y z => (x + z) * (y + z) * P m x y (z + 1) - z^2 * P m x y z

/-- Statement that P_m is symmetric for all m -/
theorem P_symmetric (m : ℕ) (x y z : ℚ) :
  P m x y z = P m y x z ∧
  P m x y z = P m x z y ∧
  P m x y z = P m y z x ∧
  P m x y z = P m z x y ∧
  P m x y z = P m z y x :=
by sorry

end NUMINAMATH_CALUDE_P_symmetric_l3782_378231


namespace NUMINAMATH_CALUDE_line_equation_through_point_with_angle_45_l3782_378260

/-- The equation of a line passing through (-4, 3) with a slope angle of 45° -/
theorem line_equation_through_point_with_angle_45 :
  ∃ (f : ℝ → ℝ),
    (∀ x y, f x = y ↔ x - y + 7 = 0) ∧
    f (-4) = 3 ∧
    (∀ x, (f x - f (-4)) / (x - (-4)) = 1) :=
by sorry

end NUMINAMATH_CALUDE_line_equation_through_point_with_angle_45_l3782_378260


namespace NUMINAMATH_CALUDE_aunt_marge_candy_distribution_l3782_378269

theorem aunt_marge_candy_distribution (total_candy : ℕ) 
  (kate_candy : ℕ) (robert_candy : ℕ) (mary_candy : ℕ) (bill_candy : ℕ) : 
  total_candy = 20 ∧ 
  robert_candy = kate_candy + 2 ∧
  bill_candy = mary_candy - 6 ∧
  mary_candy = robert_candy + 2 ∧
  kate_candy = bill_candy + 2 ∧
  total_candy = kate_candy + robert_candy + mary_candy + bill_candy →
  kate_candy = 4 := by
sorry

end NUMINAMATH_CALUDE_aunt_marge_candy_distribution_l3782_378269


namespace NUMINAMATH_CALUDE_ball_attendees_l3782_378221

theorem ball_attendees :
  ∀ (n m : ℕ),
  n + m < 50 →
  (3 * n) / 4 = (5 * m) / 7 →
  n + m = 41 :=
by
  sorry

end NUMINAMATH_CALUDE_ball_attendees_l3782_378221


namespace NUMINAMATH_CALUDE_cylinder_volume_change_l3782_378279

/-- Given a cylinder with original volume of 15 cubic feet, prove that tripling its radius and doubling its height results in a new volume of 270 cubic feet. -/
theorem cylinder_volume_change (r h : ℝ) : 
  r > 0 → h > 0 → π * r^2 * h = 15 → π * (3*r)^2 * (2*h) = 270 := by
  sorry

end NUMINAMATH_CALUDE_cylinder_volume_change_l3782_378279


namespace NUMINAMATH_CALUDE_transformed_curve_equation_l3782_378296

-- Define the original ellipse
def original_ellipse (x y : ℝ) : Prop := x^2 / 4 + y^2 = 1

-- Define the transformation
def transform (x y x' y' : ℝ) : Prop := x' = x ∧ y' = 2 * y

-- State the theorem
theorem transformed_curve_equation :
  ∀ x y x' y' : ℝ, original_ellipse x y → transform x y x' y' →
  x'^2 + y'^2 = 4 :=
sorry

end NUMINAMATH_CALUDE_transformed_curve_equation_l3782_378296


namespace NUMINAMATH_CALUDE_square_greater_than_l3782_378266

theorem square_greater_than (a b : ℝ) (h1 : a > b) (h2 : b > 0) : a^2 > b^2 := by
  sorry

end NUMINAMATH_CALUDE_square_greater_than_l3782_378266


namespace NUMINAMATH_CALUDE_rachel_math_problems_l3782_378234

def problems_per_minute : ℕ := 5
def minutes_solved : ℕ := 12
def problems_second_day : ℕ := 16

theorem rachel_math_problems :
  problems_per_minute * minutes_solved + problems_second_day = 76 := by
  sorry

end NUMINAMATH_CALUDE_rachel_math_problems_l3782_378234


namespace NUMINAMATH_CALUDE_product_expansion_l3782_378287

theorem product_expansion (x : ℝ) : (x^2 - 3*x + 3) * (x^2 + 3*x + 1) = x^4 - 5*x^2 + 6*x + 3 := by
  sorry

end NUMINAMATH_CALUDE_product_expansion_l3782_378287


namespace NUMINAMATH_CALUDE_sector_area_l3782_378277

theorem sector_area (θ : Real) (r : Real) (h1 : θ = 72 * π / 180) (h2 : r = 20) :
  (1 / 2) * θ * r^2 = 80 * π := by
  sorry

end NUMINAMATH_CALUDE_sector_area_l3782_378277


namespace NUMINAMATH_CALUDE_buffalo_count_l3782_378252

/-- A group of animals consisting of buffaloes and ducks -/
structure AnimalGroup where
  buffaloes : ℕ
  ducks : ℕ

/-- The total number of legs in the group -/
def total_legs (group : AnimalGroup) : ℕ := 4 * group.buffaloes + 2 * group.ducks

/-- The total number of heads in the group -/
def total_heads (group : AnimalGroup) : ℕ := group.buffaloes + group.ducks

/-- The main theorem: there are 12 buffaloes in the group -/
theorem buffalo_count (group : AnimalGroup) : 
  (total_legs group = 2 * total_heads group + 24) → group.buffaloes = 12 := by
  sorry

end NUMINAMATH_CALUDE_buffalo_count_l3782_378252


namespace NUMINAMATH_CALUDE_smallest_m_for_partition_l3782_378210

theorem smallest_m_for_partition (r : ℕ+) :
  (∃ (m : ℕ+), ∀ (A : Fin r → Set ℕ),
    (∀ (i j : Fin r), i ≠ j → A i ∩ A j = ∅) →
    (⋃ (i : Fin r), A i) = Finset.range m →
    (∃ (k : Fin r) (a b : ℕ), a ∈ A k ∧ b ∈ A k ∧ a ≠ 0 ∧ 1 ≤ b / a ∧ b / a ≤ 1 + 1 / 2022)) ∧
  (∀ (m : ℕ+), m < 2023 * r →
    ¬(∀ (A : Fin r → Set ℕ),
      (∀ (i j : Fin r), i ≠ j → A i ∩ A j = ∅) →
      (⋃ (i : Fin r), A i) = Finset.range m →
      (∃ (k : Fin r) (a b : ℕ), a ∈ A k ∧ b ∈ A k ∧ a ≠ 0 ∧ 1 ≤ b / a ∧ b / a ≤ 1 + 1 / 2022))) :=
by sorry

end NUMINAMATH_CALUDE_smallest_m_for_partition_l3782_378210


namespace NUMINAMATH_CALUDE_min_garden_cost_is_108_l3782_378240

/-- Represents the cost of each flower type in dollars -/
structure FlowerCost where
  asters : ℝ
  begonias : ℝ
  cannas : ℝ
  dahlias : ℝ
  easterLilies : ℝ

/-- Represents the dimensions of each region in the flower bed -/
structure RegionDimensions where
  region1 : ℝ × ℝ
  region2 : ℝ × ℝ
  region3 : ℝ × ℝ
  region4 : ℝ × ℝ
  region5 : ℝ × ℝ

/-- Calculates the minimum cost of the garden given the flower costs and region dimensions -/
def minGardenCost (costs : FlowerCost) (dimensions : RegionDimensions) : ℝ :=
  sorry

/-- Theorem stating that the minimum cost of the garden is $108 -/
theorem min_garden_cost_is_108 (costs : FlowerCost) (dimensions : RegionDimensions) :
  costs.asters = 1 ∧ 
  costs.begonias = 1.5 ∧ 
  costs.cannas = 2 ∧ 
  costs.dahlias = 2.5 ∧ 
  costs.easterLilies = 3 ∧
  dimensions.region1 = (3, 4) ∧
  dimensions.region2 = (2, 3) ∧
  dimensions.region3 = (3, 5) ∧
  dimensions.region4 = (4, 5) ∧
  dimensions.region5 = (3, 7) →
  minGardenCost costs dimensions = 108 :=
by
  sorry

end NUMINAMATH_CALUDE_min_garden_cost_is_108_l3782_378240


namespace NUMINAMATH_CALUDE_log_difference_equals_one_l3782_378293

-- Define the logarithm function
noncomputable def log (a : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log a

-- State the theorem
theorem log_difference_equals_one (a : ℝ) 
  (h1 : a > 0) 
  (h2 : a ≠ 1) 
  (h3 : log a 3 > log a 2) : 
  (log a (2 * a) - log a a = 1) → a = 2 := by
  sorry

end NUMINAMATH_CALUDE_log_difference_equals_one_l3782_378293


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l3782_378244

/-- A geometric sequence with common ratio 2 and sum of first 3 terms equal to 21 -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  (∀ n, a (n + 1) = 2 * a n) ∧ (a 1 + a 2 + a 3 = 21)

/-- The sum of the 3rd, 4th, and 5th terms of the geometric sequence is 84 -/
theorem geometric_sequence_sum (a : ℕ → ℝ) (h : GeometricSequence a) :
  a 3 + a 4 + a 5 = 84 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l3782_378244


namespace NUMINAMATH_CALUDE_number_of_students_l3782_378288

/-- Given a class where:
    1. The initial average marks of students is 100.
    2. A student's mark is wrongly noted as 50 instead of 10.
    3. The correct average marks is 96.
    Prove that the number of students in the class is 10. -/
theorem number_of_students (n : ℕ) 
    (h1 : (100 * n) / n = 100)  -- Initial average is 100
    (h2 : (100 * n - 40) / n = 96)  -- Correct average is 96
    : n = 10 := by
  sorry


end NUMINAMATH_CALUDE_number_of_students_l3782_378288


namespace NUMINAMATH_CALUDE_cos_sin_inequality_range_l3782_378276

theorem cos_sin_inequality_range (θ : Real) :
  θ ∈ Set.Icc 0 (2 * Real.pi) →
  Real.cos θ ^ 5 - Real.sin θ ^ 5 < 7 * (Real.sin θ ^ 3 - Real.cos θ ^ 3) →
  θ ∈ Set.Ioo (Real.pi / 4) (5 * Real.pi / 4) :=
by sorry

end NUMINAMATH_CALUDE_cos_sin_inequality_range_l3782_378276


namespace NUMINAMATH_CALUDE_solution_set_abs_inequality_l3782_378272

theorem solution_set_abs_inequality :
  Set.Icc (1 : ℝ) 2 = {x : ℝ | |2*x - 3| ≤ 1} := by sorry

end NUMINAMATH_CALUDE_solution_set_abs_inequality_l3782_378272


namespace NUMINAMATH_CALUDE_dog_food_weight_l3782_378257

theorem dog_food_weight (initial_amount : ℝ) (second_bag_weight : ℝ) (final_amount : ℝ) 
  (h1 : initial_amount = 15)
  (h2 : second_bag_weight = 10)
  (h3 : final_amount = 40) :
  ∃ (first_bag_weight : ℝ), 
    initial_amount + first_bag_weight + second_bag_weight = final_amount ∧ 
    first_bag_weight = 15 := by
  sorry

end NUMINAMATH_CALUDE_dog_food_weight_l3782_378257


namespace NUMINAMATH_CALUDE_candy_distribution_l3782_378250

theorem candy_distribution (total_candy : ℕ) (num_people : ℕ) (bags_per_person : ℕ) : 
  total_candy = 648 → num_people = 4 → bags_per_person = 8 →
  (total_candy / num_people / bags_per_person : ℕ) = 20 := by
  sorry

end NUMINAMATH_CALUDE_candy_distribution_l3782_378250


namespace NUMINAMATH_CALUDE_special_permutations_count_l3782_378298

/-- The number of permutations of n distinct elements where a₁ is not in the 1st position,
    a₂ is not in the 2nd position, and a₃ is not in the 3rd position. -/
def special_permutations (n : ℕ) : ℕ :=
  (n^3 - 6*n^2 + 14*n - 13) * Nat.factorial (n - 3)

/-- Theorem stating that for n ≥ 3, the number of permutations of n distinct elements
    where a₁ is not in the 1st position, a₂ is not in the 2nd position, and a₃ is not
    in the 3rd position is equal to (n³ - 6n² + 14n - 13) * (n-3)! -/
theorem special_permutations_count (n : ℕ) (h : n ≥ 3) :
  special_permutations n = (n^3 - 6*n^2 + 14*n - 13) * Nat.factorial (n - 3) := by
  sorry

end NUMINAMATH_CALUDE_special_permutations_count_l3782_378298


namespace NUMINAMATH_CALUDE_least_product_of_distinct_primes_above_50_l3782_378212

theorem least_product_of_distinct_primes_above_50 : 
  ∃ (p q : ℕ), 
    p.Prime ∧ 
    q.Prime ∧ 
    p ≠ q ∧ 
    p > 50 ∧ 
    q > 50 ∧ 
    p * q = 3127 ∧ 
    ∀ (r s : ℕ), r.Prime → s.Prime → r ≠ s → r > 50 → s > 50 → r * s ≥ 3127 :=
by sorry

end NUMINAMATH_CALUDE_least_product_of_distinct_primes_above_50_l3782_378212


namespace NUMINAMATH_CALUDE_compound_interest_rate_l3782_378273

theorem compound_interest_rate (P : ℝ) (r : ℝ) 
  (h1 : P * (1 + r / 100)^2 = 2420)
  (h2 : P * (1 + r / 100)^3 = 3025) :
  r = 25 := by
  sorry

end NUMINAMATH_CALUDE_compound_interest_rate_l3782_378273


namespace NUMINAMATH_CALUDE_intersection_equality_condition_l3782_378225

def A : Set ℝ := {x | 1 < x ∧ x ≤ 2}

def B (a : ℝ) : Set ℝ := {x | (2 : ℝ)^(2*a*x) < (2 : ℝ)^(a+x)}

theorem intersection_equality_condition (a : ℝ) :
  A ∩ B a = A ↔ a < 2/3 := by sorry

end NUMINAMATH_CALUDE_intersection_equality_condition_l3782_378225


namespace NUMINAMATH_CALUDE_magic_8_ball_probability_l3782_378208

/-- The probability of getting a positive answer from the Magic 8 Ball -/
def p : ℚ := 1/3

/-- The number of questions asked -/
def n : ℕ := 7

/-- The number of positive answers we're interested in -/
def k : ℕ := 3

/-- The probability of getting exactly k positive answers out of n questions -/
def probability_k_successes (n k : ℕ) (p : ℚ) : ℚ :=
  (n.choose k) * p^k * (1-p)^(n-k)

theorem magic_8_ball_probability :
  probability_k_successes n k p = 560/2187 := by
  sorry

end NUMINAMATH_CALUDE_magic_8_ball_probability_l3782_378208


namespace NUMINAMATH_CALUDE_min_score_for_average_l3782_378202

def total_tests : ℕ := 5
def max_score : ℕ := 120
def target_average : ℕ := 90
def first_three_scores : List ℕ := [88, 96, 105]

theorem min_score_for_average (other_score : ℕ) : 
  (List.sum first_three_scores + max_score + other_score) / total_tests = target_average →
  other_score ≥ 41 ∧ 
  ∀ (x : ℕ), x < 41 → (List.sum first_three_scores + max_score + x) / total_tests < target_average :=
by sorry

end NUMINAMATH_CALUDE_min_score_for_average_l3782_378202


namespace NUMINAMATH_CALUDE_bathroom_area_theorem_l3782_378201

/-- Calculates the square footage of a bathroom given the number of tiles and tile size -/
def bathroom_square_footage (width_tiles : ℕ) (length_tiles : ℕ) (tile_size : ℚ) : ℚ :=
  let width_feet := (width_tiles * tile_size) / 12
  let length_feet := (length_tiles * tile_size) / 12
  width_feet * length_feet

/-- Theorem: A bathroom with 10 6-inch tiles along its width and 20 6-inch tiles along its length has a square footage of 50 square feet -/
theorem bathroom_area_theorem :
  bathroom_square_footage 10 20 (6 / 1) = 50 := by
  sorry

end NUMINAMATH_CALUDE_bathroom_area_theorem_l3782_378201


namespace NUMINAMATH_CALUDE_symmetric_line_ratio_l3782_378283

-- Define the triangle ABC and points M, N
structure Triangle :=
  (A B C M N : ℝ × ℝ)

-- Define the property of AM and AN being symmetric with respect to angle bisector of A
def isSymmetric (t : Triangle) : Prop :=
  -- This is a placeholder for the actual geometric condition
  sorry

-- Define the lengths of sides and segments
def length (p q : ℝ × ℝ) : ℝ :=
  sorry

-- State the theorem
theorem symmetric_line_ratio (t : Triangle) :
  isSymmetric t →
  (length t.B t.M * length t.B t.N) / (length t.C t.M * length t.C t.N) =
  (length t.A t.C)^2 / (length t.A t.B)^2 :=
by
  sorry

end NUMINAMATH_CALUDE_symmetric_line_ratio_l3782_378283


namespace NUMINAMATH_CALUDE_toms_balloons_l3782_378227

theorem toms_balloons (sara_balloons : ℕ) (total_balloons : ℕ) 
  (h1 : sara_balloons = 8)
  (h2 : total_balloons = 17) :
  total_balloons - sara_balloons = 9 := by
  sorry

end NUMINAMATH_CALUDE_toms_balloons_l3782_378227


namespace NUMINAMATH_CALUDE_max_profit_is_270000_l3782_378236

/-- Represents the production and profit details for a company's two products. -/
structure ProductionProblem where
  materialA_for_A : ℝ  -- tons of Material A needed for 1 ton of Product A
  materialB_for_A : ℝ  -- tons of Material B needed for 1 ton of Product A
  materialA_for_B : ℝ  -- tons of Material A needed for 1 ton of Product B
  materialB_for_B : ℝ  -- tons of Material B needed for 1 ton of Product B
  profit_A : ℝ         -- profit (in RMB) for 1 ton of Product A
  profit_B : ℝ         -- profit (in RMB) for 1 ton of Product B
  max_materialA : ℝ    -- maximum available tons of Material A
  max_materialB : ℝ    -- maximum available tons of Material B

/-- Calculates the maximum profit given the production constraints. -/
def maxProfit (p : ProductionProblem) : ℝ :=
  sorry

/-- States that the maximum profit for the given problem is 270,000 RMB. -/
theorem max_profit_is_270000 (p : ProductionProblem) 
  (h1 : p.materialA_for_A = 3)
  (h2 : p.materialB_for_A = 2)
  (h3 : p.materialA_for_B = 1)
  (h4 : p.materialB_for_B = 3)
  (h5 : p.profit_A = 50000)
  (h6 : p.profit_B = 30000)
  (h7 : p.max_materialA = 13)
  (h8 : p.max_materialB = 18) :
  maxProfit p = 270000 :=
by sorry

end NUMINAMATH_CALUDE_max_profit_is_270000_l3782_378236


namespace NUMINAMATH_CALUDE_tshirt_sale_revenue_l3782_378230

/-- Calculates the total money made from selling t-shirts with a discount -/
theorem tshirt_sale_revenue (original_price discount : ℕ) (num_sold : ℕ) :
  original_price = 51 →
  discount = 8 →
  num_sold = 130 →
  (original_price - discount) * num_sold = 5590 :=
by sorry

end NUMINAMATH_CALUDE_tshirt_sale_revenue_l3782_378230


namespace NUMINAMATH_CALUDE_age_difference_l3782_378239

theorem age_difference (A B : ℕ) : 
  B = 39 → 
  A + 10 = 2 * (B - 10) → 
  A - B = 9 := by
sorry

end NUMINAMATH_CALUDE_age_difference_l3782_378239


namespace NUMINAMATH_CALUDE_polynomial_coefficient_theorem_l3782_378222

theorem polynomial_coefficient_theorem (a₀ a₁ a₂ a₃ a₄ : ℝ) :
  (∀ x : ℝ, x^4 = a₀ + a₁*(x+2) + a₂*(x+2)^2 + a₃*(x+2)^3 + a₄*(x+2)^4) →
  a₃ = -8 := by
sorry

end NUMINAMATH_CALUDE_polynomial_coefficient_theorem_l3782_378222


namespace NUMINAMATH_CALUDE_sqrt_four_fifths_simplification_l3782_378243

theorem sqrt_four_fifths_simplification :
  Real.sqrt (4 / 5) = (2 * Real.sqrt 5) / 5 := by sorry

end NUMINAMATH_CALUDE_sqrt_four_fifths_simplification_l3782_378243


namespace NUMINAMATH_CALUDE_smallest_b_value_l3782_378264

theorem smallest_b_value (a b : ℕ+) (h1 : a - b = 4) 
  (h2 : Nat.gcd ((a^3 + b^3) / (a + b)) (a * b) = 4) : 
  b ≥ 2 ∧ ∃ (a' b' : ℕ+), b' = 2 ∧ a' - b' = 4 ∧ 
    Nat.gcd ((a'^3 + b'^3) / (a' + b')) (a' * b') = 4 :=
sorry

end NUMINAMATH_CALUDE_smallest_b_value_l3782_378264


namespace NUMINAMATH_CALUDE_rationalize_denominator_l3782_378299

theorem rationalize_denominator :
  ∃ (A B C D E F : ℤ),
    (1 : ℝ) / (Real.sqrt 2 + Real.sqrt 3 + Real.sqrt 5) =
    (A * Real.sqrt 2 + B * Real.sqrt 3 + C * Real.sqrt 5 + D * Real.sqrt E) / F ∧
    F > 0 ∧
    A = 6 ∧ B = 4 ∧ C = -1 ∧ D = 1 ∧ E = 30 ∧ F = 12 := by
  sorry

end NUMINAMATH_CALUDE_rationalize_denominator_l3782_378299


namespace NUMINAMATH_CALUDE_factorization_identities_l3782_378265

theorem factorization_identities :
  (∀ m : ℝ, m^3 - 16*m = m*(m+4)*(m-4)) ∧
  (∀ a x : ℝ, -4*a^2*x + 12*a*x - 9*x = -x*(2*a-3)^2) := by
  sorry

end NUMINAMATH_CALUDE_factorization_identities_l3782_378265


namespace NUMINAMATH_CALUDE_point_on_x_axis_point_in_second_quadrant_l3782_378200

-- Define point P
def P (a : ℝ) : ℝ × ℝ := (2*a - 2, a + 5)

-- Part 1
theorem point_on_x_axis (a : ℝ) :
  P a = (-12, 0) ↔ (P a).2 = 0 :=
sorry

-- Part 2
theorem point_in_second_quadrant (a : ℝ) :
  (P a).1 < 0 ∧ (P a).2 > 0 ∧ |(P a).1| = |(P a).2| →
  a^2023 + 2023 = 2022 :=
sorry

end NUMINAMATH_CALUDE_point_on_x_axis_point_in_second_quadrant_l3782_378200


namespace NUMINAMATH_CALUDE_two_new_players_joined_l3782_378206

/-- Given an initial group of players and some new players joining, 
    calculates the number of new players based on the total lives. -/
def new_players (initial_players : ℕ) (lives_per_player : ℕ) (total_lives : ℕ) : ℕ :=
  (total_lives - initial_players * lives_per_player) / lives_per_player

/-- Proves that 2 new players joined the game given the initial conditions. -/
theorem two_new_players_joined :
  new_players 7 7 63 = 2 := by
  sorry

end NUMINAMATH_CALUDE_two_new_players_joined_l3782_378206


namespace NUMINAMATH_CALUDE_system_solution_l3782_378207

theorem system_solution (x y z : ℝ) : 
  x ≠ y ∧ y ≠ z ∧ x ≠ z ∧
  x^2 + y^2 = -x + 3*y + z ∧
  y^2 + z^2 = x + 3*y - z ∧
  z^2 + x^2 = 2*x + 2*y - z →
  ((x = 0 ∧ y = 1 ∧ z = -2) ∨ (x = -3/2 ∧ y = 5/2 ∧ z = -1/2)) :=
by sorry

end NUMINAMATH_CALUDE_system_solution_l3782_378207


namespace NUMINAMATH_CALUDE_sqrt_expression_equality_l3782_378205

theorem sqrt_expression_equality : 
  (Real.sqrt 24 - Real.sqrt 6) / Real.sqrt 3 + Real.sqrt (1/2) = (3 * Real.sqrt 2) / 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_expression_equality_l3782_378205


namespace NUMINAMATH_CALUDE_f_max_at_neg_three_l3782_378216

/-- The quadratic function f(x) = -x^2 - 6x + 12 -/
def f (x : ℝ) : ℝ := -x^2 - 6*x + 12

/-- Theorem stating that f(x) attains its maximum value when x = -3 -/
theorem f_max_at_neg_three :
  ∃ (max : ℝ), f (-3) = max ∧ ∀ x, f x ≤ max :=
sorry

end NUMINAMATH_CALUDE_f_max_at_neg_three_l3782_378216


namespace NUMINAMATH_CALUDE_parabola_arc_projection_difference_l3782_378262

/-- 
Given a parabola y = x^2 + px + q and two rays y = x and y = 2x for x ≥ 0,
prove that the difference between the projection of the right arc and 
the projection of the left arc on the x-axis is equal to 1.
-/
theorem parabola_arc_projection_difference 
  (p q : ℝ) : 
  ∃ (x₁ x₂ x₃ x₄ : ℝ), 
    (x₁ < x₂) ∧ (x₃ < x₄) ∧
    (x₁^2 + p*x₁ + q = x₁) ∧ 
    (x₂^2 + p*x₂ + q = x₂) ∧
    (x₃^2 + p*x₃ + q = 2*x₃) ∧ 
    (x₄^2 + p*x₄ + q = 2*x₄) ∧
    (x₄ - x₂) - (x₁ - x₃) = 1 := by
  sorry

end NUMINAMATH_CALUDE_parabola_arc_projection_difference_l3782_378262


namespace NUMINAMATH_CALUDE_no_fraction_value_l3782_378241

-- Define the No operator
def No : ℕ → ℕ
  | 0 => 1
  | n + 1 => (n + 1) * No n

-- State the theorem
theorem no_fraction_value :
  (No 2022) / (No 2023) = 1 / 2023 := by sorry

end NUMINAMATH_CALUDE_no_fraction_value_l3782_378241


namespace NUMINAMATH_CALUDE_nested_fraction_evaluation_l3782_378219

theorem nested_fraction_evaluation : 
  1 / (1 + 1 / (2 + 1 / (4^2))) = 33 / 49 := by
  sorry

end NUMINAMATH_CALUDE_nested_fraction_evaluation_l3782_378219


namespace NUMINAMATH_CALUDE_unique_swap_pair_l3782_378291

/-- A quadratic polynomial function -/
def QuadraticPolynomial (α : Type) [Ring α] := α → α

theorem unique_swap_pair
  (f : QuadraticPolynomial ℝ)
  (a b : ℝ)
  (h_distinct : a ≠ b)
  (h_swap : f a = b ∧ f b = a) :
  ¬∃ c d, c ≠ d ∧ (c, d) ≠ (a, b) ∧ f c = d ∧ f d = c :=
sorry

end NUMINAMATH_CALUDE_unique_swap_pair_l3782_378291


namespace NUMINAMATH_CALUDE_zhong_is_symmetrical_l3782_378233

/-- A Chinese character is represented as a structure with left and right sides -/
structure ChineseCharacter where
  left : String
  right : String

/-- A function to check if a character is symmetrical -/
def isSymmetrical (c : ChineseCharacter) : Prop :=
  c.left = c.right

/-- The Chinese character "中" -/
def zhong : ChineseCharacter :=
  { left := "|", right := "|" }

/-- Theorem stating that "中" is symmetrical -/
theorem zhong_is_symmetrical : isSymmetrical zhong := by
  sorry


end NUMINAMATH_CALUDE_zhong_is_symmetrical_l3782_378233


namespace NUMINAMATH_CALUDE_danny_travel_time_l3782_378228

/-- The time it takes Danny to reach Steve's house -/
def danny_time : ℝ := 31

/-- The time it takes Steve to reach Danny's house -/
def steve_time (t : ℝ) : ℝ := 2 * t

/-- The time difference between Steve and Danny reaching the halfway point -/
def halfway_time_difference : ℝ := 15.5

theorem danny_travel_time :
  ∀ t : ℝ,
  (steve_time t / 2 - t / 2 = halfway_time_difference) →
  t = danny_time :=
by
  sorry


end NUMINAMATH_CALUDE_danny_travel_time_l3782_378228


namespace NUMINAMATH_CALUDE_arithmetic_series_sum_problem_solution_l3782_378255

theorem arithmetic_series_sum (a₁ d n : ℕ) (h : n > 0) : 
  (n : ℝ) / 2 * (2 * a₁ + (n - 1) * d) = (n : ℝ) / 2 * (a₁ + (a₁ + (n - 1) * d)) :=
by sorry

theorem problem_solution : 
  let a₁ : ℕ := 9
  let d : ℕ := 4
  let n : ℕ := 50
  (n : ℝ) / 2 * (a₁ + (a₁ + (n - 1) * d)) = 5350 :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_series_sum_problem_solution_l3782_378255


namespace NUMINAMATH_CALUDE_farm_chicken_count_l3782_378242

/-- The number of chicken coops on the farm -/
def num_coops : ℕ := 9

/-- The number of chickens in each coop -/
def chickens_per_coop : ℕ := 60

/-- The total number of chickens on the farm -/
def total_chickens : ℕ := num_coops * chickens_per_coop

theorem farm_chicken_count : total_chickens = 540 := by
  sorry

end NUMINAMATH_CALUDE_farm_chicken_count_l3782_378242


namespace NUMINAMATH_CALUDE_auction_tv_initial_price_l3782_378235

/-- Given an auction event where:
    - The price of a TV increased by 2/5 times its initial price
    - The price of a phone, initially $400, increased by 40%
    - The total amount received after sale is $1260
    Prove that the initial price of the TV was $500 -/
theorem auction_tv_initial_price (tv_initial : ℝ) (phone_initial : ℝ) (total : ℝ) :
  phone_initial = 400 →
  total = 1260 →
  total = (tv_initial + 2/5 * tv_initial) + (phone_initial + 0.4 * phone_initial) →
  tv_initial = 500 := by
  sorry


end NUMINAMATH_CALUDE_auction_tv_initial_price_l3782_378235


namespace NUMINAMATH_CALUDE_polynomial_roots_nature_l3782_378251

def P (x : ℝ) : ℝ := x^6 - 5*x^5 + 3*x^2 - 8*x + 16

theorem polynomial_roots_nature :
  (∀ x < 0, P x > 0) ∧ 
  (∃ a b, 0 < a ∧ a < b ∧ P a * P b < 0) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_roots_nature_l3782_378251


namespace NUMINAMATH_CALUDE_fixed_point_of_linear_function_l3782_378238

theorem fixed_point_of_linear_function (m : ℝ) (hm : m ≠ 0) :
  let f : ℝ → ℝ := λ x ↦ m * x - (3 * m + 2)
  f 3 = -2 := by
  sorry

end NUMINAMATH_CALUDE_fixed_point_of_linear_function_l3782_378238


namespace NUMINAMATH_CALUDE_cuboid_volume_l3782_378204

/-- Represents a cuboid with length, width, and height -/
structure Cuboid where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Calculates the volume of a cuboid -/
def Cuboid.volume (c : Cuboid) : ℝ :=
  c.length * c.width * c.height

/-- Calculates the surface area of a cuboid -/
def Cuboid.surfaceArea (c : Cuboid) : ℝ :=
  2 * (c.length * c.width + c.length * c.height + c.width * c.height)

/-- Theorem: The volume of the cuboid is 180 cm³ -/
theorem cuboid_volume (c : Cuboid) :
  (∀ (c' : Cuboid), c'.length = c.length ∧ c'.width = c.width ∧ c'.height = c.height + 1 →
    c'.length = c'.width ∧ c'.width = c'.height) →
  (∃ (c' : Cuboid), c'.length = c.length ∧ c'.width = c.width ∧ c'.height = c.height + 1 ∧
    c'.surfaceArea = c.surfaceArea + 24) →
  c.volume = 180 := by
  sorry

end NUMINAMATH_CALUDE_cuboid_volume_l3782_378204


namespace NUMINAMATH_CALUDE_average_age_decrease_l3782_378223

theorem average_age_decrease (initial_average : ℝ) : 
  let initial_total_age := 10 * initial_average
  let new_total_age := initial_total_age - 48 + 18
  let new_average := new_total_age / 10
  initial_average - new_average = 3 := by
sorry

end NUMINAMATH_CALUDE_average_age_decrease_l3782_378223


namespace NUMINAMATH_CALUDE_triangle_sine_relation_l3782_378209

theorem triangle_sine_relation (A B C : ℝ) (h_triangle : A + B + C = π) 
  (h_relation : 3 * Real.sin B ^ 2 + 7 * Real.sin C ^ 2 = 
                2 * Real.sin A * Real.sin B * Real.sin C + 2 * Real.sin A ^ 2) : 
  Real.sin (A + π / 4) = -Real.sqrt 10 / 10 := by
  sorry

end NUMINAMATH_CALUDE_triangle_sine_relation_l3782_378209


namespace NUMINAMATH_CALUDE_simplify_and_rationalize_l3782_378290

theorem simplify_and_rationalize :
  (Real.sqrt 3 / Real.sqrt 4) * (Real.sqrt 5 / Real.sqrt 6) * (Real.sqrt 8 / Real.sqrt 9) = Real.sqrt 15 / 9 := by
  sorry

end NUMINAMATH_CALUDE_simplify_and_rationalize_l3782_378290


namespace NUMINAMATH_CALUDE_alicia_tax_deduction_l3782_378297

/-- Represents Alicia's hourly wage in dollars -/
def hourly_wage : ℚ := 25

/-- Represents the local tax rate as a decimal -/
def tax_rate : ℚ := 2 / 100

/-- Converts dollars to cents -/
def dollars_to_cents (dollars : ℚ) : ℚ := dollars * 100

/-- Calculates the tax amount in cents -/
def tax_amount_cents : ℚ := dollars_to_cents (hourly_wage * tax_rate)

theorem alicia_tax_deduction :
  tax_amount_cents = 50 := by sorry

end NUMINAMATH_CALUDE_alicia_tax_deduction_l3782_378297


namespace NUMINAMATH_CALUDE_decision_symbol_is_diamond_l3782_378271

-- Define the type for flowchart symbols
inductive FlowchartSymbol
  | Diamond
  | Rectangle
  | Oval
  | Parallelogram

-- Define the function that determines if a symbol represents a decision
def representsDecision (symbol : FlowchartSymbol) : Prop :=
  symbol = FlowchartSymbol.Diamond

-- Theorem: The symbol that represents a decision in a flowchart is a diamond-shaped box
theorem decision_symbol_is_diamond :
  ∃ (symbol : FlowchartSymbol), representsDecision symbol :=
sorry

end NUMINAMATH_CALUDE_decision_symbol_is_diamond_l3782_378271
