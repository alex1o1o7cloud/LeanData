import Mathlib

namespace NUMINAMATH_CALUDE_pascal_triangle_value_l2132_213263

/-- The number of elements in the row of Pascal's triangle we're considering -/
def row_length : Nat := 47

/-- The position of the number we're looking for in the row (1-indexed) -/
def target_position : Nat := 45

/-- The row number in Pascal's triangle (0-indexed) -/
def row_number : Nat := row_length - 1

/-- The binomial coefficient we need to calculate -/
def pascal_number : Nat := Nat.choose row_number (target_position - 1)

theorem pascal_triangle_value : pascal_number = 1035 := by sorry

end NUMINAMATH_CALUDE_pascal_triangle_value_l2132_213263


namespace NUMINAMATH_CALUDE_probability_at_least_one_even_is_65_81_l2132_213271

def valid_digits : Finset ℕ := {0, 3, 5, 7, 8, 9}
def code_length : ℕ := 4

def probability_at_least_one_even : ℚ :=
  1 - (Finset.filter (λ x => ¬ Even x) valid_digits).card ^ code_length /
      valid_digits.card ^ code_length

theorem probability_at_least_one_even_is_65_81 :
  probability_at_least_one_even = 65 / 81 := by
  sorry

end NUMINAMATH_CALUDE_probability_at_least_one_even_is_65_81_l2132_213271


namespace NUMINAMATH_CALUDE_large_cube_volume_l2132_213261

/-- The volume of a cube constructed from smaller cubes -/
theorem large_cube_volume (n : ℕ) (edge : ℝ) (h : n = 125) (h_edge : edge = 2) :
  (n : ℝ) * (edge ^ 3) = 1000 := by
  sorry

end NUMINAMATH_CALUDE_large_cube_volume_l2132_213261


namespace NUMINAMATH_CALUDE_equation_roots_sum_l2132_213232

theorem equation_roots_sum (a b c m : ℝ) : 
  (∃ x y : ℝ, 
    (x^2 - (b+1)*x) / (2*a*x - c) = (2*m-3) / (2*m+1) ∧
    (y^2 - (b+1)*y) / (2*a*y - c) = (2*m-3) / (2*m+1) ∧
    x + y = b + 1) →
  m = 1.5 := by sorry

end NUMINAMATH_CALUDE_equation_roots_sum_l2132_213232


namespace NUMINAMATH_CALUDE_count_special_numbers_eq_3776_l2132_213205

/-- A function that counts the number of five-digit numbers beginning with 2 
    that have exactly two identical digits -/
def count_special_numbers : ℕ :=
  let case1 := 4 * 8 * 8 * 8  -- Case where the two identical digits are 2s
  let case2 := 3 * 9 * 8 * 8  -- Case where the two identical digits are not 2s
  case1 + case2

/-- Theorem stating that there are exactly 3776 five-digit numbers 
    beginning with 2 that have exactly two identical digits -/
theorem count_special_numbers_eq_3776 :
  count_special_numbers = 3776 := by
  sorry

#eval count_special_numbers  -- Should output 3776

end NUMINAMATH_CALUDE_count_special_numbers_eq_3776_l2132_213205


namespace NUMINAMATH_CALUDE_additional_distance_for_average_speed_l2132_213213

theorem additional_distance_for_average_speed
  (initial_distance : ℝ)
  (initial_speed : ℝ)
  (second_speed : ℝ)
  (average_speed : ℝ)
  (h1 : initial_distance = 18)
  (h2 : initial_speed = 36)
  (h3 : second_speed = 60)
  (h4 : average_speed = 45)
  : ∃ (additional_distance : ℝ),
    (initial_distance + additional_distance) / ((initial_distance / initial_speed) + (additional_distance / second_speed)) = average_speed
    ∧ additional_distance = 18 := by
  sorry


end NUMINAMATH_CALUDE_additional_distance_for_average_speed_l2132_213213


namespace NUMINAMATH_CALUDE_percentage_calculation_l2132_213229

theorem percentage_calculation (x : ℝ) (h : 0.035 * x = 700) : 0.024 * (1.5 * x) = 720 := by
  sorry

end NUMINAMATH_CALUDE_percentage_calculation_l2132_213229


namespace NUMINAMATH_CALUDE_intersection_condition_for_singleton_zero_l2132_213252

theorem intersection_condition_for_singleton_zero (A : Set ℕ) :
  (A = {0} → A ∩ {0, 1} = {0}) ∧
  ∃ A : Set ℕ, A ∩ {0, 1} = {0} ∧ A ≠ {0} :=
by sorry

end NUMINAMATH_CALUDE_intersection_condition_for_singleton_zero_l2132_213252


namespace NUMINAMATH_CALUDE_circle_passes_through_fixed_point_tangent_circles_l2132_213262

/-- The parametric equation of a circle -/
def circle_equation (a x y : ℝ) : Prop :=
  x^2 + y^2 - 4*a*x + 2*a*y + 20*a - 20 = 0

/-- The equation of the fixed circle -/
def fixed_circle (x y : ℝ) : Prop :=
  x^2 + y^2 = 4

theorem circle_passes_through_fixed_point (a : ℝ) :
  circle_equation a 4 (-2) := by sorry

theorem tangent_circles (a : ℝ) :
  (∃ x y : ℝ, circle_equation a x y ∧ fixed_circle x y) ↔ 
  (a = 1 + Real.sqrt 5 / 5 ∨ a = 1 - Real.sqrt 5 / 5) := by sorry

end NUMINAMATH_CALUDE_circle_passes_through_fixed_point_tangent_circles_l2132_213262


namespace NUMINAMATH_CALUDE_ellipse_major_axis_length_l2132_213285

/-- The length of the major axis of an ellipse formed by intersecting a plane with a right circular cylinder --/
def major_axis_length (cylinder_radius : ℝ) (major_minor_ratio : ℝ) : ℝ :=
  2 * cylinder_radius * major_minor_ratio

/-- Theorem stating the length of the major axis for the given conditions --/
theorem ellipse_major_axis_length :
  major_axis_length 2 1.4 = 5.6 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_major_axis_length_l2132_213285


namespace NUMINAMATH_CALUDE_prob_ace_hearts_king_spades_l2132_213297

/-- Represents a standard deck of 52 playing cards -/
def StandardDeck : ℕ := 52

/-- Represents the number of cards of each suit in a standard deck -/
def CardsPerSuit : ℕ := 13

/-- The probability of drawing a specific card from a standard deck -/
def ProbFirstCard : ℚ := 1 / StandardDeck

/-- The probability of drawing a specific card from the remaining deck after one card is drawn -/
def ProbSecondCard : ℚ := 1 / (StandardDeck - 1)

/-- The probability of drawing two specific cards in order from a standard deck -/
def ProbTwoSpecificCards : ℚ := ProbFirstCard * ProbSecondCard

theorem prob_ace_hearts_king_spades : 
  ProbTwoSpecificCards = 1 / 2652 := by sorry

end NUMINAMATH_CALUDE_prob_ace_hearts_king_spades_l2132_213297


namespace NUMINAMATH_CALUDE_angle_negative_1120_in_fourth_quadrant_l2132_213206

def angle_to_standard_form (angle : ℤ) : ℤ :=
  angle % 360

def quadrant (angle : ℤ) : ℕ :=
  let standard_angle := angle_to_standard_form angle
  if 0 ≤ standard_angle ∧ standard_angle < 90 then 1
  else if 90 ≤ standard_angle ∧ standard_angle < 180 then 2
  else if 180 ≤ standard_angle ∧ standard_angle < 270 then 3
  else 4

theorem angle_negative_1120_in_fourth_quadrant :
  quadrant (-1120) = 4 := by
  sorry

end NUMINAMATH_CALUDE_angle_negative_1120_in_fourth_quadrant_l2132_213206


namespace NUMINAMATH_CALUDE_worker_b_days_l2132_213292

/-- Represents the daily wages and work days of three workers -/
structure WorkerData where
  wage_a : ℚ
  wage_b : ℚ
  wage_c : ℚ
  days_a : ℕ
  days_b : ℕ
  days_c : ℕ

/-- Calculates the total earnings of the workers -/
def totalEarnings (data : WorkerData) : ℚ :=
  data.wage_a * data.days_a + data.wage_b * data.days_b + data.wage_c * data.days_c

theorem worker_b_days (data : WorkerData) 
  (h1 : data.days_a = 6)
  (h2 : data.days_c = 4)
  (h3 : data.wage_a / data.wage_b = 3 / 4)
  (h4 : data.wage_b / data.wage_c = 4 / 5)
  (h5 : totalEarnings data = 1702)
  (h6 : data.wage_c = 115) :
  data.days_b = 9 := by
sorry

end NUMINAMATH_CALUDE_worker_b_days_l2132_213292


namespace NUMINAMATH_CALUDE_largest_of_three_consecutive_evens_l2132_213259

theorem largest_of_three_consecutive_evens (a b c : ℤ) : 
  (∃ k : ℤ, a = 2*k ∧ b = 2*k + 2 ∧ c = 2*k + 4) →  -- a, b, c are consecutive even integers
  a + b + c = 1194 →                               -- their sum is 1194
  c = 400                                          -- the largest (c) is 400
:= by sorry

end NUMINAMATH_CALUDE_largest_of_three_consecutive_evens_l2132_213259


namespace NUMINAMATH_CALUDE_specific_tetrahedron_volume_l2132_213291

/-- The volume of a tetrahedron with given edge lengths -/
def tetrahedron_volume (AB BC CD DA AC BD : ℝ) : ℝ :=
  -- Definition of volume calculation goes here
  sorry

/-- Theorem: The volume of the specific tetrahedron is √66/2 -/
theorem specific_tetrahedron_volume :
  tetrahedron_volume 1 (2 * Real.sqrt 6) 5 7 5 7 = Real.sqrt 66 / 2 := by
  sorry

end NUMINAMATH_CALUDE_specific_tetrahedron_volume_l2132_213291


namespace NUMINAMATH_CALUDE_endpoint_sum_coordinates_l2132_213225

/-- Given a line segment with one endpoint (6, -2) and midpoint (5, 5),
    the sum of coordinates of the other endpoint is 16. -/
theorem endpoint_sum_coordinates (x y : ℝ) : 
  (6 + x) / 2 = 5 ∧ (-2 + y) / 2 = 5 → x + y = 16 := by
  sorry

end NUMINAMATH_CALUDE_endpoint_sum_coordinates_l2132_213225


namespace NUMINAMATH_CALUDE_lcm_gcd_product_24_36_l2132_213239

theorem lcm_gcd_product_24_36 : Nat.lcm 24 36 * Nat.gcd 24 36 = 864 := by
  sorry

end NUMINAMATH_CALUDE_lcm_gcd_product_24_36_l2132_213239


namespace NUMINAMATH_CALUDE_unique_determination_l2132_213203

/-- Two-digit number type -/
def TwoDigitNum := {n : ℕ // n ≥ 0 ∧ n ≤ 99}

/-- The sum function as defined in the problem -/
def sum (a b c : TwoDigitNum) (X Y Z : ℕ) : ℕ :=
  a.val * X + b.val * Y + c.val * Z

/-- Function to extract a from the sum -/
def extract_a (S : ℕ) : ℕ := S % 100

/-- Function to extract b from the sum -/
def extract_b (S : ℕ) : ℕ := (S / 100) % 100

/-- Function to extract c from the sum -/
def extract_c (S : ℕ) : ℕ := S / 10000

/-- Theorem stating that a, b, and c can be uniquely determined from the sum -/
theorem unique_determination (a b c : TwoDigitNum) :
  let X : ℕ := 1
  let Y : ℕ := 100
  let Z : ℕ := 10000
  let S := sum a b c X Y Z
  (extract_a S = a.val) ∧ (extract_b S = b.val) ∧ (extract_c S = c.val) := by
  sorry

end NUMINAMATH_CALUDE_unique_determination_l2132_213203


namespace NUMINAMATH_CALUDE_stratified_sampling_female_count_l2132_213260

theorem stratified_sampling_female_count 
  (total_students : ℕ) 
  (female_students : ℕ) 
  (sample_size : ℕ) 
  (h1 : total_students = 2000)
  (h2 : female_students = 800)
  (h3 : sample_size = 50) :
  (sample_size * female_students) / total_students = 20 := by
  sorry

end NUMINAMATH_CALUDE_stratified_sampling_female_count_l2132_213260


namespace NUMINAMATH_CALUDE_expansion_properties_l2132_213222

theorem expansion_properties (x : ℝ) (n : ℕ) :
  (∃ k : ℝ, 2 * (n.choose 2) = (n.choose 1) + (n.choose 3) ∧ k ≠ 0) →
  (n = 7 ∧ ∀ r : ℕ, r ≤ n → (7 - 2*r ≠ 0)) :=
by sorry

end NUMINAMATH_CALUDE_expansion_properties_l2132_213222


namespace NUMINAMATH_CALUDE_simplify_radical_expression_l2132_213247

theorem simplify_radical_expression : 
  Real.sqrt 80 - 4 * Real.sqrt 5 + 3 * Real.sqrt (180 / 3) = Real.sqrt 540 := by
  sorry

end NUMINAMATH_CALUDE_simplify_radical_expression_l2132_213247


namespace NUMINAMATH_CALUDE_equation_is_union_of_twisted_cubics_twisted_cubic_is_parabola_like_equation_represents_two_parabolas_l2132_213295

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents the equation y^6 - 9x^6 = 3y^3 - 1 -/
def equation (p : Point3D) : Prop :=
  p.y^6 - 9*p.x^6 = 3*p.y^3 - 1

/-- Represents a twisted cubic curve -/
def twistedCubic (a b c : ℝ) (p : Point3D) : Prop :=
  p.y^3 = a*p.x^3 + b*p.x + c

/-- The equation represents the union of two twisted cubic curves -/
theorem equation_is_union_of_twisted_cubics :
  ∀ p : Point3D, equation p ↔ (twistedCubic 3 0 1 p ∨ twistedCubic (-3) 0 1 p) :=
sorry

/-- Twisted cubic curves behave like parabolas -/
theorem twisted_cubic_is_parabola_like (a b c : ℝ) :
  ∀ p : Point3D, twistedCubic a b c p → (∃ q : Point3D, twistedCubic a b c q ∧ q ≠ p) :=
sorry

/-- The equation represents two parabola-like curves -/
theorem equation_represents_two_parabolas :
  ∃ (curve1 curve2 : Point3D → Prop),
    (∀ p : Point3D, equation p ↔ (curve1 p ∨ curve2 p)) ∧
    (∀ p : Point3D, curve1 p → (∃ q : Point3D, curve1 q ∧ q ≠ p)) ∧
    (∀ p : Point3D, curve2 p → (∃ q : Point3D, curve2 q ∧ q ≠ p)) :=
sorry

end NUMINAMATH_CALUDE_equation_is_union_of_twisted_cubics_twisted_cubic_is_parabola_like_equation_represents_two_parabolas_l2132_213295


namespace NUMINAMATH_CALUDE_hat_cost_theorem_l2132_213201

def josh_shopping (initial_money : ℝ) (pencil_cost : ℝ) (cookie_cost : ℝ) (cookie_count : ℕ) (money_left : ℝ) : ℝ :=
  initial_money - (pencil_cost + cookie_cost * cookie_count) - money_left

theorem hat_cost_theorem (initial_money : ℝ) (pencil_cost : ℝ) (cookie_cost : ℝ) (cookie_count : ℕ) (money_left : ℝ) 
  (h1 : initial_money = 20)
  (h2 : pencil_cost = 2)
  (h3 : cookie_cost = 1.25)
  (h4 : cookie_count = 4)
  (h5 : money_left = 3) :
  josh_shopping initial_money pencil_cost cookie_cost cookie_count money_left = 10 := by
  sorry

#eval josh_shopping 20 2 1.25 4 3

end NUMINAMATH_CALUDE_hat_cost_theorem_l2132_213201


namespace NUMINAMATH_CALUDE_cuboidal_box_surface_area_l2132_213237

/-- A cuboidal box with given face areas has a specific total surface area -/
theorem cuboidal_box_surface_area (l w h : ℝ) : 
  l > 0 → w > 0 → h > 0 →
  l * w = 120 → w * h = 72 → l * h = 60 →
  2 * (l * w + w * h + l * h) = 504 := by
sorry

end NUMINAMATH_CALUDE_cuboidal_box_surface_area_l2132_213237


namespace NUMINAMATH_CALUDE_solve_equation_one_solve_equation_two_l2132_213234

-- Equation 1
theorem solve_equation_one (x : ℝ) : 4 * (x - 2) = 2 * x ↔ x = 4 := by
  sorry

-- Equation 2
theorem solve_equation_two (x : ℝ) : (x + 1) / 4 = 1 - (1 - x) / 3 ↔ x = -5 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_one_solve_equation_two_l2132_213234


namespace NUMINAMATH_CALUDE_f_inequality_iff_x_gt_one_l2132_213212

noncomputable def f (x : ℝ) : ℝ := Real.sin (x - 1) + Real.exp (x - 1) - Real.exp (1 - x) - x + 1

theorem f_inequality_iff_x_gt_one :
  ∀ x : ℝ, f x + f (3 - 2*x) < 0 ↔ x > 1 := by
  sorry

end NUMINAMATH_CALUDE_f_inequality_iff_x_gt_one_l2132_213212


namespace NUMINAMATH_CALUDE_quadratic_function_theorem_l2132_213250

def is_valid_quadratic (f : ℝ → ℝ) : Prop :=
  ∃ (a b c : ℝ), a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

def satisfies_bounds (f : ℝ → ℝ) : Prop :=
  ∀ x, |x| ≤ 1 → |f x| ≤ 1

def derivative_max (f : ℝ → ℝ) (K : ℝ) : Prop :=
  ∀ x, |x| ≤ 1 → |(deriv f) x| ≤ K

def max_attained (f : ℝ → ℝ) (K : ℝ) : Prop :=
  ∃ x₀, x₀ ∈ Set.Icc (-1) 1 ∧ |(deriv f) x₀| = K

theorem quadratic_function_theorem (f : ℝ → ℝ) (K : ℝ) :
  is_valid_quadratic f →
  satisfies_bounds f →
  derivative_max f K →
  max_attained f K →
  (∃ (ε : ℝ), ε = 1 ∨ ε = -1) ∧ (∀ x, f x = ε * (2 * x^2 - 1)) :=
sorry

end NUMINAMATH_CALUDE_quadratic_function_theorem_l2132_213250


namespace NUMINAMATH_CALUDE_midline_triangle_area_sum_l2132_213245

/-- The sum of areas of an infinite series of triangles, where each triangle is constructed 
    from the midlines of the previous triangle, given the area of the original triangle. -/
theorem midline_triangle_area_sum (t : ℝ) (h : t > 0) : 
  ∃ (S : ℝ), S = (∑' n, t * (3/4)^n) ∧ S = 4 * t :=
sorry

end NUMINAMATH_CALUDE_midline_triangle_area_sum_l2132_213245


namespace NUMINAMATH_CALUDE_allocation_schemes_count_l2132_213230

/-- The number of intern teachers --/
def num_teachers : ℕ := 5

/-- The number of classes --/
def num_classes : ℕ := 3

/-- The minimum number of teachers per class --/
def min_teachers_per_class : ℕ := 1

/-- The maximum number of teachers per class --/
def max_teachers_per_class : ℕ := 2

/-- A function that calculates the number of ways to allocate teachers to classes --/
def allocation_schemes (n_teachers : ℕ) (n_classes : ℕ) (min_per_class : ℕ) (max_per_class : ℕ) : ℕ :=
  sorry

/-- Theorem stating that the number of allocation schemes is 90 --/
theorem allocation_schemes_count :
  allocation_schemes num_teachers num_classes min_teachers_per_class max_teachers_per_class = 90 :=
sorry

end NUMINAMATH_CALUDE_allocation_schemes_count_l2132_213230


namespace NUMINAMATH_CALUDE_fast_food_order_l2132_213228

/-- The cost of a burger in dollars -/
def burger_cost : ℕ := 5

/-- The cost of a sandwich in dollars -/
def sandwich_cost : ℕ := 4

/-- The cost of a smoothie in dollars -/
def smoothie_cost : ℕ := 4

/-- The total cost of the order in dollars -/
def total_cost : ℕ := 17

/-- The number of smoothies ordered -/
def num_smoothies : ℕ := 2

theorem fast_food_order :
  burger_cost + sandwich_cost + smoothie_cost * num_smoothies = total_cost := by
  sorry

end NUMINAMATH_CALUDE_fast_food_order_l2132_213228


namespace NUMINAMATH_CALUDE_tangent_line_to_parabola_l2132_213255

/-- The line y = 3x + c is tangent to the parabola y^2 = 12x if and only if c = 1 -/
theorem tangent_line_to_parabola (c : ℝ) :
  (∃ x y : ℝ, y = 3 * x + c ∧ y^2 = 12 * x ∧
    ∀ x' y' : ℝ, y' = 3 * x' + c → y'^2 ≥ 12 * x') ↔ c = 1 := by
  sorry

end NUMINAMATH_CALUDE_tangent_line_to_parabola_l2132_213255


namespace NUMINAMATH_CALUDE_winning_strategy_l2132_213240

/-- Fibonacci sequence -/
def fib : ℕ → ℕ
  | 0 => 1
  | 1 => 1
  | (n + 2) => fib (n + 1) + fib n

/-- Predicate to check if a number is a Fibonacci number -/
def isFibonacci (n : ℕ) : Prop :=
  ∃ k, fib k = n

/-- Game rules -/
structure GameRules where
  n : ℕ
  n_gt_one : n > 1
  first_turn_not_all : ∀ (first_pick : ℕ), first_pick < n
  subsequent_turns : ∀ (prev_pick current_pick : ℕ), current_pick ≤ 2 * prev_pick

/-- Winning strategy for Player A -/
def playerAWins (rules : GameRules) : Prop :=
  ¬(isFibonacci rules.n)

/-- Main theorem: Player A has a winning strategy iff n is not a Fibonacci number -/
theorem winning_strategy (rules : GameRules) :
  playerAWins rules ↔ ¬(isFibonacci rules.n) := by sorry

end NUMINAMATH_CALUDE_winning_strategy_l2132_213240


namespace NUMINAMATH_CALUDE_nth_equation_sum_l2132_213299

theorem nth_equation_sum (n : ℕ) (h : n > 0) :
  (Finset.range (2 * n - 1)).sum (λ i => n + i) = (2 * n - 1)^2 := by
  sorry

end NUMINAMATH_CALUDE_nth_equation_sum_l2132_213299


namespace NUMINAMATH_CALUDE_yellow_tint_percentage_l2132_213204

/-- Calculates the percentage of yellow tint in a new mixture after adding more yellow tint --/
theorem yellow_tint_percentage 
  (original_volume : ℝ) 
  (original_yellow_percentage : ℝ) 
  (added_yellow : ℝ) : 
  original_volume = 50 → 
  original_yellow_percentage = 0.5 → 
  added_yellow = 10 → 
  (((original_volume * original_yellow_percentage + added_yellow) / (original_volume + added_yellow)) * 100 : ℝ) = 58 := by
  sorry

end NUMINAMATH_CALUDE_yellow_tint_percentage_l2132_213204


namespace NUMINAMATH_CALUDE_triangle_six_nine_equals_eleven_l2132_213223

-- Define the ▽ operation
def triangle (m n : ℚ) (x y : ℚ) : ℚ := m^2 * x + n * y - 1

-- Theorem statement
theorem triangle_six_nine_equals_eleven 
  (m n : ℚ) 
  (h : triangle m n 2 3 = 3) : 
  triangle m n 6 9 = 11 := by
sorry

end NUMINAMATH_CALUDE_triangle_six_nine_equals_eleven_l2132_213223


namespace NUMINAMATH_CALUDE_multiply_subtract_divide_l2132_213257

theorem multiply_subtract_divide : 4 * 6 * 8 - 24 / 4 = 186 := by
  sorry

end NUMINAMATH_CALUDE_multiply_subtract_divide_l2132_213257


namespace NUMINAMATH_CALUDE_slower_painter_time_l2132_213215

-- Define the start time of the slower painter (2:00 PM)
def slower_start : ℝ := 14

-- Define the finish time (0.6 past midnight, which is 24.6)
def finish_time : ℝ := 24.6

-- Theorem to prove
theorem slower_painter_time :
  finish_time - slower_start = 10.6 := by
  sorry

end NUMINAMATH_CALUDE_slower_painter_time_l2132_213215


namespace NUMINAMATH_CALUDE_yard_length_with_26_trees_32m_apart_l2132_213238

/-- The length of a yard with equally spaced trees -/
def yardLength (numTrees : ℕ) (distanceBetweenTrees : ℝ) : ℝ :=
  (numTrees - 1 : ℝ) * distanceBetweenTrees

theorem yard_length_with_26_trees_32m_apart :
  yardLength 26 32 = 800 := by
  sorry

end NUMINAMATH_CALUDE_yard_length_with_26_trees_32m_apart_l2132_213238


namespace NUMINAMATH_CALUDE_cats_in_academy_l2132_213256

/-- The number of cats that can jump -/
def jump : ℕ := 40

/-- The number of cats that can fetch -/
def fetch : ℕ := 25

/-- The number of cats that can spin -/
def spin : ℕ := 30

/-- The number of cats that can jump and fetch -/
def jump_fetch : ℕ := 20

/-- The number of cats that can fetch and spin -/
def fetch_spin : ℕ := 10

/-- The number of cats that can jump and spin -/
def jump_spin : ℕ := 15

/-- The number of cats that can do all three tricks -/
def all_tricks : ℕ := 7

/-- The number of cats that can do none of the tricks -/
def no_tricks : ℕ := 5

/-- The total number of cats in the academy -/
def total_cats : ℕ := 62

theorem cats_in_academy :
  total_cats = 
    (jump - jump_fetch - jump_spin + all_tricks) +
    (jump_fetch - all_tricks) +
    (fetch - jump_fetch - fetch_spin + all_tricks) +
    (fetch_spin - all_tricks) +
    (jump_spin - all_tricks) +
    (spin - jump_spin - fetch_spin + all_tricks) +
    all_tricks +
    no_tricks := by sorry

end NUMINAMATH_CALUDE_cats_in_academy_l2132_213256


namespace NUMINAMATH_CALUDE_symmetry_theorem_l2132_213221

-- Define the points and lines
def P : ℝ × ℝ := (1, 2)
def Q : ℝ × ℝ := (3, 0)
def l1 (x y : ℝ) : Prop := x - y - 1 = 0
def l2 (x y : ℝ) : Prop := x + 3*y - 1 = 0
def l3 (x y : ℝ) : Prop := x - y + 1 = 0

-- Define symmetry for points with respect to a line
def symmetric_point (A B : ℝ × ℝ) (l : ℝ → ℝ → Prop) : Prop :=
  let M := ((A.1 + B.1)/2, (A.2 + B.2)/2)
  l M.1 M.2 ∧ (B.1 - A.1) * (B.1 - A.1) + (B.2 - A.2) * (B.2 - A.2) = 
  4 * ((M.1 - A.1) * (M.1 - A.1) + (M.2 - A.2) * (M.2 - A.2))

-- Define symmetry for lines with respect to another line
def symmetric_line (l1 l2 l3 : ℝ → ℝ → Prop) : Prop :=
  ∀ x y : ℝ, l1 x y → ∃ x' y' : ℝ, l2 x' y' ∧ 
  symmetric_point (x, y) (x', y') l3

-- State the theorem
theorem symmetry_theorem :
  symmetric_point P Q l1 ∧
  symmetric_line l2 (fun x y => 3*x + y + 1 = 0) l3 :=
sorry

end NUMINAMATH_CALUDE_symmetry_theorem_l2132_213221


namespace NUMINAMATH_CALUDE_always_composite_l2132_213202

theorem always_composite (n : ℕ) : ∃ (a b : ℕ), a > 1 ∧ b > 1 ∧ n^4 + 64 = a * b := by
  sorry

end NUMINAMATH_CALUDE_always_composite_l2132_213202


namespace NUMINAMATH_CALUDE_equation_one_l2132_213264

theorem equation_one (x : ℝ) : x * |x| = 4 ↔ x = 2 := by sorry

end NUMINAMATH_CALUDE_equation_one_l2132_213264


namespace NUMINAMATH_CALUDE_supermarket_discount_items_l2132_213293

/-- Represents the supermarket's inventory and pricing --/
structure Supermarket where
  total_cost : ℝ
  items_a : ℕ
  items_b : ℕ
  cost_a : ℝ
  cost_b : ℝ
  price_a : ℝ
  price_b : ℝ

/-- Represents the second purchase scenario --/
structure SecondPurchase where
  sm : Supermarket
  items_b_new : ℕ
  discount_price_b : ℝ
  total_profit : ℝ

/-- The main theorem to prove --/
theorem supermarket_discount_items (sm : Supermarket) (sp : SecondPurchase) :
  sm.total_cost = 6000 ∧
  sm.items_a = 2 * sm.items_b - 30 ∧
  sm.cost_a = 22 ∧
  sm.cost_b = 30 ∧
  sm.price_a = 29 ∧
  sm.price_b = 40 ∧
  sp.sm = sm ∧
  sp.items_b_new = 3 * sm.items_b ∧
  sp.discount_price_b = sm.price_b / 2 ∧
  sp.total_profit = 2350 →
  ∃ (discount_items : ℕ), 
    discount_items = 70 ∧
    (sm.price_a - sm.cost_a) * sm.items_a + 
    (sm.price_b - sm.cost_b) * (sp.items_b_new - discount_items) +
    (sp.discount_price_b - sm.cost_b) * discount_items = sp.total_profit :=
by sorry


end NUMINAMATH_CALUDE_supermarket_discount_items_l2132_213293


namespace NUMINAMATH_CALUDE_total_fish_caught_l2132_213217

-- Define the types of fish
inductive FishType
| Trout
| Salmon
| Tuna

-- Define a function to calculate the pounds of fish caught for each type
def poundsCaught (fishType : FishType) : ℕ :=
  match fishType with
  | .Trout => 200
  | .Salmon => 200 + 200 / 2
  | .Tuna => 2 * (200 + 200 / 2)

-- Theorem statement
theorem total_fish_caught :
  (poundsCaught FishType.Trout) +
  (poundsCaught FishType.Salmon) +
  (poundsCaught FishType.Tuna) = 1100 := by
  sorry


end NUMINAMATH_CALUDE_total_fish_caught_l2132_213217


namespace NUMINAMATH_CALUDE_max_reciprocal_eccentricity_sum_l2132_213241

theorem max_reciprocal_eccentricity_sum (e₁ e₂ : ℝ) : 
  e₁ > 0 → e₂ > 0 → 
  (∃ b c : ℝ, b > 0 ∧ c > b ∧ 
    e₁ = c / Real.sqrt (c^2 + (2*b)^2) ∧ 
    e₂ = c / Real.sqrt (c^2 - b^2)) → 
  1/e₁^2 + 4/e₂^2 = 5 → 
  1/e₁ + 1/e₂ ≤ 5/2 :=
by sorry

end NUMINAMATH_CALUDE_max_reciprocal_eccentricity_sum_l2132_213241


namespace NUMINAMATH_CALUDE_no_integer_solutions_l2132_213267

theorem no_integer_solutions : ¬ ∃ (a b : ℤ), 3 * a^2 = b^2 + 1 := by
  sorry

end NUMINAMATH_CALUDE_no_integer_solutions_l2132_213267


namespace NUMINAMATH_CALUDE_smallest_common_multiple_of_9_and_6_l2132_213288

theorem smallest_common_multiple_of_9_and_6 : ∃ n : ℕ+, (∀ m : ℕ+, 9 ∣ m ∧ 6 ∣ m → n ≤ m) ∧ 9 ∣ n ∧ 6 ∣ n := by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_smallest_common_multiple_of_9_and_6_l2132_213288


namespace NUMINAMATH_CALUDE_complex_modulus_l2132_213246

theorem complex_modulus (z : ℂ) (h : z + Complex.I = 3) : Complex.abs z = Real.sqrt 10 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_l2132_213246


namespace NUMINAMATH_CALUDE_intersection_of_M_and_N_l2132_213284

-- Define the sets M and N
def M : Set ℝ := {x | -1 ≤ x ∧ x ≤ 2}
def N : Set ℝ := {y | ∃ x, y = 2^x}

-- State the theorem
theorem intersection_of_M_and_N :
  M ∩ N = {x | 0 < x ∧ x ≤ 2} :=
sorry

end NUMINAMATH_CALUDE_intersection_of_M_and_N_l2132_213284


namespace NUMINAMATH_CALUDE_derivative_y_l2132_213242

def y (x : ℝ) : ℝ := x^2 - 5*x + 4

theorem derivative_y (x : ℝ) : 
  deriv y x = 2*x - 5 := by sorry

end NUMINAMATH_CALUDE_derivative_y_l2132_213242


namespace NUMINAMATH_CALUDE_starting_lineup_combinations_l2132_213220

/-- The number of players on the team -/
def total_players : ℕ := 15

/-- The number of players in the starting lineup -/
def lineup_size : ℕ := 6

/-- The number of pre-selected players (All-Stars) -/
def preselected_players : ℕ := 3

/-- The number of different starting lineups possible -/
def num_lineups : ℕ := 220

theorem starting_lineup_combinations :
  Nat.choose (total_players - preselected_players) (lineup_size - preselected_players) = num_lineups :=
sorry

end NUMINAMATH_CALUDE_starting_lineup_combinations_l2132_213220


namespace NUMINAMATH_CALUDE_share_a_plus_c_equals_6952_l2132_213227

def total_money : ℕ := 15800
def ratio_a : ℕ := 5
def ratio_b : ℕ := 9
def ratio_c : ℕ := 6
def ratio_d : ℕ := 5

def total_ratio : ℕ := ratio_a + ratio_b + ratio_c + ratio_d

theorem share_a_plus_c_equals_6952 :
  (ratio_a + ratio_c) * (total_money / total_ratio) = 6952 := by
  sorry

end NUMINAMATH_CALUDE_share_a_plus_c_equals_6952_l2132_213227


namespace NUMINAMATH_CALUDE_point_not_in_fourth_quadrant_l2132_213249

theorem point_not_in_fourth_quadrant (m : ℝ) :
  ¬(m - 1 > 0 ∧ m + 3 < 0) :=
by sorry

end NUMINAMATH_CALUDE_point_not_in_fourth_quadrant_l2132_213249


namespace NUMINAMATH_CALUDE_S_is_three_rays_with_common_point_l2132_213210

-- Define the set S
def S : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | let x := p.1; let y := p.2;
    (3 = x + 2 ∧ y - 4 ≤ 3) ∨
    (3 = y - 4 ∧ x + 2 ≤ 3) ∨
    (x + 2 = y - 4 ∧ 3 ≤ x + 2)}

-- Define a ray
def Ray (start : ℝ × ℝ) (direction : ℝ × ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | ∃ t : ℝ, t ≥ 0 ∧ p = (start.1 + t * direction.1, start.2 + t * direction.2)}

-- Theorem statement
theorem S_is_three_rays_with_common_point :
  ∃ (common_point : ℝ × ℝ) (ray1 ray2 ray3 : Set (ℝ × ℝ)),
    S = ray1 ∪ ray2 ∪ ray3 ∧
    (∃ (d1 d2 d3 : ℝ × ℝ),
      ray1 = Ray common_point d1 ∧
      ray2 = Ray common_point d2 ∧
      ray3 = Ray common_point d3) ∧
    ray1 ∩ ray2 = {common_point} ∧
    ray2 ∩ ray3 = {common_point} ∧
    ray3 ∩ ray1 = {common_point} :=
  sorry


end NUMINAMATH_CALUDE_S_is_three_rays_with_common_point_l2132_213210


namespace NUMINAMATH_CALUDE_negation_of_exponential_inequality_l2132_213273

theorem negation_of_exponential_inequality :
  (¬ ∀ x : ℝ, x ≤ 0 → Real.exp x ≤ 1) ↔ (∃ x₀ : ℝ, x₀ ≤ 0 ∧ Real.exp x₀ > 1) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_exponential_inequality_l2132_213273


namespace NUMINAMATH_CALUDE_max_coeff_seventh_term_l2132_213208

theorem max_coeff_seventh_term (n : ℕ) : 
  (∃ k, (Nat.choose n k = Nat.choose n 6) ∧ 
        (∀ j, 0 ≤ j ∧ j ≤ n → Nat.choose n j ≤ Nat.choose n 6)) →
  n ∈ ({11, 12, 13} : Set ℕ) := by
sorry

end NUMINAMATH_CALUDE_max_coeff_seventh_term_l2132_213208


namespace NUMINAMATH_CALUDE_postage_for_420g_book_l2132_213266

/-- Calculates the postage cost for mailing a book in China. -/
def postage_cost (weight : ℕ) : ℚ :=
  let base_rate : ℚ := 7/10
  let additional_rate : ℚ := 4/10
  let base_weight : ℕ := 100
  let additional_weight := (weight - 1) / base_weight + 1
  base_rate + additional_rate * additional_weight

/-- Theorem stating that the postage cost for a 420g book is 2.3 yuan. -/
theorem postage_for_420g_book :
  postage_cost 420 = 23/10 := by sorry

end NUMINAMATH_CALUDE_postage_for_420g_book_l2132_213266


namespace NUMINAMATH_CALUDE_studentB_is_optimal_l2132_213265

-- Define the structure for a student
structure Student where
  name : String
  average : ℝ
  variance : ℝ

-- Define the students
def studentA : Student := { name := "A", average := 92, variance := 3.6 }
def studentB : Student := { name := "B", average := 95, variance := 3.6 }
def studentC : Student := { name := "C", average := 95, variance := 7.4 }
def studentD : Student := { name := "D", average := 95, variance := 8.1 }

-- Define the list of all students
def students : List Student := [studentA, studentB, studentC, studentD]

-- Function to determine if one student is better than another
def isBetterStudent (s1 s2 : Student) : Prop :=
  (s1.average > s2.average) ∨ (s1.average = s2.average ∧ s1.variance < s2.variance)

-- Theorem stating that student B is the optimal choice
theorem studentB_is_optimal : 
  ∀ s ∈ students, s.name ≠ "B" → isBetterStudent studentB s :=
by sorry

end NUMINAMATH_CALUDE_studentB_is_optimal_l2132_213265


namespace NUMINAMATH_CALUDE_choose_officers_count_l2132_213294

/-- Represents the club with its member composition -/
structure Club where
  total_members : Nat
  boys : Nat
  girls : Nat
  senior_boys : Nat
  senior_girls : Nat

/-- Calculates the number of ways to choose a president and vice-president -/
def choose_officers (club : Club) : Nat :=
  (club.senior_boys * (club.boys - 1)) + (club.senior_girls * (club.girls - 1))

/-- The specific club instance from the problem -/
def our_club : Club :=
  { total_members := 30
  , boys := 16
  , girls := 14
  , senior_boys := 3
  , senior_girls := 3 
  }

/-- Theorem stating that the number of ways to choose officers for our club is 84 -/
theorem choose_officers_count : choose_officers our_club = 84 := by
  sorry

#eval choose_officers our_club

end NUMINAMATH_CALUDE_choose_officers_count_l2132_213294


namespace NUMINAMATH_CALUDE_earliest_meeting_time_l2132_213275

def lapTime1 : ℕ := 5
def lapTime2 : ℕ := 8
def lapTime3 : ℕ := 9
def startTime : ℕ := 7 * 60  -- 7:00 AM in minutes since midnight

def meetingTime : ℕ := startTime + Nat.lcm (Nat.lcm lapTime1 lapTime2) lapTime3

theorem earliest_meeting_time :
  meetingTime = 13 * 60  -- 1:00 PM in minutes since midnight
  := by sorry

end NUMINAMATH_CALUDE_earliest_meeting_time_l2132_213275


namespace NUMINAMATH_CALUDE_border_area_calculation_l2132_213207

/-- Given a rectangular photograph and its frame, calculate the area of the border. -/
theorem border_area_calculation (photo_height photo_width border_width : ℝ) 
  (h1 : photo_height = 12)
  (h2 : photo_width = 15)
  (h3 : border_width = 3) :
  (photo_height + 2 * border_width) * (photo_width + 2 * border_width) - 
  photo_height * photo_width = 198 := by
  sorry

#check border_area_calculation

end NUMINAMATH_CALUDE_border_area_calculation_l2132_213207


namespace NUMINAMATH_CALUDE_train_length_l2132_213296

/-- The length of a train given its speed and time to cross a pole -/
theorem train_length (speed_kmh : ℝ) (time_s : ℝ) : 
  speed_kmh = 54 → time_s = 9 → speed_kmh * (1000 / 3600) * time_s = 135 := by
  sorry

#check train_length

end NUMINAMATH_CALUDE_train_length_l2132_213296


namespace NUMINAMATH_CALUDE_base_of_negative_four_cubed_l2132_213236

def base_of_power (x : ℤ) (n : ℕ) : ℤ := x

theorem base_of_negative_four_cubed :
  base_of_power (-4) 3 = -4 := by sorry

end NUMINAMATH_CALUDE_base_of_negative_four_cubed_l2132_213236


namespace NUMINAMATH_CALUDE_binomial_identities_l2132_213270

theorem binomial_identities (n k : ℕ+) :
  (Nat.choose n k + Nat.choose n (k + 1) = Nat.choose (n + 1) (k + 1)) ∧
  (Nat.choose n k = (n / k) * Nat.choose (n - 1) (k - 1)) := by
  sorry

end NUMINAMATH_CALUDE_binomial_identities_l2132_213270


namespace NUMINAMATH_CALUDE_modulus_of_z_equals_one_l2132_213276

open Complex

theorem modulus_of_z_equals_one : 
  let z : ℂ := (1 - I) / (1 + I) + 2 * I
  abs z = 1 := by sorry

end NUMINAMATH_CALUDE_modulus_of_z_equals_one_l2132_213276


namespace NUMINAMATH_CALUDE_sequence_contains_large_number_l2132_213281

theorem sequence_contains_large_number 
  (seq : Fin 20 → ℕ) 
  (distinct : ∀ i j, i ≠ j → seq i ≠ seq j) 
  (consecutive_product_square : ∀ i : Fin 19, ∃ k : ℕ, seq i * seq (i.succ) = k * k) 
  (first_num : seq 0 = 42) :
  ∃ i : Fin 20, seq i > 16000 := by
  sorry

end NUMINAMATH_CALUDE_sequence_contains_large_number_l2132_213281


namespace NUMINAMATH_CALUDE_arithmetic_sequence_middle_term_l2132_213272

theorem arithmetic_sequence_middle_term (a₁ a₃ z : ℤ) : 
  a₁ = 2^3 → a₃ = 2^5 → z = (a₁ + a₃) / 2 → z = 20 := by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_middle_term_l2132_213272


namespace NUMINAMATH_CALUDE_odd_even_sum_difference_problem_statement_l2132_213218

theorem odd_even_sum_difference : ℕ → Prop :=
  fun n =>
    let odd_sum := (n^2 + 2*n + 1)^2
    let even_sum := n * (n + 1) * (2*n + 2)
    odd_sum - even_sum = 3057

theorem problem_statement : odd_even_sum_difference 1012 := by
  sorry

end NUMINAMATH_CALUDE_odd_even_sum_difference_problem_statement_l2132_213218


namespace NUMINAMATH_CALUDE_no_real_roots_l2132_213209

-- Define the operation ⊕
def oplus (m n : ℝ) : ℝ := n^2 - m*n + 1

-- Theorem statement
theorem no_real_roots :
  ∀ x : ℝ, oplus 1 x ≠ 0 := by
  sorry

end NUMINAMATH_CALUDE_no_real_roots_l2132_213209


namespace NUMINAMATH_CALUDE_survey_sample_size_l2132_213243

/-- Represents a survey with a given population size and number of selected participants. -/
structure Survey where
  population_size : ℕ
  selected_participants : ℕ

/-- Calculates the sample size of a given survey. -/
def sample_size (s : Survey) : ℕ := s.selected_participants

/-- Theorem stating that for a survey with 4000 students and 500 randomly selected,
    the sample size is 500. -/
theorem survey_sample_size :
  let s : Survey := { population_size := 4000, selected_participants := 500 }
  sample_size s = 500 := by sorry

end NUMINAMATH_CALUDE_survey_sample_size_l2132_213243


namespace NUMINAMATH_CALUDE_fifth_number_in_list_l2132_213280

theorem fifth_number_in_list (numbers : List ℕ) : 
  numbers.length = 9 ∧ 
  numbers.sum = 207 * 9 ∧
  201 ∈ numbers ∧ 
  202 ∈ numbers ∧ 
  204 ∈ numbers ∧ 
  205 ∈ numbers ∧ 
  209 ∈ numbers ∧ 
  209 ∈ numbers ∧ 
  210 ∈ numbers ∧ 
  212 ∈ numbers ∧ 
  212 ∈ numbers →
  ∃ (fifth : ℕ), fifth ∈ numbers ∧ fifth = 211 := by
sorry

end NUMINAMATH_CALUDE_fifth_number_in_list_l2132_213280


namespace NUMINAMATH_CALUDE_coplanar_vectors_k_value_l2132_213287

def a : ℝ × ℝ × ℝ := (1, -1, 2)
def b : ℝ × ℝ × ℝ := (-2, 1, 0)
def c (k : ℝ) : ℝ × ℝ × ℝ := (-3, 1, k)

theorem coplanar_vectors_k_value :
  ∀ k : ℝ, (∃ x y : ℝ, c k = x • a + y • b) → k = 2 := by
  sorry

end NUMINAMATH_CALUDE_coplanar_vectors_k_value_l2132_213287


namespace NUMINAMATH_CALUDE_drug_price_reduction_l2132_213214

theorem drug_price_reduction (initial_price final_price : ℝ) (x : ℝ) :
  initial_price = 63800 →
  final_price = 3900 →
  final_price = initial_price * (1 - x)^2 :=
by sorry

end NUMINAMATH_CALUDE_drug_price_reduction_l2132_213214


namespace NUMINAMATH_CALUDE_max_value_of_f_l2132_213248

theorem max_value_of_f (x : ℝ) (h : x < 3) : 
  (4 / (x - 3) + x) ≤ -1 ∧ ∃ y < 3, 4 / (y - 3) + y = -1 :=
sorry

end NUMINAMATH_CALUDE_max_value_of_f_l2132_213248


namespace NUMINAMATH_CALUDE_train_cars_count_l2132_213254

-- Define the given conditions
def cars_counted : ℕ := 10
def initial_time : ℕ := 15
def total_time : ℕ := 210  -- 3 minutes and 30 seconds in seconds

-- Define the theorem
theorem train_cars_count :
  let rate : ℚ := cars_counted / initial_time
  rate * total_time = 140 := by
  sorry

end NUMINAMATH_CALUDE_train_cars_count_l2132_213254


namespace NUMINAMATH_CALUDE_middle_card_is_five_l2132_213278

def is_valid_trio (a b c : ℕ) : Prop :=
  a < b ∧ b < c ∧ a + b + c = 16

def leftmost_uncertain (a : ℕ) : Prop :=
  ∃ b₁ c₁ b₂ c₂, b₁ ≠ b₂ ∧ is_valid_trio a b₁ c₁ ∧ is_valid_trio a b₂ c₂

def rightmost_uncertain (c : ℕ) : Prop :=
  ∃ a₁ b₁ a₂ b₂, a₁ ≠ a₂ ∧ is_valid_trio a₁ b₁ c ∧ is_valid_trio a₂ b₂ c

def middle_uncertain (b : ℕ) : Prop :=
  ∃ a₁ c₁ a₂ c₂, (a₁ ≠ a₂ ∨ c₁ ≠ c₂) ∧ is_valid_trio a₁ b c₁ ∧ is_valid_trio a₂ b c₂

theorem middle_card_is_five :
  ∀ a b c : ℕ,
    is_valid_trio a b c →
    leftmost_uncertain a →
    rightmost_uncertain c →
    middle_uncertain b →
    b = 5 := by sorry

end NUMINAMATH_CALUDE_middle_card_is_five_l2132_213278


namespace NUMINAMATH_CALUDE_ratio_w_to_y_l2132_213289

theorem ratio_w_to_y (w x y z : ℝ) 
  (hw_x : w / x = 5 / 2)
  (hy_z : y / z = 2 / 3)
  (hx_z : x / z = 10) :
  w / y = 37.5 := by
sorry

end NUMINAMATH_CALUDE_ratio_w_to_y_l2132_213289


namespace NUMINAMATH_CALUDE_multiplicative_magic_square_exists_l2132_213219

/-- Represents a 3x3 matrix --/
def Matrix3x3 := Fin 3 → Fin 3 → ℕ

/-- Check if two numbers have the same digits --/
def same_digits (a b : ℕ) : Prop := sorry

/-- The original magic square --/
def original_square : Matrix3x3 := 
  fun i j => match i, j with
  | 0, 0 => 27 | 0, 1 => 20 | 0, 2 => 25
  | 1, 0 => 22 | 1, 1 => 24 | 1, 2 => 26
  | 2, 0 => 23 | 2, 1 => 28 | 2, 2 => 21

/-- Product of elements in a row --/
def row_product (m : Matrix3x3) (i : Fin 3) : ℕ :=
  (m i 0) * (m i 1) * (m i 2)

/-- Product of elements in a column --/
def col_product (m : Matrix3x3) (j : Fin 3) : ℕ :=
  (m 0 j) * (m 1 j) * (m 2 j)

/-- Product of elements in the main diagonal --/
def diag_product (m : Matrix3x3) : ℕ :=
  (m 0 0) * (m 1 1) * (m 2 2)

/-- Product of elements in the anti-diagonal --/
def antidiag_product (m : Matrix3x3) : ℕ :=
  (m 0 2) * (m 1 1) * (m 2 0)

/-- The theorem to be proved --/
theorem multiplicative_magic_square_exists : ∃ (m : Matrix3x3), 
  (∀ i j, same_digits (m i j) (original_square i j)) ∧ 
  (∀ i : Fin 3, row_product m i = 7488) ∧
  (∀ j : Fin 3, col_product m j = 7488) ∧
  (diag_product m = 7488) ∧
  (antidiag_product m = 7488) := by
  sorry

end NUMINAMATH_CALUDE_multiplicative_magic_square_exists_l2132_213219


namespace NUMINAMATH_CALUDE_exists_number_plus_digit_sum_equals_2014_l2132_213283

/-- Sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- Theorem: There exists a natural number κ such that κ plus the sum of its digits equals 2014 -/
theorem exists_number_plus_digit_sum_equals_2014 : ∃ κ : ℕ, κ + sum_of_digits κ = 2014 := by
  sorry

end NUMINAMATH_CALUDE_exists_number_plus_digit_sum_equals_2014_l2132_213283


namespace NUMINAMATH_CALUDE_sandwiches_prepared_correct_l2132_213211

/-- The number of sandwiches Ruth prepared -/
def sandwiches_prepared : ℕ := 10

/-- The number of sandwiches Ruth ate -/
def sandwiches_ruth_ate : ℕ := 1

/-- The number of sandwiches Ruth gave to her brother -/
def sandwiches_given_to_brother : ℕ := 2

/-- The number of sandwiches eaten by the first cousin -/
def sandwiches_first_cousin : ℕ := 2

/-- The number of sandwiches eaten by each of the other two cousins -/
def sandwiches_per_other_cousin : ℕ := 1

/-- The number of other cousins who ate sandwiches -/
def number_of_other_cousins : ℕ := 2

/-- The number of sandwiches left at the end -/
def sandwiches_left : ℕ := 3

/-- Theorem stating that the number of sandwiches Ruth prepared is correct -/
theorem sandwiches_prepared_correct : 
  sandwiches_prepared = 
    sandwiches_ruth_ate + 
    sandwiches_given_to_brother + 
    sandwiches_first_cousin + 
    (sandwiches_per_other_cousin * number_of_other_cousins) + 
    sandwiches_left :=
by sorry

end NUMINAMATH_CALUDE_sandwiches_prepared_correct_l2132_213211


namespace NUMINAMATH_CALUDE_boric_acid_mixture_concentration_l2132_213233

/-- Given two boric acid solutions with concentrations and volumes, 
    calculate the concentration of the resulting mixture --/
theorem boric_acid_mixture_concentration 
  (c1 : ℝ) (c2 : ℝ) (v1 : ℝ) (v2 : ℝ) 
  (h1 : c1 = 0.01) -- 1% concentration
  (h2 : c2 = 0.05) -- 5% concentration
  (h3 : v1 = 15) -- 15 mL of first solution
  (h4 : v2 = 15) -- 15 mL of second solution
  : (c1 * v1 + c2 * v2) / (v1 + v2) = 0.03 := by
  sorry

#check boric_acid_mixture_concentration

end NUMINAMATH_CALUDE_boric_acid_mixture_concentration_l2132_213233


namespace NUMINAMATH_CALUDE_games_spent_proof_l2132_213286

def total_allowance : ℚ := 50

def books_fraction : ℚ := 1/4
def apps_fraction : ℚ := 3/10
def snacks_fraction : ℚ := 1/5

def books_spent : ℚ := total_allowance * books_fraction
def apps_spent : ℚ := total_allowance * apps_fraction
def snacks_spent : ℚ := total_allowance * snacks_fraction

def other_expenses : ℚ := books_spent + apps_spent + snacks_spent

theorem games_spent_proof : total_allowance - other_expenses = 25/2 := by sorry

end NUMINAMATH_CALUDE_games_spent_proof_l2132_213286


namespace NUMINAMATH_CALUDE_partial_fraction_decomposition_l2132_213216

theorem partial_fraction_decomposition :
  ∃ (C D : ℚ), C = 16/3 ∧ D = 5/3 ∧
  ∀ x : ℚ, x ≠ 12 ∧ x ≠ -3 →
    (7*x - 4) / (x^2 - 9*x - 36) = C / (x - 12) + D / (x + 3) :=
by sorry

end NUMINAMATH_CALUDE_partial_fraction_decomposition_l2132_213216


namespace NUMINAMATH_CALUDE_existence_and_uniqueness_l2132_213268

open Real

/-- The differential equation y' = y - x^2 + 2x - 2 -/
def diff_eq (x y : ℝ) : ℝ := y - x^2 + 2*x - 2

/-- A solution to the differential equation -/
def is_solution (f : ℝ → ℝ) : Prop :=
  ∀ x, (deriv f) x = diff_eq x (f x)

theorem existence_and_uniqueness :
  ∀ (x₀ y₀ : ℝ), ∃! f : ℝ → ℝ,
    is_solution f ∧ f x₀ = y₀ :=
sorry

end NUMINAMATH_CALUDE_existence_and_uniqueness_l2132_213268


namespace NUMINAMATH_CALUDE_successful_pair_existence_l2132_213274

/-- A pair of natural numbers is successful if their arithmetic mean and geometric mean are both natural numbers. -/
def IsSuccessfulPair (a b : ℕ) : Prop :=
  ∃ m g : ℕ, 2 * m = a + b ∧ g * g = a * b

theorem successful_pair_existence (m n k : ℕ) (h1 : m > n) (h2 : n > 0) (h3 : m > k) (h4 : k > 0)
  (h5 : IsSuccessfulPair (m + n) (m - n)) (h6 : m^2 - n^2 = k^2) :
  ∃ (a b : ℕ), a ≠ b ∧ IsSuccessfulPair a b ∧ 2 * m = a + b ∧ (a ≠ m + n ∨ b ≠ m - n) := by
  sorry

end NUMINAMATH_CALUDE_successful_pair_existence_l2132_213274


namespace NUMINAMATH_CALUDE_problem_solution_l2132_213224

-- Define the line y = ax - 2a + 4
def line_a (a : ℝ) (x : ℝ) : ℝ := a * x - 2 * a + 4

-- Define the point (2, 4)
def point_2_4 : ℝ × ℝ := (2, 4)

-- Define the line y + 1 = 3x
def line_3x (x : ℝ) : ℝ := 3 * x - 1

-- Define the line x + √3y + 1 = 0
def line_sqrt3 (x y : ℝ) : Prop := x + Real.sqrt 3 * y + 1 = 0

-- Define the point (-2, 3)
def point_neg2_3 : ℝ × ℝ := (-2, 3)

-- Define the line x - 2y + 3 = 0
def line_1 (x y : ℝ) : Prop := x - 2 * y + 3 = 0

-- Define the line 2x + y + 1 = 0
def line_2 (x y : ℝ) : Prop := 2 * x + y + 1 = 0

theorem problem_solution :
  (∀ a : ℝ, line_a a (point_2_4.1) = point_2_4.2) ∧
  (line_2 point_neg2_3.1 point_neg2_3.2 ∧
   ∀ x y : ℝ, line_1 x y → line_2 x y → x = y) :=
sorry

end NUMINAMATH_CALUDE_problem_solution_l2132_213224


namespace NUMINAMATH_CALUDE_second_shop_payment_l2132_213258

/-- The amount paid for books from the second shop -/
def amount_second_shop (books_first_shop : ℕ) (books_second_shop : ℕ) 
  (price_first_shop : ℚ) (average_price : ℚ) : ℚ := 
  (average_price * (books_first_shop + books_second_shop : ℚ)) - price_first_shop

/-- Theorem stating the amount paid for books from the second shop -/
theorem second_shop_payment : 
  amount_second_shop 65 50 1160 (18088695652173913 / 1000000000000000) = 920 := by
  sorry

end NUMINAMATH_CALUDE_second_shop_payment_l2132_213258


namespace NUMINAMATH_CALUDE_germs_left_percentage_l2132_213298

/-- Represents the effectiveness of four sanitizer sprays and their overlaps -/
structure SanitizerSprays where
  /-- Kill rates for each spray -/
  spray1 : ℝ
  spray2 : ℝ
  spray3 : ℝ
  spray4 : ℝ
  /-- Two-way overlaps between sprays -/
  overlap12 : ℝ
  overlap23 : ℝ
  overlap34 : ℝ
  overlap13 : ℝ
  overlap14 : ℝ
  overlap24 : ℝ
  /-- Three-way overlaps between sprays -/
  overlap123 : ℝ
  overlap234 : ℝ

/-- Calculates the percentage of germs left after applying all sprays -/
def germsLeft (s : SanitizerSprays) : ℝ :=
  100 - (s.spray1 + s.spray2 + s.spray3 + s.spray4 - 
         (s.overlap12 + s.overlap23 + s.overlap34 + s.overlap13 + s.overlap14 + s.overlap24) -
         (s.overlap123 + s.overlap234))

/-- Theorem stating that for the given spray effectiveness and overlaps, 13.8% of germs are left -/
theorem germs_left_percentage (s : SanitizerSprays) 
  (h1 : s.spray1 = 50) (h2 : s.spray2 = 35) (h3 : s.spray3 = 20) (h4 : s.spray4 = 10)
  (h5 : s.overlap12 = 10) (h6 : s.overlap23 = 7) (h7 : s.overlap34 = 5)
  (h8 : s.overlap13 = 3) (h9 : s.overlap14 = 2) (h10 : s.overlap24 = 1)
  (h11 : s.overlap123 = 0.5) (h12 : s.overlap234 = 0.3) :
  germsLeft s = 13.8 := by
  sorry


end NUMINAMATH_CALUDE_germs_left_percentage_l2132_213298


namespace NUMINAMATH_CALUDE_unique_arrangement_l2132_213277

structure Ball :=
  (color : String)

structure Box :=
  (color : String)
  (balls : List Ball)

def valid_arrangement (boxes : List Box) : Prop :=
  boxes.length = 3 ∧
  (∃ red_box white_box yellow_box,
    boxes = [red_box, white_box, yellow_box] ∧
    red_box.color = "red" ∧ white_box.color = "white" ∧ yellow_box.color = "yellow" ∧
    (∀ ball ∈ yellow_box.balls, ball.color = "white") ∧
    (∀ ball ∈ white_box.balls, ball.color = "red") ∧
    (∀ ball ∈ red_box.balls, ball.color = "yellow") ∧
    yellow_box.balls.length > (boxes.map (λ box => box.balls.filter (λ ball => ball.color = "yellow"))).join.length ∧
    red_box.balls.length ≠ (boxes.map (λ box => box.balls.filter (λ ball => ball.color = "white"))).join.length ∧
    (boxes.map (λ box => box.balls.filter (λ ball => ball.color = "white"))).join.length < white_box.balls.length)

theorem unique_arrangement (boxes : List Box) :
  valid_arrangement boxes →
  ∃ red_box white_box yellow_box,
    boxes = [red_box, white_box, yellow_box] ∧
    (∀ ball ∈ red_box.balls, ball.color = "yellow") ∧
    (∀ ball ∈ white_box.balls, ball.color = "red") ∧
    (∀ ball ∈ yellow_box.balls, ball.color = "white") :=
by sorry

end NUMINAMATH_CALUDE_unique_arrangement_l2132_213277


namespace NUMINAMATH_CALUDE_largest_d_for_negative_five_in_range_l2132_213200

/-- The function g(x) = x^2 + 2x + d -/
def g (d : ℝ) (x : ℝ) : ℝ := x^2 + 2*x + d

/-- The largest value of d such that -5 is in the range of g(x) -/
theorem largest_d_for_negative_five_in_range : 
  (∃ (d : ℝ), ∀ (e : ℝ), (∃ (x : ℝ), g e x = -5) → e ≤ d) ∧ 
  (∃ (x : ℝ), g (-4) x = -5) :=
sorry

end NUMINAMATH_CALUDE_largest_d_for_negative_five_in_range_l2132_213200


namespace NUMINAMATH_CALUDE_amelia_dinner_l2132_213269

def dinner_problem (initial_amount : ℝ) (first_course : ℝ) (second_course_extra : ℝ) (dessert_percentage : ℝ) : Prop :=
  let second_course := first_course + second_course_extra
  let dessert := dessert_percentage * second_course
  let total_cost := first_course + second_course + dessert
  let money_left := initial_amount - total_cost
  money_left = 20

theorem amelia_dinner :
  dinner_problem 60 15 5 0.25 := by
  sorry

end NUMINAMATH_CALUDE_amelia_dinner_l2132_213269


namespace NUMINAMATH_CALUDE_min_value_of_function_l2132_213282

theorem min_value_of_function (x : ℝ) (h : x > 0) : 
  2 * x + 1 / x^6 ≥ 3 ∧ ∃ y > 0, 2 * y + 1 / y^6 = 3 :=
sorry

end NUMINAMATH_CALUDE_min_value_of_function_l2132_213282


namespace NUMINAMATH_CALUDE_min_value_f_range_of_t_l2132_213251

-- Define the function f(x)
def f (x : ℝ) : ℝ := |x + 2| + |x - 4|

-- Theorem for the minimum value of f(x)
theorem min_value_f : ∀ x : ℝ, f x ≥ 6 := by sorry

-- Theorem for the range of t
theorem range_of_t (t : ℝ) : 
  (∃ x : ℝ, -3 ≤ x ∧ x ≤ 5 ∧ f x ≤ t^2 - t) ↔ (t ≤ -2 ∨ t ≥ 3) := by sorry

end NUMINAMATH_CALUDE_min_value_f_range_of_t_l2132_213251


namespace NUMINAMATH_CALUDE_initial_men_correct_l2132_213279

/-- The number of men initially working in a garment industry -/
def initial_men : ℕ := 12

/-- The number of hours worked per day in the initial scenario -/
def initial_hours_per_day : ℕ := 8

/-- The number of days worked in the initial scenario -/
def initial_days : ℕ := 10

/-- The number of men in the second scenario -/
def second_men : ℕ := 24

/-- The number of hours worked per day in the second scenario -/
def second_hours_per_day : ℕ := 5

/-- The number of days worked in the second scenario -/
def second_days : ℕ := 8

/-- Theorem stating that the initial number of men is correct -/
theorem initial_men_correct : 
  initial_men * initial_hours_per_day * initial_days = 
  second_men * second_hours_per_day * second_days :=
by
  sorry

#check initial_men_correct

end NUMINAMATH_CALUDE_initial_men_correct_l2132_213279


namespace NUMINAMATH_CALUDE_simplify_expression_l2132_213253

theorem simplify_expression (x : ℝ) : 3 * x^2 - 1 - 2*x - 5 + 3*x - x = 3 * x^2 - 6 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l2132_213253


namespace NUMINAMATH_CALUDE_courtyard_width_l2132_213226

/-- Proves that the width of a rectangular courtyard is 18 meters -/
theorem courtyard_width (length : ℝ) (brick_length : ℝ) (brick_width : ℝ) (total_bricks : ℕ) :
  length = 25 →
  brick_length = 0.2 →
  brick_width = 0.1 →
  total_bricks = 22500 →
  (length * (total_bricks : ℝ) * brick_length * brick_width) / length = 18 :=
by sorry

end NUMINAMATH_CALUDE_courtyard_width_l2132_213226


namespace NUMINAMATH_CALUDE_berry_theorem_l2132_213231

def berry_problem (total_needed : ℕ) (strawberries : ℕ) (blueberries : ℕ) (raspberries : ℕ) : ℕ :=
  total_needed - (strawberries + blueberries + raspberries)

theorem berry_theorem : berry_problem 50 18 12 7 = 13 := by
  sorry

end NUMINAMATH_CALUDE_berry_theorem_l2132_213231


namespace NUMINAMATH_CALUDE_ray_gave_25_cents_to_peter_l2132_213235

/-- Represents the value of a nickel in cents -/
def nickel_value : ℕ := 5

/-- Represents the initial amount Ray had in cents -/
def initial_amount : ℕ := 95

/-- Represents the number of nickels Ray had left -/
def nickels_left : ℕ := 4

/-- Represents the amount given to Peter in cents -/
def amount_to_peter : ℕ := 25

/-- Proves that Ray gave 25 cents to Peter given the initial conditions -/
theorem ray_gave_25_cents_to_peter :
  let total_given := initial_amount - (nickels_left * nickel_value)
  let amount_to_randi := 2 * amount_to_peter
  total_given = amount_to_peter + amount_to_randi :=
by sorry

end NUMINAMATH_CALUDE_ray_gave_25_cents_to_peter_l2132_213235


namespace NUMINAMATH_CALUDE_cos_540_degrees_l2132_213244

theorem cos_540_degrees : Real.cos (540 * π / 180) = -1 := by
  sorry

end NUMINAMATH_CALUDE_cos_540_degrees_l2132_213244


namespace NUMINAMATH_CALUDE_school_fee_calculation_l2132_213290

/-- Represents the amount of money given by Luke's mother -/
def mother_contribution : ℕ :=
  50 + 2 * 20 + 3 * 10

/-- Represents the amount of money given by Luke's father -/
def father_contribution : ℕ :=
  4 * 50 + 20 + 10

/-- Represents the total school fee -/
def school_fee : ℕ :=
  mother_contribution + father_contribution

theorem school_fee_calculation :
  school_fee = 350 :=
by sorry

end NUMINAMATH_CALUDE_school_fee_calculation_l2132_213290
