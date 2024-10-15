import Mathlib

namespace NUMINAMATH_CALUDE_fifth_term_is_32_l2611_261101

/-- A sequence where the difference between each term and its predecessor increases by 3 each time -/
def special_sequence : ℕ → ℕ
| 0 => 2
| 1 => 5
| n + 2 => special_sequence (n + 1) + 3 * (n + 1)

theorem fifth_term_is_32 : special_sequence 4 = 32 := by
  sorry

end NUMINAMATH_CALUDE_fifth_term_is_32_l2611_261101


namespace NUMINAMATH_CALUDE_f_even_and_increasing_l2611_261108

-- Define the function f(x) = |x| + 1
def f (x : ℝ) : ℝ := |x| + 1

-- State the theorem
theorem f_even_and_increasing :
  (∀ x : ℝ, f x = f (-x)) ∧ 
  (∀ x y : ℝ, 0 < x → x < y → f x < f y) :=
sorry

end NUMINAMATH_CALUDE_f_even_and_increasing_l2611_261108


namespace NUMINAMATH_CALUDE_only_D_positive_l2611_261162

theorem only_D_positive :
  let a := -3 + 7 - 5
  let b := (1 - 2) * 3
  let c := -16 / ((-3)^2)
  let d := -(2^4) * (-6)
  (a ≤ 0 ∧ b ≤ 0 ∧ c ≤ 0 ∧ d > 0) := by sorry

end NUMINAMATH_CALUDE_only_D_positive_l2611_261162


namespace NUMINAMATH_CALUDE_quadratic_rewrite_l2611_261129

theorem quadratic_rewrite (k : ℝ) :
  ∃ (d r s : ℝ), 9 * k^2 - 6 * k + 15 = d * (k + r)^2 + s ∧ s / r = -42 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_rewrite_l2611_261129


namespace NUMINAMATH_CALUDE_finite_crosses_in_circle_l2611_261183

/-- A cross formed by the diagonals of a square with side length 1 -/
def Cross : Type := Unit

/-- A circle with radius 100 -/
def Circle : Type := Unit

/-- The maximum number of non-overlapping crosses that can fit inside the circle -/
noncomputable def maxCrosses : ℕ := sorry

/-- The theorem stating that the number of non-overlapping crosses that can fit inside the circle is finite -/
theorem finite_crosses_in_circle : ∃ n : ℕ, maxCrosses ≤ n := by sorry

end NUMINAMATH_CALUDE_finite_crosses_in_circle_l2611_261183


namespace NUMINAMATH_CALUDE_line_passes_through_point_min_triangle_area_min_area_line_equation_l2611_261147

/-- Definition of the line l with parameter k -/
def line_l (k : ℝ) (x y : ℝ) : Prop := k * x - y + 4 * k + 2 = 0

/-- Theorem stating that the line l always passes through the point (-4, 2) -/
theorem line_passes_through_point (k : ℝ) : line_l k (-4) 2 := by sorry

/-- Definition of the area of the triangle formed by the line and coordinate axes -/
noncomputable def triangle_area (k : ℝ) : ℝ := sorry

/-- Theorem stating the minimum area of the triangle -/
theorem min_triangle_area : 
  ∃ (k : ℝ), triangle_area k = 16 ∧ ∀ (k' : ℝ), triangle_area k' ≥ 16 := by sorry

/-- Theorem stating the equation of the line when the area is minimum -/
theorem min_area_line_equation (k : ℝ) : 
  triangle_area k = 16 → line_l k x y ↔ x - 2 * y + 8 = 0 := by sorry

end NUMINAMATH_CALUDE_line_passes_through_point_min_triangle_area_min_area_line_equation_l2611_261147


namespace NUMINAMATH_CALUDE_special_ellipse_equation_l2611_261100

/-- An ellipse with specific properties -/
structure SpecialEllipse where
  /-- The center of the ellipse is at the origin -/
  center_at_origin : Unit
  /-- The endpoints of the minor axis are at (0, ±1) -/
  minor_axis_endpoints : Unit
  /-- The eccentricity of the ellipse -/
  e : ℝ
  /-- The product of the eccentricity of this ellipse and that of the hyperbola y^2 - x^2 = 1 is 1 -/
  eccentricity_product : e * Real.sqrt 2 = 1

/-- The equation of the special ellipse -/
def ellipse_equation (E : SpecialEllipse) (x y : ℝ) : Prop :=
  x^2 / 2 + y^2 = 1

/-- Theorem stating that the given ellipse has the equation x^2/2 + y^2 = 1 -/
theorem special_ellipse_equation (E : SpecialEllipse) :
  ∀ x y : ℝ, ellipse_equation E x y :=
sorry

end NUMINAMATH_CALUDE_special_ellipse_equation_l2611_261100


namespace NUMINAMATH_CALUDE_imaginary_part_of_i_times_one_plus_i_l2611_261173

theorem imaginary_part_of_i_times_one_plus_i (i : ℂ) : i * i = -1 → Complex.im (i * (1 + i)) = 1 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_i_times_one_plus_i_l2611_261173


namespace NUMINAMATH_CALUDE_green_tiles_in_50th_row_l2611_261103

/-- Represents the number of tiles in a row of the tiling pattern. -/
def num_tiles (n : ℕ) : ℕ := 2 * n - 1

/-- Represents the number of green tiles in a row of the tiling pattern. -/
def num_green_tiles (n : ℕ) : ℕ := (num_tiles n - 1) / 2

theorem green_tiles_in_50th_row :
  num_green_tiles 50 = 49 := by sorry

end NUMINAMATH_CALUDE_green_tiles_in_50th_row_l2611_261103


namespace NUMINAMATH_CALUDE_candy_ratio_is_five_thirds_l2611_261121

/-- Represents the number of M&M candies Penelope has -/
def mm_candies : ℕ := 25

/-- Represents the number of Starbursts candies Penelope has -/
def starbursts_candies : ℕ := 15

/-- Represents the ratio of M&M candies to Starbursts candies -/
def candy_ratio : Rat := mm_candies / starbursts_candies

theorem candy_ratio_is_five_thirds : candy_ratio = 5 / 3 := by
  sorry

end NUMINAMATH_CALUDE_candy_ratio_is_five_thirds_l2611_261121


namespace NUMINAMATH_CALUDE_reciprocal_sum_fractions_l2611_261106

theorem reciprocal_sum_fractions : 
  (1 / (1/4 + 1/6 + 1/9) : ℚ) = 36/19 := by sorry

end NUMINAMATH_CALUDE_reciprocal_sum_fractions_l2611_261106


namespace NUMINAMATH_CALUDE_initial_mixture_volume_l2611_261172

/-- Proves that the initial volume of a milk-water mixture is 45 litres given specific conditions -/
theorem initial_mixture_volume (initial_milk : ℝ) (initial_water : ℝ) : 
  initial_milk / initial_water = 4 →
  initial_milk / (initial_water + 11) = 1.8 →
  initial_milk + initial_water = 45 :=
by
  sorry

#check initial_mixture_volume

end NUMINAMATH_CALUDE_initial_mixture_volume_l2611_261172


namespace NUMINAMATH_CALUDE_rosa_phone_book_pages_l2611_261122

/-- Rosa's phone book calling problem -/
theorem rosa_phone_book_pages : 
  let week1_pages : ℝ := 10.2
  let week2_pages : ℝ := 8.6
  let week3_pages : ℝ := 12.4
  week1_pages + week2_pages + week3_pages = 31.2 :=
by sorry

end NUMINAMATH_CALUDE_rosa_phone_book_pages_l2611_261122


namespace NUMINAMATH_CALUDE_h_function_iff_strictly_increasing_l2611_261151

/-- A function f is an H-function if for any two distinct real numbers x₁ and x₂,
    x₁ * f x₁ + x₂ * f x₂ > x₁ * f x₂ + x₂ * f x₁ -/
def is_h_function (f : ℝ → ℝ) : Prop :=
  ∀ x₁ x₂ : ℝ, x₁ ≠ x₂ → x₁ * f x₁ + x₂ * f x₂ > x₁ * f x₂ + x₂ * f x₁

/-- A function f is strictly increasing if for any two real numbers x₁ and x₂,
    x₁ < x₂ implies f x₁ < f x₂ -/
def strictly_increasing (f : ℝ → ℝ) : Prop :=
  ∀ x₁ x₂ : ℝ, x₁ < x₂ → f x₁ < f x₂

theorem h_function_iff_strictly_increasing (f : ℝ → ℝ) :
  is_h_function f ↔ strictly_increasing f :=
sorry

end NUMINAMATH_CALUDE_h_function_iff_strictly_increasing_l2611_261151


namespace NUMINAMATH_CALUDE_positive_x_solution_l2611_261140

/-- Given a system of equations, prove that the positive solution for x is 3 -/
theorem positive_x_solution (x y z : ℝ) 
  (eq1 : x * y = 6 - 2*x - 3*y)
  (eq2 : y * z = 6 - 4*y - 2*z)
  (eq3 : x * z = 30 - 4*x - 3*z)
  (x_pos : x > 0) :
  x = 3 := by
  sorry

end NUMINAMATH_CALUDE_positive_x_solution_l2611_261140


namespace NUMINAMATH_CALUDE_composition_result_l2611_261148

-- Define the two operations
def op1 (x : ℝ) : ℝ := 8 - x
def op2 (x : ℝ) : ℝ := x - 8

-- Notation for the operations
notation:max x "&" => op1 x
prefix:max "&" => op2

-- Theorem statement
theorem composition_result : &(15&) = -15 := by sorry

end NUMINAMATH_CALUDE_composition_result_l2611_261148


namespace NUMINAMATH_CALUDE_encoded_bec_value_l2611_261128

/-- Represents the encoding of a base 7 digit --/
inductive Encoding
  | A | B | C | D | E | F | G

/-- Represents a number in the encoded form --/
def EncodedNumber := List Encoding

/-- Converts an EncodedNumber to its base 10 representation --/
def to_base_10 (n : EncodedNumber) : ℕ := sorry

/-- Checks if three EncodedNumbers are consecutive integers --/
def are_consecutive (a b c : EncodedNumber) : Prop := sorry

theorem encoded_bec_value :
  ∃ (encode : Fin 7 → Encoding),
    Function.Injective encode ∧
    (∃ (x : ℕ), 
      are_consecutive 
        [encode (x % 7), encode ((x + 1) % 7), encode ((x + 2) % 7)]
        [encode ((x + 1) % 7), encode ((x + 2) % 7), encode ((x + 3) % 7)]
        [encode ((x + 2) % 7), encode ((x + 3) % 7), encode ((x + 4) % 7)]) →
    to_base_10 [Encoding.B, Encoding.E, Encoding.C] = 336 :=
sorry

end NUMINAMATH_CALUDE_encoded_bec_value_l2611_261128


namespace NUMINAMATH_CALUDE_power_difference_equality_l2611_261134

theorem power_difference_equality : (3^2)^3 - (2^3)^2 = 665 := by sorry

end NUMINAMATH_CALUDE_power_difference_equality_l2611_261134


namespace NUMINAMATH_CALUDE_negation_of_existence_leq_negation_of_proposition_l2611_261178

theorem negation_of_existence_leq (p : ℝ → Prop) :
  (¬ ∃ x₀ : ℝ, p x₀) ↔ (∀ x : ℝ, ¬ p x) :=
by sorry

theorem negation_of_proposition :
  (¬ ∃ x₀ : ℝ, Real.exp x₀ - x₀ - 1 ≤ 0) ↔ (∀ x : ℝ, Real.exp x - x - 1 > 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_existence_leq_negation_of_proposition_l2611_261178


namespace NUMINAMATH_CALUDE_tournament_games_count_l2611_261127

/-- Calculates the number of games in a round-robin tournament for a given number of teams -/
def roundRobinGames (n : ℕ) : ℕ := n * (n - 1) / 2

/-- Calculates the number of games in knockout rounds for a given number of teams -/
def knockoutGames (n : ℕ) : ℕ := n - 1

theorem tournament_games_count :
  let totalTeams : ℕ := 32
  let groupCount : ℕ := 8
  let teamsPerGroup : ℕ := 4
  let advancingTeams : ℕ := 2
  
  totalTeams = groupCount * teamsPerGroup →
  
  (groupCount * roundRobinGames teamsPerGroup) +
  (knockoutGames (groupCount * advancingTeams)) = 63 := by
  sorry

end NUMINAMATH_CALUDE_tournament_games_count_l2611_261127


namespace NUMINAMATH_CALUDE_factorization_x4_plus_81_l2611_261185

theorem factorization_x4_plus_81 (x : ℂ) : x^4 + 81 = (x^2 + 9*I)*(x^2 - 9*I) := by
  sorry

end NUMINAMATH_CALUDE_factorization_x4_plus_81_l2611_261185


namespace NUMINAMATH_CALUDE_x_squared_minus_y_squared_l2611_261142

theorem x_squared_minus_y_squared (x y : ℚ) 
  (h1 : x + y = 5/12) (h2 : x - y = 1/36) : x^2 - y^2 = 5/432 := by
  sorry

end NUMINAMATH_CALUDE_x_squared_minus_y_squared_l2611_261142


namespace NUMINAMATH_CALUDE_fraction_product_theorem_l2611_261150

theorem fraction_product_theorem : 
  (7 / 4 : ℚ) * (14 / 49 : ℚ) * (10 / 15 : ℚ) * (12 / 36 : ℚ) * 
  (21 / 14 : ℚ) * (40 / 80 : ℚ) * (33 / 22 : ℚ) * (16 / 64 : ℚ) = 1 / 12 := by
  sorry

end NUMINAMATH_CALUDE_fraction_product_theorem_l2611_261150


namespace NUMINAMATH_CALUDE_pool_filling_time_l2611_261139

theorem pool_filling_time (pipe1 pipe2 pipe3 pipe4 : ℚ) 
  (h1 : pipe1 = 1)
  (h2 : pipe2 = 1/2)
  (h3 : pipe3 = 1/3)
  (h4 : pipe4 = 1/4) :
  1 / (pipe1 + pipe2 + pipe3 + pipe4) = 12/25 := by sorry

end NUMINAMATH_CALUDE_pool_filling_time_l2611_261139


namespace NUMINAMATH_CALUDE_manufacturing_degrees_l2611_261155

/-- Represents the number of degrees in a full circle. -/
def full_circle : ℝ := 360

/-- Represents the percentage of employees in manufacturing as a decimal. -/
def manufacturing_percentage : ℝ := 0.20

/-- Calculates the number of degrees in a circle graph for a given percentage. -/
def degrees_for_percentage (percentage : ℝ) : ℝ := full_circle * percentage

/-- Theorem: The manufacturing section in the circle graph takes up 72 degrees. -/
theorem manufacturing_degrees :
  degrees_for_percentage manufacturing_percentage = 72 := by
  sorry

end NUMINAMATH_CALUDE_manufacturing_degrees_l2611_261155


namespace NUMINAMATH_CALUDE_distribute_four_to_three_l2611_261118

/-- The number of ways to distribute n distinct objects into k distinct boxes,
    where each box must contain at least one object. -/
def distribute (n k : ℕ) : ℕ := sorry

/-- Theorem: There are 36 ways to distribute 4 distinct objects into 3 distinct boxes,
    where each box must contain at least one object. -/
theorem distribute_four_to_three : distribute 4 3 = 36 := by sorry

end NUMINAMATH_CALUDE_distribute_four_to_three_l2611_261118


namespace NUMINAMATH_CALUDE_age_difference_l2611_261124

theorem age_difference (A B C : ℤ) (h : A + B = B + C + 11) : A - C = 11 := by
  sorry

end NUMINAMATH_CALUDE_age_difference_l2611_261124


namespace NUMINAMATH_CALUDE_sarah_speed_calculation_l2611_261107

def eugene_speed : ℚ := 5

def carlos_speed_ratio : ℚ := 4/5

def sarah_speed_ratio : ℚ := 6/7

def carlos_speed : ℚ := eugene_speed * carlos_speed_ratio

def sarah_speed : ℚ := carlos_speed * sarah_speed_ratio

theorem sarah_speed_calculation : sarah_speed = 24/7 := by
  sorry

end NUMINAMATH_CALUDE_sarah_speed_calculation_l2611_261107


namespace NUMINAMATH_CALUDE_horner_method_value_l2611_261194

def horner_polynomial (coeffs : List ℤ) (x : ℤ) : ℤ :=
  coeffs.foldl (fun acc a => acc * x + a) 0

def f (x : ℤ) : ℤ :=
  horner_polynomial [3, 5, 6, 79, -8, 35, 12] x

theorem horner_method_value :
  f (-4) = 220 := by
  sorry

end NUMINAMATH_CALUDE_horner_method_value_l2611_261194


namespace NUMINAMATH_CALUDE_expression_evaluation_l2611_261163

theorem expression_evaluation : ((15^15 / 15^14)^3 * 3^5) / 9^2 = 10120 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2611_261163


namespace NUMINAMATH_CALUDE_focus_line_dot_product_fixed_point_existence_l2611_261166

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 4*x

-- Define the focus of the parabola
def focus : ℝ × ℝ := (1, 0)

-- Define a line that intersects the parabola at two distinct points
def intersecting_line (t b : ℝ) (x y : ℝ) : Prop := x = t*y + b

-- Define the dot product of two vectors
def dot_product (x1 y1 x2 y2 : ℝ) : ℝ := x1*x2 + y1*y2

-- Part I: Theorem for line passing through focus
theorem focus_line_dot_product (t : ℝ) (x1 y1 x2 y2 : ℝ) :
  parabola x1 y1 ∧ parabola x2 y2 ∧
  intersecting_line t 1 x1 y1 ∧ intersecting_line t 1 x2 y2 ∧
  (x1 ≠ x2 ∨ y1 ≠ y2) →
  dot_product x1 y1 x2 y2 = -3 :=
sorry

-- Part II: Theorem for fixed point
theorem fixed_point_existence (t b : ℝ) (x1 y1 x2 y2 : ℝ) :
  parabola x1 y1 ∧ parabola x2 y2 ∧
  intersecting_line t b x1 y1 ∧ intersecting_line t b x2 y2 ∧
  (x1 ≠ x2 ∨ y1 ≠ y2) ∧
  dot_product x1 y1 x2 y2 = -4 →
  b = 2 :=
sorry

end NUMINAMATH_CALUDE_focus_line_dot_product_fixed_point_existence_l2611_261166


namespace NUMINAMATH_CALUDE_construction_paper_count_l2611_261105

/-- Represents the number of sheets in a pack of construction paper -/
structure ConstructionPaper where
  blue : ℕ
  red : ℕ

/-- Represents the daily usage of construction paper -/
structure DailyUsage where
  blue : ℕ
  red : ℕ

def initial_ratio (pack : ConstructionPaper) : Prop :=
  pack.blue * 7 = pack.red * 2

def daily_usage : DailyUsage :=
  { blue := 1, red := 3 }

def last_day_usage : DailyUsage :=
  { blue := 1, red := 3 }

def remaining_red : ℕ := 15

theorem construction_paper_count :
  ∃ (pack : ConstructionPaper),
    initial_ratio pack ∧
    ∃ (days : ℕ),
      pack.blue = daily_usage.blue * days + last_day_usage.blue ∧
      pack.red = daily_usage.red * days + last_day_usage.red + remaining_red ∧
      pack.blue + pack.red = 135 :=
sorry

end NUMINAMATH_CALUDE_construction_paper_count_l2611_261105


namespace NUMINAMATH_CALUDE_work_completion_time_l2611_261182

theorem work_completion_time (a_half_time b_third_time : ℝ) 
  (ha : a_half_time = 70)
  (hb : b_third_time = 35) :
  let a_rate := 1 / (2 * a_half_time)
  let b_rate := 1 / (3 * b_third_time)
  1 / (a_rate + b_rate) = 60 := by sorry

end NUMINAMATH_CALUDE_work_completion_time_l2611_261182


namespace NUMINAMATH_CALUDE_number_ordering_l2611_261143

theorem number_ordering : (4 : ℚ) / 5 < (801 : ℚ) / 1000 ∧ (801 : ℚ) / 1000 < 81 / 100 := by
  sorry

end NUMINAMATH_CALUDE_number_ordering_l2611_261143


namespace NUMINAMATH_CALUDE_rug_dimension_l2611_261117

theorem rug_dimension (x : ℝ) : 
  x > 0 ∧ 
  x ≤ 8 ∧
  7 ≤ 8 ∧
  x * 7 = 64 * (1 - 0.78125) →
  x = 2 := by
sorry

end NUMINAMATH_CALUDE_rug_dimension_l2611_261117


namespace NUMINAMATH_CALUDE_max_value_of_expression_l2611_261120

def S : Set ℝ := {1, 2, 3, 5, 10}

theorem max_value_of_expression (x y : ℝ) (hx : x ∈ S) (hy : y ∈ S) :
  (x / y + y / x) ≤ 10.1 :=
sorry

end NUMINAMATH_CALUDE_max_value_of_expression_l2611_261120


namespace NUMINAMATH_CALUDE_line_passes_through_fixed_point_l2611_261144

/-- The line equation passes through a fixed point for all values of m -/
theorem line_passes_through_fixed_point :
  ∀ (m : ℝ), (2 + m) * (-1) + (1 - 2*m) * (-2) + 4 - 3*m = 0 := by
  sorry

end NUMINAMATH_CALUDE_line_passes_through_fixed_point_l2611_261144


namespace NUMINAMATH_CALUDE_perimeter_decrease_percentage_l2611_261113

/-- Represents a rectangle with length and width --/
structure Rectangle where
  length : ℝ
  width : ℝ

/-- Calculates the perimeter of a rectangle --/
def perimeter (r : Rectangle) : ℝ := 2 * (r.length + r.width)

/-- Theorem: Perimeter decrease percentage for different length and width reductions --/
theorem perimeter_decrease_percentage
  (r : Rectangle)
  (h1 : perimeter { length := 0.9 * r.length, width := 0.8 * r.width } = 0.88 * perimeter r) :
  perimeter { length := 0.8 * r.length, width := 0.9 * r.width } = 0.82 * perimeter r := by
  sorry


end NUMINAMATH_CALUDE_perimeter_decrease_percentage_l2611_261113


namespace NUMINAMATH_CALUDE_rectangle_perimeter_l2611_261130

theorem rectangle_perimeter (area : ℝ) (side : ℝ) (h1 : area = 108) (h2 : side = 12) :
  2 * (side + area / side) = 42 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_perimeter_l2611_261130


namespace NUMINAMATH_CALUDE_triangle_angle_sum_triangle_angle_sum_is_540_l2611_261102

theorem triangle_angle_sum : ℝ → Prop :=
  fun total_sum =>
    ∃ (int_angles ext_angles : ℝ),
      (int_angles = 180) ∧
      (ext_angles = 360) ∧
      (total_sum = int_angles + ext_angles)

theorem triangle_angle_sum_is_540 : 
  triangle_angle_sum 540 := by sorry

end NUMINAMATH_CALUDE_triangle_angle_sum_triangle_angle_sum_is_540_l2611_261102


namespace NUMINAMATH_CALUDE_sin_1200_degrees_l2611_261111

theorem sin_1200_degrees : Real.sin (1200 * Real.pi / 180) = Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_1200_degrees_l2611_261111


namespace NUMINAMATH_CALUDE_unique_number_l2611_261115

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

def is_odd (n : ℕ) : Prop := n % 2 = 1

def is_multiple_of_9 (n : ℕ) : Prop := n % 9 = 0

def digits_product (n : ℕ) : ℕ := (n / 10) * (n % 10)

def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, m * m = n

theorem unique_number :
  ∃! n : ℕ, is_two_digit n ∧ is_odd n ∧ is_multiple_of_9 n ∧ is_perfect_square (digits_product n) ∧ n = 99 :=
sorry

end NUMINAMATH_CALUDE_unique_number_l2611_261115


namespace NUMINAMATH_CALUDE_probability_estimate_l2611_261189

def is_hit (d : Nat) : Bool := d ≥ 2 ∧ d ≤ 9

def group_has_three_hits (g : List Nat) : Bool :=
  (g.filter is_hit).length ≥ 3

def count_successful_groups (groups : List (List Nat)) : Nat :=
  (groups.filter group_has_three_hits).length

theorem probability_estimate (groups : List (List Nat))
  (h1 : groups.length = 20)
  (h2 : ∀ g ∈ groups, g.length = 4)
  (h3 : ∀ g ∈ groups, ∀ d ∈ g, d ≤ 9)
  (h4 : count_successful_groups groups = 15) :
  (count_successful_groups groups : ℚ) / groups.length = 3/4 := by
  sorry

#check probability_estimate

end NUMINAMATH_CALUDE_probability_estimate_l2611_261189


namespace NUMINAMATH_CALUDE_converse_proposition_l2611_261167

theorem converse_proposition : 
  (∀ x : ℝ, x = 3 → x^2 - 2*x - 3 = 0) ↔ 
  (∀ x : ℝ, x^2 - 2*x - 3 = 0 → x = 3) :=
by sorry

end NUMINAMATH_CALUDE_converse_proposition_l2611_261167


namespace NUMINAMATH_CALUDE_x_less_than_y_l2611_261177

theorem x_less_than_y (a b : ℝ) (h1 : 0 < a) (h2 : a < b) 
  (x y : ℝ) (hx : x = (0.1993 : ℝ)^b * (0.1997 : ℝ)^a) 
  (hy : y = (0.1993 : ℝ)^a * (0.1997 : ℝ)^b) : x < y := by
  sorry

end NUMINAMATH_CALUDE_x_less_than_y_l2611_261177


namespace NUMINAMATH_CALUDE_intersection_with_complement_l2611_261114

open Set

-- Define the sets A and B
def A : Set ℝ := {x | -1 ≤ x ∧ x ≤ 2}
def B : Set ℝ := {x | x < 1}

-- State the theorem
theorem intersection_with_complement : 
  A ∩ (𝒰 \ B) = {x : ℝ | 1 ≤ x ∧ x ≤ 2} := by sorry

end NUMINAMATH_CALUDE_intersection_with_complement_l2611_261114


namespace NUMINAMATH_CALUDE_power_product_equality_l2611_261176

theorem power_product_equality (a b : ℝ) : (a^3 * b)^2 = a^6 * b^2 := by
  sorry

end NUMINAMATH_CALUDE_power_product_equality_l2611_261176


namespace NUMINAMATH_CALUDE_certain_number_is_seven_l2611_261171

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

theorem certain_number_is_seven (n : ℕ) (h : factorial 9 / factorial n = 72) : n = 7 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_is_seven_l2611_261171


namespace NUMINAMATH_CALUDE_two_mono_triangles_probability_l2611_261112

/-- A complete graph K6 with edges colored either green or yellow -/
structure ColoredK6 where
  edges : Fin 15 → Bool  -- True for green, False for yellow

/-- The probability of an edge being green -/
def p_green : ℚ := 2/3

/-- The probability of an edge being yellow -/
def p_yellow : ℚ := 1/3

/-- The probability of a specific triangle being monochromatic -/
def p_mono_triangle : ℚ := 1/3

/-- The total number of triangles in K6 -/
def total_triangles : ℕ := 20

/-- The probability of exactly two monochromatic triangles in a ColoredK6 -/
def prob_two_mono_triangles : ℚ := 49807360/3486784401

theorem two_mono_triangles_probability (g : ColoredK6) : 
  prob_two_mono_triangles = (total_triangles.choose 2 : ℚ) * p_mono_triangle^2 * (1 - p_mono_triangle)^(total_triangles - 2) :=
sorry

end NUMINAMATH_CALUDE_two_mono_triangles_probability_l2611_261112


namespace NUMINAMATH_CALUDE_max_value_ratio_l2611_261135

theorem max_value_ratio (x y k : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hk : k ≠ 0) (h : x = k * y) :
  ∃ (M : ℝ), M = 1 ∧ ∀ x y k, x ≠ 0 → y ≠ 0 → k ≠ 0 → x = k * y →
    |x + y| / (|x| + |y|) ≤ M ∧ ∃ x y k, x ≠ 0 ∧ y ≠ 0 ∧ k ≠ 0 ∧ x = k * y ∧ |x + y| / (|x| + |y|) = M :=
sorry

end NUMINAMATH_CALUDE_max_value_ratio_l2611_261135


namespace NUMINAMATH_CALUDE_basketball_team_selection_l2611_261138

theorem basketball_team_selection (n : ℕ) (k : ℕ) (twins : ℕ) : 
  n = 15 → k = 5 → twins = 2 →
  (Nat.choose n k) - (Nat.choose (n - twins) k) = 1716 := by
sorry

end NUMINAMATH_CALUDE_basketball_team_selection_l2611_261138


namespace NUMINAMATH_CALUDE_odd_function_derivative_range_l2611_261159

open Real

theorem odd_function_derivative_range (f : ℝ → ℝ) (t : ℝ) :
  (∀ x, f (-x) = -f x) →  -- f is odd
  (∀ x ∈ Set.Ioo (-1) 1, deriv f x = 5 + cos x) →  -- f'(x) = 5 + cos(x) for x ∈ (-1, 1)
  (f (1 - t) + f (1 - t^2) < 0) →  -- given condition
  t ∈ Set.Ioo 1 (sqrt 2) :=  -- t ∈ (1, √2)
by sorry

end NUMINAMATH_CALUDE_odd_function_derivative_range_l2611_261159


namespace NUMINAMATH_CALUDE_min_value_in_region_D_l2611_261180

def region_D (x y : ℝ) : Prop :=
  y ≤ x ∧ y ≥ -x ∧ x ≤ (Real.sqrt 2) / 2

def objective_function (x y : ℝ) : ℝ :=
  x - 2 * y

theorem min_value_in_region_D :
  ∃ (min : ℝ), min = -(Real.sqrt 2) / 2 ∧
  ∀ (x y : ℝ), region_D x y → objective_function x y ≥ min :=
sorry

end NUMINAMATH_CALUDE_min_value_in_region_D_l2611_261180


namespace NUMINAMATH_CALUDE_first_stack_height_is_correct_l2611_261137

/-- The height of the first stack of blocks -/
def first_stack_height : ℕ := 7

/-- The height of the second stack of blocks -/
def second_stack_height : ℕ := first_stack_height + 5

/-- The height of the third stack of blocks -/
def third_stack_height : ℕ := second_stack_height + 7

/-- The number of blocks that fell from the second stack -/
def fallen_second_stack : ℕ := second_stack_height - 2

/-- The number of blocks that fell from the third stack -/
def fallen_third_stack : ℕ := third_stack_height - 3

/-- The total number of fallen blocks -/
def total_fallen_blocks : ℕ := 33

theorem first_stack_height_is_correct :
  first_stack_height + fallen_second_stack + fallen_third_stack = total_fallen_blocks :=
by sorry

end NUMINAMATH_CALUDE_first_stack_height_is_correct_l2611_261137


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l2611_261198

theorem quadratic_equation_solution :
  let x₁ : ℝ := (1 + Real.sqrt 3) / 2
  let x₂ : ℝ := (1 - Real.sqrt 3) / 2
  2 * x₁^2 - 2 * x₁ - 1 = 0 ∧ 2 * x₂^2 - 2 * x₂ - 1 = 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l2611_261198


namespace NUMINAMATH_CALUDE_third_group_men_count_l2611_261192

/-- The work rate of one man -/
def man_rate : ℝ := sorry

/-- The work rate of one woman -/
def woman_rate : ℝ := sorry

/-- The number of men in the third group -/
def x : ℕ := sorry

theorem third_group_men_count : x = 5 := by
  have h1 : 3 * man_rate + 8 * woman_rate = 6 * man_rate + 2 * woman_rate := by sorry
  have h2 : x * man_rate + 2 * woman_rate = (6/7) * (3 * man_rate + 8 * woman_rate) := by sorry
  sorry

end NUMINAMATH_CALUDE_third_group_men_count_l2611_261192


namespace NUMINAMATH_CALUDE_cubic_quadratic_comparison_l2611_261191

theorem cubic_quadratic_comparison (n : ℝ) :
  (n > -1 → n^3 + 1 > n^2 + n) ∧ (n < -1 → n^3 + 1 < n^2 + n) := by
  sorry

end NUMINAMATH_CALUDE_cubic_quadratic_comparison_l2611_261191


namespace NUMINAMATH_CALUDE_quadratic_distinct_roots_l2611_261187

theorem quadratic_distinct_roots (c : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ x₁^2 + 2*x₁ + 4*c = 0 ∧ x₂^2 + 2*x₂ + 4*c = 0) ↔ c < 1/4 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_distinct_roots_l2611_261187


namespace NUMINAMATH_CALUDE_ellipse_k_range_l2611_261188

/-- 
Given a real number k, if the equation x²/(9-k) + y²/(k-1) = 1 represents an ellipse 
with foci on the y-axis, then 5 < k < 9.
-/
theorem ellipse_k_range (k : ℝ) : 
  (∀ x y : ℝ, x^2 / (9 - k) + y^2 / (k - 1) = 1) → -- equation represents an ellipse
  (9 - k > 0) →  -- condition for ellipse
  (k - 1 > 0) →  -- condition for ellipse
  (k - 1 > 9 - k) →  -- foci on y-axis condition
  (5 < k ∧ k < 9) := by
sorry

end NUMINAMATH_CALUDE_ellipse_k_range_l2611_261188


namespace NUMINAMATH_CALUDE_expression_value_l2611_261109

theorem expression_value : (2018 - 18 + 20) / 2 = 1010 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l2611_261109


namespace NUMINAMATH_CALUDE_f_one_zero_range_l2611_261110

/-- The quadratic function f(x) = 3ax^2 - 2ax + 1 -/
def f (a : ℝ) (x : ℝ) : ℝ := 3 * a * x^2 - 2 * a * x + 1

/-- The property that f has exactly one zero in the interval [-1, 1] -/
def has_one_zero_in_interval (a : ℝ) : Prop :=
  ∃! x : ℝ, x ∈ Set.Icc (-1) 1 ∧ f a x = 0

/-- The theorem stating the range of a for which f has exactly one zero in [-1, 1] -/
theorem f_one_zero_range :
  ∀ a : ℝ, has_one_zero_in_interval a ↔ a = 3 ∨ (-1 < a ∧ a ≤ -1/5) :=
sorry

end NUMINAMATH_CALUDE_f_one_zero_range_l2611_261110


namespace NUMINAMATH_CALUDE_max_value_theorem_l2611_261186

theorem max_value_theorem (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : 1/a + 1/b = 2) :
  ∃ (max : ℝ), max = 25/8 ∧ ∀ (x y : ℝ), x > 0 → y > 0 → 1/x + 1/y = 2 → 1/y * (2/x + 1) ≤ max :=
sorry

end NUMINAMATH_CALUDE_max_value_theorem_l2611_261186


namespace NUMINAMATH_CALUDE_canoe_kayak_ratio_l2611_261153

/-- Represents the rental business scenario --/
structure RentalBusiness where
  canoe_cost : ℕ
  kayak_cost : ℕ
  total_revenue : ℕ
  canoe_kayak_difference : ℕ

/-- Theorem stating the ratio of canoes to kayaks rented --/
theorem canoe_kayak_ratio (rb : RentalBusiness)
  (h1 : rb.canoe_cost = 11)
  (h2 : rb.kayak_cost = 16)
  (h3 : rb.total_revenue = 460)
  (h4 : rb.canoe_kayak_difference = 5) :
  ∃ (c k : ℕ), c = k + rb.canoe_kayak_difference ∧ 
                rb.canoe_cost * c + rb.kayak_cost * k = rb.total_revenue ∧
                c * 3 = k * 4 := by
  sorry

end NUMINAMATH_CALUDE_canoe_kayak_ratio_l2611_261153


namespace NUMINAMATH_CALUDE_equation_solution_l2611_261132

theorem equation_solution : 
  ∀ x : ℝ, (2010 + x)^2 = 4*x^2 ↔ x = 2010 ∨ x = -670 := by
sorry

end NUMINAMATH_CALUDE_equation_solution_l2611_261132


namespace NUMINAMATH_CALUDE_percentage_not_sold_approx_l2611_261199

def initial_stock : ℕ := 1100
def monday_sales : ℕ := 75
def tuesday_sales : ℕ := 50
def wednesday_sales : ℕ := 64
def thursday_sales : ℕ := 78
def friday_sales : ℕ := 135

def total_sales : ℕ := monday_sales + tuesday_sales + wednesday_sales + thursday_sales + friday_sales

def books_not_sold : ℕ := initial_stock - total_sales

def percentage_not_sold : ℚ := (books_not_sold : ℚ) / (initial_stock : ℚ) * 100

theorem percentage_not_sold_approx :
  ∃ (ε : ℚ), ε > 0 ∧ ε < 0.01 ∧ |percentage_not_sold - 63.45| < ε :=
sorry

end NUMINAMATH_CALUDE_percentage_not_sold_approx_l2611_261199


namespace NUMINAMATH_CALUDE_cuboid_dimensions_sum_l2611_261133

theorem cuboid_dimensions_sum (A B C : ℝ) (h1 : A * B = 45) (h2 : B * C = 80) (h3 : C * A = 180) :
  A + B + C = 145 / 9 := by
sorry

end NUMINAMATH_CALUDE_cuboid_dimensions_sum_l2611_261133


namespace NUMINAMATH_CALUDE_opposite_of_five_l2611_261165

-- Define the concept of opposite
def opposite (a : ℝ) : ℝ := -a

-- Theorem statement
theorem opposite_of_five : opposite 5 = -5 := by
  -- The proof goes here
  sorry

-- Lemma to show that the opposite satisfies the required property
lemma opposite_property (a : ℝ) : a + opposite a = 0 := by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_opposite_of_five_l2611_261165


namespace NUMINAMATH_CALUDE_difference_of_squares_division_l2611_261125

theorem difference_of_squares_division : (245^2 - 225^2) / 20 = 470 := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_division_l2611_261125


namespace NUMINAMATH_CALUDE_function_equation_solution_l2611_261141

theorem function_equation_solution (f : ℚ → ℚ) 
  (h0 : f 0 = 0)
  (h1 : ∀ x y : ℚ, f (f x + f y) = x + y) :
  (∀ x : ℚ, f x = x) ∨ (∀ x : ℚ, f x = -x) := by
sorry

end NUMINAMATH_CALUDE_function_equation_solution_l2611_261141


namespace NUMINAMATH_CALUDE_marie_messages_theorem_l2611_261158

/-- Calculates the number of days required to read all unread messages. -/
def daysToReadMessages (initialUnread : ℕ) (readPerDay : ℕ) (newPerDay : ℕ) : ℕ :=
  if readPerDay ≤ newPerDay then 0  -- Cannot finish if receiving more than reading
  else (initialUnread + (newPerDay - 1)) / (readPerDay - newPerDay)

theorem marie_messages_theorem :
  daysToReadMessages 98 20 6 = 7 := by
sorry

end NUMINAMATH_CALUDE_marie_messages_theorem_l2611_261158


namespace NUMINAMATH_CALUDE_fraction_puzzle_l2611_261119

theorem fraction_puzzle (x y : ℚ) (h_sum : x + y = 3/4) (h_prod : x * y = 1/8) : 
  min x y = 1/4 := by
sorry

end NUMINAMATH_CALUDE_fraction_puzzle_l2611_261119


namespace NUMINAMATH_CALUDE_intersection_with_complement_l2611_261160

-- Define the sets A and B
def A : Set ℝ := {x | x ≤ 3}
def B : Set ℝ := {x | x < 2}

-- State the theorem
theorem intersection_with_complement : A ∩ (Set.univ \ B) = Set.Icc 2 3 := by
  sorry

end NUMINAMATH_CALUDE_intersection_with_complement_l2611_261160


namespace NUMINAMATH_CALUDE_amc8_paths_count_l2611_261193

/-- Represents a position on the grid --/
structure Position :=
  (x : Int) (y : Int)

/-- Represents a letter on the grid --/
inductive Letter
  | A | M | C | Eight

/-- Defines the grid layout --/
def grid : Position → Letter := sorry

/-- Checks if two positions are adjacent --/
def isAdjacent (p1 p2 : Position) : Bool := sorry

/-- Defines a valid path on the grid --/
def ValidPath : List Position → Prop := sorry

/-- Counts the number of valid paths spelling AMC8 --/
def countAMC8Paths : Nat := sorry

/-- Theorem stating that the number of valid AMC8 paths is 24 --/
theorem amc8_paths_count : countAMC8Paths = 24 := by sorry

end NUMINAMATH_CALUDE_amc8_paths_count_l2611_261193


namespace NUMINAMATH_CALUDE_mixed_tea_sale_price_l2611_261136

/-- Calculates the sale price of mixed tea to earn a specified profit -/
theorem mixed_tea_sale_price
  (tea1_weight : ℝ)
  (tea1_cost : ℝ)
  (tea2_weight : ℝ)
  (tea2_cost : ℝ)
  (profit_percentage : ℝ)
  (h1 : tea1_weight = 80)
  (h2 : tea1_cost = 15)
  (h3 : tea2_weight = 20)
  (h4 : tea2_cost = 20)
  (h5 : profit_percentage = 20)
  : ∃ (sale_price : ℝ), sale_price = 19.2 := by
  sorry

#check mixed_tea_sale_price

end NUMINAMATH_CALUDE_mixed_tea_sale_price_l2611_261136


namespace NUMINAMATH_CALUDE_sugar_water_and_triangle_inequalities_l2611_261156

theorem sugar_water_and_triangle_inequalities :
  (∀ x y m : ℝ, x > y ∧ y > 0 ∧ m > 0 → y / x < (y + m) / (x + m)) ∧
  (∀ a b c : ℝ, a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b > c ∧ b + c > a ∧ c + a > b →
    a / (b + c) + b / (a + c) + c / (a + b) < 2) :=
by sorry

end NUMINAMATH_CALUDE_sugar_water_and_triangle_inequalities_l2611_261156


namespace NUMINAMATH_CALUDE_inscribed_sphere_cone_relation_l2611_261197

/-- A right cone with a sphere inscribed in it -/
structure InscribedSphereCone where
  base_radius : ℝ
  height : ℝ
  sphere_radius : ℝ
  b : ℝ
  d : ℝ
  sphere_radius_eq : sphere_radius = b * (Real.sqrt d - 1)

/-- The theorem stating the relationship between b and d for the given cone and sphere -/
theorem inscribed_sphere_cone_relation (cone : InscribedSphereCone) 
  (h1 : cone.base_radius = 15)
  (h2 : cone.height = 30) :
  cone.b + cone.d = 12.5 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_sphere_cone_relation_l2611_261197


namespace NUMINAMATH_CALUDE_f_properties_l2611_261181

noncomputable def f (x : ℝ) : ℝ := -Real.sqrt 3 * Real.sin x ^ 2 + Real.sin x * Real.cos x

theorem f_properties :
  (f (25 * Real.pi / 6) = 0) ∧
  (∃ T : ℝ, T > 0 ∧ ∀ x : ℝ, f (x + T) = f x ∧ ∀ S : ℝ, S > 0 ∧ (∀ x : ℝ, f (x + S) = f x) → T ≤ S) ∧
  (∃ x_max : ℝ, x_max ∈ Set.Icc 0 (Real.pi / 2) ∧ 
    f x_max = (2 - Real.sqrt 3) / 2 ∧
    ∀ x ∈ Set.Icc 0 (Real.pi / 2), f x ≤ f x_max) ∧
  (∃ x_min : ℝ, x_min ∈ Set.Icc 0 (Real.pi / 2) ∧ 
    f x_min = -Real.sqrt 3 ∧
    ∀ x ∈ Set.Icc 0 (Real.pi / 2), f x_min ≤ f x) :=
by sorry

#check f_properties

end NUMINAMATH_CALUDE_f_properties_l2611_261181


namespace NUMINAMATH_CALUDE_all_propositions_false_l2611_261175

-- Define the basic geometric concepts
def Line : Type := sorry
def Plane : Type := sorry

-- Define the geometric relations
def perpendicular (l1 l2 : Line) : Prop := sorry
def parallel (l1 l2 : Line) : Prop := sorry
def parallel_planes (p1 p2 : Plane) : Prop := sorry
def perpendicular_to_plane (l : Line) (p : Plane) : Prop := sorry
def angle_with_plane (l : Line) (p : Plane) : ℝ := sorry
def intersect (l1 l2 : Line) : Prop := sorry
def skew (l1 l2 : Line) : Prop := sorry

-- State the propositions
def proposition1 : Prop :=
  ∀ l1 l2 l3 : Line, perpendicular l1 l3 → perpendicular l2 l3 → parallel l1 l2

def proposition2 : Prop :=
  ∀ p1 p2 p3 : Plane, perpendicular_to_plane p1 p3 → perpendicular_to_plane p2 p3 → parallel_planes p1 p2

def proposition3 : Prop :=
  ∀ l1 l2 : Line, ∀ p : Plane, angle_with_plane l1 p = angle_with_plane l2 p → parallel l1 l2

def proposition4 : Prop :=
  ∀ l1 l2 l3 l4 : Line, skew l1 l2 → intersect l3 l1 → intersect l3 l2 → intersect l4 l1 → intersect l4 l2 → skew l3 l4

-- Theorem stating that all propositions are false
theorem all_propositions_false :
  ¬proposition1 ∧ ¬proposition2 ∧ ¬proposition3 ∧ ¬proposition4 := by sorry

end NUMINAMATH_CALUDE_all_propositions_false_l2611_261175


namespace NUMINAMATH_CALUDE_quadratic_root_value_l2611_261184

theorem quadratic_root_value (d : ℝ) : 
  (∀ x : ℝ, x^2 + 9*x + d = 0 ↔ x = (-9 + Real.sqrt d) / 2 ∨ x = (-9 - Real.sqrt d) / 2) →
  d = 16.2 := by
sorry

end NUMINAMATH_CALUDE_quadratic_root_value_l2611_261184


namespace NUMINAMATH_CALUDE_negation_of_sum_of_squares_zero_l2611_261152

theorem negation_of_sum_of_squares_zero (a b : ℝ) :
  ¬(a^2 + b^2 = 0) ↔ (a ≠ 0 ∧ b ≠ 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_sum_of_squares_zero_l2611_261152


namespace NUMINAMATH_CALUDE_sin_cos_difference_zero_l2611_261196

theorem sin_cos_difference_zero : Real.sin (36 * π / 180) * Real.cos (36 * π / 180) - Real.cos (36 * π / 180) * Real.sin (36 * π / 180) = 0 := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_difference_zero_l2611_261196


namespace NUMINAMATH_CALUDE_perpendicular_lines_from_parallel_planes_l2611_261116

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (parallel_planes : Plane → Plane → Prop)
variable (perpendicular_line_plane : Line → Plane → Prop)
variable (parallel_line_plane : Line → Plane → Prop)
variable (perpendicular_lines : Line → Line → Prop)

-- State the theorem
theorem perpendicular_lines_from_parallel_planes
  (α β : Plane) (m n : Line)
  (h1 : parallel_planes α β)
  (h2 : perpendicular_line_plane m α)
  (h3 : parallel_line_plane n β) :
  perpendicular_lines m n :=
sorry

end NUMINAMATH_CALUDE_perpendicular_lines_from_parallel_planes_l2611_261116


namespace NUMINAMATH_CALUDE_handball_league_female_fraction_l2611_261149

/-- Represents the handball league participation data --/
structure LeagueData where
  male_last_year : ℕ
  total_increase_rate : ℚ
  male_increase_rate : ℚ
  female_increase_rate : ℚ

/-- Calculates the fraction of female participants in the current year --/
def female_fraction (data : LeagueData) : ℚ :=
  -- The actual calculation would go here
  13/27

/-- Theorem stating that given the specific conditions, the fraction of female participants is 13/27 --/
theorem handball_league_female_fraction :
  let data : LeagueData := {
    male_last_year := 25,
    total_increase_rate := 1/5,  -- 20% increase
    male_increase_rate := 1/10,  -- 10% increase
    female_increase_rate := 3/10 -- 30% increase
  }
  female_fraction data = 13/27 := by
  sorry


end NUMINAMATH_CALUDE_handball_league_female_fraction_l2611_261149


namespace NUMINAMATH_CALUDE_binomial_prob_three_l2611_261169

/-- A random variable following a binomial distribution B(n, p) -/
structure BinomialRV (n : ℕ) (p : ℝ) where
  prob : ℝ → ℝ

/-- The probability mass function of a binomial distribution -/
def binomial_pmf (n : ℕ) (p : ℝ) (k : ℕ) : ℝ :=
  (n.choose k) * p^k * (1 - p)^(n - k)

theorem binomial_prob_three (ξ : BinomialRV 5 (1/3)) :
  ξ.prob 3 = 40/243 := by sorry

end NUMINAMATH_CALUDE_binomial_prob_three_l2611_261169


namespace NUMINAMATH_CALUDE_f_neg_l2611_261168

-- Define an odd function f on the real numbers
def f : ℝ → ℝ := sorry

-- State the properties of f
axiom f_odd : ∀ x, f (-x) = -f x
axiom f_pos : ∀ x > 0, f x = x * (1 - x)

-- Theorem to prove
theorem f_neg : ∀ x < 0, f x = x * (1 + x) := by sorry

end NUMINAMATH_CALUDE_f_neg_l2611_261168


namespace NUMINAMATH_CALUDE_even_operations_l2611_261190

-- Define an even integer
def is_even (n : ℤ) : Prop := ∃ k : ℤ, n = 2 * k

-- Define a perfect square
def is_perfect_square (n : ℤ) : Prop := ∃ k : ℤ, n = k * k

theorem even_operations (n : ℤ) (h : is_even n) :
  (is_even (n^2)) ∧ 
  (∀ m : ℤ, is_even m → is_perfect_square m → is_even (Int.sqrt m)) ∧
  (∀ k : ℤ, ¬(is_even k) → is_even (n * k)) ∧
  (is_even (n^3)) :=
sorry

end NUMINAMATH_CALUDE_even_operations_l2611_261190


namespace NUMINAMATH_CALUDE_find_other_number_l2611_261195

theorem find_other_number (x y : ℤ) (h1 : 2*x + 3*y = 100) (h2 : x = 28 ∨ y = 28) : x = 8 ∨ y = 8 := by
  sorry

end NUMINAMATH_CALUDE_find_other_number_l2611_261195


namespace NUMINAMATH_CALUDE_f_lower_bound_l2611_261161

open Real

noncomputable def f (m : ℝ) (x : ℝ) : ℝ := m / x + log x

theorem f_lower_bound (m : ℝ) (x : ℝ) (hm : m > 0) (hx : x > 0) :
  m * f m x ≥ 2 * m - 1 := by
  sorry

end NUMINAMATH_CALUDE_f_lower_bound_l2611_261161


namespace NUMINAMATH_CALUDE_cow_feeding_problem_l2611_261170

theorem cow_feeding_problem (daily_feed : ℕ) (total_feed : ℕ) 
  (h1 : daily_feed = 28) (h2 : total_feed = 890) :
  ∃ (days : ℕ) (leftover : ℕ), 
    days * daily_feed + leftover = total_feed ∧ 
    days = 31 ∧ 
    leftover = 22 := by
  sorry

end NUMINAMATH_CALUDE_cow_feeding_problem_l2611_261170


namespace NUMINAMATH_CALUDE_square_floor_with_57_black_tiles_has_841_total_tiles_l2611_261104

/-- Represents a square floor tiled with congruent square tiles. -/
structure SquareFloor where
  side_length : ℕ
  is_valid : side_length > 0

/-- Calculates the number of black tiles on the diagonals of a square floor. -/
def black_tiles (floor : SquareFloor) : ℕ :=
  2 * floor.side_length - 1

/-- Calculates the total number of tiles on a square floor. -/
def total_tiles (floor : SquareFloor) : ℕ :=
  floor.side_length * floor.side_length

/-- Theorem stating that a square floor with 57 black tiles on its diagonals has 841 total tiles. -/
theorem square_floor_with_57_black_tiles_has_841_total_tiles :
  ∀ (floor : SquareFloor), black_tiles floor = 57 → total_tiles floor = 841 :=
by
  sorry

end NUMINAMATH_CALUDE_square_floor_with_57_black_tiles_has_841_total_tiles_l2611_261104


namespace NUMINAMATH_CALUDE_gcd_of_powers_minus_one_l2611_261179

theorem gcd_of_powers_minus_one (a m n : ℕ) :
  Nat.gcd (a^m - 1) (a^n - 1) = a^(Nat.gcd m n) - 1 :=
by sorry

end NUMINAMATH_CALUDE_gcd_of_powers_minus_one_l2611_261179


namespace NUMINAMATH_CALUDE_correct_average_l2611_261146

theorem correct_average (n : ℕ) (initial_avg : ℚ) (wrong_num correct_num : ℚ) :
  n = 10 →
  initial_avg = 15 →
  wrong_num = 26 →
  correct_num = 36 →
  (n : ℚ) * initial_avg + (correct_num - wrong_num) = n * 16 := by
  sorry

end NUMINAMATH_CALUDE_correct_average_l2611_261146


namespace NUMINAMATH_CALUDE_oil_container_distribution_l2611_261131

theorem oil_container_distribution :
  ∃ (n m k : ℕ),
    n + m + k = 100 ∧
    n + 10 * m + 50 * k = 500 ∧
    n = 60 ∧ m = 39 ∧ k = 1 := by
  sorry

end NUMINAMATH_CALUDE_oil_container_distribution_l2611_261131


namespace NUMINAMATH_CALUDE_nested_average_calculation_l2611_261145

-- Define the average of two numbers
def avg2 (a b : ℚ) : ℚ := (a + b) / 2

-- Define the average of three numbers
def avg3 (a b c : ℚ) : ℚ := (a + b + c) / 3

-- Theorem statement
theorem nested_average_calculation : 
  avg3 (avg3 2 4 1) (avg2 3 2) 5 = 59 / 18 := by
  sorry

end NUMINAMATH_CALUDE_nested_average_calculation_l2611_261145


namespace NUMINAMATH_CALUDE_elidas_name_length_l2611_261174

theorem elidas_name_length :
  ∀ (E A : ℕ),
  A = 2 * E - 2 →
  10 * ((E + A) / 2 : ℚ) = 65 →
  E = 5 :=
by sorry

end NUMINAMATH_CALUDE_elidas_name_length_l2611_261174


namespace NUMINAMATH_CALUDE_consecutive_odd_sum_48_l2611_261154

theorem consecutive_odd_sum_48 (a b : ℤ) : 
  (∃ k : ℤ, a = 2*k + 1) →  -- a is odd
  (∃ m : ℤ, b = 2*m + 1) →  -- b is odd
  b = a + 2 →               -- b is the next consecutive odd after a
  a + b = 48 →              -- sum is 48
  b = 25 :=                 -- larger number is 25
by sorry

end NUMINAMATH_CALUDE_consecutive_odd_sum_48_l2611_261154


namespace NUMINAMATH_CALUDE_cube_sum_inequality_l2611_261123

theorem cube_sum_inequality (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a^3 / (b*c)) + (b^3 / (c*a)) + (c^3 / (a*b)) ≥ a + b + c := by
  sorry

end NUMINAMATH_CALUDE_cube_sum_inequality_l2611_261123


namespace NUMINAMATH_CALUDE_ellipse_axis_endpoints_distance_l2611_261164

/-- The distance between an endpoint of the major axis and an endpoint of the minor axis of the ellipse 4(x-2)^2 + 16y^2 = 64 is 2√5. -/
theorem ellipse_axis_endpoints_distance : 
  let ellipse := {p : ℝ × ℝ | 4 * (p.1 - 2)^2 + 16 * p.2^2 = 64}
  let major_axis_endpoint := {p : ℝ × ℝ | p ∈ ellipse ∧ p.2 = 0 ∧ p.1 ≠ 2}
  let minor_axis_endpoint := {p : ℝ × ℝ | p ∈ ellipse ∧ p.1 = 2 ∧ p.2 ≠ 0}
  ∀ C ∈ major_axis_endpoint, ∀ D ∈ minor_axis_endpoint, 
    Real.sqrt ((C.1 - D.1)^2 + (C.2 - D.2)^2) = 2 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_axis_endpoints_distance_l2611_261164


namespace NUMINAMATH_CALUDE_abs_ab_value_l2611_261126

/-- Given an ellipse and a hyperbola with specific foci, prove that |ab| = 2√65 -/
theorem abs_ab_value (a b : ℝ) : 
  (∀ x y, x^2/a^2 + y^2/b^2 = 1 → (x = 0 ∧ y = 4) ∨ (x = 0 ∧ y = -4)) →
  (∀ x y, x^2/a^2 - y^2/b^2 = 1 → (x = 6 ∧ y = 0) ∨ (x = -6 ∧ y = 0)) →
  |a * b| = 2 * Real.sqrt 65 := by
  sorry

end NUMINAMATH_CALUDE_abs_ab_value_l2611_261126


namespace NUMINAMATH_CALUDE_parallel_line_equation_l2611_261157

/-- A line passing through point (2, -3) and parallel to y = x has equation x - y = 5 -/
theorem parallel_line_equation : 
  ∀ (x y : ℝ), 
  (∃ (m b : ℝ), y = m * x + b ∧ m = 1) →  -- Line parallel to y = x
  (2, -3) ∈ {(x, y) | y = m * x + b} →    -- Line passes through (2, -3)
  x - y = 5 :=                            -- Equation of the line
by sorry

end NUMINAMATH_CALUDE_parallel_line_equation_l2611_261157
