import Mathlib

namespace NUMINAMATH_CALUDE_total_fish_weight_l2666_266633

/-- Calculates the total weight of fish in James' three tanks -/
theorem total_fish_weight (goldfish_weight guppy_weight angelfish_weight : ℝ)
  (goldfish_count_1 guppy_count_1 : ℕ)
  (goldfish_count_2 guppy_count_2 : ℕ)
  (goldfish_count_3 guppy_count_3 angelfish_count_3 : ℕ)
  (h1 : goldfish_weight = 0.08)
  (h2 : guppy_weight = 0.05)
  (h3 : angelfish_weight = 0.14)
  (h4 : goldfish_count_1 = 15)
  (h5 : guppy_count_1 = 12)
  (h6 : goldfish_count_2 = 2 * goldfish_count_1)
  (h7 : guppy_count_2 = 3 * guppy_count_1)
  (h8 : goldfish_count_3 = 3 * goldfish_count_1)
  (h9 : guppy_count_3 = 2 * guppy_count_1)
  (h10 : angelfish_count_3 = 5) :
  goldfish_weight * (goldfish_count_1 + goldfish_count_2 + goldfish_count_3 : ℝ) +
  guppy_weight * (guppy_count_1 + guppy_count_2 + guppy_count_3 : ℝ) +
  angelfish_weight * angelfish_count_3 = 11.5 := by
  sorry


end NUMINAMATH_CALUDE_total_fish_weight_l2666_266633


namespace NUMINAMATH_CALUDE_sand_overflow_l2666_266613

/-- Represents the capacity of a bucket -/
structure BucketCapacity where
  value : ℚ
  positive : 0 < value

/-- Represents the amount of sand in a bucket -/
structure SandAmount where
  amount : ℚ
  nonnegative : 0 ≤ amount

/-- Theorem stating the overflow amount when pouring sand between buckets -/
theorem sand_overflow
  (CA : BucketCapacity) -- Capacity of Bucket A
  (sand_A : SandAmount) -- Initial sand in Bucket A
  (sand_B : SandAmount) -- Initial sand in Bucket B
  (sand_C : SandAmount) -- Initial sand in Bucket C
  (h1 : sand_A.amount = (1 : ℚ) / 4 * CA.value) -- Bucket A is 1/4 full
  (h2 : sand_B.amount = (3 : ℚ) / 8 * (CA.value / 2)) -- Bucket B is 3/8 full
  (h3 : sand_C.amount = (1 : ℚ) / 3 * (2 * CA.value)) -- Bucket C is 1/3 full
  : ∃ (overflow : ℚ), overflow = (17 : ℚ) / 48 * CA.value :=
by sorry

end NUMINAMATH_CALUDE_sand_overflow_l2666_266613


namespace NUMINAMATH_CALUDE_probability_not_red_is_three_fifths_l2666_266611

/-- Represents the duration of each traffic light color in seconds -/
structure TrafficLightDurations where
  red : ℕ
  yellow : ℕ
  green : ℕ

/-- Calculates the probability of not seeing the red light -/
def probability_not_red (d : TrafficLightDurations) : ℚ :=
  (d.yellow + d.green : ℚ) / (d.red + d.yellow + d.green)

/-- Theorem stating the probability of not seeing the red light is 3/5 -/
theorem probability_not_red_is_three_fifths :
  let d : TrafficLightDurations := ⟨30, 5, 40⟩
  probability_not_red d = 3/5 := by sorry

end NUMINAMATH_CALUDE_probability_not_red_is_three_fifths_l2666_266611


namespace NUMINAMATH_CALUDE_function_properties_imply_b_range_l2666_266684

def is_odd_function (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

def has_period (f : ℝ → ℝ) (p : ℝ) : Prop := ∀ x, f (x + p) = f x

def zero_points_count (f : ℝ → ℝ) (a b : ℝ) (n : ℕ) : Prop :=
  ∃ (zeros : Finset ℝ), zeros.card = n ∧ (∀ x ∈ zeros, a ≤ x ∧ x ≤ b ∧ f x = 0)

theorem function_properties_imply_b_range (f : ℝ → ℝ) (b : ℝ) :
  is_odd_function f →
  has_period f 4 →
  (∀ x ∈ Set.Ioo 0 2, f x = Real.log (x^2 - x + b)) →
  zero_points_count f (-2) 2 5 →
  (1/4 < b ∧ b ≤ 1) ∨ b = 5/4 :=
by sorry

end NUMINAMATH_CALUDE_function_properties_imply_b_range_l2666_266684


namespace NUMINAMATH_CALUDE_sum_of_a_and_b_is_71_l2666_266662

/-- Represents the product of a sequence following the pattern (n+1)/n from 5/3 to a/b --/
def sequence_product (a b : ℕ) : ℚ :=
  a / 3

theorem sum_of_a_and_b_is_71 (a b : ℕ) (h : sequence_product a b = 12) : a + b = 71 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_a_and_b_is_71_l2666_266662


namespace NUMINAMATH_CALUDE_equation_transformation_l2666_266656

variable (x y : ℝ)

theorem equation_transformation (h : y = x + 1/x) :
  x^4 - x^3 - 6*x^2 - x + 1 = 0 ↔ x^2 * (y^2 - y - 6) = 0 :=
by sorry

end NUMINAMATH_CALUDE_equation_transformation_l2666_266656


namespace NUMINAMATH_CALUDE_eleven_step_paths_through_F_l2666_266647

/-- A point on the 6x6 grid -/
structure Point where
  x : Nat
  y : Nat
  h_x : x ≤ 5
  h_y : y ≤ 5

/-- The number of paths between two points on the grid -/
def num_paths (start finish : Point) : Nat :=
  Nat.choose (finish.x - start.x + finish.y - start.y) (finish.x - start.x)

theorem eleven_step_paths_through_F : 
  let E : Point := ⟨0, 5, by norm_num, by norm_num⟩
  let F : Point := ⟨3, 3, by norm_num, by norm_num⟩
  let G : Point := ⟨5, 0, by norm_num, by norm_num⟩
  (num_paths E F) * (num_paths F G) = 100 := by
  sorry

end NUMINAMATH_CALUDE_eleven_step_paths_through_F_l2666_266647


namespace NUMINAMATH_CALUDE_colored_polygons_equality_l2666_266657

/-- A regular polygon with n vertices -/
structure RegularPolygon (n : ℕ) where
  vertices : Fin n → ℝ × ℝ
  is_regular : sorry

/-- A coloring of the vertices of a regular polygon -/
def Coloring (n : ℕ) := Fin n → ℕ

/-- The set of vertices of a given color -/
def colorVertices (n : ℕ) (p : RegularPolygon n) (c : Coloring n) (color : ℕ) : Set (ℝ × ℝ) :=
  {v | ∃ i, c i = color ∧ p.vertices i = v}

/-- Predicate to check if a set of vertices forms a regular polygon -/
def isRegularPolygon (vertices : Set (ℝ × ℝ)) : Prop := sorry

theorem colored_polygons_equality (n : ℕ) (p : RegularPolygon n) (c : Coloring n) :
  (∀ color, isRegularPolygon (colorVertices n p c color)) →
  ∃ color1 color2, color1 ≠ color2 ∧ 
    colorVertices n p c color1 = colorVertices n p c color2 := by
  sorry

end NUMINAMATH_CALUDE_colored_polygons_equality_l2666_266657


namespace NUMINAMATH_CALUDE_right_triangle_acute_angles_l2666_266630

theorem right_triangle_acute_angles (a b : ℝ) : 
  a > 0 ∧ b > 0 ∧  -- Angles are positive
  a + b = 90 ∧     -- Sum of acute angles in a right triangle is 90°
  a / b = 3 / 2 →  -- Ratio of angles is 3:2
  a = 54 ∧ b = 36 := by sorry

end NUMINAMATH_CALUDE_right_triangle_acute_angles_l2666_266630


namespace NUMINAMATH_CALUDE_sum_of_four_numbers_l2666_266605

theorem sum_of_four_numbers : 1357 + 7531 + 3175 + 5713 = 17776 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_four_numbers_l2666_266605


namespace NUMINAMATH_CALUDE_sequence_sum_problem_l2666_266666

/-- Sum of an arithmetic sequence -/
def arithmeticSum (a₁ : ℕ) (aₙ : ℕ) : ℕ := 
  let n := aₙ - a₁ + 1
  n * (a₁ + aₙ) / 2

theorem sequence_sum_problem : 
  (arithmeticSum 2001 2093) - (arithmeticSum 221 313) + (arithmeticSum 401 493) = 207141 := by
  sorry

end NUMINAMATH_CALUDE_sequence_sum_problem_l2666_266666


namespace NUMINAMATH_CALUDE_mean_squared_sum_l2666_266673

theorem mean_squared_sum (a b c : ℝ) 
  (h_arithmetic : (a + b + c) / 3 = 7)
  (h_geometric : (a * b * c) ^ (1/3 : ℝ) = 6)
  (h_harmonic : 3 / (1/a + 1/b + 1/c) = 5) :
  a^2 + b^2 + c^2 = 181.8 := by
  sorry

end NUMINAMATH_CALUDE_mean_squared_sum_l2666_266673


namespace NUMINAMATH_CALUDE_sum_of_net_gains_l2666_266622

def initial_revenue : ℝ := 4.7
def revenue_increase_A : ℝ := 0.1326
def revenue_increase_B : ℝ := 0.0943
def revenue_increase_C : ℝ := 0.7731
def tax_rate : ℝ := 0.235

def net_gain (initial_rev : ℝ) (rev_increase : ℝ) (tax : ℝ) : ℝ :=
  (initial_rev * (1 + rev_increase)) * (1 - tax)

theorem sum_of_net_gains :
  let net_gain_A := net_gain initial_revenue revenue_increase_A tax_rate
  let net_gain_B := net_gain initial_revenue revenue_increase_B tax_rate
  let net_gain_C := net_gain initial_revenue revenue_increase_C tax_rate
  net_gain_A + net_gain_B + net_gain_C = 14.38214 := by sorry

end NUMINAMATH_CALUDE_sum_of_net_gains_l2666_266622


namespace NUMINAMATH_CALUDE_range_of_a_l2666_266699

def prop_p (a : ℝ) : Prop := ∀ x : ℝ, x^2 + a*x + 1 > 0

def prop_q (a : ℝ) : Prop := ∃ x : ℝ, x^2 - x + a = 0

theorem range_of_a (a : ℝ) :
  (prop_p a ∨ prop_q a) ∧ ¬(prop_p a ∧ prop_q a) →
  a ≤ -2 ∨ (1/4 < a ∧ a < 2) :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l2666_266699


namespace NUMINAMATH_CALUDE_white_squares_47th_row_l2666_266612

/-- Represents the number of squares in a row of the stair-step figure -/
def totalSquares (n : ℕ) : ℕ := 2 * n - 1

/-- Represents the number of white squares in a row of the stair-step figure -/
def whiteSquares (n : ℕ) : ℕ := (totalSquares n - 1) / 2

/-- Theorem stating that the 47th row of the stair-step figure contains 46 white squares -/
theorem white_squares_47th_row :
  whiteSquares 47 = 46 := by
  sorry


end NUMINAMATH_CALUDE_white_squares_47th_row_l2666_266612


namespace NUMINAMATH_CALUDE_equation_solution_l2666_266608

theorem equation_solution : ∃! x : ℚ, (x + 4) / (x - 3) = (x - 2) / (x + 2) ∧ x = -2/11 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2666_266608


namespace NUMINAMATH_CALUDE_find_k_l2666_266676

-- Define the polynomials A and B
def A (x k : ℝ) : ℝ := 2 * x^2 + k * x - 6 * x

def B (x k : ℝ) : ℝ := -x^2 + k * x - 1

-- Define the condition for A + 2B to be independent of x
def independent_of_x (k : ℝ) : Prop :=
  ∀ x : ℝ, ∃ c : ℝ, A x k + 2 * B x k = c

-- Theorem statement
theorem find_k : ∃ k : ℝ, independent_of_x k ∧ k = 2 :=
sorry

end NUMINAMATH_CALUDE_find_k_l2666_266676


namespace NUMINAMATH_CALUDE_scientific_notation_of_0_00000065_l2666_266694

/-- Represents a number in scientific notation -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  is_valid : 1 ≤ |coefficient| ∧ |coefficient| < 10

/-- Converts a real number to scientific notation -/
def toScientificNotation (x : ℝ) : ScientificNotation :=
  sorry

theorem scientific_notation_of_0_00000065 :
  toScientificNotation 0.00000065 = ScientificNotation.mk 6.5 (-7) sorry := by
  sorry

end NUMINAMATH_CALUDE_scientific_notation_of_0_00000065_l2666_266694


namespace NUMINAMATH_CALUDE_function_derivative_problem_l2666_266659

/-- Given a function f(x) = x(x+k)(x+2k)(x-3k) where f'(0) = 6, prove that k = -1 -/
theorem function_derivative_problem (k : ℝ) : 
  (∃ f : ℝ → ℝ, (∀ x, f x = x * (x + k) * (x + 2*k) * (x - 3*k)) ∧ 
   (deriv f) 0 = 6) → 
  k = -1 := by
  sorry

end NUMINAMATH_CALUDE_function_derivative_problem_l2666_266659


namespace NUMINAMATH_CALUDE_percent_y_of_x_l2666_266669

theorem percent_y_of_x (x y : ℝ) (h : 0.5 * (x - y) = 0.3 * (x + y)) : y = 0.25 * x := by
  sorry

end NUMINAMATH_CALUDE_percent_y_of_x_l2666_266669


namespace NUMINAMATH_CALUDE_rectangle_diagonal_l2666_266637

/-- Given a rectangle with perimeter 60 meters and length-to-width ratio of 5:2,
    prove that its diagonal length is 162/7 meters. -/
theorem rectangle_diagonal (length width : ℝ) : 
  (2 * (length + width) = 60) →  -- Perimeter condition
  (length = (5/2) * width) →     -- Ratio condition
  Real.sqrt (length^2 + width^2) = 162/7 := by
sorry

end NUMINAMATH_CALUDE_rectangle_diagonal_l2666_266637


namespace NUMINAMATH_CALUDE_last_digit_89_base5_l2666_266629

def decimal_to_base5 (n : ℕ) : List ℕ :=
  if n = 0 then [0]
  else
    let rec aux (m : ℕ) (acc : List ℕ) : List ℕ :=
      if m = 0 then acc
      else aux (m / 5) ((m % 5) :: acc)
    aux n []

theorem last_digit_89_base5 : 
  (decimal_to_base5 89).getLast? = some 4 := by
  sorry

end NUMINAMATH_CALUDE_last_digit_89_base5_l2666_266629


namespace NUMINAMATH_CALUDE_josh_initial_money_l2666_266664

/-- Josh's initial amount of money given his expenses and remaining balance -/
def initial_amount (spent1 spent2 remaining : ℚ) : ℚ :=
  spent1 + spent2 + remaining

theorem josh_initial_money :
  initial_amount 1.75 1.25 6 = 9 := by
  sorry

end NUMINAMATH_CALUDE_josh_initial_money_l2666_266664


namespace NUMINAMATH_CALUDE_f_properties_l2666_266609

-- Define the function f
def f (x : ℝ) : ℝ := (x + 1)^2 - 2*(x + 1) + 7

-- Theorem statement
theorem f_properties :
  (f 2 = 10) ∧
  (∀ a, f a = a^2 + 6) ∧
  (∀ x, f x = x^2 + 6) ∧
  (∀ x, f (x + 1) = x^2 + 2*x + 7) ∧
  (∀ y, y ∈ Set.range (λ x => f (x + 1)) ↔ y ≥ 6) :=
by sorry

end NUMINAMATH_CALUDE_f_properties_l2666_266609


namespace NUMINAMATH_CALUDE_shortest_distance_parabola_line_l2666_266665

/-- The shortest distance between a point on the parabola y = x^2 - 4x + 7
    and a point on the line y = 2x - 5 is 3√5/5 -/
theorem shortest_distance_parabola_line : 
  let parabola := fun x : ℝ => x^2 - 4*x + 7
  let line := fun x : ℝ => 2*x - 5
  ∃ (min_dist : ℝ), 
    (∀ (p q : ℝ × ℝ), 
      (p.2 = parabola p.1) → 
      (q.2 = line q.1) → 
      dist p q ≥ min_dist) ∧
    (∃ (p q : ℝ × ℝ), 
      (p.2 = parabola p.1) ∧ 
      (q.2 = line q.1) ∧ 
      dist p q = min_dist) ∧
    min_dist = 3 * Real.sqrt 5 / 5 := by
  sorry

end NUMINAMATH_CALUDE_shortest_distance_parabola_line_l2666_266665


namespace NUMINAMATH_CALUDE_f_geq_kx_implies_k_range_l2666_266604

-- Define the piecewise function f
noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 0 then 2 * x^2 - 3 * x else Real.exp x + Real.exp 2

-- State the theorem
theorem f_geq_kx_implies_k_range :
  (∀ x : ℝ, f x ≥ k * x) → -3 ≤ k ∧ k ≤ Real.exp 2 :=
by sorry

end NUMINAMATH_CALUDE_f_geq_kx_implies_k_range_l2666_266604


namespace NUMINAMATH_CALUDE_bernoullis_inequality_l2666_266652

theorem bernoullis_inequality (n : ℕ) (a : ℝ) (h : a > -1) :
  (1 + a)^n ≥ n * a + 1 := by
  sorry

end NUMINAMATH_CALUDE_bernoullis_inequality_l2666_266652


namespace NUMINAMATH_CALUDE_josh_paid_six_dollars_l2666_266651

/-- The amount Josh paid for string cheese -/
def string_cheese_cost (packs : ℕ) (cheeses_per_pack : ℕ) (cents_per_cheese : ℕ) : ℚ :=
  (packs * cheeses_per_pack * cents_per_cheese : ℚ) / 100

/-- Theorem stating that Josh paid 6 dollars for the string cheese -/
theorem josh_paid_six_dollars :
  string_cheese_cost 3 20 10 = 6 := by
  sorry

end NUMINAMATH_CALUDE_josh_paid_six_dollars_l2666_266651


namespace NUMINAMATH_CALUDE_complementary_angle_of_35_30_l2666_266625

-- Define the angle in degrees and minutes
def angle_alpha : ℚ := 35 + 30 / 60

-- Define the complementary angle function
def complementary_angle (α : ℚ) : ℚ := 90 - α

-- Theorem statement
theorem complementary_angle_of_35_30 :
  let result := complementary_angle angle_alpha
  ⌊result⌋ = 54 ∧ (result - ⌊result⌋) * 60 = 30 := by
  sorry

#eval complementary_angle angle_alpha

end NUMINAMATH_CALUDE_complementary_angle_of_35_30_l2666_266625


namespace NUMINAMATH_CALUDE_right_triangle_area_l2666_266658

/-- The area of a right triangle with hypotenuse 10 and sum of other sides 14 is 24 -/
theorem right_triangle_area (a b c : ℝ) (h1 : a + b = 14) (h2 : c = 10) (h3 : a^2 + b^2 = c^2) :
  (1/2) * a * b = 24 := by
sorry

end NUMINAMATH_CALUDE_right_triangle_area_l2666_266658


namespace NUMINAMATH_CALUDE_ellipse_coincide_hyperbola_focus_l2666_266686

def ellipse_equation (a b x y : ℝ) : Prop := x^2 / a^2 + y^2 / b^2 = 1

def hyperbola_equation (x y : ℝ) : Prop := x^2 - y^2 / 3 = 1

def eccentricity (e : ℝ) : Prop := e = 1 / 2

theorem ellipse_coincide_hyperbola_focus (a b : ℝ) :
  eccentricity (1 / 2) →
  (∃ x y, ellipse_equation a b x y) →
  (∃ x y, hyperbola_equation x y) →
  (∀ x y, ellipse_equation a b x y ↔ ellipse_equation 4 (12 : ℝ).sqrt x y) :=
sorry

end NUMINAMATH_CALUDE_ellipse_coincide_hyperbola_focus_l2666_266686


namespace NUMINAMATH_CALUDE_f_4_equals_1559_l2666_266644

-- Define the polynomial f(x) = x^5 + 3x^4 - 5x^3 + 7x^2 - 9x + 11
def f (x : ℝ) : ℝ := x^5 + 3*x^4 - 5*x^3 + 7*x^2 - 9*x + 11

-- Define Horner's method for this specific polynomial
def horner (x : ℝ) : ℝ := ((((x + 3) * x - 5) * x + 7) * x - 9) * x + 11

-- Theorem stating that f(4) = 1559 using Horner's method
theorem f_4_equals_1559 : f 4 = 1559 ∧ horner 4 = 1559 := by
  sorry

end NUMINAMATH_CALUDE_f_4_equals_1559_l2666_266644


namespace NUMINAMATH_CALUDE_triangle_angle_problem_l2666_266691

theorem triangle_angle_problem (a b c : ℝ) : 
  a > 0 ∧ b > 0 ∧ c > 0 → -- angles are positive
  b = 2 * a → -- second angle is double the first
  c = a - 40 → -- third angle is 40 less than the first
  a + b + c = 180 → -- sum of angles in a triangle
  a = 55 := by sorry

end NUMINAMATH_CALUDE_triangle_angle_problem_l2666_266691


namespace NUMINAMATH_CALUDE_custom_mul_four_three_l2666_266626

/-- Custom multiplication operation -/
def custom_mul (a b : ℝ) : ℝ := a^2 - a*b + b^2

/-- Theorem stating that 4*3 = 13 under the custom multiplication -/
theorem custom_mul_four_three : custom_mul 4 3 = 13 := by
  sorry

end NUMINAMATH_CALUDE_custom_mul_four_three_l2666_266626


namespace NUMINAMATH_CALUDE_cube_sum_divisible_by_nine_l2666_266606

theorem cube_sum_divisible_by_nine (n : ℕ+) : 
  9 ∣ (n.val^3 + (n.val + 1)^3 + (n.val + 2)^3) := by
sorry

end NUMINAMATH_CALUDE_cube_sum_divisible_by_nine_l2666_266606


namespace NUMINAMATH_CALUDE_total_participants_l2666_266677

/-- Represents the exam scores and statistics -/
structure ExamStatistics where
  low_scorers : ℕ  -- Number of people scoring no more than 30
  low_avg : ℝ      -- Average score of low scorers
  high_scorers : ℕ -- Number of people scoring no less than 80
  high_avg : ℝ     -- Average score of high scorers
  above_30_avg : ℝ -- Average score of those scoring more than 30
  below_80_avg : ℝ -- Average score of those scoring less than 80

/-- Theorem stating the total number of participants in the exam -/
theorem total_participants (stats : ExamStatistics) 
  (h1 : stats.low_scorers = 153)
  (h2 : stats.low_avg = 24)
  (h3 : stats.high_scorers = 59)
  (h4 : stats.high_avg = 92)
  (h5 : stats.above_30_avg = 62)
  (h6 : stats.below_80_avg = 54) :
  stats.low_scorers + stats.high_scorers + 
  ((stats.low_scorers * (stats.below_80_avg - stats.low_avg) + 
    stats.high_scorers * (stats.high_avg - stats.above_30_avg)) / 
   (stats.above_30_avg - stats.below_80_avg)) = 1007 := by
  sorry


end NUMINAMATH_CALUDE_total_participants_l2666_266677


namespace NUMINAMATH_CALUDE_smallest_sum_of_four_consecutive_primes_divisible_by_five_l2666_266671

/-- A function that returns true if a number is prime -/
def isPrime (n : ℕ) : Prop := sorry

/-- A function that checks if four consecutive numbers are all prime -/
def fourConsecutivePrimes (a b c d : ℕ) : Prop :=
  isPrime a ∧ isPrime b ∧ isPrime c ∧ isPrime d ∧
  b = a + 1 ∧ c = b + 1 ∧ d = c + 1

/-- The main theorem -/
theorem smallest_sum_of_four_consecutive_primes_divisible_by_five :
  ∃ (a b c d : ℕ),
    fourConsecutivePrimes a b c d ∧
    (a + b + c + d) % 5 = 0 ∧
    a + b + c + d = 60 ∧
    ∀ (w x y z : ℕ),
      fourConsecutivePrimes w x y z →
      (w + x + y + z) % 5 = 0 →
      w + x + y + z ≥ 60 :=
sorry

end NUMINAMATH_CALUDE_smallest_sum_of_four_consecutive_primes_divisible_by_five_l2666_266671


namespace NUMINAMATH_CALUDE_circle_diameter_from_area_l2666_266688

/-- Given a circle with area 16π, prove its diameter is 8 -/
theorem circle_diameter_from_area : 
  ∀ (r : ℝ), π * r^2 = 16 * π → 2 * r = 8 := by
  sorry

end NUMINAMATH_CALUDE_circle_diameter_from_area_l2666_266688


namespace NUMINAMATH_CALUDE_remainder_problem_l2666_266692

theorem remainder_problem (x y z w : ℕ) 
  (hx : 4 ∣ x) (hy : 4 ∣ y) (hz : 4 ∣ z) (hw : 3 ∣ w) (hpos_x : x > 0) (hpos_y : y > 0) (hpos_z : z > 0) :
  (x^2 * (y*w + z*(x + y)^2) + 7) % 6 = 1 :=
sorry

end NUMINAMATH_CALUDE_remainder_problem_l2666_266692


namespace NUMINAMATH_CALUDE_p_is_converse_of_r_l2666_266619

-- Define propositions as functions from some type α to Prop
variable {α : Type}
variable (p q r : α → Prop)

-- Define the relationships between p, q, and r
axiom contrapositive : (∀ x, p x → q x) ↔ (∀ x, ¬q x → ¬p x)
axiom negation : (∀ x, q x) ↔ (∀ x, ¬r x)

-- Theorem to prove
theorem p_is_converse_of_r : (∀ x, p x → r x) ↔ (∀ x, r x → p x) := by sorry

end NUMINAMATH_CALUDE_p_is_converse_of_r_l2666_266619


namespace NUMINAMATH_CALUDE_cakes_served_today_l2666_266648

theorem cakes_served_today (lunch_cakes dinner_cakes : ℕ) 
  (h1 : lunch_cakes = 6) 
  (h2 : dinner_cakes = 9) : 
  lunch_cakes + dinner_cakes = 15 := by
  sorry

end NUMINAMATH_CALUDE_cakes_served_today_l2666_266648


namespace NUMINAMATH_CALUDE_sequence_problem_l2666_266639

/-- Given a sequence {a_n} and an arithmetic sequence {b_n}, prove that a_6 = 33 -/
theorem sequence_problem (a b : ℕ → ℕ) : 
  a 1 = 3 →  -- First term of {a_n} is 3
  b 1 = 2 →  -- b_1 = 2
  b 3 = 6 →  -- b_3 = 6
  (∀ n : ℕ, n > 0 → b n = a (n + 1) - a n) →  -- b_n = a_{n+1} - a_n for n ∈ ℕ*
  (∀ n : ℕ, n > 0 → ∃ d : ℕ, b (n + 1) = b n + d) →  -- {b_n} is an arithmetic sequence
  a 6 = 33 := by
sorry

end NUMINAMATH_CALUDE_sequence_problem_l2666_266639


namespace NUMINAMATH_CALUDE_ellipse_foci_coordinates_l2666_266661

/-- The coordinates of the foci of an ellipse with equation x^2/10 + y^2 = 1 are (3,0) and (-3,0) -/
theorem ellipse_foci_coordinates :
  let ellipse := {(x, y) : ℝ × ℝ | x^2/10 + y^2 = 1}
  ∃ (f₁ f₂ : ℝ × ℝ), f₁ ∈ ellipse ∧ f₂ ∈ ellipse ∧ f₁ = (3, 0) ∧ f₂ = (-3, 0) ∧
    ∀ (f : ℝ × ℝ), f ∈ ellipse → f = f₁ ∨ f = f₂ :=
by sorry

end NUMINAMATH_CALUDE_ellipse_foci_coordinates_l2666_266661


namespace NUMINAMATH_CALUDE_parity_of_expression_l2666_266627

theorem parity_of_expression (p m : ℤ) (h_p_odd : Odd p) :
  Odd (p^2 + 3*m*p) ↔ Even m := by
sorry

end NUMINAMATH_CALUDE_parity_of_expression_l2666_266627


namespace NUMINAMATH_CALUDE_product_121_54_l2666_266654

theorem product_121_54 : 121 * 54 = 6534 := by
  sorry

end NUMINAMATH_CALUDE_product_121_54_l2666_266654


namespace NUMINAMATH_CALUDE_square_difference_l2666_266696

theorem square_difference : (39 : ℕ)^2 = (40 : ℕ)^2 - 79 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_l2666_266696


namespace NUMINAMATH_CALUDE_more_boys_than_girls_l2666_266614

/-- Given a school with 34 girls and 841 boys, prove that there are 807 more boys than girls. -/
theorem more_boys_than_girls (girls : ℕ) (boys : ℕ) 
  (h1 : girls = 34) (h2 : boys = 841) : boys - girls = 807 := by
  sorry

end NUMINAMATH_CALUDE_more_boys_than_girls_l2666_266614


namespace NUMINAMATH_CALUDE_lopez_family_seating_arrangements_l2666_266663

/-- Represents a family member -/
inductive FamilyMember
| MrLopez
| MrsLopez
| Child1
| Child2
| Child3

/-- Represents a seat in the car -/
inductive Seat
| Driver
| FrontPassenger
| BackLeft
| BackMiddle
| BackRight

/-- A seating arrangement is a function from Seat to FamilyMember -/
def SeatingArrangement := Seat → FamilyMember

/-- Checks if a seating arrangement is valid -/
def isValidArrangement (arr : SeatingArrangement) : Prop :=
  (arr Seat.Driver = FamilyMember.MrLopez ∨ arr Seat.Driver = FamilyMember.MrsLopez) ∧
  (∀ s₁ s₂, s₁ ≠ s₂ → arr s₁ ≠ arr s₂)

/-- The number of valid seating arrangements -/
def numValidArrangements : ℕ := sorry

theorem lopez_family_seating_arrangements :
  numValidArrangements = 48 := by sorry

end NUMINAMATH_CALUDE_lopez_family_seating_arrangements_l2666_266663


namespace NUMINAMATH_CALUDE_total_triangles_is_16_l2666_266674

/-- Represents a square with diagonals and an inner square formed by midpoints -/
structure SquareWithDiagonalsAndInnerSquare :=
  (s : ℝ) -- Side length of the larger square
  (has_diagonals : Bool) -- The larger square has diagonals
  (has_inner_square : Bool) -- There's an inner square formed by midpoints

/-- Counts the total number of triangles in the figure -/
def count_triangles (square : SquareWithDiagonalsAndInnerSquare) : ℕ :=
  sorry -- The actual counting logic would go here

/-- Theorem stating that the total number of triangles is 16 -/
theorem total_triangles_is_16 (square : SquareWithDiagonalsAndInnerSquare) :
  square.has_diagonals = true → square.has_inner_square = true → count_triangles square = 16 :=
by sorry

end NUMINAMATH_CALUDE_total_triangles_is_16_l2666_266674


namespace NUMINAMATH_CALUDE_lars_baking_capacity_l2666_266668

/-- Represents the baking capacity of Lars' bakeshop -/
structure Bakeshop where
  baguettes_per_two_hours : ℕ
  baking_hours_per_day : ℕ
  total_breads_per_day : ℕ

/-- Calculates the number of loaves of bread that can be baked per hour -/
def loaves_per_hour (shop : Bakeshop) : ℚ :=
  let baguettes_per_day := shop.baguettes_per_two_hours * (shop.baking_hours_per_day / 2)
  let loaves_per_day := shop.total_breads_per_day - baguettes_per_day
  loaves_per_day / shop.baking_hours_per_day

/-- Theorem stating that Lars can bake 10 loaves of bread per hour -/
theorem lars_baking_capacity :
  let lars_shop : Bakeshop := {
    baguettes_per_two_hours := 30,
    baking_hours_per_day := 6,
    total_breads_per_day := 150
  }
  loaves_per_hour lars_shop = 10 := by
  sorry

end NUMINAMATH_CALUDE_lars_baking_capacity_l2666_266668


namespace NUMINAMATH_CALUDE_product_x_y_is_32_l2666_266636

/-- A parallelogram EFGH with given side lengths -/
structure Parallelogram where
  EF : ℝ
  FG : ℝ
  GH : ℝ
  HE : ℝ
  is_parallelogram : EF = GH ∧ FG = HE

/-- The product of x and y in the given parallelogram is 32 -/
theorem product_x_y_is_32 (p : Parallelogram)
  (h1 : p.EF = 42)
  (h2 : ∃ y, p.FG = 4 * y^3)
  (h3 : ∃ x, p.GH = 2 * x + 10)
  (h4 : p.HE = 32) :
  ∃ x y, x * y = 32 ∧ p.FG = 4 * y^3 ∧ p.GH = 2 * x + 10 := by
  sorry

end NUMINAMATH_CALUDE_product_x_y_is_32_l2666_266636


namespace NUMINAMATH_CALUDE_four_digit_number_with_specific_remainders_l2666_266645

theorem four_digit_number_with_specific_remainders :
  ∃! n : ℕ, 1000 ≤ n ∧ n < 10000 ∧ n % 131 = 112 ∧ n % 132 = 98 :=
by
  sorry

end NUMINAMATH_CALUDE_four_digit_number_with_specific_remainders_l2666_266645


namespace NUMINAMATH_CALUDE_negation_exactly_one_even_l2666_266617

/-- Represents the property of a natural number being even -/
def IsEven (n : ℕ) : Prop := ∃ k : ℕ, n = 2 * k

/-- Represents the property that exactly one of three natural numbers is even -/
def ExactlyOneEven (a b c : ℕ) : Prop :=
  (IsEven a ∧ ¬IsEven b ∧ ¬IsEven c) ∨
  (¬IsEven a ∧ IsEven b ∧ ¬IsEven c) ∨
  (¬IsEven a ∧ ¬IsEven b ∧ IsEven c)

/-- The main theorem stating that the negation of "exactly one even" is equivalent to "all odd or at least two even" -/
theorem negation_exactly_one_even (a b c : ℕ) :
  ¬(ExactlyOneEven a b c) ↔ (¬IsEven a ∧ ¬IsEven b ∧ ¬IsEven c) ∨ (IsEven a ∧ IsEven b) ∨ (IsEven a ∧ IsEven c) ∨ (IsEven b ∧ IsEven c) :=
sorry


end NUMINAMATH_CALUDE_negation_exactly_one_even_l2666_266617


namespace NUMINAMATH_CALUDE_dorothy_initial_money_l2666_266634

-- Define the family members
inductive FamilyMember
| Dorothy
| Brother
| Parent1
| Parent2
| Grandfather

-- Define the age of a family member
def age (member : FamilyMember) : ℕ :=
  match member with
  | .Dorothy => 15
  | .Brother => 0  -- We don't know exact age, but younger than 18
  | .Parent1 => 18 -- We don't know exact age, but at least 18
  | .Parent2 => 18 -- We don't know exact age, but at least 18
  | .Grandfather => 18 -- We don't know exact age, but at least 18

-- Define the regular ticket price
def regularTicketPrice : ℕ := 10

-- Define the discount rate for young people
def youngDiscount : ℚ := 0.3

-- Define the discounted ticket price function
def ticketPrice (member : FamilyMember) : ℚ :=
  if age member ≤ 18 then
    regularTicketPrice * (1 - youngDiscount)
  else
    regularTicketPrice

-- Define the total cost of tickets for the family
def totalTicketCost : ℚ :=
  ticketPrice FamilyMember.Dorothy +
  ticketPrice FamilyMember.Brother +
  ticketPrice FamilyMember.Parent1 +
  ticketPrice FamilyMember.Parent2 +
  ticketPrice FamilyMember.Grandfather

-- Define Dorothy's remaining money after the trip
def moneyLeftAfterTrip : ℚ := 26

-- Theorem: Dorothy's initial money was $70
theorem dorothy_initial_money :
  totalTicketCost + moneyLeftAfterTrip = 70 := by
  sorry

end NUMINAMATH_CALUDE_dorothy_initial_money_l2666_266634


namespace NUMINAMATH_CALUDE_range_of_f_greater_than_x_l2666_266632

-- Define the piecewise function f
noncomputable def f (x : ℝ) : ℝ :=
  if x ≥ 0 then (1/2) * x - 1 else 1/x

-- State the theorem
theorem range_of_f_greater_than_x :
  ∀ a : ℝ, f a > a ↔ a ∈ Set.Iio (-1) :=
by sorry

end NUMINAMATH_CALUDE_range_of_f_greater_than_x_l2666_266632


namespace NUMINAMATH_CALUDE_necessary_condition_when_m_is_one_necessary_condition_range_l2666_266615

/-- Proposition P -/
def P : Set ℝ := {x | -2 ≤ x ∧ x ≤ 10}

/-- Proposition q -/
def q (m : ℝ) : Set ℝ := {x | 1 - m ≤ x ∧ x ≤ 1 + m}

/-- P is a necessary but not sufficient condition for q -/
def necessary_not_sufficient (m : ℝ) : Prop :=
  (q m ⊆ P) ∧ (q m ≠ P) ∧ (m > 0)

theorem necessary_condition_when_m_is_one :
  necessary_not_sufficient 1 := by sorry

theorem necessary_condition_range :
  ∀ m : ℝ, necessary_not_sufficient m ↔ m ≥ 9 := by sorry

end NUMINAMATH_CALUDE_necessary_condition_when_m_is_one_necessary_condition_range_l2666_266615


namespace NUMINAMATH_CALUDE_fourth_root_simplification_l2666_266682

theorem fourth_root_simplification :
  ∃ (c d : ℕ+), (c : ℝ) * ((d : ℝ)^(1/4 : ℝ)) = (3^5 * 5^3 : ℝ)^(1/4 : ℝ) ∧ c + d = 378 := by
  sorry

end NUMINAMATH_CALUDE_fourth_root_simplification_l2666_266682


namespace NUMINAMATH_CALUDE_inequality_system_solution_l2666_266602

theorem inequality_system_solution (x : ℝ) :
  (5 * x - 2 < 3 * (x + 2)) ∧
  ((2 * x - 1) / 3 - (5 * x + 1) / 2 ≤ 1) →
  -1 ≤ x ∧ x < 4 := by
sorry

end NUMINAMATH_CALUDE_inequality_system_solution_l2666_266602


namespace NUMINAMATH_CALUDE_jakes_weight_l2666_266600

theorem jakes_weight (jake_weight sister_weight : ℝ) 
  (h1 : jake_weight - 8 = 2 * sister_weight)
  (h2 : jake_weight + sister_weight = 278) : 
  jake_weight = 188 := by
sorry

end NUMINAMATH_CALUDE_jakes_weight_l2666_266600


namespace NUMINAMATH_CALUDE_opposite_edge_angles_not_all_acute_or_obtuse_l2666_266670

/-- Represents a convex polyhedral angle -/
structure ConvexPolyhedralAngle where
  /-- All dihedral angles are 60° -/
  dihedral_angles_60 : Bool

/-- Represents the angles between opposite edges of a polyhedral angle -/
inductive OppositeEdgeAngles
  | Acute : OppositeEdgeAngles
  | Obtuse : OppositeEdgeAngles
  | Mixed : OppositeEdgeAngles

/-- 
Given a convex polyhedral angle with all dihedral angles equal to 60°, 
it's impossible for the angles between opposite edges to be simultaneously acute or simultaneously obtuse.
-/
theorem opposite_edge_angles_not_all_acute_or_obtuse (angle : ConvexPolyhedralAngle) 
  (h : angle.dihedral_angles_60 = true) : 
  ∃ (opp_angles : OppositeEdgeAngles), opp_angles = OppositeEdgeAngles.Mixed :=
sorry

end NUMINAMATH_CALUDE_opposite_edge_angles_not_all_acute_or_obtuse_l2666_266670


namespace NUMINAMATH_CALUDE_walking_time_proportional_l2666_266607

/-- Given a constant walking rate, prove that if it takes 6 minutes to walk 2 miles, 
    then it will take 12 minutes to walk 4 miles. -/
theorem walking_time_proportional (rate : ℝ) : 
  (rate * 2 = 6) → (rate * 4 = 12) := by
  sorry

end NUMINAMATH_CALUDE_walking_time_proportional_l2666_266607


namespace NUMINAMATH_CALUDE_carla_marbles_l2666_266687

/-- The number of marbles Carla has now, given her initial marbles and the number she bought -/
def total_marbles (initial : ℕ) (bought : ℕ) : ℕ := initial + bought

/-- Theorem stating that Carla now has 187 marbles -/
theorem carla_marbles : total_marbles 53 134 = 187 := by
  sorry

end NUMINAMATH_CALUDE_carla_marbles_l2666_266687


namespace NUMINAMATH_CALUDE_milena_age_l2666_266640

theorem milena_age :
  ∀ (milena_age grandmother_age grandfather_age : ℕ),
    grandmother_age = 9 * milena_age →
    grandfather_age = grandmother_age + 2 →
    grandfather_age - milena_age = 58 →
    milena_age = 7 := by
  sorry

end NUMINAMATH_CALUDE_milena_age_l2666_266640


namespace NUMINAMATH_CALUDE_milk_replacement_problem_l2666_266624

/-- Given a container initially full of milk, prove that if x liters are drawn out and 
    replaced with water twice, resulting in a milk to water ratio of 9:16 in a 
    total mixture of 15 liters, then x must equal 12 liters. -/
theorem milk_replacement_problem (x : ℝ) : 
  x > 0 →
  (15 - x) - x * ((15 - x) / 15) = (9 / 25) * 15 →
  x = 12 := by
  sorry

end NUMINAMATH_CALUDE_milk_replacement_problem_l2666_266624


namespace NUMINAMATH_CALUDE_inscribed_square_area_l2666_266653

/-- The area of a square inscribed in a circle, which is itself inscribed in an equilateral triangle -/
theorem inscribed_square_area (s : ℝ) (h : s = 6) : 
  let r := s / (2 * Real.sqrt 3)
  let d := 2 * r
  let side := d / Real.sqrt 2
  side ^ 2 = 6 := by sorry

end NUMINAMATH_CALUDE_inscribed_square_area_l2666_266653


namespace NUMINAMATH_CALUDE_shinyoung_candy_problem_l2666_266621

theorem shinyoung_candy_problem (initial_candies : ℕ) : 
  (initial_candies / 2 - (initial_candies / 2 / 3 + 5) = 5) → 
  initial_candies = 30 :=
by
  sorry

end NUMINAMATH_CALUDE_shinyoung_candy_problem_l2666_266621


namespace NUMINAMATH_CALUDE_greatest_product_of_three_l2666_266638

def S : Finset Int := {-6, -4, -2, 0, 1, 3, 5, 7}

theorem greatest_product_of_three (a b c : Int) : 
  a ∈ S → b ∈ S → c ∈ S → 
  a ≠ b → b ≠ c → a ≠ c → 
  ∀ x y z : Int, x ∈ S → y ∈ S → z ∈ S → 
  x ≠ y → y ≠ z → x ≠ z → 
  a * b * c ≤ 168 ∧ (∃ p q r : Int, p ∈ S ∧ q ∈ S ∧ r ∈ S ∧ p ≠ q ∧ q ≠ r ∧ p ≠ r ∧ p * q * r = 168) :=
by sorry

end NUMINAMATH_CALUDE_greatest_product_of_three_l2666_266638


namespace NUMINAMATH_CALUDE_XZ_length_l2666_266697

-- Define the circle and points
def Circle : Type := Unit
def Point : Type := ℝ × ℝ

-- Define the radius of the circle
def radius : ℝ := 7

-- Define the points on the circle
def X : Point := sorry
def Y : Point := sorry
def Z : Point := sorry
def W : Point := sorry

-- Define the distance function
def distance (p q : Point) : ℝ := sorry

-- State the conditions
axiom on_circle_X : distance X (0, 0) = radius
axiom on_circle_Y : distance Y (0, 0) = radius
axiom XY_distance : distance X Y = 8
axiom Z_midpoint_arc : sorry  -- Z is the midpoint of the minor arc XY
axiom W_midpoint_XZ : distance X W = distance W Z
axiom YW_distance : distance Y W = 6

-- State the theorem to be proved
theorem XZ_length : distance X Z = 8 := by sorry

end NUMINAMATH_CALUDE_XZ_length_l2666_266697


namespace NUMINAMATH_CALUDE_rectangle_ratio_is_two_l2666_266655

/-- Represents a rectangle with side lengths x and y -/
structure Rectangle where
  x : ℝ
  y : ℝ

/-- Represents a square configuration with an inner square and four surrounding rectangles -/
structure SquareConfig where
  inner_side : ℝ
  rect : Rectangle
  h_outer_area : (inner_side + 2 * rect.y)^2 = 9 * inner_side^2
  h_rect_placement : inner_side + rect.x = inner_side + 2 * rect.y

theorem rectangle_ratio_is_two (config : SquareConfig) :
  config.rect.x / config.rect.y = 2 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_ratio_is_two_l2666_266655


namespace NUMINAMATH_CALUDE_binomial_expansion_problem_l2666_266679

theorem binomial_expansion_problem (n : ℕ) (x : ℝ) :
  (∃ (a b : ℕ), a ≠ b ∧ a > 2 ∧ b > 2 ∧ (Nat.choose n a = Nat.choose n b)) →
  (n = 6 ∧ 
   ∃ (k : ℕ), k = 3 ∧
   ((-1)^k * 2^(n-k) * Nat.choose n k : ℤ) = -160) := by
  sorry

end NUMINAMATH_CALUDE_binomial_expansion_problem_l2666_266679


namespace NUMINAMATH_CALUDE_complex_modulus_sqrt_two_l2666_266646

theorem complex_modulus_sqrt_two (x y : ℝ) :
  (Complex.I + 1) * x = Complex.I * y + 1 →
  Complex.abs (x + Complex.I * y) = Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_complex_modulus_sqrt_two_l2666_266646


namespace NUMINAMATH_CALUDE_remainder_problem_l2666_266678

theorem remainder_problem (n : ℤ) (h : n % 7 = 2) : (4 * n + 5) % 7 = 6 := by
  sorry

end NUMINAMATH_CALUDE_remainder_problem_l2666_266678


namespace NUMINAMATH_CALUDE_unique_solution_condition_l2666_266635

/-- The equation (x - 3)(x - 5) = k - 4x has exactly one real solution if and only if k = 11 -/
theorem unique_solution_condition (k : ℝ) : 
  (∃! x : ℝ, (x - 3) * (x - 5) = k - 4 * x) ↔ k = 11 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_condition_l2666_266635


namespace NUMINAMATH_CALUDE_vector_difference_magnitude_l2666_266681

theorem vector_difference_magnitude : 
  let a : ℝ × ℝ := (Real.cos (π / 6), Real.sin (π / 6))
  let b : ℝ × ℝ := (Real.cos (5 * π / 6), Real.sin (5 * π / 6))
  ((a.1 - b.1)^2 + (a.2 - b.2)^2).sqrt = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_vector_difference_magnitude_l2666_266681


namespace NUMINAMATH_CALUDE_system_solution_condition_l2666_266650

theorem system_solution_condition (a : ℕ+) (A B : ℝ) :
  (∃ x y z : ℕ+, 
    x^2 + y^2 + z^2 = (B * (a : ℝ))^2 ∧
    x^2 * (A * x^2 + B * y^2) + y^2 * (A * y^2 + B * z^2) + z^2 * (A * z^2 + B * x^2) = 
      (1/4) * (2*A + B) * (B * (a : ℝ))^4) ↔
  B = 2 * A := by
sorry

end NUMINAMATH_CALUDE_system_solution_condition_l2666_266650


namespace NUMINAMATH_CALUDE_negation_divisible_by_five_l2666_266601

theorem negation_divisible_by_five (n : ℕ) : 
  ¬(∀ n : ℕ, n % 5 = 0 → n % 10 = 0) ↔ 
  ∃ n : ℕ, n % 5 = 0 ∧ n % 10 ≠ 0 :=
by sorry

end NUMINAMATH_CALUDE_negation_divisible_by_five_l2666_266601


namespace NUMINAMATH_CALUDE_water_tank_problem_l2666_266649

/-- Represents the time (in minutes) it takes for pipe A to fill the tank -/
def fill_time : ℚ := 15

/-- Represents the time (in minutes) it takes for pipe B to empty the tank -/
def empty_time : ℚ := 6

/-- Represents the time (in minutes) it takes to empty or fill the tank completely with both pipes open -/
def both_pipes_time : ℚ := 2

/-- Represents the fraction of the tank that is currently full -/
def current_fill : ℚ := 4/5

theorem water_tank_problem :
  (1 / fill_time - 1 / empty_time) * both_pipes_time = 1 - current_fill :=
by sorry

end NUMINAMATH_CALUDE_water_tank_problem_l2666_266649


namespace NUMINAMATH_CALUDE_buckingham_palace_visitor_difference_l2666_266693

/-- The number of visitors to Buckingham Palace on the previous day -/
def previous_day_visitors : ℕ := 100

/-- The number of visitors to Buckingham Palace on that day -/
def that_day_visitors : ℕ := 666

/-- The difference in visitors between that day and the previous day -/
def visitor_difference : ℕ := that_day_visitors - previous_day_visitors

theorem buckingham_palace_visitor_difference :
  visitor_difference = 566 :=
by sorry

end NUMINAMATH_CALUDE_buckingham_palace_visitor_difference_l2666_266693


namespace NUMINAMATH_CALUDE_A_D_relationship_l2666_266667

-- Define propositions
variable (A B C D : Prop)

-- Define the relationships between propositions
variable (h1 : A → B)
variable (h2 : ¬(B → A))
variable (h3 : B ↔ C)
variable (h4 : C → D)
variable (h5 : ¬(D → C))

-- Theorem to prove
theorem A_D_relationship : (A → D) ∧ ¬(D → A) := by sorry

end NUMINAMATH_CALUDE_A_D_relationship_l2666_266667


namespace NUMINAMATH_CALUDE_correlation_coefficient_inequality_l2666_266698

def X : List ℝ := [10, 11.3, 11.8, 12.5, 13]
def Y : List ℝ := [1, 2, 3, 4, 5]
def U : List ℝ := [10, 11.3, 11.8, 12.5, 13]
def V : List ℝ := [5, 4, 3, 2, 1]

def linear_correlation_coefficient (x y : List ℝ) : ℝ :=
  sorry

def r₁ : ℝ := linear_correlation_coefficient X Y
def r₂ : ℝ := linear_correlation_coefficient U V

theorem correlation_coefficient_inequality : r₂ < 0 ∧ 0 < r₁ := by
  sorry

end NUMINAMATH_CALUDE_correlation_coefficient_inequality_l2666_266698


namespace NUMINAMATH_CALUDE_dress_shirt_cost_l2666_266631

theorem dress_shirt_cost (num_shirts : ℕ) (tax_rate : ℝ) (total_paid : ℝ) :
  num_shirts = 3 ∧ tax_rate = 0.1 ∧ total_paid = 66 →
  ∃ (shirt_cost : ℝ), 
    shirt_cost * num_shirts * (1 + tax_rate) = total_paid ∧
    shirt_cost = 20 :=
by
  sorry

end NUMINAMATH_CALUDE_dress_shirt_cost_l2666_266631


namespace NUMINAMATH_CALUDE_negation_equivalence_l2666_266689

theorem negation_equivalence : 
  (¬ ∃ x : ℝ, x^2 > 1) ↔ (∀ x : ℝ, -1 ≤ x ∧ x ≤ 1) :=
by sorry

end NUMINAMATH_CALUDE_negation_equivalence_l2666_266689


namespace NUMINAMATH_CALUDE_equiangular_polygon_with_specific_angle_ratio_is_decagon_l2666_266685

theorem equiangular_polygon_with_specific_angle_ratio_is_decagon :
  ∀ (n : ℕ) (exterior_angle interior_angle : ℝ),
    n ≥ 3 →
    exterior_angle > 0 →
    interior_angle > 0 →
    exterior_angle + interior_angle = 180 →
    exterior_angle = (1 / 4) * interior_angle →
    360 / exterior_angle = 10 :=
by sorry

end NUMINAMATH_CALUDE_equiangular_polygon_with_specific_angle_ratio_is_decagon_l2666_266685


namespace NUMINAMATH_CALUDE_cone_lateral_area_l2666_266672

theorem cone_lateral_area (circumference : Real) (slant_height : Real) :
  circumference = 4 * Real.pi →
  slant_height = 3 →
  π * (circumference / (2 * π)) * slant_height = 6 * π :=
by sorry

end NUMINAMATH_CALUDE_cone_lateral_area_l2666_266672


namespace NUMINAMATH_CALUDE_binomial_expansion_coefficient_l2666_266641

theorem binomial_expansion_coefficient (x : ℝ) (a : Fin 9 → ℝ) :
  (x - 1)^8 = a 0 + a 1 * (1 + x) + a 2 * (1 + x)^2 + a 3 * (1 + x)^3 + 
              a 4 * (1 + x)^4 + a 5 * (1 + x)^5 + a 6 * (1 + x)^6 + 
              a 7 * (1 + x)^7 + a 8 * (1 + x)^8 →
  a 5 = -448 := by
sorry

end NUMINAMATH_CALUDE_binomial_expansion_coefficient_l2666_266641


namespace NUMINAMATH_CALUDE_min_value_w_l2666_266675

theorem min_value_w (x y : ℝ) : 3 * x^2 + 3 * y^2 + 12 * x - 6 * y + 30 ≥ 15 := by
  sorry

end NUMINAMATH_CALUDE_min_value_w_l2666_266675


namespace NUMINAMATH_CALUDE_q_over_p_is_five_thirds_l2666_266643

theorem q_over_p_is_five_thirds (P Q : ℤ) (h : ∀ (x : ℝ), x ≠ -6 ∧ x ≠ 0 ∧ x ≠ 6 →
  (P / (x + 6) + Q / (x^2 - 6*x) : ℝ) = (x^2 - 3*x + 12) / (x^3 + x^2 - 24*x)) :
  (Q : ℚ) / (P : ℚ) = 5 / 3 := by
sorry

end NUMINAMATH_CALUDE_q_over_p_is_five_thirds_l2666_266643


namespace NUMINAMATH_CALUDE_sum_of_four_squares_express_689_as_sum_of_squares_l2666_266620

theorem sum_of_four_squares (m n : ℕ) (h : m ≠ n) :
  ∃ (a b c d : ℕ), m^4 + 4*n^4 = a^2 + b^2 + c^2 + d^2 :=
sorry

theorem express_689_as_sum_of_squares :
  ∃ (a b c d : ℕ), 689 = a^2 + b^2 + c^2 + d^2 ∧ a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d :=
sorry

end NUMINAMATH_CALUDE_sum_of_four_squares_express_689_as_sum_of_squares_l2666_266620


namespace NUMINAMATH_CALUDE_circle_area_l2666_266610

/-- The area of the circle defined by the equation 3x^2 + 3y^2 + 12x - 9y - 27 = 0 is 49/4 * π -/
theorem circle_area (x y : ℝ) : 
  (3 * x^2 + 3 * y^2 + 12 * x - 9 * y - 27 = 0) → 
  (∃ (center : ℝ × ℝ) (r : ℝ), 
    ((x - center.1)^2 + (y - center.2)^2 = r^2) ∧ 
    (π * r^2 = 49/4 * π)) := by
  sorry

end NUMINAMATH_CALUDE_circle_area_l2666_266610


namespace NUMINAMATH_CALUDE_consecutive_even_sum_l2666_266603

def is_even (n : ℕ) : Prop := ∃ k, n = 2 * k

def is_consecutive_even (a b c : ℕ) : Prop :=
  is_even a ∧ is_even b ∧ is_even c ∧ b = a + 2 ∧ c = b + 2

def has_valid_digits (n : ℕ) : Prop :=
  n ≥ 20000 ∧ n < 30000 ∧
  n % 10 = 0 ∧
  (n / 10000 : ℕ) = 2 ∧
  ((n / 10) % 10 ≠ (n / 100) % 10) ∧
  ((n / 10) % 10 ≠ (n / 1000) % 10) ∧
  ((n / 100) % 10 ≠ (n / 1000) % 10)

theorem consecutive_even_sum (a b c : ℕ) :
  is_consecutive_even a b c →
  has_valid_digits (a * b * c) →
  a + b + c = 84 :=
by sorry

end NUMINAMATH_CALUDE_consecutive_even_sum_l2666_266603


namespace NUMINAMATH_CALUDE_coin_denomination_problem_l2666_266616

/-- Given a total of 334 coins, with 250 coins of 20 paise each, and a total sum of 7100 paise,
    the denomination of the remaining coins is 25 paise. -/
theorem coin_denomination_problem (total_coins : ℕ) (twenty_paise_coins : ℕ) (total_sum : ℕ) :
  total_coins = 334 →
  twenty_paise_coins = 250 →
  total_sum = 7100 →
  (total_coins - twenty_paise_coins) * (total_sum - twenty_paise_coins * 20) / (total_coins - twenty_paise_coins) = 25 := by
  sorry

#eval (334 - 250) * (7100 - 250 * 20) / (334 - 250)  -- Should output 25

end NUMINAMATH_CALUDE_coin_denomination_problem_l2666_266616


namespace NUMINAMATH_CALUDE_inequality_range_l2666_266680

theorem inequality_range (t : ℝ) : 
  (∀ x : ℝ, x ∈ Set.Icc 0 2 → 
    (1/8) * (2*t - t^2) ≤ x^2 - 3*x + 2 ∧ 
    x^2 - 3*x + 2 ≤ 3 - t^2) ↔ 
  t ∈ Set.Icc (-1) (1 - Real.sqrt 3) := by
sorry

end NUMINAMATH_CALUDE_inequality_range_l2666_266680


namespace NUMINAMATH_CALUDE_flowers_in_vase_l2666_266623

/-- Given that Lara bought 52 stems of flowers, gave 15 to her mom, and gave 6 more to her grandma
    than to her mom, prove that she put 16 stems in the vase. -/
theorem flowers_in_vase (total : ℕ) (to_mom : ℕ) (extra_to_grandma : ℕ)
    (h1 : total = 52)
    (h2 : to_mom = 15)
    (h3 : extra_to_grandma = 6)
    : total - (to_mom + (to_mom + extra_to_grandma)) = 16 := by
  sorry

end NUMINAMATH_CALUDE_flowers_in_vase_l2666_266623


namespace NUMINAMATH_CALUDE_simplify_expression_l2666_266642

theorem simplify_expression (a : ℝ) (h : 1 < a ∧ a < 2) :
  Real.sqrt ((a - 3)^2) + |1 - a| = 2 := by
sorry

end NUMINAMATH_CALUDE_simplify_expression_l2666_266642


namespace NUMINAMATH_CALUDE_sin_pi_over_4n_lower_bound_l2666_266690

theorem sin_pi_over_4n_lower_bound (n : ℕ) (hn : n > 0) :
  Real.sin (π / (4 * n)) ≥ Real.sqrt 2 / (2 * n) := by
  sorry

end NUMINAMATH_CALUDE_sin_pi_over_4n_lower_bound_l2666_266690


namespace NUMINAMATH_CALUDE_sin_shift_equivalence_l2666_266695

theorem sin_shift_equivalence (x : ℝ) :
  2 * Real.sin (3 * x + π / 4) = 2 * Real.sin (3 * (x + π / 12)) :=
by sorry

end NUMINAMATH_CALUDE_sin_shift_equivalence_l2666_266695


namespace NUMINAMATH_CALUDE_exam_mean_score_l2666_266618

theorem exam_mean_score (score_below mean score_above : ℝ) 
  (h1 : score_below = mean - 2 * (score_above - mean) / 5)
  (h2 : score_above = mean + 3 * (score_above - mean) / 5)
  (h3 : score_below = 60)
  (h4 : score_above = 100) : 
  mean = 76 := by
sorry

end NUMINAMATH_CALUDE_exam_mean_score_l2666_266618


namespace NUMINAMATH_CALUDE_circle_k_value_l2666_266660

def larger_circle_radius : ℝ := 15
def smaller_circle_radius : ℝ := 10
def point_P : ℝ × ℝ := (9, 12)
def point_S (k : ℝ) : ℝ × ℝ := (0, k)
def QR : ℝ := 5

theorem circle_k_value :
  ∀ k : ℝ,
  (point_P.1^2 + point_P.2^2 = larger_circle_radius^2) →
  ((point_S k).1^2 + (point_S k).2^2 = smaller_circle_radius^2) →
  (larger_circle_radius - smaller_circle_radius = QR) →
  (k = 10 ∨ k = -10) :=
by sorry

end NUMINAMATH_CALUDE_circle_k_value_l2666_266660


namespace NUMINAMATH_CALUDE_complex_equation_unique_solution_l2666_266683

theorem complex_equation_unique_solution :
  ∃! (c : ℝ), Complex.abs (1 - 2 * Complex.I - (c - 3 * Complex.I)) = 1 :=
by sorry

end NUMINAMATH_CALUDE_complex_equation_unique_solution_l2666_266683


namespace NUMINAMATH_CALUDE_reciprocal_problem_l2666_266628

theorem reciprocal_problem (x : ℚ) (h : 8 * x = 3) : 50 * (1 / x) = 400 / 3 := by
  sorry

end NUMINAMATH_CALUDE_reciprocal_problem_l2666_266628
