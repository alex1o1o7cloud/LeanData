import Mathlib

namespace NUMINAMATH_CALUDE_sum_of_reciprocals_l2292_229278

theorem sum_of_reciprocals (a b : ℝ) (ha : a^2 + a = 4) (hb : b^2 + b = 4) (hab : a ≠ b) :
  b / a + a / b = -9/4 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_reciprocals_l2292_229278


namespace NUMINAMATH_CALUDE_game_cost_l2292_229207

def initial_money : ℕ := 63
def toy_price : ℕ := 3
def toys_affordable : ℕ := 5

def remaining_money : ℕ := toy_price * toys_affordable

theorem game_cost : initial_money - remaining_money = 48 := by
  sorry

end NUMINAMATH_CALUDE_game_cost_l2292_229207


namespace NUMINAMATH_CALUDE_right_triangle_hypotenuse_l2292_229232

theorem right_triangle_hypotenuse (a b c : ℝ) 
  (h1 : a = 5) 
  (h2 : b = 12) 
  (h3 : c^2 = a^2 + b^2) : c = 13 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_hypotenuse_l2292_229232


namespace NUMINAMATH_CALUDE_invisible_dots_count_l2292_229267

/-- The total number of dots on a single die -/
def dots_per_die : ℕ := 21

/-- The sum of visible numbers on the stacked dice -/
def visible_sum : ℕ := 2 + 2 + 3 + 4 + 5 + 5 + 6 + 6

/-- The number of dice stacked -/
def num_dice : ℕ := 3

/-- The number of visible faces -/
def visible_faces : ℕ := 8

theorem invisible_dots_count : 
  num_dice * dots_per_die - visible_sum = 30 := by
sorry

end NUMINAMATH_CALUDE_invisible_dots_count_l2292_229267


namespace NUMINAMATH_CALUDE_unique_n_with_no_constant_term_l2292_229234

/-- The expansion of (1+x+x²)(x+1/x³)ⁿ has no constant term -/
def has_no_constant_term (n : ℕ) : Prop :=
  ∀ (x : ℝ), x ≠ 0 → (1 + x + x^2) * (x + 1/x^3)^n ≠ 1

theorem unique_n_with_no_constant_term :
  ∃! (n : ℕ), 2 ≤ n ∧ n ≤ 8 ∧ has_no_constant_term n ∧ n = 5 := by
  sorry

end NUMINAMATH_CALUDE_unique_n_with_no_constant_term_l2292_229234


namespace NUMINAMATH_CALUDE_convex_polygon_angle_sum_l2292_229288

theorem convex_polygon_angle_sum (n : ℕ) (angle_sum : ℝ) : n = 17 → angle_sum = 2610 → ∃ (missing_angle : ℝ), 
  0 < missing_angle ∧ 
  missing_angle < 180 ∧ 
  (180 * (n - 2) : ℝ) = angle_sum + missing_angle := by
  sorry

end NUMINAMATH_CALUDE_convex_polygon_angle_sum_l2292_229288


namespace NUMINAMATH_CALUDE_blueberries_needed_for_pies_l2292_229281

-- Define the constants
def blueberries_per_pint : ℕ := 200
def pints_per_quart : ℕ := 2
def pies_to_make : ℕ := 6

-- Define the theorem
theorem blueberries_needed_for_pies : 
  blueberries_per_pint * pints_per_quart * pies_to_make = 2400 := by
  sorry

end NUMINAMATH_CALUDE_blueberries_needed_for_pies_l2292_229281


namespace NUMINAMATH_CALUDE_inverse_f_at_4_l2292_229201

-- Define the function f
def f : ℝ → ℝ := sorry

-- Define the inverse function of f
def f_inv : ℝ → ℝ := sorry

-- State the symmetry condition
axiom symmetry_condition (x : ℝ) : f (x + 1) + f (1 - x) = 4

-- State that f has an inverse
axiom has_inverse : Function.Bijective f

-- State that f(4) = 0
axiom f_at_4 : f 4 = 0

-- Theorem to prove
theorem inverse_f_at_4 : f_inv 4 = -2 := by sorry

end NUMINAMATH_CALUDE_inverse_f_at_4_l2292_229201


namespace NUMINAMATH_CALUDE_min_distance_complex_l2292_229289

theorem min_distance_complex (z : ℂ) (h : Complex.abs (z + 2 - 2*I) = 1) :
  ∃ (min_val : ℝ), min_val = 2 ∧ ∀ w : ℂ, Complex.abs (w + 2 - 2*I) = 1 → Complex.abs (w - 1 - 2*I) ≥ min_val :=
sorry

end NUMINAMATH_CALUDE_min_distance_complex_l2292_229289


namespace NUMINAMATH_CALUDE_sqrt_2_simplest_l2292_229220

def is_simplest_quadratic_radical (x : ℝ) : Prop :=
  ∃ (y : ℝ), x = Real.sqrt y ∧ (∀ (z : ℝ), z ^ 2 = y → z = x)

theorem sqrt_2_simplest :
  is_simplest_quadratic_radical (Real.sqrt 2) ∧
  ¬is_simplest_quadratic_radical (3 ^ (1/3 : ℝ)) ∧
  ¬is_simplest_quadratic_radical (Real.sqrt (1/2)) ∧
  ¬is_simplest_quadratic_radical (Real.sqrt 16) :=
by sorry

end NUMINAMATH_CALUDE_sqrt_2_simplest_l2292_229220


namespace NUMINAMATH_CALUDE_marie_total_sales_l2292_229286

/-- The total number of items Marie sold on Saturday -/
def total_sold (newspapers : ℝ) (magazines : ℕ) : ℝ :=
  newspapers + magazines

/-- Theorem: Marie sold 425 items in total -/
theorem marie_total_sales : total_sold 275.0 150 = 425 := by
  sorry

end NUMINAMATH_CALUDE_marie_total_sales_l2292_229286


namespace NUMINAMATH_CALUDE_trapezoid_is_plane_figure_l2292_229298

-- Define a trapezoid
structure Trapezoid :=
  (hasParallelLines : Bool)

-- Define a plane figure
structure PlaneFigure

-- Theorem: A trapezoid is a plane figure
theorem trapezoid_is_plane_figure (t : Trapezoid) (h : t.hasParallelLines = true) : PlaneFigure :=
sorry

end NUMINAMATH_CALUDE_trapezoid_is_plane_figure_l2292_229298


namespace NUMINAMATH_CALUDE_cubic_factorization_l2292_229246

theorem cubic_factorization (x : ℝ) : x^3 - 5*x^2 + 4*x = x*(x-1)*(x-4) := by
  sorry

end NUMINAMATH_CALUDE_cubic_factorization_l2292_229246


namespace NUMINAMATH_CALUDE_image_of_negative_one_two_l2292_229222

-- Define the set of real pairs
def RealPair := ℝ × ℝ

-- Define the mapping f
def f (p : RealPair) : RealPair :=
  (p.1 - p.2, p.1 + p.2)

-- Theorem statement
theorem image_of_negative_one_two :
  f (-1, 2) = (-3, 1) := by
  sorry

end NUMINAMATH_CALUDE_image_of_negative_one_two_l2292_229222


namespace NUMINAMATH_CALUDE_least_x_for_even_prime_l2292_229236

theorem least_x_for_even_prime (x : ℕ+) (p : ℕ) : 
  Nat.Prime p → (x.val : ℚ) / (11 * p) = 2 → x.val ≥ 44 :=
sorry

end NUMINAMATH_CALUDE_least_x_for_even_prime_l2292_229236


namespace NUMINAMATH_CALUDE_largest_double_after_digit_removal_l2292_229224

def is_double_after_digit_removal (x : ℚ) : Prop :=
  ∃ (y : ℚ), (y > 0) ∧ (y < 1) ∧ (x = 0.1 * 3 + y) ∧ (2 * x = 0.1 * 0 + y)

theorem largest_double_after_digit_removal :
  ∀ (x : ℚ), (x > 0) → (x < 1) → is_double_after_digit_removal x → x ≤ 0.375 :=
sorry

end NUMINAMATH_CALUDE_largest_double_after_digit_removal_l2292_229224


namespace NUMINAMATH_CALUDE_lawrence_county_kids_count_lawrence_county_total_kids_l2292_229247

theorem lawrence_county_kids_count : ℕ → ℕ → ℕ
  | kids_at_home, kids_at_camp =>
    kids_at_home + kids_at_camp

theorem lawrence_county_total_kids :
  lawrence_county_kids_count 907611 455682 = 1363293 := by
  sorry

end NUMINAMATH_CALUDE_lawrence_county_kids_count_lawrence_county_total_kids_l2292_229247


namespace NUMINAMATH_CALUDE_ordering_of_a_b_c_l2292_229296

theorem ordering_of_a_b_c :
  let a := Real.tan (1/2)
  let b := Real.tan (2/π)
  let c := Real.sqrt 3 / π
  a < c ∧ c < b := by sorry

end NUMINAMATH_CALUDE_ordering_of_a_b_c_l2292_229296


namespace NUMINAMATH_CALUDE_quadratic_equation_positive_root_l2292_229209

theorem quadratic_equation_positive_root (m : ℝ) :
  ∃ x : ℝ, x > 0 ∧ (x - 1) * (x - 2) - m^2 = 0 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_positive_root_l2292_229209


namespace NUMINAMATH_CALUDE_division_negative_ten_by_five_l2292_229256

theorem division_negative_ten_by_five : -10 / 5 = -2 := by sorry

end NUMINAMATH_CALUDE_division_negative_ten_by_five_l2292_229256


namespace NUMINAMATH_CALUDE_like_terms_exponent_difference_l2292_229263

/-- Given that 2a^m * b^2 and -a^5 * b^n are like terms, prove that n-m = -3 -/
theorem like_terms_exponent_difference (a b : ℝ) (m n : ℤ) 
  (h : ∃ (k : ℝ), 2 * a^m * b^2 = k * (-a^5 * b^n)) : n - m = -3 := by
  sorry

end NUMINAMATH_CALUDE_like_terms_exponent_difference_l2292_229263


namespace NUMINAMATH_CALUDE_marbles_selection_count_l2292_229290

/-- The number of ways to choose 4 marbles from 8, with at least one red -/
def choose_marbles (total_marbles : ℕ) (red_marbles : ℕ) (choose : ℕ) : ℕ :=
  Nat.choose (total_marbles - red_marbles) (choose - 1)

/-- Theorem: There are 35 ways to choose 4 marbles from 8, with at least one red -/
theorem marbles_selection_count :
  choose_marbles 8 1 4 = 35 := by
  sorry

end NUMINAMATH_CALUDE_marbles_selection_count_l2292_229290


namespace NUMINAMATH_CALUDE_part_one_part_two_l2292_229293

-- Definitions for propositions q and p
def prop_q (a : ℝ) : Prop := ∃ x : ℝ, -2 < x ∧ x < 2 ∧ x^2 - 2*x - a = 0

def prop_p (a : ℝ) : Prop := ∃ x : ℝ, x^2 + (a-1)*x + 4 < 0

-- Part 1
theorem part_one (a : ℝ) : prop_q a ∨ prop_p a → a < -3 ∨ a ≥ -1 := by sorry

-- Definitions for part 2
def prop_p_part2 (a : ℝ) : Prop := ∃ x : ℝ, 2*a < x ∧ x < a + 1

-- Part 2
theorem part_two (a : ℝ) : 
  (∀ x : ℝ, prop_p_part2 a → prop_q a) ∧ 
  (∃ x : ℝ, prop_q a ∧ ¬prop_p_part2 a) → 
  a ≥ -1/2 := by sorry

end NUMINAMATH_CALUDE_part_one_part_two_l2292_229293


namespace NUMINAMATH_CALUDE_largest_radius_special_circle_l2292_229221

/-- A circle with the given properties -/
structure SpecialCircle where
  center : ℝ × ℝ
  radius : ℝ
  contains_points : center.1^2 + (center.2 - 11)^2 = radius^2 ∧
                    center.1^2 + (center.2 + 11)^2 = radius^2
  contains_unit_disk : ∀ (x y : ℝ), x^2 + y^2 < 1 →
                       (x - center.1)^2 + (y - center.2)^2 < radius^2

/-- The theorem stating the largest possible radius -/
theorem largest_radius_special_circle :
  ∃ (c : SpecialCircle), ∀ (c' : SpecialCircle), c'.radius ≤ c.radius ∧ c.radius = Real.sqrt 122 :=
sorry

end NUMINAMATH_CALUDE_largest_radius_special_circle_l2292_229221


namespace NUMINAMATH_CALUDE_arithmetic_progression_property_l2292_229202

/-- An arithmetic progression with n terms and common difference d -/
structure ArithmeticProgression where
  n : ℕ
  d : ℚ
  first_term : ℚ

/-- The sum of absolute values of terms in an arithmetic progression -/
def sum_of_abs_values (ap : ArithmeticProgression) : ℚ :=
  sorry

/-- Theorem stating the relation between n, d, and the sum of absolute values -/
theorem arithmetic_progression_property (ap : ArithmeticProgression) :
  (sum_of_abs_values ap = 100) ∧
  (sum_of_abs_values { ap with first_term := ap.first_term + 1 } = 100) ∧
  (sum_of_abs_values { ap with first_term := ap.first_term + 2 } = 100) →
  ap.n^2 * ap.d = 400 :=
sorry

end NUMINAMATH_CALUDE_arithmetic_progression_property_l2292_229202


namespace NUMINAMATH_CALUDE_line_direction_vector_l2292_229229

/-- Given a line with direction vector (a, -2) passing through points (-3, 7) and (2, -1),
    prove that a = 5/4 -/
theorem line_direction_vector (a : ℝ) : 
  (∃ t : ℝ, (2 : ℝ) = -3 + t * a ∧ (-1 : ℝ) = 7 + t * (-2)) → a = 5/4 := by
  sorry

end NUMINAMATH_CALUDE_line_direction_vector_l2292_229229


namespace NUMINAMATH_CALUDE_line_through_points_slope_one_l2292_229276

/-- Given a line passing through points M(-2, m) and N(m, 4) with a slope of 1, prove that m = 1 -/
theorem line_through_points_slope_one (m : ℝ) : 
  (4 - m) / (m - (-2)) = 1 → m = 1 := by
  sorry

end NUMINAMATH_CALUDE_line_through_points_slope_one_l2292_229276


namespace NUMINAMATH_CALUDE_f_continuous_at_2_l2292_229248

-- Define the piecewise function f
noncomputable def f (b : ℝ) (x : ℝ) : ℝ :=
  if x ≤ 2 then 3 * x^2 - 7 else b * (x - 2)^2 + 5

-- State the theorem
theorem f_continuous_at_2 (b : ℝ) : 
  ContinuousAt (f b) 2 := by sorry

end NUMINAMATH_CALUDE_f_continuous_at_2_l2292_229248


namespace NUMINAMATH_CALUDE_card_arrangement_probability_l2292_229200

/-- The probability of arranging cards to form a specific word --/
theorem card_arrangement_probability (n : ℕ) (n1 n2 : ℕ) (h1 : n = n1 + n2) (h2 : n1 = 2) (h3 : n2 = 3) :
  (1 : ℚ) / (n.factorial / (n1.factorial * n2.factorial)) = 1 / 10 := by
  sorry

end NUMINAMATH_CALUDE_card_arrangement_probability_l2292_229200


namespace NUMINAMATH_CALUDE_f_minimum_l2292_229277

noncomputable def f (a x : ℝ) : ℝ := x^2 + |x - a| - 1

theorem f_minimum (a : ℝ) : 
  (∀ x, f a x ≥ -a - 5/4 ∧ ∃ x, f a x = -a - 5/4) ∨
  (∀ x, f a x ≥ a^2 - 1 ∧ ∃ x, f a x = a^2 - 1) ∨
  (∀ x, f a x ≥ a - 5/4 ∧ ∃ x, f a x = a - 5/4) :=
by sorry

end NUMINAMATH_CALUDE_f_minimum_l2292_229277


namespace NUMINAMATH_CALUDE_point_not_in_fourth_quadrant_l2292_229262

theorem point_not_in_fourth_quadrant (m : ℝ) :
  ¬(m > 0 ∧ m + 1 < 0) :=
by sorry

end NUMINAMATH_CALUDE_point_not_in_fourth_quadrant_l2292_229262


namespace NUMINAMATH_CALUDE_equation_real_root_l2292_229265

theorem equation_real_root (k : ℝ) : ∃ x : ℝ, x = k^2 * (x - 1) * (x - 2) := by
  sorry

end NUMINAMATH_CALUDE_equation_real_root_l2292_229265


namespace NUMINAMATH_CALUDE_stream_current_rate_l2292_229208

/-- Proves that the rate of a stream's current is 4 kmph given the conditions of a boat's travel --/
theorem stream_current_rate (distance_one_way : ℝ) (total_time : ℝ) (still_water_speed : ℝ) 
  (h1 : distance_one_way = 6)
  (h2 : total_time = 2)
  (h3 : still_water_speed = 8) :
  ∃ c : ℝ, c = 4 ∧ 
    (distance_one_way / (still_water_speed - c) + distance_one_way / (still_water_speed + c) = total_time) :=
by sorry

end NUMINAMATH_CALUDE_stream_current_rate_l2292_229208


namespace NUMINAMATH_CALUDE_cara_speed_l2292_229250

/-- 
Proves that given a distance of 120 miles between two cities, 
if a person (Dan) leaving 60 minutes after another person (Cara) 
must exceed 40 mph to arrive first, then the first person's (Cara's) 
constant speed is 30 mph.
-/
theorem cara_speed (distance : ℝ) (dan_delay : ℝ) (dan_min_speed : ℝ) : 
  distance = 120 → 
  dan_delay = 1 → 
  dan_min_speed = 40 → 
  ∃ (cara_speed : ℝ), 
    cara_speed * (distance / dan_min_speed + dan_delay) = distance ∧ 
    cara_speed = 30 := by
  sorry


end NUMINAMATH_CALUDE_cara_speed_l2292_229250


namespace NUMINAMATH_CALUDE_stating_whack_a_mole_tickets_correct_l2292_229295

/-- Represents the number of tickets Tom won from 'whack a mole' -/
def whack_a_mole_tickets : ℕ := 32

/-- Represents the number of tickets Tom won from 'skee ball' -/
def skee_ball_tickets : ℕ := 25

/-- Represents the number of tickets Tom spent on a hat -/
def spent_tickets : ℕ := 7

/-- Represents the number of tickets Tom has left -/
def remaining_tickets : ℕ := 50

/-- 
Theorem stating that the number of tickets Tom won from 'whack a mole' 
is correct given the other known information
-/
theorem whack_a_mole_tickets_correct : 
  whack_a_mole_tickets + skee_ball_tickets = remaining_tickets + spent_tickets := by
  sorry

#check whack_a_mole_tickets_correct

end NUMINAMATH_CALUDE_stating_whack_a_mole_tickets_correct_l2292_229295


namespace NUMINAMATH_CALUDE_triangle_inequality_l2292_229231

theorem triangle_inequality (a b c : ℝ) (h : 0 < a ∧ 0 < b ∧ 0 < c ∧ a + b > c ∧ b + c > a ∧ c + a > b) :
  (a + b) * (b + c) * (c + a) ≥ 8 * (a + b - c) * (b + c - a) * (c + a - b) := by
  sorry

end NUMINAMATH_CALUDE_triangle_inequality_l2292_229231


namespace NUMINAMATH_CALUDE_projection_vector_l2292_229283

/-- Given vectors a and b in ℝ², prove that the projection of a onto b is equal to the expected result. -/
theorem projection_vector (a b : ℝ × ℝ) (ha : a = (2, 4)) (hb : b = (-1, 2)) :
  let proj := (((a.1 * b.1 + a.2 * b.2) / (b.1^2 + b.2^2)) * b.1,
               ((a.1 * b.1 + a.2 * b.2) / (b.1^2 + b.2^2)) * b.2)
  proj = (-6/5, 12/5) := by
  sorry

end NUMINAMATH_CALUDE_projection_vector_l2292_229283


namespace NUMINAMATH_CALUDE_total_donation_is_1684_l2292_229285

/-- Represents the donations to four forest reserves --/
structure ForestDonations where
  treetown : ℝ
  forest_reserve : ℝ
  animal_preservation : ℝ
  birds_sanctuary : ℝ

/-- Theorem stating the total donation given the conditions --/
theorem total_donation_is_1684 (d : ForestDonations) : 
  d.treetown = 570 ∧ 
  d.forest_reserve = d.animal_preservation + 140 ∧
  5 * d.treetown = 4 * d.forest_reserve ∧
  5 * d.treetown = 2 * d.animal_preservation ∧
  5 * d.treetown = 3 * d.birds_sanctuary →
  d.treetown + d.forest_reserve + d.animal_preservation + d.birds_sanctuary = 1684 :=
by sorry

end NUMINAMATH_CALUDE_total_donation_is_1684_l2292_229285


namespace NUMINAMATH_CALUDE_painted_cubes_ratio_l2292_229264

/-- Represents a rectangular prism with integer dimensions -/
structure RectangularPrism where
  length : ℕ
  width : ℕ
  height : ℕ

/-- Counts the number of cubes with exactly two painted faces in a rectangular prism -/
def count_two_faces (prism : RectangularPrism) : ℕ :=
  4 * ((prism.length - 2) + (prism.width - 2) + (prism.height - 2))

/-- Counts the number of cubes with exactly three painted faces in a rectangular prism -/
def count_three_faces (prism : RectangularPrism) : ℕ := 8

/-- The main theorem statement -/
theorem painted_cubes_ratio (prism : RectangularPrism)
    (h_length : prism.length = 4)
    (h_width : prism.width = 5)
    (h_height : prism.height = 6) :
    (count_two_faces prism) / (count_three_faces prism) = 9 / 2 := by
  sorry


end NUMINAMATH_CALUDE_painted_cubes_ratio_l2292_229264


namespace NUMINAMATH_CALUDE_sum_first_six_primes_mod_seventh_prime_l2292_229257

def first_seven_primes : List Nat := [2, 3, 5, 7, 11, 13, 17]

theorem sum_first_six_primes_mod_seventh_prime : 
  (List.sum (List.take 6 first_seven_primes)) % (List.get! first_seven_primes 6) = 7 := by
  sorry

end NUMINAMATH_CALUDE_sum_first_six_primes_mod_seventh_prime_l2292_229257


namespace NUMINAMATH_CALUDE_p_sufficient_not_necessary_l2292_229227

-- Define the conditions
def p (x : ℝ) : Prop := |x - 1| < 2
def q (x : ℝ) : Prop := x^2 - 5*x - 6 < 0

-- Theorem statement
theorem p_sufficient_not_necessary : 
  (∀ x : ℝ, p x → q x) ∧ 
  (∃ x : ℝ, q x ∧ ¬(p x)) := by sorry

end NUMINAMATH_CALUDE_p_sufficient_not_necessary_l2292_229227


namespace NUMINAMATH_CALUDE_derivative_at_x0_l2292_229292

theorem derivative_at_x0 (f : ℝ → ℝ) (x₀ : ℝ) (h : Differentiable ℝ f) :
  (∀ ε > 0, ∃ δ > 0, ∀ Δx ≠ 0, |Δx| < δ → 
    |(f (x₀ - 2*Δx) - f x₀) / Δx - 2| < ε) →
  deriv f x₀ = -1 := by
sorry

end NUMINAMATH_CALUDE_derivative_at_x0_l2292_229292


namespace NUMINAMATH_CALUDE_shared_property_of_shapes_l2292_229287

-- Define the basic shape
structure Shape :=
  (vertices : Fin 4 → ℝ × ℝ)

-- Define the property of having opposite sides parallel and equal
def has_opposite_sides_parallel_and_equal (s : Shape) : Prop :=
  let v := s.vertices
  (v 0 - v 1 = v 3 - v 2) ∧ (v 1 - v 2 = v 0 - v 3)

-- Define the specific shapes
def is_parallelogram (s : Shape) : Prop :=
  has_opposite_sides_parallel_and_equal s

def is_rectangle (s : Shape) : Prop :=
  is_parallelogram s ∧
  let v := s.vertices
  (v 1 - v 0) • (v 2 - v 1) = 0

def is_rhombus (s : Shape) : Prop :=
  is_parallelogram s ∧
  let v := s.vertices
  ‖v 1 - v 0‖ = ‖v 2 - v 1‖

def is_square (s : Shape) : Prop :=
  is_rectangle s ∧ is_rhombus s

-- Theorem statement
theorem shared_property_of_shapes (s : Shape) :
  (is_parallelogram s ∨ is_rectangle s ∨ is_rhombus s ∨ is_square s) →
  has_opposite_sides_parallel_and_equal s :=
sorry

end NUMINAMATH_CALUDE_shared_property_of_shapes_l2292_229287


namespace NUMINAMATH_CALUDE_distance_of_problem_lines_l2292_229297

/-- Two parallel lines in 2D space -/
structure ParallelLines where
  a : ℝ × ℝ  -- Point on the first line
  b : ℝ × ℝ  -- Point on the second line
  d : ℝ × ℝ  -- Direction vector (same for both lines)

/-- The distance between two parallel lines -/
def distance_between_parallel_lines (lines : ParallelLines) : ℝ :=
  sorry

/-- The specific parallel lines from the problem -/
def problem_lines : ParallelLines :=
  { a := (3, -2)
    b := (5, -1)
    d := (2, -5) }

theorem distance_of_problem_lines :
  distance_between_parallel_lines problem_lines = 2 * Real.sqrt 109 / 29 :=
sorry

end NUMINAMATH_CALUDE_distance_of_problem_lines_l2292_229297


namespace NUMINAMATH_CALUDE_games_in_own_division_l2292_229205

/-- Represents a baseball league with specific game scheduling rules -/
structure BaseballLeague where
  N : ℕ  -- Number of games against each team in own division
  M : ℕ  -- Number of games against each team in other division
  h1 : N > 2 * M
  h2 : M > 4
  h3 : 4 * N + 5 * M = 82

/-- The number of games a team plays within its own division is 52 -/
theorem games_in_own_division (league : BaseballLeague) : 4 * league.N = 52 := by
  sorry

end NUMINAMATH_CALUDE_games_in_own_division_l2292_229205


namespace NUMINAMATH_CALUDE_total_fishes_in_aquatic_reserve_l2292_229261

theorem total_fishes_in_aquatic_reserve (bodies_of_water : ℕ) (fishes_per_body : ℕ) 
  (h1 : bodies_of_water = 6) 
  (h2 : fishes_per_body = 175) : 
  bodies_of_water * fishes_per_body = 1050 := by
  sorry

end NUMINAMATH_CALUDE_total_fishes_in_aquatic_reserve_l2292_229261


namespace NUMINAMATH_CALUDE_chess_competition_games_l2292_229266

theorem chess_competition_games (W M : ℕ) (h1 : W = 12) (h2 : M = 24) : W * M = 288 := by
  sorry

end NUMINAMATH_CALUDE_chess_competition_games_l2292_229266


namespace NUMINAMATH_CALUDE_vector_perpendicular_l2292_229280

theorem vector_perpendicular (a b : ℝ × ℝ) :
  a = (2, 0) →
  b = (-1, 1) →
  b • (a + b) = 0 := by
sorry

end NUMINAMATH_CALUDE_vector_perpendicular_l2292_229280


namespace NUMINAMATH_CALUDE_cylinder_properties_l2292_229212

/-- Properties of a cylinder with height 15 and radius 5 -/
theorem cylinder_properties :
  let h : ℝ := 15
  let r : ℝ := 5
  let total_surface_area : ℝ := 2 * Real.pi * r * r + 2 * Real.pi * r * h
  let volume : ℝ := Real.pi * r * r * h
  (total_surface_area = 200 * Real.pi) ∧ (volume = 375 * Real.pi) := by
  sorry

end NUMINAMATH_CALUDE_cylinder_properties_l2292_229212


namespace NUMINAMATH_CALUDE_square_of_binomial_l2292_229228

theorem square_of_binomial (b : ℚ) : 
  (∃ (c : ℚ), ∀ (x : ℚ), 9*x^2 + 21*x + b = (3*x + c)^2) → b = 49/4 := by
  sorry

end NUMINAMATH_CALUDE_square_of_binomial_l2292_229228


namespace NUMINAMATH_CALUDE_reciprocals_not_arithmetic_sequence_l2292_229233

theorem reciprocals_not_arithmetic_sequence (a b c : ℝ) :
  a > 0 → b > 0 → c > 0 →
  (b - a ≠ 0) →
  (c - b = b - a) →
  ¬(1/b - 1/a = 1/c - 1/b) := by
sorry

end NUMINAMATH_CALUDE_reciprocals_not_arithmetic_sequence_l2292_229233


namespace NUMINAMATH_CALUDE_absolute_value_equation_l2292_229253

theorem absolute_value_equation (x y : ℝ) :
  |x - Real.log y| = x + Real.log y → x * (y - 1) = 0 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_equation_l2292_229253


namespace NUMINAMATH_CALUDE_min_coefficient_value_l2292_229274

theorem min_coefficient_value (a b : ℤ) (box : ℤ) : 
  (∀ x, (a * x + b) * (b * x + a) = 15 * x^2 + box * x + 15) →
  a ≠ b ∧ a ≠ box ∧ b ≠ box →
  ∃ (min_box : ℤ), (min_box = 34 ∧ box ≥ min_box) := by
sorry

end NUMINAMATH_CALUDE_min_coefficient_value_l2292_229274


namespace NUMINAMATH_CALUDE_sqrt_meaningful_range_l2292_229237

theorem sqrt_meaningful_range (x : ℝ) : 
  (∃ y : ℝ, y ^ 2 = 1 - x) ↔ x ≤ 1 := by sorry

end NUMINAMATH_CALUDE_sqrt_meaningful_range_l2292_229237


namespace NUMINAMATH_CALUDE_tangent_circles_area_ratio_l2292_229244

/-- Regular hexagon with side length 2 -/
structure RegularHexagon :=
  (side_length : ℝ)
  (is_regular : side_length = 2)

/-- Circle tangent to two lines of the hexagon -/
structure TangentCircle (h : RegularHexagon) :=
  (radius : ℝ)
  (tangent_to_side : Bool)
  (tangent_to_ef : Bool)

/-- The ratio of areas of two tangent circles is 1 -/
theorem tangent_circles_area_ratio 
  (h : RegularHexagon) 
  (c1 c2 : TangentCircle h) 
  (h1 : c1.tangent_to_side = true) 
  (h2 : c2.tangent_to_side = true) 
  (h3 : c1.tangent_to_ef = true) 
  (h4 : c2.tangent_to_ef = true) : 
  (c2.radius^2) / (c1.radius^2) = 1 := by
  sorry

end NUMINAMATH_CALUDE_tangent_circles_area_ratio_l2292_229244


namespace NUMINAMATH_CALUDE_no_solution_for_equation_l2292_229238

theorem no_solution_for_equation (x y z t : ℕ) : 3 * x^4 + 5 * y^4 + 7 * z^4 ≠ 11 * t^4 := by
  sorry

end NUMINAMATH_CALUDE_no_solution_for_equation_l2292_229238


namespace NUMINAMATH_CALUDE_equation1_no_solution_equation2_unique_solution_l2292_229282

-- Define the equations
def equation1 (x : ℝ) : Prop := (4 - x) / (x - 3) + 1 / (3 - x) = 1
def equation2 (x : ℝ) : Prop := (x + 1) / (x - 1) - 6 / (x^2 - 1) = 1

-- Theorem for equation 1
theorem equation1_no_solution : ¬∃ x : ℝ, equation1 x := by sorry

-- Theorem for equation 2
theorem equation2_unique_solution : ∃! x : ℝ, equation2 x ∧ x = 2 := by sorry

end NUMINAMATH_CALUDE_equation1_no_solution_equation2_unique_solution_l2292_229282


namespace NUMINAMATH_CALUDE_c2h6_moles_used_l2292_229211

-- Define the chemical species
structure ChemicalSpecies where
  formula : String
  moles : ℝ

-- Define the balanced chemical equation
def balancedEquation (reactant1 reactant2 product1 product2 : ChemicalSpecies) : Prop :=
  reactant1.formula = "C2H6" ∧
  reactant2.formula = "Cl2" ∧
  product1.formula = "C2Cl6" ∧
  product2.formula = "HCl" ∧
  reactant1.moles = 1 ∧
  reactant2.moles = 6 ∧
  product1.moles = 1 ∧
  product2.moles = 6

-- Define the reaction conditions
def reactionConditions (cl2 c2cl6 : ChemicalSpecies) : Prop :=
  cl2.formula = "Cl2" ∧
  cl2.moles = 6 ∧
  c2cl6.formula = "C2Cl6" ∧
  c2cl6.moles = 1

-- Theorem: The number of moles of C2H6 used in the reaction is 1
theorem c2h6_moles_used
  (reactant1 reactant2 product1 product2 cl2 c2cl6 : ChemicalSpecies)
  (h1 : balancedEquation reactant1 reactant2 product1 product2)
  (h2 : reactionConditions cl2 c2cl6) :
  ∃ c2h6 : ChemicalSpecies, c2h6.formula = "C2H6" ∧ c2h6.moles = 1 :=
sorry

end NUMINAMATH_CALUDE_c2h6_moles_used_l2292_229211


namespace NUMINAMATH_CALUDE_dissimilar_terms_eq_choose_l2292_229242

/-- The number of dissimilar terms in the expansion of (x + y + z)^8 -/
def dissimilar_terms : ℕ :=
  Nat.choose 10 2

/-- Theorem stating that the number of dissimilar terms in (x + y + z)^8 is equal to (10 choose 2) -/
theorem dissimilar_terms_eq_choose : dissimilar_terms = 45 := by
  sorry

end NUMINAMATH_CALUDE_dissimilar_terms_eq_choose_l2292_229242


namespace NUMINAMATH_CALUDE_absolute_value_inequality_solution_set_l2292_229271

theorem absolute_value_inequality_solution_set (x : ℝ) :
  (1 < |2*x - 1| ∧ |2*x - 1| < 3) ↔ ((-1 < x ∧ x < 0) ∨ (1 < x ∧ x < 2)) := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_inequality_solution_set_l2292_229271


namespace NUMINAMATH_CALUDE_coin_denominations_exist_l2292_229299

theorem coin_denominations_exist : ∃ (coins : Finset ℕ), 
  (Finset.card coins = 12) ∧ 
  (∀ n : ℕ, 1 ≤ n ∧ n ≤ 6543 → 
    ∃ (representation : Multiset ℕ), 
      (Multiset.toFinset representation ⊆ coins) ∧
      (Multiset.card representation ≤ 8) ∧
      (Multiset.sum representation = n)) :=
by sorry

end NUMINAMATH_CALUDE_coin_denominations_exist_l2292_229299


namespace NUMINAMATH_CALUDE_smallest_a_value_l2292_229223

/-- Given two quadratic equations with integer coefficients and integer roots less than -1,
    this theorem states that the smallest possible value for the constant term 'a' is 15. -/
theorem smallest_a_value (a b c : ℤ) : 
  (∃ x y : ℤ, x < -1 ∧ y < -1 ∧ x^2 + b*x + a = 0 ∧ y^2 + b*y + a = 0) →
  (∃ z w : ℤ, z < -1 ∧ w < -1 ∧ z^2 + c*z + a = 1 ∧ w^2 + c*w + a = 1) →
  (∀ a' : ℤ, (∃ b' c' : ℤ, 
    (∃ x y : ℤ, x < -1 ∧ y < -1 ∧ x^2 + b'*x + a' = 0 ∧ y^2 + b'*y + a' = 0) ∧
    (∃ z w : ℤ, z < -1 ∧ w < -1 ∧ z^2 + c'*z + a' = 1 ∧ w^2 + c'*w + a' = 1)) →
    a' ≥ 15) →
  a = 15 :=
by sorry

end NUMINAMATH_CALUDE_smallest_a_value_l2292_229223


namespace NUMINAMATH_CALUDE_absolute_value_nonnegative_l2292_229210

theorem absolute_value_nonnegative (a : ℝ) : |a| ≥ 0 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_nonnegative_l2292_229210


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l2292_229218

-- Define the universal set
def U : Set ℝ := Set.univ

-- Define set A
def A : Set ℝ := {x | ∃ y, y = Real.log (x - 1)}

-- Define set B
def B : Set ℝ := {y | ∃ x, y = Real.sqrt (x^2 + 2*x + 5)}

-- Theorem statement
theorem intersection_of_A_and_B : A ∩ B = Set.Ici 2 := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l2292_229218


namespace NUMINAMATH_CALUDE_odd_periodic_function_value_l2292_229284

-- Define an odd function f on ℝ
def isOdd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

-- Define the periodic property of f
def hasPeriod (f : ℝ → ℝ) : Prop := ∀ x, f (x + 3/2) = -f x

-- Theorem statement
theorem odd_periodic_function_value (f : ℝ → ℝ) 
  (h_odd : isOdd f) (h_period : hasPeriod f) : f (-3/2) = 0 := by
  sorry

end NUMINAMATH_CALUDE_odd_periodic_function_value_l2292_229284


namespace NUMINAMATH_CALUDE_magazine_fraction_l2292_229252

theorem magazine_fraction (initial_amount : ℚ) (grocery_fraction : ℚ) (remaining_amount : ℚ) :
  initial_amount = 600 →
  grocery_fraction = 1/5 →
  remaining_amount = 360 →
  let amount_after_groceries := initial_amount - grocery_fraction * initial_amount
  (amount_after_groceries - remaining_amount) / amount_after_groceries = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_magazine_fraction_l2292_229252


namespace NUMINAMATH_CALUDE_mat_cost_per_square_meter_l2292_229225

/-- Given a rectangular hall with specified dimensions and total expenditure for floor covering,
    calculate the cost per square meter of the mat. -/
theorem mat_cost_per_square_meter
  (length width height : ℝ)
  (total_expenditure : ℝ)
  (h_length : length = 20)
  (h_width : width = 15)
  (h_height : height = 5)
  (h_expenditure : total_expenditure = 57000) :
  total_expenditure / (length * width) = 190 := by
  sorry

end NUMINAMATH_CALUDE_mat_cost_per_square_meter_l2292_229225


namespace NUMINAMATH_CALUDE_team_leader_deputy_count_l2292_229279

def people : Nat := 5

theorem team_leader_deputy_count : 
  (people * (people - 1) : Nat) = 20 := by
  sorry

end NUMINAMATH_CALUDE_team_leader_deputy_count_l2292_229279


namespace NUMINAMATH_CALUDE_remaining_blue_fraction_after_four_changes_l2292_229245

/-- The fraction of a square's area that remains blue after one change -/
def blue_fraction_after_one_change : ℚ := 8 / 9

/-- The number of changes applied to the square -/
def num_changes : ℕ := 4

/-- The fraction of the original area that remains blue after the specified number of changes -/
def remaining_blue_fraction : ℚ := blue_fraction_after_one_change ^ num_changes

/-- Theorem stating that the remaining blue fraction after four changes is 4096/6561 -/
theorem remaining_blue_fraction_after_four_changes :
  remaining_blue_fraction = 4096 / 6561 := by
  sorry

end NUMINAMATH_CALUDE_remaining_blue_fraction_after_four_changes_l2292_229245


namespace NUMINAMATH_CALUDE_washing_loads_proof_l2292_229273

def washing_machine_capacity : ℕ := 8
def num_shirts : ℕ := 39
def num_sweaters : ℕ := 33

theorem washing_loads_proof :
  let total_clothes := num_shirts + num_sweaters
  let num_loads := (total_clothes + washing_machine_capacity - 1) / washing_machine_capacity
  num_loads = 9 := by sorry

end NUMINAMATH_CALUDE_washing_loads_proof_l2292_229273


namespace NUMINAMATH_CALUDE_adjacent_sum_divisible_by_four_l2292_229235

/-- Represents a 2006 × 2006 table filled with numbers from 1 to 2006² -/
def Table := Fin 2006 → Fin 2006 → Fin (2006^2)

/-- Checks if two positions in the table are adjacent -/
def adjacent (p q : Fin 2006 × Fin 2006) : Prop :=
  (p.1 = q.1 ∧ (p.2 = q.2 + 1 ∨ q.2 = p.2 + 1)) ∨
  (p.2 = q.2 ∧ (p.1 = q.1 + 1 ∨ q.1 = p.1 + 1)) ∨
  (p.1 = q.1 + 1 ∧ p.2 = q.2 + 1) ∨
  (q.1 = p.1 + 1 ∧ q.2 = p.2 + 1) ∨
  (p.1 = q.1 + 1 ∧ q.2 = p.2 + 1) ∨
  (q.1 = p.1 + 1 ∧ p.2 = q.2 + 1)

/-- The main theorem to be proved -/
theorem adjacent_sum_divisible_by_four (t : Table) :
  ∃ (p q : Fin 2006 × Fin 2006),
    adjacent p q ∧ (((t p.1 p.2).val + (t q.1 q.2).val + 2) % 4 = 0) := by
  sorry


end NUMINAMATH_CALUDE_adjacent_sum_divisible_by_four_l2292_229235


namespace NUMINAMATH_CALUDE_stating_acid_solution_mixing_l2292_229272

/-- 
Given an initial acid solution and a replacement acid solution,
calculate the final acid concentration after replacing a portion of the initial solution.
-/
def final_acid_concentration (initial_concentration : ℝ) (replacement_concentration : ℝ) (replaced_fraction : ℝ) : ℝ :=
  (initial_concentration * (1 - replaced_fraction) + replacement_concentration * replaced_fraction) * 100

/-- 
Theorem stating that replacing half of a 50% acid solution with a 20% acid solution 
results in a 35% acid solution.
-/
theorem acid_solution_mixing :
  final_acid_concentration 0.5 0.2 0.5 = 35 := by
sorry

#eval final_acid_concentration 0.5 0.2 0.5

end NUMINAMATH_CALUDE_stating_acid_solution_mixing_l2292_229272


namespace NUMINAMATH_CALUDE_binomial_coeff_divisible_by_two_primes_l2292_229241

theorem binomial_coeff_divisible_by_two_primes (n k : ℕ) 
  (h1 : k > 1) (h2 : k < n - 1) : 
  ∃ (p q : ℕ), Prime p ∧ Prime q ∧ p ≠ q ∧ p ∣ Nat.choose n k ∧ q ∣ Nat.choose n k :=
by sorry

end NUMINAMATH_CALUDE_binomial_coeff_divisible_by_two_primes_l2292_229241


namespace NUMINAMATH_CALUDE_max_value_expression_tightness_of_bound_l2292_229251

theorem max_value_expression (x y : ℝ) :
  (x + 3*y + 2) / Real.sqrt (2*x^2 + y^2 + 1) ≤ Real.sqrt 14 :=
sorry

theorem tightness_of_bound : 
  ∀ ε > 0, ∃ x y : ℝ, Real.sqrt 14 - (x + 3*y + 2) / Real.sqrt (2*x^2 + y^2 + 1) < ε :=
sorry

end NUMINAMATH_CALUDE_max_value_expression_tightness_of_bound_l2292_229251


namespace NUMINAMATH_CALUDE_choir_members_l2292_229294

theorem choir_members (n : ℕ) (h1 : 50 ≤ n) (h2 : n ≤ 200) 
  (h3 : n % 7 = 4) (h4 : n % 6 = 8) : 
  n = 60 ∨ n = 102 ∨ n = 144 ∨ n = 186 := by
sorry

end NUMINAMATH_CALUDE_choir_members_l2292_229294


namespace NUMINAMATH_CALUDE_work_completion_time_l2292_229206

/-- Represents the time taken to complete a work -/
structure WorkTime where
  days : ℝ
  hours_per_day : ℝ := 24
  total_hours : ℝ := days * hours_per_day

/-- Represents the rate of work -/
def WorkRate := ℝ

/-- The problem setup -/
structure WorkProblem where
  total_work : ℝ
  a_alone_time : WorkTime
  ab_initial_time : WorkTime
  a_final_time : WorkTime
  ab_together_time : WorkTime

/-- The main theorem to prove -/
theorem work_completion_time 
  (w : WorkProblem)
  (h1 : w.a_alone_time.days = 20)
  (h2 : w.ab_initial_time.days = 10)
  (h3 : w.a_final_time.days = 15)
  (h4 : w.total_work = (w.ab_initial_time.days / w.ab_together_time.days + 
                        w.a_final_time.days / w.a_alone_time.days) * w.total_work) :
  w.ab_together_time.days = 40 := by
  sorry

end NUMINAMATH_CALUDE_work_completion_time_l2292_229206


namespace NUMINAMATH_CALUDE_randy_trip_length_l2292_229204

theorem randy_trip_length :
  ∀ (total_length : ℚ),
    (1 / 4 : ℚ) * total_length +  -- gravel road
    30 +                          -- pavement
    (1 / 8 : ℚ) * total_length +  -- scenic route
    (1 / 6 : ℚ) * total_length    -- dirt road
    = total_length →
    total_length = 720 / 11 := by
  sorry

end NUMINAMATH_CALUDE_randy_trip_length_l2292_229204


namespace NUMINAMATH_CALUDE_rectangle_length_l2292_229226

/-- Proves that the length of a rectangle is 16 centimeters, given specific conditions. -/
theorem rectangle_length (square_side : ℝ) (rect_width : ℝ) : 
  square_side = 8 →
  rect_width = 4 →
  square_side * square_side = rect_width * (16 : ℝ) :=
by
  sorry

#check rectangle_length

end NUMINAMATH_CALUDE_rectangle_length_l2292_229226


namespace NUMINAMATH_CALUDE_A_intersect_B_equals_open_interval_l2292_229291

-- Define the sets A and B
def A : Set ℝ := {x : ℝ | x * (x - 2) < 0}
def B : Set ℝ := {x : ℝ | x - 1 > 0}

-- State the theorem
theorem A_intersect_B_equals_open_interval : A ∩ B = Set.Ioo 1 2 := by sorry

end NUMINAMATH_CALUDE_A_intersect_B_equals_open_interval_l2292_229291


namespace NUMINAMATH_CALUDE_divisibility_by_7_implies_37_l2292_229254

/-- Given a natural number n, returns the number consisting of n repeated digits of 1 -/
def repeatedOnes (n : ℕ) : ℕ := 
  (10^n - 1) / 9

/-- Theorem: If a number consisting of n repeated digits of 1 is divisible by 7, 
    then it is also divisible by 37 -/
theorem divisibility_by_7_implies_37 (n : ℕ) :
  (repeatedOnes n) % 7 = 0 → (repeatedOnes n) % 37 = 0 := by
  sorry

end NUMINAMATH_CALUDE_divisibility_by_7_implies_37_l2292_229254


namespace NUMINAMATH_CALUDE_spacy_subsets_count_l2292_229249

/-- A set of integers is spacy if it contains no more than one out of any four consecutive integers. -/
def IsSpacy (s : Set ℕ) : Prop :=
  ∀ n : ℕ, (n ∈ s → (n + 1 ∉ s ∧ n + 2 ∉ s ∧ n + 3 ∉ s))

/-- The number of spacy subsets of {1, 2, ..., n} -/
def NumSpacySubsets (n : ℕ) : ℕ :=
  if n ≤ 4 then
    n + 1
  else
    NumSpacySubsets (n - 1) + NumSpacySubsets (n - 4)

theorem spacy_subsets_count :
  NumSpacySubsets 15 = 181 :=
by sorry

end NUMINAMATH_CALUDE_spacy_subsets_count_l2292_229249


namespace NUMINAMATH_CALUDE_high_jump_probabilities_l2292_229243

/-- Probability of success for athlete A -/
def pA : ℝ := 0.7

/-- Probability of success for athlete B -/
def pB : ℝ := 0.6

/-- Probability that athlete A succeeds on the third attempt for the first time -/
def prob_A_third : ℝ := (1 - pA) * (1 - pA) * pA

/-- Probability that at least one athlete succeeds in their first attempt -/
def prob_at_least_one : ℝ := 1 - (1 - pA) * (1 - pB)

/-- Probability that after two attempts each, A has exactly one more successful attempt than B -/
def prob_A_one_more : ℝ := 
  2 * pA * (1 - pA) * (1 - pB) * (1 - pB) + 
  pA * pA * 2 * pB * (1 - pB)

theorem high_jump_probabilities :
  prob_A_third = 0.063 ∧ 
  prob_at_least_one = 0.88 ∧ 
  prob_A_one_more = 0.3024 := by
  sorry

end NUMINAMATH_CALUDE_high_jump_probabilities_l2292_229243


namespace NUMINAMATH_CALUDE_winProbA_le_two_p_squared_l2292_229217

/-- A tennis game between players A and B where A wins a point with probability p ≤ 1/2 -/
structure TennisGame where
  /-- The probability of player A winning a point -/
  p : ℝ
  /-- The condition that p is at most 1/2 -/
  h_p_le_half : p ≤ 1/2

/-- The probability of player A winning the entire game -/
def winProbA (game : TennisGame) : ℝ :=
  sorry

/-- Theorem stating that the probability of A winning is at most 2p² -/
theorem winProbA_le_two_p_squared (game : TennisGame) :
  winProbA game ≤ 2 * game.p^2 := by
  sorry

end NUMINAMATH_CALUDE_winProbA_le_two_p_squared_l2292_229217


namespace NUMINAMATH_CALUDE_school_classes_l2292_229230

theorem school_classes (s : ℕ) (h1 : s > 0) : 
  ∃ c : ℕ, c * s * (7 * 12) = 84 ∧ c = 1 :=
by
  sorry

end NUMINAMATH_CALUDE_school_classes_l2292_229230


namespace NUMINAMATH_CALUDE_sin_plus_cos_value_l2292_229269

theorem sin_plus_cos_value (x : Real) 
  (h1 : 0 < x ∧ x < Real.pi / 2) 
  (h2 : Real.tan (x - Real.pi / 4) = -1 / 7) : 
  Real.sin x + Real.cos x = 7 / 5 := by
  sorry

end NUMINAMATH_CALUDE_sin_plus_cos_value_l2292_229269


namespace NUMINAMATH_CALUDE_hyperbola_equation_l2292_229260

-- Define the circle C
def circle_C (x y : ℝ) : Prop := x^2 + y^2 - 6*x - 4*y + 8 = 0

-- Define the hyperbola with focus at (2, 0) and vertex at (4, 0)
def hyperbola_focus : ℝ × ℝ := (2, 0)
def hyperbola_vertex : ℝ × ℝ := (4, 0)

-- Theorem statement
theorem hyperbola_equation : 
  ∀ x y : ℝ, 
  circle_C x y →
  (hyperbola_focus = (2, 0) ∧ hyperbola_vertex = (4, 0)) →
  x^2/4 - y^2/12 = 1 :=
sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l2292_229260


namespace NUMINAMATH_CALUDE_trajectory_is_line_segment_l2292_229213

/-- The set of points P satisfying |PF₁| + |PF₂| = 10, where F₁ and F₂ are fixed points, forms a line segment. -/
theorem trajectory_is_line_segment (F₁ F₂ : ℝ × ℝ) (h₁ : F₁ = (-5, 0)) (h₂ : F₂ = (5, 0)) :
  {P : ℝ × ℝ | dist P F₁ + dist P F₂ = 10} = {P : ℝ × ℝ | ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ P = (1 - t) • F₁ + t • F₂} :=
sorry

end NUMINAMATH_CALUDE_trajectory_is_line_segment_l2292_229213


namespace NUMINAMATH_CALUDE_average_after_removing_two_numbers_l2292_229214

theorem average_after_removing_two_numbers
  (n : ℕ) (initial_avg : ℚ) (removed1 removed2 : ℚ) (final_avg : ℚ)
  (h1 : n = 50)
  (h2 : initial_avg = 38)
  (h3 : removed1 = 45)
  (h4 : removed2 = 55)
  (h5 : final_avg = 37.5) :
  initial_avg * n - (removed1 + removed2) = final_avg * (n - 2) :=
by sorry

end NUMINAMATH_CALUDE_average_after_removing_two_numbers_l2292_229214


namespace NUMINAMATH_CALUDE_cube_root_unity_sum_l2292_229219

theorem cube_root_unity_sum (w : ℂ) 
  (h1 : w^3 - 1 = 0) 
  (h2 : w^2 + w + 1 ≠ 0) : 
  w^105 + w^106 + w^107 + w^108 + w^109 + w^110 = -2 := by
  sorry

end NUMINAMATH_CALUDE_cube_root_unity_sum_l2292_229219


namespace NUMINAMATH_CALUDE_cos_2alpha_minus_2beta_l2292_229275

theorem cos_2alpha_minus_2beta (α β : ℝ) 
  (h1 : Real.sin (α + β) = 2/3)
  (h2 : Real.sin α * Real.cos β = 1/2) : 
  Real.cos (2*α - 2*β) = 7/9 := by
  sorry

end NUMINAMATH_CALUDE_cos_2alpha_minus_2beta_l2292_229275


namespace NUMINAMATH_CALUDE_product_remainder_by_ten_l2292_229268

theorem product_remainder_by_ten (a b c : ℕ) (ha : a = 3251) (hb : b = 7462) (hc : c = 93419) :
  (a * b * c) % 10 = 8 := by
  sorry

end NUMINAMATH_CALUDE_product_remainder_by_ten_l2292_229268


namespace NUMINAMATH_CALUDE_geometric_mean_of_4_and_9_l2292_229215

/-- Given line segments a and b, x is their geometric mean if x^2 = ab -/
def is_geometric_mean (a b x : ℝ) : Prop := x^2 = a * b

/-- Proof that for line segments a = 4 and b = 9, their geometric mean x equals 6 -/
theorem geometric_mean_of_4_and_9 :
  ∀ x : ℝ, is_geometric_mean 4 9 x → x = 6 := by
  sorry

end NUMINAMATH_CALUDE_geometric_mean_of_4_and_9_l2292_229215


namespace NUMINAMATH_CALUDE_student_tickets_sold_l2292_229239

theorem student_tickets_sold (student_price non_student_price total_tickets total_revenue : ℕ)
  (h1 : student_price = 9)
  (h2 : non_student_price = 11)
  (h3 : total_tickets = 2000)
  (h4 : total_revenue = 20960) :
  ∃ student_tickets : ℕ,
    student_tickets * student_price + (total_tickets - student_tickets) * non_student_price = total_revenue ∧
    student_tickets = 520 :=
by sorry

end NUMINAMATH_CALUDE_student_tickets_sold_l2292_229239


namespace NUMINAMATH_CALUDE_decimal_to_fraction_sum_l2292_229258

theorem decimal_to_fraction_sum (m n : ℕ+) : 
  (m : ℚ) / (n : ℚ) = 1824 / 10000 → 
  ∀ (a b : ℕ+), (a : ℚ) / (b : ℚ) = 1824 / 10000 → 
  (a : ℕ) ≤ (m : ℕ) ∧ (b : ℕ) ≤ (n : ℕ) →
  m + n = 739 := by
sorry


end NUMINAMATH_CALUDE_decimal_to_fraction_sum_l2292_229258


namespace NUMINAMATH_CALUDE_charity_ticket_revenue_l2292_229270

theorem charity_ticket_revenue 
  (total_tickets : ℕ) 
  (total_revenue : ℕ) 
  (full_price : ℕ) 
  (full_price_tickets : ℕ) 
  (half_price_tickets : ℕ) :
  total_tickets = 140 →
  total_revenue = 2001 →
  total_tickets = full_price_tickets + half_price_tickets →
  total_revenue = full_price * full_price_tickets + (full_price / 2) * half_price_tickets →
  full_price > 0 →
  full_price_tickets * full_price = 782 :=
by sorry

end NUMINAMATH_CALUDE_charity_ticket_revenue_l2292_229270


namespace NUMINAMATH_CALUDE_principal_amount_proof_l2292_229240

/-- Proves that given the conditions of the problem, the principal amount is 1500 --/
theorem principal_amount_proof (P : ℝ) : 
  (P * 0.04 * 4 = P - 1260) → P = 1500 := by
  sorry

end NUMINAMATH_CALUDE_principal_amount_proof_l2292_229240


namespace NUMINAMATH_CALUDE_bridge_length_calculation_l2292_229259

/-- Calculates the length of a bridge given train and crossing parameters. -/
theorem bridge_length_calculation (train_length : ℝ) (train_speed : ℝ) (crossing_time : ℝ) :
  train_length = 280 →
  train_speed = 18 →
  crossing_time = 20 →
  train_speed * crossing_time - train_length = 80 := by
  sorry

#check bridge_length_calculation

end NUMINAMATH_CALUDE_bridge_length_calculation_l2292_229259


namespace NUMINAMATH_CALUDE_family_celebration_attendees_l2292_229203

theorem family_celebration_attendees :
  ∀ (n : ℕ) (s : ℕ),
    s / n = n →
    (s - 29) / (n - 1) = n - 1 →
    n = 15 := by
  sorry

end NUMINAMATH_CALUDE_family_celebration_attendees_l2292_229203


namespace NUMINAMATH_CALUDE_floor_equation_solution_l2292_229216

theorem floor_equation_solution (x : ℝ) : x - Int.floor (x / 2016) = 2016 ↔ x = 2017 := by
  sorry

end NUMINAMATH_CALUDE_floor_equation_solution_l2292_229216


namespace NUMINAMATH_CALUDE_total_coins_is_188_l2292_229255

/-- The number of US pennies turned in -/
def us_pennies : ℕ := 38

/-- The number of US nickels turned in -/
def us_nickels : ℕ := 27

/-- The number of US dimes turned in -/
def us_dimes : ℕ := 19

/-- The number of US quarters turned in -/
def us_quarters : ℕ := 24

/-- The number of US half-dollars turned in -/
def us_half_dollars : ℕ := 13

/-- The number of US one-dollar coins turned in -/
def us_one_dollar_coins : ℕ := 17

/-- The number of US two-dollar coins turned in -/
def us_two_dollar_coins : ℕ := 5

/-- The number of Australian fifty-cent coins turned in -/
def australian_fifty_cent_coins : ℕ := 4

/-- The number of Mexican one-Peso coins turned in -/
def mexican_one_peso_coins : ℕ := 12

/-- The number of Canadian loonies turned in -/
def canadian_loonies : ℕ := 3

/-- The number of British 20 pence coins turned in -/
def british_20_pence_coins : ℕ := 7

/-- The number of pre-1965 US dimes turned in -/
def pre_1965_us_dimes : ℕ := 6

/-- The number of post-2005 Euro two-euro coins turned in -/
def euro_two_euro_coins : ℕ := 5

/-- The number of Swiss 5 franc coins turned in -/
def swiss_5_franc_coins : ℕ := 8

/-- Theorem: The total number of coins turned in is 188 -/
theorem total_coins_is_188 :
  us_pennies + us_nickels + us_dimes + us_quarters + us_half_dollars +
  us_one_dollar_coins + us_two_dollar_coins + australian_fifty_cent_coins +
  mexican_one_peso_coins + canadian_loonies + british_20_pence_coins +
  pre_1965_us_dimes + euro_two_euro_coins + swiss_5_franc_coins = 188 := by
  sorry

end NUMINAMATH_CALUDE_total_coins_is_188_l2292_229255
