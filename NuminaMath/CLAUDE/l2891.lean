import Mathlib

namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l2891_289144

theorem geometric_sequence_sum (a r : ℚ) (n : ℕ) (h1 : a = 1/2) (h2 : r = 1/3) (h3 : n = 8) :
  (a * (1 - r^n)) / (1 - r) = 9840/6561 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l2891_289144


namespace NUMINAMATH_CALUDE_product_sequence_sum_l2891_289196

theorem product_sequence_sum (a b : ℕ) (h1 : a / 3 = 18) (h2 : b = a - 1) : a + b = 107 := by
  sorry

end NUMINAMATH_CALUDE_product_sequence_sum_l2891_289196


namespace NUMINAMATH_CALUDE_total_children_l2891_289156

theorem total_children (happy : ℕ) (sad : ℕ) (neutral : ℕ) 
  (boys : ℕ) (girls : ℕ) (happy_boys : ℕ) (sad_girls : ℕ) (neutral_boys : ℕ) :
  happy = 30 →
  sad = 10 →
  neutral = 20 →
  boys = 19 →
  girls = 41 →
  happy_boys = 6 →
  sad_girls = 4 →
  neutral_boys = 7 →
  boys + girls = 60 :=
by sorry

end NUMINAMATH_CALUDE_total_children_l2891_289156


namespace NUMINAMATH_CALUDE_log_inequality_l2891_289132

theorem log_inequality (a b c : ℝ) : 
  a = Real.log (2/3) → b = Real.log (2/5) → c = Real.log (3/2) → c > a ∧ a > b :=
by sorry

end NUMINAMATH_CALUDE_log_inequality_l2891_289132


namespace NUMINAMATH_CALUDE_unique_perfect_square_solution_l2891_289166

theorem unique_perfect_square_solution (n : ℤ) : 
  (∃ k : ℤ, n^4 + 6*n^3 + 11*n^2 + 3*n + 31 = k^2) ↔ n = 10 := by
sorry

end NUMINAMATH_CALUDE_unique_perfect_square_solution_l2891_289166


namespace NUMINAMATH_CALUDE_fifth_month_sale_l2891_289189

def sales_first_four : List ℕ := [6235, 6927, 6855, 7230]
def required_sixth : ℕ := 5191
def desired_average : ℕ := 6500
def num_months : ℕ := 6

theorem fifth_month_sale :
  let total_required : ℕ := desired_average * num_months
  let sum_known : ℕ := (sales_first_four.sum + required_sixth)
  let fifth_month : ℕ := total_required - sum_known
  fifth_month = 6562 := by sorry

end NUMINAMATH_CALUDE_fifth_month_sale_l2891_289189


namespace NUMINAMATH_CALUDE_boys_camp_total_l2891_289176

theorem boys_camp_total (total : ℕ) 
  (h1 : (total : ℚ) * (1 / 5) = (total : ℚ) * (20 / 100))
  (h2 : (total : ℚ) * (1 / 5) * (3 / 10) = (total : ℚ) * (1 / 5) * (30 / 100))
  (h3 : (total : ℚ) * (1 / 5) * (7 / 10) = 56) :
  total = 400 := by
sorry

end NUMINAMATH_CALUDE_boys_camp_total_l2891_289176


namespace NUMINAMATH_CALUDE_original_class_size_l2891_289145

theorem original_class_size (N : ℕ) : 
  (N > 0) →
  (40 * N + 8 * 32 = 36 * (N + 8)) →
  N = 8 := by
sorry

end NUMINAMATH_CALUDE_original_class_size_l2891_289145


namespace NUMINAMATH_CALUDE_complex_fraction_calculation_l2891_289158

theorem complex_fraction_calculation : 
  (7/4 - 7/8 - 7/12) / (-7/8) + (-7/8) / (7/4 - 7/8 - 7/12) = -10/3 := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_calculation_l2891_289158


namespace NUMINAMATH_CALUDE_third_draw_probability_l2891_289130

/-- Represents the number of balls of each color in the box -/
structure BallCount where
  white : ℕ
  black : ℕ

/-- Calculates the probability of drawing a white ball -/
def probWhite (balls : BallCount) : ℚ :=
  balls.white / (balls.white + balls.black)

theorem third_draw_probability :
  let initial := BallCount.mk 8 7
  let after_removal := BallCount.mk (initial.white - 1) (initial.black - 1)
  probWhite after_removal = 7 / 13 := by
  sorry

end NUMINAMATH_CALUDE_third_draw_probability_l2891_289130


namespace NUMINAMATH_CALUDE_arithmetic_calculation_l2891_289180

theorem arithmetic_calculation : 
  (1.5 * ((3.6 * 0.48 * 2.50) / (0.12 * 0.09 * 0.5)) : ℝ) = 1200 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_calculation_l2891_289180


namespace NUMINAMATH_CALUDE_quadratic_inequality_problem_l2891_289177

theorem quadratic_inequality_problem (a c m : ℝ) :
  (∀ x, ax^2 + x + c > 0 ↔ 1 < x ∧ x < 3) →
  let A := {x | (-1/4)*x^2 + 2*x - 3 > 0}
  let B := {x | x + m > 0}
  A ⊆ B →
  (a = -1/4 ∧ c = -3/4) ∧ m ≥ -2 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_problem_l2891_289177


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l2891_289146

theorem sqrt_equation_solution (x : ℝ) : Real.sqrt (3 + Real.sqrt x) = 4 → x = 169 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l2891_289146


namespace NUMINAMATH_CALUDE_paige_albums_l2891_289111

def number_of_albums (total_pictures : ℕ) (first_album_pictures : ℕ) (pictures_per_album : ℕ) : ℕ :=
  (total_pictures - first_album_pictures) / pictures_per_album

theorem paige_albums : number_of_albums 35 14 7 = 3 := by
  sorry

end NUMINAMATH_CALUDE_paige_albums_l2891_289111


namespace NUMINAMATH_CALUDE_reciprocal_sum_equality_implies_zero_product_l2891_289109

theorem reciprocal_sum_equality_implies_zero_product
  (a b c : ℝ) (h : a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0) 
  (eq : 1/a + 1/b + 1/c = 1/(a+b+c)) :
  (a+b)*(b+c)*(a+c) = 0 := by
sorry

end NUMINAMATH_CALUDE_reciprocal_sum_equality_implies_zero_product_l2891_289109


namespace NUMINAMATH_CALUDE_cube_root_simplification_l2891_289187

theorem cube_root_simplification (ω : ℂ) :
  ω ≠ 1 →
  ω^3 = 1 →
  (1 - 2*ω + 3*ω^2)^3 + (2 + 3*ω - 4*ω^2)^3 = -83 := by sorry

end NUMINAMATH_CALUDE_cube_root_simplification_l2891_289187


namespace NUMINAMATH_CALUDE_line_through_ellipse_midpoint_l2891_289140

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a line in 2D space -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Checks if a point lies on the given ellipse -/
def isOnEllipse (p : Point) : Prop :=
  p.x^2 / 25 + p.y^2 / 16 = 1

/-- Checks if a point lies on the given line -/
def isOnLine (p : Point) (l : Line) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

/-- Checks if a point is the midpoint of two other points -/
def isMidpoint (m : Point) (a : Point) (b : Point) : Prop :=
  m.x = (a.x + b.x) / 2 ∧ m.y = (a.y + b.y) / 2

theorem line_through_ellipse_midpoint (M A B : Point) (l : Line) :
  isOnLine M l →
  isOnEllipse A →
  isOnEllipse B →
  isOnLine A l →
  isOnLine B l →
  isMidpoint M A B →
  M.x = 1 →
  M.y = 2 →
  l.a = 8 ∧ l.b = 25 ∧ l.c = -58 := by
  sorry


end NUMINAMATH_CALUDE_line_through_ellipse_midpoint_l2891_289140


namespace NUMINAMATH_CALUDE_tetrahedron_fits_in_box_l2891_289106

theorem tetrahedron_fits_in_box :
  ∀ (tetra_edge box_length box_width box_height : ℝ),
    tetra_edge = 12 →
    box_length = 9 ∧ box_width = 13 ∧ box_height = 15 →
    ∃ (cube_edge : ℝ),
      cube_edge = tetra_edge / Real.sqrt 2 ∧
      cube_edge ≤ box_length ∧
      cube_edge ≤ box_width ∧
      cube_edge ≤ box_height :=
by sorry

end NUMINAMATH_CALUDE_tetrahedron_fits_in_box_l2891_289106


namespace NUMINAMATH_CALUDE_multiply_decimals_l2891_289135

theorem multiply_decimals : (0.25 : ℝ) * 0.08 = 0.02 := by
  sorry

end NUMINAMATH_CALUDE_multiply_decimals_l2891_289135


namespace NUMINAMATH_CALUDE_isosceles_triangle_perimeter_isosceles_triangle_perimeter_proof_l2891_289164

/-- An isosceles triangle with two sides of length 6 and one side of length 3 has a perimeter of 15 -/
theorem isosceles_triangle_perimeter : ℝ → Prop :=
  fun perimeter =>
    ∃ (a b : ℝ),
      a = 6 ∧
      b = 3 ∧
      (a = a ∧ b ≤ a + a) ∧  -- Triangle inequality
      perimeter = a + a + b ∧
      perimeter = 15

/-- Proof of the theorem -/
theorem isosceles_triangle_perimeter_proof : isosceles_triangle_perimeter 15 := by
  sorry

#check isosceles_triangle_perimeter
#check isosceles_triangle_perimeter_proof

end NUMINAMATH_CALUDE_isosceles_triangle_perimeter_isosceles_triangle_perimeter_proof_l2891_289164


namespace NUMINAMATH_CALUDE_q_subset_p_intersect_q_iff_a_in_range_l2891_289185

-- Define sets P and Q
def P : Set ℝ := {x | 3 < x ∧ x ≤ 22}
def Q (a : ℝ) : Set ℝ := {x | 2 * a + 1 ≤ x ∧ x < 3 * a - 5}

-- State the theorem
theorem q_subset_p_intersect_q_iff_a_in_range :
  ∀ a : ℝ, (Q a).Nonempty → (Q a ⊆ (P ∩ Q a) ↔ 6 < a ∧ a ≤ 9) := by
  sorry

end NUMINAMATH_CALUDE_q_subset_p_intersect_q_iff_a_in_range_l2891_289185


namespace NUMINAMATH_CALUDE_quadratic_real_roots_alpha_range_l2891_289105

theorem quadratic_real_roots_alpha_range :
  ∀ α : ℝ, 
  (∃ x : ℝ, x^2 - 2*x + α = 0) →
  α ≤ 1 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_real_roots_alpha_range_l2891_289105


namespace NUMINAMATH_CALUDE_shaded_region_circle_diameter_l2891_289155

/-- Given two concentric circles with radii 24 and 36 units, the diameter of a new circle
    whose diameter is equal to the area of the shaded region between the two circles
    is 720π units. -/
theorem shaded_region_circle_diameter :
  let r₁ : ℝ := 24
  let r₂ : ℝ := 36
  let shaded_area := π * (r₂^2 - r₁^2)
  let new_circle_diameter := shaded_area
  new_circle_diameter = 720 * π :=
by sorry

end NUMINAMATH_CALUDE_shaded_region_circle_diameter_l2891_289155


namespace NUMINAMATH_CALUDE_fraction_value_l2891_289193

theorem fraction_value (p q : ℚ) (h : p / q = 4 / 5) :
  ∃ x : ℚ, x + (2 * q - p) / (2 * q + p) = 2 ∧ x = 11 / 7 := by
  sorry

end NUMINAMATH_CALUDE_fraction_value_l2891_289193


namespace NUMINAMATH_CALUDE_sin_cos_difference_simplification_l2891_289114

theorem sin_cos_difference_simplification :
  Real.sin (72 * π / 180) * Real.cos (12 * π / 180) - 
  Real.cos (72 * π / 180) * Real.sin (12 * π / 180) = 
  Real.sqrt 3 / 2 := by sorry

end NUMINAMATH_CALUDE_sin_cos_difference_simplification_l2891_289114


namespace NUMINAMATH_CALUDE_garden_ratio_l2891_289186

/-- A rectangular garden with given perimeter and length has a specific length-to-width ratio -/
theorem garden_ratio (perimeter : ℝ) (length : ℝ) (width : ℝ) 
  (h_perimeter : perimeter = 900)
  (h_length : length = 300)
  (h_rectangle : perimeter = 2 * (length + width)) : 
  length / width = 2 := by
  sorry

end NUMINAMATH_CALUDE_garden_ratio_l2891_289186


namespace NUMINAMATH_CALUDE_g_13_equals_201_l2891_289112

def g (n : ℕ) : ℕ := n^2 + n + 19

theorem g_13_equals_201 : g 13 = 201 := by sorry

end NUMINAMATH_CALUDE_g_13_equals_201_l2891_289112


namespace NUMINAMATH_CALUDE_leila_payment_l2891_289183

/-- The total cost Leila should pay Ali for the cakes -/
def total_cost (chocolate_quantity : ℕ) (chocolate_price : ℕ) 
               (strawberry_quantity : ℕ) (strawberry_price : ℕ) : ℕ :=
  chocolate_quantity * chocolate_price + strawberry_quantity * strawberry_price

/-- Theorem stating that Leila should pay Ali $168 for the cakes -/
theorem leila_payment : total_cost 3 12 6 22 = 168 := by
  sorry

end NUMINAMATH_CALUDE_leila_payment_l2891_289183


namespace NUMINAMATH_CALUDE_lucy_money_problem_l2891_289128

theorem lucy_money_problem (x : ℝ) : 
  let doubled := 2 * x
  let after_giving := doubled * (4/5)
  let after_losing := after_giving * (2/3)
  let after_spending := after_losing * (3/4)
  after_spending = 15 → x = 18.75 := by sorry

end NUMINAMATH_CALUDE_lucy_money_problem_l2891_289128


namespace NUMINAMATH_CALUDE_jog_time_proportional_l2891_289157

/-- Given a constant jogging pace, prove that if 3 miles takes 30 minutes, then 1.5 miles takes 15 minutes. -/
theorem jog_time_proportional (pace : ℝ → ℝ) (h_constant : ∀ x y, pace x = pace y) :
  pace 3 = 30 → pace 1.5 = 15 := by
  sorry

end NUMINAMATH_CALUDE_jog_time_proportional_l2891_289157


namespace NUMINAMATH_CALUDE_number_transformation_l2891_289174

theorem number_transformation (x : ℝ) : 2 * ((2 * (x + 1)) - 1) = 2 * x + 2 := by
  sorry

#check number_transformation

end NUMINAMATH_CALUDE_number_transformation_l2891_289174


namespace NUMINAMATH_CALUDE_smallest_norm_given_condition_l2891_289133

/-- Given a vector v in ℝ², prove that the smallest possible value of its norm,
    given that the norm of v + (4, 2) is 10, is 10 - 2√5. -/
theorem smallest_norm_given_condition (v : ℝ × ℝ) 
  (h : ‖v + (4, 2)‖ = 10) : 
  ∃ (w : ℝ × ℝ), ‖w‖ = 10 - 2 * Real.sqrt 5 ∧ 
  ∀ (u : ℝ × ℝ), ‖u + (4, 2)‖ = 10 → ‖w‖ ≤ ‖u‖ :=
sorry

end NUMINAMATH_CALUDE_smallest_norm_given_condition_l2891_289133


namespace NUMINAMATH_CALUDE_negation_equivalence_l2891_289118

theorem negation_equivalence : 
  (¬ ∃ x : ℝ, x^2 + x + 1 ≤ 0) ↔ (∀ x : ℝ, x^2 + x + 1 > 0) := by
sorry

end NUMINAMATH_CALUDE_negation_equivalence_l2891_289118


namespace NUMINAMATH_CALUDE_minute_hand_distance_l2891_289136

theorem minute_hand_distance (hand_length : ℝ) (time : ℝ) : 
  hand_length = 8 → time = 45 → 
  2 * π * hand_length * (time / 60) = 12 * π := by
sorry

end NUMINAMATH_CALUDE_minute_hand_distance_l2891_289136


namespace NUMINAMATH_CALUDE_sum_of_three_numbers_l2891_289113

theorem sum_of_three_numbers (x y z : ℝ) 
  (sum_xy : x + y = 31)
  (sum_yz : y + z = 41)
  (sum_zx : z + x = 55) :
  x + y + z = 63.5 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_three_numbers_l2891_289113


namespace NUMINAMATH_CALUDE_infinite_solutions_iff_b_eq_neg_twelve_l2891_289101

/-- The equation 4(3x-b) = 3(4x + 16) has infinitely many solutions x if and only if b = -12 -/
theorem infinite_solutions_iff_b_eq_neg_twelve (b : ℝ) : 
  (∀ x : ℝ, 4 * (3 * x - b) = 3 * (4 * x + 16)) ↔ b = -12 := by
  sorry

end NUMINAMATH_CALUDE_infinite_solutions_iff_b_eq_neg_twelve_l2891_289101


namespace NUMINAMATH_CALUDE_smallest_five_digit_multiple_correct_l2891_289143

/-- The smallest positive five-digit number divisible by 2, 3, 5, 7, and 11 -/
def smallest_five_digit_multiple : ℕ := 11550

/-- The five smallest prime numbers -/
def smallest_primes : List ℕ := [2, 3, 5, 7, 11]

theorem smallest_five_digit_multiple_correct :
  (∀ p ∈ smallest_primes, smallest_five_digit_multiple % p = 0) ∧
  smallest_five_digit_multiple ≥ 10000 ∧
  smallest_five_digit_multiple < 100000 ∧
  (∀ n : ℕ, n < smallest_five_digit_multiple →
    n < 10000 ∨ (∃ p ∈ smallest_primes, n % p ≠ 0)) :=
by sorry

end NUMINAMATH_CALUDE_smallest_five_digit_multiple_correct_l2891_289143


namespace NUMINAMATH_CALUDE_quadratic_inequality_l2891_289139

theorem quadratic_inequality (x : ℝ) : x^2 - 48*x + 576 ≤ 16 ↔ 20 ≤ x ∧ x ≤ 28 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_l2891_289139


namespace NUMINAMATH_CALUDE_x_twelve_equals_negative_one_l2891_289100

theorem x_twelve_equals_negative_one (x : ℝ) (h : x + 1/x = Real.sqrt 2) : x^12 = -1 := by
  sorry

end NUMINAMATH_CALUDE_x_twelve_equals_negative_one_l2891_289100


namespace NUMINAMATH_CALUDE_sqrt_sum_squares_l2891_289172

theorem sqrt_sum_squares : Real.sqrt (2^4 + 2^4 + 4^2) = 4 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_sum_squares_l2891_289172


namespace NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l2891_289161

theorem arithmetic_sequence_common_difference 
  (a : ℕ → ℝ) 
  (h_arithmetic : ∀ n : ℕ, a (n + 1) - a n = a (n + 2) - a (n + 1)) 
  (h_first_term : a 1 = 1) 
  (h_sum : a 1 + a 2 + a 3 = 12) : 
  a 2 - a 1 = 3 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l2891_289161


namespace NUMINAMATH_CALUDE_kids_left_playing_l2891_289179

-- Define the initial number of kids playing soccer
def initial_kids : Real := 22.5

-- Define the number of kids who decided to go home
def kids_leaving : Real := 14.3

-- Define the number of kids left playing
def kids_left : Real := initial_kids - kids_leaving

-- Theorem to prove
theorem kids_left_playing :
  kids_left = 8.2 := by sorry

end NUMINAMATH_CALUDE_kids_left_playing_l2891_289179


namespace NUMINAMATH_CALUDE_lotus_flower_problem_l2891_289117

theorem lotus_flower_problem (x : ℚ) : 
  (x / 3 + x / 5 + x / 6 + x / 4 + 6 = x) → x = 120 := by
  sorry

end NUMINAMATH_CALUDE_lotus_flower_problem_l2891_289117


namespace NUMINAMATH_CALUDE_equation_solutions_l2891_289137

theorem equation_solutions : 
  let f (x : ℝ) := (x - 2) * (x - 3) * (x - 5) * (x - 6) * (x - 3) * (x - 5) * (x - 2)
  let g (x : ℝ) := (x - 3) * (x - 5) * (x - 3)
  let h (x : ℝ) := f x / g x
  ∀ x : ℝ, h x = 1 ↔ x = (-3 + Real.sqrt 5) / 2 ∨ x = (-3 - Real.sqrt 5) / 2 :=
by
  sorry


end NUMINAMATH_CALUDE_equation_solutions_l2891_289137


namespace NUMINAMATH_CALUDE_line_points_relationship_l2891_289141

theorem line_points_relationship (m a b : ℝ) :
  ((-2 : ℝ), a) ∈ {(x, y) | y = -2*x + m} →
  ((2 : ℝ), b) ∈ {(x, y) | y = -2*x + m} →
  a > b := by
  sorry

end NUMINAMATH_CALUDE_line_points_relationship_l2891_289141


namespace NUMINAMATH_CALUDE_max_weight_for_john_and_mike_l2891_289173

/-- The maximum weight the bench can support -/
def bench_max_weight : ℝ := 1000

/-- The safety margin for one person -/
def safety_margin_one : ℝ := 0.2

/-- The safety margin for two people -/
def safety_margin_two : ℝ := 0.3

/-- John's weight -/
def john_weight : ℝ := 250

/-- Mike's weight -/
def mike_weight : ℝ := 180

/-- Theorem: The maximum weight John and Mike can put on the bar when using the bench together is 270 pounds -/
theorem max_weight_for_john_and_mike : 
  bench_max_weight * (1 - safety_margin_two) - (john_weight + mike_weight) = 270 := by
  sorry

end NUMINAMATH_CALUDE_max_weight_for_john_and_mike_l2891_289173


namespace NUMINAMATH_CALUDE_product_identity_l2891_289191

theorem product_identity (x y : ℝ) : (x + y^2) * (x - y^2) * (x^2 + y^4) = x^4 - y^8 := by
  sorry

end NUMINAMATH_CALUDE_product_identity_l2891_289191


namespace NUMINAMATH_CALUDE_radian_measure_of_15_degrees_l2891_289167

theorem radian_measure_of_15_degrees :
  let degree_to_radian (d : ℝ) := d * (Real.pi / 180)
  degree_to_radian 15 = Real.pi / 12 := by
  sorry

end NUMINAMATH_CALUDE_radian_measure_of_15_degrees_l2891_289167


namespace NUMINAMATH_CALUDE_line_intersects_circle_right_angle_l2891_289190

theorem line_intersects_circle_right_angle (k : ℝ) :
  (∃ (P Q : ℝ × ℝ), 
    P.1^2 + P.2^2 = 1 ∧ 
    Q.1^2 + Q.2^2 = 1 ∧ 
    P.2 = k * P.1 + 1 ∧ 
    Q.2 = k * Q.1 + 1 ∧ 
    (P.1 * Q.1 + P.2 * Q.2 = 0)) →
  k = 1 ∨ k = -1 :=
by sorry

end NUMINAMATH_CALUDE_line_intersects_circle_right_angle_l2891_289190


namespace NUMINAMATH_CALUDE_line_equation_l2891_289149

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2/6 + y^2/3 = 1

-- Define the line l
def line_l (x y : ℝ) : Prop := x + Real.sqrt 2 * y - 2 * Real.sqrt 2 = 0

-- Define points A, B, M, and N
def point_A : ℝ × ℝ := sorry
def point_B : ℝ × ℝ := sorry
def point_M : ℝ × ℝ := sorry
def point_N : ℝ × ℝ := sorry

-- Define the conditions
def conditions : Prop :=
  ellipse point_A.1 point_A.2 ∧
  ellipse point_B.1 point_B.2 ∧
  point_A.1 > 0 ∧ point_A.2 > 0 ∧
  point_B.1 > 0 ∧ point_B.2 > 0 ∧
  point_M.2 = 0 ∧
  point_N.1 = 0 ∧
  (point_M.1 - point_A.1)^2 + (point_M.2 - point_A.2)^2 =
    (point_N.1 - point_B.1)^2 + (point_N.2 - point_B.2)^2 ∧
  (point_M.1 - point_N.1)^2 + (point_M.2 - point_N.2)^2 = 12

-- Theorem statement
theorem line_equation (h : conditions) : 
  ∀ x y, line_l x y ↔ (x, y) ∈ {p | ∃ t, p = (1-t) • point_M + t • point_N} :=
sorry

end NUMINAMATH_CALUDE_line_equation_l2891_289149


namespace NUMINAMATH_CALUDE_power_division_sum_difference_equals_sixteen_l2891_289170

theorem power_division_sum_difference_equals_sixteen :
  (5 ^ 6 / 5 ^ 4) + 3 ^ 3 - 6 ^ 2 = 16 := by
  sorry

end NUMINAMATH_CALUDE_power_division_sum_difference_equals_sixteen_l2891_289170


namespace NUMINAMATH_CALUDE_integral_f_equals_three_l2891_289103

-- Define the function to be integrated
def f (x : ℝ) : ℝ := 2 - |1 - x|

-- State the theorem
theorem integral_f_equals_three :
  ∫ x in (0 : ℝ)..2, f x = 3 := by sorry

end NUMINAMATH_CALUDE_integral_f_equals_three_l2891_289103


namespace NUMINAMATH_CALUDE_glove_profit_is_810_l2891_289123

/-- Calculates the profit from selling gloves given the purchase and sales information. -/
def glove_profit (total_pairs : ℕ) (cost_per_pair : ℚ) (sold_pairs_high : ℕ) (price_high : ℚ) (price_low : ℚ) : ℚ :=
  let remaining_pairs := total_pairs - sold_pairs_high
  let total_cost := cost_per_pair * total_pairs
  let revenue_high := price_high * sold_pairs_high
  let revenue_low := price_low * remaining_pairs
  let total_revenue := revenue_high + revenue_low
  total_revenue - total_cost

/-- The profit from selling gloves under the given conditions is 810 yuan. -/
theorem glove_profit_is_810 :
  glove_profit 600 12 470 14 11 = 810 := by
  sorry

#eval glove_profit 600 12 470 14 11

end NUMINAMATH_CALUDE_glove_profit_is_810_l2891_289123


namespace NUMINAMATH_CALUDE_arrangement_theorem_l2891_289120

/-- The number of permutations of n elements taken r at a time -/
def permutations (n : ℕ) (r : ℕ) : ℕ := sorry

/-- The number of ways to arrange 8 people in a row with 3 specific people not adjacent -/
def arrangements_with_non_adjacent (total : ℕ) (non_adjacent : ℕ) : ℕ :=
  permutations (total - non_adjacent + 1) non_adjacent * permutations (total - non_adjacent) (total - non_adjacent)

theorem arrangement_theorem :
  arrangements_with_non_adjacent 8 3 = permutations 6 3 * permutations 5 5 := by
  sorry

end NUMINAMATH_CALUDE_arrangement_theorem_l2891_289120


namespace NUMINAMATH_CALUDE_line_point_k_value_l2891_289169

/-- Given three points on a line, calculate the value of k -/
theorem line_point_k_value (k : ℝ) : 
  (∃ (m b : ℝ), 7 = m * 3 + b ∧ k = m * 5 + b ∧ 15 = m * 11 + b) → k = 9 := by
  sorry

end NUMINAMATH_CALUDE_line_point_k_value_l2891_289169


namespace NUMINAMATH_CALUDE_factorization_of_2x_squared_minus_8_l2891_289197

theorem factorization_of_2x_squared_minus_8 (x : ℝ) : 2 * x^2 - 8 = 2 * (x - 2) * (x + 2) := by
  sorry

end NUMINAMATH_CALUDE_factorization_of_2x_squared_minus_8_l2891_289197


namespace NUMINAMATH_CALUDE_inverse_75_mod_76_l2891_289125

theorem inverse_75_mod_76 : ∃ x : ℕ, x < 76 ∧ (75 * x) % 76 = 1 :=
by
  use 75
  sorry

end NUMINAMATH_CALUDE_inverse_75_mod_76_l2891_289125


namespace NUMINAMATH_CALUDE_smallest_n_is_34_l2891_289127

/-- Given a natural number n ≥ 16, this function represents the set {16, 17, ..., n} -/
def S (n : ℕ) : Set ℕ := {x | 16 ≤ x ∧ x ≤ n}

/-- This function checks if a sequence of 15 natural numbers satisfies the required conditions -/
def valid_sequence (n : ℕ) (a : Fin 15 → ℕ) : Prop :=
  (∀ i : Fin 15, a i ∈ S n) ∧
  (∀ i : Fin 15, (i.val + 1) ∣ a i) ∧
  (∀ i j : Fin 15, i ≠ j → a i ≠ a j)

/-- The main theorem stating that 34 is the smallest n satisfying the conditions -/
theorem smallest_n_is_34 :
  (∃ a : Fin 15 → ℕ, valid_sequence 34 a) ∧
  (∀ m : ℕ, m < 34 → ¬∃ a : Fin 15 → ℕ, valid_sequence m a) :=
sorry

end NUMINAMATH_CALUDE_smallest_n_is_34_l2891_289127


namespace NUMINAMATH_CALUDE_fraction_equality_l2891_289129

theorem fraction_equality (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) 
  (h : (5*x + 2*y) / (x - 5*y) = -3) : 
  (x + 5*y) / (5*x - y) = 53/57 := by
sorry

end NUMINAMATH_CALUDE_fraction_equality_l2891_289129


namespace NUMINAMATH_CALUDE_light_flash_interval_l2891_289142

/-- Given a light that flashes 180 times in ¾ of an hour, 
    prove that the time between flashes is 15 seconds. -/
theorem light_flash_interval (flashes : ℕ) (time : ℚ) 
  (h1 : flashes = 180) 
  (h2 : time = 3/4) : 
  (time * 3600) / flashes = 15 := by
  sorry

end NUMINAMATH_CALUDE_light_flash_interval_l2891_289142


namespace NUMINAMATH_CALUDE_arithmetic_sequence_general_term_l2891_289178

/-- An arithmetic sequence with specific conditions -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  (∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d) ∧ 
  a 3 = 7 ∧
  a 5 = a 2 + 6

/-- The theorem stating the general term of the arithmetic sequence -/
theorem arithmetic_sequence_general_term (a : ℕ → ℝ) 
  (h : ArithmeticSequence a) : 
  ∀ n : ℕ, a n = 2 * n + 1 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_general_term_l2891_289178


namespace NUMINAMATH_CALUDE_square_odd_digits_iff_one_or_three_l2891_289171

/-- A function that checks if a natural number consists of only odd digits -/
def hasOnlyOddDigits (n : ℕ) : Prop :=
  ∀ d : ℕ, d ∈ n.digits 10 → d % 2 = 1

/-- Theorem stating that n^2 has only odd digits if and only if n is 1 or 3 -/
theorem square_odd_digits_iff_one_or_three (n : ℕ) :
  n > 0 → (hasOnlyOddDigits (n^2) ↔ n = 1 ∨ n = 3) :=
by sorry

end NUMINAMATH_CALUDE_square_odd_digits_iff_one_or_three_l2891_289171


namespace NUMINAMATH_CALUDE_g_expression_and_minimum_l2891_289148

noncomputable def f (a x : ℝ) : ℝ := a * x^2 - 2 * x + 1

noncomputable def M (a : ℝ) : ℝ := max (f a 1) (f a 3)

noncomputable def N (a : ℝ) : ℝ := 1 - 1/a

noncomputable def g (a : ℝ) : ℝ := M a - N a

theorem g_expression_and_minimum (a : ℝ) (h : 1/3 ≤ a ∧ a ≤ 1) :
  ((1/3 ≤ a ∧ a ≤ 1/2 → g a = a - 2 + 1/a) ∧
   (1/2 < a ∧ a ≤ 1 → g a = 9*a - 6 + 1/a)) ∧
  (∀ b, 1/3 ≤ b ∧ b ≤ 1 → g b ≥ 1/2) ∧
  g (1/2) = 1/2 := by sorry

end NUMINAMATH_CALUDE_g_expression_and_minimum_l2891_289148


namespace NUMINAMATH_CALUDE_pams_bags_to_geralds_bags_l2891_289152

/-- Represents the number of apples in each of Gerald's bags -/
def geralds_bag_size : ℕ := 40

/-- Represents the total number of apples Pam has -/
def pams_total_apples : ℕ := 1200

/-- Represents the number of bags Pam has -/
def pams_bag_count : ℕ := 10

/-- Theorem stating that each of Pam's bags equates to 3 of Gerald's bags -/
theorem pams_bags_to_geralds_bags : 
  (pams_total_apples / pams_bag_count) / geralds_bag_size = 3 := by
  sorry

end NUMINAMATH_CALUDE_pams_bags_to_geralds_bags_l2891_289152


namespace NUMINAMATH_CALUDE_divisibility_by_ten_l2891_289150

theorem divisibility_by_ten (a : ℤ) : 
  (10 ∣ (a^10 + 1)) ↔ (a % 10 = 3 ∨ a % 10 = 7 ∨ a % 10 = -3 ∨ a % 10 = -7) := by
  sorry

end NUMINAMATH_CALUDE_divisibility_by_ten_l2891_289150


namespace NUMINAMATH_CALUDE_ceiling_equation_solution_l2891_289181

theorem ceiling_equation_solution :
  ∃! b : ℝ, b + ⌈b⌉ = 14.7 ∧ b = 7.2 := by sorry

end NUMINAMATH_CALUDE_ceiling_equation_solution_l2891_289181


namespace NUMINAMATH_CALUDE_geometric_sequence_problem_l2891_289102

theorem geometric_sequence_problem (a : ℕ → ℚ) :
  (∀ n m : ℕ, a (n + m) = a n * a m) →  -- geometric sequence property
  a 5 = (5 : ℚ) / 3 →                   -- 5th term equals constant term of expansion
  a 3 * a 7 = (25 : ℚ) / 9 :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_problem_l2891_289102


namespace NUMINAMATH_CALUDE_x_value_proof_l2891_289168

theorem x_value_proof (x : ℚ) 
  (eq1 : 8 * x^2 + 7 * x - 1 = 0)
  (eq2 : 24 * x^2 + 53 * x - 7 = 0) :
  x = 1/8 := by
sorry

end NUMINAMATH_CALUDE_x_value_proof_l2891_289168


namespace NUMINAMATH_CALUDE_no_numbers_satisfy_condition_l2891_289122

def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + sum_of_digits (n / 10)

def satisfies_condition (n : ℕ) : Prop :=
  n > 0 ∧ n ≤ 1000 ∧ n = 7 * sum_of_digits n

theorem no_numbers_satisfy_condition : ¬∃ n : ℕ, satisfies_condition n := by
  sorry

end NUMINAMATH_CALUDE_no_numbers_satisfy_condition_l2891_289122


namespace NUMINAMATH_CALUDE_rams_and_ravis_selection_probability_l2891_289192

theorem rams_and_ravis_selection_probability 
  (p_ram : ℝ) 
  (p_both : ℝ) 
  (h1 : p_ram = 2/7)
  (h2 : p_both = 0.05714285714285714)
  (h3 : p_both = p_ram * (p_ravi : ℝ)) : 
  p_ravi = 1/5 := by
sorry

end NUMINAMATH_CALUDE_rams_and_ravis_selection_probability_l2891_289192


namespace NUMINAMATH_CALUDE_quadratic_discriminant_l2891_289153

theorem quadratic_discriminant : 
  let a : ℚ := 5
  let b : ℚ := 5 + 1/5
  let c : ℚ := 1/5
  let discriminant := b^2 - 4*a*c
  discriminant = 576/25 := by sorry

end NUMINAMATH_CALUDE_quadratic_discriminant_l2891_289153


namespace NUMINAMATH_CALUDE_abs_neg_two_equals_two_l2891_289154

theorem abs_neg_two_equals_two : |(-2 : ℤ)| = 2 := by
  sorry

end NUMINAMATH_CALUDE_abs_neg_two_equals_two_l2891_289154


namespace NUMINAMATH_CALUDE_space_diagonals_of_specific_polyhedron_l2891_289159

/-- A convex polyhedron with specified properties -/
structure ConvexPolyhedron where
  vertices : ℕ
  edges : ℕ
  faces : ℕ
  triangular_faces : ℕ
  quadrilateral_faces : ℕ

/-- Calculate the number of space diagonals in a convex polyhedron -/
def space_diagonals (Q : ConvexPolyhedron) : ℕ :=
  let total_line_segments := Q.vertices.choose 2
  let face_diagonals := Q.quadrilateral_faces * 2
  total_line_segments - Q.edges - face_diagonals

/-- The main theorem stating the number of space diagonals in the given polyhedron -/
theorem space_diagonals_of_specific_polyhedron :
  let Q : ConvexPolyhedron := {
    vertices := 30,
    edges := 70,
    faces := 42,
    triangular_faces := 30,
    quadrilateral_faces := 12
  }
  space_diagonals Q = 341 := by sorry

end NUMINAMATH_CALUDE_space_diagonals_of_specific_polyhedron_l2891_289159


namespace NUMINAMATH_CALUDE_subtraction_and_simplification_l2891_289119

theorem subtraction_and_simplification :
  (12 : ℚ) / 25 - (3 : ℚ) / 75 = (11 : ℚ) / 25 := by sorry

end NUMINAMATH_CALUDE_subtraction_and_simplification_l2891_289119


namespace NUMINAMATH_CALUDE_cylinder_cross_section_area_l2891_289116

/-- Represents a cylinder formed from a rectangle --/
structure Cylinder where
  rect_length : ℝ
  rect_width : ℝ
  height : ℝ

/-- Calculates the area of the cross-section through the axis of the cylinder --/
def axial_cross_section_area (c : Cylinder) : ℝ :=
  c.rect_length * c.height

theorem cylinder_cross_section_area :
  let c : Cylinder := { rect_length := 4, rect_width := 2, height := 2 }
  axial_cross_section_area c = 8 / Real.pi := by
  sorry

end NUMINAMATH_CALUDE_cylinder_cross_section_area_l2891_289116


namespace NUMINAMATH_CALUDE_parallel_vectors_second_component_l2891_289175

/-- Given vectors a and b in ℝ², if a is parallel to (a + b), then the second component of b is -3. -/
theorem parallel_vectors_second_component (a b : ℝ × ℝ) (h : a.1 = -1 ∧ a.2 = 1 ∧ b.1 = 3) :
  (∃ (k : ℝ), k ≠ 0 ∧ a = k • (a + b)) → b.2 = -3 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_second_component_l2891_289175


namespace NUMINAMATH_CALUDE_security_check_comprehensive_l2891_289126

/-- Represents a survey method -/
inductive SurveyMethod
| Comprehensive
| Sample

/-- Represents a scenario for which a survey method is chosen -/
structure Scenario where
  requiresAllChecked : Bool
  noExceptions : Bool
  populationAccessible : Bool
  populationFinite : Bool

/-- Determines the correct survey method for a given scenario -/
def correctSurveyMethod (s : Scenario) : SurveyMethod :=
  if s.requiresAllChecked && s.noExceptions && s.populationAccessible && s.populationFinite then
    SurveyMethod.Comprehensive
  else
    SurveyMethod.Sample

/-- The scenario of security checks before boarding a plane -/
def securityCheckScenario : Scenario :=
  { requiresAllChecked := true
    noExceptions := true
    populationAccessible := true
    populationFinite := true }

theorem security_check_comprehensive :
  correctSurveyMethod securityCheckScenario = SurveyMethod.Comprehensive := by
  sorry


end NUMINAMATH_CALUDE_security_check_comprehensive_l2891_289126


namespace NUMINAMATH_CALUDE_product_xyz_is_one_ninth_l2891_289160

theorem product_xyz_is_one_ninth 
  (x y z : ℝ) 
  (h1 : x + 1/y = 3) 
  (h2 : y + 1/z = 5) : 
  x * y * z = 1/9 := by
sorry

end NUMINAMATH_CALUDE_product_xyz_is_one_ninth_l2891_289160


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_l2891_289162

theorem quadratic_equation_roots (a : ℝ) : 
  (∀ x : ℝ, x^2 + 2*a*x + 1 = 0 → x < 0) → 
  (∃ x y : ℝ, x ≠ y ∧ x^2 + 2*a*x + 1 = 0 ∧ y^2 + 2*a*y + 1 = 0) → 
  a ≥ 1 :=
sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_l2891_289162


namespace NUMINAMATH_CALUDE_arithmetic_sequence_ratio_l2891_289194

/-- An arithmetic sequence -/
structure ArithmeticSequence where
  a : ℕ → ℚ
  first_term : ℚ
  common_diff : ℚ
  seq_def : ∀ n, a n = first_term + (n - 1) * common_diff

/-- Sum of first n terms of an arithmetic sequence -/
def sum_n_terms (seq : ArithmeticSequence) (n : ℕ) : ℚ :=
  n * (2 * seq.first_term + (n - 1) * seq.common_diff) / 2

theorem arithmetic_sequence_ratio (a b : ArithmeticSequence) :
  (∀ n : ℕ, n > 0 → sum_n_terms a n / sum_n_terms b n = (2 * n + 3 : ℚ) / (3 * n - 1)) →
  a.a 9 / b.a 9 = 37 / 50 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_ratio_l2891_289194


namespace NUMINAMATH_CALUDE_intersection_point_x_coordinate_l2891_289165

theorem intersection_point_x_coordinate 
  (f : ℝ → ℝ) 
  (hf : ∀ x, f x = Real.log x) 
  (P Q : ℝ × ℝ) 
  (hP : P = (2, Real.log 2)) 
  (hQ : Q = (500, Real.log 500)) 
  (hPQ : P.1 < Q.1) 
  (R : ℝ × ℝ) 
  (hR : R = ((P.1 + Q.1) / 2, (P.2 + Q.2) / 2)) 
  (T : ℝ × ℝ) 
  (hT : T.2 = R.2 ∧ f T.1 = T.2 ∧ T.1 ≠ R.1) : 
  T.1 = Real.sqrt 1000 := by
sorry

end NUMINAMATH_CALUDE_intersection_point_x_coordinate_l2891_289165


namespace NUMINAMATH_CALUDE_printer_time_relationship_l2891_289163

/-- Represents a printer's capability to print leaflets -/
structure Printer :=
  (time : ℝ)  -- Time taken to print 800 leaflets

/-- Represents a system of two printers -/
structure PrinterSystem :=
  (printer1 : Printer)
  (printer2 : Printer)
  (combined_time : ℝ)  -- Time taken by both printers together to print 800 leaflets

/-- Theorem stating the relationship between individual printer times and combined time -/
theorem printer_time_relationship (system : PrinterSystem) 
    (h1 : system.printer1.time = 12)
    (h2 : system.combined_time = 3) :
    (1 / system.printer1.time) + (1 / system.printer2.time) = (1 / system.combined_time) :=
  sorry

end NUMINAMATH_CALUDE_printer_time_relationship_l2891_289163


namespace NUMINAMATH_CALUDE_train_speed_l2891_289108

/-- The speed of a train passing through a tunnel -/
theorem train_speed (train_length : ℝ) (tunnel_length : ℝ) (pass_time : ℝ) :
  train_length = 100 →
  tunnel_length = 1.7 →
  pass_time = 1.5 / 60 →
  (train_length / 1000 + tunnel_length) / pass_time = 72 := by
  sorry

end NUMINAMATH_CALUDE_train_speed_l2891_289108


namespace NUMINAMATH_CALUDE_theresa_extra_games_video_game_comparison_l2891_289110

-- Define the number of video games for each person
def tory_games : ℕ := 6
def theresa_games : ℕ := 11

-- Define the relationship between Julia's and Tory's games
def julia_games : ℕ := tory_games / 3

-- Define the relationship between Theresa's and Julia's games
def theresa_more_than_thrice_julia : Prop :=
  theresa_games > 3 * julia_games

-- Theorem to prove
theorem theresa_extra_games :
  theresa_games - 3 * julia_games = 5 :=
by
  sorry

-- Main theorem that encapsulates the problem
theorem video_game_comparison
  (h1 : theresa_more_than_thrice_julia)
  (h2 : julia_games = tory_games / 3)
  (h3 : tory_games = 6)
  (h4 : theresa_games = 11) :
  theresa_games - 3 * julia_games = 5 :=
by
  sorry

end NUMINAMATH_CALUDE_theresa_extra_games_video_game_comparison_l2891_289110


namespace NUMINAMATH_CALUDE_problem_statement_l2891_289124

theorem problem_statement (a b c d : ℝ) 
  (h : (a - b) * (c - d) / ((b - c) * (d - a)) = 2 / 5) :
  (a - c) * (b - d) / ((a - b) * (c - d)) = -3 / 2 := by
sorry

end NUMINAMATH_CALUDE_problem_statement_l2891_289124


namespace NUMINAMATH_CALUDE_distance_between_centers_l2891_289138

-- Define the circles
def circle_M (x y : ℝ) : Prop := x^2 + y^2 = 1
def circle_N (x y : ℝ) : Prop := x^2 + (y-2)^2 = 1

-- Define the centers of the circles
def center_M : ℝ × ℝ := (0, 0)
def center_N : ℝ × ℝ := (0, 2)

-- State the theorem
theorem distance_between_centers :
  let (x₁, y₁) := center_M
  let (x₂, y₂) := center_N
  Real.sqrt ((x₂ - x₁)^2 + (y₂ - y₁)^2) = 2 := by sorry

end NUMINAMATH_CALUDE_distance_between_centers_l2891_289138


namespace NUMINAMATH_CALUDE_farmer_feed_cost_l2891_289104

theorem farmer_feed_cost (total_spent : ℝ) (chicken_feed_percent : ℝ) (chicken_discount : ℝ) : 
  total_spent = 35 →
  chicken_feed_percent = 0.4 →
  chicken_discount = 0.5 →
  let chicken_feed_cost := total_spent * chicken_feed_percent
  let goat_feed_cost := total_spent * (1 - chicken_feed_percent)
  let full_price_chicken_feed := chicken_feed_cost / (1 - chicken_discount)
  let full_price_total := full_price_chicken_feed + goat_feed_cost
  full_price_total = 49 := by sorry

end NUMINAMATH_CALUDE_farmer_feed_cost_l2891_289104


namespace NUMINAMATH_CALUDE_zeros_of_f_l2891_289182

-- Define the function f(x) = x^3 - 3x + 2
def f (x : ℝ) : ℝ := x^3 - 3*x + 2

-- Theorem stating that 1 and -2 are the zeros of f
theorem zeros_of_f : 
  (∃ x : ℝ, f x = 0) ∧ (∀ x : ℝ, f x = 0 → x = 1 ∨ x = -2) ∧ f 1 = 0 ∧ f (-2) = 0 := by
  sorry

end NUMINAMATH_CALUDE_zeros_of_f_l2891_289182


namespace NUMINAMATH_CALUDE_largest_b_for_divisibility_l2891_289121

def is_divisible_by_4 (n : ℕ) : Prop := n % 4 = 0

def five_digit_number (b : ℕ) : ℕ := 48000 + b * 100 + 56

theorem largest_b_for_divisibility :
  ∀ b : ℕ, b ≤ 9 →
    (is_divisible_by_4 (five_digit_number b) → b ≤ 8) ∧
    is_divisible_by_4 (five_digit_number 8) :=
by sorry

end NUMINAMATH_CALUDE_largest_b_for_divisibility_l2891_289121


namespace NUMINAMATH_CALUDE_lunch_cost_l2891_289199

theorem lunch_cost (x : ℝ) : 
  x + 0.04 * x + 0.06 * x = 110 → x = 100 := by
  sorry

end NUMINAMATH_CALUDE_lunch_cost_l2891_289199


namespace NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l2891_289134

theorem geometric_sequence_common_ratio
  (a : ℝ)
  (seq : ℕ → ℝ)
  (h_seq : ∀ n : ℕ, seq n = a + Real.log 3 / Real.log (2^(2^n)))
  : (∃ q : ℝ, ∀ n : ℕ, seq (n + 1) = q * seq n) ∧
    (∀ q : ℝ, (∀ n : ℕ, seq (n + 1) = q * seq n) → q = 1/3) :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l2891_289134


namespace NUMINAMATH_CALUDE_worksheets_graded_l2891_289107

theorem worksheets_graded (total_worksheets : ℕ) (problems_per_worksheet : ℕ) (problems_left : ℕ) : 
  total_worksheets = 9 →
  problems_per_worksheet = 4 →
  problems_left = 16 →
  total_worksheets - (problems_left / problems_per_worksheet) = 5 := by
sorry

end NUMINAMATH_CALUDE_worksheets_graded_l2891_289107


namespace NUMINAMATH_CALUDE_solution_satisfies_system_l2891_289147

/-- The system of linear equations -/
def system (x : ℝ × ℝ × ℝ × ℝ) : Prop :=
  let (x₁, x₂, x₃, x₄) := x
  x₁ + 2*x₂ + 3*x₃ + x₄ = 1 ∧
  3*x₁ + 13*x₂ + 13*x₃ + 5*x₄ = 3 ∧
  3*x₁ + 7*x₂ + 7*x₃ + 2*x₄ = 12 ∧
  x₁ + 5*x₂ + 3*x₃ + x₄ = 7 ∧
  4*x₁ + 5*x₂ + 6*x₃ + x₄ = 19

/-- The general solution to the system -/
def solution (α : ℝ) : ℝ × ℝ × ℝ × ℝ :=
  (4 - α, 2, α, -7 - 2*α)

/-- Theorem stating that the general solution satisfies the system for any α -/
theorem solution_satisfies_system :
  ∀ α : ℝ, system (solution α) :=
by sorry

end NUMINAMATH_CALUDE_solution_satisfies_system_l2891_289147


namespace NUMINAMATH_CALUDE_line_through_origin_and_intersection_l2891_289198

/-- The equation of the line passing through the origin and the intersection of two given lines -/
theorem line_through_origin_and_intersection (x y : ℝ) : 
  (x - 2*y + 2 = 0) →  -- Line 1 equation
  (2*x - y - 2 = 0) →  -- Line 2 equation
  (∃ t : ℝ, x = t ∧ y = t) -- Equation of the line y = x in parametric form
  := by sorry

end NUMINAMATH_CALUDE_line_through_origin_and_intersection_l2891_289198


namespace NUMINAMATH_CALUDE_target_circle_properties_l2891_289151

/-- The line equation -/
def line_eq (x y : ℝ) : Prop := 2 * x - y + 1 = 0

/-- The given circle equation -/
def given_circle_eq (x y : ℝ) : Prop := x^2 + y^2 - 2*x - 15 = 0

/-- The equation of the circle we need to prove -/
def target_circle_eq (x y : ℝ) : Prop := x^2 + y^2 + 28*x - 15*y = 0

/-- Theorem stating that the target circle passes through the intersection points
    of the line and the given circle, and also through the origin -/
theorem target_circle_properties :
  (∀ x y : ℝ, line_eq x y ∧ given_circle_eq x y → target_circle_eq x y) ∧
  target_circle_eq 0 0 := by
  sorry

end NUMINAMATH_CALUDE_target_circle_properties_l2891_289151


namespace NUMINAMATH_CALUDE_sum_of_series_equals_one_l2891_289115

/-- The sum of the infinite series ∑(n=1 to ∞) (4n-3)/(3^n) is equal to 1 -/
theorem sum_of_series_equals_one :
  (∑' n : ℕ, (4 * n - 3 : ℝ) / (3 : ℝ) ^ n) = 1 := by
sorry

end NUMINAMATH_CALUDE_sum_of_series_equals_one_l2891_289115


namespace NUMINAMATH_CALUDE_f_2007_equals_zero_l2891_289195

/-- A function f : ℝ → ℝ is even if f(-x) = f(x) for all x -/
def IsEven (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

/-- A function g : ℝ → ℝ is odd if g(-x) = -g(x) for all x -/
def IsOdd (g : ℝ → ℝ) : Prop := ∀ x, g (-x) = -g x

theorem f_2007_equals_zero
  (f g : ℝ → ℝ)
  (h_even : IsEven f)
  (h_odd : IsOdd g)
  (h_fg : ∀ x, g x = f (x - 1)) :
  f 2007 = 0 := by
  sorry

end NUMINAMATH_CALUDE_f_2007_equals_zero_l2891_289195


namespace NUMINAMATH_CALUDE_jessica_almonds_l2891_289188

theorem jessica_almonds : ∃ (j : ℕ), 
  (∃ (l : ℕ), j = l + 8 ∧ l = j / 3) → j = 12 :=
by
  sorry

end NUMINAMATH_CALUDE_jessica_almonds_l2891_289188


namespace NUMINAMATH_CALUDE_f_of_g_5_l2891_289184

-- Define the functions f and g
def g (x : ℝ) : ℝ := 4 * x + 10
def f (x : ℝ) : ℝ := 6 * x - 12

-- State the theorem
theorem f_of_g_5 : f (g 5) = 168 := by
  sorry

end NUMINAMATH_CALUDE_f_of_g_5_l2891_289184


namespace NUMINAMATH_CALUDE_quadrilateral_area_l2891_289131

/-- Represents a triangle partitioned into three triangles and a quadrilateral -/
structure PartitionedTriangle where
  /-- Area of the first triangle -/
  area1 : ℝ
  /-- Area of the second triangle -/
  area2 : ℝ
  /-- Area of the third triangle -/
  area3 : ℝ
  /-- Area of the quadrilateral -/
  areaQuad : ℝ

/-- The theorem stating that if the areas of the three triangles are 3, 7, and 7,
    then the area of the quadrilateral is 18 -/
theorem quadrilateral_area (t : PartitionedTriangle) 
    (h1 : t.area1 = 3) 
    (h2 : t.area2 = 7) 
    (h3 : t.area3 = 7) : 
    t.areaQuad = 18 := by
  sorry


end NUMINAMATH_CALUDE_quadrilateral_area_l2891_289131
