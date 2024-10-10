import Mathlib

namespace problem_statement_l1385_138598

theorem problem_statement (a b c : ℝ) 
  (h1 : a * c / (a + b) + b * a / (b + c) + c * b / (c + a) = -9)
  (h2 : b * c / (a + b) + c * a / (b + c) + a * b / (c + a) = 10) :
  b / (a + b) + c / (b + c) + a / (c + a) = 11 := by
  sorry

end problem_statement_l1385_138598


namespace work_completion_time_l1385_138593

theorem work_completion_time (b c total_time : ℝ) (total_payment c_payment : ℕ) 
  (hb : b = 8)
  (hc : c = 3)
  (htotal_payment : total_payment = 3680)
  (hc_payment : c_payment = 460) :
  ∃ a : ℝ, a = 24 / 5 ∧ 1 / a + 1 / b = 1 / c := by
sorry

end work_completion_time_l1385_138593


namespace second_third_ratio_l1385_138580

theorem second_third_ratio (A B C : ℚ) : 
  A + B + C = 98 →
  A / B = 2 / 3 →
  B = 30 →
  B / C = 5 / 8 := by
  sorry

end second_third_ratio_l1385_138580


namespace tour_group_composition_l1385_138587

/-- Represents the number of people in a tour group -/
structure TourGroup where
  total : ℕ
  children : ℕ

/-- Represents the ticket prices -/
structure TicketPrices where
  adult : ℕ
  child : ℕ

/-- The main theorem statement -/
theorem tour_group_composition 
  (group_a group_b : TourGroup) 
  (prices : TicketPrices) : 
  (group_b.total = group_a.total + 4) →
  (group_a.total + group_b.total = 18 * (group_b.total - group_a.total)) →
  (group_b.children = 3 * group_a.children - 2) →
  (prices.adult = 100) →
  (prices.child = prices.adult * 3 / 5) →
  (prices.adult * (group_a.total - group_a.children) + prices.child * group_a.children = 
   prices.adult * (group_b.total - group_b.children) + prices.child * group_b.children) →
  (group_a.total = 34 ∧ group_a.children = 6 ∧ 
   group_b.total = 38 ∧ group_b.children = 16) :=
by sorry

end tour_group_composition_l1385_138587


namespace passengers_who_got_off_l1385_138551

theorem passengers_who_got_off (initial : ℕ) (got_on : ℕ) (final : ℕ) : 
  initial = 28 → got_on = 7 → final = 26 → initial + got_on - final = 9 := by
  sorry

end passengers_who_got_off_l1385_138551


namespace ellipse_and_line_intersection_l1385_138578

/-- Ellipse C with foci at (-2,0) and (2,0), passing through (0, √5) -/
def ellipse_C : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | ∃ (x y : ℝ), p = (x, y) ∧ x^2/9 + y^2/5 = 1}

/-- Line l passing through (-2,0) with slope 1 -/
def line_l : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | ∃ (x y : ℝ), p = (x, y) ∧ y = x + 2}

/-- Theorem stating the standard equation of ellipse C and the length of PQ -/
theorem ellipse_and_line_intersection :
  (∀ (x y : ℝ), (x, y) ∈ ellipse_C ↔ x^2/9 + y^2/5 = 1) ∧
  (∃ (P Q : ℝ × ℝ), P ∈ ellipse_C ∧ Q ∈ ellipse_C ∧ P ∈ line_l ∧ Q ∈ line_l ∧
    Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2) = 30/7) :=
by sorry

end ellipse_and_line_intersection_l1385_138578


namespace binary_to_decimal_11001101_l1385_138583

/-- Converts a list of binary digits to its decimal equivalent -/
def binary_to_decimal (binary_digits : List Bool) : ℕ :=
  binary_digits.enum.foldl
    (fun acc (i, b) => acc + (if b then 2^i else 0))
    0

/-- The binary representation of the number we want to convert -/
def binary_number : List Bool := [true, false, true, true, false, false, true, true]

/-- Theorem stating that the decimal equivalent of 11001101 (binary) is 205 -/
theorem binary_to_decimal_11001101 :
  binary_to_decimal (binary_number.reverse) = 205 := by
  sorry

#eval binary_to_decimal (binary_number.reverse)

end binary_to_decimal_11001101_l1385_138583


namespace frog_jumped_farther_l1385_138556

/-- The distance jumped by the grasshopper in inches -/
def grasshopper_jump : ℕ := 9

/-- The distance jumped by the frog in inches -/
def frog_jump : ℕ := 12

/-- The difference between the frog's jump and the grasshopper's jump -/
def jump_difference : ℕ := frog_jump - grasshopper_jump

/-- Theorem stating that the frog jumped 3 inches farther than the grasshopper -/
theorem frog_jumped_farther : jump_difference = 3 := by
  sorry

end frog_jumped_farther_l1385_138556


namespace f_increasing_interval_l1385_138513

-- Define the function
def f (x : ℝ) : ℝ := x^2 - 6*x

-- State the theorem
theorem f_increasing_interval :
  ∀ x y : ℝ, x ≥ 3 → y > x → f y > f x :=
sorry

end f_increasing_interval_l1385_138513


namespace remainder_3_153_mod_8_l1385_138525

theorem remainder_3_153_mod_8 : 3^153 % 8 = 3 := by
  sorry

end remainder_3_153_mod_8_l1385_138525


namespace parallel_vectors_k_value_l1385_138531

/-- Given two vectors a and b in ℝ², prove that if they are parallel and
    a = (2, -1) and b = (k, 5/2), then k = -5. -/
theorem parallel_vectors_k_value (a b : ℝ × ℝ) (k : ℝ) :
  a = (2, -1) →
  b = (k, 5/2) →
  (∃ (t : ℝ), t ≠ 0 ∧ b = t • a) →
  k = -5 :=
by sorry

end parallel_vectors_k_value_l1385_138531


namespace f_is_linear_equation_l1385_138582

/-- A linear equation with two variables is of the form ax + by = c, where a, b, and c are constants. -/
def is_linear_equation (f : ℝ → ℝ → ℝ) : Prop :=
  ∃ (a b c : ℝ), ∀ x y, f x y = a * x + b * y - c

/-- The specific equation we want to prove is linear. -/
def f (x y : ℝ) : ℝ := 4 * x - 5 * y - 5

theorem f_is_linear_equation : is_linear_equation f := by
  sorry

end f_is_linear_equation_l1385_138582


namespace inequality_for_elements_in_M_l1385_138565

-- Define the set M
def M : Set ℝ := {x : ℝ | -1/2 < x ∧ x < 1}

-- State the theorem
theorem inequality_for_elements_in_M (a b : ℝ) (ha : a ∈ M) (hb : b ∈ M) :
  |a - b| < |1 - a * b| := by
  sorry

end inequality_for_elements_in_M_l1385_138565


namespace function_value_sum_l1385_138540

/-- A quadratic function f(x) with specific properties -/
def f (a : ℝ) (x : ℝ) : ℝ := a * (x + 2)^2 + 4

/-- The theorem stating the value of a + b + 2c for the given function -/
theorem function_value_sum (a : ℝ) :
  f a 0 = 5 ∧ f a 2 = 5 → a + 0 + 2 * 4 = 8.25 := by sorry

end function_value_sum_l1385_138540


namespace hyperbola_n_range_l1385_138539

/-- Represents a hyperbola with parameters m and n -/
structure Hyperbola (m n : ℝ) where
  eq : ∀ x y : ℝ, x^2 / (m^2 + n) - y^2 / (3 * m^2 - n) = 1

/-- The distance between the foci of the hyperbola is 4 -/
def foci_distance (m n : ℝ) : Prop :=
  (m^2 + n) + (3 * m^2 - n) = 4

/-- The theorem stating the range of n for the given hyperbola -/
theorem hyperbola_n_range (m n : ℝ) (h : Hyperbola m n) (d : foci_distance m n) :
  -1 < n ∧ n < 3 := by
  sorry

end hyperbola_n_range_l1385_138539


namespace point_on_y_axis_l1385_138581

/-- A point M with coordinates (m+3, m+1) lies on the y-axis if and only if its coordinates are (0, -2) -/
theorem point_on_y_axis (m : ℝ) : 
  (m + 3 = 0 ∧ m + 1 = -2) ↔ (m + 3 = 0 ∧ m + 1 = -2) :=
by sorry

end point_on_y_axis_l1385_138581


namespace sine_phase_shift_l1385_138591

/-- The phase shift of the sine function y = sin(4x + π/2) is π/8 units to the left. -/
theorem sine_phase_shift :
  let f : ℝ → ℝ := λ x ↦ Real.sin (4 * x + π / 2)
  ∃ (φ : ℝ), φ = π / 8 ∧
    ∀ x, f x = Real.sin (4 * (x + φ)) := by
  sorry

end sine_phase_shift_l1385_138591


namespace power_of_two_equation_l1385_138534

theorem power_of_two_equation (k : ℤ) : 
  2^2000 - 2^1999 - 2^1998 + 2^1997 = k * 2^1997 → k = 3 := by
  sorry

end power_of_two_equation_l1385_138534


namespace only_5_12_13_is_pythagorean_triple_l1385_138502

def is_pythagorean_triple (a b c : ℕ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ a * a + b * b = c * c

theorem only_5_12_13_is_pythagorean_triple :
  ¬ is_pythagorean_triple 3 4 7 ∧
  ¬ is_pythagorean_triple 1 3 5 ∧
  is_pythagorean_triple 5 12 13 :=
by sorry

end only_5_12_13_is_pythagorean_triple_l1385_138502


namespace triangle_angle_proof_l1385_138558

theorem triangle_angle_proof (a b : ℝ) (B : ℝ) (A : ℝ) : 
  a = 4 → b = 4 * Real.sqrt 2 → B = π/4 → A = π/6 := by
  sorry

end triangle_angle_proof_l1385_138558


namespace digits_of_2_pow_120_l1385_138546

theorem digits_of_2_pow_120 (h : ∃ n : ℕ, 10^60 ≤ 2^200 ∧ 2^200 < 10^61) :
  ∃ n : ℕ, 10^36 ≤ 2^120 ∧ 2^120 < 10^37 :=
sorry

end digits_of_2_pow_120_l1385_138546


namespace ladder_slide_l1385_138526

-- Define the ladder and wall setup
def ladder_length : ℝ := 25
def initial_distance : ℝ := 7
def top_slide : ℝ := 4

-- Theorem statement
theorem ladder_slide :
  let initial_height : ℝ := Real.sqrt (ladder_length^2 - initial_distance^2)
  let new_height : ℝ := initial_height - top_slide
  let new_distance : ℝ := Real.sqrt (ladder_length^2 - new_height^2)
  new_distance - initial_distance = 8 := by
  sorry

end ladder_slide_l1385_138526


namespace fifteen_exponent_division_l1385_138599

theorem fifteen_exponent_division : (15 : ℕ)^11 / (15 : ℕ)^8 = 3375 := by sorry

end fifteen_exponent_division_l1385_138599


namespace regular_octagon_interior_angle_regular_octagon_interior_angle_is_135_l1385_138559

theorem regular_octagon_interior_angle : ℝ :=
  let n : ℕ := 8  -- number of sides in an octagon
  let sum_of_interior_angles : ℝ := 180 * (n - 2)
  let one_interior_angle : ℝ := sum_of_interior_angles / n
  135

/-- The measure of one interior angle of a regular octagon is 135 degrees. -/
theorem regular_octagon_interior_angle_is_135 :
  regular_octagon_interior_angle = 135 := by
  sorry

end regular_octagon_interior_angle_regular_octagon_interior_angle_is_135_l1385_138559


namespace simplify_fraction_l1385_138521

theorem simplify_fraction : (15^30) / (45^15) = 5^15 := by
  sorry

end simplify_fraction_l1385_138521


namespace vector_sum_l1385_138553

def vector1 : ℝ × ℝ × ℝ := (4, -9, 2)
def vector2 : ℝ × ℝ × ℝ := (-1, 16, 5)

theorem vector_sum : 
  (vector1.1 + vector2.1, vector1.2.1 + vector2.2.1, vector1.2.2 + vector2.2.2) = (3, 7, 7) := by
  sorry

end vector_sum_l1385_138553


namespace wedge_volume_l1385_138504

/-- The volume of a wedge cut from a cylindrical log --/
theorem wedge_volume (d h : ℝ) (θ : ℝ) : 
  d = 20 → θ = 30 → (π * (d/2)^2 * h * θ) / 360 = (500/3) * π := by
  sorry

end wedge_volume_l1385_138504


namespace inequality_proof_l1385_138586

theorem inequality_proof (a b : ℝ) (ha : 0 ≤ a) (hb : 0 ≤ b) :
  a^2 + a*b + b^2 ≤ 3*(a - Real.sqrt (a*b) + b)^2 := by
  sorry

end inequality_proof_l1385_138586


namespace min_value_of_f_l1385_138516

/-- The function f(x) = 3/x + 1/(1-3x) has a minimum value of 16 on the interval (0, 1/3) -/
theorem min_value_of_f (x : ℝ) (hx : 0 < x ∧ x < 1/3) : 3/x + 1/(1-3*x) ≥ 16 := by
  sorry

end min_value_of_f_l1385_138516


namespace square_ratio_proof_l1385_138554

theorem square_ratio_proof (area_ratio : ℚ) :
  area_ratio = 300 / 75 →
  ∃ (a b c : ℕ), 
    (a * Real.sqrt b : ℝ) / c = Real.sqrt area_ratio ∧
    a = 2 ∧ b = 1 ∧ c = 1 ∧
    a + b + c = 4 := by
  sorry

end square_ratio_proof_l1385_138554


namespace sum_of_combined_sequence_l1385_138530

/-- Given geometric sequence {aₙ} and arithmetic sequence {bₙ} -/
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = a n * q

def arithmetic_sequence (b : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, b (n + 1) = b n + d

/-- Theorem stating the sum of the first n terms of the sequence cₙ = aₙ + bₙ -/
theorem sum_of_combined_sequence
  (a b : ℕ → ℝ)
  (h_a : geometric_sequence a)
  (h_b : arithmetic_sequence b)
  (h_a1 : a 1 = 1)
  (h_a4 : a 4 = 8)
  (h_b1 : b 1 = 3)
  (h_b4 : b 4 = 12) :
  ∃ S : ℕ → ℝ, ∀ n : ℕ, S n = 2^n - 1 + (3/2 * n^2) + (3/2 * n) :=
by sorry

end sum_of_combined_sequence_l1385_138530


namespace caterer_order_l1385_138569

theorem caterer_order (ice_cream_price sundae_price total_price : ℚ) 
  (h1 : ice_cream_price = 0.60)
  (h2 : sundae_price = 1.20)
  (h3 : total_price = 225.00)
  (h4 : ice_cream_price * x + sundae_price * x = total_price) :
  x = 125 :=
by
  sorry

#check caterer_order

end caterer_order_l1385_138569


namespace locus_of_centers_l1385_138518

-- Define the circles C1 and C2
def C1 (x y : ℝ) : Prop := x^2 + y^2 = 4
def C2 (x y : ℝ) : Prop := (x - 3)^2 + y^2 = 25

-- Define external tangency to C1
def externally_tangent_to_C1 (a b r : ℝ) : Prop := a^2 + b^2 = (r + 2)^2

-- Define internal tangency to C2
def internally_tangent_to_C2 (a b r : ℝ) : Prop := (a - 3)^2 + b^2 = (5 - r)^2

-- Define the locus equation
def locus_equation (a b : ℝ) : Prop := 13 * a^2 + 49 * b^2 - 12 * a - 1 = 0

-- Theorem statement
theorem locus_of_centers :
  ∀ a b : ℝ,
  (∃ r : ℝ, externally_tangent_to_C1 a b r ∧ internally_tangent_to_C2 a b r) ↔
  locus_equation a b :=
sorry

end locus_of_centers_l1385_138518


namespace intersection_line_passes_through_intersection_points_l1385_138528

/-- The equation of the first circle -/
def circle1 (x y : ℝ) : Prop := x^2 + y^2 + 4*x - 4*y - 12 = 0

/-- The equation of the second circle -/
def circle2 (x y : ℝ) : Prop := x^2 + y^2 + 2*x + 4*y - 4 = 0

/-- The equation of the line passing through the intersection points -/
def intersection_line (x y : ℝ) : Prop := x - 4*y - 4 = 0

/-- Theorem stating that the intersection_line passes through the intersection points of circle1 and circle2 -/
theorem intersection_line_passes_through_intersection_points :
  ∀ x y : ℝ, circle1 x y ∧ circle2 x y → intersection_line x y :=
by sorry

end intersection_line_passes_through_intersection_points_l1385_138528


namespace polynomial_factorization_l1385_138548

theorem polynomial_factorization (x y : ℝ) : 
  4 * x^2 - 4 * x - y^2 + 4 * y - 3 = (2 * x + y - 3) * (2 * x - y + 1) := by
  sorry

end polynomial_factorization_l1385_138548


namespace square_area_from_diagonal_l1385_138590

theorem square_area_from_diagonal (diagonal : Real) (area : Real) :
  diagonal = 10 → area = diagonal^2 / 2 → area = 50 := by sorry

end square_area_from_diagonal_l1385_138590


namespace power_inequality_l1385_138566

theorem power_inequality (a b : ℝ) (ha : a ≥ 0) (hb : b ≥ 0) :
  a^6 + b^6 ≥ a*b*(a^4 + b^4) := by
  sorry

end power_inequality_l1385_138566


namespace problem_solution_l1385_138506

theorem problem_solution : 2^(0^(1^9)) + ((2^0)^1)^9 = 2 := by
  sorry

end problem_solution_l1385_138506


namespace number_added_to_x_l1385_138509

theorem number_added_to_x (x : ℝ) (some_number : ℝ) : 
  x + some_number = 5 → x = 4 → some_number = 1 := by
  sorry

end number_added_to_x_l1385_138509


namespace x_value_is_three_l1385_138507

theorem x_value_is_three (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : 3 * x^2 + 6 * x * y = x^3 + x * y^2) : x = 3 := by
  sorry

end x_value_is_three_l1385_138507


namespace project_completion_l1385_138541

/-- Represents the work rate of the men (amount of work done per day per person) -/
def work_rate : ℝ := 1

/-- Represents the total amount of work in the project -/
def total_work : ℝ := 1

/-- The number of days it takes the original group to complete the project -/
def original_days : ℕ := 40

/-- The number of days it takes the reduced group to complete the project -/
def reduced_days : ℕ := 50

/-- The number of men removed from the original group -/
def men_removed : ℕ := 5

theorem project_completion (M : ℕ) : 
  (M : ℝ) * work_rate * original_days = total_work ∧
  ((M : ℝ) - men_removed) * work_rate * reduced_days = total_work →
  M = 25 := by
sorry

end project_completion_l1385_138541


namespace rotated_angle_measure_l1385_138536

/-- Given an initial angle of 60 degrees and a clockwise rotation of 300 degrees,
    the resulting positive acute angle is 120 degrees. -/
theorem rotated_angle_measure (initial_angle rotation_angle : ℝ) : 
  initial_angle = 60 →
  rotation_angle = 300 →
  (360 - (rotation_angle - initial_angle)) % 360 = 120 :=
by sorry

end rotated_angle_measure_l1385_138536


namespace quadratic_factorization_l1385_138560

theorem quadratic_factorization (a b : ℤ) :
  (∀ y : ℝ, 2 * y^2 - 5 * y - 12 = (2 * y + a) * (y + b)) →
  a - b = 7 := by
  sorry

end quadratic_factorization_l1385_138560


namespace polynomial_remainder_l1385_138555

def polynomial (x : ℝ) : ℝ := 5*x^8 + 2*x^7 - 3*x^4 + 4*x^3 - 5*x^2 + 6*x - 20

def divisor (x : ℝ) : ℝ := 3*x - 6

theorem polynomial_remainder :
  ∃ q : ℝ → ℝ, ∀ x, polynomial x = q x * divisor x + 1404 := by
  sorry

end polynomial_remainder_l1385_138555


namespace no_integer_solutions_l1385_138542

theorem no_integer_solutions : ¬∃ (x y z : ℤ), 
  (x^2 - 3*x*y + 2*y^2 - z^2 = 39) ∧ 
  (-x^2 + 6*y*z + 2*z^2 = 40) ∧ 
  (x^2 + x*y + 8*z^2 = 96) := by
  sorry

end no_integer_solutions_l1385_138542


namespace binomial_16_choose_12_l1385_138577

theorem binomial_16_choose_12 : Nat.choose 16 12 = 43680 := by
  sorry

end binomial_16_choose_12_l1385_138577


namespace seven_from_five_twos_l1385_138537

theorem seven_from_five_twos : ∃ (a b c d e f g h i j : ℕ),
  (a = 2 ∧ b = 2 ∧ c = 2 ∧ d = 2 ∧ e = 2) ∧
  (f = 2 ∧ g = 2 ∧ h = 2 ∧ i = 2 ∧ j = 2) ∧
  (a * b * c - d / e = 7) ∧
  (f + g + h + i / j = 7) ∧
  ((10 * a + b) / c - d * e = 7) :=
by sorry

end seven_from_five_twos_l1385_138537


namespace min_value_fraction_sum_l1385_138547

theorem min_value_fraction_sum (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + y = 1) :
  1/x + 4/y ≥ 9 ∧ ∃ (x₀ y₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ x₀ + y₀ = 1 ∧ 1/x₀ + 4/y₀ = 9 := by
  sorry

end min_value_fraction_sum_l1385_138547


namespace cubic_and_square_sum_l1385_138523

theorem cubic_and_square_sum (x y : ℝ) : 
  x + y = 12 → xy = 20 → (x^3 + y^3 = 1008 ∧ x^2 + y^2 = 104) := by
  sorry

end cubic_and_square_sum_l1385_138523


namespace exists_78_lines_1992_intersections_l1385_138515

/-- A configuration of lines in a plane -/
structure LineConfiguration where
  num_lines : ℕ
  num_intersections : ℕ

/-- Theorem: There exists a configuration of 78 lines with exactly 1992 intersection points -/
theorem exists_78_lines_1992_intersections :
  ∃ (config : LineConfiguration), config.num_lines = 78 ∧ config.num_intersections = 1992 :=
sorry

end exists_78_lines_1992_intersections_l1385_138515


namespace infinite_divisible_factorial_exponents_l1385_138588

/-- νₚ(n) is the exponent of p in the prime factorization of n! -/
def ν (p : Nat) (n : Nat) : Nat :=
  sorry

theorem infinite_divisible_factorial_exponents
  (d : Nat) (primes : Finset Nat) (h_primes : ∀ p ∈ primes, Nat.Prime p) :
  ∃ S : Set Nat, Set.Infinite S ∧
    ∀ n ∈ S, ∀ p ∈ primes, d ∣ ν p n :=
  sorry

end infinite_divisible_factorial_exponents_l1385_138588


namespace gwen_bookcase_total_l1385_138561

/-- The number of mystery books on each shelf -/
def mystery_books_per_shelf : Nat := 7

/-- The number of shelves for mystery books -/
def mystery_shelves : Nat := 6

/-- The number of picture books on each shelf -/
def picture_books_per_shelf : Nat := 5

/-- The number of shelves for picture books -/
def picture_shelves : Nat := 4

/-- The number of biographies on each shelf -/
def biography_books_per_shelf : Nat := 3

/-- The number of shelves for biographies -/
def biography_shelves : Nat := 3

/-- The number of sci-fi books on each shelf -/
def scifi_books_per_shelf : Nat := 9

/-- The number of shelves for sci-fi books -/
def scifi_shelves : Nat := 2

/-- The total number of books on Gwen's bookcase -/
def total_books : Nat :=
  mystery_books_per_shelf * mystery_shelves +
  picture_books_per_shelf * picture_shelves +
  biography_books_per_shelf * biography_shelves +
  scifi_books_per_shelf * scifi_shelves

theorem gwen_bookcase_total : total_books = 89 := by
  sorry

end gwen_bookcase_total_l1385_138561


namespace sufficient_not_necessary_l1385_138508

theorem sufficient_not_necessary (x : ℝ) :
  (x > (1/2 : ℝ) → 2*x^2 + x - 1 > 0) ∧
  ¬(2*x^2 + x - 1 > 0 → x > (1/2 : ℝ)) :=
by sorry

end sufficient_not_necessary_l1385_138508


namespace perpendicular_condition_l1385_138535

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the perpendicular relation for planes and lines
variable (perp_plane : Plane → Plane → Prop)
variable (perp_line_plane : Line → Plane → Prop)

-- Define the subset relation for lines and planes
variable (subset : Line → Plane → Prop)

-- Theorem statement
theorem perpendicular_condition 
  (α β : Plane) (l : Line) 
  (h_subset : subset l α) :
  (∀ β, perp_line_plane l β → perp_plane α β) ∧ 
  (∃ β, perp_plane α β ∧ ¬perp_line_plane l β) :=
sorry

end perpendicular_condition_l1385_138535


namespace nancy_bottle_caps_l1385_138545

/-- The number of bottle caps Nancy starts with -/
def initial_caps : ℕ := 91

/-- The number of bottle caps Nancy finds -/
def found_caps : ℕ := 88

/-- The total number of bottle caps Nancy ends with -/
def total_caps : ℕ := initial_caps + found_caps

theorem nancy_bottle_caps : total_caps = 179 := by sorry

end nancy_bottle_caps_l1385_138545


namespace correct_operation_result_l1385_138503

theorem correct_operation_result (x : ℝ) : 
  ((x / 8) ^ 2 = 49) → ((x * 8) * 2 = 896) := by
  sorry

end correct_operation_result_l1385_138503


namespace janice_overtime_earnings_l1385_138563

/-- Represents Janice's work schedule and earnings --/
structure WorkSchedule where
  regularDays : ℕ
  regularPayPerDay : ℕ
  overtimeShifts : ℕ
  totalEarnings : ℕ

/-- Calculates the extra amount earned per overtime shift --/
def extraPerOvertimeShift (w : WorkSchedule) : ℕ :=
  (w.totalEarnings - w.regularDays * w.regularPayPerDay) / w.overtimeShifts

/-- Theorem stating that Janice's extra earnings per overtime shift is $15 --/
theorem janice_overtime_earnings :
  let w : WorkSchedule := {
    regularDays := 5,
    regularPayPerDay := 30,
    overtimeShifts := 3,
    totalEarnings := 195
  }
  extraPerOvertimeShift w = 15 := by
  sorry

end janice_overtime_earnings_l1385_138563


namespace roots_of_polynomial_l1385_138512

theorem roots_of_polynomial : ∀ x : ℝ,
  (x = (3 + Real.sqrt 5) / 2 ∨ x = (3 - Real.sqrt 5) / 2) →
  8 * x^5 - 45 * x^4 + 84 * x^3 - 84 * x^2 + 45 * x - 8 = 0 := by
  sorry

end roots_of_polynomial_l1385_138512


namespace common_chord_length_l1385_138570

-- Define the curves C₁ and C₂
def C₁ (x y : ℝ) : Prop := x^2 + y^2 - 2*x - 1 = 0
def C₂ (x y : ℝ) : Prop := x^2 + (y - 2)^2 = 4

-- Define the common chord
def common_chord (x y : ℝ) : Prop := 2*x - 4*y + 3 = 0

-- Theorem statement
theorem common_chord_length :
  ∃ (a b c d : ℝ),
    C₁ a b ∧ C₁ c d ∧ C₂ a b ∧ C₂ c d ∧
    common_chord a b ∧ common_chord c d ∧
    ((a - c)^2 + (b - d)^2) = 11 :=
sorry

end common_chord_length_l1385_138570


namespace real_part_of_z_z_equals_1_plus_2i_purely_imaginary_l1385_138529

-- Define a complex number z as x + yi
def z (x y : ℝ) : ℂ := Complex.mk x y

-- Statement 1: The real part of z is x
theorem real_part_of_z (x y : ℝ) : (z x y).re = x := by sorry

-- Statement 2: If z = 1 + 2i, then x = 1 and y = 2
theorem z_equals_1_plus_2i (x y : ℝ) : 
  z x y = Complex.mk 1 2 → x = 1 ∧ y = 2 := by sorry

-- Statement 3: When x = 0 and y ≠ 0, z is a purely imaginary number
theorem purely_imaginary (y : ℝ) : 
  y ≠ 0 → (z 0 y).re = 0 ∧ (z 0 y).im ≠ 0 := by sorry

end real_part_of_z_z_equals_1_plus_2i_purely_imaginary_l1385_138529


namespace hemisphere_base_area_l1385_138567

/-- Given a hemisphere with total surface area 9, prove its base area is 3 -/
theorem hemisphere_base_area (r : ℝ) (h : 3 * π * r^2 = 9) : π * r^2 = 3 := by
  sorry

end hemisphere_base_area_l1385_138567


namespace min_expression_bound_l1385_138575

theorem min_expression_bound (x y : ℝ) (hx : 0 < x ∧ x < 1) (hy : 0 < y ∧ y < 1) :
  min
    (min (x^2 + x*y + y^2) (x^2 + x*(y-1) + (y-1)^2))
    (min ((x-1)^2 + (x-1)*y + y^2) ((x-1)^2 + (x-1)*(y-1) + (y-1)^2))
  ≤ 1/3 := by
sorry

end min_expression_bound_l1385_138575


namespace function_decomposition_l1385_138514

/-- A non-negative function defined on [-3, 3] -/
def NonNegativeFunction := {f : ℝ → ℝ // ∀ x ∈ Set.Icc (-3 : ℝ) 3, f x ≥ 0}

/-- An even function defined on [-3, 3] -/
def EvenFunction := {f : ℝ → ℝ // ∀ x ∈ Set.Icc (-3 : ℝ) 3, f x = f (-x)}

/-- An odd function defined on [-3, 3] -/
def OddFunction := {f : ℝ → ℝ // ∀ x ∈ Set.Icc (-3 : ℝ) 3, f x = -f (-x)}

theorem function_decomposition
  (f : EvenFunction) (g : OddFunction)
  (h : ∀ x ∈ Set.Icc (-3 : ℝ) 3, f.val x + g.val x ≥ 2007 * x * Real.sqrt (9 - x^2) + x^2006) :
  ∃ p : NonNegativeFunction,
    (∀ x ∈ Set.Icc (-3 : ℝ) 3, f.val x = x^2006 + (p.val x + p.val (-x)) / 2) ∧
    (∀ x ∈ Set.Icc (-3 : ℝ) 3, g.val x = 2007 * x * Real.sqrt (9 - x^2) + (p.val x - p.val (-x)) / 2) :=
sorry

end function_decomposition_l1385_138514


namespace turns_result_in_opposite_direction_l1385_138585

/-- Two turns result in opposite direction if they are in the same direction and sum to 180 degrees -/
def opposite_direction (turn1 : ℝ) (turn2 : ℝ) : Prop :=
  (turn1 > 0 ∧ turn2 > 0) ∧ turn1 + turn2 = 180

/-- The specific turns given in the problem -/
def first_turn : ℝ := 53
def second_turn : ℝ := 127

/-- Theorem stating that the given turns result in opposite direction -/
theorem turns_result_in_opposite_direction :
  opposite_direction first_turn second_turn := by
  sorry

#check turns_result_in_opposite_direction

end turns_result_in_opposite_direction_l1385_138585


namespace inequality_proof_l1385_138596

theorem inequality_proof (a b c : ℝ) 
  (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c)
  (h_sum : a + b + c = 1) :
  a * Real.sqrt b + b * Real.sqrt c + c * Real.sqrt a ≤ 1 / Real.sqrt 3 := by
  sorry

end inequality_proof_l1385_138596


namespace mrs_brown_payment_l1385_138510

/-- Calculates the final price after applying multiple discounts --/
def calculate_final_price (base_price : ℝ) (mother_discount : ℝ) (child_discount : ℝ) (vip_discount : ℝ) : ℝ :=
  let price_after_mother := base_price * (1 - mother_discount)
  let price_after_child := price_after_mother * (1 - child_discount)
  price_after_child * (1 - vip_discount)

/-- Theorem stating that Mrs. Brown's final payment amount is $201.10 --/
theorem mrs_brown_payment : 
  let shoes_price : ℝ := 125
  let handbag_price : ℝ := 75
  let scarf_price : ℝ := 45
  let total_price : ℝ := shoes_price + handbag_price + scarf_price
  let mother_discount : ℝ := 0.10
  let child_discount : ℝ := 0.04
  let vip_discount : ℝ := 0.05
  calculate_final_price total_price mother_discount child_discount vip_discount = 201.10 := by
  sorry


end mrs_brown_payment_l1385_138510


namespace solve_for_t_l1385_138552

theorem solve_for_t (p : ℝ) (t : ℝ) 
  (h1 : 5 = p * 3^t) 
  (h2 : 45 = p * 9^t) : 
  t = 2 := by
  sorry

end solve_for_t_l1385_138552


namespace max_rabbits_with_traits_l1385_138574

theorem max_rabbits_with_traits (N : ℕ) 
  (long_ears : ℕ) (jump_far : ℕ) (both_traits : ℕ) :
  (long_ears = 13 ∧ jump_far = 17 ∧ both_traits ≥ 3) →
  (∀ n : ℕ, n > N → ∃ arrangement : ℕ × ℕ × ℕ, 
    arrangement.1 + arrangement.2.1 + arrangement.2.2 = n ∧
    arrangement.1 + arrangement.2.1 = long_ears ∧
    arrangement.1 + arrangement.2.2 = jump_far ∧
    arrangement.1 < both_traits) →
  N = 27 :=
sorry

end max_rabbits_with_traits_l1385_138574


namespace sum_of_coefficients_zero_l1385_138579

theorem sum_of_coefficients_zero (a : ℝ) (a₀ a₁ a₂ a₃ a₄ a₅ a₆ a₇ : ℝ) :
  (∀ x : ℝ, (x^2 + x + 1) * (2*x - a)^5 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5 + a₆*x^6 + a₇*x^7) →
  a₀ = -32 →
  a₀ + a₁ + a₂ + a₃ + a₄ + a₅ + a₆ + a₇ = 0 :=
by sorry

end sum_of_coefficients_zero_l1385_138579


namespace cookies_leftover_is_four_l1385_138533

/-- The number of cookies left over when selling in packs of 10 -/
def cookies_leftover (abigail beatrice carson : ℕ) : ℕ :=
  (abigail + beatrice + carson) % 10

/-- Theorem stating that the number of cookies left over is 4 -/
theorem cookies_leftover_is_four :
  cookies_leftover 53 65 26 = 4 := by
  sorry

end cookies_leftover_is_four_l1385_138533


namespace subset_P_l1385_138501

-- Define the set P
def P : Set ℝ := {x | x ≤ 3}

-- State the theorem
theorem subset_P : {-1} ⊆ P := by sorry

end subset_P_l1385_138501


namespace largest_red_socks_l1385_138532

/-- The largest number of red socks in a drawer with specific conditions -/
theorem largest_red_socks (total : ℕ) (red : ℕ) (blue : ℕ) 
  (h1 : total ≤ 1991)
  (h2 : total = red + blue)
  (h3 : (red * (red - 1) + blue * (blue - 1)) / (total * (total - 1)) = 1/2) :
  red ≤ 990 ∧ ∃ (r : ℕ), r = 990 ∧ 
    ∃ (t b : ℕ), t ≤ 1991 ∧ t = r + b ∧ 
      (r * (r - 1) + b * (b - 1)) / (t * (t - 1)) = 1/2 :=
by sorry

end largest_red_socks_l1385_138532


namespace store_profit_l1385_138517

/-- The profit made by the store selling New Year cards -/
theorem store_profit (cost_price : ℚ) (total_sales : ℚ) (n : ℕ) (selling_price : ℚ) : 
  cost_price = 21/100 ∧ 
  total_sales = 1457/100 ∧ 
  n * selling_price = total_sales ∧ 
  selling_price ≤ 2 * cost_price →
  n * (selling_price - cost_price) = 47/10 := by
  sorry

#check store_profit

end store_profit_l1385_138517


namespace expression_equality_l1385_138589

theorem expression_equality (x : ℝ) (hx : x > 0) :
  (∃! n : ℕ, n = (if 2 * x^(x+1) = x^(x+1) + x^(x+1) then 1 else 0) +
              (if x^(2*x+2) = x^(x+1) + x^(x+1) then 1 else 0) +
              (if (3*x)^x = x^(x+1) + x^(x+1) then 1 else 0) +
              (if (3*x)^(x+1) = x^(x+1) + x^(x+1) then 1 else 0)) ∧
  n = 1 :=
by sorry

end expression_equality_l1385_138589


namespace train_passing_time_l1385_138500

theorem train_passing_time (length1 length2 speed1 speed2 : ℝ) 
  (h1 : length1 = 350)
  (h2 : length2 = 450)
  (h3 : speed1 = 63 * 1000 / 3600)
  (h4 : speed2 = 81 * 1000 / 3600)
  (h5 : speed2 > speed1) :
  (length1 + length2) / (speed2 - speed1) = 160 :=
by sorry

end train_passing_time_l1385_138500


namespace probability_select_leaders_l1385_138595

def club_sizes : List Nat := [6, 8, 9]

def num_clubs : Nat := 3

def num_selected : Nat := 4

def num_co_presidents : Nat := 2

def num_vice_presidents : Nat := 1

theorem probability_select_leaders (club_sizes : List Nat) 
  (h1 : club_sizes = [6, 8, 9]) 
  (h2 : num_clubs = 3) 
  (h3 : num_selected = 4) 
  (h4 : num_co_presidents = 2) 
  (h5 : num_vice_presidents = 1) : 
  (1 / num_clubs) * (club_sizes.map (λ n => Nat.choose (n - (num_co_presidents + num_vice_presidents)) 1 / Nat.choose n num_selected)).sum = 67 / 630 := by
  sorry

end probability_select_leaders_l1385_138595


namespace min_value_abc_minus_b_l1385_138572

def S : Finset Int := {-10, -7, -3, 0, 4, 6, 9}

theorem min_value_abc_minus_b :
  (∃ (a b c : Int), a ∈ S ∧ b ∈ S ∧ c ∈ S ∧ a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
    a * b * c - b = -546) ∧
  (∀ (a b c : Int), a ∈ S → b ∈ S → c ∈ S → a ≠ b → b ≠ c → a ≠ c →
    a * b * c - b ≥ -546) :=
by sorry

end min_value_abc_minus_b_l1385_138572


namespace number_calculation_l1385_138571

theorem number_calculation (x : ℝ) : 0.25 * x = 0.20 * 650 + 190 → x = 1280 := by
  sorry

end number_calculation_l1385_138571


namespace stamps_received_l1385_138576

/-- Given Simon's initial and final stamp counts, prove he received 27 stamps from friends -/
theorem stamps_received (initial_stamps final_stamps : ℕ) 
  (h1 : initial_stamps = 34)
  (h2 : final_stamps = 61) :
  final_stamps - initial_stamps = 27 := by
  sorry

end stamps_received_l1385_138576


namespace no_perfect_square_sum_l1385_138519

theorem no_perfect_square_sum (x y z : ℤ) (h : x^2 + y^2 + z^2 = 1993) :
  ¬ ∃ (a : ℤ), x + y + z = a^2 := by
sorry

end no_perfect_square_sum_l1385_138519


namespace inequality_proof_l1385_138584

theorem inequality_proof (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (1 / (2 * a)) + (1 / (2 * b)) + (1 / (2 * c)) ≥ (1 / (a + b)) + (1 / (b + c)) + (1 / (c + a)) := by
  sorry

end inequality_proof_l1385_138584


namespace tim_seashells_l1385_138562

/-- The number of seashells Tim initially found -/
def initial_seashells : ℕ := 679

/-- The number of seashells Tim gave away -/
def seashells_given_away : ℕ := 172

/-- The number of seashells Tim has now -/
def remaining_seashells : ℕ := initial_seashells - seashells_given_away

theorem tim_seashells : remaining_seashells = 507 := by
  sorry

end tim_seashells_l1385_138562


namespace inequality_proof_l1385_138543

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (h_abc : a * b * c = 1) :
  (a - 1 + 1/b) * (b - 1 + 1/c) * (c - 1 + 1/a) ≤ 1 := by
  sorry

end inequality_proof_l1385_138543


namespace starting_number_with_20_multiples_of_5_l1385_138520

theorem starting_number_with_20_multiples_of_5 :
  (∃! n : ℕ, n ≤ 100 ∧ 
    (∃ s : Finset ℕ, s.card = 20 ∧ 
      (∀ m ∈ s, n ≤ m ∧ m ≤ 100 ∧ m % 5 = 0) ∧
      (∀ m : ℕ, n ≤ m ∧ m ≤ 100 ∧ m % 5 = 0 → m ∈ s)) ∧
    (∀ k : ℕ, k < n → 
      ¬(∃ s : Finset ℕ, s.card = 20 ∧ 
        (∀ m ∈ s, k ≤ m ∧ m ≤ 100 ∧ m % 5 = 0) ∧
        (∀ m : ℕ, k ≤ m ∧ m ≤ 100 ∧ m % 5 = 0 → m ∈ s)))) ∧
  (∀ n : ℕ, (∃! n : ℕ, n ≤ 100 ∧ 
    (∃ s : Finset ℕ, s.card = 20 ∧ 
      (∀ m ∈ s, n ≤ m ∧ m ≤ 100 ∧ m % 5 = 0) ∧
      (∀ m : ℕ, n ≤ m ∧ m ≤ 100 ∧ m % 5 = 0 → m ∈ s)) ∧
    (∀ k : ℕ, k < n → 
      ¬(∃ s : Finset ℕ, s.card = 20 ∧ 
        (∀ m ∈ s, k ≤ m ∧ m ≤ 100 ∧ m % 5 = 0) ∧
        (∀ m : ℕ, k ≤ m ∧ m ≤ 100 ∧ m % 5 = 0 → m ∈ s)))) → n = 10) :=
by sorry

end starting_number_with_20_multiples_of_5_l1385_138520


namespace special_triangle_properties_l1385_138592

/-- An acute triangle ABC with specific properties -/
structure SpecialTriangle where
  -- The sides of the triangle
  a : ℝ
  b : ℝ
  c : ℝ
  -- The angles of the triangle
  A : ℝ
  B : ℝ
  C : ℝ
  -- Properties of the triangle
  acute : 0 < A ∧ A < π/2 ∧ 0 < B ∧ B < π/2 ∧ 0 < C ∧ C < π/2
  law_of_sines : a / Real.sin A = b / Real.sin B
  special_relation : 2 * a * Real.sin B = Real.sqrt 3 * b
  side_a : a = Real.sqrt 7
  side_c : c = 2

/-- The main theorem about the special triangle -/
theorem special_triangle_properties (t : SpecialTriangle) :
  t.A = π/3 ∧ 
  (1/2 : ℝ) * t.b * t.c * Real.sin t.A = (3 * Real.sqrt 3) / 2 := by
  sorry

end special_triangle_properties_l1385_138592


namespace grocer_sales_l1385_138549

theorem grocer_sales (sales : List ℕ) (average : ℕ) : 
  sales = [800, 900, 1000, 800, 900] →
  average = 850 →
  (sales.sum + 700) / 6 = average →
  700 = 6 * average - sales.sum :=
by sorry

end grocer_sales_l1385_138549


namespace partition_properties_l1385_138544

/-- P k l n is the number of partitions of n into no more than k parts, each not exceeding l -/
def P (k l n : ℕ) : ℕ := sorry

/-- The four properties of the partition function P -/
theorem partition_properties (k l n : ℕ) :
  (P k l n - P k (l-1) n = P (k-1) l (n-l)) ∧
  (P k l n - P (k-1) l n = P k (l-1) (n-k)) ∧
  (P k l n = P l k n) ∧
  (P k l n = P k l (k*l-n)) := by sorry

end partition_properties_l1385_138544


namespace compare_irrational_expressions_l1385_138564

theorem compare_irrational_expressions : 2 * Real.sqrt 2 - Real.sqrt 7 < Real.sqrt 6 - Real.sqrt 5 := by
  sorry

end compare_irrational_expressions_l1385_138564


namespace whale_third_hour_consumption_l1385_138594

/-- Represents the whale's plankton consumption during a feeding frenzy -/
def WhaleFeedingFrenzy (x : ℕ) : Prop :=
  let first_hour := x
  let second_hour := x + 3
  let third_hour := x + 6
  let fourth_hour := x + 9
  let fifth_hour := x + 12
  (first_hour + second_hour + third_hour + fourth_hour + fifth_hour = 450) ∧
  (third_hour = 90)

/-- Theorem stating that the whale consumes 90 kilos in the third hour -/
theorem whale_third_hour_consumption : ∃ x : ℕ, WhaleFeedingFrenzy x := by
  sorry

end whale_third_hour_consumption_l1385_138594


namespace complex_fraction_equality_l1385_138511

theorem complex_fraction_equality (a b : ℂ) (ha : a ≠ 0) (hb : b ≠ 0) 
  (h : a^2 + a*b + b^2 = 0) : (a^6 + b^6) / (a + b)^6 = 18 := by
  sorry

end complex_fraction_equality_l1385_138511


namespace f_satisfies_conditions_l1385_138597

def f (x : ℝ) := x^2

theorem f_satisfies_conditions :
  (∀ x y, x < y ∧ y < -1 → f x > f y) ∧
  (∀ x, f x = f (-x)) ∧
  (∃ m, ∀ x, f m ≤ f x) := by
  sorry

end f_satisfies_conditions_l1385_138597


namespace perpendicular_parallel_imply_perpendicular_l1385_138524

-- Define the types for lines and planes
def Line : Type := ℝ → ℝ → ℝ → Prop
def Plane : Type := ℝ → ℝ → ℝ → Prop

-- Define the relations
def parallel (l₁ l₂ : Line) : Prop := sorry
def parallel_line_plane (l : Line) (p : Plane) : Prop := sorry
def perpendicular (l₁ l₂ : Line) : Prop := sorry
def perpendicular_line_plane (l : Line) (p : Plane) : Prop := sorry

-- Define non-coincident
def non_coincident_lines (l₁ l₂ : Line) : Prop := sorry
def non_coincident_planes (p₁ p₂ : Plane) : Prop := sorry

theorem perpendicular_parallel_imply_perpendicular 
  (a b : Line) (α : Plane) 
  (h_non_coincident_lines : non_coincident_lines a b)
  (h_non_coincident_planes : non_coincident_planes α β)
  (h1 : perpendicular_line_plane a α) 
  (h2 : parallel_line_plane b α) : 
  perpendicular a b :=
sorry

end perpendicular_parallel_imply_perpendicular_l1385_138524


namespace rate_of_current_l1385_138538

/-- Proves that given a man's downstream speed, upstream speed, and still water speed, 
    the rate of current can be calculated. -/
theorem rate_of_current 
  (downstream_speed : ℝ) 
  (upstream_speed : ℝ) 
  (still_water_speed : ℝ) 
  (h1 : downstream_speed = 45) 
  (h2 : upstream_speed = 23) 
  (h3 : still_water_speed = 34) : 
  downstream_speed - still_water_speed = 11 := by
  sorry

#check rate_of_current

end rate_of_current_l1385_138538


namespace complex_fraction_equality_l1385_138527

theorem complex_fraction_equality (a b : ℂ) (ha : a ≠ 0) (hb : b ≠ 0) 
  (h : a^2 + a*b + b^2 = 0) : 
  (a^12 + b^12) / (a + b)^9 = -2 * a^3 := by
  sorry

end complex_fraction_equality_l1385_138527


namespace min_weighings_nine_medals_l1385_138568

/-- Represents a set of medals with one heavier than the others -/
structure MedalSet :=
  (total : Nat)
  (heavier_exists : total > 0)

/-- Represents a balance scale used for weighing -/
structure BalanceScale

/-- The minimum number of weighings required to find the heavier medal -/
def min_weighings (medals : MedalSet) (scale : BalanceScale) : Nat :=
  sorry

/-- Theorem stating that for 9 medals, the minimum number of weighings is 2 -/
theorem min_weighings_nine_medals :
  ∀ (scale : BalanceScale),
  min_weighings ⟨9, by norm_num⟩ scale = 2 :=
sorry

end min_weighings_nine_medals_l1385_138568


namespace scarf_cost_is_two_l1385_138573

/-- The cost of a single scarf given Kiki's spending habits -/
def scarf_cost (total_money : ℚ) (num_scarves : ℕ) (hat_percentage : ℚ) : ℚ :=
  (1 - hat_percentage) * total_money / num_scarves

/-- Proof that the cost of each scarf is $2 -/
theorem scarf_cost_is_two :
  scarf_cost 90 18 (3/5) = 2 := by
  sorry

end scarf_cost_is_two_l1385_138573


namespace silver_dollars_problem_l1385_138550

theorem silver_dollars_problem (chiu phung ha : ℕ) : 
  phung = chiu + 16 →
  ha = phung + 5 →
  chiu + phung + ha = 205 →
  chiu = 56 := by
  sorry

end silver_dollars_problem_l1385_138550


namespace max_quotient_value_l1385_138505

theorem max_quotient_value (a b : ℝ) (ha : 300 ≤ a ∧ a ≤ 500) (hb : 900 ≤ b ∧ b ≤ 1800) :
  (∀ x y, 300 ≤ x ∧ x ≤ 500 → 900 ≤ y ∧ y ≤ 1800 → x / y ≤ a / b) →
  a / b = 5 / 9 :=
by sorry

end max_quotient_value_l1385_138505


namespace system_solution_l1385_138557

theorem system_solution (k : ℝ) : 
  (∃ x y : ℝ, x - 3*y = k + 2 ∧ x - y = 4 ∧ 3*x + y = -8) → k = 12 := by
  sorry

end system_solution_l1385_138557


namespace point_in_fourth_quadrant_l1385_138522

/-- Given the equation (x-2)^2 + √(y+1) = 0, prove that the point (x,y) lies in the fourth quadrant -/
theorem point_in_fourth_quadrant (x y : ℝ) (h : (x - 2)^2 + Real.sqrt (y + 1) = 0) :
  x > 0 ∧ y < 0 := by
  sorry

end point_in_fourth_quadrant_l1385_138522
