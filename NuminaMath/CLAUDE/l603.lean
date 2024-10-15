import Mathlib

namespace NUMINAMATH_CALUDE_ferry_speed_difference_l603_60308

/-- Proves the speed difference between two ferries given their travel conditions -/
theorem ferry_speed_difference :
  -- Ferry P's travel time
  let t_p : ℝ := 2 
  -- Ferry P's speed
  let v_p : ℝ := 8
  -- Ferry Q's route length multiplier
  let route_multiplier : ℝ := 3
  -- Additional time for Ferry Q's journey
  let additional_time : ℝ := 2

  -- Distance traveled by Ferry P
  let d_p : ℝ := t_p * v_p
  -- Distance traveled by Ferry Q
  let d_q : ℝ := route_multiplier * d_p
  -- Total time for Ferry Q's journey
  let t_q : ℝ := t_p + additional_time
  -- Speed of Ferry Q
  let v_q : ℝ := d_q / t_q

  -- The speed difference between Ferry Q and Ferry P is 4 km/hour
  v_q - v_p = 4 := by sorry

end NUMINAMATH_CALUDE_ferry_speed_difference_l603_60308


namespace NUMINAMATH_CALUDE_cylinder_section_area_l603_60340

/-- The area of a plane section in a cylinder --/
theorem cylinder_section_area (r h : ℝ) (arc_angle : ℝ) : 
  r = 8 → h = 10 → arc_angle = 150 * π / 180 →
  ∃ (area : ℝ), area = (400/3) * π + 40 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_cylinder_section_area_l603_60340


namespace NUMINAMATH_CALUDE_angle_y_value_l603_60377

-- Define the triangles and angles
def triangle_ABC (A B C : ℝ) : Prop := A + B + C = 180

def right_triangle (A B : ℝ) : Prop := A + B = 90

-- State the theorem
theorem angle_y_value :
  ∀ A B C D E y : ℝ,
  triangle_ABC 50 70 C →
  right_triangle D y →
  D = C →
  y = 30 :=
by sorry

end NUMINAMATH_CALUDE_angle_y_value_l603_60377


namespace NUMINAMATH_CALUDE_students_per_class_l603_60357

theorem students_per_class 
  (total_students : ℕ) 
  (num_classrooms : ℕ) 
  (h1 : total_students = 120) 
  (h2 : num_classrooms = 24) 
  (h3 : total_students % num_classrooms = 0) -- Ensures equal distribution
  : total_students / num_classrooms = 5 := by
sorry

end NUMINAMATH_CALUDE_students_per_class_l603_60357


namespace NUMINAMATH_CALUDE_distinct_roots_root_one_k_values_l603_60391

-- Define the quadratic equation
def quadratic (k x : ℝ) : ℝ := x^2 - (2*k + 1)*x + k^2 + k

-- Theorem 1: The equation has two distinct real roots for all k
theorem distinct_roots (k : ℝ) : 
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ quadratic k x₁ = 0 ∧ quadratic k x₂ = 0 :=
sorry

-- Theorem 2: When one root is 1, k is either 0 or 1
theorem root_one_k_values : 
  ∀ k : ℝ, quadratic k 1 = 0 → k = 0 ∨ k = 1 :=
sorry

end NUMINAMATH_CALUDE_distinct_roots_root_one_k_values_l603_60391


namespace NUMINAMATH_CALUDE_bottle_and_beverage_weight_l603_60329

/-- Given a bottle and some beverage, prove the weight of the original beverage and the bottle. -/
theorem bottle_and_beverage_weight 
  (original_beverage : ℝ) 
  (bottle : ℝ) 
  (h1 : 2 * original_beverage + bottle = 5) 
  (h2 : 4 * original_beverage + bottle = 9) : 
  original_beverage = 2 ∧ bottle = 1 := by
sorry

end NUMINAMATH_CALUDE_bottle_and_beverage_weight_l603_60329


namespace NUMINAMATH_CALUDE_two_numbers_ratio_problem_l603_60365

theorem two_numbers_ratio_problem (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  x / y = 3 →
  (x^2 + y^2) / (x + y) = 5 →
  x = 6 ∧ y = 2 := by
sorry

end NUMINAMATH_CALUDE_two_numbers_ratio_problem_l603_60365


namespace NUMINAMATH_CALUDE_division_problem_l603_60397

theorem division_problem (dividend : ℕ) (divisor : ℕ) (remainder : ℕ) (quotient : ℕ) :
  dividend = 729 →
  divisor = 38 →
  remainder = 7 →
  dividend = divisor * quotient + remainder →
  quotient = 19 := by
sorry

end NUMINAMATH_CALUDE_division_problem_l603_60397


namespace NUMINAMATH_CALUDE_seed_germination_problem_l603_60364

theorem seed_germination_problem (seeds_plot1 seeds_plot2 : ℕ) 
  (germination_rate_plot1 germination_rate_total : ℚ) :
  seeds_plot1 = 300 →
  seeds_plot2 = 200 →
  germination_rate_plot1 = 25 / 100 →
  germination_rate_total = 29 / 100 →
  ∃ (germination_rate_plot2 : ℚ),
    germination_rate_plot2 = 35 / 100 ∧
    (seeds_plot1 : ℚ) * germination_rate_plot1 + (seeds_plot2 : ℚ) * germination_rate_plot2 = 
    ((seeds_plot1 + seeds_plot2) : ℚ) * germination_rate_total :=
by sorry

end NUMINAMATH_CALUDE_seed_germination_problem_l603_60364


namespace NUMINAMATH_CALUDE_max_sum_abc_l603_60354

/-- Definition of A_n -/
def A_n (a n : ℕ) : ℕ := a * (10^(3*n) - 1) / 9

/-- Definition of B_n -/
def B_n (b n : ℕ) : ℕ := b * (10^(2*n) - 1) / 9

/-- Definition of C_n -/
def C_n (c n : ℕ) : ℕ := c * (10^(2*n) - 1) / 9

/-- The main theorem -/
theorem max_sum_abc :
  ∃ (a b c : ℕ), 1 ≤ a ∧ a ≤ 9 ∧
                 1 ≤ b ∧ b ≤ 9 ∧
                 1 ≤ c ∧ c ≤ 9 ∧
                 (∃ (n : ℕ), n > 0 ∧ C_n c n - B_n b n = (A_n a n)^2) ∧
                 a + b + c = 18 ∧
                 ∀ (a' b' c' : ℕ), 1 ≤ a' ∧ a' ≤ 9 ∧
                                   1 ≤ b' ∧ b' ≤ 9 ∧
                                   1 ≤ c' ∧ c' ≤ 9 ∧
                                   (∃ (n : ℕ), n > 0 ∧ C_n c' n - B_n b' n = (A_n a' n)^2) →
                                   a' + b' + c' ≤ 18 :=
by sorry

end NUMINAMATH_CALUDE_max_sum_abc_l603_60354


namespace NUMINAMATH_CALUDE_symmetric_point_coordinates_l603_60373

/-- A point in a 2D plane. -/
structure Point where
  x : ℝ
  y : ℝ

/-- Two points are symmetric with respect to the origin if their coordinates sum to zero. -/
def symmetricToOrigin (p q : Point) : Prop :=
  p.x + q.x = 0 ∧ p.y + q.y = 0

theorem symmetric_point_coordinates :
  let A : Point := ⟨2, 4⟩
  let B : Point := ⟨-2, -4⟩
  symmetricToOrigin A B → B = ⟨-2, -4⟩ := by
  sorry

end NUMINAMATH_CALUDE_symmetric_point_coordinates_l603_60373


namespace NUMINAMATH_CALUDE_product_from_hcf_lcm_l603_60337

theorem product_from_hcf_lcm (A B : ℕ+) :
  Nat.gcd A B = 22 →
  Nat.lcm A B = 2828 →
  A * B = 62216 := by
  sorry

end NUMINAMATH_CALUDE_product_from_hcf_lcm_l603_60337


namespace NUMINAMATH_CALUDE_simplify_fraction_product_l603_60328

theorem simplify_fraction_product :
  (1 : ℝ) / (1 + Real.sqrt 3) * (1 / (1 - Real.sqrt 5)) = 1 / (1 - Real.sqrt 5 + Real.sqrt 3 - Real.sqrt 15) := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_product_l603_60328


namespace NUMINAMATH_CALUDE_arthur_reading_challenge_l603_60369

/-- Arthur's summer reading challenge -/
theorem arthur_reading_challenge
  (total_goal : ℕ)
  (book1_pages : ℕ)
  (book2_pages : ℕ)
  (book1_read_percent : ℚ)
  (book2_read_fraction : ℚ)
  (h1 : total_goal = 800)
  (h2 : book1_pages = 500)
  (h3 : book2_pages = 1000)
  (h4 : book1_read_percent = 80 / 100)
  (h5 : book2_read_fraction = 1 / 5)
  : total_goal - (↑book1_pages * book1_read_percent + ↑book2_pages * book2_read_fraction) = 200 := by
  sorry

#check arthur_reading_challenge

end NUMINAMATH_CALUDE_arthur_reading_challenge_l603_60369


namespace NUMINAMATH_CALUDE_smallest_bob_number_l603_60382

def alice_number : ℕ := 36

def has_all_prime_factors_plus_one (n m : ℕ) : Prop :=
  ∀ p : ℕ, Prime p → (p ∣ n → p ∣ m) ∧
  ∃ q : ℕ, Prime q ∧ q ∣ m ∧ ¬(q ∣ n)

theorem smallest_bob_number :
  ∃ m : ℕ, has_all_prime_factors_plus_one alice_number m ∧
  ∀ k : ℕ, has_all_prime_factors_plus_one alice_number k → m ≤ k :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_bob_number_l603_60382


namespace NUMINAMATH_CALUDE_inequality_proof_l603_60300

theorem inequality_proof (a b c : ℝ) (n : ℕ) (p q r : ℕ) 
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (hn : p + q + r = n) :
  a^n + b^n + c^n ≥ a^p * b^q * c^r + a^q * b^r * c^p + a^r * b^p * c^q :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_l603_60300


namespace NUMINAMATH_CALUDE_line_equation_equivalence_l603_60380

/-- Given a line defined by (2, -1) · ((x, y) - (4, -3)) = 0, 
    prove that it's equivalent to y = 2x - 11 -/
theorem line_equation_equivalence :
  ∀ (x y : ℝ), 
  (2 * (x - 4) + (-1) * (y - (-3)) = 0) ↔ (y = 2 * x - 11) := by
sorry

end NUMINAMATH_CALUDE_line_equation_equivalence_l603_60380


namespace NUMINAMATH_CALUDE_volume_of_specific_pyramid_l603_60303

/-- A regular triangular pyramid with specific properties -/
structure RegularTriangularPyramid where
  /-- Distance from the midpoint of the height to the lateral face -/
  midpoint_to_face : ℝ
  /-- Distance from the midpoint of the height to the lateral edge -/
  midpoint_to_edge : ℝ

/-- The volume of a regular triangular pyramid -/
noncomputable def volume (p : RegularTriangularPyramid) : ℝ := sorry

/-- Theorem stating the volume of the specific regular triangular pyramid -/
theorem volume_of_specific_pyramid :
  ∀ (p : RegularTriangularPyramid),
    p.midpoint_to_face = 2 →
    p.midpoint_to_edge = Real.sqrt 12 →
    volume p = 216 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_volume_of_specific_pyramid_l603_60303


namespace NUMINAMATH_CALUDE_beads_per_necklace_l603_60394

def total_beads : ℕ := 52
def necklaces_made : ℕ := 26

theorem beads_per_necklace : 
  total_beads / necklaces_made = 2 := by sorry

end NUMINAMATH_CALUDE_beads_per_necklace_l603_60394


namespace NUMINAMATH_CALUDE_not_divisible_by_nine_l603_60351

theorem not_divisible_by_nine (t : ℤ) (k : ℤ) (h : k = 9 * t + 8) :
  ¬ (9 ∣ (5 * (9 * t + 8) * (9 * 25 * t + 222))) := by
  sorry

end NUMINAMATH_CALUDE_not_divisible_by_nine_l603_60351


namespace NUMINAMATH_CALUDE_ellipse_foci_distance_l603_60395

/-- Given an ellipse with equation 25x^2 - 100x + 4y^2 + 8y + 4 = 0, 
    the distance between its foci is 2√21. -/
theorem ellipse_foci_distance (x y : ℝ) : 
  25 * x^2 - 100 * x + 4 * y^2 + 8 * y + 4 = 0 → 
  ∃ (c : ℝ), c = 2 * Real.sqrt 21 ∧ 
  c = (distance_between_foci : ℝ → ℝ) (25 * x^2 - 100 * x + 4 * y^2 + 8 * y + 4) :=
by sorry


end NUMINAMATH_CALUDE_ellipse_foci_distance_l603_60395


namespace NUMINAMATH_CALUDE_isosceles_triangle_perimeter_isosceles_triangle_perimeter_proof_l603_60396

/-- An isosceles triangle with sides of 5 and 10 has a perimeter of 25. -/
theorem isosceles_triangle_perimeter : ℝ → ℝ → ℝ → Prop :=
  fun a b c =>
    (a = 5 ∧ b = 10 ∧ c = 10) →  -- Isosceles triangle with sides 5 and 10
    (a + b + c = 25)             -- Perimeter is 25

-- The proof is omitted
theorem isosceles_triangle_perimeter_proof : isosceles_triangle_perimeter 5 10 10 := by
  sorry

end NUMINAMATH_CALUDE_isosceles_triangle_perimeter_isosceles_triangle_perimeter_proof_l603_60396


namespace NUMINAMATH_CALUDE_log_expression_equals_one_l603_60315

-- Define the base-10 logarithm
noncomputable def log10 (x : ℝ) := Real.log x / Real.log 10

-- State the theorem
theorem log_expression_equals_one :
  (log10 5)^2 + log10 50 * log10 2 = 1 := by sorry

end NUMINAMATH_CALUDE_log_expression_equals_one_l603_60315


namespace NUMINAMATH_CALUDE_drug_price_reduction_l603_60358

theorem drug_price_reduction (x : ℝ) : 
  (36 : ℝ) * (1 - x)^2 = 25 ↔ 
  (∃ (price1 price2 : ℝ), 
    36 * (1 - x) = price1 ∧ 
    price1 * (1 - x) = price2 ∧ 
    price2 = 25) :=
by sorry

end NUMINAMATH_CALUDE_drug_price_reduction_l603_60358


namespace NUMINAMATH_CALUDE_willie_stickers_l603_60339

/-- Given that Willie starts with 124 stickers and gives away 43 stickers,
    prove that he ends up with 81 stickers. -/
theorem willie_stickers : ∀ (initial given_away final : ℕ),
  initial = 124 →
  given_away = 43 →
  final = initial - given_away →
  final = 81 := by
  sorry

end NUMINAMATH_CALUDE_willie_stickers_l603_60339


namespace NUMINAMATH_CALUDE_divisibility_criterion_l603_60367

theorem divisibility_criterion (x : ℕ) : 
  (x ≥ 10 ∧ x ≤ 99) →
  (1207 % x = 0 ↔ (x / 10)^3 + (x % 10)^3 = 344) :=
sorry

end NUMINAMATH_CALUDE_divisibility_criterion_l603_60367


namespace NUMINAMATH_CALUDE_point_trajectory_l603_60331

/-- The trajectory of a point with constant ratio between distances to axes -/
theorem point_trajectory (k : ℝ) (h : k ≠ 0) :
  ∀ x y : ℝ, x ≠ 0 →
  (|y| / |x| = k) ↔ (y = k * x ∨ y = -k * x) :=
by sorry

end NUMINAMATH_CALUDE_point_trajectory_l603_60331


namespace NUMINAMATH_CALUDE_t_shirt_cost_is_8_l603_60386

/-- The cost of a t-shirt in dollars -/
def t_shirt_cost : ℝ := sorry

/-- The total amount Timothy has to spend in dollars -/
def total_budget : ℝ := 50

/-- The cost of a bag in dollars -/
def bag_cost : ℝ := 10

/-- The number of t-shirts Timothy buys -/
def num_t_shirts : ℕ := 2

/-- The number of bags Timothy buys -/
def num_bags : ℕ := 2

/-- The number of key chains Timothy buys -/
def num_key_chains : ℕ := 21

/-- The cost of 3 key chains in dollars -/
def cost_3_key_chains : ℝ := 2

theorem t_shirt_cost_is_8 :
  t_shirt_cost = 8 ∧
  total_budget = num_t_shirts * t_shirt_cost + num_bags * bag_cost +
    (num_key_chains / 3 : ℝ) * cost_3_key_chains :=
by sorry

end NUMINAMATH_CALUDE_t_shirt_cost_is_8_l603_60386


namespace NUMINAMATH_CALUDE_max_value_of_expression_l603_60383

theorem max_value_of_expression (a b c : ℝ) (h : a^2 + b^2 + c^2 = 9) :
  (∃ (x y z : ℝ), x^2 + y^2 + z^2 = 9 ∧ (x - y)^2 + (y - z)^2 + (z - x)^2 > (a - b)^2 + (b - c)^2 + (c - a)^2) →
  (a - b)^2 + (b - c)^2 + (c - a)^2 ≤ 27 :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_expression_l603_60383


namespace NUMINAMATH_CALUDE_cos_is_valid_g_l603_60378

def a (x : ℝ) : ℝ × ℝ := (1, x)
def b (f : ℝ → ℝ) (x : ℝ) : ℝ × ℝ := (f x, -x)

def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

def is_even (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

theorem cos_is_valid_g (f : ℝ → ℝ) (g : ℝ → ℝ) :
  (∀ x, dot_product (a x) (b f x) = g x) →
  is_even f →
  g = cos :=
by sorry

end NUMINAMATH_CALUDE_cos_is_valid_g_l603_60378


namespace NUMINAMATH_CALUDE_negation_of_universal_proposition_l603_60332

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, |x - 2| + |x - 4| > 3) ↔ (∃ x : ℝ, |x - 2| + |x - 4| ≤ 3) := by sorry

end NUMINAMATH_CALUDE_negation_of_universal_proposition_l603_60332


namespace NUMINAMATH_CALUDE_two_digit_number_representation_l603_60342

/-- Represents a two-digit number with specific properties -/
def two_digit_number (x : ℕ) : ℕ :=
  10 * (2 * x^2) + x

/-- The theorem stating the correct representation of the two-digit number -/
theorem two_digit_number_representation (x : ℕ) : 
  two_digit_number x = 20 * x^2 + x :=
by sorry

end NUMINAMATH_CALUDE_two_digit_number_representation_l603_60342


namespace NUMINAMATH_CALUDE_inequality_and_equality_condition_l603_60352

theorem inequality_and_equality_condition (a b c : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) : 
  (a^2 + b^2 + c^2 + (1/a + 1/b + 1/c)^2 ≥ 6 * Real.sqrt 3) ∧ 
  (a^2 + b^2 + c^2 + (1/a + 1/b + 1/c)^2 = 6 * Real.sqrt 3 ↔ 
   a = Real.sqrt (Real.sqrt 3) ∧ b = Real.sqrt (Real.sqrt 3) ∧ c = Real.sqrt (Real.sqrt 3)) :=
by sorry

end NUMINAMATH_CALUDE_inequality_and_equality_condition_l603_60352


namespace NUMINAMATH_CALUDE_expression_simplification_l603_60379

theorem expression_simplification (a : ℚ) (h : a = -1/2) :
  (a + 2)^2 + (a + 2) * (2 - a) - 6 * a = 9 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l603_60379


namespace NUMINAMATH_CALUDE_circle_radius_calculation_l603_60305

theorem circle_radius_calculation (d PQ QR : ℝ) (h1 : d = 15) (h2 : PQ = 10) (h3 : QR = 8) :
  ∃ r : ℝ, r = 3 * Real.sqrt 5 ∧ PQ * (PQ + QR) = (d - r) * (d + r) := by
  sorry

end NUMINAMATH_CALUDE_circle_radius_calculation_l603_60305


namespace NUMINAMATH_CALUDE_f_composition_pi_12_l603_60376

noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 0 then 4 * x^2 - 1 else Real.sin x^2 - Real.cos x^2

theorem f_composition_pi_12 : f (f (π / 12)) = 2 := by
  sorry

end NUMINAMATH_CALUDE_f_composition_pi_12_l603_60376


namespace NUMINAMATH_CALUDE_f_2008_equals_zero_l603_60370

-- Define an odd function f
def odd_function (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

-- Define the property f(1-x) = f(1+x)
def symmetric_around_one (f : ℝ → ℝ) : Prop := ∀ x, f (1-x) = f (1+x)

theorem f_2008_equals_zero 
  (f : ℝ → ℝ) 
  (h_odd : odd_function f) 
  (h_sym : symmetric_around_one f) : 
  f 2008 = 0 := by
sorry

end NUMINAMATH_CALUDE_f_2008_equals_zero_l603_60370


namespace NUMINAMATH_CALUDE_product_of_difference_and_sum_of_squares_l603_60388

theorem product_of_difference_and_sum_of_squares (a b : ℝ) 
  (h1 : a - b = 3) 
  (h2 : a^2 + b^2 = 35) : 
  a * b = 13 := by
sorry

end NUMINAMATH_CALUDE_product_of_difference_and_sum_of_squares_l603_60388


namespace NUMINAMATH_CALUDE_binomial_variance_10_2_5_l603_60393

/-- The variance of a binomial distribution B(n, p) -/
def binomial_variance (n : ℕ) (p : ℝ) : ℝ := n * p * (1 - p)

/-- Theorem: The variance of a binomial distribution B(10, 2/5) is 12/5 -/
theorem binomial_variance_10_2_5 :
  binomial_variance 10 (2/5) = 12/5 := by
  sorry

end NUMINAMATH_CALUDE_binomial_variance_10_2_5_l603_60393


namespace NUMINAMATH_CALUDE_complex_magnitude_equation_l603_60335

theorem complex_magnitude_equation (z : ℂ) :
  Complex.abs z * (3 * z + 2 * Complex.I) = 2 * (Complex.I * z - 6) →
  Complex.abs z = 2 := by
sorry

end NUMINAMATH_CALUDE_complex_magnitude_equation_l603_60335


namespace NUMINAMATH_CALUDE_students_not_reading_l603_60375

theorem students_not_reading (total_students : ℕ) (girls : ℕ) (boys : ℕ) 
  (girls_reading_fraction : ℚ) (boys_reading_fraction : ℚ) :
  total_students = girls + boys →
  girls = 12 →
  boys = 10 →
  girls_reading_fraction = 5/6 →
  boys_reading_fraction = 4/5 →
  total_students - (↑girls * girls_reading_fraction).floor - (↑boys * boys_reading_fraction).floor = 4 := by
  sorry

end NUMINAMATH_CALUDE_students_not_reading_l603_60375


namespace NUMINAMATH_CALUDE_hall_volume_l603_60348

theorem hall_volume (length width height : ℝ) : 
  length = 15 ∧ 
  width = 12 ∧ 
  2 * (length * width) = 2 * (length * height) + 2 * (width * height) → 
  length * width * height = 8004 :=
by sorry

end NUMINAMATH_CALUDE_hall_volume_l603_60348


namespace NUMINAMATH_CALUDE_expression_evaluation_l603_60349

theorem expression_evaluation (a b : ℤ) (ha : a = 1) (hb : b = -1) :
  5 * a * b^2 - (3 * a * b + 2 * (-2 * a * b^2 + a * b)) = 14 := by sorry

end NUMINAMATH_CALUDE_expression_evaluation_l603_60349


namespace NUMINAMATH_CALUDE_log_sum_theorem_l603_60317

theorem log_sum_theorem (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h1 : Real.log x / Real.log y + Real.log y / Real.log x = 4) 
  (h2 : x * y = 64) : 
  (x + y) / 2 = (64^(3/(5+Real.sqrt 3)) + 64^(1/(5+Real.sqrt 3))) / 2 := by
sorry

end NUMINAMATH_CALUDE_log_sum_theorem_l603_60317


namespace NUMINAMATH_CALUDE_freshman_psych_majors_percentage_l603_60368

theorem freshman_psych_majors_percentage
  (total_students : ℕ)
  (freshman_ratio : ℚ)
  (liberal_arts_ratio : ℚ)
  (psych_major_ratio : ℚ)
  (h1 : freshman_ratio = 2/5)
  (h2 : liberal_arts_ratio = 1/2)
  (h3 : psych_major_ratio = 1/2)
  : (freshman_ratio * liberal_arts_ratio * psych_major_ratio : ℚ) = 1/10 := by
  sorry

#check freshman_psych_majors_percentage

end NUMINAMATH_CALUDE_freshman_psych_majors_percentage_l603_60368


namespace NUMINAMATH_CALUDE_x_value_proof_l603_60359

theorem x_value_proof (x y z a b c : ℝ) 
  (ha : xy / (x - y) = a)
  (hb : xz / (x - z) = b)
  (hc : yz / (y - z) = c)
  (ha_nonzero : a ≠ 0)
  (hb_nonzero : b ≠ 0)
  (hc_nonzero : c ≠ 0) :
  x = 2 * a * b * c / (a * b + b * c + a * c) :=
sorry

end NUMINAMATH_CALUDE_x_value_proof_l603_60359


namespace NUMINAMATH_CALUDE_reciprocal_and_opposite_l603_60350

-- Define the reciprocal function
def reciprocal (x : ℚ) : ℚ := 1 / x

-- Define the opposite function
def opposite (x : ℝ) : ℝ := -x

-- Theorem statement
theorem reciprocal_and_opposite :
  (reciprocal (2 / 3) = 3 / 2) ∧ (opposite (-2.5) = 2.5) := by
  sorry

end NUMINAMATH_CALUDE_reciprocal_and_opposite_l603_60350


namespace NUMINAMATH_CALUDE_f_inequality_l603_60316

open Real

-- Define the function f on (0, +∞)
variable (f : ℝ → ℝ)

-- Define the derivative of f
variable (f' : ℝ → ℝ)

-- State that f' is indeed the derivative of f
variable (hf' : ∀ x, x > 0 → HasDerivAt f (f' x) x)

-- State the condition xf'(x) > 2f(x)
variable (h_cond : ∀ x, x > 0 → x * f' x > 2 * f x)

-- Define a and b
variable (a b : ℝ)

-- State that a > b > 0
variable (hab : a > b ∧ b > 0)

-- Theorem statement
theorem f_inequality : b^2 * f a > a^2 * f b := by
  sorry

end NUMINAMATH_CALUDE_f_inequality_l603_60316


namespace NUMINAMATH_CALUDE_reynas_lamps_l603_60362

/-- The number of light bulbs in each lamp -/
def bulbs_per_lamp : ℕ := 7

/-- The fraction of lamps with burnt-out bulbs -/
def fraction_with_burnt_bulbs : ℚ := 1 / 4

/-- The number of burnt-out bulbs in lamps with burnt-out bulbs -/
def burnt_bulbs_per_lamp : ℕ := 2

/-- The total number of working light bulbs -/
def total_working_bulbs : ℕ := 130

/-- The number of lamps Reyna has -/
def num_lamps : ℕ := 20

theorem reynas_lamps :
  (bulbs_per_lamp * num_lamps : ℚ) * (1 - fraction_with_burnt_bulbs) +
  (bulbs_per_lamp - burnt_bulbs_per_lamp : ℚ) * num_lamps * fraction_with_burnt_bulbs =
  total_working_bulbs := by sorry

end NUMINAMATH_CALUDE_reynas_lamps_l603_60362


namespace NUMINAMATH_CALUDE_probability_theorem_l603_60345

/-- Represents a unit cube with a certain number of painted faces -/
structure UnitCube where
  painted_faces : Nat

/-- Represents the large cube composed of unit cubes -/
def LargeCube : Type := List UnitCube

/-- Creates a large cube with the given specifications -/
def create_large_cube : LargeCube :=
  -- 8 cubes with 3 painted faces
  (List.replicate 8 ⟨3⟩) ++
  -- 18 cubes with 2 painted faces
  (List.replicate 18 ⟨2⟩) ++
  -- 27 cubes with 1 painted face
  (List.replicate 27 ⟨1⟩) ++
  -- Remaining cubes with 0 painted faces
  (List.replicate 72 ⟨0⟩)

/-- Calculates the probability of selecting one cube with 3 painted faces
    and one cube with 1 painted face when choosing 2 cubes at random -/
def probability_3_and_1 (cube : LargeCube) : Rat :=
  let total_combinations := (List.length cube).choose 2
  let favorable_outcomes := (cube.filter (λ c => c.painted_faces = 3)).length *
                            (cube.filter (λ c => c.painted_faces = 1)).length
  favorable_outcomes / total_combinations

/-- The main theorem to prove -/
theorem probability_theorem :
  probability_3_and_1 create_large_cube = 216 / 7750 := by
  sorry

end NUMINAMATH_CALUDE_probability_theorem_l603_60345


namespace NUMINAMATH_CALUDE_license_plate_count_l603_60302

/-- The number of consonants in the English alphabet (including Y) -/
def num_consonants : ℕ := 21

/-- The number of vowels in the English alphabet -/
def num_vowels : ℕ := 5

/-- The number of digits (0 through 9) -/
def num_digits : ℕ := 10

/-- The total number of possible license plates -/
def total_plates : ℕ := num_consonants * num_consonants * num_vowels * num_vowels * num_digits

theorem license_plate_count : total_plates = 110250 := by sorry

end NUMINAMATH_CALUDE_license_plate_count_l603_60302


namespace NUMINAMATH_CALUDE_share_percentage_problem_l603_60327

theorem share_percentage_problem (total z y x : ℝ) : 
  total = 740 →
  z = 200 →
  y = 1.2 * z →
  x = total - y - z →
  (x - y) / y * 100 = 25 := by
  sorry

end NUMINAMATH_CALUDE_share_percentage_problem_l603_60327


namespace NUMINAMATH_CALUDE_problem_solution_l603_60346

def f (x : ℝ) : ℝ := |x - 1| - |x|

theorem problem_solution :
  let u := f (5/16)
  let v := f u
  let s := f v
  s = 1/2 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l603_60346


namespace NUMINAMATH_CALUDE_mikes_marbles_l603_60336

/-- Given that Mike has 8 orange marbles initially and gives away 4 marbles,
    prove that he will have 4 orange marbles remaining. -/
theorem mikes_marbles (initial_marbles : ℕ) (marbles_given : ℕ) (remaining_marbles : ℕ) :
  initial_marbles = 8 →
  marbles_given = 4 →
  remaining_marbles = initial_marbles - marbles_given →
  remaining_marbles = 4 := by
sorry

end NUMINAMATH_CALUDE_mikes_marbles_l603_60336


namespace NUMINAMATH_CALUDE_five_spiders_make_five_webs_l603_60301

/-- The number of webs made by a given number of spiders in 5 days -/
def webs_made (num_spiders : ℕ) : ℕ :=
  num_spiders * 1

/-- Theorem stating that 5 spiders make 5 webs in 5 days -/
theorem five_spiders_make_five_webs :
  webs_made 5 = 5 := by
  sorry

end NUMINAMATH_CALUDE_five_spiders_make_five_webs_l603_60301


namespace NUMINAMATH_CALUDE_whiskers_cat_school_total_l603_60310

/-- Represents the number of cats that can perform a specific trick or combination of tricks -/
structure CatTricks where
  jump : ℕ
  sit : ℕ
  playDead : ℕ
  fetch : ℕ
  jumpSit : ℕ
  sitPlayDead : ℕ
  playDeadFetch : ℕ
  fetchJump : ℕ
  jumpSitPlayDead : ℕ
  sitPlayDeadFetch : ℕ
  playDeadFetchJump : ℕ
  jumpFetchSit : ℕ
  allFour : ℕ
  none : ℕ

/-- Calculates the total number of cats in the Whisker's Cat School -/
def totalCats (tricks : CatTricks) : ℕ :=
  let exclusiveJump := tricks.jump - (tricks.jumpSit + tricks.fetchJump + tricks.jumpSitPlayDead + tricks.allFour)
  let exclusiveSit := tricks.sit - (tricks.jumpSit + tricks.sitPlayDead + tricks.jumpSitPlayDead + tricks.allFour)
  let exclusivePlayDead := tricks.playDead - (tricks.sitPlayDead + tricks.playDeadFetch + tricks.jumpSitPlayDead + tricks.allFour)
  let exclusiveFetch := tricks.fetch - (tricks.playDeadFetch + tricks.fetchJump + tricks.sitPlayDeadFetch + tricks.allFour)
  exclusiveJump + exclusiveSit + exclusivePlayDead + exclusiveFetch +
  tricks.jumpSit + tricks.sitPlayDead + tricks.playDeadFetch + tricks.fetchJump +
  tricks.jumpSitPlayDead + tricks.sitPlayDeadFetch + tricks.playDeadFetchJump + tricks.jumpFetchSit +
  tricks.allFour + tricks.none

/-- The specific number of cats for each trick or combination at the Whisker's Cat School -/
def whiskersCatSchool : CatTricks :=
  { jump := 60
  , sit := 40
  , playDead := 35
  , fetch := 45
  , jumpSit := 20
  , sitPlayDead := 15
  , playDeadFetch := 10
  , fetchJump := 18
  , jumpSitPlayDead := 5
  , sitPlayDeadFetch := 3
  , playDeadFetchJump := 7
  , jumpFetchSit := 10
  , allFour := 2
  , none := 12 }

/-- Theorem stating that the total number of cats at the Whisker's Cat School is 143 -/
theorem whiskers_cat_school_total : totalCats whiskersCatSchool = 143 := by
  sorry

end NUMINAMATH_CALUDE_whiskers_cat_school_total_l603_60310


namespace NUMINAMATH_CALUDE_units_digit_of_sum_factorials_l603_60318

def factorial (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

def units_digit (n : ℕ) : ℕ :=
  n % 10

theorem units_digit_of_sum_factorials : 
  units_digit (3 * (factorial 1 + factorial 2 + factorial 3 + factorial 4)) = 9 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_sum_factorials_l603_60318


namespace NUMINAMATH_CALUDE_subcommittee_count_l603_60322

theorem subcommittee_count (n m k : ℕ) (h1 : n = 12) (h2 : m = 5) (h3 : k = 5) :
  (Nat.choose n k) - (Nat.choose (n - m) k) = 771 := by
  sorry

end NUMINAMATH_CALUDE_subcommittee_count_l603_60322


namespace NUMINAMATH_CALUDE_product_ABC_l603_60387

def A : ℂ := 6 + 3 * Complex.I
def B : ℂ := 2 * Complex.I
def C : ℂ := 6 - 3 * Complex.I

theorem product_ABC : A * B * C = 90 * Complex.I := by
  sorry

end NUMINAMATH_CALUDE_product_ABC_l603_60387


namespace NUMINAMATH_CALUDE_negative_two_squared_l603_60363

theorem negative_two_squared : -2^2 = -4 := by
  sorry

end NUMINAMATH_CALUDE_negative_two_squared_l603_60363


namespace NUMINAMATH_CALUDE_smallest_positive_integer_with_remainders_l603_60313

theorem smallest_positive_integer_with_remainders : ∃ b : ℕ, 
  b > 0 ∧ 
  b % 4 = 3 ∧ 
  b % 6 = 5 ∧ 
  (∀ c : ℕ, c > 0 ∧ c % 4 = 3 ∧ c % 6 = 5 → b ≤ c) ∧
  b = 11 :=
by sorry

end NUMINAMATH_CALUDE_smallest_positive_integer_with_remainders_l603_60313


namespace NUMINAMATH_CALUDE_union_of_A_and_B_l603_60399

-- Define set A
def A : Set ℝ := {x | x^2 - 4*x - 5 = 0}

-- Define set B
def B : Set ℝ := {x | x^2 - 1 = 0}

-- Theorem statement
theorem union_of_A_and_B : A ∪ B = {-1, 1, 5} := by
  sorry

end NUMINAMATH_CALUDE_union_of_A_and_B_l603_60399


namespace NUMINAMATH_CALUDE_geometric_mean_point_existence_l603_60321

/-- In a triangle ABC, point D on BC exists such that AD is the geometric mean of BD and DC
    if and only if b + c ≤ a√2, where a = BC, b = AC, and c = AB. -/
theorem geometric_mean_point_existence (a b c : ℝ) (h_pos : a > 0 ∧ b > 0 ∧ c > 0) 
  (h_triangle : a < b + c ∧ b < a + c ∧ c < a + b) :
  (∃ (t : ℝ), 0 < t ∧ t < a ∧ 
    (b^2 * t * (a - t) = a * (a - t) * t)) ↔ b + c ≤ a * Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_geometric_mean_point_existence_l603_60321


namespace NUMINAMATH_CALUDE_chocolates_bought_l603_60325

theorem chocolates_bought (cost_price selling_price : ℝ) (num_bought : ℕ) : 
  (num_bought * cost_price = 21 * selling_price) →
  ((selling_price - cost_price) / cost_price * 100 = 66.67) →
  num_bought = 35 := by
sorry

end NUMINAMATH_CALUDE_chocolates_bought_l603_60325


namespace NUMINAMATH_CALUDE_difference_c_minus_a_l603_60374

theorem difference_c_minus_a (a b c : ℝ) : 
  (a + b) / 2 = 30 → c - a = 60 → c - a = 60 := by
  sorry

end NUMINAMATH_CALUDE_difference_c_minus_a_l603_60374


namespace NUMINAMATH_CALUDE_estimate_larger_than_original_l603_60324

theorem estimate_larger_than_original 
  (u v δ γ : ℝ) 
  (hu_pos : u > 0) 
  (hv_pos : v > 0) 
  (huv : u > v) 
  (hδγ : δ > γ) 
  (hγ_pos : γ > 0) : 
  (u + δ) - (v - γ) > u - v := by
sorry

end NUMINAMATH_CALUDE_estimate_larger_than_original_l603_60324


namespace NUMINAMATH_CALUDE_board_number_after_hour_l603_60398

def digit_product (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) * digit_product (n / 10)

def next_number (n : ℕ) : ℕ :=
  digit_product n + 15

def iterate_operation (n : ℕ) (iterations : ℕ) : ℕ :=
  match iterations with
  | 0 => n
  | k + 1 => iterate_operation (next_number n) k

theorem board_number_after_hour (initial : ℕ) (h : initial = 98) :
  iterate_operation initial 60 = 24 :=
sorry

end NUMINAMATH_CALUDE_board_number_after_hour_l603_60398


namespace NUMINAMATH_CALUDE_right_triangle_hypotenuse_l603_60333

theorem right_triangle_hypotenuse (a b : ℂ) (z₁ z₂ z₃ : ℂ) : 
  (z₁^3 + a*z₁ + b = 0) → 
  (z₂^3 + a*z₂ + b = 0) → 
  (z₃^3 + a*z₃ + b = 0) → 
  (Complex.abs z₁)^2 + (Complex.abs z₂)^2 + (Complex.abs z₃)^2 = 250 →
  ∃ (x y : ℝ), (x^2 + y^2 = (Complex.abs (z₁ - z₂))^2) ∧ 
                (x^2 = (Complex.abs (z₂ - z₃))^2 ∨ y^2 = (Complex.abs (z₂ - z₃))^2) →
  (Complex.abs (z₁ - z₂))^2 + (Complex.abs (z₂ - z₃))^2 + (Complex.abs (z₃ - z₁))^2 = 2 * ((5 * Real.sqrt 15)^2) :=
by sorry

end NUMINAMATH_CALUDE_right_triangle_hypotenuse_l603_60333


namespace NUMINAMATH_CALUDE_fraction_simplification_l603_60341

theorem fraction_simplification (x : ℝ) :
  (2 * x^2 + 3) / 4 - (5 - 4 * x^2) / 6 = (14 * x^2 - 1) / 12 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l603_60341


namespace NUMINAMATH_CALUDE_complex_function_property_l603_60392

open Complex

/-- Given a function f(z) = (a+bi)z where a and b are positive real numbers,
    if f(z) is equidistant from z and 2+2i for all complex z, and |a+bi| = 10,
    then b^2 = 287/17 -/
theorem complex_function_property (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (∀ z : ℂ, ‖(a + b * I) * z - z‖ = ‖(a + b * I) * z - (2 + 2 * I)‖) →
  ‖(a : ℂ) + b * I‖ = 10 →
  b^2 = 287/17 := by
  sorry

end NUMINAMATH_CALUDE_complex_function_property_l603_60392


namespace NUMINAMATH_CALUDE_hedge_cost_proof_l603_60338

/-- The number of concrete blocks used in each section of the hedge. -/
def blocks_per_section : ℕ := 30

/-- The cost of each concrete block in dollars. -/
def cost_per_block : ℕ := 2

/-- The number of sections in the hedge. -/
def number_of_sections : ℕ := 8

/-- The total cost of concrete blocks for the hedge. -/
def total_cost : ℕ := blocks_per_section * number_of_sections * cost_per_block

theorem hedge_cost_proof : total_cost = 480 := by
  sorry

end NUMINAMATH_CALUDE_hedge_cost_proof_l603_60338


namespace NUMINAMATH_CALUDE_cube_inequality_l603_60307

theorem cube_inequality (a b : ℝ) (h : a < 0 ∧ 0 < b) : a^3 < b^3 := by
  sorry

end NUMINAMATH_CALUDE_cube_inequality_l603_60307


namespace NUMINAMATH_CALUDE_translation_problem_l603_60320

/-- A translation in the complex plane -/
def ComplexTranslation (w : ℂ) : ℂ → ℂ := fun z ↦ z + w

theorem translation_problem (T : ℂ → ℂ) (h : T = ComplexTranslation (3 + 5*I)) :
  T (3 - I) = 6 + 4*I := by
  sorry

end NUMINAMATH_CALUDE_translation_problem_l603_60320


namespace NUMINAMATH_CALUDE_power_function_value_l603_60360

-- Define a power function that passes through (2, 8)
def f : ℝ → ℝ := fun x ↦ x^3

-- Theorem statement
theorem power_function_value : f 2 = 8 ∧ f (-3) = -27 := by
  sorry


end NUMINAMATH_CALUDE_power_function_value_l603_60360


namespace NUMINAMATH_CALUDE_min_reciprocal_sum_l603_60389

theorem min_reciprocal_sum (a b : ℝ) (h1 : a * b > 0) (h2 : a + 4 * b = 1) :
  (∀ x y : ℝ, x * y > 0 ∧ x + 4 * y = 1 → 1 / a + 1 / b ≤ 1 / x + 1 / y) ∧
  (∃ x y : ℝ, x * y > 0 ∧ x + 4 * y = 1 ∧ 1 / x + 1 / y = 9) :=
sorry

end NUMINAMATH_CALUDE_min_reciprocal_sum_l603_60389


namespace NUMINAMATH_CALUDE_min_value_a_l603_60309

theorem min_value_a (a : ℝ) (ha : a > 0) : 
  (∀ x y : ℝ, x > 0 → y > 0 → (x + y) * (a / x + 4 / y) ≥ 16) ↔ a ≥ 4 := by
  sorry

end NUMINAMATH_CALUDE_min_value_a_l603_60309


namespace NUMINAMATH_CALUDE_max_sum_at_9_l603_60319

/-- An arithmetic sequence with first term 1 and common difference d -/
def arithmetic_sequence (d : ℚ) : ℕ → ℚ := λ n => 1 + (n - 1 : ℚ) * d

/-- The sum of the first n terms of the arithmetic sequence -/
def S (d : ℚ) (n : ℕ) : ℚ := (n : ℚ) * (2 + (n - 1 : ℚ) * d) / 2

/-- The theorem stating that Sn reaches its maximum when n = 9 -/
theorem max_sum_at_9 (d : ℚ) (h : -2/17 < d ∧ d < -1/9) :
  ∀ k : ℕ, S d 9 ≥ S d k :=
sorry

end NUMINAMATH_CALUDE_max_sum_at_9_l603_60319


namespace NUMINAMATH_CALUDE_tourism_revenue_scientific_notation_l603_60366

theorem tourism_revenue_scientific_notation :
  let revenue_billion : ℝ := 1480.56
  let scientific_notation : ℝ := 1.48056 * (10 ^ 11)
  revenue_billion * (10 ^ 9) = scientific_notation :=
by sorry

end NUMINAMATH_CALUDE_tourism_revenue_scientific_notation_l603_60366


namespace NUMINAMATH_CALUDE_even_a_iff_xor_inequality_l603_60372

-- Define bitwise XOR operation
def bitwiseXOR (a b : ℕ) : ℕ :=
  Nat.rec 0 (fun k res => 
    if (a / 2^k + b / 2^k - res / 2^k) % 2 = 0 
    then res 
    else res + 2^k) a

-- State the theorem
theorem even_a_iff_xor_inequality (a : ℕ) : 
  (a > 0 ∧ a % 2 = 0) ↔ 
  (∀ x y : ℕ, x > y → bitwiseXOR x (a * x) ≠ bitwiseXOR y (a * y)) := by
sorry

end NUMINAMATH_CALUDE_even_a_iff_xor_inequality_l603_60372


namespace NUMINAMATH_CALUDE_competition_results_l603_60314

def team_a_scores : List ℝ := [7, 8, 9, 7, 10, 10, 9, 10, 10, 10]
def team_b_scores : List ℝ := [10, 8, 7, 9, 8, 10, 10, 9, 10, 9]

def median (scores : List ℝ) : ℝ := sorry
def mode (scores : List ℝ) : ℝ := sorry
def average (scores : List ℝ) : ℝ := sorry
def variance (scores : List ℝ) : ℝ := sorry

theorem competition_results :
  median team_a_scores = 9.5 ∧
  mode team_b_scores = 10 ∧
  average team_b_scores = 9 ∧
  variance team_b_scores = 1 ∧
  variance team_a_scores = 1.4 ∧
  variance team_b_scores < variance team_a_scores :=
by sorry

end NUMINAMATH_CALUDE_competition_results_l603_60314


namespace NUMINAMATH_CALUDE_gcd_n_cube_plus_eight_and_n_plus_three_l603_60356

theorem gcd_n_cube_plus_eight_and_n_plus_three (n : ℕ) (h : n > 27) : 
  Nat.gcd (n^3 + 8) (n + 3) = 1 := by
sorry

end NUMINAMATH_CALUDE_gcd_n_cube_plus_eight_and_n_plus_three_l603_60356


namespace NUMINAMATH_CALUDE_total_dice_count_l603_60344

theorem total_dice_count (ivan_dice : ℕ) (jerry_dice : ℕ) : 
  ivan_dice = 20 → 
  jerry_dice = 2 * ivan_dice → 
  ivan_dice + jerry_dice = 60 := by
  sorry

end NUMINAMATH_CALUDE_total_dice_count_l603_60344


namespace NUMINAMATH_CALUDE_fraction_equality_l603_60353

theorem fraction_equality (q r s t : ℚ) 
  (h1 : q / r = 8)
  (h2 : s / r = 4)
  (h3 : s / t = 1 / 3) :
  t / q = 3 / 2 := by
sorry

end NUMINAMATH_CALUDE_fraction_equality_l603_60353


namespace NUMINAMATH_CALUDE_bottle_cap_distribution_l603_60326

theorem bottle_cap_distribution (total_caps : ℕ) (num_groups : ℕ) (caps_per_group : ℕ) : 
  total_caps = 35 → num_groups = 7 → total_caps = num_groups * caps_per_group → caps_per_group = 5 := by
  sorry

end NUMINAMATH_CALUDE_bottle_cap_distribution_l603_60326


namespace NUMINAMATH_CALUDE_max_x_is_one_l603_60390

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≤ 0 then Real.exp x + a else x^2 + 1 + a

-- State the theorem
theorem max_x_is_one (a : ℝ) :
  (∀ x : ℝ, f a (2 - x) ≥ f a x) →
  (∀ x : ℝ, x ≤ 1) :=
by sorry

end NUMINAMATH_CALUDE_max_x_is_one_l603_60390


namespace NUMINAMATH_CALUDE_prime_arithmetic_sequence_common_difference_l603_60361

theorem prime_arithmetic_sequence_common_difference (p : ℕ) (a : ℕ → ℕ) (d : ℕ) :
  Prime p →
  (∀ i, i ∈ Finset.range p → Prime (a i)) →
  (∀ i j, i < j → j < p → a i < a j) →
  (∀ i, i < p - 1 → a (i + 1) - a i = d) →
  a 0 > p →
  p ∣ d :=
sorry

end NUMINAMATH_CALUDE_prime_arithmetic_sequence_common_difference_l603_60361


namespace NUMINAMATH_CALUDE_not_always_true_point_not_in_plane_l603_60384

-- Define the basic types
variable (Point Line Plane : Type)

-- Define the necessary relations
variable (belongs_to : Point → Line → Prop)
variable (belongs_to_plane : Point → Plane → Prop)
variable (line_in_plane : Line → Plane → Prop)

-- Define the theorem
theorem not_always_true_point_not_in_plane 
  (A : Point) (l : Line) (α : Plane) : 
  ¬(∀ A l α, ¬(line_in_plane l α) → belongs_to A l → ¬(belongs_to_plane A α)) :=
by sorry

end NUMINAMATH_CALUDE_not_always_true_point_not_in_plane_l603_60384


namespace NUMINAMATH_CALUDE_min_sum_xy_l603_60312

theorem min_sum_xy (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x * y * (x - y)^2 = 1) : 
  x + y ≥ 2 := by
sorry

end NUMINAMATH_CALUDE_min_sum_xy_l603_60312


namespace NUMINAMATH_CALUDE_k_range_l603_60306

/-- The condition that for any real b, the line y = kx + b and the hyperbola x^2 - 2y^2 = 1 always have common points -/
def always_intersect (k : ℝ) : Prop :=
  ∀ b : ℝ, ∃ x y : ℝ, y = k * x + b ∧ x^2 - 2 * y^2 = 1

/-- The theorem stating the range of k given the always_intersect condition -/
theorem k_range (k : ℝ) : always_intersect k ↔ -Real.sqrt 2 / 2 < k ∧ k < Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_k_range_l603_60306


namespace NUMINAMATH_CALUDE_correct_calculation_l603_60385

theorem correct_calculation (x : ℝ) : 3 * x - 12 = 60 → (x / 3) + 12 = 20 := by
  sorry

end NUMINAMATH_CALUDE_correct_calculation_l603_60385


namespace NUMINAMATH_CALUDE_only_three_solutions_l603_60334

/-- Represents a solution to the equation AB = B^V -/
structure Solution :=
  (a b v : Nat)
  (h1 : a ≠ b ∧ a ≠ v ∧ b ≠ v)
  (h2 : a > 0 ∧ a < 10 ∧ b > 0 ∧ b < 10 ∧ v > 0 ∧ v < 10)
  (h3 : 10 * a + b = b^v)

/-- The set of all valid solutions -/
def allSolutions : Set Solution := {s | s.a > 0 ∧ s.b > 0 ∧ s.v > 0}

/-- The theorem stating that there are only three solutions -/
theorem only_three_solutions :
  allSolutions = {
    ⟨3, 2, 5, sorry, sorry, sorry⟩,
    ⟨3, 6, 2, sorry, sorry, sorry⟩,
    ⟨6, 4, 3, sorry, sorry, sorry⟩
  } := by sorry

end NUMINAMATH_CALUDE_only_three_solutions_l603_60334


namespace NUMINAMATH_CALUDE_geometric_series_ratio_l603_60371

theorem geometric_series_ratio (a : ℝ) (r : ℝ) (hr : r ≠ 1) :
  (a / (1 - r)) = 64 * (a * r^4 / (1 - r)) → r = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_geometric_series_ratio_l603_60371


namespace NUMINAMATH_CALUDE_marys_sheep_ratio_l603_60304

theorem marys_sheep_ratio (total_sheep : ℕ) (sister_fraction : ℚ) (remaining_sheep : ℕ) : 
  total_sheep = 400 →
  sister_fraction = 1/4 →
  remaining_sheep = 150 →
  let sheep_to_sister := total_sheep * sister_fraction
  let sheep_after_sister := total_sheep - sheep_to_sister
  let sheep_to_brother := sheep_after_sister - remaining_sheep
  (sheep_to_brother : ℚ) / sheep_after_sister = 1/2 := by
    sorry

end NUMINAMATH_CALUDE_marys_sheep_ratio_l603_60304


namespace NUMINAMATH_CALUDE_eggs_in_box_l603_60311

/-- The number of eggs initially in the box -/
def initial_eggs : ℝ := 47.0

/-- The number of eggs Harry adds to the box -/
def added_eggs : ℝ := 5.0

/-- The total number of eggs in the box after Harry adds eggs -/
def total_eggs : ℝ := initial_eggs + added_eggs

theorem eggs_in_box : total_eggs = 52.0 := by
  sorry

end NUMINAMATH_CALUDE_eggs_in_box_l603_60311


namespace NUMINAMATH_CALUDE_special_numbers_theorem_l603_60330

def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else n % 10 + sum_of_digits (n / 10)

def satisfies_condition (n : ℕ) : Prop :=
  n > 0 ∧ n - sum_of_digits n = 2007

theorem special_numbers_theorem : 
  {n : ℕ | satisfies_condition n} = {2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019} :=
by sorry

end NUMINAMATH_CALUDE_special_numbers_theorem_l603_60330


namespace NUMINAMATH_CALUDE_consecutive_numbers_sum_no_ten_consecutive_sum_2016_seven_consecutive_sum_2016_l603_60323

theorem consecutive_numbers_sum (n : ℕ) (sum : ℕ) : Prop :=
  ∃ a : ℕ, (n * a + n * (n - 1) / 2 = sum)

theorem no_ten_consecutive_sum_2016 : ¬ consecutive_numbers_sum 10 2016 := by
  sorry

theorem seven_consecutive_sum_2016 : consecutive_numbers_sum 7 2016 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_numbers_sum_no_ten_consecutive_sum_2016_seven_consecutive_sum_2016_l603_60323


namespace NUMINAMATH_CALUDE_cubic_minus_x_factorization_l603_60355

theorem cubic_minus_x_factorization (x : ℝ) : x^3 - x = x * (x + 1) * (x - 1) := by
  sorry

end NUMINAMATH_CALUDE_cubic_minus_x_factorization_l603_60355


namespace NUMINAMATH_CALUDE_complete_square_quadratic_l603_60381

theorem complete_square_quadratic (x : ℝ) : 
  x^2 - 2*x - 2 = 0 → (x - 1)^2 = 3 := by
  sorry

end NUMINAMATH_CALUDE_complete_square_quadratic_l603_60381


namespace NUMINAMATH_CALUDE_triangular_array_count_l603_60343

def triangular_array (bottom_row : Fin 12 → Fin 2) : ℕ :=
  let top_value := (bottom_row 0) + (bottom_row 1) + (bottom_row 10) + (bottom_row 11)
  if top_value % 5 = 0 then 1 else 0

theorem triangular_array_count :
  (Finset.univ.filter (λ f : Fin 12 → Fin 2 => triangular_array f = 1)).card = 1280 :=
sorry

end NUMINAMATH_CALUDE_triangular_array_count_l603_60343


namespace NUMINAMATH_CALUDE_additional_coins_for_eight_friends_l603_60347

/-- The minimum number of additional coins needed for unique distribution. -/
def min_additional_coins (num_friends : ℕ) (initial_coins : ℕ) : ℕ :=
  let total_needed := (num_friends * (num_friends + 1)) / 2
  if total_needed > initial_coins then
    total_needed - initial_coins
  else
    0

/-- Theorem: Given 8 friends and 28 initial coins, 8 additional coins are needed. -/
theorem additional_coins_for_eight_friends :
  min_additional_coins 8 28 = 8 := by
  sorry


end NUMINAMATH_CALUDE_additional_coins_for_eight_friends_l603_60347
