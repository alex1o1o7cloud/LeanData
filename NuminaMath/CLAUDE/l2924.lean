import Mathlib

namespace NUMINAMATH_CALUDE_balanced_sequence_equality_l2924_292478

/-- A sequence of five real numbers is balanced if, when any one number is removed, 
    the remaining four can be divided into two groups of two numbers each 
    such that the sum of one group equals the sum of the other group. -/
def IsBalanced (a b c d e : ℝ) : Prop :=
  (b + c = d + e) ∧ (a + c = d + e) ∧ (a + b = d + e) ∧
  (a + c = b + e) ∧ (a + d = b + e)

theorem balanced_sequence_equality (a b c d e : ℝ) 
  (h_pos : 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d ∧ 0 < e)
  (h_order : a ≤ b ∧ b ≤ c ∧ c ≤ d ∧ d ≤ e)
  (h_balanced : IsBalanced a b c d e)
  (h_sum1 : e + c = b + d)
  (h_sum2 : e + a = c + d) :
  a = b ∧ b = c ∧ c = d ∧ d = e := by
  sorry

end NUMINAMATH_CALUDE_balanced_sequence_equality_l2924_292478


namespace NUMINAMATH_CALUDE_power_division_rule_l2924_292442

theorem power_division_rule (x : ℝ) : x^4 / x = x^3 := by sorry

end NUMINAMATH_CALUDE_power_division_rule_l2924_292442


namespace NUMINAMATH_CALUDE_parallel_vectors_x_value_l2924_292469

/-- Two vectors in R² are parallel if their coordinates are proportional -/
def parallel (a b : ℝ × ℝ) : Prop :=
  a.1 * b.2 = a.2 * b.1

theorem parallel_vectors_x_value :
  ∀ x : ℝ, 
  let a : ℝ × ℝ := (2, 1)
  let b : ℝ × ℝ := (4, x + 1)
  parallel a b → x = 1 := by
sorry

end NUMINAMATH_CALUDE_parallel_vectors_x_value_l2924_292469


namespace NUMINAMATH_CALUDE_cos_seven_pi_fourth_l2924_292497

theorem cos_seven_pi_fourth : Real.cos (7 * π / 4) = 1 / Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_seven_pi_fourth_l2924_292497


namespace NUMINAMATH_CALUDE_intersection_complement_equality_l2924_292443

open Set

def A : Set ℝ := {x | x^2 - 1 ≤ 0}
def B : Set ℝ := {x | x < 1}

theorem intersection_complement_equality : A ∩ (𝒰 \ B) = {x | x = 1} := by sorry

end NUMINAMATH_CALUDE_intersection_complement_equality_l2924_292443


namespace NUMINAMATH_CALUDE_starters_count_l2924_292404

-- Define the total number of players
def total_players : ℕ := 16

-- Define the number of twins
def num_twins : ℕ := 2

-- Define the number of triplets
def num_triplets : ℕ := 3

-- Define the number of starters to choose
def num_starters : ℕ := 7

-- Define the function to calculate the number of ways to choose starters
def choose_starters (total : ℕ) (twins : ℕ) (triplets : ℕ) (starters : ℕ) : ℕ :=
  -- Implementation details are omitted
  sorry

-- Theorem statement
theorem starters_count :
  choose_starters total_players num_twins num_triplets num_starters = 5148 := by
  sorry

end NUMINAMATH_CALUDE_starters_count_l2924_292404


namespace NUMINAMATH_CALUDE_sum_of_powers_l2924_292496

theorem sum_of_powers : (1 : ℤ)^10 + (-1 : ℤ)^8 + (-1 : ℤ)^7 + (1 : ℤ)^5 = 2 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_powers_l2924_292496


namespace NUMINAMATH_CALUDE_max_value_sine_cosine_l2924_292402

theorem max_value_sine_cosine (x : Real) : 
  0 ≤ x → x < 2 * Real.pi → 
  ∃ (max_x : Real), max_x = 5 * Real.pi / 6 ∧
    ∀ y : Real, 0 ≤ y → y < 2 * Real.pi → 
      Real.sin x - Real.sqrt 3 * Real.cos x ≤ Real.sin max_x - Real.sqrt 3 * Real.cos max_x :=
by sorry

end NUMINAMATH_CALUDE_max_value_sine_cosine_l2924_292402


namespace NUMINAMATH_CALUDE_y_derivative_l2924_292457

noncomputable def y (x : ℝ) : ℝ :=
  2 * x - Real.log (1 + Real.sqrt (1 - Real.exp (4 * x))) - Real.exp (-2 * x) * Real.arcsin (Real.exp (2 * x))

theorem y_derivative (x : ℝ) :
  deriv y x = 2 * Real.exp (-2 * x) * Real.arcsin (Real.exp (2 * x)) :=
by sorry

end NUMINAMATH_CALUDE_y_derivative_l2924_292457


namespace NUMINAMATH_CALUDE_cube_surface_area_l2924_292436

/-- Given a cube with the sum of edge lengths equal to 36 and space diagonal length equal to 3√3,
    the total surface area is 54. -/
theorem cube_surface_area (s : ℝ) 
  (h1 : 12 * s = 36) 
  (h2 : s * Real.sqrt 3 = 3 * Real.sqrt 3) : 
  6 * s^2 = 54 := by
  sorry

end NUMINAMATH_CALUDE_cube_surface_area_l2924_292436


namespace NUMINAMATH_CALUDE_rectangle_perimeter_minus_4_l2924_292452

/-- The perimeter of a rectangle minus 4, given its width and length. -/
def perimeterMinus4 (width length : ℝ) : ℝ :=
  2 * width + 2 * length - 4

/-- Theorem: For a rectangle with width 4 cm and length 8 cm, 
    the perimeter minus 4 equals 20 cm. -/
theorem rectangle_perimeter_minus_4 :
  perimeterMinus4 4 8 = 20 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_perimeter_minus_4_l2924_292452


namespace NUMINAMATH_CALUDE_sum_of_seven_thirds_l2924_292413

theorem sum_of_seven_thirds (x : ℚ) : 
  x = 1 / 3 → x + x + x + x + x + x + x = 7 * (1 / 3) := by
  sorry

end NUMINAMATH_CALUDE_sum_of_seven_thirds_l2924_292413


namespace NUMINAMATH_CALUDE_suzanna_bike_ride_l2924_292450

/-- Suzanna's bike ride problem -/
theorem suzanna_bike_ride (rate : ℝ) (time : ℝ) : 
  rate = 3 / 10 → time = 40 → rate * time = 12 := by
  sorry

end NUMINAMATH_CALUDE_suzanna_bike_ride_l2924_292450


namespace NUMINAMATH_CALUDE_math_class_size_l2924_292492

theorem math_class_size (total_students : ℕ) (both_subjects : ℕ) :
  total_students = 75 →
  both_subjects = 15 →
  ∃ (math_only physics_only : ℕ),
    total_students = math_only + physics_only + both_subjects ∧
    math_only + both_subjects = 2 * (physics_only + both_subjects) →
  math_only + both_subjects = 60 := by
  sorry

end NUMINAMATH_CALUDE_math_class_size_l2924_292492


namespace NUMINAMATH_CALUDE_roots_relation_l2924_292410

-- Define the polynomials
def f (x : ℝ) : ℝ := x^3 + 5*x^2 + 6*x - 8
def g (x u v w : ℝ) : ℝ := x^3 + u*x^2 + v*x + w

-- Define the theorem
theorem roots_relation (p q r u v w : ℝ) : 
  (f p = 0 ∧ f q = 0 ∧ f r = 0) → 
  (g (p+q) u v w = 0 ∧ g (q+r) u v w = 0 ∧ g (r+p) u v w = 0) →
  w = 8 := by
sorry

end NUMINAMATH_CALUDE_roots_relation_l2924_292410


namespace NUMINAMATH_CALUDE_partition_naturals_with_property_l2924_292438

theorem partition_naturals_with_property : 
  ∃ (partition : ℕ → Fin 100), 
    (∀ i : Fin 100, ∃ n : ℕ, partition n = i) ∧ 
    (∀ a b c : ℕ, a + 99 * b = c → 
      partition a = partition b ∨ 
      partition a = partition c ∨ 
      partition b = partition c) := by sorry

end NUMINAMATH_CALUDE_partition_naturals_with_property_l2924_292438


namespace NUMINAMATH_CALUDE_linked_rings_length_l2924_292463

/-- Represents a sequence of linked rings with specific properties. -/
structure LinkedRings where
  ringThickness : ℝ
  topRingDiameter : ℝ
  bottomRingDiameter : ℝ
  diameterDecrease : ℝ

/-- Calculates the total length of the linked rings. -/
def totalLength (rings : LinkedRings) : ℝ :=
  sorry

/-- Theorem stating that the total length of the linked rings with given properties is 342 cm. -/
theorem linked_rings_length :
  let rings : LinkedRings := {
    ringThickness := 2,
    topRingDiameter := 40,
    bottomRingDiameter := 4,
    diameterDecrease := 2
  }
  totalLength rings = 342 := by sorry

end NUMINAMATH_CALUDE_linked_rings_length_l2924_292463


namespace NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l2924_292468

/-- A geometric sequence with first term a₁ and common ratio q -/
def geometric_sequence (a₁ : ℝ) (q : ℝ) : ℕ → ℝ :=
  λ n => a₁ * q^(n - 1)

theorem geometric_sequence_common_ratio 
  (a : ℕ → ℝ) 
  (h_geometric : ∃ (a₁ q : ℝ), ∀ n, a n = geometric_sequence a₁ q n)
  (h_a₁ : a 1 = 2)
  (h_a₄ : a 4 = 16) :
  ∃ q, ∀ n, a n = geometric_sequence 2 q n ∧ q = 2 :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l2924_292468


namespace NUMINAMATH_CALUDE_pet_store_birds_l2924_292491

/-- The number of bird cages in the pet store -/
def num_cages : ℕ := 4

/-- The number of parrots in each cage -/
def parrots_per_cage : ℕ := 8

/-- The number of parakeets in each cage -/
def parakeets_per_cage : ℕ := 2

/-- The total number of birds in the pet store -/
def total_birds : ℕ := num_cages * (parrots_per_cage + parakeets_per_cage)

theorem pet_store_birds : total_birds = 40 := by
  sorry

end NUMINAMATH_CALUDE_pet_store_birds_l2924_292491


namespace NUMINAMATH_CALUDE_hannah_cookies_sold_l2924_292477

/-- Proves that Hannah sold 40 cookies given the conditions of the problem -/
theorem hannah_cookies_sold : ℕ :=
  let cookie_price : ℚ := 8 / 10
  let cupcake_price : ℚ := 2
  let cupcakes_sold : ℕ := 30
  let spoon_set_price : ℚ := 13 / 2
  let spoon_sets_bought : ℕ := 2
  let money_left : ℚ := 79

  let cookies_sold : ℕ := 40

  have h1 : cookie_price * cookies_sold + cupcake_price * cupcakes_sold = 
            spoon_set_price * spoon_sets_bought + money_left := by sorry

  cookies_sold


end NUMINAMATH_CALUDE_hannah_cookies_sold_l2924_292477


namespace NUMINAMATH_CALUDE_half_angle_quadrant_l2924_292415

def is_second_quadrant (α : Real) : Prop :=
  ∃ k : ℤ, 2 * k * Real.pi + Real.pi / 2 < α ∧ α < 2 * k * Real.pi + Real.pi

def is_first_or_third_quadrant (α : Real) : Prop :=
  ∃ k : ℤ, k * Real.pi < α ∧ α < k * Real.pi + Real.pi / 2

theorem half_angle_quadrant (α : Real) :
  is_second_quadrant α → is_first_or_third_quadrant (α / 2) :=
by sorry

end NUMINAMATH_CALUDE_half_angle_quadrant_l2924_292415


namespace NUMINAMATH_CALUDE_xyz_sum_l2924_292494

theorem xyz_sum (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (hxy : x * y = 32) (hxz : x * z = 64) (hyz : y * z = 96) :
  x + y + z = 44 * Real.sqrt 3 / 3 := by
sorry

end NUMINAMATH_CALUDE_xyz_sum_l2924_292494


namespace NUMINAMATH_CALUDE_ticket_popcorn_difference_l2924_292447

/-- Represents the cost of items and the deal in a movie theater. -/
structure MovieTheaterCosts where
  deal : ℝ
  ticket : ℝ
  popcorn : ℝ
  drink : ℝ
  candy : ℝ

/-- The conditions of the movie theater deal problem. -/
def dealConditions (c : MovieTheaterCosts) : Prop :=
  c.deal = 20 ∧
  c.ticket = 8 ∧
  c.drink = c.popcorn + 1 ∧
  c.candy = c.drink / 2 ∧
  c.deal = c.ticket + c.popcorn + c.drink + c.candy - 2

/-- The theorem stating the difference between ticket and popcorn costs. -/
theorem ticket_popcorn_difference (c : MovieTheaterCosts) 
  (h : dealConditions c) : c.ticket - c.popcorn = 3 := by
  sorry


end NUMINAMATH_CALUDE_ticket_popcorn_difference_l2924_292447


namespace NUMINAMATH_CALUDE_hyperbola_standard_form_l2924_292411

theorem hyperbola_standard_form (x y : ℝ) :
  x^2 - 15 * y^2 = 15 ↔ x^2 / 15 - y^2 = 1 := by sorry

end NUMINAMATH_CALUDE_hyperbola_standard_form_l2924_292411


namespace NUMINAMATH_CALUDE_range_of_m_l2924_292401

def is_periodic (f : ℝ → ℝ) (p : ℝ) : Prop :=
  ∀ x, f (x + p) = f x

def smallest_positive_period (f : ℝ → ℝ) (p : ℝ) : Prop :=
  p > 0 ∧ is_periodic f p ∧ ∀ q, 0 < q ∧ q < p → ¬ is_periodic f q

def is_odd (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

theorem range_of_m (f : ℝ → ℝ) (m : ℝ) :
  is_odd f →
  smallest_positive_period f 3 →
  f 1 > -2 →
  f 2 = m^2 - m →
  m ∈ Set.Ioo (-1 : ℝ) 2 :=
by sorry

end NUMINAMATH_CALUDE_range_of_m_l2924_292401


namespace NUMINAMATH_CALUDE_rotate_point_on_circle_l2924_292427

/-- Given a circle with radius 5 centered at the origin, 
    prove that rotating the point (3,4) by 45 degrees counterclockwise 
    results in the point (-√2/2, 7√2/2) -/
theorem rotate_point_on_circle (P Q : ℝ × ℝ) : 
  P.1^2 + P.2^2 = 25 →  -- P is on the circle
  P = (3, 4) →  -- P starts at (3,4)
  Q.1 = P.1 * (Real.sqrt 2 / 2) - P.2 * (Real.sqrt 2 / 2) →  -- Q is P rotated 45°
  Q.2 = P.1 * (Real.sqrt 2 / 2) + P.2 * (Real.sqrt 2 / 2) →
  Q = (-Real.sqrt 2 / 2, 7 * Real.sqrt 2 / 2) := by
  sorry

end NUMINAMATH_CALUDE_rotate_point_on_circle_l2924_292427


namespace NUMINAMATH_CALUDE_prime_equation_solution_l2924_292471

theorem prime_equation_solution :
  ∀ p q : ℕ, 
    Nat.Prime p → Nat.Prime q →
    p^2 - 6*p*q + q^2 + 3*q - 1 = 0 →
    (p = 17 ∧ q = 3) :=
by sorry

end NUMINAMATH_CALUDE_prime_equation_solution_l2924_292471


namespace NUMINAMATH_CALUDE_perfect_square_condition_l2924_292451

theorem perfect_square_condition (n : ℤ) : 
  ∃ (k : ℤ), n^4 + 6*n^3 + 11*n^2 + 3*n + 31 = k^2 ↔ n = 10 := by
  sorry

end NUMINAMATH_CALUDE_perfect_square_condition_l2924_292451


namespace NUMINAMATH_CALUDE_circumscribed_sphere_surface_area_l2924_292429

/-- The surface area of a sphere circumscribed around a rectangular solid -/
theorem circumscribed_sphere_surface_area 
  (length width height : ℝ) 
  (h_length : length = 2)
  (h_width : width = 2)
  (h_height : height = 2 * Real.sqrt 2) : 
  4 * Real.pi * ((length^2 + width^2 + height^2) / 4) = 16 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_circumscribed_sphere_surface_area_l2924_292429


namespace NUMINAMATH_CALUDE_sum_three_fourths_power_inequality_l2924_292407

theorem sum_three_fourths_power_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  a^(3/4) + b^(3/4) + c^(3/4) > (a + b + c)^(3/4) := by
  sorry

end NUMINAMATH_CALUDE_sum_three_fourths_power_inequality_l2924_292407


namespace NUMINAMATH_CALUDE_complex_sum_simplification_l2924_292403

theorem complex_sum_simplification :
  let z₁ : ℂ := (-1 + Complex.I * Real.sqrt 3) / 2
  let z₂ : ℂ := (-1 - Complex.I * Real.sqrt 3) / 2
  z₁^12 + z₂^12 = 2 := by
sorry

end NUMINAMATH_CALUDE_complex_sum_simplification_l2924_292403


namespace NUMINAMATH_CALUDE_cash_percentage_is_twenty_percent_l2924_292464

def raw_materials : ℝ := 35000
def machinery : ℝ := 40000
def total_amount : ℝ := 93750

theorem cash_percentage_is_twenty_percent :
  (total_amount - (raw_materials + machinery)) / total_amount * 100 = 20 := by
  sorry

end NUMINAMATH_CALUDE_cash_percentage_is_twenty_percent_l2924_292464


namespace NUMINAMATH_CALUDE_students_per_group_l2924_292453

theorem students_per_group (total : ℕ) (not_picked : ℕ) (groups : ℕ) 
  (h1 : total = 17) 
  (h2 : not_picked = 5) 
  (h3 : groups = 3) :
  (total - not_picked) / groups = 4 := by
sorry

end NUMINAMATH_CALUDE_students_per_group_l2924_292453


namespace NUMINAMATH_CALUDE_expand_binomial_product_l2924_292481

theorem expand_binomial_product (x : ℝ) : (x + 4) * (x - 9) = x^2 - 5*x - 36 := by
  sorry

end NUMINAMATH_CALUDE_expand_binomial_product_l2924_292481


namespace NUMINAMATH_CALUDE_corn_ears_per_stalk_l2924_292487

/-- The number of corn stalks -/
def num_stalks : ℕ := 108

/-- The number of kernels in half of the ears -/
def kernels_half1 : ℕ := 500

/-- The number of kernels in the other half of the ears -/
def kernels_half2 : ℕ := 600

/-- The total number of kernels -/
def total_kernels : ℕ := 237600

/-- The number of ears per stalk -/
def ears_per_stalk : ℕ := 4

theorem corn_ears_per_stalk :
  num_stalks * (ears_per_stalk / 2 * kernels_half1 + ears_per_stalk / 2 * kernels_half2) = total_kernels :=
by sorry

end NUMINAMATH_CALUDE_corn_ears_per_stalk_l2924_292487


namespace NUMINAMATH_CALUDE_infinitely_many_triangular_pentagonal_pairs_l2924_292462

/-- A pair of positive integers (n, m) is a triangular-pentagonal pair if n(n+1) = m(3m-1) -/
def IsTriangularPentagonalPair (n m : ℕ) : Prop :=
  n > 0 ∧ m > 0 ∧ n * (n + 1) = m * (3 * m - 1)

/-- There exist infinitely many triangular-pentagonal pairs -/
theorem infinitely_many_triangular_pentagonal_pairs :
  ∀ k : ℕ, ∃ n m : ℕ, n > k ∧ m > k ∧ IsTriangularPentagonalPair n m :=
sorry

end NUMINAMATH_CALUDE_infinitely_many_triangular_pentagonal_pairs_l2924_292462


namespace NUMINAMATH_CALUDE_car_pedestrian_speed_ratio_l2924_292422

/-- The ratio of a car's speed to a pedestrian's speed on a bridge -/
theorem car_pedestrian_speed_ratio :
  ∀ (L : ℝ) (v_p v_c : ℝ),
  L > 0 → v_p > 0 → v_c > 0 →
  (4/9 * L) / v_p = (5/9 * L) / v_c →
  v_c / v_p = 9 := by
  sorry

end NUMINAMATH_CALUDE_car_pedestrian_speed_ratio_l2924_292422


namespace NUMINAMATH_CALUDE_smallest_bench_sections_l2924_292417

theorem smallest_bench_sections (N : ℕ) : 
  (∃ x : ℕ, x > 0 ∧ 8 * N = x ∧ 12 * N = x) ↔ N ≥ 3 :=
sorry

end NUMINAMATH_CALUDE_smallest_bench_sections_l2924_292417


namespace NUMINAMATH_CALUDE_disk_arrangement_area_sum_l2924_292449

theorem disk_arrangement_area_sum :
  ∀ (n : ℕ) (r : ℝ) (disk_radius : ℝ),
    n = 15 →
    r = 1 →
    disk_radius = 2 - Real.sqrt 3 →
    (↑n * π * disk_radius^2 : ℝ) = π * (105 - 60 * Real.sqrt 3) ∧
    105 + 60 + 3 = 168 := by
  sorry

end NUMINAMATH_CALUDE_disk_arrangement_area_sum_l2924_292449


namespace NUMINAMATH_CALUDE_inradius_less_than_half_side_height_bound_l2924_292470

/-- Triangle ABC with side lengths a, b, c, angles A, B, C, inradius r, circumradius R, and height h_a from vertex A to side BC -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ
  r : ℝ
  R : ℝ
  h_a : ℝ
  pos_sides : 0 < a ∧ 0 < b ∧ 0 < c
  pos_angles : 0 < A ∧ 0 < B ∧ 0 < C
  angle_sum : A + B + C = π
  triangle_inequality : a < b + c ∧ b < a + c ∧ c < a + b

/-- The inradius is less than half of any side length -/
theorem inradius_less_than_half_side (t : Triangle) : 2 * t.r < t.a := by sorry

/-- The height to side a is at most twice the circumradius times the square of the cosine of half angle A -/
theorem height_bound (t : Triangle) : t.h_a ≤ 2 * t.R * (Real.cos (t.A / 2))^2 := by sorry

end NUMINAMATH_CALUDE_inradius_less_than_half_side_height_bound_l2924_292470


namespace NUMINAMATH_CALUDE_max_ratio_x_y_l2924_292445

theorem max_ratio_x_y (x y a b : ℝ) : 
  x ≥ y ∧ y > 0 →
  0 ≤ a ∧ a ≤ x →
  0 ≤ b ∧ b ≤ y →
  (x - a)^2 + (y - b)^2 = x^2 + b^2 →
  (x - a)^2 + (y - b)^2 = y^2 + a^2 →
  ∃ (c : ℝ), c = x / y ∧ c ≤ 2 * Real.sqrt 3 / 3 ∧
  ∀ (d : ℝ), d = x / y → d ≤ c :=
by sorry

end NUMINAMATH_CALUDE_max_ratio_x_y_l2924_292445


namespace NUMINAMATH_CALUDE_course_selection_theorem_l2924_292419

def physical_education_courses : ℕ := 4
def art_courses : ℕ := 4
def total_courses : ℕ := physical_education_courses + art_courses

def choose (n k : ℕ) : ℕ := (Nat.factorial n) / ((Nat.factorial k) * (Nat.factorial (n - k)))

def two_course_selections : ℕ := choose physical_education_courses 1 * choose art_courses 1

def three_course_selections : ℕ := 
  choose physical_education_courses 2 * choose art_courses 1 +
  choose physical_education_courses 1 * choose art_courses 2

def total_selections : ℕ := two_course_selections + three_course_selections

theorem course_selection_theorem : total_selections = 64 := by
  sorry

end NUMINAMATH_CALUDE_course_selection_theorem_l2924_292419


namespace NUMINAMATH_CALUDE_g_of_3_eq_10_l2924_292416

/-- The function g defined for all real numbers -/
def g (x : ℝ) : ℝ := x^2 + 1

/-- Theorem stating that g(3) = 10 -/
theorem g_of_3_eq_10 : g 3 = 10 := by
  sorry

end NUMINAMATH_CALUDE_g_of_3_eq_10_l2924_292416


namespace NUMINAMATH_CALUDE_bowtie_equation_solution_l2924_292482

-- Define the bowties operation
noncomputable def bowtie (c d : ℝ) : ℝ := c - Real.sqrt (d - Real.sqrt (d - Real.sqrt d))

-- Theorem statement
theorem bowtie_equation_solution :
  ∃ x : ℝ, bowtie 7 x = 3 → x = 20 := by
  sorry

end NUMINAMATH_CALUDE_bowtie_equation_solution_l2924_292482


namespace NUMINAMATH_CALUDE_william_land_percentage_l2924_292434

def total_tax : ℝ := 3840
def tax_percentage : ℝ := 0.75
def william_tax : ℝ := 480

theorem william_land_percentage :
  let total_taxable_income := total_tax / tax_percentage
  let william_percentage := (william_tax / total_taxable_income) * 100
  william_percentage = 9.375 := by sorry

end NUMINAMATH_CALUDE_william_land_percentage_l2924_292434


namespace NUMINAMATH_CALUDE_power_of_power_l2924_292433

theorem power_of_power (a : ℝ) : (a^2)^3 = a^6 := by
  sorry

end NUMINAMATH_CALUDE_power_of_power_l2924_292433


namespace NUMINAMATH_CALUDE_madeline_work_hours_l2924_292483

/-- Calculates the minimum number of work hours needed to cover expenses and savings --/
def min_work_hours (rent : ℕ) (groceries : ℕ) (medical : ℕ) (utilities : ℕ) (savings : ℕ) (hourly_wage : ℕ) : ℕ :=
  let total_expenses := rent + groceries + medical + utilities + savings
  (total_expenses + hourly_wage - 1) / hourly_wage

theorem madeline_work_hours :
  min_work_hours 1200 400 200 60 200 15 = 138 := by
  sorry

end NUMINAMATH_CALUDE_madeline_work_hours_l2924_292483


namespace NUMINAMATH_CALUDE_time_equation_l2924_292414

-- Define variables
variable (g V V₀ c S t : ℝ)

-- State the theorem
theorem time_equation (eq1 : V = g * t + V₀ + c) (eq2 : S = (1/2) * g * t^2 + V₀ * t + c * t^2) :
  t = 2 * S / (V + V₀ - c) := by
  sorry

end NUMINAMATH_CALUDE_time_equation_l2924_292414


namespace NUMINAMATH_CALUDE_problem_solution_l2924_292498

theorem problem_solution (x y z t : ℝ) 
  (eq1 : x = y^2 - 16*x^2)
  (eq2 : y = z^2 - 4*x^2)
  (eq3 : z = t^2 - x^2)
  (eq4 : t = x - 1) :
  x = 1/9 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l2924_292498


namespace NUMINAMATH_CALUDE_a_zero_sufficient_not_necessary_l2924_292426

def M (a : ℝ) : Set ℝ := {1, a}
def N : Set ℝ := {-1, 0, 1}

theorem a_zero_sufficient_not_necessary :
  (∀ a : ℝ, a = 0 → M a ⊆ N) ∧
  (∃ a : ℝ, a ≠ 0 ∧ M a ⊆ N) :=
by sorry

end NUMINAMATH_CALUDE_a_zero_sufficient_not_necessary_l2924_292426


namespace NUMINAMATH_CALUDE_final_statue_weight_approx_l2924_292476

/-- The weight of the final statue given the initial weights and removal percentages --/
def final_statue_weight (initial_marble : ℝ) (initial_granite : ℝ) 
  (marble_removal1 : ℝ) (marble_removal2 : ℝ) (marble_removal3 : ℝ) 
  (granite_removal1 : ℝ) (granite_removal2 : ℝ) 
  (marble_removal_final : ℝ) (granite_removal_final : ℝ) : ℝ :=
  let remaining_marble1 := initial_marble * (1 - marble_removal1)
  let remaining_marble2 := remaining_marble1 * (1 - marble_removal2)
  let remaining_marble3 := remaining_marble2 * (1 - marble_removal3)
  let final_marble := remaining_marble3 * (1 - marble_removal_final)
  
  let remaining_granite1 := initial_granite * (1 - granite_removal1)
  let remaining_granite2 := remaining_granite1 * (1 - granite_removal2)
  let final_granite := remaining_granite2 * (1 - granite_removal_final)
  
  final_marble + final_granite

/-- The final weight of the statue is approximately 119.0826 kg --/
theorem final_statue_weight_approx :
  ∃ ε > 0, ε < 0.0001 ∧ 
  |final_statue_weight 225 65 0.32 0.22 0.15 0.40 0.25 0.10 0.05 - 119.0826| < ε :=
sorry

end NUMINAMATH_CALUDE_final_statue_weight_approx_l2924_292476


namespace NUMINAMATH_CALUDE_chosen_number_proof_l2924_292437

theorem chosen_number_proof (x : ℝ) : (x / 2) - 100 = 4 → x = 208 := by
  sorry

end NUMINAMATH_CALUDE_chosen_number_proof_l2924_292437


namespace NUMINAMATH_CALUDE_committee_probability_l2924_292480

def total_members : ℕ := 20
def boys : ℕ := 12
def girls : ℕ := 8
def committee_size : ℕ := 4

def probability_at_least_one_boy_and_girl : ℚ :=
  1 - (Nat.choose boys committee_size + Nat.choose girls committee_size : ℚ) / Nat.choose total_members committee_size

theorem committee_probability :
  probability_at_least_one_boy_and_girl = 4280 / 4845 :=
sorry

end NUMINAMATH_CALUDE_committee_probability_l2924_292480


namespace NUMINAMATH_CALUDE_polynomial_factorization_l2924_292409

theorem polynomial_factorization (x : ℝ) :
  x^8 + x^4 + 1 = (x^2 - Real.sqrt 3 * x + 1) * (x^2 + Real.sqrt 3 * x + 1) * (x^2 - x + 1) * (x^2 + x + 1) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l2924_292409


namespace NUMINAMATH_CALUDE_percentage_relation_l2924_292495

theorem percentage_relation (third_number : ℝ) (first_number : ℝ) (second_number : ℝ)
  (h1 : first_number = 0.08 * third_number)
  (h2 : second_number = 0.16 * third_number)
  (h3 : first_number = 0.5 * second_number) :
  first_number = 0.08 * third_number := by
sorry

end NUMINAMATH_CALUDE_percentage_relation_l2924_292495


namespace NUMINAMATH_CALUDE_square_perimeter_ratio_l2924_292459

theorem square_perimeter_ratio (area1 area2 perimeter1 perimeter2 : ℝ) :
  area1 > 0 ∧ area2 > 0 →
  area1 / area2 = 49 / 64 →
  perimeter1 / perimeter2 = 7 / 8 :=
by sorry

end NUMINAMATH_CALUDE_square_perimeter_ratio_l2924_292459


namespace NUMINAMATH_CALUDE_dinner_tip_calculation_l2924_292421

/-- Calculates the individual tip amount for a group dinner -/
theorem dinner_tip_calculation (julie_order : ℚ) (letitia_order : ℚ) (anton_order : ℚ) 
  (tip_percentage : ℚ) (num_people : ℕ) : 
  julie_order = 10 ∧ letitia_order = 20 ∧ anton_order = 30 ∧ 
  tip_percentage = 1/5 ∧ num_people = 3 →
  (julie_order + letitia_order + anton_order) * tip_percentage / num_people = 4 := by
  sorry

#check dinner_tip_calculation

end NUMINAMATH_CALUDE_dinner_tip_calculation_l2924_292421


namespace NUMINAMATH_CALUDE_trig_sum_equality_l2924_292428

theorem trig_sum_equality : 
  3.423 * Real.sin (10 * π / 180) + Real.sin (20 * π / 180) + Real.sin (30 * π / 180) + 
  Real.sin (40 * π / 180) + Real.sin (50 * π / 180) = 
  Real.sin (25 * π / 180) / (2 * Real.sin (5 * π / 180)) := by
  sorry

end NUMINAMATH_CALUDE_trig_sum_equality_l2924_292428


namespace NUMINAMATH_CALUDE_at_least_one_negative_l2924_292460

theorem at_least_one_negative (a b c d : ℝ) 
  (sum_ab : a + b = 1) 
  (sum_cd : c + d = 1) 
  (prod_sum : a * c + b * d > 1) : 
  ¬(a ≥ 0 ∧ b ≥ 0 ∧ c ≥ 0 ∧ d ≥ 0) :=
by sorry

end NUMINAMATH_CALUDE_at_least_one_negative_l2924_292460


namespace NUMINAMATH_CALUDE_probability_standard_weight_l2924_292485

theorem probability_standard_weight (total_students : ℕ) (standard_weight_students : ℕ) :
  total_students = 500 →
  standard_weight_students = 350 →
  (standard_weight_students : ℚ) / (total_students : ℚ) = 7 / 10 :=
by sorry

end NUMINAMATH_CALUDE_probability_standard_weight_l2924_292485


namespace NUMINAMATH_CALUDE_kolya_is_collection_agency_l2924_292441

-- Define the actors in the scenario
structure Person :=
  (name : String)

-- Define the book lending scenario
structure BookLendingScenario :=
  (lender : Person)
  (borrower : Person)
  (collector : Person)
  (books_lent : ℕ)
  (return_promised : Bool)
  (books_returned : Bool)
  (collector_fee : ℕ)

-- Define the characteristics of a collection agency
structure CollectionAgency :=
  (collects_items : Bool)
  (acts_on_behalf : Bool)
  (receives_fee : Bool)

-- Define Kolya's role in the scenario
def kolya_role (scenario : BookLendingScenario) : CollectionAgency :=
  { collects_items := true
  , acts_on_behalf := true
  , receives_fee := scenario.collector_fee > 0 }

-- Theorem statement
theorem kolya_is_collection_agency (scenario : BookLendingScenario) : 
  kolya_role scenario = CollectionAgency.mk true true true :=
sorry

end NUMINAMATH_CALUDE_kolya_is_collection_agency_l2924_292441


namespace NUMINAMATH_CALUDE_complex_fraction_difference_l2924_292448

theorem complex_fraction_difference (i : ℂ) (h : i * i = -1) :
  (3 + 2*i) / (2 - 3*i) - (3 - 2*i) / (2 + 3*i) = 2*i :=
by sorry

end NUMINAMATH_CALUDE_complex_fraction_difference_l2924_292448


namespace NUMINAMATH_CALUDE_straight_A_students_after_increase_l2924_292466

/-- The number of straight-A students after new students join, given the initial conditions -/
theorem straight_A_students_after_increase 
  (initial_students : ℕ) 
  (new_students : ℕ) 
  (percentage_increase : ℚ) : ℕ :=
by
  -- Assume the initial conditions
  have h1 : initial_students = 25 := by sorry
  have h2 : new_students = 7 := by sorry
  have h3 : percentage_increase = 1/10 := by sorry

  -- Define the total number of students after new students join
  let total_students : ℕ := initial_students + new_students

  -- Define the function to calculate the number of straight-A students
  let calc_straight_A (x : ℕ) (y : ℕ) : Prop :=
    (x : ℚ) / initial_students + percentage_increase = ((x + y) : ℚ) / total_students

  -- Prove that there are 16 straight-A students after the increase
  have h4 : ∃ (x y : ℕ), calc_straight_A x y ∧ x + y = 16 := by sorry

  -- Conclude the theorem
  exact 16

end NUMINAMATH_CALUDE_straight_A_students_after_increase_l2924_292466


namespace NUMINAMATH_CALUDE_ball_travel_distance_l2924_292446

/-- The distance traveled by a ball rolling down a ramp -/
def ballDistance (initialDistance : ℕ) (increase : ℕ) (time : ℕ) : ℕ :=
  let lastTerm := initialDistance + (time - 1) * increase
  time * (initialDistance + lastTerm) / 2

/-- Theorem stating the total distance traveled by the ball -/
theorem ball_travel_distance :
  ballDistance 10 8 25 = 2650 := by
  sorry

end NUMINAMATH_CALUDE_ball_travel_distance_l2924_292446


namespace NUMINAMATH_CALUDE_sqrt_meaningful_range_l2924_292423

theorem sqrt_meaningful_range (x : ℝ) : 
  (∃ y : ℝ, y^2 = x - 5) → x ≥ 5 := by
sorry

end NUMINAMATH_CALUDE_sqrt_meaningful_range_l2924_292423


namespace NUMINAMATH_CALUDE_cube_surface_area_l2924_292425

/-- The surface area of a cube that can be cut into 27 smaller cubes, each with an edge length of 4 cm, is 864 cm². -/
theorem cube_surface_area : 
  ∀ (original_cube_edge : ℝ) (small_cube_edge : ℝ),
  small_cube_edge = 4 →
  (original_cube_edge / small_cube_edge)^3 = 27 →
  6 * original_cube_edge^2 = 864 := by
sorry

end NUMINAMATH_CALUDE_cube_surface_area_l2924_292425


namespace NUMINAMATH_CALUDE_semicircle_function_max_point_max_value_max_point_trig_l2924_292435

noncomputable section

variables (R : ℝ) (x : ℝ)

def semicircle_point (R x : ℝ) : ℝ × ℝ :=
  (x, Real.sqrt (4 * R^2 - x^2))

def y (R x : ℝ) : ℝ :=
  2 * x + 3 * (2 * R - x^2 / (2 * R))

theorem semicircle_function (R : ℝ) (h : R > 0) :
  ∀ x, 0 ≤ x ∧ x ≤ 2 * R →
  y R x = -3 / (2 * R) * x^2 + 2 * x + 6 * R :=
sorry

theorem max_point (R : ℝ) (h : R > 0) :
  ∃ x_max, x_max = 2 * R / 3 ∧
  ∀ x, 0 ≤ x ∧ x ≤ 2 * R → y R x ≤ y R x_max :=
sorry

theorem max_value (R : ℝ) (h : R > 0) :
  y R (2 * R / 3) = 20 * R / 3 :=
sorry

theorem max_point_trig (R : ℝ) (h : R > 0) :
  let x_max := 2 * R / 3
  let α := Real.arccos (1 - x_max^2 / (2 * R^2))
  Real.cos α = 7 / 9 ∧ Real.sin α = 4 * Real.sqrt 2 / 9 :=
sorry

end NUMINAMATH_CALUDE_semicircle_function_max_point_max_value_max_point_trig_l2924_292435


namespace NUMINAMATH_CALUDE_lower_selling_price_l2924_292458

/-- Proves that the lower selling price is 340 given the conditions of the problem -/
theorem lower_selling_price (cost_price selling_price : ℕ) :
  cost_price = 250 →
  selling_price = 350 →
  (selling_price - cost_price : ℚ) / cost_price = 
    ((340 - cost_price : ℚ) / cost_price) + 4 / 100 →
  340 = (selling_price - cost_price) * 100 / 104 + cost_price :=
by sorry

end NUMINAMATH_CALUDE_lower_selling_price_l2924_292458


namespace NUMINAMATH_CALUDE_cereal_spending_ratio_is_two_to_one_l2924_292440

/-- The ratio of Snap's spending to Crackle's spending on cereal -/
def cereal_spending_ratio : ℚ :=
  let total_spent : ℚ := 150
  let pop_spent : ℚ := 15
  let crackle_spent : ℚ := 3 * pop_spent
  let snap_spent : ℚ := total_spent - crackle_spent - pop_spent
  snap_spent / crackle_spent

/-- Theorem stating that the ratio of Snap's spending to Crackle's spending is 2:1 -/
theorem cereal_spending_ratio_is_two_to_one :
  cereal_spending_ratio = 2 := by
  sorry

end NUMINAMATH_CALUDE_cereal_spending_ratio_is_two_to_one_l2924_292440


namespace NUMINAMATH_CALUDE_sum_of_bases_l2924_292472

/-- Given two bases R₁ and R₂, and two fractions F₁ and F₂, prove that R₁ + R₂ = 21 -/
theorem sum_of_bases (R₁ R₂ : ℕ) (F₁ F₂ : ℚ) : R₁ + R₂ = 21 :=
  by
  have h1 : F₁ = (4 * R₁ + 7) / (R₁^2 - 1) := by sorry
  have h2 : F₂ = (7 * R₁ + 4) / (R₁^2 - 1) := by sorry
  have h3 : F₁ = (R₂ + 6) / (R₂^2 - 1) := by sorry
  have h4 : F₂ = (6 * R₂ + 1) / (R₂^2 - 1) := by sorry
  sorry

end NUMINAMATH_CALUDE_sum_of_bases_l2924_292472


namespace NUMINAMATH_CALUDE_smithtown_left_handed_women_percentage_l2924_292490

theorem smithtown_left_handed_women_percentage
  (total : ℕ)
  (right_handed : ℕ)
  (left_handed : ℕ)
  (men : ℕ)
  (women : ℕ)
  (h1 : right_handed = 3 * left_handed)
  (h2 : men = 3 * (men + women) / 5)
  (h3 : women = 2 * (men + women) / 5)
  (h4 : total = right_handed + left_handed)
  (h5 : total = men + women)
  (h6 : men ≤ right_handed) :
  left_handed * 100 / total = 25 := by
sorry

end NUMINAMATH_CALUDE_smithtown_left_handed_women_percentage_l2924_292490


namespace NUMINAMATH_CALUDE_place_one_after_two_digit_number_l2924_292424

/-- Given a two-digit number with tens digit t and units digit u,
    prove that placing the digit 1 after this number results in 100t + 10u + 1 -/
theorem place_one_after_two_digit_number (t u : ℕ) :
  let original := 10 * t + u
  let new_number := original * 10 + 1
  new_number = 100 * t + 10 * u + 1 := by
sorry

end NUMINAMATH_CALUDE_place_one_after_two_digit_number_l2924_292424


namespace NUMINAMATH_CALUDE_max_profit_is_45_6_l2924_292455

/-- Profit function for location A -/
def profit_A (x : ℝ) : ℝ := 5.06 * x - 0.15 * x^2

/-- Profit function for location B -/
def profit_B (x : ℝ) : ℝ := 2 * x

/-- Total profit function -/
def total_profit (x : ℝ) : ℝ := profit_A x + profit_B (15 - x)

theorem max_profit_is_45_6 :
  ∃ x : ℝ, x ≥ 0 ∧ x ≤ 15 ∧ 
  (∀ y : ℝ, y ≥ 0 → y ≤ 15 → total_profit y ≤ total_profit x) ∧
  total_profit x = 45.6 :=
sorry

end NUMINAMATH_CALUDE_max_profit_is_45_6_l2924_292455


namespace NUMINAMATH_CALUDE_complex_equality_l2924_292412

theorem complex_equality (z : ℂ) : z = -1 + I →
  Complex.abs (z - 2) = Complex.abs (z + 4) ∧
  Complex.abs (z - 2) = Complex.abs (z + 2*I) := by
  sorry

end NUMINAMATH_CALUDE_complex_equality_l2924_292412


namespace NUMINAMATH_CALUDE_min_stamps_for_50_cents_l2924_292400

/-- Represents the number of stamps and their total value -/
structure StampCombination :=
  (threes : ℕ)
  (fours : ℕ)

/-- Calculates the total value of stamps in cents -/
def total_value (s : StampCombination) : ℕ :=
  3 * s.threes + 4 * s.fours

/-- Checks if a stamp combination is valid (totals 50 cents) -/
def is_valid (s : StampCombination) : Prop :=
  total_value s = 50

/-- Theorem: The minimum number of stamps to make 50 cents using 3 cent and 4 cent stamps is 13 -/
theorem min_stamps_for_50_cents :
  ∃ (s : StampCombination), is_valid s ∧
    (∀ (t : StampCombination), is_valid t → s.threes + s.fours ≤ t.threes + t.fours) ∧
    s.threes + s.fours = 13 :=
  sorry

end NUMINAMATH_CALUDE_min_stamps_for_50_cents_l2924_292400


namespace NUMINAMATH_CALUDE_calculation_proof_l2924_292461

theorem calculation_proof : ((-4)^2 * (-1/2)^3 - (-4+1)) = 1 := by
  sorry

end NUMINAMATH_CALUDE_calculation_proof_l2924_292461


namespace NUMINAMATH_CALUDE_M_equals_divisors_of_151_l2924_292444

def M : Set Nat :=
  {d | ∃ m n : Nat, d = Nat.gcd (2*n + 3*m + 13) (Nat.gcd (3*n + 5*m + 1) (6*n + 6*m - 1))}

theorem M_equals_divisors_of_151 : M = {d : Nat | d > 0 ∧ d ∣ 151} := by
  sorry

end NUMINAMATH_CALUDE_M_equals_divisors_of_151_l2924_292444


namespace NUMINAMATH_CALUDE_probability_of_specific_combination_l2924_292467

def total_marbles : ℕ := 12 + 8 + 5

def red_marbles : ℕ := 12
def blue_marbles : ℕ := 8
def green_marbles : ℕ := 5

def marbles_drawn : ℕ := 4

def ways_to_draw_specific_combination : ℕ := (red_marbles.choose 2) * blue_marbles * green_marbles

def total_ways_to_draw : ℕ := total_marbles.choose marbles_drawn

theorem probability_of_specific_combination :
  (ways_to_draw_specific_combination : ℚ) / total_ways_to_draw = 264 / 1265 := by sorry

end NUMINAMATH_CALUDE_probability_of_specific_combination_l2924_292467


namespace NUMINAMATH_CALUDE_circle_and_line_intersection_l2924_292430

-- Define the circle equation
def circle_equation (x y m : ℝ) : Prop := x^2 + y^2 - 2*x - 4*y + m = 0

-- Define the line equation
def line_equation (x y : ℝ) : Prop := x + 2*y - 4 = 0

-- Define the perpendicularity condition
def perpendicular (x₁ y₁ x₂ y₂ : ℝ) : Prop := x₁ * x₂ + y₁ * y₂ = 0

theorem circle_and_line_intersection :
  ∀ m : ℝ,
  (∀ x y : ℝ, circle_equation x y m → x^2 + y^2 = (x - 1)^2 + (y - 2)^2 + (5 - m)) →
  (∃ x₁ y₁ x₂ y₂ : ℝ, 
    circle_equation x₁ y₁ m ∧ 
    circle_equation x₂ y₂ m ∧ 
    line_equation x₁ y₁ ∧ 
    line_equation x₂ y₂ ∧ 
    perpendicular x₁ y₁ x₂ y₂) →
  (m < 5 ∧ m = 8/5 ∧ 
   ∀ x y : ℝ, x^2 + y^2 - (8/5)*x - (16/5)*y = 0 ↔ 
   ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ 
   x = (1-t)*x₁ + t*x₂ ∧ 
   y = (1-t)*y₁ + t*y₂) :=
sorry

end NUMINAMATH_CALUDE_circle_and_line_intersection_l2924_292430


namespace NUMINAMATH_CALUDE_mailbox_distance_l2924_292493

/-- Represents the problem of finding the distance to a mailbox --/
def MailboxProblem (initial_speed : ℝ) (return_speed : ℝ) (time_away : ℝ) : Prop :=
  let initial_speed_mpm := initial_speed * 1000 / 60
  let return_speed_mpm := return_speed * 1000 / 60
  let distance_mother_in_law := initial_speed_mpm * time_away
  let total_distance := return_speed_mpm * time_away
  let distance_to_mailbox := (total_distance + distance_mother_in_law) / 2
  distance_to_mailbox = 200

/-- The theorem stating the solution to the mailbox problem --/
theorem mailbox_distance :
  MailboxProblem 3 5 3 := by
  sorry


end NUMINAMATH_CALUDE_mailbox_distance_l2924_292493


namespace NUMINAMATH_CALUDE_empty_solution_set_implies_a_leq_5_l2924_292406

-- Define the function representing the sum of distances
def f (x : ℝ) : ℝ := |x - 2| + |x + 3|

-- Theorem statement
theorem empty_solution_set_implies_a_leq_5 :
  ∀ a : ℝ, (∀ x : ℝ, f x ≥ a) → a ≤ 5 := by
  sorry

end NUMINAMATH_CALUDE_empty_solution_set_implies_a_leq_5_l2924_292406


namespace NUMINAMATH_CALUDE_cannot_reach_all_same_l2924_292488

/-- Represents the state of the circle of numbers -/
structure CircleState where
  ones : Nat
  zeros : Nat
  deriving Repr

/-- The operation performed on the circle each second -/
def next_state (s : CircleState) : CircleState :=
  sorry

/-- Predicate to check if all numbers in the circle are the same -/
def all_same (s : CircleState) : Prop :=
  s.ones = 0 ∨ s.zeros = 0

/-- The initial state of the circle -/
def initial_state : CircleState :=
  { ones := 4, zeros := 5 }

/-- Theorem stating that it's impossible to reach a state where all numbers are the same -/
theorem cannot_reach_all_same :
  ¬ ∃ (n : Nat), all_same (n.iterate next_state initial_state) :=
sorry

end NUMINAMATH_CALUDE_cannot_reach_all_same_l2924_292488


namespace NUMINAMATH_CALUDE_sum_reciprocal_equals_two_max_weighted_sum_reciprocal_l2924_292475

-- Define the variables and conditions
variable (a b x y : ℝ)
variable (ha : a > 1)
variable (hb : b > 1)
variable (hx : a^x = 2)
variable (hy : b^y = 2)

-- Theorem 1
theorem sum_reciprocal_equals_two (hab : a * b = 4) :
  1 / x + 1 / y = 2 := by sorry

-- Theorem 2
theorem max_weighted_sum_reciprocal (hab : a^2 + b = 8) :
  ∃ (m : ℝ), m = 4 ∧ ∀ (a b x y : ℝ), a > 1 → b > 1 → a^x = 2 → b^y = 2 → a^2 + b = 8 →
    2 / x + 1 / y ≤ m := by sorry

end NUMINAMATH_CALUDE_sum_reciprocal_equals_two_max_weighted_sum_reciprocal_l2924_292475


namespace NUMINAMATH_CALUDE_fourth_part_diminished_l2924_292405

theorem fourth_part_diminished (x : ℝ) (y : ℝ) (h : x = 280) (h2 : x/5 + 7 = x/4 - y) : y = 7 := by
  sorry

end NUMINAMATH_CALUDE_fourth_part_diminished_l2924_292405


namespace NUMINAMATH_CALUDE_power_of_seven_mod_hundred_l2924_292418

theorem power_of_seven_mod_hundred : 7^2010 % 100 = 49 := by
  sorry

end NUMINAMATH_CALUDE_power_of_seven_mod_hundred_l2924_292418


namespace NUMINAMATH_CALUDE_income_data_correction_l2924_292474

theorem income_data_correction (T : ℝ) : 
  let num_families : ℕ := 1200
  let largest_correct_income : ℝ := 102000
  let largest_incorrect_income : ℝ := 1020000
  let processing_fee : ℝ := 500
  let corrected_mean := (T + (largest_correct_income - processing_fee)) / num_families
  let incorrect_mean := (T + (largest_incorrect_income - processing_fee)) / num_families
  incorrect_mean - corrected_mean = 765 := by
sorry

end NUMINAMATH_CALUDE_income_data_correction_l2924_292474


namespace NUMINAMATH_CALUDE_baseball_cap_production_l2924_292489

theorem baseball_cap_production (caps_week1 caps_week2 caps_week3 total_4_weeks : ℕ) : 
  caps_week1 = 320 →
  caps_week3 = 300 →
  (caps_week1 + caps_week2 + caps_week3 + (caps_week1 + caps_week2 + caps_week3) / 3) = total_4_weeks →
  total_4_weeks = 1360 →
  caps_week2 = 400 := by
sorry

end NUMINAMATH_CALUDE_baseball_cap_production_l2924_292489


namespace NUMINAMATH_CALUDE_quadratic_inequality_range_l2924_292484

theorem quadratic_inequality_range (a : ℝ) : 
  (∃ x : ℝ, 2 * x^2 + (a - 1) * x + (1/2 : ℝ) ≤ 0) ↔ a ≤ -1 ∨ a ≥ 3 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_range_l2924_292484


namespace NUMINAMATH_CALUDE_floor_inequality_l2924_292473

theorem floor_inequality (α β : ℝ) : 
  ⌊2 * α⌋ + ⌊2 * β⌋ ≥ ⌊α⌋ + ⌊β⌋ + ⌊α + β⌋ := by sorry

end NUMINAMATH_CALUDE_floor_inequality_l2924_292473


namespace NUMINAMATH_CALUDE_unique_solution_quadratic_l2924_292499

theorem unique_solution_quadratic (k : ℝ) : 
  (∃! x : ℝ, (x + 3) * (x + 2) = k + 3 * x) ↔ k = 5 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_quadratic_l2924_292499


namespace NUMINAMATH_CALUDE_cubic_root_function_l2924_292420

/-- Given a function y = kx^(1/3) where y = 4 when x = 8, 
    prove that y = 6 when x = 27 -/
theorem cubic_root_function (k : ℝ) :
  (∃ y : ℝ, y = k * (8 : ℝ)^(1/3) ∧ y = 4) →
  k * (27 : ℝ)^(1/3) = 6 := by
  sorry

end NUMINAMATH_CALUDE_cubic_root_function_l2924_292420


namespace NUMINAMATH_CALUDE_simplify_sqrt_difference_l2924_292431

theorem simplify_sqrt_difference : 
  Real.sqrt (12 + 8 * Real.sqrt 3) - Real.sqrt (12 - 8 * Real.sqrt 3) = 2 * Real.sqrt 3 + 2 * Real.sqrt 6 := by
  sorry

end NUMINAMATH_CALUDE_simplify_sqrt_difference_l2924_292431


namespace NUMINAMATH_CALUDE_program_result_l2924_292408

def program_output (a₀ b₀ : ℕ) : ℕ × ℕ :=
  let a₁ := a₀ + b₀
  let b₁ := b₀ * a₁
  (a₁, b₁)

theorem program_result :
  program_output 1 3 = (4, 12) := by sorry

end NUMINAMATH_CALUDE_program_result_l2924_292408


namespace NUMINAMATH_CALUDE_triangle_properties_l2924_292454

-- Define the triangle ABC
structure Triangle :=
  (a b c : ℝ)
  (A B C : ℝ)

-- State the theorem
theorem triangle_properties (t : Triangle) 
  (h1 : (2 * t.a - t.b) * Real.cos t.C + 2 * t.c * Real.sin (t.B / 2) ^ 2 = t.c)
  (h2 : t.a + t.b = 4)
  (h3 : t.c = Real.sqrt 7) :
  t.C = π / 3 ∧ 
  (1 / 2 : ℝ) * t.a * t.b * Real.sin t.C = (3 * Real.sqrt 3) / 4 :=
by sorry

end NUMINAMATH_CALUDE_triangle_properties_l2924_292454


namespace NUMINAMATH_CALUDE_ticket_cost_difference_l2924_292479

def adult_count : ℕ := 9
def child_count : ℕ := 7
def adult_ticket_price : ℚ := 11
def child_ticket_price : ℚ := 7
def discount_rate : ℚ := 0.15
def discount_threshold : ℕ := 10

def total_tickets : ℕ := adult_count + child_count

def adult_total : ℚ := adult_count * adult_ticket_price
def child_total : ℚ := child_count * child_ticket_price
def total_cost : ℚ := adult_total + child_total

def discount_applies : Prop := total_tickets > discount_threshold

def discounted_cost : ℚ := total_cost * (1 - discount_rate)

def adult_proportion : ℚ := adult_total / total_cost
def child_proportion : ℚ := child_total / total_cost

def adult_discounted : ℚ := adult_total - (discount_rate * total_cost * adult_proportion)
def child_discounted : ℚ := child_total - (discount_rate * total_cost * child_proportion)

theorem ticket_cost_difference : 
  ∃ (ε : ℚ), ε > 0 ∧ ε < 0.01 ∧ |adult_discounted - child_discounted - 42.52| < ε :=
sorry

end NUMINAMATH_CALUDE_ticket_cost_difference_l2924_292479


namespace NUMINAMATH_CALUDE_triangle_angle_C_l2924_292432

theorem triangle_angle_C (a b c : ℝ) (A B C : ℝ) :
  a > 0 ∧ b > 0 ∧ c > 0 →
  A > 0 ∧ A < π ∧ B > 0 ∧ B < π ∧ C > 0 ∧ C < π →
  A + B + C = π →
  10 * a * Real.cos B = 3 * b * Real.cos A →
  Real.cos A = 5 * Real.sqrt 26 / 26 →
  C = 3 * π / 4 := by
sorry

end NUMINAMATH_CALUDE_triangle_angle_C_l2924_292432


namespace NUMINAMATH_CALUDE_curve_properties_l2924_292486

-- Define the curve C
def C (k : ℝ) := {(x, y) : ℝ × ℝ | x^2 / (4 - k) + y^2 / (k - 1) = 1}

-- Define what it means for C to be a circle
def is_circle (k : ℝ) := ∃ r : ℝ, ∀ (x y : ℝ), (x, y) ∈ C k → x^2 + y^2 = r^2

-- Define what it means for C to be an ellipse
def is_ellipse (k : ℝ) := ∃ a b : ℝ, a ≠ b ∧ a > 0 ∧ b > 0 ∧ ∀ (x y : ℝ), (x, y) ∈ C k → x^2 / a^2 + y^2 / b^2 = 1

-- Define what it means for C to be a hyperbola
def is_hyperbola (k : ℝ) := ∃ a b : ℝ, a ≠ 0 ∧ b ≠ 0 ∧ ∀ (x y : ℝ), (x, y) ∈ C k → x^2 / a^2 - y^2 / b^2 = 1 ∨ y^2 / a^2 - x^2 / b^2 = 1

-- Define what it means for C to be an ellipse with foci on the x-axis
def is_ellipse_x_foci (k : ℝ) := is_ellipse k ∧ ∃ c : ℝ, c > 0 ∧ ∀ (x y : ℝ), (x, y) ∈ C k → (x + c, y) ∈ C k ∧ (x - c, y) ∈ C k

theorem curve_properties :
  (∃ k : ℝ, is_circle k) ∧
  (∃ k : ℝ, 1 < k ∧ k < 4 ∧ ¬is_ellipse k) ∧
  (∀ k : ℝ, is_hyperbola k → k < 1 ∨ k > 4) ∧
  (∀ k : ℝ, is_ellipse_x_foci k → 1 < k ∧ k < 5/2) :=
sorry

end NUMINAMATH_CALUDE_curve_properties_l2924_292486


namespace NUMINAMATH_CALUDE_quadratic_expression_value_l2924_292456

theorem quadratic_expression_value (x y : ℝ) 
  (eq1 : 4 * x + y = 9) 
  (eq2 : x + 4 * y = 16) : 
  18 * x^2 + 20 * x * y + 18 * y^2 = 337 := by
sorry

end NUMINAMATH_CALUDE_quadratic_expression_value_l2924_292456


namespace NUMINAMATH_CALUDE_min_value_of_f_l2924_292465

open Real

noncomputable def f (x : ℝ) : ℝ := (log x)^2 / x

theorem min_value_of_f :
  ∀ x > 0, f x ≥ 0 ∧ ∃ x₀ > 0, f x₀ = 0 :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_f_l2924_292465


namespace NUMINAMATH_CALUDE_perfect_square_difference_l2924_292439

theorem perfect_square_difference (x y : ℝ) : (x - y)^2 = x^2 - 2*x*y + y^2 := by
  sorry

end NUMINAMATH_CALUDE_perfect_square_difference_l2924_292439
