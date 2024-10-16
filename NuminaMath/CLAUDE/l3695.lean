import Mathlib

namespace NUMINAMATH_CALUDE_volume_after_density_change_l3695_369507

/-- Given a substance with initial density and a density change factor, 
    calculate the new volume of a specified mass. -/
theorem volume_after_density_change 
  (initial_mass : ℝ) 
  (initial_volume : ℝ) 
  (density_change_factor : ℝ) 
  (mass_to_calculate : ℝ) 
  (h1 : initial_mass > 0)
  (h2 : initial_volume > 0)
  (h3 : density_change_factor > 0)
  (h4 : mass_to_calculate > 0)
  (h5 : initial_mass = 500)
  (h6 : initial_volume = 1)
  (h7 : density_change_factor = 1.25)
  (h8 : mass_to_calculate = 0.001) : 
  (mass_to_calculate / (initial_mass / initial_volume * density_change_factor)) * 1000000 = 1.6 := by
  sorry

#check volume_after_density_change

end NUMINAMATH_CALUDE_volume_after_density_change_l3695_369507


namespace NUMINAMATH_CALUDE_no_prime_pair_sum_51_l3695_369548

-- Define a function to check if a number is prime
def isPrime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(n % m = 0)

-- Theorem statement
theorem no_prime_pair_sum_51 :
  ¬∃ (p q : ℕ), isPrime p ∧ isPrime q ∧ p + q = 51 :=
sorry

end NUMINAMATH_CALUDE_no_prime_pair_sum_51_l3695_369548


namespace NUMINAMATH_CALUDE_digit_sum_problem_l3695_369520

theorem digit_sum_problem (a b c d : ℕ) 
  (h_diff : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d)
  (h_digits : a < 10 ∧ b < 10 ∧ c < 10 ∧ d < 10)
  (h1 : a + c = 10)
  (h2 : b + c + 1 = 10)
  (h3 : a + d + 1 = 10) :
  a + b + c + d = 18 := by
  sorry

end NUMINAMATH_CALUDE_digit_sum_problem_l3695_369520


namespace NUMINAMATH_CALUDE_variance_of_specific_random_variable_l3695_369560

/-- A random variable that takes values 0, 1, and 2 -/
structure RandomVariable where
  prob0 : ℝ
  prob1 : ℝ
  prob2 : ℝ
  sum_to_one : prob0 + prob1 + prob2 = 1
  nonnegative : prob0 ≥ 0 ∧ prob1 ≥ 0 ∧ prob2 ≥ 0

/-- The expectation of a random variable -/
def expectation (ξ : RandomVariable) : ℝ :=
  0 * ξ.prob0 + 1 * ξ.prob1 + 2 * ξ.prob2

/-- The variance of a random variable -/
def variance (ξ : RandomVariable) : ℝ :=
  (0 - expectation ξ)^2 * ξ.prob0 +
  (1 - expectation ξ)^2 * ξ.prob1 +
  (2 - expectation ξ)^2 * ξ.prob2

/-- Theorem: If P(ξ=0) = 1/5 and E(ξ) = 1, then D(ξ) = 2/5 -/
theorem variance_of_specific_random_variable :
  ∀ (ξ : RandomVariable),
    ξ.prob0 = 1/5 →
    expectation ξ = 1 →
    variance ξ = 2/5 := by
  sorry

end NUMINAMATH_CALUDE_variance_of_specific_random_variable_l3695_369560


namespace NUMINAMATH_CALUDE_plane_equation_proof_l3695_369597

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a vector in 3D space -/
structure Vector3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Defines a plane in 3D space using a point and a normal vector -/
structure Plane where
  point : Point3D
  normal : Vector3D

/-- Checks if a given equation represents the plane defined by a point and normal vector -/
def is_plane_equation (p : Plane) (a b c d : ℝ) : Prop :=
  ∀ (x y z : ℝ),
    (a * x + b * y + c * z + d = 0) ↔
    (x - p.point.x) * p.normal.x + (y - p.point.y) * p.normal.y + (z - p.point.z) * p.normal.z = 0

/-- The main theorem: proving that x + 2y - z - 2 = 0 is the equation of the specified plane -/
theorem plane_equation_proof :
  let A : Point3D := ⟨1, 2, 3⟩
  let n : Vector3D := ⟨-1, -2, 1⟩
  let p : Plane := ⟨A, n⟩
  is_plane_equation p 1 2 (-1) (-2) := by sorry

end NUMINAMATH_CALUDE_plane_equation_proof_l3695_369597


namespace NUMINAMATH_CALUDE_nalani_puppy_price_l3695_369537

/-- The price per puppy in Nalani's sale --/
def price_per_puppy (num_dogs : ℕ) (puppies_per_dog : ℕ) (fraction_sold : ℚ) (total_revenue : ℕ) : ℚ :=
  total_revenue / (fraction_sold * (num_dogs * puppies_per_dog))

/-- Theorem stating the price per puppy in Nalani's specific case --/
theorem nalani_puppy_price :
  price_per_puppy 2 10 (3/4) 3000 = 200 := by
  sorry

#eval price_per_puppy 2 10 (3/4) 3000

end NUMINAMATH_CALUDE_nalani_puppy_price_l3695_369537


namespace NUMINAMATH_CALUDE_smallest_enclosing_circle_radius_l3695_369534

theorem smallest_enclosing_circle_radius (r : ℝ) : 
  (∃ (A B C O : ℝ × ℝ),
    -- Three unit circles touching each other
    dist A B = 2 ∧ dist B C = 2 ∧ dist C A = 2 ∧
    -- O is the center of the enclosing circle
    dist O A = r ∧ dist O B = r ∧ dist O C = r ∧
    -- r is the smallest possible radius
    ∀ (r' : ℝ), (∃ (O' : ℝ × ℝ), dist O' A ≤ r' ∧ dist O' B ≤ r' ∧ dist O' C ≤ r') → r ≤ r') →
  r = 1 + 2 / Real.sqrt 3 :=
sorry

end NUMINAMATH_CALUDE_smallest_enclosing_circle_radius_l3695_369534


namespace NUMINAMATH_CALUDE_matrix_M_property_l3695_369522

def matrix_M (x : ℝ × ℝ) : ℝ × ℝ := sorry

theorem matrix_M_property (M : ℝ × ℝ → ℝ × ℝ) 
  (h1 : M (2, -1) = (3, 0))
  (h2 : M (-3, 5) = (-1, -1)) :
  M (5, 1) = (11, -1) := by sorry

end NUMINAMATH_CALUDE_matrix_M_property_l3695_369522


namespace NUMINAMATH_CALUDE_problem_solution_l3695_369590

theorem problem_solution : (1 / ((-5^4)^2)) * (-5)^9 = -5 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l3695_369590


namespace NUMINAMATH_CALUDE_intersection_point_implies_n_equals_two_l3695_369557

theorem intersection_point_implies_n_equals_two (n : ℕ+) 
  (x y : ℤ) -- x and y are integers
  (h1 : 15 * x + 18 * y = 1005) -- First line equation
  (h2 : y = n * x + 2) -- Second line equation
  : n = 2 := by
  sorry

end NUMINAMATH_CALUDE_intersection_point_implies_n_equals_two_l3695_369557


namespace NUMINAMATH_CALUDE_single_elimination_tournament_games_l3695_369594

/-- Represents a single-elimination tournament -/
structure Tournament :=
  (num_teams : ℕ)
  (no_ties : Bool)

/-- Calculates the number of games needed to declare a winner in a single-elimination tournament -/
def games_to_winner (t : Tournament) : ℕ :=
  t.num_teams - 1

/-- Theorem: In a single-elimination tournament with 23 teams and no ties, 
    the number of games required to declare a winner is 22 -/
theorem single_elimination_tournament_games :
  ∀ (t : Tournament), t.num_teams = 23 → t.no_ties = true → 
  games_to_winner t = 22 :=
by
  sorry

end NUMINAMATH_CALUDE_single_elimination_tournament_games_l3695_369594


namespace NUMINAMATH_CALUDE_two_digit_number_sum_l3695_369512

/-- Given a two-digit number n = 10a + b, where a and b are single digits,
    if the difference between n and its reverse is 7 times the sum of its digits,
    then the sum of n and its reverse is 99. -/
theorem two_digit_number_sum (a b : ℕ) (ha : a ≤ 9) (hb : b ≤ 9) (ha_pos : a > 0) :
  (10 * a + b) - (10 * b + a) = 7 * (a + b) →
  (10 * a + b) + (10 * b + a) = 99 := by
sorry

end NUMINAMATH_CALUDE_two_digit_number_sum_l3695_369512


namespace NUMINAMATH_CALUDE_unique_quadruple_l3695_369552

theorem unique_quadruple :
  ∃! (a b c d : ℝ), 
    0 ≤ a ∧ 0 ≤ b ∧ 0 ≤ c ∧ 0 ≤ d ∧
    a^2 + b^2 + c^2 + d^2 = 4 ∧
    (a + b + c + d)^3 = 8 ∧
    a = 1 ∧ b = 1 ∧ c = 1 ∧ d = 1 :=
by sorry

end NUMINAMATH_CALUDE_unique_quadruple_l3695_369552


namespace NUMINAMATH_CALUDE_houses_on_one_side_l3695_369587

theorem houses_on_one_side (x : ℕ) : 
  x + 3 * x = 160 → x = 40 := by sorry

end NUMINAMATH_CALUDE_houses_on_one_side_l3695_369587


namespace NUMINAMATH_CALUDE_meeting_speed_l3695_369506

theorem meeting_speed (distance : ℝ) (time : ℝ) (speed_difference : ℝ) 
  (h1 : distance = 200)
  (h2 : time = 8)
  (h3 : speed_difference = 7) :
  ∃ (speed : ℝ), 
    speed > 0 ∧ 
    (speed + (speed + speed_difference)) * time = distance ∧
    speed = 9 := by
  sorry

end NUMINAMATH_CALUDE_meeting_speed_l3695_369506


namespace NUMINAMATH_CALUDE_folded_triangle_area_l3695_369577

/-- Given a triangular piece of paper with area A when folded in half,
    the total area of the unfolded paper is 2A. -/
theorem folded_triangle_area (A : ℝ) (A_pos : A > 0) :
  let folded_area := A
  let total_area := 2 * A
  folded_area > 0 → total_area = 2 * folded_area :=
by sorry

end NUMINAMATH_CALUDE_folded_triangle_area_l3695_369577


namespace NUMINAMATH_CALUDE_library_visitors_average_l3695_369551

theorem library_visitors_average (sunday_visitors : ℕ) (other_day_visitors : ℕ) 
  (days_in_month : ℕ) (h1 : sunday_visitors = 570) (h2 : other_day_visitors = 240) 
  (h3 : days_in_month = 30) :
  let sundays := (days_in_month + 6) / 7
  let other_days := days_in_month - sundays
  let total_visitors := sundays * sunday_visitors + other_days * other_day_visitors
  total_visitors / days_in_month = 295 := by
sorry

end NUMINAMATH_CALUDE_library_visitors_average_l3695_369551


namespace NUMINAMATH_CALUDE_school_committee_formation_l3695_369502

def total_people : ℕ := 14
def students : ℕ := 11
def teachers : ℕ := 3
def committee_size : ℕ := 8

theorem school_committee_formation :
  (Nat.choose total_people committee_size) - (Nat.choose students committee_size) = 2838 :=
by sorry

end NUMINAMATH_CALUDE_school_committee_formation_l3695_369502


namespace NUMINAMATH_CALUDE_inequality_solution_set_l3695_369546

theorem inequality_solution_set :
  ∀ x : ℝ, (6 - x - 2 * x^2 < 0) ↔ (x > 3/2 ∨ x < -2) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l3695_369546


namespace NUMINAMATH_CALUDE_inequality_proof_l3695_369527

theorem inequality_proof (a b : ℝ) (h1 : -1 < b) (h2 : b < 0) (h3 : a < 0) :
  a * b > a * b^2 ∧ a * b^2 > a := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3695_369527


namespace NUMINAMATH_CALUDE_manny_marbles_l3695_369569

/-- Given a total of 120 marbles distributed in the ratio 4:5:6,
    prove that the person with the middle ratio (5) receives 40 marbles. -/
theorem manny_marbles (total : ℕ) (ratio_sum : ℕ) (manny_ratio : ℕ) :
  total = 120 →
  ratio_sum = 4 + 5 + 6 →
  manny_ratio = 5 →
  manny_ratio * (total / ratio_sum) = 40 :=
by sorry

end NUMINAMATH_CALUDE_manny_marbles_l3695_369569


namespace NUMINAMATH_CALUDE_area_ratio_theorem_l3695_369539

-- Define a triangle ABC
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

-- Define the circumradius and inradius
def circumradius (t : Triangle) : ℝ := sorry
def inradius (t : Triangle) : ℝ := sorry

-- Define the points A1, B1, C1
def angle_bisector_points (t : Triangle) : Triangle := sorry

-- Define the area of a triangle
def area (t : Triangle) : ℝ := sorry

theorem area_ratio_theorem (t : Triangle) :
  let t1 := angle_bisector_points t
  area t / area t1 = 2 * inradius t / circumradius t := by
  sorry

end NUMINAMATH_CALUDE_area_ratio_theorem_l3695_369539


namespace NUMINAMATH_CALUDE_cans_collected_l3695_369595

theorem cans_collected (total_cans : ℕ) (ladonna_cans : ℕ) (prikya_cans : ℕ) (yoki_cans : ℕ) :
  total_cans = 85 →
  ladonna_cans = 25 →
  prikya_cans = 2 * ladonna_cans →
  yoki_cans = total_cans - (ladonna_cans + prikya_cans) →
  yoki_cans = 10 := by
  sorry

end NUMINAMATH_CALUDE_cans_collected_l3695_369595


namespace NUMINAMATH_CALUDE_function_inequality_l3695_369500

open Set

theorem function_inequality (f : ℝ → ℝ) (a b : ℝ) :
  (∀ x, x > 0 → f x ≥ 0) →
  (∀ x, x > 0 → HasDerivAt f (f x) x) →
  (∀ x, x > 0 → x * (deriv f x) + f x < 0) →
  0 < a → 0 < b → a < b →
  b * f b < a * f a := by
  sorry

end NUMINAMATH_CALUDE_function_inequality_l3695_369500


namespace NUMINAMATH_CALUDE_largest_power_dividing_factorial_l3695_369582

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

theorem largest_power_dividing_factorial : 
  ∃ (k : ℕ), k = 63 ∧ 
  (∀ (m : ℕ), (2023 : ℕ)^m ∣ factorial 2023 → m ≤ k) ∧
  (2023 : ℕ)^k ∣ factorial 2023 :=
sorry

end NUMINAMATH_CALUDE_largest_power_dividing_factorial_l3695_369582


namespace NUMINAMATH_CALUDE_intersection_A_B_intersection_complement_A_B_l3695_369530

-- Define the sets A and B
def A : Set ℝ := {x | 2 * x - 4 < 0}
def B : Set ℝ := {x | 0 < x ∧ x < 5}

-- Define the universal set U
def U : Set ℝ := Set.univ

-- Theorem for part (I)
theorem intersection_A_B : A ∩ B = {x | 0 < x ∧ x < 2} := by sorry

-- Theorem for part (II)
theorem intersection_complement_A_B : (Set.compl A) ∩ B = {x | 2 ≤ x ∧ x < 5} := by sorry

end NUMINAMATH_CALUDE_intersection_A_B_intersection_complement_A_B_l3695_369530


namespace NUMINAMATH_CALUDE_largest_number_with_given_hcf_and_lcm_factors_l3695_369565

/-- Given three positive integers with HCF 37 and LCM with additional prime factors 17, 19, 23, and 29,
    the largest of these numbers is 7,976,237 -/
theorem largest_number_with_given_hcf_and_lcm_factors
  (a b c : ℕ+)
  (hcf_abc : Nat.gcd a b.val = 37 ∧ Nat.gcd (Nat.gcd a b.val) c.val = 37)
  (lcm_factors : ∃ (k : ℕ+), Nat.lcm (Nat.lcm a b.val) c.val = 37 * 17 * 19 * 23 * 29 * k) :
  max a (max b c) = 7976237 := by
  sorry

end NUMINAMATH_CALUDE_largest_number_with_given_hcf_and_lcm_factors_l3695_369565


namespace NUMINAMATH_CALUDE_max_product_sides_l3695_369516

/-- A convex quadrilateral with side lengths a, b, c, d and diagonal lengths e, f --/
structure ConvexQuadrilateral where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ
  e : ℝ
  f : ℝ
  convex : True  -- Assuming convexity without formal definition
  max_side : max a (max b (max c (max d (max e f)))) = 1

/-- The maximum product of side lengths in a convex quadrilateral with max side length 1 is 2 - √3 --/
theorem max_product_sides (q : ConvexQuadrilateral) : 
  ∃ (m : ℝ), m = q.a * q.b * q.c * q.d ∧ m ≤ 2 - Real.sqrt 3 := by
  sorry

#check max_product_sides

end NUMINAMATH_CALUDE_max_product_sides_l3695_369516


namespace NUMINAMATH_CALUDE_perpendicular_vectors_l3695_369558

-- Define the vectors a and b
def a : Fin 2 → ℝ := ![1, 2]
def b (x : ℝ) : Fin 2 → ℝ := ![x, -2]

-- Define the dot product of two 2D vectors
def dot_product (v w : Fin 2 → ℝ) : ℝ := (v 0) * (w 0) + (v 1) * (w 1)

-- State the theorem
theorem perpendicular_vectors (x : ℝ) :
  dot_product (λ i => a i + b x i) (λ i => a i - b x i) = 0 → x = 1 ∨ x = -1 := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_vectors_l3695_369558


namespace NUMINAMATH_CALUDE_roots_distinct_and_sum_integer_l3695_369508

/-- Given that a, b, c are roots of x^3 - x^2 - x - 1 = 0, prove they are distinct and
    that (a^1982 - b^1982)/(a - b) + (b^1982 - c^1982)/(b - c) + (c^1982 - a^1982)/(c - a) is an integer -/
theorem roots_distinct_and_sum_integer (a b c : ℂ) : 
  (a^3 - a^2 - a - 1 = 0) → 
  (b^3 - b^2 - b - 1 = 0) → 
  (c^3 - c^2 - c - 1 = 0) → 
  (a ≠ b ∧ b ≠ c ∧ c ≠ a) ∧ 
  (∃ k : ℤ, (a^1982 - b^1982)/(a - b) + (b^1982 - c^1982)/(b - c) + (c^1982 - a^1982)/(c - a) = k) := by
  sorry

end NUMINAMATH_CALUDE_roots_distinct_and_sum_integer_l3695_369508


namespace NUMINAMATH_CALUDE_beyonce_album_songs_l3695_369591

theorem beyonce_album_songs (singles : ℕ) (albums : ℕ) (songs_per_album : ℕ) (total_songs : ℕ) : 
  singles = 5 → albums = 2 → songs_per_album = 15 → total_songs = 55 → 
  total_songs - (singles + albums * songs_per_album) = 20 := by
sorry

end NUMINAMATH_CALUDE_beyonce_album_songs_l3695_369591


namespace NUMINAMATH_CALUDE_time_addition_theorem_l3695_369547

/-- Represents time in hours, minutes, and seconds -/
structure Time where
  hours : ℕ
  minutes : ℕ
  seconds : ℕ

/-- Adds a duration to a given time, wrapping around a 12-hour clock -/
def addTime (start : Time) (durationHours durationMinutes durationSeconds : ℕ) : Time :=
  sorry

/-- Converts 24-hour time to 12-hour time -/
def to12HourFormat (time : Time) : Time :=
  sorry

/-- Calculates the sum of hours, minutes, and seconds digits -/
def sumTimeDigits (time : Time) : ℕ :=
  sorry

theorem time_addition_theorem :
  let startTime := Time.mk 15 15 20
  let finalTime := addTime startTime 198 47 36
  let finalTime12Hour := to12HourFormat finalTime
  finalTime12Hour = Time.mk 10 2 56 ∧ sumTimeDigits finalTime12Hour = 68 := by
  sorry

end NUMINAMATH_CALUDE_time_addition_theorem_l3695_369547


namespace NUMINAMATH_CALUDE_some_number_equation_l3695_369524

/-- Given the equation x - 8 / 7 * 5 + 10 = 13.285714285714286, prove that x = 9 -/
theorem some_number_equation (x : ℝ) : x - 8 / 7 * 5 + 10 = 13.285714285714286 → x = 9 := by
  sorry

end NUMINAMATH_CALUDE_some_number_equation_l3695_369524


namespace NUMINAMATH_CALUDE_andrews_mangoes_l3695_369585

/-- Given Andrew's purchase of grapes and mangoes, prove the amount of mangoes bought -/
theorem andrews_mangoes :
  ∀ (mango_kg : ℝ),
  let grape_kg : ℝ := 8
  let grape_price : ℝ := 70
  let mango_price : ℝ := 55
  let total_cost : ℝ := 1055
  (grape_kg * grape_price + mango_kg * mango_price = total_cost) →
  mango_kg = 9 := by
sorry

end NUMINAMATH_CALUDE_andrews_mangoes_l3695_369585


namespace NUMINAMATH_CALUDE_one_minus_repeating_decimal_l3695_369511

/-- The value of the repeating decimal 0.123123... -/
def repeating_decimal : ℚ := 41 / 333

/-- Theorem: 1 - 0.123123... = 292/333 -/
theorem one_minus_repeating_decimal :
  1 - repeating_decimal = 292 / 333 := by
  sorry

end NUMINAMATH_CALUDE_one_minus_repeating_decimal_l3695_369511


namespace NUMINAMATH_CALUDE_power_sum_equality_l3695_369583

theorem power_sum_equality : (-2)^23 + 5^(2^4 + 3^3 - 4^2) = -8388608 + 5^27 := by
  sorry

end NUMINAMATH_CALUDE_power_sum_equality_l3695_369583


namespace NUMINAMATH_CALUDE_select_one_from_each_l3695_369579

theorem select_one_from_each : ∀ (n m : ℕ), n = 5 → m = 4 → n * m = 20 := by
  sorry

end NUMINAMATH_CALUDE_select_one_from_each_l3695_369579


namespace NUMINAMATH_CALUDE_pizzas_served_today_l3695_369528

theorem pizzas_served_today (lunch_pizzas dinner_pizzas : ℝ) 
  (h1 : lunch_pizzas = 12.5)
  (h2 : dinner_pizzas = 8.25) : 
  lunch_pizzas + dinner_pizzas = 20.75 := by
  sorry

end NUMINAMATH_CALUDE_pizzas_served_today_l3695_369528


namespace NUMINAMATH_CALUDE_fraction_product_theorem_l3695_369540

theorem fraction_product_theorem :
  (7 / 4 : ℚ) * (8 / 14 : ℚ) * (9 / 6 : ℚ) * (10 / 25 : ℚ) * 
  (28 / 21 : ℚ) * (15 / 45 : ℚ) * (32 / 16 : ℚ) * (50 / 100 : ℚ) = 4 / 5 := by
  sorry

end NUMINAMATH_CALUDE_fraction_product_theorem_l3695_369540


namespace NUMINAMATH_CALUDE_copper_wire_length_greater_than_225_l3695_369525

/-- Represents the properties of a copper wire -/
structure CopperWire where
  density : Real
  volume : Real
  diagonal : Real

/-- Theorem: The length of a copper wire with given properties is greater than 225 meters -/
theorem copper_wire_length_greater_than_225 (wire : CopperWire)
  (h1 : wire.density = 8900)
  (h2 : wire.volume = 0.5e-3)
  (h3 : wire.diagonal = 2e-3) :
  let cross_section_area := (wire.diagonal / Real.sqrt 2) ^ 2
  let length := wire.volume / cross_section_area
  length > 225 := by
  sorry

#check copper_wire_length_greater_than_225

end NUMINAMATH_CALUDE_copper_wire_length_greater_than_225_l3695_369525


namespace NUMINAMATH_CALUDE_number_problem_l3695_369553

theorem number_problem : ∃! x : ℚ, (1 / 4 : ℚ) * x > (1 / 5 : ℚ) * (x + 1) + 1 := by
  sorry

end NUMINAMATH_CALUDE_number_problem_l3695_369553


namespace NUMINAMATH_CALUDE_inequality_proof_l3695_369521

theorem inequality_proof (a b c : ℝ) :
  a^4 + b^4 + c^4 ≥ a^2*b^2 + b^2*c^2 + c^2*a^2 ∧
  a^2*b^2 + b^2*c^2 + c^2*a^2 ≥ a*b*c*(a + b + c) :=
by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3695_369521


namespace NUMINAMATH_CALUDE_solution_set_implies_ab_l3695_369576

theorem solution_set_implies_ab (a b : ℝ) : 
  (∀ x, x^2 + a*x + b ≤ 0 ↔ -1 ≤ x ∧ x ≤ 3) → a*b = 6 := by
  sorry

end NUMINAMATH_CALUDE_solution_set_implies_ab_l3695_369576


namespace NUMINAMATH_CALUDE_odd_digits_sum_152_345_l3695_369599

/-- Converts a base 10 number to base 4 -/
def toBase4 (n : ℕ) : List ℕ :=
  sorry

/-- Counts odd digits in a list of natural numbers -/
def countOddDigits (digits : List ℕ) : ℕ :=
  sorry

theorem odd_digits_sum_152_345 : 
  let base4_152 := toBase4 152
  let base4_345 := toBase4 345
  countOddDigits base4_152 + countOddDigits base4_345 = 6 := by
  sorry

end NUMINAMATH_CALUDE_odd_digits_sum_152_345_l3695_369599


namespace NUMINAMATH_CALUDE_divisibility_property_l3695_369563

theorem divisibility_property (p : ℕ) (h1 : p > 1) (h2 : Odd p) :
  ∃ k : ℤ, (p - 1 : ℤ) ^ ((p - 1) / 2) - 1 = (p - 2) * k := by
  sorry

end NUMINAMATH_CALUDE_divisibility_property_l3695_369563


namespace NUMINAMATH_CALUDE_greatest_sum_consecutive_integers_product_less_400_l3695_369549

theorem greatest_sum_consecutive_integers_product_less_400 :
  (∀ n : ℤ, n * (n + 1) < 400 → n + (n + 1) ≤ 39) ∧
  (∃ n : ℤ, n * (n + 1) < 400 ∧ n + (n + 1) = 39) :=
by sorry

end NUMINAMATH_CALUDE_greatest_sum_consecutive_integers_product_less_400_l3695_369549


namespace NUMINAMATH_CALUDE_dividend_calculation_l3695_369505

theorem dividend_calculation (divisor quotient remainder : ℕ) 
  (h1 : divisor = 40)
  (h2 : quotient = 6)
  (h3 : remainder = 28) :
  quotient * divisor + remainder = 268 :=
by sorry

end NUMINAMATH_CALUDE_dividend_calculation_l3695_369505


namespace NUMINAMATH_CALUDE_point_in_fourth_quadrant_l3695_369504

theorem point_in_fourth_quadrant (x : ℝ) : 
  let P : ℝ × ℝ := (x^2 + 1, -2)
  P.1 > 0 ∧ P.2 < 0 :=
by sorry

end NUMINAMATH_CALUDE_point_in_fourth_quadrant_l3695_369504


namespace NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l3695_369501

theorem imaginary_part_of_complex_fraction : Complex.im ((1 + Complex.I) / (1 - Complex.I)) = 1 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l3695_369501


namespace NUMINAMATH_CALUDE_square_area_on_parabola_l3695_369571

/-- The area of a square with one side on y = 7 and endpoints on y = x^2 + 4x + 3 is 32 -/
theorem square_area_on_parabola : ∃ (s : ℝ),
  (∃ (x₁ x₂ : ℝ),
    x₁^2 + 4*x₁ + 3 = 7 ∧
    x₂^2 + 4*x₂ + 3 = 7 ∧
    s = |x₂ - x₁|) →
  s^2 = 32 := by
  sorry

end NUMINAMATH_CALUDE_square_area_on_parabola_l3695_369571


namespace NUMINAMATH_CALUDE_units_digit_of_23_times_51_squared_l3695_369550

theorem units_digit_of_23_times_51_squared : ∃ n : ℕ, 23 * 51^2 = 10 * n + 3 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_23_times_51_squared_l3695_369550


namespace NUMINAMATH_CALUDE_apple_production_formula_l3695_369535

/-- Represents an apple orchard with additional trees planted -/
structure Orchard where
  initial_trees : ℕ
  initial_avg_apples : ℕ
  decrease_per_tree : ℕ
  additional_trees : ℕ

/-- Calculates the total number of apples produced in an orchard -/
def total_apples (o : Orchard) : ℕ :=
  (o.initial_trees + o.additional_trees) * (o.initial_avg_apples - o.decrease_per_tree * o.additional_trees)

/-- Theorem stating the relationship between additional trees and total apples -/
theorem apple_production_formula (x : ℕ) :
  let o : Orchard := {
    initial_trees := 10,
    initial_avg_apples := 200,
    decrease_per_tree := 5,
    additional_trees := x
  }
  total_apples o = (10 + x) * (200 - 5 * x) := by
  sorry

end NUMINAMATH_CALUDE_apple_production_formula_l3695_369535


namespace NUMINAMATH_CALUDE_oranges_thrown_away_l3695_369526

theorem oranges_thrown_away (initial : ℕ) (new : ℕ) (final : ℕ) : 
  initial = 40 → new = 21 → final = 36 → initial - (initial - final + new) = 25 := by
  sorry

end NUMINAMATH_CALUDE_oranges_thrown_away_l3695_369526


namespace NUMINAMATH_CALUDE_total_pokemon_cards_l3695_369514

/-- The number of people with Pokemon cards -/
def num_people : ℕ := 4

/-- The number of dozens of cards each person has -/
def dozens_per_person : ℕ := 9

/-- The number of items in one dozen -/
def items_per_dozen : ℕ := 12

/-- Theorem: The total number of Pokemon cards owned by 4 people is 432,
    given that each person has 9 dozen cards and one dozen equals 12 items. -/
theorem total_pokemon_cards :
  num_people * dozens_per_person * items_per_dozen = 432 := by
  sorry

end NUMINAMATH_CALUDE_total_pokemon_cards_l3695_369514


namespace NUMINAMATH_CALUDE_ralph_tv_hours_l3695_369578

/-- The number of hours Ralph watches TV each day from Monday to Friday -/
def weekday_hours : ℝ := sorry

/-- The number of hours Ralph watches TV each day on Saturday and Sunday -/
def weekend_hours : ℝ := 6

/-- The total number of hours Ralph watches TV in one week -/
def total_weekly_hours : ℝ := 32

/-- Theorem stating that Ralph watches TV for 4 hours each day from Monday to Friday -/
theorem ralph_tv_hours : weekday_hours = 4 := by
  have h1 : 5 * weekday_hours + 2 * weekend_hours = total_weekly_hours := sorry
  sorry

end NUMINAMATH_CALUDE_ralph_tv_hours_l3695_369578


namespace NUMINAMATH_CALUDE_max_min_f_on_interval_l3695_369564

def f (x : ℝ) : ℝ := 2 * x^3 - 3 * x^2 - 12 * x + 5

theorem max_min_f_on_interval :
  ∃ (max min : ℝ), max = 5 ∧ min = -15 ∧
  (∀ x ∈ Set.Icc 0 3, f x ≤ max) ∧
  (∀ x ∈ Set.Icc 0 3, min ≤ f x) ∧
  (∃ x ∈ Set.Icc 0 3, f x = max) ∧
  (∃ x ∈ Set.Icc 0 3, f x = min) :=
sorry

end NUMINAMATH_CALUDE_max_min_f_on_interval_l3695_369564


namespace NUMINAMATH_CALUDE_freshman_psychology_percentage_l3695_369575

/-- Represents the percentage of students in a school -/
def school_percentage (school : String) : ℝ :=
  match school with
  | "Liberal Arts" => 0.20
  | "Science" => 0.15
  | "Business" => 0.25
  | "Engineering" => 0.10
  | "Education" => 0.20
  | "Health Sciences" => 0.10
  | _ => 0

/-- Represents the percentage of students in a year -/
def year_percentage (year : String) : ℝ :=
  match year with
  | "Freshman" => 0.30
  | "Sophomore" => 0.25
  | "Junior" => 0.20
  | "Senior" => 0.15
  | "Graduate" => 0.10
  | _ => 0

/-- Represents the percentage of students in a major for a specific year in Liberal Arts -/
def liberal_arts_major_percentage (year : String) (major : String) : ℝ :=
  match year, major with
  | "Freshman", "Psychology" => 0.30
  | "Freshman", "Sociology" => 0.20
  | "Freshman", "History" => 0.20
  | "Freshman", "English" => 0.15
  | "Freshman", "Philosophy" => 0.15
  | _, _ => 0  -- Other combinations not needed for this problem

/-- The theorem to be proved -/
theorem freshman_psychology_percentage : 
  school_percentage "Liberal Arts" * 
  year_percentage "Freshman" * 
  liberal_arts_major_percentage "Freshman" "Psychology" = 0.018 := by
  sorry

end NUMINAMATH_CALUDE_freshman_psychology_percentage_l3695_369575


namespace NUMINAMATH_CALUDE_part_one_part_two_l3695_369518

-- Definition of balanced numbers
def balanced (a b n : ℤ) : Prop := a + b = n

-- Part 1
theorem part_one : balanced (-6) 8 2 := by sorry

-- Part 2
theorem part_two (k : ℤ) (h : ∀ x : ℤ, ∃ n : ℤ, balanced (6*x^2 - 4*k*x + 8) (-2*(3*x^2 - 2*x + k)) n) :
  ∃ n : ℤ, (∀ x : ℤ, balanced (6*x^2 - 4*k*x + 8) (-2*(3*x^2 - 2*x + k)) n) ∧ n = 6 := by sorry

end NUMINAMATH_CALUDE_part_one_part_two_l3695_369518


namespace NUMINAMATH_CALUDE_people_who_left_line_l3695_369586

theorem people_who_left_line (initial_people : ℕ) (joined : ℕ) (final_people : ℕ) 
  (h1 : initial_people = 9)
  (h2 : joined = 3)
  (h3 : final_people = 6)
  : initial_people - (initial_people - joined + final_people) = 6 := by
  sorry

end NUMINAMATH_CALUDE_people_who_left_line_l3695_369586


namespace NUMINAMATH_CALUDE_sum_of_coordinates_is_14_l3695_369559

/-- Given two points A and B in a 2D plane, where:
  - A is at (0, 0)
  - B is on the line y = 6
  - The slope of segment AB is 3/4
  This theorem proves that the sum of the x- and y-coordinates of point B is 14. -/
theorem sum_of_coordinates_is_14 (B : ℝ × ℝ) : 
  B.2 = 6 ∧ 
  (B.2 - 0) / (B.1 - 0) = 3 / 4 →
  B.1 + B.2 = 14 := by
sorry

end NUMINAMATH_CALUDE_sum_of_coordinates_is_14_l3695_369559


namespace NUMINAMATH_CALUDE_total_accidents_l3695_369581

/-- Represents the accident rate and total traffic for a highway -/
structure HighwayData where
  accidents : ℕ
  per_vehicles : ℕ
  total_vehicles : ℕ

/-- Calculates the number of accidents for a given highway -/
def calculate_accidents (data : HighwayData) : ℕ :=
  (data.accidents * data.total_vehicles) / data.per_vehicles

/-- The given data for the three highways -/
def highway_A : HighwayData := ⟨75, 100, 2500⟩
def highway_B : HighwayData := ⟨50, 80, 1600⟩
def highway_C : HighwayData := ⟨90, 200, 1900⟩

/-- The theorem stating the total number of accidents across all three highways -/
theorem total_accidents :
  calculate_accidents highway_A +
  calculate_accidents highway_B +
  calculate_accidents highway_C = 3730 := by
  sorry

end NUMINAMATH_CALUDE_total_accidents_l3695_369581


namespace NUMINAMATH_CALUDE_log_expression_equals_two_l3695_369567

-- Define the logarithm base 10 function
noncomputable def log10 (x : ℝ) : ℝ := Real.log x / Real.log 10

-- State the theorem
theorem log_expression_equals_two :
  (log10 2)^2 + log10 2 * log10 50 + log10 25 = 2 := by sorry

end NUMINAMATH_CALUDE_log_expression_equals_two_l3695_369567


namespace NUMINAMATH_CALUDE_right_to_left_grouping_l3695_369596

/-- Evaluates an expression using right-to-left grouping -/
def evaluateRightToLeft (a b c d : ℤ) : ℚ :=
  a / (b - c * d^2)

/-- The original expression as a function -/
def originalExpression (a b c d : ℤ) : ℚ :=
  a / b - c * d^2

theorem right_to_left_grouping (a b c d : ℤ) :
  evaluateRightToLeft a b c d = originalExpression a b c d :=
sorry

end NUMINAMATH_CALUDE_right_to_left_grouping_l3695_369596


namespace NUMINAMATH_CALUDE_cyrus_family_size_cyrus_mosquito_bites_l3695_369574

theorem cyrus_family_size (cyrus_arms_legs : ℕ) (cyrus_body : ℕ) : ℕ :=
  let cyrus_total := cyrus_arms_legs + cyrus_body
  let family_total := cyrus_total / 2
  family_total

theorem cyrus_mosquito_bites : cyrus_family_size 14 10 = 12 := by
  sorry

end NUMINAMATH_CALUDE_cyrus_family_size_cyrus_mosquito_bites_l3695_369574


namespace NUMINAMATH_CALUDE_pie_eating_contest_l3695_369531

theorem pie_eating_contest (adam bill sierra : ℕ) : 
  adam = bill + 3 →
  sierra = 2 * bill →
  sierra = 12 →
  adam + bill + sierra = 27 := by
sorry

end NUMINAMATH_CALUDE_pie_eating_contest_l3695_369531


namespace NUMINAMATH_CALUDE_jellybean_difference_l3695_369509

theorem jellybean_difference (total : ℕ) (black : ℕ) (green : ℕ) (orange : ℕ) : 
  total = 27 →
  black = 8 →
  orange = green - 1 →
  total = black + green + orange →
  green - black = 2 := by
sorry

end NUMINAMATH_CALUDE_jellybean_difference_l3695_369509


namespace NUMINAMATH_CALUDE_necessary_but_not_sufficient_l3695_369580

def M : Set ℝ := {x : ℝ | 0 < x ∧ x ≤ 3}
def N : Set ℝ := {x : ℝ | 0 < x ∧ x ≤ 2}

theorem necessary_but_not_sufficient :
  (∀ a : ℝ, a ∈ N → a ∈ M) ∧
  (∃ a : ℝ, a ∈ M ∧ a ∉ N) :=
by sorry

end NUMINAMATH_CALUDE_necessary_but_not_sufficient_l3695_369580


namespace NUMINAMATH_CALUDE_parabola_chord_sum_constant_l3695_369515

/-- Theorem: For a parabola y = x^2, if there exists a constant d such that
    for all chords AB passing through D = (0,d), the sum s = 1/AD^2 + 1/BD^2 is constant,
    then d = 1/2 and s = 4. -/
theorem parabola_chord_sum_constant (d : ℝ) :
  (∃ (s : ℝ), ∀ (A B : ℝ × ℝ),
    A.2 = A.1^2 ∧ B.2 = B.1^2 ∧  -- A and B are on the parabola y = x^2
    (∃ (m : ℝ), A.2 = m * A.1 + d ∧ B.2 = m * B.1 + d) →  -- AB passes through (0,d)
    1 / ((A.1^2 + (A.2 - d)^2) : ℝ) + 1 / ((B.1^2 + (B.2 - d)^2) : ℝ) = s) →
  d = 1/2 ∧ s = 4 :=
sorry

end NUMINAMATH_CALUDE_parabola_chord_sum_constant_l3695_369515


namespace NUMINAMATH_CALUDE_product_of_numbers_l3695_369556

theorem product_of_numbers (x y : ℝ) (h1 : 0 < x) (h2 : 0 < y) (h3 : y / x = 15) (h4 : x + y = 400) : x * y = 9375 := by
  sorry

end NUMINAMATH_CALUDE_product_of_numbers_l3695_369556


namespace NUMINAMATH_CALUDE_line_intersects_circle_l3695_369593

def Circle (x y : ℝ) : Prop := x^2 + y^2 = 4

def Line (x₀ y₀ x y : ℝ) : Prop := x₀ * x + y₀ * y = 4

def PointOutsideCircle (x₀ y₀ : ℝ) : Prop := x₀^2 + y₀^2 > 4

theorem line_intersects_circle (x₀ y₀ : ℝ) 
  (h1 : PointOutsideCircle x₀ y₀) :
  ∃ x y : ℝ, Circle x y ∧ Line x₀ y₀ x y :=
sorry

end NUMINAMATH_CALUDE_line_intersects_circle_l3695_369593


namespace NUMINAMATH_CALUDE_triangle_area_l3695_369554

/-- Given a triangle with perimeter 20 cm and inradius 2.5 cm, its area is 25 cm². -/
theorem triangle_area (p r A : ℝ) : 
  p = 20 → r = 2.5 → A = r * (p / 2) → A = 25 :=
by sorry

end NUMINAMATH_CALUDE_triangle_area_l3695_369554


namespace NUMINAMATH_CALUDE_circle_equation_l3695_369544

-- Define the circle C
def Circle (center : ℝ × ℝ) (radius : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1 - center.1)^2 + (p.2 - center.2)^2 = radius^2}

-- Define the line l: 3x - y - 3 = 0
def Line : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | 3 * p.1 - p.2 - 3 = 0}

theorem circle_equation :
  ∃ (center : ℝ × ℝ) (radius : ℝ),
    (center ∈ Line) ∧
    ((2, 5) ∈ Circle center radius) ∧
    ((4, 3) ∈ Circle center radius) ∧
    (Circle center radius = {p : ℝ × ℝ | (p.1 - 2)^2 + (p.2 - 3)^2 = 4}) := by
  sorry

end NUMINAMATH_CALUDE_circle_equation_l3695_369544


namespace NUMINAMATH_CALUDE_quadrilateral_diagonal_length_l3695_369533

/-- A convex quadrilateral in a plane -/
structure ConvexQuadrilateral where
  -- We don't need to define the specific properties of a convex quadrilateral here
  -- as they are not directly used in the problem statement

/-- The theorem stating the relation between the area, sum of sides and diagonals
    in a specific convex quadrilateral -/
theorem quadrilateral_diagonal_length 
  (Q : ConvexQuadrilateral) 
  (area : ℝ) 
  (sum_sides_and_diagonal : ℝ) 
  (h1 : area = 32) 
  (h2 : sum_sides_and_diagonal = 16) : 
  ∃ (other_diagonal : ℝ), other_diagonal = 8 * Real.sqrt 2 :=
sorry

end NUMINAMATH_CALUDE_quadrilateral_diagonal_length_l3695_369533


namespace NUMINAMATH_CALUDE_butterfat_solution_l3695_369532

def butterfat_problem (x : ℝ) : Prop :=
  let initial_volume : ℝ := 8
  let added_volume : ℝ := 20
  let initial_butterfat : ℝ := x / 100
  let added_butterfat : ℝ := 10 / 100
  let final_butterfat : ℝ := 20 / 100
  let total_volume : ℝ := initial_volume + added_volume
  (initial_volume * initial_butterfat + added_volume * added_butterfat) / total_volume = final_butterfat

theorem butterfat_solution : butterfat_problem 45 := by
  sorry

end NUMINAMATH_CALUDE_butterfat_solution_l3695_369532


namespace NUMINAMATH_CALUDE_simplify_fraction_l3695_369592

theorem simplify_fraction : 18 * (8 / 12) * (1 / 6) = 2 := by sorry

end NUMINAMATH_CALUDE_simplify_fraction_l3695_369592


namespace NUMINAMATH_CALUDE_symmetric_point_xOy_l3695_369523

def xOy_plane : Set (ℝ × ℝ × ℝ) := {p | p.2.2 = 0}

def symmetric_point (p : ℝ × ℝ × ℝ) (plane : Set (ℝ × ℝ × ℝ)) : ℝ × ℝ × ℝ :=
  (p.1, p.2.1, -p.2.2)

theorem symmetric_point_xOy : 
  symmetric_point (2, 3, 4) xOy_plane = (2, 3, -4) := by sorry

end NUMINAMATH_CALUDE_symmetric_point_xOy_l3695_369523


namespace NUMINAMATH_CALUDE_sum_is_composite_l3695_369545

theorem sum_is_composite (a b : ℕ) (h : 31 * a = 54 * b) : ∃ (k m : ℕ), k > 1 ∧ m > 1 ∧ a + b = k * m := by
  sorry

end NUMINAMATH_CALUDE_sum_is_composite_l3695_369545


namespace NUMINAMATH_CALUDE_prob_n₂_div_2310_eq_l3695_369538

/-- The product of the first 25 primes -/
def n₀ : ℕ := sorry

/-- The Euler totient function -/
def φ : ℕ → ℕ := sorry

/-- The probability of choosing a divisor n of m, proportional to φ(n) -/
def prob_divisor (n m : ℕ) : ℚ := sorry

/-- The probability that a randomly chosen n₂ (which is a random divisor of n₁, 
    which itself is a random divisor of n₀) is divisible by 2310 -/
def prob_n₂_div_2310 : ℚ := sorry

/-- Main theorem: The probability that n₂ ≡ 0 (mod 2310) is 256/5929 -/
theorem prob_n₂_div_2310_eq : prob_n₂_div_2310 = 256 / 5929 := by sorry

end NUMINAMATH_CALUDE_prob_n₂_div_2310_eq_l3695_369538


namespace NUMINAMATH_CALUDE_seventh_grade_class_size_l3695_369529

theorem seventh_grade_class_size :
  let excellent_chinese : ℕ := 15
  let excellent_math : ℕ := 18
  let excellent_both : ℕ := 8
  let not_excellent : ℕ := 20
  excellent_chinese + excellent_math - excellent_both + not_excellent = 45 := by
  sorry

end NUMINAMATH_CALUDE_seventh_grade_class_size_l3695_369529


namespace NUMINAMATH_CALUDE_alarm_set_time_l3695_369519

/-- Represents time in hours and minutes -/
structure Time where
  hours : ℕ
  minutes : ℕ
  inv_minutes : minutes < 60

/-- Converts time to total minutes -/
def Time.toMinutes (t : Time) : ℕ := t.hours * 60 + t.minutes

/-- The rate at which the faster watch gains time (in minutes per hour) -/
def gainRate : ℕ := 2

/-- The time shown on the faster watch when the alarm rings -/
def fasterWatchTime : Time := ⟨4, 12, by norm_num⟩

/-- The correct time when the alarm rings -/
def correctTime : Time := ⟨4, 0, by norm_num⟩

/-- Calculates the number of hours passed based on the time difference and gain rate -/
def hoursPassed (timeDiff : ℕ) (rate : ℕ) : ℚ := timeDiff / rate

theorem alarm_set_time :
  let timeDiff := fasterWatchTime.toMinutes - correctTime.toMinutes
  let hours := hoursPassed timeDiff gainRate
  (correctTime.hours - hours.floor : ℤ) = 22 := by sorry

end NUMINAMATH_CALUDE_alarm_set_time_l3695_369519


namespace NUMINAMATH_CALUDE_complex_argument_range_l3695_369541

theorem complex_argument_range (z : ℂ) (h : Complex.abs (2 * z + z⁻¹) = 1) :
  let arg := Complex.arg z
  arg ∈ (Set.Icc (Real.arccos (Real.sqrt 2 / 4)) (Real.pi - Real.arccos (Real.sqrt 2 / 4))) ∪
           (Set.Icc (Real.pi + Real.arccos (Real.sqrt 2 / 4)) (2 * Real.pi - Real.arccos (Real.sqrt 2 / 4))) :=
by sorry

end NUMINAMATH_CALUDE_complex_argument_range_l3695_369541


namespace NUMINAMATH_CALUDE_greg_adam_marble_difference_l3695_369584

/-- Given that Adam has 29 marbles, Greg has 43 marbles, and Greg has more marbles than Adam,
    prove that Greg has 14 more marbles than Adam. -/
theorem greg_adam_marble_difference :
  ∀ (adam_marbles greg_marbles : ℕ),
    adam_marbles = 29 →
    greg_marbles = 43 →
    greg_marbles > adam_marbles →
    greg_marbles - adam_marbles = 14 := by
  sorry

end NUMINAMATH_CALUDE_greg_adam_marble_difference_l3695_369584


namespace NUMINAMATH_CALUDE_smallest_number_l3695_369542

theorem smallest_number : 
  let a := (2010 : ℝ) ^ (1 / 209)
  let b := (2009 : ℝ) ^ (1 / 200)
  let c := (2010 : ℝ)
  let d := (2010 : ℝ) / 2009
  let e := (2009 : ℝ) / 2010
  (e ≤ a) ∧ (e ≤ b) ∧ (e ≤ c) ∧ (e ≤ d) ∧ (e ≤ e) :=
by sorry

end NUMINAMATH_CALUDE_smallest_number_l3695_369542


namespace NUMINAMATH_CALUDE_abs_eq_sqrt_square_l3695_369536

theorem abs_eq_sqrt_square (x : ℝ) : |x| = Real.sqrt (x^2) := by
  sorry

end NUMINAMATH_CALUDE_abs_eq_sqrt_square_l3695_369536


namespace NUMINAMATH_CALUDE_rohan_salary_l3695_369513

/-- Rohan's monthly salary in Rupees -/
def monthly_salary : ℝ := 5000

/-- Percentage of salary spent on food -/
def food_percentage : ℝ := 40

/-- Percentage of salary spent on house rent -/
def rent_percentage : ℝ := 20

/-- Percentage of salary spent on entertainment -/
def entertainment_percentage : ℝ := 10

/-- Percentage of salary spent on conveyance -/
def conveyance_percentage : ℝ := 10

/-- Rohan's savings at the end of the month in Rupees -/
def savings : ℝ := 1000

theorem rohan_salary :
  monthly_salary * (1 - (food_percentage + rent_percentage + entertainment_percentage + conveyance_percentage) / 100) = savings :=
by sorry

end NUMINAMATH_CALUDE_rohan_salary_l3695_369513


namespace NUMINAMATH_CALUDE_ratio_of_fraction_equation_l3695_369598

theorem ratio_of_fraction_equation (P Q : ℤ) : 
  (∀ x : ℝ, x ≠ 3 ∧ x ≠ -3 ∧ x ≠ 2 ∧ x ≠ 0 → 
    (P / (x - 3) + Q / (x^2 + x - 6) = (x^2 + 3*x + 1) / (x^3 - x^2 - 12*x))) →
  Q / P = -6 / 13 := by
  sorry

end NUMINAMATH_CALUDE_ratio_of_fraction_equation_l3695_369598


namespace NUMINAMATH_CALUDE_parabola_line_intersection_l3695_369568

/-- A parabola with vertex at origin and axis of symmetry along x-axis -/
structure Parabola where
  p : ℝ
  eq : ℝ → ℝ → Prop := fun x y => y^2 = 2 * p * x

/-- A line with slope k passing through a fixed point -/
structure Line where
  k : ℝ
  fixed_point : ℝ × ℝ
  eq : ℝ → ℝ → Prop := fun x y => y = k * x + (fixed_point.2 - k * fixed_point.1)

/-- The number of intersection points between a parabola and a line -/
def intersection_count (par : Parabola) (l : Line) : ℕ :=
  sorry

theorem parabola_line_intersection 
  (par : Parabola) 
  (h_par : par.eq (1/2) (-Real.sqrt 2))
  (l : Line)
  (h_line : l.fixed_point = (-2, 1)) :
  (intersection_count par l = 2) ↔ 
  (-1 < l.k ∧ l.k < 1/2 ∧ l.k ≠ 0) :=
sorry

end NUMINAMATH_CALUDE_parabola_line_intersection_l3695_369568


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l3695_369573

theorem quadratic_equation_solution : ∃ x₁ x₂ : ℝ, 
  (x₁ = 1 + Real.sqrt 2 ∧ x₂ = 1 - Real.sqrt 2) ∧
  (x₁^2 - 2*x₁ - 1 = 0 ∧ x₂^2 - 2*x₂ - 1 = 0) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l3695_369573


namespace NUMINAMATH_CALUDE_petya_bus_catch_l3695_369570

/-- Represents the maximum distance between bus stops that allows Petya to always catch the bus -/
def max_bus_stop_distance (v_p : ℝ) : ℝ :=
  0.12

/-- Theorem stating the maximum distance between bus stops for Petya to always catch the bus -/
theorem petya_bus_catch (v_p : ℝ) (h_v_p : v_p > 0) :
  let v_b := 5 * v_p
  let max_observation_distance := 0.6
  ∀ d : ℝ, d > 0 → d ≤ max_bus_stop_distance v_p →
    (∀ t : ℝ, t ≥ 0 → 
      (v_p * t ≤ d ∧ v_b * t ≤ max_observation_distance) ∨
      (v_p * t ≤ 2 * d ∧ v_b * t ≤ d + max_observation_distance)) :=
by
  sorry

end NUMINAMATH_CALUDE_petya_bus_catch_l3695_369570


namespace NUMINAMATH_CALUDE_quadratic_form_identity_l3695_369555

theorem quadratic_form_identity 
  (a b c d e f x y z : ℝ) 
  (h : a * x^2 + b * y^2 + c * z^2 + 2 * d * y * z + 2 * e * z * x + 2 * f * x * y = 0) :
  (d * y * z + e * z * x + f * x * y)^2 - b * c * y^2 * z^2 - c * a * z^2 * x^2 - a * b * x^2 * y^2 = 
  (1/4) * (x * Real.sqrt a + y * Real.sqrt b + z * Real.sqrt c) *
          (x * Real.sqrt a - y * Real.sqrt b + z * Real.sqrt c) *
          (x * Real.sqrt a + y * Real.sqrt b - z * Real.sqrt c) *
          (x * Real.sqrt a - y * Real.sqrt b - z * Real.sqrt c) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_form_identity_l3695_369555


namespace NUMINAMATH_CALUDE_continuous_function_with_property_l3695_369510

-- Define the property of the function
def has_property (f : ℝ → ℝ) : Prop :=
  ∀ (n : ℤ) (x : ℝ), n ≠ 0 → f (x + 1 / (n : ℝ)) ≤ f x + 1 / (n : ℝ)

-- State the theorem
theorem continuous_function_with_property (f : ℝ → ℝ) 
  (hf : Continuous f) (hprop : has_property f) :
  ∃ (a : ℝ), ∀ x, f x = x + a := by
  sorry

end NUMINAMATH_CALUDE_continuous_function_with_property_l3695_369510


namespace NUMINAMATH_CALUDE_stating_probability_theorem_l3695_369572

/-- Represents the number of guests -/
def num_guests : ℕ := 3

/-- Represents the number of roll types -/
def num_roll_types : ℕ := 4

/-- Represents the total number of rolls -/
def total_rolls : ℕ := 12

/-- Represents the number of rolls per guest -/
def rolls_per_guest : ℕ := 4

/-- Represents the number of each type of roll -/
def rolls_per_type : ℕ := 3

/-- 
Calculates the probability that each guest receives one roll of each type 
when rolls are randomly distributed.
-/
def probability_all_different_rolls : ℚ := sorry

/-- 
Theorem stating that the probability of each guest receiving one roll of each type 
is equal to 2/165720
-/
theorem probability_theorem : 
  probability_all_different_rolls = 2 / 165720 := sorry

end NUMINAMATH_CALUDE_stating_probability_theorem_l3695_369572


namespace NUMINAMATH_CALUDE_geometric_arithmetic_interleaving_l3695_369561

theorem geometric_arithmetic_interleaving (n : ℕ) (h : n > 3) :
  ∃ (x y : ℕ → ℕ),
    (∀ i, i < n → x i > 0) ∧
    (∀ i, i < n → y i > 0) ∧
    (∃ r : ℚ, r > 1 ∧ ∀ i, i < n - 1 → x (i + 1) = (x i : ℚ) * r) ∧
    (∃ d : ℚ, d > 0 ∧ ∀ i, i < n - 1 → y (i + 1) = y i + d) ∧
    (∀ i, i < n - 1 → x i < y i ∧ y i < x (i + 1)) ∧
    x (n - 1) < y (n - 1) :=
by sorry

end NUMINAMATH_CALUDE_geometric_arithmetic_interleaving_l3695_369561


namespace NUMINAMATH_CALUDE_min_value_expression_l3695_369543

theorem min_value_expression (a x y z : ℝ) 
  (ha : 0 < a ∧ a < 1) 
  (hx : -a < x ∧ x < a) 
  (hy : -a < y ∧ y < a) 
  (hz : -a < z ∧ z < a) : 
  (∀ x y z, 1 / ((1 - x) * (1 - y) * (1 - z)) + 1 / ((1 + x) * (1 + y) * (1 + z)) ≥ 2 / (1 - a^2)^3) ∧ 
  (∃ x y z, 1 / ((1 - x) * (1 - y) * (1 - z)) + 1 / ((1 + x) * (1 + y) * (1 + z)) = 2 / (1 - a^2)^3) :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_l3695_369543


namespace NUMINAMATH_CALUDE_stratified_sampling_correct_l3695_369566

/-- Represents a car model with its production volume -/
structure CarModel where
  name : String
  volume : Nat

/-- Calculates the number of cars to be sampled from a given model -/
def sampleSize (model : CarModel) (totalProduction : Nat) (totalSample : Nat) : Nat :=
  (model.volume * totalSample) / totalProduction

/-- Theorem stating that the stratified sampling produces the correct sample sizes -/
theorem stratified_sampling_correct 
  (emgrand kingKong freedomShip : CarModel)
  (h1 : emgrand.volume = 1600)
  (h2 : kingKong.volume = 6000)
  (h3 : freedomShip.volume = 2000)
  (h4 : emgrand.volume + kingKong.volume + freedomShip.volume = 9600)
  (h5 : 48 ≤ 9600) :
  let totalProduction := 9600
  let totalSample := 48
  (sampleSize emgrand totalProduction totalSample = 8) ∧
  (sampleSize kingKong totalProduction totalSample = 30) ∧
  (sampleSize freedomShip totalProduction totalSample = 10) := by
  sorry

end NUMINAMATH_CALUDE_stratified_sampling_correct_l3695_369566


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l3695_369589

-- Define the sets A and B
def A : Set ℝ := {x | -2 < x ∧ x < 1}
def B : Set ℝ := {x | x * (x - 3) > 0}

-- State the theorem
theorem intersection_of_A_and_B :
  A ∩ B = {x : ℝ | -2 < x ∧ x < 0} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l3695_369589


namespace NUMINAMATH_CALUDE_point_below_right_of_line_range_of_a_below_right_of_line_l3695_369588

/-- A point (a, 1) is below and to the right of the line x-2y+4=0 if and only if a > -2 -/
theorem point_below_right_of_line (a : ℝ) : 
  (a - 2 * 1 + 4 > 0) ↔ (a > -2) :=
sorry

/-- The range of a for points (a, 1) below and to the right of the line x-2y+4=0 is (-2, +∞) -/
theorem range_of_a_below_right_of_line : 
  {a : ℝ | a - 2 * 1 + 4 > 0} = Set.Ioi (-2) :=
sorry

end NUMINAMATH_CALUDE_point_below_right_of_line_range_of_a_below_right_of_line_l3695_369588


namespace NUMINAMATH_CALUDE_eight_couples_handshakes_l3695_369503

/-- The number of handshakes in a gathering of married couples --/
def handshakes (n : ℕ) : ℕ :=
  (n * (2 * n - 1)) / 2 - n

/-- Theorem: In a gathering of 8 married couples, the total number of handshakes is 112 --/
theorem eight_couples_handshakes :
  handshakes 8 = 112 := by
  sorry

end NUMINAMATH_CALUDE_eight_couples_handshakes_l3695_369503


namespace NUMINAMATH_CALUDE_lines_do_not_form_triangle_l3695_369517

/-- Three lines in a 2D plane -/
structure ThreeLines where
  line1 : ℝ → ℝ → Prop
  line2 : ℝ → ℝ → ℝ → Prop
  line3 : ℝ → ℝ → ℝ → Prop

/-- The given three lines -/
def givenLines (m : ℝ) : ThreeLines :=
  { line1 := λ x y => 4 * x + y = 4
  , line2 := λ x y m => m * x + y = 0
  , line3 := λ x y m => 2 * x - 3 * m * y = 4 }

/-- Predicate to check if three lines form a triangle -/
def formsTriangle (lines : ThreeLines) : Prop := sorry

/-- The set of m values for which the lines do not form a triangle -/
def noTriangleValues : Set ℝ := {4, -1/6, -1, 2/3}

/-- Theorem stating the condition for the lines to not form a triangle -/
theorem lines_do_not_form_triangle (m : ℝ) :
  ¬(formsTriangle (givenLines m)) ↔ m ∈ noTriangleValues :=
sorry

end NUMINAMATH_CALUDE_lines_do_not_form_triangle_l3695_369517


namespace NUMINAMATH_CALUDE_system_solution_l3695_369562

theorem system_solution (a b : ℝ) (h : a ≠ b) :
  ∃! (x y : ℝ), (a + 1) * x + (a - 1) * y = a ∧ (b + 1) * x + (b - 1) * y = b ∧ x = (1 : ℝ) / 2 ∧ y = (1 : ℝ) / 2 := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l3695_369562
