import Mathlib

namespace NUMINAMATH_CALUDE_square_edge_sum_l2205_220510

theorem square_edge_sum (u v w x : ℕ+) : 
  u * x + u * v + v * w + w * x = 15 → u + v + w + x = 8 :=
by
  sorry

end NUMINAMATH_CALUDE_square_edge_sum_l2205_220510


namespace NUMINAMATH_CALUDE_smallest_coconut_pile_l2205_220526

def is_valid (n : ℕ) : Prop :=
  ∃ k : ℕ, (4/5)^5 * (n - 5) + 1 = 5 * k

theorem smallest_coconut_pile : 
  (is_valid 3121) ∧ 
  (∀ m : ℕ, m < 3121 → ¬(is_valid m)) :=
sorry

end NUMINAMATH_CALUDE_smallest_coconut_pile_l2205_220526


namespace NUMINAMATH_CALUDE_scavenger_hunt_ratio_l2205_220539

theorem scavenger_hunt_ratio : 
  ∀ (lewis samantha tanya : ℕ),
  lewis = samantha + 4 →
  ∃ k : ℕ, samantha = k * tanya →
  tanya = 4 →
  lewis = 20 →
  samantha / tanya = 4 := by
sorry

end NUMINAMATH_CALUDE_scavenger_hunt_ratio_l2205_220539


namespace NUMINAMATH_CALUDE_nancy_history_marks_l2205_220512

def american_literature : ℕ := 66
def home_economics : ℕ := 52
def physical_education : ℕ := 68
def art : ℕ := 89
def average_marks : ℕ := 70
def total_subjects : ℕ := 5

theorem nancy_history_marks :
  ∃ history : ℕ,
    history = average_marks * total_subjects - (american_literature + home_economics + physical_education + art) ∧
    history = 75 := by
  sorry

end NUMINAMATH_CALUDE_nancy_history_marks_l2205_220512


namespace NUMINAMATH_CALUDE_smallest_angle_satisfies_equation_l2205_220576

/-- The smallest positive angle (in degrees) that satisfies the given equation -/
noncomputable def smallest_angle : ℝ :=
  (1 / 4) * Real.arcsin (2 / 9) * (180 / Real.pi)

/-- The equation that the angle must satisfy -/
def equation (x : ℝ) : Prop :=
  9 * Real.sin x * (Real.cos x)^7 - 9 * (Real.sin x)^7 * Real.cos x = 1

theorem smallest_angle_satisfies_equation :
  equation (smallest_angle * (Real.pi / 180)) ∧
  ∀ y, 0 < y ∧ y < smallest_angle → ¬equation (y * (Real.pi / 180)) :=
by sorry

end NUMINAMATH_CALUDE_smallest_angle_satisfies_equation_l2205_220576


namespace NUMINAMATH_CALUDE_peanut_butter_duration_l2205_220563

/-- The number of days that peanut butter lasts for Phoebe and her dog -/
def peanut_butter_days (servings_per_jar : ℕ) (num_jars : ℕ) (daily_consumption : ℕ) : ℕ :=
  (servings_per_jar * num_jars) / daily_consumption

/-- Theorem stating how long 4 jars of peanut butter will last for Phoebe and her dog -/
theorem peanut_butter_duration :
  let servings_per_jar : ℕ := 15
  let num_jars : ℕ := 4
  let phoebe_consumption : ℕ := 1
  let dog_consumption : ℕ := 1
  let daily_consumption : ℕ := phoebe_consumption + dog_consumption
  peanut_butter_days servings_per_jar num_jars daily_consumption = 30 := by
  sorry

end NUMINAMATH_CALUDE_peanut_butter_duration_l2205_220563


namespace NUMINAMATH_CALUDE_remainder_theorem_l2205_220529

-- Define the polynomial P(x) = x^100 - 2x^51 + 1
def P (x : ℝ) : ℝ := x^100 - 2*x^51 + 1

-- Define the divisor polynomial D(x) = x^2 - 1
def D (x : ℝ) : ℝ := x^2 - 1

-- Define the remainder polynomial R(x) = -2x + 2
def R (x : ℝ) : ℝ := -2*x + 2

-- Theorem statement
theorem remainder_theorem : 
  ∃ (Q : ℝ → ℝ), ∀ x, P x = Q x * D x + R x :=
sorry

end NUMINAMATH_CALUDE_remainder_theorem_l2205_220529


namespace NUMINAMATH_CALUDE_quadratic_function_properties_l2205_220525

-- Define the quadratic function f
def f (x : ℝ) : ℝ := 2 * x^2 - 4 * x + 3

-- Theorem statement
theorem quadratic_function_properties :
  (∀ x, f x ≥ 1) ∧  -- Minimum value is 1
  (f 0 = 3) ∧ (f 2 = 3) ∧  -- f(0) = f(2) = 3
  (∀ x, f x = 2 * x^2 - 4 * x + 3) ∧  -- Expression of f(x)
  (∀ a, (0 < a ∧ a < 1/3) ↔ 
    (∃ x y, 3*a ≤ x ∧ x < y ∧ y ≤ a+1 ∧ f x > f y ∧ 
    ∃ z, x < z ∧ z < y ∧ f z < f y)) ∧  -- Non-monotonic condition
  (∀ m, m < -1 ↔ 
    (∀ x, -1 ≤ x ∧ x ≤ 1 → f x > 2*x + 2*m + 1)) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_function_properties_l2205_220525


namespace NUMINAMATH_CALUDE_total_legs_puppies_and_chicks_l2205_220586

/-- The number of legs for puppies and chicks -/
def total_legs (num_puppies num_chicks : ℕ) (puppy_legs chick_legs : ℕ) : ℕ :=
  num_puppies * puppy_legs + num_chicks * chick_legs

/-- Theorem: Given 3 puppies and 7 chicks, where puppies have 4 legs each and chicks have 2 legs each, the total number of legs is 26. -/
theorem total_legs_puppies_and_chicks :
  total_legs 3 7 4 2 = 26 := by
  sorry

end NUMINAMATH_CALUDE_total_legs_puppies_and_chicks_l2205_220586


namespace NUMINAMATH_CALUDE_water_segment_length_l2205_220566

/-- Represents the problem of finding the length of a water segment in a journey --/
theorem water_segment_length 
  (total_distance : ℝ) 
  (find_probability : ℝ) 
  (h1 : total_distance = 2500) 
  (h2 : find_probability = 7/10) : 
  ∃ water_length : ℝ, 
    water_length / total_distance = 1 - find_probability ∧ 
    water_length = 750 := by
  sorry

end NUMINAMATH_CALUDE_water_segment_length_l2205_220566


namespace NUMINAMATH_CALUDE_value_of_a_l2205_220579

-- Define the conversion rate between paise and rupees
def paise_per_rupee : ℚ := 100

-- Define the given percentage as a rational number
def given_percentage : ℚ := 1 / 200

-- Define the given value in paise
def given_paise : ℚ := 85

-- Theorem statement
theorem value_of_a (a : ℚ) : given_percentage * a = given_paise → a = 170 := by
  sorry

end NUMINAMATH_CALUDE_value_of_a_l2205_220579


namespace NUMINAMATH_CALUDE_f_properties_l2205_220565

def f (x : ℕ) : ℕ := x % 2

def g (x : ℕ) : ℕ := x % 3

theorem f_properties :
  (∀ x : ℕ, f (2 * x) = 0) ∧
  (∀ x : ℕ, f x + f (x + 3) = 1) := by
  sorry

end NUMINAMATH_CALUDE_f_properties_l2205_220565


namespace NUMINAMATH_CALUDE_smallest_divisor_after_437_l2205_220562

theorem smallest_divisor_after_437 (m : ℕ) (h1 : 10000 ≤ m ∧ m ≤ 99999) 
  (h2 : Odd m) (h3 : m % 437 = 0) : 
  (∃ (d : ℕ), d > 437 ∧ d ∣ m ∧ ∀ (x : ℕ), 437 < x ∧ x < d → ¬(x ∣ m)) → 
  (Nat.minFac (m / 437) = 19 ∨ Nat.minFac (m / 437) = 23) :=
sorry

end NUMINAMATH_CALUDE_smallest_divisor_after_437_l2205_220562


namespace NUMINAMATH_CALUDE_f_range_l2205_220515

-- Define the function
def f (x : ℝ) : ℝ := |x + 1| + |x - 1|

-- State the theorem
theorem f_range : 
  Set.range f = { y | y ≥ 2 } := by sorry

end NUMINAMATH_CALUDE_f_range_l2205_220515


namespace NUMINAMATH_CALUDE_fourth_roll_max_probability_l2205_220518

-- Define the dice
structure Die :=
  (sides : ℕ)
  (max_prob : ℚ)
  (other_prob : ℚ)

-- Define the three dice
def six_sided_die : Die := ⟨6, 1/6, 1/6⟩
def eight_sided_die : Die := ⟨8, 3/4, 1/28⟩
def ten_sided_die : Die := ⟨10, 4/5, 1/45⟩

-- Define the probability of choosing each die
def choose_prob : ℚ := 1/3

-- Define the event of rolling maximum value three times for a given die
def max_three_times (d : Die) : ℚ := d.max_prob^3

-- Define the total probability of rolling maximum value three times
def total_max_three_times : ℚ :=
  choose_prob * (max_three_times six_sided_die + 
                 max_three_times eight_sided_die + 
                 max_three_times ten_sided_die)

-- Define the conditional probability of using each die given three max rolls
def cond_prob (d : Die) : ℚ :=
  (choose_prob * max_three_times d) / total_max_three_times

-- Define the probability of fourth roll being max given three max rolls
def fourth_max_prob : ℚ :=
  cond_prob six_sided_die * six_sided_die.max_prob +
  cond_prob eight_sided_die * eight_sided_die.max_prob +
  cond_prob ten_sided_die * ten_sided_die.max_prob

-- The theorem to prove
theorem fourth_roll_max_probability : 
  fourth_max_prob = 1443 / 2943 := by
  sorry

end NUMINAMATH_CALUDE_fourth_roll_max_probability_l2205_220518


namespace NUMINAMATH_CALUDE_work_completion_time_l2205_220595

theorem work_completion_time (a b c : ℝ) : 
  a = 24 →  -- a completes the work in 24 days
  c = 12 →  -- c completes the work in 12 days
  1 / a + 1 / b + 1 / c = 7 / 24 →  -- combined work rate equals 7/24 (equivalent to completing in 24/7 days)
  b = 6 :=  -- b completes the work in 6 days
by sorry

end NUMINAMATH_CALUDE_work_completion_time_l2205_220595


namespace NUMINAMATH_CALUDE_function_c_injective_l2205_220593

theorem function_c_injective (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  a + 2 / a = b + 2 / b → a = b := by sorry

end NUMINAMATH_CALUDE_function_c_injective_l2205_220593


namespace NUMINAMATH_CALUDE_permutations_combinations_theorem_l2205_220549

/-- The number of ways to choose k items from n different items when order matters -/
def permutations (n k : ℕ) : ℕ := Nat.factorial n / Nat.factorial (n - k)

/-- The number of ways to choose k items from n different items when order doesn't matter -/
def combinations (n k : ℕ) : ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

theorem permutations_combinations_theorem (n k : ℕ) (h : k ≤ n) :
  permutations n k = Nat.factorial n / Nat.factorial (n - k) ∧
  combinations n k = Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k)) := by
  sorry

end NUMINAMATH_CALUDE_permutations_combinations_theorem_l2205_220549


namespace NUMINAMATH_CALUDE_rationality_of_square_roots_l2205_220507

theorem rationality_of_square_roots (x y z : ℚ) (w : ℚ) 
  (hw : w = Real.sqrt x + Real.sqrt y + Real.sqrt z) :
  ∃ (a b c : ℚ), Real.sqrt x = a ∧ Real.sqrt y = b ∧ Real.sqrt z = c :=
sorry

end NUMINAMATH_CALUDE_rationality_of_square_roots_l2205_220507


namespace NUMINAMATH_CALUDE_cuboid_length_calculation_l2205_220520

/-- The surface area of a cuboid given its length, breadth, and height -/
def cuboid_surface_area (l b h : ℝ) : ℝ := 2 * (l * b + b * h + h * l)

/-- Theorem: A cuboid with surface area 720, breadth 6, and height 10 has length 18.75 -/
theorem cuboid_length_calculation (l : ℝ) :
  cuboid_surface_area l 6 10 = 720 → l = 18.75 := by
  sorry

end NUMINAMATH_CALUDE_cuboid_length_calculation_l2205_220520


namespace NUMINAMATH_CALUDE_angle_D_value_l2205_220560

-- Define the angles
def A : ℝ := 30
def B (D : ℝ) : ℝ := 2 * D
def C (D : ℝ) : ℝ := D + 40

-- Theorem statement
theorem angle_D_value :
  ∀ D : ℝ, A + B D + C D + D = 360 → D = 72.5 := by sorry

end NUMINAMATH_CALUDE_angle_D_value_l2205_220560


namespace NUMINAMATH_CALUDE_triangle_abc_proof_l2205_220540

theorem triangle_abc_proof (A B C : ℝ) (a b c : ℝ) :
  0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = π →
  a > 0 ∧ b > 0 ∧ c > 0 →
  a * Real.cos A * Real.cos B - b * Real.sin A * Real.sin A - c * Real.cos A = 2 * b * Real.cos B →
  b = Real.sqrt 7 * a →
  (1/2) * a * c * Real.sin B = 2 * Real.sqrt 3 →
  B = 2 * π / 3 ∧ a = 2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_abc_proof_l2205_220540


namespace NUMINAMATH_CALUDE_divisor_sum_condition_l2205_220578

theorem divisor_sum_condition (n : ℕ+) :
  (∃ (a b c : ℕ+), a + b + c = n ∧ a ∣ b ∧ b ∣ c ∧ a < b ∧ b < c) ↔ 
  n ∉ ({1, 2, 3, 4, 5, 6, 8, 12, 24} : Set ℕ+) :=
by sorry

end NUMINAMATH_CALUDE_divisor_sum_condition_l2205_220578


namespace NUMINAMATH_CALUDE_min_value_theorem_l2205_220584

theorem min_value_theorem (a b : ℝ) (ha : a > 1) (hb : b > 0) (hab : a + b = 2) :
  (∀ x y : ℝ, x > 1 ∧ y > 0 ∧ x + y = 2 → 1 / (x - 1) + 2 / y ≥ 1 / (a - 1) + 2 / b) ∧
  1 / (a - 1) + 2 / b = 3 + 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_min_value_theorem_l2205_220584


namespace NUMINAMATH_CALUDE_not_value_preserving_g_value_preserving_f_condition_l2205_220555

/-- Definition of a value-preserving function on an interval [m, n] -/
def is_value_preserving (f : ℝ → ℝ) (m n : ℝ) : Prop :=
  m < n ∧ 
  Monotone (fun x => f x) ∧
  Set.range (fun x => f x) = Set.Icc m n

/-- The function g(x) = x^2 - 2x -/
def g (x : ℝ) : ℝ := x^2 - 2*x

/-- The function f(x) = 2 + 1/a - 1/(a^2x) -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := 2 + 1/a - 1/(a^2*x)

theorem not_value_preserving_g : ¬ is_value_preserving g 0 1 := by sorry

theorem value_preserving_f_condition (a : ℝ) :
  (∃ m n, is_value_preserving (f a) m n) ↔ (a > 1/2 ∨ a < -3/2) := by sorry

end NUMINAMATH_CALUDE_not_value_preserving_g_value_preserving_f_condition_l2205_220555


namespace NUMINAMATH_CALUDE_distance_k_is_14_l2205_220550

/-- Given 5 points on a straight line, this function returns the list of distances between every pair of points in increasing order -/
def distances_between_points (a b c d : ℝ) : List ℝ :=
  [a, b, c, d, a + b, a + c, a + d, b + c, b + d, c + d, a + b + c, a + b + d, a + c + d, b + c + d, a + b + c + d]

/-- Theorem stating that given 5 points on a straight line with specific distances, k must be 14 -/
theorem distance_k_is_14 (a b c d : ℝ) :
  distances_between_points a b c d = [2, 5, 6, 8, 9, 14, 15, 17, 20, 22] →
  ∃ (p : List ℝ), p.Perm (distances_between_points a b c d) ∧ p = [2, 5, 6, 8, 9, 14, 15, 17, 20, 22] := by
  sorry

#check distance_k_is_14

end NUMINAMATH_CALUDE_distance_k_is_14_l2205_220550


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l2205_220594

theorem sufficient_not_necessary_condition : 
  (∃ x : ℝ, x^2 - 1 > 0 ∧ ¬(x < -1)) ∧ 
  (∀ x : ℝ, x < -1 → x^2 - 1 > 0) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l2205_220594


namespace NUMINAMATH_CALUDE_socks_cost_theorem_l2205_220504

/-- The cost of each red pair of socks -/
def red_cost : ℝ := 3

/-- The number of red sock pairs -/
def red_pairs : ℕ := 4

/-- The number of blue sock pairs -/
def blue_pairs : ℕ := 6

/-- The cost of each blue pair of socks -/
def blue_cost : ℝ := 5

/-- The total cost of all socks -/
def total_cost : ℝ := 42

theorem socks_cost_theorem :
  red_cost * red_pairs + blue_cost * blue_pairs = total_cost :=
by sorry

end NUMINAMATH_CALUDE_socks_cost_theorem_l2205_220504


namespace NUMINAMATH_CALUDE_isosceles_triangle_base_length_l2205_220541

/-- 
An isosceles triangle with congruent sides of length 7 cm and perimeter 23 cm 
has a base of length 9 cm.
-/
theorem isosceles_triangle_base_length : 
  ∀ (base : ℝ), 
  base > 0 → 
  7 + 7 + base = 23 → 
  base = 9 :=
by
  sorry

end NUMINAMATH_CALUDE_isosceles_triangle_base_length_l2205_220541


namespace NUMINAMATH_CALUDE_parabola_vertex_y_coordinate_l2205_220543

/-- Given a quadratic function f(x) = 5x^2 + 20x + 45, 
    prove that the y-coordinate of its vertex is 25. -/
theorem parabola_vertex_y_coordinate :
  let f : ℝ → ℝ := λ x ↦ 5 * x^2 + 20 * x + 45
  ∃ h k : ℝ, (∀ x, f x = 5 * (x - h)^2 + k) ∧ k = 25 :=
by sorry

end NUMINAMATH_CALUDE_parabola_vertex_y_coordinate_l2205_220543


namespace NUMINAMATH_CALUDE_inequality_solution_l2205_220500

theorem inequality_solution (x : ℝ) (h1 : x ≠ 2) (h2 : x ≠ 0) :
  (x + 1) / (x - 2) + (x - 3) / (3 * x) ≥ 2 ↔ x ≤ -1/3 ∨ x ≥ 3 := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_l2205_220500


namespace NUMINAMATH_CALUDE_rocket_soaring_time_l2205_220569

/-- Proves that the soaring time of a rocket is 12 seconds given specific conditions -/
theorem rocket_soaring_time :
  let soaring_speed : ℝ := 150
  let plummet_distance : ℝ := 600
  let plummet_time : ℝ := 3
  let average_speed : ℝ := 160
  let soaring_time : ℝ := 12

  (soaring_speed * soaring_time + plummet_distance) / (soaring_time + plummet_time) = average_speed :=
by
  sorry


end NUMINAMATH_CALUDE_rocket_soaring_time_l2205_220569


namespace NUMINAMATH_CALUDE_probability_identical_cubes_value_l2205_220547

/-- Represents a cube with 8 faces, each face can be painted with one of three colors -/
structure Cube :=
  (faces : Fin 8 → Fin 3)

/-- The total number of ways to paint two cubes -/
def total_paintings : ℕ := 3^8 * 3^8

/-- The number of ways to paint two cubes so they look identical after rotation -/
def identical_paintings : ℕ := 831

/-- The probability that two cubes look identical after painting and possible rotations -/
def probability_identical_cubes : ℚ :=
  identical_paintings / total_paintings

theorem probability_identical_cubes_value :
  probability_identical_cubes = 831 / 43046721 :=
sorry

end NUMINAMATH_CALUDE_probability_identical_cubes_value_l2205_220547


namespace NUMINAMATH_CALUDE_angle_equality_l2205_220524

-- Define the points
variable (A B C D K : Euclidean_plane)

-- Define the quadrilateral ABCD
def is_convex_quadrilateral (A B C D : Euclidean_plane) : Prop := sorry

-- Define the equality of line segments
def seg_eq (P Q R S : Euclidean_plane) : Prop := sorry

-- Define the equality of angles
def angle_eq (P Q R S T U : Euclidean_plane) : Prop := sorry

-- Define a point being on a line segment
def point_on_seg (P Q R : Euclidean_plane) : Prop := sorry

-- Theorem statement
theorem angle_equality 
  (h1 : is_convex_quadrilateral A B C D)
  (h2 : seg_eq A B B D)
  (h3 : angle_eq A B D D B C)
  (h4 : point_on_seg K B D)
  (h5 : seg_eq B K B C) :
  angle_eq K A D K C D :=
sorry

end NUMINAMATH_CALUDE_angle_equality_l2205_220524


namespace NUMINAMATH_CALUDE_sock_combinations_proof_l2205_220538

/-- The number of ways to choose 4 socks out of 7, with at least one red sock -/
def sockCombinations : ℕ := 20

/-- The total number of socks -/
def totalSocks : ℕ := 7

/-- The number of socks to be chosen -/
def chosenSocks : ℕ := 4

/-- The number of non-red socks -/
def nonRedSocks : ℕ := 6

theorem sock_combinations_proof :
  sockCombinations = Nat.choose totalSocks chosenSocks - Nat.choose nonRedSocks chosenSocks :=
by sorry

end NUMINAMATH_CALUDE_sock_combinations_proof_l2205_220538


namespace NUMINAMATH_CALUDE_min_value_sqrt_plus_reciprocal_l2205_220587

theorem min_value_sqrt_plus_reciprocal (x : ℝ) (hx : x > 0) : 
  2 * Real.sqrt x + 1 / x ≥ 3 ∧ 
  (2 * Real.sqrt x + 1 / x = 3 ↔ x = 1) := by
sorry

end NUMINAMATH_CALUDE_min_value_sqrt_plus_reciprocal_l2205_220587


namespace NUMINAMATH_CALUDE_arithmetic_sequence_20th_term_l2205_220577

theorem arithmetic_sequence_20th_term (a : ℕ → ℤ) 
  (h_arithmetic : ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1))
  (h_sum_odd : a 1 + a 3 + a 5 = 105)
  (h_sum_even : a 2 + a 4 + a 6 = 99) :
  a 20 = 1 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_20th_term_l2205_220577


namespace NUMINAMATH_CALUDE_total_players_on_ground_l2205_220572

theorem total_players_on_ground (cricket hockey football softball : ℕ) : 
  cricket = 15 → hockey = 12 → football = 13 → softball = 15 →
  cricket + hockey + football + softball = 55 := by
  sorry

end NUMINAMATH_CALUDE_total_players_on_ground_l2205_220572


namespace NUMINAMATH_CALUDE_sequence_kth_term_l2205_220591

theorem sequence_kth_term (a : ℕ → ℝ) (S : ℕ → ℝ) (k : ℕ) :
  (∀ n, S n = n^2 - 9*n) →
  (∀ n ≥ 2, a n = S n - S (n-1)) →
  (5 < a k ∧ a k < 8) →
  k = 8 := by sorry

end NUMINAMATH_CALUDE_sequence_kth_term_l2205_220591


namespace NUMINAMATH_CALUDE_eight_friends_lineup_l2205_220582

theorem eight_friends_lineup (n : ℕ) (h : n = 8) : Nat.factorial n = 40320 := by
  sorry

end NUMINAMATH_CALUDE_eight_friends_lineup_l2205_220582


namespace NUMINAMATH_CALUDE_smallest_angle_is_90_l2205_220554

/-- Represents a trapezoid with angles in arithmetic sequence -/
structure ArithmeticTrapezoid where
  -- The smallest angle
  a : ℝ
  -- The common difference in the arithmetic sequence
  d : ℝ
  -- Assertion that angles are in arithmetic sequence
  angle_sequence : List ℝ := [a, a + d, a + 2*d, a + 3*d]
  -- Assertion that the sum of any two consecutive angles is 180°
  consecutive_sum : a + (a + d) = 180 ∧ (a + d) + (a + 2*d) = 180 ∧ (a + 2*d) + (a + 3*d) = 180
  -- Assertion that the second largest angle is 150°
  second_largest : a + 2*d = 150

/-- 
Theorem: In a trapezoid where the angles form an arithmetic sequence 
and the second largest angle is 150°, the smallest angle measures 90°.
-/
theorem smallest_angle_is_90 (t : ArithmeticTrapezoid) : t.a = 90 := by
  sorry

end NUMINAMATH_CALUDE_smallest_angle_is_90_l2205_220554


namespace NUMINAMATH_CALUDE_true_discount_is_36_l2205_220514

/-- Given a banker's discount and sum due, calculate the true discount -/
def true_discount (BD : ℚ) (SD : ℚ) : ℚ :=
  BD / (1 + BD / SD)

/-- Theorem stating that for the given banker's discount and sum due, the true discount is 36 -/
theorem true_discount_is_36 :
  true_discount 42 252 = 36 := by
  sorry

end NUMINAMATH_CALUDE_true_discount_is_36_l2205_220514


namespace NUMINAMATH_CALUDE_work_rate_problem_l2205_220511

/-- Proves that given the work rates of A and B, and the combined work rate of A, B, and C,
    we can determine how long it takes C to do the work alone. -/
theorem work_rate_problem (a b c : ℝ) (h1 : a = 6) (h2 : b = 5) 
  (h3 : 1 / a + 1 / b + 1 / c = 1 / (10 / 9)) : c = 15 / 8 := by
  sorry

#eval (15 : ℚ) / 8  -- To show that 15/8 = 1.875

end NUMINAMATH_CALUDE_work_rate_problem_l2205_220511


namespace NUMINAMATH_CALUDE_calgary_red_deer_distance_calgary_red_deer_distance_value_l2205_220506

/-- The distance between Edmonton and Red Deer in kilometers -/
def edmonton_red_deer_distance : ℝ := 220

/-- The speed of travel in kilometers per hour -/
def travel_speed : ℝ := 110

/-- The time taken to travel from Edmonton to Calgary in hours -/
def edmonton_calgary_time : ℝ := 3

/-- The distance between Edmonton and Calgary in kilometers -/
def edmonton_calgary_distance : ℝ := travel_speed * edmonton_calgary_time

/-- Calgary is south of Red Deer -/
axiom calgary_south_of_red_deer : True

theorem calgary_red_deer_distance : ℝ := by
  sorry

theorem calgary_red_deer_distance_value : calgary_red_deer_distance = 110 := by
  sorry

end NUMINAMATH_CALUDE_calgary_red_deer_distance_calgary_red_deer_distance_value_l2205_220506


namespace NUMINAMATH_CALUDE_complement_intersection_theorem_l2205_220585

def U : Set Int := {-2, -1, 0, 1, 2, 3}
def A : Set Int := {-1, 0, 1}
def B : Set Int := {1, 2, 3}

theorem complement_intersection_theorem :
  (U \ (A ∩ B)) = {-2, -1, 0, 2, 3} := by sorry

end NUMINAMATH_CALUDE_complement_intersection_theorem_l2205_220585


namespace NUMINAMATH_CALUDE_quadrilateral_diagonal_length_l2205_220502

/-- Represents a quadrilateral with diagonals intersecting at a point -/
structure Quadrilateral :=
  (A B C D O : ℝ × ℝ)

/-- The distance between two points -/
def distance (p q : ℝ × ℝ) : ℝ := sorry

theorem quadrilateral_diagonal_length 
  (ABCD : Quadrilateral) 
  (h1 : distance ABCD.B ABCD.O = 3)
  (h2 : distance ABCD.O ABCD.D = 9)
  (h3 : distance ABCD.A ABCD.O = 5)
  (h4 : distance ABCD.O ABCD.C = 2)
  (h5 : distance ABCD.A ABCD.B = 7) :
  distance ABCD.A ABCD.D = Real.sqrt 151 := by
  sorry

end NUMINAMATH_CALUDE_quadrilateral_diagonal_length_l2205_220502


namespace NUMINAMATH_CALUDE_right_angled_triangle_l2205_220545

theorem right_angled_triangle (A B C : ℝ) (h : A + B + C = π) 
  (eq : (Real.cos A) / 20 + (Real.cos B) / 21 + (Real.cos C) / 29 = 29 / 420) : 
  C = π / 2 := by
  sorry

end NUMINAMATH_CALUDE_right_angled_triangle_l2205_220545


namespace NUMINAMATH_CALUDE_decreasing_function_implies_a_nonnegative_l2205_220513

-- Define the function f
def f (x a b : ℝ) : ℝ := x^2 + |x - a| + b

-- State the theorem
theorem decreasing_function_implies_a_nonnegative 
  (h : ∀ x₁ x₂ : ℝ, x₁ < x₂ → x₂ ≤ 0 → f x₂ a b ≤ f x₁ a b) : 
  a ≥ 0 := by
  sorry

end NUMINAMATH_CALUDE_decreasing_function_implies_a_nonnegative_l2205_220513


namespace NUMINAMATH_CALUDE_curve_symmetric_about_y_eq_neg_x_l2205_220556

/-- A curve in the xy-plane -/
structure Curve where
  equation : ℝ → ℝ → Prop

/-- Symmetry about a line -/
def symmetric_about (c : Curve) (f : ℝ → ℝ) : Prop :=
  ∀ x y, c.equation x y ↔ c.equation (f y) (f x)

/-- The curve represented by xy^2 - x^2y = -2 -/
def curve : Curve :=
  { equation := λ x y ↦ x * y^2 - x^2 * y = -2 }

/-- The line y = -x -/
def line_y_eq_neg_x : ℝ → ℝ :=
  λ x ↦ -x

theorem curve_symmetric_about_y_eq_neg_x :
  symmetric_about curve line_y_eq_neg_x := by
  sorry

end NUMINAMATH_CALUDE_curve_symmetric_about_y_eq_neg_x_l2205_220556


namespace NUMINAMATH_CALUDE_simplify_expression_1_simplify_expression_2_l2205_220522

-- Problem 1
theorem simplify_expression_1 (a b : ℝ) :
  (2 * a - b) - (2 * b - 3 * a) - 2 * (a - 2 * b) = 3 * a + b := by sorry

-- Problem 2
theorem simplify_expression_2 (x y : ℝ) :
  (4 * x^2 - 5 * x * y) - (1/3 * y^2 + 2 * x^2) + 2 * (3 * x * y - 1/4 * y^2 - 1/12 * y^2) = 
  2 * x^2 + x * y - y^2 := by sorry

end NUMINAMATH_CALUDE_simplify_expression_1_simplify_expression_2_l2205_220522


namespace NUMINAMATH_CALUDE_sum_equals_product_integer_pairs_l2205_220546

theorem sum_equals_product_integer_pairs :
  ∀ x y : ℤ, x + y = x * y ↔ (x = 2 ∧ y = 2) ∨ (x = 0 ∧ y = 0) := by
sorry

end NUMINAMATH_CALUDE_sum_equals_product_integer_pairs_l2205_220546


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l2205_220574

/-- The eccentricity of a hyperbola with equation 16x^2 - 9y^2 = 144 is 5/3 -/
theorem hyperbola_eccentricity : 
  let a : ℝ := 3
  let b : ℝ := 4
  let c : ℝ := (a^2 + b^2).sqrt
  let e : ℝ := c / a
  16 * x^2 - 9 * y^2 = 144 → e = 5/3 :=
by
  sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l2205_220574


namespace NUMINAMATH_CALUDE_circle_center_on_line_ab_range_l2205_220521

theorem circle_center_on_line_ab_range :
  ∀ (a b : ℝ),
  (∃ (x y : ℝ), x^2 + y^2 + 2*x - 4*y + 1 = 0 ∧ a*x - b*y + 1 = 0) →
  a*b ≤ 1/8 ∧ ∀ ε > 0, ∃ (a' b' : ℝ), a'*b' < -ε :=
by sorry

end NUMINAMATH_CALUDE_circle_center_on_line_ab_range_l2205_220521


namespace NUMINAMATH_CALUDE_remainder_thirteen_power_thirteen_plus_thirteen_l2205_220517

theorem remainder_thirteen_power_thirteen_plus_thirteen (n : ℕ) :
  (13^13 + 13) % 14 = 12 := by
  sorry

end NUMINAMATH_CALUDE_remainder_thirteen_power_thirteen_plus_thirteen_l2205_220517


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l2205_220551

theorem quadratic_equation_solution : 
  ∃ (x₁ x₂ : ℝ), x₁ = (1 : ℝ) / 2 ∧ x₂ = 1 ∧ 
  (2 * x₁^2 - 3 * x₁ + 1 = 0) ∧ (2 * x₂^2 - 3 * x₂ + 1 = 0) :=
by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l2205_220551


namespace NUMINAMATH_CALUDE_f_composition_l2205_220588

-- Define the function f
def f (x : ℝ) : ℝ := 3 * x + 5

-- State the theorem
theorem f_composition (x : ℝ) : f (3 * x - 7) = 9 * x - 16 := by
  sorry

end NUMINAMATH_CALUDE_f_composition_l2205_220588


namespace NUMINAMATH_CALUDE_largest_divisor_of_m_l2205_220557

theorem largest_divisor_of_m (m : ℕ) (hm : m > 0) (hdiv : 847 ∣ m^3) : 
  77 = Nat.gcd m 77 ∧ ∀ k : ℕ, k > 77 → k ∣ m → k ∣ 847 := by
  sorry

end NUMINAMATH_CALUDE_largest_divisor_of_m_l2205_220557


namespace NUMINAMATH_CALUDE_pascal_triangle_30_rows_l2205_220592

/-- The number of elements in the nth row of Pascal's Triangle -/
def pascal_row_elements (n : ℕ) : ℕ := n + 1

/-- The sum of the first n natural numbers -/
def sum_first_n (n : ℕ) : ℕ := n * (n + 1) / 2

/-- The number of elements in the first 30 rows of Pascal's Triangle -/
theorem pascal_triangle_30_rows : sum_first_n 30 = 465 := by sorry

end NUMINAMATH_CALUDE_pascal_triangle_30_rows_l2205_220592


namespace NUMINAMATH_CALUDE_cos_330_degrees_l2205_220568

theorem cos_330_degrees : Real.cos (330 * π / 180) = Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_330_degrees_l2205_220568


namespace NUMINAMATH_CALUDE_intersection_M_complement_N_l2205_220599

-- Define set M
def M : Set ℝ := {x | x^2 - 2*x - 3 < 0}

-- Define set N
def N : Set ℝ := {x | 2*x < 2}

-- Theorem statement
theorem intersection_M_complement_N :
  ∀ x : ℝ, x ∈ (M ∩ (Set.univ \ N)) ↔ 1 ≤ x ∧ x < 3 := by sorry

end NUMINAMATH_CALUDE_intersection_M_complement_N_l2205_220599


namespace NUMINAMATH_CALUDE_ab_equals_six_l2205_220583

theorem ab_equals_six (a b : ℝ) (h : a / 2 = 3 / b) : a * b = 6 := by
  sorry

end NUMINAMATH_CALUDE_ab_equals_six_l2205_220583


namespace NUMINAMATH_CALUDE_nba_division_impossibility_l2205_220590

theorem nba_division_impossibility : ∀ (A B : ℕ),
  A + B = 30 →
  A * B ≠ (30 * 82) / 4 :=
by
  sorry


end NUMINAMATH_CALUDE_nba_division_impossibility_l2205_220590


namespace NUMINAMATH_CALUDE_line_properties_l2205_220597

/-- The line equation: (a+1)x + y + 2-a = 0 -/
def line_equation (a x y : ℝ) : Prop := (a + 1) * x + y + 2 - a = 0

/-- Equal intercepts on both coordinate axes -/
def equal_intercepts (a : ℝ) : Prop :=
  ∃ t : ℝ, t ≠ 0 ∧ line_equation a t 0 ∧ line_equation a 0 t

/-- Line does not pass through the second quadrant -/
def not_in_second_quadrant (a : ℝ) : Prop :=
  ∀ x y : ℝ, line_equation a x y → ¬(x < 0 ∧ y > 0)

/-- Main theorem -/
theorem line_properties :
  (∀ a : ℝ, equal_intercepts a ↔ (a = 0 ∨ a = 2)) ∧
  (∀ a : ℝ, not_in_second_quadrant a ↔ a ≤ -1) := by sorry

end NUMINAMATH_CALUDE_line_properties_l2205_220597


namespace NUMINAMATH_CALUDE_john_non_rent_expenses_l2205_220553

/-- Represents John's computer business finances --/
structure ComputerBusiness where
  parts_cost : ℝ
  selling_price_multiplier : ℝ
  computers_per_month : ℕ
  monthly_rent : ℝ
  monthly_profit : ℝ

/-- Calculates the non-rent extra expenses for John's computer business --/
def non_rent_extra_expenses (business : ComputerBusiness) : ℝ :=
  let selling_price := business.parts_cost * business.selling_price_multiplier
  let total_revenue := selling_price * business.computers_per_month
  let total_cost_components := business.parts_cost * business.computers_per_month
  let total_expenses := total_revenue - business.monthly_profit
  total_expenses - business.monthly_rent - total_cost_components

/-- Theorem stating that John's non-rent extra expenses are $3000 per month --/
theorem john_non_rent_expenses :
  let john_business : ComputerBusiness := {
    parts_cost := 800,
    selling_price_multiplier := 1.4,
    computers_per_month := 60,
    monthly_rent := 5000,
    monthly_profit := 11200
  }
  non_rent_extra_expenses john_business = 3000 := by
  sorry

end NUMINAMATH_CALUDE_john_non_rent_expenses_l2205_220553


namespace NUMINAMATH_CALUDE_circle_tangent_condition_l2205_220581

/-- A circle in the xy-plane -/
structure Circle where
  D : ℝ
  E : ℝ
  F : ℝ

/-- Predicate for a circle being tangent to the x-axis at the origin -/
def is_tangent_at_origin (c : Circle) : Prop :=
  ∀ (x y : ℝ), x^2 + y^2 + c.D * x + c.E * y + c.F = 0 → 
    (x = 0 ∧ y = 0) ∨ (x ≠ 0 ∧ y ≠ 0)

theorem circle_tangent_condition (c : Circle) :
  is_tangent_at_origin c → c.E ≠ 0 ∧ c.D = 0 ∧ c.F = 0 := by
  sorry

end NUMINAMATH_CALUDE_circle_tangent_condition_l2205_220581


namespace NUMINAMATH_CALUDE_liar_count_theorem_l2205_220544

/-- Represents the type of islander: knight or liar -/
inductive IslanderType
  | Knight
  | Liar

/-- Represents a group of islanders making a statement -/
structure IslanderGroup where
  size : Nat
  statement : Nat

/-- The problem setup -/
def islanderProblem : List IslanderGroup :=
  [⟨2, 2⟩, ⟨4, 4⟩, ⟨8, 8⟩, ⟨14, 14⟩]

/-- The total number of islanders -/
def totalIslanders : Nat := 28

/-- Function to determine if a statement is true given the actual number of liars -/
def isStatementTrue (group : IslanderGroup) (actualLiars : Nat) : Bool :=
  group.statement == actualLiars

/-- Function to determine the type of an islander based on their statement and the actual number of liars -/
def determineType (group : IslanderGroup) (actualLiars : Nat) : IslanderType :=
  if isStatementTrue group actualLiars then IslanderType.Knight else IslanderType.Liar

/-- Theorem stating that the number of liars is either 14 or 28 -/
theorem liar_count_theorem :
  ∃ (liarCount : Nat), (liarCount = 14 ∨ liarCount = 28) ∧
  (∀ (group : IslanderGroup), group ∈ islanderProblem →
    (determineType group liarCount = IslanderType.Liar) = (group.size ≤ liarCount)) ∧
  (liarCount ≤ totalIslanders) := by
  sorry

end NUMINAMATH_CALUDE_liar_count_theorem_l2205_220544


namespace NUMINAMATH_CALUDE_monotonicity_and_range_l2205_220533

noncomputable def f (a b x : ℝ) : ℝ := 2 * a * x + b * x - 1 - 2 * Real.log x

theorem monotonicity_and_range :
  (∀ a ≤ 0, ∀ x > 0, (deriv (f a 0)) x < 0) ∧
  (∀ a > 0, ∀ x ∈ Set.Ioo 0 (1/a), (deriv (f a 0)) x < 0) ∧
  (∀ a > 0, ∀ x ∈ Set.Ioi (1/a), (deriv (f a 0)) x > 0) ∧
  (∀ x > 0, f 1 b x ≥ 2 * b * x - 3 → b ≤ 2 - 2 / Real.exp 2) :=
by sorry

end NUMINAMATH_CALUDE_monotonicity_and_range_l2205_220533


namespace NUMINAMATH_CALUDE_max_value_of_a_l2205_220573

theorem max_value_of_a (a b c : ℝ) (sum_zero : a + b + c = 0) (sum_squares_six : a^2 + b^2 + c^2 = 6) :
  ∀ x : ℝ, (∃ y z : ℝ, x + y + z = 0 ∧ x^2 + y^2 + z^2 = 6) → x ≤ 2 :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_a_l2205_220573


namespace NUMINAMATH_CALUDE_garage_sale_items_l2205_220570

theorem garage_sale_items (prices : Finset ℕ) (radio_price : ℕ) : 
  prices.card = 43 ∧ radio_price ∈ prices ∧ 
  (prices.filter (λ x => x > radio_price)).card = 8 ∧
  (prices.filter (λ x => x < radio_price)).card = 34 →
  prices.card = 43 :=
by sorry

end NUMINAMATH_CALUDE_garage_sale_items_l2205_220570


namespace NUMINAMATH_CALUDE_binary_110011_is_51_l2205_220571

def binary_to_decimal (b : List Bool) : Nat :=
  b.enum.foldl (fun acc (i, bit) => acc + if bit then 2^i else 0) 0

theorem binary_110011_is_51 :
  binary_to_decimal [true, true, false, false, true, true] = 51 := by sorry

end NUMINAMATH_CALUDE_binary_110011_is_51_l2205_220571


namespace NUMINAMATH_CALUDE_exam_grading_rules_l2205_220505

-- Define the types
def Student : Type := String
def Grade : Type := String
def Essay : Type := Bool

-- Define the predicates
def all_mc_correct (s : Student) : Prop := sorry
def satisfactory_essay (s : Student) : Prop := sorry
def grade_is (s : Student) (g : Grade) : Prop := sorry

-- State the theorem
theorem exam_grading_rules (s : Student) :
  -- Condition 1
  (∀ s, all_mc_correct s → grade_is s "B" ∨ grade_is s "A") →
  -- Condition 2
  (∀ s, all_mc_correct s ∧ satisfactory_essay s → grade_is s "A") →
  -- Statement D
  (grade_is s "A" → all_mc_correct s) ∧
  -- Statement E
  (all_mc_correct s ∧ satisfactory_essay s → grade_is s "A") := by
  sorry


end NUMINAMATH_CALUDE_exam_grading_rules_l2205_220505


namespace NUMINAMATH_CALUDE_reciprocal_of_abs_negative_two_l2205_220501

theorem reciprocal_of_abs_negative_two : (|-2|)⁻¹ = (1/2 : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_reciprocal_of_abs_negative_two_l2205_220501


namespace NUMINAMATH_CALUDE_intersection_count_l2205_220527

/-- Represents a line in a 2D plane --/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The five lines given in the problem --/
def line1 : Line := { a := 3, b := -2, c := 9 }
def line2 : Line := { a := 6, b := 4, c := -12 }
def line3 : Line := { a := 1, b := 0, c := 3 }
def line4 : Line := { a := 0, b := 1, c := 1 }
def line5 : Line := { a := 2, b := 1, c := -1 }

/-- Determines if two lines intersect --/
def intersect (l1 l2 : Line) : Prop :=
  l1.a * l2.b ≠ l1.b * l2.a

/-- Counts the number of unique intersection points --/
def countIntersections (lines : List Line) : ℕ :=
  sorry

theorem intersection_count :
  countIntersections [line1, line2, line3, line4, line5] = 6 := by
  sorry

end NUMINAMATH_CALUDE_intersection_count_l2205_220527


namespace NUMINAMATH_CALUDE_quadratic_expression_value_l2205_220589

theorem quadratic_expression_value (x y : ℝ) 
  (h1 : 3 * x + y = 6) (h2 : x + 3 * y = 8) : 
  10 * x^2 + 13 * x * y + 10 * y^2 = 100 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_expression_value_l2205_220589


namespace NUMINAMATH_CALUDE_four_digit_numbers_count_l2205_220561

theorem four_digit_numbers_count : 
  (Finset.range 9000).card = (Finset.Icc 1000 9999).card := by sorry

end NUMINAMATH_CALUDE_four_digit_numbers_count_l2205_220561


namespace NUMINAMATH_CALUDE_book_cost_calculation_l2205_220523

theorem book_cost_calculation (initial_amount : ℕ) (books_bought : ℕ) (amount_left : ℕ) (cost_per_book : ℕ) : 
  initial_amount = 85 → 
  books_bought = 10 → 
  amount_left = 35 → 
  cost_per_book * books_bought = initial_amount - amount_left → 
  cost_per_book = 5 := by
sorry

end NUMINAMATH_CALUDE_book_cost_calculation_l2205_220523


namespace NUMINAMATH_CALUDE_tangent_circle_height_l2205_220552

/-- A circle tangent to y = x^3 at two points -/
structure TangentCircle where
  a : ℝ  -- x-coordinate of the tangent point
  b : ℝ  -- y-coordinate of the circle's center
  r : ℝ  -- radius of the circle

/-- The circle is tangent to y = x^3 at (a, a^3) and (-a, a^3) -/
def is_tangent (c : TangentCircle) : Prop :=
  c.a^2 + (c.a^3 - c.b)^2 = c.r^2 ∧
  c.a^6 + (1 - 2*c.b)*c.a^3 + c.b^2 - c.r^2 = 0

/-- The center of the circle is higher than the tangent points by 1/2 -/
theorem tangent_circle_height (c : TangentCircle) (h : is_tangent c) : 
  c.b - c.a^3 = 1/2 :=
sorry

end NUMINAMATH_CALUDE_tangent_circle_height_l2205_220552


namespace NUMINAMATH_CALUDE_gaspard_empty_bags_iff_even_sum_l2205_220535

/-- Represents the state of the bags -/
structure BagState where
  m : ℕ
  n : ℕ

/-- Defines the allowed operations on the bags -/
inductive Operation
  | RemoveEqual : ℕ → Operation
  | TripleOne : Bool → Operation

/-- Applies an operation to a bag state -/
def applyOperation (state : BagState) (op : Operation) : BagState :=
  match op with
  | Operation.RemoveEqual k => ⟨state.m - k, state.n - k⟩
  | Operation.TripleOne true => ⟨3 * state.m, state.n⟩
  | Operation.TripleOne false => ⟨state.m, 3 * state.n⟩

/-- Defines when a bag state is empty -/
def isEmptyState (state : BagState) : Prop :=
  state.m = 0 ∧ state.n = 0

/-- Defines when a sequence of operations can empty the bags -/
def canEmpty (initialState : BagState) : Prop :=
  ∃ (ops : List Operation), isEmptyState (ops.foldl applyOperation initialState)

/-- The main theorem: Gaspard can empty both bags iff m + n is even -/
theorem gaspard_empty_bags_iff_even_sum (m n : ℕ) (hm : m ≥ 1) (hn : n ≥ 1) :
    canEmpty ⟨m, n⟩ ↔ Even (m + n) := by
  sorry


end NUMINAMATH_CALUDE_gaspard_empty_bags_iff_even_sum_l2205_220535


namespace NUMINAMATH_CALUDE_boys_neither_happy_nor_sad_boys_neither_happy_nor_sad_is_6_l2205_220559

theorem boys_neither_happy_nor_sad (total_children : Nat) (happy_children : Nat) (sad_children : Nat) 
  (neither_children : Nat) (total_boys : Nat) (total_girls : Nat) (happy_boys : Nat) (sad_girls : Nat) : Nat :=
  by
  -- Assumptions
  have h1 : total_children = 60 := by sorry
  have h2 : happy_children = 30 := by sorry
  have h3 : sad_children = 10 := by sorry
  have h4 : neither_children = 20 := by sorry
  have h5 : total_boys = 18 := by sorry
  have h6 : total_girls = 42 := by sorry
  have h7 : happy_boys = 6 := by sorry
  have h8 : sad_girls = 4 := by sorry
  
  -- Proof
  sorry

-- The theorem statement
theorem boys_neither_happy_nor_sad_is_6 : 
  boys_neither_happy_nor_sad 60 30 10 20 18 42 6 4 = 6 := by sorry

end NUMINAMATH_CALUDE_boys_neither_happy_nor_sad_boys_neither_happy_nor_sad_is_6_l2205_220559


namespace NUMINAMATH_CALUDE_prime_product_l2205_220509

theorem prime_product (p q : ℕ) : 
  Prime p ∧ Prime q ∧ Prime (q^2 - p^2) → p * q = 6 :=
by sorry

end NUMINAMATH_CALUDE_prime_product_l2205_220509


namespace NUMINAMATH_CALUDE_sum_of_powers_of_i_l2205_220537

def i : ℂ := Complex.I

theorem sum_of_powers_of_i : 
  (Finset.range 2015).sum (λ n => i ^ n) = i :=
sorry

end NUMINAMATH_CALUDE_sum_of_powers_of_i_l2205_220537


namespace NUMINAMATH_CALUDE_geometric_sequence_property_l2205_220530

-- Define a geometric sequence
def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

-- Define the theorem
theorem geometric_sequence_property (a : ℕ → ℝ) :
  is_geometric_sequence a →
  (3 * (a 3)^2 - 11 * (a 3) + 9 = 0) →
  (3 * (a 9)^2 - 11 * (a 9) + 9 = 0) →
  (a 6 = Real.sqrt 3 ∨ a 6 = -Real.sqrt 3) :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_property_l2205_220530


namespace NUMINAMATH_CALUDE_tank_plastering_cost_l2205_220503

/-- Calculates the cost of plastering a rectangular tank's walls and bottom. -/
def plasteringCost (length width depth : ℝ) (costPerSquareMeter : ℝ) : ℝ :=
  let bottomArea := length * width
  let wallArea := 2 * (length * depth + width * depth)
  let totalArea := bottomArea + wallArea
  totalArea * costPerSquareMeter

/-- Theorem stating the cost of plastering a specific tank. -/
theorem tank_plastering_cost :
  let length : ℝ := 25
  let width : ℝ := 12
  let depth : ℝ := 6
  let costPerSquareMeter : ℝ := 0.75  -- 75 paise = 0.75 rupees
  plasteringCost length width depth costPerSquareMeter = 558 := by
  sorry

#eval plasteringCost 25 12 6 0.75

end NUMINAMATH_CALUDE_tank_plastering_cost_l2205_220503


namespace NUMINAMATH_CALUDE_smallest_k_with_remainder_one_l2205_220548

theorem smallest_k_with_remainder_one : ∃! k : ℕ,
  k > 1 ∧
  k % 23 = 1 ∧
  k % 7 = 1 ∧
  k % 3 = 1 ∧
  ∀ m : ℕ, m > 1 ∧ m % 23 = 1 ∧ m % 7 = 1 ∧ m % 3 = 1 → k ≤ m :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_k_with_remainder_one_l2205_220548


namespace NUMINAMATH_CALUDE_ellipse_condition_l2205_220558

/-- Represents the equation of a conic section -/
structure ConicSection where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ
  e : ℝ
  f : ℝ

/-- Checks if a conic section is a non-degenerate ellipse -/
def isNonDegenerateEllipse (conic : ConicSection) (m : ℝ) : Prop :=
  conic.a > 0 ∧ conic.b > 0 ∧ conic.a * conic.b * m > 0

/-- The main theorem stating the condition for the given equation to be a non-degenerate ellipse -/
theorem ellipse_condition (m : ℝ) : 
  isNonDegenerateEllipse ⟨3, 2, 0, -6, -16, -m⟩ m ↔ m > -35 := by sorry

end NUMINAMATH_CALUDE_ellipse_condition_l2205_220558


namespace NUMINAMATH_CALUDE_square_area_l2205_220534

-- Define a point in 2D space
structure Point2D where
  x : ℝ
  y : ℝ

-- Define the square
def square (p1 p2 : Point2D) : ℝ := 
  (p2.x - p1.x)^2 + (p2.y - p1.y)^2

-- Theorem statement
theorem square_area (p1 p2 : Point2D) (h : p1 = ⟨1, 2⟩ ∧ p2 = ⟨4, 6⟩) : 
  square p1 p2 = 25 := by
  sorry

end NUMINAMATH_CALUDE_square_area_l2205_220534


namespace NUMINAMATH_CALUDE_gcd_957_1537_l2205_220564

theorem gcd_957_1537 : Nat.gcd 957 1537 = 29 := by
  sorry

end NUMINAMATH_CALUDE_gcd_957_1537_l2205_220564


namespace NUMINAMATH_CALUDE_fair_haired_women_percentage_l2205_220532

/-- Given that 30% of employees are women with fair hair and 75% of employees have fair hair,
    prove that 40% of fair-haired employees are women. -/
theorem fair_haired_women_percentage
  (total_employees : ℝ)
  (women_fair_hair_percentage : ℝ)
  (fair_hair_percentage : ℝ)
  (h1 : women_fair_hair_percentage = 30 / 100)
  (h2 : fair_hair_percentage = 75 / 100) :
  (women_fair_hair_percentage * total_employees) / (fair_hair_percentage * total_employees) = 40 / 100 :=
by sorry

end NUMINAMATH_CALUDE_fair_haired_women_percentage_l2205_220532


namespace NUMINAMATH_CALUDE_total_rehabilitation_centers_l2205_220598

/-- The number of rehabilitation centers visited by Lisa, Jude, Han, and Jane -/
def total_centers (lisa jude han jane : ℕ) : ℕ := lisa + jude + han + jane

/-- Theorem stating the total number of rehabilitation centers visited -/
theorem total_rehabilitation_centers :
  ∃ (lisa jude han jane : ℕ),
    lisa = 6 ∧
    jude = lisa / 2 ∧
    han = 2 * jude - 2 ∧
    jane = 2 * han + 6 ∧
    total_centers lisa jude han jane = 27 := by
  sorry

end NUMINAMATH_CALUDE_total_rehabilitation_centers_l2205_220598


namespace NUMINAMATH_CALUDE_min_value_expression_l2205_220542

theorem min_value_expression (x y : ℝ) : 
  (x * y - 2)^2 + (x + y - 1)^2 ≥ 0 ∧ 
  ∃ (a b : ℝ), (a * b - 2)^2 + (a + b - 1)^2 = 0 := by
  sorry

end NUMINAMATH_CALUDE_min_value_expression_l2205_220542


namespace NUMINAMATH_CALUDE_zero_in_interval_l2205_220516

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log (1/2) + x - a

theorem zero_in_interval (a : ℝ) :
  a ∈ Set.Ioo 1 3 →
  ∃ x ∈ Set.Ioo 2 8, f a x = 0 ∧
  ¬(∀ a : ℝ, (∃ x ∈ Set.Ioo 2 8, f a x = 0) → a ∈ Set.Ioo 1 3) :=
by sorry

end NUMINAMATH_CALUDE_zero_in_interval_l2205_220516


namespace NUMINAMATH_CALUDE_price_difference_is_24_l2205_220528

/-- The original price of the smartphone --/
def original_price : ℚ := 800

/-- The single discount rate offered by the first store --/
def single_discount_rate : ℚ := 25 / 100

/-- The first discount rate offered by the second store --/
def first_discount_rate : ℚ := 20 / 100

/-- The second discount rate offered by the second store --/
def second_discount_rate : ℚ := 10 / 100

/-- The price after applying a single discount --/
def price_after_single_discount : ℚ := original_price * (1 - single_discount_rate)

/-- The price after applying two successive discounts --/
def price_after_successive_discounts : ℚ := 
  original_price * (1 - first_discount_rate) * (1 - second_discount_rate)

/-- Theorem stating that the difference between the two final prices is $24 --/
theorem price_difference_is_24 : 
  price_after_single_discount - price_after_successive_discounts = 24 := by
  sorry


end NUMINAMATH_CALUDE_price_difference_is_24_l2205_220528


namespace NUMINAMATH_CALUDE_not_all_prime_l2205_220508

theorem not_all_prime (a b c : ℕ) (h1 : a > b) (h2 : b > c) (h3 : c ≥ 3)
  (h4 : a ∣ b * c + b + c) (h5 : b ∣ c * a + c + a) (h6 : c ∣ a * b + a + b) :
  ¬(Nat.Prime a ∧ Nat.Prime b ∧ Nat.Prime c) := by
  sorry


end NUMINAMATH_CALUDE_not_all_prime_l2205_220508


namespace NUMINAMATH_CALUDE_f_composition_equals_126_l2205_220596

-- Define the function f
def f (x : ℝ) : ℝ := 5 * x - 4

-- State the theorem
theorem f_composition_equals_126 : f (f (f 2)) = 126 := by sorry

end NUMINAMATH_CALUDE_f_composition_equals_126_l2205_220596


namespace NUMINAMATH_CALUDE_complex_equality_l2205_220575

theorem complex_equality (z : ℂ) : Complex.abs (z + 2) = Complex.abs (z - 3) → z = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_complex_equality_l2205_220575


namespace NUMINAMATH_CALUDE_numbered_cube_consecutive_pairs_l2205_220536

/-- Represents a cube with numbers on its faces -/
structure NumberedCube where
  numbers : Fin 6 → ℕ
  distinct : ∀ i j, i ≠ j → numbers i ≠ numbers j

/-- Checks if two faces are adjacent on a cube -/
def adjacent (f1 f2 : Fin 6) : Prop := sorry

/-- Checks if two numbers are consecutive -/
def consecutive (n1 n2 : ℕ) : Prop := n2 = n1 + 1 ∨ n1 = n2 + 1

/-- Theorem: A cube numbered with consecutive integers from 1 to 6 
    has at least two pairs of adjacent faces with consecutive numbers -/
theorem numbered_cube_consecutive_pairs (c : NumberedCube) 
  (h_range : ∀ i, c.numbers i ∈ Finset.range 6) : 
  ∃ (f1 f2 f3 f4 : Fin 6), f1 ≠ f2 ∧ f3 ≠ f4 ∧ (f1, f2) ≠ (f3, f4) ∧ 
    adjacent f1 f2 ∧ adjacent f3 f4 ∧ 
    consecutive (c.numbers f1) (c.numbers f2) ∧ 
    consecutive (c.numbers f3) (c.numbers f4) := by
  sorry

end NUMINAMATH_CALUDE_numbered_cube_consecutive_pairs_l2205_220536


namespace NUMINAMATH_CALUDE_unique_function_existence_l2205_220531

theorem unique_function_existence (g : ℂ → ℂ) (ω a : ℂ) 
  (h1 : ω^3 = 1) (h2 : ω ≠ 1) :
  ∃! f : ℂ → ℂ, ∀ z : ℂ, f z + f (ω * z + a) = g z := by
  sorry

end NUMINAMATH_CALUDE_unique_function_existence_l2205_220531


namespace NUMINAMATH_CALUDE_floor_plus_self_eq_18_3_l2205_220567

theorem floor_plus_self_eq_18_3 :
  ∃! s : ℝ, ⌊s⌋ + s = 18.3 ∧ s = 9.3 := by sorry

end NUMINAMATH_CALUDE_floor_plus_self_eq_18_3_l2205_220567


namespace NUMINAMATH_CALUDE_max_value_f_positive_three_distinct_roots_condition_l2205_220580

noncomputable section

-- Define the function f
def f (x : ℝ) : ℝ :=
  if x > 0 then 1 - x^2 * Real.log x else Real.exp (-x - 2)

-- Part 1: Maximum value of f(x) for x > 0
theorem max_value_f_positive (x : ℝ) (h : x > 0) :
  f x ≤ 1 + 1 / (2 * Real.exp 1) :=
sorry

-- Part 2: Condition for three distinct real roots
theorem three_distinct_roots_condition (a b : ℝ) (h : a ≥ 0) :
  (∃ x y z : ℝ, x ≠ y ∧ y ≠ z ∧ x ≠ z ∧
    f x + a * x^2 + b * x = 0 ∧
    f y + a * y^2 + b * y = 0 ∧
    f z + a * z^2 + b * z = 0) ↔
  b < -2 * Real.sqrt 2 ∨ b ≥ 1 / Real.exp 1 :=
sorry

end NUMINAMATH_CALUDE_max_value_f_positive_three_distinct_roots_condition_l2205_220580


namespace NUMINAMATH_CALUDE_right_triangle_side_length_l2205_220519

theorem right_triangle_side_length 
  (P Q R : ℝ × ℝ) 
  (right_angle : (Q.1 - P.1) * (R.1 - P.1) + (Q.2 - P.2) * (R.2 - P.2) = 0) 
  (cos_R : Real.cos (Real.arccos ((3 * Real.sqrt 65) / 65)) = (3 * Real.sqrt 65) / 65) 
  (hypotenuse : Real.sqrt ((P.1 - R.1)^2 + (P.2 - R.2)^2) = Real.sqrt 169) :
  Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2) = (3 * Real.sqrt 65) / 5 := by
  sorry


end NUMINAMATH_CALUDE_right_triangle_side_length_l2205_220519
