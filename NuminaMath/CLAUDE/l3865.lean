import Mathlib

namespace smallest_c_is_correct_l3865_386586

/-- The number of positive divisors of a positive integer -/
def num_divisors (n : ℕ+) : ℕ := sorry

/-- The smallest positive real value c such that d(n) ≤ c * √n for all positive integers n -/
noncomputable def smallest_c : ℝ := Real.sqrt 3

theorem smallest_c_is_correct :
  (∀ n : ℕ+, (num_divisors n : ℝ) ≤ smallest_c * Real.sqrt n) ∧
  (∀ c : ℝ, 0 < c → c < smallest_c →
    ∃ n : ℕ+, (num_divisors n : ℝ) > c * Real.sqrt n) :=
by sorry

end smallest_c_is_correct_l3865_386586


namespace product_comparison_l3865_386580

theorem product_comparison (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (1.1 * a) * (1.13 * b) * (0.8 * c) < a * b * c :=
by sorry

end product_comparison_l3865_386580


namespace count_valid_bouquets_l3865_386522

/-- The number of valid bouquet combinations -/
def num_bouquets : ℕ := 11

/-- Represents a bouquet with roses and carnations -/
structure Bouquet where
  roses : ℕ
  carnations : ℕ

/-- The cost of a single rose -/
def rose_cost : ℕ := 4

/-- The cost of a single carnation -/
def carnation_cost : ℕ := 2

/-- The total budget for the bouquet -/
def total_budget : ℕ := 60

/-- Checks if a bouquet is valid according to the problem constraints -/
def is_valid_bouquet (b : Bouquet) : Prop :=
  b.roses ≥ 5 ∧
  b.roses * rose_cost + b.carnations * carnation_cost = total_budget

/-- The main theorem stating that there are exactly 11 valid bouquet combinations -/
theorem count_valid_bouquets :
  (∃ (bouquets : Finset Bouquet),
    bouquets.card = num_bouquets ∧
    (∀ b ∈ bouquets, is_valid_bouquet b) ∧
    (∀ b : Bouquet, is_valid_bouquet b → b ∈ bouquets)) :=
sorry

end count_valid_bouquets_l3865_386522


namespace annie_future_age_l3865_386534

theorem annie_future_age (anna_current_age : ℕ) (annie_current_age : ℕ) : 
  anna_current_age = 13 →
  annie_current_age = 3 * anna_current_age →
  (3 * anna_current_age + (annie_current_age - anna_current_age) = 65) :=
by
  sorry


end annie_future_age_l3865_386534


namespace consecutive_page_numbers_sum_l3865_386575

theorem consecutive_page_numbers_sum (n : ℕ) : 
  n > 0 ∧ n * (n + 1) = 19881 → n + (n + 1) = 283 := by
  sorry

end consecutive_page_numbers_sum_l3865_386575


namespace purely_imaginary_condition_l3865_386539

def is_purely_imaginary (z : ℂ) : Prop := z.re = 0 ∧ z.im ≠ 0

theorem purely_imaginary_condition (m : ℝ) :
  let z : ℂ := Complex.mk (m + 1) (m - 1)
  is_purely_imaginary z → m = -1 := by sorry

end purely_imaginary_condition_l3865_386539


namespace complement_union_theorem_l3865_386551

def U : Set ℕ := {0, 1, 2, 3, 4}
def A : Set ℕ := {1, 2, 3}
def B : Set ℕ := {2, 4}

theorem complement_union_theorem : (U \ A) ∪ B = {0, 2, 4} := by sorry

end complement_union_theorem_l3865_386551


namespace ratio_equality_l3865_386560

theorem ratio_equality (x : ℚ) : (1 : ℚ) / 3 = (5 : ℚ) / (3 * x) → x = 5 := by
  sorry

end ratio_equality_l3865_386560


namespace toothpick_problem_l3865_386530

theorem toothpick_problem (n : ℕ) : 
  n > 5000 ∧
  n % 10 = 9 ∧
  n % 9 = 8 ∧
  n % 8 = 7 ∧
  n % 7 = 6 ∧
  n % 6 = 5 ∧
  n % 5 = 4 →
  n = 5039 :=
by sorry

end toothpick_problem_l3865_386530


namespace inequality_proof_l3865_386574

theorem inequality_proof (a b x y : ℝ) 
  (h1 : x^2 + y^2 ≤ 1) 
  (h2 : a^2 + b^2 ≤ 2) : 
  |b * (x^2 - y^2) + 2 * a * x * y| ≤ Real.sqrt 2 := by
  sorry

end inequality_proof_l3865_386574


namespace packet_weight_difference_l3865_386597

theorem packet_weight_difference (a b c d e : ℝ) : 
  (a + b + c) / 3 = 84 →
  (a + b + c + d) / 4 = 80 →
  (b + c + d + e) / 4 = 79 →
  a = 75 →
  e - d = 3 :=
by sorry

end packet_weight_difference_l3865_386597


namespace product_variation_l3865_386591

theorem product_variation (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) :
  (5 * a) * b = 5 * (a * b) := by sorry

end product_variation_l3865_386591


namespace video_views_proof_l3865_386531

/-- Calculates the total views of a video given initial views and subsequent increases -/
def total_views (initial_views : ℕ) (increase_factor : ℕ) (additional_views : ℕ) : ℕ :=
  initial_views + increase_factor * initial_views + additional_views

/-- Theorem stating that given the specific conditions, the total views equal 94000 -/
theorem video_views_proof :
  let initial_views : ℕ := 4000
  let increase_factor : ℕ := 10
  let additional_views : ℕ := 50000
  total_views initial_views increase_factor additional_views = 94000 := by
  sorry

#eval total_views 4000 10 50000

end video_views_proof_l3865_386531


namespace smallest_cube_ending_112_l3865_386510

theorem smallest_cube_ending_112 : ∃ n : ℕ+, (
  n^3 ≡ 112 [ZMOD 1000] ∧
  ∀ m : ℕ+, m^3 ≡ 112 [ZMOD 1000] → n ≤ m
) ∧ n = 14 := by
  sorry

end smallest_cube_ending_112_l3865_386510


namespace tangent_to_ln_curve_l3865_386554

theorem tangent_to_ln_curve (k : ℝ) : 
  (∃ x : ℝ, x > 0 ∧ k * x = Real.log x ∧ (∀ y : ℝ, y > 0 → k * y ≤ Real.log y)) → 
  k = 1 / Real.exp 1 :=
sorry

end tangent_to_ln_curve_l3865_386554


namespace inscribed_cube_volume_l3865_386509

/-- The volume of a cube inscribed in a sphere, which is itself inscribed in a larger cube -/
theorem inscribed_cube_volume (large_cube_edge : ℝ) (h : large_cube_edge = 12) :
  let sphere_diameter := large_cube_edge
  let small_cube_diagonal := sphere_diameter
  let small_cube_edge := small_cube_diagonal / Real.sqrt 3
  let small_cube_volume := small_cube_edge ^ 3
  small_cube_volume = 192 * Real.sqrt 3 := by
  sorry

end inscribed_cube_volume_l3865_386509


namespace angle_between_vectors_l3865_386587

/-- The angle between two planar vectors -/
def angle_between (a b : ℝ × ℝ) : ℝ := sorry

theorem angle_between_vectors (a b : ℝ × ℝ) 
  (h1 : a.1 * b.1 + a.2 * b.2 = -1)  -- dot product condition
  (h2 : a.1^2 + a.2^2 = 4)           -- |a| = 2 condition
  (h3 : b.1^2 + b.2^2 = 1)           -- |b| = 1 condition
  : angle_between a b = 2 * π / 3 := by
  sorry

end angle_between_vectors_l3865_386587


namespace square_cutting_solution_l3865_386559

/-- Represents the cutting of an 8x8 square into smaller pieces -/
structure SquareCutting where
  /-- The number of 2x2 squares -/
  num_squares : ℕ
  /-- The number of 1x4 rectangles -/
  num_rectangles : ℕ
  /-- The total length of cuts -/
  total_cut_length : ℕ

/-- Theorem stating the solution to the square cutting problem -/
theorem square_cutting_solution :
  ∃ (cut : SquareCutting),
    cut.num_squares = 10 ∧
    cut.num_rectangles = 6 ∧
    cut.total_cut_length = 54 ∧
    cut.num_squares + cut.num_rectangles = 64 / 4 ∧
    8 * cut.num_squares + 10 * cut.num_rectangles = 32 + 2 * cut.total_cut_length :=
by sorry

end square_cutting_solution_l3865_386559


namespace abs_inequality_l3865_386593

theorem abs_inequality (a b : ℝ) (h : a^2 + b^2 ≤ 4) :
  |3 * a^2 - 8 * a * b - 3 * b^2| ≤ 20 := by
  sorry

end abs_inequality_l3865_386593


namespace toys_per_week_l3865_386532

/-- The number of days worked per week -/
def days_per_week : ℕ := 3

/-- The number of toys produced per day -/
def toys_per_day : ℝ := 2133.3333333333335

/-- Theorem: The number of toys produced per week is 6400 -/
theorem toys_per_week : ℕ := by
  sorry

end toys_per_week_l3865_386532


namespace prop_a_necessary_not_sufficient_for_prop_b_l3865_386524

-- Define propositions p and q
variable (p q : Prop)

-- Define Proposition A
def PropA : Prop := p → q

-- Define Proposition B
def PropB : Prop := p ↔ q

-- Theorem to prove
theorem prop_a_necessary_not_sufficient_for_prop_b :
  (PropA p q → PropB p q) ∧ ¬(PropB p q → PropA p q) :=
sorry

end prop_a_necessary_not_sufficient_for_prop_b_l3865_386524


namespace base7_subtraction_l3865_386549

/-- Converts a base 7 number to decimal --/
def base7ToDecimal (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (7 ^ i)) 0

/-- Converts a decimal number to base 7 --/
def decimalToBase7 (n : Nat) : List Nat :=
  if n = 0 then [0] else
    let rec aux (m : Nat) (acc : List Nat) :=
      if m = 0 then acc else aux (m / 7) ((m % 7) :: acc)
    aux n []

theorem base7_subtraction :
  let a := [4, 3, 2, 1]  -- 1234 in base 7
  let b := [2, 5, 6]     -- 652 in base 7
  let result := [2, 5, 2] -- 252 in base 7
  decimalToBase7 (base7ToDecimal a - base7ToDecimal b) = result := by
  sorry

end base7_subtraction_l3865_386549


namespace steve_exceeds_wayne_in_2004_l3865_386543

def money_at_year (initial : ℕ) (multiplier : ℚ) (year : ℕ) : ℚ :=
  initial * multiplier ^ year

def steve_money (year : ℕ) : ℚ := money_at_year 100 2 year

def wayne_money (year : ℕ) : ℚ := money_at_year 10000 (1/2) year

def first_year_steve_exceeds_wayne : ℕ := 2004

theorem steve_exceeds_wayne_in_2004 :
  (∀ y : ℕ, y < first_year_steve_exceeds_wayne → steve_money y ≤ wayne_money y) ∧
  steve_money first_year_steve_exceeds_wayne > wayne_money first_year_steve_exceeds_wayne :=
sorry

end steve_exceeds_wayne_in_2004_l3865_386543


namespace product_of_3_6_and_0_5_l3865_386526

theorem product_of_3_6_and_0_5 : 3.6 * 0.5 = 1.8 := by
  sorry

end product_of_3_6_and_0_5_l3865_386526


namespace product_of_reversed_digits_l3865_386519

theorem product_of_reversed_digits (A B : ℕ) (k : ℕ) : 
  0 ≤ A ∧ A ≤ 9 ∧ 0 ≤ B ∧ B ≤ 9 →
  (10 * A + B) * (10 * B + A) = k →
  (k + 1) % 101 = 0 →
  k = 403 := by
sorry

end product_of_reversed_digits_l3865_386519


namespace problem_solution_l3865_386555

theorem problem_solution (n : ℝ) (h : n + 1/n = 10) : n^2 + 1/n^2 + 5 = 103 := by
  sorry

end problem_solution_l3865_386555


namespace least_n_satisfying_inequality_l3865_386548

theorem least_n_satisfying_inequality : 
  ∃ n : ℕ+, (∀ k : ℕ+, k < n → (1 : ℚ) / k - (1 : ℚ) / (k + 1) ≥ (1 : ℚ) / 15) ∧
             ((1 : ℚ) / n - (1 : ℚ) / (n + 1) < (1 : ℚ) / 15) ∧
             n = 4 :=
by sorry

end least_n_satisfying_inequality_l3865_386548


namespace rectangle_area_with_inscribed_circle_rectangle_area_is_300_l3865_386576

theorem rectangle_area_with_inscribed_circle : ℝ → ℝ → ℝ → Prop :=
  fun radius ratio area =>
    let width := 2 * radius
    let length := ratio * width
    area = length * width

theorem rectangle_area_is_300 :
  ∃ (radius ratio : ℝ),
    radius = 5 ∧
    ratio = 3 ∧
    rectangle_area_with_inscribed_circle radius ratio 300 := by
  sorry

end rectangle_area_with_inscribed_circle_rectangle_area_is_300_l3865_386576


namespace math_city_intersections_l3865_386566

/-- Represents a city with a given number of streets -/
structure City where
  num_streets : ℕ
  no_parallel_streets : Bool
  no_three_streets_intersect : Bool

/-- Calculates the maximum number of intersections in a city -/
def max_intersections (c : City) : ℕ :=
  if c.num_streets ≤ 1 then 0
  else (c.num_streets - 1) * (c.num_streets - 2) / 2

/-- Theorem: A city with 12 streets, no parallel streets, and no three streets intersecting at a point has 66 intersections -/
theorem math_city_intersections :
  ∀ (c : City), c.num_streets = 12 → c.no_parallel_streets = true → c.no_three_streets_intersect = true →
  max_intersections c = 66 :=
by sorry

end math_city_intersections_l3865_386566


namespace train_length_l3865_386536

/-- The length of a train given its speed, time to cross a bridge, and the bridge length -/
theorem train_length (train_speed : ℝ) (crossing_time : ℝ) (bridge_length : ℝ) : 
  train_speed = 45 * 1000 / 3600 → 
  crossing_time = 30 → 
  bridge_length = 255 → 
  train_speed * crossing_time - bridge_length = 120 := by
  sorry

#check train_length

end train_length_l3865_386536


namespace unique_magnitude_quadratic_roots_l3865_386572

theorem unique_magnitude_quadratic_roots : ∃! m : ℝ, ∀ z : ℂ, z^2 - 6*z + 20 = 0 → Complex.abs z = m := by
  sorry

end unique_magnitude_quadratic_roots_l3865_386572


namespace adjacent_pairs_difference_l3865_386544

/-- Given a circular arrangement of symbols, this theorem proves that the difference
    between the number of adjacent pairs of one symbol and the number of adjacent pairs
    of another symbol equals the difference in the total count of these symbols. -/
theorem adjacent_pairs_difference (p q a b : ℕ) : 
  (p + q > 0) →  -- Ensure the circle is not empty
  (a ≤ p) →      -- Number of X pairs cannot exceed total X's
  (b ≤ q) →      -- Number of 0 pairs cannot exceed total 0's
  (a = 0 → p ≤ 1) →  -- If no X pairs, at most one X
  (b = 0 → q ≤ 1) →  -- If no 0 pairs, at most one 0
  a - b = p - q :=
by sorry

end adjacent_pairs_difference_l3865_386544


namespace solve_for_x_l3865_386538

theorem solve_for_x (x y : ℝ) (h1 : x + 3 * y = 33) (h2 : y = 10) : x = 3 := by
  sorry

end solve_for_x_l3865_386538


namespace choir_group_division_l3865_386546

theorem choir_group_division (sopranos altos tenors basses : ℕ) 
  (h_sopranos : sopranos = 10)
  (h_altos : altos = 15)
  (h_tenors : tenors = 12)
  (h_basses : basses = 18) :
  ∃ (n : ℕ), n = 3 ∧ 
  n > 0 ∧
  sopranos % n = 0 ∧ 
  altos % n = 0 ∧ 
  tenors % n = 0 ∧ 
  basses % n = 0 ∧
  sopranos / n < (altos + tenors + basses) / n ∧
  ∀ m : ℕ, m > n → 
    (sopranos % m ≠ 0 ∨ 
     altos % m ≠ 0 ∨ 
     tenors % m ≠ 0 ∨ 
     basses % m ≠ 0 ∨
     sopranos / m ≥ (altos + tenors + basses) / m) :=
by sorry

end choir_group_division_l3865_386546


namespace min_value_theorem_l3865_386527

theorem min_value_theorem (a m n : ℝ) (ha : a > 0) (ha_neq : a ≠ 1) (hm : m > 0) (hn : n > 0) :
  let f := fun x => a^(x - 1) - 2
  let A := (1, -1)
  (m * A.1 - n * A.2 - 1 = 0) →
  (∀ x, f x = -1 → x = 1) →
  (∃ (min_val : ℝ), min_val = 3 + 2 * Real.sqrt 2 ∧
    ∀ (m' n' : ℝ), m' > 0 → n' > 0 → m' * A.1 - n' * A.2 - 1 = 0 →
      1 / m' + 2 / n' ≥ min_val) :=
by sorry

end min_value_theorem_l3865_386527


namespace triangle_properties_l3865_386515

-- Define the triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the conditions
def triangle_conditions (t : Triangle) : Prop :=
  2 * t.b * Real.cos t.A + t.a = 2 * t.c ∧
  t.c = 8 ∧
  Real.sin t.A = (3 * Real.sqrt 3) / 14

-- State the theorem
theorem triangle_properties (t : Triangle) 
  (h : triangle_conditions t) : 
  t.B = π / 3 ∧ 
  (1 / 2 * t.a * t.c * Real.sin t.B) = 6 * Real.sqrt 3 := by
  sorry

end triangle_properties_l3865_386515


namespace circle_radius_from_longest_chord_l3865_386596

theorem circle_radius_from_longest_chord (c : ℝ) (h : c > 0) :
  (∃ r : ℝ, r > 0 ∧ c = 2 * r) → c / 2 = 5 :=
by sorry

end circle_radius_from_longest_chord_l3865_386596


namespace sum_of_three_numbers_l3865_386562

theorem sum_of_three_numbers : 2.12 + 0.004 + 0.345 = 2.469 := by
  sorry

end sum_of_three_numbers_l3865_386562


namespace right_triangle_pythagorean_l3865_386595

theorem right_triangle_pythagorean (a b c : ℝ) : 
  a = 1 ∧ b = Real.sqrt 3 ∧ c = 2 → a^2 + b^2 = c^2 :=
by sorry

end right_triangle_pythagorean_l3865_386595


namespace ellipse_foci_l3865_386585

/-- The equation of an ellipse in standard form -/
def is_ellipse (x y : ℝ) : Prop := x^2 / 16 + y^2 / 25 = 1

/-- The coordinates of a focus of the ellipse -/
def is_focus (x y : ℝ) : Prop := x = 0 ∧ (y = 3 ∨ y = -3)

/-- Theorem: The foci of the given ellipse are at (0, ±3) -/
theorem ellipse_foci :
  ∀ x y : ℝ, is_ellipse x y → ∃ fx fy : ℝ, is_focus fx fy :=
sorry

end ellipse_foci_l3865_386585


namespace ahmed_has_13_goats_l3865_386563

def adam_goats : ℕ := 7

def andrew_goats : ℕ := 2 * adam_goats + 5

def ahmed_goats : ℕ := andrew_goats - 6

theorem ahmed_has_13_goats : ahmed_goats = 13 := by
  sorry

end ahmed_has_13_goats_l3865_386563


namespace remainder_problem_l3865_386565

theorem remainder_problem (n : ℤ) (h : 2 * n % 15 = 2) : n % 30 = 1 := by
  sorry

end remainder_problem_l3865_386565


namespace subset_implies_a_range_l3865_386516

def A : Set ℝ := {x | |x| * (x^2 - 4*x + 3) < 0}

def B (a : ℝ) : Set ℝ := {x | 2^(1-x) + a ≤ 0 ∧ x^2 - 2*(a+7)*x + 5 ≤ 0}

theorem subset_implies_a_range (a : ℝ) : A ⊆ B a → -4 ≤ a ∧ a ≤ -1 := by
  sorry

end subset_implies_a_range_l3865_386516


namespace fred_initial_sheets_l3865_386571

def initial_sheets : ℕ → ℕ → ℕ → ℕ → Prop :=
  fun x received given final =>
    x + received - given = final

theorem fred_initial_sheets :
  ∃ x : ℕ, initial_sheets x 307 156 363 ∧ x = 212 :=
by
  sorry

end fred_initial_sheets_l3865_386571


namespace p_less_q_less_r_l3865_386517

-- Define the logarithm function (base 2)
noncomputable def lg (x : ℝ) : ℝ := Real.log x / Real.log 2

-- State the theorem
theorem p_less_q_less_r (a b : ℝ) 
  (h1 : a > b) (h2 : b > 1) 
  (P : ℝ) (hP : P = lg a * lg b)
  (Q : ℝ) (hQ : Q = lg a + lg b)
  (R : ℝ) (hR : R = lg (a * b)) :
  P < Q ∧ Q < R := by
  sorry

end p_less_q_less_r_l3865_386517


namespace multiples_of_15_between_17_and_152_l3865_386545

theorem multiples_of_15_between_17_and_152 : 
  (Finset.filter (fun n => n % 15 = 0 ∧ n > 17 ∧ n < 152) (Finset.range 152)).card = 9 := by
  sorry

end multiples_of_15_between_17_and_152_l3865_386545


namespace complex_modulus_l3865_386588

theorem complex_modulus (z : ℂ) (h : z = Complex.I / (1 + Complex.I)) : Complex.abs z = Real.sqrt 2 / 2 := by
  sorry

end complex_modulus_l3865_386588


namespace range_of_a_l3865_386547

-- Define the propositions p and q
def p (x : ℝ) : Prop := x < -3 ∨ x > 1
def q (x a : ℝ) : Prop := x > a

-- State the theorem
theorem range_of_a (h : ∀ x, ¬(p x) → ¬(q x a)) : a ≥ 1 := by
  sorry

end range_of_a_l3865_386547


namespace wheel_configuration_theorem_l3865_386535

/-- Represents a configuration of wheels with spokes -/
structure WheelConfiguration where
  total_spokes : Nat
  max_spokes_per_wheel : Nat

/-- Checks if a given number of wheels is possible for the configuration -/
def isPossible (config : WheelConfiguration) (num_wheels : Nat) : Prop :=
  num_wheels * config.max_spokes_per_wheel ≥ config.total_spokes

theorem wheel_configuration_theorem (config : WheelConfiguration) 
  (h1 : config.total_spokes = 7)
  (h2 : config.max_spokes_per_wheel = 3) :
  isPossible config 3 ∧ ¬isPossible config 2 := by
  sorry

#check wheel_configuration_theorem

end wheel_configuration_theorem_l3865_386535


namespace book_pages_theorem_l3865_386590

/-- Represents the number of pages read each night --/
structure ReadingPattern :=
  (night1 : ℕ)
  (night2 : ℕ)
  (night3 : ℕ)
  (night4 : ℕ)

/-- Calculates the total number of pages in the book --/
def totalPages (rp : ReadingPattern) : ℕ :=
  rp.night1 + rp.night2 + rp.night3 + rp.night4

/-- Theorem: The book has 100 pages in total --/
theorem book_pages_theorem (rp : ReadingPattern) 
  (h1 : rp.night1 = 15)
  (h2 : rp.night2 = 2 * rp.night1)
  (h3 : rp.night3 = rp.night2 + 5)
  (h4 : rp.night4 = 20) : 
  totalPages rp = 100 := by
  sorry


end book_pages_theorem_l3865_386590


namespace seaweed_for_livestock_l3865_386508

def total_seaweed : ℝ := 500

def fire_percentage : ℝ := 0.4
def medicinal_percentage : ℝ := 0.2
def food_and_feed_percentage : ℝ := 0.4

def human_consumption_ratio : ℝ := 0.3

theorem seaweed_for_livestock (total : ℝ) (fire_pct : ℝ) (med_pct : ℝ) (food_feed_pct : ℝ) (human_ratio : ℝ) 
    (h1 : total = total_seaweed)
    (h2 : fire_pct = fire_percentage)
    (h3 : med_pct = medicinal_percentage)
    (h4 : food_feed_pct = food_and_feed_percentage)
    (h5 : human_ratio = human_consumption_ratio)
    (h6 : fire_pct + med_pct + food_feed_pct = 1) :
  food_feed_pct * total * (1 - human_ratio) = 140 := by
  sorry

end seaweed_for_livestock_l3865_386508


namespace square_area_equals_side_perimeter_l3865_386567

/-- A square with area numerically equal to its side length has a perimeter of 4 units. -/
theorem square_area_equals_side_perimeter :
  ∀ s : ℝ, s > 0 → s^2 = s → 4 * s = 4 := by
  sorry

end square_area_equals_side_perimeter_l3865_386567


namespace coin_box_problem_l3865_386583

theorem coin_box_problem :
  ∃ (N B : ℕ), 
    N = 9 * (B - 2) ∧
    N - 6 * (B - 3) = 3 :=
by sorry

end coin_box_problem_l3865_386583


namespace floor_equation_solution_l3865_386537

theorem floor_equation_solution (a b : ℕ+) :
  (⌊(a : ℝ)^2 / b⌋ + ⌊(b : ℝ)^2 / a⌋ = ⌊((a : ℝ)^2 + (b : ℝ)^2) / (a * b)⌋ + (a * b : ℝ)) ↔
  (∃ k : ℕ+, (a = k ∧ b = k^2 + 1) ∨ (a = k^2 + 1 ∧ b = k)) :=
by sorry

end floor_equation_solution_l3865_386537


namespace pgcd_and_divisibility_properties_l3865_386520

/-- For a ≥ 2 and m ≥ n ≥ 1, prove properties of PGCD and divisibility -/
theorem pgcd_and_divisibility_properties 
  (a m n : ℕ) 
  (ha : a ≥ 2) 
  (hmn : m ≥ n) 
  (hn : n ≥ 1) :
  (Nat.gcd (a^m - 1) (a^n - 1) = Nat.gcd (a^(m-n) - 1) (a^n - 1)) ∧
  (Nat.gcd (a^m - 1) (a^n - 1) = a^(Nat.gcd m n) - 1) ∧
  ((a^m - 1) ∣ (a^n - 1) ↔ m ∣ n) :=
by sorry

end pgcd_and_divisibility_properties_l3865_386520


namespace large_monkey_doll_cost_l3865_386540

/-- The cost of a large monkey doll satisfies the given conditions --/
theorem large_monkey_doll_cost : ∃ (L : ℝ), 
  (L > 0) ∧ 
  (300 / (L - 2) = 300 / L + 25) ∧ 
  (L = 6) := by
sorry

end large_monkey_doll_cost_l3865_386540


namespace adjacent_diff_at_least_five_l3865_386541

/-- Represents a cell in the 8x8 grid -/
structure Cell :=
  (row : Fin 8)
  (col : Fin 8)

/-- Represents the 8x8 grid filled with integers from 1 to 64 -/
def Grid := Cell → Fin 64

/-- Two cells are adjacent if they share a common edge -/
def adjacent (c1 c2 : Cell) : Prop :=
  (c1.row = c2.row ∧ c1.col.val + 1 = c2.col.val) ∨
  (c1.row = c2.row ∧ c1.col.val = c2.col.val + 1) ∨
  (c1.row.val + 1 = c2.row.val ∧ c1.col = c2.col) ∨
  (c1.row.val = c2.row.val + 1 ∧ c1.col = c2.col)

/-- Main theorem: In any 8x8 grid filled with integers from 1 to 64,
    there exist two adjacent cells whose values differ by at least 5 -/
theorem adjacent_diff_at_least_five (g : Grid) : 
  ∃ c1 c2 : Cell, adjacent c1 c2 ∧ (g c1).val + 5 ≤ (g c2).val ∨ (g c2).val + 5 ≤ (g c1).val :=
sorry

end adjacent_diff_at_least_five_l3865_386541


namespace sum_equals_power_of_two_l3865_386529

theorem sum_equals_power_of_two : 29 + 12 + 23 = 2^6 := by
  sorry

end sum_equals_power_of_two_l3865_386529


namespace unique_n_satisfying_equation_l3865_386578

theorem unique_n_satisfying_equation : ∃! (n : ℕ), 
  n + Int.floor (Real.sqrt n) + Int.floor (Real.sqrt (Real.sqrt n)) = 2017 :=
by sorry

end unique_n_satisfying_equation_l3865_386578


namespace counterexample_exists_l3865_386582

/-- A function that returns the sum of digits of a natural number in base 4038 -/
def sumOfDigitsBase4038 (n : ℕ) : ℕ := sorry

/-- A predicate that checks if a natural number is "good" (sum of digits in base 4038 is divisible by 2019) -/
def isGood (n : ℕ) : Prop := sumOfDigitsBase4038 n % 2019 = 0

theorem counterexample_exists (a : ℝ) : a ≥ 2019 →
  ∃ (seq : ℕ → ℕ), 
    (∀ m n : ℕ, m ≠ n → seq m ≠ seq n) ∧ 
    (∀ n : ℕ, (seq n : ℝ) ≤ a * n) ∧
    (∃ N : ℕ, ∀ n : ℕ, n ≥ N → isGood (seq n)) :=
sorry

end counterexample_exists_l3865_386582


namespace perfect_square_with_three_or_fewer_swaps_l3865_386511

/-- Represents a permutation of digits --/
def Permutation := List Nat

/-- Checks if a permutation represents a perfect square --/
def is_perfect_square (p : Permutation) : Prop :=
  ∃ n : Nat, n * n = p.foldl (fun acc d => acc * 10 + d) 0

/-- Counts the number of swaps needed to transform one permutation into another --/
def swap_count (p1 p2 : Permutation) : Nat :=
  sorry

/-- The initial permutation of digits from 1 to 9 --/
def initial_permutation : Permutation := [1, 2, 3, 4, 5, 6, 7, 8, 9]

/-- Theorem: There exists a permutation of digits 1-9 that forms a perfect square 
    and can be achieved with 3 or fewer swaps from the initial permutation --/
theorem perfect_square_with_three_or_fewer_swaps :
  ∃ (final_perm : Permutation), 
    is_perfect_square final_perm ∧ 
    swap_count initial_permutation final_perm ≤ 3 :=
  sorry

end perfect_square_with_three_or_fewer_swaps_l3865_386511


namespace largest_good_and_smallest_bad_l3865_386518

def is_good_number (M : ℕ) : Prop :=
  ∃ a b c d : ℕ, M ≤ a ∧ a < b ∧ b ≤ c ∧ c < d ∧ d ≤ M + 49 ∧ a * d = b * c

theorem largest_good_and_smallest_bad :
  (is_good_number 576) ∧
  (∀ M : ℕ, M ≥ 577 → ¬(is_good_number M)) ∧
  (¬(is_good_number 443)) ∧
  (∀ M : ℕ, 288 < M ∧ M ≤ 442 → is_good_number M) :=
by sorry

end largest_good_and_smallest_bad_l3865_386518


namespace players_in_both_games_l3865_386553

theorem players_in_both_games (total : ℕ) (outdoor : ℕ) (indoor : ℕ) 
  (h1 : total = 400) 
  (h2 : outdoor = 350) 
  (h3 : indoor = 110) : 
  outdoor + indoor - total = 60 := by
  sorry

end players_in_both_games_l3865_386553


namespace probability_top_given_not_female_is_one_eighth_l3865_386503

/-- Represents the probability of selecting a top student given the student is not female -/
def probability_top_given_not_female (total_students : ℕ) (female_students : ℕ) (top_fraction : ℚ) (top_female_fraction : ℚ) : ℚ :=
  let male_students := total_students - female_students
  let top_students := (total_students : ℚ) * top_fraction
  let male_top_students := top_students * (1 - top_female_fraction)
  male_top_students / male_students

/-- Theorem stating the probability of selecting a top student given the student is not female -/
theorem probability_top_given_not_female_is_one_eighth :
  probability_top_given_not_female 60 20 (1/6) (1/2) = 1/8 := by
  sorry


end probability_top_given_not_female_is_one_eighth_l3865_386503


namespace cubic_root_sum_l3865_386514

theorem cubic_root_sum (a b c : ℝ) : 
  0 < a ∧ a < 1 ∧
  0 < b ∧ b < 1 ∧
  0 < c ∧ c < 1 ∧
  a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
  24 * a^3 - 36 * a^2 + 14 * a - 1 = 0 ∧
  24 * b^3 - 36 * b^2 + 14 * b - 1 = 0 ∧
  24 * c^3 - 36 * c^2 + 14 * c - 1 = 0 →
  1 / (1 - a) + 1 / (1 - b) + 1 / (1 - c) = 3 / 2 :=
by sorry

end cubic_root_sum_l3865_386514


namespace morning_faces_l3865_386577

/-- Represents a cuboid room -/
structure CuboidRoom where
  totalFaces : Nat
  eveningFaces : Nat

/-- Theorem: The number of faces Samuel painted in the morning is 3 -/
theorem morning_faces (room : CuboidRoom) 
  (h1 : room.totalFaces = 6)
  (h2 : room.eveningFaces = 3) : 
  room.totalFaces - room.eveningFaces = 3 := by
  sorry

#check morning_faces

end morning_faces_l3865_386577


namespace square_intersection_perimeter_l3865_386525

/-- Given a square with side length 2a centered at the origin, intersected by the line y = x/3,
    the perimeter of one resulting quadrilateral divided by a equals (14 + 2√10) / 3. -/
theorem square_intersection_perimeter (a : ℝ) (h : a > 0) :
  let square := {p : ℝ × ℝ | max (|p.1|) (|p.2|) = a}
  let line := {p : ℝ × ℝ | p.2 = p.1 / 3}
  let intersection := square ∩ line
  let quadrilateral_perimeter := 
    2 * (a - a / 3) +  -- vertical sides
    2 * a +            -- horizontal side
    Real.sqrt ((2*a)^2 + (2*a/3)^2) -- diagonal
  quadrilateral_perimeter / a = (14 + 2 * Real.sqrt 10) / 3 :=
by sorry

end square_intersection_perimeter_l3865_386525


namespace race_distance_l3865_386564

/-- Represents the race scenario where p runs x% faster than q, q has a head start, and the race ends in a tie -/
def race_scenario (x y : ℝ) : Prop :=
  ∀ (vq : ℝ), vq > 0 →
    let vp := vq * (1 + x / 100)
    let head_start := (x / 10) * y
    let dq := 1000 * y / x
    let dp := dq + head_start
    dq / vq = dp / vp

/-- The theorem stating that both runners cover the same distance in the given race scenario -/
theorem race_distance (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) :
  race_scenario x y →
  ∃ (d : ℝ), d = 1000 * y / x ∧ 
    (∀ (vq : ℝ), vq > 0 →
      let vp := vq * (1 + x / 100)
      let head_start := (x / 10) * y
      d = 1000 * y / x ∧ d + head_start = (10000 * y + x * y^2) / (10 * x)) :=
sorry

end race_distance_l3865_386564


namespace fibonacci_periodicity_l3865_386557

/-- Fibonacci sequence -/
def fib : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | n + 2 => fib n + fib (n + 1)

/-- The smallest positive integer k(m) satisfying the Fibonacci periodicity modulo m -/
def k_m (m : ℕ) : ℕ := sorry

theorem fibonacci_periodicity (m : ℕ) (h : m > 0) :
  (∃ i j : ℕ, 0 ≤ i ∧ i < j ∧ j ≤ m^2 ∧ fib i % m = fib j % m ∧ fib (i + 1) % m = fib (j + 1) % m) ∧
  (∃ k : ℕ, k > 0 ∧ ∀ n : ℕ, fib (n + k) % m = fib n % m) ∧
  (∀ n : ℕ, fib (n + k_m m) % m = fib n % m) ∧
  (fib (k_m m) % m = 0 ∧ fib (k_m m + 1) % m = 1) ∧
  (∀ k : ℕ, k > 0 → (∀ n : ℕ, fib (n + k) % m = fib n % m) ↔ k_m m ∣ k) :=
by sorry

end fibonacci_periodicity_l3865_386557


namespace polynomial_product_sum_l3865_386579

theorem polynomial_product_sum (b₁ b₂ b₃ c₁ c₂ c₃ : ℝ) : 
  (∀ x : ℝ, x^8 - x^7 + x^6 - x^5 + x^4 - x^3 + x^2 - x + 1 = 
    (x^2 + b₁*x + c₁) * (x^2 + b₂*x + c₂) * (x^2 + b₃*x + c₃)) →
  b₁*c₁ + b₂*c₂ + b₃*c₃ = -1 := by
  sorry

end polynomial_product_sum_l3865_386579


namespace minimum_j_10_l3865_386561

/-- A function is stringent if it satisfies the given inequality for all positive integers x and y. -/
def Stringent (f : ℕ+ → ℤ) : Prop :=
  ∀ x y : ℕ+, f x + f y ≥ 2 * x.val ^ 2 - y.val

/-- The sum of j from 1 to 15 -/
def SumJ (j : ℕ+ → ℤ) : ℤ :=
  (Finset.range 15).sum (λ i => j ⟨i + 1, Nat.succ_pos i⟩)

theorem minimum_j_10 :
  ∃ j : ℕ+ → ℤ,
    Stringent j ∧
    (∀ k : ℕ+ → ℤ, Stringent k → SumJ j ≤ SumJ k) ∧
    j ⟨10, by norm_num⟩ = 137 ∧
    (∀ k : ℕ+ → ℤ, Stringent k → (∀ i : ℕ+, SumJ j ≤ SumJ k) → j ⟨10, by norm_num⟩ ≤ k ⟨10, by norm_num⟩) :=
by sorry

end minimum_j_10_l3865_386561


namespace triangle_side_length_l3865_386589

theorem triangle_side_length (A B C : Real) (a b c : Real) :
  0 < a ∧ 0 < b ∧ 0 < c →
  A = π/3 →
  Real.cos B = (2 * Real.sqrt 7) / 7 →
  b = 3 →
  a = (3 * Real.sqrt 7) / 2 := by
  sorry

end triangle_side_length_l3865_386589


namespace smallest_k_is_two_l3865_386581

/-- A five-digit number -/
def FiveDigitNumber := Fin 100000

/-- Predicate to check if a number has digits in non-decreasing order -/
def hasNonDecreasingDigits (n : FiveDigitNumber) : Prop := sorry

/-- Predicate to check if two numbers have at least one digit in common -/
def hasCommonDigit (n m : FiveDigitNumber) : Prop := sorry

/-- The set of five-digit numbers satisfying the problem conditions -/
def SpecialSet (k : ℕ) : Set FiveDigitNumber := sorry

theorem smallest_k_is_two :
  ∀ k : ℕ,
    (∀ n : FiveDigitNumber, hasNonDecreasingDigits n →
      ∃ m ∈ SpecialSet k, hasCommonDigit n m) →
    k ≥ 2 :=
sorry

end smallest_k_is_two_l3865_386581


namespace largest_invertible_interval_for_g_l3865_386584

-- Define the function g(x)
def g (x : ℝ) : ℝ := 3 * x^2 - 9 * x + 4

-- Define the theorem
theorem largest_invertible_interval_for_g :
  ∃ (a : ℝ), 
    (∀ (I : Set ℝ), (2 ∈ I) → (∀ (x y : ℝ), x ∈ I → y ∈ I → x ≠ y → g x ≠ g y) → 
      I ⊆ {x : ℝ | a ≤ x}) ∧
    ({x : ℝ | a ≤ x} ⊆ {x : ℝ | ∀ (y : ℝ), y ∈ {x : ℝ | a ≤ x} → y ≠ x → g y ≠ g x}) ∧
    a = 3/2 := by
  sorry

end largest_invertible_interval_for_g_l3865_386584


namespace constant_sum_of_powers_l3865_386512

theorem constant_sum_of_powers (n : ℕ+) :
  (∀ x y z : ℝ, x + y + z = 0 → x * y * z = 1 → 
    ∃ c : ℝ, ∀ a b d : ℝ, a + b + d = 0 → a * b * d = 1 → 
      a^(n : ℕ) + b^(n : ℕ) + d^(n : ℕ) = c) ↔ n = 1 ∨ n = 3 :=
by sorry

end constant_sum_of_powers_l3865_386512


namespace sine_intersection_theorem_l3865_386523

theorem sine_intersection_theorem (a b : ℕ) (h : a ≠ b) :
  ∃ c : ℕ, c ≠ a ∧ c ≠ b ∧
  ∀ x : ℝ, Real.sin (a * x) = Real.sin (b * x) →
            Real.sin (c * x) = Real.sin (a * x) :=
by sorry

end sine_intersection_theorem_l3865_386523


namespace equal_real_roots_no_real_solutions_l3865_386533

-- Define the quadratic equation
def quadratic_equation (a b x : ℝ) : Prop := a * x^2 + b * x + (1/4 : ℝ) = 0

-- Part 1: Equal real roots condition
theorem equal_real_roots (a b : ℝ) (h : a ≠ 0) :
  a = 1 ∧ b = 1 → ∃! x : ℝ, quadratic_equation a b x :=
sorry

-- Part 2: No real solutions condition
theorem no_real_solutions (a b : ℝ) (h1 : a > 1) (h2 : 0 < b) (h3 : b < 1) :
  ¬∃ x : ℝ, quadratic_equation a b x :=
sorry

end equal_real_roots_no_real_solutions_l3865_386533


namespace complement_of_M_l3865_386507

def U : Set ℝ := Set.univ

def M : Set ℝ := {x | x^2 < 2*x}

theorem complement_of_M : Set.compl M = {x : ℝ | x ≤ 0 ∨ x ≥ 2} := by sorry

end complement_of_M_l3865_386507


namespace ali_flower_sales_l3865_386558

/-- Calculates the total number of flowers sold by Ali -/
def total_flowers_sold (monday : ℕ) (tuesday : ℕ) : ℕ :=
  monday + tuesday + 2 * monday

theorem ali_flower_sales : total_flowers_sold 4 8 = 20 := by
  sorry

end ali_flower_sales_l3865_386558


namespace profit_revenue_relationship_l3865_386598

/-- Represents the financial data of a company over two years -/
structure CompanyFinancials where
  prevRevenue : ℝ
  prevProfit : ℝ
  currRevenue : ℝ
  currProfit : ℝ

/-- The theorem stating the relationship between profits and revenues -/
theorem profit_revenue_relationship (c : CompanyFinancials)
  (h1 : c.currRevenue = 0.8 * c.prevRevenue)
  (h2 : c.currProfit = 0.2 * c.currRevenue)
  (h3 : c.currProfit = 1.6000000000000003 * c.prevProfit) :
  c.prevProfit / c.prevRevenue = 0.1 := by
  sorry

#check profit_revenue_relationship

end profit_revenue_relationship_l3865_386598


namespace salt_production_january_l3865_386521

/-- The salt production problem -/
theorem salt_production_january (
  monthly_increase : ℕ → ℝ)
  (average_daily_production : ℝ)
  (h1 : ∀ n : ℕ, n ≥ 1 ∧ n ≤ 11 → monthly_increase n = 100)
  (h2 : average_daily_production = 100.27397260273973)
  (h3 : ∃ january_production : ℝ,
    (january_production +
      (january_production + monthly_increase 1) +
      (january_production + monthly_increase 1 + monthly_increase 2) +
      (january_production + monthly_increase 1 + monthly_increase 2 + monthly_increase 3) +
      (january_production + monthly_increase 1 + monthly_increase 2 + monthly_increase 3 + monthly_increase 4) +
      (january_production + monthly_increase 1 + monthly_increase 2 + monthly_increase 3 + monthly_increase 4 + monthly_increase 5) +
      (january_production + monthly_increase 1 + monthly_increase 2 + monthly_increase 3 + monthly_increase 4 + monthly_increase 5 + monthly_increase 6) +
      (january_production + monthly_increase 1 + monthly_increase 2 + monthly_increase 3 + monthly_increase 4 + monthly_increase 5 + monthly_increase 6 + monthly_increase 7) +
      (january_production + monthly_increase 1 + monthly_increase 2 + monthly_increase 3 + monthly_increase 4 + monthly_increase 5 + monthly_increase 6 + monthly_increase 7 + monthly_increase 8) +
      (january_production + monthly_increase 1 + monthly_increase 2 + monthly_increase 3 + monthly_increase 4 + monthly_increase 5 + monthly_increase 6 + monthly_increase 7 + monthly_increase 8 + monthly_increase 9) +
      (january_production + monthly_increase 1 + monthly_increase 2 + monthly_increase 3 + monthly_increase 4 + monthly_increase 5 + monthly_increase 6 + monthly_increase 7 + monthly_increase 8 + monthly_increase 9 + monthly_increase 10) +
      (january_production + monthly_increase 1 + monthly_increase 2 + monthly_increase 3 + monthly_increase 4 + monthly_increase 5 + monthly_increase 6 + monthly_increase 7 + monthly_increase 8 + monthly_increase 9 + monthly_increase 10 + monthly_increase 11)) / 365 = average_daily_production) :
  ∃ january_production : ℝ, january_production = 2500 :=
by sorry

end salt_production_january_l3865_386521


namespace sequence_modulo_l3865_386570

/-- Given a prime number p > 3, we define a sequence a_n as follows:
    a_n = n for n ∈ {0, 1, ..., p-1}
    a_n = a_{n-1} + a_{n-p} for n ≥ p
    This theorem states that a_{p^3} ≡ p-1 (mod p) -/
theorem sequence_modulo (p : ℕ) (hp : p.Prime ∧ p > 3) : 
  ∃ a : ℕ → ℕ, 
    (∀ n < p, a n = n) ∧ 
    (∀ n ≥ p, a n = a (n-1) + a (n-p)) ∧ 
    a (p^3) ≡ p-1 [MOD p] := by
  sorry

end sequence_modulo_l3865_386570


namespace probability_sum_not_less_than_6_l3865_386506

/-- Represents a tetrahedral die with faces numbered 1, 2, 3, 5 -/
def TetrahedralDie : Type := Fin 4

/-- The possible face values of the tetrahedral die -/
def face_values : List ℕ := [1, 2, 3, 5]

/-- The total number of possible outcomes when rolling two dice -/
def total_outcomes : ℕ := 16

/-- Predicate to check if the sum of two face values is not less than 6 -/
def sum_not_less_than_6 (a b : ℕ) : Prop := a + b ≥ 6

/-- The number of favorable outcomes (sum not less than 6) -/
def favorable_outcomes : ℕ := 8

/-- Theorem stating that the probability of the sum being not less than 6 is 1/2 -/
theorem probability_sum_not_less_than_6 :
  (favorable_outcomes : ℚ) / total_outcomes = 1 / 2 := by sorry

end probability_sum_not_less_than_6_l3865_386506


namespace oblong_perimeter_l3865_386556

theorem oblong_perimeter :
  ∀ l w : ℕ,
    l > w →
    l * 3 = w * 4 →
    l * w = 4624 →
    2 * l + 2 * w = 182 := by
  sorry

end oblong_perimeter_l3865_386556


namespace expression_equals_one_l3865_386568

theorem expression_equals_one (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) (h_sum : a + b + c = 1) :
  (a^2 * b^2 / ((a^2 - b*c) * (b^2 - a*c))) +
  (a^2 * c^2 / ((a^2 - b*c) * (c^2 - a*b))) +
  (b^2 * c^2 / ((b^2 - a*c) * (c^2 - a*b))) = 1 := by
sorry

end expression_equals_one_l3865_386568


namespace P_consecutive_coprime_l3865_386542

-- Define the Fibonacci sequence
def fib : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | (n + 2) => fib (n + 1) + fib n

-- Define P(n) as given in the problem
def P : ℕ → ℕ
  | 0 => 0  -- Undefined in the original problem, added for completeness
  | 1 => 0  -- Undefined in the original problem, added for completeness
  | (n + 2) => 
    if n % 2 = 0 then
      (fib ((n / 2) + 1) + fib ((n / 2) - 1)) ^ 2
    else
      fib (n + 2) + fib ((n - 1) / 2)

-- State the theorem
theorem P_consecutive_coprime (k : ℕ) (h : k ≥ 3) : 
  Nat.gcd (P k) (P (k + 1)) = 1 := by
  sorry

end P_consecutive_coprime_l3865_386542


namespace first_pipe_fill_time_l3865_386504

/-- The time it takes for the first pipe to fill the cistern -/
def T : ℝ := 10

/-- The time it takes for the second pipe to fill the cistern -/
def second_pipe_time : ℝ := 12

/-- The time it takes for the third pipe to empty the cistern -/
def third_pipe_time : ℝ := 25

/-- The time it takes to fill the cistern when all pipes are opened simultaneously -/
def combined_time : ℝ := 6.976744186046512

theorem first_pipe_fill_time :
  (1 / T + 1 / second_pipe_time - 1 / third_pipe_time) * combined_time = 1 :=
sorry

end first_pipe_fill_time_l3865_386504


namespace sum_x_y_z_l3865_386573

/-- Given that:
    - 0.5% of x equals 0.65 rupees
    - 1.25% of y equals 1.04 rupees
    - 2.5% of z equals 75% of x
    Prove that the sum of x, y, and z is 4113.2 rupees -/
theorem sum_x_y_z (x y z : ℝ) 
  (hx : 0.005 * x = 0.65)
  (hy : 0.0125 * y = 1.04)
  (hz : 0.025 * z = 0.75 * x) :
  x + y + z = 4113.2 := by
  sorry

end sum_x_y_z_l3865_386573


namespace sqrt_x_minus_8_meaningful_l3865_386599

theorem sqrt_x_minus_8_meaningful (x : ℝ) : 
  (∃ y : ℝ, y ^ 2 = x - 8) ↔ x ≥ 8 := by sorry

end sqrt_x_minus_8_meaningful_l3865_386599


namespace nine_ones_squared_l3865_386513

def nine_ones : ℕ := 111111111

theorem nine_ones_squared :
  nine_ones ^ 2 = 12345678987654321 := by sorry

end nine_ones_squared_l3865_386513


namespace work_earnings_theorem_l3865_386501

/-- Given the following conditions:
  - I worked t+2 hours
  - I earned 4t-4 dollars per hour
  - Bob worked 4t-6 hours
  - Bob earned t+3 dollars per hour
  - I earned three dollars more than Bob
Prove that t = 7/2 -/
theorem work_earnings_theorem (t : ℚ) : 
  (t + 2) * (4 * t - 4) = (4 * t - 6) * (t + 3) + 3 → t = 7/2 := by
sorry

end work_earnings_theorem_l3865_386501


namespace max_sum_under_constraint_l3865_386528

theorem max_sum_under_constraint (a b c : ℝ) :
  a^2 + 4*b^2 + 9*c^2 - 2*a - 12*b + 6*c + 2 = 0 →
  a + b + c ≤ 17/3 :=
by sorry

end max_sum_under_constraint_l3865_386528


namespace snug_fit_circles_l3865_386502

/-- Given a circle of diameter 3 inches containing two circles of diameters 2 inches and 1 inch,
    the diameter of two additional identical circles that fit snugly within the larger circle
    is 12/7 inches. -/
theorem snug_fit_circles (R : ℝ) (r₁ : ℝ) (r₂ : ℝ) (d : ℝ) :
  R = 3/2 ∧ r₁ = 1 ∧ r₂ = 1/2 →
  d > 0 →
  (R - d)^2 + (R - d)^2 = (2*d)^2 →
  d = 6/7 :=
by sorry

end snug_fit_circles_l3865_386502


namespace three_false_propositions_l3865_386550

theorem three_false_propositions :
  (¬ ∀ a b : ℝ, (1 / a < 1 / b) → (a > b)) ∧
  (¬ ∀ a b c : ℝ, (a > b ∧ b > c) → (a * |c| > b * |c|)) ∧
  (¬ ∃ x₀ : ℝ, ∀ x : ℝ, x + 1/x ≥ 2) :=
by sorry

end three_false_propositions_l3865_386550


namespace cubic_sum_theorem_l3865_386552

theorem cubic_sum_theorem (x y z a b c : ℝ)
  (h1 : x + y + z = a)
  (h2 : x^2 + y^2 + z^2 = b^2)
  (h3 : x⁻¹ + y⁻¹ + z⁻¹ = c⁻¹)
  (hx : x ≠ 0)
  (hy : y ≠ 0)
  (hz : z ≠ 0) :
  x^3 + y^3 + z^3 = a^3 + (3/2) * (a^2 - b^2) * (c - a) := by
sorry

end cubic_sum_theorem_l3865_386552


namespace parabola_focus_l3865_386592

/-- The focus of a parabola with equation x = 4y^2 is at (1/16, 0) -/
theorem parabola_focus (x y : ℝ) : 
  (x = 4 * y^2) → (∃ p : ℝ, p > 0 ∧ x = y^2 / (4 * p) ∧ (1 / (16 : ℝ), 0) = (p, 0)) :=
by sorry

end parabola_focus_l3865_386592


namespace min_value_expression_l3865_386569

theorem min_value_expression (a b c : ℝ) (h1 : b > c) (h2 : c > a) (h3 : b ≠ 0) :
  ((a + 2*b)^2 + (b - 2*c)^2 + (c - 2*a)^2) / b^2 ≥ 25/3 := by
  sorry

end min_value_expression_l3865_386569


namespace det_sine_matrix_zero_l3865_386505

theorem det_sine_matrix_zero :
  let M : Matrix (Fin 3) (Fin 3) ℝ := λ i j ↦ 
    Real.sin (((i : ℕ) * 3 + (j : ℕ) + 2) : ℝ)
  Matrix.det M = 0 := by
  sorry

end det_sine_matrix_zero_l3865_386505


namespace square_sum_inequality_l3865_386594

theorem square_sum_inequality (a b : ℝ) : 2 * (a^2 + b^2) ≥ (a + b)^2 := by
  sorry

end square_sum_inequality_l3865_386594


namespace puppies_bought_l3865_386500

/-- The total number of puppies bought by Arven -/
def total_puppies : ℕ := 5

/-- The cost of each puppy on sale -/
def sale_price : ℕ := 150

/-- The cost of each puppy not on sale -/
def regular_price : ℕ := 175

/-- The number of puppies on sale -/
def sale_puppies : ℕ := 3

/-- The total cost of all puppies -/
def total_cost : ℕ := 800

theorem puppies_bought :
  total_puppies = sale_puppies + (total_cost - sale_puppies * sale_price) / regular_price :=
by sorry

end puppies_bought_l3865_386500
