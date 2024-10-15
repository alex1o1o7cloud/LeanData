import Mathlib

namespace NUMINAMATH_CALUDE_curve_to_linear_equation_l1249_124981

/-- Given a curve parameterized by (x, y) = (3t + 6, 5t - 3), where t is a real number,
    prove that it can be expressed as the linear equation y = (5/3)x - 13. -/
theorem curve_to_linear_equation :
  ∀ (t x y : ℝ), x = 3 * t + 6 ∧ y = 5 * t - 3 →
  y = (5 / 3 : ℝ) * x - 13 :=
by
  sorry

end NUMINAMATH_CALUDE_curve_to_linear_equation_l1249_124981


namespace NUMINAMATH_CALUDE_inequality_solution_l1249_124977

theorem inequality_solution (a : ℝ) :
  (a < 1/2 → ∀ x, x^2 - x + a - a^2 < 0 ↔ a < x ∧ x < 1-a) ∧
  (a > 1/2 → ∀ x, x^2 - x + a - a^2 < 0 ↔ 1-a < x ∧ x < a) ∧
  (a = 1/2 → ∀ x, ¬(x^2 - x + a - a^2 < 0)) := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_l1249_124977


namespace NUMINAMATH_CALUDE_sqrt_equation_roots_l1249_124927

theorem sqrt_equation_roots :
  ∃! (x : ℝ), x > 15 ∧ Real.sqrt (x + 15) - 7 / Real.sqrt (x + 15) = 6 ∧
  ∃ (y : ℝ), -15 < y ∧ y < -10 ∧ 
    (Real.sqrt (y + 15) - 7 / Real.sqrt (y + 15) = 6 → False) :=
by sorry

end NUMINAMATH_CALUDE_sqrt_equation_roots_l1249_124927


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_sets_l1249_124971

theorem quadratic_inequality_solution_sets (a : ℝ) :
  let S := {x : ℝ | x^2 + (a + 2) * x + 2 * a < 0}
  (a < 2 → S = {x : ℝ | -2 < x ∧ x < -a}) ∧
  (a = 2 → S = ∅) ∧
  (a > 2 → S = {x : ℝ | -a < x ∧ x < -2}) := by
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_sets_l1249_124971


namespace NUMINAMATH_CALUDE_geometric_sequence_ratio_l1249_124948

/-- A geometric sequence with positive terms -/
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, q > 0 ∧ ∀ n : ℕ, a (n + 1) = a n * q

/-- The first, third, and second terms form an arithmetic sequence -/
def arithmetic_condition (a : ℕ → ℝ) : Prop :=
  2 * ((1 / 2) * a 3) = a 1 + 2 * a 2

/-- All terms in the sequence are positive -/
def positive_terms (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, a n > 0

theorem geometric_sequence_ratio (a : ℕ → ℝ) :
  geometric_sequence a →
  positive_terms a →
  arithmetic_condition a →
  (a 9 + a 10) / (a 7 + a 8) = 3 + 2 * Real.sqrt 2 :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_ratio_l1249_124948


namespace NUMINAMATH_CALUDE_special_numbers_count_l1249_124935

def count_multiples (n : ℕ) (d : ℕ) : ℕ := (n / d : ℕ)

def count_special_numbers (upper_bound : ℕ) : ℕ :=
  count_multiples upper_bound 5 + count_multiples upper_bound 7 - count_multiples upper_bound 35

theorem special_numbers_count :
  count_special_numbers 3000 = 943 := by
  sorry

end NUMINAMATH_CALUDE_special_numbers_count_l1249_124935


namespace NUMINAMATH_CALUDE_lee_surpasses_hernandez_in_may_l1249_124902

def months : List String := ["March", "April", "May", "June", "July", "August"]

def hernandez_hrs : List Nat := [4, 8, 9, 5, 7, 6]
def lee_hrs : List Nat := [3, 9, 10, 6, 8, 8]

def cumulative_sum (list : List Nat) : List Nat :=
  list.scanl (· + ·) 0

def first_surpass (list1 list2 : List Nat) : Option Nat :=
  (list1.zip list2).findIdx? (fun (a, b) => b > a)

theorem lee_surpasses_hernandez_in_may :
  first_surpass (cumulative_sum hernandez_hrs) (cumulative_sum lee_hrs) = some 2 :=
sorry

end NUMINAMATH_CALUDE_lee_surpasses_hernandez_in_may_l1249_124902


namespace NUMINAMATH_CALUDE_num_mc_questions_is_two_l1249_124956

/-- The number of true-false questions in the quiz -/
def num_tf : ℕ := 4

/-- The number of answer choices for each multiple-choice question -/
def num_mc_choices : ℕ := 4

/-- The total number of ways to write the answer key -/
def total_ways : ℕ := 224

/-- The number of ways to answer the true-false questions, excluding all-same answers -/
def tf_ways : ℕ := 2^num_tf - 2

/-- Theorem stating that the number of multiple-choice questions is 2 -/
theorem num_mc_questions_is_two :
  ∃ (n : ℕ), tf_ways * (num_mc_choices^n) = total_ways ∧ n = 2 := by
  sorry

end NUMINAMATH_CALUDE_num_mc_questions_is_two_l1249_124956


namespace NUMINAMATH_CALUDE_isosceles_triangle_perimeter_isosceles_triangle_perimeter_proof_l1249_124999

/-- An isosceles triangle with side lengths 4 and 9 has a perimeter of 22. -/
theorem isosceles_triangle_perimeter : ℝ → ℝ → ℝ → Prop :=
  fun a b c =>
    (a = 4 ∨ a = 9) ∧ (b = 4 ∨ b = 9) ∧ (c = 4 ∨ c = 9) ∧  -- Side lengths are 4 or 9
    (a = b ∨ b = c ∨ a = c) ∧                              -- Isosceles condition
    (a + b > c ∧ b + c > a ∧ a + c > b) →                  -- Triangle inequality
    a + b + c = 22                                         -- Perimeter is 22

/-- Proof of the theorem -/
theorem isosceles_triangle_perimeter_proof : ∃ a b c, isosceles_triangle_perimeter a b c :=
  sorry

end NUMINAMATH_CALUDE_isosceles_triangle_perimeter_isosceles_triangle_perimeter_proof_l1249_124999


namespace NUMINAMATH_CALUDE_k_range_l1249_124996

-- Define the propositions p and q
def p (x k : ℝ) : Prop := x ≥ k
def q (x : ℝ) : Prop := 3 / (x + 1) < 1

-- Define the sufficient but not necessary condition
def sufficient_not_necessary (k : ℝ) : Prop :=
  (∀ x, p x k → q x) ∧ ¬(∀ x, q x → p x k)

-- Theorem statement
theorem k_range :
  ∀ k : ℝ, sufficient_not_necessary k ↔ k > 2 :=
sorry

end NUMINAMATH_CALUDE_k_range_l1249_124996


namespace NUMINAMATH_CALUDE_infinitely_many_squares_l1249_124901

theorem infinitely_many_squares (k : ℕ) (hk : k ≥ 2) :
  ∃ (f : ℕ → ℕ), Function.Injective f ∧
  ∀ (i : ℕ), ∃ (u v : ℕ), k * (f i) + 1 = u^2 ∧ (k + 1) * (f i) + 1 = v^2 :=
sorry

end NUMINAMATH_CALUDE_infinitely_many_squares_l1249_124901


namespace NUMINAMATH_CALUDE_soak_time_for_marinara_stain_l1249_124980

/-- The time needed to soak clothes for grass and marinara stains -/
theorem soak_time_for_marinara_stain 
  (grass_stain_time : ℕ) 
  (num_grass_stains : ℕ) 
  (num_marinara_stains : ℕ) 
  (total_soak_time : ℕ) 
  (h1 : grass_stain_time = 4)
  (h2 : num_grass_stains = 3)
  (h3 : num_marinara_stains = 1)
  (h4 : total_soak_time = 19) :
  total_soak_time - (grass_stain_time * num_grass_stains) = 7 := by
sorry

end NUMINAMATH_CALUDE_soak_time_for_marinara_stain_l1249_124980


namespace NUMINAMATH_CALUDE_vanessa_albums_l1249_124930

theorem vanessa_albums (phone_pics : ℕ) (camera_pics : ℕ) (pics_per_album : ℕ) : 
  phone_pics = 23 → 
  camera_pics = 7 → 
  pics_per_album = 6 → 
  (phone_pics + camera_pics) / pics_per_album = 5 := by
sorry

end NUMINAMATH_CALUDE_vanessa_albums_l1249_124930


namespace NUMINAMATH_CALUDE_quadratic_integer_solutions_l1249_124925

theorem quadratic_integer_solutions (p q x₁ x₂ : ℝ) : 
  (∃ (x : ℝ), x^2 + p*x + q = 0) →  -- Quadratic equation has real solutions
  (x₁^2 + p*x₁ + q = 0) →           -- x₁ is a solution
  (x₂^2 + p*x₂ + q = 0) →           -- x₂ is a solution
  (x₁ ≠ x₂) →                       -- Solutions are distinct
  |x₁ - x₂| = 1 →                   -- Absolute difference of solutions is 1
  |p - q| = 1 →                     -- Absolute difference of p and q is 1
  (∃ (m n k l : ℤ), (↑m : ℝ) = p ∧ (↑n : ℝ) = q ∧ (↑k : ℝ) = x₁ ∧ (↑l : ℝ) = x₂) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_integer_solutions_l1249_124925


namespace NUMINAMATH_CALUDE_correct_calculation_l1249_124928

theorem correct_calculation : 
  (5 + (-6) = -1) ∧ 
  (1 / Real.sqrt 2 ≠ Real.sqrt 2) ∧ 
  (3 * (-2) ≠ 6) ∧ 
  (Real.sin (30 * π / 180) ≠ Real.sqrt 3 / 3) :=
by sorry

end NUMINAMATH_CALUDE_correct_calculation_l1249_124928


namespace NUMINAMATH_CALUDE_student_assignment_count_l1249_124947

theorem student_assignment_count : ∀ (n m : ℕ),
  n = 4 ∧ m = 3 →
  (Nat.choose n 2 * (Nat.factorial m)) = (m * Nat.choose n 2 * 2) :=
by sorry

end NUMINAMATH_CALUDE_student_assignment_count_l1249_124947


namespace NUMINAMATH_CALUDE_income_mean_difference_l1249_124953

/-- The number of families in the dataset -/
def num_families : ℕ := 500

/-- The correct maximum income -/
def correct_max_income : ℕ := 120000

/-- The incorrect maximum income -/
def incorrect_max_income : ℕ := 1200000

/-- The sum of all incomes excluding the maximum -/
def T : ℕ := sorry

theorem income_mean_difference :
  (T + incorrect_max_income) / num_families - (T + correct_max_income) / num_families = 2160 :=
sorry

end NUMINAMATH_CALUDE_income_mean_difference_l1249_124953


namespace NUMINAMATH_CALUDE_prob_B_given_A₁_pairwise_mutually_exclusive_l1249_124942

-- Define the number of balls in each can
def can_A_red : ℕ := 5
def can_A_white : ℕ := 2
def can_A_black : ℕ := 3
def can_B_red : ℕ := 4
def can_B_white : ℕ := 3
def can_B_black : ℕ := 3

-- Define the total number of balls in each can
def total_A : ℕ := can_A_red + can_A_white + can_A_black
def total_B : ℕ := can_B_red + can_B_white + can_B_black

-- Define the events
def A₁ : Set ℕ := {x | x ≤ can_A_red}
def A₂ : Set ℕ := {x | can_A_red < x ∧ x ≤ can_A_red + can_A_white}
def A₃ : Set ℕ := {x | can_A_red + can_A_white < x ∧ x ≤ total_A}
def B : Set ℕ := {x | x ≤ can_B_red + 1}

-- Define the probability measure
noncomputable def P : Set ℕ → ℝ := sorry

-- Theorem 1: P(B|A₁) = 5/11
theorem prob_B_given_A₁ : P (B ∩ A₁) / P A₁ = 5 / 11 := by sorry

-- Theorem 2: A₁, A₂, A₃ are pairwise mutually exclusive
theorem pairwise_mutually_exclusive : 
  (A₁ ∩ A₂ = ∅) ∧ (A₁ ∩ A₃ = ∅) ∧ (A₂ ∩ A₃ = ∅) := by sorry

end NUMINAMATH_CALUDE_prob_B_given_A₁_pairwise_mutually_exclusive_l1249_124942


namespace NUMINAMATH_CALUDE_red_balls_count_l1249_124972

theorem red_balls_count (w r : ℕ) : 
  (w : ℚ) / r = 5 / 3 →  -- ratio of white to red balls
  w + 15 + r = 50 →     -- total after adding 15 white balls
  r = 12 := by           -- number of red balls
sorry

end NUMINAMATH_CALUDE_red_balls_count_l1249_124972


namespace NUMINAMATH_CALUDE_inequality_proof_l1249_124903

theorem inequality_proof (x y z : ℝ) (h : x * y * z = 1) :
  x^2 + y^2 + z^2 ≥ 1/x + 1/y + 1/z := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1249_124903


namespace NUMINAMATH_CALUDE_general_solution_valid_particular_solution_valid_l1249_124932

-- Define the general solution function
def f (C : ℝ) (x : ℝ) : ℝ := x^2 + C

-- Define the particular solution function
def g (x : ℝ) : ℝ := x^2 + 1

-- Theorem for the general solution
theorem general_solution_valid (C : ℝ) : 
  ∀ x, HasDerivAt (f C) (2 * x) x :=
sorry

-- Theorem for the particular solution
theorem particular_solution_valid : 
  g 1 = 2 ∧ ∀ x, HasDerivAt g (2 * x) x :=
sorry

end NUMINAMATH_CALUDE_general_solution_valid_particular_solution_valid_l1249_124932


namespace NUMINAMATH_CALUDE_smallest_inscribed_cube_volume_l1249_124904

theorem smallest_inscribed_cube_volume (outer_cube_edge : ℝ) : 
  outer_cube_edge = 16 →
  ∃ (largest_sphere_radius smallest_cube_edge : ℝ),
    largest_sphere_radius = outer_cube_edge / 2 ∧
    smallest_cube_edge = 16 / Real.sqrt 3 ∧
    smallest_cube_edge ^ 3 = 456 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_smallest_inscribed_cube_volume_l1249_124904


namespace NUMINAMATH_CALUDE_usual_baking_time_l1249_124965

/-- Represents the time in hours for Matthew's cake-making process -/
structure BakingTime where
  assembly : ℝ
  baking : ℝ
  decorating : ℝ

/-- The total time for Matthew's cake-making process -/
def total_time (t : BakingTime) : ℝ := t.assembly + t.baking + t.decorating

/-- Represents the scenario when the oven fails -/
def oven_failure (normal : BakingTime) : BakingTime :=
  { assembly := normal.assembly,
    baking := 2 * normal.baking,
    decorating := normal.decorating }

theorem usual_baking_time :
  ∃ (normal : BakingTime),
    normal.assembly = 1 ∧
    normal.decorating = 1 ∧
    total_time (oven_failure normal) = 5 ∧
    normal.baking = 1.5 := by
  sorry

end NUMINAMATH_CALUDE_usual_baking_time_l1249_124965


namespace NUMINAMATH_CALUDE_solution_set_inequality_1_solution_set_inequality_2_l1249_124966

-- Part 1
theorem solution_set_inequality_1 : 
  {x : ℝ | (2 * x) / (x - 2) ≤ 1} = Set.Ici (-2) ∩ Set.Iio 2 := by sorry

-- Part 2
theorem solution_set_inequality_2 (a : ℝ) (ha : a > 0) :
  {x : ℝ | a * x^2 + 2 * x + 1 > 0} = 
    if a = 1 then 
      {x : ℝ | x ≠ -1}
    else if a > 1 then 
      Set.univ
    else 
      Set.Iic ((- 1 - Real.sqrt (1 - a)) / a) ∪ Set.Ioi ((- 1 + Real.sqrt (1 - a)) / a) := by sorry

end NUMINAMATH_CALUDE_solution_set_inequality_1_solution_set_inequality_2_l1249_124966


namespace NUMINAMATH_CALUDE_p_sufficient_not_necessary_for_q_l1249_124914

-- Define p and q as predicates on real numbers
def p (x : ℝ) : Prop := |x - 3| < 1
def q (x : ℝ) : Prop := x^2 + x - 6 > 0

-- Theorem statement
theorem p_sufficient_not_necessary_for_q :
  (∀ x, p x → q x) ∧ (∃ x, q x ∧ ¬p x) :=
sorry

end NUMINAMATH_CALUDE_p_sufficient_not_necessary_for_q_l1249_124914


namespace NUMINAMATH_CALUDE_smallest_integer_satisfying_inequality_four_satisfies_inequality_four_is_smallest_integer_l1249_124978

theorem smallest_integer_satisfying_inequality :
  ∀ x : ℤ, (x : ℚ) / 4 + 3 / 7 > 4 / 3 → x ≥ 4 :=
by
  sorry

theorem four_satisfies_inequality :
  (4 : ℚ) / 4 + 3 / 7 > 4 / 3 :=
by
  sorry

theorem four_is_smallest_integer :
  ∀ x : ℤ, x < 4 → (x : ℚ) / 4 + 3 / 7 ≤ 4 / 3 :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_integer_satisfying_inequality_four_satisfies_inequality_four_is_smallest_integer_l1249_124978


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l1249_124943

def A : Set ℤ := {-1, 1, 2}
def B : Set ℤ := {x | x^2 = x}

theorem intersection_of_A_and_B : A ∩ B = {1} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l1249_124943


namespace NUMINAMATH_CALUDE_find_y_l1249_124929

theorem find_y (x : ℝ) (y : ℝ) (h1 : 1.5 * x = 0.3 * y) (h2 : x = 20) : y = 100 := by
  sorry

end NUMINAMATH_CALUDE_find_y_l1249_124929


namespace NUMINAMATH_CALUDE_no_periodic_sum_with_given_periods_l1249_124933

/-- A function is periodic if it takes at least two different values and there exists a positive period. -/
def Periodic (f : ℝ → ℝ) : Prop :=
  (∃ x y, f x ≠ f y) ∧ (∃ p > 0, ∀ x, f (x + p) = f x)

/-- The period of a function. -/
def Period (f : ℝ → ℝ) (p : ℝ) : Prop :=
  p > 0 ∧ ∀ x, f (x + p) = f x

/-- Theorem: There do not exist periodic functions g and h with periods 2 and π/2 respectively,
    such that g + h is also a periodic function. -/
theorem no_periodic_sum_with_given_periods :
  ¬ ∃ (g h : ℝ → ℝ),
    Periodic g ∧ Periodic h ∧ Period g 2 ∧ Period h (π / 2) ∧ Periodic (g + h) :=
by sorry

end NUMINAMATH_CALUDE_no_periodic_sum_with_given_periods_l1249_124933


namespace NUMINAMATH_CALUDE_intersection_M_complement_N_l1249_124920

-- Define the sets M and N
def M : Set ℝ := {x | x^2 - 2*x < 0}
def N : Set ℝ := {x | x > 1}

-- State the theorem
theorem intersection_M_complement_N : 
  M ∩ (Set.univ \ N) = {x : ℝ | 0 < x ∧ x ≤ 1} := by sorry

end NUMINAMATH_CALUDE_intersection_M_complement_N_l1249_124920


namespace NUMINAMATH_CALUDE_largest_possible_median_l1249_124962

def is_median (m : ℤ) (s : Finset ℤ) : Prop :=
  s.card = 5 ∧ (s.filter (λ x => x ≤ m)).card ≥ 3 ∧ (s.filter (λ x => x ≥ m)).card ≥ 3

theorem largest_possible_median :
  ∀ x y : ℤ, y = 2 * x →
  ∃ m : ℤ, is_median m {x, y, 3, 7, 9} ∧
    ∀ m' : ℤ, is_median m' {x, y, 3, 7, 9} → m' ≤ m ∧ m = 7 :=
by sorry

end NUMINAMATH_CALUDE_largest_possible_median_l1249_124962


namespace NUMINAMATH_CALUDE_max_q_minus_r_for_852_l1249_124974

theorem max_q_minus_r_for_852 :
  ∃ (q r : ℕ), 
    q > 0 ∧ r > 0 ∧ 
    852 = 21 * q + r ∧
    ∀ (q' r' : ℕ), q' > 0 → r' > 0 → 852 = 21 * q' + r' → q' - r' ≤ q - r ∧
    q - r = 28 :=
sorry

end NUMINAMATH_CALUDE_max_q_minus_r_for_852_l1249_124974


namespace NUMINAMATH_CALUDE_water_depth_calculation_l1249_124923

-- Define the heights of Ron and Dean
def ron_height : ℝ := 13
def dean_height : ℝ := ron_height + 4

-- Define the maximum depth at high tide
def max_depth : ℝ := 15 * dean_height

-- Define the current tide percentage and current percentage
def tide_percentage : ℝ := 0.75
def current_percentage : ℝ := 0.20

-- Theorem statement
theorem water_depth_calculation :
  let current_tide_depth := tide_percentage * max_depth
  let additional_depth := current_percentage * current_tide_depth
  current_tide_depth + additional_depth = 229.5 := by
  sorry

end NUMINAMATH_CALUDE_water_depth_calculation_l1249_124923


namespace NUMINAMATH_CALUDE_alexandras_magazines_l1249_124919

theorem alexandras_magazines (friday_magazines : ℕ) (saturday_magazines : ℕ) 
  (sunday_multiplier : ℕ) (monday_multiplier : ℕ) (chewed_magazines : ℕ) : 
  friday_magazines = 18 →
  saturday_magazines = 25 →
  sunday_multiplier = 5 →
  monday_multiplier = 3 →
  chewed_magazines = 10 →
  friday_magazines + saturday_magazines + 
  (sunday_multiplier * friday_magazines) + 
  (monday_multiplier * saturday_magazines) - 
  chewed_magazines = 198 := by
sorry

end NUMINAMATH_CALUDE_alexandras_magazines_l1249_124919


namespace NUMINAMATH_CALUDE_opposite_of_negative_seven_l1249_124970

theorem opposite_of_negative_seven :
  ∀ x : ℤ, x + (-7) = 0 → x = 7 :=
by sorry

end NUMINAMATH_CALUDE_opposite_of_negative_seven_l1249_124970


namespace NUMINAMATH_CALUDE_circle_area_circumference_ratio_l1249_124955

theorem circle_area_circumference_ratio (r₁ r₂ : ℝ) (h : r₁ > 0 ∧ r₂ > 0) :
  (π * r₁^2) / (π * r₂^2) = 16 / 25 →
  (2 * π * r₁) / (2 * π * r₂) = 4 / 5 := by
sorry

end NUMINAMATH_CALUDE_circle_area_circumference_ratio_l1249_124955


namespace NUMINAMATH_CALUDE_no_triple_exists_l1249_124916

theorem no_triple_exists : ¬∃ (a b c : ℕ+), 
  let p := (a.val - 2) * (b.val - 2) * (c.val - 2) + 12
  Nat.Prime p ∧ 
  (∃ k : ℕ+, k * p = a.val^2 + b.val^2 + c.val^2 + a.val * b.val * c.val - 2017) ∧
  p < a.val^2 + b.val^2 + c.val^2 + a.val * b.val * c.val - 2017 :=
by sorry

end NUMINAMATH_CALUDE_no_triple_exists_l1249_124916


namespace NUMINAMATH_CALUDE_x_range_l1249_124912

theorem x_range (x : ℝ) (h1 : x^2 - 2*x - 3 < 0) (h2 : 1/(x-2) < 0) : -1 < x ∧ x < 2 := by
  sorry

end NUMINAMATH_CALUDE_x_range_l1249_124912


namespace NUMINAMATH_CALUDE_fraction_1800_1809_is_7_30_l1249_124984

/-- The number of states that joined the union from 1800 to 1809 -/
def states_1800_1809 : ℕ := 7

/-- The total number of states considered (first 30 states) -/
def total_states : ℕ := 30

/-- The fraction of states that joined from 1800 to 1809 out of the first 30 states -/
def fraction_1800_1809 : ℚ := states_1800_1809 / total_states

theorem fraction_1800_1809_is_7_30 : fraction_1800_1809 = 7 / 30 := by
  sorry

end NUMINAMATH_CALUDE_fraction_1800_1809_is_7_30_l1249_124984


namespace NUMINAMATH_CALUDE_min_dot_product_l1249_124991

def OA : ℝ × ℝ := (2, 2)
def OB : ℝ × ℝ := (4, 1)

def AP (x : ℝ) : ℝ × ℝ := (x - OA.1, -OA.2)
def BP (x : ℝ) : ℝ × ℝ := (x - OB.1, -OB.2)

def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

theorem min_dot_product :
  ∃ (x : ℝ), ∀ (y : ℝ),
    dot_product (AP x) (BP x) ≤ dot_product (AP y) (BP y) ∧
    x = 3 :=
sorry

end NUMINAMATH_CALUDE_min_dot_product_l1249_124991


namespace NUMINAMATH_CALUDE_cyclist_trip_distance_l1249_124995

/-- Represents a cyclist's trip with consistent speed -/
structure CyclistTrip where
  speed : ℝ
  time : ℝ
  distance : ℝ

/-- The trip satisfies the given conditions -/
def satisfiesConditions (trip : CyclistTrip) : Prop :=
  trip.distance = trip.speed * trip.time ∧
  trip.distance = (trip.speed + 1) * (2/3 * trip.time) ∧
  trip.distance = (trip.speed - 1) * (trip.time + 1)

/-- The theorem stating that the distance is 2 miles -/
theorem cyclist_trip_distance : 
  ∀ (trip : CyclistTrip), satisfiesConditions trip → trip.distance = 2 :=
by
  sorry

end NUMINAMATH_CALUDE_cyclist_trip_distance_l1249_124995


namespace NUMINAMATH_CALUDE_triangle_relations_l1249_124941

/-- Given a triangle ABC with side lengths a, b, c, altitudes h_a, h_b, h_c, 
    inradius r, and exradii r_a, r_b, r_c, the following equations hold -/
theorem triangle_relations (a b c h_a h_b h_c r r_a r_b r_c : ℝ) 
    (ha : a > 0) (hb : b > 0) (hc : c > 0) 
    (hr : r > 0) (hr_a : r_a > 0) (hr_b : r_b > 0) (hr_c : r_c > 0) 
    (hh_a : h_a > 0) (hh_b : h_b > 0) (hh_c : h_c > 0) :
  h_a + h_b + h_c = r * (a + b + c) * (1 / a + 1 / b + 1 / c) ∧
  1 / h_a + 1 / h_b + 1 / h_c = 1 / r ∧
  1 / r = 1 / r_a + 1 / r_b + 1 / r_c ∧
  (h_a + h_b + h_c) * (1 / h_a + 1 / h_b + 1 / h_c) = (a + b + c) * (1 / a + 1 / b + 1 / c) ∧
  (h_a + h_c) / r_a + (h_c + h_a) / r_b + (h_a + h_b) / r_c = 6 :=
by sorry

end NUMINAMATH_CALUDE_triangle_relations_l1249_124941


namespace NUMINAMATH_CALUDE_circumcircle_area_l1249_124944

-- Define the triangle ABC
def Triangle (A B C : ℝ × ℝ) : Prop :=
  -- Right-angled at A
  (B.1 - A.1) * (C.1 - A.1) + (B.2 - A.2) * (C.2 - A.2) = 0 ∧
  -- AB = 6
  (B.1 - A.1)^2 + (B.2 - A.2)^2 = 36 ∧
  -- AC = 8
  (C.1 - A.1)^2 + (C.2 - A.2)^2 = 64

-- Define the circumcircle of the triangle
def Circumcircle (A B C : ℝ × ℝ) : ℝ → Prop :=
  λ r => ∃ (center : ℝ × ℝ),
    (center.1 - A.1)^2 + (center.2 - A.2)^2 = r^2 ∧
    (center.1 - B.1)^2 + (center.2 - B.2)^2 = r^2 ∧
    (center.1 - C.1)^2 + (center.2 - C.2)^2 = r^2

-- Theorem statement
theorem circumcircle_area (A B C : ℝ × ℝ) :
  Triangle A B C →
  ∃ r, Circumcircle A B C r ∧ π * r^2 = 25 * π :=
by sorry

end NUMINAMATH_CALUDE_circumcircle_area_l1249_124944


namespace NUMINAMATH_CALUDE_gake_uses_fewer_boards_l1249_124949

/-- Represents the width of a character in centimeters -/
def char_width : ℕ := 9

/-- Represents the width of a board in centimeters -/
def board_width : ℕ := 5

/-- Calculates the number of boards needed for a given total width -/
def boards_needed (total_width : ℕ) : ℕ :=
  (total_width + board_width - 1) / board_width

/-- Represents Tom's message -/
def tom_message : String := "MMO"

/-- Represents Gake's message -/
def gake_message : String := "2020"

/-- Calculates the total width needed for a message -/
def message_width (msg : String) : ℕ :=
  msg.length * char_width

theorem gake_uses_fewer_boards :
  boards_needed (message_width gake_message) < boards_needed (message_width tom_message) := by
  sorry

#eval boards_needed (message_width tom_message)
#eval boards_needed (message_width gake_message)

end NUMINAMATH_CALUDE_gake_uses_fewer_boards_l1249_124949


namespace NUMINAMATH_CALUDE_incorrect_fraction_equality_l1249_124973

theorem incorrect_fraction_equality (x y : ℝ) (h : x ≠ -y) :
  ¬ ((x - y) / (x + y) = (y - x) / (y + x)) := by
  sorry

end NUMINAMATH_CALUDE_incorrect_fraction_equality_l1249_124973


namespace NUMINAMATH_CALUDE_bridge_length_l1249_124964

/-- The length of the bridge given train crossing times and train length -/
theorem bridge_length
  (train_length : ℝ)
  (bridge_crossing_time : ℝ)
  (lamppost_crossing_time : ℝ)
  (h1 : train_length = 600)
  (h2 : bridge_crossing_time = 70)
  (h3 : lamppost_crossing_time = 20) :
  ∃ (bridge_length : ℝ), bridge_length = 1500 := by
  sorry


end NUMINAMATH_CALUDE_bridge_length_l1249_124964


namespace NUMINAMATH_CALUDE_forty_bees_honey_l1249_124957

/-- The amount of honey (in grams) produced by one honey bee in 40 days -/
def honey_per_bee : ℕ := 1

/-- The number of honey bees -/
def num_bees : ℕ := 40

/-- The amount of honey (in grams) produced by a group of honey bees in 40 days -/
def total_honey (bees : ℕ) : ℕ := bees * honey_per_bee

/-- Theorem stating that 40 honey bees produce 40 grams of honey in 40 days -/
theorem forty_bees_honey : total_honey num_bees = 40 := by
  sorry

end NUMINAMATH_CALUDE_forty_bees_honey_l1249_124957


namespace NUMINAMATH_CALUDE_four_digit_divisible_by_nine_count_l1249_124915

theorem four_digit_divisible_by_nine_count : 
  (Finset.filter (fun n => n % 9 = 0) (Finset.range 9000)).card = 1000 := by
  sorry

end NUMINAMATH_CALUDE_four_digit_divisible_by_nine_count_l1249_124915


namespace NUMINAMATH_CALUDE_f_increasing_decreasing_l1249_124987

noncomputable def f (x : ℝ) : ℝ := x^2 - Real.log x

theorem f_increasing_decreasing :
  let f : ℝ → ℝ := λ x => x^2 - Real.log x
  (∀ x, x > 0 → f x ∈ Set.univ) ∧
  (∀ x y, x > Real.sqrt 2 / 2 → y > Real.sqrt 2 / 2 → x < y → f x < f y) ∧
  (∀ x y, x > 0 → y > 0 → x < Real.sqrt 2 / 2 → y < Real.sqrt 2 / 2 → x < y → f x > f y) :=
by sorry

end NUMINAMATH_CALUDE_f_increasing_decreasing_l1249_124987


namespace NUMINAMATH_CALUDE_area_between_curves_l1249_124979

theorem area_between_curves : 
  let f (x : ℝ) := Real.sqrt x
  let g (x : ℝ) := x^2
  ∫ x in (0 : ℝ)..1, (f x - g x) = (1 : ℝ) / 3 := by sorry

end NUMINAMATH_CALUDE_area_between_curves_l1249_124979


namespace NUMINAMATH_CALUDE_solution_set_f_less_than_one_range_of_a_l1249_124998

-- Define the function f(x)
def f (x : ℝ) : ℝ := |2*x + 1| - |x - 1|

-- Theorem for the solution set of f(x) < 1
theorem solution_set_f_less_than_one :
  {x : ℝ | f x < 1} = {x : ℝ | -3 < x ∧ x < 1/3} := by sorry

-- Theorem for the range of a
theorem range_of_a (a : ℝ) :
  (∃ x : ℝ, f x ≤ a - a^2/2 + 5/2) → -2 ≤ a ∧ a ≤ 4 := by sorry

end NUMINAMATH_CALUDE_solution_set_f_less_than_one_range_of_a_l1249_124998


namespace NUMINAMATH_CALUDE_mixtape_song_length_l1249_124911

/-- Represents a mixtape with two sides -/
structure Mixtape where
  side1_songs : ℕ
  side2_songs : ℕ
  total_length : ℕ

/-- Theorem: Given a mixtape with 6 songs on side 1, 4 songs on side 2, 
    and a total length of 40 minutes, if all songs have the same length, 
    then each song is 4 minutes long. -/
theorem mixtape_song_length (m : Mixtape) 
    (h1 : m.side1_songs = 6)
    (h2 : m.side2_songs = 4)
    (h3 : m.total_length = 40) :
    m.total_length / (m.side1_songs + m.side2_songs) = 4 := by
  sorry

end NUMINAMATH_CALUDE_mixtape_song_length_l1249_124911


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_l1249_124976

theorem imaginary_part_of_z (z : ℂ) (h : (1 + Complex.I) * z = Complex.I) : 
  z.im = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_l1249_124976


namespace NUMINAMATH_CALUDE_sine_cosine_difference_l1249_124988

theorem sine_cosine_difference (θ₁ θ₂ : Real) :
  Real.sin (37.5 * π / 180) * Real.cos (7.5 * π / 180) -
  Real.cos (37.5 * π / 180) * Real.sin (7.5 * π / 180) = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_sine_cosine_difference_l1249_124988


namespace NUMINAMATH_CALUDE_daniel_paid_six_more_l1249_124918

/-- A pizza sharing scenario between Carl and Daniel -/
structure PizzaScenario where
  total_slices : ℕ
  plain_cost : ℚ
  truffle_cost : ℚ
  daniel_truffle_slices : ℕ
  daniel_plain_slices : ℕ
  carl_plain_slices : ℕ

/-- Calculate the payment difference between Daniel and Carl -/
def payment_difference (scenario : PizzaScenario) : ℚ :=
  let total_cost := scenario.plain_cost + scenario.truffle_cost
  let cost_per_slice := total_cost / scenario.total_slices
  let daniel_payment := (scenario.daniel_truffle_slices + scenario.daniel_plain_slices) * cost_per_slice
  let carl_payment := scenario.carl_plain_slices * cost_per_slice
  daniel_payment - carl_payment

/-- The specific pizza scenario described in the problem -/
def pizza : PizzaScenario :=
  { total_slices := 10
  , plain_cost := 10
  , truffle_cost := 5
  , daniel_truffle_slices := 5
  , daniel_plain_slices := 2
  , carl_plain_slices := 3 }

/-- Theorem stating that Daniel paid $6 more than Carl -/
theorem daniel_paid_six_more : payment_difference pizza = 6 := by
  sorry

end NUMINAMATH_CALUDE_daniel_paid_six_more_l1249_124918


namespace NUMINAMATH_CALUDE_simplified_sqrt_expression_l1249_124963

theorem simplified_sqrt_expression (x : ℝ) : 
  Real.sqrt (9 * x^4 + 3 * x^2) = Real.sqrt 3 * |x| * Real.sqrt (3 * x^2 + 1) := by
  sorry

end NUMINAMATH_CALUDE_simplified_sqrt_expression_l1249_124963


namespace NUMINAMATH_CALUDE_even_power_iff_even_l1249_124907

theorem even_power_iff_even (n : ℕ+) : Even (n^n.val) ↔ Even n.val := by sorry

end NUMINAMATH_CALUDE_even_power_iff_even_l1249_124907


namespace NUMINAMATH_CALUDE_car_tank_size_l1249_124952

/-- Calculates the size of a car's gas tank given the advertised mileage, actual miles driven, and the difference between advertised and actual mileage. -/
theorem car_tank_size 
  (advertised_mileage : ℝ) 
  (miles_driven : ℝ) 
  (mileage_difference : ℝ) : 
  advertised_mileage = 35 →
  miles_driven = 372 →
  mileage_difference = 4 →
  miles_driven / (advertised_mileage - mileage_difference) = 12 := by
    sorry

#check car_tank_size

end NUMINAMATH_CALUDE_car_tank_size_l1249_124952


namespace NUMINAMATH_CALUDE_sqrt_6_simplest_l1249_124945

def is_simplest_sqrt (x : ℝ) : Prop :=
  ∀ y : ℝ, y > 0 → x = Real.sqrt y → ¬∃ a b : ℝ, a > 0 ∧ b > 1 ∧ y = a * b^2

theorem sqrt_6_simplest :
  is_simplest_sqrt (Real.sqrt 6) ∧
  ¬is_simplest_sqrt (Real.sqrt 8) ∧
  ¬is_simplest_sqrt (Real.sqrt (1/3)) ∧
  ¬is_simplest_sqrt (Real.sqrt 4) :=
sorry

end NUMINAMATH_CALUDE_sqrt_6_simplest_l1249_124945


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_ratio_l1249_124968

theorem arithmetic_sequence_sum_ratio (a : ℕ → ℝ) (S : ℕ → ℝ) : 
  (∀ n, S n = (n / 2) * (2 * a 1 + (n - 1) * (a 2 - a 1))) →  -- Definition of S_n
  (∀ n, a (n + 1) - a n = a 2 - a 1) →  -- Arithmetic sequence condition
  S 3 / S 6 = 1 / 3 →
  S 6 / S 12 = 3 / 10 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_ratio_l1249_124968


namespace NUMINAMATH_CALUDE_steve_earnings_l1249_124950

/-- Calculates an author's earnings from book sales after agent's commission --/
def author_earnings (copies_sold : ℕ) (price_per_copy : ℚ) (agent_commission_rate : ℚ) : ℚ :=
  let total_revenue := copies_sold * price_per_copy
  let agent_commission := total_revenue * agent_commission_rate
  total_revenue - agent_commission

/-- Proves that given the specified conditions, the author's earnings are $1,800,000 --/
theorem steve_earnings :
  author_earnings 1000000 2 (1/10) = 1800000 := by
  sorry

end NUMINAMATH_CALUDE_steve_earnings_l1249_124950


namespace NUMINAMATH_CALUDE_max_ratio_is_half_l1249_124937

/-- A hyperbola with equation x^2 - y^2 = 1 -/
def Hyperbola : Set (ℝ × ℝ) :=
  {p | p.1^2 - p.2^2 = 1}

/-- The right focus of the hyperbola -/
def RightFocus : ℝ × ℝ := sorry

/-- The right directrix of the hyperbola -/
def RightDirectrix : Set (ℝ × ℝ) := sorry

/-- The right branch of the hyperbola -/
def RightBranch : Set (ℝ × ℝ) := sorry

/-- The projection of a point onto the right directrix -/
def ProjectOntoDirectrix (p : ℝ × ℝ) : ℝ × ℝ := sorry

/-- The distance between two points -/
def Distance (p q : ℝ × ℝ) : ℝ := sorry

/-- The midpoint of two points -/
def Midpoint (p q : ℝ × ℝ) : ℝ × ℝ := sorry

/-- Statement: The maximum value of |MN|/|AB| is 1/2 -/
theorem max_ratio_is_half :
  ∀ A B : ℝ × ℝ,
  A ∈ RightBranch →
  B ∈ RightBranch →
  Distance A RightFocus * Distance B RightFocus = 0 →  -- This represents AF ⟂ BF
  let M := Midpoint A B
  let N := ProjectOntoDirectrix M
  Distance M N / Distance A B ≤ 1/2 := by
sorry

end NUMINAMATH_CALUDE_max_ratio_is_half_l1249_124937


namespace NUMINAMATH_CALUDE_toy_piles_total_l1249_124951

theorem toy_piles_total (small_pile large_pile : ℕ) : 
  large_pile = 2 * small_pile → 
  large_pile = 80 → 
  small_pile + large_pile = 120 :=
by sorry

end NUMINAMATH_CALUDE_toy_piles_total_l1249_124951


namespace NUMINAMATH_CALUDE_complex_exp_thirteen_pi_i_over_three_l1249_124940

theorem complex_exp_thirteen_pi_i_over_three :
  Complex.exp (13 * π * Complex.I / 3) = (1 / 2 : ℂ) + Complex.I * (Real.sqrt 3 / 2) := by
  sorry

end NUMINAMATH_CALUDE_complex_exp_thirteen_pi_i_over_three_l1249_124940


namespace NUMINAMATH_CALUDE_token_game_ends_in_37_rounds_l1249_124931

/-- Represents a player in the token game -/
inductive Player : Type
  | A
  | B
  | C

/-- Represents the state of the game at any point -/
structure GameState :=
  (tokens : Player → Nat)
  (round : Nat)

/-- Determines if a player's tokens are divisible by 5 -/
def isDivisibleByFive (n : Nat) : Bool :=
  n % 5 = 0

/-- Determines the player with the most tokens -/
def playerWithMostTokens (state : GameState) : Player :=
  sorry

/-- Applies the rules of a single round to the game state -/
def applyRound (state : GameState) : GameState :=
  sorry

/-- Checks if the game has ended (i.e., a player has run out of tokens) -/
def isGameOver (state : GameState) : Bool :=
  sorry

/-- The initial state of the game -/
def initialState : GameState :=
  { tokens := λ p => match p with
    | Player.A => 17
    | Player.B => 15
    | Player.C => 14,
    round := 0 }

/-- The final state of the game -/
def finalState : GameState :=
  sorry

theorem token_game_ends_in_37_rounds :
  finalState.round = 37 ∧ isGameOver finalState :=
  sorry

end NUMINAMATH_CALUDE_token_game_ends_in_37_rounds_l1249_124931


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l1249_124967

def A : Set ℤ := {0, 1, 2, 3, 4, 5}
def B : Set ℤ := {-1, 0, 1, 6}

theorem intersection_of_A_and_B : A ∩ B = {0, 1} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l1249_124967


namespace NUMINAMATH_CALUDE_subset_implies_m_range_l1249_124934

theorem subset_implies_m_range (m : ℝ) : 
  let A := Set.Iic m
  let B := Set.Ioo 1 2
  B ⊆ A → m ∈ Set.Ici 2 := by
sorry

end NUMINAMATH_CALUDE_subset_implies_m_range_l1249_124934


namespace NUMINAMATH_CALUDE_parabola_intersection_l1249_124921

/-- The points of intersection between the parabolas y = 3x^2 - 4x + 2 and y = x^3 - 2x^2 + 5x - 1 -/
theorem parabola_intersection :
  let f (x : ℝ) := 3 * x^2 - 4 * x + 2
  let g (x : ℝ) := x^3 - 2 * x^2 + 5 * x - 1
  ∀ x y : ℝ, f x = g x ∧ y = f x ↔ (x = 1 ∧ y = 1) ∨ (x = 3 ∧ y = 17) :=
by sorry

end NUMINAMATH_CALUDE_parabola_intersection_l1249_124921


namespace NUMINAMATH_CALUDE_seafood_price_proof_l1249_124922

/-- The regular price of seafood given the sale price and discount -/
def regular_price (sale_price : ℚ) (discount_percent : ℚ) : ℚ :=
  sale_price / (1 - discount_percent)

/-- The price for a given weight of seafood at the regular price -/
def price_for_weight (price_per_unit : ℚ) (weight : ℚ) : ℚ :=
  price_per_unit * weight

theorem seafood_price_proof :
  let sale_price_per_pack : ℚ := 4
  let pack_weight : ℚ := 3/4
  let discount_percent : ℚ := 3/4
  let target_weight : ℚ := 3/2

  let regular_price_per_pack := regular_price sale_price_per_pack discount_percent
  let regular_price_per_pound := regular_price_per_pack / pack_weight
  
  price_for_weight regular_price_per_pound target_weight = 32 := by
  sorry

end NUMINAMATH_CALUDE_seafood_price_proof_l1249_124922


namespace NUMINAMATH_CALUDE_probability_x_squared_gt_one_l1249_124936

-- Define the interval
def interval : Set ℝ := Set.Icc (-2) 2

-- Define the event (x^2 > 1)
def event (x : ℝ) : Prop := x^2 > 1

-- Define the measure of the interval
def intervalMeasure : ℝ := 4

-- Define the measure of the event within the interval
def eventMeasure : ℝ := 2

-- State the theorem
theorem probability_x_squared_gt_one :
  (eventMeasure / intervalMeasure : ℝ) = 1/2 := by sorry

end NUMINAMATH_CALUDE_probability_x_squared_gt_one_l1249_124936


namespace NUMINAMATH_CALUDE_square_rectangle_area_relation_l1249_124960

theorem square_rectangle_area_relation :
  ∀ x : ℝ,
  let square_side := x - 3
  let rect_length := x - 2
  let rect_width := x + 5
  let square_area := square_side ^ 2
  let rect_area := rect_length * rect_width
  rect_area = 3 * square_area →
  ∃ x₁ x₂ : ℝ, 
    (x₁ + x₂ = 21/2) ∧ 
    (∀ y : ℝ, rect_area = 3 * square_area → y = x₁ ∨ y = x₂) :=
by
  sorry

#check square_rectangle_area_relation

end NUMINAMATH_CALUDE_square_rectangle_area_relation_l1249_124960


namespace NUMINAMATH_CALUDE_max_cube_hemisphere_ratio_l1249_124997

/-- The maximum ratio of the volume of a cube inscribed in a hemisphere to the volume of the hemisphere -/
theorem max_cube_hemisphere_ratio : 
  let r := Real.sqrt 6 / (3 * Real.pi)
  ∃ (R a : ℝ), R > 0 ∧ a > 0 ∧
  (a^2 + (Real.sqrt 2 * a / 2)^2 = R^2) ∧
  (∀ (b : ℝ), b > 0 → b^2 + (Real.sqrt 2 * b / 2)^2 ≤ R^2 → 
    b^3 / ((2/3) * Real.pi * R^3) ≤ r) ∧
  a^3 / ((2/3) * Real.pi * R^3) = r :=
sorry

end NUMINAMATH_CALUDE_max_cube_hemisphere_ratio_l1249_124997


namespace NUMINAMATH_CALUDE_grunters_win_probability_l1249_124958

/-- The probability of winning a single game for the Grunters -/
def p : ℚ := 3/5

/-- The number of games played -/
def n : ℕ := 5

/-- The probability of winning all games -/
def win_all : ℚ := p^n

theorem grunters_win_probability : win_all = 243/3125 := by
  sorry

end NUMINAMATH_CALUDE_grunters_win_probability_l1249_124958


namespace NUMINAMATH_CALUDE_orange_weight_after_water_loss_orange_weight_problem_l1249_124939

/-- Calculates the new weight of oranges after water loss -/
theorem orange_weight_after_water_loss 
  (initial_weight : ℝ) 
  (initial_water_percentage : ℝ) 
  (evaporation_loss_percentage : ℝ) 
  (skin_loss_percentage : ℝ) : ℝ :=
  let initial_water_weight := initial_weight * initial_water_percentage
  let dry_weight := initial_weight - initial_water_weight
  let evaporation_loss := initial_water_weight * evaporation_loss_percentage
  let remaining_water_after_evaporation := initial_water_weight - evaporation_loss
  let skin_loss := remaining_water_after_evaporation * skin_loss_percentage
  let total_water_loss := evaporation_loss + skin_loss
  let new_water_weight := initial_water_weight - total_water_loss
  new_water_weight + dry_weight

/-- The new weight of oranges after water loss is approximately 4.67225 kg -/
theorem orange_weight_problem : 
  ∃ ε > 0, |orange_weight_after_water_loss 5 0.95 0.05 0.02 - 4.67225| < ε :=
sorry

end NUMINAMATH_CALUDE_orange_weight_after_water_loss_orange_weight_problem_l1249_124939


namespace NUMINAMATH_CALUDE_free_time_correct_l1249_124910

/-- The time required to free Hannah's younger son -/
def free_time (total_strands : ℕ) (hannah_rate : ℕ) (son_rate : ℕ) : ℕ :=
  (total_strands + (hannah_rate + son_rate) - 1) / (hannah_rate + son_rate)

theorem free_time_correct : free_time 78 5 2 = 12 := by
  sorry

end NUMINAMATH_CALUDE_free_time_correct_l1249_124910


namespace NUMINAMATH_CALUDE_lukes_remaining_money_l1249_124994

/-- Converts an octal number to decimal --/
def octal_to_decimal (n : ℕ) : ℕ := sorry

/-- Calculates the remaining money after buying a ticket --/
def remaining_money (savings : ℕ) (ticket_cost : ℕ) : ℕ :=
  octal_to_decimal savings - ticket_cost

theorem lukes_remaining_money :
  remaining_money 0o5555 1200 = 1725 := by sorry

end NUMINAMATH_CALUDE_lukes_remaining_money_l1249_124994


namespace NUMINAMATH_CALUDE_sand_calculation_l1249_124906

def remaining_sand (initial : ℝ) (lost : ℝ) : ℝ := initial - lost

theorem sand_calculation (initial : ℝ) (lost : ℝ) :
  remaining_sand initial lost = initial - lost :=
by sorry

end NUMINAMATH_CALUDE_sand_calculation_l1249_124906


namespace NUMINAMATH_CALUDE_carol_rectangle_width_l1249_124954

/-- Given two rectangles with equal area, where one has a length of 5 inches
    and the other has dimensions of 3 inches by 40 inches,
    prove that the width of the first rectangle is 24 inches. -/
theorem carol_rectangle_width
  (length_carol : ℝ)
  (width_carol : ℝ)
  (length_jordan : ℝ)
  (width_jordan : ℝ)
  (h1 : length_carol = 5)
  (h2 : length_jordan = 3)
  (h3 : width_jordan = 40)
  (h4 : length_carol * width_carol = length_jordan * width_jordan) :
  width_carol = 24 :=
by sorry

end NUMINAMATH_CALUDE_carol_rectangle_width_l1249_124954


namespace NUMINAMATH_CALUDE_min_value_expression_l1249_124990

theorem min_value_expression (x y z : ℝ) (hx : x > 1) (hy : y > 1) (hz : z > 1) :
  (x^2 / (y - 1) + y^2 / (z - 1) + z^2 / (x - 1)) ≥ 12 :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_l1249_124990


namespace NUMINAMATH_CALUDE_right_triangle_angles_l1249_124909

theorem right_triangle_angles (a b c R r : ℝ) (h_right : a^2 + b^2 = c^2)
  (h_R : R = c / 2) (h_r : r = (a + b - c) / 2) (h_ratio : R / r = Real.sqrt 3 + 1) :
  ∃ (α β : ℝ), α + β = Real.pi / 2 ∧ 
  (α = Real.pi / 6 ∧ β = Real.pi / 3) ∨ (α = Real.pi / 3 ∧ β = Real.pi / 6) :=
sorry

end NUMINAMATH_CALUDE_right_triangle_angles_l1249_124909


namespace NUMINAMATH_CALUDE_desk_purchase_optimization_l1249_124992

/-- The total cost function for shipping and storage fees -/
def f (x : ℕ) : ℚ := 144 / x + 4 * x

/-- The number of desks to be purchased -/
def total_desks : ℕ := 36

/-- The value of each desk -/
def desk_value : ℕ := 20

/-- The shipping fee per batch -/
def shipping_fee : ℕ := 4

/-- Available funds for shipping and storage -/
def available_funds : ℕ := 48

theorem desk_purchase_optimization :
  /- 1. The total cost function is correct -/
  (∀ x : ℕ, x > 0 → f x = 144 / x + 4 * x) ∧
  /- 2. There exists an integer x between 4 and 9 inclusive that satisfies the budget -/
  (∃ x : ℕ, 4 ≤ x ∧ x ≤ 9 ∧ f x ≤ available_funds) ∧
  /- 3. The minimum value of f(x) occurs when x = 6 -/
  (∀ x : ℕ, x > 0 → f 6 ≤ f x) :=
by sorry

end NUMINAMATH_CALUDE_desk_purchase_optimization_l1249_124992


namespace NUMINAMATH_CALUDE_value_of_a_l1249_124924

-- Define the function f
def f (x : ℝ) : ℝ := (x + 1)^2 - 2*(x + 1)

-- State the theorem
theorem value_of_a (a : ℝ) (h : f a = 3) : a = 2 ∨ a = -2 := by
  sorry

end NUMINAMATH_CALUDE_value_of_a_l1249_124924


namespace NUMINAMATH_CALUDE_alligators_hiding_l1249_124917

/-- Given a zoo cage with alligators, prove the number of hiding alligators -/
theorem alligators_hiding (total_alligators : ℕ) (not_hiding : ℕ) 
  (h1 : total_alligators = 75)
  (h2 : not_hiding = 56) :
  total_alligators - not_hiding = 19 := by
  sorry

#check alligators_hiding

end NUMINAMATH_CALUDE_alligators_hiding_l1249_124917


namespace NUMINAMATH_CALUDE_base7_to_base10_conversion_l1249_124961

/-- Converts a base 7 number to base 10 -/
def base7ToBase10 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (7 ^ i)) 0

/-- The base 7 representation of the number -/
def base7Number : List Nat := [6, 3, 4, 5, 2]

theorem base7_to_base10_conversion :
  base7ToBase10 base7Number = 6740 := by
  sorry

end NUMINAMATH_CALUDE_base7_to_base10_conversion_l1249_124961


namespace NUMINAMATH_CALUDE_binomial_coefficient_seven_four_l1249_124969

theorem binomial_coefficient_seven_four : (7 : ℕ).choose 4 = 35 := by
  sorry

end NUMINAMATH_CALUDE_binomial_coefficient_seven_four_l1249_124969


namespace NUMINAMATH_CALUDE_one_absent_one_present_probability_l1249_124905

theorem one_absent_one_present_probability 
  (p_absent : ℝ) 
  (h_absent : p_absent = 1 / 20) : 
  let p_present := 1 - p_absent
  2 * (p_absent * p_present) = 19 / 200 := by
sorry

end NUMINAMATH_CALUDE_one_absent_one_present_probability_l1249_124905


namespace NUMINAMATH_CALUDE_no_simultaneously_safe_numbers_l1249_124985

def is_p_safe (n p : ℕ) : Prop :=
  ∀ k : ℤ, |n - k * p| > 3

def simultaneously_safe (n : ℕ) : Prop :=
  is_p_safe n 5 ∧ is_p_safe n 9 ∧ is_p_safe n 11

theorem no_simultaneously_safe_numbers : 
  ¬∃ n : ℕ, n > 0 ∧ n ≤ 5000 ∧ simultaneously_safe n :=
sorry

end NUMINAMATH_CALUDE_no_simultaneously_safe_numbers_l1249_124985


namespace NUMINAMATH_CALUDE_centroid_projections_sum_l1249_124900

/-- Given a triangle XYZ with sides of length 4, 3, and 5, 
    this theorem states that the sum of the distances from 
    the centroid to each side of the triangle is 47/15. -/
theorem centroid_projections_sum (X Y Z G : ℝ × ℝ) : 
  let d := λ p q : ℝ × ℝ => Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)
  (d X Y = 4) → (d X Z = 3) → (d Y Z = 5) →
  (G.1 = (X.1 + Y.1 + Z.1) / 3) → (G.2 = (X.2 + Y.2 + Z.2) / 3) →
  let dist_point_to_line := λ p a b : ℝ × ℝ => 
    |((b.2 - a.2) * p.1 - (b.1 - a.1) * p.2 + b.1 * a.2 - b.2 * a.1) / d a b|
  (dist_point_to_line G Y Z + dist_point_to_line G X Z + dist_point_to_line G X Y = 47/15) := by
sorry

end NUMINAMATH_CALUDE_centroid_projections_sum_l1249_124900


namespace NUMINAMATH_CALUDE_expansion_coefficient_l1249_124938

theorem expansion_coefficient (n : ℕ) : 
  ((-2:ℤ)^n + n * (-2:ℤ)^(n-1) = -128) → n = 6 := by
sorry

end NUMINAMATH_CALUDE_expansion_coefficient_l1249_124938


namespace NUMINAMATH_CALUDE_basketball_card_cost_l1249_124913

/-- The cost of one deck of basketball cards -/
def cost_of_deck (mary_total rose_total shoe_cost : ℕ) : ℕ :=
  (rose_total - shoe_cost) / 2

theorem basketball_card_cost :
  ∀ (mary_total rose_total shoe_cost : ℕ),
    mary_total = rose_total →
    mary_total = 200 →
    shoe_cost = 150 →
    cost_of_deck mary_total rose_total shoe_cost = 25 :=
by
  sorry

end NUMINAMATH_CALUDE_basketball_card_cost_l1249_124913


namespace NUMINAMATH_CALUDE_complete_graph_edges_six_vertices_l1249_124986

theorem complete_graph_edges_six_vertices :
  let n : ℕ := 6
  let E : ℕ := n * (n - 1) / 2
  E = 15 := by sorry

end NUMINAMATH_CALUDE_complete_graph_edges_six_vertices_l1249_124986


namespace NUMINAMATH_CALUDE_series_convergence_l1249_124993

theorem series_convergence (a : ℕ → ℝ) :
  (∃ S : ℝ, HasSum (λ n : ℕ => a n + 2 * a (n + 1)) S) →
  (∃ T : ℝ, HasSum a T) :=
by sorry

end NUMINAMATH_CALUDE_series_convergence_l1249_124993


namespace NUMINAMATH_CALUDE_percentage_runs_by_running_l1249_124982

def total_runs : ℕ := 150
def boundaries : ℕ := 6
def sixes : ℕ := 4
def no_balls : ℕ := 8
def wide_balls : ℕ := 5
def leg_byes : ℕ := 2

def runs_from_boundaries : ℕ := boundaries * 4
def runs_from_sixes : ℕ := sixes * 6
def runs_not_from_bat : ℕ := no_balls + wide_balls + leg_byes

def runs_by_running : ℕ := total_runs - runs_not_from_bat - (runs_from_boundaries + runs_from_sixes)

theorem percentage_runs_by_running :
  (runs_by_running : ℚ) / total_runs * 100 = 58 := by
  sorry

end NUMINAMATH_CALUDE_percentage_runs_by_running_l1249_124982


namespace NUMINAMATH_CALUDE_average_of_remaining_digits_l1249_124983

theorem average_of_remaining_digits
  (total_count : Nat)
  (total_avg : ℚ)
  (subset_count : Nat)
  (subset_avg : ℚ)
  (h_total_count : total_count = 10)
  (h_total_avg : total_avg = 80)
  (h_subset_count : subset_count = 6)
  (h_subset_avg : subset_avg = 58)
  : (total_count * total_avg - subset_count * subset_avg) / (total_count - subset_count) = 113 := by
  sorry

end NUMINAMATH_CALUDE_average_of_remaining_digits_l1249_124983


namespace NUMINAMATH_CALUDE_hillshire_population_l1249_124946

theorem hillshire_population (num_cities : ℕ) (avg_lower : ℕ) (avg_upper : ℕ) :
  num_cities = 25 →
  avg_lower = 5000 →
  avg_upper = 5500 →
  (num_cities : ℝ) * ((avg_lower : ℝ) + (avg_upper : ℝ)) / 2 = 131250 :=
by sorry

end NUMINAMATH_CALUDE_hillshire_population_l1249_124946


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l1249_124926

/-- The eccentricity of a hyperbola with equation x²/a² - y²/b² = 1 is √(a² + b²) / a -/
theorem hyperbola_eccentricity (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  let e := Real.sqrt (a^2 + b^2) / a
  (∀ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1) → e = Real.sqrt 7 / 2 :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l1249_124926


namespace NUMINAMATH_CALUDE_runners_in_picture_probability_l1249_124908

/-- Represents a runner on a circular track -/
structure Runner where
  lapTime : ℝ  -- Time to complete one lap in seconds
  direction : Bool  -- true for counterclockwise, false for clockwise

/-- Calculates the probability of both runners being in the picture -/
def probability_both_in_picture (runner1 runner2 : Runner) (pictureTime : ℝ) : ℚ :=
  sorry

/-- Theorem stating the probability of both runners being in the picture -/
theorem runners_in_picture_probability 
  (runner1 : Runner) 
  (runner2 : Runner) 
  (pictureTime : ℝ) 
  (h1 : runner1.lapTime = 100)
  (h2 : runner2.lapTime = 75)
  (h3 : runner1.direction = true)
  (h4 : runner2.direction = false)
  (h5 : 720 ≤ pictureTime ∧ pictureTime ≤ 780) :
  probability_both_in_picture runner1 runner2 pictureTime = 111 / 200 :=
by sorry

end NUMINAMATH_CALUDE_runners_in_picture_probability_l1249_124908


namespace NUMINAMATH_CALUDE_find_x_in_set_l1249_124975

theorem find_x_in_set (s : Finset ℝ) (x : ℝ) : 
  s = {8, 14, 20, 7, x, 16} →
  (Finset.sum s id) / (Finset.card s : ℝ) = 12 →
  x = 7 := by
sorry

end NUMINAMATH_CALUDE_find_x_in_set_l1249_124975


namespace NUMINAMATH_CALUDE_division_simplification_l1249_124959

theorem division_simplification : 240 / (12 + 12 * 2 - 3) = 240 / 33 := by
  sorry

end NUMINAMATH_CALUDE_division_simplification_l1249_124959


namespace NUMINAMATH_CALUDE_problem_solution_l1249_124989

-- Define the complex square root function
noncomputable def complexSqrt (x : ℂ) : ℂ := sorry

-- Define the statements
def statement_I : Prop :=
  complexSqrt (-4) * complexSqrt (-16) = complexSqrt ((-4) * (-16))

def statement_II : Prop :=
  complexSqrt ((-4) * (-16)) = Real.sqrt 64

def statement_III : Prop :=
  Real.sqrt 64 = 8

-- Theorem to prove
theorem problem_solution :
  (¬statement_I ∧ statement_II ∧ statement_III) := by sorry

end NUMINAMATH_CALUDE_problem_solution_l1249_124989
