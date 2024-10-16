import Mathlib

namespace NUMINAMATH_CALUDE_multitool_comparison_l54_5498

-- Define the contents of each multitool
def walmart_tools : ℕ := 2 + 4 + 1 + 1 + 1
def target_knives : ℕ := 4 * 3
def target_tools : ℕ := 3 + target_knives + 2 + 1 + 1 + 2

-- Theorem to prove the difference in tools and the ratio
theorem multitool_comparison :
  (target_tools - walmart_tools = 12) ∧
  (target_tools / walmart_tools = 7 / 3) := by
  sorry

#eval walmart_tools
#eval target_tools

end NUMINAMATH_CALUDE_multitool_comparison_l54_5498


namespace NUMINAMATH_CALUDE_ellipse_and_line_problem_l54_5455

/-- An ellipse with center at origin and foci on x-axis -/
structure Ellipse where
  a : ℝ
  b : ℝ
  c : ℝ
  h_a_pos : 0 < a
  h_b_pos : 0 < b
  h_c_pos : 0 < c
  h_a_b : b < a
  h_c_eq : c^2 = a^2 - b^2

/-- A line intersecting the ellipse -/
structure IntersectingLine where
  m : ℝ  -- slope
  k : ℝ  -- y-intercept

/-- The problem statement -/
theorem ellipse_and_line_problem 
  (E : Ellipse) 
  (l : IntersectingLine) 
  (h_arithmetic : E.c + (E.a^2 / E.c + E.c) = 4 * E.c) 
  (h_midpoint : -2 = (l.m * -2 + l.k + -2) / 2 ∧ 1 = (l.m * -2 + l.k + 1) / 2) 
  (h_length : (4 * Real.sqrt 3)^2 = 2 * ((l.m * -2 + l.k + 2)^2 + 9)) :
  l.m = 1 ∧ l.k = 3 ∧ E.a^2 = 24 ∧ E.b^2 = 12 := by sorry

end NUMINAMATH_CALUDE_ellipse_and_line_problem_l54_5455


namespace NUMINAMATH_CALUDE_binomial_eight_choose_two_l54_5485

theorem binomial_eight_choose_two : (8 : ℕ).choose 2 = 28 := by sorry

end NUMINAMATH_CALUDE_binomial_eight_choose_two_l54_5485


namespace NUMINAMATH_CALUDE_ali_baba_max_camels_l54_5426

/-- Represents the problem of maximizing the number of camels Ali Baba can buy --/
theorem ali_baba_max_camels :
  let gold_capacity : ℝ := 200
  let diamond_capacity : ℝ := 40
  let max_weight : ℝ := 100
  let gold_camel_rate : ℝ := 20
  let diamond_camel_rate : ℝ := 60
  
  ∃ (gold_weight diamond_weight : ℝ),
    gold_weight ≥ 0 ∧
    diamond_weight ≥ 0 ∧
    gold_weight + diamond_weight ≤ max_weight ∧
    gold_weight / gold_capacity + diamond_weight / diamond_capacity ≤ 1 ∧
    ∀ (g d : ℝ),
      g ≥ 0 →
      d ≥ 0 →
      g + d ≤ max_weight →
      g / gold_capacity + d / diamond_capacity ≤ 1 →
      gold_camel_rate * g + diamond_camel_rate * d ≤ gold_camel_rate * gold_weight + diamond_camel_rate * diamond_weight ∧
    gold_camel_rate * gold_weight + diamond_camel_rate * diamond_weight = 3000 := by
  sorry


end NUMINAMATH_CALUDE_ali_baba_max_camels_l54_5426


namespace NUMINAMATH_CALUDE_problem_statement_l54_5438

theorem problem_statement (x y z : ℚ) (hx : x = 4/3) (hy : y = 3/4) (hz : z = 3/2) :
  (1/2) * x^6 * y^7 * z^4 = 243/128 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l54_5438


namespace NUMINAMATH_CALUDE_no_integer_roots_l54_5430

/-- A polynomial with integer coefficients -/
def IntPolynomial := ℕ → ℤ

/-- Evaluate a polynomial at a given integer -/
def eval (p : IntPolynomial) (x : ℤ) : ℤ :=
  sorry

theorem no_integer_roots (P : IntPolynomial) 
  (h2020 : eval P 2020 = 2021) 
  (h2021 : eval P 2021 = 2021) : 
  ∀ x : ℤ, eval P x ≠ 0 := by
  sorry

end NUMINAMATH_CALUDE_no_integer_roots_l54_5430


namespace NUMINAMATH_CALUDE_greatest_of_three_consecutive_integers_l54_5463

theorem greatest_of_three_consecutive_integers (x : ℤ) 
  (h : x + (x + 1) + (x + 2) = 36) : 
  max x (max (x + 1) (x + 2)) = 13 := by
sorry

end NUMINAMATH_CALUDE_greatest_of_three_consecutive_integers_l54_5463


namespace NUMINAMATH_CALUDE_fib_gcd_consecutive_fib_gcd_identity_l54_5472

def fib : ℕ → ℕ
  | 0 => 1
  | 1 => 1
  | (n + 2) => fib n + fib (n + 1)

theorem fib_gcd_consecutive (n : ℕ) : Nat.gcd (fib n) (fib (n + 1)) = 1 := by
  sorry

theorem fib_gcd_identity (m n : ℕ) 
  (h : ∀ a b : ℕ, fib (a + b) = fib b * fib (a + 1) + fib (b - 1) * fib a) : 
  fib (Nat.gcd m n) = Nat.gcd (fib m) (fib n) := by
  sorry

end NUMINAMATH_CALUDE_fib_gcd_consecutive_fib_gcd_identity_l54_5472


namespace NUMINAMATH_CALUDE_algae_growth_time_l54_5428

/-- The growth factor of the algae population every 5 hours -/
def growth_factor : ℕ := 3

/-- The initial number of algae cells -/
def initial_cells : ℕ := 200

/-- The target number of algae cells -/
def target_cells : ℕ := 145800

/-- The time in hours for one growth cycle -/
def cycle_time : ℕ := 5

/-- The function to calculate the number of cells after a given number of cycles -/
def cells_after_cycles (n : ℕ) : ℕ :=
  initial_cells * growth_factor ^ n

/-- The theorem stating the time taken for the algae to grow to at least the target number of cells -/
theorem algae_growth_time : ∃ (t : ℕ), 
  cells_after_cycles (t / cycle_time) ≥ target_cells ∧ 
  ∀ (s : ℕ), s < t → cells_after_cycles (s / cycle_time) < target_cells :=
by sorry

end NUMINAMATH_CALUDE_algae_growth_time_l54_5428


namespace NUMINAMATH_CALUDE_bunny_burrow_exits_l54_5422

/-- Represents the total number of bunnies in the community -/
def total_bunnies : ℕ := 100

/-- Represents the number of bunnies in Group A -/
def group_a_bunnies : ℕ := 40

/-- Represents the number of bunnies in Group B -/
def group_b_bunnies : ℕ := 30

/-- Represents the number of bunnies in Group C -/
def group_c_bunnies : ℕ := 30

/-- Represents how many times a bunny in Group A comes out per minute -/
def group_a_frequency : ℚ := 3

/-- Represents how many times a bunny in Group B comes out per minute -/
def group_b_frequency : ℚ := 5 / 2

/-- Represents how many times a bunny in Group C comes out per minute -/
def group_c_frequency : ℚ := 8 / 5

/-- Represents the reduction factor in burrow-exiting behavior after environmental change -/
def reduction_factor : ℚ := 1 / 2

/-- Represents the number of weeks before environmental change -/
def weeks_before_change : ℕ := 1

/-- Represents the number of weeks after environmental change -/
def weeks_after_change : ℕ := 2

/-- Represents the total number of weeks -/
def total_weeks : ℕ := 3

/-- Represents the number of minutes in a day -/
def minutes_per_day : ℕ := 1440

/-- Represents the number of days in a week -/
def days_per_week : ℕ := 7

/-- Theorem stating that the combined number of times all bunnies come out during 3 weeks is 4,897,920 -/
theorem bunny_burrow_exits : 
  (group_a_bunnies * group_a_frequency + 
   group_b_bunnies * group_b_frequency + 
   group_c_bunnies * group_c_frequency) * 
  minutes_per_day * days_per_week * weeks_before_change +
  (group_a_bunnies * group_a_frequency + 
   group_b_bunnies * group_b_frequency + 
   group_c_bunnies * group_c_frequency) * 
  reduction_factor * 
  minutes_per_day * days_per_week * weeks_after_change = 4897920 := by
  sorry

end NUMINAMATH_CALUDE_bunny_burrow_exits_l54_5422


namespace NUMINAMATH_CALUDE_expand_product_l54_5401

theorem expand_product (x : ℝ) : (x^2 + 3*x + 3) * (x^2 - 3*x + 3) = x^4 - 3*x^2 + 9 := by
  sorry

end NUMINAMATH_CALUDE_expand_product_l54_5401


namespace NUMINAMATH_CALUDE_two_distinct_solutions_l54_5419

theorem two_distinct_solutions (a : ℝ) : 
  (16 * (a - 3) > 0) →
  (a > 0) →
  (a^2 - 16*a + 48 > 0) →
  (a ≠ 19) →
  (∃ (x₁ x₂ : ℝ), x₁ = a + 4 * Real.sqrt (a - 3) ∧ 
                   x₂ = a - 4 * Real.sqrt (a - 3) ∧ 
                   x₁ > 0 ∧ x₂ > 0 ∧ x₁ ≠ x₂) →
  (a > 3 ∧ a < 4) ∨ (a > 12 ∧ a < 19) ∨ (a > 19) :=
by sorry


end NUMINAMATH_CALUDE_two_distinct_solutions_l54_5419


namespace NUMINAMATH_CALUDE_arithmetic_mean_of_ages_l54_5424

theorem arithmetic_mean_of_ages : 
  let ages : List ℝ := [18, 27, 35, 46]
  (ages.sum / ages.length : ℝ) = 31.5 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_mean_of_ages_l54_5424


namespace NUMINAMATH_CALUDE_perfect_pairing_S8_exists_no_perfect_pairing_S5_l54_5423

def Sn (n : ℕ) : Set ℕ := {x | 1 ≤ x ∧ x ≤ 2*n}

def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, n = m^2

def is_perfect_pairing (n : ℕ) (pairing : List (ℕ × ℕ)) : Prop :=
  (∀ (pair : ℕ × ℕ), pair ∈ pairing → pair.1 ∈ Sn n ∧ pair.2 ∈ Sn n) ∧
  (∀ x ∈ Sn n, ∃ pair ∈ pairing, x = pair.1 ∨ x = pair.2) ∧
  (∀ pair ∈ pairing, is_perfect_square (pair.1 + pair.2)) ∧
  pairing.length = n

theorem perfect_pairing_S8_exists : ∃ pairing : List (ℕ × ℕ), is_perfect_pairing 8 pairing :=
sorry

theorem no_perfect_pairing_S5 : ¬∃ pairing : List (ℕ × ℕ), is_perfect_pairing 5 pairing :=
sorry

end NUMINAMATH_CALUDE_perfect_pairing_S8_exists_no_perfect_pairing_S5_l54_5423


namespace NUMINAMATH_CALUDE_sameGradePercentage_is_32_percent_l54_5407

/-- Represents the number of students who received the same grade on both tests for each grade. -/
structure SameGradeCount where
  a : ℕ
  b : ℕ
  c : ℕ
  d : ℕ
  e : ℕ

/-- Calculates the percentage of students who received the same grade on both tests. -/
def sameGradePercentage (totalStudents : ℕ) (sameGrades : SameGradeCount) : ℚ :=
  let sameGradeTotal := sameGrades.a + sameGrades.b + sameGrades.c + sameGrades.d + sameGrades.e
  (sameGradeTotal : ℚ) / (totalStudents : ℚ) * 100

/-- Theorem stating that the percentage of students who received the same grade on both tests is 32%. -/
theorem sameGradePercentage_is_32_percent :
  let totalStudents := 50
  let sameGrades := SameGradeCount.mk 4 6 3 2 1
  sameGradePercentage totalStudents sameGrades = 32 := by
  sorry

end NUMINAMATH_CALUDE_sameGradePercentage_is_32_percent_l54_5407


namespace NUMINAMATH_CALUDE_brown_hat_fraction_l54_5445

theorem brown_hat_fraction (H : ℝ) (H_pos : H > 0) : ∃ B : ℝ,
  B > 0 ∧ B < 1 ∧
  (1/5 * B * H) / (1/3 * H) = 0.15 ∧
  B = 1/4 := by
sorry

end NUMINAMATH_CALUDE_brown_hat_fraction_l54_5445


namespace NUMINAMATH_CALUDE_system_solution_proof_l54_5448

theorem system_solution_proof :
  let x₁ : ℚ := 3/2
  let x₂ : ℚ := 1/2
  (3 * x₁ - 5 * x₂ = 2) ∧ (2 * x₁ + 4 * x₂ = 5) := by
  sorry

end NUMINAMATH_CALUDE_system_solution_proof_l54_5448


namespace NUMINAMATH_CALUDE_toothpicks_required_l54_5478

/-- The number of small triangles in the base of the large triangle -/
def base_triangles : ℕ := 3000

/-- The total number of small triangles in the large triangle -/
def total_triangles : ℕ := (base_triangles * (base_triangles + 1)) / 2

/-- The total number of toothpicks if no sides were shared -/
def total_potential_toothpicks : ℕ := 3 * total_triangles

/-- The number of toothpicks on the boundary of the large triangle -/
def boundary_toothpicks : ℕ := 3 * base_triangles

/-- The theorem stating the total number of toothpicks required -/
theorem toothpicks_required : 
  (total_potential_toothpicks - boundary_toothpicks) / 2 + boundary_toothpicks = 6761700 := by
  sorry

end NUMINAMATH_CALUDE_toothpicks_required_l54_5478


namespace NUMINAMATH_CALUDE_yellow_peaches_undetermined_l54_5483

def basket_peaches (red green yellow : ℕ) : Prop :=
  red = 7 ∧ green = 8 ∧ green = red + 1

theorem yellow_peaches_undetermined :
  ∀ (red green yellow : ℕ),
    basket_peaches red green yellow →
    ¬∃ (n : ℕ), ∀ (y : ℕ), basket_peaches red green y → y = n :=
by
  sorry

end NUMINAMATH_CALUDE_yellow_peaches_undetermined_l54_5483


namespace NUMINAMATH_CALUDE_triple_hash_48_l54_5480

-- Define the # operation
def hash (N : ℝ) : ℝ := 0.75 * N + 2

-- State the theorem
theorem triple_hash_48 : hash (hash (hash 48)) = 24.875 := by
  sorry

end NUMINAMATH_CALUDE_triple_hash_48_l54_5480


namespace NUMINAMATH_CALUDE_min_cuts_for_polygons_l54_5474

theorem min_cuts_for_polygons (n : ℕ) (k : ℕ) (s : ℕ) : 
  n = 73 ∧ s = 30 ∧ k ≥ n - 1 ∧ 
  (n * ((s - 2) * π) + (k + 1 - n) * π ≤ (k + 1) * 2 * π) →
  k ≥ 1970 :=
sorry

end NUMINAMATH_CALUDE_min_cuts_for_polygons_l54_5474


namespace NUMINAMATH_CALUDE_gold_coin_distribution_l54_5418

theorem gold_coin_distribution (x y : ℕ) (h : x^2 - y^2 = 49*(x - y)) : x + y = 49 := by
  sorry

end NUMINAMATH_CALUDE_gold_coin_distribution_l54_5418


namespace NUMINAMATH_CALUDE_all_fractions_repeat_l54_5420

theorem all_fractions_repeat (n : ℕ) (h1 : 1 ≤ n) (h2 : n ≤ 20) : 
  ¬ (∃ k : ℕ, n * (5^k * 2^k) = 42 * m) :=
sorry

end NUMINAMATH_CALUDE_all_fractions_repeat_l54_5420


namespace NUMINAMATH_CALUDE_x_value_l54_5408

theorem x_value (x y : ℝ) (h1 : x - y = 6) (h2 : x + y = 12) : x = 9 := by
  sorry

end NUMINAMATH_CALUDE_x_value_l54_5408


namespace NUMINAMATH_CALUDE_range_of_function_l54_5473

theorem range_of_function : 
  ∀ y : ℝ, ∃ x : ℝ, |x + 3| - |x - 5| + 3 * x = y := by
sorry

end NUMINAMATH_CALUDE_range_of_function_l54_5473


namespace NUMINAMATH_CALUDE_line_inclination_theorem_l54_5411

theorem line_inclination_theorem (a b c : ℝ) (α : ℝ) : 
  (∃ x y, a * x + b * y + c = 0) →  -- Line exists
  (Real.tan α = -a / b) →           -- Relationship between inclination angle and coefficients
  (Real.sin α + Real.cos α = 0) →   -- Given condition
  a - b = 0 := by
sorry

end NUMINAMATH_CALUDE_line_inclination_theorem_l54_5411


namespace NUMINAMATH_CALUDE_subtraction_error_correction_l54_5450

theorem subtraction_error_correction (x y : ℕ) 
  (h1 : x - y = 8008)
  (h2 : x - 10 * y = 88) :
  x = 8888 := by
  sorry

end NUMINAMATH_CALUDE_subtraction_error_correction_l54_5450


namespace NUMINAMATH_CALUDE_distance_home_to_school_l54_5449

/-- Represents the scenario of a boy traveling between home and school. -/
structure TravelScenario where
  speed : ℝ  -- Speed in km/hr
  time_diff : ℝ  -- Time difference in hours (positive for late, negative for early)

/-- The distance between home and school satisfies the given travel scenarios. -/
def distance_satisfies (d : ℝ) (s1 s2 : TravelScenario) : Prop :=
  ∃ t : ℝ, 
    d = s1.speed * (t + s1.time_diff) ∧
    d = s2.speed * (t - s2.time_diff)

/-- The theorem stating the distance between home and school. -/
theorem distance_home_to_school : 
  ∃ d : ℝ, d = 1.5 ∧ 
    distance_satisfies d 
      { speed := 3, time_diff := 7/60 }
      { speed := 6, time_diff := -8/60 } := by
  sorry

end NUMINAMATH_CALUDE_distance_home_to_school_l54_5449


namespace NUMINAMATH_CALUDE_smallest_a_value_l54_5490

-- Define the polynomial
def polynomial (a b x : ℤ) : ℤ := x^3 - a*x^2 + b*x - 2310

-- Define the property of having three positive integer roots
def has_three_positive_integer_roots (a b : ℤ) : Prop :=
  ∃ (x y z : ℤ), x > 0 ∧ y > 0 ∧ z > 0 ∧
    polynomial a b x = 0 ∧ polynomial a b y = 0 ∧ polynomial a b z = 0

-- State the theorem
theorem smallest_a_value :
  ∀ a b : ℤ, has_three_positive_integer_roots a b →
    (∀ a' b' : ℤ, has_three_positive_integer_roots a' b' → a ≤ a') →
    a = 88 :=
sorry

end NUMINAMATH_CALUDE_smallest_a_value_l54_5490


namespace NUMINAMATH_CALUDE_triangle_angle_sine_cosine_equivalence_l54_5464

theorem triangle_angle_sine_cosine_equivalence (A B C : Real) 
  (h_triangle : A + B + C = Real.pi) 
  (h_positive : 0 < A ∧ 0 < B ∧ 0 < C) :
  (Real.sin A > Real.sin B) ↔ (Real.cos A + Real.cos (A + C) < 0) := by
  sorry

end NUMINAMATH_CALUDE_triangle_angle_sine_cosine_equivalence_l54_5464


namespace NUMINAMATH_CALUDE_total_credits_proof_l54_5415

theorem total_credits_proof (emily_credits : ℕ) 
  (h1 : emily_credits = 20)
  (h2 : ∃ aria_credits : ℕ, aria_credits = 2 * emily_credits)
  (h3 : ∃ spencer_credits : ℕ, spencer_credits = emily_credits / 2)
  (h4 : ∃ hannah_credits : ℕ, hannah_credits = 3 * (emily_credits / 2)) :
  2 * (emily_credits + 2 * emily_credits + emily_credits / 2 + 3 * (emily_credits / 2)) = 200 := by
  sorry

end NUMINAMATH_CALUDE_total_credits_proof_l54_5415


namespace NUMINAMATH_CALUDE_poultry_pricing_l54_5437

theorem poultry_pricing :
  ∃ (c d g : ℕ+),
    3 * c + d = 2 * g ∧
    c + 2 * d + 3 * g = 25 ∧
    c = 2 ∧ d = 4 ∧ g = 5 := by
  sorry

end NUMINAMATH_CALUDE_poultry_pricing_l54_5437


namespace NUMINAMATH_CALUDE_square_area_with_circles_l54_5462

/-- The area of a square containing a 3x3 grid of circles with radius 3 inches -/
theorem square_area_with_circles (r : ℝ) (h : r = 3) : 
  (3 * (2 * r))^2 = 324 := by
  sorry

end NUMINAMATH_CALUDE_square_area_with_circles_l54_5462


namespace NUMINAMATH_CALUDE_photo_count_proof_l54_5496

def final_photo_count (initial_photos : ℕ) (deleted_bad_shots : ℕ) (cat_photos : ℕ) (friend_photos : ℕ) (deleted_after_edit : ℕ) : ℕ :=
  initial_photos - deleted_bad_shots + cat_photos + friend_photos - deleted_after_edit

theorem photo_count_proof (x : ℕ) : 
  final_photo_count 63 7 15 x 3 = 68 + x := by
  sorry

end NUMINAMATH_CALUDE_photo_count_proof_l54_5496


namespace NUMINAMATH_CALUDE_student_scores_theorem_l54_5476

/-- A score is a triple of integers, each between 0 and 7 inclusive -/
def Score := { s : Fin 3 → Fin 8 // True }

/-- Given two scores, returns true if the first score is at least as high as the second for each problem -/
def ScoreGreaterEq (s1 s2 : Score) : Prop :=
  ∀ i : Fin 3, s1.val i ≥ s2.val i

theorem student_scores_theorem (scores : Fin 49 → Score) :
  ∃ i j : Fin 49, i ≠ j ∧ ScoreGreaterEq (scores i) (scores j) := by
  sorry

#check student_scores_theorem

end NUMINAMATH_CALUDE_student_scores_theorem_l54_5476


namespace NUMINAMATH_CALUDE_perfect_square_condition_l54_5412

theorem perfect_square_condition (n : ℤ) :
  (∃ k : ℤ, 7 * n + 2 = k ^ 2) ↔ 
  (∃ m : ℤ, (n = 7 * m ^ 2 + 6 * m + 1) ∨ (n = 7 * m ^ 2 - 6 * m + 1)) :=
by sorry

end NUMINAMATH_CALUDE_perfect_square_condition_l54_5412


namespace NUMINAMATH_CALUDE_cow_horse_ratio_l54_5470

theorem cow_horse_ratio (total : ℕ) (cows : ℕ) (horses : ℕ) (h1 : total = 168) (h2 : cows = 140) (h3 : total = cows + horses) (h4 : ∃ r : ℕ, cows = r * horses) : 
  cows / horses = 5 := by
  sorry

end NUMINAMATH_CALUDE_cow_horse_ratio_l54_5470


namespace NUMINAMATH_CALUDE_janes_mean_score_l54_5492

def quiz_scores : List ℝ := [99, 95, 93, 87, 90]
def exam_scores : List ℝ := [88, 92]

def all_scores : List ℝ := quiz_scores ++ exam_scores

theorem janes_mean_score :
  (all_scores.sum / all_scores.length : ℝ) = 644 / 7 := by
  sorry

end NUMINAMATH_CALUDE_janes_mean_score_l54_5492


namespace NUMINAMATH_CALUDE_pave_square_iff_integer_hypotenuse_l54_5416

/-- Represents a right triangle with integer side lengths -/
structure RightTriangle where
  a : ℕ  -- Length of side AB
  b : ℕ  -- Length of side AC
  h : a^2 + b^2 > 0  -- Ensures the triangle is non-degenerate

/-- Checks if a square can be completely paved with a given right triangle -/
def can_pave_square (t : RightTriangle) : Prop :=
  ∃ (n : ℕ), ∃ (m : ℕ), m * (t.a * t.b) = 2 * n^2 * (t.a^2 + t.b^2)

/-- The main theorem: A square can be paved if and only if the hypotenuse is an integer -/
theorem pave_square_iff_integer_hypotenuse (t : RightTriangle) :
  can_pave_square t ↔ ∃ (k : ℕ), k^2 = t.a^2 + t.b^2 :=
sorry

end NUMINAMATH_CALUDE_pave_square_iff_integer_hypotenuse_l54_5416


namespace NUMINAMATH_CALUDE_scientific_notation_4212000_l54_5488

theorem scientific_notation_4212000 :
  ∃ (a : ℝ) (n : ℤ), 
    4212000 = a * (10 : ℝ) ^ n ∧ 
    1 ≤ a ∧ 
    a < 10 ∧ 
    a = 4.212 ∧ 
    n = 6 := by
  sorry

end NUMINAMATH_CALUDE_scientific_notation_4212000_l54_5488


namespace NUMINAMATH_CALUDE_niko_sock_profit_l54_5479

theorem niko_sock_profit : ∀ (total_pairs : ℕ) (cost_per_pair : ℚ) 
  (profit_percent : ℚ) (profit_amount : ℚ) (high_profit_pairs : ℕ) (low_profit_pairs : ℕ),
  total_pairs = 9 →
  cost_per_pair = 2 →
  profit_percent = 25 / 100 →
  profit_amount = 1 / 5 →
  high_profit_pairs = 4 →
  low_profit_pairs = 5 →
  high_profit_pairs + low_profit_pairs = total_pairs →
  (high_profit_pairs : ℚ) * (cost_per_pair * profit_percent) + 
  (low_profit_pairs : ℚ) * profit_amount = 3 := by
sorry

end NUMINAMATH_CALUDE_niko_sock_profit_l54_5479


namespace NUMINAMATH_CALUDE_count_lines_4x4_grid_l54_5453

/-- A point in a 2D grid -/
structure GridPoint where
  x : Nat
  y : Nat

/-- A line in a 2D grid -/
structure GridLine where
  points : Set GridPoint

/-- A 4-by-4 grid of lattice points -/
def Grid4x4 : Set GridPoint :=
  {p | p.x < 4 ∧ p.y < 4}

/-- A function that determines if a line passes through at least two points in the grid -/
def passesThrough2Points (l : GridLine) (grid : Set GridPoint) : Prop :=
  (l.points ∩ grid).ncard ≥ 2

/-- The set of all lines that pass through at least two points in the 4-by-4 grid -/
def validLines : Set GridLine :=
  {l | passesThrough2Points l Grid4x4}

theorem count_lines_4x4_grid :
  (validLines).ncard = 88 := by sorry

end NUMINAMATH_CALUDE_count_lines_4x4_grid_l54_5453


namespace NUMINAMATH_CALUDE_remainder_double_n_l54_5451

theorem remainder_double_n (n : ℕ) (h : n % 4 = 3) : (2 * n) % 4 = 2 := by
  sorry

end NUMINAMATH_CALUDE_remainder_double_n_l54_5451


namespace NUMINAMATH_CALUDE_rectangle_area_diagonal_l54_5434

theorem rectangle_area_diagonal (length width diagonal : ℝ) (k : ℝ) : 
  length > 0 → width > 0 → diagonal > 0 →
  length / width = 3 / 2 → 
  diagonal^2 = length^2 + width^2 →
  k = 6 / 13 →
  length * width = k * diagonal^2 := by
sorry

end NUMINAMATH_CALUDE_rectangle_area_diagonal_l54_5434


namespace NUMINAMATH_CALUDE_quadratic_power_function_l54_5469

/-- A function is a power function if it's of the form f(x) = x^a for some real number a -/
def IsPowerFunction (f : ℝ → ℝ) : Prop :=
  ∃ a : ℝ, ∀ x : ℝ, f x = x^a

/-- A function is quadratic if it's of the form f(x) = ax^2 + bx + c for some real numbers a, b, c with a ≠ 0 -/
def IsQuadratic (f : ℝ → ℝ) : Prop :=
  ∃ a b c : ℝ, a ≠ 0 ∧ ∀ x : ℝ, f x = a * x^2 + b * x + c

theorem quadratic_power_function (f : ℝ → ℝ) :
  IsQuadratic f ∧ IsPowerFunction f → ∀ x : ℝ, f x = x^2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_power_function_l54_5469


namespace NUMINAMATH_CALUDE_circle_area_l54_5446

/-- The area of the circle defined by the equation 3x^2 + 3y^2 - 12x + 9y + 27 = 0 is equal to 61π/4 -/
theorem circle_area (x y : ℝ) : 
  (3 * x^2 + 3 * y^2 - 12 * x + 9 * y + 27 = 0) → 
  (∃ (center : ℝ × ℝ) (radius : ℝ), 
    ((x - center.1)^2 + (y - center.2)^2 = radius^2) ∧ 
    (π * radius^2 = 61 * π / 4)) :=
by sorry

end NUMINAMATH_CALUDE_circle_area_l54_5446


namespace NUMINAMATH_CALUDE_smallest_eight_digit_four_fours_l54_5417

def is_eight_digit (n : ℕ) : Prop := 10000000 ≤ n ∧ n ≤ 99999999

def count_digit (n : ℕ) (d : ℕ) : ℕ :=
  (n.digits 10).filter (· = d) |>.length

theorem smallest_eight_digit_four_fours : 
  ∀ n : ℕ, is_eight_digit n → count_digit n 4 = 4 → 10004444 ≤ n :=
sorry

end NUMINAMATH_CALUDE_smallest_eight_digit_four_fours_l54_5417


namespace NUMINAMATH_CALUDE_largest_integer_negative_quadratic_seven_satisfies_inequality_eight_does_not_satisfy_inequality_l54_5444

theorem largest_integer_negative_quadratic :
  ∀ n : ℤ, n^2 - 13*n + 40 < 0 → n ≤ 7 :=
by
  sorry

theorem seven_satisfies_inequality :
  7^2 - 13*7 + 40 < 0 :=
by
  sorry

theorem eight_does_not_satisfy_inequality :
  8^2 - 13*8 + 40 ≥ 0 :=
by
  sorry

end NUMINAMATH_CALUDE_largest_integer_negative_quadratic_seven_satisfies_inequality_eight_does_not_satisfy_inequality_l54_5444


namespace NUMINAMATH_CALUDE_pit_stop_duration_is_20_minutes_l54_5425

-- Define the parameters of the problem
def total_trip_time_without_stops : ℕ := 14 -- hours
def stop_interval : ℕ := 2 -- hours
def additional_food_stops : ℕ := 2
def additional_gas_stops : ℕ := 3
def total_trip_time_with_stops : ℕ := 18 -- hours

-- Calculate the number of stops
def total_stops : ℕ := 
  (total_trip_time_without_stops / stop_interval) + additional_food_stops + additional_gas_stops

-- Define the theorem
theorem pit_stop_duration_is_20_minutes : 
  (total_trip_time_with_stops - total_trip_time_without_stops) * 60 / total_stops = 20 := by
  sorry

end NUMINAMATH_CALUDE_pit_stop_duration_is_20_minutes_l54_5425


namespace NUMINAMATH_CALUDE_youngest_brother_age_l54_5432

theorem youngest_brother_age (a b c : ℕ) : 
  b = a + 1 → c = b + 1 → a + b + c = 96 → a = 31 := by
sorry

end NUMINAMATH_CALUDE_youngest_brother_age_l54_5432


namespace NUMINAMATH_CALUDE_problem_solution_l54_5457

theorem problem_solution : ∀ M N X : ℕ,
  M = 2022 / 3 →
  N = M / 3 →
  X = M + N →
  X = 898 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l54_5457


namespace NUMINAMATH_CALUDE_walter_hushpuppies_per_guest_l54_5442

/-- Calculates the number of hushpuppies per guest given the number of guests,
    cooking rate, and total cooking time. -/
def hushpuppies_per_guest (guests : ℕ) (hushpuppies_per_batch : ℕ) 
    (minutes_per_batch : ℕ) (total_minutes : ℕ) : ℕ :=
  (total_minutes / minutes_per_batch * hushpuppies_per_batch) / guests

/-- Proves that given the specified conditions, each guest will eat 5 hushpuppies. -/
theorem walter_hushpuppies_per_guest : 
  hushpuppies_per_guest 20 10 8 80 = 5 := by
  sorry

end NUMINAMATH_CALUDE_walter_hushpuppies_per_guest_l54_5442


namespace NUMINAMATH_CALUDE_prob_region_D_total_prob_is_one_l54_5452

/-- Represents the regions on the wheel of fortune -/
inductive Region
| A
| B
| C
| D

/-- The probability function for the wheel of fortune -/
def P : Region → ℚ
| Region.A => 1/4
| Region.B => 1/3
| Region.C => 1/6
| Region.D => 1 - (1/4 + 1/3 + 1/6)

/-- The theorem stating that the probability of landing on region D is 1/4 -/
theorem prob_region_D : P Region.D = 1/4 := by
  sorry

/-- The sum of all probabilities is 1 -/
theorem total_prob_is_one : P Region.A + P Region.B + P Region.C + P Region.D = 1 := by
  sorry

end NUMINAMATH_CALUDE_prob_region_D_total_prob_is_one_l54_5452


namespace NUMINAMATH_CALUDE_cubes_with_four_neighbors_eq_108_l54_5458

/-- Represents a parallelepiped with dimensions a, b, and c. -/
structure Parallelepiped where
  a : ℕ
  b : ℕ
  c : ℕ
  h1 : a > 3
  h2 : b > 3
  h3 : c > 3
  h4 : (a - 2) * (b - 2) * (c - 2) = 429

/-- The number of unit cubes with exactly 4 neighbors in a parallelepiped. -/
def cubes_with_four_neighbors (p : Parallelepiped) : ℕ :=
  4 * ((p.a - 2) + (p.b - 2) + (p.c - 2))

theorem cubes_with_four_neighbors_eq_108 (p : Parallelepiped) :
  cubes_with_four_neighbors p = 108 := by
  sorry

end NUMINAMATH_CALUDE_cubes_with_four_neighbors_eq_108_l54_5458


namespace NUMINAMATH_CALUDE_tan_sum_pi_twelfths_l54_5456

theorem tan_sum_pi_twelfths : 
  Real.tan (π / 12) + Real.tan (5 * π / 12) = 4 := by sorry

end NUMINAMATH_CALUDE_tan_sum_pi_twelfths_l54_5456


namespace NUMINAMATH_CALUDE_triangle_problem_l54_5402

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- Theorem stating the two parts of the problem -/
theorem triangle_problem (t : Triangle) :
  (t.a = 3 ∧ t.b = 5 ∧ t.B = 2 * π / 3 → Real.sin t.A = 3 * Real.sqrt 3 / 10) ∧
  (t.a = 3 ∧ t.b = 5 ∧ t.C = 2 * π / 3 → t.c = 7) := by
  sorry

end NUMINAMATH_CALUDE_triangle_problem_l54_5402


namespace NUMINAMATH_CALUDE_sqrt_21_is_11th_term_l54_5400

theorem sqrt_21_is_11th_term (a : ℕ → ℝ) :
  (∀ n, a n = Real.sqrt (2 * n - 1)) →
  a 11 = Real.sqrt 21 :=
by
  sorry

end NUMINAMATH_CALUDE_sqrt_21_is_11th_term_l54_5400


namespace NUMINAMATH_CALUDE_quadratic_inequality_roots_l54_5405

theorem quadratic_inequality_roots (c : ℝ) : 
  (∀ x, -x^2 + c*x - 8 < 0 ↔ x < 2 ∨ x > 6) → c = 8 := by
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_roots_l54_5405


namespace NUMINAMATH_CALUDE_largest_integer_with_remainder_l54_5440

theorem largest_integer_with_remainder : ∃ n : ℕ, n < 85 ∧ n % 9 = 3 ∧ ∀ m : ℕ, m < 85 ∧ m % 9 = 3 → m ≤ n :=
by sorry

end NUMINAMATH_CALUDE_largest_integer_with_remainder_l54_5440


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_l54_5486

-- Define the complex number z
def z : ℂ := Complex.I * (2 + Complex.I)

-- Theorem stating that the imaginary part of z is 2
theorem imaginary_part_of_z : z.im = 2 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_l54_5486


namespace NUMINAMATH_CALUDE_bridesmaid_dresses_completion_time_l54_5466

/-- Calculates the number of weeks needed to complete bridesmaid dresses -/
def weeks_to_complete_dresses (hours_per_dress : ℕ) (num_bridesmaids : ℕ) (hours_per_week : ℕ) : ℕ :=
  (hours_per_dress * num_bridesmaids) / hours_per_week

/-- Proves that it takes 15 weeks to complete the bridesmaid dresses under given conditions -/
theorem bridesmaid_dresses_completion_time :
  weeks_to_complete_dresses 12 5 4 = 15 := by
  sorry

#eval weeks_to_complete_dresses 12 5 4

end NUMINAMATH_CALUDE_bridesmaid_dresses_completion_time_l54_5466


namespace NUMINAMATH_CALUDE_min_stamps_for_47_cents_l54_5431

def stamps (x y : ℕ) : ℕ := 5 * x + 7 * y

theorem min_stamps_for_47_cents :
  ∃ (x y : ℕ), stamps x y = 47 ∧
  (∀ (a b : ℕ), stamps a b = 47 → x + y ≤ a + b) ∧
  x + y = 7 := by
  sorry

end NUMINAMATH_CALUDE_min_stamps_for_47_cents_l54_5431


namespace NUMINAMATH_CALUDE_equal_spending_dolls_l54_5499

/-- The number of sisters Tonya is buying gifts for -/
def num_sisters : ℕ := 2

/-- The cost of each doll in dollars -/
def doll_cost : ℕ := 15

/-- The cost of each lego set in dollars -/
def lego_cost : ℕ := 20

/-- The number of lego sets bought for the older sister -/
def num_lego_sets : ℕ := 3

/-- The total amount spent on the older sister in dollars -/
def older_sister_cost : ℕ := num_lego_sets * lego_cost

/-- The number of dolls bought for the younger sister -/
def num_dolls : ℕ := older_sister_cost / doll_cost

theorem equal_spending_dolls : num_dolls = 4 := by
  sorry

end NUMINAMATH_CALUDE_equal_spending_dolls_l54_5499


namespace NUMINAMATH_CALUDE_granola_bar_distribution_l54_5409

theorem granola_bar_distribution (total : ℕ) (eaten_by_parents : ℕ) (num_children : ℕ) :
  total = 200 →
  eaten_by_parents = 80 →
  num_children = 6 →
  (total - eaten_by_parents) / num_children = 20 :=
by
  sorry

end NUMINAMATH_CALUDE_granola_bar_distribution_l54_5409


namespace NUMINAMATH_CALUDE_max_ice_cream_servings_l54_5436

/-- Represents a day in February --/
structure FebruaryDay where
  dayOfMonth : Nat
  dayOfWeek : Nat
  h1 : dayOfMonth ≥ 1 ∧ dayOfMonth ≤ 28
  h2 : dayOfWeek ≥ 1 ∧ dayOfWeek ≤ 7

/-- Defines the ice cream eating rules --/
def iceCreamServings (day : FebruaryDay) : Nat :=
  if day.dayOfMonth % 2 = 0 ∧ (day.dayOfWeek = 3 ∨ day.dayOfWeek = 4) then 7
  else if (day.dayOfWeek = 1 ∨ day.dayOfWeek = 2) ∧ day.dayOfMonth % 2 = 1 then 3
  else if day.dayOfWeek = 5 then day.dayOfMonth
  else 0

/-- Theorem stating the maximum number of ice cream servings in February --/
theorem max_ice_cream_servings :
  (∃ (days : List FebruaryDay), days.length = 28 ∧
    (∀ d ∈ days, d.dayOfMonth ≥ 1 ∧ d.dayOfMonth ≤ 28 ∧ d.dayOfWeek ≥ 1 ∧ d.dayOfWeek ≤ 7) ∧
    (∀ i j, i ≠ j → (days.get i).dayOfMonth ≠ (days.get j).dayOfMonth) ∧
    (List.sum (days.map iceCreamServings) ≤ 110)) ∧
  (∃ (optimalDays : List FebruaryDay), optimalDays.length = 28 ∧
    (∀ d ∈ optimalDays, d.dayOfMonth ≥ 1 ∧ d.dayOfMonth ≤ 28 ∧ d.dayOfWeek ≥ 1 ∧ d.dayOfWeek ≤ 7) ∧
    (∀ i j, i ≠ j → (optimalDays.get i).dayOfMonth ≠ (optimalDays.get j).dayOfMonth) ∧
    (List.sum (optimalDays.map iceCreamServings) = 110)) := by
  sorry


end NUMINAMATH_CALUDE_max_ice_cream_servings_l54_5436


namespace NUMINAMATH_CALUDE_translation_result_l54_5460

/-- A point in the Cartesian coordinate system -/
structure Point where
  x : ℝ
  y : ℝ

/-- Translate a point horizontally -/
def translateHorizontal (p : Point) (dx : ℝ) : Point :=
  { x := p.x + dx, y := p.y }

/-- Translate a point vertically -/
def translateVertical (p : Point) (dy : ℝ) : Point :=
  { x := p.x, y := p.y + dy }

/-- The main theorem stating that translating (1, -2) by 2 units right and 3 units up results in (3, 1) -/
theorem translation_result : 
  let initial_point : Point := { x := 1, y := -2 }
  let after_horizontal : Point := translateHorizontal initial_point 2
  let final_point : Point := translateVertical after_horizontal 3
  final_point = { x := 3, y := 1 } := by
  sorry

end NUMINAMATH_CALUDE_translation_result_l54_5460


namespace NUMINAMATH_CALUDE_height_ranking_l54_5461

-- Define the siblings
inductive Sibling
| Dan
| Elena
| Finn

-- Define the height comparison relation
def taller_than : Sibling → Sibling → Prop := sorry

-- Define the properties of the height comparison
axiom taller_than_transitive : ∀ a b c : Sibling, taller_than a b → taller_than b c → taller_than a c
axiom taller_than_asymmetric : ∀ a b : Sibling, taller_than a b → ¬(taller_than b a)
axiom taller_than_trichotomous : ∀ a b : Sibling, a ≠ b → (taller_than a b ∨ taller_than b a)

-- Define the statements from the problem
def elena_not_tallest : Prop := ∃ s : Sibling, s ≠ Sibling.Elena ∧ taller_than s Sibling.Elena
def finn_tallest : Prop := ∀ s : Sibling, s ≠ Sibling.Finn → taller_than Sibling.Finn s
def dan_not_shortest : Prop := ∃ s : Sibling, s ≠ Sibling.Dan ∧ taller_than Sibling.Dan s

-- Define the condition that exactly one statement is true
def exactly_one_true : Prop :=
  (elena_not_tallest ∧ ¬finn_tallest ∧ ¬dan_not_shortest) ∨
  (¬elena_not_tallest ∧ finn_tallest ∧ ¬dan_not_shortest) ∨
  (¬elena_not_tallest ∧ ¬finn_tallest ∧ dan_not_shortest)

-- The theorem to be proved
theorem height_ranking :
  exactly_one_true →
  taller_than Sibling.Finn Sibling.Elena ∧
  taller_than Sibling.Elena Sibling.Dan :=
by sorry

end NUMINAMATH_CALUDE_height_ranking_l54_5461


namespace NUMINAMATH_CALUDE_min_h_for_circle_in_halfplane_l54_5454

/-- The minimum value of h for a circle (x-h)^2 + (y-1)^2 = 1 located within the plane region x + y + 1 ≥ 0 is √2 - 2. -/
theorem min_h_for_circle_in_halfplane :
  let C : ℝ → Set (ℝ × ℝ) := fun h => {p : ℝ × ℝ | (p.1 - h)^2 + (p.2 - 1)^2 = 1}
  let halfplane : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1 + p.2 + 1 ≥ 0}
  ∃ (h_min : ℝ), h_min = Real.sqrt 2 - 2 ∧
    (∀ h, (C h ⊆ halfplane) → h ≥ h_min) ∧
    (C h_min ⊆ halfplane) :=
by sorry

end NUMINAMATH_CALUDE_min_h_for_circle_in_halfplane_l54_5454


namespace NUMINAMATH_CALUDE_problem_1_l54_5477

def A : Set ℝ := {x | x^2 - 8*x + 15 = 0}
def B (a : ℝ) : Set ℝ := {x | a*x - 1 = 0}
def C : Set ℝ := {0, 1/3, 1/5}

theorem problem_1 : ∀ a : ℝ, B a ⊆ A ↔ a ∈ C := by sorry

end NUMINAMATH_CALUDE_problem_1_l54_5477


namespace NUMINAMATH_CALUDE_rectangle_division_even_triangles_l54_5414

theorem rectangle_division_even_triangles 
  (a b c d : ℕ) 
  (h_rect : a > 0 ∧ b > 0) 
  (h_tri : c > 0 ∧ d > 0) 
  (h_div : (a * b) % (c * d / 2) = 0) :
  ∃ k : ℕ, k % 2 = 0 ∧ k * (c * d / 2) = a * b :=
sorry

end NUMINAMATH_CALUDE_rectangle_division_even_triangles_l54_5414


namespace NUMINAMATH_CALUDE_sphere_radius_from_surface_area_l54_5410

theorem sphere_radius_from_surface_area (surface_area : Real) (radius : Real) : 
  surface_area = 64 * Real.pi → 4 * Real.pi * radius^2 = surface_area → radius = 4 := by
  sorry

end NUMINAMATH_CALUDE_sphere_radius_from_surface_area_l54_5410


namespace NUMINAMATH_CALUDE_antons_winning_strategy_l54_5468

theorem antons_winning_strategy :
  ∃ f : ℕ → ℕ, Function.Injective f ∧
  (∀ x : ℕ, 
    let n := f x
    ¬ ∃ m : ℕ, n = m * m ∧  -- n is not a perfect square
    ∃ k : ℕ, n + (n + 1) + (n + 2) = k * k) -- sum of three consecutive numbers starting from n is a perfect square
  := by sorry

end NUMINAMATH_CALUDE_antons_winning_strategy_l54_5468


namespace NUMINAMATH_CALUDE_rod_cutting_l54_5406

theorem rod_cutting (rod_length : ℝ) (piece_length : ℚ) : 
  rod_length = 58.75 →
  piece_length = 137 + 2/3 →
  ⌊(rod_length * 100) / (piece_length : ℝ)⌋ = 14 := by
  sorry

end NUMINAMATH_CALUDE_rod_cutting_l54_5406


namespace NUMINAMATH_CALUDE_publishing_profit_inequality_minimum_sets_answer_is_four_thousand_l54_5495

/-- Represents the number of thousands of sets -/
def x : ℝ := 4

/-- Fixed cost in yuan -/
def fixed_cost : ℝ := 80000

/-- Cost increase per set in yuan -/
def cost_increase : ℝ := 20

/-- Price per set in yuan -/
def price : ℝ := 100

/-- Underwriter's share of sales -/
def underwriter_share : ℝ := 0.3

/-- Publishing house's desired profit margin -/
def profit_margin : ℝ := 0.1

/-- The inequality that must be satisfied for the publishing house to achieve its desired profit -/
theorem publishing_profit_inequality :
  fixed_cost + cost_increase * 1000 * x ≤ price * (1 - underwriter_share - profit_margin) * 1000 * x :=
sorry

/-- The minimum number of sets (in thousands) that satisfies the inequality -/
theorem minimum_sets :
  x = ⌈(fixed_cost / (price * (1 - underwriter_share - profit_margin) * 1000 - cost_increase * 1000))⌉ :=
sorry

/-- Proof that 4,000 sets is the correct answer when rounded to the nearest thousand -/
theorem answer_is_four_thousand :
  ⌊x * 1000 / 1000 + 0.5⌋ * 1000 = 4000 :=
sorry

end NUMINAMATH_CALUDE_publishing_profit_inequality_minimum_sets_answer_is_four_thousand_l54_5495


namespace NUMINAMATH_CALUDE_log_equation_solutions_l54_5481

theorem log_equation_solutions :
  ∀ x y : ℝ, x > 0 → y > 0 →
  (Real.log x / Real.log 4 - Real.log y / Real.log 2 = 0) →
  (x^2 - 5*y^2 + 4 = 0) →
  ((x = 1 ∧ y = 1) ∨ (x = 4 ∧ y = 2)) := by
  sorry

end NUMINAMATH_CALUDE_log_equation_solutions_l54_5481


namespace NUMINAMATH_CALUDE_rectangle_area_l54_5413

theorem rectangle_area (width : ℝ) (length : ℝ) : 
  width = 7 → length = 4 * width → width * length = 196 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_l54_5413


namespace NUMINAMATH_CALUDE_necessary_not_sufficient_condition_l54_5467

theorem necessary_not_sufficient_condition (a b : ℝ) : 
  (∀ x y : ℝ, x < y → x < y + 1) ∧ 
  (∃ x y : ℝ, x < y + 1 ∧ ¬(x < y)) := by
  sorry

end NUMINAMATH_CALUDE_necessary_not_sufficient_condition_l54_5467


namespace NUMINAMATH_CALUDE_symmetry_of_shifted_even_function_l54_5427

-- Define an even function
def IsEven (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

-- Define symmetry about a vertical line
def SymmetricAboutLine (f : ℝ → ℝ) (a : ℝ) : Prop :=
  ∀ x, f (a + x) = f (a - x)

-- Theorem statement
theorem symmetry_of_shifted_even_function (f : ℝ → ℝ) :
  IsEven (fun x ↦ f (x + 1)) → SymmetricAboutLine f 1 := by
  sorry


end NUMINAMATH_CALUDE_symmetry_of_shifted_even_function_l54_5427


namespace NUMINAMATH_CALUDE_train_length_l54_5484

/-- The length of a train given specific conditions -/
theorem train_length (t : ℝ) (v : ℝ) (b : ℝ) (h1 : t = 30) (h2 : v = 45) (h3 : b = 205) :
  v * (1000 / 3600) * t - b = 170 :=
by sorry

end NUMINAMATH_CALUDE_train_length_l54_5484


namespace NUMINAMATH_CALUDE_age_problem_l54_5487

theorem age_problem (A B C : ℕ) : 
  A = B + 2 →
  B = 2 * C →
  A + B + C = 37 →
  B = 14 :=
by
  sorry

end NUMINAMATH_CALUDE_age_problem_l54_5487


namespace NUMINAMATH_CALUDE_simplify_and_ratio_l54_5491

theorem simplify_and_ratio (k : ℚ) : ∃ (a b : ℚ), 
  (6 * k + 12) / 3 = a * k + b ∧ a / b = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_simplify_and_ratio_l54_5491


namespace NUMINAMATH_CALUDE_book_page_sum_problem_l54_5489

theorem book_page_sum_problem :
  ∃ (n : ℕ) (p : ℕ), 
    0 < n ∧ 
    1 ≤ p ∧ 
    p ≤ n ∧ 
    n * (n + 1) / 2 + p = 2550 ∧ 
    p = 65 := by
  sorry

end NUMINAMATH_CALUDE_book_page_sum_problem_l54_5489


namespace NUMINAMATH_CALUDE_min_value_sum_l54_5482

theorem min_value_sum (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 2/x + 1/y = 1) :
  x + 2*y ≥ 8 := by
sorry

end NUMINAMATH_CALUDE_min_value_sum_l54_5482


namespace NUMINAMATH_CALUDE_tenth_valid_number_l54_5471

def digit_sum (n : ℕ) : ℕ := sorry

def is_valid_number (n : ℕ) : Prop :=
  n > 0 ∧ digit_sum n = 13

def nth_valid_number (n : ℕ) : ℕ := sorry

theorem tenth_valid_number : nth_valid_number 10 = 166 := sorry

end NUMINAMATH_CALUDE_tenth_valid_number_l54_5471


namespace NUMINAMATH_CALUDE_S_is_bounded_region_l54_5429

/-- The set S of points (x,y) in the coordinate plane where one of 5, x+1, and y-5 is greater than or equal to the other two -/
def S : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (5 ≥ p.1 + 1 ∧ 5 ≥ p.2 - 5) ∨
               (p.1 + 1 ≥ 5 ∧ p.1 + 1 ≥ p.2 - 5) ∨
               (p.2 - 5 ≥ 5 ∧ p.2 - 5 ≥ p.1 + 1)}

/-- S is a single bounded region in the quadrant -/
theorem S_is_bounded_region : 
  ∃ (a b c d : ℝ), a < b ∧ c < d ∧
  S = {p : ℝ × ℝ | a ≤ p.1 ∧ p.1 ≤ b ∧ c ≤ p.2 ∧ p.2 ≤ d} :=
by
  sorry

end NUMINAMATH_CALUDE_S_is_bounded_region_l54_5429


namespace NUMINAMATH_CALUDE_function_composition_property_l54_5433

theorem function_composition_property (f : ℤ → ℤ) (m : ℕ+) :
  (∀ n : ℤ, (f^[m] n = n + 2017)) → (m = 1 ∨ m = 2017) := by
  sorry

#check function_composition_property

end NUMINAMATH_CALUDE_function_composition_property_l54_5433


namespace NUMINAMATH_CALUDE_two_positions_from_six_candidates_l54_5404

/-- The number of ways to select two distinct positions from a group of candidates. -/
def selectTwoPositions (n : ℕ) : ℕ := n * (n - 1)

/-- The number of candidates. -/
def numCandidates : ℕ := 6

/-- The observed number of ways to select two positions. -/
def observedSelections : ℕ := 30

/-- Theorem stating that selecting 2 distinct positions from 6 candidates results in 30 possible selections. -/
theorem two_positions_from_six_candidates :
  selectTwoPositions numCandidates = observedSelections := by
  sorry


end NUMINAMATH_CALUDE_two_positions_from_six_candidates_l54_5404


namespace NUMINAMATH_CALUDE_fourth_day_temperature_l54_5421

def temperature_problem (t1 t2 t3 t4 : ℤ) : Prop :=
  let temps := [t1, t2, t3, t4]
  (t1 = -36) ∧ (t2 = 13) ∧ (t3 = -10) ∧ 
  (temps.sum / temps.length = -12) ∧
  (t4 = -15)

theorem fourth_day_temperature :
  ∃ t4 : ℤ, temperature_problem (-36) 13 (-10) t4 := by
  sorry

end NUMINAMATH_CALUDE_fourth_day_temperature_l54_5421


namespace NUMINAMATH_CALUDE_unique_solution_l54_5494

/-- The functional equation that f must satisfy for all x and y -/
def functional_equation (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (x - f y) = f (f y) + x * f y + f x - 1

/-- The theorem stating that f(x) = 1 - x²/2 is the unique solution -/
theorem unique_solution :
  ∃! f : ℝ → ℝ, functional_equation f ∧ ∀ x : ℝ, f x = 1 - x^2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_l54_5494


namespace NUMINAMATH_CALUDE_shirt_price_proof_l54_5465

-- Define the original prices
def original_shirt_price : ℝ := 60
def original_jacket_price : ℝ := 90

-- Define the reduction rate
def reduction_rate : ℝ := 0.2

-- Define the number of items bought
def num_shirts : ℕ := 5
def num_jackets : ℕ := 10

-- Define the total cost after reduction
def total_cost : ℝ := 960

-- Theorem statement
theorem shirt_price_proof :
  (1 - reduction_rate) * (num_shirts * original_shirt_price + num_jackets * original_jacket_price) = total_cost :=
by sorry

end NUMINAMATH_CALUDE_shirt_price_proof_l54_5465


namespace NUMINAMATH_CALUDE_sin_transformation_l54_5441

theorem sin_transformation (x : ℝ) : 
  2 * Real.sin (3 * x + π / 6) = 2 * Real.sin ((x + π / 6) / 3) := by
  sorry

end NUMINAMATH_CALUDE_sin_transformation_l54_5441


namespace NUMINAMATH_CALUDE_divisibility_of_2_pow_55_plus_1_l54_5443

theorem divisibility_of_2_pow_55_plus_1 : 
  ∃ k : ℤ, 2^55 + 1 = 33 * k := by
  sorry

end NUMINAMATH_CALUDE_divisibility_of_2_pow_55_plus_1_l54_5443


namespace NUMINAMATH_CALUDE_difference_of_squares_330_270_l54_5439

theorem difference_of_squares_330_270 : 330^2 - 270^2 = 36000 := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_330_270_l54_5439


namespace NUMINAMATH_CALUDE_circle_center_correct_l54_5475

/-- The equation of a circle in the form ax² + bx + cy² + dy + e = 0 -/
structure CircleEquation where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ
  e : ℝ

/-- The center of a circle -/
structure CircleCenter where
  x : ℝ
  y : ℝ

/-- Given a circle equation, find its center -/
def findCircleCenter (eq : CircleEquation) : CircleCenter :=
  sorry

theorem circle_center_correct :
  let eq := CircleEquation.mk 1 (-6) 1 10 (-7)
  findCircleCenter eq = CircleCenter.mk 3 (-5) :=
by sorry

end NUMINAMATH_CALUDE_circle_center_correct_l54_5475


namespace NUMINAMATH_CALUDE_embankment_project_additional_days_l54_5435

/-- Represents the embankment construction project -/
structure EmbankmentProject where
  initial_workers : ℕ
  initial_days : ℕ
  reassigned_workers : ℕ
  reassignment_day : ℕ
  productivity_factor : ℚ

/-- Calculates the additional days needed to complete the project -/
def additional_days_needed (project : EmbankmentProject) : ℚ :=
  let initial_rate : ℚ := 1 / (project.initial_workers * project.initial_days)
  let work_done_before_reassignment : ℚ := project.initial_workers * initial_rate * project.reassignment_day
  let remaining_work : ℚ := 1 - work_done_before_reassignment
  let remaining_workers : ℕ := project.initial_workers - project.reassigned_workers
  let new_rate : ℚ := initial_rate * project.productivity_factor
  let total_days : ℚ := project.reassignment_day + (remaining_work / (remaining_workers * new_rate))
  total_days - project.reassignment_day

/-- Theorem stating the additional days needed for the specific project -/
theorem embankment_project_additional_days :
  let project : EmbankmentProject := {
    initial_workers := 100,
    initial_days := 5,
    reassigned_workers := 40,
    reassignment_day := 2,
    productivity_factor := 3/4
  }
  additional_days_needed project = 53333 / 1000 := by sorry

end NUMINAMATH_CALUDE_embankment_project_additional_days_l54_5435


namespace NUMINAMATH_CALUDE_evaluate_expression_l54_5403

theorem evaluate_expression : -(16 / 2 * 12 - 75 + 4 * (2 * 5) + 25) = -86 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l54_5403


namespace NUMINAMATH_CALUDE_fourth_and_fifth_sum_is_138_l54_5497

/-- Represents the number of dots in a hexagonal layer -/
def hexagon_dots : ℕ → ℕ
| 0 => 1  -- Central dot
| 1 => 7  -- Second hexagon
| 2 => 19 -- Third hexagon
| n + 3 => 
  let prev_layer := hexagon_dots n - hexagon_dots (n-1)
  let two_layers_ago := hexagon_dots (n-1) - hexagon_dots (n-2)
  hexagon_dots (n+2) + prev_layer + 6 + 2 * two_layers_ago

/-- The sum of dots in the fourth and fifth hexagons -/
def fourth_and_fifth_sum : ℕ := hexagon_dots 3 + hexagon_dots 4

theorem fourth_and_fifth_sum_is_138 : fourth_and_fifth_sum = 138 := by
  sorry

end NUMINAMATH_CALUDE_fourth_and_fifth_sum_is_138_l54_5497


namespace NUMINAMATH_CALUDE_square_difference_equality_l54_5459

theorem square_difference_equality : 1010^2 - 990^2 - 1005^2 + 995^2 = 20000 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_equality_l54_5459


namespace NUMINAMATH_CALUDE_prob_A_squared_zero_correct_l54_5447

/-- Probability that A² = O for an n × n matrix A with exactly two 1's -/
def prob_A_squared_zero (n : ℕ) : ℚ :=
  if n < 2 then 0
  else (n - 1) * (n - 2) / (n * (n + 1))

/-- Theorem stating the probability that A² = O for the given conditions -/
theorem prob_A_squared_zero_correct (n : ℕ) (h : n ≥ 2) :
  prob_A_squared_zero n = (n - 1) * (n - 2) / (n * (n + 1)) :=
sorry

end NUMINAMATH_CALUDE_prob_A_squared_zero_correct_l54_5447


namespace NUMINAMATH_CALUDE_defective_clock_correct_time_fraction_l54_5493

/-- Represents a 12-hour digital clock with a defect that displays '7' instead of '1'. -/
structure DefectiveClock where
  /-- The number of hours in the clock cycle -/
  hours : Nat
  /-- The number of minutes in an hour -/
  minutes_per_hour : Nat
  /-- The number of hours displayed correctly -/
  correct_hours : Nat
  /-- The number of minutes displayed correctly in each hour -/
  correct_minutes : Nat

/-- The fraction of the day that the defective clock displays the correct time -/
def correct_time_fraction (clock : DefectiveClock) : ℚ :=
  (clock.correct_hours : ℚ) / clock.hours * (clock.correct_minutes : ℚ) / clock.minutes_per_hour

/-- Theorem stating that the fraction of the day the defective clock displays the correct time is 1/2 -/
theorem defective_clock_correct_time_fraction :
  ∃ (clock : DefectiveClock),
    clock.hours = 12 ∧
    clock.minutes_per_hour = 60 ∧
    clock.correct_hours = 8 ∧
    clock.correct_minutes = 45 ∧
    correct_time_fraction clock = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_defective_clock_correct_time_fraction_l54_5493
