import Mathlib

namespace NUMINAMATH_CALUDE_geometric_sequence_sine_values_l2264_226468

theorem geometric_sequence_sine_values (α β γ : Real) :
  (β = 2 * α ∧ γ = 4 * α) →  -- geometric sequence condition
  (0 ≤ α ∧ α ≤ 2 * Real.pi) →  -- α ∈ [0, 2π]
  ((Real.sin β) / (Real.sin α) = (Real.sin γ) / (Real.sin β)) →  -- sine values form geometric sequence
  ((α = 2 * Real.pi / 3 ∧ β = 4 * Real.pi / 3 ∧ γ = 8 * Real.pi / 3) ∨
   (α = 4 * Real.pi / 3 ∧ β = 8 * Real.pi / 3 ∧ γ = 16 * Real.pi / 3)) :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_sine_values_l2264_226468


namespace NUMINAMATH_CALUDE_positive_numbers_l2264_226485

theorem positive_numbers (a b c : ℝ) 
  (sum_positive : a + b + c > 0)
  (sum_products_positive : a * b + b * c + c * a > 0)
  (product_positive : a * b * c > 0) :
  a > 0 ∧ b > 0 ∧ c > 0 := by
  sorry

end NUMINAMATH_CALUDE_positive_numbers_l2264_226485


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_l2264_226452

theorem quadratic_equation_roots (a : ℝ) : 
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ 
    x₁^2 + a*x₁ + 5 = 0 ∧ 
    x₂^2 + a*x₂ + 5 = 0 ∧ 
    x₁^2 + 250/(19*x₂^3) = x₂^2 + 250/(19*x₁^3)) → 
  a = 10 := by sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_l2264_226452


namespace NUMINAMATH_CALUDE_brilliant_permutations_l2264_226499

def word := "BRILLIANT"

/-- The number of permutations of the letters in 'BRILLIANT' where no two adjacent letters are the same -/
def valid_permutations : ℕ :=
  Nat.factorial 9 / (Nat.factorial 2 * Nat.factorial 2) -
  (Nat.factorial 8 / Nat.factorial 2 +
   Nat.factorial 8 / Nat.factorial 2 -
   Nat.factorial 7)

theorem brilliant_permutations :
  valid_permutations = 55440 :=
sorry

end NUMINAMATH_CALUDE_brilliant_permutations_l2264_226499


namespace NUMINAMATH_CALUDE_rotated_logarithm_function_l2264_226460

-- Define the logarithm base 10 function
noncomputable def lg (x : ℝ) := Real.log x / Real.log 10

-- Define the rotation transformation
def rotate_counterclockwise_pi_over_2 (x y : ℝ) : ℝ × ℝ := (y, -x)

-- State the theorem
theorem rotated_logarithm_function (f : ℝ → ℝ) :
  (∀ x, rotate_counterclockwise_pi_over_2 (f x) x = (lg (x + 1), x)) →
  (∀ x, f x = 10^(-x) - 1) :=
by sorry

end NUMINAMATH_CALUDE_rotated_logarithm_function_l2264_226460


namespace NUMINAMATH_CALUDE_sum_of_averages_equals_155_l2264_226418

def even_integers_to_100 : List ℕ := List.range 51 |> List.map (· * 2)
def even_integers_to_50 : List ℕ := List.range 26 |> List.map (· * 2)
def even_perfect_squares_to_250 : List ℕ := [0, 4, 16, 36, 64, 100, 144, 196]

def average (l : List ℕ) : ℚ :=
  (l.sum : ℚ) / l.length

theorem sum_of_averages_equals_155 :
  average even_integers_to_100 +
  average even_integers_to_50 +
  average even_perfect_squares_to_250 = 155 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_averages_equals_155_l2264_226418


namespace NUMINAMATH_CALUDE_ellipse_max_major_axis_l2264_226456

theorem ellipse_max_major_axis 
  (a b : ℝ) 
  (h_ab : a > b ∧ b > 0) 
  (e : ℝ) 
  (h_e : e ∈ Set.Icc (1/2) (Real.sqrt 2 / 2)) 
  (h_perp : ∀ (x₁ y₁ x₂ y₂ : ℝ), 
    x₁^2/a^2 + y₁^2/b^2 = 1 → 
    y₁ = -x₁ + 1 → 
    x₂^2/a^2 + y₂^2/b^2 = 1 → 
    y₂ = -x₂ + 1 → 
    x₁*x₂ + y₁*y₂ = 0) 
  (h_ecc : e^2 = 1 - b^2/a^2) :
  ∃ (max_axis : ℝ), max_axis = Real.sqrt 6 ∧ 
    ∀ (axis : ℝ), axis = 2*a → axis ≤ max_axis :=
sorry

end NUMINAMATH_CALUDE_ellipse_max_major_axis_l2264_226456


namespace NUMINAMATH_CALUDE_solution_product_l2264_226484

theorem solution_product (p q : ℝ) : 
  (p - 4) * (3 * p + 11) = p^2 - 19 * p + 72 →
  (q - 4) * (3 * q + 11) = q^2 - 19 * q + 72 →
  p ≠ q →
  (p + 4) * (q + 4) = -78 :=
by
  sorry

end NUMINAMATH_CALUDE_solution_product_l2264_226484


namespace NUMINAMATH_CALUDE_min_value_of_f_l2264_226457

def f (x a : ℝ) : ℝ := |x - a| + |x - 15| + |x - (a + 15)|

theorem min_value_of_f (a : ℝ) (h1 : 0 < a) (h2 : a < 15) :
  ∃ Q : ℝ, Q = 15 ∧ ∀ x : ℝ, a ≤ x → x ≤ 15 → f x a ≥ Q :=
sorry

end NUMINAMATH_CALUDE_min_value_of_f_l2264_226457


namespace NUMINAMATH_CALUDE_sum_of_a_and_b_l2264_226486

theorem sum_of_a_and_b (a b : ℚ) 
  (eq1 : 2 * a + 5 * b = 31) 
  (eq2 : 4 * a + 3 * b = 35) : 
  a + b = 68 / 7 := by
sorry

end NUMINAMATH_CALUDE_sum_of_a_and_b_l2264_226486


namespace NUMINAMATH_CALUDE_sophist_statements_l2264_226426

/-- Represents the types of inhabitants on the Isle of Logic. -/
inductive Inhabitant
  | Knight
  | Liar
  | Sophist

/-- The total number of knights on the island. -/
def num_knights : ℕ := 40

/-- The total number of liars on the island. -/
def num_liars : ℕ := 25

/-- A function that determines if a statement about the number of knights is valid for a sophist. -/
def valid_knight_statement (n : ℕ) : Prop :=
  n ≠ num_knights ∧ n = num_knights

/-- A function that determines if a statement about the number of liars is valid for a sophist. -/
def valid_liar_statement (n : ℕ) : Prop :=
  n ≠ num_liars ∧ n = num_liars + 1

/-- The main theorem stating that the only valid sophist statements are 40 knights and 26 liars. -/
theorem sophist_statements :
  (∃! k : ℕ, valid_knight_statement k) ∧
  (∃! l : ℕ, valid_liar_statement l) ∧
  valid_knight_statement 40 ∧
  valid_liar_statement 26 := by
  sorry

end NUMINAMATH_CALUDE_sophist_statements_l2264_226426


namespace NUMINAMATH_CALUDE_no_negative_values_range_monotonicity_when_even_l2264_226496

-- Define the function f(x) with parameter m
def f (m : ℝ) (x : ℝ) : ℝ := 4 * x^2 + (m - 2) * x + 1

-- Theorem for part (I)
theorem no_negative_values_range (m : ℝ) :
  (∀ x, f m x ≥ 0) ↔ m ∈ Set.Icc (-2 : ℝ) 6 :=
sorry

-- Theorem for part (II)
theorem monotonicity_when_even (m : ℝ) :
  (∀ x, f m x = f m (-x)) →
  (∀ x y, x ≤ 0 ∧ y ≤ x → f m y ≥ f m x) ∧
  (∀ x y, 0 ≤ x ∧ x ≤ y → f m x ≤ f m y) :=
sorry

end NUMINAMATH_CALUDE_no_negative_values_range_monotonicity_when_even_l2264_226496


namespace NUMINAMATH_CALUDE_A_div_B_eq_37_l2264_226413

-- Define the series A
def A : ℝ := sorry

-- Define the series B
def B : ℝ := sorry

-- Theorem stating the relationship between A and B
theorem A_div_B_eq_37 : A / B = 37 := by sorry

end NUMINAMATH_CALUDE_A_div_B_eq_37_l2264_226413


namespace NUMINAMATH_CALUDE_f_extrema_l2264_226440

-- Define the function f
def f (x y : ℝ) : ℝ := 2 * x^2 - 2 * y^2

-- Define the disk
def disk (x y : ℝ) : Prop := x^2 + y^2 ≤ 9

-- Theorem statement
theorem f_extrema :
  (∃ x y : ℝ, disk x y ∧ f x y = 18) ∧
  (∃ x y : ℝ, disk x y ∧ f x y = -18) ∧
  (∀ x y : ℝ, disk x y → f x y ≤ 18) ∧
  (∀ x y : ℝ, disk x y → f x y ≥ -18) := by
  sorry

end NUMINAMATH_CALUDE_f_extrema_l2264_226440


namespace NUMINAMATH_CALUDE_nested_fraction_evaluation_l2264_226401

theorem nested_fraction_evaluation : 
  (1 : ℚ) / (3 - 1 / (3 - 1 / (3 - 1 / 3))) = 8 / 21 := by
  sorry

end NUMINAMATH_CALUDE_nested_fraction_evaluation_l2264_226401


namespace NUMINAMATH_CALUDE_geometric_number_difference_l2264_226441

/-- A 4-digit number is geometric if it has 4 distinct digits forming a geometric sequence from left to right. -/
def IsGeometric (n : ℕ) : Prop :=
  1000 ≤ n ∧ n < 10000 ∧
  ∃ a b c d r : ℕ,
    n = a * 1000 + b * 100 + c * 10 + d ∧
    a ≠ 0 ∧ a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
    b = a * r ∧ c = a * r^2 ∧ d = a * r^3

/-- The largest 4-digit geometric number -/
def LargestGeometric : ℕ := 9648

/-- The smallest 4-digit geometric number -/
def SmallestGeometric : ℕ := 1248

theorem geometric_number_difference :
  IsGeometric LargestGeometric ∧
  IsGeometric SmallestGeometric ∧
  (∀ n : ℕ, IsGeometric n → SmallestGeometric ≤ n ∧ n ≤ LargestGeometric) ∧
  LargestGeometric - SmallestGeometric = 8400 := by
  sorry

end NUMINAMATH_CALUDE_geometric_number_difference_l2264_226441


namespace NUMINAMATH_CALUDE_age_difference_l2264_226448

/-- Proves that the difference between Rahul's and Sachin's ages is 9 years -/
theorem age_difference (sachin_age rahul_age : ℝ) : 
  sachin_age = 31.5 → 
  sachin_age / rahul_age = 7 / 9 → 
  rahul_age - sachin_age = 9 :=
by
  sorry


end NUMINAMATH_CALUDE_age_difference_l2264_226448


namespace NUMINAMATH_CALUDE_max_common_segment_for_coprime_l2264_226493

/-- The maximum length of the common initial segment of two sequences with coprime periods -/
def max_common_segment (m n : ℕ) : ℕ :=
  m + n - 2

/-- Theorem: For coprime positive integers m and n, the maximum length of the common
    initial segment of two sequences with periods m and n respectively is m + n - 2 -/
theorem max_common_segment_for_coprime (m n : ℕ) (h_pos_m : 0 < m) (h_pos_n : 0 < n) 
    (h_coprime : Nat.Coprime m n) : 
  max_common_segment m n = m + n - 2 := by
  sorry

#check max_common_segment_for_coprime

end NUMINAMATH_CALUDE_max_common_segment_for_coprime_l2264_226493


namespace NUMINAMATH_CALUDE_sqrt_six_div_sqrt_three_eq_sqrt_two_l2264_226427

theorem sqrt_six_div_sqrt_three_eq_sqrt_two :
  Real.sqrt 6 / Real.sqrt 3 = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_six_div_sqrt_three_eq_sqrt_two_l2264_226427


namespace NUMINAMATH_CALUDE_triangle_inequality_cube_l2264_226415

/-- Given an arbitrary triangle with sides a, b, and c, prove that a^3 + b^3 + 3abc > c^3. -/
theorem triangle_inequality_cube (a b c : ℝ) (h : 0 < a ∧ 0 < b ∧ 0 < c) 
  (triangle_ineq : a + b > c ∧ b + c > a ∧ c + a > b) : 
  a^3 + b^3 + 3*a*b*c > c^3 := by
  sorry

end NUMINAMATH_CALUDE_triangle_inequality_cube_l2264_226415


namespace NUMINAMATH_CALUDE_project_time_ratio_l2264_226463

/-- Given a project where three people (Pat, Kate, and Mark) charged time, 
    prove that the ratio of time charged by Pat to Mark is 1:3 -/
theorem project_time_ratio (total_hours kate_hours pat_hours mark_hours : ℕ) : 
  total_hours = 216 →
  pat_hours = 2 * kate_hours →
  mark_hours = kate_hours + 120 →
  total_hours = kate_hours + pat_hours + mark_hours →
  pat_hours * 3 = mark_hours := by
  sorry

end NUMINAMATH_CALUDE_project_time_ratio_l2264_226463


namespace NUMINAMATH_CALUDE_garden_perimeter_proof_l2264_226474

/-- The perimeter of a rectangular garden with given length and breadth. -/
def garden_perimeter (length breadth : ℝ) : ℝ :=
  2 * length + 2 * breadth

/-- Theorem: The perimeter of a rectangular garden with length 375 m and breadth 100 m is 950 m. -/
theorem garden_perimeter_proof :
  garden_perimeter 375 100 = 950 := by
  sorry

end NUMINAMATH_CALUDE_garden_perimeter_proof_l2264_226474


namespace NUMINAMATH_CALUDE_power_two_geq_double_plus_two_l2264_226411

theorem power_two_geq_double_plus_two (n : ℕ) (h : n ≥ 3) : 2^n ≥ 2*(n+1) := by
  sorry

end NUMINAMATH_CALUDE_power_two_geq_double_plus_two_l2264_226411


namespace NUMINAMATH_CALUDE_min_value_of_2x_plus_y_min_value_is_one_min_value_exists_l2264_226476

theorem min_value_of_2x_plus_y (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : 2 * x + y + 2 * x * y = 5/4) : 
  ∀ a b : ℝ, a > 0 → b > 0 → 2 * a + b + 2 * a * b = 5/4 → 2 * x + y ≤ 2 * a + b :=
by sorry

theorem min_value_is_one (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : 2 * x + y + 2 * x * y = 5/4) : 
  2 * x + y ≥ 1 :=
by sorry

theorem min_value_exists (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : 2 * x + y + 2 * x * y = 5/4) : 
  ∃ a b : ℝ, a > 0 ∧ b > 0 ∧ 2 * a + b + 2 * a * b = 5/4 ∧ 2 * a + b = 1 :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_2x_plus_y_min_value_is_one_min_value_exists_l2264_226476


namespace NUMINAMATH_CALUDE_probability_one_white_one_red_l2264_226430

theorem probability_one_white_one_red (total : ℕ) (white : ℕ) (red : ℕ) :
  total = white + red →
  total = 15 →
  white = 10 →
  red = 5 →
  (white.choose 1 * red.choose 1 : ℚ) / total.choose 2 = 10 / 21 := by
  sorry

end NUMINAMATH_CALUDE_probability_one_white_one_red_l2264_226430


namespace NUMINAMATH_CALUDE_three_from_eight_l2264_226466

theorem three_from_eight : Nat.choose 8 3 = 56 := by
  sorry

end NUMINAMATH_CALUDE_three_from_eight_l2264_226466


namespace NUMINAMATH_CALUDE_two_numbers_with_sum_and_gcd_l2264_226459

theorem two_numbers_with_sum_and_gcd : ∃ (a b : ℕ), a + b = 168 ∧ Nat.gcd a b = 24 := by
  sorry

end NUMINAMATH_CALUDE_two_numbers_with_sum_and_gcd_l2264_226459


namespace NUMINAMATH_CALUDE_sum_x_y_equals_2700_l2264_226489

theorem sum_x_y_equals_2700 (x y : ℝ) : 
  (0.9 * 600 = 0.5 * x) → 
  (0.6 * x = 0.4 * y) → 
  x + y = 2700 := by
sorry

end NUMINAMATH_CALUDE_sum_x_y_equals_2700_l2264_226489


namespace NUMINAMATH_CALUDE_smallest_integer_fraction_l2264_226410

theorem smallest_integer_fraction (y : ℤ) : (7 : ℚ) / 11 < (y : ℚ) / 17 ↔ 11 ≤ y := by sorry

end NUMINAMATH_CALUDE_smallest_integer_fraction_l2264_226410


namespace NUMINAMATH_CALUDE_upstream_downstream_time_relation_stream_speed_is_twelve_l2264_226446

/-- Represents the speed of a boat in still water -/
def boat_speed : ℝ := 36

/-- Represents the speed of the stream -/
def stream_speed : ℝ := 12

/-- The time taken to row upstream is twice the time taken to row downstream -/
theorem upstream_downstream_time_relation (d : ℝ) (h : d > 0) :
  d / (boat_speed - stream_speed) = 2 * (d / (boat_speed + stream_speed)) :=
by sorry

/-- Proves that the stream speed is 12 kmph given the conditions -/
theorem stream_speed_is_twelve :
  stream_speed = 12 :=
by sorry

end NUMINAMATH_CALUDE_upstream_downstream_time_relation_stream_speed_is_twelve_l2264_226446


namespace NUMINAMATH_CALUDE_square_of_difference_l2264_226405

theorem square_of_difference (y : ℝ) (h : y^2 ≥ 49) :
  (7 - Real.sqrt (y^2 - 49))^2 = y^2 - 14 * Real.sqrt (y^2 - 49) := by
  sorry

end NUMINAMATH_CALUDE_square_of_difference_l2264_226405


namespace NUMINAMATH_CALUDE_conference_handshakes_l2264_226431

/-- The number of handshakes in a conference with non-committee members shaking hands --/
def max_handshakes (total_participants : ℕ) (committee_members : ℕ) : ℕ :=
  let non_committee := total_participants - committee_members
  (non_committee * (non_committee - 1)) / 2

/-- Theorem stating the maximum number of handshakes in the given conference scenario --/
theorem conference_handshakes :
  max_handshakes 50 10 = 780 := by
  sorry

#eval max_handshakes 50 10

end NUMINAMATH_CALUDE_conference_handshakes_l2264_226431


namespace NUMINAMATH_CALUDE_stratified_sampling_business_personnel_l2264_226429

theorem stratified_sampling_business_personnel 
  (total_employees : ℕ) 
  (business_personnel : ℕ) 
  (sample_size : ℕ) 
  (h1 : total_employees = 160) 
  (h2 : business_personnel = 120) 
  (h3 : sample_size = 20) :
  (business_personnel * sample_size) / total_employees = 15 := by
sorry

end NUMINAMATH_CALUDE_stratified_sampling_business_personnel_l2264_226429


namespace NUMINAMATH_CALUDE_error_percentage_division_vs_multiplication_l2264_226458

theorem error_percentage_division_vs_multiplication :
  ∀ x : ℝ, x ≠ 0 →
  (((5 * x - x / 5) / (5 * x)) * 100 : ℝ) = 96 := by
  sorry

end NUMINAMATH_CALUDE_error_percentage_division_vs_multiplication_l2264_226458


namespace NUMINAMATH_CALUDE_saree_price_calculation_l2264_226462

theorem saree_price_calculation (P : ℝ) : 
  P * (1 - 0.20) * (1 - 0.10) = 108 → P = 150 := by
  sorry

end NUMINAMATH_CALUDE_saree_price_calculation_l2264_226462


namespace NUMINAMATH_CALUDE_michael_gave_two_crates_l2264_226480

/-- Calculates the number of crates Michael gave to Susan -/
def crates_given_to_susan (crates_tuesday : ℕ) (crates_thursday : ℕ) (eggs_per_crate : ℕ) (eggs_remaining : ℕ) : ℕ :=
  let total_crates := crates_tuesday + crates_thursday
  let total_eggs := total_crates * eggs_per_crate
  (total_eggs - eggs_remaining) / eggs_per_crate

theorem michael_gave_two_crates :
  crates_given_to_susan 6 5 30 270 = 2 := by
  sorry

end NUMINAMATH_CALUDE_michael_gave_two_crates_l2264_226480


namespace NUMINAMATH_CALUDE_songs_downloaded_l2264_226403

def internet_speed : ℝ := 20
def download_time : ℝ := 0.5
def song_size : ℝ := 5

theorem songs_downloaded : 
  ⌊(internet_speed * download_time * 3600) / song_size⌋ = 7200 := by sorry

end NUMINAMATH_CALUDE_songs_downloaded_l2264_226403


namespace NUMINAMATH_CALUDE_name_tag_paper_perimeter_l2264_226425

theorem name_tag_paper_perimeter :
  ∀ (num_students : ℕ) (tag_side_length : ℝ) (paper_width : ℝ) (unused_width : ℝ),
    num_students = 24 →
    tag_side_length = 4 →
    paper_width = 34 →
    unused_width = 2 →
    (paper_width - unused_width) / tag_side_length * tag_side_length * 
      (num_students / ((paper_width - unused_width) / tag_side_length)) = 
      paper_width - unused_width →
    2 * (paper_width + (num_students / ((paper_width - unused_width) / tag_side_length)) * tag_side_length) = 92 := by
  sorry

end NUMINAMATH_CALUDE_name_tag_paper_perimeter_l2264_226425


namespace NUMINAMATH_CALUDE_cost_calculation_l2264_226453

/-- The cost of buying pens and notebooks -/
def total_cost (pen_price notebook_price : ℝ) : ℝ :=
  5 * pen_price + 8 * notebook_price

/-- Theorem: The total cost of 5 pens at 'a' yuan each and 8 notebooks at 'b' yuan each is 5a + 8b yuan -/
theorem cost_calculation (a b : ℝ) : total_cost a b = 5 * a + 8 * b := by
  sorry

end NUMINAMATH_CALUDE_cost_calculation_l2264_226453


namespace NUMINAMATH_CALUDE_difference_of_squares_factorization_l2264_226488

theorem difference_of_squares_factorization (a : ℝ) : a^2 - 6 = (a + Real.sqrt 6) * (a - Real.sqrt 6) := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_factorization_l2264_226488


namespace NUMINAMATH_CALUDE_sequence_monotonicity_l2264_226407

theorem sequence_monotonicity (b : ℝ) :
  (∀ n : ℕ, n^2 + b*n < (n+1)^2 + b*(n+1)) ↔ b > -3 :=
sorry

end NUMINAMATH_CALUDE_sequence_monotonicity_l2264_226407


namespace NUMINAMATH_CALUDE_surface_area_of_sliced_solid_l2264_226432

/-- A right prism with equilateral triangular bases -/
structure RightPrism where
  height : ℝ
  base_side : ℝ

/-- Midpoints of edges in the prism -/
structure Midpoints where
  X : ℝ × ℝ × ℝ
  Y : ℝ × ℝ × ℝ
  Z : ℝ × ℝ × ℝ

/-- The solid formed by slicing off a part of the prism -/
def SlicedSolid (p : RightPrism) (m : Midpoints) : ℝ × ℝ × ℝ × ℝ := sorry

/-- Surface area of the sliced solid -/
def surfaceArea (s : ℝ × ℝ × ℝ × ℝ) : ℝ := sorry

/-- Main theorem -/
theorem surface_area_of_sliced_solid (p : RightPrism) (m : Midpoints) :
  p.height = 20 ∧ p.base_side = 10 →
  surfaceArea (SlicedSolid p m) = 100 + 25 * Real.sqrt 3 / 4 := by
  sorry

end NUMINAMATH_CALUDE_surface_area_of_sliced_solid_l2264_226432


namespace NUMINAMATH_CALUDE_complex_addition_simplification_l2264_226417

theorem complex_addition_simplification :
  (-5 + 3*Complex.I) + (2 - 7*Complex.I) = -3 - 4*Complex.I :=
by sorry

end NUMINAMATH_CALUDE_complex_addition_simplification_l2264_226417


namespace NUMINAMATH_CALUDE_car_distribution_l2264_226483

/-- The number of cars produced annually by American carmakers -/
def total_cars : ℕ := 5650000

/-- The number of car suppliers -/
def num_suppliers : ℕ := 5

/-- The number of cars received by the first supplier -/
def first_supplier : ℕ := 1000000

/-- The number of cars received by the second supplier -/
def second_supplier : ℕ := first_supplier + 500000

/-- The number of cars received by the third supplier -/
def third_supplier : ℕ := first_supplier + second_supplier

/-- The number of cars received by each of the fourth and fifth suppliers -/
def fourth_fifth_supplier : ℕ := (total_cars - (first_supplier + second_supplier + third_supplier)) / 2

theorem car_distribution :
  fourth_fifth_supplier = 325000 :=
sorry

end NUMINAMATH_CALUDE_car_distribution_l2264_226483


namespace NUMINAMATH_CALUDE_student_skills_l2264_226470

theorem student_skills (total : ℕ) (chess_unable : ℕ) (puzzle_unable : ℕ) (code_unable : ℕ) :
  total = 120 →
  chess_unable = 50 →
  puzzle_unable = 75 →
  code_unable = 40 →
  (∃ (two_skills : ℕ), two_skills = 75 ∧
    two_skills = (total - chess_unable) + (total - puzzle_unable) + (total - code_unable) - total) :=
by sorry

end NUMINAMATH_CALUDE_student_skills_l2264_226470


namespace NUMINAMATH_CALUDE_derivative_at_pi_third_l2264_226438

theorem derivative_at_pi_third (f : ℝ → ℝ) (hf : Differentiable ℝ f) 
  (h : ∀ x, f x = x^2 * (deriv f (π/3)) + Real.sin x) : 
  deriv f (π/3) = 3 / (6 - 4*π) := by
  sorry

end NUMINAMATH_CALUDE_derivative_at_pi_third_l2264_226438


namespace NUMINAMATH_CALUDE_only_B_is_true_l2264_226400

-- Define a point in 2D space
structure Point2D where
  x : ℝ
  y : ℝ

-- Define a line in 2D space
structure Line2D where
  a : ℝ
  b : ℝ
  c : ℝ

-- Proposition A
def propA (P₀ : Point2D) (l : Line2D) : Prop :=
  ∃ k : ℝ, ∀ x y : ℝ, l.a * x + l.b * y + l.c = 0 ↔ y - P₀.y = k * (x - P₀.x)

-- Proposition B
def propB (P₁ P₂ : Point2D) (l : Line2D) : Prop :=
  P₁ ≠ P₂ → ∀ x y : ℝ, l.a * x + l.b * y + l.c = 0 ↔ 
    (y - P₁.y) * (P₂.x - P₁.x) = (x - P₁.x) * (P₂.y - P₁.y)

-- Proposition C
def propC (l : Line2D) : Prop :=
  ∃ a b : ℝ, a ≠ 0 ∧ b ≠ 0 ∧ ∀ x y : ℝ, l.a * x + l.b * y + l.c = 0 ↔ x / a + y / b = 1

-- Proposition D
def propD (b : ℝ) (l : Line2D) : Prop :=
  ∃ k : ℝ, ∀ x y : ℝ, l.a * x + l.b * y + l.c = 0 ↔ y = k * x + b

theorem only_B_is_true :
  (∃ P₀ : Point2D, ∀ l : Line2D, propA P₀ l) = false ∧
  (∀ P₁ P₂ : Point2D, ∀ l : Line2D, propB P₁ P₂ l) = true ∧
  (∀ l : Line2D, propC l) = false ∧
  (∃ b : ℝ, ∀ l : Line2D, propD b l) = false :=
sorry

end NUMINAMATH_CALUDE_only_B_is_true_l2264_226400


namespace NUMINAMATH_CALUDE_stone_piles_problem_l2264_226475

theorem stone_piles_problem (x y : ℕ) : 
  (y + 100 = 2 * (x - 100)) → 
  (∃ z : ℕ, x + z = 5 * (y - z)) → 
  x ≥ 170 → 
  (x = 170 ∧ y = 40) ∨ x > 170 :=
sorry

end NUMINAMATH_CALUDE_stone_piles_problem_l2264_226475


namespace NUMINAMATH_CALUDE_inequality_proof_l2264_226428

theorem inequality_proof (x y : ℝ) (h : x^4 + y^4 ≥ 2) :
  |x^16 - y^16| + 4 * x^8 * y^8 ≥ 4 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2264_226428


namespace NUMINAMATH_CALUDE_unique_solution_quadratic_l2264_226497

theorem unique_solution_quadratic (a : ℝ) : 
  (∃! x : ℝ, a * x^2 + 2 * x - a = 0) ↔ a = 0 := by
sorry

end NUMINAMATH_CALUDE_unique_solution_quadratic_l2264_226497


namespace NUMINAMATH_CALUDE_neighbor_cans_is_46_l2264_226454

/-- Represents the recycling problem Collin faces --/
structure RecyclingProblem where
  /-- The amount earned per aluminum can in dollars --/
  earnings_per_can : ℚ
  /-- The number of cans found at home --/
  cans_at_home : ℕ
  /-- The factor by which the number of cans at grandparents' house exceeds those at home --/
  grandparents_factor : ℕ
  /-- The number of cans brought by dad from the office --/
  cans_from_office : ℕ
  /-- The amount Collin has to put into savings in dollars --/
  savings_amount : ℚ

/-- Calculates the number of cans Collin's neighbor gave him --/
def neighbor_cans (p : RecyclingProblem) : ℕ :=
  sorry

/-- Theorem stating that the number of cans Collin's neighbor gave him is 46 --/
theorem neighbor_cans_is_46 (p : RecyclingProblem)
  (h1 : p.earnings_per_can = 1/4)
  (h2 : p.cans_at_home = 12)
  (h3 : p.grandparents_factor = 3)
  (h4 : p.cans_from_office = 250)
  (h5 : p.savings_amount = 43) :
  neighbor_cans p = 46 :=
  sorry

end NUMINAMATH_CALUDE_neighbor_cans_is_46_l2264_226454


namespace NUMINAMATH_CALUDE_f_derivative_at_2_l2264_226495

noncomputable def f (a b x : ℝ) : ℝ := a * Real.log x + b / x

theorem f_derivative_at_2 (a b : ℝ) :
  f a b 1 = -2 → (deriv (f a b)) 1 = 0 → (deriv (f a b)) 2 = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_f_derivative_at_2_l2264_226495


namespace NUMINAMATH_CALUDE_product_sum_in_base_l2264_226467

/-- Represents a number in base b --/
structure BaseNumber (b : ℕ) where
  value : ℕ

/-- Converts a base b number to its decimal representation --/
def to_decimal (b : ℕ) (n : BaseNumber b) : ℕ := sorry

/-- Converts a decimal number to its representation in base b --/
def from_decimal (b : ℕ) (n : ℕ) : BaseNumber b := sorry

/-- Multiplies two numbers in base b --/
def mul_base (b : ℕ) (x y : BaseNumber b) : BaseNumber b := sorry

/-- Adds two numbers in base b --/
def add_base (b : ℕ) (x y : BaseNumber b) : BaseNumber b := sorry

theorem product_sum_in_base (b : ℕ) 
  (h : mul_base b (mul_base b (from_decimal b 14) (from_decimal b 17)) (from_decimal b 18) = from_decimal b 6180) :
  add_base b (add_base b (from_decimal b 14) (from_decimal b 17)) (from_decimal b 18) = from_decimal b 53 := by
  sorry

end NUMINAMATH_CALUDE_product_sum_in_base_l2264_226467


namespace NUMINAMATH_CALUDE_female_fox_terriers_count_l2264_226408

theorem female_fox_terriers_count 
  (total_dogs : ℕ) 
  (total_females : ℕ) 
  (total_fox_terriers : ℕ) 
  (male_shih_tzus : ℕ) 
  (h1 : total_dogs = 2012)
  (h2 : total_females = 1110)
  (h3 : total_fox_terriers = 1506)
  (h4 : male_shih_tzus = 202) :
  total_fox_terriers - (total_dogs - total_females - male_shih_tzus) = 806 :=
by
  sorry

end NUMINAMATH_CALUDE_female_fox_terriers_count_l2264_226408


namespace NUMINAMATH_CALUDE_cl2_moles_required_l2264_226436

-- Define the reaction components
structure Reaction where
  c2h6 : ℝ
  cl2 : ℝ
  c2cl6 : ℝ
  hcl : ℝ

-- Define the balanced equation ratios
def balancedRatio : Reaction := {
  c2h6 := 1,
  cl2 := 6,
  c2cl6 := 1,
  hcl := 6
}

-- Define the given reaction
def givenReaction : Reaction := {
  c2h6 := 2,
  cl2 := 0,  -- This is what we need to prove
  c2cl6 := 2,
  hcl := 12
}

-- Theorem statement
theorem cl2_moles_required (r : Reaction) :
  r.c2h6 = givenReaction.c2h6 ∧
  r.c2cl6 = givenReaction.c2cl6 ∧
  r.hcl = givenReaction.hcl →
  r.cl2 = 12 :=
by sorry

end NUMINAMATH_CALUDE_cl2_moles_required_l2264_226436


namespace NUMINAMATH_CALUDE_thirty_sided_polygon_diagonals_l2264_226424

/-- The number of diagonals in a convex polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- A convex polygon with 30 sides has 405 diagonals -/
theorem thirty_sided_polygon_diagonals : num_diagonals 30 = 405 := by
  sorry

end NUMINAMATH_CALUDE_thirty_sided_polygon_diagonals_l2264_226424


namespace NUMINAMATH_CALUDE_greatest_five_digit_multiple_l2264_226487

theorem greatest_five_digit_multiple : ∃ n : ℕ, 
  n ≤ 99999 ∧ 
  n ≥ 10000 ∧
  n % 9 = 0 ∧ 
  n % 6 = 0 ∧ 
  n % 2 = 0 ∧
  ∀ m : ℕ, m ≤ 99999 ∧ m ≥ 10000 ∧ m % 9 = 0 ∧ m % 6 = 0 ∧ m % 2 = 0 → m ≤ n :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_greatest_five_digit_multiple_l2264_226487


namespace NUMINAMATH_CALUDE_saree_discount_problem_l2264_226439

/-- Calculates the second discount percentage given the original price, first discount percentage, and final sale price. -/
def second_discount_percentage (original_price first_discount_percent final_price : ℚ) : ℚ :=
  let price_after_first_discount := original_price * (1 - first_discount_percent / 100)
  let second_discount_amount := price_after_first_discount - final_price
  (second_discount_amount / price_after_first_discount) * 100

/-- Theorem stating that for the given conditions, the second discount percentage is 15%. -/
theorem saree_discount_problem :
  second_discount_percentage 450 20 306 = 15 := by sorry

end NUMINAMATH_CALUDE_saree_discount_problem_l2264_226439


namespace NUMINAMATH_CALUDE_direction_vector_of_line_l2264_226443

/-- Given a line l with equation x + y + 1 = 0, prove that (1, -1) is a direction vector of l. -/
theorem direction_vector_of_line (l : Set (ℝ × ℝ)) :
  (∀ p : ℝ × ℝ, p ∈ l ↔ p.1 + p.2 + 1 = 0) →
  ∃ t : ℝ, (1 + t, -1 + t) ∈ l := by sorry

end NUMINAMATH_CALUDE_direction_vector_of_line_l2264_226443


namespace NUMINAMATH_CALUDE_equation_solution_l2264_226472

theorem equation_solution : ∃ x : ℤ, x * (x + 2) + 1 = 36 ∧ x = 5 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2264_226472


namespace NUMINAMATH_CALUDE_value_of_M_l2264_226482

theorem value_of_M : ∀ M : ℝ, (0.25 * M = 0.55 * 1500) → M = 3300 := by
  sorry

end NUMINAMATH_CALUDE_value_of_M_l2264_226482


namespace NUMINAMATH_CALUDE_square_difference_equality_l2264_226434

theorem square_difference_equality : 1103^2 - 1097^2 - 1101^2 + 1099^2 = 8800 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_equality_l2264_226434


namespace NUMINAMATH_CALUDE_smallest_prime_sum_l2264_226435

theorem smallest_prime_sum (a b c d : ℕ) : 
  (Prime a ∧ Prime b ∧ Prime c ∧ Prime d) →
  (a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d) →
  Prime (a + b + c + d) →
  (Prime (a + b) ∧ Prime (a + c) ∧ Prime (a + d) ∧ 
   Prime (b + c) ∧ Prime (b + d) ∧ Prime (c + d)) →
  (Prime (a + b + c) ∧ Prime (a + b + d) ∧ 
   Prime (a + c + d) ∧ Prime (b + c + d)) →
  a + b + c + d ≥ 31 :=
by sorry

end NUMINAMATH_CALUDE_smallest_prime_sum_l2264_226435


namespace NUMINAMATH_CALUDE_probability_of_blank_in_specific_lottery_l2264_226492

/-- The probability of getting a blank in a lottery with prizes and blanks. -/
def probability_of_blank (prizes : ℕ) (blanks : ℕ) : ℚ :=
  blanks / (prizes + blanks)

/-- Theorem stating that the probability of getting a blank in a lottery 
    with 10 prizes and 25 blanks is 5/7. -/
theorem probability_of_blank_in_specific_lottery : 
  probability_of_blank 10 25 = 5 / 7 := by
  sorry

end NUMINAMATH_CALUDE_probability_of_blank_in_specific_lottery_l2264_226492


namespace NUMINAMATH_CALUDE_exponent_of_five_in_30_factorial_l2264_226461

theorem exponent_of_five_in_30_factorial : 
  ∃ k : ℕ, (30 : ℕ).factorial = 5^7 * k ∧ k % 5 ≠ 0 := by
  sorry

end NUMINAMATH_CALUDE_exponent_of_five_in_30_factorial_l2264_226461


namespace NUMINAMATH_CALUDE_sum_24_probability_l2264_226442

/-- The number of ways to achieve a sum of 24 with 10 fair standard 6-sided dice -/
def ways_to_sum_24 : ℕ := 817190

/-- The number of possible outcomes when throwing 10 fair standard 6-sided dice -/
def total_outcomes : ℕ := 6^10

/-- The probability of achieving a sum of 24 when throwing 10 fair standard 6-sided dice -/
def prob_sum_24 : ℚ := ways_to_sum_24 / total_outcomes

theorem sum_24_probability :
  ways_to_sum_24 = 817190 ∧
  total_outcomes = 6^10 ∧
  prob_sum_24 = ways_to_sum_24 / total_outcomes :=
sorry

end NUMINAMATH_CALUDE_sum_24_probability_l2264_226442


namespace NUMINAMATH_CALUDE_problem_solution_l2264_226491

theorem problem_solution : (2023^2 - 2023 - 1) / 2023 = 2022 - 1 / 2023 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l2264_226491


namespace NUMINAMATH_CALUDE_line_circle_intersection_l2264_226433

/-- A line in 2D space -/
structure Line where
  k : ℝ
  b : ℝ

/-- A circle in 2D space -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Check if a line and a circle intersect -/
def intersect (l : Line) (c : Circle) : Prop :=
  ∃ x y : ℝ, y = l.k * x + l.b ∧ (x - c.center.1)^2 + (y - c.center.2)^2 = c.radius^2

theorem line_circle_intersection (k : ℝ) :
  (∀ l : Line, l.b = 1 → intersect l (Circle.mk (0, 1) 1)) ∧
  (∃ l : Line, l.b ≠ 1 ∧ intersect l (Circle.mk (0, 1) 1)) :=
sorry

end NUMINAMATH_CALUDE_line_circle_intersection_l2264_226433


namespace NUMINAMATH_CALUDE_absolute_value_difference_l2264_226412

theorem absolute_value_difference (x p : ℝ) (h1 : |x - 5| = p) (h2 : x > 5) : x - p = 5 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_difference_l2264_226412


namespace NUMINAMATH_CALUDE_f_value_at_sqrt3_over_2_main_theorem_l2264_226406

-- Define the function f
def f (x : ℝ) : ℝ := 1 - 2 * x^2

-- Theorem statement
theorem f_value_at_sqrt3_over_2 : f (Real.sqrt 3 / 2) = -1/2 := by
  sorry

-- The main theorem that corresponds to the original problem
theorem main_theorem : 
  (∀ x, f (Real.sin x) = 1 - 2 * (Real.sin x)^2) → f (Real.sqrt 3 / 2) = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_f_value_at_sqrt3_over_2_main_theorem_l2264_226406


namespace NUMINAMATH_CALUDE_direct_proportion_only_f3_l2264_226421

/-- A function f: ℝ → ℝ is a direct proportion function if there exists a constant k such that f(x) = k * x for all x. -/
def is_direct_proportion (f : ℝ → ℝ) : Prop :=
  ∃ k : ℝ, ∀ x : ℝ, f x = k * x

/-- Function 1: f(x) = 3x - 4 -/
def f1 : ℝ → ℝ := λ x ↦ 3 * x - 4

/-- Function 2: f(x) = -2x + 1 -/
def f2 : ℝ → ℝ := λ x ↦ -2 * x + 1

/-- Function 3: f(x) = 3x -/
def f3 : ℝ → ℝ := λ x ↦ 3 * x

/-- Function 4: f(x) = 4 -/
def f4 : ℝ → ℝ := λ _ ↦ 4

theorem direct_proportion_only_f3 :
  ¬ is_direct_proportion f1 ∧
  ¬ is_direct_proportion f2 ∧
  is_direct_proportion f3 ∧
  ¬ is_direct_proportion f4 :=
sorry

end NUMINAMATH_CALUDE_direct_proportion_only_f3_l2264_226421


namespace NUMINAMATH_CALUDE_train_speed_calculation_l2264_226447

/-- Calculates the speed of a train crossing a bridge -/
theorem train_speed_calculation 
  (train_length : Real) 
  (bridge_length : Real) 
  (crossing_time : Real) 
  (h1 : train_length = 100) 
  (h2 : bridge_length = 275) 
  (h3 : crossing_time = 30) : 
  (train_length + bridge_length) / crossing_time * 3.6 = 45 := by
  sorry

end NUMINAMATH_CALUDE_train_speed_calculation_l2264_226447


namespace NUMINAMATH_CALUDE_base15_divisible_by_9_l2264_226477

/-- Converts a base-15 integer to decimal --/
def base15ToDecimal (digits : List Nat) : Nat :=
  digits.enum.foldr (fun (i, d) acc => acc + d * (15 ^ i)) 0

/-- The base-15 representation of 2643₁₅ --/
def base15Number : List Nat := [3, 4, 6, 2]

/-- Theorem stating that 2643₁₅ divided by 9 has a remainder of 0 --/
theorem base15_divisible_by_9 :
  (base15ToDecimal base15Number) % 9 = 0 := by
  sorry

end NUMINAMATH_CALUDE_base15_divisible_by_9_l2264_226477


namespace NUMINAMATH_CALUDE_max_a_value_l2264_226469

theorem max_a_value (a b : ℤ) (h1 : a > b) (h2 : b > 0) (h3 : a + b + a * b = 119) :
  a ≤ 59 ∧ ∃ (a' b' : ℤ), a' = 59 ∧ b' = 1 ∧ a' > b' ∧ b' > 0 ∧ a' + b' + a' * b' = 119 :=
sorry

end NUMINAMATH_CALUDE_max_a_value_l2264_226469


namespace NUMINAMATH_CALUDE_little_john_sweets_expenditure_l2264_226445

/-- Proof of the amount spent on sweets by Little John --/
theorem little_john_sweets_expenditure 
  (initial_amount : ℚ)
  (amount_per_friend : ℚ)
  (num_friends : ℕ)
  (final_amount : ℚ)
  (h1 : initial_amount = 20.10)
  (h2 : amount_per_friend = 1)
  (h3 : num_friends = 2)
  (h4 : final_amount = 17.05) :
  initial_amount - (↑num_friends * amount_per_friend) - final_amount = 1.05 := by
  sorry

#check little_john_sweets_expenditure

end NUMINAMATH_CALUDE_little_john_sweets_expenditure_l2264_226445


namespace NUMINAMATH_CALUDE_smallest_solution_for_floor_equation_l2264_226498

theorem smallest_solution_for_floor_equation :
  ∃ (x : ℝ), x > 0 ∧ 
  (⌊x^3⌋ : ℝ) - x * (⌊x⌋ : ℝ) = 18 ∧
  (∀ (y : ℝ), y > 0 → (⌊y^3⌋ : ℝ) - y * (⌊y⌋ : ℝ) = 18 → y ≥ x) ∧
  x = 369 / 19 := by
  sorry

end NUMINAMATH_CALUDE_smallest_solution_for_floor_equation_l2264_226498


namespace NUMINAMATH_CALUDE_perp_planes_necessary_not_sufficient_l2264_226437

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the perpendicular relation between planes and between a line and a plane
variable (perp_planes : Plane → Plane → Prop)
variable (perp_line_plane : Line → Plane → Prop)

-- Define the subset relation for a line being contained in a plane
variable (subset_line_plane : Line → Plane → Prop)

-- Theorem statement
theorem perp_planes_necessary_not_sufficient 
  (α β : Plane) (m : Line) 
  (h_subset : subset_line_plane m α) :
  (∀ m α β, perp_line_plane m β → perp_planes α β) ∧ 
  (∃ m α β, perp_planes α β ∧ ¬perp_line_plane m β) :=
sorry

end NUMINAMATH_CALUDE_perp_planes_necessary_not_sufficient_l2264_226437


namespace NUMINAMATH_CALUDE_inequality_proof_l2264_226479

theorem inequality_proof (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  x*y / (x^2 + y^2 + 2*z^2) + y*z / (2*x^2 + y^2 + z^2) + z*x / (x^2 + 2*y^2 + z^2) ≤ 3/4 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2264_226479


namespace NUMINAMATH_CALUDE_scores_mode_is_80_l2264_226422

def scores : List Nat := [70, 80, 100, 60, 80, 70, 90, 50, 80, 70, 80, 70, 90, 80, 90, 80, 70, 90, 60, 80]

def mode (l : List α) [DecidableEq α] : Option α :=
  l.foldl (fun acc x =>
    match acc with
    | none => some x
    | some y => if l.count x > l.count y then some x else some y
  ) none

theorem scores_mode_is_80 : mode scores = some 80 := by
  sorry

end NUMINAMATH_CALUDE_scores_mode_is_80_l2264_226422


namespace NUMINAMATH_CALUDE_trihedral_angle_properties_l2264_226402

-- Define a trihedral angle
structure TrihedralAngle where
  planeAngle1 : ℝ
  planeAngle2 : ℝ
  planeAngle3 : ℝ
  dihedralAngle1 : ℝ
  dihedralAngle2 : ℝ
  dihedralAngle3 : ℝ

-- State the theorem
theorem trihedral_angle_properties (t : TrihedralAngle) :
  t.planeAngle1 + t.planeAngle2 + t.planeAngle3 < 2 * Real.pi ∧
  t.dihedralAngle1 + t.dihedralAngle2 + t.dihedralAngle3 > Real.pi :=
by sorry

end NUMINAMATH_CALUDE_trihedral_angle_properties_l2264_226402


namespace NUMINAMATH_CALUDE_factor_expression_l2264_226465

theorem factor_expression (b : ℝ) : 52 * b^2 + 208 * b = 52 * b * (b + 4) := by
  sorry

end NUMINAMATH_CALUDE_factor_expression_l2264_226465


namespace NUMINAMATH_CALUDE_problems_per_page_is_three_l2264_226444

/-- The number of problems on each page of homework -/
def problems_per_page : ℕ := sorry

/-- The number of pages of math homework -/
def math_pages : ℕ := 6

/-- The number of pages of reading homework -/
def reading_pages : ℕ := 4

/-- The total number of problems -/
def total_problems : ℕ := 30

/-- Theorem stating that the number of problems per page is 3 -/
theorem problems_per_page_is_three :
  problems_per_page = 3 ∧
  (math_pages + reading_pages) * problems_per_page = total_problems :=
sorry

end NUMINAMATH_CALUDE_problems_per_page_is_three_l2264_226444


namespace NUMINAMATH_CALUDE_four_hearts_probability_l2264_226481

-- Define a standard deck of cards
def standard_deck : ℕ := 52

-- Define the number of hearts in a standard deck
def hearts_in_deck : ℕ := 13

-- Define the number of cards we're drawing
def cards_drawn : ℕ := 4

-- Define the probability of drawing four hearts
def prob_four_hearts : ℚ := 2 / 95

-- Theorem statement
theorem four_hearts_probability :
  (hearts_in_deck.factorial / (hearts_in_deck - cards_drawn).factorial) /
  (standard_deck.factorial / (standard_deck - cards_drawn).factorial) = prob_four_hearts := by
  sorry

end NUMINAMATH_CALUDE_four_hearts_probability_l2264_226481


namespace NUMINAMATH_CALUDE_subtracted_amount_l2264_226414

theorem subtracted_amount (N : ℝ) (A : ℝ) (h1 : N = 200) (h2 : 0.4 * N - A = 50) : A = 30 := by
  sorry

end NUMINAMATH_CALUDE_subtracted_amount_l2264_226414


namespace NUMINAMATH_CALUDE_ten_books_left_to_read_l2264_226404

/-- The number of books left to read in the 'crazy silly school' series -/
def books_left_to_read (total_books read_books : ℕ) : ℕ :=
  total_books - read_books

/-- Theorem stating that there are 10 books left to read -/
theorem ten_books_left_to_read :
  books_left_to_read 22 12 = 10 := by
  sorry

#eval books_left_to_read 22 12

end NUMINAMATH_CALUDE_ten_books_left_to_read_l2264_226404


namespace NUMINAMATH_CALUDE_total_peppers_weight_l2264_226423

/-- The weight of green peppers bought by Hannah's Vegetarian Restaurant -/
def green_peppers : ℚ := 0.3333333333333333

/-- The weight of red peppers bought by Hannah's Vegetarian Restaurant -/
def red_peppers : ℚ := 0.3333333333333333

/-- The total weight of peppers bought by Hannah's Vegetarian Restaurant -/
def total_peppers : ℚ := green_peppers + red_peppers

/-- Theorem stating that the total weight of peppers is 0.6666666666666666 pounds -/
theorem total_peppers_weight : total_peppers = 0.6666666666666666 := by
  sorry

end NUMINAMATH_CALUDE_total_peppers_weight_l2264_226423


namespace NUMINAMATH_CALUDE_hyperbola_perpendicular_points_sum_l2264_226471

/-- Given a hyperbola x²/a² - y²/b² = 1 with 0 < a < b, and points A and B on the hyperbola
    such that OA is perpendicular to OB, prove that 1/|OA|² + 1/|OB|² = 1/a² - 1/b² -/
theorem hyperbola_perpendicular_points_sum (a b : ℝ) (ha : 0 < a) (hb : a < b)
  (A B : ℝ × ℝ) (hA : A.1^2 / a^2 - A.2^2 / b^2 = 1) (hB : B.1^2 / a^2 - B.2^2 / b^2 = 1)
  (hperp : A.1 * B.1 + A.2 * B.2 = 0) :
  1 / (A.1^2 + A.2^2) + 1 / (B.1^2 + B.2^2) = 1 / a^2 - 1 / b^2 :=
sorry

end NUMINAMATH_CALUDE_hyperbola_perpendicular_points_sum_l2264_226471


namespace NUMINAMATH_CALUDE_bills_difference_l2264_226416

/-- The number of bills each person had at the beginning -/
structure Bills where
  geric : ℕ
  kyla : ℕ
  jessa : ℕ

/-- The conditions of the problem -/
def problem_conditions (b : Bills) : Prop :=
  b.geric = 2 * b.kyla ∧
  b.geric = 16 ∧
  b.jessa - 3 = 7

/-- The theorem to prove -/
theorem bills_difference (b : Bills) 
  (h : problem_conditions b) : b.jessa - b.kyla = 2 := by
  sorry

end NUMINAMATH_CALUDE_bills_difference_l2264_226416


namespace NUMINAMATH_CALUDE_land_area_increase_l2264_226490

theorem land_area_increase :
  let initial_side : ℝ := 6
  let increase : ℝ := 1
  let new_side := initial_side + increase
  let initial_area := initial_side ^ 2
  let new_area := new_side ^ 2
  new_area - initial_area = 13 := by
sorry

end NUMINAMATH_CALUDE_land_area_increase_l2264_226490


namespace NUMINAMATH_CALUDE_circle_passes_through_focus_l2264_226464

-- Define the parabola
def parabola (p : ℝ × ℝ) : Prop := p.2^2 = 8 * p.1

-- Define the line
def line (x : ℝ) : Prop := x + 2 = 0

-- Define a circle
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define tangency of a circle to a line
def tangent_to_line (c : Circle) : Prop :=
  c.radius = |c.center.1 + 2|

-- Main theorem
theorem circle_passes_through_focus :
  ∀ c : Circle,
  parabola c.center →
  tangent_to_line c →
  c.center.1^2 + c.center.2^2 = (2 - c.center.1)^2 + c.center.2^2 :=
sorry

end NUMINAMATH_CALUDE_circle_passes_through_focus_l2264_226464


namespace NUMINAMATH_CALUDE_arithmetic_calculation_l2264_226449

theorem arithmetic_calculation : 12 - (-18) + (-7) = 23 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_calculation_l2264_226449


namespace NUMINAMATH_CALUDE_emberly_walk_distance_l2264_226450

/-- Emberly's walking problem -/
theorem emberly_walk_distance :
  ∀ (total_days : ℕ) (days_not_walked : ℕ) (total_miles : ℕ),
    total_days = 31 →
    days_not_walked = 4 →
    total_miles = 108 →
    (total_miles : ℚ) / (total_days - days_not_walked : ℚ) = 4 := by
  sorry

end NUMINAMATH_CALUDE_emberly_walk_distance_l2264_226450


namespace NUMINAMATH_CALUDE_distinctFourDigitNumbers_eq_360_l2264_226451

/-- The number of distinct four-digit numbers that can be formed using the digits 1, 2, 3, 4, 5,
    where exactly one digit repeats once. -/
def distinctFourDigitNumbers : ℕ :=
  let digits : Fin 5 := 5
  let positionsForRepeatedDigit : ℕ := Nat.choose 4 2
  let remainingDigitChoices : ℕ := 4 * 3
  digits * positionsForRepeatedDigit * remainingDigitChoices

/-- Theorem stating that the number of distinct four-digit numbers under the given conditions is 360. -/
theorem distinctFourDigitNumbers_eq_360 : distinctFourDigitNumbers = 360 := by
  sorry

end NUMINAMATH_CALUDE_distinctFourDigitNumbers_eq_360_l2264_226451


namespace NUMINAMATH_CALUDE_roots_magnitude_l2264_226494

theorem roots_magnitude (p : ℝ) (r₁ r₂ : ℝ) : 
  (r₁ ≠ r₂) →  -- r₁ and r₂ are distinct
  (r₁^2 + p*r₁ + 12 = 0) →  -- r₁ is a root of the equation
  (r₂^2 + p*r₂ + 12 = 0) →  -- r₂ is a root of the equation
  (abs r₁ > 3 ∨ abs r₂ > 3) := by
sorry

end NUMINAMATH_CALUDE_roots_magnitude_l2264_226494


namespace NUMINAMATH_CALUDE_vector_problem_l2264_226473

/-- Given two perpendicular vectors a and c, and two parallel vectors b and c in ℝ², 
    prove that x = 4, y = -8, and the magnitude of a + b is 10. -/
theorem vector_problem (x y : ℝ) 
  (a b c : ℝ × ℝ)
  (ha : a = (x, 2))
  (hb : b = (4, y))
  (hc : c = (1, -2))
  (hac_perp : a.1 * c.1 + a.2 * c.2 = 0)
  (hbc_par : b.1 * c.2 = b.2 * c.1) :
  x = 4 ∧ y = -8 ∧ Real.sqrt ((a.1 + b.1)^2 + (a.2 + b.2)^2) = 10 := by
  sorry

end NUMINAMATH_CALUDE_vector_problem_l2264_226473


namespace NUMINAMATH_CALUDE_forty_percent_of_fifty_percent_l2264_226455

theorem forty_percent_of_fifty_percent (x : ℝ) : (0.4 * (0.5 * x)) = (0.2 * x) := by
  sorry

end NUMINAMATH_CALUDE_forty_percent_of_fifty_percent_l2264_226455


namespace NUMINAMATH_CALUDE_sweets_expenditure_l2264_226419

theorem sweets_expenditure (initial_amount : ℚ) (friends_count : ℕ) (amount_per_friend : ℚ) (final_amount : ℚ) 
  (h1 : initial_amount = 71/10)
  (h2 : friends_count = 2)
  (h3 : amount_per_friend = 1)
  (h4 : final_amount = 405/100) :
  initial_amount - friends_count * amount_per_friend - final_amount = 21/20 := by
  sorry

end NUMINAMATH_CALUDE_sweets_expenditure_l2264_226419


namespace NUMINAMATH_CALUDE_largest_two_digit_prime_factor_of_binomial_l2264_226420

theorem largest_two_digit_prime_factor_of_binomial : 
  ∃ (p : ℕ), p.Prime ∧ 10 ≤ p ∧ p < 100 ∧ p ∣ Nat.choose 300 150 ∧
  ∀ (q : ℕ), q.Prime → 10 ≤ q → q < 100 → q ∣ Nat.choose 300 150 → q ≤ p :=
by sorry

end NUMINAMATH_CALUDE_largest_two_digit_prime_factor_of_binomial_l2264_226420


namespace NUMINAMATH_CALUDE_three_card_selection_l2264_226478

/-- The number of cards in a standard deck -/
def standard_deck_size : ℕ := 52

/-- The number of cards to be picked -/
def cards_to_pick : ℕ := 3

/-- The number of ways to pick three different cards from a standard deck where order matters -/
def ways_to_pick_three_cards : ℕ := standard_deck_size * (standard_deck_size - 1) * (standard_deck_size - 2)

theorem three_card_selection :
  ways_to_pick_three_cards = 132600 :=
sorry

end NUMINAMATH_CALUDE_three_card_selection_l2264_226478


namespace NUMINAMATH_CALUDE_consecutive_even_numbers_l2264_226409

/-- Given three consecutive even integers and a condition, prove the value of k. -/
theorem consecutive_even_numbers (N₁ N₂ N₃ k : ℤ) : 
  N₃ = 19 →
  N₂ = N₁ + 2 →
  N₃ = N₂ + 2 →
  3 * N₁ = k * N₃ + 7 →
  k = 2 := by
sorry


end NUMINAMATH_CALUDE_consecutive_even_numbers_l2264_226409
