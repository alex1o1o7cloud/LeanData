import Mathlib

namespace NUMINAMATH_CALUDE_monotone_increasing_intervals_l1325_132533

/-- The function f(x) = 2x^3 - 3x^2 - 36x + 16 -/
def f (x : ℝ) : ℝ := 2 * x^3 - 3 * x^2 - 36 * x + 16

/-- The derivative of f(x) -/
def f' (x : ℝ) : ℝ := 6 * x^2 - 6 * x - 36

theorem monotone_increasing_intervals :
  MonotoneOn f (Set.Ici (-2) ∩ Set.Iic (-2)) ∧
  MonotoneOn f (Set.Ici 3) :=
sorry

end NUMINAMATH_CALUDE_monotone_increasing_intervals_l1325_132533


namespace NUMINAMATH_CALUDE_log_inequality_l1325_132557

theorem log_inequality (m n : ℝ) (hm : m > 0) (hn : n > 0) :
  (Real.log 3 / Real.log m < Real.log 3 / Real.log n) ∧ (Real.log 3 / Real.log n < 0) →
  1 > m ∧ m > n ∧ n > 0 := by
sorry

end NUMINAMATH_CALUDE_log_inequality_l1325_132557


namespace NUMINAMATH_CALUDE_gcd_lcm_product_360_l1325_132563

theorem gcd_lcm_product_360 : 
  ∃! (s : Finset ℕ), 
    (∀ d ∈ s, d > 0 ∧ ∃ a b : ℕ, a > 0 ∧ b > 0 ∧ d = Nat.gcd a b ∧ d * Nat.lcm a b = 360) ∧ 
    s.card = 17 :=
by sorry

end NUMINAMATH_CALUDE_gcd_lcm_product_360_l1325_132563


namespace NUMINAMATH_CALUDE_harry_says_1111_l1325_132555

/-- Represents a student in the counting game -/
inductive Student
| Adam
| Beth
| Claire
| Debby
| Eva
| Frank
| Gina
| Harry

/-- Defines the rules for each student's counting pattern -/
def countingRule (s : Student) : ℕ → Prop :=
  match s with
  | Student.Adam => λ n => n % 4 ≠ 0
  | Student.Beth => λ n => (n % 4 = 0) ∧ (n % 3 ≠ 2)
  | Student.Claire => λ n => (n % 4 = 0) ∧ (n % 3 = 2) ∧ (n % 3 ≠ 0)
  | Student.Debby => λ n => (n % 4 = 0) ∧ (n % 3 = 2) ∧ (n % 3 = 0) ∧ (n % 2 ≠ 0)
  | Student.Eva => λ n => (n % 4 = 0) ∧ (n % 3 = 2) ∧ (n % 3 = 0) ∧ (n % 2 = 0) ∧ (n % 3 ≠ 0)
  | Student.Frank => λ n => (n % 4 = 0) ∧ (n % 3 = 2) ∧ (n % 3 = 0) ∧ (n % 2 = 0) ∧ (n % 3 = 0) ∧ (n % 2 ≠ 0)
  | Student.Gina => λ n => (n % 4 = 0) ∧ (n % 3 = 2) ∧ (n % 3 = 0) ∧ (n % 2 = 0) ∧ (n % 3 = 0) ∧ (n % 2 = 0) ∧ (n % 3 ≠ 0)
  | Student.Harry => λ n => (n % 4 = 0) ∧ (n % 3 = 2) ∧ (n % 3 = 0) ∧ (n % 2 = 0) ∧ (n % 3 = 0) ∧ (n % 2 = 0) ∧ (n % 3 = 0)

/-- The theorem stating that Harry says the number 1111 -/
theorem harry_says_1111 : countingRule Student.Harry 1111 := by
  sorry

end NUMINAMATH_CALUDE_harry_says_1111_l1325_132555


namespace NUMINAMATH_CALUDE_max_purple_points_theorem_l1325_132582

/-- The maximum number of purple points in a configuration of blue and red lines -/
def max_purple_points (n : ℕ) : ℕ :=
  n * (n - 1) * (n - 2) * (n - 3) * (n - 4) / 8

/-- Theorem stating the maximum number of purple points given n blue lines -/
theorem max_purple_points_theorem (n : ℕ) (h : n ≥ 5) :
  let blue_lines := n
  let no_parallel := true
  let no_concurrent := true
  max_purple_points n = n * (n - 1) * (n - 2) * (n - 3) * (n - 4) / 8 :=
by
  sorry

#check max_purple_points_theorem

end NUMINAMATH_CALUDE_max_purple_points_theorem_l1325_132582


namespace NUMINAMATH_CALUDE_power_product_cube_l1325_132568

theorem power_product_cube (a b : ℝ) : (a * b) ^ 3 = a ^ 3 * b ^ 3 := by
  sorry

end NUMINAMATH_CALUDE_power_product_cube_l1325_132568


namespace NUMINAMATH_CALUDE_max_value_difference_l1325_132578

-- Define the function f(x)
def f (x : ℝ) : ℝ := 3 * x - x^3

-- Define a as the point where f(x) reaches its maximum value
def a : ℝ := 1

-- Define b as the maximum value of f(x)
def b : ℝ := f a

-- Theorem statement
theorem max_value_difference (x : ℝ) : a - b = -1 := by
  sorry

end NUMINAMATH_CALUDE_max_value_difference_l1325_132578


namespace NUMINAMATH_CALUDE_touching_spheres_bounds_l1325_132565

/-- Represents a tetrahedron -/
structure Tetrahedron where
  faces : Fin 4 → Real
  volume : Real

/-- Represents a sphere touching all face planes of a tetrahedron -/
structure TouchingSphere where
  radius : Real
  center : Fin 3 → Real

/-- The number of spheres touching all face planes of a tetrahedron -/
def num_touching_spheres (t : Tetrahedron) : ℕ :=
  sorry

/-- Theorem stating the bounds on the number of touching spheres -/
theorem touching_spheres_bounds (t : Tetrahedron) :
  5 ≤ num_touching_spheres t ∧ num_touching_spheres t ≤ 8 :=
sorry

end NUMINAMATH_CALUDE_touching_spheres_bounds_l1325_132565


namespace NUMINAMATH_CALUDE_ashley_exam_marks_l1325_132573

theorem ashley_exam_marks (marks_secured : ℕ) (percentage : ℚ) (max_marks : ℕ) : 
  marks_secured = 332 → percentage = 83/100 → 
  (marks_secured : ℚ) / (max_marks : ℚ) = percentage →
  max_marks = 400 := by
sorry

end NUMINAMATH_CALUDE_ashley_exam_marks_l1325_132573


namespace NUMINAMATH_CALUDE_zeros_of_f_shifted_l1325_132529

-- Define the function f
def f (x : ℝ) : ℝ := x^2 - 1

-- State the theorem
theorem zeros_of_f_shifted (x : ℝ) :
  f (x - 1) = 0 ↔ x = 0 ∨ x = 2 :=
by sorry

end NUMINAMATH_CALUDE_zeros_of_f_shifted_l1325_132529


namespace NUMINAMATH_CALUDE_matts_work_schedule_l1325_132504

/-- Matt's work schedule problem -/
theorem matts_work_schedule (monday_minutes : ℕ) (wednesday_minutes : ℕ) : 
  monday_minutes = 450 →
  wednesday_minutes = 300 →
  wednesday_minutes - (monday_minutes / 2) = 75 := by
  sorry

end NUMINAMATH_CALUDE_matts_work_schedule_l1325_132504


namespace NUMINAMATH_CALUDE_extreme_value_derivative_l1325_132505

/-- A function has an extreme value at a point -/
def has_extreme_value (f : ℝ → ℝ) (x : ℝ) : Prop :=
  ∃ ε > 0, ∀ y ∈ Set.Ioo (x - ε) (x + ε), f y ≤ f x ∨ f y ≥ f x

/-- The relationship between extreme values and derivative -/
theorem extreme_value_derivative (f : ℝ → ℝ) (x : ℝ) 
  (hf : Differentiable ℝ f) :
  (has_extreme_value f x → deriv f x = 0) ∧
  ∃ g : ℝ → ℝ, Differentiable ℝ g ∧ deriv g 0 = 0 ∧ ¬ has_extreme_value g 0 := by
  sorry

end NUMINAMATH_CALUDE_extreme_value_derivative_l1325_132505


namespace NUMINAMATH_CALUDE_platform_length_l1325_132507

/-- Given a train of length 450 m that crosses a platform in 56 sec and a signal pole in 24 sec,
    the length of the platform is 600 m. -/
theorem platform_length (train_length : ℝ) (platform_time : ℝ) (pole_time : ℝ) 
    (h1 : train_length = 450)
    (h2 : platform_time = 56)
    (h3 : pole_time = 24) : 
  train_length * (platform_time / pole_time - 1) = 600 := by
  sorry

end NUMINAMATH_CALUDE_platform_length_l1325_132507


namespace NUMINAMATH_CALUDE_problem_statement_l1325_132500

theorem problem_statement : (2002^3 + 4 * 2002^2 + 6006) / (2002^2 + 2002) = 2005 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l1325_132500


namespace NUMINAMATH_CALUDE_simplify_product_l1325_132559

theorem simplify_product : 8 * (15 / 4) * (-24 / 25) = -144 / 5 := by
  sorry

end NUMINAMATH_CALUDE_simplify_product_l1325_132559


namespace NUMINAMATH_CALUDE_permutations_of_eight_distinct_objects_l1325_132576

theorem permutations_of_eight_distinct_objects : Nat.factorial 8 = 40320 := by
  sorry

end NUMINAMATH_CALUDE_permutations_of_eight_distinct_objects_l1325_132576


namespace NUMINAMATH_CALUDE_smallest_gcd_multiple_l1325_132579

theorem smallest_gcd_multiple (m n : ℕ+) (h : Nat.gcd m n = 7) :
  ∃ (k : ℕ+), k = Nat.gcd (8 * m) (6 * n) ∧ ∀ (l : ℕ+), l = Nat.gcd (8 * m) (6 * n) → k ≤ l :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_gcd_multiple_l1325_132579


namespace NUMINAMATH_CALUDE_consistency_condition_l1325_132527

theorem consistency_condition (a b c d x y z : ℝ) 
  (eq1 : y + z = a)
  (eq2 : x + y = b)
  (eq3 : x + z = c)
  (eq4 : x + y + z = d) :
  a + b + c = 2 * d := by
  sorry

end NUMINAMATH_CALUDE_consistency_condition_l1325_132527


namespace NUMINAMATH_CALUDE_wire_division_l1325_132562

/-- Given a wire of length 49 cm divided into 7 equal parts, prove that each part is 7 cm long -/
theorem wire_division (wire_length : ℝ) (num_parts : ℕ) (part_length : ℝ) 
  (h1 : wire_length = 49)
  (h2 : num_parts = 7)
  (h3 : part_length * num_parts = wire_length) :
  part_length = 7 := by
  sorry

end NUMINAMATH_CALUDE_wire_division_l1325_132562


namespace NUMINAMATH_CALUDE_number_divisibility_l1325_132521

theorem number_divisibility (a b : ℕ) : 
  (∃ k : ℤ, (1001 * a + 110 * b : ℤ) = 11 * k) ∧ 
  (∃ m : ℤ, (111000 * a + 111 * b : ℤ) = 37 * m) ∧
  (∃ n : ℤ, (101010 * a + 10101 * b : ℤ) = 7 * n) ∧
  (∃ p q : ℤ, (909 * (a - b) : ℤ) = 9 * p ∧ (909 * (a - b) : ℤ) = 101 * q) :=
by sorry

end NUMINAMATH_CALUDE_number_divisibility_l1325_132521


namespace NUMINAMATH_CALUDE_total_age_is_22_l1325_132550

/-- Given three people a, b, and c, where:
  - a is two years older than b
  - b is twice as old as c
  - b is 8 years old
  Prove that the total of their ages is 22 years. -/
theorem total_age_is_22 (a b c : ℕ) : 
  b = 8 → 
  a = b + 2 → 
  b = 2 * c → 
  a + b + c = 22 := by sorry

end NUMINAMATH_CALUDE_total_age_is_22_l1325_132550


namespace NUMINAMATH_CALUDE_decreasing_implies_a_le_10_l1325_132592

/-- A quadratic function f(x) = x^2 + 2(a-5)x - 6 -/
def f (a : ℝ) (x : ℝ) : ℝ := x^2 + 2*(a-5)*x - 6

/-- The function f is decreasing on the interval (-∞, -5] -/
def is_decreasing_on_interval (a : ℝ) : Prop :=
  ∀ x y, x ≤ y → y ≤ -5 → f a x ≥ f a y

theorem decreasing_implies_a_le_10 (a : ℝ) :
  is_decreasing_on_interval a → a ≤ 10 := by sorry

end NUMINAMATH_CALUDE_decreasing_implies_a_le_10_l1325_132592


namespace NUMINAMATH_CALUDE_number_of_boys_l1325_132558

theorem number_of_boys (initial_avg : ℝ) (incorrect_height : ℝ) (correct_height : ℝ) (actual_avg : ℝ) :
  initial_avg = 184 →
  incorrect_height = 166 →
  correct_height = 106 →
  actual_avg = 182 →
  ∃ n : ℕ, n * initial_avg - (incorrect_height - correct_height) = n * actual_avg ∧ n = 30 :=
by sorry

end NUMINAMATH_CALUDE_number_of_boys_l1325_132558


namespace NUMINAMATH_CALUDE_combined_land_area_l1325_132585

/-- The combined area of two rectangular tracts of land -/
theorem combined_land_area (length1 width1 length2 width2 : ℕ) 
  (h1 : length1 = 300) 
  (h2 : width1 = 500) 
  (h3 : length2 = 250) 
  (h4 : width2 = 630) : 
  length1 * width1 + length2 * width2 = 307500 := by
  sorry

#check combined_land_area

end NUMINAMATH_CALUDE_combined_land_area_l1325_132585


namespace NUMINAMATH_CALUDE_sum_of_fractions_equals_two_ninths_l1325_132591

theorem sum_of_fractions_equals_two_ninths :
  let sum := (1 / (3 * 4 : ℚ)) + (1 / (4 * 5 : ℚ)) + (1 / (5 * 6 : ℚ)) +
              (1 / (6 * 7 : ℚ)) + (1 / (7 * 8 : ℚ)) + (1 / (8 * 9 : ℚ))
  sum = 2 / 9 := by
    sorry

end NUMINAMATH_CALUDE_sum_of_fractions_equals_two_ninths_l1325_132591


namespace NUMINAMATH_CALUDE_arithmetic_simplification_l1325_132552

theorem arithmetic_simplification :
  (5 + 7 + 3) / 3 - 2 / 3 = 13 / 3 := by sorry

end NUMINAMATH_CALUDE_arithmetic_simplification_l1325_132552


namespace NUMINAMATH_CALUDE_truck_speed_problem_l1325_132593

/-- 
Proves that given two trucks 1025 km apart, with Driver A starting at 90 km/h 
and Driver B starting 1 hour later, if Driver A has driven 145 km farther than 
Driver B when they meet, then Driver B's average speed is 485/6 km/h.
-/
theorem truck_speed_problem (distance : ℝ) (speed_A : ℝ) (extra_distance : ℝ) 
  (h1 : distance = 1025)
  (h2 : speed_A = 90)
  (h3 : extra_distance = 145) : 
  ∃ (speed_B : ℝ) (time : ℝ), 
    speed_B = 485 / 6 ∧ 
    time > 0 ∧
    speed_A * (time + 1) = speed_B * time + extra_distance ∧
    speed_A * (time + 1) + speed_B * time = distance :=
by sorry


end NUMINAMATH_CALUDE_truck_speed_problem_l1325_132593


namespace NUMINAMATH_CALUDE_paper_clips_count_l1325_132594

/-- The number of paper clips in 2 cases -/
def paper_clips_in_two_cases (c b : ℕ) : ℕ := 2 * (c * b) * 600

/-- Theorem stating the number of paper clips in 2 cases -/
theorem paper_clips_count (c b : ℕ) :
  paper_clips_in_two_cases c b = 2 * (c * b) * 600 := by
  sorry

end NUMINAMATH_CALUDE_paper_clips_count_l1325_132594


namespace NUMINAMATH_CALUDE_inequality_system_solution_l1325_132567

theorem inequality_system_solution (x : ℝ) :
  x + 3 ≥ 2 ∧ 2 * (x + 4) > 4 * x + 2 → -1 ≤ x ∧ x < 3 := by
  sorry

end NUMINAMATH_CALUDE_inequality_system_solution_l1325_132567


namespace NUMINAMATH_CALUDE_smallest_b_value_l1325_132531

theorem smallest_b_value (a b : ℝ) : 
  (2 < a ∧ a < b) →
  (2 + a ≤ b) →
  (1 / a + 1 / b ≤ 2) →
  ∀ ε > 0, ∃ b₀ : ℝ, 2 < b₀ ∧ b₀ < 2 + ε ∧
    ∃ a₀ : ℝ, 2 < a₀ ∧ a₀ < b₀ ∧
    (2 + a₀ ≤ b₀) ∧
    (1 / a₀ + 1 / b₀ ≤ 2) :=
by sorry

end NUMINAMATH_CALUDE_smallest_b_value_l1325_132531


namespace NUMINAMATH_CALUDE_xy_greater_than_xz_l1325_132534

theorem xy_greater_than_xz (x y z : ℝ) (h1 : x > y) (h2 : y > z) (h3 : x + y + z = 0) : x * y > x * z := by
  sorry

end NUMINAMATH_CALUDE_xy_greater_than_xz_l1325_132534


namespace NUMINAMATH_CALUDE_sum_of_squares_of_roots_l1325_132586

theorem sum_of_squares_of_roots (a b α β : ℝ) : 
  (∀ x, (x - a) * (x - b) = 1 ↔ x = α ∨ x = β) →
  (∃ x₁ x₂, (x₁ - α) * (x₁ - β) = -1 ∧ (x₂ - α) * (x₂ - β) = -1 ∧ x₁ ≠ x₂) →
  x₁^2 + x₂^2 = a^2 + b^2 :=
sorry

end NUMINAMATH_CALUDE_sum_of_squares_of_roots_l1325_132586


namespace NUMINAMATH_CALUDE_blue_water_bottles_l1325_132513

theorem blue_water_bottles (red black : ℕ) (total removed remaining : ℕ) (blue : ℕ) : 
  red = 2 →
  black = 3 →
  total = red + black + blue →
  removed = 5 →
  remaining = 4 →
  total = removed + remaining →
  blue = 4 := by
sorry

end NUMINAMATH_CALUDE_blue_water_bottles_l1325_132513


namespace NUMINAMATH_CALUDE_residue_7_2023_mod_19_l1325_132532

theorem residue_7_2023_mod_19 : 7^2023 % 19 = 3 := by
  sorry

end NUMINAMATH_CALUDE_residue_7_2023_mod_19_l1325_132532


namespace NUMINAMATH_CALUDE_polygon_sum_l1325_132597

/-- Given a polygon JKLMNO with specific properties, prove that MN + NO = 14.5 -/
theorem polygon_sum (area_JKLMNO : ℝ) (JK KL NO : ℝ) :
  area_JKLMNO = 68 ∧ JK = 10 ∧ KL = 11 ∧ NO = 7 →
  ∃ (MN : ℝ), MN + NO = 14.5 := by
  sorry

end NUMINAMATH_CALUDE_polygon_sum_l1325_132597


namespace NUMINAMATH_CALUDE_equation_system_solution_l1325_132580

theorem equation_system_solution (a b : ℝ) : 
  (∃ (a' : ℝ), a' * (-1) + 5 * (-1) = 15 ∧ 4 * (-1) - b * (-1) = -2) →
  (∃ (b' : ℝ), a * 5 + 5 * 2 = 15 ∧ 4 * 5 - b' * 2 = -2) →
  (a + 4 * b)^2 = 9 :=
by
  sorry

end NUMINAMATH_CALUDE_equation_system_solution_l1325_132580


namespace NUMINAMATH_CALUDE_worker_save_fraction_l1325_132551

/-- Represents the worker's monthly savings scenario -/
structure WorkerSavings where
  monthly_pay : ℝ
  save_fraction : ℝ
  (monthly_pay_positive : monthly_pay > 0)
  (save_fraction_valid : 0 ≤ save_fraction ∧ save_fraction ≤ 1)

/-- The total amount saved over a year -/
def yearly_savings (w : WorkerSavings) : ℝ := 12 * w.save_fraction * w.monthly_pay

/-- The amount not saved from monthly pay -/
def monthly_unsaved (w : WorkerSavings) : ℝ := (1 - w.save_fraction) * w.monthly_pay

/-- Theorem stating the fraction of monthly take-home pay saved -/
theorem worker_save_fraction (w : WorkerSavings) 
  (h : yearly_savings w = 5 * monthly_unsaved w) : 
  w.save_fraction = 5 / 17 := by
  sorry

end NUMINAMATH_CALUDE_worker_save_fraction_l1325_132551


namespace NUMINAMATH_CALUDE_arithmetic_sequence_120th_term_l1325_132526

/-- The nth term of an arithmetic sequence -/
def arithmetic_sequence (a₁ : ℝ) (d : ℝ) (n : ℕ) : ℝ :=
  a₁ + (n - 1 : ℝ) * d

/-- The 120th term of the specific arithmetic sequence -/
def term_120 : ℝ :=
  arithmetic_sequence 6 6 120

theorem arithmetic_sequence_120th_term :
  term_120 = 720 := by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_120th_term_l1325_132526


namespace NUMINAMATH_CALUDE_arithmetic_sequence_log_theorem_l1325_132577

-- Define the logarithm base 2
noncomputable def lg (x : ℝ) : ℝ := Real.log x / Real.log 2

-- Define the arithmetic sequence condition
def is_arithmetic_sequence (a b c : ℝ) : Prop := b - a = c - b

-- State the theorem
theorem arithmetic_sequence_log_theorem (x : ℝ) :
  is_arithmetic_sequence (lg 2) (lg (2^x - 1)) (lg (2^x + 3)) →
  x = Real.log 5 / Real.log 2 :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_log_theorem_l1325_132577


namespace NUMINAMATH_CALUDE_smaller_solution_of_quadratic_l1325_132541

theorem smaller_solution_of_quadratic (x : ℝ) : 
  x^2 + 12*x - 64 = 0 ∧ 
  (∀ y : ℝ, y^2 + 12*y - 64 = 0 → x ≤ y) → 
  x = -16 := by
sorry

end NUMINAMATH_CALUDE_smaller_solution_of_quadratic_l1325_132541


namespace NUMINAMATH_CALUDE_scores_mode_and_median_l1325_132544

def scores : List ℕ := [97, 88, 85, 93, 85]

def mode (l : List ℕ) : ℕ := sorry

def median (l : List ℕ) : ℕ := sorry

theorem scores_mode_and_median :
  mode scores = 85 ∧ median scores = 88 := by sorry

end NUMINAMATH_CALUDE_scores_mode_and_median_l1325_132544


namespace NUMINAMATH_CALUDE_product_of_roots_l1325_132590

theorem product_of_roots (x : ℝ) : (x + 2) * (x - 3) = -10 → 
  ∃ (r₁ r₂ : ℝ), r₁ * r₂ = 4 ∧ (x = r₁ ∨ x = r₂) := by
  sorry

end NUMINAMATH_CALUDE_product_of_roots_l1325_132590


namespace NUMINAMATH_CALUDE_kitten_puppy_difference_l1325_132587

theorem kitten_puppy_difference (kittens puppies : ℕ) : 
  kittens = 78 → puppies = 32 → kittens - 2 * puppies = 14 :=
by
  sorry

end NUMINAMATH_CALUDE_kitten_puppy_difference_l1325_132587


namespace NUMINAMATH_CALUDE_sqrt_sum_of_squares_l1325_132598

theorem sqrt_sum_of_squares : 
  Real.sqrt ((43 * 17)^2 + (43 * 26)^2 + (17 * 26)^2) = 1407 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_sum_of_squares_l1325_132598


namespace NUMINAMATH_CALUDE_circle_pair_relation_infinite_quadrilaterals_l1325_132509

/-- A structure representing a pair of circles with a quadrilateral inscribed in one and circumscribed around the other. -/
structure CirclePair where
  R : ℝ  -- Radius of the larger circle
  r : ℝ  -- Radius of the smaller circle
  d : ℝ  -- Distance between the centers of the circles
  h_positive_R : R > 0
  h_positive_r : r > 0
  h_positive_d : d > 0
  h_d_less_R : d < R

/-- The main theorem stating the relationship between the radii and distance of the circles. -/
theorem circle_pair_relation (cp : CirclePair) :
  1 / (cp.R + cp.d)^2 + 1 / (cp.R - cp.d)^2 = 1 / cp.r^2 :=
sorry

/-- There exist infinitely many quadrilaterals satisfying the conditions. -/
theorem infinite_quadrilaterals (R r d : ℝ) (h_R : R > 0) (h_r : r > 0) (h_d : d > 0) (h_d_R : d < R) :
  ∃ (cp : CirclePair), cp.R = R ∧ cp.r = r ∧ cp.d = d :=
sorry

end NUMINAMATH_CALUDE_circle_pair_relation_infinite_quadrilaterals_l1325_132509


namespace NUMINAMATH_CALUDE_water_needed_for_mixture_l1325_132575

/-- Given a mixture of nutrient concentrate and water, calculate the amount of water needed to prepare a larger volume of the same mixture. -/
theorem water_needed_for_mixture (concentrate : ℝ) (initial_water : ℝ) (total_desired : ℝ) : 
  concentrate = 0.05 → 
  initial_water = 0.03 → 
  total_desired = 0.72 → 
  (initial_water / (concentrate + initial_water)) * total_desired = 0.27 := by
sorry

end NUMINAMATH_CALUDE_water_needed_for_mixture_l1325_132575


namespace NUMINAMATH_CALUDE_inequality_solution_existence_l1325_132514

theorem inequality_solution_existence (a : ℝ) (ha : a > 0) :
  (∃ x : ℝ, |x - 4| + |x - 3| < a) ↔ a > 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_existence_l1325_132514


namespace NUMINAMATH_CALUDE_expression_evaluation_l1325_132553

-- Define the ceiling function
def ceiling (x : ℚ) : ℤ := Int.ceil x

-- Define the main expression
def main_expression : ℚ :=
  (ceiling ((21 : ℚ) / 5 - ceiling ((35 : ℚ) / 23))) /
  (ceiling ((35 : ℚ) / 5 + ceiling ((5 * 23 : ℚ) / 35)))

-- Theorem statement
theorem expression_evaluation :
  main_expression = 3 / 11 := by sorry

end NUMINAMATH_CALUDE_expression_evaluation_l1325_132553


namespace NUMINAMATH_CALUDE_evaluate_expression_l1325_132561

theorem evaluate_expression : (8^6 : ℝ) / (4 * 8^3) = 128 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l1325_132561


namespace NUMINAMATH_CALUDE_triangle_area_l1325_132589

theorem triangle_area (a c B : ℝ) (h1 : a = 1) (h2 : c = 2) (h3 : B = Real.pi / 3) :
  (1/2) * a * c * Real.sin B = Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_l1325_132589


namespace NUMINAMATH_CALUDE_probability_not_all_same_dice_l1325_132518

def num_sides : ℕ := 8
def num_dice : ℕ := 5

theorem probability_not_all_same_dice :
  1 - (num_sides : ℚ) / (num_sides ^ num_dice) = 4095 / 4096 := by
  sorry

end NUMINAMATH_CALUDE_probability_not_all_same_dice_l1325_132518


namespace NUMINAMATH_CALUDE_complex_modulus_one_l1325_132543

theorem complex_modulus_one (z : ℂ) (h : (1 - z) / (1 + z) = Complex.I * 2) : Complex.abs z = 1 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_one_l1325_132543


namespace NUMINAMATH_CALUDE_impossibility_of_filling_l1325_132596

/-- A brick is made of four unit cubes: one unit cube with three unit cubes
    attached to three of its faces, all sharing a common vertex. -/
structure Brick :=
  (cubes : Fin 4 → Unit)

/-- A rectangular parallelepiped with dimensions 11 × 12 × 13 -/
def Parallelepiped := Fin 11 × Fin 12 × Fin 13

/-- A function that represents filling the parallelepiped with bricks -/
def FillParallelepiped := Parallelepiped → Brick

/-- Theorem stating that it's impossible to fill the 11 × 12 × 13 parallelepiped with the given bricks -/
theorem impossibility_of_filling :
  ¬ ∃ (f : FillParallelepiped), True :=
sorry

end NUMINAMATH_CALUDE_impossibility_of_filling_l1325_132596


namespace NUMINAMATH_CALUDE_work_left_theorem_l1325_132522

def work_left (p_days q_days collab_days : ℚ) : ℚ :=
  1 - collab_days * (1 / p_days + 1 / q_days)

theorem work_left_theorem (p_days q_days collab_days : ℚ) 
  (hp : p_days = 15)
  (hq : q_days = 20)
  (hc : collab_days = 4) :
  work_left p_days q_days collab_days = 8 / 15 := by
  sorry

#eval work_left 15 20 4

end NUMINAMATH_CALUDE_work_left_theorem_l1325_132522


namespace NUMINAMATH_CALUDE_geometric_sequence_product_l1325_132570

theorem geometric_sequence_product (a : ℕ → ℝ) :
  (∀ n, a n > 0) →
  (∀ n, ∃ r, a (n + 1) = r * a n) →
  (a 1)^2 - 10*(a 1) + 16 = 0 →
  (a 19)^2 - 10*(a 19) + 16 = 0 →
  a 8 * a 10 * a 12 = 64 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_product_l1325_132570


namespace NUMINAMATH_CALUDE_sam_not_buying_book_probability_l1325_132584

theorem sam_not_buying_book_probability (p : ℚ) 
  (h : p = 5 / 8) : 1 - p = 3 / 8 := by
  sorry

end NUMINAMATH_CALUDE_sam_not_buying_book_probability_l1325_132584


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1325_132542

-- Define an arithmetic sequence
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

-- State the theorem
theorem arithmetic_sequence_sum (a : ℕ → ℝ) :
  is_arithmetic_sequence a →
  a 1 + a 3 = 2 →
  a 2 + a 4 = 6 →
  a 1 + a 7 = 10 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1325_132542


namespace NUMINAMATH_CALUDE_sum_abc_equals_negative_three_l1325_132501

theorem sum_abc_equals_negative_three
  (a b c : ℝ)
  (h_distinct : a ≠ b ∧ b ≠ c ∧ a ≠ c)
  (h_common_root1 : ∃ x : ℝ, x^2 + a*x + 1 = 0 ∧ x^2 + b*x + c = 0)
  (h_common_root2 : ∃ x : ℝ, x^2 + x + a = 0 ∧ x^2 + c*x + b = 0) :
  a + b + c = -3 :=
by sorry

end NUMINAMATH_CALUDE_sum_abc_equals_negative_three_l1325_132501


namespace NUMINAMATH_CALUDE_log_equation_solution_l1325_132545

theorem log_equation_solution (x : ℝ) (h : x > 0) (eq : Real.log (729 : ℝ) / Real.log (3 * x) = x) :
  x = 3 ∧ ¬ ∃ n : ℕ, x = n^2 ∧ ¬ ∃ m : ℕ, x = m^3 ∧ ∃ k : ℕ, x = k := by
  sorry

end NUMINAMATH_CALUDE_log_equation_solution_l1325_132545


namespace NUMINAMATH_CALUDE_largest_three_digit_special_number_l1325_132508

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

def distinct_nonzero_digits (n : ℕ) : Prop :=
  let hundreds := n / 100
  let tens := (n % 100) / 10
  let ones := n % 10
  hundreds ≠ 0 ∧ hundreds ≠ tens ∧ hundreds ≠ ones ∧ tens ≠ ones ∧ tens ≠ 0 ∧ ones ≠ 0

def sum_of_digits (n : ℕ) : ℕ :=
  (n / 100) + ((n % 100) / 10) + (n % 10)

def divisible_by_digits (n : ℕ) : Prop :=
  let hundreds := n / 100
  let tens := (n % 100) / 10
  let ones := n % 10
  n % hundreds = 0 ∧ (tens ≠ 0 → n % tens = 0) ∧ (ones ≠ 0 → n % ones = 0)

theorem largest_three_digit_special_number :
  ∀ n : ℕ, 100 ≤ n → n < 1000 →
    (distinct_nonzero_digits n ∧
     is_prime (sum_of_digits n) ∧
     divisible_by_digits n) →
    n ≤ 963 :=
sorry

end NUMINAMATH_CALUDE_largest_three_digit_special_number_l1325_132508


namespace NUMINAMATH_CALUDE_parallelogram_condition_inscribed_quadrilateral_condition_l1325_132502

-- Define a quadrilateral
structure Quadrilateral :=
  (A B C D : Point)

-- Define a parallelogram
def is_parallelogram (q : Quadrilateral) : Prop := sorry

-- Define parallel sides
def parallel_sides (q : Quadrilateral) (side1 side2 : Segment) : Prop := sorry

-- Define equal sides
def equal_sides (side1 side2 : Segment) : Prop := sorry

-- Define supplementary angles
def supplementary_angles (a1 a2 : Angle) : Prop := sorry

-- Define inscribed in a circle
def inscribed_in_circle (q : Quadrilateral) : Prop := sorry

-- Theorem 1
theorem parallelogram_condition (q : Quadrilateral) 
  (side1 side2 : Segment) :
  parallel_sides q side1 side2 → 
  equal_sides side1 side2 → 
  is_parallelogram q :=
sorry

-- Theorem 2
theorem inscribed_quadrilateral_condition (q : Quadrilateral) 
  (a1 a2 a3 a4 : Angle) :
  supplementary_angles a1 a3 → 
  supplementary_angles a2 a4 → 
  inscribed_in_circle q :=
sorry

end NUMINAMATH_CALUDE_parallelogram_condition_inscribed_quadrilateral_condition_l1325_132502


namespace NUMINAMATH_CALUDE_min_n_constant_term_is_correct_l1325_132510

/-- The minimum natural number n for which (x^2 + 1/(2x^3))^n contains a constant term -/
def min_n_constant_term : ℕ := 5

/-- Predicate to check if the expansion contains a constant term -/
def has_constant_term (n : ℕ) : Prop :=
  ∃ r : ℕ, 2 * n = 5 * r

theorem min_n_constant_term_is_correct :
  (∀ k < min_n_constant_term, ¬ has_constant_term k) ∧
  has_constant_term min_n_constant_term := by sorry

end NUMINAMATH_CALUDE_min_n_constant_term_is_correct_l1325_132510


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l1325_132540

-- Define the quadratic function
def f (a b x : ℝ) := x^2 - a*x - b

-- Define the solution set of the quadratic inequality
def solution_set (a b : ℝ) := {x : ℝ | 1 < x ∧ x < 3}

-- Theorem statement
theorem quadratic_inequality_solution 
  (a b : ℝ) 
  (h : ∀ x, x ∈ solution_set a b ↔ f a b x < 0) :
  a = 4 ∧ b = -3 ∧
  (∀ x, (2*x + a) / (x + b) > 1 ↔ x > -7 ∨ x > 3) :=
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l1325_132540


namespace NUMINAMATH_CALUDE_max_profit_price_l1325_132564

/-- The profit function for the bookstore -/
def profit_function (p : ℝ) : ℝ := 150 * p - 4 * p^2 - 200

/-- The theorem stating the price that maximizes profit -/
theorem max_profit_price :
  ∃ (p : ℝ), 0 ≤ p ∧ p ≤ 30 ∧
  ∀ (q : ℝ), 0 ≤ q ∧ q ≤ 30 → profit_function p ≥ profit_function q ∧
  p = 18.75 := by
  sorry

end NUMINAMATH_CALUDE_max_profit_price_l1325_132564


namespace NUMINAMATH_CALUDE_max_students_above_mean_l1325_132511

/-- Given a class of 150 students, proves that the maximum number of students
    who can have a score higher than the class mean is 149. -/
theorem max_students_above_mean (scores : Fin 150 → ℝ) :
  (Finset.filter (fun i => scores i > Finset.sum Finset.univ scores / 150) Finset.univ).card ≤ 149 :=
by
  sorry

end NUMINAMATH_CALUDE_max_students_above_mean_l1325_132511


namespace NUMINAMATH_CALUDE_brianna_cd_purchase_l1325_132539

/-- Brianna's CD purchase problem -/
theorem brianna_cd_purchase (m : ℚ) (c : ℚ) : 
  (1 / 4 : ℚ) * m = (1 / 2 : ℚ) * c → 
  m - c = (1 / 2 : ℚ) * m := by
  sorry

end NUMINAMATH_CALUDE_brianna_cd_purchase_l1325_132539


namespace NUMINAMATH_CALUDE_knives_percentage_after_trade_l1325_132546

/-- Represents Carolyn's silverware set --/
structure SilverwareSet where
  knives : ℕ
  forks : ℕ
  spoons : ℕ

/-- Calculates the total number of pieces in a silverware set --/
def SilverwareSet.total (s : SilverwareSet) : ℕ :=
  s.knives + s.forks + s.spoons

/-- Represents the initial state of Carolyn's silverware set --/
def initial_set : SilverwareSet :=
  { knives := 6
  , forks := 12
  , spoons := 6 * 3 }

/-- Represents the trade operation --/
def trade (s : SilverwareSet) : SilverwareSet :=
  { knives := s.knives - 6
  , forks := s.forks
  , spoons := s.spoons + 6 }

/-- Theorem stating that after the trade, 0% of Carolyn's silverware is knives --/
theorem knives_percentage_after_trade :
  let final_set := trade initial_set
  (final_set.knives : ℚ) / (final_set.total : ℚ) * 100 = 0 := by
  sorry

end NUMINAMATH_CALUDE_knives_percentage_after_trade_l1325_132546


namespace NUMINAMATH_CALUDE_six_balls_three_boxes_l1325_132530

/-- The number of ways to place distinguishable balls into distinguishable boxes -/
def place_balls (num_balls : ℕ) (num_boxes : ℕ) : ℕ :=
  num_boxes ^ num_balls

/-- Theorem: There are 729 ways to place 6 distinguishable balls into 3 distinguishable boxes -/
theorem six_balls_three_boxes : place_balls 6 3 = 729 := by
  sorry

end NUMINAMATH_CALUDE_six_balls_three_boxes_l1325_132530


namespace NUMINAMATH_CALUDE_new_average_score_is_correct_l1325_132595

/-- Represents the grace mark criteria for different score ranges -/
inductive GraceMarkCriteria where
  | below30 : GraceMarkCriteria
  | between30and40 : GraceMarkCriteria
  | above40 : GraceMarkCriteria

/-- Returns the grace marks for a given criteria -/
def graceMarks (c : GraceMarkCriteria) : ℕ :=
  match c with
  | GraceMarkCriteria.below30 => 5
  | GraceMarkCriteria.between30and40 => 3
  | GraceMarkCriteria.above40 => 1

/-- Calculates the new average score after applying grace marks -/
def newAverageScore (
  classSize : ℕ
  ) (initialAverage : ℚ
  ) (studentsPerRange : ℕ
  ) : ℚ :=
  let initialTotal := classSize * initialAverage
  let totalGraceMarks := 
    studentsPerRange * (graceMarks GraceMarkCriteria.below30 + 
                        graceMarks GraceMarkCriteria.between30and40 + 
                        graceMarks GraceMarkCriteria.above40)
  (initialTotal + totalGraceMarks) / classSize

/-- Theorem stating that the new average score is approximately 39.57 -/
theorem new_average_score_is_correct :
  let classSize := 35
  let initialAverage := 37
  let studentsPerRange := 10
  abs (newAverageScore classSize initialAverage studentsPerRange - 39.57) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_new_average_score_is_correct_l1325_132595


namespace NUMINAMATH_CALUDE_decreasing_linear_function_not_in_third_quadrant_l1325_132519

/-- A linear function y = kx + 1 where k ≠ 0 and y decreases as x increases -/
structure DecreasingLinearFunction where
  k : ℝ
  hk_nonzero : k ≠ 0
  hk_negative : k < 0

/-- The third quadrant -/
def ThirdQuadrant : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1 > 0 ∧ p.2 < 0}

/-- The graph of a linear function y = kx + 1 -/
def LinearFunctionGraph (f : DecreasingLinearFunction) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.2 = f.k * p.1 + 1}

/-- The theorem stating that the graph of a decreasing linear function
    does not pass through the third quadrant -/
theorem decreasing_linear_function_not_in_third_quadrant
  (f : DecreasingLinearFunction) :
  LinearFunctionGraph f ∩ ThirdQuadrant = ∅ := by
  sorry

end NUMINAMATH_CALUDE_decreasing_linear_function_not_in_third_quadrant_l1325_132519


namespace NUMINAMATH_CALUDE_corrected_mean_l1325_132512

theorem corrected_mean (n : ℕ) (original_mean : ℚ) (incorrect_value correct_value : ℚ) :
  n = 50 ∧ original_mean = 36 ∧ incorrect_value = 23 ∧ correct_value = 60 →
  (n : ℚ) * original_mean + (correct_value - incorrect_value) / n = 36.74 := by
  sorry

end NUMINAMATH_CALUDE_corrected_mean_l1325_132512


namespace NUMINAMATH_CALUDE_min_distance_squared_l1325_132547

/-- Given real numbers a, b, c, d satisfying the condition,
    the minimum value of (a - c)^2 + (b - d)^2 is 25/2 -/
theorem min_distance_squared (a b c d : ℝ) 
  (h : (a - 2 * Real.exp a) / b = (2 - c) / (d - 1) ∧ (a - 2 * Real.exp a) / b = 1) :
  ∃ (min : ℝ), min = 25 / 2 ∧ ∀ (x y : ℝ), 
    (x - 2 * Real.exp x) / y = (2 - c) / (d - 1) ∧ (x - 2 * Real.exp x) / y = 1 →
    (x - c)^2 + (y - d)^2 ≥ min :=
by sorry

end NUMINAMATH_CALUDE_min_distance_squared_l1325_132547


namespace NUMINAMATH_CALUDE_linear_function_fixed_point_l1325_132528

theorem linear_function_fixed_point :
  ∀ (k : ℝ), (2 * k - 3) * 2 + (k + 1) * (-3) - (k - 9) = 0 := by
sorry

end NUMINAMATH_CALUDE_linear_function_fixed_point_l1325_132528


namespace NUMINAMATH_CALUDE_x_gt_one_sufficient_not_necessary_for_x_gt_zero_l1325_132517

theorem x_gt_one_sufficient_not_necessary_for_x_gt_zero :
  (∀ x : ℝ, x > 1 → x > 0) ∧ (∃ x : ℝ, x > 0 ∧ x ≤ 1) := by
  sorry

end NUMINAMATH_CALUDE_x_gt_one_sufficient_not_necessary_for_x_gt_zero_l1325_132517


namespace NUMINAMATH_CALUDE_square_perimeter_sum_l1325_132538

theorem square_perimeter_sum (x y : ℕ) : 
  (x : ℤ) ^ 2 - (y : ℤ) ^ 2 = 19 → 
  4 * x + 4 * y = 76 :=
by
  sorry

end NUMINAMATH_CALUDE_square_perimeter_sum_l1325_132538


namespace NUMINAMATH_CALUDE_sock_distribution_l1325_132548

-- Define the total number of socks
def total_socks : ℕ := 9

-- Define the property that among any 4 socks, at least 2 belong to the same child
def at_least_two_same (socks : Finset ℕ) : Prop :=
  ∀ (s : Finset ℕ), s ⊆ socks → s.card = 4 → ∃ (child : ℕ), (s.filter (λ x => x = child)).card ≥ 2

-- Define the property that among any 5 socks, no more than 3 belong to the same child
def no_more_than_three (socks : Finset ℕ) : Prop :=
  ∀ (s : Finset ℕ), s ⊆ socks → s.card = 5 → ∀ (child : ℕ), (s.filter (λ x => x = child)).card ≤ 3

-- Theorem statement
theorem sock_distribution (socks : Finset ℕ) 
  (h_total : socks.card = total_socks)
  (h_at_least_two : at_least_two_same socks)
  (h_no_more_than_three : no_more_than_three socks) :
  ∃ (children : Finset ℕ), 
    children.card = 3 ∧ 
    (∀ child ∈ children, (socks.filter (λ x => x = child)).card = 3) :=
sorry

end NUMINAMATH_CALUDE_sock_distribution_l1325_132548


namespace NUMINAMATH_CALUDE_paint_usage_fraction_l1325_132588

theorem paint_usage_fraction (total_paint : ℝ) (paint_used_total : ℝ) :
  total_paint = 360 →
  paint_used_total = 168 →
  let paint_used_first_week := total_paint / 3
  let paint_remaining := total_paint - paint_used_first_week
  let paint_used_second_week := paint_used_total - paint_used_first_week
  paint_used_second_week / paint_remaining = 1 / 5 := by
sorry

end NUMINAMATH_CALUDE_paint_usage_fraction_l1325_132588


namespace NUMINAMATH_CALUDE_book_sale_loss_l1325_132520

/-- Represents the sale of two books with given conditions -/
def book_sale (total_cost cost_book1 loss_percent1 gain_percent2 : ℚ) : ℚ :=
  let cost_book2 := total_cost - cost_book1
  let selling_price1 := cost_book1 * (1 - loss_percent1 / 100)
  let selling_price2 := cost_book2 * (1 + gain_percent2 / 100)
  let total_selling_price := selling_price1 + selling_price2
  total_cost - total_selling_price

/-- Theorem stating the overall loss from the book sale -/
theorem book_sale_loss :
  book_sale 460 268.33 15 19 = 3.8322 := by
  sorry

end NUMINAMATH_CALUDE_book_sale_loss_l1325_132520


namespace NUMINAMATH_CALUDE_cos_135_degrees_l1325_132599

theorem cos_135_degrees : Real.cos (135 * π / 180) = -Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_135_degrees_l1325_132599


namespace NUMINAMATH_CALUDE_relationship_between_D_and_A_l1325_132515

theorem relationship_between_D_and_A (A B C D : Prop) 
  (h1 : A → B)
  (h2 : ¬(B → A))
  (h3 : B → C)
  (h4 : ¬(C → B))
  (h5 : D ↔ C) :
  (D → A) ∧ ¬(A → D) :=
sorry

end NUMINAMATH_CALUDE_relationship_between_D_and_A_l1325_132515


namespace NUMINAMATH_CALUDE_number_of_students_in_section_B_l1325_132556

theorem number_of_students_in_section_B (students_A : ℕ) (avg_weight_A : ℚ) (avg_weight_B : ℚ) (avg_weight_total : ℚ) :
  students_A = 26 →
  avg_weight_A = 50 →
  avg_weight_B = 30 →
  avg_weight_total = 38.67 →
  ∃ (students_B : ℕ), 
    (students_A * avg_weight_A + students_B * avg_weight_B : ℚ) / (students_A + students_B : ℚ) = avg_weight_total ∧
    students_B = 34 :=
by sorry

end NUMINAMATH_CALUDE_number_of_students_in_section_B_l1325_132556


namespace NUMINAMATH_CALUDE_monotonic_quadratic_function_l1325_132581

/-- The function f(x) = x^2 - 2mx + 3 is monotonic on the interval [1, 3] if and only if m ≤ 1 or m ≥ 3 -/
theorem monotonic_quadratic_function (m : ℝ) :
  (∀ x ∈ Set.Icc 1 3, Monotone (fun x => x^2 - 2*m*x + 3)) ↔ (m ≤ 1 ∨ m ≥ 3) := by
  sorry

end NUMINAMATH_CALUDE_monotonic_quadratic_function_l1325_132581


namespace NUMINAMATH_CALUDE_sin_cos_45_degrees_l1325_132549

theorem sin_cos_45_degrees : 
  let θ : Real := Real.pi / 4  -- 45 degrees in radians
  Real.sin θ = 1 / Real.sqrt 2 ∧ Real.cos θ = 1 / Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_45_degrees_l1325_132549


namespace NUMINAMATH_CALUDE_modular_inverse_of_5_mod_26_l1325_132535

theorem modular_inverse_of_5_mod_26 : ∃ x : ℕ, x ≤ 25 ∧ (5 * x) % 26 = 1 :=
by
  use 21
  sorry

end NUMINAMATH_CALUDE_modular_inverse_of_5_mod_26_l1325_132535


namespace NUMINAMATH_CALUDE_union_condition_intersection_condition_l1325_132516

-- Define sets A and B
def A : Set ℝ := {x | x^2 + 4*x = 0}
def B (a : ℝ) : Set ℝ := {x | x^2 + 2*(a+1)*x + a^2 - 1 = 0}

-- Theorem for the first question
theorem union_condition (a : ℝ) : A ∪ B a = B a → a = 1 := by sorry

-- Theorem for the second question
theorem intersection_condition (a : ℝ) : A ∩ B a = B a → a ≤ -1 ∨ a = 1 := by sorry

end NUMINAMATH_CALUDE_union_condition_intersection_condition_l1325_132516


namespace NUMINAMATH_CALUDE_area_triangle_QCA_l1325_132572

/-- The area of triangle QCA given the coordinates of points Q, A, and C -/
theorem area_triangle_QCA (p : ℝ) : 
  let Q : ℝ × ℝ := (0, 15)
  let A : ℝ × ℝ := (3, 15)
  let C : ℝ × ℝ := (0, p)
  let area := (1/2) * (A.1 - Q.1) * (Q.2 - C.2)
  area = 45/2 - 3*p/2 := by
sorry

end NUMINAMATH_CALUDE_area_triangle_QCA_l1325_132572


namespace NUMINAMATH_CALUDE_remaining_money_l1325_132583

def octal_to_decimal (n : ℕ) : ℕ := sorry

def john_savings : ℕ := 5555

def airline_cost : ℕ := 1200
def lodging_cost : ℕ := 800
def food_cost : ℕ := 400

def total_expenses : ℕ := airline_cost + lodging_cost + food_cost

theorem remaining_money :
  octal_to_decimal john_savings - total_expenses = 525 := by sorry

end NUMINAMATH_CALUDE_remaining_money_l1325_132583


namespace NUMINAMATH_CALUDE_cubic_expression_value_l1325_132566

theorem cubic_expression_value : 7^3 - 4 * 7^2 + 4 * 7 - 2 = 173 := by
  sorry

end NUMINAMATH_CALUDE_cubic_expression_value_l1325_132566


namespace NUMINAMATH_CALUDE_trees_around_square_theorem_l1325_132560

/-- Represents a rectangle with trees planted along its sides -/
structure TreeRectangle where
  side_ad : ℕ  -- Number of trees along side AD
  side_ab : ℕ  -- Number of trees along side AB

/-- Calculates the number of trees around a square with side length equal to the longer side of the rectangle -/
def trees_around_square (rect : TreeRectangle) : ℕ :=
  4 * (rect.side_ad - 1) + 4

/-- Theorem stating that for a rectangle with 49 trees along AD and 25 along AB,
    the number of trees around the corresponding square is 196 -/
theorem trees_around_square_theorem (rect : TreeRectangle) 
        (h1 : rect.side_ad = 49) (h2 : rect.side_ab = 25) : 
        trees_around_square rect = 196 := by
  sorry

#eval trees_around_square ⟨49, 25⟩

end NUMINAMATH_CALUDE_trees_around_square_theorem_l1325_132560


namespace NUMINAMATH_CALUDE_probability_of_U_in_SHUXUE_l1325_132569

def pinyin : String := "SHUXUE"

theorem probability_of_U_in_SHUXUE : 
  (pinyin.toList.filter (· = 'U')).length / pinyin.length = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_probability_of_U_in_SHUXUE_l1325_132569


namespace NUMINAMATH_CALUDE_equal_projections_imply_relation_l1325_132571

-- Define the points A, B, and C
def A (a : ℝ) : ℝ × ℝ := (a, 1)
def B (b : ℝ) : ℝ × ℝ := (2, b)
def C : ℝ × ℝ := (3, 4)

-- Define vectors OA, OB, and OC
def OA (a : ℝ) : ℝ × ℝ := A a
def OB (b : ℝ) : ℝ × ℝ := B b
def OC : ℝ × ℝ := C

-- Define the dot product of two 2D vectors
def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

-- Theorem statement
theorem equal_projections_imply_relation (a b : ℝ) :
  dot_product (OA a) OC = dot_product (OB b) OC →
  3 * a - 4 * b = 2 := by
  sorry


end NUMINAMATH_CALUDE_equal_projections_imply_relation_l1325_132571


namespace NUMINAMATH_CALUDE_deck_size_problem_l1325_132503

theorem deck_size_problem (r b : ℕ) : 
  -- Initial probability of selecting a red card
  (r : ℚ) / (r + b) = 2 / 5 →
  -- Probability after adding 6 black cards
  (r : ℚ) / (r + b + 6) = 1 / 3 →
  -- Total number of cards initially
  r + b = 5 := by
sorry

end NUMINAMATH_CALUDE_deck_size_problem_l1325_132503


namespace NUMINAMATH_CALUDE_chandler_apples_per_week_l1325_132574

/-- The number of apples Chandler can eat per week -/
def chandler_apples : ℕ := 23

/-- The number of apples Lucy can eat per week -/
def lucy_apples : ℕ := 19

/-- The number of apples ordered for a month -/
def monthly_order : ℕ := 168

/-- The number of weeks in a month -/
def weeks_per_month : ℕ := 4

/-- Theorem stating that Chandler can eat 23 apples per week -/
theorem chandler_apples_per_week :
  chandler_apples * weeks_per_month + lucy_apples * weeks_per_month = monthly_order :=
by sorry

end NUMINAMATH_CALUDE_chandler_apples_per_week_l1325_132574


namespace NUMINAMATH_CALUDE_train_length_l1325_132506

/-- The length of a train given its speed and time to cross a pole -/
theorem train_length (speed : ℝ) (time : ℝ) : 
  speed = 52 → time = 9 → ∃ length : ℝ, 
  (abs (length - 129.96) < 0.01) ∧ (length = speed * 1000 / 3600 * time) := by
  sorry

#check train_length

end NUMINAMATH_CALUDE_train_length_l1325_132506


namespace NUMINAMATH_CALUDE_ellipse_m_relation_l1325_132536

/-- Represents an ellipse with parameter m -/
structure Ellipse (m : ℝ) where
  eq : ∀ (x y : ℝ), x^2 / (10 - m) + y^2 / (m - 2) = 1
  major_axis_y : m - 2 > 10 - m
  focal_distance : ℝ

/-- The theorem stating the relationship between m and the focal distance -/
theorem ellipse_m_relation (m : ℝ) (e : Ellipse m) (h : e.focal_distance = 4) :
  16 = 2 * m - 12 := by
  sorry


end NUMINAMATH_CALUDE_ellipse_m_relation_l1325_132536


namespace NUMINAMATH_CALUDE_aaron_age_l1325_132523

/-- Proves that Aaron is 16 years old given the conditions of the problem -/
theorem aaron_age : 
  ∀ (aaron_age henry_age sister_age : ℕ),
  sister_age = 3 * aaron_age →
  henry_age = 4 * sister_age →
  henry_age + sister_age = 240 →
  aaron_age = 16 :=
by
  sorry

end NUMINAMATH_CALUDE_aaron_age_l1325_132523


namespace NUMINAMATH_CALUDE_rectangle_area_l1325_132524

/-- A rectangle with perimeter 36 and length three times its width has area 60.75 -/
theorem rectangle_area (width : ℝ) (length : ℝ) : 
  (width + length) * 2 = 36 →
  length = 3 * width →
  width * length = 60.75 := by
sorry

end NUMINAMATH_CALUDE_rectangle_area_l1325_132524


namespace NUMINAMATH_CALUDE_probability_edge_endpoints_is_correct_l1325_132554

structure RegularIcosahedron where
  vertices : Finset (Fin 12)
  edges : Finset (Fin 12 × Fin 12)
  vertex_degree : ∀ v : Fin 12, (edges.filter (λ e => e.1 = v ∨ e.2 = v)).card = 5

def probability_edge_endpoints (I : RegularIcosahedron) : ℚ :=
  5 / 11

theorem probability_edge_endpoints_is_correct (I : RegularIcosahedron) :
  probability_edge_endpoints I = 5 / 11 := by
  sorry

end NUMINAMATH_CALUDE_probability_edge_endpoints_is_correct_l1325_132554


namespace NUMINAMATH_CALUDE_set_b_forms_triangle_l1325_132537

/-- Triangle inequality theorem: the sum of the lengths of any two sides of a triangle
    must be greater than the length of the remaining side. -/
def triangle_inequality (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

/-- A function to check if three line segments can form a triangle. -/
def can_form_triangle (a b c : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ triangle_inequality a b c

/-- Theorem: The set of line segments (6, 7, 8) can form a triangle. -/
theorem set_b_forms_triangle : can_form_triangle 6 7 8 := by
  sorry

end NUMINAMATH_CALUDE_set_b_forms_triangle_l1325_132537


namespace NUMINAMATH_CALUDE_same_color_probability_l1325_132525

/-- Represents a 12-sided die with colored sides -/
structure ColoredDie :=
  (pink : Nat)
  (green : Nat)
  (blue : Nat)
  (total : Nat)
  (h_total : pink + green + blue = total)

/-- The probability of two dice showing the same color -/
def samColorProbability (d : ColoredDie) : Rat :=
  (d.pink^2 + d.green^2 + d.blue^2) / d.total^2

/-- Two 12-sided dice with 3 pink, 4 green, and 5 blue sides each -/
def twelveSidedDie : ColoredDie :=
  { pink := 3
  , green := 4
  , blue := 5
  , total := 12
  , h_total := by rfl }

theorem same_color_probability :
  samColorProbability twelveSidedDie = 25 / 72 := by
  sorry

end NUMINAMATH_CALUDE_same_color_probability_l1325_132525
