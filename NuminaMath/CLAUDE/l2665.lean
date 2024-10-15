import Mathlib

namespace NUMINAMATH_CALUDE_sales_tax_difference_l2665_266597

theorem sales_tax_difference (price : ℝ) (tax_rate1 : ℝ) (tax_rate2 : ℝ) : 
  price = 30 → tax_rate1 = 0.0675 → tax_rate2 = 0.055 → 
  price * tax_rate1 - price * tax_rate2 = 0.375 := by
  sorry

#eval (30 * 0.0675 - 30 * 0.055)

end NUMINAMATH_CALUDE_sales_tax_difference_l2665_266597


namespace NUMINAMATH_CALUDE_butterfat_percentage_of_added_milk_l2665_266516

/-- Prove that the percentage of butterfat in the added milk is 10% -/
theorem butterfat_percentage_of_added_milk
  (initial_volume : ℝ)
  (initial_butterfat_percentage : ℝ)
  (added_volume : ℝ)
  (final_butterfat_percentage : ℝ)
  (h_initial_volume : initial_volume = 8)
  (h_initial_butterfat : initial_butterfat_percentage = 35)
  (h_added_volume : added_volume = 12)
  (h_final_butterfat : final_butterfat_percentage = 20)
  (h_total_volume : initial_volume + added_volume = 20) :
  let added_butterfat_percentage :=
    (final_butterfat_percentage * (initial_volume + added_volume) -
     initial_butterfat_percentage * initial_volume) / added_volume
  added_butterfat_percentage = 10 :=
by sorry

end NUMINAMATH_CALUDE_butterfat_percentage_of_added_milk_l2665_266516


namespace NUMINAMATH_CALUDE_first_step_error_l2665_266583

-- Define the original equation
def original_equation (x : ℝ) : Prop :=
  (x - 1) / 2 + 1 = (2 * x + 1) / 3

-- Define the incorrect step 1 result
def incorrect_step1 (x : ℝ) : Prop :=
  3 * (x - 1) + 2 = 2 * x + 1

-- Define the correct step 1 result
def correct_step1 (x : ℝ) : Prop :=
  3 * (x - 1) + 6 = 2 * x + 1

-- Theorem stating that the first step is erroneous
theorem first_step_error :
  ∃ x : ℝ, original_equation x ∧ ¬(incorrect_step1 x ↔ correct_step1 x) :=
sorry

end NUMINAMATH_CALUDE_first_step_error_l2665_266583


namespace NUMINAMATH_CALUDE_abc_divides_sum_pow13_l2665_266537

theorem abc_divides_sum_pow13 (a b c : ℕ+) 
  (h1 : a ∣ b^3) 
  (h2 : b ∣ c^3) 
  (h3 : c ∣ a^3) : 
  (a * b * c) ∣ (a + b + c)^13 := by
  sorry

end NUMINAMATH_CALUDE_abc_divides_sum_pow13_l2665_266537


namespace NUMINAMATH_CALUDE_coefficient_of_x_is_17_l2665_266549

def expression (x : ℝ) : ℝ := 5 * (x - 6) + 6 * (9 - 3 * x^2 + 7 * x) - 10 * (3 * x - 2)

theorem coefficient_of_x_is_17 :
  ∃ (a b c : ℝ), expression = λ x => a * x^2 + 17 * x + c :=
by sorry

end NUMINAMATH_CALUDE_coefficient_of_x_is_17_l2665_266549


namespace NUMINAMATH_CALUDE_school_children_count_l2665_266506

theorem school_children_count (total_bananas : ℕ) : 
  (∃ (children : ℕ), 
    total_bananas = 2 * children ∧ 
    total_bananas = 4 * (children - 360)) →
  ∃ (children : ℕ), children = 720 := by
sorry

end NUMINAMATH_CALUDE_school_children_count_l2665_266506


namespace NUMINAMATH_CALUDE_total_seashells_l2665_266525

theorem total_seashells (sam mary lucy : ℕ) 
  (h_sam : sam = 18) 
  (h_mary : mary = 47) 
  (h_lucy : lucy = 32) : 
  sam + mary + lucy = 97 := by
  sorry

end NUMINAMATH_CALUDE_total_seashells_l2665_266525


namespace NUMINAMATH_CALUDE_video_release_week_l2665_266599

/-- Proves that the number of days in a week is 7, given John's video release schedule --/
theorem video_release_week (short_video_length : ℕ) (long_video_multiplier : ℕ) 
  (videos_per_day : ℕ) (short_videos_per_day : ℕ) (total_weekly_minutes : ℕ) :
  short_video_length = 2 →
  long_video_multiplier = 6 →
  videos_per_day = 3 →
  short_videos_per_day = 2 →
  total_weekly_minutes = 112 →
  (total_weekly_minutes / (short_videos_per_day * short_video_length + 
    (videos_per_day - short_videos_per_day) * (long_video_multiplier * short_video_length))) = 7 := by
  sorry

#check video_release_week

end NUMINAMATH_CALUDE_video_release_week_l2665_266599


namespace NUMINAMATH_CALUDE_point_on_line_k_l2665_266564

/-- A line passing through the origin with slope 1/5 -/
def line_k (x y : ℝ) : Prop := y = (1/5) * x

theorem point_on_line_k (x y : ℝ) :
  line_k x 1 →
  line_k 5 y →
  y = 1 := by
  sorry

end NUMINAMATH_CALUDE_point_on_line_k_l2665_266564


namespace NUMINAMATH_CALUDE_geometry_theorem_l2665_266587

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (intersects : Plane → Plane → Line → Prop)
variable (parallel : Line → Plane → Prop)
variable (contains : Plane → Line → Prop)
variable (perpendicular : Line → Plane → Prop)
variable (perp_planes : Plane → Plane → Prop)
variable (perp_lines : Line → Line → Prop)

-- State the theorem
theorem geometry_theorem 
  (α β γ : Plane) (l m : Line)
  (h1 : intersects β γ l)
  (h2 : parallel l α)
  (h3 : contains α m)
  (h4 : perpendicular m γ) :
  perp_planes α γ ∧ perp_lines l m :=
sorry

end NUMINAMATH_CALUDE_geometry_theorem_l2665_266587


namespace NUMINAMATH_CALUDE_marsh_bird_difference_l2665_266575

theorem marsh_bird_difference (canadian_geese mallard_ducks great_egrets red_winged_blackbirds : ℕ) 
  (h1 : canadian_geese = 58)
  (h2 : mallard_ducks = 37)
  (h3 : great_egrets = 21)
  (h4 : red_winged_blackbirds = 15) :
  canadian_geese - mallard_ducks = 21 := by
  sorry

end NUMINAMATH_CALUDE_marsh_bird_difference_l2665_266575


namespace NUMINAMATH_CALUDE_even_function_implies_b_zero_solution_set_inequality_l2665_266566

-- Define the function f
def f (b : ℝ) (x : ℝ) : ℝ := x^2 + b*x + 1

-- Theorem 1: If f is even, then b = 0
theorem even_function_implies_b_zero (b : ℝ) :
  (∀ x : ℝ, f b x = f b (-x)) → b = 0 := by sorry

-- Define the specific function f with b = 0
def f_zero (x : ℝ) : ℝ := x^2 + 1

-- Theorem 2: Solution set of f(x-1) < |x|
theorem solution_set_inequality :
  {x : ℝ | f_zero (x - 1) < |x|} = {x : ℝ | 1 < x ∧ x < 2} := by sorry

end NUMINAMATH_CALUDE_even_function_implies_b_zero_solution_set_inequality_l2665_266566


namespace NUMINAMATH_CALUDE_investment_decrease_l2665_266512

/-- Given an initial investment that increases by 50% in the first year
    and has a net increase of 4.999999999999982% after two years,
    prove that the percentage decrease in the second year is 30%. -/
theorem investment_decrease (initial : ℝ) (initial_pos : initial > 0) :
  let first_year := initial * 1.5
  let final := initial * 1.04999999999999982
  let second_year_factor := final / first_year
  second_year_factor = 0.7 := by sorry

end NUMINAMATH_CALUDE_investment_decrease_l2665_266512


namespace NUMINAMATH_CALUDE_smallest_number_proof_l2665_266513

/-- A function that checks if a natural number contains all digits from 0 to 9 exactly once -/
def has_all_digits_once (n : ℕ) : Prop := sorry

/-- A function that returns the smallest natural number that is a multiple of 36 and contains all digits from 0 to 9 exactly once -/
def smallest_number_with_all_digits_divisible_by_36 : ℕ := sorry

theorem smallest_number_proof :
  smallest_number_with_all_digits_divisible_by_36 = 1023457896 ∧
  has_all_digits_once smallest_number_with_all_digits_divisible_by_36 ∧
  smallest_number_with_all_digits_divisible_by_36 % 36 = 0 ∧
  ∀ m : ℕ, m < smallest_number_with_all_digits_divisible_by_36 →
    ¬(has_all_digits_once m ∧ m % 36 = 0) :=
by sorry

end NUMINAMATH_CALUDE_smallest_number_proof_l2665_266513


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l2665_266554

theorem quadratic_equation_solution : 
  ∃ (x₁ x₂ : ℝ), x₁ = 5 ∧ x₂ = 1 ∧ 
  (x₁^2 - 6*x₁ + 5 = 0) ∧ (x₂^2 - 6*x₂ + 5 = 0) :=
by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l2665_266554


namespace NUMINAMATH_CALUDE_tan_ratio_given_sin_condition_l2665_266577

theorem tan_ratio_given_sin_condition (α : Real) 
  (h : 5 * Real.sin (2 * α) = Real.sin (2 * (π / 180))) : 
  Real.tan (α + π / 180) / Real.tan (α - π / 180) = -3/2 := by
  sorry

end NUMINAMATH_CALUDE_tan_ratio_given_sin_condition_l2665_266577


namespace NUMINAMATH_CALUDE_smallest_norwegian_l2665_266585

def is_norwegian (n : ℕ) : Prop :=
  ∃ (a b c : ℕ), a < b ∧ b < c ∧ a ∣ n ∧ b ∣ n ∧ c ∣ n ∧ a + b + c = 2022

theorem smallest_norwegian : ∀ n : ℕ, is_norwegian n → n ≥ 1344 :=
sorry

end NUMINAMATH_CALUDE_smallest_norwegian_l2665_266585


namespace NUMINAMATH_CALUDE_triangle_count_l2665_266504

def stick_lengths : List ℕ := [1, 2, 3, 4, 5]

def is_valid_triangle (a b c : ℕ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

def count_valid_triangles (lengths : List ℕ) : ℕ :=
  (lengths.toFinset.powerset.filter (fun s => s.card = 3)).card

theorem triangle_count : count_valid_triangles stick_lengths = 22 := by
  sorry

end NUMINAMATH_CALUDE_triangle_count_l2665_266504


namespace NUMINAMATH_CALUDE_complex_power_difference_l2665_266553

-- Define the complex number i
def i : ℂ := Complex.I

-- State the theorem
theorem complex_power_difference (h : i^2 = -1) : (1 + i)^20 - (1 - i)^20 = 0 := by
  sorry

end NUMINAMATH_CALUDE_complex_power_difference_l2665_266553


namespace NUMINAMATH_CALUDE_joan_money_found_l2665_266503

def total_money (dimes_jacket : ℕ) (dimes_shorts : ℕ) (nickels_shorts : ℕ) 
  (quarters_jeans : ℕ) (pennies_jeans : ℕ) (nickels_backpack : ℕ) (pennies_backpack : ℕ) : ℚ :=
  (dimes_jacket + dimes_shorts) * (10 : ℚ) / 100 +
  (nickels_shorts + nickels_backpack) * (5 : ℚ) / 100 +
  quarters_jeans * (25 : ℚ) / 100 +
  (pennies_jeans + pennies_backpack) * (1 : ℚ) / 100

theorem joan_money_found :
  total_money 15 4 7 12 2 8 23 = (590 : ℚ) / 100 := by
  sorry

end NUMINAMATH_CALUDE_joan_money_found_l2665_266503


namespace NUMINAMATH_CALUDE_statue_weight_proof_l2665_266560

/-- The weight of a statue after three successive cuts -/
def final_statue_weight (initial_weight : ℝ) : ℝ :=
  let weight_after_first_cut := initial_weight * (1 - 0.25)
  let weight_after_second_cut := weight_after_first_cut * (1 - 0.15)
  let weight_after_third_cut := weight_after_second_cut * (1 - 0.10)
  weight_after_third_cut

/-- Theorem stating that the final weight of the statue is 109.0125 kg -/
theorem statue_weight_proof :
  final_statue_weight 190 = 109.0125 := by
  sorry

end NUMINAMATH_CALUDE_statue_weight_proof_l2665_266560


namespace NUMINAMATH_CALUDE_wall_penetrating_skill_l2665_266556

theorem wall_penetrating_skill (n : ℕ) : 
  (8 * Real.sqrt (8 / n) = Real.sqrt (8 * (8 / n))) ↔ n = 63 := by
  sorry

end NUMINAMATH_CALUDE_wall_penetrating_skill_l2665_266556


namespace NUMINAMATH_CALUDE_bobs_age_multiple_l2665_266594

theorem bobs_age_multiple (bob_age carol_age : ℕ) (m : ℚ) : 
  bob_age = 16 →
  carol_age = 50 →
  carol_age = m * bob_age + 2 →
  m = 3 := by
sorry

end NUMINAMATH_CALUDE_bobs_age_multiple_l2665_266594


namespace NUMINAMATH_CALUDE_poverty_alleviation_volunteers_l2665_266563

/-- Represents the age group frequencies in the histogram -/
structure AgeDistribution :=
  (f1 f2 f3 f4 f5 : ℝ)
  (sum_to_one : f1 + f2 + f3 + f4 + f5 = 1)

/-- Represents the stratified sample -/
structure StratifiedSample :=
  (total : ℕ)
  (under_35 : ℕ)
  (over_35 : ℕ)
  (sum_equal : under_35 + over_35 = total)

/-- The main theorem -/
theorem poverty_alleviation_volunteers 
  (dist : AgeDistribution) 
  (sample : StratifiedSample) 
  (h1 : dist.f1 = 0.01)
  (h2 : dist.f2 = 0.02)
  (h3 : dist.f3 = 0.04)
  (h5 : dist.f5 = 0.07)
  (h_sample : sample.total = 10 ∧ sample.under_35 = 6 ∧ sample.over_35 = 4) :
  dist.f4 = 0.06 ∧ 
  ∃ (X : Fin 4 → ℝ), 
    X 0 = 1/30 ∧ 
    X 1 = 3/10 ∧ 
    X 2 = 1/2 ∧ 
    X 3 = 1/6 ∧
    (X 0 * 0 + X 1 * 1 + X 2 * 2 + X 3 * 3 = 1.8) := by
  sorry

end NUMINAMATH_CALUDE_poverty_alleviation_volunteers_l2665_266563


namespace NUMINAMATH_CALUDE_counterexample_exists_l2665_266500

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d ∣ n → d = 1 ∨ d = n

theorem counterexample_exists : ∃ n : ℕ, 
  ¬ is_prime n ∧ ¬ is_prime (n - 3) ∧ n = 15 := by
  sorry

end NUMINAMATH_CALUDE_counterexample_exists_l2665_266500


namespace NUMINAMATH_CALUDE_power_relation_l2665_266551

theorem power_relation (a m n : ℝ) (hm : a^m = 2) (hn : a^n = 5) : 
  a^(3*m - 2*n) = 8/25 := by
sorry

end NUMINAMATH_CALUDE_power_relation_l2665_266551


namespace NUMINAMATH_CALUDE_birdwatching_sites_l2665_266501

theorem birdwatching_sites (x : ℕ) : 
  (7 * x + 5 * x + 80) / (2 * x + 10) = 7 → x + x = 10 := by
  sorry

end NUMINAMATH_CALUDE_birdwatching_sites_l2665_266501


namespace NUMINAMATH_CALUDE_katy_brownies_l2665_266598

/-- The number of brownies Katy made and ate over three days. -/
def brownies_problem (monday : ℕ) : Prop :=
  ∃ (tuesday wednesday : ℕ),
    tuesday = 2 * monday ∧
    wednesday = 3 * tuesday ∧
    monday + tuesday + wednesday = 45

/-- Theorem stating that Katy made 45 brownies in total. -/
theorem katy_brownies : brownies_problem 5 := by
  sorry

end NUMINAMATH_CALUDE_katy_brownies_l2665_266598


namespace NUMINAMATH_CALUDE_abc_inequality_l2665_266580

theorem abc_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  a^a * b^b * c^c ≥ (a*b*c)^((a+b+c)/3) := by
  sorry

end NUMINAMATH_CALUDE_abc_inequality_l2665_266580


namespace NUMINAMATH_CALUDE_chubby_checkerboard_black_squares_l2665_266586

/-- Represents a square on the checkerboard -/
inductive Square
| Black
| Red

/-- Represents a checkerboard -/
def Checkerboard := Array (Array Square)

/-- Creates a checkerboard with the given dimensions and pattern -/
def createCheckerboard (n : Nat) : Checkerboard :=
  sorry

/-- Counts the number of black squares on the checkerboard -/
def countBlackSquares (board : Checkerboard) : Nat :=
  sorry

theorem chubby_checkerboard_black_squares :
  let board := createCheckerboard 29
  countBlackSquares board = 421 := by
  sorry

end NUMINAMATH_CALUDE_chubby_checkerboard_black_squares_l2665_266586


namespace NUMINAMATH_CALUDE_pau_total_chicken_l2665_266543

def kobe_order : ℕ := 5

def pau_order (kobe : ℕ) : ℕ := 2 * kobe

def total_pau_order (initial : ℕ) : ℕ := 2 * initial

theorem pau_total_chicken :
  total_pau_order (pau_order kobe_order) = 20 := by
  sorry

end NUMINAMATH_CALUDE_pau_total_chicken_l2665_266543


namespace NUMINAMATH_CALUDE_calculate_expression_l2665_266559

theorem calculate_expression : (2023 - Real.pi) ^ 0 - (1 / 4)⁻¹ + |(-2)| + Real.sqrt 9 = 2 := by
  sorry

end NUMINAMATH_CALUDE_calculate_expression_l2665_266559


namespace NUMINAMATH_CALUDE_inverse_prop_t_times_function_no_linear_2k_times_function_quadratic_5_times_function_l2665_266547

/-- Definition of a "t times function" on [a,b] -/
def is_t_times_function (f : ℝ → ℝ) (t a b : ℝ) : Prop :=
  a < b ∧ t > 0 ∧ ∀ x ∈ Set.Icc a b, t * a ≤ f x ∧ f x ≤ t * b

/-- Part 1: Inverse proportional function -/
theorem inverse_prop_t_times_function :
  ∀ t > 0, is_t_times_function (fun x ↦ 2023 / x) t 1 2023 ↔ t = 1 := by sorry

/-- Part 2: Non-existence of linear "2k times function" -/
theorem no_linear_2k_times_function :
  ∀ k > 0, ∀ a b : ℝ, a < b →
    ¬∃ (c : ℝ), is_t_times_function (fun x ↦ k * x + c) (2 * k) a b := by sorry

/-- Part 3: Quadratic "5 times function" -/
theorem quadratic_5_times_function :
  ∀ a b : ℝ, is_t_times_function (fun x ↦ x^2 - 4*x - 7) 5 a b ↔
    (a = -2 ∧ b = 1) ∨ (a = -11/5 ∧ b = (9 + Real.sqrt 109) / 2) := by sorry

end NUMINAMATH_CALUDE_inverse_prop_t_times_function_no_linear_2k_times_function_quadratic_5_times_function_l2665_266547


namespace NUMINAMATH_CALUDE_unique_solution_g100_l2665_266514

-- Define g₀(x)
def g₀ (x : ℝ) : ℝ := 2 * x + |x - 50| - |x + 50|

-- Define gₙ(x) recursively
def g (n : ℕ) (x : ℝ) : ℝ :=
  match n with
  | 0 => g₀ x
  | n + 1 => |g n x| - 2

-- Theorem statement
theorem unique_solution_g100 :
  ∃! x, g 100 x = 0 :=
sorry

end NUMINAMATH_CALUDE_unique_solution_g100_l2665_266514


namespace NUMINAMATH_CALUDE_zeros_of_f_l2665_266539

-- Define the piecewise function f
noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 0 then x^2 - 2*x - 3 else -2 + Real.log x

-- State the theorem about the zeros of f
theorem zeros_of_f :
  ∃ (x₁ x₂ : ℝ), x₁ = -1 ∧ x₂ = Real.exp 2 ∧
  (∀ x : ℝ, f x = 0 ↔ x = x₁ ∨ x = x₂) :=
sorry

end NUMINAMATH_CALUDE_zeros_of_f_l2665_266539


namespace NUMINAMATH_CALUDE_recommended_apps_proof_l2665_266518

/-- The recommended number of apps for Roger's phone -/
def recommended_apps : ℕ := 35

/-- The maximum number of apps for optimal function -/
def max_optimal_apps : ℕ := 50

/-- The number of apps Roger currently has -/
def rogers_current_apps : ℕ := 2 * recommended_apps

/-- The number of apps Roger needs to delete -/
def apps_to_delete : ℕ := 20

theorem recommended_apps_proof :
  (rogers_current_apps = max_optimal_apps + apps_to_delete) ∧
  (rogers_current_apps = 2 * recommended_apps) ∧
  (max_optimal_apps = 50) ∧
  (apps_to_delete = 20) →
  recommended_apps = 35 := by sorry

end NUMINAMATH_CALUDE_recommended_apps_proof_l2665_266518


namespace NUMINAMATH_CALUDE_x_difference_is_22_l2665_266509

theorem x_difference_is_22 (x : ℝ) (h : (x + 3)^2 / (3*x + 65) = 2) :
  ∃ (x₁ x₂ : ℝ), ((x₁ + 3)^2 / (3*x₁ + 65) = 2) ∧
                 ((x₂ + 3)^2 / (3*x₂ + 65) = 2) ∧
                 (x₁ ≠ x₂) ∧
                 (x₁ - x₂ = 22 ∨ x₂ - x₁ = 22) :=
by sorry

end NUMINAMATH_CALUDE_x_difference_is_22_l2665_266509


namespace NUMINAMATH_CALUDE_village_population_l2665_266532

theorem village_population (partial_population : ℕ) (percentage : ℚ) (total_population : ℕ) : 
  percentage = 80 / 100 →
  partial_population = 32000 →
  percentage * (total_population : ℚ) = partial_population →
  total_population = 40000 :=
by
  sorry

end NUMINAMATH_CALUDE_village_population_l2665_266532


namespace NUMINAMATH_CALUDE_optimal_rectangle_area_l2665_266546

/-- Given a rectangle with perimeter 400 feet, length at least 100 feet, and width at least 50 feet,
    the maximum possible area is 10,000 square feet. -/
theorem optimal_rectangle_area (l w : ℝ) (h1 : l + w = 200) (h2 : l ≥ 100) (h3 : w ≥ 50) :
  l * w ≤ 10000 :=
by sorry

end NUMINAMATH_CALUDE_optimal_rectangle_area_l2665_266546


namespace NUMINAMATH_CALUDE_unique_triple_l2665_266538

theorem unique_triple : ∃! (x y z : ℕ), 
  x > 1 ∧ y > 1 ∧ z > 1 ∧
  (yz - 1) % x = 0 ∧ 
  (zx - 1) % y = 0 ∧ 
  (xy - 1) % z = 0 ∧
  x = 5 ∧ y = 3 ∧ z = 2 := by
  sorry

end NUMINAMATH_CALUDE_unique_triple_l2665_266538


namespace NUMINAMATH_CALUDE_F_properties_l2665_266517

-- Define the function f
def f (a b : ℝ) (h1 : 0 < b) (h2 : b < -a) : ℝ → ℝ := sorry

-- Define the function F
def F (a b : ℝ) (h1 : 0 < b) (h2 : b < -a) (x : ℝ) : ℝ :=
  (f a b h1 h2 x)^2 - (f a b h1 h2 (-x))^2

-- State the theorem
theorem F_properties (a b : ℝ) (h1 : 0 < b) (h2 : b < -a) :
  (∀ x, F a b h1 h2 x ≠ 0) →
  (∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → f a b h1 h2 x < f a b h1 h2 y) →
  (∀ x, F a b h1 h2 x = 0 ∨ -b ≤ x ∧ x ≤ b) ∧
  (∀ x, F a b h1 h2 (-x) = -(F a b h1 h2 x)) :=
by sorry

end NUMINAMATH_CALUDE_F_properties_l2665_266517


namespace NUMINAMATH_CALUDE_playground_total_l2665_266565

/-- The number of people on a playground --/
structure Playground where
  girls : ℕ
  boys : ℕ
  thirdGradeGirls : ℕ
  thirdGradeBoys : ℕ
  teachers : ℕ
  maleTeachers : ℕ
  femaleTeachers : ℕ

/-- The total number of people on the playground is 67 --/
theorem playground_total (p : Playground)
  (h1 : p.girls = 28)
  (h2 : p.boys = 35)
  (h3 : p.thirdGradeGirls = 15)
  (h4 : p.thirdGradeBoys = 18)
  (h5 : p.teachers = 4)
  (h6 : p.maleTeachers = 2)
  (h7 : p.femaleTeachers = 2)
  (h8 : p.teachers = p.maleTeachers + p.femaleTeachers) :
  p.girls + p.boys + p.teachers = 67 := by
  sorry

#check playground_total

end NUMINAMATH_CALUDE_playground_total_l2665_266565


namespace NUMINAMATH_CALUDE_expression_evaluation_l2665_266548

theorem expression_evaluation : (120 / 6 * 2 / 3 : ℚ) = 40 / 3 := by sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2665_266548


namespace NUMINAMATH_CALUDE_diophantine_equation_solutions_l2665_266571

theorem diophantine_equation_solutions :
  ∀ (x y : ℕ) (p : ℕ), 
    Prime p → 
    p^x - y^p = 1 → 
    ((x = 1 ∧ y = 1 ∧ p = 2) ∨ (x = 2 ∧ y = 2 ∧ p = 3)) :=
by sorry

end NUMINAMATH_CALUDE_diophantine_equation_solutions_l2665_266571


namespace NUMINAMATH_CALUDE_zack_traveled_20_countries_l2665_266528

def alex_countries : ℕ := 30

def george_countries (alex : ℕ) : ℚ := (3 : ℚ) / 5 * alex

def joseph_countries (george : ℚ) : ℚ := (1 : ℚ) / 3 * george

def patrick_countries (joseph : ℚ) : ℚ := (4 : ℚ) / 3 * joseph

def zack_countries (patrick : ℚ) : ℚ := (5 : ℚ) / 2 * patrick

theorem zack_traveled_20_countries :
  zack_countries (patrick_countries (joseph_countries (george_countries alex_countries))) = 20 := by
  sorry

end NUMINAMATH_CALUDE_zack_traveled_20_countries_l2665_266528


namespace NUMINAMATH_CALUDE_triangle_side_length_l2665_266510

theorem triangle_side_length (a b c : ℝ) (A B C : ℝ) :
  -- Triangle ABC exists
  0 < a ∧ 0 < b ∧ 0 < c ∧
  0 < A ∧ A < pi ∧ 0 < B ∧ B < pi ∧ 0 < C ∧ C < pi ∧
  A + B + C = pi →
  -- Given conditions
  a = 2 →
  B = pi / 3 →
  b = Real.sqrt 7 →
  -- Conclusion
  c = 3 := by
  sorry

end NUMINAMATH_CALUDE_triangle_side_length_l2665_266510


namespace NUMINAMATH_CALUDE_arctan_of_tan_difference_l2665_266557

-- Define the problem parameters
def angle₁ : Real := 80
def angle₂ : Real := 30

-- Define the theorem
theorem arctan_of_tan_difference (h : 0 ≤ angle₁ ∧ angle₁ ≤ 180 ∧ 0 ≤ angle₂ ∧ angle₂ ≤ 180) :
  Real.arctan (Real.tan (angle₁ * π / 180) - 3 * Real.tan (angle₂ * π / 180)) * 180 / π = angle₁ := by
  sorry


end NUMINAMATH_CALUDE_arctan_of_tan_difference_l2665_266557


namespace NUMINAMATH_CALUDE_min_value_reciprocal_sum_l2665_266588

theorem min_value_reciprocal_sum (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x + 3 * y = 1) :
  (∀ x' y' : ℝ, x' > 0 → y' > 0 → x' + 3 * y' = 1 → 1 / x' + 1 / y' ≥ 1 / x + 1 / y) →
  1 / x + 1 / y = 3 + Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_min_value_reciprocal_sum_l2665_266588


namespace NUMINAMATH_CALUDE_equilateral_triangle_height_l2665_266536

/-- Given an equilateral triangle with two vertices at (0,0) and (10,0), 
    and the third vertex (x,y) in the first quadrant, 
    prove that the y-coordinate of the third vertex is 5√3. -/
theorem equilateral_triangle_height : 
  ∀ (x y : ℝ), 
  x ≥ 0 → y > 0 →  -- First quadrant condition
  (x^2 + y^2 = 100) →  -- Distance from (0,0) to (x,y) is 10
  ((x-10)^2 + y^2 = 100) →  -- Distance from (10,0) to (x,y) is 10
  y = 5 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_equilateral_triangle_height_l2665_266536


namespace NUMINAMATH_CALUDE_reflected_quad_area_l2665_266567

/-- A convex quadrilateral in the plane -/
structure ConvexQuadrilateral where
  -- We don't need to define the specifics of the quadrilateral,
  -- just that it exists and has an area
  area : ℝ
  area_pos : area > 0

/-- The quadrilateral formed by reflecting a point inside a convex quadrilateral 
    with respect to the midpoints of its sides -/
def reflectedQuadrilateral (Q : ConvexQuadrilateral) : ConvexQuadrilateral where
  -- We don't need to define how this quadrilateral is constructed,
  -- just that it exists and is related to the original quadrilateral
  area := 2 * Q.area
  area_pos := by
    -- The proof that the area is positive
    sorry

/-- Theorem stating that the area of the reflected quadrilateral 
    is twice the area of the original quadrilateral -/
theorem reflected_quad_area (Q : ConvexQuadrilateral) :
  (reflectedQuadrilateral Q).area = 2 * Q.area := by
  -- The proof of the theorem
  sorry

end NUMINAMATH_CALUDE_reflected_quad_area_l2665_266567


namespace NUMINAMATH_CALUDE_sin_cos_shift_l2665_266540

theorem sin_cos_shift (x : ℝ) : 
  Real.sin (2 * x + π / 6) = Real.cos (2 * x - π / 6 + π / 2 - π / 12) := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_shift_l2665_266540


namespace NUMINAMATH_CALUDE_pyramid_frustum_theorem_l2665_266570

-- Define the pyramid
structure Pyramid :=
  (base_side : ℝ)
  (height : ℝ)

-- Define the frustum
structure Frustum :=
  (base_side : ℝ)
  (top_side : ℝ)
  (height : ℝ)

-- Define the theorem
theorem pyramid_frustum_theorem (P : Pyramid) (F : Frustum) (P' : Pyramid) :
  P.base_side = 10 →
  P.height = 15 →
  F.base_side = P.base_side →
  F.top_side = P'.base_side →
  F.height + P'.height = P.height →
  (P.base_side^2 * P.height) = 9 * (P'.base_side^2 * P'.height) →
  ∃ (S : ℝ × ℝ × ℝ) (V : ℝ × ℝ × ℝ),
    S.2.2 = F.height / 2 + P'.height ∧
    V.2.2 = P.height ∧
    Real.sqrt ((S.1 - V.1)^2 + (S.2.1 - V.2.1)^2 + (S.2.2 - V.2.2)^2) = 2.5 :=
by sorry

end NUMINAMATH_CALUDE_pyramid_frustum_theorem_l2665_266570


namespace NUMINAMATH_CALUDE_money_division_l2665_266590

theorem money_division (total : ℕ) (p q r : ℕ) : 
  p + q + r = total →
  3 * p = 7 * q →
  7 * q = 12 * r →
  q - p = 4500 →
  r - q = 4500 →
  q - p = 3600 := by
sorry

end NUMINAMATH_CALUDE_money_division_l2665_266590


namespace NUMINAMATH_CALUDE_basketball_lineup_combinations_l2665_266541

theorem basketball_lineup_combinations (total_players : ℕ) (lineup_size : ℕ) (guaranteed_players : ℕ) :
  total_players = 15 →
  lineup_size = 6 →
  guaranteed_players = 2 →
  Nat.choose (total_players - guaranteed_players) (lineup_size - guaranteed_players) = 715 :=
by sorry

end NUMINAMATH_CALUDE_basketball_lineup_combinations_l2665_266541


namespace NUMINAMATH_CALUDE_right_triangle_ratio_l2665_266527

theorem right_triangle_ratio (a b c r s : ℝ) : 
  a > 0 → b > 0 → c > 0 → r > 0 → s > 0 →
  a^2 + b^2 = c^2 →  -- Pythagorean theorem
  c = r + s →        -- c is divided into r and s
  r * c = a^2 →      -- Geometric mean theorem for r
  s * c = b^2 →      -- Geometric mean theorem for s
  a / b = 2 / 5 →    -- Given ratio of a to b
  r / s = 4 / 25     -- Conclusion: ratio of r to s
  := by sorry

end NUMINAMATH_CALUDE_right_triangle_ratio_l2665_266527


namespace NUMINAMATH_CALUDE_inverse_proportion_k_value_l2665_266511

theorem inverse_proportion_k_value (k : ℝ) (h1 : k ≠ 0) :
  (∀ x : ℝ, x ≠ 0 → (k / x) = -1/2 ↔ x = 4) → k = -2 := by
  sorry

end NUMINAMATH_CALUDE_inverse_proportion_k_value_l2665_266511


namespace NUMINAMATH_CALUDE_swimming_time_against_current_l2665_266561

theorem swimming_time_against_current 
  (swimming_speed : ℝ) 
  (water_speed : ℝ) 
  (time_with_current : ℝ) 
  (h1 : swimming_speed = 4) 
  (h2 : water_speed = 2) 
  (h3 : time_with_current = 4) : 
  (swimming_speed + water_speed) * time_with_current / (swimming_speed - water_speed) = 12 := by
  sorry

end NUMINAMATH_CALUDE_swimming_time_against_current_l2665_266561


namespace NUMINAMATH_CALUDE_min_sum_reciprocals_equality_condition_l2665_266533

theorem min_sum_reciprocals (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  a / (3 * b) + b / (5 * c) + c / (7 * a) ≥ 3 / Real.rpow 105 (1/3) :=
sorry

theorem equality_condition (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a / (3 * b) = b / (5 * c) ∧ b / (5 * c) = c / (7 * a)) ↔
  (a / (3 * b) = 1 / Real.rpow 105 (1/3) ∧
   b / (5 * c) = 1 / Real.rpow 105 (1/3) ∧
   c / (7 * a) = 1 / Real.rpow 105 (1/3)) :=
sorry

end NUMINAMATH_CALUDE_min_sum_reciprocals_equality_condition_l2665_266533


namespace NUMINAMATH_CALUDE_complement_union_problem_l2665_266562

def U : Finset Nat := {1, 2, 3, 4, 5}
def A : Finset Nat := {1, 3, 5}
def B : Finset Nat := {2, 3}

theorem complement_union_problem : 
  (U \ A) ∪ B = {2, 3, 4} := by sorry

end NUMINAMATH_CALUDE_complement_union_problem_l2665_266562


namespace NUMINAMATH_CALUDE_tangent_circle_height_difference_l2665_266526

/-- A circle tangent to the parabola y = x^2 + 1 at two points and lying inside the parabola -/
structure TangentCircle where
  /-- x-coordinate of the point of tangency -/
  a : ℝ
  /-- y-coordinate of the center of the circle -/
  b : ℝ
  /-- radius of the circle -/
  r : ℝ
  /-- The circle is tangent to the parabola at (a, a^2 + 1) and (-a, a^2 + 1) -/
  tangent_point : b = a^2 + 1/2
  /-- The circle equation satisfies the tangency condition -/
  circle_eq : b^2 - r^2 = a^4 + 1

/-- The difference in height between the center of the circle and the points of tangency is -1/2 -/
theorem tangent_circle_height_difference (c : TangentCircle) :
  c.b - (c.a^2 + 1) = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_tangent_circle_height_difference_l2665_266526


namespace NUMINAMATH_CALUDE_binomial_26_3_minus_10_l2665_266508

theorem binomial_26_3_minus_10 : Nat.choose 26 3 - 10 = 2590 := by sorry

end NUMINAMATH_CALUDE_binomial_26_3_minus_10_l2665_266508


namespace NUMINAMATH_CALUDE_similar_triangles_leg_sum_l2665_266521

theorem similar_triangles_leg_sum (a b c d : ℝ) : 
  a > 0 → b > 0 → c > 0 → d > 0 →
  (1/2) * a * b = 24 →
  (1/2) * c * d = 600 →
  a^2 + b^2 = 100 →
  (c / a)^2 = 25 →
  (d / b)^2 = 25 →
  c + d = 70 := by
sorry

end NUMINAMATH_CALUDE_similar_triangles_leg_sum_l2665_266521


namespace NUMINAMATH_CALUDE_geometric_series_common_ratio_l2665_266544

theorem geometric_series_common_ratio 
  (a : ℕ → ℝ) 
  (h_geometric : ∀ n : ℕ, a (n + 1) = a n * (a 2 / a 1)) 
  (h_sum1 : a 1 + a 3 = 10) 
  (h_sum2 : a 4 + a 6 = 5/4) : 
  a 2 / a 1 = 1/2 := by
sorry

end NUMINAMATH_CALUDE_geometric_series_common_ratio_l2665_266544


namespace NUMINAMATH_CALUDE_trig_expression_equality_l2665_266555

theorem trig_expression_equality : 
  let sin30 : ℝ := 1/2
  let cos45 : ℝ := Real.sqrt 2 / 2
  let tan60 : ℝ := Real.sqrt 3
  sin30 - Real.sqrt 3 * cos45 + Real.sqrt 2 * tan60 = (1 + Real.sqrt 6) / 2 := by
sorry

end NUMINAMATH_CALUDE_trig_expression_equality_l2665_266555


namespace NUMINAMATH_CALUDE_rectangular_box_surface_area_l2665_266534

theorem rectangular_box_surface_area 
  (a b c : ℝ) 
  (h1 : 4 * a + 4 * b + 4 * c = 160) 
  (h2 : Real.sqrt (a^2 + b^2 + c^2) = 25) : 
  2 * (a * b + b * c + c * a) = 975 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_box_surface_area_l2665_266534


namespace NUMINAMATH_CALUDE_solve_equation_l2665_266576

theorem solve_equation (x : ℝ) : (x ^ 3).sqrt = 9 * (81 ^ (1 / 9 : ℝ)) → x = 9 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l2665_266576


namespace NUMINAMATH_CALUDE_left_handed_jazz_lovers_count_l2665_266505

/-- Represents a club with members and their characteristics -/
structure Club where
  total_members : ℕ
  left_handed : ℕ
  jazz_lovers : ℕ
  right_handed_non_jazz : ℕ

/-- The number of left-handed jazz lovers in the club -/
def left_handed_jazz_lovers (c : Club) : ℕ :=
  c.total_members + c.right_handed_non_jazz - c.left_handed - c.jazz_lovers

/-- Theorem stating the number of left-handed jazz lovers in the given club -/
theorem left_handed_jazz_lovers_count (c : Club) 
  (h1 : c.total_members = 30)
  (h2 : c.left_handed = 12)
  (h3 : c.jazz_lovers = 22)
  (h4 : c.right_handed_non_jazz = 4)
  (h5 : c.left_handed + (c.total_members - c.left_handed) = c.total_members) :
  left_handed_jazz_lovers c = 8 := by
  sorry

#eval left_handed_jazz_lovers ⟨30, 12, 22, 4⟩

end NUMINAMATH_CALUDE_left_handed_jazz_lovers_count_l2665_266505


namespace NUMINAMATH_CALUDE_student_number_problem_l2665_266573

theorem student_number_problem (x : ℝ) : 3 * x - 220 = 110 → x = 110 := by
  sorry

end NUMINAMATH_CALUDE_student_number_problem_l2665_266573


namespace NUMINAMATH_CALUDE_tan_alpha_value_l2665_266502

theorem tan_alpha_value (α : Real) (h : Real.tan (α + π/4) = 2) : Real.tan α = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_tan_alpha_value_l2665_266502


namespace NUMINAMATH_CALUDE_kyuhyung_candies_l2665_266542

theorem kyuhyung_candies :
  ∀ (k d : ℕ), -- k for Kyuhyung's candies, d for Dongmin's candies
  d = k + 5 →   -- Dongmin has 5 more candies than Kyuhyung
  k + d = 43 → -- The sum of their candies is 43
  k = 19       -- Kyuhyung has 19 candies
  := by sorry

end NUMINAMATH_CALUDE_kyuhyung_candies_l2665_266542


namespace NUMINAMATH_CALUDE_remove_four_gives_desired_average_l2665_266552

def original_list : List Nat := [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]

def remove_number (list : List Nat) (n : Nat) : List Nat :=
  list.filter (· ≠ n)

def average (list : List Nat) : Rat :=
  (list.sum : Rat) / list.length

theorem remove_four_gives_desired_average :
  average (remove_number original_list 4) = 29/4 := by
sorry

end NUMINAMATH_CALUDE_remove_four_gives_desired_average_l2665_266552


namespace NUMINAMATH_CALUDE_square_sum_given_sum_and_product_l2665_266522

theorem square_sum_given_sum_and_product (a b : ℝ) 
  (h1 : a + b = 7) (h2 : a * b = 10) : a^2 + b^2 = 29 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_given_sum_and_product_l2665_266522


namespace NUMINAMATH_CALUDE_remainder_product_theorem_l2665_266535

theorem remainder_product_theorem (P Q R k : ℤ) (hk : k > 0) (hprod : P * Q = R) :
  (P % k * Q % k) % k = R % k :=
by sorry

end NUMINAMATH_CALUDE_remainder_product_theorem_l2665_266535


namespace NUMINAMATH_CALUDE_problem_statement_l2665_266531

theorem problem_statement (a b : ℝ) (h : (a + 1)^2 + |b - 2| = 0) :
  a^2006 + (a + b)^2007 = 2 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l2665_266531


namespace NUMINAMATH_CALUDE_rectangle_segment_sum_l2665_266524

theorem rectangle_segment_sum (a b : ℝ) (n : ℕ) (h1 : a = 4) (h2 : b = 3) (h3 : n = 168) :
  let diagonal := Real.sqrt (a^2 + b^2)
  let segment_sum := n * diagonal
  segment_sum = 840 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_segment_sum_l2665_266524


namespace NUMINAMATH_CALUDE_replaced_person_weight_l2665_266515

def group_size : ℕ := 8
def average_weight_increase : ℝ := 2.5
def new_person_weight : ℝ := 90

theorem replaced_person_weight :
  let total_weight_increase : ℝ := group_size * average_weight_increase
  let replaced_weight : ℝ := new_person_weight - total_weight_increase
  replaced_weight = 70 := by sorry

end NUMINAMATH_CALUDE_replaced_person_weight_l2665_266515


namespace NUMINAMATH_CALUDE_instantaneous_velocity_at_2_l2665_266519

/-- The displacement function of the object -/
def h (t : ℝ) : ℝ := 14 * t - t^2

/-- The velocity function of the object -/
def v (t : ℝ) : ℝ := 14 - 2 * t

/-- Theorem: The instantaneous velocity at t = 2 seconds is 10 meters/second -/
theorem instantaneous_velocity_at_2 : v 2 = 10 := by sorry

end NUMINAMATH_CALUDE_instantaneous_velocity_at_2_l2665_266519


namespace NUMINAMATH_CALUDE_x_squared_minus_one_is_quadratic_l2665_266596

/-- Definition of a quadratic equation in one variable -/
def is_quadratic_equation (f : ℝ → ℝ) : Prop :=
  ∃ (a b c : ℝ), a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

/-- The function representing x^2 - 1 -/
def f (x : ℝ) : ℝ := x^2 - 1

/-- Theorem: x^2 - 1 = 0 is a quadratic equation in one variable -/
theorem x_squared_minus_one_is_quadratic : is_quadratic_equation f := by
  sorry

end NUMINAMATH_CALUDE_x_squared_minus_one_is_quadratic_l2665_266596


namespace NUMINAMATH_CALUDE_max_profit_toy_sales_exists_max_profit_price_l2665_266572

/-- Represents the profit function for toy sales -/
def profit_function (x : ℝ) : ℝ := -10 * x^2 + 1300 * x - 30000

/-- Represents the sales volume function for toy sales -/
def sales_volume (x : ℝ) : ℝ := -10 * x + 1000

/-- The maximum profit theorem for toy sales -/
theorem max_profit_toy_sales :
  ∀ x : ℝ,
  (x ≥ 44) →
  (x ≤ 46) →
  (sales_volume x ≥ 540) →
  profit_function x ≤ 8640 :=
by
  sorry

/-- The existence of a selling price that achieves the maximum profit -/
theorem exists_max_profit_price :
  ∃ x : ℝ,
  (x ≥ 44) ∧
  (x ≤ 46) ∧
  (sales_volume x ≥ 540) ∧
  profit_function x = 8640 :=
by
  sorry

end NUMINAMATH_CALUDE_max_profit_toy_sales_exists_max_profit_price_l2665_266572


namespace NUMINAMATH_CALUDE_adiabatic_compression_work_l2665_266529

/-- Adiabatic compression work in a cylindrical vessel -/
theorem adiabatic_compression_work
  (V₀ V₁ p₀ k : ℝ)
  (h₀ : V₀ > 0)
  (h₁ : V₁ > 0)
  (h₂ : p₀ > 0)
  (h₃ : k > 1)
  (h₄ : V₁ < V₀) :
  ∃ W : ℝ, W = (p₀ * V₀ / (k - 1)) * ((V₀ / V₁) ^ (k - 1) - 1) :=
by sorry

end NUMINAMATH_CALUDE_adiabatic_compression_work_l2665_266529


namespace NUMINAMATH_CALUDE_distribute_teachers_count_l2665_266591

/-- Represents the number of schools --/
def num_schools : ℕ := 3

/-- Represents the number of teachers --/
def num_teachers : ℕ := 5

/-- Represents the constraint that each school must have at least one teacher --/
def min_teachers_per_school : ℕ := 1

/-- The function that calculates the number of ways to distribute teachers --/
def distribute_teachers : ℕ := sorry

/-- The theorem stating that the number of ways to distribute teachers is 150 --/
theorem distribute_teachers_count : distribute_teachers = 150 := by sorry

end NUMINAMATH_CALUDE_distribute_teachers_count_l2665_266591


namespace NUMINAMATH_CALUDE_master_percentage_is_76_l2665_266593

/-- Represents a team of junior and master players -/
structure Team where
  juniors : ℕ
  masters : ℕ

/-- The average score of the entire team -/
def teamAverage (t : Team) (juniorAvg masterAvg : ℚ) : ℚ :=
  (juniorAvg * t.juniors + masterAvg * t.masters) / (t.juniors + t.masters)

/-- The percentage of masters in the team -/
def masterPercentage (t : Team) : ℚ :=
  t.masters * 100 / (t.juniors + t.masters)

theorem master_percentage_is_76 (t : Team) :
  teamAverage t 22 47 = 41 →
  masterPercentage t = 76 := by
  sorry

#check master_percentage_is_76

end NUMINAMATH_CALUDE_master_percentage_is_76_l2665_266593


namespace NUMINAMATH_CALUDE_largest_prime_factor_is_17_l2665_266545

def numbers : List Nat := [210, 255, 143, 187, 169]

def is_prime (n : Nat) : Prop := 
  n > 1 ∧ ∀ m : Nat, m > 1 → m < n → ¬(n % m = 0)

def prime_factors (n : Nat) : Set Nat :=
  {p : Nat | is_prime p ∧ n % p = 0}

theorem largest_prime_factor_is_17 : 
  ∃ (n : Nat), n ∈ numbers ∧ 
    (∃ (p : Nat), p ∈ prime_factors n ∧ p = 17 ∧ 
      ∀ (m : Nat) (q : Nat), m ∈ numbers → q ∈ prime_factors m → q ≤ 17) :=
by sorry

end NUMINAMATH_CALUDE_largest_prime_factor_is_17_l2665_266545


namespace NUMINAMATH_CALUDE_complement_of_M_l2665_266507

-- Define the set M
def M : Set ℝ := {x | x^2 - 2*x > 0}

-- State the theorem
theorem complement_of_M : 
  (Set.univ : Set ℝ) \ M = Set.Icc 0 2 := by sorry

end NUMINAMATH_CALUDE_complement_of_M_l2665_266507


namespace NUMINAMATH_CALUDE_ferry_tourists_sum_l2665_266595

/-- The number of trips the ferry makes in a day -/
def num_trips : ℕ := 6

/-- The number of tourists on the first trip -/
def initial_tourists : ℕ := 100

/-- The decrease in number of tourists per trip -/
def tourist_decrease : ℕ := 1

/-- The sum of an arithmetic sequence -/
def arithmetic_sum (a₁ : ℕ) (d : ℕ) (n : ℕ) : ℕ :=
  n * (2 * a₁ + (n - 1) * d) / 2

/-- The total number of tourists transported by the ferry in a day -/
def total_tourists : ℕ := arithmetic_sum initial_tourists tourist_decrease num_trips

theorem ferry_tourists_sum :
  total_tourists = 585 := by sorry

end NUMINAMATH_CALUDE_ferry_tourists_sum_l2665_266595


namespace NUMINAMATH_CALUDE_total_cost_proof_l2665_266578

def price_AVN : ℝ := 12
def price_TheDark : ℝ := 2 * price_AVN
def num_TheDark : ℕ := 2
def num_AVN : ℕ := 1
def ratio_90s : ℝ := 0.4
def num_90s : ℕ := 5

theorem total_cost_proof :
  let cost_main := price_TheDark * num_TheDark + price_AVN * num_AVN
  let cost_90s := ratio_90s * cost_main * num_90s
  cost_main + cost_90s = 180 := by sorry

end NUMINAMATH_CALUDE_total_cost_proof_l2665_266578


namespace NUMINAMATH_CALUDE_min_perimeter_area_l2665_266589

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x^2 - y^2/8 = 1

-- Define the right focus F
def rightFocus : ℝ × ℝ := (3, 0)

-- Define point A
def A : ℝ × ℝ := (0, 4)

-- Define a point P on the left branch of the hyperbola
def P : ℝ × ℝ := sorry

-- Define the perimeter of triangle APF
def perimeter (p : ℝ × ℝ) : ℝ := sorry

-- Define the area of triangle APF
def area (p : ℝ × ℝ) : ℝ := sorry

-- Theorem statement
theorem min_perimeter_area :
  ∃ (p : ℝ × ℝ), 
    hyperbola p.1 p.2 ∧ 
    p.1 < 0 ∧ 
    (∀ q : ℝ × ℝ, hyperbola q.1 q.2 ∧ q.1 < 0 → perimeter p ≤ perimeter q) ∧
    area p = 36/7 :=
sorry

end NUMINAMATH_CALUDE_min_perimeter_area_l2665_266589


namespace NUMINAMATH_CALUDE_acute_angle_relation_l2665_266584

theorem acute_angle_relation (α β : Real) (h1 : 0 < α ∧ α < π/2) (h2 : 0 < β ∧ β < π/2)
  (h3 : Real.sin α = (1/2) * Real.sin (α + β)) : α < β := by
  sorry

end NUMINAMATH_CALUDE_acute_angle_relation_l2665_266584


namespace NUMINAMATH_CALUDE_area_of_graph_region_l2665_266579

/-- The equation of the graph -/
def graph_equation (x y : ℝ) : Prop :=
  |x - 80| + |y| = |x / 5|

/-- The region enclosed by the graph -/
def enclosed_region : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | graph_equation p.1 p.2}

/-- The area of the enclosed region -/
noncomputable def area_of_region : ℝ := sorry

theorem area_of_graph_region :
  area_of_region = 800 :=
sorry

end NUMINAMATH_CALUDE_area_of_graph_region_l2665_266579


namespace NUMINAMATH_CALUDE_connie_tickets_l2665_266568

/-- The number of tickets Connie redeemed -/
def total_tickets : ℕ := 50

/-- The number of tickets spent on earbuds -/
def earbuds_tickets : ℕ := 10

/-- The number of tickets spent on glow bracelets -/
def glow_bracelets_tickets : ℕ := 15

/-- Theorem stating that Connie redeemed 50 tickets -/
theorem connie_tickets : 
  (total_tickets / 2 : ℚ) + earbuds_tickets + glow_bracelets_tickets = total_tickets := by
  sorry

#check connie_tickets

end NUMINAMATH_CALUDE_connie_tickets_l2665_266568


namespace NUMINAMATH_CALUDE_arithmetic_sequence_a7_l2665_266530

/-- An arithmetic sequence is a sequence where the difference between
    successive terms is constant. -/
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_a7 (a : ℕ → ℝ) :
  is_arithmetic_sequence a → a 3 = 2 → a 5 = 7 → a 7 = 12 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_a7_l2665_266530


namespace NUMINAMATH_CALUDE_animal_weight_comparison_l2665_266558

theorem animal_weight_comparison (chicken_weight duck_weight cow_weight : ℕ) 
  (h1 : chicken_weight = 3)
  (h2 : duck_weight = 6)
  (h3 : cow_weight = 624) :
  (cow_weight / chicken_weight = 208) ∧ (cow_weight / duck_weight = 104) := by
  sorry

end NUMINAMATH_CALUDE_animal_weight_comparison_l2665_266558


namespace NUMINAMATH_CALUDE_vacation_cost_is_120_l2665_266574

/-- Calculates the total cost of a vacation for two people. -/
def vacationCost (planeTicketCost hotelCostPerDay : ℕ) (durationInDays : ℕ) : ℕ :=
  2 * planeTicketCost + 2 * hotelCostPerDay * durationInDays

/-- Proves that the total cost of the vacation is $120. -/
theorem vacation_cost_is_120 :
  vacationCost 24 12 3 = 120 := by
  sorry

end NUMINAMATH_CALUDE_vacation_cost_is_120_l2665_266574


namespace NUMINAMATH_CALUDE_remainder_theorem_l2665_266592

theorem remainder_theorem (x : ℤ) (h : x % 11 = 7) : (x^3 - (2*x)^2) % 11 = 4 := by
  sorry

end NUMINAMATH_CALUDE_remainder_theorem_l2665_266592


namespace NUMINAMATH_CALUDE_consecutive_fibonacci_coprime_l2665_266569

def fib : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | n + 2 => fib (n + 1) + fib n

theorem consecutive_fibonacci_coprime (n : ℕ) (h : n ≥ 1) : 
  Nat.gcd (fib n) (fib (n - 1)) = 1 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_fibonacci_coprime_l2665_266569


namespace NUMINAMATH_CALUDE_function_inequality_l2665_266520

theorem function_inequality (f : ℝ → ℝ) 
  (h_diff : Differentiable ℝ f) 
  (h_cond : ∀ x, (x - 1) * deriv f x ≥ 0) : 
  f 0 + f 2 ≥ 2 * f 1 := by
  sorry

end NUMINAMATH_CALUDE_function_inequality_l2665_266520


namespace NUMINAMATH_CALUDE_correct_conclusions_l2665_266582

theorem correct_conclusions :
  (∀ x : ℝ, |x| = |-3| → x = 3 ∨ x = -3) ∧
  (∀ a b c : ℚ, a ≠ 0 → b ≠ 0 → c ≠ 0 →
    a < 0 → a + b < 0 → a + b + c < 0 →
    (|a| / a + |b| / b + |c| / c - |a * b * c| / (a * b * c) = 2 ∨
     |a| / a + |b| / b + |c| / c - |a * b * c| / (a * b * c) = -2)) :=
by sorry

end NUMINAMATH_CALUDE_correct_conclusions_l2665_266582


namespace NUMINAMATH_CALUDE_derivative_at_one_l2665_266550

-- Define the function f(x) = (x+1)^2(x-1)
def f (x : ℝ) : ℝ := (x + 1)^2 * (x - 1)

-- State the theorem
theorem derivative_at_one :
  deriv f 1 = 4 := by sorry

end NUMINAMATH_CALUDE_derivative_at_one_l2665_266550


namespace NUMINAMATH_CALUDE_work_time_ratio_l2665_266581

theorem work_time_ratio (time_A : ℝ) (combined_rate : ℝ) : 
  time_A = 10 → combined_rate = 0.3 → 
  ∃ time_B : ℝ, time_B / time_A = 1 / 2 :=
by
  sorry

end NUMINAMATH_CALUDE_work_time_ratio_l2665_266581


namespace NUMINAMATH_CALUDE_sum_of_fifth_terms_l2665_266523

def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

def is_geometric_sequence (b : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, b (n + 1) = b n * r

theorem sum_of_fifth_terms (a b : ℕ → ℝ) :
  is_arithmetic_sequence a →
  is_geometric_sequence b →
  a 1 + b 1 = 3 →
  a 2 + b 2 = 7 →
  a 3 + b 3 = 15 →
  a 4 + b 4 = 35 →
  a 5 + b 5 = 91 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_fifth_terms_l2665_266523
