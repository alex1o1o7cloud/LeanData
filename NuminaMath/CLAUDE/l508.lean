import Mathlib

namespace NUMINAMATH_CALUDE_problem_solutions_l508_50874

theorem problem_solutions :
  (∀ x : ℝ, 2 * x^2 - 3 * x + 4 > 0) ∧
  (∃ x : ℕ, x^2 ≤ x) ∧
  (∃ x : ℕ, 29 % x = 0) := by
  sorry

end NUMINAMATH_CALUDE_problem_solutions_l508_50874


namespace NUMINAMATH_CALUDE_abs_neg_three_l508_50875

theorem abs_neg_three : |(-3 : ℤ)| = 3 := by sorry

end NUMINAMATH_CALUDE_abs_neg_three_l508_50875


namespace NUMINAMATH_CALUDE_cut_scene_is_six_minutes_l508_50881

/-- The length of a cut scene from a movie --/
def cut_scene_length (original_length final_length : ℕ) : ℕ :=
  original_length - final_length

/-- Theorem: The length of the cut scene is 6 minutes --/
theorem cut_scene_is_six_minutes :
  let original_length : ℕ := 60  -- One hour in minutes
  let final_length : ℕ := 54
  cut_scene_length original_length final_length = 6 := by
  sorry

#eval cut_scene_length 60 54  -- This should output 6

end NUMINAMATH_CALUDE_cut_scene_is_six_minutes_l508_50881


namespace NUMINAMATH_CALUDE_work_completion_theorem_l508_50857

/-- Represents the number of days it takes for a person to complete the work alone -/
structure WorkRate :=
  (days : ℝ)
  (positive : days > 0)

/-- Represents the state of the work project -/
structure WorkProject :=
  (rate_a : WorkRate)
  (rate_b : WorkRate)
  (total_days : ℝ)
  (a_left_before : Bool)

/-- Calculate the number of days A left before completion -/
def days_a_left_before (project : WorkProject) : ℝ :=
  sorry

theorem work_completion_theorem (project : WorkProject) 
  (h1 : project.rate_a.days = 10)
  (h2 : project.rate_b.days = 20)
  (h3 : project.total_days = 10)
  (h4 : project.a_left_before = true) :
  days_a_left_before project = 5 := by
  sorry

end NUMINAMATH_CALUDE_work_completion_theorem_l508_50857


namespace NUMINAMATH_CALUDE_parabola_has_one_x_intercept_l508_50804

/-- The parabola equation -/
def parabola_equation (x y : ℝ) : Prop := x = -3 * y^2 + 2 * y + 2

/-- An x-intercept is a point where the parabola crosses the x-axis (y = 0) -/
def is_x_intercept (x : ℝ) : Prop := parabola_equation x 0

/-- The theorem stating that the parabola has exactly one x-intercept -/
theorem parabola_has_one_x_intercept : ∃! x : ℝ, is_x_intercept x := by sorry

end NUMINAMATH_CALUDE_parabola_has_one_x_intercept_l508_50804


namespace NUMINAMATH_CALUDE_A_minus_2B_specific_value_A_minus_2B_independent_of_x_l508_50890

/-- The algebraic expression A -/
def A (x y : ℝ) : ℝ := 2 * x^2 + 3 * x * y + 2 * y

/-- The algebraic expression B -/
def B (x y : ℝ) : ℝ := x^2 - x * y + x

/-- Theorem 1: A - 2B equals -20 when x = -2 and y = 3 -/
theorem A_minus_2B_specific_value : A (-2) 3 - 2 * B (-2) 3 = -20 := by sorry

/-- Theorem 2: A - 2B is independent of x when y = 2/5 -/
theorem A_minus_2B_independent_of_x : 
  ∀ x : ℝ, A x (2/5) - 2 * B x (2/5) = A 0 (2/5) - 2 * B 0 (2/5) := by sorry

end NUMINAMATH_CALUDE_A_minus_2B_specific_value_A_minus_2B_independent_of_x_l508_50890


namespace NUMINAMATH_CALUDE_m_equals_two_sufficient_not_necessary_l508_50859

-- Define sets A and B
def A (m : ℝ) : Set ℝ := {1, m^2}
def B : Set ℝ := {2, 4}

-- Define the property we want to prove
def property (m : ℝ) : Prop := A m ∩ B = {4}

-- Theorem statement
theorem m_equals_two_sufficient_not_necessary :
  (∀ m : ℝ, m = 2 → property m) ∧
  (∃ m : ℝ, m ≠ 2 ∧ property m) :=
sorry

end NUMINAMATH_CALUDE_m_equals_two_sufficient_not_necessary_l508_50859


namespace NUMINAMATH_CALUDE_principal_amount_proof_l508_50868

/-- Proves that given the specified conditions, the principal amount is 7200 --/
theorem principal_amount_proof (rate : ℝ) (time : ℝ) (diff : ℝ) (P : ℝ) 
  (h1 : rate = 5 / 100)
  (h2 : time = 2)
  (h3 : diff = 18)
  (h4 : P * (1 + rate)^time - P - (P * rate * time) = diff) :
  P = 7200 := by
  sorry

#check principal_amount_proof

end NUMINAMATH_CALUDE_principal_amount_proof_l508_50868


namespace NUMINAMATH_CALUDE_binomial_sum_problem_l508_50811

theorem binomial_sum_problem (n : ℕ) (M N : ℝ) : 
  M = (5 - 1/2)^n →  -- Sum of coefficients of (5x - 1/√x)^n
  N = 2^n →          -- Sum of binomial coefficients
  M - N = 240 → 
  N = 16 := by
sorry

end NUMINAMATH_CALUDE_binomial_sum_problem_l508_50811


namespace NUMINAMATH_CALUDE_root_sum_reciprocal_diff_l508_50805

-- Define the polynomial
def f (x : ℝ) : ℝ := 20 * x^3 - 40 * x^2 + 24 * x - 2

-- State the theorem
theorem root_sum_reciprocal_diff (a b c : ℝ) :
  f a = 0 → f b = 0 → f c = 0 →  -- a, b, c are roots of f
  a ≠ b → b ≠ c → a ≠ c →        -- roots are distinct
  0 < a → a < 1 →                -- a is between 0 and 1
  0 < b → b < 1 →                -- b is between 0 and 1
  0 < c → c < 1 →                -- c is between 0 and 1
  (1 / (1 - a)) + (1 / (1 - b)) + (1 / (1 - c)) = 1 :=
by sorry

end NUMINAMATH_CALUDE_root_sum_reciprocal_diff_l508_50805


namespace NUMINAMATH_CALUDE_gum_sticks_in_twelve_boxes_l508_50892

/-- Represents the number of sticks of gum in a given number of brown boxes -/
def sticks_in_boxes (full_boxes : ℕ) (half_boxes : ℕ) : ℕ :=
  let sticks_per_pack : ℕ := 5
  let packs_per_carton : ℕ := 7
  let cartons_per_full_box : ℕ := 6
  let cartons_per_half_box : ℕ := 3
  let sticks_per_carton : ℕ := sticks_per_pack * packs_per_carton
  let sticks_per_full_box : ℕ := sticks_per_carton * cartons_per_full_box
  let sticks_per_half_box : ℕ := sticks_per_carton * cartons_per_half_box
  full_boxes * sticks_per_full_box + half_boxes * sticks_per_half_box

/-- Theorem stating that 12 brown boxes with 2 half-full boxes contain 2310 sticks of gum -/
theorem gum_sticks_in_twelve_boxes : sticks_in_boxes 10 2 = 2310 := by
  sorry

end NUMINAMATH_CALUDE_gum_sticks_in_twelve_boxes_l508_50892


namespace NUMINAMATH_CALUDE_unpainted_cubes_count_l508_50834

/-- Represents a 6x6x6 cube composed of unit cubes -/
structure Cube :=
  (size : Nat)
  (total_units : Nat)
  (painted_per_face : Nat)
  (unpainted_columns : Nat)
  (unpainted_rows : Nat)

/-- The number of unpainted unit cubes in the cube -/
def unpainted_cubes (c : Cube) : Nat :=
  c.total_units - (c.painted_per_face * 6 - 24)

/-- Theorem stating the number of unpainted cubes in the specific cube configuration -/
theorem unpainted_cubes_count (c : Cube) 
  (h1 : c.size = 6)
  (h2 : c.total_units = 216)
  (h3 : c.painted_per_face = 10)
  (h4 : c.unpainted_columns = 2)
  (h5 : c.unpainted_rows = 2) :
  unpainted_cubes c = 168 := by
  sorry

end NUMINAMATH_CALUDE_unpainted_cubes_count_l508_50834


namespace NUMINAMATH_CALUDE_inequality_proof_l508_50854

theorem inequality_proof (x y z : ℝ) 
  (hx : 0 ≤ x ∧ x ≤ 1) 
  (hy : 0 ≤ y ∧ y ≤ 1) 
  (hz : 0 ≤ z ∧ z ≤ 1) : 
  x / (y + z + 1) + y / (z + x + 1) + z / (x + y + 1) ≤ 1 - (1 - x) * (1 - y) * (1 - z) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l508_50854


namespace NUMINAMATH_CALUDE_inequality_solution_range_l508_50823

theorem inequality_solution_range (a : ℝ) : 
  (∃! (x y : ℤ), x ≠ y ∧ 
    (∀ (z : ℤ), z^2 - (a+1)*z + a < 0 ↔ (z = x ∨ z = y))) ↔ 
  (a ∈ Set.Icc (-2) (-1) ∪ Set.Ioc 3 4) :=
sorry

end NUMINAMATH_CALUDE_inequality_solution_range_l508_50823


namespace NUMINAMATH_CALUDE_total_video_time_l508_50843

def cat_video_length : ℕ := 4

def dog_video_length (cat : ℕ) : ℕ := 2 * cat

def gorilla_video_length (cat : ℕ) : ℕ := cat ^ 2

def elephant_video_length (cat dog gorilla : ℕ) : ℕ := cat + dog + gorilla

def penguin_video_length (cat dog gorilla elephant : ℕ) : ℕ := (cat + dog + gorilla + elephant) ^ 3

def dolphin_video_length (cat dog gorilla elephant penguin : ℕ) : ℕ :=
  cat + dog + gorilla + elephant + penguin

theorem total_video_time :
  let cat := cat_video_length
  let dog := dog_video_length cat
  let gorilla := gorilla_video_length cat
  let elephant := elephant_video_length cat dog gorilla
  let penguin := penguin_video_length cat dog gorilla elephant
  let dolphin := dolphin_video_length cat dog gorilla elephant penguin
  cat + dog + gorilla + elephant + penguin + dolphin = 351344 := by
  sorry

end NUMINAMATH_CALUDE_total_video_time_l508_50843


namespace NUMINAMATH_CALUDE_zoo_consumption_theorem_l508_50815

/-- Represents the daily consumption of fish for an animal -/
structure DailyConsumption where
  trout : Float
  salmon : Float

/-- Calculates the total monthly consumption for all animals -/
def totalMonthlyConsumption (animals : List DailyConsumption) (days : Nat) : Float :=
  let dailyTotal := animals.foldl (fun acc x => acc + x.trout + x.salmon) 0
  dailyTotal * days.toFloat

/-- Theorem stating the total monthly consumption for the given animals -/
theorem zoo_consumption_theorem (pb1 pb2 pb3 sl1 sl2 : DailyConsumption)
    (h1 : pb1 = { trout := 0.2, salmon := 0.4 })
    (h2 : pb2 = { trout := 0.3, salmon := 0.5 })
    (h3 : pb3 = { trout := 0.25, salmon := 0.45 })
    (h4 : sl1 = { trout := 0.1, salmon := 0.15 })
    (h5 : sl2 = { trout := 0.2, salmon := 0.25 }) :
    totalMonthlyConsumption [pb1, pb2, pb3, sl1, sl2] 30 = 84 := by
  sorry


end NUMINAMATH_CALUDE_zoo_consumption_theorem_l508_50815


namespace NUMINAMATH_CALUDE_basketball_score_proof_l508_50803

-- Define the scores for each quarter
def alpha_scores (a r : ℝ) : Fin 4 → ℝ
| 0 => a
| 1 => a * r
| 2 => a * r^2
| 3 => a * r^3

def beta_scores (b d : ℝ) : Fin 4 → ℝ
| 0 => b
| 1 => b + d
| 2 => b + 2*d
| 3 => b + 3*d

-- Define the theorem
theorem basketball_score_proof 
  (a r b d : ℝ) 
  (h1 : 0 < r) -- Ensure increasing geometric sequence
  (h2 : 0 < d) -- Ensure increasing arithmetic sequence
  (h3 : alpha_scores a r 0 + alpha_scores a r 1 = beta_scores b d 0 + beta_scores b d 1) -- Tied at second quarter
  (h4 : (alpha_scores a r 0 + alpha_scores a r 1 + alpha_scores a r 2 + alpha_scores a r 3) = 
        (beta_scores b d 0 + beta_scores b d 1 + beta_scores b d 2 + beta_scores b d 3) + 2) -- Alpha wins by 2
  (h5 : (alpha_scores a r 0 + alpha_scores a r 1 + alpha_scores a r 2 + alpha_scores a r 3) ≤ 100) -- Alpha's total ≤ 100
  (h6 : (beta_scores b d 0 + beta_scores b d 1 + beta_scores b d 2 + beta_scores b d 3) ≤ 100) -- Beta's total ≤ 100
  : (alpha_scores a r 0 + alpha_scores a r 1 + beta_scores b d 0 + beta_scores b d 1) = 24 :=
by sorry


end NUMINAMATH_CALUDE_basketball_score_proof_l508_50803


namespace NUMINAMATH_CALUDE_fraction_equality_l508_50867

theorem fraction_equality (q r s t : ℚ) 
  (h1 : q / r = 8)
  (h2 : s / r = 12)
  (h3 : s / t = 4) :
  t / q = 3 / 8 := by
sorry

end NUMINAMATH_CALUDE_fraction_equality_l508_50867


namespace NUMINAMATH_CALUDE_percent_equivalence_l508_50853

theorem percent_equivalence (x : ℝ) (h : 0.3 * 0.05 * x = 18) : 0.05 * 0.3 * x = 18 := by
  sorry

end NUMINAMATH_CALUDE_percent_equivalence_l508_50853


namespace NUMINAMATH_CALUDE_gcd_problem_l508_50877

theorem gcd_problem (b : ℤ) (h : ∃ k : ℤ, b = 2 * k * 953) :
  Int.gcd (3 * b^2 + 17 * b + 23) (b + 19) = 34 := by
  sorry

end NUMINAMATH_CALUDE_gcd_problem_l508_50877


namespace NUMINAMATH_CALUDE_marble_difference_l508_50800

theorem marble_difference (total : ℕ) (yellow : ℕ) (h1 : total = 913) (h2 : yellow = 514) :
  yellow - (total - yellow) = 115 := by
  sorry

end NUMINAMATH_CALUDE_marble_difference_l508_50800


namespace NUMINAMATH_CALUDE_container_volume_ratio_l508_50833

/-- Theorem: Container Volume Ratio
Given two containers where the first is 3/7 full and transfers all its water to the second,
making it 2/3 full, the ratio of the volume of the first container to the volume of the second
container is 14/9.
-/
theorem container_volume_ratio (container1 container2 : ℝ) :
  container1 > 0 ∧ container2 > 0 →  -- Ensure containers have positive volume
  (3 / 7 : ℝ) * container1 = (2 / 3 : ℝ) * container2 → -- Water transfer equation
  container1 / container2 = 14 / 9 := by
sorry


end NUMINAMATH_CALUDE_container_volume_ratio_l508_50833


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_range_l508_50898

theorem quadratic_inequality_solution_range (a : ℝ) : 
  (∃ x : ℝ, 1 < x ∧ x < 4 ∧ x^2 - 3*x - 2 - a > 0) → a < 2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_range_l508_50898


namespace NUMINAMATH_CALUDE_weight_of_b_l508_50883

theorem weight_of_b (a b c d : ℝ) 
  (h1 : (a + b + c + d) / 4 = 40)
  (h2 : (a + b) / 2 = 25)
  (h3 : (b + c) / 2 = 28)
  (h4 : (c + d) / 2 = 32) :
  b = 46 := by
sorry

end NUMINAMATH_CALUDE_weight_of_b_l508_50883


namespace NUMINAMATH_CALUDE_no_clax_is_snapp_l508_50851

-- Define the sets
variable (U : Type) -- Universe set
variable (Clax Ell Snapp Plott : Set U)

-- Define the conditions
variable (h1 : Clax ⊆ Ellᶜ)
variable (h2 : ∃ x, x ∈ Ell ∩ Snapp)
variable (h3 : Snapp ∩ Plott = ∅)

-- State the theorem
theorem no_clax_is_snapp : Clax ∩ Snapp = ∅ := by
  sorry

end NUMINAMATH_CALUDE_no_clax_is_snapp_l508_50851


namespace NUMINAMATH_CALUDE_matrix_sum_proof_l508_50844

theorem matrix_sum_proof : 
  let A : Matrix (Fin 2) (Fin 2) ℤ := !![4, -3; 0, 5]
  let B : Matrix (Fin 2) (Fin 2) ℤ := !![-6, 8; 7, -10]
  A + B = !![-2, 5; 7, -5] := by
  sorry

end NUMINAMATH_CALUDE_matrix_sum_proof_l508_50844


namespace NUMINAMATH_CALUDE_competition_results_l508_50880

def seventh_grade_scores : List ℕ := [3, 6, 7, 6, 6, 8, 6, 9, 6, 10]
def eighth_grade_scores : List ℕ := [5, 6, 8, 7, 5, 8, 7, 9, 8, 8]

def mode (l : List ℕ) : ℕ := sorry

def average (l : List ℕ) : ℚ := sorry

def median (l : List ℕ) : ℚ := sorry

theorem competition_results :
  (mode seventh_grade_scores = 6) ∧
  (average eighth_grade_scores = 71 / 10) ∧
  (7 > median seventh_grade_scores) ∧
  (7 < median eighth_grade_scores) := by sorry

end NUMINAMATH_CALUDE_competition_results_l508_50880


namespace NUMINAMATH_CALUDE_binomial_coefficient_22_5_l508_50808

theorem binomial_coefficient_22_5 (h1 : Nat.choose 20 3 = 1140)
                                  (h2 : Nat.choose 20 4 = 4845)
                                  (h3 : Nat.choose 20 5 = 15504) :
  Nat.choose 22 5 = 26334 := by
  sorry

end NUMINAMATH_CALUDE_binomial_coefficient_22_5_l508_50808


namespace NUMINAMATH_CALUDE_age_problem_l508_50888

theorem age_problem (a b : ℕ) (h1 : 5 * b = 3 * a) (h2 : 7 * (b + 6) = 5 * (a + 6)) : a = 15 := by
  sorry

end NUMINAMATH_CALUDE_age_problem_l508_50888


namespace NUMINAMATH_CALUDE_some_number_value_l508_50825

theorem some_number_value (n m : ℚ) : 
  n = 40 → (n / 20) * (n / m) = 1 → m = 2 := by
  sorry

end NUMINAMATH_CALUDE_some_number_value_l508_50825


namespace NUMINAMATH_CALUDE_price_decrease_percentage_l508_50866

theorem price_decrease_percentage (base_price : ℝ) (regular_price : ℝ) (promotional_price : ℝ) : 
  regular_price = base_price * (1 + 0.25) ∧ 
  promotional_price = base_price →
  (regular_price - promotional_price) / regular_price = 0.20 := by
sorry

end NUMINAMATH_CALUDE_price_decrease_percentage_l508_50866


namespace NUMINAMATH_CALUDE_percentage_problem_l508_50862

theorem percentage_problem (p : ℝ) (h1 : 0.5 * 10 = p / 100 * 500 - 20) : p = 5 := by
  sorry

end NUMINAMATH_CALUDE_percentage_problem_l508_50862


namespace NUMINAMATH_CALUDE_sum_of_cubes_l508_50829

theorem sum_of_cubes (a b : ℝ) (h1 : a + b = 11) (h2 : a * b = 21) : a^3 + b^3 = 638 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_cubes_l508_50829


namespace NUMINAMATH_CALUDE_eccentricity_ratio_for_common_point_l508_50871

/-- The eccentricity of an ellipse -/
def eccentricity_ellipse (a b : ℝ) : ℝ := sorry

/-- The eccentricity of a hyperbola -/
def eccentricity_hyperbola (a b : ℝ) : ℝ := sorry

/-- The distance between two points in ℝ² -/
def distance (p q : ℝ × ℝ) : ℝ := sorry

theorem eccentricity_ratio_for_common_point 
  (F₁ F₂ P : ℝ × ℝ) 
  (e₁ : ℝ) 
  (e₂ : ℝ) 
  (h_ellipse : e₁ = eccentricity_ellipse (distance F₁ P) (distance F₂ P))
  (h_hyperbola : e₂ = eccentricity_hyperbola (distance F₁ P) (distance F₂ P))
  (h_common_point : distance P F₁ + distance P F₂ = distance F₁ F₂) :
  (e₁ * e₂) / Real.sqrt (e₁^2 + e₂^2) = Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_eccentricity_ratio_for_common_point_l508_50871


namespace NUMINAMATH_CALUDE_stating_sweet_apple_percentage_correct_l508_50832

/-- Represents the percentage of sweet apples in Chang's Garden. -/
def sweet_apple_percentage : ℝ := 75

/-- Represents the total number of apples sold. -/
def total_apples : ℕ := 100

/-- Represents the price of a sweet apple in dollars. -/
def sweet_apple_price : ℝ := 0.5

/-- Represents the price of a sour apple in dollars. -/
def sour_apple_price : ℝ := 0.1

/-- Represents the total earnings from selling all apples in dollars. -/
def total_earnings : ℝ := 40

/-- 
Theorem stating that the percentage of sweet apples is correct given the conditions.
-/
theorem sweet_apple_percentage_correct : 
  sweet_apple_price * (sweet_apple_percentage / 100 * total_apples) + 
  sour_apple_price * ((100 - sweet_apple_percentage) / 100 * total_apples) = 
  total_earnings := by sorry

end NUMINAMATH_CALUDE_stating_sweet_apple_percentage_correct_l508_50832


namespace NUMINAMATH_CALUDE_interior_angles_sum_l508_50882

theorem interior_angles_sum (n : ℕ) :
  (180 * (n - 2) = 2340) →
  (180 * ((n + 3) - 2) = 2880) :=
by sorry

end NUMINAMATH_CALUDE_interior_angles_sum_l508_50882


namespace NUMINAMATH_CALUDE_base6_addition_theorem_l508_50841

/-- Represents a number in base 6 as a list of digits (least significant first) -/
def Base6 := List Nat

/-- Converts a base 6 number to its decimal representation -/
def to_decimal (n : Base6) : Nat :=
  n.enum.foldl (fun acc (i, d) => acc + d * (6 ^ i)) 0

/-- Adds two base 6 numbers -/
noncomputable def base6_add (a b : Base6) : Base6 :=
  sorry

theorem base6_addition_theorem :
  let a : Base6 := [2, 3, 5, 4]  -- 4532₆
  let b : Base6 := [2, 1, 4, 3]  -- 3412₆
  let result : Base6 := [4, 1, 4, 0, 1]  -- 10414₆
  base6_add a b = result := by sorry

end NUMINAMATH_CALUDE_base6_addition_theorem_l508_50841


namespace NUMINAMATH_CALUDE_multiply_polynomials_l508_50870

theorem multiply_polynomials (x y : ℝ) :
  (3 * x^4 - 2 * y^3) * (9 * x^8 + 6 * x^4 * y^3 + 4 * y^6) = 27 * x^12 - 8 * y^9 := by
  sorry

end NUMINAMATH_CALUDE_multiply_polynomials_l508_50870


namespace NUMINAMATH_CALUDE_sequence_identity_l508_50878

def is_valid_sequence (a : ℕ → ℕ) : Prop :=
  ∀ n : ℕ, n ≥ 1 → a n ≥ 1 ∧ a (a n) + a n = 2 * n

theorem sequence_identity (a : ℕ → ℕ) (h : is_valid_sequence a) :
  ∀ n : ℕ, n ≥ 1 → a n = n :=
sorry

end NUMINAMATH_CALUDE_sequence_identity_l508_50878


namespace NUMINAMATH_CALUDE_sin_2alpha_value_l508_50884

theorem sin_2alpha_value (a α : ℝ) 
  (h : Real.sin (a + π/4) = Real.sqrt 2 * (Real.sin α + 2 * Real.cos α)) : 
  Real.sin (2 * α) = -3/5 := by
  sorry

end NUMINAMATH_CALUDE_sin_2alpha_value_l508_50884


namespace NUMINAMATH_CALUDE_exist_six_numbers_l508_50827

theorem exist_six_numbers : ∃ (a b c d e f : ℕ), 
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ a ≠ f ∧ 
  b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ b ≠ f ∧ 
  c ≠ d ∧ c ≠ e ∧ c ≠ f ∧ 
  d ≠ e ∧ d ≠ f ∧ 
  e ≠ f ∧
  a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧ e > 0 ∧ f > 0 ∧
  (a + b + c + d + e + f : ℚ) / (1 / a + 1 / b + 1 / c + 1 / d + 1 / e + 1 / f) = 2012 := by
  sorry

end NUMINAMATH_CALUDE_exist_six_numbers_l508_50827


namespace NUMINAMATH_CALUDE_divide_people_eq_280_l508_50860

/-- The number of ways to divide 8 people into three groups -/
def divide_people : ℕ :=
  let total_people : ℕ := 8
  let group_1_size : ℕ := 3
  let group_2_size : ℕ := 3
  let group_3_size : ℕ := 2
  let ways_to_choose_group_1 := Nat.choose total_people group_1_size
  let ways_to_choose_group_2 := Nat.choose (total_people - group_1_size) group_2_size
  let ways_to_choose_group_3 := Nat.choose group_3_size group_3_size
  let arrangements_of_identical_groups : ℕ := 2  -- 2! for two identical groups of 3
  (ways_to_choose_group_1 * ways_to_choose_group_2 * ways_to_choose_group_3) / arrangements_of_identical_groups

theorem divide_people_eq_280 : divide_people = 280 := by
  sorry

end NUMINAMATH_CALUDE_divide_people_eq_280_l508_50860


namespace NUMINAMATH_CALUDE_part_one_part_two_l508_50879

/-- Definition of a midpoint equation -/
def is_midpoint_equation (a b : ℚ) : Prop :=
  a ≠ 0 ∧ (- b / a) = (a + b) / 2

/-- Part 1: Prove that 4x - 8/3 = 0 is a midpoint equation -/
theorem part_one : is_midpoint_equation 4 (-8/3) := by
  sorry

/-- Part 2: Prove that for 5x + m - 1 = 0 to be a midpoint equation, m = -18/7 -/
theorem part_two : ∃ m : ℚ, is_midpoint_equation 5 (m - 1) ↔ m = -18/7 := by
  sorry

end NUMINAMATH_CALUDE_part_one_part_two_l508_50879


namespace NUMINAMATH_CALUDE_smallest_binary_multiple_of_ten_l508_50895

def is_binary_number (n : ℕ) : Prop :=
  ∀ d : ℕ, d ∈ n.digits 10 → d = 0 ∨ d = 1

theorem smallest_binary_multiple_of_ten :
  ∃ X : ℕ, X > 0 ∧ 
  (∃ T : ℕ, T > 0 ∧ is_binary_number T ∧ T = 10 * X) ∧
  (∀ Y : ℕ, Y > 0 → 
    (∃ S : ℕ, S > 0 ∧ is_binary_number S ∧ S = 10 * Y) → 
    X ≤ Y) ∧
  X = 1 :=
sorry

end NUMINAMATH_CALUDE_smallest_binary_multiple_of_ten_l508_50895


namespace NUMINAMATH_CALUDE_cat_monitored_area_percentage_l508_50865

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a rectangular room -/
structure Room where
  width : ℝ
  height : ℝ

/-- Calculates the area of a room -/
def roomArea (r : Room) : ℝ := r.width * r.height

/-- Calculates the area that a cat can monitor in a room -/
noncomputable def monitoredArea (r : Room) (catPosition : Point) : ℝ := sorry

/-- Theorem stating that a cat at (3, 8) in a 10x8 room monitors 66.875% of the area -/
theorem cat_monitored_area_percentage (r : Room) (catPos : Point) :
  r.width = 10 ∧ r.height = 8 ∧ catPos.x = 3 ∧ catPos.y = 8 →
  monitoredArea r catPos / roomArea r = 66.875 / 100 := by sorry

end NUMINAMATH_CALUDE_cat_monitored_area_percentage_l508_50865


namespace NUMINAMATH_CALUDE_roots_equality_l508_50845

theorem roots_equality (a : ℝ) (x₁ x₂ : ℝ) 
  (h1 : x₁ ≠ x₂) 
  (h2 : x₁^2 - x₁ + a = 0) 
  (h3 : x₂^2 - x₂ + a = 0) : 
  |x₁^2 - x₂^2| = 1 ↔ |x₁^3 - x₂^3| = 1 := by
sorry

end NUMINAMATH_CALUDE_roots_equality_l508_50845


namespace NUMINAMATH_CALUDE_final_rope_length_l508_50822

/-- Calculates the final length of a rope made by tying multiple pieces together -/
theorem final_rope_length
  (rope_lengths : List ℝ)
  (knot_loss : ℝ)
  (h_lengths : rope_lengths = [8, 20, 2, 2, 2, 7])
  (h_knot_loss : knot_loss = 1.2)
  : (rope_lengths.sum - knot_loss * (rope_lengths.length - 1 : ℝ)) = 35 := by
  sorry

end NUMINAMATH_CALUDE_final_rope_length_l508_50822


namespace NUMINAMATH_CALUDE_rectangle_covers_curve_l508_50839

/-- A plane curve is a continuous function from a closed interval to ℝ² -/
def PlaneCurve := Set.Icc 0 1 → ℝ × ℝ

/-- The length of a plane curve -/
def curveLength (γ : PlaneCurve) : ℝ := sorry

/-- A rectangle in the plane -/
structure Rectangle where
  bottomLeft : ℝ × ℝ
  topRight : ℝ × ℝ

/-- The area of a rectangle -/
def Rectangle.area (r : Rectangle) : ℝ := sorry

/-- Predicate to check if a rectangle covers a curve -/
def covers (r : Rectangle) (γ : PlaneCurve) : Prop := sorry

/-- Main theorem: For any plane curve of length 1, there exists a rectangle of area 1/4 that covers it -/
theorem rectangle_covers_curve (γ : PlaneCurve) (h : curveLength γ = 1) :
  ∃ r : Rectangle, r.area = 1/4 ∧ covers r γ := by sorry

end NUMINAMATH_CALUDE_rectangle_covers_curve_l508_50839


namespace NUMINAMATH_CALUDE_double_cone_is_cone_l508_50894

/-- Represents a point in cylindrical coordinates -/
structure CylindricalPoint where
  r : ℝ
  θ : ℝ
  z : ℝ

/-- Defines the set of points satisfying the given equations -/
def DoubleConeSurface (c : ℝ) : Set CylindricalPoint :=
  {p : CylindricalPoint | p.θ = c ∧ p.r = |p.z|}

/-- Defines a cone surface in cylindrical coordinates -/
def ConeSurface : Set CylindricalPoint :=
  {p : CylindricalPoint | ∃ (k : ℝ), p.r = k * |p.z|}

/-- Theorem stating that the surface defined by the equations is a cone -/
theorem double_cone_is_cone (c : ℝ) :
  ∃ (k : ℝ), DoubleConeSurface c ⊆ ConeSurface :=
sorry

end NUMINAMATH_CALUDE_double_cone_is_cone_l508_50894


namespace NUMINAMATH_CALUDE_expand_expression_l508_50848

theorem expand_expression (x : ℝ) : (7*x + 11) * (3*x^2 + 2*x) = 21*x^3 + 47*x^2 + 22*x := by
  sorry

end NUMINAMATH_CALUDE_expand_expression_l508_50848


namespace NUMINAMATH_CALUDE_inequality_proof_equality_condition_theorem_l508_50842

-- Define the theorem
theorem inequality_proof (a b c : ℝ) :
  Real.sqrt (a^2 + b^2 - a*b) + Real.sqrt (b^2 + c^2 - b*c) ≥ Real.sqrt (a^2 + c^2 + a*c) := by
  sorry

-- Define the equality condition
def equality_condition (a b c : ℝ) : Prop :=
  (a * c = a * b + b * c) ∧ (a * b + a * c + b * c - 2 * b^2 ≥ 0)

-- Theorem for the equality condition
theorem equality_condition_theorem (a b c : ℝ) :
  (Real.sqrt (a^2 + b^2 - a*b) + Real.sqrt (b^2 + c^2 - b*c) = Real.sqrt (a^2 + c^2 + a*c)) ↔
  equality_condition a b c := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_equality_condition_theorem_l508_50842


namespace NUMINAMATH_CALUDE_chess_tournament_proof_l508_50855

theorem chess_tournament_proof (i g n : ℕ) (I G : ℚ) :
  g = 10 * i →
  n = i + g →
  G = (9/2) * I →
  (n * (n - 1)) / 2 = I + G →
  i = 1 ∧ g = 10 ∧ (n * (n - 1)) / 2 = 55 :=
by sorry

end NUMINAMATH_CALUDE_chess_tournament_proof_l508_50855


namespace NUMINAMATH_CALUDE_least_4_light_four_digit_l508_50864

def is_4_light (n : ℕ) : Prop := n % 9 < 4

def is_four_digit (n : ℕ) : Prop := 1000 ≤ n ∧ n ≤ 9999

theorem least_4_light_four_digit : 
  (∀ n : ℕ, is_four_digit n → is_4_light n → 1000 ≤ n) ∧ is_four_digit 1000 ∧ is_4_light 1000 :=
sorry

end NUMINAMATH_CALUDE_least_4_light_four_digit_l508_50864


namespace NUMINAMATH_CALUDE_arithmetic_sequence_properties_l508_50861

def arithmetic_sequence (a : ℕ → ℤ) := ∀ n, a (n + 1) - a n = a 1 - a 0

def sum_arithmetic_sequence (a : ℕ → ℤ) (n : ℕ) : ℤ := 
  n * (a 0 + a (n - 1)) / 2

theorem arithmetic_sequence_properties 
  (a : ℕ → ℤ) 
  (h_arithmetic : arithmetic_sequence a)
  (h_sum3 : sum_arithmetic_sequence a 3 = 42)
  (h_sum6 : sum_arithmetic_sequence a 6 = 57) :
  (∀ n, a n = 20 - 3 * n) ∧ 
  (∀ n, n ≤ 6 → sum_arithmetic_sequence a n ≤ sum_arithmetic_sequence a 6) :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_properties_l508_50861


namespace NUMINAMATH_CALUDE_diamond_equation_solution_l508_50806

/-- Define the diamond operation -/
def diamond (A B : ℝ) : ℝ := 5 * A + 3 * B + 7

/-- Theorem: If A ◇ 5 = 82, then A = 12 -/
theorem diamond_equation_solution :
  ∀ A : ℝ, diamond A 5 = 82 → A = 12 := by
  sorry

end NUMINAMATH_CALUDE_diamond_equation_solution_l508_50806


namespace NUMINAMATH_CALUDE_work_completion_time_l508_50846

theorem work_completion_time (a b : ℝ) (h1 : a = 2 * b) (h2 : 1 / (a + b) = 1 / 20) :
  1 / a = 1 / 30 := by
  sorry

end NUMINAMATH_CALUDE_work_completion_time_l508_50846


namespace NUMINAMATH_CALUDE_linear_function_quadrants_l508_50807

/-- A proportional function with a negative slope -/
structure NegativeSlopeProportionalFunction where
  k : ℝ
  k_nonzero : k ≠ 0
  k_negative : k < 0

/-- The linear function y = 2x + k -/
def linear_function (f : NegativeSlopeProportionalFunction) (x : ℝ) : ℝ := 2 * x + f.k

/-- Quadrants of the Cartesian plane -/
inductive Quadrant
  | I
  | II
  | III
  | IV

/-- Check if a point (x, y) is in a given quadrant -/
def in_quadrant (x y : ℝ) (q : Quadrant) : Prop :=
  match q with
  | Quadrant.I  => x > 0 ∧ y > 0
  | Quadrant.II => x < 0 ∧ y > 0
  | Quadrant.III => x < 0 ∧ y < 0
  | Quadrant.IV => x > 0 ∧ y < 0

/-- The theorem stating that the linear function passes through Quadrants I, III, and IV -/
theorem linear_function_quadrants (f : NegativeSlopeProportionalFunction) :
  (∃ x y : ℝ, y = linear_function f x ∧ in_quadrant x y Quadrant.I) ∧
  (∃ x y : ℝ, y = linear_function f x ∧ in_quadrant x y Quadrant.III) ∧
  (∃ x y : ℝ, y = linear_function f x ∧ in_quadrant x y Quadrant.IV) :=
sorry

end NUMINAMATH_CALUDE_linear_function_quadrants_l508_50807


namespace NUMINAMATH_CALUDE_max_fourth_power_sum_l508_50802

theorem max_fourth_power_sum (a b c d : ℝ) (h : a^3 + b^3 + c^3 + d^3 = 4) :
  ∃ (m : ℝ), m = 64 / (4^(1/3)) ∧ a^4 + b^4 + c^4 + d^4 ≤ m :=
by sorry

end NUMINAMATH_CALUDE_max_fourth_power_sum_l508_50802


namespace NUMINAMATH_CALUDE_expression_simplification_l508_50849

theorem expression_simplification (a b : ℚ) (h1 : a = 1) (h2 : b = -2) :
  ((a - 2*b)^2 - (a - 2*b)*(a + 2*b) + 4*b^2) / (-2*b) = 14 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l508_50849


namespace NUMINAMATH_CALUDE_red_lettuce_cost_l508_50801

/-- The amount spent on red lettuce given the total cost and cost of green lettuce -/
def amount_spent_on_red_lettuce (total_cost green_cost : ℕ) : ℕ :=
  total_cost - green_cost

/-- Proof that the amount spent on red lettuce is $6 -/
theorem red_lettuce_cost : amount_spent_on_red_lettuce 14 8 = 6 := by
  sorry

end NUMINAMATH_CALUDE_red_lettuce_cost_l508_50801


namespace NUMINAMATH_CALUDE_square_rearrangement_theorem_l508_50873

-- Define a type for square sheets of paper
def Square : Type := Unit

-- Define a function that represents the possibility of cutting and rearranging squares
def can_cut_and_rearrange (n : ℕ) : Prop :=
  ∀ (squares : Fin n → Square), ∃ (new_square : Square), True

-- State the theorem
theorem square_rearrangement_theorem (n : ℕ) (h : n > 1) :
  can_cut_and_rearrange n :=
sorry

end NUMINAMATH_CALUDE_square_rearrangement_theorem_l508_50873


namespace NUMINAMATH_CALUDE_milk_replacement_percentage_l508_50885

theorem milk_replacement_percentage (x : ℝ) : 
  (((100 - x) / 100) * ((100 - x) / 100) * ((100 - x) / 100)) * 100 = 51.20000000000001 → 
  x = 20 := by
sorry

end NUMINAMATH_CALUDE_milk_replacement_percentage_l508_50885


namespace NUMINAMATH_CALUDE_discount_percentage_l508_50886

theorem discount_percentage (srp mp paid : ℝ) : 
  srp = 1.2 * mp →  -- SRP is 20% higher than MP
  paid = 0.6 * mp →  -- John paid 60% of MP (40% off)
  paid / srp = 0.5 :=  -- John paid 50% of SRP
by sorry

end NUMINAMATH_CALUDE_discount_percentage_l508_50886


namespace NUMINAMATH_CALUDE_julies_landscaping_hours_l508_50826

/-- Julie's landscaping business problem -/
theorem julies_landscaping_hours (mowing_rate pulling_rate pulling_hours total_earnings : ℕ) :
  mowing_rate = 4 →
  pulling_rate = 8 →
  pulling_hours = 3 →
  total_earnings = 248 →
  ∃ (mowing_hours : ℕ),
    2 * (mowing_rate * mowing_hours + pulling_rate * pulling_hours) = total_earnings ∧
    mowing_hours = 25 :=
by sorry

end NUMINAMATH_CALUDE_julies_landscaping_hours_l508_50826


namespace NUMINAMATH_CALUDE_inverse_on_negative_T_to_0_l508_50852

-- Define a periodic function f with period T
def isPeriodic (f : ℝ → ℝ) (T : ℝ) : Prop :=
  ∀ x, f (x + T) = f x

-- Define the smallest positive period
def isSmallestPositivePeriod (f : ℝ → ℝ) (T : ℝ) : Prop :=
  T > 0 ∧ isPeriodic f T ∧ ∀ S, 0 < S ∧ S < T → ¬isPeriodic f S

-- Define the inverse function on (0, T)
def inverseOn0T (f : ℝ → ℝ) (T : ℝ) (D : Set ℝ) (fInv : ℝ → ℝ) : Prop :=
  ∀ x ∈ D, 0 < x ∧ x < T → f (fInv x) = x ∧ fInv (f x) = x

-- Main theorem
theorem inverse_on_negative_T_to_0
  (f : ℝ → ℝ) (T : ℝ) (D : Set ℝ) (fInv : ℝ → ℝ)
  (h_periodic : isPeriodic f T)
  (h_smallest : isSmallestPositivePeriod f T)
  (h_inverse : inverseOn0T f T D fInv) :
  ∀ x ∈ D, -T < x ∧ x < 0 → f (fInv x - T) = x ∧ fInv x - T = f⁻¹ x :=
sorry

end NUMINAMATH_CALUDE_inverse_on_negative_T_to_0_l508_50852


namespace NUMINAMATH_CALUDE_right_triangle_hypotenuse_l508_50816

theorem right_triangle_hypotenuse (a b c : ℝ) : 
  a = 15 → b = 36 → c^2 = a^2 + b^2 → c = 39 :=
by sorry

end NUMINAMATH_CALUDE_right_triangle_hypotenuse_l508_50816


namespace NUMINAMATH_CALUDE_sally_cost_theorem_l508_50830

def lightning_cost : ℝ := 140000

def mater_cost : ℝ := 0.1 * lightning_cost

def sally_cost : ℝ := 3 * mater_cost

theorem sally_cost_theorem : sally_cost = 42000 := by
  sorry

end NUMINAMATH_CALUDE_sally_cost_theorem_l508_50830


namespace NUMINAMATH_CALUDE_max_remainder_eleven_l508_50814

theorem max_remainder_eleven (y : ℕ+) : ∃ (q r : ℕ), y = 11 * q + r ∧ r < 11 ∧ r ≤ 10 :=
sorry

end NUMINAMATH_CALUDE_max_remainder_eleven_l508_50814


namespace NUMINAMATH_CALUDE_simplify_sqrt_difference_l508_50889

theorem simplify_sqrt_difference : 
  (Real.sqrt 648 / Real.sqrt 81) - (Real.sqrt 294 / Real.sqrt 49) = 2 * Real.sqrt 2 - Real.sqrt 42 := by
  sorry

end NUMINAMATH_CALUDE_simplify_sqrt_difference_l508_50889


namespace NUMINAMATH_CALUDE_problem_solution_l508_50824

/-- Calculates the total earnings given investment ratios, percentage return ratios, and the difference between B's and A's earnings -/
def totalEarnings (investmentRatio : Fin 3 → ℕ) (returnRatio : Fin 3 → ℕ) (bMinusAEarnings : ℕ) : ℕ :=
  let earnings := λ i => investmentRatio i * returnRatio i
  let totalEarnings := (earnings 0) + (earnings 1) + (earnings 2)
  totalEarnings * (bMinusAEarnings / ((investmentRatio 1 * returnRatio 1) - (investmentRatio 0 * returnRatio 0)))

/-- The total earnings for the given problem -/
theorem problem_solution :
  totalEarnings
    (λ i => [3, 4, 5].get i)
    (λ i => [6, 5, 4].get i)
    250 = 7250 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l508_50824


namespace NUMINAMATH_CALUDE_boisjoli_farm_egg_production_l508_50836

/-- Represents the chicken coop at Boisjoli farm -/
structure ChickenCoop where
  total_hens : ℕ
  total_roosters : ℕ
  laying_percentage : ℚ
  morning_laying_percentage : ℚ
  afternoon_laying_percentage : ℚ
  unusable_egg_percentage : ℚ
  eggs_per_box : ℕ
  days_per_week : ℕ

/-- Calculates the number of boxes of usable eggs filled in a week -/
def boxes_filled_per_week (coop : ChickenCoop) : ℕ :=
  sorry

/-- The main theorem stating the number of boxes filled per week -/
theorem boisjoli_farm_egg_production :
  let coop : ChickenCoop := {
    total_hens := 270,
    total_roosters := 3,
    laying_percentage := 9/10,
    morning_laying_percentage := 4/10,
    afternoon_laying_percentage := 5/10,
    unusable_egg_percentage := 1/20,
    eggs_per_box := 7,
    days_per_week := 7
  }
  boxes_filled_per_week coop = 203 := by
  sorry

end NUMINAMATH_CALUDE_boisjoli_farm_egg_production_l508_50836


namespace NUMINAMATH_CALUDE_equation_solution_l508_50872

theorem equation_solution (a b x : ℝ) :
  (a ≠ b ∧ a ≠ -b ∧ b ≠ 0 → x = a^2 - b^2 → 
    1 / (a + b) + (a - b) / x = 1 / (a - b) + (a - b) / x) ∧
  (b = 0 ∧ a ≠ 0 ∧ x ≠ 0 → 
    1 / (a + b) + (a - b) / x = 1 / (a - b) + (a - b) / x) :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l508_50872


namespace NUMINAMATH_CALUDE_smallest_four_digit_with_digit_product_12_l508_50891

def is_four_digit (n : ℕ) : Prop := 1000 ≤ n ∧ n ≤ 9999

def digit_product (n : ℕ) : ℕ :=
  (n / 1000) * ((n / 100) % 10) * ((n / 10) % 10) * (n % 10)

theorem smallest_four_digit_with_digit_product_12 :
  ∀ n : ℕ, is_four_digit n → digit_product n = 12 → 1126 ≤ n :=
by sorry

end NUMINAMATH_CALUDE_smallest_four_digit_with_digit_product_12_l508_50891


namespace NUMINAMATH_CALUDE_reggie_brother_long_shots_l508_50893

-- Define the point values for each shot type
def layup_points : ℕ := 1
def free_throw_points : ℕ := 2
def long_shot_points : ℕ := 3

-- Define Reggie's shots
def reggie_layups : ℕ := 3
def reggie_free_throws : ℕ := 2
def reggie_long_shots : ℕ := 1

-- Define the point difference
def point_difference : ℕ := 2

-- Theorem to prove
theorem reggie_brother_long_shots :
  let reggie_points := reggie_layups * layup_points + reggie_free_throws * free_throw_points + reggie_long_shots * long_shot_points
  let brother_points := reggie_points + point_difference
  brother_points / long_shot_points = 4 :=
by sorry

end NUMINAMATH_CALUDE_reggie_brother_long_shots_l508_50893


namespace NUMINAMATH_CALUDE_sum_of_first_n_naturals_l508_50837

theorem sum_of_first_n_naturals (n : ℕ) : 
  (n * (n + 1)) / 2 = 3675 ↔ n = 81 := by sorry

end NUMINAMATH_CALUDE_sum_of_first_n_naturals_l508_50837


namespace NUMINAMATH_CALUDE_roots_are_irrational_l508_50831

/-- Given a real number k, this function represents the quadratic equation x^2 - 3kx + 2k^2 - 1 = 0 --/
def quadratic_equation (k : ℝ) (x : ℝ) : Prop :=
  x^2 - 3*k*x + 2*k^2 - 1 = 0

/-- The product of the roots of the quadratic equation is 7 --/
axiom root_product (k : ℝ) : 
  ∃ x₁ x₂ : ℝ, quadratic_equation k x₁ ∧ quadratic_equation k x₂ ∧ x₁ * x₂ = 7

/-- Definition of an irrational number --/
def is_irrational (x : ℝ) : Prop :=
  ∀ p q : ℤ, q ≠ 0 → x ≠ p / q

/-- The main theorem: the roots of the quadratic equation are irrational --/
theorem roots_are_irrational (k : ℝ) :
  ∃ x₁ x₂ : ℝ, quadratic_equation k x₁ ∧ quadratic_equation k x₂ ∧ 
             is_irrational x₁ ∧ is_irrational x₂ :=
by sorry

end NUMINAMATH_CALUDE_roots_are_irrational_l508_50831


namespace NUMINAMATH_CALUDE_jane_mean_score_l508_50810

def jane_scores : List ℝ := [96, 95, 90, 87, 91, 75]

theorem jane_mean_score :
  (jane_scores.sum / jane_scores.length : ℝ) = 89 := by
  sorry

end NUMINAMATH_CALUDE_jane_mean_score_l508_50810


namespace NUMINAMATH_CALUDE_star_two_three_l508_50897

-- Define the star operation
def star (a b : ℝ) : ℝ := a * b^3 - b + 2

-- Theorem statement
theorem star_two_three : star 2 3 = 53 := by
  sorry

end NUMINAMATH_CALUDE_star_two_three_l508_50897


namespace NUMINAMATH_CALUDE_collinearity_necessary_not_sufficient_l508_50817

/-- Two vectors are collinear if one is a scalar multiple of the other -/
def collinear (v w : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, v.1 = k * w.1 ∧ v.2 = k * w.2

/-- The statement to be proved -/
theorem collinearity_necessary_not_sufficient :
  ¬(∀ a : ℝ, collinear (a, a^2) (1, 2) → a = 2) ∧
  (a = 2 → collinear (a, a^2) (1, 2)) :=
by sorry

end NUMINAMATH_CALUDE_collinearity_necessary_not_sufficient_l508_50817


namespace NUMINAMATH_CALUDE_A_star_B_equals_zero_three_l508_50850

def A : Set ℕ := {0, 1, 2}
def B : Set ℕ := {1, 2, 3}

def star (X Y : Set ℕ) : Set ℕ :=
  {x | x ∈ X ∨ x ∈ Y ∧ x ∉ X ∩ Y}

theorem A_star_B_equals_zero_three : star A B = {0, 3} := by
  sorry

end NUMINAMATH_CALUDE_A_star_B_equals_zero_three_l508_50850


namespace NUMINAMATH_CALUDE_specific_arc_rectangle_boundary_l508_50887

/-- Represents a rectangle with quarter-circle arcs on its corners -/
structure ArcRectangle where
  area : ℝ
  length_width_ratio : ℝ
  divisions : ℕ

/-- Calculates the boundary length of the ArcRectangle -/
def boundary_length (r : ArcRectangle) : ℝ :=
  sorry

/-- Theorem stating the boundary length of a specific ArcRectangle -/
theorem specific_arc_rectangle_boundary :
  let r : ArcRectangle := { area := 72, length_width_ratio := 2, divisions := 3 }
  boundary_length r = 4 * Real.pi + 24 := by
  sorry

end NUMINAMATH_CALUDE_specific_arc_rectangle_boundary_l508_50887


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_sets_quadratic_inequality_parameter_range_l508_50813

-- Problem 1
theorem quadratic_inequality_solution_sets (a c : ℝ) :
  (∀ x : ℝ, ax^2 + 2*x + c > 0 ↔ -1/3 < x ∧ x < 1/2) →
  (∀ x : ℝ, c*x^2 - 2*x + a < 0 ↔ -2 < x ∧ x < 3) :=
sorry

-- Problem 2
theorem quadratic_inequality_parameter_range (m : ℝ) :
  (∀ x : ℝ, x > 0 → x^2 - m*x + 4 > 0) ↔ m < 4 :=
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_sets_quadratic_inequality_parameter_range_l508_50813


namespace NUMINAMATH_CALUDE_solution_implies_a_value_l508_50869

theorem solution_implies_a_value (x y a : ℝ) : 
  x = -2 → y = 1 → 2 * x + a * y = 3 → a = 7 := by sorry

end NUMINAMATH_CALUDE_solution_implies_a_value_l508_50869


namespace NUMINAMATH_CALUDE_math_competition_score_l508_50840

theorem math_competition_score (total_questions n_correct n_wrong n_unanswered : ℕ) 
  (new_score old_score : ℕ) :
  total_questions = 50 ∧ 
  new_score = 150 ∧ 
  old_score = 118 ∧ 
  new_score = 6 * n_correct + 3 * n_unanswered ∧ 
  old_score = 40 + 5 * n_correct - 2 * n_wrong ∧ 
  total_questions = n_correct + n_wrong + n_unanswered →
  n_unanswered = 16 := by
sorry

end NUMINAMATH_CALUDE_math_competition_score_l508_50840


namespace NUMINAMATH_CALUDE_complex_exponent_calculation_l508_50847

theorem complex_exponent_calculation : 
  ((-8 : ℂ) ^ (2/3 : ℂ)) * ((1 / Real.sqrt 2) ^ (-2 : ℂ)) * ((27 : ℂ) ^ (-1/3 : ℂ)) = 8/3 := by
  sorry

end NUMINAMATH_CALUDE_complex_exponent_calculation_l508_50847


namespace NUMINAMATH_CALUDE_not_both_prime_2n_plus_minus_one_l508_50820

theorem not_both_prime_2n_plus_minus_one (n : ℕ) (h : n > 2) :
  ¬(Nat.Prime (2^n - 1) ∧ Nat.Prime (2^n + 1)) :=
by sorry

end NUMINAMATH_CALUDE_not_both_prime_2n_plus_minus_one_l508_50820


namespace NUMINAMATH_CALUDE_sum_of_A_and_C_is_six_l508_50896

theorem sum_of_A_and_C_is_six (A B C D : ℕ) : 
  A ∈ ({1, 2, 3, 4, 5} : Set ℕ) →
  B ∈ ({1, 2, 3, 4, 5} : Set ℕ) →
  C ∈ ({1, 2, 3, 4, 5} : Set ℕ) →
  D ∈ ({1, 2, 3, 4, 5} : Set ℕ) →
  A ≠ B → A ≠ C → A ≠ D → B ≠ C → B ≠ D → C ≠ D →
  (A : ℚ) / B + (C : ℚ) / D = 2 →
  A + C = 6 := by
sorry

end NUMINAMATH_CALUDE_sum_of_A_and_C_is_six_l508_50896


namespace NUMINAMATH_CALUDE_rational_inequality_solution_l508_50876

theorem rational_inequality_solution (x : ℝ) : 
  x ≠ -5 → ((x - 2) / (x + 5) ≥ 0 ↔ x < -5 ∨ x ≥ 2) :=
by sorry

end NUMINAMATH_CALUDE_rational_inequality_solution_l508_50876


namespace NUMINAMATH_CALUDE_christopher_age_l508_50856

theorem christopher_age (c g : ℕ) : 
  c = 2 * g →                  -- Christopher is 2 times as old as Gabriela now
  c - 9 = 5 * (g - 9) →        -- Nine years ago, Christopher was 5 times as old as Gabriela
  c = 24                       -- Christopher's current age is 24
  := by sorry

end NUMINAMATH_CALUDE_christopher_age_l508_50856


namespace NUMINAMATH_CALUDE_smallest_non_special_number_twenty_two_is_non_special_l508_50863

def triangle_number (k : ℕ) : ℕ := k * (k + 1) / 2

def is_prime_power (n : ℕ) : Prop :=
  ∃ (p k : ℕ), p.Prime ∧ k > 0 ∧ n = p ^ k

def is_prime_plus_one (n : ℕ) : Prop :=
  ∃ p : ℕ, p.Prime ∧ n = p + 1

theorem smallest_non_special_number :
  ∀ n : ℕ, n < 22 →
    (∃ k : ℕ, n = triangle_number k) ∨
    is_prime_power n ∨
    is_prime_plus_one n :=
  sorry

theorem twenty_two_is_non_special :
  ¬(∃ k : ℕ, 22 = triangle_number k) ∧
  ¬is_prime_power 22 ∧
  ¬is_prime_plus_one 22 :=
  sorry

end NUMINAMATH_CALUDE_smallest_non_special_number_twenty_two_is_non_special_l508_50863


namespace NUMINAMATH_CALUDE_sequence_formula_l508_50812

theorem sequence_formula (a : ℕ → ℕ) :
  a 1 = 1 ∧
  (∀ n : ℕ, n ≥ 1 → a (n + 1) = a n + 2 * n) →
  ∀ n : ℕ, n ≥ 1 → a n = n^2 - n + 1 :=
by
  sorry

end NUMINAMATH_CALUDE_sequence_formula_l508_50812


namespace NUMINAMATH_CALUDE_tan_difference_l508_50828

theorem tan_difference (α β : Real) 
  (h1 : Real.tan (α + π/3) = -3)
  (h2 : Real.tan (β - π/6) = 5) : 
  Real.tan (α - β) = -7/4 := by
sorry

end NUMINAMATH_CALUDE_tan_difference_l508_50828


namespace NUMINAMATH_CALUDE_solve_system_l508_50835

theorem solve_system (x y : ℚ) (eq1 : 3 * x - 2 * y = 7) (eq2 : x + 3 * y = 8) : x = 37 / 11 := by
  sorry

end NUMINAMATH_CALUDE_solve_system_l508_50835


namespace NUMINAMATH_CALUDE_equation_solution_l508_50821

theorem equation_solution : 
  ∃! x : ℝ, (2 + x ≠ 0 ∧ 3 * x - 1 ≠ 0) ∧ (1 / (2 + x) = 2 / (3 * x - 1)) ∧ x = 5 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l508_50821


namespace NUMINAMATH_CALUDE_altitude_length_l508_50838

/-- An isosceles triangle with given side lengths and altitude --/
structure IsoscelesTriangle where
  -- Side lengths
  ab : ℝ
  bc : ℝ
  -- Isosceles condition
  isIsosceles : ab = ac
  -- Altitude
  ad : ℝ
  -- Altitude meets base at midpoint
  isMidpoint : bd = bc / 2

/-- The theorem stating the length of the altitude in the given isosceles triangle --/
theorem altitude_length (t : IsoscelesTriangle) 
  (h1 : t.ab = 10) 
  (h2 : t.bc = 16) : 
  t.ad = 6 := by
  sorry

#check altitude_length

end NUMINAMATH_CALUDE_altitude_length_l508_50838


namespace NUMINAMATH_CALUDE_complement_of_N_in_M_l508_50809

def M : Set ℕ := {1, 2, 3, 4, 5}
def N : Set ℕ := {2, 5}

theorem complement_of_N_in_M :
  M \ N = {1, 3, 4} := by sorry

end NUMINAMATH_CALUDE_complement_of_N_in_M_l508_50809


namespace NUMINAMATH_CALUDE_karls_total_distance_l508_50818

/-- Represents the problem of calculating Karl's total driving distance --/
def karls_drive (miles_per_gallon : ℚ) (tank_capacity : ℚ) (initial_distance : ℚ) 
  (refuel_amount : ℚ) (final_tank_fraction : ℚ) : Prop :=
  let initial_fuel_used : ℚ := initial_distance / miles_per_gallon
  let remaining_fuel : ℚ := refuel_amount - (tank_capacity * final_tank_fraction)
  let additional_distance : ℚ := remaining_fuel * miles_per_gallon
  let total_distance : ℚ := initial_distance + additional_distance
  total_distance = 517

/-- Theorem stating that Karl drove 517 miles given the problem conditions --/
theorem karls_total_distance : 
  karls_drive 25 16 400 10 (1/3) :=
by
  sorry

end NUMINAMATH_CALUDE_karls_total_distance_l508_50818


namespace NUMINAMATH_CALUDE_cube_third_times_eighth_equals_one_over_216_l508_50858

theorem cube_third_times_eighth_equals_one_over_216 :
  (1 / 3 : ℚ)^3 * (1 / 8 : ℚ) = 1 / 216 := by sorry

end NUMINAMATH_CALUDE_cube_third_times_eighth_equals_one_over_216_l508_50858


namespace NUMINAMATH_CALUDE_complex_magnitude_problem_l508_50819

theorem complex_magnitude_problem (z₁ z₂ : ℂ) : 
  z₁ = -1 + I → z₁ * z₂ = -2 → Complex.abs (z₂ + 2*I) = Real.sqrt 10 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_problem_l508_50819


namespace NUMINAMATH_CALUDE_kahi_memorized_words_l508_50899

theorem kahi_memorized_words (total : ℕ) (yesterday_fraction : ℚ) (today_fraction : ℚ)
  (h_total : total = 810)
  (h_yesterday : yesterday_fraction = 1 / 9)
  (h_today : today_fraction = 1 / 4) :
  (total - yesterday_fraction * total) * today_fraction = 180 := by
  sorry

end NUMINAMATH_CALUDE_kahi_memorized_words_l508_50899
