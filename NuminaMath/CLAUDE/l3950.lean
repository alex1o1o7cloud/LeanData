import Mathlib

namespace NUMINAMATH_CALUDE_final_load_is_30600_l3950_395080

def initial_load : ℝ := 50000

def first_unload_percent : ℝ := 0.1
def second_unload_percent : ℝ := 0.2
def third_unload_percent : ℝ := 0.15

def remaining_after_first (load : ℝ) : ℝ :=
  load * (1 - first_unload_percent)

def remaining_after_second (load : ℝ) : ℝ :=
  load * (1 - second_unload_percent)

def remaining_after_third (load : ℝ) : ℝ :=
  load * (1 - third_unload_percent)

theorem final_load_is_30600 :
  remaining_after_third (remaining_after_second (remaining_after_first initial_load)) = 30600 := by
  sorry

end NUMINAMATH_CALUDE_final_load_is_30600_l3950_395080


namespace NUMINAMATH_CALUDE_brian_shirts_l3950_395085

theorem brian_shirts (steven_shirts andrew_shirts brian_shirts : ℕ) 
  (h1 : steven_shirts = 4 * andrew_shirts)
  (h2 : andrew_shirts = 6 * brian_shirts)
  (h3 : steven_shirts = 72) : 
  brian_shirts = 3 := by
sorry

end NUMINAMATH_CALUDE_brian_shirts_l3950_395085


namespace NUMINAMATH_CALUDE_corrected_mean_is_36_4_l3950_395001

/-- Calculates the corrected mean of a set of observations after fixing an error --/
def corrected_mean (n : ℕ) (original_mean : ℚ) (wrong_value : ℚ) (correct_value : ℚ) : ℚ :=
  let original_sum := n * original_mean
  let difference := correct_value - wrong_value
  let corrected_sum := original_sum + difference
  corrected_sum / n

/-- Proves that the corrected mean is 36.4 given the specified conditions --/
theorem corrected_mean_is_36_4 :
  corrected_mean 50 36 23 43 = 36.4 := by
sorry

end NUMINAMATH_CALUDE_corrected_mean_is_36_4_l3950_395001


namespace NUMINAMATH_CALUDE_at_least_one_not_less_than_two_l3950_395063

theorem at_least_one_not_less_than_two (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a + 1/b ≥ 2) ∨ (b + 1/c ≥ 2) ∨ (c + 1/a ≥ 2) := by
  sorry

end NUMINAMATH_CALUDE_at_least_one_not_less_than_two_l3950_395063


namespace NUMINAMATH_CALUDE_log_equality_l3950_395078

theorem log_equality (a b : ℝ) (ha : a = Real.log 625 / Real.log 16) (hb : b = Real.log 25 / Real.log 4) :
  a = b := by
  sorry

end NUMINAMATH_CALUDE_log_equality_l3950_395078


namespace NUMINAMATH_CALUDE_triangle_area_l3950_395005

theorem triangle_area (A B C : ℝ × ℝ) : 
  let AB := Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)
  let BC := Real.sqrt ((B.1 - C.1)^2 + (B.2 - C.2)^2)
  let AC := Real.sqrt ((A.1 - C.1)^2 + (A.2 - C.2)^2)
  let angle_C := Real.arccos ((AB^2 + BC^2 - AC^2) / (2 * AB * BC))
  AB = 2 * Real.sqrt 3 ∧ BC = 2 ∧ angle_C = π / 3 →
  (1 / 2) * AB * BC * Real.sin angle_C = 2 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_l3950_395005


namespace NUMINAMATH_CALUDE_trays_from_second_table_l3950_395057

def trays_per_trip : ℕ := 7
def total_trips : ℕ := 4
def trays_from_first_table : ℕ := 23

theorem trays_from_second_table :
  trays_per_trip * total_trips - trays_from_first_table = 5 := by
  sorry

end NUMINAMATH_CALUDE_trays_from_second_table_l3950_395057


namespace NUMINAMATH_CALUDE_f_continuous_iff_a_eq_5_l3950_395040

-- Define the piecewise function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x > 3 then x^2 + 2 else 2*x + a

-- State the theorem
theorem f_continuous_iff_a_eq_5 (a : ℝ) :
  Continuous (f a) ↔ a = 5 := by
  sorry

end NUMINAMATH_CALUDE_f_continuous_iff_a_eq_5_l3950_395040


namespace NUMINAMATH_CALUDE_teacup_box_ratio_l3950_395027

theorem teacup_box_ratio : 
  let total_boxes : ℕ := 26
  let pan_boxes : ℕ := 6
  let cups_per_box : ℕ := 5 * 4
  let broken_cups_per_box : ℕ := 2
  let remaining_cups : ℕ := 180
  
  let non_pan_boxes : ℕ := total_boxes - pan_boxes
  let teacup_boxes : ℕ := remaining_cups / (cups_per_box - broken_cups_per_box)
  let decoration_boxes : ℕ := non_pan_boxes - teacup_boxes
  
  (decoration_boxes : ℚ) / total_boxes = 5 / 13 :=
by sorry

end NUMINAMATH_CALUDE_teacup_box_ratio_l3950_395027


namespace NUMINAMATH_CALUDE_teena_speed_is_55_l3950_395013

-- Define the given conditions
def yoe_speed : ℝ := 40
def initial_distance : ℝ := 7.5
def final_relative_distance : ℝ := 15
def time : ℝ := 1.5  -- 90 minutes in hours

-- Define Teena's speed as a variable
def teena_speed : ℝ := 55

-- Theorem statement
theorem teena_speed_is_55 :
  yoe_speed * time + initial_distance + final_relative_distance = teena_speed * time :=
by sorry

end NUMINAMATH_CALUDE_teena_speed_is_55_l3950_395013


namespace NUMINAMATH_CALUDE_triangular_array_sum_of_digits_l3950_395094

def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + sum_of_digits (n / 10)

theorem triangular_array_sum_of_digits :
  ∀ N : ℕ, N > 0 → N * (N + 1) / 2 = 3003 → sum_of_digits N = 14 :=
by
  sorry

end NUMINAMATH_CALUDE_triangular_array_sum_of_digits_l3950_395094


namespace NUMINAMATH_CALUDE_roads_per_neighborhood_is_four_l3950_395028

/-- The number of roads passing through each neighborhood in a town with the following properties:
  * The town has 10 neighborhoods.
  * Each road has 250 street lights on each opposite side.
  * The total number of street lights in the town is 20000.
-/
def roads_per_neighborhood : ℕ := 4

/-- Theorem stating that the number of roads passing through each neighborhood is 4 -/
theorem roads_per_neighborhood_is_four :
  let neighborhoods : ℕ := 10
  let lights_per_side : ℕ := 250
  let total_lights : ℕ := 20000
  roads_per_neighborhood * neighborhoods * (2 * lights_per_side) = total_lights :=
by sorry

end NUMINAMATH_CALUDE_roads_per_neighborhood_is_four_l3950_395028


namespace NUMINAMATH_CALUDE_ap_80th_term_l3950_395072

/-- An arithmetic progression (AP) with given properties -/
structure AP where
  /-- Sum of the first 20 terms -/
  sum20 : ℚ
  /-- Sum of the first 60 terms -/
  sum60 : ℚ
  /-- The property that sum20 = 200 -/
  sum20_eq : sum20 = 200
  /-- The property that sum60 = 180 -/
  sum60_eq : sum60 = 180

/-- The 80th term of the AP -/
def term80 (ap : AP) : ℚ := -573/40

/-- Theorem stating that the 80th term of the AP with given properties is -573/40 -/
theorem ap_80th_term (ap : AP) : term80 ap = -573/40 := by
  sorry

end NUMINAMATH_CALUDE_ap_80th_term_l3950_395072


namespace NUMINAMATH_CALUDE_log_absolute_equality_l3950_395076

/-- Given a function f(x) = |log x|, prove that if 0 < a < b and f(a) = f(b), then ab = 1 -/
theorem log_absolute_equality (a b : ℝ) (h1 : 0 < a) (h2 : a < b) 
  (h3 : |Real.log a| = |Real.log b|) : a * b = 1 := by
  sorry

end NUMINAMATH_CALUDE_log_absolute_equality_l3950_395076


namespace NUMINAMATH_CALUDE_penny_socks_l3950_395067

def sock_problem (initial_amount : ℕ) (sock_cost : ℕ) (hat_cost : ℕ) (remaining_amount : ℕ) : Prop :=
  ∃ (num_socks : ℕ), 
    initial_amount = sock_cost * num_socks + hat_cost + remaining_amount

theorem penny_socks : sock_problem 20 2 7 5 → ∃ (num_socks : ℕ), num_socks = 4 := by
  sorry

end NUMINAMATH_CALUDE_penny_socks_l3950_395067


namespace NUMINAMATH_CALUDE_tea_consumption_l3950_395061

theorem tea_consumption (total : ℕ) (days : ℕ) (diff : ℕ) : 
  total = 120 → days = 6 → diff = 4 → 
  ∃ (first : ℕ), 
    (first + 3 * diff = 22) ∧ 
    (days * (2 * first + (days - 1) * diff) / 2 = total) := by
  sorry

end NUMINAMATH_CALUDE_tea_consumption_l3950_395061


namespace NUMINAMATH_CALUDE_factorization_of_polynomial_l3950_395070

theorem factorization_of_polynomial (x : ℝ) :
  x^2 - 6*x + 9 - 64*x^4 = (-8*x^2 + x - 3)*(8*x^2 - x + 3) ∧
  (∀ a b c d e f : ℤ, (-8*x^2 + x - 3) = a*x^2 + b*x + c ∧
                       (8*x^2 - x + 3) = d*x^2 + e*x + f ∧
                       a < d) :=
by sorry

end NUMINAMATH_CALUDE_factorization_of_polynomial_l3950_395070


namespace NUMINAMATH_CALUDE_quadratic_shift_l3950_395048

/-- The original quadratic function -/
def g (x : ℝ) : ℝ := (x + 1)^2 + 3

/-- The transformed quadratic function -/
def f (x : ℝ) : ℝ := (x - 1)^2 + 2

/-- Theorem stating that f is the result of shifting g 2 units right and 1 unit down -/
theorem quadratic_shift (x : ℝ) : f x = g (x - 2) - 1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_shift_l3950_395048


namespace NUMINAMATH_CALUDE_grid_arrangement_impossibility_l3950_395016

theorem grid_arrangement_impossibility :
  ¬ ∃ (grid : Fin 25 → Fin 41 → ℤ),
    (∀ i j i' j', grid i j = grid i' j' → (i = i' ∧ j = j')) ∧
    (∀ i j,
      (i.val + 1 < 25 → |grid i j - grid ⟨i.val + 1, sorry⟩ j| ≤ 16) ∧
      (j.val + 1 < 41 → |grid i j - grid i ⟨j.val + 1, sorry⟩| ≤ 16)) :=
sorry

end NUMINAMATH_CALUDE_grid_arrangement_impossibility_l3950_395016


namespace NUMINAMATH_CALUDE_equation_solution_l3950_395008

theorem equation_solution (x : ℝ) :
  Real.sqrt ((3 / x) + 3) = 5 / 3 → x = -27 / 2 := by
sorry

end NUMINAMATH_CALUDE_equation_solution_l3950_395008


namespace NUMINAMATH_CALUDE_curve_self_intersection_l3950_395082

/-- The x-coordinate of a point on the curve as a function of t -/
def x (t : ℝ) : ℝ := t^2 - 4

/-- The y-coordinate of a point on the curve as a function of t -/
def y (t : ℝ) : ℝ := t^3 - 6*t + 3

/-- Theorem stating that (2,3) is the only self-intersection point of the curve -/
theorem curve_self_intersection :
  ∃! p : ℝ × ℝ, p.1 = 2 ∧ p.2 = 3 ∧
  ∃ a b : ℝ, a ≠ b ∧ 
    x a = x b ∧ y a = y b ∧
    x a = p.1 ∧ y a = p.2 :=
sorry

end NUMINAMATH_CALUDE_curve_self_intersection_l3950_395082


namespace NUMINAMATH_CALUDE_flag_design_count_l3950_395092

/-- The number of possible colors for each stripe -/
def num_colors : ℕ := 3

/-- The number of stripes on the flag -/
def num_stripes : ℕ := 3

/-- The number of possible flag designs -/
def num_flag_designs : ℕ := num_colors ^ num_stripes

theorem flag_design_count :
  num_flag_designs = 27 :=
sorry

end NUMINAMATH_CALUDE_flag_design_count_l3950_395092


namespace NUMINAMATH_CALUDE_y_value_at_x_2_l3950_395019

/-- Given y₁ = x² - 7x + 6, y₂ = 7x - 3, and y = y₁ + xy₂, prove that when x = 2, y = 18. -/
theorem y_value_at_x_2 :
  let y₁ : ℝ → ℝ := λ x => x^2 - 7*x + 6
  let y₂ : ℝ → ℝ := λ x => 7*x - 3
  let y : ℝ → ℝ := λ x => y₁ x + x * y₂ x
  y 2 = 18 := by sorry

end NUMINAMATH_CALUDE_y_value_at_x_2_l3950_395019


namespace NUMINAMATH_CALUDE_otimes_inequality_implies_a_range_l3950_395081

-- Define the ⊗ operation
def otimes (x y : ℝ) := x * (1 - y)

-- State the theorem
theorem otimes_inequality_implies_a_range (a : ℝ) :
  (∀ x : ℝ, otimes (x - a) (x + a) < 1) →
  -1/2 < a ∧ a < 3/2 :=
by sorry

end NUMINAMATH_CALUDE_otimes_inequality_implies_a_range_l3950_395081


namespace NUMINAMATH_CALUDE_reciprocal_of_negative_nine_l3950_395017

theorem reciprocal_of_negative_nine (x : ℚ) : 
  (x * (-9) = 1) → x = -1/9 := by
  sorry

end NUMINAMATH_CALUDE_reciprocal_of_negative_nine_l3950_395017


namespace NUMINAMATH_CALUDE_repeating_decimal_sum_l3950_395045

theorem repeating_decimal_sum (a b : ℕ+) : 
  (a.val : ℚ) / b.val = 4 / 11 → Nat.gcd a.val b.val = 1 → a.val + b.val = 15 := by
  sorry

end NUMINAMATH_CALUDE_repeating_decimal_sum_l3950_395045


namespace NUMINAMATH_CALUDE_literary_readers_count_l3950_395029

theorem literary_readers_count (total : ℕ) (sci_fi : ℕ) (both : ℕ) (literary : ℕ) : 
  total = 150 → sci_fi = 120 → both = 60 → literary = total - sci_fi + both → literary = 90 :=
by
  sorry

end NUMINAMATH_CALUDE_literary_readers_count_l3950_395029


namespace NUMINAMATH_CALUDE_simplify_expression_l3950_395060

theorem simplify_expression (x y z : ℝ) 
  (hx : x ≠ 2) (hy : y ≠ 3) (hz : z ≠ 4) : 
  (x - 2) / (4 - z) * (y - 3) / (2 - x) * (z - 4) / (3 - y) = -1 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l3950_395060


namespace NUMINAMATH_CALUDE_keenan_essay_words_l3950_395020

/-- Represents Keenan's essay writing scenario -/
structure EssayWriting where
  initial_rate : ℕ  -- Words per hour for first two hours
  later_rate : ℕ    -- Words per hour after first two hours
  total_time : ℕ    -- Total time available in hours

/-- Calculates the total number of words Keenan can write -/
def total_words (e : EssayWriting) : ℕ :=
  (e.initial_rate * 2) + (e.later_rate * (e.total_time - 2))

/-- Theorem stating that Keenan can write 1200 words given the conditions -/
theorem keenan_essay_words :
  ∃ (e : EssayWriting), e.initial_rate = 400 ∧ e.later_rate = 200 ∧ e.total_time = 4 ∧ total_words e = 1200 := by
  sorry

end NUMINAMATH_CALUDE_keenan_essay_words_l3950_395020


namespace NUMINAMATH_CALUDE_pens_in_pack_l3950_395007

/-- The number of pens in each pack -/
def pens_per_pack : ℕ := sorry

/-- The number of packs Kendra has -/
def kendra_packs : ℕ := 4

/-- The number of packs Tony has -/
def tony_packs : ℕ := 2

/-- The number of pens Kendra and Tony keep for themselves -/
def pens_kept : ℕ := 4

/-- The number of friends they give pens to -/
def friends : ℕ := 14

theorem pens_in_pack : 
  (kendra_packs + tony_packs) * pens_per_pack - pens_kept - friends = 0 ∧ 
  pens_per_pack = 3 := by sorry

end NUMINAMATH_CALUDE_pens_in_pack_l3950_395007


namespace NUMINAMATH_CALUDE_slope_value_l3950_395086

theorem slope_value (m : ℝ) : 
  let A : ℝ × ℝ := (-m, 6)
  let B : ℝ × ℝ := (1, 3*m)
  let slope := (B.2 - A.2) / (B.1 - A.1)
  slope = 12 → m = -2 := by
sorry

end NUMINAMATH_CALUDE_slope_value_l3950_395086


namespace NUMINAMATH_CALUDE_p_sufficient_not_necessary_for_not_q_l3950_395083

theorem p_sufficient_not_necessary_for_not_q :
  (∀ x : ℝ, x^2 + 2*x - 3 ≤ 0 → |x| ≤ 3) ∧
  (∃ x : ℝ, |x| ≤ 3 ∧ x^2 + 2*x - 3 > 0) :=
by sorry

end NUMINAMATH_CALUDE_p_sufficient_not_necessary_for_not_q_l3950_395083


namespace NUMINAMATH_CALUDE_four_digit_number_proof_l3950_395031

/-- Represents a four-digit number -/
structure FourDigitNumber where
  value : ℕ
  is_four_digit : 1000 ≤ value ∧ value ≤ 9999

/-- Returns the largest number that can be formed by rearranging the digits of a given number -/
def largest_rearrangement (n : FourDigitNumber) : ℕ :=
  sorry

/-- Returns the smallest number that can be formed by rearranging the digits of a given number -/
def smallest_rearrangement (n : FourDigitNumber) : ℕ :=
  sorry

/-- Checks if a number has any digit equal to 0 -/
def has_zero_digit (n : ℕ) : Bool :=
  sorry

theorem four_digit_number_proof :
  ∃ (A : FourDigitNumber),
    largest_rearrangement A = A.value + 7668 ∧
    smallest_rearrangement A = A.value - 594 ∧
    ¬ has_zero_digit A.value ∧
    A.value = 1963 :=
  sorry

end NUMINAMATH_CALUDE_four_digit_number_proof_l3950_395031


namespace NUMINAMATH_CALUDE_average_of_solutions_is_zero_l3950_395035

theorem average_of_solutions_is_zero :
  let solutions := {x : ℝ | Real.sqrt (3 * x^2 + 4) = Real.sqrt 28}
  ∃ (x₁ x₂ : ℝ), x₁ ∈ solutions ∧ x₂ ∈ solutions ∧ x₁ ≠ x₂ ∧
    (x₁ + x₂) / 2 = 0 ∧
    ∀ (x : ℝ), x ∈ solutions → x = x₁ ∨ x = x₂ :=
by sorry

end NUMINAMATH_CALUDE_average_of_solutions_is_zero_l3950_395035


namespace NUMINAMATH_CALUDE_right_triangle_sin_d_l3950_395068

theorem right_triangle_sin_d (D E F : ℝ) (h1 : 0 < D) (h2 : D < π / 2) : 
  5 * Real.sin D = 12 * Real.cos D → Real.sin D = 12 / 13 := by
sorry

end NUMINAMATH_CALUDE_right_triangle_sin_d_l3950_395068


namespace NUMINAMATH_CALUDE_morgan_sat_score_l3950_395064

theorem morgan_sat_score (second_score : ℝ) (improvement_rate : ℝ) :
  second_score = 1100 →
  improvement_rate = 0.1 →
  ∃ (first_score : ℝ), first_score * (1 + improvement_rate) = second_score ∧ first_score = 1000 :=
by
  sorry

end NUMINAMATH_CALUDE_morgan_sat_score_l3950_395064


namespace NUMINAMATH_CALUDE_unique_numbers_l3950_395091

/-- Checks if a three-digit number has distinct digits in ascending order -/
def has_ascending_digits (n : ℕ) : Prop :=
  n ≥ 100 ∧ n < 1000 ∧
  ∃ (a b c : ℕ), n = 100 * a + 10 * b + c ∧ a < b ∧ b < c

/-- Checks if a three-digit number has identical digits -/
def has_identical_digits (n : ℕ) : Prop :=
  n ≥ 100 ∧ n < 1000 ∧
  ∃ (a : ℕ), n = 100 * a + 10 * a + a

/-- Checks if all words in the name of a number start with the same letter -/
def name_starts_same_letter (n : ℕ) : Prop :=
  -- This is a placeholder for the actual condition
  n = 147

/-- Checks if all words in the name of a number start with different letters -/
def name_starts_different_letters (n : ℕ) : Prop :=
  -- This is a placeholder for the actual condition
  n = 111

theorem unique_numbers :
  (∃! n : ℕ, has_ascending_digits n ∧ name_starts_same_letter n) ∧
  (∃! n : ℕ, has_identical_digits n ∧ name_starts_different_letters n) :=
sorry

end NUMINAMATH_CALUDE_unique_numbers_l3950_395091


namespace NUMINAMATH_CALUDE_pens_distribution_ways_l3950_395003

/-- The number of ways to distribute n identical objects among k recipients,
    where each recipient must receive at least one object. -/
def distribute (n k : ℕ) : ℕ := Nat.choose (n - 1) (k - 1)

/-- The number of ways to distribute 9 pens among 3 friends,
    where each friend must receive at least one pen. -/
def distribute_pens : ℕ := distribute 9 3

theorem pens_distribution_ways : distribute_pens = 28 := by sorry

end NUMINAMATH_CALUDE_pens_distribution_ways_l3950_395003


namespace NUMINAMATH_CALUDE_chord_max_surface_area_l3950_395047

/-- 
Given a circle with radius R, the chord of length R√2 maximizes the surface area 
of the cylindrical shell formed when rotating the chord around the diameter parallel to it.
-/
theorem chord_max_surface_area (R : ℝ) (R_pos : R > 0) : 
  let chord_length (x : ℝ) := 2 * x
  let surface_area (x : ℝ) := 4 * Real.pi * x * Real.sqrt (R^2 - x^2)
  ∃ (x : ℝ), x > 0 ∧ x < R ∧ 
    chord_length x = R * Real.sqrt 2 ∧
    ∀ (y : ℝ), y > 0 → y < R → surface_area x ≥ surface_area y :=
by sorry

end NUMINAMATH_CALUDE_chord_max_surface_area_l3950_395047


namespace NUMINAMATH_CALUDE_max_value_even_function_l3950_395058

def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

theorem max_value_even_function 
  (f : ℝ → ℝ) 
  (h_even : is_even_function f) 
  (h_max : ∃ x ∈ Set.Icc (-3) (-1), ∀ y ∈ Set.Icc (-3) (-1), f y ≤ f x ∧ f x = 6) :
  ∃ x ∈ Set.Icc 1 3, ∀ y ∈ Set.Icc 1 3, f y ≤ f x ∧ f x = 6 :=
sorry

end NUMINAMATH_CALUDE_max_value_even_function_l3950_395058


namespace NUMINAMATH_CALUDE_exponent_multiplication_l3950_395071

theorem exponent_multiplication (a : ℝ) : a^4 * a^3 = a^7 := by sorry

end NUMINAMATH_CALUDE_exponent_multiplication_l3950_395071


namespace NUMINAMATH_CALUDE_inequality_solution_fractional_equation_no_solution_l3950_395043

-- Part 1: Inequality
theorem inequality_solution (x : ℝ) : 
  (1 - x) / 3 - x < 3 - (x + 2) / 4 ↔ x > -2 :=
sorry

-- Part 2: Fractional equation
theorem fractional_equation_no_solution :
  ¬∃ (x : ℝ), (x - 2) / (2 * x - 1) + 1 = 3 / (2 * (1 - 2 * x)) :=
sorry

end NUMINAMATH_CALUDE_inequality_solution_fractional_equation_no_solution_l3950_395043


namespace NUMINAMATH_CALUDE_limit_point_sequence_a_l3950_395004

def sequence_a (n : ℕ) : ℚ := (n + 1) / n

theorem limit_point_sequence_a :
  ∀ ε > 0, ∃ N : ℕ, ∀ n ≥ N, |sequence_a n - 1| < ε :=
sorry

end NUMINAMATH_CALUDE_limit_point_sequence_a_l3950_395004


namespace NUMINAMATH_CALUDE_greatest_integer_radius_l3950_395050

theorem greatest_integer_radius (A : ℝ) (h : A < 90 * Real.pi) :
  ∃ (r : ℕ), r^2 * Real.pi = A ∧ ∀ (n : ℕ), n^2 * Real.pi < 90 * Real.pi → n ≤ r ∧ r ≤ 9 :=
sorry

end NUMINAMATH_CALUDE_greatest_integer_radius_l3950_395050


namespace NUMINAMATH_CALUDE_perpendicular_parallel_implication_l3950_395024

-- Define a structure for a line in 3D space
structure Line3D where
  direction : ℝ × ℝ × ℝ

-- Define perpendicularity for lines
def perpendicular (l1 l2 : Line3D) : Prop :=
  sorry

-- Define parallelism for lines
def parallel (l1 l2 : Line3D) : Prop :=
  sorry

-- Theorem statement
theorem perpendicular_parallel_implication (a b c : Line3D) 
  (h1 : perpendicular a b) (h2 : parallel b c) : perpendicular a c :=
sorry

end NUMINAMATH_CALUDE_perpendicular_parallel_implication_l3950_395024


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_l3950_395034

theorem quadratic_equation_roots (a b c : ℝ) (h1 : a > 0) (h2 : c < 0) :
  ∃ x y : ℝ, x ≠ y ∧ a * x^2 + b * x + c = 0 ∧ a * y^2 + b * y + c = 0 :=
sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_l3950_395034


namespace NUMINAMATH_CALUDE_coin_problem_l3950_395056

theorem coin_problem :
  ∀ (x y : ℕ),
    x + y = 15 →
    2 * x + 5 * y = 51 →
    x = y + 1 :=
by
  sorry

end NUMINAMATH_CALUDE_coin_problem_l3950_395056


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_l3950_395079

theorem quadratic_equation_roots (a : ℝ) : 
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ a * x₁^2 + 2 * x₁ + 3 = 0 ∧ a * x₂^2 + 2 * x₂ + 3 = 0) →
  a = -2 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_l3950_395079


namespace NUMINAMATH_CALUDE_two_digit_number_18_l3950_395062

/-- Represents a two-digit number as a pair of natural numbers -/
def TwoDigitNumber := Nat × Nat

/-- Converts a two-digit number to its decimal representation -/
def toDecimal (n : TwoDigitNumber) : ℚ :=
  (n.1 * 10 + n.2 : ℚ) / 100

/-- Converts a two-digit number to its repeating decimal representation -/
def toRepeatingDecimal (n : TwoDigitNumber) : ℚ :=
  1 + (n.1 * 10 + n.2 : ℚ) / 99

theorem two_digit_number_18 (c d : Nat) (h_c : c < 10) (h_d : d < 10) :
  55 * toRepeatingDecimal (c, d) - 55 * (1 + toDecimal (c, d)) = 1 →
  c = 1 ∧ d = 8 := by
  sorry

end NUMINAMATH_CALUDE_two_digit_number_18_l3950_395062


namespace NUMINAMATH_CALUDE_parabola_distance_theorem_l3950_395037

/-- Parabola type representing y^2 = 8x -/
structure Parabola where
  equation : ℝ → ℝ → Prop
  focus : ℝ × ℝ
  directrix : ℝ → ℝ → Prop

/-- Represents a point on the parabola -/
structure ParabolaPoint (p : Parabola) where
  point : ℝ × ℝ
  on_parabola : p.equation point.1 point.2

theorem parabola_distance_theorem (p : Parabola) 
  (h_equation : p.equation = fun x y ↦ y^2 = 8*x)
  (P : ParabolaPoint p)
  (A : ℝ × ℝ)
  (h_perpendicular : (P.point.1 - A.1) * (P.point.2 - A.2) = 0)
  (h_on_directrix : p.directrix A.1 A.2)
  (h_slope : (A.2 - p.focus.2) / (A.1 - p.focus.1) = -Real.sqrt 3) :
  Real.sqrt ((P.point.1 - p.focus.1)^2 + (P.point.2 - p.focus.2)^2) = 8 := by
  sorry

end NUMINAMATH_CALUDE_parabola_distance_theorem_l3950_395037


namespace NUMINAMATH_CALUDE_range_of_p_l3950_395041

-- Define the function f
def f (x : ℝ) : ℝ := x^3 - x^2 - 10*x

-- Define the derivative of f
def f' (x : ℝ) : ℝ := 3*x^2 - 2*x - 10

-- Define set A
def A : Set ℝ := {x | f' x ≤ 0}

-- Define set B
def B (p : ℝ) : Set ℝ := {x | p + 1 ≤ x ∧ x ≤ 2*p - 1}

-- Theorem statement
theorem range_of_p (p : ℝ) : A ∪ B p = A → p ≤ 3 := by
  sorry

end NUMINAMATH_CALUDE_range_of_p_l3950_395041


namespace NUMINAMATH_CALUDE_distance_to_directrix_l3950_395011

/-- The distance from a point on a parabola to its directrix -/
theorem distance_to_directrix (p : ℝ) (x y : ℝ) (h : y^2 = 2*p*x) :
  x + p/2 = 9/4 :=
sorry

end NUMINAMATH_CALUDE_distance_to_directrix_l3950_395011


namespace NUMINAMATH_CALUDE_contrapositive_isosceles_equal_angles_l3950_395012

-- Define a triangle
structure Triangle where
  A : Point
  B : Point
  C : Point

-- Define properties of a triangle
def Triangle.isIsosceles (t : Triangle) : Prop := sorry
def Triangle.hasEqualInteriorAngles (t : Triangle) : Prop := sorry

-- State the theorem
theorem contrapositive_isosceles_equal_angles (t : Triangle) :
  (¬(t.isIsosceles) → ¬(t.hasEqualInteriorAngles)) ↔
  (t.hasEqualInteriorAngles → t.isIsosceles) := by sorry

end NUMINAMATH_CALUDE_contrapositive_isosceles_equal_angles_l3950_395012


namespace NUMINAMATH_CALUDE_intersection_points_slope_l3950_395033

theorem intersection_points_slope :
  ∀ (s x y : ℝ), 
    (2 * x + 3 * y = 8 * s + 5) →
    (x + 2 * y = 3 * s + 2) →
    y = -(7/2) * x + 1/2 := by
  sorry

end NUMINAMATH_CALUDE_intersection_points_slope_l3950_395033


namespace NUMINAMATH_CALUDE_candidate_votes_l3950_395055

theorem candidate_votes (total_votes : ℕ) (invalid_percent : ℚ) (candidate_percent : ℚ) :
  total_votes = 560000 →
  invalid_percent = 15 / 100 →
  candidate_percent = 80 / 100 →
  ∃ (valid_votes : ℕ) (candidate_votes : ℕ),
    valid_votes = (1 - invalid_percent) * total_votes ∧
    candidate_votes = candidate_percent * valid_votes ∧
    candidate_votes = 380800 := by
  sorry

end NUMINAMATH_CALUDE_candidate_votes_l3950_395055


namespace NUMINAMATH_CALUDE_rectangle_area_l3950_395023

/-- The area of a rectangle composed of 24 congruent squares arranged in a 6x4 grid, with a diagonal of 10 cm, is 600/13 square cm. -/
theorem rectangle_area (diagonal : ℝ) (rows columns : ℕ) (num_squares : ℕ) : 
  diagonal = 10 → 
  rows = 6 → 
  columns = 4 → 
  num_squares = 24 → 
  (rows * columns : ℝ) * (diagonal^2 / ((rows : ℝ)^2 + (columns : ℝ)^2)) = 600 / 13 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_l3950_395023


namespace NUMINAMATH_CALUDE_unique_square_sum_l3950_395042

theorem unique_square_sum : ∃! x : ℕ+, 
  (∃ m : ℕ+, (x : ℕ) + 100 = m^2) ∧ 
  (∃ n : ℕ+, (x : ℕ) + 168 = n^2) :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_unique_square_sum_l3950_395042


namespace NUMINAMATH_CALUDE_sum_product_difference_l3950_395054

theorem sum_product_difference (x y : ℝ) (h1 : x + y = 25) (h2 : x * y = 126) : 
  |x - y| = 11 := by
sorry

end NUMINAMATH_CALUDE_sum_product_difference_l3950_395054


namespace NUMINAMATH_CALUDE_xy_square_sum_l3950_395010

theorem xy_square_sum (x y : ℝ) (h1 : x + y = -2) (h2 : x * y = -3) :
  x^2 * y + x * y^2 = 6 := by
  sorry

end NUMINAMATH_CALUDE_xy_square_sum_l3950_395010


namespace NUMINAMATH_CALUDE_remainder_problem_l3950_395052

theorem remainder_problem (m : ℤ) : (((8 - m) + (m + 4)) % 5) % 3 = 2 := by
  sorry

end NUMINAMATH_CALUDE_remainder_problem_l3950_395052


namespace NUMINAMATH_CALUDE_complement_intersection_equals_set_l3950_395049

universe u

def U : Set ℕ := {0, 1, 2, 3, 4}
def A : Set ℕ := {0, 1, 2, 3}
def B : Set ℕ := {2, 3, 4}

theorem complement_intersection_equals_set : 
  (A ∩ B)ᶜ = {0, 1, 4} := by sorry

end NUMINAMATH_CALUDE_complement_intersection_equals_set_l3950_395049


namespace NUMINAMATH_CALUDE_sum_of_solutions_is_zero_l3950_395018

theorem sum_of_solutions_is_zero (x : ℝ) (h : x^2 - 4 = 36) :
  ∃ y : ℝ, y^2 - 4 = 36 ∧ x + y = 0 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_solutions_is_zero_l3950_395018


namespace NUMINAMATH_CALUDE_fraction_change_l3950_395099

theorem fraction_change (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) :
  (0.6 * x) / (0.4 * y) = 1.5 * (x / y) := by
sorry

end NUMINAMATH_CALUDE_fraction_change_l3950_395099


namespace NUMINAMATH_CALUDE_elizabeth_steak_knife_cost_l3950_395084

def steak_knife_cost (sets : ℕ) (knives_per_set : ℕ) (cost_per_set : ℚ) : ℚ :=
  (sets * cost_per_set) / (sets * knives_per_set)

theorem elizabeth_steak_knife_cost :
  steak_knife_cost 2 4 80 = 20 := by
  sorry

end NUMINAMATH_CALUDE_elizabeth_steak_knife_cost_l3950_395084


namespace NUMINAMATH_CALUDE_evaluate_complex_expression_l3950_395077

theorem evaluate_complex_expression :
  let M := (Real.sqrt (Real.sqrt 7 + 3) + Real.sqrt (Real.sqrt 7 - 3)) / Real.sqrt (Real.sqrt 7 + 2) + Real.sqrt (5 - 2 * Real.sqrt 6)
  M = 2 * Real.sqrt 2 - 1 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_complex_expression_l3950_395077


namespace NUMINAMATH_CALUDE_smallest_n_for_g_greater_than_15_l3950_395046

def g (n : ℕ+) : ℕ := 
  (Nat.digits 10 ((10^n.val) / (7^n.val))).sum

theorem smallest_n_for_g_greater_than_15 : 
  ∀ k : ℕ+, k < 12 → g k ≤ 15 ∧ g 12 > 15 := by sorry

end NUMINAMATH_CALUDE_smallest_n_for_g_greater_than_15_l3950_395046


namespace NUMINAMATH_CALUDE_third_term_is_six_l3950_395006

/-- An arithmetic sequence with specific properties -/
structure ArithmeticSequence where
  a : ℕ → ℝ  -- The sequence
  is_arithmetic : ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)
  sum_second_fourth : a 2 + a 4 = 10
  fourth_minus_third : a 4 = a 3 + 2

/-- The third term of the arithmetic sequence is 6 -/
theorem third_term_is_six (seq : ArithmeticSequence) : seq.a 3 = 6 := by
  sorry

end NUMINAMATH_CALUDE_third_term_is_six_l3950_395006


namespace NUMINAMATH_CALUDE_product_is_three_digit_l3950_395036

def smallest_three_digit_number : ℕ := 100
def largest_single_digit_number : ℕ := 9

def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

theorem product_is_three_digit : 
  is_three_digit (smallest_three_digit_number * largest_single_digit_number) := by
  sorry

end NUMINAMATH_CALUDE_product_is_three_digit_l3950_395036


namespace NUMINAMATH_CALUDE_cellphone_selection_theorem_l3950_395015

/-- The number of service providers -/
def total_providers : ℕ := 25

/-- The number of siblings (including Laura) -/
def num_siblings : ℕ := 4

/-- The number of ways to select providers for all siblings -/
def ways_to_select_providers : ℕ := 
  (total_providers - 1) * (total_providers - 2) * (total_providers - 3)

theorem cellphone_selection_theorem :
  ways_to_select_providers = 12144 := by
  sorry

end NUMINAMATH_CALUDE_cellphone_selection_theorem_l3950_395015


namespace NUMINAMATH_CALUDE_compressor_stations_configuration_l3950_395014

/-- Represents a triangle with side lengths x, y, and z. -/
structure Triangle where
  x : ℝ
  y : ℝ
  z : ℝ
  triangle_inequality : x < y + z ∧ y < x + z ∧ z < x + y

/-- Theorem about the specific triangle configuration described in the problem. -/
theorem compressor_stations_configuration (a : ℝ) :
  ∃ (t : Triangle),
    t.x + t.y = 3 * t.z ∧
    t.z + t.y = t.x + a ∧
    t.x + t.z = 60 →
    0 < a ∧ a < 60 ∧
    (a = 30 → t.x = 35 ∧ t.y = 40 ∧ t.z = 25) :=
by sorry

#check compressor_stations_configuration

end NUMINAMATH_CALUDE_compressor_stations_configuration_l3950_395014


namespace NUMINAMATH_CALUDE_intersection_M_N_l3950_395022

def M : Set ℝ := {x | x^2 + 3*x + 2 > 0}
def N : Set ℝ := {-2, -1, 0, 1, 2}

theorem intersection_M_N : M ∩ N = {0, 1, 2} := by sorry

end NUMINAMATH_CALUDE_intersection_M_N_l3950_395022


namespace NUMINAMATH_CALUDE_negation_equivalence_l3950_395059

theorem negation_equivalence (a : ℝ) :
  (¬ ∃ x : ℝ, x^2 + a*x + 1 < 0) ↔ (∀ x : ℝ, x^2 + a*x + 1 ≥ 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_equivalence_l3950_395059


namespace NUMINAMATH_CALUDE_stating_min_problems_olympiad_l3950_395044

/-- The number of students in the olympiad -/
def num_students : ℕ := 55

/-- 
The function that calculates the maximum number of distinct pairs of "+" and "-" scores
for a given number of problems.
-/
def max_distinct_pairs (num_problems : ℕ) : ℕ :=
  (num_problems + 1) * (num_problems + 2) / 2

/-- 
Theorem stating that the minimum number of problems needed in the olympiad is 9,
given that there are 55 students and no two students can have the same number of "+" and "-" scores.
-/
theorem min_problems_olympiad :
  ∃ (n : ℕ), n = 9 ∧ max_distinct_pairs n = num_students ∧
  ∀ (m : ℕ), m < n → max_distinct_pairs m < num_students :=
by sorry

end NUMINAMATH_CALUDE_stating_min_problems_olympiad_l3950_395044


namespace NUMINAMATH_CALUDE_calculation_proofs_l3950_395065

theorem calculation_proofs (x y a b : ℝ) :
  (((1/2) * x * y)^2 * (6 * x^2 * y) = (3/2) * x^4 * y^3) ∧
  ((2*a + b)^2 = 4*a^2 + 4*a*b + b^2) := by sorry

end NUMINAMATH_CALUDE_calculation_proofs_l3950_395065


namespace NUMINAMATH_CALUDE_complex_square_l3950_395051

theorem complex_square (z : ℂ) (i : ℂ) (h1 : z = 5 - 3*i) (h2 : i^2 = -1) :
  z^2 = 16 - 30*i := by sorry

end NUMINAMATH_CALUDE_complex_square_l3950_395051


namespace NUMINAMATH_CALUDE_dividend_calculation_l3950_395089

theorem dividend_calculation (divisor quotient remainder : ℕ) 
  (h_divisor : divisor = 19)
  (h_quotient : quotient = 7)
  (h_remainder : remainder = 6) :
  divisor * quotient + remainder = 139 := by
sorry

end NUMINAMATH_CALUDE_dividend_calculation_l3950_395089


namespace NUMINAMATH_CALUDE_tiger_distance_is_160_l3950_395097

/-- Calculates the distance traveled given speed and time -/
def distance (speed : ℝ) (time : ℝ) : ℝ := speed * time

/-- The total distance traveled by the escaped tiger -/
def tiger_distance : ℝ :=
  distance 25 1 + distance 35 2 + distance 20 1.5 + distance 10 1 + distance 50 0.5

/-- Theorem stating that the tiger traveled 160 miles -/
theorem tiger_distance_is_160 : tiger_distance = 160 := by
  sorry

end NUMINAMATH_CALUDE_tiger_distance_is_160_l3950_395097


namespace NUMINAMATH_CALUDE_complex_equality_sum_l3950_395093

-- Define the complex number i
def i : ℂ := Complex.I

-- Define the theorem
theorem complex_equality_sum (a b : ℝ) (h : (1 + i) * i = a + b * i) : a + b = 0 := by
  sorry

end NUMINAMATH_CALUDE_complex_equality_sum_l3950_395093


namespace NUMINAMATH_CALUDE_series_sum_l3950_395090

/-- The positive real solution to x^3 + (2/5)x - 1 = 0 -/
noncomputable def r : ℝ :=
  Real.sqrt (Real.sqrt (1 + 2/5))

/-- The sum of the series r^2 + 2r^5 + 3r^8 + 4r^11 + ... -/
noncomputable def S : ℝ :=
  ∑' n, (n + 1) * r^(3*n + 2)

theorem series_sum : 
  r > 0 ∧ r^3 + 2/5 * r - 1 = 0 → S = 25/4 :=
by
  sorry

#check series_sum

end NUMINAMATH_CALUDE_series_sum_l3950_395090


namespace NUMINAMATH_CALUDE_new_line_equation_l3950_395021

/-- Given a line y = mx + b, proves that a new line with half the slope
    and triple the y-intercept has the equation y = (m/2)x + 3b -/
theorem new_line_equation (m b : ℝ) :
  let original_line := fun x => m * x + b
  let new_line := fun x => (m / 2) * x + 3 * b
  ∀ x, new_line x = (m / 2) * x + 3 * b :=
by sorry

end NUMINAMATH_CALUDE_new_line_equation_l3950_395021


namespace NUMINAMATH_CALUDE_vector_properties_l3950_395000

/-- Given vectors a, b, c and x ∈ [0,π], prove two statements about x and sin(x + π/6) -/
theorem vector_properties (x : Real) 
  (hx : x ∈ Set.Icc 0 Real.pi)
  (a : Fin 2 → Real)
  (ha : a = fun i => if i = 0 then Real.sin x else Real.sqrt 3 * Real.cos x)
  (b : Fin 2 → Real)
  (hb : b = fun i => if i = 0 then -1 else 1)
  (c : Fin 2 → Real)
  (hc : c = fun i => if i = 0 then 1 else -1) :
  (∃ (k : Real), (a + b) = k • c → x = 5 * Real.pi / 6) ∧
  (a • b = 1 / 2 → Real.sin (x + Real.pi / 6) = Real.sqrt 15 / 4) := by
sorry

end NUMINAMATH_CALUDE_vector_properties_l3950_395000


namespace NUMINAMATH_CALUDE_smallest_twice_square_three_cube_l3950_395095

def is_twice_perfect_square (n : ℕ) : Prop :=
  ∃ k : ℕ, n = 2 * k^2

def is_three_times_perfect_cube (n : ℕ) : Prop :=
  ∃ m : ℕ, n = 3 * m^3

theorem smallest_twice_square_three_cube :
  (∀ n : ℕ, n > 0 ∧ n < 648 → ¬(is_twice_perfect_square n ∧ is_three_times_perfect_cube n)) ∧
  (is_twice_perfect_square 648 ∧ is_three_times_perfect_cube 648) :=
sorry

end NUMINAMATH_CALUDE_smallest_twice_square_three_cube_l3950_395095


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l3950_395038

/-- A geometric sequence with positive terms -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r > 0 ∧ ∀ n : ℕ, a (n + 1) = r * a n

theorem geometric_sequence_sum (a : ℕ → ℝ) :
  GeometricSequence a →
  a 3 = 3 →
  a 6 = 1 / 9 →
  a 4 + a 5 = 4 / 3 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l3950_395038


namespace NUMINAMATH_CALUDE_tree_planting_event_l3950_395098

theorem tree_planting_event (boys girls : ℕ) : 
  girls = boys + 400 →
  girls > boys →
  (60 : ℚ) / 100 * (boys + girls : ℚ) = 960 →
  boys = 600 := by
sorry

end NUMINAMATH_CALUDE_tree_planting_event_l3950_395098


namespace NUMINAMATH_CALUDE_inequality_order_l3950_395073

theorem inequality_order (a b : ℝ) (ha : a = 6) (hb : b = 3) :
  (a + 3*b) / 4 < (a^2 * b)^(1/3) ∧ (a^2 * b)^(1/3) < (a + 3*b)^2 / (4*(a + b)) := by
  sorry

end NUMINAMATH_CALUDE_inequality_order_l3950_395073


namespace NUMINAMATH_CALUDE_quadratic_roots_ratio_l3950_395053

theorem quadratic_roots_ratio (m : ℝ) : 
  (∃ r s : ℝ, r ≠ 0 ∧ s ≠ 0 ∧ r / s = 3 / 2 ∧ 
   r + s = -10 ∧ r * s = m ∧ 
   ∀ x : ℝ, x^2 + 10*x + m = 0 ↔ (x = r ∨ x = s)) → 
  m = 24 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_ratio_l3950_395053


namespace NUMINAMATH_CALUDE_jesses_friends_bananas_l3950_395026

/-- The total number of bananas given the number of friends and bananas per friend -/
def total_bananas (num_friends : ℝ) (bananas_per_friend : ℝ) : ℝ :=
  num_friends * bananas_per_friend

/-- Theorem: Jesse's friends have 63.0 bananas in total -/
theorem jesses_friends_bananas :
  total_bananas 3.0 21.0 = 63.0 := by
  sorry

end NUMINAMATH_CALUDE_jesses_friends_bananas_l3950_395026


namespace NUMINAMATH_CALUDE_hundredth_term_value_l3950_395066

/-- A geometric sequence with first term 5 and second term -15 -/
def geometric_sequence (n : ℕ) : ℚ :=
  5 * (-3)^(n - 1)

/-- The 100th term of the geometric sequence -/
def a_100 : ℚ := geometric_sequence 100

theorem hundredth_term_value : a_100 = -5 * 3^99 := by
  sorry

end NUMINAMATH_CALUDE_hundredth_term_value_l3950_395066


namespace NUMINAMATH_CALUDE_right_triangle_area_l3950_395075

theorem right_triangle_area (DF EF : ℝ) (angle_DEF : ℝ) :
  DF = 4 →
  angle_DEF = π / 4 →
  DF = EF →
  (1 / 2) * DF * EF = 8 :=
by sorry

end NUMINAMATH_CALUDE_right_triangle_area_l3950_395075


namespace NUMINAMATH_CALUDE_base6_addition_theorem_l3950_395074

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

end NUMINAMATH_CALUDE_base6_addition_theorem_l3950_395074


namespace NUMINAMATH_CALUDE_units_digit_E_1000_l3950_395009

def E (n : ℕ) : ℕ := 3^(3^n) + 1

theorem units_digit_E_1000 : E 1000 % 10 = 4 := by sorry

end NUMINAMATH_CALUDE_units_digit_E_1000_l3950_395009


namespace NUMINAMATH_CALUDE_line_inclination_theorem_l3950_395039

/-- Given a line ax + by + c = 0 with inclination angle α, and sin α + cos α = 0, then a - b = 0 -/
theorem line_inclination_theorem (a b c : ℝ) (α : ℝ) : 
  (∃ x y, a * x + b * y + c = 0) →  -- line exists
  (Real.tan α = -a / b) →           -- definition of inclination angle
  (Real.sin α + Real.cos α = 0) →   -- given condition
  a - b = 0 := by
sorry

end NUMINAMATH_CALUDE_line_inclination_theorem_l3950_395039


namespace NUMINAMATH_CALUDE_haley_concert_spending_l3950_395069

/-- The amount spent on concert tickets -/
def concert_spending (ticket_price : ℕ) (tickets_for_self : ℕ) (extra_tickets : ℕ) : ℕ :=
  ticket_price * (tickets_for_self + extra_tickets)

theorem haley_concert_spending :
  concert_spending 4 3 5 = 32 := by
  sorry

end NUMINAMATH_CALUDE_haley_concert_spending_l3950_395069


namespace NUMINAMATH_CALUDE_boisjoli_farm_egg_production_l3950_395087

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

end NUMINAMATH_CALUDE_boisjoli_farm_egg_production_l3950_395087


namespace NUMINAMATH_CALUDE_sushi_lollipops_l3950_395032

theorem sushi_lollipops (x y : ℕ) : x + y = 27 :=
  by
    have h1 : x + y = 5 + (3 * 5) + 7 := by sorry
    have h2 : 5 + (3 * 5) + 7 = 27 := by sorry
    rw [h1, h2]

end NUMINAMATH_CALUDE_sushi_lollipops_l3950_395032


namespace NUMINAMATH_CALUDE_coin_toss_count_l3950_395002

theorem coin_toss_count (total_tosses : ℕ) (tail_count : ℕ) (head_count : ℕ) :
  total_tosses = 14 →
  tail_count = 5 →
  total_tosses = head_count + tail_count →
  head_count = 9 := by
  sorry

end NUMINAMATH_CALUDE_coin_toss_count_l3950_395002


namespace NUMINAMATH_CALUDE_m_range_isosceles_perimeter_l3950_395025

-- Define the triangle ABC
structure Triangle where
  AB : ℝ
  BC : ℝ
  AC : ℝ

-- Define the specific triangle from the problem
def triangleABC (m : ℝ) : Triangle where
  AB := 17
  BC := 8
  AC := 2 * m - 1

-- Theorem for the range of m
theorem m_range (m : ℝ) : 
  (∃ t : Triangle, t = triangleABC m ∧ t.AB + t.BC > t.AC ∧ t.AB + t.AC > t.BC ∧ t.AC + t.BC > t.AB) 
  ↔ (5 < m ∧ m < 13) := by sorry

-- Theorem for the perimeter when isosceles
theorem isosceles_perimeter (m : ℝ) :
  (∃ t : Triangle, t = triangleABC m ∧ (t.AB = t.AC ∨ t.AB = t.BC ∨ t.AC = t.BC)) →
  (∃ t : Triangle, t = triangleABC m ∧ t.AB + t.BC + t.AC = 42) := by sorry

end NUMINAMATH_CALUDE_m_range_isosceles_perimeter_l3950_395025


namespace NUMINAMATH_CALUDE_derivative_at_one_l3950_395030

/-- Given a differentiable function f: ℝ → ℝ where x > 0, 
    if f(x) = 2e^x * f'(1) + 3ln(x), then f'(1) = 3 / (1 - 2e) -/
theorem derivative_at_one (f : ℝ → ℝ) (hf : Differentiable ℝ f) 
    (h : ∀ x > 0, f x = 2 * Real.exp x * deriv f 1 + 3 * Real.log x) : 
  deriv f 1 = 3 / (1 - 2 * Real.exp 1) := by
  sorry

end NUMINAMATH_CALUDE_derivative_at_one_l3950_395030


namespace NUMINAMATH_CALUDE_simplify_radicals_l3950_395088

theorem simplify_radicals : Real.sqrt 18 - Real.sqrt 8 = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_simplify_radicals_l3950_395088


namespace NUMINAMATH_CALUDE_subset_union_theorem_l3950_395096

theorem subset_union_theorem (n : ℕ) (X : Finset ℕ) (m : ℕ) 
  (A : Fin m → Finset ℕ) :
  n > 6 →
  X.card = n →
  (∀ i : Fin m, (A i).card = 5) →
  (∀ i j : Fin m, i ≠ j → A i ≠ A j) →
  m > n * (n - 1) * (n - 2) * (n - 3) * (4 * n - 15) / 600 →
  ∃ (i₁ i₂ i₃ i₄ i₅ i₆ : Fin m), 
    i₁ < i₂ ∧ i₂ < i₃ ∧ i₃ < i₄ ∧ i₄ < i₅ ∧ i₅ < i₆ ∧
    (A i₁ ∪ A i₂ ∪ A i₃ ∪ A i₄ ∪ A i₅ ∪ A i₆) = X :=
by sorry

end NUMINAMATH_CALUDE_subset_union_theorem_l3950_395096
