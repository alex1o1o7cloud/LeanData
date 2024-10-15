import Mathlib

namespace NUMINAMATH_GPT_annual_increase_rate_l499_49901

theorem annual_increase_rate (r : ℝ) : 
  (6400 * (1 + r) * (1 + r) = 8100) → r = 0.125 :=
by sorry

end NUMINAMATH_GPT_annual_increase_rate_l499_49901


namespace NUMINAMATH_GPT_simplify_expression_l499_49994

theorem simplify_expression
  (h0 : (Real.pi / 2) < 2 ∧ 2 < Real.pi)  -- Given conditions on 2 related to π.
  (h1 : Real.sin 2 > 0)  -- Given condition that sin 2 is positive.
  (h2 : Real.cos 2 < 0)  -- Given condition that cos 2 is negative.
  : 2 * Real.sqrt (1 + Real.sin 4) + Real.sqrt (2 + 2 * Real.cos 4) = 2 * Real.sin 2 :=
sorry

end NUMINAMATH_GPT_simplify_expression_l499_49994


namespace NUMINAMATH_GPT_trig_identity_example_l499_49920

theorem trig_identity_example :
  256 * (Real.sin (10 * Real.pi / 180)) * (Real.sin (30 * Real.pi / 180)) *
    (Real.sin (50 * Real.pi / 180)) * (Real.sin (70 * Real.pi / 180)) = 16 := by
  sorry

end NUMINAMATH_GPT_trig_identity_example_l499_49920


namespace NUMINAMATH_GPT_remaining_surface_area_unchanged_l499_49972

noncomputable def original_cube_surface_area : Nat := 6 * 4 * 4

def corner_cube_surface_area : Nat := 3 * 2 * 2

def remaining_surface_area (original_cube_surface_area : Nat) (corner_cube_surface_area : Nat) : Nat :=
  original_cube_surface_area

theorem remaining_surface_area_unchanged :
  remaining_surface_area original_cube_surface_area corner_cube_surface_area = 96 := 
by
  sorry

end NUMINAMATH_GPT_remaining_surface_area_unchanged_l499_49972


namespace NUMINAMATH_GPT_simplify_fractions_l499_49953

theorem simplify_fractions :
  (240 / 18) * (6 / 135) * (9 / 4) = 4 / 3 :=
by
  sorry

end NUMINAMATH_GPT_simplify_fractions_l499_49953


namespace NUMINAMATH_GPT_fraction_inequality_l499_49967

theorem fraction_inequality 
  (a b x y : ℝ) 
  (ha : 0 < a) 
  (hb : 0 < b) 
  (hx : 0 < x) 
  (hy : 0 < y) 
  (h1 : 1 / a > 1 / b)
  (h2 : x > y) : 
  x / (x + a) > y / (y + b) := 
  sorry

end NUMINAMATH_GPT_fraction_inequality_l499_49967


namespace NUMINAMATH_GPT_binomial_problem_l499_49965

-- Definition of the binomial coefficient
def binomial (n k : ℕ) : ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

-- The problem statement: prove that binomial(13, 11) * 2 = 156
theorem binomial_problem : binomial 13 11 * 2 = 156 := by
  sorry

end NUMINAMATH_GPT_binomial_problem_l499_49965


namespace NUMINAMATH_GPT_Tamara_height_l499_49957

-- Define the conditions and goal as a theorem
theorem Tamara_height (K T : ℕ) (h1 : T = 3 * K - 4) (h2 : K + T = 92) : T = 68 :=
by
  sorry

end NUMINAMATH_GPT_Tamara_height_l499_49957


namespace NUMINAMATH_GPT_gcd_poly_multiple_l499_49933

theorem gcd_poly_multiple {x : ℤ} (h : ∃ k : ℤ, x = 54321 * k) :
  Int.gcd ((3 * x + 4) * (8 * x + 5) * (15 * x + 11) * (x + 14)) x = 1 :=
sorry

end NUMINAMATH_GPT_gcd_poly_multiple_l499_49933


namespace NUMINAMATH_GPT_f_at_one_is_zero_f_is_increasing_range_of_x_l499_49968

open Function

-- Define the conditions
variable {f : ℝ → ℝ}
variable (h1 : ∀ x > 1, f x > 0)
variable (h2 : ∀ x y, f (x * y) = f x + f y)

-- Problem Statements
theorem f_at_one_is_zero : f 1 = 0 := 
sorry

theorem f_is_increasing (x₁ x₂ : ℝ) (hx₁ : x₁ > 0) (hx₂ : x₂ > 0) (h : x₁ > x₂) : 
  f x₁ > f x₂ := 
sorry

theorem range_of_x (f3_eq_1 : f 3 = 1) (x : ℝ) (h3 : x ≥ 1 + Real.sqrt 10) : 
  f x - f (1 / (x - 2)) ≥ 2 := 
sorry

end NUMINAMATH_GPT_f_at_one_is_zero_f_is_increasing_range_of_x_l499_49968


namespace NUMINAMATH_GPT_probability_hardcover_liberal_arts_probability_liberal_arts_then_hardcover_l499_49922

-- Definitions based on the conditions provided
def total_books : ℕ := 100
def liberal_arts_books : ℕ := 40
def hardcover_books : ℕ := 70
def softcover_science_books : ℕ := 20
def hardcover_liberal_arts_books : ℕ := 30
def softcover_liberal_arts_books : ℕ := liberal_arts_books - hardcover_liberal_arts_books
def total_events_2 : ℕ := total_books * total_books

-- Statement part 1: Probability of selecting a hardcover liberal arts book
theorem probability_hardcover_liberal_arts :
  (hardcover_liberal_arts_books : ℝ) / total_books = 0.3 :=
sorry

-- Statement part 2: Probability of selecting a liberal arts book then a hardcover book (with replacement)
theorem probability_liberal_arts_then_hardcover :
  ((liberal_arts_books : ℝ) / total_books) * ((hardcover_books : ℝ) / total_books) = 0.28 :=
sorry

end NUMINAMATH_GPT_probability_hardcover_liberal_arts_probability_liberal_arts_then_hardcover_l499_49922


namespace NUMINAMATH_GPT_constant_term_binomial_expansion_l499_49981

theorem constant_term_binomial_expansion :
  ∃ (r : ℕ), (8 - 2 * r = 0) ∧ Nat.choose 8 r = 70 := by
  sorry

end NUMINAMATH_GPT_constant_term_binomial_expansion_l499_49981


namespace NUMINAMATH_GPT_car_clock_correctness_l499_49904

variables {t_watch t_car : ℕ} 
--  Variable declarations for time on watch (accurate) and time on car clock.

-- Define the initial times at 8:00 AM
def initial_time_watch : ℕ := 8 * 60 -- 8:00 AM in minutes
def initial_time_car : ℕ := 8 * 60 -- also 8:00 AM in minutes

-- Define the known times in the afternoon
def afternoon_time_watch : ℕ := 14 * 60 -- 2:00 PM in minutes
def afternoon_time_car : ℕ := 14 * 60 + 10 -- 2:10 PM in minutes

-- Car clock runs 37 minutes in the time the watch runs 36 minutes
def car_clock_rate : ℕ × ℕ := (37, 36)

-- Check the car clock time when the accurate watch shows 10:00 PM
def car_time_at_10pm_watch : ℕ := 22 * 60 -- 10:00 PM in minutes

-- Define the actual time that we need to prove
def actual_time_at_10pm_car : ℕ := 21 * 60 + 47 -- 9:47 PM in minutes

theorem car_clock_correctness : 
  (t_watch = actual_time_at_10pm_car) ↔ 
  (t_car = car_time_at_10pm_watch) ∧ 
  (initial_time_watch = initial_time_car) ∧ 
  (afternoon_time_watch = 14 * 60) ∧ 
  (afternoon_time_car = 14 * 60 + 10) ∧ 
  (car_clock_rate = (37, 36)) :=
sorry

end NUMINAMATH_GPT_car_clock_correctness_l499_49904


namespace NUMINAMATH_GPT_intersection_of_A_and_B_union_of_A_and_B_l499_49906

def A : Set ℝ := {x | x * (9 - x) > 0}
def B : Set ℝ := {x | x ≤ 3}

theorem intersection_of_A_and_B : A ∩ B = {x | 0 < x ∧ x ≤ 3} :=
sorry

theorem union_of_A_and_B : A ∪ B = {x | x < 9} :=
sorry

end NUMINAMATH_GPT_intersection_of_A_and_B_union_of_A_and_B_l499_49906


namespace NUMINAMATH_GPT_nina_max_digits_l499_49923

-- Define the conditions
def sam_digits (C : ℕ) := C + 6
def mina_digits := 24
def nina_digits (C : ℕ) := (7 * C) / 2

-- Define Carlos's digits and the sum condition
def carlos_digits := mina_digits / 6
def total_digits (C : ℕ) := C + sam_digits C + mina_digits + nina_digits C

-- Prove the maximum number of digits Nina could memorize
theorem nina_max_digits : ∀ C : ℕ, C = carlos_digits →
  total_digits C ≤ 100 → nina_digits C ≤ 62 :=
by
  intro C hC htotal
  sorry

end NUMINAMATH_GPT_nina_max_digits_l499_49923


namespace NUMINAMATH_GPT_find_n_minus_m_l499_49951

-- Define the circles
def circle1 (x y : ℝ) : Prop := x^2 + y^2 = 49
def circle2 (x y r : ℝ) : Prop := x^2 + y^2 - 6 * x - 8 * y + 25 - r^2 = 0

-- Given conditions
def circles_intersect (r : ℝ) : Prop :=
(r > 0) ∧ (∃ x y, circle1 x y ∧ circle2 x y r)

-- Prove the range of r for intersection
theorem find_n_minus_m : 
(∀ (r : ℝ), 2 ≤ r ∧ r ≤ 12 ↔ circles_intersect r) → 
12 - 2 = 10 :=
by
  sorry

end NUMINAMATH_GPT_find_n_minus_m_l499_49951


namespace NUMINAMATH_GPT_mark_new_phone_plan_cost_l499_49956

noncomputable def total_new_plan_cost (old_plan_cost old_internet_cost old_intl_call_cost : ℝ) (percent_increase_plan percent_increase_internet percent_decrease_intl : ℝ) : ℝ :=
  let new_plan_cost := old_plan_cost * (1 + percent_increase_plan)
  let new_internet_cost := old_internet_cost * (1 + percent_increase_internet)
  let new_intl_call_cost := old_intl_call_cost * (1 - percent_decrease_intl)
  new_plan_cost + new_internet_cost + new_intl_call_cost

theorem mark_new_phone_plan_cost :
  let old_plan_cost := 150
  let old_internet_cost := 50
  let old_intl_call_cost := 30
  let percent_increase_plan := 0.30
  let percent_increase_internet := 0.20
  let percent_decrease_intl := 0.15
  total_new_plan_cost old_plan_cost old_internet_cost old_intl_call_cost percent_increase_plan percent_increase_internet percent_decrease_intl = 280.50 :=
by
  sorry

end NUMINAMATH_GPT_mark_new_phone_plan_cost_l499_49956


namespace NUMINAMATH_GPT_part1_part2_part3_l499_49935

open Set

-- Define the sets A and B and the universal set
def A : Set ℝ := { x | 3 ≤ x ∧ x < 7 }
def B : Set ℝ := { x | 2 < x ∧ x < 10 }
def U : Set ℝ := univ  -- Universal set R

theorem part1 : A ∩ B = { x | 3 ≤ x ∧ x < 7 } :=
by { sorry }

theorem part2 : U \ A = { x | x < 3 ∨ x ≥ 7 } :=
by { sorry }

theorem part3 : U \ (A ∪ B) = { x | x ≤ 2 ∨ x ≥ 10 } :=
by { sorry }

end NUMINAMATH_GPT_part1_part2_part3_l499_49935


namespace NUMINAMATH_GPT_number_of_younger_employees_correct_l499_49996

noncomputable def total_employees : ℕ := 200
noncomputable def younger_employees : ℕ := 120
noncomputable def sample_size : ℕ := 25

def number_of_younger_employees_to_be_drawn (total younger sample : ℕ) : ℕ :=
  sample * younger / total

theorem number_of_younger_employees_correct :
  number_of_younger_employees_to_be_drawn total_employees younger_employees sample_size = 15 := by
  sorry

end NUMINAMATH_GPT_number_of_younger_employees_correct_l499_49996


namespace NUMINAMATH_GPT_no_adjacent_same_color_probability_zero_l499_49908

-- Define the number of each color bead
def num_red_beads : ℕ := 5
def num_white_beads : ℕ := 3
def num_blue_beads : ℕ := 2

-- Define the total number of beads
def total_beads : ℕ := num_red_beads + num_white_beads + num_blue_beads

-- Calculate the probability that no two neighboring beads are the same color
noncomputable def probability_no_adjacent_same_color : ℚ :=
  if (num_red_beads > num_white_beads + num_blue_beads + 1) then 0 else sorry

theorem no_adjacent_same_color_probability_zero :
  probability_no_adjacent_same_color = 0 :=
by {
  sorry
}

end NUMINAMATH_GPT_no_adjacent_same_color_probability_zero_l499_49908


namespace NUMINAMATH_GPT_moles_of_NaCl_formed_l499_49931

theorem moles_of_NaCl_formed (hcl moles : ℕ) (nahco3 moles : ℕ) (reaction : ℕ → ℕ → ℕ) :
  hcl = 3 → nahco3 = 3 → reaction 1 1 = 1 →
  reaction hcl nahco3 = 3 :=
by 
  intros h1 h2 h3
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_moles_of_NaCl_formed_l499_49931


namespace NUMINAMATH_GPT_toby_sharing_proof_l499_49926

theorem toby_sharing_proof (initial_amt amount_left num_brothers : ℕ) 
(h_init : initial_amt = 343)
(h_left : amount_left = 245)
(h_bros : num_brothers = 2) : 
(initial_amt - amount_left) / (initial_amt * num_brothers) = 1 / 7 := 
sorry

end NUMINAMATH_GPT_toby_sharing_proof_l499_49926


namespace NUMINAMATH_GPT_ducks_killed_is_20_l499_49958

variable (x : ℕ)

def killed_ducks_per_year (x : ℕ) : Prop :=
  let initial_flock := 100
  let annual_births := 30
  let years := 5
  let additional_flock := 150
  let final_flock := 300
  initial_flock + years * (annual_births - x) + additional_flock = final_flock

theorem ducks_killed_is_20 : killed_ducks_per_year 20 :=
by
  sorry

end NUMINAMATH_GPT_ducks_killed_is_20_l499_49958


namespace NUMINAMATH_GPT_highest_sum_vertex_l499_49936

theorem highest_sum_vertex (a b c d e f : ℕ) (h₀ : a + d = 8) (h₁ : b + e = 8) (h₂ : c + f = 8) : 
  a + b + c ≤ 11 ∧ b + c + d ≤ 11 ∧ c + d + e ≤ 11 ∧ d + e + f ≤ 11 ∧ e + f + a ≤ 11 ∧ f + a + b ≤ 11 :=
sorry

end NUMINAMATH_GPT_highest_sum_vertex_l499_49936


namespace NUMINAMATH_GPT_product_8_40_product_5_1_6_sum_6_instances_500_l499_49934

-- The product of 8 and 40 is 320
theorem product_8_40 : 8 * 40 = 320 := sorry

-- 5 times 1/6 is 5/6
theorem product_5_1_6 : 5 * (1 / 6) = 5 / 6 := sorry

-- The sum of 6 instances of 500 ends with 3 zeros and the sum is 3000
theorem sum_6_instances_500 :
  (500 * 6 = 3000) ∧ ((3000 % 1000) = 0) := sorry

end NUMINAMATH_GPT_product_8_40_product_5_1_6_sum_6_instances_500_l499_49934


namespace NUMINAMATH_GPT_function_even_periodic_l499_49982

theorem function_even_periodic (f : ℝ → ℝ) :
  (∀ x : ℝ, f (10 + x) = f (10 - x)) ∧ (∀ x : ℝ, f (5 - x) = f (5 + x)) →
  (∀ x : ℝ, f x = f (-x)) ∧ (∀ x : ℝ, f (x + 10) = f x) :=
by
  sorry

end NUMINAMATH_GPT_function_even_periodic_l499_49982


namespace NUMINAMATH_GPT_calculate_area_of_square_field_l499_49940

def area_of_square_field (t: ℕ) (v: ℕ) (d: ℕ) (s: ℕ) (a: ℕ) : Prop :=
  t = 10 ∧ v = 16 ∧ d = v * t ∧ 4 * s = d ∧ a = s^2

theorem calculate_area_of_square_field (t v d s a : ℕ) 
  (h1: t = 10) (h2: v = 16) (h3: d = v * t) (h4: 4 * s = d) 
  (h5: a = s^2) : a = 1600 := by
  sorry

end NUMINAMATH_GPT_calculate_area_of_square_field_l499_49940


namespace NUMINAMATH_GPT_sin_gt_cos_lt_nec_suff_l499_49959

-- Define the triangle and the angles
variables {A B C : ℝ}
variables (t : triangle A B C)

-- Define conditions in the triangle: sum of angles is 180 degrees
axiom angle_sum : A + B + C = 180

-- Define sin and cos using the sides of the triangle
noncomputable def sin_A (A : ℝ) : ℝ := sorry -- placeholder for actual definition
noncomputable def sin_B (B : ℝ) : ℝ := sorry
noncomputable def cos_A (A : ℝ) : ℝ := sorry
noncomputable def cos_B (B : ℝ) : ℝ := sorry

-- The proposition to prove
theorem sin_gt_cos_lt_nec_suff {A B : ℝ} (h1 : sin_A A > sin_B B) :
  cos_A A < cos_B B ↔ sin_A A > sin_B B := sorry

end NUMINAMATH_GPT_sin_gt_cos_lt_nec_suff_l499_49959


namespace NUMINAMATH_GPT_sqrt_expression_eval_l499_49993

theorem sqrt_expression_eval :
    (Real.sqrt 8 - 2 * Real.sqrt (1 / 2) + (2 - Real.sqrt 3) * (2 + Real.sqrt 3)) = Real.sqrt 2 + 1 := 
by
  sorry

end NUMINAMATH_GPT_sqrt_expression_eval_l499_49993


namespace NUMINAMATH_GPT_equivalent_problem_l499_49997

theorem equivalent_problem :
  let a : ℤ := (-6)
  let b : ℤ := 6
  let c : ℤ := 2
  let d : ℤ := 4
  (a^4 / b^2 - c^5 + d^2 = 20) :=
by
  sorry

end NUMINAMATH_GPT_equivalent_problem_l499_49997


namespace NUMINAMATH_GPT_sqrt_fraction_fact_l499_49990

noncomputable def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

theorem sqrt_fraction_fact :
  Real.sqrt (factorial 9 / 210 : ℝ) = 24 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_GPT_sqrt_fraction_fact_l499_49990


namespace NUMINAMATH_GPT_rounding_proof_l499_49909

def rounding_question : Prop :=
  let num := 9.996
  let rounded_value := ((num * 100).round / 100)
  rounded_value ≠ 10.00

theorem rounding_proof : rounding_question :=
by
  sorry

end NUMINAMATH_GPT_rounding_proof_l499_49909


namespace NUMINAMATH_GPT_find_C_l499_49944

theorem find_C (A B C : ℝ) (h1 : A + B + C = 500) (h2 : A + C = 200) (h3 : B + C = 330) : C = 30 := 
sorry

end NUMINAMATH_GPT_find_C_l499_49944


namespace NUMINAMATH_GPT_julia_played_tag_l499_49966

/-
Problem:
Let m be the number of kids Julia played with on Monday.
Let t be the number of kids Julia played with on Tuesday.
m = 24
m = t + 18
Show that t = 6
-/

theorem julia_played_tag (m t : ℕ) (h1 : m = 24) (h2 : m = t + 18) : t = 6 :=
by
  sorry

end NUMINAMATH_GPT_julia_played_tag_l499_49966


namespace NUMINAMATH_GPT_probability_of_prime_or_odd_is_half_l499_49971

-- Define the list of sections on the spinner
def sections : List ℕ := [3, 6, 1, 4, 8, 10, 2, 7]

-- Define a function to check if a number is prime
def is_prime (n : ℕ) : Bool :=
  if n < 2 then false else List.foldr (λ p b => b && (n % p ≠ 0)) true (List.range (n - 2) |>.map (λ x => x + 2))

-- Define a function to check if a number is odd
def is_odd (n : ℕ) : Bool := n % 2 ≠ 0

-- Define the condition of being either prime or odd
def is_prime_or_odd (n : ℕ) : Bool := is_prime n || is_odd n

-- List of favorable outcomes where the number is either prime or odd
def favorable_outcomes : List ℕ := sections.filter is_prime_or_odd

-- Calculate the probability
def probability_prime_or_odd : ℚ := (favorable_outcomes.length : ℚ) / (sections.length : ℚ)

-- Statement to prove the probability is 1/2
theorem probability_of_prime_or_odd_is_half : probability_prime_or_odd = 1 / 2 := by
  sorry

end NUMINAMATH_GPT_probability_of_prime_or_odd_is_half_l499_49971


namespace NUMINAMATH_GPT_ratio_of_values_l499_49949

-- Define the geometric sequence with first term and common ratio
def geom_seq_term (a₁ : ℚ) (r : ℚ) (n : ℕ) : ℚ := a₁ * r^(n-1)

-- Define the sum of the first n terms of the geometric sequence
def geom_seq_sum (a : ℚ) (r : ℚ) (n : ℕ) : ℚ :=
  if r = 1 then a * n else a * (1 - r^n) / (1 - r)

-- Sum of the first n terms for given sequence
noncomputable def S_n (n : ℕ) : ℚ :=
  geom_seq_sum (3/2) (-1/2) n

-- Define the function f(t) = t - 1/t
def f (t : ℚ) : ℚ := t - 1 / t

-- Define the maximum and minimum values of f(S_n) and their ratio
noncomputable def ratio_max_min_values : ℚ :=
  let max_val := f (3/2)
  let min_val := f (3/4)
  max_val / min_val

-- The theorem to prove the ratio of the maximum and minimum values
theorem ratio_of_values :
  ratio_max_min_values = -10/7 := by
  sorry

end NUMINAMATH_GPT_ratio_of_values_l499_49949


namespace NUMINAMATH_GPT_eval_floor_ceil_sum_l499_49902

noncomputable def floor (x : ℝ) : ℤ := Int.floor x
noncomputable def ceil (x : ℝ) : ℤ := Int.ceil x

theorem eval_floor_ceil_sum : floor (-3.67) + ceil 34.7 = 31 := by
  sorry

end NUMINAMATH_GPT_eval_floor_ceil_sum_l499_49902


namespace NUMINAMATH_GPT_units_digit_of_5_pow_150_plus_7_l499_49980

theorem units_digit_of_5_pow_150_plus_7 : (5^150 + 7) % 10 = 2 := by
  sorry

end NUMINAMATH_GPT_units_digit_of_5_pow_150_plus_7_l499_49980


namespace NUMINAMATH_GPT_min_value_of_g_inequality_f_l499_49991

def f (x m : ℝ) : ℝ := abs (x - m)
def g (x m : ℝ) : ℝ := 2 * f x m - f (x + m) m

theorem min_value_of_g (m : ℝ) (hm : m > 0) (h : ∀ x, g x m ≥ -1) : m = 1 :=
sorry

theorem inequality_f {m a b : ℝ} (hm : m > 0) (ha : abs a < m) (hb : abs b < m) (h0 : a ≠ 0) :
  f (a * b) m > abs a * f (b / a) m :=
sorry

end NUMINAMATH_GPT_min_value_of_g_inequality_f_l499_49991


namespace NUMINAMATH_GPT_ratio_e_f_l499_49910

theorem ratio_e_f (a b c d e f : ℚ)
  (h1 : a / b = 1 / 3)
  (h2 : b / c = 2)
  (h3 : c / d = 1 / 2)
  (h4 : d / e = 3)
  (h5 : a * b * c / (d * e * f) = 0.25) :
  e / f = 9 / 4 :=
sorry

end NUMINAMATH_GPT_ratio_e_f_l499_49910


namespace NUMINAMATH_GPT_fruit_basket_count_l499_49976

theorem fruit_basket_count :
  let apples := 6
  let oranges := 8
  let min_apples := 2
  let min_fruits := 1
  (0 <= oranges ∧ oranges <= 8) ∧ (min_apples <= apples ∧ apples <= 6) ∧ (min_fruits <= (apples + oranges)) →
  (5 * 9 = 45) :=
by
  intro h
  sorry

end NUMINAMATH_GPT_fruit_basket_count_l499_49976


namespace NUMINAMATH_GPT_geometric_sequence_second_term_l499_49939

theorem geometric_sequence_second_term (a_1 q a_3 a_4 : ℝ) (h3 : a_1 * q^2 = 12) (h4 : a_1 * q^3 = 18) : a_1 * q = 8 :=
by
  sorry

end NUMINAMATH_GPT_geometric_sequence_second_term_l499_49939


namespace NUMINAMATH_GPT_man_walking_speed_l499_49977

-- This statement introduces the assumptions and goals of the proof problem.
theorem man_walking_speed
  (x : ℝ)
  (h1 : (25 * (1 / 12)) = (x * (1 / 3)))
  : x = 6.25 :=
sorry

end NUMINAMATH_GPT_man_walking_speed_l499_49977


namespace NUMINAMATH_GPT_bank_policy_advantageous_for_retirees_l499_49969

theorem bank_policy_advantageous_for_retirees
  (special_programs : Prop)
  (higher_deposit_rates : Prop)
  (lower_credit_rates : Prop)
  (reliable_loan_payers : Prop)
  (stable_income : Prop)
  (family_interest : Prop)
  (savings_tendency : Prop)
  (regular_income : Prop)
  (long_term_deposits : Prop) :
  reliable_loan_payers ∧ stable_income ∧ family_interest ∧ savings_tendency ∧ regular_income ∧ long_term_deposits → 
  special_programs ∧ higher_deposit_rates ∧ lower_credit_rates :=
sorry

end NUMINAMATH_GPT_bank_policy_advantageous_for_retirees_l499_49969


namespace NUMINAMATH_GPT_part1_solution_set_k_3_part2_solution_set_k_lt_0_l499_49999

open Set

-- Definitions
def inequality (k : ℝ) (x : ℝ) : Prop :=
  k * x^2 + (k - 2) * x - 2 < 0

-- Part 1: When k = 3
theorem part1_solution_set_k_3 : ∀ x : ℝ, inequality 3 x ↔ -1 < x ∧ x < (2 / 3) :=
by
  sorry

-- Part 2: When k < 0
theorem part2_solution_set_k_lt_0 :
  ∀ k : ℝ, k < 0 → 
    (k = -2 → ∀ x : ℝ, inequality k x ↔ x ≠ -1) ∧
    (k < -2 → ∀ x : ℝ, inequality k x ↔ x < -1 ∨ x > 2 / k) ∧
    (-2 < k → ∀ x : ℝ, inequality k x ↔ x > -1 ∨ x < 2 / k) :=
by
  sorry

end NUMINAMATH_GPT_part1_solution_set_k_3_part2_solution_set_k_lt_0_l499_49999


namespace NUMINAMATH_GPT_area_of_sheet_is_correct_l499_49916

noncomputable def area_of_rolled_sheet (length width height thickness : ℝ) : ℝ :=
  (length * width * height) / thickness

theorem area_of_sheet_is_correct :
  area_of_rolled_sheet 80 20 5 0.1 = 80000 :=
by
  -- The proof is omitted (sorry).
  sorry

end NUMINAMATH_GPT_area_of_sheet_is_correct_l499_49916


namespace NUMINAMATH_GPT_second_tap_emptying_time_l499_49903

theorem second_tap_emptying_time :
  ∀ (T : ℝ), (∀ (f e : ℝ),
  (f = 1 / 3) →
  (∀ (n : ℝ), (n = 1 / 4.5) →
  (n = f - e ↔ e = 1 / T))) →
  T = 9 :=
by
  sorry

end NUMINAMATH_GPT_second_tap_emptying_time_l499_49903


namespace NUMINAMATH_GPT_arccos_one_eq_zero_l499_49978

theorem arccos_one_eq_zero :
  Real.arccos 1 = 0 :=
sorry

end NUMINAMATH_GPT_arccos_one_eq_zero_l499_49978


namespace NUMINAMATH_GPT_exact_two_solutions_l499_49948

theorem exact_two_solutions (a : ℝ) : 
  (∃! x : ℝ, x^2 + 2*x + 2*|x+1| = a) ↔ a > -1 :=
sorry

end NUMINAMATH_GPT_exact_two_solutions_l499_49948


namespace NUMINAMATH_GPT_antiderivative_correct_l499_49974

def f (x : ℝ) : ℝ := 2 * x
def F (x : ℝ) : ℝ := x^2 + 2

theorem antiderivative_correct :
  (∀ x, f x = deriv (F) x) ∧ (F 1 = 3) :=
by
  sorry

end NUMINAMATH_GPT_antiderivative_correct_l499_49974


namespace NUMINAMATH_GPT_problem_statements_correct_l499_49913

theorem problem_statements_correct :
    (∀ (select : ℕ) (male female : ℕ), male = 4 → female = 3 → 
      (select = (4 * 3 + 3)) → select ≥ 12 = false) ∧
    (∀ (a1 a2 a3 : ℕ), 
      a2 = 0 ∨ a2 = 1 ∨ a2 = 2 →
      (∃ (cases : ℕ), cases = 14) →
      cases = 14) ∧
    (∀ (ways enter exit : ℕ), enter = 4 → exit = 4 - 1 →
      (ways = enter * exit) → ways = 12 = false) ∧
    (∀ (a b : ℕ),
      a > 0 ∧ a < 10 ∧ b > 0 ∧ b < 10 →
      (∃ (log_val : ℕ), log_val = 54) →
      log_val = 54) := by
  admit

end NUMINAMATH_GPT_problem_statements_correct_l499_49913


namespace NUMINAMATH_GPT_triangle_count_l499_49938

theorem triangle_count (a b c : ℕ) (h1 : a + b + c = 15) (h2 : a ≤ b) (h3 : b ≤ c) (h4 : a + b > c) :
  ∃ (n : ℕ), n = 7 :=
by
  -- Proceed with the proof steps, using a, b, c satisfying the given conditions
  sorry

end NUMINAMATH_GPT_triangle_count_l499_49938


namespace NUMINAMATH_GPT_part1_part2_l499_49915

-- Define the sets P and Q
def P (a : ℝ) : Set ℝ := {x | a + 1 ≤ x ∧ x ≤ 2 * a + 1}
def Q : Set ℝ := {x | -2 ≤ x ∧ x ≤ 5}

-- Part (1)
theorem part1 (a : ℝ) (h : a = 3) : (P 3)ᶜ ∩ Q = {x | -2 ≤ x ∧ x < 4} :=
by
  sorry

-- Part (2)
theorem part2 (a : ℝ) : (∀ x, x ∈ P a → x ∈ Q) ∧ (∃ x, x ∈ Q ∧ x ∉ P a) → 0 ≤ a ∧ a ≤ 2 :=
by
  sorry

end NUMINAMATH_GPT_part1_part2_l499_49915


namespace NUMINAMATH_GPT_teresa_class_size_l499_49960

theorem teresa_class_size :
  ∃ (a : ℤ), 50 < a ∧ a < 100 ∧ 
  (a % 3 = 2) ∧ 
  (a % 4 = 2) ∧ 
  (a % 5 = 2) ∧ 
  a = 62 := 
by {
  sorry
}

end NUMINAMATH_GPT_teresa_class_size_l499_49960


namespace NUMINAMATH_GPT_mean_of_remaining_two_l499_49970

theorem mean_of_remaining_two (a b c d e : ℝ) (h : (a + b + c = 3 * 2010)) : 
  (a + b + c + d + e) / 5 = 2010 → (d + e) / 2 = 2011.5 :=
by
  sorry 

end NUMINAMATH_GPT_mean_of_remaining_two_l499_49970


namespace NUMINAMATH_GPT_workerB_time_to_complete_job_l499_49929

theorem workerB_time_to_complete_job 
  (time_A : ℝ) (time_together: ℝ) (time_B : ℝ) 
  (h1 : time_A = 5) 
  (h2 : time_together = 3.333333333333333) 
  (h3 : 1 / time_A + 1 / time_B = 1 / time_together) 
  : time_B = 10 := 
  sorry

end NUMINAMATH_GPT_workerB_time_to_complete_job_l499_49929


namespace NUMINAMATH_GPT_total_flour_used_l499_49928

def wheat_flour : ℝ := 0.2
def white_flour : ℝ := 0.1

theorem total_flour_used : wheat_flour + white_flour = 0.3 :=
by
  sorry

end NUMINAMATH_GPT_total_flour_used_l499_49928


namespace NUMINAMATH_GPT_topsoil_cost_l499_49964

theorem topsoil_cost (cost_per_cubic_foot : ℝ) (cubic_yards : ℝ) (conversion_factor : ℝ) : 
  cubic_yards = 8 →
  cost_per_cubic_foot = 7 →
  conversion_factor = 27 →
  ∃ total_cost : ℝ, total_cost = 1512 :=
by
  intros h1 h2 h3
  sorry

end NUMINAMATH_GPT_topsoil_cost_l499_49964


namespace NUMINAMATH_GPT_simplify_expression_l499_49924

variable {R : Type*} [Field R]

theorem simplify_expression (x y z : R) (h₁ : x ≠ 0) (h₂ : y ≠ 0) (h₃ : z ≠ 0) :
  (x + y + z)⁻¹ * (x⁻¹ + y⁻¹ + z⁻¹) = x⁻¹ * y⁻¹ * z⁻¹ :=
sorry

end NUMINAMATH_GPT_simplify_expression_l499_49924


namespace NUMINAMATH_GPT_fibonacci_150_mod_9_l499_49937

def fibonacci (n : ℕ) : ℕ :=
  if h : n < 2 then n else fibonacci (n - 1) + fibonacci (n - 2)

theorem fibonacci_150_mod_9 : fibonacci 150 % 9 = 8 :=
  sorry

end NUMINAMATH_GPT_fibonacci_150_mod_9_l499_49937


namespace NUMINAMATH_GPT_intersection_of_lines_l499_49907

theorem intersection_of_lines :
  ∃ x y : ℚ, (12 * x - 3 * y = 33) ∧ (8 * x + 2 * y = 18) ∧ (x = 29 / 12 ∧ y = -2 / 3) :=
by {
  sorry
}

end NUMINAMATH_GPT_intersection_of_lines_l499_49907


namespace NUMINAMATH_GPT_marquita_gardens_l499_49911

open Nat

theorem marquita_gardens (num_mancino_gardens : ℕ) 
  (length_mancino_garden width_mancino_garden : ℕ) 
  (num_marquita_gardens : ℕ) 
  (length_marquita_garden width_marquita_garden : ℕ)
  (total_area : ℕ) 
  (h1 : num_mancino_gardens = 3)
  (h2 : length_mancino_garden = 16)
  (h3 : width_mancino_garden = 5)
  (h4 : length_marquita_garden = 8)
  (h5 : width_marquita_garden = 4)
  (h6 : total_area = 304)
  (hmancino_area : num_mancino_gardens * (length_mancino_garden * width_mancino_garden) = 3 * (16 * 5))
  (hcombined_area : total_area = num_mancino_gardens * (length_mancino_garden * width_mancino_garden) + num_marquita_gardens * (length_marquita_garden * width_marquita_garden)) :
  num_marquita_gardens = 2 :=
sorry

end NUMINAMATH_GPT_marquita_gardens_l499_49911


namespace NUMINAMATH_GPT_squares_end_with_76_l499_49975

noncomputable def validNumbers : List ℕ := [24, 26, 74, 76]

theorem squares_end_with_76 (x : ℕ) (h₁ : x % 10 = 4 ∨ x % 10 = 6) 
    (h₂ : (x * x) % 100 = 76) : x ∈ validNumbers := by
  sorry

end NUMINAMATH_GPT_squares_end_with_76_l499_49975


namespace NUMINAMATH_GPT_total_oysters_eaten_l499_49983

/-- Squido eats 200 oysters -/
def Squido_eats := 200

/-- Crabby eats at least twice as many oysters as Squido -/
def Crabby_eats := 2 * Squido_eats

/-- Total oysters eaten by Squido and Crabby -/
theorem total_oysters_eaten : Squido_eats + Crabby_eats = 600 := 
by
  sorry

end NUMINAMATH_GPT_total_oysters_eaten_l499_49983


namespace NUMINAMATH_GPT_avg_root_area_avg_volume_correlation_coefficient_total_volume_estimate_l499_49905

open Real
open List

-- Conditions
def x_vals : List ℝ := [0.04, 0.06, 0.04, 0.08, 0.08, 0.05, 0.05, 0.07, 0.07, 0.06]
def y_vals : List ℝ := [0.25, 0.40, 0.22, 0.54, 0.51, 0.34, 0.36, 0.46, 0.42, 0.40]
def sum_x : ℝ := 0.6
def sum_y : ℝ := 3.9
def sum_x_squared : ℝ := 0.038
def sum_y_squared : ℝ := 1.6158
def sum_xy : ℝ := 0.2474
def total_root_area : ℝ := 186

-- Proof problems
theorem avg_root_area : (List.sum x_vals / 10) = 0.06 := by
  sorry

theorem avg_volume : (List.sum y_vals / 10) = 0.39 := by
  sorry

theorem correlation_coefficient : 
  let mean_x := List.sum x_vals / 10;
  let mean_y := List.sum y_vals / 10;
  let numerator := List.sum (List.zipWith (λ x y => (x - mean_x) * (y - mean_y)) x_vals y_vals);
  let denominator := sqrt ((List.sum (List.map (λ x => (x - mean_x) ^ 2) x_vals)) * (List.sum (List.map (λ y => (y - mean_y) ^ 2) y_vals)));
  (numerator / denominator) = 0.97 := by 
  sorry

theorem total_volume_estimate : 
  let avg_x := sum_x / 10;
  let avg_y := sum_y / 10;
  (avg_y / avg_x) * total_root_area = 1209 := by
  sorry

end NUMINAMATH_GPT_avg_root_area_avg_volume_correlation_coefficient_total_volume_estimate_l499_49905


namespace NUMINAMATH_GPT_max_positive_integer_value_of_n_l499_49952

-- Define the arithmetic sequence with common difference d and first term a₁.
variable {d a₁ : ℝ}

-- The quadratic inequality condition which provides the solution set [0,9].
def inequality_condition (d a₁ : ℝ) : Prop :=
  ∀ (x : ℝ), (0 ≤ x ∧ x ≤ 9) → d * x^2 + 2 * a₁ * x ≥ 0

-- Maximum integer n such that the sum of the first n terms of the sequence is maximum.
noncomputable def max_n (d a₁ : ℝ) : ℕ :=
  if d < 0 then 5 else 0

-- Statement to be proved.
theorem max_positive_integer_value_of_n (d a₁ : ℝ) 
  (h : inequality_condition d a₁) : max_n d a₁ = 5 :=
sorry

end NUMINAMATH_GPT_max_positive_integer_value_of_n_l499_49952


namespace NUMINAMATH_GPT_sum_of_m_integers_l499_49900

theorem sum_of_m_integers :
  ∀ (m : ℤ), 
    (∀ (x : ℚ), (x - 10) / 5 ≤ -1 - x / 5 ∧ x - 1 > -m / 2) → 
    (∃ x_max x_min : ℤ, x_max + x_min = -2 ∧ 
                        (x_max ≤ 5 / 2 ∧ x_min ≤ 5 / 2) ∧ 
                        (1 - m / 2 < x_min ∧ 1 - m / 2 < x_max)) →
  (10 < m ∧ m ≤ 12) → m = 11 ∨ m = 12 → 11 + 12 = 23 :=
by sorry

end NUMINAMATH_GPT_sum_of_m_integers_l499_49900


namespace NUMINAMATH_GPT_pool_width_40_l499_49943

theorem pool_width_40
  (hose_rate : ℕ)
  (pool_length : ℕ)
  (pool_depth : ℕ)
  (pool_capacity_percent : ℚ)
  (drain_time : ℕ)
  (water_drained : ℕ)
  (total_capacity : ℚ)
  (pool_width : ℚ) :
  hose_rate = 60 ∧
  pool_length = 150 ∧
  pool_depth = 10 ∧
  pool_capacity_percent = 0.8 ∧
  drain_time = 800 ∧
  water_drained = hose_rate * drain_time ∧
  total_capacity = water_drained / pool_capacity_percent ∧
  total_capacity = pool_length * pool_width * pool_depth →
  pool_width = 40 :=
by
  sorry

end NUMINAMATH_GPT_pool_width_40_l499_49943


namespace NUMINAMATH_GPT_dilute_lotion_l499_49942

/-- Determine the number of ounces of water needed to dilute 12 ounces
    of a shaving lotion containing 60% alcohol to a lotion containing 45% alcohol. -/
theorem dilute_lotion (W : ℝ) : 
  ∃ W, 12 * (0.60 : ℝ) / (12 + W) = 0.45 ∧ W = 4 :=
by
  use 4
  sorry

end NUMINAMATH_GPT_dilute_lotion_l499_49942


namespace NUMINAMATH_GPT_find_f_2016_minus_f_2015_l499_49925

-- Definitions for the given conditions

def odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = -f (x)

def periodic_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (x + 2) = -f x

def specific_values (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, (0 < x ∧ x ≤ 1) → f x = 2^x

-- Main theorem statement
theorem find_f_2016_minus_f_2015 {f : ℝ → ℝ} 
    (H1 : odd_function f) 
    (H2 : periodic_function f)
    (H3 : specific_values f)
    : f 2016 - f 2015 = 2 := 
sorry

end NUMINAMATH_GPT_find_f_2016_minus_f_2015_l499_49925


namespace NUMINAMATH_GPT_arithmetic_sequence_zero_l499_49986

noncomputable def f (x : ℝ) : ℝ :=
  0.3 ^ x - Real.log x / Real.log 2

theorem arithmetic_sequence_zero (a b c x : ℝ) (h_seq : a < b ∧ b < c) (h_pos_diff : b - a = c - b)
    (h_f_product : f a * f b * f c > 0) (h_fx_zero : f x = 0) : ¬ (x < a) :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_zero_l499_49986


namespace NUMINAMATH_GPT_train_length_l499_49932

noncomputable def length_of_train (time_in_seconds : ℝ) (speed_in_kmh : ℝ) : ℝ :=
  let speed_in_mps := speed_in_kmh * (5 / 18)
  speed_in_mps * time_in_seconds

theorem train_length :
  length_of_train 2.3998080153587713 210 = 140 :=
by
  sorry

end NUMINAMATH_GPT_train_length_l499_49932


namespace NUMINAMATH_GPT_tree_height_relationship_l499_49914

theorem tree_height_relationship (x : ℕ) : ∃ h : ℕ, h = 80 + 2 * x :=
by
  sorry

end NUMINAMATH_GPT_tree_height_relationship_l499_49914


namespace NUMINAMATH_GPT_cubic_with_root_p_sq_l499_49950

theorem cubic_with_root_p_sq (p : ℝ) (hp : p^3 + p - 3 = 0) : (p^2 : ℝ) ^ 3 + 2 * (p^2) ^ 2 + p^2 - 9 = 0 :=
sorry

end NUMINAMATH_GPT_cubic_with_root_p_sq_l499_49950


namespace NUMINAMATH_GPT_fern_pays_228_11_usd_l499_49988

open Real

noncomputable def high_heels_price : ℝ := 66
noncomputable def ballet_slippers_price : ℝ := (2 / 3) * high_heels_price
noncomputable def purse_price : ℝ := 49.5
noncomputable def scarf_price : ℝ := 27.5
noncomputable def high_heels_discount : ℝ := 0.10 * high_heels_price
noncomputable def discounted_high_heels_price : ℝ := high_heels_price - high_heels_discount
noncomputable def total_cost_before_tax : ℝ := discounted_high_heels_price + ballet_slippers_price + purse_price + scarf_price
noncomputable def sales_tax : ℝ := 0.075 * total_cost_before_tax
noncomputable def total_cost_after_tax : ℝ := total_cost_before_tax + sales_tax
noncomputable def exchange_rate : ℝ := 1 / 0.85
noncomputable def total_cost_in_usd : ℝ := total_cost_after_tax * exchange_rate

theorem fern_pays_228_11_usd: total_cost_in_usd = 228.11 := by
  sorry

end NUMINAMATH_GPT_fern_pays_228_11_usd_l499_49988


namespace NUMINAMATH_GPT_math_proof_problem_l499_49962

-- Definitions
def PropA : Prop := ¬ (∀ n : ℤ, (3 ∣ n → ¬ (n % 2 = 1)))
def PropB : Prop := ¬ (¬ (∃ x : ℝ, x^2 + x + 1 ≥ 0))
def PropC : Prop := ∀ (α β : ℝ) (k : ℤ), α = k * Real.pi + β ↔ Real.tan α = Real.tan β
def PropD : Prop := ∀ (a b : ℝ), a ≠ 0 → a * b ≠ 0 → b ≠ 0

def correct_options : Prop := PropA ∧ PropC ∧ ¬PropB ∧ PropD

-- The theorem to be proven
theorem math_proof_problem : correct_options :=
by
  sorry

end NUMINAMATH_GPT_math_proof_problem_l499_49962


namespace NUMINAMATH_GPT_negation_of_universal_sin_pos_l499_49917

theorem negation_of_universal_sin_pos :
  ¬ (∀ x : ℝ, Real.sin x > 0) ↔ ∃ x : ℝ, Real.sin x ≤ 0 :=
by sorry

end NUMINAMATH_GPT_negation_of_universal_sin_pos_l499_49917


namespace NUMINAMATH_GPT_m_value_for_power_function_l499_49995

theorem m_value_for_power_function (m : ℝ) :
  (3 * m - 1 = 1) → (m = 2 / 3) :=
by
  sorry

end NUMINAMATH_GPT_m_value_for_power_function_l499_49995


namespace NUMINAMATH_GPT_conic_section_is_parabola_l499_49992

def isParabola (equation : String) : Prop := 
  equation = "|y - 3| = sqrt((x + 4)^2 + (y - 1)^2)"

theorem conic_section_is_parabola : isParabola "|y - 3| = sqrt((x + 4)^2 + (y - 1)^2)" :=
  by
  sorry

end NUMINAMATH_GPT_conic_section_is_parabola_l499_49992


namespace NUMINAMATH_GPT_ratio_B_to_A_l499_49985

theorem ratio_B_to_A (A B S : ℕ) 
  (h1 : A = 2 * S)
  (h2 : A = 80)
  (h3 : B - S = 200) :
  B / A = 3 :=
by sorry

end NUMINAMATH_GPT_ratio_B_to_A_l499_49985


namespace NUMINAMATH_GPT_sum_of_two_digit_and_reverse_l499_49919

theorem sum_of_two_digit_and_reverse (a b : ℕ) (h1 : 1 ≤ a) (h2 : a ≤ 9) (h3 : 0 ≤ b) (h4 : b ≤ 9)
  (h5 : (10 * a + b) - (10 * b + a) = 9 * (a + b)) : (10 * a + b) + (10 * b + a) = 11 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_two_digit_and_reverse_l499_49919


namespace NUMINAMATH_GPT_boat_speed_in_still_water_equals_6_l499_49984

def river_flow_rate : ℝ := 2
def distance_upstream : ℝ := 40
def distance_downstream : ℝ := 40
def total_time : ℝ := 15

theorem boat_speed_in_still_water_equals_6 :
  ∃ b : ℝ, (40 / (b - river_flow_rate) + 40 / (b + river_flow_rate) = total_time) ∧ b = 6 :=
sorry

end NUMINAMATH_GPT_boat_speed_in_still_water_equals_6_l499_49984


namespace NUMINAMATH_GPT_finite_integer_solutions_l499_49961

theorem finite_integer_solutions (n : ℕ) : 
  ∃ (S : Finset (ℤ × ℤ)), ∀ (x y : ℤ), (x^3 + y^3 = n) → (x, y) ∈ S := 
sorry

end NUMINAMATH_GPT_finite_integer_solutions_l499_49961


namespace NUMINAMATH_GPT_sum_first_five_terms_arith_seq_l499_49927

theorem sum_first_five_terms_arith_seq (a : ℕ → ℤ)
  (h4 : a 4 = 3) (h5 : a 5 = 7) (h6 : a 6 = 11) :
  a 1 + a 2 + a 3 + a 4 + a 5 = -5 :=
by
  sorry

end NUMINAMATH_GPT_sum_first_five_terms_arith_seq_l499_49927


namespace NUMINAMATH_GPT_calculate_fraction_l499_49918

theorem calculate_fraction :
  (18 - 6) / ((3 + 3) * 2) = 1 := by
  sorry

end NUMINAMATH_GPT_calculate_fraction_l499_49918


namespace NUMINAMATH_GPT_num_ways_to_pay_16_rubles_l499_49912

theorem num_ways_to_pay_16_rubles :
  ∃! (n : ℕ), n = 13 ∧ ∀ (x y z : ℕ), (x ≥ 0) ∧ (y ≥ 0) ∧ (z ≥ 0) ∧ 
  (10 * x + 2 * y + 1 * z = 16) ∧ (x < 2) ∧ (y + z > 0) := sorry

end NUMINAMATH_GPT_num_ways_to_pay_16_rubles_l499_49912


namespace NUMINAMATH_GPT_list_price_l499_49954

theorem list_price (P : ℝ) (h₀ : 0.83817 * P = 56.16) : P = 67 :=
sorry

end NUMINAMATH_GPT_list_price_l499_49954


namespace NUMINAMATH_GPT_count_valid_three_digit_numbers_l499_49963

theorem count_valid_three_digit_numbers : 
  ∃ n : ℕ, n = 36 ∧ 
    (∀ (a b c : ℕ), a ≠ 0 ∧ c ≠ 0 → 
    ((10 * b + c) % 4 = 0 ∧ (10 * b + a) % 4 = 0) → 
    n = 36) :=
sorry

end NUMINAMATH_GPT_count_valid_three_digit_numbers_l499_49963


namespace NUMINAMATH_GPT_average_speed_of_bus_l499_49998

theorem average_speed_of_bus (speed_bicycle : ℝ)
  (start_distance : ℝ) (catch_up_time : ℝ)
  (h1 : speed_bicycle = 15)
  (h2 : start_distance = 195)
  (h3 : catch_up_time = 3) : 
  (start_distance + speed_bicycle * catch_up_time) / catch_up_time = 80 :=
by
  sorry

end NUMINAMATH_GPT_average_speed_of_bus_l499_49998


namespace NUMINAMATH_GPT_least_subtract_divisible_by_10_least_subtract_divisible_by_100_least_subtract_divisible_by_1000_l499_49930

-- The numbers involved and the requirements described
def num : ℕ := 427398

def least_to_subtract_10 : ℕ := 8
def least_to_subtract_100 : ℕ := 98
def least_to_subtract_1000 : ℕ := 398

-- Proving the conditions:
-- 1. (num - least_to_subtract_10) is divisible by 10
-- 2. (num - least_to_subtract_100) is divisible by 100
-- 3. (num - least_to_subtract_1000) is divisible by 1000

theorem least_subtract_divisible_by_10 : (num - least_to_subtract_10) % 10 = 0 := 
by 
  sorry

theorem least_subtract_divisible_by_100 : (num - least_to_subtract_100) % 100 = 0 := 
by 
  sorry

theorem least_subtract_divisible_by_1000 : (num - least_to_subtract_1000) % 1000 = 0 := 
by 
  sorry

end NUMINAMATH_GPT_least_subtract_divisible_by_10_least_subtract_divisible_by_100_least_subtract_divisible_by_1000_l499_49930


namespace NUMINAMATH_GPT_arithmetic_progression_sum_l499_49945

variable {α : Type*} [LinearOrderedField α]

def arithmetic_progression (S : ℕ → α) :=
  ∃ (a d : α), ∀ n, S n = (n * (2 * a + (n - 1) * d)) / 2

theorem arithmetic_progression_sum :
  ∀ (S : ℕ → α),
  arithmetic_progression S →
  (S 4) / (S 8) = 1 / 7 →
  (S 12) / (S 4) = 43 :=
by
  intros S h_arith_prog h_ratio
  sorry

end NUMINAMATH_GPT_arithmetic_progression_sum_l499_49945


namespace NUMINAMATH_GPT_quadratic_has_real_roots_range_l499_49973

-- Lean 4 statement

theorem quadratic_has_real_roots_range (m : ℝ) :
  (∀ x : ℝ, (m - 3) * x^2 + 4 * x + 1 = 0) → m ≤ 7 ∧ m ≠ 3 :=
by
  sorry

end NUMINAMATH_GPT_quadratic_has_real_roots_range_l499_49973


namespace NUMINAMATH_GPT_marbles_difference_l499_49979

def lostMarbles : ℕ := 8
def foundMarbles : ℕ := 10

theorem marbles_difference (lostMarbles foundMarbles : ℕ) : foundMarbles - lostMarbles = 2 := 
by
  sorry

end NUMINAMATH_GPT_marbles_difference_l499_49979


namespace NUMINAMATH_GPT_one_cow_one_bag_l499_49955

theorem one_cow_one_bag (h : 50 * 1 * 50 = 50 * 50) : 50 = 50 :=
by
  sorry

end NUMINAMATH_GPT_one_cow_one_bag_l499_49955


namespace NUMINAMATH_GPT_face_value_of_shares_l499_49921

-- Define the problem conditions
variables (F : ℝ) (D R : ℝ)

-- Assume conditions
axiom h1 : D = 0.155 * F
axiom h2 : R = 0.25 * 31
axiom h3 : D = R

-- State the theorem
theorem face_value_of_shares : F = 50 :=
by 
  -- Here should be the proof which we are skipping
  sorry

end NUMINAMATH_GPT_face_value_of_shares_l499_49921


namespace NUMINAMATH_GPT_jonah_fishes_per_day_l499_49987

theorem jonah_fishes_per_day (J G J_total : ℕ) (days : ℕ) (total : ℕ)
  (hJ : J = 6) (hG : G = 8) (hdays : days = 5) (htotal : total = 90) 
  (fish_total : days * J + days * G + days * J_total = total) : 
  J_total = 4 :=
by
  sorry

end NUMINAMATH_GPT_jonah_fishes_per_day_l499_49987


namespace NUMINAMATH_GPT_cookies_on_ninth_plate_l499_49941

-- Define the geometric sequence
def cookies_on_plate (n : ℕ) : ℕ :=
  2 * 2^(n - 1)

-- State the theorem
theorem cookies_on_ninth_plate : cookies_on_plate 9 = 512 :=
by
  sorry

end NUMINAMATH_GPT_cookies_on_ninth_plate_l499_49941


namespace NUMINAMATH_GPT_exponent_value_l499_49946

theorem exponent_value (exponent : ℕ) (y: ℕ) :
  (12 ^ exponent) * (6 ^ 4) / 432 = y → y = 36 → exponent = 1 :=
by
  intro h1 h2
  sorry

end NUMINAMATH_GPT_exponent_value_l499_49946


namespace NUMINAMATH_GPT_sofie_total_distance_l499_49947

-- Definitions for the conditions
def side1 : ℝ := 25
def side2 : ℝ := 35
def side3 : ℝ := 20
def side4 : ℝ := 40
def side5 : ℝ := 30
def laps_initial : ℕ := 2
def laps_additional : ℕ := 5
def perimeter : ℝ := side1 + side2 + side3 + side4 + side5

-- Theorem statement
theorem sofie_total_distance : laps_initial * perimeter + laps_additional * perimeter = 1050 := by
  sorry

end NUMINAMATH_GPT_sofie_total_distance_l499_49947


namespace NUMINAMATH_GPT_proposition_negation_l499_49989

theorem proposition_negation (p : Prop) : 
  (∃ x : ℝ, x < 1 ∧ x^2 < 1) ↔ (∀ x : ℝ, x < 1 → x^2 ≥ 1) :=
sorry

end NUMINAMATH_GPT_proposition_negation_l499_49989
