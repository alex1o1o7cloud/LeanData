import Mathlib

namespace NUMINAMATH_GPT_number_of_integers_in_double_inequality_l451_45179

noncomputable def pi_approx : ℝ := 3.14
noncomputable def sqrt_pi_approx : ℝ := Real.sqrt pi_approx
noncomputable def lower_bound : ℝ := -12 * sqrt_pi_approx
noncomputable def upper_bound : ℝ := 15 * pi_approx

theorem number_of_integers_in_double_inequality : 
  ∃ n : ℕ, n = 13 ∧ ∀ k : ℤ, lower_bound ≤ (k^2 : ℝ) ∧ (k^2 : ℝ) ≤ upper_bound → (-6 ≤ k ∧ k ≤ 6) :=
by
  sorry

end NUMINAMATH_GPT_number_of_integers_in_double_inequality_l451_45179


namespace NUMINAMATH_GPT_lindsay_dolls_problem_l451_45105

theorem lindsay_dolls_problem :
  let blonde_dolls := 6
  let brown_dolls := 3 * blonde_dolls
  let black_dolls := brown_dolls / 2
  let red_dolls := 2 * black_dolls
  let combined_dolls := black_dolls + brown_dolls + red_dolls
  combined_dolls - blonde_dolls = 39 :=
by
  sorry

end NUMINAMATH_GPT_lindsay_dolls_problem_l451_45105


namespace NUMINAMATH_GPT_alice_and_bob_pies_l451_45185

theorem alice_and_bob_pies (T : ℝ) : (T / 5 = T / 6 + 2) → T = 60 := by
  sorry

end NUMINAMATH_GPT_alice_and_bob_pies_l451_45185


namespace NUMINAMATH_GPT_value_of_each_other_toy_l451_45121

-- Definitions for the conditions
def total_toys : ℕ := 9
def total_worth : ℕ := 52
def single_toy_value : ℕ := 12

-- Definition to represent the value of each of the other toys
def other_toys_value (same_value : ℕ) : Prop :=
  (total_worth - single_toy_value) / (total_toys - 1) = same_value

-- The theorem to be proven
theorem value_of_each_other_toy : other_toys_value 5 :=
  sorry

end NUMINAMATH_GPT_value_of_each_other_toy_l451_45121


namespace NUMINAMATH_GPT_find_constants_PQR_l451_45144

theorem find_constants_PQR :
  ∃ P Q R : ℚ, 
    (P = (-8 / 15)) ∧ 
    (Q = (-7 / 6)) ∧ 
    (R = (27 / 10)) ∧
    (∀ x : ℚ, 
      (x - 1) ≠ 0 ∧ (x - 4) ≠ 0 ∧ (x - 6) ≠ 0 →
      (x^2 - 9) / ((x - 1) * (x - 4) * (x - 6)) = 
      P / (x - 1) + Q / (x - 4) + R / (x - 6)) :=
by
  sorry

end NUMINAMATH_GPT_find_constants_PQR_l451_45144


namespace NUMINAMATH_GPT_lunch_people_count_l451_45159

theorem lunch_people_count
  (C : ℝ)   -- total lunch cost including gratuity
  (G : ℝ)   -- gratuity rate
  (P : ℝ)   -- average price per person excluding gratuity
  (n : ℕ)   -- number of people
  (h1 : C = 207.0)  -- condition: total cost with gratuity
  (h2 : G = 0.15)   -- condition: gratuity rate of 15%
  (h3 : P = 12.0)   -- condition: average price per person
  (h4 : C = (1 + G) * n * P) -- condition: total cost with gratuity is (1 + gratuity rate) * number of people * average price per person
  : n = 15 :=       -- conclusion: number of people
sorry

end NUMINAMATH_GPT_lunch_people_count_l451_45159


namespace NUMINAMATH_GPT_find_area_of_oblique_triangle_l451_45192

noncomputable def area_triangle (a b c : ℝ) (A B C : ℝ) : ℝ :=
  1 / 2 * a * b * Real.sin C

theorem find_area_of_oblique_triangle
  (A B C a b c : ℝ)
  (h1 : c = Real.sqrt 21)
  (h2 : c * Real.sin A = Real.sqrt 3 * a * Real.cos C)
  (h3 : Real.sin C + Real.sin (B - A) = 5 * Real.sin (2 * A))
  (hA : 0 < A) (hB : 0 < B) (hC : 0 < C) (h_sum_ABC : A + B + C = Real.pi)
  (habc_pos : 0 < a ∧ 0 < b ∧ 0 < c)
  (tri_angle_pos : 0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = Real.pi) :
  area_triangle a b c A B C = 5 * Real.sqrt 3 / 4 := 
sorry

end NUMINAMATH_GPT_find_area_of_oblique_triangle_l451_45192


namespace NUMINAMATH_GPT_young_or_old_woman_lawyer_probability_l451_45183

/-- 
40 percent of the members of a study group are women.
Among these women, 30 percent are young lawyers.
10 percent are old lawyers.
Prove the probability that a member randomly selected is a young or old woman lawyer is 0.16.
-/
theorem young_or_old_woman_lawyer_probability :
  let total_members := 100
  let women_percentage := 40
  let young_lawyers_percentage := 30
  let old_lawyers_percentage := 10
  let total_women := (women_percentage * total_members) / 100
  let young_women_lawyers := (young_lawyers_percentage * total_women) / 100
  let old_women_lawyers := (old_lawyers_percentage * total_women) / 100
  let women_lawyers := young_women_lawyers + old_women_lawyers
  let probability := women_lawyers / total_members
  probability = 0.16 := 
by {
  sorry
}

end NUMINAMATH_GPT_young_or_old_woman_lawyer_probability_l451_45183


namespace NUMINAMATH_GPT_calculate_fraction_square_mul_l451_45187

theorem calculate_fraction_square_mul :
  ((8 / 9) ^ 2) * ((1 / 3) ^ 2) = 64 / 729 :=
by
  sorry

end NUMINAMATH_GPT_calculate_fraction_square_mul_l451_45187


namespace NUMINAMATH_GPT_largest_area_polygons_l451_45157

-- Define the area of each polygon
def area_P := 4
def area_Q := 6
def area_R := 3 + 3 * (1 / 2)
def area_S := 6 * (1 / 2)
def area_T := 5 + 2 * (1 / 2)

-- Proof of the polygons with the largest area
theorem largest_area_polygons : (area_Q = 6 ∧ area_T = 6) ∧ area_Q ≥ area_P ∧ area_Q ≥ area_R ∧ area_Q ≥ area_S :=
by
  sorry

end NUMINAMATH_GPT_largest_area_polygons_l451_45157


namespace NUMINAMATH_GPT_puppies_count_l451_45104

theorem puppies_count 
  (dogs : ℕ := 3)
  (dog_meal_weight : ℕ := 4)
  (dog_meals_per_day : ℕ := 3)
  (total_food : ℕ := 108)
  (puppy_meal_multiplier : ℕ := 2)
  (puppy_meal_frequency_multiplier : ℕ := 3) :
  ∃ (puppies : ℕ), puppies = 4 :=
by
  let dog_daily_food := dog_meal_weight * dog_meals_per_day
  let puppy_meal_weight := dog_meal_weight / puppy_meal_multiplier
  let puppy_daily_food := puppy_meal_weight * puppy_meal_frequency_multiplier * dog_meals_per_day
  let total_dog_food := dogs * dog_daily_food
  let total_puppy_food := total_food - total_dog_food
  let puppies := total_puppy_food / puppy_daily_food
  use puppies
  have h_puppies_correct : puppies = 4 := sorry
  exact h_puppies_correct

end NUMINAMATH_GPT_puppies_count_l451_45104


namespace NUMINAMATH_GPT_find_m_l451_45152

variables (m : ℝ)
def a : ℝ × ℝ := (2, -1)
def b : ℝ × ℝ := (-1, m)
def c : ℝ × ℝ := (-1, 2)

-- Define the property of vector parallelism in ℝ.
def parallel (u v : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, u.1 = k * v.1 ∧ u.2 = k * v.2

-- Statement to be proven
theorem find_m :
    parallel (1, m - 1) c →
    m = -1 :=
by
  sorry

end NUMINAMATH_GPT_find_m_l451_45152


namespace NUMINAMATH_GPT_find_common_ratio_l451_45141

variable {a : ℕ → ℝ}

def is_geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n, a (n + 1) = a n * q

theorem find_common_ratio (a1 a4 q : ℝ) (hq : q ^ 3 = 8) (ha1 : a1 = 8) (ha4 : a4 = 64)
  (a_def : is_geometric_sequence (fun n => a1 * q ^ n) q) :
  q = 2 :=
by
  sorry

end NUMINAMATH_GPT_find_common_ratio_l451_45141


namespace NUMINAMATH_GPT_inequality_solution_ab_l451_45115

theorem inequality_solution_ab (a b : ℝ) (h : ∀ x : ℝ, 2 < x ∧ x < 4 ↔ |x + a| < b) : a * b = -3 := 
by
  sorry

end NUMINAMATH_GPT_inequality_solution_ab_l451_45115


namespace NUMINAMATH_GPT_blue_balls_count_l451_45103

theorem blue_balls_count (Y B : ℕ) (h_ratio : 4 * B = 3 * Y) (h_total : Y + B = 35) : B = 15 :=
sorry

end NUMINAMATH_GPT_blue_balls_count_l451_45103


namespace NUMINAMATH_GPT_positive_difference_16_l451_45135

def avg_is_37 (y : ℤ) : Prop := (45 + y) / 2 = 37

def positive_difference (a b : ℤ) : ℤ := if a > b then a - b else b - a

theorem positive_difference_16 (y : ℤ) (h : avg_is_37 y) : positive_difference 45 y = 16 :=
by
  sorry

end NUMINAMATH_GPT_positive_difference_16_l451_45135


namespace NUMINAMATH_GPT_decrease_hours_by_13_percent_l451_45167

theorem decrease_hours_by_13_percent (W H : ℝ) (hW_pos : W > 0) (hH_pos : H > 0) :
  let W_new := 1.15 * W
  let H_new := H / 1.15
  let income_decrease_percentage := (1 - H_new / H) * 100
  abs (income_decrease_percentage - 13.04) < 0.01 := 
by
  sorry

end NUMINAMATH_GPT_decrease_hours_by_13_percent_l451_45167


namespace NUMINAMATH_GPT_min_distance_racetracks_l451_45147

theorem min_distance_racetracks : 
  ∀ A B : ℝ × ℝ, (A.1 ^ 2 + A.2 ^ 2 = 1) ∧ (((B.1 - 1) ^ 2) / 16 + (B.2 ^ 2) / 4 = 1) → 
  dist A B ≥ (Real.sqrt 33 - 3) / 3 := by
  sorry

end NUMINAMATH_GPT_min_distance_racetracks_l451_45147


namespace NUMINAMATH_GPT_intersection_eq_l451_45146

def P : Set ℕ := {1, 2, 3, 4}
def Q : Set ℕ := {x | x ≤ 2}

theorem intersection_eq : P ∩ Q = {1, 2} :=
by
  sorry

end NUMINAMATH_GPT_intersection_eq_l451_45146


namespace NUMINAMATH_GPT_speed_of_car_first_hour_98_l451_45126

def car_speed_in_first_hour_is_98 (x : ℕ) : Prop :=
  (70 + x) / 2 = 84 → x = 98

theorem speed_of_car_first_hour_98 (x : ℕ) (h : car_speed_in_first_hour_is_98 x) : x = 98 :=
  by
  sorry

end NUMINAMATH_GPT_speed_of_car_first_hour_98_l451_45126


namespace NUMINAMATH_GPT_inequality_solution_l451_45117

theorem inequality_solution (a : ℝ) (h : a^2 > 2 * a - 1) : a ≠ 1 := 
sorry

end NUMINAMATH_GPT_inequality_solution_l451_45117


namespace NUMINAMATH_GPT_inequality_negative_solution_l451_45148

theorem inequality_negative_solution (a : ℝ) (h : a ≥ -17/4 ∧ a < 4) : 
  ∃ x : ℝ, x < 0 ∧ x^2 < 4 - |x - a| :=
by
  sorry

end NUMINAMATH_GPT_inequality_negative_solution_l451_45148


namespace NUMINAMATH_GPT_remainder_of_sum_modulo_9_l451_45160

theorem remainder_of_sum_modulo_9 : 
  (8230 + 8231 + 8232 + 8233 + 8234 + 8235) % 9 = 0 := by
  sorry

end NUMINAMATH_GPT_remainder_of_sum_modulo_9_l451_45160


namespace NUMINAMATH_GPT_exponent_equation_l451_45196

theorem exponent_equation (n : ℕ) (h : 8^4 = 16^n) : n = 3 :=
by sorry

end NUMINAMATH_GPT_exponent_equation_l451_45196


namespace NUMINAMATH_GPT_minimum_a_l451_45173

theorem minimum_a (a : ℝ) : 
  (∀ x : ℝ, 0 < x ∧ x ≤ 1/2 → x^2 + a * x + 1 ≥ 0) → 
  a ≥ -5/2 :=
sorry

end NUMINAMATH_GPT_minimum_a_l451_45173


namespace NUMINAMATH_GPT_annual_income_of_A_l451_45178

variable (Cm : ℝ)
variable (Bm : ℝ)
variable (Am : ℝ)
variable (Aa : ℝ)

-- Given conditions
axiom h1 : Cm = 12000
axiom h2 : Bm = Cm + 0.12 * Cm
axiom h3 : (Am / Bm) = 5 / 2

-- Statement to prove
theorem annual_income_of_A : Aa = 403200 := by
  sorry

end NUMINAMATH_GPT_annual_income_of_A_l451_45178


namespace NUMINAMATH_GPT_cost_to_open_store_l451_45191

-- Define the conditions as constants
def revenue_per_month : ℕ := 4000
def expenses_per_month : ℕ := 1500
def months_to_payback : ℕ := 10

-- Theorem stating the cost to open the store
theorem cost_to_open_store : (revenue_per_month - expenses_per_month) * months_to_payback = 25000 :=
by
  sorry

end NUMINAMATH_GPT_cost_to_open_store_l451_45191


namespace NUMINAMATH_GPT_probability_of_at_most_one_white_ball_l451_45110

open Nat

-- Definitions based on conditions in a)
def black_balls : ℕ := 10
def red_balls : ℕ := 12
def white_balls : ℕ := 3
def total_balls : ℕ := black_balls + red_balls + white_balls
def select_balls : ℕ := 3

-- The combinatorial function C(n, k) as defined in combinatorics
def C (n k : ℕ) : ℕ := Nat.choose n k

-- Defining the expression and correct answer
def expr : ℚ := (C white_balls 1 * C (black_balls + red_balls) 2 + C (black_balls + red_balls) 3 : ℚ) / (C total_balls 3 : ℚ)
def correct_answer : ℚ := (C white_balls 0 * C (black_balls + red_balls) 3 + C white_balls 1 * C (black_balls + red_balls) 2 : ℚ) / (C total_balls 3 : ℚ)

-- Lean 4 theorem statement
theorem probability_of_at_most_one_white_ball :
  expr = correct_answer := sorry

end NUMINAMATH_GPT_probability_of_at_most_one_white_ball_l451_45110


namespace NUMINAMATH_GPT_second_number_value_l451_45133

theorem second_number_value 
  (a b c : ℚ)
  (h1 : a + b + c = 120)
  (h2 : a / b = 3 / 4)
  (h3 : b / c = 2 / 5) :
  b = 480 / 17 :=
by
  sorry

end NUMINAMATH_GPT_second_number_value_l451_45133


namespace NUMINAMATH_GPT_cube_difference_l451_45171

theorem cube_difference (a b : ℝ) (h1 : a - b = 7) (h2 : a^2 + b^2 = 53) : a^3 - b^3 = 385 :=
sorry

end NUMINAMATH_GPT_cube_difference_l451_45171


namespace NUMINAMATH_GPT_max_value_ineq_l451_45172

variables {R : Type} [LinearOrderedField R]

theorem max_value_ineq (a b c x y z : R) 
  (h1 : 0 ≤ a) (h2 : 0 ≤ b) (h3 : 0 ≤ c)
  (h4 : 0 ≤ x) (h5 : 0 ≤ y) (h6 : 0 ≤ z)
  (h7 : a + b + c = 1) (h8 : x + y + z = 1) :
  (a - x^2) * (b - y^2) * (c - z^2) ≤ 1 / 16 :=
sorry

end NUMINAMATH_GPT_max_value_ineq_l451_45172


namespace NUMINAMATH_GPT_square_of_complex_l451_45131

def z : Complex := 5 - 2 * Complex.I

theorem square_of_complex : z^2 = 21 - 20 * Complex.I := by
  sorry

end NUMINAMATH_GPT_square_of_complex_l451_45131


namespace NUMINAMATH_GPT_orthogonal_projection_l451_45181

noncomputable def proj (u v : ℝ × ℝ) : ℝ × ℝ := 
  let dot_uv := u.1 * v.1 + u.2 * v.2
  let norm_u_squared := u.1 * u.1 + u.2 * u.2
  (dot_uv / norm_u_squared * u.1, dot_uv / norm_u_squared * u.2)

theorem orthogonal_projection
  (a b : ℝ × ℝ)
  (h_orth : a.1 * b.1 + a.2 * b.2 = 0)
  (h_proj_a : proj a (4, -4) = (-4/5, -8/5)) :
  proj b (4, -4) = (24/5, -12/5) :=
sorry

end NUMINAMATH_GPT_orthogonal_projection_l451_45181


namespace NUMINAMATH_GPT_second_person_more_heads_probability_l451_45111

noncomputable def coin_flip_probability (n m : ℕ) : ℚ :=
  if n < m then 1 / 2 else 0

theorem second_person_more_heads_probability :
  coin_flip_probability 10 11 = 1 / 2 :=
by
  sorry

end NUMINAMATH_GPT_second_person_more_heads_probability_l451_45111


namespace NUMINAMATH_GPT_isosceles_triangle_side_length_condition_l451_45100

theorem isosceles_triangle_side_length_condition (x y : ℕ) :
    y = x + 1 ∧ 2 * x + y = 16 → (y = 6 → x = 5) :=
by sorry

end NUMINAMATH_GPT_isosceles_triangle_side_length_condition_l451_45100


namespace NUMINAMATH_GPT_product_of_even_and_odd_is_odd_l451_45140

def even_function (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x
def odd_function (g : ℝ → ℝ) : Prop := ∀ x, g (-x) = -g x
def odd_product (f g : ℝ → ℝ) : Prop := ∀ x, (f x) * (g x) = - (f x) * (g x)
 
theorem product_of_even_and_odd_is_odd 
  (f : ℝ → ℝ) (g : ℝ → ℝ) 
  (h1 : even_function f) 
  (h2 : odd_function g) : odd_product f g :=
by
  sorry

end NUMINAMATH_GPT_product_of_even_and_odd_is_odd_l451_45140


namespace NUMINAMATH_GPT_sum_of_coefficients_l451_45166

theorem sum_of_coefficients (a_0 a_1 a_2 a_3 a_4 a_5 a_6 a_7 a_8 a_9 : ℝ) :
  (1 - 2 * x)^9 = a_0 + a_1 * x + a_2 * x^2 + a_3 * x^3 + a_4 * x^4 + a_5 * x^5 + 
                  a_6 * x^6 + a_7 * x^7 + a_8 * x^8 + a_9 * x^9 →
  a_0 + a_1 + a_2 + a_3 + a_4 + a_5 + a_6 + a_7 + a_8 + a_9 = -1 :=
sorry

end NUMINAMATH_GPT_sum_of_coefficients_l451_45166


namespace NUMINAMATH_GPT_minimum_value_of_f_l451_45118

noncomputable def f (x : ℝ) : ℝ := 1 / (x^2) + 2 * x

theorem minimum_value_of_f (h : ∀ x > 0, f x ≥ 3) : ∃ x, x > 0 ∧ f x = 3 :=
by
  sorry

end NUMINAMATH_GPT_minimum_value_of_f_l451_45118


namespace NUMINAMATH_GPT_no_solution_fraction_equation_l451_45142

theorem no_solution_fraction_equation (x : ℝ) (h : x ≠ 2) : 
  (1 - x) / (x - 2) + 2 = 1 / (2 - x) → false :=
by 
  intro h_eq
  sorry

end NUMINAMATH_GPT_no_solution_fraction_equation_l451_45142


namespace NUMINAMATH_GPT_slowerPainterDuration_l451_45164

def slowerPainterStartTime : ℝ := 14 -- 2:00 PM in 24-hour format
def fasterPainterStartTime : ℝ := slowerPainterStartTime + 3 -- 3 hours later
def finishTime : ℝ := 24.6 -- 0.6 hours past midnight

theorem slowerPainterDuration :
  finishTime - slowerPainterStartTime = 10.6 :=
by
  sorry

end NUMINAMATH_GPT_slowerPainterDuration_l451_45164


namespace NUMINAMATH_GPT_IncorrectStatement_l451_45153

-- Definitions of the events
def EventA (planeShot : ℕ → Prop) : Prop := planeShot 1 ∧ planeShot 2
def EventB (planeShot : ℕ → Prop) : Prop := ¬planeShot 1 ∧ ¬planeShot 2
def EventC (planeShot : ℕ → Prop) : Prop := (planeShot 1 ∧ ¬planeShot 2) ∨ (¬planeShot 1 ∧ planeShot 2)
def EventD (planeShot : ℕ → Prop) : Prop := planeShot 1 ∨ planeShot 2

-- Theorem statement to be proved (negation of the incorrect statement)
theorem IncorrectStatement (planeShot : ℕ → Prop) :
  ¬((EventA planeShot ∨ EventC planeShot) = (EventB planeShot ∨ EventD planeShot)) :=
by
  sorry

end NUMINAMATH_GPT_IncorrectStatement_l451_45153


namespace NUMINAMATH_GPT_quotient_correct_l451_45158

def dividend : ℤ := 474232
def divisor : ℤ := 800
def remainder : ℤ := -968

theorem quotient_correct : (dividend + abs remainder) / divisor = 594 := by
  sorry

end NUMINAMATH_GPT_quotient_correct_l451_45158


namespace NUMINAMATH_GPT_triangle_perimeter_correct_l451_45129

noncomputable def triangle_perimeter (a b c : ℕ) : ℕ :=
    a + b + c

theorem triangle_perimeter_correct (a b c : ℕ) (h1 : a = b - 1) (h2 : b = c - 1) (h3 : c = 2 * a) : triangle_perimeter a b c = 15 :=
    sorry

end NUMINAMATH_GPT_triangle_perimeter_correct_l451_45129


namespace NUMINAMATH_GPT_determine_chris_age_l451_45128

theorem determine_chris_age (a b c : ℚ)
  (h1 : (a + b + c) / 3 = 10)
  (h2 : c - 5 = 2 * a)
  (h3 : b + 4 = (3 / 4) * (a + 4)) :
  c = 283 / 15 :=
by
  sorry

end NUMINAMATH_GPT_determine_chris_age_l451_45128


namespace NUMINAMATH_GPT_arccos_range_l451_45168

theorem arccos_range (a : ℝ) (x : ℝ) (h₀ : x = Real.sin a) 
  (h₁ : -Real.pi / 4 ≤ a ∧ a ≤ 3 * Real.pi / 4) :
  ∀ y, y = Real.arccos x → 0 ≤ y ∧ y ≤ 3 * Real.pi / 4 := 
sorry

end NUMINAMATH_GPT_arccos_range_l451_45168


namespace NUMINAMATH_GPT_solution_set_for_f_l451_45113

noncomputable def f (x : ℝ) : ℝ :=
if x ≥ 0 then x^2 + x else -x^2 + x

theorem solution_set_for_f (x : ℝ) :
  f (x^2 - x + 1) < 12 ↔ -1 < x ∧ x < 2 :=
sorry

end NUMINAMATH_GPT_solution_set_for_f_l451_45113


namespace NUMINAMATH_GPT_find_n_tan_eq_348_l451_45145

theorem find_n_tan_eq_348 (n : ℤ) (h1 : -90 < n) (h2 : n < 90) : 
  (Real.tan (n * Real.pi / 180) = Real.tan (348 * Real.pi / 180)) ↔ (n = -12) := by
  sorry

end NUMINAMATH_GPT_find_n_tan_eq_348_l451_45145


namespace NUMINAMATH_GPT_roots_ratio_sum_l451_45176

theorem roots_ratio_sum (a b m : ℝ) 
  (m1 m2 : ℝ)
  (h_roots : a ≠ b ∧ b ≠ 0 ∧ m ≠ 0 ∧ a ≠ 0 ∧ 
    ∀ x : ℝ, m * (x^2 - 3 * x) + 2 * x + 7 = 0 → (x = a ∨ x = b)) 
  (h_ratio : (a / b) + (b / a) = 7 / 3)
  (h_m1_m2_eq : ((3 * m - 2) ^ 2) / (7 * m) - 2 = 7 / 3)
  (h_m_vieta : (3 * m - 2) ^ 2 - 27 * m * (91 / 3) = 0) :
  (m1 + m2 = 127 / 27) ∧ (m1 * m2 = 4 / 9) →
  ((m1 / m2) + (m2 / m1) = 47.78) :=
sorry

end NUMINAMATH_GPT_roots_ratio_sum_l451_45176


namespace NUMINAMATH_GPT_solve_system_eqs_l451_45190
noncomputable section

theorem solve_system_eqs (x y z : ℝ) :
  (x * y = 5 * (x + y) ∧ x * z = 4 * (x + z) ∧ y * z = 2 * (y + z))
  ↔ (x = 0 ∧ y = 0 ∧ z = 0)
  ∨ (x = -40 ∧ y = 40 / 9 ∧ z = 40 / 11) := sorry

end NUMINAMATH_GPT_solve_system_eqs_l451_45190


namespace NUMINAMATH_GPT_sum_seven_consecutive_integers_l451_45122

theorem sum_seven_consecutive_integers (n : ℕ) : 
  ∃ k : ℕ, (n + (n + 1) + (n + 2) + (n + 3) + (n + 4) + (n + 5) + (n + 6)) = 7 * k := 
by 
  -- Use sum of integers and factor to demonstrate that the sum is multiple of 7
  sorry

end NUMINAMATH_GPT_sum_seven_consecutive_integers_l451_45122


namespace NUMINAMATH_GPT_find_c_if_quadratic_lt_zero_l451_45134

theorem find_c_if_quadratic_lt_zero (c : ℝ) : 
  (∀ x : ℝ, -x^2 + c * x - 12 < 0 ↔ (x < 2 ∨ x > 7)) → c = 9 := 
by
  sorry

end NUMINAMATH_GPT_find_c_if_quadratic_lt_zero_l451_45134


namespace NUMINAMATH_GPT_mean_of_squares_of_first_four_odd_numbers_l451_45108

theorem mean_of_squares_of_first_four_odd_numbers :
  (1^2 + 3^2 + 5^2 + 7^2) / 4 = 21 := 
by
  sorry

end NUMINAMATH_GPT_mean_of_squares_of_first_four_odd_numbers_l451_45108


namespace NUMINAMATH_GPT_min_value_ineq_solve_ineq_l451_45107

theorem min_value_ineq (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (1 / a^3 + 1 / b^3 + 1 / c^3 + 3 * a * b * c) ≥ 6 :=
sorry

theorem solve_ineq (x : ℝ) (h : |x + 1| - 2 * x < 6) : x > -7/3 :=
sorry

end NUMINAMATH_GPT_min_value_ineq_solve_ineq_l451_45107


namespace NUMINAMATH_GPT_abc_plus_2_gt_a_plus_b_plus_c_l451_45180

theorem abc_plus_2_gt_a_plus_b_plus_c (a b c : ℝ) (h1 : |a| < 1) (h2 : |b| < 1) (h3 : |c| < 1) : abc + 2 > a + b + c :=
by
  sorry

end NUMINAMATH_GPT_abc_plus_2_gt_a_plus_b_plus_c_l451_45180


namespace NUMINAMATH_GPT_inequality_proof_l451_45120

theorem inequality_proof (f : ℝ → ℝ) (a b : ℝ) :
  (∀ x : ℝ, f x = x^2 + 3 * x + 2) →
  a > 0 →
  b > 0 →
  b ≤ a / 7 →
  (∀ x : ℝ, |x + 2| < b → |f x + 4| < a) :=
by
  sorry

end NUMINAMATH_GPT_inequality_proof_l451_45120


namespace NUMINAMATH_GPT_find_FC_l451_45127

-- Define all given values and relationships
variables (DC CB AD AB ED FC : ℝ)
variables (h1 : DC = 9) (h2 : CB = 6)
variables (h3 : AB = (1/3) * AD)
variables (h4 : ED = (2/3) * AD)

-- Define the goal
theorem find_FC :
  FC = 9 :=
sorry

end NUMINAMATH_GPT_find_FC_l451_45127


namespace NUMINAMATH_GPT_div_trans_l451_45124

variable {a b c : ℝ}

theorem div_trans :
  a / b = 3 → b / c = 5 / 2 → c / a = 2 / 15 :=
  by
  intro h1 h2
  sorry

end NUMINAMATH_GPT_div_trans_l451_45124


namespace NUMINAMATH_GPT_find_k_of_division_property_l451_45169

theorem find_k_of_division_property (k : ℝ) :
  (3 * (1 / 3)^3 - k * (1 / 3)^2 + 4) % (3 * (1 / 3) - 1) = 5 → k = -8 :=
by sorry

end NUMINAMATH_GPT_find_k_of_division_property_l451_45169


namespace NUMINAMATH_GPT_find_c_value_l451_45162

theorem find_c_value (c : ℝ)
  (h : 4 * (3.6 * 0.48 * c / (0.12 * 0.09 * 0.5)) = 3200.0000000000005) :
  c = 2.5 :=
by sorry

end NUMINAMATH_GPT_find_c_value_l451_45162


namespace NUMINAMATH_GPT_sequence_value_x_l451_45154

theorem sequence_value_x (x : ℕ) (h1 : 1 + 3 = 4) (h2 : 4 + 3 = 7) (h3 : 7 + 3 = 10) (h4 : 10 + 3 = x) (h5 : x + 3 = 16) : x = 13 := by
  sorry

end NUMINAMATH_GPT_sequence_value_x_l451_45154


namespace NUMINAMATH_GPT_triangle_median_perpendicular_l451_45112

theorem triangle_median_perpendicular (x1 y1 x2 y2 x3 y3 : ℝ) 
  (h1 : (x1 - (x2 + x3) / 2) * (x2 - (x1 + x3) / 2) + (y1 - (y2 + y3) / 2) * (y2 - (y1 + y3) / 2) = 0)
  (h2 : (x2 - x3) ^ 2 + (y2 - y3) ^ 2 = 64)
  (h3 : (x1 - x3) ^ 2 + (y1 - y3) ^ 2 = 25) : 
  (x1 - x2) ^ 2 + (y1 - y2) ^ 2 = 22.25 := sorry

end NUMINAMATH_GPT_triangle_median_perpendicular_l451_45112


namespace NUMINAMATH_GPT_find_a3_in_arith_geo_seq_l451_45175

theorem find_a3_in_arith_geo_seq
  (a : ℕ → ℚ)
  (S : ℕ → ℚ)
  (h1 : S 6 / S 3 = -19 / 8)
  (h2 : a 4 - a 2 = -15 / 8) :
  a 3 = 9 / 4 :=
sorry

end NUMINAMATH_GPT_find_a3_in_arith_geo_seq_l451_45175


namespace NUMINAMATH_GPT_average_age_of_team_l451_45116

theorem average_age_of_team (A : ℝ) : 
    (11 * A =
         9 * (A - 1) + 53) → 
    A = 31 := 
by 
  sorry

end NUMINAMATH_GPT_average_age_of_team_l451_45116


namespace NUMINAMATH_GPT_prob_is_correct_l451_45109

def total_balls : ℕ := 500
def white_balls : ℕ := 200
def green_balls : ℕ := 100
def yellow_balls : ℕ := 70
def blue_balls : ℕ := 50
def red_balls : ℕ := 30
def purple_balls : ℕ := 20
def orange_balls : ℕ := 30

noncomputable def probability_green_yellow_blue : ℚ :=
  (green_balls + yellow_balls + blue_balls) / total_balls

theorem prob_is_correct :
  probability_green_yellow_blue = 0.44 := 
  by
  sorry

end NUMINAMATH_GPT_prob_is_correct_l451_45109


namespace NUMINAMATH_GPT_k_at_1_value_l451_45149

def h (x p : ℝ) := x^3 + p * x^2 + 2 * x + 20
def k (x p q r : ℝ) := x^4 + 2 * x^3 + q * x^2 + 50 * x + r

theorem k_at_1_value (p q r : ℝ) (h_distinct_roots : ∀ x₁ x₂ x₃ : ℝ, x₁ ≠ x₂ ∧ x₂ ≠ x₃ ∧ x₁ ≠ x₃ → h x₁ p = 0 → h x₂ p = 0 → h x₃ p = 0 → k x₁ p q r = 0 ∧ k x₂ p q r = 0 ∧ k x₃ p q r = 0):
  k 1 (-28) (2 - -28 * -30) (-20 * -30) = -155 :=
by
  sorry

end NUMINAMATH_GPT_k_at_1_value_l451_45149


namespace NUMINAMATH_GPT_volunteers_correct_l451_45101

-- Definitions of given conditions and the required result
def sheets_per_member : ℕ := 10
def cookies_per_sheet : ℕ := 16
def total_cookies : ℕ := 16000

-- Number of members who volunteered
def members : ℕ := total_cookies / (sheets_per_member * cookies_per_sheet)

-- Proof statement
theorem volunteers_correct :
  members = 100 :=
sorry

end NUMINAMATH_GPT_volunteers_correct_l451_45101


namespace NUMINAMATH_GPT_x_gt_y_neither_sufficient_nor_necessary_for_x_sq_gt_y_sq_l451_45102

theorem x_gt_y_neither_sufficient_nor_necessary_for_x_sq_gt_y_sq (x y : ℝ) :
  ¬((x > y) → (x^2 > y^2)) ∧ ¬((x^2 > y^2) → (x > y)) :=
by
  sorry

end NUMINAMATH_GPT_x_gt_y_neither_sufficient_nor_necessary_for_x_sq_gt_y_sq_l451_45102


namespace NUMINAMATH_GPT_mary_shirts_left_l451_45170

theorem mary_shirts_left :
  let blue_shirts := 35
  let brown_shirts := 48
  let red_shirts := 27
  let yellow_shirts := 36
  let green_shirts := 18
  let blue_given_away := 4 / 5 * blue_shirts
  let brown_given_away := 5 / 6 * brown_shirts
  let red_given_away := 2 / 3 * red_shirts
  let yellow_given_away := 3 / 4 * yellow_shirts
  let green_given_away := 1 / 3 * green_shirts
  let blue_left := blue_shirts - blue_given_away
  let brown_left := brown_shirts - brown_given_away
  let red_left := red_shirts - red_given_away
  let yellow_left := yellow_shirts - yellow_given_away
  let green_left := green_shirts - green_given_away
  blue_left + brown_left + red_left + yellow_left + green_left = 45 := by
  sorry

end NUMINAMATH_GPT_mary_shirts_left_l451_45170


namespace NUMINAMATH_GPT_tickets_per_box_l451_45165

-- Definitions
def boxes (G: Type) : ℕ := 9
def total_tickets (G: Type) : ℕ := 45

-- Theorem statement
theorem tickets_per_box (G: Type) : total_tickets G / boxes G = 5 :=
by
  sorry

end NUMINAMATH_GPT_tickets_per_box_l451_45165


namespace NUMINAMATH_GPT_max_value_l451_45198

-- Define the weights and values of gemstones
def weight_sapphire : ℕ := 6
def value_sapphire : ℕ := 15
def weight_ruby : ℕ := 3
def value_ruby : ℕ := 9
def weight_diamond : ℕ := 2
def value_diamond : ℕ := 5

-- Define the weight capacity
def max_weight : ℕ := 24

-- Define the availability constraint
def min_availability : ℕ := 10

-- The goal is to prove that the maximum value is 72
theorem max_value : ∃ (num_sapphire num_ruby num_diamond : ℕ),
  num_sapphire >= min_availability ∧
  num_ruby >= min_availability ∧
  num_diamond >= min_availability ∧
  num_sapphire * weight_sapphire + num_ruby * weight_ruby + num_diamond * weight_diamond ≤ max_weight ∧
  num_sapphire * value_sapphire + num_ruby * value_ruby + num_diamond * value_diamond = 72 :=
by sorry

end NUMINAMATH_GPT_max_value_l451_45198


namespace NUMINAMATH_GPT_wire_length_before_cutting_l451_45143

theorem wire_length_before_cutting (L S : ℝ) (h1 : S = 40) (h2 : S = 2 / 5 * L) : L + S = 140 :=
by
  sorry

end NUMINAMATH_GPT_wire_length_before_cutting_l451_45143


namespace NUMINAMATH_GPT_breadth_is_13_l451_45137

variable (b l : ℕ) (breadth : ℕ)

/-
We have the following conditions:
1. The area of the rectangular plot is 23 times its breadth.
2. The difference between the length and the breadth is 10 metres.
We need to prove that the breadth of the plot is 13 metres.
-/

theorem breadth_is_13
  (h1 : l * b = 23 * b)
  (h2 : l - b = 10) :
  b = 13 := 
sorry

end NUMINAMATH_GPT_breadth_is_13_l451_45137


namespace NUMINAMATH_GPT_cylinder_volume_l451_45195

theorem cylinder_volume (r h : ℝ) (hr : r = 5) (hh : h = 10) :
    π * r^2 * h = 250 * π := by
  -- We leave the actual proof as sorry for now
  sorry

end NUMINAMATH_GPT_cylinder_volume_l451_45195


namespace NUMINAMATH_GPT_fifth_inequality_l451_45177

theorem fifth_inequality :
  1 + (1 / (2^2 : ℝ)) + (1 / (3^2 : ℝ)) + (1 / (4^2 : ℝ)) + (1 / (5^2 : ℝ)) + (1 / (6^2 : ℝ)) < (11 / 6 : ℝ) :=
by
  sorry

end NUMINAMATH_GPT_fifth_inequality_l451_45177


namespace NUMINAMATH_GPT_crabapple_recipients_sequence_count_l451_45189

/-- Mrs. Crabapple teaches a class of 15 students and her advanced literature class meets three times a week.
    She picks a new student each period to receive a crabapple, ensuring no student receives more than one
    crabapple in a week. Prove that the number of different sequences of crabapple recipients is 2730. -/
theorem crabapple_recipients_sequence_count :
  ∃ sequence_count : ℕ, sequence_count = 15 * 14 * 13 ∧ sequence_count = 2730 :=
by
  sorry

end NUMINAMATH_GPT_crabapple_recipients_sequence_count_l451_45189


namespace NUMINAMATH_GPT_substract_repeating_decimal_l451_45130

noncomputable def repeating_decimal : ℝ := 1 / 3

theorem substract_repeating_decimal (x : ℝ) (h : x = repeating_decimal) : 
  1 - x = 2 / 3 :=
by
  sorry

end NUMINAMATH_GPT_substract_repeating_decimal_l451_45130


namespace NUMINAMATH_GPT_sequence_nonzero_l451_45125

def seq (a : ℕ → ℤ) : Prop :=
  a 1 = 1 ∧
  a 2 = 2 ∧
  ∀ n ≥ 3, 
    if (a (n - 1) * a (n - 2)) % 2 = 0 then 
      a n = 5 * (a (n - 1)) - 3 * (a (n - 2)) 
    else 
      a n = (a (n - 1)) - (a (n - 2))

theorem sequence_nonzero (a : ℕ → ℤ) (h : seq a) : ∀ n : ℕ, a n ≠ 0 := 
by sorry

end NUMINAMATH_GPT_sequence_nonzero_l451_45125


namespace NUMINAMATH_GPT_Adam_total_shopping_cost_l451_45150

theorem Adam_total_shopping_cost :
  let sandwiches := 3
  let sandwich_cost := 3
  let water_cost := 2
  (sandwiches * sandwich_cost + water_cost) = 11 := 
by
  sorry

end NUMINAMATH_GPT_Adam_total_shopping_cost_l451_45150


namespace NUMINAMATH_GPT_hyperbola_problem_l451_45199

theorem hyperbola_problem (s : ℝ) :
    (∃ b > 0, ∀ (x y : ℝ), (x, y) = (-4, 5) → (x^2 / 9) - (y^2 / b^2) = 1)
    ∧ (∀ (x y : ℝ), (x, y) = (-3, 0) → (x^2 / 9) - (y^2 / b^2) = 1)
    ∧ (∀ b > 0, (x, y) = (s, 3) → (x^2 / 9) - (7 * y^2 / 225) = 1)
    → s^2 = (288 / 25) :=
by
  sorry

end NUMINAMATH_GPT_hyperbola_problem_l451_45199


namespace NUMINAMATH_GPT_intersection_M_N_l451_45174

def M : Set ℝ := { x | x^2 ≤ 4 }
def N : Set ℝ := { x | Real.log x / Real.log 2 ≥ 1 }

theorem intersection_M_N : M ∩ N = {2} := by
  sorry

end NUMINAMATH_GPT_intersection_M_N_l451_45174


namespace NUMINAMATH_GPT_probability_same_color_l451_45197

/-- Define the number of green plates. -/
def green_plates : ℕ := 7

/-- Define the number of yellow plates. -/
def yellow_plates : ℕ := 5

/-- Define the total number of plates. -/
def total_plates : ℕ := green_plates + yellow_plates

/-- Calculate the binomial coefficient for choosing k items from a set of n items. -/
def binomial_coeff (n k : ℕ) : ℕ :=
  Nat.choose n k

/-- Prove that the probability of selecting three plates of the same color is 9/44. -/
theorem probability_same_color :
  (binomial_coeff green_plates 3 + binomial_coeff yellow_plates 3) / binomial_coeff total_plates 3 = 9 / 44 :=
by
  sorry

end NUMINAMATH_GPT_probability_same_color_l451_45197


namespace NUMINAMATH_GPT_base7_addition_XY_l451_45139

theorem base7_addition_XY (X Y : ℕ) (h1 : (Y + 2) % 7 = X % 7) (h2 : (X + 5) % 7 = 9 % 7) : X + Y = 6 :=
by sorry

end NUMINAMATH_GPT_base7_addition_XY_l451_45139


namespace NUMINAMATH_GPT_same_heads_probability_l451_45186

theorem same_heads_probability
  (fair_coin : Real := 1/2)
  (biased_coin : Real := 5/8)
  (prob_Jackie_eq_Phil : Real := 77/225) :
  let m := 77
  let n := 225
  (m : ℕ) + (n : ℕ) = 302 := 
by {
  -- The proof would involve constructing the generating functions,
  -- calculating the sum of corresponding coefficients and showing that the
  -- resulting probability reduces to 77/225
  sorry
}

end NUMINAMATH_GPT_same_heads_probability_l451_45186


namespace NUMINAMATH_GPT_carol_points_loss_l451_45138

theorem carol_points_loss 
  (first_round_points : ℕ) (second_round_points : ℕ) (end_game_points : ℕ) 
  (h1 : first_round_points = 17) 
  (h2 : second_round_points = 6) 
  (h3 : end_game_points = 7) : 
  (first_round_points + second_round_points - end_game_points = 16) :=
by 
  sorry

end NUMINAMATH_GPT_carol_points_loss_l451_45138


namespace NUMINAMATH_GPT_geometric_sequence_formula_l451_45156

noncomputable def a_n (q : ℝ) (n : ℕ) : ℝ := if n = 0 then 0 else 2^(n - 1)

theorem geometric_sequence_formula (q : ℝ) (S : ℕ → ℝ) (n : ℕ) (hn : n > 0) :
  a_n q n = 2^(n - 1) :=
sorry

end NUMINAMATH_GPT_geometric_sequence_formula_l451_45156


namespace NUMINAMATH_GPT_reciprocals_expression_eq_zero_l451_45182

theorem reciprocals_expression_eq_zero {m n : ℝ} (h : m * n = 1) : (2 * m - 2 / n) * (1 / m + n) = 0 :=
by
  sorry

end NUMINAMATH_GPT_reciprocals_expression_eq_zero_l451_45182


namespace NUMINAMATH_GPT_find_first_number_l451_45161

theorem find_first_number 
  (first_number second_number hcf lcm : ℕ) 
  (hCF_condition : hcf = 12) 
  (lCM_condition : lcm = 396) 
  (one_number_condition : first_number = 99) 
  (relation_condition : first_number * second_number = hcf * lcm) : 
  second_number = 48 :=
by
  sorry

end NUMINAMATH_GPT_find_first_number_l451_45161


namespace NUMINAMATH_GPT_third_side_range_l451_45132

theorem third_side_range (a : ℝ) (h₃ : 0 < a ∧ a ≠ 0) (h₅ : 0 < a ∧ a ≠ 0): 
  (2 < a ∧ a < 8) ↔ (3 - 5 < a ∧ a < 3 + 5) :=
by
  sorry

end NUMINAMATH_GPT_third_side_range_l451_45132


namespace NUMINAMATH_GPT_positive_roots_of_x_pow_x_eq_one_over_sqrt_two_l451_45194

theorem positive_roots_of_x_pow_x_eq_one_over_sqrt_two (x : ℝ) (h : x > 0) : 
  (x^x = 1 / Real.sqrt 2) ↔ (x = 1 / 2 ∨ x = 1 / 4) := by
  sorry

end NUMINAMATH_GPT_positive_roots_of_x_pow_x_eq_one_over_sqrt_two_l451_45194


namespace NUMINAMATH_GPT_find_divisor_l451_45136

theorem find_divisor (d : ℕ) (h1 : 109 % d = 1) (h2 : 109 / d = 9) : d = 12 := by
  sorry

end NUMINAMATH_GPT_find_divisor_l451_45136


namespace NUMINAMATH_GPT_regular_polygon_sides_l451_45155

theorem regular_polygon_sides (n : ℕ) 
  (h1 : (n - 2) * 180 = 2 * 360) : 
  n = 6 :=
sorry

end NUMINAMATH_GPT_regular_polygon_sides_l451_45155


namespace NUMINAMATH_GPT_josette_paid_correct_amount_l451_45163

-- Define the number of small and large bottles
def num_small_bottles : ℕ := 3
def num_large_bottles : ℕ := 2

-- Define the cost of each type of bottle
def cost_per_small_bottle : ℝ := 1.50
def cost_per_large_bottle : ℝ := 2.40

-- Define the total number of bottles purchased
def total_bottles : ℕ := num_small_bottles + num_large_bottles

-- Define the discount rate applicable when purchasing 5 or more bottles
def discount_rate : ℝ := 0.10

-- Calculate the initial total cost before any discount
def total_cost_before_discount : ℝ :=
  (num_small_bottles * cost_per_small_bottle) + 
  (num_large_bottles * cost_per_large_bottle)

-- Calculate the discount amount if applicable
def discount_amount : ℝ :=
  if total_bottles >= 5 then
    discount_rate * total_cost_before_discount
  else
    0

-- Calculate the final amount Josette paid after applying any discount
def final_amount_paid : ℝ :=
  total_cost_before_discount - discount_amount

-- Prove that the final amount paid is €8.37
theorem josette_paid_correct_amount :
  final_amount_paid = 8.37 :=
by
  sorry

end NUMINAMATH_GPT_josette_paid_correct_amount_l451_45163


namespace NUMINAMATH_GPT_ab_value_l451_45188

-- Defining the conditions as Lean assumptions
theorem ab_value (a b c : ℝ) (h1 : a - b = 3) (h2 : a^2 + b^2 = 27) (h3 : a + b + c = 10) : a * b = 9 :=
by
  sorry

end NUMINAMATH_GPT_ab_value_l451_45188


namespace NUMINAMATH_GPT_sum_of_eight_numbers_l451_45151

theorem sum_of_eight_numbers (average : ℝ) (h : average = 5) :
  (8 * average) = 40 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_eight_numbers_l451_45151


namespace NUMINAMATH_GPT_transform_polynomial_eq_correct_factorization_positive_polynomial_gt_zero_l451_45123

-- Define the polynomial transformation
def transform_polynomial (x : ℝ) : ℝ := x^2 + 8 * x - 1

-- Transformation problem
theorem transform_polynomial_eq (x m n : ℝ) :
  (x + 4)^2 - 17 = transform_polynomial x := 
sorry

-- Define the polynomial for correction
def factor_polynomial (x : ℝ) : ℝ := x^2 - 3 * x - 40

-- Factoring correction problem
theorem correct_factorization (x : ℝ) :
  factor_polynomial x = (x + 5) * (x - 8) := 
sorry

-- Define the polynomial for the positivity proof
def positive_polynomial (x y : ℝ) : ℝ := x^2 + y^2 - 2 * x - 4 * y + 16

-- Positive polynomial proof
theorem positive_polynomial_gt_zero (x y : ℝ) :
  positive_polynomial x y > 0 := 
sorry

end NUMINAMATH_GPT_transform_polynomial_eq_correct_factorization_positive_polynomial_gt_zero_l451_45123


namespace NUMINAMATH_GPT_quadratic_to_completed_square_l451_45114

-- Define the given quadratic function.
def quadratic_function (x : ℝ) : ℝ := x^2 + 2 * x - 2

-- Define the completed square form of the function.
def completed_square_form (x : ℝ) : ℝ := (x + 1)^2 - 3

-- The theorem statement that needs to be proven.
theorem quadratic_to_completed_square :
  ∀ x : ℝ, quadratic_function x = completed_square_form x :=
by sorry

end NUMINAMATH_GPT_quadratic_to_completed_square_l451_45114


namespace NUMINAMATH_GPT_plane_perpendicular_l451_45193

-- Define types for lines and planes
axiom Line : Type
axiom Plane : Type

-- Define the relationships between lines and planes
axiom Parallel (l : Line) (p : Plane) : Prop
axiom Perpendicular (l : Line) (p : Plane) : Prop
axiom PlanePerpendicular (p1 p2 : Plane) : Prop

-- The setting conditions
variables (c : Line) (α β : Plane)

-- The given conditions
axiom c_perpendicular_β : Perpendicular c β
axiom c_parallel_α : Parallel c α

-- The proof goal (without the proof body)
theorem plane_perpendicular : PlanePerpendicular α β :=
by
  sorry

end NUMINAMATH_GPT_plane_perpendicular_l451_45193


namespace NUMINAMATH_GPT_domain_of_g_l451_45119

noncomputable def f : ℝ → ℝ := sorry  -- Placeholder for the function f

theorem domain_of_g :
  {x : ℝ | (0 ≤ x ∧ x < 2) ∨ (2 < x ∧ x ≤ 3)} = -- Expected domain of g(x)
  { x : ℝ |
    (0 ≤ x ∧ x ≤ 6) ∧ -- Domain of f is 0 ≤ x ≤ 6
    2 * x ≤ 6 ∧ -- For g(x) to be in the domain of f(2x)
    0 ≤ 2 * x ∧ -- Ensures 2x fits within the domain 0 < 2x < 6
    x ≠ 2 } -- x cannot be 2
:= sorry

end NUMINAMATH_GPT_domain_of_g_l451_45119


namespace NUMINAMATH_GPT_jogging_time_after_two_weeks_l451_45184

noncomputable def daily_jogging_hours : ℝ := 1.5
noncomputable def days_in_two_weeks : ℕ := 14

theorem jogging_time_after_two_weeks : daily_jogging_hours * days_in_two_weeks = 21 := by
  sorry

end NUMINAMATH_GPT_jogging_time_after_two_weeks_l451_45184


namespace NUMINAMATH_GPT_quadratic_discriminant_constraint_l451_45106

theorem quadratic_discriminant_constraint (c : ℝ) : 
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ x1^2 - 4*x1 + c = 0 ∧ x2^2 - 4*x2 + c = 0) ↔ c < 4 := 
by
  sorry

end NUMINAMATH_GPT_quadratic_discriminant_constraint_l451_45106
