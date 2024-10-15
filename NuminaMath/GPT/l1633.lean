import Mathlib

namespace NUMINAMATH_GPT_initial_number_of_men_l1633_163355

theorem initial_number_of_men (M : ℝ) (P : ℝ) (h1 : P = M * 20) (h2 : P = (M + 200) * 16.67) : M = 1000 :=
by
  sorry

end NUMINAMATH_GPT_initial_number_of_men_l1633_163355


namespace NUMINAMATH_GPT_shaded_area_represents_correct_set_l1633_163334

theorem shaded_area_represents_correct_set :
  ∀ (U A B : Set ℕ), 
    U = {0, 1, 2, 3, 4} → 
    A = {1, 2, 3} → 
    B = {2, 4} → 
    (U \ (A ∪ B)) ∪ (A ∩ B) = {0, 2} :=
by
  intros U A B hU hA hB
  -- The rest of the proof would go here
  sorry

end NUMINAMATH_GPT_shaded_area_represents_correct_set_l1633_163334


namespace NUMINAMATH_GPT_same_color_probability_l1633_163365

/-- There are 7 red plates and 5 blue plates. We want to prove that the probability of
    selecting 3 plates, where all are of the same color, is 9/44. -/
theorem same_color_probability :
  let total_plates := 12
  let total_ways_to_choose := Nat.choose total_plates 3
  let red_plates := 7
  let blue_plates := 5
  let ways_to_choose_red := Nat.choose red_plates 3
  let ways_to_choose_blue := Nat.choose blue_plates 3
  let favorable_ways_to_choose := ways_to_choose_red + ways_to_choose_blue
  ∃ (prob : ℚ), prob = (favorable_ways_to_choose : ℚ) / (total_ways_to_choose : ℚ) ∧
                 prob = 9 / 44 :=
by
  sorry

end NUMINAMATH_GPT_same_color_probability_l1633_163365


namespace NUMINAMATH_GPT_usual_time_is_20_l1633_163393

-- Define the problem
variables (T T': ℕ)

-- Conditions
axiom condition1 : T' = T + 5
axiom condition2 : T' = 5 * T / 4

-- Proof statement
theorem usual_time_is_20 : T = 20 :=
  sorry

end NUMINAMATH_GPT_usual_time_is_20_l1633_163393


namespace NUMINAMATH_GPT_jerky_batch_size_l1633_163306

theorem jerky_batch_size
  (total_order_bags : ℕ)
  (initial_bags : ℕ)
  (days_to_fulfill : ℕ)
  (remaining_bags : ℕ := total_order_bags - initial_bags)
  (production_per_day : ℕ := remaining_bags / days_to_fulfill) :
  total_order_bags = 60 →
  initial_bags = 20 →
  days_to_fulfill = 4 →
  production_per_day = 10 :=
by
  intros
  sorry

end NUMINAMATH_GPT_jerky_batch_size_l1633_163306


namespace NUMINAMATH_GPT_average_age_of_school_l1633_163377

theorem average_age_of_school 
  (total_students : ℕ)
  (average_age_boys : ℕ)
  (average_age_girls : ℕ)
  (number_of_girls : ℕ)
  (number_of_boys : ℕ := total_students - number_of_girls)
  (total_age_boys : ℕ := average_age_boys * number_of_boys)
  (total_age_girls : ℕ := average_age_girls * number_of_girls)
  (total_age_students : ℕ := total_age_boys + total_age_girls) :
  total_students = 640 →
  average_age_boys = 12 →
  average_age_girls = 11 →
  number_of_girls = 160 →
  (total_age_students : ℝ) / (total_students : ℝ) = 11.75 :=
by
  intros h1 h2 h3 h4
  sorry

end NUMINAMATH_GPT_average_age_of_school_l1633_163377


namespace NUMINAMATH_GPT_product_of_ratios_eq_l1633_163307

theorem product_of_ratios_eq :
  (∃ x_1 y_1 x_2 y_2 x_3 y_3 : ℝ,
    (x_1^3 - 3 * x_1 * y_1^2 = 2006) ∧
    (y_1^3 - 3 * x_1^2 * y_1 = 2007) ∧
    (x_2^3 - 3 * x_2 * y_2^2 = 2006) ∧
    (y_2^3 - 3 * x_2^2 * y_2 = 2007) ∧
    (x_3^3 - 3 * x_3 * y_3^2 = 2006) ∧
    (y_3^3 - 3 * x_3^2 * y_3 = 2007)) →
    (1 - x_1 / y_1) * (1 - x_2 / y_2) * (1 - x_3 / y_3) = 1 / 1003 :=
by
  sorry

end NUMINAMATH_GPT_product_of_ratios_eq_l1633_163307


namespace NUMINAMATH_GPT_find_value_of_expression_l1633_163389

theorem find_value_of_expression (a : ℝ) (h : a^2 - a - 1 = 0) : a^3 - a^2 - a + 2023 = 2023 :=
by
  sorry

end NUMINAMATH_GPT_find_value_of_expression_l1633_163389


namespace NUMINAMATH_GPT_solve_fraction_l1633_163329

theorem solve_fraction (x : ℚ) : (x^2 + 3*x + 5) / (x + 6) = x + 7 ↔ x = -37 / 10 :=
by
  sorry

end NUMINAMATH_GPT_solve_fraction_l1633_163329


namespace NUMINAMATH_GPT_intersection_length_l1633_163361

theorem intersection_length 
  (A B : ℝ × ℝ) 
  (hA : A.1^2 + A.2^2 = 1) 
  (hB : B.1^2 + B.2^2 = 1) 
  (hA_on_line : A.1 = A.2) 
  (hB_on_line : B.1 = B.2) 
  (hAB : A ≠ B) :
  dist A B = 2 :=
by sorry

end NUMINAMATH_GPT_intersection_length_l1633_163361


namespace NUMINAMATH_GPT_n_minus_two_is_square_of_natural_number_l1633_163350

theorem n_minus_two_is_square_of_natural_number 
  (n m : ℕ) 
  (hn: n ≥ 3) 
  (hm: m = n * (n - 1) / 2) 
  (hm_odd: m % 2 = 1)
  (unique_rem: ∀ i j : ℕ, i ≠ j → (i + j) % m ≠ (i + j) % m) :
  ∃ k : ℕ, n - 2 = k * k := 
sorry

end NUMINAMATH_GPT_n_minus_two_is_square_of_natural_number_l1633_163350


namespace NUMINAMATH_GPT_maximize_product_numbers_l1633_163324

theorem maximize_product_numbers (a b : ℕ) (ha : a = 96420) (hb : b = 87531) (cond: a * b = 96420 * 87531):
  b = 87531 := 
by sorry

end NUMINAMATH_GPT_maximize_product_numbers_l1633_163324


namespace NUMINAMATH_GPT_suitable_b_values_l1633_163330

theorem suitable_b_values (b : ℤ) :
  (∃ (c d e f : ℤ), 35 * c * d + (c * f + d * e) * b + 35 = 0 ∧
    c * e = 35 ∧ d * f = 35) →
  (∃ (k : ℤ), b = 2 * k) :=
by
  intro h
  sorry

end NUMINAMATH_GPT_suitable_b_values_l1633_163330


namespace NUMINAMATH_GPT_trading_organization_increase_price_l1633_163346

theorem trading_organization_increase_price 
  (initial_moisture_content : ℝ)
  (final_moisture_content : ℝ)
  (solid_mass : ℝ)
  (initial_total_mass final_total_mass : ℝ) :
  initial_moisture_content = 0.99 → 
  final_moisture_content = 0.98 →
  initial_total_mass = 100 →
  solid_mass = initial_total_mass * (1 - initial_moisture_content) →
  final_total_mass = solid_mass / (1 - final_moisture_content) →
  (final_total_mass / initial_total_mass) = 0.5 →
  100 * (1 - (final_total_mass / initial_total_mass)) = 100 :=
by sorry

end NUMINAMATH_GPT_trading_organization_increase_price_l1633_163346


namespace NUMINAMATH_GPT_line_parallel_l1633_163380

theorem line_parallel (x y : ℝ) :
  ∃ m b : ℝ, 
    y = m * (x - 2) + (-4) ∧ 
    m = 2 ∧ 
    (∀ (x y : ℝ), y = 2 * x - 8 → 2 * x - y - 8 = 0) :=
sorry

end NUMINAMATH_GPT_line_parallel_l1633_163380


namespace NUMINAMATH_GPT_laborers_employed_l1633_163383

theorem laborers_employed 
    (H L : ℕ) 
    (h1 : H + L = 35) 
    (h2 : 140 * H + 90 * L = 3950) : 
    L = 19 :=
by
  sorry

end NUMINAMATH_GPT_laborers_employed_l1633_163383


namespace NUMINAMATH_GPT_fruitseller_apples_l1633_163301

theorem fruitseller_apples (x : ℝ) (sold_percent remaining_apples : ℝ) 
  (h_sold : sold_percent = 0.80) 
  (h_remaining : remaining_apples = 500) 
  (h_equation : (1 - sold_percent) * x = remaining_apples) : 
  x = 2500 := 
by 
  sorry

end NUMINAMATH_GPT_fruitseller_apples_l1633_163301


namespace NUMINAMATH_GPT_list_size_is_2017_l1633_163332

def has_sum (L : List ℤ) (n : ℤ) : Prop :=
  List.sum L = n

def has_product (L : List ℤ) (n : ℤ) : Prop :=
  List.prod L = n

def includes (L : List ℤ) (n : ℤ) : Prop :=
  n ∈ L

theorem list_size_is_2017 
(L : List ℤ) :
  has_sum L 2018 ∧ 
  has_product L 2018 ∧ 
  includes L 2018 
  → L.length = 2017 :=
by 
  sorry

end NUMINAMATH_GPT_list_size_is_2017_l1633_163332


namespace NUMINAMATH_GPT_sum_of_abcd_l1633_163399

variable (a b c d : ℚ)

def condition (x : ℚ) : Prop :=
  x = a + 3 ∧
  x = b + 7 ∧
  x = c + 5 ∧
  x = d + 9 ∧
  x = a + b + c + d + 13

theorem sum_of_abcd (x : ℚ) (h : condition a b c d x) : a + b + c + d = -28 / 3 := 
by sorry

end NUMINAMATH_GPT_sum_of_abcd_l1633_163399


namespace NUMINAMATH_GPT_polynomial_irreducible_segment_intersect_l1633_163348

-- Part (a)
theorem polynomial_irreducible 
  (f : Polynomial ℤ) 
  (h_def : f = Polynomial.C 12 + Polynomial.X * Polynomial.C 9 + Polynomial.X^2 * Polynomial.C 6 + Polynomial.X^3 * Polynomial.C 3 + Polynomial.X^4) : 
  ¬ ∃ (p q : Polynomial ℤ), (Polynomial.degree p = 2) ∧ (Polynomial.degree q = 2) ∧ (f = p * q) :=
sorry

-- Part (b)
theorem segment_intersect 
  (n : ℕ) 
  (segments : Fin (2*n+1) → Set (ℝ × ℝ)) 
  (h_intersect : ∀ i, ∃ n_indices : Finset (Fin (2*n+1)), n_indices.card = n ∧ ∀ j ∈ n_indices, (segments i ∩ segments j).Nonempty) :
  ∃ i, ∀ j, i ≠ j → (segments i ∩ segments j).Nonempty :=
sorry


end NUMINAMATH_GPT_polynomial_irreducible_segment_intersect_l1633_163348


namespace NUMINAMATH_GPT_rate_of_stream_is_5_l1633_163319

-- Define the conditions
def boat_speed : ℝ := 16  -- Boat speed in still water
def time_downstream : ℝ := 3  -- Time taken downstream
def distance_downstream : ℝ := 63  -- Distance covered downstream

-- Define the rate of the stream as an unknown variable
def rate_of_stream (v : ℝ) : Prop := 
  distance_downstream = (boat_speed + v) * time_downstream

-- Statement to prove
theorem rate_of_stream_is_5 : 
  ∃ (v : ℝ), rate_of_stream v ∧ v = 5 :=
by
  use 5
  simp [boat_speed, time_downstream, distance_downstream, rate_of_stream]
  sorry

end NUMINAMATH_GPT_rate_of_stream_is_5_l1633_163319


namespace NUMINAMATH_GPT_product_of_roots_l1633_163314

-- Let x₁ and x₂ be roots of the quadratic equation x^2 + x - 1 = 0
theorem product_of_roots (x₁ x₂ : ℝ) (h₁ : x₁^2 + x₁ - 1 = 0) (h₂ : x₂^2 + x₂ - 1 = 0) :
  x₁ * x₂ = -1 :=
sorry

end NUMINAMATH_GPT_product_of_roots_l1633_163314


namespace NUMINAMATH_GPT_exponential_function_inequality_l1633_163311

theorem exponential_function_inequality {a : ℝ} (h0 : 0 < a) (h1 : a < 1) :
  (a^3) * (a^2) < a^2 :=
by
  sorry

end NUMINAMATH_GPT_exponential_function_inequality_l1633_163311


namespace NUMINAMATH_GPT_find_f2_l1633_163375

noncomputable def f (x a b : ℝ) : ℝ := x^3 + a*x^2 + b*x + a^2

theorem find_f2 (a b : ℝ) (h : (∃ x : ℝ, f x a b = 10 ∧ x = 1)):
  f 2 a b = 18 ∨ f 2 a b = 11 :=
sorry

end NUMINAMATH_GPT_find_f2_l1633_163375


namespace NUMINAMATH_GPT_rotation_locus_l1633_163381

-- Definitions for points and structure of the cube
structure Point3D :=
(x : ℝ) (y : ℝ) (z : ℝ)

structure Cube :=
(A : Point3D) (B : Point3D) (C : Point3D) (D : Point3D)
(E : Point3D) (F : Point3D) (G : Point3D) (H : Point3D)

-- Function to perform the required rotations and return the locus geometrical representation
noncomputable def locus_points_on_surface (c : Cube) : Set Point3D :=
sorry

-- Mathematical problem rephrased in Lean 4 statement
theorem rotation_locus (c : Cube) :
  locus_points_on_surface c = {c.D, c.A} ∪ {c.A, c.C} ∪ {c.C, c.D} :=
sorry

end NUMINAMATH_GPT_rotation_locus_l1633_163381


namespace NUMINAMATH_GPT_lance_hourly_earnings_l1633_163344

theorem lance_hourly_earnings
  (hours_per_week : ℕ)
  (workdays_per_week : ℕ)
  (daily_earnings : ℕ)
  (total_weekly_earnings : ℕ)
  (hourly_wage : ℕ)
  (h1 : hours_per_week = 35)
  (h2 : workdays_per_week = 5)
  (h3 : daily_earnings = 63)
  (h4 : total_weekly_earnings = daily_earnings * workdays_per_week)
  (h5 : total_weekly_earnings = hourly_wage * hours_per_week)
  : hourly_wage = 9 :=
sorry

end NUMINAMATH_GPT_lance_hourly_earnings_l1633_163344


namespace NUMINAMATH_GPT_g_at_negative_two_l1633_163302

-- Function definition
def g (x : ℝ) : ℝ := 3*x^5 - 4*x^4 + 2*x^3 - 5*x^2 - x + 8

-- Theorem statement
theorem g_at_negative_two : g (-2) = -186 :=
by
  -- Proof will go here, but it is skipped with sorry
  sorry

end NUMINAMATH_GPT_g_at_negative_two_l1633_163302


namespace NUMINAMATH_GPT_moles_of_NaHCO3_needed_l1633_163326

theorem moles_of_NaHCO3_needed 
  (HC2H3O2_moles: ℕ)
  (H2O_moles: ℕ)
  (NaHCO3_HC2H3O2_molar_ratio: ℕ)
  (reaction: NaHCO3_HC2H3O2_molar_ratio = 1 ∧ H2O_moles = 3) :
  ∃ NaHCO3_moles : ℕ, NaHCO3_moles = 3 :=
by
  sorry

end NUMINAMATH_GPT_moles_of_NaHCO3_needed_l1633_163326


namespace NUMINAMATH_GPT_laura_park_time_percentage_l1633_163376

theorem laura_park_time_percentage (num_trips: ℕ) (time_in_park: ℝ) (walking_time: ℝ) 
    (total_percentage_in_park: ℝ) 
    (h1: num_trips = 6) 
    (h2: time_in_park = 2) 
    (h3: walking_time = 0.5) 
    (h4: total_percentage_in_park = 80) : 
    (time_in_park * num_trips) / ((time_in_park + walking_time) * num_trips) * 100 = total_percentage_in_park :=
by
  sorry

end NUMINAMATH_GPT_laura_park_time_percentage_l1633_163376


namespace NUMINAMATH_GPT_friendly_triangle_angle_l1633_163342

theorem friendly_triangle_angle (α : ℝ) (β : ℝ) (γ : ℝ) (hα12β : α = 2 * β) (h_sum : α + β + γ = 180) :
    (α = 42 ∨ α = 84 ∨ α = 92) ∧ (42 = β ∨ 42 = γ) := 
sorry

end NUMINAMATH_GPT_friendly_triangle_angle_l1633_163342


namespace NUMINAMATH_GPT_compare_polynomials_l1633_163367

theorem compare_polynomials (x : ℝ) (h : x ≥ 0) : 
  (x > 2 → 5*x^2 - 1 > 3*x^2 + 3*x + 1) ∧ 
  (x = 2 → 5*x^2 - 1 = 3*x^2 + 3*x + 1) ∧ 
  (0 ≤ x → x < 2 → 5*x^2 - 1 < 3*x^2 + 3*x + 1) :=
sorry

end NUMINAMATH_GPT_compare_polynomials_l1633_163367


namespace NUMINAMATH_GPT_like_terms_exponent_equality_l1633_163374

theorem like_terms_exponent_equality (m n : ℕ) (a b : ℝ) 
    (H : 3 * a^m * b^2 = 2/3 * a * b^n) : m = 1 ∧ n = 2 :=
by
  sorry

end NUMINAMATH_GPT_like_terms_exponent_equality_l1633_163374


namespace NUMINAMATH_GPT_probability_quarter_circle_is_pi_div_16_l1633_163322

open Real

noncomputable def probability_quarter_circle : ℝ :=
  let side_length := 2
  let total_area := side_length * side_length
  let quarter_circle_area := π / 4
  quarter_circle_area / total_area

theorem probability_quarter_circle_is_pi_div_16 :
  probability_quarter_circle = π / 16 :=
by
  sorry

end NUMINAMATH_GPT_probability_quarter_circle_is_pi_div_16_l1633_163322


namespace NUMINAMATH_GPT_part_I_part_II_l1633_163373

noncomputable def A : Set ℝ := {x | 2*x^2 - 5*x - 3 <= 0}
noncomputable def B (a : ℝ) : Set ℝ := {x | (x - (2*a + 1)) * (x - (a - 1)) < 0}

theorem part_I :
  (A ∪ B 0 = {x : ℝ | -1 < x ∧ x ≤ 3}) :=
by sorry

theorem part_II (a : ℝ) :
  (A ∩ B a = ∅) →
  (a ≤ -3/4 ∨ a ≥ 4) ∧ a ≠ -2 :=
by sorry


end NUMINAMATH_GPT_part_I_part_II_l1633_163373


namespace NUMINAMATH_GPT_remainder_division_Q_l1633_163360

noncomputable def Q_rest : Polynomial ℝ := -(Polynomial.X : Polynomial ℝ) + 125

theorem remainder_division_Q (Q : Polynomial ℝ) :
  Q.eval 20 = 105 ∧ Q.eval 105 = 20 →
  ∃ R : Polynomial ℝ, Q = (Polynomial.X - 20) * (Polynomial.X - 105) * R + Q_rest :=
by sorry

end NUMINAMATH_GPT_remainder_division_Q_l1633_163360


namespace NUMINAMATH_GPT_Leila_donated_2_bags_l1633_163385

theorem Leila_donated_2_bags (L : ℕ) (h1 : 25 * L + 7 = 57) : L = 2 :=
by
  sorry

end NUMINAMATH_GPT_Leila_donated_2_bags_l1633_163385


namespace NUMINAMATH_GPT_math_problem_l1633_163335

theorem math_problem (f : ℝ → ℝ) (hf : Differentiable ℝ f) 
  (h : ∀ x, (x - 1) * (deriv f x) ≥ 0) : 
  f 0 + f 2 ≥ 2 * f 1 :=
sorry

end NUMINAMATH_GPT_math_problem_l1633_163335


namespace NUMINAMATH_GPT_steve_take_home_pay_l1633_163391

def annual_salary : ℝ := 40000
def tax_rate : ℝ := 0.20
def healthcare_rate : ℝ := 0.10
def union_dues : ℝ := 800

theorem steve_take_home_pay : 
  (annual_salary - (annual_salary * tax_rate + annual_salary * healthcare_rate + union_dues)) = 27200 := 
by 
  sorry

end NUMINAMATH_GPT_steve_take_home_pay_l1633_163391


namespace NUMINAMATH_GPT_find_BM_length_l1633_163386

variables (A M C : Type) (MA CA BC BM : ℝ) (x d h : ℝ)

-- Conditions
def condition1 : Prop := MA + (BC - BM) = 2 * CA
def condition2 : Prop := MA = x
def condition3 : Prop := CA = d
def condition4 : Prop := BC = h

-- The proof problem statement
theorem find_BM_length (A M C : Type) (MA CA BC BM : ℝ) (x d h : ℝ)
  (h1 : condition1 MA CA BC BM)
  (h2 : condition2 MA x)
  (h3 : condition3 CA d)
  (h4 : condition4 BC h) :
  BM = 2 * d :=
sorry

end NUMINAMATH_GPT_find_BM_length_l1633_163386


namespace NUMINAMATH_GPT_original_average_age_l1633_163362

-- Definitions based on conditions
def original_strength : ℕ := 12
def new_student_count : ℕ := 12
def new_student_average_age : ℕ := 32
def age_decrease : ℕ := 4
def total_student_count : ℕ := original_strength + new_student_count
def combined_total_age (A : ℕ) : ℕ := original_strength * A + new_student_count * new_student_average_age
def new_average_age (A : ℕ) : ℕ := A - age_decrease

-- Statement of the problem
theorem original_average_age (A : ℕ) (h : combined_total_age A / total_student_count = new_average_age A) : A = 40 := 
by 
  sorry

end NUMINAMATH_GPT_original_average_age_l1633_163362


namespace NUMINAMATH_GPT_total_students_l1633_163356

theorem total_students (ratio_boys_girls : ℕ) (girls : ℕ) (boys : ℕ) (total_students : ℕ)
  (h1 : ratio_boys_girls = 2)     -- The simplified ratio of boys to girls
  (h2 : girls = 200)              -- There are 200 girls
  (h3 : boys = ratio_boys_girls * girls) -- Number of boys is ratio * number of girls
  (h4 : total_students = boys + girls)   -- Total number of students is the sum of boys and girls
  : total_students = 600 :=             -- Prove that the total number of students is 600
sorry

end NUMINAMATH_GPT_total_students_l1633_163356


namespace NUMINAMATH_GPT_find_k_l1633_163347

-- Define vector a and vector b
def vec_a : (ℝ × ℝ) := (1, 1)
def vec_b : (ℝ × ℝ) := (-3, 1)

-- Define the expression for k * vec_a - vec_b
def k_vec_a_minus_vec_b (k : ℝ) : ℝ × ℝ :=
  (k * vec_a.1 - vec_b.1, k * vec_a.2 - vec_b.2)

-- Define the dot product condition for perpendicular vectors
def dot_product (u v : ℝ × ℝ) : ℝ :=
  u.1 * v.1 + u.2 * v.2

-- The theorem to be proved: k = -1 is the value that makes the dot product zero
theorem find_k : ∃ k : ℝ, dot_product (k_vec_a_minus_vec_b k) vec_a = 0 :=
by
  use -1
  sorry

end NUMINAMATH_GPT_find_k_l1633_163347


namespace NUMINAMATH_GPT_collinear_points_k_value_l1633_163328

theorem collinear_points_k_value : 
  (∀ k : ℝ, ∃ (a : ℝ) (b : ℝ), ∀ (x : ℝ) (y : ℝ),
    ((x, y) = (1, -2) ∨ (x, y) = (3, 2) ∨ (x, y) = (6, k / 3)) → y = a * x + b) → k = 24 :=
by
sorry

end NUMINAMATH_GPT_collinear_points_k_value_l1633_163328


namespace NUMINAMATH_GPT_john_max_books_l1633_163349

theorem john_max_books (h₁ : 4575 ≥ 0) (h₂ : 325 > 0) : 
  ∃ (x : ℕ), x = 14 ∧ ∀ n : ℕ, n ≤ x ↔ n * 325 ≤ 4575 := 
  sorry

end NUMINAMATH_GPT_john_max_books_l1633_163349


namespace NUMINAMATH_GPT_no_rational_roots_of_polynomial_l1633_163379

theorem no_rational_roots_of_polynomial :
  ¬ ∃ (x : ℚ), (3 * x^4 - 7 * x^3 - 4 * x^2 + 8 * x + 3 = 0) :=
by
  sorry

end NUMINAMATH_GPT_no_rational_roots_of_polynomial_l1633_163379


namespace NUMINAMATH_GPT_problem_part1_problem_part2_l1633_163395

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := 2 * (Real.cos x) ^ 2 + 2 * Real.sqrt 3 * (Real.sin x) * (Real.cos x) + a

noncomputable def g (x : ℝ) : ℝ := 2 * Real.sin (4 * x - Real.pi / 6) + 3

theorem problem_part1 (h : ∀ x ∈ Set.Icc 0 (Real.pi / 2), f x 2 ≥ 2) :
    ∃ a : ℝ, a = 2 ∧ 
    ∀ k : ℤ, ∀ x ∈ Set.Icc (k * Real.pi - Real.pi / 3) (k * Real.pi + Real.pi / 6), 
    ∃ m : ℤ, x = (m * Real.pi / 2 + Real.pi / 12) ∨ x = (m * Real.pi / 2 + Real.pi / 4) := sorry

theorem problem_part2 :
    ∀ x ∈ Set.Icc 0 (Real.pi / 2), g x = 4 → 
    ∃ s : ℝ, s = Real.pi / 3 := sorry

end NUMINAMATH_GPT_problem_part1_problem_part2_l1633_163395


namespace NUMINAMATH_GPT_sum_of_first_seven_terms_l1633_163372

variable (a : ℕ → ℝ)
variable (d : ℝ)

-- Define the arithmetic sequence
def arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n, a (n + 1) = a n + d

-- Given condition
axiom a3_a4_a5_sum : a 3 + a 4 + a 5 = 12

-- Statement to prove
theorem sum_of_first_seven_terms (h : arithmetic_sequence a d) : 
  a 1 + a 2 + a 3 + a 4 + a 5 + a 6 + a 7 = 28 :=
sorry

end NUMINAMATH_GPT_sum_of_first_seven_terms_l1633_163372


namespace NUMINAMATH_GPT_frac_square_between_half_and_one_l1633_163398

theorem frac_square_between_half_and_one :
  let fraction := (11 : ℝ) / 12
  let expr := fraction^2
  (1 / 2) < expr ∧ expr < 1 :=
by
  let fraction := (11 : ℝ) / 12
  let expr := fraction^2
  have h1 : (1 / 2) < expr := sorry
  have h2 : expr < 1 := sorry
  exact ⟨h1, h2⟩

end NUMINAMATH_GPT_frac_square_between_half_and_one_l1633_163398


namespace NUMINAMATH_GPT_correct_calculation_l1633_163397

theorem correct_calculation : (Real.sqrt 3) ^ 2 = 3 := by
  sorry

end NUMINAMATH_GPT_correct_calculation_l1633_163397


namespace NUMINAMATH_GPT_shire_total_population_l1633_163315

theorem shire_total_population :
  let n := 25
  let avg_pop_min := 5400
  let avg_pop_max := 5700
  let avg_pop := (avg_pop_min + avg_pop_max) / 2
  n * avg_pop = 138750 :=
by
  let n := 25
  let avg_pop_min := 5400
  let avg_pop_max := 5700
  let avg_pop := (avg_pop_min + avg_pop_max) / 2
  show n * avg_pop = 138750
  sorry

end NUMINAMATH_GPT_shire_total_population_l1633_163315


namespace NUMINAMATH_GPT_product_divisible_by_12_l1633_163378

theorem product_divisible_by_12 (a b c d : ℤ) :
  12 ∣ (b - a) * (c - a) * (d - a) * (d - c) * (d - b) * (c - b) := 
by {
  sorry
}

end NUMINAMATH_GPT_product_divisible_by_12_l1633_163378


namespace NUMINAMATH_GPT_find_equation_of_perpendicular_line_l1633_163320

noncomputable def line_through_point_perpendicular
    (A : ℝ × ℝ) (a b c : ℝ) (hA : A = (2, 3)) (hLine : a = 2 ∧ b = 1 ∧ c = -5) :
    Prop :=
  ∃ (m : ℝ) (b1 : ℝ), (m = (1 / 2)) ∧
    (b1 = 3 - m * 2) ∧
    (∀ (x y : ℝ), y = m * (x - 2) + 3 → a * x + b * y + c = 0 → x - 2 * y + 4 = 0)

theorem find_equation_of_perpendicular_line :
  line_through_point_perpendicular (2, 3) 2 1 (-5) rfl ⟨rfl, rfl, rfl⟩ :=
sorry

end NUMINAMATH_GPT_find_equation_of_perpendicular_line_l1633_163320


namespace NUMINAMATH_GPT_banana_ratio_proof_l1633_163392

-- Definitions based on conditions
def initial_bananas := 310
def bananas_left_on_tree := 100
def bananas_eaten := 70

-- Auxiliary calculations for clarity
def bananas_cut := initial_bananas - bananas_left_on_tree
def bananas_remaining := bananas_cut - bananas_eaten

-- Theorem we need to prove
theorem banana_ratio_proof :
  bananas_remaining / bananas_eaten = 2 :=
by
  sorry

end NUMINAMATH_GPT_banana_ratio_proof_l1633_163392


namespace NUMINAMATH_GPT_constant_term_expansion_l1633_163341

theorem constant_term_expansion (r : Nat) (h : 12 - 3 * r = 0) :
  (Nat.choose 6 r) * 2^r = 240 :=
sorry

end NUMINAMATH_GPT_constant_term_expansion_l1633_163341


namespace NUMINAMATH_GPT_isosceles_triangle_congruent_l1633_163327

theorem isosceles_triangle_congruent (A B C C1 : ℝ) 
(h₁ : A = B) 
(h₂ : C = C1) 
: A = B ∧ C = C1 :=
by
  sorry

end NUMINAMATH_GPT_isosceles_triangle_congruent_l1633_163327


namespace NUMINAMATH_GPT_cats_remaining_l1633_163321

theorem cats_remaining 
  (n_initial n_given_away : ℝ) 
  (h_initial : n_initial = 17.0) 
  (h_given_away : n_given_away = 14.0) : 
  (n_initial - n_given_away) = 3.0 :=
by
  rw [h_initial, h_given_away]
  norm_num

end NUMINAMATH_GPT_cats_remaining_l1633_163321


namespace NUMINAMATH_GPT_vendor_pepsi_volume_l1633_163351

theorem vendor_pepsi_volume 
    (liters_maaza : ℕ)
    (liters_sprite : ℕ)
    (num_cans : ℕ)
    (h1 : liters_maaza = 40)
    (h2 : liters_sprite = 368)
    (h3 : num_cans = 69)
    (volume_pepsi : ℕ)
    (total_volume : ℕ)
    (h4 : total_volume = liters_maaza + liters_sprite + volume_pepsi)
    (h5 : total_volume = num_cans * n)
    (h6 : 408 % num_cans = 0) :
  volume_pepsi = 75 :=
sorry

end NUMINAMATH_GPT_vendor_pepsi_volume_l1633_163351


namespace NUMINAMATH_GPT_trigonometric_identity_l1633_163309

theorem trigonometric_identity :
  7 * 6 * (1 / Real.tan (2 * Real.pi * 10 / 360) + Real.tan (2 * Real.pi * 5 / 360)) 
  = 7 * 6 * (1 / Real.sin (2 * Real.pi * 10 / 360)) := 
sorry

end NUMINAMATH_GPT_trigonometric_identity_l1633_163309


namespace NUMINAMATH_GPT_olivia_hair_length_l1633_163345

def emilys_hair_length (logan_hair : ℕ) : ℕ := logan_hair + 6
def kates_hair_length (emily_hair : ℕ) : ℕ := emily_hair / 2
def jacks_hair_length (kate_hair : ℕ) : ℕ := (7 * kate_hair) / 2
def olivias_hair_length (jack_hair : ℕ) : ℕ := (2 * jack_hair) / 3

theorem olivia_hair_length
  (logan_hair : ℕ)
  (h_logan : logan_hair = 20)
  (h_emily : emilys_hair_length logan_hair = logan_hair + 6)
  (h_emily_value : emilys_hair_length logan_hair = 26)
  (h_kate : kates_hair_length (emilys_hair_length logan_hair) = 13)
  (h_jack : jacks_hair_length (kates_hair_length (emilys_hair_length logan_hair)) = 45)
  (h_olivia : olivias_hair_length (jacks_hair_length (kates_hair_length (emilys_hair_length logan_hair))) = 30) :
  olivias_hair_length
    (jacks_hair_length
      (kates_hair_length (emilys_hair_length logan_hair))) = 30 := by
  sorry

end NUMINAMATH_GPT_olivia_hair_length_l1633_163345


namespace NUMINAMATH_GPT_original_lettuce_cost_l1633_163305

theorem original_lettuce_cost
  (original_cost: ℝ) (tomatoes_original: ℝ) (tomatoes_new: ℝ) (celery_original: ℝ) (celery_new: ℝ) (lettuce_new: ℝ)
  (delivery_tip: ℝ) (new_bill: ℝ)
  (H1: original_cost = 25)
  (H2: tomatoes_original = 0.99) (H3: tomatoes_new = 2.20)
  (H4: celery_original = 1.96) (H5: celery_new = 2.00)
  (H6: lettuce_new = 1.75)
  (H7: delivery_tip = 8.00)
  (H8: new_bill = 35) :
  ∃ (lettuce_original: ℝ), lettuce_original = 1.00 :=
by
  let tomatoes_diff := tomatoes_new - tomatoes_original
  let celery_diff := celery_new - celery_original
  let new_cost_without_lettuce := original_cost + tomatoes_diff + celery_diff
  let new_cost_excl_delivery := new_bill - delivery_tip
  have lettuce_diff := new_cost_excl_delivery - new_cost_without_lettuce
  let lettuce_original := lettuce_new - lettuce_diff
  exists lettuce_original
  sorry

end NUMINAMATH_GPT_original_lettuce_cost_l1633_163305


namespace NUMINAMATH_GPT_expression_may_not_hold_l1633_163304

theorem expression_may_not_hold (a b c : ℝ) (h : a = b) (hc : c = 0) :
  a = b → ¬ (a / c = b / c) := 
by
  intro hab
  intro h_div
  sorry

end NUMINAMATH_GPT_expression_may_not_hold_l1633_163304


namespace NUMINAMATH_GPT_total_outlets_needed_l1633_163354

-- Definitions based on conditions:
def outlets_per_room : ℕ := 6
def number_of_rooms : ℕ := 7

-- Theorem to prove the total number of outlets is 42
theorem total_outlets_needed : outlets_per_room * number_of_rooms = 42 := by
  -- Simple proof with mathematics:
  sorry

end NUMINAMATH_GPT_total_outlets_needed_l1633_163354


namespace NUMINAMATH_GPT_abs_f_x_minus_f_a_lt_l1633_163364

variable {R : Type*} [LinearOrderedField R]

def f (x : R) (c : R) := x ^ 2 - x + c

theorem abs_f_x_minus_f_a_lt (x a c : R) (h : abs (x - a) < 1) : 
  abs (f x c - f a c) < 2 * (abs a + 1) :=
by
  sorry

end NUMINAMATH_GPT_abs_f_x_minus_f_a_lt_l1633_163364


namespace NUMINAMATH_GPT_daria_multiple_pizzas_l1633_163336

variable (m : ℝ)
variable (don_pizzas : ℝ) (total_pizzas : ℝ)

axiom don_pizzas_def : don_pizzas = 80
axiom total_pizzas_def : total_pizzas = 280

theorem daria_multiple_pizzas (m : ℝ) (don_pizzas : ℝ) (total_pizzas : ℝ) 
    (h1 : don_pizzas = 80) (h2 : total_pizzas = 280) 
    (h3 : total_pizzas = don_pizzas + m * don_pizzas) : 
    m = 2.5 :=
by sorry

end NUMINAMATH_GPT_daria_multiple_pizzas_l1633_163336


namespace NUMINAMATH_GPT_rose_flyers_l1633_163359

theorem rose_flyers (total_flyers made: ℕ) (flyers_jack: ℕ) (flyers_left: ℕ) 
(h1 : total_flyers = 1236)
(h2 : flyers_jack = 120)
(h3 : flyers_left = 796)
: total_flyers - flyers_jack - flyers_left = 320 :=
by
  sorry

end NUMINAMATH_GPT_rose_flyers_l1633_163359


namespace NUMINAMATH_GPT_evaluate_expression_l1633_163310

theorem evaluate_expression :
  (4 * 6) / (12 * 14) * ((8 * 12 * 14) / (4 * 6 * 8)) = 1 := 
by 
  sorry

end NUMINAMATH_GPT_evaluate_expression_l1633_163310


namespace NUMINAMATH_GPT_john_thrice_tom_years_ago_l1633_163357

-- Define the ages of Tom and John
def T : ℕ := 16
def J : ℕ := 36

-- Condition that John will be 2 times Tom's age in 4 years
def john_twice_tom_in_4_years (J T : ℕ) : Prop := J + 4 = 2 * (T + 4)

-- The number of years ago John was thrice as old as Tom
def years_ago (J T x : ℕ) : Prop := J - x = 3 * (T - x)

-- Prove that the number of years ago John was thrice as old as Tom is 6
theorem john_thrice_tom_years_ago (h1 : john_twice_tom_in_4_years 36 16) : years_ago 36 16 6 :=
by
  -- Import initial values into the context
  unfold john_twice_tom_in_4_years at h1
  unfold years_ago
  -- Solve the steps, more details in the actual solution
  sorry

end NUMINAMATH_GPT_john_thrice_tom_years_ago_l1633_163357


namespace NUMINAMATH_GPT_simple_interest_years_l1633_163390

theorem simple_interest_years (P : ℝ) (hP : P > 0) (R : ℝ := 2.5) (SI : ℝ := P / 5) : 
  ∃ T : ℝ, P * R * T / 100 = SI ∧ T = 8 :=
by
  sorry

end NUMINAMATH_GPT_simple_interest_years_l1633_163390


namespace NUMINAMATH_GPT_max_diameter_min_diameter_l1633_163339

-- Definitions based on problem conditions
def base_diameter : ℝ := 30
def positive_tolerance : ℝ := 0.03
def negative_tolerance : ℝ := 0.04

-- The corresponding proof problem statements in Lean 4
theorem max_diameter : base_diameter + positive_tolerance = 30.03 := sorry
theorem min_diameter : base_diameter - negative_tolerance = 29.96 := sorry

end NUMINAMATH_GPT_max_diameter_min_diameter_l1633_163339


namespace NUMINAMATH_GPT_average_age_increase_39_l1633_163338

variable (n : ℕ) (A : ℝ)
noncomputable def average_age_increase (r : ℝ) : Prop :=
  (r = 7) →
  (n + 1) * (A + r) = n * A + 39 →
  (n + 1) * (A - 1) = n * A + 15 →
  r = 7

theorem average_age_increase_39 : ∀ (n : ℕ) (A : ℝ), average_age_increase n A 7 :=
by
  intros n A
  unfold average_age_increase
  intros hr h1 h2
  exact hr

end NUMINAMATH_GPT_average_age_increase_39_l1633_163338


namespace NUMINAMATH_GPT_parallel_planes_of_skew_lines_l1633_163323

variables {Plane : Type*} {Line : Type*}
variables (α β : Plane)
variables (a b : Line)

-- Conditions
def is_parallel (p1 p2 : Plane) : Prop := sorry -- Parallel planes relation
def line_in_plane (l : Line) (p : Plane) : Prop := sorry -- Line in plane relation
def line_parallel_plane (l : Line) (p : Plane) : Prop := sorry -- Line parallel to plane relation
def is_skew_lines (l1 l2 : Line) : Prop := sorry -- Skew lines relation

-- Theorem to prove
theorem parallel_planes_of_skew_lines 
  (h1 : line_in_plane a α)
  (h2 : line_in_plane b β)
  (h3 : line_parallel_plane a β)
  (h4 : line_parallel_plane b α)
  (h5 : is_skew_lines a b) :
  is_parallel α β :=
sorry

end NUMINAMATH_GPT_parallel_planes_of_skew_lines_l1633_163323


namespace NUMINAMATH_GPT_integral_log_eq_ln2_l1633_163318

theorem integral_log_eq_ln2 :
  ∫ x in (0 : ℝ)..(1 : ℝ), (1 / (x + 1)) = Real.log 2 :=
by
  sorry

end NUMINAMATH_GPT_integral_log_eq_ln2_l1633_163318


namespace NUMINAMATH_GPT_find_third_month_sale_l1633_163396

def sales_first_month : ℕ := 3435
def sales_second_month : ℕ := 3927
def sales_fourth_month : ℕ := 4230
def sales_fifth_month : ℕ := 3562
def sales_sixth_month : ℕ := 1991
def required_average_sale : ℕ := 3500

theorem find_third_month_sale (S3 : ℕ) :
  (sales_first_month + sales_second_month + S3 + sales_fourth_month + sales_fifth_month + sales_sixth_month) / 6 = required_average_sale →
  S3 = 3855 := by
  sorry

end NUMINAMATH_GPT_find_third_month_sale_l1633_163396


namespace NUMINAMATH_GPT_maciek_total_cost_l1633_163352

-- Define the cost of pretzels and the additional cost percentage for chips
def cost_pretzel : ℝ := 4
def cost_chip := cost_pretzel + (cost_pretzel * 0.75)

-- Number of packets Maciek bought for pretzels and chips
def num_pretzels : ℕ := 2
def num_chips : ℕ := 2

-- Total cost calculation
def total_cost := (cost_pretzel * num_pretzels) + (cost_chip * num_chips)

-- The final theorem statement
theorem maciek_total_cost :
  total_cost = 22 := by
  sorry

end NUMINAMATH_GPT_maciek_total_cost_l1633_163352


namespace NUMINAMATH_GPT_negation_of_P_is_exists_Q_l1633_163388

def P (x : ℝ) : Prop := x^2 - x + 3 > 0

theorem negation_of_P_is_exists_Q :
  (¬ (∀ x : ℝ, P x)) ↔ (∃ x : ℝ, ¬ P x) :=
sorry

end NUMINAMATH_GPT_negation_of_P_is_exists_Q_l1633_163388


namespace NUMINAMATH_GPT_graphs_intersect_at_three_points_l1633_163353

noncomputable def is_invertible (f : ℝ → ℝ) := ∃ (g : ℝ → ℝ), ∀ x : ℝ, f (g x) = x ∧ g (f x) = x

theorem graphs_intersect_at_three_points (f : ℝ → ℝ) (h_inv : is_invertible f) :
  ∃ (xs : Finset ℝ), (∀ x ∈ xs, f (x^2) = f (x^6)) ∧ xs.card = 3 :=
by 
  sorry

end NUMINAMATH_GPT_graphs_intersect_at_three_points_l1633_163353


namespace NUMINAMATH_GPT_part1_eq_part2_if_empty_intersection_then_a_geq_3_l1633_163358

open Set

variable {U : Type} {a : ℝ}

def universal_set : Set ℝ := univ
def A : Set ℝ := {x : ℝ | 1 < x ∧ x < 3}
def B1 (a : ℝ) : Set ℝ := {x : ℝ | x > a}
def complement_B1 (a : ℝ) : Set ℝ := {x : ℝ | x ≤ a}
def intersection_with_complement (a : ℝ) : Set ℝ := A ∩ complement_B1 a

-- Statement for part (1)
theorem part1_eq {a : ℝ} (h : a = 2) : intersection_with_complement a = {x : ℝ | 1 < x ∧ x ≤ 2} :=
by sorry

-- Statement for part (2)
theorem part2_if_empty_intersection_then_a_geq_3 
(h : A ∩ B1 a = ∅) : a ≥ 3 :=
by sorry

end NUMINAMATH_GPT_part1_eq_part2_if_empty_intersection_then_a_geq_3_l1633_163358


namespace NUMINAMATH_GPT_exponent_multiplication_correct_l1633_163337

theorem exponent_multiplication_correct (a : ℝ) : a^3 * a^4 = a^7 := by
  sorry

end NUMINAMATH_GPT_exponent_multiplication_correct_l1633_163337


namespace NUMINAMATH_GPT_abs_neg_is_2_l1633_163312

theorem abs_neg_is_2 (a : ℝ) (h1 : a < 0) (h2 : |a| = 2) : a = -2 :=
by sorry

end NUMINAMATH_GPT_abs_neg_is_2_l1633_163312


namespace NUMINAMATH_GPT_no_high_quality_triangle_exist_high_quality_quadrilateral_l1633_163316

-- Define the necessary predicate for a number being a perfect square
def is_perfect_square (n : ℕ) : Prop :=
  ∃ m : ℕ, m * m = n

-- Define the property of being a high-quality triangle
def high_quality_triangle (a b c : ℕ) : Prop :=
  is_perfect_square (a + b) ∧ is_perfect_square (b + c) ∧ is_perfect_square (c + a)

-- Define the property of non-existence of a high-quality triangle
theorem no_high_quality_triangle (a b c : ℕ) (ha : Prime a) (hb : Prime b) (hc : Prime c) : 
  ¬high_quality_triangle a b c := by sorry

-- Define the property of being a high-quality quadrilateral
def high_quality_quadrilateral (a b c d : ℕ) : Prop :=
  is_perfect_square (a + b) ∧ is_perfect_square (b + c) ∧ is_perfect_square (c + d) ∧ is_perfect_square (d + a)

-- Define the property of existence of a high-quality quadrilateral
theorem exist_high_quality_quadrilateral (a b c d : ℕ) (ha : Prime a) (hb : Prime b) (hc : Prime c) (hd : Prime d) : 
  high_quality_quadrilateral a b c d := by sorry

end NUMINAMATH_GPT_no_high_quality_triangle_exist_high_quality_quadrilateral_l1633_163316


namespace NUMINAMATH_GPT_arithmetic_sequence_geometric_sequence_added_number_l1633_163325

theorem arithmetic_sequence_geometric_sequence_added_number 
  (a : ℕ → ℤ)
  (h1 : a 1 = -8)
  (h2 : a 2 = -6)
  (h_arith : ∀ n, a n = -8 + (n-1) * 2)  -- derived from the conditions
  (x : ℤ)
  (h_geo : (-8 + x) * x = (-2 + x) * (-2 + x)) :
  x = -1 := 
sorry

end NUMINAMATH_GPT_arithmetic_sequence_geometric_sequence_added_number_l1633_163325


namespace NUMINAMATH_GPT_solve_for_y_l1633_163313

theorem solve_for_y (x y : ℤ) (h1 : x + y = 260) (h2 : x - y = 200) : y = 30 := by
  sorry

end NUMINAMATH_GPT_solve_for_y_l1633_163313


namespace NUMINAMATH_GPT_new_weight_is_77_l1633_163303

theorem new_weight_is_77 (weight_increase_per_person : ℝ) (number_of_persons : ℕ) (old_weight : ℝ) 
  (total_weight_increase : ℝ) (new_weight : ℝ) 
  (h1 : weight_increase_per_person = 1.5)
  (h2 : number_of_persons = 8)
  (h3 : old_weight = 65)
  (h4 : total_weight_increase = number_of_persons * weight_increase_per_person)
  (h5 : new_weight = old_weight + total_weight_increase) :
  new_weight = 77 :=
sorry

end NUMINAMATH_GPT_new_weight_is_77_l1633_163303


namespace NUMINAMATH_GPT_circle_equation_translation_l1633_163300

theorem circle_equation_translation (x y : ℝ) :
  x^2 + y^2 - 4 * x + 6 * y - 68 = 0 → (x - 2)^2 + (y + 3)^2 = 81 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_circle_equation_translation_l1633_163300


namespace NUMINAMATH_GPT_exists_permutation_ab_minus_cd_ge_two_l1633_163368

theorem exists_permutation_ab_minus_cd_ge_two (p q r s : ℝ) 
  (h1 : p + q + r + s = 9) 
  (h2 : p^2 + q^2 + r^2 + s^2 = 21) :
  ∃ (a b c d : ℝ), (a, b, c, d) = (p, q, r, s) ∨ (a, b, c, d) = (p, q, s, r) ∨ 
  (a, b, c, d) = (p, r, q, s) ∨ (a, b, c, d) = (p, r, s, q) ∨ 
  (a, b, c, d) = (p, s, q, r) ∨ (a, b, c, d) = (p, s, r, q) ∨ 
  (a, b, c, d) = (q, p, r, s) ∨ (a, b, c, d) = (q, p, s, r) ∨ 
  (a, b, c, d) = (q, r, p, s) ∨ (a, b, c, d) = (q, r, s, p) ∨ 
  (a, b, c, d) = (q, s, p, r) ∨ (a, b, c, d) = (q, s, r, p) ∨ 
  (a, b, c, d) = (r, p, q, s) ∨ (a, b, c, d) = (r, p, s, q) ∨ 
  (a, b, c, d) = (r, q, p, s) ∨ (a, b, c, d) = (r, q, s, p) ∨ 
  (a, b, c, d) = (r, s, p, q) ∨ (a, b, c, d) = (r, s, q, p) ∨ 
  (a, b, c, d) = (s, p, q, r) ∨ (a, b, c, d) = (s, p, r, q) ∨ 
  (a, b, c, d) = (s, q, p, r) ∨ (a, b, c, d) = (s, q, r, p) ∨ 
  (a, b, c, d) = (s, r, p, q) ∨ (a, b, c, d) = (s, r, q, p) ∧ ab - cd ≥ 2 :=
sorry

end NUMINAMATH_GPT_exists_permutation_ab_minus_cd_ge_two_l1633_163368


namespace NUMINAMATH_GPT_tan_a1_a13_eq_sqrt3_l1633_163333

-- Definition of required constants and properties of the geometric sequence
noncomputable def a (n : Nat) : ℝ := sorry -- Geometric sequence definition (abstract)

-- Given condition: a_3 * a_11 + 2 * a_7^2 = 4π
axiom geom_seq_cond : a 3 * a 11 + 2 * (a 7)^2 = 4 * Real.pi

-- Property of geometric sequence: a_3 * a_11 = a_7^2
axiom geom_seq_property : a 3 * a 11 = (a 7)^2

-- To prove: tan(a_1 * a_13) = √3
theorem tan_a1_a13_eq_sqrt3 : Real.tan (a 1 * a 13) = Real.sqrt 3 := by
  sorry

end NUMINAMATH_GPT_tan_a1_a13_eq_sqrt3_l1633_163333


namespace NUMINAMATH_GPT_problem1_problem2_l1633_163366

-- Problem 1: Solution set for x(7 - x) >= 12
theorem problem1 (x : ℝ) : x * (7 - x) ≥ 12 ↔ (3 ≤ x ∧ x ≤ 4) :=
by
  sorry

-- Problem 2: Solution set for x^2 > 2(x - 1)
theorem problem2 (x : ℝ) : x^2 > 2 * (x - 1) ↔ true :=
by
  sorry

end NUMINAMATH_GPT_problem1_problem2_l1633_163366


namespace NUMINAMATH_GPT_average_value_of_powers_l1633_163369

theorem average_value_of_powers (z : ℝ) : 
  (z^2 + 3*z^2 + 6*z^2 + 12*z^2 + 24*z^2) / 5 = 46*z^2 / 5 :=
by
  sorry

end NUMINAMATH_GPT_average_value_of_powers_l1633_163369


namespace NUMINAMATH_GPT_product_of_possible_values_of_N_l1633_163331

theorem product_of_possible_values_of_N (M L N : ℝ) (h1 : M = L + N) (h2 : M - 5 = (L + N) - 5) (h3 : L + 3 = L + 3) (h4 : |(L + N - 5) - (L + 3)| = 2) : 10 * 6 = 60 := by
  sorry

end NUMINAMATH_GPT_product_of_possible_values_of_N_l1633_163331


namespace NUMINAMATH_GPT_difference_red_white_l1633_163382

/-
Allie picked 100 wildflowers. The categories of flowers are given as below:
- 13 of the flowers were yellow and white
- 17 of the flowers were red and yellow
- 14 of the flowers were red and white
- 16 of the flowers were blue and yellow
- 9 of the flowers were blue and white
- 8 of the flowers were red, blue, and yellow
- 6 of the flowers were red, white, and blue

The goal is to define the number of flowers containing red and white, and
prove that the difference between the number of flowers containing red and 
those containing white is 3.
-/

def total_flowers : ℕ := 100
def yellow_and_white : ℕ := 13
def red_and_yellow : ℕ := 17
def red_and_white : ℕ := 14
def blue_and_yellow : ℕ := 16
def blue_and_white : ℕ := 9
def red_blue_and_yellow : ℕ := 8
def red_white_and_blue : ℕ := 6

def flowers_with_red : ℕ := red_and_yellow + red_and_white + red_blue_and_yellow + red_white_and_blue
def flowers_with_white : ℕ := yellow_and_white + red_and_white + blue_and_white + red_white_and_blue

theorem difference_red_white : flowers_with_red - flowers_with_white = 3 := by
  rw [flowers_with_red, flowers_with_white]
  sorry

end NUMINAMATH_GPT_difference_red_white_l1633_163382


namespace NUMINAMATH_GPT_system_solution_l1633_163371

theorem system_solution (x y : ℝ) :
  (x + y = 4) ∧ (2 * x - y = 2) → x = 2 ∧ y = 2 := by 
sorry

end NUMINAMATH_GPT_system_solution_l1633_163371


namespace NUMINAMATH_GPT_spa_polish_total_digits_l1633_163387

theorem spa_polish_total_digits (girls : ℕ) (digits_per_girl : ℕ) (total_digits : ℕ)
  (h1 : girls = 5) (h2 : digits_per_girl = 20) : total_digits = 100 :=
by
  sorry

end NUMINAMATH_GPT_spa_polish_total_digits_l1633_163387


namespace NUMINAMATH_GPT_handshaking_remainder_l1633_163308

-- Define number of people
def num_people := 11

-- Define N as the number of possible handshaking ways
def N : ℕ :=
sorry -- This will involve complicated combinatorial calculations

-- Define the target result to be proven
theorem handshaking_remainder : N % 1000 = 120 :=
sorry

end NUMINAMATH_GPT_handshaking_remainder_l1633_163308


namespace NUMINAMATH_GPT_find_245th_digit_in_decimal_rep_of_13_div_17_l1633_163340

-- Definition of the repeating sequence for the fractional division
def repeating_sequence_13_div_17 : List Char := ['7', '6', '4', '7', '0', '5', '8', '8', '2', '3', '5', '2', '9', '4', '1', '1']

-- Period of the repeating sequence
def period : ℕ := 16

-- Function to find the n-th digit in a repeating sequence
def nth_digit_in_repeating_sequence (seq : List Char) (period : ℕ) (n : ℕ) : Char :=
  seq.get! ((n - 1) % period)

-- Hypothesis: The repeating sequence of 13/17 and its period
axiom repeating_sequence_period : repeating_sequence_13_div_17.length = period

-- The theorem to prove
theorem find_245th_digit_in_decimal_rep_of_13_div_17 : nth_digit_in_repeating_sequence repeating_sequence_13_div_17 period 245 = '7' := 
  by
  sorry

end NUMINAMATH_GPT_find_245th_digit_in_decimal_rep_of_13_div_17_l1633_163340


namespace NUMINAMATH_GPT_product_of_possible_values_l1633_163343

theorem product_of_possible_values :
  (∀ x : ℝ, abs (18 / x + 4) = 3 → x = -18 ∨ x = -18 / 7) →
  (∀ x1 x2 : ℝ, x1 = -18 → x2 = -18 / 7 → x1 * x2 = 324 / 7) :=
by
  intros h x1 x2 hx1 hx2
  rw [hx1, hx2]
  norm_num

end NUMINAMATH_GPT_product_of_possible_values_l1633_163343


namespace NUMINAMATH_GPT_average_rst_l1633_163363

variable (r s t : ℝ)

theorem average_rst
  (h : (4 / 3) * (r + s + t) = 12) :
  (r + s + t) / 3 = 3 :=
sorry

end NUMINAMATH_GPT_average_rst_l1633_163363


namespace NUMINAMATH_GPT_Sam_scored_points_l1633_163394

theorem Sam_scored_points (total_points friend_points S: ℕ) (h1: friend_points = 12) (h2: total_points = 87) (h3: total_points = S + friend_points) : S = 75 :=
by
  sorry

end NUMINAMATH_GPT_Sam_scored_points_l1633_163394


namespace NUMINAMATH_GPT_fifth_number_l1633_163370

def sequence_sum (a b : ℕ) : ℕ :=
  a + b + (a + b) + (a + 2 * b) + (2 * a + 3 * b) + (3 * a + 5 * b)

theorem fifth_number (a b : ℕ) (h : sequence_sum a b = 2008) : 2 * a + 3 * b = 502 := by
  sorry

end NUMINAMATH_GPT_fifth_number_l1633_163370


namespace NUMINAMATH_GPT_problem_statement_l1633_163384

def nabla (a b : ℕ) : ℕ := 3 + b ^ a

theorem problem_statement : nabla (nabla 2 3) 4 = 16777219 := by
  sorry

end NUMINAMATH_GPT_problem_statement_l1633_163384


namespace NUMINAMATH_GPT_fraction_inequality_l1633_163317

theorem fraction_inequality (a b c : ℝ) : 
  (a / (a + 2 * b + c)) + (b / (a + b + 2 * c)) + (c / (2 * a + b + c)) ≥ 3 / 4 := 
by
  sorry

end NUMINAMATH_GPT_fraction_inequality_l1633_163317
