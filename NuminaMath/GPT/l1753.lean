import Mathlib

namespace NUMINAMATH_GPT_original_number_is_45_l1753_175386

theorem original_number_is_45 (x y : ℕ) (h1 : x + y = 9) (h2 : 10 * y + x = 10 * x + y + 9) : 10 * x + y = 45 := by
  sorry

end NUMINAMATH_GPT_original_number_is_45_l1753_175386


namespace NUMINAMATH_GPT_exists_bound_for_expression_l1753_175342

theorem exists_bound_for_expression :
  ∃ (C : ℝ), (∀ (k : ℤ), abs ((k^8 - 2*k + 1 : ℤ) / (k^4 - 3 : ℤ)) < C) := 
sorry

end NUMINAMATH_GPT_exists_bound_for_expression_l1753_175342


namespace NUMINAMATH_GPT_calc_expression_l1753_175379

theorem calc_expression : 
  |1 - Real.sqrt 2| - Real.sqrt 8 + (Real.sqrt 2 - 1)^0 = -Real.sqrt 2 :=
by
  sorry

end NUMINAMATH_GPT_calc_expression_l1753_175379


namespace NUMINAMATH_GPT_max_overlap_l1753_175378

variable (A : Type) [Fintype A] [DecidableEq A]
variable (P1 P2 : A → Prop)

theorem max_overlap (hP1 : ∃ X : Finset A, (X.card : ℝ) / Fintype.card A = 0.20 ∧ ∀ a ∈ X, P1 a)
                    (hP2 : ∃ Y : Finset A, (Y.card : ℝ) / Fintype.card A = 0.70 ∧ ∀ a ∈ Y, P2 a) :
  ∃ Z : Finset A, (Z.card : ℝ) / Fintype.card A = 0.20 ∧ ∀ a ∈ Z, P1 a ∧ P2 a :=
sorry

end NUMINAMATH_GPT_max_overlap_l1753_175378


namespace NUMINAMATH_GPT_i_pow_2006_l1753_175380

-- Definitions based on given conditions
def i : ℂ := Complex.I

-- Cyclic properties of i (imaginary unit)
axiom i_pow_1 : i^1 = i
axiom i_pow_2 : i^2 = -1
axiom i_pow_3 : i^3 = -i
axiom i_pow_4 : i^4 = 1
axiom i_pow_5 : i^5 = i

-- The proof statement
theorem i_pow_2006 : (i^2006 = -1) :=
by
  sorry

end NUMINAMATH_GPT_i_pow_2006_l1753_175380


namespace NUMINAMATH_GPT_exist_pairs_sum_and_diff_l1753_175356

theorem exist_pairs_sum_and_diff (N : ℕ) : ∃ a b c d : ℕ, 
  (a + b = c + d) ∧ (a * b + N = c * d ∨ a * b = c * d + N) := sorry

end NUMINAMATH_GPT_exist_pairs_sum_and_diff_l1753_175356


namespace NUMINAMATH_GPT_farm_field_area_l1753_175347

variable (A D : ℕ)

theorem farm_field_area
  (h1 : 160 * D = A)
  (h2 : 85 * (D + 2) + 40 = A) :
  A = 480 :=
by
  sorry

end NUMINAMATH_GPT_farm_field_area_l1753_175347


namespace NUMINAMATH_GPT_find_values_l1753_175335

theorem find_values (h t u : ℕ) 
  (h0 : u = h - 5) 
  (h1 : (h * 100 + t * 10 + u) - (h * 100 + u * 10 + t) = 96)
  (hu : h < 10 ∧ t < 10 ∧ u < 10) :
  h = 5 ∧ t = 9 ∧ u = 0 :=
by 
  sorry

end NUMINAMATH_GPT_find_values_l1753_175335


namespace NUMINAMATH_GPT_number_of_straight_A_students_l1753_175367

-- Define the initial conditions and numbers
variables {x y : ℕ}

-- Define the initial student count and conditions on percentages
def initial_student_count := 25
def new_student_count := 7
def total_student_count := initial_student_count + new_student_count
def initial_percentage (x : ℕ) := (x : ℚ) / initial_student_count * 100
def new_percentage (x y : ℕ) := ((x + y : ℚ) / total_student_count) * 100

theorem number_of_straight_A_students
  (x y : ℕ)
  (h : initial_percentage x + 10 = new_percentage x y) :
  (x + y = 16) :=
sorry

end NUMINAMATH_GPT_number_of_straight_A_students_l1753_175367


namespace NUMINAMATH_GPT_find_number_l1753_175338

theorem find_number (x : ℝ) (h : (3.242 * 16) / x = 0.051871999999999995) : x = 1000 :=
by
  sorry

end NUMINAMATH_GPT_find_number_l1753_175338


namespace NUMINAMATH_GPT_percentage_of_students_who_received_certificates_l1753_175309

theorem percentage_of_students_who_received_certificates
  (total_boys : ℕ)
  (total_girls : ℕ)
  (perc_boys_certificates : ℝ)
  (perc_girls_certificates : ℝ)
  (h1 : total_boys = 30)
  (h2 : total_girls = 20)
  (h3 : perc_boys_certificates = 0.1)
  (h4 : perc_girls_certificates = 0.2)
  : (3 + 4) / (30 + 20) * 100 = 14 :=
by
  sorry

end NUMINAMATH_GPT_percentage_of_students_who_received_certificates_l1753_175309


namespace NUMINAMATH_GPT_range_of_a_l1753_175395

open Set

theorem range_of_a (a : ℝ) (M N : Set ℝ) (hM : ∀ x, x ∈ M ↔ x < 2) (hN : ∀ x, x ∈ N ↔ x < a) (hMN : M ⊆ N) : 2 ≤ a :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_l1753_175395


namespace NUMINAMATH_GPT_existential_proposition_l1753_175369

theorem existential_proposition :
  (∃ x y : ℝ, x + y > 1) ∧ (∀ P : Prop, (∃ x y : ℝ, x + y > 1 → P) → P) :=
sorry

end NUMINAMATH_GPT_existential_proposition_l1753_175369


namespace NUMINAMATH_GPT_units_digit_of_147_pow_is_7_some_exponent_units_digit_l1753_175353

theorem units_digit_of_147_pow_is_7 (n : ℕ) : (147 ^ 25) % 10 = 7 % 10 :=
by
  sorry

theorem some_exponent_units_digit (n : ℕ) (hn : n % 4 = 2) : ((147 ^ 25) ^ n) % 10 = 9 :=
by
  have base_units_digit := units_digit_of_147_pow_is_7 25
  sorry

end NUMINAMATH_GPT_units_digit_of_147_pow_is_7_some_exponent_units_digit_l1753_175353


namespace NUMINAMATH_GPT_tomatoes_left_l1753_175357

theorem tomatoes_left (initial_tomatoes : ℕ) (birds : ℕ) (fraction : ℕ) (E1 : initial_tomatoes = 21) 
  (E2 : birds = 2) (E3 : fraction = 3) : 
  initial_tomatoes - initial_tomatoes / fraction = 14 :=
by 
  sorry

end NUMINAMATH_GPT_tomatoes_left_l1753_175357


namespace NUMINAMATH_GPT_todd_ate_cupcakes_l1753_175393

def total_cupcakes_baked := 68
def packages := 6
def cupcakes_per_package := 6
def total_packaged_cupcakes := packages * cupcakes_per_package
def remaining_cupcakes := total_cupcakes_baked - total_packaged_cupcakes

theorem todd_ate_cupcakes : total_cupcakes_baked - remaining_cupcakes = 36 := by
  sorry

end NUMINAMATH_GPT_todd_ate_cupcakes_l1753_175393


namespace NUMINAMATH_GPT_find_complex_number_purely_imaginary_l1753_175334

theorem find_complex_number_purely_imaginary :
  ∃ z : ℂ, (∃ b : ℝ, b ≠ 0 ∧ z = 1 + b * I) ∧ (∀ a b : ℝ, z = a + b * I → a^2 - b^2 + 3 = 0) :=
by
  -- Proof will go here
  sorry

end NUMINAMATH_GPT_find_complex_number_purely_imaginary_l1753_175334


namespace NUMINAMATH_GPT_canoe_row_probability_l1753_175345

theorem canoe_row_probability :
  let p_left_works := 3 / 5
  let p_right_works := 3 / 5
  let p_left_breaks := 1 - p_left_works
  let p_right_breaks := 1 - p_right_works
  let p_can_still_row := (p_left_works * p_right_works) + (p_left_works * p_right_breaks) + (p_left_breaks * p_right_works)
  p_can_still_row = 21 / 25 :=
by
  sorry

end NUMINAMATH_GPT_canoe_row_probability_l1753_175345


namespace NUMINAMATH_GPT_find_y_values_l1753_175313

variable (x y : ℝ)

theorem find_y_values 
    (h1 : 3 * x^2 + 9 * x + 4 * y - 2 = 0)
    (h2 : 3 * x + 2 * y - 6 = 0) : 
    y^2 - 13 * y + 26 = 0 := by
  sorry

end NUMINAMATH_GPT_find_y_values_l1753_175313


namespace NUMINAMATH_GPT_minimize_b_plus_4c_l1753_175385

noncomputable def triangle := Type

variable {ABC : triangle}
variable (a b c : ℝ) -- sides of the triangle
variable (BAC : ℝ) -- angle BAC
variable (D : triangle → ℝ) -- angle bisector intersecting BC at D
variable (AD : ℝ) -- length of AD
variable (min_bc : ℝ) -- minimum value of b + 4c

-- Conditions
variable (h1 : BAC = 120)
variable (h2 : D ABC = 1)
variable (h3 : AD = 1)

-- Proof statement
theorem minimize_b_plus_4c (h1 : BAC = 120) (h2 : D ABC = 1) (h3 : AD = 1) : min_bc = 9 := 
sorry

end NUMINAMATH_GPT_minimize_b_plus_4c_l1753_175385


namespace NUMINAMATH_GPT_sum_m_n_l1753_175354

-- Define the conditions and the result

def probabilityOfNo3x3RedSquare : ℚ :=
  65408 / 65536

def gcd_65408_65536 := Nat.gcd 65408 65536

def simplifiedProbability : ℚ :=
  probabilityOfNo3x3RedSquare / gcd_65408_65536

def m : ℕ :=
  511

def n : ℕ :=
  512

theorem sum_m_n : m + n = 1023 := by
  sorry

end NUMINAMATH_GPT_sum_m_n_l1753_175354


namespace NUMINAMATH_GPT_abs_inequality_solution_l1753_175363

theorem abs_inequality_solution (x : ℝ) : |2 * x - 5| > 1 ↔ x < 2 ∨ x > 3 := sorry

end NUMINAMATH_GPT_abs_inequality_solution_l1753_175363


namespace NUMINAMATH_GPT_percentage_of_democrats_l1753_175359

variable (D R : ℝ)

theorem percentage_of_democrats (h1 : D + R = 100) (h2 : 0.75 * D + 0.20 * R = 53) :
  D = 60 :=
by
  sorry

end NUMINAMATH_GPT_percentage_of_democrats_l1753_175359


namespace NUMINAMATH_GPT_evaluate_expression_l1753_175394

variable (m n p q s : ℝ)

theorem evaluate_expression :
  m / (n - (p + q * s)) = m / (n - p - q * s) :=
by
  sorry

end NUMINAMATH_GPT_evaluate_expression_l1753_175394


namespace NUMINAMATH_GPT_apple_price_33kg_l1753_175321

theorem apple_price_33kg
  (l q : ℝ)
  (h1 : 10 * l = 3.62)
  (h2 : 30 * l + 6 * q = 12.48) :
  30 * l + 3 * q = 11.67 :=
by
  sorry

end NUMINAMATH_GPT_apple_price_33kg_l1753_175321


namespace NUMINAMATH_GPT_find_x_value_l1753_175364

-- Let's define the conditions
def equation (x y : ℝ) : Prop := x^2 - 4 * x + y = 0
def y_value : ℝ := 4

-- Define the theorem which states that x = 2 satisfies the conditions
theorem find_x_value (x : ℝ) (h : equation x y_value) : x = 2 :=
by
  sorry

end NUMINAMATH_GPT_find_x_value_l1753_175364


namespace NUMINAMATH_GPT_number_of_buses_l1753_175322

-- Definitions based on the given conditions
def vans : ℕ := 6
def people_per_van : ℕ := 6
def people_per_bus : ℕ := 18
def total_people : ℕ := 180

-- Theorem to prove the number of buses
theorem number_of_buses : 
  ∃ buses : ℕ, buses = (total_people - (vans * people_per_van)) / people_per_bus ∧ buses = 8 :=
by
  sorry

end NUMINAMATH_GPT_number_of_buses_l1753_175322


namespace NUMINAMATH_GPT_exists_100_digit_number_divisible_by_sum_of_digits_l1753_175336

-- Definitions
def is_100_digit_number (n : ℕ) : Prop :=
  10^99 ≤ n ∧ n < 10^100

def no_zero_digits (n : ℕ) : Prop :=
  ∀ d ∈ n.digits 10, d ≠ 0

def sum_of_digits (n : ℕ) : ℕ :=
  (n.digits 10).sum

def is_divisible_by_sum_of_digits (n : ℕ) : Prop :=
  n % sum_of_digits n = 0

-- Main theorem statement
theorem exists_100_digit_number_divisible_by_sum_of_digits :
  ∃ n : ℕ, is_100_digit_number n ∧ no_zero_digits n ∧ is_divisible_by_sum_of_digits n :=
sorry

end NUMINAMATH_GPT_exists_100_digit_number_divisible_by_sum_of_digits_l1753_175336


namespace NUMINAMATH_GPT_A_intersection_B_complement_l1753_175305

noncomputable
def universal_set : Set ℝ := Set.univ

def set_A : Set ℝ := {x | x > 1}

def set_B : Set ℝ := {y | -1 < y ∧ y < 2}

def B_complement : Set ℝ := {y | y <= -1 ∨ y >= 2}

def intersection : Set ℝ := {x | x >= 2}

theorem A_intersection_B_complement :
  (set_A ∩ B_complement) = intersection :=
  sorry

end NUMINAMATH_GPT_A_intersection_B_complement_l1753_175305


namespace NUMINAMATH_GPT_initial_big_bottles_l1753_175384

theorem initial_big_bottles (B : ℝ)
  (initial_small : ℝ := 6000)
  (sold_small : ℝ := 0.11)
  (sold_big : ℝ := 0.12)
  (remaining_total : ℝ := 18540) :
  (initial_small * (1 - sold_small) + B * (1 - sold_big) = remaining_total) → B = 15000 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_initial_big_bottles_l1753_175384


namespace NUMINAMATH_GPT_car_speed_first_hour_l1753_175351

-- Definitions based on the conditions in the problem
noncomputable def speed_second_hour := 30
noncomputable def average_speed := 45
noncomputable def total_time := 2

-- Assertion based on the problem's question and correct answer
theorem car_speed_first_hour: ∃ (x : ℕ), (average_speed * total_time) = (x + speed_second_hour) ∧ x = 60 :=
by
  sorry

end NUMINAMATH_GPT_car_speed_first_hour_l1753_175351


namespace NUMINAMATH_GPT_inequality_incorrect_l1753_175324

theorem inequality_incorrect (a b : ℝ) (h : a > b) : ¬(-a + 2 > -b + 2) :=
by
  sorry

end NUMINAMATH_GPT_inequality_incorrect_l1753_175324


namespace NUMINAMATH_GPT_solution_correct_l1753_175346

noncomputable def quadratic_inequality_solution (x : ℝ) : Prop :=
  x^2 - 36 * x + 320 ≤ 16

theorem solution_correct (x : ℝ) : quadratic_inequality_solution x ↔ 16 ≤ x ∧ x ≤ 19 :=
by sorry

end NUMINAMATH_GPT_solution_correct_l1753_175346


namespace NUMINAMATH_GPT_polynomial_p_l1753_175316

variable {a b c : ℝ}

theorem polynomial_p (a b c : ℝ) : 
  a * (b - c)^2 + b * (c - a)^2 + c * (a - b)^2 = (a - b) * (b - c) * (c - a) * 2 :=
by
  sorry

end NUMINAMATH_GPT_polynomial_p_l1753_175316


namespace NUMINAMATH_GPT_trig_expression_l1753_175383

theorem trig_expression (α : ℝ) (h : Real.tan α = 2) : 
  (Real.sin α + Real.cos α) / (Real.sin α - Real.cos α) + Real.cos α ^ 2 = 16 / 5 := 
  sorry

end NUMINAMATH_GPT_trig_expression_l1753_175383


namespace NUMINAMATH_GPT_even_function_l1753_175310

-- Definition of f and F with the given conditions
variable (f : ℝ → ℝ)
variable (a : ℝ)
variable (x : ℝ)

-- Condition that x is in the interval (-a, a)
def in_interval (a x : ℝ) : Prop := x > -a ∧ x < a

-- Definition of F(x)
def F (x : ℝ) : ℝ := f x + f (-x)

-- The proposition that we want to prove
theorem even_function (h : in_interval a x) : F f x = F f (-x) :=
by
  unfold F
  sorry

end NUMINAMATH_GPT_even_function_l1753_175310


namespace NUMINAMATH_GPT_Adam_smiley_count_l1753_175301

theorem Adam_smiley_count :
  ∃ (adam mojmir petr pavel : ℕ), adam + mojmir + petr + pavel = 52 ∧
  petr + pavel = 33 ∧ adam >= 1 ∧ mojmir >= 1 ∧ petr >= 1 ∧ pavel >= 1 ∧
  mojmir > max petr pavel ∧ adam = 1 :=
by
  sorry

end NUMINAMATH_GPT_Adam_smiley_count_l1753_175301


namespace NUMINAMATH_GPT_tetrahedron_BC_squared_l1753_175312

theorem tetrahedron_BC_squared (AB AC BC R r : ℝ) 
  (h1 : AB = 1) 
  (h2 : AC = 1) 
  (h3 : 1 ≤ BC) 
  (h4 : R = 4 * r) 
  (concentric : AB = AC ∧ R > 0 ∧ r > 0) :
  BC^2 = 1 + Real.sqrt (7 / 15) := 
by 
sorry

end NUMINAMATH_GPT_tetrahedron_BC_squared_l1753_175312


namespace NUMINAMATH_GPT_greatest_integer_equality_l1753_175304

theorem greatest_integer_equality (m : ℝ) (h : m ≥ 3) :
  Int.floor ((m * (m + 1)) / (2 * (2 * m - 1))) = Int.floor ((m + 1) / 4) :=
  sorry

end NUMINAMATH_GPT_greatest_integer_equality_l1753_175304


namespace NUMINAMATH_GPT_simplify_expression_l1753_175331

variables (x y : ℝ)

theorem simplify_expression :
  (3 * x)^4 + (4 * x) * (x^3) + (5 * y)^2 = 85 * x^4 + 25 * y^2 :=
by
  sorry

end NUMINAMATH_GPT_simplify_expression_l1753_175331


namespace NUMINAMATH_GPT_pow_expression_eq_l1753_175381

theorem pow_expression_eq : (-3)^(4^2) + 2^(3^2) = 43047233 := by
  sorry

end NUMINAMATH_GPT_pow_expression_eq_l1753_175381


namespace NUMINAMATH_GPT_person_a_work_days_l1753_175320

theorem person_a_work_days (x : ℝ) :
  (2 * (1 / x + 1 / 45) = 1 / 9) → (x = 30) :=
by
  sorry

end NUMINAMATH_GPT_person_a_work_days_l1753_175320


namespace NUMINAMATH_GPT_task_completion_time_l1753_175390

theorem task_completion_time (A B : ℝ) : 
  (14 * A / 80 + 10 * B / 96) = (20 * (A + B)) →
  (1 / (14 * A / 80 + 10 * B / 96)) = 480 / (84 * A + 50 * B) :=
by
  intros h
  sorry

end NUMINAMATH_GPT_task_completion_time_l1753_175390


namespace NUMINAMATH_GPT_Elaine_rent_increase_l1753_175343

noncomputable def Elaine_rent_percent (E: ℝ) : ℝ :=
  let last_year_rent := 0.20 * E
  let this_year_earnings := 1.25 * E
  let this_year_rent := 0.30 * this_year_earnings
  let ratio := (this_year_rent / last_year_rent) * 100
  ratio

theorem Elaine_rent_increase (E : ℝ) : Elaine_rent_percent E = 187.5 :=
by 
  -- The proof would go here.
  sorry

end NUMINAMATH_GPT_Elaine_rent_increase_l1753_175343


namespace NUMINAMATH_GPT_sum_of_possible_values_l1753_175368

theorem sum_of_possible_values (x y : ℝ) (h : x * y - x / y^3 - y / x^3 = 2) :
  (x - 2) * (y - 2) = 6 ∨ (x - 2) * (y - 2) = 9 →
  (if (x - 2) * (y - 2) = 6 then 6 else 0) + (if (x - 2) * (y - 2) = 9 then 9 else 0) = 15 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_possible_values_l1753_175368


namespace NUMINAMATH_GPT_parallel_lines_l1753_175398

theorem parallel_lines (a : ℝ) :
  (∀ x y, x + a^2 * y + 6 = 0 → (a - 2) * x + 3 * a * y + 2 * a = 0) ↔ (a = 0 ∨ a = -1) :=
by
  sorry

end NUMINAMATH_GPT_parallel_lines_l1753_175398


namespace NUMINAMATH_GPT_razorback_tshirt_sales_l1753_175373

theorem razorback_tshirt_sales 
  (price_per_tshirt : ℕ) (total_money_made : ℕ)
  (h1 : price_per_tshirt = 16) (h2 : total_money_made = 720) :
  total_money_made / price_per_tshirt = 45 :=
by
  sorry

end NUMINAMATH_GPT_razorback_tshirt_sales_l1753_175373


namespace NUMINAMATH_GPT_gcd_256_180_600_l1753_175315

theorem gcd_256_180_600 : Nat.gcd (Nat.gcd 256 180) 600 = 4 :=
by
  -- This proof is marked as 'sorry' to indicate it is a place holder
  sorry

end NUMINAMATH_GPT_gcd_256_180_600_l1753_175315


namespace NUMINAMATH_GPT_area_of_enclosed_figure_l1753_175344

theorem area_of_enclosed_figure:
  ∫ (x : ℝ) in (1/2)..2, x⁻¹ = 2 * Real.log 2 :=
by
  sorry

end NUMINAMATH_GPT_area_of_enclosed_figure_l1753_175344


namespace NUMINAMATH_GPT_ball_in_78th_position_is_green_l1753_175300

-- Definition of colors in the sequence
inductive Color
| red
| yellow
| green
| blue
| violet

open Color

-- Function to compute the color of a ball at a given position within a cycle
def ball_color (n : Nat) : Color :=
  match n % 5 with
  | 0 => red    -- 78 % 5 == 3, hence 3 + 1 == 4 ==> Using 0 for red to 4 for violet
  | 1 => yellow
  | 2 => green
  | 3 => blue
  | 4 => violet
  | _ => red  -- default case, should not be reached

-- Theorem stating the desired proof problem
theorem ball_in_78th_position_is_green : ball_color 78 = green :=
by
  sorry

end NUMINAMATH_GPT_ball_in_78th_position_is_green_l1753_175300


namespace NUMINAMATH_GPT_car_speed_l1753_175366

theorem car_speed (v : ℝ) (h1 : 1 / 900 * 3600 = 4) (h2 : 1 / v * 3600 = 6) : v = 600 :=
by
  sorry

end NUMINAMATH_GPT_car_speed_l1753_175366


namespace NUMINAMATH_GPT_waffle_bowl_more_scoops_l1753_175303

-- Definitions based on conditions
def single_cone_scoops : ℕ := 1
def banana_split_scoops : ℕ := 3 * single_cone_scoops
def double_cone_scoops : ℕ := 2 * single_cone_scoops
def total_scoops : ℕ := 10
def remaining_scoops : ℕ := total_scoops - (banana_split_scoops + single_cone_scoops + double_cone_scoops)

-- Question: Prove that the waffle bowl has 1 more scoop than the banana split
theorem waffle_bowl_more_scoops : remaining_scoops - banana_split_scoops = 1 := by
  have h1 : single_cone_scoops = 1 := rfl
  have h2 : banana_split_scoops = 3 * single_cone_scoops := rfl
  have h3 : double_cone_scoops = 2 * single_cone_scoops := rfl
  have h4 : total_scoops = 10 := rfl
  have h5 : remaining_scoops = total_scoops - (banana_split_scoops + single_cone_scoops + double_cone_scoops) := rfl
  sorry

end NUMINAMATH_GPT_waffle_bowl_more_scoops_l1753_175303


namespace NUMINAMATH_GPT_club_members_after_four_years_l1753_175341

theorem club_members_after_four_years
  (b : ℕ → ℕ)
  (h_initial : b 0 = 20)
  (h_recursive : ∀ k, b (k + 1) = 3 * (b k) - 10) :
  b 4 = 1220 :=
sorry

end NUMINAMATH_GPT_club_members_after_four_years_l1753_175341


namespace NUMINAMATH_GPT_sin_15_deg_eq_l1753_175325

theorem sin_15_deg_eq : 
  Real.sin (15 * Real.pi / 180) = (Real.sqrt 6 - Real.sqrt 2) / 4 := 
by
  -- conditions
  have h1 : Real.sin (45 * Real.pi / 180) = Real.sqrt 2 / 2 := by sorry
  have h2 : Real.cos (45 * Real.pi / 180) = Real.sqrt 2 / 2 := by sorry
  have h3 : Real.sin (30 * Real.pi / 180) = 1 / 2 := by sorry
  have h4 : Real.cos (30 * Real.pi / 180) = Real.sqrt 3 / 2 := by sorry
  
  -- proof
  sorry

end NUMINAMATH_GPT_sin_15_deg_eq_l1753_175325


namespace NUMINAMATH_GPT_cheenu_time_difference_l1753_175382

-- Define the conditions in terms of Cheenu's activities

variable (boy_run_distance : ℕ) (boy_run_time : ℕ)
variable (midage_bike_distance : ℕ) (midage_bike_time : ℕ)
variable (old_walk_distance : ℕ) (old_walk_time : ℕ)

-- Define the problem with these variables
theorem cheenu_time_difference:
    boy_run_distance = 20 ∧ boy_run_time = 240 ∧
    midage_bike_distance = 30 ∧ midage_bike_time = 120 ∧
    old_walk_distance = 8 ∧ old_walk_time = 240 →
    (old_walk_time / old_walk_distance - midage_bike_time / midage_bike_distance) = 26 := by
    sorry

end NUMINAMATH_GPT_cheenu_time_difference_l1753_175382


namespace NUMINAMATH_GPT_trigonometric_identity_l1753_175397

open Real

theorem trigonometric_identity (θ : ℝ) (h₁ : 0 < θ ∧ θ < π/2) (h₂ : cos θ = sqrt 10 / 10) :
  (cos (2 * θ) / (sin (2 * θ) + (cos θ)^2)) = -8 / 7 := 
sorry

end NUMINAMATH_GPT_trigonometric_identity_l1753_175397


namespace NUMINAMATH_GPT_parabola_focus_line_ratio_l1753_175339

noncomputable def ratio_AF_BF : ℝ := (Real.sqrt 5 + 3) / 2

theorem parabola_focus_line_ratio :
  ∀ (F A B : ℝ × ℝ), 
    F = (1, 0) ∧ 
    (A.2 = 2 * A.1 - 2 ∧ A.2^2 = 4 * A.1 ) ∧ 
    (B.2 = 2 * B.1 - 2 ∧ B.2^2 = 4 * B.1) ∧ 
    A.2 > 0 -> 
  |(A.1 - F.1) / (B.1 - F.1)| = ratio_AF_BF :=
by
  sorry

end NUMINAMATH_GPT_parabola_focus_line_ratio_l1753_175339


namespace NUMINAMATH_GPT_roots_cubic_eq_sum_fraction_l1753_175302

theorem roots_cubic_eq_sum_fraction (p q r : ℝ)
  (h1 : p + q + r = 8)
  (h2 : p * q + p * r + q * r = 10)
  (h3 : p * q * r = 3) :
  p / (q * r + 2) + q / (p * r + 2) + r / (p * q + 2) = 8 / 69 := 
sorry

end NUMINAMATH_GPT_roots_cubic_eq_sum_fraction_l1753_175302


namespace NUMINAMATH_GPT_range_of_a_l1753_175374

variable {R : Type*} [LinearOrderedField R]

def setA (a : R) : Set R := {x | x^2 - 2*x + a ≤ 0}

def setB : Set R := {x | x^2 - 3*x + 2 ≤ 0}

theorem range_of_a (a : R) (h : setB ⊆ setA a) : a ≤ 0 := sorry

end NUMINAMATH_GPT_range_of_a_l1753_175374


namespace NUMINAMATH_GPT_picnic_weather_condition_l1753_175365

variables (P Q : Prop)

theorem picnic_weather_condition (h : ¬P → ¬Q) : Q → P := 
by sorry

end NUMINAMATH_GPT_picnic_weather_condition_l1753_175365


namespace NUMINAMATH_GPT_find_intersection_point_l1753_175396

-- Define the problem conditions and question in Lean
theorem find_intersection_point 
  (slope_l1 : ℝ) (slope_l2 : ℝ) (p : ℝ × ℝ) (P : ℝ × ℝ)
  (h_l1_slope : slope_l1 = 2) 
  (h_parallel : slope_l1 = slope_l2)
  (h_passes_through : p = (-1, 1)) :
  P = (0, 3) := sorry

end NUMINAMATH_GPT_find_intersection_point_l1753_175396


namespace NUMINAMATH_GPT_length_of_segment_l1753_175376

theorem length_of_segment (x : ℤ) (hx : |x - 3| = 4) : 
  let a := 7
  let b := -1
  a - b = 8 := by
    sorry

end NUMINAMATH_GPT_length_of_segment_l1753_175376


namespace NUMINAMATH_GPT_lisa_max_non_a_quizzes_l1753_175333

def lisa_goal : ℕ := 34
def quizzes_total : ℕ := 40
def quizzes_taken_first : ℕ := 25
def quizzes_with_a_first : ℕ := 20
def remaining_quizzes : ℕ := quizzes_total - quizzes_taken_first
def additional_a_needed : ℕ := lisa_goal - quizzes_with_a_first

theorem lisa_max_non_a_quizzes : 
  additional_a_needed ≤ remaining_quizzes → 
  remaining_quizzes - additional_a_needed ≤ 1 :=
by
  sorry

end NUMINAMATH_GPT_lisa_max_non_a_quizzes_l1753_175333


namespace NUMINAMATH_GPT_calculate_neg2_add3_l1753_175328

theorem calculate_neg2_add3 : (-2) + 3 = 1 :=
  sorry

end NUMINAMATH_GPT_calculate_neg2_add3_l1753_175328


namespace NUMINAMATH_GPT_prime_p_in_range_l1753_175318

theorem prime_p_in_range (p : ℕ) (prime_p : Nat.Prime p) 
    (h : ∃ a b : ℤ, a * b = -530 * p ∧ a + b = p) : 43 < p ∧ p ≤ 53 := 
sorry

end NUMINAMATH_GPT_prime_p_in_range_l1753_175318


namespace NUMINAMATH_GPT_largest_non_sum_217_l1753_175362

def is_prime (n : ℕ) : Prop := sorry -- This would be some definition of primality

noncomputable def largest_non_sum_of_composite (n : ℕ) : Prop :=
  ∀ (a b : ℕ), 0 ≤ b ∧ b < 30 ∧ (∀ i : ℕ, i < a → is_prime (b + 30 * i)) →
  n ≤ 30 * a + b

theorem largest_non_sum_217 : ∃ n, largest_non_sum_of_composite n ∧ n = 217 := sorry

end NUMINAMATH_GPT_largest_non_sum_217_l1753_175362


namespace NUMINAMATH_GPT_find_n_l1753_175317

theorem find_n : ∃ n : ℕ, 2^7 * 3^3 * 5 * n = Nat.factorial 12 ∧ n = 27720 :=
by
  use 27720
  have h1 : 2^7 * 3^3 * 5 * 27720 = Nat.factorial 12 :=
  sorry -- This will be the place to prove the given equation eventually.
  exact ⟨h1, rfl⟩

end NUMINAMATH_GPT_find_n_l1753_175317


namespace NUMINAMATH_GPT_quadratic_inequality_false_iff_l1753_175308

open Real

theorem quadratic_inequality_false_iff (a : ℝ) : 
  (¬ ∃ x : ℝ, x^2 + (a - 1) * x + 1 ≤ 0) ↔ -1 < a ∧ a < 3 :=
sorry

end NUMINAMATH_GPT_quadratic_inequality_false_iff_l1753_175308


namespace NUMINAMATH_GPT_max_squares_on_checkerboard_l1753_175319

theorem max_squares_on_checkerboard (n : ℕ) (h1 : n = 7) (h2 : ∀ s : ℕ, s = 2) : ∃ max_squares : ℕ, max_squares = 18 := sorry

end NUMINAMATH_GPT_max_squares_on_checkerboard_l1753_175319


namespace NUMINAMATH_GPT_transform_circle_to_ellipse_l1753_175388

theorem transform_circle_to_ellipse (x y x'' y'' : ℝ) (h_circle : x^2 + y^2 = 1)
  (hx_trans : x = x'' / 2) (hy_trans : y = y'' / 3) :
  (x''^2 / 4) + (y''^2 / 9) = 1 :=
by {
  sorry
}

end NUMINAMATH_GPT_transform_circle_to_ellipse_l1753_175388


namespace NUMINAMATH_GPT_ratio_e_to_f_l1753_175358

theorem ratio_e_to_f {a b c d e f : ℝ}
  (h1 : a / b = 1 / 3)
  (h2 : b / c = 2)
  (h3 : c / d = 1 / 2)
  (h4 : d / e = 3)
  (h5 : a * b * c / (d * e * f) = 0.75) :
  e / f = 0.5 :=
sorry

end NUMINAMATH_GPT_ratio_e_to_f_l1753_175358


namespace NUMINAMATH_GPT_lottery_probability_l1753_175326

theorem lottery_probability (p: ℝ) :
  (∀ n, 1 ≤ n ∧ n ≤ 15 → p = 2/3) →
  (true → p = 0.6666666666666666) →
  p = 2/3 :=
by
  intros h h'
  sorry

end NUMINAMATH_GPT_lottery_probability_l1753_175326


namespace NUMINAMATH_GPT_equation_of_chord_line_l1753_175340

theorem equation_of_chord_line (m n s t : ℝ)
  (h₀ : m > 0) (h₁ : n > 0) (h₂ : s > 0) (h₃ : t > 0)
  (h₄ : m + n = 3)
  (h₅ : m / s + n / t = 1)
  (h₆ : m < n)
  (h₇ : s + t = 3 + 2 * Real.sqrt 2)
  (h₈ : ∃ x1 x2 y1 y2 : ℝ, 
        (x1 + x2) / 2 = m ∧ (y1 + y2) / 2 = n ∧
        x1 ^ 2 / 4 + y1 ^ 2 / 16 = 1 ∧
        x2 ^ 2 / 4 + y2 ^ 2 / 16 = 1) 
  : 2 * m + n - 4 = 0 := sorry

end NUMINAMATH_GPT_equation_of_chord_line_l1753_175340


namespace NUMINAMATH_GPT_num_ways_to_arrange_BANANA_l1753_175370

theorem num_ways_to_arrange_BANANA : 
  let total_letters := 6
  let num_A := 3
  let num_N := 2
  let total_arrangements := Nat.factorial total_letters / (Nat.factorial num_A * Nat.factorial num_N)
  total_arrangements = 60 :=
by
  sorry

end NUMINAMATH_GPT_num_ways_to_arrange_BANANA_l1753_175370


namespace NUMINAMATH_GPT_problem_correctness_l1753_175314

variable (f : ℝ → ℝ)
variable (h₀ : ∀ x, f x > 0)
variable (h₁ : ∀ a b, f a * f b = f (a + b))

theorem problem_correctness :
  (f 0 = 1) ∧
  (∀ a, f (-a) = 1 / f a) ∧
  (∀ a, f a = (f (3 * a)) ^ (1 / 3)) :=
by 
  -- Using the hypotheses provided
  sorry

end NUMINAMATH_GPT_problem_correctness_l1753_175314


namespace NUMINAMATH_GPT_line_b_y_intercept_l1753_175327

theorem line_b_y_intercept :
  ∃ c : ℝ, (∀ x : ℝ, (-3) * x + c = -3 * x + 7) ∧ ∃ p : ℝ × ℝ, (p = (5, -2)) → -3 * 5 + c = -2 →
  c = 13 :=
by
  sorry

end NUMINAMATH_GPT_line_b_y_intercept_l1753_175327


namespace NUMINAMATH_GPT_average_rainfall_is_4_l1753_175329

namespace VirginiaRainfall

def march_rainfall : ℝ := 3.79
def april_rainfall : ℝ := 4.5
def may_rainfall : ℝ := 3.95
def june_rainfall : ℝ := 3.09
def july_rainfall : ℝ := 4.67

theorem average_rainfall_is_4 :
  (march_rainfall + april_rainfall + may_rainfall + june_rainfall + july_rainfall) / 5 = 4 := by
  sorry

end VirginiaRainfall

end NUMINAMATH_GPT_average_rainfall_is_4_l1753_175329


namespace NUMINAMATH_GPT_angle_value_l1753_175348

theorem angle_value (α : ℝ) (h1 : 0 ≤ α) (h2 : α < 360) 
(h3 : (Real.sin 215 * π / 180, Real.cos 215 * π / 180) = (Real.sin α, Real.cos α)) :
α = 235 :=
sorry

end NUMINAMATH_GPT_angle_value_l1753_175348


namespace NUMINAMATH_GPT_factor_of_change_l1753_175375

-- Given conditions
def avg_marks_before : ℕ := 45
def avg_marks_after : ℕ := 90
def num_students : ℕ := 30

-- Prove the factor F by which marks are changed
theorem factor_of_change : ∃ F : ℕ, avg_marks_before * F = avg_marks_after := 
by
  use 2
  have h1 : 30 * avg_marks_before = 30 * 45 := rfl
  have h2 : 30 * avg_marks_after = 30 * 90 := rfl
  sorry

end NUMINAMATH_GPT_factor_of_change_l1753_175375


namespace NUMINAMATH_GPT_find_x_when_y_neg_five_l1753_175337

-- Definitions based on the conditions provided
variable (x y : ℝ)
def inversely_proportional (x y : ℝ) := ∃ (k : ℝ), x * y = k

-- Proving the main result
theorem find_x_when_y_neg_five (h_prop : inversely_proportional x y) (hx4 : x = 4) (hy2 : y = 2) :
    (y = -5) → x = - 8 / 5 := by
  sorry

end NUMINAMATH_GPT_find_x_when_y_neg_five_l1753_175337


namespace NUMINAMATH_GPT_difference_of_squares_153_147_l1753_175392

theorem difference_of_squares_153_147 : (153^2 - 147^2) = 1800 := by
  sorry

end NUMINAMATH_GPT_difference_of_squares_153_147_l1753_175392


namespace NUMINAMATH_GPT_incorrect_statement_A_l1753_175391

theorem incorrect_statement_A (p q : Prop) : (p ∨ q) → ¬ (p ∧ q) := by
  intros h
  cases h with
  | inl hp => sorry
  | inr hq => sorry

end NUMINAMATH_GPT_incorrect_statement_A_l1753_175391


namespace NUMINAMATH_GPT_ratio_b_c_l1753_175360

theorem ratio_b_c (a b c d e f : ℝ) 
  (h1 : a / b = 1 / 3)
  (h2 : a * b * c / (d * e * f) = 0.1875)
  (h3 : c / d = 1 / 2)
  (h4 : d / e = 3)
  (h5 : e / f = 1 / 8) : 
  b / c = 3 :=
sorry

end NUMINAMATH_GPT_ratio_b_c_l1753_175360


namespace NUMINAMATH_GPT_question_2024_polynomials_l1753_175349

open Polynomial

noncomputable def P (x : ℝ) : Polynomial ℝ := sorry
noncomputable def Q (x : ℝ) : Polynomial ℝ := sorry

-- Main statement
theorem question_2024_polynomials (P Q : Polynomial ℝ) (hP : P.degree = 2024) (hQ : Q.degree = 2024)
    (hPm : P.leadingCoeff = 1) (hQm : Q.leadingCoeff = 1) (h : ∀ x : ℝ, P.eval x ≠ Q.eval x) :
    ∀ (α : ℝ), α ≠ 0 → ∃ x : ℝ, P.eval (x - α) = Q.eval (x + α) :=
by
  sorry

end NUMINAMATH_GPT_question_2024_polynomials_l1753_175349


namespace NUMINAMATH_GPT_function_relation_l1753_175352

theorem function_relation (f : ℝ → ℝ) 
  (h0 : ∀ x, f (-x) = f x)
  (h1 : ∀ x, f (x + 2) = f x)
  (h2 : ∀ x y, 0 ≤ x ∧ x < y ∧ y ≤ 1 → f x < f y) :
  f 0 < f (-6.5) ∧ f (-6.5) < f (-1) := 
by
  sorry

end NUMINAMATH_GPT_function_relation_l1753_175352


namespace NUMINAMATH_GPT_total_rainfall_correct_l1753_175350

-- Define the individual rainfall amounts
def rainfall_mon1 : ℝ := 0.17
def rainfall_wed1 : ℝ := 0.42
def rainfall_fri : ℝ := 0.08
def rainfall_mon2 : ℝ := 0.37
def rainfall_wed2 : ℝ := 0.51

-- Define the total rainfall
def total_rainfall : ℝ := rainfall_mon1 + rainfall_wed1 + rainfall_fri + rainfall_mon2 + rainfall_wed2

-- Theorem statement to prove the total rainfall is 1.55 cm
theorem total_rainfall_correct : total_rainfall = 1.55 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_total_rainfall_correct_l1753_175350


namespace NUMINAMATH_GPT_simplify_sqrt_expression_l1753_175389

theorem simplify_sqrt_expression (t : ℝ) : (Real.sqrt (t^6 + t^4) = t^2 * Real.sqrt (t^2 + 1)) :=
by sorry

end NUMINAMATH_GPT_simplify_sqrt_expression_l1753_175389


namespace NUMINAMATH_GPT_tank_length_l1753_175377

variable (rate : ℝ)
variable (time : ℝ)
variable (width : ℝ)
variable (depth : ℝ)
variable (volume : ℝ)
variable (length : ℝ)

-- Given conditions
axiom rate_cond : rate = 5 -- cubic feet per hour
axiom time_cond : time = 60 -- hours
axiom width_cond : width = 6 -- feet
axiom depth_cond : depth = 5 -- feet

-- Derived volume from the rate and time
axiom volume_cond : volume = rate * time

-- Definition of length from volume, width, and depth
axiom length_def : length = volume / (width * depth)

-- The proof problem to show
theorem tank_length : length = 10 := by
  -- conditions provided and we expect the length to be computed
  sorry

end NUMINAMATH_GPT_tank_length_l1753_175377


namespace NUMINAMATH_GPT_semicircle_perimeter_l1753_175361

/-- The perimeter of a semicircle with radius 6.3 cm is approximately 32.382 cm. -/
theorem semicircle_perimeter (r : ℝ) (h : r = 6.3) : 
  (π * r + 2 * r = 32.382) :=
by
  sorry

end NUMINAMATH_GPT_semicircle_perimeter_l1753_175361


namespace NUMINAMATH_GPT_one_minus_repeating_six_l1753_175332

noncomputable def repeating_six : Real := 2 / 3

theorem one_minus_repeating_six : 1 - repeating_six = 1 / 3 :=
by
  sorry

end NUMINAMATH_GPT_one_minus_repeating_six_l1753_175332


namespace NUMINAMATH_GPT_cannot_tile_surface_square_hexagon_l1753_175306

-- Definitions of internal angles of the tile shapes
def internal_angle_triangle := 60
def internal_angle_square := 90
def internal_angle_hexagon := 120
def internal_angle_octagon := 135

-- The theorem to prove that square and hexagon cannot tile a surface without gaps or overlaps
theorem cannot_tile_surface_square_hexagon : ∀ (m n : ℕ), internal_angle_square * m + internal_angle_hexagon * n ≠ 360 := 
by sorry

end NUMINAMATH_GPT_cannot_tile_surface_square_hexagon_l1753_175306


namespace NUMINAMATH_GPT_williams_tips_august_l1753_175372

variable (A : ℝ) (total_tips : ℝ)
variable (tips_August : ℝ) (average_monthly_tips_other_months : ℝ)

theorem williams_tips_august (h1 : tips_August = 0.5714285714285714 * total_tips)
                               (h2 : total_tips = 7 * average_monthly_tips_other_months) 
                               (h3 : total_tips = tips_August + 6 * average_monthly_tips_other_months) :
                               tips_August = 8 * average_monthly_tips_other_months :=
by
  sorry

end NUMINAMATH_GPT_williams_tips_august_l1753_175372


namespace NUMINAMATH_GPT_ellipse_foci_y_axis_range_l1753_175371

noncomputable def is_ellipse_with_foci_on_y_axis (k : ℝ) : Prop :=
  (k > 5) ∧ (k < 10) ∧ (10 - k > k - 5)

theorem ellipse_foci_y_axis_range (k : ℝ) :
  is_ellipse_with_foci_on_y_axis k ↔ 5 < k ∧ k < 7.5 := 
by
  sorry

end NUMINAMATH_GPT_ellipse_foci_y_axis_range_l1753_175371


namespace NUMINAMATH_GPT_escalator_steps_l1753_175330

theorem escalator_steps (T : ℝ) (E : ℝ) (N : ℝ) (h1 : N - 11 = 2 * (N - 29)) : N = 47 :=
by
  sorry

end NUMINAMATH_GPT_escalator_steps_l1753_175330


namespace NUMINAMATH_GPT_next_performance_together_in_90_days_l1753_175387

theorem next_performance_together_in_90_days :
  Nat.lcm (Nat.lcm 5 6) (Nat.lcm 9 10) = 90 := by
  sorry

end NUMINAMATH_GPT_next_performance_together_in_90_days_l1753_175387


namespace NUMINAMATH_GPT_sum_of_all_possible_values_of_N_with_equation_l1753_175311

def satisfiesEquation (N : ℝ) : Prop :=
  N * (N - 4) = -7

theorem sum_of_all_possible_values_of_N_with_equation :
  (∀ N, satisfiesEquation N → N + (4 - N) = 4) :=
sorry

end NUMINAMATH_GPT_sum_of_all_possible_values_of_N_with_equation_l1753_175311


namespace NUMINAMATH_GPT_eval_expression_l1753_175307

theorem eval_expression : 
  (20-19 + 18-17 + 16-15 + 14-13 + 12-11 + 10-9 + 8-7 + 6-5 + 4-3 + 2-1) / 
  (1-2 + 3-4 + 5-6 + 7-8 + 9-10 + 11-12 + 13-14 + 15-16 + 17-18 + 19-20) = -1 := by
  sorry

end NUMINAMATH_GPT_eval_expression_l1753_175307


namespace NUMINAMATH_GPT_hilt_has_2_pennies_l1753_175355

-- Define the total value of coins each person has without considering Mrs. Hilt's pennies
def dimes : ℕ := 2
def nickels : ℕ := 2
def hilt_base_amount : ℕ := dimes * 10 + nickels * 5 -- 30 cents

def jacob_pennies : ℕ := 4
def jacob_nickels : ℕ := 1
def jacob_dimes : ℕ := 1
def jacob_amount : ℕ := jacob_pennies * 1 + jacob_nickels * 5 + jacob_dimes * 10 -- 19 cents

def difference : ℕ := 13
def hilt_pennies : ℕ := 2 -- The solution's correct answer

theorem hilt_has_2_pennies : hilt_base_amount - jacob_amount + hilt_pennies = difference := by sorry

end NUMINAMATH_GPT_hilt_has_2_pennies_l1753_175355


namespace NUMINAMATH_GPT_thirty_thousand_times_thirty_thousand_l1753_175399

-- Define the number thirty thousand
def thirty_thousand : ℕ := 30000

-- Define the product of thirty thousand times thirty thousand
def product_thirty_thousand : ℕ := thirty_thousand * thirty_thousand

-- State the theorem that this product equals nine hundred million
theorem thirty_thousand_times_thirty_thousand :
  product_thirty_thousand = 900000000 :=
sorry -- Proof goes here

end NUMINAMATH_GPT_thirty_thousand_times_thirty_thousand_l1753_175399


namespace NUMINAMATH_GPT_max_result_l1753_175323

-- Define the expressions as Lean definitions
def expr1 : Int := 2 + (-2)
def expr2 : Int := 2 - (-2)
def expr3 : Int := 2 * (-2)
def expr4 : Int := 2 / (-2)

-- State the theorem
theorem max_result : 
  (expr2 = 4) ∧ (expr2 > expr1) ∧ (expr2 > expr3) ∧ (expr2 > expr4) :=
by
  sorry

end NUMINAMATH_GPT_max_result_l1753_175323
