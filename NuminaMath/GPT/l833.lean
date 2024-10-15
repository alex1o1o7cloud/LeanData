import Mathlib

namespace NUMINAMATH_GPT_flying_scotsman_more_carriages_l833_83372

theorem flying_scotsman_more_carriages :
  ∀ (E N No F T D : ℕ),
    E = 130 →
    E = N + 20 →
    No = 100 →
    T = 460 →
    D = F - No →
    F + E + N + No = T →
    D = 20 :=
by
  intros E N No F T D hE1 hE2 hNo hT hD hSum
  sorry

end NUMINAMATH_GPT_flying_scotsman_more_carriages_l833_83372


namespace NUMINAMATH_GPT_moles_NaOH_combined_with_HCl_l833_83331

-- Definitions for given conditions
def NaOH : Type := Unit
def HCl : Type := Unit
def NaCl : Type := Unit
def H2O : Type := Unit

def balanced_reaction (nHCl nNaOH nNaCl nH2O : ℕ) : Prop :=
  nHCl = nNaOH ∧ nNaOH = nNaCl ∧ nNaCl = nH2O

def mole_mass_H2O : ℕ := 18

-- Given: certain amount of NaOH combined with 1 mole of HCl
def initial_moles_HCl : ℕ := 1

-- Given: 18 grams of H2O formed
def grams_H2O : ℕ := 18

-- Molar mass of H2O is approximately 18 g/mol, so 18 grams is 1 mole
def moles_H2O : ℕ := grams_H2O / mole_mass_H2O

-- Prove that number of moles of NaOH combined with HCl is 1 mole
theorem moles_NaOH_combined_with_HCl : 
  balanced_reaction initial_moles_HCl 1 1 moles_H2O →
  moles_H2O = 1 →
  1 = 1 :=
by
  intros h1 h2
  sorry

end NUMINAMATH_GPT_moles_NaOH_combined_with_HCl_l833_83331


namespace NUMINAMATH_GPT_one_point_one_seven_three_billion_in_scientific_notation_l833_83322

theorem one_point_one_seven_three_billion_in_scientific_notation :
  (1.173 * 10^9 = 1.173 * 1000000000) :=
by
  sorry

end NUMINAMATH_GPT_one_point_one_seven_three_billion_in_scientific_notation_l833_83322


namespace NUMINAMATH_GPT_angle_measure_l833_83368

theorem angle_measure (x : ℝ) (h : 180 - x = 4 * (90 - x)) : x = 60 :=
by {
  sorry
}

end NUMINAMATH_GPT_angle_measure_l833_83368


namespace NUMINAMATH_GPT_roots_poly_sum_cubed_eq_l833_83351

theorem roots_poly_sum_cubed_eq :
  ∀ (r s t : ℝ), (r + s + t = 0) 
  → (∀ x, 9 * x^3 + 2023 * x + 4047 = 0 → x = r ∨ x = s ∨ x = t) 
  → (r + s) ^ 3 + (s + t) ^ 3 + (t + r) ^ 3 = 1349 :=
by
  intros r s t h_sum h_roots
  sorry

end NUMINAMATH_GPT_roots_poly_sum_cubed_eq_l833_83351


namespace NUMINAMATH_GPT_probability_A_seventh_week_l833_83399

/-
Conditions:
1. There are four different passwords: A, B, C, and D.
2. Each week, one of these passwords is used.
3. Each week, the password is chosen at random and equally likely from the three passwords that were not used in the previous week.
4. Password A is used in the first week.

Goal:
Prove that the probability that password A will be used in the seventh week is 61/243.
-/

def prob_password_A_in_seventh_week : ℚ :=
  let Pk (k : ℕ) : ℚ := 
    if k = 1 then 1
    else if k >= 2 then ((-1 / 3)^(k - 1) * (3 / 4) + 1 / 4) else 0
  Pk 7

theorem probability_A_seventh_week : prob_password_A_in_seventh_week = 61 / 243 := by
  sorry

end NUMINAMATH_GPT_probability_A_seventh_week_l833_83399


namespace NUMINAMATH_GPT_square_area_eq_36_l833_83356

theorem square_area_eq_36 :
  let triangle_side1 := 5.5
  let triangle_side2 := 7.5
  let triangle_side3 := 11
  let triangle_perimeter := triangle_side1 + triangle_side2 + triangle_side3
  let square_perimeter := triangle_perimeter
  let square_side_length := square_perimeter / 4
  let square_area := square_side_length * square_side_length
  square_area = 36 := by
  sorry

end NUMINAMATH_GPT_square_area_eq_36_l833_83356


namespace NUMINAMATH_GPT_complex_modulus_problem_l833_83334

open Complex

def modulus_of_z (z : ℂ) (h : (z - 2 * I) * (1 - I) = -2) : Prop :=
  abs z = Real.sqrt 2

theorem complex_modulus_problem (z : ℂ) (h : (z - 2 * I) * (1 - I) = -2) : 
  modulus_of_z z h :=
sorry

end NUMINAMATH_GPT_complex_modulus_problem_l833_83334


namespace NUMINAMATH_GPT_rent_increase_l833_83361

theorem rent_increase (monthly_rent_first_3_years : ℕ) (months_first_3_years : ℕ) 
  (total_paid : ℕ) (total_years : ℕ) (months_in_a_year : ℕ) (new_monthly_rent : ℕ) :
  monthly_rent_first_3_years * (months_in_a_year * 3) + new_monthly_rent * (months_in_a_year * (total_years - 3)) = total_paid →
  new_monthly_rent = 350 :=
by
  intros h
  -- proof development
  sorry

end NUMINAMATH_GPT_rent_increase_l833_83361


namespace NUMINAMATH_GPT_minimize_x_expr_minimized_l833_83371

noncomputable def minimize_x_expr (x : ℝ) : ℝ :=
  x + 4 / (x + 1)

theorem minimize_x_expr_minimized 
  (hx : x > -1) 
  : x = 1 ↔ minimize_x_expr x = minimize_x_expr 1 :=
by
  sorry

end NUMINAMATH_GPT_minimize_x_expr_minimized_l833_83371


namespace NUMINAMATH_GPT_businessman_expenditure_l833_83391

theorem businessman_expenditure (P : ℝ) (h1 : P * 1.21 = 24200) : P = 20000 := 
by sorry

end NUMINAMATH_GPT_businessman_expenditure_l833_83391


namespace NUMINAMATH_GPT_expenditure_recorded_neg_20_l833_83381

-- Define the condition where income of 60 yuan is recorded as +60 yuan
def income_recorded (income : ℤ) : ℤ :=
  income

-- Define what expenditure is given the condition
def expenditure_recorded (expenditure : ℤ) : ℤ :=
  -expenditure

-- Prove that an expenditure of 20 yuan is recorded as -20 yuan
theorem expenditure_recorded_neg_20 :
  expenditure_recorded 20 = -20 :=
by
  sorry

end NUMINAMATH_GPT_expenditure_recorded_neg_20_l833_83381


namespace NUMINAMATH_GPT_sequence_an_square_l833_83313

theorem sequence_an_square (a : ℕ → ℝ) (h1 : a 1 = 1)
  (h2 : ∀ n : ℕ, a (n + 1) > a n) 
  (h3 : ∀ n : ℕ, a (n + 1)^2 + a n^2 + 1 = 2 * (a (n + 1) * a n + a (n + 1) + a n)) :
  ∀ n : ℕ, a n = n^2 :=
by
  sorry

end NUMINAMATH_GPT_sequence_an_square_l833_83313


namespace NUMINAMATH_GPT_option1_payment_correct_option2_payment_correct_most_cost_effective_for_30_l833_83387

variables (x : ℕ) (hx : x > 10)

def suit_price : ℕ := 1000
def tie_price : ℕ := 200
def num_suits : ℕ := 10

-- Option 1: Buy one suit, get one tie for free
def option1_payment : ℕ := 200 * x + 8000

-- Option 2: All items sold at a 10% discount
def option2_payment : ℕ := (10 * 1000 + x * 200) * 9 / 10

-- For x = 30, which option is more cost-effective
def x_value := 30
def option1_payment_30 : ℕ := 200 * x_value + 8000
def option2_payment_30 : ℕ := (10 * 1000 + x_value * 200) * 9 / 10
def more_cost_effective_option_30 : ℕ := if option1_payment_30 < option2_payment_30 then option1_payment_30 else option2_payment_30

-- Most cost-effective option for x = 30 with new combination plan
def combination_payment_30 : ℕ := 10000 + 20 * 200 * 9 / 10

-- Statements to be proved
theorem option1_payment_correct : option1_payment x = 200 * x + 8000 := sorry

theorem option2_payment_correct : option2_payment x = (10 * 1000 + x * 200) * 9 / 10 := sorry

theorem most_cost_effective_for_30 :
  option1_payment_30 = 14000 ∧ 
  option2_payment_30 = 14400 ∧ 
  more_cost_effective_option_30 = 14000 ∧
  combination_payment_30 = 13600 := sorry

end NUMINAMATH_GPT_option1_payment_correct_option2_payment_correct_most_cost_effective_for_30_l833_83387


namespace NUMINAMATH_GPT_rectangle_dimensions_l833_83311

-- Define the dimensions and properties of the rectangle
variables {a b : ℕ}

-- Theorem statement
theorem rectangle_dimensions 
  (h1 : b = a + 3)
  (h2 : 2 * a + 2 * b + a = a * b) : 
  (a = 3 ∧ b = 6) :=
by
  sorry

end NUMINAMATH_GPT_rectangle_dimensions_l833_83311


namespace NUMINAMATH_GPT_jelly_ratio_l833_83395

theorem jelly_ratio (G S R P : ℕ) 
  (h1 : G = 2 * S)
  (h2 : R = 2 * P) 
  (h3 : P = 6) 
  (h4 : S = 18) : 
  R / G = 1 / 3 := by
  sorry

end NUMINAMATH_GPT_jelly_ratio_l833_83395


namespace NUMINAMATH_GPT_range_of_a_l833_83345

def A : Set ℝ := {x | x > 1}
def B (a : ℝ) : Set ℝ := {x | x < a}

theorem range_of_a (a : ℝ) (h : (A ∩ B a).Nonempty) : a > 1 :=
sorry

end NUMINAMATH_GPT_range_of_a_l833_83345


namespace NUMINAMATH_GPT_probability_at_least_one_head_and_die_3_l833_83358

-- Define the probability of an event happening
noncomputable def probability_of_event (total_outcomes : ℕ) (successful_outcomes : ℕ) : ℚ :=
  successful_outcomes / total_outcomes

-- Define the problem specific values
def total_coin_outcomes : ℕ := 4
def successful_coin_outcomes : ℕ := 3
def total_die_outcomes : ℕ := 8
def successful_die_outcome : ℕ := 1
def total_outcomes : ℕ := total_coin_outcomes * total_die_outcomes
def successful_outcomes : ℕ := successful_coin_outcomes * successful_die_outcome

-- Prove that the probability of at least one head in two coin flips and die showing a 3 is 3/32
theorem probability_at_least_one_head_and_die_3 : 
  probability_of_event total_outcomes successful_outcomes = 3 / 32 := by
  sorry

end NUMINAMATH_GPT_probability_at_least_one_head_and_die_3_l833_83358


namespace NUMINAMATH_GPT_correct_option_d_l833_83319

variable (m t x1 x2 y1 y2 : ℝ)

theorem correct_option_d (h_m : m > 0)
  (h_y1 : y1 = m * x1^2 - 2 * m * x1 + 1)
  (h_y2 : y2 = m * x2^2 - 2 * m * x2 + 1)
  (h_x1 : t < x1 ∧ x1 < t + 1)
  (h_x2 : t + 2 < x2 ∧ x2 < t + 3)
  (h_t_geq1 : t ≥ 1) :
  y1 < y2 := sorry

end NUMINAMATH_GPT_correct_option_d_l833_83319


namespace NUMINAMATH_GPT_pentagon_sum_of_sides_and_vertices_eq_10_l833_83349

-- Define the number of sides of a pentagon
def number_of_sides : ℕ := 5

-- Define the number of vertices of a pentagon
def number_of_vertices : ℕ := 5

-- Define the sum of sides and vertices
def sum_of_sides_and_vertices : ℕ :=
  number_of_sides + number_of_vertices

-- The theorem to prove that the sum is 10
theorem pentagon_sum_of_sides_and_vertices_eq_10 : sum_of_sides_and_vertices = 10 :=
by
  sorry

end NUMINAMATH_GPT_pentagon_sum_of_sides_and_vertices_eq_10_l833_83349


namespace NUMINAMATH_GPT_celina_paid_multiple_of_diego_l833_83388

theorem celina_paid_multiple_of_diego
  (D : ℕ) (x : ℕ)
  (h_total : (x + 1) * D + 1000 = 50000)
  (h_positive : D > 0) :
  x = 48 :=
sorry

end NUMINAMATH_GPT_celina_paid_multiple_of_diego_l833_83388


namespace NUMINAMATH_GPT_infinite_geometric_series_common_ratio_l833_83390

theorem infinite_geometric_series_common_ratio 
  (a S r : ℝ) 
  (ha : a = 400) 
  (hS : S = 2500)
  (h_sum : S = a / (1 - r)) :
  r = 0.84 :=
by
  -- Proof will go here
  sorry

end NUMINAMATH_GPT_infinite_geometric_series_common_ratio_l833_83390


namespace NUMINAMATH_GPT_perpendicular_line_through_point_l833_83307

noncomputable def line_eq_perpendicular (x y : ℝ) : Prop :=
  3 * x - 6 * y = 9

noncomputable def slope_intercept_form (m b x y : ℝ) : Prop :=
  y = m * x + b

theorem perpendicular_line_through_point
  (x y : ℝ)
  (hx : x = 2)
  (hy : y = -3) :
  ∀ x y, line_eq_perpendicular x y →
  ∃ m b, slope_intercept_form m b x y ∧ m = -2 ∧ b = 1 :=
sorry

end NUMINAMATH_GPT_perpendicular_line_through_point_l833_83307


namespace NUMINAMATH_GPT_smallest_four_digit_integer_mod_8_eq_3_l833_83310

theorem smallest_four_digit_integer_mod_8_eq_3 : ∃ n : ℕ, 1000 ≤ n ∧ n < 10000 ∧ n % 8 = 3 ∧ n = 1003 := by
  -- Proof will be provided here
  sorry

end NUMINAMATH_GPT_smallest_four_digit_integer_mod_8_eq_3_l833_83310


namespace NUMINAMATH_GPT_circle_numbers_contradiction_l833_83333

theorem circle_numbers_contradiction :
  ¬ ∃ (f : Fin 25 → Fin 25), ∀ i : Fin 25, 
  let a := f i
  let b := f ((i + 1) % 25)
  (b = a + 10 ∨ b = a - 10 ∨ ∃ k : Int, b = a * k) :=
by
  sorry

end NUMINAMATH_GPT_circle_numbers_contradiction_l833_83333


namespace NUMINAMATH_GPT_suff_and_necc_l833_83373

variable (x : ℝ)

def A : Set ℝ := { x | x > 2 }
def B : Set ℝ := { x | x < 0 }
def C : Set ℝ := { x | x * (x - 2) > 0 }

theorem suff_and_necc : (x ∈ (A ∪ B)) ↔ (x ∈ C) := by
  sorry

end NUMINAMATH_GPT_suff_and_necc_l833_83373


namespace NUMINAMATH_GPT_jeremy_is_40_l833_83305

-- Definitions for Jeremy (J), Sebastian (S), and Sophia (So)
def JeremyCurrentAge : ℕ := 40
def SebastianCurrentAge : ℕ := JeremyCurrentAge + 4
def SophiaCurrentAge : ℕ := 60 - 3

-- Assertion properties
axiom age_sum_in_3_years : (JeremyCurrentAge + 3) + (SebastianCurrentAge + 3) + (SophiaCurrentAge + 3) = 150
axiom sebastian_older_by_4 : SebastianCurrentAge = JeremyCurrentAge + 4
axiom sophia_age_in_3_years : SophiaCurrentAge + 3 = 60

-- The theorem to prove that Jeremy is currently 40 years old
theorem jeremy_is_40 : JeremyCurrentAge = 40 := by
  sorry

end NUMINAMATH_GPT_jeremy_is_40_l833_83305


namespace NUMINAMATH_GPT_find_u_l833_83339

-- Definitions for given points lying on a straight line
def point := (ℝ × ℝ)

-- Points
def p1 : point := (2, 8)
def p2 : point := (6, 20)
def p3 : point := (10, 32)

-- Function to check if point is on the line derived from p1, p2, p3
def is_on_line (x y : ℝ) : Prop :=
  ∃ m b : ℝ, y = m * x + b ∧
  p1.2 = m * p1.1 + b ∧ 
  p2.2 = m * p2.1 + b ∧
  p3.2 = m * p3.1 + b

-- Statement to prove
theorem find_u (u : ℝ) (hu : is_on_line 50 u) : u = 152 :=
sorry

end NUMINAMATH_GPT_find_u_l833_83339


namespace NUMINAMATH_GPT_find_a_parallel_l833_83332

-- Define the lines
def line1 (a : ℝ) (x : ℝ) (y : ℝ) : Prop :=
  (a + 1) * x + 2 * y = 2

def line2 (a : ℝ) (x : ℝ) (y : ℝ) : Prop :=
  x + a * y = 1

-- Define the parallel condition
def are_parallel (a : ℝ) : Prop :=
  ∀ x y : ℝ, line1 a x y → line2 a x y

-- The theorem stating our problem
theorem find_a_parallel (a : ℝ) : are_parallel a → a = -2 :=
by
  sorry

end NUMINAMATH_GPT_find_a_parallel_l833_83332


namespace NUMINAMATH_GPT_other_root_of_quadratic_l833_83357

theorem other_root_of_quadratic (a c : ℝ) (h : a ≠ 0) (h_root : 4 * a * 0^2 - 2 * a * 0 + c = 0) :
  ∃ t : ℝ, (4 * a * t^2 - 2 * a * t + c = 0) ∧ t = 1 / 2 :=
by
  sorry

end NUMINAMATH_GPT_other_root_of_quadratic_l833_83357


namespace NUMINAMATH_GPT_only_other_list_with_same_product_l833_83365

-- Assigning values to letters
def letter_value (ch : Char) : ℕ :=
  match ch with
  | 'A' => 1 | 'B' => 2 | 'C' => 3 | 'D' => 4 | 'E' => 5 | 'F' => 6 | 'G' => 7 | 'H' => 8
  | 'I' => 9 | 'J' => 10| 'K' => 11| 'L' => 12| 'M' => 13| 'N' => 14| 'O' => 15| 'P' => 16
  | 'Q' => 17| 'R' => 18| 'S' => 19| 'T' => 20| 'U' => 21| 'V' => 22| 'W' => 23| 'X' => 24
  | 'Y' => 25| 'Z' => 26| _ => 0

-- Define the product function for a list of 4 letters
def product_of_list (lst : List Char) : ℕ :=
  lst.map letter_value |> List.prod

-- Define the specific lists
def BDFH : List Char := ['B', 'D', 'F', 'H']
def BCDH : List Char := ['B', 'C', 'D', 'H']

-- The main statement to prove
theorem only_other_list_with_same_product : 
  product_of_list BCDH = product_of_list BDFH :=
by
  -- Sorry is a placeholder for the proof
  sorry

end NUMINAMATH_GPT_only_other_list_with_same_product_l833_83365


namespace NUMINAMATH_GPT_second_mechanic_hours_l833_83312

theorem second_mechanic_hours (x y : ℕ) (h1 : 45 * x + 85 * y = 1100) (h2 : x + y = 20) : y = 5 :=
by
  sorry

end NUMINAMATH_GPT_second_mechanic_hours_l833_83312


namespace NUMINAMATH_GPT_gcd_1855_1120_l833_83342

theorem gcd_1855_1120 : Int.gcd 1855 1120 = 35 :=
by
  sorry

end NUMINAMATH_GPT_gcd_1855_1120_l833_83342


namespace NUMINAMATH_GPT_total_buyers_in_three_days_l833_83335

theorem total_buyers_in_three_days
  (D_minus_2 : ℕ)
  (D_minus_1 : ℕ)
  (D_0 : ℕ)
  (h1 : D_minus_2 = 50)
  (h2 : D_minus_1 = D_minus_2 / 2)
  (h3 : D_0 = D_minus_1 + 40) :
  D_minus_2 + D_minus_1 + D_0 = 140 :=
by
  sorry

end NUMINAMATH_GPT_total_buyers_in_three_days_l833_83335


namespace NUMINAMATH_GPT_paper_boat_travel_time_l833_83347

-- Defining the conditions as constants
def distance_embankment : ℝ := 50
def speed_downstream : ℝ := 10
def speed_upstream : ℝ := 12.5

-- Definitions for the speeds of the boat and current
noncomputable def v_boat : ℝ := (speed_upstream + speed_downstream) / 2
noncomputable def v_current : ℝ := (speed_downstream - speed_upstream) / 2

-- Statement to prove the time taken for the paper boat
theorem paper_boat_travel_time :
  (distance_embankment / v_current) = 40 := by
  sorry

end NUMINAMATH_GPT_paper_boat_travel_time_l833_83347


namespace NUMINAMATH_GPT_video_duration_correct_l833_83383

/-
Define the conditions as given:
1. Vasya's time from home to school
2. Petya's time from school to home
3. Meeting conditions
-/

-- Define the times for Vasya and Petya
def vasya_time : ℕ := 8
def petya_time : ℕ := 5

-- Define the total video duration when correctly merged
def video_duration : ℕ := 5

-- State the theorem to be proved in Lean:
theorem video_duration_correct : vasya_time = 8 → petya_time = 5 → video_duration = 5 :=
by
  intros h1 h2
  exact sorry

end NUMINAMATH_GPT_video_duration_correct_l833_83383


namespace NUMINAMATH_GPT_find_angle_A_area_bound_given_a_l833_83314

-- (1) Given the condition, prove that \(A = \frac{\pi}{3}\).
theorem find_angle_A
  {A B C : ℝ} {a b c : ℝ}
  (h1 : a / (Real.cos A) + b / (Real.cos B) + c / (Real.cos C) = (Real.sqrt 3) * c * (Real.sin B) / (Real.cos B * Real.cos C)) :
  A = Real.pi / 3 :=
sorry

-- (2) Given a = 4, prove the area S satisfies \(S \leq 4\sqrt{3}\).
theorem area_bound_given_a
  {A B C : ℝ} {a b c S : ℝ}
  (ha : a = 4)
  (hA : A = Real.pi / 3)
  (h1 : a / (Real.cos A) + b / (Real.cos B) + c / (Real.cos C) = (Real.sqrt 3) * c * (Real.sin B) / (Real.cos B * Real.cos C))
  (hS : S = 1 / 2 * b * c * Real.sin A) :
  S ≤ 4 * Real.sqrt 3 :=
sorry

end NUMINAMATH_GPT_find_angle_A_area_bound_given_a_l833_83314


namespace NUMINAMATH_GPT_max_value_condition_l833_83325

noncomputable def f (a x : ℝ) : ℝ :=
if h : 0 < x ∧ x ≤ a then
  Real.log x
else
  if x > a then
    a / x
  else
    0 -- This case should not happen given the domain conditions

theorem max_value_condition (a : ℝ) : 
  (∃ M, ∀ x > 0, x ≤ a → f a x ≤ M) ∧ (∀ x > a, f a x ≤ M) ↔ a ≥ Real.exp 1 :=
sorry

end NUMINAMATH_GPT_max_value_condition_l833_83325


namespace NUMINAMATH_GPT_volume_ratio_of_trapezoidal_pyramids_l833_83366

theorem volume_ratio_of_trapezoidal_pyramids 
  (V U : ℝ) (m n m₁ n₁ : ℝ)
  (hV : V > 0) (hU : U > 0) (hm : m > 0) (hn : n > 0) (hm₁ : m₁ > 0) (hn₁ : n₁ > 0)
  (h_ratio : U / V = (m₁ + n₁)^2 / (m + n)^2) :
  U / V = (m₁ + n₁)^2 / (m + n)^2 :=
sorry

end NUMINAMATH_GPT_volume_ratio_of_trapezoidal_pyramids_l833_83366


namespace NUMINAMATH_GPT_num_digits_c_l833_83396

theorem num_digits_c (a b c : ℕ) (ha : 10 ^ 2010 ≤ a ∧ a < 10 ^ 2011)
  (hb : 10 ^ 2011 ≤ b ∧ b < 10 ^ 2012)
  (h1 : a < b) (h2 : b < c)
  (div1 : ∃ k : ℕ, b + a = k * (b - a))
  (div2 : ∃ m : ℕ, c + b = m * (c - b)) :
  10 ^ 4 ≤ c ∧ c < 10 ^ 5 :=
sorry

end NUMINAMATH_GPT_num_digits_c_l833_83396


namespace NUMINAMATH_GPT_f_equals_one_l833_83320

-- Define the functions f, g, h with the given properties

def f : ℕ → ℕ := sorry
def g : ℕ → ℕ := sorry
def h : ℕ → ℕ := sorry

-- Condition 1: h is injective
axiom h_injective : ∀ {a b : ℕ}, h a = h b → a = b

-- Condition 2: g is surjective
axiom g_surjective : ∀ n : ℕ, ∃ m : ℕ, g m = n

-- Condition 3: Definition of f in terms of g and h
axiom f_def : ∀ n : ℕ, f n = g n - h n + 1

-- Prove that f(n) = 1 for all n ∈ ℕ
theorem f_equals_one : ∀ n : ℕ, f n = 1 := by
  sorry

end NUMINAMATH_GPT_f_equals_one_l833_83320


namespace NUMINAMATH_GPT_circles_intersect_l833_83362

-- Define the parameters and conditions given in the problem.
def r1 : ℝ := 5  -- Radius of circle O1
def r2 : ℝ := 8  -- Radius of circle O2
def d : ℝ := 8   -- Distance between the centers of O1 and O2

-- The main theorem that needs to be proven.
theorem circles_intersect (r1 r2 d : ℝ) (h_r1 : r1 = 5) (h_r2 : r2 = 8) (h_d : d = 8) :
  r2 - r1 < d ∧ d < r1 + r2 :=
by
  sorry

end NUMINAMATH_GPT_circles_intersect_l833_83362


namespace NUMINAMATH_GPT_simplify_expression_l833_83385

theorem simplify_expression : 
  2^345 - 3^4 * (3^2)^2 = 2^345 - 6561 := by
sorry

end NUMINAMATH_GPT_simplify_expression_l833_83385


namespace NUMINAMATH_GPT_units_produced_today_l833_83302

theorem units_produced_today (n : ℕ) (X : ℕ) 
  (h1 : n = 9) 
  (h2 : (360 + X) / (n + 1) = 45) 
  (h3 : 40 * n = 360) : 
  X = 90 := 
sorry

end NUMINAMATH_GPT_units_produced_today_l833_83302


namespace NUMINAMATH_GPT_rectangular_plot_breadth_l833_83353

theorem rectangular_plot_breadth (b l : ℝ) (A : ℝ)
  (h1 : l = 3 * b)
  (h2 : A = l * b)
  (h3 : A = 2700) : b = 30 :=
by sorry

end NUMINAMATH_GPT_rectangular_plot_breadth_l833_83353


namespace NUMINAMATH_GPT_cathy_wallet_left_money_l833_83330

noncomputable def amount_left_in_wallet (initial : ℝ) (dad_amount : ℝ) (book_cost : ℝ) (saving_percentage : ℝ) : ℝ :=
  let mom_amount := 2 * dad_amount
  let total_initial := initial + dad_amount + mom_amount
  let after_purchase := total_initial - book_cost
  let saved_amount := saving_percentage * after_purchase
  after_purchase - saved_amount

theorem cathy_wallet_left_money :
  amount_left_in_wallet 12 25 15 0.20 = 57.60 :=
by 
  sorry

end NUMINAMATH_GPT_cathy_wallet_left_money_l833_83330


namespace NUMINAMATH_GPT_find_certain_number_l833_83355

theorem find_certain_number (x : ℝ) 
    (h : 7 * x - 6 - 12 = 4 * x) : x = 6 := 
by
  sorry

end NUMINAMATH_GPT_find_certain_number_l833_83355


namespace NUMINAMATH_GPT_inequality_solution_range_l833_83336

theorem inequality_solution_range (a : ℝ) :
  (∀ x : ℝ, 1 ≤ x ∧ x ≤ 5 → x ^ 2 + a * x + 4 < 0) ↔ a < -4 :=
by 
  sorry

end NUMINAMATH_GPT_inequality_solution_range_l833_83336


namespace NUMINAMATH_GPT_probability_of_forming_phrase_l833_83394

theorem probability_of_forming_phrase :
  let cards := ["中", "国", "梦"]
  let n := 6
  let m := 1
  ∃ (p : ℚ), p = (m / n : ℚ) ∧ p = 1 / 6 :=
by
  sorry

end NUMINAMATH_GPT_probability_of_forming_phrase_l833_83394


namespace NUMINAMATH_GPT_can_form_triangle_l833_83308

-- Define the function to check for the triangle inequality
def is_triangle (a b c : ℕ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

-- Problem statement: Prove that only the set (3, 4, 6) can form a triangle
theorem can_form_triangle :
  (¬ is_triangle 3 4 8) ∧
  (¬ is_triangle 5 6 11) ∧
  (¬ is_triangle 5 8 15) ∧
  (is_triangle 3 4 6) :=
by
  sorry

end NUMINAMATH_GPT_can_form_triangle_l833_83308


namespace NUMINAMATH_GPT_max_area_triang_ABC_l833_83341

noncomputable def max_area_triang (a b c : ℝ) (M : ℝ) (BM : ℝ := 2) (AM : ℝ := c - b) : ℝ :=
if M = (b + c) / 2 then 2 * Real.sqrt 3 else 0

theorem max_area_triang_ABC (a b c : ℝ) (M : ℝ) (BM : ℝ := 2) (AM : ℝ := c - b) (M_midpoint : M = (b + c) / 2) :
  max_area_triang a b c M BM AM = 2 * Real.sqrt 3 :=
by
  sorry

end NUMINAMATH_GPT_max_area_triang_ABC_l833_83341


namespace NUMINAMATH_GPT_range_of_m_l833_83398

noncomputable def quadratic_polynomial (m : ℝ) (x : ℝ) : ℝ :=
  x^2 + (m - 1) * x + m^2 - 2

theorem range_of_m (m : ℝ) (h1 : ∃ x1 x2 : ℝ, x1 < -1 ∧ x2 > 1 ∧ quadratic_polynomial m x1 = 0 ∧ quadratic_polynomial m x2 = 0) :
  0 < m ∧ m < 1 :=
sorry

end NUMINAMATH_GPT_range_of_m_l833_83398


namespace NUMINAMATH_GPT_numerator_of_fraction_l833_83369

/-- 
Given:
1. The denominator of a fraction is 7 less than 3 times the numerator.
2. The fraction is equivalent to 2/5.
Prove that the numerator of the fraction is 14.
-/
theorem numerator_of_fraction {x : ℕ} (h : x / (3 * x - 7) = 2 / 5) : x = 14 :=
  sorry

end NUMINAMATH_GPT_numerator_of_fraction_l833_83369


namespace NUMINAMATH_GPT_smallest_possible_AC_l833_83323

-- Constants and assumptions
variables (AC CD : ℕ)
def BD_squared : ℕ := 68

-- Prime number constraint for CD
def is_prime (n : ℕ) : Prop := n = 2 ∨ n = 3 ∨ n = 5 ∨ n = 7

-- Given facts
axiom eq_ab_ac (AB : ℕ) : AB = AC
axiom perp_bd_ac (BD AC : ℕ) : BD^2 = BD_squared
axiom int_ac_cd : AC = (CD^2 + BD_squared) / (2 * CD)

theorem smallest_possible_AC :
  ∃ AC : ℕ, (∃ CD : ℕ, is_prime CD ∧ CD < 10 ∧ AC = (CD^2 + BD_squared) / (2 * CD)) ∧ AC = 18 :=
by
  sorry

end NUMINAMATH_GPT_smallest_possible_AC_l833_83323


namespace NUMINAMATH_GPT_molecular_weight_of_compound_l833_83370

def hydrogen_atomic_weight : ℝ := 1.008
def chromium_atomic_weight : ℝ := 51.996
def oxygen_atomic_weight : ℝ := 15.999

def compound_molecular_weight (h_atoms : ℕ) (cr_atoms : ℕ) (o_atoms : ℕ) : ℝ :=
  h_atoms * hydrogen_atomic_weight + cr_atoms * chromium_atomic_weight + o_atoms * oxygen_atomic_weight

theorem molecular_weight_of_compound :
  compound_molecular_weight 2 1 4 = 118.008 :=
by
  sorry

end NUMINAMATH_GPT_molecular_weight_of_compound_l833_83370


namespace NUMINAMATH_GPT_milk_mixture_l833_83300

theorem milk_mixture:
  ∀ (x : ℝ), 0.40 * x + 1.6 = 0.20 * (x + 16) → x = 8 := 
by
  intro x
  sorry

end NUMINAMATH_GPT_milk_mixture_l833_83300


namespace NUMINAMATH_GPT_tan_identity_l833_83374

theorem tan_identity
  (α : ℝ)
  (h : Real.tan (π / 3 - α) = 1 / 3) :
  Real.tan (2 * π / 3 + α) = -1 / 3 := 
sorry

end NUMINAMATH_GPT_tan_identity_l833_83374


namespace NUMINAMATH_GPT_amount_leaked_during_repairs_l833_83317

theorem amount_leaked_during_repairs:
  let total_leaked := 6206
  let leaked_before_repairs := 2475
  total_leaked - leaked_before_repairs = 3731 :=
by
  sorry

end NUMINAMATH_GPT_amount_leaked_during_repairs_l833_83317


namespace NUMINAMATH_GPT_find_theta_l833_83364

theorem find_theta (θ : Real) : 
  (0 ≤ θ ∧ θ ≤ 2 * Real.pi) →
  (∀ x : Real, 0 ≤ x ∧ x ≤ 1 → 
    x ^ 3 * Real.sin θ + x ^ 2 * Real.cos θ - x * (1 - x) + (1 - x) ^ 2 * Real.sin θ > 0) → 
  Real.sin θ > 0 → 
  Real.cos θ + Real.sin θ > 0 → 
  (Real.pi / 12 < θ ∧ θ < 5 * Real.pi / 12) :=
by
  intro θ_range all_x_condition sin_pos cos_sin_pos
  sorry

end NUMINAMATH_GPT_find_theta_l833_83364


namespace NUMINAMATH_GPT_chords_intersect_probability_l833_83326

noncomputable def probability_chords_intersect (n m : ℕ) : ℚ :=
  if (n > 6 ∧ m = 2023) then
    1 / 72
  else
    0

theorem chords_intersect_probability :
  probability_chords_intersect 6 2023 = 1 / 72 :=
by
  sorry

end NUMINAMATH_GPT_chords_intersect_probability_l833_83326


namespace NUMINAMATH_GPT_inequality_proof_l833_83318

theorem inequality_proof
  (a1 b1 c1 a2 b2 c2 : ℝ)
  (ha1 : 0 < a1) (hb1 : 0 < b1) (hc1 : 0 < c1)
  (ha2 : 0 < a2) (hb2 : 0 < b2) (hc2 : 0 < c2)
  (h1: b1^2 ≤ a1 * c1)
  (h2: b2^2 ≤ a2 * c2) :
  (a1 + a2 + 5) * (c1 + c2 + 2) > (b1 + b2 + 3)^2 :=
by
  sorry

end NUMINAMATH_GPT_inequality_proof_l833_83318


namespace NUMINAMATH_GPT_smallest_prime_divisor_of_sum_of_powers_l833_83350

theorem smallest_prime_divisor_of_sum_of_powers :
  ∃ p, Prime p ∧ p = Nat.gcd (3 ^ 25 + 11 ^ 19) 2 := by
  sorry

end NUMINAMATH_GPT_smallest_prime_divisor_of_sum_of_powers_l833_83350


namespace NUMINAMATH_GPT_center_of_circle_l833_83397

theorem center_of_circle :
  ∀ (x y : ℝ), (x - 1)^2 + (y - 1)^2 = 2 → (x, y) = (1, 1) :=
by
  sorry

end NUMINAMATH_GPT_center_of_circle_l833_83397


namespace NUMINAMATH_GPT_propositions_true_false_l833_83329

theorem propositions_true_false :
  (∃ x : ℝ, x ^ 3 < 1) ∧ 
  ¬ (∃ x : ℚ, x ^ 2 = 2) ∧ 
  ¬ (∀ x : ℕ, x ^ 3 > x ^ 2) ∧ 
  (∀ x : ℝ, x ^ 2 + 1 > 0) :=
by
  sorry

end NUMINAMATH_GPT_propositions_true_false_l833_83329


namespace NUMINAMATH_GPT_find_weight_l833_83384

-- Define the weight of each box before taking out 20 kg as W
variable (W : ℚ)

-- Define the condition given in the problem
def condition : Prop := 7 * (W - 20) = 3 * W

-- The proof goal is to prove W = 35 under the given condition
theorem find_weight (h : condition W) : W = 35 := by
  sorry

end NUMINAMATH_GPT_find_weight_l833_83384


namespace NUMINAMATH_GPT_basketball_shots_l833_83393

variable (x y : ℕ)

theorem basketball_shots : 3 * x + 2 * y = 26 ∧ x + y = 11 → x = 4 :=
by
  intros h
  sorry

end NUMINAMATH_GPT_basketball_shots_l833_83393


namespace NUMINAMATH_GPT_sin_eq_product_one_eighth_l833_83316

open Real

theorem sin_eq_product_one_eighth :
  (∀ (n k m : ℕ), 1 ≤ n → n ≤ 5 → 1 ≤ k → k ≤ 5 → 1 ≤ m → m ≤ 5 →
    sin (π * n / 12) * sin (π * k / 12) * sin (π * m / 12) = 1 / 8) ↔ (n = 2 ∧ k = 2 ∧ m = 2) := by
  sorry

end NUMINAMATH_GPT_sin_eq_product_one_eighth_l833_83316


namespace NUMINAMATH_GPT_problem_l833_83379

theorem problem (w x y z : ℕ) (h : 3^w * 5^x * 7^y * 11^z = 2310) : 3 * w + 5 * x + 7 * y + 11 * z = 26 :=
sorry

end NUMINAMATH_GPT_problem_l833_83379


namespace NUMINAMATH_GPT_first_term_of_arithmetic_sequence_l833_83321

theorem first_term_of_arithmetic_sequence (a : ℕ → ℚ) (S : ℕ → ℚ) (d : ℚ)
  (h1 : a 3 = 3) (h2 : S 9 - S 6 = 27)
  (h3 : ∀ n, a n = a 1 + (n - 1) * d)
  (h4 : ∀ n, S n = n * (a 1 + a n) / 2) : a 1 = 3 / 5 :=
by
  sorry

end NUMINAMATH_GPT_first_term_of_arithmetic_sequence_l833_83321


namespace NUMINAMATH_GPT_number_of_planting_methods_l833_83346

theorem number_of_planting_methods :
  let vegetables := ["cucumbers", "cabbages", "rape", "flat beans"]
  let plots := ["plot1", "plot2", "plot3"]
  (∀ v ∈ vegetables, v = "cucumbers") →
  (∃! n : ℕ, n = 18)
:= by
  sorry

end NUMINAMATH_GPT_number_of_planting_methods_l833_83346


namespace NUMINAMATH_GPT_remainder_when_divided_by_44_l833_83392

theorem remainder_when_divided_by_44 (N Q R : ℕ) :
  (N = 44 * 432 + R) ∧ (N = 39 * Q + 15) → R = 0 :=
by
  sorry

end NUMINAMATH_GPT_remainder_when_divided_by_44_l833_83392


namespace NUMINAMATH_GPT_exponent_rule_example_l833_83376

theorem exponent_rule_example : 3000 * (3000 ^ 3000) = 3000 ^ 3001 := by
  sorry

end NUMINAMATH_GPT_exponent_rule_example_l833_83376


namespace NUMINAMATH_GPT_power_function_value_l833_83338

theorem power_function_value (f : ℝ → ℝ) (h : ∃ a : ℝ, ∀ x : ℝ, f x = x ^ a) (h₁ : f 4 = 1 / 2) :
  f (1 / 16) = 4 :=
sorry

end NUMINAMATH_GPT_power_function_value_l833_83338


namespace NUMINAMATH_GPT_scores_greater_than_18_l833_83315

noncomputable def olympiad_scores (scores : Fin 20 → ℕ) :=
∀ i j k : Fin 20, i ≠ j → i ≠ k → j ≠ k → scores i < scores j + scores k

theorem scores_greater_than_18 (scores : Fin 20 → ℕ) (h1 : ∀ i j, i < j → scores i < scores j)
  (h2 : olympiad_scores scores) : ∀ i, 18 < scores i :=
by
  intro i
  sorry

end NUMINAMATH_GPT_scores_greater_than_18_l833_83315


namespace NUMINAMATH_GPT_find_f_at_4_l833_83343

theorem find_f_at_4 (f : ℝ → ℝ) (h : ∀ x : ℝ, x ≠ 0 → 3 * f x - 2 * f (1 / x) = x) : 
  f 4 = 5 / 2 :=
sorry

end NUMINAMATH_GPT_find_f_at_4_l833_83343


namespace NUMINAMATH_GPT_cubic_sum_expression_l833_83380

theorem cubic_sum_expression (x y z p q r : ℝ) (h1 : x * y = p) (h2 : x * z = q) (h3 : y * z = r) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0) :
  x^3 + y^3 + z^3 = (p^2 * q^2 + p^2 * r^2 + q^2 * r^2) / (p * q * r) :=
by
  sorry

end NUMINAMATH_GPT_cubic_sum_expression_l833_83380


namespace NUMINAMATH_GPT_jackson_entertainment_expense_l833_83348

noncomputable def total_spent_on_entertainment_computer_game_original_price : ℝ :=
  66 / 0.85

noncomputable def movie_ticket_price_with_tax : ℝ :=
  12 * 1.10

noncomputable def total_movie_tickets_cost : ℝ :=
  3 * movie_ticket_price_with_tax

noncomputable def total_snacks_and_transportation_cost : ℝ :=
  7 + 5

noncomputable def total_spent : ℝ :=
  66 + total_movie_tickets_cost + total_snacks_and_transportation_cost

theorem jackson_entertainment_expense :
  total_spent = 117.60 :=
by
  sorry

end NUMINAMATH_GPT_jackson_entertainment_expense_l833_83348


namespace NUMINAMATH_GPT_horse_cow_difference_l833_83337

def initial_conditions (h c : ℕ) : Prop :=
  4 * c = h

def transaction (h c : ℕ) : Prop :=
  (h - 15) * 7 = (c + 15) * 13

def final_difference (h c : ℕ) : Prop := 
  h - 15 - (c + 15) = 30

theorem horse_cow_difference (h c : ℕ) (hc : initial_conditions h c) (ht : transaction h c) : final_difference h c :=
    by
      sorry

end NUMINAMATH_GPT_horse_cow_difference_l833_83337


namespace NUMINAMATH_GPT_negation_of_forall_log_gt_one_l833_83344

noncomputable def negation_of_p : Prop :=
∃ x : ℝ, Real.log x ≤ 1

theorem negation_of_forall_log_gt_one :
  (¬ (∀ x : ℝ, Real.log x > 1)) ↔ negation_of_p :=
by
  sorry

end NUMINAMATH_GPT_negation_of_forall_log_gt_one_l833_83344


namespace NUMINAMATH_GPT_find_a_and_mono_l833_83309

open Real

noncomputable def f (x : ℝ) (a : ℝ) := (a * 2^x + a - 2) / (2^x + 1)

theorem find_a_and_mono :
  (∀ x : ℝ, f x a + f (-x) a = 0) →
  a = 1 ∧ f 3 1 = 7 / 9 ∧ ∀ x1 x2 : ℝ, x1 < x2 → f x1 1 < f x2 1 :=
by
  sorry

end NUMINAMATH_GPT_find_a_and_mono_l833_83309


namespace NUMINAMATH_GPT_jose_share_of_profit_l833_83327

-- Definitions from problem conditions
def tom_investment : ℕ := 30000
def jose_investment : ℕ := 45000
def profit : ℕ := 27000
def months_total : ℕ := 12
def months_jose_investment : ℕ := 10

-- Derived calculations
def tom_month_investment := tom_investment * months_total
def jose_month_investment := jose_investment * months_jose_investment
def total_month_investment := tom_month_investment + jose_month_investment

-- Prove Jose's share of profit
theorem jose_share_of_profit : (jose_month_investment * profit) / total_month_investment = 15000 := by
  -- This is where the step-by-step proof would go
  sorry

end NUMINAMATH_GPT_jose_share_of_profit_l833_83327


namespace NUMINAMATH_GPT_boat_speed_in_still_water_l833_83363

theorem boat_speed_in_still_water (B S : ℝ) (h1 : B + S = 13) (h2 : B - S = 9) : B = 11 :=
by
  sorry

end NUMINAMATH_GPT_boat_speed_in_still_water_l833_83363


namespace NUMINAMATH_GPT_fixed_point_on_line_l833_83389

theorem fixed_point_on_line (m x y : ℝ) (h : ∀ m : ℝ, m * x - y + 2 * m + 1 = 0) : 
  (x = -2 ∧ y = 1) :=
sorry

end NUMINAMATH_GPT_fixed_point_on_line_l833_83389


namespace NUMINAMATH_GPT_line_AB_eq_x_plus_3y_zero_l833_83324

/-- 
Consider two circles defined by:
C1: x^2 + y^2 - 4x + 6y = 0
C2: x^2 + y^2 - 6x = 0

Prove that the equation of the line through the intersection points of these two circles (line AB)
is x + 3y = 0.
-/
theorem line_AB_eq_x_plus_3y_zero (x y : ℝ) :
  (x^2 + y^2 - 4 * x + 6 * y = 0) ∧ (x^2 + y^2 - 6 * x = 0) → (x + 3 * y = 0) :=
by
  sorry

end NUMINAMATH_GPT_line_AB_eq_x_plus_3y_zero_l833_83324


namespace NUMINAMATH_GPT_max_dance_counts_possible_l833_83359

noncomputable def max_dance_counts : ℕ := 29

theorem max_dance_counts_possible (boys girls : ℕ) (dance_count : ℕ → ℕ) :
   boys = 29 → girls = 15 → 
   (∀ b, b < boys → dance_count b ≤ girls) → 
   (∀ g, g < girls → ∃ d, d ≤ boys ∧ dance_count d = g) →
   (∃ d, d ≤ max_dance_counts ∧
     (∀ k, k ≤ d → (∃ b, b < boys ∧ dance_count b = k) ∨ (∃ g, g < girls ∧ dance_count g = k))) := 
sorry

end NUMINAMATH_GPT_max_dance_counts_possible_l833_83359


namespace NUMINAMATH_GPT_age_ratio_l833_83303

theorem age_ratio (B A : ℕ) (h1 : B = 4) (h2 : A - B = 12) :
  A / B = 4 :=
by
  sorry

end NUMINAMATH_GPT_age_ratio_l833_83303


namespace NUMINAMATH_GPT_largest_number_is_89_l833_83328

theorem largest_number_is_89 (a b c d : ℕ) 
  (h1 : a + b + c = 180) 
  (h2 : a + b + d = 197) 
  (h3 : a + c + d = 208) 
  (h4 : b + c + d = 222) : 
  max a (max b (max c d)) = 89 := 
by sorry

end NUMINAMATH_GPT_largest_number_is_89_l833_83328


namespace NUMINAMATH_GPT_intersect_P_Q_l833_83352

open Set

def P : Set ℤ := { x | (x - 3) * (x - 6) ≤ 0 }
def Q : Set ℤ := { 5, 7 }

theorem intersect_P_Q : P ∩ Q = {5} :=
sorry

end NUMINAMATH_GPT_intersect_P_Q_l833_83352


namespace NUMINAMATH_GPT_positive_integer_pairs_divisibility_l833_83340

theorem positive_integer_pairs_divisibility (a b : ℕ) (h : a * b^2 + b + 7 ∣ a^2 * b + a + b) :
  (a = 11 ∧ b = 1) ∨ (a = 49 ∧ b = 1) ∨ ∃ k : ℕ, k > 0 ∧ a = 7 * k^2 ∧ b = 7 * k :=
sorry

end NUMINAMATH_GPT_positive_integer_pairs_divisibility_l833_83340


namespace NUMINAMATH_GPT_Larry_sessions_per_day_eq_2_l833_83377

variable (x : ℝ)
variable (sessions_per_day_time : ℝ)
variable (feeding_time_per_day : ℝ)
variable (total_time_per_day : ℝ)

theorem Larry_sessions_per_day_eq_2
  (h1: sessions_per_day_time = 30 * x)
  (h2: feeding_time_per_day = 12)
  (h3: total_time_per_day = 72) :
  x = 2 := by
  sorry

end NUMINAMATH_GPT_Larry_sessions_per_day_eq_2_l833_83377


namespace NUMINAMATH_GPT_min_value_of_quadratic_l833_83306

theorem min_value_of_quadratic : ∃ x : ℝ, 7 * x^2 - 28 * x + 1702 = 1674 ∧ ∀ y : ℝ, 7 * y^2 - 28 * y + 1702 ≥ 1674 :=
by
  sorry

end NUMINAMATH_GPT_min_value_of_quadratic_l833_83306


namespace NUMINAMATH_GPT_minimum_value_of_f_l833_83301

noncomputable def f (x : ℝ) : ℝ := x + 1 / (x - 2)

theorem minimum_value_of_f :
  (∀ x : ℝ, x > 2 → f x ≥ 4) ∧ (∃ x : ℝ, x > 2 ∧ f x = 4) :=
by {
  sorry
}

end NUMINAMATH_GPT_minimum_value_of_f_l833_83301


namespace NUMINAMATH_GPT_trajectory_of_center_l833_83360

-- Define the fixed circle C as x^2 + (y + 3)^2 = 1
def fixed_circle (p : ℝ × ℝ) : Prop :=
  (p.1)^2 + (p.2 + 3)^2 = 1

-- Define the line y = 2
def tangent_line (p : ℝ × ℝ) : Prop :=
  p.2 = 2

-- The main theorem stating the trajectory of the center of circle M is x^2 = -12y
theorem trajectory_of_center :
  ∀ (M : ℝ × ℝ), 
  tangent_line M → (∃ r : ℝ, fixed_circle (M.1, M.2 - r) ∧ r > 0) →
  (M.1)^2 = -12 * M.2 :=
sorry

end NUMINAMATH_GPT_trajectory_of_center_l833_83360


namespace NUMINAMATH_GPT_binom_16_12_eq_1820_l833_83354

theorem binom_16_12_eq_1820 : Nat.choose 16 12 = 1820 :=
by
  sorry

end NUMINAMATH_GPT_binom_16_12_eq_1820_l833_83354


namespace NUMINAMATH_GPT_prob_correct_l833_83367

noncomputable def prob_train_there_when_sam_arrives : ℚ :=
  let total_area := (60 : ℚ) * 60
  let triangle_area := (1 / 2 : ℚ) * 15 * 15
  let parallelogram_area := (30 : ℚ) * 15
  let shaded_area := triangle_area + parallelogram_area
  shaded_area / total_area

theorem prob_correct : prob_train_there_when_sam_arrives = 25 / 160 :=
  sorry

end NUMINAMATH_GPT_prob_correct_l833_83367


namespace NUMINAMATH_GPT_increase_by_150_percent_l833_83386

theorem increase_by_150_percent (x : ℝ) (h : x = 80) : x + (1.5 * x) = 200 := 
by
  -- The proof goes here, but is omitted as per instructions.
  sorry

end NUMINAMATH_GPT_increase_by_150_percent_l833_83386


namespace NUMINAMATH_GPT_geometric_series_sum_l833_83375

theorem geometric_series_sum :
  ∀ (a r : ℚ) (n : ℕ), 
  a = 1 / 5 → 
  r = -1 / 5 → 
  n = 6 →
  (a - a * r^n) / (1 - r) = 1562 / 9375 :=
by 
  intro a r n ha hr hn
  rw [ha, hr, hn]
  sorry

end NUMINAMATH_GPT_geometric_series_sum_l833_83375


namespace NUMINAMATH_GPT_option_d_correct_l833_83304

variable (a b : ℝ)

theorem option_d_correct : (-a^3)^4 = a^(12) := by sorry

end NUMINAMATH_GPT_option_d_correct_l833_83304


namespace NUMINAMATH_GPT_sum_of_cube_faces_l833_83382

theorem sum_of_cube_faces (a b c d e f : ℕ) (h_pos: 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d ∧ 0 < e ∧ 0 < f) 
    (h_eq_sum: (a * b * c) + (a * e * c) + (a * b * f) + (a * e * f) + (d * b * c) + (d * e * c) + (d * b * f) + (d * e * f) = 1089) :
    a + b + c + d + e + f = 31 := 
by
  sorry

end NUMINAMATH_GPT_sum_of_cube_faces_l833_83382


namespace NUMINAMATH_GPT_max_value_of_symmetric_f_l833_83378

noncomputable def f (x : ℝ) (a b : ℝ) : ℝ :=
  (1 - x^2) * (x^2 + a * x + b)

theorem max_value_of_symmetric_f :
  ∀ (a b : ℝ),
    (f 1 a b = 0) →
    (f (-1) a b = 0) →
    (f (-5) a b = 0) →
    (f (-3) a b = 0) →
    (∃ x : ℝ, f x 8 15 = 16) :=
by
  sorry

end NUMINAMATH_GPT_max_value_of_symmetric_f_l833_83378
