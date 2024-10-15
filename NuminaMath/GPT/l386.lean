import Mathlib

namespace NUMINAMATH_GPT_algebraic_expression_value_l386_38652

variable (a b : ℝ)

theorem algebraic_expression_value
  (h : a^2 + 2 * b^2 - 1 = 0) :
  (a - b)^2 + b * (2 * a + b) = 1 :=
by
  sorry

end NUMINAMATH_GPT_algebraic_expression_value_l386_38652


namespace NUMINAMATH_GPT_max_at_pi_six_l386_38667

theorem max_at_pi_six : ∃ (x : ℝ), (0 ≤ x ∧ x ≤ π / 2) ∧ (∀ y, (0 ≤ y ∧ y ≤ π / 2) → (x + 2 * Real.cos x) ≥ (y + 2 * Real.cos y)) ∧ x = π / 6 := sorry

end NUMINAMATH_GPT_max_at_pi_six_l386_38667


namespace NUMINAMATH_GPT_product_of_possible_values_l386_38616

noncomputable def math_problem (x : ℚ) : Prop :=
  |(10 / x) - 4| = 3

theorem product_of_possible_values :
  let x1 := 10 / 7
  let x2 := 10
  (x1 * x2) = (100 / 7) :=
by
  sorry

end NUMINAMATH_GPT_product_of_possible_values_l386_38616


namespace NUMINAMATH_GPT_f_5_eq_2_l386_38628

def f : ℕ → ℤ :=
sorry

axiom f_initial_condition : f 1 = 2

axiom f_functional_eq (a b : ℕ) : f (a + b) = 2 * f a + 2 * f b - 3 * f (a * b)

theorem f_5_eq_2 : f 5 = 2 :=
sorry

end NUMINAMATH_GPT_f_5_eq_2_l386_38628


namespace NUMINAMATH_GPT_choice_first_question_range_of_P2_l386_38686

theorem choice_first_question (P1 P2 a b : ℚ) (hP1 : P1 = 1/2) (hP2 : P2 = 1/3) :
  (P1 * (1 - P2) * a + P1 * P2 * (a + b) - P2 * (1 - P1) * b - P1 * P2 * (a + b) > 0) ↔ a > b / 2 :=
sorry

theorem range_of_P2 (a b P1 P2 : ℚ) (ha : a = 10) (hb : b = 20) (hP1 : P1 = 2/5) :
  P1 * (1 - P2) * a + P1 * P2 * (a + b) - P2 * (1 - P1) * b - P1 * P2 * (a + b) ≥ 0 ↔ (0 ≤ P2 ∧ P2 ≤ P1 / (2 - P1)) :=
sorry

end NUMINAMATH_GPT_choice_first_question_range_of_P2_l386_38686


namespace NUMINAMATH_GPT_find_a_range_l386_38640

noncomputable def monotonic_func_a_range : Set ℝ :=
  {a : ℝ | ∀ x : ℝ, -2 ≤ x ∧ x ≤ 3 → (3 * x^2 + a ≥ 0 ∨ 3 * x^2 + a ≤ 0)}

theorem find_a_range :
  monotonic_func_a_range = {a | a ≤ -27} ∪ {a | a ≥ 0} :=
by
  sorry

end NUMINAMATH_GPT_find_a_range_l386_38640


namespace NUMINAMATH_GPT_problem_1_l386_38600

theorem problem_1 : (-(5 / 8) / (14 / 3) * (-(16 / 5)) / (-(6 / 7))) = -1 / 2 :=
  sorry

end NUMINAMATH_GPT_problem_1_l386_38600


namespace NUMINAMATH_GPT_dozen_pen_cost_l386_38674

-- Definitions based on the conditions
def cost_of_pen (x : ℝ) : ℝ := 5 * x
def cost_of_pencil (x : ℝ) : ℝ := x
def total_cost (x : ℝ) (y : ℝ) : ℝ := 3 * cost_of_pen x + y * cost_of_pencil x

open Classical
noncomputable def cost_dozen_pens (x : ℝ) : ℝ := 12 * cost_of_pen x

theorem dozen_pen_cost (x y : ℝ) (h : total_cost x y = 150) : cost_dozen_pens x = 60 * x :=
by
  sorry

end NUMINAMATH_GPT_dozen_pen_cost_l386_38674


namespace NUMINAMATH_GPT_inequality_no_solution_l386_38638

theorem inequality_no_solution : 
  ∀ x : ℝ, -2 < (x^2 - 10 * x + 9) / (x^2 - 4 * x + 8) ∧ (x^2 - 10 * x + 9) / (x^2 - 4 * x + 8) < 2 → false :=
by sorry

end NUMINAMATH_GPT_inequality_no_solution_l386_38638


namespace NUMINAMATH_GPT_system_solution_l386_38680

theorem system_solution (x y : ℝ) (h1 : x + 5*y = 5) (h2 : 3*x - y = 3) : x + y = 2 := 
by
  sorry

end NUMINAMATH_GPT_system_solution_l386_38680


namespace NUMINAMATH_GPT_range_of_a_l386_38614

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, x > 1 → x + 1 / (x - 1) ≥ a) → a ≤ 3 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_range_of_a_l386_38614


namespace NUMINAMATH_GPT_matching_pair_probability_l386_38699

theorem matching_pair_probability :
  let total_socks := 22
  let blue_socks := 12
  let red_socks := 10
  let total_ways := (total_socks * (total_socks - 1)) / 2
  let blue_ways := (blue_socks * (blue_socks - 1)) / 2
  let red_ways := (red_socks * (red_socks - 1)) / 2
  let matching_ways := blue_ways + red_ways
  total_ways = 231 →
  blue_ways = 66 →
  red_ways = 45 →
  matching_ways = 111 →
  (matching_ways : ℝ) / total_ways = 111 / 231 := by sorry

end NUMINAMATH_GPT_matching_pair_probability_l386_38699


namespace NUMINAMATH_GPT_find_second_number_l386_38617

theorem find_second_number 
  (h₁ : (20 + 40 + 60) / 3 = (10 + x + 15) / 3 + 5) :
  x = 80 :=
  sorry

end NUMINAMATH_GPT_find_second_number_l386_38617


namespace NUMINAMATH_GPT_no_solution_exists_l386_38692

theorem no_solution_exists (m n : ℕ) : ¬ (m^2 = n^2 + 2014) :=
by
  sorry

end NUMINAMATH_GPT_no_solution_exists_l386_38692


namespace NUMINAMATH_GPT_smallest_k_divides_polynomial_l386_38678

theorem smallest_k_divides_polynomial :
  ∃ k : ℕ, 0 < k ∧ (∀ z : ℂ, (z^10 + z^9 + z^8 + z^6 + z^5 + z^4 + z + 1) ∣ (z^k - 1)) ∧ k = 84 :=
by
  sorry

end NUMINAMATH_GPT_smallest_k_divides_polynomial_l386_38678


namespace NUMINAMATH_GPT_find_m_value_l386_38671

/-- 
If the function y = (m + 1)x^(m^2 + 3m + 4) is a quadratic function, 
then the value of m is -2.
--/
theorem find_m_value 
  (m : ℝ)
  (h1 : m^2 + 3 * m + 4 = 2)
  (h2 : m + 1 ≠ 0) : 
  m = -2 := 
sorry

end NUMINAMATH_GPT_find_m_value_l386_38671


namespace NUMINAMATH_GPT_g_extreme_values_l386_38679

-- Definitions based on the conditions
def f (x : ℝ) := x^3 - 2 * x^2 + x
def g (x : ℝ) := f x + 1

-- Theorem statement
theorem g_extreme_values : 
  (g (1/3) = 31/27) ∧ (g 1 = 1) := sorry

end NUMINAMATH_GPT_g_extreme_values_l386_38679


namespace NUMINAMATH_GPT_molecular_weight_of_Carbonic_acid_l386_38657

theorem molecular_weight_of_Carbonic_acid :
  let H_weight := 1.008
  let C_weight := 12.011
  let O_weight := 15.999
  let H_atoms := 2
  let C_atoms := 1
  let O_atoms := 3
  (H_atoms * H_weight + C_atoms * C_weight + O_atoms * O_weight) = 62.024 :=
by 
  let H_weight := 1.008
  let C_weight := 12.011
  let O_weight := 15.999
  let H_atoms := 2
  let C_atoms := 1
  let O_atoms := 3
  sorry

end NUMINAMATH_GPT_molecular_weight_of_Carbonic_acid_l386_38657


namespace NUMINAMATH_GPT_sales_price_reduction_l386_38689

theorem sales_price_reduction
  (current_sales : ℝ := 20)
  (current_profit_per_shirt : ℝ := 40)
  (sales_increase_per_dollar : ℝ := 2)
  (desired_profit : ℝ := 1200) :
  ∃ x : ℝ, (40 - x) * (20 + 2 * x) = 1200 ∧ x = 20 :=
by
  use 20
  sorry

end NUMINAMATH_GPT_sales_price_reduction_l386_38689


namespace NUMINAMATH_GPT_max_value_of_f_l386_38604

noncomputable def f (t : ℝ) : ℝ := ((2^(t+1) - 4*t) * t) / (16^t)

theorem max_value_of_f : ∃ t : ℝ, ∀ u : ℝ, f u ≤ f t ∧ f t = 1 / 16 := by
  sorry

end NUMINAMATH_GPT_max_value_of_f_l386_38604


namespace NUMINAMATH_GPT_overtaking_time_l386_38627

theorem overtaking_time :
  ∀ t t_k : ℕ,
  (30 * t = 40 * (t - 5)) ∧ 
  (30 * t = 60 * t_k) →
  t = 20 ∧ t_k = 10 ∧ (20 - 10 = 10) :=
by
  sorry

end NUMINAMATH_GPT_overtaking_time_l386_38627


namespace NUMINAMATH_GPT_rectangular_field_length_l386_38672

noncomputable def area_triangle (base height : ℝ) : ℝ :=
  (base * height) / 2

noncomputable def length_rectangle (area width : ℝ) : ℝ :=
  area / width

theorem rectangular_field_length (base height width : ℝ) (h_base : base = 7.2) (h_height : height = 7) (h_width : width = 4) :
  length_rectangle (area_triangle base height) width = 6.3 :=
by
  -- sorry would be replaced by the actual proof.
  sorry

end NUMINAMATH_GPT_rectangular_field_length_l386_38672


namespace NUMINAMATH_GPT_am_gm_inequality_l386_38658

-- Let's define the problem statement
theorem am_gm_inequality (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) 
  (h : (a + 1) * (b + 1) * (c + 1) = 8) : a + b + c ≥ 3 := by
  sorry

end NUMINAMATH_GPT_am_gm_inequality_l386_38658


namespace NUMINAMATH_GPT_infinitely_many_n_divisible_by_n_squared_l386_38647

theorem infinitely_many_n_divisible_by_n_squared :
  ∃ (n : ℕ → ℕ), (∀ k : ℕ, 0 < n k) ∧ (∀ k : ℕ, n k^2 ∣ 2^(n k) + 3^(n k)) :=
sorry

end NUMINAMATH_GPT_infinitely_many_n_divisible_by_n_squared_l386_38647


namespace NUMINAMATH_GPT_probability_dice_sum_12_l386_38629

def total_outcomes : ℕ := 216
def favorable_outcomes : ℕ := 25

theorem probability_dice_sum_12 :
  (favorable_outcomes : ℚ) / total_outcomes = 25 / 216 := by
  sorry

end NUMINAMATH_GPT_probability_dice_sum_12_l386_38629


namespace NUMINAMATH_GPT_sin_2017pi_div_3_l386_38687

theorem sin_2017pi_div_3 : Real.sin (2017 * Real.pi / 3) = Real.sqrt 3 / 2 := 
  sorry

end NUMINAMATH_GPT_sin_2017pi_div_3_l386_38687


namespace NUMINAMATH_GPT_range_of_x_l386_38630

def interval1 : Set ℝ := {x | 2 ≤ x ∧ x ≤ 5}
def interval2 : Set ℝ := {x | x < 1 ∨ x > 4}
def false_statement (x : ℝ) : Prop := x ∈ interval1 ∨ x ∈ interval2

theorem range_of_x (x : ℝ) (h : ¬ false_statement x) : x ∈ Set.Ico 1 2 :=
by
  sorry

end NUMINAMATH_GPT_range_of_x_l386_38630


namespace NUMINAMATH_GPT_minimum_value_y_l386_38642

noncomputable def y (x : ℚ) : ℚ := |3 - x| + |x - 2| + |-1 + x|

theorem minimum_value_y : ∃ x : ℚ, y x = 2 :=
by
  sorry

end NUMINAMATH_GPT_minimum_value_y_l386_38642


namespace NUMINAMATH_GPT_books_loaned_out_l386_38677

theorem books_loaned_out (initial_books returned_percent final_books : ℕ) (h1 : initial_books = 75) (h2 : returned_percent = 65) (h3 : final_books = 61) : 
  ∃ x : ℕ, initial_books - final_books = x - (returned_percent * x / 100) ∧ x = 40 :=
by {
  sorry 
}

end NUMINAMATH_GPT_books_loaned_out_l386_38677


namespace NUMINAMATH_GPT_ammonium_nitrate_formed_l386_38661

-- Definitions based on conditions in the problem
def NH3_moles : ℕ := 3
def HNO3_moles (NH3 : ℕ) : ℕ := NH3 -- 1:1 molar ratio with NH3 for HNO3

-- Definition of the outcome
def NH4NO3_moles (NH3 NH4NO3 : ℕ) : Prop :=
  NH4NO3 = NH3

-- The theorem to prove that 3 moles of NH3 combined with sufficient HNO3 produces 3 moles of NH4NO3
theorem ammonium_nitrate_formed (NH3 NH4NO3 : ℕ) (h : NH3 = 3) :
  NH4NO3_moles NH3 NH4NO3 → NH4NO3 = 3 :=
by
  intro hn
  rw [h] at hn
  exact hn

end NUMINAMATH_GPT_ammonium_nitrate_formed_l386_38661


namespace NUMINAMATH_GPT_sample_size_of_survey_l386_38651

def eighth_grade_students : ℕ := 350
def selected_students : ℕ := 50

theorem sample_size_of_survey : selected_students = 50 :=
by sorry

end NUMINAMATH_GPT_sample_size_of_survey_l386_38651


namespace NUMINAMATH_GPT_actual_length_of_tunnel_in_km_l386_38691

-- Define the conditions
def scale_factor : ℝ := 30000
def length_on_map_cm : ℝ := 7

-- Using the conditions, we need to prove the actual length is 2.1 km
theorem actual_length_of_tunnel_in_km :
  (length_on_map_cm * scale_factor / 100000) = 2.1 :=
by sorry

end NUMINAMATH_GPT_actual_length_of_tunnel_in_km_l386_38691


namespace NUMINAMATH_GPT_initial_gift_card_value_l386_38644

-- The price per pound of coffee
def cost_per_pound : ℝ := 8.58

-- The number of pounds of coffee bought by Rita
def pounds_bought : ℝ := 4.0

-- The remaining balance on Rita's gift card after buying coffee
def remaining_balance : ℝ := 35.68

-- The total cost of the coffee Rita bought
def total_cost_of_coffee : ℝ := cost_per_pound * pounds_bought

-- The initial value of Rita's gift card
def initial_value_of_gift_card : ℝ := total_cost_of_coffee + remaining_balance

-- Statement of the proof problem
theorem initial_gift_card_value : initial_value_of_gift_card = 70.00 :=
by
  -- Placeholder for the proof
  sorry

end NUMINAMATH_GPT_initial_gift_card_value_l386_38644


namespace NUMINAMATH_GPT_length_of_CD_l386_38611

theorem length_of_CD (L : ℝ) (r : ℝ) (V_total : ℝ) (cylinder_vol : ℝ) (hemisphere_vol : ℝ) : 
  r = 5 ∧ V_total = 900 * Real.pi ∧ cylinder_vol = Real.pi * r^2 * L ∧ hemisphere_vol = (2/3) *Real.pi * r^3 → 
  V_total = cylinder_vol + 2 * hemisphere_vol → 
  L = 88 / 3 := 
by
  sorry

end NUMINAMATH_GPT_length_of_CD_l386_38611


namespace NUMINAMATH_GPT_age_difference_constant_l386_38608

theorem age_difference_constant (a b x : ℕ) : (a + x) - (b + x) = a - b :=
by
  sorry

end NUMINAMATH_GPT_age_difference_constant_l386_38608


namespace NUMINAMATH_GPT_xy_identity_l386_38668

theorem xy_identity (x y : ℝ) (h : 4 * x^2 - 5 * x * y + 4 * y^2 = 5) : x^2 + y^2 = 5 :=
  sorry

end NUMINAMATH_GPT_xy_identity_l386_38668


namespace NUMINAMATH_GPT_first_term_exceeding_1000_l386_38633

variable (a₁ : Int := 2)
variable (d : Int := 3)

def arithmetic_sequence (n : Int) : Int :=
  a₁ + (n - 1) * d

theorem first_term_exceeding_1000 :
  ∃ n : Int, n = 334 ∧ arithmetic_sequence n > 1000 := by
  sorry

end NUMINAMATH_GPT_first_term_exceeding_1000_l386_38633


namespace NUMINAMATH_GPT_card_area_after_shortening_l386_38662

/-- Given a card with dimensions 3 inches by 7 inches, prove that 
  if the length is shortened by 1 inch and the width is shortened by 2 inches, 
  then the resulting area is 10 square inches. -/
theorem card_area_after_shortening :
  let length := 3
  let width := 7
  let new_length := length - 1
  let new_width := width - 2
  new_length * new_width = 10 :=
by
  let length := 3
  let width := 7
  let new_length := length - 1
  let new_width := width - 2
  show new_length * new_width = 10
  sorry

end NUMINAMATH_GPT_card_area_after_shortening_l386_38662


namespace NUMINAMATH_GPT_value_of_a4_l386_38637

open Nat

def sequence (a : ℕ → ℕ) : Prop :=
  a 1 = 2 ∧ ∀ n, a (n + 1) = 2 * a n + 1

theorem value_of_a4 (a : ℕ → ℕ) (h : sequence a) : a 4 = 23 :=
by
  -- Proof to be provided or implemented
  sorry

end NUMINAMATH_GPT_value_of_a4_l386_38637


namespace NUMINAMATH_GPT_probability_face_not_red_is_five_sixths_l386_38603

-- Definitions based on the conditions
def total_faces : ℕ := 6
def green_faces : ℕ := 3
def blue_faces : ℕ := 2
def red_faces : ℕ := 1

-- Definition for the probability calculation
def probability_not_red (total : ℕ) (not_red : ℕ) : ℚ := not_red / total

-- The main statement to prove
theorem probability_face_not_red_is_five_sixths :
  probability_not_red total_faces (green_faces + blue_faces) = 5 / 6 :=
by sorry

end NUMINAMATH_GPT_probability_face_not_red_is_five_sixths_l386_38603


namespace NUMINAMATH_GPT_geometric_sequence_common_ratio_l386_38694

theorem geometric_sequence_common_ratio (a : ℕ → ℝ)
  (h : ∀ n, a n * a (n + 1) = 16^n) :
  ∃ r : ℝ, r = 4 ∧ ∀ n, a (n + 1) = a n * r :=
sorry

end NUMINAMATH_GPT_geometric_sequence_common_ratio_l386_38694


namespace NUMINAMATH_GPT_rational_sum_zero_l386_38676

theorem rational_sum_zero {a b c : ℚ} (h : (a + b + c) * (a + b - c) = 4 * c^2) : a + b = 0 := 
sorry

end NUMINAMATH_GPT_rational_sum_zero_l386_38676


namespace NUMINAMATH_GPT_find_2a_plus_b_l386_38673

open Real

-- Define the given conditions
variables (a b : ℝ)

-- a and b are acute angles
axiom acute_a : 0 < a ∧ a < π / 2
axiom acute_b : 0 < b ∧ b < π / 2

axiom condition1 : 4 * sin a ^ 2 + 3 * sin b ^ 2 = 1
axiom condition2 : 4 * sin (2 * a) - 3 * sin (2 * b) = 0

-- Define the theorem we want to prove
theorem find_2a_plus_b : 2 * a + b = π / 2 :=
sorry

end NUMINAMATH_GPT_find_2a_plus_b_l386_38673


namespace NUMINAMATH_GPT_man_wage_l386_38650

variable (m w b : ℝ) -- wages of man, woman, boy respectively
variable (W : ℝ) -- number of women equivalent to 5 men and 8 boys

-- Conditions given in the problem
axiom condition1 : 5 * m = W * w
axiom condition2 : W * w = 8 * b
axiom condition3 : 5 * m + 8 * b + 8 * b = 90

-- Prove the wage of one man
theorem man_wage : m = 6 := 
by
  -- proof steps would be here, but skipped as per instructions
  sorry

end NUMINAMATH_GPT_man_wage_l386_38650


namespace NUMINAMATH_GPT_abc_inequality_l386_38606

variable {a b c : ℝ}

theorem abc_inequality (h₀ : a ≠ b ∧ b ≠ c ∧ c ≠ a)
  (h₁ : a + b + c = 6)
  (h₂ : a * b + b * c + c * a = 9) :
  0 < a * b * c ∧ a * b * c < 4 := by
  sorry

end NUMINAMATH_GPT_abc_inequality_l386_38606


namespace NUMINAMATH_GPT_trillion_value_l386_38697

def ten_thousand : ℕ := 10^4
def million : ℕ := 10^6
def billion : ℕ := ten_thousand * million

theorem trillion_value : (ten_thousand * ten_thousand * billion) = 10^16 :=
by
  sorry

end NUMINAMATH_GPT_trillion_value_l386_38697


namespace NUMINAMATH_GPT_pentagon_probability_l386_38609

/-- Ten points are equally spaced around the circumference of a regular pentagon,
with each side being divided into two equal segments.

We need to prove that the probability of choosing two points randomly and
having them be exactly one side of the pentagon apart is 2/9.
-/
theorem pentagon_probability : 
  let total_points := 10
  let favorable_pairs := 10
  let total_pairs := total_points * (total_points - 1) / 2
  (favorable_pairs / total_pairs : ℚ) = 2 / 9 :=
by
  sorry

end NUMINAMATH_GPT_pentagon_probability_l386_38609


namespace NUMINAMATH_GPT_problem_l386_38612

theorem problem (x : ℝ) (h : 8 * x = 3) : 200 * (1 / x) = 533.33 := by
  sorry

end NUMINAMATH_GPT_problem_l386_38612


namespace NUMINAMATH_GPT_total_servings_l386_38623

/-- The first jar contains 24 2/3 tablespoons of peanut butter. -/
def first_jar_pb : ℚ := 74 / 3

/-- The second jar contains 19 1/2 tablespoons of peanut butter. -/
def second_jar_pb : ℚ := 39 / 2

/-- One serving size is 3 tablespoons. -/
def serving_size : ℚ := 3

/-- The total servings of peanut butter in both jars is 14 13/18 servings. -/
theorem total_servings : (first_jar_pb + second_jar_pb) / serving_size = 14 + 13 / 18 :=
by
  sorry

end NUMINAMATH_GPT_total_servings_l386_38623


namespace NUMINAMATH_GPT_range_of_a_l386_38683

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, |x - 3| - |x + 1| ≤ a^2 - 3 * a) ↔ a ≤ -1 ∨ 4 ≤ a := 
sorry

end NUMINAMATH_GPT_range_of_a_l386_38683


namespace NUMINAMATH_GPT_regular_n_gon_center_inside_circle_l386_38670

-- Define a regular n-gon
structure RegularNGon (n : ℕ) :=
(center : ℝ × ℝ)
(vertices : Fin n → (ℝ × ℝ))

-- Define the condition to be able to roll and reflect the n-gon over any of its sides
def canReflectSymmetrically (n : ℕ) (g : RegularNGon n) : Prop := sorry

-- Definition of a circle with a given center and radius
structure Circle :=
(center : ℝ × ℝ)
(radius : ℝ)

-- Define the problem for determining if reflection can bring the center of n-gon inside any circle
def canCenterBeInsideCircle (n : ℕ) (g : RegularNGon n) (c : Circle) : Prop :=
  ∃ (f : ℝ × ℝ → ℝ × ℝ), -- Some function representing the reflections
    canReflectSymmetrically n g ∧ f g.center = c.center

-- State the main theorem determining for which n-gons the assertion is true
theorem regular_n_gon_center_inside_circle (n : ℕ) 
  (h : n ≠ 3 ∧ n ≠ 4 ∧ n ≠ 6) : 
  ∀ (g : RegularNGon n) (c : Circle), canCenterBeInsideCircle n g c :=
sorry

end NUMINAMATH_GPT_regular_n_gon_center_inside_circle_l386_38670


namespace NUMINAMATH_GPT_sarah_problem_solution_l386_38620

def two_digit_number := {x : ℕ // 10 ≤ x ∧ x < 100}
def three_digit_number := {y : ℕ // 100 ≤ y ∧ y < 1000}

theorem sarah_problem_solution (x : two_digit_number) (y : three_digit_number) 
    (h_eq : 1000 * x.1 + y.1 = 8 * x.1 * y.1) : 
    x.1 = 15 ∧ y.1 = 126 ∧ (x.1 + y.1 = 141) := 
by 
  sorry

end NUMINAMATH_GPT_sarah_problem_solution_l386_38620


namespace NUMINAMATH_GPT_gcd_example_l386_38698

theorem gcd_example : Nat.gcd (101^11 + 1) (101^11 + 101^3 + 1) = 1 := by
  sorry

end NUMINAMATH_GPT_gcd_example_l386_38698


namespace NUMINAMATH_GPT_simplify_and_evaluate_l386_38649

-- Define the constants
def a : ℤ := -1
def b : ℤ := 2

-- Declare the expression
def expr : ℤ := 7 * a ^ 2 * b + (-4 * a ^ 2 * b + 5 * a * b ^ 2) - (2 * a ^ 2 * b - 3 * a * b ^ 2)

-- Declare the final evaluated result
def result : ℤ := 2 * ((-1 : ℤ) ^ 2) + 8 * (-1) * (2 : ℤ) ^ 2 

-- The theorem we want to prove
theorem simplify_and_evaluate : expr = result :=
by
  sorry

end NUMINAMATH_GPT_simplify_and_evaluate_l386_38649


namespace NUMINAMATH_GPT_probability_at_least_one_blown_l386_38653

theorem probability_at_least_one_blown (P_A P_B P_AB : ℝ)  
  (hP_A : P_A = 0.085) 
  (hP_B : P_B = 0.074) 
  (hP_AB : P_AB = 0.063) : 
  P_A + P_B - P_AB = 0.096 :=
by
  sorry

end NUMINAMATH_GPT_probability_at_least_one_blown_l386_38653


namespace NUMINAMATH_GPT_total_right_handed_players_l386_38643

-- Defining the conditions and the given values
def total_players : ℕ := 61
def throwers : ℕ := 37
def non_throwers : ℕ := total_players - throwers
def left_handed_non_throwers : ℕ := non_throwers / 3
def right_handed_non_throwers : ℕ := non_throwers - left_handed_non_throwers

-- The proof goal
theorem total_right_handed_players 
  (h1 : total_players = 61)
  (h2 : throwers = 37)
  (h3 : non_throwers = total_players - throwers)
  (h4 : left_handed_non_throwers = non_throwers / 3)
  (h5 : right_handed_non_throwers = non_throwers - left_handed_non_throwers)
  (h6 : left_handed_non_throwers * 3 = non_throwers)
  : throwers + right_handed_non_throwers = 53 :=
sorry

end NUMINAMATH_GPT_total_right_handed_players_l386_38643


namespace NUMINAMATH_GPT_divide_square_into_smaller_squares_l386_38685

def P (n : ℕ) : Prop := sorry /- Define the property of dividing a square into n smaller squares -/

theorem divide_square_into_smaller_squares (n : ℕ) (h : n > 5) : P n :=
  sorry

end NUMINAMATH_GPT_divide_square_into_smaller_squares_l386_38685


namespace NUMINAMATH_GPT_Thabo_owns_more_paperback_nonfiction_than_hardcover_nonfiction_l386_38610

noncomputable def Thabo_book_count_problem : Prop :=
  let P := Nat
  let F := Nat
  ∃ (P F : Nat), 
    -- Conditions
    (P > 40) ∧ 
    (F = 2 * P) ∧ 
    (F + P + 40 = 220) ∧ 
    -- Conclusion
    (P - 40 = 20)

theorem Thabo_owns_more_paperback_nonfiction_than_hardcover_nonfiction : Thabo_book_count_problem :=
  sorry

end NUMINAMATH_GPT_Thabo_owns_more_paperback_nonfiction_than_hardcover_nonfiction_l386_38610


namespace NUMINAMATH_GPT_tracy_feeds_dogs_times_per_day_l386_38682

theorem tracy_feeds_dogs_times_per_day : 
  let cups_per_meal_per_dog := 1.5
  let dogs := 2
  let total_pounds_per_day := 4
  let cups_per_pound := 2.25
  (total_pounds_per_day * cups_per_pound) / (dogs * cups_per_meal_per_dog) = 3 :=
by
  sorry

end NUMINAMATH_GPT_tracy_feeds_dogs_times_per_day_l386_38682


namespace NUMINAMATH_GPT_cost_of_each_notebook_is_3_l386_38641

noncomputable def notebooks_cost (total_spent : ℕ) (backpack_cost : ℕ) (pens_cost : ℕ) (pencils_cost : ℕ) (num_notebooks : ℕ) : ℕ :=
  (total_spent - (backpack_cost + pens_cost + pencils_cost)) / num_notebooks

theorem cost_of_each_notebook_is_3 :
  notebooks_cost 32 15 1 1 5 = 3 :=
by
  sorry

end NUMINAMATH_GPT_cost_of_each_notebook_is_3_l386_38641


namespace NUMINAMATH_GPT_range_of_y_l386_38634

theorem range_of_y (a b y : ℝ) (hab : a + b = 2) (hbl : b ≤ 2) (hy : y = a^2 + 2*a - 2) : y ≥ -2 :=
by
  sorry

end NUMINAMATH_GPT_range_of_y_l386_38634


namespace NUMINAMATH_GPT_john_ate_cookies_l386_38648

-- Definitions for conditions
def dozen := 12

-- Given conditions
def initial_cookies : ℕ := 2 * dozen
def cookies_left : ℕ := 21

-- Problem statement
theorem john_ate_cookies : initial_cookies - cookies_left = 3 :=
by
  -- Solution steps omitted, only statement provided
  sorry

end NUMINAMATH_GPT_john_ate_cookies_l386_38648


namespace NUMINAMATH_GPT_k_value_l386_38675

theorem k_value (k : ℝ) (h : (k / 4) + (-k / 3) = 2) : k = -24 :=
by
  sorry

end NUMINAMATH_GPT_k_value_l386_38675


namespace NUMINAMATH_GPT_bryden_receives_10_dollars_l386_38688

-- Define the face value of one quarter
def face_value_quarter : ℝ := 0.25

-- Define the number of quarters Bryden has
def num_quarters : ℕ := 8

-- Define the multiplier for 500%
def multiplier : ℝ := 5

-- Calculate the total face value of eight quarters
def total_face_value : ℝ := num_quarters * face_value_quarter

-- Calculate the amount Bryden will receive
def amount_received : ℝ := total_face_value * multiplier

-- The proof goal: Bryden will receive 10 dollars
theorem bryden_receives_10_dollars : amount_received = 10 :=
by
  sorry

end NUMINAMATH_GPT_bryden_receives_10_dollars_l386_38688


namespace NUMINAMATH_GPT_sarah_bought_3_bottle_caps_l386_38696

theorem sarah_bought_3_bottle_caps
  (orig_caps : ℕ)
  (new_caps : ℕ)
  (h_orig_caps : orig_caps = 26)
  (h_new_caps : new_caps = 29) :
  new_caps - orig_caps = 3 :=
by
  sorry

end NUMINAMATH_GPT_sarah_bought_3_bottle_caps_l386_38696


namespace NUMINAMATH_GPT_product_of_three_numbers_l386_38665

theorem product_of_three_numbers (x y z n : ℝ)
  (h_sum : x + y + z = 180)
  (h_n_eq_8x : n = 8 * x)
  (h_n_eq_y_minus_10 : n = y - 10)
  (h_n_eq_z_plus_10 : n = z + 10) :
  x * y * z = (180 / 17) * ((1440 / 17) ^ 2 - 100) := by
  sorry

end NUMINAMATH_GPT_product_of_three_numbers_l386_38665


namespace NUMINAMATH_GPT_minyoung_gave_nine_notebooks_l386_38636

theorem minyoung_gave_nine_notebooks (original left given : ℕ) (h1 : original = 17) (h2 : left = 8) (h3 : given = original - left) : given = 9 :=
by
  rw [h1, h2] at h3
  exact h3

end NUMINAMATH_GPT_minyoung_gave_nine_notebooks_l386_38636


namespace NUMINAMATH_GPT_min_k_period_at_least_15_l386_38613

theorem min_k_period_at_least_15 (a b : ℚ) (h_pos_a : 0 < a) (h_pos_b : 0 < b)
    (h_period_a : ∃ m, a = m / (10^30 - 1))
    (h_period_b : ∃ n, b = n / (10^30 - 1))
    (h_period_ab : ∃ p, (a - b) = p / (10^30 - 1) ∧ 10^15 + 1 ∣ p) :
    ∃ k : ℕ, k = 6 ∧ (∃ q, (a + k * b) = q / (10^30 - 1) ∧ 10^15 + 1 ∣ q) :=
sorry

end NUMINAMATH_GPT_min_k_period_at_least_15_l386_38613


namespace NUMINAMATH_GPT_solution_set_inequality_range_of_t_l386_38631

noncomputable def f (x : ℝ) : ℝ := |x| - 2 * |x + 3|

-- Problem (1)
theorem solution_set_inequality :
  { x : ℝ | f x ≥ 2 } = { x : ℝ | -4 ≤ x ∧ x ≤ - (8 / 3) } :=
by
  sorry

-- Problem (2)
theorem range_of_t (t : ℝ) :
  (∃ x : ℝ, f x - |3 * t - 2| ≥ 0) ↔ (- (1 / 3) ≤ t ∧ t ≤ 5 / 3) :=
by
  sorry

end NUMINAMATH_GPT_solution_set_inequality_range_of_t_l386_38631


namespace NUMINAMATH_GPT_parents_can_catch_ka_liang_l386_38646

-- Definitions according to the problem statement.
-- Define the condition of the roads and the speed of the participants.
def grid_with_roads : Prop :=
  ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧
  -- 4 roads forming the sides of a square with side length a
  True ∧
  -- 2 roads connecting the midpoints of opposite sides of the square
  True

def ka_liang_speed : ℝ := 2

def parent_speed : ℝ := 1

-- Condition that Ka Liang, father, and mother can see each other
def mutual_visibility (a b : ℝ) : Prop := True

-- The main proposition
theorem parents_can_catch_ka_liang (a b : ℝ) (hgrid : grid_with_roads)
    (hspeed : ka_liang_speed = 2 * parent_speed) (hvis : mutual_visibility a b) :
  True := 
sorry

end NUMINAMATH_GPT_parents_can_catch_ka_liang_l386_38646


namespace NUMINAMATH_GPT_Tom_earns_per_week_l386_38621

-- Definitions based on conditions
def crab_buckets_per_day := 8
def crabs_per_bucket := 12
def price_per_crab := 5
def days_per_week := 7

-- The proof goal
theorem Tom_earns_per_week :
  (crab_buckets_per_day * crabs_per_bucket * price_per_crab * days_per_week) = 3360 := by
  sorry

end NUMINAMATH_GPT_Tom_earns_per_week_l386_38621


namespace NUMINAMATH_GPT_Martha_blocks_end_l386_38639

variable (Ronald_blocks : ℕ) (Martha_start_blocks : ℕ) (Martha_found_blocks : ℕ)
variable (Ronald_has_blocks : Ronald_blocks = 13)
variable (Martha_has_start_blocks : Martha_start_blocks = 4)
variable (Martha_finds_more_blocks : Martha_found_blocks = 80)

theorem Martha_blocks_end : Martha_start_blocks + Martha_found_blocks = 84 :=
by
  have Martha_start_blocks := Martha_has_start_blocks
  have Martha_found_blocks := Martha_finds_more_blocks
  sorry

end NUMINAMATH_GPT_Martha_blocks_end_l386_38639


namespace NUMINAMATH_GPT_smartphone_demand_inverse_proportional_l386_38607

theorem smartphone_demand_inverse_proportional (k : ℝ) (d d' p p' : ℝ) 
  (h1 : d = 30)
  (h2 : p = 600)
  (h3 : p' = 900)
  (h4 : d * p = k) :
  d' * p' = k → d' = 20 := 
by 
  sorry

end NUMINAMATH_GPT_smartphone_demand_inverse_proportional_l386_38607


namespace NUMINAMATH_GPT_slope_and_angle_of_inclination_l386_38690

noncomputable def line_slope_and_inclination : Prop :=
  ∀ (x y : ℝ), (x - y - 3 = 0) → (∃ m : ℝ, m = 1) ∧ (∃ θ : ℝ, θ = 45)

theorem slope_and_angle_of_inclination (x y : ℝ) (h : x - y - 3 = 0) : line_slope_and_inclination :=
by
  sorry

end NUMINAMATH_GPT_slope_and_angle_of_inclination_l386_38690


namespace NUMINAMATH_GPT_determine_x_l386_38666

theorem determine_x 
  (w : ℤ) (hw : w = 90)
  (z : ℤ) (hz : z = 4 * w + 40)
  (y : ℤ) (hy : y = 3 * z + 15)
  (x : ℤ) (hx : x = 2 * y + 6) :
  x = 2436 := 
by
  sorry

end NUMINAMATH_GPT_determine_x_l386_38666


namespace NUMINAMATH_GPT_smallest_k_l386_38660

theorem smallest_k (a b : ℚ) (h_a_period : ∀ n, a ≠ (10^30 - 1) * n)
  (h_b_period : ∀ n, b ≠ (10^30 - 1) * n)
  (h_diff_period : ∀ n, a - b ≠ (10^15 - 1) * n) :
  ∃ k : ℕ, k = 6 ∧ (a + (k:ℚ) * b) ≠ (10^15 - 1) :=
sorry

end NUMINAMATH_GPT_smallest_k_l386_38660


namespace NUMINAMATH_GPT_prob_at_least_two_correct_l386_38619

-- Probability of guessing a question correctly
def prob_correct := 1 / 6

-- Probability of guessing a question incorrectly
def prob_incorrect := 5 / 6

-- Binomial probability mass function for k successes out of n trials
def binom_pmf (n k : ℕ) (p : ℚ) : ℚ :=
  (Nat.choose n k : ℚ) * (p ^ k) * ((1 - p) ^ (n - k))

-- Calculate probability P(X = 0)
def prob_X0 := binom_pmf 6 0 prob_correct

-- Calculate probability P(X = 1)
def prob_X1 := binom_pmf 6 1 prob_correct

-- Theorem for the desired probability
theorem prob_at_least_two_correct : 
  1 - (prob_X0 + prob_X1) = 34369 / 58420 := by
  sorry

end NUMINAMATH_GPT_prob_at_least_two_correct_l386_38619


namespace NUMINAMATH_GPT_find_sum_l386_38605

theorem find_sum (P R : ℝ) (T : ℝ) (hT : T = 3) (h1 : P * (R + 1) * 3 = P * R * 3 + 2500) : 
  P = 2500 := by
  sorry

end NUMINAMATH_GPT_find_sum_l386_38605


namespace NUMINAMATH_GPT_one_number_is_zero_l386_38656

variable {a b c : ℤ}
variable (cards : Fin 30 → ℤ)

theorem one_number_is_zero (h_distinct : a ≠ b ∧ a ≠ c ∧ b ≠ c)
    (h_cards : ∀ i : Fin 30, cards i = a ∨ cards i = b ∨ cards i = c)
    (h_sum_zero : ∀ (S : Finset (Fin 30)) (hS : S.card = 5),
        ∃ T : Finset (Fin 30), T.card = 5 ∧ (S ∪ T).sum cards = 0) :
    b = 0 := 
sorry

end NUMINAMATH_GPT_one_number_is_zero_l386_38656


namespace NUMINAMATH_GPT_bales_in_barn_now_l386_38625

-- Define the initial number of bales
def initial_bales : ℕ := 28

-- Define the number of bales added by Tim
def added_bales : ℕ := 26

-- Define the total number of bales
def total_bales : ℕ := initial_bales + added_bales

-- Theorem stating the total number of bales
theorem bales_in_barn_now : total_bales = 54 := by
  sorry

end NUMINAMATH_GPT_bales_in_barn_now_l386_38625


namespace NUMINAMATH_GPT_trig_fraction_identity_l386_38695

noncomputable def cos_63 := Real.cos (Real.pi * 63 / 180)
noncomputable def cos_3 := Real.cos (Real.pi * 3 / 180)
noncomputable def cos_87 := Real.cos (Real.pi * 87 / 180)
noncomputable def cos_27 := Real.cos (Real.pi * 27 / 180)
noncomputable def cos_132 := Real.cos (Real.pi * 132 / 180)
noncomputable def cos_72 := Real.cos (Real.pi * 72 / 180)
noncomputable def cos_42 := Real.cos (Real.pi * 42 / 180)
noncomputable def cos_18 := Real.cos (Real.pi * 18 / 180)
noncomputable def tan_24 := Real.tan (Real.pi * 24 / 180)

theorem trig_fraction_identity :
  (cos_63 * cos_3 - cos_87 * cos_27) / 
  (cos_132 * cos_72 - cos_42 * cos_18) = 
  -tan_24 := 
by
  sorry

end NUMINAMATH_GPT_trig_fraction_identity_l386_38695


namespace NUMINAMATH_GPT_number_of_zeros_of_g_l386_38664

open Real

noncomputable def g (x : ℝ) : ℝ := cos (π * log x + x)

theorem number_of_zeros_of_g : ¬ ∃ (x : ℝ), 1 < x ∧ x < exp 2 ∧ g x = 0 :=
sorry

end NUMINAMATH_GPT_number_of_zeros_of_g_l386_38664


namespace NUMINAMATH_GPT_solve_for_x_l386_38601

theorem solve_for_x (x : ℚ) (h : 2 / 3 + 1 / x = 7 / 9) : x = 9 :=
sorry

end NUMINAMATH_GPT_solve_for_x_l386_38601


namespace NUMINAMATH_GPT_value_of_expression_l386_38693

-- Given conditions as definitions
axiom cond1 (x y : ℝ) : -x + 2*y = 5

-- The theorem we want to prove
theorem value_of_expression (x y : ℝ) (h : -x + 2*y = 5) : 
  5 * (x - 2 * y)^2 - 3 * (x - 2 * y) - 60 = 80 :=
by
  -- The proof part is omitted here.
  sorry

end NUMINAMATH_GPT_value_of_expression_l386_38693


namespace NUMINAMATH_GPT_symmetry_center_of_g_l386_38626

open Real

noncomputable def g (x : ℝ) : ℝ := cos ((1 / 2) * x - π / 6)

def center_of_symmetry : Set (ℝ × ℝ) := { p | ∃ k : ℤ, p = (2 * k * π + 4 * π / 3, 0) }

theorem symmetry_center_of_g :
  (∃ p : ℝ × ℝ, p ∈ center_of_symmetry) :=
sorry

end NUMINAMATH_GPT_symmetry_center_of_g_l386_38626


namespace NUMINAMATH_GPT_number_of_ways_to_form_team_l386_38669

theorem number_of_ways_to_form_team (boys girls : ℕ) (select_boys select_girls : ℕ)
    (H_boys : boys = 7) (H_girls : girls = 9) (H_select_boys : select_boys = 2) (H_select_girls : select_girls = 3) :
    (Nat.choose boys select_boys) * (Nat.choose girls select_girls) = 1764 := by
  rw [H_boys, H_girls, H_select_boys, H_select_girls]
  sorry

end NUMINAMATH_GPT_number_of_ways_to_form_team_l386_38669


namespace NUMINAMATH_GPT_tan_150_eq_neg_inv_sqrt3_l386_38663

theorem tan_150_eq_neg_inv_sqrt3 : Real.tan (150 * Real.pi / 180) = -1 / Real.sqrt 3 :=
by
  sorry

end NUMINAMATH_GPT_tan_150_eq_neg_inv_sqrt3_l386_38663


namespace NUMINAMATH_GPT_number_of_wheels_l386_38624

theorem number_of_wheels (V : ℕ) (W_2 : ℕ) (n : ℕ) 
  (hV : V = 16) 
  (h_eq : 2 * W_2 + 16 * n = 66) : 
  n = 4 := 
by 
  sorry

end NUMINAMATH_GPT_number_of_wheels_l386_38624


namespace NUMINAMATH_GPT_maximize_profit_l386_38632

def revenue (x : ℝ) : ℝ := 17 * x^2
def cost (x : ℝ) : ℝ := 2 * x^3 - x^2
def profit (x : ℝ) : ℝ := revenue x - cost x

theorem maximize_profit : ∃ x > 0, profit x = 18 * x^2 - 2 * x^3 ∧ (∀ y > 0, y ≠ x → profit y < profit x) :=
by
  sorry

end NUMINAMATH_GPT_maximize_profit_l386_38632


namespace NUMINAMATH_GPT_find_richards_score_l386_38681

variable (R B : ℕ)

theorem find_richards_score (h1 : B = R - 14) (h2 : B = 48) : R = 62 := by
  sorry

end NUMINAMATH_GPT_find_richards_score_l386_38681


namespace NUMINAMATH_GPT_probability_no_3by3_red_grid_correct_l386_38684

noncomputable def probability_no_3by3_red_grid : ℚ := 813 / 819

theorem probability_no_3by3_red_grid_correct :
  ∀ (m n : ℕ), Nat.gcd m n = 1 →
  (∀ square : Fin 4 × Fin 4, square.1 = square.2 ∨ square.1 ≠ square.2) →
    m = 813 ∧ n = 819 ∧ probability_no_3by3_red_grid = m / n :=
by
  sorry

end NUMINAMATH_GPT_probability_no_3by3_red_grid_correct_l386_38684


namespace NUMINAMATH_GPT_find_fraction_value_l386_38645

theorem find_fraction_value {m n r t : ℚ}
  (h1 : m / n = 5 / 2)
  (h2 : r / t = 7 / 5) :
  (2 * m * r - 3 * n * t) / (5 * n * t - 4 * m * r) = -4 / 9 :=
by
  sorry

end NUMINAMATH_GPT_find_fraction_value_l386_38645


namespace NUMINAMATH_GPT_sum_of_coefficients_eq_one_l386_38635

theorem sum_of_coefficients_eq_one (a a₁ a₂ a₃ a₄ : ℝ) :
  (∀ x : ℝ, (2 * x - 3) ^ 4 = a + a₁ * x + a₂ * x^2 + a₃ * x^3 + a₄ * x^4) →
  a + a₁ + a₂ + a₃ + a₄ = 1 :=
by
  intros h
  specialize h 1
  -- Specific calculation steps would go here
  sorry

end NUMINAMATH_GPT_sum_of_coefficients_eq_one_l386_38635


namespace NUMINAMATH_GPT_solve_for_x_l386_38654

def custom_mul (a b : ℝ) : ℝ := 4 * a - 2 * b

theorem solve_for_x (x : ℝ) : (custom_mul 3 (custom_mul 6 x) = 2) → (x = 19 / 2) :=
sorry

end NUMINAMATH_GPT_solve_for_x_l386_38654


namespace NUMINAMATH_GPT_actual_distance_map_l386_38622

theorem actual_distance_map (scale : ℕ) (map_distance : ℕ) (actual_distance_km : ℕ) (h1 : scale = 500000) (h2 : map_distance = 4) :
  actual_distance_km = 20 :=
by
  -- definitions and assumptions
  let actual_distance_cm := map_distance * scale
  have cm_to_km_conversion : actual_distance_km = actual_distance_cm / 100000 := sorry
  -- calculation
  have actual_distance_sol : actual_distance_cm = 4 * 500000 := sorry
  have actual_distance_eq : actual_distance_km = (4 * 500000) / 100000 := sorry
  -- final answer
  have answer_correct : actual_distance_km = 20 := sorry
  exact answer_correct

end NUMINAMATH_GPT_actual_distance_map_l386_38622


namespace NUMINAMATH_GPT_average_of_first_21_multiples_of_7_l386_38659

theorem average_of_first_21_multiples_of_7 :
  let a1 := 7
  let d := 7
  let n := 21
  let an := a1 + (n - 1) * d
  let Sn := n / 2 * (a1 + an)
  Sn / n = 77 :=
by
  let a1 := 7
  let d := 7
  let n := 21
  let an := a1 + (n - 1) * d
  let Sn := n / 2 * (a1 + an)
  have h1 : an = 147 := by
    sorry
  have h2 : Sn = 1617 := by
    sorry
  have h3 : Sn / n = 77 := by
    sorry
  exact h3

end NUMINAMATH_GPT_average_of_first_21_multiples_of_7_l386_38659


namespace NUMINAMATH_GPT_find_x_l386_38602

def operation_star (a b c d : ℤ) : ℤ × ℤ :=
  (a + c, b - 2 * d)

theorem find_x (x y : ℤ) (h : operation_star (x+1) (y-1) 1 3 = (2, -4)) : x = 0 :=
by 
  sorry

end NUMINAMATH_GPT_find_x_l386_38602


namespace NUMINAMATH_GPT_find_alpha_plus_beta_l386_38618

open Real

theorem find_alpha_plus_beta 
  (α β : ℝ)
  (h1 : sin α = sqrt 5 / 5)
  (h2 : sin β = sqrt 10 / 10)
  (h3 : π / 2 < α ∧ α < π)
  (h4 : π / 2 < β ∧ β < π) :
  α + β = 7 * π / 4 :=
sorry

end NUMINAMATH_GPT_find_alpha_plus_beta_l386_38618


namespace NUMINAMATH_GPT_cannot_determine_right_triangle_l386_38615

-- Define what a right triangle is
def is_right_triangle (a b c : ℕ) : Prop :=
  a^2 + b^2 = c^2

-- Define the conditions
def condition_A (A B C : ℕ) : Prop :=
  A / B = 3 / 4 ∧ A / C = 3 / 5 ∧ B / C = 4 / 5

def condition_B (a b c : ℕ) : Prop :=
  a = 5 ∧ b = 12 ∧ c = 13

def condition_C (A B C : ℕ) : Prop :=
  A - B = C

def condition_D (a b c : ℕ) : Prop :=
  a^2 = b^2 - c^2

-- Define the problem in Lean
theorem cannot_determine_right_triangle :
  (∃ A B C, condition_A A B C → ¬is_right_triangle A B C) ∧
  (∀ (a b c : ℕ), condition_B a b c → is_right_triangle a b c) ∧
  (∀ A B C, condition_C A B C → A = 90) ∧
  (∀ (a b c : ℕ),  condition_D a b c → is_right_triangle a b c)
:=
by sorry

end NUMINAMATH_GPT_cannot_determine_right_triangle_l386_38615


namespace NUMINAMATH_GPT_smallest_integer_solution_of_inequality_l386_38655

theorem smallest_integer_solution_of_inequality : ∃ x : ℤ, (3 * x ≥ x - 5) ∧ (∀ y : ℤ, 3 * y ≥ y - 5 → y ≥ -2) := 
sorry

end NUMINAMATH_GPT_smallest_integer_solution_of_inequality_l386_38655
