import Mathlib

namespace NUMINAMATH_GPT_arithmetic_sequence_sum_l1971_197180

theorem arithmetic_sequence_sum (x y : ℕ) (h₀: ∃ (n : ℕ), x = 3 + n * 4) (h₁: ∃ (m : ℕ), y = 3 + m * 4) (h₂: y = 31 - 4) (h₃: x = y - 4) : x + y = 50 := by
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_sum_l1971_197180


namespace NUMINAMATH_GPT_probability_factor_less_than_eight_l1971_197106

theorem probability_factor_less_than_eight (n : ℕ) (h72 : n = 72) :
  (∃ k < 8, k ∣ n) →
  (∃ p q, p/q = 5/12) :=
by
  sorry

end NUMINAMATH_GPT_probability_factor_less_than_eight_l1971_197106


namespace NUMINAMATH_GPT_courtyard_length_eq_40_l1971_197147

/-- Defining the dimensions of a paving stone -/
def stone_length : ℝ := 4
def stone_width : ℝ := 2

/-- Defining the width of the courtyard -/
def courtyard_width : ℝ := 20

/-- Number of paving stones used -/
def num_stones : ℝ := 100

/-- Area covered by one paving stone -/
def stone_area : ℝ := stone_length * stone_width

/-- Total area covered by the paving stones -/
def total_area : ℝ := num_stones * stone_area

/-- The main statement to be proved -/
theorem courtyard_length_eq_40 (h1 : total_area = num_stones * stone_area)
(h2 : total_area = 800)
(h3 : courtyard_width = 20) : total_area / courtyard_width = 40 :=
by sorry

end NUMINAMATH_GPT_courtyard_length_eq_40_l1971_197147


namespace NUMINAMATH_GPT_find_other_denomination_l1971_197129

theorem find_other_denomination
  (total_spent : ℕ)
  (twenty_bill_value : ℕ) (other_denomination_value : ℕ)
  (twenty_bill_count : ℕ) (other_bill_count : ℕ)
  (h1 : total_spent = 80)
  (h2 : twenty_bill_value = 20)
  (h3 : other_bill_count = 2)
  (h4 : twenty_bill_count = other_bill_count + 1)
  (h5 : total_spent = twenty_bill_value * twenty_bill_count + other_denomination_value * other_bill_count) : 
  other_denomination_value = 10 :=
by
  sorry

end NUMINAMATH_GPT_find_other_denomination_l1971_197129


namespace NUMINAMATH_GPT_inequality_solution_set_l1971_197137

theorem inequality_solution_set :
  { x : ℝ | 1 < x ∧ x < 2 } = { x : ℝ | (x - 2) / (1 - x) > 0 } :=
by sorry

end NUMINAMATH_GPT_inequality_solution_set_l1971_197137


namespace NUMINAMATH_GPT_match_proverbs_l1971_197190

-- Define each condition as a Lean definition
def condition1 : Prop :=
"As cold comes and heat goes, the four seasons change" = "Things are developing"

def condition2 : Prop :=
"Thousands of flowers arranged, just waiting for the first thunder" = 
"Decisively seize the opportunity to promote qualitative change"

def condition3 : Prop :=
"Despite the intention to plant flowers, they don't bloom; unintentionally planting willows, they grow into shade" = 
"The unity of contradictions"

def condition4 : Prop :=
"There will be times when the strong winds break the waves, and we will sail across the sea with clouds" = 
"The future is bright"

-- The theorem we need to prove, using the condition definitions
theorem match_proverbs : condition2 ∧ condition4 :=
sorry

end NUMINAMATH_GPT_match_proverbs_l1971_197190


namespace NUMINAMATH_GPT_sqrt_expr_is_integer_l1971_197187

theorem sqrt_expr_is_integer (x : ℤ) (n : ℤ) (h : n^2 = x^2 - x + 1) : x = 0 ∨ x = 1 := by
  sorry

end NUMINAMATH_GPT_sqrt_expr_is_integer_l1971_197187


namespace NUMINAMATH_GPT_revenue_per_investment_l1971_197161

theorem revenue_per_investment (Banks_investments : ℕ) (Elizabeth_investments : ℕ) (Elizabeth_revenue_per_investment : ℕ) (revenue_difference : ℕ) :
  Banks_investments = 8 →
  Elizabeth_investments = 5 →
  Elizabeth_revenue_per_investment = 900 →
  revenue_difference = 500 →
  ∃ (R : ℤ), R = (5 * 900 - 500) / 8 :=
by
  intros h1 h2 h3 h4
  let T_elizabeth := 5 * Elizabeth_revenue_per_investment
  let T_banks := T_elizabeth - revenue_difference
  let R := T_banks / 8
  use R
  sorry

end NUMINAMATH_GPT_revenue_per_investment_l1971_197161


namespace NUMINAMATH_GPT_problem_proof_l1971_197173

theorem problem_proof (x y : ℝ) (h : x / (2 * y) = 3 / 2) : (7 * x + 2 * y) / (x - 2 * y) = 23 :=
by sorry

end NUMINAMATH_GPT_problem_proof_l1971_197173


namespace NUMINAMATH_GPT_man_speed_with_stream_l1971_197125

-- Define the man's rate in still water
def man_rate_in_still_water : ℝ := 6

-- Define the man's rate against the stream
def man_rate_against_stream (stream_speed : ℝ) : ℝ :=
  man_rate_in_still_water - stream_speed

-- The given condition that the man's rate against the stream is 10 km/h
def man_rate_against_condition : Prop := ∃ (stream_speed : ℝ), man_rate_against_stream stream_speed = 10

-- We aim to prove that the man's speed with the stream is 10 km/h
theorem man_speed_with_stream (stream_speed : ℝ) (h : man_rate_against_stream stream_speed = 10) :
  man_rate_in_still_water + stream_speed = 10 := by
  sorry

end NUMINAMATH_GPT_man_speed_with_stream_l1971_197125


namespace NUMINAMATH_GPT_locus_of_points_equidistant_from_axes_l1971_197184

-- Define the notion of being equidistant from the x-axis and the y-axis
def is_equidistant_from_axes (P : (ℝ × ℝ)) : Prop :=
  abs P.1 = abs P.2

-- The proof problem: given a moving point, the locus equation when P is equidistant from both axes
theorem locus_of_points_equidistant_from_axes (x y : ℝ) :
  is_equidistant_from_axes (x, y) → abs x - abs y = 0 :=
by
  intros h
  exact sorry

end NUMINAMATH_GPT_locus_of_points_equidistant_from_axes_l1971_197184


namespace NUMINAMATH_GPT_find_Y_l1971_197143

theorem find_Y (Y : ℕ) 
  (h_top : 2 + 1 + Y + 3 = 6 + Y)
  (h_bottom : 4 + 3 + 1 + 5 = 13)
  (h_equal : 6 + Y = 13) : 
  Y = 7 := 
by
  sorry

end NUMINAMATH_GPT_find_Y_l1971_197143


namespace NUMINAMATH_GPT_units_digit_of_n_squared_plus_2_n_is_7_l1971_197124

def n : ℕ := 2023 ^ 2 + 2 ^ 2023

theorem units_digit_of_n_squared_plus_2_n_is_7 : (n ^ 2 + 2 ^ n) % 10 = 7 := 
by
  sorry

end NUMINAMATH_GPT_units_digit_of_n_squared_plus_2_n_is_7_l1971_197124


namespace NUMINAMATH_GPT_systematic_sampling_method_l1971_197122

theorem systematic_sampling_method :
  ∀ (num_classes num_students_per_class selected_student : ℕ),
    num_classes = 12 →
    num_students_per_class = 50 →
    selected_student = 40 →
    (∃ (start_interval: ℕ) (interval: ℕ) (total_population: ℕ), 
      total_population > 100 ∧ start_interval < interval ∧ interval * num_classes = total_population ∧
      ∀ (c : ℕ), c < num_classes → (start_interval + c * interval) % num_students_per_class = selected_student - 1) →
    "Systematic Sampling" = "Systematic Sampling" :=
by
  intros num_classes num_students_per_class selected_student h_classes h_students h_selected h_conditions
  sorry

end NUMINAMATH_GPT_systematic_sampling_method_l1971_197122


namespace NUMINAMATH_GPT_tan_product_l1971_197172

theorem tan_product : 
(1 + Real.tan (Real.pi / 60)) * (1 + Real.tan (Real.pi / 30)) * (1 + Real.tan (Real.pi / 20)) * (1 + Real.tan (Real.pi / 15)) * (1 + Real.tan (Real.pi / 12)) * (1 + Real.tan (Real.pi / 10)) * (1 + Real.tan (Real.pi / 9)) * (1 + Real.tan (Real.pi / 6)) = 2^8 :=
by
  sorry 

end NUMINAMATH_GPT_tan_product_l1971_197172


namespace NUMINAMATH_GPT_painted_rooms_l1971_197136

def total_rooms : ℕ := 12
def hours_per_room : ℕ := 7
def remaining_hours : ℕ := 49

theorem painted_rooms : total_rooms - (remaining_hours / hours_per_room) = 5 := by
  sorry

end NUMINAMATH_GPT_painted_rooms_l1971_197136


namespace NUMINAMATH_GPT_find_natural_numbers_eq_36_sum_of_digits_l1971_197177

-- Define the sum of digits function
def sum_of_digits (x : ℕ) : ℕ := 
  if x = 0 then 0
  else sum_of_digits (x / 10) + (x % 10)

-- Lean theorem statement proving the given problem
theorem find_natural_numbers_eq_36_sum_of_digits :
  {x : ℕ | x = 36 * (sum_of_digits x)} = {324, 648} :=
sorry

end NUMINAMATH_GPT_find_natural_numbers_eq_36_sum_of_digits_l1971_197177


namespace NUMINAMATH_GPT_range_of_a_l1971_197126
noncomputable section

open Real

theorem range_of_a (a : ℝ) (h : ∀ x : ℝ, x^2 + 2 * x + a + 2 > 0) : a > -1 :=
sorry

end NUMINAMATH_GPT_range_of_a_l1971_197126


namespace NUMINAMATH_GPT_strongest_erosive_power_l1971_197128

-- Definition of the options
inductive Period where
  | MayToJune : Period
  | JuneToJuly : Period
  | JulyToAugust : Period
  | AugustToSeptember : Period

-- Definition of the eroding power function (stub)
def erosivePower : Period → ℕ
| Period.MayToJune => 1
| Period.JuneToJuly => 2
| Period.JulyToAugust => 3
| Period.AugustToSeptember => 1

-- Statement that July to August has the maximum erosive power
theorem strongest_erosive_power : erosivePower Period.JulyToAugust = 3 := 
by 
  sorry

end NUMINAMATH_GPT_strongest_erosive_power_l1971_197128


namespace NUMINAMATH_GPT_smallest_x_value_l1971_197148

theorem smallest_x_value : ∀ x : ℚ, (14 * x^2 - 40 * x + 18) / (4 * x - 3) + 6 * x = 7 * x - 2 → x = 4 / 5 :=
by
  intros x hx
  sorry

end NUMINAMATH_GPT_smallest_x_value_l1971_197148


namespace NUMINAMATH_GPT_scientific_notation_l1971_197192

theorem scientific_notation : (0.000000005 : ℝ) = 5 * 10^(-9 : ℤ) := 
by
  sorry

end NUMINAMATH_GPT_scientific_notation_l1971_197192


namespace NUMINAMATH_GPT_no_five_consecutive_divisible_by_2025_l1971_197113

def seq (n : ℕ) : ℕ := 1 + 2^n + 3^n + 4^n + 5^n

theorem no_five_consecutive_divisible_by_2025 : 
  ¬ ∃ (a : ℕ), (∀ (i : ℕ), i < 5 → 2025 ∣ seq (a + i)) := 
sorry

end NUMINAMATH_GPT_no_five_consecutive_divisible_by_2025_l1971_197113


namespace NUMINAMATH_GPT_quilt_squares_count_l1971_197149

theorem quilt_squares_count (total_squares : ℕ) (additional_squares : ℕ)
  (h1 : total_squares = 4 * additional_squares)
  (h2 : additional_squares = 24) :
  total_squares = 32 :=
by
  -- Proof would go here
  -- The proof would involve showing that total_squares indeed equals 32 given h1 and h2
  sorry

end NUMINAMATH_GPT_quilt_squares_count_l1971_197149


namespace NUMINAMATH_GPT_circle_center_radius_l1971_197160

theorem circle_center_radius :
  ∃ (h k r : ℝ), (∀ x y : ℝ, (x - 2)^2 + (y + 1)^2 = 4 ↔ (x - h)^2 + (y - k)^2 = r^2) ∧ h = 2 ∧ k = -1 ∧ r = 2 :=
by
  sorry

end NUMINAMATH_GPT_circle_center_radius_l1971_197160


namespace NUMINAMATH_GPT_find_costs_of_accessories_max_type_a_accessories_l1971_197191

theorem find_costs_of_accessories (x y : ℕ) 
  (h1 : x + 3 * y = 530) 
  (h2 : 3 * x + 2 * y = 890) : 
  x = 230 ∧ y = 100 := 
by 
  sorry

theorem max_type_a_accessories (m n : ℕ) 
  (m_n_sum : m + n = 30) 
  (cost_constraint : 230 * m + 100 * n ≤ 4180) : 
  m ≤ 9 := 
by 
  sorry

end NUMINAMATH_GPT_find_costs_of_accessories_max_type_a_accessories_l1971_197191


namespace NUMINAMATH_GPT_max_possible_value_xv_l1971_197127

noncomputable def max_xv_distance (x y z w v : ℝ)
  (h1 : |x - y| = 1)
  (h2 : |y - z| = 2)
  (h3 : |z - w| = 3)
  (h4 : |w - v| = 5) : ℝ :=
|x - v|

theorem max_possible_value_xv 
  (x y z w v : ℝ)
  (h1 : |x - y| = 1)
  (h2 : |y - z| = 2)
  (h3 : |z - w| = 3)
  (h4 : |w - v| = 5) :
  max_xv_distance x y z w v h1 h2 h3 h4 = 11 :=
sorry

end NUMINAMATH_GPT_max_possible_value_xv_l1971_197127


namespace NUMINAMATH_GPT_find_positive_integers_l1971_197114

noncomputable def is_power_of_two (n : ℕ) : Prop :=
  ∃ k : ℕ, n = 2 ^ k

theorem find_positive_integers (a b c : ℕ) (hpos_a : 0 < a) (hpos_b : 0 < b) (hpos_c : 0 < c)
  (hab_c : is_power_of_two (a * b - c))
  (hbc_a : is_power_of_two (b * c - a))
  (hca_b : is_power_of_two (c * a - b)) :
  (a = 2 ∧ b = 2 ∧ c = 2) ∨
  (a = 2 ∧ b = 2 ∧ c = 3) ∨
  (a = 3 ∧ b = 5 ∧ c = 7) ∨
  (a = 2 ∧ b = 6 ∧ c = 11) :=
sorry

end NUMINAMATH_GPT_find_positive_integers_l1971_197114


namespace NUMINAMATH_GPT_math_proof_problem_l1971_197103

theorem math_proof_problem
  (a b c : ℝ)
  (h : a ≠ b)
  (h1 : b ≠ c)
  (h2 : c ≠ a)
  (h3 : (a / (2 * (b - c))) + (b / (2 * (c - a))) + (c / (2 * (a - b))) = 0) :
  (a / (b - c)^3) + (b / (c - a)^3) + (c / (a - b)^3) = 0 := 
by
  sorry

end NUMINAMATH_GPT_math_proof_problem_l1971_197103


namespace NUMINAMATH_GPT_min_value_of_ab_l1971_197133

theorem min_value_of_ab {a b : ℝ} (ha : a > 0) (hb : b > 0)
    (h : 1 / a + 1 / b = 1) : a + b ≥ 4 :=
sorry

end NUMINAMATH_GPT_min_value_of_ab_l1971_197133


namespace NUMINAMATH_GPT_unique_x2_range_of_a_l1971_197151

noncomputable def f (x : ℝ) (k a : ℝ) : ℝ :=
if x >= 0
then k*x + k*(1 - a^2)
else x^2 + (a^2 - 4*a)*x + (3 - a)^2

theorem unique_x2 (k a : ℝ) (x1 : ℝ) (hx1 : x1 ≠ 0) (hx2 : ∃ x2 : ℝ, x2 ≠ 0 ∧ x2 ≠ x1 ∧ f x2 k a = f x1 k a) :
f 0 k a = k*(1 - a^2) →
0 ≤ a ∧ a < 1 →
k = (3 - a)^2 / (1 - a^2) :=
sorry

variable (a : ℝ)

theorem range_of_a :
0 ≤ a ∧ a < 1 ↔ a^2 - 4*a ≤ 0 :=
sorry

end NUMINAMATH_GPT_unique_x2_range_of_a_l1971_197151


namespace NUMINAMATH_GPT_barbell_percentage_increase_l1971_197171

def old_barbell_cost : ℕ := 250
def new_barbell_cost : ℕ := 325

theorem barbell_percentage_increase :
  (new_barbell_cost - old_barbell_cost : ℚ) / old_barbell_cost * 100 = 30 := 
by
  sorry

end NUMINAMATH_GPT_barbell_percentage_increase_l1971_197171


namespace NUMINAMATH_GPT_chili_pepper_cost_l1971_197182

theorem chili_pepper_cost :
  ∃ x : ℝ, 
    (3 * 2.50 + 4 * 1.50 + 5 * x = 18) ∧ 
    x = 0.90 :=
by
  use 0.90
  sorry

end NUMINAMATH_GPT_chili_pepper_cost_l1971_197182


namespace NUMINAMATH_GPT_sin_cos_105_l1971_197175

theorem sin_cos_105 (h1 : ∀ x : ℝ, Real.sin x * Real.cos x = 1 / 2 * Real.sin (2 * x))
                    (h2 : ∀ x : ℝ, Real.sin (180 * Real.pi / 180 + x) = - Real.sin x)
                    (h3 : Real.sin (30 * Real.pi / 180) = 1 / 2) :
  Real.sin (105 * Real.pi / 180) * Real.cos (105 * Real.pi / 180) = - 1 / 4 :=
by
  sorry

end NUMINAMATH_GPT_sin_cos_105_l1971_197175


namespace NUMINAMATH_GPT_problem1_problem2_l1971_197157

-- Problem 1
theorem problem1 {a b c : ℝ} (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (1 / a) + (1 / b) + (1 / c) ≥ (1 / (Real.sqrt (a * b))) + (1 / (Real.sqrt (b * c))) + (1 / (Real.sqrt (a * c))) :=
sorry

-- Problem 2
theorem problem2 {x y : ℝ} :
  Real.sin x + Real.sin y ≤ 1 + Real.sin x * Real.sin y :=
sorry

end NUMINAMATH_GPT_problem1_problem2_l1971_197157


namespace NUMINAMATH_GPT_total_cookies_l1971_197158

def MonaCookies : ℕ := 20
def JasmineCookies : ℕ := MonaCookies - 5
def RachelCookies : ℕ := JasmineCookies + 10

theorem total_cookies : MonaCookies + JasmineCookies + RachelCookies = 60 := by
  -- Since we don't need to provide the solution steps, we simply use sorry.
  sorry

end NUMINAMATH_GPT_total_cookies_l1971_197158


namespace NUMINAMATH_GPT_sqrt_five_eq_l1971_197153

theorem sqrt_five_eq (m n a b c d : ℤ)
  (h : m + n * Real.sqrt 5 = (a + b * Real.sqrt 5) * (c + d * Real.sqrt 5)) :
  m - n * Real.sqrt 5 = (a - b * Real.sqrt 5) * (c - d * Real.sqrt 5) := by
  sorry

end NUMINAMATH_GPT_sqrt_five_eq_l1971_197153


namespace NUMINAMATH_GPT_tangent_line_eq_range_f_l1971_197193

-- Given the function f(x) = 2x^3 - 9x^2 + 12x
def f(x : ℝ) : ℝ := 2 * x^3 - 9 * x^2 + 12 * x

-- (1) Prove that the equation of the tangent line to y = f(x) at (0, f(0)) is y = 12x
theorem tangent_line_eq : ∀ x, x = 0 → f x = 0 → (∃ m, m = 12 ∧ (∀ y, y = 12 * x)) :=
by
  sorry

-- (2) Prove that the range of f(x) on the interval [0, 3] is [0, 9]
theorem range_f : Set.Icc 0 9 = Set.image f (Set.Icc (0 : ℝ) 3) :=
by
  sorry

end NUMINAMATH_GPT_tangent_line_eq_range_f_l1971_197193


namespace NUMINAMATH_GPT_find_m_value_l1971_197152

theorem find_m_value (a m : ℤ) (h : a ≠ 1) (hx : ∀ x y : ℤ, (x > 0) ∧ (y > 0) ∧ (a - 1) * x^2 - m * x + a = 0 ∧ (a - 1) * y^2 - m * y + a = 0) : m = 3 :=
sorry

end NUMINAMATH_GPT_find_m_value_l1971_197152


namespace NUMINAMATH_GPT_option_d_not_true_l1971_197194

variable (a b : ℝ)
variable (h1 : 0 < a)
variable (h2 : 0 < b)

theorem option_d_not_true : (2 * a * b) / (a + b) ≤ Real.sqrt (a * b) := sorry

end NUMINAMATH_GPT_option_d_not_true_l1971_197194


namespace NUMINAMATH_GPT_marbles_count_l1971_197189

def num_violet_marbles := 64

def num_red_marbles := 14

def total_marbles (violet : Nat) (red : Nat) : Nat :=
  violet + red

theorem marbles_count :
  total_marbles num_violet_marbles num_red_marbles = 78 := by
  sorry

end NUMINAMATH_GPT_marbles_count_l1971_197189


namespace NUMINAMATH_GPT_intersection_complement_l1971_197174

-- Declare variables for sets
variable (I A B : Set ℤ)

-- Define the universal set I
def universal_set : Set ℤ := { x | -3 < x ∧ x < 3 }

-- Define sets A and B
def set_A : Set ℤ := { -2, 0, 1 }
def set_B : Set ℤ := { -1, 0, 1, 2 }

-- Main theorem statement
theorem intersection_complement
  (hI : I = universal_set)
  (hA : A = set_A)
  (hB : B = set_B) :
  B ∩ (I \ A) = { -1, 2 } :=
sorry

end NUMINAMATH_GPT_intersection_complement_l1971_197174


namespace NUMINAMATH_GPT_perp_bisector_eq_l1971_197178

/-- The circles x^2+y^2=4 and x^2+y^2-4x+6y=0 intersect at points A and B. 
Find the equation of the perpendicular bisector of line segment AB. -/

theorem perp_bisector_eq : 
  let C1 := (0, 0)
  let C2 := (2, -3)
  ∃ (a b c : ℝ), a = 3 ∧ b = 2 ∧ c = 0 ∧ ∀ (x y : ℝ), a * x + b * y + c = 0 := 
by
  sorry

end NUMINAMATH_GPT_perp_bisector_eq_l1971_197178


namespace NUMINAMATH_GPT_seeds_per_flowerbed_l1971_197162

theorem seeds_per_flowerbed (total_seeds flowerbeds : ℕ) (h1 : total_seeds = 32) (h2 : flowerbeds = 8) :
  total_seeds / flowerbeds = 4 :=
by {
  sorry
}

end NUMINAMATH_GPT_seeds_per_flowerbed_l1971_197162


namespace NUMINAMATH_GPT_no_real_solution_of_fraction_eq_l1971_197108

theorem no_real_solution_of_fraction_eq (m : ℝ) :
  (∀ x : ℝ, (x - 1) / (x + 4) ≠ m / (x + 4)) → m = -5 :=
sorry

end NUMINAMATH_GPT_no_real_solution_of_fraction_eq_l1971_197108


namespace NUMINAMATH_GPT_vector_perpendicular_l1971_197156

open Real

theorem vector_perpendicular (t : ℝ) (a b : ℝ × ℝ) (h_a : a = (1, 2)) (h_b : b = (4, 3)) :
  a.1 * (t * a.1 + b.1) + a.2 * (t * a.2 + b.2) = 0 ↔ t = -2 := by
  sorry

end NUMINAMATH_GPT_vector_perpendicular_l1971_197156


namespace NUMINAMATH_GPT_union_P_Q_l1971_197197

-- Definition of sets P and Q
def P : Set ℝ := { x | 0 ≤ x ∧ x ≤ 4 }
def Q : Set ℝ := { x | -3 < x ∧ x < 3 }

-- Statement to prove
theorem union_P_Q :
  P ∪ Q = { x : ℝ | -3 < x ∧ x ≤ 4 } :=
sorry

end NUMINAMATH_GPT_union_P_Q_l1971_197197


namespace NUMINAMATH_GPT_share_money_3_people_l1971_197181

theorem share_money_3_people (total_money : ℝ) (amount_per_person : ℝ) (h1 : total_money = 3.75) (h2 : amount_per_person = 1.25) : 
  total_money / amount_per_person = 3 := by
  sorry

end NUMINAMATH_GPT_share_money_3_people_l1971_197181


namespace NUMINAMATH_GPT_sum_a_b_max_power_l1971_197116

theorem sum_a_b_max_power (a b : ℕ) (h_pos : 0 < a) (h_b_gt_1 : 1 < b) (h_lt_600 : a ^ b < 600) : a + b = 26 :=
sorry

end NUMINAMATH_GPT_sum_a_b_max_power_l1971_197116


namespace NUMINAMATH_GPT_train_passenger_count_l1971_197104

theorem train_passenger_count (P : ℕ) (total_passengers : ℕ) (r : ℕ)
  (h1 : r = 60)
  (h2 : total_passengers = P + r + 3 * (P + r))
  (h3 : total_passengers = 640) :
  P = 100 :=
by
  sorry

end NUMINAMATH_GPT_train_passenger_count_l1971_197104


namespace NUMINAMATH_GPT_final_silver_tokens_l1971_197100

structure TokenCounts :=
  (red : ℕ)
  (blue : ℕ)

def initial_tokens : TokenCounts := { red := 100, blue := 50 }

def exchange_booth1 (tokens : TokenCounts) : TokenCounts :=
  { red := tokens.red - 3, blue := tokens.blue + 2 }

def exchange_booth2 (tokens : TokenCounts) : TokenCounts :=
  { red := tokens.red + 1, blue := tokens.blue - 3 }

noncomputable def max_exchanges (initial : TokenCounts) : ℕ × ℕ :=
  let x := 48
  let y := 47
  (x, y)

noncomputable def silver_tokens (x y : ℕ) : ℕ := x + y

theorem final_silver_tokens (x y : ℕ) (tokens : TokenCounts) 
  (hx : tokens.red = initial_tokens.red - 3 * x + y)
  (hy : tokens.blue = initial_tokens.blue + 2 * x - 3 * y) 
  (hx_le : tokens.red >= 3 → false)
  (hy_le : tokens.blue >= 3 → false) : 
  silver_tokens x y = 95 :=
by {
  sorry
}

end NUMINAMATH_GPT_final_silver_tokens_l1971_197100


namespace NUMINAMATH_GPT_determine_Q_l1971_197112

def P : Set ℕ := {1, 2}

def Q : Set ℕ := {z | ∃ x y, x ∈ P ∧ y ∈ P ∧ z = x + y}

theorem determine_Q : Q = {2, 3, 4} :=
by
  sorry

end NUMINAMATH_GPT_determine_Q_l1971_197112


namespace NUMINAMATH_GPT_lcm_18_30_eq_90_l1971_197169

theorem lcm_18_30_eq_90 : Nat.lcm 18 30 = 90 := 
by {
  sorry
}

end NUMINAMATH_GPT_lcm_18_30_eq_90_l1971_197169


namespace NUMINAMATH_GPT_expand_product_l1971_197101

theorem expand_product (x : ℝ) : 4 * (x + 3) * (x + 6) = 4 * x^2 + 36 * x + 72 :=
by
  sorry

end NUMINAMATH_GPT_expand_product_l1971_197101


namespace NUMINAMATH_GPT_unique_positive_b_solution_exists_l1971_197155

theorem unique_positive_b_solution_exists (c : ℝ) (k : ℝ) :
  (∃b : ℝ, b > 0 ∧ ∀x : ℝ, x^2 + (b + 1/b) * x + c = 0 → x = 0) ∧
  (∀b : ℝ, b^4 + (2 - 4 * c) * b^2 + k = 0) → c = 1 :=
by
  sorry

end NUMINAMATH_GPT_unique_positive_b_solution_exists_l1971_197155


namespace NUMINAMATH_GPT_infinite_sqrt_solution_l1971_197186

noncomputable def infinite_sqrt (x : ℝ) : ℝ := Real.sqrt (20 + x)

theorem infinite_sqrt_solution : 
  ∃ x : ℝ, infinite_sqrt x = x ∧ x ≥ 0 ∧ x = 5 :=
by
  sorry

end NUMINAMATH_GPT_infinite_sqrt_solution_l1971_197186


namespace NUMINAMATH_GPT_time_fraction_reduced_l1971_197168

theorem time_fraction_reduced (T D : ℝ) (h1 : D = 30 * T) :
  D = 40 * ((3/4) * T) → 1 - (3/4) = 1/4 :=
sorry

end NUMINAMATH_GPT_time_fraction_reduced_l1971_197168


namespace NUMINAMATH_GPT_empty_solution_set_range_l1971_197141

theorem empty_solution_set_range (m : ℝ) : 
  (¬ ∃ x : ℝ, (m * x^2 + 2 * m * x + 1) < 0) ↔ (m = 0 ∨ (0 < m ∧ m ≤ 1)) :=
by sorry

end NUMINAMATH_GPT_empty_solution_set_range_l1971_197141


namespace NUMINAMATH_GPT_g_of_neg2_l1971_197105

def g (x : ℤ) : ℤ := x^3 - x^2 + x

theorem g_of_neg2 : g (-2) = -14 := 
by
  sorry

end NUMINAMATH_GPT_g_of_neg2_l1971_197105


namespace NUMINAMATH_GPT_subset_M_P_N_l1971_197111

def setM : Set (ℝ × ℝ) := {p | |p.1| + |p.2| < 1}

def setN : Set (ℝ × ℝ) := 
  {p | (Real.sqrt ((p.1 - 1 / 2) ^ 2 + (p.2 + 1 / 2) ^ 2) + Real.sqrt ((p.1 + 1 / 2) ^ 2 + (p.2 - 1 / 2) ^ 2)) < 2 * Real.sqrt 2}

def setP : Set (ℝ × ℝ) := 
  {p | |p.1 + p.2| < 1 ∧ |p.1| < 1 ∧ |p.2| < 1}

theorem subset_M_P_N : setM ⊆ setP ∧ setP ⊆ setN := by
  sorry

end NUMINAMATH_GPT_subset_M_P_N_l1971_197111


namespace NUMINAMATH_GPT_largest_k_exists_l1971_197176

noncomputable def largest_k := 3

theorem largest_k_exists :
  ∃ (k : ℕ), (k = largest_k) ∧ ∀ m : ℕ, 
    (∀ n : ℕ, ∃ a b : ℕ, m + n = a^2 + b^2) ∧ 
    (∀ n : ℕ, ∃ seq : ℕ → ℕ,
      (∀ i : ℕ, seq i = a^2 + b^2) ∧
      (∀ j : ℕ, m ≤ j → a^2 + b^2 ≠ 3 + 4 * j)
    ) := ⟨3, rfl, sorry⟩

end NUMINAMATH_GPT_largest_k_exists_l1971_197176


namespace NUMINAMATH_GPT_arithmetic_sequence_num_terms_l1971_197139

theorem arithmetic_sequence_num_terms (a_1 d S_n n : ℕ) 
  (h1 : a_1 = 4) (h2 : d = 3) (h3 : S_n = 650)
  (h4 : S_n = (n / 2) * (2 * a_1 + (n - 1) * d)) : n = 20 := by
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_num_terms_l1971_197139


namespace NUMINAMATH_GPT_ratio_of_triangle_areas_l1971_197166

-- Define the given conditions
variables (m n x a : ℝ) (S T1 T2 : ℝ)

-- Conditions
def area_of_square : Prop := S = x^2
def area_of_triangle_1 : Prop := T1 = m * x^2
def length_relation : Prop := x = n * a

-- The proof goal
theorem ratio_of_triangle_areas (h1 : area_of_square S x) 
                                (h2 : area_of_triangle_1 T1 m x)
                                (h3 : length_relation x n a) : 
                                T2 / S = m / n^2 := 
sorry

end NUMINAMATH_GPT_ratio_of_triangle_areas_l1971_197166


namespace NUMINAMATH_GPT_find_extrema_of_S_l1971_197120

theorem find_extrema_of_S (x y z : ℚ) (h1 : 3 * x + 2 * y + z = 5) (h2 : x + y - z = 2) (hx : 0 ≤ x) (hy : 0 ≤ y) (hz : 0 ≤ z) :
  2 ≤ 2 * x + y - z ∧ 2 * x + y - z ≤ 3 :=
by
  sorry

end NUMINAMATH_GPT_find_extrema_of_S_l1971_197120


namespace NUMINAMATH_GPT_strawberry_candies_count_l1971_197135

theorem strawberry_candies_count (S G : ℕ) (h1 : S + G = 240) (h2 : G = S - 2) : S = 121 :=
by
  sorry

end NUMINAMATH_GPT_strawberry_candies_count_l1971_197135


namespace NUMINAMATH_GPT_marie_erasers_l1971_197142

theorem marie_erasers (initial_erasers : ℕ) (lost_erasers : ℕ) (final_erasers : ℕ) 
  (h1 : initial_erasers = 95) (h2 : lost_erasers = 42) : final_erasers = 53 :=
by
  sorry

end NUMINAMATH_GPT_marie_erasers_l1971_197142


namespace NUMINAMATH_GPT_central_angle_of_sector_l1971_197140

theorem central_angle_of_sector (r S α : ℝ) (h1 : r = 10) (h2 : S = 100)
  (h3 : S = 1/2 * α * r^2) : α = 2 :=
by
  -- Given radius r and area S, substituting into the formula for the area of the sector,
  -- we derive the central angle α.
  sorry

end NUMINAMATH_GPT_central_angle_of_sector_l1971_197140


namespace NUMINAMATH_GPT_smallest_repeating_block_of_5_over_13_l1971_197198

theorem smallest_repeating_block_of_5_over_13 : 
  ∃ n, n = 6 ∧ (∃ m, (5 / 13 : ℚ) = (m/(10^6) : ℚ) ) := 
sorry

end NUMINAMATH_GPT_smallest_repeating_block_of_5_over_13_l1971_197198


namespace NUMINAMATH_GPT_nine_pow_1000_mod_13_l1971_197164

theorem nine_pow_1000_mod_13 :
  (9^1000) % 13 = 9 :=
by
  have h1 : 9^1 % 13 = 9 := by sorry
  have h2 : 9^2 % 13 = 3 := by sorry
  have h3 : 9^3 % 13 = 1 := by sorry
  have cycle : ∀ n, 9^(3 * n + 1) % 13 = 9 := by sorry
  exact (cycle 333)

end NUMINAMATH_GPT_nine_pow_1000_mod_13_l1971_197164


namespace NUMINAMATH_GPT_kelly_initially_had_l1971_197123

def kelly_needs_to_pick : ℕ := 49
def kelly_will_have : ℕ := 105

theorem kelly_initially_had :
  kelly_will_have - kelly_needs_to_pick = 56 :=
by
  sorry

end NUMINAMATH_GPT_kelly_initially_had_l1971_197123


namespace NUMINAMATH_GPT_rectangle_ratio_expression_value_l1971_197130

theorem rectangle_ratio_expression_value (l w : ℝ) (S : ℝ) (h1 : l / w = (2 * (l + w)) / (2 * l)) (h2 : S = w / l) :
  S ^ (S ^ (S^2 + 1/S) + 1/S) + 1/S = Real.sqrt 5 :=
by
  sorry

end NUMINAMATH_GPT_rectangle_ratio_expression_value_l1971_197130


namespace NUMINAMATH_GPT_set_intersection_complement_l1971_197179

open Set

variable (A B U : Set ℕ)

theorem set_intersection_complement (A B : Set ℕ) (U : Set ℕ) (hU : U = {1, 2, 3, 4})
  (h1 : compl (A ∪ B) = {4}) (h2 : B = {1, 2}) :
  A ∩ compl B = {3} :=
by
  sorry

end NUMINAMATH_GPT_set_intersection_complement_l1971_197179


namespace NUMINAMATH_GPT_equality_of_ha_l1971_197134

theorem equality_of_ha 
  {p a b α β γ : ℝ} 
  (h1 : h_a = (2 * (p - a) * Real.cos (β / 2) * Real.cos (γ / 2)) / Real.cos (α / 2))
  (h2 : h_a = (2 * (p - b) * Real.sin (β / 2) * Real.cos (γ / 2)) / Real.sin (α / 2)) : 
  (2 * (p - a) * Real.cos (β / 2) * Real.cos (γ / 2)) / Real.cos (α / 2) = 
  (2 * (p - b) * Real.sin (β / 2) * Real.cos (γ / 2)) / Real.sin (α / 2) :=
by sorry

end NUMINAMATH_GPT_equality_of_ha_l1971_197134


namespace NUMINAMATH_GPT_wire_length_l1971_197170

theorem wire_length (r_sphere r_wire : ℝ) (h : ℝ) (V : ℝ)
  (h₁ : r_sphere = 24) (h₂ : r_wire = 16)
  (h₃ : V = 4 / 3 * Real.pi * r_sphere ^ 3)
  (h₄ : V = Real.pi * r_wire ^ 2 * h): 
  h = 72 := by
  -- we can use provided condition to show that h = 72, proof details omitted
  sorry

end NUMINAMATH_GPT_wire_length_l1971_197170


namespace NUMINAMATH_GPT_geometric_triangle_condition_right_geometric_triangle_condition_l1971_197117

-- Definitions for the geometric progression
def geometric_sequence (a b c q : ℝ) : Prop :=
  b = a * q ∧ c = a * q^2

-- Conditions for forming a triangle
def forms_triangle (a b c : ℝ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

-- Conditions for forming a right triangle using Pythagorean theorem
def right_triangle (a b c : ℝ) : Prop :=
  a^2 + b^2 = c^2 ∨ a^2 + c^2 = b^2 ∨ b^2 + c^2 = a^2

theorem geometric_triangle_condition (a q : ℝ) (h1 : 1 ≤ q) (h2 : q < (1 + Real.sqrt 5) / 2) :
  ∃ (b c : ℝ), geometric_sequence a b c q ∧ forms_triangle a b c := 
sorry

theorem right_geometric_triangle_condition (a q : ℝ) :
  q = Real.sqrt ((1 + Real.sqrt 5) / 2) →
  ∃ (b c : ℝ), geometric_sequence a b c q ∧ right_triangle a b c :=
sorry

end NUMINAMATH_GPT_geometric_triangle_condition_right_geometric_triangle_condition_l1971_197117


namespace NUMINAMATH_GPT_star_polygon_net_of_pyramid_l1971_197146

theorem star_polygon_net_of_pyramid (R r : ℝ) (h : R > r) : R > 2 * r :=
by
  sorry

end NUMINAMATH_GPT_star_polygon_net_of_pyramid_l1971_197146


namespace NUMINAMATH_GPT_quadrilateral_iff_segments_lt_half_l1971_197196

theorem quadrilateral_iff_segments_lt_half (a b c d : ℝ) (h₁ : a + b + c + d = 1) (h₂ : a ≤ b) (h₃ : b ≤ c) (h₄ : c ≤ d) : 
    (a + b > d) ∧ (a + c > d) ∧ (a + b + c > d) ∧ (b + c > d) ↔ a < 1/2 ∧ b < 1/2 ∧ c < 1/2 ∧ d < 1/2 :=
by
  sorry

end NUMINAMATH_GPT_quadrilateral_iff_segments_lt_half_l1971_197196


namespace NUMINAMATH_GPT_card_probability_l1971_197144

-- Definitions to capture the problem's conditions in Lean
def total_cards : ℕ := 52
def remaining_after_first : ℕ := total_cards - 1
def remaining_after_second : ℕ := total_cards - 2

def kings : ℕ := 4
def non_heart_kings : ℕ := 3
def non_kings_in_hearts : ℕ := 12
def spades_and_diamonds : ℕ := 26

-- Define probabilities for each step
def prob_first_king : ℚ := non_heart_kings / total_cards
def prob_second_heart : ℚ := non_kings_in_hearts / remaining_after_first
def prob_third_spade_or_diamond : ℚ := spades_and_diamonds / remaining_after_second

-- Calculate total probability
def total_probability : ℚ := prob_first_king * prob_second_heart * prob_third_spade_or_diamond

-- Theorem statement that encapsulates the problem
theorem card_probability : total_probability = 26 / 3675 :=
by sorry

end NUMINAMATH_GPT_card_probability_l1971_197144


namespace NUMINAMATH_GPT_nuts_per_box_l1971_197107

theorem nuts_per_box (N : ℕ)  
  (h1 : ∀ (boxes bolts_per_box : ℕ), boxes = 7 ∧ bolts_per_box = 11 → boxes * bolts_per_box = 77)
  (h2 : ∀ (boxes: ℕ), boxes = 3 → boxes * N = 3 * N)
  (h3 : ∀ (used_bolts purchased_bolts remaining_bolts : ℕ), purchased_bolts = 77 ∧ remaining_bolts = 3 → used_bolts = purchased_bolts - remaining_bolts)
  (h4 : ∀ (used_nuts purchased_nuts remaining_nuts : ℕ), purchased_nuts = 3 * N ∧ remaining_nuts = 6 → used_nuts = purchased_nuts - remaining_nuts)
  (h5 : ∀ (used_bolts used_nuts total_used : ℕ), used_bolts = 74 ∧ used_nuts = 3 * N - 6 → total_used = used_bolts + used_nuts)
  (h6 : total_used_bolts_and_nuts = 113) :
  N = 15 :=
by
  sorry

end NUMINAMATH_GPT_nuts_per_box_l1971_197107


namespace NUMINAMATH_GPT_least_range_product_multiple_840_l1971_197159

def is_multiple (x y : Nat) : Prop :=
  ∃ k : Nat, y = k * x

theorem least_range_product_multiple_840 : 
  ∃ (a : Nat), a > 0 ∧ ∀ (n : Nat), (n = 3) → is_multiple 840 (List.foldr (· * ·) 1 (List.range' a n)) := 
by {
  sorry
}

end NUMINAMATH_GPT_least_range_product_multiple_840_l1971_197159


namespace NUMINAMATH_GPT_percentage_of_Y_salary_l1971_197188

variable (X Y : ℝ)
variable (total_salary Y_salary : ℝ)
variable (P : ℝ)

theorem percentage_of_Y_salary :
  total_salary = 638 ∧ Y_salary = 290 ∧ X = (P / 100) * Y_salary → P = 120 := by
  sorry

end NUMINAMATH_GPT_percentage_of_Y_salary_l1971_197188


namespace NUMINAMATH_GPT_lcm_135_468_l1971_197199

theorem lcm_135_468 : Nat.lcm 135 468 = 7020 := by
  sorry

end NUMINAMATH_GPT_lcm_135_468_l1971_197199


namespace NUMINAMATH_GPT_hyperbola_equation_l1971_197118

noncomputable def hyperbola (a b : ℝ) := ∀ x y : ℝ, (x^2 / a^2) - (y^2 / b^2) = 1

def parabola_focus_same_as_hyperbola_focus (c : ℝ) : Prop :=
  ∃ x y : ℝ, y^2 = 4 * (10:ℝ).sqrt * x ∧ (c, 0) = ((10:ℝ).sqrt, 0)

def hyperbola_eccentricity (c a : ℝ) := (c / a) = (10:ℝ).sqrt / 3

theorem hyperbola_equation :
  ∃ a b : ℝ, (hyperbola a b) ∧
  (parabola_focus_same_as_hyperbola_focus ((10:ℝ).sqrt)) ∧
  (hyperbola_eccentricity ((10:ℝ).sqrt) a) ∧
  ((a = 3) ∧ (b = 1)) :=
sorry

end NUMINAMATH_GPT_hyperbola_equation_l1971_197118


namespace NUMINAMATH_GPT_chess_tournament_possible_l1971_197163

section ChessTournament

structure Player :=
  (name : String)
  (wins : ℕ)
  (draws : ℕ)
  (losses : ℕ)

def points (p : Player) : ℕ :=
  p.wins + p.draws / 2

def is_possible (A B C : Player) : Prop :=
  (points A > points B) ∧ (points A > points C) ∧
  (points C < points B) ∧
  (A.wins < B.wins) ∧ (A.wins < C.wins) ∧
  (C.wins > B.wins)

theorem chess_tournament_possible (A B C : Player) :
  is_possible A B C :=
  sorry

end ChessTournament

end NUMINAMATH_GPT_chess_tournament_possible_l1971_197163


namespace NUMINAMATH_GPT_tank_empties_in_4320_minutes_l1971_197138

-- Define the initial conditions
def tankVolumeCubicFeet: ℝ := 30
def inletPipeRateCubicInchesPerMin: ℝ := 5
def outletPipe1RateCubicInchesPerMin: ℝ := 9
def outletPipe2RateCubicInchesPerMin: ℝ := 8
def feetToInches: ℝ := 12

-- Conversion from cubic feet to cubic inches
def tankVolumeCubicInches: ℝ := tankVolumeCubicFeet * feetToInches^3

-- Net rate of emptying in cubic inches per minute
def netRateOfEmptying: ℝ := (outletPipe1RateCubicInchesPerMin + outletPipe2RateCubicInchesPerMin) - inletPipeRateCubicInchesPerMin

-- Time to empty the tank
noncomputable def timeToEmptyTank: ℝ := tankVolumeCubicInches / netRateOfEmptying

-- The theorem to prove
theorem tank_empties_in_4320_minutes :
  timeToEmptyTank = 4320 := by
  sorry

end NUMINAMATH_GPT_tank_empties_in_4320_minutes_l1971_197138


namespace NUMINAMATH_GPT_solve_expression_l1971_197109

theorem solve_expression : (0.76 ^ 3 - 0.008) / (0.76 ^ 2 + 0.76 * 0.2 + 0.04) = 0.560 := 
by
  sorry

end NUMINAMATH_GPT_solve_expression_l1971_197109


namespace NUMINAMATH_GPT_largest_integer_l1971_197121

def bin_op (n : ℤ) : ℤ := n - 5 * n

theorem largest_integer (n : ℤ) (h : 0 < n) (h' : bin_op n < 18) : n = 4 := sorry

end NUMINAMATH_GPT_largest_integer_l1971_197121


namespace NUMINAMATH_GPT_ratio_boys_to_girls_l1971_197195

theorem ratio_boys_to_girls (total_students girls : ℕ) (h1 : total_students = 455) (h2 : girls = 175) :
  let boys := total_students - girls
  (boys : ℕ) / Nat.gcd boys girls = 8 / 1 ∧ (girls : ℕ) / Nat.gcd boys girls = 5 / 1 :=
by
  sorry

end NUMINAMATH_GPT_ratio_boys_to_girls_l1971_197195


namespace NUMINAMATH_GPT_triangle_area_ratio_l1971_197102

theorem triangle_area_ratio {A B C : ℝ} {a b c : ℝ} 
  (h : 2 * Real.sin A * Real.cos (B - C) + Real.sin (2 * A) = 2 / 3) 
  (S1 : ℝ) (S2 : ℝ) :
  S1 / S2 = 1 / (3 * Real.pi) :=
sorry

end NUMINAMATH_GPT_triangle_area_ratio_l1971_197102


namespace NUMINAMATH_GPT_part1_part2_l1971_197150

variable {a b c : ℝ}

theorem part1 (hpos : a > 0 ∧ b > 0 ∧ c > 0) (hsum : a + b + c = 1) : 
  a * b + b * c + a * c ≤ 1 / 3 := 
sorry 

theorem part2 (hpos : a > 0 ∧ b > 0 ∧ c > 0) (hsum : a + b + c = 1) : 
  a^2 / b + b^2 / c + c^2 / a ≥ 1 := 
sorry

end NUMINAMATH_GPT_part1_part2_l1971_197150


namespace NUMINAMATH_GPT_books_total_pages_l1971_197154

theorem books_total_pages (x y z : ℕ) 
  (h1 : (2 / 3 : ℚ) * x - (1 / 3 : ℚ) * x = 20)
  (h2 : (3 / 5 : ℚ) * y - (2 / 5 : ℚ) * y = 15)
  (h3 : (3 / 4 : ℚ) * z - (1 / 4 : ℚ) * z = 30) : 
  x = 60 ∧ y = 75 ∧ z = 60 :=
by
  sorry

end NUMINAMATH_GPT_books_total_pages_l1971_197154


namespace NUMINAMATH_GPT_john_pays_2010_dollars_l1971_197131

-- Define the main problem as the number of ways to pay 2010$ using 2, 5, and 10$ notes.
theorem john_pays_2010_dollars :
  ∃ (count : ℕ), count = 20503 ∧
  ∀ (x y z : ℕ), (2 * x + 5 * y + 10 * z = 2010) → (x % 5 = 0) → (y % 2 = 0) → count = 20503 :=
by sorry

end NUMINAMATH_GPT_john_pays_2010_dollars_l1971_197131


namespace NUMINAMATH_GPT_sufficient_but_not_necessary_condition_for_hyperbola_l1971_197110

theorem sufficient_but_not_necessary_condition_for_hyperbola (k : ℝ) :
  (∃ k : ℝ, k > 3 ∧ (∃ x y : ℝ, (x^2) / (k - 3) - (y^2) / (k + 3) = 1)) ∧ 
  (∃ k : ℝ, k < -3 ∧ (∃ x y : ℝ, (x^2) / (k - 3) - (y^2) / (k + 3) = 1)) :=
    sorry

end NUMINAMATH_GPT_sufficient_but_not_necessary_condition_for_hyperbola_l1971_197110


namespace NUMINAMATH_GPT_sum_zero_l1971_197132

variable {a b c d : ℝ}

-- Pairwise distinct real numbers
axiom h1 : a ≠ b
axiom h2 : a ≠ c
axiom h3 : a ≠ d
axiom h4 : b ≠ c
axiom h5 : b ≠ d
axiom h6 : c ≠ d

-- Given condition
axiom h : (a^2 + b^2 - 1) * (a + b) = (b^2 + c^2 - 1) * (b + c) ∧ 
          (b^2 + c^2 - 1) * (b + c) = (c^2 + d^2 - 1) * (c + d)

theorem sum_zero : a + b + c + d = 0 :=
sorry

end NUMINAMATH_GPT_sum_zero_l1971_197132


namespace NUMINAMATH_GPT_part1_part2_l1971_197167

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := (Real.exp x / x) - Real.log x + x - a

theorem part1 (a : ℝ) (h : ∀ x > 0, f x a ≥ 0) : a ≤ Real.exp 1 + 1 := 
sorry

theorem part2 {a : ℝ} {x1 x2 : ℝ} (hx1 : 0 < x1) (hx2 : 0 < x2) 
  (hz1 : f x1 a = 0) (hz2 : f x2 a = 0) : x1 * x2 < 1 := 
sorry

end NUMINAMATH_GPT_part1_part2_l1971_197167


namespace NUMINAMATH_GPT_spent_on_veggies_l1971_197119

noncomputable def total_amount : ℕ := 167
noncomputable def spent_on_meat : ℕ := 17
noncomputable def spent_on_chicken : ℕ := 22
noncomputable def spent_on_eggs : ℕ := 5
noncomputable def spent_on_dog_food : ℕ := 45
noncomputable def amount_left : ℕ := 35

theorem spent_on_veggies : 
  total_amount - (spent_on_meat + spent_on_chicken + spent_on_eggs + spent_on_dog_food + amount_left) = 43 := 
by 
  sorry

end NUMINAMATH_GPT_spent_on_veggies_l1971_197119


namespace NUMINAMATH_GPT_find_f_value_l1971_197165

-- Condition 1: f is an odd function
def is_odd_function (f : ℝ → ℝ) := ∀ x, f (-x) = -f x

-- Condition 2: f has a period of 5
def is_periodic (f : ℝ → ℝ) (p : ℝ) := ∀ x, f (x + p) = f x

-- Condition 3: f(-3) = -4
def f_value_at_neg3 (f : ℝ → ℝ) := f (-3) = -4

-- Condition 4: cos(α) = 1 / 2
def cos_alpha_value (α : ℝ) := Real.cos α = 1 / 2

noncomputable def f (x : ℝ) : ℝ := sorry
noncomputable def α : ℝ := sorry

theorem find_f_value (h_odd : is_odd_function f)
                     (h_periodic : is_periodic f 5)
                     (h_f_neg3 : f_value_at_neg3 f)
                     (h_cos_alpha : cos_alpha_value α) :
  f (4 * Real.cos (2 * α)) = 4 := 
sorry

end NUMINAMATH_GPT_find_f_value_l1971_197165


namespace NUMINAMATH_GPT_no_real_solution_for_x_l1971_197185

theorem no_real_solution_for_x (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0)
  (h1 : x + 1 / y = 8) (h2 : y + 1 / x = 7 / 20) : false :=
by sorry

end NUMINAMATH_GPT_no_real_solution_for_x_l1971_197185


namespace NUMINAMATH_GPT_F_final_coordinates_l1971_197183

-- Define the original coordinates of point F
def F : ℝ × ℝ := (5, 2)

-- Reflection over the y-axis changes the sign of the x-coordinate
def reflect_y_axis (p : ℝ × ℝ) : ℝ × ℝ := (-p.1, p.2)

-- Reflection over the line y = x involves swapping x and y coordinates
def reflect_y_eq_x (p : ℝ × ℝ) : ℝ × ℝ := (p.2, p.1)

-- The combined transformation: reflect over the y-axis, then reflect over y = x
def F_final : ℝ × ℝ := reflect_y_eq_x (reflect_y_axis F)

-- The proof statement
theorem F_final_coordinates : F_final = (2, -5) :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_F_final_coordinates_l1971_197183


namespace NUMINAMATH_GPT_problem_statement_l1971_197115

variable {F : Type*} [Field F]

theorem problem_statement (m : F) (h : m + 1 / m = 6) : m^2 + 1 / m^2 + 4 = 38 :=
by
  sorry

end NUMINAMATH_GPT_problem_statement_l1971_197115


namespace NUMINAMATH_GPT_incorrect_average_l1971_197145

theorem incorrect_average (S : ℕ) (A_correct : ℕ) (A_incorrect : ℕ) (S_correct : ℕ) 
  (h1 : S = 135)
  (h2 : A_correct = 19)
  (h3 : A_incorrect = (S + 25) / 10)
  (h4 : S_correct = (S + 55) / 10)
  (h5 : S_correct = A_correct) :
  A_incorrect = 16 :=
by
  -- The proof will go here, which is skipped with a 'sorry'
  sorry

end NUMINAMATH_GPT_incorrect_average_l1971_197145
