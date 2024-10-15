import Mathlib

namespace NUMINAMATH_GPT_compute_value_l1417_141744

variables {p q r : ℝ}

theorem compute_value (h1 : (p * q) / (p + r) + (q * r) / (q + p) + (r * p) / (r + q) = -7)
                      (h2 : (p * r) / (p + r) + (q * p) / (q + p) + (r * q) / (r + q) = 8) :
  (q / (p + q) + r / (q + r) + p / (r + p)) = 9 :=
sorry

end NUMINAMATH_GPT_compute_value_l1417_141744


namespace NUMINAMATH_GPT_compare_log_exp_powers_l1417_141747

variable (a b c : ℝ)

theorem compare_log_exp_powers (h1 : a = Real.log 0.3 / Real.log 2)
                               (h2 : b = Real.exp (Real.log 2 * 0.1))
                               (h3 : c = Real.exp (Real.log 0.2 * 1.3)) :
  a < c ∧ c < b :=
by
  sorry

end NUMINAMATH_GPT_compare_log_exp_powers_l1417_141747


namespace NUMINAMATH_GPT_find_m_l1417_141746

def U : Set ℤ := {-1, 2, 3, 6}
def A (m : ℤ) : Set ℤ := {x | x^2 - 5 * x + m = 0}
def complement_U_A (m : ℤ) : Set ℤ := U \ A m

theorem find_m (m : ℤ) (hU : U = {-1, 2, 3, 6}) (hcomp : complement_U_A m = {2, 3}) :
  m = -6 := by
  sorry

end NUMINAMATH_GPT_find_m_l1417_141746


namespace NUMINAMATH_GPT_find_n_and_d_l1417_141730

theorem find_n_and_d (n d : ℕ) (hn_pos : 0 < n) (hd_digit : d < 10)
    (h1 : 3 * n^2 + 2 * n + d = 263)
    (h2 : 3 * n^2 + 2 * n + 4 = 1 * 8^3 + 1 * 8^2 + d * 8 + 1) :
    n + d = 12 := 
sorry

end NUMINAMATH_GPT_find_n_and_d_l1417_141730


namespace NUMINAMATH_GPT_chocolates_sold_in_second_week_l1417_141716

theorem chocolates_sold_in_second_week
  (c₁ c₂ c₃ c₄ c₅ : ℕ)
  (h₁ : c₁ = 75)
  (h₃ : c₃ = 75)
  (h₄ : c₄ = 70)
  (h₅ : c₅ = 68)
  (h_mean : (c₁ + c₂ + c₃ + c₄ + c₅) / 5 = 71) :
  c₂ = 67 := 
sorry

end NUMINAMATH_GPT_chocolates_sold_in_second_week_l1417_141716


namespace NUMINAMATH_GPT_ratio_of_width_to_perimeter_l1417_141713

-- Condition definitions
def length := 22
def width := 13
def perimeter := 2 * (length + width)

-- Statement of the problem in Lean 4
theorem ratio_of_width_to_perimeter : width = 13 ∧ length = 22 → width * 70 = 13 * perimeter :=
by
  sorry

end NUMINAMATH_GPT_ratio_of_width_to_perimeter_l1417_141713


namespace NUMINAMATH_GPT_Carly_injured_week_miles_l1417_141729

def week1_miles : ℕ := 2
def week2_miles : ℕ := week1_miles * 2 + 3
def week3_miles : ℕ := week2_miles * 9 / 7
def week4_miles : ℕ := week3_miles - 5

theorem Carly_injured_week_miles : week4_miles = 4 :=
  by
    sorry

end NUMINAMATH_GPT_Carly_injured_week_miles_l1417_141729


namespace NUMINAMATH_GPT_jean_average_mark_l1417_141781

/-
  Jean writes five tests and achieves the following marks: 80, 70, 60, 90, and 80.
  Prove that her average mark on these five tests is 76.
-/
theorem jean_average_mark : 
  let marks := [80, 70, 60, 90, 80]
  let total_marks := marks.sum
  let number_of_tests := marks.length
  let average_mark := total_marks / number_of_tests
  average_mark = 76 :=
by 
  let marks := [80, 70, 60, 90, 80]
  let total_marks := marks.sum
  let number_of_tests := marks.length
  let average_mark := total_marks / number_of_tests
  sorry

end NUMINAMATH_GPT_jean_average_mark_l1417_141781


namespace NUMINAMATH_GPT_smallest_integer_2023m_54321n_l1417_141767

theorem smallest_integer_2023m_54321n : ∃ (m n : ℤ), 2023 * m + 54321 * n = 1 :=
sorry

end NUMINAMATH_GPT_smallest_integer_2023m_54321n_l1417_141767


namespace NUMINAMATH_GPT_coin_difference_l1417_141792

theorem coin_difference (h : ∃ x y z : ℕ, 5*x + 10*y + 20*z = 40) : (∃ x : ℕ, 5*x = 40) → (∃ y : ℕ, 20*y = 40) → 8 - 2 = 6 :=
by
  intros h1 h2
  exact rfl

end NUMINAMATH_GPT_coin_difference_l1417_141792


namespace NUMINAMATH_GPT_students_got_on_second_stop_l1417_141738

-- Given conditions translated into definitions and hypotheses
def students_after_first_stop := 39
def students_after_second_stop := 68

-- The proof statement we aim to prove
theorem students_got_on_second_stop : (students_after_second_stop - students_after_first_stop) = 29 := by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_students_got_on_second_stop_l1417_141738


namespace NUMINAMATH_GPT_total_cost_of_returned_packets_l1417_141761

/--
  Martin bought 10 packets of milk with varying prices.
  The average price (arithmetic mean) of all the packets is 25¢.
  If Martin returned three packets to the retailer, and the average price of the remaining packets was 20¢,
  then the total cost, in cents, of the three returned milk packets is 110¢.
-/
theorem total_cost_of_returned_packets 
  (T10 : ℕ) (T7 : ℕ) (average_price_10 : T10 / 10 = 25)
  (average_price_7 : T7 / 7 = 20) :
  (T10 - T7 = 110) := 
sorry

end NUMINAMATH_GPT_total_cost_of_returned_packets_l1417_141761


namespace NUMINAMATH_GPT_probability_one_out_of_three_l1417_141794

def probability_passing_exactly_one (p : ℚ) (n k : ℕ) :=
  (Nat.choose n k) * (p^k) * ((1 - p)^(n - k))

theorem probability_one_out_of_three :
  probability_passing_exactly_one (1/3) 3 1 = 4/9 :=
by sorry

end NUMINAMATH_GPT_probability_one_out_of_three_l1417_141794


namespace NUMINAMATH_GPT_project_completion_time_l1417_141779

def process_duration (a b c d e f : Nat) : Nat :=
  let duration_c := max a b + c
  let duration_d := duration_c + d
  let duration_e := duration_c + e
  let duration_f := max duration_d duration_e + f
  duration_f

theorem project_completion_time :
  ∀ (a b c d e f : Nat), a = 2 → b = 3 → c = 2 → d = 5 → e = 4 → f = 1 →
  process_duration a b c d e f = 11 := by
  intros
  subst_vars
  sorry

end NUMINAMATH_GPT_project_completion_time_l1417_141779


namespace NUMINAMATH_GPT_maximum_volume_regular_triangular_pyramid_l1417_141791

-- Given values
def R : ℝ := 1

-- Prove the maximum volume
theorem maximum_volume_regular_triangular_pyramid : 
  ∃ (V_max : ℝ), V_max = (8 * Real.sqrt 3) / 27 := 
by 
  sorry

end NUMINAMATH_GPT_maximum_volume_regular_triangular_pyramid_l1417_141791


namespace NUMINAMATH_GPT_gnomes_in_fifth_house_l1417_141733

-- Defining the problem conditions
def num_houses : Nat := 5
def gnomes_per_house : Nat := 3
def total_gnomes : Nat := 20

-- Defining the condition for the first four houses
def gnomes_in_first_four_houses : Nat := 4 * gnomes_per_house

-- Statement of the problem
theorem gnomes_in_fifth_house : 20 - (4 * 3) = 8 := by
  sorry

end NUMINAMATH_GPT_gnomes_in_fifth_house_l1417_141733


namespace NUMINAMATH_GPT_countDistinguishedDigitsTheorem_l1417_141711

-- Define a function to count numbers with four distinct digits where leading zeros are allowed
def countDistinguishedDigits : Nat :=
  10 * 9 * 8 * 7

-- State the theorem we need to prove
theorem countDistinguishedDigitsTheorem :
  countDistinguishedDigits = 5040 := 
by
  sorry

end NUMINAMATH_GPT_countDistinguishedDigitsTheorem_l1417_141711


namespace NUMINAMATH_GPT_number_of_ways_to_divide_l1417_141795

-- Define the given shape
structure Shape :=
  (sides : Nat) -- Number of 3x1 stripes along the sides
  (centre : Nat) -- Size of the central square (3x3)

-- Define the specific problem shape
def problem_shape : Shape :=
  { sides := 4, centre := 9 } -- 3x1 stripes on all sides and a 3x3 centre

-- Theorem stating the number of ways to divide the shape into 1x3 rectangles
theorem number_of_ways_to_divide (s : Shape) (h1 : s.sides = 4) (h2 : s.centre = 9) : 
  ∃ ways, ways = 2 :=
by
  -- The proof is skipped
  sorry

end NUMINAMATH_GPT_number_of_ways_to_divide_l1417_141795


namespace NUMINAMATH_GPT_find_m_l1417_141774

-- Define the pattern of splitting cubes into odd numbers
def split_cubes (m : ℕ) : List ℕ := 
  let rec odd_numbers (n : ℕ) : List ℕ :=
    if n = 0 then []
    else (2 * n - 1) :: odd_numbers (n - 1)
  odd_numbers m

-- Define the condition that 59 is part of the split numbers of m^3
def is_split_number (m : ℕ) (n : ℕ) : Prop :=
  n ∈ (split_cubes m)

-- Prove that if 59 is part of the split numbers of m^3, then m = 8
theorem find_m (m : ℕ) (h : is_split_number m 59) : m = 8 := 
sorry

end NUMINAMATH_GPT_find_m_l1417_141774


namespace NUMINAMATH_GPT_math_problem_l1417_141754

-- Define the individual numbers
def a : Int := 153
def b : Int := 39
def c : Int := 27
def d : Int := 21

-- Define the entire expression and its expected result
theorem math_problem : (a + b + c + d) * 2 = 480 := by
  sorry

end NUMINAMATH_GPT_math_problem_l1417_141754


namespace NUMINAMATH_GPT_y_intercept_of_tangent_line_l1417_141793

def point (x y : ℝ) : Prop := true

def circle_eq (x y : ℝ) : ℝ := x^2 + y^2 + 4*x - 2*y + 3

theorem y_intercept_of_tangent_line :
  ∃ m b : ℝ,
  (∀ x : ℝ, circle_eq x (m*x + b) = 0 → m * m = 1) ∧
  (∃ P: ℝ × ℝ, P = (-1, 0)) ∧
  ∀ b : ℝ, (∃ m : ℝ, m = 1 ∧ (∃ P: ℝ × ℝ, P = (-1, 0)) ∧ b = 1) := 
sorry

end NUMINAMATH_GPT_y_intercept_of_tangent_line_l1417_141793


namespace NUMINAMATH_GPT_no_perfect_square_in_seq_l1417_141705

noncomputable def seq : ℕ → ℕ
| 0       => 2
| 1       => 7
| (n + 2) => 4 * seq (n + 1) - seq n

theorem no_perfect_square_in_seq :
  ¬ ∃ (n : ℕ), ∃ (k : ℕ), (seq n) = k * k :=
sorry

end NUMINAMATH_GPT_no_perfect_square_in_seq_l1417_141705


namespace NUMINAMATH_GPT_solve_abs_eq_zero_l1417_141714

theorem solve_abs_eq_zero : ∃ x : ℝ, |5 * x - 3| = 0 ↔ x = 3 / 5 :=
by
  sorry

end NUMINAMATH_GPT_solve_abs_eq_zero_l1417_141714


namespace NUMINAMATH_GPT_find_minimum_value_l1417_141706

theorem find_minimum_value (c : ℝ) : 
  (∀ c : ℝ, (c = -12) ↔ (∀ d : ℝ, (1 / 3) * d^2 + 8 * d - 7 ≥ (1 / 3) * (-12)^2 + 8 * (-12) - 7)) :=
sorry

end NUMINAMATH_GPT_find_minimum_value_l1417_141706


namespace NUMINAMATH_GPT_alcohol_percentage_new_mixture_l1417_141752

namespace AlcoholMixtureProblem

def original_volume : ℝ := 3
def alcohol_percentage : ℝ := 0.33
def additional_water_volume : ℝ := 1
def new_volume : ℝ := original_volume + additional_water_volume
def alcohol_amount : ℝ := original_volume * alcohol_percentage

theorem alcohol_percentage_new_mixture : (alcohol_amount / new_volume) * 100 = 24.75 := by
  sorry

end AlcoholMixtureProblem

end NUMINAMATH_GPT_alcohol_percentage_new_mixture_l1417_141752


namespace NUMINAMATH_GPT_original_number_is_106_25_l1417_141756

theorem original_number_is_106_25 (x : ℝ) (h : (x + 0.375 * x) - (x - 0.425 * x) = 85) : x = 106.25 := by
  sorry

end NUMINAMATH_GPT_original_number_is_106_25_l1417_141756


namespace NUMINAMATH_GPT_sufficient_but_not_necessary_l1417_141765

theorem sufficient_but_not_necessary (x : ℝ) :
  (x > 1 → x^2 + 2 * x > 0) ∧ ¬(x^2 + 2 * x > 0 → x > 1) :=
by
  sorry

end NUMINAMATH_GPT_sufficient_but_not_necessary_l1417_141765


namespace NUMINAMATH_GPT_number_of_zeros_l1417_141739

noncomputable def f : ℝ → ℝ := sorry
noncomputable def f' : ℝ → ℝ := sorry
noncomputable def f'' : ℝ → ℝ := sorry

def odd_function (f : ℝ → ℝ) := ∀ x, f (-x) = -f x

def conditions (f : ℝ → ℝ) (f'' : ℝ → ℝ) :=
  odd_function f ∧ ∀ x : ℝ, x < 0 → (2 * f x + x * f'' x < x * f x)

theorem number_of_zeros (f : ℝ → ℝ) (f'' : ℝ → ℝ) (h : conditions f f'') :
  ∃! x : ℝ, f x = 0 :=
sorry

end NUMINAMATH_GPT_number_of_zeros_l1417_141739


namespace NUMINAMATH_GPT_kittens_given_to_Jessica_is_3_l1417_141771

def kittens_initial := 18
def kittens_given_to_Sara := 6
def kittens_now := 9

def kittens_after_Sara := kittens_initial - kittens_given_to_Sara
def kittens_given_to_Jessica := kittens_after_Sara - kittens_now

theorem kittens_given_to_Jessica_is_3 : kittens_given_to_Jessica = 3 := by
  sorry

end NUMINAMATH_GPT_kittens_given_to_Jessica_is_3_l1417_141771


namespace NUMINAMATH_GPT_no_ratio_p_squared_l1417_141724

theorem no_ratio_p_squared {p : ℕ} (hp : Nat.Prime p) :
  ∀ l n m : ℕ, 1 ≤ l → (∃ k : ℕ, k = p^l) → ((2 * (n*(n+1)) = (m*(m+1))*p^(2*l)) → false) := 
sorry

end NUMINAMATH_GPT_no_ratio_p_squared_l1417_141724


namespace NUMINAMATH_GPT_vacation_expense_sharing_l1417_141740

def alice_paid : ℕ := 90
def bob_paid : ℕ := 150
def charlie_paid : ℕ := 120
def donna_paid : ℕ := 240
def total_paid : ℕ := alice_paid + bob_paid + charlie_paid + donna_paid
def individual_share : ℕ := total_paid / 4

def alice_owes : ℕ := individual_share - alice_paid
def charlie_owes : ℕ := individual_share - charlie_paid
def donna_owes : ℕ := donna_paid - individual_share

def a : ℕ := charlie_owes
def b : ℕ := donna_owes - (donna_owes - charlie_owes)

theorem vacation_expense_sharing : a - b = 0 :=
by
  sorry

end NUMINAMATH_GPT_vacation_expense_sharing_l1417_141740


namespace NUMINAMATH_GPT_gcd_computation_l1417_141700

theorem gcd_computation (a b : ℕ) (h₁ : a = 7260) (h₂ : b = 540) : 
  Nat.gcd a b - 12 + 5 = 53 :=
by
  rw [h₁, h₂]
  sorry

end NUMINAMATH_GPT_gcd_computation_l1417_141700


namespace NUMINAMATH_GPT_solution_set_inequality_l1417_141727

theorem solution_set_inequality (x : ℝ) :
  ((x^2 - 4) * (x - 6)^2 ≤ 0) ↔ (-2 ≤ x ∧ x ≤ 2 ∨ x = 6) :=
  sorry

end NUMINAMATH_GPT_solution_set_inequality_l1417_141727


namespace NUMINAMATH_GPT_total_berries_l1417_141799

theorem total_berries (S_stacy S_steve S_skylar : ℕ) 
  (h1 : S_stacy = 800)
  (h2 : S_stacy = 4 * S_steve)
  (h3 : S_steve = 2 * S_skylar) :
  S_stacy + S_steve + S_skylar = 1100 :=
by
  sorry

end NUMINAMATH_GPT_total_berries_l1417_141799


namespace NUMINAMATH_GPT_same_yield_among_squares_l1417_141748

-- Define the conditions
def rectangular_schoolyard (length : ℝ) (width : ℝ) := length = 70 ∧ width = 35

def total_harvest (harvest : ℝ) := harvest = 1470 -- in kilograms (14.7 quintals)

def smaller_square (side : ℝ) := side = 0.7

-- Define the proof problem
theorem same_yield_among_squares :
  ∃ side : ℝ, smaller_square side ∧
  ∃ length width harvest : ℝ, rectangular_schoolyard length width ∧ total_harvest harvest →
  ∃ (yield1 yield2 : ℝ), yield1 = yield2 ∧ yield1 ≠ 0 ∧ yield2 ≠ 0 :=
by sorry

end NUMINAMATH_GPT_same_yield_among_squares_l1417_141748


namespace NUMINAMATH_GPT_perpendicular_lines_l1417_141719

theorem perpendicular_lines :
  ∃ m₁ m₄, (m₁ : ℚ) * (m₄ : ℚ) = -1 ∧
  (∀ x y : ℚ, 4 * y - 3 * x = 16 → y = m₁ * x + 4) ∧
  (∀ x y : ℚ, 3 * y + 4 * x = 15 → y = m₄ * x + 5) :=
by sorry

end NUMINAMATH_GPT_perpendicular_lines_l1417_141719


namespace NUMINAMATH_GPT_SumataFamilyTotalMiles_l1417_141758

def miles_per_day := 250
def days := 5

theorem SumataFamilyTotalMiles : miles_per_day * days = 1250 :=
by
  sorry

end NUMINAMATH_GPT_SumataFamilyTotalMiles_l1417_141758


namespace NUMINAMATH_GPT_subtraction_of_largest_three_digit_from_smallest_five_digit_l1417_141728

def largest_three_digit_number : ℕ := 999
def smallest_five_digit_number : ℕ := 10000

theorem subtraction_of_largest_three_digit_from_smallest_five_digit :
  smallest_five_digit_number - largest_three_digit_number = 9001 :=
by
  sorry

end NUMINAMATH_GPT_subtraction_of_largest_three_digit_from_smallest_five_digit_l1417_141728


namespace NUMINAMATH_GPT_calculate_moment_of_inertia_l1417_141745

noncomputable def moment_of_inertia (a ρ₀ k : ℝ) : ℝ :=
  8 * (a ^ (9/2)) * ((ρ₀ / 7) + (k * a / 9))

theorem calculate_moment_of_inertia (a ρ₀ k : ℝ) 
  (h₀ : 0 ≤ a) :
  moment_of_inertia a ρ₀ k = 8 * a ^ (9/2) * ((ρ₀ / 7) + (k * a / 9)) :=
sorry

end NUMINAMATH_GPT_calculate_moment_of_inertia_l1417_141745


namespace NUMINAMATH_GPT_inequality_f_x_f_a_l1417_141763

noncomputable def f (x : ℝ) : ℝ := x * x + x + 13

theorem inequality_f_x_f_a (a x : ℝ) (h : |x - a| < 1) : |f x * f a| < 2 * (|a| + 1) := 
sorry

end NUMINAMATH_GPT_inequality_f_x_f_a_l1417_141763


namespace NUMINAMATH_GPT_historical_fiction_new_releases_fraction_l1417_141773

noncomputable def HF_fraction_total_inventory : ℝ := 0.4
noncomputable def Mystery_fraction_total_inventory : ℝ := 0.3
noncomputable def SF_fraction_total_inventory : ℝ := 0.2
noncomputable def Romance_fraction_total_inventory : ℝ := 0.1

noncomputable def HF_new_release_percentage : ℝ := 0.35
noncomputable def Mystery_new_release_percentage : ℝ := 0.60
noncomputable def SF_new_release_percentage : ℝ := 0.45
noncomputable def Romance_new_release_percentage : ℝ := 0.80

noncomputable def historical_fiction_new_releases : ℝ := HF_fraction_total_inventory * HF_new_release_percentage
noncomputable def mystery_new_releases : ℝ := Mystery_fraction_total_inventory * Mystery_new_release_percentage
noncomputable def sf_new_releases : ℝ := SF_fraction_total_inventory * SF_new_release_percentage
noncomputable def romance_new_releases : ℝ := Romance_fraction_total_inventory * Romance_new_release_percentage

noncomputable def total_new_releases : ℝ :=
  historical_fiction_new_releases + mystery_new_releases + sf_new_releases + romance_new_releases

theorem historical_fiction_new_releases_fraction :
  (historical_fiction_new_releases / total_new_releases) = (2 / 7) :=
by
  sorry

end NUMINAMATH_GPT_historical_fiction_new_releases_fraction_l1417_141773


namespace NUMINAMATH_GPT_solve_f_inv_zero_l1417_141723

noncomputable def f (a b x : ℝ) : ℝ := 1 / (a * x + b)
noncomputable def f_inv (a b x : ℝ) : ℝ := sorry -- this is where the inverse function definition would go

theorem solve_f_inv_zero (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) : f_inv a b 0 = (1 / b) :=
by sorry

end NUMINAMATH_GPT_solve_f_inv_zero_l1417_141723


namespace NUMINAMATH_GPT_limit_log_div_x_alpha_l1417_141721

open Real

theorem limit_log_div_x_alpha (α : ℝ) (hα : α > 0) :
  (Filter.Tendsto (fun x => (log x) / (x^α)) Filter.atTop (nhds 0)) :=
by
  sorry

end NUMINAMATH_GPT_limit_log_div_x_alpha_l1417_141721


namespace NUMINAMATH_GPT_ratio_a7_b7_l1417_141750

variable (a b : ℕ → ℕ) -- Define sequences a and b
variable (S T : ℕ → ℕ) -- Define sums S and T

-- Define conditions: arithmetic sequences and given ratio
variable (h_arith_a : ∀ n, a (n + 1) - a n = a 1)
variable (h_arith_b : ∀ n, b (n + 1) - b n = b 1)
variable (h_sum_a : ∀ n, S n = (n + 1) * a 1 + n * a n)
variable (h_sum_b : ∀ n, T n = (n + 1) * b 1 + n * b n)
variable (h_ratio : ∀ n, (S n) / (T n) = (3 * n + 2) / (2 * n))

-- Define the problem statement using the given conditions
theorem ratio_a7_b7 : (a 7) / (b 7) = 41 / 26 :=
by
  sorry

end NUMINAMATH_GPT_ratio_a7_b7_l1417_141750


namespace NUMINAMATH_GPT_negative_root_no_positive_l1417_141732

theorem negative_root_no_positive (a : ℝ) : 
  (∃ x : ℝ, x < 0 ∧ |x| = ax + 1) ∧ (¬ ∃ x : ℝ, x > 0 ∧ |x| = ax + 1) → a > -1 :=
by
  sorry

end NUMINAMATH_GPT_negative_root_no_positive_l1417_141732


namespace NUMINAMATH_GPT_min_f_value_l1417_141790

noncomputable def f (x y z : ℝ) : ℝ := x^2 + 4 * x * y + 9 * y^2 + 8 * y * z + 3 * z^2

theorem min_f_value (x y z : ℝ) (hxyz_pos : 0 < x ∧ 0 < y ∧ 0 < z) (hxyz : x * y * z = 1) :
  f x y z ≥ 18 :=
sorry

end NUMINAMATH_GPT_min_f_value_l1417_141790


namespace NUMINAMATH_GPT_range_of_y_coordinate_of_C_l1417_141718

-- Define the given parabola equation
def on_parabola (x y : ℝ) : Prop := y^2 = x + 4

-- Define the coordinates for point A
def A : (ℝ × ℝ) := (0, 2)

-- Determine if points B and C lies on the parabola
def point_on_parabola (B C : ℝ × ℝ) : Prop :=
  on_parabola B.1 B.2 ∧ on_parabola C.1 C.2

-- Determine if lines AB and BC are perpendicular
def perpendicular_slopes (B C : ℝ × ℝ) : Prop :=
  let k_AB := (B.2 - A.2) / (B.1 - A.1)
  let k_BC := (C.2 - B.2) / (C.1 - B.1)
  k_AB * k_BC = -1

-- Prove the range for y-coordinate of C
theorem range_of_y_coordinate_of_C (B C : ℝ × ℝ) (h1 : point_on_parabola B C) (h2 : perpendicular_slopes B C) :
  C.2 ≤ 0 ∨ C.2 ≥ 4 := sorry

end NUMINAMATH_GPT_range_of_y_coordinate_of_C_l1417_141718


namespace NUMINAMATH_GPT_correct_equation_l1417_141797

variables (x y : ℕ)

-- Conditions
def condition1 : Prop := y = 8 * x - 3
def condition2 : Prop := y = 7 * x + 4

-- Theorem to prove
theorem correct_equation (h1 : condition1 x y) (h2 : condition2 x y) : (y + 3) / 8 = (y - 4) / 7 :=
sorry

end NUMINAMATH_GPT_correct_equation_l1417_141797


namespace NUMINAMATH_GPT_distinct_elements_triangle_not_isosceles_l1417_141725

theorem distinct_elements_triangle_not_isosceles
  {a b c : ℝ} (h1 : a ≠ b) (h2 : b ≠ c) (h3 : a ≠ c)
  (h4 : a + b > c) (h5 : a + c > b) (h6 : b + c > a) :
  ¬(a = b ∨ b = c ∨ a = c) := by
  sorry

end NUMINAMATH_GPT_distinct_elements_triangle_not_isosceles_l1417_141725


namespace NUMINAMATH_GPT_freda_flag_dimensions_l1417_141734

/--  
Given the area of the dove is 192 cm², and the perimeter of the dove consists of quarter-circles or straight lines,
prove that the dimensions of the flag are 24 cm by 16 cm.
-/
theorem freda_flag_dimensions (area_dove : ℝ) (h1 : area_dove = 192) : 
∃ (length width : ℝ), length = 24 ∧ width = 16 := 
sorry

end NUMINAMATH_GPT_freda_flag_dimensions_l1417_141734


namespace NUMINAMATH_GPT_least_value_MX_l1417_141789

-- Definitions of points and lines
variables (A B C D M P X : ℝ × ℝ)
variables (y : ℝ)

-- Hypotheses based on the conditions
variables (h1 : A = (0, 0))
variables (h2 : B = (33, 0))
variables (h3 : C = (33, 56))
variables (h4 : D = (0, 56))
variables (h5 : M = (33 / 2, 0)) -- M is midpoint of AB
variables (h6 : P = (33, y)) -- P is on BC
variables (hy_range : 0 ≤ y ∧ y ≤ 56) -- y is within the bounds of BC

-- Additional derived hypotheses needed for the proof
variables (h7 : ∃ x, X = (x, sqrt (816.75))) -- X is intersection point on DA

-- The theorem statement
theorem least_value_MX : ∃ y, 0 ≤ y ∧ y ≤ 56 ∧ MX = 33 :=
by
  use 28
  sorry

end NUMINAMATH_GPT_least_value_MX_l1417_141789


namespace NUMINAMATH_GPT_find_first_term_of_geometric_series_l1417_141726

theorem find_first_term_of_geometric_series 
  (r : ℚ) (S : ℚ) (a : ℚ) 
  (hr : r = -1/3) (hS : S = 9)
  (h_sum_formula : S = a / (1 - r)) : 
  a = 12 := 
by
  sorry

end NUMINAMATH_GPT_find_first_term_of_geometric_series_l1417_141726


namespace NUMINAMATH_GPT_mark_has_seven_butterfingers_l1417_141778

/-
  Mark has 12 candy bars in total between Mars bars, Snickers, and Butterfingers.
  He has 3 Snickers and 2 Mars bars.
  Prove that he has 7 Butterfingers.
-/

noncomputable def total_candy_bars : Nat := 12
noncomputable def snickers : Nat := 3
noncomputable def mars_bars : Nat := 2
noncomputable def butterfingers : Nat := total_candy_bars - (snickers + mars_bars)

theorem mark_has_seven_butterfingers : butterfingers = 7 := by
  sorry

end NUMINAMATH_GPT_mark_has_seven_butterfingers_l1417_141778


namespace NUMINAMATH_GPT_directrix_of_parabola_l1417_141751

theorem directrix_of_parabola (x y : ℝ) (h : y = x^2) : y = -1 / 4 :=
sorry

end NUMINAMATH_GPT_directrix_of_parabola_l1417_141751


namespace NUMINAMATH_GPT_evaluate_expression_l1417_141768

theorem evaluate_expression : 2 - 1 / (2 + 1 / (2 - 1 / 3)) = 21 / 13 := by
  sorry

end NUMINAMATH_GPT_evaluate_expression_l1417_141768


namespace NUMINAMATH_GPT_mark_cans_correct_l1417_141753

variable (R : ℕ) -- Rachel's cans
variable (J : ℕ) -- Jaydon's cans
variable (M : ℕ) -- Mark's cans
variable (T : ℕ) -- Total cans 

-- Conditions
def jaydon_cans (R : ℕ) : ℕ := 2 * R + 5
def mark_cans (J : ℕ) : ℕ := 4 * J
def total_cans (R : ℕ) (J : ℕ) (M : ℕ) : ℕ := R + J + M

theorem mark_cans_correct (R : ℕ) (J : ℕ) 
  (h1 : J = jaydon_cans R) 
  (h2 : M = mark_cans J) 
  (h3 : total_cans R J M = 135) : 
  M = 100 := 
sorry

end NUMINAMATH_GPT_mark_cans_correct_l1417_141753


namespace NUMINAMATH_GPT_parabola_vertex_and_point_l1417_141787

/-- The vertex form of the parabola is at (7, -6) and passes through the point (1,0).
    Verify that the equation parameters a, b, c satisfy a + b + c = -43 / 6. -/
theorem parabola_vertex_and_point (a b c : ℚ)
  (h_eq : ∀ y, (a * y^2 + b * y + c) = a * (y + 6)^2 + 7)
  (h_vertex : ∃ x y, x = a * y^2 + b * y + c ∧ y = -6 ∧ x = 7)
  (h_point : ∃ x y, x = a * y^2 + b * y + c ∧ x = 1 ∧ y = 0) :
  a + b + c = -43 / 6 :=
by
  sorry

end NUMINAMATH_GPT_parabola_vertex_and_point_l1417_141787


namespace NUMINAMATH_GPT_trigonometric_identity_proof_l1417_141796

theorem trigonometric_identity_proof
  (α : Real)
  (h1 : Real.sin (Real.pi + α) = -Real.sin α)
  (h2 : Real.cos (Real.pi + α) = -Real.cos α)
  (h3 : Real.cos (-α) = Real.cos α)
  (h4 : Real.sin α ^ 2 + Real.cos α ^ 2 = 1) :
  Real.sin (Real.pi + α) ^ 2 - Real.cos (Real.pi + α) * Real.cos (-α) + 1 = 2 := 
by
  sorry

end NUMINAMATH_GPT_trigonometric_identity_proof_l1417_141796


namespace NUMINAMATH_GPT_roots_equal_and_real_l1417_141769

theorem roots_equal_and_real:
  (∀ (x : ℝ), x ^ 2 - 3 * x * y + y ^ 2 + 2 * x - 9 * y + 1 = 0 → (y = 0 ∨ y = -24 / 5)) ∧
  (∀ (x : ℝ), x ^ 2 - 3 * x * y + y ^ 2 + 2 * x - 9 * y + 1 = 0 → (y ≥ 0 ∨ y ≤ -24 / 5)) :=
  by sorry

end NUMINAMATH_GPT_roots_equal_and_real_l1417_141769


namespace NUMINAMATH_GPT_bucket_weight_one_third_l1417_141720

theorem bucket_weight_one_third 
    (x y c b : ℝ) 
    (h1 : x + 3/4 * y = c)
    (h2 : x + 1/2 * y = b) :
    x + 1/3 * y = 5/3 * b - 2/3 * c :=
by
  sorry

end NUMINAMATH_GPT_bucket_weight_one_third_l1417_141720


namespace NUMINAMATH_GPT_kira_memory_space_is_140_l1417_141701

def kira_songs_memory_space 
  (n_m : ℕ) -- number of songs downloaded in the morning
  (n_d : ℕ) -- number of songs downloaded later that day
  (n_n : ℕ) -- number of songs downloaded at night
  (s : ℕ) -- size of each song in MB
  : ℕ := (n_m + n_d + n_n) * s

theorem kira_memory_space_is_140 :
  kira_songs_memory_space 10 15 3 5 = 140 := 
by
  sorry

end NUMINAMATH_GPT_kira_memory_space_is_140_l1417_141701


namespace NUMINAMATH_GPT_find_N_l1417_141735

theorem find_N :
  ∃ N : ℕ, N > 1 ∧
    (1743 % N = 2019 % N) ∧ (2019 % N = 3008 % N) ∧ N = 23 :=
by
  sorry

end NUMINAMATH_GPT_find_N_l1417_141735


namespace NUMINAMATH_GPT_cos_seven_pi_six_l1417_141786

theorem cos_seven_pi_six : (Real.cos (7 * Real.pi / 6) = - Real.sqrt 3 / 2) :=
sorry

end NUMINAMATH_GPT_cos_seven_pi_six_l1417_141786


namespace NUMINAMATH_GPT_solve_equation_l1417_141770

theorem solve_equation (x : ℝ) :
  (3 / x - (1 / x * 6 / x) = -2.5) ↔ (x = (-3 + Real.sqrt 69) / 5 ∨ x = (-3 - Real.sqrt 69) / 5) :=
by {
  sorry
}

end NUMINAMATH_GPT_solve_equation_l1417_141770


namespace NUMINAMATH_GPT_trigonometric_identity_l1417_141703

theorem trigonometric_identity (α : ℝ) (h : Real.tan α = 3) : 
    (2 * Real.sin α - Real.cos α) / (Real.sin α + 3 * Real.cos α) = 5 / 6 :=
by
  sorry

end NUMINAMATH_GPT_trigonometric_identity_l1417_141703


namespace NUMINAMATH_GPT_percentage_subtraction_l1417_141755

theorem percentage_subtraction (P : ℝ) : (700 - (P / 100 * 7000) = 700) → P = 0 :=
by
  sorry

end NUMINAMATH_GPT_percentage_subtraction_l1417_141755


namespace NUMINAMATH_GPT_find_d_given_n_eq_cda_div_a_minus_d_l1417_141749

theorem find_d_given_n_eq_cda_div_a_minus_d (a c d n : ℝ) (h : n = c * d * a / (a - d)) :
  d = n * a / (c * d + n) := 
by
  sorry

end NUMINAMATH_GPT_find_d_given_n_eq_cda_div_a_minus_d_l1417_141749


namespace NUMINAMATH_GPT_proof_U_eq_A_union_complement_B_l1417_141784

noncomputable def U : Set Nat := {1, 2, 3, 4, 5, 7}
noncomputable def A : Set Nat := {1, 3, 5, 7}
noncomputable def B : Set Nat := {3, 5}
noncomputable def complement_U_B := U \ B

theorem proof_U_eq_A_union_complement_B : U = A ∪ complement_U_B := by
  sorry

end NUMINAMATH_GPT_proof_U_eq_A_union_complement_B_l1417_141784


namespace NUMINAMATH_GPT_find_m_value_l1417_141707

theorem find_m_value
  (m : ℝ)
  (h1 : 10 - m > 0)
  (h2 : m - 2 > 0)
  (h3 : 2 * Real.sqrt (10 - m - (m - 2)) = 4) :
  m = 4 := by
sorry

end NUMINAMATH_GPT_find_m_value_l1417_141707


namespace NUMINAMATH_GPT_find_a_plus_b_l1417_141762

-- Conditions for the lines
def line_l0 (x y : ℝ) : Prop := x - y + 1 = 0
def line_l1 (a : ℝ) (x y : ℝ) : Prop := a * x - 2 * y + 1 = 0
def line_l2 (b : ℝ) (x y : ℝ) : Prop := x + b * y + 3 = 0

-- Perpendicularity condition for l1 to l0
def perpendicular (a : ℝ) : Prop := 1 * a + (-1) * (-2) = 0

-- Parallel condition for l2 to l0
def parallel (b : ℝ) : Prop := 1 * b = (-1) * 1

-- Prove the value of a + b given the conditions
theorem find_a_plus_b (a b : ℝ) 
  (h1 : perpendicular a)
  (h2 : parallel b) : a + b = -3 :=
sorry

end NUMINAMATH_GPT_find_a_plus_b_l1417_141762


namespace NUMINAMATH_GPT_winner_percentage_l1417_141742

theorem winner_percentage (W L V : ℕ) 
    (hW : W = 868) 
    (hDiff : W - L = 336)
    (hV : V = W + L) : 
    (W * 100 / V) = 62 := 
by 
    sorry

end NUMINAMATH_GPT_winner_percentage_l1417_141742


namespace NUMINAMATH_GPT_stock_yield_percentage_l1417_141737

noncomputable def FaceValue : ℝ := 100
noncomputable def AnnualYield : ℝ := 0.20 * FaceValue
noncomputable def MarketPrice : ℝ := 166.66666666666669
noncomputable def ExpectedYieldPercentage : ℝ := 12

theorem stock_yield_percentage :
  (AnnualYield / MarketPrice) * 100 = ExpectedYieldPercentage :=
by
  -- given conditions directly from the problem
  have h1 : FaceValue = 100 := rfl
  have h2 : AnnualYield = 0.20 * FaceValue := rfl
  have h3 : MarketPrice = 166.66666666666669 := rfl
  
  -- we are proving that the yield percentage is 12%
  sorry

end NUMINAMATH_GPT_stock_yield_percentage_l1417_141737


namespace NUMINAMATH_GPT_john_umbrella_in_car_l1417_141717

variable (UmbrellasInHouse : Nat)
variable (CostPerUmbrella : Nat)
variable (TotalAmountPaid : Nat)

theorem john_umbrella_in_car
  (h1 : UmbrellasInHouse = 2)
  (h2 : CostPerUmbrella = 8)
  (h3 : TotalAmountPaid = 24) :
  (TotalAmountPaid / CostPerUmbrella) - UmbrellasInHouse = 1 := by
  sorry

end NUMINAMATH_GPT_john_umbrella_in_car_l1417_141717


namespace NUMINAMATH_GPT_min_value_expr_sum_of_squares_inequality_l1417_141772

-- Given conditions
variables (a b : ℝ)
variables (ha : a > 0) (hb : b > 0) (hab : a + b = 2)

-- Problem (1): Prove minimum value of (2 / a + 8 / b) is 9
theorem min_value_expr : ∃ a b, a > 0 ∧ b > 0 ∧ a + b = 2 ∧ ((2 / a) + (8 / b) = 9) := sorry

-- Problem (2): Prove a^2 + b^2 ≥ 2
theorem sum_of_squares_inequality : a^2 + b^2 ≥ 2 :=
by { sorry }

end NUMINAMATH_GPT_min_value_expr_sum_of_squares_inequality_l1417_141772


namespace NUMINAMATH_GPT_probability_journalist_A_to_group_A_l1417_141712

open Nat

theorem probability_journalist_A_to_group_A :
  let group_A := 0
  let group_B := 1
  let group_C := 2
  let journalists := [0, 1, 2, 3]  -- four journalists

  -- total number of ways to distribute 4 journalists into 3 groups such that each group has at least one journalist
  let total_ways := 36

  -- number of ways to assign journalist 0 to group A specifically
  let favorable_ways := 12

  -- probability calculation
  ∃ (prob : ℚ), prob = favorable_ways / total_ways ∧ prob = 1 / 3 :=
sorry

end NUMINAMATH_GPT_probability_journalist_A_to_group_A_l1417_141712


namespace NUMINAMATH_GPT_Dave_ticket_count_l1417_141757

variable (T C total : ℕ)

theorem Dave_ticket_count
  (hT1 : T = 12)
  (hC1 : C = 7)
  (hT2 : T = C + 5) :
  total = T + C → total = 19 := by
  sorry

end NUMINAMATH_GPT_Dave_ticket_count_l1417_141757


namespace NUMINAMATH_GPT_cos_240_eq_neg_half_l1417_141722

theorem cos_240_eq_neg_half : Real.cos (4 * Real.pi / 3) = -1/2 :=
by
  sorry

end NUMINAMATH_GPT_cos_240_eq_neg_half_l1417_141722


namespace NUMINAMATH_GPT_triangle_area_of_tangent_circles_l1417_141776

/-- 
Given three circles with radii 1, 3, and 5, that are mutually externally tangent and all tangent to 
the same line, the area of the triangle determined by the points where each circle is tangent to the line 
is 6.
-/
theorem triangle_area_of_tangent_circles :
  let r1 := 1
  let r2 := 3
  let r3 := 5
  ∃ (A B C : ℝ × ℝ),
    A = (0, -(r1 : ℝ)) ∧ B = (0, -(r2 : ℝ)) ∧ C = (0, -(r3 : ℝ)) ∧
    (∃ (h : ℝ), ∃ (b : ℝ), h = 4 ∧ b = 3 ∧
    (1 / 2) * h * b = 6) := 
by
  sorry

end NUMINAMATH_GPT_triangle_area_of_tangent_circles_l1417_141776


namespace NUMINAMATH_GPT_percentage_of_people_with_diploma_l1417_141766

variable (P : Type) -- P is the type representing people in Country Z.

-- Given Conditions:
def no_diploma_job (population : ℝ) : ℝ := 0.18 * population
def people_with_job (population : ℝ) : ℝ := 0.40 * population
def diploma_no_job (population : ℝ) : ℝ := 0.25 * (0.60 * population)

-- To Prove:
theorem percentage_of_people_with_diploma (population : ℝ) :
  no_diploma_job population + (diploma_no_job population) + (people_with_job population - no_diploma_job population) = 0.37 * population := 
by
  sorry

end NUMINAMATH_GPT_percentage_of_people_with_diploma_l1417_141766


namespace NUMINAMATH_GPT_fireflies_win_l1417_141704

theorem fireflies_win 
  (initial_hornets : ℕ) (initial_fireflies : ℕ) 
  (hornets_scored : ℕ) (fireflies_scored : ℕ) 
  (three_point_baskets : ℕ) (two_point_baskets : ℕ)
  (h1 : initial_hornets = 86)
  (h2 : initial_fireflies = 74)
  (h3 : three_point_baskets = 7)
  (h4 : two_point_baskets = 2)
  (h5 : fireflies_scored = three_point_baskets * 3)
  (h6 : hornets_scored = two_point_baskets * 2)
  : initial_fireflies + fireflies_scored - (initial_hornets + hornets_scored) = 5 := 
sorry

end NUMINAMATH_GPT_fireflies_win_l1417_141704


namespace NUMINAMATH_GPT_sum_of_roots_l1417_141708

-- Define the polynomial equation
def poly (x : ℝ) := (3 * x + 4) * (x - 5) + (3 * x + 4) * (x - 7)

-- The theorem claiming the sum of the roots
theorem sum_of_roots : 
  (∀ x : ℝ, poly x = 0 → (x = -4/3 ∨ x = 6)) → 
  (∀ s : ℝ, s = -4 / 3 + 6) → s = 14 / 3 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_roots_l1417_141708


namespace NUMINAMATH_GPT_LeanProof_l1417_141782

noncomputable def ProblemStatement : Prop :=
  let AB_parallel_YZ := True -- given condition that AB is parallel to YZ
  let AZ := 36 
  let BQ := 15
  let QY := 20
  let similarity_ratio := BQ / QY = 3 / 4
  ∃ QZ : ℝ, AZ = (3 / 4) * QZ + QZ ∧ QZ = 144 / 7

theorem LeanProof : ProblemStatement :=
sorry

end NUMINAMATH_GPT_LeanProof_l1417_141782


namespace NUMINAMATH_GPT_variance_of_data_set_is_4_l1417_141783

/-- The data set for which we want to calculate the variance --/
def data_set : List ℝ := [2, 4, 5, 6, 8]

/-- The mean of the data set --/
noncomputable def mean (l : List ℝ) : ℝ :=
  (l.sum) / (l.length)

-- Calculation of the variance of a list given its mean
noncomputable def variance (l : List ℝ) (μ : ℝ) : ℝ :=
  (l.map (λ x => (x - μ) ^ 2)).sum / l.length

theorem variance_of_data_set_is_4 :
  variance data_set (mean data_set) = 4 :=
by
  sorry

end NUMINAMATH_GPT_variance_of_data_set_is_4_l1417_141783


namespace NUMINAMATH_GPT_no_perfect_square_l1417_141702

theorem no_perfect_square (p q : ℕ) (hp : Nat.Prime p) (hq : Nat.Prime q) 
  (h : ∃ (a : ℕ), p + q^2 = a^2) : ∀ (n : ℕ), n > 0 → ¬ (∃ (b : ℕ), p^2 + q^n = b^2) := 
by
  sorry

end NUMINAMATH_GPT_no_perfect_square_l1417_141702


namespace NUMINAMATH_GPT_triangle_is_isosceles_l1417_141780

theorem triangle_is_isosceles (α β γ δ ε : ℝ)
  (h1 : α + β + γ = 180) 
  (h2 : α + β = δ) 
  (h3 : β + γ = ε) : 
  α = γ ∨ β = γ ∨ α = β := 
sorry

end NUMINAMATH_GPT_triangle_is_isosceles_l1417_141780


namespace NUMINAMATH_GPT_sum_last_two_digits_l1417_141710

theorem sum_last_two_digits (h1 : 9 ^ 23 ≡ a [MOD 100]) (h2 : 11 ^ 23 ≡ b [MOD 100]) :
  (a + b) % 100 = 60 := 
  sorry

end NUMINAMATH_GPT_sum_last_two_digits_l1417_141710


namespace NUMINAMATH_GPT_number_of_nurses_l1417_141764

theorem number_of_nurses (total : ℕ) (ratio_d_to_n : ℕ → ℕ) (h1 : total = 250) (h2 : ratio_d_to_n 2 = 3) : ∃ n : ℕ, n = 150 := 
by
  sorry

end NUMINAMATH_GPT_number_of_nurses_l1417_141764


namespace NUMINAMATH_GPT_distance_between_lamps_l1417_141775

/-- 
A rectangular classroom measures 10 meters in length. Two lamps emitting conical light beams with a 90° opening angle 
are installed on the ceiling. The first lamp is located at the center of the ceiling and illuminates a circle on the 
floor with a diameter of 6 meters. The second lamp is adjusted such that the illuminated area along the length 
of the classroom spans a 10-meter section without reaching the opposite walls. Prove that the distance between the 
two lamps is 4 meters.
-/
theorem distance_between_lamps : 
  ∀ (length width height : ℝ) (center_illum_radius illum_length : ℝ) (d_center_to_lamp1 d_center_to_lamp2 dist_lamps : ℝ),
  length = 10 ∧ d_center_to_lamp1 = 3 ∧ d_center_to_lamp2 = 1 ∧ dist_lamps = 4 → d_center_to_lamp1 - d_center_to_lamp2 = dist_lamps :=
by
  intros length width height center_illum_radius illum_length d_center_to_lamp1 d_center_to_lamp2 dist_lamps conditions
  sorry

end NUMINAMATH_GPT_distance_between_lamps_l1417_141775


namespace NUMINAMATH_GPT_how_many_more_rolls_needed_l1417_141760

variable (total_needed sold_to_grandmother sold_to_uncle sold_to_neighbor : ℕ)

theorem how_many_more_rolls_needed (h1 : total_needed = 45)
  (h2 : sold_to_grandmother = 1)
  (h3 : sold_to_uncle = 10)
  (h4 : sold_to_neighbor = 6) :
  total_needed - (sold_to_grandmother + sold_to_uncle + sold_to_neighbor) = 28 := by
  sorry

end NUMINAMATH_GPT_how_many_more_rolls_needed_l1417_141760


namespace NUMINAMATH_GPT_smallest_prime_after_six_nonprimes_l1417_141731

-- Define the set of natural numbers and prime numbers
def is_natural (n : ℕ) : Prop := n ≥ 1
def is_prime (n : ℕ) : Prop := 1 < n ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n
def is_nonprime (n : ℕ) : Prop := ¬ is_prime n

-- The condition of six consecutive nonprime numbers
def six_consecutive_nonprime (n : ℕ) : Prop := 
  is_nonprime n ∧ 
  is_nonprime (n + 1) ∧ 
  is_nonprime (n + 2) ∧ 
  is_nonprime (n + 3) ∧ 
  is_nonprime (n + 4) ∧ 
  is_nonprime (n + 5)

-- The main theorem stating that 37 is the smallest prime following six consecutive nonprime numbers
theorem smallest_prime_after_six_nonprimes : 
  ∃ (n : ℕ), six_consecutive_nonprime n ∧ is_prime (n + 6) ∧ (∀ m, m < (n + 6) → ¬ is_prime m) :=
sorry

end NUMINAMATH_GPT_smallest_prime_after_six_nonprimes_l1417_141731


namespace NUMINAMATH_GPT_fallen_tree_trunk_length_l1417_141736

noncomputable def tiger_speed (tiger_length : ℕ) (time_pass_grass : ℕ) : ℕ := tiger_length / time_pass_grass

theorem fallen_tree_trunk_length
  (tiger_length : ℕ)
  (time_pass_grass : ℕ)
  (time_pass_tree : ℕ)
  (speed := tiger_speed tiger_length time_pass_grass) :
  tiger_length = 5 →
  time_pass_grass = 1 →
  time_pass_tree = 5 →
  (speed * time_pass_tree) = 25 :=
by
  intros h_tiger_length h_time_pass_grass h_time_pass_tree
  sorry

end NUMINAMATH_GPT_fallen_tree_trunk_length_l1417_141736


namespace NUMINAMATH_GPT_tom_sara_age_problem_l1417_141715

-- Define the given conditions as hypotheses and variables
variables (t s : ℝ)
variables (h1 : t - 3 = 2 * (s - 3))
variables (h2 : t - 8 = 3 * (s - 8))

-- Lean statement of the problem
theorem tom_sara_age_problem :
  ∃ x : ℝ, (t + x) / (s + x) = 3 / 2 ∧ x = 7 :=
by
  sorry

end NUMINAMATH_GPT_tom_sara_age_problem_l1417_141715


namespace NUMINAMATH_GPT_hyperbola_range_m_l1417_141798

theorem hyperbola_range_m (m : ℝ) : 
  (∃ x y : ℝ, (m - 2) ≠ 0 ∧ (m + 3) ≠ 0 ∧ (x^2 / (m - 2) + y^2 / (m + 3) = 1)) ↔ (-3 < m ∧ m < 2) :=
by
  sorry

end NUMINAMATH_GPT_hyperbola_range_m_l1417_141798


namespace NUMINAMATH_GPT_conference_handshakes_l1417_141785

theorem conference_handshakes (n : ℕ) (h : n = 10) :
  (n * (n - 1)) / 2 = 45 :=
by
  sorry

end NUMINAMATH_GPT_conference_handshakes_l1417_141785


namespace NUMINAMATH_GPT_determine_n_l1417_141759

theorem determine_n : ∃ n : ℤ, 0 ≤ n ∧ n < 8 ∧ -2222 % 8 = n := by
  use 2
  sorry

end NUMINAMATH_GPT_determine_n_l1417_141759


namespace NUMINAMATH_GPT_square_side_length_l1417_141788

theorem square_side_length (d : ℝ) (s : ℝ) (h : d = Real.sqrt 2) (h2 : d = Real.sqrt 2 * s) : s = 1 :=
by
  sorry

end NUMINAMATH_GPT_square_side_length_l1417_141788


namespace NUMINAMATH_GPT_relative_magnitude_of_reciprocal_l1417_141777

theorem relative_magnitude_of_reciprocal 
  (a b : ℝ) (hab : a < 1 / b) :
  (a > 0 ∧ b > 0 ∧ 1 / a > b) ∨ (a < 0 ∧ b < 0 ∧ 1 / a > b)
   ∨ (a > 0 ∧ b < 0 ∧ 1 / a < b) ∨ (a < 0 ∧ b > 0 ∧ 1 / a < b) :=
by sorry

end NUMINAMATH_GPT_relative_magnitude_of_reciprocal_l1417_141777


namespace NUMINAMATH_GPT_bank_account_balance_l1417_141741

theorem bank_account_balance : 
  ∀ (initial_amount withdraw_amount deposited_amount final_amount : ℕ),
  initial_amount = 230 →
  withdraw_amount = 60 →
  deposited_amount = 2 * withdraw_amount →
  final_amount = initial_amount - withdraw_amount + deposited_amount →
  final_amount = 290 :=
by
  intros
  sorry

end NUMINAMATH_GPT_bank_account_balance_l1417_141741


namespace NUMINAMATH_GPT_range_of_m_l1417_141743

noncomputable def f (x : ℝ) : ℝ := (1 / 2) * x^2 - ((Real.exp x - 1) / (Real.exp x + 1))

theorem range_of_m (m : ℝ) (h : f (4 - m) - f m ≥ 8 - 4 * m) : 2 ≤ m := by
  sorry

end NUMINAMATH_GPT_range_of_m_l1417_141743


namespace NUMINAMATH_GPT_find_perpendicular_line_through_intersection_l1417_141709

theorem find_perpendicular_line_through_intersection : 
  (∃ (M : ℚ × ℚ), 
    (M.1 - 2 * M.2 + 3 = 0) ∧ 
    (2 * M.1 + 3 * M.2 - 8 = 0) ∧ 
    (∃ (c : ℚ), M.1 + 3 * M.2 + c = 0 ∧ 3 * M.1 - M.2 + 1 = 0)) → 
  ∃ (c : ℚ), x + 3 * y + c = 0 :=
sorry

end NUMINAMATH_GPT_find_perpendicular_line_through_intersection_l1417_141709
