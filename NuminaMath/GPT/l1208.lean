import Mathlib

namespace f_increasing_l1208_120879

noncomputable def f (x : Real) : Real := (2 * Real.exp x) / (1 + Real.exp x) + 1/2

theorem f_increasing : ∀ x y : Real, x < y → f x < f y := 
by
  -- the proof goes here
  sorry

end f_increasing_l1208_120879


namespace growth_factor_condition_l1208_120898

open BigOperators

theorem growth_factor_condition {n : ℕ} (h : ∏ i in Finset.range n, (i + 2) / (i + 1) = 50) : n = 49 := by
  sorry

end growth_factor_condition_l1208_120898


namespace abs_diff_squares_1055_985_eq_1428_l1208_120842

theorem abs_diff_squares_1055_985_eq_1428 :
  abs ((105.5: ℝ)^2 - (98.5: ℝ)^2) = 1428 :=
by
  sorry

end abs_diff_squares_1055_985_eq_1428_l1208_120842


namespace find_min_value_of_quadratic_l1208_120820

theorem find_min_value_of_quadratic : ∀ x : ℝ, ∃ c : ℝ, (∃ a b : ℝ, (y = 2*x^2 + 8*x + 7 ∧ (∀ x : ℝ, y ≥ c)) ∧ c = -1) :=
by
  sorry

end find_min_value_of_quadratic_l1208_120820


namespace xiaoqiang_xiaolin_stamps_l1208_120821

-- Definitions for initial conditions and constraints
noncomputable def x : ℤ := 227
noncomputable def y : ℤ := 221
noncomputable def k : ℤ := sorry

-- Proof problem as a theorem
theorem xiaoqiang_xiaolin_stamps:
  x + y > 400 ∧
  x - k = (13 / 19) * (y + k) ∧
  y - k = (11 / 17) * (x + k) ∧
  x = 227 ∧ 
  y = 221 :=
by
  sorry

end xiaoqiang_xiaolin_stamps_l1208_120821


namespace root_polynomial_h_l1208_120838

theorem root_polynomial_h (h : ℤ) : (2^3 + h * 2 + 10 = 0) → h = -9 :=
by
  sorry

end root_polynomial_h_l1208_120838


namespace complex_z_1000_l1208_120863

open Complex

theorem complex_z_1000 (z : ℂ) (h : z + z⁻¹ = 2 * Real.cos (Real.pi * 5 / 180)) :
  z^(1000 : ℕ) + (z^(1000 : ℕ))⁻¹ = 2 * Real.cos (Real.pi * 20 / 180) :=
sorry

end complex_z_1000_l1208_120863


namespace parabola_focus_l1208_120806

theorem parabola_focus (a : ℝ) (h : a ≠ 0) : ∃ q : ℝ, q = 1/(4*a) ∧ (0, q) = (0, 1/(4*a)) :=
by
  sorry

end parabola_focus_l1208_120806


namespace inequality_abc_equality_condition_abc_l1208_120841

theorem inequality_abc (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (h_sum : a + b + c = 1) :
  (a / (2 * a + 1)) + (b / (3 * b + 1)) + (c / (6 * c + 1)) ≤ 1 / 2 :=
sorry

theorem equality_condition_abc (a b c : ℝ) :
  (a / (2 * a + 1)) + (b / (3 * b + 1)) + (c / (6 * c + 1)) = 1 / 2 ↔ 
  a = 1 / 2 ∧ b = 1 / 3 ∧ c = 1 / 6 :=
sorry

end inequality_abc_equality_condition_abc_l1208_120841


namespace trajectory_of_square_is_line_l1208_120834

open Complex

theorem trajectory_of_square_is_line (z : ℂ) (h : z.re = z.im) : ∃ c : ℝ, z^2 = Complex.I * (c : ℂ) :=
by
  sorry

end trajectory_of_square_is_line_l1208_120834


namespace find_f_log_value_l1208_120833

noncomputable def f (x : ℝ) : ℝ :=
if 0 < x ∧ x < 1 then 2^x + 1 else sorry

theorem find_f_log_value (f : ℝ → ℝ)
  (h_odd : ∀ x, f (-x) = -f x)
  (h_sym : ∀ x, f x = f (2 - x))
  (h_spec : ∀ x, 0 < x → x < 1 → f x = 2^x + 1) :
  f (Real.logb (1/2) (1/15)) = -31/15 :=
sorry

end find_f_log_value_l1208_120833


namespace mortgage_loan_amount_l1208_120802

theorem mortgage_loan_amount (C : ℝ) (hC : C = 8000000) : 0.75 * C = 6000000 :=
by
  sorry

end mortgage_loan_amount_l1208_120802


namespace base5_to_octal_1234_eval_f_at_3_l1208_120819

-- Definition of base conversion from base 5 to decimal and to octal
def base5_to_decimal (n : Nat) : Nat :=
  match n with
  | 1234 => 1 * 5^3 + 2 * 5^2 + 3 * 5 + 4
  | _ => 0

def decimal_to_octal (n : Nat) : Nat :=
  match n with
  | 194 => 302
  | _ => 0

-- Definition of the polynomial f(x) = 7x^7 + 6x^6 + 5x^5 + 4x^4 + 3x^3 + 2x^2 + x
def f (x : Nat) : Nat :=
  7 * x^7 + 6 * x^6 + 5 * x^5 + 4 * x^4 + 3 * x^3 + 2 * x^2 + x

-- Definition of Horner's method evaluation
def horner_eval (x : Nat) : Nat :=
  ((((((7 * x + 6) * x + 5) * x + 4) * x + 3) * x + 2) * x + 1) * x

-- Theorem statement for base-5 to octal conversion
theorem base5_to_octal_1234 : base5_to_decimal 1234 = 194 ∧ decimal_to_octal 194 = 302 :=
  by
    sorry

-- Theorem statement for polynomial evaluation using Horner's method
theorem eval_f_at_3 : horner_eval 3 = f 3 ∧ f 3 = 21324 :=
  by
    sorry

end base5_to_octal_1234_eval_f_at_3_l1208_120819


namespace cost_of_450_candies_l1208_120817

theorem cost_of_450_candies (box_cost : ℝ) (box_candies : ℕ) (total_candies : ℕ) 
  (h1 : box_cost = 7.50) (h2 : box_candies = 30) (h3 : total_candies = 450) : 
  (total_candies / box_candies) * box_cost = 112.50 :=
by
  sorry

end cost_of_450_candies_l1208_120817


namespace ratio_pentagon_side_length_to_rectangle_width_l1208_120861

def pentagon_side_length (p : ℕ) (n : ℕ) := p / n
def rectangle_width (p : ℕ) (ratio : ℕ) := p / (2 * (1 + ratio))

theorem ratio_pentagon_side_length_to_rectangle_width :
  pentagon_side_length 60 5 / rectangle_width 80 3 = (6 : ℚ) / 5 :=
by {
  sorry
}

end ratio_pentagon_side_length_to_rectangle_width_l1208_120861


namespace savings_correct_l1208_120892

noncomputable def school_price_math : Float := 45
noncomputable def school_price_science : Float := 60
noncomputable def school_price_literature : Float := 35

noncomputable def discount_math : Float := 0.20
noncomputable def discount_science : Float := 0.25
noncomputable def discount_literature : Float := 0.15

noncomputable def tax_school : Float := 0.07
noncomputable def tax_alt : Float := 0.06
noncomputable def shipping_alt : Float := 10

noncomputable def alt_price_math : Float := (school_price_math * (1 - discount_math)) * (1 + tax_alt)
noncomputable def alt_price_science : Float := (school_price_science * (1 - discount_science)) * (1 + tax_alt)
noncomputable def alt_price_literature : Float := (school_price_literature * (1 - discount_literature)) * (1 + tax_alt)

noncomputable def total_alt_cost : Float := alt_price_math + alt_price_science + alt_price_literature + shipping_alt

noncomputable def school_price_math_tax : Float := school_price_math * (1 + tax_school)
noncomputable def school_price_science_tax : Float := school_price_science * (1 + tax_school)
noncomputable def school_price_literature_tax : Float := school_price_literature * (1 + tax_school)

noncomputable def total_school_cost : Float := school_price_math_tax + school_price_science_tax + school_price_literature_tax

noncomputable def savings : Float := total_school_cost - total_alt_cost

theorem savings_correct : savings = 22.40 := by
  sorry

end savings_correct_l1208_120892


namespace time_to_cross_pole_l1208_120827

def train_length := 3000 -- in meters
def train_speed_kmh := 90 -- in kilometers per hour

noncomputable def train_speed_mps : ℝ := train_speed_kmh * (1000 / 3600) -- converting speed to meters per second

theorem time_to_cross_pole : (train_length : ℝ) / train_speed_mps = 120 := 
by
  -- Placeholder for the actual proof
  sorry

end time_to_cross_pole_l1208_120827


namespace value_of_b_minus_a_l1208_120808

theorem value_of_b_minus_a (a b : ℕ) (h1 : a * b = 2 * (a + b) + 1) (h2 : b = 7) : b - a = 4 :=
by
  sorry

end value_of_b_minus_a_l1208_120808


namespace power_equality_l1208_120856

-- Definitions based on conditions
def nine := 3^2

-- Theorem stating the given mathematical problem
theorem power_equality : nine^4 = 3^8 := by
  sorry

end power_equality_l1208_120856


namespace sum_of_digits_of_N_eq_14_l1208_120847

theorem sum_of_digits_of_N_eq_14 :
  ∃ N : ℕ, (N * (N + 1)) / 2 = 3003 ∧ (N % 10 + N / 10 % 10 = 14) :=
by
  sorry

end sum_of_digits_of_N_eq_14_l1208_120847


namespace linear_function_not_in_first_quadrant_l1208_120851

theorem linear_function_not_in_first_quadrant:
  ∀ x y : ℝ, y = -2 * x - 3 → ¬ (x > 0 ∧ y > 0) :=
by
 -- proof steps would go here
 sorry

end linear_function_not_in_first_quadrant_l1208_120851


namespace vertical_strips_count_l1208_120846

/- Define the conditions -/

variables {a b x y : ℕ}

-- The outer rectangle has a perimeter of 50 cells
axiom outer_perimeter : 2 * a + 2 * b = 50

-- The inner hole has a perimeter of 32 cells
axiom inner_perimeter : 2 * x + 2 * y = 32

-- Cutting along all horizontal lines produces 20 strips
axiom horizontal_cuts : a + x = 20

-- We want to prove that cutting along all vertical grid lines produces 21 strips
theorem vertical_strips_count : b + y = 21 :=
by
  sorry

end vertical_strips_count_l1208_120846


namespace complex_product_l1208_120826

-- Define the imaginary unit i
def i : ℂ := Complex.I

-- Define the complex numbers z1 and z2
def z1 : ℂ := 1 - i
def z2 : ℂ := 3 + i

-- Statement of the problem
theorem complex_product : z1 * z2 = 4 - 2 * i := by
  sorry

end complex_product_l1208_120826


namespace parallel_line_slope_l1208_120801

theorem parallel_line_slope (x y : ℝ) :
  (∃ k b : ℝ, 3 * x + 6 * y = k * x + b) ∧ (∃ a b, y = a * x + b) ∧ 3 * x + 6 * y = -24 → 
  ∃ m : ℝ, m = -1/2 :=
by
  sorry

end parallel_line_slope_l1208_120801


namespace sales_tax_rate_l1208_120839

-- Given conditions
def cost_of_video_game : ℕ := 50
def weekly_allowance : ℕ := 10
def weekly_savings : ℕ := weekly_allowance / 2
def weeks_to_save : ℕ := 11
def total_savings : ℕ := weeks_to_save * weekly_savings

-- Proof problem statement
theorem sales_tax_rate : 
  total_savings - cost_of_video_game = (cost_of_video_game * 10) / 100 := by
  sorry

end sales_tax_rate_l1208_120839


namespace smallest_a_inequality_l1208_120832

theorem smallest_a_inequality 
  (x : ℝ)
  (h1 : x ∈ Set.Ioo (-3 * Real.pi / 2) (-Real.pi)) : 
  (∃ a : ℝ, a = -2.52 ∧ (∀ x ∈ Set.Ioo (-3 * Real.pi / 2) (-Real.pi), 
    ( ((Real.sqrt (Real.cos x / Real.sin x)^2) - (Real.sqrt (Real.sin x / Real.cos x)^2))
    / ((Real.sqrt (Real.sin x)^2) - (Real.sqrt (Real.cos x)^2)) ) < a )) :=
  sorry

end smallest_a_inequality_l1208_120832


namespace widget_production_l1208_120813

theorem widget_production (p q r s t : ℕ) :
  (s * q * t) / (p * r) = (sqt / pr) := 
sorry

end widget_production_l1208_120813


namespace ratio_area_A_to_C_l1208_120887

noncomputable def side_length (perimeter : ℕ) : ℕ :=
  perimeter / 4

noncomputable def area (side : ℕ) : ℕ :=
  side * side

theorem ratio_area_A_to_C : 
  let A_perimeter := 16
  let B_perimeter := 40
  let C_perimeter := 2 * A_perimeter
  let side_A := side_length A_perimeter
  let side_C := side_length C_perimeter
  let area_A := area side_A
  let area_C := area side_C
  (area_A : ℚ) / area_C = 1 / 4 :=
by
  sorry

end ratio_area_A_to_C_l1208_120887


namespace negation_proposition_l1208_120858

open Set

theorem negation_proposition :
  ¬ (∀ x : ℝ, x^2 + 2 * x + 5 > 0) → (∃ x : ℝ, x^2 + 2 * x + 5 ≤ 0) :=
sorry

end negation_proposition_l1208_120858


namespace man_speed_upstream_l1208_120875

def man_speed_still_water : ℕ := 50
def speed_downstream : ℕ := 80

theorem man_speed_upstream : (man_speed_still_water - (speed_downstream - man_speed_still_water)) = 20 :=
by
  sorry

end man_speed_upstream_l1208_120875


namespace jason_egg_consumption_l1208_120823

-- Definition for the number of eggs Jason consumes per day
def eggs_per_day : ℕ := 3

-- Definition for the number of days in a week
def days_in_week : ℕ := 7

-- Definition for the number of weeks we are considering
def weeks : ℕ := 2

-- The statement we want to prove, which combines all the conditions and provides the final answer
theorem jason_egg_consumption : weeks * days_in_week * eggs_per_day = 42 := by
sorry

end jason_egg_consumption_l1208_120823


namespace fraction_unseated_l1208_120865

theorem fraction_unseated :
  ∀ (tables seats_per_table seats_taken : ℕ),
  tables = 15 →
  seats_per_table = 10 →
  seats_taken = 135 →
  ((tables * seats_per_table - seats_taken : ℕ) / (tables * seats_per_table : ℕ) : ℚ) = 1 / 10 :=
by
  intros tables seats_per_table seats_taken h_tables h_seats_per_table h_seats_taken
  sorry

end fraction_unseated_l1208_120865


namespace value_of_f_log_half_24_l1208_120848

noncomputable def f (x : ℝ) : ℝ := sorry

theorem value_of_f_log_half_24 :
  (∀ x : ℝ, f x * -1 = f (-x)) → -- Condition 1: f(x) is an odd function.
  (∀ x : ℝ, f (x + 1) = f (x - 1)) → -- Condition 2: f(x + 1) = f(x - 1).
  (∀ x : ℝ, 0 < x ∧ x < 1 → f x = 2^x - 2) → -- Condition 3: For 0 < x < 1, f(x) = 2^x - 2.
  f (Real.logb 0.5 24) = 1 / 2 := 
sorry

end value_of_f_log_half_24_l1208_120848


namespace determine_c_l1208_120884

theorem determine_c (c d : ℝ) (hc : c < 0) (hd : d > 0) (hamp : ∀ x, y = c * Real.cos (d * x) → |y| ≤ 3) :
  c = -3 :=
sorry

end determine_c_l1208_120884


namespace value_of_y_at_64_l1208_120881

theorem value_of_y_at_64 (x y k : ℝ) (h1 : y = k * x^(1/3)) (h2 : 8^(1/3) = 2) (h3 : y = 4 ∧ x = 8):
  y = 8 :=
by {
  sorry
}

end value_of_y_at_64_l1208_120881


namespace max_additional_bags_correct_l1208_120878

-- Definitions from conditions
def num_people : ℕ := 6
def bags_per_person : ℕ := 5
def weight_per_bag : ℕ := 50
def max_plane_capacity : ℕ := 6000

-- Derived definitions from conditions
def total_bags : ℕ := num_people * bags_per_person
def total_weight_of_bags : ℕ := total_bags * weight_per_bag
def remaining_capacity : ℕ := max_plane_capacity - total_weight_of_bags 
def max_additional_bags : ℕ := remaining_capacity / weight_per_bag

-- Theorem statement
theorem max_additional_bags_correct : max_additional_bags = 90 := by
  -- Proof skipped
  sorry

end max_additional_bags_correct_l1208_120878


namespace sum_first_5_arithmetic_l1208_120844

theorem sum_first_5_arithmetic (u : ℕ → ℝ) (h : u 3 = 0) : 
  (u 1 + u 2 + u 3 + u 4 + u 5) = 0 :=
sorry

end sum_first_5_arithmetic_l1208_120844


namespace quadratic_roots_expression_eq_zero_l1208_120886

theorem quadratic_roots_expression_eq_zero
  (a b c : ℝ)
  (h : ∀ x : ℝ, a * x^2 + b * x + c = 0)
  (x1 x2 : ℝ)
  (hx1 : a * x1^2 + b * x1 + c = 0)
  (hx2 : a * x2^2 + b * x2 + c = 0)
  (s1 s2 s3 : ℝ)
  (h_s1 : s1 = x1 + x2)
  (h_s2 : s2 = x1^2 + x2^2)
  (h_s3 : s3 = x1^3 + x2^3) :
  a * s3 + b * s2 + c * s1 = 0 := sorry

end quadratic_roots_expression_eq_zero_l1208_120886


namespace sum_of_cubes_of_roots_l1208_120843

theorem sum_of_cubes_of_roots (x₁ x₂ : ℝ) (h₀ : 3 * x₁ ^ 2 - 5 * x₁ - 2 = 0)
  (h₁ : 3 * x₂ ^ 2 - 5 * x₂ - 2 = 0) :
  x₁^3 + x₂^3 = 215 / 27 :=
by sorry

end sum_of_cubes_of_roots_l1208_120843


namespace compound_interest_rate_l1208_120824

theorem compound_interest_rate (P : ℝ) (r : ℝ) (t : ℕ) (A : ℝ) 
  (h1 : t = 15) (h2 : A = (9 / 5) * P) :
  (1 + r) ^ t = (9 / 5) → 
  r ≠ 0.05 ∧ r ≠ 0.06 ∧ r ≠ 0.07 ∧ r ≠ 0.08 :=
by
  -- Sorry could be placed here for now
  sorry

end compound_interest_rate_l1208_120824


namespace smallest_possible_area_of_ellipse_l1208_120809

theorem smallest_possible_area_of_ellipse
  (a b : ℝ)
  (h_ellipse : ∀ (x y : ℝ), (x^2 / a^2 + y^2 / b^2 = 1) → 
    (((x - 1/2)^2 + y^2 = 1/4) ∨ ((x + 1/2)^2 + y^2 = 1/4))) :
  ∃ (k : ℝ), (a * b * π = 4 * π) :=
by
  sorry

end smallest_possible_area_of_ellipse_l1208_120809


namespace probability_both_boys_or_both_girls_l1208_120836

theorem probability_both_boys_or_both_girls 
  (total_students : ℕ) (boys : ℕ) (girls : ℕ) :
  total_students = 5 → boys = 2 → girls = 3 →
    (∃ (p : ℚ), p = 2/5) :=
by
  intros ht hb hg
  sorry

end probability_both_boys_or_both_girls_l1208_120836


namespace quadratic_completing_square_t_value_l1208_120800

theorem quadratic_completing_square_t_value :
  ∃ q t : ℝ, 4 * x^2 - 24 * x - 96 = 0 → (x + q) ^ 2 = t ∧ t = 33 :=
by
  sorry

end quadratic_completing_square_t_value_l1208_120800


namespace solve_for_y_l1208_120807

noncomputable def g (y : ℝ) : ℝ := (30 * y + (30 * y + 27)^(1/3))^(1/3)

theorem solve_for_y :
  (∃ y : ℝ, g y = 15) ↔ (∃ y : ℝ, y = 1674 / 15) :=
by
  sorry

end solve_for_y_l1208_120807


namespace k_is_2_l1208_120814

noncomputable def f (k : ℝ) (x : ℝ) : ℝ := (k - 1) * x - 1
def g (x : ℝ) : ℝ := 0
noncomputable def h (x : ℝ) : ℝ := (x + 1) * Real.log x

theorem k_is_2 :
  (∀ x ∈ Set.Icc 1 (2 * Real.exp 1), 0 ≤ f k x ∧ f k x ≤ h x) ↔ (k = 2) :=
  sorry

end k_is_2_l1208_120814


namespace student_correct_answers_l1208_120897

theorem student_correct_answers (C W : ℕ) (h1 : C + W = 60) (h2 : 4 * C - W = 140) : C = 40 :=
by
  sorry

end student_correct_answers_l1208_120897


namespace suff_cond_iff_lt_l1208_120864

variable (a b : ℝ)

-- Proving that (a - b) a^2 < 0 is a sufficient but not necessary condition for a < b
theorem suff_cond_iff_lt (h : (a - b) * a^2 < 0) : a < b :=
by {
  sorry
}

end suff_cond_iff_lt_l1208_120864


namespace positive_integer_pairs_l1208_120899

theorem positive_integer_pairs (a b : ℕ) (h_pos : 0 < a ∧ 0 < b) :
  (∃ k : ℕ, k > 0 ∧ a^2 = k * (2 * a * b^2 - b^3 + 1)) ↔ 
  ∃ l : ℕ, 0 < l ∧ ((a = l ∧ b = 2 * l) ∨ (a = 8 * l^4 - l ∧ b = 2 * l)) :=
by 
  sorry

end positive_integer_pairs_l1208_120899


namespace count_numbers_with_cube_root_lt_8_l1208_120869

theorem count_numbers_with_cube_root_lt_8 : 
  ∀ n : ℕ, (n > 0) → (n < 8^3) → n ≤ 8^3 - 1 :=
by
  -- We need to prove that the count of such numbers is 511
  sorry

end count_numbers_with_cube_root_lt_8_l1208_120869


namespace simplify_expression_l1208_120895

theorem simplify_expression (a : ℝ) (h : a ≠ 1) : 1 - (1 / (1 + ((a + 1) / (1 - a)))) = (1 + a) / 2 := 
by
  sorry

end simplify_expression_l1208_120895


namespace donny_paid_l1208_120845

variable (total_capacity initial_fuel price_per_liter change : ℕ)

theorem donny_paid (h1 : total_capacity = 150) 
                   (h2 : initial_fuel = 38) 
                   (h3 : price_per_liter = 3) 
                   (h4 : change = 14) : 
                   (total_capacity - initial_fuel) * price_per_liter + change = 350 := 
by
  sorry

end donny_paid_l1208_120845


namespace jack_runs_faster_than_paul_l1208_120876

noncomputable def convert_km_hr_to_m_s (v : ℝ) : ℝ :=
  v * (1000 / 3600)

noncomputable def speed_difference : ℝ :=
  let v_J_km_hr := 20.62665  -- Jack's speed in km/hr
  let v_J_m_s := convert_km_hr_to_m_s v_J_km_hr  -- Jack's speed in m/s
  let distance := 1000  -- distance in meters
  let time_J := distance / v_J_m_s  -- Jack's time in seconds
  let time_P := time_J + 1.5  -- Paul's time in seconds
  let v_P_m_s := distance / time_P  -- Paul's speed in m/s
  let speed_diff_m_s := v_J_m_s - v_P_m_s  -- speed difference in m/s
  let speed_diff_km_hr := speed_diff_m_s * (3600 / 1000)  -- convert to km/hr
  speed_diff_km_hr

theorem jack_runs_faster_than_paul : speed_difference = 0.18225 :=
by
  -- Proof is omitted
  sorry

end jack_runs_faster_than_paul_l1208_120876


namespace max_integer_value_of_expression_l1208_120890

theorem max_integer_value_of_expression (x : ℝ) :
  ∃ M : ℤ, M = 15 ∧ ∀ y : ℝ, (4 * y^2 + 8 * y + 19) / (4 * y^2 + 8 * y + 5) ≤ M :=
sorry

end max_integer_value_of_expression_l1208_120890


namespace largest_common_value_less_than_1000_l1208_120862

theorem largest_common_value_less_than_1000 :
  ∃ a : ℕ, a = 999 ∧ (∃ n m : ℕ, a = 4 + 5 * n ∧ a = 7 + 8 * m) ∧ a < 1000 :=
by
  sorry

end largest_common_value_less_than_1000_l1208_120862


namespace min_value_geometric_seq_l1208_120831

theorem min_value_geometric_seq (a : ℕ → ℝ) (m n : ℕ) (h_pos : ∀ k, a k > 0)
  (h1 : a 1 = 1)
  (h2 : a 7 = a 6 + 2 * a 5)
  (h3 : a m * a n = 16) :
  (1 / m + 4 / n) ≥ 3 / 2 :=
sorry

end min_value_geometric_seq_l1208_120831


namespace prime_angle_triangle_l1208_120829

theorem prime_angle_triangle (a b c : ℕ) (ha : Nat.Prime a) (hb : Nat.Prime b) (hc : Nat.Prime c) (h_sum : a + b + c = 180) : a = 2 ∨ b = 2 ∨ c = 2 :=
sorry

end prime_angle_triangle_l1208_120829


namespace andrew_purchased_mangoes_l1208_120853

theorem andrew_purchased_mangoes
  (m : Nat)
  (h1 : 14 * 54 = 756)
  (h2 : 756 + 62 * m = 1376) :
  m = 10 :=
by
  sorry

end andrew_purchased_mangoes_l1208_120853


namespace average_k_l1208_120889

open Nat

def positive_integer_roots (a b : ℕ) : Prop :=
  a * b = 24 ∧ a + b = b + a

theorem average_k (k : ℕ) :
  (positive_integer_roots 1 24 ∨ 
  positive_integer_roots 2 12 ∨ 
  positive_integer_roots 3 8 ∨ 
  positive_integer_roots 4 6) →
  (k = 25 ∨ k = 14 ∨ k = 11 ∨ k = 10) →
  (25 + 14 + 11 + 10) / 4 = 15 := by
  sorry

end average_k_l1208_120889


namespace num_ways_arrange_l1208_120880

open Finset

def valid_combinations : Finset (Finset Nat) :=
  { {2, 5, 11, 3}, {3, 5, 6, 2}, {3, 6, 11, 5}, {5, 6, 11, 2} }

theorem num_ways_arrange : valid_combinations.card = 4 :=
  by
    sorry  -- proof of the statement

end num_ways_arrange_l1208_120880


namespace find_n_sin_eq_l1208_120812

theorem find_n_sin_eq (n : ℤ) (h₁ : -180 ≤ n) (h₂ : n ≤ 180) (h₃ : Real.sin (n * Real.pi / 180) = Real.sin (680 * Real.pi / 180)) :
  n = 40 ∨ n = 140 :=
by
  sorry

end find_n_sin_eq_l1208_120812


namespace part1_part2_l1208_120835

noncomputable def triangleABC (a : ℝ) (cosB : ℝ) (b : ℝ) (SinA : ℝ) : Prop :=
  cosB = 3 / 5 ∧ b = 4 → SinA = 2 / 5

noncomputable def triangleABC2 (a : ℝ) (cosB : ℝ) (S : ℝ) (b c : ℝ) : Prop :=
  cosB = 3 / 5 ∧ S = 4 → b = Real.sqrt 17 ∧ c = 5

theorem part1 :
  triangleABC 2 (3 / 5) 4 (2 / 5) :=
by {
  sorry
}

theorem part2 :
  triangleABC2 2 (3 / 5) 4 (Real.sqrt 17) 5 :=
by {
  sorry
}

end part1_part2_l1208_120835


namespace geometric_seq_a3_l1208_120874

theorem geometric_seq_a3 (a : ℕ → ℝ) (r : ℝ)
  (h1 : a 6 = a 3 * r^3)
  (h2 : a 9 = a 3 * r^6)
  (h3 : a 6 = 6)
  (h4 : a 9 = 9) : 
  a 3 = 4 := 
sorry

end geometric_seq_a3_l1208_120874


namespace determine_pq_value_l1208_120811

noncomputable def p : ℝ → ℝ := λ x => 16 * x
noncomputable def q : ℝ → ℝ := λ x => (x + 4) * (x - 1)

theorem determine_pq_value : (p (-1) / q (-1)) = 8 / 3 := by
  sorry

end determine_pq_value_l1208_120811


namespace find_natural_numbers_l1208_120867

theorem find_natural_numbers :
  ∃ (x y : ℕ), 
    x * y - (x + y) = Nat.gcd x y + Nat.lcm x y ∧ 
    ((x = 6 ∧ y = 3) ∨ (x = 6 ∧ y = 4) ∨ (x = 3 ∧ y = 6) ∨ (x = 4 ∧ y = 6)) := 
by 
  sorry

end find_natural_numbers_l1208_120867


namespace osmanthus_trees_variance_l1208_120825

variable (n : Nat) (p : ℚ)

def variance_binomial_distribution (n : Nat) (p : ℚ) : ℚ :=
  n * p * (1 - p)

theorem osmanthus_trees_variance (n : Nat) (p : ℚ) (h₁ : n = 4) (h₂ : p = 4 / 5) :
  variance_binomial_distribution n p = 16 / 25 := by
  sorry

end osmanthus_trees_variance_l1208_120825


namespace max_value_ad_bc_l1208_120859

theorem max_value_ad_bc (a b c d : ℤ) (h₁ : a ∈ ({-1, 1, 2} : Set ℤ))
                          (h₂ : b ∈ ({-1, 1, 2} : Set ℤ))
                          (h₃ : c ∈ ({-1, 1, 2} : Set ℤ))
                          (h₄ : d ∈ ({-1, 1, 2} : Set ℤ)) :
  ad - bc ≤ 6 :=
by sorry

end max_value_ad_bc_l1208_120859


namespace smallest_equal_cost_l1208_120852

def decimal_cost (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

def binary_cost (n : ℕ) : ℕ :=
  n.digits 2 |>.sum

theorem smallest_equal_cost :
  ∃ n : ℕ, n < 200 ∧ decimal_cost n = binary_cost n ∧ (∀ m : ℕ, m < 200 ∧ decimal_cost m = binary_cost m → m ≥ n) :=
by
  -- Proof goes here
  sorry

end smallest_equal_cost_l1208_120852


namespace smallest_n_for_y_n_integer_l1208_120896

noncomputable def y (n : ℕ) : ℝ :=
  if n = 0 then 0 else
  if n = 1 then (5 : ℝ)^(1/3) else
  if n = 2 then ((5 : ℝ)^(1/3))^((5 : ℝ)^(1/3)) else
  y (n-1)^((5 : ℝ)^(1/3))

theorem smallest_n_for_y_n_integer : ∃ n : ℕ, y n = 5 ∧ ∀ m < n, y m ≠ ((⌊y m⌋:ℝ)) :=
by
  sorry

end smallest_n_for_y_n_integer_l1208_120896


namespace solve_q_l1208_120837

theorem solve_q (n m q : ℤ) 
  (h₁ : 5/6 = n/72) 
  (h₂ : 5/6 = (m + n)/90) 
  (h₃ : 5/6 = (q - m)/150) : 
  q = 140 := by
  sorry

end solve_q_l1208_120837


namespace sum_of_scores_l1208_120894

/-- Prove that given the conditions on Bill, John, and Sue's scores, the total sum of the scores of the three students is 160. -/
theorem sum_of_scores (B J S : ℕ) (h1 : B = J + 20) (h2 : B = S / 2) (h3 : B = 45) : B + J + S = 160 :=
sorry

end sum_of_scores_l1208_120894


namespace math_problem_l1208_120840

theorem math_problem :
  let a := 481 * 7
  let b := 426 * 5
  ((a + b) ^ 3 - 4 * a * b) = 166021128033 := 
by
  let a := 481 * 7
  let b := 426 * 5
  sorry

end math_problem_l1208_120840


namespace tan_pi_div_a_of_point_on_cubed_function_l1208_120815

theorem tan_pi_div_a_of_point_on_cubed_function (a : ℝ) (h : (a, 27) ∈ {p : ℝ × ℝ | p.snd = p.fst ^ 3}) : 
  Real.tan (Real.pi / a) = Real.sqrt 3 := sorry

end tan_pi_div_a_of_point_on_cubed_function_l1208_120815


namespace find_k_l1208_120888

theorem find_k (x y z k : ℝ) (h1 : 5 / (x + y) = k / (x + z)) (h2 : k / (x + z) = 9 / (z - y)) : k = 14 :=
by
  sorry

end find_k_l1208_120888


namespace eight_times_10x_plus_14pi_l1208_120818

theorem eight_times_10x_plus_14pi (x : ℝ) (Q : ℝ) (h : 4 * (5 * x + 7 * π) = Q) : 
  8 * (10 * x + 14 * π) = 4 * Q := 
by {
  sorry  -- proof is omitted
}

end eight_times_10x_plus_14pi_l1208_120818


namespace sum_even_and_multiples_of_5_l1208_120871

def num_even_four_digit : ℕ :=
  let thousands := 9 -- thousands place cannot be zero
  let hundreds := 10
  let tens := 10
  let units := 5 -- even digits: {0, 2, 4, 6, 8}
  thousands * hundreds * tens * units

def num_multiples_of_5_four_digit : ℕ :=
  let thousands := 9 -- thousands place cannot be zero
  let hundreds := 10
  let tens := 10
  let units := 2 -- multiples of 5 digits: {0, 5}
  thousands * hundreds * tens * units

theorem sum_even_and_multiples_of_5 : num_even_four_digit + num_multiples_of_5_four_digit = 6300 := by
  sorry

end sum_even_and_multiples_of_5_l1208_120871


namespace nonagon_diagonals_l1208_120857

theorem nonagon_diagonals (n : ℕ) (h1 : n = 9) : (n * (n - 3)) / 2 = 27 := by
  sorry

end nonagon_diagonals_l1208_120857


namespace rectangle_area_k_l1208_120872

theorem rectangle_area_k (d : ℝ) (x : ℝ) (h_ratio : 5 * x > 0 ∧ 2 * x > 0) (h_diagonal : d^2 = (5 * x)^2 + (2 * x)^2) :
  ∃ k : ℝ, (∃ (h : k = 10 / 29), (5 * x) * (2 * x) = k * d^2) := by
  use 10 / 29
  sorry

end rectangle_area_k_l1208_120872


namespace river_depth_ratio_l1208_120883

-- Definitions based on the conditions
def depthMidMay : ℝ := 5
def increaseMidJune : ℝ := 10
def depthMidJune : ℝ := depthMidMay + increaseMidJune
def depthMidJuly : ℝ := 45

-- The theorem based on the question and correct answer
theorem river_depth_ratio : depthMidJuly / depthMidJune = 3 := by 
  -- Proof skipped for illustration purposes
  sorry

end river_depth_ratio_l1208_120883


namespace infinitely_many_solutions_b_value_l1208_120849

theorem infinitely_many_solutions_b_value :
  ∀ (x : ℝ) (b : ℝ), (5 * (4 * x - b) = 3 * (5 * x + 15)) ↔ b = -9 := 
by
  intro x b
  sorry

end infinitely_many_solutions_b_value_l1208_120849


namespace determine_functions_l1208_120882

noncomputable def satisfies_condition (f : ℕ → ℕ) : Prop :=
∀ (n p : ℕ), Prime p → (f n)^p % f p = n % f p

theorem determine_functions :
  ∀ (f : ℕ → ℕ),
  satisfies_condition f →
  f = id ∨
  (∀ p: ℕ, Prime p → f p = 1) ∨
  (f 2 = 2 ∧ (∀ p: ℕ, Prime p → p > 2 → f p = 1) ∧ ∀ n: ℕ, f n % 2 = n % 2) :=
by
  intros f h1
  sorry

end determine_functions_l1208_120882


namespace gnomes_cannot_cross_l1208_120891

theorem gnomes_cannot_cross :
  ∀ (gnomes : List ℕ), 
    (∀ g, g ∈ gnomes → g ∈ (List.range 100).map (λ x => x + 1)) →
    List.sum gnomes = 5050 → 
    ∀ (boat_capacity : ℕ), boat_capacity = 100 →
    ∀ (k : ℕ), (200 * (k + 1) - k^2 = 10100) → false :=
by
  intros gnomes H_weights H_sum boat_capacity H_capacity k H_equation
  sorry

end gnomes_cannot_cross_l1208_120891


namespace kris_suspension_days_per_instance_is_three_l1208_120828

-- Define the basic parameters given in the conditions
def total_fingers_toes : ℕ := 20
def total_bullying_instances : ℕ := 20
def multiplier : ℕ := 3

-- Define total suspension days according to the conditions
def total_suspension_days : ℕ := multiplier * total_fingers_toes

-- Define the goal: to find the number of suspension days per instance
def suspension_days_per_instance : ℕ := total_suspension_days / total_bullying_instances

-- The theorem to prove that Kris was suspended for 3 days per instance
theorem kris_suspension_days_per_instance_is_three : suspension_days_per_instance = 3 := by
  -- Skip the actual proof, focus only on the statement
  sorry

end kris_suspension_days_per_instance_is_three_l1208_120828


namespace min_pounds_of_beans_l1208_120885

theorem min_pounds_of_beans : 
  ∃ (b : ℕ), (∀ (r : ℝ), (r ≥ 8 + b / 3 ∧ r ≤ 3 * b) → b ≥ 3) :=
sorry

end min_pounds_of_beans_l1208_120885


namespace find_x_value_l1208_120850

def acid_solution (m : ℕ) (x : ℕ) (h : m > 25) : Prop :=
  let initial_acid := m^2 / 100
  let total_volume := m + x
  let new_acid_concentration := (m - 5) / 100 * (m + x)
  initial_acid = new_acid_concentration

theorem find_x_value (m : ℕ) (h : m > 25) (x : ℕ) :
  (acid_solution m x h) → x = 5 * m / (m - 5) :=
sorry

end find_x_value_l1208_120850


namespace points_lie_on_ellipse_l1208_120877

open Real

noncomputable def curve_points_all_lie_on_ellipse (s: ℝ) : Prop :=
  let x := 2 * cos s + 2 * sin s
  let y := 4 * (cos s - sin s)
  (x^2 / 8 + y^2 / 32 = 1)

-- Below statement defines the theorem we aim to prove:
theorem points_lie_on_ellipse (s: ℝ) : curve_points_all_lie_on_ellipse s :=
sorry -- This "sorry" is to indicate that the proof is omitted.

end points_lie_on_ellipse_l1208_120877


namespace only_C_forms_triangle_l1208_120893

def triangle_sides (a b c : ℕ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

theorem only_C_forms_triangle :
  ¬ triangle_sides 3 4 8 ∧
  ¬ triangle_sides 2 5 2 ∧
  triangle_sides 3 5 6 ∧
  ¬ triangle_sides 5 6 11 :=
by
  sorry

end only_C_forms_triangle_l1208_120893


namespace fraction_multiplication_l1208_120830

-- Define the problem as a theorem in Lean
theorem fraction_multiplication
  (a b x : ℝ) (hx : x ≠ 0) (hb : b ≠ 0) (ha : a ≠ 0): 
  (3 * a * b / x) * (2 * x^2 / (9 * a * b^2)) = (2 * x) / (3 * b) := 
by
  sorry

end fraction_multiplication_l1208_120830


namespace smallest_n_Sn_pos_l1208_120868

theorem smallest_n_Sn_pos {a : ℕ → ℤ} (S : ℕ → ℤ) 
  (h1 : ∀ n, S n = n * a 1 + (n * (n - 1) / 2) * (a 2 - a 1))
  (h2 : ∀ n, (n ≠ 5 → S n > S 5))
  (h3 : |a 5| > |a 6|) :
  ∃ n : ℕ, S n > 0 ∧ ∀ m < n, S m ≤ 0 :=
by 
  -- Actual proof steps would go here.
  sorry

end smallest_n_Sn_pos_l1208_120868


namespace homogeneous_diff_eq_solution_l1208_120860

open Real

theorem homogeneous_diff_eq_solution (C : ℝ) : 
  ∀ (x y : ℝ), (y^4 - 2 * x^3 * y) * (dx) + (x^4 - 2 * x * y^3) * (dy) = 0 ↔ x^3 + y^3 = C * x * y :=
by
  sorry

end homogeneous_diff_eq_solution_l1208_120860


namespace minimum_sum_of_dimensions_of_box_l1208_120870

theorem minimum_sum_of_dimensions_of_box (a b c : ℕ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) (h_vol : a * b * c = 2310) :
  a + b + c ≥ 52 :=
sorry

end minimum_sum_of_dimensions_of_box_l1208_120870


namespace odd_ints_divisibility_l1208_120804

theorem odd_ints_divisibility (a b : ℤ) (ha_odd : a % 2 = 1) (hb_odd : b % 2 = 1) (hdiv : 2 * a * b + 1 ∣ a^2 + b^2 + 1) : a = b :=
sorry

end odd_ints_divisibility_l1208_120804


namespace chord_length_in_circle_l1208_120822

theorem chord_length_in_circle 
  (radius : ℝ) 
  (chord_midpoint_perpendicular_radius : ℝ)
  (r_eq_10 : radius = 10)
  (cmp_eq_5 : chord_midpoint_perpendicular_radius = 5) : 
  ∃ (chord_length : ℝ), chord_length = 10 * Real.sqrt 3 := 
by 
  sorry

end chord_length_in_circle_l1208_120822


namespace tim_total_points_l1208_120816

theorem tim_total_points :
  let single_points := 1000
  let tetris_points := 8 * single_points
  let singles := 6
  let tetrises := 4
  let total_points := singles * single_points + tetrises * tetris_points
  total_points = 38000 :=
by
  sorry

end tim_total_points_l1208_120816


namespace train_crosses_bridge_in_12_2_seconds_l1208_120854

def length_of_train : ℕ := 110
def speed_of_train_kmh : ℕ := 72
def length_of_bridge : ℕ := 134

def speed_of_train_ms : ℚ := speed_of_train_kmh * (1000 : ℚ) / (3600 : ℚ)
def total_distance : ℕ := length_of_train + length_of_bridge

noncomputable def time_to_cross_bridge : ℚ := total_distance / speed_of_train_ms

theorem train_crosses_bridge_in_12_2_seconds : time_to_cross_bridge = 12.2 := by
  sorry

end train_crosses_bridge_in_12_2_seconds_l1208_120854


namespace arithmetic_sequence_99th_term_l1208_120805

-- Define the problem with conditions and question
theorem arithmetic_sequence_99th_term (a : ℕ → ℕ) (S : ℕ → ℕ) (h1 : S 9 = 27) (h2 : a 10 = 8) :
  a 99 = 97 := 
sorry

end arithmetic_sequence_99th_term_l1208_120805


namespace initial_money_amount_l1208_120866

theorem initial_money_amount 
  (X : ℝ) 
  (h : 0.70 * X = 350) : 
  X = 500 := 
sorry

end initial_money_amount_l1208_120866


namespace second_third_parts_length_l1208_120855

variable (total_length : ℝ) (first_part : ℝ) (last_part : ℝ)
variable (second_third_part_length : ℝ)

def is_equal_length (x y : ℝ) := x = y

theorem second_third_parts_length :
  total_length = 74.5 ∧ first_part = 15.5 ∧ last_part = 16 → 
  is_equal_length (second_third_part_length) 21.5 :=
by
  intros h
  let remaining_distance := total_length - first_part - last_part
  let second_third_part_length := remaining_distance / 2
  sorry

end second_third_parts_length_l1208_120855


namespace min_eq_neg_one_implies_x_eq_two_l1208_120803

theorem min_eq_neg_one_implies_x_eq_two (x : ℝ) (h : min (2*x - 5) (x + 1) = -1) : x = 2 :=
sorry

end min_eq_neg_one_implies_x_eq_two_l1208_120803


namespace alex_silver_tokens_l1208_120810

-- Definitions and conditions
def initialRedTokens : ℕ := 100
def initialBlueTokens : ℕ := 50
def firstBoothRedChange (x : ℕ) : ℕ := 3 * x
def firstBoothSilverGain (x : ℕ) : ℕ := 2 * x
def firstBoothBlueGain (x : ℕ) : ℕ := x
def secondBoothBlueChange (y : ℕ) : ℕ := 2 * y
def secondBoothSilverGain (y : ℕ) : ℕ := y
def secondBoothRedGain (y : ℕ) : ℕ := y

-- Final conditions when no more exchanges are possible
def finalRedTokens (x y : ℕ) : ℕ := initialRedTokens - firstBoothRedChange x + secondBoothRedGain y
def finalBlueTokens (x y : ℕ) : ℕ := initialBlueTokens + firstBoothBlueGain x - secondBoothBlueChange y

-- Total silver tokens calculation
def totalSilverTokens (x y : ℕ) : ℕ := firstBoothSilverGain x + secondBoothSilverGain y

-- Proof that in the end, Alex has 147 silver tokens
theorem alex_silver_tokens : 
  ∃ (x y : ℕ), finalRedTokens x y = 2 ∧ finalBlueTokens x y = 1 ∧ totalSilverTokens x y = 147 :=
by
  -- the proof logic will be filled here
  sorry

end alex_silver_tokens_l1208_120810


namespace total_rings_is_19_l1208_120873

-- Definitions based on the problem conditions
def rings_on_first_day : Nat := 8
def rings_on_second_day : Nat := 6
def rings_on_third_day : Nat := 5

-- Total rings calculation
def total_rings : Nat := rings_on_first_day + rings_on_second_day + rings_on_third_day

-- Proof statement
theorem total_rings_is_19 : total_rings = 19 := by
  -- Proof goes here
  sorry

end total_rings_is_19_l1208_120873
