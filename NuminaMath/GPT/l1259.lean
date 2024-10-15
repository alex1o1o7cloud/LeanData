import Mathlib

namespace NUMINAMATH_GPT_katya_attached_squares_perimeter_l1259_125925

theorem katya_attached_squares_perimeter :
  let p1 := 100 -- Perimeter of the larger square
  let p2 := 40  -- Perimeter of the smaller square
  let s1 := p1 / 4 -- Side length of the larger square
  let s2 := p2 / 4 -- Side length of the smaller square
  let combined_perimeter_without_internal_sides := p1 + p2
  let actual_perimeter := combined_perimeter_without_internal_sides - 2 * s2
  actual_perimeter = 120 :=
by
  sorry

end NUMINAMATH_GPT_katya_attached_squares_perimeter_l1259_125925


namespace NUMINAMATH_GPT_armistice_day_is_wednesday_l1259_125991

-- Define the starting date
def start_day : Nat := 5 -- 5 represents Friday if we consider 0 = Sunday

-- Define the number of days after which armistice was signed
def days_after : Nat := 2253

-- Define the target day (Wednesday = 3)
def expected_day : Nat := 3

-- Define the function to calculate the day of the week after a number of days
def day_after_n_days (start_day : Nat) (n : Nat) : Nat :=
  (start_day + n) % 7

-- Define the theorem to prove the equivalent mathematical problem
theorem armistice_day_is_wednesday : day_after_n_days start_day days_after = expected_day := by
  sorry

end NUMINAMATH_GPT_armistice_day_is_wednesday_l1259_125991


namespace NUMINAMATH_GPT_fourth_powers_count_l1259_125967

theorem fourth_powers_count (n m : ℕ) (h₁ : n^4 ≥ 100) (h₂ : m^4 ≤ 10000) :
  ∃ k, k = m - n + 1 ∧ k = 7 :=
by
  sorry

end NUMINAMATH_GPT_fourth_powers_count_l1259_125967


namespace NUMINAMATH_GPT_problem_statement_l1259_125955

   noncomputable def g (x : ℝ) : ℝ := (3 * x + 4) / (x + 3)

   def T := {y : ℝ | ∃ (x : ℝ), x ≥ 0 ∧ y = g x}

   theorem problem_statement :
     (∃ N, (∀ y ∈ T, y ≤ N) ∧ N = 3 ∧ N ∉ T) ∧
     (∃ n, (∀ y ∈ T, y ≥ n) ∧ n = 4/3 ∧ n ∈ T) :=
   by
     sorry
   
end NUMINAMATH_GPT_problem_statement_l1259_125955


namespace NUMINAMATH_GPT_shop_makes_off_each_jersey_l1259_125982

theorem shop_makes_off_each_jersey :
  ∀ (T : ℝ) (jersey_earnings : ℝ),
  (T = 25) →
  (jersey_earnings = T + 90) →
  jersey_earnings = 115 := by
  intros T jersey_earnings ht hj
  sorry

end NUMINAMATH_GPT_shop_makes_off_each_jersey_l1259_125982


namespace NUMINAMATH_GPT_optimal_sampling_methods_l1259_125944

/-
We define the conditions of the problem.
-/
def households := 500
def high_income_households := 125
def middle_income_households := 280
def low_income_households := 95
def sample_households := 100

def soccer_players := 12
def sample_soccer_players := 3

/-
We state the goal as a theorem.
-/
theorem optimal_sampling_methods :
  (sample_households == 100) ∧
  (sample_soccer_players == 3) ∧
  (high_income_households + middle_income_households + low_income_households == households) →
  ("stratified" = "stratified" ∧ "random" = "random") :=
by
  -- Sorry to skip the proof
  sorry

end NUMINAMATH_GPT_optimal_sampling_methods_l1259_125944


namespace NUMINAMATH_GPT_false_proposition_of_quadratic_l1259_125968

theorem false_proposition_of_quadratic
  (a : ℝ) (h0 : a ≠ 0)
  (h1 : ¬(5 = a * (1/2)^2 + (-a^2 - 1) * (1/2) + a))
  (h2 : (a^2 + 1) / (2 * a) > 0)
  (h3 : (0, a) = (0, x) ∧ x > 0)
  (h4 : ∀ x : ℝ, a * x^2 + (-a^2 - 1) * x + a ≤ 0) :
  false :=
sorry

end NUMINAMATH_GPT_false_proposition_of_quadratic_l1259_125968


namespace NUMINAMATH_GPT_new_boarders_day_scholars_ratio_l1259_125952

theorem new_boarders_day_scholars_ratio
  (initial_boarders : ℕ)
  (initial_day_scholars : ℕ)
  (ratio_boarders_day_scholars : ℕ → ℕ → Prop)
  (additional_boarders : ℕ)
  (new_boarders : ℕ)
  (new_ratio : ℕ → ℕ → Prop)
  (r1 r2 : ℕ)
  (h1 : ratio_boarders_day_scholars 7 16)
  (h2 : initial_boarders = 560)
  (h3 : initial_day_scholars = 1280)
  (h4 : additional_boarders = 80)
  (h5 : new_boarders = initial_boarders + additional_boarders)
  (h6 : new_ratio new_boarders initial_day_scholars) :
  new_ratio r1 r2 → r1 = 1 ∧ r2 = 2 :=
by {
    sorry
}

end NUMINAMATH_GPT_new_boarders_day_scholars_ratio_l1259_125952


namespace NUMINAMATH_GPT_number_of_ordered_pairs_l1259_125987

theorem number_of_ordered_pairs : ∃ (s : Finset (ℂ × ℂ)), 
    (∀ (a b : ℂ), (a, b) ∈ s → a^5 * b^3 = 1 ∧ a^9 * b^2 = 1) ∧ 
    s.card = 17 := 
by
  sorry

end NUMINAMATH_GPT_number_of_ordered_pairs_l1259_125987


namespace NUMINAMATH_GPT_courtyard_width_l1259_125905

def width_of_courtyard (w : ℝ) : Prop :=
  28 * 100 * 100 * w = 13788 * 22 * 12

theorem courtyard_width :
  ∃ w : ℝ, width_of_courtyard w ∧ abs (w - 13.012) < 0.001 :=
by
  sorry

end NUMINAMATH_GPT_courtyard_width_l1259_125905


namespace NUMINAMATH_GPT_nancy_age_l1259_125912

variable (n g : ℕ)

theorem nancy_age (h1 : g = 10 * n) (h2 : g - n = 45) : n = 5 :=
by
  sorry

end NUMINAMATH_GPT_nancy_age_l1259_125912


namespace NUMINAMATH_GPT_center_of_circle_l1259_125999

theorem center_of_circle : ∃ c : ℝ × ℝ, 
  (∃ r : ℝ, ∀ x y : ℝ, (x - c.1) * (x - c.1) + (y - c.2) * (y - c.2) = r ↔ x^2 + y^2 - 6*x - 2*y - 15 = 0) → c = (3, 1) :=
by 
  sorry

end NUMINAMATH_GPT_center_of_circle_l1259_125999


namespace NUMINAMATH_GPT_train_length_l1259_125948

noncomputable def jogger_speed_kmh : ℝ := 9
noncomputable def train_speed_kmh : ℝ := 45
noncomputable def head_start : ℝ := 270
noncomputable def passing_time : ℝ := 39

noncomputable def kmh_to_ms (speed: ℝ) : ℝ := speed * (1000 / 3600)

theorem train_length (l : ℝ) 
  (v_j := kmh_to_ms jogger_speed_kmh)
  (v_t := kmh_to_ms train_speed_kmh)
  (d_h := head_start)
  (t := passing_time) :
  l = 120 :=
by 
  sorry

end NUMINAMATH_GPT_train_length_l1259_125948


namespace NUMINAMATH_GPT_min_value_of_m_n_l1259_125984

variable {a b : ℝ}
variable (ab_eq_4 : a * b = 4)
variable (m : ℝ := b + 1 / a)
variable (n : ℝ := a + 1 / b)

theorem min_value_of_m_n (h1 : 0 < a) (h2 : 0 < b) : m + n = 5 :=
sorry

end NUMINAMATH_GPT_min_value_of_m_n_l1259_125984


namespace NUMINAMATH_GPT_solve_eq1_solve_eq2_l1259_125954

-- Define the first equation
def eq1 (x : ℚ) : Prop := x / (x - 1) = 3 / (2*x - 2) - 2

-- Define the valid solution for the first equation
def sol1 : ℚ := 7 / 6

-- Theorem for the first equation
theorem solve_eq1 : eq1 sol1 :=
by
  sorry

-- Define the second equation
def eq2 (x : ℚ) : Prop := (5*x + 2) / (x^2 + x) = 3 / (x + 1)

-- Theorem for the second equation: there is no valid solution
theorem solve_eq2 : ¬ ∃ x : ℚ, eq2 x :=
by
  sorry

end NUMINAMATH_GPT_solve_eq1_solve_eq2_l1259_125954


namespace NUMINAMATH_GPT_count_equilateral_triangles_in_hexagonal_lattice_l1259_125916

-- Definitions based on conditions in problem (hexagonal lattice setup)
def hexagonal_lattice (dist : ℕ) : Prop :=
  -- Define properties of the points in hexagonal lattice
  -- Placeholder for actual structure defining the hexagon and surrounding points
  sorry

def equilateral_triangles (n : ℕ) : Prop :=
  -- Define a method to count equilateral triangles in the given lattice setup
  sorry

-- Theorem stating that 10 equilateral triangles can be formed in the lattice
theorem count_equilateral_triangles_in_hexagonal_lattice (dist : ℕ) (h : dist = 1 ∨ dist = 2) :
  equilateral_triangles 10 :=
by
  -- Proof to be completed
  sorry

end NUMINAMATH_GPT_count_equilateral_triangles_in_hexagonal_lattice_l1259_125916


namespace NUMINAMATH_GPT_min_brilliant_triple_product_l1259_125956

theorem min_brilliant_triple_product :
  ∃ a b c : ℕ, a > b ∧ b > c ∧ Prime a ∧ Prime b ∧ Prime c ∧ (a = b + 2 * c) ∧ (∃ k : ℕ, (a + b + c) = k^2) ∧ (a * b * c = 35651) :=
by
  sorry

end NUMINAMATH_GPT_min_brilliant_triple_product_l1259_125956


namespace NUMINAMATH_GPT_compare_neg_fractions_and_neg_values_l1259_125911

theorem compare_neg_fractions_and_neg_values :
  (- (3 : ℚ) / 4 > - (4 : ℚ) / 5) ∧ (-(-3 : ℤ) > -|(3 : ℤ)|) :=
by
  apply And.intro
  sorry
  sorry

end NUMINAMATH_GPT_compare_neg_fractions_and_neg_values_l1259_125911


namespace NUMINAMATH_GPT_responses_needed_l1259_125931

-- Define the given conditions
def rate : ℝ := 0.80
def num_mailed : ℕ := 375

-- Statement to prove
theorem responses_needed :
  rate * num_mailed = 300 := by
  sorry

end NUMINAMATH_GPT_responses_needed_l1259_125931


namespace NUMINAMATH_GPT_remainder_product_mod_five_l1259_125903

-- Define the conditions as congruences
def num1 : ℕ := 14452
def num2 : ℕ := 15652
def num3 : ℕ := 16781

-- State the main theorem using the conditions and the given problem
theorem remainder_product_mod_five : 
  (num1 % 5 = 2) → 
  (num2 % 5 = 2) → 
  (num3 % 5 = 1) → 
  ((num1 * num2 * num3) % 5 = 4) :=
by
  intros
  sorry

end NUMINAMATH_GPT_remainder_product_mod_five_l1259_125903


namespace NUMINAMATH_GPT_find_length_of_PB_l1259_125936

theorem find_length_of_PB
  (PA : ℝ) -- Define PA
  (h_PA : PA = 4) -- Condition PA = 4
  (PB : ℝ) -- Define PB
  (PT : ℝ) -- Define PT
  (h_PT : PT = PB - 2 * PA) -- Condition PT = PB - 2 * PA
  (h_power_of_a_point : PA * PB = PT^2) -- Condition PA * PB = PT^2
  : PB = 16 :=
sorry

end NUMINAMATH_GPT_find_length_of_PB_l1259_125936


namespace NUMINAMATH_GPT_mass_percentage_of_H_in_H2O_is_11_19_l1259_125900

def mass_of_hydrogen : Float := 1.008
def mass_of_oxygen : Float := 16.00
def mass_of_H2O : Float := 2 * mass_of_hydrogen + mass_of_oxygen
def mass_percentage_hydrogen : Float :=
  (2 * mass_of_hydrogen / mass_of_H2O) * 100

theorem mass_percentage_of_H_in_H2O_is_11_19 :
  mass_percentage_hydrogen = 11.19 :=
  sorry

end NUMINAMATH_GPT_mass_percentage_of_H_in_H2O_is_11_19_l1259_125900


namespace NUMINAMATH_GPT_circle_center_coordinates_l1259_125960

theorem circle_center_coordinates :
  ∃ (h k : ℝ), (x^2 + y^2 - 2*x + 4*y = 0) → (x - h)^2 + (y + k)^2 = 5 :=
sorry

end NUMINAMATH_GPT_circle_center_coordinates_l1259_125960


namespace NUMINAMATH_GPT_sin_cos_relationship_l1259_125923

theorem sin_cos_relationship (α : ℝ) (h1 : Real.pi / 2 < α) (h2 : α < Real.pi) : 
  Real.sin α - Real.cos α > 1 :=
sorry

end NUMINAMATH_GPT_sin_cos_relationship_l1259_125923


namespace NUMINAMATH_GPT_checkerboard_probability_l1259_125918

def total_squares (n : ℕ) : ℕ :=
  n * n

def perimeter_squares (n : ℕ) : ℕ :=
  4 * n - 4

def non_perimeter_squares (n : ℕ) : ℕ :=
  total_squares n - perimeter_squares n

def probability_non_perimeter_square (n : ℕ) : ℚ :=
  non_perimeter_squares n / total_squares n

theorem checkerboard_probability :
  probability_non_perimeter_square 10 = 16 / 25 :=
by
  sorry

end NUMINAMATH_GPT_checkerboard_probability_l1259_125918


namespace NUMINAMATH_GPT_different_values_count_l1259_125962

theorem different_values_count (i : ℕ) (h : 1 ≤ i ∧ i ≤ 2015) : 
  ∃ l : Finset ℕ, (∀ j ∈ l, ∃ i : ℕ, (1 ≤ i ∧ i ≤ 2015) ∧ j = (i^2 / 2015)) ∧
  l.card = 2016 := 
sorry

end NUMINAMATH_GPT_different_values_count_l1259_125962


namespace NUMINAMATH_GPT_root_of_quadratic_l1259_125933

theorem root_of_quadratic (x m : ℝ) (h : x = -1 ∧ x^2 + m*x - 1 = 0) : m = 0 :=
sorry

end NUMINAMATH_GPT_root_of_quadratic_l1259_125933


namespace NUMINAMATH_GPT_part1_part2_part3_l1259_125953

section ShoppingMall

variable (x y a b : ℝ)
variable (cpaA spaA cpaB spaB : ℝ)
variable (n total_y yuan : ℝ)

-- Conditions given in the problem
def cost_price_A := 160
def selling_price_A := 220
def cost_price_B := 120
def selling_price_B := 160
def total_clothing := 100
def min_A_clothing := 60
def max_budget := 15000
def discount_diff := 4
def max_profit_with_discount := 4950

-- Definitions applied from conditions
def profit_per_piece_A := selling_price_A - cost_price_A
def profit_per_piece_B := selling_price_B - cost_price_B

-- Question 1: Functional relationship between y and x
theorem part1 : 
  (∀ (x : ℝ), x ≥ 0 → x ≤ total_clothing → 
  y = profit_per_piece_A * x + profit_per_piece_B * (total_clothing - x)) →
  y = 20 * x + 4000 := 
sorry

-- Question 2: Maximum profit under given cost constraints
theorem part2 : 
  (min_A_clothing ≤ x ∧ x ≤ 75 ∧ 
  (cost_price_A * x + cost_price_B * (total_clothing - x) ≤ max_budget)) →
  y = 20 * 75 + 4000 → 
  y = 5500 :=
sorry

-- Question 3: Determine a under max profit condition
theorem part3 : 
  (a - b = discount_diff ∧ 0 < a ∧ a < 20 ∧ 
  (20 - a) * 75 + 4000 + 100 * a - 400 = max_profit_with_discount) →
  a = 9 :=
sorry

end ShoppingMall

end NUMINAMATH_GPT_part1_part2_part3_l1259_125953


namespace NUMINAMATH_GPT_surplus_by_end_of_week_is_14_estimated_monthly_income_is_1860_l1259_125922

-- Given conditions
def income_per_day : List Int := [65, 68, 50, 66, 50, 75, 74]
def expenditure_per_day : List Int := [-60, -64, -63, -58, -60, -64, -65]

-- Part 1: Proving the surplus by the end of the week is 14 yuan
theorem surplus_by_end_of_week_is_14 :
  List.sum income_per_day + List.sum expenditure_per_day = 14 :=
by
  sorry

-- Part 2: Proving the estimated income needed per month to maintain normal expenses is 1860 yuan
theorem estimated_monthly_income_is_1860 :
  (List.sum (List.map Int.natAbs expenditure_per_day) / 7) * 30 = 1860 :=
by
  sorry

end NUMINAMATH_GPT_surplus_by_end_of_week_is_14_estimated_monthly_income_is_1860_l1259_125922


namespace NUMINAMATH_GPT_side_length_of_square_l1259_125934

theorem side_length_of_square (d : ℝ) (h_d : d = 2 * Real.sqrt 2) : 
  ∃ s : ℝ, d = s * Real.sqrt 2 ∧ s = 2 := by
  sorry

end NUMINAMATH_GPT_side_length_of_square_l1259_125934


namespace NUMINAMATH_GPT_math_problem_l1259_125904

theorem math_problem :
  (1 / (1 / (1 / (1 / (3 + 2 : ℝ) - 1 : ℝ) - 1 : ℝ) - 1 : ℝ) - 1 : ℝ) = -13 / 9 :=
by
  -- proof goes here
  sorry

end NUMINAMATH_GPT_math_problem_l1259_125904


namespace NUMINAMATH_GPT_shorter_diagonal_of_rhombus_l1259_125921

variable (d s : ℝ)  -- d for shorter diagonal, s for the side length of the rhombus

theorem shorter_diagonal_of_rhombus 
  (h1 : ∀ (s : ℝ), s = 39)
  (h2 : ∀ (a b : ℝ), a^2 + b^2 = s^2)
  (h3 : ∀ (d a : ℝ), (d / 2)^2 + a^2 = 39^2)
  (h4 : 72 / 2 = 36)
  : d = 30 := 
by 
  sorry

end NUMINAMATH_GPT_shorter_diagonal_of_rhombus_l1259_125921


namespace NUMINAMATH_GPT_percentage_difference_l1259_125998

theorem percentage_difference (n z x y y_decreased : ℝ)
  (h1 : x = 8 * y)
  (h2 : y = 2 * |z - n|)
  (h3 : z = 1.1 * n)
  (h4 : y_decreased = 0.75 * y) :
  (x - y_decreased) / x * 100 = 90.625 := by
sorry

end NUMINAMATH_GPT_percentage_difference_l1259_125998


namespace NUMINAMATH_GPT_sequence_bounds_l1259_125990

theorem sequence_bounds (θ : ℝ) (n : ℕ) (a : ℕ → ℝ) (hθ : 0 < θ ∧ θ < π / 2) 
  (h1 : a 1 = 1) 
  (h2 : a 2 = 1 - 2 * (Real.sin θ * Real.cos θ)^2) 
  (h_recurrence : ∀ n, a (n + 2) - a (n + 1) + a n * (Real.sin θ * Real.cos θ)^2 = 0) :
  1 / 2 ^ (n - 1) ≤ a n ∧ a n ≤ 1 - (Real.sin (2 * θ))^n * (1 - 1 / 2 ^ (n - 1)) := 
sorry

end NUMINAMATH_GPT_sequence_bounds_l1259_125990


namespace NUMINAMATH_GPT_kaylin_is_younger_by_five_l1259_125971

def Freyja_age := 10
def Kaylin_age := 33
def Eli_age := Freyja_age + 9
def Sarah_age := 2 * Eli_age
def age_difference := Sarah_age - Kaylin_age

theorem kaylin_is_younger_by_five : age_difference = 5 := 
by
  show 5 = Sarah_age - Kaylin_age
  sorry

end NUMINAMATH_GPT_kaylin_is_younger_by_five_l1259_125971


namespace NUMINAMATH_GPT_first_number_is_nine_l1259_125964

theorem first_number_is_nine (x : ℤ) (h : 11 * x = 3 * (x + 4) + 16 + 4 * (x + 2)) : x = 9 :=
by {
  sorry
}

end NUMINAMATH_GPT_first_number_is_nine_l1259_125964


namespace NUMINAMATH_GPT_geometric_series_ratio_l1259_125997

theorem geometric_series_ratio (a r : ℝ) (h : a ≠ 0) (h2 : 0 < r ∧ r < 1) :
    (a / (1 - r)) = 81 * (ar^4 / (1 - r)) → r = 1 / 3 :=
by
  sorry

end NUMINAMATH_GPT_geometric_series_ratio_l1259_125997


namespace NUMINAMATH_GPT_standard_polar_representation_l1259_125992

theorem standard_polar_representation {r θ : ℝ} (hr : r < 0) (hθ : θ = 5 * Real.pi / 6) :
  ∃ (r' θ' : ℝ), r' > 0 ∧ 0 ≤ θ' ∧ θ' < 2 * Real.pi ∧ (r', θ') = (5, 11 * Real.pi / 6) := 
by {
  sorry
}

end NUMINAMATH_GPT_standard_polar_representation_l1259_125992


namespace NUMINAMATH_GPT_arthur_muffins_l1259_125938

variable (arthur_baked : ℕ)
variable (james_baked : ℕ := 1380)
variable (times_as_many : ℕ := 12)

theorem arthur_muffins : arthur_baked * times_as_many = james_baked -> arthur_baked = 115 := by
  sorry

end NUMINAMATH_GPT_arthur_muffins_l1259_125938


namespace NUMINAMATH_GPT_convert_444_quinary_to_octal_l1259_125914

def quinary_to_decimal (n : ℕ) : ℕ :=
  let d2 := (n / 100) * 25
  let d1 := ((n % 100) / 10) * 5
  let d0 := (n % 10)
  d2 + d1 + d0

def decimal_to_octal (n : ℕ) : ℕ :=
  let r2 := (n / 64)
  let n2 := (n % 64)
  let r1 := (n2 / 8)
  let r0 := (n2 % 8)
  r2 * 100 + r1 * 10 + r0

theorem convert_444_quinary_to_octal :
  decimal_to_octal (quinary_to_decimal 444) = 174 := by
  sorry

end NUMINAMATH_GPT_convert_444_quinary_to_octal_l1259_125914


namespace NUMINAMATH_GPT_green_paint_quarts_l1259_125901

theorem green_paint_quarts (blue green white : ℕ) (h_ratio : 3 = blue ∧ 2 = green ∧ 4 = white) 
  (h_white_paint : white = 12) : green = 6 := 
by
  sorry

end NUMINAMATH_GPT_green_paint_quarts_l1259_125901


namespace NUMINAMATH_GPT_percentage_difference_l1259_125947

variable {P Q : ℝ}

theorem percentage_difference (P Q : ℝ) : (100 * (Q - P)) / Q = ((Q - P) / Q) * 100 :=
by
  sorry

end NUMINAMATH_GPT_percentage_difference_l1259_125947


namespace NUMINAMATH_GPT_acute_angle_sum_l1259_125908

theorem acute_angle_sum (n : ℕ) (hn : n ≥ 4) (M m: ℕ) 
  (hM : M = 3) (hm : m = 0) : M + m = 3 := 
by 
  sorry

end NUMINAMATH_GPT_acute_angle_sum_l1259_125908


namespace NUMINAMATH_GPT_simplify_and_evaluate_sqrt_log_product_property_l1259_125975

-- Problem I
theorem simplify_and_evaluate_sqrt (a : ℝ) (h : 0 < a) : 
  Real.sqrt (a^(1/4) * Real.sqrt (a * Real.sqrt a)) = Real.sqrt a := 
by
  sorry

-- Problem II
theorem log_product_property : 
  Real.log 3 / Real.log 2 * Real.log 5 / Real.log 3 * Real.log 4 / Real.log 5 = 2 := 
by
  sorry

end NUMINAMATH_GPT_simplify_and_evaluate_sqrt_log_product_property_l1259_125975


namespace NUMINAMATH_GPT_approx_average_sqft_per_person_l1259_125979

noncomputable def average_sqft_per_person 
  (population : ℕ) 
  (land_area_sqmi : ℕ) 
  (sqft_per_sqmi : ℕ) : ℕ :=
(sqft_per_sqmi * land_area_sqmi) / population

theorem approx_average_sqft_per_person :
  average_sqft_per_person 331000000 3796742 (5280 ^ 2) = 319697 := 
sorry

end NUMINAMATH_GPT_approx_average_sqft_per_person_l1259_125979


namespace NUMINAMATH_GPT_sum_of_interior_angles_6_find_n_from_300_degrees_l1259_125910

-- Definitions and statement for part 1:
def sum_of_interior_angles (n : ℕ) : ℕ :=
  (n - 2) * 180

theorem sum_of_interior_angles_6 :
  sum_of_interior_angles 6 = 720 := 
by
  sorry

-- Definitions and statement for part 2:
def find_n_from_angles (angle : ℕ) : ℕ := 
  (angle / 180) + 2

theorem find_n_from_300_degrees :
  find_n_from_angles 900 = 7 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_interior_angles_6_find_n_from_300_degrees_l1259_125910


namespace NUMINAMATH_GPT_mod_multiplication_example_l1259_125950

theorem mod_multiplication_example :
  (98 % 75) * (202 % 75) % 75 = 71 :=
by
  have h1 : 98 % 75 = 23 := by sorry
  have h2 : 202 % 75 = 52 := by sorry
  have h3 : 1196 % 75 = 71 := by sorry
  exact h3

end NUMINAMATH_GPT_mod_multiplication_example_l1259_125950


namespace NUMINAMATH_GPT_perpendicular_vectors_l1259_125949

theorem perpendicular_vectors (b : ℚ) : 
  (4 * b - 15 = 0) → (b = 15 / 4) :=
by
  intro h
  sorry

end NUMINAMATH_GPT_perpendicular_vectors_l1259_125949


namespace NUMINAMATH_GPT_sphere_radius_is_16_25_l1259_125965

def sphere_in_cylinder_radius (r : ℝ) : Prop := 
  ∃ (x : ℝ), (x ^ 2 + 15 ^ 2 = r ^ 2) ∧ ((x + 10) ^ 2 = r ^ 2) ∧ (r = 16.25)

theorem sphere_radius_is_16_25 : 
  sphere_in_cylinder_radius 16.25 :=
sorry

end NUMINAMATH_GPT_sphere_radius_is_16_25_l1259_125965


namespace NUMINAMATH_GPT_minimum_value_l1259_125996

theorem minimum_value (x y : ℝ) (h1 : x > y) (h2 : y > 0) (h3 : x + y = 1) :
  (2 / (x + 3 * y) + 1 / (x - y)) = (3 + 2 * Real.sqrt 2) / 2 := sorry

end NUMINAMATH_GPT_minimum_value_l1259_125996


namespace NUMINAMATH_GPT_equation_is_true_l1259_125907

theorem equation_is_true :
  10 * 6 - (9 - 3) * 2 = 48 :=
by
  sorry

end NUMINAMATH_GPT_equation_is_true_l1259_125907


namespace NUMINAMATH_GPT_part_one_extreme_value_part_two_max_k_l1259_125942

noncomputable def f (x : ℝ) (k : ℝ) : ℝ :=
  x * Real.log x - k * (x - 1)

theorem part_one_extreme_value :
  ∃ x : ℝ, x > 0 ∧ ∀ y > 0, f y 1 ≥ f x 1 ∧ f x 1 = 0 := 
  sorry

theorem part_two_max_k :
  ∀ x : ℝ, ∃ k : ℕ, (1 < x) -> (f x (k:ℝ) + x > 0) ∧ k = 3 :=
  sorry

end NUMINAMATH_GPT_part_one_extreme_value_part_two_max_k_l1259_125942


namespace NUMINAMATH_GPT_percentage_increase_overtime_rate_l1259_125927

theorem percentage_increase_overtime_rate :
  let regular_rate := 16
  let regular_hours_limit := 30
  let total_earnings := 760
  let total_hours_worked := 40
  let overtime_rate := 28 -- This is calculated as $280/10 from the solution.
  let increase_in_hourly_rate := overtime_rate - regular_rate
  let percentage_increase := (increase_in_hourly_rate / regular_rate) * 100
  percentage_increase = 75 :=
by {
  sorry
}

end NUMINAMATH_GPT_percentage_increase_overtime_rate_l1259_125927


namespace NUMINAMATH_GPT_find_smallest_result_l1259_125986

namespace small_result

def num_set : Set Int := { -10, -4, 0, 2, 7 }

def all_results : Set Int := 
  { z | ∃ x ∈ num_set, ∃ y ∈ num_set, z = x * y ∨ z = x + y }

def smallest_result := -70

theorem find_smallest_result : ∃ z ∈ all_results, z = smallest_result :=
by
  sorry

end small_result

end NUMINAMATH_GPT_find_smallest_result_l1259_125986


namespace NUMINAMATH_GPT_no_integer_solutions_quadratic_l1259_125930

theorem no_integer_solutions_quadratic (n : ℤ) (s : ℕ) (pos_odd_s : s % 2 = 1) :
  ¬ ∃ x : ℤ, x^2 - 16 * n * x + 7^s = 0 :=
sorry

end NUMINAMATH_GPT_no_integer_solutions_quadratic_l1259_125930


namespace NUMINAMATH_GPT_doubled_money_is_1_3_l1259_125940

-- Define the amounts of money Alice and Bob have
def alice_money := (2 : ℚ) / 5
def bob_money := (1 : ℚ) / 4

-- Define the total money before doubling
def total_money_before_doubling := alice_money + bob_money

-- Define the total money after doubling
def total_money_after_doubling := 2 * total_money_before_doubling

-- State the proposition to prove
theorem doubled_money_is_1_3 : total_money_after_doubling = 1.3 := by
  -- The proof will be filled in here
  sorry

end NUMINAMATH_GPT_doubled_money_is_1_3_l1259_125940


namespace NUMINAMATH_GPT_point_inside_circle_l1259_125993

theorem point_inside_circle :
  let center := (2, 3)
  let radius := 2
  let P := (1, 2)
  let distance_squared := (P.1 - center.1)^2 + (P.2 - center.2)^2
  distance_squared < radius^2 :=
by
  -- Definitions
  let center := (2, 3)
  let radius := 2
  let P := (1, 2)
  let distance_squared := (P.1 - center.1) ^ 2 + (P.2 - center.2) ^ 2

  -- Goal
  show distance_squared < radius ^ 2
  
  -- Skip Proof
  sorry

end NUMINAMATH_GPT_point_inside_circle_l1259_125993


namespace NUMINAMATH_GPT_hybrids_with_full_headlights_l1259_125958

-- Definitions for the conditions
def total_cars : ℕ := 600
def percentage_hybrids : ℕ := 60
def percentage_one_headlight : ℕ := 40

-- The proof statement
theorem hybrids_with_full_headlights :
  (percentage_hybrids * total_cars) / 100 - (percentage_one_headlight * (percentage_hybrids * total_cars) / 100) / 100 = 216 :=
by
  sorry

end NUMINAMATH_GPT_hybrids_with_full_headlights_l1259_125958


namespace NUMINAMATH_GPT_max_dinners_for_7_people_max_dinners_for_8_people_l1259_125909

def max_dinners_with_new_neighbors (n : ℕ) : ℕ :=
  if n = 7 ∨ n = 8 then 3 else 0

theorem max_dinners_for_7_people : max_dinners_with_new_neighbors 7 = 3 := sorry

theorem max_dinners_for_8_people : max_dinners_with_new_neighbors 8 = 3 := sorry

end NUMINAMATH_GPT_max_dinners_for_7_people_max_dinners_for_8_people_l1259_125909


namespace NUMINAMATH_GPT_coin_diameter_l1259_125985

theorem coin_diameter (r : ℝ) (h : r = 7) : 2 * r = 14 := by
  rw [h]
  norm_num

end NUMINAMATH_GPT_coin_diameter_l1259_125985


namespace NUMINAMATH_GPT_arithmetic_sequence_a12_bound_l1259_125915

theorem arithmetic_sequence_a12_bound (a_1 d : ℤ) (h8 : a_1 + 7 * d ≥ 15) (h9 : a_1 + 8 * d ≤ 13) : 
  a_1 + 11 * d ≤ 7 :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_a12_bound_l1259_125915


namespace NUMINAMATH_GPT_total_telephone_bill_second_month_l1259_125902

variable (F C : ℝ)

-- Elvin's total telephone bill for January is 40 dollars
axiom january_bill : F + C = 40

-- The charge for calls in the second month is twice the charge for calls in January
axiom second_month_call_charge : ∃ C2, C2 = 2 * C

-- Proof that the total telephone bill for the second month is 40 + C
theorem total_telephone_bill_second_month : 
  ∃ S, S = F + 2 * C ∧ S = 40 + C :=
sorry

end NUMINAMATH_GPT_total_telephone_bill_second_month_l1259_125902


namespace NUMINAMATH_GPT_remaining_distance_l1259_125994

-- Definitions of conditions
def distance_to_grandmother : ℕ := 300
def speed_per_hour : ℕ := 60
def time_elapsed : ℕ := 2

-- Statement of the proof problem
theorem remaining_distance : distance_to_grandmother - (speed_per_hour * time_elapsed) = 180 :=
by 
  sorry

end NUMINAMATH_GPT_remaining_distance_l1259_125994


namespace NUMINAMATH_GPT_sam_bought_new_books_l1259_125983

   def books_question (a m u : ℕ) : ℕ := (a + m) - u

   theorem sam_bought_new_books (a m u : ℕ) (h1 : a = 13) (h2 : m = 17) (h3 : u = 15) :
     books_question a m u = 15 :=
   by sorry
   
end NUMINAMATH_GPT_sam_bought_new_books_l1259_125983


namespace NUMINAMATH_GPT_multiple_is_eight_l1259_125972

theorem multiple_is_eight (m : ℝ) (h : 17 = m * 2.625 - 4) : m = 8 :=
by
  sorry

end NUMINAMATH_GPT_multiple_is_eight_l1259_125972


namespace NUMINAMATH_GPT_alice_age_l1259_125980

theorem alice_age (x : ℕ) (h1 : ∃ n : ℕ, x - 4 = n^2) (h2 : ∃ m : ℕ, x + 2 = m^3) : x = 58 :=
sorry

end NUMINAMATH_GPT_alice_age_l1259_125980


namespace NUMINAMATH_GPT_shooter_scores_l1259_125928

theorem shooter_scores
    (x y z : ℕ)
    (hx : x + y + z > 11)
    (hscore: 8 * x + 9 * y + 10 * z = 100) :
    (x + y + z = 12) ∧ ((x = 10 ∧ y = 0 ∧ z = 2) ∨ (x = 9 ∧ y = 2 ∧ z = 1) ∨ (x = 8 ∧ y = 4 ∧ z = 0)) :=
by
  sorry

end NUMINAMATH_GPT_shooter_scores_l1259_125928


namespace NUMINAMATH_GPT_sum_of_x_values_satisfying_eq_l1259_125977

noncomputable def rational_eq_sum (x : ℝ) : Prop :=
3 = (x^3 - 3 * x^2 - 12 * x) / (x + 3)

theorem sum_of_x_values_satisfying_eq :
  (∃ (x : ℝ), rational_eq_sum x) ∧ (x ≠ -3 → (x_1 + x_2) = 6) :=
sorry

end NUMINAMATH_GPT_sum_of_x_values_satisfying_eq_l1259_125977


namespace NUMINAMATH_GPT_observations_decrement_l1259_125919

theorem observations_decrement (n : ℤ) (h_n_pos : n > 0) : 200 - 15 = 185 :=
by
  sorry

end NUMINAMATH_GPT_observations_decrement_l1259_125919


namespace NUMINAMATH_GPT_speed_for_remaining_distance_l1259_125906

theorem speed_for_remaining_distance
  (t_total : ℝ) (v1 : ℝ) (d_total : ℝ)
  (t_total_def : t_total = 1.4)
  (v1_def : v1 = 4)
  (d_total_def : d_total = 5.999999999999999) :
  ∃ v2 : ℝ, v2 = 5 := 
by
  sorry

end NUMINAMATH_GPT_speed_for_remaining_distance_l1259_125906


namespace NUMINAMATH_GPT_matthews_contribution_l1259_125920

theorem matthews_contribution 
  (total_cost : ℝ) (yen_amount : ℝ) (conversion_rate : ℝ)
  (h1 : total_cost = 18)
  (h2 : yen_amount = 2500)
  (h3 : conversion_rate = 140) :
  (total_cost - (yen_amount / conversion_rate)) = 0.143 :=
by sorry

end NUMINAMATH_GPT_matthews_contribution_l1259_125920


namespace NUMINAMATH_GPT_average_distinct_k_values_l1259_125974

theorem average_distinct_k_values (k : ℕ) (h : ∃ r1 r2 : ℕ, r1 * r2 = 24 ∧ r1 + r2 = k ∧ r1 > 0 ∧ r2 > 0) : k = 15 :=
sorry

end NUMINAMATH_GPT_average_distinct_k_values_l1259_125974


namespace NUMINAMATH_GPT_quadratic_roots_inverse_sum_l1259_125988

theorem quadratic_roots_inverse_sum (t q α β : ℝ) (h1 : α + β = t) (h2 : α * β = q) 
  (h3 : ∀ n : ℕ, n ≥ 1 → α^n + β^n = t) : (1 / α^2011 + 1 / β^2011) = 2 := 
by 
  sorry

end NUMINAMATH_GPT_quadratic_roots_inverse_sum_l1259_125988


namespace NUMINAMATH_GPT_max_min_product_l1259_125963

theorem max_min_product (A B : ℕ) (h : A + B = 100) : 
  (∃ (maxProd : ℕ), maxProd = 2500 ∧ (∀ (A B : ℕ), A + B = 100 → A * B ≤ maxProd)) ∧
  (∃ (minProd : ℕ), minProd = 0 ∧ (∀ (A B : ℕ), A + B = 100 → minProd ≤ A * B)) :=
by 
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_max_min_product_l1259_125963


namespace NUMINAMATH_GPT_chocolates_per_student_class_7B_l1259_125917

theorem chocolates_per_student_class_7B :
  (∃ (x : ℕ), 9 * x < 288 ∧ 10 * x > 300 ∧ x = 31) :=
by
  use 31
  -- proof steps omitted here
  sorry

end NUMINAMATH_GPT_chocolates_per_student_class_7B_l1259_125917


namespace NUMINAMATH_GPT_product_of_coordinates_of_D_l1259_125973

theorem product_of_coordinates_of_D (D : ℝ × ℝ) (N : ℝ × ℝ) (C : ℝ × ℝ) 
  (hN : N = (4, 3)) (hC : C = (5, -1)) (midpoint : N = ((C.1 + D.1) / 2, (C.2 + D.2) / 2)) :
  D.1 * D.2 = 21 :=
by
  sorry

end NUMINAMATH_GPT_product_of_coordinates_of_D_l1259_125973


namespace NUMINAMATH_GPT_tangent_line_at_1_2_l1259_125995

noncomputable def f (x : ℝ) : ℝ := x^3 - x^2 + x + 1

def f' (x : ℝ) : ℝ := 3 * x^2 - 2 * x + 1

def tangent_eq (x y : ℝ) : Prop := y = 2 * x

theorem tangent_line_at_1_2 : tangent_eq 1 2 :=
by
  have f_1 := 1
  have f'_1 := 2
  sorry

end NUMINAMATH_GPT_tangent_line_at_1_2_l1259_125995


namespace NUMINAMATH_GPT_subtraction_problem_digits_sum_l1259_125945

theorem subtraction_problem_digits_sum :
  ∃ (K L M N : ℕ), K < 10 ∧ L < 10 ∧ M < 10 ∧ N < 10 ∧ 
  ((6000 + K * 100 + 0 + L) - (900 + N * 10 + 4) = 2011) ∧ 
  (K + L + M + N = 17) :=
by
  sorry

end NUMINAMATH_GPT_subtraction_problem_digits_sum_l1259_125945


namespace NUMINAMATH_GPT_combined_proposition_range_l1259_125941

def p (a : ℝ) : Prop := ∀ x ∈ ({1, 2} : Set ℝ), 3 * x^2 - a ≥ 0

def q (a : ℝ) : Prop := ∃ x : ℝ, x^2 + 2 * a * x + 2 - a = 0

theorem combined_proposition_range (a : ℝ) : 
  (p a ∧ q a) ↔ (a ≤ -2 ∨ (1 ≤ a ∧ a ≤ 3)) := 
  sorry

end NUMINAMATH_GPT_combined_proposition_range_l1259_125941


namespace NUMINAMATH_GPT_nonneg_reals_inequality_l1259_125959

theorem nonneg_reals_inequality (a b c d : ℝ) (h₀ : 0 ≤ a) (h₁ : 0 ≤ b) (h₂ : 0 ≤ c) (h₃ : 0 ≤ d) (h₄ : a^2 + b^2 + c^2 + d^2 = 1) : 
  a + b + c + d - 1 ≥ 16 * a * b * c * d := 
by 
  sorry

end NUMINAMATH_GPT_nonneg_reals_inequality_l1259_125959


namespace NUMINAMATH_GPT_marian_balance_proof_l1259_125976

noncomputable def marian_new_balance : ℝ :=
  let initial_balance := 126.00
  let uk_purchase := 50.0
  let uk_discount := 0.10
  let uk_rate := 1.39
  let france_purchase := 70.0
  let france_discount := 0.15
  let france_rate := 1.18
  let japan_purchase := 10000.0
  let japan_discount := 0.05
  let japan_rate := 0.0091
  let towel_return := 45.0
  let interest_rate := 0.015
  let uk_usd := (uk_purchase * (1 - uk_discount)) * uk_rate
  let france_usd := (france_purchase * (1 - france_discount)) * france_rate
  let japan_usd := (japan_purchase * (1 - japan_discount)) * japan_rate
  let gas_usd := (uk_purchase / 2) * uk_rate
  let balance_before_interest := initial_balance + uk_usd + france_usd + japan_usd + gas_usd - towel_return
  let interest := balance_before_interest * interest_rate
  balance_before_interest + interest

theorem marian_balance_proof :
  abs (marian_new_balance - 340.00) < 1 :=
by
  sorry

end NUMINAMATH_GPT_marian_balance_proof_l1259_125976


namespace NUMINAMATH_GPT_binomial_coefficient_multiple_of_4_l1259_125946

theorem binomial_coefficient_multiple_of_4 :
  ∃ (S : Finset ℕ), (∀ k ∈ S, 0 ≤ k ∧ k ≤ 2014 ∧ (Nat.choose 2014 k) % 4 = 0) ∧ S.card = 991 :=
sorry

end NUMINAMATH_GPT_binomial_coefficient_multiple_of_4_l1259_125946


namespace NUMINAMATH_GPT_minimum_trains_needed_l1259_125943

theorem minimum_trains_needed (n : ℕ) (h : 50 * n >= 645) : n = 13 :=
by
  sorry

end NUMINAMATH_GPT_minimum_trains_needed_l1259_125943


namespace NUMINAMATH_GPT_at_least_one_not_less_than_two_l1259_125939

theorem at_least_one_not_less_than_two
  (x y z : ℝ) (h1 : 0 < x) (h2 : 0 < y) (h3 : 0 < z) :
  (2 ≤ (y / x + y / z)) ∨ (2 ≤ (z / x + z / y)) ∨ (2 ≤ (x / z + x / y)) :=
sorry

end NUMINAMATH_GPT_at_least_one_not_less_than_two_l1259_125939


namespace NUMINAMATH_GPT_right_triangle_inequality_l1259_125932

theorem right_triangle_inequality (a b c : ℝ) (h : a^2 + b^2 = c^2) :
  a^4 + b^4 < c^4 :=
by
  sorry

end NUMINAMATH_GPT_right_triangle_inequality_l1259_125932


namespace NUMINAMATH_GPT_remainder_130_div_k_l1259_125937

theorem remainder_130_div_k (k : ℕ) (h_positive : k > 0)
  (h_remainder : 84 % (k*k) = 20) : 
  130 % k = 2 := 
by sorry

end NUMINAMATH_GPT_remainder_130_div_k_l1259_125937


namespace NUMINAMATH_GPT_digit_appears_in_3n_l1259_125926

-- Define a function to check if a digit is in a number
def contains_digit (n : ℕ) (d : ℕ) : Prop :=
  ∃ k : ℕ, n / 10^k % 10 = d

-- Define the statement that n does not contain the digits 1, 2, or 9
def does_not_contain_1_2_9 (n : ℕ) : Prop :=
  ¬ (contains_digit n 1 ∨ contains_digit n 2 ∨ contains_digit n 9)

theorem digit_appears_in_3n (n : ℕ) (hn : 1 ≤ n) (h : does_not_contain_1_2_9 n) :
  contains_digit (3 * n) 1 ∨ contains_digit (3 * n) 2 ∨ contains_digit (3 * n) 9 :=
by
  sorry

end NUMINAMATH_GPT_digit_appears_in_3n_l1259_125926


namespace NUMINAMATH_GPT_abc_divisibility_l1259_125929

theorem abc_divisibility (a b c : ℕ) (h1 : 1 < a) (h2 : a < b) (h3 : b < c) : 
  (a - 1) * (b - 1) * (c - 1) ∣ (a * b * c - 1) ↔ (a = 2 ∧ b = 4 ∧ c = 8) ∨ (a = 3 ∧ b = 5 ∧ c = 15) :=
by {
  sorry  -- proof to be filled in
}

end NUMINAMATH_GPT_abc_divisibility_l1259_125929


namespace NUMINAMATH_GPT_range_of_a_l1259_125957

theorem range_of_a (a : ℝ) : 
  (∀ (x : ℝ) (θ : ℝ), 0 ≤ θ ∧ θ ≤ π / 2 → 
    (x + 3 + 2 * (Real.sin θ) * (Real.cos θ))^2 + (x + a * (Real.sin θ) + a * (Real.cos θ))^2 ≥ 1 / 8) → 
  a ≥ 7 / 2 ∨ a ≤ Real.sqrt 6 :=
sorry

end NUMINAMATH_GPT_range_of_a_l1259_125957


namespace NUMINAMATH_GPT_compare_A_B_C_l1259_125951

-- Define the expressions A, B, and C
def A : ℚ := (2010 / 2009) + (2010 / 2011)
def B : ℚ := (2010 / 2011) + (2012 / 2011)
def C : ℚ := (2011 / 2010) + (2011 / 2012)

-- The statement asserting A is the greatest
theorem compare_A_B_C : A > B ∧ A > C := by
  sorry

end NUMINAMATH_GPT_compare_A_B_C_l1259_125951


namespace NUMINAMATH_GPT_constant_seq_is_arith_not_always_geom_l1259_125978

theorem constant_seq_is_arith_not_always_geom (c : ℝ) (seq : ℕ → ℝ) (h : ∀ n, seq n = c) :
  (∀ n, seq (n + 1) - seq n = 0) ∧ (c = 0 ∨ (∀ n, seq (n + 1) / seq n = 1)) :=
by
  sorry

end NUMINAMATH_GPT_constant_seq_is_arith_not_always_geom_l1259_125978


namespace NUMINAMATH_GPT_problem_1_problem_2_l1259_125966

def A : Set ℝ := { x | x^2 - 3 * x + 2 < 0 }

def B (a : ℝ) : Set ℝ := { x | a - 1 < x ∧ x < 3 * a + 1 }

-- Problem 1
theorem problem_1 (a : ℝ) (h : a = 1 / 4) : A ∩ B a = { x | 1 < x ∧ x < 7 / 4 } :=
by
  rw [h]
  sorry

-- Problem 2
theorem problem_2 : (∀ x, A x → B a x) → ∀ a, 1 / 3 ≤ a ∧ a ≤ 2 :=
by
  sorry

end NUMINAMATH_GPT_problem_1_problem_2_l1259_125966


namespace NUMINAMATH_GPT_at_least_one_ge_two_l1259_125913

theorem at_least_one_ge_two (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
    max (a + 1/b) (max (b + 1/c) (c + 1/a)) ≥ 2 :=
sorry

end NUMINAMATH_GPT_at_least_one_ge_two_l1259_125913


namespace NUMINAMATH_GPT_bus_driver_hours_l1259_125924

theorem bus_driver_hours (h : ℕ) (regular_rate : ℕ) (extra_rate1 : ℕ) (extra_rate2 : ℕ) (total_earnings : ℕ)
  (h1 : regular_rate = 14)
  (h2 : extra_rate1 = (14 + (14 * 35 / 100)))
  (h3: extra_rate2 = (14 + (14 * 75 / 100)))
  (h4: total_earnings = 1230)
  (h5: total_earnings = 40 * regular_rate + 10 * extra_rate1 + (h - 50) * extra_rate2)
  (condition : 50 < h) :
  h = 69 :=
by
  sorry

end NUMINAMATH_GPT_bus_driver_hours_l1259_125924


namespace NUMINAMATH_GPT_triangle_identity_l1259_125970

theorem triangle_identity
  (A B C : ℝ) (a b c: ℝ)
  (h1: A + B + C = Real.pi)
  (h2: a = 2 * R * Real.sin A)
  (h3: b = 2 * R * Real.sin B)
  (h4: c = 2 * R * Real.sin C)
  (h5: Real.sin A = Real.sin B * Real.cos C + Real.cos B * Real.sin C) :
  (b * Real.cos C + c * Real.cos B) / a = 1 := 
  by 
  sorry

end NUMINAMATH_GPT_triangle_identity_l1259_125970


namespace NUMINAMATH_GPT_cube_edge_length_l1259_125969

theorem cube_edge_length (total_edge_length : ℕ) (num_edges : ℕ) (h1 : total_edge_length = 108) (h2 : num_edges = 12) : total_edge_length / num_edges = 9 := by 
  -- additional formal mathematical steps can follow here
  sorry

end NUMINAMATH_GPT_cube_edge_length_l1259_125969


namespace NUMINAMATH_GPT_smallest_base_for_101_l1259_125961

theorem smallest_base_for_101 : ∃ b : ℕ, b = 10 ∧ b ≤ 101 ∧ 101 < b^2 :=
by
  -- We state the simplest form of the theorem,
  -- then use the answer from the solution step.
  use 10
  sorry

end NUMINAMATH_GPT_smallest_base_for_101_l1259_125961


namespace NUMINAMATH_GPT_compute_cd_l1259_125989

-- Define the variables c and d as real numbers
variables (c d : ℝ)

-- Define the conditions
def condition1 : Prop := c + d = 10
def condition2 : Prop := c^3 + d^3 = 370

-- State the theorem we need to prove
theorem compute_cd (h1 : condition1 c d) (h2 : condition2 c d) : c * d = 21 :=
by
  sorry

end NUMINAMATH_GPT_compute_cd_l1259_125989


namespace NUMINAMATH_GPT_Xingyou_age_is_3_l1259_125981

theorem Xingyou_age_is_3 (x : ℕ) (h1 : x = x) (h2 : x + 3 = 2 * x) : x = 3 :=
by
  sorry

end NUMINAMATH_GPT_Xingyou_age_is_3_l1259_125981


namespace NUMINAMATH_GPT_true_proposition_among_options_l1259_125935

theorem true_proposition_among_options :
  (∀ (x y : ℝ), (x > |y|) → (x > y)) ∧
  (¬ (∀ (x : ℝ), (x > 1) → (x^2 > 1))) ∧
  (¬ (∀ (x : ℤ), (x = 1) → (x^2 + x - 2 = 0))) ∧
  (¬ (∀ (x : ℝ), (x^2 > 0) → (x > 1))) :=
by
  sorry

end NUMINAMATH_GPT_true_proposition_among_options_l1259_125935
