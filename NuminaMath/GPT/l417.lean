import Mathlib

namespace expression_of_quadratic_function_coordinates_of_vertex_l417_41768

def quadratic_function_through_points (a b : ℝ) : Prop :=
  (0 = a * (-3)^2 + b * (-3) + 3) ∧ (-5 = a * 2^2 + b * 2 + 3)

theorem expression_of_quadratic_function :
  ∃ a b : ℝ, quadratic_function_through_points a b ∧ ∀ x : ℝ, -x^2 - 2 * x + 3 = a * x^2 + b * x + 3 :=
by
  sorry

theorem coordinates_of_vertex :
  - (1 : ℝ) * (1 : ℝ) = (-1) / (2 * (-1)) ∧ 4 = -(1 - (-1) + 3) + 4 :=
by
  sorry

end expression_of_quadratic_function_coordinates_of_vertex_l417_41768


namespace geometric_progression_common_ratio_l417_41796

-- Definitions and theorems
variable {α : Type*} [OrderedCommRing α]

theorem geometric_progression_common_ratio
  (a : α) (r : α)
  (h_pos : a > 0)
  (h_geometric : ∀ n : ℕ, a * r^n = (a * r^(n + 1)) * (a * r^(n + 2))):
  r = 1 := by
  sorry

end geometric_progression_common_ratio_l417_41796


namespace nominal_rate_of_interest_l417_41720

theorem nominal_rate_of_interest
  (EAR : ℝ)
  (n : ℕ)
  (h_EAR : EAR = 0.0609)
  (h_n : n = 2) :
  ∃ i : ℝ, (1 + i / n)^n - 1 = EAR ∧ i = 0.059 := 
by 
  sorry

end nominal_rate_of_interest_l417_41720


namespace perpendicular_vectors_l417_41761

theorem perpendicular_vectors (b : ℝ) :
  (5 * b - 12 = 0) → b = 12 / 5 :=
by
  intro h
  sorry

end perpendicular_vectors_l417_41761


namespace sharpened_off_length_l417_41725

-- Define the conditions
def original_length : ℤ := 31
def length_after_sharpening : ℤ := 14

-- Define the theorem to prove the length sharpened off is 17 inches
theorem sharpened_off_length : original_length - length_after_sharpening = 17 := sorry

end sharpened_off_length_l417_41725


namespace exponent_of_two_gives_n_l417_41785

theorem exponent_of_two_gives_n (x: ℝ) (n: ℝ) (b: ℝ)
  (h1: n = 2 ^ x)
  (h2: n ^ b = 8)
  (h3: b = 12) : x = 3 / 12 :=
by
  sorry

end exponent_of_two_gives_n_l417_41785


namespace triangle_inequality_lt_l417_41783

theorem triangle_inequality_lt {a b c : ℝ} (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (h1 : a < b + c) (h2 : b < a + c) (h3 : c < a + b) : a^2 + b^2 + c^2 < 2 * (a*b + b*c + c*a) := 
sorry

end triangle_inequality_lt_l417_41783


namespace chess_tournament_participants_l417_41762

theorem chess_tournament_participants (n : ℕ) (h : n * (n - 1) / 2 = 120) : n = 16 :=
sorry

end chess_tournament_participants_l417_41762


namespace fraction_meaningful_cond_l417_41799

theorem fraction_meaningful_cond (x : ℝ) : (x + 2 ≠ 0) ↔ (x ≠ -2) := 
by
  sorry

end fraction_meaningful_cond_l417_41799


namespace power_function_solution_l417_41767

theorem power_function_solution (m : ℝ) 
    (h1 : m^2 - 3 * m + 3 = 1) 
    (h2 : m - 1 ≠ 0) : m = 2 := 
by
  sorry

end power_function_solution_l417_41767


namespace area_percentage_change_l417_41795

variable (a b : ℝ)

def initial_area : ℝ := a * b

def new_length (a : ℝ) : ℝ := a * 1.35

def new_width (b : ℝ) : ℝ := b * 0.86

def new_area (a b : ℝ) : ℝ := (new_length a) * (new_width b)

theorem area_percentage_change :
    ((new_area a b) / (initial_area a b)) = 1.161 :=
by
  sorry

end area_percentage_change_l417_41795


namespace sum_of_maximum_and_minimum_of_u_l417_41772

theorem sum_of_maximum_and_minimum_of_u :
  ∀ (x y z : ℝ),
    0 ≤ x → 0 ≤ y → 0 ≤ z →
    3 * x + 2 * y + z = 5 →
    2 * x + y - 3 * z = 1 →
    3 * x + y - 7 * z = 3 * z - 2 →
    (-5 : ℝ) / 7 + (-1 : ℝ) / 11 = -62 / 77 :=
by
  sorry

end sum_of_maximum_and_minimum_of_u_l417_41772


namespace necessary_and_sufficient_condition_l417_41750

theorem necessary_and_sufficient_condition 
  (a : ℕ) 
  (A B : ℝ) 
  (x y z : ℤ) 
  (h1 : (x^2 + y^2 + z^2 : ℝ) = (B * ↑a)^2) 
  (h2 : (x^2 * (A * x^2 + B * y^2) + y^2 * (A * y^2 + B * z^2) + z^2 * (A * z^2 + B * x^2) : ℝ) = (1 / 4) * (2 * A + B) * (B * (↑a)^4)) :
  B = 2 * A :=
by
  sorry

end necessary_and_sufficient_condition_l417_41750


namespace find_angle_D_l417_41718

theorem find_angle_D (A B C D E F : ℝ) (hA : A = 50) (hB : B = 35) (hC : C = 40) 
  (triangle_sum1 : A + B + C + E + F = 180) (triangle_sum2 : D + E + F = 180) : 
  D = 125 :=
by
  -- Only adding a comment, proof omitted for the purpose of this task
  sorry

end find_angle_D_l417_41718


namespace probability_of_3_black_face_cards_l417_41728

-- Definitions based on conditions
def total_cards : ℕ := 36
def total_black_face_cards : ℕ := 8
def total_other_cards : ℕ := total_cards - total_black_face_cards
def draw_cards : ℕ := 6
def draw_black_face_cards : ℕ := 3
def draw_other_cards := draw_cards - draw_black_face_cards

-- Calculation using combinations
noncomputable def combination (n k : ℕ) : ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

noncomputable def total_combinations : ℕ := combination total_cards draw_cards
noncomputable def favorable_combinations : ℕ := combination total_black_face_cards draw_black_face_cards * combination total_other_cards draw_other_cards

-- Calculating probability
noncomputable def probability : ℚ := favorable_combinations / total_combinations

-- The theorem to be proved
theorem probability_of_3_black_face_cards : probability = 11466 / 121737 := by
  -- proof
  sorry

end probability_of_3_black_face_cards_l417_41728


namespace line_equation_l417_41745

theorem line_equation (a : ℝ) (P : ℝ × ℝ) (hx : P = (5, 6)) 
                      (cond : (a ≠ 0) ∧ (2 * a = 17)) : 
  ∃ (m b : ℝ), - (m * (0 : ℝ) + b) = a ∧ (- m * 17 / 2 + b) = 6 ∧ 
               (x + 2 * y - 17 =  0) := sorry

end line_equation_l417_41745


namespace normal_time_to_finish_bs_l417_41770

theorem normal_time_to_finish_bs (P : ℕ) (H1 : P = 5) (H2 : ∀ total_time, total_time = 6 → total_time = (3 / 4) * (P + B)) : B = (8 - P) :=
by sorry

end normal_time_to_finish_bs_l417_41770


namespace day_after_60_days_is_monday_l417_41716

theorem day_after_60_days_is_monday
    (birthday_is_thursday : ∃ d : ℕ, d % 7 = 0) :
    ∃ d : ℕ, (d + 60) % 7 = 4 :=
by
  -- Proof steps are omitted here
  sorry

end day_after_60_days_is_monday_l417_41716


namespace union_of_intervals_l417_41727

theorem union_of_intervals :
  let M := {x : ℝ | x^2 - 3 * x - 4 ≤ 0}
  let N := {x : ℝ | x^2 - 16 ≤ 0}
  M ∪ N = {x : ℝ | -4 ≤ x ∧ x ≤ 4} :=
by
  sorry

end union_of_intervals_l417_41727


namespace number_in_pattern_l417_41755

theorem number_in_pattern (m n : ℕ) (h : 8 * m - 5 = 2023) (hn : n = 5) : m + n = 258 :=
by
  sorry

end number_in_pattern_l417_41755


namespace find_digit_A_l417_41724

theorem find_digit_A (A : ℕ) (h1 : 0 ≤ A ∧ A ≤ 9) (h2 : (2 + A + 3 + A) % 9 = 0) : A = 2 :=
by
  sorry

end find_digit_A_l417_41724


namespace cuboid_length_l417_41756

theorem cuboid_length (A b h : ℝ) (A_eq : A = 2400) (b_eq : b = 10) (h_eq : h = 16) :
    ∃ l : ℝ, 2 * (l * b + b * h + h * l) = A ∧ l = 40 := by
  sorry

end cuboid_length_l417_41756


namespace product_of_slope_and_intercept_l417_41788

theorem product_of_slope_and_intercept {x1 y1 x2 y2 : ℝ} (h1 : x1 = -4) (h2 : y1 = -2) (h3 : x2 = 1) (h4 : y2 = 3) :
  let m := (y2 - y1) / (x2 - x1)
  let b := y1 - m * x1
  m * b = 2 :=
by
  sorry

end product_of_slope_and_intercept_l417_41788


namespace cosine_evaluation_l417_41754

variable (α : ℝ)

theorem cosine_evaluation
  (h : Real.sin (Real.pi / 6 + α) = 1 / 3) :
  Real.cos (Real.pi / 3 - α) = 1 / 3 :=
sorry

end cosine_evaluation_l417_41754


namespace simplify_sqrt_l417_41779

theorem simplify_sqrt (x : ℝ) (h : x = (Real.sqrt 3) + 1) : Real.sqrt (x^2) = Real.sqrt 3 + 1 :=
by
  -- This will serve as the placeholder for the proof.
  sorry

end simplify_sqrt_l417_41779


namespace five_colored_flags_l417_41748

def num_different_flags (colors total_stripes : ℕ) : ℕ :=
  Nat.choose colors total_stripes * Nat.factorial total_stripes

theorem five_colored_flags : num_different_flags 11 5 = 55440 := by
  sorry

end five_colored_flags_l417_41748


namespace inequality_proof_l417_41782

theorem inequality_proof (x y : ℝ) (n : ℕ) (hx : 0 < x) (hy : 0 < y) (hn : 0 < n):
  x^n / (1 + x^2) + y^n / (1 + y^2) ≤ (x^n + y^n) / (1 + x * y) :=
by
  sorry

end inequality_proof_l417_41782


namespace maximum_distance_l417_41721

noncomputable def point_distance (x1 y1 x2 y2 : ℝ) : ℝ := 
  Real.sqrt ((x1 - x2)^2 + (y1 - y2)^2)

def square_side_length := 2

def distance_condition (u v w : ℝ) : Prop := 
  u^2 + v^2 = 2 * w^2

theorem maximum_distance 
  (x y : ℝ) 
  (h1 : point_distance x y 0 0 = u) 
  (h2 : point_distance x y 2 0 = v) 
  (h3 : point_distance x y 2 2 = w)
  (h4 : distance_condition u v w) :
  ∃ (d : ℝ), d = point_distance x y 0 2 ∧ d = 2 * Real.sqrt 5 := sorry

end maximum_distance_l417_41721


namespace square_of_cube_of_third_smallest_prime_l417_41719

-- Definition of the third smallest prime number
def third_smallest_prime : Nat := 5

-- Definition of the cube of a number
def cube (n : Nat) : Nat := n ^ 3

-- Definition of the square of a number
def square (n : Nat) : Nat := n ^ 2

-- Theorem stating that the square of the cube of the third smallest prime number is 15625
theorem square_of_cube_of_third_smallest_prime : 
  square (cube third_smallest_prime) = 15625 := by 
  sorry

end square_of_cube_of_third_smallest_prime_l417_41719


namespace apples_in_basket_l417_41703

-- Definitions based on conditions
def total_apples : ℕ := 138
def apples_per_box : ℕ := 18

-- Problem: prove the number of apples in the basket
theorem apples_in_basket : (total_apples % apples_per_box) = 12 :=
by 
  -- Skip the proof part by adding sorry
  sorry

end apples_in_basket_l417_41703


namespace is_triangle_inequality_set_B_valid_triangle_set_A_not_triangle_set_C_not_triangle_set_D_not_triangle_l417_41773

theorem is_triangle_inequality (a b c: ℝ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

theorem set_B_valid_triangle :
  is_triangle_inequality 5 5 6 := by
  sorry

theorem set_A_not_triangle :
  ¬ is_triangle_inequality 7 4 2 := by
  sorry

theorem set_C_not_triangle :
  ¬ is_triangle_inequality 3 4 8 := by
  sorry

theorem set_D_not_triangle :
  ¬ is_triangle_inequality 2 3 5 := by
  sorry

end is_triangle_inequality_set_B_valid_triangle_set_A_not_triangle_set_C_not_triangle_set_D_not_triangle_l417_41773


namespace point_B_represent_l417_41758

-- Given conditions
def point_A := -2
def units_moved := 4

-- Lean statement to prove
theorem point_B_represent : 
  ∃ B : ℤ, (B = point_A - units_moved) ∨ (B = point_A + units_moved) := by
    sorry

end point_B_represent_l417_41758


namespace sum_of_terms_arithmetic_sequence_l417_41746

variable {S : ℕ → ℕ}
variable {k : ℕ}

-- Given conditions
axiom S_k : S k = 2
axiom S_3k : S (3 * k) = 18

-- The statement to prove
theorem sum_of_terms_arithmetic_sequence : S (4 * k) = 32 := by
  sorry

end sum_of_terms_arithmetic_sequence_l417_41746


namespace find_multiplier_value_l417_41749

def number : ℤ := 18
def increase : ℤ := 198

theorem find_multiplier_value (x : ℤ) (h : number * x = number + increase) : x = 12 :=
by
  sorry

end find_multiplier_value_l417_41749


namespace xyz_product_condition_l417_41707

theorem xyz_product_condition (x y z : ℝ) (h : x^2 + y^2 = x * y * (z + 1 / z)) : 
  x = y * z ∨ y = x * z :=
sorry

end xyz_product_condition_l417_41707


namespace find_n_in_geometric_series_l417_41787

theorem find_n_in_geometric_series (a : ℕ → ℕ) (S : ℕ → ℕ) (n : ℕ) :
  a 1 = 2 →
  (∀ k, a (k + 1) = 2 * a k) →
  S n = 126 →
  S n = a 1 * (2^n - 1) / (2 - 1) →
  n = 6 :=
by
  intros h₁ h₂ h₃ h₄
  sorry

end find_n_in_geometric_series_l417_41787


namespace man_speed_was_5_kmph_l417_41775

theorem man_speed_was_5_kmph (time_in_minutes : ℕ) (distance_in_km : ℝ)
  (h_time : time_in_minutes = 30)
  (h_distance : distance_in_km = 2.5) :
  (distance_in_km / (time_in_minutes / 60 : ℝ) = 5) :=
by
  sorry

end man_speed_was_5_kmph_l417_41775


namespace greatest_divisor_remainders_l417_41740

theorem greatest_divisor_remainders (x : ℕ) (h1 : 1255 % x = 8) (h2 : 1490 % x = 11) : x = 29 :=
by
  -- The proof steps would go here, but for now, we use sorry.
  sorry

end greatest_divisor_remainders_l417_41740


namespace ratio_a6_b6_l417_41793

-- Definitions for sequences and sums
variable {α : Type*} [LinearOrderedField α] 
variable (a b : ℕ → α) 
variable (S T : ℕ → α)

-- Main theorem stating the problem
theorem ratio_a6_b6 (h : ∀ n, S n / T n = (2 * n - 5) / (4 * n + 3)) :
    a 6 / b 6 = 17 / 47 :=
sorry

end ratio_a6_b6_l417_41793


namespace sqrt_number_is_169_l417_41764

theorem sqrt_number_is_169 (a b : ℝ) 
  (h : a^2 + b^2 + (4 * a - 6 * b + 13) = 0) : 
  (a^2 + b^2)^2 = 169 :=
sorry

end sqrt_number_is_169_l417_41764


namespace car_speed_first_hour_l417_41706

theorem car_speed_first_hour (x : ℝ) (h1 : (x + 75) / 2 = 82.5) : x = 90 :=
sorry

end car_speed_first_hour_l417_41706


namespace extremum_at_x_1_max_integer_k_l417_41711

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := (x - 1) * Real.log x - (a + 1) * x

theorem extremum_at_x_1 (a : ℝ) : (∀ x : ℝ, 0 < x → ((Real.log x - 1 / x - a = 0) ↔ x = 1))
  → a = -1 ∧
  (∀ x : ℝ, 0 < x → (Real.log x - 1 / x + 1) < 0 → f x (-1) < f 1 (-1) ∧
  (Real.log x - 1 / x + 1) > 0 → f 1 (-1) < f x (-1)) :=
sorry

theorem max_integer_k (k : ℤ) :
  (∀ x : ℝ, 0 < x → (f x 1 > k))
  → k ≤ -4 :=
sorry

end extremum_at_x_1_max_integer_k_l417_41711


namespace simplify_f_value_of_f_l417_41766

noncomputable def f (α : ℝ) : ℝ :=
  (Real.sin (α - (5 * Real.pi) / 2) * Real.cos ((3 * Real.pi) / 2 + α) * Real.tan (Real.pi - α)) /
  (Real.tan (-α - Real.pi) * Real.sin (Real.pi - α))

theorem simplify_f (α : ℝ) : f α = -Real.cos α := by
  sorry

theorem value_of_f (α : ℝ)
  (h : Real.cos (α + (3 * Real.pi) / 2) = 1 / 5)
  (h2 : α > Real.pi / 2 ∧ α < Real.pi ) : 
  f α = 2 * Real.sqrt 6 / 5 := by
  sorry

end simplify_f_value_of_f_l417_41766


namespace Jack_hands_in_l417_41759

def num_hundred_bills := 2
def num_fifty_bills := 1
def num_twenty_bills := 5
def num_ten_bills := 3
def num_five_bills := 7
def num_one_bills := 27
def to_leave_in_till := 300

def total_money_in_notes : Nat :=
  (num_hundred_bills * 100) +
  (num_fifty_bills * 50) +
  (num_twenty_bills * 20) +
  (num_ten_bills * 10) +
  (num_five_bills * 5) +
  (num_one_bills * 1)

def money_to_hand_in := total_money_in_notes - to_leave_in_till

theorem Jack_hands_in : money_to_hand_in = 142 := by
  sorry

end Jack_hands_in_l417_41759


namespace prime_in_range_l417_41742

theorem prime_in_range (p: ℕ) (h_prime: Nat.Prime p) (h_int_roots: ∃ a b: ℤ, a ≠ b ∧ a + b = -p ∧ a * b = -520 * p) : 11 < p ∧ p ≤ 21 := 
by
  sorry

end prime_in_range_l417_41742


namespace days_in_month_l417_41734

theorem days_in_month
  (monthly_production : ℕ)
  (production_per_half_hour : ℚ)
  (hours_per_day : ℕ)
  (daily_production : ℚ)
  (days_in_month : ℚ) :
  monthly_production = 8400 ∧
  production_per_half_hour = 6.25 ∧
  hours_per_day = 24 ∧
  daily_production = production_per_half_hour * 2 * hours_per_day ∧
  days_in_month = monthly_production / daily_production
  → days_in_month = 28 :=
by
  sorry

end days_in_month_l417_41734


namespace problem_statement_l417_41736

-- Given: x, y, z are real numbers such that x < 0 and x < y < z
variables {x y z : ℝ} 

-- Conditions
axiom h1 : x < 0
axiom h2 : x < y
axiom h3 : y < z

-- Statement to prove: x + y < y + z
theorem problem_statement : x + y < y + z :=
by {
  sorry
}

end problem_statement_l417_41736


namespace smallest_integer_CC4_DD6_rep_l417_41757

-- Lean 4 Statement
theorem smallest_integer_CC4_DD6_rep (C D : ℕ) (hC : C < 4) (hD : D < 6) :
  (5 * C = 7 * D) → (5 * C = 35 ∧ 7 * D = 35) :=
by
  sorry

end smallest_integer_CC4_DD6_rep_l417_41757


namespace average_percentage_decrease_l417_41778

theorem average_percentage_decrease (x : ℝ) : 60 * (1 - x) * (1 - x) = 48.6 → x = 0.1 :=
by sorry

end average_percentage_decrease_l417_41778


namespace cannot_make_62_cents_with_five_coins_l417_41780

theorem cannot_make_62_cents_with_five_coins :
  ∀ (p n d q : ℕ), p + n + d + q = 5 ∧ q ≤ 1 →
  1 * p + 5 * n + 10 * d + 25 * q ≠ 62 := by
  intro p n d q h
  sorry

end cannot_make_62_cents_with_five_coins_l417_41780


namespace negation_example_l417_41784

theorem negation_example : ¬(∀ x : ℝ, x > 1 → x^2 > 1) ↔ ∃ x : ℝ, x > 1 ∧ x^2 ≤ 1 := by
  sorry

end negation_example_l417_41784


namespace correct_location_l417_41763

-- Define the possible options
inductive Location
| A : Location
| B : Location
| C : Location
| D : Location

-- Define the conditions
def option_A : Prop := ¬(∃ d, d ≠ "right")
def option_B : Prop := ¬(∃ d, d ≠ 900)
def option_C : Prop := ¬(∃ d, d ≠ "west")
def option_D : Prop := (∃ d₁ d₂, d₁ = "west" ∧ d₂ = 900)

-- The objective is to prove that option D is the correct description of the location
theorem correct_location : ∃ l, l = Location.D → 
  (option_A ∧ option_B ∧ option_C ∧ option_D) :=
by
  sorry

end correct_location_l417_41763


namespace mia_days_not_worked_l417_41704

theorem mia_days_not_worked :
  ∃ (y : ℤ), (∃ (x : ℤ), 
  x + y = 30 ∧ 80 * x - 40 * y = 1600) ∧ y = 20 :=
by
  sorry

end mia_days_not_worked_l417_41704


namespace vikki_worked_42_hours_l417_41760

-- Defining the conditions
def hourly_pay_rate : ℝ := 10
def tax_deduction : ℝ := 0.20 * hourly_pay_rate
def insurance_deduction : ℝ := 0.05 * hourly_pay_rate
def union_dues : ℝ := 5
def take_home_pay : ℝ := 310

-- Equation derived from the given conditions
def total_hours_worked (h : ℝ) : Prop :=
  hourly_pay_rate * h - (tax_deduction * h + insurance_deduction * h + union_dues) = take_home_pay

-- Prove that Vikki worked for 42 hours given the conditions
theorem vikki_worked_42_hours : total_hours_worked 42 := by
  sorry

end vikki_worked_42_hours_l417_41760


namespace three_distinct_solutions_no_solution_for_2009_l417_41713

-- Problem 1: Show that the equation has at least three distinct solutions if it has one
theorem three_distinct_solutions (n : ℕ) (hn : n > 0) :
  (∃ x y : ℤ, x^3 - 3*x*y^2 + y^3 = n) →
  (∃ (x1 y1 x2 y2 x3 y3 : ℤ), 
    x1^3 - 3*x1*y1^2 + y1^3 = n ∧ 
    x2^3 - 3*x2*y2^2 + y2^3 = n ∧ 
    x3^3 - 3*x3*y3^2 + y3^3 = n ∧ 
    (x1, y1) ≠ (x2, y2) ∧ 
    (x1, y1) ≠ (x3, y3) ∧ 
    (x2, y2) ≠ (x3, y3)) :=
sorry

-- Problem 2: Show that the equation has no solutions when n = 2009
theorem no_solution_for_2009 :
  ¬ ∃ x y : ℤ, x^3 - 3*x*y^2 + y^3 = 2009 :=
sorry

end three_distinct_solutions_no_solution_for_2009_l417_41713


namespace largest_two_digit_n_l417_41753

theorem largest_two_digit_n (x : ℕ) (n : ℕ) (hx : x < 10) (hx_nonzero : 0 < x)
  (hn : n = 12 * x * x) (hn_two_digit : n < 100) : n = 48 :=
by sorry

end largest_two_digit_n_l417_41753


namespace rectangle_area_increase_l417_41751

theorem rectangle_area_increase (a b : ℝ) :
  let new_length := (1 + 1/4) * a
  let new_width := (1 + 1/5) * b
  let original_area := a * b
  let new_area := new_length * new_width
  let area_increase := new_area - original_area
  (area_increase / original_area) = 1/2 := 
by
  sorry

end rectangle_area_increase_l417_41751


namespace prob_top_odd_correct_l417_41791

def total_dots : Nat := 78
def faces : List Nat := [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

-- Probability calculation for odd dots after removal
def prob_odd_dot (n : Nat) : Rat :=
  if n % 2 = 1 then
    1 - (n : Rat) / total_dots
  else
    (n : Rat) / total_dots

-- Probability that the top face shows an odd number of dots
noncomputable def prob_top_odd : Rat :=
  (1 / (faces.length : Rat)) * (faces.map prob_odd_dot).sum

theorem prob_top_odd_correct :
  prob_top_odd = 523 / 936 :=
by
  sorry

end prob_top_odd_correct_l417_41791


namespace polygon_proof_l417_41739

-- Define the conditions and the final proof problem.
theorem polygon_proof 
  (interior_angle : ℝ) 
  (side_length : ℝ) 
  (h1 : interior_angle = 160) 
  (h2 : side_length = 4) 
  : ∃ n : ℕ, ∃ P : ℝ, (interior_angle = 180 * (n - 2) / n) ∧ (P = n * side_length) ∧ (n = 18) ∧ (P = 72) :=
by
  sorry

end polygon_proof_l417_41739


namespace total_students_l417_41786

theorem total_students (T : ℝ) (h : 0.50 * T = 440) : T = 880 := 
by {
  sorry
}

end total_students_l417_41786


namespace max_value_k_l417_41743

theorem max_value_k (x y : ℝ) (k : ℝ) (h₁ : x^2 + y^2 = 1) (h₂ : ∀ x y, x^2 + y^2 = 1 → x + y - k ≥ 0) : 
  k ≤ -Real.sqrt 2 :=
sorry

end max_value_k_l417_41743


namespace f_bounds_l417_41738

-- Define the function f with the given properties
def f : ℝ → ℝ :=
sorry 

-- Specify the conditions on f
axiom f_0 : f 0 = 0
axiom f_1 : f 1 = 1
axiom f_ratio (x y z : ℝ) (h1 : 0 ≤ x) (h2 : x < y) (h3 : y < z) (h4 : z ≤ 1) 
  (h5 : z - y = y - x) : 1/2 ≤ (f z - f y) / (f y - f x) ∧ (f z - f y) / (f y - f x) ≤ 2

-- State the theorem to be proven
theorem f_bounds : 1 / 7 ≤ f (1 / 3) ∧ f (1 / 3) ≤ 4 / 7 :=
sorry

end f_bounds_l417_41738


namespace least_integer_square_eq_double_plus_64_l417_41726

theorem least_integer_square_eq_double_plus_64 :
  ∃ x : ℤ, x^2 = 2 * x + 64 ∧ ∀ y : ℤ, y^2 = 2 * y + 64 → y ≥ x → x = -8 :=
by
  sorry

end least_integer_square_eq_double_plus_64_l417_41726


namespace croissants_for_breakfast_l417_41732

def total_items (C : ℕ) : Prop :=
  C + 18 + 30 = 110

theorem croissants_for_breakfast (C : ℕ) (h : total_items C) : C = 62 :=
by {
  -- The proof might be here, but since it's not required:
  sorry
}

end croissants_for_breakfast_l417_41732


namespace part1_part2_l417_41701

variable {R : Type} [LinearOrderedField R]

def f (x : R) : R := abs (x - 2) + 2
def g (m : R) (x : R) : R := m * abs x

theorem part1 (x : R) : f x > 5 ↔ x < -1 ∨ x > 5 := by
  sorry

theorem part2 (m : R) : (∀ x : R, f x ≥ g m x) → m ∈ Set.Iic (1 : R) := by
  sorry

end part1_part2_l417_41701


namespace ball_travel_distance_l417_41737

theorem ball_travel_distance 
    (initial_height : ℕ)
    (half : ℕ → ℕ)
    (num_bounces : ℕ)
    (height_after_bounce : ℕ → ℕ)
    (total_distance : ℕ) :
    initial_height = 16 ∧ 
    (∀ n, half n = n / 2) ∧ 
    num_bounces = 4 ∧ 
    (height_after_bounce 0 = initial_height) ∧
    (∀ n, height_after_bounce (n + 1) = half (height_after_bounce n))
→ total_distance = 46 :=
by
  sorry

end ball_travel_distance_l417_41737


namespace shara_savings_l417_41798

theorem shara_savings 
  (original_price : ℝ)
  (discount1 : ℝ := 0.08)
  (discount2 : ℝ := 0.05)
  (sales_tax : ℝ := 0.06)
  (final_price : ℝ := 184)
  (h : (original_price * (1 - discount1) * (1 - discount2) * (1 + sales_tax)) = final_price) :
  original_price - final_price = 25.78 :=
sorry

end shara_savings_l417_41798


namespace bowling_average_change_l417_41792

theorem bowling_average_change (old_avg : ℝ) (wickets_last : ℕ) (runs_last : ℕ) (wickets_before : ℕ)
  (h_old_avg : old_avg = 12.4)
  (h_wickets_last : wickets_last = 8)
  (h_runs_last : runs_last = 26)
  (h_wickets_before : wickets_before = 175) :
  old_avg - ((old_avg * wickets_before + runs_last)/(wickets_before + wickets_last)) = 0.4 :=
by {
  sorry
}

end bowling_average_change_l417_41792


namespace ratio_of_coefficients_l417_41714

theorem ratio_of_coefficients (x y c d : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hd : d ≠ 0)
  (H1 : 8 * x - 6 * y = c) (H2 : 12 * y - 18 * x = d) :
  c / d = -4 / 9 := 
by {
  sorry
}

end ratio_of_coefficients_l417_41714


namespace algebraic_simplification_evaluate_expression_for_x2_evaluate_expression_for_x_neg2_l417_41774

theorem algebraic_simplification (x : ℤ) (h1 : -3 < x) (h2 : x < 3) (h3 : x ≠ 0) (h4 : x ≠ 1) (h5 : x ≠ -1) :
  (x - (x / (x + 1))) / (1 + (1 / (x^2 - 1))) = x - 1 :=
sorry

theorem evaluate_expression_for_x2 (h1 : -3 < 2) (h2 : 2 < 3) (h3 : 2 ≠ 0) (h4 : 2 ≠ 1) (h5 : 2 ≠ -1) :
  (2 - (2 / (2 + 1))) / (1 + (1 / (2^2 - 1))) = 1 :=
sorry

theorem evaluate_expression_for_x_neg2 (h1 : -3 < -2) (h2 : -2 < 3) (h3 : -2 ≠ 0) (h4 : -2 ≠ 1) (h5 : -2 ≠ -1) :
  (-2 - (-2 / (-2 + 1))) / (1 + (1 / ((-2)^2 - 1))) = -3 :=
sorry

end algebraic_simplification_evaluate_expression_for_x2_evaluate_expression_for_x_neg2_l417_41774


namespace height_of_triangle_l417_41705

-- Define the dimensions of the rectangle
variable (l w : ℝ)

-- Assume the base of the triangle is equal to the length of the rectangle
-- We need to prove that the height of the triangle h = 2w

theorem height_of_triangle (h : ℝ) (hl_eq_length : l > 0) (hw_eq_width : w > 0) :
  (l * w) = (1 / 2) * l * h → h = 2 * w :=
by
  sorry

end height_of_triangle_l417_41705


namespace pizza_slices_l417_41700

theorem pizza_slices (total_slices pepperoni_slices mushroom_slices : ℕ) 
  (h_total : total_slices = 24)
  (h_pepperoni : pepperoni_slices = 15)
  (h_mushrooms : mushroom_slices = 16)
  (h_at_least_one : total_slices = pepperoni_slices + mushroom_slices - both_slices)
  : both_slices = 7 :=
by
  have h1 : total_slices = 24 := h_total
  have h2 : pepperoni_slices = 15 := h_pepperoni
  have h3 : mushroom_slices = 16 := h_mushrooms
  have h4 : total_slices = 24 := by sorry
  sorry

end pizza_slices_l417_41700


namespace microphotonics_budget_allocation_l417_41731

theorem microphotonics_budget_allocation
    (home_electronics : ℕ)
    (food_additives : ℕ)
    (gen_mod_microorg : ℕ)
    (ind_lubricants : ℕ)
    (basic_astrophysics_degrees : ℕ)
    (full_circle_degrees : ℕ := 360)
    (total_budget_percentage : ℕ := 100)
    (basic_astrophysics_percentage : ℕ) :
  home_electronics = 24 →
  food_additives = 15 →
  gen_mod_microorg = 19 →
  ind_lubricants = 8 →
  basic_astrophysics_degrees = 72 →
  basic_astrophysics_percentage = (basic_astrophysics_degrees * total_budget_percentage) / full_circle_degrees →
  (total_budget_percentage -
    (home_electronics + food_additives + gen_mod_microorg + ind_lubricants + basic_astrophysics_percentage)) = 14 :=
by
  intros he fa gmm il bad bp
  sorry

end microphotonics_budget_allocation_l417_41731


namespace maximum_value_of_a_l417_41735

theorem maximum_value_of_a (a : ℝ) : 
  (∀ x : ℝ, x^2 + |2 * x - 6| ≥ a) ↔ a ≤ 5 :=
by
  sorry

end maximum_value_of_a_l417_41735


namespace shirt_tie_combinations_l417_41709

noncomputable def shirts : ℕ := 8
noncomputable def ties : ℕ := 7
noncomputable def forbidden_combinations : ℕ := 2

theorem shirt_tie_combinations :
  shirts * ties - forbidden_combinations = 54 := by
  sorry

end shirt_tie_combinations_l417_41709


namespace more_girls_than_boys_l417_41797

def initial_girls : ℕ := 632
def initial_boys : ℕ := 410
def new_girls_joined : ℕ := 465
def total_girls : ℕ := initial_girls + new_girls_joined

theorem more_girls_than_boys :
  total_girls - initial_boys = 687 :=
by
  -- Proof goes here
  sorry


end more_girls_than_boys_l417_41797


namespace apple_crisps_calculation_l417_41765

theorem apple_crisps_calculation (apples crisps : ℕ) (h : crisps = 3 ∧ apples = 12) : 
  (36 / apples) * crisps = 9 := by
  sorry

end apple_crisps_calculation_l417_41765


namespace polygon_interior_angle_sum_l417_41729

theorem polygon_interior_angle_sum (n : ℕ) (h : 180 * (n - 2) = 2340) :
  180 * (n - 2 + 3) = 2880 := by
  sorry

end polygon_interior_angle_sum_l417_41729


namespace initial_boys_l417_41777

-- Define the initial condition
def initial_girls : ℕ := 18
def additional_girls : ℕ := 7
def quitting_boys : ℕ := 4
def total_children_after_changes : ℕ := 36

-- Define the initial number of boys
variable (B : ℕ)

-- State the main theorem
theorem initial_boys (h : 25 + (B - 4) = 36) : B = 15 :=
by
  sorry

end initial_boys_l417_41777


namespace statement_II_must_be_true_l417_41781

-- Define the set of all creatures
variable (Creature : Type)

-- Define properties for being a dragon, mystical, and fire-breathing
variable (Dragon Mystical FireBreathing : Creature → Prop)

-- Given conditions
-- All dragons breathe fire
axiom all_dragons_breathe_fire : ∀ c, Dragon c → FireBreathing c
-- Some mystical creatures are dragons
axiom some_mystical_creatures_are_dragons : ∃ c, Mystical c ∧ Dragon c

-- Questions to prove (we will only formalize the must be true statement)
-- Statement II: Some fire-breathing creatures are mystical creatures

theorem statement_II_must_be_true : ∃ c, FireBreathing c ∧ Mystical c :=
by
  sorry

end statement_II_must_be_true_l417_41781


namespace freds_total_marbles_l417_41730

theorem freds_total_marbles :
  let red := 38
  let green := red / 2
  let dark_blue := 6
  red + green + dark_blue = 63 := by
  sorry

end freds_total_marbles_l417_41730


namespace radius_range_of_circle_l417_41741

theorem radius_range_of_circle (r : ℝ) :
  (∃ (x y : ℝ), (x - 3)^2 + (y + 5)^2 = r^2 ∧ 
    (∃ a b : ℝ, 4*a - 3*b - 2 = 0 ∧ ∃ c d : ℝ, 4*c - 3*d - 2 = 0 ∧ 
      (a - x)^2 + (b - y)^2 = 1 ∧ (c - x)^2 + (d - y)^2 = 1 ∧
       a ≠ c ∧ b ≠ d)) ↔ 4 < r ∧ r < 6 :=
by
  sorry

end radius_range_of_circle_l417_41741


namespace dog_adult_weight_l417_41776

theorem dog_adult_weight 
  (w7 : ℕ) (w7_eq : w7 = 6)
  (w9 : ℕ) (w9_eq : w9 = 2 * w7)
  (w3m : ℕ) (w3m_eq : w3m = 2 * w9)
  (w5m : ℕ) (w5m_eq : w5m = 2 * w3m)
  (w1y : ℕ) (w1y_eq : w1y = w5m + 30) :
  w1y = 78 := by
  -- Proof is not required, so we leave it with sorry.
  sorry

end dog_adult_weight_l417_41776


namespace proof_problem_l417_41710

open Set Real

def M : Set ℝ := { x : ℝ | ∃ y : ℝ, y = log (1 - 2 / x) }
def N : Set ℝ := { x : ℝ | ∃ y : ℝ, y = sqrt (x - 1) }

theorem proof_problem : N ∩ (U \ M) = Icc 1 2 := by
  sorry

end proof_problem_l417_41710


namespace find_k_for_quadratic_has_one_real_root_l417_41702

theorem find_k_for_quadratic_has_one_real_root (k : ℝ) : 
  (∃ x : ℝ, (3 * x - 4) * (x + 6) = -53 + k * x) ↔ (k = 14 + 2 * Real.sqrt 87 ∨ k = 14 - 2 * Real.sqrt 87) :=
sorry

end find_k_for_quadratic_has_one_real_root_l417_41702


namespace product_of_primes_l417_41717

theorem product_of_primes :
  (7 * 97 * 89) = 60431 :=
by
  sorry

end product_of_primes_l417_41717


namespace largest_integer_remainder_condition_l417_41708

theorem largest_integer_remainder_condition (number : ℤ) (h1 : number < 100) (h2 : number % 7 = 4) :
  number = 95 := sorry

end largest_integer_remainder_condition_l417_41708


namespace value_of_fraction_l417_41723

theorem value_of_fraction (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0)
  (h : (b / c = 2005) ∧ (c / b = 2005)) : (b + c) / (a + b) = 2005 :=
by
  sorry

end value_of_fraction_l417_41723


namespace k_plus_a_equals_three_halves_l417_41769

theorem k_plus_a_equals_three_halves :
  ∃ (k a : ℝ), (2 = k * 4 ^ a) ∧ (k + a = 3 / 2) :=
sorry

end k_plus_a_equals_three_halves_l417_41769


namespace angie_age_l417_41733

variables (age : ℕ)

theorem angie_age (h : 2 * age + 4 = 20) : age = 8 :=
sorry

end angie_age_l417_41733


namespace sufficient_condition_l417_41715

theorem sufficient_condition (a b c : ℤ) : (a = c + 1) → (b = a - 1) → a * (a - b) + b * (b - c) + c * (c - a) = 2 :=
by
  intros h1 h2
  sorry

end sufficient_condition_l417_41715


namespace Bruno_wants_2_5_dozens_l417_41712

theorem Bruno_wants_2_5_dozens (total_pens : ℕ) (dozen_pens : ℕ) (h_total_pens : total_pens = 30) (h_dozen_pens : dozen_pens = 12) : (total_pens / dozen_pens : ℚ) = 2.5 :=
by 
  sorry

end Bruno_wants_2_5_dozens_l417_41712


namespace sum_of_three_integers_mod_53_l417_41790

theorem sum_of_three_integers_mod_53 (a b c : ℕ) (h1 : a % 53 = 31) 
                                     (h2 : b % 53 = 22) (h3 : c % 53 = 7) : 
                                     (a + b + c) % 53 = 7 :=
by
  sorry

end sum_of_three_integers_mod_53_l417_41790


namespace Lyka_savings_l417_41789

def Smartphone_cost := 800
def Initial_savings := 200
def Gym_cost_per_month := 50
def Total_months := 4
def Weeks_per_month := 4
def Savings_per_week_initial := 50
def Savings_per_week_after_raise := 80

def Total_savings : Nat :=
  let initial_savings := Savings_per_week_initial * Weeks_per_month * 2
  let increased_savings := Savings_per_week_after_raise * Weeks_per_month * 2
  initial_savings + increased_savings

theorem Lyka_savings :
  (Initial_savings + Total_savings) = 1040 := by
  sorry

end Lyka_savings_l417_41789


namespace expression_evaluation_l417_41747

theorem expression_evaluation :
  2 - 3 * (-4) + 5 - (-6) * 7 = 61 :=
sorry

end expression_evaluation_l417_41747


namespace findPrincipalAmount_l417_41752

noncomputable def principalAmount (r : ℝ) (t : ℝ) (diff : ℝ) : ℝ :=
  let n := 2 -- compounded semi-annually
  let rate_per_period := (1 + r / n)
  let num_periods := n * t
  (diff / (rate_per_period^num_periods - 1 - r * t))

theorem findPrincipalAmount :
  let r := 0.05
  let t := 3
  let diff := 25
  abs (principalAmount r t diff - 2580.39) < 0.01 := 
by 
  sorry

end findPrincipalAmount_l417_41752


namespace distance_from_origin_to_line_l417_41794

def ellipse (x y : ℝ) : Prop := (x^2 / 2) + y^2 = 1

-- definition of the perpendicular property of chords
def perpendicular (O A B : ℝ × ℝ) : Prop := (A.1 * B.1 + A.2 * B.2 = 0)

theorem distance_from_origin_to_line
  (xA yA xB yB : ℝ)
  (hA : ellipse xA yA)
  (hB : ellipse xB yB)
  (h_perpendicular : perpendicular (0, 0) (xA, yA) (xB, yB))
  : ∃ d : ℝ, d = (Real.sqrt 6) / 3 :=
sorry

end distance_from_origin_to_line_l417_41794


namespace profit_at_end_of_first_year_l417_41744

theorem profit_at_end_of_first_year :
  let total_amount := 50000
  let part1 := 30000
  let interest_rate1 := 0.10
  let part2 := total_amount - part1
  let interest_rate2 := 0.20
  let time_period := 1
  let interest1 := part1 * interest_rate1 * time_period
  let interest2 := part2 * interest_rate2 * time_period
  let total_profit := interest1 + interest2
  total_profit = 7000 := 
by 
  sorry

end profit_at_end_of_first_year_l417_41744


namespace vertex_angle_of_obtuse_isosceles_triangle_l417_41722

noncomputable def isosceles_obtuse_triangle (a b h : ℝ) (φ : ℝ) : Prop :=
  a^2 = 2 * b * h ∧
  b = 2 * a * Real.cos ((180 - φ) / 2) ∧
  h = a * Real.sin ((180 - φ) / 2) ∧
  90 < φ ∧ φ < 180

theorem vertex_angle_of_obtuse_isosceles_triangle (a b h : ℝ) (φ : ℝ) :
  isosceles_obtuse_triangle a b h φ → φ = 150 :=
by
  sorry

end vertex_angle_of_obtuse_isosceles_triangle_l417_41722


namespace shift_parabola_two_units_right_l417_41771

-- Define the original parabola
def original_parabola (x : ℝ) : ℝ := x^2

-- Define the shift function
def shift (f : ℝ → ℝ) (h : ℝ) (x : ℝ) : ℝ := f (x - h)

-- Define the new parabola equation after shifting 2 units to the right
def shifted_parabola (x : ℝ) : ℝ := (x - 2)^2

-- The theorem stating that shifting the original parabola 2 units to the right equals the new parabola equation
theorem shift_parabola_two_units_right :
  ∀ x : ℝ, shift original_parabola 2 x = shifted_parabola x :=
by
  intros
  sorry

end shift_parabola_two_units_right_l417_41771
