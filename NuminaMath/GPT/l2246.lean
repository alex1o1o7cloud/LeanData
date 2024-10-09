import Mathlib

namespace vector_addition_in_triangle_l2246_224690

theorem vector_addition_in_triangle
  (A B C D : Type)
  [AddCommGroup A] [AddCommGroup B] [AddCommGroup C] [AddCommGroup D] 
  [Module ℝ A] [Module ℝ B] [Module ℝ C] [Module ℝ D]
  (AB AC AD BD DC : A)
  (h1 : BD = 2 • DC) :
  AD = (1/3 : ℝ) • AB + (2/3 : ℝ) • AC :=
sorry

end vector_addition_in_triangle_l2246_224690


namespace base9_to_decimal_l2246_224686

theorem base9_to_decimal : (8 * 9^1 + 5 * 9^0) = 77 := 
by
  sorry

end base9_to_decimal_l2246_224686


namespace total_money_amount_l2246_224630

-- Define the conditions
def num_bills : ℕ := 3
def value_per_bill : ℕ := 20
def initial_amount : ℕ := 75

-- Define the statement about the total amount of money James has
theorem total_money_amount : num_bills * value_per_bill + initial_amount = 135 := 
by 
  -- Since the proof is not required, we use 'sorry' to skip it
  sorry

end total_money_amount_l2246_224630


namespace olivia_money_left_l2246_224626

-- Defining hourly wages
def wage_monday : ℕ := 10
def wage_wednesday : ℕ := 12
def wage_friday : ℕ := 14
def wage_saturday : ℕ := 20

-- Defining hours worked each day
def hours_monday : ℕ := 5
def hours_wednesday : ℕ := 4
def hours_friday : ℕ := 3
def hours_saturday : ℕ := 2

-- Defining business-related expenses and tax rate
def expenses : ℕ := 50
def tax_rate : ℝ := 0.15

-- Calculate total earnings
def total_earnings : ℕ :=
  (hours_monday * wage_monday) +
  (hours_wednesday * wage_wednesday) +
  (hours_friday * wage_friday) +
  (hours_saturday * wage_saturday)

-- Earnings after expenses
def earnings_after_expenses : ℕ :=
  total_earnings - expenses

-- Calculate tax amount
def tax_amount : ℝ :=
  tax_rate * (total_earnings : ℝ)

-- Final amount Olivia has left
def remaining_amount : ℝ :=
  (earnings_after_expenses : ℝ) - tax_amount

theorem olivia_money_left : remaining_amount = 103 := by
  sorry

end olivia_money_left_l2246_224626


namespace fractional_sum_l2246_224673

noncomputable def greatest_integer (t : ℝ) : ℝ := ⌊t⌋
noncomputable def fractional_part (t : ℝ) : ℝ := t - greatest_integer t

theorem fractional_sum (x : ℝ) (h : x^3 + (1/x)^3 = 18) : 
  fractional_part x + fractional_part (1/x) = 1 :=
sorry

end fractional_sum_l2246_224673


namespace soda_cans_purchase_l2246_224683

noncomputable def cans_of_soda (S Q D : ℕ) : ℕ :=
  10 * D * S / Q

theorem soda_cans_purchase (S Q D : ℕ) :
  (1 : ℕ) * 10 * D / Q = (10 * D * S) / Q := by
  sorry

end soda_cans_purchase_l2246_224683


namespace find_g_product_l2246_224669

theorem find_g_product 
  (x1 x2 x3 x4 x5 : ℝ)
  (h_root1 : x1^5 - x1^3 + 1 = 0)
  (h_root2 : x2^5 - x2^3 + 1 = 0)
  (h_root3 : x3^5 - x3^3 + 1 = 0)
  (h_root4 : x4^5 - x4^3 + 1 = 0)
  (h_root5 : x5^5 - x5^3 + 1 = 0)
  (g : ℝ → ℝ) 
  (hg : ∀ x, g x = x^2 - 3) :
  g x1 * g x2 * g x3 * g x4 * g x5 = 107 := 
sorry

end find_g_product_l2246_224669


namespace min_value_g_squared_plus_f_l2246_224650

def f (a b : ℝ) (x : ℝ) : ℝ := a * x + b
def g (a c : ℝ) (x : ℝ) : ℝ := a * x + c

theorem min_value_g_squared_plus_f (a b c : ℝ) (h : a ≠ 0) 
  (min_f_squared_plus_g : ∀ x : ℝ, (f a b x)^2 + g a c x ≥ 4)
  (exists_x_min : ∃ x : ℝ, (f a b x)^2 + g a c x = 4) :
  ∃ x : ℝ, (g a c x)^2 + f a b x = -9 / 2 :=
sorry

end min_value_g_squared_plus_f_l2246_224650


namespace problem1_problem2_l2246_224677

variable (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b)
def f (x : ℝ) : ℝ := abs (x - a) + 2 * abs (x + b)

theorem problem1 (h3 : ∃ x, f x = 1) : a + b = 1 := sorry

theorem problem2 (h4 : a + b = 1) (m : ℝ) (h5 : ∀ m, m ≤ 1/a + 2/b)
: m ≤ 3 + 2 * Real.sqrt 2 := sorry

end problem1_problem2_l2246_224677


namespace inverse_proportional_fraction_l2246_224617

theorem inverse_proportional_fraction (N : ℝ) (d f : ℝ) (h : N ≠ 0):
  d * f = N :=
sorry

end inverse_proportional_fraction_l2246_224617


namespace trig_identity_l2246_224697

noncomputable def f (x : ℝ) : ℝ := Real.sin x + Real.cos x
noncomputable def f' (x : ℝ) : ℝ := Real.cos x - Real.sin x

theorem trig_identity (x : ℝ) (h : f x = 2 * f' x) : 
  (1 + Real.sin x ^ 2) / (Real.cos x ^ 2 - Real.sin x * Real.cos x) = 11 / 6 := by
  sorry

end trig_identity_l2246_224697


namespace sandwiches_left_l2246_224613

theorem sandwiches_left 
    (initial_sandwiches : ℕ)
    (first_coworker : ℕ)
    (second_coworker : ℕ)
    (third_coworker : ℕ)
    (kept_sandwiches : ℕ) :
    initial_sandwiches = 50 →
    first_coworker = 4 →
    second_coworker = 3 →
    third_coworker = 2 * first_coworker →
    kept_sandwiches = 3 * second_coworker →
    initial_sandwiches - (first_coworker + second_coworker + third_coworker + kept_sandwiches) = 26 :=
by
  intros h_initial h_first h_second h_third h_kept
  rw [h_initial, h_first, h_second, h_third, h_kept]
  simp
  norm_num
  sorry

end sandwiches_left_l2246_224613


namespace common_ratio_of_geometric_series_l2246_224688

theorem common_ratio_of_geometric_series (a b : ℚ) (h1 : a = 4 / 7) (h2 : b = 16 / 21) :
  b / a = 4 / 3 :=
by
  sorry

end common_ratio_of_geometric_series_l2246_224688


namespace ned_total_mows_l2246_224608

def ned_mowed_front (spring summer fall : Nat) : Nat :=
  spring + summer + fall

def ned_mowed_backyard (spring summer fall : Nat) : Nat :=
  spring + summer + fall

theorem ned_total_mows :
  let front_spring := 6
  let front_summer := 5
  let front_fall := 4
  let backyard_spring := 5
  let backyard_summer := 7
  let backyard_fall := 3
  ned_mowed_front front_spring front_summer front_fall +
  ned_mowed_backyard backyard_spring backyard_summer backyard_fall = 30 := by
  sorry

end ned_total_mows_l2246_224608


namespace value_of_x_l2246_224663

variable (x y z a b c : ℝ)
variable (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0)
variable (h1 : x * y / (x + y) = a)
variable (h2 : x * z / (x + z) = b)
variable (h3 : y * z / (y + z) = c)

theorem value_of_x : x = 2 * a * b * c / (a * c + b * c - a * b) :=
by sorry

end value_of_x_l2246_224663


namespace illegally_parked_percentage_l2246_224656

theorem illegally_parked_percentage (total_cars : ℕ) (towed_cars : ℕ)
  (ht : towed_cars = 2 * total_cars / 100) (not_towed_percentage : ℕ)
  (hp : not_towed_percentage = 80) : 
  (100 * (5 * towed_cars) / total_cars) = 10 :=
by
  sorry

end illegally_parked_percentage_l2246_224656


namespace x_eq_one_l2246_224627

theorem x_eq_one (x y : ℕ) (hx : 0 < x) (hy : 0 < y)
  (div_cond : ∀ n : ℕ, 0 < n → (2^n * y + 1) ∣ (x^(2^n) - 1)) : x = 1 := by
  sorry

end x_eq_one_l2246_224627


namespace cube_of_sum_l2246_224653

theorem cube_of_sum :
  (100 + 2) ^ 3 = 1061208 :=
by
  sorry

end cube_of_sum_l2246_224653


namespace hypotenuse_length_l2246_224645

theorem hypotenuse_length (x a b: ℝ) (h1: a = 7) (h2: b = x - 1) (h3: a^2 + b^2 = x^2) : x = 25 :=
by {
  -- Condition h1 states that one leg 'a' is 7 cm.
  -- Condition h2 states that the other leg 'b' is 1 cm shorter than the hypotenuse 'x', i.e., b = x - 1.
  -- Condition h3 is derived from the Pythagorean theorem, i.e., a^2 + b^2 = x^2.
  -- We need to prove that x = 25 cm.
  sorry
}

end hypotenuse_length_l2246_224645


namespace angle_in_third_quadrant_l2246_224605

theorem angle_in_third_quadrant (α : ℝ) (h1 : Real.sin α < 0) (h2 : Real.tan α > 0) : 
  (π < α ∧ α < 3 * π / 2) :=
by
  sorry

end angle_in_third_quadrant_l2246_224605


namespace air_conditioner_usage_l2246_224680

-- Define the given data and the theorem to be proven
theorem air_conditioner_usage (h : ℝ) (rate : ℝ) (days : ℝ) (total_consumption : ℝ) :
  rate = 0.9 → days = 5 → total_consumption = 27 → (days * h * rate = total_consumption) → h = 6 :=
by
  intros hr dr tc h_eq
  sorry

end air_conditioner_usage_l2246_224680


namespace work_problem_solution_l2246_224606

theorem work_problem_solution :
  (∃ C: ℝ, 
    B_work_days = 8 ∧ 
    (1 / A_work_rate + 1 / B_work_days + C = 1 / 3) ∧ 
    C = 1 / 8
  ) → 
  A_work_days = 12 :=
by
  sorry

end work_problem_solution_l2246_224606


namespace range_of_a_plus_b_l2246_224609

theorem range_of_a_plus_b (a b : ℝ) (h1 : 0 < a) (h2 : a < b)
    (h3 : |2 - a^2| = |2 - b^2|) : 2 < a + b ∧ a + b < 2 * Real.sqrt 2 :=
by
  sorry

end range_of_a_plus_b_l2246_224609


namespace value_of_expression_l2246_224678

theorem value_of_expression {x y z w : ℝ} (h1 : 4 * x * z + y * w = 3) (h2 : x * w + y * z = 6) :
  (2 * x + y) * (2 * z + w) = 15 :=
by
  sorry

end value_of_expression_l2246_224678


namespace part_a_part_b_l2246_224659

theorem part_a (p : ℕ) (hp : Nat.Prime p) (a b : ℤ) (h : a ≡ b [ZMOD p]) : a ^ p ≡ b ^ p [ZMOD p^2] :=
  sorry

theorem part_b (p : ℕ) (hp : Nat.Prime p) : 
  Nat.card { n | n ∈ Finset.range (p^2) ∧ ∃ x, x ^ p ≡ n [ZMOD p^2] } = p :=
  sorry

end part_a_part_b_l2246_224659


namespace find_C_and_D_l2246_224633

theorem find_C_and_D (C D : ℚ) :
  (∀ x : ℚ, ((6 * x - 8) / (2 * x^2 + 5 * x - 3) = (C / (x - 1)) + (D / (2 * x + 3)))) →
  (2*x^2 + 5*x - 3 = (2*x - 1)*(x + 3)) →
  (∀ x : ℚ, ((C*(2*x + 3) + D*(x - 1)) / ((2*x - 1)*(x + 3))) = ((6*x - 8) / ((2*x - 1)*(x + 3)))) →
  (∀ x : ℚ, C*(2*x + 3) + D*(x - 1) = 6*x - 8) →
  C = -2/5 ∧ D = 34/5 := 
by 
  sorry

end find_C_and_D_l2246_224633


namespace product_gcd_lcm_24_60_l2246_224666

theorem product_gcd_lcm_24_60 : Nat.gcd 24 60 * Nat.lcm 24 60 = 1440 := by
  sorry

end product_gcd_lcm_24_60_l2246_224666


namespace triangle_iff_inequality_l2246_224654

variable {a b c : ℝ}

theorem triangle_iff_inequality :
  (a + b > c ∧ b + c > a ∧ c + a > b) ↔ (2 * (a^4 + b^4 + c^4) < (a^2 + b^2 + c^2)^2) := sorry

end triangle_iff_inequality_l2246_224654


namespace combined_weight_l2246_224623

theorem combined_weight (a b c : ℕ) (h1 : a + b = 122) (h2 : b + c = 125) (h3 : c + a = 127) : 
  a + b + c = 187 :=
by
  sorry

end combined_weight_l2246_224623


namespace rick_has_eaten_servings_l2246_224698

theorem rick_has_eaten_servings (calories_per_serving block_servings remaining_calories total_calories servings_eaten : ℝ) 
  (h1 : calories_per_serving = 110) 
  (h2 : block_servings = 16) 
  (h3 : remaining_calories = 1210) 
  (h4 : total_calories = block_servings * calories_per_serving)
  (h5 : servings_eaten = (total_calories - remaining_calories) / calories_per_serving) :
  servings_eaten = 5 :=
by 
  sorry

end rick_has_eaten_servings_l2246_224698


namespace angle_measure_x_l2246_224668

theorem angle_measure_x
    (angle_CBE : ℝ)
    (angle_EBD : ℝ)
    (angle_ABE : ℝ)
    (sum_angles_TRIA : ∀ a b c : ℝ, a + b + c = 180)
    (sum_straight_ANGLE : ∀ a b : ℝ, a + b = 180) :
    angle_CBE = 124 → angle_EBD = 33 → angle_ABE = 19 → x = 91 :=
by
    sorry

end angle_measure_x_l2246_224668


namespace parabola_directrix_l2246_224622

theorem parabola_directrix (x y : ℝ) (h : x^2 = 2 * y) : y = -1 / 2 := 
  sorry

end parabola_directrix_l2246_224622


namespace probability_change_needed_l2246_224612

noncomputable def toy_prices : List ℝ := List.range' 1 11 |>.map (λ n => n * 0.25)

def favorite_toy_price : ℝ := 2.25

def total_quarters : ℕ := 12

def total_toy_count : ℕ := 10

def total_orders : ℕ := Nat.factorial total_toy_count

def ways_to_buy_without_change : ℕ :=
  (Nat.factorial (total_toy_count - 1)) + 2 * (Nat.factorial (total_toy_count - 2))

def probability_without_change : ℚ :=
  ↑ways_to_buy_without_change / ↑total_orders

def probability_with_change : ℚ :=
  1 - probability_without_change

theorem probability_change_needed : probability_with_change = 79 / 90 :=
  sorry

end probability_change_needed_l2246_224612


namespace luncheon_cost_l2246_224610

theorem luncheon_cost
  (s c p : ℝ)
  (h1 : 3 * s + 7 * c + p = 3.15)
  (h2 : 4 * s + 10 * c + p = 4.20) :
  s + c + p = 1.05 :=
by sorry

end luncheon_cost_l2246_224610


namespace set_subset_condition_l2246_224664

theorem set_subset_condition (a : ℝ) :
  (∀ x, (1 < a * x ∧ a * x < 2) → (-1 < x ∧ x < 1)) → (|a| ≥ 2 ∨ a = 0) :=
by
  intro h
  sorry

end set_subset_condition_l2246_224664


namespace dots_per_ladybug_l2246_224695

-- Define the conditions as variables
variables (m t : ℕ) (total_dots : ℕ) (d : ℕ)

-- Setting actual values for the variables based on the given conditions
def m_val : ℕ := 8
def t_val : ℕ := 5
def total_dots_val : ℕ := 78

-- Defining the total number of ladybugs and the average dots per ladybug
def total_ladybugs : ℕ := m_val + t_val

-- To prove: Each ladybug has 6 dots on average
theorem dots_per_ladybug : total_dots_val / total_ladybugs = 6 :=
by
  have m := m_val
  have t := t_val
  have total_dots := total_dots_val
  have d := 6
  sorry

end dots_per_ladybug_l2246_224695


namespace parabola_y_intercepts_l2246_224672

theorem parabola_y_intercepts : 
  ∃ (n : ℕ), n = 2 ∧ 
  ∀ (x : ℝ), x = 0 → 
  ∃ (y : ℝ), 3 * y^2 - 5 * y - 2 = 0 :=
sorry

end parabola_y_intercepts_l2246_224672


namespace simplified_expression_value_l2246_224631

noncomputable def a : ℝ := Real.sqrt 3 + 1
noncomputable def b : ℝ := Real.sqrt 3 - 1

theorem simplified_expression_value :
  ( (a ^ 2 / (a - b) - (2 * a * b - b ^ 2) / (a - b)) / (a - b) * a * b ) = 2 := by
  sorry

end simplified_expression_value_l2246_224631


namespace largest_three_digit_perfect_square_and_cube_l2246_224614

theorem largest_three_digit_perfect_square_and_cube :
  ∃ (n : ℕ), (100 ≤ n ∧ n ≤ 999) ∧ (∃ (a : ℕ), n = a^6) ∧ ∀ (m : ℕ), ((100 ≤ m ∧ m ≤ 999) ∧ (∃ (b : ℕ), m = b^6)) → m ≤ n := 
by 
  sorry

end largest_three_digit_perfect_square_and_cube_l2246_224614


namespace green_sequins_per_row_correct_l2246_224657

def total_blue_sequins : ℕ := 6 * 8
def total_purple_sequins : ℕ := 5 * 12
def total_green_sequins : ℕ := 162 - (total_blue_sequins + total_purple_sequins)
def green_sequins_per_row : ℕ := total_green_sequins / 9

theorem green_sequins_per_row_correct : green_sequins_per_row = 6 := 
by 
  sorry

end green_sequins_per_row_correct_l2246_224657


namespace distance_between_house_and_school_l2246_224624

variable (T D : ℝ)

axiom cond1 : 9 * (T + 20 / 60) = D
axiom cond2 : 12 * (T - 20 / 60) = D
axiom cond3 : 15 * (T - 40 / 60) = D

theorem distance_between_house_and_school : D = 24 := 
by
  sorry

end distance_between_house_and_school_l2246_224624


namespace coordinates_of_point_P_l2246_224637

theorem coordinates_of_point_P 
  (P : ℝ × ℝ)
  (h1 : P.1 < 0 ∧ P.2 < 0) 
  (h2 : abs P.2 = 3)
  (h3 : abs P.1 = 5) :
  P = (-5, -3) :=
sorry

end coordinates_of_point_P_l2246_224637


namespace line_parabola_intersect_l2246_224667

theorem line_parabola_intersect {k : ℝ} 
    (h1: ∀ x y : ℝ, y = k*x - 2 → y^2 = 8*x → x ≠ y)
    (h2: ∀ x1 x2 y1 y2 : ℝ, y1 = k*x1 - 2 → y2 = k*x2 - 2 → y1^2 = 8*x1 → y2^2 = 8*x2 → (x1 + x2) / 2 = 2) : 
    k = 2 := 
sorry

end line_parabola_intersect_l2246_224667


namespace teresa_spends_40_dollars_l2246_224696

-- Definitions of the conditions
def sandwich_cost : ℝ := 7.75
def num_sandwiches : ℝ := 2

def salami_cost : ℝ := 4.00

def brie_cost : ℝ := 3 * salami_cost

def olives_cost_per_pound : ℝ := 10.00
def amount_of_olives : ℝ := 0.25

def feta_cost_per_pound : ℝ := 8.00
def amount_of_feta : ℝ := 0.5

def french_bread_cost : ℝ := 2.00

-- Total cost calculation
def total_cost : ℝ :=
  num_sandwiches * sandwich_cost + salami_cost + brie_cost + olives_cost_per_pound * amount_of_olives + feta_cost_per_pound * amount_of_feta + french_bread_cost

-- Proof statement
theorem teresa_spends_40_dollars :
  total_cost = 40.0 :=
by
  sorry

end teresa_spends_40_dollars_l2246_224696


namespace least_sum_of_bases_l2246_224607

theorem least_sum_of_bases (a b : ℕ) (h1 : 0 < a) (h2 : 0 < b)
  (h3 : 4 * a + 7 = 7 * b + 4) (h4 : 4 * a + 3 % 7 = 0) :
  a + b = 24 :=
sorry

end least_sum_of_bases_l2246_224607


namespace number_of_pairs_satisfying_l2246_224681

theorem number_of_pairs_satisfying (h1 : 2 ^ 2013 < 5 ^ 867) (h2 : 5 ^ 867 < 2 ^ 2014) :
  ∃ k, k = 279 ∧ ∀ (m n : ℕ), 1 ≤ m ∧ m ≤ 2012 ∧ 5 ^ n < 2 ^ m ∧ 2 ^ (m + 2) < 5 ^ (n + 1) → 
  ∃ (count : ℕ), count = 279 :=
by
  sorry

end number_of_pairs_satisfying_l2246_224681


namespace correct_option_C_l2246_224687

def U : Set ℕ := {1, 2, 3, 4, 5, 6}
def M : Set ℕ := {1, 5}
def P : Set ℕ := {2, 4}

theorem correct_option_C : 3 ∈ U \ (M ∪ P) :=
by
  sorry

end correct_option_C_l2246_224687


namespace paige_folders_l2246_224601

def initial_files : Nat := 135
def deleted_files : Nat := 27
def files_per_folder : Rat := 8.5
def folders_rounded_up (files_left : Nat) (per_folder : Rat) : Nat :=
  (Rat.ceil (Rat.ofInt files_left / per_folder)).toNat

theorem paige_folders :
  folders_rounded_up (initial_files - deleted_files) files_per_folder = 13 :=
by
  sorry

end paige_folders_l2246_224601


namespace choir_members_count_l2246_224644

theorem choir_members_count (n : ℕ) 
  (h1 : 150 < n) 
  (h2 : n < 300) 
  (h3 : n % 6 = 1) 
  (h4 : n % 8 = 3) 
  (h5 : n % 9 = 2) : 
  n = 163 :=
sorry

end choir_members_count_l2246_224644


namespace quadratic_radical_type_equivalence_l2246_224692

def is_same_type_as_sqrt2 (x : ℝ) : Prop := ∃ k : ℚ, x = k * (Real.sqrt 2)

theorem quadratic_radical_type_equivalence (A B C D : ℝ) (hA : A = (Real.sqrt 8) / 7)
  (hB : B = Real.sqrt 3) (hC : C = Real.sqrt (1 / 3)) (hD : D = Real.sqrt 12) :
  is_same_type_as_sqrt2 A ∧ ¬ is_same_type_as_sqrt2 B ∧ ¬ is_same_type_as_sqrt2 C ∧ ¬ is_same_type_as_sqrt2 D :=
by
  sorry

end quadratic_radical_type_equivalence_l2246_224692


namespace person_age_in_1954_l2246_224674

theorem person_age_in_1954 
  (x : ℤ)
  (cond1 : ∃ k1 : ℤ, 7 * x = 13 * k1 + 11)
  (cond2 : ∃ k2 : ℤ, 13 * x = 11 * k2 + 7)
  (input_year : ℤ) :
  input_year = 1954 → x = 1868 → input_year - x = 86 :=
by
  sorry

end person_age_in_1954_l2246_224674


namespace area_of_square_plot_l2246_224693

theorem area_of_square_plot (s : ℕ) (price_per_foot total_cost: ℕ)
  (h_price : price_per_foot = 58)
  (h_total_cost : total_cost = 3944) :
  (s * s = 289) :=
by
  sorry

end area_of_square_plot_l2246_224693


namespace probability_of_convex_quadrilateral_l2246_224634

def binomial (n k : ℕ) : ℕ := Nat.choose n k

theorem probability_of_convex_quadrilateral :
  let num_points := 8
  let total_chords := binomial num_points 2
  let total_ways_to_select_4_chords := binomial total_chords 4
  let favorable_ways := binomial num_points 4
  (favorable_ways : ℚ) / (total_ways_to_select_4_chords : ℚ) = 2 / 585 :=
by
  -- definitions
  let num_points := 8
  let total_chords := binomial 8 2
  let total_ways_to_select_4_chords := binomial total_chords 4
  let favorable_ways := binomial num_points 4
  
  -- assertion of result
  have h : (favorable_ways : ℚ) / (total_ways_to_select_4_chords : ℚ) = 2 / 585 :=
    sorry
  exact h

end probability_of_convex_quadrilateral_l2246_224634


namespace necessary_but_not_sufficient_condition_l2246_224682

variable {a : ℕ → ℝ}
variable {q : ℝ}

def geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n * q

theorem necessary_but_not_sufficient_condition
    (a1_pos : a 1 > 0)
    (geo_seq : geometric_sequence a q)
    (a3_lt_a6 : a 3 < a 6) :
  (a 1 < a 3) ↔ ∃ k : ℝ, k > 1 ∧ a 1 * k^2 < a 1 * k^5 :=
by
  sorry

end necessary_but_not_sufficient_condition_l2246_224682


namespace batter_sugar_is_one_l2246_224655

-- Definitions based on the conditions given
def initial_sugar : ℕ := 3
def sugar_per_bag : ℕ := 6
def num_bags : ℕ := 2
def frosting_sugar_per_dozen : ℕ := 2
def total_dozen_cupcakes : ℕ := 5

-- Total sugar Lillian has
def total_sugar : ℕ := initial_sugar + num_bags * sugar_per_bag

-- Sugar needed for frosting
def frosting_sugar_needed : ℕ := frosting_sugar_per_dozen * total_dozen_cupcakes

-- Sugar used for the batter
def batter_sugar_total : ℕ := total_sugar - frosting_sugar_needed

-- Question asked in the problem
def batter_sugar_per_dozen : ℕ := batter_sugar_total / total_dozen_cupcakes

theorem batter_sugar_is_one :
  batter_sugar_per_dozen = 1 :=
by
  sorry -- Proof is not required here

end batter_sugar_is_one_l2246_224655


namespace odd_function_zero_unique_l2246_224611

variable (f : ℝ → ℝ)

def odd_function (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f x = - f (- x)

def functional_eq (f : ℝ → ℝ) : Prop := ∀ x y : ℝ, f (x + y) * f (x - y) = f x ^ 2 * f y ^ 2

theorem odd_function_zero_unique
  (h_odd : odd_function f)
  (h_func_eq : functional_eq f) :
  ∀ x : ℝ, f x = 0 :=
by
  sorry

end odd_function_zero_unique_l2246_224611


namespace total_weight_cashew_nuts_and_peanuts_l2246_224600

theorem total_weight_cashew_nuts_and_peanuts (weight_cashew_nuts weight_peanuts : ℕ) (h1 : weight_cashew_nuts = 3) (h2 : weight_peanuts = 2) : 
  weight_cashew_nuts + weight_peanuts = 5 := 
by
  sorry

end total_weight_cashew_nuts_and_peanuts_l2246_224600


namespace A_speed_ratio_B_speed_l2246_224684

-- Define the known conditions
def B_speed : ℚ := 1 / 12
def total_speed : ℚ := 1 / 4

-- Define the problem statement
theorem A_speed_ratio_B_speed : ∃ (A_speed : ℚ), A_speed + B_speed = total_speed ∧ (A_speed / B_speed = 2) :=
by
  sorry

end A_speed_ratio_B_speed_l2246_224684


namespace workers_allocation_l2246_224635

-- Definitions based on conditions
def num_workers := 90
def bolt_per_worker := 15
def nut_per_worker := 24
def bolt_matching_requirement := 2

-- Statement of the proof problem
theorem workers_allocation (x y : ℕ) :
  x + y = num_workers ∧
  bolt_matching_requirement * bolt_per_worker * x = nut_per_worker * y →
  x = 40 ∧ y = 50 :=
by
  sorry

end workers_allocation_l2246_224635


namespace exists_int_squares_l2246_224619

theorem exists_int_squares (a b n : ℕ) (ha : a > 0) (hb : b > 0) (hn : n > 0) :
  ∃ x y : ℤ, (a^2 + b^2)^n = x^2 + y^2 :=
by
  sorry

end exists_int_squares_l2246_224619


namespace arc_length_l2246_224639

theorem arc_length (circumference : ℝ) (angle : ℝ) (h1 : circumference = 72) (h2 : angle = 45) :
  ∃ length : ℝ, length = 9 :=
by
  sorry

end arc_length_l2246_224639


namespace evaluate_expression_l2246_224676

def improper_fraction (n : Int) (a : Int) (b : Int) : Rat :=
  n + (a : Rat) / b

def expression (x : Rat) : Rat :=
  (x * 1.65 - x + (7 / 20) * x) * 47.5 * 0.8 * 2.5

theorem evaluate_expression : 
  expression (improper_fraction 20 94 95) = 1994 := 
by 
  sorry

end evaluate_expression_l2246_224676


namespace even_function_b_eq_zero_l2246_224625

theorem even_function_b_eq_zero (b : ℝ) :
  (∀ x : ℝ, (x^2 + b * x) = (x^2 - b * x)) → b = 0 :=
by sorry

end even_function_b_eq_zero_l2246_224625


namespace unique_diff_of_cubes_l2246_224636

theorem unique_diff_of_cubes (n k : ℕ) (h : 61 = n^3 - k^3) : n = 5 ∧ k = 4 :=
sorry

end unique_diff_of_cubes_l2246_224636


namespace ways_to_score_at_least_7_points_l2246_224646

-- Definitions based on the given conditions
def red_balls : Nat := 4
def white_balls : Nat := 6
def points_red : Nat := 2
def points_white : Nat := 1

-- Function to count the number of combinations for choosing k elements from n elements
def choose (n : Nat) (k : Nat) : Nat :=
  if h : k ≤ n then
    Nat.descFactorial n k / Nat.factorial k
  else
    0

-- The main theorem to prove the number of ways to get at least 7 points by choosing 5 balls out
theorem ways_to_score_at_least_7_points : 
  (choose red_balls 4 * choose white_balls 1) +
  (choose red_balls 3 * choose white_balls 2) +
  (choose red_balls 2 * choose white_balls 3) = 186 := 
sorry

end ways_to_score_at_least_7_points_l2246_224646


namespace parallelogram_vector_sum_l2246_224648

theorem parallelogram_vector_sum (A B C D : ℝ × ℝ) (parallelogram : A ≠ B ∧ B ≠ C ∧ C ≠ D ∧ A ≠ D ∧ (C - A = D - B) ∧ (B - D = A - C)) :
  (B - A) + (C - B) = C - A :=
by
  sorry

end parallelogram_vector_sum_l2246_224648


namespace points_on_inverse_proportion_l2246_224649

theorem points_on_inverse_proportion (y_1 y_2 : ℝ) :
  (2:ℝ) = 5 / y_1 → (3:ℝ) = 5 / y_2 → y_1 > y_2 :=
by
  intros h1 h2
  sorry

end points_on_inverse_proportion_l2246_224649


namespace equivalent_proof_problem_l2246_224620

def math_problem (x y : ℚ) : ℚ :=
((x + y) * (3 * x - y) + y^2) / (-x)

theorem equivalent_proof_problem (hx : x = 4) (hy : y = -(1/4)) :
  math_problem x y = -23 / 2 :=
by
  sorry

end equivalent_proof_problem_l2246_224620


namespace find_f_five_thirds_l2246_224685

variable {R : Type*} [LinearOrderedField R]

-- Define the odd function and its properties
variable {f : R → R}
variable (oddf : ∀ x : R, f (-x) = -f x)
variable (propf : ∀ x : R, f (1 + x) = f (-x))
variable (val : f (- (1 / 3 : R)) = 1 / 3)

theorem find_f_five_thirds : f (5 / 3 : R) = 1 / 3 := by
  sorry

end find_f_five_thirds_l2246_224685


namespace Petya_has_24_chips_l2246_224661

noncomputable def PetyaChips (x y : ℕ) : ℕ := 3 * x - 3

theorem Petya_has_24_chips (x y : ℕ) (h1 : y = x - 2) (h2 : 3 * x - 3 = 4 * y - 4) : PetyaChips x y = 24 :=
by
  sorry

end Petya_has_24_chips_l2246_224661


namespace domain_tan_2x_plus_pi_over_3_l2246_224618

noncomputable def domain_tan : Set ℝ := {x | ∃ k : ℤ, x = k * Real.pi + Real.pi / 2}

noncomputable def domain_tan_transformed : Set ℝ :=
  {x | ∃ k : ℤ, x = k * (Real.pi / 2) + Real.pi / 12}

theorem domain_tan_2x_plus_pi_over_3 :
  (∀ x, ¬ (x ∈ domain_tan)) ↔ (∀ x, ¬ (x ∈ domain_tan_transformed)) :=
by
  sorry

end domain_tan_2x_plus_pi_over_3_l2246_224618


namespace length_of_train_is_400_meters_l2246_224675

noncomputable def relative_speed (speed_train speed_man : ℝ) : ℝ :=
  speed_train - speed_man

noncomputable def km_per_hr_to_m_per_s (speed_km_per_hr : ℝ) : ℝ :=
  speed_km_per_hr * (1000 / 3600)

noncomputable def length_of_train (relative_speed_m_per_s time_seconds : ℝ) : ℝ :=
  relative_speed_m_per_s * time_seconds

theorem length_of_train_is_400_meters :
  let speed_train := 30 -- km/hr
  let speed_man := 6 -- km/hr
  let time_to_cross := 59.99520038396929 -- seconds
  let rel_speed := km_per_hr_to_m_per_s (relative_speed speed_train speed_man)
  length_of_train rel_speed time_to_cross = 400 :=
by
  sorry

end length_of_train_is_400_meters_l2246_224675


namespace police_emergency_number_prime_factor_l2246_224671

theorem police_emergency_number_prime_factor (N : ℕ) (h1 : N % 1000 = 133) : 
  ∃ p : ℕ, Nat.Prime p ∧ p > 7 ∧ p ∣ N :=
sorry

end police_emergency_number_prime_factor_l2246_224671


namespace fraction_inequality_solution_set_l2246_224621

theorem fraction_inequality_solution_set : 
  {x : ℝ | (2 - x) / (x + 4) > 0} = {x : ℝ | -4 < x ∧ x < 2} :=
by sorry

end fraction_inequality_solution_set_l2246_224621


namespace functional_equation_initial_condition_unique_f3_l2246_224604

noncomputable def f : ℝ → ℝ := sorry

theorem functional_equation (x y : ℝ) : f (f x + y) = f (x ^ 2 - y) + 2 * f x * y := sorry

theorem initial_condition : f 1 = 1 := sorry

theorem unique_f3 : f 3 = 9 := sorry

end functional_equation_initial_condition_unique_f3_l2246_224604


namespace find_a_if_f_is_odd_function_l2246_224628

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x^2 * (a * 2^x - 2^(-x))

def is_odd_function (f : ℝ → ℝ) : Prop :=
∀ x : ℝ, f (-x) = -f x

theorem find_a_if_f_is_odd_function : 
  ∀ a : ℝ, is_odd_function (f a) → a = 1 :=
by
  sorry

end find_a_if_f_is_odd_function_l2246_224628


namespace parrot_age_is_24_l2246_224662

variable (cat_age : ℝ) (rabbit_age : ℝ) (dog_age : ℝ) (parrot_age : ℝ)

def ages (cat_age rabbit_age dog_age parrot_age : ℝ) : Prop :=
  cat_age = 8 ∧
  rabbit_age = cat_age / 2 ∧
  dog_age = rabbit_age * 3 ∧
  parrot_age = cat_age + rabbit_age + dog_age

theorem parrot_age_is_24 (cat_age rabbit_age dog_age parrot_age : ℝ) :
  ages cat_age rabbit_age dog_age parrot_age → parrot_age = 24 :=
by
  intro h
  sorry

end parrot_age_is_24_l2246_224662


namespace middle_integer_is_five_l2246_224643

-- Define the conditions of the problem
def consecutive_one_digit_positive_odd_integers (a b c : ℤ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ 
  a % 2 = 1 ∧ b % 2 = 1 ∧ c % 2 = 1 ∧
  a + 2 = b ∧ b + 2 = c ∨ a + 2 = c ∧ c + 2 = b

def sum_is_one_seventh_of_product (a b c : ℤ) : Prop :=
  a + b + c = (a * b * c) / 7

-- Define the theorem to prove
theorem middle_integer_is_five :
  ∃ (b : ℤ), consecutive_one_digit_positive_odd_integers (b - 2) b (b + 2) ∧
             sum_is_one_seventh_of_product (b - 2) b (b + 2) ∧
             b = 5 :=
sorry

end middle_integer_is_five_l2246_224643


namespace circle_radius_condition_l2246_224689

theorem circle_radius_condition (c: ℝ):
  (∃ x y : ℝ, (x^2 + y^2 + 4 * x - 2 * y - 5 * c = 0)) → c > -1 :=
by
  sorry

end circle_radius_condition_l2246_224689


namespace wuzhen_conference_arrangements_l2246_224615

theorem wuzhen_conference_arrangements 
  (countries : Finset ℕ)
  (hotels : Finset ℕ)
  (h_countries_count : countries.card = 5)
  (h_hotels_count : hotels.card = 3) :
  ∃ f : ℕ → ℕ,
  (∀ c ∈ countries, f c ∈ hotels) ∧
  (∀ h ∈ hotels, ∃ c ∈ countries, f c = h) ∧
  (Finset.card (Set.toFinset (f '' countries)) = 3) ∧
  ∃ n : ℕ,
  n = 150 := 
sorry

end wuzhen_conference_arrangements_l2246_224615


namespace slope_of_line_l2246_224638

theorem slope_of_line (x₁ y₁ x₂ y₂ : ℝ) (h₁ : x₁ = 1) (h₂ : y₁ = 3) (h₃ : x₂ = 4) (h₄ : y₂ = -6) : 
  (y₂ - y₁) / (x₂ - x₁) = -3 := by
  sorry

end slope_of_line_l2246_224638


namespace focus_of_parabola_x_squared_eq_4y_is_0_1_l2246_224670

theorem focus_of_parabola_x_squared_eq_4y_is_0_1 :
  ∃ (x y : ℝ), (0, 1) = (x, y) ∧ (∀ a b : ℝ, a^2 = 4 * b → (x, y) = (0, 1)) :=
sorry

end focus_of_parabola_x_squared_eq_4y_is_0_1_l2246_224670


namespace sum_sum_sum_sum_eq_one_l2246_224642

def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

-- Mathematical problem statement
theorem sum_sum_sum_sum_eq_one :
  sum_of_digits (sum_of_digits (sum_of_digits (sum_of_digits (2017^2017)))) = 1 := 
sorry

end sum_sum_sum_sum_eq_one_l2246_224642


namespace weight_of_one_liter_ghee_brand_b_l2246_224602

theorem weight_of_one_liter_ghee_brand_b (wa w_mix : ℕ) (vol_a vol_b : ℕ) (w_mix_total : ℕ) (wb : ℕ) :
  wa = 900 ∧ vol_a = 3 ∧ vol_b = 2 ∧ w_mix = 3360 →
  (vol_a * wa + vol_b * wb = w_mix →
  wb = 330) :=
by
  intros h_eq h_eq2
  obtain ⟨h_wa, h_vol_a, h_vol_b, h_w_mix⟩ := h_eq
  rw [h_wa, h_vol_a, h_vol_b, h_w_mix] at h_eq2
  sorry

end weight_of_one_liter_ghee_brand_b_l2246_224602


namespace ratio_P_S_l2246_224691

theorem ratio_P_S (S N P : ℝ) 
  (hN : N = S / 4) 
  (hP : P = N / 4) : 
  P / S = 1 / 16 := 
by 
  sorry

end ratio_P_S_l2246_224691


namespace unique_number_outside_range_f_l2246_224616

noncomputable def f (a b c d : ℝ) (x : ℝ) : ℝ := (a * x + b) / (c * x + d)

theorem unique_number_outside_range_f (a b c d : ℝ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : c ≠ 0) (h4 : d ≠ 0)
  (h5 : f a b c d 19 = 19) (h6 : f a b c d 97 = 97)
  (h7 : ∀ x, x ≠ -d / c → f a b c d (f a b c d x) = x) : 
  ∀ y : ℝ, y ≠ 58 → ∃ x : ℝ, f a b c d x ≠ y :=
sorry

end unique_number_outside_range_f_l2246_224616


namespace mean_equals_sum_of_squares_l2246_224660

noncomputable def arithmetic_mean (x y z : ℝ) := (x + y + z) / 3
noncomputable def geometric_mean (x y z : ℝ) := (x * y * z) ^ (1 / 3)
noncomputable def harmonic_mean (x y z : ℝ) := 3 / ((1 / x) + (1 / y) + (1 / z))

theorem mean_equals_sum_of_squares (x y z : ℝ) (h1 : arithmetic_mean x y z = 10)
  (h2 : geometric_mean x y z = 6) (h3 : harmonic_mean x y z = 4) :
  x^2 + y^2 + z^2 = 576 :=
  sorry

end mean_equals_sum_of_squares_l2246_224660


namespace find_number_of_breeding_rabbits_l2246_224632

def breeding_rabbits_condition (B : ℕ) : Prop :=
  ∃ (kittens_first_spring remaining_kittens_first_spring kittens_second_spring remaining_kittens_second_spring : ℕ),
    kittens_first_spring = 10 * B ∧
    remaining_kittens_first_spring = 5 * B + 5 ∧
    kittens_second_spring = 60 ∧
    remaining_kittens_second_spring = kittens_second_spring - 4 ∧
    B + remaining_kittens_first_spring + remaining_kittens_second_spring = 121

theorem find_number_of_breeding_rabbits (B : ℕ) : breeding_rabbits_condition B → B = 10 :=
by
  sorry

end find_number_of_breeding_rabbits_l2246_224632


namespace goldfish_in_first_tank_l2246_224658

-- Definitions of conditions
def num_fish_third_tank : Nat := 10
def num_fish_second_tank := 3 * num_fish_third_tank
def num_fish_first_tank := num_fish_second_tank / 2
def goldfish_and_beta_sum (G : Nat) : Prop := G + 8 = num_fish_first_tank

-- Theorem to prove the number of goldfish in the first fish tank
theorem goldfish_in_first_tank (G : Nat) (h : goldfish_and_beta_sum G) : G = 7 :=
by
  sorry

end goldfish_in_first_tank_l2246_224658


namespace equation_satisfying_solution_l2246_224629

theorem equation_satisfying_solution (x y : ℤ) :
  (x = 1 ∧ y = 4 → x + 3 * y ≠ 7) ∧
  (x = 2 ∧ y = 1 → x + 3 * y ≠ 7) ∧
  (x = -2 ∧ y = 3 → x + 3 * y = 7) ∧
  (x = 4 ∧ y = 2 → x + 3 * y ≠ 7) :=
by
  sorry

end equation_satisfying_solution_l2246_224629


namespace johns_donation_l2246_224652

theorem johns_donation (A J : ℝ) 
  (h1 : (75 / 1.5) = A) 
  (h2 : A * 2 = 100)
  (h3 : (100 + J) / 3 = 75) : 
  J = 125 :=
by 
  sorry

end johns_donation_l2246_224652


namespace roof_length_width_difference_l2246_224641

theorem roof_length_width_difference (w l : ℝ) 
  (h1 : l = 5 * w) 
  (h2 : l * w = 720) : l - w = 48 := 
sorry

end roof_length_width_difference_l2246_224641


namespace problem_1_problem_2_l2246_224694

-- Problem 1: Prove that (\frac{1}{5} - \frac{2}{3} - \frac{3}{10}) × (-60) = 46
theorem problem_1 : (1/5 - 2/3 - 3/10) * -60 = 46 := by
  sorry

-- Problem 2: Prove that (-1)^{2024} + 24 ÷ (-2)^3 - 15^2 × (1/15)^2 = -3
theorem problem_2 : (-1)^2024 + 24 / (-2)^3 - 15^2 * (1/15)^2 = -3 := by
  sorry

end problem_1_problem_2_l2246_224694


namespace shuttle_speed_in_kph_l2246_224647

def sec_per_min := 60
def min_per_hour := 60
def sec_per_hour := sec_per_min * min_per_hour
def speed_in_kps := 12
def speed_in_kph := speed_in_kps * sec_per_hour

theorem shuttle_speed_in_kph :
  speed_in_kph = 43200 :=
by
  -- No proof needed
  sorry

end shuttle_speed_in_kph_l2246_224647


namespace length_of_cable_l2246_224679

-- Conditions
def condition1 (x y z : ℝ) : Prop := x + y + z = 8
def condition2 (x y z : ℝ) : Prop := x * y + y * z + x * z = -18

-- Conclusion we want to prove
theorem length_of_cable (x y z : ℝ) (h1 : condition1 x y z) (h2 : condition2 x y z) :
  4 * π * Real.sqrt (59 / 3) = 4 * π * (Real.sqrt ((x^2 + y^2 + z^2 - ((x + y + z)^2 - 4*(x*y + y*z + x*z))) / 3)) :=
sorry

end length_of_cable_l2246_224679


namespace no_intersection_of_ellipses_l2246_224640

theorem no_intersection_of_ellipses :
  (∀ (x y : ℝ), (9*x^2 + y^2 = 9) ∧ (x^2 + 16*y^2 = 16) → false) :=
sorry

end no_intersection_of_ellipses_l2246_224640


namespace height_of_parallelogram_l2246_224699

theorem height_of_parallelogram (area base height : ℝ) (h1 : area = 240) (h2 : base = 24) : height = 10 :=
by
  sorry

end height_of_parallelogram_l2246_224699


namespace floor_neg_seven_thirds_l2246_224665

theorem floor_neg_seven_thirds : ⌊-7 / 3⌋ = -3 :=
sorry

end floor_neg_seven_thirds_l2246_224665


namespace find_p_l2246_224603

theorem find_p 
  (h : {x | x^2 - 5 * x + p ≥ 0} = {x | x ≤ -1 ∨ x ≥ 6}) : p = -6 :=
by
  sorry

end find_p_l2246_224603


namespace lcm_1404_972_l2246_224651

def num1 := 1404
def num2 := 972

theorem lcm_1404_972 : Nat.lcm num1 num2 = 88452 := 
by 
  sorry

end lcm_1404_972_l2246_224651
