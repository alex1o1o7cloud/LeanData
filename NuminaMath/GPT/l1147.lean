import Mathlib

namespace NUMINAMATH_GPT_sum_b4_b6_l1147_114795

theorem sum_b4_b6
  (b : ℕ → ℝ)
  (h₁ : ∀ n : ℕ, n > 0 → ∃ d : ℝ, ∀ m : ℕ, m > 0 → (1 / b (m + 1) - 1 / b m) = d)
  (h₂ : b 1 + b 2 + b 3 + b 4 + b 5 + b 6 + b 7 + b 8 + b 9 = 90) :
  b 4 + b 6 = 20 := by
  sorry

end NUMINAMATH_GPT_sum_b4_b6_l1147_114795


namespace NUMINAMATH_GPT_find_xyz_l1147_114744

theorem find_xyz (x y z : ℝ) 
  (h1 : (x + y + z) * (x * y + x * z + y * z) = 49) 
  (h2 : x^2 * (y + z) + y^2 * (x + z) + z^2 * (x + y) = 17) 
  (h3 : x^3 + y^3 + z^3 = 27) : 
  x * y * z = 32 / 3 :=
  sorry

end NUMINAMATH_GPT_find_xyz_l1147_114744


namespace NUMINAMATH_GPT_pool_length_l1147_114773

theorem pool_length (r : ℕ) (t : ℕ) (w : ℕ) (d : ℕ) (L : ℕ) 
  (H1 : r = 60)
  (H2 : t = 2000)
  (H3 : w = 80)
  (H4 : d = 10)
  (H5 : L = (r * t) / (w * d)) : L = 150 :=
by
  rw [H1, H2, H3, H4] at H5
  exact H5


end NUMINAMATH_GPT_pool_length_l1147_114773


namespace NUMINAMATH_GPT_find_abc_l1147_114769

theorem find_abc (a b c : ℕ) (h_coprime_ab : gcd a b = 1) (h_coprime_ac : gcd a c = 1) 
  (h_coprime_bc : gcd b c = 1) (h1 : ab + bc + ac = 431) (h2 : a + b + c = 39) 
  (h3 : a + b + (ab / c) = 18) : 
  a = 7 ∧ b = 9 ∧ c = 23 := 
sorry

end NUMINAMATH_GPT_find_abc_l1147_114769


namespace NUMINAMATH_GPT_price_per_half_pound_of_basil_l1147_114759

theorem price_per_half_pound_of_basil
    (cost_per_pound_eggplant : ℝ)
    (pounds_eggplant : ℝ)
    (cost_per_pound_zucchini : ℝ)
    (pounds_zucchini : ℝ)
    (cost_per_pound_tomato : ℝ)
    (pounds_tomato : ℝ)
    (cost_per_pound_onion : ℝ)
    (pounds_onion : ℝ)
    (quarts_ratatouille : ℝ)
    (cost_per_quart : ℝ) :
    pounds_eggplant = 5 → cost_per_pound_eggplant = 2 →
    pounds_zucchini = 4 → cost_per_pound_zucchini = 2 →
    pounds_tomato = 4 → cost_per_pound_tomato = 3.5 →
    pounds_onion = 3 → cost_per_pound_onion = 1 →
    quarts_ratatouille = 4 → cost_per_quart = 10 →
    (cost_per_quart * quarts_ratatouille - 
    (cost_per_pound_eggplant * pounds_eggplant + 
    cost_per_pound_zucchini * pounds_zucchini + 
    cost_per_pound_tomato * pounds_tomato + 
    cost_per_pound_onion * pounds_onion)) / 2 = 2.5 :=
by
    intros h₁ h₂ h₃ h₄ h₅ h₆ h₇ h₈ h₉ h₀
    rw [h₁, h₂, h₃, h₄, h₅, h₆, h₇, h₈, h₉, h₀]
    sorry

end NUMINAMATH_GPT_price_per_half_pound_of_basil_l1147_114759


namespace NUMINAMATH_GPT_part1_daily_sales_profit_final_max_daily_sales_profit_l1147_114738

-- Conditions from part (a)
def original_selling_price : ℚ := 30
def cost_price : ℚ := 15
def original_sales_volume : ℚ := 60
def sales_increase_per_yuan : ℚ := 10

-- Part (1): Daily sales profit if the price is reduced by 2 yuan
def new_selling_price1 : ℚ := original_selling_price - 2
def new_sales_volume1 : ℚ := original_sales_volume + (2 * sales_increase_per_yuan)
def profit_per_kilogram1 : ℚ := new_selling_price1 - cost_price
def daily_sales_profit1 : ℚ := profit_per_kilogram1 * new_sales_volume1

theorem part1_daily_sales_profit : daily_sales_profit1 = 1040 := by
  sorry

-- Part (2): Maximum daily sales profit and corresponding selling price
def selling_price_at_max_profit : ℚ := 51 / 2

def daily_profit (x : ℚ) : ℚ :=
  (x - cost_price) * (original_sales_volume + (original_selling_price - x) * sales_increase_per_yuan)

theorem final_max_daily_sales_profit :
  (∀ x : ℚ, daily_profit x ≤ daily_profit selling_price_at_max_profit) ∧ daily_profit selling_price_at_max_profit = 1102.5 := by
  sorry

end NUMINAMATH_GPT_part1_daily_sales_profit_final_max_daily_sales_profit_l1147_114738


namespace NUMINAMATH_GPT_find_point_P_l1147_114717

structure Point3D :=
  (x : ℝ)
  (y : ℝ)
  (z : ℝ)

def isEquidistant (p1 p2 : Point3D) (q : Point3D) : Prop :=
  (q.x - p1.x)^2 + (q.y - p1.y)^2 + (q.z - p1.z)^2 = (q.x - p2.x)^2 + (q.y - p2.y)^2 + (q.z - p2.z)^2

theorem find_point_P (P : Point3D) :
  (∀ (Q : Point3D), isEquidistant ⟨2, 3, -4⟩ P Q → (8 * Q.x - 6 * Q.y + 18 * Q.z = 70)) →
  P = ⟨6, 0, 5⟩ :=
by 
  sorry

end NUMINAMATH_GPT_find_point_P_l1147_114717


namespace NUMINAMATH_GPT_insurance_covers_80_percent_of_medical_bills_l1147_114736

theorem insurance_covers_80_percent_of_medical_bills 
    (vaccine_cost : ℕ) (num_vaccines : ℕ) (doctor_visit_cost trip_cost : ℕ) (amount_tom_pays : ℕ) 
    (total_cost := num_vaccines * vaccine_cost + doctor_visit_cost) 
    (total_trip_cost := trip_cost + total_cost)
    (insurance_coverage := total_trip_cost - amount_tom_pays)
    (percent_covered := (insurance_coverage * 100) / total_cost) :
    vaccine_cost = 45 → num_vaccines = 10 → doctor_visit_cost = 250 → trip_cost = 1200 → amount_tom_pays = 1340 →
    percent_covered = 80 := 
by
  sorry

end NUMINAMATH_GPT_insurance_covers_80_percent_of_medical_bills_l1147_114736


namespace NUMINAMATH_GPT_maximum_achievable_score_l1147_114764

def robot_initial_iq : Nat := 25
def problem_scores : List Nat := [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

theorem maximum_achievable_score 
  (initial_iq : Nat := robot_initial_iq) 
  (scores : List Nat := problem_scores) 
  : Nat :=
  31

end NUMINAMATH_GPT_maximum_achievable_score_l1147_114764


namespace NUMINAMATH_GPT_log_expansion_l1147_114796

theorem log_expansion (a : ℝ) (h : a = Real.log 4 / Real.log 5) : Real.log 64 / Real.log 5 - 2 * (Real.log 20 / Real.log 5) = a - 2 :=
by
  sorry

end NUMINAMATH_GPT_log_expansion_l1147_114796


namespace NUMINAMATH_GPT_initial_provisions_last_l1147_114733

theorem initial_provisions_last (x : ℕ) (h : 2000 * (x - 20) = 4000 * 10) : x = 40 :=
by sorry

end NUMINAMATH_GPT_initial_provisions_last_l1147_114733


namespace NUMINAMATH_GPT_floor_expression_equality_l1147_114761

theorem floor_expression_equality :
  ⌊((2023^3 : ℝ) / (2021 * 2022) - (2021^3 : ℝ) / (2022 * 2023))⌋ = 8 := 
sorry

end NUMINAMATH_GPT_floor_expression_equality_l1147_114761


namespace NUMINAMATH_GPT_min_int_solution_inequality_l1147_114789

theorem min_int_solution_inequality : ∃ x : ℤ, 4 * (x + 1) + 2 > x - 1 ∧ ∀ y : ℤ, 4 * (y + 1) + 2 > y - 1 → y ≥ x := 
by 
  sorry

end NUMINAMATH_GPT_min_int_solution_inequality_l1147_114789


namespace NUMINAMATH_GPT_students_who_won_first_prize_l1147_114734

theorem students_who_won_first_prize :
  ∃ x : ℤ, 30 ≤ x ∧ x ≤ 55 ∧ (x % 3 = 2) ∧ (x % 5 = 4) ∧ (x % 7 = 2) ∧ x = 44 :=
by
  sorry

end NUMINAMATH_GPT_students_who_won_first_prize_l1147_114734


namespace NUMINAMATH_GPT_find_a_value_l1147_114768

theorem find_a_value
  (a : ℝ)
  (h : ∀ x, 0 ≤ x ∧ x ≤ (π / 2) → a * Real.sin x + Real.cos x ≤ 2)
  (h_max : ∃ x, 0 ≤ x ∧ x ≤ (π / 2) ∧ a * Real.sin x + Real.cos x = 2) :
  a = Real.sqrt 3 :=
sorry

end NUMINAMATH_GPT_find_a_value_l1147_114768


namespace NUMINAMATH_GPT_negation_of_universal_quantifier_proposition_l1147_114784

variable (x : ℝ)

theorem negation_of_universal_quantifier_proposition :
  (¬ ∀ x : ℝ, x^2 - x + 1/4 ≥ 0) ↔ (∃ x : ℝ, x^2 - x + 1/4 < 0) :=
sorry

end NUMINAMATH_GPT_negation_of_universal_quantifier_proposition_l1147_114784


namespace NUMINAMATH_GPT_degrees_for_basic_astrophysics_correct_l1147_114721

-- Definitions for conditions
def percentage_allocations : List ℚ := [13, 24, 15, 29, 8]
def total_percentage : ℚ := percentage_allocations.sum
def remaining_percentage : ℚ := 100 - total_percentage

-- The question to answer
def total_degrees : ℚ := 360
def degrees_for_basic_astrophysics : ℚ := remaining_percentage / 100 * total_degrees

-- Prove that the degrees for basic astrophysics is 39.6
theorem degrees_for_basic_astrophysics_correct :
  degrees_for_basic_astrophysics = 39.6 :=
by
  sorry

end NUMINAMATH_GPT_degrees_for_basic_astrophysics_correct_l1147_114721


namespace NUMINAMATH_GPT_cos_squared_value_l1147_114755

theorem cos_squared_value (α : ℝ) (h : Real.sin (2 * α) = 2 / 3) :
  Real.cos (α + π / 4) ^ 2 = 1 / 6 :=
sorry

end NUMINAMATH_GPT_cos_squared_value_l1147_114755


namespace NUMINAMATH_GPT_find_natural_numbers_l1147_114783

theorem find_natural_numbers (n : ℕ) : 
  (∃ d : ℕ, d ≤ 9 ∧ 10 * n + d = 13 * n) ↔ n = 1 ∨ n = 2 ∨ n = 3 :=
by {
  sorry
}

end NUMINAMATH_GPT_find_natural_numbers_l1147_114783


namespace NUMINAMATH_GPT_find_a8_in_arithmetic_sequence_l1147_114707

variable {a : ℕ → ℕ} -- Define a as a function from natural numbers to natural numbers

-- Assume a is an arithmetic sequence
axiom arithmetic_sequence (a : ℕ → ℕ) : ∃ d : ℤ, ∀ n : ℕ, a (n + 1) = a n + d

theorem find_a8_in_arithmetic_sequence (h : a 4 + a 6 + a 8 + a 10 + a 12 = 120) : a 8 = 24 :=
by
  sorry  -- Proof to be filled in separately

end NUMINAMATH_GPT_find_a8_in_arithmetic_sequence_l1147_114707


namespace NUMINAMATH_GPT_sqrt_expression_l1147_114770

theorem sqrt_expression :
  Real.sqrt 18 - 3 * Real.sqrt (1 / 2) + Real.sqrt 2 = (5 * Real.sqrt 2) / 2 :=
by
  sorry

end NUMINAMATH_GPT_sqrt_expression_l1147_114770


namespace NUMINAMATH_GPT_problem_equivalence_l1147_114762

theorem problem_equivalence : (7^2 - 3^2)^4 = 2560000 :=
by
  sorry

end NUMINAMATH_GPT_problem_equivalence_l1147_114762


namespace NUMINAMATH_GPT_bouquet_combinations_l1147_114719

theorem bouquet_combinations :
  ∃ n : ℕ, (∀ r c t : ℕ, 4 * r + 3 * c + 2 * t = 60 → true) ∧ n = 13 :=
sorry

end NUMINAMATH_GPT_bouquet_combinations_l1147_114719


namespace NUMINAMATH_GPT_rectangle_perimeter_l1147_114780

variable (a b : ℝ)
variable (h1 : a * b = 24)
variable (h2 : a^2 + b^2 = 121)

theorem rectangle_perimeter : 2 * (a + b) = 26 := 
by
  sorry

end NUMINAMATH_GPT_rectangle_perimeter_l1147_114780


namespace NUMINAMATH_GPT_Mike_exercises_l1147_114778

theorem Mike_exercises :
  let pull_ups_per_visit := 2
  let push_ups_per_visit := 5
  let squats_per_visit := 10
  let office_visits_per_day := 5
  let kitchen_visits_per_day := 8
  let living_room_visits_per_day := 7
  let days_per_week := 7
  let total_pull_ups := pull_ups_per_visit * office_visits_per_day * days_per_week
  let total_push_ups := push_ups_per_visit * kitchen_visits_per_day * days_per_week
  let total_squats := squats_per_visit * living_room_visits_per_day * days_per_week
  total_pull_ups = 70 ∧ total_push_ups = 280 ∧ total_squats = 490 :=
by
  let pull_ups_per_visit := 2
  let push_ups_per_visit := 5
  let squats_per_visit := 10
  let office_visits_per_day := 5
  let kitchen_visits_per_day := 8
  let living_room_visits_per_day := 7
  let days_per_week := 7
  let total_pull_ups := pull_ups_per_visit * office_visits_per_day * days_per_week
  let total_push_ups := push_ups_per_visit * kitchen_visits_per_day * days_per_week
  let total_squats := squats_per_visit * living_room_visits_per_day * days_per_week
  have h1 : total_pull_ups = 2 * 5 * 7 := rfl
  have h2 : total_push_ups = 5 * 8 * 7 := rfl
  have h3 : total_squats = 10 * 7 * 7 := rfl
  show total_pull_ups = 70 ∧ total_push_ups = 280 ∧ total_squats = 490
  sorry

end NUMINAMATH_GPT_Mike_exercises_l1147_114778


namespace NUMINAMATH_GPT_sin_1200_eq_sqrt3_div_2_l1147_114704

theorem sin_1200_eq_sqrt3_div_2 : Real.sin (1200 * Real.pi / 180) = Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_GPT_sin_1200_eq_sqrt3_div_2_l1147_114704


namespace NUMINAMATH_GPT_range_of_a_l1147_114775

-- Define the quadratic inequality
def quadratic_inequality (a x : ℝ) : ℝ := (a-1)*x^2 + (a-1)*x + 1

theorem range_of_a :
  (∀ x : ℝ, quadratic_inequality a x > 0) ↔ (1 ≤ a ∧ a < 5) :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_l1147_114775


namespace NUMINAMATH_GPT_sum_first_100_odd_l1147_114777

-- Define the sequence of odd numbers.
def odd (n : ℕ) : ℕ := 2 * n + 1

-- Define the sum of the first n odd natural numbers.
def sumOdd (n : ℕ) : ℕ := (n * (n + 1))

-- State the theorem.
theorem sum_first_100_odd : sumOdd 100 = 10000 :=
by
  -- Skipping the proof as per the instructions
  sorry

end NUMINAMATH_GPT_sum_first_100_odd_l1147_114777


namespace NUMINAMATH_GPT_supplement_of_complement_of_75_degree_angle_l1147_114772

def angle : ℕ := 75
def complement_angle (a : ℕ) := 90 - a
def supplement_angle (a : ℕ) := 180 - a

theorem supplement_of_complement_of_75_degree_angle : supplement_angle (complement_angle angle) = 165 :=
by
  sorry

end NUMINAMATH_GPT_supplement_of_complement_of_75_degree_angle_l1147_114772


namespace NUMINAMATH_GPT_false_statement_of_quadratic_l1147_114760

-- Define the function f and the conditions
def f (a b c x : ℝ) := a * x^2 + b * x + c

theorem false_statement_of_quadratic (a b c x0 : ℝ) (h₀ : a > 0) (h₁ : 2 * a * x0 + b = 0) :
  ¬ ∀ x : ℝ, f a b c x ≤ f a b c x0 := by
  sorry

end NUMINAMATH_GPT_false_statement_of_quadratic_l1147_114760


namespace NUMINAMATH_GPT_candies_problem_l1147_114706

theorem candies_problem (emily jennifer bob : ℕ) (h1 : emily = 6) 
  (h2 : jennifer = 2 * emily) (h3 : jennifer = 3 * bob) : bob = 4 := by
  -- Lean code to skip the proof
  sorry

end NUMINAMATH_GPT_candies_problem_l1147_114706


namespace NUMINAMATH_GPT_original_rectangle_length_l1147_114739

-- Define the problem conditions
def length_three_times_width (l w : ℕ) : Prop :=
  l = 3 * w

def length_decreased_width_increased (l w : ℕ) : Prop :=
  l - 5 = w + 5

-- Define the proof problem
theorem original_rectangle_length (l w : ℕ) (H1 : length_three_times_width l w) (H2 : length_decreased_width_increased l w) : l = 15 :=
sorry

end NUMINAMATH_GPT_original_rectangle_length_l1147_114739


namespace NUMINAMATH_GPT_find_unknown_l1147_114746

theorem find_unknown (x : ℝ) :
  300 * 2 + (x + 4) * (1 / 8) = 602 → x = 12 :=
by 
  sorry

end NUMINAMATH_GPT_find_unknown_l1147_114746


namespace NUMINAMATH_GPT_proof_problem_l1147_114758

noncomputable def problem : ℕ :=
  let p := 588
  let q := 0
  let r := 1
  p + q + r

theorem proof_problem
  (AB : ℝ) (P Q : ℝ) (AP BP PQ : ℝ) (angle_POQ : ℝ) 
  (h1 : AB = 1200)
  (h2 : AP + PQ = BP)
  (h3 : BP - Q = 600)
  (h4 : angle_POQ = 30)
  (h5 : PQ = 500)
  : problem = 589 := by
    sorry

end NUMINAMATH_GPT_proof_problem_l1147_114758


namespace NUMINAMATH_GPT_solve_abc_l1147_114722

theorem solve_abc (a b c : ℝ) (h1 : a ≤ b ∧ b ≤ c) (h2 : a + b + c = -1) (h3 : a * b + b * c + a * c = -4) (h4 : a * b * c = -2) :
  a = -1 - Real.sqrt 3 ∧ b = -1 + Real.sqrt 3 ∧ c = 1 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_solve_abc_l1147_114722


namespace NUMINAMATH_GPT_find_a_l1147_114720

noncomputable def f (a : ℝ) (x : ℝ) := (1/2) * a * x^2 + Real.log x

theorem find_a (h_max : ∃ (x : Set.Icc (0 : ℝ) 1), f (-Real.exp 1) x = -1) : 
  ∀ a : ℝ, (∀ x : ℝ, 0 < x → x ≤ 1 → f a x ≤ -1) → a = -Real.exp 1 :=
sorry

end NUMINAMATH_GPT_find_a_l1147_114720


namespace NUMINAMATH_GPT_find_a_of_even_function_l1147_114750

-- Define the function f
def f (x a : ℝ) := (x + 1) * (x + a)

-- State the theorem to be proven
theorem find_a_of_even_function (a : ℝ) (h : ∀ x : ℝ, f x a = f (-x) a) : a = -1 :=
by
  -- The actual proof goes here
  sorry

end NUMINAMATH_GPT_find_a_of_even_function_l1147_114750


namespace NUMINAMATH_GPT_solution_m_in_interval_l1147_114745

noncomputable def f (x : ℝ) (m : ℝ) : ℝ :=
if x < 1 then -x^2 + 2 * m * x - 2 else 1 + Real.log x

theorem solution_m_in_interval :
  ∃ m : ℝ, (1 ≤ m ∧ m ≤ 2) ∧
  (∀ x < 1, ∀ y < 1, x ≤ y → f x m ≤ f y m) ∧
  (∀ x ≥ 1, ∀ y ≥ 1, x ≤ y → f x m ≤ f y m) ∧
  (∀ x < 1, ∀ y ≥ 1, f x m ≤ f y m) :=
by
  sorry

end NUMINAMATH_GPT_solution_m_in_interval_l1147_114745


namespace NUMINAMATH_GPT_relationship_of_variables_l1147_114752

variable {a b c d : ℝ}

theorem relationship_of_variables 
  (h1 : d - a < c - b) 
  (h2 : c - b < 0) 
  (h3 : d - b = c - a) : 
  d < c ∧ c < b ∧ b < a := 
sorry

end NUMINAMATH_GPT_relationship_of_variables_l1147_114752


namespace NUMINAMATH_GPT_opposite_of_one_over_2023_l1147_114701

def one_over_2023 : ℚ := 1 / 2023

theorem opposite_of_one_over_2023 : -one_over_2023 = -1 / 2023 :=
by
  sorry

end NUMINAMATH_GPT_opposite_of_one_over_2023_l1147_114701


namespace NUMINAMATH_GPT_vector_dot_product_l1147_114708

-- Define the vectors a and b
def a : ℝ × ℝ := (1, 1)
def b : ℝ × ℝ := (1, -1)

-- Prove that the scalar product a · (a - 2b) equals 2
theorem vector_dot_product :
  let u := a
  let v := b
  u • (u - (2 • v)) = 2 :=
by 
  -- Placeholder for the proof
  sorry

end NUMINAMATH_GPT_vector_dot_product_l1147_114708


namespace NUMINAMATH_GPT_value_of_f_sin_7pi_over_6_l1147_114705

def f (x : ℝ) : ℝ := 4 * x^2 + 2 * x

theorem value_of_f_sin_7pi_over_6 :
  f (Real.sin (7 * Real.pi / 6)) = 0 :=
by
  sorry

end NUMINAMATH_GPT_value_of_f_sin_7pi_over_6_l1147_114705


namespace NUMINAMATH_GPT_range_of_m_l1147_114797

open Real

theorem range_of_m (m : ℝ) : (∀ x : ℝ, 4 * cos x + sin x ^ 2 + m - 4 = 0) ↔ 0 ≤ m ∧ m ≤ 8 :=
sorry

end NUMINAMATH_GPT_range_of_m_l1147_114797


namespace NUMINAMATH_GPT_pen_price_relationship_l1147_114756

variable (x : ℕ) -- x represents the number of pens
variable (y : ℝ) -- y represents the total selling price in dollars
variable (p : ℝ) -- p represents the price per pen

-- Each box contains 10 pens
def pens_per_box := 10

-- Each box is sold for $16
def price_per_box := 16

-- Given the conditions, prove the relationship between y and x
theorem pen_price_relationship (hx : x = 10) (hp : p = 16) :
  y = 1.6 * x := sorry

end NUMINAMATH_GPT_pen_price_relationship_l1147_114756


namespace NUMINAMATH_GPT_digit_B_divisibility_l1147_114786

theorem digit_B_divisibility :
  ∃ B : ℕ, B < 10 ∧ (2 * 100 + B * 10 + 9) % 13 = 0 ↔ B = 0 :=
by
  sorry

end NUMINAMATH_GPT_digit_B_divisibility_l1147_114786


namespace NUMINAMATH_GPT_product_sum_of_roots_l1147_114716

theorem product_sum_of_roots
  {p q r : ℝ}
  (h : (∀ x : ℝ, (4 * x^3 - 8 * x^2 + 16 * x - 12) = 0 → (x = p ∨ x = q ∨ x = r))) :
  p * q + q * r + r * p = 4 := 
sorry

end NUMINAMATH_GPT_product_sum_of_roots_l1147_114716


namespace NUMINAMATH_GPT_downstream_distance_80_l1147_114711

-- Conditions
variables (Speed_boat Speed_stream Distance_upstream : ℝ)

-- Assign given values
def speed_boat := 36 -- kmph
def speed_stream := 12 -- kmph
def distance_upstream := 40 -- km

-- Effective speeds
def speed_downstream := speed_boat + speed_stream -- kmph
def speed_upstream := speed_boat - speed_stream -- kmph

-- Downstream distance
noncomputable def distance_downstream : ℝ := 80 -- km

-- Theorem
theorem downstream_distance_80 :
  speed_boat = 36 → speed_stream = 12 → distance_upstream = 40 →
  (distance_upstream / speed_upstream = distance_downstream / speed_downstream) :=
by
  sorry

end NUMINAMATH_GPT_downstream_distance_80_l1147_114711


namespace NUMINAMATH_GPT_hexagon_side_lengths_l1147_114763

theorem hexagon_side_lengths (n m : ℕ) (AB BC : ℕ) (P : ℕ) :
  n + m = 6 ∧ n * 4 + m * 7 = 38 ∧ AB = 4 ∧ BC = 7 → m = 4 :=
by
  sorry

end NUMINAMATH_GPT_hexagon_side_lengths_l1147_114763


namespace NUMINAMATH_GPT_total_tubes_in_consignment_l1147_114757

theorem total_tubes_in_consignment (N : ℕ) 
  (h : (5 / (N : ℝ)) * (4 / (N - 1 : ℝ)) = 0.05263157894736842) : 
  N = 20 := 
sorry

end NUMINAMATH_GPT_total_tubes_in_consignment_l1147_114757


namespace NUMINAMATH_GPT_brett_red_marbles_l1147_114792

variables (r b : ℕ)

-- Define the conditions
axiom h1 : b = r + 24
axiom h2 : b = 5 * r

theorem brett_red_marbles : r = 6 :=
by
  sorry

end NUMINAMATH_GPT_brett_red_marbles_l1147_114792


namespace NUMINAMATH_GPT_alice_cell_phone_cost_l1147_114742

theorem alice_cell_phone_cost
  (base_cost : ℕ)
  (included_hours : ℕ)
  (text_cost_per_message : ℕ)
  (extra_minute_cost : ℕ)
  (messages_sent : ℕ)
  (hours_spent : ℕ) :
  base_cost = 25 →
  included_hours = 40 →
  text_cost_per_message = 4 →
  extra_minute_cost = 5 →
  messages_sent = 150 →
  hours_spent = 42 →
  (base_cost + (messages_sent * text_cost_per_message) / 100 + ((hours_spent - included_hours) * 60 * extra_minute_cost) / 100) = 37 := 
by
  intros h1 h2 h3 h4 h5 h6
  sorry

end NUMINAMATH_GPT_alice_cell_phone_cost_l1147_114742


namespace NUMINAMATH_GPT_simplify_expression_l1147_114715

noncomputable def x : ℝ := Real.sqrt 3 + 1

theorem simplify_expression (x : ℝ) (h : x = Real.sqrt 3 + 1) :
  ((1 - (x / (x + 1))) / ((x^2 - 1) / (x^2 + 2*x + 1))) = Real.sqrt 3 / 3 :=
by
  sorry

end NUMINAMATH_GPT_simplify_expression_l1147_114715


namespace NUMINAMATH_GPT_intersection_is_2_l1147_114735

-- Define the sets A and B
def A : Set ℝ := { x | x < 1 }
def B : Set ℝ := { -1, 0, 2 }

-- Define the complement of A
def A_complement : Set ℝ := { x | x ≥ 1 }

-- Define the intersection of the complement of A and B
def intersection : Set ℝ := A_complement ∩ B

-- Prove that the intersection is {2}
theorem intersection_is_2 : intersection = {2} := by
  sorry

end NUMINAMATH_GPT_intersection_is_2_l1147_114735


namespace NUMINAMATH_GPT_exists_universal_accessible_city_l1147_114724

-- Define the basic structure for cities and flights
structure Country :=
  (City : Type)
  (accessible : City → City → Prop)

namespace Country

-- Define the properties of accessibility in the country
variables {C : Country}

-- Axiom: Each city is accessible from itself
axiom self_accessible (A : C.City) : C.accessible A A

-- Axiom: For any two cities, there exists a city from which both are accessible
axiom exists_intermediate (P Q : C.City) : ∃ R : C.City, C.accessible R P ∧ C.accessible R Q

-- Definition of the main theorem
theorem exists_universal_accessible_city :
  ∃ U : C.City, ∀ A : C.City, C.accessible U A :=
sorry

end Country

end NUMINAMATH_GPT_exists_universal_accessible_city_l1147_114724


namespace NUMINAMATH_GPT_incorrect_conclusion_l1147_114714

-- Define the linear regression model
def model (x : ℝ) : ℝ := 0.85 * x - 85.71

-- Define the conditions
axiom linear_correlation : ∀ (x y : ℝ), ∃ (x_i y_i : ℝ) (i : ℕ), model x = y

-- The theorem to prove the statement for x = 170 is false
theorem incorrect_conclusion (x : ℝ) (h : x = 170) : ¬ (model x = 58.79) :=
  by sorry

end NUMINAMATH_GPT_incorrect_conclusion_l1147_114714


namespace NUMINAMATH_GPT_marcel_corn_l1147_114799

theorem marcel_corn (C : ℕ) (H1 : ∃ D, D = C / 2) (H2 : 27 = C + C / 2 + 8 + 4) : C = 10 :=
sorry

end NUMINAMATH_GPT_marcel_corn_l1147_114799


namespace NUMINAMATH_GPT_books_brought_back_l1147_114779

def initial_books : ℕ := 235
def taken_out_tuesday : ℕ := 227
def taken_out_friday : ℕ := 35
def books_remaining : ℕ := 29

theorem books_brought_back (B : ℕ) :
  B = 56 ↔ (initial_books - taken_out_tuesday + B - taken_out_friday = books_remaining) :=
by
  -- proof steps would go here
  sorry

end NUMINAMATH_GPT_books_brought_back_l1147_114779


namespace NUMINAMATH_GPT_value_of_expression_l1147_114790

theorem value_of_expression : ((25 + 8)^2 - (8^2 + 25^2) = 400) :=
by 
  sorry

end NUMINAMATH_GPT_value_of_expression_l1147_114790


namespace NUMINAMATH_GPT_least_common_addition_of_primes_l1147_114767

theorem least_common_addition_of_primes (x y : ℕ) (hx : Nat.Prime x) (hy : Nat.Prime y) (hxy : x < y) (h : 4 * x + y = 87) : x + y = 81 := 
sorry

end NUMINAMATH_GPT_least_common_addition_of_primes_l1147_114767


namespace NUMINAMATH_GPT_tan_a_over_tan_b_plus_tan_b_over_tan_a_l1147_114787

theorem tan_a_over_tan_b_plus_tan_b_over_tan_a {a b : ℝ} 
  (h1 : (Real.sin a / Real.cos b) + (Real.sin b / Real.cos a) = 2)
  (h2 : (Real.cos a / Real.sin b) + (Real.cos b / Real.sin a) = 4) :
  (Real.tan a / Real.tan b) + (Real.tan b / Real.tan a) = 44 / 5 :=
sorry

end NUMINAMATH_GPT_tan_a_over_tan_b_plus_tan_b_over_tan_a_l1147_114787


namespace NUMINAMATH_GPT_ages_correct_in_2018_l1147_114727

-- Define the initial ages in the year 2000
def age_marianne_2000 : ℕ := 20
def age_bella_2000 : ℕ := 8
def age_carmen_2000 : ℕ := 15

-- Define the birth year of Elli
def birth_year_elli : ℕ := 2003

-- Define the target year when Bella turns 18
def year_bella_turns_18 : ℕ := 2000 + 18

-- Define the ages to be proven
def age_marianne_2018 : ℕ := 30
def age_carmen_2018 : ℕ := 33
def age_elli_2018 : ℕ := 15

theorem ages_correct_in_2018 :
  age_marianne_2018 = age_marianne_2000 + (year_bella_turns_18 - 2000) ∧
  age_carmen_2018 = age_carmen_2000 + (year_bella_turns_18 - 2000) ∧
  age_elli_2018 = year_bella_turns_18 - birth_year_elli :=
by 
  -- The proof would go here
  sorry

end NUMINAMATH_GPT_ages_correct_in_2018_l1147_114727


namespace NUMINAMATH_GPT_range_of_a_l1147_114747

open Real

theorem range_of_a (a : ℝ) :
  (∀ x, |x - 1| < 3 → (x + 2) * (x + a) < 0) ∧ ¬ (∀ x, (x + 2) * (x + a) < 0 → |x - 1| < 3) →
  a < -4 :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_l1147_114747


namespace NUMINAMATH_GPT_even_function_value_l1147_114700

theorem even_function_value (f : ℝ → ℝ) (h_even : ∀ x, f x = f (-x))
  (h_neg : ∀ x, x < 0 → f x = 2 ^ x) :
  f (Real.log 9 / Real.log 4) = 1 / 3 :=
by
  sorry

end NUMINAMATH_GPT_even_function_value_l1147_114700


namespace NUMINAMATH_GPT_paperclip_day_l1147_114748

theorem paperclip_day:
  ∃ k : ℕ, 5 * 3 ^ k > 500 ∧ ∀ m : ℕ, m < k → 5 * 3 ^ m ≤ 500 ∧ k % 7 = 5 :=
sorry

end NUMINAMATH_GPT_paperclip_day_l1147_114748


namespace NUMINAMATH_GPT_smallest_positive_multiple_of_45_is_45_l1147_114709

-- Define that a positive multiple of 45 is parameterized by a positive integer
def is_positive_multiple (m : ℕ) : Prop := ∃ x : ℕ, x > 0 ∧ m = 45 * x

-- State the theorem
theorem smallest_positive_multiple_of_45_is_45 :
  (∀ m : ℕ, is_positive_multiple m → (45 ≤ m)) :=
by
  sorry

end NUMINAMATH_GPT_smallest_positive_multiple_of_45_is_45_l1147_114709


namespace NUMINAMATH_GPT_truck_travel_distance_l1147_114766

theorem truck_travel_distance (b t : ℝ) (ht : t > 0) (ht30 : t + 30 > 0) : 
  let converted_feet := 4 * 60
  let time_half := converted_feet / 2
  let speed_first_half := b / 4
  let speed_second_half := b / 4
  let distance_first_half := speed_first_half * time_half / t
  let distance_second_half := speed_second_half * time_half / (t + 30)
  let total_distance_feet := distance_first_half + distance_second_half
  let result_yards := total_distance_feet / 3
  result_yards = (10 * b / t) + (10 * b / (t + 30))
:= by
  -- proof skipped
  sorry

end NUMINAMATH_GPT_truck_travel_distance_l1147_114766


namespace NUMINAMATH_GPT_mark_sold_8_boxes_less_l1147_114740

theorem mark_sold_8_boxes_less (T M A x : ℕ) (hT : T = 9) 
    (hM : M = T - x) (hA : A = T - 2) 
    (hM_ge_1 : 1 ≤ M) (hA_ge_1 : 1 ≤ A) 
    (h_sum_lt_T : M + A < T) : x = 8 := 
by
  sorry

end NUMINAMATH_GPT_mark_sold_8_boxes_less_l1147_114740


namespace NUMINAMATH_GPT_simple_interest_borrowed_rate_l1147_114732

theorem simple_interest_borrowed_rate
  (P_borrowed P_lent : ℝ)
  (n_years : ℕ)
  (gain_per_year : ℝ)
  (simple_interest_lent_rate : ℝ)
  (SI_lending : ℝ := P_lent * simple_interest_lent_rate * n_years / 100)
  (total_gain : ℝ := gain_per_year * n_years) :
  SI_lending = 1000 →
  total_gain = 100 →
  ∀ (SI_borrowing : ℝ), SI_borrowing = SI_lending - total_gain →
  ∀ (R_borrowed : ℝ), SI_borrowing = P_borrowed * R_borrowed * n_years / 100 →
  R_borrowed = 9 := 
by
  sorry

end NUMINAMATH_GPT_simple_interest_borrowed_rate_l1147_114732


namespace NUMINAMATH_GPT_sequence_a6_value_l1147_114791

theorem sequence_a6_value :
  ∃ (a : ℕ → ℚ), (a 1 = 1) ∧ (∀ n, a (n + 1) = a n / (2 * a n + 1)) ∧ (a 6 = 1 / 11) :=
by
  sorry

end NUMINAMATH_GPT_sequence_a6_value_l1147_114791


namespace NUMINAMATH_GPT_acid_solution_replacement_percentage_l1147_114723

theorem acid_solution_replacement_percentage 
  (original_concentration fraction_replaced final_concentration replaced_percentage : ℝ)
  (h₁ : original_concentration = 0.50)
  (h₂ : fraction_replaced = 0.5)
  (h₃ : final_concentration = 0.40)
  (h₄ : 0.25 + fraction_replaced * replaced_percentage = final_concentration) :
  replaced_percentage = 0.30 :=
by
  sorry

end NUMINAMATH_GPT_acid_solution_replacement_percentage_l1147_114723


namespace NUMINAMATH_GPT_product_mod_self_inverse_l1147_114726

theorem product_mod_self_inverse 
  {n : ℕ} (hn : 0 < n) (a b : ℤ) (ha : a * a % n = 1) (hb : b * b % n = 1) :
  (a * b) % n = 1 := 
sorry

end NUMINAMATH_GPT_product_mod_self_inverse_l1147_114726


namespace NUMINAMATH_GPT_trailing_zeros_of_9_pow_999_plus_1_l1147_114774

theorem trailing_zeros_of_9_pow_999_plus_1 :
  ∃ n : ℕ, n = 999 ∧ (9^n + 1) % 10 = 0 ∧ (9^n + 1) % 100 ≠ 0 :=
by
  sorry

end NUMINAMATH_GPT_trailing_zeros_of_9_pow_999_plus_1_l1147_114774


namespace NUMINAMATH_GPT_arithmetic_mean_124_4_31_l1147_114753

theorem arithmetic_mean_124_4_31 :
  let numbers := [12, 25, 39, 48]
  let total := 124
  let count := 4
  (total / count : ℝ) = 31 := by
  sorry

end NUMINAMATH_GPT_arithmetic_mean_124_4_31_l1147_114753


namespace NUMINAMATH_GPT_movie_ticket_cost_l1147_114741

variable (x : ℝ)
variable (h1 : x * 2 + 1.59 + 13.95 = 36.78)

theorem movie_ticket_cost : x = 10.62 :=
by
  sorry

end NUMINAMATH_GPT_movie_ticket_cost_l1147_114741


namespace NUMINAMATH_GPT_magnitude_product_l1147_114718

-- Definitions based on conditions
def z1 : Complex := ⟨7, -4⟩
def z2 : Complex := ⟨3, 10⟩

-- Statement of the theorem to be proved
theorem magnitude_product :
  Complex.abs (z1 * z2) = Real.sqrt 7085 := by
  sorry

end NUMINAMATH_GPT_magnitude_product_l1147_114718


namespace NUMINAMATH_GPT_sum_of_digits_floor_large_number_div_50_eq_457_l1147_114785

-- Define a helper function to sum the digits of a number
def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

-- Define the large number as the sum of its components
def large_number : ℕ :=
  51 * 10^96 + 52 * 10^94 + 53 * 10^92 + 54 * 10^90 + 55 * 10^88 + 56 * 10^86 + 
  57 * 10^84 + 58 * 10^82 + 59 * 10^80 + 60 * 10^78 + 61 * 10^76 + 62 * 10^74 + 
  63 * 10^72 + 64 * 10^70 + 65 * 10^68 + 66 * 10^66 + 67 * 10^64 + 68 * 10^62 + 
  69 * 10^60 + 70 * 10^58 + 71 * 10^56 + 72 * 10^54 + 73 * 10^52 + 74 * 10^50 + 
  75 * 10^48 + 76 * 10^46 + 77 * 10^44 + 78 * 10^42 + 79 * 10^40 + 80 * 10^38 + 
  81 * 10^36 + 82 * 10^34 + 83 * 10^32 + 84 * 10^30 + 85 * 10^28 + 86 * 10^26 + 
  87 * 10^24 + 88 * 10^22 + 89 * 10^20 + 90 * 10^18 + 91 * 10^16 + 92 * 10^14 + 
  93 * 10^12 + 94 * 10^10 + 95 * 10^8 + 96 * 10^6 + 97 * 10^4 + 98 * 10^2 + 99

-- Define the main statement to be proven
theorem sum_of_digits_floor_large_number_div_50_eq_457 : 
    sum_of_digits (Nat.floor (large_number / 50)) = 457 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_digits_floor_large_number_div_50_eq_457_l1147_114785


namespace NUMINAMATH_GPT_probability_of_F_l1147_114794

-- Definitions for the probabilities of regions D, E, and the total probability
def P_D : ℚ := 3 / 8
def P_E : ℚ := 1 / 4
def total_probability : ℚ := 1

-- The hypothesis
lemma total_probability_eq_one : P_D + P_E + (1 - P_D - P_E) = total_probability :=
by
  simp [P_D, P_E, total_probability]

-- The goal is to prove this statement
theorem probability_of_F : 1 - P_D - P_E = 3 / 8 :=
by
  -- Using the total_probability_eq_one hypothesis
  have h := total_probability_eq_one
  -- This is a structured approach where verification using hypothesis and simplification can be done
  sorry

end NUMINAMATH_GPT_probability_of_F_l1147_114794


namespace NUMINAMATH_GPT_shopper_savings_percentage_l1147_114712

theorem shopper_savings_percentage
  (amount_saved : ℝ) (final_price : ℝ)
  (h_saved : amount_saved = 3)
  (h_final : final_price = 27) :
  (amount_saved / (final_price + amount_saved)) * 100 = 10 := 
by
  sorry

end NUMINAMATH_GPT_shopper_savings_percentage_l1147_114712


namespace NUMINAMATH_GPT_Mille_suckers_l1147_114730

theorem Mille_suckers:
  let pretzels := 64
  let goldfish := 4 * pretzels
  let baggies := 16
  let items_per_baggie := 22
  let total_items_needed := baggies * items_per_baggie
  let total_pretzels_and_goldfish := pretzels + goldfish
  let suckers := total_items_needed - total_pretzels_and_goldfish
  suckers = 32 := 
by sorry

end NUMINAMATH_GPT_Mille_suckers_l1147_114730


namespace NUMINAMATH_GPT_fourth_number_is_2_eighth_number_is_2_l1147_114765

-- Conditions as given in the problem
def initial_board := [1]

/-- Medians recorded in Mitya's notebook for the first 10 numbers -/
def medians := [1, 2, 3, 2.5, 3, 2.5, 2, 2, 2, 2.5]

/-- Prove that the fourth number written on the board is 2 given initial conditions. -/
theorem fourth_number_is_2 (board : ℕ → ℤ)  
  (h1 : board 0 = 1)
  (h2 : medians = [1, 2, 3, 2.5, 3, 2.5, 2, 2, 2, 2.5])
  : board 3 = 2 :=
sorry

/-- Prove that the eighth number written on the board is 2 given initial conditions. -/
theorem eighth_number_is_2 (board : ℕ → ℤ) 
  (h1 : board 0 = 1)
  (h2 : medians = [1, 2, 3, 2.5, 3, 2.5, 2, 2, 2, 2.5])
  : board 7 = 2 :=
sorry

end NUMINAMATH_GPT_fourth_number_is_2_eighth_number_is_2_l1147_114765


namespace NUMINAMATH_GPT_field_day_difference_l1147_114788

theorem field_day_difference :
  let girls_4th_first := 12
  let boys_4th_first := 13
  let girls_4th_second := 15
  let boys_4th_second := 11
  let girls_5th_first := 9
  let boys_5th_first := 13
  let girls_5th_second := 10
  let boys_5th_second := 11
  let total_girls := girls_4th_first + girls_4th_second + girls_5th_first + girls_5th_second
  let total_boys := boys_4th_first + boys_4th_second + boys_5th_first + boys_5th_second
  total_boys - total_girls = 2 :=
by
  let girls_4th_first := 12
  let boys_4th_first := 13
  let girls_4th_second := 15
  let boys_4th_second := 11
  let girls_5th_first := 9
  let boys_5th_first := 13
  let girls_5th_second := 10
  let boys_5th_second := 11
  let total_girls := girls_4th_first + girls_4th_second + girls_5th_first + girls_5th_second
  let total_boys := boys_4th_first + boys_4th_second + boys_5th_first + boys_5th_second
  have h1 : total_girls = 46 := rfl
  have h2 : total_boys = 48 := rfl
  have h3 : total_boys - total_girls = 2 := rfl
  exact h3

end NUMINAMATH_GPT_field_day_difference_l1147_114788


namespace NUMINAMATH_GPT_total_distance_traveled_l1147_114749

noncomputable def row_speed_still_water : ℝ := 8
noncomputable def river_speed : ℝ := 2

theorem total_distance_traveled (h : (3.75 / (row_speed_still_water - river_speed)) + (3.75 / (row_speed_still_water + river_speed)) = 1) : 
  2 * 3.75 = 7.5 :=
by
  sorry

end NUMINAMATH_GPT_total_distance_traveled_l1147_114749


namespace NUMINAMATH_GPT_parabola_vertex_example_l1147_114751

noncomputable def parabola_vertex (a b c : ℝ) := (-b / (2 * a), (4 * a * c - b^2) / (4 * a))

theorem parabola_vertex_example : parabola_vertex (-4) (-16) (-20) = (-2, -4) :=
by
  sorry

end NUMINAMATH_GPT_parabola_vertex_example_l1147_114751


namespace NUMINAMATH_GPT_original_volume_of_cube_l1147_114729

theorem original_volume_of_cube (a : ℕ) 
  (h1 : (a + 2) * (a - 2) * (a + 3) = a^3 - 7) : 
  a = 3 :=
by sorry

end NUMINAMATH_GPT_original_volume_of_cube_l1147_114729


namespace NUMINAMATH_GPT_remainder_eq_one_l1147_114781

theorem remainder_eq_one (n : ℤ) (h : n % 6 = 1) : (n + 150) % 6 = 1 := 
by
  sorry

end NUMINAMATH_GPT_remainder_eq_one_l1147_114781


namespace NUMINAMATH_GPT_soccer_team_games_l1147_114725

theorem soccer_team_games :
  ∃ G : ℕ, G % 2 = 0 ∧ 
           45 / 100 * 36 = 16 ∧ 
           ∀ R, R = G - 36 → (16 + 75 / 100 * R) = 62 / 100 * G ∧
           G = 84 :=
sorry

end NUMINAMATH_GPT_soccer_team_games_l1147_114725


namespace NUMINAMATH_GPT_find_b_perpendicular_lines_l1147_114731

theorem find_b_perpendicular_lines (b : ℚ)
  (line1 : (3 : ℚ) * x + 4 * y - 6 = 0)
  (line2 : b * x + 4 * y - 6 = 0)
  (perpendicular : ( - (3 : ℚ) / 4 ) * ( - (b / 4) ) = -1) :
  b = - (16 : ℚ) / 3 := 
sorry

end NUMINAMATH_GPT_find_b_perpendicular_lines_l1147_114731


namespace NUMINAMATH_GPT_part_I_part_II_l1147_114793

-- Define the function f(x) as per the problem's conditions
def f (x a : ℝ) : ℝ := abs (x - 2 * a) + abs (x - a)

theorem part_I (x : ℝ) (h₁ : 1 ≠ 0) : 
  (f x 1 > 2) ↔ (x < 1 / 2 ∨ x > 5 / 2) :=
by
  sorry

theorem part_II (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) :
  f b a ≥ f a a ∧ (f b a = f a a ↔ ((2 * a - b ≥ 0 ∧ b - a ≥ 0) ∨ (2 * a - b ≤ 0 ∧ b - a ≤ 0) ∨ (2 * a - b = 0) ∨ (b - a = 0))) :=
by
  sorry

end NUMINAMATH_GPT_part_I_part_II_l1147_114793


namespace NUMINAMATH_GPT_x_range_l1147_114737

theorem x_range (x : ℝ) : (x + 2) > 0 → (3 - x) ≥ 0 → (-2 < x ∧ x ≤ 3) :=
by
  intro h1 h2
  constructor
  { linarith }
  { linarith }

end NUMINAMATH_GPT_x_range_l1147_114737


namespace NUMINAMATH_GPT_find_polynomial_l1147_114776

theorem find_polynomial
  (M : ℝ → ℝ)
  (h : ∀ x, M x + 5 * x^2 - 4 * x - 3 = -1 * x^2 - 3 * x) :
  ∀ x, M x = -6 * x^2 + x + 3 :=
sorry

end NUMINAMATH_GPT_find_polynomial_l1147_114776


namespace NUMINAMATH_GPT_exists_c_same_digit_occurrences_l1147_114728

theorem exists_c_same_digit_occurrences (m n : ℕ) (hm : m > 0) (hn : n > 0) :
  ∃ c : ℕ, c > 0 ∧ ∀ d : ℕ, d ≠ 0 → 
    (Nat.digits 10 (c * m)).count d = (Nat.digits 10 (c * n)).count d := sorry

end NUMINAMATH_GPT_exists_c_same_digit_occurrences_l1147_114728


namespace NUMINAMATH_GPT_min_value_frac_l1147_114798

theorem min_value_frac (m n : ℝ) (hm : m > 0) (hn : n > 0) (h : m + n = 10) : 
  ∃ x, (x = (1 / m) + (4 / n)) ∧ (∀ y, y = (1 / m) + (4 / n) → y ≥ 9 / 10) :=
sorry

end NUMINAMATH_GPT_min_value_frac_l1147_114798


namespace NUMINAMATH_GPT_problem_statement_l1147_114743

theorem problem_statement (a b : ℝ) (h0 : 0 < b) (h1 : b < 1/2) (h2 : 1/2 < a) (h3 : a < 1) :
  (0 < a - b) ∧ (a - b < 1) ∧ (ab < a^2) ∧ (a - 1/b < b - 1/a) :=
by 
  sorry

end NUMINAMATH_GPT_problem_statement_l1147_114743


namespace NUMINAMATH_GPT_miles_mike_l1147_114710

def cost_mike (M : ℕ) : ℝ := 2.50 + 0.25 * M
def cost_annie (A : ℕ) : ℝ := 2.50 + 5.00 + 0.25 * A

theorem miles_mike {M A : ℕ} (annie_ride_miles : A = 16) (same_cost : cost_mike M = cost_annie A) : M = 36 :=
by
  rw [cost_annie, annie_ride_miles] at same_cost
  simp [cost_mike] at same_cost
  sorry

end NUMINAMATH_GPT_miles_mike_l1147_114710


namespace NUMINAMATH_GPT_intersection_eq_singleton_l1147_114702

-- Defining the sets M and N
def M : Set ℤ := {-1, 1, -2, 2}
def N : Set ℤ := {1, 4}

-- Stating the intersection problem
theorem intersection_eq_singleton :
  M ∩ N = {1} := 
by 
  sorry

end NUMINAMATH_GPT_intersection_eq_singleton_l1147_114702


namespace NUMINAMATH_GPT_isosceles_triangle_base_length_l1147_114782

noncomputable def length_of_base (a b : ℝ) (h_isosceles : a = b) (h_side : a = 3) (h_perimeter : a + b + x = 12) : ℝ :=
  (12 - 2 * a) / 2

theorem isosceles_triangle_base_length (a b : ℝ) (h_isosceles : a = b) (h_side : a = 3) (h_perimeter : a + b + x = 12) : length_of_base a b h_isosceles h_side h_perimeter = 4.5 :=
sorry

end NUMINAMATH_GPT_isosceles_triangle_base_length_l1147_114782


namespace NUMINAMATH_GPT_probability_same_color_opposite_foot_l1147_114771

def total_shoes := 28

def black_pairs := 7
def brown_pairs := 4
def gray_pairs := 2
def red_pair := 1

def total_pairs := black_pairs + brown_pairs + gray_pairs + red_pair

theorem probability_same_color_opposite_foot : 
  (7 + 4 + 2 + 1) * 2 = total_shoes →
  (14 / 28 * (7 / 27) + 8 / 28 * (4 / 27) + 4 / 28 * (2 / 27) + 2 / 28 * (1 / 27)) = (20 / 63) :=
by
  sorry

end NUMINAMATH_GPT_probability_same_color_opposite_foot_l1147_114771


namespace NUMINAMATH_GPT_calculate_Al2O3_weight_and_H2_volume_l1147_114713

noncomputable def weight_of_Al2O3 (moles : ℕ) : ℝ :=
  moles * ((2 * 26.98) + (3 * 16.00))

noncomputable def volume_of_H2_at_STP (moles_of_Al2O3 : ℕ) : ℝ :=
  (moles_of_Al2O3 * 3) * 22.4

theorem calculate_Al2O3_weight_and_H2_volume :
  weight_of_Al2O3 6 = 611.76 ∧ volume_of_H2_at_STP 6 = 403.2 :=
by
  sorry

end NUMINAMATH_GPT_calculate_Al2O3_weight_and_H2_volume_l1147_114713


namespace NUMINAMATH_GPT_number_of_cloth_bags_l1147_114703

-- Definitions based on the conditions
def dozen := 12

def total_peaches : ℕ := 5 * dozen
def peaches_in_knapsack : ℕ := 12
def peaches_per_bag : ℕ := 2 * peaches_in_knapsack

-- The proof statement
theorem number_of_cloth_bags :
  (total_peaches - peaches_in_knapsack) / peaches_per_bag = 2 := by
  sorry

end NUMINAMATH_GPT_number_of_cloth_bags_l1147_114703


namespace NUMINAMATH_GPT_jane_sandwich_count_l1147_114754

noncomputable def total_sandwiches : ℕ := 5 * 7 * 4

noncomputable def turkey_swiss_reduction : ℕ := 5 * 1 * 1

noncomputable def salami_bread_reduction : ℕ := 5 * 1 * 4

noncomputable def correct_sandwich_count : ℕ := 115

theorem jane_sandwich_count : total_sandwiches - turkey_swiss_reduction - salami_bread_reduction = correct_sandwich_count :=
by
  sorry

end NUMINAMATH_GPT_jane_sandwich_count_l1147_114754
