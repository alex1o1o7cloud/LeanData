import Mathlib

namespace set_D_is_empty_l2242_224218

theorem set_D_is_empty :
  {x : ℝ | x > 6 ∧ x < 1} = ∅ :=
by
  sorry

end set_D_is_empty_l2242_224218


namespace reciprocal_of_neg2019_l2242_224237

theorem reciprocal_of_neg2019 : (1 / -2019) = - (1 / 2019) := 
by
  sorry

end reciprocal_of_neg2019_l2242_224237


namespace sum_of_fraction_numerator_and_denominator_l2242_224267

theorem sum_of_fraction_numerator_and_denominator (x : ℚ) (a b : ℤ) :
  x = (45 / 99 : ℚ) ∧ (a = 5) ∧ (b = 11) → (a + b = 16) :=
by
  sorry

end sum_of_fraction_numerator_and_denominator_l2242_224267


namespace gcd_of_78_and_36_l2242_224231

theorem gcd_of_78_and_36 : Int.gcd 78 36 = 6 := by
  sorry

end gcd_of_78_and_36_l2242_224231


namespace day_of_50th_day_l2242_224273

theorem day_of_50th_day (days_250_N days_150_N1 : ℕ) 
  (h₁ : days_250_N % 7 = 5) (h₂ : days_150_N1 % 7 = 5) : 
  ((50 + 315 - 150 + 365 * 2) % 7) = 4 := 
  sorry

end day_of_50th_day_l2242_224273


namespace example_theorem_l2242_224240

theorem example_theorem :
∀ x : ℝ, 0 ≤ x ∧ x < 2 * Real.pi → (Real.sin x - Real.cos x = Real.sqrt 2) → x = 3 * Real.pi / 4 :=
by
  intros x h_range h_eq
  sorry

end example_theorem_l2242_224240


namespace sin_cos_product_neg_l2242_224220

theorem sin_cos_product_neg (α : ℝ) (h : Real.tan α < 0) : Real.sin α * Real.cos α < 0 :=
sorry

end sin_cos_product_neg_l2242_224220


namespace D_working_alone_completion_time_l2242_224203

variable (A_rate D_rate : ℝ)
variable (A_job_hours D_job_hours : ℝ)

-- Conditions
def A_can_complete_in_15_hours : Prop := (A_job_hours = 15)
def A_and_D_together_complete_in_10_hours : Prop := (1/A_rate + 1/D_rate = 10)

-- Proof statement
theorem D_working_alone_completion_time
  (hA : A_job_hours = 15)
  (hAD : 1/A_rate + 1/D_rate = 10) :
  D_job_hours = 30 := sorry

end D_working_alone_completion_time_l2242_224203


namespace diameter_of_circular_field_l2242_224254

noncomputable def diameter (C : ℝ) : ℝ := C / Real.pi

theorem diameter_of_circular_field :
  let cost_per_meter := 3
  let total_cost := 376.99
  let circumference := total_cost / cost_per_meter
  diameter circumference = 40 :=
by
  let cost_per_meter : ℝ := 3
  let total_cost : ℝ := 376.99
  let circumference : ℝ := total_cost / cost_per_meter
  have : circumference = 125.66333333333334 := by sorry
  have : diameter circumference = 40 := by sorry
  sorry

end diameter_of_circular_field_l2242_224254


namespace solve_for_x_l2242_224281

theorem solve_for_x (x : ℝ) 
  (h : (2 / (x + 3)) + (3 * x / (x + 3)) - (5 / (x + 3)) = 2) : 
  x = 9 := 
by
  sorry

end solve_for_x_l2242_224281


namespace john_payment_correct_l2242_224250

noncomputable def camera_value : ℝ := 5000
noncomputable def base_rental_fee_per_week : ℝ := 0.10 * camera_value
noncomputable def high_demand_fee_per_week : ℝ := base_rental_fee_per_week + 0.03 * camera_value
noncomputable def low_demand_fee_per_week : ℝ := base_rental_fee_per_week - 0.02 * camera_value
noncomputable def total_rental_fee : ℝ :=
  high_demand_fee_per_week + low_demand_fee_per_week + high_demand_fee_per_week + low_demand_fee_per_week
noncomputable def insurance_fee : ℝ := 0.05 * camera_value
noncomputable def pre_tax_total_cost : ℝ := total_rental_fee + insurance_fee
noncomputable def tax : ℝ := 0.08 * pre_tax_total_cost
noncomputable def total_cost : ℝ := pre_tax_total_cost + tax

noncomputable def mike_contribution : ℝ := 0.20 * total_cost
noncomputable def sarah_contribution : ℝ := min (0.30 * total_cost) 1000
noncomputable def alex_contribution : ℝ := min (0.10 * total_cost) 700
noncomputable def total_friends_contributions : ℝ := mike_contribution + sarah_contribution + alex_contribution

noncomputable def john_final_payment : ℝ := total_cost - total_friends_contributions

theorem john_payment_correct : john_final_payment = 1015.20 :=
by
  sorry

end john_payment_correct_l2242_224250


namespace Humphrey_birds_l2242_224235

-- Definitions for the given conditions:
def Marcus_birds : ℕ := 7
def Darrel_birds : ℕ := 9
def average_birds : ℕ := 9
def number_of_people : ℕ := 3

-- Proof statement
theorem Humphrey_birds : ∀ x : ℕ, (average_birds * number_of_people = Marcus_birds + Darrel_birds + x) → x = 11 :=
by
  intro x h
  sorry

end Humphrey_birds_l2242_224235


namespace equal_acutes_l2242_224270

open Real

theorem equal_acutes (a b c : ℝ) (ha : 0 < a ∧ a < π / 2) (hb : 0 < b ∧ b < π / 2) (hc : 0 < c ∧ c < π / 2)
  (h1 : sin b = (sin a + sin c) / 2) (h2 : cos b ^ 2 = cos a * cos c) : a = b ∧ b = c := 
by
  -- We have to fill the proof steps here.
  sorry

end equal_acutes_l2242_224270


namespace painting_combinations_l2242_224264

-- Define the conditions and the problem statement
def top_row_paint_count := 2
def total_lockers_per_row := 4
def valid_paintings := Nat.choose total_lockers_per_row top_row_paint_count

theorem painting_combinations : valid_paintings = 6 := by
  -- Use the derived conditions to provide the proof
  sorry

end painting_combinations_l2242_224264


namespace find_pairs_l2242_224241

def isDivisible (m n : ℕ) : Prop := ∃ k : ℕ, m = k * n
def isPrime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n
def satisfiesConditions (a b : ℕ) : Prop :=
  (isDivisible (a^2 + 6 * a + 8) b ∧
  (a^2 + a * b - 6 * b^2 - 15 * b - 9 = 0) ∧
  ¬ (a + 2 * b + 2) % 4 = 0 ∧
  isPrime (a + 6 * b + 2)) ∨
  (isDivisible (a^2 + 6 * a + 8) b ∧
  (a^2 + a * b - 6 * b^2 - 15 * b - 9 = 0) ∧
  ¬ (a + 2 * b + 2) % 4 = 0 ∧
  ¬ isPrime (a + 6 * b + 2))

theorem find_pairs (a b : ℕ) :
  (a = 5 ∧ b = 1) ∨ 
  (a = 17 ∧ b = 7) → 
  satisfiesConditions a b :=
by
  -- Proof to be completed
  sorry

end find_pairs_l2242_224241


namespace product_of_g_of_roots_l2242_224296

noncomputable def f (x : ℝ) : ℝ := x^5 - 2*x^3 + x + 1
noncomputable def g (x : ℝ) : ℝ := x^3 - 3*x + 1

theorem product_of_g_of_roots (x₁ x₂ x₃ x₄ x₅ : ℝ)
  (h₁ : f x₁ = 0) (h₂ : f x₂ = 0) (h₃ : f x₃ = 0)
  (h₄ : f x₄ = 0) (h₅ : f x₅ = 0) :
  g x₁ * g x₂ * g x₃ * g x₄ * g x₅ = f (-1 + Real.sqrt 2) * f (-1 - Real.sqrt 2) :=
by
  sorry

end product_of_g_of_roots_l2242_224296


namespace determine_1000g_weight_l2242_224210

-- Define the weights
def weights : List ℕ := [1000, 1001, 1002, 1004, 1007]

-- Define the weight sets
def Group1 : List ℕ := [weights.get! 0, weights.get! 1]
def Group2 : List ℕ := [weights.get! 2, weights.get! 3]
def Group3 : List ℕ := [weights.get! 4]

-- Definition to choose the lighter group or determine equality
def lighterGroup (g1 g2 : List ℕ) : List ℕ :=
  if g1.sum = g2.sum then Group3 else if g1.sum < g2.sum then g1 else g2

-- Determine the 1000 g weight functionally
def identify1000gWeightUsing3Weighings : ℕ :=
  let firstWeighing := lighterGroup Group1 Group2
  if firstWeighing = Group3 then Group3.get! 0 else
  let remainingWeights := firstWeighing
  if remainingWeights.get! 0 = remainingWeights.get! 1 then Group3.get! 0
  else if remainingWeights.get! 0 < remainingWeights.get! 1 then remainingWeights.get! 0 else remainingWeights.get! 1

theorem determine_1000g_weight : identify1000gWeightUsing3Weighings = 1000 :=
sorry

end determine_1000g_weight_l2242_224210


namespace avg_weight_class_l2242_224211

-- Definitions based on the conditions
def students_section_A : Nat := 36
def students_section_B : Nat := 24
def avg_weight_section_A : ℝ := 30.0
def avg_weight_section_B : ℝ := 30.0

-- The statement we want to prove
theorem avg_weight_class :
  (avg_weight_section_A * students_section_A + avg_weight_section_B * students_section_B) / (students_section_A + students_section_B) = 30.0 := 
by
  sorry

end avg_weight_class_l2242_224211


namespace negation_of_universal_statement_l2242_224202

theorem negation_of_universal_statement :
  (¬ ∀ x : ℝ, x^4 - x^3 + x^2 + 5 ≤ 0) ↔ (∃ x : ℝ, x^4 - x^3 + x^2 + 5 > 0) :=
by sorry

end negation_of_universal_statement_l2242_224202


namespace range_of_a_l2242_224263

open Real

theorem range_of_a (a : ℝ) 
  (h : ¬ ∃ x₀ : ℝ, 2 ^ x₀ - 2 ≤ a ^ 2 - 3 * a) : 1 ≤ a ∧ a ≤ 2 := 
by
  sorry

end range_of_a_l2242_224263


namespace find_quadratic_polynomial_l2242_224276

def quadratic_polynomial (a b c x : ℝ) : ℝ :=
  a * x^2 + b * x + c

theorem find_quadratic_polynomial : 
  ∃ a b c: ℝ, (∀ x : ℂ, quadratic_polynomial a b c x.re = 0 → (x = 3 + 4*I) ∨ (x = 3 - 4*I)) 
  ∧ (b = 8) 
  ∧ (a = -4/3) 
  ∧ (c = -50/3) :=
by
  sorry

end find_quadratic_polynomial_l2242_224276


namespace initial_investment_B_l2242_224294

theorem initial_investment_B (A_initial : ℝ) (B : ℝ) (total_profit : ℝ) (A_profit : ℝ) 
(A_withdraw : ℝ) (B_advance : ℝ) : 
  A_initial = 3000 → B_advance = 1000 → A_withdraw = 1000 → total_profit = 756 → A_profit = 288 → 
  (8 * A_initial + 4 * (A_initial - A_withdraw)) / (8 * B + 4 * (B + B_advance)) = A_profit / (total_profit - A_profit) → 
  B = 4000 := 
by 
  intros h1 h2 h3 h4 h5 h6 
  sorry

end initial_investment_B_l2242_224294


namespace hamburger_count_l2242_224214

-- Define the number of condiments and their possible combinations
def condiment_combinations : ℕ := 2 ^ 10

-- Define the number of choices for meat patties
def meat_patties_choices : ℕ := 4

-- Define the total count of different hamburgers
def total_hamburgers : ℕ := condiment_combinations * meat_patties_choices

-- The theorem statement proving the total number of different hamburgers
theorem hamburger_count : total_hamburgers = 4096 := by
  sorry

end hamburger_count_l2242_224214


namespace increase_in_expenses_is_20_percent_l2242_224284

noncomputable def man's_salary : ℝ := 6500
noncomputable def initial_savings : ℝ := 0.20 * man's_salary
noncomputable def new_savings : ℝ := 260
noncomputable def reduction_in_savings : ℝ := initial_savings - new_savings
noncomputable def initial_expenses : ℝ := 0.80 * man's_salary
noncomputable def increase_in_expenses_percentage : ℝ := (reduction_in_savings / initial_expenses) * 100

theorem increase_in_expenses_is_20_percent :
  increase_in_expenses_percentage = 20 := by
  sorry

end increase_in_expenses_is_20_percent_l2242_224284


namespace length_of_train_l2242_224279

noncomputable def train_length : ℕ := 1200

theorem length_of_train 
  (L : ℝ) 
  (speed_km_per_hr : ℝ) 
  (time_min : ℕ) 
  (speed_m_per_s : ℝ) 
  (time_sec : ℕ) 
  (distance : ℝ) 
  (cond1 : L = L)
  (cond2 : speed_km_per_hr = 144) 
  (cond3 : time_min = 1)
  (cond4 : speed_m_per_s = speed_km_per_hr * 1000 / 3600)
  (cond5 : time_sec = time_min * 60)
  (cond6 : distance = speed_m_per_s * time_sec)
  (cond7 : 2 * L = distance)
  : L = train_length := 
sorry

end length_of_train_l2242_224279


namespace inequality_solution_l2242_224239

noncomputable def log_b (b x : ℝ) := Real.log x / Real.log b

noncomputable def lhs (x : ℝ) := 
  log_b 5 250 + ((4 - (log_b 5 2) ^ 2) / (2 + log_b 5 2))

noncomputable def rhs (x : ℝ) := 
  125 ^ (log_b 5 x) ^ 2 - 24 * x ^ (log_b 5 x)

theorem inequality_solution (x : ℝ) : 
  (lhs x <= rhs x) ↔ (0 < x ∧ x ≤ 1/5) ∨ (5 ≤ x) := 
sorry

end inequality_solution_l2242_224239


namespace math_problem_l2242_224289

theorem math_problem
  (x y z : ℕ)
  (h1 : z = 4)
  (h2 : x + y = 7)
  (h3 : x + z = 8) :
  x + y + z = 11 := 
by
  sorry

end math_problem_l2242_224289


namespace horizontal_asymptote_of_rational_function_l2242_224219

theorem horizontal_asymptote_of_rational_function :
  ∀ (x : ℝ), (y = (7 * x^2 - 5) / (4 * x^2 + 6 * x + 3)) → (∃ b : ℝ, b = 7 / 4) :=
by
  intro x y
  sorry

end horizontal_asymptote_of_rational_function_l2242_224219


namespace Riku_stickers_more_times_l2242_224233

theorem Riku_stickers_more_times (Kristoff_stickers Riku_stickers : ℕ) 
  (h1 : Kristoff_stickers = 85) (h2 : Riku_stickers = 2210) : 
  Riku_stickers / Kristoff_stickers = 26 := 
by
  sorry

end Riku_stickers_more_times_l2242_224233


namespace cos_B_equals_3_over_4_l2242_224259

variables {A B C : ℝ} {a b c R : ℝ} (h₁ : b * Real.sin B - a * Real.sin A = (1/2) * a * Real.sin C)
  (h₂ :  2 * R ^ 2 * Real.sin B * (1 - Real.cos (2 * A)) = (1 / 2) * a * b * Real.sin C)

theorem cos_B_equals_3_over_4 : Real.cos B = 3 / 4 := by
  sorry

end cos_B_equals_3_over_4_l2242_224259


namespace find_S11_l2242_224268

variable {a : ℕ → ℚ} -- Define the arithmetic sequence as a function

-- Define conditions
def arithmetic_sequence (a : ℕ → ℚ) :=
∀ n m, a (n + m) = a n + a m

def S (n : ℕ) (a : ℕ → ℚ) : ℚ := (n / 2 : ℚ) * (a 1 + a n)

-- Define the problem statement to be proved
theorem find_S11 (h_arith : arithmetic_sequence a) (h_eq : a 3 + a 6 + a 9 = 54) : 
  S 11 a = 198 :=
sorry

end find_S11_l2242_224268


namespace tetrahedron_sum_l2242_224206

theorem tetrahedron_sum :
  let edges := 6
  let corners := 4
  let faces := 4
  edges + corners + faces = 14 :=
by
  sorry

end tetrahedron_sum_l2242_224206


namespace number_line_distance_l2242_224228

theorem number_line_distance (x : ℝ) : (abs (-3 - x) = 2) ↔ (x = -5 ∨ x = -1) :=
by
  sorry

end number_line_distance_l2242_224228


namespace problem_statement_l2242_224230

open Real Polynomial

theorem problem_statement (a1 a2 a3 d1 d2 d3 : ℝ) 
  (h : ∀ x : ℝ, x^8 - x^7 + x^6 - x^5 + x^4 - x^3 + x^2 - x + 1 =
                 (x^2 + a1 * x + d1) * (x^2 + a2 * x + d2) * (x^2 + a3 * x + d3) * (x^2 - 1)) :
  a1 * d1 + a2 * d2 + a3 * d3 = -1 := 
sorry

end problem_statement_l2242_224230


namespace percentage_of_alcohol_in_vessel_Q_l2242_224269

theorem percentage_of_alcohol_in_vessel_Q
  (x : ℝ)
  (h_mix : 2.5 + 0.04 * x = 6) :
  x = 87.5 :=
by
  sorry

end percentage_of_alcohol_in_vessel_Q_l2242_224269


namespace ratio_of_capitals_l2242_224280

noncomputable def Ashok_loss (total_loss : ℝ) (Pyarelal_loss : ℝ) : ℝ := total_loss - Pyarelal_loss

theorem ratio_of_capitals (total_loss : ℝ) (Pyarelal_loss : ℝ) (Ashok_capital Pyarelal_capital : ℝ) 
    (h_total_loss : total_loss = 1200)
    (h_Pyarelal_loss : Pyarelal_loss = 1080)
    (h_Ashok_capital : Ashok_capital = 120)
    (h_Pyarelal_capital : Pyarelal_capital = 1080) :
    Ashok_capital / Pyarelal_capital = 1 / 9 :=
by
  sorry

end ratio_of_capitals_l2242_224280


namespace exists_function_f_l2242_224222

theorem exists_function_f :
  ∃ f : ℕ → ℕ, ∀ n : ℕ, f (f n) = n * n :=
by
  sorry

end exists_function_f_l2242_224222


namespace production_analysis_l2242_224217

def daily_change (day: ℕ) : ℤ :=
  match day with
  | 0 => 40    -- Monday
  | 1 => -30   -- Tuesday
  | 2 => 90    -- Wednesday
  | 3 => -50   -- Thursday
  | 4 => -20   -- Friday
  | 5 => -10   -- Saturday
  | 6 => 20    -- Sunday
  | _ => 0     -- Invalid day, just in case

def planned_daily_production : ℤ := 500

def actual_production (day: ℕ) : ℤ :=
  planned_daily_production + (List.sum (List.map daily_change (List.range (day + 1))))

def total_production : ℤ :=
  List.sum (List.map actual_production (List.range 7))

theorem production_analysis :
  ∃ largest_increase_day smallest_increase_day : ℕ,
    largest_increase_day = 2 ∧  -- Wednesday
    smallest_increase_day = 1 ∧  -- Tuesday
    total_production = 3790 ∧
    total_production > 7 * planned_daily_production := by
  sorry

end production_analysis_l2242_224217


namespace inequality_holds_iff_m_range_l2242_224201

theorem inequality_holds_iff_m_range (m : ℝ) : (∀ x : ℝ, m * x^2 - 2 * m * x - 3 < 0) ↔ (-3 < m ∧ m ≤ 0) :=
by
  sorry

end inequality_holds_iff_m_range_l2242_224201


namespace two_b_is_16667_percent_of_a_l2242_224287

theorem two_b_is_16667_percent_of_a {a b : ℝ} (h : a = 1.2 * b) : (2 * b / a) = 5 / 3 := by
  sorry

end two_b_is_16667_percent_of_a_l2242_224287


namespace sequence_a_n_a_99_value_l2242_224244

theorem sequence_a_n_a_99_value :
  ∃ (a : ℕ → ℝ), a 1 = 3 ∧ (∀ n, 2 * (a (n + 1)) - 2 * (a n) = 1) ∧ a 99 = 52 :=
by {
  sorry
}

end sequence_a_n_a_99_value_l2242_224244


namespace cost_price_computer_table_l2242_224283

theorem cost_price_computer_table 
  (CP SP : ℝ)
  (h1 : SP = CP * 1.20)
  (h2 : SP = 8400) :
  CP = 7000 :=
by
  sorry

end cost_price_computer_table_l2242_224283


namespace files_remaining_correct_l2242_224238

-- Definitions for the original number of files
def music_files_original : ℕ := 4
def video_files_original : ℕ := 21
def document_files_original : ℕ := 12
def photo_files_original : ℕ := 30
def app_files_original : ℕ := 7

-- Definitions for the number of deleted files
def video_files_deleted : ℕ := 15
def document_files_deleted : ℕ := 10
def photo_files_deleted : ℕ := 18
def app_files_deleted : ℕ := 3

-- Definitions for the remaining number of files
def music_files_remaining : ℕ := music_files_original
def video_files_remaining : ℕ := video_files_original - video_files_deleted
def document_files_remaining : ℕ := document_files_original - document_files_deleted
def photo_files_remaining : ℕ := photo_files_original - photo_files_deleted
def app_files_remaining : ℕ := app_files_original - app_files_deleted

-- The proof problem statement
theorem files_remaining_correct : 
  music_files_remaining + video_files_remaining + document_files_remaining + photo_files_remaining + app_files_remaining = 28 :=
by
  rw [music_files_remaining, video_files_remaining, document_files_remaining, photo_files_remaining, app_files_remaining]
  exact rfl


end files_remaining_correct_l2242_224238


namespace smallest_n_inequality_l2242_224261

variable {x y z : ℝ}

theorem smallest_n_inequality :
  ∃ (n : ℕ), (∀ (x y z : ℝ), (x^2 + y^2 + z^2)^2 ≤ n * (x^4 + y^4 + z^4)) ∧
    (∀ m : ℕ, (∀ (x y z : ℝ), (x^2 + y^2 + z^2)^2 ≤ m * (x^4 + y^4 + z^4)) → n ≤ m) :=
sorry

end smallest_n_inequality_l2242_224261


namespace alpha_minus_beta_l2242_224236

-- Providing the conditions
variable (α β : ℝ)
variable (hα1 : 0 < α ∧ α < Real.pi / 2)
variable (hβ1 : 0 < β ∧ β < Real.pi / 2)
variable (hα2 : Real.tan α = 4 / 3)
variable (hβ2 : Real.tan β = 1 / 7)

-- The goal is to show that α - β = π / 4 given the conditions
theorem alpha_minus_beta :
  α - β = Real.pi / 4 := by
  sorry

end alpha_minus_beta_l2242_224236


namespace average_height_l2242_224288

theorem average_height (avg1 avg2 : ℕ) (n1 n2 : ℕ) (total_students : ℕ)
  (h1 : avg1 = 20) (h2 : avg2 = 20) (h3 : n1 = 20) (h4 : n2 = 11) (h5 : total_students = 31) :
  (n1 * avg1 + n2 * avg2) / total_students = 20 :=
by
  -- Placeholder for the proof
  sorry

end average_height_l2242_224288


namespace number_of_groups_of_bananas_l2242_224251

theorem number_of_groups_of_bananas (total_bananas : ℕ) (bananas_per_group : ℕ) (H_total_bananas : total_bananas = 290) (H_bananas_per_group : bananas_per_group = 145) :
    (total_bananas / bananas_per_group) = 2 :=
by {
  sorry
}

end number_of_groups_of_bananas_l2242_224251


namespace import_tax_applied_amount_l2242_224258

theorem import_tax_applied_amount 
    (total_value : ℝ) 
    (import_tax_paid : ℝ)
    (tax_rate : ℝ) 
    (excess_amount : ℝ) 
    (condition1 : total_value = 2580) 
    (condition2 : import_tax_paid = 110.60) 
    (condition3 : tax_rate = 0.07) 
    (condition4 : import_tax_paid = tax_rate * (total_value - excess_amount)) : 
    excess_amount = 1000 :=
by
  sorry

end import_tax_applied_amount_l2242_224258


namespace profit_function_equation_maximum_profit_l2242_224252

noncomputable def production_cost (x : ℝ) : ℝ := x^3 - 24*x^2 + 63*x + 10
noncomputable def sales_revenue (x : ℝ) : ℝ := 18*x
noncomputable def production_profit (x : ℝ) : ℝ := sales_revenue x - production_cost x

theorem profit_function_equation (x : ℝ) : production_profit x = -x^3 + 24*x^2 - 45*x - 10 :=
  by
    unfold production_profit sales_revenue production_cost
    sorry

theorem maximum_profit : (production_profit 15 = 1340) ∧ ∀ x, production_profit 15 ≥ production_profit x :=
  by
    sorry

end profit_function_equation_maximum_profit_l2242_224252


namespace initial_fee_correct_l2242_224285

-- Define the relevant values
def initialFee := 2.25
def chargePerSegment := 0.4
def totalDistance := 3.6
def totalCharge := 5.85
noncomputable def segments := (totalDistance * (5 / 2))
noncomputable def costForDistance := segments * chargePerSegment

-- Define the theorem
theorem initial_fee_correct :
  totalCharge = initialFee + costForDistance :=
by
  -- Proof is omitted.
  sorry

end initial_fee_correct_l2242_224285


namespace comparison_l2242_224272

noncomputable def a := Real.log 3000 / Real.log 9
noncomputable def b := Real.log 2023 / Real.log 4
noncomputable def c := (11 * Real.exp (0.01 * Real.log 1.001)) / 2

theorem comparison : a < b ∧ b < c :=
by
  sorry

end comparison_l2242_224272


namespace gcd_1113_1897_l2242_224221

theorem gcd_1113_1897 : Int.gcd 1113 1897 = 7 := by
  sorry

end gcd_1113_1897_l2242_224221


namespace rotate_circle_sectors_l2242_224243

theorem rotate_circle_sectors (n : ℕ) (h : n > 0) :
  (∀ i, i < n → ∃ θ : ℝ, θ < (π / (n^2 - n + 1))) →
  ∃ θ : ℝ, 0 < θ ∧ θ < 2 * π ∧
  (∀ i : ℕ, i < n → (θ * i) % (2 * π) > (π / (n^2 - n + 1))) :=
sorry

end rotate_circle_sectors_l2242_224243


namespace wrapping_paper_solution_l2242_224291

variable (P1 P2 P3 : ℝ)

def wrapping_paper_problem : Prop :=
  P1 = 2 ∧
  P3 = P1 + P2 ∧
  P1 + P2 + P3 = 7 →
  (P2 / P1) = 3 / 4

theorem wrapping_paper_solution : wrapping_paper_problem P1 P2 P3 :=
by
  sorry

end wrapping_paper_solution_l2242_224291


namespace periodic_even_function_l2242_224275

open Real

noncomputable def f : ℝ → ℝ := sorry

theorem periodic_even_function (f : ℝ → ℝ)
  (h1 : ∀ x, f (x + 2) = f x)
  (h2 : ∀ x, f (-x) = f x)
  (h3 : ∀ x, 2 ≤ x ∧ x ≤ 3 → f x = x) :
  ∀ x, -2 ≤ x ∧ x ≤ 0 → f x = 3 - abs (x + 1) :=
sorry

end periodic_even_function_l2242_224275


namespace liza_final_balance_l2242_224262

def initial_balance : ℕ := 800
def rent : ℕ := 450
def paycheck : ℕ := 1500
def electricity_bill : ℕ := 117
def internet_bill : ℕ := 100
def phone_bill : ℕ := 70

theorem liza_final_balance :
  initial_balance - rent + paycheck - (electricity_bill + internet_bill) - phone_bill = 1563 := by
  sorry

end liza_final_balance_l2242_224262


namespace fill_time_correct_l2242_224277

-- Define the conditions
def rightEyeTime := 2 * 24 -- hours
def leftEyeTime := 3 * 24 -- hours
def rightFootTime := 4 * 24 -- hours
def throatTime := 6       -- hours

def rightEyeRate := 1 / rightEyeTime
def leftEyeRate := 1 / leftEyeTime
def rightFootRate := 1 / rightFootTime
def throatRate := 1 / throatTime

-- Combined rate calculation
def combinedRate := rightEyeRate + leftEyeRate + rightFootRate + throatRate

-- Goal definition
def fillTime := 288 / 61 -- hours

-- Prove that the calculated time to fill the pool matches the given answer
theorem fill_time_correct : (1 / combinedRate) = fillTime :=
by {
  sorry
}

end fill_time_correct_l2242_224277


namespace evaluate_expr_l2242_224282

def x := 2
def y := -1
def z := 3
def expr := 2 * x^2 + y^2 - z^2 + 3 * x * y

theorem evaluate_expr : expr = -6 :=
by sorry

end evaluate_expr_l2242_224282


namespace discount_profit_percentage_l2242_224209

theorem discount_profit_percentage (CP : ℝ) (P_no_discount : ℝ) (D : ℝ) (profit_with_discount : ℝ) (SP_no_discount : ℝ) (SP_discount : ℝ) :
  P_no_discount = 50 ∧ D = 4 ∧ SP_no_discount = CP + 0.5 * CP ∧ SP_discount = SP_no_discount - (D / 100) * SP_no_discount ∧ profit_with_discount = SP_discount - CP →
  (profit_with_discount / CP) * 100 = 44 :=
by sorry

end discount_profit_percentage_l2242_224209


namespace can_construct_prism_with_fewer_than_20_shapes_l2242_224223

/-
  We have 5 congruent unit cubes glued together to form complex shapes.
  4 of these cubes form a 4-unit high prism, and the fifth is attached to one of the inner cubes with a full face.
  Prove that we can construct a solid rectangular prism using fewer than 20 of these shapes.
-/

theorem can_construct_prism_with_fewer_than_20_shapes :
  ∃ (n : ℕ), n < 20 ∧ (∃ (length width height : ℕ), length * width * height = 5 * n) :=
sorry

end can_construct_prism_with_fewer_than_20_shapes_l2242_224223


namespace binary_add_sub_l2242_224246

theorem binary_add_sub : 
  (1101 + 111 - 101 + 1001 - 11 : ℕ) = (10101 : ℕ) := by
  sorry

end binary_add_sub_l2242_224246


namespace jerry_initial_action_figures_l2242_224245

theorem jerry_initial_action_figures 
(A : ℕ) 
(h1 : ∀ A, A + 7 = 9 + 3)
: A = 5 :=
by
  sorry

end jerry_initial_action_figures_l2242_224245


namespace avery_donation_l2242_224299

theorem avery_donation (shirts pants shorts : ℕ)
  (h_shirts : shirts = 4)
  (h_pants : pants = 2 * shirts)
  (h_shorts : shorts = pants / 2) :
  shirts + pants + shorts = 16 := by
  sorry

end avery_donation_l2242_224299


namespace jackie_sleeping_hours_l2242_224248

def hours_in_a_day : ℕ := 24
def work_hours : ℕ := 8
def exercise_hours : ℕ := 3
def free_time_hours : ℕ := 5
def accounted_hours : ℕ := work_hours + exercise_hours + free_time_hours

theorem jackie_sleeping_hours :
  hours_in_a_day - accounted_hours = 8 := by
  sorry

end jackie_sleeping_hours_l2242_224248


namespace mass_percentage_I_in_CaI2_l2242_224255

theorem mass_percentage_I_in_CaI2 :
  let molar_mass_Ca : ℝ := 40.08
  let molar_mass_I : ℝ := 126.90
  let molar_mass_CaI2 : ℝ := molar_mass_Ca + 2 * molar_mass_I
  let mass_percentage_I : ℝ := (2 * molar_mass_I / molar_mass_CaI2) * 100
  mass_percentage_I = 86.36 := by
  sorry

end mass_percentage_I_in_CaI2_l2242_224255


namespace largest_possible_a_l2242_224232

theorem largest_possible_a (a b c d : ℕ) (ha : a < 2 * b) (hb : b < 3 * c) (hc : c < 4 * d) (hd : d < 100) : 
  a ≤ 2367 :=
sorry

end largest_possible_a_l2242_224232


namespace triangle_sides_length_a_triangle_perimeter_l2242_224215

theorem triangle_sides_length_a (A B C : ℝ) (a b c : ℝ) 
  (hA : A = π / 3) (h1 : (b + c) / (Real.sin B + Real.sin C) = 2) :
  a = Real.sqrt 3 :=
sorry

theorem triangle_perimeter (A B C : ℝ) (a b c : ℝ) 
  (hA : A = π / 3) (h1 : (b + c) / (Real.sin B + Real.sin C) = 2) 
  (h2 : (b * c * Real.sin (π / 3)) / 2 = Real.sqrt 3 / 2) :
  a + b + c = 3 + Real.sqrt 3 :=
sorry

end triangle_sides_length_a_triangle_perimeter_l2242_224215


namespace certain_number_unique_l2242_224226

theorem certain_number_unique (x : ℝ) (hx1 : 213 * x = 3408) (hx2 : 21.3 * x = 340.8) : x = 16 :=
by
  sorry

end certain_number_unique_l2242_224226


namespace interior_angle_of_regular_hexagon_l2242_224207

theorem interior_angle_of_regular_hexagon : 
  ∀ (n : ℕ), n = 6 → (∃ sumInteriorAngles : ℕ, sumInteriorAngles = (n - 2) * 180) →
  ∀ (interiorAngle : ℕ), (∃ sumInteriorAngles : ℕ, sumInteriorAngles = 720) → 
  interiorAngle = sumInteriorAngles / 6 →
  interiorAngle = 120 :=
by
  sorry

end interior_angle_of_regular_hexagon_l2242_224207


namespace quadratic_function_opens_downwards_l2242_224234

theorem quadratic_function_opens_downwards (m : ℝ) (h₁ : m - 1 < 0) (h₂ : m^2 + 1 = 2) : m = -1 :=
by {
  -- Proof would go here.
  sorry
}

end quadratic_function_opens_downwards_l2242_224234


namespace Thelma_cuts_each_tomato_into_8_slices_l2242_224247

-- Conditions given in the problem
def slices_per_meal := 20
def family_size := 8
def tomatoes_needed := 20

-- The quantity we want to prove
def slices_per_tomato := 8

-- Statement to be proven: Thelma cuts each green tomato into the correct number of slices
theorem Thelma_cuts_each_tomato_into_8_slices :
  (slices_per_meal * family_size) = (tomatoes_needed * slices_per_tomato) :=
by 
  sorry

end Thelma_cuts_each_tomato_into_8_slices_l2242_224247


namespace number_is_composite_l2242_224242

theorem number_is_composite : ∃ k l : ℕ, k * l = 53 * 83 * 109 + 40 * 66 * 96 ∧ k > 1 ∧ l > 1 :=
by
  have h1 : 53 + 96 = 149 := by norm_num
  have h2 : 83 + 66 = 149 := by norm_num
  have h3 : 109 + 40 = 149 := by norm_num
  sorry

end number_is_composite_l2242_224242


namespace express_y_in_terms_of_x_l2242_224204

theorem express_y_in_terms_of_x (x y : ℝ) (h : 4 * x - y = 7) : y = 4 * x - 7 :=
sorry

end express_y_in_terms_of_x_l2242_224204


namespace drops_of_glue_needed_l2242_224257

def number_of_clippings (friend : ℕ) : ℕ :=
  match friend with
  | 1 => 4
  | 2 => 7
  | 3 => 5
  | 4 => 3
  | 5 => 5
  | 6 => 8
  | 7 => 2
  | 8 => 6
  | _ => 0

def total_drops_of_glue : ℕ :=
  (number_of_clippings 1 +
   number_of_clippings 2 +
   number_of_clippings 3 +
   number_of_clippings 4 +
   number_of_clippings 5 +
   number_of_clippings 6 +
   number_of_clippings 7 +
   number_of_clippings 8) * 6

theorem drops_of_glue_needed : total_drops_of_glue = 240 :=
by
  sorry

end drops_of_glue_needed_l2242_224257


namespace solution_set_of_inequality_l2242_224292

theorem solution_set_of_inequality :
  {x : ℝ | (3 * x - 1) / (2 - x) ≥ 0} = {x : ℝ | 1/3 ≤ x ∧ x < 2} :=
by
  sorry

end solution_set_of_inequality_l2242_224292


namespace union_set_equiv_l2242_224227

namespace ProofProblem

-- Define the sets A and B
def A : Set ℝ := { x | x - 1 > 0 }
def B : Set ℝ := { x | x^2 - x - 2 > 0 }

-- Define the union of A and B
def unionAB : Set ℝ := A ∪ B

-- State the proof problem
theorem union_set_equiv : unionAB = (Set.Iio (-1)) ∪ (Set.Ioi 1) := by
  sorry

end ProofProblem

end union_set_equiv_l2242_224227


namespace problem_integer_pairs_l2242_224249

theorem problem_integer_pairs (a b q r : ℕ) (h1 : a > 0) (h2 : b > 0) (h3 : a^2 + b^2 = q * (a + b) + r) (h4 : q^2 + r = 1977) :
    (a, b) = (50, 7) ∨ (a, b) = (50, 37) ∨ (a, b) = (7, 50) ∨ (a, b) = (37, 50) :=
sorry

end problem_integer_pairs_l2242_224249


namespace necessary_but_not_sufficient_condition_l2242_224200

theorem necessary_but_not_sufficient_condition (p : ℝ) : 
  p < 2 → (¬(p^2 - 4 < 0) → ∃ q, q < p ∧ q^2 - 4 < 0) :=
by
  sorry

end necessary_but_not_sufficient_condition_l2242_224200


namespace range_of_a_l2242_224225

-- Definitions related to the conditions in the problem
def polynomial (x : ℝ) (a : ℝ) (b : ℝ) : ℝ := 3 * x ^ 5 - 4 * a * x ^ 3 + 2 * b ^ 2 * x ^ 2 + 1

def v_2 (x : ℝ) (a : ℝ) : ℝ := (3 * x + 0) * x - 4 * a

def v_3 (x : ℝ) (a : ℝ) (b : ℝ) : ℝ := (((3 * x + 0) * x - 4 * a) * x + 2 * b ^ 2)

-- The main statement to prove
theorem range_of_a (x a b : ℝ) (h1 : x = 2) (h2 : ∀ b : ℝ, (v_2 x a) < (v_3 x a b)) : a < 3 :=
by
  sorry

end range_of_a_l2242_224225


namespace correlation_coefficient_value_relation_between_gender_and_electric_car_expectation_X_value_l2242_224216

-- Definition 1: Variance and regression coefficients and correlation coefficient calculation
noncomputable def correlation_coefficient : ℝ := 4.7 * (Real.sqrt (2 / 50))

-- Theorem 1: Correlation coefficient computation
theorem correlation_coefficient_value :
  correlation_coefficient = 0.94 :=
sorry

-- Definition 2: Chi-square calculation for independence test
noncomputable def chi_square : ℝ :=
  (100 * ((30 * 35 - 20 * 15)^2 : ℝ)) / (50 * 50 * 45 * 55)

-- Theorem 2: Chi-square test result
theorem relation_between_gender_and_electric_car :
  chi_square > 6.635 :=
sorry

-- Definition 3: Probability distribution and expectation calculation
def probability_distribution : Finset ℚ :=
{(21/55), (28/55), (6/55)}

noncomputable def expectation_X : ℚ :=
(0 * (21/55) + 1 * (28/55) + 2 * (6/55))

-- Theorem 3: Expectation of X calculation
theorem expectation_X_value :
  expectation_X = 8/11 :=
sorry

end correlation_coefficient_value_relation_between_gender_and_electric_car_expectation_X_value_l2242_224216


namespace freddy_travel_time_l2242_224213

theorem freddy_travel_time (dist_A_B : ℝ) (time_Eddy : ℝ) (dist_A_C : ℝ) (speed_ratio : ℝ) (travel_time_Freddy : ℝ) :
  dist_A_B = 540 ∧ time_Eddy = 3 ∧ dist_A_C = 300 ∧ speed_ratio = 2.4 →
  travel_time_Freddy = dist_A_C / (dist_A_B / time_Eddy / speed_ratio) :=
  sorry

end freddy_travel_time_l2242_224213


namespace total_pages_is_905_l2242_224265

def history_pages : ℕ := 160
def geography_pages : ℕ := history_pages + 70
def math_pages : ℕ := (history_pages + geography_pages) / 2
def science_pages : ℕ := 2 * history_pages
def total_pages : ℕ := history_pages + geography_pages + math_pages + science_pages

theorem total_pages_is_905 : total_pages = 905 := by
  sorry

end total_pages_is_905_l2242_224265


namespace basketball_scores_l2242_224295

theorem basketball_scores :
  ∃ P: Finset ℕ, (∀ x y: ℕ, (x + y = 7 → P = {p | ∃ x y: ℕ, p = 3 * x + 2 * y})) ∧ (P.card = 8) :=
sorry

end basketball_scores_l2242_224295


namespace trigonometric_identity_l2242_224271

open Real

theorem trigonometric_identity
  (α β γ φ : ℝ)
  (h1 : sin α + 7 * sin β = 4 * (sin γ + 2 * sin φ))
  (h2 : cos α + 7 * cos β = 4 * (cos γ + 2 * cos φ)) :
  2 * cos (α - φ) = 7 * cos (β - γ) :=
by sorry

end trigonometric_identity_l2242_224271


namespace inequality_problem_l2242_224256

theorem inequality_problem (a b c : ℝ) (h : a < b ∧ b < 0) : a^2 > a * b ∧ a * b > b^2 :=
by
  -- The proof is supposed to be here
  sorry

end inequality_problem_l2242_224256


namespace max_complexity_51_l2242_224260

-- Define the complexity of a number 
def complexity (x : ℚ) : ℕ := sorry -- Placeholder for the actual complexity function definition

-- Define the sequence for m values
def m_sequence (k : ℕ) : List ℕ :=
  List.range' 1 (2^(k-1)) |>.filter (λ n => n % 2 = 1)

-- Define the candidate number
def candidate_number (k : ℕ) : ℚ :=
  (2^(k + 1) + (-1)^k) / (3 * 2^k)

theorem max_complexity_51 : 
  ∃ m, m ∈ m_sequence 50 ∧ 
  (∀ n, n ∈ m_sequence 50 → complexity (n / 2^50) ≤ complexity (candidate_number 50 / 2^50)) :=
sorry

end max_complexity_51_l2242_224260


namespace problem_diamond_value_l2242_224278

def diamond (x y : ℕ) : ℕ := 4 * x + 6 * y

theorem problem_diamond_value :
  diamond 3 4 = 36 := 
by
  sorry

end problem_diamond_value_l2242_224278


namespace find_b_value_l2242_224205

theorem find_b_value (a b : ℤ) (h₁ : a + 2 * b = 32) (h₂ : |a| > 2) (h₃ : a = 4) : b = 14 :=
by
  -- proof goes here
  sorry

end find_b_value_l2242_224205


namespace rotations_needed_to_reach_goal_l2242_224286

-- Define the given conditions
def rotations_per_block : ℕ := 200
def blocks_goal : ℕ := 8
def current_rotations : ℕ := 600

-- Define total_rotations_needed and more_rotations_needed
def total_rotations_needed : ℕ := blocks_goal * rotations_per_block
def more_rotations_needed : ℕ := total_rotations_needed - current_rotations

-- Theorem stating the solution
theorem rotations_needed_to_reach_goal : more_rotations_needed = 1000 := by
  -- proof steps are omitted
  sorry

end rotations_needed_to_reach_goal_l2242_224286


namespace Clara_sells_third_type_boxes_l2242_224290

variable (total_cookies boxes_first boxes_second boxes_third : ℕ)
variable (cookies_per_first cookies_per_second cookies_per_third : ℕ)

theorem Clara_sells_third_type_boxes (h1 : cookies_per_first = 12)
                                    (h2 : boxes_first = 50)
                                    (h3 : cookies_per_second = 20)
                                    (h4 : boxes_second = 80)
                                    (h5 : cookies_per_third = 16)
                                    (h6 : total_cookies = 3320) :
                                    boxes_third = 70 :=
by
  sorry

end Clara_sells_third_type_boxes_l2242_224290


namespace overall_percentage_decrease_l2242_224266

theorem overall_percentage_decrease (P x y : ℝ) (hP : P = 100) 
  (h : (P - (x / 100) * P) - (y / 100) * (P - (x / 100) * P) = 55) : 
  ((P - 55) / P) * 100 = 45 := 
by 
  sorry

end overall_percentage_decrease_l2242_224266


namespace Faye_created_rows_l2242_224297

theorem Faye_created_rows (total_crayons : ℕ) (crayons_per_row : ℕ) (rows : ℕ) 
  (h1 : total_crayons = 210) (h2 : crayons_per_row = 30) : rows = 7 :=
by
  sorry

end Faye_created_rows_l2242_224297


namespace a_and_b_are_kth_powers_l2242_224274

theorem a_and_b_are_kth_powers (k : ℕ) (h_k : 1 < k) (a b : ℤ) (h_rel_prime : Int.gcd a b = 1)
  (c : ℤ) (h_ab_power : a * b = c^k) : ∃ (m n : ℤ), a = m^k ∧ b = n^k :=
by
  sorry

end a_and_b_are_kth_powers_l2242_224274


namespace boat_navigation_under_arch_l2242_224224

theorem boat_navigation_under_arch (h_arch : ℝ) (w_arch: ℝ) (boat_width: ℝ) (boat_height: ℝ) (boat_above_water: ℝ) :
  (h_arch = 5) → 
  (w_arch = 8) → 
  (boat_width = 4) → 
  (boat_height = 2) → 
  (boat_above_water = 0.75) →
  (h_arch - 2 = 3) :=
by
  intros h_arch_eq w_arch_eq boat_w_eq boat_h_eq boat_above_water_eq
  sorry

end boat_navigation_under_arch_l2242_224224


namespace new_savings_after_expense_increase_l2242_224293

theorem new_savings_after_expense_increase
    (monthly_salary : ℝ)
    (initial_saving_percent : ℝ)
    (expense_increase_percent : ℝ)
    (initial_salary : monthly_salary = 20000)
    (saving_rate : initial_saving_percent = 0.1)
    (increase_rate : expense_increase_percent = 0.1) :
    monthly_salary - (monthly_salary * (1 - initial_saving_percent + (1 - initial_saving_percent) * expense_increase_percent)) = 200 :=
by
  sorry

end new_savings_after_expense_increase_l2242_224293


namespace baron_munchausen_correct_l2242_224298

noncomputable def P (x : ℕ) : ℕ := sorry -- Assume non-constant polynomial with non-negative integer coefficients
noncomputable def Q (x : ℕ) : ℕ := sorry -- Assume non-constant polynomial with non-negative integer coefficients

theorem baron_munchausen_correct (b p0 : ℕ) 
  (hP2 : P 2 = b) 
  (hPp2 : P b = p0) 
  (hQ2 : Q 2 = b) 
  (hQp2 : Q b = p0) : 
  P = Q := sorry

end baron_munchausen_correct_l2242_224298


namespace geometric_sequence_first_term_l2242_224229

noncomputable def first_term_of_geometric_sequence (a r : ℝ) : ℝ :=
  a

theorem geometric_sequence_first_term 
  (a r : ℝ)
  (h1 : a * r^3 = 720)   -- The fourth term is 6!
  (h2 : a * r^6 = 5040)  -- The seventh term is 7!
  : first_term_of_geometric_sequence a r = 720 / 7 :=
sorry

end geometric_sequence_first_term_l2242_224229


namespace sales_volume_expression_reduction_for_desired_profit_l2242_224208

-- Initial conditions definitions.
def initial_purchase_price : ℝ := 3
def initial_selling_price : ℝ := 5
def initial_sales_volume : ℝ := 100
def sales_increase_per_0_1_yuan : ℝ := 20
def desired_profit : ℝ := 300
def minimum_sales_volume : ℝ := 220

-- Question (1): Sales Volume Expression
theorem sales_volume_expression (x : ℝ) : initial_sales_volume + (sales_increase_per_0_1_yuan * 10 * x) = 100 + 200 * x :=
by sorry

-- Question (2): Determine Reduction for Desired Profit and Minimum Sales Volume
theorem reduction_for_desired_profit (x : ℝ) 
  (hx : (initial_selling_price - initial_purchase_price - x) * (initial_sales_volume + (sales_increase_per_0_1_yuan * 10 * x)) = desired_profit)
  (hy : initial_sales_volume + (sales_increase_per_0_1_yuan * 10 * x) >= minimum_sales_volume) :
  x = 1 :=
by sorry

end sales_volume_expression_reduction_for_desired_profit_l2242_224208


namespace students_exceed_rabbits_l2242_224253

theorem students_exceed_rabbits (students_per_classroom rabbits_per_classroom number_of_classrooms : ℕ) 
  (h_students : students_per_classroom = 18)
  (h_rabbits : rabbits_per_classroom = 2)
  (h_classrooms : number_of_classrooms = 4) : 
  (students_per_classroom * number_of_classrooms) - (rabbits_per_classroom * number_of_classrooms) = 64 :=
by {
  sorry
}

end students_exceed_rabbits_l2242_224253


namespace find_f_value_l2242_224212

noncomputable def f (a b c : ℝ) (x : ℝ) : ℝ :=
  a * x^5 - b * x^3 + c * x - 3

theorem find_f_value (a b c : ℝ) (h : f a b c (-3) = 7) : f a b c 3 = -13 :=
by
  sorry

end find_f_value_l2242_224212
