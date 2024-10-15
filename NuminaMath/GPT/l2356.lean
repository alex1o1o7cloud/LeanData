import Mathlib

namespace NUMINAMATH_GPT_Dacid_weighted_average_l2356_235627

noncomputable def DacidMarks := 86 * 3 + 85 * 4 + 92 * 4 + 87 * 3 + 95 * 3 + 89 * 2 + 75 * 1
noncomputable def TotalCreditHours := 3 + 4 + 4 + 3 + 3 + 2 + 1
noncomputable def WeightedAverageMarks := (DacidMarks : ℝ) / (TotalCreditHours : ℝ)

theorem Dacid_weighted_average :
  WeightedAverageMarks = 88.25 :=
sorry

end NUMINAMATH_GPT_Dacid_weighted_average_l2356_235627


namespace NUMINAMATH_GPT_g_possible_values_l2356_235655

noncomputable def g (x : ℝ) : ℝ := 
  Real.arctan x + Real.arctan ((x - 1) / (x + 1)) + Real.arctan (1 / x)

theorem g_possible_values (x : ℝ) (hx₁ : x ≠ 0) (hx₂ : x ≠ -1) (hx₃ : x ≠ 1) :
  g x = (Real.pi / 4) ∨ g x = (5 * Real.pi / 4) :=
sorry

end NUMINAMATH_GPT_g_possible_values_l2356_235655


namespace NUMINAMATH_GPT_range_of_x_satisfying_inequality_l2356_235676

theorem range_of_x_satisfying_inequality (x : ℝ) : x^2 < |x| ↔ (x > -1 ∧ x < 0) ∨ (x > 0 ∧ x < 1) :=
by
  sorry

end NUMINAMATH_GPT_range_of_x_satisfying_inequality_l2356_235676


namespace NUMINAMATH_GPT_possible_values_a_l2356_235659

noncomputable def setA (a : ℝ) : Set ℝ := { x | a * x + 2 = 0 }
def setB : Set ℝ := {-1, 2}

theorem possible_values_a :
  ∀ a : ℝ, setA a ⊆ setB ↔ a = -1 ∨ a = 0 ∨ a = 2 :=
by
  intro a
  sorry

end NUMINAMATH_GPT_possible_values_a_l2356_235659


namespace NUMINAMATH_GPT_slope_range_l2356_235652

theorem slope_range (k : ℝ) : 
  (∃ (x : ℝ), ∀ (y : ℝ), y = k * (x - 1) + 1) ∧ (0 < 1 - k ∧ 1 - k < 2) → (-1 < k ∧ k < 1) :=
by
  sorry

end NUMINAMATH_GPT_slope_range_l2356_235652


namespace NUMINAMATH_GPT_sum_of_positive_factors_of_90_eq_234_l2356_235693

theorem sum_of_positive_factors_of_90_eq_234 : 
  let factors := [1, 2, 3, 5, 6, 9, 10, 15, 18, 30, 45, 90]
  List.sum factors = 234 :=
by
  -- List the positive factors of 90
  let factors := [1, 2, 3, 5, 6, 9, 10, 15, 18, 30, 45, 90]
  -- Prove that the sum of these factors is 234
  have h_sum_factors : List.sum factors = 234 := sorry
  exact h_sum_factors

end NUMINAMATH_GPT_sum_of_positive_factors_of_90_eq_234_l2356_235693


namespace NUMINAMATH_GPT_quadratic_solution_l2356_235689

theorem quadratic_solution (x : ℝ) : 2 * x * (x + 1) = 3 * (x + 1) ↔ (x = -1 ∨ x = 3 / 2) := by
  sorry

end NUMINAMATH_GPT_quadratic_solution_l2356_235689


namespace NUMINAMATH_GPT_val_need_33_stamps_l2356_235674

def valerie_needs_total_stamps 
    (thank_you_cards : ℕ) 
    (bills_water : ℕ) 
    (bills_electric : ℕ) 
    (bills_internet : ℕ) 
    (rebate_addition : ℕ) 
    (rebate_stamps : ℕ) 
    (job_apps_multiplier : ℕ) 
    (job_app_stamps : ℕ) 
    (total_stamps : ℕ) : Prop :=
    thank_you_cards = 3 ∧
    bills_water = 1 ∧
    bills_electric = 2 ∧
    bills_internet = 3 ∧
    rebate_addition = 3 ∧
    rebate_stamps = 2 ∧
    job_apps_multiplier = 2 ∧
    job_app_stamps = 1 ∧
    total_stamps = 33

theorem val_need_33_stamps : 
  valerie_needs_total_stamps 3 1 2 3 3 2 2 1 33 :=
by 
  -- proof skipped
  sorry

end NUMINAMATH_GPT_val_need_33_stamps_l2356_235674


namespace NUMINAMATH_GPT_range_of_m_l2356_235643

-- Definitions
def is_circle_eqn (d e f : ℝ) : Prop :=
  d^2 + e^2 - 4 * f > 0

-- Main statement 
theorem range_of_m (m : ℝ) : 
  is_circle_eqn (-2) (-4) m → m < 5 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_range_of_m_l2356_235643


namespace NUMINAMATH_GPT_find_required_water_amount_l2356_235635

-- Definitions based on the conditions
def sanitizer_volume : ℝ := 12
def initial_alcohol_concentration : ℝ := 0.60
def desired_alcohol_concentration : ℝ := 0.40

-- Statement of the proof problem
theorem find_required_water_amount : 
  ∃ (x : ℝ), x = 6 ∧ sanitizer_volume * initial_alcohol_concentration = desired_alcohol_concentration * (sanitizer_volume + x) :=
sorry

end NUMINAMATH_GPT_find_required_water_amount_l2356_235635


namespace NUMINAMATH_GPT_cherries_count_l2356_235610

theorem cherries_count (b s r c : ℝ) 
  (h1 : b + s + r + c = 360)
  (h2 : s = 2 * b)
  (h3 : r = 4 * s)
  (h4 : c = 2 * r) : 
  c = 640 / 3 :=
by 
  sorry

end NUMINAMATH_GPT_cherries_count_l2356_235610


namespace NUMINAMATH_GPT_c_minus_b_seven_l2356_235600

theorem c_minus_b_seven {a b c d : ℕ} (ha : a^6 = b^5) (hb : c^4 = d^3) (hc : c - a = 31) : c - b = 7 :=
sorry

end NUMINAMATH_GPT_c_minus_b_seven_l2356_235600


namespace NUMINAMATH_GPT_parabola_rotation_180_equivalent_l2356_235657

-- Define the original parabola equation
def original_parabola (x : ℝ) : ℝ := 2 * (x - 3)^2 - 2

-- Define the expected rotated parabola equation
def rotated_parabola (x : ℝ) : ℝ := -2 * (x - 3)^2 - 2

-- Prove that the rotated parabola is correctly transformed
theorem parabola_rotation_180_equivalent :
  ∀ x, rotated_parabola x = -2 * (x - 3)^2 - 2 := 
by
  intro x
  unfold rotated_parabola
  sorry

end NUMINAMATH_GPT_parabola_rotation_180_equivalent_l2356_235657


namespace NUMINAMATH_GPT_inequality_problems_l2356_235665

theorem inequality_problems
  (m n l : ℝ)
  (h1 : m > n)
  (h2 : n > l) :
  (m + 1/m > n + 1/n) ∧ (m + 1/n > n + 1/m) :=
by
  sorry

end NUMINAMATH_GPT_inequality_problems_l2356_235665


namespace NUMINAMATH_GPT_min_value_of_inverse_sum_l2356_235663

theorem min_value_of_inverse_sum (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a + 3 * b = 1) : 
  ∃ c : ℝ, c = 4 + 2 * Real.sqrt 3 ∧ ∀x : ℝ, (x = (1 / a + 1 / b)) → x ≥ c :=
by
  sorry

end NUMINAMATH_GPT_min_value_of_inverse_sum_l2356_235663


namespace NUMINAMATH_GPT_seats_scientific_notation_l2356_235609

theorem seats_scientific_notation : 
  (13000 = 1.3 * 10^4) := 
by 
  sorry 

end NUMINAMATH_GPT_seats_scientific_notation_l2356_235609


namespace NUMINAMATH_GPT_factorial_equation_solution_l2356_235616

theorem factorial_equation_solution (a b c : ℕ) (h_pos_a : a > 0) (h_pos_b : b > 0) (h_pos_c : c > 0) :
  a.factorial * b.factorial = a.factorial + b.factorial + c.factorial → a = 3 ∧ b = 3 ∧ c = 4 := by
  sorry

end NUMINAMATH_GPT_factorial_equation_solution_l2356_235616


namespace NUMINAMATH_GPT_anne_distance_diff_l2356_235696

def track_length := 300
def min_distance := 100

-- Define distances functions as described
def distance_AB (t : ℝ) : ℝ := sorry  -- Distance function between Anne and Beth over time 
def distance_AC (t : ℝ) : ℝ := sorry  -- Distance function between Anne and Carmen over time 

theorem anne_distance_diff (Anne_speed Beth_speed Carmen_speed : ℝ) 
  (hneA : Anne_speed ≠ Beth_speed)
  (hneC : Anne_speed ≠ Carmen_speed) :
  ∃ α ≥ 0, min_distance ≤ distance_AB α ∧ min_distance ≤ distance_AC α :=
sorry

end NUMINAMATH_GPT_anne_distance_diff_l2356_235696


namespace NUMINAMATH_GPT_min_value_reciprocal_sum_l2356_235678

theorem min_value_reciprocal_sum (a b : ℝ) (ha : a > 0) (hb : b > 0) (hmean : (a + b) / 2 = 1 / 2) : 
  ∃ c, c = (1 / a + 1 / b) ∧ c ≥ 4 := 
sorry

end NUMINAMATH_GPT_min_value_reciprocal_sum_l2356_235678


namespace NUMINAMATH_GPT_Sheila_attendance_probability_l2356_235666

-- Definitions as per given conditions
def P_rain := 0.5
def P_sunny := 0.3
def P_cloudy := 0.2
def P_Sheila_goes_given_rain := 0.3
def P_Sheila_goes_given_sunny := 0.7
def P_Sheila_goes_given_cloudy := 0.5

-- Define the probability calculation
def P_Sheila_attends := 
  (P_rain * P_Sheila_goes_given_rain) + 
  (P_sunny * P_Sheila_goes_given_sunny) + 
  (P_cloudy * P_Sheila_goes_given_cloudy)

-- Final theorem statement
theorem Sheila_attendance_probability : P_Sheila_attends = 0.46 := by
  sorry

end NUMINAMATH_GPT_Sheila_attendance_probability_l2356_235666


namespace NUMINAMATH_GPT_correct_option_l2356_235688

-- Definitions based on the conditions of the problem
def exprA (a : ℝ) : Prop := 7 * a + a = 7 * a^2
def exprB (x y : ℝ) : Prop := 3 * x^2 * y - 2 * x^2 * y = x^2 * y
def exprC (y : ℝ) : Prop := 5 * y - 3 * y = 2
def exprD (a b : ℝ) : Prop := 3 * a + 2 * b = 5 * a * b

-- Proof problem statement verifying the correctness of the given expressions
theorem correct_option (x y : ℝ) : exprB x y :=
by
  -- (No proof is required, the statement is sufficient)
  sorry

end NUMINAMATH_GPT_correct_option_l2356_235688


namespace NUMINAMATH_GPT_find_BP_l2356_235679

-- Define points
variables {A B C D P : Type}  

-- Define lengths
variables (AP PC BP DP BD : ℝ)

-- Provided conditions
axiom h1 : AP = 10
axiom h2 : PC = 2
axiom h3 : BD = 9

-- Assume intersect and lengths relations setup
axiom intersect : BP < DP
axiom power_of_point : AP * PC = BP * DP

-- Target statement
theorem find_BP (h1 : AP = 10) (h2 : PC = 2) (h3 : BD = 9)
  (intersect : BP < DP) (power_of_point : AP * PC = BP * DP) : BP = 4 :=
  sorry

end NUMINAMATH_GPT_find_BP_l2356_235679


namespace NUMINAMATH_GPT_rectangle_diagonal_length_proof_parallel_l2356_235640

-- Definition of a rectangle whose sides are parallel to the coordinate axes
structure RectangleParallel :=
  (a b : ℕ)
  (area_eq : a * b = 2018)
  (diagonal_length : ℕ)

-- Prove that the length of the diagonal of the given rectangle is sqrt(1018085)
def rectangle_diagonal_length_parallel : RectangleParallel → Prop :=
  fun r => r.diagonal_length = Int.sqrt (r.a * r.a + r.b * r.b)

theorem rectangle_diagonal_length_proof_parallel (r : RectangleParallel)
  (h1 : r.a * r.b = 2018)
  (h2 : r.a ≠ r.b)
  (h3 : r.diagonal_length = Int.sqrt (r.a * r.a + r.b * r.b)) :
  r.diagonal_length = Int.sqrt 1018085 := 
  sorry

end NUMINAMATH_GPT_rectangle_diagonal_length_proof_parallel_l2356_235640


namespace NUMINAMATH_GPT_g_domain_l2356_235614

noncomputable def g (x : ℝ) : ℝ := Real.tan (Real.arccos (x^3))

theorem g_domain : { x : ℝ | -1 ≤ x ∧ x ≤ 1 ∧ x ≠ 0 } = (Set.Icc (-1) 0 ∪ Set.Icc 0 1) \ {0} :=
by
  sorry

end NUMINAMATH_GPT_g_domain_l2356_235614


namespace NUMINAMATH_GPT_triangle_inequality_for_min_segments_l2356_235618

theorem triangle_inequality_for_min_segments
  (a b c d : ℝ)
  (a1 b1 c1 : ℝ)
  (h1 : a1 = min a d)
  (h2 : b1 = min b d)
  (h3 : c1 = min c d)
  (h_triangle : c < a + b) :
  a1 + b1 > c1 ∧ a1 + c1 > b1 ∧ b1 + c1 > a1 := sorry

end NUMINAMATH_GPT_triangle_inequality_for_min_segments_l2356_235618


namespace NUMINAMATH_GPT_find_range_of_k_l2356_235671

noncomputable def f (x k : ℝ) : ℝ := |x^2 - 1| + x^2 + k * x

theorem find_range_of_k :
  (∀ x : ℝ, 0 < x → 0 ≤ f x k) → (-1 ≤ k) :=
by
  sorry

end NUMINAMATH_GPT_find_range_of_k_l2356_235671


namespace NUMINAMATH_GPT_maximize_takehome_pay_l2356_235661

noncomputable def tax_initial (income : ℝ) : ℝ :=
  if income ≤ 20000 then 0.10 * income else 2000 + 0.05 * ((income - 20000) / 10000) * income

noncomputable def tax_beyond (income : ℝ) : ℝ :=
  (income - 20000) * ((0.005 * ((income - 20000) / 10000)) * income)

noncomputable def tax_total (income : ℝ) : ℝ :=
  if income ≤ 20000 then tax_initial income else tax_initial 20000 + tax_beyond income

noncomputable def takehome_pay_function (income : ℝ) : ℝ :=
  income - tax_total income

theorem maximize_takehome_pay : ∃ x, takehome_pay_function x = takehome_pay_function 30000 := 
sorry

end NUMINAMATH_GPT_maximize_takehome_pay_l2356_235661


namespace NUMINAMATH_GPT_range_of_a_l2356_235662

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a / x + x^2
def g (x : ℝ) : ℝ := x^3 - x^2 - 3

theorem range_of_a (a : ℝ) (h : ∀ s t : ℝ, (1/2 ≤ s ∧ s ≤ 2) → (1/2 ≤ t ∧ t ≤ 2) → f a s ≥ g t) : a ≥ 1 :=
sorry

end NUMINAMATH_GPT_range_of_a_l2356_235662


namespace NUMINAMATH_GPT_calculate_liquids_l2356_235641

def water_ratio := 60 -- mL of water for every 400 mL of flour
def milk_ratio := 80 -- mL of milk for every 400 mL of flour
def flour_ratio := 400 -- mL of flour in one portion

def flour_quantity := 1200 -- mL of flour available

def number_of_portions := flour_quantity / flour_ratio

def total_water := number_of_portions * water_ratio
def total_milk := number_of_portions * milk_ratio

theorem calculate_liquids :
  total_water = 180 ∧ total_milk = 240 :=
by
  -- Proof will be filled in here. Skipping with sorry for now.
  sorry

end NUMINAMATH_GPT_calculate_liquids_l2356_235641


namespace NUMINAMATH_GPT_jasmine_milk_gallons_l2356_235629

theorem jasmine_milk_gallons (G : ℝ) 
  (coffee_cost_per_pound : ℝ) (milk_cost_per_gallon : ℝ) (total_cost : ℝ)
  (coffee_pounds : ℝ) :
  coffee_cost_per_pound = 2.50 →
  milk_cost_per_gallon = 3.50 →
  total_cost = 17 →
  coffee_pounds = 4 →
  total_cost - coffee_pounds * coffee_cost_per_pound = G * milk_cost_per_gallon →
  G = 2 :=
by
  intros
  sorry

end NUMINAMATH_GPT_jasmine_milk_gallons_l2356_235629


namespace NUMINAMATH_GPT_pallets_of_paper_cups_l2356_235680

theorem pallets_of_paper_cups (total_pallets paper_towels tissues paper_plates : ℕ) 
  (H1 : total_pallets = 20) 
  (H2 : paper_towels = total_pallets / 2)
  (H3 : tissues = total_pallets / 4)
  (H4 : paper_plates = total_pallets / 5) : 
  total_pallets - paper_towels - tissues - paper_plates = 1 := 
  by
    sorry

end NUMINAMATH_GPT_pallets_of_paper_cups_l2356_235680


namespace NUMINAMATH_GPT_inequality_solution_set_minimum_value_mn_squared_l2356_235623

noncomputable def f (x : ℝ) := |x - 2| + |x + 1|

theorem inequality_solution_set : 
  (∀ x, f x > 7 ↔ x > 4 ∨ x < -3) :=
by sorry

theorem minimum_value_mn_squared (m n : ℝ) (hm : n > 0) (hmin : ∀ x, f x ≥ m + n) :
  m^2 + n^2 = 9 / 2 ∧ m = 3 / 2 ∧ n = 3 / 2 :=
by sorry

end NUMINAMATH_GPT_inequality_solution_set_minimum_value_mn_squared_l2356_235623


namespace NUMINAMATH_GPT_percentage_increase_l2356_235673

theorem percentage_increase (G P : ℝ) (h1 : G = 15 + (P / 100) * 15) 
                            (h2 : 15 + 2 * G = 51) : P = 20 :=
by 
  sorry

end NUMINAMATH_GPT_percentage_increase_l2356_235673


namespace NUMINAMATH_GPT_intersect_once_l2356_235613

theorem intersect_once (x : ℝ) : 
  (∀ y, y = 3 * Real.log x ↔ y = Real.log (3 * x)) → (∃! x, 3 * Real.log x = Real.log (3 * x)) :=
by 
  sorry

end NUMINAMATH_GPT_intersect_once_l2356_235613


namespace NUMINAMATH_GPT_find_factor_l2356_235647

theorem find_factor (x : ℕ) (f : ℕ) (h1 : x = 9)
  (h2 : (2 * x + 6) * f = 72) : f = 3 := by
  sorry

end NUMINAMATH_GPT_find_factor_l2356_235647


namespace NUMINAMATH_GPT_range_of_m_l2356_235604

theorem range_of_m (m : ℝ) (H : ∀ x, x ≥ 4 → (m^2 * x - 1) / (m * x + 1) < 0) : m < -1 / 2 :=
sorry

end NUMINAMATH_GPT_range_of_m_l2356_235604


namespace NUMINAMATH_GPT_eccentricity_of_ellipse_l2356_235625

open Real

def ellipse_eq (a b x y : ℝ) : Prop :=
  a > b ∧ b > 0 ∧ (x^2 / a^2 + y^2 / b^2 = 1)

def foci_dist_eq (a c : ℝ) : Prop :=
  2 * c / (2 * a) = sqrt 6 / 2

noncomputable def eccentricity (c a : ℝ) : ℝ :=
  c / a

theorem eccentricity_of_ellipse (a b x y c : ℝ)
  (h1 : ellipse_eq a b x y)
  (h2 : foci_dist_eq a c) :
  eccentricity c a = sqrt 6 / 3 :=
sorry

end NUMINAMATH_GPT_eccentricity_of_ellipse_l2356_235625


namespace NUMINAMATH_GPT_cost_price_of_toy_l2356_235695

theorem cost_price_of_toy 
  (cost_price : ℝ)
  (SP : ℝ := 120000)
  (num_toys : ℕ := 40)
  (profit_per_toy : ℝ := 500)
  (gain_per_toy : ℝ := cost_price + profit_per_toy)
  (total_gain : ℝ := 8 * cost_price + profit_per_toy * num_toys)
  (total_cost_price : ℝ := num_toys * cost_price)
  (SP_eq_cost_plus_gain : SP = total_cost_price + total_gain) :
  cost_price = 2083.33 :=
by
  sorry

end NUMINAMATH_GPT_cost_price_of_toy_l2356_235695


namespace NUMINAMATH_GPT_find_m_l2356_235683

-- Given conditions
variable (U : Set ℕ) (A : Set ℕ) (m : ℕ)
variable (hU : U = {1, 2, 3, 4})
variable (hA : A = { x ∈ U | x^2 - 5 * x + m = 0 })
variable (hCUA : U \ A = {1, 4})

-- Prove that m = 6
theorem find_m (U A : Set ℕ) (m : ℕ) 
               (hU : U = {1, 2, 3, 4}) 
               (hA : A = { x ∈ U | x^2 - 5 * x + m = 0 }) 
               (hCUA : U \ A = {1, 4}) : 
  m = 6 := 
sorry

end NUMINAMATH_GPT_find_m_l2356_235683


namespace NUMINAMATH_GPT_correct_propositions_are_123_l2356_235633

theorem correct_propositions_are_123
  (f : ℝ → ℝ)
  (h1 : ∀ x, f (x-1) = -f x → f x = f (x-2))
  (h2 : ∀ x, f (1 - x) = f (x - 1) → f (1 - x) = -f x)
  (h3 : ∀ x, f (x) = -f (-x)) :
  (∀ x, f (x-1) = -f x → ∃ c, c * (f (1-1)) = -f x) ∧
  (∀ x, f (1 - x) = f (x - 1) → ∀ x, f x = f (-x)) ∧
  (∀ x, f (x-1) = -f x → ∀ x, f (x - 2) = f x) :=
sorry

end NUMINAMATH_GPT_correct_propositions_are_123_l2356_235633


namespace NUMINAMATH_GPT_factorize_expression_l2356_235642

theorem factorize_expression (a b : ℝ) : b^2 - ab + a - b = (b - 1) * (b - a) :=
by
  sorry

end NUMINAMATH_GPT_factorize_expression_l2356_235642


namespace NUMINAMATH_GPT_common_ratio_of_geometric_sequence_l2356_235687

theorem common_ratio_of_geometric_sequence (a : ℕ → ℝ) (q : ℝ)
  (h1 : a 3 - 3 * a 2 = 2)
  (h2 : 5 * a 4 = (12 * a 3 + 2 * a 5) / 2) :
  (∃ a1 : ℝ, ∃ q : ℝ,
    (∀ n, a n = a1 * q ^ (n - 1)) ∧ 
    q = 2) := 
by 
  sorry

end NUMINAMATH_GPT_common_ratio_of_geometric_sequence_l2356_235687


namespace NUMINAMATH_GPT_not_perfect_square_2023_l2356_235606

theorem not_perfect_square_2023 : ¬ (∃ x : ℤ, x^2 = 5^2023) := 
sorry

end NUMINAMATH_GPT_not_perfect_square_2023_l2356_235606


namespace NUMINAMATH_GPT_time_after_seconds_l2356_235621

def initial_time : Nat × Nat × Nat := (4, 45, 0)
def seconds_to_add : Nat := 12345
def final_time : Nat × Nat × Nat := (8, 30, 45)

theorem time_after_seconds (h : initial_time = (4, 45, 0) ∧ seconds_to_add = 12345) : 
  ∃ (h' : Nat × Nat × Nat), h' = final_time := by
  sorry

end NUMINAMATH_GPT_time_after_seconds_l2356_235621


namespace NUMINAMATH_GPT_new_price_of_sugar_l2356_235612

theorem new_price_of_sugar (C : ℝ) (H : 10 * C = P * (0.7692307692307693 * C)) : P = 13 := by
  sorry

end NUMINAMATH_GPT_new_price_of_sugar_l2356_235612


namespace NUMINAMATH_GPT_vampire_needs_7_gallons_per_week_l2356_235626

-- Define conditions given in the problem
def pints_per_person : ℕ := 2
def people_per_day : ℕ := 4
def days_per_week : ℕ := 7
def pints_per_gallon : ℕ := 8

-- Prove the vampire needs 7 gallons of blood per week to survive
theorem vampire_needs_7_gallons_per_week :
  (pints_per_person * people_per_day * days_per_week) / pints_per_gallon = 7 := 
by 
  sorry

end NUMINAMATH_GPT_vampire_needs_7_gallons_per_week_l2356_235626


namespace NUMINAMATH_GPT_rows_of_seats_l2356_235639

theorem rows_of_seats (r : ℕ) (h : r * 4 = 80) : r = 20 :=
sorry

end NUMINAMATH_GPT_rows_of_seats_l2356_235639


namespace NUMINAMATH_GPT_sqrt8_same_type_as_sqrt2_l2356_235636

def same_type_sqrt_2 (x : Real) : Prop := ∃ k : Real, k * Real.sqrt 2 = x

theorem sqrt8_same_type_as_sqrt2 : same_type_sqrt_2 (Real.sqrt 8) :=
  sorry

end NUMINAMATH_GPT_sqrt8_same_type_as_sqrt2_l2356_235636


namespace NUMINAMATH_GPT_no_arith_prog_of_sines_l2356_235664

theorem no_arith_prog_of_sines (x₁ x₂ x₃ : ℝ) (h₁ : x₁ ≠ x₂) (h₂ : x₂ ≠ x₃) (h₃ : x₁ ≠ x₃)
    (hx : 0 < x₁ ∧ x₁ < (Real.pi / 2))
    (hy : 0 < x₂ ∧ x₂ < (Real.pi / 2))
    (hz : 0 < x₃ ∧ x₃ < (Real.pi / 2))
    (h : 2 * Real.sin x₂ = Real.sin x₁ + Real.sin x₃) :
    ¬ (x₁ + x₃ = 2 * x₂) :=
sorry

end NUMINAMATH_GPT_no_arith_prog_of_sines_l2356_235664


namespace NUMINAMATH_GPT_simplify_expr1_simplify_expr2_l2356_235638

-- Define the variables a and b
variables (a b : ℝ)

-- First problem: simplify 2a^2 - 3a^3 + 5a + 2a^3 - a^2 to a^2 - a^3 + 5a
theorem simplify_expr1 : 2*a^2 - 3*a^3 + 5*a + 2*a^3 - a^2 = a^2 - a^3 + 5*a :=
  by sorry

-- Second problem: simplify (2 / 3) (2 * a - b) + 2 (b - 2 * a) - 3 (2 * a - b) - (4 / 3) (b - 2 * a) to -6 * a + 3 * b
theorem simplify_expr2 : 
  (2 / 3) * (2 * a - b) + 2 * (b - 2 * a) - 3 * (2 * a - b) - (4 / 3) * (b - 2 * a) = -6 * a + 3 * b :=
  by sorry

end NUMINAMATH_GPT_simplify_expr1_simplify_expr2_l2356_235638


namespace NUMINAMATH_GPT_arithmetic_sequence_sum_l2356_235608

theorem arithmetic_sequence_sum {a : ℕ → ℝ}
  (h1 : a 1 + a 5 = 6) 
  (h2 : a 2 + a 14 = 26) :
  (10 / 2) * (a 1 + a 10) = 80 :=
by sorry

end NUMINAMATH_GPT_arithmetic_sequence_sum_l2356_235608


namespace NUMINAMATH_GPT_ellipse_intersection_l2356_235668

open Real

noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

theorem ellipse_intersection (f1 f2 : ℝ × ℝ)
    (h1 : f1 = (0, 5))
    (h2 : f2 = (4, 0))
    (origin_intersection : distance (0, 0) f1 + distance (0, 0) f2 = 5) :
    ∃ x : ℝ, (distance (x, 0) f1 + distance (x, 0) f2 = 5 ∧ x > 0 ∧ x ≠ 0 → x = 28 / 9) :=
by 
  sorry

end NUMINAMATH_GPT_ellipse_intersection_l2356_235668


namespace NUMINAMATH_GPT_professionals_work_days_l2356_235605

theorem professionals_work_days (cost_per_hour_1 cost_per_hour_2 hours_per_day total_cost : ℝ) (h_cost1: cost_per_hour_1 = 15) (h_cost2: cost_per_hour_2 = 15) (h_hours: hours_per_day = 6) (h_total: total_cost = 1260) : (∃ d : ℝ, total_cost = d * hours_per_day * (cost_per_hour_1 + cost_per_hour_2) ∧ d = 7) :=
by
  use 7
  rw [h_cost1, h_cost2, h_hours, h_total]
  simp
  sorry

end NUMINAMATH_GPT_professionals_work_days_l2356_235605


namespace NUMINAMATH_GPT_num_dimes_l2356_235615

/--
Given eleven coins consisting of pennies, nickels, dimes, quarters, and half-dollars,
having a total value of $1.43, with at least one coin of each type,
prove that there must be exactly 4 dimes.
-/
theorem num_dimes (p n d q h : ℕ) :
  1 ≤ p ∧ 1 ≤ n ∧ 1 ≤ d ∧ 1 ≤ q ∧ 1 ≤ h ∧ 
  p + n + d + q + h = 11 ∧ 
  (1 * p + 5 * n + 10 * d + 25 * q + 50 * h) = 143
  → d = 4 :=
by
  sorry

end NUMINAMATH_GPT_num_dimes_l2356_235615


namespace NUMINAMATH_GPT_sufficient_condition_abs_sum_gt_one_l2356_235656

theorem sufficient_condition_abs_sum_gt_one (x y : ℝ) (h : y ≤ -2) : |x| + |y| > 1 :=
  sorry

end NUMINAMATH_GPT_sufficient_condition_abs_sum_gt_one_l2356_235656


namespace NUMINAMATH_GPT_percentage_failed_both_l2356_235653

theorem percentage_failed_both 
    (p_h p_e p_p p_pe : ℝ)
    (h_p_h : p_h = 32)
    (h_p_e : p_e = 56)
    (h_p_p : p_p = 24)
    : p_pe = 12 := by 
    sorry

end NUMINAMATH_GPT_percentage_failed_both_l2356_235653


namespace NUMINAMATH_GPT_find_angle_B_l2356_235637

theorem find_angle_B (A B C : ℝ) (a b c : ℝ)
  (hAngleA : A = 120) (ha : a = 2) (hb : b = 2 * Real.sqrt 3 / 3) : B = 30 :=
sorry

end NUMINAMATH_GPT_find_angle_B_l2356_235637


namespace NUMINAMATH_GPT_repeating_decimals_subtraction_l2356_235690

def x : Rat := 1 / 3
def y : Rat := 2 / 99

theorem repeating_decimals_subtraction :
  x - y = 31 / 99 :=
sorry

end NUMINAMATH_GPT_repeating_decimals_subtraction_l2356_235690


namespace NUMINAMATH_GPT_last_rope_length_l2356_235682

def totalRopeLength : ℝ := 35
def rope1 : ℝ := 8
def rope2 : ℝ := 20
def rope3a : ℝ := 2
def rope3b : ℝ := 2
def rope3c : ℝ := 2
def knotLoss : ℝ := 1.2
def numKnots : ℝ := 4

theorem last_rope_length : 
  (35 + (4 * 1.2)) = (8 + 20 + 2 + 2 + 2 + x) → (x = 5.8) :=
sorry

end NUMINAMATH_GPT_last_rope_length_l2356_235682


namespace NUMINAMATH_GPT_weight_of_six_moles_BaF2_l2356_235667

variable (atomic_weight_Ba : ℝ := 137.33) -- Atomic weight of Barium in g/mol
variable (atomic_weight_F : ℝ := 19.00) -- Atomic weight of Fluorine in g/mol
variable (moles_BaF2 : ℝ := 6) -- Number of moles of BaF2

theorem weight_of_six_moles_BaF2 :
  moles_BaF2 * (atomic_weight_Ba + 2 * atomic_weight_F) = 1051.98 :=
by sorry

end NUMINAMATH_GPT_weight_of_six_moles_BaF2_l2356_235667


namespace NUMINAMATH_GPT_solve_abs_inequality_l2356_235694

theorem solve_abs_inequality (x : ℝ) (h : x ≠ 1) : 
  abs ((3 * x - 2) / (x - 1)) > 3 ↔ (5 / 6 < x ∧ x < 1) ∨ (x > 1) := 
by 
  sorry

end NUMINAMATH_GPT_solve_abs_inequality_l2356_235694


namespace NUMINAMATH_GPT_computation_problems_count_l2356_235681

theorem computation_problems_count
    (C W : ℕ)
    (h1 : 3 * C + 5 * W = 110)
    (h2 : C + W = 30) :
    C = 20 :=
by
  sorry

end NUMINAMATH_GPT_computation_problems_count_l2356_235681


namespace NUMINAMATH_GPT_marian_returned_amount_l2356_235646

theorem marian_returned_amount
  (B : ℕ) (G : ℕ) (H : ℕ) (N : ℕ)
  (hB : B = 126) (hG : G = 60) (hH : H = G / 2) (hN : N = 171) :
  (B + G + H - N) = 45 := 
by
  sorry

end NUMINAMATH_GPT_marian_returned_amount_l2356_235646


namespace NUMINAMATH_GPT_number_of_people_entered_l2356_235632

-- Define the total number of placards
def total_placards : ℕ := 5682

-- Define the number of placards each person takes
def placards_per_person : ℕ := 2

-- The Lean theorem to prove the number of people who entered the stadium
theorem number_of_people_entered : total_placards / placards_per_person = 2841 :=
by
  -- Proof will be inserted here
  sorry

end NUMINAMATH_GPT_number_of_people_entered_l2356_235632


namespace NUMINAMATH_GPT_shaded_area_eight_l2356_235648

-- Definitions based on given conditions
def arcAQB (r : ℝ) : Prop := r = 2
def arcBRC (r : ℝ) : Prop := r = 2
def midpointQ (r : ℝ) : Prop := arcAQB r
def midpointR (r : ℝ) : Prop := arcBRC r
def midpointS (r : ℝ) : Prop := arcAQB r ∧ arcBRC r ∧ (arcAQB r ∨ arcBRC r)
def arcQRS (r : ℝ) : Prop := r = 2 ∧ midpointS r

-- The theorem to prove
theorem shaded_area_eight (r : ℝ) : arcAQB r ∧ arcBRC r ∧ arcQRS r → area_shaded_region = 8 := by
  sorry

end NUMINAMATH_GPT_shaded_area_eight_l2356_235648


namespace NUMINAMATH_GPT_cube_edge_length_proof_l2356_235675

-- Define the edge length of the cube
def edge_length_of_cube := 15

-- Define the volume of the cube
def volume_of_cube (a : ℕ) := a^3

-- Define the volume of the displaced water
def volume_of_displaced_water := 20 * 15 * 11.25

-- The theorem to prove
theorem cube_edge_length_proof : ∃ a : ℕ, volume_of_cube a = 3375 ∧ a = edge_length_of_cube := 
by {
  sorry
}

end NUMINAMATH_GPT_cube_edge_length_proof_l2356_235675


namespace NUMINAMATH_GPT_maximum_area_of_rectangle_with_given_perimeter_l2356_235611

noncomputable def perimeter : ℝ := 30
noncomputable def area (length width : ℝ) : ℝ := length * width
noncomputable def max_area : ℝ := 56.25

theorem maximum_area_of_rectangle_with_given_perimeter :
  ∃ length width : ℝ, 2 * length + 2 * width = perimeter ∧ area length width = max_area :=
sorry

end NUMINAMATH_GPT_maximum_area_of_rectangle_with_given_perimeter_l2356_235611


namespace NUMINAMATH_GPT_complement_union_covers_until_1_l2356_235634

open Set

noncomputable def S := {x : ℝ | x > -2}
noncomputable def T := {x : ℝ | x^2 + 3*x - 4 ≤ 0}
noncomputable def complement_R_S := {x : ℝ | x ≤ -2}
noncomputable def union := complement_R_S ∪ T

theorem complement_union_covers_until_1 : union = {x : ℝ | x ≤ 1} := by
  sorry

end NUMINAMATH_GPT_complement_union_covers_until_1_l2356_235634


namespace NUMINAMATH_GPT_find_tricksters_l2356_235644

def inhab_group : Type := { n : ℕ // n < 65 }
def is_knight (i : inhab_group) : Prop := ∀ g : inhab_group, true -- Placeholder for the actual property

theorem find_tricksters (inhabitants : inhab_group → Prop)
  (is_knight : inhab_group → Prop)
  (knight_always_tells_truth : ∀ i : inhab_group, is_knight i → inhabitants i = true)
  (tricksters_2_and_rest_knights : ∃ t1 t2 : inhab_group, t1 ≠ t2 ∧ ¬is_knight t1 ∧ ¬is_knight t2 ∧
    (∀ i : inhab_group, i ≠ t1 → i ≠ t2 → is_knight i)) :
  ∃ find_them : inhab_group → inhab_group → Prop, (∀ q_count : ℕ, q_count ≤ 16) → 
  ∃ t1 t2 : inhab_group, find_them t1 t2 :=
by 
  -- The proof goes here
  sorry

end NUMINAMATH_GPT_find_tricksters_l2356_235644


namespace NUMINAMATH_GPT_new_person_age_l2356_235669

theorem new_person_age (T : ℕ) (A : ℕ) (n : ℕ) 
  (avg_age : ℕ) (new_avg_age : ℕ) 
  (h1 : avg_age = T / n) 
  (h2 : T = 14 * n)
  (h3 : n = 17) 
  (h4 : new_avg_age = 15) 
  (h5 : new_avg_age = (T + A) / (n + 1)) 
  : A = 32 := 
by 
  sorry

end NUMINAMATH_GPT_new_person_age_l2356_235669


namespace NUMINAMATH_GPT_inequality_system_correctness_l2356_235651

theorem inequality_system_correctness :
  (∀ (x a b : ℝ), 
    (x - a ≥ 1) ∧ (x - b < 2) →
    ((∀ x, -1 ≤ x ∧ x < 3 → (a = -2 ∧ b = 1)) ∧
     (a = b → (a + 1 ≤ x ∧ x < a + 2)) ∧
     (¬(∃ x, a + 1 ≤ x ∧ x < b + 2) → a > b + 1) ∧
     ((∃ n : ℤ, n < 0 ∧ n ≥ -6 - a ∧ n ≥ -5) → -7 < a ∧ a ≤ -6))) :=
sorry

end NUMINAMATH_GPT_inequality_system_correctness_l2356_235651


namespace NUMINAMATH_GPT_inequality_transitive_l2356_235649

theorem inequality_transitive (a b c : ℝ) : a * c^2 > b * c^2 → a > b :=
sorry

end NUMINAMATH_GPT_inequality_transitive_l2356_235649


namespace NUMINAMATH_GPT_quadratic_solution_l2356_235686

theorem quadratic_solution :
  (∀ x : ℝ, 3 * x^2 - 13 * x + 5 = 0 → 
           x = (13 + Real.sqrt 109) / 6 ∨ x = (13 - Real.sqrt 109) / 6) 
  := by
  sorry

end NUMINAMATH_GPT_quadratic_solution_l2356_235686


namespace NUMINAMATH_GPT_vector_condition_l2356_235622

def vec_a : ℝ × ℝ := (5, 2)
def vec_b : ℝ × ℝ := (-4, -3)
def vec_c : ℝ × ℝ := (-23, -12)

theorem vector_condition : 3 • (vec_a.1, vec_a.2) - 2 • (vec_b.1, vec_b.2) + vec_c = (0, 0) :=
by
  sorry

end NUMINAMATH_GPT_vector_condition_l2356_235622


namespace NUMINAMATH_GPT_quadratic_factorization_l2356_235698

theorem quadratic_factorization (a b : ℕ) (h1 : x^2 - 20 * x + 96 = (x - a) * (x - b)) (h2 : a > b) : 2 * b - a = 4 :=
sorry

end NUMINAMATH_GPT_quadratic_factorization_l2356_235698


namespace NUMINAMATH_GPT_white_mice_count_l2356_235619

variable (T W B : ℕ) -- Declare variables T (total), W (white), B (brown)

def W_condition := W = (2 / 3) * T  -- White mice condition
def B_condition := B = 7           -- Brown mice condition
def T_condition := T = W + B       -- Total mice condition

theorem white_mice_count : W = 14 :=
by
  sorry  -- Proof to be filled in

end NUMINAMATH_GPT_white_mice_count_l2356_235619


namespace NUMINAMATH_GPT_janice_work_days_l2356_235697

variable (dailyEarnings : Nat)
variable (overtimeEarnings : Nat)
variable (numOvertimeShifts : Nat)
variable (totalEarnings : Nat)

theorem janice_work_days
    (h1 : dailyEarnings = 30)
    (h2 : overtimeEarnings = 15)
    (h3 : numOvertimeShifts = 3)
    (h4 : totalEarnings = 195)
    : let overtimeTotal := numOvertimeShifts * overtimeEarnings
      let regularEarnings := totalEarnings - overtimeTotal
      let workDays := regularEarnings / dailyEarnings
      workDays = 5 :=
by
  sorry

end NUMINAMATH_GPT_janice_work_days_l2356_235697


namespace NUMINAMATH_GPT_decreasing_linear_function_l2356_235631

theorem decreasing_linear_function (k : ℝ) : 
  (∀ x1 x2 : ℝ, x1 < x2 → (k - 3) * x1 + 2 > (k - 3) * x2 + 2) → k < 3 := 
by 
  sorry

end NUMINAMATH_GPT_decreasing_linear_function_l2356_235631


namespace NUMINAMATH_GPT_solve_for_x_l2356_235699

-- We define that the condition and what we need to prove.
theorem solve_for_x (x : ℝ) : (x + 7) / (x - 4) = (x - 3) / (x + 6) → x = -3 / 2 :=
by sorry

end NUMINAMATH_GPT_solve_for_x_l2356_235699


namespace NUMINAMATH_GPT_monotonic_increasing_interval_l2356_235603

noncomputable def f (x : ℝ) : ℝ := (1 / 2)^(x^2 - 2 * x + 6)

theorem monotonic_increasing_interval : 
  ∀ x y : ℝ, x < y → y < 1 → f x < f y :=
by
  sorry

end NUMINAMATH_GPT_monotonic_increasing_interval_l2356_235603


namespace NUMINAMATH_GPT_point_coordinates_in_second_quadrant_l2356_235617

theorem point_coordinates_in_second_quadrant
    (P : ℝ × ℝ)
    (h1 : P.1 < 0)
    (h2 : P.2 > 0)
    (h3 : |P.2| = 4)
    (h4 : |P.1| = 5) :
    P = (-5, 4) :=
sorry

end NUMINAMATH_GPT_point_coordinates_in_second_quadrant_l2356_235617


namespace NUMINAMATH_GPT_robin_bobin_can_meet_prescription_l2356_235660

def large_gr_pill : ℝ := 11
def medium_gr_pill : ℝ := -1.1
def small_gr_pill : ℝ := -0.11
def prescribed_gr : ℝ := 20.13

theorem robin_bobin_can_meet_prescription :
  ∃ (large : ℕ) (medium : ℕ) (small : ℕ), large ≥ 1 ∧ medium ≥ 1 ∧ small ≥ 1 ∧
  large_gr_pill * large + medium_gr_pill * medium + small_gr_pill * small = prescribed_gr :=
sorry

end NUMINAMATH_GPT_robin_bobin_can_meet_prescription_l2356_235660


namespace NUMINAMATH_GPT_g100_value_l2356_235684

-- Define the function g and its properties
def g (x : ℝ) : ℝ := sorry

theorem g100_value 
  (h : ∀ (x y : ℝ), 0 < x → 0 < y → x * g y - y * g x = g (x / y) + x - y) : 
  g 100 = 99 / 2 := 
sorry

end NUMINAMATH_GPT_g100_value_l2356_235684


namespace NUMINAMATH_GPT_harvest_rate_l2356_235691

def days := 3
def total_sacks := 24
def sacks_per_day := total_sacks / days

theorem harvest_rate :
  sacks_per_day = 8 :=
by
  sorry

end NUMINAMATH_GPT_harvest_rate_l2356_235691


namespace NUMINAMATH_GPT_area_ratio_proof_l2356_235677

variables (BE CE DE AE : ℝ)
variables (S_alpha S_beta S_gamma S_delta : ℝ)
variables (x : ℝ)

-- Definitions for the given conditions
def BE_val := 80
def CE_val := 60
def DE_val := 40
def AE_val := 30

-- Expressing the ratios
def S_alpha_ratio := 2
def S_beta_ratio := 2

-- Assuming areas in terms of x
def S_alpha_val := 2 * x
def S_beta_val := 2 * x
def S_delta_val := x
def S_gamma_val := 2 * x

-- Problem statement
theorem area_ratio_proof
  (BE := BE_val)
  (CE := CE_val)
  (DE := DE_val)
  (AE := AE_val)
  (S_alpha := S_alpha_val)
  (S_beta := S_beta_val)
  (S_gamma := S_gamma_val)
  (S_delta := S_delta_val) :
  (S_gamma + S_delta) / (S_alpha + S_beta) = 5 / 4 :=
by
  sorry

end NUMINAMATH_GPT_area_ratio_proof_l2356_235677


namespace NUMINAMATH_GPT_no_rational_multiples_pi_tan_sum_two_l2356_235607

theorem no_rational_multiples_pi_tan_sum_two (x y : ℚ) (hx : 0 < x * π ∧ x * π < y * π ∧ y * π < π / 2) (hxy : Real.tan (x * π) + Real.tan (y * π) = 2) : False :=
sorry

end NUMINAMATH_GPT_no_rational_multiples_pi_tan_sum_two_l2356_235607


namespace NUMINAMATH_GPT_cylinder_radius_original_l2356_235654

theorem cylinder_radius_original (r : ℝ) (h : ℝ) (h_given : h = 4) 
    (V_increase_radius : π * (r + 4) ^ 2 * h = π * r ^ 2 * (h + 4)) : 
    r = 12 := 
  by
    sorry

end NUMINAMATH_GPT_cylinder_radius_original_l2356_235654


namespace NUMINAMATH_GPT_purchasing_methods_l2356_235672

theorem purchasing_methods :
  ∃ (s : Finset (ℕ × ℕ)), s.card = 7 ∧
    ∀ (x y : ℕ), (x, y) ∈ s ↔ 60 * x + 70 * y ≤ 500 ∧ 3 ≤ x ∧ 2 ≤ y :=
sorry

end NUMINAMATH_GPT_purchasing_methods_l2356_235672


namespace NUMINAMATH_GPT_valid_starting_lineups_correct_l2356_235650

-- Define the parameters from the problem
def volleyball_team : Finset ℕ := Finset.range 18
def quadruplets : Finset ℕ := {0, 1, 2, 3}

-- Define the main computation: total lineups excluding those where all quadruplets are chosen
noncomputable def valid_starting_lineups : ℕ :=
  (volleyball_team.card.choose 7) - ((volleyball_team \ quadruplets).card.choose 3)

-- The theorem states that the number of valid starting lineups is 31460
theorem valid_starting_lineups_correct : valid_starting_lineups = 31460 := by
  sorry

end NUMINAMATH_GPT_valid_starting_lineups_correct_l2356_235650


namespace NUMINAMATH_GPT_comic_story_books_proportion_l2356_235630

theorem comic_story_books_proportion (x : ℕ) :
  let initial_comic_books := 140
  let initial_story_books := 100
  let borrowed_books_per_day := 4
  let comic_books_after_x_days := initial_comic_books - borrowed_books_per_day * x
  let story_books_after_x_days := initial_story_books - borrowed_books_per_day * x
  (comic_books_after_x_days = 3 * story_books_after_x_days) -> x = 20 :=
by
  sorry

end NUMINAMATH_GPT_comic_story_books_proportion_l2356_235630


namespace NUMINAMATH_GPT_chris_money_left_over_l2356_235645

-- Define the constants based on the conditions given in the problem.
def video_game_cost : ℕ := 60
def candy_cost : ℕ := 5
def earnings_per_hour : ℕ := 8
def hours_worked : ℕ := 9

-- Define the intermediary results based on the problem's conditions.
def total_cost : ℕ := video_game_cost + candy_cost
def total_earnings : ℕ := earnings_per_hour * hours_worked

-- Define the final result to be proven.
def total_leftover : ℕ := total_earnings - total_cost

-- State the proof problem as a Lean theorem.
theorem chris_money_left_over : total_leftover = 7 := by
  sorry

end NUMINAMATH_GPT_chris_money_left_over_l2356_235645


namespace NUMINAMATH_GPT_final_concentration_after_procedure_l2356_235628

open Real

def initial_salt_concentration : ℝ := 0.16
def final_salt_concentration : ℝ := 0.107

def volume_ratio_large : ℝ := 10
def volume_ratio_medium : ℝ := 4
def volume_ratio_small : ℝ := 3

def overflow_due_to_small_ball : ℝ := 0.1

theorem final_concentration_after_procedure :
  (initial_salt_concentration * (overflow_due_to_small_ball)) * volume_ratio_small / (volume_ratio_large + volume_ratio_medium + volume_ratio_small) =
  final_salt_concentration :=
sorry

end NUMINAMATH_GPT_final_concentration_after_procedure_l2356_235628


namespace NUMINAMATH_GPT_point_above_line_l2356_235685

theorem point_above_line (t : ℝ) : (∃ y : ℝ, y = (2 : ℝ)/3) → (t > (2 : ℝ)/3) :=
  by
  intro h
  sorry

end NUMINAMATH_GPT_point_above_line_l2356_235685


namespace NUMINAMATH_GPT_probability_exactly_half_red_balls_l2356_235620

def binomial_probability (n k : ℕ) (p : ℚ) : ℚ :=
  (Nat.factorial n) / ((Nat.factorial k) * (Nat.factorial (n - k))) * p^k * (1 - p)^(n - k)

theorem probability_exactly_half_red_balls :
  binomial_probability 8 4 (1/2) = 35/128 :=
by
  sorry

end NUMINAMATH_GPT_probability_exactly_half_red_balls_l2356_235620


namespace NUMINAMATH_GPT_weekly_earnings_l2356_235624

theorem weekly_earnings (total_earnings : ℕ) (weeks : ℕ) (h1 : total_earnings = 133) (h2 : weeks = 19) : 
  round (total_earnings / weeks : ℝ) = 7 := 
by 
  sorry

end NUMINAMATH_GPT_weekly_earnings_l2356_235624


namespace NUMINAMATH_GPT_find_y_l2356_235658

def star (a b : ℝ) : ℝ := 2 * a * b - 3 * b - a

theorem find_y (y : ℝ) (h : star 4 y = 80) : y = 16.8 :=
by
  sorry

end NUMINAMATH_GPT_find_y_l2356_235658


namespace NUMINAMATH_GPT_negation_of_existential_l2356_235601

theorem negation_of_existential :
  (¬ ∃ x_0 : ℝ, x_0^2 + 2 * x_0 - 3 > 0) = (∀ x : ℝ, x^2 + 2 * x - 3 ≤ 0) := 
by
  sorry

end NUMINAMATH_GPT_negation_of_existential_l2356_235601


namespace NUMINAMATH_GPT_impossible_path_2018_grid_l2356_235602

theorem impossible_path_2018_grid :
  ¬((∃ (path : Finset (Fin 2018 × Fin 2018)), 
    (0, 0) ∈ path ∧ (2017, 2017) ∈ path ∧ 
    (∀ {x y}, (x, y) ∈ path → (x + 1, y) ∈ path ∨ (x, y + 1) ∈ path ∨ (x - 1, y) ∈ path ∨ (x, y - 1) ∈ path) ∧ 
    (∀ {x y}, (x, y) ∈ path → (Finset.card path = 2018 * 2018)))) :=
by 
  sorry

end NUMINAMATH_GPT_impossible_path_2018_grid_l2356_235602


namespace NUMINAMATH_GPT_solve_for_n_l2356_235670

theorem solve_for_n (n : ℚ) (h : (1 / (n + 2)) + (2 / (n + 2)) + (n / (n + 2)) = 3) : n = -3/2 := 
by
  sorry

end NUMINAMATH_GPT_solve_for_n_l2356_235670


namespace NUMINAMATH_GPT_length_of_square_side_l2356_235692

theorem length_of_square_side 
  (r : ℝ) 
  (A : ℝ) 
  (h : A = 42.06195997410015) 
  (side_length : ℝ := 2 * r)
  (area_of_square : ℝ := side_length ^ 2)
  (segment_area : ℝ := 4 * (π * r * r / 4))
  (enclosed_area: ℝ := area_of_square - segment_area)
  (h2 : enclosed_area = A) :
  side_length = 14 :=
by sorry

end NUMINAMATH_GPT_length_of_square_side_l2356_235692
