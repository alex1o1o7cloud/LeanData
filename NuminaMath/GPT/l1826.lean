import Mathlib

namespace NUMINAMATH_GPT_value_of_a_l1826_182643

theorem value_of_a (a : ℝ) :
  (∀ x : ℝ, -1 < x ∧ x < 2 → x^2 - x + a < 0) → a = -2 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_value_of_a_l1826_182643


namespace NUMINAMATH_GPT_johns_subtraction_l1826_182629

theorem johns_subtraction 
  (a : ℕ) 
  (h₁ : (51 : ℕ)^2 = (50 : ℕ)^2 + 101) 
  (h₂ : (49 : ℕ)^2 = (50 : ℕ)^2 - b) 
  : b = 99 := 
by 
  sorry

end NUMINAMATH_GPT_johns_subtraction_l1826_182629


namespace NUMINAMATH_GPT_sandwiches_ordered_l1826_182654

-- Define the cost per sandwich
def cost_per_sandwich : ℝ := 5

-- Define the delivery fee
def delivery_fee : ℝ := 20

-- Define the tip percentage
def tip_percentage : ℝ := 0.10

-- Define the total amount received
def total_received : ℝ := 121

-- Define the equation representing the total amount received
def total_equation (x : ℝ) : Prop :=
  cost_per_sandwich * x + delivery_fee + (cost_per_sandwich * x + delivery_fee) * tip_percentage = total_received

-- Define the theorem that needs to be proved
theorem sandwiches_ordered (x : ℝ) : total_equation x ↔ x = 18 :=
sorry

end NUMINAMATH_GPT_sandwiches_ordered_l1826_182654


namespace NUMINAMATH_GPT_train_length_l1826_182680
-- Import all necessary libraries from Mathlib

-- Define the given conditions and prove the target
theorem train_length (L_t L_p : ℝ) (h1 : L_t = L_p) (h2 : 54 * (1000 / 3600) * 60 = 2 * L_t) : L_t = 450 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_train_length_l1826_182680


namespace NUMINAMATH_GPT_geometric_sequence_a3_equals_4_l1826_182613

variable {a : ℕ → ℝ}

def is_geometric_sequence (a : ℕ → ℝ) := ∃ r : ℝ, ∀ i, a (i+1) = a i * r

theorem geometric_sequence_a3_equals_4 
    (a_seq : is_geometric_sequence a) 
    (a_6_eq : a 6 = 6)
    (a_9_eq : a 9 = 9) : 
    a 3 = 4 := 
sorry

end NUMINAMATH_GPT_geometric_sequence_a3_equals_4_l1826_182613


namespace NUMINAMATH_GPT_total_amount_paid_l1826_182603

theorem total_amount_paid (g_p g_q m_p m_q : ℝ) (g_d g_t m_d m_t : ℝ) : 
    g_p = 70 -> g_q = 8 -> g_d = 0.05 -> g_t = 0.08 -> 
    m_p = 55 -> m_q = 9 -> m_d = 0.07 -> m_t = 0.11 -> 
    (g_p * g_q * (1 - g_d) * (1 + g_t) + m_p * m_q * (1 - m_d) * (1 + m_t)) = 1085.55 := by 
    sorry

end NUMINAMATH_GPT_total_amount_paid_l1826_182603


namespace NUMINAMATH_GPT_jigsaw_puzzle_pieces_l1826_182642

theorem jigsaw_puzzle_pieces
  (P : ℝ)
  (h1 : ∃ P, P = 0.90 * P + 0.72 * 0.10 * P + 0.504 * 0.08 * P + 504)
  (h2 : 0.504 * P = 504) :
  P = 1000 :=
by
  sorry

end NUMINAMATH_GPT_jigsaw_puzzle_pieces_l1826_182642


namespace NUMINAMATH_GPT_find_k_l1826_182600

noncomputable def f (a b c : ℤ) (x : ℤ) := a * x^2 + b * x + c

theorem find_k (a b c k : ℤ) 
  (h1 : f a b c 1 = 0) 
  (h2 : 50 < f a b c 7) (h2' : f a b c 7 < 60) 
  (h3 : 70 < f a b c 8) (h3' : f a b c 8 < 80) 
  (h4 : 5000 * k < f a b c 100) (h4' : f a b c 100 < 5000 * (k + 1)) : 
  k = 3 := 
sorry

end NUMINAMATH_GPT_find_k_l1826_182600


namespace NUMINAMATH_GPT_positive_difference_l1826_182679

theorem positive_difference
  (x y : ℝ)
  (h1 : x + y = 10)
  (h2 : x^2 - y^2 = 40) : abs (x - y) = 4 :=
sorry

end NUMINAMATH_GPT_positive_difference_l1826_182679


namespace NUMINAMATH_GPT_minimum_value_expression_l1826_182620

theorem minimum_value_expression (x : ℝ) : ∃ y : ℝ, (x + 1) * (x + 3) * (x + 5) * (x + 7) + 2050 = y ∧ ∀ z : ℝ, ((x + 1) * (x + 3) * (x + 5) * (x + 7) + 2050 ≥ z) ↔ (z = 2034) :=
by
  sorry

end NUMINAMATH_GPT_minimum_value_expression_l1826_182620


namespace NUMINAMATH_GPT_parallel_lines_a_perpendicular_lines_a_l1826_182699

-- Definitions of the lines
def l1 (a x y : ℝ) := a * x + 2 * y + 6 = 0
def l2 (a x y : ℝ) := x + (a - 1) * y + a^2 - 1 = 0

-- Statement for parallel lines problem
theorem parallel_lines_a (a : ℝ) :
  (∀ x y : ℝ, l1 a x y → l2 a x y) → (a = -1) :=
by
  sorry

-- Statement for perpendicular lines problem
theorem perpendicular_lines_a (a : ℝ) :
  (∀ x y : ℝ, l1 a x y → l2 a x y → (-a / 2) * (1 / (a - 1)) = -1) → (a = 2 / 3) :=
by
  sorry

end NUMINAMATH_GPT_parallel_lines_a_perpendicular_lines_a_l1826_182699


namespace NUMINAMATH_GPT_mary_flour_l1826_182672

-- Defining the conditions
def total_flour : ℕ := 11
def total_sugar : ℕ := 7
def flour_difference : ℕ := 2

-- The problem we want to prove
theorem mary_flour (F : ℕ) (C : ℕ) (S : ℕ)
  (h1 : C + 2 = S)
  (h2 : total_flour = F + C)
  (h3 : S = total_sugar) :
  F = 2 :=
by
  sorry

end NUMINAMATH_GPT_mary_flour_l1826_182672


namespace NUMINAMATH_GPT_mean_exercise_days_correct_l1826_182674

def students_exercise_days : List (Nat × Nat) := 
  [ (2, 0), (4, 1), (5, 2), (7, 3), (5, 4), (3, 5), (1, 6)]

def total_days_exercised : Nat := 
  List.sum (students_exercise_days.map (λ (count, days) => count * days))

def total_students : Nat := 
  List.sum (students_exercise_days.map Prod.fst)

def mean_exercise_days : Float := 
  total_days_exercised.toFloat / total_students.toFloat

theorem mean_exercise_days_correct : Float.round (mean_exercise_days * 100) / 100 = 2.81 :=
by
  sorry -- proof not required

end NUMINAMATH_GPT_mean_exercise_days_correct_l1826_182674


namespace NUMINAMATH_GPT_quadratic_inequality_cond_l1826_182691

theorem quadratic_inequality_cond (a : ℝ) :
  (∀ x : ℝ, ax^2 - ax + 1 > 0) ↔ (0 < a ∧ a < 4) :=
sorry

end NUMINAMATH_GPT_quadratic_inequality_cond_l1826_182691


namespace NUMINAMATH_GPT_circle_equation_solution_l1826_182655

theorem circle_equation_solution
  (a : ℝ)
  (h1 : a ^ 2 = a + 2)
  (h2 : (2 * a / (a + 2)) ^ 2 - 4 * a / (a + 2) > 0) : 
  a = -1 := 
sorry

end NUMINAMATH_GPT_circle_equation_solution_l1826_182655


namespace NUMINAMATH_GPT_combined_area_is_256_l1826_182621

-- Define the conditions
def side_length : ℝ := 16
def area_square : ℝ := side_length ^ 2

-- Define the property of the sides r and s
def r_s_property (r s : ℝ) : Prop :=
  (r + s)^2 + (r - s)^2 = side_length^2

-- The combined area of the four triangles
def combined_area_of_triangles (r s : ℝ) : ℝ :=
  2 * (r ^ 2 + s ^ 2)

-- Prove the final statement
theorem combined_area_is_256 (r s : ℝ) (h : r_s_property r s) :
  combined_area_of_triangles r s = 256 := by
  sorry

end NUMINAMATH_GPT_combined_area_is_256_l1826_182621


namespace NUMINAMATH_GPT_intersection_A_B_l1826_182690

noncomputable def A : Set ℝ := { y | ∃ x : ℝ, y = Real.sin x }
noncomputable def B : Set ℝ := { y | ∃ x : ℝ, y = x^2 }

theorem intersection_A_B : A ∩ B = { y | 0 ≤ y ∧ y ≤ 1 } :=
by 
  sorry

end NUMINAMATH_GPT_intersection_A_B_l1826_182690


namespace NUMINAMATH_GPT_line_AB_bisects_segment_DE_l1826_182694

variables {A B C D E : Type} [Inhabited A] [Inhabited B] [Inhabited C] [Inhabited D] [Inhabited E]
  {trapezoid : A × B × C × D} (AC CD : Prop) (BD_sym : Prop) (intersect_E : Prop)
  (line_AB : Prop) (bisects_DE : Prop)

-- Given a trapezoid ABCD
def is_trapezoid (A B C D : Type) : Prop := sorry

-- Given the diagonal AC is equal to the side CD
def diagonal_eq_leg (AC CD : Prop) : Prop := sorry

-- Given line BD is symmetric with respect to AD intersects AC at point E
def symmetric_line_intersect (BD_sym AD AC E : Prop) : Prop := sorry

-- Prove that line AB bisects segment DE
theorem line_AB_bisects_segment_DE
  (h_trapezoid : is_trapezoid A B C D)
  (h_diagonal_eq_leg : diagonal_eq_leg AC CD)
  (h_symmetric_line_intersect : symmetric_line_intersect BD_sym (sorry : Prop) AC intersect_E)
  (h_line_AB : line_AB) :
  bisects_DE := sorry

end NUMINAMATH_GPT_line_AB_bisects_segment_DE_l1826_182694


namespace NUMINAMATH_GPT_unique_solution_eq_l1826_182632

theorem unique_solution_eq (x : ℝ) : 
  (x ≠ 0 ∧ x ≠ 5) ∧ (∀ x, (3 * x^3 - 15 * x^2) / (x^2 - 5 * x) = x - 2) 
  → ∃! (x : ℝ), (3 * x ^ 3 - 15 * x ^ 2) / (x^2 - 5 * x) = x - 2 := 
by sorry

end NUMINAMATH_GPT_unique_solution_eq_l1826_182632


namespace NUMINAMATH_GPT_binary_subtraction_to_decimal_l1826_182665

theorem binary_subtraction_to_decimal :
  (511 - 63 = 448) :=
by
  sorry

end NUMINAMATH_GPT_binary_subtraction_to_decimal_l1826_182665


namespace NUMINAMATH_GPT_circle_circumference_difference_l1826_182624

theorem circle_circumference_difference (d_inner : ℝ) (h_inner : d_inner = 100) 
  (d_outer : ℝ) (h_outer : d_outer = d_inner + 30) :
  ((π * d_outer) - (π * d_inner)) = 30 * π :=
by 
  sorry

end NUMINAMATH_GPT_circle_circumference_difference_l1826_182624


namespace NUMINAMATH_GPT_initial_tomatoes_count_l1826_182677

-- Definitions and conditions
def birds_eat_fraction : ℚ := 1/3
def tomatoes_left : ℚ := 14
def fraction_tomatoes_left : ℚ := 2/3

-- We want to prove the initial number of tomatoes
theorem initial_tomatoes_count (initial_tomatoes : ℚ) 
  (h1 : tomatoes_left = fraction_tomatoes_left * initial_tomatoes) : 
  initial_tomatoes = 21 := 
by
  -- skipping the proof for now
  sorry

end NUMINAMATH_GPT_initial_tomatoes_count_l1826_182677


namespace NUMINAMATH_GPT_no_14_consecutive_divisible_by_2_to_11_l1826_182606

theorem no_14_consecutive_divisible_by_2_to_11 :
  ¬ ∃ (a : ℕ), ∀ i, i < 14 → ∃ p, Nat.Prime p ∧ 2 ≤ p ∧ p ≤ 11 ∧ (a + i) % p = 0 :=
by sorry

end NUMINAMATH_GPT_no_14_consecutive_divisible_by_2_to_11_l1826_182606


namespace NUMINAMATH_GPT_travel_time_and_speed_l1826_182653

theorem travel_time_and_speed :
  (total_time : ℝ) = 5.5 →
  (bus_whole_journey : ℝ) = 1 →
  (bus_half_journey : ℝ) = bus_whole_journey / 2 →
  (walk_half_journey : ℝ) = total_time - bus_half_journey →
  (walk_whole_journey : ℝ) = 2 * walk_half_journey →
  (bus_speed_factor : ℝ) = walk_whole_journey / bus_whole_journey →
  walk_whole_journey = 10 ∧ bus_speed_factor = 10 := by
  intros h1 h2 h3 h4 h5 h6
  sorry

end NUMINAMATH_GPT_travel_time_and_speed_l1826_182653


namespace NUMINAMATH_GPT_one_sixths_in_fraction_l1826_182641

theorem one_sixths_in_fraction :
  (11 / 3) / (1 / 6) = 22 :=
sorry

end NUMINAMATH_GPT_one_sixths_in_fraction_l1826_182641


namespace NUMINAMATH_GPT_max_product_distance_l1826_182625

-- Definitions for the conditions
def ellipse (x y : ℝ) : Prop := (x^2 / 9) + (y^2 / 4) = 1
def is_focus (F : ℝ × ℝ) : Prop := F = (3, 0) ∨ F = (-3, 0)

-- The theorem statement
theorem max_product_distance (M : ℝ × ℝ) (F1 F2 : ℝ × ℝ) 
  (h1 : ellipse M.1 M.2) 
  (h2 : is_focus F1) 
  (h3 : is_focus F2) : 
  (∃ x y, M = (x, y) ∧ ellipse x y) → 
  |(M.1 - F1.1)^2 + (M.2 - F1.2)^2| * |(M.1 - F2.1)^2 + (M.2 - F2.2)^2| ≤ 81 := 
sorry

end NUMINAMATH_GPT_max_product_distance_l1826_182625


namespace NUMINAMATH_GPT_ratio_depth_to_height_l1826_182649

theorem ratio_depth_to_height
  (Dean_height : ℝ := 9)
  (additional_depth : ℝ := 81)
  (water_depth : ℝ := Dean_height + additional_depth) :
  water_depth / Dean_height = 10 :=
by
  -- Dean_height = 9
  -- additional_depth = 81
  -- water_depth = 9 + 81 = 90
  -- water_depth / Dean_height = 90 / 9 = 10
  sorry

end NUMINAMATH_GPT_ratio_depth_to_height_l1826_182649


namespace NUMINAMATH_GPT_CoreyCandies_l1826_182645

theorem CoreyCandies (T C : ℕ) (h1 : T + C = 66) (h2 : T = C + 8) : C = 29 :=
by
  sorry

end NUMINAMATH_GPT_CoreyCandies_l1826_182645


namespace NUMINAMATH_GPT_monotonicity_of_f_range_of_k_for_three_zeros_l1826_182619

noncomputable def f (x k : ℝ) : ℝ := x^3 - k * x + k^2

def f_derivative (x k : ℝ) : ℝ := 3 * x^2 - k

theorem monotonicity_of_f (k : ℝ) : 
  (∀ x : ℝ, 0 <= f_derivative x k) ↔ k <= 0 :=
by sorry

theorem range_of_k_for_three_zeros : 
  (∃ x1 x2 x3 : ℝ, x1 ≠ x2 ∧ x2 ≠ x3 ∧ x1 ≠ x3 ∧ f x1 k = 0 ∧ f x2 k = 0 ∧ f x3 k = 0) ↔ (0 < k ∧ k < 4 / 27) :=
by sorry

end NUMINAMATH_GPT_monotonicity_of_f_range_of_k_for_three_zeros_l1826_182619


namespace NUMINAMATH_GPT_grill_cost_difference_l1826_182651

theorem grill_cost_difference:
  let in_store_price : Float := 129.99
  let payment_per_installment : Float := 32.49
  let number_of_installments : Float := 4
  let shipping_handling : Float := 9.99
  let total_tv_cost : Float := (number_of_installments * payment_per_installment) + shipping_handling
  let cost_difference : Float := in_store_price - total_tv_cost
  cost_difference * 100 = -996 := by
    sorry

end NUMINAMATH_GPT_grill_cost_difference_l1826_182651


namespace NUMINAMATH_GPT_ratio_of_running_to_swimming_l1826_182670

variable (Speed_swimming Time_swimming Distance_total Speed_factor : ℕ)

theorem ratio_of_running_to_swimming :
  let Distance_swimming := Speed_swimming * Time_swimming
  let Distance_running := Distance_total - Distance_swimming
  let Speed_running := Speed_factor * Speed_swimming
  let Time_running := Distance_running / Speed_running
  (Distance_total = 12) ∧
  (Speed_swimming = 2) ∧
  (Time_swimming = 2) ∧
  (Speed_factor = 4) →
  (Time_running : ℕ) / Time_swimming = 1 / 2 :=
by
  intros
  sorry

end NUMINAMATH_GPT_ratio_of_running_to_swimming_l1826_182670


namespace NUMINAMATH_GPT_interval_length_t_subset_interval_t_l1826_182671

-- Statement (1)
theorem interval_length_t (t : ℝ) (h : (Real.log t / Real.log 2) - 2 = 3) : t = 32 :=
  sorry

-- Statement (2)
theorem subset_interval_t (t : ℝ) (h : 2 ≤ Real.log t / Real.log 2 ∧ Real.log t / Real.log 2 ≤ 5) :
  0 < t ∧ t ≤ 32 :=
  sorry

end NUMINAMATH_GPT_interval_length_t_subset_interval_t_l1826_182671


namespace NUMINAMATH_GPT_initial_water_amount_l1826_182627

theorem initial_water_amount (W : ℝ) 
  (evap_per_day : ℝ := 0.0008) 
  (days : ℤ := 50) 
  (percentage_evap : ℝ := 0.004) 
  (evap_total : ℝ := evap_per_day * days) 
  (evap_eq : evap_total = percentage_evap * W) : 
  W = 10 := 
by
  sorry

end NUMINAMATH_GPT_initial_water_amount_l1826_182627


namespace NUMINAMATH_GPT_nested_sqrt_eq_two_l1826_182601

theorem nested_sqrt_eq_two (y : ℝ) (h : y = Real.sqrt (2 + y)) : y = 2 :=
by
  sorry

end NUMINAMATH_GPT_nested_sqrt_eq_two_l1826_182601


namespace NUMINAMATH_GPT_ratio_nephews_l1826_182626

variable (N : ℕ) -- The number of nephews Alden has now.
variable (Alden_had_50 : Prop := 50 = 50)
variable (Vihaan_more_60 : Prop := Vihaan = N + 60)
variable (Together_260 : Prop := N + (N + 60) = 260)

theorem ratio_nephews (N : ℕ) 
  (H1 : Alden_had_50)
  (H2 : Vihaan_more_60)
  (H3 : Together_260) :
  50 / N = 1 / 2 :=
by
  sorry

end NUMINAMATH_GPT_ratio_nephews_l1826_182626


namespace NUMINAMATH_GPT_initial_comparison_discount_comparison_B_based_on_discounted_A_l1826_182666

noncomputable section

-- Definitions based on the problem conditions
def A_price (x : ℝ) : ℝ := x
def B_price (x : ℝ) : ℝ := (0.2 * 2 * x + 0.3 * 3 * x + 0.4 * 4 * x) / 3
def A_discount_price (x : ℝ) : ℝ := 0.9 * x

-- Initial comparison
theorem initial_comparison (x : ℝ) (h : 0 < x) : B_price x < A_price x :=
by {
  sorry
}

-- After A's discount comparison
theorem discount_comparison (x : ℝ) (h : 0 < x) : A_discount_price x < B_price x :=
by {
  sorry
}

-- B's price based on A’s discounted price comparison
theorem B_based_on_discounted_A (x : ℝ) (h : 0 < x) : B_price (A_discount_price x) < A_discount_price x :=
by {
  sorry
}

end NUMINAMATH_GPT_initial_comparison_discount_comparison_B_based_on_discounted_A_l1826_182666


namespace NUMINAMATH_GPT_probability_of_matching_correctly_l1826_182634

-- Define the number of plants and seedlings.
def num_plants : ℕ := 4

-- Define the number of total arrangements.
def total_arrangements : ℕ := Nat.factorial num_plants

-- Define the number of correct arrangements.
def correct_arrangements : ℕ := 1

-- Define the probability of a correct guess.
def probability_of_correct_guess : ℚ := correct_arrangements / total_arrangements

-- The problem requires to prove that the probability of correct guess is 1/24
theorem probability_of_matching_correctly :
  probability_of_correct_guess = 1 / 24 :=
  by
    sorry

end NUMINAMATH_GPT_probability_of_matching_correctly_l1826_182634


namespace NUMINAMATH_GPT_sum_fraction_equals_two_l1826_182687

theorem sum_fraction_equals_two
  (a b c d : ℝ) (h₁ : a ≠ -1) (h₂ : b ≠ -1) (h₃ : c ≠ -1) (h₄ : d ≠ -1)
  (ω : ℂ) (h₅ : ω^4 = 1) (h₆ : ω ≠ 1)
  (h₇ : (1 / (a + ω)) + (1 / (b + ω)) + (1 / (c + ω)) + (1 / (d + ω)) = (4 / (ω^2))) 
  (h₈ : a + b + c + d = a * b * c * d)
  (h₉ : a * b + a * c + a * d + b * c + b * d + c * d = a * b * c + a * b * d + a * c * d + b * c * d) :
  (1 / (a + 1)) + (1 / (b + 1)) + (1 / (c + 1)) + (1 / (d + 1)) = 2 := 
sorry

end NUMINAMATH_GPT_sum_fraction_equals_two_l1826_182687


namespace NUMINAMATH_GPT_annual_income_from_investment_l1826_182661

theorem annual_income_from_investment
  (I : ℝ) (P : ℝ) (R : ℝ)
  (hI : I = 6800) (hP : P = 136) (hR : R = 0.60) :
  (I / P) * 100 * R = 3000 := by
  sorry

end NUMINAMATH_GPT_annual_income_from_investment_l1826_182661


namespace NUMINAMATH_GPT_f_of_x_squared_domain_l1826_182695

structure FunctionDomain (f : ℝ → ℝ) :=
  (domain : Set ℝ)
  (domain_eq : domain = Set.Icc 0 1)

theorem f_of_x_squared_domain (f : ℝ → ℝ) (h : FunctionDomain f) :
  FunctionDomain (fun x => f (x ^ 2)) :=
{
  domain := Set.Icc (-1) 1,
  domain_eq := sorry
}

end NUMINAMATH_GPT_f_of_x_squared_domain_l1826_182695


namespace NUMINAMATH_GPT_area_of_quadrilateral_ABFG_l1826_182658

/-- 
Given conditions:
1. Rectangle with dimensions AC = 40 and AE = 24.
2. Points B and F are midpoints of sides AC and AE, respectively.
3. G is the midpoint of DE.
Prove that the area of quadrilateral ABFG is 600 square units.
-/
theorem area_of_quadrilateral_ABFG (AC AE : ℝ) (B F G : ℤ) 
  (hAC : AC = 40) (hAE : AE = 24) (hB : B = 1/2 * AC) (hF : F = 1/2 * AE) (hG : G = 1/2 * AE):
  area_of_ABFG = 600 :=
by
  sorry

end NUMINAMATH_GPT_area_of_quadrilateral_ABFG_l1826_182658


namespace NUMINAMATH_GPT_average_visitors_per_day_correct_l1826_182633

-- Define the average number of visitors on Sundays
def avg_visitors_sunday : ℕ := 660

-- Define the average number of visitors on other days
def avg_visitors_other : ℕ := 240

-- Define the number of Sundays in a 30-day month starting with a Sunday
def num_sundays_in_month : ℕ := 5

-- Define the number of other days in a 30-day month starting with a Sunday
def num_other_days_in_month : ℕ := 25

-- Calculate the total number of visitors in the month
def total_visitors_in_month : ℕ :=
  (num_sundays_in_month * avg_visitors_sunday) + (num_other_days_in_month * avg_visitors_other)

-- Define the number of days in the month
def days_in_month : ℕ := 30

-- Define the average number of visitors per day
def avg_visitors_per_day := total_visitors_in_month / days_in_month

-- State the theorem to be proved
theorem average_visitors_per_day_correct :
  avg_visitors_per_day = 310 :=
by
  sorry

end NUMINAMATH_GPT_average_visitors_per_day_correct_l1826_182633


namespace NUMINAMATH_GPT_sqrt_div_equality_l1826_182692

noncomputable def sqrt_div (x y : ℝ) : ℝ := Real.sqrt x / Real.sqrt y

theorem sqrt_div_equality (x y : ℝ)
  (h : ( ( (1/3 : ℝ) ^ 2 + (1/4 : ℝ) ^ 2 ) / ( (1/5 : ℝ) ^ 2 + (1/6 : ℝ) ^ 2 ) = 25 * x / (73 * y) )) :
  sqrt_div x y = 5 / 2 :=
sorry

end NUMINAMATH_GPT_sqrt_div_equality_l1826_182692


namespace NUMINAMATH_GPT_find_first_offset_l1826_182656

variable (d y A x : ℝ)

theorem find_first_offset (h_d : d = 40) (h_y : y = 6) (h_A : A = 300) :
    x = 9 :=
by
  sorry

end NUMINAMATH_GPT_find_first_offset_l1826_182656


namespace NUMINAMATH_GPT_total_min_waiting_time_total_max_waiting_time_total_expected_waiting_time_l1826_182616

variables (a b: ℕ) (n m: ℕ)

def C (x y : ℕ) : ℕ := x.choose y

def T_min (a n m : ℕ) : ℕ :=
  a * C n 2 + a * m * n + b * C m 2

def T_max (a n m : ℕ) : ℕ :=
  a * C n 2 + b * m * n + b * C m 2

def E_T (a b n m : ℕ) : ℕ :=
  C (n + m) 2 * ((b * m + a * n) / (m + n))

theorem total_min_waiting_time (a b : ℕ) : T_min 1 5 3 = 40 :=
  by sorry

theorem total_max_waiting_time (a b : ℕ) : T_max 1 5 3 = 100 :=
  by sorry

theorem total_expected_waiting_time (a b : ℕ) : E_T 1 5 5 3 = 70 :=
  by sorry

end NUMINAMATH_GPT_total_min_waiting_time_total_max_waiting_time_total_expected_waiting_time_l1826_182616


namespace NUMINAMATH_GPT_evaluate_expression_l1826_182608

theorem evaluate_expression : 4 * (8 - 3) - 7 = 13 :=
by sorry

end NUMINAMATH_GPT_evaluate_expression_l1826_182608


namespace NUMINAMATH_GPT_eval_expression_l1826_182668

theorem eval_expression :
  Int.ceil (7 / 3 : ℚ) + Int.floor (-7 / 3 : ℚ) - Int.ceil (2 / 3 : ℚ) = -1 := 
by 
  sorry

end NUMINAMATH_GPT_eval_expression_l1826_182668


namespace NUMINAMATH_GPT_isha_original_length_l1826_182660

variable (current_length sharpened_off : ℕ)

-- Condition 1: Isha's pencil is now 14 inches long
def isha_current_length : current_length = 14 := sorry

-- Condition 2: She sharpened off 17 inches of her pencil
def isha_sharpened_off : sharpened_off = 17 := sorry

-- Statement to prove:
theorem isha_original_length (current_length sharpened_off : ℕ) 
  (h1 : current_length = 14) (h2 : sharpened_off = 17) :
  current_length + sharpened_off = 31 :=
by
  sorry

end NUMINAMATH_GPT_isha_original_length_l1826_182660


namespace NUMINAMATH_GPT_find_integer_pairs_l1826_182684

theorem find_integer_pairs :
  ∃ (x y : ℤ), (x = 30 ∧ y = 21) ∨ (x = -21 ∧ y = -30) ∧ (x^2 + y^2 + 27 = 456 * Int.sqrt (x - y)) :=
by
  sorry

end NUMINAMATH_GPT_find_integer_pairs_l1826_182684


namespace NUMINAMATH_GPT_find_p_q_l1826_182648

theorem find_p_q (p q : ℤ) (h : ∀ x : ℤ, (x - 5) * (x + 2) = x^2 + p * x + q) :
  p = -3 ∧ q = -10 :=
by {
  -- The proof would go here, but for now we'll use sorry to indicate it's incomplete.
  sorry
}

end NUMINAMATH_GPT_find_p_q_l1826_182648


namespace NUMINAMATH_GPT_trapezoid_area_l1826_182683

variable (A B C D K : Type)
variable [Nonempty A] [Nonempty B] [Nonempty C] [Nonempty D] [Nonempty K]

-- Define the lengths as given in the conditions
def AK : ℝ := 16
def DK : ℝ := 4
def CD : ℝ := 6

-- Define the property that the trapezoid ABCD has an inscribed circle
axiom trapezoid_with_inscribed_circle (ABCD : Prop) : Prop

-- The Lean theorem statement
theorem trapezoid_area (ABCD : Prop) (AK DK CD : ℝ) 
  (H1 : trapezoid_with_inscribed_circle ABCD)
  (H2 : AK = 16)
  (H3 : DK = 4)
  (H4 : CD = 6) : 
  ∃ (area : ℝ), area = 432 :=
by
  sorry

end NUMINAMATH_GPT_trapezoid_area_l1826_182683


namespace NUMINAMATH_GPT_minimize_quadratic_l1826_182610

theorem minimize_quadratic (y : ℝ) : 
  ∃ m, m = 3 * y ^ 2 - 18 * y + 11 ∧ 
       (∀ z : ℝ, 3 * z ^ 2 - 18 * z + 11 ≥ m) ∧ 
       m = -16 := 
sorry

end NUMINAMATH_GPT_minimize_quadratic_l1826_182610


namespace NUMINAMATH_GPT_polynomial_is_2y2_l1826_182663

variables (x y : ℝ)

theorem polynomial_is_2y2 (P : ℝ → ℝ → ℝ) (h : P x y + (x^2 - y^2) = x^2 + y^2) : 
  P x y = 2 * y^2 :=
by
  sorry

end NUMINAMATH_GPT_polynomial_is_2y2_l1826_182663


namespace NUMINAMATH_GPT_summation_problem_l1826_182637

open BigOperators

theorem summation_problem : 
  (∑ i in Finset.range 50, ∑ j in Finset.range 75, 2 * (i + 1) + 3 * (j + 1) + (i + 1) * (j + 1)) = 4275000 :=
by
  sorry

end NUMINAMATH_GPT_summation_problem_l1826_182637


namespace NUMINAMATH_GPT_rods_in_one_mile_l1826_182611

/-- Definitions based on given conditions -/
def miles_to_furlongs := 8
def furlongs_to_rods := 40

/-- The theorem stating the number of rods in one mile -/
theorem rods_in_one_mile : (miles_to_furlongs * furlongs_to_rods) = 320 := 
  sorry

end NUMINAMATH_GPT_rods_in_one_mile_l1826_182611


namespace NUMINAMATH_GPT_shelby_rain_time_l1826_182669

noncomputable def speedNonRainy : ℚ := 30 / 60
noncomputable def speedRainy : ℚ := 20 / 60
noncomputable def totalDistance : ℚ := 16
noncomputable def totalTime : ℚ := 40

theorem shelby_rain_time : 
  ∃ x : ℚ, (speedNonRainy * (totalTime - x) + speedRainy * x = totalDistance) ∧ x = 24 := 
by
  sorry

end NUMINAMATH_GPT_shelby_rain_time_l1826_182669


namespace NUMINAMATH_GPT_solve_equation_l1826_182698

theorem solve_equation (x : ℝ) (h1 : x ≠ 2 / 3) :
  (7 * x + 3) / (3 * x ^ 2 + 7 * x - 6) = (3 * x) / (3 * x - 2) ↔
  x = (-1 + Real.sqrt 10) / 3 ∨ x = (-1 - Real.sqrt 10) / 3 :=
by sorry

end NUMINAMATH_GPT_solve_equation_l1826_182698


namespace NUMINAMATH_GPT_minimum_value_x_plus_y_l1826_182652

theorem minimum_value_x_plus_y (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x^2 + y^2 = 1) : x + y = 16 :=
sorry

end NUMINAMATH_GPT_minimum_value_x_plus_y_l1826_182652


namespace NUMINAMATH_GPT_areas_of_triangle_and_parallelogram_are_equal_l1826_182615

theorem areas_of_triangle_and_parallelogram_are_equal (b : ℝ) :
  let parallelogram_height := 100
  let triangle_height := 200
  let area_parallelogram := b * parallelogram_height
  let area_triangle := (1/2) * b * triangle_height
  area_parallelogram = area_triangle :=
by
  -- conditions
  let parallelogram_height := 100
  let triangle_height := 200
  let area_parallelogram := b * parallelogram_height
  let area_triangle := (1 / 2) * b * triangle_height
  -- relationship
  show area_parallelogram = area_triangle
  sorry

end NUMINAMATH_GPT_areas_of_triangle_and_parallelogram_are_equal_l1826_182615


namespace NUMINAMATH_GPT_gcd_sub_12_eq_36_l1826_182604

theorem gcd_sub_12_eq_36 :
  Nat.gcd 7344 48 - 12 = 36 := 
by 
  sorry

end NUMINAMATH_GPT_gcd_sub_12_eq_36_l1826_182604


namespace NUMINAMATH_GPT_marias_workday_ends_at_six_pm_l1826_182667

theorem marias_workday_ends_at_six_pm :
  ∀ (start_time : ℕ) (work_hours : ℕ) (lunch_start_time : ℕ) (lunch_duration : ℕ) (afternoon_break_time : ℕ) (afternoon_break_duration : ℕ) (end_time : ℕ),
    start_time = 8 ∧
    work_hours = 8 ∧
    lunch_start_time = 13 ∧
    lunch_duration = 1 ∧
    afternoon_break_time = 15 * 60 + 30 ∧  -- Converting 3:30 P.M. to minutes
    afternoon_break_duration = 15 ∧
    end_time = 18  -- 6:00 P.M. in 24-hour format
    → end_time = 18 :=
by
  -- map 13:00 -> 1:00 P.M.,  15:30 -> 3:30 P.M.; convert 6:00 P.M. back 
  sorry

end NUMINAMATH_GPT_marias_workday_ends_at_six_pm_l1826_182667


namespace NUMINAMATH_GPT_find_phi_l1826_182607

open Real

noncomputable def f (x φ : ℝ) : ℝ := cos (2 * x + φ)
noncomputable def g (x φ : ℝ) : ℝ := cos (2 * x - π/2 + φ)

theorem find_phi 
  (h1 : 0 < φ) 
  (h2 : φ < π) 
  (symmetry_condition : ∀ x, g (π/2 - x) φ = g (π/2 + x) φ) 
  : φ = π / 2 
:= by 
  sorry

end NUMINAMATH_GPT_find_phi_l1826_182607


namespace NUMINAMATH_GPT_find_k_parallel_find_k_perpendicular_l1826_182689

noncomputable def veca : (ℝ × ℝ) := (1, 2)
noncomputable def vecb : (ℝ × ℝ) := (-3, 2)

def is_parallel (u v : (ℝ × ℝ)) : Prop := 
  ∃ k : ℝ, k ≠ 0 ∧ u = (k * v.1, k * v.2)

def is_perpendicular (u v : (ℝ × ℝ)) : Prop := 
  u.1 * v.1 + u.2 * v.2 = 0

def calc_vector (k : ℝ) (a b : (ℝ × ℝ)) : (ℝ × ℝ) :=
  (k * a.1 + b.1, k * a.2 + b.2)

theorem find_k_parallel : 
  ∃ k : ℝ, is_parallel (calc_vector k veca vecb) (calc_vector 1 veca (-2 * vecb)) := sorry

theorem find_k_perpendicular :
  ∃ k : ℝ, k = 25 / 3 ∧ is_perpendicular (calc_vector k veca vecb) (calc_vector 1 veca (-2 * vecb)) := sorry

end NUMINAMATH_GPT_find_k_parallel_find_k_perpendicular_l1826_182689


namespace NUMINAMATH_GPT_math_problem_l1826_182605

theorem math_problem (x y n : ℕ) (hx : 10 ≤ x ∧ x < 100) (hy : 10 ≤ y ∧ y < 100)
  (hy_reverse : ∃ a b, x = 10 * a + b ∧ y = 10 * b + a) 
  (h_xy_square_sum : x^2 + y^2 = n^2) : x + y + n = 132 :=
sorry

end NUMINAMATH_GPT_math_problem_l1826_182605


namespace NUMINAMATH_GPT_count_even_positive_integers_satisfy_inequality_l1826_182681

open Int

noncomputable def countEvenPositiveIntegersInInterval : ℕ :=
  (List.filter (fun n : ℕ => n % 2 = 0) [2, 4, 6, 8, 10, 12]).length

theorem count_even_positive_integers_satisfy_inequality :
  countEvenPositiveIntegersInInterval = 6 := by
  sorry

end NUMINAMATH_GPT_count_even_positive_integers_satisfy_inequality_l1826_182681


namespace NUMINAMATH_GPT_local_minimum_at_2_l1826_182657

open Real

noncomputable def f (x : ℝ) : ℝ := x^3 - 12 * x

def f' (x : ℝ) : ℝ := 3 * x^2 - 12

theorem local_minimum_at_2 :
  (∀ x : ℝ, -2 < x ∧ x < 2 → f' x < 0) →
  (∀ x : ℝ, x > 2 → f' x > 0) →
  (∃ ε > 0, ∀ x : ℝ, abs (x - 2) < ε → f x > f 2) :=
by
  sorry

end NUMINAMATH_GPT_local_minimum_at_2_l1826_182657


namespace NUMINAMATH_GPT_find_circle_radius_l1826_182638

-- Definitions based on the given conditions
def circle_eq (x y : ℝ) : Prop := (x^2 - 8*x + y^2 - 10*y + 34 = 0)

-- Problem statement
theorem find_circle_radius (x y : ℝ) : circle_eq x y → ∃ r : ℝ, r = Real.sqrt 7 :=
by
  sorry

end NUMINAMATH_GPT_find_circle_radius_l1826_182638


namespace NUMINAMATH_GPT_tiered_water_pricing_usage_l1826_182636

theorem tiered_water_pricing_usage (total_cost : ℤ) (water_used : ℤ) :
  (total_cost = 60) →
  (water_used > 12 ∧ water_used ≤ 18) →
  (3 * 12 + (water_used - 12) * 6 = total_cost) →
  water_used = 16 :=
by
  intros h_cost h_range h_eq
  sorry

end NUMINAMATH_GPT_tiered_water_pricing_usage_l1826_182636


namespace NUMINAMATH_GPT_arithmetic_sequence_first_term_range_l1826_182685

theorem arithmetic_sequence_first_term_range (a_1 : ℝ) (d : ℝ) (a_10 : ℝ) (a_11 : ℝ) :
  d = (Real.pi / 8) → 
  (a_1 + 9 * d ≤ 0) → 
  (a_1 + 10 * d ≥ 0) → 
  - (5 * Real.pi / 4) ≤ a_1 ∧ a_1 ≤ - (9 * Real.pi / 8) :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_first_term_range_l1826_182685


namespace NUMINAMATH_GPT_max_value_y_l1826_182650

theorem max_value_y (x : ℝ) : ∃ y, y = -3 * x^2 + 6 ∧ ∀ z, (∃ x', z = -3 * x'^2 + 6) → z ≤ y :=
by sorry

end NUMINAMATH_GPT_max_value_y_l1826_182650


namespace NUMINAMATH_GPT_fraction_identity_l1826_182686

theorem fraction_identity (a b : ℚ) (h1 : 3 * a = 4 * b) (h2 : a ≠ 0 ∧ b ≠ 0) : (a + b) / a = 7 / 4 :=
by
  sorry

end NUMINAMATH_GPT_fraction_identity_l1826_182686


namespace NUMINAMATH_GPT_perimeter_regular_polygon_l1826_182696

-- Condition definitions
def is_regular_polygon (n : ℕ) (s : ℝ) : Prop := 
  n * s > 0

def exterior_angle (E : ℝ) (n : ℕ) : Prop := 
  E = 360 / n

def side_length (s : ℝ) : Prop :=
  s = 6

-- Theorem statement to prove the perimeter is 24 units
theorem perimeter_regular_polygon 
  (n : ℕ) (s E : ℝ)
  (h1 : is_regular_polygon n s)
  (h2 : exterior_angle E n)
  (h3 : side_length s)
  (h4 : E = 90) :
  4 * s = 24 :=
by
  sorry

end NUMINAMATH_GPT_perimeter_regular_polygon_l1826_182696


namespace NUMINAMATH_GPT_sum_of_digits_is_32_l1826_182622

/-- 
Prove that the sum of digits \( A, B, C, D, E \) is 32 given the constraints
1. \( A, B, C, D, E \) are single digits.
2. The sum of the units column 3E results in 1 (units place of 2011).
3. The sum of the hundreds column 3A and carry equals 20 (hundreds place of 2011).
-/
theorem sum_of_digits_is_32
  (A B C D E : ℕ)
  (h1 : A < 10)
  (h2 : B < 10)
  (h3 : C < 10)
  (h4 : D < 10)
  (h5 : E < 10)
  (units_condition : 3 * E % 10 = 1)
  (hundreds_condition : ∃ carry: ℕ, carry < 10 ∧ 3 * A + carry = 20) :
  A + B + C + D + E = 32 := 
sorry

end NUMINAMATH_GPT_sum_of_digits_is_32_l1826_182622


namespace NUMINAMATH_GPT_find_alpha_l1826_182612

-- Declare the conditions
variables (α : ℝ) (h₀ : 0 < α) (h₁ : α < 90) (h₂ : Real.sin (α - 10 * Real.pi / 180) = Real.sqrt 3 / 2)

theorem find_alpha : α = 70 * Real.pi / 180 :=
sorry

end NUMINAMATH_GPT_find_alpha_l1826_182612


namespace NUMINAMATH_GPT_negation_of_existential_proposition_l1826_182647

theorem negation_of_existential_proposition :
  (¬ ∃ x : ℝ, Real.exp x < 0) = (∀ x : ℝ, Real.exp x ≥ 0) :=
sorry

end NUMINAMATH_GPT_negation_of_existential_proposition_l1826_182647


namespace NUMINAMATH_GPT_total_sugar_weight_l1826_182659

theorem total_sugar_weight (x y : ℝ) (h1 : y - x = 8) (h2 : x - 1 = 0.6 * (y + 1)) : x + y = 40 := by
  sorry

end NUMINAMATH_GPT_total_sugar_weight_l1826_182659


namespace NUMINAMATH_GPT_jungkook_mother_age_four_times_jungkook_age_l1826_182639

-- Definitions of conditions
def jungkoo_age : ℕ := 16
def mother_age : ℕ := 46

-- Theorem statement for the problem
theorem jungkook_mother_age_four_times_jungkook_age :
  ∃ (x : ℕ), (mother_age - x = 4 * (jungkoo_age - x)) ∧ x = 6 :=
by
  sorry

end NUMINAMATH_GPT_jungkook_mother_age_four_times_jungkook_age_l1826_182639


namespace NUMINAMATH_GPT_prod_one_minus_nonneg_reals_ge_half_l1826_182664

theorem prod_one_minus_nonneg_reals_ge_half (x1 x2 x3 : ℝ) 
  (h1 : 0 ≤ x1) (h2 : 0 ≤ x2) (h3 : 0 ≤ x3)
  (h_sum : x1 + x2 + x3 ≤ 1/2) : 
  (1 - x1) * (1 - x2) * (1 - x3) ≥ 1 / 2 := 
by
  sorry

end NUMINAMATH_GPT_prod_one_minus_nonneg_reals_ge_half_l1826_182664


namespace NUMINAMATH_GPT_calendars_ordered_l1826_182697

theorem calendars_ordered 
  (C D : ℝ) 
  (h1 : C + D = 500) 
  (h2 : 0.75 * C + 0.50 * D = 300) 
  : C = 200 :=
by
  sorry

end NUMINAMATH_GPT_calendars_ordered_l1826_182697


namespace NUMINAMATH_GPT_rival_awards_l1826_182676

theorem rival_awards (jessie_multiple : ℕ) (scott_awards : ℕ) (rival_multiple : ℕ) 
  (h1 : jessie_multiple = 3) 
  (h2 : scott_awards = 4) 
  (h3 : rival_multiple = 2) 
  : (rival_multiple * (jessie_multiple * scott_awards) = 24) :=
by 
  sorry

end NUMINAMATH_GPT_rival_awards_l1826_182676


namespace NUMINAMATH_GPT_salt_added_correctly_l1826_182688

-- Define the problem's conditions and the correct answer in Lean
variable (x : ℝ) (y : ℝ)
variable (S : ℝ := 0.2 * x) -- original salt
variable (E : ℝ := (1 / 4) * x) -- evaporated water
variable (New_volume : ℝ := x - E + 10) -- new volume after adding water

theorem salt_added_correctly :
  x = 150 → y = (1 / 3) * New_volume - S :=
by
  sorry

end NUMINAMATH_GPT_salt_added_correctly_l1826_182688


namespace NUMINAMATH_GPT_consecutive_integers_satisfy_inequality_l1826_182678

theorem consecutive_integers_satisfy_inequality :
  ∀ (n m : ℝ), n + 1 = m ∧ n < Real.sqrt 26 ∧ Real.sqrt 26 < m → m + n = 11 :=
by
  sorry

end NUMINAMATH_GPT_consecutive_integers_satisfy_inequality_l1826_182678


namespace NUMINAMATH_GPT_derivative_at_2_l1826_182673

noncomputable def f (x : ℝ) : ℝ := (x + 1)^2 * (x - 1)

theorem derivative_at_2 : deriv f 2 = 15 := by
  sorry

end NUMINAMATH_GPT_derivative_at_2_l1826_182673


namespace NUMINAMATH_GPT_inclination_angle_l1826_182682

theorem inclination_angle (α : ℝ) (t : ℝ) (h : 0 < α ∧ α < π / 2) :
  let x := 1 + t * Real.cos (α + 3 * π / 2)
  let y := 2 + t * Real.sin (α + 3 * π / 2)
  ∃ θ, θ = α + π / 2 := by
  sorry

end NUMINAMATH_GPT_inclination_angle_l1826_182682


namespace NUMINAMATH_GPT_simplify_complex_expr_l1826_182618

theorem simplify_complex_expr : ∀ (i : ℂ), (4 - 2 * i) - (7 - 2 * i) + (6 - 3 * i) = 3 - 3 * i := by
  intro i
  sorry

end NUMINAMATH_GPT_simplify_complex_expr_l1826_182618


namespace NUMINAMATH_GPT_tom_bought_new_books_l1826_182614

def original_books : ℕ := 5
def sold_books : ℕ := 4
def current_books : ℕ := 39

def new_books (original_books sold_books current_books : ℕ) : ℕ :=
  current_books - (original_books - sold_books)

theorem tom_bought_new_books :
  new_books original_books sold_books current_books = 38 :=
by
  sorry

end NUMINAMATH_GPT_tom_bought_new_books_l1826_182614


namespace NUMINAMATH_GPT_octagon_area_difference_is_512_l1826_182640

noncomputable def octagon_area_difference (side_length : ℝ) : ℝ :=
  let initial_octagon_area := 2 * (1 + Real.sqrt 2) * side_length^2
  let triangle_area := (1 / 2) * side_length^2
  let total_triangle_area := 8 * triangle_area
  let inner_octagon_area := initial_octagon_area - total_triangle_area
  initial_octagon_area - inner_octagon_area

theorem octagon_area_difference_is_512 :
  octagon_area_difference 16 = 512 :=
by
  -- This is where the proof would be filled in.
  sorry

end NUMINAMATH_GPT_octagon_area_difference_is_512_l1826_182640


namespace NUMINAMATH_GPT_find_5b_l1826_182631

-- Define variables and conditions
variables (a b : ℝ)
axiom h1 : 6 * a + 3 * b = 0
axiom h2 : a = b - 3

-- State the theorem to prove
theorem find_5b : 5 * b = 10 :=
sorry

end NUMINAMATH_GPT_find_5b_l1826_182631


namespace NUMINAMATH_GPT_largest_multiple_of_15_less_than_500_l1826_182609

-- Define the conditions
def is_multiple_of_15 (n : Nat) : Prop :=
  ∃ (k : Nat), n = 15 * k

def is_positive (n : Nat) : Prop :=
  n > 0

def is_less_than_500 (n : Nat) : Prop :=
  n < 500

-- Define the main theorem statement
theorem largest_multiple_of_15_less_than_500 :
  ∀ (n : Nat), is_multiple_of_15 n ∧ is_positive n ∧ is_less_than_500 n → n ≤ 495 :=
sorry

end NUMINAMATH_GPT_largest_multiple_of_15_less_than_500_l1826_182609


namespace NUMINAMATH_GPT_total_points_other_members_18_l1826_182635

-- Definitions
def total_points (x : ℕ) (S : ℕ) (T : ℕ) (M : ℕ) (y : ℕ) :=
  S + T + M + y = x

def Sam_scored (x S : ℕ) := S = x / 3

def Taylor_scored (x T : ℕ) := T = 3 * x / 8

def Morgan_scored (M : ℕ) := M = 21

def other_members_scored (y : ℕ) := ∃ (a b c d e f g h : ℕ),
  a ≤ 3 ∧ b ≤ 3 ∧ c ≤ 3 ∧ d ≤ 3 ∧ e ≤ 3 ∧ f ≤ 3 ∧ g ≤ 3 ∧ h ≤ 3 ∧
  y = a + b + c + d + e + f + g + h

-- Theorem
theorem total_points_other_members_18 (x y S T M : ℕ) :
  Sam_scored x S → Taylor_scored x T → Morgan_scored M → total_points x S T M y → other_members_scored y → y = 18 :=
by
  intros hSam hTaylor hMorgan hTotal hOther
  sorry

end NUMINAMATH_GPT_total_points_other_members_18_l1826_182635


namespace NUMINAMATH_GPT_increase_150_percent_of_80_l1826_182644

theorem increase_150_percent_of_80 : 80 * 1.5 + 80 = 200 :=
by
  sorry

end NUMINAMATH_GPT_increase_150_percent_of_80_l1826_182644


namespace NUMINAMATH_GPT_number_of_digits_in_N_l1826_182617

noncomputable def N : ℕ := 2^12 * 5^8

theorem number_of_digits_in_N : (Nat.digits 10 N).length = 10 := by
  sorry

end NUMINAMATH_GPT_number_of_digits_in_N_l1826_182617


namespace NUMINAMATH_GPT_maximize_f_l1826_182662

open Nat

-- Define the combination function
def comb (n k : ℕ) : ℕ := choose n k

-- Define the probability function f(n)
def f (n : ℕ) : ℚ := 
  (comb n 2 * comb (100 - n) 8 : ℚ) / comb 100 10

-- Define the theorem to find the value of n that maximizes f(n)
theorem maximize_f : ∃ n : ℕ, 2 ≤ n ∧ n ≤ 92 ∧ (∀ m : ℕ, 2 ≤ m ∧ m ≤ 92 → f n ≥ f m) ∧ n = 20 :=
by
  sorry

end NUMINAMATH_GPT_maximize_f_l1826_182662


namespace NUMINAMATH_GPT_stripe_area_is_480pi_l1826_182693

noncomputable def stripeArea (diameter : ℝ) (height : ℝ) (width : ℝ) (revolutions : ℕ) : ℝ :=
  let radius := diameter / 2
  let circumference := 2 * Real.pi * radius
  let stripeLength := circumference * revolutions
  let area := width * stripeLength
  area

theorem stripe_area_is_480pi : stripeArea 40 90 4 3 = 480 * Real.pi :=
  by
    show stripeArea 40 90 4 3 = 480 * Real.pi
    sorry

end NUMINAMATH_GPT_stripe_area_is_480pi_l1826_182693


namespace NUMINAMATH_GPT_inequality_count_l1826_182630

theorem inequality_count
  (x y a b : ℝ)
  (hx_pos : 0 < x)
  (hy_pos : 0 < y)
  (ha_pos : 0 < a)
  (hb_pos : 0 < b)
  (hx_lt_one : x < 1)
  (hy_lt_one : y < 1)
  (hx_lt_a : x < a)
  (hy_lt_b : y < b)
  (h_sum : x + y = a - b) :
  ({(x + y < a + b), (x - y < a - b), (x * y < a * b)}:Finset Prop).card = 3 :=
by
  sorry

end NUMINAMATH_GPT_inequality_count_l1826_182630


namespace NUMINAMATH_GPT_unoccupied_volume_correct_l1826_182602

-- Define the conditions given in the problem
def tank_length := 12 -- inches
def tank_width := 8 -- inches
def tank_height := 10 -- inches
def water_fraction := 1 / 3
def ice_cube_side := 1 -- inches
def num_ice_cubes := 12

-- Calculate the occupied volume
noncomputable def tank_volume : ℝ := tank_length * tank_width * tank_height
noncomputable def water_volume : ℝ := tank_volume * water_fraction
noncomputable def ice_cube_volume : ℝ := ice_cube_side^3
noncomputable def total_ice_volume : ℝ := ice_cube_volume * num_ice_cubes
noncomputable def total_occupied_volume : ℝ := water_volume + total_ice_volume

-- Calculate the unoccupied volume
noncomputable def unoccupied_volume : ℝ := tank_volume - total_occupied_volume

-- State the problem
theorem unoccupied_volume_correct : unoccupied_volume = 628 := by
  sorry

end NUMINAMATH_GPT_unoccupied_volume_correct_l1826_182602


namespace NUMINAMATH_GPT_robert_ate_more_l1826_182628

variable (robert_chocolates : ℕ) (nickel_chocolates : ℕ)
variable (robert_ate_9 : robert_chocolates = 9) (nickel_ate_2 : nickel_chocolates = 2)

theorem robert_ate_more : robert_chocolates - nickel_chocolates = 7 :=
  by
    sorry

end NUMINAMATH_GPT_robert_ate_more_l1826_182628


namespace NUMINAMATH_GPT_not_possible_2002_pieces_l1826_182646

theorem not_possible_2002_pieces (k : ℤ) : ¬ (1 + 7 * k = 2002) :=
by
  sorry

end NUMINAMATH_GPT_not_possible_2002_pieces_l1826_182646


namespace NUMINAMATH_GPT_number_of_democrats_in_senate_l1826_182623

/-
This Lean statement captures the essence of the problem: proving the number of Democrats in the Senate (S_D) is 55,
under given conditions involving the House's and Senate's number of Democrats and Republicans.
-/

theorem number_of_democrats_in_senate
  (D R S_D S_R : ℕ)
  (h1 : D + R = 434)
  (h2 : R = D + 30)
  (h3 : S_D + S_R = 100)
  (h4 : S_D * 4 = S_R * 5) :
  S_D = 55 := by
  sorry

end NUMINAMATH_GPT_number_of_democrats_in_senate_l1826_182623


namespace NUMINAMATH_GPT_train_stoppage_time_l1826_182675

theorem train_stoppage_time (speed_excluding_stoppages speed_including_stoppages : ℝ) 
(H1 : speed_excluding_stoppages = 54) 
(H2 : speed_including_stoppages = 36) : (18 / (54 / 60)) = 20 :=
by
  sorry

end NUMINAMATH_GPT_train_stoppage_time_l1826_182675
