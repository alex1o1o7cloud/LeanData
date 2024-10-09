import Mathlib

namespace find_first_offset_l208_20846

variable (d y A x : ℝ)

theorem find_first_offset (h_d : d = 40) (h_y : y = 6) (h_A : A = 300) :
    x = 9 :=
by
  sorry

end find_first_offset_l208_20846


namespace total_sugar_weight_l208_20862

theorem total_sugar_weight (x y : ℝ) (h1 : y - x = 8) (h2 : x - 1 = 0.6 * (y + 1)) : x + y = 40 := by
  sorry

end total_sugar_weight_l208_20862


namespace CoreyCandies_l208_20870

theorem CoreyCandies (T C : ℕ) (h1 : T + C = 66) (h2 : T = C + 8) : C = 29 :=
by
  sorry

end CoreyCandies_l208_20870


namespace area_of_quadrilateral_ABFG_l208_20829

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

end area_of_quadrilateral_ABFG_l208_20829


namespace average_visitors_per_day_correct_l208_20851

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

end average_visitors_per_day_correct_l208_20851


namespace first_team_odd_is_correct_l208_20899

noncomputable def odd_for_first_team : Real := 
  let odd2 := 5.23
  let odd3 := 3.25
  let odd4 := 2.05
  let bet_amount := 5.00
  let expected_win := 223.0072
  let total_odds := expected_win / bet_amount
  let denominator := odd2 * odd3 * odd4
  total_odds / denominator

theorem first_team_odd_is_correct : 
  odd_for_first_team = 1.28 := by 
  sorry

end first_team_odd_is_correct_l208_20899


namespace circle_equation_solution_l208_20845

theorem circle_equation_solution
  (a : ℝ)
  (h1 : a ^ 2 = a + 2)
  (h2 : (2 * a / (a + 2)) ^ 2 - 4 * a / (a + 2) > 0) : 
  a = -1 := 
sorry

end circle_equation_solution_l208_20845


namespace find_integer_pairs_l208_20855

theorem find_integer_pairs :
  ∃ (x y : ℤ), (x = 30 ∧ y = 21) ∨ (x = -21 ∧ y = -30) ∧ (x^2 + y^2 + 27 = 456 * Int.sqrt (x - y)) :=
by
  sorry

end find_integer_pairs_l208_20855


namespace find_circle_radius_l208_20863

-- Definitions based on the given conditions
def circle_eq (x y : ℝ) : Prop := (x^2 - 8*x + y^2 - 10*y + 34 = 0)

-- Problem statement
theorem find_circle_radius (x y : ℝ) : circle_eq x y → ∃ r : ℝ, r = Real.sqrt 7 :=
by
  sorry

end find_circle_radius_l208_20863


namespace donald_juice_l208_20814

variable (P D : ℕ)

theorem donald_juice (h1 : P = 3) (h2 : D = 2 * P + 3) : D = 9 := by
  sorry

end donald_juice_l208_20814


namespace tiered_water_pricing_usage_l208_20821

theorem tiered_water_pricing_usage (total_cost : ℤ) (water_used : ℤ) :
  (total_cost = 60) →
  (water_used > 12 ∧ water_used ≤ 18) →
  (3 * 12 + (water_used - 12) * 6 = total_cost) →
  water_used = 16 :=
by
  intros h_cost h_range h_eq
  sorry

end tiered_water_pricing_usage_l208_20821


namespace unique_solution_eq_l208_20824

theorem unique_solution_eq (x : ℝ) : 
  (x ≠ 0 ∧ x ≠ 5) ∧ (∀ x, (3 * x^3 - 15 * x^2) / (x^2 - 5 * x) = x - 2) 
  → ∃! (x : ℝ), (3 * x ^ 3 - 15 * x ^ 2) / (x^2 - 5 * x) = x - 2 := 
by sorry

end unique_solution_eq_l208_20824


namespace marias_workday_ends_at_six_pm_l208_20872

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

end marias_workday_ends_at_six_pm_l208_20872


namespace intersection_A_B_l208_20866

noncomputable def A : Set ℝ := { y | ∃ x : ℝ, y = Real.sin x }
noncomputable def B : Set ℝ := { y | ∃ x : ℝ, y = x^2 }

theorem intersection_A_B : A ∩ B = { y | 0 ≤ y ∧ y ≤ 1 } :=
by 
  sorry

end intersection_A_B_l208_20866


namespace roots_sum_of_quadratic_l208_20896

theorem roots_sum_of_quadratic :
  ∀ x1 x2 : ℝ, (Polynomial.eval x1 (Polynomial.X ^ 2 + 2 * Polynomial.X - 1) = 0) →
              (Polynomial.eval x2 (Polynomial.X ^ 2 + 2 * Polynomial.X - 1) = 0) →
              x1 + x2 = -2 :=
by
  intros x1 x2 h1 h2
  sorry

end roots_sum_of_quadratic_l208_20896


namespace one_sixths_in_fraction_l208_20857

theorem one_sixths_in_fraction :
  (11 / 3) / (1 / 6) = 22 :=
sorry

end one_sixths_in_fraction_l208_20857


namespace binary_subtraction_to_decimal_l208_20831

theorem binary_subtraction_to_decimal :
  (511 - 63 = 448) :=
by
  sorry

end binary_subtraction_to_decimal_l208_20831


namespace maximum_sets_l208_20809

-- define the initial conditions
def dinner_forks : Nat := 6
def knives : Nat := dinner_forks + 9
def soup_spoons : Nat := 2 * knives
def teaspoons : Nat := dinner_forks / 2
def dessert_forks : Nat := teaspoons / 3
def butter_knives : Nat := 2 * dessert_forks

def max_capacity_g : Nat := 20000

def weight_dinner_fork : Nat := 80
def weight_knife : Nat := 100
def weight_soup_spoon : Nat := 85
def weight_teaspoon : Nat := 50
def weight_dessert_fork : Nat := 70
def weight_butter_knife : Nat := 65

-- Calculate the total weight of the existing cutlery
def total_weight_existing : Nat := 
  (dinner_forks * weight_dinner_fork) + 
  (knives * weight_knife) + 
  (soup_spoons * weight_soup_spoon) + 
  (teaspoons * weight_teaspoon) + 
  (dessert_forks * weight_dessert_fork) + 
  (butter_knives * weight_butter_knife)

-- Calculate the weight of one 2-piece cutlery set (1 knife + 1 dinner fork)
def weight_set : Nat := weight_knife + weight_dinner_fork

-- The remaining capacity in the drawer
def remaining_capacity_g : Nat := max_capacity_g - total_weight_existing

-- The maximum number of 2-piece cutlery sets that can be added
def max_2_piece_sets : Nat := remaining_capacity_g / weight_set

-- Theorem: maximum number of 2-piece cutlery sets that can be added is 84
theorem maximum_sets : max_2_piece_sets = 84 :=
by
  sorry

end maximum_sets_l208_20809


namespace find_5b_l208_20822

-- Define variables and conditions
variables (a b : ℝ)
axiom h1 : 6 * a + 3 * b = 0
axiom h2 : a = b - 3

-- State the theorem to prove
theorem find_5b : 5 * b = 10 :=
sorry

end find_5b_l208_20822


namespace mean_exercise_days_correct_l208_20826

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

end mean_exercise_days_correct_l208_20826


namespace sum_fraction_equals_two_l208_20874

theorem sum_fraction_equals_two
  (a b c d : ℝ) (h₁ : a ≠ -1) (h₂ : b ≠ -1) (h₃ : c ≠ -1) (h₄ : d ≠ -1)
  (ω : ℂ) (h₅ : ω^4 = 1) (h₆ : ω ≠ 1)
  (h₇ : (1 / (a + ω)) + (1 / (b + ω)) + (1 / (c + ω)) + (1 / (d + ω)) = (4 / (ω^2))) 
  (h₈ : a + b + c + d = a * b * c * d)
  (h₉ : a * b + a * c + a * d + b * c + b * d + c * d = a * b * c + a * b * d + a * c * d + b * c * d) :
  (1 / (a + 1)) + (1 / (b + 1)) + (1 / (c + 1)) + (1 / (d + 1)) = 2 := 
sorry

end sum_fraction_equals_two_l208_20874


namespace fraction_identity_l208_20825

theorem fraction_identity (a b : ℚ) (h1 : 3 * a = 4 * b) (h2 : a ≠ 0 ∧ b ≠ 0) : (a + b) / a = 7 / 4 :=
by
  sorry

end fraction_identity_l208_20825


namespace local_minimum_at_2_l208_20828

open Real

noncomputable def f (x : ℝ) : ℝ := x^3 - 12 * x

def f' (x : ℝ) : ℝ := 3 * x^2 - 12

theorem local_minimum_at_2 :
  (∀ x : ℝ, -2 < x ∧ x < 2 → f' x < 0) →
  (∀ x : ℝ, x > 2 → f' x > 0) →
  (∃ ε > 0, ∀ x : ℝ, abs (x - 2) < ε → f x > f 2) :=
by
  sorry

end local_minimum_at_2_l208_20828


namespace arithmetic_sequence_first_term_range_l208_20819

theorem arithmetic_sequence_first_term_range (a_1 : ℝ) (d : ℝ) (a_10 : ℝ) (a_11 : ℝ) :
  d = (Real.pi / 8) → 
  (a_1 + 9 * d ≤ 0) → 
  (a_1 + 10 * d ≥ 0) → 
  - (5 * Real.pi / 4) ≤ a_1 ∧ a_1 ≤ - (9 * Real.pi / 8) :=
by
  sorry

end arithmetic_sequence_first_term_range_l208_20819


namespace area_of_annulus_l208_20883

section annulus
variables {R r x : ℝ}
variable (h1 : R > r)
variable (h2 : R^2 - r^2 = x^2)

theorem area_of_annulus (R r x : ℝ) (h1 : R > r) (h2 : R^2 - r^2 = x^2) : 
  π * R^2 - π * r^2 = π * x^2 :=
sorry

end annulus

end area_of_annulus_l208_20883


namespace sandwiches_ordered_l208_20844

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

end sandwiches_ordered_l208_20844


namespace increase_150_percent_of_80_l208_20869

theorem increase_150_percent_of_80 : 80 * 1.5 + 80 = 200 :=
by
  sorry

end increase_150_percent_of_80_l208_20869


namespace log_identity_l208_20880

noncomputable def log (base : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log base

theorem log_identity : log 2 5 * log 3 2 * log 5 3 = 1 :=
by sorry

end log_identity_l208_20880


namespace strips_overlap_area_l208_20813

theorem strips_overlap_area (L1 L2 AL AR S : ℝ) (hL1 : L1 = 9) (hL2 : L2 = 7) (hAL : AL = 27) (hAR : AR = 18) 
    (hrel : (AL + S) / (AR + S) = L1 / L2) : S = 13.5 := 
by
  sorry

end strips_overlap_area_l208_20813


namespace ratio_depth_to_height_l208_20836

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

end ratio_depth_to_height_l208_20836


namespace prod_one_minus_nonneg_reals_ge_half_l208_20877

theorem prod_one_minus_nonneg_reals_ge_half (x1 x2 x3 : ℝ) 
  (h1 : 0 ≤ x1) (h2 : 0 ≤ x2) (h3 : 0 ≤ x3)
  (h_sum : x1 + x2 + x3 ≤ 1/2) : 
  (1 - x1) * (1 - x2) * (1 - x3) ≥ 1 / 2 := 
by
  sorry

end prod_one_minus_nonneg_reals_ge_half_l208_20877


namespace max_value_y_l208_20837

theorem max_value_y (x : ℝ) : ∃ y, y = -3 * x^2 + 6 ∧ ∀ z, (∃ x', z = -3 * x'^2 + 6) → z ≤ y :=
by sorry

end max_value_y_l208_20837


namespace grill_cost_difference_l208_20878

theorem grill_cost_difference:
  let in_store_price : Float := 129.99
  let payment_per_installment : Float := 32.49
  let number_of_installments : Float := 4
  let shipping_handling : Float := 9.99
  let total_tv_cost : Float := (number_of_installments * payment_per_installment) + shipping_handling
  let cost_difference : Float := in_store_price - total_tv_cost
  cost_difference * 100 = -996 := by
    sorry

end grill_cost_difference_l208_20878


namespace negation_of_existential_proposition_l208_20865

theorem negation_of_existential_proposition :
  (¬ ∃ x : ℝ, Real.exp x < 0) = (∀ x : ℝ, Real.exp x ≥ 0) :=
sorry

end negation_of_existential_proposition_l208_20865


namespace initial_tomatoes_count_l208_20834

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

end initial_tomatoes_count_l208_20834


namespace sum_of_possible_values_l208_20890

theorem sum_of_possible_values (A B : ℕ) 
  (hA1 : A < 10) (hA2 : 0 < A) (hB1 : B < 10) (hB2 : 0 < B)
  (h1 : 3 / 12 < A / 12) (h2 : A / 12 < 7 / 12)
  (h3 : 1 / 10 < 1 / B) (h4 : 1 / B < 1 / 3) :
  3 + 6 = 9 :=
by
  sorry

end sum_of_possible_values_l208_20890


namespace area_relation_l208_20807

noncomputable def area_triangle (A B C : ℝ × ℝ) : ℝ := 
  0.5 * (A.1 * (B.2 - C.2) + B.1 * (C.2 - A.2) + C.1 * (A.2 - B.2))

theorem area_relation (A B C A' B' C' : ℝ × ℝ) (hAA'BB'CC'parallel: 
  ∃ k : ℝ, (A'.1 - A.1 = k * (B'.1 - B.1)) ∧ (A'.2 - A.2 = k * (B'.2 - B.2)) ∧ 
           (B'.1 - B.1 = k * (C'.1 - C.1)) ∧ (B'.2 - B.2 = k * (C'.2 - C.2))) :
  3 * (area_triangle A B C + area_triangle A' B' C') = 
    area_triangle A B' C' + area_triangle B C' A' + area_triangle C A' B' +
    area_triangle A' B C + area_triangle B' C A + area_triangle C' A B := 
sorry

end area_relation_l208_20807


namespace initial_comparison_discount_comparison_B_based_on_discounted_A_l208_20871

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

end initial_comparison_discount_comparison_B_based_on_discounted_A_l208_20871


namespace summation_problem_l208_20852

open BigOperators

theorem summation_problem : 
  (∑ i in Finset.range 50, ∑ j in Finset.range 75, 2 * (i + 1) + 3 * (j + 1) + (i + 1) * (j + 1)) = 4275000 :=
by
  sorry

end summation_problem_l208_20852


namespace add_fractions_l208_20895

theorem add_fractions (a : ℝ) (h : a ≠ 0): (3 / a) + (2 / a) = 5 / a := by
  sorry

end add_fractions_l208_20895


namespace area_of_triangle_OAB_is_5_l208_20805

-- Define the parameters and assumptions
def OA : ℝ × ℝ := (-2, 1)
def OB : ℝ × ℝ := (4, 3)

noncomputable def area_triangle_OAB (OA OB : ℝ × ℝ) : ℝ :=
  1 / 2 * (OA.1 * OB.2 - OA.2 * OB.1)

-- The theorem we want to prove:
theorem area_of_triangle_OAB_is_5 : area_triangle_OAB OA OB = 5 := by
  sorry

end area_of_triangle_OAB_is_5_l208_20805


namespace minimum_k_conditions_l208_20886

theorem minimum_k_conditions (k : ℝ) :
  (∀ (a b c : ℝ), a ≠ 0 → b ≠ 0 → c ≠ 0 → (|a - b| ≤ k ∨ |1/a - 1/b| ≤ k)) ↔ k = 3/2 :=
sorry

end minimum_k_conditions_l208_20886


namespace polynomial_is_2y2_l208_20876

variables (x y : ℝ)

theorem polynomial_is_2y2 (P : ℝ → ℝ → ℝ) (h : P x y + (x^2 - y^2) = x^2 + y^2) : 
  P x y = 2 * y^2 :=
by
  sorry

end polynomial_is_2y2_l208_20876


namespace total_balloons_correct_l208_20815

-- Definitions based on the conditions
def brookes_initial_balloons : Nat := 12
def brooke_additional_balloons : Nat := 8

def tracys_initial_balloons : Nat := 6
def tracy_additional_balloons : Nat := 24

-- Calculate the number of balloons each person has after the additions and Tracy popping half
def brookes_final_balloons : Nat := brookes_initial_balloons + brooke_additional_balloons
def tracys_balloons_after_addition : Nat := tracys_initial_balloons + tracy_additional_balloons
def tracys_final_balloons : Nat := tracys_balloons_after_addition / 2

-- Total number of balloons
def total_balloons : Nat := brookes_final_balloons + tracys_final_balloons

-- The proof statement
theorem total_balloons_correct : total_balloons = 35 := by
  -- Proof would go here (but we'll skip with sorry)
  sorry

end total_balloons_correct_l208_20815


namespace rival_awards_l208_20840

theorem rival_awards (jessie_multiple : ℕ) (scott_awards : ℕ) (rival_multiple : ℕ) 
  (h1 : jessie_multiple = 3) 
  (h2 : scott_awards = 4) 
  (h3 : rival_multiple = 2) 
  : (rival_multiple * (jessie_multiple * scott_awards) = 24) :=
by 
  sorry

end rival_awards_l208_20840


namespace decrypt_nbui_is_math_l208_20891

-- Define the sets A and B as the 26 English letters
def A := {c : Char | c ≥ 'a' ∧ c ≤ 'z'}
def B := A

-- Define the mapping f from A to B
def f (c : Char) : Char :=
  if c = 'z' then 'a'
  else Char.ofNat (c.toNat + 1)

-- Define the decryption function g (it reverses the mapping f)
def g (c : Char) : Char :=
  if c = 'a' then 'z'
  else Char.ofNat (c.toNat - 1)

-- Define the decryption of the given ciphertext
def decrypt (ciphertext : String) : String :=
  ciphertext.map g

-- Prove that the decryption of "nbui" is "math"
theorem decrypt_nbui_is_math : decrypt "nbui" = "math" :=
  by
  sorry

end decrypt_nbui_is_math_l208_20891


namespace value_of_a_l208_20859

theorem value_of_a (a : ℝ) :
  (∀ x : ℝ, -1 < x ∧ x < 2 → x^2 - x + a < 0) → a = -2 :=
by
  intro h
  sorry

end value_of_a_l208_20859


namespace height_of_table_l208_20803

variable (h l w h3 : ℝ)

-- Conditions from the problem
def condition1 : Prop := h3 = 4
def configurationA : Prop := l + h - w = 50
def configurationB : Prop := w + h + h3 - l = 44

-- Statement to prove
theorem height_of_table (h l w h3 : ℝ) 
  (cond1 : condition1 h3)
  (confA : configurationA h l w)
  (confB : configurationB h l w h3) : 
  h = 45 := 
by 
  sorry

end height_of_table_l208_20803


namespace octagon_area_difference_is_512_l208_20856

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

end octagon_area_difference_is_512_l208_20856


namespace not_possible_2002_pieces_l208_20873

theorem not_possible_2002_pieces (k : ℤ) : ¬ (1 + 7 * k = 2002) :=
by
  sorry

end not_possible_2002_pieces_l208_20873


namespace total_apples_l208_20802

theorem total_apples (A B C : ℕ) (h1 : A + B = 11) (h2 : B + C = 18) (h3 : A + C = 19) : A + B + C = 24 :=  
by
  -- Skip the proof
  sorry

end total_apples_l208_20802


namespace consecutive_integers_satisfy_inequality_l208_20867

theorem consecutive_integers_satisfy_inequality :
  ∀ (n m : ℝ), n + 1 = m ∧ n < Real.sqrt 26 ∧ Real.sqrt 26 < m → m + n = 11 :=
by
  sorry

end consecutive_integers_satisfy_inequality_l208_20867


namespace interval_length_t_subset_interval_t_l208_20861

-- Statement (1)
theorem interval_length_t (t : ℝ) (h : (Real.log t / Real.log 2) - 2 = 3) : t = 32 :=
  sorry

-- Statement (2)
theorem subset_interval_t (t : ℝ) (h : 2 ≤ Real.log t / Real.log 2 ∧ Real.log t / Real.log 2 ≤ 5) :
  0 < t ∧ t ≤ 32 :=
  sorry

end interval_length_t_subset_interval_t_l208_20861


namespace joan_total_spent_on_clothing_l208_20810

theorem joan_total_spent_on_clothing :
  let shorts_cost := 15.00
  let jacket_cost := 14.82
  let shirt_cost := 12.51
  let shoes_cost := 21.67
  let hat_cost := 8.75
  let belt_cost := 6.34
  shorts_cost + jacket_cost + shirt_cost + shoes_cost + hat_cost + belt_cost = 79.09 :=
by
  sorry

end joan_total_spent_on_clothing_l208_20810


namespace quadratic_inequality_cond_l208_20820

theorem quadratic_inequality_cond (a : ℝ) :
  (∀ x : ℝ, ax^2 - ax + 1 > 0) ↔ (0 < a ∧ a < 4) :=
sorry

end quadratic_inequality_cond_l208_20820


namespace area_inside_C_outside_A_B_l208_20879

-- Define the given circles with corresponding radii and positions
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define the circles A, B, and C with the specific properties given
def CircleA : Circle := { center := (0, 0), radius := 1 }
def CircleB : Circle := { center := (2, 0), radius := 1 }
def CircleC : Circle := { center := (1, 2), radius := 2 }

-- Given that Circle C is tangent to the midpoint M of the line segment AB
-- Prove the area inside Circle C but outside Circle A and B
theorem area_inside_C_outside_A_B : 
  let area_inside_C := π * CircleC.radius ^ 2
  let overlap_area := (π - 2)
  area_inside_C - overlap_area = 3 * π + 2 := by
  sorry

end area_inside_C_outside_A_B_l208_20879


namespace stripe_area_is_480pi_l208_20881

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

end stripe_area_is_480pi_l208_20881


namespace imons_no_entanglements_l208_20892

-- Define the fundamental structure for imons and their entanglements.
universe u
variable {α : Type u}

-- Define a graph structure to represent imons and their entanglement.
structure Graph (α : Type u) where
  vertices : Finset α
  edges : Finset (α × α)
  edge_sym : ∀ {x y}, (x, y) ∈ edges → (y, x) ∈ edges

-- Define the operations that can be performed on imons.
structure ImonOps (G : Graph α) where
  destroy : {v : α} → G.vertices.card % 2 = 1
  double : Graph α

-- Prove the main theorem
theorem imons_no_entanglements (G : Graph α) (op : ImonOps G) : 
  ∃ seq : List (ImonOps G), ∀ g : Graph α, g ∈ (seq.map (λ h => h.double)) → g.edges = ∅ :=
by
  sorry -- The proof would be constructed here.

end imons_no_entanglements_l208_20892


namespace salt_added_correctly_l208_20875

-- Define the problem's conditions and the correct answer in Lean
variable (x : ℝ) (y : ℝ)
variable (S : ℝ := 0.2 * x) -- original salt
variable (E : ℝ := (1 / 4) * x) -- evaporated water
variable (New_volume : ℝ := x - E + 10) -- new volume after adding water

theorem salt_added_correctly :
  x = 150 → y = (1 / 3) * New_volume - S :=
by
  sorry

end salt_added_correctly_l208_20875


namespace minimum_value_x_plus_y_l208_20817

theorem minimum_value_x_plus_y (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x^2 + y^2 = 1) : x + y = 16 :=
sorry

end minimum_value_x_plus_y_l208_20817


namespace derivative_at_2_l208_20830

noncomputable def f (x : ℝ) : ℝ := (x + 1)^2 * (x - 1)

theorem derivative_at_2 : deriv f 2 = 15 := by
  sorry

end derivative_at_2_l208_20830


namespace julia_total_spend_l208_20897

noncomputable def total_cost_julia_puppy : ℝ :=
  let adoption_fee := 20.00
  let dog_food := 20.00
  let treat_cost := 2.50
  let treat_count := 2
  let treats := treat_cost * treat_count
  let toys := 15.00
  let crate := 20.00
  let bed := 20.00
  let collar_leash := 15.00
  let total_supplies := dog_food + treats + toys + crate + bed + collar_leash
  let discount := 0.20 * total_supplies
  let final_supplies := total_supplies - discount
  final_supplies + adoption_fee

theorem julia_total_spend : total_cost_julia_puppy = 96.00 :=
by
  sorry

end julia_total_spend_l208_20897


namespace minnie_takes_more_time_l208_20888

def minnie_speed_flat : ℝ := 25
def minnie_speed_downhill : ℝ := 35
def minnie_speed_uphill : ℝ := 10
def penny_speed_flat : ℝ := 35
def penny_speed_downhill : ℝ := 45
def penny_speed_uphill : ℝ := 15

def distance_A_to_B : ℝ := 15
def distance_B_to_D : ℝ := 20
def distance_D_to_C : ℝ := 25

def distance_C_to_B : ℝ := 20
def distance_D_to_A : ℝ := 25

noncomputable def time_minnie : ℝ :=
(distance_A_to_B / minnie_speed_uphill) + 
(distance_B_to_D / minnie_speed_downhill) + 
(distance_D_to_C / minnie_speed_flat)

noncomputable def time_penny : ℝ :=
(distance_C_to_B / penny_speed_uphill) + 
(distance_B_to_D / penny_speed_downhill) + 
(distance_D_to_A / penny_speed_flat)

noncomputable def time_diff : ℝ := (time_minnie - time_penny) * 60

theorem minnie_takes_more_time : time_diff = 10 := by
  sorry

end minnie_takes_more_time_l208_20888


namespace inclination_angle_l208_20853

theorem inclination_angle (α : ℝ) (t : ℝ) (h : 0 < α ∧ α < π / 2) :
  let x := 1 + t * Real.cos (α + 3 * π / 2)
  let y := 2 + t * Real.sin (α + 3 * π / 2)
  ∃ θ, θ = α + π / 2 := by
  sorry

end inclination_angle_l208_20853


namespace find_k_parallel_find_k_perpendicular_l208_20839

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

end find_k_parallel_find_k_perpendicular_l208_20839


namespace space_shuttle_speed_kmph_l208_20804

-- Question: Prove that the speed of the space shuttle in kilometers per hour is 32400, given it travels at 9 kilometers per second and there are 3600 seconds in an hour.
theorem space_shuttle_speed_kmph :
  (9 * 3600 = 32400) :=
by
  sorry

end space_shuttle_speed_kmph_l208_20804


namespace original_number_fraction_l208_20808

theorem original_number_fraction (x : ℚ) (h : 1 + 1/x = 9/4) : x = 4/5 := by
  sorry

end original_number_fraction_l208_20808


namespace find_p_q_l208_20835

theorem find_p_q (p q : ℤ) (h : ∀ x : ℤ, (x - 5) * (x + 2) = x^2 + p * x + q) :
  p = -3 ∧ q = -10 :=
by {
  -- The proof would go here, but for now we'll use sorry to indicate it's incomplete.
  sorry
}

end find_p_q_l208_20835


namespace union_A_B_l208_20898

noncomputable def A : Set ℝ := { x | x^2 - 3 * x + 2 = 0 }
noncomputable def B : Set ℝ := { x | x^3 = x }

theorem union_A_B : A ∪ B = { -1, 0, 1, 2 } := by
  sorry

end union_A_B_l208_20898


namespace sqrt_div_equality_l208_20860

noncomputable def sqrt_div (x y : ℝ) : ℝ := Real.sqrt x / Real.sqrt y

theorem sqrt_div_equality (x y : ℝ)
  (h : ( ( (1/3 : ℝ) ^ 2 + (1/4 : ℝ) ^ 2 ) / ( (1/5 : ℝ) ^ 2 + (1/6 : ℝ) ^ 2 ) = 25 * x / (73 * y) )) :
  sqrt_div x y = 5 / 2 :=
sorry

end sqrt_div_equality_l208_20860


namespace bill_left_with_money_l208_20811

def foolsgold (ounces_sold : Nat) (price_per_ounce : Nat) (fine : Nat): Int :=
  (ounces_sold * price_per_ounce) - fine

theorem bill_left_with_money :
  foolsgold 8 9 50 = 22 :=
by
  sorry

end bill_left_with_money_l208_20811


namespace jungkook_mother_age_four_times_jungkook_age_l208_20864

-- Definitions of conditions
def jungkoo_age : ℕ := 16
def mother_age : ℕ := 46

-- Theorem statement for the problem
theorem jungkook_mother_age_four_times_jungkook_age :
  ∃ (x : ℕ), (mother_age - x = 4 * (jungkoo_age - x)) ∧ x = 6 :=
by
  sorry

end jungkook_mother_age_four_times_jungkook_age_l208_20864


namespace eval_expression_l208_20843

theorem eval_expression :
  Int.ceil (7 / 3 : ℚ) + Int.floor (-7 / 3 : ℚ) - Int.ceil (2 / 3 : ℚ) = -1 := 
by 
  sorry

end eval_expression_l208_20843


namespace maximize_f_l208_20841

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

end maximize_f_l208_20841


namespace probability_of_matching_correctly_l208_20850

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

end probability_of_matching_correctly_l208_20850


namespace parallel_lines_a_values_l208_20889

theorem parallel_lines_a_values (a : Real) : 
  (∃ k : Real, 2 = k * a ∧ -a = k * (-8)) ↔ (a = 4 ∨ a = -4) := sorry

end parallel_lines_a_values_l208_20889


namespace booknote_unique_elements_l208_20894

def booknote_string : String := "booknote"
def booknote_set : Finset Char := { 'b', 'o', 'k', 'n', 't', 'e' }

theorem booknote_unique_elements : booknote_set.card = 6 :=
by
  sorry

end booknote_unique_elements_l208_20894


namespace travel_time_and_speed_l208_20818

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

end travel_time_and_speed_l208_20818


namespace jane_mistake_l208_20801

theorem jane_mistake (x y z : ℤ) (h1 : x - y + z = 15) (h2 : x - y - z = 7) : x - y = 11 :=
by sorry

end jane_mistake_l208_20801


namespace trapezoid_area_l208_20854

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

end trapezoid_area_l208_20854


namespace find_x_value_l208_20806

theorem find_x_value (x : ℝ) (h1 : Real.sin (π / 2 - x) = -Real.sqrt 3 / 2) (h2 : π < x ∧ x < 2 * π) : x = 7 * π / 6 :=
sorry

end find_x_value_l208_20806


namespace baker_sold_cakes_l208_20816

theorem baker_sold_cakes :
  ∀ (C : ℕ),  -- C is the number of cakes Baker sold
    (∃ (cakes pastries : ℕ), 
      cakes = 14 ∧ 
      pastries = 153 ∧ 
      (∃ (sold_pastries : ℕ), sold_pastries = 8 ∧ 
      C = 89 + sold_pastries)) 
  → C = 97 :=
by
  intros C h
  rcases h with ⟨cakes, pastries, cakes_eq, pastries_eq, ⟨sold_pastries, sold_pastries_eq, C_eq⟩⟩
  -- Fill in the proof details
  sorry

end baker_sold_cakes_l208_20816


namespace percent_savings_per_roll_l208_20884

theorem percent_savings_per_roll 
  (cost_case : ℕ := 900) -- In cents, equivalent to $9
  (cost_individual : ℕ := 100) -- In cents, equivalent to $1
  (num_rolls : ℕ := 12) :
  (cost_individual - (cost_case / num_rolls)) * 100 / cost_individual = 25 := 
sorry

end percent_savings_per_roll_l208_20884


namespace international_postage_surcharge_l208_20885

theorem international_postage_surcharge 
  (n_letters : ℕ) 
  (std_postage_per_letter : ℚ) 
  (n_international : ℕ) 
  (total_cost : ℚ) 
  (cents_per_dollar : ℚ) 
  (std_total_cost : ℚ) 
  : 
  n_letters = 4 →
  std_postage_per_letter = 108 / 100 →
  n_international = 2 →
  total_cost = 460 / 100 →
  cents_per_dollar = 100 →
  std_total_cost = n_letters * std_postage_per_letter →
  (total_cost - std_total_cost) / n_international * cents_per_dollar = 14 := 
sorry

end international_postage_surcharge_l208_20885


namespace total_points_other_members_18_l208_20842

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

end total_points_other_members_18_l208_20842


namespace mary_flour_l208_20847

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

end mary_flour_l208_20847


namespace triangle_side_lengths_approx_l208_20812

noncomputable def approx_side_lengths (AB : ℝ) (BAC ABC : ℝ) : ℝ × ℝ :=
  let α := BAC * Real.pi / 180
  let β := ABC * Real.pi / 180
  let c := AB
  let β1 := (90 - (BAC)) * Real.pi / 180
  let m := 2 * c * α * (β1 + 3) / (9 - α * β1)
  let c1 := 2 * c * β1 * (α + 3) / (9 - α * β1)
  let β2 := β1 - β
  let γ1 := α + β
  let a1 := β2 / γ1 * (γ1 + 3) / (β2 + 3) * m
  let a := (9 - β2 * γ1) / (2 * γ1 * (β2 + 3)) * m
  let b := c1 - a1
  (a, b)

theorem triangle_side_lengths_approx (AB : ℝ) (BAC ABC : ℝ) (hAB : AB = 441) (hBAC : BAC = 16.2) (hABC : ABC = 40.6) :
  approx_side_lengths AB BAC ABC = (147, 344) := by
  sorry

end triangle_side_lengths_approx_l208_20812


namespace count_even_positive_integers_satisfy_inequality_l208_20827

open Int

noncomputable def countEvenPositiveIntegersInInterval : ℕ :=
  (List.filter (fun n : ℕ => n % 2 = 0) [2, 4, 6, 8, 10, 12]).length

theorem count_even_positive_integers_satisfy_inequality :
  countEvenPositiveIntegersInInterval = 6 := by
  sorry

end count_even_positive_integers_satisfy_inequality_l208_20827


namespace train_stoppage_time_l208_20823

theorem train_stoppage_time (speed_excluding_stoppages speed_including_stoppages : ℝ) 
(H1 : speed_excluding_stoppages = 54) 
(H2 : speed_including_stoppages = 36) : (18 / (54 / 60)) = 20 :=
by
  sorry

end train_stoppage_time_l208_20823


namespace shelby_rain_time_l208_20848

noncomputable def speedNonRainy : ℚ := 30 / 60
noncomputable def speedRainy : ℚ := 20 / 60
noncomputable def totalDistance : ℚ := 16
noncomputable def totalTime : ℚ := 40

theorem shelby_rain_time : 
  ∃ x : ℚ, (speedNonRainy * (totalTime - x) + speedRainy * x = totalDistance) ∧ x = 24 := 
by
  sorry

end shelby_rain_time_l208_20848


namespace bill_score_l208_20893

theorem bill_score
  (J B S : ℕ)
  (h1 : B = J + 20)
  (h2 : B = S / 2)
  (h3 : J + B + S = 160) : 
  B = 45 := 
by 
  sorry

end bill_score_l208_20893


namespace symm_y_axis_l208_20882

noncomputable def f (x : ℝ) : ℝ := abs x

theorem symm_y_axis (x : ℝ) : f (-x) = f (x) := by
  sorry

end symm_y_axis_l208_20882


namespace positive_difference_l208_20868

theorem positive_difference
  (x y : ℝ)
  (h1 : x + y = 10)
  (h2 : x^2 - y^2 = 40) : abs (x - y) = 4 :=
sorry

end positive_difference_l208_20868


namespace ratio_of_running_to_swimming_l208_20849

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

end ratio_of_running_to_swimming_l208_20849


namespace initial_y_percentage_proof_l208_20800

variable (initial_volume : ℝ) (added_volume : ℝ) (initial_percentage_x : ℝ) (result_percentage_x : ℝ)

-- Conditions
def initial_volume_condition : Prop := initial_volume = 80
def added_volume_condition : Prop := added_volume = 20
def initial_percentage_x_condition : Prop := initial_percentage_x = 0.30
def result_percentage_x_condition : Prop := result_percentage_x = 0.44

-- Question
def initial_percentage_y (initial_volume added_volume initial_percentage_x result_percentage_x : ℝ) : ℝ :=
  1 - initial_percentage_x

-- Theorem
theorem initial_y_percentage_proof 
  (h1 : initial_volume_condition initial_volume)
  (h2 : added_volume_condition added_volume)
  (h3 : initial_percentage_x_condition initial_percentage_x)
  (h4 : result_percentage_x_condition result_percentage_x) :
  initial_percentage_y initial_volume added_volume initial_percentage_x result_percentage_x = 0.70 := 
sorry

end initial_y_percentage_proof_l208_20800


namespace train_length_l208_20838
-- Import all necessary libraries from Mathlib

-- Define the given conditions and prove the target
theorem train_length (L_t L_p : ℝ) (h1 : L_t = L_p) (h2 : 54 * (1000 / 3600) * 60 = 2 * L_t) : L_t = 450 :=
by
  -- Proof goes here
  sorry

end train_length_l208_20838


namespace isha_original_length_l208_20832

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

end isha_original_length_l208_20832


namespace annual_income_from_investment_l208_20833

theorem annual_income_from_investment
  (I : ℝ) (P : ℝ) (R : ℝ)
  (hI : I = 6800) (hP : P = 136) (hR : R = 0.60) :
  (I / P) * 100 * R = 3000 := by
  sorry

end annual_income_from_investment_l208_20833


namespace jigsaw_puzzle_pieces_l208_20858

theorem jigsaw_puzzle_pieces
  (P : ℝ)
  (h1 : ∃ P, P = 0.90 * P + 0.72 * 0.10 * P + 0.504 * 0.08 * P + 504)
  (h2 : 0.504 * P = 504) :
  P = 1000 :=
by
  sorry

end jigsaw_puzzle_pieces_l208_20858


namespace point_direction_form_eq_l208_20887

-- Define the conditions
def point := (1, 2)
def direction_vector := (3, -4)

-- Define a function to represent the line equation based on point and direction
def line_equation (x y : ℝ) : Prop :=
  (x - point.1) / direction_vector.1 = (y - point.2) / direction_vector.2

-- State the theorem
theorem point_direction_form_eq (x y : ℝ) :
  (x - 1) / 3 = (y - 2) / -4 →
  line_equation x y :=
sorry

end point_direction_form_eq_l208_20887
