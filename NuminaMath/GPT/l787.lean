import Mathlib

namespace quadratic_eq_coeffs_l787_78762

theorem quadratic_eq_coeffs (x : ℝ) : 
  ∃ a b c : ℝ, 3 * x^2 + 1 - 6 * x = a * x^2 + b * x + c ∧ a = 3 ∧ b = -6 ∧ c = 1 :=
by sorry

end quadratic_eq_coeffs_l787_78762


namespace sixth_equation_l787_78748

theorem sixth_equation :
  (6 + 7 + 8 + 9 + 10 + 11 + 12 + 13 + 14 + 15 + 16 = 121) :=
by
  sorry

end sixth_equation_l787_78748


namespace Mikaela_savings_l787_78725

theorem Mikaela_savings
  (hourly_rate : ℕ)
  (first_month_hours : ℕ)
  (additional_hours_second_month : ℕ)
  (spending_fraction : ℚ)
  (earnings_first_month := hourly_rate * first_month_hours)
  (hours_second_month := first_month_hours + additional_hours_second_month)
  (earnings_second_month := hourly_rate * hours_second_month)
  (total_earnings := earnings_first_month + earnings_second_month)
  (amount_spent := spending_fraction * total_earnings)
  (amount_saved := total_earnings - amount_spent) :
  hourly_rate = 10 →
  first_month_hours = 35 →
  additional_hours_second_month = 5 →
  spending_fraction = 4 / 5 →
  amount_saved = 150 :=
by
  sorry

end Mikaela_savings_l787_78725


namespace find_y_l787_78764

theorem find_y (t : ℚ) (x y : ℚ) (h1 : x = 3 - 2 * t) (h2 : y = 3 * t + 10) (hx : x = -4) : y = 41 / 2 :=
by
  sorry

end find_y_l787_78764


namespace Marnie_can_make_9_bracelets_l787_78721

def number_of_beads : Nat :=
  (5 * 50) + (2 * 100)

def beads_per_bracelet : Nat := 50

def total_bracelets (total_beads : Nat) (beads_per_bracelet : Nat) : Nat :=
  total_beads / beads_per_bracelet

theorem Marnie_can_make_9_bracelets :
  total_bracelets number_of_beads beads_per_bracelet = 9 :=
by
  -- proof goes here
  sorry

end Marnie_can_make_9_bracelets_l787_78721


namespace quadratic_inequality_l787_78797

theorem quadratic_inequality (x : ℝ) : x^2 - x + 1 ≥ 0 :=
sorry

end quadratic_inequality_l787_78797


namespace Harkamal_purchase_grapes_l787_78740

theorem Harkamal_purchase_grapes
  (G : ℕ) -- The number of kilograms of grapes
  (cost_grapes_per_kg : ℕ := 70)
  (kg_mangoes : ℕ := 9)
  (cost_mangoes_per_kg : ℕ := 55)
  (total_paid : ℕ := 1195) :
  70 * G + 55 * 9 = 1195 → G = 10 := 
by
  sorry

end Harkamal_purchase_grapes_l787_78740


namespace intersection_of_A_and_B_l787_78759

def A : Set ℝ := {y | y > 1}
def B : Set ℝ := {x | Real.log x ≥ 0}
def Intersect : Set ℝ := {x | x > 1}

theorem intersection_of_A_and_B : A ∩ B = Intersect :=
by
  sorry

end intersection_of_A_and_B_l787_78759


namespace average_calls_per_day_l787_78743

def calls_Monday : ℕ := 35
def calls_Tuesday : ℕ := 46
def calls_Wednesday : ℕ := 27
def calls_Thursday : ℕ := 61
def calls_Friday : ℕ := 31

def total_calls : ℕ := calls_Monday + calls_Tuesday + calls_Wednesday + calls_Thursday + calls_Friday
def number_of_days : ℕ := 5

theorem average_calls_per_day : (total_calls / number_of_days) = 40 := 
by 
  -- calculations and proof steps go here.
  sorry

end average_calls_per_day_l787_78743


namespace count_perfect_cubes_between_bounds_l787_78713

theorem count_perfect_cubes_between_bounds :
  let lower_bound := 3^6 + 1
  let upper_bound := 3^12 + 1
  -- the number of perfect cubes k^3 such that 3^6 + 1 < k^3 < 3^12 + 1 inclusive is 72
  (730 < k * k * k ∧ k * k * k <= 531442 ∧ 10 <= k ∧ k <= 81 → k = 72) :=
by
  let lower_bound : ℕ := 3^6 + 1
  let upper_bound : ℕ := 3^12 + 1
  sorry

end count_perfect_cubes_between_bounds_l787_78713


namespace cos_135_eq_neg_inv_sqrt_2_l787_78782

theorem cos_135_eq_neg_inv_sqrt_2 : Real.cos (135 * Real.pi / 180) = -1 / Real.sqrt 2 := by
  sorry

end cos_135_eq_neg_inv_sqrt_2_l787_78782


namespace belt_length_sufficient_l787_78790

theorem belt_length_sufficient (r O_1O_2 O_1O_3 O_3_plane : ℝ) 
(O_1O_2_eq : O_1O_2 = 12) (O_1O_3_eq : O_1O_3 = 10) (O_3_plane_eq : O_3_plane = 8) (r_eq : r = 2) : 
(∃ L₁ L₂, L₁ = 32 + 4 * Real.pi ∧ L₂ = 22 + 2 * Real.sqrt 97 + 4 * Real.pi ∧ 
L₁ ≠ 54 ∧ L₂ > 54) := 
by 
  sorry

end belt_length_sufficient_l787_78790


namespace functional_eq_solution_l787_78795

theorem functional_eq_solution (f : ℕ → ℕ) (h : ∀ m n : ℕ, f (m + f n) = f m + n) : ∀ n, f n = n := 
by
  sorry

end functional_eq_solution_l787_78795


namespace students_making_stars_l787_78772

theorem students_making_stars (total_stars stars_per_student : ℕ) (h1 : total_stars = 372) (h2 : stars_per_student = 3) : 
  total_stars / stars_per_student = 124 :=
by
  sorry

end students_making_stars_l787_78772


namespace rational_expression_equals_3_l787_78786

theorem rational_expression_equals_3 (x : ℝ) (hx : x^3 + x - 1 = 0) :
  (x^4 - 2*x^3 + x^2 - 3*x + 5) / (x^5 - x^2 - x + 2) = 3 := 
by
  sorry

end rational_expression_equals_3_l787_78786


namespace tan_alpha_value_l787_78727

open Real

theorem tan_alpha_value
  (α : ℝ)
  (h₀ : 0 < α)
  (h₁ : α < π / 2)
  (h₂ : cos (2 * α) = (2 * sqrt 5 / 5) * sin (α + π / 4)) :
  tan α = 1 / 3 :=
sorry

end tan_alpha_value_l787_78727


namespace solve_for_n_l787_78745

theorem solve_for_n (n : ℕ) : (3^n * 3^n * 3^n * 3^n = 81^2) → n = 2 :=
by
  sorry

end solve_for_n_l787_78745


namespace sum_of_coefficients_is_225_l787_78770

theorem sum_of_coefficients_is_225 :
  let C4 := 1
  let C41 := 4
  let C42 := 6
  let C43 := 4
  (C4 + C41 + C42 + C43)^2 = 225 :=
by
  sorry

end sum_of_coefficients_is_225_l787_78770


namespace intersection_product_of_circles_l787_78755

theorem intersection_product_of_circles :
  (∀ x y : ℝ, (x^2 + 2 * x + y^2 + 4 * y + 5 = 0) ∧ (x^2 + 6 * x + y^2 + 4 * y + 9 = 0) →
  x * y = 2) :=
sorry

end intersection_product_of_circles_l787_78755


namespace evaluate_expression_l787_78761

theorem evaluate_expression (a : ℚ) (h : a = 3/2) : 
  ((5 * a^2 - 13 * a + 4) * (2 * a - 3)) = 0 := by
  sorry

end evaluate_expression_l787_78761


namespace find_line_equation_l787_78728

-- Define the first line equation
def line1 (x y : ℝ) : Prop := 2 * x - y - 5 = 0

-- Define the second line equation
def line2 (x y : ℝ) : Prop := x + y + 2 = 0

-- Define the parallel line equation with a variable constant term
def line_parallel (x y m : ℝ) : Prop := 3 * x + y + m = 0

-- State the intersection point
def intersect_point (x y : ℝ) : Prop := line1 x y ∧ line2 x y

-- The desired equation of the line passing through the intersection point
theorem find_line_equation (x y : ℝ) (h : intersect_point x y) : ∃ m, line_parallel x y m := by
  sorry

end find_line_equation_l787_78728


namespace line_passes_fixed_point_l787_78708

theorem line_passes_fixed_point (a b : ℝ) (h : a + 2 * b = 1) : 
  a * (1/2) + 3 * (-1/6) + b = 0 :=
by
  sorry

end line_passes_fixed_point_l787_78708


namespace intersection_nonempty_iff_m_lt_one_l787_78749

open Set Real

variable {m : ℝ}

theorem intersection_nonempty_iff_m_lt_one 
  (A : Set ℝ) (B : Set ℝ) (U : Set ℝ := univ) 
  (hA : A = {x | x + m >= 0}) 
  (hB : B = {x | -1 < x ∧ x < 5}) : 
  (U \ A ∩ B ≠ ∅) ↔ m < 1 := by
  sorry

end intersection_nonempty_iff_m_lt_one_l787_78749


namespace johns_disposable_income_increase_l787_78751

noncomputable def percentage_increase_of_johns_disposable_income
  (weekly_income_before : ℝ) (weekly_income_after : ℝ)
  (tax_rate_before : ℝ) (tax_rate_after : ℝ)
  (monthly_expense : ℝ) : ℝ :=
  let disposable_income_before := (weekly_income_before * (1 - tax_rate_before) * 4 - monthly_expense)
  let disposable_income_after := (weekly_income_after * (1 - tax_rate_after) * 4 - monthly_expense)
  (disposable_income_after - disposable_income_before) / disposable_income_before * 100

theorem johns_disposable_income_increase :
  percentage_increase_of_johns_disposable_income 60 70 0.15 0.18 100 = 24.62 :=
  by
  sorry

end johns_disposable_income_increase_l787_78751


namespace find_constants_for_B_l787_78799
open Matrix

def B : Matrix (Fin 3) (Fin 3) ℝ :=
  ![![0, 2, 4], ![2, 0, 2], ![4, 2, 0]]

def I3 : Matrix (Fin 3) (Fin 3) ℝ := 1

def zeros : Matrix (Fin 3) (Fin 3) ℝ := 0

theorem find_constants_for_B : 
  ∃ (s t u : ℝ), s = 0 ∧ t = -36 ∧ u = -48 ∧ (B^3 + s • B^2 + t • B + u • I3 = zeros) :=
sorry

end find_constants_for_B_l787_78799


namespace mike_age_proof_l787_78771

theorem mike_age_proof (a m : ℝ) (h1 : m = 3 * a - 20) (h2 : m + a = 70) : m = 47.5 := 
by {
  sorry
}

end mike_age_proof_l787_78771


namespace dentist_age_considered_years_ago_l787_78778

theorem dentist_age_considered_years_ago (A : ℕ) (X : ℕ) (H1 : A = 32) (H2 : (1/6 : ℚ) * (A - X) = (1/10 : ℚ) * (A + 8)) : X = 8 :=
sorry

end dentist_age_considered_years_ago_l787_78778


namespace three_digit_numbers_divisible_by_5_l787_78742

theorem three_digit_numbers_divisible_by_5 : 
  let first_term := 100
  let last_term := 995
  let common_difference := 5 
  (last_term - first_term) / common_difference + 1 = 180 :=
by
  sorry

end three_digit_numbers_divisible_by_5_l787_78742


namespace identical_digits_satisfy_l787_78730

theorem identical_digits_satisfy (n : ℕ) (hn : n ≥ 2) (x y z : ℕ) :
  (∃ (x y z : ℕ),
     (∃ (x y z : ℕ), 
         x = 3 ∧ y = 2 ∧ z = 1) ∨
     (∃ (x y z : ℕ), 
         x = 6 ∧ y = 8 ∧ z = 4) ∨
     (∃ (x y z : ℕ), 
         x = 8 ∧ y = 3 ∧ z = 7)) :=
by sorry

end identical_digits_satisfy_l787_78730


namespace correctLikeTermsPair_l787_78733

def areLikeTerms (term1 term2 : String) : Bool :=
  -- Define the criteria for like terms (variables and their respective powers)
  sorry

def pairA : (String × String) := ("-2x^3", "-2x")
def pairB : (String × String) := ("-1/2ab", "18ba")
def pairC : (String × String) := ("x^2y", "-xy^2")
def pairD : (String × String) := ("4m", "4mn")

theorem correctLikeTermsPair :
  areLikeTerms pairA.1 pairA.2 = false ∧
  areLikeTerms pairB.1 pairB.2 = true ∧
  areLikeTerms pairC.1 pairC.2 = false ∧
  areLikeTerms pairD.1 pairD.2 = false :=
sorry

end correctLikeTermsPair_l787_78733


namespace increase_in_area_l787_78746

theorem increase_in_area :
  let original_side := 6
  let increase := 1
  let new_side := original_side + increase
  let original_area := original_side * original_side
  let new_area := new_side * new_side
  let area_increase := new_area - original_area
  area_increase = 13 :=
by
  let original_side := 6
  let increase := 1
  let new_side := original_side + increase
  let original_area := original_side * original_side
  let new_area := new_side * new_side
  let area_increase := new_area - original_area
  sorry

end increase_in_area_l787_78746


namespace present_age_of_R_l787_78776

variables (P_p Q_p R_p : ℝ)

-- Conditions from the problem
axiom h1 : P_p - 8 = 1/2 * (Q_p - 8)
axiom h2 : Q_p - 8 = 2/3 * (R_p - 8)
axiom h3 : Q_p = 2 * Real.sqrt R_p
axiom h4 : P_p = 3/5 * Q_p

theorem present_age_of_R : R_p = 400 :=
by
  sorry

end present_age_of_R_l787_78776


namespace range_of_a_l787_78701

theorem range_of_a (x : ℝ) (h : 1 < x) : ∀ a, (∀ x, 1 < x → x + 1 / (x - 1) ≥ a) → a ≤ 3 :=
by
sorry

end range_of_a_l787_78701


namespace Kate_has_223_pennies_l787_78788

-- Definition of the conditions
variables (J K : ℕ)
variable (h1 : J = 388)
variable (h2 : J = K + 165)

-- Prove the question equals the answer
theorem Kate_has_223_pennies : K = 223 :=
by
  sorry

end Kate_has_223_pennies_l787_78788


namespace triangle_area_l787_78719

theorem triangle_area (a b c p : ℕ) (h_ratio : a = 5 * p) (h_ratio2 : b = 12 * p) (h_ratio3 : c = 13 * p) (h_perimeter : a + b + c = 300) : 
  (1 / 4) * Real.sqrt ((a + b + c) * (a + b - c) * (a + c - b) * (b + c - a)) = 3000 := 
by 
  sorry

end triangle_area_l787_78719


namespace alice_meeting_distance_l787_78766

noncomputable def distanceAliceWalks (t : ℝ) : ℝ :=
  6 * t

theorem alice_meeting_distance :
  ∃ t : ℝ, 
    distanceAliceWalks t = 
      (900 * Real.sqrt 2 - Real.sqrt 630000) / 11 ∧
    (5 * t) ^ 2 =
      (6 * t) ^ 2 + 150 ^ 2 - 2 * 6 * t * 150 * Real.cos (Real.pi / 4) :=
sorry

end alice_meeting_distance_l787_78766


namespace relationship_among_abc_l787_78767

noncomputable def a : ℝ := Real.log (1/4) / Real.log 2
noncomputable def b : ℝ := 2.1^(1/3)
noncomputable def c : ℝ := (4/5)^2

theorem relationship_among_abc : a < c ∧ c < b :=
by
  -- Definitions
  have ha : a = Real.log (1/4) / Real.log 2 := rfl
  have hb : b = 2.1^(1/3) := rfl
  have hc : c = (4/5)^2 := rfl
  sorry

end relationship_among_abc_l787_78767


namespace male_students_count_l787_78704

theorem male_students_count
  (average_all_students : ℕ → ℕ → ℚ → Prop)
  (average_male_students : ℕ → ℚ → Prop)
  (average_female_students : ℕ → ℚ → Prop)
  (F : ℕ)
  (total_average : average_all_students (F + M) (83 * M + 92 * F) 90)
  (male_average : average_male_students M 83)
  (female_average : average_female_students 28 92) :
  ∃ (M : ℕ), M = 8 :=
by {
  sorry
}

end male_students_count_l787_78704


namespace orange_balls_count_l787_78731

theorem orange_balls_count :
  ∀ (total red blue orange pink : ℕ), 
  total = 50 → red = 20 → blue = 10 → 
  total = red + blue + orange + pink → 3 * orange = pink → 
  orange = 5 :=
by
  intros total red blue orange pink h_total h_red h_blue h_total_eq h_ratio
  sorry

end orange_balls_count_l787_78731


namespace evaluate_expr_l787_78792

noncomputable def expr : ℚ :=
  2013 * (5.7 * 4.2 + (21 / 5) * 4.3) / ((14 / 73) * 15 + (5 / 73) * 177 + 656)

theorem evaluate_expr : expr = 126 := by
  sorry

end evaluate_expr_l787_78792


namespace sequence_general_term_l787_78769

theorem sequence_general_term (a : ℕ → ℕ) 
  (h₀ : a 1 = 4) 
  (h₁ : ∀ n : ℕ, a (n + 1) = 2 * a n + n^2) : 
  ∀ n : ℕ, a n = 5 * 2^n - n^2 - 2*n - 3 :=
by
  sorry

end sequence_general_term_l787_78769


namespace gcd_90_450_l787_78787

theorem gcd_90_450 : Nat.gcd 90 450 = 90 := by
  sorry

end gcd_90_450_l787_78787


namespace partition_subset_sum_l787_78753

variable {p k : ℕ}

def V_p (p : ℕ) := {k : ℕ | p ∣ (k * (k + 1) / 2) ∧ k ≥ 2 * p - 1}

theorem partition_subset_sum (p : ℕ) (hp : Nat.Prime p) (k : ℕ) : k ∈ V_p p := sorry

end partition_subset_sum_l787_78753


namespace extreme_point_l787_78737

noncomputable def f (x : ℝ) : ℝ := (x^4 / 4) - (x^3 / 3)
noncomputable def f_prime (x : ℝ) : ℝ := deriv f x

theorem extreme_point (x : ℝ) : f_prime 1 = 0 ∧
  (∀ y, y < 1 → f_prime y < 0) ∧
  (∀ z, z > 1 → f_prime z > 0) :=
by
  sorry

end extreme_point_l787_78737


namespace remainder_845307_div_6_l787_78709

theorem remainder_845307_div_6 :
  let n := 845307
  ∃ r : ℕ, n % 6 = r ∧ r = 3 :=
by
  let n := 845307
  have h_div_2 : ¬(n % 2 = 0) := by sorry
  have h_div_3 : n % 3 = 0 := by sorry
  exact ⟨3, by sorry, rfl⟩

end remainder_845307_div_6_l787_78709


namespace below_sea_level_notation_l787_78796

theorem below_sea_level_notation (depth_above_sea_level : Int) (depth_below_sea_level: Int) 
  (h: depth_above_sea_level = 9050 ∧ depth_below_sea_level = -10907) : 
  depth_below_sea_level = -10907 :=
by 
  sorry

end below_sea_level_notation_l787_78796


namespace solve_for_x_l787_78714

theorem solve_for_x (x : ℝ) (h : 3^(3 * x - 2) = (1 : ℝ) / 27) : x = -(1 : ℝ) / 3 :=
sorry

end solve_for_x_l787_78714


namespace union_A_B_range_of_a_l787_78706

-- Definitions of sets A, B, and C
def A : Set ℝ := { x | 3 ≤ x ∧ x ≤ 9 }
def B : Set ℝ := { x | 2 < x ∧ x < 5 }
def C (a : ℝ) : Set ℝ := { x | x > a }

-- Problem 1: Proving A ∪ B = { x | 2 < x ≤ 9 }
theorem union_A_B : A ∪ B = { x | 2 < x ∧ x ≤ 9 } :=
sorry

-- Problem 2: Proving the range of 'a' given B ∩ C = ∅
theorem range_of_a (a : ℝ) (h : B ∩ C a = ∅) : a ≥ 5 :=
sorry

end union_A_B_range_of_a_l787_78706


namespace cost_of_each_soda_l787_78700

theorem cost_of_each_soda (total_cost sandwiches_cost : ℝ) (number_of_sodas : ℕ)
  (h_total_cost : total_cost = 6.46)
  (h_sandwiches_cost : sandwiches_cost = 2 * 1.49) :
  total_cost - sandwiches_cost = 4 * 0.87 := by
  sorry

end cost_of_each_soda_l787_78700


namespace no_positive_integers_satisfy_l787_78705

theorem no_positive_integers_satisfy (a b : ℕ) (ha : a > 0) (hb : b > 0) : 
  ¬ (3 * a^2 = b^2 + 1) := 
sorry

end no_positive_integers_satisfy_l787_78705


namespace smallest_log_log_x0_l787_78711

noncomputable def f (x : ℝ) : ℝ := Real.log x - 6 + 2 * x

theorem smallest_log_log_x0 (x₀ : ℝ) (h₀ : f x₀ = 0) (h_dom : 2 < x₀ ∧ x₀ < Real.exp 1) :
  min (min (Real.log x₀) (Real.log (Real.sqrt x₀))) (min (Real.log (Real.log x₀)) ((Real.log x₀)^2)) = Real.log (Real.log x₀) :=
sorry

end smallest_log_log_x0_l787_78711


namespace expression_equals_a5_l787_78752

theorem expression_equals_a5 (a : ℝ) : a^4 * a = a^5 := 
by sorry

end expression_equals_a5_l787_78752


namespace rectangle_area_proof_l787_78729

variable (x y : ℕ) -- Declaring the variables to represent length and width of the rectangle.

-- Declaring the conditions as hypotheses.
def condition1 := (x + 3) * (y - 1) = x * y
def condition2 := (x - 3) * (y + 2) = x * y
def condition3 := (x + 4) * (y - 2) = x * y

-- The theorem to prove the area is 36 given the above conditions.
theorem rectangle_area_proof (h1 : condition1 x y) (h2 : condition2 x y) (h3 : condition3 x y) : x * y = 36 :=
by
  sorry

end rectangle_area_proof_l787_78729


namespace trebled_resultant_l787_78724

theorem trebled_resultant (n : ℕ) (h : n = 20) : 3 * ((2 * n) + 5) = 135 := 
by
  sorry

end trebled_resultant_l787_78724


namespace mod_inverse_9_mod_23_l787_78794

theorem mod_inverse_9_mod_23 : ∃ (a : ℤ), 0 ≤ a ∧ a < 23 ∧ (9 * a) % 23 = 1 :=
by
  use 18
  sorry

end mod_inverse_9_mod_23_l787_78794


namespace flat_fee_rate_l787_78715

-- Definitions for the variables
variable (F n : ℝ)

-- Conditions based on the problem statement
axiom mark_cost : F + 4.6 * n = 310
axiom lucy_cost : F + 6.2 * n = 410

-- Problem Statement
theorem flat_fee_rate : F = 22.5 ∧ n = 62.5 :=
by
  sorry

end flat_fee_rate_l787_78715


namespace circle_circumference_ratio_l787_78702

theorem circle_circumference_ratio (A₁ A₂ : ℝ) (h : A₁ / A₂ = 16 / 25) :
  ∃ C₁ C₂ : ℝ, (C₁ / C₂ = 4 / 5) :=
by
  -- Definitions and calculations to be done here
  sorry

end circle_circumference_ratio_l787_78702


namespace infinitely_many_n_l787_78780

theorem infinitely_many_n (h : ℤ) : ∃ (S : Set ℤ), S ≠ ∅ ∧ ∀ n ∈ S, ∃ k : ℕ, ⌊n * Real.sqrt (h^2 + 1)⌋ = k^2 :=
by
  sorry

end infinitely_many_n_l787_78780


namespace min_sum_of_a_and_b_l787_78765

theorem min_sum_of_a_and_b (a b : ℕ) (h1 : 0 < a) (h2 : 0 < b) (h3 : a > 4 * b) : a + b ≥ 6 :=
by
  sorry

end min_sum_of_a_and_b_l787_78765


namespace cost_of_each_box_is_8_33_l787_78785

noncomputable def cost_per_box (boxes pens_per_box pens_packaged price_per_packaged price_per_set profit_total : ℕ) : ℝ :=
  let total_pens := boxes * pens_per_box
  let packaged_pens := pens_packaged * pens_per_box
  let packages := packaged_pens / 6
  let revenue_packages := packages * price_per_packaged
  let remaining_pens := total_pens - packaged_pens
  let sets := remaining_pens / 3
  let revenue_sets := sets * price_per_set
  let total_revenue := revenue_packages + revenue_sets
  let cost_total := total_revenue - profit_total
  cost_total / boxes

theorem cost_of_each_box_is_8_33 :
  cost_per_box 12 30 5 3 2 115 = 100 / 12 :=
by
  unfold cost_per_box
  sorry

end cost_of_each_box_is_8_33_l787_78785


namespace inscribed_circle_theta_l787_78798

/-- Given that a circle inscribed in triangle ABC is tangent to sides BC, CA, and AB at points
    where the tangential angles are 120 degrees, 130 degrees, and theta degrees respectively,
    we need to prove that theta is 110 degrees. -/
theorem inscribed_circle_theta 
  (ABC : Type)
  (A B C : ABC)
  (theta : ℝ)
  (tangent_angle_BC : ℝ)
  (tangent_angle_CA : ℝ) 
  (tangent_angle_AB : ℝ) 
  (h1 : tangent_angle_BC = 120)
  (h2 : tangent_angle_CA = 130) 
  (h3 : tangent_angle_AB = theta) : 
  theta = 110 :=
by
  sorry

end inscribed_circle_theta_l787_78798


namespace problem_solution_l787_78775

theorem problem_solution (a b : ℝ) (h1 : 2 + 3 = -b) (h2 : 2 * 3 = -2 * a) : a + b = -8 :=
by
  sorry

end problem_solution_l787_78775


namespace reciprocal_of_neg_2023_l787_78777

theorem reciprocal_of_neg_2023 : 1 / (-2023) = -1 / 2023 :=
by
  sorry

end reciprocal_of_neg_2023_l787_78777


namespace solve_percentage_increase_length_l787_78754

def original_length (L : ℝ) : Prop := true
def original_breadth (B : ℝ) : Prop := true

def new_breadth (B' : ℝ) (B : ℝ) : Prop := B' = 1.25 * B

def new_length (L' : ℝ) (L : ℝ) (x : ℝ) : Prop := L' = L * (1 + x / 100)

def original_area (L : ℝ) (B : ℝ) (A : ℝ) : Prop := A = L * B

def new_area (A' : ℝ) (A : ℝ) : Prop := A' = 1.375 * A

def percentage_increase_length (x : ℝ) : Prop := x = 10

theorem solve_percentage_increase_length (L B A A' L' B' x : ℝ)
  (hL : original_length L)
  (hB : original_breadth B)
  (hB' : new_breadth B' B)
  (hL' : new_length L' L x)
  (hA : original_area L B A)
  (hA' : new_area A' A)
  (h_eqn : L' * B' = A') :
  percentage_increase_length x :=
by
  sorry

end solve_percentage_increase_length_l787_78754


namespace patrons_per_golf_cart_l787_78732

theorem patrons_per_golf_cart (patrons_from_cars patrons_from_bus golf_carts total_patrons patrons_per_cart : ℕ) 
  (h1 : patrons_from_cars = 12)
  (h2 : patrons_from_bus = 27)
  (h3 : golf_carts = 13)
  (h4 : total_patrons = patrons_from_cars + patrons_from_bus)
  (h5 : patrons_per_cart = total_patrons / golf_carts) : 
  patrons_per_cart = 3 := 
by
  sorry

end patrons_per_golf_cart_l787_78732


namespace value_of_a_l787_78738

theorem value_of_a (a : ℝ) (h : (2 : ℝ)^a = (1 / 2 : ℝ)) : a = -1 := 
sorry

end value_of_a_l787_78738


namespace solve_equation_l787_78718

theorem solve_equation : ∀ x : ℝ, (2 * x - 1)^2 - (1 - 3 * x)^2 = 5 * (1 - x) * (x + 1) → x = 5 / 2 :=
by
  intro x
  intro h
  sorry

end solve_equation_l787_78718


namespace midpoint_of_interception_l787_78744

theorem midpoint_of_interception (x1 x2 y1 y2 : ℝ) 
  (h1 : y1^2 = 4 * x1) 
  (h2 : y2^2 = 4 * x2) 
  (h3 : y1 = x1 - 1) 
  (h4 : y2 = x2 - 1) : 
  ( (x1 + x2) / 2, (y1 + y2) / 2 ) = (3, 2) :=
by 
  sorry

end midpoint_of_interception_l787_78744


namespace find_magical_points_on_specific_quad_find_t_for_unique_magical_point_l787_78756

-- Define what it means to be a "magical point"
def is_magical_point (m : ℝ) (A : ℝ × ℝ) : Prop :=
  A = (m, 2 * m)

-- Specialize for the specific quadratic function y = x^2 - x - 4
def on_specific_quadratic (m : ℝ) (A : ℝ × ℝ) : Prop :=
  A = (m, m^2 - m - 4)

-- Theorem for part 1: Find the magical points on y = x^2 - x - 4
theorem find_magical_points_on_specific_quad (m : ℝ) (A : ℝ × ℝ) :
  is_magical_point m A ∧ on_specific_quadratic m A →
  (A = (4, 8) ∨ A = (-1, -2)) :=
sorry

-- Define the quadratic function for part 2
def on_general_quadratic (t m : ℝ) (A : ℝ × ℝ) : Prop :=
  A = (m, t * m^2 + (t-2) * m - 4)

-- Theorem for part 2: Find the t values for unique magical points
theorem find_t_for_unique_magical_point (t m : ℝ) (A : ℝ × ℝ) :
  ( ∀ m, is_magical_point m A ∧ on_general_quadratic t m A → 
    (t * m^2 + (t-4) * m - 4 = 0) ) → 
  ( ∃! m, is_magical_point m A ∧ on_general_quadratic t m A ) →
  t = -4 :=
sorry

end find_magical_points_on_specific_quad_find_t_for_unique_magical_point_l787_78756


namespace purely_imaginary_m_complex_division_a_plus_b_l787_78712

-- Problem 1: Prove that m=-2 for z to be purely imaginary
theorem purely_imaginary_m (m : ℝ) (h : ∀ z : ℂ, z = (m - 1) * (m + 2) + (m - 1) * I → z.im = z.im) : m = -2 :=
sorry

-- Problem 2: Prove a+b = 13/10 with given conditions
theorem complex_division_a_plus_b (a b : ℝ) (m : ℝ) (h_m : m = 2) 
  (h_z : z = 4 + I) (h_eq : (z + I) / (z - I) = a + b * I) : a + b = 13 / 10 :=
sorry

end purely_imaginary_m_complex_division_a_plus_b_l787_78712


namespace annual_salary_is_20_l787_78734

-- Define the conditions
variable (months_worked : ℝ) (total_received : ℝ) (turban_price : ℝ)
variable (S : ℝ)

-- Actual values from the problem
axiom h1 : months_worked = 9 / 12
axiom h2 : total_received = 55
axiom h3 : turban_price = 50

-- Define the statement to prove
theorem annual_salary_is_20 : S = 20 := by
  -- Conditions derived from the problem
  have cash_received := total_received - turban_price
  have fraction_of_salary := months_worked * S
  -- Given the servant worked 9 months and received Rs. 55 including Rs. 50 turban
  have : cash_received = fraction_of_salary := by sorry
  -- Solving the equation 3/4 S = 5 for S
  have : S = 20 := by sorry
  sorry -- Final proof step

end annual_salary_is_20_l787_78734


namespace solution_set_of_inequality_l787_78717

theorem solution_set_of_inequality :
  {x : ℝ | |x^2 - 2| < 2} = {x : ℝ | (x > -2 ∧ x < 0) ∨ (x > 0 ∧ x < 2)} :=
by
  sorry

end solution_set_of_inequality_l787_78717


namespace water_added_l787_78791

theorem water_added (W : ℝ) : 
  (15 + W) * 0.20833333333333336 = 3.75 → W = 3 :=
by
  intro h
  sorry

end water_added_l787_78791


namespace derivative_of_f_l787_78773

-- Define the function
def f (x : ℝ) : ℝ := x^2 - x

-- State the theorem to prove
theorem derivative_of_f : ∀ x : ℝ,  (deriv f x = 2 * x - 1) :=
by sorry

end derivative_of_f_l787_78773


namespace certain_number_is_310_l787_78789

theorem certain_number_is_310 (x : ℤ) (h : 3005 - x + 10 = 2705) : x = 310 :=
by
  sorry

end certain_number_is_310_l787_78789


namespace supplement_of_complementary_angle_l787_78741

theorem supplement_of_complementary_angle (α β : ℝ) 
  (h1 : α + β = 90) (h2 : α = 30) : 180 - β = 120 :=
by sorry

end supplement_of_complementary_angle_l787_78741


namespace intersection_point_of_lines_l787_78781

theorem intersection_point_of_lines : 
  (∃ x y : ℚ, (8 * x - 3 * y = 5) ∧ (5 * x + 2 * y = 20)) ↔ (x = 70 / 31 ∧ y = 135 / 31) :=
sorry

end intersection_point_of_lines_l787_78781


namespace line_through_point_intersects_yaxis_triangular_area_l787_78783

theorem line_through_point_intersects_yaxis_triangular_area 
  (a T : ℝ) 
  (h : 0 < a) 
  (line_eqn : ∀ x y : ℝ, x = -a * y + a → 2 * T * x + a^2 * y - 2 * a * T = 0) 
  : ∃ (m b : ℝ), (forall x y : ℝ, y = m * x + b) := 
by
  sorry

end line_through_point_intersects_yaxis_triangular_area_l787_78783


namespace chime_date_is_march_22_2003_l787_78707

-- Definitions
def clock_chime (n : ℕ) : ℕ := n % 12

def half_hour_chimes (half_hours : ℕ) : ℕ := half_hours
def hourly_chimes (hours : List ℕ) : ℕ := hours.map clock_chime |>.sum

-- Problem conditions and result
def initial_chimes_and_half_hours : ℕ := half_hour_chimes 9
def initial_hourly_chimes : ℕ := hourly_chimes [4, 5, 6, 7, 8, 9, 10, 11, 0]
def chimes_on_february_28_2003 : ℕ := initial_chimes_and_half_hours + initial_hourly_chimes

def half_hour_chimes_per_day : ℕ := half_hour_chimes 24
def hourly_chimes_per_day : ℕ := hourly_chimes (List.range 12 ++ List.range 12)
def total_chimes_per_day : ℕ := half_hour_chimes_per_day + hourly_chimes_per_day

def remaining_chimes_needed : ℕ := 2003 - chimes_on_february_28_2003
def full_days_needed : ℕ := remaining_chimes_needed / total_chimes_per_day
def additional_chimes_needed : ℕ := remaining_chimes_needed % total_chimes_per_day

-- Lean theorem statement
theorem chime_date_is_march_22_2003 :
    (full_days_needed = 21) → (additional_chimes_needed < total_chimes_per_day) → 
    true :=
by
  sorry

end chime_date_is_march_22_2003_l787_78707


namespace num_circles_rectangle_l787_78779

structure Rectangle (α : Type*) [Field α] :=
  (A B C D : α × α)
  (AB_parallel_CD : B.1 = A.1 ∧ D.1 = C.1)
  (AD_parallel_BC : D.2 = A.2 ∧ C.2 = B.2)

def num_circles_with_diameter_vertices (R : Rectangle ℝ) : ℕ :=
  sorry

theorem num_circles_rectangle (R : Rectangle ℝ) : num_circles_with_diameter_vertices R = 5 :=
  sorry

end num_circles_rectangle_l787_78779


namespace correct_div_value_l787_78793

theorem correct_div_value (x : ℝ) (h : 25 * x = 812) : x / 4 = 8.12 :=
by sorry

end correct_div_value_l787_78793


namespace distribution_difference_l787_78739

theorem distribution_difference 
  (total_amnt : ℕ)
  (p_amnt : ℕ) 
  (q_amnt : ℕ) 
  (r_amnt : ℕ)
  (s_amnt : ℕ)
  (h_total : total_amnt = 1000)
  (h_p : p_amnt = 2 * q_amnt)
  (h_s : s_amnt = 4 * r_amnt)
  (h_qr : q_amnt = r_amnt) :
  s_amnt - p_amnt = 250 := 
sorry

end distribution_difference_l787_78739


namespace solve_686_l787_78726

theorem solve_686 : ∃ (x y z : ℕ), x > 0 ∧ y > 0 ∧ z > 0 ∧ x^2 + y^2 + z^2 = 686 := 
by
  sorry

end solve_686_l787_78726


namespace trapezoid_area_l787_78774

theorem trapezoid_area :
  ∃ S, (S = 6 ∨ S = 10) ∧ 
  ((∃ (a b c d : ℝ), a = 1 ∧ b = 4 ∧ c = 4 ∧ d = 5 ∧ 
    (∃ (is_isosceles_trapezoid : Prop), is_isosceles_trapezoid)) ∨
   (∃ (a b c d : ℝ), a = 1 ∧ b = 4 ∧ c = 5 ∧ d = 4 ∧ 
    (∃ (is_right_angled_trapezoid : Prop), is_right_angled_trapezoid)) ∨ 
   (∃ (a b c d : ℝ), (a = 1 ∨ b = 1 ∨ c = 1 ∨ d = 1) →
   (∀ (is_impossible_trapezoid : Prop), ¬ is_impossible_trapezoid))) :=
sorry

end trapezoid_area_l787_78774


namespace number_of_shirts_proof_l787_78757

def regular_price := 50
def discount_percentage := 20
def total_paid := 240

def sale_price (rp : ℕ) (dp : ℕ) : ℕ := rp * (100 - dp) / 100

def number_of_shirts (tp : ℕ) (sp : ℕ) : ℕ := tp / sp

theorem number_of_shirts_proof : 
  number_of_shirts total_paid (sale_price regular_price discount_percentage) = 6 :=
by 
  sorry

end number_of_shirts_proof_l787_78757


namespace regular_polygon_sides_and_interior_angle_l787_78758

theorem regular_polygon_sides_and_interior_angle (n : ℕ) (H : (n - 2) * 180 = 3 * 360 + 180) :
  n = 9 ∧ (n - 2) * 180 / n = 140 :=
by
-- This marks the start of the proof, but the proof is omitted.
sorry

end regular_polygon_sides_and_interior_angle_l787_78758


namespace sum_of_two_squares_l787_78735

theorem sum_of_two_squares (n : ℕ) (h : ∀ m, m = n → n = 2 ∨ (n = 2 * 10 + m) → n % 8 = m) :
  (∃ a b : ℕ, n = a^2 + b^2) ↔ n = 2 := by
  sorry

end sum_of_two_squares_l787_78735


namespace kendra_minivans_l787_78723

theorem kendra_minivans (afternoon: ℕ) (evening: ℕ) (h1: afternoon = 4) (h2: evening = 1) : afternoon + evening = 5 :=
by sorry

end kendra_minivans_l787_78723


namespace next_perfect_square_l787_78716

theorem next_perfect_square (n : ℤ) (hn : Even n) (x : ℤ) (hx : x = n^2) : 
  ∃ y : ℤ, y = x + 2 * n + 1 ∧ (∃ m : ℤ, y = m^2) ∧ m > n :=
by
  sorry

end next_perfect_square_l787_78716


namespace scientific_notation_correct_l787_78747

theorem scientific_notation_correct :
  52000000 = 5.2 * 10^7 :=
sorry

end scientific_notation_correct_l787_78747


namespace jack_sugar_final_l787_78760

-- Conditions
def initial_sugar := 65
def sugar_used := 18
def sugar_bought := 50

-- Question and proof goal
theorem jack_sugar_final : initial_sugar - sugar_used + sugar_bought = 97 := by
  sorry

end jack_sugar_final_l787_78760


namespace quadratic_residues_count_l787_78720

theorem quadratic_residues_count (p : ℕ) (hp : Nat.Prime p) (hp2 : p % 2 = 1) :
  ∃ (q_residues : Finset (ZMod p)), q_residues.card = (p - 1) / 2 ∧
  ∃ (nq_residues : Finset (ZMod p)), nq_residues.card = (p - 1) / 2 ∧
  ∀ d ∈ q_residues, ∃ x y : ZMod p, x^2 = d ∧ y^2 = d ∧ x ≠ y :=
by
  sorry

end quadratic_residues_count_l787_78720


namespace probability_area_l787_78703

noncomputable def probability_x_y_le_five (x y : ℝ) : ℚ :=
  if 0 ≤ x ∧ x ≤ 4 ∧ 0 ≤ y ∧ y ≤ 8 ∧ x + y ≤ 5 then 1 else 0

theorem probability_area {P : ℚ} :
  (∀ (x y : ℝ), 0 ≤ x ∧ x ≤ 4 ∧ 0 ≤ y ∧ y ≤ 8 → P = probability_x_y_le_five x y / (4 * 8)) →
  P = 5 / 16 :=
by
  sorry

end probability_area_l787_78703


namespace cyclist_speed_ratio_l787_78750

-- Define the conditions
def speeds_towards_each_other (v1 v2 : ℚ) : Prop :=
  v1 + v2 = 25

def speeds_apart_with_offset (v1 v2 : ℚ) : Prop :=
  v1 - v2 = 10 / 3

-- The proof problem to show the required ratio of speeds
theorem cyclist_speed_ratio (v1 v2 : ℚ) (h1 : speeds_towards_each_other v1 v2) (h2 : speeds_apart_with_offset v1 v2) :
  v1 / v2 = 17 / 13 :=
sorry

end cyclist_speed_ratio_l787_78750


namespace bottle_caps_sum_l787_78736

theorem bottle_caps_sum : 
  let starting_caps := 91
  let found_caps := 88
  starting_caps + found_caps = 179 :=
by
  sorry

end bottle_caps_sum_l787_78736


namespace arithmetic_sequence_sum_l787_78722

theorem arithmetic_sequence_sum (a : ℕ → ℕ) (S : ℕ → ℕ) (d : ℤ) :
  (∀ n, a n = a 1 + (n - 1) * d) → 
  (∀ n, S n = n * (a 1 + a n) / 2) → 
  (a 3 + 4 = a 2 + a 7) → 
  S 11 = 44 :=
by 
  sorry

end arithmetic_sequence_sum_l787_78722


namespace outfits_without_matching_color_l787_78768

theorem outfits_without_matching_color (red_shirts green_shirts pairs_pants green_hats red_hats : ℕ) 
  (h_red_shirts : red_shirts = 5) 
  (h_green_shirts : green_shirts = 5) 
  (h_pairs_pants : pairs_pants = 6) 
  (h_green_hats : green_hats = 8) 
  (h_red_hats : red_hats = 8) : 
  (red_shirts * pairs_pants * green_hats) + (green_shirts * pairs_pants * red_hats) = 480 := 
by 
  sorry

end outfits_without_matching_color_l787_78768


namespace arithmetic_sequence_inequality_l787_78710

noncomputable def arithmetic_seq (a d : ℕ) (n : ℕ) : ℕ := a + (n - 1) * d

noncomputable def sum_arithmetic_seq (a d : ℕ) (n : ℕ) : ℕ :=
  n * (2 * a + (n - 1) * d) / 2

theorem arithmetic_sequence_inequality
  (a d : ℕ)
  (i j k l : ℕ)
  (hi : i ≤ j)
  (hj : j ≤ k)
  (hk : k ≤ l)
  (hij: i + l = j + k)
  : (arithmetic_seq a d i) * (arithmetic_seq a d l) ≤ (arithmetic_seq a d j) * (arithmetic_seq a d k) :=
sorry

end arithmetic_sequence_inequality_l787_78710


namespace number_divisible_by_7_last_digits_l787_78763

theorem number_divisible_by_7_last_digits :
  ∀ d : ℕ, d ≤ 9 → ∃ n : ℕ, n % 7 = 0 ∧ n % 10 = d :=
by
  sorry

end number_divisible_by_7_last_digits_l787_78763


namespace gain_percent_l787_78784

theorem gain_percent (CP SP : ℝ) (hCP : CP = 100) (hSP : SP = 115) : 
  ((SP - CP) / CP) * 100 = 15 := 
by 
  sorry

end gain_percent_l787_78784
