import Mathlib

namespace number_of_subsets_l274_274088

-- Defining the type of the elements
variable {α : Type*}

-- Statement of the problem in Lean 4
theorem number_of_subsets (s : Finset α) (h : s.card = n) : (Finset.powerset s).card = 2^n := 
sorry

end number_of_subsets_l274_274088


namespace recent_quarter_revenue_l274_274246

theorem recent_quarter_revenue :
  let revenue_year_ago : Float := 69.0
  let percentage_decrease : Float := 30.434782608695656
  let decrease_in_revenue : Float := revenue_year_ago * (percentage_decrease / 100)
  let recent_quarter_revenue := revenue_year_ago - decrease_in_revenue
  recent_quarter_revenue = 48.0 := by
  sorry

end recent_quarter_revenue_l274_274246


namespace wicket_keeper_age_difference_l274_274777

def cricket_team_average_age : Nat := 24
def total_members : Nat := 11
def remaining_members : Nat := 9
def age_difference : Nat := 1

theorem wicket_keeper_age_difference :
  let total_age := cricket_team_average_age * total_members
  let remaining_average_age := cricket_team_average_age - age_difference
  let remaining_total_age := remaining_average_age * remaining_members
  let combined_age := total_age - remaining_total_age
  let average_age := cricket_team_average_age
  let wicket_keeper_age := combined_age - average_age
  wicket_keeper_age - average_age = 9 := 
by
  sorry

end wicket_keeper_age_difference_l274_274777


namespace plumber_charge_shower_l274_274546

theorem plumber_charge_shower (S : ℝ) 
  (sink_cost : ℝ := 30) 
  (toilet_cost : ℝ := 50)
  (max_earning : ℝ := 250)
  (first_job_toilets : ℝ := 3) (first_job_sinks : ℝ := 3)
  (second_job_toilets : ℝ := 2) (second_job_sinks : ℝ := 5)
  (third_job_toilets : ℝ := 1) (third_job_showers : ℝ := 2) (third_job_sinks : ℝ := 3) :
  2 * S + 1 * toilet_cost + 3 * sink_cost ≤ max_earning → S ≤ 55 :=
by
  sorry

end plumber_charge_shower_l274_274546


namespace message_hours_needed_l274_274684

-- Define the sequence and the condition
def S (n : ℕ) : ℕ := 2^(n + 1) - 2

theorem message_hours_needed : ∃ n : ℕ, S n > 55 ∧ n = 5 := by
  sorry

end message_hours_needed_l274_274684


namespace inequality_solution_l274_274110

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a + 2 / (2^x + 1)

lemma monotone_decreasing (a : ℝ) : ∀ x y : ℝ, x < y → f a y < f a x := 
sorry

lemma odd_function (a : ℝ) (h : ∀ x : ℝ, f a (-x) = -f a x) : a = 0 := 
sorry

theorem inequality_solution (t : ℝ) (a : ℝ) (h_monotone : ∀ x y : ℝ, x < y → f a y < f a x)
    (h_odd : ∀ x : ℝ, f a (-x) = -f a x) : t ≥ 4 / 3 ↔ f a (2 * t + 1) + f a (t - 5) ≤ 0 := 
sorry

end inequality_solution_l274_274110


namespace min_quadratic_expression_value_l274_274192

def quadratic_expression (x : ℝ) : ℝ := 3 * x^2 - 18 * x + 2205

theorem min_quadratic_expression_value : 
  ∃ x : ℝ, quadratic_expression x = 2178 :=
sorry

end min_quadratic_expression_value_l274_274192


namespace round_robin_10_players_l274_274237

theorem round_robin_10_players : @Nat.choose 10 2 = 45 := by
  sorry

end round_robin_10_players_l274_274237


namespace slope_of_line_l274_274272

theorem slope_of_line (x y : ℝ) :
  (∀ (x y : ℝ), (x / 4 + y / 5 = 1) → (∃ m b : ℝ, y = m * x + b ∧ m = -5 / 4)) :=
by
  sorry

end slope_of_line_l274_274272


namespace series_remainder_is_zero_l274_274425

theorem series_remainder_is_zero :
  let a : ℕ := 4
  let d : ℕ := 6
  let n : ℕ := 17
  let l : ℕ := a + d * (n - 1) -- last term
  let S : ℕ := n * (a + l) / 2 -- sum of the series
  S % 17 = 0 := by
  sorry

end series_remainder_is_zero_l274_274425


namespace exactly_one_correct_l274_274468

theorem exactly_one_correct (P_A P_B : ℚ) (hA : P_A = 1/5) (hB : P_B = 1/4) :
  P_A * (1 - P_B) + (1 - P_A) * P_B = 7/20 :=
by
  sorry

end exactly_one_correct_l274_274468


namespace parallel_lines_slope_l274_274586

theorem parallel_lines_slope (m : ℝ) 
  (h1 : ∀ x y : ℝ, x + 2 * y - 1 = 0 → x = -2 * y + 1)
  (h2 : ∀ x y : ℝ, m * x - y = 0 → y = m * x) : 
  m = -1 / 2 :=
by
  sorry

end parallel_lines_slope_l274_274586


namespace solve_quadratic_roots_l274_274788

theorem solve_quadratic_roots : ∀ x : ℝ, (x - 1)^2 = 1 → (x = 2 ∨ x = 0) :=
by
  sorry

end solve_quadratic_roots_l274_274788


namespace students_per_configuration_l274_274643

theorem students_per_configuration (students_per_column : ℕ → ℕ) :
  students_per_column 1 = 15 ∧
  students_per_column 2 = 1 ∧
  students_per_column 3 = 1 ∧
  students_per_column 4 = 6 ∧
  ∀ i j, (i ≠ j ∧ i ≤ 12 ∧ j ≤ 12) → students_per_column i ≠ students_per_column j →
  (∃ n, 13 ≤ n ∧ ∀ k, k < 13 → students_per_column k * n = 60) :=
by
  sorry

end students_per_configuration_l274_274643


namespace find_a_in_terms_of_x_l274_274140

variable (a b x : ℝ)

theorem find_a_in_terms_of_x (h1 : a ≠ b) (h2 : a^3 - b^3 = 27 * x^3) (h3 : a - b = 3 * x) : a = 3 * x :=
sorry

end find_a_in_terms_of_x_l274_274140


namespace factor_expression_l274_274985

theorem factor_expression (a b c : ℝ) :
  ((a^3 - b^3)^3 + (b^3 - c^3)^3 + (c^3 - a^3)^3) / ((a - b)^3 + (b - c)^3 + (c - a)^3) =
  (a^2 + a * b + b^2) * (b^2 + b * c + c^2) * (c^2 + c * a + a^2) :=
by
  sorry

end factor_expression_l274_274985


namespace count_negative_numbers_l274_274613

theorem count_negative_numbers : 
  let n1 := abs (-2)
  let n2 := - abs (3^2)
  let n3 := - (3^2)
  let n4 := (-2)^(2023)
  (if n1 < 0 then 1 else 0) + (if n2 < 0 then 1 else 0) + (if n3 < 0 then 1 else 0) + (if n4 < 0 then 1 else 0) = 3 := 
by
  sorry

end count_negative_numbers_l274_274613


namespace minimum_omega_l274_274862

theorem minimum_omega 
  (ω : ℝ)
  (hω : ω > 0)
  (h_shift : ∃ T > 0, T = 2 * π / ω ∧ T = 2 * π / 3) : 
  ω = 3 := 
sorry

end minimum_omega_l274_274862


namespace no_solutions_l274_274989

/-- Prove that there are no pairs of positive integers (x, y) such that x² + y² + x = 2x³. -/
theorem no_solutions : ∀ x y : ℕ, 0 < x → 0 < y → (x^2 + y^2 + x = 2 * x^3) → false :=
by
  sorry

end no_solutions_l274_274989


namespace DVDs_sold_is_168_l274_274607

variables (C D : ℕ)
variables (h1 : D = (16 * C) / 10)
variables (h2 : D + C = 273)

theorem DVDs_sold_is_168 : D = 168 := by
  sorry

end DVDs_sold_is_168_l274_274607


namespace range_frequency_l274_274593

-- Define the sample data
def sample_data : List ℝ := [10, 8, 6, 10, 13, 8, 10, 12, 11, 7, 8, 9, 11, 9, 12, 9, 10, 11, 12, 11]

-- Define the condition representing the frequency count
def frequency_count : ℝ := 0.2 * 20

-- Define the proof problem
theorem range_frequency (s : List ℝ) (range_start range_end : ℝ) : 
  s = sample_data → 
  range_start = 11.5 →
  range_end = 13.5 → 
  (s.filter (λ x => range_start ≤ x ∧ x < range_end)).length = frequency_count := 
by 
  intros
  sorry

end range_frequency_l274_274593


namespace compute_star_l274_274563

def star (x y : ℕ) := 4 * x + 6 * y

theorem compute_star : star 3 4 = 36 := 
by
  sorry

end compute_star_l274_274563


namespace work_together_days_l274_274231

-- Define the days it takes for A and B to complete the work individually.
def days_A : ℕ := 3
def days_B : ℕ := 6

-- Define the combined work rate.
def combined_work_rate : ℚ := (1 / days_A) + (1 / days_B)

-- State the theorem for the number of days A and B together can complete the work.
theorem work_together_days :
  1 / combined_work_rate = 2 := by
  sorry

end work_together_days_l274_274231


namespace Q_is_perfect_square_trinomial_l274_274394

def is_perfect_square_trinomial (p : ℤ → ℤ) :=
∃ (b : ℤ), ∀ a : ℤ, p a = (a + b) * (a + b)

def P (a b : ℤ) : ℤ := a^2 + 2 * a * b - b^2
def Q (a : ℤ) : ℤ := a^2 + 2 * a + 1
def R (a b : ℤ) : ℤ := a^2 + a * b + b^2
def S (a : ℤ) : ℤ := a^2 + 2 * a - 1

theorem Q_is_perfect_square_trinomial : is_perfect_square_trinomial Q :=
sorry -- Proof goes here

end Q_is_perfect_square_trinomial_l274_274394


namespace size_of_angle_C_max_value_of_a_add_b_l274_274750

variable (A B C a b c : ℝ)
variable (h₀ : 0 < A ∧ A < π / 2)
variable (h₁ : 0 < B ∧ B < π / 2)
variable (h₂ : 0 < C ∧ C < π / 2)
variable (h₃ : a = 2 * c * sin A / sqrt 3)
variable (h₄ : a * a + b * b - 2 * a * b * cos (π / 3) = c * c)

theorem size_of_angle_C (h₅: a ≠ 0):
  C = π / 3 :=
by sorry

theorem max_value_of_a_add_b (h₆: c = 2):
  a + b ≤ 4 :=
by sorry

end size_of_angle_C_max_value_of_a_add_b_l274_274750


namespace find_k_value_l274_274292

-- Definitions based on conditions
variables {k b x y : ℝ} -- k, b, x, and y are real numbers

-- Conditions given in the problem
def linear_function (k b x : ℝ) : ℝ := k * x + b

-- Proposition: Given the conditions, prove that k = 2
theorem find_k_value (h₁ : ∀ x y, y = linear_function k b x → y + 6 = linear_function k b (x + 3)) : k = 2 :=
by
  sorry

end find_k_value_l274_274292


namespace marsha_pay_per_mile_l274_274633

variable (distance1 distance2 payment : ℝ)
variable (distance3 : ℝ := distance2 / 2)
variable (totalDistance := distance1 + distance2 + distance3)

noncomputable def payPerMile (payment : ℝ) (totalDistance : ℝ) : ℝ :=
  payment / totalDistance

theorem marsha_pay_per_mile
  (distance1: ℝ := 10)
  (distance2: ℝ := 28)
  (payment: ℝ := 104)
  (distance3: ℝ := distance2 / 2)
  (totalDistance: ℝ := distance1 + distance2 + distance3)
  : payPerMile payment totalDistance = 2 := by
  sorry

end marsha_pay_per_mile_l274_274633


namespace expression_value_l274_274259

theorem expression_value : 7^4 + 4 * 7^3 + 6 * 7^2 + 4 * 7 + 1 = 4096 := 
by 
  -- proof goes here 
  sorry

end expression_value_l274_274259


namespace sum_of_powers_of_minus_one_l274_274078

theorem sum_of_powers_of_minus_one : (-1) ^ 2010 + (-1) ^ 2011 + 1 ^ 2012 - 1 ^ 2013 + (-1) ^ 2014 = -1 := by
  sorry

end sum_of_powers_of_minus_one_l274_274078


namespace outfits_count_l274_274009

theorem outfits_count (shirts ties pants belts : ℕ) (h_shirts : shirts = 7) (h_ties : ties = 5) (h_pants : pants = 4) (h_belts : belts = 2) : 
  (shirts * pants * (ties + 1) * (belts + 1 + 1) = 504) :=
by
  rw [h_shirts, h_ties, h_pants, h_belts]
  sorry

end outfits_count_l274_274009


namespace rational_root_of_p_l274_274287

noncomputable def p (n : ℕ) (x : ℚ) : ℚ :=
  x^n + (2 + x)^n + (2 - x)^n

theorem rational_root_of_p :
  ∀ n : ℕ, n > 0 → (∃ x : ℚ, p n x = 0) ↔ n = 1 := by
  sorry

end rational_root_of_p_l274_274287


namespace part1_part2_l274_274896

def P (a : ℝ) := ∀ x : ℝ, x^2 - a * x + a + 5 / 4 > 0
def Q (a : ℝ) := 4 * a + 7 ≠ 0 ∧ a - 3 ≠ 0 ∧ (4 * a + 7) * (a - 3) < 0

theorem part1 (h : Q a) : -7 / 4 < a ∧ a < 3 := sorry

theorem part2 (h : (P a ∨ Q a) ∧ ¬(P a ∧ Q a)) :
  (-7 / 4 < a ∧ a ≤ -1) ∨ (3 ≤ a ∧ a < 5) := sorry

end part1_part2_l274_274896


namespace village_food_sales_l274_274802

theorem village_food_sales :
  ∀ (customers_per_month heads_of_lettuce_per_person tomatoes_per_person 
      price_per_head_of_lettuce price_per_tomato : ℕ) 
    (H1 : customers_per_month = 500)
    (H2 : heads_of_lettuce_per_person = 2)
    (H3 : tomatoes_per_person = 4)
    (H4 : price_per_head_of_lettuce = 1)
    (H5 : price_per_tomato = 1 / 2), 
  customers_per_month * ((heads_of_lettuce_per_person * price_per_head_of_lettuce) 
    + (tomatoes_per_person * (price_per_tomato : ℝ))) = 2000 := 
by 
  intros customers_per_month heads_of_lettuce_per_person tomatoes_per_person 
         price_per_head_of_lettuce price_per_tomato 
         H1 H2 H3 H4 H5
  sorry

end village_food_sales_l274_274802


namespace tangent_lines_passing_through_point_l274_274647

theorem tangent_lines_passing_through_point :
  ∀ (x0 y0 : ℝ) (p : ℝ × ℝ), 
  (p = (1, 1)) ∧ (y0 = x0 ^ 3) → 
  (y0 - 1 = 3 * x0 ^ 2 * (1 - x0)) → 
  (x0 = 1 ∨ x0 = -1/2) → 
  ((y - (3 * 1 - 2)) * (y - (3/4 * x0 + 1/4))) = 0 :=
sorry

end tangent_lines_passing_through_point_l274_274647


namespace area_of_triangle_PDE_l274_274646

noncomputable def length (a b : Point) : ℝ := -- define length between two points
sorry

def distance_from_line (P D E : Point) : ℝ := -- define perpendicular distance from P to line DE
sorry

structure Point :=
(x : ℝ)
(y : ℝ)

def area_triangle (P D E : Point) : ℝ :=
0.5 -- define area given conditions

theorem area_of_triangle_PDE (D E : Point) (hD_E : D ≠ E) :
  { P : Point | area_triangle P D E = 0.5 } =
  { P : Point | distance_from_line P D E = 1 / (length D E) } :=
sorry

end area_of_triangle_PDE_l274_274646


namespace square_area_in_ellipse_l274_274550

theorem square_area_in_ellipse :
  (∃ t : ℝ, 
    (∀ x y : ℝ, ((x = t ∨ x = -t) ∧ (y = t ∨ y = -t)) → (x^2 / 4 + y^2 / 8 = 1)) 
    ∧ t > 0 
    ∧ ((2 * t)^2 = 32 / 3)) :=
sorry

end square_area_in_ellipse_l274_274550


namespace dvd_sold_168_l274_274605

/-- 
Proof that the number of DVDs sold (D) is 168 given the conditions:
1) D = 1.6 * C
2) D + C = 273 
-/
theorem dvd_sold_168 (C D : ℝ) (h1 : D = 1.6 * C) (h2 : D + C = 273) : D = 168 := 
sorry

end dvd_sold_168_l274_274605


namespace suzanne_donation_l274_274012

theorem suzanne_donation :
  let base_donation := 10
  let total_distance := 5
  let total_donation := (List.range total_distance).foldl (fun acc km => acc + base_donation * 2 ^ km) 0
  total_donation = 310 :=
by
  let base_donation := 10
  let total_distance := 5
  let total_donation := (List.range total_distance).foldl (fun acc km => acc + base_donation * 2 ^ km) 0
  sorry

end suzanne_donation_l274_274012


namespace molecular_weight_of_NH4I_l274_274378

-- Define the conditions in Lean
def molecular_weight (moles grams: ℕ) : Prop :=
  grams / moles = 145

-- Statement of the proof problem
theorem molecular_weight_of_NH4I :
  molecular_weight 9 1305 :=
by
  -- Proof is omitted 
  sorry

end molecular_weight_of_NH4I_l274_274378


namespace find_least_integer_l274_274103

theorem find_least_integer (x : ℤ) : (3 * |x| - 4 < 20) → (x ≥ -7) :=
by
  sorry

end find_least_integer_l274_274103


namespace expr_simplify_l274_274876

variable {a b c d m : ℚ}
variable {b_nonzero : b ≠ 0}
variable {m_nat : ℕ}
variable {m_bound : 0 ≤ m_nat ∧ m_nat < 2}

def expr_value (a b c d m : ℚ) : ℚ :=
  m - (c * d) + (a + b) / 2023 + a / b

theorem expr_simplify (h1 : a = -b) (h2 : c * d = 1) (h3 : m = (m_nat : ℚ)) :
  expr_value a b c d m = -1 ∨ expr_value a b c d m = -2 := by
  sorry

end expr_simplify_l274_274876


namespace problem_1_problem_2_l274_274305

def f (x : ℝ) : ℝ := |2 * x - 1|

theorem problem_1 : {x : ℝ | f x > 2} = {x : ℝ | x < -1 / 2 ∨ x > 3 / 2} := sorry

theorem problem_2 (m : ℝ) : (∀ x : ℝ, f x + |2 * (x + 3)| - 4 > m * x) → m ≤ -11 := sorry

end problem_1_problem_2_l274_274305


namespace a5_is_16_S8_is_255_l274_274116

-- Define the sequence
def seq (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | n + 1 => 2 * seq n

-- Definition of the geometric sum
def geom_sum (n : ℕ) : ℕ :=
  (2 ^ (n + 1) - 1)

-- Prove that a₅ = 16
theorem a5_is_16 : seq 5 = 16 :=
  by
  unfold seq
  sorry

-- Prove that the sum of the first 8 terms, S₈ = 255
theorem S8_is_255 : geom_sum 7 = 255 :=
  by 
  unfold geom_sum
  sorry

end a5_is_16_S8_is_255_l274_274116


namespace integer_solutions_count_l274_274310

theorem integer_solutions_count :
  ∃ (s : Finset (ℤ × ℤ)), (∀ (x y : ℤ), (6 * y^2 + 3 * x * y + x + 2 * y - 72 = 0) ↔ ((x, y) ∈ s)) ∧ s.card = 4 :=
begin
  sorry
end

end integer_solutions_count_l274_274310


namespace nonnegative_values_ineq_l274_274843

theorem nonnegative_values_ineq {x : ℝ} : 
  (x^2 - 6*x + 9) / (9 - x^3) ≥ 0 ↔ x ∈ Set.Iic 3 := 
sorry

end nonnegative_values_ineq_l274_274843


namespace product_consecutive_even_div_48_l274_274640

theorem product_consecutive_even_div_48 (k : ℤ) : 
  (2 * k) * (2 * k + 2) * (2 * k + 4) % 48 = 0 :=
by
  sorry

end product_consecutive_even_div_48_l274_274640


namespace domain_of_v_l274_274526

noncomputable def v (x : ℝ) : ℝ := 1 / Real.sqrt (Real.cos x)

theorem domain_of_v :
  (∀ x : ℝ, (∃ n : ℤ, 2 * n * Real.pi - Real.pi / 2 < x ∧ x < 2 * n * Real.pi + Real.pi / 2) ↔ 
    ∀ x : ℝ, ∀ x_in_domain : ℝ, (0 < Real.cos x ∧ 1 / Real.sqrt (Real.cos x) = x_in_domain)) :=
sorry

end domain_of_v_l274_274526


namespace function_for_negative_x_l274_274719

def odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

def given_function (f : ℝ → ℝ) : Prop :=
  ∀ x, (0 < x) → f x = x * (1 - x)

theorem function_for_negative_x {f : ℝ → ℝ} :
  odd_function f → given_function f → ∀ x, x < 0 → f x = x * (1 + x) :=
by
  intros h1 h2
  sorry

end function_for_negative_x_l274_274719


namespace donna_pays_total_l274_274090

def original_price_vase : ℝ := 250
def discount_vase : ℝ := original_price_vase * 0.25

def original_price_teacups : ℝ := 350
def discount_teacups : ℝ := original_price_teacups * 0.30

def original_price_plate : ℝ := 450
def discount_plate : ℝ := 0

def original_price_ornament : ℝ := 150
def discount_ornament : ℝ := original_price_ornament * 0.20

def membership_discount_vase : ℝ := (original_price_vase - discount_vase) * 0.05
def membership_discount_plate : ℝ := original_price_plate * 0.05

def tax_vase : ℝ := ((original_price_vase - discount_vase - membership_discount_vase) * 0.12)
def tax_teacups : ℝ := ((original_price_teacups - discount_teacups) * 0.08)
def tax_plate : ℝ := ((original_price_plate - membership_discount_plate) * 0.10)
def tax_ornament : ℝ := ((original_price_ornament - discount_ornament) * 0.06)

def final_price_vase : ℝ := (original_price_vase - discount_vase - membership_discount_vase) + tax_vase
def final_price_teacups : ℝ := (original_price_teacups - discount_teacups) + tax_teacups
def final_price_plate : ℝ := (original_price_plate - membership_discount_plate) + tax_plate
def final_price_ornament : ℝ := (original_price_ornament - discount_ornament) + tax_ornament

def total_price : ℝ := final_price_vase + final_price_teacups + final_price_plate + final_price_ornament

theorem donna_pays_total :
  total_price = 1061.55 :=
by
  sorry

end donna_pays_total_l274_274090


namespace perimeter_of_rectangle_l274_274932

theorem perimeter_of_rectangle (L W : ℝ) (h1 : L / W = 5 / 2) (h2 : L * W = 4000) : 2 * L + 2 * W = 280 :=
sorry

end perimeter_of_rectangle_l274_274932


namespace range_of_3a_minus_b_l274_274446

theorem range_of_3a_minus_b (a b : ℝ) (h1 : 1 ≤ a + b ∧ a + b ≤ 4) (h2 : -1 ≤ a - b ∧ a - b ≤ 2) :
  -1 ≤ (3 * a - b) ∧ (3 * a - b) ≤ 8 :=
sorry

end range_of_3a_minus_b_l274_274446


namespace range_of_m_for_line_to_intersect_ellipse_twice_l274_274564

theorem range_of_m_for_line_to_intersect_ellipse_twice (m : ℝ) :
  (∃ A B : ℝ × ℝ, A ≠ B ∧ 
   (A.2 = 4 * A.1 + m) ∧
   (B.2 = 4 * B.1 + m) ∧
   ((A.1 ^ 2) / 4 + (A.2 ^ 2) / 3 = 1) ∧
   ((B.1 ^ 2) / 4 + (B.2 ^ 2) / 3 = 1) ∧
   (A.1 + B.1) / 2 = 0 ∧ 
   (A.2 + B.2) / 2 = 4 * 0 + m) ↔
   - (2 * Real.sqrt 13) / 13 < m ∧ m < (2 * Real.sqrt 13) / 13
 :=
sorry

end range_of_m_for_line_to_intersect_ellipse_twice_l274_274564


namespace part1_part2_l274_274124

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := Real.exp x * abs (x^2 - a)

-- Define the two main proofs to be shown
theorem part1 (a : ℝ) (h : a = 1) : 
  ∃ I1 I2 : Set ℝ, I1 = Set.Icc (-1 - Real.sqrt 2) (-1) ∧ I2 = Set.Icc (-1 + Real.sqrt 2) (1) ∧ 
  ∀ x ∈ I1 ∪ I2, ∀ y ∈ I1 ∪ I2, x ≤ y → f y 1 ≤ f x 1 :=
sorry

theorem part2 (a : ℝ) (h : a ≥ 0) (h_roots : ∀ m : ℝ, (∃ x : ℝ, x > 0 ∧ f x a = m) ∧ (∃ x : ℝ, x < 0 ∧ f x a = m)) : 
  ∃ m : ℝ, m = 4 / (Real.exp 2) :=
sorry

end part1_part2_l274_274124


namespace math_problem_l274_274863

noncomputable def problem_statement : Prop :=
  ∃ b c : ℝ, 
  (∀ x : ℝ, (x^2 - b * x + c < 0) ↔ (-3 < x ∧ x < 2)) ∧ 
  (b + c = -7)

theorem math_problem : problem_statement := 
by
  sorry

end math_problem_l274_274863


namespace maximum_x_value_l274_274338

theorem maximum_x_value (x y z : ℝ) (h1 : x + y + z = 10) (h2 : x * y + x * z + y * z = 20) : 
  x ≤ 10 / 3 := sorry

end maximum_x_value_l274_274338


namespace power_expansion_l274_274426

theorem power_expansion (x y : ℝ) : (-2 * x^2 * y)^3 = -8 * x^6 * y^3 := 
by 
  sorry

end power_expansion_l274_274426


namespace least_positive_integer_solution_l274_274034

theorem least_positive_integer_solution :
  ∃ b : ℕ, b ≡ 1 [MOD 4] ∧ b ≡ 2 [MOD 5] ∧ b ≡ 3 [MOD 6] ∧ b = 37 :=
by
  sorry

end least_positive_integer_solution_l274_274034


namespace number_of_foals_l274_274969

theorem number_of_foals (t f : ℕ) (h1 : t + f = 11) (h2 : 2 * t + 4 * f = 30) : f = 4 :=
by
  sorry

end number_of_foals_l274_274969


namespace average_of_a_b_l274_274083

theorem average_of_a_b (a b : ℚ) (h1 : b = 2 * a) (h2 : (4 + 6 + 8 + a + b) / 5 = 17) : (a + b) / 2 = 33.5 := 
by
  sorry

end average_of_a_b_l274_274083


namespace smallest_digit_for_divisibility_by_3_l274_274282

theorem smallest_digit_for_divisibility_by_3 : ∃ x : ℕ, x < 10 ∧ (5 + 2 + 6 + x + 1 + 8) % 3 = 0 ∧ ∀ y : ℕ, y < 10 ∧ (5 + 2 + 6 + y + 1 + 8) % 3 = 0 → x ≤ y := by
  sorry

end smallest_digit_for_divisibility_by_3_l274_274282


namespace sum_of_midpoints_l274_274791

theorem sum_of_midpoints (d e f : ℝ) (h : d + e + f = 15) :
  (d + e) / 2 + (d + f) / 2 + (e + f) / 2 = 15 :=
by sorry

end sum_of_midpoints_l274_274791


namespace intersection_of_P_and_Q_l274_274996

def P : Set ℤ := {x | -4 ≤ x ∧ x ≤ 2 ∧ x ∈ Set.univ}
def Q : Set ℤ := {x | -3 < x ∧ x < 1}

theorem intersection_of_P_and_Q :
  P ∩ Q = {-2, -1, 0} :=
sorry

end intersection_of_P_and_Q_l274_274996


namespace expressions_equal_iff_l274_274566

theorem expressions_equal_iff (x y z : ℝ) : x + y + z = 0 ↔ x + yz = (x + y) * (x + z) :=
by
  sorry

end expressions_equal_iff_l274_274566


namespace meters_of_cloth_sold_l274_274829

-- Definitions based on conditions
def total_selling_price : ℕ := 8925
def profit_per_meter : ℕ := 20
def cost_price_per_meter : ℕ := 85
def selling_price_per_meter : ℕ := cost_price_per_meter + profit_per_meter

-- The proof statement
theorem meters_of_cloth_sold : ∃ x : ℕ, selling_price_per_meter * x = total_selling_price ∧ x = 85 := by
  sorry

end meters_of_cloth_sold_l274_274829


namespace air_quality_probability_l274_274322

variable (p_good_day : ℝ) (p_good_two_days : ℝ)

theorem air_quality_probability
  (h1 : p_good_day = 0.75)
  (h2 : p_good_two_days = 0.6) :
  (p_good_two_days / p_good_day = 0.8) :=
by
  rw [h1, h2]
  norm_num

end air_quality_probability_l274_274322


namespace no_such_abc_l274_274689

theorem no_such_abc :
  ¬ ∃ (a b c : ℕ+),
    (∃ k1 : ℕ, a ^ 2 * b * c + 2 = k1 ^ 2) ∧
    (∃ k2 : ℕ, b ^ 2 * c * a + 2 = k2 ^ 2) ∧
    (∃ k3 : ℕ, c ^ 2 * a * b + 2 = k3 ^ 2) := 
sorry

end no_such_abc_l274_274689


namespace least_n_prime_condition_l274_274852

theorem least_n_prime_condition : ∃ n : ℕ, (∀ p : ℕ, Prime p → ¬ Prime (p^2 + n)) ∧ (∀ m : ℕ, 
 (m > 0 ∧ ∀ p : ℕ, Prime p → ¬ Prime (p^2 + m)) → m ≥ 5) ∧ n = 5 := by
  sorry

end least_n_prime_condition_l274_274852


namespace directrix_of_parabola_l274_274779

theorem directrix_of_parabola (p : ℝ) : y^2 = -8 * x → x = 2 :=
by 
  assume h
  sorry

end directrix_of_parabola_l274_274779


namespace son_age_l274_274067

theorem son_age {M S : ℕ} 
  (h1 : M = S + 18) 
  (h2 : M + 2 = 2 * (S + 2)) : 
  S = 16 := 
by
  sorry

end son_age_l274_274067


namespace necess_suff_cond_odd_function_l274_274763

noncomputable def f (ω : ℝ) (ϕ : ℝ) (x : ℝ) := Real.sin (ω * x + ϕ)

def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

def P (ω ϕ : ℝ) : Prop := f ω ϕ 0 = 0
def Q (ω ϕ : ℝ) : Prop := is_odd (f ω ϕ)

theorem necess_suff_cond_odd_function (ω ϕ : ℝ) : P ω ϕ ↔ Q ω ϕ := by
  sorry

end necess_suff_cond_odd_function_l274_274763


namespace third_smallest_four_digit_number_in_pascals_triangle_l274_274198

theorem third_smallest_four_digit_number_in_pascals_triangle : 
  ∃ n, is_four_digit (binomial n k) ∧ third_smallest n = 1002 :=
by
  assume P : (∀ k n, 0 ≤ k ∧ k ≤ n → ∃ m, m = binomial n k)
  sorry

end third_smallest_four_digit_number_in_pascals_triangle_l274_274198


namespace graphGn_planarity_l274_274975

open Nat

def isPrime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def isConnected (a b : ℕ) : Prop := isPrime (a + b)

def graphGn (n : ℕ) : SimpleGraph ℕ :=
{ adj := λ a b, a ≠ b ∧ isConnected a b,
  symm := λ a b ab, ⟨ab.1.symm, ab.2⟩,
  loopless := λ a aa, aa.1 rfl }

theorem graphGn_planarity (n : ℕ) : (graphGn n).planar ↔ n ≤ 8 := sorry

end graphGn_planarity_l274_274975


namespace prime_divisor_exponent_l274_274761

theorem prime_divisor_exponent (a n : ℕ) (p : ℕ) 
    (ha : a ≥ 2)
    (hn : n ≥ 1) 
    (hp : Nat.Prime p) 
    (hdiv : p ∣ a^(2^n) + 1) :
    2^(n+1) ∣ (p-1) :=
by
  sorry

end prime_divisor_exponent_l274_274761


namespace swimmer_speed_in_still_water_l274_274417

-- Define the various given conditions as constants in Lean
def swimmer_distance : ℝ := 3
def river_current_speed : ℝ := 1.7
def time_taken : ℝ := 2.3076923076923075

-- Define what we need to prove: the swimmer's speed in still water
theorem swimmer_speed_in_still_water (v : ℝ) :
  swimmer_distance = (v - river_current_speed) * time_taken → 
  v = 3 := by
  sorry

end swimmer_speed_in_still_water_l274_274417


namespace solve_for_a_l274_274454

-- Given conditions
def x : ℕ := 2
def y : ℕ := 2
def equation (a : ℚ) : Prop := a * x + y = 5

-- Our goal is to prove that "a = 3/2" given the conditions
theorem solve_for_a : ∃ a : ℚ, equation a ∧ a = 3 / 2 :=
by
  sorry

end solve_for_a_l274_274454


namespace find_alpha_beta_l274_274715

-- Define the conditions of the problem
variables (α β : ℝ) (hα : 0 < α ∧ α < π) (hβ : π < β ∧ β < 2 * π)
variable (h_eq : ∀ x : ℝ, cos (x + α) + sin (x + β) + sqrt 2 * cos x = 0)

-- State the required proof as a theorem
theorem find_alpha_beta : α = 3 * π / 4 ∧ β = 7 * π / 4 :=
by
  sorry

end find_alpha_beta_l274_274715


namespace probability_red_or_white_correct_l274_274670

-- Define the conditions
def totalMarbles : ℕ := 30
def blueMarbles : ℕ := 5
def redMarbles : ℕ := 9
def whiteMarbles : ℕ := totalMarbles - (blueMarbles + redMarbles)

-- Define the calculated probability
def probabilityRedOrWhite : ℚ := (redMarbles + whiteMarbles) / totalMarbles

-- Verify the probability is equal to 5 / 6
theorem probability_red_or_white_correct :
  probabilityRedOrWhite = 5 / 6 := by
  sorry

end probability_red_or_white_correct_l274_274670


namespace condition_sufficiency_l274_274023

theorem condition_sufficiency (x₁ x₂ : ℝ) :
  (x₁ > 4 ∧ x₂ > 4) → (x₁ + x₂ > 8 ∧ x₁ * x₂ > 16) ∧ ¬ ((x₁ + x₂ > 8 ∧ x₁ * x₂ > 16) → (x₁ > 4 ∧ x₂ > 4)) :=
by 
  sorry

end condition_sufficiency_l274_274023


namespace ratio_frogs_to_dogs_l274_274148

variable (D C F : ℕ)

-- Define the conditions as given in the problem statement
def cats_eq_dogs_implied : Prop := C = Nat.div (4 * D) 5
def frogs : Prop := F = 160
def total_animals : Prop := D + C + F = 304

-- Define the statement to be proved
theorem ratio_frogs_to_dogs (h1 : cats_eq_dogs_implied D C) (h2 : frogs F) (h3 : total_animals D C F) : F / D = 2 := by
  sorry

end ratio_frogs_to_dogs_l274_274148


namespace largest_possible_A_l274_274667

theorem largest_possible_A (A B C : ℕ) (h1 : 10 = A * B + C) (h2 : B = C) : A ≤ 9 :=
by sorry

end largest_possible_A_l274_274667


namespace geom_arith_seq_first_term_is_two_l274_274927

theorem geom_arith_seq_first_term_is_two (b q a d : ℝ) 
  (hq : q ≠ 1) 
  (h_geom_first : b = a + d) 
  (h_geom_second : b * q = a + 3 * d) 
  (h_geom_third : b * q^2 = a + 6 * d) 
  (h_prod : b * b * q * b * q^2 = 64) :
  b = 2 :=
by
  sorry

end geom_arith_seq_first_term_is_two_l274_274927


namespace actual_distance_traveled_l274_274947

theorem actual_distance_traveled (D : ℕ) 
  (h : D / 10 = (D + 36) / 16) : D = 60 := by
  sorry

end actual_distance_traveled_l274_274947


namespace tank_fraction_after_adding_water_l274_274730

noncomputable def fraction_of_tank_full 
  (initial_fraction : ℚ) 
  (additional_water : ℚ) 
  (total_capacity : ℚ) 
  : ℚ :=
(initial_fraction * total_capacity + additional_water) / total_capacity

theorem tank_fraction_after_adding_water 
  (initial_fraction : ℚ) 
  (additional_water : ℚ) 
  (total_capacity : ℚ) 
  (h_initial : initial_fraction = 3 / 4) 
  (h_addition : additional_water = 4) 
  (h_capacity : total_capacity = 32) 
: fraction_of_tank_full initial_fraction additional_water total_capacity = 7 / 8 :=
by
  sorry

end tank_fraction_after_adding_water_l274_274730


namespace sum_of_digits_of_expression_l274_274665

theorem sum_of_digits_of_expression : 
  (sum_digits (decimal_repr (2^2010 * 5^2008 * 7)) = 10) :=
sorry

end sum_of_digits_of_expression_l274_274665


namespace value_of_a_plus_b_l274_274304

noncomputable def f (a b x : ℝ) : ℝ := a * x^3 + b * x + 1

theorem value_of_a_plus_b (a b : ℝ) (h1 : 3 * a + b = 4) (h2 : a + b + 1 = 3) : a + b = 2 :=
by
  sorry

end value_of_a_plus_b_l274_274304


namespace domain_of_fractional_sqrt_function_l274_274924

theorem domain_of_fractional_sqrt_function :
  ∀ x : ℝ, (x + 4 ≥ 0) ∧ (x - 1 ≠ 0) ↔ (x ∈ (Set.Ici (-4) \ {1})) :=
by
  sorry

end domain_of_fractional_sqrt_function_l274_274924


namespace tank_fill_time_l274_274245

theorem tank_fill_time (R L : ℝ) (h1 : (R - L) * 8 = 1) (h2 : L * 56 = 1) :
  (1 / R) = 7 :=
by
  sorry

end tank_fill_time_l274_274245


namespace compare_abc_l274_274622

noncomputable def a : ℝ := 2 * Real.log (21 / 20)
noncomputable def b : ℝ := Real.log (11 / 10)
noncomputable def c : ℝ := Real.sqrt 1.2 - 1

theorem compare_abc : a > b ∧ b < c ∧ a > c :=
by {
  sorry
}

end compare_abc_l274_274622


namespace vika_pairs_exactly_8_ways_l274_274372

theorem vika_pairs_exactly_8_ways :
  ∃ d : ℕ, (d ∣ 30) ∧ (Finset.card (Finset.filter (λ d, d ∣ 30) (Finset.range 31)) = 8) := 
sorry

end vika_pairs_exactly_8_ways_l274_274372


namespace units_digit_n_l274_274704

theorem units_digit_n (m n : ℕ) (h1 : m * n = 14^5) (h2 : m % 10 = 8) : n % 10 = 3 :=
sorry

end units_digit_n_l274_274704


namespace units_digit_periodic_10_l274_274912

theorem units_digit_periodic_10:
  ∀ n: ℕ, (n * (n + 1) * (n + 2)) % 10 = ((n + 10) * (n + 11) * (n + 12)) % 10 :=
by
  sorry

end units_digit_periodic_10_l274_274912


namespace part_one_part_two_l274_274448

noncomputable def f (x a : ℝ) : ℝ :=
  x^2 - (a + 1/a) * x + 1

theorem part_one (x : ℝ) : f x (1/2) ≤ 0 ↔ (1/2 ≤ x ∧ x ≤ 2) :=
by
  sorry

theorem part_two (x a : ℝ) (h : a > 0) : 
  ((a < 1) → (f x a ≤ 0 ↔ (a ≤ x ∧ x ≤ 1/a))) ∧
  ((a > 1) → (f x a ≤ 0 ↔ (1/a ≤ x ∧ x ≤ a))) ∧
  ((a = 1) → (f x a ≤ 0 ↔ (x = 1))) :=
by
  sorry

end part_one_part_two_l274_274448


namespace system1_solution_system2_solution_l274_274916

theorem system1_solution (x y : ℝ) (h1 : 3 * x + y = 4) (h2 : 3 * x + 2 * y = 6) : x = 2 / 3 ∧ y = 2 :=
by
  sorry

theorem system2_solution (x y : ℝ) (h1 : 2 * x + y = 3) (h2 : 3 * x - 5 * y = 11) : x = 2 ∧ y = -1 :=
by
  sorry

end system1_solution_system2_solution_l274_274916


namespace least_positive_integer_satifies_congruences_l274_274045

theorem least_positive_integer_satifies_congruences :
  ∃ x : ℕ, x ≡ 1 [MOD 4] ∧ x ≡ 2 [MOD 5] ∧ x ≡ 3 [MOD 6] ∧ x = 17 :=
sorry

end least_positive_integer_satifies_congruences_l274_274045


namespace divisibility_by_37_l274_274641

def sum_of_segments (n : ℕ) : ℕ :=
  let rec split_and_sum (num : ℕ) (acc : ℕ) : ℕ :=
    if num < 1000 then acc + num
    else split_and_sum (num / 1000) (acc + num % 1000)
  split_and_sum n 0

theorem divisibility_by_37 (A : ℕ) : 
  (37 ∣ A) ↔ (37 ∣ sum_of_segments A) :=
sorry

end divisibility_by_37_l274_274641


namespace cover_condition_l274_274994

theorem cover_condition (n : ℕ) :
  (∃ (f : ℕ) (h1 : f = n^2), f % 2 = 0) ↔ (n % 2 = 0) := 
sorry

end cover_condition_l274_274994


namespace real_roots_polynomial_ab_leq_zero_l274_274639

theorem real_roots_polynomial_ab_leq_zero
  {a b c : ℝ}
  (h : ∀ x, Polynomial.eval x (Polynomial.C 1 * X^4 + Polynomial.C a * X^3 + Polynomial.C b * X + Polynomial.C c) = 0 → x ∈ ℝ) :
  a * b ≤ 0 := 
begin
  sorry
end

end real_roots_polynomial_ab_leq_zero_l274_274639


namespace find_third_number_l274_274181

noncomputable def averageFirstSet (x : ℝ) : ℝ := (20 + 40 + x) / 3
noncomputable def averageSecondSet : ℝ := (10 + 70 + 16) / 3

theorem find_third_number (x : ℝ) (h : averageFirstSet x = averageSecondSet + 8) : x = 60 :=
by
  sorry

end find_third_number_l274_274181


namespace find_possible_numbers_l274_274401

theorem find_possible_numbers (N : ℕ) (a : ℕ) (h : 1 ≤ a ∧ a ≤ 9)
  (cond : nat.num_digits N = 200 ∧ N = 10^199 * 5 * a ∧ 10^199 * 5 * a < 10^200) :
  (∃ a, N = 125 * a * 10^197 ∧ (1 ≤ a ∧ a ≤ 3)) :=
by
  sorry

end find_possible_numbers_l274_274401


namespace eq1_solution_eq2_solution_l274_274693


-- Theorem for the first equation (4(x + 1)^2 - 25 = 0)
theorem eq1_solution (x : ℝ) : (4 * (x + 1)^2 - 25 = 0) ↔ (x = 3 / 2 ∨ x = -7 / 2) :=
by
  sorry

-- Theorem for the second equation ((x + 10)^3 = -125)
theorem eq2_solution (x : ℝ) : ((x + 10)^3 = -125) ↔ (x = -15) :=
by
  sorry

end eq1_solution_eq2_solution_l274_274693


namespace exists_adj_diff_gt_3_max_min_adj_diff_l274_274190
-- Import needed libraries

-- Definition of the given problem and statement of the parts (a) and (b)

-- Part (a)
theorem exists_adj_diff_gt_3 (arrangement : Fin 18 → Fin 18) (adj : Fin 18 → Fin 18 → Prop) :
  (∀ i j : Fin 18, adj i j → i ≠ j) →
  (∃ i j : Fin 18, adj i j ∧ |arrangement i - arrangement j| > 3) :=
sorry

-- Part (b)
theorem max_min_adj_diff (arrangement : Fin 18 → Fin 18) (adj : Fin 18 → Fin 18 → Prop) :
  (∀ i j : Fin 18, adj i j → i ≠ j) →
  (∀ i j : Fin 18, adj i j → |arrangement i - arrangement j| ≥ 6) :=
sorry

end exists_adj_diff_gt_3_max_min_adj_diff_l274_274190


namespace solve_eq_l274_274437

theorem solve_eq :
  { x : ℝ | (14 * x - x^2) / (x + 2) * (x + (14 - x) / (x + 2)) = 48 } =
  {4, (1 + Real.sqrt 193) / 2, (1 - Real.sqrt 193) / 2} :=
by
  sorry

end solve_eq_l274_274437


namespace factor_diff_of_squares_l274_274277

theorem factor_diff_of_squares (y : ℝ) : 25 - 16 * y^2 = (5 - 4 * y) * (5 + 4 * y) := 
sorry

end factor_diff_of_squares_l274_274277


namespace h_is_decreasing_intervals_l274_274706

noncomputable def f (x : ℝ) := if x >= 1 then x - 2 else 0
noncomputable def g (x : ℝ) := if x <= 2 then -2 * x + 3 else 0

noncomputable def h (x : ℝ) :=
  if x >= 1 ∧ x <= 2 then f x * g x
  else if x >= 1 then f x
  else if x <= 2 then g x
  else 0

theorem h_is_decreasing_intervals :
  (∀ x1 x2 : ℝ, x1 < x2 → x1 < 1 → h x1 > h x2) ∧
  (∀ x1 x2 : ℝ, x1 < x2 → x1 ≥ 7 / 4 → x2 ≤ 2 → h x1 ≥ h x2) :=
by
  sorry

end h_is_decreasing_intervals_l274_274706


namespace arithmetic_seq_75th_term_difference_l274_274421

theorem arithmetic_seq_75th_term_difference :
  ∃ (d : ℝ), 300 * (50 + d) = 15000 ∧ -30 / 299 ≤ d ∧ d ≤ 30 / 299 ∧
  let L := 50 - 225 * (30 / 299)
  let G := 50 + 225 * (30 / 299)
  G - L = 13500 / 299 := by
sorry

end arithmetic_seq_75th_term_difference_l274_274421


namespace spade_problem_l274_274993

def spade (x y : ℝ) : ℝ := (x + y) * (x - y)

theorem spade_problem : spade 2 (spade 3 (spade 1 4)) = -46652 := 
by sorry

end spade_problem_l274_274993


namespace proposition_correctness_l274_274669

theorem proposition_correctness :
  (∀ x : ℝ, (|x-1| < 2) → (x < 3)) ∧
  (∀ (P Q : Prop), (Q → ¬ P) → (P → ¬ Q)) :=
by 
sorry

end proposition_correctness_l274_274669


namespace inequality_geq_27_l274_274341

theorem inequality_geq_27 (a b c : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c)
    (h_eq : a + b + c + 2 = a * b * c) : (a + 1) * (b + 1) * (c + 1) ≥ 27 := 
    sorry

end inequality_geq_27_l274_274341


namespace find_sum_l274_274573

theorem find_sum (a b : ℝ) 
  (h₁ : (a + Real.sqrt b) + (a - Real.sqrt b) = -8) 
  (h₂ : (a + Real.sqrt b) * (a - Real.sqrt b) = 4) : 
  a + b = 8 := 
sorry

end find_sum_l274_274573


namespace leak_empties_tank_in_8_hours_l274_274820

theorem leak_empties_tank_in_8_hours (capacity : ℕ) (inlet_rate_per_minute : ℕ) (time_with_inlet_open : ℕ) (time_without_inlet_open : ℕ) : 
  capacity = 8640 ∧ inlet_rate_per_minute = 6 ∧ time_with_inlet_open = 12 ∧ time_without_inlet_open = 8 := 
by 
  sorry

end leak_empties_tank_in_8_hours_l274_274820


namespace smallest_value_a_b_l274_274867

theorem smallest_value_a_b (a b : ℕ) (h : 2^6 * 3^9 = a^b) : a > 0 ∧ b > 0 ∧ (a + b = 111) :=
by
  sorry

end smallest_value_a_b_l274_274867


namespace square_inscribed_in_ellipse_area_l274_274549

theorem square_inscribed_in_ellipse_area :
  ∀ t : ℝ,
    (∃ t, (t ≠ 0 ∧ (t * t / 4 + t * t / 8 = 1))) →
    let side_length := 2 * t in
    side_length ^ 2 = 32 / 3 :=
by
  -- Proof skipped for now
  sorry

end square_inscribed_in_ellipse_area_l274_274549


namespace correct_operation_l274_274809

theorem correct_operation (a : ℝ) :
  (a^2)^3 = a^6 :=
by
  sorry

end correct_operation_l274_274809


namespace find_number_l274_274314

theorem find_number (x : ℝ) (h : 0.65 * x = 0.05 * 60 + 23) : x = 40 :=
sorry

end find_number_l274_274314


namespace sqrt5_lt_sqrt2_plus_1_l274_274836

theorem sqrt5_lt_sqrt2_plus_1 : Real.sqrt 5 < Real.sqrt 2 + 1 :=
sorry

end sqrt5_lt_sqrt2_plus_1_l274_274836


namespace lisa_ratio_l274_274632

theorem lisa_ratio (L J T : ℝ) 
  (h1 : L + J + T = 60) 
  (h2 : T = L / 2) 
  (h3 : L = T + 15) : 
  L / 60 = 1 / 2 :=
by 
  sorry

end lisa_ratio_l274_274632


namespace annie_age_when_anna_three_times_current_age_l274_274423

theorem annie_age_when_anna_three_times_current_age
  (anna_age : ℕ) (annie_age : ℕ)
  (h1 : anna_age = 13)
  (h2 : annie_age = 3 * anna_age) :
  annie_age + 2 * anna_age = 65 :=
by
  sorry

end annie_age_when_anna_three_times_current_age_l274_274423


namespace map_length_conversion_l274_274001

-- Define the given condition: 12 cm on the map represents 72 km in reality.
def length_on_map := 12 -- in cm
def distance_in_reality := 72 -- in km

-- Define the length in cm we want to find the real-world distance for.
def query_length := 17 -- in cm

-- State the proof problem.
theorem map_length_conversion :
  (distance_in_reality / length_on_map) * query_length = 102 :=
by
  -- placeholder for the proof
  sorry

end map_length_conversion_l274_274001


namespace total_monthly_bill_working_from_home_l274_274903

-- Definitions based on conditions
def original_bill : ℝ := 60
def increase_rate : ℝ := 0.45
def additional_internet_cost : ℝ := 25
def additional_cloud_cost : ℝ := 15

-- The theorem to prove
theorem total_monthly_bill_working_from_home : 
  original_bill * (1 + increase_rate) + additional_internet_cost + additional_cloud_cost = 127 := by
  sorry

end total_monthly_bill_working_from_home_l274_274903


namespace park_area_is_correct_l274_274659

-- Define the side of the square
def side_length : ℕ := 30

-- Define the area function for a square
def area_of_square (side: ℕ) : ℕ := side * side

-- State the theorem we're going to prove
theorem park_area_is_correct : area_of_square side_length = 900 := 
sorry -- proof not required

end park_area_is_correct_l274_274659


namespace area_of_triangle_l274_274741

theorem area_of_triangle (a b c : ℝ) (A B C : ℝ) (h₁ : b = 2) (h₂ : c = 2 * Real.sqrt 2) (h₃ : C = Real.pi / 4) :
  1 / 2 * b * c * Real.sin (Real.pi - B - C) = Real.sqrt 3 + 1 := 
by
  sorry

end area_of_triangle_l274_274741


namespace minimum_value_polynomial_l274_274019

def polynomial (x y : ℝ) : ℝ := 5 * x^2 - 4 * x * y + 4 * y^2 + 12 * x + 25

theorem minimum_value_polynomial : ∃ (m : ℝ), (∀ (x y : ℝ), polynomial x y ≥ m) ∧ m = 16 :=
by
  sorry

end minimum_value_polynomial_l274_274019


namespace number_of_clips_after_k_steps_l274_274524

theorem number_of_clips_after_k_steps (k : ℕ) : 
  ∃ (c : ℕ), c = 2^(k-1) + 1 :=
by sorry

end number_of_clips_after_k_steps_l274_274524


namespace last_place_is_Fedya_l274_274635

def position_is_valid (position : ℕ) := position >= 1 ∧ position <= 4

variable (Misha Anton Petya Fedya : ℕ)

axiom Misha_statement: position_is_valid Misha → Misha ≠ 1 ∧ Misha ≠ 4
axiom Anton_statement: position_is_valid Anton → Anton ≠ 4
axiom Petya_statement: position_is_valid Petya → Petya = 1
axiom Fedya_statement: position_is_valid Fedya → Fedya = 4

theorem last_place_is_Fedya : ∃ (x : ℕ), x = Fedya ∧ Fedya = 4 :=
by
  sorry

end last_place_is_Fedya_l274_274635


namespace matrix_arithmetic_sequence_sum_l274_274612

theorem matrix_arithmetic_sequence_sum (a : ℕ → ℕ → ℕ)
  (h_row1 : ∀ i, 2 * a 4 2 = a 4 (i - 1) + a 4 (i + 1))
  (h_row2 : ∀ i, 2 * a 5 2 = a 5 (i - 1) + a 5 (i + 1))
  (h_row3 : ∀ i, 2 * a 6 2 = a 6 (i - 1) + a 6 (i + 1))
  (h_col1 : ∀ i, 2 * a 5 2 = a (i - 1) 2 + a (i + 1) 2)
  (h_sum : a 4 1 + a 4 2 + a 4 3 + a 5 1 + a 5 2 + a 5 3 + a 6 1 + a 6 2 + a 6 3 = 63)
  : a 5 2 = 7 := sorry

end matrix_arithmetic_sequence_sum_l274_274612


namespace overlap_length_in_mm_l274_274944

theorem overlap_length_in_mm {sheets : ℕ} {length_per_sheet : ℝ} {perimeter : ℝ} 
  (h_sheets : sheets = 12)
  (h_length_per_sheet : length_per_sheet = 18)
  (h_perimeter : perimeter = 210) : 
  (length_per_sheet * sheets - perimeter) / sheets * 10 = 5 := by
  sorry

end overlap_length_in_mm_l274_274944


namespace alice_chicken_weight_l274_274660

theorem alice_chicken_weight (total_cost_needed : ℝ)
  (amount_to_spend_more : ℝ)
  (cost_lettuce : ℝ)
  (cost_tomatoes : ℝ)
  (sweet_potato_quantity : ℝ)
  (cost_per_sweet_potato : ℝ)
  (broccoli_quantity : ℝ)
  (cost_per_broccoli : ℝ)
  (brussel_sprouts_weight : ℝ)
  (cost_per_brussel_sprouts : ℝ)
  (cost_per_pound_chicken : ℝ)
  (total_cost_excluding_chicken : ℝ) :
  total_cost_needed = 35 ∧
  amount_to_spend_more = 11 ∧
  cost_lettuce = 3 ∧
  cost_tomatoes = 2.5 ∧
  sweet_potato_quantity = 4 ∧
  cost_per_sweet_potato = 0.75 ∧
  broccoli_quantity = 2 ∧
  cost_per_broccoli = 2 ∧
  brussel_sprouts_weight = 1 ∧
  cost_per_brussel_sprouts = 2.5 ∧
  total_cost_excluding_chicken = (cost_lettuce + cost_tomatoes + sweet_potato_quantity * cost_per_sweet_potato + broccoli_quantity * cost_per_broccoli + brussel_sprouts_weight * cost_per_brussel_sprouts) →
  (total_cost_needed - amount_to_spend_more - total_cost_excluding_chicken) / cost_per_pound_chicken = 1.5 :=
by
  intros
  sorry

end alice_chicken_weight_l274_274660


namespace contrapositive_example_l274_274776

theorem contrapositive_example (x : ℝ) : 
  (x^2 - 3*x + 2 = 0 → x = 1) ↔ (x ≠ 1 → x^2 - 3*x + 2 ≠ 0) := 
by
  sorry

end contrapositive_example_l274_274776


namespace find_y_l274_274547

noncomputable def x : Real := 2.6666666666666665

theorem find_y (y : Real) (h : (x * y) / 3 = x^2) : y = 8 :=
sorry

end find_y_l274_274547


namespace number_of_intersections_l274_274782

   -- Definitions corresponding to conditions
   def C1 (x y : ℝ) : Prop := x^2 - y^2 + 4*y - 3 = 0
   def C2 (a x y : ℝ) : Prop := y = a*x^2
   def positive_real (a : ℝ) : Prop := a > 0

   -- Final statement converting the question, conditions, and correct answer into Lean code
   theorem number_of_intersections (a : ℝ) (ha : positive_real a) :
     ∃ (count : ℕ), (count = 4) ∧
     (∀ x y : ℝ, C1 x y → C2 a x y → True) := sorry
   
end number_of_intersections_l274_274782


namespace sqrt_range_real_l274_274464

theorem sqrt_range_real (x : ℝ) (h : 1 - 3 * x ≥ 0) : x ≤ 1 / 3 :=
sorry

end sqrt_range_real_l274_274464


namespace ab_perpendicular_cd_l274_274723

variables {A B C D : Type} [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace D]

-- Assuming points are members of a metric space and distances are calculated using the distance function
variables (a b c d : A)

-- Given condition
def given_condition : Prop := 
  dist a c ^ 2 + dist b d ^ 2 = dist a d ^ 2 + dist b c ^ 2

-- Statement that needs to be proven
theorem ab_perpendicular_cd (h : given_condition a b c d) : dist a b * dist c d = 0 :=
sorry

end ab_perpendicular_cd_l274_274723


namespace remainder_eq_159_l274_274089

def x : ℕ := 2^40
def numerator : ℕ := 2^160 + 160
def denominator : ℕ := 2^80 + 2^40 + 1

theorem remainder_eq_159 : (numerator % denominator) = 159 := 
by {
  -- Proof will be filled in here.
  sorry
}

end remainder_eq_159_l274_274089


namespace division_by_fraction_l274_274279

theorem division_by_fraction :
  (3 : ℚ) / (6 / 11) = 11 / 2 :=
by
  sorry

end division_by_fraction_l274_274279


namespace solution_set_of_inequality_l274_274363

theorem solution_set_of_inequality :
  {x : ℝ | (x + 1) / x ≤ 3} = {x : ℝ | x < 0} ∪ {x : ℝ | x ≥ 1 / 2} :=
by
  sorry

end solution_set_of_inequality_l274_274363


namespace mother_daughter_ages_l274_274411

theorem mother_daughter_ages :
  ∃ (x y : ℕ), (y = x + 22) ∧ (2 * x = (x + 22) - x) ∧ (x = 11) ∧ (y = 33) :=
by
  sorry

end mother_daughter_ages_l274_274411


namespace sam_age_two_years_ago_l274_274672

variables (S J : ℕ)
variables (h1 : J = 3 * S) (h2 : J + 9 = 2 * (S + 9))

theorem sam_age_two_years_ago : S - 2 = 7 := by
  sorry

end sam_age_two_years_ago_l274_274672


namespace total_customers_l274_274954

namespace math_proof

-- Definitions based on the problem's conditions.
def tables : ℕ := 9
def women_per_table : ℕ := 7
def men_per_table : ℕ := 3

-- The theorem stating the problem's question and correct answer.
theorem total_customers : tables * (women_per_table + men_per_table) = 90 := 
by
  -- This would be expanded into a proof, but we use sorry to bypass it here.
  sorry

end math_proof

end total_customers_l274_274954


namespace functional_equation_solution_l274_274847

-- Define ℕ* (positive integers) as a subtype of ℕ
def Nat.star := {n : ℕ // n > 0}

-- Define the problem statement
theorem functional_equation_solution (f : Nat.star → Nat.star) :
  (∀ m n : Nat.star, m.val ^ 2 + (f n).val ∣ m.val * (f m).val + n.val) →
  (∀ n : Nat.star, f n = n) :=
by
  intro h
  sorry

end functional_equation_solution_l274_274847


namespace evaluate_expression_at_2_l274_274276

theorem evaluate_expression_at_2 : ∀ (x : ℕ), x = 2 → (x^x)^(x^(x^x)) = 4294967296 := by
  intros x h
  rw [h]
  sorry

end evaluate_expression_at_2_l274_274276


namespace find_y_l274_274312

theorem find_y (x y : ℝ) (h1 : x - y = 10) (h2 : x + y = 18) : y = 4 := 
by 
  sorry

end find_y_l274_274312


namespace twenty_four_game_l274_274475

-- Definition of the cards' values
def card2 : ℕ := 2
def card5 : ℕ := 5
def cardJ : ℕ := 11
def cardQ : ℕ := 12

-- Theorem stating the proof
theorem twenty_four_game : card2 * (cardJ - card5) + cardQ = 24 :=
by
  sorry

end twenty_four_game_l274_274475


namespace third_smallest_four_digit_number_in_pascals_triangle_l274_274197

theorem third_smallest_four_digit_number_in_pascals_triangle : 
  ∃ n, is_four_digit (binomial n k) ∧ third_smallest n = 1002 :=
by
  assume P : (∀ k n, 0 ≤ k ∧ k ≤ n → ∃ m, m = binomial n k)
  sorry

end third_smallest_four_digit_number_in_pascals_triangle_l274_274197


namespace number_of_valid_pairing_ways_l274_274373

-- Define a natural number as a condition.
def is_natural (n : ℕ) : Prop := 0 < n

-- Define that 60 cards can be paired with the same modulus difference.
def pair_cards_same_modulus_difference (d : ℕ) (k : ℕ) : Prop :=
  60 = 2 * d * k

-- Define what it means for d to be a divisor of 30.
def is_divisor_of_30 (d : ℕ) : Prop :=
  ∃ k, 30 = d * k

theorem number_of_valid_pairing_ways :
  (finset.univ.filter is_divisor_of_30).card = 8 :=
begin
  sorry
end

end number_of_valid_pairing_ways_l274_274373


namespace domain_of_f_l274_274925

theorem domain_of_f : 
  ∀ x, (2 - x ≥ 0) ∧ (x + 1 > 0) ↔ (-1 < x ∧ x ≤ 2) := by
  sorry

end domain_of_f_l274_274925


namespace pencils_ratio_l274_274941

theorem pencils_ratio
  (Sarah_pencils : ℕ)
  (Tyrah_pencils : ℕ)
  (Tim_pencils : ℕ)
  (h1 : Tyrah_pencils = 12)
  (h2 : Tim_pencils = 16)
  (h3 : Tim_pencils = 8 * Sarah_pencils) :
  Tyrah_pencils / Sarah_pencils = 6 :=
by
  sorry

end pencils_ratio_l274_274941


namespace thirty_five_times_ninety_nine_is_not_thirty_five_times_hundred_plus_thirty_five_l274_274397

theorem thirty_five_times_ninety_nine_is_not_thirty_five_times_hundred_plus_thirty_five :
  (35 * 99 ≠ 35 * 100 + 35) :=
by
  sorry

end thirty_five_times_ninety_nine_is_not_thirty_five_times_hundred_plus_thirty_five_l274_274397


namespace find_a_l274_274738

theorem find_a (f : ℕ → ℕ) (a : ℕ) 
  (h1 : ∀ x : ℕ, f (x + 1) = x) 
  (h2 : f a = 8) : a = 9 :=
sorry

end find_a_l274_274738


namespace prod_sum_rel_prime_l274_274185

theorem prod_sum_rel_prime (a b : ℕ) 
  (h1 : a * b + a + b = 119)
  (h2 : Nat.gcd a b = 1)
  (h3 : a < 25)
  (h4 : b < 25) : 
  a + b = 27 := 
sorry

end prod_sum_rel_prime_l274_274185


namespace age_ratio_l274_274822

theorem age_ratio (S : ℕ) (M : ℕ) (h1 : S = 28) (h2 : M = S + 30) : 
  ((M + 2) / (S + 2) = 2) := 
by
  sorry

end age_ratio_l274_274822


namespace infinitely_many_87_b_seq_l274_274167

def a_seq : ℕ → ℕ
| 0 => 3
| (n + 1) => 3 ^ (a_seq n)

def b_seq (n : ℕ) : ℕ := (a_seq n) % 100

theorem infinitely_many_87_b_seq (n : ℕ) (hn : n ≥ 2) : b_seq n = 87 := by
  sorry

end infinitely_many_87_b_seq_l274_274167


namespace can_measure_all_weights_l274_274519

theorem can_measure_all_weights (a b c : ℕ) 
  (h_sum : a + b + c = 10) 
  (h_unique : (a = 1 ∧ b = 2 ∧ c = 7) ∨ (a = 1 ∧ b = 3 ∧ c = 6)) : 
  ∀ w : ℕ, 1 ≤ w ∧ w ≤ 10 → 
    ∃ (k l m : ℤ), w = k * a + l * b + m * c ∨ w = k * -a + l * -b + m * -c :=
  sorry

end can_measure_all_weights_l274_274519


namespace horner_v3_value_l274_274189

-- Define constants
def a_n : ℤ := 2 -- Leading coefficient of x^5
def a_3 : ℤ := -3 -- Coefficient of x^3
def a_2 : ℤ := 5 -- Coefficient of x^2
def a_0 : ℤ := -4 -- Constant term
def x : ℤ := 2 -- Given value of x

-- Horner's method sequence for the coefficients
def v_0 : ℤ := a_n -- Initial value v_0
def v_1 : ℤ := v_0 * x -- Calculated as v_0 * x
def v_2 : ℤ := v_1 * x + a_3 -- Calculated as v_1 * x + a_3 (coefficient of x^3)
def v_3 : ℤ := v_2 * x + a_2 -- Calculated as v_2 * x + a_2 (coefficient of x^2)

theorem horner_v3_value : v_3 = 15 := 
by
  -- Formal proof would go here, skipped due to problem specifications
  sorry

end horner_v3_value_l274_274189


namespace calculate_sum_l274_274077

theorem calculate_sum : (-2) + 1 = -1 :=
by 
  sorry

end calculate_sum_l274_274077


namespace cyrus_shots_percentage_l274_274320

theorem cyrus_shots_percentage (total_shots : ℕ) (missed_shots : ℕ) (made_shots : ℕ)
  (h_total : total_shots = 20)
  (h_missed : missed_shots = 4)
  (h_made : made_shots = total_shots - missed_shots) :
  (made_shots / total_shots : ℚ) * 100 = 80 := by
  sorry

end cyrus_shots_percentage_l274_274320


namespace remove_one_and_average_l274_274666

theorem remove_one_and_average (l : List ℕ) (n : ℕ) (avg : ℚ) :
  l = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15] →
  avg = 8.5 →
  (l.sum - n : ℚ) = 14 * avg →
  n = 1 :=
by
  intros hlist havg hsum
  sorry

end remove_one_and_average_l274_274666


namespace value_of_f_g3_l274_274734

def g (x : ℝ) : ℝ := 4 * x - 5
def f (x : ℝ) : ℝ := 6 * x + 11

theorem value_of_f_g3 : f (g 3) = 53 := by
  sorry

end value_of_f_g3_l274_274734


namespace tank_capacity_l274_274410

variable (C : ℝ)

noncomputable def leak_rate := C / 6 -- litres per hour
noncomputable def inlet_rate := 6 * 60 -- litres per hour
noncomputable def net_emptying_rate := C / 12 -- litres per hour

theorem tank_capacity : 
  (360 - leak_rate C = net_emptying_rate C) → 
  C = 1440 :=
by 
  sorry

end tank_capacity_l274_274410


namespace find_constant_term_l274_274579

theorem find_constant_term (x y C : ℤ) 
    (h1 : 5 * x + y = 19) 
    (h2 : 3 * x + 2 * y = 10) 
    (h3 : C = x + 3 * y) 
    : C = 1 := 
by 
  sorry

end find_constant_term_l274_274579


namespace Patricia_read_21_books_l274_274265

theorem Patricia_read_21_books
  (Candice_books Amanda_books Kara_books Patricia_books : ℕ)
  (h1 : Candice_books = 18)
  (h2 : Candice_books = 3 * Amanda_books)
  (h3 : Kara_books = Amanda_books / 2)
  (h4 : Patricia_books = 7 * Kara_books) :
  Patricia_books = 21 :=
by
  sorry

end Patricia_read_21_books_l274_274265


namespace min_value_of_expression_l274_274998

theorem min_value_of_expression (a b : ℝ) (h_pos_b : 0 < b) (h_eq : 2 * a + b = 1) : 
  42 + b^2 + 1 / (a * b) ≥ 17 / 2 := 
sorry

end min_value_of_expression_l274_274998


namespace belize_homes_l274_274511

theorem belize_homes (H : ℝ) 
  (h1 : (3 / 5) * (3 / 4) * H = 240) : 
  H = 400 :=
sorry

end belize_homes_l274_274511


namespace initial_birds_l274_274520

-- Define the initial number of birds (B) and the fact that 13 more birds flew up to the tree
-- Define that the total number of birds after 13 more birds joined is 42
theorem initial_birds (B : ℕ) (h : B + 13 = 42) : B = 29 :=
by
  sorry

end initial_birds_l274_274520


namespace trapezium_area_l274_274100

theorem trapezium_area (a b h : ℝ) (ha : a = 20) (hb : b = 18) (hh : h = 15) : 
  (1/2) * (a + b) * h = 285 :=
by
  rw [ha, hb, hh]
  norm_num

end trapezium_area_l274_274100


namespace multiple_of_k_l274_274139

theorem multiple_of_k (k : ℕ) (m : ℕ) (h₁ : 7 ^ k = 2) (h₂ : 7 ^ (m * k + 2) = 784) : m = 2 :=
sorry

end multiple_of_k_l274_274139


namespace int_solution_count_l274_274307

theorem int_solution_count :
  let count_solutions (eq : ℤ → ℤ → Bool) : Nat :=
    Finset.card (Finset.filter (λ ⟨(y, x)⟩, eq y x) Finset.univ.prod Finset.univ)
  count_solutions (λ y x, 6 * y^2 + 3 * y * x + x + 2 * y + 180 = 0) = 6 :=
sorry

end int_solution_count_l274_274307


namespace minimum_packs_needed_l274_274007

theorem minimum_packs_needed (n : ℕ) :
  (∃ x y z : ℕ, 30 * x + 18 * y + 9 * z = 120 ∧ x + y + z = n ∧ x ≥ 2 ∧ z' = if x ≥ 2 then z + 1 else z) → n = 4 := 
by
  sorry

end minimum_packs_needed_l274_274007


namespace slope_of_given_line_l274_274381

theorem slope_of_given_line : ∀ (x y : ℝ), (4 / x + 5 / y = 0) → (y = (-5 / 4) * x) := 
by 
  intros x y h
  sorry

end slope_of_given_line_l274_274381


namespace range_of_x_l274_274868

open Real

theorem range_of_x (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 2 / a + 1 / b = 1) :
  a + b ≥ 3 + 2 * sqrt 2 :=
sorry

end range_of_x_l274_274868


namespace like_terms_exponent_l274_274731

theorem like_terms_exponent (m n : ℤ) (h₁ : n = 2) (h₂ : m = 1) : m - n = -1 :=
by
  sorry

end like_terms_exponent_l274_274731


namespace minimum_width_for_fence_l274_274497

theorem minimum_width_for_fence (w : ℝ) (h : 0 ≤ 20) : 
  (w * (w + 20) ≥ 150) → w ≥ 10 :=
by
  sorry

end minimum_width_for_fence_l274_274497


namespace find_set_of_points_B_l274_274130

noncomputable def is_incenter (A B C I : Point) : Prop :=
  -- define the incenter condition
  sorry

noncomputable def angle_less_than (A B C : Point) (α : ℝ) : Prop :=
  -- define the condition that all angles of triangle ABC are less than α
  sorry

theorem find_set_of_points_B (A I : Point) (α : ℝ) (hα1 : 60 < α) (hα2 : α < 90) :
  ∃ B : Point, ∃ C : Point,
    is_incenter A B C I ∧ angle_less_than A B C α :=
by
  -- The proof will go here
  sorry

end find_set_of_points_B_l274_274130


namespace range_of_m_l274_274452

open Set

variable {m x : ℝ}

def A (m : ℝ) : Set ℝ := {x | m + 1 ≤ x ∧ x ≤ 2m - 1}
def B : Set ℝ := {x | x^2 - 7 * x + 10 ≤ 0}

theorem range_of_m (m : ℝ) (hA : ∀ x, x ∈ A m → x ∈ B) : 2 ≤ m ∧ m ≤ 3 :=
by 
  sorry

end range_of_m_l274_274452


namespace determine_a_l274_274981

theorem determine_a (a : ℚ) (x : ℚ) : 
  (∃ r s : ℚ, (r*x + s)^2 = a*x^2 + 18*x + 16) → 
  a = 81/16 := 
sorry

end determine_a_l274_274981


namespace calculate_expression_l274_274692

theorem calculate_expression : (235 - 2 * 3 * 5) * 7 / 5 = 287 := 
by
  sorry

end calculate_expression_l274_274692


namespace triangle_constructibility_l274_274361

noncomputable def constructible_triangle (a b w_c : ℝ) : Prop :=
  (2 * a * b) / (a + b) > w_c

theorem triangle_constructibility {a b w_c : ℝ} (h : (a > 0) ∧ (b > 0) ∧ (w_c > 0)) :
  constructible_triangle a b w_c ↔ True :=
by
  sorry

end triangle_constructibility_l274_274361


namespace outer_term_in_proportion_l274_274885

theorem outer_term_in_proportion (a b x : ℝ) (h_ab : a * b = 1) (h_x : x = 0.2) : b = 5 :=
by
  sorry

end outer_term_in_proportion_l274_274885


namespace tangent_perpendicular_point_l274_274123

open Real

noncomputable def f (x : ℝ) : ℝ := exp x - (1 / 2) * x^2

theorem tangent_perpendicular_point :
  ∃ x0, (f x0 = 1) ∧ (x0 = 0) :=
sorry

end tangent_perpendicular_point_l274_274123


namespace find_y_l274_274106

-- Define vectors as tuples
def vector_1 : ℝ × ℝ := (3, 4)
def vector_2 (y : ℝ) : ℝ × ℝ := (y, -5)

-- Define dot product
def dot_product (u v : ℝ × ℝ) : ℝ :=
  u.1 * v.1 + u.2 * v.2

-- Condition for orthogonality
def orthogonal (u v : ℝ × ℝ) : Prop :=
  dot_product u v = 0

-- The theorem we want to prove
theorem find_y (y : ℝ) :
  orthogonal vector_1 (vector_2 y) → y = (20 / 3) :=
by
  sorry

end find_y_l274_274106


namespace third_smallest_four_digit_in_pascals_triangle_l274_274195

-- First we declare the conditions
def every_positive_integer_in_pascals_triangle : Prop :=
  ∀ n : ℕ, ∃ a b : ℕ, nat.choose b a = n

def number_1000_in_pascals_triangle : Prop :=
  ∃ b : ℕ, nat.choose b 1 = 1000

def number_1001_in_pascals_triangle : Prop :=
  ∃ b : ℕ, nat.choose b 1 = 1001

def number_1002_in_pascals_triangle : Prop :=
  ∃ b : ℕ, nat.choose b 2 = 1002

-- Define the theorem
theorem third_smallest_four_digit_in_pascals_triangle
  (h1 : every_positive_integer_in_pascals_triangle)
  (h2 : number_1000_in_pascals_triangle)
  (h3 : number_1001_in_pascals_triangle)
  (h4 : number_1002_in_pascals_triangle) :
  (∃ n : ℕ, nat.choose n 2 = 1002 ∧ (∀ k, k < n → nat.choose k 1 < 1002)) := sorry

end third_smallest_four_digit_in_pascals_triangle_l274_274195


namespace fish_population_l274_274321

theorem fish_population (N : ℕ) (hN : (2 / 50 : ℚ) = (30 / N : ℚ)) : N = 750 :=
by
  sorry

end fish_population_l274_274321


namespace real_part_of_i_squared_times_1_plus_i_l274_274935

noncomputable def imaginary_unit : ℂ := Complex.I

theorem real_part_of_i_squared_times_1_plus_i :
  (Complex.re (imaginary_unit^2 * (1 + imaginary_unit))) = -1 :=
by
  sorry

end real_part_of_i_squared_times_1_plus_i_l274_274935


namespace tan_half_sum_l274_274165

variable (p q : ℝ)

-- Given conditions
def cos_condition : Prop := (Real.cos p + Real.cos q = 1 / 3)
def sin_condition : Prop := (Real.sin p + Real.sin q = 4 / 9)

-- Prove the target expression
theorem tan_half_sum (h1 : cos_condition p q) (h2 : sin_condition p q) : 
  Real.tan ((p + q) / 2) = 4 / 3 :=
sorry

-- For better readability, I included variable declarations and definitions separately

end tan_half_sum_l274_274165


namespace identify_1000g_weight_l274_274188

-- Define the masses of the weights
def masses : List ℕ := [1000, 1001, 1002, 1004, 1007]

-- The statement that needs to be proven
theorem identify_1000g_weight (masses : List ℕ) (h : masses = [1000, 1001, 1002, 1004, 1007]) :
  ∃ w, w ∈ masses ∧ w = 1000 ∧ by sorry :=
sorry

end identify_1000g_weight_l274_274188


namespace other_root_of_quadratic_l274_274172

variable (p : ℝ)

theorem other_root_of_quadratic (h1: 3 * (-2) * r_2 = -6) : r_2 = 1 :=
by
  sorry

end other_root_of_quadratic_l274_274172


namespace inequality_proof_l274_274491

theorem inequality_proof
  (x1 x2 x3 x4 x5 : ℝ)
  (hx1 : 0 < x1)
  (hx2 : 0 < x2)
  (hx3 : 0 < x3)
  (hx4 : 0 < x4)
  (hx5 : 0 < x5) :
  x1^2 + x2^2 + x3^2 + x4^2 + x5^2 ≥ x1 * (x2 + x3 + x4 + x5) :=
by
  sorry

end inequality_proof_l274_274491


namespace irrational_sqrt_3_l274_274420

theorem irrational_sqrt_3 : ¬ ∃ (q : ℚ), (q : ℝ) = Real.sqrt 3 := by
  sorry

end irrational_sqrt_3_l274_274420


namespace smallest_value_of_n_l274_274645

theorem smallest_value_of_n (a b c m n : ℕ) (h1 : a ≥ b) (h2 : b ≥ c) (h3 : a + b + c = 2010) (h4 : (a! * b! * c!) = m * 10 ^ n) : ∃ n, n = 500 := 
sorry

end smallest_value_of_n_l274_274645


namespace least_positive_integer_congruences_l274_274052

theorem least_positive_integer_congruences :
  ∃ n : ℕ, 
    n > 0 ∧ 
    (n % 4 = 1) ∧ 
    (n % 5 = 2) ∧ 
    (n % 6 = 3) ∧ 
    (n = 57) :=
by
  sorry

end least_positive_integer_congruences_l274_274052


namespace regular_price_adult_ticket_l274_274991

theorem regular_price_adult_ticket : 
  ∀ (concessions_cost_children cost_adult1 cost_adult2 cost_adult3 cost_adult4 cost_adult5
       ticket_cost_child cost_discount1 cost_discount2 cost_discount3 total_cost : ℝ),
  (concessions_cost_children = 3) → 
  (cost_adult1 = 5) → 
  (cost_adult2 = 6) → 
  (cost_adult3 = 7) → 
  (cost_adult4 = 4) → 
  (cost_adult5 = 9) → 
  (ticket_cost_child = 7) → 
  (cost_discount1 = 3) → 
  (cost_discount2 = 2) → 
  (cost_discount3 = 1) → 
  (total_cost = 139) → 
  (∀ A : ℝ, total_cost = 
    (2 * concessions_cost_children + cost_adult1 + cost_adult2 + cost_adult3 + cost_adult4 + cost_adult5) + 
    (2 * ticket_cost_child + (2 * A + (A - cost_discount1) + (A - cost_discount2) + (A - cost_discount3))) → 
    5 * A - 6 = 88 →
    A = 18.80) :=
by
  intros
  sorry

end regular_price_adult_ticket_l274_274991


namespace andy_older_than_rahim_l274_274744

-- Define Rahim's current age
def Rahim_current_age : ℕ := 6

-- Define Andy's age in 5 years
def Andy_age_in_5_years : ℕ := 2 * Rahim_current_age

-- Define Andy's current age
def Andy_current_age : ℕ := Andy_age_in_5_years - 5

-- Define the difference in age between Andy and Rahim right now
def age_difference : ℕ := Andy_current_age - Rahim_current_age

-- Theorem stating the age difference between Andy and Rahim right now is 1 year
theorem andy_older_than_rahim : age_difference = 1 :=
by
  -- Proof is skipped
  sorry

end andy_older_than_rahim_l274_274744


namespace intersecting_lines_sum_c_d_l274_274781

theorem intersecting_lines_sum_c_d 
  (c d : ℚ)
  (h1 : 2 = 1 / 5 * (3 : ℚ) + c)
  (h2 : 3 = 1 / 5 * (2 : ℚ) + d) : 
  c + d = 4 :=
by sorry

end intersecting_lines_sum_c_d_l274_274781


namespace student_failed_by_40_marks_l274_274828

theorem student_failed_by_40_marks (total_marks : ℕ) (passing_percentage : ℝ) (marks_obtained : ℕ) (h1 : total_marks = 500) (h2 : passing_percentage = 33) (h3 : marks_obtained = 125) :
  ((passing_percentage / 100) * total_marks - marks_obtained : ℝ) = 40 :=
sorry

end student_failed_by_40_marks_l274_274828


namespace certain_positive_integer_value_l274_274318

theorem certain_positive_integer_value :
  ∃ (i m p : ℕ), (x = 2 ^ i * 3 ^ 2 * 5 ^ m * 7 ^ p) ∧ (i + 2 + m + p = 11) :=
by
  let x := 40320 -- 8!
  sorry

end certain_positive_integer_value_l274_274318


namespace even_goals_more_likely_l274_274681

theorem even_goals_more_likely (p₁ : ℝ) (q₁ : ℝ) 
  (h₁ : q₁ = 1 - p₁)
  (independent_halves : (p₁ * p₁ + q₁ * q₁) > (2 * p₁ * q₁)) :
  (p₁ * p₁ + q₁ * q₁) > (1 - (p₁ * p₁ + q₁ * q₁)) :=
by
  sorry

end even_goals_more_likely_l274_274681


namespace suresh_completion_time_l274_274921

theorem suresh_completion_time (S : ℕ) 
  (ashu_time : ℕ := 30) 
  (suresh_work_time : ℕ := 9) 
  (ashu_remaining_time : ℕ := 12) 
  (ashu_fraction : ℚ := ashu_remaining_time / ashu_time) :
  (suresh_work_time / S + ashu_fraction = 1) → S = 15 :=
by
  intro h
  -- Proof here
  sorry

end suresh_completion_time_l274_274921


namespace find_x_l274_274678

variable (x : ℝ)

theorem find_x (h : 0.60 * x = (1/3) * x + 110) : x = 412.5 :=
sorry

end find_x_l274_274678


namespace least_positive_integer_condition_l274_274032

theorem least_positive_integer_condition
  (a : ℤ) (ha1 : a % 4 = 1) (ha2 : a % 5 = 2) (ha3 : a % 6 = 3) :
  a > 0 → a = 57 :=
by
  intro ha_pos
  -- Proof omitted for brevity
  sorry

end least_positive_integer_condition_l274_274032


namespace age_problem_l274_274409

theorem age_problem (age x : ℕ) (h : age = 64) :
  (1 / 2 : ℝ) * (8 * (age + x) - 8 * (age - 8)) = age → x = 8 :=
by
  sorry

end age_problem_l274_274409


namespace number_of_men_in_first_group_l274_274177

-- Define the conditions as hypotheses in Lean
def work_completed_in_25_days (x : ℕ) : Prop := x * 25 * (1 : ℚ) / (25 * x) = (1 : ℚ)
def twenty_men_complete_in_15_days : Prop := 20 * 15 * (1 : ℚ) / 15 = (1 : ℚ)

-- Define the theorem to prove the number of men in the first group
theorem number_of_men_in_first_group (x : ℕ) (h1 : work_completed_in_25_days x)
  (h2 : twenty_men_complete_in_15_days) : x = 20 :=
  sorry

end number_of_men_in_first_group_l274_274177


namespace repair_time_and_earnings_l274_274158

-- Definitions based on given conditions
def cars : ℕ := 10
def cars_repair_50min : ℕ := 6
def repair_time_50min : ℕ := 50 -- minutes per car
def longer_percentage : ℕ := 80 -- 80% longer for the remaining cars
def wage_per_hour : ℕ := 30 -- dollars per hour

-- Remaining cars to repair
def remaining_cars : ℕ := cars - cars_repair_50min

-- Calculate total repair time for each type of cars and total repair time
def repair_time_remaining_cars : ℕ := repair_time_50min + (repair_time_50min * longer_percentage) / 100
def total_repair_time : ℕ := (cars_repair_50min * repair_time_50min) + (remaining_cars * repair_time_remaining_cars)

-- Convert total repair time from minutes to hours
def total_repair_hours : ℕ := total_repair_time / 60

-- Calculate total earnings
def total_earnings : ℕ := wage_per_hour * total_repair_hours

-- The theorem to be proved: total_repair_time == 660 and total_earnings == 330
theorem repair_time_and_earnings :
  total_repair_time = 660 ∧ total_earnings = 330 := by
  sorry

end repair_time_and_earnings_l274_274158


namespace avg_temp_correct_l274_274824

-- Defining the temperatures for each day from March 1st to March 5th
def day_1_temp := 55.0
def day_2_temp := 59.0
def day_3_temp := 60.0
def day_4_temp := 57.0
def day_5_temp := 64.0

-- Calculating the average temperature
def avg_temp := (day_1_temp + day_2_temp + day_3_temp + day_4_temp + day_5_temp) / 5.0

-- Proving that the average temperature equals 59.0°F
theorem avg_temp_correct : avg_temp = 59.0 := sorry

end avg_temp_correct_l274_274824


namespace expected_participants_in_2005_l274_274326

open Nat

def initial_participants : ℕ := 500
def annual_increase_rate : ℚ := 1.2
def num_years : ℕ := 5
def expected_participants_2005 : ℚ := 1244

theorem expected_participants_in_2005 :
  (initial_participants : ℚ) * annual_increase_rate ^ num_years = expected_participants_2005 := by
  sorry

end expected_participants_in_2005_l274_274326


namespace completely_factored_form_l274_274266

theorem completely_factored_form (x : ℤ) :
  (12 * x ^ 3 + 95 * x - 6) - (-3 * x ^ 3 + 5 * x - 6) = 15 * x * (x ^ 2 + 6) :=
by
  sorry

end completely_factored_form_l274_274266


namespace jennifer_initial_pears_l274_274157

def initialPears (P: ℕ) : Prop := (P + 20 + 2 * P - 6 = 44)

theorem jennifer_initial_pears (P: ℕ) (h : initialPears P) : P = 10 := by
  sorry

end jennifer_initial_pears_l274_274157


namespace find_x2_plus_y2_l274_274434

theorem find_x2_plus_y2 
  (x y : ℕ) 
  (hx : x > 0) 
  (hy : y > 0) 
  (h1 : x * y + x + y = 117) 
  (h2 : x^2 * y + x * y^2 = 1512) : 
  x^2 + y^2 = 549 := 
sorry

end find_x2_plus_y2_l274_274434


namespace arithmetic_sequence_problem_l274_274753

variable {α : Type*} [LinearOrderedRing α]

theorem arithmetic_sequence_problem
  (a : ℕ → α)
  (h : ∀ n, a (n + 1) = a n + (a 1 - a 0))
  (h_seq : a 5 + a 6 + a 7 + a 8 + a 9 = 450) :
  a 3 + a 11 = 180 :=
sorry

end arithmetic_sequence_problem_l274_274753


namespace trapezium_area_l274_274097

-- Define the lengths of the parallel sides and the distance between them
def side_a : ℝ := 20
def side_b : ℝ := 18
def height : ℝ := 15

-- Define the formula for the area of a trapezium
def area_trapezium (a b h : ℝ) : ℝ :=
  (1 / 2) * (a + b) * h

-- State the theorem
theorem trapezium_area :
  area_trapezium side_a side_b height = 285 :=
by
  sorry

end trapezium_area_l274_274097


namespace largest_multiple_of_11_lt_neg150_l274_274528

theorem largest_multiple_of_11_lt_neg150 : ∃ (x : ℤ), (x % 11 = 0) ∧ (x < -150) ∧ (∀ y : ℤ, y % 11 = 0 → y < -150 → y ≤ x) ∧ x = -154 :=
by
  sorry

end largest_multiple_of_11_lt_neg150_l274_274528


namespace triangle_hypotenuse_segments_l274_274504

theorem triangle_hypotenuse_segments :
  ∀ (x : ℝ) (BC AC : ℝ),
  BC / AC = 3 / 7 →
  ∃ (h : ℝ) (BD AD : ℝ),
    h = 42 ∧
    BD * AD = h^2 ∧
    BD / AD = 9 / 49 ∧
    BD = 18 ∧
    AD = 98 :=
by
  sorry

end triangle_hypotenuse_segments_l274_274504


namespace height_of_water_in_cylinder_l274_274556

theorem height_of_water_in_cylinder
  (r_cone : ℝ) (h_cone : ℝ) (r_cylinder : ℝ) (V_cone : ℝ) (V_cylinder : ℝ) (h_cylinder : ℝ) :
  r_cone = 15 → h_cone = 25 → r_cylinder = 20 →
  V_cone = (1 / 3) * π * r_cone^2 * h_cone →
  V_cylinder = V_cone → V_cylinder = π * r_cylinder^2 * h_cylinder →
  h_cylinder = 4.7 :=
by
  intros r_cone_eq h_cone_eq r_cylinder_eq V_cone_eq V_cylinder_eq volume_eq
  sorry

end height_of_water_in_cylinder_l274_274556


namespace reduced_price_after_exchange_rate_fluctuation_l274_274819

-- Definitions based on conditions
variables (P : ℝ) -- Original price per kg

def reduced_price_per_kg : ℝ := 0.9 * P

axiom six_kg_costs_900 : 6 * reduced_price_per_kg P = 900

-- Additional conditions
def exchange_rate_factor : ℝ := 1.02

-- Question restated as the theorem to prove
theorem reduced_price_after_exchange_rate_fluctuation : 
  ∃ P : ℝ, reduced_price_per_kg P * exchange_rate_factor = 153 :=
sorry

end reduced_price_after_exchange_rate_fluctuation_l274_274819


namespace find_pqr_eq_1680_l274_274008

theorem find_pqr_eq_1680
  {p q r : ℤ} (hpqz : p ≠ 0) (hqqz : q ≠ 0) (hrqz : r ≠ 0)
  (h_sum : p + q + r = 30)
  (h_cond : (1:ℚ) / p + (1:ℚ) / q + (1:ℚ) / r + 390 / (p * q * r) = 1) :
  p * q * r = 1680 :=
sorry

end find_pqr_eq_1680_l274_274008


namespace decreasing_function_solution_set_l274_274482

theorem decreasing_function_solution_set {f : ℝ → ℝ} (h : ∀ x y, x < y → f y < f x) :
  {x : ℝ | f 2 < f (2*x + 1)} = {x : ℝ | x < 1/2} :=
by
  sorry

end decreasing_function_solution_set_l274_274482


namespace mary_total_money_l274_274171

def num_quarters : ℕ := 21
def quarters_worth : ℚ := 0.25
def dimes_worth : ℚ := 0.10

def num_dimes (Q : ℕ) : ℕ := (Q - 7) / 2

def total_money (Q : ℕ) (D : ℕ) : ℚ :=
  Q * quarters_worth + D * dimes_worth

theorem mary_total_money : 
  total_money num_quarters (num_dimes num_quarters) = 5.95 := 
by
  sorry

end mary_total_money_l274_274171


namespace quadratic_rewrite_constants_l274_274978

theorem quadratic_rewrite_constants (a b c : ℤ) 
    (h1 : -4 * (x - 2) ^ 2 + 144 = -4 * x ^ 2 + 16 * x + 128) 
    (h2 : a = -4)
    (h3 : b = -2)
    (h4 : c = 144) 
    : a + b + c = 138 := by
  sorry

end quadratic_rewrite_constants_l274_274978


namespace f_zero_f_increasing_on_negative_l274_274869

noncomputable def f : ℝ → ℝ := sorry
variable {x : ℝ}

-- Assume f is an odd function
axiom odd_f : ∀ x, f (-x) = -f x

-- Assume f is increasing on (0, +∞)
axiom increasing_f_on_positive :
  ∀ ⦃x₁ x₂⦄, 0 < x₁ → x₁ < x₂ → f x₁ < f x₂

-- Prove that f(0) = 0
theorem f_zero : f 0 = 0 := sorry

-- Prove that f is increasing on (-∞, 0)
theorem f_increasing_on_negative :
  ∀ ⦃x₁ x₂⦄, x₁ < x₂ → x₂ < 0 → f x₁ < f x₂ := sorry

end f_zero_f_increasing_on_negative_l274_274869


namespace car_distance_covered_by_car_l274_274817

theorem car_distance_covered_by_car
  (V : ℝ)                               -- Initial speed of the car
  (D : ℝ)                               -- Distance covered by the car
  (h1 : D = V * 6)                      -- The car takes 6 hours to cover the distance at speed V
  (h2 : D = 56 * 9)                     -- The car takes 9 hours to cover the distance at speed 56
  : D = 504 :=                          -- Prove that the distance D is 504 kilometers
by
  sorry

end car_distance_covered_by_car_l274_274817


namespace stratified_sampling_l274_274022

theorem stratified_sampling (total_students : ℕ) (ratio_grade1 ratio_grade2 ratio_grade3 : ℕ) (sample_size : ℕ) (h_ratio : ratio_grade1 = 3 ∧ ratio_grade2 = 3 ∧ ratio_grade3 = 4) (h_sample_size : sample_size = 50) : 
  (ratio_grade2 / (ratio_grade1 + ratio_grade2 + ratio_grade3) : ℚ) * sample_size = 15 := 
by
  sorry

end stratified_sampling_l274_274022


namespace prime_if_and_only_if_digit_is_nine_l274_274655

theorem prime_if_and_only_if_digit_is_nine (B : ℕ) (h : 0 ≤ B ∧ B < 10) :
  Prime (303200 + B) ↔ B = 9 := 
by
  sorry

end prime_if_and_only_if_digit_is_nine_l274_274655


namespace negation_proposition_false_l274_274649

theorem negation_proposition_false : 
  (¬ ∃ x : ℝ, x^2 + 2 ≤ 0) :=
by sorry

end negation_proposition_false_l274_274649


namespace ratio_of_ab_l274_274630

noncomputable theory
open Complex

theorem ratio_of_ab (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0)
  (h : ∃ c : ℝ, (3 - 8 * I) * (a + b * I) = c * I) : a / b = -8 / 3 :=
by sorry

end ratio_of_ab_l274_274630


namespace rate_of_first_batch_l274_274972

theorem rate_of_first_batch (x : ℝ) 
  (cost_second_batch : ℝ := 20 * 14.25)
  (total_cost : ℝ := 30 * x + 285)
  (weight_mixture : ℝ := 30 + 20)
  (selling_price_per_kg : ℝ := 15.12) :
  (total_cost * 1.20 / weight_mixture = selling_price_per_kg) → x = 11.50 :=
by
  sorry

end rate_of_first_batch_l274_274972


namespace machines_working_time_l274_274939

theorem machines_working_time (y: ℝ) 
  (h1 : y + 8 > 0)  -- condition for time taken by S
  (h2 : y + 2 > 0)  -- condition for time taken by T
  (h3 : 2 * y > 0)  -- condition for time taken by U
  : (1 / (y + 8) + 1 / (y + 2) + 1 / (2 * y) = 1 / y) ↔ (y = 3 / 2) := 
by
  have h4 : y ≠ 0 := by linarith [h1, h2, h3]
  sorry

end machines_working_time_l274_274939


namespace compare_sqrt_expression_l274_274694

theorem compare_sqrt_expression : 2 * Real.sqrt 3 < 3 * Real.sqrt 2 := 
sorry

end compare_sqrt_expression_l274_274694


namespace no_n_exists_l274_274982

theorem no_n_exists (n : ℕ) : ¬ ∃ n : ℕ, (n^2 + 6 * n + 2019) % 100 = 0 :=
by {
  sorry
}

end no_n_exists_l274_274982


namespace surveys_on_tuesday_l274_274252

theorem surveys_on_tuesday
  (num_surveys_monday: ℕ) -- number of surveys Bart completed on Monday
  (earnings_monday: ℕ) -- earning per survey on Monday
  (total_earnings: ℕ) -- total earnings over the two days
  (earnings_per_survey: ℕ) -- earnings Bart gets per survey
  (monday_earnings_eq : earnings_monday = num_surveys_monday * earnings_per_survey)
  (total_earnings_eq : total_earnings = earnings_monday + (8 : ℕ))
  (earnings_per_survey_eq : earnings_per_survey = 2)
  : ((8 : ℕ) / earnings_per_survey = 4) := sorry

end surveys_on_tuesday_l274_274252


namespace number_of_integers_l274_274436

theorem number_of_integers (n : ℤ) : 
  (16 < n^2) → (n^2 < 121) → n = -10 ∨ n = -9 ∨ n = -8 ∨ n = -7 ∨ n = -6 ∨ n = -5 ∨ n = 5 ∨ n = 6 ∨ n = 7 ∨ n = 8 ∨ n = 9 ∨ n = 10 := 
by
  sorry

end number_of_integers_l274_274436


namespace solution_correct_l274_274354

-- Define the conditions
def abs_inequality (x : ℝ) : Prop := abs (x - 3) + abs (x + 4) < 8
def quadratic_eq (x : ℝ) : Prop := x^2 - x - 12 = 0

-- Define the main statement to prove
theorem solution_correct : ∃ (x : ℝ), abs_inequality x ∧ quadratic_eq x ∧ x = -3 := sorry

end solution_correct_l274_274354


namespace same_function_l274_274832

noncomputable def f (x : ℝ) : ℝ := x
noncomputable def g (t : ℝ) : ℝ := (t^3 + t) / (t^2 + 1)

theorem same_function : ∀ x : ℝ, f x = g x :=
by sorry

end same_function_l274_274832


namespace pascal_third_smallest_four_digit_number_l274_274210

theorem pascal_third_smallest_four_digit_number :
  (∃ n k: ℕ, 1000 ≤ binom n k ∧ binom n k ≤ 1002) ∧ ¬(∃ n k: ℕ, 1003 ≤ binom n k) :=
sorry

end pascal_third_smallest_four_digit_number_l274_274210


namespace linear_function_solution_l274_274400

theorem linear_function_solution (f : ℝ → ℝ) (h1 : ∀ x, f (f x) = 16 * x - 15) :
  (∀ x, f x = 4 * x - 3) ∨ (∀ x, f x = -4 * x + 5) :=
sorry

end linear_function_solution_l274_274400


namespace area_is_12_5_l274_274754

-- Define the triangle XYZ
structure Triangle := 
  (X Y Z : Type) 
  (XZ YZ : ℝ) 
  (angleX angleY angleZ : ℝ)

-- Provided conditions in the problem
def triangleXYZ : Triangle := {
  X := ℝ, 
  Y := ℝ, 
  Z := ℝ, 
  XZ := 5,
  YZ := 5,
  angleX := 45,
  angleY := 45,
  angleZ := 90
}

-- Lean statement to prove the area of triangle XYZ
theorem area_is_12_5 (t : Triangle) 
  (h1 : t.angleZ = 90)
  (h2 : t.angleX = 45)
  (h3 : t.angleY = 45)
  (h4 : t.XZ = 5)
  (h5 : t.YZ = 5) : 
  (1/2 * t.XZ * t.YZ) = 12.5 :=
sorry

end area_is_12_5_l274_274754


namespace find_slope_of_intersecting_line_l274_274486

-- Define the conditions
def line_p (x : ℝ) : ℝ := 2 * x + 3
def line_q (x : ℝ) (m : ℝ) : ℝ := m * x + 1

-- Define the point of intersection
def intersection_point : ℝ × ℝ := (4, 11)

-- Prove that the slope m of line q such that both lines intersect at (4, 11) is 2.5
theorem find_slope_of_intersecting_line (m : ℝ) :
  line_q 4 m = 11 → m = 2.5 :=
by
  intro h
  sorry

end find_slope_of_intersecting_line_l274_274486


namespace find_number_l274_274273

theorem find_number (x : ℝ) (h : ((x / 8) + 8 - 30) * 6 = 12) : x = 192 :=
sorry

end find_number_l274_274273


namespace total_fish_weight_is_25_l274_274432

-- Define the conditions and the problem
def num_trout : ℕ := 4
def weight_trout : ℝ := 2
def num_catfish : ℕ := 3
def weight_catfish : ℝ := 1.5
def num_bluegills : ℕ := 5
def weight_bluegill : ℝ := 2.5

-- Calculate the total weight of each type of fish
def total_weight_trout : ℝ := num_trout * weight_trout
def total_weight_catfish : ℝ := num_catfish * weight_catfish
def total_weight_bluegills : ℝ := num_bluegills * weight_bluegill

-- Calculate the total weight of all fish
def total_weight_fish : ℝ := total_weight_trout + total_weight_catfish + total_weight_bluegills

-- Statement to be proved
theorem total_fish_weight_is_25 : total_weight_fish = 25 := by
  sorry

end total_fish_weight_is_25_l274_274432


namespace product_value_l274_274441

noncomputable def product_of_integers (A B C D : ℕ) : ℕ :=
  A * B * C * D

theorem product_value :
  ∃ (A B C D : ℕ), A + B + C + D = 72 ∧ 
                    A + 2 = B - 2 ∧ 
                    A + 2 = C * 2 ∧ 
                    A + 2 = D / 2 ∧ 
                    product_of_integers A B C D = 64512 :=
by
  sorry

end product_value_l274_274441


namespace sandy_total_spent_l274_274913

def shorts_price : ℝ := 13.99
def shirt_price : ℝ := 12.14
def jacket_price : ℝ := 7.43
def total_spent : ℝ := shorts_price + shirt_price + jacket_price

theorem sandy_total_spent : total_spent = 33.56 :=
by
  sorry

end sandy_total_spent_l274_274913


namespace reciprocal_and_fraction_l274_274601

theorem reciprocal_and_fraction (a b : ℝ) (h1 : a * b = 1) (h2 : (2/5) * a = 20) : 
  b = (1/a) ∧ (1/3) * a = (50/3) := 
by 
  sorry

end reciprocal_and_fraction_l274_274601


namespace ratio_a_d_l274_274946

theorem ratio_a_d 
  (a b c d : ℕ) 
  (h1 : a / b = 1 / 4) 
  (h2 : b / c = 13 / 9) 
  (h3 : c / d = 5 / 13) : 
  a / d = 5 / 36 :=
sorry

end ratio_a_d_l274_274946


namespace find_a4_in_geometric_seq_l274_274888

variable {q : ℝ} -- q is the common ratio of the geometric sequence

noncomputable def geometric_seq (q : ℝ) (n : ℕ) : ℝ := 16 * q ^ (n - 1)

theorem find_a4_in_geometric_seq (h1 : geometric_seq q 1 = 16)
  (h2 : geometric_seq q 6 = 2 * geometric_seq q 5 * geometric_seq q 7) :
  geometric_seq q 4 = 2 := 
  sorry

end find_a4_in_geometric_seq_l274_274888


namespace M_is_correct_l274_274117

def A : Set ℕ := {1, 2, 3}
def B : Set ℕ := {x | x > 2}

def M := {x | x ∈ A ∧ x ∉ B}

theorem M_is_correct : M = {1, 2} := by
  -- Proof needed here
  sorry

end M_is_correct_l274_274117


namespace largest_cyclic_decimal_l274_274424

def digits_on_circle := [1, 3, 9, 5, 7, 9, 1, 3, 9, 5, 7, 1]

def max_cyclic_decimal : ℕ := sorry

theorem largest_cyclic_decimal :
  max_cyclic_decimal = 957913 :=
sorry

end largest_cyclic_decimal_l274_274424


namespace ramu_profit_percent_l274_274673

noncomputable def profit_percent (purchase_price repair_cost selling_price : ℕ) : ℕ :=
  let total_cost := purchase_price + repair_cost
  let profit := selling_price - total_cost
  (profit * 100) / total_cost

theorem ramu_profit_percent :
  profit_percent 42000 13000 64900 = 18 := by
  sorry

end ramu_profit_percent_l274_274673


namespace father_twice_as_old_in_years_l274_274846

-- Conditions
def father_age : ℕ := 42
def son_age : ℕ := 14
def years : ℕ := 14

-- Proof statement
theorem father_twice_as_old_in_years : (father_age + years) = 2 * (son_age + years) :=
by
  -- Proof content is omitted as per the instruction.
  sorry

end father_twice_as_old_in_years_l274_274846


namespace third_smallest_four_digit_in_pascals_triangle_l274_274218

-- Definitions for Pascal's Triangle and four-digit numbers
def is_in_pascals_triangle (n : ℕ) : Prop :=
  ∃ (r : ℕ) (k : ℕ), r.choose k = n

def is_four_digit (n : ℕ) : Prop :=
  1000 ≤ n ∧ n < 10000

-- Proposition stating the third smallest four-digit number in Pascal's Triangle
theorem third_smallest_four_digit_in_pascals_triangle :
  ∃ (n : ℕ), is_in_pascals_triangle n ∧ is_four_digit n ∧ 
  (∃ m1 m2 m3, is_in_pascals_triangle m1 ∧ is_four_digit m1 ∧ 
                is_in_pascals_triangle m2 ∧ is_four_digit m2 ∧ 
                is_in_pascals_triangle m3 ∧ is_four_digit m3 ∧ 
                1000 ≤ m1 ∧ m1 < 1001 ∧ 1001 ≤ m2 ∧ m2 < 1002 ∧ 1002 ≤ n ∧ n < 1003) ∧ 
  n = 1002 :=
sorry

end third_smallest_four_digit_in_pascals_triangle_l274_274218


namespace sine_tangent_coincide_3_decimal_places_l274_274111

open Real

noncomputable def deg_to_rad (d : ℝ) : ℝ := d * (π / 180)

theorem sine_tangent_coincide_3_decimal_places :
  ∀ θ : ℝ,
    0 ≤ θ ∧ θ ≤ deg_to_rad (4 + 20 / 60) →
    |sin θ - tan θ| < 0.0005 :=
by
  intros θ hθ
  sorry

end sine_tangent_coincide_3_decimal_places_l274_274111


namespace total_order_cost_l274_274698

theorem total_order_cost :
  let c := 2 * 30
  let w := 9 * 15
  let s := 50
  c + w + s = 245 := 
by
  linarith

end total_order_cost_l274_274698


namespace area_of_stripe_l274_274407

def cylindrical_tank.diameter : ℝ := 40
def cylindrical_tank.height : ℝ := 100
def green_stripe.width : ℝ := 4
def green_stripe.revolutions : ℝ := 3

theorem area_of_stripe :
  let diameter := cylindrical_tank.diameter
  let height := cylindrical_tank.height
  let width := green_stripe.width
  let revolutions := green_stripe.revolutions
  let circumference := Real.pi * diameter
  let length := revolutions * circumference
  let area := length * width
  area = 480 * Real.pi := by
  sorry

end area_of_stripe_l274_274407


namespace negation_of_universal_l274_274506

theorem negation_of_universal :
  ¬ (∀ x : ℝ, 2 * x ^ 2 + x - 1 ≤ 0) ↔ ∃ x : ℝ, 2 * x ^ 2 + x - 1 > 0 := 
by 
  sorry

end negation_of_universal_l274_274506


namespace class3_qualifies_l274_274545

/-- Data structure representing a class's tardiness statistics. -/
structure ClassStats where
  mean : ℕ
  median : ℕ
  variance : ℕ
  mode : Option ℕ -- mode is optional because not all classes might have a unique mode.

def class1 : ClassStats := { mean := 3, median := 3, variance := 0, mode := none }
def class2 : ClassStats := { mean := 2, median := 0, variance := 1, mode := none }
def class3 : ClassStats := { mean := 2, median := 0, variance := 2, mode := none }
def class4 : ClassStats := { mean := 0, median := 2, variance := 0, mode := some 2 }

/-- Predicate to check if a class qualifies for the flag, meaning no more than 5 students tardy each day for 5 consecutive days. -/
def qualifies (cs : ClassStats) : Prop :=
  cs.mean = 2 ∧ cs.variance = 2

theorem class3_qualifies : qualifies class3 :=
by
  sorry

end class3_qualifies_l274_274545


namespace difference_between_hit_and_unreleased_l274_274915

-- Define the conditions as constants
def hit_songs : Nat := 25
def top_100_songs : Nat := hit_songs + 10
def total_songs : Nat := 80

-- Define the question, conditional on the definitions above
theorem difference_between_hit_and_unreleased : 
  let unreleased_songs := total_songs - (hit_songs + top_100_songs)
  hit_songs - unreleased_songs = 5 :=
by
  sorry

end difference_between_hit_and_unreleased_l274_274915


namespace tournament_total_players_l274_274746

noncomputable def total_players_in_tournament : ℕ := 24

theorem tournament_total_players (n : ℕ) : 
  let total_players := n + 12 in
  let total_games := (total_players * (total_players - 1)) / 2 in
  let low_players_games := (12 * (12 - 1)) / 2 = 66 in
  let low_players_points := low_players_games * 2 = 132 in
  let high_players_points := (n * (n - 1)) + 132 in
  let total_points := high_players_points = total_games in
  n = 12 → total_players = 24 :=
by 
  intros n;
  let total_players = n + 12;
  let total_games = (total_players * (total_players - 1)) / 2;
  let low_players_games = (12 * (12 - 1)) / 2;
  let low_players_points = low_players_games * 2;
  let high_players_points = (n * (n - 1)) + 132;
  let total_points = high_players_points = total_games;
  assume n_eq_12 : n = 12;
  rw n_eq_12 at ⊢ total_players;
  exact total_players = 24,
  sorry


end tournament_total_players_l274_274746


namespace sum_of_midpoints_y_coordinates_l274_274789

theorem sum_of_midpoints_y_coordinates (d e f : ℝ) (h : d + e + f = 15) : 
  (d + e) / 2 + (d + f) / 2 + (e + f) / 2 = 15 :=
by
  sorry

end sum_of_midpoints_y_coordinates_l274_274789


namespace probability_five_digit_palindrome_div_by_11_l274_274683

noncomputable
def five_digit_palindrome_div_by_11_probability : ℚ :=
  let total_palindromes := 900
  let valid_palindromes := 80
  valid_palindromes / total_palindromes

theorem probability_five_digit_palindrome_div_by_11 :
  five_digit_palindrome_div_by_11_probability = 2 / 25 := by
  sorry

end probability_five_digit_palindrome_div_by_11_l274_274683


namespace grandson_age_l274_274902

theorem grandson_age (M S G : ℕ) (h1 : M = 2 * S) (h2 : S = 2 * G) (h3 : M + S + G = 140) : G = 20 :=
by 
  sorry

end grandson_age_l274_274902


namespace find_principal_amount_l274_274249

-- Define the conditions as constants and assumptions
def monthly_interest_payment : ℝ := 216
def annual_interest_rate : ℝ := 0.09

-- Define the Lean statement to show that the amount of the investment is 28800
theorem find_principal_amount (monthly_payment : ℝ) (annual_rate : ℝ) (P : ℝ) :
  monthly_payment = 216 →
  annual_rate = 0.09 →
  P = 28800 :=
by
  intros 
  sorry

end find_principal_amount_l274_274249


namespace snowfall_on_friday_l274_274487

def snowstorm (snow_wednesday snow_thursday total_snow : ℝ) : ℝ :=
  total_snow - (snow_wednesday + snow_thursday)

theorem snowfall_on_friday :
  snowstorm 0.33 0.33 0.89 = 0.23 := 
by
  -- (Conditions)
  -- snow_wednesday = 0.33
  -- snow_thursday = 0.33
  -- total_snow = 0.89
  -- (Conclusion) snowstorm 0.33 0.33 0.89 = 0.23
  sorry

end snowfall_on_friday_l274_274487


namespace triangle_shape_area_l274_274509

theorem triangle_shape_area (a b : ℕ) (area_small area_middle area_large : ℕ) :
  a = 2 →
  b = 2 →
  area_small = (1 / 2) * a * b →
  area_middle = 2 * area_small →
  area_large = 2 * area_middle →
  area_small + area_middle + area_large = 14 :=
by
  intros
  sorry

end triangle_shape_area_l274_274509


namespace part1_part2_part3_l274_274120

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  (a * x^2 - Real.log x) * (x - Real.log x) + 1

variable {a : ℝ}

-- Prove that for all x > 0, if ax^2 > ln x, then f(x) ≥ ax^2 - ln x + 1
theorem part1 (h : ∀ x > 0, a*x^2 > Real.log x) (x : ℝ) (hx : x > 0) :
  f a x ≥ a*x^2 - Real.log x + 1 := sorry

-- Find the maximum value of a given there exists x₀ ∈ (0, +∞) where f(x₀) = 1 + x₀ ln x₀ - ln² x₀
theorem part2 (h : ∃ x₀ > 0, f a x₀ = 1 + x₀ * Real.log x₀ - (Real.log x₀)^2) :
  a ≤ 1 / Real.exp 1 := sorry

-- Prove that for all 1 < x < 2, we have f(x) > ax(2-ax)
theorem part3 (h : ∀ x, 1 < x ∧ x < 2) (x : ℝ) (hx1 : 1 < x) (hx2 : x < 2) :
  f a x > a * x * (2 - a * x) := sorry

end part1_part2_part3_l274_274120


namespace absolute_difference_m_n_l274_274624

theorem absolute_difference_m_n (m n : ℝ) (h1 : m * n = 6) (h2 : m + n = 7) : |m - n| = 5 := 
by 
  sorry

end absolute_difference_m_n_l274_274624


namespace sum_of_coefficients_proof_l274_274433

-- Problem statement: Define the expressions and prove the sum of the coefficients
def expr1 (c : ℝ) : ℝ := -(3 - c) * (c + 2 * (3 - c))
def expanded_form (c : ℝ) : ℝ := -c^2 + 9 * c - 18
def sum_of_coefficients (p : ℝ) := -1 + 9 - 18

theorem sum_of_coefficients_proof (c : ℝ) : sum_of_coefficients (expr1 c) = -10 := by
  sorry

end sum_of_coefficients_proof_l274_274433


namespace decrease_in_profit_when_one_loom_idles_l274_274553

def num_looms : ℕ := 125
def total_sales_value : ℕ := 500000
def total_manufacturing_expenses : ℕ := 150000
def monthly_establishment_charges : ℕ := 75000
def sales_value_per_loom : ℕ := total_sales_value / num_looms
def manufacturing_expense_per_loom : ℕ := total_manufacturing_expenses / num_looms
def decrease_in_sales_value : ℕ := sales_value_per_loom
def decrease_in_manufacturing_expenses : ℕ := manufacturing_expense_per_loom
def net_decrease_in_profit : ℕ := decrease_in_sales_value - decrease_in_manufacturing_expenses

theorem decrease_in_profit_when_one_loom_idles : net_decrease_in_profit = 2800 := by
  sorry

end decrease_in_profit_when_one_loom_idles_l274_274553


namespace felix_brother_lifting_capacity_is_600_l274_274569

-- Define the conditions
def felix_lifting_capacity (felix_weight : ℝ) : ℝ := 1.5 * felix_weight
def felix_brother_weight (felix_weight : ℝ) : ℝ := 2 * felix_weight
def felix_brother_lifting_capacity (brother_weight : ℝ) : ℝ := 3 * brother_weight
def felix_actual_lifting_capacity : ℝ := 150

-- Define the proof problem
theorem felix_brother_lifting_capacity_is_600 :
  ∃ felix_weight : ℝ,
    felix_lifting_capacity felix_weight = felix_actual_lifting_capacity ∧
    felix_brother_lifting_capacity (felix_brother_weight felix_weight) = 600 :=
by
  sorry

end felix_brother_lifting_capacity_is_600_l274_274569


namespace least_positive_integer_congruences_l274_274050

theorem least_positive_integer_congruences :
  ∃ n : ℕ, 
    n > 0 ∧ 
    (n % 4 = 1) ∧ 
    (n % 5 = 2) ∧ 
    (n % 6 = 3) ∧ 
    (n = 57) :=
by
  sorry

end least_positive_integer_congruences_l274_274050


namespace anyas_hair_loss_l274_274834

theorem anyas_hair_loss (H : ℝ) 
  (washes_hair_loss : H > 0) 
  (brushes_hair_loss : H / 2 > 0) 
  (grows_back : ∃ h : ℝ, h = 49 ∧ H + H / 2 + 1 = h) :
  H = 32 :=
by
  sorry

end anyas_hair_loss_l274_274834


namespace trigonometric_identity_l274_274289

theorem trigonometric_identity
  (α : ℝ)
  (h : Real.tan α = 2) :
  (4 * Real.sin α ^ 3 - 2 * Real.cos α) / (5 * Real.cos α + 3 * Real.sin α) = 2 / 5 :=
by
  sorry

end trigonometric_identity_l274_274289


namespace unique_three_digit_numbers_unique_three_digit_odd_numbers_l274_274445

def num_digits: ℕ := 10

theorem unique_three_digit_numbers (num_digits : ℕ) :
  ∃ n : ℕ, n = 648 ∧ n = (num_digits - 1) * (num_digits - 1) * (num_digits - 2) + 2 * (num_digits - 1) * (num_digits - 1) :=
  sorry

theorem unique_three_digit_odd_numbers (num_digits : ℕ) :
  ∃ n : ℕ, n = 320 ∧ ∀ odd_digit_nums : ℕ, odd_digit_nums ≥ 1 → odd_digit_nums = 5 → 
  n = odd_digit_nums * (num_digits - 2) * (num_digits - 2) :=
  sorry

end unique_three_digit_numbers_unique_three_digit_odd_numbers_l274_274445


namespace trapezium_area_l274_274095

theorem trapezium_area (a b h : ℝ) (ha : a = 20) (hb : b = 18) (hh : h = 15) :
  (1/2) * (a + b) * h = 285 := by
  sorry

end trapezium_area_l274_274095


namespace gas_pipe_probability_l274_274818

-- Define the problem statement in Lean.
theorem gas_pipe_probability :
  let total_area := 400 * 400 / 2
  let usable_area := (300 - 100) * (300 - 100) / 2
  usable_area / total_area = 1 / 4 :=
by
  -- Sorry will be placeholder for the proof
  sorry

end gas_pipe_probability_l274_274818


namespace smallest_number_jungkook_l274_274333

theorem smallest_number_jungkook (jungkook yoongi yuna : ℕ) 
  (hj : jungkook = 6 - 3) (hy : yoongi = 4) (hu : yuna = 5) : 
  jungkook < yoongi ∧ jungkook < yuna :=
by
  sorry

end smallest_number_jungkook_l274_274333


namespace least_positive_integer_satifies_congruences_l274_274047

theorem least_positive_integer_satifies_congruences :
  ∃ x : ℕ, x ≡ 1 [MOD 4] ∧ x ≡ 2 [MOD 5] ∧ x ≡ 3 [MOD 6] ∧ x = 17 :=
sorry

end least_positive_integer_satifies_congruences_l274_274047


namespace repeating_decimal_427_diff_l274_274336

theorem repeating_decimal_427_diff :
  let G := 0.427427427427
  let num := 427
  let denom := 999
  num.gcd denom = 1 →
  denom - num = 572 :=
by
  intros G num denom gcd_condition
  sorry

end repeating_decimal_427_diff_l274_274336


namespace solve_x_l274_274583

theorem solve_x (x : ℝ) (A B : ℝ × ℝ) (a : ℝ × ℝ) 
  (hA : A = (1, 3)) (hB : B = (2, 4))
  (ha : a = (2 * x - 1, x ^ 2 + 3 * x - 3))
  (hab : a = (B.1 - A.1, B.2 - A.2)) : x = 1 :=
by {
  sorry
}

end solve_x_l274_274583


namespace ellipse_x_intercepts_l274_274555

noncomputable def distances_sum (x : ℝ) (y : ℝ) (f₁ f₂ : ℝ × ℝ) :=
  (Real.sqrt ((x - f₁.1)^2 + (y - f₁.2)^2)) + (Real.sqrt ((x - f₂.1)^2 + (y - f₂.2)^2))

def is_on_ellipse (x y : ℝ) : Prop := 
  distances_sum x y (0, 3) (4, 0) = 7

theorem ellipse_x_intercepts 
  (h₀ : is_on_ellipse 0 0) 
  (hx_intercept : ∀ x : ℝ, is_on_ellipse x 0 → x = 0 ∨ x = 20 / 7) :
  ∀ x : ℝ, is_on_ellipse x 0 ↔ x = 0 ∨ x = 20 / 7 :=
by
  sorry

end ellipse_x_intercepts_l274_274555


namespace problem_rect_ratio_l274_274887

theorem problem_rect_ratio (W X Y Z U V R S : ℝ × ℝ) 
  (hYZ : Y = (0, 0))
  (hW : W = (0, 6))
  (hZ : Z = (7, 6))
  (hX : X = (7, 4))
  (hU : U = (5, 0))
  (hV : V = (4, 4))
  (hR : R = (5 / 3, 4))
  (hS : S = (0, 4))
  : (dist R S) / (dist X V) = 5 / 9 := 
sorry

end problem_rect_ratio_l274_274887


namespace positive_difference_solutions_l274_274379

theorem positive_difference_solutions (r₁ r₂ : ℝ) (h_r₁ : (r₁^2 - 5 * r₁ - 22) / (r₁ + 4) = 3 * r₁ + 8) (h_r₂ : (r₂^2 - 5 * r₂ - 22) / (r₂ + 4) = 3 * r₂ + 8) (h_r₁_ne : r₁ ≠ -4) (h_r₂_ne : r₂ ≠ -4) :
  |r₁ - r₂| = 3 / 2 := 
sorry


end positive_difference_solutions_l274_274379


namespace matrix_inverse_problem_l274_274018

theorem matrix_inverse_problem
  (x y z w : ℚ)
  (h1 : 2 * x + 3 * w = 1)
  (h2 : x * z = 15)
  (h3 : 4 * w = -8)
  (h4 : 4 * z = 5 * y) :
  x * y * z * w = -102.857 := by
    sorry

end matrix_inverse_problem_l274_274018


namespace carina_total_coffee_l274_274428

def number_of_ten_ounce_packages : ℕ := 4
def number_of_five_ounce_packages : ℕ := number_of_ten_ounce_packages + 2
def ounces_in_each_ten_ounce_package : ℕ := 10
def ounces_in_each_five_ounce_package : ℕ := 5

def total_coffee_ounces : ℕ := 
  (number_of_ten_ounce_packages * ounces_in_each_ten_ounce_package) +
  (number_of_five_ounce_packages * ounces_in_each_five_ounce_package)

theorem carina_total_coffee : total_coffee_ounces = 70 := by
  -- proof to be provided
  sorry

end carina_total_coffee_l274_274428


namespace determine_b_l274_274919

noncomputable def f (x : ℝ) (b : ℝ) : ℝ := 1 / (3 * x + b)
noncomputable def f_inv (x : ℝ) : ℝ := (2 - 3 * x) / (3 * x)

theorem determine_b (b : ℝ) :
    (∀ x : ℝ, f_inv (f x b) = x) ↔ b = -3 :=
by
  sorry

end determine_b_l274_274919


namespace derivative_of_y_l274_274574

variable (x : ℝ)

def y := x^3 + 3 * x^2 + 6 * x - 10

theorem derivative_of_y : (deriv y) x = 3 * x^2 + 6 * x + 6 :=
sorry

end derivative_of_y_l274_274574


namespace john_total_time_spent_l274_274756

-- Define conditions
def num_pictures : ℕ := 10
def draw_time_per_picture : ℝ := 2
def color_time_reduction : ℝ := 0.3

-- Define the actual color time per picture
def color_time_per_picture : ℝ := draw_time_per_picture * (1 - color_time_reduction)

-- Define the total time per picture
def total_time_per_picture : ℝ := draw_time_per_picture + color_time_per_picture

-- Define the total time for all pictures
def total_time_for_all_pictures : ℝ := total_time_per_picture * num_pictures

-- The theorem we need to prove
theorem john_total_time_spent : total_time_for_all_pictures = 34 :=
by
sorry

end john_total_time_spent_l274_274756


namespace magnitude_a_eq_3sqrt2_l274_274340

open Real

def a (x: ℝ) : ℝ × ℝ := (3, x)
def b : ℝ × ℝ := (-1, 1)
def perpendicular (u v : ℝ × ℝ) : Prop := u.1 * v.1 + u.2 * v.2 = 0

theorem magnitude_a_eq_3sqrt2 (x : ℝ) (h : perpendicular (a x) b) :
  ‖a 3‖ = 3 * sqrt 2 := by
  sorry

end magnitude_a_eq_3sqrt2_l274_274340


namespace sheep_to_cow_ratio_l274_274346

theorem sheep_to_cow_ratio : 
  ∀ (cows sheep : ℕ) (cow_water sheep_water : ℕ),
  cows = 40 →
  cow_water = 80 →
  sheep_water = cow_water / 4 →
  7 * (cows * cow_water + sheep * sheep_water) = 78400 →
  sheep / cows = 10 :=
by
  intros cows sheep cow_water sheep_water hcows hcow_water hsheep_water htotal
  sorry

end sheep_to_cow_ratio_l274_274346


namespace find_slope_of_line_l274_274585

theorem find_slope_of_line
  (k : ℝ)
  (P : ℝ × ℝ)
  (hP : P = (3, 0))
  (C : ℝ → ℝ → Prop)
  (hC : ∀ x y, C x y ↔ x^2 - y^2 / 3 = 1)
  (A B : ℝ × ℝ)
  (hA : C A.1 A.2)
  (hB : C B.1 B.2)
  (line : ℝ → ℝ → Prop)
  (hline : ∀ x y, line x y ↔ y = k * (x - 3))
  (hintersectA : line A.1 A.2)
  (hintersectB : line B.1 B.2)
  (F : ℝ × ℝ)
  (hF : F = (1, 0))
  (hfoci_sum : ∀ z : ℝ × ℝ, |z.1 - F.1| + |z.2 - F.2| = 16) :
  k = 3 ∨ k = -3 :=
by
  sorry

end find_slope_of_line_l274_274585


namespace largest_whole_number_lt_div_l274_274930

theorem largest_whole_number_lt_div {x : ℕ} (hx : 8 * x < 80) : x ≤ 9 :=
by
  sorry

end largest_whole_number_lt_div_l274_274930


namespace value_of_y_l274_274142

variables (x y : ℝ)

theorem value_of_y (h1 : x - y = 16) (h2 : x + y = 8) : y = -4 :=
by
  sorry

end value_of_y_l274_274142


namespace age_sum_squares_l274_274107

theorem age_sum_squares (a b c : ℕ) (h1 : 5 * a + 2 * b = 3 * c) (h2 : 3 * c^2 = 4 * a^2 + b^2) (h3 : Nat.gcd (Nat.gcd a b) c = 1) : a^2 + b^2 + c^2 = 18 :=
sorry

end age_sum_squares_l274_274107


namespace Lance_daily_earnings_l274_274274

theorem Lance_daily_earnings :
  ∀ (hours_per_week : ℕ) (workdays_per_week : ℕ) (hourly_rate : ℕ) (total_earnings : ℕ) (daily_earnings : ℕ),
  hours_per_week = 35 →
  workdays_per_week = 5 →
  hourly_rate = 9 →
  total_earnings = hours_per_week * hourly_rate →
  daily_earnings = total_earnings / workdays_per_week →
  daily_earnings = 63 := 
by
  intros hours_per_week workdays_per_week hourly_rate total_earnings daily_earnings
  intros H1 H2 H3 H4 H5
  sorry

end Lance_daily_earnings_l274_274274


namespace initial_money_proof_l274_274560

-- Definition: Dan's initial money, the money spent, and the money left.
def initial_money : ℝ := sorry
def spent_money : ℝ := 1.0
def left_money : ℝ := 2.0

-- Theorem: Prove that Dan's initial money is the sum of the money spent and the money left.
theorem initial_money_proof : initial_money = spent_money + left_money :=
sorry

end initial_money_proof_l274_274560


namespace tom_searching_days_l274_274367

variable (d : ℕ) (total_cost : ℕ)

theorem tom_searching_days :
  (∀ n, n ≤ 5 → total_cost = n * 100 + (d - n) * 60) →
  (∀ n, n > 5 → total_cost = 5 * 100 + (d - 5) * 60) →
  total_cost = 800 →
  d = 10 :=
by
  intros h1 h2 h3
  sorry

end tom_searching_days_l274_274367


namespace stickers_per_page_l274_274515

theorem stickers_per_page (total_pages total_stickers : ℕ) (h1 : total_pages = 22) (h2 : total_stickers = 220) : (total_stickers / total_pages) = 10 :=
by
  sorry

end stickers_per_page_l274_274515


namespace combined_mpg_rate_l274_274493

-- Conditions of the problem
def ray_mpg : ℝ := 48
def tom_mpg : ℝ := 24
def ray_distance (s : ℝ) : ℝ := 2 * s
def tom_distance (s : ℝ) : ℝ := s

-- Theorem to prove the combined rate of miles per gallon
theorem combined_mpg_rate (s : ℝ) (h : s > 0) : 
  let total_distance := tom_distance s + ray_distance s
  let ray_gas_usage := ray_distance s / ray_mpg
  let tom_gas_usage := tom_distance s / tom_mpg
  let total_gas_usage := ray_gas_usage + tom_gas_usage
  total_distance / total_gas_usage = 36 := 
by
  sorry

end combined_mpg_rate_l274_274493


namespace y_less_than_z_by_40_percent_l274_274146

variable {x y z : ℝ}

theorem y_less_than_z_by_40_percent (h1 : x = 1.3 * y) (h2 : x = 0.78 * z) : y = 0.6 * z :=
by
  -- The proof will be provided here
  -- We are demonstrating that y = 0.6 * z is a consequence of h1 and h2
  sorry

end y_less_than_z_by_40_percent_l274_274146


namespace least_positive_integer_satifies_congruences_l274_274048

theorem least_positive_integer_satifies_congruences :
  ∃ x : ℕ, x ≡ 1 [MOD 4] ∧ x ≡ 2 [MOD 5] ∧ x ≡ 3 [MOD 6] ∧ x = 17 :=
sorry

end least_positive_integer_satifies_congruences_l274_274048


namespace arcsin_sqrt2_over_2_eq_pi_over_4_l274_274837

theorem arcsin_sqrt2_over_2_eq_pi_over_4 :
  Real.arcsin (Real.sqrt 2 / 2) = Real.pi / 4 :=
sorry

end arcsin_sqrt2_over_2_eq_pi_over_4_l274_274837


namespace part1_part2_l274_274122

noncomputable def f (x a : ℝ) : ℝ := cos x ^ 2 + a * sin x + 2 * a - 1

-- Part 1: For a = 1, find the maximum and minimum of the function.
theorem part1 (x : ℝ) : (f x 1 ≤ 9 / 4) ∧ (0 ≤ f x 1) :=
  sorry

-- Part 2: Determine the range of values for a given that f(x) ≤ 5 for all x in [-π/2, π/2].
theorem part2 (a : ℝ) (x : ℝ) (hx : x ∈ Set.Icc (-Real.pi / 2) (Real.pi / 2)) : (f x a ≤ 5) → (a ≤ 2) :=
  sorry

end part1_part2_l274_274122


namespace smallest_k_for_positive_roots_5_l274_274580

noncomputable def smallest_k_for_positive_roots : ℕ := 5

theorem smallest_k_for_positive_roots_5
  (k p q : ℕ) 
  (hk : k = smallest_k_for_positive_roots)
  (hq_pos : 0 < q)
  (h_distinct_pos_roots : ∃ (x₁ x₂ : ℝ), 0 < x₁ ∧ x₁ < x₂ ∧ x₂ < 1 ∧ 
    k * x₁ * x₂ = q ∧ k * x₁ + k * x₂ > p ∧ k * x₁ * x₂ < q * ( 1 / (x₁*(1 - x₁) * x₂ * (1 - x₂)))) :
  k = 5 :=
by
  sorry

end smallest_k_for_positive_roots_5_l274_274580


namespace rectangle_circle_area_ratio_l274_274783

theorem rectangle_circle_area_ratio (w r: ℝ) (h1: 3 * w = π * r) (h2: 2 * w = l) :
  (l * w) / (π * r ^ 2) = 2 * π / 9 :=
by 
  sorry

end rectangle_circle_area_ratio_l274_274783


namespace sum_of_perimeters_l274_274422

theorem sum_of_perimeters (a : ℝ) : 
    ∑' n : ℕ, (3 * a) * (1/3)^n = 9 * a / 2 :=
by sorry

end sum_of_perimeters_l274_274422


namespace pages_written_on_wednesday_l274_274909

variable (minutesMonday minutesTuesday rateMonday rateTuesday : ℕ)
variable (totalPages : ℕ)

def pagesOnMonday (minutesMonday rateMonday : ℕ) : ℕ :=
  minutesMonday / rateMonday

def pagesOnTuesday (minutesTuesday rateTuesday : ℕ) : ℕ :=
  minutesTuesday / rateTuesday

def totalPagesMondayAndTuesday (minutesMonday minutesTuesday rateMonday rateTuesday : ℕ) : ℕ :=
  pagesOnMonday minutesMonday rateMonday + pagesOnTuesday minutesTuesday rateTuesday

def pagesOnWednesday (minutesMonday minutesTuesday rateMonday rateTuesday totalPages : ℕ) : ℕ :=
  totalPages - totalPagesMondayAndTuesday minutesMonday minutesTuesday rateMonday rateTuesday

theorem pages_written_on_wednesday :
  pagesOnWednesday 60 45 30 15 10 = 5 := by
  sorry

end pages_written_on_wednesday_l274_274909


namespace total_apple_trees_is_800_l274_274418

variable (T P A : ℕ) -- Total number of trees, peach trees, and apple trees respectively
variable (samples_peach samples_apple : ℕ) -- Sampled peach trees and apple trees respectively
variable (sampled_percentage : ℕ) -- Percentage of total trees sampled

-- Given conditions
axiom H1 : sampled_percentage = 10
axiom H2 : samples_peach = 50
axiom H3 : samples_apple = 80

-- Theorem to prove the number of apple trees
theorem total_apple_trees_is_800 : A = 800 :=
by sorry

end total_apple_trees_is_800_l274_274418


namespace find_c_for_two_solutions_in_real_l274_274990

noncomputable def system_two_solutions (x y c : ℝ) : Prop := (|x + y| = 2007 ∧ |x - y| = c)

theorem find_c_for_two_solutions_in_real : ∃ c : ℝ, (∀ x y : ℝ, system_two_solutions x y c) ↔ (c = 0) :=
by
  sorry

end find_c_for_two_solutions_in_real_l274_274990


namespace problem1_problem2_l274_274343

-- Definitions of the sets A and B
def set_A (x : ℝ) : Prop := x ≤ -1 ∨ x ≥ 4
def set_B (x a : ℝ) : Prop := 2 * a ≤ x ∧ x ≤ a + 2

-- Problem 1: If A ∩ B ≠ ∅, find the range of a
theorem problem1 (a : ℝ) : (∃ x : ℝ, set_A x ∧ set_B x a) → a ≤ -1 / 2 ∨ a = 2 :=
sorry

-- Problem 2: If A ∩ B = B, find the value of a
theorem problem2 (a : ℝ) : (∀ x : ℝ, set_B x a → set_A x) → a ≤ -1 / 2 ∨ a ≥ 2 :=
sorry

end problem1_problem2_l274_274343


namespace number_of_solutions_of_sine_eq_third_to_x_l274_274854

open Real

theorem number_of_solutions_of_sine_eq_third_to_x : 
  (set.Icc 0 (200 * π)).countable.count (λ x, sin x = (1/3)^x) = 200 :=
by
  sorry

end number_of_solutions_of_sine_eq_third_to_x_l274_274854


namespace prob_abs_diff_gt_one_third_l274_274068

noncomputable theory

-- Definitions corresponding to given conditions
def coin_flip_outcome : Type := ℕ
def coin_flip : ℕ → coin_flip_outcome := λ _, if (rand.uniform 0 1) < 0.5 then 0 else 1
def generate_number : list coin_flip_outcome → ℝ
| [0, 0, 0]        := 0
| [1, 1, 1]        := 1
| _                := rand.uniform 0 1

def choose_number : ℝ :=
  let results := list.ret <| repeat (coin_flip 3) 3 in
  generate_number results

-- Theorem statement
theorem prob_abs_diff_gt_one_third :
  let x := choose_number
  let y := choose_number
  P (|x - y| > 1 / 3) = 3 / 32 :=
sorry

end prob_abs_diff_gt_one_third_l274_274068


namespace isosceles_triangle_perimeter_l274_274883

-- Definitions of the conditions
def is_isosceles (a b : ℕ) : Prop :=
  a = b

def has_side_lengths (a b : ℕ) (c : ℕ) : Prop :=
  true

-- The statement to be proved
theorem isosceles_triangle_perimeter (a b c : ℕ) 
  (h₁ : is_isosceles a b) (h₂ : has_side_lengths a b c) :
  (a + b + c = 16 ∨ a + b + c = 17) :=
sorry

end isosceles_triangle_perimeter_l274_274883


namespace third_smallest_four_digit_number_in_pascals_triangle_l274_274225

theorem third_smallest_four_digit_number_in_pascals_triangle : 
    (∃ n, binomial n (n - 2) = 1002 ∧ binomial (n-1) (n - 1 - 2) = 1001 ∧ binomial (n-2) (n - 2 - 2) = 1000) :=
by
  sorry

end third_smallest_four_digit_number_in_pascals_triangle_l274_274225


namespace distribution_ways_5_to_3_l274_274133

noncomputable def num_ways (n m : ℕ) : ℕ :=
  m ^ n

theorem distribution_ways_5_to_3 : num_ways 5 3 = 243 := by
  sorry

end distribution_ways_5_to_3_l274_274133


namespace divisibility_by_7_l274_274638

theorem divisibility_by_7 (m a : ℤ) (h : 0 ≤ a ∧ a ≤ 9) (B : ℤ) (hB : B = m - 2 * a) (h7 : B % 7 = 0) : (10 * m + a) % 7 = 0 := 
sorry

end divisibility_by_7_l274_274638


namespace abs_diff_eq_5_l274_274629

-- Definitions of m and n, and conditions provided in the problem
variables (m n : ℝ)
hypothesis (h1 : m * n = 6)
hypothesis (h2 : m + n = 7)

-- Statement to prove
theorem abs_diff_eq_5 : |m - n| = 5 :=
by
  sorry

end abs_diff_eq_5_l274_274629


namespace minimum_p_l274_274141

-- Define the problem constants and conditions
noncomputable def problem_statement :=
  ∃ p q : ℕ, 
    0 < p ∧ 0 < q ∧ 
    (2008 / 2009 < p / (q : ℚ)) ∧ (p / (q : ℚ) < 2009 / 2010) ∧ 
    (∀ p' q' : ℕ, (0 < p' ∧ 0 < q' ∧ (2008 / 2009 < p' / (q' : ℚ)) ∧ (p' / (q' : ℚ) < 2009 / 2010)) → p ≤ p') 

-- The proof
theorem minimum_p (h : problem_statement) :
  ∃ p q : ℕ, 
    0 < p ∧ 0 < q ∧ 
    (2008 / 2009 < p / (q : ℚ)) ∧ (p / (q : ℚ) < 2009 / 2010) ∧
    p = 4017 :=
sorry

end minimum_p_l274_274141


namespace johns_original_earnings_l274_274617

-- Definitions from conditions
variables (x : ℝ) (raise_percentage : ℝ) (new_salary : ℝ)

-- Conditions
def conditions : Prop :=
  raise_percentage = 0.25 ∧ new_salary = 75 ∧ x + raise_percentage * x = new_salary

-- Theorem statement
theorem johns_original_earnings (h : conditions x 0.25 75) : x = 60 :=
sorry

end johns_original_earnings_l274_274617


namespace find_angle_B_find_side_b_l274_274450

variable {A B C : ℝ}
variable {a b c : ℝ}
variable {m n : ℝ × ℝ}
variable {dot_product_max : ℝ}

-- Conditions
def triangle_condition (a b c : ℝ) (A B C : ℝ) : Prop :=
  a * Real.sin A + c * Real.sin C - b * Real.sin B = Real.sqrt 2 * a * Real.sin C

def vectors (m n : ℝ × ℝ) := 
  m = (Real.cos A, Real.cos (2 * A)) ∧ n = (12, -5)

def side_length_a (a : ℝ) := 
  a = 4

-- Questions and Proof Problems
theorem find_angle_B (A B C : ℝ) (a b c : ℝ) (h1 : triangle_condition a b c A B C) : 
  B = π / 4 :=
sorry

theorem find_side_b (A B C : ℝ) (a b c : ℝ) 
  (m n : ℝ × ℝ) (max_dot_product_condition : Real.cos A = 3 / 5) 
  (ha : side_length_a a) (hb : b = a * Real.sin B / Real.sin A) : 
  b = 5 * Real.sqrt 2 / 2 :=
sorry

end find_angle_B_find_side_b_l274_274450


namespace fg_eq_gf_condition_l274_274462

theorem fg_eq_gf_condition (m n p q : ℝ) (f g : ℝ → ℝ)
  (hf : ∀ x, f x = m * x + n) 
  (hg : ∀ x, g x = p * x + q) : 
  (∀ x, f (g x) = g (f x)) ↔ n * (1 - p) = q * (1 - m) := 
sorry

end fg_eq_gf_condition_l274_274462


namespace third_smallest_four_digit_in_pascals_triangle_l274_274216

theorem third_smallest_four_digit_in_pascals_triangle :
  ∃ n k, n > 0 ∧ k > 0 ∧ Nat.choose n k = 1035 ∧ 
  (∀ (m l : ℕ), m > 0 ∧ l > 0 ∧ Nat.choose m l < 1000 → Nat.choose m l < 1035) ∧
  (∀ (p q : ℕ), p > 0 ∧ q > 0 ∧ 1000 ≤ Nat.choose p q ∧ Nat.choose p q < 1035 → 
    (Nat.choose m l = 1000 ∨ Nat.choose m l = 1001 ∨ Nat.choose m l = 1035)) :=
sorry

end third_smallest_four_digit_in_pascals_triangle_l274_274216


namespace max_unique_solution_l274_274430

theorem max_unique_solution (x y : ℕ) (m : ℕ) (h : 2005 * x + 2007 * y = m) : 
  m = 2 * 2005 * 2007 ↔ ∃! (x y : ℕ), 2005 * x + 2007 * y = m :=
sorry

end max_unique_solution_l274_274430


namespace find_a2_l274_274300

def arithmetic_sequence (a : ℕ → ℚ) := 
  (a 1 = 1) ∧ ∀ n, a (n + 2) - a n = 3

theorem find_a2 (a : ℕ → ℚ) (h : arithmetic_sequence a) : 
  a 2 = 5 / 2 := 
by
  -- Conditions
  have a1 : a 1 = 1 := h.1
  have h_diff : ∀ n, a (n + 2) - a n = 3 := h.2
  -- Proof steps can be written here
  sorry

end find_a2_l274_274300


namespace largest_c_in_range_l274_274102

theorem largest_c_in_range (c : ℝ) (h : ∃ x : ℝ,  2 * x ^ 2 - 4 * x + c = 5) : c ≤ 7 :=
by sorry

end largest_c_in_range_l274_274102


namespace number_of_starting_lineups_l274_274015

-- Defining the conditions as sets and their respective sizes
def players : Finset ℕ := {1, 2, ..., 15}  -- representing players with numbers
def Tim : ℕ := 1
def Tom : ℕ := 2
def Kim : ℕ := 3

-- Finset of players excluding Tim, Tom, and Kim
def other_players := players \ {Tim, Tom, Kim}

-- Theorem stating the number of valid starting lineups
theorem number_of_starting_lineups :
    ∃ (lineups : Finset (Finset ℕ)), 
    (∀ lineup ∈ lineups, lineup.card = 5 ∧ (Tim ∈ lineup ∨ Tom ∈ lineup) ∧ ¬(Tim ∈ lineup ∧ Tom ∈ lineup ∧ Kim ∈ lineup)) ∧
    lineups.card = 1210 :=
sorry

end number_of_starting_lineups_l274_274015


namespace rebecca_eggs_l274_274353

/-- Rebecca wants to split a collection of eggs into 4 groups. Each group will have 2 eggs. -/
def number_of_groups : Nat := 4

def eggs_per_group : Nat := 2

theorem rebecca_eggs : (number_of_groups * eggs_per_group) = 8 := by
  sorry

end rebecca_eggs_l274_274353


namespace find_a1_l274_274115

noncomputable def a (n : ℕ) : ℤ := sorry -- the definition of sequence a_n is not computable without initial terms
noncomputable def S (n : ℕ) : ℤ := sorry -- similarly, the definition of S_n without initial terms isn't given

axiom recurrence_relation (n : ℕ) (h : n ≥ 3): 
  a (n) = a (n - 1) - a (n - 2)

axiom S9 : S 9 = 6
axiom S10 : S 10 = 5

theorem find_a1 : a 1 = 1 :=
by
  sorry

end find_a1_l274_274115


namespace max_value_cos2_sin_l274_274358

noncomputable def max_cos2_sin (x : Real) : Real := 
  (Real.cos x) ^ 2 + Real.sin x

theorem max_value_cos2_sin : 
  ∃ x : Real, (-1 ≤ Real.sin x) ∧ (Real.sin x ≤ 1) ∧ 
    max_cos2_sin x = 5 / 4 :=
sorry

end max_value_cos2_sin_l274_274358


namespace f_g_x_eq_l274_274483

noncomputable def f (x : ℝ) : ℝ := (x * (x + 1)) / 3
noncomputable def g (x : ℝ) : ℝ := x + 3

theorem f_g_x_eq (x : ℝ) : f (g x) = (x^2 + 7*x + 12) / 3 := by
  sorry

end f_g_x_eq_l274_274483


namespace left_handed_rock_lovers_l274_274187

def total_people := 30
def left_handed := 12
def like_rock_music := 20
def right_handed_dislike_rock := 3

theorem left_handed_rock_lovers : ∃ x, x + (left_handed - x) + (like_rock_music - x) + right_handed_dislike_rock = total_people ∧ x = 5 :=
by
  sorry

end left_handed_rock_lovers_l274_274187


namespace average_birth_rate_l274_274470

theorem average_birth_rate (B : ℕ) 
  (death_rate : ℕ := 3)
  (daily_net_increase : ℕ := 86400) 
  (intervals_per_day : ℕ := 86400 / 2) 
  (net_increase : ℕ := (B - death_rate) * intervals_per_day) : 
  net_increase = daily_net_increase → 
  B = 5 := 
sorry

end average_birth_rate_l274_274470


namespace sufficient_but_not_necessary_perpendicular_l274_274399

theorem sufficient_but_not_necessary_perpendicular (a : ℝ) :
  (∃ a' : ℝ, a' = -1 ∧ (a' = -1 → (0 : ℝ) ≠ 3 * a' - 1)) ∨
  (∃ a' : ℝ, a' ≠ -1 ∧ (a' ≠ -1 → (0 : ℝ) ≠ 3 * a' - 1)) →
  (3 * a' - 1) * (a' - 3) = -1 := sorry

end sufficient_but_not_necessary_perpendicular_l274_274399


namespace sum_p_q_r_l274_274356

theorem sum_p_q_r :
  ∃ (p q r : ℤ), 
    (∀ x : ℤ, x ^ 2 + 20 * x + 96 = (x + p) * (x + q)) ∧ 
    (∀ x : ℤ, x ^ 2 - 22 * x + 120 = (x - q) * (x - r)) ∧ 
    p + q + r = 30 :=
by 
  sorry

end sum_p_q_r_l274_274356


namespace find_x_l274_274676

def op (a b : ℤ) : ℤ := -2 * a + b

theorem find_x (x : ℤ) (h : op x (-5) = 3) : x = -4 :=
by
  sorry

end find_x_l274_274676


namespace point_in_fourth_quadrant_l274_274752

def point (x y : ℝ) := (x, y)
def x_positive (p : ℝ × ℝ) : Prop := p.1 > 0
def y_negative (p : ℝ × ℝ) : Prop := p.2 < 0
def in_fourth_quadrant (p : ℝ × ℝ) : Prop := x_positive p ∧ y_negative p

theorem point_in_fourth_quadrant : in_fourth_quadrant (2, -4) :=
by
  -- The proof states that (2, -4) is in the fourth quadrant.
  sorry

end point_in_fourth_quadrant_l274_274752


namespace pairs_sum_gcd_l274_274849

theorem pairs_sum_gcd (a b : ℕ) (h_sum : a + b = 288) (h_gcd : Int.gcd a b = 36) :
  (a = 36 ∧ b = 252) ∨ (a = 252 ∧ b = 36) ∨ (a = 108 ∧ b = 180) ∨ (a = 180 ∧ b = 108) :=
by {
   sorry
}

end pairs_sum_gcd_l274_274849


namespace find_angle_l274_274359

noncomputable def angle_between_generatrix_and_base (k : ℝ) (h : k > 3 / 2) : ℝ :=
  real.arctan ((2 : ℝ) / real.sqrt (2 * k - 3))

-- Theorem
theorem find_angle (k : ℝ) (h : k > 3 / 2) : 
  angle_between_generatrix_and_base k h = real.arctan ((2 : ℝ) / real.sqrt (2 * k - 3)) :=
sorry

end find_angle_l274_274359


namespace suzanne_donation_total_l274_274011

theorem suzanne_donation_total : 
  (10 + 10 * 2 + 10 * 2^2 + 10 * 2^3 + 10 * 2^4 = 310) :=
by
  sorry

end suzanne_donation_total_l274_274011


namespace total_volume_is_10_l274_274679

noncomputable def total_volume_of_final_mixture (V : ℝ) : ℝ :=
  2.5 + V

theorem total_volume_is_10 :
  ∃ (V : ℝ), 
  (0.30 * 2.5 + 0.50 * V = 0.45 * (2.5 + V)) ∧ 
  total_volume_of_final_mixture V = 10 :=
by
  sorry

end total_volume_is_10_l274_274679


namespace polynomial_evaluation_l274_274256

theorem polynomial_evaluation :
  7^4 + 4 * 7^3 + 6 * 7^2 + 4 * 7 + 1 = 4096 :=
by
  sorry

end polynomial_evaluation_l274_274256


namespace tangent_line_parallel_curve_l274_274457

def curve (x : ℝ) : ℝ := x^4

def line_parallel_to_curve (l : ℝ → ℝ → Prop) : Prop :=
  ∃ x0 y0 : ℝ, l x0 y0 ∧ curve x0 = y0 ∧ ∀ (x : ℝ), l x (curve x)

theorem tangent_line_parallel_curve :
  ∃ (l : ℝ → ℝ → Prop), line_parallel_to_curve l ∧ ∀ x y, l x y ↔ 8 * x + 16 * y + 3 = 0 :=
by
  sorry

end tangent_line_parallel_curve_l274_274457


namespace profit_percent_is_26_l274_274395

variables (P C : ℝ)
variables (h1 : (2/3) * P = 0.84 * C)

theorem profit_percent_is_26 :
  ((P - C) / C) * 100 = 26 :=
by
  sorry

end profit_percent_is_26_l274_274395


namespace train_length_l274_274830

theorem train_length (L : ℕ) (speed : ℕ) 
  (h1 : L + 1200 = speed * 45) 
  (h2 : L + 180 = speed * 15) : 
  L = 330 := 
sorry

end train_length_l274_274830


namespace trig_identity_l274_274589

noncomputable def tan_eq_neg_4_over_3 (theta : ℝ) : Prop := 
  Real.tan theta = -4 / 3

theorem trig_identity (theta : ℝ) (h : tan_eq_neg_4_over_3 theta) : 
  (Real.cos (π / 2 + θ) - Real.sin (-π - θ)) / (Real.cos (11 * π / 2 - θ) + Real.sin (9 * π / 2 + θ)) = 8 / 7 :=
by
  sorry

end trig_identity_l274_274589


namespace midpoint_plane_distance_l274_274297

noncomputable def midpoint_distance (A B : ℝ) (dA dB : ℝ) : ℝ :=
  (dA + dB) / 2

theorem midpoint_plane_distance (A B : ℝ) (dA dB : ℝ) (hA : dA = 1) (hB : dB = 3) :
  midpoint_distance A B dA dB = 1 ∨ midpoint_distance A B dA dB = 2 :=
by
  sorry

end midpoint_plane_distance_l274_274297


namespace ball_height_less_than_10_after_16_bounces_l274_274402

noncomputable def bounce_height (initial : ℝ) (ratio : ℝ) (bounces : ℕ) : ℝ :=
  initial * ratio^bounces

theorem ball_height_less_than_10_after_16_bounces :
  let initial_height := 800
  let bounce_ratio := 3 / 4
  ∃ k : ℕ, k = 16 ∧ bounce_height initial_height bounce_ratio k < 10 := by
  let initial_height := 800
  let bounce_ratio := 3 / 4
  use 16
  sorry

end ball_height_less_than_10_after_16_bounces_l274_274402


namespace angles_terminal_yaxis_l274_274360

theorem angles_terminal_yaxis :
  {θ : ℝ | ∃ k : ℤ, θ = 2 * k * Real.pi + Real.pi / 2 ∨ θ = 2 * k * Real.pi + 3 * Real.pi / 2} =
  {θ : ℝ | ∃ n : ℤ, θ = n * Real.pi + Real.pi / 2} :=
by sorry

end angles_terminal_yaxis_l274_274360


namespace macey_weeks_to_save_l274_274488

theorem macey_weeks_to_save :
  ∀ (total_cost amount_saved weekly_savings : ℝ),
    total_cost = 22.45 →
    amount_saved = 7.75 →
    weekly_savings = 1.35 →
    ⌈(total_cost - amount_saved) / weekly_savings⌉ = 11 :=
by
  intros total_cost amount_saved weekly_savings h_total_cost h_amount_saved h_weekly_savings
  sorry

end macey_weeks_to_save_l274_274488


namespace number_of_newborn_members_in_group_l274_274884

noncomputable def N : ℝ :=
  let p_death := 1 / 10
  let p_survive := 1 - p_death
  let prob_survive_3_months := p_survive * p_survive * p_survive
  218.7 / prob_survive_3_months

theorem number_of_newborn_members_in_group : N = 300 := by
  sorry

end number_of_newborn_members_in_group_l274_274884


namespace parabola_locus_l274_274759

variables (a c k : ℝ) (a_pos : 0 < a) (c_pos : 0 < c) (k_pos : 0 < k)

theorem parabola_locus :
  ∀ t : ℝ, ∃ x y : ℝ,
    x = -kt / (2 * a) ∧ y = - k^2 * t^2 / (4 * a) + c ∧
    y = - (k^2 / (4 * a)) * x^2 + c :=
sorry

end parabola_locus_l274_274759


namespace total_limes_picked_l274_274581

-- Define the number of limes each person picked
def fred_limes : Nat := 36
def alyssa_limes : Nat := 32
def nancy_limes : Nat := 35
def david_limes : Nat := 42
def eileen_limes : Nat := 50

-- Formal statement of the problem
theorem total_limes_picked : 
  fred_limes + alyssa_limes + nancy_limes + david_limes + eileen_limes = 195 := by
  -- Add proof
  sorry

end total_limes_picked_l274_274581


namespace unique_three_digit_numbers_unique_three_digit_odd_numbers_l274_274443

-- Part 1: Total unique three-digit numbers
theorem unique_three_digit_numbers : 
  (finset.univ.card) * (finset.univ.erase 0).card * (finset.univ.erase 0).erase 9.card = 648 :=
sorry

-- Part 2: Total unique three-digit odd numbers
theorem unique_three_digit_odd_numbers :
  5 * ((finset.univ.erase 5).erase 0).card * ((finset.univ.erase 5).erase 0).erase 9.card = 320 :=
sorry

end unique_three_digit_numbers_unique_three_digit_odd_numbers_l274_274443


namespace remainder_when_divided_by_x_minus_4_l274_274054

-- Define the polynomial function
def f (x : ℝ) : ℝ := x^5 - 8 * x^4 + 15 * x^3 + 20 * x^2 - 5 * x - 20

-- State the problem as a theorem
theorem remainder_when_divided_by_x_minus_4 : 
    (f 4 = 216) := 
by 
    -- Calculation goes here
    sorry

end remainder_when_divided_by_x_minus_4_l274_274054


namespace trapezium_area_l274_274099

theorem trapezium_area (a b h : ℝ) (ha : a = 20) (hb : b = 18) (hh : h = 15) : 
  (1/2) * (a + b) * h = 285 :=
by
  rw [ha, hb, hh]
  norm_num

end trapezium_area_l274_274099


namespace tan_half_angles_l274_274460

theorem tan_half_angles (a b : ℝ) (ha : 3 * (Real.cos a + Real.cos b) + 5 * (Real.cos a * Real.cos b - 1) = 0) :
  ∃ z : ℝ, z = Real.tan (a / 2) * Real.tan (b / 2) ∧ (z = Real.sqrt (6 / 13) ∨ z = -Real.sqrt (6 / 13)) :=
by
  sorry

end tan_half_angles_l274_274460


namespace complement_union_A_B_l274_274722

open Set

def A : Set ℝ := {x | -1 < x ∧ x < 5}
def B : Set ℝ := {x | x ≤ -2 ∨ x ≥ 2}
def U : Set ℝ := A ∪ B
def R : Set ℝ := univ

theorem complement_union_A_B : (R \ U) = {x | -2 < x ∧ x ≤ -1} :=
by
  sorry

end complement_union_A_B_l274_274722


namespace polynomial_roots_bc_product_l274_274658

theorem polynomial_roots_bc_product : ∃ (b c : ℤ), 
  (∀ x, (x^2 - 2*x - 1 = 0 → x^5 - b*x^3 - c*x^2 = 0)) ∧ (b * c = 348) := by 
  sorry

end polynomial_roots_bc_product_l274_274658


namespace total_number_of_trees_l274_274518

variable {T : ℕ} -- Define T as a natural number (total number of trees)
variable (h1 : 70 / 100 * T + 105 = T) -- Indicates 30% of T is 105

theorem total_number_of_trees (h1 : 70 / 100 * T + 105 = T) : T = 350 :=
by
sorry

end total_number_of_trees_l274_274518


namespace probability_three_primes_from_1_to_30_l274_274797

noncomputable def prob_three_primes : ℚ := 
  let primes := {x ∈ Finset.range 31 | Nat.Prime x }.card
  let total_combinations := (Finset.range 31).card.choose 3
  let prime_combinations := primes.choose 3
  prime_combinations / total_combinations

theorem probability_three_primes_from_1_to_30 : prob_three_primes = 10 / 339 := 
  by
    sorry

end probability_three_primes_from_1_to_30_l274_274797


namespace largest_digit_divisible_by_6_l274_274191

theorem largest_digit_divisible_by_6 : ∃ N : ℕ, N ≤ 9 ∧ (∃ d : ℕ, 3456 * 10 + N = 6 * d) ∧ (∀ M : ℕ, M ≤ 9 ∧ (∃ d : ℕ, 3456 * 10 + M = 6 * d) → M ≤ N) :=
sorry

end largest_digit_divisible_by_6_l274_274191


namespace solve_rational_eq_l274_274280

theorem solve_rational_eq (x : ℝ) :
  (1 / (x^2 + 14*x - 36)) + (1 / (x^2 + 5*x - 14)) + (1 / (x^2 - 16*x - 36)) = 0 ↔ 
  x = 9 ∨ x = -4 ∨ x = 12 ∨ x = 3 :=
sorry

end solve_rational_eq_l274_274280


namespace fruits_eaten_l274_274347

theorem fruits_eaten (initial_cherries initial_strawberries initial_blueberries left_cherries left_strawberries left_blueberries : ℕ)
  (h1 : initial_cherries = 16) (h2 : initial_strawberries = 10) (h3 : initial_blueberries = 20)
  (h4 : left_cherries = 6) (h5 : left_strawberries = 8) (h6 : left_blueberries = 15) :
  (initial_cherries - left_cherries) + (initial_strawberries - left_strawberries) + (initial_blueberries - left_blueberries) = 17 := 
by
  sorry

end fruits_eaten_l274_274347


namespace ways_to_make_30_cents_is_17_l274_274920

-- Define the value of each type of coin
def penny_value : ℕ := 1
def nickel_value : ℕ := 5
def dime_value : ℕ := 10
def quarter_value : ℕ := 25

-- Define the function that counts the number of ways to make 30 cents
def count_ways_to_make_30_cents : ℕ :=
  let ways_with_1_quarter := (if 30 - quarter_value == 5 then 2 else 0)
  let ways_with_0_quarters :=
    let ways_with_2_dimes := (if 30 - 2 * dime_value == 10 then 3 else 0)
    let ways_with_1_dime := (if 30 - dime_value == 20 then 5 else 0)
    let ways_with_0_dimes := (if 30 == 30 then 7 else 0)
    ways_with_2_dimes + ways_with_1_dime + ways_with_0_dimes
  2 + ways_with_1_quarter + ways_with_0_quarters

-- Proof statement
theorem ways_to_make_30_cents_is_17 : count_ways_to_make_30_cents = 17 := sorry

end ways_to_make_30_cents_is_17_l274_274920


namespace infinite_primes_of_form_2px_plus_1_l274_274342

theorem infinite_primes_of_form_2px_plus_1 (p : ℕ) (hp : Nat.Prime p) (odd_p : p % 2 = 1) : 
  ∃ᶠ (n : ℕ) in at_top, Nat.Prime (2 * p * n + 1) :=
sorry

end infinite_primes_of_form_2px_plus_1_l274_274342


namespace least_positive_integer_l274_274040

theorem least_positive_integer (n : ℕ) : 
  (n % 4 = 1) ∧ (n % 5 = 2) ∧ (n % 6 = 3) → n = 57 := by
sorry

end least_positive_integer_l274_274040


namespace tree_ratio_l274_274962

theorem tree_ratio (native_trees : ℕ) (total_planted : ℕ) (M : ℕ) 
  (h1 : native_trees = 30) 
  (h2 : total_planted = 80) 
  (h3 : total_planted = M + M / 3) :
  (native_trees + M) / native_trees = 3 :=
sorry

end tree_ratio_l274_274962


namespace max_value_x2y_l274_274315

theorem max_value_x2y : 
  ∃ (x y : ℕ), 
    7 * x + 4 * y = 140 ∧
    (∀ (x' y' : ℕ),
       7 * x' + 4 * y' = 140 → 
       x' ^ 2 * y' ≤ x ^ 2 * y) ∧
    x ^ 2 * y = 2016 :=
by {
  sorry
}

end max_value_x2y_l274_274315


namespace fraction_proof_l274_274066

theorem fraction_proof (x y : ℕ) (h1 : y = 7) (h2 : x = 22) : 
  (y / (x - 1) = 1 / 3) ∧ ((y + 4) / x = 1 / 2) := by
  sorry

end fraction_proof_l274_274066


namespace a_eq_zero_l274_274780

theorem a_eq_zero (a b : ℤ) (h : ∀ n : ℕ, ∃ k : ℤ, 2^n * a + b = k^2) : a = 0 :=
sorry

end a_eq_zero_l274_274780


namespace find_other_polynomial_l274_274871

variables {a b c d : ℤ}

theorem find_other_polynomial (h : ∀ P Q : ℤ, P - Q = c^2 * d^2 - a^2 * b^2) 
  (P : ℤ) (hP : P = a^2 * b^2 + c^2 * d^2 - 2 * a * b * c * d) : 
  (∃ Q : ℤ, Q = 2 * c^2 * d^2 - 2 * a * b * c * d) ∨ 
  (∃ Q : ℤ, Q = 2 * a^2 * b^2 - 2 * a * b * c * d) :=
by {
  sorry
}

end find_other_polynomial_l274_274871


namespace polynomial_evaluation_l274_274257

theorem polynomial_evaluation :
  7^4 + 4 * 7^3 + 6 * 7^2 + 4 * 7 + 1 = 4096 :=
by
  sorry

end polynomial_evaluation_l274_274257


namespace cos_inequality_range_l274_274021

theorem cos_inequality_range (x : ℝ) (h₁ : 0 ≤ x) (h₂ : x ≤ 2 * Real.pi) (h₃ : Real.cos x ≤ 1 / 2) :
  x ∈ Set.Icc (Real.pi / 3) (5 * Real.pi / 3) := 
sorry

end cos_inequality_range_l274_274021


namespace infinite_non_expressible_integers_l274_274498

theorem infinite_non_expressible_integers :
  ∃ (S : Set ℤ), S.Infinite ∧ (∀ n ∈ S, ∀ a b c : ℕ, n ≠ 2^a + 3^b - 5^c) :=
sorry

end infinite_non_expressible_integers_l274_274498


namespace barrel_capacity_is_16_l274_274541

noncomputable def capacity_of_barrel (midway_tap_rate bottom_tap_rate used_bottom_tap_early_time assistant_use_time : Nat) : Nat :=
  let midway_draw := used_bottom_tap_early_time / midway_tap_rate
  let bottom_draw_assistant := assistant_use_time / bottom_tap_rate
  let total_extra_draw := midway_draw + bottom_draw_assistant
  2 * total_extra_draw

theorem barrel_capacity_is_16 :
  capacity_of_barrel 6 4 24 16 = 16 :=
by
  sorry

end barrel_capacity_is_16_l274_274541


namespace third_smallest_four_digit_in_pascals_triangle_l274_274217

-- Definitions for Pascal's Triangle and four-digit numbers
def is_in_pascals_triangle (n : ℕ) : Prop :=
  ∃ (r : ℕ) (k : ℕ), r.choose k = n

def is_four_digit (n : ℕ) : Prop :=
  1000 ≤ n ∧ n < 10000

-- Proposition stating the third smallest four-digit number in Pascal's Triangle
theorem third_smallest_four_digit_in_pascals_triangle :
  ∃ (n : ℕ), is_in_pascals_triangle n ∧ is_four_digit n ∧ 
  (∃ m1 m2 m3, is_in_pascals_triangle m1 ∧ is_four_digit m1 ∧ 
                is_in_pascals_triangle m2 ∧ is_four_digit m2 ∧ 
                is_in_pascals_triangle m3 ∧ is_four_digit m3 ∧ 
                1000 ≤ m1 ∧ m1 < 1001 ∧ 1001 ≤ m2 ∧ m2 < 1002 ∧ 1002 ≤ n ∧ n < 1003) ∧ 
  n = 1002 :=
sorry

end third_smallest_four_digit_in_pascals_triangle_l274_274217


namespace problem_solution_l274_274851

theorem problem_solution (x : ℝ) (h1 : x ≠ -2) (h2 : x ≠ 2):
    (x ∈ Set.Iio (-2) ∪ Set.Ioo (-2) ((1 - Real.sqrt 129)/8) ∪ Set.Ioo 2 3 ∪ Set.Ioi ((1 + (Real.sqrt 129))/8)) ↔
    (2 * x^2 / (x + 2) ≥ 3 / (x - 2) + 6 / 4) :=
by
  sorry

end problem_solution_l274_274851


namespace find_least_number_l274_274532

theorem find_least_number (x : ℕ) :
  (∀ k, 24 ∣ k + 7 → 32 ∣ k + 7 → 36 ∣ k + 7 → 54 ∣ k + 7 → x = k) → 
  x + 7 = Nat.lcm (Nat.lcm (Nat.lcm 24 32) 36) 54 → x = 857 :=
by
  sorry

end find_least_number_l274_274532


namespace third_smallest_four_digit_in_pascal_l274_274204

def is_four_digit (n : ℕ) : Prop := 1000 ≤ n ∧ n < 10000

def pascal (n k : ℕ) : ℕ := Nat.choose n k

theorem third_smallest_four_digit_in_pascal :
  ∃ n k : ℕ, is_four_digit (pascal n k) ∧ (pascal n k = 1002) :=
sorry

end third_smallest_four_digit_in_pascal_l274_274204


namespace minimum_value_y_l274_274584

variable {x y : ℝ}

theorem minimum_value_y (h : y * Real.log y = Real.exp (2 * x) - y * Real.log (2 * x)) : y ≥ Real.exp 1 :=
sorry

end minimum_value_y_l274_274584


namespace impossible_even_n_m_if_n3_plus_m3_is_odd_l274_274597

theorem impossible_even_n_m_if_n3_plus_m3_is_odd
  (n m : ℤ) (h : (n^3 + m^3) % 2 = 1) : ¬((n % 2 = 0) ∧ (m % 2 = 0)) := by
  sorry

end impossible_even_n_m_if_n3_plus_m3_is_odd_l274_274597


namespace least_positive_integer_solution_l274_274037

theorem least_positive_integer_solution :
  ∃ b : ℕ, b ≡ 1 [MOD 4] ∧ b ≡ 2 [MOD 5] ∧ b ≡ 3 [MOD 6] ∧ b = 37 :=
by
  sorry

end least_positive_integer_solution_l274_274037


namespace power_multiplication_l274_274255

theorem power_multiplication :
  2^4 * 5^4 = 10000 := 
by
  sorry

end power_multiplication_l274_274255


namespace tan_sum_trig_identity_l274_274288

variable {α : ℝ}

-- Part (I)
theorem tan_sum (h : Real.tan α = 2) : Real.tan (α + Real.pi / 4) = -3 :=
by
  sorry

-- Part (II)
theorem trig_identity (h : Real.tan α = 2) : 
  (Real.sin (2 * α) - Real.cos α ^ 2) / (1 + Real.cos (2 * α)) = 3 / 2 :=
by
  sorry

end tan_sum_trig_identity_l274_274288


namespace price_of_each_bracelet_l274_274368

-- The conditions
def bike_cost : ℕ := 112
def days_in_two_weeks : ℕ := 14
def bracelets_per_day : ℕ := 8
def total_bracelets := days_in_two_weeks * bracelets_per_day

-- The question and the expected answer
def price_per_bracelet : ℕ := bike_cost / total_bracelets

theorem price_of_each_bracelet :
  price_per_bracelet = 1 := 
by
  sorry

end price_of_each_bracelet_l274_274368


namespace vika_card_pairing_l274_274371

theorem vika_card_pairing :
  ∃ (d ∈ {1, 2, 3, 5, 6, 10, 15, 30}), ∃ (k : ℕ), 60 = 2 * d * k :=
by sorry

end vika_card_pairing_l274_274371


namespace slope_of_line_determined_by_any_two_solutions_l274_274385

theorem slope_of_line_determined_by_any_two_solutions 
  (x₁ y₁ x₂ y₂ : ℝ) 
  (h₁ : 4 / x₁ + 5 / y₁ = 0) 
  (h₂ : 4 / x₂ + 5 / y₂ = 0) 
  (h_distinct : x₁ ≠ x₂) : 
  (y₂ - y₁) / (x₂ - x₁) = -5 / 4 := 
sorry

end slope_of_line_determined_by_any_two_solutions_l274_274385


namespace stickers_per_page_l274_274516

theorem stickers_per_page (n_pages total_stickers : ℕ) (h_n_pages : n_pages = 22) (h_total_stickers : total_stickers = 220) : total_stickers / n_pages = 10 :=
by
  sorry

end stickers_per_page_l274_274516


namespace tom_wins_with_smallest_n_l274_274294

def tom_and_jerry_game_proof_problem (n : ℕ) : Prop :=
  ∀ (pos : ℕ), pos ≥ 1 ∧ pos ≤ 2018 → 
  ∀ (move : ℕ), move ≥ 1 ∧ move ≤ n →
  (∃ n_min : ℕ, n_min ≤ n ∧ ∀ pos, (pos ≤ n_min ∨ pos > 2018 - n_min) → false)

theorem tom_wins_with_smallest_n : tom_and_jerry_game_proof_problem 1010 :=
sorry

end tom_wins_with_smallest_n_l274_274294


namespace root_of_quadratic_eq_l274_274281

open Complex

theorem root_of_quadratic_eq :
  ∃ z1 z2 : ℂ, (z1 = 3.5 - I) ∧ (z2 = -2.5 + I) ∧ (∀ z : ℂ, z^2 - z = 6 - 6 * I → (z = z1 ∨ z = z2)) := 
sorry

end root_of_quadratic_eq_l274_274281


namespace largest_multiple_of_11_less_than_minus_150_l274_274530

theorem largest_multiple_of_11_less_than_minus_150 : 
  ∃ n : ℤ, (n * 11 < -150) ∧ (∀ m : ℤ, (m * 11 < -150) →  n * 11 ≥ m * 11) ∧ (n * 11 = -154) :=
by
  sorry

end largest_multiple_of_11_less_than_minus_150_l274_274530


namespace cobbler_hours_per_day_l274_274404

-- Defining some conditions based on our problem statement
def cobbler_rate : ℕ := 3  -- pairs of shoes per hour
def friday_hours : ℕ := 3  -- number of hours worked on Friday
def friday_pairs : ℕ := cobbler_rate * friday_hours  -- pairs mended on Friday
def weekly_pairs : ℕ := 105  -- total pairs mended in a week
def mon_thu_pairs : ℕ := weekly_pairs - friday_pairs  -- pairs mended from Monday to Thursday
def mon_thu_hours : ℕ := mon_thu_pairs / cobbler_rate  -- total hours worked from Monday to Thursday

-- Thm statement: If a cobbler works h hours daily from Mon to Thu, then h = 8 implies total = 105 pairs
theorem cobbler_hours_per_day (h : ℕ) : (4 * h = mon_thu_hours) ↔ (h = 8) :=
by
  sorry

end cobbler_hours_per_day_l274_274404


namespace total_players_is_28_l274_274540

def total_players (A B C AB BC AC ABC : ℕ) : ℕ :=
  A + B + C - (AB + BC + AC) + ABC

theorem total_players_is_28 :
  total_players 10 15 18 8 6 4 3 = 28 :=
by
  -- as per inclusion-exclusion principle
  -- T = A + B + C - (AB + BC + AC) + ABC
  -- substituting given values we repeatedly perform steps until final answer
  -- take user inputs to build your final answer.
  sorry

end total_players_is_28_l274_274540


namespace smallest_N_l274_274544

theorem smallest_N (N : ℕ) : (N * 3 ≥ 75) ∧ (N * 2 < 75) → N = 25 :=
by {
  sorry
}

end smallest_N_l274_274544


namespace cos_eq_neg_1_over_4_of_sin_eq_1_over_4_l274_274112

theorem cos_eq_neg_1_over_4_of_sin_eq_1_over_4
  (α : ℝ)
  (h : Real.sin (α + π / 3) = 1 / 4) :
  Real.cos (α + 5 * π / 6) = -1 / 4 :=
sorry

end cos_eq_neg_1_over_4_of_sin_eq_1_over_4_l274_274112


namespace archer_probability_less_than_8_l274_274587

-- Define the conditions as probabilities for hitting the 10-ring, 9-ring, and 8-ring.
def p_10 : ℝ := 0.24
def p_9 : ℝ := 0.28
def p_8 : ℝ := 0.19

-- Define the probability that the archer scores at least 8.
def p_at_least_8 : ℝ := p_10 + p_9 + p_8

-- Calculate the probability of the archer scoring less than 8.
def p_less_than_8 : ℝ := 1 - p_at_least_8

-- Now, state the theorem to prove that this probability is equal to 0.29.
theorem archer_probability_less_than_8 : p_less_than_8 = 0.29 := by sorry

end archer_probability_less_than_8_l274_274587


namespace ratio_of_areas_l274_274784

variable (l w r : ℝ)
variable (h1 : l = 2 * w)
variable (h2 : 6 * w = 2 * real.pi * r)

theorem ratio_of_areas : (l * w) / (real.pi * r^2) = 2 * real.pi / 9 :=
by sorry

end ratio_of_areas_l274_274784


namespace fraction_to_decimal_l274_274093

theorem fraction_to_decimal :
  (3 / 8 : ℝ) = 0.375 :=
sorry

end fraction_to_decimal_l274_274093


namespace perimeter_of_fence_l274_274092

noncomputable def n : ℕ := 18
noncomputable def w : ℝ := 0.5
noncomputable def d : ℝ := 4

theorem perimeter_of_fence : 3 * ((n / 3 - 1) * d + n / 3 * w) = 69 := by
  sorry

end perimeter_of_fence_l274_274092


namespace average_candies_correct_l274_274568

noncomputable def Eunji_candies : ℕ := 35
noncomputable def Jimin_candies : ℕ := Eunji_candies + 6
noncomputable def Jihyun_candies : ℕ := Eunji_candies - 3
noncomputable def Total_candies : ℕ := Eunji_candies + Jimin_candies + Jihyun_candies
noncomputable def Average_candies : ℚ := Total_candies / 3

theorem average_candies_correct :
  Average_candies = 36 := by
  sorry

end average_candies_correct_l274_274568


namespace stratified_sampling_correct_l274_274405

-- Definitions for the conditions
def total_employees : ℕ := 750
def young_employees : ℕ := 350
def middle_aged_employees : ℕ := 250
def elderly_employees : ℕ := 150
def sample_size : ℕ := 15
def sampling_proportion : ℚ := sample_size / total_employees

-- Statement to prove
theorem stratified_sampling_correct :
  (young_employees * sampling_proportion = 7) ∧
  (middle_aged_employees * sampling_proportion = 5) ∧
  (elderly_employees * sampling_proportion = 3) :=
by
  sorry

end stratified_sampling_correct_l274_274405


namespace total_arrangements_l274_274062

theorem total_arrangements (students communities : ℕ) 
  (h_students : students = 5) 
  (h_communities : communities = 3)
  (h_conditions :
    ∀(student : Fin students) (community : Fin communities), 
      true 
  ) : 150 = 150 :=
by sorry

end total_arrangements_l274_274062


namespace least_positive_integer_l274_274043

theorem least_positive_integer (n : ℕ) : 
  (n % 4 = 1) ∧ (n % 5 = 2) ∧ (n % 6 = 3) → n = 57 := by
sorry

end least_positive_integer_l274_274043


namespace perfect_square_trinomial_m_l274_274879

theorem perfect_square_trinomial_m (m : ℝ) :
  (∃ (a b : ℝ), ∀ x : ℝ, x^2 + (m-1)*x + 9 = (a*x + b)^2) ↔ (m = 7 ∨ m = -5) :=
sorry

end perfect_square_trinomial_m_l274_274879


namespace minimum_value_xy_minimum_value_x_plus_2y_l274_274898

-- (1) Prove that the minimum value of \(xy\) given \(x, y > 0\) and \(\frac{1}{x} + \frac{9}{y} = 1\) is \(36\).
theorem minimum_value_xy (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 1 / x + 9 / y = 1) : x * y ≥ 36 := 
sorry

-- (2) Prove that the minimum value of \(x + 2y\) given \(x, y > 0\) and \(\frac{1}{x} + \frac{9}{y} = 1\) is \(19 + 6\sqrt{2}\).
theorem minimum_value_x_plus_2y (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 1 / x + 9 / y = 1) : x + 2 * y ≥ 19 + 6 * Real.sqrt 2 := 
sorry

end minimum_value_xy_minimum_value_x_plus_2y_l274_274898


namespace find_a_n_l274_274295

noncomputable def is_arithmetic_seq (a b : ℕ) (seq : ℕ → ℕ) : Prop :=
  ∀ n, seq n = a + n * b

noncomputable def is_geometric_seq (b a : ℕ) (seq : ℕ → ℕ) : Prop :=
  ∀ n, seq n = b * a ^ n

theorem find_a_n (a b : ℕ) 
  (a_positive : a > 1)
  (b_positive : b > 1)
  (a_seq : ℕ → ℕ)
  (b_seq : ℕ → ℕ)
  (arith_seq : is_arithmetic_seq a b a_seq)
  (geom_seq : is_geometric_seq b a b_seq)
  (init_condition : a_seq 0 < b_seq 0)
  (next_condition : b_seq 1 < a_seq 2)
  (relation_condition : ∀ n, ∃ m, a_seq m + 3 = b_seq n) :
  ∀ n, a_seq n = 5 * n - 3 :=
sorry

end find_a_n_l274_274295


namespace students_who_did_not_receive_an_A_l274_274472

def total_students : ℕ := 40
def a_in_literature : ℕ := 10
def a_in_science : ℕ := 18
def a_in_both : ℕ := 6

theorem students_who_did_not_receive_an_A :
  total_students - ((a_in_literature + a_in_science) - a_in_both) = 18 :=
by
  sorry

end students_who_did_not_receive_an_A_l274_274472


namespace P_neither_l274_274313

-- Definition of probabilities according to given conditions
def P_A : ℝ := 0.63      -- Probability of answering the first question correctly
def P_B : ℝ := 0.50      -- Probability of answering the second question correctly
def P_A_and_B : ℝ := 0.33  -- Probability of answering both questions correctly

-- Theorem to prove the probability of answering neither of the questions correctly
theorem P_neither : (1 - (P_A + P_B - P_A_and_B)) = 0.20 := by
  sorry

end P_neither_l274_274313


namespace least_number_with_remainder_l274_274531

theorem least_number_with_remainder (N : ℕ) : (∃ k : ℕ, N = 12 * k + 4) → N = 256 :=
by
  intro h
  sorry

end least_number_with_remainder_l274_274531


namespace paper_pattern_after_unfolding_l274_274685

-- Define the number of layers after folding the square paper four times
def folded_layers (initial_layers : ℕ) : ℕ :=
  initial_layers * 2 ^ 4

-- Define the number of quarter-circles removed based on the layers
def quarter_circles_removed (layers : ℕ) : ℕ :=
  layers

-- Define the number of complete circles from the quarter circles
def complete_circles (quarter_circles : ℕ) : ℕ :=
  quarter_circles / 4

-- The main theorem that we need to prove
theorem paper_pattern_after_unfolding :
  (complete_circles (quarter_circles_removed (folded_layers 1)) = 4) :=
by
  sorry

end paper_pattern_after_unfolding_l274_274685


namespace smallest_integer_not_expressible_in_form_l274_274577

theorem smallest_integer_not_expressible_in_form :
  ∀ (n : ℕ), (0 < n ∧ (∀ a b c d : ℕ, n ≠ (2^a - 2^b) / (2^c - 2^d))) ↔ n = 11 :=
by
  sorry

end smallest_integer_not_expressible_in_form_l274_274577


namespace age_difference_l274_274510

theorem age_difference (sum_ages : ℕ) (eldest_age : ℕ) (age_diff : ℕ) 
(h1 : sum_ages = 50) (h2 : eldest_age = 14) :
  14 + (14 - age_diff) + (14 - 2 * age_diff) + (14 - 3 * age_diff) + (14 - 4 * age_diff) = 50 → age_diff = 2 := 
by
  intro h
  sorry

end age_difference_l274_274510


namespace minimum_ticket_cost_correct_l274_274056

noncomputable def minimum_ticket_cost : Nat :=
let adults := 8
let children := 4
let adult_ticket_price := 100
let child_ticket_price := 50
let group_ticket_price := 70
let group_size := 10
-- Calculate the cost of group tickets for 10 people and regular tickets for 2 children
let total_cost := (group_size * group_ticket_price) + (2 * child_ticket_price)
total_cost

theorem minimum_ticket_cost_correct :
  minimum_ticket_cost = 800 := by
  sorry

end minimum_ticket_cost_correct_l274_274056


namespace real_solution_count_l274_274702

noncomputable def f (x : ℝ) : ℝ :=
  (1/(x - 1)) + (2/(x - 2)) + (3/(x - 3)) + (4/(x - 4)) + 
  (5/(x - 5)) + (6/(x - 6)) + (7/(x - 7)) + (8/(x - 8)) + 
  (9/(x - 9)) + (10/(x - 10))

theorem real_solution_count : ∃ n : ℕ, n = 11 ∧ 
  ∃ x : ℝ, f x = x :=
sorry

end real_solution_count_l274_274702


namespace absolute_error_2175000_absolute_error_1730000_l274_274889

noncomputable def absolute_error (a : ℕ) : ℕ :=
  if a = 2175000 then 1
  else if a = 1730000 then 10000
  else 0

theorem absolute_error_2175000 : absolute_error 2175000 = 1 :=
by sorry

theorem absolute_error_1730000 : absolute_error 1730000 = 10000 :=
by sorry

end absolute_error_2175000_absolute_error_1730000_l274_274889


namespace average_person_funding_l274_274642

-- Define the conditions from the problem
def total_amount_needed : ℝ := 1000
def amount_already_have : ℝ := 200
def number_of_people : ℝ := 80

-- Define the correct answer
def average_funding_per_person : ℝ := 10

-- Formulate the proof statement
theorem average_person_funding :
  (total_amount_needed - amount_already_have) / number_of_people = average_funding_per_person :=
by
  sorry

end average_person_funding_l274_274642


namespace arithmetic_sequence_sum_l274_274135

theorem arithmetic_sequence_sum (a : ℕ → ℕ) (d : ℕ → ℕ)
  (is_arithmetic_seq : ∀ n, a (n + 1) = a n + d n)
  (h : (a 2) + (a 5) + (a 8) = 39) :
  (a 1) + (a 2) + (a 3) + (a 4) + (a 5) + (a 6) + (a 7) + (a 8) + (a 9) = 117 := 
sorry

end arithmetic_sequence_sum_l274_274135


namespace greatest_divisor_of_arithmetic_sum_l274_274806

theorem greatest_divisor_of_arithmetic_sum (a d : ℕ) (ha : 0 < a) (hd : 0 < d) :
  ∃ k : ℕ, k = 6 ∧ ∀ a d : ℕ, 12 * a + 66 * d % k = 0 :=
by sorry

end greatest_divisor_of_arithmetic_sum_l274_274806


namespace count_negative_numbers_l274_274074

theorem count_negative_numbers : 
  (List.filter (λ x => x < (0:ℚ)) [-14, 7, 0, -2/3, -5/16]).length = 3 := 
by
  sorry

end count_negative_numbers_l274_274074


namespace katie_books_ratio_l274_274268

theorem katie_books_ratio
  (d : ℕ)
  (k : ℚ)
  (g : ℕ)
  (total_books : ℕ)
  (hd : d = 6)
  (hk : ∃ k : ℚ, k = (k : ℚ))
  (hg : g = 5 * (d + k * d))
  (ht : total_books = d + k * d + g)
  (htotal : total_books = 54) :
  k = 1 / 2 :=
by
  sorry

end katie_books_ratio_l274_274268


namespace new_student_weight_l274_274503

theorem new_student_weight (W_new : ℝ) (W : ℝ) (avg_decrease : ℝ) (num_students : ℝ) (old_weight : ℝ) (new_weight : ℝ) :
  avg_decrease = 5 → old_weight = 86 → num_students = 8 →
  W_new = W - old_weight + new_weight → W_new = W - avg_decrease * num_students →
  new_weight = 46 :=
by
  intros avg_decrease_eq old_weight_eq num_students_eq W_new_eq avg_weight_decrease_eq
  rw [avg_decrease_eq, old_weight_eq, num_students_eq] at *
  sorry

end new_student_weight_l274_274503


namespace average_of_values_l274_274695

theorem average_of_values (z : ℝ) : 
  (0 + 1 + 2 + 4 + 8 + 32 : ℝ) * z / (6 : ℝ) = 47 * z / 6 :=
by
  sorry

end average_of_values_l274_274695


namespace factor_evaluate_l274_274845

theorem factor_evaluate (a b : ℤ) (h1 : a = 2) (h2 : b = -2) : 
  5 * a * (b - 2) + 2 * a * (2 - b) = -24 := by
  sorry

end factor_evaluate_l274_274845


namespace proof_problem_l274_274650

noncomputable def polar_to_cartesian_O1 : Prop :=
  ∀ (ρ : ℝ) (θ : ℝ), ρ = 4 * Real.cos θ → (ρ^2 = 4 * ρ * Real.cos θ)

noncomputable def cartesian_O1 : Prop :=
  ∀ (x y : ℝ), x^2 + y^2 = 4 * x → x^2 + y^2 - 4 * x = 0

noncomputable def polar_to_cartesian_O2 : Prop :=
  ∀ (ρ : ℝ) (θ : ℝ), ρ = -4 * Real.sin θ → (ρ^2 = -4 * ρ * Real.sin θ)

noncomputable def cartesian_O2 : Prop :=
  ∀ (x y : ℝ), x^2 + y^2 = -4 * y → x^2 + y^2 + 4 * y = 0

noncomputable def intersections_O1_O2 : Prop :=
  ∀ (x y : ℝ), (x^2 + y^2 - 4 * x = 0) ∧ (x^2 + y^2 + 4 * y = 0) →
  (x = 0 ∧ y = 0) ∨ (x = 2 ∧ y = -2)

noncomputable def line_through_intersections : Prop :=
  ∀ (x y : ℝ), ((x = 0 ∧ y = 0) ∨ (x = 2 ∧ y = -2)) → y = -x

theorem proof_problem : polar_to_cartesian_O1 ∧ cartesian_O1 ∧ polar_to_cartesian_O2 ∧ cartesian_O2 ∧ intersections_O1_O2 ∧ line_through_intersections :=
  sorry

end proof_problem_l274_274650


namespace remainder_of_3y_l274_274668

theorem remainder_of_3y (y : ℕ) (hy : y % 9 = 5) : (3 * y) % 9 = 6 :=
sorry

end remainder_of_3y_l274_274668


namespace problems_per_page_l274_274154

def total_problems : ℕ := 72
def finished_problems : ℕ := 32
def remaining_pages : ℕ := 5
def remaining_problems : ℕ := total_problems - finished_problems

theorem problems_per_page : remaining_problems / remaining_pages = 8 := 
by
  sorry

end problems_per_page_l274_274154


namespace notebooks_per_child_if_half_l274_274861

theorem notebooks_per_child_if_half (C N : ℕ) 
    (h1 : N = C / 8) 
    (h2 : C * N = 512) : 
    512 / (C / 2) = 16 :=
by
    sorry

end notebooks_per_child_if_half_l274_274861


namespace third_smallest_four_digit_in_pascals_triangle_is_1002_l274_274205

theorem third_smallest_four_digit_in_pascals_triangle_is_1002 :
  ∃ n k, 1000 <= binomial n k ∧ binomial n k <= 9999 ∧ (third_smallest (λ n k, 1000 <= binomial n k ∧ binomial n k <= 9999) = 1002) :=
sorry

end third_smallest_four_digit_in_pascals_triangle_is_1002_l274_274205


namespace male_teacher_classes_per_month_l274_274182

theorem male_teacher_classes_per_month (x y a : ℕ) :
  (15 * x = 6 * (x + y)) ∧ (a * y = 6 * (x + y)) → a = 10 :=
by
  sorry

end male_teacher_classes_per_month_l274_274182


namespace complex_norm_wz_l274_274113

open Complex

theorem complex_norm_wz (w z : ℂ) (h₁ : ‖w + z‖ = 2) (h₂ : ‖w^2 + z^2‖ = 8) : 
  ‖w^4 + z^4‖ = 56 := 
  sorry

end complex_norm_wz_l274_274113


namespace min_ab_l274_274505

theorem min_ab (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : 1 / a + 1 / b = 1) : ab = 4 :=
  sorry

end min_ab_l274_274505


namespace largest_square_area_l274_274147

theorem largest_square_area (total_string_length : ℕ) (h : total_string_length = 32) : ∃ (area : ℕ), area = 64 := 
  by
    sorry

end largest_square_area_l274_274147


namespace cube_edge_length_l274_274025

theorem cube_edge_length (total_edge_length : ℕ) (num_edges : ℕ) (h1 : total_edge_length = 108) (h2 : num_edges = 12) : total_edge_length / num_edges = 9 := by 
  -- additional formal mathematical steps can follow here
  sorry

end cube_edge_length_l274_274025


namespace books_got_rid_of_l274_274551

-- Define the number of books they originally had
def original_books : ℕ := 87

-- Define the number of shelves used
def shelves_used : ℕ := 9

-- Define the number of books per shelf
def books_per_shelf : ℕ := 6

-- Define the number of books left after placing them on shelves
def remaining_books : ℕ := shelves_used * books_per_shelf

-- The statement to prove
theorem books_got_rid_of : original_books - remaining_books = 33 := 
by 
-- here is proof body you need to fill in 
  sorry

end books_got_rid_of_l274_274551


namespace find_a5_l274_274325

def arithmetic_sequence (a : ℕ → ℤ) (d : ℤ) : Prop :=
  ∀ n : ℕ, n > 0 → a n = a 1 + (n - 1) * d

theorem find_a5 (a : ℕ → ℤ) (d : ℤ)
  (h_seq : arithmetic_sequence a d)
  (h1 : a 1 + a 5 = 8)
  (h4 : a 4 = 7) : 
  a 5 = 10 := sorry

end find_a5_l274_274325


namespace initial_earning_members_l274_274502

theorem initial_earning_members (average_income_before: ℝ) (average_income_after: ℝ) (income_deceased: ℝ) (n: ℝ)
    (H1: average_income_before = 735)
    (H2: average_income_after = 650)
    (H3: income_deceased = 990)
    (H4: n * average_income_before - (n - 1) * average_income_after = income_deceased)
    : n = 4 := 
by 
  rw [H1, H2, H3] at H4
  linarith


end initial_earning_members_l274_274502


namespace binomial_expansion_value_calculation_result_final_result_l274_274261

theorem binomial_expansion_value :
  7^4 + 4 * 7^3 + 6 * 7^2 + 4 * 7 + 1 = (7 + 1)^4 := 
sorry

theorem calculation_result :
  (7 + 1)^4 = 4096 := 
sorry

theorem final_result :
  7^4 + 4 * 7^3 + 6 * 7^2 + 4 * 7 + 1 = 4096 := 
by
  calc
    7^4 + 4 * 7^3 + 6 * 7^2 + 4 * 7 + 1 = (7 + 1)^4 := binomial_expansion_value
    ... = 4096 := calculation_result

end binomial_expansion_value_calculation_result_final_result_l274_274261


namespace part1_part2_l274_274721

noncomputable def f (x : ℝ) (m : ℝ) : ℝ := 
  Real.log x + 0.5 * m * x^2 - 2 

def perpendicular_slope_condition (m : ℝ) : Prop := 
  let k := (1 / 1 + m)
  k = -1 / 2

def inequality_condition (m : ℝ) : Prop := 
  ∀ x > 0, 
  Real.log x - 0.5 * m * x^2 + (1 - m) * x + 1 ≤ 0

theorem part1 : perpendicular_slope_condition (-3/2) :=
  sorry

theorem part2 : ∃ m : ℤ, m ≥ 2 ∧ inequality_condition m :=
  sorry

end part1_part2_l274_274721


namespace line_segment_parametric_curve_l274_274778

noncomputable def parametric_curve (θ : ℝ) := 
  (2 + Real.cos θ ^ 2, 1 - Real.sin θ ^ 2)

theorem line_segment_parametric_curve : 
  (∀ θ : ℝ, 0 ≤ θ ∧ θ < 2 * Real.pi → 
    ∃ x y : ℝ, (x, y) = parametric_curve θ ∧ 2 ≤ x ∧ x ≤ 3 ∧ x - y = 2) := 
sorry

end line_segment_parametric_curve_l274_274778


namespace velocity_of_current_l274_274682

theorem velocity_of_current (v : ℝ) 
  (row_speed : ℝ) (distance : ℝ) (total_time : ℝ) 
  (h_row_speed : row_speed = 5)
  (h_distance : distance = 2.4)
  (h_total_time : total_time = 1)
  (h_equation : distance / (row_speed + v) + distance / (row_speed - v) = total_time) :
  v = 1 :=
sorry

end velocity_of_current_l274_274682


namespace zoo_visitors_sunday_l274_274768

-- Definitions based on conditions
def friday_visitors : ℕ := 1250
def saturday_multiplier : ℚ := 3
def sunday_decrease_percent : ℚ := 0.15

-- Assert the equivalence
theorem zoo_visitors_sunday : 
  let saturday_visitors := friday_visitors * saturday_multiplier
  let sunday_visitors := saturday_visitors * (1 - sunday_decrease_percent)
  round (sunday_visitors : ℚ) = 3188 :=
by
  sorry

end zoo_visitors_sunday_l274_274768


namespace number_of_ways_to_select_president_and_vice_president_l274_274465

-- Define the given conditions
def num_candidates : Nat := 4

-- Define the problem to prove
theorem number_of_ways_to_select_president_and_vice_president : (num_candidates * (num_candidates - 1)) = 12 :=
by
  -- This is where the proof would go, but we are skipping it
  sorry

end number_of_ways_to_select_president_and_vice_president_l274_274465


namespace polynomial_expansion_sum_eq_l274_274619

theorem polynomial_expansion_sum_eq :
  (∀ (x : ℝ), (2 * x - 1)^5 = a_0 + a_1 * (x - 1) + a_2 * (x - 1)^2 + a_3 * (x - 1)^3 + a_4 * (x - 1)^4 + a_5 * (x - 1)^5) →
  (a_0 + a_1 + a_2 + a_3 + a_4 + a_5 = 243) :=
by
  sorry

end polynomial_expansion_sum_eq_l274_274619


namespace number_of_teams_in_league_l274_274319

theorem number_of_teams_in_league (n : ℕ) :
  (6 * n * (n - 1)) / 2 = 396 ↔ n = 12 :=
by
  sorry

end number_of_teams_in_league_l274_274319


namespace p_and_q_necessary_not_sufficient_l274_274126

variable (a m x : ℝ) (P Q : Prop)

def p (a m : ℝ) : Prop := a < 0 ∧ m^2 - 4 * a * m + 3 * a^2 < 0

def q (m : ℝ) : Prop := ∀ x > 0, x + 4 / x ≥ 1 - m

theorem p_and_q_necessary_not_sufficient :
  (∀ (a m : ℝ), p a m → q m) ∧ (∀ a : ℝ, -1 ≤ a ∧ a < 0) :=
sorry

end p_and_q_necessary_not_sufficient_l274_274126


namespace identical_dice_probability_l274_274799

def num_ways_to_paint_die : ℕ := 3^6

def total_ways_to_paint_dice (n : ℕ) : ℕ := (num_ways_to_paint_die ^ n)

def count_identical_ways : ℕ := 1 + 324 + 8100

def probability_identical_dice : ℚ :=
  (count_identical_ways : ℚ) / (total_ways_to_paint_dice 2 : ℚ)

theorem identical_dice_probability : probability_identical_dice = 8425 / 531441 := by
  sorry

end identical_dice_probability_l274_274799


namespace minimum_value_expression_l274_274853

theorem minimum_value_expression (x : ℝ) : (x^2 + 9) / Real.sqrt (x^2 + 5) ≥ 4 :=
by
  sorry

end minimum_value_expression_l274_274853


namespace simplify_expr_for_a_neq_0_1_neg1_final_value_when_a_2_l274_274006

theorem simplify_expr_for_a_neq_0_1_neg1 (a : ℝ) (h1 : a ≠ 1) (h0 : a ≠ 0) (h_neg1 : a ≠ -1) :
  ( (a - 1)^2 / ((a + 1) * (a - 1)) ) / (a - (2 * a / (a + 1))) = 1 / a := by
  sorry

theorem final_value_when_a_2 :
  ( (2 - 1)^2 / ((2 + 1) * (2 - 1)) ) / (2 - (2 * 2 / (2 + 1))) = 1 / 2 := by
  sorry

end simplify_expr_for_a_neq_0_1_neg1_final_value_when_a_2_l274_274006


namespace seq_fixed_point_l274_274293

theorem seq_fixed_point (a_0 b_0 : ℝ) (a b : ℕ → ℝ)
  (h1 : a 0 = a_0)
  (h2 : b 0 = b_0)
  (h3 : ∀ n, a (n + 1) = a n + b n)
  (h4 : ∀ n, b (n + 1) = a n * b n) :
  a 2022 = a_0 ∧ b 2022 = b_0 ↔ b_0 = 0 := sorry

end seq_fixed_point_l274_274293


namespace stickers_per_page_l274_274517

theorem stickers_per_page (n_pages total_stickers : ℕ) (h_n_pages : n_pages = 22) (h_total_stickers : total_stickers = 220) : total_stickers / n_pages = 10 :=
by
  sorry

end stickers_per_page_l274_274517


namespace trains_cross_time_l274_274950

def L : ℕ := 120 -- Length of each train in meters

def t1 : ℕ := 10 -- Time for the first train to cross the telegraph post in seconds
def t2 : ℕ := 12 -- Time for the second train to cross the telegraph post in seconds

def V1 : ℕ := L / t1 -- Speed of the first train (in m/s)
def V2 : ℕ := L / t2 -- Speed of the second train (in m/s)

def Vr : ℕ := V1 + V2 -- Relative speed when traveling in opposite directions

def TotalDistance : ℕ := 2 * L -- Total distance when both trains cross each other

def T : ℚ := TotalDistance / Vr -- Time for the trains to cross each other

theorem trains_cross_time : T = 11 := sorry

end trains_cross_time_l274_274950


namespace third_smallest_four_digit_in_pascal_l274_274202

def is_four_digit (n : ℕ) : Prop := 1000 ≤ n ∧ n < 10000

def pascal (n k : ℕ) : ℕ := Nat.choose n k

theorem third_smallest_four_digit_in_pascal :
  ∃ n k : ℕ, is_four_digit (pascal n k) ∧ (pascal n k = 1002) :=
sorry

end third_smallest_four_digit_in_pascal_l274_274202


namespace max_remainder_l274_274131

theorem max_remainder (y : ℕ) : 
  ∃ q r : ℕ, y = 11 * q + r ∧ r < 11 ∧ r = 10 := by sorry

end max_remainder_l274_274131


namespace muffins_sugar_l274_274413

theorem muffins_sugar (cups_muffins_ratio : 24 * 3 = 72 * s / 9) : s = 9 := by
  sorry

end muffins_sugar_l274_274413


namespace max_true_statements_l274_274164

theorem max_true_statements 
  (a b : ℝ) 
  (cond1 : a > 0) 
  (cond2 : b > 0) : 
  ( 
    ( (1 / a > 1 / b) ∧ (a^2 < b^2) 
      ∧ (a > b) ∧ (a > 0) ∧ (b > 0) ) 
    ∨ 
    ( (1 / a > 1 / b) ∧ ¬(a^2 < b^2) 
      ∧ (a > b) ∧ (a > 0) ∧ (b > 0) ) 
    ∨ 
    ( ¬(1 / a > 1 / b) ∧ (a^2 < b^2) 
      ∧ (a > b) ∧ (a > 0) ∧ (b > 0) ) 
    ∨ 
    ( ¬(1 / a > 1 / b) ∧ ¬(a^2 < b^2) 
      ∧ (a > b) ∧ (a > 0) ∧ (b > 0) ) 
  ) 
→ 
  (true ∧ true ∧ true ∧ true → 4 = 4) :=
sorry

end max_true_statements_l274_274164


namespace solution_set_of_inequality_l274_274362

theorem solution_set_of_inequality :
  {x : ℝ | (x + 1) / x ≤ 3} = {x : ℝ | x < 0} ∪ {x : ℝ | x ≥ 1 / 2} :=
by
  sorry

end solution_set_of_inequality_l274_274362


namespace degrees_subtraction_l274_274974

theorem degrees_subtraction :
  (108 * 3600 + 18 * 60 + 25) - (56 * 3600 + 23 * 60 + 32) = (51 * 3600 + 54 * 60 + 53) :=
by sorry

end degrees_subtraction_l274_274974


namespace clock_correction_l274_274831

def gain_per_day : ℚ := 13 / 4
def hours_per_day : ℕ := 24
def days_passed : ℕ := 9
def extra_hours : ℕ := 8
def total_hours : ℕ := days_passed * hours_per_day + extra_hours
def gain_per_hour : ℚ := gain_per_day / hours_per_day
def total_gain : ℚ := total_hours * gain_per_hour
def required_correction : ℚ := 30.33

theorem clock_correction :
  total_gain = required_correction :=
  by sorry

end clock_correction_l274_274831


namespace factorization_of_expression_l274_274984

theorem factorization_of_expression (a b c : ℝ) : 
  ((a^3 - b^3)^3 + (b^3 - c^3)^3 + (c^3 - a^3)^3) / 
  ((a - b)^3 + (b - c)^3 + (c - a)^3) = 
  (a^2 + ab + b^2) * (b^2 + bc + c^2) * (c^2 + ca + a^2) :=
by
  sorry

end factorization_of_expression_l274_274984


namespace exists_base_and_digit_l274_274608

def valid_digit_in_base (B : ℕ) (V : ℕ) : Prop :=
  V^2 % B = V ∧ V ≠ 0 ∧ V ≠ 1

theorem exists_base_and_digit :
  ∃ B V, valid_digit_in_base B V :=
by {
  sorry
}

end exists_base_and_digit_l274_274608


namespace arcsin_one_eq_pi_div_two_l274_274082

theorem arcsin_one_eq_pi_div_two : Real.arcsin 1 = (Real.pi / 2) :=
by
  sorry

end arcsin_one_eq_pi_div_two_l274_274082


namespace candy_bars_per_bag_l274_274179

theorem candy_bars_per_bag (total_candy_bars : ℕ) (number_of_bags : ℕ) (h1 : total_candy_bars = 15) (h2 : number_of_bags = 5) : total_candy_bars / number_of_bags = 3 :=
by
  sorry

end candy_bars_per_bag_l274_274179


namespace geometric_sequence_characterization_l274_274807

theorem geometric_sequence_characterization (a : ℕ → ℝ) :
  (∀ n, a (n + 1) / a n = (a (n + 2) - a (n + 1)) / (a (n + 1) - a n)) →
  (∃ r : ℝ, ∀ n, a (n + 1) = r * a n) :=
by
  sorry

end geometric_sequence_characterization_l274_274807


namespace find_c_share_l274_274496

noncomputable def shares (a b c d : ℝ) : Prop :=
  (5 * a = 4 * c) ∧ (7 * b = 4 * c) ∧ (2 * d = 4 * c) ∧ (a + b + c + d = 1200)

theorem find_c_share (A B C D : ℝ) (h : shares A B C D) : C = 275 :=
  by
  sorry

end find_c_share_l274_274496


namespace first_even_number_l274_274145

theorem first_even_number (x : ℤ) (h : x + (x + 2) + (x + 4) = 1194) : x = 396 :=
by
  -- the proof is skipped as per instructions
  sorry

end first_even_number_l274_274145


namespace lisa_total_spoons_l274_274169

def children_count : ℕ := 6
def spoons_per_child : ℕ := 4
def decorative_spoons : ℕ := 4
def large_spoons : ℕ := 20
def dessert_spoons : ℕ := 10
def soup_spoons : ℕ := 15
def tea_spoons : ℕ := 25

def baby_spoons_total : ℕ := children_count * spoons_per_child
def cutlery_set_total : ℕ := large_spoons + dessert_spoons + soup_spoons + tea_spoons

def total_spoons : ℕ := cutlery_set_total + baby_spoons_total + decorative_spoons

theorem lisa_total_spoons : total_spoons = 98 :=
by
  sorry

end lisa_total_spoons_l274_274169


namespace dolphins_to_be_trained_next_month_l274_274513

-- Definition of conditions
def total_dolphins : ℕ := 20
def fraction_fully_trained := 1 / 4
def fraction_currently_training := 2 / 3

-- Lean 4 statement for the proof problem
theorem dolphins_to_be_trained_next_month :
  let fully_trained := total_dolphins * fraction_fully_trained
  let remaining := total_dolphins - fully_trained
  let currently_training := remaining * fraction_currently_training
  remaining - currently_training = 5 := by
begin
  -- Calculation core based on the given conditions
  let fully_trained := total_dolphins * fraction_fully_trained,
  let remaining := total_dolphins - fully_trained,
  let currently_training := remaining * fraction_currently_training,
  show remaining - currently_training = 5,
  sorry  -- Proof should go here
end

end dolphins_to_be_trained_next_month_l274_274513


namespace cats_weigh_more_by_5_kg_l274_274724

def puppies_weight (num_puppies : ℕ) (weight_per_puppy : ℝ) : ℝ :=
  num_puppies * weight_per_puppy

def cats_weight (num_cats : ℕ) (weight_per_cat : ℝ) : ℝ :=
  num_cats * weight_per_cat

theorem cats_weigh_more_by_5_kg :
  puppies_weight 4 7.5  = 30 ∧ cats_weight 14 2.5 = 35 → (cats_weight 14 2.5 - puppies_weight 4 7.5 = 5) := 
by
  intro h
  sorry

end cats_weigh_more_by_5_kg_l274_274724


namespace constant_term_of_second_eq_l274_274285

theorem constant_term_of_second_eq (x y : ℝ) 
  (h1 : 7*x + y = 19) 
  (h2 : 2*x + y = 5) : 
  ∃ k : ℝ, x + 3*y = k ∧ k = 15 := 
by
  sorry

end constant_term_of_second_eq_l274_274285


namespace trig_identity_l274_274438

-- Proving the equality (we state the problem here)
theorem trig_identity :
  Real.sin (40 * Real.pi / 180) * (Real.tan (10 * Real.pi / 180) - Real.sqrt 3) = -8 / 3 :=
by
  sorry

end trig_identity_l274_274438


namespace smaller_cuboid_length_l274_274065

-- Definitions based on conditions
def original_cuboid_volume : ℝ := 18 * 15 * 2
def smaller_cuboid_volume (L : ℝ) : ℝ := 4 * 3 * L
def smaller_cuboids_total_volume (L : ℝ) : ℝ := 7.5 * smaller_cuboid_volume L

-- Theorem statement
theorem smaller_cuboid_length :
  ∃ L : ℝ, smaller_cuboids_total_volume L = original_cuboid_volume ∧ L = 6 := 
by
  sorry

end smaller_cuboid_length_l274_274065


namespace initial_notebooks_l274_274881

variable (a n : ℕ)
variable (h1 : n = 13 * a + 8)
variable (h2 : n = 15 * a)

theorem initial_notebooks : n = 60 := by
  -- additional details within the proof
  sorry

end initial_notebooks_l274_274881


namespace least_positive_integer_l274_274042

theorem least_positive_integer (n : ℕ) : 
  (n % 4 = 1) ∧ (n % 5 = 2) ∧ (n % 6 = 3) → n = 57 := by
sorry

end least_positive_integer_l274_274042


namespace slope_of_line_determined_by_any_two_solutions_l274_274386

theorem slope_of_line_determined_by_any_two_solutions 
  (x₁ y₁ x₂ y₂ : ℝ) 
  (h₁ : 4 / x₁ + 5 / y₁ = 0) 
  (h₂ : 4 / x₂ + 5 / y₂ = 0) 
  (h_distinct : x₁ ≠ x₂) : 
  (y₂ - y₁) / (x₂ - x₁) = -5 / 4 := 
sorry

end slope_of_line_determined_by_any_two_solutions_l274_274386


namespace rational_smaller_than_neg_half_l274_274637

theorem rational_smaller_than_neg_half : ∃ q : ℚ, q < -1/2 := by
  use (-1 : ℚ)
  sorry

end rational_smaller_than_neg_half_l274_274637


namespace expression_equals_k_times_10_pow_1007_l274_274262

theorem expression_equals_k_times_10_pow_1007 :
  (3^1006 + 7^1007)^2 - (3^1006 - 7^1007)^2 = 588 * 10^1007 := by
  sorry

end expression_equals_k_times_10_pow_1007_l274_274262


namespace faster_pump_rate_ratio_l274_274523

theorem faster_pump_rate_ratio (S F : ℝ) 
  (h1 : S + F = 1/5) 
  (h2 : S = 1/12.5) : F / S = 1.5 :=
by
  sorry

end faster_pump_rate_ratio_l274_274523


namespace simplify_and_evaluate_expression_l274_274175

theorem simplify_and_evaluate_expression (x : ℝ) (h : x = Real.sqrt 2 - 1) :
  (1 - (x + 1) / x) / ((x^2 - 1) / (x^2 - x)) = -Real.sqrt 2 / 2 := by
  sorry

end simplify_and_evaluate_expression_l274_274175


namespace quadrilateral_inequality_l274_274956

variable (x1 y1 x2 y2 x3 y3 x4 y4 : ℝ)

def distance_squared (x1 y1 x2 y2 : ℝ) : ℝ := (x1 - x2)^2 + (y1 - y2)^2

-- Given a convex quadrilateral ABCD with vertices at (x1, y1), (x2, y2), (x3, y3), and (x4, y4), prove:
theorem quadrilateral_inequality :
  (distance_squared x1 y1 x3 y3 + distance_squared x2 y2 x4 y4) ≤
  (distance_squared x1 y1 x2 y2 + distance_squared x2 y2 x3 y3 +
   distance_squared x3 y3 x4 y4 + distance_squared x4 y4 x1 y1) :=
sorry

end quadrilateral_inequality_l274_274956


namespace intersection_A_B_l274_274337

def A := {x : ℝ | x^2 - ⌊x⌋ = 2}
def B := {x : ℝ | -2 < x ∧ x < 2}

theorem intersection_A_B :
  A ∩ B = {-1, Real.sqrt 3} :=
sorry

end intersection_A_B_l274_274337


namespace pure_imaginary_condition_l274_274870

-- Define the problem
theorem pure_imaginary_condition (θ : ℝ) :
  (∀ k : ℤ, θ = (3 * Real.pi / 4) + k * Real.pi) →
  ∀ z : ℂ, z = (Complex.cos θ - Complex.sin θ * Complex.I) * (1 + Complex.I) →
  ∃ k : ℤ, θ = (3 * Real.pi / 4) + k * Real.pi → 
  (Complex.re z = 0 ∧ Complex.im z ≠ 0) :=
  sorry

end pure_imaginary_condition_l274_274870


namespace cats_weight_more_than_puppies_l274_274726

theorem cats_weight_more_than_puppies :
  let num_puppies := 4
  let weight_per_puppy := 7.5
  let num_cats := 14
  let weight_per_cat := 2.5
  (num_cats * weight_per_cat) - (num_puppies * weight_per_puppy) = 5 :=
by 
  let num_puppies := 4
  let weight_per_puppy := 7.5
  let num_cats := 14
  let weight_per_cat := 2.5
  sorry

end cats_weight_more_than_puppies_l274_274726


namespace find_triples_l274_274701

-- Definitions of the problem conditions
def is_prime (p : ℕ) : Prop := Nat.Prime p

def factorial (n : ℕ) : ℕ := 
  if n = 0 then 1 else n * factorial (n - 1)

def satisfies_equation (a b p : ℕ) : Prop := a^p = factorial b + p

-- The main theorem statement based on the problem conditions
theorem find_triples :
  (satisfies_equation 2 2 2 ∧ is_prime 2) ∧
  (satisfies_equation 3 4 3 ∧ is_prime 3) ∧
  (∀ (a b p : ℕ), (satisfies_equation a b p ∧ is_prime p) → (a, b, p) = (2, 2, 2) ∨ (a, b, p) = (3, 4, 3)) :=
by
  -- Proof to be filled
  sorry

end find_triples_l274_274701


namespace train_speed_l274_274536

theorem train_speed (L V : ℝ) (h1 : L = V * 10) (h2 : L + 500 = V * 35) : V = 20 :=
by {
  -- Proof is skipped with sorry
  sorry
}

end train_speed_l274_274536


namespace calculate_mari_buttons_l274_274764

theorem calculate_mari_buttons (Sue_buttons : ℕ) (Kendra_buttons : ℕ) (Mari_buttons : ℕ)
  (h_sue : Sue_buttons = 6)
  (h_kendra : Sue_buttons = 0.5 * Kendra_buttons)
  (h_mari : Mari_buttons = 4 + 5 * Kendra_buttons) :
  Mari_buttons = 64 := 
by
  sorry

end calculate_mari_buttons_l274_274764


namespace proof_problem_l274_274603

-- Define the system of equations
def system_of_equations (x y a : ℝ) : Prop :=
  (3 * x + y = 2 + 3 * a) ∧ (x + 3 * y = 2 + a)

-- Define the condition x + y < 0
def condition (x y : ℝ) : Prop := x + y < 0

-- Prove that if the system of equations has a solution with x + y < 0, then a < -1 and |1 - a| + |a + 1 / 2| = 1 / 2 - 2 * a
theorem proof_problem (x y a : ℝ) (h1 : system_of_equations x y a) (h2 : condition x y) :
  a < -1 ∧ |1 - a| + |a + 1 / 2| = (1 / 2) - 2 * a := 
sorry

end proof_problem_l274_274603


namespace geometric_progression_quadrilateral_exists_l274_274614

theorem geometric_progression_quadrilateral_exists :
  ∃ (a1 r : ℝ), a1 > 0 ∧ r > 0 ∧ 
  (1 + r + r^2 > r^3) ∧
  (1 + r + r^3 > r^2) ∧
  (1 + r^2 + r^3 > r) ∧
  (r + r^2 + r^3 > 1) := 
sorry

end geometric_progression_quadrilateral_exists_l274_274614


namespace gnollish_valid_sentences_count_is_50_l274_274500

def gnollish_words : List String := ["splargh", "glumph", "amr", "blort"]

def is_valid_sentence (sentence : List String) : Prop :=
  match sentence with
  | [_, "splargh", "glumph"] => False
  | ["splargh", "glumph", _] => False
  | [_, "blort", "amr"] => False
  | ["blort", "amr", _] => False
  | _ => True

def count_valid_sentences (n : Nat) : Nat :=
  (List.replicate n gnollish_words).mapM id |>.length

theorem gnollish_valid_sentences_count_is_50 : count_valid_sentences 3 = 50 :=
by 
  sorry

end gnollish_valid_sentences_count_is_50_l274_274500


namespace fraction_of_girls_is_one_half_l274_274250

def fraction_of_girls (total_students_jasper : ℕ) (ratio_jasper : ℕ × ℕ) (total_students_brookstone : ℕ) (ratio_brookstone : ℕ × ℕ) : ℚ :=
  let (boys_ratio_jasper, girls_ratio_jasper) := ratio_jasper
  let (boys_ratio_brookstone, girls_ratio_brookstone) := ratio_brookstone
  let girls_jasper := (total_students_jasper * girls_ratio_jasper) / (boys_ratio_jasper + girls_ratio_jasper)
  let girls_brookstone := (total_students_brookstone * girls_ratio_brookstone) / (boys_ratio_brookstone + girls_ratio_brookstone)
  let total_girls := girls_jasper + girls_brookstone
  let total_students := total_students_jasper + total_students_brookstone
  total_girls / total_students

theorem fraction_of_girls_is_one_half :
  fraction_of_girls 360 (7, 5) 240 (3, 5) = 1 / 2 :=
  sorry

end fraction_of_girls_is_one_half_l274_274250


namespace max_profit_l274_274408

noncomputable def annual_profit (x : ℝ) : ℝ :=
  if 0 < x ∧ x ≤ 5 then 
    -0.5 * x^2 + 3.5 * x - 0.5 
  else if x > 5 then 
    17 - 2.5 * x 
  else 
    0

theorem max_profit :
  ∀ x : ℝ, (annual_profit 3.5 = 5.625) :=
by
  -- Proof omitted
  sorry

end max_profit_l274_274408


namespace original_number_l274_274369

theorem original_number (x : ℝ) (h : 20 = 0.4 * (x - 5)) : x = 55 :=
sorry

end original_number_l274_274369


namespace common_root_for_equations_l274_274286

theorem common_root_for_equations : 
  ∃ p x : ℤ, 3 * x^2 - 4 * x + p - 2 = 0 ∧ x^2 - 2 * p * x + 5 = 0 ∧ p = 3 ∧ x = 1 :=
by
  sorry

end common_root_for_equations_l274_274286


namespace slope_of_given_line_l274_274382

theorem slope_of_given_line : ∀ (x y : ℝ), (4 / x + 5 / y = 0) → (y = (-5 / 4) * x) := 
by 
  intros x y h
  sorry

end slope_of_given_line_l274_274382


namespace novel_pages_l274_274908

theorem novel_pages (x : ℕ)
  (h1 : x - ((1 / 6 : ℝ) * x + 10) = (5 / 6 : ℝ) * x - 10)
  (h2 : (5 / 6 : ℝ) * x - 10 - ((1 / 5 : ℝ) * ((5 / 6 : ℝ) * x - 10) + 20) = (2 / 3 : ℝ) * x - 28)
  (h3 : (2 / 3 : ℝ) * x - 28 - ((1 / 4 : ℝ) * ((2 / 3 : ℝ) * x - 28) + 25) = (1 / 2 : ℝ) * x - 46) :
  (1 / 2 : ℝ) * x - 46 = 80 → x = 252 :=
by
  sorry

end novel_pages_l274_274908


namespace multiplier_for_difference_l274_274971

variable (x y k : ℕ)
variable (h1 : x + y = 81)
variable (h2 : x^2 - y^2 = k * (x - y))
variable (h3 : x ≠ y)

theorem multiplier_for_difference : k = 81 := 
by
  sorry

end multiplier_for_difference_l274_274971


namespace planning_committee_selection_l274_274070

theorem planning_committee_selection (x : ℕ) (h1 : (x.choose 3) = 35) : (x.choose 4) = 35 :=
by
  have h_x_eq_7 : x = 7 :=
    by sorry
  rw h_x_eq_7
  norm_num
  sorry

end planning_committee_selection_l274_274070


namespace value_of_p_l274_274999

theorem value_of_p (a : ℕ → ℚ) (m : ℕ) (p : ℚ)
  (h1 : a 1 = 111)
  (h2 : a 2 = 217)
  (h3 : ∀ n : ℕ, 3 ≤ n ∧ n ≤ m → a n = a (n - 2) - (n - p) / a (n - 1))
  (h4 : m = 220) :
  p = 110 / 109 :=
by
  sorry

end value_of_p_l274_274999


namespace absolute_difference_m_n_l274_274625

theorem absolute_difference_m_n (m n : ℝ) (h1 : m * n = 6) (h2 : m + n = 7) : |m - n| = 5 := 
by 
  sorry

end absolute_difference_m_n_l274_274625


namespace isosceles_triangle_vertex_angle_l274_274833

theorem isosceles_triangle_vertex_angle (α β γ : ℝ) (h_triangle : α + β + γ = 180)
  (h_isosceles : α = β ∨ β = α ∨ α = γ ∨ γ = α ∨ β = γ ∨ γ = β)
  (h_ratio : α / γ = 1 / 4 ∨ γ / α = 1 / 4) :
  (γ = 20 ∨ γ = 120) :=
sorry

end isosceles_triangle_vertex_angle_l274_274833


namespace third_place_amount_l274_274893

noncomputable def total_people : ℕ := 13
noncomputable def money_per_person : ℝ := 5
noncomputable def total_money : ℝ := total_people * money_per_person

noncomputable def first_place_percentage : ℝ := 0.65
noncomputable def second_third_place_percentage : ℝ := 0.35
noncomputable def split_factor : ℝ := 0.5

noncomputable def first_place_money : ℝ := first_place_percentage * total_money
noncomputable def second_third_place_money : ℝ := second_third_place_percentage * total_money
noncomputable def third_place_money : ℝ := split_factor * second_third_place_money

theorem third_place_amount : third_place_money = 11.38 := by
  sorry

end third_place_amount_l274_274893


namespace sum_first_12_terms_geom_seq_l274_274149

def geometric_sequence_periodic (a : ℕ → ℕ) (k : ℕ) : Prop :=
  ∀ n : ℕ, a n * a (n + 1) * a (n + 2) = k

theorem sum_first_12_terms_geom_seq :
  ∃ a : ℕ → ℕ,
    a 1 = 1 ∧
    a 2 = 2 ∧
    a 3 = 4 ∧
    geometric_sequence_periodic a 8 ∧
    (a 1 + a 2 + a 3 + a 4 + a 5 + a 6 + a 7 + a 8 + a 9 + a 10 + a 11 + a 12) = 28 :=
by
  sorry

end sum_first_12_terms_geom_seq_l274_274149


namespace number_of_skirts_l274_274567

theorem number_of_skirts (T Ca Cs S : ℕ) (hT : T = 50) (hCa : Ca = 20) (hCs : Cs = 15) (hS : T - Ca = S * Cs) : S = 2 := by
  sorry

end number_of_skirts_l274_274567


namespace third_smallest_four_digit_in_pascals_triangle_l274_274199

theorem third_smallest_four_digit_in_pascals_triangle :
  ∃ n m k, binom n m = 1002 ∧ binom n (m + 1) = 1001 ∧ binom n (m + 2) = 1000 :=
sorry

end third_smallest_four_digit_in_pascals_triangle_l274_274199


namespace multiply_polynomials_l274_274905

theorem multiply_polynomials (x : ℝ) : (x^4 + 50 * x^2 + 625) * (x^2 - 25) = x^6 - 15625 := by
  sorry

end multiply_polynomials_l274_274905


namespace interest_rate_supposed_to_be_invested_l274_274961

variable (P T : ℕ) (additional_interest interest_rate_15 interest_rate_R : ℚ)

def simple_interest (principal: ℚ) (time: ℚ) (rate: ℚ) : ℚ := (principal * time * rate) / 100

theorem interest_rate_supposed_to_be_invested :
  P = 15000 → T = 2 → additional_interest = 900 → interest_rate_15 = 15 →
  simple_interest P T interest_rate_15 = simple_interest P T interest_rate_R + additional_interest →
  interest_rate_R = 12 := by
  intros hP hT h_add h15 h_interest
  simp [simple_interest] at *
  sorry

end interest_rate_supposed_to_be_invested_l274_274961


namespace total_hours_worked_l274_274239

-- Definition of the given conditions.
def hours_software : ℕ := 24
def hours_help_user : ℕ := 17
def percentage_other_services : ℚ := 0.4

-- Statement to prove.
theorem total_hours_worked : ∃ (T : ℕ), hours_software + hours_help_user + percentage_other_services * T = T ∧ T = 68 :=
by {
  -- The proof will go here.
  sorry
}

end total_hours_worked_l274_274239


namespace solution_set_of_inequality_l274_274857

theorem solution_set_of_inequality : 
  { x : ℝ | (1 : ℝ) * (2 * x + 1) < (x + 1) } = { x | -1 < x ∧ x < 0 } :=
by
  sorry

end solution_set_of_inequality_l274_274857


namespace F_8_not_true_F_6_might_be_true_l274_274398

variable {n : ℕ}

-- Declare the proposition F
variable (F : ℕ → Prop)

-- Placeholder conditions
axiom condition1 : ¬ F 7
axiom condition2 : ∀ k : ℕ, k > 0 → (F k → F (k + 1))

-- Proof statements
theorem F_8_not_true : ¬ F 8 :=
by {
  sorry
}

theorem F_6_might_be_true : ¬ ¬ F 6 :=
by {
  sorry
}

end F_8_not_true_F_6_might_be_true_l274_274398


namespace tennis_balls_in_each_container_l274_274156

theorem tennis_balls_in_each_container :
  let total_balls := 100
  let given_away := total_balls / 2
  let remaining := total_balls - given_away
  let containers := 5
  remaining / containers = 10 := 
by
  sorry

end tennis_balls_in_each_container_l274_274156


namespace initial_candles_count_l274_274970

section

variable (C : ℝ)
variable (h_Alyssa : C / 2 = C / 2)
variable (h_Chelsea : C / 2 - 0.7 * (C / 2) = 6)

theorem initial_candles_count : C = 40 := 
by sorry

end

end initial_candles_count_l274_274970


namespace remainder_19_pow_19_plus_19_mod_20_l274_274952

theorem remainder_19_pow_19_plus_19_mod_20 : (19^19 + 19) % 20 = 18 := 
by
  sorry

end remainder_19_pow_19_plus_19_mod_20_l274_274952


namespace no_nat_pairs_divisibility_l274_274848

theorem no_nat_pairs_divisibility (a b : ℕ) (hab : b^a ∣ a^b - 1) : false :=
sorry

end no_nat_pairs_divisibility_l274_274848


namespace money_total_l274_274461

theorem money_total (s j m : ℝ) (h1 : 3 * s = 80) (h2 : j / 2 = 70) (h3 : 2.5 * m = 100) :
  s + j + m = 206.67 :=
sorry

end money_total_l274_274461


namespace greatest_divisible_by_11_l274_274242

theorem greatest_divisible_by_11 :
  ∃ (A B C : ℕ), A ≠ C ∧ A ≠ B ∧ B ≠ C ∧ 
  (∀ n, n = 10000 * A + 1000 * B + 100 * C + 10 * B + A → n = 96569) ∧
  (10000 * A + 1000 * B + 100 * C + 10 * B + A) % 11 = 0 :=
sorry

end greatest_divisible_by_11_l274_274242


namespace main_theorem_l274_274339

-- Declare nonzero complex numbers
variables {x y z : ℂ} 

-- State the conditions
def conditions (x y z : ℂ) : Prop :=
  x ≠ 0 ∧ y ≠ 0 ∧ z ≠ 0 ∧
  x + y + z = 30 ∧
  (x - y)^2 + (x - z)^2 + (y - z)^2 = 2 * x * y * z

-- Prove the main statement given the conditions
theorem main_theorem (h : conditions x y z) : 
  (x^3 + y^3 + z^3) / (x * y * z) = 33 :=
by
  sorry

end main_theorem_l274_274339


namespace imaginary_part_of_fraction_l274_274456

theorem imaginary_part_of_fraction (i : ℂ) (hi : i * i = -1) : (1 + i) / (1 - i) = 1 :=
by
  -- Skipping the proof
  sorry

end imaginary_part_of_fraction_l274_274456


namespace pascal_third_smallest_four_digit_number_l274_274208

theorem pascal_third_smallest_four_digit_number :
  (∃ n k: ℕ, 1000 ≤ binom n k ∧ binom n k ≤ 1002) ∧ ¬(∃ n k: ℕ, 1003 ≤ binom n k) :=
sorry

end pascal_third_smallest_four_digit_number_l274_274208


namespace right_triangle_eqn_roots_indeterminate_l274_274866

theorem right_triangle_eqn_roots_indeterminate 
  (a b c : ℝ) (h : a^2 + c^2 = b^2) : 
  ¬(∃ Δ, Δ = 4 - 4 * c^2 ∧ (Δ > 0 ∨ Δ = 0 ∨ Δ < 0)) →
  (¬∃ x, a * (x^2 - 1) - 2 * x + b * (x^2 + 1) = 0 ∨
  ∃ x₁ x₂, x₁ ≠ x₂ ∧ a * (x₁^2 - 1) - 2 * x₁ + b * (x₁^2 + 1) = 0 ∧ a * (x₂^2 - 1) - 2 * x₂ + b * (x₂^2 + 1) = 0) :=
by
  sorry

end right_triangle_eqn_roots_indeterminate_l274_274866


namespace combined_water_leak_l274_274808

theorem combined_water_leak
  (largest_rate : ℕ)
  (medium_rate : ℕ)
  (smallest_rate : ℕ)
  (time_minutes : ℕ)
  (h1 : largest_rate = 3)
  (h2 : medium_rate = largest_rate / 2)
  (h3 : smallest_rate = medium_rate / 3)
  (h4 : time_minutes = 120) :
  largest_rate * time_minutes + medium_rate * time_minutes + smallest_rate * time_minutes = 600 := by
  sorry

end combined_water_leak_l274_274808


namespace distance_from_origin_is_correct_l274_274825

noncomputable def is_distance_8_from_x_axis (x y : ℝ) := y = 8
noncomputable def is_distance_12_from_point (x y : ℝ) := (x - 1)^2 + (y - 6)^2 = 144
noncomputable def x_greater_than_1 (x : ℝ) := x > 1
noncomputable def distance_from_origin (x y : ℝ) := Real.sqrt (x^2 + y^2)

theorem distance_from_origin_is_correct (x y : ℝ)
  (h1 : is_distance_8_from_x_axis x y)
  (h2 : is_distance_12_from_point x y)
  (h3 : x_greater_than_1 x) :
  distance_from_origin x y = Real.sqrt (205 + 2 * Real.sqrt 140) :=
by
  sorry

end distance_from_origin_is_correct_l274_274825


namespace alec_correct_problems_l274_274331

-- Definitions of conditions and proof problem
theorem alec_correct_problems (c w : ℕ) (s : ℕ) (H1 : s = 30 + 4 * c - w) (H2 : s > 90)
  (H3 : ∀ s', 90 < s' ∧ s' < s → ¬(∃ c', ∃ w', s' = 30 + 4 * c' - w')) :
  c = 16 :=
by
  sorry

end alec_correct_problems_l274_274331


namespace monday_rainfall_l274_274890

theorem monday_rainfall (tuesday_rainfall monday_rainfall: ℝ) 
(less_rain: ℝ) (h1: tuesday_rainfall = 0.2) 
(h2: less_rain = 0.7) 
(h3: tuesday_rainfall = monday_rainfall - less_rain): 
monday_rainfall = 0.9 :=
by sorry

end monday_rainfall_l274_274890


namespace solve_for_x2_minus_y2_minus_z2_l274_274740

theorem solve_for_x2_minus_y2_minus_z2
  (x y z : ℝ)
  (h1 : x + y + z = 12)
  (h2 : x - y = 4)
  (h3 : y + z = 7) :
  x^2 - y^2 - z^2 = -12 :=
by
  sorry

end solve_for_x2_minus_y2_minus_z2_l274_274740


namespace swimming_problem_l274_274823

/-- The swimming problem where a man swims downstream 30 km and upstream a certain distance 
    taking 6 hours each time. Given his speed in still water is 4 km/h, we aim to prove the 
    distance swam upstream is 18 km. -/
theorem swimming_problem 
  (V_m : ℝ) (Distance_downstream : ℝ) (Time_downstream : ℝ) (Time_upstream : ℝ) 
  (Distance_upstream : ℝ) (V_s : ℝ)
  (h1 : V_m = 4)
  (h2 : Distance_downstream = 30)
  (h3 : Time_downstream = 6)
  (h4 : Time_upstream = 6)
  (h5 : V_m + V_s = Distance_downstream / Time_downstream)
  (h6 : V_m - V_s = Distance_upstream / Time_upstream) :
  Distance_upstream = 18 := 
sorry

end swimming_problem_l274_274823


namespace proposition_false_at_4_l274_274964

theorem proposition_false_at_4 (P : ℕ → Prop) (hp : ∀ k : ℕ, k > 0 → (P k → P (k + 1))) (h4 : ¬ P 5) : ¬ P 4 :=
by {
    sorry
}

end proposition_false_at_4_l274_274964


namespace find_m_l274_274933

theorem find_m (m : ℝ) (f : ℝ → ℝ) 
  (h1 : ∀ x, f x = (m^2 - 5*m + 7)*x^(m-2)) 
  (h2 : ∀ x, f (-x) = - f x) : 
  m = 3 :=
by
  sorry

end find_m_l274_274933


namespace min_abs_val_sum_l274_274931

theorem min_abs_val_sum : ∃ x : ℝ, (∀ y : ℝ, |y - 1| + |y - 2| + |y - 3| ≥ |x - 1| + |x - 2| + |x - 3|) ∧ |x - 1| + |x - 2| + |x - 3| = 1 :=
sorry

end min_abs_val_sum_l274_274931


namespace multiply_polynomials_l274_274904

theorem multiply_polynomials (x : ℝ) : (x^4 + 50 * x^2 + 625) * (x^2 - 25) = x^6 - 15625 := by
  sorry

end multiply_polynomials_l274_274904


namespace third_smallest_four_digit_Pascal_triangle_l274_274220

theorem third_smallest_four_digit_Pascal_triangle : 
  ∃ n : ℕ, (binomial (n + 2) (n + 2) = 1002) ∧ ∀ k : ℕ, (k < (n + 2) → binomial (k + 2) (k + 2) < 1002) :=
by
  sorry

end third_smallest_four_digit_Pascal_triangle_l274_274220


namespace sin_exp_eq_200_l274_274855

theorem sin_exp_eq_200 (x : ℝ) :
  (∃ x ∈ (0, 200 * real.pi), sin x = (1 / 3) ^ x) ↔ 200 := sorry

end sin_exp_eq_200_l274_274855


namespace least_positive_integer_solution_l274_274035

theorem least_positive_integer_solution :
  ∃ b : ℕ, b ≡ 1 [MOD 4] ∧ b ≡ 2 [MOD 5] ∧ b ≡ 3 [MOD 6] ∧ b = 37 :=
by
  sorry

end least_positive_integer_solution_l274_274035


namespace find_certain_number_l274_274542

theorem find_certain_number (x : ℝ) 
    (h : 7 * x - 6 - 12 = 4 * x) : x = 6 := 
by
  sorry

end find_certain_number_l274_274542


namespace percentage_emails_moved_to_work_folder_l274_274918

def initialEmails : ℕ := 400
def trashedEmails : ℕ := initialEmails / 2
def remainingEmailsAfterTrash : ℕ := initialEmails - trashedEmails
def emailsLeftInInbox : ℕ := 120
def emailsMovedToWorkFolder : ℕ := remainingEmailsAfterTrash - emailsLeftInInbox

theorem percentage_emails_moved_to_work_folder :
  (emailsMovedToWorkFolder * 100 / remainingEmailsAfterTrash) = 40 := by
  sorry

end percentage_emails_moved_to_work_folder_l274_274918


namespace num_integers_units_digit_condition_l274_274595

theorem num_integers_units_digit_condition :
  ∃ (count : ℕ), count = 81 ∧
  ∀ n : ℕ, (1000 < n ∧ n < 2050) →
  (n % 10 = (n / 10 % 10) + (n / 100 % 10) + (n / 1000 % 10)) →
  count = (∑ n in finset.range 1050 \ finset.range 1001, if (n % 10 = (n / 10 % 10) + (n / 100 % 10) + (n / 1000 % 10)) then 1 else 0)
:= by
  sorry

end num_integers_units_digit_condition_l274_274595


namespace percentage_increase_l274_274317

noncomputable def percentMoreThan (a b : ℕ) : ℕ :=
  ((a - b) * 100) / b

theorem percentage_increase (x y z : ℕ) (h1 : z = 300) (h2 : x = 5 * y / 4) (h3 : x + y + z = 1110) :
  percentMoreThan y z = 20 := by
  sorry

end percentage_increase_l274_274317


namespace find_y_when_x_is_6_l274_274936

variable (x y : ℝ)
variable (h₁ : x > 0)
variable (h₂ : y > 0)
variable (k : ℝ)

axiom inverse_proportional : 3 * x^2 * y = k
axiom initial_condition : 3 * 3^2 * 30 = k

theorem find_y_when_x_is_6 (h : x = 6) : y = 7.5 :=
by
  sorry

end find_y_when_x_is_6_l274_274936


namespace find_f_neg_eight_l274_274859

-- Conditions based on the given problem
variable (f : ℤ → ℤ)
axiom func_property : ∀ x y : ℤ, f (x + y) = f x + f y + x * y + 1
axiom f1_is_one : f 1 = 1

-- Main theorem
theorem find_f_neg_eight : f (-8) = 19 := by
  sorry

end find_f_neg_eight_l274_274859


namespace trapezium_area_l274_274098

-- Define the lengths of the parallel sides and the distance between them
def side_a : ℝ := 20
def side_b : ℝ := 18
def height : ℝ := 15

-- Define the formula for the area of a trapezium
def area_trapezium (a b h : ℝ) : ℝ :=
  (1 / 2) * (a + b) * h

-- State the theorem
theorem trapezium_area :
  area_trapezium side_a side_b height = 285 :=
by
  sorry

end trapezium_area_l274_274098


namespace exponentiation_rule_proof_l274_274079

-- Definitions based on conditions
def x : ℕ := 3
def a : ℕ := 4
def b : ℕ := 2

-- The rule that relates the exponents
def rule (x a b : ℕ) : ℕ := x^(a * b)

-- Proposition that we need to prove
theorem exponentiation_rule_proof : rule x a b = 6561 :=
by
  -- sorry is used to indicate the proof is omitted
  sorry

end exponentiation_rule_proof_l274_274079


namespace problem_theorem_l274_274533

theorem problem_theorem (x y z : ℤ) 
  (h1 : x = 10 * y + 3)
  (h2 : 2 * x = 21 * y + 1)
  (h3 : 3 * x = 5 * z + 2) : 
  11 * y - x + 7 * z = 219 := 
by
  sorry

end problem_theorem_l274_274533


namespace geo_seq_product_l274_274747

noncomputable def geometric_sequence (a : ℕ → ℝ) : Prop :=
∀ n m : ℕ, a (n + m) = a n * a m / a 1

theorem geo_seq_product
  {a : ℕ → ℝ}
  (h_pos : ∀ n, a n > 0)
  (h_seq : geometric_sequence a)
  (h_roots : ∃ x y, (x*x - 10 * x + 16 = 0) ∧ (y*y - 10 * y + 16 = 0) ∧ a 1 = x ∧ a 19 = y) :
  a 8 * a 10 * a 12 = 64 := 
sorry

end geo_seq_product_l274_274747


namespace tetrahedron_pythagorean_theorem_l274_274474

noncomputable section

variables {a b c : ℝ} {S_ABC S_VAB S_VBC S_VAC : ℝ}

-- Conditions
def is_right_triangle (a b c : ℝ) := c^2 = a^2 + b^2
def is_right_tetrahedron (S_ABC S_VAB S_VBC S_VAC : ℝ) := 
  S_ABC^2 = S_VAB^2 + S_VBC^2 + S_VAC^2

-- Theorem Statement
theorem tetrahedron_pythagorean_theorem (a b c S_ABC S_VAB S_VBC S_VAC : ℝ) 
  (h1 : is_right_triangle a b c)
  (h2 : S_ABC^2 = S_VAB^2 + S_VBC^2 + S_VAC^2) :
  S_ABC^2 = S_VAB^2 + S_VBC^2 + S_VAC^2 := 
by sorry

end tetrahedron_pythagorean_theorem_l274_274474


namespace conditional_independence_law_equiv_l274_274163

noncomputable theory
open ProbabilityTheory

variables {α : Type*} {β : Type*} [MeasurableSpace α] [MeasurableSpace β]
variables (X : ℕ → α) (Y : ℕ → β) (n : ℕ)

def independence_pairwise (X : ℕ → α) (Y : ℕ → β) :=
  ∀ i j, i ≠ j → indep (λ ω, (X i ω, Y i ω)) (λ ω, (X j ω, Y j ω))

def sigma_algebra_Y (Y : ℕ → β) (n : ℕ) : MeasurableSpace α :=
  measurable_space.comap (λ x, \(n : ℕ), (Y n x)) measurable_space.{β}

theorem conditional_independence (h_indep : independence_pairwise X Y) :
  ∀ i, i ∈ finset.range n → indep_cond (λ ω, X ω) (λ ω, Y ω) :=
begin
  sorry
end

theorem law_equiv (h_indep : independence_pairwise X Y) :
  ∀ i, i ∈ finset.range n → 
  (∀ (B : set α), measurable_set B → 
  P[X i ∈ B | sigma_algebra_Y Y n] = P[X i ∈ B | Y i]) :=
begin
  sorry
end

end conditional_independence_law_equiv_l274_274163


namespace division_problem_l274_274323

theorem division_problem (D : ℕ) (Quotient Dividend Remainder : ℕ) 
    (h1 : Quotient = 36) 
    (h2 : Dividend = 3086) 
    (h3 : Remainder = 26) 
    (h_div : Dividend = (D * Quotient) + Remainder) : 
    D = 85 := 
by 
  -- Steps to prove the theorem will go here
  sorry

end division_problem_l274_274323


namespace max_pqrs_squared_l274_274161

theorem max_pqrs_squared (p q r s : ℝ)
  (h1 : p + q = 18)
  (h2 : pq + r + s = 85)
  (h3 : pr + qs = 190)
  (h4 : rs = 120) :
  p^2 + q^2 + r^2 + s^2 ≤ 886 :=
sorry

end max_pqrs_squared_l274_274161


namespace multiples_of_4_count_l274_274132

theorem multiples_of_4_count (a b : ℕ) (h₁ : a = 100) (h₂ : b = 400) :
  ∃ n : ℕ, n = 75 ∧ ∀ k : ℕ, (k >= a ∧ k <= b ∧ k % 4 = 0) ↔ (k / 4 - 25 ≥ 1 ∧ k / 4 - 25 ≤ n) :=
sorry

end multiples_of_4_count_l274_274132


namespace ac_plus_bd_eq_neg_10_l274_274877

theorem ac_plus_bd_eq_neg_10 (a b c d : ℝ)
  (h1 : a + b + c = 1)
  (h2 : a + b + d = 3)
  (h3 : a + c + d = 8)
  (h4 : b + c + d = 6) :
  a * c + b * d = -10 :=
by
  sorry

end ac_plus_bd_eq_neg_10_l274_274877


namespace handshakes_total_l274_274251

def num_couples : ℕ := 15
def total_people : ℕ := 30
def men : ℕ := 15
def women : ℕ := 15
def youngest_man_handshakes : ℕ := 0
def men_handshakes : ℕ := (14 * 13) / 2
def men_women_handshakes : ℕ := 15 * 14

theorem handshakes_total : men_handshakes + men_women_handshakes = 301 :=
by
  -- Proof goes here
  sorry

end handshakes_total_l274_274251


namespace min_distinct_sums_l274_274235

theorem min_distinct_sums (n : ℕ) (hn : n ≥ 5) (s : Finset ℕ) 
  (hs : s.card = n) : 
  ∃ (t : Finset ℕ), (∀ (x y : ℕ), x ∈ s → y ∈ s → x < y → (x + y) ∈ t) ∧ t.card = 2 * n - 3 :=
by
  sorry

end min_distinct_sums_l274_274235


namespace range_of_k_for_real_roots_l274_274709

theorem range_of_k_for_real_roots (k : ℝ) : 
  (∃ x, 2 * x^2 - 3 * x = k) ↔ k ≥ -9/8 :=
by
  sorry

end range_of_k_for_real_roots_l274_274709


namespace symmetric_line_equation_y_axis_l274_274926

theorem symmetric_line_equation_y_axis (x y : ℝ) : 
  (∃ m n : ℝ, (y = 3 * x + 1) ∧ (x + m = 0) ∧ (y = n) ∧ (n = 3 * m + 1)) → 
  y = -3 * x + 1 :=
by
  sorry

end symmetric_line_equation_y_axis_l274_274926


namespace liam_drinks_17_glasses_l274_274490

def minutes_in_hours (h : ℕ) : ℕ := h * 60

def total_time_in_minutes (hours : ℕ) (extra_minutes : ℕ) : ℕ := 
  minutes_in_hours hours + extra_minutes

def rate_of_drinking (drink_interval : ℕ) (total_minutes : ℕ) : ℕ :=
  total_minutes / drink_interval

theorem liam_drinks_17_glasses : 
  rate_of_drinking 20 (total_time_in_minutes 5 40) = 17 :=
by
  sorry

end liam_drinks_17_glasses_l274_274490


namespace least_positive_integer_congruences_l274_274051

theorem least_positive_integer_congruences :
  ∃ n : ℕ, 
    n > 0 ∧ 
    (n % 4 = 1) ∧ 
    (n % 5 = 2) ∧ 
    (n % 6 = 3) ∧ 
    (n = 57) :=
by
  sorry

end least_positive_integer_congruences_l274_274051


namespace sequence_10th_term_l274_274873

theorem sequence_10th_term (a : ℕ → ℝ) 
  (h_initial : a 1 = 1) 
  (h_recursive : ∀ n, a (n + 1) = 2 * a n / (a n + 2)) : 
  a 10 = 2 / 11 :=
sorry

end sequence_10th_term_l274_274873


namespace inscribed_circle_radius_DEF_l274_274980

noncomputable def radius_inscribed_circle (DE DF EF : ℕ) : ℝ :=
  let s := (DE + DF + EF) / 2
  let K := Real.sqrt (s * (s - DE) * (s - DF) * (s - EF))
  K / s

theorem inscribed_circle_radius_DEF :
  radius_inscribed_circle 26 16 20 = 5 * Real.sqrt 511.5 / 31 :=
by
  sorry

end inscribed_circle_radius_DEF_l274_274980


namespace celestia_badges_l274_274594

theorem celestia_badges (H L C : ℕ) (total_badges : ℕ) (h1 : H = 14) (h2 : L = 17) (h3 : total_badges = 83) (h4 : H + L + C = total_badges) : C = 52 :=
by
  sorry

end celestia_badges_l274_274594


namespace third_smallest_four_digit_in_pascals_triangle_l274_274214

theorem third_smallest_four_digit_in_pascals_triangle :
  ∃ n k, n > 0 ∧ k > 0 ∧ Nat.choose n k = 1035 ∧ 
  (∀ (m l : ℕ), m > 0 ∧ l > 0 ∧ Nat.choose m l < 1000 → Nat.choose m l < 1035) ∧
  (∀ (p q : ℕ), p > 0 ∧ q > 0 ∧ 1000 ≤ Nat.choose p q ∧ Nat.choose p q < 1035 → 
    (Nat.choose m l = 1000 ∨ Nat.choose m l = 1001 ∨ Nat.choose m l = 1035)) :=
sorry

end third_smallest_four_digit_in_pascals_triangle_l274_274214


namespace infinite_series_equivalence_l274_274484

theorem infinite_series_equivalence (x y : ℝ) (hy : y ≠ 0 ∧ y ≠ 1) 
  (series_cond : ∑' n : ℕ, x / (y^(n+1)) = 3) :
  ∑' n : ℕ, x / ((x + 2*y)^(n+1)) = 3 * (y - 1) / (5*y - 4) := 
by
  sorry

end infinite_series_equivalence_l274_274484


namespace probability_two_shirts_one_pants_one_socks_l274_274886

def total_items := 5 + 6 + 7

def ways_pick_4 := Nat.choose total_items 4

def ways_pick_2_shirts := Nat.choose 5 2

def ways_pick_1_pants := Nat.choose 6 1

def ways_pick_1_socks := Nat.choose 7 1

def favorable_outcomes := ways_pick_2_shirts * ways_pick_1_pants * ways_pick_1_socks

def probability := (favorable_outcomes : ℚ) / (ways_pick_4 : ℚ)

theorem probability_two_shirts_one_pants_one_socks :
  probability = 7 / 51 :=
by
  sorry

end probability_two_shirts_one_pants_one_socks_l274_274886


namespace arcsin_sin_solution_l274_274176

theorem arcsin_sin_solution (x : ℝ) (h : - (3 * Real.pi / 2) ≤ x ∧ x ≤ 3 * Real.pi / 2) :
  arcsin (sin x) = x / 3 ↔ x = -3 * Real.pi ∨ x = - Real.pi ∨ x = 0 ∨ x = Real.pi ∨ x = 3 * Real.pi :=
sorry

end arcsin_sin_solution_l274_274176


namespace problem_l274_274144

noncomputable def x : ℝ := 123.75
noncomputable def y : ℝ := 137.5
noncomputable def original_value : ℝ := 125

theorem problem (y_more : y = original_value + 0.1 * original_value) (x_less : x = y * 0.9) : y = 137.5 :=
by
  sorry

end problem_l274_274144


namespace men_to_complete_work_l274_274958

theorem men_to_complete_work (x : ℕ) (h1 : 10 * 80 = x * 40) : x = 20 :=
by
  sorry

end men_to_complete_work_l274_274958


namespace abs_difference_of_mn_6_and_sum_7_l274_274627

theorem abs_difference_of_mn_6_and_sum_7 (m n : ℝ) (h₁ : m * n = 6) (h₂ : m + n = 7) : |m - n| = 5 := 
sorry

end abs_difference_of_mn_6_and_sum_7_l274_274627


namespace range_of_a_l274_274129

open Set Real

noncomputable def A (a : ℝ) : Set ℝ := {x | x * (x - a) < 0}
def B : Set ℝ := {x | x^2 - 7 * x - 18 < 0}

theorem range_of_a (a : ℝ) : A a ⊆ B → (-2 : ℝ) ≤ a ∧ a ≤ 9 :=
by sorry

end range_of_a_l274_274129


namespace fencing_required_l274_274965

-- Conditions
def L : ℕ := 20
def A : ℕ := 680

-- Statement to prove
theorem fencing_required : ∃ W : ℕ, A = L * W ∧ 2 * W + L = 88 :=
by
  -- Here you would normally need the logical steps to arrive at the proof
  sorry

end fencing_required_l274_274965


namespace bike_ride_ratio_l274_274076

theorem bike_ride_ratio (J : ℕ) (B : ℕ) (M : ℕ) (hB : B = 17) (hM : M = J + 10) (hTotal : B + J + M = 95) :
  J / B = 2 :=
by
  sorry

end bike_ride_ratio_l274_274076


namespace largest_5_digit_congruent_15_mod_24_l274_274377

theorem largest_5_digit_congruent_15_mod_24 : ∃ x, 10000 ≤ x ∧ x < 100000 ∧ x % 24 = 15 ∧ x = 99999 := by
  sorry

end largest_5_digit_congruent_15_mod_24_l274_274377


namespace least_positive_integer_congruences_l274_274053

theorem least_positive_integer_congruences :
  ∃ n : ℕ, 
    n > 0 ∧ 
    (n % 4 = 1) ∧ 
    (n % 5 = 2) ∧ 
    (n % 6 = 3) ∧ 
    (n = 57) :=
by
  sorry

end least_positive_integer_congruences_l274_274053


namespace tanC_div_tanA_plus_tanC_div_tanB_eq_four_l274_274473

theorem tanC_div_tanA_plus_tanC_div_tanB_eq_four
  (a b c A B C : ℝ)
  (h1 : 0 < A) (h2 : 0 < B) (h3 : 0 < C)
  (h4 : a = b * (Real.cos C))
  (h5 : a * a + b * b = 3 * c * c / 2)
  (h6 : Real.tan A = (Real.sin A) / (Real.cos A))
  (h7 : Real.tan B = (Real.sin B) / (Real.cos B))
  (h8 : Real.tan C = (Real.sin C) / (Real.cos C)) :
  (Real.tan C / Real.tan A) + (Real.tan C / Real.tan B) = 4 :=
  sorry

end tanC_div_tanA_plus_tanC_div_tanB_eq_four_l274_274473


namespace calculate_inverse_y3_minus_y_l274_274599

theorem calculate_inverse_y3_minus_y
  (i : ℂ) (y : ℂ)
  (h_i : i = Complex.I)
  (h_y : y = (1 + i * Real.sqrt 3) / 2) :
  (1 / (y^3 - y)) = -1/2 + i * (Real.sqrt 3) / 6 :=
by
  sorry

end calculate_inverse_y3_minus_y_l274_274599


namespace lcm_180_616_l274_274028

theorem lcm_180_616 : Nat.lcm 180 616 = 27720 := 
by
  sorry

end lcm_180_616_l274_274028


namespace third_smallest_four_digit_in_pascals_triangle_l274_274194

-- First we declare the conditions
def every_positive_integer_in_pascals_triangle : Prop :=
  ∀ n : ℕ, ∃ a b : ℕ, nat.choose b a = n

def number_1000_in_pascals_triangle : Prop :=
  ∃ b : ℕ, nat.choose b 1 = 1000

def number_1001_in_pascals_triangle : Prop :=
  ∃ b : ℕ, nat.choose b 1 = 1001

def number_1002_in_pascals_triangle : Prop :=
  ∃ b : ℕ, nat.choose b 2 = 1002

-- Define the theorem
theorem third_smallest_four_digit_in_pascals_triangle
  (h1 : every_positive_integer_in_pascals_triangle)
  (h2 : number_1000_in_pascals_triangle)
  (h3 : number_1001_in_pascals_triangle)
  (h4 : number_1002_in_pascals_triangle) :
  (∃ n : ℕ, nat.choose n 2 = 1002 ∧ (∀ k, k < n → nat.choose k 1 < 1002)) := sorry

end third_smallest_four_digit_in_pascals_triangle_l274_274194


namespace third_smallest_four_digit_in_pascals_triangle_is_1002_l274_274207

theorem third_smallest_four_digit_in_pascals_triangle_is_1002 :
  ∃ n k, 1000 <= binomial n k ∧ binomial n k <= 9999 ∧ (third_smallest (λ n k, 1000 <= binomial n k ∧ binomial n k <= 9999) = 1002) :=
sorry

end third_smallest_four_digit_in_pascals_triangle_is_1002_l274_274207


namespace sereja_picked_more_berries_l274_274696

theorem sereja_picked_more_berries (total_berries : ℕ) (sereja_rate : ℕ) (dima_rate : ℕ)
  (sereja_pattern_input : ℕ) (sereja_pattern_eat : ℕ)
  (dima_pattern_input : ℕ) (dima_pattern_eat : ℕ)
  (rate_relation : sereja_rate = 2 * dima_rate)
  (total_berries_collected : sereja_rate + dima_rate = total_berries) :
  let sereja_collected := total_berries_collected * (sereja_pattern_input / (sereja_pattern_input + sereja_pattern_eat)),
      dima_collected := total_berries_collected * (dima_pattern_input / (dima_pattern_input + dima_pattern_eat)) in
  sereja_collected > dima_collected ∧ sereja_collected - dima_collected = 50 :=
by {
  sorry
}

end sereja_picked_more_berries_l274_274696


namespace stripe_area_l274_274406

theorem stripe_area :
  ∀ (diameter height width revolutions : ℝ), 
    diameter = 40 → 
    height = 100 → 
    width = 4 → 
    revolutions = 3 → 
    let circumference := Real.pi * diameter in
    let total_length := circumference * revolutions in
    let area := width * total_length in
    area = 480 * Real.pi :=
by
  intros diameter height width revolutions h1 h2 h3 h4 circumference total_length area
  sorry

end stripe_area_l274_274406


namespace mario_savings_percentage_l274_274324

-- Define the price of one ticket
def ticket_price : ℝ := sorry

-- Define the conditions
-- Condition 1: 5 tickets can be purchased for the usual price of 3 tickets
def price_for_5_tickets := 3 * ticket_price

-- Condition 2: Mario bought 5 tickets
def mario_tickets := 5 * ticket_price

-- Condition 3: Usual price for 5 tickets
def usual_price_5_tickets := 5 * ticket_price

-- Calculate the amount saved
def amount_saved := usual_price_5_tickets - price_for_5_tickets

theorem mario_savings_percentage
  (ticket_price: ℝ)
  (h1 : price_for_5_tickets = 3 * ticket_price)
  (h2 : mario_tickets = 5 * ticket_price)
  (h3 : usual_price_5_tickets = 5 * ticket_price)
  (h4 : amount_saved = usual_price_5_tickets - price_for_5_tickets):
  (amount_saved / usual_price_5_tickets) * 100 = 40 := 
by {
    -- Placeholder
    sorry
}

end mario_savings_percentage_l274_274324


namespace least_positive_integer_condition_l274_274033

theorem least_positive_integer_condition
  (a : ℤ) (ha1 : a % 4 = 1) (ha2 : a % 5 = 2) (ha3 : a % 6 = 3) :
  a > 0 → a = 57 :=
by
  intro ha_pos
  -- Proof omitted for brevity
  sorry

end least_positive_integer_condition_l274_274033


namespace DVDs_sold_is_168_l274_274606

variables (C D : ℕ)
variables (h1 : D = (16 * C) / 10)
variables (h2 : D + C = 273)

theorem DVDs_sold_is_168 : D = 168 := by
  sorry

end DVDs_sold_is_168_l274_274606


namespace patricia_books_read_l274_274264

noncomputable def calculate_books := 
  λ (Candice_read : ℕ) =>
    let Amanda_read := Candice_read / 3
    let Kara_read := Amanda_read / 2
    let Patricia_read := 7 * Kara_read
    Patricia_read

theorem patricia_books_read (Candice_books : ℕ) (hC : Candice_books = 18) :
  calculate_books Candice_books = 21 := by
  rw [hC]
  unfold calculate_books
  simp
  sorry

end patricia_books_read_l274_274264


namespace stamps_in_last_page_l274_274767

-- Define the total number of books, pages per book, and stamps per original page.
def total_books : ℕ := 6
def pages_per_book : ℕ := 30
def original_stamps_per_page : ℕ := 7

-- Define the new stamps per page after reorganization.
def new_stamps_per_page : ℕ := 9

-- Define the number of fully filled books and pages in the fourth book.
def filled_books : ℕ := 3
def pages_in_fourth_book : ℕ := 26

-- Define the total number of stamps originally.
def total_original_stamps : ℕ := total_books * pages_per_book * original_stamps_per_page

-- Prove that the last page in the fourth book contains 9 stamps under the given conditions.
theorem stamps_in_last_page : 
  total_original_stamps / new_stamps_per_page - (filled_books * pages_per_book + pages_in_fourth_book) * new_stamps_per_page = 9 :=
by
  sorry

end stamps_in_last_page_l274_274767


namespace john_time_spent_on_pictures_l274_274755

noncomputable def time_spent (num_pictures : ℕ) (draw_time color_time : ℝ) : ℝ := 
  (num_pictures * draw_time) + (num_pictures * color_time)

theorem john_time_spent_on_pictures :
  ∀ (num_pictures : ℕ) (draw_time : ℝ) (color_percentage_less : ℝ)
    (color_time : ℝ),
    num_pictures = 10 →
    draw_time = 2 →
    color_percentage_less = 0.30 →
    color_time = draw_time - (color_percentage_less * draw_time) →
    time_spent num_pictures draw_time color_time = 34 :=
begin
  sorry,
end

end john_time_spent_on_pictures_l274_274755


namespace polygon_perimeter_is_35_l274_274966

-- Define the concept of a regular polygon with given side length and exterior angle
def regular_polygon_perimeter (n : ℕ) (side_length : ℕ) : ℕ := 
  n * side_length

theorem polygon_perimeter_is_35 (side_length : ℕ) (exterior_angle : ℕ) (n : ℕ)
  (h1 : side_length = 7) (h2 : exterior_angle = 72) (h3 : 360 / exterior_angle = n) :
  regular_polygon_perimeter n side_length = 35 :=
by
  -- We skip the proof body as only the statement is required
  sorry

end polygon_perimeter_is_35_l274_274966


namespace greatest_common_divisor_XYXY_pattern_l274_274376

theorem greatest_common_divisor_XYXY_pattern (X Y : ℕ) (hX : X ≥ 0 ∧ X ≤ 9) (hY : Y ≥ 0 ∧ Y ≤ 9) :
  ∃ k, 11 * k = 1001 * X + 10 * Y :=
by
  sorry

end greatest_common_divisor_XYXY_pattern_l274_274376


namespace largest_three_digit_divisible_by_13_l274_274842

theorem largest_three_digit_divisible_by_13 :
  ∃ n, (n ≤ 999 ∧ n ≥ 100 ∧ 13 ∣ n) ∧ (∀ m, m ≤ 999 ∧ m ≥ 100 ∧ 13 ∣ m → m ≤ 987) :=
by
  sorry

end largest_three_digit_divisible_by_13_l274_274842


namespace june_biking_time_l274_274757

theorem june_biking_time :
  ∀ (d_jj d_jb : ℕ) (t_jj : ℕ), (d_jj = 2) → (t_jj = 8) → (d_jb = 6) →
  (t_jb : ℕ) → t_jb = (d_jb * t_jj) / d_jj → t_jb = 24 :=
by
  intros d_jj d_jb t_jj h_djj h_tjj h_djb t_jb h_eq
  rw [h_djj, h_tjj, h_djb] at h_eq
  simp at h_eq
  exact h_eq

end june_biking_time_l274_274757


namespace expression_value_l274_274258

theorem expression_value : 7^4 + 4 * 7^3 + 6 * 7^2 + 4 * 7 + 1 = 4096 := 
by 
  -- proof goes here 
  sorry

end expression_value_l274_274258


namespace factor_diff_of_squares_l274_274278

theorem factor_diff_of_squares (y : ℝ) : 25 - 16 * y^2 = (5 - 4 * y) * (5 + 4 * y) := 
sorry

end factor_diff_of_squares_l274_274278


namespace ellipse_equation_l274_274554

theorem ellipse_equation 
  (m n : ℝ) (h1 : m > 0) (h2 : n > 0) (h3 : m ≠ n)
  (h4 : ∀ A B : ℝ × ℝ, (m * A.1^2 + n * A.2^2 = 1) ∧ (m * B.1^2 + n * B.2^2 = 1) ∧ (A.1 + A.2 = 1) ∧ (B.1 + B.2 = 1) → dist A B = 2 * (2:ℝ).sqrt)
  (h5 : ∀ A B : ℝ × ℝ, (m * A.1^2 + n * A.2^2 = 1) ∧ (m * B.1^2 + n * B.2^2 = 1) ∧ (A.1 + A.2 = 1) ∧ (B.1 + B.2 = 1) → 
    (A.2 + B.2) / (A.1 + B.1) = (2:ℝ).sqrt / 2) :
  m = 1 / 3 → n = (2:ℝ).sqrt / 3 → 
  (∀ x y : ℝ, (1 / 3) * x^2 + ((2:ℝ).sqrt / 3) * y^2 = 1) :=
by
  sorry

end ellipse_equation_l274_274554


namespace multiply_polynomials_l274_274907

theorem multiply_polynomials (x : ℝ) : (x^4 + 50 * x^2 + 625) * (x^2 - 25) = x^6 - 15625 := by
  sorry

end multiply_polynomials_l274_274907


namespace cats_weigh_more_than_puppies_l274_274728

theorem cats_weigh_more_than_puppies :
  let puppies_weight := 4 * 7.5
  let cats_weight := 14 * 2.5
  cats_weight - puppies_weight = 5 :=
by
  let puppies_weight := 4 * 7.5
  let cats_weight := 14 * 2.5
  show cats_weight - puppies_weight = 5 from sorry

end cats_weigh_more_than_puppies_l274_274728


namespace same_terminal_side_angles_l274_274654

theorem same_terminal_side_angles (α : ℝ) : 
  (∃ k : ℤ, α = -457 + k * 360) ↔ (∃ k : ℤ, α = 263 + k * 360) :=
sorry

end same_terminal_side_angles_l274_274654


namespace slope_of_line_determined_by_solutions_eq_l274_274384

theorem slope_of_line_determined_by_solutions_eq :
  ∀ (x y : ℝ), (4 / x + 5 / y = 0) → ∃ m : ℝ, m = -5 / 4 :=
by
  intro x y h
  use -5 / 4
  sorry

end slope_of_line_determined_by_solutions_eq_l274_274384


namespace inequality_proof_l274_274351

theorem inequality_proof (a b c : ℝ) (h₀ : 0 < a) (h₁ : 0 < b) (h₂ : 0 < c) : 
  a^4 + b^4 + c^4 ≥ a * b * c * (a + b + c) := 
by 
  sorry

end inequality_proof_l274_274351


namespace least_positive_integer_l274_274039

theorem least_positive_integer (n : ℕ) : 
  (n % 4 = 1) ∧ (n % 5 = 2) ∧ (n % 6 = 3) → n = 57 := by
sorry

end least_positive_integer_l274_274039


namespace find_y_l274_274949

theorem find_y (x y : ℤ)
  (h1 : (100 + 200300 + x) / 3 = 250)
  (h2 : (300 + 150100 + x + y) / 4 = 200) :
  y = -4250 :=
sorry

end find_y_l274_274949


namespace int_pairs_satisfy_conditions_l274_274840

theorem int_pairs_satisfy_conditions (m n : ℤ) :
  (∃ a b : ℤ, m^2 + n = a^2 ∧ n^2 + m = b^2) ↔ 
  ∃ k : ℤ, (m = 0 ∧ n = k^2) ∨ (m = k^2 ∧ n = 0) ∨ (m = 1 ∧ n = -1) ∨ (m = -1 ∧ n = 1) := by
  sorry

end int_pairs_satisfy_conditions_l274_274840


namespace savings_calculation_l274_274537

theorem savings_calculation (x : ℕ) (h1 : 15 * x = 15000) : (15000 - 8 * x = 7000) :=
sorry

end savings_calculation_l274_274537


namespace inequality_holds_for_k_2_l274_274703

theorem inequality_holds_for_k_2 (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) : 
  a * b + b * c + c * a + 2 * (1 / a + 1 / b + 1 / c) ≥ 9 := 
by 
  sorry

end inequality_holds_for_k_2_l274_274703


namespace S_17_33_50_sum_l274_274162

def S (n : ℕ) : ℤ :=
  if n % 2 = 0 then
    - (n / 2)
  else
    (n + 1) / 2

theorem S_17_33_50_sum : S 17 + S 33 + S 50 = 1 :=
by
  sorry

end S_17_33_50_sum_l274_274162


namespace third_smallest_four_digit_in_pascal_triangle_l274_274211

-- Defining Pascal's triangle
def pascal (n k : ℕ) : ℕ := if k = 0 ∨ k = n then 1 else pascal (n - 1) (k - 1) + pascal (n - 1) k

/--
Condition:
1. In Pascal's triangle, each number is the sum of the two numbers directly above it.
2. Four-digit numbers start appearing regularly in higher rows (from row 45 onwards).
-/
theorem third_smallest_four_digit_in_pascal_triangle : 
  ∃ n k : ℕ, pascal n k = 1002 ∧ ∀ m l : ℕ, (pascal m l < 1000 ∨ pascal m l = 1001) → (m < n ∨ (m = n ∧ l < k)) :=
sorry

end third_smallest_four_digit_in_pascal_triangle_l274_274211


namespace least_positive_integer_congruences_l274_274049

theorem least_positive_integer_congruences :
  ∃ n : ℕ, 
    n > 0 ∧ 
    (n % 4 = 1) ∧ 
    (n % 5 = 2) ∧ 
    (n % 6 = 3) ∧ 
    (n = 57) :=
by
  sorry

end least_positive_integer_congruences_l274_274049


namespace volume_of_inscribed_sphere_l274_274739

noncomputable def volume_of_tetrahedron (R : ℝ) (S1 S2 S3 S4 : ℝ) : ℝ :=
  R * (S1 + S2 + S3 + S4)

theorem volume_of_inscribed_sphere (R S1 S2 S3 S4 V : ℝ) :
  V = R * (S1 + S2 + S3 + S4) :=
sorry

end volume_of_inscribed_sphere_l274_274739


namespace find_positive_solutions_l274_274988

theorem find_positive_solutions (x₁ x₂ x₃ x₄ x₅ : ℝ) (h_pos : 0 < x₁ ∧ 0 < x₂ ∧ 0 < x₃ ∧ 0 < x₄ ∧ 0 < x₅)
    (h1 : x₁ + x₂ = x₃^2)
    (h2 : x₂ + x₃ = x₄^2)
    (h3 : x₃ + x₄ = x₅^2)
    (h4 : x₄ + x₅ = x₁^2)
    (h5 : x₅ + x₁ = x₂^2) :
    x₁ = 2 ∧ x₂ = 2 ∧ x₃ = 2 ∧ x₄ = 2 ∧ x₅ = 2 := 
    by {
        -- Proof goes here
        sorry
    }

end find_positive_solutions_l274_274988


namespace least_positive_integer_condition_l274_274031

theorem least_positive_integer_condition
  (a : ℤ) (ha1 : a % 4 = 1) (ha2 : a % 5 = 2) (ha3 : a % 6 = 3) :
  a > 0 → a = 57 :=
by
  intro ha_pos
  -- Proof omitted for brevity
  sorry

end least_positive_integer_condition_l274_274031


namespace xyz_problem_l274_274109

theorem xyz_problem (x y : ℝ) (h1 : x + y - x * y = 155) (h2 : x^2 + y^2 = 325) : |x^3 - y^3| = 4375 := by
  sorry

end xyz_problem_l274_274109


namespace quadratic_points_range_l274_274451

theorem quadratic_points_range (a : ℝ) (y1 y2 y3 y4 : ℝ) :
  (∀ (x : ℝ), 
    (x = -4 → y1 = a * x^2 + 4 * a * x - 6) ∧ 
    (x = -3 → y2 = a * x^2 + 4 * a * x - 6) ∧ 
    (x = 0 → y3 = a * x^2 + 4 * a * x - 6) ∧ 
    (x = 2 → y4 = a * x^2 + 4 * a * x - 6)) →
  (∃! (y : ℝ), y > 0 ∧ (y = y1 ∨ y = y2 ∨ y = y3 ∨ y = y4)) →
  (a < -2 ∨ a > 1 / 2) :=
by
  sorry

end quadratic_points_range_l274_274451


namespace smallest_positive_angle_l274_274298

theorem smallest_positive_angle (α : ℝ) (h : (Real.sin (2 * Real.pi / 3), Real.cos (2 * Real.pi / 3)) = (Real.sin α, Real.cos α)) : 
  α = 11 * Real.pi / 6 := by
sorry

end smallest_positive_angle_l274_274298


namespace integer_solutions_of_equation_l274_274308

theorem integer_solutions_of_equation:
  (number_of_int_solutions (λ x y, 6 * y^2 + 3 * x * y + x + 2 * y + 180) = 6) :=
begin
  sorry
end

end integer_solutions_of_equation_l274_274308


namespace simplify_evaluate_expr_l274_274004

theorem simplify_evaluate_expr (x y : ℚ) (h₁ : x = -1) (h₂ : y = -1 / 2) :
  (4 * x * y + (2 * x^2 + 5 * x * y - y^2) - 2 * (x^2 + 3 * x * y)) = 5 / 4 :=
by
  rw [h₁, h₂]
  -- Here we would include the specific algebra steps to convert the LHS to 5/4.
  sorry

end simplify_evaluate_expr_l274_274004


namespace trigonometric_identity_l274_274134

theorem trigonometric_identity (α : ℝ) (h : Real.sin α = 2 * Real.cos α) :
  Real.sin α ^ 2 + 2 * Real.cos α ^ 2 = 6 / 5 := 
by 
  sorry

end trigonometric_identity_l274_274134


namespace Alpha_Beta_meet_at_Alpha_Beta_meet_again_l274_274770

open Real

-- Definitions and conditions
def A : ℝ := -24
def B : ℝ := -10
def C : ℝ := 10
def Alpha_speed : ℝ := 4
def Beta_speed : ℝ := 6

-- Question 1: Prove that Alpha and Beta meet at -10.4
theorem Alpha_Beta_meet_at : 
  ∃ t : ℝ, (A + Alpha_speed * t = C - Beta_speed * t) ∧ (A + Alpha_speed * t = -10.4) :=
  sorry

-- Question 2: Prove that after reversing at t = 2, Alpha and Beta meet again at -44
theorem Alpha_Beta_meet_again :
  ∃ t z : ℝ, 
    ((t = 2) ∧ (4 * t + (14 - 4 * t) + (14 - 4 * t + 20) = 40) ∧ 
     (A + Alpha_speed * t - Alpha_speed * z = C - Beta_speed * t - Beta_speed * z) ∧ 
     (A + Alpha_speed * t - Alpha_speed * z = -44)) :=
  sorry  

end Alpha_Beta_meet_at_Alpha_Beta_meet_again_l274_274770


namespace triangle_perimeter_l274_274299

/-- Given the lengths of two sides of a triangle are 1 and 4,
    and the length of the third side is an integer, 
    prove that the perimeter of the triangle is 9 -/
theorem triangle_perimeter
  (a b : ℕ)
  (c : ℤ)
  (h₁ : a = 1)
  (h₂ : b = 4)
  (h₃ : 3 < c ∧ c < 5) :
  a + b + c = 9 :=
by sorry

end triangle_perimeter_l274_274299


namespace largest_multiple_of_11_less_than_minus_150_l274_274529

theorem largest_multiple_of_11_less_than_minus_150 : 
  ∃ n : ℤ, (n * 11 < -150) ∧ (∀ m : ℤ, (m * 11 < -150) →  n * 11 ≥ m * 11) ∧ (n * 11 = -154) :=
by
  sorry

end largest_multiple_of_11_less_than_minus_150_l274_274529


namespace center_of_circle_param_eq_l274_274507

theorem center_of_circle_param_eq (θ : ℝ) : 
  (∃ c : ℝ × ℝ, ∀ θ, 
    ∃ (x y : ℝ), 
      (x = 2 + 2 * Real.cos θ) ∧ 
      (y = 2 * Real.sin θ) ∧ 
      (x - c.1)^2 + y^2 = 4) 
  ↔ 
  c = (2, 0) :=
by
  sorry

end center_of_circle_param_eq_l274_274507


namespace probability_grunters_win_all_5_games_l274_274501

noncomputable def probability_grunters_win_game : ℚ := 4 / 5

theorem probability_grunters_win_all_5_games :
  (probability_grunters_win_game ^ 5) = 1024 / 3125 := 
  by 
  sorry

end probability_grunters_win_all_5_games_l274_274501


namespace task_completion_days_l274_274677

theorem task_completion_days (a b c: ℕ) :
  (b = a + 6) → (c = b + 3) → 
  (3 / a + 4 / b = 9 / c) →
  a = 18 ∧ b = 24 ∧ c = 27 :=
  by
  sorry

end task_completion_days_l274_274677


namespace find_x_proportionally_l274_274760

theorem find_x_proportionally (k m x z : ℝ) (h1 : ∀ y, x = k * y^2) (h2 : ∀ z, y = m / (Real.sqrt z)) (h3 : x = 7 ∧ z = 16) :
  ∃ x, x = 7 / 9 := by
  sorry

end find_x_proportionally_l274_274760


namespace pieces_picked_by_olivia_l274_274489

-- Define the conditions
def picked_by_edward : ℕ := 3
def total_picked : ℕ := 19

-- Prove the number of pieces picked up by Olivia
theorem pieces_picked_by_olivia (O : ℕ) (h : O + picked_by_edward = total_picked) : O = 16 :=
by sorry

end pieces_picked_by_olivia_l274_274489


namespace cakes_sold_correct_l274_274548

def total_cakes_baked_today : Nat := 5
def total_cakes_baked_yesterday : Nat := 3
def cakes_left : Nat := 2

def total_cakes : Nat := total_cakes_baked_today + total_cakes_baked_yesterday
def cakes_sold : Nat := total_cakes - cakes_left

theorem cakes_sold_correct :
  cakes_sold = 6 :=
by
  -- proof goes here
  sorry

end cakes_sold_correct_l274_274548


namespace log_x3y2_value_l274_274733

open Real

noncomputable def log_identity (x y : ℝ) : Prop :=
  log (x * y^4) = 1 ∧ log (x^3 * y) = 1

theorem log_x3y2_value (x y : ℝ) (h : log_identity x y) : log (x^3 * y^2) = 13 / 11 :=
  by
  sorry

end log_x3y2_value_l274_274733


namespace problem_statements_l274_274075

open Real

theorem problem_statements :
  ¬(∀ x ∈ Ioo (-π / 3) (π / 6), (2 * cos (2 * x + π / 3)) > 0) ∧
  ((∀ x, cos (x + π / 3) = cos (2 * (π / 6 - x))) ∧
  (¬ (∀ x, tan (x + π / 3) = tan (π / 6 - x))) ∧
  (∀ x, 3 * sin (2 * (x - π / 6) + π / 3) = 3 * sin (2 * x)) :=
by
  sorry

end problem_statements_l274_274075


namespace abs_diff_eq_5_l274_274628

-- Definitions of m and n, and conditions provided in the problem
variables (m n : ℝ)
hypothesis (h1 : m * n = 6)
hypothesis (h2 : m + n = 7)

-- Statement to prove
theorem abs_diff_eq_5 : |m - n| = 5 :=
by
  sorry

end abs_diff_eq_5_l274_274628


namespace inequality_abc_ge_1_sqrt_abcd_l274_274479

theorem inequality_abc_ge_1_sqrt_abcd
  (a b c d : ℝ)
  (h₀ : 0 ≤ a) (h₁ : 0 ≤ b) (h₂ : 0 ≤ c) (h₃ : 0 ≤ d)
  (h_sum : a^2 + b^2 + c^2 + d^2 = 4) :
  (a + b + c + d) / 2 ≥ 1 + Real.sqrt (a * b * c * d) :=
by
  sorry

end inequality_abc_ge_1_sqrt_abcd_l274_274479


namespace fuel_cost_is_50_cents_l274_274414

-- Define the capacities of the tanks
def small_tank_capacity : ℕ := 60
def large_tank_capacity : ℕ := 60 * 3 / 2 -- 50% larger than small tank

-- Define the number of planes
def number_of_small_planes : ℕ := 2
def number_of_large_planes : ℕ := 2

-- Define the service charge per plane
def service_charge_per_plane : ℕ := 100
def total_service_charge : ℕ :=
  service_charge_per_plane * (number_of_small_planes + number_of_large_planes)

-- Define the total cost to fill all planes
def total_cost : ℕ := 550

-- Define the total fuel capacity
def total_fuel_capacity : ℕ :=
  number_of_small_planes * small_tank_capacity + number_of_large_planes * large_tank_capacity

-- Define the total fuel cost
def total_fuel_cost : ℕ := total_cost - total_service_charge

-- Define the fuel cost per liter
def fuel_cost_per_liter : ℕ :=
  total_fuel_cost / total_fuel_capacity

theorem fuel_cost_is_50_cents :
  fuel_cost_per_liter = 50 / 100 := by
sorry

end fuel_cost_is_50_cents_l274_274414


namespace good_carrots_l274_274306

theorem good_carrots (haley_picked : ℕ) (mom_picked : ℕ) (bad_carrots : ℕ) :
  haley_picked = 39 → mom_picked = 38 → bad_carrots = 13 →
  (haley_picked + mom_picked - bad_carrots) = 64 :=
by
  sorry  -- Proof is omitted.

end good_carrots_l274_274306


namespace minimally_intersecting_triples_count_correct_l274_274429

open Finset

def minimally_intersecting_triples_count : Nat :=
  let universe := range 8
  (universe.card.choose 3) * (universe.erase x.card.choose 2) * (universe.erase y.card.choose 1) % 1000

theorem minimally_intersecting_triples_count_correct :
  minimally_intersecting_triples_count = 80 := by
  sorry

end minimally_intersecting_triples_count_correct_l274_274429


namespace breadth_of_rectangular_plot_l274_274538

theorem breadth_of_rectangular_plot (b l A : ℕ) (h1 : A = 20 * b) (h2 : l = b + 10) 
    (h3 : A = l * b) : b = 10 := by
  sorry

end breadth_of_rectangular_plot_l274_274538


namespace geometric_sequence_common_ratio_l274_274152

theorem geometric_sequence_common_ratio {a : ℕ → ℝ} (q : ℝ) 
  (h_geom : ∀ n, a (n + 1) = a n * q)
  (h_start : a 1 < 0)
  (h_increasing : ∀ n, a n < a (n + 1)) : 0 < q ∧ q < 1 :=
by
  sorry

end geometric_sequence_common_ratio_l274_274152


namespace part1_part2_part3_l274_274874

-- Definitions for the given functions
def y1 (x : ℝ) : ℝ := -x + 1
def y2 (x : ℝ) : ℝ := -3 * x + 2

-- Part (1)
theorem part1 (a : ℝ) : (∃ x : ℝ, y1 x = a + y2 x ∧ x > 0) ↔ (a > -1) := sorry

-- Part (2)
theorem part2 (x y : ℝ) (h1 : y = y1 x) (h2 : y = y2 x) : 12*x^2 + 12*x*y + 3*y^2 = 27/4 := sorry

-- Part (3)
theorem part3 (A B : ℝ) (x : ℝ) (h : (4 - 2 * x) / ((3 * x - 2) * (x - 1)) = A / y1 x + B / y2 x) : (A / B + B / A) = -17 / 4 := sorry

end part1_part2_part3_l274_274874


namespace rotated_and_shifted_line_eq_l274_274495

theorem rotated_and_shifted_line_eq :
  let rotate_line_90 (x y : ℝ) := ( -y, x )
  let shift_right (x y : ℝ) := (x + 1, y)
  ∃ (new_a new_b new_c : ℝ), 
  (∀ (x y : ℝ), (y = 3 * x → x * new_a + y * new_b + new_c = 0)) ∧ 
  (new_a = 1) ∧ (new_b = 3) ∧ (new_c = -1) := by
  sorry

end rotated_and_shifted_line_eq_l274_274495


namespace log_evaluation_l274_274878

theorem log_evaluation
  (x : ℝ)
  (h : x = (Real.log 3 / Real.log 5) ^ (Real.log 5 / Real.log 3)) :
  Real.log x / Real.log 7 = -(Real.log 5 / Real.log 3) * (Real.log (Real.log 5 / Real.log 3) / Real.log 7) :=
by
  sorry

end log_evaluation_l274_274878


namespace original_number_l274_274811

theorem original_number (x : ℝ) (h : 1.4 * x = 700) : x = 500 :=
sorry

end original_number_l274_274811


namespace shaded_region_is_correct_l274_274611

noncomputable def area_shaded_region : ℝ :=
  let r_small := (3 : ℝ) / 2
  let r_large := (15 : ℝ) / 2
  let area_small := (1 / 2) * Real.pi * r_small^2
  let area_large := (1 / 2) * Real.pi * r_large^2
  (area_large - 2 * area_small + 3 * area_small)

theorem shaded_region_is_correct :
  area_shaded_region = (117 / 4) * Real.pi :=
by
  -- The proof will go here.
  sorry

end shaded_region_is_correct_l274_274611


namespace piggy_bank_exceed_five_dollars_l274_274478

noncomputable def sequence_sum (n : ℕ) : ℕ := 2^n - 1

theorem piggy_bank_exceed_five_dollars (n : ℕ) (start_day : Nat) (day_of_week : Fin 7) :
  ∃ (n : ℕ), sequence_sum n > 500 ∧ n = 9 ∧ (start_day + n) % 7 = 2 := 
sorry

end piggy_bank_exceed_five_dollars_l274_274478


namespace time_after_2023_minutes_l274_274328

def start_time : Nat := 1 * 60 -- Start time is 1:00 a.m. in minutes from midnight, which is 60 minutes.
def elapsed_time : Nat := 2023 -- The elapsed time is 2023 minutes.

theorem time_after_2023_minutes : (start_time + elapsed_time) % 1440 = 643 := 
by
  -- 1440 represents the total minutes in a day (24 hours * 60 minutes).
  -- 643 represents the time 10:43 a.m. in minutes from midnight. This is obtained as 10 * 60 + 43 = 643.
  sorry

end time_after_2023_minutes_l274_274328


namespace repeated_digit_squares_l274_274858

theorem repeated_digit_squares :
  {n : ℕ | ∃ d : Fin 10, n = d ^ 2 ∧ (∀ m < n, m % 10 = d % 10)} ⊆ {0, 1, 4, 9} := by
  sorry

end repeated_digit_squares_l274_274858


namespace C3PO_Optimal_Play_Wins_l274_274180

def initial_number : List ℕ := [1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1]

-- Conditions for the game
structure GameConditions where
  number : List ℕ
  robots : List String
  cannot_swap : List (ℕ × ℕ) -- Pair of digits that cannot be swapped again
  cannot_start_with_zero : Bool
  c3po_starts : Bool

-- Define the initial conditions
def initial_conditions : GameConditions :=
{
  number := initial_number,
  robots := ["C3PO", "R2D2"],
  cannot_swap := [],
  cannot_start_with_zero := true,
  c3po_starts := true
}

-- Define the winning condition for C3PO
def C3PO_wins : Prop :=
  ∀ game : GameConditions, game = initial_conditions → ∃ is_c3po_winner : Bool, is_c3po_winner = true

-- The theorem statement
theorem C3PO_Optimal_Play_Wins : C3PO_wins :=
by
  sorry

end C3PO_Optimal_Play_Wins_l274_274180


namespace sum_of_values_for_one_solution_l274_274938

noncomputable def sum_of_a_values (a1 a2 : ℝ) : ℝ :=
  a1 + a2

theorem sum_of_values_for_one_solution :
  ∃ a1 a2 : ℝ, 
  (∀ x : ℝ, 4 * x^2 + (a1 + 8) * x + 9 = 0 ∨ 4 * x^2 + (a2 + 8) * x + 9 = 0) ∧
  ((a1 + 8)^2 - 144 = 0) ∧ ((a2 + 8)^2 - 144 = 0) ∧
  sum_of_a_values a1 a2 = -16 :=
by
  sorry

end sum_of_values_for_one_solution_l274_274938


namespace total_fish_weight_l274_274431

noncomputable def trout_count : ℕ := 4
noncomputable def catfish_count : ℕ := 3
noncomputable def bluegill_count : ℕ := 5

noncomputable def trout_weight : ℝ := 2
noncomputable def catfish_weight : ℝ := 1.5
noncomputable def bluegill_weight : ℝ := 2.5

theorem total_fish_weight :
  let total_trout_weight := trout_count * trout_weight,
      total_catfish_weight := catfish_count * catfish_weight,
      total_bluegill_weight := bluegill_count * bluegill_weight,
      total_weight := total_trout_weight + total_catfish_weight + total_bluegill_weight
  in total_weight = 25 := 
  by 
  sorry

end total_fish_weight_l274_274431


namespace fill_tank_time_l274_274636

variable (A_rate := 1/3)
variable (B_rate := 1/4)
variable (C_rate := -1/4)

def combined_rate := A_rate + B_rate + C_rate

theorem fill_tank_time (hA : A_rate = 1/3) (hB : B_rate = 1/4) (hC : C_rate = -1/4) : (1 / combined_rate) = 3 := by
  sorry

end fill_tank_time_l274_274636


namespace roof_ratio_l274_274787

theorem roof_ratio (L W : ℕ) (h1 : L * W = 768) (h2 : L - W = 32) : L / W = 3 := 
sorry

end roof_ratio_l274_274787


namespace condition_on_p_l274_274143

theorem condition_on_p (p q r M : ℝ) (hq : 0 < q ∧ q < 100) (hr : 0 < r ∧ r < 100) (hM : 0 < M) :
  p > (100 * (q + r)) / (100 - q - r) → 
  M * (1 + p / 100) * (1 - q / 100) * (1 - r / 100) > M :=
by
  intro h
  -- The proof will go here
  sorry

end condition_on_p_l274_274143


namespace solve_for_y_l274_274773

theorem solve_for_y (y : ℝ) (h : 6 * y^(1/3) - 3 * (y / y^(2/3)) = 10 + 2 * y^(1/3)) : y = 1000 := 
by
  sorry

end solve_for_y_l274_274773


namespace isosceles_triangles_with_perimeter_21_l274_274794

theorem isosceles_triangles_with_perimeter_21 : 
  ∃ n : ℕ, n = 5 ∧ (∀ (a b c : ℕ), a ≤ b ∧ b = c ∧ a + 2*b = 21 → 1 ≤ a ∧ a ≤ 10) :=
sorry

end isosceles_triangles_with_perimeter_21_l274_274794


namespace least_positive_integer_satifies_congruences_l274_274046

theorem least_positive_integer_satifies_congruences :
  ∃ x : ℕ, x ≡ 1 [MOD 4] ∧ x ≡ 2 [MOD 5] ∧ x ≡ 3 [MOD 6] ∧ x = 17 :=
sorry

end least_positive_integer_satifies_congruences_l274_274046


namespace basketball_court_length_difference_l274_274020

theorem basketball_court_length_difference :
  ∃ (l w : ℕ), l = 31 ∧ w = 17 ∧ l - w = 14 := by
  sorry

end basketball_court_length_difference_l274_274020


namespace minimum_distance_on_line_l274_274119

-- Define the line as a predicate
def on_line (P : ℝ × ℝ) : Prop := P.1 - P.2 = 1

-- Define the expression to be minimized
def distance_squared (P : ℝ × ℝ) : ℝ := (P.1 - 2)^2 + (P.2 - 2)^2

theorem minimum_distance_on_line :
  ∃ P : ℝ × ℝ, on_line P ∧ distance_squared P = 1 / 2 :=
sorry

end minimum_distance_on_line_l274_274119


namespace unique_three_digit_numbers_unique_three_digit_odd_numbers_l274_274442

-- Part 1: Total unique three-digit numbers
theorem unique_three_digit_numbers : 
  (finset.univ.card) * (finset.univ.erase 0).card * (finset.univ.erase 0).erase 9.card = 648 :=
sorry

-- Part 2: Total unique three-digit odd numbers
theorem unique_three_digit_odd_numbers :
  5 * ((finset.univ.erase 5).erase 0).card * ((finset.univ.erase 5).erase 0).erase 9.card = 320 :=
sorry

end unique_three_digit_numbers_unique_three_digit_odd_numbers_l274_274442


namespace set_intersection_l274_274485

noncomputable def A : Set ℝ := { x | x / (x - 1) < 0 }
noncomputable def B : Set ℝ := { x | 0 < x ∧ x < 3 }
noncomputable def expected_intersection : Set ℝ := { x | 0 < x ∧ x < 1 }

theorem set_intersection (x : ℝ) : (x ∈ A ∧ x ∈ B) ↔ x ∈ expected_intersection :=
by
  sorry

end set_intersection_l274_274485


namespace x_squared_eq_1_iff_x_eq_1_l274_274291

theorem x_squared_eq_1_iff_x_eq_1 (x : ℝ) : (x^2 = 1 → x = 1) ↔ false ∧ (x = 1 → x^2 = 1) :=
by
  sorry

end x_squared_eq_1_iff_x_eq_1_l274_274291


namespace alice_minimum_speed_l274_274813

-- Conditions
def distance : ℝ := 60 -- The distance from City A to City B in miles
def bob_speed : ℝ := 40 -- Bob's constant speed in miles per hour
def alice_delay : ℝ := 0.5 -- Alice's delay in hours before she starts

-- Question as a proof statement
theorem alice_minimum_speed : ∀ (alice_speed : ℝ), alice_speed > 60 → 
  (alice_speed * (1.5 - alice_delay) < distance) → true :=
by
  sorry

end alice_minimum_speed_l274_274813


namespace point_cannot_exist_on_line_l274_274775

theorem point_cannot_exist_on_line (m k : ℝ) (h : m * k > 0) : ¬ (2000 * m + k = 0) :=
sorry

end point_cannot_exist_on_line_l274_274775


namespace debbie_total_tape_l274_274977

def large_box_tape : ℕ := 4
def medium_box_tape : ℕ := 2
def small_box_tape : ℕ := 1
def label_tape : ℕ := 1

def large_boxes_packed : ℕ := 2
def medium_boxes_packed : ℕ := 8
def small_boxes_packed : ℕ := 5

def total_tape_used : ℕ := 
  (large_boxes_packed * (large_box_tape + label_tape)) +
  (medium_boxes_packed * (medium_box_tape + label_tape)) +
  (small_boxes_packed * (small_box_tape + label_tape))

theorem debbie_total_tape : total_tape_used = 44 := by
  sorry

end debbie_total_tape_l274_274977


namespace mixed_feed_cost_l274_274940

/-- Tim and Judy mix two kinds of feed for pedigreed dogs. They made 35 pounds of feed by mixing 
    one kind worth $0.18 per pound with another worth $0.53 per pound. They used 17 pounds of the cheaper kind in the mix.
    We are to prove that the cost per pound of the mixed feed is $0.36 per pound. -/
theorem mixed_feed_cost
  (total_weight : ℝ) (cheaper_cost : ℝ) (expensive_cost : ℝ) (cheaper_weight : ℝ)
  (total_weight_eq : total_weight = 35)
  (cheaper_cost_eq : cheaper_cost = 0.18)
  (expensive_cost_eq : expensive_cost = 0.53)
  (cheaper_weight_eq : cheaper_weight = 17) :
  ((cheaper_weight * cheaper_cost + (total_weight - cheaper_weight) * expensive_cost) / total_weight) = 0.36 :=
by
  sorry

end mixed_feed_cost_l274_274940


namespace smallest_x_l274_274389

theorem smallest_x (x : ℕ) : (x % 3 = 2) ∧ (x % 4 = 3) ∧ (x % 5 = 4) → x = 59 :=
by
  intro h
  sorry

end smallest_x_l274_274389


namespace mike_travel_time_l274_274562

-- Definitions of conditions
def dave_steps_per_min : ℕ := 85
def dave_step_length_cm : ℕ := 70
def dave_time_min : ℕ := 20
def mike_steps_per_min : ℕ := 95
def mike_step_length_cm : ℕ := 65

-- Calculate Dave's speed in cm/min
def dave_speed_cm_per_min := dave_steps_per_min * dave_step_length_cm

-- Calculate the distance to school in cm
def school_distance_cm := dave_speed_cm_per_min * dave_time_min

-- Calculate Mike's speed in cm/min
def mike_speed_cm_per_min := mike_steps_per_min * mike_step_length_cm

-- Calculate the time for Mike to get to school in minutes as a rational number
def mike_time_min := (school_distance_cm : ℚ) / mike_speed_cm_per_min

-- The proof problem statement
theorem mike_travel_time :
  mike_time_min = 19 + 2 / 7 :=
sorry

end mike_travel_time_l274_274562


namespace volumes_comparison_l274_274366

variable (a : ℝ) (h_a : a ≠ 3)

def volume_A := 3 * 3 * 3
def volume_B := 3 * 3 * a
def volume_C := a * a * 3
def volume_D := a * a * a

theorem volumes_comparison (h_a : a ≠ 3) :
  (volume_A + volume_D) > (volume_B + volume_C) :=
by
  have volume_A : ℝ := 27
  have volume_B := 9 * a
  have volume_C := 3 * a * a
  have volume_D := a * a * a
  sorry

end volumes_comparison_l274_274366


namespace two_people_lying_l274_274910

def is_lying (A B C D : Prop) : Prop :=
  (A ↔ ¬B) ∧ (B ↔ ¬C) ∧ (C ↔ ¬B) ∧ (D ↔ ¬A)

theorem two_people_lying (A B C D : Prop) (LA LB LC LD : Prop) :
  is_lying A B C D → (LA → ¬A) → (LB → ¬B) → (LC → ¬C) → (LD → ¬D) → (LA ∧ LC ∧ ¬LB ∧ ¬LD) :=
by
  sorry

end two_people_lying_l274_274910


namespace binomial_expansion_value_calculation_result_final_result_l274_274260

theorem binomial_expansion_value :
  7^4 + 4 * 7^3 + 6 * 7^2 + 4 * 7 + 1 = (7 + 1)^4 := 
sorry

theorem calculation_result :
  (7 + 1)^4 = 4096 := 
sorry

theorem final_result :
  7^4 + 4 * 7^3 + 6 * 7^2 + 4 * 7 + 1 = 4096 := 
by
  calc
    7^4 + 4 * 7^3 + 6 * 7^2 + 4 * 7 + 1 = (7 + 1)^4 := binomial_expansion_value
    ... = 4096 := calculation_result

end binomial_expansion_value_calculation_result_final_result_l274_274260


namespace count_valid_n_l274_274707

theorem count_valid_n : 
  let valid_n := {n : ℕ | 1 ≤ n ∧ n ≤ 9 ∧ (250 + n) % n = 0} 
  in valid_n.to_finset.card = 3 :=
by
  sorry

end count_valid_n_l274_274707


namespace raja_monthly_income_l274_274492

theorem raja_monthly_income (X : ℝ) 
  (h1 : 0.1 * X = 5000) : X = 50000 :=
sorry

end raja_monthly_income_l274_274492


namespace find_probability_between_0_and_1_l274_274631

-- Define a random variable X following a normal distribution N(μ, σ²)
variables {X : ℝ → ℝ} {μ σ : ℝ}
-- Define conditions:
-- Condition 1: X follows a normal distribution with mean μ and variance σ²
def normal_dist (X : ℝ → ℝ) (μ σ : ℝ) : Prop :=
  sorry  -- Assume properties of normal distribution are satisfied

-- Condition 2: P(X < 1) = 1/2
def P_X_lt_1 : Prop := 
  sorry  -- Assume that P(X < 1) = 1/2

-- Condition 3: P(X > 2) = p
def P_X_gt_2 (p : ℝ) : Prop := 
  sorry  -- Assume that P(X > 2) = p

noncomputable
def probability_X_between_0_and_1 (p : ℝ) : ℝ :=
  1/2 - p

theorem find_probability_between_0_and_1 (X : ℝ → ℝ) {μ σ p : ℝ} 
  (hX : normal_dist X μ σ)
  (h1 : P_X_lt_1)
  (h2 : P_X_gt_2 p) :
  probability_X_between_0_and_1 p = 1/2 - p := 
  sorry

end find_probability_between_0_and_1_l274_274631


namespace residue_neg_1234_mod_32_l274_274565

theorem residue_neg_1234_mod_32 : -1234 % 32 = 14 := 
by sorry

end residue_neg_1234_mod_32_l274_274565


namespace cows_and_sheep_bushels_l274_274267

theorem cows_and_sheep_bushels (bushels_per_chicken: Int) (total_bushels: Int) (num_chickens: Int) 
  (bushels_chickens: Int) (bushels_cows_sheep: Int) (num_cows: Int) (num_sheep: Int):
  bushels_per_chicken = 3 ∧ total_bushels = 35 ∧ num_chickens = 7 ∧
  bushels_chickens = num_chickens * bushels_per_chicken ∧ bushels_chickens = 21 ∧ bushels_cows_sheep = total_bushels - bushels_chickens → 
  bushels_cows_sheep = 14 := by
  sorry

end cows_and_sheep_bushels_l274_274267


namespace find_m_l274_274732

theorem find_m (a b m : ℝ) :
  (∀ x : ℝ, (x^2 - b * x + b^2) / (a * x^2 - b^2) = (m - 1) / (m + 1) → (∀ y : ℝ, x = y ∧ x = -y)) →
  c = b^2 →
  m = (a - 1) / (a + 1) :=
by
  sorry

end find_m_l274_274732


namespace towel_area_decrease_l274_274072

theorem towel_area_decrease (L B : ℝ) (hL : 0 < L) (hB : 0 < B) :
  let A := L * B
  let L' := 0.80 * L
  let B' := 0.90 * B
  let A' := L' * B'
  A' = 0.72 * A →
  ((A - A') / A) * 100 = 28 :=
by
  intros _ _ _ _
  sorry

end towel_area_decrease_l274_274072


namespace total_students_l274_274745

theorem total_students (initial_candies leftover_candies girls boys : ℕ) (h1 : initial_candies = 484)
  (h2 : leftover_candies = 4) (h3 : boys = girls + 3) (h4 : (2 * girls + boys) * (2 * girls + boys) = initial_candies - leftover_candies) :
  2 * girls + boys = 43 :=
  sorry

end total_students_l274_274745


namespace proof_f_3_eq_9_ln_3_l274_274592

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * x * Real.log x

theorem proof_f_3_eq_9_ln_3 (a : ℝ) (h : deriv (deriv (f a)) 1 = 3) : f a 3 = 9 * Real.log 3 :=
by
  sorry

end proof_f_3_eq_9_ln_3_l274_274592


namespace original_square_area_is_144_square_centimeters_l274_274069

noncomputable def area_of_original_square (x : ℝ) : ℝ :=
  x^2 - (x - 3) * (x - 5)

theorem original_square_area_is_144_square_centimeters (x : ℝ) (h : area_of_original_square x = 81) :
  (x = 12) → (x^2 = 144) :=
by
  sorry

end original_square_area_is_144_square_centimeters_l274_274069


namespace value_of_T_l274_274173

theorem value_of_T (T : ℝ) (h : (1 / 3) * (1 / 6) * T = (1 / 4) * (1 / 8) * 120) : T = 67.5 :=
sorry

end value_of_T_l274_274173


namespace isosceles_triangle_sides_l274_274687

theorem isosceles_triangle_sides (length_rope : ℝ) (one_side : ℝ) (a b : ℝ) :
  length_rope = 18 ∧ one_side = 5 ∧ a + a + one_side = length_rope ∧ b = one_side ∨ b + b + one_side = length_rope -> (a = 6.5 ∨ a = 5) ∧ (b = 6.5 ∨ b = 5) :=
by
  sorry

end isosceles_triangle_sides_l274_274687


namespace set_equality_l274_274127

theorem set_equality (a : ℤ) : 
  {z : ℤ | ∃ x : ℤ, (x - a = z ∧ a - 1 ≤ x ∧ x ≤ a + 1)} = {-1, 0, 1} :=
by {
  sorry
}

end set_equality_l274_274127


namespace a7_plus_a11_l274_274588

variable {a : ℕ → ℤ} (d : ℤ) (a₁ : ℤ)

-- Definitions based on given conditions
def S_n (n : ℕ) := (n * (2 * a₁ + (n - 1) * d)) / 2
def a_n (n : ℕ) := a₁ + (n - 1) * d

-- Condition: S_17 = 51
axiom h : S_n 17 = 51

-- Theorem to prove the question is equivalent to the answer
theorem a7_plus_a11 (h : S_n 17 = 51) : a_n 7 + a_n 11 = 6 :=
by
  -- This is where you'd fill in the actual proof, but we'll use sorry for now
  sorry

end a7_plus_a11_l274_274588


namespace monotonic_if_and_only_if_extreme_point_inequality_l274_274125

noncomputable def f (x a : ℝ) : ℝ := x^2 - 1 + a * Real.log (1 - x)

def is_monotonic (a : ℝ) : Prop := 
  ∀ x y : ℝ, x < y → f x a ≤ f y a

theorem monotonic_if_and_only_if (a : ℝ) : 
  is_monotonic a ↔ a ≥ 0.5 :=
sorry

theorem extreme_point_inequality (a : ℝ) (x1 x2 : ℝ) (hₐ : 0 < a ∧ a < 0.5) 
  (hx : x1 < x2) (hx₁₂ : f x1 a = f x2 a) : 
  f x1 a / x2 > f x2 a / x1 :=
sorry

end monotonic_if_and_only_if_extreme_point_inequality_l274_274125


namespace probability_solved_l274_274539

theorem probability_solved (pA pB pA_and_B : ℚ) :
  pA = 2 / 3 → pB = 3 / 4 → pA_and_B = (2 / 3) * (3 / 4) →
  pA + pB - pA_and_B = 11 / 12 :=
by
  intros hA hB hA_and_B
  rw [hA, hB, hA_and_B]
  sorry

end probability_solved_l274_274539


namespace hoseok_subtraction_result_l274_274412

theorem hoseok_subtraction_result:
  ∃ x : ℤ, 15 * x = 45 ∧ x - 1 = 2 :=
by
  sorry

end hoseok_subtraction_result_l274_274412


namespace least_add_to_divisible_by_17_l274_274104

/-- Given that the remainder when 433124 is divided by 17 is 2,
    prove that the least number that must be added to 433124 to make 
    it divisible by 17 is 15. -/
theorem least_add_to_divisible_by_17: 
  (433124 % 17 = 2) → 
  (∃ n, n ≥ 0 ∧ (433124 + n) % 17 = 0 ∧ n = 15) := 
by
  sorry

end least_add_to_divisible_by_17_l274_274104


namespace initial_population_l274_274785

theorem initial_population (P : ℝ) (h : P * 1.21 = 12000) : P = 12000 / 1.21 :=
by sorry

end initial_population_l274_274785


namespace least_positive_integer_solution_l274_274038

theorem least_positive_integer_solution :
  ∃ b : ℕ, b ≡ 1 [MOD 4] ∧ b ≡ 2 [MOD 5] ∧ b ≡ 3 [MOD 6] ∧ b = 37 :=
by
  sorry

end least_positive_integer_solution_l274_274038


namespace Serezha_puts_more_berries_l274_274697

theorem Serezha_puts_more_berries (berries : ℕ) 
    (Serezha_puts : ℕ) (Serezha_eats : ℕ)
    (Dima_puts : ℕ) (Dima_eats : ℕ)
    (Serezha_rate : ℕ) (Dima_rate : ℕ)
    (total_berries : berries = 450)
    (Serezha_pattern : Serezha_puts = 1 ∧ Serezha_eats = 1)
    (Dima_pattern : Dima_puts = 2 ∧ Dima_eats = 1)
    (Serezha_faster : Serezha_rate = 2 * Dima_rate) : 
    ∃ (Serezha_in_basket : ℕ) (Dima_in_basket : ℕ),
      Serezha_in_basket > Dima_in_basket ∧ Serezha_in_basket - Dima_in_basket = 50 :=
by
  sorry -- Skip the proof

end Serezha_puts_more_berries_l274_274697


namespace reciprocals_sum_of_roots_l274_274168

theorem reciprocals_sum_of_roots (r s γ δ : ℚ) (h1 : 7 * r^2 + 5 * r + 3 = 0) (h2 : 7 * s^2 + 5 * s + 3 = 0) (h3 : γ = 1/r) (h4 : δ = 1/s) :
  γ + δ = -5/3 := 
  by 
    sorry

end reciprocals_sum_of_roots_l274_274168


namespace mari_made_64_buttons_l274_274765

def mari_buttons (kendra : ℕ) : ℕ := 5 * kendra + 4

theorem mari_made_64_buttons (sue kendra mari : ℕ) (h_sue : sue = 6) (h_kendra : kendra = 2 * sue) (h_mari : mari = mari_buttons kendra) : mari = 64 :=
by 
  rw [h_sue, h_kendra, h_mari]
  simp [mari_buttons]
  sorry

end mari_made_64_buttons_l274_274765


namespace arithmetic_mean_midpoint_l274_274349

theorem arithmetic_mean_midpoint (a b : ℝ) : ∃ m : ℝ, m = (a + b) / 2 ∧ m = a + (b - a) / 2 :=
by
  sorry

end arithmetic_mean_midpoint_l274_274349


namespace maximum_elephants_l274_274953

theorem maximum_elephants (e_1 e_2 : ℕ) :
  (∃ e_1 e_2 : ℕ, 28 * e_1 + 37 * e_2 = 1036 ∧ (∀ k, 28 * e_1 + 37 * e_2 = k → k ≤ 1036 )) → 
  28 * e_1 + 37 * e_2 = 1036 :=
sorry

end maximum_elephants_l274_274953


namespace third_smallest_four_digit_Pascal_triangle_l274_274221

theorem third_smallest_four_digit_Pascal_triangle : 
  ∃ n : ℕ, (binomial (n + 2) (n + 2) = 1002) ∧ ∀ k : ℕ, (k < (n + 2) → binomial (k + 2) (k + 2) < 1002) :=
by
  sorry

end third_smallest_four_digit_Pascal_triangle_l274_274221


namespace inscribed_circle_area_l274_274717

/-- Defining the inscribed circle problem and its area. -/
theorem inscribed_circle_area (l : ℝ) (h₁ : 90 = 90) (h₂ : true) : 
  ∃ r : ℝ, (r = (2 * (Real.sqrt 2 - 1) * l / Real.pi)) ∧ ((Real.pi * r ^ 2) = (12 - 8 * Real.sqrt 2) * l ^ 2 / Real.pi) :=
  sorry

end inscribed_circle_area_l274_274717


namespace least_positive_integer_solution_l274_274036

theorem least_positive_integer_solution :
  ∃ b : ℕ, b ≡ 1 [MOD 4] ∧ b ≡ 2 [MOD 5] ∧ b ≡ 3 [MOD 6] ∧ b = 37 :=
by
  sorry

end least_positive_integer_solution_l274_274036


namespace alpha_value_l274_274656

theorem alpha_value (b : ℝ) : (∀ x : ℝ, (|2 * x - 3| < 2) ↔ (x^2 + -3 * x + b < 0)) :=
by
  sorry

end alpha_value_l274_274656


namespace smallest_N_circular_table_l274_274543

theorem smallest_N_circular_table (N chairs : ℕ) (circular_seating : N < chairs) :
  (∀ new_person_reserved : ℕ, new_person_reserved < chairs →
    (∃ i : ℕ, (i < N) ∧ (new_person_reserved = (i + 1) % chairs ∨ 
                           new_person_reserved = (i - 1) % chairs))) ↔ N = 18 := by
sorry

end smallest_N_circular_table_l274_274543


namespace certain_number_l274_274055

theorem certain_number (n q1 q2: ℕ) (h1 : 49 = n * q1 + 4) (h2 : 66 = n * q2 + 6): n = 15 :=
sorry

end certain_number_l274_274055


namespace number_of_even_multiples_of_3_l274_274841

theorem number_of_even_multiples_of_3 :
  ∃ n, n = (198 - 6) / 6 + 1 := by
  sorry

end number_of_even_multiples_of_3_l274_274841


namespace distinct_ways_to_place_digits_l274_274598

theorem distinct_ways_to_place_digits : 
  (∃ (grid : fin 6 → option ℕ), 
    multiset.card (multiset.filter (λ x, x ≠ none) (multiset.map grid (multiset.fin_range 6)))
    = 4 ∧ (multiset.erase_dup $ multiset.filter_map id $ multiset.map grid (multiset.fin_range 6)) = {1,2,3,4}) → 15 * (nat.factorial 4) = 360 :=
by
  sorry

end distinct_ways_to_place_digits_l274_274598


namespace trig_expr_correct_l274_274508

noncomputable def trig_expr : ℝ := Real.sin (20 * Real.pi / 180) * Real.cos (10 * Real.pi / 180) - 
                                   Real.cos (160 * Real.pi / 180) * Real.sin (170 * Real.pi / 180)

theorem trig_expr_correct : trig_expr = 1 / 2 := 
  sorry

end trig_expr_correct_l274_274508


namespace sum_of_squares_of_roots_eq_14_l274_274720

theorem sum_of_squares_of_roots_eq_14 {α β γ : ℝ}
  (h1: ∀ x: ℝ, (x^3 - 6*x^2 + 11*x - 6 = 0) → (x = α ∨ x = β ∨ x = γ)) :
  α^2 + β^2 + γ^2 = 14 :=
by
  sorry

end sum_of_squares_of_roots_eq_14_l274_274720


namespace units_digit_of_k_squared_plus_2_to_k_l274_274623

theorem units_digit_of_k_squared_plus_2_to_k (k : ℕ) (h : k = 2012 ^ 2 + 2 ^ 2014) : (k ^ 2 + 2 ^ k) % 10 = 5 := by
  sorry

end units_digit_of_k_squared_plus_2_to_k_l274_274623


namespace ratio_of_heights_eq_three_twentieths_l274_274827

noncomputable def base_circumference : ℝ := 32 * Real.pi
noncomputable def original_height : ℝ := 60
noncomputable def shorter_volume : ℝ := 768 * Real.pi

theorem ratio_of_heights_eq_three_twentieths
  (base_circumference : ℝ)
  (original_height : ℝ)
  (shorter_volume : ℝ)
  (h' : ℝ)
  (ratio : ℝ) :
  base_circumference = 32 * Real.pi →
  original_height = 60 →
  shorter_volume = 768 * Real.pi →
  (1 / 3 * Real.pi * (base_circumference / (2 * Real.pi))^2 * h') = shorter_volume →
  ratio = h' / original_height →
  ratio = 3 / 20 := 
by
  intros h₁ h₂ h₃ h₄ h₅
  sorry

end ratio_of_heights_eq_three_twentieths_l274_274827


namespace polar_to_rectangular_4sqrt2_pi_over_4_l274_274559

theorem polar_to_rectangular_4sqrt2_pi_over_4 :
  let r := 4 * Real.sqrt 2
  let θ := Real.pi / 4
  let x := r * Real.cos θ
  let y := r * Real.sin θ
  (x, y) = (4, 4) :=
by
  let r := 4 * Real.sqrt 2
  let θ := Real.pi / 4
  let x := r * Real.cos θ
  let y := r * Real.sin θ
  sorry

end polar_to_rectangular_4sqrt2_pi_over_4_l274_274559


namespace regular_polygon_sides_l274_274826

theorem regular_polygon_sides (O A B : Type) (angle_OAB : ℝ) 
  (h_angle : angle_OAB = 72) : 
  (360 / angle_OAB = 5) := 
by 
  sorry

end regular_polygon_sides_l274_274826


namespace find_ratio_l274_274284

-- Definition of the system of equations with k = 5
def system_of_equations (x y z : ℝ) :=
  x + 10 * y + 5 * z = 0 ∧
  2 * x + 5 * y + 4 * z = 0 ∧
  3 * x + 6 * y + 5 * z = 0

-- Proof that if (x, y, z) solves the system, then yz / x^2 = -3 / 49
theorem find_ratio (x y z : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0) (h : system_of_equations x y z) :
  (y * z) / (x ^ 2) = -3 / 49 :=
by
  -- Substitute the system of equations and solve for the ratio.
  sorry

end find_ratio_l274_274284


namespace number_of_liars_l274_274471

theorem number_of_liars {n : ℕ} (h1 : n ≥ 1) (h2 : n ≤ 200) (h3 : ∃ k : ℕ, k < n ∧ k ≥ 1) : 
  (∃ l : ℕ, l = 199 ∨ l = 200) := 
sorry

end number_of_liars_l274_274471


namespace remainder_sum_of_squares_mod_13_l274_274380

-- Define the sum of squares of the first n natural numbers
def sum_of_squares (n : ℕ) : ℕ := (n * (n + 1) * (2 * n + 1)) / 6

-- Prove that the remainder when the sum of squares of the first 20 natural numbers
-- is divided by 13 is 10
theorem remainder_sum_of_squares_mod_13 : sum_of_squares 20 % 13 = 10 := 
by
  -- Here you can imagine the relevant steps or intermediate computations might go, if needed.
  sorry -- Placeholder for the proof.

end remainder_sum_of_squares_mod_13_l274_274380


namespace books_left_over_l274_274466

def total_books (box_count : ℕ) (books_per_box : ℤ) : ℤ :=
  box_count * books_per_box

theorem books_left_over
  (box_count : ℕ)
  (books_per_box : ℤ)
  (new_box_capacity : ℤ)
  (books_total : ℤ := total_books box_count books_per_box) :
  box_count = 1500 →
  books_per_box = 35 →
  new_box_capacity = 43 →
  books_total % new_box_capacity = 40 :=
by
  intros
  sorry

end books_left_over_l274_274466


namespace actual_distance_traveled_l274_274060

-- Given conditions
variables (D : ℝ)
variables (H : D / 5 = (D + 20) / 15)

-- The proof problem statement
theorem actual_distance_traveled : D = 10 :=
by
  sorry

end actual_distance_traveled_l274_274060


namespace complete_square_l274_274229

theorem complete_square {x : ℝ} :
  x^2 - 6 * x - 8 = 0 ↔ (x - 3)^2 = 17 :=
sorry

end complete_square_l274_274229


namespace sqrt_simplify_l274_274254

theorem sqrt_simplify (p : ℝ) :
  (Real.sqrt (12 * p) * Real.sqrt (7 * p^3) * Real.sqrt (15 * p^5)) =
  6 * p^4 * Real.sqrt (35 * p) :=
by
  sorry

end sqrt_simplify_l274_274254


namespace abc_inequality_l274_274651

theorem abc_inequality (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (h1 : a^2 < 16 * b * c) (h2 : b^2 < 16 * c * a) (h3 : c^2 < 16 * a * b) :
  a^2 + b^2 + c^2 < 2 * (a * b + b * c + c * a) :=
by sorry

end abc_inequality_l274_274651


namespace count_divisible_digits_l274_274708

def is_divisible (a b : ℕ) : Prop := b ≠ 0 ∧ a % b = 0

theorem count_divisible_digits :
  ∃! (s : Finset ℕ), s = {n | n ∈ Finset.range 10 ∧ n ≠ 0 ∧ is_divisible (25 * n) n} ∧ (Finset.card s = 3) := 
by
  sorry

end count_divisible_digits_l274_274708


namespace correct_equation_l274_274150

variable (x : ℕ)

def three_people_per_cart_and_two_empty_carts (x : ℕ) :=
  x / 3 + 2

def two_people_per_cart_and_nine_walking (x : ℕ) :=
  (x - 9) / 2

theorem correct_equation (x : ℕ) :
  three_people_per_cart_and_two_empty_carts x = two_people_per_cart_and_nine_walking x :=
by
  sorry

end correct_equation_l274_274150


namespace union_sets_l274_274345

-- Define the sets A and B
def A : Set ℤ := {0, 1}
def B : Set ℤ := {x : ℤ | (x + 2) * (x - 1) < 0}

-- The theorem to be proven
theorem union_sets : A ∪ B = {-1, 0, 1} :=
by
  sorry

end union_sets_l274_274345


namespace scott_runs_84_miles_in_a_month_l274_274003

-- Define the number of miles Scott runs from Monday to Wednesday in a week.
def milesMonToWed : ℕ := 3 * 3

-- Define the number of miles Scott runs on Thursday and Friday in a week.
def milesThuFri : ℕ := 3 * 2 * 2

-- Define the total number of miles Scott runs in a week.
def totalMilesPerWeek : ℕ := milesMonToWed + milesThuFri

-- Define the number of weeks in a month.
def weeksInMonth : ℕ := 4

-- Define the total number of miles Scott runs in a month.
def totalMilesInMonth : ℕ := totalMilesPerWeek * weeksInMonth

-- Statement to prove that Scott runs 84 miles in a month with 4 weeks.
theorem scott_runs_84_miles_in_a_month : totalMilesInMonth = 84 := by
  -- The proof is omitted for this example.
  sorry

end scott_runs_84_miles_in_a_month_l274_274003


namespace total_students_in_school_l274_274644

theorem total_students_in_school
  (students_per_group : ℕ) (groups_per_class : ℕ) (number_of_classes : ℕ)
  (h1 : students_per_group = 7) (h2 : groups_per_class = 9) (h3 : number_of_classes = 13) :
  students_per_group * groups_per_class * number_of_classes = 819 := by
  -- The proof steps would go here
  sorry

end total_students_in_school_l274_274644


namespace third_smallest_four_digit_in_pascals_triangle_l274_274200

theorem third_smallest_four_digit_in_pascals_triangle :
  ∃ n m k, binom n m = 1002 ∧ binom n (m + 1) = 1001 ∧ binom n (m + 2) = 1000 :=
sorry

end third_smallest_four_digit_in_pascals_triangle_l274_274200


namespace sin_double_alpha_l274_274714

theorem sin_double_alpha (α : ℝ) 
  (h : Real.tan (α - Real.pi / 4) = Real.sqrt 2 / 4) : 
  Real.sin (2 * α) = 7 / 9 := by
  sorry

end sin_double_alpha_l274_274714


namespace marsha_pay_per_mile_l274_274634

variable (distance1 distance2 payment : ℝ)
variable (distance3 : ℝ := distance2 / 2)
variable (totalDistance := distance1 + distance2 + distance3)

noncomputable def payPerMile (payment : ℝ) (totalDistance : ℝ) : ℝ :=
  payment / totalDistance

theorem marsha_pay_per_mile
  (distance1: ℝ := 10)
  (distance2: ℝ := 28)
  (payment: ℝ := 104)
  (distance3: ℝ := distance2 / 2)
  (totalDistance: ℝ := distance1 + distance2 + distance3)
  : payPerMile payment totalDistance = 2 := by
  sorry

end marsha_pay_per_mile_l274_274634


namespace plane_equation_of_points_l274_274086

theorem plane_equation_of_points :
  ∃ A B C D : ℤ, A > 0 ∧ Int.gcd (Int.gcd (Int.gcd A (Int.natAbs B)) (Int.natAbs C)) (Int.natAbs D) = 1 ∧
  ∀ x y z : ℤ, (15 * x + 7 * y + 17 * z - 26 = 0) ↔
  (A * x + B * y + C * z + D = 0) :=
by
  sorry

end plane_equation_of_points_l274_274086


namespace remainder_two_when_divided_by_3_l274_274350

-- Define the main theorem stating that for any positive integer n,
-- n^3 + 3/2 * n^2 + 1/2 * n - 1 leaves a remainder of 2 when divided by 3.

theorem remainder_two_when_divided_by_3 (n : ℕ) (h : n > 0) : 
  (n^3 + (3 / 2) * n^2 + (1 / 2) * n - 1) % 3 = 2 := 
sorry

end remainder_two_when_divided_by_3_l274_274350


namespace line_through_point_equidistant_l274_274416

open Real

structure Point where
  x : ℝ
  y : ℝ

def line_equation (a b c : ℝ) (p : Point) : Prop :=
  a * p.x + b * p.y + c = 0

def equidistant (p1 p2 : Point) (l : ℝ × ℝ × ℝ) : Prop :=
  let (a, b, c) := l
  let dist_from_p1 := abs (a * p1.x + b * p1.y + c) / sqrt (a^2 + b^2)
  let dist_from_p2 := abs (a * p2.x + b * p2.y + c) / sqrt (a^2 + b^2)
  dist_from_p1 = dist_from_p2

theorem line_through_point_equidistant (a b c : ℝ)
  (P : Point) (A : Point) (B : Point) :
  (P = ⟨1, 2⟩) →
  (A = ⟨2, 2⟩) →
  (B = ⟨4, -6⟩) →
  line_equation a b c P →
  equidistant A B (a, b, c) →
  (a = 2 ∧ b = 1 ∧ c = -4) :=
by
  sorry

end line_through_point_equidistant_l274_274416


namespace greatest_measure_length_l274_274233

theorem greatest_measure_length :
  let l1 := 18000
  let l2 := 50000
  let l3 := 1520
  ∃ d, d = Int.gcd (Int.gcd l1 l2) l3 ∧ d = 40 :=
by
  sorry

end greatest_measure_length_l274_274233


namespace circle_tangent_l274_274662

variables {O M : ℝ} {R : ℝ}

theorem circle_tangent
  (r : ℝ)
  (hOM_pos : O ≠ M)
  (hO : O > 0)
  (hR : R > 0)
  (h_distinct : ∀ (m n : ℝ), m ≠ n → abs (m - n) ≠ 0) :
  (r = abs (O - M) - R) ∨ (r = abs (O - M) + R) ∨ (r = R - abs (O - M)) →
  (abs ((O - M)^2 + r^2 - R^2) = 2 * R * r) :=
sorry

end circle_tangent_l274_274662


namespace kabadi_players_l274_274917

def people_play_kabadi (Kho_only Both Total : ℕ) : Prop :=
  ∃ K : ℕ, Kho_only = 20 ∧ Both = 5 ∧ Total = 30 ∧ K = Total - Kho_only ∧ (K + Both) = 15

theorem kabadi_players :
  people_play_kabadi 20 5 30 :=
by
  sorry

end kabadi_players_l274_274917


namespace secret_known_on_monday_l274_274769

def students_know_secret (n : ℕ) : ℕ :=
  (3^(n + 1) - 1) / 2

theorem secret_known_on_monday :
  ∃ n : ℕ, students_know_secret n = 3280 ∧ (n + 1) % 7 = 0 :=
by
  sorry

end secret_known_on_monday_l274_274769


namespace min_f_triangle_sides_l274_274875

noncomputable def f (x : ℝ) : ℝ :=
  let a := (2 * Real.cos x ^ 2, Real.sqrt 3)
  let b := (1, Real.sin (2 * x))
  (a.1 * b.1 + a.2 * b.2) - 2

theorem min_f (x : ℝ) (h1 : -Real.pi / 6 ≤ x) (h2 : x ≤ Real.pi / 3) :
  ∃ x₀, f x₀ = -2 ∧ ∀ x, -Real.pi / 6 ≤ x ∧ x ≤ Real.pi / 3 → f x ≥ -2 :=
  sorry

theorem triangle_sides (a b C : ℝ) (h1 : f C = 1) (h2 : C = Real.pi / 6)
  (h3 : 1 = 1) (h4 : a * b = 2 * Real.sqrt 3) (h5 : a > b) :
  a = 2 ∧ b = Real.sqrt 3 :=
  sorry

end min_f_triangle_sides_l274_274875


namespace nth_equation_identity_l274_274000

theorem nth_equation_identity (n : ℕ) (h : n ≥ 1) : 
  (n / (n + 2 : ℚ)) * (1 - 1 / (n + 1 : ℚ)) = (n^2 / ((n + 1) * (n + 2) : ℚ)) := 
by 
  sorry

end nth_equation_identity_l274_274000


namespace us2_eq_3958_div_125_l274_274897

-- Definitions based on conditions
def t (x : ℚ) : ℚ := 5 * x - 12
def s (t_x : ℚ) : ℚ := (2 : ℚ) ^ 2 + 3 * 2 - 2
def u (s_t_x : ℚ) : ℚ := (14 : ℚ) / 5 ^ 3 + 2 * (14 / 5) ^ 2 - 14 / 5 + 4

-- Prove that u(s(2)) = 3958 / 125
theorem us2_eq_3958_div_125 : u (s (2)) = 3958 / 125 := by
  sorry

end us2_eq_3958_div_125_l274_274897


namespace area_of_enclosed_region_l274_274929

open Real

noncomputable def integral_area : ℝ :=
  2 * (∫ (θ : ℝ) in (π / 4)..(3 * π / 4), 4 + 4 * cos (2 * θ) + (cos (2 * θ)) ^ 2) sorry

theorem area_of_enclosed_region :
  integral_area = (9 * π / 2) - 8 := sorry

end area_of_enclosed_region_l274_274929


namespace power_mod_result_l274_274691

theorem power_mod_result :
  (47 ^ 1235 - 22 ^ 1235) % 8 = 7 := by
  sorry

end power_mod_result_l274_274691


namespace conditions_for_unique_solution_l274_274710

noncomputable def is_solution (n p x y z : ℕ) : Prop :=
x + p * y = n ∧ x + y = p^z

def unique_positive_integer_solution (n p : ℕ) : Prop :=
∃! (x y z : ℕ), 0 < x ∧ 0 < y ∧ 0 < z ∧ is_solution n p x y z

theorem conditions_for_unique_solution {n p : ℕ} :
  (1 < p) ∧ ((n - 1) % (p - 1) = 0) ∧ ∀ k : ℕ, n ≠ p^k ↔ unique_positive_integer_solution n p :=
sorry

end conditions_for_unique_solution_l274_274710


namespace notebooks_difference_l274_274329

theorem notebooks_difference :
  ∀ (Jac_left Jac_Paula Jac_Mike Ger_not Jac_init : ℕ),
  Ger_not = 8 →
  Jac_left = 10 →
  Jac_Paula = 5 →
  Jac_Mike = 6 →
  Jac_init = Jac_left + Jac_Paula + Jac_Mike →
  Jac_init - Ger_not = 13 := 
by
  intros Jac_left Jac_Paula Jac_Mike Ger_not Jac_init
  intros Ger_not_8 Jac_left_10 Jac_Paula_5 Jac_Mike_6 Jac_init_def
  sorry

end notebooks_difference_l274_274329


namespace tan_half_angle_is_two_l274_274718

-- Define the setup
variables (α : ℝ) (H1 : α ∈ Icc (π/2) π) (H2 : 3 * Real.sin α + 4 * Real.cos α = 0)

-- Define the main theorem
theorem tan_half_angle_is_two : Real.tan (α / 2) = 2 :=
sorry

end tan_half_angle_is_two_l274_274718


namespace derivative_odd_function_l274_274344

theorem derivative_odd_function (a b c : ℝ) (f : ℝ → ℝ) (h : ∀ x, f x = a * x^3 + b * x^2 + c * x + 2) 
    (h_deriv_odd : ∀ x, deriv f (-x) = - deriv f x) : a^2 + c^2 ≠ 0 :=
by
  sorry

end derivative_odd_function_l274_274344


namespace sum_100_consecutive_from_neg49_l274_274387

noncomputable def sum_of_consecutive_integers (n : ℕ) (first_term : ℤ) : ℤ :=
  n * ( first_term + (first_term + n - 1) ) / 2

theorem sum_100_consecutive_from_neg49 : sum_of_consecutive_integers 100 (-49) = 50 :=
by sorry

end sum_100_consecutive_from_neg49_l274_274387


namespace balance_two_diamonds_three_bullets_l274_274795

-- Define the variables
variables (a b c : ℝ)

-- Define the conditions as hypotheses
def condition1 : Prop := 3 * a + b = 9 * c
def condition2 : Prop := a = b + c

-- Goal is to prove two diamonds (2 * b) balance three bullets (3 * c)
theorem balance_two_diamonds_three_bullets (h1 : condition1 a b c) (h2 : condition2 a b c) : 
  2 * b = 3 * c := 
by 
  sorry

end balance_two_diamonds_three_bullets_l274_274795


namespace least_number_to_add_l274_274951

theorem least_number_to_add (n d : ℕ) (h₁ : n = 1054) (h₂ : d = 23) : ∃ x, (n + x) % d = 0 ∧ x = 4 := by
  sorry

end least_number_to_add_l274_274951


namespace suzanne_donation_total_l274_274010

theorem suzanne_donation_total : 
  (10 + 10 * 2 + 10 * 2^2 + 10 * 2^3 + 10 * 2^4 = 310) :=
by
  sorry

end suzanne_donation_total_l274_274010


namespace factor_expression_l274_274987

theorem factor_expression (x : ℝ) : 45 * x^2 + 135 * x = 45 * x * (x + 3) := 
by
  sorry

end factor_expression_l274_274987


namespace pie_chart_probability_l274_274963

theorem pie_chart_probability
  (P_W P_X P_Z : ℚ)
  (h_W : P_W = 1/4)
  (h_X : P_X = 1/3)
  (h_Z : P_Z = 1/6) :
  1 - P_W - P_X - P_Z = 1/4 :=
by
  -- The detailed proof steps are omitted as per the requirement.
  sorry

end pie_chart_probability_l274_274963


namespace students_voted_for_meat_l274_274749

theorem students_voted_for_meat (total_votes veggies_votes : ℕ) (h_total: total_votes = 672) (h_veggies: veggies_votes = 337) :
  total_votes - veggies_votes = 335 := 
by
  -- Proof steps go here
  sorry

end students_voted_for_meat_l274_274749


namespace apples_rate_per_kg_l274_274798

variable (A : ℝ)

theorem apples_rate_per_kg (h : 8 * A + 9 * 65 = 1145) : A = 70 :=
sorry

end apples_rate_per_kg_l274_274798


namespace prob_three_cards_in_sequence_l274_274967

theorem prob_three_cards_in_sequence : 
  let total_cards := 52
  let spades_count := 13
  let hearts_count := 13
  let sequence_prob := (spades_count / total_cards) * (hearts_count / (total_cards - 1)) * ((spades_count - 1) / (total_cards - 2))
  sequence_prob = (78 / 5100) :=
by
  sorry

end prob_three_cards_in_sequence_l274_274967


namespace solution_set_of_inequality_l274_274364

theorem solution_set_of_inequality :
  {x : ℝ | (x + 1) / x ≤ 3} = {x : ℝ | x < 0} ∪ {x : ℝ | x ≥ 1 / 2} :=
by
  sorry

end solution_set_of_inequality_l274_274364


namespace Andy_is_1_year_older_l274_274743

variable Rahim_current_age : ℕ
variable Rahim_age_in_5_years : ℕ := Rahim_current_age + 5
variable Andy_age_in_5_years : ℕ := 2 * Rahim_current_age
variable Andy_current_age : ℕ := Andy_age_in_5_years - 5

theorem Andy_is_1_year_older :
  Rahim_current_age = 6 → Andy_current_age = Rahim_current_age + 1 :=
by
  sorry

end Andy_is_1_year_older_l274_274743


namespace total_fortunate_numbers_is_65_largest_odd_fortunate_number_is_1995_l274_274621

-- Definition of properties required as per the given conditions
def is_fortunate_number (abcd ab cd : ℕ) : Prop :=
  abcd = 100 * ab + cd ∧
  ab ≠ cd ∧
  ab ∣ cd ∧
  cd ∣ abcd

-- Total number of fortunate numbers is 65
theorem total_fortunate_numbers_is_65 : 
  ∃ n : ℕ, n = 65 ∧ 
  ∀(abcd ab cd : ℕ), is_fortunate_number abcd ab cd → n = 65 :=
sorry

-- Largest odd fortunate number is 1995
theorem largest_odd_fortunate_number_is_1995 : 
  ∃ abcd : ℕ, abcd = 1995 ∧ 
  ∀(abcd' ab cd : ℕ), is_fortunate_number abcd' ab cd ∧ cd % 2 = 1 → abcd = 1995 :=
sorry

end total_fortunate_numbers_is_65_largest_odd_fortunate_number_is_1995_l274_274621


namespace dvd_sold_168_l274_274604

/-- 
Proof that the number of DVDs sold (D) is 168 given the conditions:
1) D = 1.6 * C
2) D + C = 273 
-/
theorem dvd_sold_168 (C D : ℝ) (h1 : D = 1.6 * C) (h2 : D + C = 273) : D = 168 := 
sorry

end dvd_sold_168_l274_274604


namespace income_increase_l274_274742

variable (a : ℝ)

theorem income_increase (h : ∃ a : ℝ, a > 0):
  a * 1.142 = a * 1 + a * 0.142 :=
by
  sorry

end income_increase_l274_274742


namespace percent_pelicans_non_swans_l274_274159

noncomputable def percent_geese := 0.20
noncomputable def percent_swans := 0.30
noncomputable def percent_herons := 0.10
noncomputable def percent_ducks := 0.25
noncomputable def percent_pelicans := 0.15

theorem percent_pelicans_non_swans :
  (percent_pelicans / (1 - percent_swans)) * 100 = 21.43 := 
by 
  sorry

end percent_pelicans_non_swans_l274_274159


namespace sum_even_102_to_200_l274_274793

theorem sum_even_102_to_200 :
  let sum_first_50_even := (50 / 2) * (2 + 100)
  let sum_102_to_200 := 3 * sum_first_50_even
  sum_102_to_200 = 7650 :=
by
  let sum_first_50_even := (50 / 2) * (2 + 100)
  let sum_102_to_200 := 3 * sum_first_50_even
  have : sum_102_to_200 = 7650 := by
    rw [sum_first_50_even, sum_102_to_200]
    sorry
  exact this

end sum_even_102_to_200_l274_274793


namespace ones_digit_of_22_to_22_11_11_l274_274576

theorem ones_digit_of_22_to_22_11_11 : (22 ^ (22 * (11 ^ 11))) % 10 = 4 :=
by
  sorry

end ones_digit_of_22_to_22_11_11_l274_274576


namespace denomination_of_four_bills_l274_274766

theorem denomination_of_four_bills (X : ℕ) (h1 : 10 * 20 + 8 * 10 + 4 * X = 300) : X = 5 :=
by
  -- proof goes here
  sorry

end denomination_of_four_bills_l274_274766


namespace find_m_l274_274850

theorem find_m (a b c m : ℕ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c) (h_pos_m : 0 < m) (h : a * b * c * m = 1 + a^2 + b^2 + c^2) : 
  m = 4 :=
sorry

end find_m_l274_274850


namespace same_color_probability_l274_274311

/-- There are 7 red plates and 5 blue plates. We want to prove that the probability of
    selecting 3 plates, where all are of the same color, is 9/44. -/
theorem same_color_probability :
  let total_plates := 12
  let total_ways_to_choose := Nat.choose total_plates 3
  let red_plates := 7
  let blue_plates := 5
  let ways_to_choose_red := Nat.choose red_plates 3
  let ways_to_choose_blue := Nat.choose blue_plates 3
  let favorable_ways_to_choose := ways_to_choose_red + ways_to_choose_blue
  ∃ (prob : ℚ), prob = (favorable_ways_to_choose : ℚ) / (total_ways_to_choose : ℚ) ∧
                 prob = 9 / 44 :=
by
  sorry

end same_color_probability_l274_274311


namespace greatest_divisor_of_arithmetic_sequence_sum_l274_274805

theorem greatest_divisor_of_arithmetic_sequence_sum :
  ∀ (x c : ℕ), (∃ (n : ℕ), n = 12 * x + 66 * c) → (6 ∣ n) :=
by
  intro x c _ h
  cases h with n h_eq
  use 6
  sorry

end greatest_divisor_of_arithmetic_sequence_sum_l274_274805


namespace total_number_of_bottles_l274_274403

def water_bottles := 2 * 12
def orange_juice_bottles := (7 / 4) * 12
def apple_juice_bottles := water_bottles + 6
def total_bottles := water_bottles + orange_juice_bottles + apple_juice_bottles

theorem total_number_of_bottles :
  total_bottles = 75 :=
by
  sorry

end total_number_of_bottles_l274_274403


namespace smallest_b_for_factoring_l274_274283

theorem smallest_b_for_factoring :
  ∃ b : ℕ, b > 0 ∧
    (∀ r s : ℤ, r * s = 2016 → r + s ≠ b) ∧
    (∀ r s : ℤ, r * s = 2016 → r + s = b → b = 92) :=
sorry

end smallest_b_for_factoring_l274_274283


namespace abs_difference_of_mn_6_and_sum_7_l274_274626

theorem abs_difference_of_mn_6_and_sum_7 (m n : ℝ) (h₁ : m * n = 6) (h₂ : m + n = 7) : |m - n| = 5 := 
sorry

end abs_difference_of_mn_6_and_sum_7_l274_274626


namespace find_y_l274_274943

theorem find_y (x y : ℕ) (h_pos_y : 0 < y) (h_rem : x % y = 7) (h_div : x = 86 * y + (1 / 10) * y) :
  y = 70 :=
sorry

end find_y_l274_274943


namespace total_order_cost_l274_274699

theorem total_order_cost :
  let c := 2 * 30
  let w := 9 * 15
  let s := 50
  c + w + s = 245 := 
by
  linarith

end total_order_cost_l274_274699


namespace sum_n_terms_max_sum_n_l274_274582

variable {a : ℕ → ℚ} (S : ℕ → ℚ)
variable (d a_1 : ℚ)

-- Conditions given in the problem
axiom sum_first_10 : S 10 = 125 / 7
axiom sum_first_20 : S 20 = -250 / 7
axiom sum_arithmetic_seq : ∀ n, S n = n * (a 1 + a n) / 2

-- Define the first term and common difference for the arithmetic sequence
axiom common_difference : ∀ n, a n = a_1 + (n - 1) * d

-- Theorem 1: Sum of the first n terms
theorem sum_n_terms (n : ℕ) : S n = (75 * n - 5 * n^2) / 14 := 
  sorry

-- Theorem 2: Value of n that maximizes S_n
theorem max_sum_n : n = 7 ∨ n = 8 ↔ (∀ m, S m ≤ S 7 ∨ S m ≤ S 8) := 
  sorry

end sum_n_terms_max_sum_n_l274_274582


namespace dolphins_to_be_trained_next_month_l274_274512

theorem dolphins_to_be_trained_next_month :
  ∀ (total_dolphins fully_trained remaining trained_next_month : ℕ),
    total_dolphins = 20 →
    fully_trained = (1 / 4 : ℚ) * total_dolphins →
    remaining = total_dolphins - fully_trained →
    (2 / 3 : ℚ) * remaining = 10 →
    trained_next_month = remaining - 10 →
    trained_next_month = 5 :=
by
  intros total_dolphins fully_trained remaining trained_next_month
  intro h1 h2 h3 h4 h5
  sorry

end dolphins_to_be_trained_next_month_l274_274512


namespace maximum_value_attains_maximum_value_l274_274481

theorem maximum_value
  (a b c : ℝ)
  (h₀ : 0 ≤ a) (h₁ : 0 ≤ b) (h₂ : 0 ≤ c)
  (h₃ : a + b + c = 1) :
  (ab / (a + b)) + (ac / (a + c)) + (bc / (b + c)) ≤ 1 / 2 :=
sorry

theorem attains_maximum_value :
  ∃ (a b c : ℝ), 0 ≤ a ∧ 0 ≤ b ∧ 0 ≤ c ∧ a + b + c = 1 ∧
  (ab / (a + b)) + (ac / (a + c)) + (bc / (b + c)) = 1 / 2 :=
sorry

end maximum_value_attains_maximum_value_l274_274481


namespace composite_A_l274_274911

def A : ℕ := 10^1962 + 1

theorem composite_A : ∃ p q : ℕ, 1 < p ∧ 1 < q ∧ A = p * q :=
  sorry

end composite_A_l274_274911


namespace function_values_l274_274440

noncomputable def f (a b c x : ℝ) : ℝ := a * Real.cos x + b * x^2 + c

theorem function_values (a b c : ℝ) : 
  f a b c 1 = 1 ∧ f a b c (-1) = 1 := 
by
  sorry

end function_values_l274_274440


namespace smallest_x_l274_274388

theorem smallest_x (x : ℕ) : (x % 3 = 2) ∧ (x % 4 = 3) ∧ (x % 5 = 4) → x = 59 :=
by
  intro h
  sorry

end smallest_x_l274_274388


namespace find_t_l274_274178

variables {a b c r s t : ℝ}

theorem find_t (h1 : a + b + c = -3)
             (h2 : a * b + b * c + c * a = 4)
             (h3 : a * b * c = -1)
             (h4 : ∀ x, x^3 + 3*x^2 + 4*x + 1 = 0 → (x = a ∨ x = b ∨ x = c))
             (h5 : ∀ y, y^3 + r*y^2 + s*y + t = 0 → (y = a + b ∨ y = b + c ∨ y = c + a))
             : t = 11 :=
sorry

end find_t_l274_274178


namespace find_number_l274_274241

noncomputable def question (x : ℝ) : Prop :=
  (2 * x^2 + Real.sqrt 6)^3 = 19683

theorem find_number : ∃ x : ℝ, question x ∧ (x = Real.sqrt ((27 - Real.sqrt 6) / 2) ∨ x = -Real.sqrt ((27 - Real.sqrt 6) / 2)) :=
  sorry

end find_number_l274_274241


namespace determine_m_l274_274186

-- Define the conditions: the quadratic equation and the sum of roots
def quadratic_eq (x m : ℝ) : Prop :=
  x^2 + m * x + 2 = 0

def sum_of_roots (x1 x2 : ℝ) : ℝ := x1 + x2

-- Problem Statement: Prove that m = 4
theorem determine_m (x1 x2 m : ℝ) 
  (h1 : quadratic_eq x1 m) 
  (h2 : quadratic_eq x2 m)
  (h3 : sum_of_roots x1 x2 = -4) : 
  m = 4 :=
by
  sorry

end determine_m_l274_274186


namespace sufficient_but_not_necessary_condition_l274_274901

theorem sufficient_but_not_necessary_condition (x : ℝ) :
  (2 * x^2 + x - 1 ≥ 0) → (x ≥ 1/2) ∨ (x ≤ -1) :=
by
  -- The given inequality and condition imply this result.
  sorry

end sufficient_but_not_necessary_condition_l274_274901


namespace third_smallest_four_digit_in_pascals_triangle_l274_274228

def pascal (n k : ℕ) : ℕ :=
  if k = 0 ∨ k = n then 1
  else pascal (n - 1) (k - 1) + pascal (n - 1) k

theorem third_smallest_four_digit_in_pascals_triangle :
  ∃ n k, pascal n k = 1002 ∧ 
  ∀ m l, (pascal m l ≥ 1000 ∧ pascal m l < 1002) → (m = n ∧ l = k ∨ m ≠ n ∨ l ≠ k) :=
begin
  sorry
end

end third_smallest_four_digit_in_pascals_triangle_l274_274228


namespace planning_committee_ways_is_20_l274_274552

-- Define the number of students in the council
def num_students : ℕ := 6

-- Define the ways to choose a 3-person committee from num_students
def committee_ways (x : ℕ) : ℕ := Nat.choose x 3

-- Given condition: number of ways to choose the welcoming committee is 20
axiom welcoming_committee_condition : committee_ways num_students = 20

-- Statement to prove
theorem planning_committee_ways_is_20 : committee_ways num_students = 20 := by
  exact welcoming_committee_condition

end planning_committee_ways_is_20_l274_274552


namespace matrix_eq_sum_35_l274_274480

theorem matrix_eq_sum_35 (a b c d : ℤ) (h1 : 2 * a = 14 * a - 15 * b)
  (h2 : 2 * b = 9 * a - 10 * b)
  (h3 : 3 * c = 14 * c - 15 * d)
  (h4 : 3 * d = 9 * c - 10 * d) :
  a + b + c + d = 35 :=
sorry

end matrix_eq_sum_35_l274_274480


namespace smallest_hamburger_packages_l274_274269

theorem smallest_hamburger_packages (h_num : ℕ) (b_num : ℕ) (h_bag_num : h_num = 10) (b_bag_num : b_num = 15) :
  ∃ (n : ℕ), n = 3 ∧ (n * h_num) = (2 * b_num) := by
  sorry

end smallest_hamburger_packages_l274_274269


namespace jaydee_typing_speed_l274_274892

theorem jaydee_typing_speed (hours : ℕ) (total_words : ℕ) (minutes_per_hour : ℕ := 60) 
  (h1 : hours = 2) (h2 : total_words = 4560) : (total_words / (hours * minutes_per_hour) = 38) :=
by
  sorry

end jaydee_typing_speed_l274_274892


namespace minimum_xy_l274_274063

noncomputable def f (x y : ℝ) := 2 * x + y + 6

theorem minimum_xy (x y : ℝ) (h : 0 < x ∧ 0 < y) (h1 : f x y = x * y) : x * y = 18 :=
by
  sorry

end minimum_xy_l274_274063


namespace relationship_of_f_values_l274_274118

noncomputable def f : ℝ → ℝ := sorry  -- placeholder for the actual function 

def is_even (f : ℝ → ℝ) : Prop := ∀ x, f (x + 2) = f (-x + 2)

def is_increasing (f : ℝ → ℝ) (a b : ℝ) : Prop := a < b → f a < f b

theorem relationship_of_f_values (h1 : is_increasing f 0 2) (h2 : is_even f) :
  f (5/2) > f 1 ∧ f 1 > f (7/2) :=
sorry -- proof goes here

end relationship_of_f_values_l274_274118


namespace mosquito_distance_ratio_l274_274922

-- Definition of the clock problem conditions
structure ClockInsects where
  distance_from_center : ℕ
  initial_time : ℕ := 1

-- Prove the ratio of distances traveled by mosquito and fly over 12 hours
theorem mosquito_distance_ratio (c : ClockInsects) :
  let mosquito_distance := (83 : ℚ)/12
  let fly_distance := (73 : ℚ)/12
  mosquito_distance / fly_distance = 83 / 73 :=
by 
  sorry

end mosquito_distance_ratio_l274_274922


namespace percentage_seeds_germinated_l274_274108

/-- There were 300 seeds planted in the first plot and 200 seeds planted in the second plot. 
    30% of the seeds in the first plot germinated and 32% of the total seeds germinated.
    Prove that 35% of the seeds in the second plot germinated. -/
theorem percentage_seeds_germinated 
  (s1 s2 : ℕ) (p1 p2 t : ℚ)
  (h1 : s1 = 300) 
  (h2 : s2 = 200) 
  (h3 : p1 = 30) 
  (h4 : t = 32) 
  (h5 : 0.30 * s1 + p2 * s2 = 0.32 * (s1 + s2)) :
  p2 = 35 :=
by 
  -- Proof goes here
  sorry

end percentage_seeds_germinated_l274_274108


namespace probability_of_three_primes_l274_274796

-- Defining the range of integers
def range := {n : ℕ | 1 ≤ n ∧ n ≤ 30}

-- Function to check if a number is prime
def is_prime (n : ℕ) : Prop := n > 1 ∧ (∀ m : ℕ, m ∣ n → m = 1 ∨ m = n)

-- Defining the set of prime numbers from 1 to 30
def primes : set ℕ := {n | n ∈ range ∧ is_prime n}

-- Define binomial coefficient
def binomial (n k : ℕ) : ℕ := n.choose k

-- Define probability function
def probability_three_primes : ℚ :=
  binomial 10 3 / binomial 30 3

theorem probability_of_three_primes : 
  probability_three_primes = 6 / 203 :=
by sorry

end probability_of_three_primes_l274_274796


namespace range_of_f_at_most_7_l274_274335

theorem range_of_f_at_most_7 (f : ℤ × ℤ → ℝ)
  (H : ∀ (x y m n : ℤ), f (x + 3 * m - 2 * n, y - 4 * m + 5 * n) = f (x, y)) :
  ∃ (s : Finset ℝ), s.card ≤ 7 ∧ ∀ (a : ℤ × ℤ), f a ∈ s :=
sorry

end range_of_f_at_most_7_l274_274335


namespace simplified_value_l274_274463

-- Define the operation ∗
def operation (m n p q : ℚ) : ℚ :=
  m * p * (n / q)

-- Prove that the simplified value of 5/4 ∗ 6/2 is 60
theorem simplified_value : operation 5 4 6 2 = 60 :=
by
  sorry

end simplified_value_l274_274463


namespace S_eq_T_l274_274736

-- Define the sets S and T
def S : Set ℤ := {x | ∃ n : ℕ, x = 3 * n + 1}
def T : Set ℤ := {x | ∃ k : ℤ, x = 3 * k - 2}

-- Prove that S = T
theorem S_eq_T : S = T := 
by {
  sorry
}

end S_eq_T_l274_274736


namespace calculate_pow_zero_l274_274427

theorem calculate_pow_zero: (2023 - Real.pi) ≠ 0 → (2023 - Real.pi)^0 = 1 := by
  -- Proof
  sorry

end calculate_pow_zero_l274_274427


namespace third_smallest_four_digit_in_pascals_triangle_is_1002_l274_274206

theorem third_smallest_four_digit_in_pascals_triangle_is_1002 :
  ∃ n k, 1000 <= binomial n k ∧ binomial n k <= 9999 ∧ (third_smallest (λ n k, 1000 <= binomial n k ∧ binomial n k <= 9999) = 1002) :=
sorry

end third_smallest_four_digit_in_pascals_triangle_is_1002_l274_274206


namespace find_g3_l274_274357

noncomputable def g : ℝ → ℝ := sorry

theorem find_g3 (h : ∀ x : ℝ, g (3^x) + x * g (3^(-x)) = x) : g 3 = 1 :=
sorry

end find_g3_l274_274357


namespace max_number_of_kids_on_school_bus_l274_274661

-- Definitions based on the conditions from the problem
def totalRowsLowerDeck : ℕ := 15
def totalRowsUpperDeck : ℕ := 10
def capacityLowerDeckRow : ℕ := 5
def capacityUpperDeckRow : ℕ := 3
def reservedSeatsLowerDeck : ℕ := 10
def staffMembers : ℕ := 4

-- The total capacity of the lower and upper decks
def totalCapacityLowerDeck := totalRowsLowerDeck * capacityLowerDeckRow
def totalCapacityUpperDeck := totalRowsUpperDeck * capacityUpperDeckRow
def totalCapacity := totalCapacityLowerDeck + totalCapacityUpperDeck

-- The maximum number of different kids that can ride the bus
def maxKids := totalCapacity - reservedSeatsLowerDeck - staffMembers

theorem max_number_of_kids_on_school_bus : maxKids = 91 := 
by 
  -- Step-by-step proof not required for this task
  sorry

end max_number_of_kids_on_school_bus_l274_274661


namespace work_completion_time_l274_274812

theorem work_completion_time
  (W : ℝ) -- Total work
  (p_rate : ℝ := W / 40) -- p's work rate
  (q_rate : ℝ := W / 24) -- q's work rate
  (work_done_by_p_alone : ℝ := 8 * p_rate) -- Work done by p in first 8 days
  (remaining_work : ℝ := W - work_done_by_p_alone) -- Remaining work after 8 days
  (combined_rate : ℝ := p_rate + q_rate) -- Combined work rate of p and q
  (time_to_complete_remaining_work : ℝ := remaining_work / combined_rate) -- Time to complete remaining work
  : (8 + time_to_complete_remaining_work) = 20 :=
by
  sorry

end work_completion_time_l274_274812


namespace rational_sum_eq_neg2_l274_274882

theorem rational_sum_eq_neg2 (a b : ℚ) (h : |a + 6| + (b - 4)^2 = 0) : a + b = -2 :=
sorry

end rational_sum_eq_neg2_l274_274882


namespace multiply_polynomials_l274_274906

theorem multiply_polynomials (x : ℝ) : (x^4 + 50 * x^2 + 625) * (x^2 - 25) = x^6 - 15625 := by
  sorry

end multiply_polynomials_l274_274906


namespace range_of_a_l274_274085

-- Function definition for op
def op (x y : ℝ) : ℝ := x * (2 - y)

-- Predicate that checks the inequality for all t
def inequality_holds_for_all_t (a : ℝ) : Prop :=
  ∀ t : ℝ, (op (t - a) (t + a)) < 1

-- Prove that the range of a is (0, 2)
theorem range_of_a : 
  ∀ a : ℝ, inequality_holds_for_all_t a ↔ 0 < a ∧ a < 2 := 
by
  sorry

end range_of_a_l274_274085


namespace sum_x_y_eq_two_l274_274900

theorem sum_x_y_eq_two (x y : ℝ) 
  (h1 : (x-1)^3 + 2003*(x-1) = -1) 
  (h2 : (y-1)^3 + 2003*(y-1) = 1) : 
  x + y = 2 :=
by
  sorry

end sum_x_y_eq_two_l274_274900


namespace kaleb_boxes_required_l274_274477

/-- Kaleb's Games Packing Problem -/
theorem kaleb_boxes_required (initial_games sold_games box_capacity : ℕ) (h1 : initial_games = 76) (h2 : sold_games = 46) (h3 : box_capacity = 5) :
  ((initial_games - sold_games) / box_capacity) = 6 :=
by
  -- Skipping the proof
  sorry

end kaleb_boxes_required_l274_274477


namespace largest_multiple_of_11_lt_neg150_l274_274527

theorem largest_multiple_of_11_lt_neg150 : ∃ (x : ℤ), (x % 11 = 0) ∧ (x < -150) ∧ (∀ y : ℤ, y % 11 = 0 → y < -150 → y ≤ x) ∧ x = -154 :=
by
  sorry

end largest_multiple_of_11_lt_neg150_l274_274527


namespace abs_eq_zero_iff_l274_274137

theorem abs_eq_zero_iff {a : ℝ} (h : |a + 3| = 0) : a = -3 :=
sorry

end abs_eq_zero_iff_l274_274137


namespace fraction_of_male_gerbils_is_correct_l274_274558

def total_pets := 90
def total_gerbils := 66
def total_hamsters := total_pets - total_gerbils
def fraction_hamsters_male := 1/3
def total_males := 25
def male_hamsters := fraction_hamsters_male * total_hamsters
def male_gerbils := total_males - male_hamsters
def fraction_gerbils_male := male_gerbils / total_gerbils

theorem fraction_of_male_gerbils_is_correct : fraction_gerbils_male = 17 / 66 := by
  sorry

end fraction_of_male_gerbils_is_correct_l274_274558


namespace max_hardcover_books_l274_274467

-- Define the conditions as provided in the problem
def total_books : ℕ := 36
def is_composite (n : ℕ) : Prop := 
  ∃ a b : ℕ, 2 ≤ a ∧ 2 ≤ b ∧ a * b = n

-- The logical statement we need to prove
theorem max_hardcover_books :
  ∃ h : ℕ, (∃ c : ℕ, is_composite c ∧ 2 * h + c = total_books) ∧ 
  ∀ h' c', is_composite c' ∧ 2 * h' + c' = total_books → h' ≤ h :=
sorry

end max_hardcover_books_l274_274467


namespace thickness_of_layer_l274_274415

noncomputable def volume_of_sphere (r : ℝ) : ℝ := (4/3) * Real.pi * r^3

noncomputable def volume_of_cylinder (r h : ℝ) : ℝ := Real.pi * r^2 * h

theorem thickness_of_layer (radius_sphere radius_cylinder : ℝ) (volume_sphere volume_cylinder : ℝ) (h : ℝ) : 
  radius_sphere = 3 → 
  radius_cylinder = 10 →
  volume_sphere = volume_of_sphere radius_sphere →
  volume_cylinder = volume_of_cylinder radius_cylinder h →
  volume_sphere = volume_cylinder → 
  h = 9 / 25 :=
by
  intros
  sorry

end thickness_of_layer_l274_274415


namespace third_smallest_four_digit_in_pascals_triangle_l274_274215

theorem third_smallest_four_digit_in_pascals_triangle :
  ∃ n k, n > 0 ∧ k > 0 ∧ Nat.choose n k = 1035 ∧ 
  (∀ (m l : ℕ), m > 0 ∧ l > 0 ∧ Nat.choose m l < 1000 → Nat.choose m l < 1035) ∧
  (∀ (p q : ℕ), p > 0 ∧ q > 0 ∧ 1000 ≤ Nat.choose p q ∧ Nat.choose p q < 1035 → 
    (Nat.choose m l = 1000 ∨ Nat.choose m l = 1001 ∨ Nat.choose m l = 1035)) :=
sorry

end third_smallest_four_digit_in_pascals_triangle_l274_274215


namespace problem_counts_correct_pairs_l274_274578

noncomputable def count_valid_pairs : ℝ :=
  sorry

theorem problem_counts_correct_pairs :
  count_valid_pairs = 128 :=
by
  sorry

end problem_counts_correct_pairs_l274_274578


namespace single_elimination_game_count_l274_274968

theorem single_elimination_game_count (n : Nat) (h : n = 23) : n - 1 = 22 :=
by
  sorry

end single_elimination_game_count_l274_274968


namespace tan_15_eq_sqrt3_l274_274247

theorem tan_15_eq_sqrt3 :
  (1 + Real.tan (Real.pi / 12)) / (1 - Real.tan (Real.pi / 12)) = Real.sqrt 3 :=
sorry

end tan_15_eq_sqrt3_l274_274247


namespace last_digit_2_pow_1000_last_digit_3_pow_1000_last_digit_7_pow_1000_l274_274027

-- Define the cycle period used in the problem
def cycle_period_2 := [2, 4, 8, 6]
def cycle_period_3 := [3, 9, 7, 1]
def cycle_period_7 := [7, 9, 3, 1]

-- Define a function to get the last digit from the cycle for given n
def last_digit_from_cycle (cycle : List ℕ) (n : ℕ) : ℕ :=
  let cycle_length := cycle.length
  cycle.get! ((n % cycle_length) - 1)

-- Problem statements
theorem last_digit_2_pow_1000 : last_digit_from_cycle cycle_period_2 1000 = 6 := sorry
theorem last_digit_3_pow_1000 : last_digit_from_cycle cycle_period_3 1000 = 1 := sorry
theorem last_digit_7_pow_1000 : last_digit_from_cycle cycle_period_7 1000 = 1 := sorry

end last_digit_2_pow_1000_last_digit_3_pow_1000_last_digit_7_pow_1000_l274_274027


namespace third_smallest_four_digit_Pascal_triangle_l274_274222

theorem third_smallest_four_digit_Pascal_triangle : 
  ∃ n : ℕ, (binomial (n + 2) (n + 2) = 1002) ∧ ∀ k : ℕ, (k < (n + 2) → binomial (k + 2) (k + 2) < 1002) :=
by
  sorry

end third_smallest_four_digit_Pascal_triangle_l274_274222


namespace weight_of_person_replaced_l274_274748

def initial_total_weight (W : ℝ) : ℝ := W
def new_person_weight : ℝ := 137
def average_increase : ℝ := 7.2
def group_size : ℕ := 10

theorem weight_of_person_replaced 
(W : ℝ) 
(weight_replaced : ℝ) 
(h1 : (W / group_size) + average_increase = (W - weight_replaced + new_person_weight) / group_size) : 
weight_replaced = 65 := 
sorry

end weight_of_person_replaced_l274_274748


namespace trivia_team_points_l274_274688

theorem trivia_team_points : 
  let member1_points := 8
  let member2_points := 12
  let member3_points := 9
  let member4_points := 5
  let member5_points := 10
  let member6_points := 7
  let member7_points := 14
  let member8_points := 11
  (member1_points + member2_points + member3_points + member4_points + member5_points + member6_points + member7_points + member8_points) = 76 :=
by
  let member1_points := 8
  let member2_points := 12
  let member3_points := 9
  let member4_points := 5
  let member5_points := 10
  let member6_points := 7
  let member7_points := 14
  let member8_points := 11
  sorry

end trivia_team_points_l274_274688


namespace youngest_sibling_age_l274_274232

theorem youngest_sibling_age
  (Y : ℕ)
  (h1 : Y + (Y + 3) + (Y + 6) + (Y + 7) = 120) :
  Y = 26 :=
by
  -- proof steps would be here 
  sorry

end youngest_sibling_age_l274_274232


namespace fundraiser_contribution_l274_274002

theorem fundraiser_contribution :
  let sasha_muffins := 30
  let melissa_muffins := 4 * sasha_muffins
  let tiffany_muffins := (sasha_muffins + melissa_muffins) / 2
  let total_muffins := sasha_muffins + melissa_muffins + tiffany_muffins
  let price_per_muffin := 4
  total_muffins * price_per_muffin = 900 :=
by
  let sasha_muffins := 30
  let melissa_muffins := 4 * sasha_muffins
  let tiffany_muffins := (sasha_muffins + melissa_muffins) / 2
  let total_muffins := sasha_muffins + melissa_muffins + tiffany_muffins
  let price_per_muffin := 4
  sorry

end fundraiser_contribution_l274_274002


namespace stickers_per_page_l274_274514

theorem stickers_per_page (total_pages total_stickers : ℕ) (h1 : total_pages = 22) (h2 : total_stickers = 220) : (total_stickers / total_pages) = 10 :=
by
  sorry

end stickers_per_page_l274_274514


namespace total_spent_on_birthday_presents_l274_274160

noncomputable def leonards_total_before_discount :=
  (3 * 35.50) + (2 * 120.75) + 44.25

noncomputable def leonards_total_after_discount :=
  leonards_total_before_discount - (0.10 * leonards_total_before_discount)

noncomputable def michaels_total_before_discount :=
  89.50 + (3 * 54.50) + 24.75

noncomputable def michaels_total_after_discount :=
  michaels_total_before_discount - (0.15 * michaels_total_before_discount)

noncomputable def emilys_total_before_tax :=
  (2 * 69.25) + (4 * 14.80)

noncomputable def emilys_total_after_tax :=
  emilys_total_before_tax + (0.08 * emilys_total_before_tax)

noncomputable def total_amount_spent :=
  leonards_total_after_discount + michaels_total_after_discount + emilys_total_after_tax

theorem total_spent_on_birthday_presents :
  total_amount_spent = 802.64 :=
by
  sorry

end total_spent_on_birthday_presents_l274_274160


namespace income_in_scientific_notation_l274_274073

theorem income_in_scientific_notation :
  10870 = 1.087 * 10^4 := 
sorry

end income_in_scientific_notation_l274_274073


namespace range_of_x_l274_274786

noncomputable def range_of_independent_variable (x : ℝ) : Prop :=
  1 - x > 0

theorem range_of_x (x : ℝ) : range_of_independent_variable x → x < 1 :=
by sorry

end range_of_x_l274_274786


namespace area_excluding_garden_proof_l274_274469

noncomputable def area_land_excluding_garden (length width r : ℝ) : ℝ :=
  let area_rec := length * width
  let area_circle := Real.pi * (r ^ 2)
  area_rec - area_circle

theorem area_excluding_garden_proof :
  area_land_excluding_garden 8 12 3 = 96 - 9 * Real.pi :=
by
  unfold area_land_excluding_garden
  sorry

end area_excluding_garden_proof_l274_274469


namespace value_of_a_l274_274600

theorem value_of_a (a : ℕ) : (∃ (x1 x2 x3 : ℤ),
  abs (abs (x1 - 3) - 1) = a ∧
  abs (abs (x2 - 3) - 1) = a ∧
  abs (abs (x3 - 3) - 1) = a ∧
  x1 ≠ x2 ∧ x1 ≠ x3 ∧ x2 ≠ x3)
  → a = 1 :=
by
  sorry

end value_of_a_l274_274600


namespace sum_of_first_1000_terms_l274_274084

def sequence_block_sum (n : ℕ) : ℕ :=
  1 + 3 * n

def sequence_sum_up_to (k : ℕ) : ℕ :=
  if k = 0 then 0 else (1 + 3 * (k * (k - 1) / 2)) + k

def nth_term_position (n : ℕ) : ℕ :=
  n * (n + 1) / 2 + n

theorem sum_of_first_1000_terms : sequence_sum_up_to 43 + (1000 - nth_term_position 43) * 3 = 2912 :=
sorry

end sum_of_first_1000_terms_l274_274084


namespace ratio_of_45_and_9_l274_274663

theorem ratio_of_45_and_9 : (45 / 9) = 5 := 
by
  sorry

end ratio_of_45_and_9_l274_274663


namespace arun_weight_average_l274_274671

theorem arun_weight_average :
  (∀ w : ℝ, 65 < w ∧ w < 72 → 60 < w ∧ w < 70 → w ≤ 68 → 66 ≤ w ∧ w ≤ 69 → 64 ≤ w ∧ w ≤ 67.5 → 
    (66.75 = (66 + 67.5) / 2)) := by
  sorry

end arun_weight_average_l274_274671


namespace third_smallest_four_digit_in_pascals_triangle_l274_274193

-- First we declare the conditions
def every_positive_integer_in_pascals_triangle : Prop :=
  ∀ n : ℕ, ∃ a b : ℕ, nat.choose b a = n

def number_1000_in_pascals_triangle : Prop :=
  ∃ b : ℕ, nat.choose b 1 = 1000

def number_1001_in_pascals_triangle : Prop :=
  ∃ b : ℕ, nat.choose b 1 = 1001

def number_1002_in_pascals_triangle : Prop :=
  ∃ b : ℕ, nat.choose b 2 = 1002

-- Define the theorem
theorem third_smallest_four_digit_in_pascals_triangle
  (h1 : every_positive_integer_in_pascals_triangle)
  (h2 : number_1000_in_pascals_triangle)
  (h3 : number_1001_in_pascals_triangle)
  (h4 : number_1002_in_pascals_triangle) :
  (∃ n : ℕ, nat.choose n 2 = 1002 ∧ (∀ k, k < n → nat.choose k 1 < 1002)) := sorry

end third_smallest_four_digit_in_pascals_triangle_l274_274193


namespace tickets_used_l274_274557

variable (C T : Nat)

theorem tickets_used (h1 : C = 7) (h2 : T = C + 5) : T = 12 := by
  sorry

end tickets_used_l274_274557


namespace find_b_l274_274059

theorem find_b (a b : ℕ) (h1 : a = 105) (h2 : a ^ 3 = 21 * 25 * 15 * b) : b = 147 :=
sorry

end find_b_l274_274059


namespace cuboid_dimensions_l274_274804

-- Define the problem conditions and the goal
theorem cuboid_dimensions (x y v : ℕ) :
  (v * (x * y - 1) = 602) ∧ (x * (v * y - 1) = 605) →
  v = x + 3 →
  x = 11 ∧ y = 4 ∧ v = 14 :=
by
  sorry

end cuboid_dimensions_l274_274804


namespace not_perfect_square_l274_274814

theorem not_perfect_square (n : ℕ) (h : 0 < n) : ¬ ∃ k : ℕ, k * k = 2551 * 543^n - 2008 * 7^n :=
by
  sorry

end not_perfect_square_l274_274814


namespace value_of_a_l274_274602

theorem value_of_a (a b : ℝ) (h1 : b = 4 * a) (h2 : b = 24 - 4 * a) : a = 3 :=
by
  sorry

end value_of_a_l274_274602


namespace buffy_breath_time_l274_274334

theorem buffy_breath_time (k : ℕ) (b : ℕ) (f : ℕ) 
  (h1 : k = 3 * 60) 
  (h2 : b = k - 20) 
  (h3 : f = b - 40) :
  f = 120 :=
by {
  sorry
}

end buffy_breath_time_l274_274334


namespace sin2_cos3_tan4_lt_zero_l274_274080

theorem sin2_cos3_tan4_lt_zero (h1 : Real.sin 2 > 0) (h2 : Real.cos 3 < 0) (h3 : Real.tan 4 > 0) : Real.sin 2 * Real.cos 3 * Real.tan 4 < 0 :=
sorry

end sin2_cos3_tan4_lt_zero_l274_274080


namespace jafaris_candy_l274_274992

-- Define the conditions
variable (candy_total : Nat)
variable (taquon_candy : Nat)
variable (mack_candy : Nat)

-- Assume the conditions from the problem
axiom candy_total_def : candy_total = 418
axiom taquon_candy_def : taquon_candy = 171
axiom mack_candy_def : mack_candy = 171

-- Define the statement to be proved
theorem jafaris_candy : (candy_total - (taquon_candy + mack_candy)) = 76 :=
by
  -- Proof goes here
  sorry

end jafaris_candy_l274_274992


namespace find_perpendicular_line_l274_274101

theorem find_perpendicular_line (x y : ℝ) (h₁ : y = (1/2) * x + 1)
    (h₂ : (x, y) = (2, 0)) : y = -2 * x + 4 :=
sorry

end find_perpendicular_line_l274_274101


namespace jason_messages_l274_274476

theorem jason_messages :
  ∃ M : ℕ, (M + M / 2 + 150) / 5 = 96 ∧ M = 220 := by
  sorry

end jason_messages_l274_274476


namespace height_inequality_triangle_l274_274899

theorem height_inequality_triangle (a b c h_a h_b h_c Δ : ℝ) (n : ℝ) 
  (ha : h_a = 2 * Δ / a)
  (hb : h_b = 2 * Δ / b)
  (hc : h_c = 2 * Δ / c)
  (n_pos : n > 0) :
  (a * h_b)^n + (b * h_c)^n + (c * h_a)^n ≥ 3 * 2^n * Δ^n := 
sorry

end height_inequality_triangle_l274_274899


namespace least_positive_integer_condition_l274_274029

theorem least_positive_integer_condition
  (a : ℤ) (ha1 : a % 4 = 1) (ha2 : a % 5 = 2) (ha3 : a % 6 = 3) :
  a > 0 → a = 57 :=
by
  intro ha_pos
  -- Proof omitted for brevity
  sorry

end least_positive_integer_condition_l274_274029


namespace least_positive_integer_l274_274041

theorem least_positive_integer (n : ℕ) : 
  (n % 4 = 1) ∧ (n % 5 = 2) ∧ (n % 6 = 3) → n = 57 := by
sorry

end least_positive_integer_l274_274041


namespace trapezium_area_l274_274096

theorem trapezium_area (a b h : ℝ) (ha : a = 20) (hb : b = 18) (hh : h = 15) :
  (1/2) * (a + b) * h = 285 := by
  sorry

end trapezium_area_l274_274096


namespace first_problem_second_problem_l274_274263

variable (x : ℝ)

-- Proof for the first problem
theorem first_problem : 6 * x^3 / (-3 * x^2) = -2 * x := by
sorry

-- Proof for the second problem
theorem second_problem : (2 * x + 3) * (2 * x - 3) - 4 * (x - 2)^2 = 16 * x - 25 := by
sorry

end first_problem_second_problem_l274_274263


namespace suzanne_donation_l274_274013

theorem suzanne_donation :
  let base_donation := 10
  let total_distance := 5
  let total_donation := (List.range total_distance).foldl (fun acc km => acc + base_donation * 2 ^ km) 0
  total_donation = 310 :=
by
  let base_donation := 10
  let total_distance := 5
  let total_donation := (List.range total_distance).foldl (fun acc km => acc + base_donation * 2 ^ km) 0
  sorry

end suzanne_donation_l274_274013


namespace find_number_l274_274674

theorem find_number (x : ℝ) (h : (x / 6) * 12 = 15) : x = 7.5 :=
sorry

end find_number_l274_274674


namespace quadratic_real_roots_opposite_signs_l274_274016

theorem quadratic_real_roots_opposite_signs (c : ℝ) : 
  (c < 0 → (∃ x1 x2 : ℝ, x1 * x2 = c ∧ x1 + x2 = -1 ∧ x1 ≠ x2 ∧ (x1 < 0 ∧ x2 > 0 ∨ x1 > 0 ∧ x2 < 0))) ∧ 
  (∃ x1 x2 : ℝ, x1 * x2 = c ∧ x1 + x2 = -1 ∧ x1 ≠ x2 ∧ (x1 < 0 ∧ x2 > 0 ∨ x1 > 0 ∧ x2 < 0) → c < 0) :=
by 
  sorry

end quadratic_real_roots_opposite_signs_l274_274016


namespace john_remaining_amount_l274_274616

theorem john_remaining_amount (initial_amount games: ℕ) (food souvenirs: ℕ) :
  initial_amount = 100 →
  games = 20 →
  food = 3 * games →
  souvenirs = (1 / 2 : ℚ) * games →
  initial_amount - (games + food + souvenirs) = 10 :=
by
  sorry

end john_remaining_amount_l274_274616


namespace magnitude_of_a_l274_274296

open Real

-- Assuming the standard inner product space for vectors in Euclidean space

variables (a b : ℝ) -- Vectors in R^n (could be general but simplified to real numbers for this example)
variable (θ : ℝ)    -- Angle between vectors
axiom angle_ab : θ = 60 -- Given angle between vectors

-- Conditions:
axiom non_zero_a : a ≠ 0
axiom non_zero_b : b ≠ 0
axiom norm_b : abs b = 1
axiom norm_2a_minus_b : abs (2 * a - b) = 1

-- To prove:
theorem magnitude_of_a : abs a = 1 / 2 :=
sorry

end magnitude_of_a_l274_274296


namespace reciprocal_neg_two_l274_274653

theorem reciprocal_neg_two : 1 / (-2) = - (1 / 2) :=
by
  sorry

end reciprocal_neg_two_l274_274653


namespace profit_ratio_l274_274061

theorem profit_ratio (P_invest Q_invest : ℕ) (hP : P_invest = 500000) (hQ : Q_invest = 1000000) :
  (P_invest:ℚ) / Q_invest = 1 / 2 := 
  by
  rw [hP, hQ]
  norm_num

end profit_ratio_l274_274061


namespace competitive_exam_candidates_l274_274948

theorem competitive_exam_candidates (x : ℝ)
  (A_selected : ℝ := 0.06 * x) 
  (B_selected : ℝ := 0.07 * x) 
  (h : B_selected = A_selected + 81) :
  x = 8100 := by
  sorry

end competitive_exam_candidates_l274_274948


namespace mouse_lives_count_l274_274680

-- Define the basic conditions
def catLives : ℕ := 9
def dogLives : ℕ := catLives - 3
def mouseLives : ℕ := dogLives + 7

-- The main theorem to prove
theorem mouse_lives_count : mouseLives = 13 :=
by
  -- proof steps go here
  sorry

end mouse_lives_count_l274_274680


namespace solution_set_of_inequality_l274_274290

theorem solution_set_of_inequality (a : ℝ) (h : 0 < a) :
  {x : ℝ | x ^ 2 - 4 * a * x - 5 * a ^ 2 < 0} = {x : ℝ | -a < x ∧ x < 5 * a} :=
sorry

end solution_set_of_inequality_l274_274290


namespace fraction_simplification_l274_274700

theorem fraction_simplification :
  (3 / 7 + 5 / 8 + 2 / 9) / (5 / 12 + 1 / 4) = 643 / 336 :=
by
  sorry

end fraction_simplification_l274_274700


namespace cats_weight_more_than_puppies_l274_274727

theorem cats_weight_more_than_puppies :
  let num_puppies := 4
  let weight_per_puppy := 7.5
  let num_cats := 14
  let weight_per_cat := 2.5
  (num_cats * weight_per_cat) - (num_puppies * weight_per_puppy) = 5 :=
by 
  let num_puppies := 4
  let weight_per_puppy := 7.5
  let num_cats := 14
  let weight_per_cat := 2.5
  sorry

end cats_weight_more_than_puppies_l274_274727


namespace therapist_charge_l274_274238

theorem therapist_charge (F A : ℝ) 
    (h1 : F + 4 * A = 400)
    (h2 : F + 2 * A = 252) : F - A = 30 := 
by 
    sorry

end therapist_charge_l274_274238


namespace least_common_multiple_l274_274942

theorem least_common_multiple (x : ℕ) (hx : x > 0) : 
  lcm (1 / (4 * x)) (lcm (1 / (6 * x)) (1 / (9 * x))) = 1 / (36 * x) :=
by
  sorry

end least_common_multiple_l274_274942


namespace roots_polynomial_identity_l274_274166

theorem roots_polynomial_identity (a b c : ℝ) (h1 : a + b + c = 15) (h2 : a * b + b * c + c * a = 22) (h3 : a * b * c = 8) :
  (2 + a) * (2 + b) * (2 + c) = 120 :=
by
  sorry

end roots_polynomial_identity_l274_274166


namespace cats_weigh_more_than_puppies_l274_274729

theorem cats_weigh_more_than_puppies :
  let puppies_weight := 4 * 7.5
  let cats_weight := 14 * 2.5
  cats_weight - puppies_weight = 5 :=
by
  let puppies_weight := 4 * 7.5
  let cats_weight := 14 * 2.5
  show cats_weight - puppies_weight = 5 from sorry

end cats_weigh_more_than_puppies_l274_274729


namespace simplify_evaluate_expr_l274_274005

theorem simplify_evaluate_expr (x y : ℚ) (h₁ : x = -1) (h₂ : y = -1 / 2) :
  (4 * x * y + (2 * x^2 + 5 * x * y - y^2) - 2 * (x^2 + 3 * x * y)) = 5 / 4 :=
by
  rw [h₁, h₂]
  -- Here we would include the specific algebra steps to convert the LHS to 5/4.
  sorry

end simplify_evaluate_expr_l274_274005


namespace part1_part2_l274_274590

noncomputable def f (x : ℝ) : ℝ := x * Real.log x

theorem part1 (m : ℝ) : (∃ x, deriv f x = 2 ∧ f x = 2 * x + m) → m = -Real.exp 1 :=
sorry

theorem part2 : ∀ x > 0, -1 / Real.exp 1 ≤ f x ∧ f x < Real.exp x / (2 * x) :=
sorry

end part1_part2_l274_274590


namespace solve_trig_system_l274_274774

open Real

theorem solve_trig_system (x y : ℝ) (k : ℤ) (hx : tan x * tan y = 1/6) (hy : sin x * sin y = 1/(5 * sqrt 2)) :
  (x + y = ±(π / 4) + 2 * π * k) ∧ (x - y = ±(arccos ((7 / (5 * sqrt 2))) + 2 * π * k)) :=
sorry

end solve_trig_system_l274_274774


namespace third_smallest_four_digit_in_pascal_triangle_l274_274213

-- Defining Pascal's triangle
def pascal (n k : ℕ) : ℕ := if k = 0 ∨ k = n then 1 else pascal (n - 1) (k - 1) + pascal (n - 1) k

/--
Condition:
1. In Pascal's triangle, each number is the sum of the two numbers directly above it.
2. Four-digit numbers start appearing regularly in higher rows (from row 45 onwards).
-/
theorem third_smallest_four_digit_in_pascal_triangle : 
  ∃ n k : ℕ, pascal n k = 1002 ∧ ∀ m l : ℕ, (pascal m l < 1000 ∨ pascal m l = 1001) → (m < n ∨ (m = n ∧ l < k)) :=
sorry

end third_smallest_four_digit_in_pascal_triangle_l274_274213


namespace integer_solutions_count_l274_274309

theorem integer_solutions_count :
  let eq : Int -> Int -> Int := fun x y => 6 * y ^ 2 + 3 * x * y + x + 2 * y - 72
  ∃ (sols : List (Int × Int)), 
    (∀ x y, eq x y = 0 → (x, y) ∈ sols) ∧
    (∀ p ∈ sols, ∃ x y, p = (x, y) ∧ eq x y = 0) ∧
    sols.length = 4 :=
by
  sorry

end integer_solutions_count_l274_274309


namespace eliminating_y_l274_274860

theorem eliminating_y (x y : ℝ) (h1 : y = x + 3) (h2 : 2 * x - y = 5) : 2 * x - x - 3 = 5 :=
by {
  sorry
}

end eliminating_y_l274_274860


namespace exists_x_y_not_divisible_by_3_l274_274895

theorem exists_x_y_not_divisible_by_3 (k : ℕ) (h_pos : 0 < k) :
  ∃ x y : ℤ, (x^2 + 2 * y^2 = 3^k) ∧ (x % 3 ≠ 0) ∧ (y % 3 ≠ 0) := 
sorry

end exists_x_y_not_divisible_by_3_l274_274895


namespace three_distinct_real_solutions_l274_274737

theorem three_distinct_real_solutions (b c : ℝ):
  (∀ x : ℝ, x^2 + b * |x| + c = 0 → x = 0) ∧ (∃! x : ℝ, x^2 + b * |x| + c = 0) →
  b < 0 ∧ c = 0 :=
by {
  sorry
}

end three_distinct_real_solutions_l274_274737


namespace Jungkook_has_most_apples_l274_274057

-- Conditions
def Yoongi_apples : ℕ := 4
def Jungkook_apples_initial : ℕ := 6
def Jungkook_apples_additional : ℕ := 3
def Jungkook_total_apples : ℕ := Jungkook_apples_initial + Jungkook_apples_additional
def Yuna_apples : ℕ := 5

-- Statement (to prove)
theorem Jungkook_has_most_apples : Jungkook_total_apples > Yoongi_apples ∧ Jungkook_total_apples > Yuna_apples := by
  sorry

end Jungkook_has_most_apples_l274_274057


namespace find_S_2013_l274_274151

variable {a : ℕ → ℤ} -- the arithmetic sequence
variable {S : ℕ → ℤ} -- the sum of the first n terms

-- Conditions
axiom a1_eq_neg2011 : a 1 = -2011
axiom sum_sequence : ∀ n, S n = (n * (a 1 + a n)) / 2
axiom condition_eq : (S 2012 / 2012) - (S 2011 / 2011) = 1

-- The Lean statement to prove that S 2013 = 2013
theorem find_S_2013 : S 2013 = 2013 := by
  sorry

end find_S_2013_l274_274151


namespace fraction_evaluation_l274_274973

theorem fraction_evaluation :
  (7 / 18 * (9 / 2) + 1 / 6) / ((40 / 3) - (15 / 4) / (5 / 16)) * (23 / 8) =
  4 + 17 / 128 :=
by
  -- conditions based on mixed number simplification
  have h1 : 4 + 1 / 2 = (9 : ℚ) / 2 := by sorry
  have h2 : 13 + 1 / 3 = (40 : ℚ) / 3 := by sorry
  have h3 : 3 + 3 / 4 = (15 : ℚ) / 4 := by sorry
  have h4 : 2 + 7 / 8 = (23 : ℚ) / 8 := by sorry
  -- the main proof
  sorry

end fraction_evaluation_l274_274973


namespace inequality_part_1_inequality_part_2_l274_274303

noncomputable def f (x : ℝ) := |x - 2| + 2
noncomputable def g (x : ℝ) (m : ℝ) := m * |x|

theorem inequality_part_1 (x : ℝ) : f x > 5 ↔ x < -1 ∨ x > 5 := by
  sorry

theorem inequality_part_2 (m : ℝ) : (∀ x, f x ≥ g x m) ↔ m ≤ 1 := by
  sorry

end inequality_part_1_inequality_part_2_l274_274303


namespace find_older_friend_age_l274_274355

theorem find_older_friend_age (A B C : ℕ) 
  (h1 : A - B = 2) 
  (h2 : A - C = 5) 
  (h3 : A + B + C = 110) : 
  A = 39 := 
by 
  sorry

end find_older_friend_age_l274_274355


namespace gcd_1458_1479_l274_274979

def a : ℕ := 1458
def b : ℕ := 1479
def gcd_ab : ℕ := 21

theorem gcd_1458_1479 : Nat.gcd a b = gcd_ab := sorry

end gcd_1458_1479_l274_274979


namespace total_cuts_length_eq_60_l274_274686

noncomputable def total_length_of_cuts (side_length : ℝ) (num_rectangles : ℕ) : ℝ :=
  if side_length = 36 ∧ num_rectangles = 3 then 60 else 0

theorem total_cuts_length_eq_60 :
  ∀ (side_length : ℝ) (num_rectangles : ℕ),
    side_length = 36 ∧ num_rectangles = 3 →
    total_length_of_cuts side_length num_rectangles = 60 := by
  intros
  simp [total_length_of_cuts]
  sorry

end total_cuts_length_eq_60_l274_274686


namespace solution_set_of_inequality_l274_274957

theorem solution_set_of_inequality (x : ℝ) : 
  |x + 3| - |x - 2| ≥ 3 ↔ x ≥ 1 :=
sorry

end solution_set_of_inequality_l274_274957


namespace third_smallest_four_digit_in_pascals_triangle_l274_274201

theorem third_smallest_four_digit_in_pascals_triangle :
  ∃ n m k, binom n m = 1002 ∧ binom n (m + 1) = 1001 ∧ binom n (m + 2) = 1000 :=
sorry

end third_smallest_four_digit_in_pascals_triangle_l274_274201


namespace find_a_l274_274327

theorem find_a : (a : ℕ) = 103 * 97 * 10009 → a = 99999919 := by
  intro h
  sorry

end find_a_l274_274327


namespace B_investment_amount_l274_274240

-- Definitions based on given conditions
variable (A_investment : ℕ := 300) -- A's investment in dollars
variable (B_investment : ℕ)        -- B's investment in dollars
variable (A_time : ℕ := 12)        -- Time A's investment was in the business in months
variable (B_time : ℕ := 6)         -- Time B's investment was in the business in months
variable (profit : ℕ := 100)       -- Total profit in dollars
variable (A_share : ℕ := 75)       -- A's share of the profit in dollars

-- The mathematically equivalent proof problem to prove that B invested $200
theorem B_investment_amount (h : A_share * (A_investment * A_time + B_investment * B_time) / profit = A_investment * A_time) : 
  B_investment = 200 := by
  sorry

end B_investment_amount_l274_274240


namespace number_of_people_needed_to_lift_car_l274_274891

-- Define the conditions as Lean definitions
def twice_as_many_people_to_lift_truck (C T : ℕ) : Prop :=
  T = 2 * C

def people_needed_for_cars_and_trucks (C T total_people : ℕ) : Prop :=
  60 = 6 * C + 3 * T

-- Define the theorem statement using the conditions
theorem number_of_people_needed_to_lift_car :
  ∃ C, (∃ T, twice_as_many_people_to_lift_truck C T) ∧ people_needed_for_cars_and_trucks C T 60 ∧ C = 5 :=
sorry

end number_of_people_needed_to_lift_car_l274_274891


namespace jia_winning_strategy_l274_274243

variables {p q : ℝ}
def is_quadratic_real_roots (a b c : ℝ) : Prop := b ^ 2 - 4 * a * c > 0

def quadratic_with_roots (x1 x2 : ℝ) :=
  x1 > 0 ∧ x2 > 0 ∧ x1 ≠ x2 ∧ is_quadratic_real_roots 1 (- (x1 + x2)) (x1 * x2)

def modify_jia (p q x1 : ℝ) : (ℝ × ℝ) := (p + 1, q - x1)

def modify_yi1 (p q : ℝ) : (ℝ × ℝ) := (p - 1, q)

def modify_yi2 (p q x2 : ℝ) : (ℝ × ℝ) := (p - 1, q + x2)

def winning_strategy_jia (x1 x2 : ℝ) : Prop :=
  ∃ n : ℕ, ∀ m ≥ n, ∀ p q, quadratic_with_roots x1 x2 → 
  (¬ is_quadratic_real_roots 1 p q) ∨ (q ≤ 0)

theorem jia_winning_strategy (x1 x2 : ℝ)
  (h: quadratic_with_roots x1 x2) : 
  winning_strategy_jia x1 x2 :=
sorry

end jia_winning_strategy_l274_274243


namespace smallest_other_integer_l274_274017

-- Definitions of conditions
def gcd_condition (a b : ℕ) (x : ℕ) : Prop := 
  Nat.gcd a b = x + 5

def lcm_condition (a b : ℕ) (x : ℕ) : Prop := 
  Nat.lcm a b = x * (x + 5)

def sum_condition (a b : ℕ) : Prop := 
  a + b < 100

-- Main statement incorporating all conditions
theorem smallest_other_integer {x b : ℕ} (hx_pos : x > 0)
  (h_gcd : gcd_condition 45 b x)
  (h_lcm : lcm_condition 45 b x)
  (h_sum : sum_condition 45 b) :
  b = 12 :=
sorry

end smallest_other_integer_l274_274017


namespace complex_expression_identity_l274_274762

noncomputable section

variable (x y : ℂ) 

theorem complex_expression_identity (hx : x ≠ 0) (hy : y ≠ 0) (hxy : x^2 + x*y + y^2 = 0) : 
  (x / (x + y)) ^ 1990 + (y / (x + y)) ^ 1990 = -1 := 
by 
  sorry

end complex_expression_identity_l274_274762


namespace g_of_g_of_2_l274_274596

def g (x : ℝ) : ℝ := 4 * x^2 - 3

theorem g_of_g_of_2 : g (g 2) = 673 := 
by 
  sorry

end g_of_g_of_2_l274_274596


namespace vika_card_pairs_l274_274375

theorem vika_card_pairs : 
  let numbers := finset.range 61 \ finset.singleton 0 in
  let divs := {d | d ∈ finset.divisors 30} in
  numbers.card = 60 →
  ∀ d ∈ divs, ∀ pair : finset (ℕ × ℕ),
    pair.card = 30 →
    finset.forall₂ pair (λ x y, |x.1 - x.2| % d = |y.1 - y.2| % d) → 
    ∃ (number_of_pairs : ℕ), number_of_pairs = 8 :=
by 
  intro numbers divs hc hd hp hpairs,
  sorry

end vika_card_pairs_l274_274375


namespace min_cubes_l274_274270

-- Define the conditions as properties
structure FigureViews :=
  (front_view : ℕ)
  (side_view : ℕ)
  (top_view : ℕ)
  (adjacency_requirement : Bool)

-- Define the given views
def given_views : FigureViews := {
  front_view := 3,  -- as described: 2 cubes at bottom + 1 on top
  side_view := 3,   -- same as front view
  top_view := 3,    -- L-shape consists of 3 cubes
  adjacency_requirement := true
}

-- The theorem to state that the minimum number of cubes is 3
theorem min_cubes (views : FigureViews) : views.front_view = 3 ∧ views.side_view = 3 ∧ views.top_view = 3 ∧ views.adjacency_requirement = true → ∃ n, n = 3 :=
by {
  sorry
}

end min_cubes_l274_274270


namespace union_of_A_and_B_intersection_of_complement_A_and_B_l274_274128

-- Define sets A and B
def A : Set ℝ := {x | 3 ≤ x ∧ x ≤ 7}
def B : Set ℝ := {x | 3 < 2 * x - 1 ∧ 2 * x - 1 < 19}

-- Define the universal set here, which encompass all real numbers
def universal_set : Set ℝ := {x | true}

-- Define the complement of A with respect to the real numbers
def C_R (S : Set ℝ) : Set ℝ := {x | x ∉ S}

-- Prove that A ∪ B is {x | 2 < x < 10}
theorem union_of_A_and_B : A ∪ B = {x | 2 < x ∧ x < 10} := by
  sorry

-- Prove that (C_R A) ∩ B is {x | 2 < x < 3 ∨ 7 < x < 10}
theorem intersection_of_complement_A_and_B : (C_R A) ∪ B = {x | (2 < x ∧ x < 3) ∨ (7 < x ∧ x < 10)} := by
  sorry

end union_of_A_and_B_intersection_of_complement_A_and_B_l274_274128


namespace kitchen_width_l274_274170

theorem kitchen_width (length : ℕ) (height : ℕ) (rate : ℕ) (hours : ℕ) (coats : ℕ) 
  (total_painted : ℕ) (half_walls_area : ℕ) (total_walls_area : ℕ)
  (width : ℕ) : 
  length = 12 ∧ height = 10 ∧ rate = 40 ∧ hours = 42 ∧ coats = 3 ∧ 
  total_painted = rate * hours ∧ total_painted = coats * total_walls_area ∧
  half_walls_area = 2 * length * height ∧ total_walls_area = half_walls_area + 2 * width * height ∧
  2 * (total_walls_area - half_walls_area / 2) = 2 * width * height →
  width = 16 := 
by
  sorry

end kitchen_width_l274_274170


namespace SplitWinnings_l274_274675

noncomputable def IstvanInitialContribution : ℕ := 5000 * 20
noncomputable def IstvanSecondPeriodContribution : ℕ := (5000 + 4000) * 30
noncomputable def IstvanThirdPeriodContribution : ℕ := (5000 + 4000 - 2500) * 40
noncomputable def IstvanTotalContribution : ℕ := IstvanInitialContribution + IstvanSecondPeriodContribution + IstvanThirdPeriodContribution

noncomputable def KalmanContribution : ℕ := 4000 * 70
noncomputable def LaszloContribution : ℕ := 2500 * 40
noncomputable def MiklosContributionAdjustment : ℕ := 2000 * 90

noncomputable def IstvanExpectedShare : ℕ := IstvanTotalContribution * 12 / 100
noncomputable def KalmanExpectedShare : ℕ := KalmanContribution * 12 / 100
noncomputable def LaszloExpectedShare : ℕ := LaszloContribution * 12 / 100
noncomputable def MiklosExpectedShare : ℕ := MiklosContributionAdjustment * 12 / 100

noncomputable def IstvanActualShare : ℕ := IstvanExpectedShare * 7 / 8
noncomputable def KalmanActualShare : ℕ := (KalmanExpectedShare - MiklosExpectedShare) * 7 / 8
noncomputable def LaszloActualShare : ℕ := LaszloExpectedShare * 7 / 8
noncomputable def MiklosActualShare : ℕ := MiklosExpectedShare * 7 / 8

theorem SplitWinnings :
  IstvanActualShare = 54600 ∧ KalmanActualShare = 7800 ∧ LaszloActualShare = 10500 ∧ MiklosActualShare = 18900 :=
by
  sorry

end SplitWinnings_l274_274675


namespace probability_exactly_3_divisible_by_3_of_7_fair_12_sided_dice_l274_274690
-- Import all necessary libraries

-- Define the conditions as variables
variable (n k : ℕ) (p q : ℚ)
variable (dice_divisible_by_3_prob : ℚ)
variable (dice_not_divisible_by_3_prob : ℚ)

-- Assign values based on the problem statement
noncomputable def cond_replicate_n_fair_12_sided_dice := n = 7
noncomputable def cond_exactly_k_divisible_by_3 := k = 3
noncomputable def cond_prob_divisible_by_3 := dice_divisible_by_3_prob = 1 / 3
noncomputable def cond_prob_not_divisible_by_3 := dice_not_divisible_by_3_prob = 2 / 3

-- The theorem statement with the final answer incorporated
theorem probability_exactly_3_divisible_by_3_of_7_fair_12_sided_dice :
  cond_replicate_n_fair_12_sided_dice n →
  cond_exactly_k_divisible_by_3 k →
  cond_prob_divisible_by_3 dice_divisible_by_3_prob →
  cond_prob_not_divisible_by_3 dice_not_divisible_by_3_prob →
  p = (35 : ℚ) * ((1 / 3) ^ 3) * ((2 / 3) ^ 4) →
  q = (560 / 2187 : ℚ) →
  p = q :=
by
  intros
  sorry

end probability_exactly_3_divisible_by_3_of_7_fair_12_sided_dice_l274_274690


namespace range_of_a_l274_274591

noncomputable def f (a x : ℝ) : ℝ := a / (x + 1) + Real.log x

theorem range_of_a (a : ℝ) :
  (∀ (x₁ x₂ : ℝ), x₁ ∈ Ioc 0 2 → x₂ ∈ Ioc 0 2 → x₁ ≠ x₂ → (f a x₂ - f a x₁) / (x₂ - x₁) > -1) ↔ a ≤ 27 / 4 :=
sorry

end range_of_a_l274_274591


namespace kenny_pieces_used_l274_274995

-- Definitions based on conditions
def mushrooms_cut := 22
def pieces_per_mushroom := 4
def karla_pieces := 42
def remaining_pieces := 8
def total_pieces := mushrooms_cut * pieces_per_mushroom

-- Theorem to be proved
theorem kenny_pieces_used :
  total_pieces - (karla_pieces + remaining_pieces) = 38 := 
by 
  sorry

end kenny_pieces_used_l274_274995


namespace number_of_sons_l274_274821

noncomputable def land_area_hectares : ℕ := 3
noncomputable def hectare_to_m2 : ℕ := 10000
noncomputable def profit_per_section_per_3months : ℕ := 500
noncomputable def section_area_m2 : ℕ := 750
noncomputable def profit_per_son_per_year : ℕ := 10000
noncomputable def months_in_year : ℕ := 12
noncomputable def months_per_season : ℕ := 3

theorem number_of_sons :
  let total_land_area_m2 := land_area_hectares * hectare_to_m2
  let yearly_profit_per_section := profit_per_section_per_3months * (months_in_year / months_per_season)
  let number_of_sections := total_land_area_m2 / section_area_m2
  let total_yearly_profit := number_of_sections * yearly_profit_per_section
  let n := total_yearly_profit / profit_per_son_per_year
  n = 8 :=
by
  sorry

end number_of_sons_l274_274821


namespace value_of_expression_l274_274105

theorem value_of_expression (x : ℝ) (h : 7 * x^2 - 2 * x - 4 = 4 * x + 11) : 
  (5 * x - 7)^2 = 11.63265306 := 
by 
  sorry

end value_of_expression_l274_274105


namespace sum_of_midpoints_l274_274792

theorem sum_of_midpoints (d e f : ℝ) (h : d + e + f = 15) :
  (d + e) / 2 + (d + f) / 2 + (e + f) / 2 = 15 :=
by sorry

end sum_of_midpoints_l274_274792


namespace simplify_and_evaluate_expr_l274_274174

-- Define x
def x : ℝ := Real.sqrt 2 - 1

-- Define the expression
def expr : ℝ := (1 - (x + 1) / x) / ((x^2 - 1) / (x^2 - x))

-- State the theorem which asserts the equality
theorem simplify_and_evaluate_expr : expr = -Real.sqrt 2 / 2 := 
by 
  sorry

end simplify_and_evaluate_expr_l274_274174


namespace dan_marbles_l274_274976

theorem dan_marbles (original_marbles : ℕ) (given_marbles : ℕ) (remaining_marbles : ℕ) :
  original_marbles = 128 →
  given_marbles = 32 →
  remaining_marbles = original_marbles - given_marbles →
  remaining_marbles = 96 :=
by
  intros h1 h2 h3
  rw [h1, h2] at h3
  exact h3

end dan_marbles_l274_274976


namespace number_of_arrangements_l274_274937

theorem number_of_arrangements (V T : ℕ) (hV : V = 3) (hT : T = 4) :
  ∃ n : ℕ, n = 36 :=
by
  sorry

end number_of_arrangements_l274_274937


namespace no_valid_angles_l274_274087

open Real

theorem no_valid_angles (θ : ℝ) (h1 : 0 < θ) (h2 : θ < 2 * π)
    (h3 : ∀ k : ℤ, θ ≠ k * (π / 2))
    (h4 : cos θ * tan θ = sin θ ^ 3) : false :=
by
  -- The proof goes here
  sorry

end no_valid_angles_l274_274087


namespace book_problem_part1_book_problem_part2_l274_274534

variables (costA costB : ℝ) (x y W : ℝ)

theorem book_problem_part1 (h1 : costA = 1.5 * costB) 
  (h2 : 540 / costA + 3 = 450 / costB) :
  costA = 45 ∧ costB = 30 :=
by
  sorry

theorem book_problem_part2 (costA costB : ℝ) (x y : ℝ)
  (h1 : costA = 1.5 * costB)
  (h2 : 540 / costA + 3 = 450 / costB)
  (h3 : x + y = 50)
  (h4 : x ≥ y + 6)
  (h5 : W = costA * x + costB * y) :
  x = 28 ∧ y = 22 ∧ W = 1920 :=
by
  sorry

end book_problem_part1_book_problem_part2_l274_274534


namespace holes_in_compartment_l274_274236

theorem holes_in_compartment :
  ∀ (rect : Type) (holes : ℕ) (compartments : ℕ),
  compartments = 9 →
  holes = 20 →
  (∃ (compartment : rect ) (n : ℕ), n ≥ 3) :=
by
  intros rect holes compartments h_compartments h_holes
  sorry

end holes_in_compartment_l274_274236


namespace unique_three_digit_numbers_unique_three_digit_odd_numbers_l274_274444

def num_digits: ℕ := 10

theorem unique_three_digit_numbers (num_digits : ℕ) :
  ∃ n : ℕ, n = 648 ∧ n = (num_digits - 1) * (num_digits - 1) * (num_digits - 2) + 2 * (num_digits - 1) * (num_digits - 1) :=
  sorry

theorem unique_three_digit_odd_numbers (num_digits : ℕ) :
  ∃ n : ℕ, n = 320 ∧ ∀ odd_digit_nums : ℕ, odd_digit_nums ≥ 1 → odd_digit_nums = 5 → 
  n = odd_digit_nums * (num_digits - 2) * (num_digits - 2) :=
  sorry

end unique_three_digit_numbers_unique_three_digit_odd_numbers_l274_274444


namespace village_foods_sales_l274_274800

-- Definitions based on conditions
def customer_count : Nat := 500
def lettuce_per_customer : Nat := 2
def tomato_per_customer : Nat := 4
def price_per_lettuce : Nat := 1
def price_per_tomato : Nat := 1 / 2 -- Note: Handling decimal requires careful type choice

-- Main statement to prove
theorem village_foods_sales : 
  customer_count * (lettuce_per_customer * price_per_lettuce + tomato_per_customer * price_per_tomato) = 2000 := 
by
  sorry

end village_foods_sales_l274_274800


namespace sequences_identity_l274_274914

variables {α β γ : ℤ}
variables {a b : ℕ → ℤ}

-- Define the recurrence relations conditions
def conditions (a b : ℕ → ℤ) (α β γ : ℤ) : Prop :=
  a 0 = 1 ∧ b 0 = 1 ∧
  (∀ n, a (n + 1) = α * a n + β * b n) ∧
  (∀ n, b (n + 1) = β * a n + γ * b n) ∧
  α < γ ∧ α * γ = β^2 + 1

-- Define the main statement
theorem sequences_identity (a b : ℕ → ℤ) 
  (h : conditions a b α β γ) (m n : ℕ) :
  a (m + n) + b (m + n) = a m * a n + b m * b n :=
sorry

end sequences_identity_l274_274914


namespace limit_calculation_l274_274447

noncomputable def f (x : ℝ) : ℝ := Real.exp (-x)

theorem limit_calculation :
  (Real.exp (-1) * Real.exp 0 - Real.exp (-1) * Real.exp 0) / 0 = -3 / Real.exp 1 := by
  sorry

end limit_calculation_l274_274447


namespace brads_zip_code_l274_274835

theorem brads_zip_code (A B C D E : ℕ) (h1 : A + B + C + D + E = 20)
                        (h2 : B = A + 1) (h3 : C = A)
                        (h4 : D = 2 * A) (h5 : D + E = 13)
                        (h6 : Nat.Prime (A*10000 + B*1000 + C*100 + D*10 + E)) :
                        A*10000 + B*1000 + C*100 + D*10 + E = 34367 := 
sorry

end brads_zip_code_l274_274835


namespace part_I_part_II_part_III_l274_274302

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x * Real.log x - (1 / 2) * a * x^2

-- Part (Ⅰ)
theorem part_I (x : ℝ) : (0 < x) → (f 1 x < f 1 (x+1)) := sorry

-- Part (Ⅱ)
theorem part_II (f_has_two_distinct_extreme_values : ∀ a : ℝ, (∃ x y : ℝ, x ≠ y ∧ (f a x = f a y))) : 0 < a ∧ a < 1 := sorry

-- Part (Ⅲ)
theorem part_III (f_has_two_distinct_zeros : ∀ a : ℝ, (∃ x y : ℝ, x ≠ y ∧ f a x = 0 ∧ f a y = 0)) : 0 < a ∧ a < (2 / Real.exp 1) := sorry

end part_I_part_II_part_III_l274_274302


namespace smallest_x_l274_274391

open Classical
noncomputable theory

def conditions (x : ℕ) : Prop :=
  x % 3 = 2 ∧ x % 4 = 3 ∧ x % 5 = 4

theorem smallest_x : ∃ (x : ℕ), conditions x ∧ (∀ (y : ℕ), conditions y → x ≤ y) ∧ x = 59 :=
by {
  sorry
}

end smallest_x_l274_274391


namespace solution_set_of_inequality_l274_274865

noncomputable def f : ℝ → ℝ := sorry

axiom ax1 : ∀ (x1 x2 : ℝ), (0 < x1) → (0 < x2) → (x1 ≠ x2) → 
  (x1 * f x2 - x2 * f x1) / (x2 - x1) > 1

axiom ax2 : f 3 = 2

theorem solution_set_of_inequality :
  {x : ℝ | 0 < x ∧ f x < x - 1} = {x : ℝ | 0 < x ∧ x < 3} := by
  sorry

end solution_set_of_inequality_l274_274865


namespace gcd_condition_implies_equality_l274_274894

theorem gcd_condition_implies_equality (a b : ℤ) (h : ∀ n : ℤ, n ≥ 1 → Int.gcd (a + n) (b + n) > 1) : a = b :=
sorry

end gcd_condition_implies_equality_l274_274894


namespace mul_72516_9999_l274_274945

theorem mul_72516_9999 : 72516 * 9999 = 724787484 :=
by
  sorry

end mul_72516_9999_l274_274945


namespace orthocentric_tetrahedron_equivalence_l274_274396

def isOrthocentricTetrahedron 
  (sums_of_squares_of_opposite_edges_equal : Prop) 
  (products_of_cosines_of_opposite_dihedral_angles_equal : Prop)
  (angles_between_opposite_edges_equal : Prop) : Prop :=
  sums_of_squares_of_opposite_edges_equal ∨
  products_of_cosines_of_opposite_dihedral_angles_equal ∨
  angles_between_opposite_edges_equal

theorem orthocentric_tetrahedron_equivalence
  (sums_of_squares_of_opposite_edges_equal 
    products_of_cosines_of_opposite_dihedral_angles_equal
    angles_between_opposite_edges_equal : Prop) :
  isOrthocentricTetrahedron
    sums_of_squares_of_opposite_edges_equal
    products_of_cosines_of_opposite_dihedral_angles_equal
    angles_between_opposite_edges_equal :=
sorry

end orthocentric_tetrahedron_equivalence_l274_274396


namespace smallest_x_l274_274392

theorem smallest_x (x : ℕ) (h₁ : x % 3 = 2) (h₂ : x % 4 = 3) (h₃ : x % 5 = 4) : x = 59 :=
by
  sorry

end smallest_x_l274_274392


namespace remainder_r15_minus_1_l274_274856

theorem remainder_r15_minus_1 (r : ℝ) : 
    (r^15 - 1) % (r - 1) = 0 :=
sorry

end remainder_r15_minus_1_l274_274856


namespace third_smallest_four_digit_in_pascals_triangle_l274_274226

def pascal (n k : ℕ) : ℕ :=
  if k = 0 ∨ k = n then 1
  else pascal (n - 1) (k - 1) + pascal (n - 1) k

theorem third_smallest_four_digit_in_pascals_triangle :
  ∃ n k, pascal n k = 1002 ∧ 
  ∀ m l, (pascal m l ≥ 1000 ∧ pascal m l < 1002) → (m = n ∧ l = k ∨ m ≠ n ∨ l ≠ k) :=
begin
  sorry
end

end third_smallest_four_digit_in_pascals_triangle_l274_274226


namespace undefined_values_of_fraction_l274_274711

theorem undefined_values_of_fraction (b : ℝ) : b^2 - 9 = 0 ↔ b = 3 ∨ b = -3 := by
  sorry

end undefined_values_of_fraction_l274_274711


namespace karthik_weight_average_l274_274758

theorem karthik_weight_average
  (weight : ℝ)
  (hKarthik: 55 < weight )
  (hBrother: weight < 58 )
  (hFather : 56 < weight )
  (hSister: 54 < weight ∧ weight < 57) :
  (56 < weight ∧ weight < 57) → (weight = 56.5) :=
by 
  sorry

end karthik_weight_average_l274_274758


namespace third_smallest_four_digit_in_pascals_triangle_l274_274227

def pascal (n k : ℕ) : ℕ :=
  if k = 0 ∨ k = n then 1
  else pascal (n - 1) (k - 1) + pascal (n - 1) k

theorem third_smallest_four_digit_in_pascals_triangle :
  ∃ n k, pascal n k = 1002 ∧ 
  ∀ m l, (pascal m l ≥ 1000 ∧ pascal m l < 1002) → (m = n ∧ l = k ∨ m ≠ n ∨ l ≠ k) :=
begin
  sorry
end

end third_smallest_four_digit_in_pascals_triangle_l274_274227


namespace average_speed_is_37_5_l274_274959

-- Define the conditions
def distance_local : ℕ := 60
def speed_local : ℕ := 30
def distance_gravel : ℕ := 10
def speed_gravel : ℕ := 20
def distance_highway : ℕ := 105
def speed_highway : ℕ := 60
def traffic_delay : ℚ := 15 / 60
def obstruction_delay : ℚ := 10 / 60

-- Define the total distance
def total_distance : ℕ := distance_local + distance_gravel + distance_highway

-- Define the total time
def total_time : ℚ :=
  (distance_local / speed_local) +
  (distance_gravel / speed_gravel) +
  (distance_highway / speed_highway) +
  traffic_delay +
  obstruction_delay

-- Define the average speed as distance divided by time
def average_speed : ℚ := total_distance / total_time

theorem average_speed_is_37_5 :
  average_speed = 37.5 := by sorry

end average_speed_is_37_5_l274_274959


namespace whitney_money_left_over_l274_274810

def total_cost (posters_cost : ℝ) (notebooks_cost : ℝ) (bookmarks_cost : ℝ) (pencils_cost : ℝ) (tax_rate : ℝ) :=
  let pre_tax := (3 * posters_cost) + (4 * notebooks_cost) + (5 * bookmarks_cost) + (2 * pencils_cost)
  let tax := pre_tax * tax_rate
  pre_tax + tax

def money_left_over (initial_money : ℝ) (total_cost : ℝ) :=
  initial_money - total_cost

theorem whitney_money_left_over :
  let initial_money := 40
  let posters_cost := 7.50
  let notebooks_cost := 5.25
  let bookmarks_cost := 3.10
  let pencils_cost := 1.15
  let tax_rate := 0.08
  money_left_over initial_money (total_cost posters_cost notebooks_cost bookmarks_cost pencils_cost tax_rate) = -26.20 :=
by
  sorry

end whitney_money_left_over_l274_274810


namespace range_of_a_l274_274121

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a^x - 1

noncomputable def g (a : ℝ) (x : ℝ) : ℝ := f a (x + 1) - 4

theorem range_of_a (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) (h3 : f a 1 > 1) (h4 : ∀ x, g a x ≤ 0 → ¬(x < 0 ∧ g a x > 0)) :
  2 < a ∧ a ≤ 5 :=
by
  sorry

end range_of_a_l274_274121


namespace towel_area_decrease_28_percent_l274_274071

variable (L B : ℝ)

def original_area (L B : ℝ) : ℝ := L * B

def new_length (L : ℝ) : ℝ := L * 0.80

def new_breadth (B : ℝ) : ℝ := B * 0.90

def new_area (L B : ℝ) : ℝ := (new_length L) * (new_breadth B)

def percentage_decrease_in_area (L B : ℝ) : ℝ :=
  ((original_area L B - new_area L B) / original_area L B) * 100

theorem towel_area_decrease_28_percent (L B : ℝ) :
  percentage_decrease_in_area L B = 28 := by
  sorry

end towel_area_decrease_28_percent_l274_274071


namespace eval_expression_solve_inequalities_l274_274955

-- Problem 1: Evaluation of the expression equals sqrt(2)
theorem eval_expression : (1 - 1^2023 + Real.sqrt 9 - (Real.pi - 3)^0 + |Real.sqrt 2 - 1|) = Real.sqrt 2 := 
by sorry

-- Problem 2: Solution set of the inequality system
theorem solve_inequalities (x : ℝ) : 
  ((3 * x + 1) / 2 ≥ (4 * x + 3) / 3 ∧ 2 * x + 7 ≥ 5 * x - 17) ↔ (3 ≤ x ∧ x ≤ 8) :=
by sorry

end eval_expression_solve_inequalities_l274_274955


namespace village_foods_sales_l274_274801

-- Definitions based on conditions
def customer_count : Nat := 500
def lettuce_per_customer : Nat := 2
def tomato_per_customer : Nat := 4
def price_per_lettuce : Nat := 1
def price_per_tomato : Nat := 1 / 2 -- Note: Handling decimal requires careful type choice

-- Main statement to prove
theorem village_foods_sales : 
  customer_count * (lettuce_per_customer * price_per_lettuce + tomato_per_customer * price_per_tomato) = 2000 := 
by
  sorry

end village_foods_sales_l274_274801


namespace inequality_transformation_l274_274864

theorem inequality_transformation (x : ℝ) (n : ℕ) (hn : 0 < n) (hx : 0 < x) : 
  x + (n^n) / (x^n) ≥ n + 1 := 
sorry

end inequality_transformation_l274_274864


namespace angle_B_value_value_of_k_l274_274153

variable {A B C a b c : ℝ}
variable {k : ℝ}
variable {m n : ℝ × ℝ}

theorem angle_B_value
  (h1 : (2 * a - c) * Real.cos B = b * Real.cos C) :
  B = Real.pi / 3 :=
by sorry

theorem value_of_k
  (hA : 0 < A ∧ A < 2 * Real.pi / 3)
  (hm : m = (Real.sin A, Real.cos (2 * A)))
  (hn : n = (4 * k, 1))
  (hM : 4 * k * Real.sin A + Real.cos (2 * A) = 7) :
  k = 2 :=
by sorry

end angle_B_value_value_of_k_l274_274153


namespace negation_of_universal_proposition_l274_274183

theorem negation_of_universal_proposition :
  ¬ (∀ x : ℝ, x^2 + 2*x + 2 > 0) ↔ ∃ x : ℝ, x^2 + 2*x + 2 ≤ 0 :=
by sorry

end negation_of_universal_proposition_l274_274183


namespace probability_one_hits_l274_274522

theorem probability_one_hits 
  (p_A : ℚ) (p_B : ℚ)
  (hA : p_A = 1 / 2) (hB : p_B = 1 / 3):
  p_A * (1 - p_B) + (1 - p_A) * p_B = 1 / 2 := by
  sorry

end probability_one_hits_l274_274522


namespace third_smallest_four_digit_in_pascal_l274_274203

def is_four_digit (n : ℕ) : Prop := 1000 ≤ n ∧ n < 10000

def pascal (n k : ℕ) : ℕ := Nat.choose n k

theorem third_smallest_four_digit_in_pascal :
  ∃ n k : ℕ, is_four_digit (pascal n k) ∧ (pascal n k = 1002) :=
sorry

end third_smallest_four_digit_in_pascal_l274_274203


namespace smallest_x_l274_274390

open Classical
noncomputable theory

def conditions (x : ℕ) : Prop :=
  x % 3 = 2 ∧ x % 4 = 3 ∧ x % 5 = 4

theorem smallest_x : ∃ (x : ℕ), conditions x ∧ (∀ (y : ℕ), conditions y → x ≤ y) ∧ x = 59 :=
by {
  sorry
}

end smallest_x_l274_274390


namespace inequality_proof_l274_274449

theorem inequality_proof (x y : ℝ) (hx : 0 < x) (hy : 0 < y) : x + y ≤ (y^2 / x) + (x^2 / y) :=
sorry

end inequality_proof_l274_274449


namespace ounces_per_cup_l274_274091

theorem ounces_per_cup (total_ounces : ℕ) (total_cups : ℕ) 
  (h : total_ounces = 264 ∧ total_cups = 33) : total_ounces / total_cups = 8 :=
by
  sorry

end ounces_per_cup_l274_274091


namespace present_age_ratio_l274_274934

-- Define the conditions as functions in Lean.
def age_difference (M R : ℝ) : Prop := M - R = 7.5
def future_age_ratio (M R : ℝ) : Prop := (R + 10) / (M + 10) = 2 / 3

-- Define the goal as a proof problem in Lean.
theorem present_age_ratio (M R : ℝ) 
  (h1 : age_difference M R) 
  (h2 : future_age_ratio M R) : 
  R / M = 2 / 5 := 
by 
  sorry  -- Proof to be completed

end present_age_ratio_l274_274934


namespace slope_of_line_determined_by_solutions_eq_l274_274383

theorem slope_of_line_determined_by_solutions_eq :
  ∀ (x y : ℝ), (4 / x + 5 / y = 0) → ∃ m : ℝ, m = -5 / 4 :=
by
  intro x y h
  use -5 / 4
  sorry

end slope_of_line_determined_by_solutions_eq_l274_274383


namespace no_integer_solution_l274_274575

theorem no_integer_solution (a b : ℤ) : ¬ (3 * a ^ 2 = b ^ 2 + 1) :=
by
  -- Proof omitted
  sorry

end no_integer_solution_l274_274575


namespace symmetric_curve_eq_l274_274435

-- Define the original curve equation and line of symmetry
def original_curve (x y : ℝ) : Prop := y^2 = 4 * x
def line_of_symmetry (x : ℝ) : Prop := x = 2

-- The equivalent Lean 4 statement
theorem symmetric_curve_eq (x y : ℝ) (hx : line_of_symmetry 2) :
  (∀ (x' y' : ℝ), original_curve (4 - x') y' → y^2 = 16 - 4 * x) :=
sorry

end symmetric_curve_eq_l274_274435


namespace find_a4_l274_274716

variable (a : ℕ → ℝ)
variable (q : ℝ)
variable (T_7 : ℝ)

-- Conditions
axiom geom_seq (n : ℕ) : a (n + 1) = q * a n
axiom common_ratio_ne_one : q ≠ 1
axiom product_first_seven_terms : (a 1) * (a 2) * (a 3) * (a 4) * (a 5) * (a 6) * (a 7) = 128

-- Goal
theorem find_a4 : a 4 = 2 :=
sorry

end find_a4_l274_274716


namespace age_of_youngest_child_l274_274024

theorem age_of_youngest_child (x : ℕ) 
  (h : x + (x + 2) + (x + 4) + (x + 6) + (x + 8) = 50) : x = 6 :=
sorry

end age_of_youngest_child_l274_274024


namespace pascal_third_smallest_four_digit_number_l274_274209

theorem pascal_third_smallest_four_digit_number :
  (∃ n k: ℕ, 1000 ≤ binom n k ∧ binom n k ≤ 1002) ∧ ¬(∃ n k: ℕ, 1003 ≤ binom n k) :=
sorry

end pascal_third_smallest_four_digit_number_l274_274209


namespace chucks_team_final_score_l274_274610

variable (RedTeamScore : ℕ) (scoreDifference : ℕ)

-- Given conditions
def red_team_score := RedTeamScore = 76
def score_difference := scoreDifference = 19

-- Question: What was the final score of Chuck's team?
def chucks_team_score (RedTeamScore scoreDifference : ℕ) : ℕ := 
  RedTeamScore + scoreDifference

-- Proof statement
theorem chucks_team_final_score : red_team_score 76 ∧ score_difference 19 → chucks_team_score 76 19 = 95 :=
by
  sorry

end chucks_team_final_score_l274_274610


namespace village_food_sales_l274_274803

theorem village_food_sales :
  ∀ (customers_per_month heads_of_lettuce_per_person tomatoes_per_person 
      price_per_head_of_lettuce price_per_tomato : ℕ) 
    (H1 : customers_per_month = 500)
    (H2 : heads_of_lettuce_per_person = 2)
    (H3 : tomatoes_per_person = 4)
    (H4 : price_per_head_of_lettuce = 1)
    (H5 : price_per_tomato = 1 / 2), 
  customers_per_month * ((heads_of_lettuce_per_person * price_per_head_of_lettuce) 
    + (tomatoes_per_person * (price_per_tomato : ℝ))) = 2000 := 
by 
  intros customers_per_month heads_of_lettuce_per_person tomatoes_per_person 
         price_per_head_of_lettuce price_per_tomato 
         H1 H2 H3 H4 H5
  sorry

end village_food_sales_l274_274803


namespace ellipse_minimum_distance_point_l274_274570

theorem ellipse_minimum_distance_point :
  ∃ (x y : ℝ), (x^2 / 16 + y^2 / 12 = 1) ∧ (∀ p, x - 2 * y - 12 = 0 → dist (x, y) p ≥ dist (2, -3) p) :=
sorry

end ellipse_minimum_distance_point_l274_274570


namespace problem_solution_l274_274419

def is_quadratic (y : ℝ → ℝ) : Prop :=
  ∃ (a b c : ℝ), ∀ x, y x = a * x^2 + b * x + c

def not_quadratic_func := 
  let yA := fun x => -2 * x^2
  let yB := fun x => 2 * (x - 1)^2 + 1
  let yC := fun x => (x - 3)^2 - x^2
  let yD := fun a => a * (8 - a)
  (¬ is_quadratic yC) ∧ (is_quadratic yA) ∧ (is_quadratic yB) ∧ (is_quadratic yD)

theorem problem_solution : not_quadratic_func := 
sorry

end problem_solution_l274_274419


namespace Darla_electricity_bill_l274_274561

theorem Darla_electricity_bill :
  let tier1_rate := 4
  let tier2_rate := 3.5
  let tier3_rate := 3
  let tier1_limit := 300
  let tier2_limit := 500
  let late_fee1 := 150
  let late_fee2 := 200
  let late_fee3 := 250
  let consumption := 1200
  let cost_tier1 := tier1_limit * tier1_rate
  let cost_ttier2 := tier2_limit * tier2_rate
  let cost_tier3 := (consumption - (tier1_limit + tier2_limit)) * tier3_rate
  let total_cost := cost_tier1 + cost_tier2 + cost_tier3
  let late_fee := late_fee3
  let final_cost := total_cost + late_fee
  final_cost = 4400 :=
by
  sorry

end Darla_electricity_bill_l274_274561


namespace greatest_possible_remainder_l274_274459

theorem greatest_possible_remainder (x : ℕ) : ∃ r, r < 9 ∧ x % 9 = r ∧ r = 8 :=
by
  use 8
  sorry -- Proof to be filled in

end greatest_possible_remainder_l274_274459


namespace solve_equation_l274_274572

theorem solve_equation (x : ℝ) :
  (1 / (x^2 + 17 * x - 8) + 1 / (x^2 + 4 * x - 8) + 1 / (x^2 - 9 * x - 8) = 0) →
  (x = 1 ∨ x = -8 ∨ x = 2 ∨ x = -4) :=
by
  sorry

end solve_equation_l274_274572


namespace third_smallest_four_digit_in_pascals_triangle_l274_274219

-- Definitions for Pascal's Triangle and four-digit numbers
def is_in_pascals_triangle (n : ℕ) : Prop :=
  ∃ (r : ℕ) (k : ℕ), r.choose k = n

def is_four_digit (n : ℕ) : Prop :=
  1000 ≤ n ∧ n < 10000

-- Proposition stating the third smallest four-digit number in Pascal's Triangle
theorem third_smallest_four_digit_in_pascals_triangle :
  ∃ (n : ℕ), is_in_pascals_triangle n ∧ is_four_digit n ∧ 
  (∃ m1 m2 m3, is_in_pascals_triangle m1 ∧ is_four_digit m1 ∧ 
                is_in_pascals_triangle m2 ∧ is_four_digit m2 ∧ 
                is_in_pascals_triangle m3 ∧ is_four_digit m3 ∧ 
                1000 ≤ m1 ∧ m1 < 1001 ∧ 1001 ≤ m2 ∧ m2 < 1002 ∧ 1002 ≤ n ∧ n < 1003) ∧ 
  n = 1002 :=
sorry

end third_smallest_four_digit_in_pascals_triangle_l274_274219


namespace freshman_class_count_l274_274014

theorem freshman_class_count : ∃ n : ℤ, n < 500 ∧ n % 25 = 24 ∧ n % 19 = 11 ∧ n = 49 := by
  sorry

end freshman_class_count_l274_274014


namespace bobs_sisters_mile_time_l274_274253

theorem bobs_sisters_mile_time (bobs_current_time_minutes : ℕ) (bobs_current_time_seconds : ℕ) (improvement_percentage : ℝ) :
  bobs_current_time_minutes = 10 → bobs_current_time_seconds = 40 → improvement_percentage = 9.062499999999996 →
  bobs_sisters_time_minutes = 9 ∧ bobs_sisters_time_seconds = 42 :=
by
  -- Definitions from conditions
  let bobs_time_in_seconds := bobs_current_time_minutes * 60 + bobs_current_time_seconds
  let improvement_in_seconds := bobs_time_in_seconds * improvement_percentage / 100
  let target_time_in_seconds := bobs_time_in_seconds - improvement_in_seconds
  let bobs_sisters_time_minutes := target_time_in_seconds / 60
  let bobs_sisters_time_seconds := target_time_in_seconds % 60
  
  sorry

end bobs_sisters_mile_time_l274_274253


namespace cats_weigh_more_by_5_kg_l274_274725

def puppies_weight (num_puppies : ℕ) (weight_per_puppy : ℝ) : ℝ :=
  num_puppies * weight_per_puppy

def cats_weight (num_cats : ℕ) (weight_per_cat : ℝ) : ℝ :=
  num_cats * weight_per_cat

theorem cats_weigh_more_by_5_kg :
  puppies_weight 4 7.5  = 30 ∧ cats_weight 14 2.5 = 35 → (cats_weight 14 2.5 - puppies_weight 4 7.5 = 5) := 
by
  intro h
  sorry

end cats_weigh_more_by_5_kg_l274_274725


namespace leak_empty_tank_time_l274_274772

-- Definitions based on given conditions
def rate_A := 1 / 2 -- Rate of Pipe A (1 tank per 2 hours)
def rate_A_plus_L := 2 / 5 -- Combined rate of Pipe A and leak

-- Theorem states the time leak takes to empty full tank is 10 hours
theorem leak_empty_tank_time : 1 / (rate_A - rate_A_plus_L) = 10 :=
by
  -- Proof steps would go here
  sorry

end leak_empty_tank_time_l274_274772


namespace probability_fourth_term_integer_l274_274275

-- Define the initial condition
def initial_term : ℕ := 10

-- Define the sequence generation rules
  
def next_term (x : ℝ) (outcome : Bool) : ℝ :=
  if outcome then 2 * x - 1 else x / 2 - 1

-- Define the process to generate the sequence up to the fourth term

def term_seq (n : ℕ) : ℝ :=
  if n = 0 then initial_term
  else if n = 1 then next_term initial_term sorry -- the first coin result
  else if n = 2 then next_term (next_term initial_term sorry) sorry
  else if n = 3 then next_term (next_term (next_term initial_term sorry) sorry) sorry
  else next_term (next_term (next_term (next_term initial_term sorry) sorry) sorry) sorry

-- Proving the probability of the fourth term being an integer is 1/2
theorem probability_fourth_term_integer : 
  (probability {w : ℝ | ∃ a, term_seq 3 = a ∧ a ∈ ℤ} = 1 / 2) :=
by 
  -- To be completed
  sorry

end probability_fourth_term_integer_l274_274275


namespace solution_set_of_inequality_l274_274365

theorem solution_set_of_inequality :
  {x : ℝ | (x + 1) / x ≤ 3} = {x : ℝ | x < 0} ∪ {x : ℝ | x ≥ 1 / 2} :=
by
  sorry

end solution_set_of_inequality_l274_274365


namespace investment_recovery_l274_274332

-- Define the conditions and the goal
theorem investment_recovery (c : ℕ) : 
  (15 * c - 5 * c) ≥ 8000 ↔ c ≥ 800 := 
sorry

end investment_recovery_l274_274332


namespace J_speed_is_4_l274_274615

noncomputable def J_speed := 4
variable (v_J v_P : ℝ)

axiom condition1 : v_J > v_P
axiom condition2 : v_J + v_P = 7
axiom condition3 : (24 / v_J) + (24 / v_P) = 14

theorem J_speed_is_4 : v_J = J_speed :=
by
  sorry

end J_speed_is_4_l274_274615


namespace third_smallest_four_digit_number_in_pascals_triangle_l274_274223

theorem third_smallest_four_digit_number_in_pascals_triangle : 
    (∃ n, binomial n (n - 2) = 1002 ∧ binomial (n-1) (n - 1 - 2) = 1001 ∧ binomial (n-2) (n - 2 - 2) = 1000) :=
by
  sorry

end third_smallest_four_digit_number_in_pascals_triangle_l274_274223


namespace project_time_for_A_l274_274816

/--
A can complete a project in some days and B can complete the same project in 30 days.
If A and B start working on the project together and A quits 5 days before the project is 
completed, the project will be completed in 15 days.
Prove that A can complete the project alone in 20 days.
-/
theorem project_time_for_A (x : ℕ) (h : 10 * (1 / x + 1 / 30) + 5 * (1 / 30) = 1) : x = 20 :=
sorry

end project_time_for_A_l274_274816


namespace Ray_has_4_nickels_left_l274_274352

def Ray_initial_cents := 95
def Ray_cents_to_Peter := 25
def Ray_cents_to_Randi := 2 * Ray_cents_to_Peter

-- There are 5 cents in each nickel
def cents_per_nickel := 5

-- Nickels Ray originally has
def Ray_initial_nickels := Ray_initial_cents / cents_per_nickel
-- Nickels given to Peter
def Ray_nickels_to_Peter := Ray_cents_to_Peter / cents_per_nickel
-- Nickels given to Randi
def Ray_nickels_to_Randi := Ray_cents_to_Randi / cents_per_nickel
-- Total nickels given away
def Ray_nickels_given_away := Ray_nickels_to_Peter + Ray_nickels_to_Randi
-- Nickels left with Ray
def Ray_nickels_left := Ray_initial_nickels - Ray_nickels_given_away

theorem Ray_has_4_nickels_left :
  Ray_nickels_left = 4 :=
by
  sorry

end Ray_has_4_nickels_left_l274_274352


namespace units_digit_of_result_is_3_l274_274648

def hundreds_digit_relation (c : ℕ) (a : ℕ) : Prop :=
  a = 2 * c - 3

def original_number_expression (a b c : ℕ) : ℕ :=
  100 * a + 10 * b + c

def reversed_number_expression (a b c : ℕ) : ℕ :=
  100 * c + 10 * b + a + 50

def subtraction_result (orig rev : ℕ) : ℕ :=
  orig - rev

def units_digit (n : ℕ) : ℕ :=
  n % 10

theorem units_digit_of_result_is_3 (a b c : ℕ) (h : hundreds_digit_relation c a) :
  units_digit (subtraction_result (original_number_expression a b c)
                                  (reversed_number_expression a b c)) = 3 :=
by
  sorry

end units_digit_of_result_is_3_l274_274648


namespace max_value_of_y_l274_274620

noncomputable def max_y (x y : ℝ) : ℝ :=
  if h : x^2 + y^2 = 20*x + 54*y then y else 0

theorem max_value_of_y (x y : ℝ) (h : x^2 + y^2 = 20*x + 54*y) :
  max_y x y ≤ 27 + Real.sqrt 829 :=
sorry

end max_value_of_y_l274_274620


namespace prod_sum_rel_prime_l274_274184

theorem prod_sum_rel_prime (a b : ℕ) 
  (h1 : a * b + a + b = 119)
  (h2 : Nat.gcd a b = 1)
  (h3 : a < 25)
  (h4 : b < 25) : 
  a + b = 27 := 
sorry

end prod_sum_rel_prime_l274_274184


namespace solution_to_logarithmic_equation_l274_274657

noncomputable def log_base (a b : ℝ) := Real.log b / Real.log a

def equation (x : ℝ) := log_base 2 x + 1 / log_base (x + 1) 2 = 1

theorem solution_to_logarithmic_equation :
  ∃ x > 0, equation x ∧ x = 1 :=
by
  sorry

end solution_to_logarithmic_equation_l274_274657


namespace extreme_value_f_at_a_eq_1_monotonic_intervals_f_exists_no_common_points_l274_274872

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := x^2 - a * x - a * Real.log (x - 1)

-- Problem 1: Prove that the extreme value of f(x) when a = 1 is \frac{3}{4} + \ln 2
theorem extreme_value_f_at_a_eq_1 : 
  f (3/2) 1 = 3/4 + Real.log 2 :=
sorry

-- Problem 2: Prove the monotonic intervals of f(x) based on the value of a
theorem monotonic_intervals_f :
  ∀ a : ℝ, 
    (if a ≤ 0 then 
      ∀ x, 1 < x → f x' a > 0
     else
      ∀ x, 1 < x ∧ x ≤ (a + 2) / 2 → f x a ≤ 0 ∧ ∀ x, x ≥ (a + 2) / 2 → f x a > 0) :=
sorry

-- Problem 3: Prove that for a ≥ 1, there exists an a such that f(x) has no common points with y = \frac{5}{8} + \ln 2
theorem exists_no_common_points (h : 1 ≤ a) :
  ∃ x : ℝ, f x a ≠ 5/8 + Real.log 2 :=
sorry

end extreme_value_f_at_a_eq_1_monotonic_intervals_f_exists_no_common_points_l274_274872


namespace sum_of_midpoints_y_coordinates_l274_274790

theorem sum_of_midpoints_y_coordinates (d e f : ℝ) (h : d + e + f = 15) : 
  (d + e) / 2 + (d + f) / 2 + (e + f) / 2 = 15 :=
by
  sorry

end sum_of_midpoints_y_coordinates_l274_274790


namespace compute_expression_l274_274838

theorem compute_expression (y : ℕ) (h : y = 3) : (y^8 + 10 * y^4 + 25) / (y^4 + 5) = 86 :=
by
  rw [h]
  sorry

end compute_expression_l274_274838


namespace lowest_die_exactly_3_prob_l274_274735

noncomputable def fair_die_prob_at_least (n : ℕ) : ℚ :=
  if h : 1 ≤ n ∧ n ≤ 6 then (6 - n + 1) / 6 else 0

noncomputable def prob_lowest_die_exactly_3 : ℚ :=
  let p_at_least_3 := fair_die_prob_at_least 3
  let p_at_least_4 := fair_die_prob_at_least 4
  (p_at_least_3 ^ 4) - (p_at_least_4 ^ 4)

theorem lowest_die_exactly_3_prob :
  prob_lowest_die_exactly_3 = 175 / 1296 := by
  sorry

end lowest_die_exactly_3_prob_l274_274735


namespace angle_relation_l274_274609

theorem angle_relation
  (x y z w : ℝ)
  (h_sum : x + y + z + (360 - w) = 360) :
  x = w - y - z :=
by
  sorry

end angle_relation_l274_274609


namespace determine_h_l274_274839

theorem determine_h (x : ℝ) (h : ℝ → ℝ) :
  2 * x ^ 5 + 4 * x ^ 3 + h x = 7 * x ^ 3 - 5 * x ^ 2 + 9 * x + 3 →
  h x = -2 * x ^ 5 + 3 * x ^ 3 - 5 * x ^ 2 + 9 * x + 3 :=
by
  intro h_eq
  sorry

end determine_h_l274_274839


namespace find_circumference_l274_274815

theorem find_circumference
  (C : ℕ)
  (h1 : ∃ (vA vB : ℕ), C > 0 ∧ vA > 0 ∧ vB > 0 ∧ 
                        (120 * (C/2 + 80)) = ((C - 80) * (C/2 - 120)) ∧
                        (C - 240) / vA = (C + 240) / vB) :
  C = 520 := 
  sorry

end find_circumference_l274_274815


namespace increase_by_percentage_l274_274138

theorem increase_by_percentage (a b : ℝ) (percentage : ℝ) (final : ℝ) : b = a * percentage → final = a + b → final = 437.5 :=
by
  sorry

end increase_by_percentage_l274_274138


namespace least_positive_integer_condition_l274_274030

theorem least_positive_integer_condition
  (a : ℤ) (ha1 : a % 4 = 1) (ha2 : a % 5 = 2) (ha3 : a % 6 = 3) :
  a > 0 → a = 57 :=
by
  intro ha_pos
  -- Proof omitted for brevity
  sorry

end least_positive_integer_condition_l274_274030


namespace fox_jeans_price_l274_274713

theorem fox_jeans_price (F : ℝ) (P : ℝ) 
  (pony_price : P = 18) 
  (total_savings : 3 * F * 0.08 + 2 * P * 0.14 = 8.64)
  (total_discount_rate : 0.08 + 0.14 = 0.22)
  (pony_discount_rate : 0.14 = 13.999999999999993 / 100) 
  : F = 15 :=
by
  sorry

end fox_jeans_price_l274_274713


namespace total_amount_shared_l274_274960

theorem total_amount_shared (J Jo B : ℝ) (r1 r2 r3 : ℝ)
  (H1 : r1 = 2) (H2 : r2 = 4) (H3 : r3 = 6) (H4 : J = 1600) (part_value : ℝ)
  (H5 : part_value = J / r1) (H6 : Jo = r2 * part_value) (H7 : B = r3 * part_value) :
  J + Jo + B = 9600 :=
sorry

end total_amount_shared_l274_274960


namespace third_smallest_four_digit_in_pascal_triangle_l274_274212

-- Defining Pascal's triangle
def pascal (n k : ℕ) : ℕ := if k = 0 ∨ k = n then 1 else pascal (n - 1) (k - 1) + pascal (n - 1) k

/--
Condition:
1. In Pascal's triangle, each number is the sum of the two numbers directly above it.
2. Four-digit numbers start appearing regularly in higher rows (from row 45 onwards).
-/
theorem third_smallest_four_digit_in_pascal_triangle : 
  ∃ n k : ℕ, pascal n k = 1002 ∧ ∀ m l : ℕ, (pascal m l < 1000 ∨ pascal m l = 1001) → (m < n ∨ (m = n ∧ l < k)) :=
sorry

end third_smallest_four_digit_in_pascal_triangle_l274_274212


namespace total_rent_calculation_l274_274348

variables (x y : ℕ) -- x: number of rooms rented for $40, y: number of rooms rented for $60
variable (rent_total : ℕ)

-- Condition: Each room at the motel was rented for either $40 or $60
-- Condition: If 10 of the rooms that were rented for $60 had instead been rented for $40, the total rent would have been reduced by 50 percent

theorem total_rent_calculation 
  (h1 : 40 * (x + 10) + 60 * (y - 10) = (40 * x + 60 * y) / 2) :
  40 * x + 60 * y = 800 :=
sorry

end total_rent_calculation_l274_274348


namespace factor_expression_l274_274986

theorem factor_expression (a b c : ℝ) :
  ((a^3 - b^3)^3 + (b^3 - c^3)^3 + (c^3 - a^3)^3) / ((a - b)^3 + (b - c)^3 + (c - a)^3) =
  (a^2 + a * b + b^2) * (b^2 + b * c + c^2) * (c^2 + c * a + a^2) :=
by
  sorry

end factor_expression_l274_274986


namespace third_smallest_four_digit_number_in_pascals_triangle_l274_274224

theorem third_smallest_four_digit_number_in_pascals_triangle : 
    (∃ n, binomial n (n - 2) = 1002 ∧ binomial (n-1) (n - 1 - 2) = 1001 ∧ binomial (n-2) (n - 2 - 2) = 1000) :=
by
  sorry

end third_smallest_four_digit_number_in_pascals_triangle_l274_274224


namespace resultant_force_correct_l274_274521

-- Define the conditions
def P1 : ℝ := 80
def P2 : ℝ := 130
def distance : ℝ := 12.035
def theta1 : ℝ := 125
def theta2 : ℝ := 135.1939

-- Calculate the correct answer
def result_magnitude : ℝ := 209.299
def result_direction : ℝ := 131.35

-- The goal statement to be proved
theorem resultant_force_correct :
  ∃ (R : ℝ) (theta_R : ℝ), 
    R = result_magnitude ∧ theta_R = result_direction := 
sorry

end resultant_force_correct_l274_274521


namespace value_of_expression_l274_274997

theorem value_of_expression (a b : ℝ) (h : a - b = 1) : a^2 - b^2 - 2 * b = 1 := 
by
  sorry

end value_of_expression_l274_274997


namespace part1_part2_l274_274458
noncomputable def f (x : ℝ) : ℝ := abs (x - 1) + abs (x - 2)

theorem part1 : {x : ℝ | f x ≥ 3} = {x | x ≤ 0} ∪ {x | x ≥ 3} :=
by
  sorry

theorem part2 (a : ℝ) : (∃ x : ℝ, f x ≤ -a^2 + a + 7) ↔ -2 ≤ a ∧ a ≤ 3 :=
by
  sorry

end part1_part2_l274_274458


namespace price_increase_after_reduction_l274_274652

theorem price_increase_after_reduction (P : ℝ) (h : P > 0) : 
  let reduced_price := P * 0.85
  let increase_factor := 1 / 0.85
  let percentage_increase := (increase_factor - 1) * 100
  percentage_increase = 17.65 := by
  sorry

end price_increase_after_reduction_l274_274652


namespace value_of_k_l274_274880

theorem value_of_k (k x : ℕ) (h1 : 2^x - 2^(x - 2) = k * 2^10) (h2 : x = 12) : k = 3 := by
  sorry

end value_of_k_l274_274880


namespace smaller_part_volume_l274_274114

noncomputable def volume_of_smaller_part (a : ℝ) : ℝ :=
  (25 / 144) * (a^3)

theorem smaller_part_volume (a : ℝ) (h_pos : 0 < a) :
  ∃ v : ℝ, v = volume_of_smaller_part a :=
  sorry

end smaller_part_volume_l274_274114


namespace geometric_sequence_first_term_l274_274928

variable {b q a d : ℝ}

-- Hypotheses and Conditions
hypothesis h_geom_seq : ∀ n, b * q ^ n
hypothesis h_arith_seq : ∀ n, a + n * d
hypothesis h_product : b * (b * q) * (b * q^2) = 64

-- Theorems to prove b == 8/3
theorem geometric_sequence_first_term :
  (∃ b q a d : ℝ, (∀ n, b * q ^ n = a + n * d) ∧ (b * (b * q) * (b * q^2) = 64)) → b = 8/3 := by 
  sorry

end geometric_sequence_first_term_l274_274928


namespace sum_of_cubes_of_consecutive_integers_l274_274705

theorem sum_of_cubes_of_consecutive_integers (n : ℕ) (h : (n-1)^2 + n^2 + (n+1)^2 = 8450) : 
  (n-1)^3 + n^3 + (n+1)^3 = 446949 := 
sorry

end sum_of_cubes_of_consecutive_integers_l274_274705


namespace fraction_susan_can_eat_l274_274499

theorem fraction_susan_can_eat
  (v t n nf : ℕ)
  (h₁ : v = 6)
  (h₂ : n = 4)
  (h₃ : 1/3 * t = v)
  (h₄ : nf = v - n) :
  nf / t = 1 / 9 :=
sorry

end fraction_susan_can_eat_l274_274499


namespace gwen_money_received_from_dad_l274_274439

variables (D : ℕ)

-- Conditions
def mom_received := 8
def mom_more_than_dad := 3

-- Question and required proof
theorem gwen_money_received_from_dad : 
  (mom_received = D + mom_more_than_dad) -> D = 5 := 
by
  sorry

end gwen_money_received_from_dad_l274_274439


namespace least_positive_integer_satifies_congruences_l274_274044

theorem least_positive_integer_satifies_congruences :
  ∃ x : ℕ, x ≡ 1 [MOD 4] ∧ x ≡ 2 [MOD 5] ∧ x ≡ 3 [MOD 6] ∧ x = 17 :=
sorry

end least_positive_integer_satifies_congruences_l274_274044


namespace ratio_of_areas_l274_274494

theorem ratio_of_areas 
  (lenA : ℕ) (brdA : ℕ) (lenB : ℕ) (brdB : ℕ)
  (h_lenA : lenA = 48) 
  (h_brdA : brdA = 30)
  (h_lenB : lenB = 60) 
  (h_brdB : brdB = 35) :
  (lenA * brdA : ℚ) / (lenB * brdB) = 24 / 35 :=
by
  sorry

end ratio_of_areas_l274_274494


namespace card_pairing_modulus_l274_274374

theorem card_pairing_modulus (cards : Finset ℕ) (h : cards = (Finset.range 60).image (λ n, n + 1)) :
  ∃ n, n = 8 ∧ ∀ (pairs : Finset (ℕ × ℕ)), (∀ (p ∈ pairs), (p.1 ∈ cards ∧ p.2 ∈ cards ∧ (|p.1 - p.2| = d))) → pairs.card = 30 :=
sorry

end card_pairing_modulus_l274_274374


namespace area_inequality_equality_condition_l274_274618

variable (a b c d S : ℝ)
variable (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (h4 : d > 0) 
variable (s : ℝ) (h5 : s = (a + b + c + d) / 2)
variable (h6 : S = Real.sqrt ((s - a) * (s - b) * (s - c) * (s - d)))

theorem area_inequality (h : S = Real.sqrt ((s - a) * (s - b) * (s - c) * (s - d)) ∧ s = (a + b + c + d) / 2) :
  S ≤ Real.sqrt (a * b * c * d) :=
sorry

theorem equality_condition (h : S = Real.sqrt ((s - a) * (s - b) * (s - c) * (s - d)) ∧ s = (a + b + c + d) / 2) :
  (S = Real.sqrt (a * b * c * d)) ↔ (a = c ∧ b = d ∨ a = d ∧ b = c) :=
sorry

end area_inequality_equality_condition_l274_274618


namespace taco_beef_per_taco_l274_274244

open Real

theorem taco_beef_per_taco
  (total_beef : ℝ)
  (sell_price : ℝ)
  (cost_per_taco : ℝ)
  (profit : ℝ)
  (h1 : total_beef = 100)
  (h2 : sell_price = 2)
  (h3 : cost_per_taco = 1.5)
  (h4 : profit = 200) :
  ∃ (x : ℝ), x = 1/4 := 
by
  -- The proof will go here.
  sorry

end taco_beef_per_taco_l274_274244


namespace reading_time_difference_in_minutes_l274_274230

noncomputable def xanthia_reading_speed : ℝ := 120 -- pages per hour
noncomputable def molly_reading_speed : ℝ := 60 -- pages per hour
noncomputable def book_length : ℝ := 360 -- pages

theorem reading_time_difference_in_minutes :
  let time_for_xanthia := book_length / xanthia_reading_speed
  let time_for_molly := book_length / molly_reading_speed
  let difference_in_hours := time_for_molly - time_for_xanthia
  difference_in_hours * 60 = 180 :=
by
  sorry

end reading_time_difference_in_minutes_l274_274230


namespace range_of_m_l274_274771

theorem range_of_m (m n : ℝ) (h1 : n = 2 / m) (h2 : n ≥ -1) :
  m ≤ -2 ∨ m > 0 := 
sorry

end range_of_m_l274_274771


namespace inequality_not_always_hold_l274_274136

variables {a b c d : ℝ}

theorem inequality_not_always_hold 
  (h1 : a > b) 
  (h2 : c > d) 
: ¬ (a + d > b + c) :=
  sorry

end inequality_not_always_hold_l274_274136


namespace evaluate_sum_l274_274453

variable {a b c : ℝ}

theorem evaluate_sum 
  (h : a / (30 - a) + b / (75 - b) + c / (55 - c) = 8) :
  6 / (30 - a) + 15 / (75 - b) + 11 / (55 - c) = 187 / 30 :=
by
  sorry

end evaluate_sum_l274_274453


namespace number_of_real_values_p_l274_274844

theorem number_of_real_values_p (p : ℝ) :
  (∀ p: ℝ, x^2 - (p + 1) * x + (p + 1)^2 = 0 -> (p + 1) ^ 2 = 0) ↔ p = -1 := by
  sorry

end number_of_real_values_p_l274_274844


namespace supplement_of_double_complement_l274_274525

def angle : ℝ := 30

def complement (θ : ℝ) : ℝ :=
  90 - θ

def double_complement (θ : ℝ) : ℝ :=
  2 * (complement θ)

def supplement (θ : ℝ) : ℝ :=
  180 - θ

theorem supplement_of_double_complement (θ : ℝ) (h : θ = angle) : supplement (double_complement θ) = 60 :=
by
  sorry

end supplement_of_double_complement_l274_274525


namespace n_times_2pow_nplus1_plus_1_is_square_l274_274712

theorem n_times_2pow_nplus1_plus_1_is_square (n : ℕ) (h : 0 < n) :
  ∃ m : ℤ, n * 2 ^ (n + 1) + 1 = m * m ↔ n = 3 := 
by
  sorry

end n_times_2pow_nplus1_plus_1_is_square_l274_274712


namespace sum_of_digits_is_10_l274_274664

def sum_of_digits_of_expression : ℕ :=
  let expression := 2^2010 * 5^2008 * 7
  let simplified := 280000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
  2 + 8

/-- The sum of the digits of the decimal representation of 2^2010 * 5^2008 * 7 is 10 -/
theorem sum_of_digits_is_10 :
  sum_of_digits_of_expression = 10 :=
by sorry

end sum_of_digits_is_10_l274_274664


namespace seedling_sales_problem_l274_274064

variable (x y z : ℕ)

theorem seedling_sales_problem
  (h1 : 3 * x + 2 * y + z = 29000)
  (h2 : x = (1 / 2) * y)
  (h3 : y = (3 / 4) * z) :
  x + y + z = 17000 :=
sorry

end seedling_sales_problem_l274_274064


namespace new_total_energy_l274_274026

-- Define the problem conditions
def identical_point_charges_positioned_at_vertices_of_equilateral_triangle (charges : ℕ) (initial_energy : ℝ) : Prop :=
  charges = 3 ∧ initial_energy = 18

def charge_moved_one_third_along_side (move_fraction : ℝ) : Prop :=
  move_fraction = 1/3

-- Define the theorem and proof goal
theorem new_total_energy (charges : ℕ) (initial_energy : ℝ) (move_fraction : ℝ) :
  identical_point_charges_positioned_at_vertices_of_equilateral_triangle charges initial_energy →
  charge_moved_one_third_along_side move_fraction →
  ∃ (new_energy : ℝ), new_energy = 21 :=
by
  intros h_triangle h_move
  sorry

end new_total_energy_l274_274026


namespace smallest_x_l274_274393

theorem smallest_x (x : ℕ) (h₁ : x % 3 = 2) (h₂ : x % 4 = 3) (h₃ : x % 5 = 4) : x = 59 :=
by
  sorry

end smallest_x_l274_274393


namespace find_integers_satisfying_equation_l274_274094

theorem find_integers_satisfying_equation :
  ∃ (a b c : ℤ), (a = 1 ∧ b = 0 ∧ c = 0) ∨ (a = 0 ∧ b = 1 ∧ c = 0) ∨ (a = 0 ∧ b = 0 ∧ c = 1) ∨
                  (a = 2 ∧ b = -1 ∧ c = -1) ∨ (a = -1 ∧ b = 2 ∧ c = -1) ∨ (a = -1 ∧ b = -1 ∧ c = 2)
  ↔ (∃ (a b c : ℤ), 1 / 2 * (a + b) * (b + c) * (c + a) + (a + b + c) ^ 3 = 1 - a * b * c) := sorry

end find_integers_satisfying_equation_l274_274094


namespace third_smallest_four_digit_number_in_pascals_triangle_l274_274196

theorem third_smallest_four_digit_number_in_pascals_triangle : 
  ∃ n, is_four_digit (binomial n k) ∧ third_smallest n = 1002 :=
by
  assume P : (∀ k n, 0 ≤ k ∧ k ≤ n → ∃ m, m = binomial n k)
  sorry

end third_smallest_four_digit_number_in_pascals_triangle_l274_274196


namespace company_p_employees_in_january_l274_274058

-- Conditions
def employees_in_december (january_employees : ℝ) : ℝ := january_employees + 0.15 * january_employees

theorem company_p_employees_in_january (january_employees : ℝ) :
  employees_in_december january_employees = 490 → january_employees = 426 :=
by
  intro h
  -- The proof steps will be filled here.
  sorry

end company_p_employees_in_january_l274_274058


namespace find_original_amount_l274_274081

-- Let X be the original amount of money in Christina's account.
variable (X : ℝ)

-- Condition 1: Remaining balance after transferring 20% is $30,000.
def initial_transfer (X : ℝ) : Prop :=
  0.80 * X = 30000

-- Prove that the original amount before the initial transfer was $37,500.
theorem find_original_amount (h : initial_transfer X) : X = 37500 :=
  sorry

end find_original_amount_l274_274081


namespace factorization_of_expression_l274_274983

theorem factorization_of_expression (a b c : ℝ) : 
  ((a^3 - b^3)^3 + (b^3 - c^3)^3 + (c^3 - a^3)^3) / 
  ((a - b)^3 + (b - c)^3 + (c - a)^3) = 
  (a^2 + ab + b^2) * (b^2 + bc + c^2) * (c^2 + ca + a^2) :=
by
  sorry

end factorization_of_expression_l274_274983


namespace failed_in_english_l274_274751

/- Lean definitions and statement -/

def total_percentage := 100
def failed_H := 32
def failed_H_and_E := 12
def passed_H_or_E := 24

theorem failed_in_english (total_percentage failed_H failed_H_and_E passed_H_or_E : ℕ) (h1 : total_percentage = 100) (h2 : failed_H = 32) (h3 : failed_H_and_E = 12) (h4 : passed_H_or_E = 24) :
  total_percentage - (failed_H + (total_percentage - passed_H_or_E - failed_H_and_E)) = 56 :=
by sorry

end failed_in_english_l274_274751


namespace gcd_polynomial_l274_274455

theorem gcd_polynomial (b : ℤ) (k : ℤ) (hk : k % 2 = 1) (h_b : b = 1193 * k) :
  Int.gcd (2 * b^2 + 31 * b + 73) (b + 17) = 1 := 
  sorry

end gcd_polynomial_l274_274455


namespace jack_jogging_speed_needed_l274_274155

noncomputable def jack_normal_speed : ℝ :=
  let normal_melt_time : ℝ := 10
  let faster_melt_factor : ℝ := 0.75
  let adjusted_melt_time : ℝ := normal_melt_time * faster_melt_factor
  let adjusted_melt_time_hours : ℝ := adjusted_melt_time / 60
  let distance_to_beach : ℝ := 2
  let required_speed : ℝ := distance_to_beach / adjusted_melt_time_hours
  let slope_reduction_factor : ℝ := 0.8
  required_speed / slope_reduction_factor

theorem jack_jogging_speed_needed
  (normal_melt_time : ℝ := 10) 
  (faster_melt_factor : ℝ := 0.75) 
  (distance_to_beach : ℝ := 2) 
  (slope_reduction_factor : ℝ := 0.8) :
  jack_normal_speed = 20 := 
by
  sorry

end jack_jogging_speed_needed_l274_274155


namespace relationship_between_a_b_c_l274_274271

theorem relationship_between_a_b_c
  (a : ℝ) (b : ℝ) (c : ℝ)
  (h₁ : a = (10 ^ 1988 + 1) / (10 ^ 1989 + 1))
  (h₂ : b = (10 ^ 1987 + 1) / (10 ^ 1988 + 1))
  (h₃ : c = (10 ^ 1987 + 9) / (10 ^ 1988 + 9)) :
  a < b ∧ b < c := 
sorry

end relationship_between_a_b_c_l274_274271


namespace domain_f_2x_plus_1_eq_l274_274301

-- Conditions
def domain_fx_plus_1 : Set ℝ := {x : ℝ | -2 < x ∧ x < -1}

-- Question and Correct Answer
theorem domain_f_2x_plus_1_eq :
  (∃ (x : ℝ), x ∈ domain_fx_plus_1) →
  {x : ℝ | -1 < x ∧ x < -1/2} = {x : ℝ | (2*x + 1 ∈ domain_fx_plus_1)} :=
by
  sorry

end domain_f_2x_plus_1_eq_l274_274301


namespace b_contribution_l274_274535

/-- A starts business with Rs. 3500.
    After 9 months, B joins as a partner.
    After a year, the profit is divided in the ratio 2:3.
    Prove that B's contribution to the capital is Rs. 21000. -/
theorem b_contribution (a_capital : ℕ) (months_a : ℕ) (b_time : ℕ) (profit_ratio_num : ℕ) (profit_ratio_den : ℕ)
  (h_a_capital : a_capital = 3500)
  (h_months_a : months_a = 12)
  (h_b_time : b_time = 3)
  (h_profit_ratio : profit_ratio_num = 2 ∧ profit_ratio_den = 3) :
  (21000 * b_time * profit_ratio_num) / (3 * profit_ratio_den) = 3500 * months_a := by
  sorry

end b_contribution_l274_274535


namespace quadratic_intersection_y_axis_l274_274923

theorem quadratic_intersection_y_axis :
  (∃ y, y = 3 * (0: ℝ)^2 - 4 * (0: ℝ) + 5 ∧ (0, y) = (0, 5)) :=
by
  sorry

end quadratic_intersection_y_axis_l274_274923


namespace solution_xyz_uniqueness_l274_274234

theorem solution_xyz_uniqueness (x y z : ℝ) :
  x + y + z = 3 ∧ x^2 + y^2 + z^2 = 3 ∧ x^3 + y^3 + z^3 = 3 → x = 1 ∧ y = 1 ∧ z = 1 :=
by
  sorry

end solution_xyz_uniqueness_l274_274234


namespace subset_zero_in_A_l274_274316

def A := { x : ℝ | x > -1 }

theorem subset_zero_in_A : {0} ⊆ A :=
by sorry

end subset_zero_in_A_l274_274316


namespace correct_assignment_statement_l274_274248

def is_assignment_statement (stmt : String) : Prop :=
  stmt = "a = 2a"

theorem correct_assignment_statement : is_assignment_statement "a = 2a" :=
by
  sorry

end correct_assignment_statement_l274_274248


namespace card_paiting_modulus_l274_274370

theorem card_paiting_modulus (cards : Finset ℕ) (H : cards = Finset.range 61 \ {0}) :
  ∃ d : ℕ, ∀ n ∈ cards, ∃! k, (∀ x ∈ cards, (x + n ≡ k [MOD d])) ∧ (d ∣ 30) ∧ (∃! n : ℕ, 1 ≤ n ∧ n ≤ 8) :=
sorry

end card_paiting_modulus_l274_274370


namespace john_saved_120_dollars_l274_274330

-- Defining the conditions
def num_machines : ℕ := 10
def ball_bearings_per_machine : ℕ := 30
def total_ball_bearings : ℕ := num_machines * ball_bearings_per_machine
def regular_price_per_bearing : ℝ := 1
def sale_price_per_bearing : ℝ := 0.75
def bulk_discount : ℝ := 0.20
def discounted_price_per_bearing : ℝ := sale_price_per_bearing - (bulk_discount * sale_price_per_bearing)

-- Calculate total costs
def total_cost_without_sale : ℝ := total_ball_bearings * regular_price_per_bearing
def total_cost_with_sale : ℝ := total_ball_bearings * discounted_price_per_bearing

-- Calculate the savings
def savings : ℝ := total_cost_without_sale - total_cost_with_sale

-- The theorem we want to prove
theorem john_saved_120_dollars : savings = 120 := by
  sorry

end john_saved_120_dollars_l274_274330


namespace find_x_range_l274_274571

theorem find_x_range : 
  {x : ℝ | (2 / (x + 2) + 4 / (x + 8) ≤ 3 / 4)} = 
  {x : ℝ | (-4 < x ∧ x ≤ -2) ∨ (4 ≤ x)} := by
  sorry

end find_x_range_l274_274571
