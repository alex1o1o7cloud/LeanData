import Mathlib

namespace prairie_total_area_l1529_152931

theorem prairie_total_area (dust : ℕ) (untouched : ℕ) (total : ℕ) 
  (h1 : dust = 64535) (h2 : untouched = 522) : total = dust + untouched :=
by
  sorry

end prairie_total_area_l1529_152931


namespace parametric_to_ordinary_eq_l1529_152962

-- Define the parametric equations and the domain of the parameter t
def parametric_eqns (t : ℝ) : ℝ × ℝ := (t + 1, 3 - t^2)

-- Define the target equation to be proved
def target_eqn (x y : ℝ) : Prop := y = -x^2 + 2*x + 2

-- Prove that, given the parametric equations, the target ordinary equation holds
theorem parametric_to_ordinary_eq :
  ∃ (t : ℝ) (x y : ℝ), parametric_eqns t = (x, y) ∧ target_eqn x y :=
by
  sorry

end parametric_to_ordinary_eq_l1529_152962


namespace isosceles_triangle_perimeter_l1529_152959

theorem isosceles_triangle_perimeter (a b : ℕ) (h1 : a = 4 ∨ a = 7) (h2 : b = 4 ∨ b = 7) (h3 : a ≠ b) :
  (a + a + b = 15 ∨ a + a + b = 18) ∨ (a + b + b = 15 ∨ a + b + b = 18) :=
sorry

end isosceles_triangle_perimeter_l1529_152959


namespace common_factor_extraction_l1529_152988

-- Define the polynomial
def poly (a b c : ℝ) := 8 * a^3 * b^2 + 12 * a^3 * b * c - 4 * a^2 * b

-- Define the common factor
def common_factor (a b : ℝ) := 4 * a^2 * b

-- State the theorem
theorem common_factor_extraction (a b c : ℝ) :
  ∃ p : ℝ, poly a b c = common_factor a b * p := by
  sorry

end common_factor_extraction_l1529_152988


namespace positive_integers_not_in_E_are_perfect_squares_l1529_152956

open Set

def E : Set ℕ := {m | ∃ n : ℕ, m = Int.floor (n + Real.sqrt n + 0.5)}

theorem positive_integers_not_in_E_are_perfect_squares (m : ℕ) (h_pos : 0 < m) :
  m ∉ E ↔ ∃ t : ℕ, m = t^2 := 
by
    sorry

end positive_integers_not_in_E_are_perfect_squares_l1529_152956


namespace value_of_r_minus_p_l1529_152951

-- Define the arithmetic mean conditions
def arithmetic_mean1 (p q : ℝ) : Prop :=
  (p + q) / 2 = 10

def arithmetic_mean2 (q r : ℝ) : Prop :=
  (q + r) / 2 = 27

-- Prove that r - p = 34 based on the conditions
theorem value_of_r_minus_p (p q r : ℝ)
  (h1 : arithmetic_mean1 p q)
  (h2 : arithmetic_mean2 q r) :
  r - p = 34 :=
by
  sorry

end value_of_r_minus_p_l1529_152951


namespace find_p_value_l1529_152936

open Set

/-- Given the parabola C: y^2 = 2px with p > 0, point A(0, sqrt(3)),
    and point B on the parabola such that AB is perpendicular to AF,
    and |BF| = 4. Determine the value of p. -/
theorem find_p_value (p : ℝ) (h : p > 0) :
  ∃ p, p = 2 ∨ p = 6 :=
sorry

end find_p_value_l1529_152936


namespace exists_pos_ints_l1529_152983

open Nat

noncomputable def f (a : ℕ) : ℕ :=
  a^2 + 3 * a + 2

noncomputable def g (b c : ℕ) : ℕ :=
  b^2 - b + 3 * c^2 + 3 * c

theorem exists_pos_ints (a : ℕ) (ha : 0 < a) :
  ∃ (b c : ℕ), 0 < b ∧ 0 < c ∧ f a = g b c :=
sorry

end exists_pos_ints_l1529_152983


namespace unique_7tuple_exists_l1529_152920

theorem unique_7tuple_exists 
  (x : Fin 7 → ℝ) 
  (h : (1 - x 0)^2 + (x 0 - x 1)^2 + (x 1 - x 2)^2 + (x 2 - x 3)^2 + (x 3 - x 4)^2 + (x 4 - x 5)^2 + (x 5 - x 6)^2 + x 6^2 = 1 / 7) 
  : ∃! (x : Fin 7 → ℝ), (1 - x 0)^2 + (x 0 - x 1)^2 + (x 1 - x 2)^2 + (x 2 - x 3)^2 + (x 3 - x 4)^2 + (x 4 - x 5)^2 + (x 5 - x 6)^2 + x 6^2 = 1 / 7 :=
sorry

end unique_7tuple_exists_l1529_152920


namespace age_30_years_from_now_l1529_152934

variables (ElderSonAge : ℕ) (DeclanAgeDiff : ℕ) (YoungerSonAgeDiff : ℕ) (ThirdSiblingAgeDiff : ℕ)

-- Given conditions
def elder_son_age : ℕ := 40
def declan_age : ℕ := elder_son_age + 25
def younger_son_age : ℕ := elder_son_age - 10
def third_sibling_age : ℕ := younger_son_age - 5

-- To prove the ages 30 years from now
def younger_son_age_30_years_from_now : ℕ := younger_son_age + 30
def third_sibling_age_30_years_from_now : ℕ := third_sibling_age + 30

-- The proof statement
theorem age_30_years_from_now : 
  younger_son_age_30_years_from_now = 60 ∧ 
  third_sibling_age_30_years_from_now = 55 :=
by
  sorry

end age_30_years_from_now_l1529_152934


namespace winning_candidate_percentage_l1529_152982

theorem winning_candidate_percentage 
    (votes_winner : ℕ)
    (votes_total : ℕ)
    (votes_majority : ℕ)
    (H1 : votes_total = 900)
    (H2 : votes_majority = 360)
    (H3 : votes_winner - (votes_total - votes_winner) = votes_majority) :
    (votes_winner : ℕ) * 100 / (votes_total : ℕ) = 70 := by
    sorry

end winning_candidate_percentage_l1529_152982


namespace janet_more_siblings_than_carlos_l1529_152907

theorem janet_more_siblings_than_carlos :
  ∀ (masud_siblings : ℕ),
  masud_siblings = 60 →
  (janets_siblings : ℕ) →
  janets_siblings = 4 * masud_siblings - 60 →
  (carlos_siblings : ℕ) →
  carlos_siblings = 3 * masud_siblings / 4 →
  janets_siblings - carlos_siblings = 45 :=
by
  intros masud_siblings hms janets_siblings hjs carlos_siblings hcs
  sorry

end janet_more_siblings_than_carlos_l1529_152907


namespace area_of_circle_portion_l1529_152972

theorem area_of_circle_portion :
  (∀ x y : ℝ, (x^2 + 6 * x + y^2 = 50) → y ≤ x - 3 → y ≤ 0 → (y^2 + (x + 3)^2 ≤ 59)) →
  (∃ area : ℝ, area = (59 * Real.pi / 4)) :=
by
  sorry

end area_of_circle_portion_l1529_152972


namespace trigonometric_identity_l1529_152914

theorem trigonometric_identity (α : ℝ) (h : Real.sin α = 2 * Real.cos α) :
  Real.sin α ^ 2 + 2 * Real.cos α ^ 2 = 6 / 5 := 
by 
  sorry

end trigonometric_identity_l1529_152914


namespace largest_x_value_satisfies_largest_x_value_l1529_152987

theorem largest_x_value (x : ℚ) (h : Real.sqrt (3 * x) = 5 * x) : x ≤ 3 / 25 :=
sorry

theorem satisfies_largest_x_value : 
  ∃ x : ℚ, Real.sqrt (3 * x) = 5 * x ∧ x = 3 / 25 :=
sorry

end largest_x_value_satisfies_largest_x_value_l1529_152987


namespace seashells_found_l1529_152916

theorem seashells_found (C B : ℤ) (h1 : 9 * B = 7 * C) (h2 : B = C - 12) : C = 54 :=
by
  sorry

end seashells_found_l1529_152916


namespace positive_difference_of_solutions_l1529_152998

theorem positive_difference_of_solutions :
  let a := 1
  let b := -6
  let c := -28
  let discriminant := b^2 - 4 * a * c
  let solution1 := 3 + (Real.sqrt discriminant) / 2
  let solution2 := 3 - (Real.sqrt discriminant) / 2
  have h_discriminant : discriminant = 148 := by sorry
  Real.sqrt 148 = 2 * Real.sqrt 37 :=
 sorry

end positive_difference_of_solutions_l1529_152998


namespace initial_customers_l1529_152909

theorem initial_customers (x : ℕ) (h1 : x - 31 + 26 = 28) : x = 33 := 
by 
  sorry

end initial_customers_l1529_152909


namespace max_boxes_in_large_box_l1529_152974

def max_boxes (l_L w_L h_L : ℕ) (l_S w_S h_S : ℕ) : ℕ :=
  (l_L * w_L * h_L) / (l_S * w_S * h_S)

theorem max_boxes_in_large_box :
  let l_L := 8 * 100 -- converted to cm
  let w_L := 7 * 100 -- converted to cm
  let h_L := 6 * 100 -- converted to cm
  let l_S := 4
  let w_S := 7
  let h_S := 6
  max_boxes l_L w_L h_L l_S w_S h_S = 2000000 :=
by {
  let l_L := 800 -- converted to cm
  let w_L := 700 -- converted to cm
  let h_L := 600 -- converted to cm
  let l_S := 4
  let w_S := 7
  let h_S := 6
  trivial
}

end max_boxes_in_large_box_l1529_152974


namespace wallpaper_three_layers_l1529_152902

theorem wallpaper_three_layers
  (A B C : ℝ)
  (hA : A = 300)
  (hB : B = 30)
  (wall_area : ℝ)
  (h_wall_area : wall_area = 180)
  (hC : C = A - (wall_area - B) - B)
  : C = 120 := by
  sorry

end wallpaper_three_layers_l1529_152902


namespace monthly_payment_l1529_152977

theorem monthly_payment (price : ℝ) (discount_rate : ℝ) (down_payment : ℝ) (months : ℕ) (monthly_payment : ℝ) :
  price = 480 ∧ discount_rate = 0.05 ∧ down_payment = 150 ∧ months = 3 ∧
  monthly_payment = (price * (1 - discount_rate) - down_payment) / months →
  monthly_payment = 102 :=
by
  sorry

end monthly_payment_l1529_152977


namespace partial_fraction_decomposition_product_l1529_152961

theorem partial_fraction_decomposition_product :
  ∃ A B C : ℚ,
    (A + 2) * (A - 3) *
    (B - 2) * (B - 3) *
    (C - 2) * (C + 2) = x^2 - 12 ∧
    (A = -2) ∧
    (B = 2/5) ∧
    (C = 3/5) ∧
    (A * B * C = -12/25) :=
  sorry

end partial_fraction_decomposition_product_l1529_152961


namespace evaluate_expression_l1529_152949

theorem evaluate_expression (x y : ℕ) (h1 : x = 3) (h2 : y = 7) :
  (x^5 + 3 * y^3) / 9 = 141 :=
by
  sorry

end evaluate_expression_l1529_152949


namespace laurie_shells_l1529_152980

def alan_collected : ℕ := 48
def ben_collected (alan : ℕ) : ℕ := alan / 4
def laurie_collected (ben : ℕ) : ℕ := ben * 3

theorem laurie_shells (a : ℕ) (b : ℕ) (l : ℕ) (h1 : alan_collected = a)
  (h2 : ben_collected a = b) (h3 : laurie_collected b = l) : l = 36 := 
by
  sorry

end laurie_shells_l1529_152980


namespace factor_1_factor_2_factor_3_l1529_152922

-- Consider the variables a, b, x, y
variable (a b x y : ℝ)

-- Statement 1: Factorize 3a^3 - 6a^2 + 3a
theorem factor_1 : 3 * a^3 - 6 * a^2 + 3 * a = 3 * a * (a - 1)^2 :=
by
  sorry
  
-- Statement 2: Factorize a^2(x - y) + b^2(y - x)
theorem factor_2 : a^2 * (x - y) + b^2 * (y - x) = (x - y) * (a^2 - b^2) :=
by
  sorry

-- Statement 3: Factorize 16(a + b)^2 - 9(a - b)^2
theorem factor_3 : 16 * (a + b)^2 - 9 * (a - b)^2 = (a + 7 * b) * (7 * a + b) :=
by
  sorry

end factor_1_factor_2_factor_3_l1529_152922


namespace hike_on_saturday_l1529_152939

-- Define the conditions
variables (x : Real) -- distance hiked on Saturday
variables (y : Real) -- distance hiked on Sunday
variables (z : Real) -- total distance hiked

-- Define given values
def hiked_on_sunday : Real := 1.6
def total_hiked : Real := 9.8

-- The hypothesis: y + x = z
axiom hike_total : y + x = z

theorem hike_on_saturday : x = 8.2 :=
by
  sorry

end hike_on_saturday_l1529_152939


namespace sandy_spent_home_currency_l1529_152918

variable (A B C D : ℝ)

def total_spent_home_currency (A B C D : ℝ) : ℝ :=
  let total_foreign := A + B + C
  total_foreign * D

theorem sandy_spent_home_currency (D : ℝ) : 
  total_spent_home_currency 13.99 12.14 7.43 D = 33.56 * D := 
by
  sorry

end sandy_spent_home_currency_l1529_152918


namespace pat_more_hours_than_jane_l1529_152924

theorem pat_more_hours_than_jane (H P K M J : ℝ) 
  (h_total : H = P + K + M + J)
  (h_pat : P = 2 * K)
  (h_mark : M = (1/3) * P)
  (h_jane : J = (1/2) * M)
  (H290 : H = 290) :
  P - J = 120.83 := 
by
  sorry

end pat_more_hours_than_jane_l1529_152924


namespace f_monotonicity_g_min_l1529_152940

-- Definitions
noncomputable def f (x : ℝ) (a : ℝ) : ℝ := 2 * a ^ x - 2 * a ^ (-x)
noncomputable def g (x : ℝ) (a : ℝ) : ℝ := a ^ (2 * x) + a ^ (-2 * x) - 2 * f x a

-- Conditions
variable {a : ℝ} 
variable (a_pos : 0 < a) (a_ne_one : a ≠ 1) (f_one : f 1 a = 3) (x : ℝ) (h : 0 ≤ x ∧ x ≤ 3)

-- Monotonicity of f(x)
theorem f_monotonicity : 
  (∀ x y, x < y → f x a < f y a) ∨ (∀ x y, x < y → f y a < f x a) :=
sorry

-- Minimum value of g(x)
theorem g_min : ∃ x' : ℝ, 0 ≤ x' ∧ x' ≤ 3 ∧ g x' a = -2 :=
sorry

end f_monotonicity_g_min_l1529_152940


namespace compute_one_plus_i_power_four_l1529_152905

theorem compute_one_plus_i_power_four (i : ℂ) (h : i^2 = -1) : (1 + i)^4 = -4 :=
by
  sorry

end compute_one_plus_i_power_four_l1529_152905


namespace triangle_area_l1529_152923

theorem triangle_area (a b c : ℝ) (C : ℝ) 
  (h1 : c^2 = (a - b)^2 + 6)
  (h2 : C = Real.pi / 3) : 
  (1/2) * a * b * Real.sin C = 3 * Real.sqrt 3 / 2 := 
by 
  sorry

end triangle_area_l1529_152923


namespace proof_least_sum_l1529_152917

noncomputable def least_sum (m n : ℕ) (h1 : Nat.gcd (m + n) 330 = 1) 
                           (h2 : n^n ∣ m^m) (h3 : ¬(n ∣ m)) : ℕ :=
  m + n

theorem proof_least_sum :
  ∃ m n : ℕ, Nat.gcd (m + n) 330 = 1 ∧ n^n ∣ m^m ∧ ¬(n ∣ m) ∧ m + n = 390 :=
by
  sorry

end proof_least_sum_l1529_152917


namespace add_pure_water_to_achieve_solution_l1529_152971

theorem add_pure_water_to_achieve_solution
  (w : ℝ) (h_salt_content : 0.15 * 40 = 6) (h_new_concentration : 6 / (40 + w) = 0.1) :
  w = 20 :=
sorry

end add_pure_water_to_achieve_solution_l1529_152971


namespace solve_system_l1529_152976

noncomputable def system_solutions (x y z : ℤ) : Prop :=
  x^3 + y^3 + z^3 = 8 ∧
  x^2 + y^2 + z^2 = 22 ∧
  (1 / x + 1 / y + 1 / z = - (z / (x * y)))

theorem solve_system :
  ∀ (x y z : ℤ), system_solutions x y z ↔ 
    (x = 3 ∧ y = 2 ∧ z = -3) ∨
    (x = -3 ∧ y = 2 ∧ z = 3) ∨
    (x = 2 ∧ y = 3 ∧ z = -3) ∨
    (x = 2 ∧ y = -3 ∧ z = 3) := by
  sorry

end solve_system_l1529_152976


namespace cylinder_surface_area_and_volume_l1529_152984

noncomputable def cylinder_total_surface_area (r h : ℝ) : ℝ :=
  2 * Real.pi * r^2 + 2 * Real.pi * r * h

noncomputable def cylinder_volume (r h : ℝ) : ℝ :=
  Real.pi * r^2 * h

theorem cylinder_surface_area_and_volume (r h : ℝ) (hr : r = 5) (hh : h = 15) :
  cylinder_total_surface_area r h = 200 * Real.pi ∧ cylinder_volume r h = 375 * Real.pi :=
by
  sorry -- Proof omitted

end cylinder_surface_area_and_volume_l1529_152984


namespace sum_of_solutions_l1529_152967

theorem sum_of_solutions : 
  ∃ x1 x2 x3 : ℝ, (x1 = 10 ∧ x2 = 50/7 ∧ x3 = 50 ∧ (x1 + x2 + x3 = 470 / 7) ∧ 
  (∀ x : ℝ, x = abs (3 * x - abs (50 - 3 * x)) → (x = x1 ∨ x = x2 ∨ x = x3))) := 
sorry

end sum_of_solutions_l1529_152967


namespace olympic_medals_l1529_152937

theorem olympic_medals (total_sprinters british_sprinters non_british_sprinters ways_case1 ways_case2 ways_case3 : ℕ)
  (h_total : total_sprinters = 10)
  (h_british : british_sprinters = 4)
  (h_non_british : non_british_sprinters = 6)
  (h_case1 : ways_case1 = 6 * 5 * 4)
  (h_case2 : ways_case2 = 4 * 3 * (6 * 5))
  (h_case3 : ways_case3 = (4 * 3) * (3 * 2) * 6) :
  ways_case1 + ways_case2 + ways_case3 = 912 := by
  sorry

end olympic_medals_l1529_152937


namespace race_length_l1529_152926

theorem race_length (covered_meters remaining_meters race_length : ℕ)
  (h_covered : covered_meters = 721)
  (h_remaining : remaining_meters = 279)
  (h_race_length : race_length = covered_meters + remaining_meters) :
  race_length = 1000 :=
by
  rw [h_covered, h_remaining] at h_race_length
  exact h_race_length

end race_length_l1529_152926


namespace parabola_intersects_y_axis_l1529_152906

theorem parabola_intersects_y_axis (m n : ℝ) :
  (∃ (x y : ℝ), y = x^2 + m * x + n ∧ 
  ((x = -1 ∧ y = -6) ∨ (x = 1 ∧ y = 0))) →
  (0, (-4)) = (0, n) :=
by
  sorry

end parabola_intersects_y_axis_l1529_152906


namespace island_of_misfortune_l1529_152954

def statement (n : ℕ) (knight : ℕ → Prop) (liar : ℕ → Prop) : Prop :=
  ∀ k : ℕ, k < n → (
    if k = 0 then ∀ m : ℕ, (m % 2 = 1) ↔ liar m
    else if k = 1 then ∀ m : ℕ, (m % 3 = 1) ↔ liar m
    else ∀ m : ℕ, (m % (k + 1) = 1) ↔ liar m
  )

theorem island_of_misfortune :
  ∃ n : ℕ, n >= 2 ∧ statement n knight liar
:= sorry

end island_of_misfortune_l1529_152954


namespace geometric_sequence_first_term_l1529_152928

theorem geometric_sequence_first_term (a b c : ℕ) 
    (h1 : 16 = a * (2^3)) 
    (h2 : 32 = a * (2^4)) : 
    a = 2 := 
sorry

end geometric_sequence_first_term_l1529_152928


namespace harper_water_duration_l1529_152989

theorem harper_water_duration
  (half_bottle_per_day : ℝ)
  (bottles_per_case : ℕ)
  (cost_per_case : ℝ)
  (total_spending : ℝ)
  (cases_bought : ℕ)
  (days_per_case : ℕ)
  (total_days : ℕ) :
  half_bottle_per_day = 1/2 →
  bottles_per_case = 24 →
  cost_per_case = 12 →
  total_spending = 60 →
  cases_bought = total_spending / cost_per_case →
  days_per_case = bottles_per_case * 2 →
  total_days = days_per_case * cases_bought →
  total_days = 240 :=
by
  -- The proof will be added here
  sorry

end harper_water_duration_l1529_152989


namespace factorization_A_factorization_B_factorization_C_factorization_D_incorrect_factorization_D_correct_l1529_152948

theorem factorization_A (x y : ℝ) : x^2 - 2 * x * y = x * (x - 2 * y) :=
  by sorry

theorem factorization_B (x y : ℝ) : x^2 - 25 * y^2 = (x - 5 * y) * (x + 5 * y) :=
  by sorry

theorem factorization_C (x : ℝ) : 4 * x^2 - 4 * x + 1 = (2 * x - 1)^2 :=
  by sorry

theorem factorization_D_incorrect (x : ℝ) : x^2 + x - 2 ≠ (x - 2) * (x + 1) :=
  by sorry

theorem factorization_D_correct (x : ℝ) : x^2 + x - 2 = (x + 2) * (x - 1) :=
  by sorry

end factorization_A_factorization_B_factorization_C_factorization_D_incorrect_factorization_D_correct_l1529_152948


namespace inequlity_proof_l1529_152986

theorem inequlity_proof (a b : ℝ) : a^2 + a * b + b^2 ≥ 3 * (a + b - 1) := 
  sorry

end inequlity_proof_l1529_152986


namespace find_k_from_roots_ratio_l1529_152941

theorem find_k_from_roots_ratio (k : ℝ) (x1 x2 : ℝ) 
  (h1 : x1 + x2 = -10)
  (h2 : x1 * x2 = k)
  (h3 : x1/x2 = 3) : 
  k = 18.75 := 
sorry

end find_k_from_roots_ratio_l1529_152941


namespace counterexample_to_strict_inequality_l1529_152981

theorem counterexample_to_strict_inequality :
  ∃ (a1 a2 b1 b2 c1 c2 d1 d2 : ℕ),
  (0 < a1) ∧ (0 < a2) ∧ (0 < b1) ∧ (0 < b2) ∧ (0 < c1) ∧ (0 < c2) ∧ (0 < d1) ∧ (0 < d2) ∧
  (a1 * b2 < a2 * b1) ∧ (c1 * d2 < c2 * d1) ∧ ¬ (a1 + c1) * (b2 + d2) < (a2 + c2) * (b1 + d1) :=
sorry

end counterexample_to_strict_inequality_l1529_152981


namespace concentric_circles_ratio_l1529_152953

theorem concentric_circles_ratio
  (a b : ℝ) (ha : 0 < a) (hb : 0 < b)
  (h : π * b^2 - π * a^2 = 4 * (π * a^2)) :
  a / b = 1 / Real.sqrt 5 :=
by
  sorry

end concentric_circles_ratio_l1529_152953


namespace simplify_complex_expression_l1529_152991

open Complex

theorem simplify_complex_expression :
  let a := (4 : ℂ) + 6 * I
  let b := (4 : ℂ) - 6 * I
  ((a / b) - (b / a) = (24 * I) / 13) := by
  sorry

end simplify_complex_expression_l1529_152991


namespace probability_of_shaded_triangle_l1529_152921

def total_triangles : ℕ := 9
def shaded_triangles : ℕ := 3

theorem probability_of_shaded_triangle :
  total_triangles > 5 →
  (shaded_triangles : ℚ) / total_triangles = 1 / 3 :=
by
  intros h
  -- proof here
  sorry

end probability_of_shaded_triangle_l1529_152921


namespace triangle_perimeter_l1529_152973

theorem triangle_perimeter (A B C : Type) 
  (x : ℝ) 
  (a b c : ℝ) 
  (h₁ : a = x + 1) 
  (h₂ : b = x) 
  (h₃ : c = x - 1) 
  (α β γ : ℝ) 
  (angle_condition : α = 2 * γ) 
  (law_of_sines : a / Real.sin α = c / Real.sin γ)
  (law_of_cosines : Real.cos γ = ((a^2 + b^2 - c^2) / (2 * b * a))) :
  a + b + c = 15 :=
  by
  sorry

end triangle_perimeter_l1529_152973


namespace range_of_expression_l1529_152999

theorem range_of_expression (x y : ℝ) (h1 : x * y = 1) (h2 : 3 ≥ x ∧ x ≥ 4 * y ∧ 4 * y > 0) :
  ∃ A B, A = 4 ∧ B = 5 ∧ ∀ z, z = (x^2 + 4 * y^2) / (x - 2 * y) → 4 ≤ z ∧ z ≤ 5 :=
by
  sorry

end range_of_expression_l1529_152999


namespace copper_zinc_ratio_l1529_152979

theorem copper_zinc_ratio (total_weight : ℝ) (zinc_weight : ℝ) 
  (h_total_weight : total_weight = 70) (h_zinc_weight : zinc_weight = 31.5) : 
  (70 - 31.5) / 31.5 = 77 / 63 :=
by
  have h_copper_weight : total_weight - zinc_weight = 38.5 :=
    by rw [h_total_weight, h_zinc_weight]; norm_num
  sorry

end copper_zinc_ratio_l1529_152979


namespace successful_multiplications_in_one_hour_l1529_152960

variable (multiplications_per_second : ℕ)
variable (error_rate_percentage : ℕ)

theorem successful_multiplications_in_one_hour
  (h1 : multiplications_per_second = 15000)
  (h2 : error_rate_percentage = 5)
  : (multiplications_per_second * 3600 * (100 - error_rate_percentage) / 100) 
    + (multiplications_per_second * 3600 * error_rate_percentage / 100) = 54000000 := by
  sorry

end successful_multiplications_in_one_hour_l1529_152960


namespace smallest_perfect_square_greater_than_x_l1529_152966

theorem smallest_perfect_square_greater_than_x (x : ℤ)
  (h₁ : ∃ k : ℤ, k^2 ≠ x)
  (h₂ : x ≥ 0) :
  ∃ n : ℤ, n^2 > x ∧ ∀ m : ℤ, m^2 > x → n^2 ≤ m^2 :=
sorry

end smallest_perfect_square_greater_than_x_l1529_152966


namespace part_1_part_2_part_3_l1529_152965

noncomputable def f (x : ℝ) (m : ℝ) : ℝ := x + m / x

theorem part_1 (h : f 1 m = 5) : m = 4 :=
sorry

theorem part_2 (m : ℝ) (h : m = 4) : ∀ x : ℝ, f (-x) m = -f x m :=
sorry

theorem part_3 (m : ℝ) (h : m = 4) : ∀ x1 x2 : ℝ, 2 < x1 → x1 < x2 → f x1 m < f x2 m :=
sorry

end part_1_part_2_part_3_l1529_152965


namespace total_boxes_correct_l1529_152942

def boxes_chocolate : ℕ := 2
def boxes_sugar : ℕ := 5
def boxes_gum : ℕ := 2
def total_boxes : ℕ := boxes_chocolate + boxes_sugar + boxes_gum

theorem total_boxes_correct : total_boxes = 9 := by
  sorry

end total_boxes_correct_l1529_152942


namespace colin_speed_l1529_152955

noncomputable def B : Real := 1
noncomputable def T : Real := 2 * B
noncomputable def Br : Real := (1/3) * T
noncomputable def C : Real := 6 * Br

theorem colin_speed : C = 4 := by
  sorry

end colin_speed_l1529_152955


namespace sum_eighth_row_interior_numbers_l1529_152958

-- Define the sum of the interior numbers in the nth row of Pascal's Triangle.
def sum_interior_numbers (n : ℕ) : ℕ := 2^(n-1) - 2

-- Problem statement: Prove the sum of the interior numbers of Pascal's Triangle in the eighth row is 126,
-- given the sums for the fifth and sixth rows.
theorem sum_eighth_row_interior_numbers :
  sum_interior_numbers 5 = 14 →
  sum_interior_numbers 6 = 30 →
  sum_interior_numbers 8 = 126 :=
by
  sorry

end sum_eighth_row_interior_numbers_l1529_152958


namespace dot_product_calculation_l1529_152935

def vec_a : ℝ × ℝ := (1, 0)
def vec_b : ℝ × ℝ := (2, 3)
def vec_s : ℝ × ℝ := (2 * vec_a.1 - vec_b.1, 2 * vec_a.2 - vec_b.2)
def vec_t : ℝ × ℝ := (vec_a.1 + vec_b.1, vec_a.2 + vec_b.2)
def dot_product (u v : ℝ × ℝ) : ℝ := u.1 * v.1 + u.2 * v.2

theorem dot_product_calculation :
  dot_product vec_s vec_t = -9 := by
  sorry

end dot_product_calculation_l1529_152935


namespace bread_last_days_l1529_152900

def total_consumption_per_member_breakfast : ℕ := 4
def total_consumption_per_member_snacks : ℕ := 3
def total_consumption_per_member : ℕ := total_consumption_per_member_breakfast + total_consumption_per_member_snacks
def family_members : ℕ := 6
def daily_family_consumption : ℕ := family_members * total_consumption_per_member
def slices_per_loaf : ℕ := 10
def total_loaves : ℕ := 5
def total_bread_slices : ℕ := total_loaves * slices_per_loaf

theorem bread_last_days : total_bread_slices / daily_family_consumption = 1 :=
by
  sorry

end bread_last_days_l1529_152900


namespace Robie_gave_away_boxes_l1529_152968

theorem Robie_gave_away_boxes :
  ∀ (total_cards cards_per_box boxes_with_him remaining_cards : ℕ)
  (h_total_cards : total_cards = 75)
  (h_cards_per_box : cards_per_box = 10)
  (h_boxes_with_him : boxes_with_him = 5)
  (h_remaining_cards : remaining_cards = 5),
  (total_cards / cards_per_box) - boxes_with_him = 2 :=
by
  intros total_cards cards_per_box boxes_with_him remaining_cards
  intros h_total_cards h_cards_per_box h_boxes_with_him h_remaining_cards
  sorry

end Robie_gave_away_boxes_l1529_152968


namespace team_A_processes_fraction_l1529_152995

theorem team_A_processes_fraction (A B : ℕ) (total_calls : ℚ) 
  (h1 : A = (5/8) * B) 
  (h2 : (8 / 11) * total_calls = TeamB_calls_processed)
  (frac_TeamA_calls : ℚ := (1 - (8 / 11)) * total_calls)
  (calls_per_member_A : ℚ := frac_TeamA_calls / A)
  (calls_per_member_B : ℚ := (8 / 11) * total_calls / B) : 
  calls_per_member_A / calls_per_member_B = 3 / 5 := 
by
  sorry

end team_A_processes_fraction_l1529_152995


namespace sum_of_three_numbers_l1529_152964

theorem sum_of_three_numbers :
  ∃ (a b c : ℕ), 
    (a ≤ b ∧ b ≤ c) ∧ 
    (b = 8) ∧ 
    ((a + b + c) / 3 = a + 8) ∧ 
    ((a + b + c) / 3 = c - 20) ∧ 
    (a + b + c = 60) :=
by
  sorry

end sum_of_three_numbers_l1529_152964


namespace side_salad_cost_l1529_152933

theorem side_salad_cost (T S : ℝ)
  (h1 : T + S + 4 + 2 = 2 * T) 
  (h2 : (T + S + 4 + 2) + T = 24) : S = 2 :=
by
  sorry

end side_salad_cost_l1529_152933


namespace trips_and_weights_l1529_152904

theorem trips_and_weights (x : ℕ) (w : ℕ) (trips_Bill Jean_total limit_total: ℕ)
  (h1 : x + (x + 6) = 40)
  (h2 : trips_Bill = x)
  (h3 : Jean_total = x + 6)
  (h4 : w = 7850)
  (h5 : limit_total = 8000)
  : 
  trips_Bill = 17 ∧ 
  Jean_total = 23 ∧ 
  (w : ℝ) / 40 = 196.25 := 
by 
  sorry

end trips_and_weights_l1529_152904


namespace retail_profit_percent_l1529_152930

variable (CP : ℝ) (MP : ℝ) (SP : ℝ)
variable (h_marked : MP = CP + 0.60 * CP)
variable (h_discount : SP = MP - 0.25 * MP)

theorem retail_profit_percent : CP = 100 → MP = CP + 0.60 * CP → SP = MP - 0.25 * MP → 
       (SP - CP) / CP * 100 = 20 := 
by
  intros h1 h2 h3
  sorry

end retail_profit_percent_l1529_152930


namespace part_a_part_b_l1529_152913

/- Part (a) -/
theorem part_a (a b c d : ℝ) (h1 : (a + b ≠ c + d)) (h2 : (a + c ≠ b + d)) (h3 : (a + d ≠ b + c)) :
  ∃ (spheres : ℕ), spheres = 8 := sorry

/- Part (b) -/
theorem part_b (a b c d : ℝ) (h : (a + b = c + d) ∨ (a + c = b + d) ∨ (a + d = b + c)) :
  ∃ (spheres : ℕ), ∀ (n : ℕ), n > 0 → spheres = n := sorry

end part_a_part_b_l1529_152913


namespace sum_of_non_visible_faces_l1529_152996

theorem sum_of_non_visible_faces
    (d1 d2 d3 d4 : Fin 6 → Nat)
    (visible_faces : List Nat)
    (hv : visible_faces = [1, 2, 3, 4, 4, 5, 5, 6]) :
    let total_sum := 4 * 21
    let visible_sum := List.sum visible_faces
    total_sum - visible_sum = 54 := by
  sorry

end sum_of_non_visible_faces_l1529_152996


namespace lisa_flight_time_l1529_152915

noncomputable def distance : ℝ := 519.5
noncomputable def speed : ℝ := 54.75
noncomputable def time : ℝ := 9.49

theorem lisa_flight_time : distance / speed = time :=
by
  sorry

end lisa_flight_time_l1529_152915


namespace percentage_x_eq_six_percent_y_l1529_152947

variable {x y : ℝ}

theorem percentage_x_eq_six_percent_y (h1 : ∃ P : ℝ, (P / 100) * x = (6 / 100) * y)
  (h2 : (18 / 100) * x = (9 / 100) * y) : 
  ∃ P : ℝ, P = 12 := 
sorry

end percentage_x_eq_six_percent_y_l1529_152947


namespace cost_of_traveling_roads_is_2600_l1529_152946

-- Define the lawn, roads, and the cost parameters
def width_lawn : ℝ := 80
def length_lawn : ℝ := 60
def road_width : ℝ := 10
def cost_per_sq_meter : ℝ := 2

-- Area calculations
def area_road_1 : ℝ := road_width * length_lawn
def area_road_2 : ℝ := road_width * width_lawn
def area_intersection : ℝ := road_width * road_width

def total_area_roads : ℝ := area_road_1 + area_road_2 - area_intersection

def total_cost : ℝ := total_area_roads * cost_per_sq_meter

theorem cost_of_traveling_roads_is_2600 :
  total_cost = 2600 :=
by
  sorry

end cost_of_traveling_roads_is_2600_l1529_152946


namespace simplify_expression_l1529_152943

noncomputable def p (a b c x k : ℝ) := 
  k * (((x + a) ^ 2 / ((a - b) * (a - c))) +
       ((x + b) ^ 2 / ((b - a) * (b - c))) +
       ((x + c) ^ 2 / ((c - a) * (c - b))))

theorem simplify_expression (a b c k : ℝ) (h₀ : a ≠ b) (h₁ : a ≠ c) (h₂ : b ≠ c) (h₃ : k ≠ 0) :
  p a b c x k = k :=
sorry

end simplify_expression_l1529_152943


namespace arithmetic_sequence_problem_l1529_152950

-- Define the arithmetic sequence
def arithmetic_sequence (a : ℕ → ℝ) (a1 d : ℝ) : Prop :=
∀ n, a n = a1 + (n - 1) * d

-- Given condition
def given_condition (a : ℕ → ℝ) (a1 d : ℝ) : Prop :=
3 * a 9 - a 15 - a 3 = 20

-- Question to prove
def question (a : ℕ → ℝ) (a1 d : ℝ) : Prop :=
2 * a 8 - a 7 = 20

-- Main theorem
theorem arithmetic_sequence_problem (a: ℕ → ℝ) (a1 d: ℝ):
  arithmetic_sequence a a1 d →
  given_condition a a1 d →
  question a a1 d :=
by
  sorry

end arithmetic_sequence_problem_l1529_152950


namespace sequence_value_a8_b8_l1529_152925

theorem sequence_value_a8_b8
(a b : ℝ) 
(h1 : a + b = 1) 
(h2 : a^2 + b^2 = 3) 
(h3 : a^3 + b^3 = 4) 
(h4 : a^4 + b^4 = 7) 
(h5 : a^5 + b^5 = 11) 
(h6 : a^6 + b^6 = 18) : 
a^8 + b^8 = 47 :=
sorry

end sequence_value_a8_b8_l1529_152925


namespace malou_average_score_l1529_152978

def quiz1_score := 91
def quiz2_score := 90
def quiz3_score := 92

def sum_of_scores := quiz1_score + quiz2_score + quiz3_score
def number_of_quizzes := 3

theorem malou_average_score : sum_of_scores / number_of_quizzes = 91 :=
by
  sorry

end malou_average_score_l1529_152978


namespace terminal_side_of_half_angle_quadrant_l1529_152997

def is_angle_in_third_quadrant (α : ℝ) (k : ℤ) : Prop :=
  k * 360 + 180 < α ∧ α < k * 360 + 270

def is_terminal_side_of_half_angle_in_quadrant (α : ℝ) : Prop :=
  (∃ n : ℤ, n * 360 + 90 < α / 2 ∧ α / 2 < n * 360 + 135) ∨ 
  (∃ n : ℤ, n * 360 + 270 < α / 2 ∧ α / 2 < n * 360 + 315)

theorem terminal_side_of_half_angle_quadrant (α : ℝ) (k : ℤ) :
  is_angle_in_third_quadrant α k → is_terminal_side_of_half_angle_in_quadrant α := 
sorry

end terminal_side_of_half_angle_quadrant_l1529_152997


namespace exists_solution_l1529_152952

noncomputable def smallest_c0 (a b : ℕ) (ha : 0 < a) (hb : 0 < b) (h : Nat.gcd a b = 1) : ℕ :=
  a * b - a - b + 1

theorem exists_solution (a b : ℕ) (ha : 0 < a) (hb : 0 < b) (h : Nat.gcd a b = 1) :
  ∃ c0, (c0 = smallest_c0 a b ha hb h) ∧ ∀ c : ℕ, c ≥ c0 → ∃ x y : ℕ, a * x + b * y = c :=
sorry

end exists_solution_l1529_152952


namespace sale_price_lower_by_2_5_percent_l1529_152963

open Real

theorem sale_price_lower_by_2_5_percent (x : ℝ) : 
  let increased_price := 1.30 * x
  let sale_price := 0.75 * increased_price 
  sale_price = 0.975 * x :=
by
  let increased_price := 1.30 * x
  let sale_price := 0.75 * increased_price 
  show sale_price = 0.975 * x
  sorry

end sale_price_lower_by_2_5_percent_l1529_152963


namespace place_value_ratio_l1529_152932

theorem place_value_ratio :
  let val_6 := 1000
  let val_2 := 0.1
  val_6 / val_2 = 10000 :=
by
  -- the proof would go here
  sorry

end place_value_ratio_l1529_152932


namespace segment_halving_1M_l1529_152994

noncomputable def segment_halving_sum (k : ℕ) : ℕ :=
  3^k + 1

theorem segment_halving_1M : segment_halving_sum 1000000 = 3^1000000 + 1 :=
by
  sorry

end segment_halving_1M_l1529_152994


namespace reciprocal_geometric_sum_l1529_152929

variable (n : ℕ) (r s : ℝ)
variable (h_r_nonzero : r ≠ 0)
variable (h_sum_original : (1 - r^(2 * n)) / (1 - r^2) = s^3)

theorem reciprocal_geometric_sum (n : ℕ) (r s : ℝ) (h_r_nonzero : r ≠ 0)
  (h_sum_original : (1 - r^(2 * n)) / (1 - r^2) = s^3) :
  ((1 - (1 / r^2)^n) / (1 - 1 / r^2)) = s^3 / r^2 :=
sorry

end reciprocal_geometric_sum_l1529_152929


namespace opposite_of_fraction_reciprocal_of_fraction_absolute_value_of_fraction_l1529_152957

def improper_fraction : ℚ := -4/3

theorem opposite_of_fraction : -improper_fraction = 4/3 :=
by sorry

theorem reciprocal_of_fraction : (improper_fraction⁻¹) = -3/4 :=
by sorry

theorem absolute_value_of_fraction : |improper_fraction| = 4/3 :=
by sorry

end opposite_of_fraction_reciprocal_of_fraction_absolute_value_of_fraction_l1529_152957


namespace student_B_speed_l1529_152970

theorem student_B_speed (d : ℝ) (ratio : ℝ) (t_diff : ℝ) (sB : ℝ) : 
  d = 12 → ratio = 1.2 → t_diff = 1/6 → 
  (d / sB - t_diff = d / (ratio * sB)) → 
  sB = 12 :=
by
  intros h1 h2 h3 h4
  sorry

end student_B_speed_l1529_152970


namespace least_N_l1529_152910

theorem least_N :
  ∃ N : ℕ, 
    (N % 2 = 1) ∧ 
    (N % 3 = 2) ∧ 
    (N % 5 = 3) ∧ 
    (N % 7 = 4) ∧ 
    (∀ M : ℕ, 
      (M % 2 = 1) ∧ 
      (M % 3 = 2) ∧ 
      (M % 5 = 3) ∧ 
      (M % 7 = 4) → 
      N ≤ M) :=
  sorry

end least_N_l1529_152910


namespace min_value_frac_sqrt_l1529_152944

theorem min_value_frac_sqrt (x : ℝ) (h : x > 1) : 
  (x + 10) / Real.sqrt (x - 1) ≥ 2 * Real.sqrt 11 :=
sorry

end min_value_frac_sqrt_l1529_152944


namespace boy_speed_in_kmph_l1529_152908

-- Define the conditions
def side_length : ℕ := 35
def time_seconds : ℕ := 56

-- Perimeter of the square field
def perimeter : ℕ := 4 * side_length

-- Speed in meters per second
def speed_mps : ℚ := perimeter / time_seconds

-- Speed in kilometers per hour
def speed_kmph : ℚ := speed_mps * (3600 / 1000)

-- Theorem stating the boy's speed is 9 km/hr
theorem boy_speed_in_kmph : speed_kmph = 9 :=
by
  sorry

end boy_speed_in_kmph_l1529_152908


namespace max_value_of_x_and_y_l1529_152990

theorem max_value_of_x_and_y (x y : ℕ) 
  (hx : 0 < x) 
  (hy : 0 < y) 
  (h : (x - 4) * (x - 10) = 2 ^ y) : x + y ≤ 16 :=
sorry

end max_value_of_x_and_y_l1529_152990


namespace exchanges_divisible_by_26_l1529_152912

variables (p a d : ℕ) -- Define the variables for the number of exchanges

theorem exchanges_divisible_by_26 (t : ℕ) (h1 : p = 4 * a + d) (h2 : p = a + 5 * d) :
  ∃ k : ℕ, a + p + d = 26 * k :=
by {
  -- Replace these sorry placeholders with the actual proof where needed
  sorry
}

end exchanges_divisible_by_26_l1529_152912


namespace equalize_money_l1529_152969

theorem equalize_money (ann_money : ℕ) (bill_money : ℕ) : 
  ann_money = 777 → 
  bill_money = 1111 → 
  ∃ x, bill_money - x = ann_money + x :=
by
  sorry

end equalize_money_l1529_152969


namespace ascending_order_proof_l1529_152975

noncomputable def frac1 : ℚ := 1 / 2
noncomputable def frac2 : ℚ := 3 / 4
noncomputable def frac3 : ℚ := 1 / 5
noncomputable def dec1 : ℚ := 0.25
noncomputable def dec2 : ℚ := 0.42

theorem ascending_order_proof :
  frac3 < dec1 ∧ dec1 < dec2 ∧ dec2 < frac1 ∧ frac1 < frac2 :=
by {
  -- The proof will show the conversions mentioned in solution steps
  sorry
}

end ascending_order_proof_l1529_152975


namespace power_mod_l1529_152938

theorem power_mod (h : 5 ^ 200 ≡ 1 [MOD 1000]) : 5 ^ 6000 ≡ 1 [MOD 1000] :=
by
  sorry

end power_mod_l1529_152938


namespace area_of_triangle_l1529_152927

noncomputable def circumradius (a b c : ℝ) (α : ℝ) : ℝ := a / (2 * Real.sin α)

theorem area_of_triangle (A B C a b c R : ℝ) (h₁ : b * Real.cos C + c * Real.cos B = Real.sqrt 3 * R)
  (h₂ : a = 2) (h₃ : b + c = 4) : 
  1 / 2 * b * (c * Real.sin A) = Real.sqrt 3 :=
by
  sorry

end area_of_triangle_l1529_152927


namespace find_exp_l1529_152992

noncomputable def a : ℝ := sorry
noncomputable def m : ℤ := sorry
noncomputable def n : ℤ := sorry

axiom a_m_eq_six : a ^ m = 6
axiom a_n_eq_six : a ^ n = 6

theorem find_exp : a ^ (2 * m - n) = 6 :=
by
  sorry

end find_exp_l1529_152992


namespace complex_division_l1529_152911

theorem complex_division (i : ℂ) (h_i : i * i = -1) : (3 - 4 * i) / i = 4 - 3 * i :=
by
  sorry

end complex_division_l1529_152911


namespace douglas_votes_percentage_l1529_152903

theorem douglas_votes_percentage 
  (V : ℝ)
  (hx : 0.62 * 2 * V + 0.38 * V = 1.62 * V)
  (hy : 3 * V > 0) : 
  ((1.62 * V) / (3 * V)) * 100 = 54 := 
by
  sorry

end douglas_votes_percentage_l1529_152903


namespace bernardo_wins_at_5_l1529_152901

theorem bernardo_wins_at_5 :
  ∃ N : ℕ, 0 ≤ N ∧ N ≤ 499 ∧ 27 * N + 360 < 500 ∧ ∀ M : ℕ, (0 ≤ M ∧ M ≤ 499 ∧ 27 * M + 360 < 500 → N ≤ M) :=
by
  sorry

end bernardo_wins_at_5_l1529_152901


namespace inequality_proof_l1529_152985

variable (a b c d e p q : ℝ)

theorem inequality_proof
  (h₀ : 0 < p)
  (h₁ : p ≤ a) (h₂ : a ≤ q)
  (h₃ : p ≤ b) (h₄ : b ≤ q)
  (h₅ : p ≤ c) (h₆ : c ≤ q)
  (h₇ : p ≤ d) (h₈ : d ≤ q)
  (h₉ : p ≤ e) (h₁₀ : e ≤ q) :
  (a + b + c + d + e) * (1 / a + 1 / b + 1 / c + 1 / d + 1 / e) 
  ≤ 25 + 6 * (Real.sqrt (p / q) - Real.sqrt (q / p))^2 := 
by
  sorry -- The actual proof will be filled here

end inequality_proof_l1529_152985


namespace find_m_l1529_152919

theorem find_m 
  (h : ∀ x, (0 < x ∧ x < 2) ↔ ( - (1 / 2) * x^2 + 2 * x > m * x )) :
  m = 1 :=
sorry

end find_m_l1529_152919


namespace days_in_month_l1529_152945

-- The number of days in the month
variable (D : ℕ)

-- The conditions provided in the problem
def mean_daily_profit (D : ℕ) := 350
def mean_first_fifteen_days := 225
def mean_last_fifteen_days := 475
def total_profit := mean_first_fifteen_days * 15 + mean_last_fifteen_days * 15

-- The Lean statement to prove the number of days in the month
theorem days_in_month : D = 30 :=
by
  -- mean_daily_profit(D) * D should be equal to total_profit
  have h : mean_daily_profit D * D = total_profit := sorry
  -- solve for D
  sorry

end days_in_month_l1529_152945


namespace cars_with_neither_feature_l1529_152993

theorem cars_with_neither_feature 
  (total_cars : ℕ) 
  (power_steering : ℕ) 
  (power_windows : ℕ) 
  (both_features : ℕ) 
  (h1 : total_cars = 65) 
  (h2 : power_steering = 45) 
  (h3 : power_windows = 25) 
  (h4 : both_features = 17)
  : total_cars - (power_steering + power_windows - both_features) = 12 :=
by
  sorry

end cars_with_neither_feature_l1529_152993
