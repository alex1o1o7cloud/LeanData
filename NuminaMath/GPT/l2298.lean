import Mathlib

namespace marie_profit_l2298_229881

-- Define constants and conditions
def loaves_baked : ℕ := 60
def morning_price : ℝ := 3.00
def discount : ℝ := 0.25
def afternoon_price : ℝ := morning_price * (1 - discount)
def cost_per_loaf : ℝ := 1.00
def donated_loaves : ℕ := 5

-- Define the number of loaves sold and revenue
def morning_loaves : ℕ := loaves_baked / 3
def morning_revenue : ℝ := morning_loaves * morning_price

def remaining_after_morning : ℕ := loaves_baked - morning_loaves
def afternoon_loaves : ℕ := remaining_after_morning / 2
def afternoon_revenue : ℝ := afternoon_loaves * afternoon_price

def remaining_after_afternoon : ℕ := remaining_after_morning - afternoon_loaves
def unsold_loaves : ℕ := remaining_after_afternoon - donated_loaves

-- Define the total revenue and cost
def total_revenue : ℝ := morning_revenue + afternoon_revenue
def total_cost : ℝ := loaves_baked * cost_per_loaf

-- Define the profit
def profit : ℝ := total_revenue - total_cost

-- State the proof problem
theorem marie_profit : profit = 45 := by
  sorry

end marie_profit_l2298_229881


namespace angle_same_terminal_side_l2298_229811

theorem angle_same_terminal_side (α : ℝ) : 
  (∃ k : ℤ, α = k * 360 - 100) ↔ (∃ k : ℤ, α = k * 360 + (-100)) :=
sorry

end angle_same_terminal_side_l2298_229811


namespace equation_of_curve_E_equation_of_line_l_through_origin_intersecting_E_l2298_229815

noncomputable def F₁ : ℝ × ℝ := (-Real.sqrt 3, 0)
noncomputable def F₂ : ℝ × ℝ := (Real.sqrt 3, 0)

def is_ellipse (P : ℝ × ℝ) : Prop :=
  Real.sqrt ((P.1 - F₁.1) ^ 2 + P.2 ^ 2) + Real.sqrt ((P.1 - F₂.1) ^ 2 + P.2 ^ 2) = 4

theorem equation_of_curve_E :
  ∀ P : ℝ × ℝ, is_ellipse P ↔ (P.1 ^ 2 / 4 + P.2 ^ 2 = 1) :=
sorry

def intersects_at_origin (C D : ℝ × ℝ) : Prop :=
  C.1 * D.1 + C.2 * D.2 = 0

theorem equation_of_line_l_through_origin_intersecting_E :
  ∀ (l : ℝ → ℝ) (C D : ℝ × ℝ),
    (l 0 = -2) →
    (∀ P : ℝ × ℝ, is_ellipse P ↔ (P.1, P.2) = (C.1, l C.1) ∨ (P.1, P.2) = (D.1, l D.1)) →
    intersects_at_origin C D →
    (∀ x, l x = 2 * x - 2) ∨ (∀ x, l x = -2 * x - 2) :=
sorry

end equation_of_curve_E_equation_of_line_l_through_origin_intersecting_E_l2298_229815


namespace simplify_expression_l2298_229863

theorem simplify_expression :
  5 * (18 / 7) * (21 / -45) = -6 / 5 := 
sorry

end simplify_expression_l2298_229863


namespace Brian_Frodo_ratio_l2298_229888

-- Definitions from the conditions
def Lily_tennis_balls : Int := 3
def Frodo_tennis_balls : Int := Lily_tennis_balls + 8
def Brian_tennis_balls : Int := 22

-- The proof statement
theorem Brian_Frodo_ratio :
  Brian_tennis_balls / Frodo_tennis_balls = 2 := by
  sorry

end Brian_Frodo_ratio_l2298_229888


namespace bob_pays_more_than_samantha_l2298_229886

theorem bob_pays_more_than_samantha
  (total_slices : ℕ := 12)
  (cost_plain_pizza : ℝ := 12)
  (cost_olives : ℝ := 3)
  (slices_one_third_pizza : ℕ := total_slices / 3)
  (total_cost : ℝ := cost_plain_pizza + cost_olives)
  (cost_per_slice : ℝ := total_cost / total_slices)
  (bob_slices_total : ℕ := slices_one_third_pizza + 3)
  (samantha_slices_total : ℕ := total_slices - bob_slices_total)
  (bob_total_cost : ℝ := bob_slices_total * cost_per_slice)
  (samantha_total_cost : ℝ := samantha_slices_total * cost_per_slice) :
  bob_total_cost - samantha_total_cost = 2.5 :=
by
  sorry

end bob_pays_more_than_samantha_l2298_229886


namespace florist_bouquets_is_36_l2298_229872

noncomputable def florist_bouquets : Prop :=
  let r := 125
  let y := 125
  let o := 125
  let p := 125
  let rk := 45
  let yk := 61
  let ok := 30
  let pk := 40
  let initial_flowers := r + y + o + p
  let total_killed := rk + yk + ok + pk
  let remaining_flowers := initial_flowers - total_killed
  let flowers_per_bouquet := 9
  let bouquets := remaining_flowers / flowers_per_bouquet
  bouquets = 36

theorem florist_bouquets_is_36 : florist_bouquets :=
  by
    sorry

end florist_bouquets_is_36_l2298_229872


namespace find_m_l2298_229856

theorem find_m (m : ℝ) : (∀ x : ℝ, m * x^2 + 2 < 2) ∧ (m^2 + m = 2) → m = -2 :=
by
  sorry

end find_m_l2298_229856


namespace sqrt_one_div_four_is_one_div_two_l2298_229861

theorem sqrt_one_div_four_is_one_div_two : Real.sqrt (1 / 4) = 1 / 2 :=
by
  sorry

end sqrt_one_div_four_is_one_div_two_l2298_229861


namespace cylinder_volume_ratio_l2298_229896

theorem cylinder_volume_ratio (a b : ℕ) (h_dim : (a, b) = (9, 12)) :
  let r₁ := (a : ℝ) / (2 * Real.pi)
  let h₁ := (↑b : ℝ)
  let V₁ := (Real.pi * r₁^2 * h₁)
  let r₂ := (b : ℝ) / (2 * Real.pi)
  let h₂ := (↑a : ℝ)
  let V₂ := (Real.pi * r₂^2 * h₂)
  (if V₂ > V₁ then V₂ / V₁ else V₁ / V₂) = (16 / 3) :=
by {
  sorry
}

end cylinder_volume_ratio_l2298_229896


namespace find_f_100_l2298_229862

theorem find_f_100 (f : ℝ → ℝ) (k : ℝ) (h_nonzero : k ≠ 0) 
(h_func : ∀ x y : ℝ, 0 < x → 0 < y → k * (x * f y - y * f x) = f (x / y)) : 
f 100 = 0 := 
by
  sorry

end find_f_100_l2298_229862


namespace average_of_new_set_l2298_229809

theorem average_of_new_set (s : List ℝ) (h₁ : s.length = 10) (h₂ : (s.sum / 10) = 7) : 
  ((s.map (λ x => x * 12)).sum / 10) = 84 :=
by
  sorry

end average_of_new_set_l2298_229809


namespace rectangle_area_l2298_229887

open Classical

noncomputable def point := {x : ℝ × ℝ // x.1 >= 0 ∧ x.2 >= 0}

structure Triangle :=
  (X Y Z : point)

structure Rectangle :=
  (P Q R S : point)

def height_from (t : Triangle) : ℝ :=
  8

def xz_length (t : Triangle) : ℝ :=
  15

def ps_on_xz (r : Rectangle) (t : Triangle) : Prop :=
  r.S.val.1 = r.P.val.1 ∧ r.S.val.1 = t.X.val.1 ∧ r.S.val.2 = 0 ∧ r.P.val.2 = 0

def pq_is_one_third_ps (r : Rectangle) : Prop :=
  dist r.P.1 r.Q.1 = (1/3) * dist r.P.1 r.S.1

theorem rectangle_area : ∀ (R : Rectangle) (T : Triangle),
  height_from T = 8 → xz_length T = 15 → ps_on_xz R T → pq_is_one_third_ps R →
  (dist R.P.1 R.Q.1) * (dist R.P.1 R.S.1) = 4800/169 :=
by
  intros
  sorry

end rectangle_area_l2298_229887


namespace kate_change_l2298_229835

def candyCost : ℝ := 0.54
def amountGiven : ℝ := 1.00
def change (amountGiven candyCost : ℝ) : ℝ := amountGiven - candyCost

theorem kate_change : change amountGiven candyCost = 0.46 := by
  sorry

end kate_change_l2298_229835


namespace james_total_payment_l2298_229844

noncomputable def total_amount_paid : ℕ :=
  let dirt_bike_count := 3
  let off_road_vehicle_count := 4
  let atv_count := 2
  let moped_count := 5
  let scooter_count := 3
  let dirt_bike_cost := dirt_bike_count * 150
  let off_road_vehicle_cost := off_road_vehicle_count * 300
  let atv_cost := atv_count * 450
  let moped_cost := moped_count * 200
  let scooter_cost := scooter_count * 100
  let registration_dirt_bike := dirt_bike_count * 25
  let registration_off_road_vehicle := off_road_vehicle_count * 25
  let registration_atv := atv_count * 30
  let registration_moped := moped_count * 15
  let registration_scooter := scooter_count * 20
  let maintenance_dirt_bike := dirt_bike_count * 50
  let maintenance_off_road_vehicle := off_road_vehicle_count * 75
  let maintenance_atv := atv_count * 100
  let maintenance_moped := moped_count * 60
  let total_cost_of_vehicles := dirt_bike_cost + off_road_vehicle_cost + atv_cost + moped_cost + scooter_cost
  let total_registration_costs := registration_dirt_bike + registration_off_road_vehicle + registration_atv + registration_moped + registration_scooter
  let total_maintenance_costs := maintenance_dirt_bike + maintenance_off_road_vehicle + maintenance_atv + maintenance_moped
  total_cost_of_vehicles + total_registration_costs + total_maintenance_costs

theorem james_total_payment : total_amount_paid = 5170 := by
  -- The proof would be written here
  sorry

end james_total_payment_l2298_229844


namespace right_triangle_relation_l2298_229851

theorem right_triangle_relation (a b c x : ℝ)
  (h : c^2 = a^2 + b^2)
  (altitude : a * b = c * x) :
  (1 / x^2) = (1 / a^2) + (1 / b^2) :=
sorry

end right_triangle_relation_l2298_229851


namespace num_even_divisors_of_8_l2298_229841

def factorial (n : Nat) : Nat :=
  match n with
  | 0     => 1
  | Nat.succ n' => Nat.succ n' * factorial n'

-- Define the prime factorization of 8!
def prime_factors_eight_factorial : Nat := 2^7 * 3^2 * 5 * 7

-- Definition of an even divisor of 8!
def is_even_divisor (d : Nat) : Prop :=
  d ∣ prime_factors_eight_factorial ∧ 2 ∣ d

-- Calculation of number of even divisors of 8!
def num_even_divisors_8! : Nat :=
  7 * 3 * 2 * 2

theorem num_even_divisors_of_8! :
  num_even_divisors_8! = 84 :=
sorry

end num_even_divisors_of_8_l2298_229841


namespace two_n_plus_m_is_36_l2298_229820

theorem two_n_plus_m_is_36 (n m : ℤ) (h1 : 3 * n - m < 5) (h2 : n + m > 26) (h3 : 3 * m - 2 * n < 46) : 
  2 * n + m = 36 :=
sorry

end two_n_plus_m_is_36_l2298_229820


namespace determine_base_l2298_229892

theorem determine_base (x : ℕ) (h : 2 * x^3 + x + 6 = x^3 + 2 * x + 342) : x = 7 := 
sorry

end determine_base_l2298_229892


namespace line_through_point_intersecting_circle_eq_l2298_229842

theorem line_through_point_intersecting_circle_eq :
  ∃ k l : ℝ, (x + 2*y + 9 = 0 ∨ 2*x - y + 3 = 0) ∧ 
    ∀ L : ℝ × ℝ,  
      (L = (-3, -3)) ∧ (x^2 + y^2 + 4*y - 21 = 0) → 
      (L = (-3,-3) → (x + 2*y + 9 = 0 ∨ 2*x - y + 3 = 0)) := 
sorry

end line_through_point_intersecting_circle_eq_l2298_229842


namespace multiplications_in_three_hours_l2298_229818

theorem multiplications_in_three_hours :
  let rate := 15000  -- multiplications per second
  let seconds_in_three_hours := 3 * 3600  -- seconds in three hours
  let total_multiplications := rate * seconds_in_three_hours
  total_multiplications = 162000000 :=
by
  let rate := 15000
  let seconds_in_three_hours := 3 * 3600
  let total_multiplications := rate * seconds_in_three_hours
  have h : total_multiplications = 162000000 := sorry
  exact h

end multiplications_in_three_hours_l2298_229818


namespace max_x_value_l2298_229826

theorem max_x_value (x y z : ℝ) (h1 : x + y + z = 7) (h2 : x * y + x * z + y * z = 12) : x ≤ 1 :=
by sorry

end max_x_value_l2298_229826


namespace polynomial_coefficients_sum_l2298_229857

theorem polynomial_coefficients_sum :
  ∀ (a_0 a_1 a_2 a_3 a_4 a_5 a_6 a_7 a_8 a_9 : ℤ), 
  (∀ x : ℚ, (3 * x - 2)^9 = a_0 + a_1 * x + a_2 * x^2 + a_3 * x^3 +
                            a_4 * x^4 + a_5 * x^5 + a_6 * x^6 + 
                            a_7 * x^7 + a_8 * x^8 + a_9 * x^9) →
  (a_0 = -512) →
  ((a_0 + a_1 * (1/3) + a_2 * (1/3)^2 + a_3 * (1/3)^3 + 
    a_4 * (1/3)^4 + a_5 * (1/3)^5 + a_6 * (1/3)^6 + 
    a_7 * (1/3)^7 + a_8 * (1/3)^8 + a_9 * (1/3)^9) = -1) →
  (a_1 / 3 + a_2 / 3^2 + a_3 / 3^3 + a_4 / 3^4 + a_5 / 3^5 + 
   a_6 / 3^6 + a_7 / 3^7 + a_8 / 3^8 + a_9 / 3^9 = 511) :=
by 
  -- The proof would go here
  sorry

end polynomial_coefficients_sum_l2298_229857


namespace find_a_for_even_function_l2298_229832

theorem find_a_for_even_function (a : ℝ) (f : ℝ → ℝ) (h : ∀ x : ℝ, f x = (x + 1)*(x - a) ∧ f (-x) = f x) : a = 1 :=
sorry

end find_a_for_even_function_l2298_229832


namespace input_for_output_16_l2298_229813

theorem input_for_output_16 (x : ℝ) (y : ℝ) : 
  (y = (if x < 0 then (x + 1)^2 else (x - 1)^2)) → 
  y = 16 → 
  (x = 5 ∨ x = -5) :=
by sorry

end input_for_output_16_l2298_229813


namespace smallest_multiple_of_18_and_40_l2298_229805

-- Define the conditions
def multiple_of_18 (n : ℕ) : Prop := n % 18 = 0
def multiple_of_40 (n : ℕ) : Prop := n % 40 = 0

-- Prove that the smallest number that meets the conditions is 360
theorem smallest_multiple_of_18_and_40 : ∃ n : ℕ, multiple_of_18 n ∧ multiple_of_40 n ∧ ∀ m : ℕ, (multiple_of_18 m ∧ multiple_of_40 m) → n ≤ m :=
  by
    let n := 360
    -- We have to prove that 360 is the smallest number that is a multiple of both 18 and 40
    sorry

end smallest_multiple_of_18_and_40_l2298_229805


namespace distance_to_Rock_Mist_Mountains_l2298_229895

theorem distance_to_Rock_Mist_Mountains (d_Sky_Falls : ℕ) (multiplier : ℕ) (d_Rock_Mist : ℕ) :
  d_Sky_Falls = 8 → multiplier = 50 → d_Rock_Mist = d_Sky_Falls * multiplier → d_Rock_Mist = 400 :=
by 
  intros h₁ h₂ h₃
  rw [h₁, h₂] at h₃
  exact h₃

end distance_to_Rock_Mist_Mountains_l2298_229895


namespace water_added_l2298_229859

theorem water_added (x : ℝ) (salt_initial_percentage : ℝ) (salt_final_percentage : ℝ) 
   (evap_fraction : ℝ) (salt_added : ℝ) (W : ℝ) 
   (hx : x = 150) (h_initial_salt : salt_initial_percentage = 0.2) 
   (h_final_salt : salt_final_percentage = 1 / 3) 
   (h_evap_fraction : evap_fraction = 1 / 4) 
   (h_salt_added : salt_added = 20) : 
  W = 37.5 :=
by
  sorry

end water_added_l2298_229859


namespace diamonds_balance_emerald_l2298_229850

theorem diamonds_balance_emerald (D E : ℝ) (h1 : 9 * D = 4 * E) (h2 : 9 * D + E = 4 * E) : 3 * D = E := by
  sorry

end diamonds_balance_emerald_l2298_229850


namespace value_of_expression_l2298_229802

theorem value_of_expression (m : ℝ) (α : ℝ) (h : m < 0) (h_M : M = (3 * m, -m)) :
  let sin_alpha := -m / (Real.sqrt 10 * -m)
  let cos_alpha := 3 * m / (Real.sqrt 10 * -m)
  (1 / (2 * sin_alpha * cos_alpha + cos_alpha^2) = 10 / 3) :=
by
  sorry

end value_of_expression_l2298_229802


namespace distinct_balls_boxes_l2298_229824

theorem distinct_balls_boxes :
  (3^5 = 243) :=
by sorry

end distinct_balls_boxes_l2298_229824


namespace differential_savings_l2298_229890

-- Defining conditions given in the problem
def initial_tax_rate : ℝ := 0.45
def new_tax_rate : ℝ := 0.30
def annual_income : ℝ := 48000

-- Statement of the theorem to prove the differential savings
theorem differential_savings : (annual_income * initial_tax_rate) - (annual_income * new_tax_rate) = 7200 := by
  sorry  -- providing the proof is not required

end differential_savings_l2298_229890


namespace stratified_sampling_total_sample_size_l2298_229812

-- Definitions based on conditions
def pure_milk_brands : ℕ := 30
def yogurt_brands : ℕ := 10
def infant_formula_brands : ℕ := 35
def adult_milk_powder_brands : ℕ := 25
def sampled_infant_formula_brands : ℕ := 7

-- The goal is to prove that the total sample size n is 20.
theorem stratified_sampling_total_sample_size : 
  let total_brands := pure_milk_brands + yogurt_brands + infant_formula_brands + adult_milk_powder_brands
  let sampling_fraction := sampled_infant_formula_brands / infant_formula_brands
  let pure_milk_samples := pure_milk_brands * sampling_fraction
  let yogurt_samples := yogurt_brands * sampling_fraction
  let adult_milk_samples := adult_milk_powder_brands * sampling_fraction
  let n := pure_milk_samples + yogurt_samples + sampled_infant_formula_brands + adult_milk_samples
  n = 20 :=
by
  sorry

end stratified_sampling_total_sample_size_l2298_229812


namespace mutually_exclusive_not_opposite_l2298_229882

universe u

-- Define the colors and people involved
inductive Color
| black
| red
| white

inductive Person 
| A
| B
| C

-- Define a function that distributes the cards amongst the people
def distributes (cards : List Color) (people : List Person) : People -> Color :=
  sorry

-- Define events as propositions
def A_gets_red (d : Person -> Color) : Prop :=
  d Person.A = Color.red

def B_gets_red (d : Person -> Color) : Prop :=
  d Person.B = Color.red

-- The main theorem stating the problem
theorem mutually_exclusive_not_opposite 
  (d : Person -> Color)
  (h : A_gets_red d → ¬ B_gets_red d) : 
  ¬ ( ∀ (p : Prop), A_gets_red d ↔ p ) → B_gets_red d :=
sorry

end mutually_exclusive_not_opposite_l2298_229882


namespace area_of_triangle_with_medians_l2298_229821

theorem area_of_triangle_with_medians
  (s_a s_b s_c : ℝ) :
  (∃ t : ℝ, t = (1 / 3 : ℝ) * ((s_a + s_b + s_c) * (s_b + s_c - s_a) * (s_a + s_c - s_b) * (s_a + s_b - s_c)).sqrt) :=
sorry

end area_of_triangle_with_medians_l2298_229821


namespace number_of_people_l2298_229871

-- Define the total number of candy bars
def total_candy_bars : ℝ := 5.0

-- Define the amount of candy each person gets
def candy_per_person : ℝ := 1.66666666699999

-- Define a theorem to state that dividing the total candy bars by candy per person gives 3 people
theorem number_of_people : total_candy_bars / candy_per_person = 3 :=
  by
  -- Proof omitted
  sorry

end number_of_people_l2298_229871


namespace product_of_2020_numbers_even_l2298_229855

theorem product_of_2020_numbers_even (a : ℕ → ℕ) 
  (h : (Finset.sum (Finset.range 2020) a) % 2 = 1) : 
  (Finset.prod (Finset.range 2020) a) % 2 = 0 :=
sorry

end product_of_2020_numbers_even_l2298_229855


namespace simplify_polynomials_l2298_229800

theorem simplify_polynomials (x : ℝ) :
  (3 * x^2 + 8 * x - 5) - (2 * x^2 + 3 * x - 15) = x^2 + 5 * x + 10 :=
by 
  sorry

end simplify_polynomials_l2298_229800


namespace solution_xy_l2298_229848

noncomputable def find_xy (x y : ℚ) : Prop :=
  (x - 10)^2 + (y - 11)^2 + (x - y)^2 = 1 / 3

theorem solution_xy :
  find_xy (10 + 1 / 3) (10 + 2 / 3) :=
by
  sorry

end solution_xy_l2298_229848


namespace range_of_k_l2298_229889

theorem range_of_k (k : ℝ) : 
  (∃ (x₁ x₂ x₃ : ℝ), (x₁ ≠ x₂ ∧ x₂ ≠ x₃ ∧ x₃ ≠ x₁) ∧ 
   (x₁^3 - 3*x₁ = k ∧ x₂^3 - 3*x₂ = k ∧ x₃^3 - 3*x₃ = k)) ↔ (-2 < k ∧ k < 2) :=
sorry

end range_of_k_l2298_229889


namespace equation_of_symmetric_line_l2298_229877

theorem equation_of_symmetric_line
  (a b : ℝ) (a_ne_zero : a ≠ 0) (b_ne_zero : b ≠ 0) :
  (∀ x : ℝ, ∃ y : ℝ, (x = a * y + b)) → (∀ x : ℝ, ∃ y : ℝ, (y = (1/a) * x - (b/a))) :=
by
  sorry

end equation_of_symmetric_line_l2298_229877


namespace find_tangent_value_l2298_229822

noncomputable def tangent_value (a : ℝ) : Prop :=
  ∃ (x₀ y₀ : ℝ), y₀ = x₀ + 1 ∧ y₀ = Real.log (x₀ + a) ∧
  (1 / (x₀ + a) = 1)

theorem find_tangent_value : tangent_value 2 :=
  sorry

end find_tangent_value_l2298_229822


namespace no_such_integers_and_function_l2298_229876

theorem no_such_integers_and_function (f : ℝ → ℝ) (m n : ℤ) (h1 : ∀ x, f (f x) = 2 * f x - x - 2) (h2 : (m : ℝ) ≤ (n : ℝ) ∧ f m = n) : False :=
sorry

end no_such_integers_and_function_l2298_229876


namespace polygon_sides_l2298_229860

theorem polygon_sides (h : ∀ (n : ℕ), 360 / n = 36) : 10 = 10 := by
  sorry

end polygon_sides_l2298_229860


namespace age_ratio_correct_l2298_229880

noncomputable def RahulDeepakAgeRatio : Prop :=
  let R := 20
  let D := 8
  R / D = 5 / 2

theorem age_ratio_correct (R D : ℕ) (h1 : R + 6 = 26) (h2 : D = 8) : RahulDeepakAgeRatio :=
by
  -- Proof omitted
  sorry

end age_ratio_correct_l2298_229880


namespace laptop_price_l2298_229847

theorem laptop_price (cost upfront : ℝ) (upfront_percentage : ℝ) (upfront_eq : upfront = 240) (upfront_percentage_eq : upfront_percentage = 20) : 
  cost = 1200 :=
by
  sorry

end laptop_price_l2298_229847


namespace ben_mms_count_l2298_229810

theorem ben_mms_count (S M : ℕ) (hS : S = 50) (h_diff : S = M + 30) : M = 20 := by
  sorry

end ben_mms_count_l2298_229810


namespace georgina_parrot_days_l2298_229814

theorem georgina_parrot_days
  (total_phrases : ℕ)
  (phrases_per_week : ℕ)
  (initial_phrases : ℕ)
  (phrases_now : total_phrases = 17)
  (teaching_rate : phrases_per_week = 2)
  (initial_known : initial_phrases = 3) :
  (49 : ℕ) = (((17 - 3) / 2) * 7) :=
by
  -- proof will be here
  sorry

end georgina_parrot_days_l2298_229814


namespace q1_correct_q2_correct_l2298_229808

-- Defining the necessary operations
def q1_lhs := 8 / (-2) - (-4) * (-3)
def q2_lhs := (-2) ^ 3 / 4 * (5 - (-3) ^ 2)

-- Theorem statements to prove that they are equal to 8
theorem q1_correct : q1_lhs = 8 := sorry
theorem q2_correct : q2_lhs = 8 := sorry

end q1_correct_q2_correct_l2298_229808


namespace sin_cos_value_sin_plus_cos_value_l2298_229883

noncomputable def given_condition (θ : ℝ) : Prop := 
  (Real.tan θ + 1 / Real.tan θ = 2)

theorem sin_cos_value (θ : ℝ) (h : given_condition θ) : 
  Real.sin θ * Real.cos θ = 1 / 2 :=
sorry

theorem sin_plus_cos_value (θ : ℝ) (h : given_condition θ) : 
  Real.sin θ + Real.cos θ = Real.sqrt 2 ∨ Real.sin θ + Real.cos θ = -Real.sqrt 2 :=
sorry

end sin_cos_value_sin_plus_cos_value_l2298_229883


namespace infinite_n_divisible_by_p_l2298_229843

theorem infinite_n_divisible_by_p (p : ℕ) (hp : Nat.Prime p) : 
  ∃ᶠ n in Filter.atTop, p ∣ (2^n - n) :=
by
  sorry

end infinite_n_divisible_by_p_l2298_229843


namespace shape_volume_to_surface_area_ratio_l2298_229852

/-- 
Define the volume and surface area of our specific shape with given conditions:
1. Five unit cubes in a straight line.
2. An additional cube on top of the second cube.
3. Another cube beneath the fourth cube.

Prove that the ratio of the volume to the surface area is \( \frac{1}{4} \).
-/
theorem shape_volume_to_surface_area_ratio :
  let volume := 7
  let surface_area := 28
  volume / surface_area = 1 / 4 :=
by
  sorry

end shape_volume_to_surface_area_ratio_l2298_229852


namespace range_of_S_l2298_229858

theorem range_of_S (x y : ℝ) (h : 2 * x^2 + 3 * y^2 = 1) (S : ℝ) (hS : S = 3 * x^2 - 2 * y^2) :
  -2 / 3 < S ∧ S ≤ 3 / 2 :=
sorry

end range_of_S_l2298_229858


namespace sum_g_h_l2298_229891

theorem sum_g_h (d g h : ℝ) 
  (h1 : (8 * d^2 - 4 * d + g) * (4 * d^2 + h * d + 7) = 32 * d^4 + (4 * h - 16) * d^3 - (14 * d^2 - 28 * d - 56)) :
  g + h = -8 :=
sorry

end sum_g_h_l2298_229891


namespace sufficient_but_not_necessary_l2298_229801

def sequence_increasing (a : ℕ → ℝ) : Prop :=
  ∀ n, a n < a (n + 1)

def abs_condition (a : ℕ → ℝ) : Prop :=
  ∀ n, a (n + 1) > abs (a n)

theorem sufficient_but_not_necessary (a : ℕ → ℝ) :
  (abs_condition a → sequence_increasing a) ∧ ¬ (sequence_increasing a → abs_condition a) :=
by
  sorry

end sufficient_but_not_necessary_l2298_229801


namespace largest_possible_difference_l2298_229884

theorem largest_possible_difference 
  (weight_A weight_B weight_C : ℝ)
  (hA : 24.9 ≤ weight_A ∧ weight_A ≤ 25.1)
  (hB : 24.8 ≤ weight_B ∧ weight_B ≤ 25.2)
  (hC : 24.7 ≤ weight_C ∧ weight_C ≤ 25.3) :
  ∃ w1 w2 : ℝ, (w1 = weight_C ∧ w2 = weight_C ∧ abs (w1 - w2) = 0.6) :=
by
  sorry

end largest_possible_difference_l2298_229884


namespace jason_seashells_remaining_l2298_229827

-- Define the initial number of seashells Jason found
def initial_seashells : ℕ := 49

-- Define the number of seashells Jason gave to Tim
def seashells_given_to_tim : ℕ := 13

-- Define the number of seashells Jason now has
def seashells_now : ℕ := initial_seashells - seashells_given_to_tim

-- The theorem to prove: 
theorem jason_seashells_remaining : seashells_now = 36 := 
by
  -- Proof steps will go here
  sorry

end jason_seashells_remaining_l2298_229827


namespace circle_equation_l2298_229833

theorem circle_equation (a b r : ℝ) 
    (h₁ : b = -4 * a)
    (h₂ : abs (a + b - 1) / Real.sqrt 2 = r)
    (h₃ : (b + 2) / (a - 3) * (-1) = -1)
    (h₄ : a = 1)
    (h₅ : b = -4)
    (h₆ : r = 2 * Real.sqrt 2) :
    ∀ x y: ℝ, (x - 1) ^ 2 + (y + 4) ^ 2 = 8 := 
by
  intros
  sorry

end circle_equation_l2298_229833


namespace geometric_series_common_ratio_l2298_229874

theorem geometric_series_common_ratio (a S r : ℝ) 
  (hS : S = a / (1 - r)) 
  (h_modified : (a * r^2) / (1 - r) = S / 16) : 
  r = 1/4 ∨ r = -1/4 :=
by
  sorry

end geometric_series_common_ratio_l2298_229874


namespace casey_marathon_time_l2298_229870

theorem casey_marathon_time (C : ℝ) (h : (C + (4 / 3) * C) / 2 = 7) : C = 10.5 :=
by
  sorry

end casey_marathon_time_l2298_229870


namespace calculate_expression_l2298_229868

theorem calculate_expression : (0.0088 * 4.5) / (0.05 * 0.1 * 0.008) = 990 := by
  sorry

end calculate_expression_l2298_229868


namespace even_function_a_eq_one_l2298_229825

noncomputable def f (x a : ℝ) : ℝ := x * Real.log (x + Real.sqrt (a + x ^ 2))

theorem even_function_a_eq_one (a : ℝ) :
  (∀ x : ℝ, f x a = f (-x) a) → a = 1 :=
by
  sorry

end even_function_a_eq_one_l2298_229825


namespace gcd_of_three_numbers_l2298_229869

theorem gcd_of_three_numbers (a b c : ℕ) (h1 : a = 15378) (h2 : b = 21333) (h3 : c = 48906) :
  Nat.gcd (Nat.gcd a b) c = 3 :=
by
  rw [h1, h2, h3]
  sorry

end gcd_of_three_numbers_l2298_229869


namespace quadratic_result_l2298_229846

noncomputable def quadratic_has_two_positive_integer_roots (k p : ℕ) : Prop :=
  ∃ x1 x2 : ℕ, x1 > 0 ∧ x2 > 0 ∧ (k - 1) * x1 * x1 - p * x1 + k = 0 ∧ (k - 1) * x2 * x2 - p * x2 + k = 0

theorem quadratic_result (k p : ℕ) (h1 : k = 2) (h2 : quadratic_has_two_positive_integer_roots k p) :
  k^(k*p) * (p^p + k^k) = 1984 :=
by
  sorry

end quadratic_result_l2298_229846


namespace read_time_proof_l2298_229866

noncomputable def read_time_problem : Prop :=
  ∃ (x y : ℕ), 
    x > 0 ∧
    y = 480 / x ∧
    (y - 5) = 480 / (x + 16) ∧
    y = 15

theorem read_time_proof : read_time_problem := 
sorry

end read_time_proof_l2298_229866


namespace sin_half_pi_plus_A_l2298_229864

theorem sin_half_pi_plus_A (A : Real) (h : Real.cos (Real.pi + A) = -1 / 2) :
  Real.sin (Real.pi / 2 + A) = 1 / 2 := by
  sorry

end sin_half_pi_plus_A_l2298_229864


namespace find_a_l2298_229867

theorem find_a (a : ℕ) (h : a * 2 * 2^3 = 2^6) : a = 4 := 
by 
  sorry

end find_a_l2298_229867


namespace arithmetic_sequence_sum_ratio_l2298_229898

variable {a : ℕ → ℚ}
variable {S : ℕ → ℚ}

-- Definition of arithmetic sequence sum
def arithmeticSum (n : ℕ) : ℚ :=
  (n / 2) * (a 1 + a n)

-- Given condition
axiom condition : (a 6) / (a 5) = 9 / 11

theorem arithmetic_sequence_sum_ratio :
  (S 11) / (S 9) = 1 :=
by
  sorry

end arithmetic_sequence_sum_ratio_l2298_229898


namespace describe_graph_l2298_229831

theorem describe_graph : 
  ∀ (x y : ℝ), x^2 * (x + y + 1) = y^3 * (x + y + 1) ↔ (x^2 = y^3 ∨ y = -x - 1)
:= sorry

end describe_graph_l2298_229831


namespace prove_percent_liquid_X_in_new_solution_l2298_229849

variable (initial_solution total_weight_x total_weight_y total_weight_new)

def percent_liquid_X_in_new_solution : Prop :=
  let liquid_X_in_initial := 0.45 * 12
  let water_in_initial := 0.55 * 12
  let remaining_liquid_X := liquid_X_in_initial
  let remaining_water := water_in_initial - 5
  let liquid_X_in_added := 0.45 * 7
  let water_in_added := 0.55 * 7
  let total_liquid_X := remaining_liquid_X + liquid_X_in_added
  let total_water := remaining_water + water_in_added
  let total_weight := total_liquid_X + total_water
  (total_liquid_X / total_weight) * 100 = 61.07

theorem prove_percent_liquid_X_in_new_solution :
  percent_liquid_X_in_new_solution := by
  sorry

end prove_percent_liquid_X_in_new_solution_l2298_229849


namespace max_modulus_z_i_l2298_229899

open Complex

theorem max_modulus_z_i (z : ℂ) (hz : abs z = 2) : ∃ z₂ : ℂ, abs z₂ = 2 ∧ abs (z₂ - I) = 3 :=
sorry

end max_modulus_z_i_l2298_229899


namespace minimize_sum_of_distances_l2298_229838

theorem minimize_sum_of_distances (P : ℝ × ℝ) (A : ℝ × ℝ) (F : ℝ × ℝ) 
  (hP_on_parabola : P.2 ^ 2 = 2 * P.1)
  (hA : A = (3, 2)) 
  (hF : F = (1/2, 0)) : 
  |P - A| + |P - F| ≥ |(2, 2) - A| + |(2, 2) - F| :=
by sorry

end minimize_sum_of_distances_l2298_229838


namespace triangle_altitude_l2298_229830

theorem triangle_altitude {A b h : ℝ} (hA : A = 720) (hb : b = 40) (hArea : A = 1 / 2 * b * h) : h = 36 :=
by
  sorry

end triangle_altitude_l2298_229830


namespace selling_price_correct_l2298_229878

theorem selling_price_correct (C P_rate : ℝ) (hC : C = 50) (hP_rate : P_rate = 0.40) : 
  C + (P_rate * C) = 70 :=
by
  sorry

end selling_price_correct_l2298_229878


namespace tan_value_l2298_229834

theorem tan_value (α : ℝ) (h1 : α ∈ (Set.Ioo (π/2) π)) (h2 : Real.sin α = 4/5) : Real.tan α = -4/3 :=
sorry

end tan_value_l2298_229834


namespace small_branches_per_branch_l2298_229839

theorem small_branches_per_branch (x : ℕ) (h1 : 1 + x + x^2 = 57) : x = 7 :=
by {
  sorry
}

end small_branches_per_branch_l2298_229839


namespace parabola_constants_sum_l2298_229836

-- Definition based on the given conditions
structure Parabola where
  a: ℝ
  b: ℝ
  c: ℝ
  vertex_x: ℝ
  vertex_y: ℝ
  point_x: ℝ
  point_y: ℝ

-- Definitions of the specific parabola based on the problem's conditions
noncomputable def givenParabola : Parabola := {
  a := -1/4,
  b := -5/2,
  c := -1/4,
  vertex_x := 6,
  vertex_y := -5,
  point_x := 2,
  point_y := -1
}

-- Theorem proving the required value of a + b + c
theorem parabola_constants_sum : givenParabola.a + givenParabola.b + givenParabola.c = -3.25 :=
  by
  sorry

end parabola_constants_sum_l2298_229836


namespace larger_number_hcf_lcm_l2298_229873

theorem larger_number_hcf_lcm (a b : ℕ) (hcf : ℕ) (factor1 factor2 : ℕ) 
  (h_hcf : hcf = 20) 
  (h_factor1 : factor1 = 13) 
  (h_factor2 : factor2 = 14) 
  (h_ab_hcf : Nat.gcd a b = hcf)
  (h_ab_lcm : Nat.lcm a b = hcf * factor1 * factor2) :
  max a b = 280 :=
by 
  sorry

end larger_number_hcf_lcm_l2298_229873


namespace email_sequence_correct_l2298_229845

theorem email_sequence_correct :
    ∀ (a b c d e f : Prop),
    (a → (e → (b → (c → (d → f))))) :=
by 
  sorry

end email_sequence_correct_l2298_229845


namespace remainder_52_l2298_229893

theorem remainder_52 (x y : ℕ) (k m : ℤ)
  (h₁ : x = 246 * k + 37)
  (h₂ : y = 357 * m + 53) :
  (x + y + 97) % 123 = 52 := by
  sorry

end remainder_52_l2298_229893


namespace solve_ineq_case1_solve_ineq_case2_l2298_229840

theorem solve_ineq_case1 {a x : ℝ} (ha_pos : 0 < a) (ha_lt_one : a < 1) : 
  a^(x + 5) < a^(4 * x - 1) ↔ x < 2 :=
sorry

theorem solve_ineq_case2 {a x : ℝ} (ha_gt_one : a > 1) : 
  a^(x + 5) < a^(4 * x - 1) ↔ x > 2 :=
sorry

end solve_ineq_case1_solve_ineq_case2_l2298_229840


namespace max_trig_expression_l2298_229894

open Real

theorem max_trig_expression (x y z : ℝ) :
  (sin (2 * x) + sin y + sin (3 * z)) * (cos (2 * x) + cos y + cos (3 * z)) ≤ 4.5 := sorry

end max_trig_expression_l2298_229894


namespace move_point_right_l2298_229853

theorem move_point_right (x y : ℝ) (h₁ : x = 1) (h₂ : y = 1) (dx : ℝ) (h₃ : dx = 2) : (x + dx, y) = (3, 1) :=
by
  rw [h₁, h₂, h₃]
  simp
  sorry

end move_point_right_l2298_229853


namespace quadratic_min_value_l2298_229803

theorem quadratic_min_value (k : ℝ) :
  (∀ x : ℝ, 3 ≤ x ∧ x ≤ 5 → y = (1/2) * (x - 1) ^ 2 + k) ∧
  (∀ y : ℝ, 3 ≤ y ∧ y ≤ 5 → y ≥ 3) → k = 1 :=
sorry

end quadratic_min_value_l2298_229803


namespace number_of_proper_subsets_l2298_229854

theorem number_of_proper_subsets (S : Finset ℕ) (h : S = {1, 2, 3, 4}) : S.powerset.card - 1 = 15 := by
  sorry

end number_of_proper_subsets_l2298_229854


namespace inequality_sol_range_t_l2298_229879

def f (x : ℝ) : ℝ := abs (2 * x + 1) - abs (x - 2)

theorem inequality_sol : {x : ℝ | f x > 2} = {x : ℝ | x < -5} ∪ {x : ℝ | 1 < x} :=
sorry

theorem range_t (t : ℝ) : (∀ x : ℝ, f x ≥ t^2 - 11/2 * t) ↔ (1/2 ≤ t ∧ t ≤ 5) :=
sorry

end inequality_sol_range_t_l2298_229879


namespace expr_for_pos_x_min_value_l2298_229837

section
variable {f : ℝ → ℝ}
variable {a : ℝ}

def even_func (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)
def func_def (f : ℝ → ℝ) (a : ℝ) : Prop := ∀ x, x ≤ 0 → f x = 4^(-x) - a * 2^(-x)

-- Assuming f is even and specified as in the problem for x ≤ 0
axiom ev_func : even_func f
axiom f_condition : 0 < a

theorem expr_for_pos_x (f : ℝ → ℝ) (a : ℝ) (h1 : even_func f) (h2 : func_def f a) : 
  ∀ x, 0 < x → f x = 4^x - a * 2^x :=
sorry -- this aims to prove the function's form for positive x.

theorem min_value (f : ℝ → ℝ) (a : ℝ) (h1 : even_func f) (h2 : func_def f a) :
  (0 < a ∧ a ≤ 2 → ∃ x, 0 < x ∧ f x = 1 - a) ∧
  (2 < a → ∃ x, 0 < x ∧ f x = -a^2 / 4) :=
sorry -- this aims to prove the minimum value on the interval (0, +∞).
end

end expr_for_pos_x_min_value_l2298_229837


namespace proof_minimum_value_l2298_229807

noncomputable def minimum_value_inequality (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a + 2 * b = 1) : Prop :=
  (1 / a + a / b) ≥ 1 + 2 * Real.sqrt 2

theorem proof_minimum_value (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a + 2 * b = 1) : minimum_value_inequality a b h1 h2 h3 :=
  by
    sorry

end proof_minimum_value_l2298_229807


namespace quadratic_inequality_solution_range_l2298_229816

open Set Real

theorem quadratic_inequality_solution_range
  (a : ℝ) : (∃ (x1 x2 : ℤ), x1 ≠ x2 ∧ (∀ x : ℝ, x^2 - a * x + 2 * a < 0 ↔ ↑x1 < x ∧ x < ↑x2)) ↔ 
    (a ∈ Icc (-1 : ℝ) ((-1:ℝ)/3)) ∨ (a ∈ Ioo (25 / 3 : ℝ) 9) :=
sorry

end quadratic_inequality_solution_range_l2298_229816


namespace trig_identity_l2298_229885

variable (α : ℝ)
variable (h : Real.sin α = 3 / 5)

theorem trig_identity : Real.sin (Real.pi / 2 + 2 * α) = 7 / 25 :=
by
  sorry

end trig_identity_l2298_229885


namespace number_of_cubes_l2298_229806

theorem number_of_cubes (L W H V_cube : ℝ) (L_eq : L = 9) (W_eq : W = 12) (H_eq : H = 3) (V_cube_eq : V_cube = 3) :
  L * W * H / V_cube = 108 :=
by
  sorry

end number_of_cubes_l2298_229806


namespace hurricane_damage_in_euros_l2298_229897

-- Define the conditions
def usd_damage : ℝ := 45000000  -- Damage in US dollars
def exchange_rate : ℝ := 0.9    -- Exchange rate from US dollars to Euros

-- Define the target value in Euros
def eur_damage : ℝ := 40500000  -- Expected damage in Euros

-- The theorem to prove
theorem hurricane_damage_in_euros :
  usd_damage * exchange_rate = eur_damage :=
by
  sorry

end hurricane_damage_in_euros_l2298_229897


namespace find_a_l2298_229817

noncomputable def point1 : ℝ × ℝ := (-3, 6)
noncomputable def point2 : ℝ × ℝ := (2, -1)

theorem find_a (a : ℝ) :
  let direction : ℝ × ℝ := (point2.1 - point1.1, point2.2 - point1.2)
  direction = (5, -7) →
  let normalized_direction : ℝ × ℝ := (direction.1 / -7, direction.2 / -7)
  normalized_direction = (a, -1) →
  a = -5 / 7 :=
by 
  intros 
  sorry

end find_a_l2298_229817


namespace min_distance_exists_l2298_229829

open Real

-- Define the distance formula function
noncomputable def distance (x : ℝ) : ℝ :=
sqrt ((x - 1) ^ 2 + (3 - 2 * x) ^ 2 + (3 * x - 3) ^ 2)

theorem min_distance_exists :
  ∃ (x : ℝ), distance x = sqrt (14 * x^2 - 32 * x + 19) ∧
               ∀ y, distance y ≥ (sqrt 35) / 7 :=
sorry

end min_distance_exists_l2298_229829


namespace rectangle_area_expression_l2298_229875

theorem rectangle_area_expression {d x : ℝ} (h : d^2 = 29 * x^2) :
  ∃ k : ℝ, (5 * x) * (2 * x) = k * d^2 ∧ k = (10 / 29) :=
by {
 sorry
}

end rectangle_area_expression_l2298_229875


namespace composite_of_squares_l2298_229804

theorem composite_of_squares (n : ℕ) (h1 : 8 * n + 1 = x^2) (h2 : 24 * n + 1 = y^2) (h3 : n > 1) : ∃ a b : ℕ, a ∣ (8 * n + 3) ∧ b ∣ (8 * n + 3) ∧ a ≠ 1 ∧ b ≠ 1 ∧ a ≠ (8 * n + 3) ∧ b ≠ (8 * n + 3) := by
  sorry

end composite_of_squares_l2298_229804


namespace solve_fraction_l2298_229819

theorem solve_fraction :
  (144^2 - 100^2) / 22 = 488 := 
by 
  sorry

end solve_fraction_l2298_229819


namespace abs_neg_three_eq_three_l2298_229865

theorem abs_neg_three_eq_three : abs (-3) = 3 :=
by sorry

end abs_neg_three_eq_three_l2298_229865


namespace method_one_cost_eq_300_method_two_cost_eq_300_method_one_more_cost_effective_l2298_229823

noncomputable def method_one_cost (x : ℕ) : ℕ := 120 + 10 * x

noncomputable def method_two_cost (x : ℕ) : ℕ := 15 * x

theorem method_one_cost_eq_300 (x : ℕ) : method_one_cost x = 300 ↔ x = 18 :=
by sorry

theorem method_two_cost_eq_300 (x : ℕ) : method_two_cost x = 300 ↔ x = 20 :=
by sorry

theorem method_one_more_cost_effective (x : ℕ) :
  x ≥ 40 → method_one_cost x < method_two_cost x :=
by sorry

end method_one_cost_eq_300_method_two_cost_eq_300_method_one_more_cost_effective_l2298_229823


namespace normal_level_short_of_capacity_l2298_229828

noncomputable def total_capacity (water_amount : ℕ) (percentage : ℝ) : ℝ :=
  water_amount / percentage

noncomputable def normal_level (water_amount : ℕ) : ℕ :=
  water_amount / 2

theorem normal_level_short_of_capacity (water_amount : ℕ) (percentage : ℝ) (capacity : ℝ) (normal : ℕ) : 
  water_amount = 30 ∧ percentage = 0.75 ∧ capacity = total_capacity water_amount percentage ∧ normal = normal_level water_amount →
  (capacity - ↑normal) = 25 :=
by
  intros h
  sorry

end normal_level_short_of_capacity_l2298_229828
