import Mathlib

namespace total_pennies_donated_l2187_218790

def cassandra_pennies : ℕ := 5000
def james_pennies : ℕ := cassandra_pennies - 276
def total_pennies : ℕ := cassandra_pennies + james_pennies

theorem total_pennies_donated : total_pennies = 9724 := by
  sorry

end total_pennies_donated_l2187_218790


namespace solution_set_l2187_218732

noncomputable def f : ℝ → ℝ := sorry

axiom f_cond1 : ∀ x : ℝ, f x + deriv f x > 1
axiom f_cond2 : f 0 = 4

theorem solution_set (x : ℝ) : e^x * f x > e^x + 3 ↔ x > 0 :=
by sorry

end solution_set_l2187_218732


namespace find_a_l2187_218748

theorem find_a (a : ℤ) (A B : Set ℤ) (hA : A = {0, 1}) (hB : B = {-1, 0, a + 3}) (h : A ⊆ B) : a = -2 := by
  sorry

end find_a_l2187_218748


namespace domain_of_f_l2187_218792

noncomputable def f (x : ℝ) : ℝ := (1 / Real.sqrt (2 - x)) + Real.log (x+1)

theorem domain_of_f : {x : ℝ | (2 - x) > 0 ∧ (x + 1) > 0} = {x : ℝ | -1 < x ∧ x < 2} := 
by
  ext x
  simp
  sorry

end domain_of_f_l2187_218792


namespace points_above_y_eq_x_l2187_218715

theorem points_above_y_eq_x (x y : ℝ) : (y > x) → (y, x) ∈ {p : ℝ × ℝ | p.2 < p.1} :=
by
  intro h
  sorry

end points_above_y_eq_x_l2187_218715


namespace remaining_dimes_l2187_218700

-- Define the initial quantity of dimes Joan had
def initial_dimes : Nat := 5

-- Define the quantity of dimes Joan spent
def dimes_spent : Nat := 2

-- State the theorem we need to prove
theorem remaining_dimes : initial_dimes - dimes_spent = 3 := by
  sorry

end remaining_dimes_l2187_218700


namespace backyard_area_proof_l2187_218727

-- Condition: Walking the length of 40 times covers 1000 meters
def length_times_40_eq_1000 (L: ℝ) : Prop := 40 * L = 1000

-- Condition: Walking the perimeter 8 times covers 1000 meters
def perimeter_times_8_eq_1000 (P: ℝ) : Prop := 8 * P = 1000

-- Given the conditions, we need to find the Length and Width of the backyard
def is_backyard_dimensions (L W: ℝ) : Prop := 
  length_times_40_eq_1000 L ∧ 
  perimeter_times_8_eq_1000 (2 * (L + W))

-- We need to calculate the area
def backyard_area (L W: ℝ) : ℝ := L * W

-- The theorem to prove
theorem backyard_area_proof (L W: ℝ) 
  (h1: length_times_40_eq_1000 L) 
  (h2: perimeter_times_8_eq_1000 (2 * (L + W))) :
  backyard_area L W = 937.5 := 
  by 
    sorry

end backyard_area_proof_l2187_218727


namespace find_other_subject_given_conditions_l2187_218779

theorem find_other_subject_given_conditions :
  ∀ (P C M : ℕ),
  P = 65 →
  (P + C + M) / 3 = 85 →
  (P + M) / 2 = 90 →
  ∃ (S : ℕ), (P + S) / 2 = 70 ∧ S = C :=
by
  sorry

end find_other_subject_given_conditions_l2187_218779


namespace cubic_roots_identity_l2187_218734

noncomputable def roots_of_cubic (a b c : ℝ) : Prop :=
  (5 * a^3 - 2019 * a + 4029 = 0) ∧ 
  (5 * b^3 - 2019 * b + 4029 = 0) ∧ 
  (5 * c^3 - 2019 * c + 4029 = 0)

theorem cubic_roots_identity (a b c : ℝ) (h_roots : roots_of_cubic a b c) : 
  (a + b)^3 + (b + c)^3 + (c + a)^3 = 12087 / 5 :=
by 
  -- proof steps
  sorry

end cubic_roots_identity_l2187_218734


namespace Earl_owes_Fred_l2187_218708

-- Define initial amounts of money each person has
def Earl_initial : ℤ := 90
def Fred_initial : ℤ := 48
def Greg_initial : ℤ := 36

-- Define debts
def Fred_owes_Greg : ℤ := 32
def Greg_owes_Earl : ℤ := 40

-- Define the total money Greg and Earl have together after debts are settled
def Greg_Earl_total_after_debts : ℤ := 130

-- Define the final amounts after debts are settled
def Earl_final (E : ℤ) : ℤ := Earl_initial - E + Greg_owes_Earl
def Fred_final (E : ℤ) : ℤ := Fred_initial + E - Fred_owes_Greg
def Greg_final : ℤ := Greg_initial + Fred_owes_Greg - Greg_owes_Earl

-- Prove that the total money Greg and Earl have together after debts are settled is 130
theorem Earl_owes_Fred (E : ℤ) (H : Greg_final + Earl_final E = Greg_Earl_total_after_debts) : E = 28 := 
by sorry

end Earl_owes_Fred_l2187_218708


namespace power_equality_l2187_218740

theorem power_equality (y : ℕ) (h : 3^y = 81) : 3^(y+3) = 2187 :=
sorry

end power_equality_l2187_218740


namespace average_height_l2187_218798

def heights : List ℕ := [145, 142, 138, 136, 143, 146, 138, 144, 137, 141]

theorem average_height :
  (heights.sum : ℕ) / heights.length = 141 := by
  sorry

end average_height_l2187_218798


namespace line_through_point_equal_intercepts_locus_equidistant_lines_l2187_218728

theorem line_through_point_equal_intercepts (x y : ℝ) (hx : x = 1) (hy : y = 3) :
  (∃ k : ℝ, y = k * x ∧ k = 3) ∨ (∃ a : ℝ, x + y = a ∧ a = 4) :=
sorry

theorem locus_equidistant_lines (x y : ℝ) :
  ∀ (a b : ℝ), (2 * x + 3 * y - a = 0) ∧ (4 * x + 6 * y + b = 0) →
  ∀ b : ℝ, |b + 10| = |b - 8| → b = -9 → 
  4 * x + 6 * y - 9 = 0 :=
sorry

end line_through_point_equal_intercepts_locus_equidistant_lines_l2187_218728


namespace smallest_stable_triangle_side_length_l2187_218720

/-- The smallest possible side length that can appear in any stable triangle with side lengths that 
are multiples of 5, 80, and 112, respectively, is 20. -/
theorem smallest_stable_triangle_side_length {a b c : ℕ} 
  (hab : ∃ k₁, a = 5 * k₁) 
  (hbc : ∃ k₂, b = 80 * k₂) 
  (hac : ∃ k₃, c = 112 * k₃) 
  (abc_triangle : a + b > c ∧ a + c > b ∧ b + c > a) : 
  a = 20 ∨ b = 20 ∨ c = 20 :=
sorry

end smallest_stable_triangle_side_length_l2187_218720


namespace vector_addition_and_scalar_multiplication_l2187_218796

-- Specify the vectors a and b
def a : ℝ × ℝ := (1, 2)
def b : ℝ × ℝ := (-2, 1)

-- Define the theorem we want to prove
theorem vector_addition_and_scalar_multiplication :
  a + 2 • b = (-3, 4) :=
sorry

end vector_addition_and_scalar_multiplication_l2187_218796


namespace shortest_chord_line_intersect_circle_l2187_218766

-- Define the equation of the circle C
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 - 2*x - 3 = 0

-- Define the fixed point
def fixed_point : ℝ × ℝ := (0, 1)

-- Define the center of the circle
def center : ℝ × ℝ := (1, 0)

-- Define the equation of the line l
def line_eq (x y : ℝ) : Prop := x - y + 1 = 0

-- The theorem that needs to be proven
theorem shortest_chord_line_intersect_circle :
  ∃ k : ℝ, ∀ x y : ℝ, (circle_eq x y ∧ y = k * x + 1) ↔ line_eq x y :=
by
  sorry

end shortest_chord_line_intersect_circle_l2187_218766


namespace odd_prime_2wy_factors_l2187_218795

theorem odd_prime_2wy_factors (w y : ℕ) (h1 : Nat.Prime w) (h2 : Nat.Prime y) (h3 : ¬ Even w) (h4 : ¬ Even y) (h5 : w < y) (h6 : Nat.totient (2 * w * y) = 8) :
  w = 3 :=
sorry

end odd_prime_2wy_factors_l2187_218795


namespace eval_expression_at_a_l2187_218777

theorem eval_expression_at_a (a : ℝ) (h : a = 1 / 2) : (2 * a⁻¹ + a⁻¹ / 2) / a = 10 :=
by
  sorry

end eval_expression_at_a_l2187_218777


namespace bus_speed_l2187_218736

theorem bus_speed (t : ℝ) (d : ℝ) (h : t = 42 / 60) (d_eq : d = 35) : d / t = 50 :=
by
  -- Assume
  sorry

end bus_speed_l2187_218736


namespace number_of_toys_l2187_218762

-- Definitions based on conditions
def selling_price : ℝ := 18900
def cost_price_per_toy : ℝ := 900
def gain_per_toy : ℝ := 3 * cost_price_per_toy

-- The number of toys sold
noncomputable def number_of_toys_sold (SP CP gain : ℝ) : ℝ :=
  (SP - gain) / CP

-- The theorem statement to prove
theorem number_of_toys (SP CP gain : ℝ) : number_of_toys_sold SP CP gain = 18 :=
by
  have h1: SP = 18900 := by sorry
  have h2: CP = 900 := by sorry
  have h3: gain = 3 * CP := by sorry
  -- Further steps to establish the proof
  sorry

end number_of_toys_l2187_218762


namespace age_solution_l2187_218739

theorem age_solution (M S : ℕ) (h1 : M = S + 16) (h2 : M + 2 = 2 * (S + 2)) : S = 14 :=
by sorry

end age_solution_l2187_218739


namespace James_will_take_7_weeks_l2187_218775

def pages_per_hour : ℕ := 5
def hours_per_day : ℕ := 4 - 1
def pages_per_day : ℕ := hours_per_day * pages_per_hour
def total_pages : ℕ := 735
def days_to_finish : ℕ := total_pages / pages_per_day
def weeks_to_finish : ℕ := days_to_finish / 7

theorem James_will_take_7_weeks :
  weeks_to_finish = 7 :=
by
  -- You can add the necessary proof steps here
  sorry

end James_will_take_7_weeks_l2187_218775


namespace child_growth_l2187_218788

-- Define variables for heights
def current_height : ℝ := 41.5
def previous_height : ℝ := 38.5

-- Define the problem statement in Lean 4
theorem child_growth :
  current_height - previous_height = 3 :=
by 
  sorry

end child_growth_l2187_218788


namespace apple_price_theorem_l2187_218757

-- Given conditions
def apple_counts : List Nat := [20, 40, 60, 80, 100, 120, 140]

-- Helper function to calculate revenue for a given apple count.
def revenue (apples : Nat) (price_per_batch : Nat) (price_per_leftover : Nat) (batch_size : Nat) : Nat :=
  (apples / batch_size) * price_per_batch + (apples % batch_size) * price_per_leftover

-- Theorem stating that the price per 7 apples is 1 cent and 3 cents per leftover apple ensures equal revenue.
theorem apple_price_theorem : 
  ∀ seller ∈ apple_counts, 
  revenue seller 1 3 7 = 20 :=
by
  intros seller h_seller
  -- Proof will follow here
  sorry

end apple_price_theorem_l2187_218757


namespace lcm_48_180_value_l2187_218711

def lcm_48_180 : ℕ := Nat.lcm 48 180

theorem lcm_48_180_value : lcm_48_180 = 720 :=
by
-- Proof not required, insert sorry
sorry

end lcm_48_180_value_l2187_218711


namespace find_number_l2187_218764

theorem find_number (x : ℕ) (hx : (x / 100) * 100 = 20) : x = 20 :=
sorry

end find_number_l2187_218764


namespace ana_additional_payment_l2187_218729

theorem ana_additional_payment (A B L : ℝ) (h₁ : A < B) (h₂ : A < L) : 
  (A + (B + L - 2 * A) / 3 = ((A + B + L) / 3)) :=
by
  sorry

end ana_additional_payment_l2187_218729


namespace ratio_expression_x_2y_l2187_218725

theorem ratio_expression_x_2y :
  ∀ (x y : ℝ), x / (2 * y) = 27 → (7 * x + 6 * y) / (x - 2 * y) = 96 / 13 :=
by
  intros x y h
  sorry

end ratio_expression_x_2y_l2187_218725


namespace second_bounce_distance_correct_l2187_218738

noncomputable def second_bounce_distance (R v g : ℝ) : ℝ := 2 * R - (2 * v / 3) * (Real.sqrt (R / g))

theorem second_bounce_distance_correct (R v g : ℝ) (hR : R > 0) (hv : v > 0) (hg : g > 0) :
  second_bounce_distance R v g = 2 * R - (2 * v / 3) * (Real.sqrt (R / g)) := 
by
  -- Placeholder for the proof
  sorry

end second_bounce_distance_correct_l2187_218738


namespace purely_imaginary_iff_m_eq_1_l2187_218789

theorem purely_imaginary_iff_m_eq_1 (m : ℝ) :
  (m^2 - 1 = 0 ∧ m + 1 ≠ 0) → m = 1 :=
by
  sorry

end purely_imaginary_iff_m_eq_1_l2187_218789


namespace barbara_shopping_l2187_218761

theorem barbara_shopping :
  let total_paid := 56
  let tuna_cost := 5 * 2
  let water_cost := 4 * 1.5
  let other_goods_cost := total_paid - tuna_cost - water_cost
  other_goods_cost = 40 :=
by
  sorry

end barbara_shopping_l2187_218761


namespace employee_pay_l2187_218702

theorem employee_pay (y : ℝ) (x : ℝ) (h1 : x = 1.2 * y) (h2 : x + y = 700) : y = 318.18 :=
by
  sorry

end employee_pay_l2187_218702


namespace min_val_proof_l2187_218783

noncomputable def minimum_value (x y z: ℝ) := 9 / x + 4 / y + 1 / z

theorem min_val_proof (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (h : x + 2 * y + 3 * z = 12) :
  minimum_value x y z ≥ 49 / 12 :=
by {
  sorry
}

end min_val_proof_l2187_218783


namespace semicircle_inequality_l2187_218782

-- Define the points on the semicircle
variables (A B C D E : ℝ)
-- Define the length function
def length (X Y : ℝ) : ℝ := abs (X - Y)

-- This is the main theorem statement
theorem semicircle_inequality {A B C D E : ℝ} :
  length A B ^ 2 + length B C ^ 2 + length C D ^ 2 + length D E ^ 2 +
  length A B * length B C * length C D + length B C * length C D * length D E < 4 :=
sorry

end semicircle_inequality_l2187_218782


namespace mike_picked_12_pears_l2187_218703

theorem mike_picked_12_pears
  (jason_pears : ℕ)
  (keith_pears : ℕ)
  (total_pears : ℕ)
  (H1 : jason_pears = 46)
  (H2 : keith_pears = 47)
  (H3 : total_pears = 105) :
  (total_pears - (jason_pears + keith_pears)) = 12 :=
by
  sorry

end mike_picked_12_pears_l2187_218703


namespace find_prime_pairs_l2187_218784

open Nat

def is_prime (p : ℕ) : Prop := p > 1 ∧ ∀ d : ℕ, d ∣ p → d = 1 ∨ d = p

theorem find_prime_pairs :
  ∀ m n : ℕ,
  is_prime m → is_prime n → (m < n ∧ n < 5 * m) → is_prime (m + 3 * n) →
  (m = 2 ∧ (n = 3 ∨ n = 5 ∨ n = 7)) :=
by
  sorry

end find_prime_pairs_l2187_218784


namespace rotational_homothety_commutes_l2187_218780

-- Definitions for our conditions
variable (H1 H2 : Point → Point)

-- Definition of rotational homothety. 
-- You would define it based on your bespoke library/formalization.
axiom is_rot_homothety : ∀ (H : Point → Point), Prop

-- Main theorem statement
theorem rotational_homothety_commutes (H1 H2 : Point → Point) (A : Point) 
    (h1_rot : is_rot_homothety H1) (h2_rot : is_rot_homothety H2) : 
    (H1 ∘ H2 = H2 ∘ H1) ↔ (H1 (H2 A) = H2 (H1 A)) :=
sorry

end rotational_homothety_commutes_l2187_218780


namespace taxi_company_charges_l2187_218776

theorem taxi_company_charges
  (X : ℝ)  -- charge for the first 1/5 of a mile
  (C : ℝ)  -- charge for each additional 1/5 of a mile
  (total_charge : ℝ)  -- total charge for an 8-mile ride
  (remaining_distance_miles : ℝ)  -- remaining miles after the first 1/5 mile
  (remaining_increments : ℝ)  -- remaining 1/5 mile increments
  (charge_increments : ℝ)  -- total charge for remaining increments
  (X_val : X = 2.50)
  (C_val : C = 0.40)
  (total_charge_val : total_charge = 18.10)
  (remaining_distance_miles_val : remaining_distance_miles = 7.8)
  (remaining_increments_val : remaining_increments = remaining_distance_miles * 5)
  (charge_increments_val : charge_increments = remaining_increments * C)
  (proof_1: charge_increments = 15.60)
  (proof_2: total_charge - charge_increments = X) : X = 2.50 := 
by
  sorry

end taxi_company_charges_l2187_218776


namespace role_of_scatter_plot_correct_l2187_218741

-- Definitions for problem context
def role_of_scatter_plot (role : String) : Prop :=
  role = "Roughly judging whether variables are linearly related"

-- Problem and conditions
theorem role_of_scatter_plot_correct :
  role_of_scatter_plot "Roughly judging whether variables are linearly related" :=
by 
  sorry

end role_of_scatter_plot_correct_l2187_218741


namespace mrs_hilt_initial_money_l2187_218760

def initial_amount (pencil_cost candy_cost left_money : ℕ) := 
  pencil_cost + candy_cost + left_money

theorem mrs_hilt_initial_money :
  initial_amount 20 5 18 = 43 :=
by
  -- initial_amount 20 5 18 
  -- = 20 + 5 + 18
  -- = 25 + 18 
  -- = 43
  sorry

end mrs_hilt_initial_money_l2187_218760


namespace value_of_hash_l2187_218771

def hash (a b c d : ℝ) : ℝ := b^2 - 4 * a * c * d

theorem value_of_hash : hash 2 3 2 1 = -7 := by
  sorry

end value_of_hash_l2187_218771


namespace sum_of_roots_l2187_218742

open Polynomial

noncomputable def f (a b : ℝ) : Polynomial ℝ := Polynomial.C b + Polynomial.C a * X + X^2
noncomputable def g (c d : ℝ) : Polynomial ℝ := Polynomial.C d + Polynomial.C c * X + X^2

theorem sum_of_roots (a b c d : ℝ)
  (h1 : eval 1 (f a b) = eval 2 (g c d))
  (h2 : eval 1 (g c d) = eval 2 (f a b))
  (hf_roots : ∃ r1 r2 : ℝ, (f a b).roots = {r1, r2})
  (hg_roots : ∃ s1 s2 : ℝ, (g c d).roots = {s1, s2}) :
  (-(a + c) = 6) :=
sorry

end sum_of_roots_l2187_218742


namespace isosceles_right_triangle_area_l2187_218794

-- Define the isosceles right triangle and its properties

theorem isosceles_right_triangle_area 
  (h : ℝ)
  (hyp : h = 6) :
  let l : ℝ := h / Real.sqrt 2
  let A : ℝ := (l^2) / 2
  A = 9 :=
by
  -- The proof steps are skipped with sorry
  sorry

end isosceles_right_triangle_area_l2187_218794


namespace integral_value_l2187_218781

theorem integral_value (a : ℝ) (h : -35 * a^3 = -280) : ∫ x in a..2 * Real.exp 1, 1 / x = 1 := by
  sorry

end integral_value_l2187_218781


namespace possible_new_perimeters_l2187_218706

theorem possible_new_perimeters
  (initial_tiles := 8)
  (initial_shape := "L")
  (initial_perimeter := 12)
  (additional_tiles := 2)
  (new_perimeters := [12, 14, 16]) :
  True := sorry

end possible_new_perimeters_l2187_218706


namespace book_club_meeting_days_l2187_218772

theorem book_club_meeting_days :
  Nat.lcm (Nat.lcm 5 6) (Nat.lcm 8 (Nat.lcm 9 10)) = 360 := 
by sorry

end book_club_meeting_days_l2187_218772


namespace hexagon_interior_angles_l2187_218763

theorem hexagon_interior_angles
  (A B C D E F : ℝ)
  (hA : A = 90)
  (hB : B = 120)
  (hCD : C = D)
  (hE : E = 2 * C + 20)
  (hF : F = 60)
  (hsum : A + B + C + D + E + F = 720) :
  D = 107.5 := 
by
  -- formal proof required here
  sorry

end hexagon_interior_angles_l2187_218763


namespace train_cross_post_time_proof_l2187_218769

noncomputable def train_cross_post_time (speed_kmh : ℝ) (length_m : ℝ) : ℝ :=
  let speed_ms := speed_kmh * 1000 / 3600
  length_m / speed_ms

theorem train_cross_post_time_proof : train_cross_post_time 40 190.0152 = 17.1 := by
  sorry

end train_cross_post_time_proof_l2187_218769


namespace bookseller_original_cost_l2187_218710

theorem bookseller_original_cost
  (x y z : ℝ)
  (h1 : 1.10 * x = 11.00)
  (h2 : 1.10 * y = 16.50)
  (h3 : 1.10 * z = 24.20) :
  x + y + z = 47.00 := by
  sorry

end bookseller_original_cost_l2187_218710


namespace complex_exponential_sum_l2187_218746

theorem complex_exponential_sum (γ δ : ℝ) 
  (h : Complex.exp (Complex.I * γ) + Complex.exp (Complex.I * δ) = -1 / 2 + 5 / 4 * Complex.I) :
  Complex.exp (-Complex.I * γ) + Complex.exp (-Complex.I * δ) = -1 / 2 - 5 / 4 * Complex.I :=
by
  sorry

end complex_exponential_sum_l2187_218746


namespace fraction_meaningful_iff_l2187_218701

theorem fraction_meaningful_iff (x : ℝ) : (∃ y, y = 1 / (x + 1)) ↔ x ≠ -1 :=
by
  sorry

end fraction_meaningful_iff_l2187_218701


namespace total_flag_distance_moved_l2187_218797

def flagpole_length : ℕ := 60

def initial_raise_distance : ℕ := flagpole_length

def lower_to_half_mast_distance : ℕ := flagpole_length / 2

def raise_from_half_mast_distance : ℕ := flagpole_length / 2

def final_lower_distance : ℕ := flagpole_length

theorem total_flag_distance_moved :
  initial_raise_distance + lower_to_half_mast_distance + raise_from_half_mast_distance + final_lower_distance = 180 :=
by
  sorry

end total_flag_distance_moved_l2187_218797


namespace min_omega_l2187_218751

noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := Real.sin (ω * x + 1)
noncomputable def g (ω : ℝ) (x : ℝ) : ℝ := Real.sin (ω * (x - 1) + 1)

def condition1 (ω : ℝ) : Prop := ω > 0
def condition2 (ω : ℝ) (x : ℝ) : Prop := g ω x = Real.sin (ω * x - ω + 1)
def condition3 (ω : ℝ) (k : ℤ) : Prop := ∃ k : ℤ, ω = 1 - k * Real.pi

theorem min_omega (ω : ℝ) (k : ℤ) (x : ℝ) : condition1 ω → condition2 ω x → condition3 ω k → ω = 1 :=
by
  intros h1 h2 h3
  sorry

end min_omega_l2187_218751


namespace james_drive_time_to_canada_l2187_218786

theorem james_drive_time_to_canada : 
  ∀ (distance speed stop_time : ℕ), 
    speed = 60 → 
    distance = 360 → 
    stop_time = 1 → 
    (distance / speed) + stop_time = 7 :=
by
  intros distance speed stop_time h1 h2 h3
  sorry

end james_drive_time_to_canada_l2187_218786


namespace marco_total_time_l2187_218718

def marco_run_time (laps distance1 distance2 speed1 speed2 : ℕ ) : ℝ :=
  let time1 := distance1 / speed1
  let time2 := distance2 / speed2
  laps * (time1 + time2)

theorem marco_total_time :
  marco_run_time 7 150 350 3 4 = 962.5 :=
by
  sorry

end marco_total_time_l2187_218718


namespace double_root_values_l2187_218705

theorem double_root_values (b₃ b₂ b₁ s : ℤ) (h : ∀ x : ℤ, (x * (x - s)) ∣ (x^4 + b₃ * x^3 + b₂ * x^2 + b₁ * x + 36)) 
  : s = -6 ∨ s = -3 ∨ s = -2 ∨ s = -1 ∨ s = 1 ∨ s = 2 ∨ s = 3 ∨ s = 6 :=
sorry

end double_root_values_l2187_218705


namespace units_digit_pow_prod_l2187_218749

theorem units_digit_pow_prod : 
  ((2 ^ 2023) * (5 ^ 2024) * (11 ^ 2025)) % 10 = 0 :=
by
  sorry

end units_digit_pow_prod_l2187_218749


namespace Rob_has_three_dimes_l2187_218743

theorem Rob_has_three_dimes (quarters dimes nickels pennies : ℕ) 
                            (val_quarters val_nickels val_pennies : ℚ)
                            (total_amount : ℚ) :
  quarters = 7 →
  nickels = 5 →
  pennies = 12 →
  val_quarters = 0.25 →
  val_nickels = 0.05 →
  val_pennies = 0.01 →
  total_amount = 2.42 →
  (7 * 0.25 + 5 * 0.05 + 12 * 0.01 + dimes * 0.10 = total_amount) →
  dimes = 3 :=
by sorry

end Rob_has_three_dimes_l2187_218743


namespace solve_for_x_l2187_218714

theorem solve_for_x (x : ℚ) :
  (3 + 1 / (2 + 1 / (3 + 3 / (4 + x)))) = 225 / 73 ↔ x = -647 / 177 :=
by sorry

end solve_for_x_l2187_218714


namespace average_percent_score_is_65_point_25_l2187_218774

theorem average_percent_score_is_65_point_25 :
  let percent_score : List (ℕ × ℕ) := [(95, 10), (85, 20), (75, 40), (65, 50), (55, 60), (45, 15), (35, 5)]
  let total_students : ℕ := 200
  let total_score : ℕ := percent_score.foldl (fun acc p => acc + p.1 * p.2) 0
  (total_score : ℚ) / (total_students : ℚ) = 65.25 := by
{
  sorry
}

end average_percent_score_is_65_point_25_l2187_218774


namespace smallest_x_for_perfect_cube_l2187_218712

theorem smallest_x_for_perfect_cube (x N : ℕ) (hN : 1260 * x = N^3) (h_fact : 1260 = 2^2 * 3^2 * 5 * 7): x = 7350 := sorry

end smallest_x_for_perfect_cube_l2187_218712


namespace remainder_when_divided_by_19_l2187_218707

theorem remainder_when_divided_by_19 {N : ℤ} (h : N % 342 = 47) : N % 19 = 9 :=
sorry

end remainder_when_divided_by_19_l2187_218707


namespace lindy_total_distance_l2187_218745

-- Definitions derived from the conditions
def jack_speed : ℕ := 5
def christina_speed : ℕ := 7
def lindy_speed : ℕ := 12
def initial_distance : ℕ := 360

theorem lindy_total_distance :
  lindy_speed * (initial_distance / (jack_speed + christina_speed)) = 360 := by
  sorry

end lindy_total_distance_l2187_218745


namespace timmy_needs_speed_l2187_218717

variable (s1 s2 s3 : ℕ) (extra_speed : ℕ)

theorem timmy_needs_speed
  (h_s1 : s1 = 36)
  (h_s2 : s2 = 34)
  (h_s3 : s3 = 38)
  (h_extra_speed : extra_speed = 4) :
  (s1 + s2 + s3) / 3 + extra_speed = 40 := 
sorry

end timmy_needs_speed_l2187_218717


namespace integral_solutions_l2187_218724

theorem integral_solutions (a b c : ℤ) (h : a^2 + b^2 + c^2 = a^2 * b^2) : a = 0 ∧ b = 0 ∧ c = 0 :=
sorry

end integral_solutions_l2187_218724


namespace final_rider_is_C_l2187_218713

def initial_order : List Char := ['A', 'B', 'C']

def leader_changes : Nat := 19
def third_place_changes : Nat := 17

def B_finishes_third (final_order: List Char) : Prop :=
  final_order.get! 2 = 'B'

def total_transpositions (a b : Nat) : Nat :=
  a + b

theorem final_rider_is_C (final_order: List Char) :
  B_finishes_third final_order →
  total_transpositions leader_changes third_place_changes % 2 = 0 →
  final_order = ['C', 'A', 'B'] → 
  final_order.get! 0 = 'C' :=
by
  sorry

end final_rider_is_C_l2187_218713


namespace perimeter_of_shaded_area_l2187_218756

theorem perimeter_of_shaded_area (AB AD : ℝ) (h1 : AB = 14) (h2 : AD = 12) : 
  2 * AB + 2 * AD = 52 := 
by
  sorry

end perimeter_of_shaded_area_l2187_218756


namespace sum_of_two_numbers_l2187_218737

theorem sum_of_two_numbers :
  ∀ (A B : ℚ), (A - B = 8) → (1 / 4 * (A + B) = 6) → (A = 16) → (A + B = 24) :=
by
  intros A B h1 h2 h3
  sorry

end sum_of_two_numbers_l2187_218737


namespace polygon_sides_l2187_218733

theorem polygon_sides (n : ℕ) : 
  ((n - 2) * 180 = 4 * 360) → n = 10 :=
by
  sorry

end polygon_sides_l2187_218733


namespace sequence_general_term_l2187_218778

theorem sequence_general_term :
  ∀ (a : ℕ → ℝ), a 1 = 2 ^ (5 / 2) ∧ 
  (∀ n, a (n+1) = 4 * (4 * a n) ^ (1/4)) →
  ∀ n, a n = 2 ^ (10 / 3 * (1 - 1 / 4 ^ n)) :=
by
  intros a h1 h_rec
  sorry

end sequence_general_term_l2187_218778


namespace smallest_positive_e_for_polynomial_l2187_218793

theorem smallest_positive_e_for_polynomial :
  ∃ a b c d e : ℤ, e = 168 ∧
  (a * x ^ 4 + b * x ^ 3 + c * x ^ 2 + d * x + e = 0) ∧
  (a * (x + 3) * (x - 7) * (x - 8) * (4 * x + 1) = a * x ^ 4 + b * x ^ 3 + c * x ^ 2 + d * x + e) := sorry

end smallest_positive_e_for_polynomial_l2187_218793


namespace coordinates_of_D_l2187_218716
-- Importing the necessary library

-- Defining the conditions as given in the problem
def AB : ℝ × ℝ := (5, 3)
def C : ℝ × ℝ := (-1, 3)
def CD : ℝ × ℝ := (2 * 5, 2 * 3)

-- The target proof statement
theorem coordinates_of_D :
  ∃ D : ℝ × ℝ, CD = D - C ∧ D = (9, -3) :=
by
  sorry

end coordinates_of_D_l2187_218716


namespace non_upgraded_sensor_ratio_l2187_218770

theorem non_upgraded_sensor_ratio 
  (N U S : ℕ) 
  (units : ℕ := 24) 
  (fraction_upgraded : ℚ := 1 / 7) 
  (fraction_non_upgraded : ℚ := 6 / 7)
  (h1 : U / S = fraction_upgraded)
  (h2 : units * N = (fraction_non_upgraded * S)) : 
  N / U = 1 / 4 := 
by 
  sorry

end non_upgraded_sensor_ratio_l2187_218770


namespace smallest_number_of_sparrows_in_each_flock_l2187_218726

theorem smallest_number_of_sparrows_in_each_flock (P : ℕ) (H : 14 * P ≥ 182) : 
  ∃ S : ℕ, S = 14 ∧ S ∣ 182 ∧ (∃ P : ℕ, S ∣ (14 * P)) := 
by 
  sorry

end smallest_number_of_sparrows_in_each_flock_l2187_218726


namespace non_congruent_right_triangles_count_l2187_218773

def isRightTriangle (a b c : ℕ) : Prop :=
  a * a + b * b = c * c

def areaEqualsFourTimesPerimeter (a b c : ℕ) : Prop :=
  a * b = 8 * (a + b + c)

theorem non_congruent_right_triangles_count :
  {n : ℕ // ∃ (a b c : ℕ), a > 0 ∧ b > 0 ∧ isRightTriangle a b c ∧ areaEqualsFourTimesPerimeter a b c ∧ n = 3} := sorry

end non_congruent_right_triangles_count_l2187_218773


namespace probability_of_factor_less_than_ten_is_half_l2187_218753

-- Definitions for the factors and counts
def numFactors (n : ℕ) : ℕ :=
  let psa := 1;
  let psb := 2;
  let psc := 1;
  (psa + 1) * (psb + 1) * (psc + 1)

def factorsLessThanTen (n : ℕ) : List ℕ :=
  if n = 90 then [1, 2, 3, 5, 6, 9] else []

def probabilityLessThanTen (n : ℕ) : ℚ :=
  let totalFactors := numFactors n;
  let lessThanTenFactors := factorsLessThanTen n;
  let favorableOutcomes := lessThanTenFactors.length;
  favorableOutcomes / totalFactors

-- The proof statement
theorem probability_of_factor_less_than_ten_is_half :
  probabilityLessThanTen 90 = 1 / 2 := sorry

end probability_of_factor_less_than_ten_is_half_l2187_218753


namespace simple_interest_time_period_l2187_218721

theorem simple_interest_time_period 
  (P : ℝ) (R : ℝ := 4) (T : ℝ) (SI : ℝ := (2 / 5) * P) :
  SI = P * R * T / 100 → T = 10 :=
by {
  sorry
}

end simple_interest_time_period_l2187_218721


namespace ellipse_k_values_l2187_218767

theorem ellipse_k_values (k : ℝ) :
  (∃ a b : ℝ, a = (k + 8) ∧ b = 9 ∧ 
  (b > a → (a * (1 - (1 / 2) ^ 2) = b - a) ∧ k = 4) ∧ 
  (a > b → (b * (1 - (1 / 2) ^ 2) = a - b) ∧ k = -5/4)) :=
sorry

end ellipse_k_values_l2187_218767


namespace Billy_weight_l2187_218765

variables (Billy Brad Carl Dave Edgar : ℝ)

-- Conditions
def conditions :=
  Carl = 145 ∧
  Dave = Carl + 8 ∧
  Brad = Dave / 2 ∧
  Billy = Brad + 9 ∧
  Edgar = 3 * Dave ∧
  Edgar = Billy + 20

-- The statement to prove
theorem Billy_weight (Billy Brad Carl Dave Edgar : ℝ) (h : conditions Billy Brad Carl Dave Edgar) : Billy = 85.5 :=
by
  -- Proof would go here
  sorry

end Billy_weight_l2187_218765


namespace sum_of_coefficients_l2187_218730

/-- If (2x - 1)^4 = a₄x^4 + a₃x^3 + a₂x^2 + a₁x + a₀, then the sum of the coefficients a₀ + a₁ + a₂ + a₃ + a₄ is 1. -/
theorem sum_of_coefficients :
  ∃ a₄ a₃ a₂ a₁ a₀ : ℝ, (2 * x - 1) ^ 4 = a₄ * x ^ 4 + a₃ * x ^ 3 + a₂ * x ^ 2 + a₁ * x + a₀ → 
  a₀ + a₁ + a₂ + a₃ + a₄ = 1 :=
by
  sorry

end sum_of_coefficients_l2187_218730


namespace rhombus_diagonal_length_l2187_218709

theorem rhombus_diagonal_length (area d1 d2 : ℝ) (h₁ : area = 24) (h₂ : d1 = 8) (h₃ : area = (d1 * d2) / 2) : d2 = 6 := 
by sorry

end rhombus_diagonal_length_l2187_218709


namespace number_of_shirts_in_first_batch_minimum_selling_price_per_shirt_l2187_218754

-- Define the conditions
def first_batch_cost : ℝ := 13200
def second_batch_cost : ℝ := 28800
def unit_price_difference : ℝ := 10
def discount_rate : ℝ := 0.8
def profit_margin : ℝ := 1.25
def last_batch_count : ℕ := 50

-- Define the theorem for the first part
theorem number_of_shirts_in_first_batch (x : ℕ) (h₁ : first_batch_cost / x + unit_price_difference = second_batch_cost / (2 * x)) : x = 120 :=
sorry

-- Define the theorem for the second part
theorem minimum_selling_price_per_shirt (x : ℕ) (y : ℝ)
  (h₁ : first_batch_cost / x + unit_price_difference = second_batch_cost / (2 * x))
  (h₂ : x = 120)
  (h₃ : (3 * x - last_batch_count) * y + last_batch_count * discount_rate * y ≥ (first_batch_cost + second_batch_cost) * profit_margin) : y ≥ 150 :=
sorry

end number_of_shirts_in_first_batch_minimum_selling_price_per_shirt_l2187_218754


namespace ball_hits_ground_l2187_218723

theorem ball_hits_ground 
  (y : ℝ → ℝ) 
  (height_eq : ∀ t, y t = -3 * t^2 - 6 * t + 90) :
  ∃ t : ℝ, y t = 0 ∧ t = 5.00 :=
by
  sorry

end ball_hits_ground_l2187_218723


namespace desk_chair_production_l2187_218744

theorem desk_chair_production (x : ℝ) (h₁ : x > 0) (h₂ : 540 / x - 540 / (x + 2) = 3) : 
  ∃ x, 540 / x - 540 / (x + 2) = 3 := 
by
  sorry

end desk_chair_production_l2187_218744


namespace man_is_older_by_20_l2187_218704

variables (M S : ℕ)
axiom h1 : S = 18
axiom h2 : M + 2 = 2 * (S + 2)

theorem man_is_older_by_20 :
  M - S = 20 :=
by {
  sorry
}

end man_is_older_by_20_l2187_218704


namespace no_k_for_linear_function_not_in_second_quadrant_l2187_218791

theorem no_k_for_linear_function_not_in_second_quadrant :
  ¬∃ k : ℝ, ∀ x < 0, (k-1)*x + k ≤ 0 :=
by
  sorry

end no_k_for_linear_function_not_in_second_quadrant_l2187_218791


namespace simplify_fraction_multiplication_l2187_218719

theorem simplify_fraction_multiplication :
  (15/35) * (28/45) * (75/28) = 5/7 :=
by
  sorry

end simplify_fraction_multiplication_l2187_218719


namespace danny_reaches_steve_house_in_31_minutes_l2187_218768

theorem danny_reaches_steve_house_in_31_minutes:
  ∃ (t : ℝ), 2 * t - t = 15.5 * 2 ∧ t = 31 := sorry

end danny_reaches_steve_house_in_31_minutes_l2187_218768


namespace part_a_part_b_l2187_218799

-- Part A: Proving the specific values of p and q
theorem part_a (p q : ℝ) : 
  (∀ x : ℝ, (x + 3) ^ 2 + (7 * x + p) ^ 2 = (kx + m) ^ 2) ∧
  (∀ x : ℝ, (3 * x + 5) ^ 2 + (p * x + q) ^ 2 = (cx + d) ^ 2) → 
  p = 21 ∧ q = 35 :=
sorry

-- Part B: Proving the new polynomial is a square of a linear polynomial
theorem part_b (a b c A B C : ℝ) (hab : a ≠ 0) (hA : A ≠ 0) (hb : b ≠ 0) (hB : B ≠ 0)
  (habc : (∀ x : ℝ, (a * x + b) ^ 2 + (A * x + B) ^ 2 = (kx + m) ^ 2) ∧
         (∀ x : ℝ, (b * x + c) ^ 2 + (B * x + C) ^ 2 = (cx + d) ^ 2)) :
  ∀ x : ℝ, (c * x + a) ^ 2 + (C * x + A) ^ 2 = (lx + n) ^ 2 :=
sorry

end part_a_part_b_l2187_218799


namespace cases_needed_to_raise_funds_l2187_218722

-- Define conditions as lemmas that will be used in the main theorem.
lemma packs_per_case : ℕ := 3
lemma muffins_per_pack : ℕ := 4
lemma muffin_price : ℕ := 2
lemma fundraising_goal : ℕ := 120

-- Calculate muffins per case
noncomputable def muffins_per_case : ℕ := packs_per_case * muffins_per_pack

-- Calculate money earned per case
noncomputable def money_per_case : ℕ := muffins_per_case * muffin_price

-- The main theorem to prove the number of cases needed
theorem cases_needed_to_raise_funds : 
  (fundraising_goal / money_per_case) = 5 :=
by
  sorry

end cases_needed_to_raise_funds_l2187_218722


namespace right_square_pyramid_height_l2187_218735

theorem right_square_pyramid_height :
  ∀ (h x : ℝ),
    let topBaseSide := 3
    let bottomBaseSide := 6
    let lateralArea := 4 * (1/2) * (topBaseSide + bottomBaseSide) * x
    let baseAreasSum := topBaseSide^2 + bottomBaseSide^2
    lateralArea = baseAreasSum →
    x = 5/2 →
    h = 2 :=
by
  intros h x topBaseSide bottomBaseSide lateralArea baseAreasSum lateralEq baseEq
  sorry

end right_square_pyramid_height_l2187_218735


namespace area_of_triangle_MEF_correct_l2187_218785

noncomputable def area_of_triangle_MEF : ℝ :=
  let r := 10
  let chord_length := 12
  let parallel_segment_length := 15
  let angle_MOA := 30.0
  (1 / 2) * chord_length * (2 * Real.sqrt 21)

theorem area_of_triangle_MEF_correct :
  area_of_triangle_MEF = 12 * Real.sqrt 21 :=
by
  -- proof will go here
  sorry

end area_of_triangle_MEF_correct_l2187_218785


namespace range_of_a_l2187_218755

noncomputable def f (a x : ℝ) : ℝ :=
  Real.exp x + x^2 + (3 * a + 2) * x

theorem range_of_a (a : ℝ) :
  (∃ x ∈ Set.Ioo (-1 : ℝ) 0, ∀ y ∈ Set.Ioo (-1 : ℝ) 0, f a x ≤ f a y) →
  a ∈ Set.Ioo (-1 : ℝ) (-1 / (3 * Real.exp 1)) :=
sorry

end range_of_a_l2187_218755


namespace coloring_triangles_l2187_218750

theorem coloring_triangles (n : ℕ) (k : ℕ) (h_n : n = 18) (h_k : k = 6) :
  (Nat.choose n k) = 18564 :=
by
  rw [h_n, h_k]
  sorry

end coloring_triangles_l2187_218750


namespace negation_of_proposition_l2187_218752

theorem negation_of_proposition (p : Prop) : 
  (∀ x : ℝ, x ≥ 0 → x^2 - x + 1 ≥ 0) ↔ ¬(∃ x : ℝ, x ≥ 0 ∧ x^2 - x + 1 < 0) :=
by sorry

end negation_of_proposition_l2187_218752


namespace martha_initial_crayons_l2187_218747

theorem martha_initial_crayons : ∃ (x : ℕ), (x / 2 + 20 = 29) ∧ x = 18 :=
by
  sorry

end martha_initial_crayons_l2187_218747


namespace sale_in_fifth_month_l2187_218758

def sales_month_1 := 6635
def sales_month_2 := 6927
def sales_month_3 := 6855
def sales_month_4 := 7230
def sales_month_6 := 4791
def target_average := 6500
def number_of_months := 6

def total_sales := sales_month_1 + sales_month_2 + sales_month_3 + sales_month_4 + sales_month_6

theorem sale_in_fifth_month :
  (target_average * number_of_months) - total_sales = 6562 :=
by
  sorry

end sale_in_fifth_month_l2187_218758


namespace number_of_students_l2187_218787

theorem number_of_students (N T : ℕ) (h1 : T = 80 * N)
  (h2 : 95 * (N - 5) = T - 100) : N = 25 :=
by
  sorry

end number_of_students_l2187_218787


namespace product_of_constants_l2187_218759

theorem product_of_constants :
  ∃ M₁ M₂ : ℝ, 
    (∀ x : ℝ, x ≠ 2 ∧ x ≠ 3 → (45 * x - 82) / (x^2 - 5 * x + 6) = M₁ / (x - 2) + M₂ / (x - 3)) ∧ 
    M₁ * M₂ = -424 :=
by
  sorry

end product_of_constants_l2187_218759


namespace inequality_holds_for_all_real_numbers_l2187_218731

theorem inequality_holds_for_all_real_numbers (x : ℝ) : 3 * x - 5 ≤ 12 - 2 * x + x^2 :=
by sorry

end inequality_holds_for_all_real_numbers_l2187_218731
