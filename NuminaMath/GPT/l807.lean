import Mathlib

namespace poles_on_each_side_l807_80733

theorem poles_on_each_side (total_poles : ℕ) (sides_equal : ℕ)
  (h1 : total_poles = 104) (h2 : sides_equal = 4) : 
  (total_poles / sides_equal) = 26 :=
by
  sorry

end poles_on_each_side_l807_80733


namespace k_range_l807_80762

def y_increasing (k : ℝ) : Prop := ∀ x₁ x₂ : ℝ, x₁ < x₂ → k * x₁ + 1 < k * x₂ + 1
def y_max_min (k : ℝ) : Prop := (∃ x : ℝ, 0 ≤ x ∧ x ≤ k ∧ (x^2 - 2 * x + 3 = 2)) ∧ (∃ x : ℝ, 0 ≤ x ∧ x ≤ k ∧ (x^2 - 2 * x + 3 = 3))

theorem k_range (k : ℝ) (hk : (¬ (0 < k ∧ y_max_min k) ∧ (0 < k ∨ y_max_min k))) : 
  (0 < k ∧ k < 1) ∨ (k > 2) :=
sorry

end k_range_l807_80762


namespace probability_correct_l807_80757

namespace ProbabilitySongs

/-- Define the total number of ways to choose 2 out of 4 songs -/ 
def total_ways : ℕ := Nat.choose 4 2

/-- Define the number of ways to choose 2 songs such that neither A nor B is chosen (only C and D can be chosen) -/
def ways_without_AB : ℕ := Nat.choose 2 2

/-- The probability of playing at least one of A and B is calculated via the complementary rule -/
def probability_at_least_one_AB_played : ℚ := 1 - (ways_without_AB / total_ways)

theorem probability_correct : probability_at_least_one_AB_played = 5 / 6 := sorry
end ProbabilitySongs

end probability_correct_l807_80757


namespace fifteenth_odd_multiple_of_5_is_145_l807_80796

def sequence_term (n : ℕ) : ℤ :=
  10 * n - 5

theorem fifteenth_odd_multiple_of_5_is_145 : sequence_term 15 = 145 :=
by
  sorry

end fifteenth_odd_multiple_of_5_is_145_l807_80796


namespace max_value_of_k_proof_l807_80725

noncomputable def maximum_value_of_k (x y k : ℝ) (h1: x > 0) (h2: y > 0) (h3: k > 0) 
(h4: 5 = k^2 * (x^2 / y^2 + y^2 / x^2) + k * (x / y + y / x)) : Prop :=
  k = (-1 + Real.sqrt 17) / 2

-- This is the statement that needs to be proven:
theorem max_value_of_k_proof (x y k : ℝ) (h1: x > 0) (h2: y > 0) (h3: k > 0) 
(h4: 5 = k^2 * (x^2 / y^2 + y^2 / x^2) + k * (x / y + y / x)) : maximum_value_of_k x y k h1 h2 h3 h4 :=
sorry

end max_value_of_k_proof_l807_80725


namespace ruda_received_clock_on_correct_date_l807_80726

/-- Ruda's clock problem -/
def ruda_clock_problem : Prop :=
  ∃ receive_date : ℕ → ℕ × ℕ × ℕ, -- A function mapping the number of presses to a date (Year, Month, Day)
  (∀ days_after_received, 
    receive_date days_after_received = 
    if days_after_received <= 45 then (2022, 10, 27 - (45 - days_after_received)) -- Calculating the receive date.
    else receive_date 45)
  ∧
  receive_date 45 = (2022, 12, 11) -- The day he checked the clock has to be December 11th

-- We want to prove that:
theorem ruda_received_clock_on_correct_date : ruda_clock_problem :=
by
  sorry

end ruda_received_clock_on_correct_date_l807_80726


namespace expand_simplify_correct_l807_80723

noncomputable def expand_and_simplify (x : ℕ) : ℕ :=
  (x + 4) * (x - 9)

theorem expand_simplify_correct (x : ℕ) : 
  (x + 4) * (x - 9) = x^2 - 5*x - 36 := 
by
  sorry

end expand_simplify_correct_l807_80723


namespace number_of_points_l807_80787

theorem number_of_points (x : ℕ) (h : (x * (x - 1)) / 2 = 45) : x = 10 :=
by
  -- Proof to be done here
  sorry

end number_of_points_l807_80787


namespace not_parallel_to_a_l807_80766

noncomputable def is_parallel (u v : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, v = (k * u.1, k * u.2)

theorem not_parallel_to_a : ∀ k : ℝ, ¬ is_parallel (k^2 + 1, k^2 + 1) (1, -2) :=
sorry

end not_parallel_to_a_l807_80766


namespace odd_function_condition_l807_80777

noncomputable def f (A ω φ : ℝ) (x : ℝ) : ℝ :=
  A * Real.sin (ω * x + φ)

theorem odd_function_condition (A ω : ℝ) (hA : 0 < A) (hω : 0 < ω) (φ : ℝ) :
  (f A ω φ 0 = 0) ↔ (f A ω φ) = fun x => -f A ω φ (-x) := 
by
  sorry

end odd_function_condition_l807_80777


namespace solution_unique_for_alpha_neg_one_l807_80728

noncomputable def alpha : ℝ := sorry

axiom alpha_nonzero : alpha ≠ 0

def functional_eqn (f : ℝ → ℝ) (x y : ℝ) : Prop :=
  f (f (x + y)) = f (x + y) + f (x) * f (y) + alpha * x * y

theorem solution_unique_for_alpha_neg_one (f : ℝ → ℝ) :
  (alpha = -1 → (∀ x : ℝ, f x = x)) ∧ (alpha ≠ -1 → ¬ ∃ f : ℝ → ℝ, ∀ x y : ℝ, functional_eqn f x y) :=
sorry

end solution_unique_for_alpha_neg_one_l807_80728


namespace duration_of_investment_l807_80791

-- Define the constants as given in the conditions
def Principal : ℝ := 7200
def Rate : ℝ := 17.5
def SimpleInterest : ℝ := 3150

-- Define the time variable we want to prove
def Time : ℝ := 2.5

-- Prove that the calculated time matches the expected value
theorem duration_of_investment :
  SimpleInterest = (Principal * Rate * Time) / 100 :=
sorry

end duration_of_investment_l807_80791


namespace gcd_1729_867_l807_80724

theorem gcd_1729_867 : Nat.gcd 1729 867 = 1 :=
by
  sorry

end gcd_1729_867_l807_80724


namespace avg_tickets_male_l807_80744

theorem avg_tickets_male (M F : ℕ) (w : ℕ) 
  (h1 : M / F = 1 / 2) 
  (h2 : (M + F) * 66 = M * w + F * 70) 
  : w = 58 := 
sorry

end avg_tickets_male_l807_80744


namespace shape_of_constant_phi_l807_80716

-- Define the spherical coordinates structure
structure SphericalCoordinates where
  ρ : ℝ
  θ : ℝ
  φ : ℝ

-- Define the condition that φ is a constant c
def constant_phi (c : ℝ) (coords : SphericalCoordinates) : Prop :=
  coords.φ = c

-- Define the type for shapes
inductive Shape
  | Line : Shape
  | Circle : Shape
  | Plane : Shape
  | Sphere : Shape
  | Cylinder : Shape
  | Cone : Shape

-- The theorem statement
theorem shape_of_constant_phi (c : ℝ) (coords : SphericalCoordinates) 
  (h : constant_phi c coords) : Shape :=
  Shape.Cone

end shape_of_constant_phi_l807_80716


namespace age_ratio_rahul_deepak_l807_80713

/--
Prove that the ratio between Rahul and Deepak's current ages is 4:3 given the following conditions:
1. After 10 years, Rahul's age will be 26 years.
2. Deepak's current age is 12 years.
-/
theorem age_ratio_rahul_deepak (R D : ℕ) (h1 : R + 10 = 26) (h2 : D = 12) : R / D = 4 / 3 :=
by sorry

end age_ratio_rahul_deepak_l807_80713


namespace program_output_eq_l807_80707

theorem program_output_eq : ∀ (n : ℤ), n^2 + 3 * n - (2 * n^2 - n) = -n^2 + 4 * n := by
  intro n
  sorry

end program_output_eq_l807_80707


namespace horse_revolutions_l807_80794

-- Defining the problem conditions
def radius_outer : ℝ := 30
def radius_inner : ℝ := 10
def revolutions_outer : ℕ := 25

-- The question we need to prove
theorem horse_revolutions :
  (revolutions_outer : ℝ) * (radius_outer / radius_inner) = 75 := 
by
  sorry

end horse_revolutions_l807_80794


namespace closest_point_on_ellipse_l807_80730

theorem closest_point_on_ellipse : 
  ∃ (x y : ℝ), (7 * x^2 + 4 * y^2 = 28 ∧ 3 * x - 2 * y - 16 = 0) ∧ 
  (∀ (x' y' : ℝ), 7 * x'^2 + 4 * y'^2 = 28 → dist (x, y) (0, 0) ≤ dist (x', y') (0, 0)) :=
sorry

end closest_point_on_ellipse_l807_80730


namespace heath_time_spent_l807_80721

variables (rows_per_carrot : ℕ) (plants_per_row : ℕ) (carrots_per_hour : ℕ) (total_hours : ℕ)

def total_carrots (rows_per_carrot plants_per_row : ℕ) : ℕ :=
  rows_per_carrot * plants_per_row

def time_spent (total_carrots carrots_per_hour : ℕ) : ℕ :=
  total_carrots / carrots_per_hour

theorem heath_time_spent
  (h1 : rows_per_carrot = 400)
  (h2 : plants_per_row = 300)
  (h3 : carrots_per_hour = 6000)
  (h4 : total_hours = 20) :
  time_spent (total_carrots rows_per_carrot plants_per_row) carrots_per_hour = total_hours :=
by
  sorry

end heath_time_spent_l807_80721


namespace right_triangle_angle_l807_80740

open Real

theorem right_triangle_angle (a b c : ℝ) (h : a^2 + b^2 = c^2) (h2 : c^2 = 2 * a * b) : 
  ∃ θ : ℝ, θ = 45 ∧ tan θ = a / b := 
by sorry

end right_triangle_angle_l807_80740


namespace find_b_l807_80793

-- Definitions
def quadratic (x b c : ℝ) : ℝ := x^2 + b * x + c

theorem find_b (b c : ℝ) 
  (h_diff : ∀ x : ℝ, 1 ≤ x ∧ x ≤ 7 → (∀ y : ℝ, 1 ≤ y ∧ y ≤ 7 → quadratic x b c - quadratic y b c = 25)) :
  b = -4 ∨ b = -12 :=
by sorry

end find_b_l807_80793


namespace valid_x_y_sum_l807_80734

-- Setup the initial conditions as variables.
variables (x y : ℕ)

-- Declare the conditions as hypotheses.
theorem valid_x_y_sum (h1 : 0 < x) (h2 : x < 25)
  (h3 : 0 < y) (h4 : y < 25) (h5 : x + y + x * y = 119) :
  x + y = 27 ∨ x + y = 24 ∨ x + y = 21 ∨ x + y = 20 :=
sorry

end valid_x_y_sum_l807_80734


namespace max_triangles_convex_polygon_l807_80731

theorem max_triangles_convex_polygon (vertices : ℕ) (interior_points : ℕ) (total_points : ℕ) : 
  vertices = 13 ∧ interior_points = 200 ∧ total_points = 213 ∧ (∀ (x y z : ℕ), (x < total_points ∧ y < total_points ∧ z < total_points) → x ≠ y ∧ y ≠ z ∧ x ≠ z) →
  (∃ triangles : ℕ, triangles = 411) :=
by
  sorry

end max_triangles_convex_polygon_l807_80731


namespace problem_l807_80751

theorem problem (q r : ℕ) (hq : 1259 = 23 * q + r) (hq_pos : 0 < q) (hr_pos : 0 < r) :
  q - r ≤ 37 :=
sorry

end problem_l807_80751


namespace value_of_neg_a_squared_sub_3a_l807_80769

variable (a : ℝ)
variable (h : a^2 + 3 * a - 5 = 0)

theorem value_of_neg_a_squared_sub_3a : -a^2 - 3*a = -5 :=
by
  sorry

end value_of_neg_a_squared_sub_3a_l807_80769


namespace max_abs_value_l807_80705

open Complex Real

theorem max_abs_value (z : ℂ) (h : abs (z - 8) + abs (z + 6 * I) = 10) : abs z ≤ 8 :=
sorry

example : ∃ z : ℂ, abs (z - 8) + abs (z + 6 * I) = 10 ∧ abs z = 8 :=
sorry

end max_abs_value_l807_80705


namespace cylinder_surface_area_l807_80738

theorem cylinder_surface_area (a b : ℝ) (h1 : a = 4 * Real.pi) (h2 : b = 8 * Real.pi) :
  (∃ S, S = 32 * Real.pi^2 + 8 * Real.pi ∨ S = 32 * Real.pi^2 + 32 * Real.pi) :=
by
  sorry

end cylinder_surface_area_l807_80738


namespace flour_already_put_in_l807_80792

theorem flour_already_put_in (total_flour flour_still_needed: ℕ) (h1: total_flour = 9) (h2: flour_still_needed = 6) : total_flour - flour_still_needed = 3 := 
by
  -- Here we will state the proof
  sorry

end flour_already_put_in_l807_80792


namespace cost_of_60_tulips_l807_80701

-- Definition of conditions
def cost_of_bouquet (n : ℕ) : ℝ :=
  if n ≤ 40 then n * 2
  else 40 * 2 + (n - 40) * 3

-- The main statement
theorem cost_of_60_tulips : cost_of_bouquet 60 = 140 := by
  sorry

end cost_of_60_tulips_l807_80701


namespace train_crosses_second_platform_l807_80718

theorem train_crosses_second_platform (
  length_train length_platform1 length_platform2 : ℝ) 
  (time_platform1 : ℝ) 
  (H1 : length_train = 100)
  (H2 : length_platform1 = 200)
  (H3 : length_platform2 = 300)
  (H4 : time_platform1 = 15) :
  ∃ t : ℝ, t = 20 := by
  sorry

end train_crosses_second_platform_l807_80718


namespace find_quadratic_function_l807_80711

-- Define the quadratic function
def quadratic_function (a b c x : ℝ) : ℝ := a * x^2 + b * x + c

-- Define the discriminant condition
def discriminant_zero (a b c : ℝ) : Prop := b^2 - 4 * a * c = 0

-- Given derivative
def given_derivative (x : ℝ) : ℝ := 2 * x + 2

-- Prove that if these conditions hold, then f(x) = x^2 + 2x + 1
theorem find_quadratic_function :
  ∃ (a b c : ℝ), (∀ x, quadratic_function a b c x = 0 → discriminant_zero a b c) ∧
                (∀ x, (2 * a * x + b) = given_derivative x) ∧
                (quadratic_function a b c x = x^2 + 2 * x + 1) := 
by
  sorry

end find_quadratic_function_l807_80711


namespace polynomial_division_l807_80717

open Polynomial

-- Define the theorem statement
theorem polynomial_division (f g : ℤ[X])
  (h : ∀ n : ℤ, f.eval n ∣ g.eval n) :
  ∃ (h : ℤ[X]), g = f * h :=
sorry

end polynomial_division_l807_80717


namespace greatest_value_of_4a_l807_80797

-- Definitions of the given conditions
def hundreds_digit (x : ℕ) : ℕ := x / 100
def tens_digit (x : ℕ) : ℕ := (x / 10) % 10
def units_digit (x : ℕ) : ℕ := x % 10

def satisfies_conditions (a b c x : ℕ) : Prop :=
  hundreds_digit x = a ∧
  tens_digit x = b ∧
  units_digit x = c ∧
  4 * a = 2 * b ∧
  2 * b = c ∧
  a > 0

def difference_of_two_greatest_x : ℕ := 124

theorem greatest_value_of_4a (x1 x2 a1 a2 b1 b2 c1 c2 : ℕ) :
  satisfies_conditions a1 b1 c1 x1 →
  satisfies_conditions a2 b2 c2 x2 →
  x1 - x2 = difference_of_two_greatest_x →
  4 * a1 = 8 :=
by
  sorry

end greatest_value_of_4a_l807_80797


namespace area_of_EPGQ_l807_80742

noncomputable def area_of_region (length_rect width_rect half_length_rect : ℝ) : ℝ :=
  half_length_rect * width_rect

theorem area_of_EPGQ :
  let length_rect := 10.0
  let width_rect := 6.0
  let P_half_length := length_rect / 2
  let Q_half_length := length_rect / 2
  (area_of_region length_rect width_rect P_half_length) = 30.0 :=
by
  sorry

end area_of_EPGQ_l807_80742


namespace triangle_inequality_from_condition_l807_80781

theorem triangle_inequality_from_condition (a b c : ℝ)
  (h : (a^2 + b^2 + c^2)^2 > 2 * (a^4 + b^4 + c^4)) :
  a + b > c ∧ a + c > b ∧ b + c > a :=
by 
  sorry

end triangle_inequality_from_condition_l807_80781


namespace diana_shops_for_newborns_l807_80753

theorem diana_shops_for_newborns (total_children : ℕ) (num_toddlers : ℕ) (teenager_ratio : ℕ) (num_teens : ℕ) (num_newborns : ℕ)
    (h1 : total_children = 40) (h2 : num_toddlers = 6) (h3 : teenager_ratio = 5) (h4 : num_teens = teenager_ratio * num_toddlers) 
    (h5 : num_newborns = total_children - num_teens - num_toddlers) : 
    num_newborns = 4 := sorry

end diana_shops_for_newborns_l807_80753


namespace Steven_has_more_peaches_l807_80714

variable (Steven_peaches : Nat) (Jill_peaches : Nat)
variable (h1 : Steven_peaches = 19) (h2 : Jill_peaches = 6)

theorem Steven_has_more_peaches : Steven_peaches - Jill_peaches = 13 :=
by
  sorry

end Steven_has_more_peaches_l807_80714


namespace other_investment_interest_rate_l807_80784

open Real

-- Definitions of the given conditions
def total_investment : ℝ := 22000
def investment_at_8_percent : ℝ := 17000
def total_interest : ℝ := 1710
def interest_rate_8_percent : ℝ := 0.08

-- Derived definitions from the conditions
def other_investment_amount : ℝ := total_investment - investment_at_8_percent
def interest_from_8_percent : ℝ := investment_at_8_percent * interest_rate_8_percent
def interest_from_other : ℝ := total_interest - interest_from_8_percent

-- Proof problem: Prove that the percentage of the other investment is 0.07 (or 7%).
theorem other_investment_interest_rate :
  interest_from_other / other_investment_amount = 0.07 := by
  sorry

end other_investment_interest_rate_l807_80784


namespace h_inch_approx_l807_80772

noncomputable def h_cm : ℝ := 14.5 - 2 * 1.7
noncomputable def cm_to_inch (cm : ℝ) : ℝ := cm / 2.54
noncomputable def h_inch : ℝ := cm_to_inch h_cm

theorem h_inch_approx : abs (h_inch - 4.37) < 1e-2 :=
by
  -- The proof is omitted
  sorry

end h_inch_approx_l807_80772


namespace conditional_prob_l807_80775

noncomputable def prob_A := 0.7
noncomputable def prob_AB := 0.4

theorem conditional_prob : prob_AB / prob_A = 4 / 7 :=
by
  sorry

end conditional_prob_l807_80775


namespace united_airlines_discount_l807_80795

theorem united_airlines_discount :
  ∀ (delta_price original_price_u discount_delta discount_u saved_amount cheapest_price: ℝ),
    delta_price = 850 →
    original_price_u = 1100 →
    discount_delta = 0.20 →
    saved_amount = 90 →
    cheapest_price = delta_price * (1 - discount_delta) - saved_amount →
    discount_u = (original_price_u - cheapest_price) / original_price_u →
    discount_u = 0.4636363636 :=
by
  intros delta_price original_price_u discount_delta discount_u saved_amount cheapest_price δeq ueq deq saeq cpeq dueq
  -- Placeholder for the actual proof steps
  sorry

end united_airlines_discount_l807_80795


namespace emily_spending_l807_80747

theorem emily_spending (X Y : ℝ) 
  (h1 : (X + 2*X + 3*X + 12*X) = Y) : 
  X = Y / 18 := 
by
  sorry

end emily_spending_l807_80747


namespace record_loss_of_10_l807_80743

-- Definition of profit and loss recording
def record (x : Int) : Int :=
  if x ≥ 0 then x else -x

-- Condition: A profit of $20 should be recorded as +$20
axiom profit_recording : ∀ (p : Int), p ≥ 0 → record p = p

-- Condition: A loss should be recorded as a negative amount
axiom loss_recording : ∀ (l : Int), l < 0 → record l = l

-- Question: How should a loss of $10 be recorded?
-- Prove that if a small store lost $10, it should be recorded as -$10
theorem record_loss_of_10 : record (-10) = -10 :=
by sorry

end record_loss_of_10_l807_80743


namespace area_of_picture_l807_80737

theorem area_of_picture {x y : ℕ} (hx : x > 1) (hy : y > 1) 
  (h : (2 * x + 3) * (y + 2) - x * y = 34) : x * y = 8 := 
by
  sorry

end area_of_picture_l807_80737


namespace renu_work_rate_l807_80709

theorem renu_work_rate (R : ℝ) :
  (∀ (renu_rate suma_rate combined_rate : ℝ),
    renu_rate = 1 / R ∧
    suma_rate = 1 / 6 ∧
    combined_rate = 1 / 3 ∧    
    combined_rate = renu_rate + suma_rate) → 
    R = 6 :=
by
  sorry

end renu_work_rate_l807_80709


namespace sum_of_reciprocals_sum_of_square_reciprocals_sum_of_cubic_reciprocals_l807_80770

variable (p q : ℝ) (x1 x2 : ℝ)

-- Define the condition: Roots of the quadratic equation
def quadratic_equation_condition : Prop :=
  x1^2 + p * x1 + q = 0 ∧ x2^2 + p * x2 + q = 0

-- Define the identities for calculations based on properties of roots
def properties_of_roots : Prop :=
  x1 + x2 = -p ∧ x1 * x2 = q

-- First proof problem
theorem sum_of_reciprocals (h1 : quadratic_equation_condition p q x1 x2) 
                           (h2 : properties_of_roots p q x1 x2) :
  1 / x1 + 1 / x2 = -p / q := 
by sorry

-- Second proof problem
theorem sum_of_square_reciprocals (h1 : quadratic_equation_condition p q x1 x2) 
                                  (h2 : properties_of_roots p q x1 x2) :
  1 / (x1^2) + 1 / (x2^2) = (p^2 - 2*q) / (q^2) := 
by sorry

-- Third proof problem
theorem sum_of_cubic_reciprocals (h1 : quadratic_equation_condition p q x1 x2) 
                                 (h2 : properties_of_roots p q x1 x2) :
  1 / (x1^3) + 1 / (x2^3) = p * (3*q - p^2) / (q^3) := 
by sorry

end sum_of_reciprocals_sum_of_square_reciprocals_sum_of_cubic_reciprocals_l807_80770


namespace remainder_13_pow_2031_mod_100_l807_80763

theorem remainder_13_pow_2031_mod_100 : (13^2031) % 100 = 17 :=
by sorry

end remainder_13_pow_2031_mod_100_l807_80763


namespace line_circle_no_intersection_l807_80720

theorem line_circle_no_intersection :
  ∀ (x y : ℝ), (3 * x + 4 * y = 12) ∧ (x^2 + y^2 = 4) → false :=
by
  -- Sorry to skip the actual proof
  sorry

end line_circle_no_intersection_l807_80720


namespace algebraic_expression_eval_l807_80712

theorem algebraic_expression_eval (a b c : ℝ) (h : a * (-5:ℝ)^4 + b * (-5)^2 + c = 3): 
  a * (5:ℝ)^4 + b * (5)^2 + c = 3 :=
by
  sorry

end algebraic_expression_eval_l807_80712


namespace solve_absolute_value_equation_l807_80745

theorem solve_absolute_value_equation (y : ℝ) : (|y - 4| + 3 * y = 15) ↔ (y = 19 / 4) := by
  sorry

end solve_absolute_value_equation_l807_80745


namespace find_p_4_l807_80776

-- Define the polynomial p(x)
def p (x : ℕ) : ℚ := sorry

-- Given conditions
axiom h1 : p 1 = 1
axiom h2 : p 2 = 1 / 4
axiom h3 : p 3 = 1 / 9
axiom h4 : p 5 = 1 / 25

-- Prove that p(4) = -1/30
theorem find_p_4 : p 4 = -1 / 30 := 
  by sorry

end find_p_4_l807_80776


namespace certain_event_positive_integers_sum_l807_80736

theorem certain_event_positive_integers_sum :
  ∀ (a b : ℕ), a > 0 → b > 0 → a + b > 1 :=
by
  intros a b ha hb
  sorry

end certain_event_positive_integers_sum_l807_80736


namespace rectangle_area_l807_80774

def radius : ℝ := 10
def width : ℝ := 2 * radius
def length : ℝ := 3 * width
def area_of_rectangle : ℝ := length * width

theorem rectangle_area : area_of_rectangle = 1200 :=
  by sorry

end rectangle_area_l807_80774


namespace ball_hits_ground_time_l807_80799

noncomputable def h (t : ℝ) : ℝ := -16 * t^2 - 30 * t + 180

theorem ball_hits_ground_time :
  ∃ t : ℝ, h t = 0 ∧ t = 2.545 :=
by
  sorry

end ball_hits_ground_time_l807_80799


namespace piggy_bank_balance_l807_80748

theorem piggy_bank_balance (original_amount : ℕ) (taken_out : ℕ) : original_amount = 5 ∧ taken_out = 2 → original_amount - taken_out = 3 :=
by sorry

end piggy_bank_balance_l807_80748


namespace greater_number_is_twenty_two_l807_80722

theorem greater_number_is_twenty_two (x y : ℝ) (h1 : x + y = 40) (h2 : 3 * (x - y) = 12) : x = 22 :=
sorry

end greater_number_is_twenty_two_l807_80722


namespace smallest_n_value_l807_80754

theorem smallest_n_value :
  ∃ n, (∀ (sheets : Fin 2000 → Fin 4 → Fin 4),
        (∀ (n : Nat) (h : n ≤ 2000) (a b c d : Fin n) (h' : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d),
          ∃ (i j k : Fin 5), sheets a i = sheets b i ∧ sheets a j = sheets b j ∧ sheets a k = sheets b k → ¬ sheets a i = sheets c i ∧ ¬ sheets b j = sheets c j ∧ ¬ sheets a k = sheets c k)) ↔ n = 25 :=
sorry

end smallest_n_value_l807_80754


namespace correct_factorization_l807_80768

theorem correct_factorization (a b : ℝ) : a^2 - 4 * a * b + 4 * b^2 = (a - 2 * b)^2 :=
by sorry

end correct_factorization_l807_80768


namespace b_plus_d_l807_80767

noncomputable def f (a b c d x : ℝ) : ℝ := a * x^3 + b * x^2 + c * x + d

theorem b_plus_d 
  (a b c d : ℝ) 
  (h1 : f a b c d 1 = 20) 
  (h2 : f a b c d (-1) = 16) 
: b + d = 18 :=
sorry

end b_plus_d_l807_80767


namespace veggies_count_l807_80785

def initial_tomatoes := 500
def picked_tomatoes := 325
def initial_potatoes := 400
def picked_potatoes := 270
def initial_cucumbers := 300
def planted_cucumber_plants := 200
def cucumbers_per_plant := 2
def initial_cabbages := 100
def picked_cabbages := 50
def planted_cabbage_plants := 80
def cabbages_per_cabbage_plant := 3

noncomputable def remaining_tomatoes : Nat :=
  initial_tomatoes - picked_tomatoes

noncomputable def remaining_potatoes : Nat :=
  initial_potatoes - picked_potatoes

noncomputable def remaining_cucumbers : Nat :=
  initial_cucumbers + planted_cucumber_plants * cucumbers_per_plant

noncomputable def remaining_cabbages : Nat :=
  (initial_cabbages - picked_cabbages) + planted_cabbage_plants * cabbages_per_cabbage_plant

theorem veggies_count :
  remaining_tomatoes = 175 ∧
  remaining_potatoes = 130 ∧
  remaining_cucumbers = 700 ∧
  remaining_cabbages = 290 :=
by
  sorry

end veggies_count_l807_80785


namespace geometric_series_sum_squares_l807_80704

theorem geometric_series_sum_squares (a r : ℝ) (hr : -1 < r) (hr2 : r < 1) :
  (∑' n : ℕ, a^2 * r^(3 * n)) = a^2 / (1 - r^3) :=
by
  -- Note: Proof goes here
  sorry

end geometric_series_sum_squares_l807_80704


namespace larger_root_of_degree_11_l807_80708

theorem larger_root_of_degree_11 {x : ℝ} :
  (∃ x₁, x₁ > 0 ∧ (x₁ + x₁^2 + x₁^3 + x₁^4 + x₁^5 + x₁^6 + x₁^7 + x₁^8 = 8 - 10 * x₁^9)) ∧
  (∃ x₂, x₂ > 0 ∧ (x₂ + x₂^2 + x₂^3 + x₂^4 + x₂^5 + x₂^6 + x₂^7 + x₂^8 + x₂^9 + x₂^10 = 8 - 10 * x₂^11)) →
  (∃ x₁ x₂, x₁ > 0 ∧ x₂ > 0 ∧
    (x₁ + x₁^2 + x₁^3 + x₁^4 + x₁^5 + x₁^6 + x₁^7 + x₁^8 = 8 - 10 * x₁^9) ∧
    (x₂ + x₂^2 + x₂^3 + x₂^4 + x₂^5 + x₂^6 + x₂^7 + x₂^8 + x₂^9 + x₂^10 = 8 - 10 * x₂^11) ∧
    x₁ < x₂) :=
by
  sorry

end larger_root_of_degree_11_l807_80708


namespace scheme_A_yield_percentage_l807_80732

-- Define the initial investments and yields
def initial_investment_A : ℝ := 300
def initial_investment_B : ℝ := 200
def yield_B : ℝ := 0.5 -- 50% yield

-- Define the equation given in the problem
def yield_A_equation (P : ℝ) : Prop :=
  initial_investment_A + (initial_investment_A * (P / 100)) = initial_investment_B + (initial_investment_B * yield_B) + 90

-- The proof statement we need to prove
theorem scheme_A_yield_percentage : yield_A_equation 30 :=
by
  sorry -- Proof is omitted

end scheme_A_yield_percentage_l807_80732


namespace library_width_l807_80789

theorem library_width 
  (num_libraries : ℕ) 
  (length_per_library : ℕ) 
  (total_area_km2 : ℝ) 
  (conversion_factor : ℝ) 
  (total_area : ℝ) 
  (area_of_one_library : ℝ) 
  (width_of_library : ℝ) :

  num_libraries = 8 →
  length_per_library = 300 →
  total_area_km2 = 0.6 →
  conversion_factor = 1000000 →
  total_area = total_area_km2 * conversion_factor →
  area_of_one_library = total_area / num_libraries →
  width_of_library = area_of_one_library / length_per_library →
  width_of_library = 250 :=
by
  intros;
  sorry

end library_width_l807_80789


namespace find_f_values_l807_80798

noncomputable def f : ℕ → ℕ := sorry

axiom condition1 : ∀ (a b : ℕ), a ≠ b → (a * f a + b * f b > a * f b + b * f a)
axiom condition2 : ∀ (n : ℕ), f (f n) = 3 * n

theorem find_f_values : f 1 + f 6 + f 28 = 66 := 
by
  sorry

end find_f_values_l807_80798


namespace f_prime_neg_one_l807_80760

-- Given conditions and definitions
def f (x : ℝ) (a b c : ℝ) := a * x^4 + b * x^2 + c

def f_prime (x : ℝ) (a b : ℝ) := 4 * a * x^3 + 2 * b * x

-- The theorem we need to prove
theorem f_prime_neg_one (a b c : ℝ) (h : f_prime 1 a b = 2) : f_prime (-1) a b = -2 := by
  sorry

end f_prime_neg_one_l807_80760


namespace radian_measure_of_minute_hand_rotation_l807_80729

theorem radian_measure_of_minute_hand_rotation :
  ∀ (t : ℝ), (t = 10) → (2 * π / 60 * t = -π/3) := by
  sorry

end radian_measure_of_minute_hand_rotation_l807_80729


namespace vladimir_can_invest_more_profitably_l807_80702

-- Conditions and parameters
def p_buckwheat_initial : ℝ := 70 -- initial price of buckwheat in RUB/kg
def p_buckwheat_2017 : ℝ := 85 -- price of buckwheat in early 2017 in RUB/kg
def rate_2015 : ℝ := 0.16 -- interest rate for annual deposit in 2015
def rate_2016 : ℝ := 0.10 -- interest rate for annual deposit in 2016
def rate_2yr : ℝ := 0.15 -- interest rate for two-year deposit per year

-- Amounts after investments
def amount_annual : ℝ := p_buckwheat_initial * (1 + rate_2015) * (1 + rate_2016)
def amount_2yr : ℝ := p_buckwheat_initial * (1 + rate_2yr)^2

-- Prove that the best investment amount is greater than the 2017 buckwheat price
theorem vladimir_can_invest_more_profitably : max amount_annual amount_2yr > p_buckwheat_2017 := by
  sorry

end vladimir_can_invest_more_profitably_l807_80702


namespace find_lengths_of_segments_l807_80783

variable (b c : ℝ)

theorem find_lengths_of_segments (CK AK AB CT AC AT : ℝ)
  (h1 : CK = AK + AB)
  (h2 : CK = (b + c) / 2)
  (h3 : CT = AC - AT)
  (h4 : AC = b) :
  AT = (b + c) / 2 ∧ CT = (b - c) / 2 := 
sorry

end find_lengths_of_segments_l807_80783


namespace parallel_lines_slope_l807_80746

theorem parallel_lines_slope (m : ℝ) :
  (∀ x y : ℝ, (m + 3) * x + 4 * y + 3 * m - 5 = 0) ∧ (∀ x y : ℝ, 2 * x + (m + 5) * y - 8 = 0) →
  m = -7 :=
by
  intro H
  sorry

end parallel_lines_slope_l807_80746


namespace length_of_bridge_l807_80715

-- Definitions based on the conditions
def walking_speed_kmph : ℝ := 10 -- speed in km/hr
def time_minutes : ℝ := 24 -- crossing time in minutes
def conversion_factor_km_to_m : ℝ := 1000
def conversion_factor_hr_to_min : ℝ := 60

-- The main statement to prove
theorem length_of_bridge :
  let walking_speed_m_per_min := walking_speed_kmph * conversion_factor_km_to_m / conversion_factor_hr_to_min;
  walking_speed_m_per_min * time_minutes = 4000 := 
by
  let walking_speed_m_per_min := walking_speed_kmph * conversion_factor_km_to_m / conversion_factor_hr_to_min;
  sorry

end length_of_bridge_l807_80715


namespace no_quadruples_solution_l807_80706

theorem no_quadruples_solution (a b c d : ℝ) :
    a^3 + c^3 = 2 ∧
    a^2 * b + c^2 * d = 0 ∧
    b^3 + d^3 = 1 ∧
    a * b^2 + c * d^2 = -6 →
    false :=
by 
  intros h
  sorry

end no_quadruples_solution_l807_80706


namespace stream_speed_fraction_l807_80759

theorem stream_speed_fraction (B S : ℝ) (h1 : B = 3 * S) 
  (h2 : (1 / (B - S)) = 2 * (1 / (B + S))) : (S / B) = 1 / 3 :=
sorry

end stream_speed_fraction_l807_80759


namespace angle_sum_and_relation_l807_80786

variable {A B : ℝ}

theorem angle_sum_and_relation (h1 : A + B = 180) (h2 : A = 5 * B) : A = 150 := by
  sorry

end angle_sum_and_relation_l807_80786


namespace simplify_expression_l807_80788

theorem simplify_expression (a b c : ℝ) (h1 : a + b + c = 0) (h2 : a ≠ 0) (h3 : b ≠ 0) (h4 : c ≠ 0) :
    a * (1 / b + 1 / c) + b * (1 / a + 1 / c) + c * (1 / a + 1 / b) = -3 :=
by
  sorry

end simplify_expression_l807_80788


namespace combine_monomials_x_plus_y_l807_80779

theorem combine_monomials_x_plus_y : ∀ (x y : ℤ),
  7 * x = 2 - 4 * y →
  y + 7 = 2 * x →
  x + y = -1 :=
by
  intros x y h1 h2
  sorry

end combine_monomials_x_plus_y_l807_80779


namespace traveler_arrangements_l807_80780

theorem traveler_arrangements :
  let travelers := 6
  let rooms := 3
  ∃ (arrangements : Nat), arrangements = 240 := by
  sorry

end traveler_arrangements_l807_80780


namespace relationship_x_y_l807_80755

variable (a b x y : ℝ)

theorem relationship_x_y (h1: 0 < a) (h2: a < b)
  (hx : x = (Real.sqrt (a + b) - Real.sqrt b))
  (hy : y = (Real.sqrt b - Real.sqrt (b - a))) :
  x < y :=
  sorry

end relationship_x_y_l807_80755


namespace square_of_105_l807_80700

/-- Prove that 105^2 = 11025. -/
theorem square_of_105 : 105^2 = 11025 :=
by
  sorry

end square_of_105_l807_80700


namespace multiply_negatives_l807_80764

theorem multiply_negatives : (-3) * (-4) * (-1) = -12 := 
by sorry

end multiply_negatives_l807_80764


namespace polynomial_roots_l807_80739

theorem polynomial_roots (k r : ℝ) (hk_pos : k > 0) 
(h_sum : r + 1 = 2 * k) (h_prod : r * 1 = k) : 
  r = 1 ∧ (∀ x, (x - 1) * (x - 1) = x^2 - 2 * x + 1) := 
by 
  sorry

end polynomial_roots_l807_80739


namespace money_lent_to_C_is_3000_l807_80703

def principal_B : ℕ := 5000
def time_B : ℕ := 2
def time_C : ℕ := 4
def rate_of_interest : ℕ := 12
def total_interest : ℕ := 2640
def interest_rate : ℚ := (rate_of_interest : ℚ) / 100
def interest_B : ℚ := principal_B * interest_rate * time_B
def interest_C (P_C : ℚ) : ℚ := P_C * interest_rate * time_C

theorem money_lent_to_C_is_3000 :
  ∃ P_C : ℚ, interest_B + interest_C P_C = total_interest ∧ P_C = 3000 :=
by
  use 3000
  unfold interest_B interest_C interest_rate principal_B time_B time_C rate_of_interest total_interest
  sorry

end money_lent_to_C_is_3000_l807_80703


namespace coffee_serving_time_between_1_and_2_is_correct_l807_80756

theorem coffee_serving_time_between_1_and_2_is_correct
    (x : ℝ)
    (h_pos: 0 < x)
    (h_lt: x < 60) :
    30 + (x / 2) = 360 - (6 * x) → x = 660 / 13 :=
by
  sorry

end coffee_serving_time_between_1_and_2_is_correct_l807_80756


namespace exponential_sequence_term_eq_l807_80771

-- Definitions for the conditions
variable {α : Type} [CommRing α] (q : α)
def a (n : ℕ) : α := q * (q ^ (n - 1))

-- Statement of the problem
theorem exponential_sequence_term_eq : a q 9 = a q 3 * a q 7 := by
  sorry

end exponential_sequence_term_eq_l807_80771


namespace solve_for_s_l807_80735

theorem solve_for_s : ∃ s, (∃ x, 4 * x^2 - 8 * x - 320 = 0) ∧ s = 81 :=
by {
  -- Sorry is used to skip the actual proof.
  sorry
}

end solve_for_s_l807_80735


namespace exit_condition_l807_80719

-- Define the loop structure in a way that is consistent with how the problem is described
noncomputable def program_loop (k : ℕ) : ℕ :=
  if k < 7 then 35 else sorry -- simulate the steps of the program

-- The proof goal is to show that the condition which stops the loop when s = 35 is k ≥ 7
theorem exit_condition (k : ℕ) (s : ℕ) : 
  (program_loop k = 35) → (k ≥ 7) :=
by {
  sorry
}

end exit_condition_l807_80719


namespace greenwood_school_l807_80752

theorem greenwood_school (f s : ℕ) (h : (3 / 4) * f = (1 / 3) * s) : s = 3 * f :=
by
  sorry

end greenwood_school_l807_80752


namespace school_bought_50_cartons_of_markers_l807_80761

theorem school_bought_50_cartons_of_markers
  (n_puzzles : ℕ := 200)  -- the remaining amount after buying pencils
  (cost_per_carton_marker : ℕ := 4)  -- the cost per carton of markers
  :
  (n_puzzles / cost_per_carton_marker = 50) := -- the theorem to prove
by
  -- Provide skeleton proof strategy here
  sorry  -- details of the proof

end school_bought_50_cartons_of_markers_l807_80761


namespace discount_problem_l807_80758

theorem discount_problem (n : ℕ) : 
  (∀ x : ℝ, 0 < x → (1 - n / 100 : ℝ) * x < min (0.72 * x) (min (0.6724 * x) (0.681472 * x))) ↔ n ≥ 33 :=
by
  sorry

end discount_problem_l807_80758


namespace paul_reading_novel_l807_80778

theorem paul_reading_novel (x : ℕ) 
  (h1 : x - ((1 / 6) * x + 10) - ((1 / 5) * (x - ((1 / 6) * x + 10)) + 14) - ((1 / 4) * ((x - ((1 / 6) * x + 10) - ((1 / 5) * (x - ((1 / 6) * x + 10)) + 14)) + 16)) = 48) : 
  x = 161 :=
by sorry

end paul_reading_novel_l807_80778


namespace probability_light_change_l807_80765

noncomputable def total_cycle_duration : ℕ := 45 + 5 + 50
def change_intervals : ℕ := 15

theorem probability_light_change :
  (15 : ℚ) / total_cycle_duration = 3 / 20 :=
by
  sorry

end probability_light_change_l807_80765


namespace distance_Tim_covers_l807_80750

theorem distance_Tim_covers (initial_distance : ℕ) (tim_speed elan_speed : ℕ) (double_speed_time : ℕ)
  (h_initial_distance : initial_distance = 30)
  (h_tim_speed : tim_speed = 10)
  (h_elan_speed : elan_speed = 5)
  (h_double_speed_time : double_speed_time = 1) :
  ∃ t d : ℕ, d = 20 ∧ t ∈ {t | t = d / tim_speed + (initial_distance - d) / (tim_speed * 2)} :=
sorry

end distance_Tim_covers_l807_80750


namespace maximum_value_product_cube_expression_l807_80727

theorem maximum_value_product_cube_expression (x y z : ℝ) (h1 : 0 ≤ x) (h2 : 0 ≤ y) (h3 : 0 ≤ z) (h4 : x + y + z = 3) :
  (x^3 - x * y^2 + y^3) * (x^3 - x^2 * z + z^3) * (y^3 - y^2 * z + z^3) ≤ 1 :=
sorry

end maximum_value_product_cube_expression_l807_80727


namespace equilateral_triangle_l807_80782

theorem equilateral_triangle (a b c : ℝ) (h1 : a^4 = b^4 + c^4 - b^2 * c^2) (h2 : b^4 = a^4 + c^4 - a^2 * c^2) : 
  a = b ∧ b = c ∧ c = a :=
by sorry

end equilateral_triangle_l807_80782


namespace negation_is_false_l807_80741

-- Definitions corresponding to the conditions
def prop (x : ℝ) := x > 0 → x^2 > 0

-- Statement of the proof problem in Lean 4
theorem negation_is_false : ¬(∀ x : ℝ, ¬(x > 0 → x^2 > 0)) = false :=
by {
  sorry
}

end negation_is_false_l807_80741


namespace find_pairs_l807_80749

noncomputable def f (k : ℤ) (x y : ℝ) : ℝ :=
  if k = 0 then 0 else (x^k + y^k + (-1)^k * (x + y)^k) / k

theorem find_pairs (x y : ℝ) (hxy : x ≠ 0 ∧ y ≠ 0 ∧ x + y ≠ 0) :
  ∃ (m n : ℤ), m ≠ 0 ∧ n ≠ 0 ∧ m ≤ n ∧ m + n ≠ 0 ∧ 
    (∀ (x y : ℝ), x ≠ 0 ∧ y ≠ 0 ∧ x + y ≠ 0 → f m x y * f n x y = f (m + n) x y) :=
  sorry

end find_pairs_l807_80749


namespace carpet_area_l807_80710

/-- A rectangular floor with a length of 15 feet and a width of 12 feet needs 20 square yards of carpet to cover it. -/
theorem carpet_area (length_feet : ℕ) (width_feet : ℕ) (feet_per_yard : ℕ) (length_yards : ℕ) (width_yards : ℕ) (area_sq_yards : ℕ) :
  length_feet = 15 ∧
  width_feet = 12 ∧
  feet_per_yard = 3 ∧
  length_yards = length_feet / feet_per_yard ∧
  width_yards = width_feet / feet_per_yard ∧
  area_sq_yards = length_yards * width_yards → 
  area_sq_yards = 20 :=
by
  sorry

end carpet_area_l807_80710


namespace tan_half_angle_product_l807_80790

theorem tan_half_angle_product (a b : ℝ) 
  (h : 7 * (Real.cos a + Real.sin b) + 6 * (Real.cos a * Real.cos b - 1) = 0) :
  (Real.tan (a / 2)) * (Real.tan (b / 2)) = 5 ∨ (Real.tan (a / 2)) * (Real.tan (b / 2)) = -5 :=
by 
  sorry

end tan_half_angle_product_l807_80790


namespace sequence_general_term_l807_80773

theorem sequence_general_term 
  (a : ℕ → ℝ)
  (h₀ : a 1 = 1)
  (h₁ : a 2 = 1 / 3)
  (h₂ : ∀ n : ℕ, 2 ≤ n → a n * a (n - 1) + a n * a (n + 1) = 2 * a (n - 1) * a (n + 1)) :
  ∀ n : ℕ, 1 ≤ n → a n = 1 / (2 * n - 1) := 
by
  sorry

end sequence_general_term_l807_80773
