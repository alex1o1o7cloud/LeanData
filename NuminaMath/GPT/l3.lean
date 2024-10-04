import Mathlib

namespace digits_to_replace_l3_3647

theorem digits_to_replace (a b c d e f : ℕ) :
  (a = 1) →
  (b < 5) →
  (c = 8) →
  (d = 1) →
  (e = 0) →
  (f = 4) →
  (100 * a + 10 * b + c)^2 = 10000 * d + 1000 * e + 100 * f + 10 * f + f :=
  by
    intros ha hb hc hd he hf 
    sorry

end digits_to_replace_l3_3647


namespace scientific_notation_l3_3618

theorem scientific_notation (a n : ℝ) (h1 : 100000000 = a * 10^n) (h2 : 1 ≤ a) (h3 : a < 10) : 
  a = 1 ∧ n = 8 :=
by
  sorry

end scientific_notation_l3_3618


namespace highest_x_value_satisfies_equation_l3_3234

theorem highest_x_value_satisfies_equation:
  ∃ x, x ≤ 4 ∧ (∀ x1, x1 ≤ 4 → x1 = 4 ↔ (15 * x1^2 - 40 * x1 + 18) / (4 * x1 - 3) + 7 * x1 = 9 * x1 - 2) :=
by
  sorry

end highest_x_value_satisfies_equation_l3_3234


namespace sin_330_deg_l3_3866

noncomputable theory

open Real

theorem sin_330_deg :
  sin (330 * (π / 180)) = -1 / 2 :=
by sorry

end sin_330_deg_l3_3866


namespace total_plums_l3_3670

def alyssa_plums : Nat := 17
def jason_plums : Nat := 10

theorem total_plums : alyssa_plums + jason_plums = 27 := 
by
  -- proof goes here
  sorry

end total_plums_l3_3670


namespace jordan_total_points_l3_3139

-- Definitions based on conditions in the problem
def jordan_attempts (x y : ℕ) : Prop :=
  x + y = 40

def points_from_three_point_shots (x : ℕ) : ℝ :=
  0.75 * x

def points_from_two_point_shots (y : ℕ) : ℝ :=
  0.8 * y

-- Main theorem to prove the total points scored by Jordan
theorem jordan_total_points (x y : ℕ) 
  (h_attempts : jordan_attempts x y) : 
  points_from_three_point_shots x + points_from_two_point_shots y = 30 := 
by
  sorry

end jordan_total_points_l3_3139


namespace quadratic_equation_unique_solution_l3_3183

theorem quadratic_equation_unique_solution (a c : ℝ) : 
  16 - 4 * a * c = 0 ∧ a + c = 5 ∧ a < c → (a, c) = (1, 4) :=
by
  sorry

end quadratic_equation_unique_solution_l3_3183


namespace fractions_product_simplified_l3_3681

theorem fractions_product_simplified : (2/3 : ℚ) * (4/7) * (9/11) = 24/77 := by
  sorry

end fractions_product_simplified_l3_3681


namespace sin_330_eq_neg_half_l3_3844

theorem sin_330_eq_neg_half 
  (h1: (330 : ℝ) = 330)
  (h2: (330 : ℝ = 360 - 30))
  (h3: 30 ∈ reference_angle_set ∧ sin 30 = 1/2)
  (h4: 330 ∈ fourth_quadrant)
  (h5: ∀ θ ∈ fourth_quadrant, sin θ < 0) :
  sin 330 = -1/2 := 
by
  sorry

end sin_330_eq_neg_half_l3_3844


namespace new_area_is_726_l3_3512

variable (l w : ℝ)
variable (h_area : l * w = 576)
variable (l' : ℝ := 1.20 * l)
variable (w' : ℝ := 1.05 * w)

theorem new_area_is_726 : l' * w' = 726 := by
  sorry

end new_area_is_726_l3_3512


namespace modular_arithmetic_proof_l3_3285

open Nat

theorem modular_arithmetic_proof (m : ℕ) (h0 : 0 ≤ m ∧ m < 37) (h1 : 4 * m ≡ 1 [MOD 37]) :
  (3^m)^4 ≡ 27 + 3 [MOD 37] :=
by
  -- Although some parts like modular inverse calculation or finding specific m are skipped,
  -- the conclusion directly should reflect (3^m)^4 ≡ 27 + 3 [MOD 37]
  -- Considering (3^m)^4 - 3 ≡ 24 [MOD 37] translates to the above statement
  sorry

end modular_arithmetic_proof_l3_3285


namespace four_digit_numbers_sum_even_l3_3424

theorem four_digit_numbers_sum_even : 
  ∃ N : ℕ, 
    (∀ (digits : Finset ℕ) (thousands hundreds tens units : ℕ), 
      digits = {1, 2, 3, 4, 5, 6} ∧ 
      ∀ n ∈ digits, (0 < n ∧ n < 10) ∧ 
      (thousands ∈ digits ∧ hundreds ∈ digits ∧ tens ∈ digits ∧ units ∈ digits) ∧ 
      (thousands ≠ hundreds ∧ thousands ≠ tens ∧ thousands ≠ units ∧ 
       hundreds ≠ tens ∧ hundreds ≠ units ∧ tens ≠ units) ∧ 
      (tens + units) % 2 = 0 → N = 324) :=
sorry

end four_digit_numbers_sum_even_l3_3424


namespace eval_exp_l3_3094

theorem eval_exp {a b : ℝ} (h : a = 3^4) : a^(5/4) = 243 :=
by
  sorry

end eval_exp_l3_3094


namespace sin_330_deg_l3_3850

theorem sin_330_deg : Real.sin (330 * Real.pi / 180) = -1 / 2 := by
  sorry

end sin_330_deg_l3_3850


namespace percent_area_covered_by_hexagons_l3_3317

theorem percent_area_covered_by_hexagons (a : ℝ) (h1 : 0 < a) :
  let large_square_area := 4 * a^2
  let hexagon_contribution := a^2 / 4
  (hexagon_contribution / large_square_area) * 100 = 25 := 
by
  sorry

end percent_area_covered_by_hexagons_l3_3317


namespace number_of_routes_l3_3366

structure RailwayStation :=
  (A B C D E F G H I J K L M : ℕ)

def initialize_station : RailwayStation :=
  ⟨1, 1, 1, 1, 2, 2, 3, 3, 3, 6, 9, 9, 18⟩

theorem number_of_routes (station : RailwayStation) : station.M = 18 :=
  by sorry

end number_of_routes_l3_3366


namespace solve_system_of_equations_l3_3468

theorem solve_system_of_equations : ∃ (x y : ℝ), (2 * x - y = 3) ∧ (3 * x + 2 * y = 8) ∧ (x = 2) ∧ (y = 1) := by
  sorry

end solve_system_of_equations_l3_3468


namespace sequence_a_5_l3_3107

theorem sequence_a_5 (S : ℕ → ℝ) (a : ℕ → ℝ) (h1 : ∀ n : ℕ, n > 0 → S n = 2 * a n - 3) (h2 : ∀ n : ℕ, n > 0 → a n = S n - S (n - 1)) :
  a 5 = 48 := by
  -- The proof and implementations are omitted
  sorry

end sequence_a_5_l3_3107


namespace sin_330_eq_neg_one_half_l3_3884

theorem sin_330_eq_neg_one_half : sin (330 * π / 180) = -1 / 2 := by
  sorry

end sin_330_eq_neg_one_half_l3_3884


namespace sum_of_exponents_l3_3435

theorem sum_of_exponents (n : ℕ) (h : n = 2^11 + 2^10 + 2^5 + 2^4 + 2^2) : 11 + 10 + 5 + 4 + 2 = 32 :=
by {
  -- The proof could be written here
  sorry
}

end sum_of_exponents_l3_3435


namespace conditional_probability_l3_3191

variable (Ω : Type) [Fintype Ω] [DecidableEq Ω]

namespace conditional_probability_problem

-- Define the sample space for the experiment
def sample_space : Finset (Finset Ω) := 
  {s ∈ (univ : Finset (Finset Ω)) | s.card = 3}

-- Define events A and B
def A (s : Finset Ω) : Prop := s.card = 3
def B (s : Finset Ω) : Prop := ∃ a ∈ s, ∀ b ∈ s, b ≠ a

-- Given the sample space Ω, define the conditional probability
def P (A B : Finset Ω → Prop) : ℝ := 
  (Finset.card (sample_space.filter (λ s, A s ∧ B s))).to_real / 
  (Finset.card (sample_space.filter B)).to_real

theorem conditional_probability :
  P A B = 1 / 2 := by
  sorry

end conditional_probability_problem

end conditional_probability_l3_3191


namespace mixed_groups_count_l3_3026

/-- Define the initial conditions --/
def number_of_children : ℕ := 300
def number_of_groups : ℕ := 100
def group_size : ℕ := 3
def boy_boy_photos : ℕ := 100
def girl_girl_photos : ℕ := 56

/-- Define the proof problem -/
theorem mixed_groups_count : 
    (number_of_children = 300) →
    (number_of_groups = 100) →
    (group_size = 3) →
    (boy_boy_photos = 100) →
    (girl_girl_photos = 56) →
    (∀ total_photos, total_photos = number_of_groups * group_size) →
    (∃ mixed_groups, mixed_groups = (total_photos - boy_boy_photos - girl_girl_photos) / 2) →
    mixed_groups = 72 :=
by
    intros h1 h2 h3 h4 h5 h6 h7
    sorry

end mixed_groups_count_l3_3026


namespace greatest_x_l3_3341

theorem greatest_x (x : ℕ) (h : x > 0 ∧ (x^4 / x^2 : ℚ) < 18) : x ≤ 4 :=
by
  sorry

end greatest_x_l3_3341


namespace nina_money_l3_3463

variable (C : ℝ)

def original_widget_count : ℕ := 6
def new_widget_count : ℕ := 8
def price_reduction : ℝ := 1.5

theorem nina_money (h : original_widget_count * C = new_widget_count * (C - price_reduction)) :
  original_widget_count * C = 36 := by
  sorry

end nina_money_l3_3463


namespace dinner_cakes_today_6_l3_3222

-- Definitions based on conditions
def lunch_cakes_today : ℕ := 5
def dinner_cakes_today (x : ℕ) : ℕ := x
def yesterday_cakes : ℕ := 3
def total_cakes_served : ℕ := 14

-- Lean statement to prove the mathematical equivalence
theorem dinner_cakes_today_6 (x : ℕ) (h : lunch_cakes_today + dinner_cakes_today x + yesterday_cakes = total_cakes_served) : x = 6 :=
by {
  sorry -- Proof to be completed.
}

end dinner_cakes_today_6_l3_3222


namespace find_value_x_y_cube_l3_3539

variables (x y k c m : ℝ)

theorem find_value_x_y_cube
  (h1 : x^3 * y^3 = k)
  (h2 : 1 / x^3 + 1 / y^3 = c)
  (h3 : x + y = m) :
  (x + y)^3 = c * k + 3 * k^(1/3) * m :=
by
  sorry

end find_value_x_y_cube_l3_3539


namespace skittles_total_correct_l3_3188

def number_of_students : ℕ := 9
def skittles_per_student : ℕ := 3
def total_skittles : ℕ := 27

theorem skittles_total_correct : number_of_students * skittles_per_student = total_skittles := by
  sorry

end skittles_total_correct_l3_3188


namespace value_of_expression_l3_3043

theorem value_of_expression : 4 * (8 - 6) - 7 = 1 := by
  -- Calculation steps would go here
  sorry

end value_of_expression_l3_3043


namespace probability_Q_within_three_units_of_origin_l3_3364

noncomputable def probability_within_three_units_of_origin :=
  let radius := 3
  let square_side := 10
  let circle_area := Real.pi * radius^2
  let square_area := square_side^2
  circle_area / square_area

theorem probability_Q_within_three_units_of_origin :
  probability_within_three_units_of_origin = 9 * Real.pi / 100 :=
by
  -- Since this proof is not required, we skip it with sorry.
  sorry

end probability_Q_within_three_units_of_origin_l3_3364


namespace speed_of_water_current_l3_3517

theorem speed_of_water_current (v : ℝ) 
  (swimmer_speed_still_water : ℝ := 4) 
  (distance : ℝ := 3) 
  (time : ℝ := 1.5)
  (effective_speed_against_current : ℝ := swimmer_speed_still_water - v) :
  effective_speed_against_current = distance / time → v = 2 := 
by
  -- Proof
  sorry

end speed_of_water_current_l3_3517


namespace solve_equation1_solve_equation2_l3_3764

-- Define the first equation and state the theorem that proves its roots
def equation1 (x : ℝ) : Prop := 2 * x^2 + 1 = 3 * x

theorem solve_equation1 (x : ℝ) : equation1 x ↔ (x = 1 ∨ x = 1/2) :=
by sorry

-- Define the second equation and state the theorem that proves its roots
def equation2 (x : ℝ) : Prop := (2 * x - 1)^2 = (3 - x)^2

theorem solve_equation2 (x : ℝ) : equation2 x ↔ (x = -2 ∨ x = 4 / 3) :=
by sorry

end solve_equation1_solve_equation2_l3_3764


namespace sin_330_correct_l3_3974

noncomputable def sin_330 : ℝ := sorry

theorem sin_330_correct : sin_330 = -1 / 2 :=
  sorry

end sin_330_correct_l3_3974


namespace sin_330_l3_3900

theorem sin_330 : Real.sin (330 * Real.pi / 180) = -1 / 2 := 
by
  -- Outline the proof here without providing it
  -- sorry to delay the proof
  sorry

end sin_330_l3_3900


namespace sin_330_correct_l3_3973

noncomputable def sin_330 : ℝ := sorry

theorem sin_330_correct : sin_330 = -1 / 2 :=
  sorry

end sin_330_correct_l3_3973


namespace original_deck_total_l3_3066

theorem original_deck_total (b y : ℕ) 
    (h1 : (b : ℚ) / (b + y) = 2 / 5)
    (h2 : (b : ℚ) / (b + y + 6) = 5 / 14) :
    b + y = 50 := by
  sorry

end original_deck_total_l3_3066


namespace function_equiv_proof_l3_3102

noncomputable def function_solution (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (f x * f y) = f x * y

theorem function_equiv_proof : ∀ f : ℝ → ℝ,
  function_solution f ↔ (∀ x : ℝ, f x = 0 ∨ f x = x ∨ f x = -x) := 
sorry

end function_equiv_proof_l3_3102


namespace average_after_11th_inning_is_30_l3_3049

-- Define the conditions as Lean 4 definitions
def score_in_11th_inning : ℕ := 80
def increase_in_avg : ℕ := 5
def innings_before_11th : ℕ := 10

-- Define the average before 11th inning
def average_before (x : ℕ) : ℕ := x

-- Define the total runs before 11th inning
def total_runs_before (x : ℕ) : ℕ := innings_before_11th * (average_before x)

-- Define the total runs after 11th inning
def total_runs_after (x : ℕ) : ℕ := total_runs_before x + score_in_11th_inning

-- Define the new average after 11th inning
def new_average_after (x : ℕ) : ℕ := total_runs_after x / (innings_before_11th + 1)

-- Theorem statement
theorem average_after_11th_inning_is_30 : 
  ∃ (x : ℕ), new_average_after x = average_before x + increase_in_avg → new_average_after 25 = 30 :=
by
  sorry

end average_after_11th_inning_is_30_l3_3049


namespace sin_330_eq_neg_half_l3_3834

theorem sin_330_eq_neg_half : sin (330 * (π / 180)) = -1 / 2 := 
by
  -- condition: \(330^\circ = 360^\circ - 30^\circ\)
  -- condition: \(\sin 30^\circ = \frac{1}{2}\)
  -- condition: sine is negative in the fourth quadrant
  sorry

end sin_330_eq_neg_half_l3_3834


namespace tickets_spent_on_beanie_l3_3233

variable (initial_tickets won_tickets tickets_left tickets_spent: ℕ)

theorem tickets_spent_on_beanie
  (h1 : initial_tickets = 49)
  (h2 : won_tickets = 6)
  (h3 : tickets_left = 30)
  (h4 : tickets_spent = initial_tickets + won_tickets - tickets_left) :
  tickets_spent = 25 :=
by
  sorry

end tickets_spent_on_beanie_l3_3233


namespace daves_earnings_l3_3692

theorem daves_earnings
  (hourly_wage : ℕ)
  (monday_hours : ℕ)
  (tuesday_hours : ℕ)
  (monday_earning : monday_hours * hourly_wage = 36)
  (tuesday_earning : tuesday_hours * hourly_wage = 12) :
  monday_hours * hourly_wage + tuesday_hours * hourly_wage = 48 :=
by
  sorry

end daves_earnings_l3_3692


namespace expand_expression_l3_3238

theorem expand_expression :
  (6 * (Polynomial.C (Complex.ofReal 1) * (Polynomial.X - 3)) * (Polynomial.X^2 + 4 * Polynomial.X + 16)).coeffs
   = (6 * Polynomial.X^3 + 6 * Polynomial.X^2 + 24 * Polynomial.X - 288).coeffs := by
  sorry

end expand_expression_l3_3238


namespace triangle_area_l3_3456

def vec2 := ℝ × ℝ

def area_of_triangle (a b : vec2) : ℝ :=
  0.5 * |a.1 * b.2 - a.2 * b.1|

def a : vec2 := (2, -3)
def b : vec2 := (4, -1)

theorem triangle_area : area_of_triangle a b = 5 := by
  sorry

end triangle_area_l3_3456


namespace sin_330_deg_l3_3868

noncomputable theory

open Real

theorem sin_330_deg :
  sin (330 * (π / 180)) = -1 / 2 :=
by sorry

end sin_330_deg_l3_3868


namespace number_of_rabbits_l3_3067

theorem number_of_rabbits (C D : ℕ) (hC : C = 49) (hD : D = 37) (h : D + R = C + 9) :
  R = 21 :=
by
    sorry

end number_of_rabbits_l3_3067


namespace son_completion_time_l3_3509

theorem son_completion_time (M S F : ℝ) 
  (h1 : M = 1 / 10) 
  (h2 : M + S = 1 / 5) 
  (h3 : S + F = 1 / 4) : 
  1 / S = 10 := 
  sorry

end son_completion_time_l3_3509


namespace sin_330_deg_l3_3854

theorem sin_330_deg : Real.sin (330 * Real.pi / 180) = -1 / 2 := by
  sorry

end sin_330_deg_l3_3854


namespace factorize_expr_l3_3239

theorem factorize_expr (x y : ℝ) : 3 * x^2 + 6 * x * y + 3 * y^2 = 3 * (x + y)^2 := 
  sorry

end factorize_expr_l3_3239


namespace combine_terms_implies_mn_l3_3433

theorem combine_terms_implies_mn {m n : ℕ} (h1 : m = 2) (h2 : n = 3) : m ^ n = 8 :=
by
  -- We will skip the proof here
  sorry

end combine_terms_implies_mn_l3_3433


namespace curves_intersect_l3_3574

-- Declare points and curve definitions
def C1 (t : ℝ) : ℝ × ℝ := (4 + t, 5 + 2 * t)
def C2 (α : ℝ) : ℝ × ℝ := (3 + 5 * Real.cos α, 5 + 5 * Real.sin α)

-- Equation forms of the curves
def C1_cartesian (x y : ℝ) : Prop := y = 2 * x - 3
def C2_cartesian (x y : ℝ) : Prop := (x - 3)^2 + (y - 5)^2 = 25

theorem curves_intersect : ∃ x y, C1_cartesian x y ∧ C2_cartesian x y :=
begin
  sorry
end

end curves_intersect_l3_3574


namespace cost_of_one_dozen_pens_l3_3350

theorem cost_of_one_dozen_pens
  (x : ℝ)
  (hx : 20 * x = 150) :
  12 * 5 * (150 / 20) = 450 :=
by
  sorry

end cost_of_one_dozen_pens_l3_3350


namespace negation_proposition_l3_3178

-- Definitions based on the conditions
def original_proposition : Prop := ∃ x : ℝ, x^2 + 3*x + 2 < 0

-- Theorem requiring proof
theorem negation_proposition : (¬ original_proposition) = ∀ x : ℝ, x^2 + 3*x + 2 ≥ 0 :=
by
  sorry

end negation_proposition_l3_3178


namespace cost_per_slice_in_cents_l3_3625

def loaves : ℕ := 3
def slices_per_loaf : ℕ := 20
def total_payment : ℕ := 2 * 20
def change : ℕ := 16
def total_cost : ℕ := total_payment - change
def total_slices : ℕ := loaves * slices_per_loaf

theorem cost_per_slice_in_cents :
  (total_cost : ℕ) * 100 / total_slices = 40 :=
by
  sorry

end cost_per_slice_in_cents_l3_3625


namespace exhaust_pipe_leak_time_l3_3063

theorem exhaust_pipe_leak_time : 
  (∃ T : Real, T > 0 ∧ 
                (1 / 10 - 1 / T) = 1 / 59.999999999999964 ∧ 
                T = 12) :=
by
  sorry

end exhaust_pipe_leak_time_l3_3063


namespace soaps_in_one_package_l3_3479

theorem soaps_in_one_package (boxes : ℕ) (packages_per_box : ℕ) (total_packages : ℕ) (total_soaps : ℕ) : 
  boxes = 2 → packages_per_box = 6 → total_packages = boxes * packages_per_box → total_soaps = 2304 → (total_soaps / total_packages) = 192 :=
by
  intros h_boxes h_packages_per_box h_total_packages h_total_soaps
  sorry

end soaps_in_one_package_l3_3479


namespace gcf_7fact_8fact_l3_3412

-- Definitions based on the conditions
def factorial (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | (n + 1) => (n + 1) * factorial n

noncomputable def greatest_common_divisor (a b : ℕ) : ℕ :=
  Nat.gcd a b

-- Theorem statement
theorem gcf_7fact_8fact : greatest_common_divisor (factorial 7) (factorial 8) = 5040 := by
  sorry

end gcf_7fact_8fact_l3_3412


namespace sin_330_eq_neg_one_half_l3_3961

theorem sin_330_eq_neg_one_half : Real.sin (330 * Real.pi / 180) = -1 / 2 := by
  sorry

end sin_330_eq_neg_one_half_l3_3961


namespace sin_330_eq_neg_sqrt3_div_2_l3_3997

theorem sin_330_eq_neg_sqrt3_div_2 
  (R : ℝ × ℝ)
  (hR : R = (1/2, -sqrt(3)/2))
  : Real.sin (330 * Real.pi / 180) = -sqrt(3)/2 :=
by
  sorry

end sin_330_eq_neg_sqrt3_div_2_l3_3997


namespace sin_330_eq_neg_half_l3_3921

open Real

theorem sin_330_eq_neg_half : sin (330 * (π / 180)) = - (1 / 2) := by
  have angle_eq : 330 * (π / 180) = 2 * π - (30 * (π / 180)) := by sorry
  have sin_30_pos : sin (30 * (π / 180)) = 1 / 2 := by sorry
  have sin_330_neg : sin (2 * π - (30 * (π / 180))) = - (sin (30 * (π / 180))) := by sorry
  rw [angle_eq, sin_330_neg, sin_30_pos]
  rfl

end sin_330_eq_neg_half_l3_3921


namespace bowling_ball_weight_l3_3420

-- Definitions based on given conditions
variable (k b : ℕ)

-- Condition 1: one kayak weighs 35 pounds
def kayak_weight : Prop := k = 35

-- Condition 2: four kayaks weigh the same as five bowling balls
def balance_equation : Prop := 4 * k = 5 * b

-- Goal: prove the weight of one bowling ball is 28 pounds
theorem bowling_ball_weight (hk : kayak_weight k) (hb : balance_equation k b) : b = 28 :=
by
  sorry

end bowling_ball_weight_l3_3420


namespace sin_330_eq_neg_one_half_l3_3959

theorem sin_330_eq_neg_one_half : Real.sin (330 * Real.pi / 180) = -1 / 2 := by
  sorry

end sin_330_eq_neg_one_half_l3_3959


namespace retail_price_l3_3491

/-- A retailer bought a machine at a wholesale price of $99 and later sold it after a 10% discount of the retail price.
If the retailer made a profit equivalent to 20% of the wholesale price, then the retail price of the machine before the discount was $132. -/
theorem retail_price (wholesale_price : ℝ) (profit_percent discount_percent : ℝ) (P : ℝ) 
  (h₁ : wholesale_price = 99) 
  (h₂ : profit_percent = 0.20) 
  (h₃ : discount_percent = 0.10)
  (h₄ : (1 - discount_percent) * P = wholesale_price + profit_percent * wholesale_price) : 
  P = 132 := 
by
  sorry

end retail_price_l3_3491


namespace smallest_whole_number_larger_than_any_triangle_perimeter_l3_3197

def is_valid_triangle (a b c : ℕ) : Prop := 
  (a + b > c) ∧ (a + c > b) ∧ (b + c > a)

theorem smallest_whole_number_larger_than_any_triangle_perimeter : 
  ∀ (s : ℕ), 16 < s ∧ s < 30 → is_valid_triangle 7 23 s → 
    60 = (Nat.succ (7 + 23 + s - 1)) := 
by 
  sorry

end smallest_whole_number_larger_than_any_triangle_perimeter_l3_3197


namespace age_product_difference_l3_3373

theorem age_product_difference (age_today : ℕ) (product_today : ℕ) (product_next_year : ℕ) :
  age_today = 7 →
  product_today = age_today * age_today →
  product_next_year = (age_today + 1) * (age_today + 1) →
  product_next_year - product_today = 15 :=
by
  sorry

end age_product_difference_l3_3373


namespace sin_330_correct_l3_3966

noncomputable def sin_330 : ℝ := sorry

theorem sin_330_correct : sin_330 = -1 / 2 :=
  sorry

end sin_330_correct_l3_3966


namespace f_2008_eq_zero_l3_3118

noncomputable def f : ℝ → ℝ := sorry

-- f is odd function
axiom f_odd : ∀ x : ℝ, f (-x) = -f x

-- f satisfies f(x + 2) = -f(x)
axiom f_periodic : ∀ x : ℝ, f (x + 2) = -f x

theorem f_2008_eq_zero : f 2008 = 0 :=
by
  sorry

end f_2008_eq_zero_l3_3118


namespace difference_between_two_smallest_integers_l3_3623

-- We will use the concepts of modular arithmetic and LCM
theorem difference_between_two_smallest_integers (k : ℕ) (hk : 2 ≤ k ∧ k ≤ 12) :
  let n1 := nat.lcm (list.range' 2 11) + 1,
      n2 := 2 * nat.lcm (list.range' 2 11) + 1
  in n2 - n1 = 4620 := 
by
  sorry

end difference_between_two_smallest_integers_l3_3623


namespace find_b_in_quadratic_eqn_l3_3769

theorem find_b_in_quadratic_eqn :
  ∃ (b : ℝ), ∃ (p : ℝ), 
  (∀ x, x^2 + b*x + 64 = (x + p)^2 + 16) → 
  b = 8 * Real.sqrt 3 :=
by 
  sorry

end find_b_in_quadratic_eqn_l3_3769


namespace calculate_yield_l3_3737

-- Define the conditions
def x := 6
def x_pos := 3
def x_tot := 3 * x
def nuts_x_pos := x + x_pos
def nuts_x := x
def nuts_x_neg := x - x_pos
def yield_x_pos := 60
def yield_x := 120
def avg_yield := 100

-- Calculate yields
def nuts_x_pos_yield : ℕ := nuts_x_pos * yield_x_pos
def nuts_x_yield : ℕ := nuts_x * yield_x
noncomputable def total_yield (yield_x_neg : ℕ) : ℕ :=
  nuts_x_pos_yield + nuts_x_yield + nuts_x_neg * yield_x_neg

-- Equation combining all
lemma yield_per_tree : (total_yield Y) / x_tot = avg_yield := sorry

-- Prove Y = 180
theorem calculate_yield : (x = 6 → ((nuts_x_neg * 180 = 540) ∧ rate = 180)) := sorry

end calculate_yield_l3_3737


namespace max_writers_and_editors_l3_3206

theorem max_writers_and_editors (T W : ℕ) (E : ℕ) (x : ℕ) (hT : T = 100) (hW : W = 35) (hE : E > 38) (h_comb : W + E + x = T)
    (h_neither : T = W + E + x) : x = 26 := by
  sorry

end max_writers_and_editors_l3_3206


namespace stratified_sampling_medium_supermarkets_l3_3567

theorem stratified_sampling_medium_supermarkets
  (large_supermarkets : ℕ)
  (medium_supermarkets : ℕ)
  (small_supermarkets : ℕ)
  (sample_size : ℕ)
  (total_supermarkets : ℕ)
  (medium_proportion : ℚ) :
  large_supermarkets = 200 →
  medium_supermarkets = 400 →
  small_supermarkets = 1400 →
  sample_size = 100 →
  total_supermarkets = large_supermarkets + medium_supermarkets + small_supermarkets →
  medium_proportion = (medium_supermarkets : ℚ) / (total_supermarkets : ℚ) →
  medium_supermarkets_to_sample = sample_size * medium_proportion →
  medium_supermarkets_to_sample = 20 :=
sorry

end stratified_sampling_medium_supermarkets_l3_3567


namespace smallest_date_for_first_Saturday_after_second_Monday_following_second_Thursday_l3_3485

theorem smallest_date_for_first_Saturday_after_second_Monday_following_second_Thursday :
  ∃ d : ℕ, d = 17 :=
by
  -- Assuming the starting condition that the month starts such that the second Thursday is on the 8th
  let second_thursday := 8

  -- Calculate second Monday after the second Thursday
  let second_monday := second_thursday + 4
  
  -- Calculate first Saturday after the second Monday
  let first_saturday := second_monday + 5

  have smallest_date : first_saturday = 17 := rfl
  
  exact ⟨first_saturday, smallest_date⟩

end smallest_date_for_first_Saturday_after_second_Monday_following_second_Thursday_l3_3485


namespace coin_flip_sequences_l3_3211

theorem coin_flip_sequences :
  let total_sequences := 2^10
  let sequences_starting_with_two_heads := 2^8
  total_sequences - sequences_starting_with_two_heads = 768 :=
by
  sorry

end coin_flip_sequences_l3_3211


namespace conditional_probability_l3_3192

variable (Ω : Type) [Fintype Ω] [DecidableEq Ω]

namespace conditional_probability_problem

-- Define the sample space for the experiment
def sample_space : Finset (Finset Ω) := 
  {s ∈ (univ : Finset (Finset Ω)) | s.card = 3}

-- Define events A and B
def A (s : Finset Ω) : Prop := s.card = 3
def B (s : Finset Ω) : Prop := ∃ a ∈ s, ∀ b ∈ s, b ≠ a

-- Given the sample space Ω, define the conditional probability
def P (A B : Finset Ω → Prop) : ℝ := 
  (Finset.card (sample_space.filter (λ s, A s ∧ B s))).to_real / 
  (Finset.card (sample_space.filter B)).to_real

theorem conditional_probability :
  P A B = 1 / 2 := by
  sorry

end conditional_probability_problem

end conditional_probability_l3_3192


namespace find_maximum_marks_l3_3223

theorem find_maximum_marks (M : ℝ) 
  (h1 : 0.60 * M = 270)
  (h2 : ∀ x : ℝ, 220 + 50 = x → x = 270) : 
  M = 450 :=
by
  sorry

end find_maximum_marks_l3_3223


namespace labourer_monthly_income_l3_3471

-- Define the conditions
def total_expense_first_6_months : ℕ := 90 * 6
def total_expense_next_4_months : ℕ := 60 * 4
def debt_cleared_and_savings : ℕ := 30

-- Define the monthly income
def monthly_income : ℕ := 81

-- The statement to be proven
theorem labourer_monthly_income (I D : ℕ) (h1 : 6 * I + D = total_expense_first_6_months) 
                               (h2 : 4 * I - D = total_expense_next_4_months + debt_cleared_and_savings) :
  I = monthly_income :=
by {
  sorry
}

end labourer_monthly_income_l3_3471


namespace minimum_value_expression_l3_3744

variable (a b c k : ℝ)
variable (h_pos : a > 0 ∧ b > 0 ∧ c > 0)
variable (h_eq : a = k ∧ b = k ∧ c = k)

theorem minimum_value_expression : 
  (a + b + c) * (1 / (a + b) + 1 / (a + c) + 1 / (b + c)) = 9 / 2 :=
by
  sorry

end minimum_value_expression_l3_3744


namespace find_angle_l3_3673

theorem find_angle (x : Real) : 
  (x - (1 / 2) * (180 - x) = -18 - 24/60 - 36/3600) -> 
  x = 47 + 43/60 + 36/3600 :=
by
  sorry

end find_angle_l3_3673


namespace suitable_comprehensive_survey_l3_3519

def investigate_service_life_of_lamps : Prop := 
  -- This would typically involve checking a subset rather than every lamp
  sorry

def investigate_water_quality : Prop := 
  -- This would typically involve sampling rather than checking every point
  sorry

def investigate_sports_activities : Prop := 
  -- This would typically involve sampling rather than collecting data on every student
  sorry

def test_components_of_rocket : Prop := 
  -- Given the critical importance and manageable number of components, this requires comprehensive examination
  sorry

def most_suitable_for_comprehensive_survey : Prop :=
  test_components_of_rocket ∧ ¬investigate_service_life_of_lamps ∧ 
  ¬investigate_water_quality ∧ ¬investigate_sports_activities

theorem suitable_comprehensive_survey : most_suitable_for_comprehensive_survey :=
  sorry

end suitable_comprehensive_survey_l3_3519


namespace sin_330_eq_neg_half_l3_3843

theorem sin_330_eq_neg_half 
  (h1: (330 : ℝ) = 330)
  (h2: (330 : ℝ = 360 - 30))
  (h3: 30 ∈ reference_angle_set ∧ sin 30 = 1/2)
  (h4: 330 ∈ fourth_quadrant)
  (h5: ∀ θ ∈ fourth_quadrant, sin θ < 0) :
  sin 330 = -1/2 := 
by
  sorry

end sin_330_eq_neg_half_l3_3843


namespace B_gain_correct_l3_3562

noncomputable def compound_interest (P : ℝ) (r : ℝ) (n : ℕ) (t : ℝ) : ℝ :=
  P * (1 + r / n) ^ (n * t)

noncomputable def gain_of_B : ℝ :=
  let principal : ℝ := 3150
  let interest_rate_A_to_B : ℝ := 0.08
  let annual_compound : ℕ := 1
  let time_A_to_B : ℝ := 3

  let interest_rate_B_to_C : ℝ := 0.125
  let semiannual_compound : ℕ := 2
  let time_B_to_C : ℝ := 2.5

  let amount_A_to_B := compound_interest principal interest_rate_A_to_B annual_compound time_A_to_B
  let amount_B_to_C := compound_interest principal interest_rate_B_to_C semiannual_compound time_B_to_C

  amount_B_to_C - amount_A_to_B

theorem B_gain_correct : gain_of_B = 282.32 :=
  sorry

end B_gain_correct_l3_3562


namespace arithmetic_sequences_count_l3_3290

noncomputable def countArithmeticSequences (n : ℕ) : ℕ :=
  if n % 2 = 0 then (n^2) / 4 else (n^2 - 1) / 4

theorem arithmetic_sequences_count :
  ∀ n : ℕ, countArithmeticSequences n = if n % 2 = 0 then (n^2) / 4 else (n^2 - 1) / 4 :=
by sorry

end arithmetic_sequences_count_l3_3290


namespace percentage_decrease_in_area_l3_3518

variable (L B : ℝ)

def original_area (L B : ℝ) : ℝ := L * B
def new_length (L : ℝ) : ℝ := 0.70 * L
def new_breadth (B : ℝ) : ℝ := 0.85 * B
def new_area (L B : ℝ) : ℝ := new_length L * new_breadth B

theorem percentage_decrease_in_area (L B : ℝ) (hL : L > 0) (hB : B > 0) :
  ((original_area L B - new_area L B) / original_area L B) * 100 = 40.5 :=
by
  sorry

end percentage_decrease_in_area_l3_3518


namespace simplify_expression_l3_3303

theorem simplify_expression (x : ℝ) : (2 * x + 20) + (150 * x + 20) = 152 * x + 40 := 
by 
  sorry

end simplify_expression_l3_3303


namespace trigonometric_identity_l3_3010

theorem trigonometric_identity :
  (1 / Real.cos 80) - (Real.sqrt 3 / Real.cos 10) = 4 :=
by
  sorry

end trigonometric_identity_l3_3010


namespace mixed_groups_count_l3_3034

-- Define the facts about the groups and photographs
def numberOfChildren : Nat := 300
def numberOfGroups : Nat := 100
def childrenPerGroup : Nat := 3
def b_b_photos : Nat := 100
def g_g_photos : Nat := 56

-- Define the function to calculate mixed groups
def mixedGroups (totalPhotos b_b_photos g_g_photos : Nat) : Nat := 
  (totalPhotos - b_b_photos - g_g_photos) / 2

-- State the theorem
theorem mixed_groups_count : 
  mixedGroups (numberOfGroups * childrenPerGroup) b_b_photos g_g_photos = 72 := by
  rfl

end mixed_groups_count_l3_3034


namespace min_moves_to_visit_all_non_forbidden_squares_l3_3256

def min_diagonal_moves (n : ℕ) : ℕ :=
  2 * (n / 2) - 1

theorem min_moves_to_visit_all_non_forbidden_squares (n : ℕ) :
  min_diagonal_moves n = 2 * (n / 2) - 1 := by
  sorry

end min_moves_to_visit_all_non_forbidden_squares_l3_3256


namespace total_litter_pieces_l3_3057

-- Define the number of glass bottles and aluminum cans as constants.
def glass_bottles : ℕ := 10
def aluminum_cans : ℕ := 8

-- State the theorem that the sum of glass bottles and aluminum cans is 18.
theorem total_litter_pieces : glass_bottles + aluminum_cans = 18 := by
  sorry

end total_litter_pieces_l3_3057


namespace sin_330_eq_neg_sin_30_l3_3981

theorem sin_330_eq_neg_sin_30 :
  sin (330 : ℝ) = - sin (30 : ℝ) := sorry

end sin_330_eq_neg_sin_30_l3_3981


namespace balloons_left_after_distribution_l3_3804

-- Definitions for the conditions
def red_balloons : ℕ := 23
def blue_balloons : ℕ := 39
def green_balloons : ℕ := 71
def yellow_balloons : ℕ := 89
def total_balloons : ℕ := red_balloons + blue_balloons + green_balloons + yellow_balloons
def number_of_friends : ℕ := 10

-- Statement to prove the correct answer
theorem balloons_left_after_distribution : total_balloons % number_of_friends = 2 :=
by
  -- The proof would go here
  sorry

end balloons_left_after_distribution_l3_3804


namespace no_point_on_line_y_eq_2x_l3_3593

theorem no_point_on_line_y_eq_2x
  (marked : Set (ℕ × ℕ))
  (initial_points : { p // p ∈ [(1, 1), (2, 3), (4, 5), (999, 111)] })
  (rule1 : ∀ a b, (a, b) ∈ marked → (b, a) ∈ marked ∧ (a - b, a + b) ∈ marked)
  (rule2 : ∀ a b c d, (a, b) ∈ marked ∧ (c, d) ∈ marked → (a * d + b * c, 4 * a * c - 4 * b * d) ∈ marked) :
  ∃ x, (x, 2 * x) ∈ marked → False := sorry

end no_point_on_line_y_eq_2x_l3_3593


namespace sin_330_eq_neg_half_l3_3838

theorem sin_330_eq_neg_half 
  (h1: (330 : ℝ) = 330)
  (h2: (330 : ℝ = 360 - 30))
  (h3: 30 ∈ reference_angle_set ∧ sin 30 = 1/2)
  (h4: 330 ∈ fourth_quadrant)
  (h5: ∀ θ ∈ fourth_quadrant, sin θ < 0) :
  sin 330 = -1/2 := 
by
  sorry

end sin_330_eq_neg_half_l3_3838


namespace inequality_proof_l3_3731

theorem inequality_proof (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) :
  (1 / (2 * x) + 1 / (2 * y) + 1 / (2 * z)) > 
  (1 / (y + z) + 1 / (z + x) + 1 / (x + y)) :=
  by
    let a := y + z
    let b := z + x
    let c := x + y
    have x_def : x = (a + c - b) / 2 := sorry
    have y_def : y = (a + b - c) / 2 := sorry
    have z_def : z = (b + c - a) / 2 := sorry
    sorry

end inequality_proof_l3_3731


namespace five_fridays_in_september_l3_3768

theorem five_fridays_in_september (year : ℕ) :
  (∃ (july_wednesdays : ℕ × ℕ × ℕ × ℕ × ℕ), 
     (july_wednesdays = (1, 8, 15, 22, 29) ∨ 
      july_wednesdays = (2, 9, 16, 23, 30) ∨ 
      july_wednesdays = (3, 10, 17, 24, 31)) ∧ 
      september_days = 30) → 
  ∃ (september_fridays : ℕ × ℕ × ℕ × ℕ × ℕ), 
  (september_fridays = (1, 8, 15, 22, 29)) :=
by
  sorry

end five_fridays_in_september_l3_3768


namespace sin_330_correct_l3_3972

noncomputable def sin_330 : ℝ := sorry

theorem sin_330_correct : sin_330 = -1 / 2 :=
  sorry

end sin_330_correct_l3_3972


namespace sum_of_fractions_equals_16_l3_3823

def list_of_fractions : List (ℚ) := [
  2 / 10,
  4 / 10,
  6 / 10,
  8 / 10,
  10 / 10,
  15 / 10,
  20 / 10,
  25 / 10,
  30 / 10,
  40 / 10
]

theorem sum_of_fractions_equals_16 : list_of_fractions.sum = 16 := by
  sorry

end sum_of_fractions_equals_16_l3_3823


namespace eggs_not_eaten_is_6_l3_3756

noncomputable def eggs_not_eaten_each_week 
  (trays_purchased : ℕ) 
  (eggs_per_tray : ℕ) 
  (eggs_morning : ℕ) 
  (days_in_week : ℕ) 
  (eggs_night : ℕ) : ℕ :=
  let total_eggs := trays_purchased * eggs_per_tray
  let eggs_eaten_son_daughter := eggs_morning * days_in_week
  let eggs_eaten_rhea_husband := eggs_night * days_in_week
  let eggs_eaten_total := eggs_eaten_son_daughter + eggs_eaten_rhea_husband
  total_eggs - eggs_eaten_total

theorem eggs_not_eaten_is_6 
  (trays_purchased : ℕ := 2) 
  (eggs_per_tray : ℕ := 24) 
  (eggs_morning : ℕ := 2) 
  (days_in_week : ℕ := 7) 
  (eggs_night : ℕ := 4) : 
  eggs_not_eaten_each_week trays_purchased eggs_per_tray eggs_morning days_in_week eggs_night = 6 :=
by
  -- Here should be proof steps, but we use sorry to skip it as per instruction
  sorry

end eggs_not_eaten_is_6_l3_3756


namespace factor_product_l3_3798

theorem factor_product : 2^2 * 3^2 * 5^2 * 7 = 6300 := by
  sorry

end factor_product_l3_3798


namespace minimal_fraction_difference_l3_3286

theorem minimal_fraction_difference :
  ∃ (p q : ℕ), 0 < p ∧ 0 < q ∧ (3 : ℚ) / 5 < (p : ℚ) / q ∧ (p : ℚ) / q < (2 : ℚ) / 3 ∧
  (∀ r s : ℕ, (3 : ℚ) / 5 < (r : ℚ) / s ∧ (r : ℚ) / s < (2 : ℚ) / 3 → 0 < s → q ≤ s) ∧
  q - p = 3 :=
begin
  sorry
end

end minimal_fraction_difference_l3_3286


namespace markup_is_correct_l3_3779

-- The mathematical interpretation of the given conditions
def purchase_price : ℝ := 48
def overhead_percentage : ℝ := 0.05
def net_profit : ℝ := 12

-- Define the overhead cost calculation
def overhead_cost : ℝ := overhead_percentage * purchase_price

-- Define the total cost calculation
def total_cost : ℝ := purchase_price + overhead_cost

-- Define the selling price calculation
def selling_price : ℝ := total_cost + net_profit

-- Define the markup calculation
def markup : ℝ := selling_price - purchase_price

-- The statement we want to prove
theorem markup_is_correct : markup = 14.40 :=
by
  -- We will eventually prove this, but for now we use sorry as a placeholder
  sorry

end markup_is_correct_l3_3779


namespace scientific_notation_400000000_l3_3370

theorem scientific_notation_400000000 : 400000000 = 4 * 10^8 :=
by
  sorry

end scientific_notation_400000000_l3_3370


namespace exponentiation_81_5_4_eq_243_l3_3091

theorem exponentiation_81_5_4_eq_243 : 81^(5/4) = 243 := by
  sorry

end exponentiation_81_5_4_eq_243_l3_3091


namespace tickets_not_went_to_concert_l3_3295

theorem tickets_not_went_to_concert :
  let total_tickets := 900
  let before_start := total_tickets * 3 / 4
  let remaining_after_start := total_tickets - before_start
  let after_first_song := remaining_after_start * 5 / 9
  let during_middle := 80
  remaining_after_start - (after_first_song + during_middle) = 20 := 
by
  let total_tickets := 900
  let before_start := total_tickets * 3 / 4
  let remaining_after_start := total_tickets - before_start
  let after_first_song := remaining_after_start * 5 / 9
  let during_middle := 80
  show remaining_after_start - (after_first_song + during_middle) = 20
  sorry

end tickets_not_went_to_concert_l3_3295


namespace second_reduction_is_18_point_1_percent_l3_3667

noncomputable def second_reduction_percentage (P : ℝ) : ℝ :=
  let first_price := 0.91 * P
  let second_price := 0.819 * P
  let R := (first_price - second_price) / first_price
  R * 100

theorem second_reduction_is_18_point_1_percent (P : ℝ) : second_reduction_percentage P = 18.1 :=
by
  -- Proof omitted
  sorry

end second_reduction_is_18_point_1_percent_l3_3667


namespace red_peaches_count_l3_3189

/-- Math problem statement:
There are some red peaches and 16 green peaches in the basket.
There is 1 more red peach than green peaches in the basket.
Prove that the number of red peaches in the basket is 17.
--/

-- Let G be the number of green peaches and R be the number of red peaches.
def G : ℕ := 16
def R : ℕ := G + 1

theorem red_peaches_count : R = 17 := by
  sorry

end red_peaches_count_l3_3189


namespace sin_330_eq_neg_half_l3_3925

open Real

theorem sin_330_eq_neg_half : sin (330 * (π / 180)) = - (1 / 2) := by
  have angle_eq : 330 * (π / 180) = 2 * π - (30 * (π / 180)) := by sorry
  have sin_30_pos : sin (30 * (π / 180)) = 1 / 2 := by sorry
  have sin_330_neg : sin (2 * π - (30 * (π / 180))) = - (sin (30 * (π / 180))) := by sorry
  rw [angle_eq, sin_330_neg, sin_30_pos]
  rfl

end sin_330_eq_neg_half_l3_3925


namespace find_angle_l3_3672

def degree := ℝ

def complement (x : degree) : degree := 180 - x

def angle_condition (x : degree) : Prop :=
  x - (complement x / 2) = -18 - 24/60 - 36/3600

theorem find_angle : ∃ x : degree, angle_condition x ∧ x = 47 + 43/60 + 36/3600 :=
by
  sorry

end find_angle_l3_3672


namespace sin_330_eq_neg_half_l3_3889

theorem sin_330_eq_neg_half : Real.sin (330 * Real.pi / 180) = -1 / 2 := by
  -- Definitions and conditions from the problem
  have h1 : 330 = 360 - 30 := rfl
  have h2 : ∀ θ, Real.sin θ = -Real.sin (θ - 2 * Real.pi) := sorry
  have h3 : Real.sin (30 * Real.pi / 180) = 1 / 2 := sorry

  -- Use the given conditions to prove the equality
  calc
    Real.sin (330 * Real.pi / 180)
        = Real.sin (-(30 * Real.pi / 180) + 2 * Real.pi) : by rw [←h1, Real.sin_angle_sub_pi]
    ... = -Real.sin (30 * Real.pi / 180) : by rw [h2]
    ... = -1 / 2 : by rw [h3]

end sin_330_eq_neg_half_l3_3889


namespace mixed_groups_count_l3_3030

theorem mixed_groups_count :
  ∀ (total_children groups_of_3 total_photos boys_photos girls_photos : ℕ),
  total_children = 300 ∧
  groups_of_3 = 100 ∧
  total_photos = 300 ∧
  boys_photos = 100 ∧
  girls_photos = 56 →
  let mixed_photos := total_photos - boys_photos - girls_photos in
  let mixed_groups := mixed_photos / 2 in
  mixed_groups = 72 :=
by
  intros total_children groups_of_3 total_photos boys_photos girls_photos h,
  have h1 : mixed_photos = total_photos - boys_photos - girls_photos := rfl,
  have h2 : mixed_groups = mixed_photos / 2 := rfl,
  rw [h1, h2],
  simp [h],
  sorry

end mixed_groups_count_l3_3030


namespace total_ingredients_l3_3628

theorem total_ingredients (water : ℕ) (flour : ℕ) (salt : ℕ)
  (h_water : water = 10)
  (h_flour : flour = 16)
  (h_salt : salt = flour / 2) :
  water + flour + salt = 34 :=
by
  sorry

end total_ingredients_l3_3628


namespace greatest_integer_floor_div_l3_3195

-- Define the parameters
def a : ℕ := 3^100 + 2^105
def b : ℕ := 3^96 + 2^101

-- Formulate the proof statement
theorem greatest_integer_floor_div (a b : ℕ) : 
  a = 3^100 + 2^105 →
  b = 3^96 + 2^101 →
  (a / b) = 16 := 
by
  intros ha hb
  sorry

end greatest_integer_floor_div_l3_3195


namespace fraction_product_simplified_l3_3688

theorem fraction_product_simplified:
  (2 / 3) * (4 / 7) * (9 / 11) = 24 / 77 := by
  sorry

end fraction_product_simplified_l3_3688


namespace negation_of_exisential_inequality_l3_3767

open Classical

theorem negation_of_exisential_inequality :
  ¬ (∃ x : ℝ, x^2 - x + 1/4 ≤ 0) ↔ ∀ x : ℝ, x^2 - x + 1/4 > 0 := 
by 
sorry

end negation_of_exisential_inequality_l3_3767


namespace difference_of_coordinates_l3_3299

-- Define point and its properties in Lean.
structure Point where
  x : ℝ
  y : ℝ

-- Define the midpoint property.
def is_midpoint (M A B : Point) : Prop :=
  M.x = (A.x + B.x) / 2 ∧ M.y = (A.y + B.y) / 2

-- Given points A and M
def A : Point := {x := 8, y := 0}
def M : Point := {x := 4, y := 1}

-- Assume B is a point with coordinates x and y
variable (B : Point)

-- The theorem to prove.
theorem difference_of_coordinates :
  is_midpoint M A B → B.x - B.y = -2 :=
by
  sorry

end difference_of_coordinates_l3_3299


namespace sin_330_eq_neg_half_l3_3940

noncomputable def Q : ℝ × ℝ := (Real.sqrt 3 / 2, -1 / 2)

theorem sin_330_eq_neg_half : Real.sin (330 * Real.pi / 180) = -1 / 2 :=
by
  have h1 : 330 * Real.pi / 180 = 11 * Real.pi / 6 := by sorry
  have coord_y : (Q.snd) = -1 / 2 := by sorry
  rw [h1, coord_y]
  sorry

end sin_330_eq_neg_half_l3_3940


namespace min_tan_of_acute_angle_l3_3454

def is_ocular_ray (u : ℚ) (x y : ℕ) : Prop :=
  1 ≤ x ∧ x ≤ 20 ∧ 1 ≤ y ∧ y ≤ 20 ∧ u = x / y

def acute_angle_tangent (u v : ℚ) : ℚ :=
  |(u - v) / (1 + u * v)|

theorem min_tan_of_acute_angle :
  ∃ θ : ℚ, (∀ u v : ℚ, (∃ x1 y1 x2 y2 : ℕ, is_ocular_ray u x1 y1 ∧ is_ocular_ray v x2 y2 ∧ u ≠ v) 
  → acute_angle_tangent u v ≥ θ) ∧ θ = 1 / 722 :=
sorry

end min_tan_of_acute_angle_l3_3454


namespace sector_area_72_20_eq_80pi_l3_3608

open Real

def sectorArea (θ : ℝ) (r : ℝ) : ℝ :=
  (θ / 360) * π * r^2

theorem sector_area_72_20_eq_80pi :
  sectorArea 72 20 = 80 * π := by
  sorry

end sector_area_72_20_eq_80pi_l3_3608


namespace bottles_more_than_apples_l3_3068

def regular_soda : ℕ := 72
def diet_soda : ℕ := 32
def apples : ℕ := 78

def total_bottles : ℕ := regular_soda + diet_soda

theorem bottles_more_than_apples : total_bottles - apples = 26 := by
  -- Proof will go here
  sorry

end bottles_more_than_apples_l3_3068


namespace sin_330_eq_neg_half_l3_3886

theorem sin_330_eq_neg_half : Real.sin (330 * Real.pi / 180) = -1 / 2 := by
  -- Definitions and conditions from the problem
  have h1 : 330 = 360 - 30 := rfl
  have h2 : ∀ θ, Real.sin θ = -Real.sin (θ - 2 * Real.pi) := sorry
  have h3 : Real.sin (30 * Real.pi / 180) = 1 / 2 := sorry

  -- Use the given conditions to prove the equality
  calc
    Real.sin (330 * Real.pi / 180)
        = Real.sin (-(30 * Real.pi / 180) + 2 * Real.pi) : by rw [←h1, Real.sin_angle_sub_pi]
    ... = -Real.sin (30 * Real.pi / 180) : by rw [h2]
    ... = -1 / 2 : by rw [h3]

end sin_330_eq_neg_half_l3_3886


namespace total_winter_clothing_l3_3345

theorem total_winter_clothing (boxes : ℕ) (scarves_per_box mittens_per_box : ℕ) (h_boxes : boxes = 8) (h_scarves : scarves_per_box = 4) (h_mittens : mittens_per_box = 6) : 
  boxes * (scarves_per_box + mittens_per_box) = 80 := 
by
  sorry

end total_winter_clothing_l3_3345


namespace exists_route_same_republic_l3_3440

noncomputable def cities := (Fin 100)
def republics := (Fin 3)
def connections (c : cities) : Finset cities := sorry
def owning_republic (c : cities) : republics := sorry

theorem exists_route_same_republic (H : ∃ (S : Finset cities), S.card ≥ 70 ∧ ∀ c ∈ S, (connections c).card ≥ 70) : 
  ∃ c₁ c₂ : cities, c₁ ≠ c₂ ∧ owning_republic c₁ = owning_republic c₂ ∧ connected_route c₁ c₂ :=
by
  sorry

end exists_route_same_republic_l3_3440


namespace initial_money_equals_26_l3_3294

def cost_jumper : ℕ := 9
def cost_tshirt : ℕ := 4
def cost_heels : ℕ := 5
def money_left : ℕ := 8

def total_cost_items : ℕ := cost_jumper + cost_tshirt + cost_heels

theorem initial_money_equals_26 : total_cost_items + money_left = 26 := by
  sorry

end initial_money_equals_26_l3_3294


namespace smallest_a₁_l3_3587

-- We define the sequence a_n and its recurrence relation
def a (n : ℕ) (a₁ : ℝ) : ℝ :=
  match n with
  | 0     => 0  -- this case is not used, but included for function completeness
  | 1     => a₁
  | (n+2) => 11 * a (n+1) a₁ - (n+2)

theorem smallest_a₁ : ∃ a₁ : ℝ, (a₁ = 21 / 100) ∧ ∀ n > 1, a n a₁ > 0 := 
  sorry

end smallest_a₁_l3_3587


namespace find_annual_interest_rate_l3_3247

-- Define the given conditions
def principal : ℝ := 10000
def time : ℝ := 1  -- since 12 months is 1 year for annual rate
def simple_interest : ℝ := 800

-- Define the annual interest rate to be proved
def annual_interest_rate : ℝ := 0.08

-- The theorem stating the problem
theorem find_annual_interest_rate (P : ℝ) (T : ℝ) (SI : ℝ) : 
  P = principal → 
  T = time → 
  SI = simple_interest → 
  SI = P * annual_interest_rate * T := 
by
  intros hP hT hSI
  rw [hP, hT, hSI]
  unfold annual_interest_rate
  -- here's where we skip the proof
  sorry

end find_annual_interest_rate_l3_3247


namespace robot_cost_max_units_A_l3_3466

noncomputable def cost_price_A (x : ℕ) := 1600
noncomputable def cost_price_B (x : ℕ) := 2800

theorem robot_cost (x : ℕ) (y : ℕ) (a : ℕ) (b : ℕ) :
  y = 2 * x - 400 →
  a = 96000 →
  b = 168000 →
  a / x = 6000 →
  b / y = 6000 →
  (x = 1600 ∧ y = 2800) :=
by sorry

theorem max_units_A (m n total_units : ℕ) : 
  total_units = 100 →
  m + n = 100 →
  m ≤ 2 * n →
  m ≤ 66 :=
by sorry

end robot_cost_max_units_A_l3_3466


namespace proof_problem_l3_3112

variables {m n : ℝ}

theorem proof_problem (h1 : m > 0) (h2 : n > 0) (h3 : m + n = 2 * m * n) :
  (mn : ℝ) ≥ 1 ∧ (m^2 + n^2 ≥ 2) :=
  sorry

end proof_problem_l3_3112


namespace alternating_sum_of_coefficients_l3_3537

noncomputable def polynomial_expansion (x : ℝ) : ℝ :=
  (2 * x + 1)^5

theorem alternating_sum_of_coefficients :
  ∀ (a_0 a_1 a_2 a_3 a_4 a_5 : ℝ), polynomial_expansion x = 
    a_0 + a_1 * x + a_2 * x^2 + a_3 * x^3 + a_4 * x^4 + a_5 * x^5 →
    a_0 - a_1 + a_2 - a_3 + a_4 - a_5 = -1 :=
by
  intros a_0 a_1 a_2 a_3 a_4 a_5 h
  sorry

end alternating_sum_of_coefficients_l3_3537


namespace average_speed_entire_journey_l3_3205

-- Define the average speed for the journey from x to y
def speed_xy := 60

-- Define the average speed for the journey from y to x
def speed_yx := 30

-- Definition for the distance (D) (it's an abstract value, so we don't need to specify)
variable (D : ℝ) (hD : D > 0)

-- Theorem stating that the average speed for the entire journey is 40 km/hr
theorem average_speed_entire_journey : 
  2 * D / ((D / speed_xy) + (D / speed_yx)) = 40 := 
by 
  sorry

end average_speed_entire_journey_l3_3205


namespace length_increase_100_l3_3297

theorem length_increase_100 (n : ℕ) (h : (n + 2) / 2 = 100) : n = 198 :=
sorry

end length_increase_100_l3_3297


namespace concert_attendance_problem_l3_3296

theorem concert_attendance_problem:
  (total_tickets sold before_start minutes_after first_song during_middle_part remaining not_go: ℕ) 
  (H1: total_tickets = 900)
  (H2: sold_before_start = (3 * total_tickets) / 4)
  (H3: remaining = total_tickets - sold_before_start)
  (H4: minutes_after_first_song = (5 * remaining) / 9)
  (H5: during_middle_part = 80)
  (H5_remaining: remaining - minutes_after_first_song - during_middle_part = not_go) :
  not_go = 20 :=
sorry

end concert_attendance_problem_l3_3296


namespace money_left_l3_3662

variable (S : ℚ)

-- Given conditions
def house_rent := (2/5) * S
def food := (3/10) * S
def conveyance := (1/8) * S
def total_food_conveyance := 3400

-- Given that total expenditure on food and conveyance is $3400
axiom h_food_conveyance : food + conveyance = total_food_conveyance

-- Prove the total money left after all expenditures
theorem money_left : S = 8000 → S - (house_rent + food + conveyance) = 1400 :=
by
  intros hS
  have h_total_expenditure : house_rent + food + conveyance = 3200 + 3400 :=
  sorry  -- This will be proven using further steps, skipping here for brevity
  show S - (house_rent + food + conveyance) = 1400 from
  sorry  -- Similar skipping of proof steps to comply with instructions

end money_left_l3_3662


namespace b_completion_days_l3_3347

theorem b_completion_days (x : ℝ) :
  (7 * (1 / 24 + 1 / x + 1 / 40) + 4 * (1 / 24 + 1 / x) = 1) → x = 26.25 := 
by 
  sorry

end b_completion_days_l3_3347


namespace combine_ingredients_l3_3627

theorem combine_ingredients : 
  ∃ (water flour salt : ℕ), 
    water = 10 ∧ flour = 16 ∧ salt = 1 / 2 * flour ∧ 
    (water + flour = 26) ∧ (salt = 8) :=
by
  sorry

end combine_ingredients_l3_3627


namespace andrew_game_night_expenses_l3_3453

theorem andrew_game_night_expenses : 
  let cost_per_game := 9 
  let number_of_games := 5 
  total_money_spent = cost_per_game * number_of_games 
→ total_money_spent = 45 := 
by
  intro cost_per_game number_of_games total_money_spent
  sorry

end andrew_game_night_expenses_l3_3453


namespace medium_supermarkets_in_sample_l3_3568

-- Define the conditions
def large_supermarkets : ℕ := 200
def medium_supermarkets : ℕ := 400
def small_supermarkets : ℕ := 1400
def total_supermarkets : ℕ := large_supermarkets + medium_supermarkets + small_supermarkets
def sample_size : ℕ := 100
def proportion_medium := (medium_supermarkets : ℚ) / (total_supermarkets : ℚ)

-- The main theorem to prove
theorem medium_supermarkets_in_sample : sample_size * proportion_medium = 20 := by
  sorry

end medium_supermarkets_in_sample_l3_3568


namespace total_revenue_from_sale_l3_3229

def total_weight_of_potatoes : ℕ := 6500
def weight_of_damaged_potatoes : ℕ := 150
def weight_per_bag : ℕ := 50
def price_per_bag : ℕ := 72

theorem total_revenue_from_sale :
  (total_weight_of_potatoes - weight_of_damaged_potatoes) / weight_per_bag * price_per_bag = 9144 := 
begin
  sorry
end

end total_revenue_from_sale_l3_3229


namespace money_left_after_shopping_l3_3758

-- Define the initial amount of money Sandy took for shopping
def initial_amount : ℝ := 310

-- Define the percentage of money spent in decimal form
def percentage_spent : ℝ := 0.30

-- Define the remaining money as per the given conditions
def remaining_money : ℝ := initial_amount * (1 - percentage_spent)

-- The statement we need to prove
theorem money_left_after_shopping :
  remaining_money = 217 :=
by
  sorry

end money_left_after_shopping_l3_3758


namespace cubed_expression_value_l3_3157

open Real

theorem cubed_expression_value (a b c : ℝ)
  (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0)
  (h : a + b + 2 * c = 0) :
  (a^3 + b^3 + 2 * c^3) / (a * b * c) = -3 * (a^2 - a * b + b^2) / (2 * a * b) :=
  sorry

end cubed_expression_value_l3_3157


namespace sin_330_l3_3901

theorem sin_330 : Real.sin (330 * Real.pi / 180) = -1 / 2 := 
by
  -- Outline the proof here without providing it
  -- sorry to delay the proof
  sorry

end sin_330_l3_3901


namespace valid_word_combinations_l3_3577

-- Definition of valid_combination based on given conditions
def valid_combination : ℕ :=
  26 * 5 * 26

-- Statement to prove the number of valid four-letter combinations is 3380
theorem valid_word_combinations : valid_combination = 3380 := by
  sorry

end valid_word_combinations_l3_3577


namespace sin_330_deg_l3_3846

theorem sin_330_deg : Real.sin (330 * Real.pi / 180) = -1 / 2 := by
  sorry

end sin_330_deg_l3_3846


namespace power_calculation_l3_3522

noncomputable def a : ℕ := 3 ^ 1006
noncomputable def b : ℕ := 7 ^ 1007
noncomputable def lhs : ℕ := (a + b)^2 - (a - b)^2
noncomputable def rhs : ℕ := 42 * (10 ^ 1007)

theorem power_calculation : lhs = rhs := by
  sorry

end power_calculation_l3_3522


namespace find_salary_J_l3_3054

variables {J F M A May : ℝ}
variables (h1 : (J + F + M + A) / 4 = 8000)
variables (h2 : (F + M + A + May) / 4 = 8200)
variables (h3 : May = 6500)

theorem find_salary_J : J = 5700 :=
by
  sorry

end find_salary_J_l3_3054


namespace point_on_line_l3_3615

theorem point_on_line :
  ∃ a b : ℝ, (a ≠ 0) ∧
  (∀ x y : ℝ, (x = 4 ∧ y = 5) ∨ (x = 8 ∧ y = 17) ∨ (x = 12 ∧ y = 29) → y = a * x + b) →
  (∃ t : ℝ, (15, t) ∈ {(x, y) | y = a * x + b} ∧ t = 38) :=
by
  sorry

end point_on_line_l3_3615


namespace circle_center_radius_sum_l3_3742

-- We define the circle equation as a predicate
def circle_eq (x y : ℝ) : Prop :=
  x^2 - 14 * x + y^2 + 16 * y + 100 = 0

-- We need to find that the center and radius satisfy a specific relationship
theorem circle_center_radius_sum :
  let a' := 7
  let b' := -8
  let r' := Real.sqrt 13
  a' + b' + r' = -1 + Real.sqrt 13 :=
by
  sorry

end circle_center_radius_sum_l3_3742


namespace probability_in_given_interval_l3_3511

noncomputable def length_interval (a b : ℝ) : ℝ := b - a

noncomputable def probability_in_interval (a b c d : ℝ) : ℝ :=
  (length_interval a b) / (length_interval c d)

theorem probability_in_given_interval : 
  probability_in_interval (-1) 1 (-2) 3 = 2 / 5 :=
by
  sorry

end probability_in_given_interval_l3_3511


namespace eval_power_l3_3087

theorem eval_power (h : 81 = 3^4) : 81^(5/4) = 243 := by
  sorry

end eval_power_l3_3087


namespace monthly_profit_10000_daily_profit_15000_maximize_profit_l3_3505

noncomputable def price_increase (c p: ℕ) (x: ℕ) : ℕ := c + x - p
noncomputable def sales_volume (s d: ℕ) (x: ℕ) : ℕ := s - d * x
noncomputable def monthly_profit (price cost volume: ℕ) : ℕ := (price - cost) * volume
noncomputable def monthly_profit_equation (x: ℕ) : ℕ := (40 + x - 30) * (600 - 10 * x)

theorem monthly_profit_10000 (x: ℕ) : monthly_profit_equation x = 10000 ↔ x = 10 ∨ x = 40 :=
by sorry

theorem daily_profit_15000 (x: ℕ) : ¬∃ x, monthly_profit_equation x = 15000 :=
by sorry

theorem maximize_profit (x p y: ℕ) : (∀ x, monthly_profit (40 + x) 30 (600 - 10 * x) ≤ y) ∧ y = 12250 ∧ x = 65 :=
by sorry

end monthly_profit_10000_daily_profit_15000_maximize_profit_l3_3505


namespace sin_330_eq_neg_one_half_l3_3885

theorem sin_330_eq_neg_one_half : sin (330 * π / 180) = -1 / 2 := by
  sorry

end sin_330_eq_neg_one_half_l3_3885


namespace evaluate_81_power_5_div_4_l3_3096

-- Define the conditions
def base_factorized : ℕ := 3 ^ 4
def power_rule (b : ℕ) (m n : ℝ) : ℝ := (b : ℝ) ^ m ^ n

-- Define the primary calculation
noncomputable def power_calculation : ℝ := 81 ^ (5 / 4)

-- Prove that the calculation equals 243
theorem evaluate_81_power_5_div_4 : power_calculation = 243 := 
by
  have h1 : base_factorized = 81 := by sorry
  have h2 : power_rule 3 4 (5 / 4) = 3 ^ 5 := by sorry
  have h3 : 3 ^ 5 = 243 := by sorry
  have h4 : power_calculation = power_rule 3 4 (5 / 4) := by sorry
  rw [h1, h2, h3, h4]
  exact h3

end evaluate_81_power_5_div_4_l3_3096


namespace exists_airline_route_within_same_republic_l3_3439

theorem exists_airline_route_within_same_republic
  (C : Type) [Fintype C] [DecidableEq C]
  (R : Type) [Fintype R] [DecidableEq R]
  (belongs_to : C → R)
  (airline_route : C → C → Prop)
  (country_size : Fintype.card C = 100)
  (republics_size : Fintype.card R = 3)
  (millionaire_cities : {c : C // ∃ n : ℕ, n ≥ 70 ∧ (∃ S : Finset C, S.card = n ∧ ∀ x ∈ S, airline_route c x) })
  (at_least_70_millionaire_cities : ∃ n : ℕ, n ≥ 70 ∧ (∃ S : Finset {c : C // ∃ n : ℕ, n ≥ 70 ∧ ( ∃ S : Finset C, S.card = n ∧ ∀ x ∈ S, airline_route c x )}, S.card = n)):
  ∃ (c1 c2 : C), airline_route c1 c2 ∧ belongs_to c1 = belongs_to c2 := 
sorry

end exists_airline_route_within_same_republic_l3_3439


namespace sales_tax_difference_l3_3053

/-- The difference in sales tax calculation given the changes in rate. -/
theorem sales_tax_difference 
  (market_price : ℝ := 9000) 
  (original_rate : ℝ := 0.035) 
  (new_rate : ℝ := 0.0333) 
  (difference : ℝ := 15.3) :
  market_price * original_rate - market_price * new_rate = difference :=
by
  /- The proof is omitted as per the instructions. -/
  sorry

end sales_tax_difference_l3_3053


namespace age_difference_two_children_l3_3058

/-!
# Age difference between two children in a family

## Given:
- 10 years ago, the average age of a family of 4 members was 24 years.
- Two children have been born since then.
- The present average age of the family (now 6 members) is the same, 24 years.
- The present age of the youngest child (Y1) is 3 years.

## Prove:
The age difference between the two children is 2 years.
-/

theorem age_difference_two_children :
  let Y1 := 3
  let Y2 := 5
  let total_age_10_years_ago := 4 * 24
  let total_age_now := 6 * 24
  let increase_age_10_years := total_age_now - total_age_10_years_ago
  let increase_due_to_original_members := 4 * 10
  let increase_due_to_children := increase_age_10_years - increase_due_to_original_members
  Y1 + Y2 = increase_due_to_children
  → Y2 - Y1 = 2 :=
by
  intros
  sorry

end age_difference_two_children_l3_3058


namespace only_setA_forms_triangle_l3_3488

-- Define the sets of line segments
def setA := [3, 5, 7]
def setB := [3, 6, 10]
def setC := [5, 5, 11]
def setD := [5, 6, 11]

-- Define a function to check the triangle inequality
def satisfies_triangle_inequality (a b c : Nat) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

-- Formalize the question
theorem only_setA_forms_triangle :
  satisfies_triangle_inequality 3 5 7 ∧
  ¬(satisfies_triangle_inequality 3 6 10) ∧
  ¬(satisfies_triangle_inequality 5 5 11) ∧
  ¬(satisfies_triangle_inequality 5 6 11) :=
by
  sorry

end only_setA_forms_triangle_l3_3488


namespace quadratic_roots_range_l3_3137

theorem quadratic_roots_range (k : ℝ) : (x^2 - 6*x + k = 0) → k < 9 := 
by
  sorry

end quadratic_roots_range_l3_3137


namespace shaded_area_l3_3320

theorem shaded_area (PQ : ℝ) (n_squares : ℕ) (d_intersect : ℝ)
  (h1 : PQ = 8) (h2 : n_squares = 20) (h3 : d_intersect = 8) : ∃ (A : ℝ), A = 160 := 
by {
  sorry
}

end shaded_area_l3_3320


namespace eliminate_denominators_l3_3343

theorem eliminate_denominators (x : ℝ) :
  (6 : ℝ) * ((x - 1) / 3) = (6 : ℝ) * (4 - (2 * x + 1) / 2) ↔ 2 * (x - 1) = 24 - 3 * (2 * x + 1) :=
by
  intros
  sorry

end eliminate_denominators_l3_3343


namespace pipe_fills_cistern_l3_3071

theorem pipe_fills_cistern (t : ℕ) (h : t = 5) : 11 * t = 55 :=
by
  sorry

end pipe_fills_cistern_l3_3071


namespace min_value_a_cube_plus_b_cube_no_exist_2a_plus_3b_eq_6_l3_3543

section ProofProblem

theorem min_value_a_cube_plus_b_cube {a b : ℝ} (ha : 0 < a) (hb : 0 < b) (hfra : 1/a + 1/b = Real.sqrt (a * b)) :
  a^3 + b^3 = 4 * Real.sqrt 2 := 
sorry

theorem no_exist_2a_plus_3b_eq_6 {a b : ℝ} (ha : 0 < a) (hb : 0 < b) (hfra : 1/a + 1/b = Real.sqrt (a * b)) :
  ¬ (2 * a + 3 * b = 6) :=
sorry

end ProofProblem

end min_value_a_cube_plus_b_cube_no_exist_2a_plus_3b_eq_6_l3_3543


namespace sin_330_l3_3934

theorem sin_330 : Real.sin (330 * Real.pi / 180) = -1 / 2 := 
by
  sorry

end sin_330_l3_3934


namespace nearest_integer_to_expression_correct_l3_3793

open Real

noncomputable def nearest_integer_to_expression : ℝ :=
  let a := (3 + (sqrt 5))^6
  let b := (3 - (sqrt 5))^6
  let s := a + b
  floor s

theorem nearest_integer_to_expression_correct :
  nearest_integer_to_expression = 74608 :=
begin
  sorry
end

end nearest_integer_to_expression_correct_l3_3793


namespace smallest_initial_number_sum_of_digits_l3_3677

theorem smallest_initial_number_sum_of_digits : ∃ (N : ℕ), 
  (0 ≤ N ∧ N < 1000) ∧ 
  ∃ (k : ℕ), 16 * N + 700 + 50 * k < 1000 ∧ 
  (N = 16) ∧ 
  (Nat.digits 10 N).sum = 7 := 
by
  sorry

end smallest_initial_number_sum_of_digits_l3_3677


namespace binom_2p_p_mod_p_l3_3745

theorem binom_2p_p_mod_p (p : ℕ) (hp : p.Prime) : Nat.choose (2 * p) p ≡ 2 [MOD p] := 
by
  sorry

end binom_2p_p_mod_p_l3_3745


namespace mixed_groups_count_l3_3028

theorem mixed_groups_count :
  ∀ (total_children groups_of_3 total_photos boys_photos girls_photos : ℕ),
  total_children = 300 ∧
  groups_of_3 = 100 ∧
  total_photos = 300 ∧
  boys_photos = 100 ∧
  girls_photos = 56 →
  let mixed_photos := total_photos - boys_photos - girls_photos in
  let mixed_groups := mixed_photos / 2 in
  mixed_groups = 72 :=
by
  intros total_children groups_of_3 total_photos boys_photos girls_photos h,
  have h1 : mixed_photos = total_photos - boys_photos - girls_photos := rfl,
  have h2 : mixed_groups = mixed_photos / 2 := rfl,
  rw [h1, h2],
  simp [h],
  sorry

end mixed_groups_count_l3_3028


namespace george_painting_combinations_l3_3252

namespace Combinations

/-- George's painting problem -/
theorem george_painting_combinations :
  let colors := 10
  let colors_to_pick := 3
  let textures := 2
  ((colors) * (colors - 1) * (colors - 2) / (colors_to_pick * (colors_to_pick - 1) * 1)) * (textures ^ colors_to_pick) = 960 :=
by
  sorry

end Combinations

end george_painting_combinations_l3_3252


namespace width_of_rectangular_prism_l3_3268

theorem width_of_rectangular_prism (l h d : ℕ) (w : ℤ) 
  (hl : l = 3) (hh : h = 12) (hd : d = 13) 
  (diag_eq : d = Int.sqrt (l^2 + w^2 + h^2)) : 
  w = 4 := by
  sorry

end width_of_rectangular_prism_l3_3268


namespace bernardo_wins_l3_3675

/-- 
Bernardo and Silvia play the following game. An integer between 0 and 999 inclusive is selected
and given to Bernardo. Whenever Bernardo receives a number, he doubles it and passes the result 
to Silvia. Whenever Silvia receives a number, she adds 50 to it and passes the result back. 
The winner is the last person who produces a number less than 1000. The smallest initial number 
that results in a win for Bernardo is 16, and the sum of the digits of 16 is 7.
-/
def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

theorem bernardo_wins (N : ℕ) (h : 16 ≤ N ∧ N ≤ 18) : sum_of_digits 16 = 7 :=
by
  sorry

end bernardo_wins_l3_3675


namespace sin_330_l3_3932

theorem sin_330 : Real.sin (330 * Real.pi / 180) = -1 / 2 := 
by
  sorry

end sin_330_l3_3932


namespace range_of_b_l3_3261

theorem range_of_b (f g : ℝ → ℝ) (a b : ℝ)
  (hf : ∀ x, f x = Real.exp x - 1)
  (hg : ∀ x, g x = -x^2 + 4*x - 3)
  (h : f a = g b) :
  2 - Real.sqrt 2 < b ∧ b < 2 + Real.sqrt 2 := by
  sorry

end range_of_b_l3_3261


namespace sin_330_degree_l3_3952

theorem sin_330_degree : Real.sin (330 * Real.pi / 180) = -1 / 2 :=
by
  sorry

end sin_330_degree_l3_3952


namespace area_convex_quadrilateral_l3_3064

theorem area_convex_quadrilateral (x y : ℝ) :
  (x^2 + y^2 = 73 ∧ x * y = 24) →
  -- You can place a formal statement specifying the four vertices here if needed
  ∃ a b c d : ℝ × ℝ,
  a.1^2 + a.2^2 = 73 ∧
  a.1 * a.2 = 24 ∧
  b.1^2 + b.2^2 = 73 ∧
  b.1 * b.2 = 24 ∧
  c.1^2 + c.2^2 = 73 ∧
  c.1 * c.2 = 24 ∧
  d.1^2 + d.2^2 = 73 ∧
  d.1 * d.2 = 24 ∧
  -- Ensure the quadrilateral forms a rectangle (additional conditions here)
  -- Compute the side lengths and area
  -- Specify finally the area and prove it equals 110
  True :=
sorry

end area_convex_quadrilateral_l3_3064


namespace mixed_groups_count_l3_3023

/-- Define the initial conditions --/
def number_of_children : ℕ := 300
def number_of_groups : ℕ := 100
def group_size : ℕ := 3
def boy_boy_photos : ℕ := 100
def girl_girl_photos : ℕ := 56

/-- Define the proof problem -/
theorem mixed_groups_count : 
    (number_of_children = 300) →
    (number_of_groups = 100) →
    (group_size = 3) →
    (boy_boy_photos = 100) →
    (girl_girl_photos = 56) →
    (∀ total_photos, total_photos = number_of_groups * group_size) →
    (∃ mixed_groups, mixed_groups = (total_photos - boy_boy_photos - girl_girl_photos) / 2) →
    mixed_groups = 72 :=
by
    intros h1 h2 h3 h4 h5 h6 h7
    sorry

end mixed_groups_count_l3_3023


namespace sin_330_deg_l3_3852

theorem sin_330_deg : Real.sin (330 * Real.pi / 180) = -1 / 2 := by
  sorry

end sin_330_deg_l3_3852


namespace turnip_pulled_by_mice_l3_3501

theorem turnip_pulled_by_mice :
  ∀ (M B G D J C : ℕ),
    D = 2 * B →
    B = 3 * G →
    G = 4 * J →
    J = 5 * C →
    C = 6 * M →
    (D + B + G + J + C + M) ≥ (D + B + G + J + C) + M → 
    1237 * M ≤ (D + B + G + J + C + M) :=
by
  intros M B G D J C h1 h2 h3 h4 h5 h6
  rw [h1, h2, h3, h4, h5]
  linarith

end turnip_pulled_by_mice_l3_3501


namespace odd_function_product_nonpositive_l3_3104

noncomputable def is_odd_function (f : ℝ → ℝ) := 
  ∀ x : ℝ, f (-x) = -f x

theorem odd_function_product_nonpositive (f : ℝ → ℝ) 
  (h_odd : is_odd_function f) : 
  ∀ x : ℝ, f x * f (-x) ≤ 0 :=
by 
  sorry

end odd_function_product_nonpositive_l3_3104


namespace factorize_expression_l3_3100

variable {x y : ℝ}

theorem factorize_expression :
  3 * x^2 - 27 * y^2 = 3 * (x + 3 * y) * (x - 3 * y) :=
by
  sorry

end factorize_expression_l3_3100


namespace arithmetic_geometric_sequence_solution_l3_3123

theorem arithmetic_geometric_sequence_solution 
  (a1 a2 b1 b2 b3 : ℝ) 
  (h1 : -2 * 2 + a2 = a1)
  (h2 : a1 * 2 - 8 = a2)
  (h3 : b2 ^ 2 = -2 * -8)
  (h4 : b2 = -4) :
  (a2 - a1) / b2 = 1 / 2 :=
by 
  sorry

end arithmetic_geometric_sequence_solution_l3_3123


namespace sin_330_eq_neg_half_l3_3944

noncomputable def Q : ℝ × ℝ := (Real.sqrt 3 / 2, -1 / 2)

theorem sin_330_eq_neg_half : Real.sin (330 * Real.pi / 180) = -1 / 2 :=
by
  have h1 : 330 * Real.pi / 180 = 11 * Real.pi / 6 := by sorry
  have coord_y : (Q.snd) = -1 / 2 := by sorry
  rw [h1, coord_y]
  sorry

end sin_330_eq_neg_half_l3_3944


namespace translate_right_one_unit_l3_3669

theorem translate_right_one_unit (x y : ℤ) (hx : x = 4) (hy : y = -3) : (x + 1, y) = (5, -3) :=
by
  -- The proof would go here
  sorry

end translate_right_one_unit_l3_3669


namespace option_C_correct_l3_3644

theorem option_C_correct {a : ℝ} : a^2 * a^3 = a^5 := by
  -- Proof to be filled
  sorry

end option_C_correct_l3_3644


namespace sin_330_deg_l3_3849

theorem sin_330_deg : Real.sin (330 * Real.pi / 180) = -1 / 2 := by
  sorry

end sin_330_deg_l3_3849


namespace brownies_pieces_count_l3_3159

theorem brownies_pieces_count:
  let pan_width := 24
  let pan_length := 15
  let piece_width := 3
  let piece_length := 2
  pan_width * pan_length / (piece_width * piece_length) = 60 := 
by
  sorry

end brownies_pieces_count_l3_3159


namespace tap_fills_tank_without_leakage_in_12_hours_l3_3131

theorem tap_fills_tank_without_leakage_in_12_hours 
  (R_t R_l : ℝ)
  (h1 : (R_t - R_l) * 18 = 1)
  (h2 : R_l * 36 = 1) :
  1 / R_t = 12 := 
by
  sorry

end tap_fills_tank_without_leakage_in_12_hours_l3_3131


namespace possible_shapes_l3_3035

def is_valid_shapes (T S C : ℕ) : Prop :=
  T + S + C = 24 ∧ T = 7 * S

theorem possible_shapes :
  ∃ (T S C : ℕ), is_valid_shapes T S C ∧ 
    (T = 0 ∧ S = 0 ∧ C = 24) ∨
    (T = 7 ∧ S = 1 ∧ C = 16) ∨
    (T = 14 ∧ S = 2 ∧ C = 8) ∨
    (T = 21 ∧ S = 3 ∧ C = 0) :=
by
  sorry

end possible_shapes_l3_3035


namespace B_works_alone_in_24_days_l3_3816

noncomputable def B_completion_days (A B : ℝ) (h1 : A = B) (h2 : (A + B) / 12 = 1) : ℝ :=
24

theorem B_works_alone_in_24_days (A B : ℝ) (h1 : A = B) (h2 : (A + B) / 12 = 1) : 
  B_completion_days A B h1 h2 = 24 :=
sorry

end B_works_alone_in_24_days_l3_3816


namespace square_101_l3_3388

theorem square_101:
  (101 : ℕ)^2 = 10201 :=
by
  sorry

end square_101_l3_3388


namespace polynomial_inequality_l3_3765

theorem polynomial_inequality (x : ℝ) : -6 * x^2 + 2 * x - 8 < 0 :=
sorry

end polynomial_inequality_l3_3765


namespace area_proportions_and_point_on_line_l3_3316

theorem area_proportions_and_point_on_line (T : ℝ × ℝ) :
  (∃ r s : ℝ, T = (r, s) ∧ s = -(5 / 3) * r + 10 ∧ 1 / 2 * 6 * s = 7.5) 
  ↔ T.1 + T.2 = 7 :=
by { sorry }

end area_proportions_and_point_on_line_l3_3316


namespace vending_machine_users_l3_3186

theorem vending_machine_users (p_fail p_double p_single : ℚ) (total_snacks : ℕ) (P : ℕ) :
  p_fail = 1 / 6 ∧ p_double = 1 / 10 ∧ p_single = 1 - 1 / 6 - 1 / 10 ∧
  total_snacks = 28 →
  P = 30 :=
by
  intros h
  sorry

end vending_machine_users_l3_3186


namespace cindy_envelopes_left_l3_3380

def total_envelopes : ℕ := 37
def envelopes_per_friend : ℕ := 3
def number_of_friends : ℕ := 5

theorem cindy_envelopes_left : total_envelopes - (envelopes_per_friend * number_of_friends) = 22 :=
by
  sorry

end cindy_envelopes_left_l3_3380


namespace mixed_groups_count_l3_3020

-- Defining the conditions
def total_children : ℕ := 300
def groups_count : ℕ := 100
def group_size : ℕ := 3
def photographs_per_group : ℕ := group_size
def total_photographs : ℕ := groups_count * photographs_per_group
def boy_boy_photos : ℕ := 100
def girl_girl_photos : ℕ := 56
def mixed_photos : ℕ := total_photographs - boy_boy_photos - girl_girl_photos
def mixed_groups : ℕ := mixed_photos / 2

theorem mixed_groups_count : mixed_groups = 72 := by
  -- skipping the proof
  sorry

end mixed_groups_count_l3_3020


namespace proportionality_cube_and_fourth_root_l3_3557

variables (x y z : ℝ) (k j m n : ℝ)

theorem proportionality_cube_and_fourth_root (h1 : x = k * y^3) (h2 : y = j * z^(1/4)) : 
  ∃ m : ℝ, ∃ n : ℝ, x = m * z^n ∧ n = 3/4 :=
by
  sorry

end proportionality_cube_and_fourth_root_l3_3557


namespace sin_330_eq_neg_half_l3_3829

theorem sin_330_eq_neg_half : sin (330 * (π / 180)) = -1 / 2 := 
by
  -- condition: \(330^\circ = 360^\circ - 30^\circ\)
  -- condition: \(\sin 30^\circ = \frac{1}{2}\)
  -- condition: sine is negative in the fourth quadrant
  sorry

end sin_330_eq_neg_half_l3_3829


namespace candy_pack_cost_l3_3525

theorem candy_pack_cost (c : ℝ) (h1 : 20 + 78 = 98) (h2 : 2 * c = 98) : c = 49 :=
by {
  sorry
}

end candy_pack_cost_l3_3525


namespace sin_330_l3_3905

theorem sin_330 : Real.sin (330 * Real.pi / 180) = -1 / 2 := 
by
  -- Outline the proof here without providing it
  -- sorry to delay the proof
  sorry

end sin_330_l3_3905


namespace volume_box_values_l3_3648

theorem volume_box_values :
  let V := (x + 3) * (x - 3) * (x^2 - 10*x + 25)
  ∃ (x_values : Finset ℕ),
    ∀ x ∈ x_values, V < 1000 ∧ x > 0 ∧ x_values.card = 3 :=
by
  sorry

end volume_box_values_l3_3648


namespace sets_equal_l3_3551

def E : Set ℝ := { x | ∃ n : ℤ, x = Real.cos (n * Real.pi / 3) }
def F : Set ℝ := { x | ∃ m : ℤ, x = Real.sin ((2 * m - 3) * Real.pi / 6) }

theorem sets_equal : E = F :=
  sorry

end sets_equal_l3_3551


namespace solve_system_of_equations_l3_3766

variables {a1 a2 a3 a4 : ℝ}

theorem solve_system_of_equations (h_distinct: a1 ≠ a2 ∧ a1 ≠ a3 ∧ a1 ≠ a4 ∧ a2 ≠ a3 ∧ a2 ≠ a4 ∧ a3 ≠ a4) :
  ∃ (x1 x2 x3 x4 : ℝ),
    (|a1 - a2| * x2 + |a1 - a3| * x3 + |a1 - a4| * x4 = 1) ∧
    (|a2 - a1| * x1 + |a2 - a3| * x3 + |a2 - a4| * x4 = 1) ∧
    (|a3 - a1| * x1 + |a3 - a2| * x2 + |a3 - a4| * x4 = 1) ∧
    (|a4 - a1| * x1 + |a4 - a2| * x2 + |a4 - a3| * x3 = 1) ∧
    (x1 = 1 / (a1 - a4)) ∧ (x2 = 0) ∧ (x3 = 0) ∧ (x4 = 1 / (a1 - a4)) :=
sorry

end solve_system_of_equations_l3_3766


namespace angle_measure_l3_3637

theorem angle_measure (x : ℝ) : 
  (180 - x = 7 * (90 - x)) → x = 75 :=
by
  sorry

end angle_measure_l3_3637


namespace find_b10_l3_3743

def seq (b : ℕ → ℕ) :=
  (b 1 = 2)
  ∧ (∀ m n, b (m + n) = b m + b n + 2 * m * n)

theorem find_b10 (b : ℕ → ℕ) (h : seq b) : b 10 = 110 :=
by 
  -- Proof omitted, as requested.
  sorry

end find_b10_l3_3743


namespace sin_330_correct_l3_3971

noncomputable def sin_330 : ℝ := sorry

theorem sin_330_correct : sin_330 = -1 / 2 :=
  sorry

end sin_330_correct_l3_3971


namespace necessary_but_not_sufficient_l3_3425

def angle_of_inclination (α : ℝ) : Prop :=
  α > Real.pi / 4

def slope_of_line (k : ℝ) : Prop :=
  k > 1

theorem necessary_but_not_sufficient (α k : ℝ) :
  angle_of_inclination α → (slope_of_line k → (k = Real.tan α)) → (angle_of_inclination α → slope_of_line k) ∧ ¬(slope_of_line k → angle_of_inclination α) :=
by
  sorry

end necessary_but_not_sufficient_l3_3425


namespace find_number_l3_3060

theorem find_number (x n : ℕ) (h1 : 3 * x + n = 48) (h2 : x = 4) : n = 36 :=
by
  sorry

end find_number_l3_3060


namespace smallest_positive_integer_divisible_conditions_l3_3642

theorem smallest_positive_integer_divisible_conditions :
  ∃ (M : ℕ), M % 4 = 3 ∧ M % 5 = 4 ∧ M % 6 = 5 ∧ M % 7 = 6 ∧ M = 419 :=
sorry

end smallest_positive_integer_divisible_conditions_l3_3642


namespace quadratic_function_two_distinct_roots_l3_3250

-- Define the discriminant of the quadratic function
def discriminant (a b c : ℝ) : ℝ := b^2 - 4 * a * c

-- Define the function that checks the conditions for the quadratic to have two distinct real roots
theorem quadratic_function_two_distinct_roots (a : ℝ) : 
  (0 < a ∧ a < 2) → (discriminant a (-4) 2 > 0) :=
by
  sorry

end quadratic_function_two_distinct_roots_l3_3250


namespace period_tan_half_l3_3196

noncomputable def period_of_tan_half : Real :=
  2 * Real.pi

theorem period_tan_half (f : Real → Real) (h : ∀ x, f x = Real.tan (x / 2)) :
  ∀ x, f (x + period_of_tan_half) = f x := 
by 
  sorry

end period_tan_half_l3_3196


namespace find_a_l3_3128

theorem find_a 
  (a b c : ℚ) 
  (h1 : a + b = c) 
  (h2 : b + c + 2 * b = 11) 
  (h3 : c = 7) :
  a = 17 / 3 :=
by
  sorry

end find_a_l3_3128


namespace Z_divisible_by_10001_l3_3155

def is_eight_digit_integer (Z : Nat) : Prop :=
  (10^7 ≤ Z) ∧ (Z < 10^8)

def first_four_equal_last_four (Z : Nat) : Prop :=
  ∃ (a b c d : Nat), a ≠ 0 ∧ (Z = 1001 * (1000 * a + 100 * b + 10 * c + d))

theorem Z_divisible_by_10001 (Z : Nat) (h1 : is_eight_digit_integer Z) (h2 : first_four_equal_last_four Z) : 
  10001 ∣ Z :=
sorry

end Z_divisible_by_10001_l3_3155


namespace problem1_l3_3355

theorem problem1
  (a b c : ℝ)
  (h1: a > 0)
  (h2: b > 0)
  (h3: c > 0) :
  a + b + c ≥ Real.sqrt (a * b) + Real.sqrt (b * c) + Real.sqrt (c * a) :=
sorry

end problem1_l3_3355


namespace medium_supermarkets_in_sample_l3_3569

-- Define the conditions
def large_supermarkets : ℕ := 200
def medium_supermarkets : ℕ := 400
def small_supermarkets : ℕ := 1400
def total_supermarkets : ℕ := large_supermarkets + medium_supermarkets + small_supermarkets
def sample_size : ℕ := 100
def proportion_medium := (medium_supermarkets : ℚ) / (total_supermarkets : ℚ)

-- The main theorem to prove
theorem medium_supermarkets_in_sample : sample_size * proportion_medium = 20 := by
  sorry

end medium_supermarkets_in_sample_l3_3569


namespace scientific_notation_of_100000000_l3_3619

theorem scientific_notation_of_100000000 :
  100000000 = 1 * 10^8 :=
sorry

end scientific_notation_of_100000000_l3_3619


namespace minimize_area_eq_l3_3809

theorem minimize_area_eq {l : ℝ → ℝ → Prop}
  (P : ℝ × ℝ) (A B : ℝ × ℝ)
  (condition1 : l P.1 P.2)
  (condition2 : A.1 > 0 ∧ A.2 = 0)
  (condition3 : B.1 = 0 ∧ B.2 > 0)
  (line_eq : ∀ x y : ℝ, l x y ↔ (2 * x + y = 4)) :
  ∀ (a b : ℝ), a = 2 → b = 4 → 2 * P.1 + P.2 = 4 :=
by sorry

end minimize_area_eq_l3_3809


namespace rightmost_four_digits_of_5_pow_2023_l3_3334

theorem rightmost_four_digits_of_5_pow_2023 :
  5 ^ 2023 % 5000 = 3125 :=
  sorry

end rightmost_four_digits_of_5_pow_2023_l3_3334


namespace factorial_divisibility_l3_3510

theorem factorial_divisibility 
  {n : ℕ} 
  (hn : bit0 (n.bits.count 1) == 1995) : 
  (2^(n-1995)) ∣ n! := 
sorry

end factorial_divisibility_l3_3510


namespace sufficient_not_necessary_condition_parallel_lines_l3_3802

theorem sufficient_not_necessary_condition_parallel_lines :
  ∀ (a : ℝ), (a = 1/2 → (∀ x y : ℝ, x + 2*a*y = 1 ↔ (x - x + 1) ≠ 0) 
            ∧ ((∃ a', a' ≠ 1/2 ∧ (∀ x y : ℝ, x + 2*a'*y = 1 ↔ (x - x + 1) ≠ 0)) → (a ≠ 1/2))) :=
by
  intro a
  sorry

end sufficient_not_necessary_condition_parallel_lines_l3_3802


namespace nearest_integer_to_power_six_l3_3788

noncomputable def nearest_integer (x : ℝ) : ℤ := 
if x - real.floor x < real.ceil x - x then real.floor x else real.ceil x

theorem nearest_integer_to_power_six : 
  let x := 3 + real.sqrt 5 in
  let y := 3 - real.sqrt 5 in
  nearest_integer (x^6) = 2654 :=
by
  sorry

end nearest_integer_to_power_six_l3_3788


namespace sin_330_deg_l3_3874

noncomputable theory

open Real

theorem sin_330_deg :
  sin (330 * (π / 180)) = -1 / 2 :=
by sorry

end sin_330_deg_l3_3874


namespace k_range_l3_3124

noncomputable def f (x : ℝ) : ℝ :=
  if 0 < x ∧ x < 1 then -x^3 + 2*x^2 - x
  else if 1 ≤ x then Real.log x
  else 0 -- Technically, we don't care outside (0, +∞), so this else case doesn't matter.

theorem k_range (k : ℝ) :
  (∀ t : ℝ, 0 < t → f t < k * t) ↔ k ∈ (Set.Ioi (1 / Real.exp 1)) :=
by
  sorry

end k_range_l3_3124


namespace sin_330_l3_3933

theorem sin_330 : Real.sin (330 * Real.pi / 180) = -1 / 2 := 
by
  sorry

end sin_330_l3_3933


namespace smallest_positive_n_l3_3796

theorem smallest_positive_n (n : ℕ) (h : 77 * n ≡ 308 [MOD 385]) : n = 4 :=
sorry

end smallest_positive_n_l3_3796


namespace sin_330_eq_neg_half_l3_3828

theorem sin_330_eq_neg_half : sin (330 * (π / 180)) = -1 / 2 := 
by
  -- condition: \(330^\circ = 360^\circ - 30^\circ\)
  -- condition: \(\sin 30^\circ = \frac{1}{2}\)
  -- condition: sine is negative in the fourth quadrant
  sorry

end sin_330_eq_neg_half_l3_3828


namespace common_ratio_l3_3154

-- Definitions for the geometric sequence
variables {a_n : ℕ → ℝ} {S_n q : ℝ}

-- Conditions provided in the problem
def condition1 (a_n : ℕ → ℝ) (S_n : ℕ → ℝ) : Prop :=
  S_n 3 = a_n 1 + a_n 2 + a_n 3

def condition2 (a_n : ℕ → ℝ) (S_n : ℝ) : Prop :=
  3 * (a_n 1 + a_n 2 + a_n 3) = a_n 4 - 2

def condition3 (a_n : ℕ → ℝ) (S_n : ℝ) : Prop :=
  3 * (a_n 1 + a_n 2) = a_n 3 - 2

-- The theorem we want to prove
theorem common_ratio (a_n : ℕ → ℝ) (q : ℝ) :
  condition2 a_n S_n ∧ condition3 a_n S_n → q = 4 :=
by
  sorry

end common_ratio_l3_3154


namespace question_1_question_2_question_3_l3_3646

def deck_size : Nat := 32

theorem question_1 :
  let hands_when_order_matters := deck_size * (deck_size - 1)
  hands_when_order_matters = 992 :=
by
  let hands_when_order_matters := deck_size * (deck_size - 1)
  sorry

theorem question_2 :
  let hands_when_order_does_not_matter := (deck_size * (deck_size - 1)) / 2
  hands_when_order_does_not_matter = 496 :=
by
  let hands_when_order_does_not_matter := (deck_size * (deck_size - 1)) / 2
  sorry

theorem question_3 :
  let hands_3_cards_order_does_not_matter := (deck_size * (deck_size - 1) * (deck_size - 2)) / 6
  hands_3_cards_order_does_not_matter = 4960 :=
by
  let hands_3_cards_order_does_not_matter := (deck_size * (deck_size - 1) * (deck_size - 2)) / 6
  sorry

end question_1_question_2_question_3_l3_3646


namespace domain_of_function_l3_3244

theorem domain_of_function :
  {x : ℝ | x^3 + 5*x^2 + 6*x ≠ 0} =
  {x : ℝ | x < -3} ∪ {x : ℝ | -3 < x ∧ x < -2} ∪ {x : ℝ | -2 < x ∧ x < 0} ∪ {x : ℝ | 0 < x} :=
by
  sorry

end domain_of_function_l3_3244


namespace emails_received_in_afternoon_l3_3148

theorem emails_received_in_afternoon (A : ℕ) 
  (h1 : 4 + (A - 3) = 9) : 
  A = 8 :=
by
  sorry

end emails_received_in_afternoon_l3_3148


namespace max_value_expr_l3_3182

theorem max_value_expr (a b c d : ℝ) (ha : -12.5 ≤ a ∧ a ≤ 12.5) (hb : -12.5 ≤ b ∧ b ≤ 12.5) (hc : -12.5 ≤ c ∧ c ≤ 12.5) (hd : -12.5 ≤ d ∧ d ≤ 12.5) :
  a + 2 * b + c + 2 * d - a * b - b * c - c * d - d * a ≤ 650 :=
sorry

end max_value_expr_l3_3182


namespace sin_330_is_minus_sqrt3_over_2_l3_3988

-- Define the angle theta as 330 degrees
def theta := 330 * Real.toRad

-- State that point corresponding to 330 degrees lies in the fourth quadrant
def point_in_fourth_quadrant (θ : ℝ) := 330 * Real.toRad = θ ∧ θ > 270 * Real.toRad ∧ θ < 360 * Real.toRad

-- Define a proof problem that states computing sin of 330° equals -√3/2
theorem sin_330_is_minus_sqrt3_over_2 : point_in_fourth_quadrant theta → Real.sin theta = -Real.sqrt 3 / 2 := by
  sorry

end sin_330_is_minus_sqrt3_over_2_l3_3988


namespace sufficient_not_necessary_condition_l3_3309

-- Define the condition on a
def condition (a : ℝ) : Prop := a > 0

-- Define the quadratic inequality
def quadratic_inequality (a : ℝ) : Prop := a^2 + a ≥ 0

-- The proof statement that "a > 0" is a sufficient but not necessary condition for "a^2 + a ≥ 0"
theorem sufficient_not_necessary_condition (a : ℝ) : condition a → quadratic_inequality a :=
by
    intro ha
    -- [The remaining part of the proof is skipped.]
    sorry

end sufficient_not_necessary_condition_l3_3309


namespace width_of_rectangular_prism_l3_3269

theorem width_of_rectangular_prism (l h d : ℕ) (w : ℤ) 
  (hl : l = 3) (hh : h = 12) (hd : d = 13) 
  (diag_eq : d = Int.sqrt (l^2 + w^2 + h^2)) : 
  w = 4 := by
  sorry

end width_of_rectangular_prism_l3_3269


namespace greatest_possible_x_l3_3338

theorem greatest_possible_x (x : ℕ) (h : x^4 / x^2 < 18) : x ≤ 4 :=
sorry

end greatest_possible_x_l3_3338


namespace fraction_married_men_l3_3200

-- Define the problem conditions
def num_faculty : ℕ := 100
def women_perc : ℕ := 60
def married_perc : ℕ := 60
def single_men_perc : ℚ := 3/4

-- We need to calculate the fraction of men who are married.
theorem fraction_married_men :
  (60 : ℚ) / 100 = women_perc / num_faculty →
  (60 : ℚ) / 100 = married_perc / num_faculty →
  (3/4 : ℚ) = single_men_perc →
  ∃ (fraction : ℚ), fraction = 1/4 :=
by
  intro h1 h2 h3
  sorry

end fraction_married_men_l3_3200


namespace sin_330_eq_neg_one_half_l3_3958

theorem sin_330_eq_neg_one_half : Real.sin (330 * Real.pi / 180) = -1 / 2 := by
  sorry

end sin_330_eq_neg_one_half_l3_3958


namespace quadratic_roots_range_l3_3138

theorem quadratic_roots_range (k : ℝ) : (x^2 - 6*x + k = 0) → k < 9 := 
by
  sorry

end quadratic_roots_range_l3_3138


namespace area_of_quadrilateral_l3_3711
noncomputable def c := sqrt (16 - 4) -- √12 = 2√3

theorem area_of_quadrilateral (a b : ℝ) (P Q F1 F2 : ℝ×ℝ) :
  let e : set (ℝ × ℝ) := {p | (p.1^2 / a^2) + (p.2^2 / b^2) = 1} in
  let ellipse_symmetric := (P ∈ e) ∧ (Q ∈ e) ∧ (P.1 = -Q.1) ∧ (P.2 = -Q.2) ∧ dist (P, Q) = 2 * c in
  let F1F2 := 2 * c in
  let mn := 8 in
  (dist (P, F1) + dist (P, F2) = 2 * a) →
  (dist (P, F1)^2 + dist (P, F2)^2 = F1F2^2 / 4 * (a^2 - b^2)) →
  (mn = 8) →
  area_of_quadrilateral P F1 Q F2 = 8 :=
by
  sorry

end area_of_quadrilateral_l3_3711


namespace eval_power_l3_3088

theorem eval_power (h : 81 = 3^4) : 81^(5/4) = 243 := by
  sorry

end eval_power_l3_3088


namespace simplify_radicals_l3_3600

theorem simplify_radicals :
  (Real.sqrt (10 + 6 * Real.sqrt 3) + Real.sqrt (10 - 6 * Real.sqrt 3)) = 2 * Real.sqrt 6 :=
by 
  sorry

end simplify_radicals_l3_3600


namespace inequality_triangle_area_l3_3421

-- Define the triangles and their properties
variables {α β γ : Real} -- Internal angles of triangle ABC
variables {r : Real} -- Circumradius of triangle ABC
variables {P Q : Real} -- Areas of triangles ABC and A'B'C' respectively

-- Define the bisectors and intersect points
-- Note: For the purpose of this proof, we're not explicitly defining the geometry
-- of the inner bisectors and intersect points but working from the given conditions.

theorem inequality_triangle_area
  (h1 : P = r^2 * (Real.sin (2 * α) + Real.sin (2 * β) + Real.sin (2 * γ)) / 2)
  (h2 : Q = r^2 * (Real.sin (β + γ) + Real.sin (γ + α) + Real.sin (α + β)) / 2) :
  16 * Q^3 ≥ 27 * r^4 * P :=
sorry

end inequality_triangle_area_l3_3421


namespace num_ways_distribute_pens_l3_3109

-- Conditions:
def num_friends : ℕ := 4
def num_pens : ℕ := 10
def at_least_one_pen_each (dist : fin num_friends → ℕ) : Prop := 
  ∀ i : fin num_friends, dist i ≥ 1
def at_most_five_pens_each (dist : fin num_friends → ℕ) : Prop :=
  ∀ i : fin num_friends, dist i ≤ 5
def total_pens_distributed (dist : fin num_friends → ℕ) : Prop :=
  finset.univ.sum dist = num_pens

-- Proof Problem:
theorem num_ways_distribute_pens :
  {dist : fin num_friends → ℕ // 
    at_least_one_pen_each dist ∧ at_most_five_pens_each dist ∧ total_pens_distributed dist}.card = 50 := 
sorry

end num_ways_distribute_pens_l3_3109


namespace proof_problem_l3_3431

theorem proof_problem (a b : ℝ) (H1 : ∀ x : ℝ, (ax^2 - 3*x + 6 > 4) ↔ (x < 1 ∨ x > b)) :
  a = 1 ∧ b = 2 ∧
  (∀ c : ℝ, (ax^2 - (a*c + b)*x + b*c < 0) ↔ 
   (if c > 2 then 2 < x ∧ x < c
    else if c < 2 then c < x ∧ x < 2
    else false)) :=
by
  sorry

end proof_problem_l3_3431


namespace Teresa_current_age_l3_3771

-- Definitions of the conditions
def Morio_current_age := 71
def Morio_age_when_Michiko_born := 38
def Teresa_age_when_Michiko_born := 26

-- Definition of Michiko's current age
def Michiko_current_age := Morio_current_age - Morio_age_when_Michiko_born

-- The Theorem statement
theorem Teresa_current_age : Teresa_age_when_Michiko_born + Michiko_current_age = 59 :=
by
  -- Skip the proof
  sorry

end Teresa_current_age_l3_3771


namespace mice_needed_l3_3500

-- Definitions for relative strength in terms of M (Mouse strength)
def C (M : ℕ) : ℕ := 6 * M
def J (M : ℕ) : ℕ := 5 * C M
def G (M : ℕ) : ℕ := 4 * J M
def B (M : ℕ) : ℕ := 3 * G M
def D (M : ℕ) : ℕ := 2 * B M

-- Condition: all together can pull up the Turnip with strength 1237M
def total_strength_with_mouse (M : ℕ) : ℕ :=
  D M + B M + G M + J M + C M + M

-- Condition: without the Mouse, they cannot pull up the Turnip
def total_strength_without_mouse (M : ℕ) : ℕ :=
  D M + B M + G M + J M + C M

theorem mice_needed (M : ℕ) (h : total_strength_with_mouse M = 1237 * M) (h2 : total_strength_without_mouse M < 1237 * M) :
  1237 = 1237 :=
by
  -- using sorry to indicate proof is not provided
  sorry

end mice_needed_l3_3500


namespace sin_330_degree_l3_3954

theorem sin_330_degree : Real.sin (330 * Real.pi / 180) = -1 / 2 :=
by
  sorry

end sin_330_degree_l3_3954


namespace number_of_groups_of_three_marbles_l3_3630

-- Define the problem conditions 
def red_marble : ℕ := 1
def blue_marble : ℕ := 1
def black_marble : ℕ := 1
def white_marbles : ℕ := 4

-- The proof problem statement
theorem number_of_groups_of_three_marbles (red_marble blue_marble black_marble white_marbles : ℕ) :
  (white_marbles.choose 3) + (3.choose 2 * white_marbles.choose 1) = 16 :=
by
  sorry

end number_of_groups_of_three_marbles_l3_3630


namespace avg_int_values_between_l3_3483

theorem avg_int_values_between (N : ℤ) :
  (5 : ℚ) / 12 < N / 48 ∧ N / 48 < 1 / 3 →
  (N = 17 ∨ N = 18 ∨ N = 19) ∧
  (N = 17 ∨ N = 18 ∨ N = 19 →
  (17 + 18 + 19) / 3 = 18) :=
by
  sorry

end avg_int_values_between_l3_3483


namespace solution_correct_l3_3304

noncomputable def solve_system (a b c : ℝ) : ℝ × ℝ × ℝ :=
  let x := (3 * c - a - b) / 4
  let y := (3 * b - a - c) / 4
  let z := (3 * a - b - c) / 4
  (x, y, z)

theorem solution_correct (a b c : ℝ) (x y z : ℝ) :
  (x + y + 2 * z = a) →
  (x + 2 * y + z = b) →
  (2 * x + y + z = c) →
  (x, y, z) = solve_system a b c :=
by sorry

end solution_correct_l3_3304


namespace product_identity_l3_3524

theorem product_identity (x y : ℝ) : (3 * x^2 - 4 * y^3) * (9 * x^4 + 12 * x^2 * y^3 + 16 * y^6) = 27 * x^6 - 64 * y^9 :=
by
  sorry

end product_identity_l3_3524


namespace spicy_hot_noodles_plates_l3_3507

theorem spicy_hot_noodles_plates (total_plates lobster_rolls seafood_noodles spicy_hot_noodles : ℕ) :
  total_plates = 55 →
  lobster_rolls = 25 →
  seafood_noodles = 16 →
  spicy_hot_noodles = total_plates - (lobster_rolls + seafood_noodles) →
  spicy_hot_noodles = 14 := by
  intros h_total h_lobster h_seafood h_eq
  rw [h_total, h_lobster, h_seafood] at h_eq
  exact h_eq

end spicy_hot_noodles_plates_l3_3507


namespace mixed_groups_count_l3_3017

namespace YoungPhotographerClub

theorem mixed_groups_count
  (total_children : ℕ)
  (total_groups : ℕ)
  (children_per_group : ℕ)
  (photographs_per_group : ℕ)
  (boy_boy_photographs : ℕ)
  (girl_girl_photographs : ℕ)
  (total_photographs : ℕ)
  (mixed_photographs : ℕ)
  (mixed_groups : ℕ)
  (H1 : total_children = 300)
  (H2 : total_groups = 100)
  (H3 : children_per_group = 3)
  (H4 : photographs_per_group = 3)
  (H5 : boy_boy_photographs = 100)
  (H6 : girl_girl_photographs = 56)
  (H7 : total_photographs = total_groups * photographs_per_group)
  (H8 : mixed_photographs = total_photographs - boy_boy_photographs - girl_girl_photographs)
  (H9 : mixed_groups = mixed_photographs / 2) : mixed_groups = 72 := 
by
  rw [H1, H2, H3, H4, H5, H6, H7, H8, H9]
  sorry

end YoungPhotographerClub

end mixed_groups_count_l3_3017


namespace optionD_is_equation_l3_3487

-- Definitions for options
def optionA (x : ℕ) := 2 * x - 3
def optionB := 2 + 4 = 6
def optionC (x : ℕ) := x > 2
def optionD (x : ℕ) := 2 * x - 1 = 3

-- Goal: prove that option D is an equation.
theorem optionD_is_equation (x : ℕ) : (optionD x) = True :=
sorry

end optionD_is_equation_l3_3487


namespace mixed_groups_count_l3_3031

-- Define the facts about the groups and photographs
def numberOfChildren : Nat := 300
def numberOfGroups : Nat := 100
def childrenPerGroup : Nat := 3
def b_b_photos : Nat := 100
def g_g_photos : Nat := 56

-- Define the function to calculate mixed groups
def mixedGroups (totalPhotos b_b_photos g_g_photos : Nat) : Nat := 
  (totalPhotos - b_b_photos - g_g_photos) / 2

-- State the theorem
theorem mixed_groups_count : 
  mixedGroups (numberOfGroups * childrenPerGroup) b_b_photos g_g_photos = 72 := by
  rfl

end mixed_groups_count_l3_3031


namespace find_n_l3_3665

theorem find_n 
  (N : ℕ) 
  (hn : ¬ (N = 0))
  (parts_inv_prop : ∀ k, 1 ≤ k → k ≤ n → N / (k * (k + 1)) = x / (n * (n + 1))) 
  (smallest_part : (N : ℝ) / 400 = N / (n * (n + 1))) : 
  n = 20 :=
sorry

end find_n_l3_3665


namespace difference_approx_l3_3311

-- Let L be the larger number and S be the smaller number
variables (L S : ℝ)

-- Conditions given:
-- 1. L is approximately 1542.857
def approx_L : Prop := abs (L - 1542.857) < 1

-- 2. When L is divided by S, quotient is 8 and remainder is 15
def division_condition : Prop := L = 8 * S + 15

-- The theorem stating the difference L - S is approximately 1351.874
theorem difference_approx (hL : approx_L L) (hdiv : division_condition L S) :
  abs ((L - S) - 1351.874) < 1 :=
sorry

#check difference_approx

end difference_approx_l3_3311


namespace cindy_envelopes_left_l3_3379

def total_envelopes : ℕ := 37
def envelopes_per_friend : ℕ := 3
def number_of_friends : ℕ := 5

theorem cindy_envelopes_left : total_envelopes - (envelopes_per_friend * number_of_friends) = 22 :=
by
  sorry

end cindy_envelopes_left_l3_3379


namespace total_cans_l3_3478

theorem total_cans (total_oil : ℕ) (oil_in_8_liter_cans : ℕ) (number_of_8_liter_cans : ℕ) (remaining_oil : ℕ) 
(oil_per_15_liter_can : ℕ) (number_of_15_liter_cans : ℕ) :
  total_oil = 290 ∧ oil_in_8_liter_cans = 8 ∧ number_of_8_liter_cans = 10 ∧ oil_per_15_liter_can = 15 ∧
  remaining_oil = total_oil - (number_of_8_liter_cans * oil_in_8_liter_cans) ∧
  number_of_15_liter_cans = remaining_oil / oil_per_15_liter_can →
  (number_of_8_liter_cans + number_of_15_liter_cans) = 24 := sorry

end total_cans_l3_3478


namespace sin_330_degree_l3_3955

theorem sin_330_degree : Real.sin (330 * Real.pi / 180) = -1 / 2 :=
by
  sorry

end sin_330_degree_l3_3955


namespace isosceles_triangle_perimeter_l3_3572

theorem isosceles_triangle_perimeter 
  (a b c : ℝ)  (h_iso : a = b ∨ b = c ∨ c = a)
  (h_len1 : a = 4 ∨ b = 4 ∨ c = 4)
  (h_len2 : a = 9 ∨ b = 9 ∨ c = 9)
  (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b) :
  a + b + c = 22 :=
sorry

end isosceles_triangle_perimeter_l3_3572


namespace exponentiation_81_5_4_eq_243_l3_3092

theorem exponentiation_81_5_4_eq_243 : 81^(5/4) = 243 := by
  sorry

end exponentiation_81_5_4_eq_243_l3_3092


namespace sin_330_l3_3899

theorem sin_330 : Real.sin (330 * Real.pi / 180) = -1 / 2 := 
by
  -- Outline the proof here without providing it
  -- sorry to delay the proof
  sorry

end sin_330_l3_3899


namespace distance_from_circle_center_to_line_l3_3428

-- Define the center of the circle
def circle_center : ℝ × ℝ := (1, -2)

-- Define the equation of the line
def line_eq (x y : ℝ) : ℝ := 2 * x + y - 5

-- Define the distance function from a point to a line
noncomputable def distance_to_line (p : ℝ × ℝ) (a b c : ℝ) : ℝ :=
  |a * p.1 + b * p.2 + c| / Real.sqrt (a ^ 2 + b ^ 2)

-- Define the actual proof problem
theorem distance_from_circle_center_to_line : 
  distance_to_line circle_center 2 1 (-5) = Real.sqrt 5 :=
by
  sorry

end distance_from_circle_center_to_line_l3_3428


namespace fractions_product_simplified_l3_3682

theorem fractions_product_simplified : (2/3 : ℚ) * (4/7) * (9/11) = 24/77 := by
  sorry

end fractions_product_simplified_l3_3682


namespace square_area_from_diagonal_l3_3516

theorem square_area_from_diagonal :
  ∀ (d : ℝ), d = 10 * Real.sqrt 2 → (d / Real.sqrt 2) ^ 2 = 100 :=
by
  intros d hd
  sorry -- Skipping the proof

end square_area_from_diagonal_l3_3516


namespace total_handshakes_l3_3077

theorem total_handshakes (players_team1 players_team2 referees : ℕ) 
  (h1 : players_team1 = 11) (h2 : players_team2 = 11) (h3 : referees = 3) : 
  players_team1 * players_team2 + (players_team1 + players_team2) * referees = 187 := 
by
  sorry

end total_handshakes_l3_3077


namespace sec_120_eq_neg_2_l3_3396

noncomputable def sec (θ : ℝ) : ℝ := 1 / Real.cos θ

theorem sec_120_eq_neg_2 : sec 120 = -2 := by
  sorry

end sec_120_eq_neg_2_l3_3396


namespace evaluate_fraction_l3_3693

-- Define the custom operations x@y and x#y
def op_at (x y : ℝ) : ℝ := x * y - y^2
def op_hash (x y : ℝ) : ℝ := x + y - x * y^2 + x^2

-- State the proof goal
theorem evaluate_fraction : (op_at 7 3) / (op_hash 7 3) = -3 :=
by
  -- Calculations to prove the theorem
  sorry

end evaluate_fraction_l3_3693


namespace sin_330_eq_neg_half_l3_3893

theorem sin_330_eq_neg_half : Real.sin (330 * Real.pi / 180) = -1 / 2 := by
  -- Definitions and conditions from the problem
  have h1 : 330 = 360 - 30 := rfl
  have h2 : ∀ θ, Real.sin θ = -Real.sin (θ - 2 * Real.pi) := sorry
  have h3 : Real.sin (30 * Real.pi / 180) = 1 / 2 := sorry

  -- Use the given conditions to prove the equality
  calc
    Real.sin (330 * Real.pi / 180)
        = Real.sin (-(30 * Real.pi / 180) + 2 * Real.pi) : by rw [←h1, Real.sin_angle_sub_pi]
    ... = -Real.sin (30 * Real.pi / 180) : by rw [h2]
    ... = -1 / 2 : by rw [h3]

end sin_330_eq_neg_half_l3_3893


namespace gcf_7fact_8fact_l3_3413

-- Definitions based on the conditions
def factorial (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | (n + 1) => (n + 1) * factorial n

noncomputable def greatest_common_divisor (a b : ℕ) : ℕ :=
  Nat.gcd a b

-- Theorem statement
theorem gcf_7fact_8fact : greatest_common_divisor (factorial 7) (factorial 8) = 5040 := by
  sorry

end gcf_7fact_8fact_l3_3413


namespace ninth_term_geometric_sequence_l3_3177

noncomputable def geometric_seq (a r : ℝ) (n : ℕ) : ℝ := a * r ^ (n - 1)

theorem ninth_term_geometric_sequence (a r : ℝ) (h_positive : ∀ n, 0 < geometric_seq a r n)
  (h_fifth_term : geometric_seq a r 5 = 32)
  (h_eleventh_term : geometric_seq a r 11 = 2) :
  geometric_seq a r 9 = 2 :=
by
{
  sorry
}

end ninth_term_geometric_sequence_l3_3177


namespace probability_of_matching_pair_l3_3480

noncomputable def num_socks := 22
noncomputable def red_socks := 12
noncomputable def blue_socks := 10

def ways_to_choose_two (n : ℕ) : ℕ :=
  n * (n - 1) / 2

noncomputable def probability_same_color : ℚ :=
  (ways_to_choose_two red_socks + ways_to_choose_two blue_socks : ℚ) / ways_to_choose_two num_socks

theorem probability_of_matching_pair :
  probability_same_color = 37 / 77 := 
by
  -- proof goes here
  sorry

end probability_of_matching_pair_l3_3480


namespace scientific_notation_example_l3_3477

def scientific_notation (n : ℝ) (a : ℝ) (b : ℤ) : Prop :=
  n = a * 10^b

theorem scientific_notation_example : 
  scientific_notation 0.00519 5.19 (-3) :=
by 
  sorry

end scientific_notation_example_l3_3477


namespace sin_330_degree_l3_3950

theorem sin_330_degree : Real.sin (330 * Real.pi / 180) = -1 / 2 :=
by
  sorry

end sin_330_degree_l3_3950


namespace smallest_initial_number_sum_of_digits_l3_3678

theorem smallest_initial_number_sum_of_digits : ∃ (N : ℕ), 
  (0 ≤ N ∧ N < 1000) ∧ 
  ∃ (k : ℕ), 16 * N + 700 + 50 * k < 1000 ∧ 
  (N = 16) ∧ 
  (Nat.digits 10 N).sum = 7 := 
by
  sorry

end smallest_initial_number_sum_of_digits_l3_3678


namespace sin_330_correct_l3_3968

noncomputable def sin_330 : ℝ := sorry

theorem sin_330_correct : sin_330 = -1 / 2 :=
  sorry

end sin_330_correct_l3_3968


namespace expand_and_simplify_l3_3099

variable (x : ℝ)

theorem expand_and_simplify : (7 * x - 3) * 3 * x^2 = 21 * x^3 - 9 * x^2 := by
  sorry

end expand_and_simplify_l3_3099


namespace cube_edges_after_cuts_l3_3392

theorem cube_edges_after_cuts (V E : ℕ) (hV : V = 8) (hE : E = 12) : 
  12 + 24 = 36 := by
  sorry

end cube_edges_after_cuts_l3_3392


namespace arithmetic_sequence_properties_l3_3584

variable {a : ℕ → ℤ} (S : ℕ → ℤ) (d : ℤ)

-- Definitions and conditions
def is_arithmetic_sequence (a : ℕ → ℤ) (d : ℤ) : Prop :=
  ∀ n, a (n + 1) = a n + d

def sum_of_first_n_terms (a : ℕ → ℤ) (n : ℕ) : ℤ :=
  n * a 1 + (n * (n - 1) / 2) * d

axiom a1_val : a 1 = 30
axiom S12_eq_S19 : sum_of_first_n_terms a 12 = sum_of_first_n_terms a 19

-- Prove that d = -2 and S_n ≤ S_15 for any n
theorem arithmetic_sequence_properties :
  is_arithmetic_sequence a d →
  (∀ n, S n = sum_of_first_n_terms a n) →
  d = -2 ∧ ∀ n, S n ≤ S 15 :=
by
  intros h_arith h_sum
  sorry

end arithmetic_sequence_properties_l3_3584


namespace balls_to_boxes_l3_3554

theorem balls_to_boxes (balls boxes : ℕ) (h1 : balls = 5) (h2 : boxes = 3) :
  ∃ ways : ℕ, ways = 150 := by
  sorry

end balls_to_boxes_l3_3554


namespace medium_size_shoes_initially_stocked_l3_3073

variable {M : ℕ}  -- The number of medium-size shoes initially stocked

noncomputable def initial_pairs_eq (M : ℕ) := 22 + M + 24
noncomputable def shoes_sold (M : ℕ) := initial_pairs_eq M - 13

theorem medium_size_shoes_initially_stocked :
  shoes_sold M = 83 → M = 26 :=
by
  sorry

end medium_size_shoes_initially_stocked_l3_3073


namespace ellipse_foci_x_axis_l3_3775

theorem ellipse_foci_x_axis (k : ℝ) : 
  (0 < k ∧ k < 2) ↔ (∃ a b : ℝ, a > 0 ∧ b > 0 ∧ ∀ x y : ℝ, (x^2 / a^2 + y^2 / b^2 = 1) ∧ a > b) := 
sorry

end ellipse_foci_x_axis_l3_3775


namespace even_digit_perfect_squares_odd_digit_perfect_squares_l3_3397

-- Define the property of being a four-digit number
def is_four_digit (n : ℕ) : Prop :=
  1000 ≤ n ∧ n < 10000

-- Define the property of having even digits
def is_even_digit_number (n : ℕ) : Prop :=
  ∀ digit ∈ (n.digits 10), digit % 2 = 0

-- Define the property of having odd digits
def is_odd_digit_number (n : ℕ) : Prop :=
  ∀ digit ∈ (n.digits 10), digit % 2 = 1

-- Part (a) statement
theorem even_digit_perfect_squares :
  ∀ n : ℕ, is_four_digit n ∧ is_even_digit_number n ∧ ∃ m : ℕ, n = m * m ↔ 
    n = 4624 ∨ n = 6084 ∨ n = 6400 ∨ n = 8464 :=
sorry

-- Part (b) statement
theorem odd_digit_perfect_squares :
  ∀ n : ℕ, is_four_digit n ∧ is_odd_digit_number n ∧ ∃ m : ℕ, n = m * m → false :=
sorry

end even_digit_perfect_squares_odd_digit_perfect_squares_l3_3397


namespace AgOH_moles_formed_l3_3103

noncomputable def number_of_moles_of_AgOH (n_AgNO3 n_NaOH : ℕ) : ℕ :=
  if n_AgNO3 = n_NaOH then n_AgNO3 else 0

theorem AgOH_moles_formed :
  number_of_moles_of_AgOH 3 3 = 3 := by
  sorry

end AgOH_moles_formed_l3_3103


namespace extreme_values_range_of_a_l3_3724

noncomputable def f (x : ℝ) := x^2 * Real.exp x
noncomputable def y (x : ℝ) (a : ℝ) := f x - a * x

theorem extreme_values :
  ∃ x_max x_min,
    (x_max = -2 ∧ f x_max = 4 / Real.exp 2) ∧
    (x_min = 0 ∧ f x_min = 0) := sorry

theorem range_of_a (a : ℝ) :
  (∃ x₁ x₂, x₁ ≠ 0 ∧ x₂ ≠ 0 ∧ y x₁ a = 0 ∧ y x₂ a = 0) ↔
  -1 / Real.exp 1 < a ∧ a < 0 := sorry

end extreme_values_range_of_a_l3_3724


namespace cubic_function_increasing_l3_3720

noncomputable def f (a x : ℝ) := x ^ 3 + a * x ^ 2 + 7 * a * x

theorem cubic_function_increasing (a : ℝ) (h : 0 ≤ a ∧ a ≤ 21) :
    ∀ x y : ℝ, x ≤ y → f a x ≤ f a y :=
sorry

end cubic_function_increasing_l3_3720


namespace ratio_of_areas_l3_3755

-- Definitions of perimeter in Lean terms
def P_A : ℕ := 16
def P_B : ℕ := 32

-- Ratio of the area of region A to region C
theorem ratio_of_areas (s_A s_C : ℕ) (h₀ : 4 * s_A = P_A)
  (h₁ : 4 * s_C = 12) : s_A^2 / s_C^2 = 1 / 9 :=
by 
  sorry

end ratio_of_areas_l3_3755


namespace value_of_f_neg2_l3_3047

def f (x : ℤ) : ℤ := x^2 - 3 * x + 1

theorem value_of_f_neg2 : f (-2) = 11 := by
  sorry

end value_of_f_neg2_l3_3047


namespace functional_equation_solution_l3_3529

noncomputable def f (t : ℝ) (x : ℝ) := (t * (x - t)) / (t + 1)

noncomputable def g (t : ℝ) (x : ℝ) := t * (x - t)

theorem functional_equation_solution (t : ℝ) (ht : t ≠ -1) :
  ∀ x y : ℝ, f t (x + g t y) = x * f t y - y * f t x + g t x :=
by
  intros x y
  let fx := f t
  let gx := g t
  sorry

end functional_equation_solution_l3_3529


namespace sin_330_eq_neg_one_half_l3_3856

theorem sin_330_eq_neg_one_half :
  ∃ θ, θ = 330 ∧ (sin (Real.pi * (θ / 180)) = -1 / 2) :=
by
  use 330
  split
  · rfl
  · sorry

end sin_330_eq_neg_one_half_l3_3856


namespace probability_non_defective_pencils_l3_3496

theorem probability_non_defective_pencils :
  let total_pencils := 8
  let defective_pencils := 2
  let selected_pencils := 3
  let non_defective_pencils := total_pencils - defective_pencils
  let total_combinations := Nat.choose total_pencils selected_pencils
  let non_defective_combinations := Nat.choose non_defective_pencils selected_pencils
  (non_defective_combinations:ℚ) / (total_combinations:ℚ) = 5 / 14 := by
  sorry

end probability_non_defective_pencils_l3_3496


namespace sin_330_eq_neg_sin_30_l3_3982

theorem sin_330_eq_neg_sin_30 :
  sin (330 : ℝ) = - sin (30 : ℝ) := sorry

end sin_330_eq_neg_sin_30_l3_3982


namespace root_in_interval_l3_3134

noncomputable def f (m x : ℝ) := m * 3^x - x + 3

theorem root_in_interval (m : ℝ) (h1 : m < 0) (h2 : ∃ x : ℝ, 0 < x ∧ x < 1 ∧ f m x = 0) : -3 < m ∧ m < -2/3 :=
by
  sorry

end root_in_interval_l3_3134


namespace hanna_has_money_l3_3432

variable (total_roses money_spent : ℕ)
variable (rose_price : ℕ := 2)

def hanna_gives_roses (total_roses : ℕ) : Bool :=
  (1 / 3 * total_roses + 1 / 2 * total_roses) = 125

theorem hanna_has_money (H : hanna_gives_roses total_roses) : money_spent = 300 := sorry

end hanna_has_money_l3_3432


namespace solution_set_inequality_l3_3391

theorem solution_set_inequality (x : ℝ) : (3 * x^2 + 7 * x ≤ 2) ↔ (-2 ≤ x ∧ x ≤ 1 / 3) :=
by
  sorry

end solution_set_inequality_l3_3391


namespace sin_330_is_minus_sqrt3_over_2_l3_3995

-- Define the angle theta as 330 degrees
def theta := 330 * Real.toRad

-- State that point corresponding to 330 degrees lies in the fourth quadrant
def point_in_fourth_quadrant (θ : ℝ) := 330 * Real.toRad = θ ∧ θ > 270 * Real.toRad ∧ θ < 360 * Real.toRad

-- Define a proof problem that states computing sin of 330° equals -√3/2
theorem sin_330_is_minus_sqrt3_over_2 : point_in_fourth_quadrant theta → Real.sin theta = -Real.sqrt 3 / 2 := by
  sorry

end sin_330_is_minus_sqrt3_over_2_l3_3995


namespace sum_first_9_terms_l3_3739

noncomputable def sum_of_first_n_terms (a1 d : Int) (n : Int) : Int :=
  n * (2 * a1 + (n - 1) * d) / 2

theorem sum_first_9_terms (a1 d : ℤ) 
  (h1 : a1 + (a1 + 3 * d) + (a1 + 6 * d) = 39)
  (h2 : (a1 + 2 * d) + (a1 + 5 * d) + (a1 + 8 * d) = 27) :
  sum_of_first_n_terms a1 d 9 = 99 := by
  sorry

end sum_first_9_terms_l3_3739


namespace original_volume_of_ice_cube_l3_3520

theorem original_volume_of_ice_cube
  (V : ℝ)
  (h1 : V * (1/2) * (2/3) * (3/4) * (4/5) = 30)
  : V = 150 :=
sorry

end original_volume_of_ice_cube_l3_3520


namespace cards_from_country_correct_l3_3048

def total_cards : ℝ := 403.0
def cards_from_home : ℝ := 287.0
def cards_from_country : ℝ := total_cards - cards_from_home

theorem cards_from_country_correct : cards_from_country = 116.0 := by
  -- proof to be added
  sorry

end cards_from_country_correct_l3_3048


namespace scout_earnings_weekend_l3_3302

-- Define the conditions
def base_pay_per_hour : ℝ := 10.00
def saturday_hours : ℝ := 6
def saturday_customers : ℝ := 5
def saturday_tip_per_customer : ℝ := 5.00
def sunday_hours : ℝ := 8
def sunday_customers_with_3_tip : ℝ := 5
def sunday_customers_with_7_tip : ℝ := 5
def sunday_tip_3_per_customer : ℝ := 3.00
def sunday_tip_7_per_customer : ℝ := 7.00
def overtime_multiplier : ℝ := 1.5

-- Statement to prove earnings for the weekend is $255.00
theorem scout_earnings_weekend : 
  (base_pay_per_hour * saturday_hours + saturday_customers * saturday_tip_per_customer) +
  (base_pay_per_hour * overtime_multiplier * sunday_hours + 
   sunday_customers_with_3_tip * sunday_tip_3_per_customer +
   sunday_customers_with_7_tip * sunday_tip_7_per_customer) = 255 :=
by
  sorry

end scout_earnings_weekend_l3_3302


namespace count_ordered_triples_lcm_l3_3581

def lcm_of_pair (a b : ℕ) : ℕ := a * b / Nat.gcd a b

theorem count_ordered_triples_lcm :
  (∃ (count : ℕ), count = 70 ∧
   ∀ (a b c : ℕ), (0 < a) → (0 < b) → (0 < c) →
   lcm_of_pair a b = 1000 → lcm_of_pair b c = 2000 → lcm_of_pair c a = 2000 → count = 70) :=
sorry

end count_ordered_triples_lcm_l3_3581


namespace sum_of_n_and_k_l3_3314

open Nat

theorem sum_of_n_and_k (n k : ℕ)
  (h1 : 2 = n - 3 * k)
  (h2 : 8 = 2 * n - 5 * k) :
  n + k = 18 :=
sorry

end sum_of_n_and_k_l3_3314


namespace solve_inequality_system_l3_3169

-- Define the conditions and the correct answer
def system_of_inequalities (x : ℝ) : Prop :=
  (5 * x - 1 > 3 * (x + 1)) ∧ ((1/2) * x - 1 ≤ 7 - (3/2) * x)

def solution_set (x : ℝ) : Prop :=
  2 < x ∧ x ≤ 4

-- State that solving the system of inequalities is equivalent to the solution set
theorem solve_inequality_system (x : ℝ) : system_of_inequalities x ↔ solution_set x :=
  sorry

end solve_inequality_system_l3_3169


namespace sin_330_eq_neg_half_l3_3887

theorem sin_330_eq_neg_half : Real.sin (330 * Real.pi / 180) = -1 / 2 := by
  -- Definitions and conditions from the problem
  have h1 : 330 = 360 - 30 := rfl
  have h2 : ∀ θ, Real.sin θ = -Real.sin (θ - 2 * Real.pi) := sorry
  have h3 : Real.sin (30 * Real.pi / 180) = 1 / 2 := sorry

  -- Use the given conditions to prove the equality
  calc
    Real.sin (330 * Real.pi / 180)
        = Real.sin (-(30 * Real.pi / 180) + 2 * Real.pi) : by rw [←h1, Real.sin_angle_sub_pi]
    ... = -Real.sin (30 * Real.pi / 180) : by rw [h2]
    ... = -1 / 2 : by rw [h3]

end sin_330_eq_neg_half_l3_3887


namespace arithmetic_sequence_problem_l3_3259

variable (d a1 : ℝ)
variable (h1 : a1 ≠ d)
variable (h2 : d ≠ 0)

theorem arithmetic_sequence_problem (S20 M : ℝ)
  (h3 : S20 = 10 * M)
  (x y : ℝ)
  (h4 : M = x * (a1 + 9 * d) + y * d) :
  x = 2 ∧ y = 1 := 
by 
  sorry

end arithmetic_sequence_problem_l3_3259


namespace infinite_geometric_series_first_term_l3_3232

theorem infinite_geometric_series_first_term 
  (r : ℝ) 
  (S : ℝ) 
  (a : ℝ) 
  (h1 : r = -3/7) 
  (h2 : S = 18) 
  (h3 : S = a / (1 - r)) : 
  a = 180 / 7 := by
  -- omitted proof
  sorry

end infinite_geometric_series_first_term_l3_3232


namespace combine_ingredients_l3_3626

theorem combine_ingredients : 
  ∃ (water flour salt : ℕ), 
    water = 10 ∧ flour = 16 ∧ salt = 1 / 2 * flour ∧ 
    (water + flour = 26) ∧ (salt = 8) :=
by
  sorry

end combine_ingredients_l3_3626


namespace james_weekly_earnings_l3_3281

def rate_per_hour : ℕ := 20
def hours_per_day : ℕ := 8
def days_per_week : ℕ := 4

def daily_earnings : ℕ := rate_per_hour * hours_per_day
def weekly_earnings : ℕ := daily_earnings * days_per_week

theorem james_weekly_earnings : weekly_earnings = 640 := sorry

end james_weekly_earnings_l3_3281


namespace find_line_equation_l3_3544

open Real

noncomputable def line_equation (x y : ℝ) (k : ℝ) : ℝ := k * x - y + 4 - 3 * k

noncomputable def distance_to_line (x1 y1 k : ℝ) : ℝ :=
  abs (k * x1 - y1 + 4 - 3 * k) / sqrt (k^2 + 1)

theorem find_line_equation :
  (∃ k : ℝ, (k = 2 ∨ k = -2 / 3) ∧
    (∀ x y, (x, y) = (3, 4) → (2 * x - y - 2 = 0 ∨ 2 * x + 3 * y - 18 = 0)))
    ∧ (line_equation (-2) 2 2 = line_equation 4 (-2) 2)
    ∧ (line_equation (-2) 2 (-2 / 3) = line_equation 4 (-2) (-2 / 3)) :=
sorry

end find_line_equation_l3_3544


namespace max_independent_set_l3_3276

-- Define the given conditions
variables (P : Type) [fintype P]
variables (acquainted : P → P → Prop)
variables (h_size : fintype.card P = 30)
variables (h_acquainted_max : ∀ p : P, (finset.filter (λ q, acquainted p q) finset.univ).card ≤ 5)
variables (h_group_of_five : ∀ (s : finset P), s.card = 5 → ∃ (p q : P), p ∈ s ∧ q ∈ s ∧ ¬acquainted p q)

-- Define the independent set
def independent_set (s : finset P) : Prop :=
  ∀ p q ∈ s, p ≠ q → ¬acquainted p q

-- State the main theorem
theorem max_independent_set : ∃ s : finset P, s.card = 6 ∧ independent_set s :=
sorry

end max_independent_set_l3_3276


namespace min_sum_bn_l3_3427

theorem min_sum_bn (b : ℕ → ℤ) 
  (h₁ : ∀ n, b n = 2 * n - 31) 
  (h₂ : ∀ n, b n ∈ Int) :
  ∃ n : ℕ, n = 15 ∧ (∀ m : ℕ, m ≠ 15 → (∑ i in Finset.range (m + 1), b i) > (∑ i in Finset.range (15 + 1), b i)) :=
sorry

end min_sum_bn_l3_3427


namespace solve_equation_l3_3165

noncomputable def equation := 
  (λ x : ℂ, (x^3 + 3 * x^2 * Complex.sqrt 3 + 8 * x + 2 * Complex.sqrt 3))

theorem solve_equation :
  ∀ x : ℂ, equation x = 0 ↔ (x = Complex.sqrt 3 ∨ x = Complex.sqrt 3 + Complex.I * Complex.sqrt 2 ∨ x = Complex.sqrt 3 - Complex.I * Complex.sqrt 2) :=
by
  sorry

end solve_equation_l3_3165


namespace jack_afternoon_emails_l3_3149

theorem jack_afternoon_emails : 
  ∀ (morning_emails afternoon_emails : ℕ), 
  morning_emails = 6 → 
  afternoon_emails = morning_emails + 2 → 
  afternoon_emails = 8 := 
by
  intros morning_emails afternoon_emails hm ha
  rw [hm] at ha
  exact ha

end jack_afternoon_emails_l3_3149


namespace line_equations_through_point_with_intercepts_l3_3611

theorem line_equations_through_point_with_intercepts (x y : ℝ) :
  (x = -10 ∧ y = 10) ∧ (∃ a : ℝ, 4 * a = intercept_x ∧ a = intercept_y) →
  (x + y = 0 ∨ x + 4 * y - 30 = 0) :=
by
  sorry

end line_equations_through_point_with_intercepts_l3_3611


namespace stratified_sampling_medium_supermarkets_l3_3566

theorem stratified_sampling_medium_supermarkets
  (large_supermarkets : ℕ)
  (medium_supermarkets : ℕ)
  (small_supermarkets : ℕ)
  (sample_size : ℕ)
  (total_supermarkets : ℕ)
  (medium_proportion : ℚ) :
  large_supermarkets = 200 →
  medium_supermarkets = 400 →
  small_supermarkets = 1400 →
  sample_size = 100 →
  total_supermarkets = large_supermarkets + medium_supermarkets + small_supermarkets →
  medium_proportion = (medium_supermarkets : ℚ) / (total_supermarkets : ℚ) →
  medium_supermarkets_to_sample = sample_size * medium_proportion →
  medium_supermarkets_to_sample = 20 :=
sorry

end stratified_sampling_medium_supermarkets_l3_3566


namespace num_triangles_in_circle_l3_3193

noncomputable def num_triangles (n : ℕ) : ℕ :=
  n.choose 3

theorem num_triangles_in_circle (n : ℕ) :
  num_triangles n = n.choose 3 :=
by
  sorry

end num_triangles_in_circle_l3_3193


namespace find_5_digit_number_l3_3241

theorem find_5_digit_number {A B C D E : ℕ} 
  (hA_even : A % 2 = 0) 
  (hB_even : B % 2 = 0) 
  (hA_half_B : A = B / 2) 
  (hC_sum : C = A + B) 
  (hDE_prime : Prime (10 * D + E)) 
  (hD_3B : D = 3 * B) : 
  10000 * A + 1000 * B + 100 * C + 10 * D + E = 48247 := 
sorry

end find_5_digit_number_l3_3241


namespace combined_income_is_16800_l3_3696

-- Given conditions
def ErnieOldIncome : ℕ := 6000
def ErnieCurrentIncome : ℕ := (4 * ErnieOldIncome) / 5
def JackCurrentIncome : ℕ := 2 * ErnieOldIncome

-- Proof that their combined income is $16800
theorem combined_income_is_16800 : ErnieCurrentIncome + JackCurrentIncome = 16800 := by
  sorry

end combined_income_is_16800_l3_3696


namespace product_arithmetic_sequence_mod_100_l3_3416

def is_arithmetic_sequence (a : ℕ) (d : ℕ) (n : ℕ → Prop) : Prop :=
  ∀ k, n k → k = a + d * (k / d)

theorem product_arithmetic_sequence_mod_100 :
  ∀ P : ℕ,
    (∀ k, 7 ≤ k ∧ k ≤ 1999 ∧ ((k - 7) % 12 = 0) → P = k) →
    (P % 100 = 75) :=
by {
  sorry
}

end product_arithmetic_sequence_mod_100_l3_3416


namespace age_of_son_l3_3051

theorem age_of_son (S M : ℕ) 
  (h1 : M = S + 22)
  (h2 : M + 2 = 2 * (S + 2)) : 
  S = 20 := 
sorry

end age_of_son_l3_3051


namespace eval_exp_l3_3095

theorem eval_exp {a b : ℝ} (h : a = 3^4) : a^(5/4) = 243 :=
by
  sorry

end eval_exp_l3_3095


namespace allens_mothers_age_l3_3228

-- Define the conditions
variables (A M S : ℕ) -- Declare variables for ages of Allen, his mother, and his sister

-- Define Allen is 30 years younger than his mother
axiom h1 : A = M - 30

-- Define Allen's sister is 5 years older than him
axiom h2 : S = A + 5

-- Define in 7 years, the sum of their ages will be 110
axiom h3 : (A + 7) + (M + 7) + (S + 7) = 110

-- Define the age difference between Allen's mother and sister is 25 years
axiom h4 : M - S = 25

-- State the theorem: what is the present age of Allen's mother
theorem allens_mothers_age : M = 48 :=
by sorry

end allens_mothers_age_l3_3228


namespace math_problem_l3_3643

theorem math_problem :
  (1 / (1 / (1 / (1 / (3 + 2 : ℝ) - 1 : ℝ) - 1 : ℝ) - 1 : ℝ) - 1 : ℝ) = -13 / 9 :=
by
  -- proof goes here
  sorry

end math_problem_l3_3643


namespace gcf_7_factorial_8_factorial_l3_3404

theorem gcf_7_factorial_8_factorial :
  let factorial (n : ℕ) := Nat.factorial n in
  let seven_factorial := factorial 7 in
  let eight_factorial := factorial 8 in
  ∃ (gcf : ℕ), gcf = Nat.gcd seven_factorial eight_factorial ∧ gcf = 5040 :=
by
  let factorial (n : ℕ) := Nat.factorial n
  let seven_factorial := factorial 7
  let eight_factorial := factorial 8
  have seven_factorial_eq : seven_factorial = 5040 := by sorry
  have gcf_eq_seven_factorial : Nat.gcd seven_factorial eight_factorial = seven_factorial := by sorry
  exact ⟨seven_factorial, gcf_eq_seven_factorial, seven_factorial_eq⟩

end gcf_7_factorial_8_factorial_l3_3404


namespace challenging_math_problem_l3_3763

theorem challenging_math_problem :
  ((9^2 + (3^3 - 1) * 4^2) % 6) * Real.sqrt 49 + (15 - 3 * 5) = 35 :=
by
  sorry

end challenging_math_problem_l3_3763


namespace correct_statements_l3_3735

variable (a_1 a_2 b_1 b_2 : ℝ)

def ellipse1 := ∀ x y : ℝ, x^2 / a_1^2 + y^2 / b_1^2 = 1
def ellipse2 := ∀ x y : ℝ, x^2 / a_2^2 + y^2 / b_2^2 = 1

axiom a1_pos : a_1 > 0
axiom b1_pos : b_1 > 0
axiom a2_gt_b2_pos : a_2 > b_2 ∧ b_2 > 0
axiom same_foci : a_1^2 - b_1^2 = a_2^2 - b_2^2
axiom a1_gt_a2 : a_1 > a_2

theorem correct_statements : 
  (¬(∃ x y, (x^2 / a_1^2 + y^2 / b_1^2 = 1) ∧ (x^2 / a_2^2 + y^2 / b_2^2 = 1))) ∧ 
  (a_1^2 - a_2^2 = b_1^2 - b_2^2) :=
by 
  sorry

end correct_statements_l3_3735


namespace sum_even_integers_l3_3324

theorem sum_even_integers (sum_first_50_even : Nat) (sum_from_100_to_200 : Nat) : 
  sum_first_50_even = 2550 → sum_from_100_to_200 = 7550 :=
by
  sorry

end sum_even_integers_l3_3324


namespace sin_330_eq_neg_one_half_l3_3878

theorem sin_330_eq_neg_one_half : sin (330 * π / 180) = -1 / 2 := by
  sorry

end sin_330_eq_neg_one_half_l3_3878


namespace red_given_red_l3_3353

def p_i (i : ℕ) : ℚ := sorry
axiom lights_probs_eq : p_i 1 + p_i 2 = 2 / 3
axiom lights_probs_eq2 : p_i 1 + p_i 3 = 2 / 3
axiom green_given_green : p_i 1 / (p_i 1 + p_i 2) = 3 / 4
axiom total_prob : p_i 1 + p_i 2 + p_i 3 + p_i 4 = 1

theorem red_given_red : (p_i 4 / (p_i 3 + p_i 4)) = 1 / 2 := 
sorry

end red_given_red_l3_3353


namespace find_k_inv_h_of_10_l3_3171

-- Assuming h and k are functions with appropriate properties
variables (h k : ℝ → ℝ)
variables (h_inv : ℝ → ℝ) (k_inv : ℝ → ℝ)

-- Given condition: h_inv (k(x)) = 4 * x - 5
axiom h_inv_k_eq : ∀ x, h_inv (k x) = 4 * x - 5

-- Statement to prove
theorem find_k_inv_h_of_10 :
  k_inv (h 10) = 15 / 4 := 
sorry

end find_k_inv_h_of_10_l3_3171


namespace ratio_of_m1_m2_l3_3458

open Real

theorem ratio_of_m1_m2 :
  ∀ (m : ℝ) (p q : ℝ), p ≠ 0 ∧ q ≠ 0 ∧ m ≠ 0 ∧
    (p + q = -((3 - 2 * m) / m)) ∧ 
    (p * q = 4 / m) ∧ 
    (p / q + q / p = 2) → 
   ∃ (m1 m2 : ℝ), 
    (4 * m1^2 - 28 * m1 + 9 = 0) ∧
    (4 * m2^2 - 28 * m2 + 9 = 0) ∧ 
    (m1 ≠ m2) ∧ 
    (m1 + m2 = 7) ∧ 
    (m1 * m2 = 9 / 4) ∧ 
    (m1 / m2 + m2 / m1 = 178 / 9) :=
by sorry

end ratio_of_m1_m2_l3_3458


namespace unique_solution_system_eqns_l3_3602

theorem unique_solution_system_eqns (a b c : ℕ) :
  (a^3 - b^3 - c^3 = 3 * a * b * c) ∧ (a^2 = 2 * (b + c)) ↔ (a = 2 ∧ b = 1 ∧ c = 1) := by 
  sorry

end unique_solution_system_eqns_l3_3602


namespace gcf_7_8_fact_l3_3410

noncomputable def factorial : ℕ → ℕ
| 0     := 1
| (n+1) := (n+1) * factorial n

theorem gcf_7_8_fact : Nat.gcd (factorial 7) (factorial 8) = 5040 := by
  sorry

end gcf_7_8_fact_l3_3410


namespace sin_330_eq_neg_one_half_l3_3914

theorem sin_330_eq_neg_one_half :
  ∃ θ : ℝ, sin (330 * real.pi / 180) = -1 / 2 :=
begin
  have h1 : sin (360 * real.pi / 180 - 30 * real.pi / 180) = -sin (30 * real.pi / 180),
  { rw sin_sub,
    ring,
    rw sin_pi_div_six, -- sin 30° = 1 / 2
  },
  have h2 : sin (30 * real.pi / 180) = 1 / 2 := by exact real.sin_pi_div_six,
  use 330,
  calc
    sin (330 * real.pi / 180)
        = sin ((360 - 30) * real.pi / 180)      : by ring
    ... = sin (360 * real.pi / 180 - 30 * real.pi / 180) : by congr' 1; ring
    ... = -sin (30 * real.pi / 180)            : by exact h1
    ... = -1 / 2                               : by rw h2,
end

end sin_330_eq_neg_one_half_l3_3914


namespace largest_integer_in_interval_l3_3415

theorem largest_integer_in_interval :
  (∃ x : ℤ, (1 : ℚ) / 4 < x / 6 ∧ x / 6 < 7 / 11 ∧ ∀ y : ℤ, (1 : ℚ) / 4 < y / 6 → y / 6 < 7 / 11 → y ≤ x) :=
begin
  use 3,
  split,
  { norm_num,
    split,
    { norm_num },
    { intros y hy1 hy2,
      norm_num at hy1 hy2,
      linarith }
  },
  sorry
end

end largest_integer_in_interval_l3_3415


namespace sin_330_deg_l3_3869

noncomputable theory

open Real

theorem sin_330_deg :
  sin (330 * (π / 180)) = -1 / 2 :=
by sorry

end sin_330_deg_l3_3869


namespace probability_of_selecting_specific_letters_l3_3372

theorem probability_of_selecting_specific_letters :
  let total_cards := 15
  let amanda_cards := 6
  let chloe_or_ethan_cards := 9
  let prob_amanda_then_chloe_or_ethan := (amanda_cards / total_cards) * (chloe_or_ethan_cards / (total_cards - 1))
  let prob_chloe_or_ethan_then_amanda := (chloe_or_ethan_cards / total_cards) * (amanda_cards / (total_cards - 1))
  let total_prob := prob_amanda_then_chloe_or_ethan + prob_chloe_or_ethan_then_amanda
  total_prob = 18 / 35 :=
by
  sorry

end probability_of_selecting_specific_letters_l3_3372


namespace focus_of_parabola_l3_3549

theorem focus_of_parabola (a : ℝ) (h1 : a > 0)
  (h2 : ∀ x, y = 3 * x → 3 / a = 3) :
  ∃ (focus : ℝ × ℝ), focus = (0, 1 / 8) :=
by
  -- The proof goes here
  sorry

end focus_of_parabola_l3_3549


namespace find_value_m_sq_plus_2m_plus_n_l3_3289

noncomputable def m_n_roots (x : ℝ) : Prop := x^2 + x - 1001 = 0

theorem find_value_m_sq_plus_2m_plus_n
  (m n : ℝ)
  (hm : m_n_roots m)
  (hn : m_n_roots n)
  (h_sum : m + n = -1)
  (h_prod : m * n = -1001) :
  m^2 + 2 * m + n = 1000 :=
sorry

end find_value_m_sq_plus_2m_plus_n_l3_3289


namespace sin_330_eq_neg_half_l3_3832

theorem sin_330_eq_neg_half : sin (330 * (π / 180)) = -1 / 2 := 
by
  -- condition: \(330^\circ = 360^\circ - 30^\circ\)
  -- condition: \(\sin 30^\circ = \frac{1}{2}\)
  -- condition: sine is negative in the fourth quadrant
  sorry

end sin_330_eq_neg_half_l3_3832


namespace number_of_medium_boxes_l3_3236

def large_box_tape := 4
def medium_box_tape := 2
def small_box_tape := 1
def label_tape := 1

def num_large_boxes := 2
def num_small_boxes := 5
def total_tape := 44

theorem number_of_medium_boxes :
  let tape_used_large_boxes := num_large_boxes * (large_box_tape + label_tape)
  let tape_used_small_boxes := num_small_boxes * (small_box_tape + label_tape)
  let tape_used_medium_boxes := total_tape - (tape_used_large_boxes + tape_used_small_boxes)
  let medium_box_total_tape := medium_box_tape + label_tape
  let num_medium_boxes := tape_used_medium_boxes / medium_box_total_tape
  num_medium_boxes = 8 :=
by
  sorry

end number_of_medium_boxes_l3_3236


namespace sin_330_eq_neg_one_half_l3_3963

theorem sin_330_eq_neg_one_half : Real.sin (330 * Real.pi / 180) = -1 / 2 := by
  sorry

end sin_330_eq_neg_one_half_l3_3963


namespace volume_of_defined_region_l3_3419

noncomputable def volume_of_region (x y z : ℝ) : ℝ :=
if x + y ≤ 5 ∧ z ≤ 5 ∧ 0 ≤ x ∧ 0 ≤ y ∧ 0 ≤ z ∧ x ≤ 2 then 15 else 0

theorem volume_of_defined_region :
  ∀ (x y z : ℝ),
  (0 ≤ x) → (0 ≤ y) → (0 ≤ z) → (x ≤ 2) →
  (|x + y + z| + |x + y - z| ≤ 10) →
  volume_of_region x y z = 15 :=
sorry

end volume_of_defined_region_l3_3419


namespace factorize_m_sq_minus_one_l3_3240

theorem factorize_m_sq_minus_one (m : ℝ) : m^2 - 1 = (m + 1) * (m - 1) := 
by
  sorry

end factorize_m_sq_minus_one_l3_3240


namespace tangent_line_eq_l3_3472

-- Define the function f(x)
def f (x : ℝ) : ℝ := x^2 - 3 * x + 1

-- Define the point at which we are evaluating the tangent
def point : ℝ × ℝ := (1, -1)

-- Define the derivative of the function f(x)
def f' (x : ℝ) : ℝ := 2 * x - 3

-- The desired theorem
theorem tangent_line_eq :
  ∀ x y : ℝ, (x, y) = point → (y = -x) :=
by sorry

end tangent_line_eq_l3_3472


namespace carousel_ticket_cost_l3_3300

theorem carousel_ticket_cost :
  ∃ (x : ℕ), 
  (2 * 5) + (3 * x) = 19 ∧ x = 3 :=
by
  sorry

end carousel_ticket_cost_l3_3300


namespace transform_expression_to_product_l3_3784

open Real

noncomputable def transform_expression (α : ℝ) : ℝ :=
  4.66 * sin (5 * π / 2 + 4 * α) - (sin (5 * π / 2 + 2 * α)) ^ 6 + (cos (7 * π / 2 - 2 * α)) ^ 6

theorem transform_expression_to_product (α : ℝ) :
  transform_expression α = (1 / 8) * sin (4 * α) * sin (8 * α) :=
by
  sorry

end transform_expression_to_product_l3_3784


namespace fraction_of_track_in_forest_l3_3042

theorem fraction_of_track_in_forest (n : ℕ) (l : ℝ) (A B C : ℝ) :
  (∃ x, x = 2*l/3 ∨ x = l/3) → (∃ f, 0 < f ∧ f ≤ 1 ∧ (f = 2/3 ∨ f = 1/3)) :=
by
  -- sorry, the proof will go here
  sorry

end fraction_of_track_in_forest_l3_3042


namespace set_union_example_l3_3120

open Set

theorem set_union_example :
  let A := ({1, 3, 5, 6} : Set ℤ)
  let B := ({-1, 5, 7} : Set ℤ)
  A ∪ B = ({-1, 1, 3, 5, 6, 7} : Set ℤ) :=
by
  intros
  sorry

end set_union_example_l3_3120


namespace sin_330_deg_l3_3855

theorem sin_330_deg : Real.sin (330 * Real.pi / 180) = -1 / 2 := by
  sorry

end sin_330_deg_l3_3855


namespace sin_330_is_minus_sqrt3_over_2_l3_3989

-- Define the angle theta as 330 degrees
def theta := 330 * Real.toRad

-- State that point corresponding to 330 degrees lies in the fourth quadrant
def point_in_fourth_quadrant (θ : ℝ) := 330 * Real.toRad = θ ∧ θ > 270 * Real.toRad ∧ θ < 360 * Real.toRad

-- Define a proof problem that states computing sin of 330° equals -√3/2
theorem sin_330_is_minus_sqrt3_over_2 : point_in_fourth_quadrant theta → Real.sin theta = -Real.sqrt 3 / 2 := by
  sorry

end sin_330_is_minus_sqrt3_over_2_l3_3989


namespace conversion_7_dms_to_cms_conversion_5_hectares_to_sms_conversion_600_hectares_to_sqkms_conversion_200_sqsmeters_to_smeters_l3_3208

theorem conversion_7_dms_to_cms :
  7 * 100 = 700 :=
by
  sorry

theorem conversion_5_hectares_to_sms :
  5 * 10000 = 50000 :=
by
  sorry

theorem conversion_600_hectares_to_sqkms :
  600 / 100 = 6 :=
by
  sorry

theorem conversion_200_sqsmeters_to_smeters :
  200 / 100 = 2 :=
by
  sorry

end conversion_7_dms_to_cms_conversion_5_hectares_to_sms_conversion_600_hectares_to_sqkms_conversion_200_sqsmeters_to_smeters_l3_3208


namespace sum_of_digits_of_N_plus_2021_is_10_l3_3476

-- The condition that N is the smallest positive integer whose digits add to 41.
def smallest_integer_with_digit_sum_41 (N : ℕ) : Prop :=
  (N > 0) ∧ ((N.digits 10).sum = 41)

-- The Lean 4 statement to prove the problem.
theorem sum_of_digits_of_N_plus_2021_is_10 :
  ∃ N : ℕ, smallest_integer_with_digit_sum_41 N ∧ ((N + 2021).digits 10).sum = 10 :=
by
  -- The proof would go here
  sorry

end sum_of_digits_of_N_plus_2021_is_10_l3_3476


namespace nearest_integer_is_11304_l3_3787

def nearest_integer_to_a_plus_b_pow_six (a b : ℝ) (h : b = Real.sqrt 5) : ℝ :=
  (a + b) ^ 6

theorem nearest_integer_is_11304 : nearest_integer_to_a_plus_b_pow_six 3 (Real.sqrt 5) rfl = 11304 := 
  sorry

end nearest_integer_is_11304_l3_3787


namespace sin_330_eq_neg_half_l3_3892

theorem sin_330_eq_neg_half : Real.sin (330 * Real.pi / 180) = -1 / 2 := by
  -- Definitions and conditions from the problem
  have h1 : 330 = 360 - 30 := rfl
  have h2 : ∀ θ, Real.sin θ = -Real.sin (θ - 2 * Real.pi) := sorry
  have h3 : Real.sin (30 * Real.pi / 180) = 1 / 2 := sorry

  -- Use the given conditions to prove the equality
  calc
    Real.sin (330 * Real.pi / 180)
        = Real.sin (-(30 * Real.pi / 180) + 2 * Real.pi) : by rw [←h1, Real.sin_angle_sub_pi]
    ... = -Real.sin (30 * Real.pi / 180) : by rw [h2]
    ... = -1 / 2 : by rw [h3]

end sin_330_eq_neg_half_l3_3892


namespace opposite_neg_three_over_two_l3_3475

-- Define the concept of the opposite number
def opposite (x : ℚ) : ℚ := -x

-- State the problem: The opposite number of -3/2 is 3/2
theorem opposite_neg_three_over_two :
  opposite (- (3 / 2 : ℚ)) = (3 / 2 : ℚ) := 
  sorry

end opposite_neg_three_over_two_l3_3475


namespace problem_statement_l3_3800

theorem problem_statement : 2^2 * 3^2 * 5^2 * 7 = 6300 := by
  sorry

end problem_statement_l3_3800


namespace arrange_leopards_correct_l3_3749

-- Definitions for conditions
def num_shortest : ℕ := 3
def total_leopards : ℕ := 9
def factorial (n : ℕ) : ℕ := if n = 0 then 1 else n * factorial (n - 1)

-- Calculation of total ways to arrange given conditions
def arrange_leopards (num_shortest : ℕ) (total_leopards : ℕ) : ℕ :=
  let choose2short := (num_shortest * (num_shortest - 1)) / 2
  let arrange2short := 2 * factorial (total_leopards - num_shortest)
  choose2short * arrange2short * factorial (total_leopards - num_shortest)

theorem arrange_leopards_correct :
  arrange_leopards num_shortest total_leopards = 30240 := by
  sorry

end arrange_leopards_correct_l3_3749


namespace gcf_factorial_seven_eight_l3_3402

theorem gcf_factorial_seven_eight (a b : ℕ) (h : a = 7! ∧ b = 8!) : Nat.gcd a b = 7! := 
by 
  sorry

end gcf_factorial_seven_eight_l3_3402


namespace length_of_AX_l3_3279

theorem length_of_AX 
  (A B C X : Type) 
  (AB AC BC AX BX : ℕ) 
  (hx : AX + BX = AB)
  (h_angle_bisector : AC * BX = BC * AX)
  (h_AB : AB = 40)
  (h_BC : BC = 35)
  (h_AC : AC = 21) : 
  AX = 15 :=
by
  sorry

end length_of_AX_l3_3279


namespace sin_330_eq_neg_one_half_l3_3956

theorem sin_330_eq_neg_one_half : Real.sin (330 * Real.pi / 180) = -1 / 2 := by
  sorry

end sin_330_eq_neg_one_half_l3_3956


namespace sin_330_eq_neg_one_half_l3_3877

theorem sin_330_eq_neg_one_half : sin (330 * π / 180) = -1 / 2 := by
  sorry

end sin_330_eq_neg_one_half_l3_3877


namespace mixed_groups_count_l3_3013

theorem mixed_groups_count (total_children : ℕ) (total_groups : ℕ)
  (group_size : ℕ) (total_boy_boy_photos : ℕ)
  (total_girl_girl_photos : ℕ)
  (total_photos : ℕ)
  (each_group_photos : ℕ)
  (total_mixed_photos : ℕ)
  (mixed_group_count : ℕ):
  total_children = 300 ∧
  total_groups = 100 ∧
  group_size = 3 ∧
  total_boy_boy_photos = 100 ∧
  total_girl_girl_photos = 56 ∧
  each_group_photos = 3 ∧
  total_photos = 100 * each_group_photos ∧
  total_mixed_photos = total_photos - total_boy_boy_photos - total_girl_girl_photos ∧
  mixed_group_count = total_mixed_photos / 2 →
  mixed_group_count = 72 :=
by
  sorry

end mixed_groups_count_l3_3013


namespace tan_alpha_minus_beta_l3_3714

theorem tan_alpha_minus_beta
  (α β : ℝ)
  (tan_alpha : Real.tan α = 2)
  (tan_beta : Real.tan β = -7) :
  Real.tan (α - β) = -9 / 13 :=
by sorry

end tan_alpha_minus_beta_l3_3714


namespace sin_330_is_minus_sqrt3_over_2_l3_3992

-- Define the angle theta as 330 degrees
def theta := 330 * Real.toRad

-- State that point corresponding to 330 degrees lies in the fourth quadrant
def point_in_fourth_quadrant (θ : ℝ) := 330 * Real.toRad = θ ∧ θ > 270 * Real.toRad ∧ θ < 360 * Real.toRad

-- Define a proof problem that states computing sin of 330° equals -√3/2
theorem sin_330_is_minus_sqrt3_over_2 : point_in_fourth_quadrant theta → Real.sin theta = -Real.sqrt 3 / 2 := by
  sorry

end sin_330_is_minus_sqrt3_over_2_l3_3992


namespace problem_statement_l3_3565

theorem problem_statement (p q : Prop) :
  ¬(p ∧ q) ∧ ¬¬p → ¬q := 
by 
  sorry

end problem_statement_l3_3565


namespace largest_three_digit_divisible_by_6_l3_3484

-- Defining what it means for a number to be divisible by 6, 2, and 3
def divisible_by (n d : ℕ) : Prop := ∃ k : ℕ, n = d * k

-- Conditions extracted from the problem
def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999
def last_digit_even (n : ℕ) : Prop := (n % 10) % 2 = 0
def sum_of_digits_divisible_by_3 (n : ℕ) : Prop := ((n / 100) + (n / 10 % 10) + (n % 10)) % 3 = 0

-- Define what it means for a number to be divisible by 6 according to the conditions
def divisible_by_6 (n : ℕ) : Prop := last_digit_even n ∧ sum_of_digits_divisible_by_3 n

-- Prove that 996 is the largest three-digit number that satisfies these conditions
theorem largest_three_digit_divisible_by_6 (n : ℕ) : is_three_digit n ∧ divisible_by_6 n → n ≤ 996 :=
by
    sorry

end largest_three_digit_divisible_by_6_l3_3484


namespace nearest_integer_to_expr_l3_3794

theorem nearest_integer_to_expr : 
  let a := 3 + Real.sqrt 5
  let b := (a)^6
  abs (b - 2744) < 1
:= sorry

end nearest_integer_to_expr_l3_3794


namespace range_of_a_l3_3585

theorem range_of_a (a : ℝ) (h : ∀ x : ℝ, (x^2 + (a + 2) * x + 1) * ((3 - 2 * a) * x^2 + 5 * x + (3 - 2 * a)) ≥ 0) : a ∈ Set.Icc (-4 : ℝ) 0 := sorry

end range_of_a_l3_3585


namespace simplify_sqrt_144000_l3_3599

theorem simplify_sqrt_144000 :
  (sqrt 144000 = 120 * sqrt 10) :=
by
  -- Assume given conditions
  have h1 : 144000 = 144 * 1000 := by
    calc 144000 = 144 * 1000 : by rfl

  have h2 : 144 = 12^2 := by rfl

  have h3 : sqrt (a * b) = sqrt a * sqrt b := by sorry

  have h4 : sqrt (10 ^ 3) = 10 * sqrt 10 := by sorry

  -- Prove the target
  calc
    sqrt 144000
    = sqrt (144 * 1000) : by rw [←h1]
    = sqrt (12^2 * 10^3) : by rw [h2, pow_succ]
    = sqrt (12^2) * sqrt (10^3) : by rw [h3]
    = 12 * sqrt (10^3) : by rw [sqrt_sq', h2, pow_two]
    = 12 * (10 * sqrt 10) : by rw [h4]
    = 12 * 10 * sqrt 10 : by rw [mul_assoc]
    = 120 * sqrt 10 : by sorry

-- sqrt_sq' and pow_two are used to simplify sqrt (12^2) == 12.

end simplify_sqrt_144000_l3_3599


namespace no_real_roots_other_than_zero_l3_3108

theorem no_real_roots_other_than_zero (k : ℝ) (h : k ≠ 0):
  ¬(∃ x : ℝ, x^2 + 2 * k * x + 3 * k^2 = 0) :=
by
  sorry

end no_real_roots_other_than_zero_l3_3108


namespace max_surface_area_of_rectangular_solid_on_sphere_l3_3270

noncomputable def max_surface_area_rectangular_solid (a b c : ℝ) :=
  2 * a * b + 2 * a * c + 2 * b * c

theorem max_surface_area_of_rectangular_solid_on_sphere :
  (∀ (a b c : ℝ), a^2 + b^2 + c^2 = 36 → max_surface_area_rectangular_solid a b c ≤ 72) :=
by
  intros a b c h
  sorry

end max_surface_area_of_rectangular_solid_on_sphere_l3_3270


namespace sin_330_eq_neg_one_half_l3_3860

theorem sin_330_eq_neg_one_half :
  ∃ θ, θ = 330 ∧ (sin (Real.pi * (θ / 180)) = -1 / 2) :=
by
  use 330
  split
  · rfl
  · sorry

end sin_330_eq_neg_one_half_l3_3860


namespace remainder_when_divided_l3_3176

theorem remainder_when_divided (L S R : ℕ) (h1: L - S = 1365) (h2: S = 270) (h3: L = 6 * S + R) : 
  R = 15 := 
by 
  sorry

end remainder_when_divided_l3_3176


namespace sin_330_eq_neg_sin_30_l3_3983

theorem sin_330_eq_neg_sin_30 :
  sin (330 : ℝ) = - sin (30 : ℝ) := sorry

end sin_330_eq_neg_sin_30_l3_3983


namespace sin_330_eq_neg_half_l3_3942

noncomputable def Q : ℝ × ℝ := (Real.sqrt 3 / 2, -1 / 2)

theorem sin_330_eq_neg_half : Real.sin (330 * Real.pi / 180) = -1 / 2 :=
by
  have h1 : 330 * Real.pi / 180 = 11 * Real.pi / 6 := by sorry
  have coord_y : (Q.snd) = -1 / 2 := by sorry
  rw [h1, coord_y]
  sorry

end sin_330_eq_neg_half_l3_3942


namespace sin_330_eq_neg_sin_30_l3_3979

theorem sin_330_eq_neg_sin_30 :
  sin (330 : ℝ) = - sin (30 : ℝ) := sorry

end sin_330_eq_neg_sin_30_l3_3979


namespace sin_330_deg_l3_3867

noncomputable theory

open Real

theorem sin_330_deg :
  sin (330 * (π / 180)) = -1 / 2 :=
by sorry

end sin_330_deg_l3_3867


namespace problem1_problem2_exists_largest_k_real_problem3_exists_largest_k_int_l3_3761

-- Problem 1: Prove the inequality for all real numbers x, y
theorem problem1 (x y : ℝ) : x^2 + y^2 + 1 > x * (y + 1) :=
sorry

-- Problem 2: Prove the largest k = sqrt(2) for the inequality with reals
theorem problem2_exists_largest_k_real : ∃ (k : ℝ), (∀ (x y : ℝ), x^2 + y^2 + 1 ≥ k * x * (y + 1)) ∧ k = Real.sqrt 2 :=
sorry

-- Problem 3: Prove the largest k = 3/2 for the inequality with integers
theorem problem3_exists_largest_k_int : ∃ (k : ℝ), (∀ (m n : ℤ), m^2 + n^2 + 1 ≥ k * m * (n + 1)) ∧ k = 3 / 2 :=
sorry

end problem1_problem2_exists_largest_k_real_problem3_exists_largest_k_int_l3_3761


namespace boat_speed_in_still_water_l3_3363

-- Problem Definitions
def V_s : ℕ := 16
def t : ℕ := sorry -- t is arbitrary positive value
def V_b : ℕ := 48

-- Conditions
def upstream_time := 2 * t
def downstream_time := t
def upstream_distance := (V_b - V_s) * upstream_time
def downstream_distance := (V_b + V_s) * downstream_time

-- Proof Problem
theorem boat_speed_in_still_water :
  upstream_distance = downstream_distance → V_b = 48 :=
by sorry

end boat_speed_in_still_water_l3_3363


namespace ratatouille_cost_per_quart_l3_3301

theorem ratatouille_cost_per_quart:
  let eggplants_pounds := 5
  let eggplants_cost_per_pound := 2.00
  let zucchini_pounds := 4
  let zucchini_cost_per_pound := 2.00
  let tomatoes_pounds := 4
  let tomatoes_cost_per_pound := 3.50
  let onions_pounds := 3
  let onions_cost_per_pound := 1.00
  let basil_pounds := 1
  let basil_cost_per_half_pound := 2.50
  let total_quarts := 4
  let eggplants_cost := eggplants_pounds * eggplants_cost_per_pound
  let zucchini_cost := zucchini_pounds * zucchini_cost_per_pound
  let tomatoes_cost := tomatoes_pounds * tomatoes_cost_per_pound
  let onions_cost := onions_pounds * onions_cost_per_pound
  let basil_cost := basil_pounds * (basil_cost_per_half_pound / 0.5)
  let total_cost := eggplants_cost + zucchini_cost + tomatoes_cost + onions_cost + basil_cost
  let cost_per_quart := total_cost / total_quarts
  cost_per_quart = 10.00 :=
  by
    sorry

end ratatouille_cost_per_quart_l3_3301


namespace boaster_guarantee_distinct_balls_l3_3719

noncomputable def canGuaranteeDistinctBallCounts (boxes : Fin 2018 → ℕ) (pairs : Fin 4032 → (Fin 2018 × Fin 2018)) : Prop :=
  ∀ i j : Fin 2018, i ≠ j → boxes i ≠ boxes j

theorem boaster_guarantee_distinct_balls :
  ∃ (boxes : Fin 2018 → ℕ) (pairs : Fin 4032 → (Fin 2018 × Fin 2018)),
  canGuaranteeDistinctBallCounts boxes pairs :=
sorry

end boaster_guarantee_distinct_balls_l3_3719


namespace bikers_meet_again_in_36_minutes_l3_3631

theorem bikers_meet_again_in_36_minutes :
    Nat.lcm 12 18 = 36 :=
sorry

end bikers_meet_again_in_36_minutes_l3_3631


namespace sin_330_eq_neg_one_half_l3_3882

theorem sin_330_eq_neg_one_half : sin (330 * π / 180) = -1 / 2 := by
  sorry

end sin_330_eq_neg_one_half_l3_3882


namespace sin_330_eq_neg_half_l3_3842

theorem sin_330_eq_neg_half 
  (h1: (330 : ℝ) = 330)
  (h2: (330 : ℝ = 360 - 30))
  (h3: 30 ∈ reference_angle_set ∧ sin 30 = 1/2)
  (h4: 330 ∈ fourth_quadrant)
  (h5: ∀ θ ∈ fourth_quadrant, sin θ < 0) :
  sin 330 = -1/2 := 
by
  sorry

end sin_330_eq_neg_half_l3_3842


namespace taylor_one_basket_in_three_tries_l3_3394

theorem taylor_one_basket_in_three_tries (P_no_make : ℚ) (h : P_no_make = 1/3) : 
  (∃ P_make : ℚ, P_make = 1 - P_no_make ∧ P_make * P_no_make * P_no_make * 3 = 2/9) := 
by
  sorry

end taylor_one_basket_in_three_tries_l3_3394


namespace sin_330_eq_neg_one_half_l3_3957

theorem sin_330_eq_neg_one_half : Real.sin (330 * Real.pi / 180) = -1 / 2 := by
  sorry

end sin_330_eq_neg_one_half_l3_3957


namespace sin_330_eq_neg_half_l3_3895

theorem sin_330_eq_neg_half : Real.sin (330 * Real.pi / 180) = -1 / 2 := by
  -- Definitions and conditions from the problem
  have h1 : 330 = 360 - 30 := rfl
  have h2 : ∀ θ, Real.sin θ = -Real.sin (θ - 2 * Real.pi) := sorry
  have h3 : Real.sin (30 * Real.pi / 180) = 1 / 2 := sorry

  -- Use the given conditions to prove the equality
  calc
    Real.sin (330 * Real.pi / 180)
        = Real.sin (-(30 * Real.pi / 180) + 2 * Real.pi) : by rw [←h1, Real.sin_angle_sub_pi]
    ... = -Real.sin (30 * Real.pi / 180) : by rw [h2]
    ... = -1 / 2 : by rw [h3]

end sin_330_eq_neg_half_l3_3895


namespace initial_price_of_TV_l3_3376

theorem initial_price_of_TV (T : ℤ) (phone_price_increase : ℤ) (total_amount : ℤ) 
    (h1 : phone_price_increase = (400: ℤ) + (40 * 400 / 100)) 
    (h2 : total_amount = T + (2 * T / 5) + phone_price_increase) 
    (h3 : total_amount = 1260) : 
    T = 500 := by
  sorry

end initial_price_of_TV_l3_3376


namespace solve_inequality_l3_3550

theorem solve_inequality (x : ℝ) : -4 * x - 8 > 0 → x < -2 := sorry

end solve_inequality_l3_3550


namespace gcf_factorial_seven_eight_l3_3400

theorem gcf_factorial_seven_eight (a b : ℕ) (h : a = 7! ∧ b = 8!) : Nat.gcd a b = 7! := 
by 
  sorry

end gcf_factorial_seven_eight_l3_3400


namespace mixed_groups_count_l3_3033

-- Define the facts about the groups and photographs
def numberOfChildren : Nat := 300
def numberOfGroups : Nat := 100
def childrenPerGroup : Nat := 3
def b_b_photos : Nat := 100
def g_g_photos : Nat := 56

-- Define the function to calculate mixed groups
def mixedGroups (totalPhotos b_b_photos g_g_photos : Nat) : Nat := 
  (totalPhotos - b_b_photos - g_g_photos) / 2

-- State the theorem
theorem mixed_groups_count : 
  mixedGroups (numberOfGroups * childrenPerGroup) b_b_photos g_g_photos = 72 := by
  rfl

end mixed_groups_count_l3_3033


namespace ratio_of_sides_l3_3367

theorem ratio_of_sides (s r : ℝ) (h : s^2 = 2 * r^2 * Real.sqrt 2) : r / s = 1 / Real.sqrt (2 * Real.sqrt 2) := 
by
  sorry

end ratio_of_sides_l3_3367


namespace print_time_l3_3221

/-- Define the number of pages per minute printed by the printer -/
def pages_per_minute : ℕ := 25

/-- Define the total number of pages to be printed -/
def total_pages : ℕ := 350

/-- Prove that the time to print 350 pages at a rate of 25 pages per minute is 14 minutes -/
theorem print_time :
  (total_pages / pages_per_minute) = 14 :=
by
  sorry

end print_time_l3_3221


namespace total_votes_cast_l3_3649

theorem total_votes_cast (V : ℕ) (h1 : V > 0) (h2 : ∃ c r : ℕ, c = 40 * V / 100 ∧ r = 40 * V / 100 + 5000 ∧ c + r = V):
  V = 25000 :=
by
  sorry

end total_votes_cast_l3_3649


namespace fewest_posts_required_l3_3514

def dimensions_garden : ℕ × ℕ := (32, 72)
def post_spacing : ℕ := 8

theorem fewest_posts_required
  (d : ℕ × ℕ := dimensions_garden)
  (s : ℕ := post_spacing) :
  d = (32, 72) ∧ s = 8 → 
  ∃ N, N = 26 := 
by 
  sorry

end fewest_posts_required_l3_3514


namespace sin_330_eq_neg_sqrt3_div_2_l3_3999

theorem sin_330_eq_neg_sqrt3_div_2 
  (R : ℝ × ℝ)
  (hR : R = (1/2, -sqrt(3)/2))
  : Real.sin (330 * Real.pi / 180) = -sqrt(3)/2 :=
by
  sorry

end sin_330_eq_neg_sqrt3_div_2_l3_3999


namespace parts_of_diagonal_in_rectangle_l3_3513

/-- Proving that a 24x60 rectangle divided by its diagonal results in 1512 parts --/

theorem parts_of_diagonal_in_rectangle :
  let m := 24
  let n := 60
  let gcd_mn := gcd m n
  let unit_squares := m * n
  let diagonal_intersections := m + n - gcd_mn
  unit_squares + diagonal_intersections = 1512 :=
by
  sorry

end parts_of_diagonal_in_rectangle_l3_3513


namespace sin_330_degree_l3_3949

theorem sin_330_degree : Real.sin (330 * Real.pi / 180) = -1 / 2 :=
by
  sorry

end sin_330_degree_l3_3949


namespace probability_of_stopping_on_H_l3_3369

theorem probability_of_stopping_on_H (y : ℚ)
  (h1 : (1 / 5) + (1 / 4) + y + y + (1 / 10) = 1)
  : y = 9 / 40 :=
sorry

end probability_of_stopping_on_H_l3_3369


namespace jean_vs_pauline_cost_l3_3162

-- Definitions based on the conditions given
def patty_cost (ida_cost : ℕ) : ℕ := ida_cost + 10
def ida_cost (jean_cost : ℕ) : ℕ := jean_cost + 30
def pauline_cost : ℕ := 30

noncomputable def total_cost (jean_cost : ℕ) : ℕ :=
jean_cost + ida_cost jean_cost + patty_cost (ida_cost jean_cost) + pauline_cost

-- Lean 4 statement to prove the required condition
theorem jean_vs_pauline_cost :
  ∃ (jean_cost : ℕ), total_cost jean_cost = 160 ∧ pauline_cost - jean_cost = 10 :=
by
  sorry

end jean_vs_pauline_cost_l3_3162


namespace find_B_value_l3_3361

theorem find_B_value (A C B : ℕ) (h1 : A = 634) (h2 : A = C + 593) (h3 : B = C + 482) : B = 523 :=
by {
  -- Proof would go here
  sorry
}

end find_B_value_l3_3361


namespace num_roots_of_unity_satisfy_cubic_l3_3515

def root_of_unity (z : ℂ) (n : ℕ) : Prop :=
  z ^ n = 1

def cubic_eqn_root (z : ℂ) (a b c : ℤ) : Prop :=
  z^3 + (a:ℂ) * z^2 + (b:ℂ) * z + (c:ℂ) = 0

theorem num_roots_of_unity_satisfy_cubic (a b c : ℤ) (n : ℕ) 
    (h_n : n ≥ 1) : ∃! z : ℂ, root_of_unity z n ∧ cubic_eqn_root z a b c := sorry

end num_roots_of_unity_satisfy_cubic_l3_3515


namespace max_sum_of_arithmetic_seq_l3_3255

theorem max_sum_of_arithmetic_seq (a : ℕ → ℤ) (S : ℕ → ℤ) 
  (h₁ : a 1 = 11) (h₂ : a 5 = -1) 
  (h₃ : ∀ n, a n = 14 - 3 * (n - 1)) 
  : ∀ n, (S n = (n * (a 1 + a n) / 2)) → max (S n) = 26 :=
sorry

end max_sum_of_arithmetic_seq_l3_3255


namespace sin_330_eq_neg_half_l3_3916

open Real

theorem sin_330_eq_neg_half : sin (330 * (π / 180)) = - (1 / 2) := by
  have angle_eq : 330 * (π / 180) = 2 * π - (30 * (π / 180)) := by sorry
  have sin_30_pos : sin (30 * (π / 180)) = 1 / 2 := by sorry
  have sin_330_neg : sin (2 * π - (30 * (π / 180))) = - (sin (30 * (π / 180))) := by sorry
  rw [angle_eq, sin_330_neg, sin_30_pos]
  rfl

end sin_330_eq_neg_half_l3_3916


namespace age_ratio_in_future_l3_3110

variables (t j x : ℕ)

theorem age_ratio_in_future:
  (t - 4 = 5 * (j - 4)) → 
  (t - 10 = 6 * (j - 10)) →
  (t + x = 3 * (j + x)) →
  x = 26 := 
by {
  sorry
}

end age_ratio_in_future_l3_3110


namespace sin_330_eq_neg_one_half_l3_3908

theorem sin_330_eq_neg_one_half :
  ∃ θ : ℝ, sin (330 * real.pi / 180) = -1 / 2 :=
begin
  have h1 : sin (360 * real.pi / 180 - 30 * real.pi / 180) = -sin (30 * real.pi / 180),
  { rw sin_sub,
    ring,
    rw sin_pi_div_six, -- sin 30° = 1 / 2
  },
  have h2 : sin (30 * real.pi / 180) = 1 / 2 := by exact real.sin_pi_div_six,
  use 330,
  calc
    sin (330 * real.pi / 180)
        = sin ((360 - 30) * real.pi / 180)      : by ring
    ... = sin (360 * real.pi / 180 - 30 * real.pi / 180) : by congr' 1; ring
    ... = -sin (30 * real.pi / 180)            : by exact h1
    ... = -1 / 2                               : by rw h2,
end

end sin_330_eq_neg_one_half_l3_3908


namespace bee_fraction_remaining_l3_3212

theorem bee_fraction_remaining (N : ℕ) (L : ℕ) (D : ℕ) (hN : N = 80000) (hL : L = 1200) (hD : D = 50) :
  (N - (L * D)) / N = 1 / 4 :=
by
  sorry

end bee_fraction_remaining_l3_3212


namespace gcf_7fact_8fact_l3_3411

-- Definitions based on the conditions
def factorial (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | (n + 1) => (n + 1) * factorial n

noncomputable def greatest_common_divisor (a b : ℕ) : ℕ :=
  Nat.gcd a b

-- Theorem statement
theorem gcf_7fact_8fact : greatest_common_divisor (factorial 7) (factorial 8) = 5040 := by
  sorry

end gcf_7fact_8fact_l3_3411


namespace total_goals_other_members_l3_3576

theorem total_goals_other_members (x y : ℕ) (h1 : y = (7 * x) / 15 - 18)
  (h2 : 1 / 3 * x + 1 / 5 * x + 18 + y = x)
  (h3 : ∀ n, 0 ≤ n ∧ n ≤ 3 → ¬(n * 8 > y))
  : y = 24 :=
by
  sorry

end total_goals_other_members_l3_3576


namespace original_children_count_l3_3305

theorem original_children_count (x : ℕ) (h1 : 46800 / x + 1950 = 46800 / (x - 2))
    : x = 8 :=
sorry

end original_children_count_l3_3305


namespace sequence_term_2012_l3_3542

theorem sequence_term_2012 :
  ∃ (a : ℕ → ℤ), a 1 = 3 ∧ a 2 = 6 ∧ (∀ n, a (n + 2) = a (n + 1) - a n) ∧ a 2012 = 6 :=
sorry

end sequence_term_2012_l3_3542


namespace solve_inequality_system_l3_3166

theorem solve_inequality_system (x : ℝ) : 
  (5 * x - 1 > 3 * (x + 1)) →
  ((1 / 2) * x - 1 ≤ 7 - (3 / 2) * x) →
  (2 < x ∧ x ≤ 4) :=
by
  intro h1 h2
  sorry

end solve_inequality_system_l3_3166


namespace Henry_has_four_Skittles_l3_3821

-- Defining the initial amount of Skittles Bridget has
def Bridget_initial := 4

-- Defining the final amount of Skittles Bridget has after receiving all of Henry's Skittles
def Bridget_final := 8

-- Defining the amount of Skittles Henry has
def Henry_Skittles := Bridget_final - Bridget_initial

-- The proof statement to be proven
theorem Henry_has_four_Skittles : Henry_Skittles = 4 := by
  sorry

end Henry_has_four_Skittles_l3_3821


namespace sandcastle_height_difference_l3_3818

theorem sandcastle_height_difference :
  let Miki_height := 0.8333333333333334
  let Sister_height := 0.5
  Miki_height - Sister_height = 0.3333333333333334 :=
by
  sorry

end sandcastle_height_difference_l3_3818


namespace total_snowfall_l3_3275

theorem total_snowfall (morning_snowfall : ℝ) (afternoon_snowfall : ℝ) (h_morning : morning_snowfall = 0.125) (h_afternoon : afternoon_snowfall = 0.5) :
  morning_snowfall + afternoon_snowfall = 0.625 :=
by 
  sorry

end total_snowfall_l3_3275


namespace daves_apps_count_l3_3691

theorem daves_apps_count (x : ℕ) : 
  let initial_apps : ℕ := 21
  let added_apps : ℕ := 89
  let total_apps : ℕ := initial_apps + added_apps
  let deleted_apps : ℕ := x
  let more_added_apps : ℕ := x + 3
  total_apps - deleted_apps + more_added_apps = 113 :=
by
  sorry

end daves_apps_count_l3_3691


namespace mixed_groups_count_l3_3011

theorem mixed_groups_count (total_children : ℕ) (total_groups : ℕ)
  (group_size : ℕ) (total_boy_boy_photos : ℕ)
  (total_girl_girl_photos : ℕ)
  (total_photos : ℕ)
  (each_group_photos : ℕ)
  (total_mixed_photos : ℕ)
  (mixed_group_count : ℕ):
  total_children = 300 ∧
  total_groups = 100 ∧
  group_size = 3 ∧
  total_boy_boy_photos = 100 ∧
  total_girl_girl_photos = 56 ∧
  each_group_photos = 3 ∧
  total_photos = 100 * each_group_photos ∧
  total_mixed_photos = total_photos - total_boy_boy_photos - total_girl_girl_photos ∧
  mixed_group_count = total_mixed_photos / 2 →
  mixed_group_count = 72 :=
by
  sorry

end mixed_groups_count_l3_3011


namespace solve_inequality_system_l3_3167

theorem solve_inequality_system (x : ℝ) : 
  (5 * x - 1 > 3 * (x + 1)) →
  ((1 / 2) * x - 1 ≤ 7 - (3 / 2) * x) →
  (2 < x ∧ x ≤ 4) :=
by
  intro h1 h2
  sorry

end solve_inequality_system_l3_3167


namespace initial_price_of_sugar_per_kg_l3_3006

theorem initial_price_of_sugar_per_kg
  (initial_price : ℝ)
  (final_price : ℝ)
  (required_reduction : ℝ)
  (initial_price_eq : initial_price = 6)
  (final_price_eq : final_price = 7.5)
  (required_reduction_eq : required_reduction = 0.19999999999999996) :
  initial_price = 6 :=
by
  sorry

end initial_price_of_sugar_per_kg_l3_3006


namespace mixed_groups_count_l3_3022

-- Defining the conditions
def total_children : ℕ := 300
def groups_count : ℕ := 100
def group_size : ℕ := 3
def photographs_per_group : ℕ := group_size
def total_photographs : ℕ := groups_count * photographs_per_group
def boy_boy_photos : ℕ := 100
def girl_girl_photos : ℕ := 56
def mixed_photos : ℕ := total_photographs - boy_boy_photos - girl_girl_photos
def mixed_groups : ℕ := mixed_photos / 2

theorem mixed_groups_count : mixed_groups = 72 := by
  -- skipping the proof
  sorry

end mixed_groups_count_l3_3022


namespace smallest_N_sum_of_digits_eq_six_l3_3819

def bernardo_wins (N : ℕ) : Prop :=
  let b1 := 3 * N
  let s1 := b1 - 30
  let b2 := 3 * s1
  let s2 := b2 - 30
  let b3 := 3 * s2
  let s3 := b3 - 30
  let b4 := 3 * s3
  b4 < 800

def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n
  else sum_of_digits (n / 10) + (n % 10)

theorem smallest_N_sum_of_digits_eq_six :
  ∃ N : ℕ, bernardo_wins N ∧ sum_of_digits N = 6 :=
by
  sorry

end smallest_N_sum_of_digits_eq_six_l3_3819


namespace incorrect_transformation_l3_3486

-- Definitions based on conditions
variable (a b c : ℝ)

-- Conditions
axiom eq_add_six (h : a = b) : a + 6 = b + 6
axiom eq_div_nine (h : a = b) : a / 9 = b / 9
axiom eq_mul_c (h : a / c = b / c) (hc : c ≠ 0) : a = b
axiom eq_div_neg_two (h : -2 * a = -2 * b) : a = b

-- Proving the incorrect transformation statement
theorem incorrect_transformation : ¬ (a = -b) ∧ (-2 * a = -2 * b → a = b) := by
  sorry

end incorrect_transformation_l3_3486


namespace ferris_wheel_height_expression_best_visual_effect_time_l3_3307

noncomputable def ferris_wheel_height (t : ℝ) : ℝ :=
  -50 * Real.cos ((2 * Real.pi / 3) * t) + 50

theorem ferris_wheel_height_expression :
  ∀ t : ℝ, ferris_wheel_height t = -50 * Real.cos ((2 * Real.pi / 3) * t) + 50 :=
by intro t; rfl

theorem best_visual_effect_time :
  t = 3 - (3 / Real.pi) * Real.arccos (-7/10) :=
sorry

end ferris_wheel_height_expression_best_visual_effect_time_l3_3307


namespace lemon_heads_distribution_l3_3078

-- Conditions
def total_lemon_heads := 72
def number_of_friends := 6

-- Desired answer
def lemon_heads_per_friend := 12

-- Lean 4 statement
theorem lemon_heads_distribution : total_lemon_heads / number_of_friends = lemon_heads_per_friend := by 
  sorry

end lemon_heads_distribution_l3_3078


namespace billy_restaurant_total_payment_l3_3375

noncomputable def cost_of_meal
  (adult_count child_count : ℕ)
  (adult_cost child_cost : ℕ) : ℕ :=
  adult_count * adult_cost + child_count * child_cost

noncomputable def cost_of_dessert
  (total_people : ℕ)
  (dessert_cost : ℕ) : ℕ :=
  total_people * dessert_cost

noncomputable def total_cost_before_discount
  (adult_count child_count : ℕ)
  (adult_cost child_cost dessert_cost : ℕ) : ℕ :=
  (cost_of_meal adult_count child_count adult_cost child_cost) +
  (cost_of_dessert (adult_count + child_count) dessert_cost)

noncomputable def discount_amount
  (total : ℕ)
  (discount_rate : ℝ) : ℝ :=
  total * discount_rate

noncomputable def total_amount_to_pay
  (total : ℕ)
  (discount : ℝ) : ℝ :=
  total - discount

theorem billy_restaurant_total_payment :
  total_amount_to_pay
  (total_cost_before_discount 2 5 7 3 2)
  (discount_amount (total_cost_before_discount 2 5 7 3 2) 0.15) = 36.55 := by
  sorry

end billy_restaurant_total_payment_l3_3375


namespace exists_airline_route_within_same_republic_l3_3442

-- Define the concept of a country with cities, republics, and airline routes
def City : Type := ℕ
def Republic : Type := ℕ
def country : Set City := {n | n < 100}
noncomputable def cities_in_republic (R : Set City) : Prop :=
  ∃ x : City, x ∈ R

-- Conditions
def connected_by_route (c1 c2 : City) : Prop := sorry -- Placeholder for being connected by a route
def is_millionaire_city (c : City) : Prop := ∃ (routes : Set City), routes.card ≥ 70 ∧ ∀ r ∈ routes, connected_by_route c r

-- Theorem to be proved
theorem exists_airline_route_within_same_republic :
  country.card = 100 →
  ∃ republics : Set (Set City), republics.card = 3 ∧
    (∀ R ∈ republics, R.nonempty ∧ R.card ≤ 30) →
  ∃ millionaire_cities : Set City, millionaire_cities.card ≥ 70 ∧
    (∀ m ∈ millionaire_cities, is_millionaire_city m) →
  ∃ c1 c2 : City, ∃ R : Set City, R ∈ republics ∧ c1 ∈ R ∧ c2 ∈ R ∧ connected_by_route c1 c2 :=
begin
  -- Proof outline
  exact sorry
end

end exists_airline_route_within_same_republic_l3_3442


namespace angle_measure_l3_3641

theorem angle_measure (x : ℝ) :
  (180 - x) = 7 * (90 - x) → 
  x = 75 :=
by
  intro h
  sorry

end angle_measure_l3_3641


namespace total_lemons_produced_l3_3663

def normal_tree_lemons_per_year := 60
def production_factor := 1.5
def grove_rows := 50
def grove_cols := 30
def years := 5

theorem total_lemons_produced :
  let engineered_tree_lemons_per_year := normal_tree_lemons_per_year * production_factor in
  let total_trees := grove_rows * grove_cols in
  let lemons_per_year := total_trees * engineered_tree_lemons_per_year in
  let total_lemons := lemons_per_year * years in
  total_lemons = 675000 :=
by
  sorry

end total_lemons_produced_l3_3663


namespace smallest_digit_d_l3_3532

theorem smallest_digit_d (d : ℕ) (hd : d < 10) :
  (∃ d, (20 - (8 + d)) % 11 = 0 ∧ d < 10) → d = 1 :=
by
  sorry

end smallest_digit_d_l3_3532


namespace ordered_triple_solution_l3_3284

theorem ordered_triple_solution (a b c : ℝ) (h1 : a > 5) (h2 : b > 5) (h3 : c > 5)
  (h4 : (a + 3) * (a + 3) / (b + c - 5) + (b + 5) * (b + 5) / (c + a - 7) + (c + 7) * (c + 7) / (a + b - 9) = 49) :
  (a, b, c) = (13, 9, 6) :=
sorry

end ordered_triple_solution_l3_3284


namespace sum_binomial_coefficients_l3_3621

theorem sum_binomial_coefficients :
  let a := 1
  let b := 1
  let binomial := (2 * a + 2 * b)
  (binomial)^7 = 16384 := by
  -- Proof omitted
  sorry

end sum_binomial_coefficients_l3_3621


namespace cube_volume_and_diagonal_from_surface_area_l3_3065

theorem cube_volume_and_diagonal_from_surface_area
    (A : ℝ) (h : A = 150) :
    ∃ (V : ℝ) (d : ℝ), V = 125 ∧ d = 5 * Real.sqrt 3 :=
by
  sorry

end cube_volume_and_diagonal_from_surface_area_l3_3065


namespace correct_conclusion_l3_3226

theorem correct_conclusion (x : ℝ) (hx : x > 1/2) : -2 * x + 1 < 0 :=
by
  -- sorry placeholder
  sorry

end correct_conclusion_l3_3226


namespace total_balloons_l3_3452

-- Define the conditions
def joan_balloons : ℕ := 9
def sally_balloons : ℕ := 5
def jessica_balloons : ℕ := 2

-- The statement we want to prove
theorem total_balloons : joan_balloons + sally_balloons + jessica_balloons = 16 :=
by
  sorry

end total_balloons_l3_3452


namespace x_plus_y_bound_l3_3156

-- Definitions based on given conditions
variable {x y : ℝ}
variable [x_floor_equality : y = 4 * (⌊x⌋ : ℝ) + 1]
variable [x_minus_1_floor_equality : y = 2 * (⌊x - 1⌋ : ℝ) + 7]
variable [x_not_int : x ∉ ℤ]

-- Problem Statement
theorem x_plus_y_bound : 11 < x + y ∧ x + y < 12 := 
begin
  sorry
end

end x_plus_y_bound_l3_3156


namespace problem_statement_l3_3258

open Complex

theorem problem_statement (a : ℝ) : (∃ x : ℝ, (1 + a * Complex.i) / (2 + Complex.i) = x) → a = 1 / 2 :=
by
  sorry

end problem_statement_l3_3258


namespace notebook_cost_proof_l3_3215

-- Let n be the cost of the notebook and p be the cost of the pen.
variable (n p : ℝ)

-- Conditions:
def total_cost : Prop := n + p = 2.50
def notebook_more_pen : Prop := n = 2 + p

-- Theorem: Prove that the cost of the notebook is $2.25
theorem notebook_cost_proof (h1 : total_cost n p) (h2 : notebook_more_pen n p) : n = 2.25 := 
by 
  sorry

end notebook_cost_proof_l3_3215


namespace evaluate_expression_l3_3697

variable (x y : ℝ)

theorem evaluate_expression
  (hx : x ≠ 0)
  (hy : y ≠ 0)
  (hsum_sq : x^2 + y^2 ≠ 0)
  (hsum : x + y ≠ 0) :
    (x^2 + y^2)⁻¹ * ((x + y)⁻¹ + (x / y)⁻¹) = (1 + y) / ((x^2 + y^2) * (x + y)) :=
sorry

end evaluate_expression_l3_3697


namespace mean_score_of_students_who_failed_l3_3613

noncomputable def mean_failed_score : ℝ := sorry

theorem mean_score_of_students_who_failed (t p proportion_passed proportion_failed : ℝ) (h1 : t = 6) (h2 : p = 8) (h3 : proportion_passed = 0.6) (h4 : proportion_failed = 0.4) : mean_failed_score = 3 :=
by
  sorry

end mean_score_of_students_who_failed_l3_3613


namespace print_time_l3_3220

/-- Define the number of pages per minute printed by the printer -/
def pages_per_minute : ℕ := 25

/-- Define the total number of pages to be printed -/
def total_pages : ℕ := 350

/-- Prove that the time to print 350 pages at a rate of 25 pages per minute is 14 minutes -/
theorem print_time :
  (total_pages / pages_per_minute) = 14 :=
by
  sorry

end print_time_l3_3220


namespace film_cost_eq_five_l3_3580

variable (F : ℕ)

theorem film_cost_eq_five (H1 : 9 * F + 4 * 4 + 6 * 3 = 79) : F = 5 :=
by
  -- This is a placeholder for your proof
  sorry

end film_cost_eq_five_l3_3580


namespace sin_330_eq_neg_sin_30_l3_3978

theorem sin_330_eq_neg_sin_30 :
  sin (330 : ℝ) = - sin (30 : ℝ) := sorry

end sin_330_eq_neg_sin_30_l3_3978


namespace tangent_line_at_point_is_correct_l3_3313

theorem tangent_line_at_point_is_correct :
  ∀ (x : ℝ), (x • exp x + 2 * x + 1) for given x = 0 := 
  ∃ (f : ℝ → ℝ), (∀ (x : ℝ),
  f x = x * exp x + 2 * x + 1 ∧
  (∀ (p : ℝ × ℝ), p = (0, 1) →
    (∃ (m : ℝ), m = (derivative f) 0 ∧
    (∃ (b : ℝ), ∀ (y : ℝ), y = m * x + b ∧
    b = 1 ∧ m = 3))))

sorry

end tangent_line_at_point_is_correct_l3_3313


namespace fraction_product_simplified_l3_3686

theorem fraction_product_simplified:
  (2 / 3) * (4 / 7) * (9 / 11) = 24 / 77 := by
  sorry

end fraction_product_simplified_l3_3686


namespace candy_lasts_for_days_l3_3237

-- Definitions based on conditions
def candy_from_neighbors : ℕ := 75
def candy_from_sister : ℕ := 130
def candy_traded : ℕ := 25
def candy_lost : ℕ := 15
def candy_eaten_per_day : ℕ := 7

-- Total candy calculation
def total_candy : ℕ := candy_from_neighbors + candy_from_sister - candy_traded - candy_lost
def days_candy_lasts : ℕ := total_candy / candy_eaten_per_day

-- Proof statement
theorem candy_lasts_for_days : days_candy_lasts = 23 := by
  -- sorry is used to skip the actual proof
  sorry

end candy_lasts_for_days_l3_3237


namespace limit_of_function_l3_3690

-- Define the function f
def f (x : ℝ) : ℝ := (x - 2 * Real.pi) ^ 2 / Real.tan (Real.cos x - 1)

-- Define the limit point "2 * Real.pi"
def limit_point : ℝ := 2 * Real.pi

-- Statement of the limit problem
theorem limit_of_function : Filter.Tendsto f (nhds limit_point) (nhds (-2)) := by
  sorry

end limit_of_function_l3_3690


namespace mid_segment_PQ_l3_3143

variables {A B C A' B' P Q D : EuclideanSpace ℝ}
variables {circumcircle : Circle (triangle ABC)}
variables {AA' BB' BD AD : Line}

-- Condition: Triangle ABC is acute
def acute_triangle (A B C : EuclideanSpace ℝ) : Prop := ∀ angle (ABC), angle < π / 2

-- Condition: AA' and BB' are altitudes
def is_altitude (A' : EuclideanSpace ℝ) (A B C : EuclideanSpace ℝ) : Prop := 
  ∀ (line_from_A' : Line), perpendicular line_from_A' (Line A C)

-- Condition: D is on arc ACB of the circumcircle
def arc_ACB (D : EuclideanSpace ℝ) (circumcircle : Circle (triangle ABC)) : Prop := 
  on_circle D circumcircle ∧ between A C B D

-- Condition: Line AA' intersects BD at P, Line BB' intersects AD at Q
def intersects (l1 l2 : Line) : EuclideanSpace ℝ := ∃ P, lies_on P l1 ∧ lies_on P l2

theorem mid_segment_PQ (ABC : {ABC : EuclideanSpace ℝ // acute_triangle A B C}) 
  (A' B' P Q : EuclideanSpace ℝ) (AA' BB' BD AD : Line)
  (Ha : is_altitude A' A B C) (Hb : is_altitude B' A B C) (Hc : arc_ACB D circumcircle)
  (Hintersect1 : intersects AA' BD = P) (Hintersect2 : intersects BB' AD = Q) : 
  passes_through (line A' B') (midpoint P Q) :=
sorry

end mid_segment_PQ_l3_3143


namespace percentage_chain_l3_3130

theorem percentage_chain (n : ℝ) (h : n = 6000) : 0.1 * (0.3 * (0.5 * n)) = 90 := by
  sorry

end percentage_chain_l3_3130


namespace distance_from_point_to_asymptote_l3_3000

theorem distance_from_point_to_asymptote :
  ∃ (d : ℝ), ∀ (x₀ y₀ : ℝ), (x₀, y₀) = (3, 0) ∧ 3 * x₀ - 4 * y₀ = 0 →
  d = 9 / 5 :=
by
  sorry

end distance_from_point_to_asymptote_l3_3000


namespace mice_needed_l3_3499

-- Definitions for relative strength in terms of M (Mouse strength)
def C (M : ℕ) : ℕ := 6 * M
def J (M : ℕ) : ℕ := 5 * C M
def G (M : ℕ) : ℕ := 4 * J M
def B (M : ℕ) : ℕ := 3 * G M
def D (M : ℕ) : ℕ := 2 * B M

-- Condition: all together can pull up the Turnip with strength 1237M
def total_strength_with_mouse (M : ℕ) : ℕ :=
  D M + B M + G M + J M + C M + M

-- Condition: without the Mouse, they cannot pull up the Turnip
def total_strength_without_mouse (M : ℕ) : ℕ :=
  D M + B M + G M + J M + C M

theorem mice_needed (M : ℕ) (h : total_strength_with_mouse M = 1237 * M) (h2 : total_strength_without_mouse M < 1237 * M) :
  1237 = 1237 :=
by
  -- using sorry to indicate proof is not provided
  sorry

end mice_needed_l3_3499


namespace mixed_groups_count_l3_3012

theorem mixed_groups_count (total_children : ℕ) (total_groups : ℕ)
  (group_size : ℕ) (total_boy_boy_photos : ℕ)
  (total_girl_girl_photos : ℕ)
  (total_photos : ℕ)
  (each_group_photos : ℕ)
  (total_mixed_photos : ℕ)
  (mixed_group_count : ℕ):
  total_children = 300 ∧
  total_groups = 100 ∧
  group_size = 3 ∧
  total_boy_boy_photos = 100 ∧
  total_girl_girl_photos = 56 ∧
  each_group_photos = 3 ∧
  total_photos = 100 * each_group_photos ∧
  total_mixed_photos = total_photos - total_boy_boy_photos - total_girl_girl_photos ∧
  mixed_group_count = total_mixed_photos / 2 →
  mixed_group_count = 72 :=
by
  sorry

end mixed_groups_count_l3_3012


namespace base7_to_base10_div_l3_3604

theorem base7_to_base10_div (x y : ℕ) (h : 546 = x * 10^2 + y * 10 + 9) : (x + y + 9) / 21 = 6 / 7 :=
by {
  sorry
}

end base7_to_base10_div_l3_3604


namespace calculate_y_l3_3368

theorem calculate_y (x : ℤ) (y : ℤ) (h1 : x = 121) (h2 : 2 * x - y = 102) : y = 140 :=
by
  -- Placeholder proof
  sorry

end calculate_y_l3_3368


namespace max_xy_min_function_l3_3653

-- Problem 1: Prove that the maximum value of xy is 8 given the conditions
theorem max_xy (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : x + 2 * y = 8) : xy ≤ 8 :=
sorry

-- Problem 2: Prove that the minimum value of the function is 9 given the conditions
theorem min_function (x : ℝ) (hx : -1 < x) : (x + 4 / (x + 1) + 6) ≥ 9 :=
sorry

end max_xy_min_function_l3_3653


namespace cistern_wet_surface_area_l3_3050

noncomputable def total_wet_surface_area (length : ℝ) (width : ℝ) (depth : ℝ) : ℝ :=
  let bottom_surface_area := length * width
  let longer_side_area := 2 * (depth * length)
  let shorter_side_area := 2 * (depth * width)
  bottom_surface_area + longer_side_area + shorter_side_area

theorem cistern_wet_surface_area :
  total_wet_surface_area 9 4 1.25 = 68.5 :=
by
  sorry

end cistern_wet_surface_area_l3_3050


namespace smallest_nat_number_l3_3214

theorem smallest_nat_number (x : ℕ) 
  (h1 : ∃ z : ℕ, x + 3 = 5 * z) 
  (h2 : ∃ n : ℕ, x - 3 = 6 * n) : x = 27 := 
sorry

end smallest_nat_number_l3_3214


namespace Minjeong_family_juice_consumption_l3_3750

theorem Minjeong_family_juice_consumption :
  (∀ (amount_per_time : ℝ) (times_per_day : ℕ) (days_per_week : ℕ),
  amount_per_time = 0.2 → times_per_day = 3 → days_per_week = 7 → 
  amount_per_time * times_per_day * days_per_week = 4.2) :=
by
  intros amount_per_time times_per_day days_per_week h1 h2 h3
  rw [h1, h2, h3]
  norm_num

end Minjeong_family_juice_consumption_l3_3750


namespace distance_between_foci_of_ellipse_l3_3674

theorem distance_between_foci_of_ellipse :
  let center : (ℝ × ℝ) := (8, 2)
  let a : ℝ := 16 / 2 -- half the length of the major axis
  let b : ℝ := 4 / 2  -- half the length of the minor axis
  let c : ℝ := Real.sqrt (a^2 - b^2) -- distance from the center to each focus
  2 * c = 4 * Real.sqrt 15 :=
by
  let center : (ℝ × ℝ) := (8, 2)
  let a : ℝ := 16 / 2 -- half the length of the major axis
  let b : ℝ := 4 / 2  -- half the length of the minor axis
  let c : ℝ := Real.sqrt (a^2 - b^2) -- distance from the center to each focus
  show 2 * c = 4 * Real.sqrt 15
  sorry

end distance_between_foci_of_ellipse_l3_3674


namespace evaluate_81_power_5_div_4_l3_3097

-- Define the conditions
def base_factorized : ℕ := 3 ^ 4
def power_rule (b : ℕ) (m n : ℝ) : ℝ := (b : ℝ) ^ m ^ n

-- Define the primary calculation
noncomputable def power_calculation : ℝ := 81 ^ (5 / 4)

-- Prove that the calculation equals 243
theorem evaluate_81_power_5_div_4 : power_calculation = 243 := 
by
  have h1 : base_factorized = 81 := by sorry
  have h2 : power_rule 3 4 (5 / 4) = 3 ^ 5 := by sorry
  have h3 : 3 ^ 5 = 243 := by sorry
  have h4 : power_calculation = power_rule 3 4 (5 / 4) := by sorry
  rw [h1, h2, h3, h4]
  exact h3

end evaluate_81_power_5_div_4_l3_3097


namespace total_number_of_cantelopes_l3_3706

def number_of_cantelopes_fred : ℕ := 38
def number_of_cantelopes_tim : ℕ := 44

theorem total_number_of_cantelopes : number_of_cantelopes_fred + number_of_cantelopes_tim = 82 := by
  sorry

end total_number_of_cantelopes_l3_3706


namespace total_pages_of_book_l3_3474

theorem total_pages_of_book (P : ℝ) (h : 0.4 * P = 16) : P = 40 :=
sorry

end total_pages_of_book_l3_3474


namespace cameron_books_ratio_l3_3377

theorem cameron_books_ratio (Boris_books : ℕ) (Cameron_books : ℕ)
  (Boris_after_donation : ℕ) (Cameron_after_donation : ℕ)
  (total_books_after_donation : ℕ) (ratio : ℚ) :
  Boris_books = 24 → 
  Cameron_books = 30 → 
  Boris_after_donation = Boris_books - (Boris_books / 4) →
  total_books_after_donation = 38 →
  Cameron_after_donation = total_books_after_donation - Boris_after_donation →
  ratio = (Cameron_books - Cameron_after_donation) / Cameron_books →
  ratio = 1 / 3 :=
by
  -- Proof goes here.
  sorry

end cameron_books_ratio_l3_3377


namespace sin_330_eq_neg_half_l3_3938

noncomputable def Q : ℝ × ℝ := (Real.sqrt 3 / 2, -1 / 2)

theorem sin_330_eq_neg_half : Real.sin (330 * Real.pi / 180) = -1 / 2 :=
by
  have h1 : 330 * Real.pi / 180 = 11 * Real.pi / 6 := by sorry
  have coord_y : (Q.snd) = -1 / 2 := by sorry
  rw [h1, coord_y]
  sorry

end sin_330_eq_neg_half_l3_3938


namespace sin_330_eq_neg_half_l3_3943

noncomputable def Q : ℝ × ℝ := (Real.sqrt 3 / 2, -1 / 2)

theorem sin_330_eq_neg_half : Real.sin (330 * Real.pi / 180) = -1 / 2 :=
by
  have h1 : 330 * Real.pi / 180 = 11 * Real.pi / 6 := by sorry
  have coord_y : (Q.snd) = -1 / 2 := by sorry
  rw [h1, coord_y]
  sorry

end sin_330_eq_neg_half_l3_3943


namespace triangle_formation_ways_l3_3632

-- Given conditions
def parallel_tracks : Prop := true -- The tracks are parallel, implicit condition not affecting calculation
def first_track_checkpoints := 6
def second_track_checkpoints := 10

-- The proof problem
theorem triangle_formation_ways : 
  (first_track_checkpoints * Nat.choose second_track_checkpoints 2) = 270 := by
  sorry

end triangle_formation_ways_l3_3632


namespace sum_f_to_2017_l3_3722

noncomputable def f (x : ℕ) : ℝ := Real.cos (x * Real.pi / 3)

theorem sum_f_to_2017 : (Finset.range 2017).sum f = 1 / 2 :=
by
  sorry

end sum_f_to_2017_l3_3722


namespace sin_330_l3_3898

theorem sin_330 : Real.sin (330 * Real.pi / 180) = -1 / 2 := 
by
  -- Outline the proof here without providing it
  -- sorry to delay the proof
  sorry

end sin_330_l3_3898


namespace church_path_count_is_321_l3_3811

/-- A person starts at the bottom-left corner of an m x n grid and can only move north, east, or 
    northeast. Prove that the number of distinct paths to the top-right corner is 321 
    for a specific grid size (abstracted parameters included). -/
def distinct_paths_to_church (m n : ℕ) : ℕ :=
  let rec P : ℕ → ℕ → ℕ
    | 0, 0 => 1
    | i + 1, 0 => 1
    | 0, j + 1 => 1
    | i + 1, j + 1 => P i (j + 1) + P (i + 1) j + P i j
  P m n

theorem church_path_count_is_321 : distinct_paths_to_church m n = 321 :=
sorry

end church_path_count_is_321_l3_3811


namespace gcf_7_8_fact_l3_3409

noncomputable def factorial : ℕ → ℕ
| 0     := 1
| (n+1) := (n+1) * factorial n

theorem gcf_7_8_fact : Nat.gcd (factorial 7) (factorial 8) = 5040 := by
  sorry

end gcf_7_8_fact_l3_3409


namespace horizontal_asymptote_l3_3732

def numerator (x : ℝ) : ℝ :=
  15 * x^4 + 3 * x^3 + 7 * x^2 + 6 * x + 2

def denominator (x : ℝ) : ℝ :=
  5 * x^4 + x^3 + 4 * x^2 + 2 * x + 1

noncomputable def rational_function (x : ℝ) : ℝ :=
  numerator x / denominator x

theorem horizontal_asymptote :
  ∃ y : ℝ, (∀ x : ℝ, x ≠ 0 → rational_function x = y) ↔ y = 3 :=
by
  sorry

end horizontal_asymptote_l3_3732


namespace sqrt_144000_simplified_l3_3598

theorem sqrt_144000_simplified : Real.sqrt 144000 = 120 * Real.sqrt 10 := by
  sorry

end sqrt_144000_simplified_l3_3598


namespace gcf_factorial_seven_eight_l3_3401

theorem gcf_factorial_seven_eight (a b : ℕ) (h : a = 7! ∧ b = 8!) : Nat.gcd a b = 7! := 
by 
  sorry

end gcf_factorial_seven_eight_l3_3401


namespace major_premise_incorrect_l3_3555

theorem major_premise_incorrect (a b : ℝ) (h : a > b) : ¬ (a^2 > b^2) :=
by {
  sorry
}

end major_premise_incorrect_l3_3555


namespace sum_midpoint_x_coords_l3_3323

theorem sum_midpoint_x_coords (a b c : ℝ) (h1 : a + b + c = 15) (h2 : a - b = 3) :
    (a + (a - 3)) / 2 + (a + c) / 2 + ((a - 3) + c) / 2 = 15 := 
by 
  sorry

end sum_midpoint_x_coords_l3_3323


namespace sin_330_eq_neg_half_l3_3923

open Real

theorem sin_330_eq_neg_half : sin (330 * (π / 180)) = - (1 / 2) := by
  have angle_eq : 330 * (π / 180) = 2 * π - (30 * (π / 180)) := by sorry
  have sin_30_pos : sin (30 * (π / 180)) = 1 / 2 := by sorry
  have sin_330_neg : sin (2 * π - (30 * (π / 180))) = - (sin (30 * (π / 180))) := by sorry
  rw [angle_eq, sin_330_neg, sin_30_pos]
  rfl

end sin_330_eq_neg_half_l3_3923


namespace sin_330_eq_neg_half_l3_3831

theorem sin_330_eq_neg_half : sin (330 * (π / 180)) = -1 / 2 := 
by
  -- condition: \(330^\circ = 360^\circ - 30^\circ\)
  -- condition: \(\sin 30^\circ = \frac{1}{2}\)
  -- condition: sine is negative in the fourth quadrant
  sorry

end sin_330_eq_neg_half_l3_3831


namespace total_students_l3_3438

variables (F G B N : ℕ)
variables (hF : F = 41) (hG : G = 22) (hB : B = 9) (hN : N = 6)

theorem total_students (F G B N : ℕ) (hF : F = 41) (hG : G = 22) (hB : B = 9) (hN : N = 6) : 
  F + G - B + N = 60 := by
sorry

end total_students_l3_3438


namespace distance_to_hole_l3_3371

-- Define the variables from the problem
variables (distance_first_turn distance_second_turn beyond_hole total_distance hole_distance : ℝ)

-- Given conditions
def conditions : Prop :=
  distance_first_turn = 180 ∧
  distance_second_turn = distance_first_turn / 2 ∧
  beyond_hole = 20 ∧
  total_distance = distance_first_turn + distance_second_turn

-- The main statement we need to prove
theorem distance_to_hole : conditions →
  hole_distance = total_distance - beyond_hole → hole_distance = 250 :=
by
  sorry

end distance_to_hole_l3_3371


namespace mixed_groups_count_l3_3032

-- Define the facts about the groups and photographs
def numberOfChildren : Nat := 300
def numberOfGroups : Nat := 100
def childrenPerGroup : Nat := 3
def b_b_photos : Nat := 100
def g_g_photos : Nat := 56

-- Define the function to calculate mixed groups
def mixedGroups (totalPhotos b_b_photos g_g_photos : Nat) : Nat := 
  (totalPhotos - b_b_photos - g_g_photos) / 2

-- State the theorem
theorem mixed_groups_count : 
  mixedGroups (numberOfGroups * childrenPerGroup) b_b_photos g_g_photos = 72 := by
  rfl

end mixed_groups_count_l3_3032


namespace sin_330_eq_neg_one_half_l3_3880

theorem sin_330_eq_neg_one_half : sin (330 * π / 180) = -1 / 2 := by
  sorry

end sin_330_eq_neg_one_half_l3_3880


namespace problem_statement_l3_3799

theorem problem_statement : 2^2 * 3^2 * 5^2 * 7 = 6300 := by
  sorry

end problem_statement_l3_3799


namespace line_through_point_equal_intercepts_l3_3777

theorem line_through_point_equal_intercepts (a b : ℝ) : 
  ((∃ (k : ℝ), k ≠ 0 ∧ (3 = 2 * k) ∧ b = k) ∨ ((a ≠ 0) ∧ (5/a = 1))) → 
  (a = 1 ∧ b = 1) ∨ (3 * a - 2 * b = 0) := 
by 
  sorry

end line_through_point_equal_intercepts_l3_3777


namespace residential_ratio_l3_3655

theorem residential_ratio (B R O E : ℕ) (h1 : B = 300) (h2 : E = 75) (h3 : E = O ∧ R + 2 * E = B) : R / B = 1 / 2 :=
by
  sorry

end residential_ratio_l3_3655


namespace sin_330_eq_neg_half_l3_3888

theorem sin_330_eq_neg_half : Real.sin (330 * Real.pi / 180) = -1 / 2 := by
  -- Definitions and conditions from the problem
  have h1 : 330 = 360 - 30 := rfl
  have h2 : ∀ θ, Real.sin θ = -Real.sin (θ - 2 * Real.pi) := sorry
  have h3 : Real.sin (30 * Real.pi / 180) = 1 / 2 := sorry

  -- Use the given conditions to prove the equality
  calc
    Real.sin (330 * Real.pi / 180)
        = Real.sin (-(30 * Real.pi / 180) + 2 * Real.pi) : by rw [←h1, Real.sin_angle_sub_pi]
    ... = -Real.sin (30 * Real.pi / 180) : by rw [h2]
    ... = -1 / 2 : by rw [h3]

end sin_330_eq_neg_half_l3_3888


namespace sin_330_l3_3926

theorem sin_330 : Real.sin (330 * Real.pi / 180) = -1 / 2 := 
by
  sorry

end sin_330_l3_3926


namespace negation_exists_to_forall_l3_3180

theorem negation_exists_to_forall (P : ℝ → Prop) (h : ∃ x : ℝ, x^2 + 3 * x + 2 < 0) :
  (¬ (∃ x : ℝ, x^2 + 3 * x + 2 < 0)) ↔ (∀ x : ℝ, x^2 + 3 * x + 2 ≥ 0) := by
sorry

end negation_exists_to_forall_l3_3180


namespace matrix_vector_addition_l3_3389

def A : Matrix (Fin 2) (Fin 2) ℤ := ![![4, -2], ![-5, 6]]
def v : Fin 2 → ℤ := ![5, -2]
def w : Fin 2 → ℤ := ![1, -1]

theorem matrix_vector_addition :
  (A.mulVec v + w) = ![25, -38] :=
by
  sorry

end matrix_vector_addition_l3_3389


namespace angle_measure_supplement_complement_l3_3635

theorem angle_measure_supplement_complement (x : ℝ) 
    (h1 : 180 - x = 7 * (90 - x)) : 
    x = 75 := by
  sorry

end angle_measure_supplement_complement_l3_3635


namespace sin_330_deg_l3_3851

theorem sin_330_deg : Real.sin (330 * Real.pi / 180) = -1 / 2 := by
  sorry

end sin_330_deg_l3_3851


namespace t_over_s_possible_values_l3_3460

-- Define the initial conditions
variables (n : ℕ) (h : n ≥ 3)

-- The theorem statement
theorem t_over_s_possible_values (s t : ℕ) (h_s : s > 0) (h_t : t > 0) : 
  (∃ r : ℚ, r = t / s ∧ 1 ≤ r ∧ r < (n - 1)) :=
sorry

end t_over_s_possible_values_l3_3460


namespace print_time_l3_3218

-- Conditions
def printer_pages_per_minute : ℕ := 25
def total_pages : ℕ := 350

-- Theorem
theorem print_time :
  (total_pages / printer_pages_per_minute : ℕ) = 14 :=
by sorry

end print_time_l3_3218


namespace intersection_M_N_l3_3126

def M : Set ℕ := {1, 3, 4}
def N : Set ℕ := {x | x^2 - 4 * x + 3 = 0}

theorem intersection_M_N : M ∩ N = {1, 3} :=
by sorry

end intersection_M_N_l3_3126


namespace total_number_of_cantelopes_l3_3707

def number_of_cantelopes_fred : ℕ := 38
def number_of_cantelopes_tim : ℕ := 44

theorem total_number_of_cantelopes : number_of_cantelopes_fred + number_of_cantelopes_tim = 82 := by
  sorry

end total_number_of_cantelopes_l3_3707


namespace sin_330_eq_neg_one_half_l3_3960

theorem sin_330_eq_neg_one_half : Real.sin (330 * Real.pi / 180) = -1 / 2 := by
  sorry

end sin_330_eq_neg_one_half_l3_3960


namespace sugar_percentage_is_7_5_l3_3489

theorem sugar_percentage_is_7_5 
  (V1 : ℕ := 340)
  (p_water : ℝ := 88/100)
  (p_kola : ℝ := 5/100)
  (p_sugar : ℝ := 7/100)
  (V_sugar_add : ℝ := 3.2)
  (V_water_add : ℝ := 10)
  (V_kola_add : ℝ := 6.8) : 
  (
    (23.8 + 3.2) / (340 + 3.2 + 10 + 6.8) * 100 = 7.5
  ) :=
  by
  sorry

end sugar_percentage_is_7_5_l3_3489


namespace prism_width_l3_3266

/-- A rectangular prism with dimensions l, w, h such that the diagonal length is 13
    and given l = 3 and h = 12, has width w = 4. -/
theorem prism_width (w : ℕ) 
  (h : ℕ) (l : ℕ) 
  (diag_len : ℕ) 
  (hl : l = 3) 
  (hh : h = 12) 
  (hd : diag_len = 13) 
  (h_diag : diag_len = Int.sqrt (l^2 + w^2 + h^2)) : 
  w = 4 := 
  sorry

end prism_width_l3_3266


namespace smallest_integer_divisible_20_perfect_cube_square_l3_3084

theorem smallest_integer_divisible_20_perfect_cube_square :
  ∃ (n : ℕ), n > 0 ∧ n % 20 = 0 ∧ (∃ (m : ℕ), n^2 = m^3) ∧ (∃ (k : ℕ), n^3 = k^2) ∧ n = 1000000 :=
by {
  sorry -- Replace this placeholder with an appropriate proof.
}

end smallest_integer_divisible_20_perfect_cube_square_l3_3084


namespace sin_330_eq_neg_one_half_l3_3913

theorem sin_330_eq_neg_one_half :
  ∃ θ : ℝ, sin (330 * real.pi / 180) = -1 / 2 :=
begin
  have h1 : sin (360 * real.pi / 180 - 30 * real.pi / 180) = -sin (30 * real.pi / 180),
  { rw sin_sub,
    ring,
    rw sin_pi_div_six, -- sin 30° = 1 / 2
  },
  have h2 : sin (30 * real.pi / 180) = 1 / 2 := by exact real.sin_pi_div_six,
  use 330,
  calc
    sin (330 * real.pi / 180)
        = sin ((360 - 30) * real.pi / 180)      : by ring
    ... = sin (360 * real.pi / 180 - 30 * real.pi / 180) : by congr' 1; ring
    ... = -sin (30 * real.pi / 180)            : by exact h1
    ... = -1 / 2                               : by rw h2,
end

end sin_330_eq_neg_one_half_l3_3913


namespace no_real_solution_range_of_a_l3_3652

theorem no_real_solution_range_of_a (a : ℝ) :
  (∀ x : ℝ, ¬(|x + 1| + |x - 2| < a)) → a ≤ 3 :=
by
  sorry  -- Proof skipped

end no_real_solution_range_of_a_l3_3652


namespace percentage_students_passed_is_35_l3_3277

/-
The problem is to prove the percentage of students who passed the examination, given that 520 out of 800 students failed, is 35%.
-/

def total_students : ℕ := 800
def failed_students : ℕ := 520
def passed_students : ℕ := total_students - failed_students

def percentage_passed : ℕ := (passed_students * 100) / total_students

theorem percentage_students_passed_is_35 : percentage_passed = 35 :=
by
  -- Here the proof will go.
  sorry

end percentage_students_passed_is_35_l3_3277


namespace sin_330_eq_neg_one_half_l3_3910

theorem sin_330_eq_neg_one_half :
  ∃ θ : ℝ, sin (330 * real.pi / 180) = -1 / 2 :=
begin
  have h1 : sin (360 * real.pi / 180 - 30 * real.pi / 180) = -sin (30 * real.pi / 180),
  { rw sin_sub,
    ring,
    rw sin_pi_div_six, -- sin 30° = 1 / 2
  },
  have h2 : sin (30 * real.pi / 180) = 1 / 2 := by exact real.sin_pi_div_six,
  use 330,
  calc
    sin (330 * real.pi / 180)
        = sin ((360 - 30) * real.pi / 180)      : by ring
    ... = sin (360 * real.pi / 180 - 30 * real.pi / 180) : by congr' 1; ring
    ... = -sin (30 * real.pi / 180)            : by exact h1
    ... = -1 / 2                               : by rw h2,
end

end sin_330_eq_neg_one_half_l3_3910


namespace sin_330_l3_3929

theorem sin_330 : Real.sin (330 * Real.pi / 180) = -1 / 2 := 
by
  sorry

end sin_330_l3_3929


namespace circumference_ratio_l3_3609

theorem circumference_ratio (C D : ℝ) (hC : C = 94.2) (hD : D = 30) : C / D = 3.14 :=
by {
  sorry
}

end circumference_ratio_l3_3609


namespace solve_rings_l3_3293

variable (B : ℝ) (S : ℝ)

def conditions := (S = (5/8) * (Real.sqrt B)) ∧ (S + B = 52)

theorem solve_rings : conditions B S → (S + B = 52) := by
  intros h
  sorry

end solve_rings_l3_3293


namespace largest_consecutive_integer_product_2520_l3_3398

theorem largest_consecutive_integer_product_2520 :
  ∃ (n : ℕ), n * (n + 1) * (n + 2) * (n + 3) = 2520 ∧ (n + 3) = 8 :=
by {
  sorry
}

end largest_consecutive_integer_product_2520_l3_3398


namespace twice_original_price_l3_3273

theorem twice_original_price (P : ℝ) (h : 377 = 1.30 * P) : 2 * P = 580 :=
by {
  -- proof steps will go here
  sorry
}

end twice_original_price_l3_3273


namespace division_quotient_less_dividend_l3_3801

theorem division_quotient_less_dividend
  (a1 : (6 : ℝ) > 0)
  (a2 : (5 / 7 : ℝ) > 0)
  (a3 : (3 / 8 : ℝ) > 0)
  (h1 : (3 / 5 : ℝ) < 1)
  (h2 : (5 / 4 : ℝ) > 1)
  (h3 : (5 / 12 : ℝ) < 1):
  (6 / (3 / 5) > 6) ∧ (5 / 7 / (5 / 4) < 5 / 7) ∧ (3 / 8 / (5 / 12) > 3 / 8) :=
by
  sorry

end division_quotient_less_dividend_l3_3801


namespace exists_route_within_same_republic_l3_3441

-- Conditions
def city := ℕ
def republic := ℕ
def airline_routes (c1 c2 : city) : Prop := sorry -- A predicate representing airline routes

constant n_cities : ℕ := 100
constant n_republics : ℕ := 3
constant cities_in_republic : city → republic
constant very_connected_city : city → Prop
axiom at_least_70_very_connected : ∃ S : set city, S.card ≥ 70 ∧ ∀ c ∈ S, (cardinal.mk {d : city | airline_routes c d}.to_finset) ≥ 70

-- Question
theorem exists_route_within_same_republic : ∃ c1 c2 : city, c1 ≠ c2 ∧ cities_in_republic c1 = cities_in_republic c2 ∧ airline_routes c1 c2 :=
sorry

end exists_route_within_same_republic_l3_3441


namespace prism_width_l3_3267

/-- A rectangular prism with dimensions l, w, h such that the diagonal length is 13
    and given l = 3 and h = 12, has width w = 4. -/
theorem prism_width (w : ℕ) 
  (h : ℕ) (l : ℕ) 
  (diag_len : ℕ) 
  (hl : l = 3) 
  (hh : h = 12) 
  (hd : diag_len = 13) 
  (h_diag : diag_len = Int.sqrt (l^2 + w^2 + h^2)) : 
  w = 4 := 
  sorry

end prism_width_l3_3267


namespace valentine_floral_requirement_l3_3147

theorem valentine_floral_requirement:
  let nursing_home_roses := 90
  let nursing_home_tulips := 80
  let nursing_home_lilies := 100
  let shelter_roses := 120
  let shelter_tulips := 75
  let shelter_lilies := 95
  let maternity_ward_roses := 100
  let maternity_ward_tulips := 110
  let maternity_ward_lilies := 85
  let total_roses := nursing_home_roses + shelter_roses + maternity_ward_roses
  let total_tulips := nursing_home_tulips + shelter_tulips + maternity_ward_tulips
  let total_lilies := nursing_home_lilies + shelter_lilies + maternity_ward_lilies
  let total_flowers := total_roses + total_tulips + total_lilies
  total_roses = 310 ∧
  total_tulips = 265 ∧
  total_lilies = 280 ∧
  total_flowers = 855 :=
by
  sorry

end valentine_floral_requirement_l3_3147


namespace original_number_of_men_l3_3346

-- Define the conditions
def work_days_by_men (M : ℕ) (days : ℕ) : ℕ := M * days
def additional_men (M : ℕ) : ℕ := M + 10
def completed_days : ℕ := 9

-- The main theorem
theorem original_number_of_men : ∀ (M : ℕ), 
  work_days_by_men M 12 = work_days_by_men (additional_men M) completed_days → 
  M = 30 :=
by
  intros M h
  sorry

end original_number_of_men_l3_3346


namespace mixed_groups_count_l3_3019

-- Defining the conditions
def total_children : ℕ := 300
def groups_count : ℕ := 100
def group_size : ℕ := 3
def photographs_per_group : ℕ := group_size
def total_photographs : ℕ := groups_count * photographs_per_group
def boy_boy_photos : ℕ := 100
def girl_girl_photos : ℕ := 56
def mixed_photos : ℕ := total_photographs - boy_boy_photos - girl_girl_photos
def mixed_groups : ℕ := mixed_photos / 2

theorem mixed_groups_count : mixed_groups = 72 := by
  -- skipping the proof
  sorry

end mixed_groups_count_l3_3019


namespace Eve_spend_l3_3393

noncomputable def hand_mitts := 14.00
noncomputable def apron := 16.00
noncomputable def utensils_set := 10.00
noncomputable def small_knife := 2 * utensils_set
noncomputable def total_cost_for_one_niece := hand_mitts + apron + utensils_set + small_knife
noncomputable def total_cost_for_three_nieces := 3 * total_cost_for_one_niece
noncomputable def discount := 0.25 * total_cost_for_three_nieces
noncomputable def final_cost := total_cost_for_three_nieces - discount

theorem Eve_spend : final_cost = 135.00 :=
by sorry

end Eve_spend_l3_3393


namespace sin_330_is_minus_sqrt3_over_2_l3_3987

-- Define the angle theta as 330 degrees
def theta := 330 * Real.toRad

-- State that point corresponding to 330 degrees lies in the fourth quadrant
def point_in_fourth_quadrant (θ : ℝ) := 330 * Real.toRad = θ ∧ θ > 270 * Real.toRad ∧ θ < 360 * Real.toRad

-- Define a proof problem that states computing sin of 330° equals -√3/2
theorem sin_330_is_minus_sqrt3_over_2 : point_in_fourth_quadrant theta → Real.sin theta = -Real.sqrt 3 / 2 := by
  sorry

end sin_330_is_minus_sqrt3_over_2_l3_3987


namespace sin_330_eq_neg_half_l3_3839

theorem sin_330_eq_neg_half 
  (h1: (330 : ℝ) = 330)
  (h2: (330 : ℝ = 360 - 30))
  (h3: 30 ∈ reference_angle_set ∧ sin 30 = 1/2)
  (h4: 330 ∈ fourth_quadrant)
  (h5: ∀ θ ∈ fourth_quadrant, sin θ < 0) :
  sin 330 = -1/2 := 
by
  sorry

end sin_330_eq_neg_half_l3_3839


namespace total_stickers_l3_3651

-- Definitions for the given conditions
def stickers_per_page : ℕ := 10
def number_of_pages : ℕ := 22

-- The theorem to be proven
theorem total_stickers : stickers_per_page * number_of_pages = 220 := by
  sorry

end total_stickers_l3_3651


namespace sin_330_deg_l3_3870

noncomputable theory

open Real

theorem sin_330_deg :
  sin (330 * (π / 180)) = -1 / 2 :=
by sorry

end sin_330_deg_l3_3870


namespace smallest_possible_value_of_M_l3_3457

theorem smallest_possible_value_of_M :
  ∀ (a b c d e f : ℕ), a > 0 → b > 0 → c > 0 → d > 0 → e > 0 → f > 0 →
  a + b + c + d + e + f = 4020 →
  (∃ M : ℕ, M = max (a + b) (max (b + c) (max (c + d) (max (d + e) (e + f)))) ∧
    (∀ (M' : ℕ), (∀ (a b c d e f : ℕ), a > 0 → b > 0 → c > 0 → d > 0 → e > 0 → f > 0 →
      a + b + c + d + e + f = 4020 →
      M' = max (a + b) (max (b + c) (max (c + d) (max (d + e) (e + f)))) → M' ≥ 804) → M = 804)) := by
  sorry

end smallest_possible_value_of_M_l3_3457


namespace parabola_directrix_distance_l3_3271

theorem parabola_directrix_distance (m : ℝ) (h : |1 / (4 * m)| = 2) : m = 1/8 ∨ m = -1/8 :=
by { sorry }

end parabola_directrix_distance_l3_3271


namespace angle_measure_l3_3640

theorem angle_measure (x : ℝ) :
  (180 - x) = 7 * (90 - x) → 
  x = 75 :=
by
  intro h
  sorry

end angle_measure_l3_3640


namespace union_of_A_and_B_l3_3538

variables (A B : Set ℤ)
variable (a : ℤ)
theorem union_of_A_and_B : (A = {4, a^2}) → (B = {a-6, 1+a, 9}) → (A ∩ B = {9}) → (A ∪ B = {-9, -2, 4, 9}) :=
by
  intros hA hB hInt
  sorry

end union_of_A_and_B_l3_3538


namespace proof_aim_l3_3725

variables (a : ℝ)

def p : Prop := ∀ x : ℝ, 1 ≤ x ∧ x ≤ 2 → x^2 - a ≥ 0
def q : Prop := ∃ x₀ : ℝ, x₀^2 + 2*a*x₀ + (2 - a) = 0

theorem proof_aim (hp : p a) (hq : q a) : a ≤ -2 ∨ a = 1 :=
sorry

end proof_aim_l3_3725


namespace value_of_x_l3_3175

theorem value_of_x (x : ℚ) (h : (x + 10 + 17 + 3 * x + 15 + 3 * x + 6) / 5 = 26) : x = 82 / 7 :=
by
  sorry

end value_of_x_l3_3175


namespace find_m_collinear_l3_3070

-- Definition of a point in 2D space
structure Point2D where
  x : ℤ
  y : ℤ

-- Predicate to check if three points are collinear 
def collinear_points (p1 p2 p3 : Point2D) : Prop :=
  (p3.x - p2.x) * (p2.y - p1.y) = (p2.x - p1.x) * (p3.y - p2.y)

-- Given points A, B, and C
def A : Point2D := ⟨2, 3⟩
def B (m : ℤ) : Point2D := ⟨-4, m⟩
def C : Point2D := ⟨-12, -1⟩

-- Theorem stating the value of m such that points A, B, and C are collinear
theorem find_m_collinear : ∃ (m : ℤ), collinear_points A (B m) C ∧ m = 9 / 7 := sorry

end find_m_collinear_l3_3070


namespace smallest_q_difference_l3_3287

theorem smallest_q_difference (p q : ℕ) (hp_pos : 0 < p) (hq_pos : 0 < q)
  (h_fraction1 : 3 * q < 5 * p)
  (h_fraction2 : 5 * p < 6 * q)
  (h_smallest : ∀ r s : ℕ, 0 < s → 3 * s < 5 * r → 5 * r < 6 * s → q ≤ s) :
  q - p = 3 :=
by
  sorry

end smallest_q_difference_l3_3287


namespace find_f_2_l3_3723

theorem find_f_2 (f : ℕ → ℕ) (h : ∀ x, f (x + 1) = 2 * x + 3) : f 2 = 5 :=
sorry

end find_f_2_l3_3723


namespace three_times_value_intervals_correctness_l3_3312

open Function

theorem three_times_value_intervals_correctness :
  (∀ a b : ℝ, a > 0 ∧ b > 0 ∧ a * b = 1/3 ∧ (∀ x, a ≤ x → x ≤ b → IsDecreasing (λ x, x⁻¹))) ∨
  (∀ a b : ℝ, 0 ≤ a ∧ 0 ≤ b ∧ a^2 = 3 * a ∧ b^2 = 3 * b ∧ (∀ x, a ≤ x → x ≤ b → (λ x, x^2)')) :=
sorry

end three_times_value_intervals_correctness_l3_3312


namespace sin_330_eq_neg_sin_30_l3_3984

theorem sin_330_eq_neg_sin_30 :
  sin (330 : ℝ) = - sin (30 : ℝ) := sorry

end sin_330_eq_neg_sin_30_l3_3984


namespace sin_330_correct_l3_3967

noncomputable def sin_330 : ℝ := sorry

theorem sin_330_correct : sin_330 = -1 / 2 :=
  sorry

end sin_330_correct_l3_3967


namespace problem_equiv_proof_l3_3455

noncomputable def prob_b1_div_b2_div_b3 : ℚ :=
  let T := { d | ∃ (e1 : ℕ) (e2 : ℕ), d = 2^e1 * 3^e2 ∧ e1 ≤ 5 ∧ e2 ≤ 10 }
  let choices := (T.to_finset.card : ℚ) ^ 3
  let valid_pairs := (nat.ascents 3 5).card * (nat.ascents 3 10).card
  valid_pairs / choices

theorem problem_equiv_proof :
  prob_b1_div_b2_div_b3 = 77 / 1387 := by
    sorry

end problem_equiv_proof_l3_3455


namespace num_rose_bushes_approximation_l3_3657

noncomputable def num_rose_bushes (radius spacing : ℝ) : ℝ :=
  2 * real.pi * radius / spacing

theorem num_rose_bushes_approximation :
  num_rose_bushes 15 0.75 ≈ 126 := 
by 
  sorry

end num_rose_bushes_approximation_l3_3657


namespace greatest_possible_x_l3_3339

theorem greatest_possible_x (x : ℕ) (h : x^4 / x^2 < 18) : x ≤ 4 :=
sorry

end greatest_possible_x_l3_3339


namespace election_result_l3_3448

theorem election_result (total_votes : ℕ) (invalid_vote_percentage valid_vote_percentage : ℚ) 
  (candidate_A_percentage : ℚ) (hv: valid_vote_percentage = 1 - invalid_vote_percentage) 
  (ht: total_votes = 560000) 
  (hi: invalid_vote_percentage = 0.15) 
  (hc: candidate_A_percentage = 0.80) : 
  (candidate_A_percentage * valid_vote_percentage * total_votes = 380800) :=
by 
  sorry

end election_result_l3_3448


namespace min_a2_plus_b2_quartic_eq_l3_3319

theorem min_a2_plus_b2_quartic_eq (a b : ℝ) (x : ℝ) 
  (h : x^4 + a * x^3 + b * x^2 + a * x + 1 = 0) : 
  a^2 + b^2 ≥ 4/5 := 
sorry

end min_a2_plus_b2_quartic_eq_l3_3319


namespace triangle_to_rectangle_ratio_l3_3076

def triangle_perimeter := 60
def rectangle_perimeter := 60

def is_equilateral_triangle (side_length: ℝ) : Prop :=
  3 * side_length = triangle_perimeter

def is_valid_rectangle (length width: ℝ) : Prop :=
  2 * (length + width) = rectangle_perimeter ∧ length = 2 * width

theorem triangle_to_rectangle_ratio (s l w: ℝ) 
  (ht: is_equilateral_triangle s) 
  (hr: is_valid_rectangle l w) : 
  s / w = 2 := by
  sorry

end triangle_to_rectangle_ratio_l3_3076


namespace clinton_shoes_count_l3_3386

def num_hats : ℕ := 5
def num_belts : ℕ := num_hats + 2
def num_shoes : ℕ := 2 * num_belts

theorem clinton_shoes_count : num_shoes = 14 := by
  -- proof goes here
  sorry

end clinton_shoes_count_l3_3386


namespace compute_expression_l3_3387

theorem compute_expression : -8 * 4 - (-6 * -3) + (-10 * -5) = 0 := by sorry

end compute_expression_l3_3387


namespace gcf_factorial_seven_eight_l3_3399

theorem gcf_factorial_seven_eight (a b : ℕ) (h : a = 7! ∧ b = 8!) : Nat.gcd a b = 7! := 
by 
  sorry

end gcf_factorial_seven_eight_l3_3399


namespace sin_330_eq_neg_half_l3_3939

noncomputable def Q : ℝ × ℝ := (Real.sqrt 3 / 2, -1 / 2)

theorem sin_330_eq_neg_half : Real.sin (330 * Real.pi / 180) = -1 / 2 :=
by
  have h1 : 330 * Real.pi / 180 = 11 * Real.pi / 6 := by sorry
  have coord_y : (Q.snd) = -1 / 2 := by sorry
  rw [h1, coord_y]
  sorry

end sin_330_eq_neg_half_l3_3939


namespace equilateral_triangle_perimeter_l3_3606

theorem equilateral_triangle_perimeter (s : ℝ) (h1 : s ≠ 0) (h2 : (s ^ 2 * real.sqrt 3) / 4 = 2 * s) : 
  3 * s = 8 * real.sqrt 3 := 
by
  sorry

end equilateral_triangle_perimeter_l3_3606


namespace nearest_integer_3_add_sqrt_5_pow_6_l3_3791

noncomputable def approx (x : ℝ) : ℕ := Real.floor (x + 0.5)

theorem nearest_integer_3_add_sqrt_5_pow_6 : 
  approx ((3 + Real.sqrt 5)^6) = 22608 :=
by
  -- Proof omitted, sorry
  sorry

end nearest_integer_3_add_sqrt_5_pow_6_l3_3791


namespace sum_of_numbers_l3_3007

theorem sum_of_numbers (a b c : ℝ) (h_ratio : a / 1 = b / 2 ∧ b / 2 = c / 3) (h_sum_squares : a^2 + b^2 + c^2 = 2744) : 
  a + b + c = 84 := 
sorry

end sum_of_numbers_l3_3007


namespace correct_statement_D_l3_3645

theorem correct_statement_D (h : 3.14 < Real.pi) : -3.14 > -Real.pi := by
sorry

end correct_statement_D_l3_3645


namespace sculpture_and_base_height_l3_3201

def height_in_inches (feet: ℕ) (inches: ℕ) : ℕ :=
  feet * 12 + inches

theorem sculpture_and_base_height
  (sculpture_feet: ℕ) (sculpture_inches: ℕ) (base_inches: ℕ)
  (hf: sculpture_feet = 2)
  (hi: sculpture_inches = 10)
  (hb: base_inches = 8)
  : height_in_inches sculpture_feet sculpture_inches + base_inches = 42 :=
by
  -- Placeholder for the proof
  sorry

end sculpture_and_base_height_l3_3201


namespace negation_exists_to_forall_l3_3181

theorem negation_exists_to_forall (P : ℝ → Prop) (h : ∃ x : ℝ, x^2 + 3 * x + 2 < 0) :
  (¬ (∃ x : ℝ, x^2 + 3 * x + 2 < 0)) ↔ (∀ x : ℝ, x^2 + 3 * x + 2 ≥ 0) := by
sorry

end negation_exists_to_forall_l3_3181


namespace find_x_plus_y_squared_l3_3561

variable (x y a b : ℝ)

def condition1 := x * y = b
def condition2 := (1 / (x ^ 2)) + (1 / (y ^ 2)) = a

theorem find_x_plus_y_squared (h1 : condition1 x y b) (h2 : condition2 x y a) : 
  (x + y) ^ 2 = a * b ^ 2 + 2 * b :=
by
  sorry

end find_x_plus_y_squared_l3_3561


namespace sin_330_is_minus_sqrt3_over_2_l3_3994

-- Define the angle theta as 330 degrees
def theta := 330 * Real.toRad

-- State that point corresponding to 330 degrees lies in the fourth quadrant
def point_in_fourth_quadrant (θ : ℝ) := 330 * Real.toRad = θ ∧ θ > 270 * Real.toRad ∧ θ < 360 * Real.toRad

-- Define a proof problem that states computing sin of 330° equals -√3/2
theorem sin_330_is_minus_sqrt3_over_2 : point_in_fourth_quadrant theta → Real.sin theta = -Real.sqrt 3 / 2 := by
  sorry

end sin_330_is_minus_sqrt3_over_2_l3_3994


namespace g_value_at_5_l3_3002

noncomputable def g : ℝ → ℝ := sorry

theorem g_value_at_5 (h : ∀ x : ℝ, g x + 3 * g (2 - x) = 4 * x ^ 2) : g 5 = 1 := 
by 
  sorry

end g_value_at_5_l3_3002


namespace intersection_of_P_and_Q_l3_3121

def P (x : ℝ) : Prop := 1 < x ∧ x < 4
def Q (x : ℝ) : Prop := 2 < x ∧ x < 3

theorem intersection_of_P_and_Q (x : ℝ) : P x ∧ Q x ↔ 2 < x ∧ x < 3 := by
  sorry

end intersection_of_P_and_Q_l3_3121


namespace gcf_7_8_fact_l3_3408

noncomputable def factorial : ℕ → ℕ
| 0     := 1
| (n+1) := (n+1) * factorial n

theorem gcf_7_8_fact : Nat.gcd (factorial 7) (factorial 8) = 5040 := by
  sorry

end gcf_7_8_fact_l3_3408


namespace values_of_z_l3_3248

theorem values_of_z (x z : ℝ) 
  (h1 : 3 * x^2 + 9 * x + 7 * z + 2 = 0)
  (h2 : 3 * x + z + 4 = 0) : 
  z^2 + 20 * z - 14 = 0 := 
sorry

end values_of_z_l3_3248


namespace point_and_sum_of_coordinates_l3_3469

-- Definitions
def point_on_graph_of_g_over_3 (g : ℝ → ℝ) (p : ℝ × ℝ) : Prop :=
  p.2 = (g p.1) / 3

def point_on_graph_of_inv_g_over_3 (g : ℝ → ℝ) (q : ℝ × ℝ) : Prop :=
  q.2 = (g⁻¹ q.1) / 3

-- Main statement
theorem point_and_sum_of_coordinates {g : ℝ → ℝ} (h : point_on_graph_of_g_over_3 g (2, 3)) :
  point_on_graph_of_inv_g_over_3 g (9, 2 / 3) ∧ (9 + 2 / 3 = 29 / 3) :=
by
  sorry

end point_and_sum_of_coordinates_l3_3469


namespace sin_330_eq_neg_one_half_l3_3857

theorem sin_330_eq_neg_one_half :
  ∃ θ, θ = 330 ∧ (sin (Real.pi * (θ / 180)) = -1 / 2) :=
by
  use 330
  split
  · rfl
  · sorry

end sin_330_eq_neg_one_half_l3_3857


namespace product_of_two_numbers_l3_3332

theorem product_of_two_numbers 
  (x y : ℝ) 
  (h1 : x - y = 2) 
  (h2 : x + y = 8 * (x - y)) 
  (h3 : x * y = 40 * (x - y)) 
  : x * y = 63 := 
by 
  sorry

end product_of_two_numbers_l3_3332


namespace undefined_expression_real_val_l3_3423

theorem undefined_expression_real_val (a : ℝ) :
  a = 2 → (a^3 - 8 = 0) :=
by
  intros
  sorry

end undefined_expression_real_val_l3_3423


namespace sin_330_eq_neg_sin_30_l3_3977

theorem sin_330_eq_neg_sin_30 :
  sin (330 : ℝ) = - sin (30 : ℝ) := sorry

end sin_330_eq_neg_sin_30_l3_3977


namespace polynomial_integer_roots_l3_3812

theorem polynomial_integer_roots (b1 b2 : ℤ) (x : ℤ) (h : x^3 + b2 * x^2 + b1 * x + 18 = 0) :
  x = -18 ∨ x = -9 ∨ x = -6 ∨ x = -3 ∨ x = -2 ∨ x = -1 ∨ x = 1 ∨ x = 2 ∨ x = 3 ∨ x = 6 ∨ x = 9 ∨ x = 18 :=
sorry

end polynomial_integer_roots_l3_3812


namespace greatest_x_l3_3340

theorem greatest_x (x : ℕ) (h : x > 0 ∧ (x^4 / x^2 : ℚ) < 18) : x ≤ 4 :=
by
  sorry

end greatest_x_l3_3340


namespace arithmetic_sequence_ratio_a10_b10_l3_3728

variable {a : ℕ → ℕ} {b : ℕ → ℕ}
variable {S T : ℕ → ℕ}

-- We assume S_n and T_n are the sums of the first n terms of sequences a and b respectively.
-- We also assume the provided ratio condition between S_n and T_n.
axiom sum_of_first_n_terms_a (n : ℕ) : S n = (n * (2 * a 1 + (n - 1) * (a 2 - a 1))) / 2
axiom sum_of_first_n_terms_b (n : ℕ) : T n = (n * (2 * b 1 + (n - 1) * (b 2 - b 1))) / 2
axiom ratio_condition (n : ℕ) : (S n) / (T n) = (3 * n - 1) / (2 * n + 3)

theorem arithmetic_sequence_ratio_a10_b10 : (a 10) / (b 10) = 56 / 41 :=
by sorry

end arithmetic_sequence_ratio_a10_b10_l3_3728


namespace arithmetic_geometric_sum_l3_3734

theorem arithmetic_geometric_sum (a b c : ℝ) (h1 : a ≠ b) (h2 : b ≠ c) (h3 : c ≠ a)
  (h_arith : 2 * b = a + c) (h_geom : a^2 = b * c) 
  (h_sum : a + 3 * b + c = 10) : a = -4 :=
by
  sorry

end arithmetic_geometric_sum_l3_3734


namespace sin_330_eq_neg_half_l3_3937

noncomputable def Q : ℝ × ℝ := (Real.sqrt 3 / 2, -1 / 2)

theorem sin_330_eq_neg_half : Real.sin (330 * Real.pi / 180) = -1 / 2 :=
by
  have h1 : 330 * Real.pi / 180 = 11 * Real.pi / 6 := by sorry
  have coord_y : (Q.snd) = -1 / 2 := by sorry
  rw [h1, coord_y]
  sorry

end sin_330_eq_neg_half_l3_3937


namespace exponentiation_81_5_4_eq_243_l3_3090

theorem exponentiation_81_5_4_eq_243 : 81^(5/4) = 243 := by
  sorry

end exponentiation_81_5_4_eq_243_l3_3090


namespace range_of_m_l3_3260

noncomputable def f (x : ℝ) : ℝ := 2 * sin (3 * x + π / 6) + 1

theorem range_of_m (m : ℝ) :
  (∀ x ∈ Icc (0 : ℝ) (π / 3), f x + m = 0 →  ∃! x1, ∃! x2, x1 ≠ x2) ↔ m ∈ Icc (-3 : ℝ) (-2) :=
by sorry

end range_of_m_l3_3260


namespace sin_330_eq_neg_sqrt3_div_2_l3_3998

theorem sin_330_eq_neg_sqrt3_div_2 
  (R : ℝ × ℝ)
  (hR : R = (1/2, -sqrt(3)/2))
  : Real.sin (330 * Real.pi / 180) = -sqrt(3)/2 :=
by
  sorry

end sin_330_eq_neg_sqrt3_div_2_l3_3998


namespace turtles_remaining_l3_3660

-- Define the initial number of turtles
def initial_turtles : ℕ := 9

-- Define the number of turtles that climbed onto the log
def climbed_turtles : ℕ := 3 * initial_turtles - 2

-- Define the total number of turtles on the log before any jump off
def total_turtles_before_jumping : ℕ := initial_turtles + climbed_turtles

-- Define the number of turtles remaining after half jump off
def remaining_turtles : ℕ := total_turtles_before_jumping / 2

theorem turtles_remaining : remaining_turtles = 17 :=
  by
  -- Placeholder for the proof
  sorry

end turtles_remaining_l3_3660


namespace total_pigs_in_barn_l3_3781

-- Define the number of pigs initially in the barn
def initial_pigs : ℝ := 2465.25

-- Define the number of pigs that join
def joining_pigs : ℝ := 5683.75

-- Define the total number of pigs after they join
def total_pigs : ℝ := 8149

-- The theorem that states the total number of pigs is the sum of initial and joining pigs
theorem total_pigs_in_barn : initial_pigs + joining_pigs = total_pigs := 
by
  sorry

end total_pigs_in_barn_l3_3781


namespace sin_330_eq_neg_one_half_l3_3965

theorem sin_330_eq_neg_one_half : Real.sin (330 * Real.pi / 180) = -1 / 2 := by
  sorry

end sin_330_eq_neg_one_half_l3_3965


namespace common_difference_of_arithmetic_sequence_l3_3278

theorem common_difference_of_arithmetic_sequence
  (a : ℕ → ℤ)
  (d : ℤ)
  (h1 : a 1 + a 9 = 10)
  (h2 : a 2 = -1)
  (h3 : ∀ n, a n = a 1 + (n - 1) * d) :
  d = 2 :=
by
  sorry

end common_difference_of_arithmetic_sequence_l3_3278


namespace point_A_inside_circle_O_l3_3718

-- Definitions based on conditions in the problem
def radius := 5 -- in cm
def distance_to_center := 4 -- in cm

-- The theorem to be proven
theorem point_A_inside_circle_O (r d : ℝ) (hr : r = 5) (hd : d = 4) (h : r > d) : true :=
by {
  sorry
}

end point_A_inside_circle_O_l3_3718


namespace domain_of_f_l3_3610

noncomputable def f (x : ℝ) : ℝ := 1 / (Real.log (x + 1)) + Real.sqrt (4 - x^2)

theorem domain_of_f :
  {x : ℝ | x + 1 > 0 ∧ Real.log (x + 1) ≠ 0 ∧ 4 - x^2 ≥ 0} =
  {x : ℝ | -1 < x ∧ x < 0} ∪ {x : ℝ | 0 < x ∧ x ≤ 2} :=
by
  sorry

end domain_of_f_l3_3610


namespace parametric_to_cartesian_l3_3262

variable (R t : ℝ)

theorem parametric_to_cartesian (x y : ℝ) (h1 : x = R * Real.cos t) (h2 : y = R * Real.sin t) : 
  x^2 + y^2 = R^2 := 
by
  sorry

end parametric_to_cartesian_l3_3262


namespace sin_330_eq_neg_half_l3_3936

noncomputable def Q : ℝ × ℝ := (Real.sqrt 3 / 2, -1 / 2)

theorem sin_330_eq_neg_half : Real.sin (330 * Real.pi / 180) = -1 / 2 :=
by
  have h1 : 330 * Real.pi / 180 = 11 * Real.pi / 6 := by sorry
  have coord_y : (Q.snd) = -1 / 2 := by sorry
  rw [h1, coord_y]
  sorry

end sin_330_eq_neg_half_l3_3936


namespace cubic_sum_l3_3172

theorem cubic_sum (a b c : ℤ) (h1 : a + b + c = 7) (h2 : a * b + a * c + b * c = 11) (h3 : a * b * c = -6) :
  a^3 + b^3 + c^3 = 223 :=
by
  sorry

end cubic_sum_l3_3172


namespace correct_result_l3_3344

-- Define the conditions
variables (x : ℤ)
axiom condition1 : (x - 27 + 19 = 84)

-- Define the goal
theorem correct_result : x - 19 + 27 = 100 :=
  sorry

end correct_result_l3_3344


namespace factor_product_l3_3797

theorem factor_product : 2^2 * 3^2 * 5^2 * 7 = 6300 := by
  sorry

end factor_product_l3_3797


namespace mixed_groups_count_l3_3027

theorem mixed_groups_count :
  ∀ (total_children groups_of_3 total_photos boys_photos girls_photos : ℕ),
  total_children = 300 ∧
  groups_of_3 = 100 ∧
  total_photos = 300 ∧
  boys_photos = 100 ∧
  girls_photos = 56 →
  let mixed_photos := total_photos - boys_photos - girls_photos in
  let mixed_groups := mixed_photos / 2 in
  mixed_groups = 72 :=
by
  intros total_children groups_of_3 total_photos boys_photos girls_photos h,
  have h1 : mixed_photos = total_photos - boys_photos - girls_photos := rfl,
  have h2 : mixed_groups = mixed_photos / 2 := rfl,
  rw [h1, h2],
  simp [h],
  sorry

end mixed_groups_count_l3_3027


namespace find_d_plus_q_l3_3283

noncomputable def arithmetic_sequence (a₁ d : ℝ) (n : ℕ) : ℝ := a₁ + (n - 1) * d

noncomputable def geometric_sequence (b₁ q : ℝ) (n : ℕ) : ℝ := b₁ * q ^ (n - 1)

noncomputable def sum_arithmetic_sequence (a₁ d : ℝ) (n : ℕ) : ℝ :=
  n * a₁ + d * (n * (n - 1) / 2)

noncomputable def sum_geometric_sequence (b₁ q : ℝ) (n : ℕ) : ℝ :=
  if q = 1 then n * b₁
  else b₁ * (q ^ n - 1) / (q - 1)

noncomputable def sum_combined_sequence (a₁ d b₁ q : ℝ) (n : ℕ) : ℝ :=
  sum_arithmetic_sequence a₁ d n + sum_geometric_sequence b₁ q n

theorem find_d_plus_q (a₁ d b₁ q : ℝ) (h_seq: ∀ n : ℕ, 0 < n → sum_combined_sequence a₁ d b₁ q n = n^2 - n + 2^n - 1) :
  d + q = 4 :=
  sorry

end find_d_plus_q_l3_3283


namespace concert_ticket_cost_l3_3150

-- Definitions based on the conditions
def hourlyWage : ℝ := 18
def hoursPerWeek : ℝ := 30
def drinkTicketCost : ℝ := 7
def numberOfDrinkTickets : ℝ := 5
def outingPercentage : ℝ := 0.10
def weeksPerMonth : ℝ := 4

-- Proof statement
theorem concert_ticket_cost (hourlyWage hoursPerWeek drinkTicketCost numberOfDrinkTickets outingPercentage weeksPerMonth : ℝ)
  (monthlySalary := weeksPerMonth * (hoursPerWeek * hourlyWage))
  (outingAmount := outingPercentage * monthlySalary)
  (costOfDrinkTickets := numberOfDrinkTickets * drinkTicketCost)
  (costOfConcertTicket := outingAmount - costOfDrinkTickets)
  : costOfConcertTicket = 181 := 
sorry

end concert_ticket_cost_l3_3150


namespace find_n_l3_3133

noncomputable def f (x : ℝ) : ℝ := Real.log x + 2 * x - 6

theorem find_n (n : ℤ) (h : ∃ x, n < x ∧ x < n+1 ∧ f x = 0) : n = 2 :=
sorry

end find_n_l3_3133


namespace sin_330_eq_neg_half_l3_3891

theorem sin_330_eq_neg_half : Real.sin (330 * Real.pi / 180) = -1 / 2 := by
  -- Definitions and conditions from the problem
  have h1 : 330 = 360 - 30 := rfl
  have h2 : ∀ θ, Real.sin θ = -Real.sin (θ - 2 * Real.pi) := sorry
  have h3 : Real.sin (30 * Real.pi / 180) = 1 / 2 := sorry

  -- Use the given conditions to prove the equality
  calc
    Real.sin (330 * Real.pi / 180)
        = Real.sin (-(30 * Real.pi / 180) + 2 * Real.pi) : by rw [←h1, Real.sin_angle_sub_pi]
    ... = -Real.sin (30 * Real.pi / 180) : by rw [h2]
    ... = -1 / 2 : by rw [h3]

end sin_330_eq_neg_half_l3_3891


namespace max_frac_sum_l3_3115

theorem max_frac_sum (n a b c d : ℕ) (hn : 1 < n) (hab : 0 < a) (hcd : 0 < c)
    (hfrac : (a / b) + (c / d) < 1) (hsum : a + c ≤ n) :
    (∃ (b_val : ℕ), 2 ≤ b_val ∧ b_val ≤ n ∧ 
    1 - 1 / (b_val * (b_val * (n + 1 - b_val) + 1)) = 
    1 - 1 / ((2 * n / 3 + 7 / 6) * ((2 * n / 3 + 7 / 6) * (n - (2 * n / 3 + 1 / 6)) + 1))) :=
sorry

end max_frac_sum_l3_3115


namespace unique_solution_of_quadratic_l3_3603

theorem unique_solution_of_quadratic (b c x : ℝ) (h_eqn : 9 * x^2 + b * x + c = 0) (h_one_solution : ∀ y: ℝ, 9 * y^2 + b * y + c = 0 → y = x) (h_b2_4c : b^2 = 4 * c) : 
  x = -b / 18 := 
by 
  sorry

end unique_solution_of_quadratic_l3_3603


namespace minimum_number_of_guests_l3_3351

theorem minimum_number_of_guests (total_food : ℝ) (max_food_per_guest : ℝ) (H₁ : total_food = 406) (H₂ : max_food_per_guest = 2.5) : 
  ∃ n : ℕ, (n : ℝ) ≥ 163 ∧ total_food / max_food_per_guest ≤ (n : ℝ) := 
by
  sorry

end minimum_number_of_guests_l3_3351


namespace closest_to_2010_l3_3671

theorem closest_to_2010 :
  let A := 2008 * 2012
  let B := 1000 * Real.pi
  let C := 58 * 42
  let D := (48.3 ^ 2 - 2 * 8.3 * 48.3 + 8.3 ^ 2)
  abs (2010 - D) < abs (2010 - A) ∧
  abs (2010 - D) < abs (2010 - B) ∧
  abs (2010 - D) < abs (2010 - C) :=
by
  sorry

end closest_to_2010_l3_3671


namespace sin_330_eq_neg_one_half_l3_3912

theorem sin_330_eq_neg_one_half :
  ∃ θ : ℝ, sin (330 * real.pi / 180) = -1 / 2 :=
begin
  have h1 : sin (360 * real.pi / 180 - 30 * real.pi / 180) = -sin (30 * real.pi / 180),
  { rw sin_sub,
    ring,
    rw sin_pi_div_six, -- sin 30° = 1 / 2
  },
  have h2 : sin (30 * real.pi / 180) = 1 / 2 := by exact real.sin_pi_div_six,
  use 330,
  calc
    sin (330 * real.pi / 180)
        = sin ((360 - 30) * real.pi / 180)      : by ring
    ... = sin (360 * real.pi / 180 - 30 * real.pi / 180) : by congr' 1; ring
    ... = -sin (30 * real.pi / 180)            : by exact h1
    ... = -1 / 2                               : by rw h2,
end

end sin_330_eq_neg_one_half_l3_3912


namespace max_discount_l3_3656

theorem max_discount (cost_price selling_price : ℝ) (min_profit_margin : ℝ) (x : ℝ) : 
  cost_price = 400 → selling_price = 500 → min_profit_margin = 0.0625 → 
  (selling_price * (1 - x / 100) - cost_price ≥ min_profit_margin * cost_price) → x ≤ 15 :=
by
  intros h1 h2 h3 h4
  sorry

end max_discount_l3_3656


namespace raisin_fraction_of_mixture_l3_3202

noncomputable def raisin_nut_cost_fraction (R : ℝ) : ℝ :=
  let raisin_cost := 3 * R
  let nut_cost := 4 * (4 * R)
  let total_cost := raisin_cost + nut_cost
  raisin_cost / total_cost

theorem raisin_fraction_of_mixture (R : ℝ) : raisin_nut_cost_fraction R = 3 / 19 :=
by
  sorry

end raisin_fraction_of_mixture_l3_3202


namespace bobbie_letters_to_remove_l3_3757

-- Definitions of the conditions
def samanthaLastNameLength := 7
def bobbieLastNameLength := samanthaLastNameLength + 3
def jamieLastNameLength := 4
def targetBobbieLastNameLength := 2 * jamieLastNameLength

-- Question: How many letters does Bobbie need to take off to have a last name twice the length of Jamie's?
theorem bobbie_letters_to_remove : 
  bobbieLastNameLength - targetBobbieLastNameLength = 2 := by 
  sorry

end bobbie_letters_to_remove_l3_3757


namespace volume_of_bounded_figure_l3_3080

-- Define the volume of a cube with edge length 1
def volume_of_cube (a : ℝ) : ℝ := a^3

-- Define the edge length of the smaller cubes
def small_cube_edge_length (a : ℝ) : ℝ := a / 2

-- Define the volume of a small cube
def volume_of_small_cube (a : ℝ) : ℝ := volume_of_cube (small_cube_edge_length a)

-- Theorem: Proving the volume of the bounded figure
theorem volume_of_bounded_figure (a : ℝ) : volume_of_cube a = 1 → 
  let V := volume_of_small_cube a in 8 * (V / 2) = 1 / 2 :=
begin
  sorry
end

end volume_of_bounded_figure_l3_3080


namespace correct_addition_l3_3207

-- Define the initial conditions and goal
theorem correct_addition (x : ℕ) : (x + 26 = 61) → (x + 62 = 97) :=
by
  intro h
  -- Proof steps would be provided here
  sorry

end correct_addition_l3_3207


namespace sin_330_is_minus_sqrt3_over_2_l3_3991

-- Define the angle theta as 330 degrees
def theta := 330 * Real.toRad

-- State that point corresponding to 330 degrees lies in the fourth quadrant
def point_in_fourth_quadrant (θ : ℝ) := 330 * Real.toRad = θ ∧ θ > 270 * Real.toRad ∧ θ < 360 * Real.toRad

-- Define a proof problem that states computing sin of 330° equals -√3/2
theorem sin_330_is_minus_sqrt3_over_2 : point_in_fourth_quadrant theta → Real.sin theta = -Real.sqrt 3 / 2 := by
  sorry

end sin_330_is_minus_sqrt3_over_2_l3_3991


namespace sin_330_correct_l3_3969

noncomputable def sin_330 : ℝ := sorry

theorem sin_330_correct : sin_330 = -1 / 2 :=
  sorry

end sin_330_correct_l3_3969


namespace sin_330_eq_neg_half_l3_3894

theorem sin_330_eq_neg_half : Real.sin (330 * Real.pi / 180) = -1 / 2 := by
  -- Definitions and conditions from the problem
  have h1 : 330 = 360 - 30 := rfl
  have h2 : ∀ θ, Real.sin θ = -Real.sin (θ - 2 * Real.pi) := sorry
  have h3 : Real.sin (30 * Real.pi / 180) = 1 / 2 := sorry

  -- Use the given conditions to prove the equality
  calc
    Real.sin (330 * Real.pi / 180)
        = Real.sin (-(30 * Real.pi / 180) + 2 * Real.pi) : by rw [←h1, Real.sin_angle_sub_pi]
    ... = -Real.sin (30 * Real.pi / 180) : by rw [h2]
    ... = -1 / 2 : by rw [h3]

end sin_330_eq_neg_half_l3_3894


namespace train_passing_time_l3_3224

theorem train_passing_time 
  (length_of_train : ℕ) 
  (length_of_platform : ℕ) 
  (time_to_pass_pole : ℕ) 
  (speed_of_train : ℕ) 
  (combined_length : ℕ) 
  (time_to_pass_platform : ℕ) 
  (h1 : length_of_train = 240) 
  (h2 : length_of_platform = 650)
  (h3 : time_to_pass_pole = 24)
  (h4 : speed_of_train = length_of_train / time_to_pass_pole)
  (h5 : combined_length = length_of_train + length_of_platform)
  (h6 : time_to_pass_platform = combined_length / speed_of_train) : 
  time_to_pass_platform = 89 :=
sorry

end train_passing_time_l3_3224


namespace solve_frac_eqn_l3_3242

theorem solve_frac_eqn (x : ℝ) :
  (1 / ((x - 1) * (x - 3)) + 1 / ((x - 3) * (x - 5)) +
   1 / ((x - 5) * (x - 7)) + 1 / ((x - 7) * (x - 9)) = 1 / 8) ↔ 
  (x = 13 ∨ x = -3) :=
by
  sorry

end solve_frac_eqn_l3_3242


namespace pelican_speed_l3_3231

theorem pelican_speed
  (eagle_speed falcon_speed hummingbird_speed total_distance time : ℕ)
  (eagle_distance falcon_distance hummingbird_distance : ℕ)
  (H1 : eagle_speed = 15)
  (H2 : falcon_speed = 46)
  (H3 : hummingbird_speed = 30)
  (H4 : time = 2)
  (H5 : total_distance = 248)
  (H6 : eagle_distance = eagle_speed * time)
  (H7 : falcon_distance = falcon_speed * time)
  (H8 : hummingbird_distance = hummingbird_speed * time)
  (total_other_birds_distance : ℕ)
  (H9 : total_other_birds_distance = eagle_distance + falcon_distance + hummingbird_distance)
  (pelican_distance : ℕ)
  (H10 : pelican_distance = total_distance - total_other_birds_distance)
  (pelican_speed : ℕ)
  (H11 : pelican_speed = pelican_distance / time) :
  pelican_speed = 33 := 
  sorry

end pelican_speed_l3_3231


namespace giraffes_difference_l3_3782

theorem giraffes_difference :
  ∃ n : ℕ, (300 = 3 * n) ∧ (300 - n = 200) :=
by
  sorry

end giraffes_difference_l3_3782


namespace certain_amount_is_19_l3_3321

theorem certain_amount_is_19 (x y certain_amount : ℤ) 
  (h1 : x + y = 15)
  (h2 : 3 * x = 5 * y - certain_amount)
  (h3 : x = 7) : 
  certain_amount = 19 :=
by
  sorry

end certain_amount_is_19_l3_3321


namespace sin_330_l3_3931

theorem sin_330 : Real.sin (330 * Real.pi / 180) = -1 / 2 := 
by
  sorry

end sin_330_l3_3931


namespace arithmetic_sequence_properties_l3_3127

theorem arithmetic_sequence_properties (a b c : ℝ) (h1 : ∃ d : ℝ, [2, a, b, c, 9] = [2, 2 + d, 2 + 2 * d, 2 + 3 * d, 2 + 4 * d]) : 
  c - a = 7 / 2 := 
by
  -- We assume the proof here
  sorry

end arithmetic_sequence_properties_l3_3127


namespace necessary_but_not_sufficient_condition_for_inequality_l3_3075

theorem necessary_but_not_sufficient_condition_for_inequality 
    {a b c : ℝ} (h : a * c^2 ≥ b * c^2) : ¬(a > b → (a * c^2 < b * c^2)) :=
by
  sorry

end necessary_but_not_sufficient_condition_for_inequality_l3_3075


namespace train_length_l3_3052

theorem train_length (speed_kmph : ℝ) (time_sec : ℝ) (length : ℝ) : 
  speed_kmph = 60 → time_sec = 12 → 
  length = speed_kmph * (1000 / 3600) * time_sec → 
  length = 200.04 :=
by
  intros h_speed h_time h_length
  sorry

end train_length_l3_3052


namespace sin_330_eq_neg_half_l3_3841

theorem sin_330_eq_neg_half 
  (h1: (330 : ℝ) = 330)
  (h2: (330 : ℝ = 360 - 30))
  (h3: 30 ∈ reference_angle_set ∧ sin 30 = 1/2)
  (h4: 330 ∈ fourth_quadrant)
  (h5: ∀ θ ∈ fourth_quadrant, sin θ < 0) :
  sin 330 = -1/2 := 
by
  sorry

end sin_330_eq_neg_half_l3_3841


namespace largest_divisor_of_exp_and_linear_combination_l3_3701

theorem largest_divisor_of_exp_and_linear_combination :
  ∃ x : ℕ, (∀ y : ℕ, x ∣ (7^y + 12*y - 1)) ∧ x = 18 :=
by
  sorry

end largest_divisor_of_exp_and_linear_combination_l3_3701


namespace helicopter_A_highest_altitude_helicopter_A_final_altitude_helicopter_B_5th_performance_l3_3086

def heights_A : List ℝ := [3.6, -2.4, 2.8, -1.5, 0.9]
def heights_B : List ℝ := [3.8, -2, 4.1, -2.3]

theorem helicopter_A_highest_altitude :
  List.maximum heights_A = some 3.6 :=
by sorry

theorem helicopter_A_final_altitude :
  List.sum heights_A = 3.4 :=
by sorry

theorem helicopter_B_5th_performance :
  ∃ (x : ℝ), List.sum heights_B + x = 3.4 ∧ x = -0.2 :=
by sorry

end helicopter_A_highest_altitude_helicopter_A_final_altitude_helicopter_B_5th_performance_l3_3086


namespace apple_counts_l3_3161

theorem apple_counts (x y : ℤ) (h1 : y - x = 2) (h2 : y = 3 * x - 4) : x = 3 ∧ y = 5 := 
by
  sorry

end apple_counts_l3_3161


namespace sin_330_deg_l3_3872

noncomputable theory

open Real

theorem sin_330_deg :
  sin (330 * (π / 180)) = -1 / 2 :=
by sorry

end sin_330_deg_l3_3872


namespace sin_330_eq_neg_one_half_l3_3962

theorem sin_330_eq_neg_one_half : Real.sin (330 * Real.pi / 180) = -1 / 2 := by
  sorry

end sin_330_eq_neg_one_half_l3_3962


namespace sin_330_eq_neg_half_l3_3917

open Real

theorem sin_330_eq_neg_half : sin (330 * (π / 180)) = - (1 / 2) := by
  have angle_eq : 330 * (π / 180) = 2 * π - (30 * (π / 180)) := by sorry
  have sin_30_pos : sin (30 * (π / 180)) = 1 / 2 := by sorry
  have sin_330_neg : sin (2 * π - (30 * (π / 180))) = - (sin (30 * (π / 180))) := by sorry
  rw [angle_eq, sin_330_neg, sin_30_pos]
  rfl

end sin_330_eq_neg_half_l3_3917


namespace mixed_groups_count_l3_3018

namespace YoungPhotographerClub

theorem mixed_groups_count
  (total_children : ℕ)
  (total_groups : ℕ)
  (children_per_group : ℕ)
  (photographs_per_group : ℕ)
  (boy_boy_photographs : ℕ)
  (girl_girl_photographs : ℕ)
  (total_photographs : ℕ)
  (mixed_photographs : ℕ)
  (mixed_groups : ℕ)
  (H1 : total_children = 300)
  (H2 : total_groups = 100)
  (H3 : children_per_group = 3)
  (H4 : photographs_per_group = 3)
  (H5 : boy_boy_photographs = 100)
  (H6 : girl_girl_photographs = 56)
  (H7 : total_photographs = total_groups * photographs_per_group)
  (H8 : mixed_photographs = total_photographs - boy_boy_photographs - girl_girl_photographs)
  (H9 : mixed_groups = mixed_photographs / 2) : mixed_groups = 72 := 
by
  rw [H1, H2, H3, H4, H5, H6, H7, H8, H9]
  sorry

end YoungPhotographerClub

end mixed_groups_count_l3_3018


namespace square_of_neg_three_l3_3327

theorem square_of_neg_three : (-3 : ℤ)^2 = 9 := by
  sorry

end square_of_neg_three_l3_3327


namespace length_of_living_room_l3_3813

theorem length_of_living_room (L : ℝ) (width : ℝ) (border_width : ℝ) (border_area : ℝ) 
  (h1 : width = 10)
  (h2 : border_width = 2)
  (h3 : border_area = 72) :
  L = 12 :=
by
  sorry

end length_of_living_room_l3_3813


namespace sin_330_degree_l3_3946

theorem sin_330_degree : Real.sin (330 * Real.pi / 180) = -1 / 2 :=
by
  sorry

end sin_330_degree_l3_3946


namespace sin_330_eq_neg_half_l3_3922

open Real

theorem sin_330_eq_neg_half : sin (330 * (π / 180)) = - (1 / 2) := by
  have angle_eq : 330 * (π / 180) = 2 * π - (30 * (π / 180)) := by sorry
  have sin_30_pos : sin (30 * (π / 180)) = 1 / 2 := by sorry
  have sin_330_neg : sin (2 * π - (30 * (π / 180))) = - (sin (30 * (π / 180))) := by sorry
  rw [angle_eq, sin_330_neg, sin_30_pos]
  rfl

end sin_330_eq_neg_half_l3_3922


namespace linear_eq_implies_m_eq_1_l3_3257

theorem linear_eq_implies_m_eq_1 (x y m : ℝ) (h : 3 * (x ^ |m|) + (m + 1) * y = 6) (hm_abs : |m| = 1) (hm_ne_zero : m + 1 ≠ 0) : m = 1 :=
  sorry

end linear_eq_implies_m_eq_1_l3_3257


namespace value_range_of_log_function_l3_3622

noncomputable def quadratic_function (x : ℝ) : ℝ :=
  x^2 - 2*x + 4

noncomputable def log_base_3 (x : ℝ) : ℝ :=
  Real.log x / Real.log 3

theorem value_range_of_log_function :
  ∀ x : ℝ, log_base_3 (quadratic_function x) ≥ 1 := by
  sorry

end value_range_of_log_function_l3_3622


namespace geom_series_common_ratio_l3_3322

theorem geom_series_common_ratio (a r S : ℝ) (h1 : S = a / (1 - r)) 
  (h2 : (ar^4) / (1 - r) = S / 64) : r = 1 / 2 :=
sorry

end geom_series_common_ratio_l3_3322


namespace john_has_leftover_bulbs_l3_3740

-- Definitions of the problem statements
def initial_bulbs : ℕ := 40
def used_bulbs : ℕ := 16
def remaining_bulbs_after_use : ℕ := initial_bulbs - used_bulbs
def given_to_friend : ℕ := remaining_bulbs_after_use / 2

-- Statement to prove
theorem john_has_leftover_bulbs :
  remaining_bulbs_after_use - given_to_friend = 12 :=
by
  sorry

end john_has_leftover_bulbs_l3_3740


namespace altitude_of_triangle_l3_3492

theorem altitude_of_triangle (b h_t h_p : ℝ) (hb : b ≠ 0) 
  (area_eq : b * h_p = (1/2) * b * h_t) 
  (h_p_def : h_p = 100) : h_t = 200 :=
by
  sorry

end altitude_of_triangle_l3_3492


namespace problem_solution_l3_3056

noncomputable def proof_problem (x1 x2 x3 x4 x5 : ℝ) (h1 : 0 < x1) (h2 : 0 < x2) (h3 : 0 < x3) (h4 : 0 < x4) (h5 : 0 < x5) : Prop :=
  ((x1^2 - x3 * x5) * (x2^2 - x3 * x5) ≤ 0) ∧
  ((x2^2 - x4 * x1) * (x3^2 - x4 * x1) ≤ 0) ∧
  ((x3^2 - x5 * x2) * (x4^2 - x5 * x2) ≤ 0) ∧
  ((x4^2 - x1 * x3) * (x5^2 - x1 * x3) ≤ 0) ∧
  ((x5^2 - x2 * x4) * (x1^2 - x2 * x4) ≤ 0) → 
  x1 = x2 ∧ x2 = x3 ∧ x3 = x4 ∧ x4 = x5

theorem problem_solution (x1 x2 x3 x4 x5 : ℝ) (h1 : 0 < x1) (h2 : 0 < x2) (h3 : 0 < x3) (h4 : 0 < x4) (h5 : 0 < x5) :
  proof_problem x1 x2 x3 x4 x5 h1 h2 h3 h4 h5 :=
  by
    sorry

end problem_solution_l3_3056


namespace shorter_side_of_room_l3_3072

theorem shorter_side_of_room
  (P : ℕ) (A : ℕ) (a b : ℕ)
  (perimeter_eq : 2 * a + 2 * b = P)
  (area_eq : a * b = A) (partition_len : ℕ) (partition_cond : partition_len = 5)
  (room_perimeter : P = 60)
  (room_area : A = 200) :
  b = 10 := 
by
  sorry

end shorter_side_of_room_l3_3072


namespace sin_330_deg_l3_3853

theorem sin_330_deg : Real.sin (330 * Real.pi / 180) = -1 / 2 := by
  sorry

end sin_330_deg_l3_3853


namespace phone_calls_to_reach_Davina_l3_3306

theorem phone_calls_to_reach_Davina : 
  (∀ (a b : ℕ), (0 ≤ a ∧ a < 10) ∧ (0 ≤ b ∧ b < 10)) → (least_num_calls : ℕ) = 100 :=
by
  sorry

end phone_calls_to_reach_Davina_l3_3306


namespace solve_problem_l3_3326

open Matrix

variables {R : Type*} [Field R]
variables {S : Matrix (Fin 3) (Fin 1) R → Matrix (Fin 3) (Fin 1) R}
variables {a b : R} {u v : Matrix (Fin 3) (Fin 1) R}
variables {w x : Matrix (Fin 3) (Fin 1) Int}

-- conditions
def cond1 : Prop := ∀ (a b : R) (u v : Matrix (Fin 3) (Fin 1) R), 
  S (a • u + b • v) = a • (S u) + b • (S v)
def cond2 : Prop := ∀ (u v : Matrix (Fin 3) (Fin 1) R),
  S (u × v) = S u × S v
def cond3 : S (λ i, ![5, 2, 7]) = (λ i, ![1, 3, 4])
def cond4 : S (λ i, ![3, 7, 2]) = (λ i, ![4, 6, 5])

-- desired outcome
def target : Prop := S (λ i, ![4, 11, 9]) = (λ i, ![5, 10, 9])

theorem solve_problem
  (h1 : cond1) (h2 : cond2) (h3 : cond3) (h4 : cond4) : target := by
  sorry

end solve_problem_l3_3326


namespace total_cantaloupes_l3_3708

theorem total_cantaloupes (fred_cantaloupes : ℕ) (tim_cantaloupes : ℕ) (h1 : fred_cantaloupes = 38) (h2 : tim_cantaloupes = 44) : fred_cantaloupes + tim_cantaloupes = 82 :=
by sorry

end total_cantaloupes_l3_3708


namespace find_a_range_empty_solution_set_l3_3125

theorem find_a_range_empty_solution_set :
  ∀ a : ℝ, (∀ x : ℝ, (a^2 - 4) * x^2 + (a + 2) * x - 1 ≥ 0 → false) ↔ (-2 ≤ a ∧ a < 6 / 5) :=
by sorry

end find_a_range_empty_solution_set_l3_3125


namespace income_of_deceased_member_l3_3805

theorem income_of_deceased_member
  (A B C : ℝ) -- Incomes of the three members
  (h1 : (A + B + C) / 3 = 735)
  (h2 : (A + B) / 2 = 650) :
  C = 905 :=
by
  sorry

end income_of_deceased_member_l3_3805


namespace sum_of_intersections_l3_3390

theorem sum_of_intersections :
  (∃ x1 y1 x2 y2 x3 y3 x4 y4, 
    y1 = (x1 - 1)^2 ∧ y2 = (x2 - 1)^2 ∧ y3 = (x3 - 1)^2 ∧ y4 = (x4 - 1)^2 ∧
    x1 - 2 = (y1 + 1)^2 ∧ x2 - 2 = (y2 + 1)^2 ∧ x3 - 2 = (y3 + 1)^2 ∧ x4 - 2 = (y4 + 1)^2 ∧
    (x1 + x2 + x3 + x4 + y1 + y2 + y3 + y4) = 2) :=
sorry

end sum_of_intersections_l3_3390


namespace sin_330_eq_neg_half_l3_3945

noncomputable def Q : ℝ × ℝ := (Real.sqrt 3 / 2, -1 / 2)

theorem sin_330_eq_neg_half : Real.sin (330 * Real.pi / 180) = -1 / 2 :=
by
  have h1 : 330 * Real.pi / 180 = 11 * Real.pi / 6 := by sorry
  have coord_y : (Q.snd) = -1 / 2 := by sorry
  rw [h1, coord_y]
  sorry

end sin_330_eq_neg_half_l3_3945


namespace total_difference_proof_l3_3360

-- Definitions for the initial quantities
def initial_tomatoes : ℕ := 17
def initial_carrots : ℕ := 13
def initial_cucumbers : ℕ := 8

-- Definitions for the picked quantities
def picked_tomatoes : ℕ := 5
def picked_carrots : ℕ := 6

-- Definitions for the given away quantities
def given_away_tomatoes : ℕ := 3
def given_away_carrots : ℕ := 2

-- Definitions for the remaining quantities 
def remaining_tomatoes : ℕ := initial_tomatoes - (picked_tomatoes - given_away_tomatoes)
def remaining_carrots : ℕ := initial_carrots - (picked_carrots - given_away_carrots)

-- Definitions for the difference quantities
def difference_tomatoes : ℕ := initial_tomatoes - remaining_tomatoes
def difference_carrots : ℕ := initial_carrots - remaining_carrots

-- Definition for the total difference
def total_difference : ℕ := difference_tomatoes + difference_carrots

-- Lean Theorem Statement
theorem total_difference_proof : total_difference = 6 := by
  -- Proof is omitted
  sorry

end total_difference_proof_l3_3360


namespace largest_possible_b_l3_3426

theorem largest_possible_b (a b c : ℤ) (h1 : a > b) (h2 : b > c) (h3 : c > 2) (h4 : a * b * c = 360) : b = 10 :=
sorry

end largest_possible_b_l3_3426


namespace gcf_7_8_fact_l3_3407

noncomputable def factorial : ℕ → ℕ
| 0     := 1
| (n+1) := (n+1) * factorial n

theorem gcf_7_8_fact : Nat.gcd (factorial 7) (factorial 8) = 5040 := by
  sorry

end gcf_7_8_fact_l3_3407


namespace sin_330_eq_neg_one_half_l3_3861

theorem sin_330_eq_neg_one_half :
  ∃ θ, θ = 330 ∧ (sin (Real.pi * (θ / 180)) = -1 / 2) :=
by
  use 330
  split
  · rfl
  · sorry

end sin_330_eq_neg_one_half_l3_3861


namespace hash_op_correct_l3_3495

-- Definition of the custom operation #
def hash_op (a b : ℕ) : ℕ := a * b - b + b ^ 2

-- The theorem to prove that 3 # 8 = 80
theorem hash_op_correct : hash_op 3 8 = 80 :=
by
  sorry

end hash_op_correct_l3_3495


namespace find_k_l3_3727

theorem find_k 
  (t k r : ℝ)
  (h1 : t = 5 / 9 * (k - 32))
  (h2 : r = 3 * t)
  (h3 : r = 150) : 
  k = 122 := 
sorry

end find_k_l3_3727


namespace triangle_at_most_one_right_angle_l3_3041

-- Definition of a triangle with its angles adding up to 180 degrees
def triangle (α β γ : ℝ) : Prop := α + β + γ = 180

-- The main theorem stating that a triangle can have at most one right angle.
theorem triangle_at_most_one_right_angle (α β γ : ℝ) 
  (h₁ : triangle α β γ) 
  (h₂ : α = 90 ∨ β = 90 ∨ γ = 90) : 
  (α = 90 → β ≠ 90 ∧ γ ≠ 90) ∧ 
  (β = 90 → α ≠ 90 ∧ γ ≠ 90) ∧ 
  (γ = 90 → α ≠ 90 ∧ β ≠ 90) :=
sorry

end triangle_at_most_one_right_angle_l3_3041


namespace cubic_identity_l3_3560

theorem cubic_identity (a b c : ℝ) 
  (h1 : a + b + c = 12)
  (h2 : ab + ac + bc = 30)
  : a^3 + b^3 + c^3 - 3 * a * b * c = 648 := by
  sorry

end cubic_identity_l3_3560


namespace sin_330_correct_l3_3975

noncomputable def sin_330 : ℝ := sorry

theorem sin_330_correct : sin_330 = -1 / 2 :=
  sorry

end sin_330_correct_l3_3975


namespace total_ingredients_l3_3629

theorem total_ingredients (water : ℕ) (flour : ℕ) (salt : ℕ)
  (h_water : water = 10)
  (h_flour : flour = 16)
  (h_salt : salt = flour / 2) :
  water + flour + salt = 34 :=
by
  sorry

end total_ingredients_l3_3629


namespace new_trailer_homes_added_l3_3331

theorem new_trailer_homes_added
  (n : ℕ) (avg_age_3_years_ago avg_age_today age_increase new_home_age : ℕ) (k : ℕ) :
  n = 30 → avg_age_3_years_ago = 15 → avg_age_today = 12 → age_increase = 3 → new_home_age = 3 →
  (n * (avg_age_3_years_ago + age_increase) + k * new_home_age) / (n + k) = avg_age_today →
  k = 20 :=
by
  intros h_n h_avg_age_3y h_avg_age_today h_age_increase h_new_home_age h_eq
  sorry

end new_trailer_homes_added_l3_3331


namespace envelopes_left_l3_3381

theorem envelopes_left (initial_envelopes : ℕ) (envelopes_per_friend : ℕ) (number_of_friends : ℕ) (remaining_envelopes : ℕ) :
  initial_envelopes = 37 → envelopes_per_friend = 3 → number_of_friends = 5 → remaining_envelopes = initial_envelopes - (envelopes_per_friend * number_of_friends) → remaining_envelopes = 22 :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3] at h4
  norm_num at h4
  exact h4

end envelopes_left_l3_3381


namespace age_of_person_l3_3490

theorem age_of_person (x : ℕ) (h : 3 * (x + 3) - 3 * (x - 3) = x) : x = 18 :=
  sorry

end age_of_person_l3_3490


namespace find_C_in_terms_of_D_l3_3158

noncomputable def h (C D x : ℝ) : ℝ := C * x - 3 * D ^ 2
noncomputable def k (D x : ℝ) : ℝ := D * x + 1

theorem find_C_in_terms_of_D (C D : ℝ) (h_eq : h C D (k D 2) = 0) (h_def : ∀ x, h C D x = C * x - 3 * D ^ 2) (k_def : ∀ x, k D x = D * x + 1) (D_ne_neg1 : D ≠ -1) : 
C = (3 * D ^ 2) / (2 * D + 1) := 
by 
  sorry

end find_C_in_terms_of_D_l3_3158


namespace arithmetic_sequence_common_difference_l3_3583

theorem arithmetic_sequence_common_difference
  (a : ℕ → ℝ) (S : ℕ → ℝ) (d : ℝ)
  (h1 : a 1 = 30)
  (h2 : ∀ n, S n = n * (a 1 + (n - 1) / 2 * d))
  (h3 : S 12 = S 19) :
  d = -2 :=
by
  sorry

end arithmetic_sequence_common_difference_l3_3583


namespace print_time_l3_3219

-- Conditions
def printer_pages_per_minute : ℕ := 25
def total_pages : ℕ := 350

-- Theorem
theorem print_time :
  (total_pages / printer_pages_per_minute : ℕ) = 14 :=
by sorry

end print_time_l3_3219


namespace box_weight_difference_l3_3751

theorem box_weight_difference:
  let w1 := 2
  let w2 := 3
  let w3 := 13
  let w4 := 7
  let w5 := 10
  (max (max (max (max w1 w2) w3) w4) w5) - (min (min (min (min w1 w2) w3) w4) w5) = 11 :=
by
  sorry

end box_weight_difference_l3_3751


namespace polynomial_coefficients_l3_3713

theorem polynomial_coefficients (a : ℕ → ℤ) :
  (∀ x : ℤ, (2 * x - 1) * ((x + 1) ^ 7) = (a 0) + (a 1) * x + (a 2) * x^2 + (a 3) * x^3 + 
  (a 4) * x^4 + (a 5) * x^5 + (a 6) * x^6 + (a 7) * x^7 + (a 8) * x^8) →
  (a 0 = -1) ∧
  (a 0 + a 2 + a 4 + a 6 + a 8 = 64) ∧
  (a 1 + 2 * (a 2) + 3 * (a 3) + 4 * (a 4) + 5 * (a 5) + 6 * (a 6) + 7 * (a 7) + 8 * (a 8) = 704) := by
  sorry

end polynomial_coefficients_l3_3713


namespace eval_f_neg2_l3_3044

-- Define the function f
def f (x : ℤ) : ℤ := x^2 - 3*x + 1

-- Theorem statement
theorem eval_f_neg2 : f (-2) = 11 := by
  sorry

end eval_f_neg2_l3_3044


namespace factorize_expr1_factorize_expr2_l3_3101

variable (x y a b : ℝ)

theorem factorize_expr1 : 2 * x^2 - 4 * x + 2 = 2 * (x - 1)^2 := sorry

theorem factorize_expr2 : a^2 * (x - y) + b^2 * (y - x) = (x - y) * (a + b) * (a - b) := sorry

end factorize_expr1_factorize_expr2_l3_3101


namespace arithmetic_sequence_a3_l3_3450

theorem arithmetic_sequence_a3 :
  ∃ (a : ℕ → ℝ) (d : ℝ), 
    (∀ n, a n = 2 + (n - 1) * d) ∧
    (a 1 = 2) ∧
    (a 5 = a 4 + 2) →
    a 3 = 6 :=
sorry

end arithmetic_sequence_a3_l3_3450


namespace find_value_of_expression_l3_3249

variable {x y : ℝ}
variable (hx : x ≠ 0)
variable (hy : y ≠ 0)
variable (h : x + y = x * y + 1)

theorem find_value_of_expression (h : x + y = x * y + 1) : 
  (1 / x) + (1 / y) = 1 + (1 / (x * y)) :=
  sorry

end find_value_of_expression_l3_3249


namespace sin_330_deg_l3_3875

noncomputable theory

open Real

theorem sin_330_deg :
  sin (330 * (π / 180)) = -1 / 2 :=
by sorry

end sin_330_deg_l3_3875


namespace solve_for_x_l3_3729

theorem solve_for_x (x : ℤ) (h : 3 * x - 5 = 4 * x + 10) : x = -15 :=
sorry

end solve_for_x_l3_3729


namespace race_problem_l3_3437

theorem race_problem
  (total_distance : ℕ)
  (A_time : ℕ)
  (B_extra_time : ℕ)
  (A_speed B_speed : ℕ)
  (A_distance B_distance : ℕ)
  (H1 : total_distance = 120)
  (H2 : A_time = 8)
  (H3 : B_extra_time = 7)
  (H4 : A_speed = total_distance / A_time)
  (H5 : B_speed = total_distance / (A_time + B_extra_time))
  (H6 : A_distance = total_distance)
  (H7 : B_distance = B_speed * A_time) :
  A_distance - B_distance = 56 := 
sorry

end race_problem_l3_3437


namespace mixed_groups_count_l3_3015

namespace YoungPhotographerClub

theorem mixed_groups_count
  (total_children : ℕ)
  (total_groups : ℕ)
  (children_per_group : ℕ)
  (photographs_per_group : ℕ)
  (boy_boy_photographs : ℕ)
  (girl_girl_photographs : ℕ)
  (total_photographs : ℕ)
  (mixed_photographs : ℕ)
  (mixed_groups : ℕ)
  (H1 : total_children = 300)
  (H2 : total_groups = 100)
  (H3 : children_per_group = 3)
  (H4 : photographs_per_group = 3)
  (H5 : boy_boy_photographs = 100)
  (H6 : girl_girl_photographs = 56)
  (H7 : total_photographs = total_groups * photographs_per_group)
  (H8 : mixed_photographs = total_photographs - boy_boy_photographs - girl_girl_photographs)
  (H9 : mixed_groups = mixed_photographs / 2) : mixed_groups = 72 := 
by
  rw [H1, H2, H3, H4, H5, H6, H7, H8, H9]
  sorry

end YoungPhotographerClub

end mixed_groups_count_l3_3015


namespace geometric_mean_problem_l3_3451

theorem geometric_mean_problem
  (a : Nat) (a1 : Nat) (a8 : Nat) (r : Rat) 
  (h1 : a1 = 6) (h2 : a8 = 186624) 
  (h3 : a8 = a1 * r^7) 
  : a = a1 * r^3 → a = 1296 := 
by
  sorry

end geometric_mean_problem_l3_3451


namespace sin_330_eq_neg_one_half_l3_3858

theorem sin_330_eq_neg_one_half :
  ∃ θ, θ = 330 ∧ (sin (Real.pi * (θ / 180)) = -1 / 2) :=
by
  use 330
  split
  · rfl
  · sorry

end sin_330_eq_neg_one_half_l3_3858


namespace odd_divisibility_l3_3333

theorem odd_divisibility (n : ℕ) (k : ℕ) (x y : ℤ) (h : n = 2 * k + 1) : (x^n + y^n) % (x + y) = 0 :=
by sorry

end odd_divisibility_l3_3333


namespace sin_330_l3_3896

theorem sin_330 : Real.sin (330 * Real.pi / 180) = -1 / 2 := 
by
  -- Outline the proof here without providing it
  -- sorry to delay the proof
  sorry

end sin_330_l3_3896


namespace markup_percentage_l3_3778

-- Definitions coming from conditions
variables (C : ℝ) (M : ℝ) (S : ℝ)
-- Markup formula
def markup_formula : Prop := M = 0.10 * C
-- Selling price formula
def selling_price_formula : Prop := S = C + M

-- Given the conditions, we need to prove that the markup is 9.09% of the selling price
theorem markup_percentage (h1 : markup_formula C M) (h2 : selling_price_formula C M S) :
  (M / S) * 100 = 9.09 :=
sorry

end markup_percentage_l3_3778


namespace mad_hatter_waiting_time_march_hare_waiting_time_waiting_time_l3_3461

-- Definitions
def mad_hatter_clock_rate := 5 / 4
def march_hare_clock_rate := 5 / 6
def time_at_dormouse_clock := 5 -- 5:00 PM

-- Real time calculation based on clock rates
def real_time (clock_rate : ℚ) (clock_time : ℚ) : ℚ := clock_time * (1 / clock_rate)

-- Mad Hatter's and March Hare's arrival times in real time
def mad_hatter_real_time := real_time mad_hatter_clock_rate time_at_dormouse_clock
def march_hare_real_time := real_time march_hare_clock_rate time_at_dormouse_clock

-- Theorems to be proved
theorem mad_hatter_waiting_time : mad_hatter_real_time = 4 := sorry
theorem march_hare_waiting_time : march_hare_real_time = 6 := sorry

-- Main theorem
theorem waiting_time : march_hare_real_time - mad_hatter_real_time = 2 := sorry

end mad_hatter_waiting_time_march_hare_waiting_time_waiting_time_l3_3461


namespace school_student_count_l3_3814

-- Definition of the conditions
def students_in_school (n : ℕ) : Prop :=
  200 ≤ n ∧ n ≤ 300 ∧
  n % 6 = 1 ∧
  n % 8 = 2 ∧
  n % 9 = 3

-- The main proof statement
theorem school_student_count : ∃ n, students_in_school n ∧ n = 265 :=
by
  sorry  -- Proof would go here

end school_student_count_l3_3814


namespace count_three_digit_perfect_squares_l3_3553

def is_perfect_square (n : ℕ) : Prop :=
  ∃ k : ℕ, n = k * k

def is_three_digit_number (n : ℕ) : Prop :=
  100 ≤ n ∧ n ≤ 999

theorem count_three_digit_perfect_squares : 
  ∃ (count : ℕ), count = 22 ∧
  ∀ (n : ℕ), is_three_digit_number n → is_perfect_square n → true :=
sorry

end count_three_digit_perfect_squares_l3_3553


namespace angle_AXC_angle_ACB_l3_3780

-- Definitions of the problem conditions
variables (A B C D X : Type) [Nonempty A] [Nonempty B] [Nonempty C] [Nonempty D] [Nonempty X]
variables (AD DC: Type) [Nonempty AD] [Nonempty DC]
variables (angleB angleXDC angleAXC angleACB : ℝ)
variables (AB BX: ℝ)

-- Given conditions
axiom equal_sides: AD = DC
axiom pointX: BX = AB
axiom given_angleB: angleB = 34
axiom given_angleXDC: angleXDC = 52

-- Proof goals (no proof included, only the statements)
theorem angle_AXC: angleAXC = 107 :=
sorry

theorem angle_ACB: angleACB = 47 :=
sorry

end angle_AXC_angle_ACB_l3_3780


namespace sin_330_eq_neg_one_half_l3_3863

theorem sin_330_eq_neg_one_half :
  ∃ θ, θ = 330 ∧ (sin (Real.pi * (θ / 180)) = -1 / 2) :=
by
  use 330
  split
  · rfl
  · sorry

end sin_330_eq_neg_one_half_l3_3863


namespace quadratic_fixed_points_l3_3616

noncomputable def quadratic_function (a x : ℝ) : ℝ :=
  a * x^2 + (3 * a - 1) * x - (10 * a + 3)

theorem quadratic_fixed_points (a : ℝ) (h : a ≠ 0) :
  quadratic_function a 2 = -5 ∧ quadratic_function a (-5) = 2 :=
by sorry

end quadratic_fixed_points_l3_3616


namespace value_of_f_neg2_l3_3046

def f (x : ℤ) : ℤ := x^2 - 3 * x + 1

theorem value_of_f_neg2 : f (-2) = 11 := by
  sorry

end value_of_f_neg2_l3_3046


namespace point_on_x_axis_right_of_origin_is_3_units_away_l3_3142

theorem point_on_x_axis_right_of_origin_is_3_units_away :
  ∃ (P : ℝ × ℝ), P.2 = 0 ∧ P.1 > 0 ∧ dist (P.1, P.2) (0, 0) = 3 ∧ P = (3, 0) := 
by
  sorry

end point_on_x_axis_right_of_origin_is_3_units_away_l3_3142


namespace num_elements_in_B_inter_C_l3_3291

open Set

-- Defining sets A, B, and C
def A : Set ℕ := { x | 1 ≤ x ∧ x ≤ 99 }
def B : Set ℕ := { x | ∃ y ∈ A, x = 2 * y }
def C : Set ℕ := { x | 2 * x ∈ A }

-- Theorem to prove
theorem num_elements_in_B_inter_C : 
  (Finset.card ((A.filter (λ x, 2 * x ∈ A)).image (λ x, 2 * x)) : ℕ) ∩ 
  (Finset.card (A.filter (λ x, 2 * x ∈ A)) : ℕ) = 24 := 
sorry

end num_elements_in_B_inter_C_l3_3291


namespace sum_geometric_series_l3_3082

noncomputable def S_n (n : ℕ) : ℝ :=
  3 - 3 * ((2 / 3)^n)

theorem sum_geometric_series (a : ℝ) (r : ℝ) (n : ℕ) (h_a : a = 1) (h_r : r = 2 / 3) :
  S_n n = a * (1 - r^n) / (1 - r) :=
by
  sorry

end sum_geometric_series_l3_3082


namespace find_quotient_l3_3521

-- Definitions for the variables and conditions
variables (D d q r : ℕ)

-- Conditions
axiom eq1 : D = q * d + r
axiom eq2 : D + 65 = q * (d + 5) + r

-- Theorem statement
theorem find_quotient : q = 13 :=
by
  sorry

end find_quotient_l3_3521


namespace sin_330_l3_3904

theorem sin_330 : Real.sin (330 * Real.pi / 180) = -1 / 2 := 
by
  -- Outline the proof here without providing it
  -- sorry to delay the proof
  sorry

end sin_330_l3_3904


namespace price_of_silver_l3_3820

theorem price_of_silver
  (side : ℕ) (side_eq : side = 3)
  (weight_per_cubic_inch : ℕ) (weight_per_cubic_inch_eq : weight_per_cubic_inch = 6)
  (selling_price : ℝ) (selling_price_eq : selling_price = 4455)
  (markup_percentage : ℝ) (markup_percentage_eq : markup_percentage = 1.10)
  : 4050 / 162 = 25 :=
by
  -- Given conditions are side_eq, weight_per_cubic_inch_eq, selling_price_eq, and markup_percentage_eq
  -- The statement requiring proof, i.e., price per ounce calculation, is provided.
  sorry

end price_of_silver_l3_3820


namespace largest_corner_sum_l3_3001

noncomputable def sum_faces (cube : ℕ → ℕ) : Prop :=
  cube 1 + cube 7 = 8 ∧ 
  cube 2 + cube 6 = 8 ∧ 
  cube 3 + cube 5 = 8 ∧ 
  cube 4 + cube 4 = 8

theorem largest_corner_sum (cube : ℕ → ℕ) 
  (h : sum_faces cube) : 
  ∃ n, n = 17 ∧ 
  ∀ a b c, (a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ 
            (cube a = 7 ∧ cube b = 6 ∧ cube c = 4 ∨ 
             cube a = 6 ∧ cube b = 4 ∧ cube c = 7 ∨ 
             cube a = 4 ∧ cube b = 7 ∧ cube c = 6)) → 
            a + b + c = n := sorry

end largest_corner_sum_l3_3001


namespace value_of_f_5_l3_3503

noncomputable def f (a b : ℝ) (x : ℝ) : ℝ := a * x^3 - b * Real.sin x - 2

theorem value_of_f_5 (a b : ℝ) (hf : f a b (-5) = 17) : f a b 5 = -21 := by
  sorry

end value_of_f_5_l3_3503


namespace unused_square_is_teal_l3_3164

-- Define the set of colors
inductive Color
| Cyan
| Magenta
| Lime
| Purple
| Teal
| Silver
| Violet

open Color

-- Define the condition that Lime is opposite Purple in the cube
def opposite (a b : Color) : Prop :=
  (a = Lime ∧ b = Purple) ∨ (a = Purple ∧ b = Lime)

-- Define the problem: seven squares are colored and one color remains unused.
def seven_squares_set (hinge : List Color) : Prop :=
  hinge.length = 6 ∧ 
  opposite Lime Purple ∧
  Color.Cyan ∈ hinge ∧
  Color.Magenta ∈ hinge ∧ 
  Color.Lime ∈ hinge ∧ 
  Color.Purple ∈ hinge ∧ 
  Color.Teal ∈ hinge ∧ 
  Color.Silver ∈ hinge ∧ 
  Color.Violet ∈ hinge

theorem unused_square_is_teal :
  ∃ hinge : List Color, seven_squares_set hinge ∧ ¬ (Teal ∈ hinge) := 
by sorry

end unused_square_is_teal_l3_3164


namespace ratio_twice_width_to_length_l3_3005

theorem ratio_twice_width_to_length (L W : ℝ) (k : ℤ)
  (h1 : L = 24)
  (h2 : W = 13.5)
  (h3 : L = k * W - 3) :
  2 * W / L = 9 / 8 := by
  sorry

end ratio_twice_width_to_length_l3_3005


namespace value_of_a_l3_3263

-- Definitions of sets A and B
def A : Set ℝ := {x | x^2 = 1}
def B (a : ℝ) : Set ℝ := {x | a * x = 1}

-- The main theorem statement
theorem value_of_a (a : ℝ) (H : B a ⊆ A) : a = 0 ∨ a = 1 ∨ a = -1 :=
by 
  sorry

end value_of_a_l3_3263


namespace airline_route_within_republic_l3_3444

theorem airline_route_within_republic (cities : Finset α) (republics : Finset (Finset α))
  (routes : α → Finset α) (h_cities : cities.card = 100) (h_republics : republics.card = 3)
  (h_partition : ∀ r ∈ republics, disjoint r (Finset.univ \ r) ∧ Finset.univ = r ∪ (Finset.univ \ r))
  (h_millionaire : ∃ m ∈ cities, 70 ≤ (routes m).card) :
  ∃ c1 c2 ∈ cities, ∃ r ∈ republics, (routes c1).member c2 ∧ c1 ≠ c2 ∧ c1 ∈ r ∧ c2 ∈ r :=
by sorry

end airline_route_within_republic_l3_3444


namespace sum_m_n_eq_zero_l3_3715

theorem sum_m_n_eq_zero (m n p : ℝ) (h1 : m * n + p^2 + 4 = 0) (h2 : m - n = 4) : m + n = 0 := 
  sorry

end sum_m_n_eq_zero_l3_3715


namespace find_f_2_pow_2011_l3_3612

noncomputable def f (x : ℝ) : ℝ := sorry

axiom f_positive (x : ℝ) : x > 0 → f x > 0

axiom f_initial_condition : f 1 + f 2 = 10

axiom f_functional_equation (a b : ℝ) : f a + f b = f (a+b) - 2 * Real.sqrt (f a * f b)

theorem find_f_2_pow_2011 : f (2^2011) = 2^4023 := 
by 
  sorry

end find_f_2_pow_2011_l3_3612


namespace k_value_for_z_perfect_square_l3_3785

theorem k_value_for_z_perfect_square (Z K : ℤ) (h1 : 500 < Z ∧ Z < 1000) (h2 : K > 1) (h3 : Z = K * K^2) :
  ∃ K : ℤ, Z = 729 ∧ K = 9 :=
by {
  sorry
}

end k_value_for_z_perfect_square_l3_3785


namespace quadrilateral_area_is_8_l3_3712

noncomputable section
open Real

def f1 : ℝ × ℝ := (-2, 0)
def f2 : ℝ × ℝ := (2, 0)

def ellipse (x y : ℝ) : Prop := x^2 / 16 + y^2 / 4 = 1

def origin_symmetric (P Q : ℝ × ℝ) : Prop := P.1 = -Q.1 ∧ P.2 = -Q.2

def distance (A B : ℝ × ℝ) : ℝ := sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)

def is_quadrilateral (P Q F1 F2 : ℝ × ℝ) : Prop :=
  ∃ a b c d, a = P ∧ b = F1 ∧ c = Q ∧ d = F2

def area_of_quadrilateral (A B C D : ℝ × ℝ) : ℝ :=
  0.5 * abs (A.1*B.2 + B.1*C.2 + C.1*D.2 + D.1*A.2 - (B.1*A.2 + C.1*B.2 + D.1*C.2 + A.1*D.2))

theorem quadrilateral_area_is_8 (P Q : ℝ × ℝ) :
  ellipse P.1 P.2 →
  ellipse Q.1 Q.2 →
  origin_symmetric P Q →
  distance P Q = distance f1 f2 →
  is_quadrilateral P Q f1 f2 →
  area_of_quadrilateral P f1 Q f2 = 8 := 
by
  sorry

end quadrilateral_area_is_8_l3_3712


namespace unique_solution_to_function_equation_l3_3245

theorem unique_solution_to_function_equation (f : ℕ → ℕ)
  (h1 : ∀ n : ℕ, f (2 * n) = 2 * f n)
  (h2 : ∀ n : ℕ, f (2 * n + 1) = 2 * f n + 1) :
  ∀ n : ℕ, f n = n :=
by
  sorry

end unique_solution_to_function_equation_l3_3245


namespace fraction_product_simplified_l3_3687

theorem fraction_product_simplified:
  (2 / 3) * (4 / 7) * (9 / 11) = 24 / 77 := by
  sorry

end fraction_product_simplified_l3_3687


namespace sum_of_squares_consecutive_nat_l3_3325

theorem sum_of_squares_consecutive_nat (n : ℕ) (h : n = 26) : (n - 1) ^ 2 + n ^ 2 + (n + 1) ^ 2 = 2030 :=
by
  sorry

end sum_of_squares_consecutive_nat_l3_3325


namespace sin_330_eq_neg_half_l3_3840

theorem sin_330_eq_neg_half 
  (h1: (330 : ℝ) = 330)
  (h2: (330 : ℝ = 360 - 30))
  (h3: 30 ∈ reference_angle_set ∧ sin 30 = 1/2)
  (h4: 330 ∈ fourth_quadrant)
  (h5: ∀ θ ∈ fourth_quadrant, sin θ < 0) :
  sin 330 = -1/2 := 
by
  sorry

end sin_330_eq_neg_half_l3_3840


namespace sin_330_l3_3927

theorem sin_330 : Real.sin (330 * Real.pi / 180) = -1 / 2 := 
by
  sorry

end sin_330_l3_3927


namespace find_number_that_gives_200_9_when_8_036_divided_by_it_l3_3356

theorem find_number_that_gives_200_9_when_8_036_divided_by_it (
  x : ℝ
) : (8.036 / x = 200.9) → (x = 0.04) :=
by
  intro h
  sorry

end find_number_that_gives_200_9_when_8_036_divided_by_it_l3_3356


namespace thirty_percent_of_x_l3_3204

noncomputable def x : ℝ := 160 / 0.40

theorem thirty_percent_of_x (h : 0.40 * x = 160) : 0.30 * x = 120 :=
sorry

end thirty_percent_of_x_l3_3204


namespace degree_of_p_is_unbounded_l3_3534

theorem degree_of_p_is_unbounded (p : Polynomial ℝ) (h : ∀ x : ℝ, p.eval (x^2 - 1) = (p.eval x) * (p.eval (-x))) : False :=
sorry

end degree_of_p_is_unbounded_l3_3534


namespace sheep_count_l3_3055

theorem sheep_count (S H : ℕ) (h1 : S / H = 2 / 7) (h2 : H * 230 = 12880) : S = 16 :=
by 
  -- Lean proof goes here
  sorry

end sheep_count_l3_3055


namespace maximum_ratio_l3_3588

-- Define the conditions
def is_two_digit_positive_integer (n : ℕ) : Prop :=
  10 ≤ n ∧ n ≤ 99

def mean_is_45 (x y : ℕ) : Prop :=
  (x + y) / 2 = 45

-- State the theorem
theorem maximum_ratio (x y : ℕ) (hx : is_two_digit_positive_integer x) (hy : is_two_digit_positive_integer y) (h_mean : mean_is_45 x y) : 
  ∃ (k: ℕ), (x / y = k) ∧ k = 8 :=
sorry

end maximum_ratio_l3_3588


namespace eval_f_neg2_l3_3045

-- Define the function f
def f (x : ℤ) : ℤ := x^2 - 3*x + 1

-- Theorem statement
theorem eval_f_neg2 : f (-2) = 11 := by
  sorry

end eval_f_neg2_l3_3045


namespace friends_contribution_l3_3571

theorem friends_contribution (x : ℝ) 
  (h1 : 4 * (x - 5) = 0.75 * 4 * x) : 
  0.75 * 4 * x = 60 :=
by 
  sorry

end friends_contribution_l3_3571


namespace circle_radius_eq_one_l3_3417

theorem circle_radius_eq_one (x y : ℝ) : (16 * x^2 - 32 * x + 16 * y^2 + 64 * y + 64 = 0) → (1 = 1) :=
by
  intros h
  sorry

end circle_radius_eq_one_l3_3417


namespace sin_330_degree_l3_3947

theorem sin_330_degree : Real.sin (330 * Real.pi / 180) = -1 / 2 :=
by
  sorry

end sin_330_degree_l3_3947


namespace mixed_groups_count_l3_3024

/-- Define the initial conditions --/
def number_of_children : ℕ := 300
def number_of_groups : ℕ := 100
def group_size : ℕ := 3
def boy_boy_photos : ℕ := 100
def girl_girl_photos : ℕ := 56

/-- Define the proof problem -/
theorem mixed_groups_count : 
    (number_of_children = 300) →
    (number_of_groups = 100) →
    (group_size = 3) →
    (boy_boy_photos = 100) →
    (girl_girl_photos = 56) →
    (∀ total_photos, total_photos = number_of_groups * group_size) →
    (∃ mixed_groups, mixed_groups = (total_photos - boy_boy_photos - girl_girl_photos) / 2) →
    mixed_groups = 72 :=
by
    intros h1 h2 h3 h4 h5 h6 h7
    sorry

end mixed_groups_count_l3_3024


namespace trigonometric_identity_l3_3264

theorem trigonometric_identity (α : ℝ) (h : Real.tan α = 2) :
  (Real.sin α + Real.cos α) / (2 * Real.sin α - 3 * Real.cos α) = 3 := 
by 
  sorry

end trigonometric_identity_l3_3264


namespace length_of_each_lateral_edge_l3_3564

-- Define the concept of a prism with a certain number of vertices and lateral edges
structure Prism where
  vertices : ℕ
  lateral_edges : ℕ

-- Example specific to the problem: Define the conditions given in the problem statement
def given_prism : Prism := { vertices := 12, lateral_edges := 6 }
def sum_lateral_edges : ℕ := 30

-- The main proof statement: Prove the length of each lateral edge
theorem length_of_each_lateral_edge (p : Prism) (h : p = given_prism) :
  (sum_lateral_edges / p.lateral_edges) = 5 :=
by 
  -- The details of the proof will replace 'sorry'
  sorry

end length_of_each_lateral_edge_l3_3564


namespace angle_measure_supplement_complement_l3_3633

theorem angle_measure_supplement_complement (x : ℝ) 
    (h1 : 180 - x = 7 * (90 - x)) : 
    x = 75 := by
  sorry

end angle_measure_supplement_complement_l3_3633


namespace rightmost_four_digits_of_5_pow_2023_l3_3335

theorem rightmost_four_digits_of_5_pow_2023 :
  5 ^ 2023 % 5000 = 3125 :=
  sorry

end rightmost_four_digits_of_5_pow_2023_l3_3335


namespace polynomials_exist_l3_3280

theorem polynomials_exist (p : ℕ) (hp : Nat.Prime p) :
  ∃ (P Q : Polynomial ℤ),
  ¬(Polynomial.degree P = 0) ∧ ¬(Polynomial.degree Q = 0) ∧
  (∀ n, (Polynomial.coeff (P * Q) n).natAbs % p =
    if n = 0 then 1
    else if n = 4 then 1
    else if n = 2 then p - 2
    else 0) :=
sorry

end polynomials_exist_l3_3280


namespace minimum_value_of_x_is_4_l3_3730

-- Given conditions
variable {x : ℝ} (hx_pos : 0 < x) (h : log x ≥ log 2 + 1/2 * log x)

-- The minimum value of x is 4
theorem minimum_value_of_x_is_4 : x ≥ 4 :=
by
  sorry

end minimum_value_of_x_is_4_l3_3730


namespace scientific_notation_of_100000000_l3_3620

theorem scientific_notation_of_100000000 :
  100000000 = 1 * 10^8 :=
sorry

end scientific_notation_of_100000000_l3_3620


namespace nancy_antacids_l3_3592

theorem nancy_antacids :
  ∀ (x : ℕ),
  (3 * 3 + x * 2 + 1 * 2) * 4 = 60 → x = 2 :=
by
  sorry

end nancy_antacids_l3_3592


namespace difference_of_two_numbers_l3_3310

-- Definitions as per conditions
def L : ℕ := 1656
def S : ℕ := 273
def quotient : ℕ := 6
def remainder : ℕ := 15

-- Statement of the proof problem
theorem difference_of_two_numbers (h1 : L = 6 * S + 15) : L - S = 1383 :=
by sorry

end difference_of_two_numbers_l3_3310


namespace age_difference_of_siblings_l3_3184

theorem age_difference_of_siblings (x : ℝ) 
  (h1 : 19 * x + 20 = 230) :
  |4 * x - 3 * x| = 210 / 19 := by
    sorry

end age_difference_of_siblings_l3_3184


namespace butterfly_black_dots_l3_3187

theorem butterfly_black_dots (b f : ℕ) (total_butterflies : b = 397) (total_black_dots : f = 4764) : f / b = 12 :=
by
  sorry

end butterfly_black_dots_l3_3187


namespace sin_330_is_minus_sqrt3_over_2_l3_3993

-- Define the angle theta as 330 degrees
def theta := 330 * Real.toRad

-- State that point corresponding to 330 degrees lies in the fourth quadrant
def point_in_fourth_quadrant (θ : ℝ) := 330 * Real.toRad = θ ∧ θ > 270 * Real.toRad ∧ θ < 360 * Real.toRad

-- Define a proof problem that states computing sin of 330° equals -√3/2
theorem sin_330_is_minus_sqrt3_over_2 : point_in_fourth_quadrant theta → Real.sin theta = -Real.sqrt 3 / 2 := by
  sorry

end sin_330_is_minus_sqrt3_over_2_l3_3993


namespace relationship_among_a_b_c_l3_3122

variable (x y : ℝ)
variable (hx_pos : x > 0) (hy_pos : y > 0) (hxy_ne : x ≠ y)

noncomputable def a := (x + y) / 2
noncomputable def b := Real.sqrt (x * y)
noncomputable def c := 2 / ((1 / x) + (1 / y))

theorem relationship_among_a_b_c :
    a > b ∧ b > c := by
    sorry

end relationship_among_a_b_c_l3_3122


namespace bricks_needed_for_wall_l3_3494

noncomputable def brick_volume (length width height : ℝ) : ℝ :=
  length * width * height

noncomputable def wall_volume (length height thickness : ℝ) : ℝ :=
  length * height * thickness

theorem bricks_needed_for_wall :
  let length_wall := 800
  let height_wall := 600
  let thickness_wall := 22.5
  let length_brick := 100
  let width_brick := 11.25
  let height_brick := 6
  let vol_wall := wall_volume length_wall height_wall thickness_wall
  let vol_brick := brick_volume length_brick width_brick height_brick
  vol_wall / vol_brick = 1600 :=
by
  sorry

end bricks_needed_for_wall_l3_3494


namespace eval_power_l3_3089

theorem eval_power (h : 81 = 3^4) : 81^(5/4) = 243 := by
  sorry

end eval_power_l3_3089


namespace chess_team_boys_count_l3_3807

theorem chess_team_boys_count : 
  ∃ (B G : ℕ), B + G = 30 ∧ (2 / 3 : ℚ) * G + B = 18 ∧ B = 6 := by
  sorry

end chess_team_boys_count_l3_3807


namespace g_f_x_not_quadratic_l3_3129

open Real

noncomputable def f : ℝ → ℝ := sorry
noncomputable def g : ℝ → ℝ := sorry

theorem g_f_x_not_quadratic (h : ∃ x : ℝ, x - f (g x) = 0) :
  ∀ x : ℝ, g (f x) ≠ x^2 + x + 1 / 5 := sorry

end g_f_x_not_quadratic_l3_3129


namespace quadratic_two_distinct_real_roots_l3_3136

theorem quadratic_two_distinct_real_roots (k : ℝ) : 
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ x^2 - 6 * x + k = 0) ↔ k < 9 :=
by
  sorry

end quadratic_two_distinct_real_roots_l3_3136


namespace sin_330_eq_neg_one_half_l3_3915

theorem sin_330_eq_neg_one_half :
  ∃ θ : ℝ, sin (330 * real.pi / 180) = -1 / 2 :=
begin
  have h1 : sin (360 * real.pi / 180 - 30 * real.pi / 180) = -sin (30 * real.pi / 180),
  { rw sin_sub,
    ring,
    rw sin_pi_div_six, -- sin 30° = 1 / 2
  },
  have h2 : sin (30 * real.pi / 180) = 1 / 2 := by exact real.sin_pi_div_six,
  use 330,
  calc
    sin (330 * real.pi / 180)
        = sin ((360 - 30) * real.pi / 180)      : by ring
    ... = sin (360 * real.pi / 180 - 30 * real.pi / 180) : by congr' 1; ring
    ... = -sin (30 * real.pi / 180)            : by exact h1
    ... = -1 / 2                               : by rw h2,
end

end sin_330_eq_neg_one_half_l3_3915


namespace jelly_bean_probability_l3_3661

theorem jelly_bean_probability :
  ∀ (P_red P_orange P_green P_yellow : ℝ),
  P_red = 0.1 →
  P_orange = 0.4 →
  P_green = 0.2 →
  P_red + P_orange + P_green + P_yellow = 1 →
  P_yellow = 0.3 :=
by
  intros P_red P_orange P_green P_yellow h_red h_orange h_green h_sum
  sorry

end jelly_bean_probability_l3_3661


namespace tilly_star_count_l3_3039

theorem tilly_star_count (stars_east : ℕ) (stars_west : ℕ) (total_stars : ℕ) 
  (h1 : stars_east = 120)
  (h2 : stars_west = 6 * stars_east)
  (h3 : total_stars = stars_east + stars_west) :
  total_stars = 840 :=
sorry

end tilly_star_count_l3_3039


namespace percentage_increase_l3_3558

variable (S : ℝ) (P : ℝ)
variable (h1 : S + 0.10 * S = 330)
variable (h2 : S + P * S = 324)

theorem percentage_increase : P = 0.08 := sorry

end percentage_increase_l3_3558


namespace counterfeit_probability_correct_l3_3374

noncomputable def calc_probability (
  P_C : ℝ,
  P_R : ℝ,
  P_L : ℝ,
  P_L_C : ℝ,
  P_L_R : ℝ,
  P_T_counterfeit : ℝ,
  P_T_real : ℝ
) : ℝ :=
  let P_T := P_T_counterfeit * P_C * P_L_C + P_T_real * P_R * P_L_R in
  (P_T_counterfeit * P_C) / P_T

theorem counterfeit_probability_correct :
  calc_probability
    (1 / 100)     -- P(C)
    (99 / 100)    -- P(R)
    0.05          -- P(L)
    1             -- P(L | C)
    0.05          -- P(L | R)
    0.90          -- P(T | counterfeit)
    0.10          -- P(T | real)
  = 19 / 28 :=
sorry

end counterfeit_probability_correct_l3_3374


namespace soccer_goal_difference_l3_3738

theorem soccer_goal_difference (n : ℕ) (h : n = 2020) :
  ¬ ∃ g : Fin n → ℤ,
    (∀ i j : Fin n, i < j → (g i < g j)) ∧ 
    (∀ i : Fin n, ∃ x y : ℕ, x + y = n - 1 ∧ 3 * x = (n - 1 - x) ∧ g i = x - y) :=
by
  sorry

end soccer_goal_difference_l3_3738


namespace polynomial_divisibility_l3_3418

theorem polynomial_divisibility (A B : ℝ)
  (h: ∀ (x : ℂ), x^2 + x + 1 = 0 → x^104 + A * x^3 + B * x = 0) :
  A + B = 0 :=
by
  sorry

end polynomial_divisibility_l3_3418


namespace ratio_ashley_mary_l3_3773

-- Definitions based on conditions
def sum_ages (A M : ℕ) := A + M = 22
def ashley_age (A : ℕ) := A = 8

-- Theorem stating the ratio of Ashley's age to Mary's age
theorem ratio_ashley_mary (A M : ℕ) 
  (h1 : sum_ages A M)
  (h2 : ashley_age A) : 
  (A : ℚ) / (M : ℚ) = 4 / 7 :=
by
  -- Skipping the proof as specified
  sorry

end ratio_ashley_mary_l3_3773


namespace sin_330_eq_neg_sin_30_l3_3980

theorem sin_330_eq_neg_sin_30 :
  sin (330 : ℝ) = - sin (30 : ℝ) := sorry

end sin_330_eq_neg_sin_30_l3_3980


namespace sin_330_deg_l3_3873

noncomputable theory

open Real

theorem sin_330_deg :
  sin (330 * (π / 180)) = -1 / 2 :=
by sorry

end sin_330_deg_l3_3873


namespace no_negative_roots_l3_3752

theorem no_negative_roots (x : ℝ) : 4 * x^4 - 7 * x^3 - 20 * x^2 - 13 * x + 25 ≠ 0 ∨ x ≥ 0 := 
sorry

end no_negative_roots_l3_3752


namespace six_digit_number_l3_3365

theorem six_digit_number : ∃ x : ℕ, 100000 ≤ x ∧ x < 1000000 ∧ 3 * x = (x - 300000) * 10 + 3 ∧ x = 428571 :=
by
sorry

end six_digit_number_l3_3365


namespace numberOfSolutions_l3_3246

noncomputable def numberOfRealPositiveSolutions(x : ℝ) : Prop := 
  (x^6 + 1) * (x^4 + x^2 + 1) = 6 * x^5

theorem numberOfSolutions : ∃! x : ℝ, numberOfRealPositiveSolutions x := 
by
  sorry

end numberOfSolutions_l3_3246


namespace correct_equations_l3_3760

variable (x y : ℝ)

theorem correct_equations :
  (18 * x = y + 3) ∧ (17 * x = y - 4) ↔ (18 * x = y + 3) ∧ (17 * x = y - 4) :=
by
  sorry

end correct_equations_l3_3760


namespace sin_330_deg_l3_3847

theorem sin_330_deg : Real.sin (330 * Real.pi / 180) = -1 / 2 := by
  sorry

end sin_330_deg_l3_3847


namespace inequality_condition_l3_3668

-- Define the inequality (x - 2) * (x + 2) > 0
def inequality_holds (x : ℝ) : Prop := (x - 2) * (x + 2) > 0

-- The sufficient and necessary condition for the inequality to hold is x > 2 or x < -2
theorem inequality_condition (x : ℝ) : inequality_holds x ↔ (x > 2 ∨ x < -2) :=
  sorry

end inequality_condition_l3_3668


namespace range_of_a_l3_3265

theorem range_of_a (x a : ℝ) (h1 : -2 < x) (h2 : x ≤ 1) (h3 : |x - 2| < a) : a ≤ 0 :=
sorry

end range_of_a_l3_3265


namespace range_of_ab_l3_3716

theorem range_of_ab (a b : ℝ) 
  (h1: ∀ x y : ℝ, x^2 + y^2 + 2 * x - 4 * y + 1 = 0 → (2 * a * x - b * y + 2 = 0)) : 
  ab ≤ 0 :=
sorry

end range_of_ab_l3_3716


namespace sin_330_degree_l3_3953

theorem sin_330_degree : Real.sin (330 * Real.pi / 180) = -1 / 2 :=
by
  sorry

end sin_330_degree_l3_3953


namespace smallest_possible_value_of_N_l3_3666

noncomputable def smallest_N (N : ℕ) : Prop :=
  ∃ l m n : ℕ, l * m * n = N ∧ (l - 1) * (m - 1) * (n - 1) = 378

theorem smallest_possible_value_of_N : smallest_N 560 :=
  by {
    sorry
  }

end smallest_possible_value_of_N_l3_3666


namespace fraction_of_loss_l3_3815

theorem fraction_of_loss
  (SP CP : ℚ) (hSP : SP = 16) (hCP : CP = 17) :
  (CP - SP) / CP = 1 / 17 :=
by
  sorry

end fraction_of_loss_l3_3815


namespace sin_330_eq_neg_half_l3_3918

open Real

theorem sin_330_eq_neg_half : sin (330 * (π / 180)) = - (1 / 2) := by
  have angle_eq : 330 * (π / 180) = 2 * π - (30 * (π / 180)) := by sorry
  have sin_30_pos : sin (30 * (π / 180)) = 1 / 2 := by sorry
  have sin_330_neg : sin (2 * π - (30 * (π / 180))) = - (sin (30 * (π / 180))) := by sorry
  rw [angle_eq, sin_330_neg, sin_30_pos]
  rfl

end sin_330_eq_neg_half_l3_3918


namespace total_points_of_players_l3_3570

variables (Samanta Mark Eric Daisy Jake : ℕ)
variables (h1 : Samanta = Mark + 8)
variables (h2 : Mark = 3 / 2 * Eric)
variables (h3 : Eric = 6)
variables (h4 : Daisy = 3 / 4 * (Samanta + Mark + Eric))
variables (h5 : Jake = Samanta - Eric)
 
theorem total_points_of_players :
  Samanta + Mark + Eric + Daisy + Jake = 67 :=
sorry

end total_points_of_players_l3_3570


namespace intersect_three_points_l3_3152

noncomputable def f (x : ℝ) : ℝ := (Real.cos x) / x
noncomputable def g (x : ℝ) (a : ℝ) : ℝ := Real.sin x + a * x

theorem intersect_three_points (a : ℝ) :
  (∃ (t1 t2 t3 : ℝ), t1 > 0 ∧ t2 > 0 ∧ t3 > 0 ∧ t1 ≠ t2 ∧ t2 ≠ t3 ∧ t1 ≠ t3 ∧ 
    f t1 = g t1 a ∧ f t2 = g t2 a ∧ f t3 = g t3 a) ↔ 
  a ∈ Set.Ioo (2 / (7 * Real.pi)) (2 / (3 * Real.pi)) ∨ a = -2 / (5 * Real.pi) :=
sorry

end intersect_three_points_l3_3152


namespace exists_route_within_republic_l3_3443

theorem exists_route_within_republic :
  ∃ (cities : Finset ℕ) (republics : Finset (Finset ℕ)),
    (Finset.card cities = 100) ∧
    (by ∃ (R1 R2 R3 : Finset ℕ), R1 ∪ R2 ∪ R3 = cities ∧ Finset.card R1 ≤ 30 ∧ Finset.card R2 ≤ 30 ∧ Finset.card R3 ≤ 30) ∧
    (∃ (millionaire_cities : Finset ℕ), Finset.card millionaire_cities ≥ 70 ∧ ∀ city ∈ millionaire_cities, ∃ routes : Finset ℕ, Finset.card routes ≥ 70 ∧ routes ⊆ cities) →
  ∃ (city1 city2 : ℕ) (republic : Finset ℕ), city1 ∈ republic ∧ city2 ∈ republic ∧
    city1 ≠ city2 ∧ (∃ route : ℕ, route = (city1, city2) ∨ route = (city2, city1)) :=
sorry

end exists_route_within_republic_l3_3443


namespace brian_tape_needed_l3_3079

-- Definitions of conditions
def tape_needed_for_box (short_side: ℕ) (long_side: ℕ) : ℕ := 
  2 * short_side + long_side

def total_tape_needed (num_short_long_boxes: ℕ) (short_side: ℕ) (long_side: ℕ) (num_square_boxes: ℕ) (side: ℕ) : ℕ := 
  (num_short_long_boxes * tape_needed_for_box short_side long_side) + (num_square_boxes * 3 * side)

-- Theorem statement
theorem brian_tape_needed : total_tape_needed 5 15 30 2 40 = 540 := 
by 
  sorry

end brian_tape_needed_l3_3079


namespace multiply_and_simplify_fractions_l3_3683

theorem multiply_and_simplify_fractions :
  (2 / 3) * (4 / 7) * (9 / 11) = 24 / 77 := 
by
  sorry

end multiply_and_simplify_fractions_l3_3683


namespace great_eighteen_hockey_league_games_l3_3772

theorem great_eighteen_hockey_league_games :
  (let teams_per_division := 9
   let games_intra_division_per_team := 8 * 3
   let games_inter_division_per_team := teams_per_division * 2
   let total_games_per_team := games_intra_division_per_team + games_inter_division_per_team
   let total_game_instances := 18 * total_games_per_team
   let unique_games := total_game_instances / 2
   unique_games = 378) :=
by
  sorry

end great_eighteen_hockey_league_games_l3_3772


namespace jordan_walk_distance_l3_3579

theorem jordan_walk_distance
  (d t : ℝ)
  (flat_speed uphill_speed walk_speed : ℝ)
  (total_time : ℝ)
  (h1 : flat_speed = 18)
  (h2 : uphill_speed = 6)
  (h3 : walk_speed = 4)
  (h4 : total_time = 3)
  (h5 : d / (3 * 18) + d / (3 * 6) + d / (3 * 4) = total_time) :
  t = 6.6 :=
by
  -- Proof goes here
  sorry

end jordan_walk_distance_l3_3579


namespace no_triples_of_consecutive_numbers_l3_3059

theorem no_triples_of_consecutive_numbers (n : ℤ) (a : ℕ) (h : 1 ≤ a ∧ a ≤ 9) :
  ¬(3 * n^2 + 2 = 1111 * a) :=
by sorry

end no_triples_of_consecutive_numbers_l3_3059


namespace abs_neg_one_div_three_l3_3308

open Real

theorem abs_neg_one_div_three : abs (-1 / 3) = 1 / 3 :=
by
  sorry

end abs_neg_one_div_three_l3_3308


namespace students_with_all_three_pets_correct_l3_3446

noncomputable def students_with_all_three_pets (total_students dog_owners cat_owners bird_owners dog_and_cat_owners cat_and_bird_owners dog_and_bird_owners : ℕ) : ℕ :=
  total_students - (dog_owners + cat_owners + bird_owners - dog_and_cat_owners - cat_and_bird_owners - dog_and_bird_owners)

theorem students_with_all_three_pets_correct : 
  students_with_all_three_pets 50 30 35 10 8 5 3 = 7 :=
by
  rw [students_with_all_three_pets]
  norm_num
  sorry

end students_with_all_three_pets_correct_l3_3446


namespace Mary_age_is_10_l3_3753

-- Define the parameters for the ages of Rahul and Mary
variables (Rahul Mary : ℕ)

-- Conditions provided in the problem
def condition1 := Rahul = Mary + 30
def condition2 := Rahul + 20 = 2 * (Mary + 20)

-- Stating the theorem to be proved
theorem Mary_age_is_10 (Rahul Mary : ℕ) 
  (h1 : Rahul = Mary + 30) 
  (h2 : Rahul + 20 = 2 * (Mary + 20)) : 
  Mary = 10 :=
by 
  sorry

end Mary_age_is_10_l3_3753


namespace man_l3_3810

theorem man's_speed_upstream :
  ∀ (R : ℝ), (R + 1.5 = 11) → (R - 1.5 = 8) :=
by
  intros R h
  sorry

end man_l3_3810


namespace smallest_k_l3_3009

-- Define the non-decreasing property of digits in a five-digit number
def non_decreasing (n : Fin 5 → ℕ) : Prop :=
  n 0 ≤ n 1 ∧ n 1 ≤ n 2 ∧ n 2 ≤ n 3 ∧ n 3 ≤ n 4

-- Define the overlap property in at least one digit
def overlap (n1 n2 : Fin 5 → ℕ) : Prop :=
  ∃ i : Fin 5, n1 i = n2 i

-- The main theorem stating the problem
theorem smallest_k {N1 Nk : Fin 5 → ℕ} :
  (∀ n : Fin 5 → ℕ, non_decreasing n → overlap N1 n ∨ overlap Nk n) → 
  ∃ (k : Nat), k = 2 :=
sorry

end smallest_k_l3_3009


namespace Cary_walked_miles_round_trip_l3_3689

theorem Cary_walked_miles_round_trip : ∀ (m : ℕ), 
  150 * m - 200 = 250 → m = 3 := 
by
  intros m h
  sorry

end Cary_walked_miles_round_trip_l3_3689


namespace range_of_y_l3_3559

theorem range_of_y (y : ℝ) (h1 : y < 0) (h2 : Int.ceil y * Int.floor y = 72) : 
  -9 < y ∧ y < -8 :=
sorry

end range_of_y_l3_3559


namespace factorize_expression_l3_3698

theorem factorize_expression (a b : ℝ) : 2 * a^2 - 8 * b^2 = 2 * (a + 2 * b) * (a - 2 * b) :=
by sorry

end factorize_expression_l3_3698


namespace evaluate_81_power_5_div_4_l3_3098

-- Define the conditions
def base_factorized : ℕ := 3 ^ 4
def power_rule (b : ℕ) (m n : ℝ) : ℝ := (b : ℝ) ^ m ^ n

-- Define the primary calculation
noncomputable def power_calculation : ℝ := 81 ^ (5 / 4)

-- Prove that the calculation equals 243
theorem evaluate_81_power_5_div_4 : power_calculation = 243 := 
by
  have h1 : base_factorized = 81 := by sorry
  have h2 : power_rule 3 4 (5 / 4) = 3 ^ 5 := by sorry
  have h3 : 3 ^ 5 = 243 := by sorry
  have h4 : power_calculation = power_rule 3 4 (5 / 4) := by sorry
  rw [h1, h2, h3, h4]
  exact h3

end evaluate_81_power_5_div_4_l3_3098


namespace fraction_of_sand_is_one_third_l3_3359

noncomputable def total_weight : ℝ := 24
noncomputable def weight_of_water (total_weight : ℝ) : ℝ := total_weight / 4
noncomputable def weight_of_gravel : ℝ := 10
noncomputable def weight_of_sand (total_weight weight_of_water weight_of_gravel : ℝ) : ℝ :=
  total_weight - weight_of_water - weight_of_gravel
noncomputable def fraction_of_sand (weight_of_sand total_weight : ℝ) : ℝ :=
  weight_of_sand / total_weight

theorem fraction_of_sand_is_one_third :
  fraction_of_sand (weight_of_sand total_weight (weight_of_water total_weight) weight_of_gravel) total_weight
  = 1/3 := by
  sorry

end fraction_of_sand_is_one_third_l3_3359


namespace weight_of_one_baseball_l3_3699

structure Context :=
  (numberBaseballs : ℕ)
  (numberBicycles : ℕ)
  (weightBicycles : ℕ)
  (weightTotalBicycles : ℕ)

def problem (ctx : Context) :=
  ctx.weightTotalBicycles = ctx.numberBicycles * ctx.weightBicycles ∧
  ctx.numberBaseballs * ctx.weightBicycles = ctx.weightTotalBicycles →
  (ctx.weightTotalBicycles / ctx.numberBaseballs) = 8

theorem weight_of_one_baseball (ctx : Context) : problem ctx :=
sorry

end weight_of_one_baseball_l3_3699


namespace nearest_integer_to_expr_l3_3795

theorem nearest_integer_to_expr : 
  let a := 3 + Real.sqrt 5
  let b := (a)^6
  abs (b - 2744) < 1
:= sorry

end nearest_integer_to_expr_l3_3795


namespace correct_operation_l3_3198

theorem correct_operation :
  (∀ {a : ℝ}, a^6 / a^3 = a^3) = false ∧
  (∀ {a b : ℝ}, (a + b) * (a - b) = a^2 - b^2) ∧
  (∀ {a : ℝ}, (-a^3)^3 = -a^9) = false ∧
  (∀ {a : ℝ}, 2 * a^2 + 3 * a^3 = 5 * a^5) = false :=
by
  sorry

end correct_operation_l3_3198


namespace math_problem_l3_3824

theorem math_problem :
  ((-1)^2023 - (27^(1/3)) - (16^(1/2)) + (|1 - Real.sqrt 3|)) = -9 + Real.sqrt 3 :=
by
  sorry

end math_problem_l3_3824


namespace graph_single_point_c_eq_7_l3_3173

theorem graph_single_point_c_eq_7 (x y : ℝ) (c : ℝ) :
  (∃ p : ℝ × ℝ, ∀ x y : ℝ, 3 * x^2 + 4 * y^2 + 6 * x - 8 * y + c = 0 ↔ (x, y) = p) →
  c = 7 :=
by
  sorry

end graph_single_point_c_eq_7_l3_3173


namespace least_milk_l3_3597

theorem least_milk (seokjin jungkook yoongi : ℚ) (h_seokjin : seokjin = 11 / 10)
  (h_jungkook : jungkook = 1.3) (h_yoongi : yoongi = 7 / 6) :
  seokjin < jungkook ∧ seokjin < yoongi :=
by
  sorry

end least_milk_l3_3597


namespace sin_330_l3_3903

theorem sin_330 : Real.sin (330 * Real.pi / 180) = -1 / 2 := 
by
  -- Outline the proof here without providing it
  -- sorry to delay the proof
  sorry

end sin_330_l3_3903


namespace fraction_of_field_planted_l3_3140

theorem fraction_of_field_planted : 
  let field_area := 5 * 6
  let triangle_area := (5 * 6) / 2
  let a := (41 * 3) / 33  -- derived from the given conditions
  let square_area := a^2
  let planted_area := triangle_area - square_area
  (planted_area / field_area) = (404 / 841) := 
by
  sorry

end fraction_of_field_planted_l3_3140


namespace actual_distance_traveled_l3_3348

theorem actual_distance_traveled (T : ℝ) :
  ∀ D : ℝ, (D = 4 * T) → (D + 6 = 5 * T) → D = 24 :=
by
  intro D h1 h2
  sorry

end actual_distance_traveled_l3_3348


namespace nearest_integer_to_power_six_l3_3789

noncomputable def nearest_integer (x : ℝ) : ℤ := 
if x - real.floor x < real.ceil x - x then real.floor x else real.ceil x

theorem nearest_integer_to_power_six : 
  let x := 3 + real.sqrt 5 in
  let y := 3 - real.sqrt 5 in
  nearest_integer (x^6) = 2654 :=
by
  sorry

end nearest_integer_to_power_six_l3_3789


namespace quadratic_solution_property_l3_3459

theorem quadratic_solution_property :
  (∃ p q : ℝ, 3 * p^2 + 7 * p - 6 = 0 ∧ 3 * q^2 + 7 * q - 6 = 0 ∧ (p - 2) * (q - 2) = 6) :=
by
  sorry

end quadratic_solution_property_l3_3459


namespace eq_triangle_perimeter_l3_3607

theorem eq_triangle_perimeter (s : ℝ) (h : (s^2 * Real.sqrt 3) / 4 = 2 * s) : 
  3 * s = 8 * Real.sqrt 3 :=
by
  sorry

end eq_triangle_perimeter_l3_3607


namespace vernal_equinox_shadow_length_l3_3575

-- Lean 4 statement
theorem vernal_equinox_shadow_length :
  ∀ (a : ℕ → ℝ), (a 4 = 10.5) → (a 10 = 4.5) → 
  (∀ (n m : ℕ), a (n + 1) = a n + (a 2 - a 1)) → 
  a 7 = 7.5 :=
by
  intros a h_4 h_10 h_progression
  sorry

end vernal_equinox_shadow_length_l3_3575


namespace center_of_circle_is_2_1_l3_3654

-- Definition of the circle
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 - 4 * x - 2 * y - 5 = 0

-- Theorem stating the center of the circle
theorem center_of_circle_is_2_1 (x y : ℝ) (h : circle_eq x y) : (x, y) = (2, 1) := sorry

end center_of_circle_is_2_1_l3_3654


namespace no_rational_solution_l3_3163

theorem no_rational_solution :
  ¬ ∃ (x y z : ℚ), 
  x + y + z = 0 ∧ x^2 + y^2 + z^2 = 100 := sorry

end no_rational_solution_l3_3163


namespace count_dna_sequences_Rthea_l3_3596

-- Definition of bases
inductive Base | H | M | N | T

-- Function to check whether two bases can be adjacent on the same strand
def can_be_adjacent (x y : Base) : Prop :=
  match x, y with
  | Base.H, Base.M => False
  | Base.M, Base.H => False
  | Base.N, Base.T => False
  | Base.T, Base.N => False
  | _, _ => True

-- Function to count the number of valid sequences
noncomputable def count_valid_sequences : Nat := 12 * 7^4

-- Theorem stating the expected count of valid sequences
theorem count_dna_sequences_Rthea : count_valid_sequences = 28812 := by
  sorry

end count_dna_sequences_Rthea_l3_3596


namespace min_time_to_complete_tasks_l3_3199

-- Define the conditions as individual time durations for each task in minutes
def bed_making_time : ℕ := 3
def teeth_washing_time : ℕ := 4
def water_boiling_time : ℕ := 10
def breakfast_time : ℕ := 7
def dish_washing_time : ℕ := 1
def backpack_organizing_time : ℕ := 2
def milk_making_time : ℕ := 1

-- Define the total minimum time required to complete all tasks
def min_completion_time : ℕ := 18

-- A theorem stating that given the times for each task, the minimum completion time is 18 minutes
theorem min_time_to_complete_tasks :
  bed_making_time + teeth_washing_time + water_boiling_time + 
  breakfast_time + dish_washing_time + backpack_organizing_time + milk_making_time - 
  (bed_making_time + teeth_washing_time + backpack_organizing_time + milk_making_time) <=
  min_completion_time := by
  sorry

end min_time_to_complete_tasks_l3_3199


namespace scientific_notation_l3_3617

theorem scientific_notation (a n : ℝ) (h1 : 100000000 = a * 10^n) (h2 : 1 ≤ a) (h3 : a < 10) : 
  a = 1 ∧ n = 8 :=
by
  sorry

end scientific_notation_l3_3617


namespace amount_received_by_sam_l3_3497

def P : ℝ := 15000
def r : ℝ := 0.10
def n : ℝ := 2
def t : ℝ := 1

noncomputable def compoundInterest (P r n t : ℝ) : ℝ :=
  P * (1 + r / n) ^ (n * t)

theorem amount_received_by_sam : compoundInterest P r n t = 16537.50 := by
  sorry

end amount_received_by_sam_l3_3497


namespace first_class_circular_permutations_second_class_circular_permutations_l3_3702

section CircularPermutations

def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

noncomputable def perm_count (a b c : ℕ) : ℕ :=
  factorial (a + b + c) / (factorial a * factorial b * factorial c)

theorem first_class_circular_permutations : perm_count 2 2 4 / 8 = 52 := by
  sorry

theorem second_class_circular_permutations : perm_count 2 2 4 / 2 / 4 = 33 := by
  sorry

end CircularPermutations

end first_class_circular_permutations_second_class_circular_permutations_l3_3702


namespace angle_measure_l3_3638

theorem angle_measure (x : ℝ) : 
  (180 - x = 7 * (90 - x)) → x = 75 :=
by
  sorry

end angle_measure_l3_3638


namespace efficiency_percentage_l3_3362

-- Define the conditions
def E_A : ℚ := 1 / 23
noncomputable def days_B : ℚ := 299 / 10
def E_B : ℚ := 1 / days_B
def combined_efficiency : ℚ := 1 / 13

-- Define the main theorem
theorem efficiency_percentage : 
  E_A + E_B = combined_efficiency → 
  ((E_A / E_B) * 100) ≈ 1300 :=
by
  sorry

end efficiency_percentage_l3_3362


namespace multiply_and_simplify_fractions_l3_3685

theorem multiply_and_simplify_fractions :
  (2 / 3) * (4 / 7) * (9 / 11) = 24 / 77 := 
by
  sorry

end multiply_and_simplify_fractions_l3_3685


namespace findMonthlyIncome_l3_3754

-- Variables and conditions
variable (I : ℝ) -- Raja's monthly income
variable (saving : ℝ) (r1 r2 r3 r4 r5 : ℝ) -- savings and monthly percentages

-- Conditions
def condition1 : r1 = 0.45 := by sorry
def condition2 : r2 = 0.12 := by sorry
def condition3 : r3 = 0.08 := by sorry
def condition4 : r4 = 0.15 := by sorry
def condition5 : r5 = 0.10 := by sorry
def conditionSaving : saving = 5000 := by sorry

-- Define the main equation
def mainEquation (I : ℝ) (r1 r2 r3 r4 r5 saving : ℝ) : Prop :=
  (r1 * I) + (r2 * I) + (r3 * I) + (r4 * I) + (r5 * I) + saving = I

-- Main theorem to prove
theorem findMonthlyIncome (I : ℝ) (r1 r2 r3 r4 r5 saving : ℝ) 
  (h1 : r1 = 0.45) (h2 : r2 = 0.12) (h3 : r3 = 0.08) (h4 : r4 = 0.15) (h5 : r5 = 0.10) (hSaving : saving = 5000) :
  mainEquation I r1 r2 r3 r4 r5 saving → I = 50000 :=
  by sorry

end findMonthlyIncome_l3_3754


namespace two_beta_plus_alpha_eq_pi_div_two_l3_3113

theorem two_beta_plus_alpha_eq_pi_div_two
  (α β : ℝ) (hα1 : 0 < α) (hα2 : α < π / 2)
  (hβ1 : 0 < β) (hβ2 : β < π / 2)
  (h : Real.tan α + Real.tan β = 1 / Real.cos α) :
  2 * β + α = π / 2 :=
sorry

end two_beta_plus_alpha_eq_pi_div_two_l3_3113


namespace infinitely_many_singular_pairs_l3_3105

def largestPrimeFactor (n : ℕ) : ℕ := sorry -- definition of largest prime factor

def isSingularPair (p q : ℕ) : Prop :=
  p ≠ q ∧ ∀ (n : ℕ), n ≥ 2 → largestPrimeFactor n * largestPrimeFactor (n + 1) ≠ p * q

theorem infinitely_many_singular_pairs : ∃ (S : ℕ → (ℕ × ℕ)), ∀ i, isSingularPair (S i).1 (S i).2 :=
sorry

end infinitely_many_singular_pairs_l3_3105


namespace negation_proposition_l3_3179

-- Definitions based on the conditions
def original_proposition : Prop := ∃ x : ℝ, x^2 + 3*x + 2 < 0

-- Theorem requiring proof
theorem negation_proposition : (¬ original_proposition) = ∀ x : ℝ, x^2 + 3*x + 2 ≥ 0 :=
by
  sorry

end negation_proposition_l3_3179


namespace sin_330_eq_neg_half_l3_3941

noncomputable def Q : ℝ × ℝ := (Real.sqrt 3 / 2, -1 / 2)

theorem sin_330_eq_neg_half : Real.sin (330 * Real.pi / 180) = -1 / 2 :=
by
  have h1 : 330 * Real.pi / 180 = 11 * Real.pi / 6 := by sorry
  have coord_y : (Q.snd) = -1 / 2 := by sorry
  rw [h1, coord_y]
  sorry

end sin_330_eq_neg_half_l3_3941


namespace DansAgeCalculation_l3_3235

theorem DansAgeCalculation (D x : ℕ) (h1 : D = 8) (h2 : D + 20 = 7 * (D - x)) : x = 4 :=
by
  sorry

end DansAgeCalculation_l3_3235


namespace primes_eq_2_3_7_l3_3526

theorem primes_eq_2_3_7 (p : ℕ) (hp : Nat.Prime p) :
  (∃ x y : ℕ, 0 < x ∧ 0 < y ∧ x * (y^2 - p) + y * (x^2 - p) = 5 * p) ↔ p = 2 ∨ p = 3 ∨ p = 7 :=
by
  sorry

end primes_eq_2_3_7_l3_3526


namespace spadesuit_calculation_l3_3535

def spadesuit (x y : ℝ) : ℝ := (x + y) * (x - y)

theorem spadesuit_calculation : spadesuit 5 (spadesuit 3 2) = 0 :=
by
  sorry

end spadesuit_calculation_l3_3535


namespace kolya_time_segment_DE_l3_3145

-- Definitions representing the conditions
def time_petya_route : ℝ := 12  -- Petya takes 12 minutes
def time_kolya_route : ℝ := 12  -- Kolya also takes 12 minutes
def kolya_speed_factor : ℝ := 1.2

-- Proof problem: Prove that Kolya spends 1 minute traveling the segment D-E
theorem kolya_time_segment_DE 
    (v : ℝ)  -- Assume v is Petya's speed
    (time_petya_A_B_C : ℝ := time_petya_route)  
    (time_kolya_A_D_E_F_C : ℝ := time_kolya_route)
    (kolya_fast_factor : ℝ := kolya_speed_factor)
    : (time_petya_A_B_C / kolya_fast_factor - time_petya_A_B_C) / (2 / kolya_fast_factor) = 1 := 
by 
    sorry

end kolya_time_segment_DE_l3_3145


namespace exists_complex_on_line_y_eq_neg_x_l3_3733

open Complex

theorem exists_complex_on_line_y_eq_neg_x :
  ∃ (z : ℂ), ∃ (a b : ℝ), z = a + b * I ∧ b = -a :=
by
  use 1 - I
  use 1, -1
  sorry

end exists_complex_on_line_y_eq_neg_x_l3_3733


namespace sin_330_degree_l3_3951

theorem sin_330_degree : Real.sin (330 * Real.pi / 180) = -1 / 2 :=
by
  sorry

end sin_330_degree_l3_3951


namespace Maria_trip_time_l3_3160

/-- 
Given:
- Maria drove 80 miles on a freeway.
- Maria drove 20 miles on a rural road.
- Her speed on the rural road was half of her speed on the freeway.
- Maria spent 40 minutes driving on the rural road.

Prove that Maria's entire trip took 120 minutes.
-/ 
theorem Maria_trip_time
  (distance_freeway : ℕ)
  (distance_rural : ℕ)
  (rural_speed_ratio : ℕ → ℕ)
  (time_rural_minutes : ℕ) 
  (time_freeway : ℕ)
  (total_time : ℕ) 
  (speed_rural : ℕ)
  (speed_freeway : ℕ) 
  :
  distance_freeway = 80 ∧
  distance_rural = 20 ∧ 
  rural_speed_ratio (speed_freeway) = speed_rural ∧ 
  time_rural_minutes = 40 ∧
  time_rural_minutes = 20 / speed_rural ∧
  speed_freeway = 2 * speed_rural ∧
  time_freeway = distance_freeway / speed_freeway ∧
  total_time = time_rural_minutes + time_freeway → 
  total_time = 120 :=
by
  intros
  sorry

end Maria_trip_time_l3_3160


namespace gcf_7fact_8fact_l3_3414

-- Definitions based on the conditions
def factorial (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | (n + 1) => (n + 1) * factorial n

noncomputable def greatest_common_divisor (a b : ℕ) : ℕ :=
  Nat.gcd a b

-- Theorem statement
theorem gcf_7fact_8fact : greatest_common_divisor (factorial 7) (factorial 8) = 5040 := by
  sorry

end gcf_7fact_8fact_l3_3414


namespace volume_of_figure_eq_half_l3_3081

-- Define a cube data structure and its properties
structure Cube where
  edge_length : ℝ
  h_el : edge_length = 1

-- Define a function to calculate volume of the figure
noncomputable def volume_of_figure (c : Cube) : ℝ := sorry

-- Example cube
def example_cube : Cube := { edge_length := 1, h_el := rfl }

-- Theorem statement
theorem volume_of_figure_eq_half (c : Cube) : volume_of_figure c = 1 / 2 := by
  sorry

end volume_of_figure_eq_half_l3_3081


namespace basic_astrophysics_degrees_l3_3808

def budget_allocation : Nat := 100
def microphotonics_perc : Nat := 14
def home_electronics_perc : Nat := 19
def food_additives_perc : Nat := 10
def genetically_modified_perc : Nat := 24
def industrial_lubricants_perc : Nat := 8

def arc_of_sector (percentage : Nat) : Nat := percentage * 360 / budget_allocation

theorem basic_astrophysics_degrees :
  arc_of_sector (budget_allocation - (microphotonics_perc + home_electronics_perc + food_additives_perc + genetically_modified_perc + industrial_lubricants_perc)) = 90 :=
  by
  sorry

end basic_astrophysics_degrees_l3_3808


namespace find_x_l3_3493

def f (x: ℝ) : ℝ := 3 * x - 5

theorem find_x (x : ℝ) (h : 2 * (f x) - 10 = f (x - 2)) : x = 3 :=
by
  sorry

end find_x_l3_3493


namespace malvina_card_value_sum_l3_3578

noncomputable def possible_values_sum: ℝ :=
  let value1 := 1
  let value2 := (-1 + Real.sqrt 5) / 2
  (value1 + value2) / 2

theorem malvina_card_value_sum
  (hx : ∃ x : ℝ, 0 < x ∧ x < Real.pi / 2 ∧ 
                 (x = Real.pi / 4 ∨ (Real.sin x = (-1 + Real.sqrt 5) / 2))):
  possible_values_sum = (1 + Real.sqrt 5) / 2 :=
by
  sorry

end malvina_card_value_sum_l3_3578


namespace sin_330_eq_neg_one_half_l3_3964

theorem sin_330_eq_neg_one_half : Real.sin (330 * Real.pi / 180) = -1 / 2 := by
  sorry

end sin_330_eq_neg_one_half_l3_3964


namespace visible_black_area_ratio_l3_3464

-- Definitions for circle areas as nonnegative real numbers
variables (A_b A_g A_w : ℝ) (hA_b : 0 ≤ A_b) (hA_g : 0 ≤ A_g) (hA_w : 0 ≤ A_w)
-- Condition: Initial visible black area is 7 times the white area
axiom initial_visible_black_area : 7 * A_w = A_b

-- Definition of new visible black area after movement
def new_visible_black_area := A_b - A_w

-- Prove the ratio of the visible black regions before and after moving the circles
theorem visible_black_area_ratio :
  (7 * A_w) / ((7 * A_w) - A_w) = 7 / 6 :=
by { sorry }

end visible_black_area_ratio_l3_3464


namespace abs_inequality_range_l3_3556

theorem abs_inequality_range (x : ℝ) (b : ℝ) (h : 0 < b) : (b > 2) ↔ ∃ x : ℝ, |x - 5| + |x - 7| < b :=
sorry

end abs_inequality_range_l3_3556


namespace mixed_groups_count_l3_3016

namespace YoungPhotographerClub

theorem mixed_groups_count
  (total_children : ℕ)
  (total_groups : ℕ)
  (children_per_group : ℕ)
  (photographs_per_group : ℕ)
  (boy_boy_photographs : ℕ)
  (girl_girl_photographs : ℕ)
  (total_photographs : ℕ)
  (mixed_photographs : ℕ)
  (mixed_groups : ℕ)
  (H1 : total_children = 300)
  (H2 : total_groups = 100)
  (H3 : children_per_group = 3)
  (H4 : photographs_per_group = 3)
  (H5 : boy_boy_photographs = 100)
  (H6 : girl_girl_photographs = 56)
  (H7 : total_photographs = total_groups * photographs_per_group)
  (H8 : mixed_photographs = total_photographs - boy_boy_photographs - girl_girl_photographs)
  (H9 : mixed_groups = mixed_photographs / 2) : mixed_groups = 72 := 
by
  rw [H1, H2, H3, H4, H5, H6, H7, H8, H9]
  sorry

end YoungPhotographerClub

end mixed_groups_count_l3_3016


namespace sin_330_l3_3935

theorem sin_330 : Real.sin (330 * Real.pi / 180) = -1 / 2 := 
by
  sorry

end sin_330_l3_3935


namespace sin_330_eq_neg_half_l3_3830

theorem sin_330_eq_neg_half : sin (330 * (π / 180)) = -1 / 2 := 
by
  -- condition: \(330^\circ = 360^\circ - 30^\circ\)
  -- condition: \(\sin 30^\circ = \frac{1}{2}\)
  -- condition: sine is negative in the fourth quadrant
  sorry

end sin_330_eq_neg_half_l3_3830


namespace width_of_wall_is_6_l3_3352

-- Definitions of the conditions given in the problem
def height_of_wall (w : ℝ) := 4 * w
def length_of_wall (h : ℝ) := 3 * h
def volume_of_wall (w h l : ℝ) := w * h * l

-- Proof statement that the width of the wall is 6 meters given the conditions
theorem width_of_wall_is_6 :
  ∃ w : ℝ, 
  (height_of_wall w = 4 * w) ∧ 
  (length_of_wall (height_of_wall w) = 3 * (height_of_wall w)) ∧ 
  (volume_of_wall w (height_of_wall w) (length_of_wall (height_of_wall w)) = 10368) ∧ 
  (w = 6) :=
sorry

end width_of_wall_is_6_l3_3352


namespace total_students_l3_3736

theorem total_students (T : ℕ)
  (A_cond : (2/9 : ℚ) * T = (a_real : ℚ))
  (B_cond : (1/3 : ℚ) * T = (b_real : ℚ))
  (C_cond : (2/9 : ℚ) * T = (c_real : ℚ))
  (D_cond : (1/9 : ℚ) * T = (d_real : ℚ))
  (E_cond : 15 = e_real) :
  (2/9 : ℚ) * T + (1/3 : ℚ) * T + (2/9 : ℚ) * T + (1/9 : ℚ) * T + 15 = T → T = 135 :=
by
  sorry

end total_students_l3_3736


namespace sin_330_deg_l3_3871

noncomputable theory

open Real

theorem sin_330_deg :
  sin (330 * (π / 180)) = -1 / 2 :=
by sorry

end sin_330_deg_l3_3871


namespace sin_330_eq_neg_one_half_l3_3906

theorem sin_330_eq_neg_one_half :
  ∃ θ : ℝ, sin (330 * real.pi / 180) = -1 / 2 :=
begin
  have h1 : sin (360 * real.pi / 180 - 30 * real.pi / 180) = -sin (30 * real.pi / 180),
  { rw sin_sub,
    ring,
    rw sin_pi_div_six, -- sin 30° = 1 / 2
  },
  have h2 : sin (30 * real.pi / 180) = 1 / 2 := by exact real.sin_pi_div_six,
  use 330,
  calc
    sin (330 * real.pi / 180)
        = sin ((360 - 30) * real.pi / 180)      : by ring
    ... = sin (360 * real.pi / 180 - 30 * real.pi / 180) : by congr' 1; ring
    ... = -sin (30 * real.pi / 180)            : by exact h1
    ... = -1 / 2                               : by rw h2,
end

end sin_330_eq_neg_one_half_l3_3906


namespace range_of_a_l3_3726

noncomputable def A := { x : ℝ | 0 < x ∧ x < 2 }
noncomputable def B (a : ℝ) := { x : ℝ | 0 < x ∧ x < (2 / a) }

theorem range_of_a (a : ℝ) (h : 0 < a) : (A ∩ (B a)) = A → 0 < a ∧ a ≤ 1 := by
  sorry

end range_of_a_l3_3726


namespace min_rows_required_l3_3329

-- Condition definitions
def number_of_students : ℕ := 2016
def seats_per_row : ℕ := 168
def max_students_per_school : ℕ := 45

-- Theorem statement to prove the minimum number of rows
theorem min_rows_required : (∀ students : ℕ, students ≤ number_of_students → 
  ∀ max_per_school : ℕ, max_per_school = max_students_per_school → 
  ∀ seats : ℕ, seats = seats_per_row → 
  ∃ rows : ℕ, rows = 16) :=
begin
  sorry
end

end min_rows_required_l3_3329


namespace sum_of_edges_of_square_l3_3552

theorem sum_of_edges_of_square (u v w x : ℕ) (hu : 0 < u) (hv : 0 < v) (hw : 0 < w) (hx : 0 < x) 
(hsum : u * x + u * v + v * w + w * x = 15) : u + v + w + x = 8 :=
by
  sorry

end sum_of_edges_of_square_l3_3552


namespace eval_exp_l3_3093

theorem eval_exp {a b : ℝ} (h : a = 3^4) : a^(5/4) = 243 :=
by
  sorry

end eval_exp_l3_3093


namespace minimize_J_l3_3170

noncomputable def H (p q : ℝ) : ℝ :=
  -3 * p * q + 4 * p * (1 - q) + 4 * (1 - p) * q - 5 * (1 - p) * (1 - q)

noncomputable def J (p : ℝ) : ℝ :=
  if p < 0 then 0 else if p > 1 then 1 else if (9 * p - 5 > 4 - 7 * p) then 9 * p - 5 else 4 - 7 * p

theorem minimize_J :
  ∃ (p : ℝ), 0 ≤ p ∧ p ≤ 1 ∧ J p = J (9 / 16) := by
  sorry

end minimize_J_l3_3170


namespace mixed_groups_count_l3_3021

-- Defining the conditions
def total_children : ℕ := 300
def groups_count : ℕ := 100
def group_size : ℕ := 3
def photographs_per_group : ℕ := group_size
def total_photographs : ℕ := groups_count * photographs_per_group
def boy_boy_photos : ℕ := 100
def girl_girl_photos : ℕ := 56
def mixed_photos : ℕ := total_photographs - boy_boy_photos - girl_girl_photos
def mixed_groups : ℕ := mixed_photos / 2

theorem mixed_groups_count : mixed_groups = 72 := by
  -- skipping the proof
  sorry

end mixed_groups_count_l3_3021


namespace at_least_one_gt_one_of_sum_gt_two_l3_3288

theorem at_least_one_gt_one_of_sum_gt_two (x y : ℝ) (h : x + y > 2) : x > 1 ∨ y > 1 := 
by sorry

end at_least_one_gt_one_of_sum_gt_two_l3_3288


namespace linear_regression_decrease_l3_3253

theorem linear_regression_decrease (x : ℝ) (y : ℝ) :
  (h : ∃ c₀ c₁, (c₀ = 2) ∧ (c₁ = -1.5) ∧ y = c₀ - c₁ * x) →
  ( ∃ Δx, Δx = 1 → ∃ Δy, Δy = -1.5) :=
by 
  sorry

end linear_regression_decrease_l3_3253


namespace Cindy_envelopes_left_l3_3384

theorem Cindy_envelopes_left :
  ∀ (initial_envelopes envelopes_per_friend friends : ℕ), 
    initial_envelopes = 37 →
    envelopes_per_friend = 3 →
    friends = 5 →
    initial_envelopes - envelopes_per_friend * friends = 22 :=
by
  intros initial_envelopes envelopes_per_friend friends h1 h2 h3
  rw [h1, h2, h3]
  norm_num
  sorry

end Cindy_envelopes_left_l3_3384


namespace mixed_groups_count_l3_3025

/-- Define the initial conditions --/
def number_of_children : ℕ := 300
def number_of_groups : ℕ := 100
def group_size : ℕ := 3
def boy_boy_photos : ℕ := 100
def girl_girl_photos : ℕ := 56

/-- Define the proof problem -/
theorem mixed_groups_count : 
    (number_of_children = 300) →
    (number_of_groups = 100) →
    (group_size = 3) →
    (boy_boy_photos = 100) →
    (girl_girl_photos = 56) →
    (∀ total_photos, total_photos = number_of_groups * group_size) →
    (∃ mixed_groups, mixed_groups = (total_photos - boy_boy_photos - girl_girl_photos) / 2) →
    mixed_groups = 72 :=
by
    intros h1 h2 h3 h4 h5 h6 h7
    sorry

end mixed_groups_count_l3_3025


namespace commission_rate_correct_l3_3481

variables (weekly_earnings : ℕ) (commission : ℕ) (total_earnings : ℕ) (sales : ℕ) (commission_rate : ℕ)

-- Base earnings per week without commission
def base_earnings : ℕ := 190

-- Total earnings target
def earnings_goal : ℕ := 500

-- Minimum sales required to meet the earnings goal
def sales_needed : ℕ := 7750

-- Definition of the commission as needed to meet the goal
def needed_commission : ℕ := earnings_goal - base_earnings

-- Definition of the actual commission rate
def commission_rate_per_sale : ℕ := (needed_commission * 100) / sales_needed

-- Proof goal: Show that commission_rate_per_sale is 4
theorem commission_rate_correct : commission_rate_per_sale = 4 :=
by
  sorry

end commission_rate_correct_l3_3481


namespace henry_skittles_l3_3822

theorem henry_skittles (b_initial: ℕ) (b_final: ℕ) (skittles_henry: ℕ) : 
  b_initial = 4 → b_final = 8 → b_final = b_initial + skittles_henry → skittles_henry = 4 :=
by
  intros h_initial h_final h_transfer
  rw [h_initial, h_final, add_comm] at h_transfer
  exact eq_of_add_eq_add_right h_transfer

end henry_skittles_l3_3822


namespace simplify_expression_eq_sqrt3_l3_3601

theorem simplify_expression_eq_sqrt3
  (a : ℝ)
  (h : a = Real.sqrt 3 + 1) :
  ( (a + 1) / a / (a - (1 + 2 * a^2) / (3 * a)) ) = Real.sqrt 3 := sorry

end simplify_expression_eq_sqrt3_l3_3601


namespace abs_neg_one_tenth_l3_3605

theorem abs_neg_one_tenth : |(-1 : ℚ) / 10| = 1 / 10 :=
by
  sorry

end abs_neg_one_tenth_l3_3605


namespace sin_330_is_minus_sqrt3_over_2_l3_3990

-- Define the angle theta as 330 degrees
def theta := 330 * Real.toRad

-- State that point corresponding to 330 degrees lies in the fourth quadrant
def point_in_fourth_quadrant (θ : ℝ) := 330 * Real.toRad = θ ∧ θ > 270 * Real.toRad ∧ θ < 360 * Real.toRad

-- Define a proof problem that states computing sin of 330° equals -√3/2
theorem sin_330_is_minus_sqrt3_over_2 : point_in_fourth_quadrant theta → Real.sin theta = -Real.sqrt 3 / 2 := by
  sorry

end sin_330_is_minus_sqrt3_over_2_l3_3990


namespace sin_330_eq_neg_one_half_l3_3911

theorem sin_330_eq_neg_one_half :
  ∃ θ : ℝ, sin (330 * real.pi / 180) = -1 / 2 :=
begin
  have h1 : sin (360 * real.pi / 180 - 30 * real.pi / 180) = -sin (30 * real.pi / 180),
  { rw sin_sub,
    ring,
    rw sin_pi_div_six, -- sin 30° = 1 / 2
  },
  have h2 : sin (30 * real.pi / 180) = 1 / 2 := by exact real.sin_pi_div_six,
  use 330,
  calc
    sin (330 * real.pi / 180)
        = sin ((360 - 30) * real.pi / 180)      : by ring
    ... = sin (360 * real.pi / 180 - 30 * real.pi / 180) : by congr' 1; ring
    ... = -sin (30 * real.pi / 180)            : by exact h1
    ... = -1 / 2                               : by rw h2,
end

end sin_330_eq_neg_one_half_l3_3911


namespace dad_strawberries_now_weight_l3_3589

-- Definitions based on the conditions given
def total_weight : ℕ := 36
def weight_lost_by_dad : ℕ := 8
def weight_of_marco_strawberries : ℕ := 12

-- Theorem to prove the question as an equality
theorem dad_strawberries_now_weight :
  total_weight - weight_lost_by_dad - weight_of_marco_strawberries = 16 := by
  sorry

end dad_strawberries_now_weight_l3_3589


namespace chef_served_173_guests_l3_3061

noncomputable def total_guests_served : ℕ :=
  let adults := 58
  let children := adults - 35
  let seniors := 2 * children
  let teenagers := seniors - 15
  let toddlers := teenagers / 2
  adults + children + seniors + teenagers + toddlers

theorem chef_served_173_guests : total_guests_served = 173 :=
  by
    -- Proof will be provided here.
    sorry

end chef_served_173_guests_l3_3061


namespace pi_approx_by_jews_l3_3146

theorem pi_approx_by_jews (S D C : ℝ) (h1 : 4 * S = (5 / 4) * C) (h2 : D = S) (h3 : C = π * D) : π = 3 := by
  sorry

end pi_approx_by_jews_l3_3146


namespace range_of_a_l3_3721

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
if x ≤ 2 then -x + 5 else a^x + 2 * a + 2

theorem range_of_a (a : ℝ) (h₁ : a > 0) (h₂ : a ≠ 1) :
  (∀ y ∈ Set.range (f a), y ≥ 3) ↔ (a ∈ Set.Ici (1/2) ∪ Set.Ioi 1) :=
sorry

end range_of_a_l3_3721


namespace fraction_addition_l3_3614

def simplest_fraction (r : ℚ) : ℚ :=
let n := r.num.gcd(r.den)
in ⟨ r.num / n, r.den / n, by sorry ⟩

theorem fraction_addition (a b : ℕ) (ha : 519 = a) (hb : 1600 = b) :
  0.324375 = (519 : ℚ) / 1600 ∧ a + b = 2119 :=
by {
  have fraction_rep : 0.324375 = (519 : ℚ) / 1600 := by sorry,
  have sum_ab : a + b = 2119 := by {
    rw [ha, hb],
    exact rfl,
  },
  exact ⟨fraction_rep, sum_ab⟩,
}

end fraction_addition_l3_3614


namespace petrol_price_increase_l3_3318

theorem petrol_price_increase
  (P P_new : ℝ)
  (C : ℝ)
  (h1 : P * C = P_new * (C * 0.7692307692307693))
  (h2 : C * (1 - 0.23076923076923073) = C * 0.7692307692307693) :
  ((P_new - P) / P) * 100 = 30 := 
  sorry

end petrol_price_increase_l3_3318


namespace sin_330_l3_3928

theorem sin_330 : Real.sin (330 * Real.pi / 180) = -1 / 2 := 
by
  sorry

end sin_330_l3_3928


namespace range_of_a_l3_3586

theorem range_of_a (a : ℝ) (h : ∀ x : ℝ, (x^2 + (a + 2) * x + 1) * ((3 - 2 * a) * x^2 + 5 * x + (3 - 2 * a)) ≥ 0) : a ∈ Set.Icc (-4 : ℝ) 0 := sorry

end range_of_a_l3_3586


namespace count_valid_three_digit_numbers_l3_3116

theorem count_valid_three_digit_numbers : 
  ∃ n : ℕ, (100 ≤ n ∧ n < 1000) ∧ 
           (∃ a b c : ℕ, n = 100 * a + 10 * b + c ∧ 
                          a ≥ 1 ∧ a ≤ 9 ∧ 
                          b ≥ 0 ∧ b ≤ 9 ∧ 
                          c ≥ 0 ∧ c ≤ 9 ∧ 
                          (a = b ∨ b = c ∨ a = c ∨ 
                           a + b > c ∧ a + c > b ∧ b + c > a)) ∧
           n = 57 := 
sorry

end count_valid_three_digit_numbers_l3_3116


namespace filter_replacement_month_l3_3151

theorem filter_replacement_month (n : ℕ) (h : n = 25) : (7 * (n - 1)) % 12 = 0 → "January" = "January" :=
by
  intros
  sorry

end filter_replacement_month_l3_3151


namespace part1_part2_l3_3114

-- Define the function f
def f (x a : ℝ) : ℝ := abs (x + a - 1) + abs (x - 2 * a)

-- Part (1) of the proof problem
theorem part1 (a : ℝ) : f 1 a < 3 → - (2 : ℝ)/3 < a ∧ a < 4 / 3 := sorry

-- Part (2) of the proof problem
theorem part2 (a x : ℝ) : a ≥ 1 → f x a ≥ 2 := sorry

end part1_part2_l3_3114


namespace calculate_product_l3_3679

theorem calculate_product :
  6^5 * 3^5 = 1889568 := by
  sorry

end calculate_product_l3_3679


namespace boxes_to_fill_l3_3251

theorem boxes_to_fill (total_boxes filled_boxes : ℝ) (h₁ : total_boxes = 25.75) (h₂ : filled_boxes = 17.5) : 
  total_boxes - filled_boxes = 8.25 := 
by
  sorry

end boxes_to_fill_l3_3251


namespace least_number_to_subtract_l3_3203

theorem least_number_to_subtract (n : ℕ) (k : ℕ) (h : 1387 = n + k * 15) : n = 7 :=
by
  sorry

end least_number_to_subtract_l3_3203


namespace sin_330_is_minus_sqrt3_over_2_l3_3986

-- Define the angle theta as 330 degrees
def theta := 330 * Real.toRad

-- State that point corresponding to 330 degrees lies in the fourth quadrant
def point_in_fourth_quadrant (θ : ℝ) := 330 * Real.toRad = θ ∧ θ > 270 * Real.toRad ∧ θ < 360 * Real.toRad

-- Define a proof problem that states computing sin of 330° equals -√3/2
theorem sin_330_is_minus_sqrt3_over_2 : point_in_fourth_quadrant theta → Real.sin theta = -Real.sqrt 3 / 2 := by
  sorry

end sin_330_is_minus_sqrt3_over_2_l3_3986


namespace tilly_counts_total_stars_l3_3037

open Nat

def stars_to_east : ℕ := 120
def factor_west_stars : ℕ := 6
def stars_to_west : ℕ := factor_west_stars * stars_to_east
def total_stars : ℕ := stars_to_east + stars_to_west

theorem tilly_counts_total_stars :
  total_stars = 840 := by
  sorry

end tilly_counts_total_stars_l3_3037


namespace gcf_7_factorial_8_factorial_l3_3406

theorem gcf_7_factorial_8_factorial :
  let factorial (n : ℕ) := Nat.factorial n in
  let seven_factorial := factorial 7 in
  let eight_factorial := factorial 8 in
  ∃ (gcf : ℕ), gcf = Nat.gcd seven_factorial eight_factorial ∧ gcf = 5040 :=
by
  let factorial (n : ℕ) := Nat.factorial n
  let seven_factorial := factorial 7
  let eight_factorial := factorial 8
  have seven_factorial_eq : seven_factorial = 5040 := by sorry
  have gcf_eq_seven_factorial : Nat.gcd seven_factorial eight_factorial = seven_factorial := by sorry
  exact ⟨seven_factorial, gcf_eq_seven_factorial, seven_factorial_eq⟩

end gcf_7_factorial_8_factorial_l3_3406


namespace opposite_sides_range_a_l3_3132

theorem opposite_sides_range_a (a: ℝ) :
  ((1 - 2 * a + 1) * (a + 4 + 1) < 0) ↔ (a < -5 ∨ a > 1) :=
by
  sorry

end opposite_sides_range_a_l3_3132


namespace gcf_7_factorial_8_factorial_l3_3405

theorem gcf_7_factorial_8_factorial :
  let factorial (n : ℕ) := Nat.factorial n in
  let seven_factorial := factorial 7 in
  let eight_factorial := factorial 8 in
  ∃ (gcf : ℕ), gcf = Nat.gcd seven_factorial eight_factorial ∧ gcf = 5040 :=
by
  let factorial (n : ℕ) := Nat.factorial n
  let seven_factorial := factorial 7
  let eight_factorial := factorial 8
  have seven_factorial_eq : seven_factorial = 5040 := by sorry
  have gcf_eq_seven_factorial : Nat.gcd seven_factorial eight_factorial = seven_factorial := by sorry
  exact ⟨seven_factorial, gcf_eq_seven_factorial, seven_factorial_eq⟩

end gcf_7_factorial_8_factorial_l3_3405


namespace faster_speed_14_l3_3216

theorem faster_speed_14 
    (d₁ : ℕ) -- actual distance traveled
    (d₂ : ℕ) -- additional distance at faster speed
    (s₁ : ℕ) -- initial speed
    (s₂ : ℕ) -- faster speed
    (h₁ : d₁ = 50)
    (h₂ : s₁ = 10)
    (h₃ : d₂ = 20) : 
    s₂ = 14 :=
by
  have t := d₁ / s₁     -- Calculate time taken to travel distance d₁ at speed s₁
  have d := d₁ + d₂     -- Total distance covered when walking at the faster speed s₂
  have s := d / t       -- Calculate the faster speed s₂ as total distance divided by time
  have eq1 : t = 5 := by simp [h₁, h₂]
  have eq2 : d = 70 := by simp [h₁, h₃]
  exact by simp [eq1, eq2]

end faster_speed_14_l3_3216


namespace tilly_counts_total_stars_l3_3038

open Nat

def stars_to_east : ℕ := 120
def factor_west_stars : ℕ := 6
def stars_to_west : ℕ := factor_west_stars * stars_to_east
def total_stars : ℕ := stars_to_east + stars_to_west

theorem tilly_counts_total_stars :
  total_stars = 840 := by
  sorry

end tilly_counts_total_stars_l3_3038


namespace determine_points_on_line_l3_3695

def pointA : ℝ × ℝ := (2, 5)
def pointB : ℝ × ℝ := (1, 2.2)
def line_eq (x y : ℝ) : ℝ := 3 * x - 5 * y + 8

theorem determine_points_on_line :
  (line_eq pointA.1 pointA.2 ≠ 0) ∧ (line_eq pointB.1 pointB.2 = 0) :=
by
  sorry

end determine_points_on_line_l3_3695


namespace cubic_identity_l3_3540

theorem cubic_identity (a b c : ℝ) (h1 : a + b + c = 7) (h2 : ab + ac + bc = 11) (h3 : abc = -6) : a^3 + b^3 + c^3 = 94 :=
by
  sorry

end cubic_identity_l3_3540


namespace Cindy_envelopes_left_l3_3383

theorem Cindy_envelopes_left :
  ∀ (initial_envelopes envelopes_per_friend friends : ℕ), 
    initial_envelopes = 37 →
    envelopes_per_friend = 3 →
    friends = 5 →
    initial_envelopes - envelopes_per_friend * friends = 22 :=
by
  intros initial_envelopes envelopes_per_friend friends h1 h2 h3
  rw [h1, h2, h3]
  norm_num
  sorry

end Cindy_envelopes_left_l3_3383


namespace smallest_positive_b_l3_3704

theorem smallest_positive_b (b : ℤ) :
  b % 5 = 1 ∧ b % 4 = 2 ∧ b % 7 = 3 → b = 86 :=
by
  sorry

end smallest_positive_b_l3_3704


namespace susan_avg_speed_l3_3498

variable (d1 d2 : ℕ) (s1 s2 : ℕ)

def time (d s : ℕ) : ℚ := d / s

theorem susan_avg_speed 
  (h1 : d1 = 40) 
  (h2 : s1 = 30) 
  (h3 : d2 = 40) 
  (h4 : s2 = 15) : 
  (d1 + d2) / (time d1 s1 + time d2 s2) = 20 := 
by 
  -- Sorry to skip the proof.
  sorry

end susan_avg_speed_l3_3498


namespace potato_sales_l3_3230

theorem potato_sales :
  let total_weight := 6500
  let damaged_weight := 150
  let bag_weight := 50
  let price_per_bag := 72
  let sellable_weight := total_weight - damaged_weight
  let num_bags := sellable_weight / bag_weight
  let total_revenue := num_bags * price_per_bag
  total_revenue = 9144 :=
by
  sorry

end potato_sales_l3_3230


namespace sin_330_eq_neg_half_l3_3890

theorem sin_330_eq_neg_half : Real.sin (330 * Real.pi / 180) = -1 / 2 := by
  -- Definitions and conditions from the problem
  have h1 : 330 = 360 - 30 := rfl
  have h2 : ∀ θ, Real.sin θ = -Real.sin (θ - 2 * Real.pi) := sorry
  have h3 : Real.sin (30 * Real.pi / 180) = 1 / 2 := sorry

  -- Use the given conditions to prove the equality
  calc
    Real.sin (330 * Real.pi / 180)
        = Real.sin (-(30 * Real.pi / 180) + 2 * Real.pi) : by rw [←h1, Real.sin_angle_sub_pi]
    ... = -Real.sin (30 * Real.pi / 180) : by rw [h2]
    ... = -1 / 2 : by rw [h3]

end sin_330_eq_neg_half_l3_3890


namespace airline_flights_increase_l3_3817

theorem airline_flights_increase (n k : ℕ) 
  (h : (n + k) * (n + k - 1) / 2 - n * (n - 1) / 2 = 76) :
  (n = 6 ∧ n + k = 14) ∨ (n = 76 ∧ n + k = 77) :=
by
  sorry

end airline_flights_increase_l3_3817


namespace cooperative_payment_divisibility_l3_3508

theorem cooperative_payment_divisibility (T_old : ℕ) (N : ℕ) 
  (hN : N = 99 * T_old / 100) : 99 ∣ N :=
by
  sorry

end cooperative_payment_divisibility_l3_3508


namespace trigonometric_identity_l3_3465

theorem trigonometric_identity 
  (α β γ : ℝ)
  (h : (1 - Real.sin α) * (1 - Real.sin β) * (1 - Real.sin γ) = (1 + Real.sin α) * (1 + Real.sin β) * (1 + Real.sin γ)) :
  (1 - Real.sin α) * (1 - Real.sin β) * (1 - Real.sin γ) = 
  abs (Real.cos α * Real.cos β * Real.cos γ) ∧
  (1 + Real.sin α) * (1 + Real.sin β) * (1 + Real.sin γ) = 
  abs (Real.cos α * Real.cos β * Real.cos γ) := by
  sorry

end trigonometric_identity_l3_3465


namespace mixed_groups_count_l3_3014

theorem mixed_groups_count (total_children : ℕ) (total_groups : ℕ)
  (group_size : ℕ) (total_boy_boy_photos : ℕ)
  (total_girl_girl_photos : ℕ)
  (total_photos : ℕ)
  (each_group_photos : ℕ)
  (total_mixed_photos : ℕ)
  (mixed_group_count : ℕ):
  total_children = 300 ∧
  total_groups = 100 ∧
  group_size = 3 ∧
  total_boy_boy_photos = 100 ∧
  total_girl_girl_photos = 56 ∧
  each_group_photos = 3 ∧
  total_photos = 100 * each_group_photos ∧
  total_mixed_photos = total_photos - total_boy_boy_photos - total_girl_girl_photos ∧
  mixed_group_count = total_mixed_photos / 2 →
  mixed_group_count = 72 :=
by
  sorry

end mixed_groups_count_l3_3014


namespace length_of_lateral_edge_l3_3563

theorem length_of_lateral_edge (vertices : ℕ) (total_length : ℝ) (num_lateral_edges : ℕ) (length_each_edge : ℝ) : 
  vertices = 12 ∧ total_length = 30 ∧ num_lateral_edges = 6 → length_each_edge = 5 :=
by
  intros h
  cases h with h_vertices h_rest
  cases h_rest with h_length h_num_edges
  have h_calculation : length_each_edge = total_length / num_lateral_edges := sorry
  rw [h_vertices, h_length, h_num_edges] at h_calculation
  norm_num at h_calculation
  exact h_calculation
  sorry

end length_of_lateral_edge_l3_3563


namespace freds_total_marbles_l3_3536

theorem freds_total_marbles :
  let red := 38
  let green := red / 2
  let dark_blue := 6
  red + green + dark_blue = 63 := by
  sorry

end freds_total_marbles_l3_3536


namespace cheesecake_needs_more_eggs_l3_3062

def chocolate_eggs_per_cake := 3
def cheesecake_eggs_per_cake := 8
def num_chocolate_cakes := 5
def num_cheesecakes := 9

theorem cheesecake_needs_more_eggs :
  cheesecake_eggs_per_cake * num_cheesecakes - chocolate_eggs_per_cake * num_chocolate_cakes = 57 :=
by
  sorry

end cheesecake_needs_more_eggs_l3_3062


namespace max_area_basketball_court_l3_3624

theorem max_area_basketball_court : 
  ∃ l w : ℝ, 2 * l + 2 * w = 400 ∧ l ≥ 100 ∧ w ≥ 50 ∧ l * w = 10000 :=
by {
  -- We are skipping the proof for now
  sorry
}

end max_area_basketball_court_l3_3624


namespace ricciana_jump_distance_l3_3595

theorem ricciana_jump_distance (R : ℕ) :
  let Ricciana_run := 20
  let Margarita_run := 18
  let Margarita_jump := 2 * R - 1
  let Total_distance_Margarita := Margarita_run + Margarita_jump
  let Total_distance_Ricciana := Ricciana_run + R + 1
  Total_distance_Margarita = Total_distance_Ricciana → R = 22 :=
by
  -- Definitions
  let Ricciana_run := 20
  let Margarita_run := 18
  let Margarita_jump := 2 * R - 1
  let Total_distance_Margarita := Margarita_run + Margarita_jump
  let Total_distance_Ricciana := Ricciana_run + R + 1

  -- Given condition
  intro h
  sorry

end ricciana_jump_distance_l3_3595


namespace catch_up_time_l3_3357

def A_departure_time : ℕ := 8 * 60 -- in minutes
def B_departure_time : ℕ := 6 * 60 -- in minutes
def relative_speed (v : ℕ) : ℕ := 5 * v / 4 -- (2.5v effective) converted to integer math
def initial_distance (v : ℕ) : ℕ := 2 * v * 2 -- 4v distance (B's 2 hours lead)

theorem catch_up_time (v : ℕ) :  A_departure_time + ((initial_distance v * 4) / (relative_speed v - v)) = 1080 :=
by
  sorry

end catch_up_time_l3_3357


namespace lemons_needed_l3_3354

theorem lemons_needed (lemons32 : ℕ) (lemons4 : ℕ) (h1 : lemons32 = 24) (h2 : (24 : ℕ) / 32 = (lemons4 : ℕ) / 4) : lemons4 = 3 := 
sorry

end lemons_needed_l3_3354


namespace find_larger_number_l3_3298

theorem find_larger_number
  (x y : ℝ)
  (h1 : y = 2 * x + 3)
  (h2 : x + y = 27)
  : y = 19 :=
by
  sorry

end find_larger_number_l3_3298


namespace max_f_at_1_f_inequality_l3_3548

-- Define the function f(x)
def f (a b x : ℝ) := a * log x + 0.5 * b * x^2 - (b + a) * x

-- Problem (I): Maximum value of f(x) when a = 1 and b = 0
theorem max_f_at_1 (x : ℝ) (h_pos : 0 < x) : f 1 0 x ≤ f 1 0 1 :=
by sorry -- Prove that f(x) reaches its maximum value at x = 1

-- Problem (II): Prove the inequality for f(x) when b = 1
theorem f_inequality 
  (a : ℝ) (e : ℝ) (h_e : real.exp 1 = e)  
  (h1 : 1 < a) (h2 : a ≤ e) (x1 x2 : ℝ) (h3 : 1 ≤ x1) (h4 : x1 ≤ a) (h5 : 1 ≤ x2) (h6 : x2 ≤ a) :
  |f a 1 x1 - f a 1 x2| < 1 :=
by sorry -- Prove the required inequality

end max_f_at_1_f_inequality_l3_3548


namespace nearest_integer_to_expression_correct_l3_3792

open Real

noncomputable def nearest_integer_to_expression : ℝ :=
  let a := (3 + (sqrt 5))^6
  let b := (3 - (sqrt 5))^6
  let s := a + b
  floor s

theorem nearest_integer_to_expression_correct :
  nearest_integer_to_expression = 74608 :=
begin
  sorry
end

end nearest_integer_to_expression_correct_l3_3792


namespace inequality_solution_exists_l3_3272

theorem inequality_solution_exists (x m : ℝ) (h1: 1 < x) (h2: x ≤ 2) (h3: x > m) : m < 2 :=
sorry

end inequality_solution_exists_l3_3272


namespace rightmost_four_digits_of_5_pow_2023_l3_3337

theorem rightmost_four_digits_of_5_pow_2023 :
  (5 ^ 2023) % 10000 = 8125 :=
sorry

end rightmost_four_digits_of_5_pow_2023_l3_3337


namespace sin_330_eq_neg_half_l3_3837

theorem sin_330_eq_neg_half 
  (h1: (330 : ℝ) = 330)
  (h2: (330 : ℝ = 360 - 30))
  (h3: 30 ∈ reference_angle_set ∧ sin 30 = 1/2)
  (h4: 330 ∈ fourth_quadrant)
  (h5: ∀ θ ∈ fourth_quadrant, sin θ < 0) :
  sin 330 = -1/2 := 
by
  sorry

end sin_330_eq_neg_half_l3_3837


namespace years_to_rise_to_chief_l3_3282

-- Definitions based on the conditions
def ageWhenRetired : ℕ := 46
def ageWhenJoined : ℕ := 18
def additionalYearsAsMasterChief : ℕ := 10
def multiplierForChiefToMasterChief : ℚ := 1.25

-- Total years spent in the military
def totalYearsInMilitary : ℕ := ageWhenRetired - ageWhenJoined

-- Given conditions and correct answer
theorem years_to_rise_to_chief (x : ℚ) (h : totalYearsInMilitary = x + multiplierForChiefToMasterChief * x + additionalYearsAsMasterChief) :
  x = 8 := by
  sorry

end years_to_rise_to_chief_l3_3282


namespace trig_identity_t_half_l3_3533

theorem trig_identity_t_half (a t : ℝ) (ht : t = Real.tan (a / 2)) :
  Real.sin a = (2 * t) / (1 + t^2) ∧
  Real.cos a = (1 - t^2) / (1 + t^2) ∧
  Real.tan a = (2 * t) / (1 - t^2) := 
sorry

end trig_identity_t_half_l3_3533


namespace nearest_integer_3_add_sqrt_5_pow_6_l3_3790

noncomputable def approx (x : ℝ) : ℕ := Real.floor (x + 0.5)

theorem nearest_integer_3_add_sqrt_5_pow_6 : 
  approx ((3 + Real.sqrt 5)^6) = 22608 :=
by
  -- Proof omitted, sorry
  sorry

end nearest_integer_3_add_sqrt_5_pow_6_l3_3790


namespace quadratic_function_points_l3_3003

theorem quadratic_function_points (a c y1 y2 y3 y4 : ℝ) (h_a : a < 0)
    (h_A : y1 = a * (-2)^2 - 4 * a * (-2) + c)
    (h_B : y2 = a * 0^2 - 4 * a * 0 + c)
    (h_C : y3 = a * 3^2 - 4 * a * 3 + c)
    (h_D : y4 = a * 5^2 - 4 * a * 5 + c)
    (h_condition : y2 * y4 < 0) : y1 * y3 < 0 :=
by
  sorry

end quadratic_function_points_l3_3003


namespace room_width_is_12_l3_3315

variable (w : ℕ)

-- Definitions of given conditions
def room_length := 19
def veranda_width := 2
def veranda_area := 140

-- Statement that needs to be proven
theorem room_width_is_12
  (h1 : veranda_width = 2)
  (h2 : veranda_area = 140)
  (h3 : room_length = 19) :
  w = 12 :=
by
  sorry

end room_width_is_12_l3_3315


namespace sin_330_eq_neg_one_half_l3_3862

theorem sin_330_eq_neg_one_half :
  ∃ θ, θ = 330 ∧ (sin (Real.pi * (θ / 180)) = -1 / 2) :=
by
  use 330
  split
  · rfl
  · sorry

end sin_330_eq_neg_one_half_l3_3862


namespace conner_collected_on_day_two_l3_3770

variable (s0 : ℕ) (c0 : ℕ) (s1 : ℕ) (c1 : ℕ) (c2 : ℕ) (s3 : ℕ) (c3 : ℕ) (total_sydney : ℕ) (total_conner : ℕ)

theorem conner_collected_on_day_two :
  s0 = 837 ∧ c0 = 723 ∧ 
  s1 = 4 ∧ c1 = 8 * s1 ∧
  s3 = 2 * c1 ∧ c3 = 27 ∧
  total_sydney = s0 + s1 + s3 ∧
  total_conner = c0 + c1 + c2 + c3 ∧
  total_conner >= total_sydney
  → c2 = 123 :=
by
  sorry

end conner_collected_on_day_two_l3_3770


namespace area_of_quadrilateral_PF1QF2_l3_3710

theorem area_of_quadrilateral_PF1QF2 (x y : ℝ) (F1 F2 P Q : ℝ×ℝ) 
  (h1 : ∀ p : ℝ×ℝ, p ∈ set_of (λ q, q.1^2/16 + q.2^2/4 = 1))
  (h2 : F1 = (4, 0) ∧ F2 = (-4, 0)) 
  (h3 : Q = (-P.1, -P.2))
  (h4 : dist P Q = dist F1 F2) :
  let a := 8 in
  let c := 4 in
  let b_sq := a^2 - c^2 in
  let m := |dist P F1| in
  let n := |dist P F2| in
  m * n = 8 :=
by sorry

end area_of_quadrilateral_PF1QF2_l3_3710


namespace turnip_pulled_by_mice_l3_3502

theorem turnip_pulled_by_mice :
  ∀ (M B G D J C : ℕ),
    D = 2 * B →
    B = 3 * G →
    G = 4 * J →
    J = 5 * C →
    C = 6 * M →
    (D + B + G + J + C + M) ≥ (D + B + G + J + C) + M → 
    1237 * M ≤ (D + B + G + J + C + M) :=
by
  intros M B G D J C h1 h2 h3 h4 h5 h6
  rw [h1, h2, h3, h4, h5]
  linarith

end turnip_pulled_by_mice_l3_3502


namespace sequence_n_value_l3_3254

theorem sequence_n_value (a : ℕ → ℕ) (n : ℕ) (h1 : a 1 = 1) (h2 : ∀ n, a (n + 1) = a n + 3) (h3 : a n = 2008) : n = 670 :=
by
 sorry

end sequence_n_value_l3_3254


namespace angle_measure_l3_3639

theorem angle_measure (x : ℝ) :
  (180 - x) = 7 * (90 - x) → 
  x = 75 :=
by
  intro h
  sorry

end angle_measure_l3_3639


namespace max_annual_profit_l3_3227

noncomputable def R (x : ℝ) : ℝ :=
  if x < 40 then 10 * x^2 + 300 * x
  else (901 * x^2 - 9450 * x + 10000) / x

noncomputable def W (x : ℝ) : ℝ :=
  if x < 40 then -10 * x^2 + 600 * x - 260
  else -x + 9190 - 10000 / x

theorem max_annual_profit : ∃ x : ℝ, W 100 = 8990 :=
by {
  use 100,
  sorry
}

end max_annual_profit_l3_3227


namespace solve_inequality_system_l3_3168

-- Define the conditions and the correct answer
def system_of_inequalities (x : ℝ) : Prop :=
  (5 * x - 1 > 3 * (x + 1)) ∧ ((1/2) * x - 1 ≤ 7 - (3/2) * x)

def solution_set (x : ℝ) : Prop :=
  2 < x ∧ x ≤ 4

-- State that solving the system of inequalities is equivalent to the solution set
theorem solve_inequality_system (x : ℝ) : system_of_inequalities x ↔ solution_set x :=
  sorry

end solve_inequality_system_l3_3168


namespace envelopes_left_l3_3382

theorem envelopes_left (initial_envelopes : ℕ) (envelopes_per_friend : ℕ) (number_of_friends : ℕ) (remaining_envelopes : ℕ) :
  initial_envelopes = 37 → envelopes_per_friend = 3 → number_of_friends = 5 → remaining_envelopes = initial_envelopes - (envelopes_per_friend * number_of_friends) → remaining_envelopes = 22 :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3] at h4
  norm_num at h4
  exact h4

end envelopes_left_l3_3382


namespace nearest_integer_is_11304_l3_3786

def nearest_integer_to_a_plus_b_pow_six (a b : ℝ) (h : b = Real.sqrt 5) : ℝ :=
  (a + b) ^ 6

theorem nearest_integer_is_11304 : nearest_integer_to_a_plus_b_pow_six 3 (Real.sqrt 5) rfl = 11304 := 
  sorry

end nearest_integer_is_11304_l3_3786


namespace sin_330_correct_l3_3970

noncomputable def sin_330 : ℝ := sorry

theorem sin_330_correct : sin_330 = -1 / 2 :=
  sorry

end sin_330_correct_l3_3970


namespace terminating_fraction_count_l3_3705

theorem terminating_fraction_count : 
  (∃ (n : ℕ), 1 ≤ n ∧ n ≤ 299 ∧ (∃ k, n = 3 * k)) ∧ 
  (∃ (count : ℕ), count = 99) :=
by
  sorry

end terminating_fraction_count_l3_3705


namespace charge_for_cat_l3_3591

theorem charge_for_cat (D N_D N_C T C : ℝ) 
  (h1 : D = 60) (h2 : N_D = 20) (h3 : N_C = 60) (h4 : T = 3600)
  (h5 : 20 * D + 60 * C = T) :
  C = 40 := by
  sorry

end charge_for_cat_l3_3591


namespace age_ratio_problem_l3_3194

def age_condition (s a : ℕ) : Prop :=
  s - 2 = 2 * (a - 2) ∧ s - 4 = 3 * (a - 4)

def future_ratio (s a x : ℕ) : Prop :=
  (s + x) * 2 = (a + x) * 3

theorem age_ratio_problem :
  ∃ s a x : ℕ, age_condition s a ∧ future_ratio s a x ∧ x = 2 :=
by
  sorry

end age_ratio_problem_l3_3194


namespace folding_positions_l3_3776

theorem folding_positions (positions : Finset ℕ) (h_conditions: positions = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13}) : 
  ∃ valid_positions : Finset ℕ, valid_positions = {1, 2, 3, 4, 9, 10, 11, 12} ∧ valid_positions.card = 8 :=
by
  sorry

end folding_positions_l3_3776


namespace youngest_child_cakes_l3_3292

theorem youngest_child_cakes : 
  let total_cakes := 60
  let oldest_cakes := (1 / 4 : ℚ) * total_cakes
  let second_oldest_cakes := (3 / 10 : ℚ) * total_cakes
  let middle_cakes := (1 / 6 : ℚ) * total_cakes
  let second_youngest_cakes := (1 / 5 : ℚ) * total_cakes
  let distributed_cakes := oldest_cakes + second_oldest_cakes + middle_cakes + second_youngest_cakes
  let youngest_cakes := total_cakes - distributed_cakes
  youngest_cakes = 5 := 
by
  exact sorry

end youngest_child_cakes_l3_3292


namespace problem_l3_3434

open Complex

-- Given condition: smallest positive integer n greater than 3
def smallest_n_gt_3 (n : ℕ) : Prop :=
  n > 3 ∧ ∀ m : ℕ, m > 3 → m < n → False

-- Given condition: equation holds for complex numbers
def equation_holds (a b : ℝ) (n : ℕ) : Prop :=
  (a + b * I)^n + a = (a - b * I)^n + b

-- Proof problem: Given conditions, prove b / a = 1
theorem problem (n : ℕ) (a b : ℝ)
  (h1 : smallest_n_gt_3 n)
  (h2 : 0 < a) (h3 : 0 < b)
  (h4 : equation_holds a b n) :
  b / a = 1 :=
by
  sorry

end problem_l3_3434


namespace total_amount_in_wallet_l3_3467

theorem total_amount_in_wallet
  (num_10_bills : ℕ)
  (num_20_bills : ℕ)
  (num_5_bills : ℕ)
  (amount_10_bills : ℕ)
  (num_20_bills_eq : num_20_bills = 4)
  (amount_10_bills_eq : amount_10_bills = 50)
  (total_num_bills : ℕ)
  (total_num_bills_eq : total_num_bills = 13)
  (num_10_bills_eq : num_10_bills = amount_10_bills / 10)
  (total_amount : ℕ)
  (total_amount_eq : total_amount = amount_10_bills + num_20_bills * 20 + num_5_bills * 5)
  (num_bills_accounted : ℕ)
  (num_bills_accounted_eq : num_bills_accounted = num_10_bills + num_20_bills)
  (num_5_bills_eq : num_5_bills = total_num_bills - num_bills_accounted)
  : total_amount = 150 :=
by
  sorry

end total_amount_in_wallet_l3_3467


namespace fractions_product_simplified_l3_3680

theorem fractions_product_simplified : (2/3 : ℚ) * (4/7) * (9/11) = 24/77 := by
  sorry

end fractions_product_simplified_l3_3680


namespace train_speed_l3_3225

noncomputable def speed_of_train_kmph (L V : ℝ) : ℝ :=
  3.6 * V

theorem train_speed
  (L V : ℝ)
  (h1 : L = 18 * V)
  (h2 : L + 340 = 35 * V) :
  speed_of_train_kmph L V = 72 :=
by
  sorry

end train_speed_l3_3225


namespace smallest_n_logarithm_l3_3085

theorem smallest_n_logarithm :
  ∃ n : ℕ, 0 < n ∧ 
  (Real.log (Real.log n / Real.log 3) / Real.log 3^2 =
  Real.log (Real.log n / Real.log 2) / Real.log 2^3) ∧ 
  n = 9 :=
by
  sorry

end smallest_n_logarithm_l3_3085


namespace incorrect_statement_c_l3_3803

-- Definitions based on conditions
variable (p q : Prop)

-- Lean 4 statement to check the logical proposition
theorem incorrect_statement_c (h : ¬(p ∧ q)) : ¬p ∨ ¬q :=
by
  sorry

end incorrect_statement_c_l3_3803


namespace cube_coloring_schemes_l3_3523

theorem cube_coloring_schemes (colors : Finset ℕ) (h : colors.card = 6) :
  ∃ schemes : Nat, schemes = 230 :=
by
  sorry

end cube_coloring_schemes_l3_3523


namespace multiply_and_simplify_fractions_l3_3684

theorem multiply_and_simplify_fractions :
  (2 / 3) * (4 / 7) * (9 / 11) = 24 / 77 := 
by
  sorry

end multiply_and_simplify_fractions_l3_3684


namespace frog_paths_l3_3209

theorem frog_paths (n : ℕ) : (∃ e_2n e_2n_minus_1 : ℕ,
  e_2n_minus_1 = 0 ∧
  e_2n = (1 / Real.sqrt 2) * ((2 + Real.sqrt 2) ^ (n - 1) - (2 - Real.sqrt 2) ^ (n - 1))) :=
by {
  sorry
}

end frog_paths_l3_3209


namespace no_nat_triplet_exists_l3_3083

theorem no_nat_triplet_exists (x y z : ℕ) : ¬ (x ^ 2 + y ^ 2 = 7 * z ^ 2) := 
sorry

end no_nat_triplet_exists_l3_3083


namespace half_vectorAB_is_2_1_l3_3119

def point := ℝ × ℝ -- Define a point as a pair of real numbers
def vector := ℝ × ℝ -- Define a vector as a pair of real numbers

def A : point := (-1, 0) -- Define point A
def B : point := (3, 2) -- Define point B

noncomputable def vectorAB : vector := (B.1 - A.1, B.2 - A.2) -- Define vector AB as B - A

noncomputable def half_vectorAB : vector := (1 / 2 * vectorAB.1, 1 / 2 * vectorAB.2) -- Define half of vector AB

theorem half_vectorAB_is_2_1 : half_vectorAB = (2, 1) := by
  -- Sorry is a placeholder for the proof
  sorry

end half_vectorAB_is_2_1_l3_3119


namespace best_trip_representation_l3_3759

structure TripConditions where
  initial_walk_moderate : Prop
  main_road_speed_up : Prop
  bird_watching : Prop
  return_same_route : Prop
  coffee_stop : Prop
  final_walk_moderate : Prop

theorem best_trip_representation (conds : TripConditions) : 
  conds.initial_walk_moderate →
  conds.main_road_speed_up →
  conds.bird_watching →
  conds.return_same_route →
  conds.coffee_stop →
  conds.final_walk_moderate →
  True := 
by 
  intros 
  exact True.intro

end best_trip_representation_l3_3759


namespace sin_330_eq_neg_half_l3_3919

open Real

theorem sin_330_eq_neg_half : sin (330 * (π / 180)) = - (1 / 2) := by
  have angle_eq : 330 * (π / 180) = 2 * π - (30 * (π / 180)) := by sorry
  have sin_30_pos : sin (30 * (π / 180)) = 1 / 2 := by sorry
  have sin_330_neg : sin (2 * π - (30 * (π / 180))) = - (sin (30 * (π / 180))) := by sorry
  rw [angle_eq, sin_330_neg, sin_30_pos]
  rfl

end sin_330_eq_neg_half_l3_3919


namespace shaded_area_correct_l3_3213

noncomputable def side_length : ℝ := 24
noncomputable def radius : ℝ := side_length / 4
noncomputable def area_of_square : ℝ := side_length ^ 2
noncomputable def area_of_one_circle : ℝ := Real.pi * radius ^ 2
noncomputable def total_area_of_circles : ℝ := 5 * area_of_one_circle
noncomputable def shaded_area : ℝ := area_of_square - total_area_of_circles

theorem shaded_area_correct :
  shaded_area = 576 - 180 * Real.pi := by
  sorry

end shaded_area_correct_l3_3213


namespace determine_scores_l3_3447

variables {M Q S K : ℕ}

theorem determine_scores (h1 : Q > M ∨ K > M) 
                          (h2 : M ≠ K) 
                          (h3 : S ≠ Q) 
                          (h4 : S ≠ M) : 
  (Q, S, M) = (Q, S, M) :=
by
  -- We state the theorem as true
  sorry

end determine_scores_l3_3447


namespace total_cantaloupes_l3_3709

theorem total_cantaloupes (fred_cantaloupes : ℕ) (tim_cantaloupes : ℕ) (h1 : fred_cantaloupes = 38) (h2 : tim_cantaloupes = 44) : fred_cantaloupes + tim_cantaloupes = 82 :=
by sorry

end total_cantaloupes_l3_3709


namespace find_page_words_l3_3358
open Nat

-- Define the conditions
def condition1 : Nat := 150
def condition2 : Nat := 221
def total_words_modulo : Nat := 220
def upper_bound_words : Nat := 120

-- Define properties
def is_solution (p : Nat) : Prop :=
  Nat.Prime p ∧ p ≤ upper_bound_words ∧ (condition1 * p) % condition2 = total_words_modulo

-- The theorem to prove
theorem find_page_words (p : Nat) (hp : is_solution p) : p = 67 :=
by
  sorry

end find_page_words_l3_3358


namespace stratified_sampling_third_year_l3_3069

theorem stratified_sampling_third_year :
  ∀ (total students_first_year students_second_year sample_size students_third_year sampled_students : ℕ),
  (total = 900) →
  (students_first_year = 240) →
  (students_second_year = 260) →
  (sample_size = 45) →
  (students_third_year = total - students_first_year - students_second_year) →
  (sampled_students = sample_size * students_third_year / total) →
  sampled_students = 20 :=
by
  intros
  sorry

end stratified_sampling_third_year_l3_3069


namespace chord_length_intercepted_by_curve_l3_3774

theorem chord_length_intercepted_by_curve
(param_eqns : ∀ θ : ℝ, (x = 2 * Real.cos θ ∧ y = 1 + 2 * Real.sin θ))
(line_eqn : 3 * x - 4 * y - 1 = 0) :
  ∃ (chord_length : ℝ), chord_length = 2 * Real.sqrt 3 := 
sorry

end chord_length_intercepted_by_curve_l3_3774


namespace sin_330_eq_neg_half_l3_3845

theorem sin_330_eq_neg_half 
  (h1: (330 : ℝ) = 330)
  (h2: (330 : ℝ = 360 - 30))
  (h3: 30 ∈ reference_angle_set ∧ sin 30 = 1/2)
  (h4: 330 ∈ fourth_quadrant)
  (h5: ∀ θ ∈ fourth_quadrant, sin θ < 0) :
  sin 330 = -1/2 := 
by
  sorry

end sin_330_eq_neg_half_l3_3845


namespace totalLemonProductionIn5Years_l3_3664

-- Definition of a normal lemon tree's production rate
def normalLemonProduction : ℕ := 60

-- Definition of the percentage increase for Jim's lemon trees (50%)
def percentageIncrease : ℕ := 50

-- Calculate Jim's lemon tree production per year
def jimLemonProduction : ℕ := normalLemonProduction * (100 + percentageIncrease) / 100

-- Calculate the total number of trees in Jim's grove
def treesInGrove : ℕ := 50 * 30

-- Calculate the total lemon production by Jim's grove in one year
def annualLemonProduction : ℕ := treesInGrove * jimLemonProduction

-- Calculate the total lemon production by Jim's grove in 5 years
def fiveYearLemonProduction : ℕ := 5 * annualLemonProduction

-- Theorem: Prove that the total lemon production in 5 years is 675000
theorem totalLemonProductionIn5Years : fiveYearLemonProduction = 675000 := by
  -- Proof needs to be filled in
  sorry

end totalLemonProductionIn5Years_l3_3664


namespace sin_330_eq_neg_half_l3_3920

open Real

theorem sin_330_eq_neg_half : sin (330 * (π / 180)) = - (1 / 2) := by
  have angle_eq : 330 * (π / 180) = 2 * π - (30 * (π / 180)) := by sorry
  have sin_30_pos : sin (30 * (π / 180)) = 1 / 2 := by sorry
  have sin_330_neg : sin (2 * π - (30 * (π / 180))) = - (sin (30 * (π / 180))) := by sorry
  rw [angle_eq, sin_330_neg, sin_30_pos]
  rfl

end sin_330_eq_neg_half_l3_3920


namespace sin_330_l3_3897

theorem sin_330 : Real.sin (330 * Real.pi / 180) = -1 / 2 := 
by
  -- Outline the proof here without providing it
  -- sorry to delay the proof
  sorry

end sin_330_l3_3897


namespace proof_cost_A_B_schools_proof_renovation_plans_l3_3783

noncomputable def cost_A_B_schools : Prop :=
  ∃ (x y : ℝ), 2 * x + 3 * y = 78 ∧ 3 * x + y = 54 ∧ x = 12 ∧ y = 18

noncomputable def renovation_plans : Prop :=
  ∃ (a : ℕ), 3 ≤ a ∧ a ≤ 5 ∧ 
    (1200 - 300) * a + (1800 - 500) * (10 - a) ≤ 11800 ∧
    300 * a + 500 * (10 - a) ≥ 4000

theorem proof_cost_A_B_schools : cost_A_B_schools :=
sorry

theorem proof_renovation_plans : renovation_plans :=
sorry

end proof_cost_A_B_schools_proof_renovation_plans_l3_3783


namespace area_of_triangle_ABC_l3_3144

-- Define the sides of the triangle
def AB : ℝ := 12
def BC : ℝ := 9

-- Define the expected area of the triangle
def expectedArea : ℝ := 54

-- Prove the area of the triangle using the given conditions
theorem area_of_triangle_ABC : (1/2) * AB * BC = expectedArea := 
by
  sorry

end area_of_triangle_ABC_l3_3144


namespace band_fundraising_goal_exceed_l3_3008

theorem band_fundraising_goal_exceed
    (goal : ℕ)
    (basic_wash_cost deluxe_wash_cost premium_wash_cost cookie_cost : ℕ)
    (basic_wash_families deluxe_wash_families premium_wash_families sold_cookies : ℕ)
    (total_earnings : ℤ) :
    
    goal = 150 →
    basic_wash_cost = 5 →
    deluxe_wash_cost = 8 →
    premium_wash_cost = 12 →
    cookie_cost = 2 →
    basic_wash_families = 10 →
    deluxe_wash_families = 6 →
    premium_wash_families = 2 →
    sold_cookies = 30 →
    total_earnings = 
        (basic_wash_cost * basic_wash_families +
         deluxe_wash_cost * deluxe_wash_families +
         premium_wash_cost * premium_wash_families +
         cookie_cost * sold_cookies : ℤ) →
    (goal : ℤ) - total_earnings = -32 :=
by
  intros h_goal h_basic h_deluxe h_premium h_cookie h_basic_fam h_deluxe_fam h_premium_fam h_sold_cookies h_total_earnings
  sorry

end band_fundraising_goal_exceed_l3_3008


namespace sin_330_eq_neg_one_half_l3_3859

theorem sin_330_eq_neg_one_half :
  ∃ θ, θ = 330 ∧ (sin (Real.pi * (θ / 180)) = -1 / 2) :=
by
  use 330
  split
  · rfl
  · sorry

end sin_330_eq_neg_one_half_l3_3859


namespace max_area_ratio_l3_3545

noncomputable theory

variables (A B C I P : ℝ)
variables (r : ℝ := 2) (PI : ℝ := 1) (S_APB S_APC : ℝ)

-- Define the conditions given in the problem
def conditions := 
  (inradius : r = 2) ∧   -- Radius of incircle
  (center : I = 2*r) ∧   -- Center I derived from equilateral triangle property
  (pointP : (P - I) = 1) -- PI = 1

-- Define the areas of the triangles
def area_ratio := S_APB / S_APC

-- The main theorem statement
theorem max_area_ratio (h : conditions) : 
  ∃ S_APB S_APC, area_ratio = (3 + Real.sqrt 5) / 2 :=
begin
  -- We would provide proof here, but since it is requested to skip the proof...
  sorry,
end

end max_area_ratio_l3_3545


namespace unique_triangle_constructions_l3_3547

structure Triangle :=
(a b c : ℝ) (A B C : ℝ)

-- Definitions for the conditions
def SSS (t : Triangle) : Prop := 
  t.a > 0 ∧ t.b > 0 ∧ t.c > 0

def SAS (t : Triangle) : Prop :=
  t.a > 0 ∧ t.b > 0 ∧ t.A > 0 ∧ t.A < 180

def ASA (t : Triangle) : Prop :=
  t.A > 0 ∧ t.B > 0 ∧ t.c > 0 ∧ t.A + t.B < 180

def SSA (t : Triangle) : Prop :=
  t.a > 0 ∧ t.b > 0 ∧ t.A > 0 ∧ t.A < 180 

-- The formally stated proof goal
theorem unique_triangle_constructions (t : Triangle) :
  (SSS t ∨ SAS t ∨ ASA t) ∧ ¬(SSA t) :=
by
  sorry

end unique_triangle_constructions_l3_3547


namespace sum_of_consecutive_odd_integers_l3_3473

-- Definitions of conditions
def consecutive_odd_integers (a b : ℤ) : Prop :=
  b = a + 2 ∧ (a % 2 = 1) ∧ (b % 2 = 1)

def five_times_smaller_minus_two_condition (a b : ℤ) : Prop :=
  b = 5 * a - 2

-- Theorem statement
theorem sum_of_consecutive_odd_integers (a b : ℤ)
  (h1 : consecutive_odd_integers a b)
  (h2 : five_times_smaller_minus_two_condition a b) : a + b = 4 :=
by
  sorry

end sum_of_consecutive_odd_integers_l3_3473


namespace cubic_sum_l3_3274

theorem cubic_sum (a b c : ℝ) (h1 : a + b + c = 5) (h2 : a * b + b * c + c * a = 7) (h3 : a * b * c = 2) :
  a^3 + b^3 + c^3 = 26 :=
by
  sorry

end cubic_sum_l3_3274


namespace faster_speed_l3_3217

theorem faster_speed (v : ℝ) (h1 : ∀ (t : ℝ), t = 50 / 10) (h2 : ∀ (d : ℝ), d = 50 + 20) (h3 : ∀ (t : ℝ), t = 70 / v) : v = 14 :=
by
  sorry

end faster_speed_l3_3217


namespace triangle_problems_l3_3117

open Real

variables {A B C a b c : ℝ}
variables {m n : ℝ × ℝ}

def triangle_sides_and_angles (a b c : ℝ) (A B C : ℝ) : Prop := a > 0 ∧ b > 0 ∧ c > 0 ∧ A + B + C = π

def perpendicular (m n : ℝ × ℝ) : Prop := m.1 * n.1 + m.2 * n.2 = 0

noncomputable def area_of_triangle (a b c : ℝ) (A : ℝ) : ℝ :=
  1 / 2 * b * c * sin A

theorem triangle_problems
  (h1 : triangle_sides_and_angles a b c A B C)
  (h2 : m = (1, 1))
  (h3 : n = (sqrt 3 / 2 - sin B * sin C, cos B * cos C))
  (h4 : perpendicular m n)
  (h5 : a = 1)
  (h6 : b = sqrt 3 * c) :
  A = π / 6 ∧ area_of_triangle a b c A = sqrt 3 / 4 :=
by
  sorry

end triangle_problems_l3_3117


namespace find_divisor_l3_3349

theorem find_divisor (D Q R d: ℕ) (hD: D = 16698) (hQ: Q = 89) (hR: R = 14) (hDiv: D = d * Q + R): d = 187 := 
by 
  sorry

end find_divisor_l3_3349


namespace sin_330_eq_neg_one_half_l3_3881

theorem sin_330_eq_neg_one_half : sin (330 * π / 180) = -1 / 2 := by
  sorry

end sin_330_eq_neg_one_half_l3_3881


namespace min_distance_curveC1_curveC2_l3_3449

-- Definitions of the conditions
def curveC1 (P : ℝ × ℝ) : Prop :=
  ∃ θ : ℝ, P.1 = 3 + Real.cos θ ∧ P.2 = 4 + Real.sin θ

def curveC2 (P : ℝ × ℝ) : Prop :=
  P.1^2 + P.2^2 = 1

-- Proof statement
theorem min_distance_curveC1_curveC2 :
  (∀ A B : ℝ × ℝ,
    curveC1 A →
    curveC2 B →
    ∃ m : ℝ, m = 3 ∧ ∀ d : ℝ, (d = Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)) → d ≥ m) := 
  sorry

end min_distance_curveC1_curveC2_l3_3449


namespace least_value_MX_l3_3141

-- Definitions of points and lines
variables (A B C D M P X : ℝ × ℝ)
variables (y : ℝ)

-- Hypotheses based on the conditions
variables (h1 : A = (0, 0))
variables (h2 : B = (33, 0))
variables (h3 : C = (33, 56))
variables (h4 : D = (0, 56))
variables (h5 : M = (33 / 2, 0)) -- M is midpoint of AB
variables (h6 : P = (33, y)) -- P is on BC
variables (hy_range : 0 ≤ y ∧ y ≤ 56) -- y is within the bounds of BC

-- Additional derived hypotheses needed for the proof
variables (h7 : ∃ x, X = (x, sqrt (816.75))) -- X is intersection point on DA

-- The theorem statement
theorem least_value_MX : ∃ y, 0 ≤ y ∧ y ≤ 56 ∧ MX = 33 :=
by
  use 28
  sorry

end least_value_MX_l3_3141


namespace sin_330_eq_neg_sin_30_l3_3976

theorem sin_330_eq_neg_sin_30 :
  sin (330 : ℝ) = - sin (30 : ℝ) := sorry

end sin_330_eq_neg_sin_30_l3_3976


namespace sin_330_eq_neg_one_half_l3_3907

theorem sin_330_eq_neg_one_half :
  ∃ θ : ℝ, sin (330 * real.pi / 180) = -1 / 2 :=
begin
  have h1 : sin (360 * real.pi / 180 - 30 * real.pi / 180) = -sin (30 * real.pi / 180),
  { rw sin_sub,
    ring,
    rw sin_pi_div_six, -- sin 30° = 1 / 2
  },
  have h2 : sin (30 * real.pi / 180) = 1 / 2 := by exact real.sin_pi_div_six,
  use 330,
  calc
    sin (330 * real.pi / 180)
        = sin ((360 - 30) * real.pi / 180)      : by ring
    ... = sin (360 * real.pi / 180 - 30 * real.pi / 180) : by congr' 1; ring
    ... = -sin (30 * real.pi / 180)            : by exact h1
    ... = -1 / 2                               : by rw h2,
end

end sin_330_eq_neg_one_half_l3_3907


namespace sin_330_eq_neg_half_l3_3827

theorem sin_330_eq_neg_half : sin (330 * (π / 180)) = -1 / 2 := 
by
  -- condition: \(330^\circ = 360^\circ - 30^\circ\)
  -- condition: \(\sin 30^\circ = \frac{1}{2}\)
  -- condition: sine is negative in the fourth quadrant
  sorry

end sin_330_eq_neg_half_l3_3827


namespace min_cuts_for_30_sided_polygons_l3_3074

theorem min_cuts_for_30_sided_polygons (n : ℕ) (h : n = 73) : 
  ∃ k : ℕ, (∀ m : ℕ, m < k → (m + 1) ≤ 2 * m - 1972) ∧ (k = 1970) :=
sorry

end min_cuts_for_30_sided_polygons_l3_3074


namespace leopard_arrangement_l3_3748

theorem leopard_arrangement : ∃ (f : Fin 9 → ℕ), 
  (∀ i j, i ≠ j → f i ≠ f j) ∧
  let shortest_three := {f 0, f 1, f 2} in
  let middle_six := {f 3, f 4, f 5, f 6, f 7, f 8} in
  shortest_three = {0, 1, 2} ∧
  ∃ (perm : Finset.perm (Fin 9)),
    Finset.card shortest_three * Finset.card middle_six = 3! * 6! ∧ 
    3! * 6! = 4320 :=
by sorry

end leopard_arrangement_l3_3748


namespace amount_diana_owes_l3_3528

-- Problem definitions
def principal : ℝ := 75
def rate : ℝ := 0.07
def time : ℝ := 1
def interest := principal * rate * time
def total_owed := principal + interest

-- Theorem to prove that the total amount owed is $80.25
theorem amount_diana_owes : total_owed = 80.25 := by
  sorry

end amount_diana_owes_l3_3528


namespace fantasy_gala_handshakes_l3_3330

theorem fantasy_gala_handshakes
    (gremlins imps : ℕ)
    (gremlin_handshakes : ℕ)
    (imp_handshakes : ℕ)
    (imp_gremlin_handshakes : ℕ)
    (total_handshakes : ℕ)
    (h1 : gremlins = 30)
    (h2 : imps = 20)
    (h3 : gremlin_handshakes = (30 * 29) / 2)
    (h4 : imp_handshakes = (20 * 5) / 2)
    (h5 : imp_gremlin_handshakes = 20 * 30)
    (h6 : total_handshakes = gremlin_handshakes + imp_handshakes + imp_gremlin_handshakes) :
    total_handshakes = 1085 := by
    sorry

end fantasy_gala_handshakes_l3_3330


namespace graph_of_equation_is_two_lines_l3_3594

theorem graph_of_equation_is_two_lines :
  ∀ (x y : ℝ), (x * y - 2 * x + 3 * y - 6 = 0) ↔ ((x + 3 = 0) ∨ (y - 2 = 0)) := 
by
  intro x y
  sorry

end graph_of_equation_is_two_lines_l3_3594


namespace range_of_a2_l3_3541

theorem range_of_a2 (a : ℕ → ℝ) (S : ℕ → ℝ) (a2 : ℝ) (a3 a6 : ℝ) (h1: 3 * a3 = a6 + 4) (h2 : S 5 < 10) :
  a2 < 2 := 
sorry

end range_of_a2_l3_3541


namespace sin_330_eq_neg_half_l3_3836

theorem sin_330_eq_neg_half 
  (h1: (330 : ℝ) = 330)
  (h2: (330 : ℝ = 360 - 30))
  (h3: 30 ∈ reference_angle_set ∧ sin 30 = 1/2)
  (h4: 330 ∈ fourth_quadrant)
  (h5: ∀ θ ∈ fourth_quadrant, sin θ < 0) :
  sin 330 = -1/2 := 
by
  sorry

end sin_330_eq_neg_half_l3_3836


namespace abs_h_eq_2_l3_3700

-- Definitions based on the given conditions
def sum_of_squares_of_roots (h : ℝ) : Prop :=
  let a := 1
  let b := -4 * h
  let c := -8
  let sum_of_roots := -b / a
  let prod_of_roots := c / a
  let sum_of_squares := sum_of_roots^2 - 2 * prod_of_roots
  sum_of_squares = 80

-- Theorem to prove the absolute value of h is 2
theorem abs_h_eq_2 (h : ℝ) (h_condition : sum_of_squares_of_roots h) : |h| = 2 :=
by
  sorry

end abs_h_eq_2_l3_3700


namespace sin_330_eq_neg_one_half_l3_3876

theorem sin_330_eq_neg_one_half : sin (330 * π / 180) = -1 / 2 := by
  sorry

end sin_330_eq_neg_one_half_l3_3876


namespace lengths_AC_CB_ratio_GJ_JH_coords_F_on_DE_values_p_q_KL_l3_3806

-- Problem 1 - Lengths of AC and CB are 15 and 5 respectively.
theorem lengths_AC_CB (x1 y1 x2 y2 x3 y3 : ℝ) :
  (x1, y1) = (1,2) ∧ (x2, y2) = (17,14) ∧ (x3, y3) = (13,11) →
  ∃ (AC CB : ℝ), AC = 15 ∧ CB = 5 :=
by
  sorry

-- Problem 2 - Ratio of GJ and JH is 3:2.
theorem ratio_GJ_JH (x1 y1 x2 y2 x3 y3 : ℝ) :
  (x1, y1) = (11,2) ∧ (x2, y2) = (1,7) ∧ (x3, y3) = (5,5) →
  ∃ (GJ JH : ℝ), GJ / JH = 3 / 2 :=
by
  sorry

-- Problem 3 - Coordinates of point F on DE with ratio 1:2 is (3,7).
theorem coords_F_on_DE (x1 y1 x2 y2 : ℝ) :
  (x1, y1) = (1,6) ∧ (x2, y2) = (7,9) →
  ∃ (x y : ℝ), (x, y) = (3,7) :=
by
  sorry

-- Problem 4 - Values of p and q for point M on KL with ratio 3:4 are p = 15 and q = 2.
theorem values_p_q_KL (x1 y1 x2 y2 x3 y3 : ℝ) :
  (x1, y1) = (1, q) ∧ (x2, y2) = (p, 9) ∧ (x3, y3) = (7,5) →
  ∃ (p q : ℝ), p = 15 ∧ q = 2 :=
by
  sorry

end lengths_AC_CB_ratio_GJ_JH_coords_F_on_DE_values_p_q_KL_l3_3806


namespace plane_through_points_l3_3531

def point := (ℝ × ℝ × ℝ)

def plane_equation (A B C D : ℤ) (x y z : ℝ) : Prop :=
  A * x + B * y + C * z + D = 0

theorem plane_through_points : 
  ∃ (A B C D : ℤ), (A > 0) ∧ (Int.gcd (Int.gcd (Int.gcd (Int.natAbs A) (Int.natAbs B)) (Int.natAbs C)) (Int.natAbs D) = 1) ∧
  plane_equation A B C D 2 (-3) 5 ∧
  plane_equation A B C D (-1) (-3) 7 ∧
  plane_equation A B C D (-4) (-5) 6 ∧
  (A = 2) ∧ (B = -9) ∧ (C = 3) ∧ (D = -46) :=
sorry

end plane_through_points_l3_3531


namespace distance_points_3_12_and_10_0_l3_3243

theorem distance_points_3_12_and_10_0 : 
  Real.sqrt ((10 - 3)^2 + (0 - 12)^2) = Real.sqrt 193 := 
by
  sorry

end distance_points_3_12_and_10_0_l3_3243


namespace number_of_correct_conclusions_l3_3582

-- Define the conditions as hypotheses
variable {a_n : ℕ → ℝ}
variable {S : ℕ → ℝ}
variable {d : ℝ}
variable {a_1 : ℝ}
variable {n : ℕ}

-- Arithmetic sequence definition for a_n
def arithmetic_sequence (a_n : ℕ → ℝ) (a_1 d : ℝ) : Prop :=
  ∀ n, a_n n = a_1 + (n - 1) * d

-- Sum of the first n terms of the arithmetic sequence
def sum_arithmetic_sequence (S : ℕ → ℝ) (a_1 d : ℝ) : Prop :=
  ∀ n, S n = n / 2 * (2 * a_1 + (n - 1) * d)

-- Problem statement
theorem number_of_correct_conclusions 
  (h_seq : arithmetic_sequence a_n a_1 d)
  (h_sum : sum_arithmetic_sequence S a_1 d)
  (h1 : S 5 < S 6)
  (h2 : S 6 = S 7 ∧ S 7 > S 8) :
  ∃ n, n = 3 ∧ 
       (d < 0) ∧ 
       (a_n 7 = 0) ∧ 
       ¬(S 9 = S 5) ∧ 
       (S 6 = S 7 ∧ ∀ m, m > 7 → S m < S 6) := 
sorry

end number_of_correct_conclusions_l3_3582


namespace sin_330_eq_neg_sin_30_l3_3985

theorem sin_330_eq_neg_sin_30 :
  sin (330 : ℝ) = - sin (30 : ℝ) := sorry

end sin_330_eq_neg_sin_30_l3_3985


namespace range_of_a_l3_3111

noncomputable def f (a x : ℝ) : ℝ := x^3 + a * x^2 + (a + 6) * x + 1

theorem range_of_a (a : ℝ) (h : ∀ x, ∃ y, y = (3 : ℝ) * x^2 + 2 * a * x + (a + 6) ∧ (y = 0)) :
  (a < -3 ∨ a > 6) :=
by { sorry }

end range_of_a_l3_3111


namespace no_real_b_for_inequality_l3_3527

theorem no_real_b_for_inequality : ¬ ∃ b : ℝ, (∃ x : ℝ, |x^2 + 3 * b * x + 4 * b| = 5 ∧ ∀ y : ℝ, y ≠ x → |y^2 + 3 * b * y + 4 * b| > 5) := sorry

end no_real_b_for_inequality_l3_3527


namespace mixed_groups_count_l3_3029

theorem mixed_groups_count :
  ∀ (total_children groups_of_3 total_photos boys_photos girls_photos : ℕ),
  total_children = 300 ∧
  groups_of_3 = 100 ∧
  total_photos = 300 ∧
  boys_photos = 100 ∧
  girls_photos = 56 →
  let mixed_photos := total_photos - boys_photos - girls_photos in
  let mixed_groups := mixed_photos / 2 in
  mixed_groups = 72 :=
by
  intros total_children groups_of_3 total_photos boys_photos girls_photos h,
  have h1 : mixed_photos = total_photos - boys_photos - girls_photos := rfl,
  have h2 : mixed_groups = mixed_photos / 2 := rfl,
  rw [h1, h2],
  simp [h],
  sorry

end mixed_groups_count_l3_3029


namespace airline_route_same_republic_exists_l3_3445

theorem airline_route_same_republic_exists
  (cities : Finset ℕ) (republics : Finset (Finset ℕ)) (routes : ℕ → ℕ → Prop)
  (H1 : cities.card = 100)
  (H2 : ∃ R1 R2 R3 : Finset ℕ, R1 ∈ republics ∧ R2 ∈ republics ∧ R3 ∈ republics ∧
        R1 ≠ R2 ∧ R2 ≠ R3 ∧ R1 ≠ R3 ∧ 
        (∀ (R : Finset ℕ), R ∈ republics → R.card ≤ 30) ∧ 
        R1 ∪ R2 ∪ R3 = cities)
  (H3 : ∃ (S : Finset ℕ), S ⊆ cities ∧ 70 ≤ S.card ∧ 
        (∀ x ∈ S, (routes x).filter (λ y, y ∈ cities).card ≥ 70)) :
  ∃ (x y : ℕ), x ∈ cities ∧ y ∈ cities ∧ (x = y ∨ ∃ R ∈ republics, x ∈ R ∧ y ∈ R) ∧ routes x y :=
begin
  sorry
end

end airline_route_same_republic_exists_l3_3445


namespace sin_330_eq_neg_sqrt3_div_2_l3_3996

theorem sin_330_eq_neg_sqrt3_div_2 
  (R : ℝ × ℝ)
  (hR : R = (1/2, -sqrt(3)/2))
  : Real.sin (330 * Real.pi / 180) = -sqrt(3)/2 :=
by
  sorry

end sin_330_eq_neg_sqrt3_div_2_l3_3996


namespace gcf_7_factorial_8_factorial_l3_3403

theorem gcf_7_factorial_8_factorial :
  let factorial (n : ℕ) := Nat.factorial n in
  let seven_factorial := factorial 7 in
  let eight_factorial := factorial 8 in
  ∃ (gcf : ℕ), gcf = Nat.gcd seven_factorial eight_factorial ∧ gcf = 5040 :=
by
  let factorial (n : ℕ) := Nat.factorial n
  let seven_factorial := factorial 7
  let eight_factorial := factorial 8
  have seven_factorial_eq : seven_factorial = 5040 := by sorry
  have gcf_eq_seven_factorial : Nat.gcd seven_factorial eight_factorial = seven_factorial := by sorry
  exact ⟨seven_factorial, gcf_eq_seven_factorial, seven_factorial_eq⟩

end gcf_7_factorial_8_factorial_l3_3403


namespace quadratic_two_distinct_real_roots_l3_3135

theorem quadratic_two_distinct_real_roots (k : ℝ) : 
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ x^2 - 6 * x + k = 0) ↔ k < 9 :=
by
  sorry

end quadratic_two_distinct_real_roots_l3_3135


namespace sin_330_l3_3930

theorem sin_330 : Real.sin (330 * Real.pi / 180) = -1 / 2 := 
by
  sorry

end sin_330_l3_3930


namespace intersecting_lines_l3_3482

-- Definitions based on conditions
def line1 (m : ℝ) (x : ℝ) : ℝ := m * x + 4
def line2 (b : ℝ) (x : ℝ) : ℝ := 3 * x + b

-- Lean 4 Statement of the problem
theorem intersecting_lines (m b : ℝ) (h1 : line1 m 6 = 10) (h2 : line2 b 6 = 10) : b + m = -7 :=
by
  sorry

end intersecting_lines_l3_3482


namespace probability_of_diamond_king_ace_l3_3190

noncomputable def probability_three_cards : ℚ :=
  (11 / 52) * (4 / 51) * (4 / 50) + 
  (1 / 52) * (3 / 51) * (4 / 50) + 
  (1 / 52) * (4 / 51) * (3 / 50)

theorem probability_of_diamond_king_ace :
  probability_three_cards = 284 / 132600 := 
by
  sorry

end probability_of_diamond_king_ace_l3_3190


namespace rectangle_side_excess_percentage_l3_3573

theorem rectangle_side_excess_percentage (A B : ℝ) (x : ℝ) (h : A * (1 + x) * B * (1 - 0.04) = A * B * 1.008) : x = 0.05 :=
by
  sorry

end rectangle_side_excess_percentage_l3_3573


namespace angle_measure_l3_3636

theorem angle_measure (x : ℝ) : 
  (180 - x = 7 * (90 - x)) → x = 75 :=
by
  sorry

end angle_measure_l3_3636


namespace sum_of_ages_l3_3590

variables (M A : ℕ)

def Maria_age_relation : Prop :=
  M = A + 8

def future_age_relation : Prop :=
  M + 10 = 3 * (A - 6)

theorem sum_of_ages (h₁ : Maria_age_relation M A) (h₂ : future_age_relation M A) : M + A = 44 :=
by
  sorry

end sum_of_ages_l3_3590


namespace range_of_a_for_domain_of_f_l3_3430

open Real

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := sqrt (-5 / (a * x^2 + a * x - 3))

theorem range_of_a_for_domain_of_f :
  {a : ℝ | ∀ x : ℝ, a * x^2 + a * x - 3 < 0} = {a : ℝ | -12 < a ∧ a ≤ 0} :=
by
  sorry

end range_of_a_for_domain_of_f_l3_3430


namespace sin_330_eq_neg_half_l3_3835

theorem sin_330_eq_neg_half : sin (330 * (π / 180)) = -1 / 2 := 
by
  -- condition: \(330^\circ = 360^\circ - 30^\circ\)
  -- condition: \(\sin 30^\circ = \frac{1}{2}\)
  -- condition: sine is negative in the fourth quadrant
  sorry

end sin_330_eq_neg_half_l3_3835


namespace minimum_rows_required_l3_3328

theorem minimum_rows_required (total_students : ℕ) (max_students_per_school : ℕ) (seats_per_row : ℕ) (num_schools : ℕ) 
    (h_total_students : total_students = 2016) 
    (h_max_students_per_school : max_students_per_school = 45) 
    (h_seats_per_row : seats_per_row = 168) 
    (h_num_schools : num_schools = 46) : 
    ∃ (min_rows : ℕ), min_rows = 16 := 
by 
  -- Proof omitted
  sorry

end minimum_rows_required_l3_3328


namespace smallest_integer_x_l3_3342

-- Conditions
def condition1 (x : ℤ) : Prop := 7 - 5 * x < 25
def condition2 (x : ℤ) : Prop := ∃ y : ℤ, y = 10 ∧ y - 3 * x > 6

-- Statement
theorem smallest_integer_x : ∃ x : ℤ, condition1 x ∧ condition2 x ∧ ∀ z : ℤ, condition1 z ∧ condition2 z → x ≤ z :=
  sorry

end smallest_integer_x_l3_3342


namespace tea_leaves_costs_l3_3506

theorem tea_leaves_costs (a_1 b_1 a_2 b_2 : ℕ) (c_A c_B : ℝ) :
  a_1 * c_A = 4000 ∧ 
  b_1 * c_B = 8400 ∧ 
  b_1 = a_1 + 10 ∧ 
  c_B = 1.4 * c_A ∧ 
  a_2 + b_2 = 100 ∧ 
  (300 - c_A) * (a_2 / 2) + (300 * 0.7 - c_A) * (a_2 / 2) + 
  (400 - c_B) * (b_2 / 2) + (400 * 0.7 - c_B) * (b_2 / 2) = 5800 
  → c_A = 200 ∧ c_B = 280 ∧ a_2 = 40 ∧ b_2 = 60 := 
sorry

end tea_leaves_costs_l3_3506


namespace polynomial_factorization_l3_3395

theorem polynomial_factorization (x : ℝ) :
  x^6 + 6*x^5 + 15*x^4 + 20*x^3 + 15*x^2 + 6*x + 1 = (x + 1)^6 :=
by {
  -- proof goes here
  sorry
}

end polynomial_factorization_l3_3395


namespace sin_330_eq_neg_one_half_l3_3883

theorem sin_330_eq_neg_one_half : sin (330 * π / 180) = -1 / 2 := by
  sorry

end sin_330_eq_neg_one_half_l3_3883


namespace seating_arrangement_count_l3_3504

def seating_arrangements : ℕ :=
  (5.fact) * (Nat.choose 6 4) * (4.fact)

theorem seating_arrangement_count : seating_arrangements = 43200 := by
  sorry

end seating_arrangement_count_l3_3504


namespace intersection_A_B_l3_3153

-- Define the set A
def A : Set (ℝ × ℝ) := { p | ∃ x : ℝ, p = (x, x + 1) }

-- Define the set B
def B : Set (ℝ × ℝ) := { p | ∃ x : ℝ, p = (x, -2*x + 4) }

-- State the theorem to prove A ∩ B = {(1, 2)}
theorem intersection_A_B : A ∩ B = { (1, 2) } :=
by
  sorry

end intersection_A_B_l3_3153


namespace sin_330_deg_l3_3848

theorem sin_330_deg : Real.sin (330 * Real.pi / 180) = -1 / 2 := by
  sorry

end sin_330_deg_l3_3848


namespace sin_330_eq_neg_one_half_l3_3909

theorem sin_330_eq_neg_one_half :
  ∃ θ : ℝ, sin (330 * real.pi / 180) = -1 / 2 :=
begin
  have h1 : sin (360 * real.pi / 180 - 30 * real.pi / 180) = -sin (30 * real.pi / 180),
  { rw sin_sub,
    ring,
    rw sin_pi_div_six, -- sin 30° = 1 / 2
  },
  have h2 : sin (30 * real.pi / 180) = 1 / 2 := by exact real.sin_pi_div_six,
  use 330,
  calc
    sin (330 * real.pi / 180)
        = sin ((360 - 30) * real.pi / 180)      : by ring
    ... = sin (360 * real.pi / 180 - 30 * real.pi / 180) : by congr' 1; ring
    ... = -sin (30 * real.pi / 180)            : by exact h1
    ... = -1 / 2                               : by rw h2,
end

end sin_330_eq_neg_one_half_l3_3909


namespace sin_330_eq_neg_one_half_l3_3865

theorem sin_330_eq_neg_one_half :
  ∃ θ, θ = 330 ∧ (sin (Real.pi * (θ / 180)) = -1 / 2) :=
by
  use 330
  split
  · rfl
  · sorry

end sin_330_eq_neg_one_half_l3_3865


namespace permits_cost_l3_3741

-- Definitions based on conditions
def total_cost : ℕ := 2950
def contractor_hourly_rate : ℕ := 150
def contractor_hours_per_day : ℕ := 5
def contractor_days : ℕ := 3
def inspector_discount_rate : ℕ := 80

-- Proving the cost of permits
theorem permits_cost : ∃ (permits_cost : ℕ), permits_cost = 250 :=
by
  let contractor_hours := contractor_days * contractor_hours_per_day
  let contractor_cost := contractor_hours * contractor_hourly_rate
  let inspector_hourly_rate := contractor_hourly_rate - (contractor_hourly_rate * inspector_discount_rate / 100)
  let inspector_cost := contractor_hours * inspector_hourly_rate
  let total_cost_without_permits := contractor_cost + inspector_cost
  let permits_cost := total_cost - total_cost_without_permits
  use permits_cost
  sorry

end permits_cost_l3_3741


namespace absolute_value_of_neg_eight_l3_3470

/-- Absolute value of a number is the distance from 0 on the number line. -/
def absolute_value (x : ℤ) : ℤ :=
  if x >= 0 then x else -x

theorem absolute_value_of_neg_eight : absolute_value (-8) = 8 := by
  -- Proof is omitted
  sorry

end absolute_value_of_neg_eight_l3_3470


namespace rectangle_area_l3_3004

-- Definitions of the conditions
variables (Length Width Area : ℕ)
variable (h1 : Length = 4 * Width)
variable (h2 : Length = 20)

-- Statement to prove
theorem rectangle_area : Area = Length * Width → Area = 100 :=
by
  sorry

end rectangle_area_l3_3004


namespace mad_hatter_wait_time_l3_3462

theorem mad_hatter_wait_time 
  (mad_hatter_fast_rate : ℝ := 15 / 60)
  (march_hare_slow_rate : ℝ := 10 / 60)
  (meeting_time : ℝ := 5) :
  let mad_hatter_real_time := meeting_time * (60 / (60 + mad_hatter_fast_rate * 60)),
      march_hare_real_time := meeting_time * (60 / (60 - march_hare_slow_rate * 60)),
      waiting_time := march_hare_real_time - mad_hatter_real_time
  in waiting_time = 2 :=
by 
  sorry

end mad_hatter_wait_time_l3_3462


namespace sampling_method_is_systematic_l3_3658

-- Definition of the conditions
def factory_produces_product := True  -- Assuming the factory is always producing
def uses_conveyor_belt := True  -- Assuming the conveyor belt is always in use
def samples_taken_every_10_minutes := True  -- Sampling at specific intervals

-- Definition corresponding to the systematic sampling
def systematic_sampling := True

-- Theorem: Prove that given the conditions, the sampling method is systematic sampling.
theorem sampling_method_is_systematic :
  factory_produces_product → uses_conveyor_belt → samples_taken_every_10_minutes → systematic_sampling :=
by
  intros _ _ _
  trivial

end sampling_method_is_systematic_l3_3658


namespace number_of_strictly_increasing_sequences_l3_3703

def strictly_increasing_sequences (n : ℕ) : ℕ :=
if n = 0 then 1 else if n = 1 then 1 else strictly_increasing_sequences (n - 1) + strictly_increasing_sequences (n - 2)

theorem number_of_strictly_increasing_sequences :
  strictly_increasing_sequences 12 = 144 :=
by
  sorry

end number_of_strictly_increasing_sequences_l3_3703


namespace triplet_solution_l3_3694

theorem triplet_solution (a b c : ℝ)
  (h1 : a^2 + b = c^2)
  (h2 : b^2 + c = a^2)
  (h3 : c^2 + a = b^2) :
  (a = 0 ∧ b = 0 ∧ c = 0) ∨
  (a = 0 ∧ b = 1 ∧ c = -1) ∨
  (a = -1 ∧ b = 0 ∧ c = 1) ∨
  (a = 1 ∧ b = -1 ∧ c = 0) :=
sorry

end triplet_solution_l3_3694


namespace value_of_expression_l3_3185

theorem value_of_expression : 8 * (6 - 4) + 2 = 18 := by
  sorry

end value_of_expression_l3_3185


namespace proposition_false_n4_l3_3717

variable {P : ℕ → Prop}

theorem proposition_false_n4
  (h_ind : ∀ (k : ℕ), k ≠ 0 → P k → P (k + 1))
  (h_false_5 : P 5 = False) :
  P 4 = False :=
sorry

end proposition_false_n4_l3_3717


namespace clinton_shoes_count_l3_3385

theorem clinton_shoes_count : 
  let hats := 5
  let belts := hats + 2
  let shoes := 2 * belts
  shoes = 14 := 
by
  -- Define the number of hats
  let hats := 5
  -- Define the number of belts
  let belts := hats + 2
  -- Define the number of shoes
  let shoes := 2 * belts
  -- Assert that the number of shoes is 14
  show shoes = 14 from sorry

end clinton_shoes_count_l3_3385


namespace problem1_l3_3825

theorem problem1 : abs (-3) + (-1: ℤ)^2021 * (Real.pi - 3.14)^0 - (- (1/2: ℝ))⁻¹ = 4 := 
  sorry

end problem1_l3_3825


namespace sin_330_degree_l3_3948

theorem sin_330_degree : Real.sin (330 * Real.pi / 180) = -1 / 2 :=
by
  sorry

end sin_330_degree_l3_3948


namespace girl_buys_roses_l3_3659

theorem girl_buys_roses 
  (x y : ℤ)
  (h1 : y = 1)
  (h2 : x > 0)
  (h3 : (200 : ℤ) / (x + 10) < (100 : ℤ) / x)
  (h4 : (80 : ℤ) / 12 = ((100 : ℤ) / x) - ((200 : ℤ) / (x + 10))) :
  x = 5 ∧ y = 1 :=
by
  sorry

end girl_buys_roses_l3_3659


namespace min_xy_l3_3436

variable {x y : ℝ}

theorem min_xy (hx : x > 0) (hy : y > 0) (h : 10 * x + 2 * y + 60 = x * y) : x * y ≥ 180 := 
sorry

end min_xy_l3_3436


namespace rightmost_four_digits_of_5_pow_2023_l3_3336

theorem rightmost_four_digits_of_5_pow_2023 :
  (5 ^ 2023) % 10000 = 8125 :=
sorry

end rightmost_four_digits_of_5_pow_2023_l3_3336


namespace arc_length_of_f_l3_3378

noncomputable def f (x : ℝ) : ℝ := 2 - Real.exp x

theorem arc_length_of_f :
  ∫ x in Real.log (Real.sqrt 3)..Real.log (Real.sqrt 8), Real.sqrt (1 + (Real.exp x)^2) = 1 + 1/2 * Real.log (3 / 2) :=
by
  sorry

end arc_length_of_f_l3_3378


namespace side_length_square_l3_3762

theorem side_length_square (s : ℝ) (h1 : ∃ (s : ℝ), (s > 0)) (h2 : 6 * s^2 = 3456) : s = 24 :=
sorry

end side_length_square_l3_3762


namespace count_integer_values_not_satisfying_inequality_l3_3106

theorem count_integer_values_not_satisfying_inequality : 
  ∃ n : ℕ, 
  (n = 3) ∧ (∀ x : ℤ, (4 * x^2 + 22 * x + 21 ≤ 25) → (-2 ≤ x ∧ x ≤ 0)) :=
by
  sorry

end count_integer_values_not_satisfying_inequality_l3_3106


namespace sin_330_eq_neg_half_l3_3826

theorem sin_330_eq_neg_half : sin (330 * (π / 180)) = -1 / 2 := 
by
  -- condition: \(330^\circ = 360^\circ - 30^\circ\)
  -- condition: \(\sin 30^\circ = \frac{1}{2}\)
  -- condition: sine is negative in the fourth quadrant
  sorry

end sin_330_eq_neg_half_l3_3826


namespace ferris_break_length_l3_3650

-- Definitions of the given conditions
def audrey_work_rate := 1 / 4  -- Audrey completes 1/4 of the job per hour
def ferris_work_rate := 1 / 3  -- Ferris completes 1/3 of the job per hour
def total_work_time := 2       -- They worked together for 2 hours
def num_breaks := 6            -- Ferris took 6 breaks during the work period

-- The theorem to prove the length of each break Ferris took
theorem ferris_break_length (break_length : ℝ) :
  (audrey_work_rate * total_work_time) + 
  (ferris_work_rate * (total_work_time - (break_length / 60) * num_breaks)) = 1 →
  break_length = 2.5 :=
by
  sorry

end ferris_break_length_l3_3650


namespace lisa_total_distance_l3_3747

-- Definitions for distances and counts of trips
def plane_distance : ℝ := 256.0
def train_distance : ℝ := 120.5
def bus_distance : ℝ := 35.2

def plane_trips : ℕ := 32
def train_trips : ℕ := 16
def bus_trips : ℕ := 42

-- Definition of total distance traveled
def total_distance_traveled : ℝ :=
  (plane_distance * plane_trips)
  + (train_distance * train_trips)
  + (bus_distance * bus_trips)

-- The statement to be proven
theorem lisa_total_distance :
  total_distance_traveled = 11598.4 := by
  sorry

end lisa_total_distance_l3_3747


namespace age_ratio_in_years_l3_3036

variable (s d x : ℕ)

theorem age_ratio_in_years (h1 : s - 3 = 2 * (d - 3)) (h2 : s - 7 = 3 * (d - 7)) (hx : (s + x) = 3 * (d + x) / 2) : x = 5 := sorry

end age_ratio_in_years_l3_3036


namespace extreme_value_and_tangent_line_l3_3429

noncomputable def f (a b x : ℝ) : ℝ := a * x^3 + b * x^2 - 3 * x

theorem extreme_value_and_tangent_line (a b : ℝ) (h1 : f a b 1 = 0) (h2 : f a b (-1) = 0) :
  (f 1 0 (-1) = 2) ∧ (f 1 0 1 = -2) ∧ (∀ x : ℝ, x = -2 → (9 * x - (x^3 - 3 * x) + 16 = 0)) :=
by
  sorry

end extreme_value_and_tangent_line_l3_3429


namespace bernardo_wins_l3_3676

/-- 
Bernardo and Silvia play the following game. An integer between 0 and 999 inclusive is selected
and given to Bernardo. Whenever Bernardo receives a number, he doubles it and passes the result 
to Silvia. Whenever Silvia receives a number, she adds 50 to it and passes the result back. 
The winner is the last person who produces a number less than 1000. The smallest initial number 
that results in a win for Bernardo is 16, and the sum of the digits of 16 is 7.
-/
def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

theorem bernardo_wins (N : ℕ) (h : 16 ≤ N ∧ N ≤ 18) : sum_of_digits 16 = 7 :=
by
  sorry

end bernardo_wins_l3_3676


namespace sin_330_l3_3902

theorem sin_330 : Real.sin (330 * Real.pi / 180) = -1 / 2 := 
by
  -- Outline the proof here without providing it
  -- sorry to delay the proof
  sorry

end sin_330_l3_3902


namespace three_digit_sum_9_l3_3422

theorem three_digit_sum_9 : 
  {abc : ℕ // 100 ≤ abc ∧ abc < 1000 ∧ (abc.digits 10).sum = 9}.card = 45 := 
by
  sorry

end three_digit_sum_9_l3_3422


namespace ArithmeticSequenceSum_l3_3546

theorem ArithmeticSequenceSum (a : ℕ → ℕ) (d : ℕ) 
  (h1 : a 1 + a 2 = 10) 
  (h2 : a 4 = a 3 + 2)
  (h3 : ∀ n : ℕ, a n = a 1 + (n - 1) * d) :
  a 3 + a 4 = 18 :=
by
  sorry

end ArithmeticSequenceSum_l3_3546


namespace sin_330_eq_neg_half_l3_3924

open Real

theorem sin_330_eq_neg_half : sin (330 * (π / 180)) = - (1 / 2) := by
  have angle_eq : 330 * (π / 180) = 2 * π - (30 * (π / 180)) := by sorry
  have sin_30_pos : sin (30 * (π / 180)) = 1 / 2 := by sorry
  have sin_330_neg : sin (2 * π - (30 * (π / 180))) = - (sin (30 * (π / 180))) := by sorry
  rw [angle_eq, sin_330_neg, sin_30_pos]
  rfl

end sin_330_eq_neg_half_l3_3924


namespace sin_330_eq_neg_one_half_l3_3879

theorem sin_330_eq_neg_one_half : sin (330 * π / 180) = -1 / 2 := by
  sorry

end sin_330_eq_neg_one_half_l3_3879


namespace tilly_star_count_l3_3040

theorem tilly_star_count (stars_east : ℕ) (stars_west : ℕ) (total_stars : ℕ) 
  (h1 : stars_east = 120)
  (h2 : stars_west = 6 * stars_east)
  (h3 : total_stars = stars_east + stars_west) :
  total_stars = 840 :=
sorry

end tilly_star_count_l3_3040


namespace problem_solution_l3_3530

noncomputable def solve_equation (x : ℝ) : Prop :=
  x ≠ 4 ∧ (x + 36 / (x - 4) = -9)

theorem problem_solution : {x : ℝ | solve_equation x} = {0, -5} :=
by
  sorry

end problem_solution_l3_3530


namespace distance_covered_downstream_l3_3210

-- Conditions
def boat_speed_still_water : ℝ := 16
def stream_rate : ℝ := 5
def time_downstream : ℝ := 6

-- Effective speed downstream
def effective_speed_downstream := boat_speed_still_water + stream_rate

-- Distance covered downstream
def distance_downstream := effective_speed_downstream * time_downstream

-- Theorem to prove
theorem distance_covered_downstream :
  (distance_downstream = 126) :=
by
  sorry

end distance_covered_downstream_l3_3210


namespace sin_330_eq_neg_one_half_l3_3864

theorem sin_330_eq_neg_one_half :
  ∃ θ, θ = 330 ∧ (sin (Real.pi * (θ / 180)) = -1 / 2) :=
by
  use 330
  split
  · rfl
  · sorry

end sin_330_eq_neg_one_half_l3_3864


namespace sin_330_eq_neg_half_l3_3833

theorem sin_330_eq_neg_half : sin (330 * (π / 180)) = -1 / 2 := 
by
  -- condition: \(330^\circ = 360^\circ - 30^\circ\)
  -- condition: \(\sin 30^\circ = \frac{1}{2}\)
  -- condition: sine is negative in the fourth quadrant
  sorry

end sin_330_eq_neg_half_l3_3833


namespace no_real_intersection_l3_3746

theorem no_real_intersection (f : ℝ → ℝ) 
  (h1 : ∀ x y : ℝ, x * f y = y * f x) 
  (h2 : f 1 = -1) : ¬∃ x : ℝ, f x = x^2 + 1 :=
by
  sorry

end no_real_intersection_l3_3746


namespace stone_counting_l3_3174

theorem stone_counting (n : ℕ) (m : ℕ) : 
    10 > 0 ∧  (n ≡ 6 [MOD 20]) ∧ m = 126 → n = 6 := 
by
  sorry

end stone_counting_l3_3174


namespace angle_measure_supplement_complement_l3_3634

theorem angle_measure_supplement_complement (x : ℝ) 
    (h1 : 180 - x = 7 * (90 - x)) : 
    x = 75 := by
  sorry

end angle_measure_supplement_complement_l3_3634
