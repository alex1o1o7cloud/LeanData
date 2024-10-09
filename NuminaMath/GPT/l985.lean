import Mathlib

namespace total_boxes_is_4575_l985_98589

-- Define the number of boxes in each warehouse
def num_boxes_in_warehouse_A (x : ℕ) := x
def num_boxes_in_warehouse_B (x : ℕ) := 3 * x
def num_boxes_in_warehouse_C (x : ℕ) := (3 * x) / 2 + 100
def num_boxes_in_warehouse_D (x : ℕ) := 2 * ((3 * x) / 2 + 100) - 50
def num_boxes_in_warehouse_E (x : ℕ) := x + (2 * ((3 * x) / 2 + 100) - 50) - 200

-- Define the condition that warehouse B has 300 more boxes than warehouse E
def condition_B_E (x : ℕ) := 3 * x = num_boxes_in_warehouse_E x + 300

-- Define the total number of boxes calculation
def total_boxes (x : ℕ) := 
    num_boxes_in_warehouse_A x +
    num_boxes_in_warehouse_B x +
    num_boxes_in_warehouse_C x +
    num_boxes_in_warehouse_D x +
    num_boxes_in_warehouse_E x

-- The statement of the problem
theorem total_boxes_is_4575 (x : ℕ) (h : condition_B_E x) : total_boxes x = 4575 :=
by
    sorry

end total_boxes_is_4575_l985_98589


namespace carbon_emission_l985_98585

theorem carbon_emission (x y : ℕ) (h1 : x + y = 70) (h2 : x = 5 * y - 8) : y = 13 ∧ x = 57 := by
  sorry

end carbon_emission_l985_98585


namespace williams_tips_fraction_l985_98539

theorem williams_tips_fraction
  (A : ℝ) -- average tips for months other than August
  (h : ∀ A, A > 0) -- assuming some positivity constraint for non-degenerate mean
  (h_august : A ≠ 0) -- assuming average can’t be zero
  (august_tips : ℝ := 10 * A)
  (other_months_tips : ℝ := 6 * A)
  (total_tips : ℝ := 16 * A) :
  (august_tips / total_tips) = (5 / 8) := 
sorry

end williams_tips_fraction_l985_98539


namespace composite_square_perimeter_l985_98574

theorem composite_square_perimeter (p1 p2 : ℝ) (h1 : p1 = 40) (h2 : p2 = 100) : 
  let s1 := p1 / 4
  let s2 := p2 / 4
  (p1 + p2 - 2 * s1) = 120 := 
by
  -- proof goes here
  sorry

end composite_square_perimeter_l985_98574


namespace positive_integer_power_of_two_l985_98564

theorem positive_integer_power_of_two (n : ℕ) (hn : 0 < n) :
  (∃ m : ℤ, (2^n - 1) ∣ (m^2 + 9)) ↔ (∃ k : ℕ, n = 2^k) :=
by
  sorry

end positive_integer_power_of_two_l985_98564


namespace houses_with_only_one_pet_l985_98588

theorem houses_with_only_one_pet (h_total : ∃ t : ℕ, t = 75)
                                 (h_dogs : ∃ d : ℕ, d = 40)
                                 (h_cats : ∃ c : ℕ, c = 30)
                                 (h_dogs_and_cats : ∃ dc : ℕ, dc = 10)
                                 (h_birds : ∃ b : ℕ, b = 8)
                                 (h_cats_and_birds : ∃ cb : ℕ, cb = 5)
                                 (h_no_dogs_and_birds : ∀ db : ℕ, ¬ (∃ db : ℕ, db = 1)) :
  ∃ n : ℕ, n = 48 :=
by
  have only_dogs := 40 - 10
  have only_cats := 30 - 10 - 5
  have only_birds := 8 - 5
  have result := only_dogs + only_cats + only_birds
  exact ⟨result, sorry⟩

end houses_with_only_one_pet_l985_98588


namespace factorize_poly1_factorize_poly2_l985_98598

variable (a b m n : ℝ)

theorem factorize_poly1 : 3 * a^2 - 6 * a * b + 3 * b^2 = 3 * (a - b)^2 :=
sorry

theorem factorize_poly2 : 4 * m^2 - 9 * n^2 = (2 * m - 3 * n) * (2 * m + 3 * n) :=
sorry

end factorize_poly1_factorize_poly2_l985_98598


namespace sum_even_minus_odd_from_1_to_100_l985_98582

noncomputable def sum_even_numbers : Nat :=
  (List.range' 2 99 2).sum

noncomputable def sum_odd_numbers : Nat :=
  (List.range' 1 100 2).sum

theorem sum_even_minus_odd_from_1_to_100 :
  sum_even_numbers - sum_odd_numbers = 50 :=
by
  sorry

end sum_even_minus_odd_from_1_to_100_l985_98582


namespace triangle_solution_proof_l985_98599

noncomputable def solve_triangle_proof (a b c : ℝ) (alpha beta gamma : ℝ) : Prop :=
  a = 631.28 ∧
  alpha = 63 + 35 / 60 + 30 / 3600 ∧
  b - c = 373 ∧
  beta = 88 + 12 / 60 + 15 / 3600 ∧
  gamma = 28 + 12 / 60 + 15 / 3600 ∧
  b = 704.55 ∧
  c = 331.55

theorem triangle_solution_proof : solve_triangle_proof 631.28 704.55 331.55 (63 + 35 / 60 + 30 / 3600) (88 + 12 / 60 + 15 / 3600) (28 + 12 / 60 + 15 / 3600) :=
  by { sorry }

end triangle_solution_proof_l985_98599


namespace negation_of_forall_geq_l985_98573

theorem negation_of_forall_geq {x : ℝ} : ¬ (∀ x : ℝ, x^2 - x ≥ 0) ↔ ∃ x : ℝ, x^2 - x < 0 :=
by
  sorry

end negation_of_forall_geq_l985_98573


namespace contradiction_assumption_l985_98596

variable (x y z : ℝ)

/-- The negation of "at least one is positive" for proof by contradiction is 
    "all three numbers are non-positive". -/
theorem contradiction_assumption (h : ¬ (x > 0 ∨ y > 0 ∨ z > 0)) : 
  (x ≤ 0 ∧ y ≤ 0 ∧ z ≤ 0) :=
by
  sorry

end contradiction_assumption_l985_98596


namespace min_value_of_f_l985_98538

-- Define the function f
def f (a b c x y z : ℤ) : ℤ := a * x + b * y + c * z

-- Define the gcd function for three integers
def gcd3 (a b c : ℕ) : ℕ := Nat.gcd (Nat.gcd a b) c

-- Define the main theorem to prove
theorem min_value_of_f (a b c : ℕ) (a_pos : a > 0) (b_pos : b > 0) (c_pos : c > 0) :
  ∃ (x y z : ℤ), f a b c x y z = gcd3 a b c := 
by
  sorry

end min_value_of_f_l985_98538


namespace triangle_perimeter_upper_bound_l985_98595

theorem triangle_perimeter_upper_bound (a b : ℕ) (s : ℕ) (h₁ : a = 7) (h₂ : b = 23) 
  (h₃ : 16 < s) (h₄ : s < 30) : 
  ∃ n : ℕ, n = 60 ∧ n > a + b + s := 
by
  sorry

end triangle_perimeter_upper_bound_l985_98595


namespace frog_jumps_10_inches_more_than_grasshopper_frog_jumps_10_inches_farther_than_grasshopper_l985_98566

-- Definitions of conditions
def grasshopper_jump : ℕ := 19
def mouse_jump_frog (frog_jump : ℕ) : ℕ := frog_jump + 20
def mouse_jump_grasshopper : ℕ := grasshopper_jump + 30

-- The proof problem statement
theorem frog_jumps_10_inches_more_than_grasshopper (frog_jump : ℕ) :
  mouse_jump_frog frog_jump = mouse_jump_grasshopper → frog_jump = 29 :=
by
  sorry

-- The ultimate question in the problem
theorem frog_jumps_10_inches_farther_than_grasshopper : 
  (∃ (frog_jump : ℕ), frog_jump = 29) → (frog_jump - grasshopper_jump = 10) :=
by
  sorry

end frog_jumps_10_inches_more_than_grasshopper_frog_jumps_10_inches_farther_than_grasshopper_l985_98566


namespace visible_steps_on_escalator_l985_98576

variable (steps_visible : ℕ) -- The number of steps visible on the escalator
variable (al_steps : ℕ := 150) -- Al walks down 150 steps
variable (bob_steps : ℕ := 75) -- Bob walks up 75 steps
variable (al_speed : ℕ := 3) -- Al's walking speed
variable (bob_speed : ℕ := 1) -- Bob's walking speed
variable (escalator_speed : ℚ) -- The speed of the escalator

theorem visible_steps_on_escalator : steps_visible = 120 :=
by
  -- Define times taken by Al and Bob
  let al_time := al_steps / al_speed
  let bob_time := bob_steps / bob_speed

  -- Define effective speeds considering escalator speed 'escalator_speed'
  let al_effective_speed := al_speed - escalator_speed
  let bob_effective_speed := bob_speed + escalator_speed

  -- Calculate the total steps walked if the escalator was stopped (same total steps)
  have al_total_steps := al_effective_speed * al_time
  have bob_total_steps := bob_effective_speed * bob_time

  -- Set up the equation
  have eq := al_total_steps = bob_total_steps

  -- Substitute and solve for escalator_speed
  sorry

end visible_steps_on_escalator_l985_98576


namespace opposite_of_six_is_neg_six_l985_98546

-- Define the condition that \( a \) is the opposite of \( 6 \)
def is_opposite_of_six (a : Int) : Prop := a = -6

-- Prove that \( a = -6 \) given that \( a \) is the opposite of \( 6 \)
theorem opposite_of_six_is_neg_six (a : Int) (h : is_opposite_of_six a) : a = -6 :=
by
  sorry

end opposite_of_six_is_neg_six_l985_98546


namespace claudia_filled_5oz_glasses_l985_98504

theorem claudia_filled_5oz_glasses :
  ∃ (n : ℕ), n = 6 ∧ 4 * 8 + 15 * 4 + n * 5 = 122 :=
by
  sorry

end claudia_filled_5oz_glasses_l985_98504


namespace scientific_notation_correct_l985_98509

def distance_moon_km : ℕ := 384000

def scientific_notation (n : ℕ) : ℝ := 3.84 * 10^5

theorem scientific_notation_correct : scientific_notation distance_moon_km = 3.84 * 10^5 := by
  sorry

end scientific_notation_correct_l985_98509


namespace intersection_A_B_l985_98590

open Set

def SetA : Set ℤ := {x | ∃ n : ℤ, x = 2 * n}
def SetB : Set ℤ := {x | 0 ≤ x ∧ x ≤ 4}

theorem intersection_A_B :
  (SetA ∩ SetB) = ( {0, 2, 4} : Set ℤ ) :=
by
  sorry

end intersection_A_B_l985_98590


namespace P_subsetneq_M_l985_98517

def M := {x : ℝ | x > 1}
def P := {x : ℝ | x^2 - 6*x + 9 = 0}

theorem P_subsetneq_M : P ⊂ M := by
  sorry

end P_subsetneq_M_l985_98517


namespace fraction_of_number_l985_98518

variable (N : ℝ) (F : ℝ)

theorem fraction_of_number (h1 : 0.5 * N = F * N + 2) (h2 : N = 8.0) : F = 0.25 := by
  sorry

end fraction_of_number_l985_98518


namespace penelope_mandm_candies_l985_98557

theorem penelope_mandm_candies (m n : ℕ) (r : ℝ) :
  (m / n = 5 / 3) → (n = 15) → (m = 25) :=
by
  sorry

end penelope_mandm_candies_l985_98557


namespace cost_of_adult_ticket_is_15_l985_98550

variable (A : ℕ) -- Cost of an adult ticket
variable (total_tickets : ℕ) (cost_child_ticket : ℕ) (total_revenue : ℕ)
variable (adult_tickets_sold : ℕ)

theorem cost_of_adult_ticket_is_15
  (h1 : total_tickets = 522)
  (h2 : cost_child_ticket = 8)
  (h3 : total_revenue = 5086)
  (h4 : adult_tickets_sold = 130) 
  (h5 : (total_tickets - adult_tickets_sold) * cost_child_ticket + adult_tickets_sold * A = total_revenue) :
  A = 15 :=
by
  sorry

end cost_of_adult_ticket_is_15_l985_98550


namespace solve_r_l985_98578

theorem solve_r (k r : ℝ) (h1 : 3 = k * 2^r) (h2 : 15 = k * 4^r) : 
  r = Real.log 5 / Real.log 2 := 
sorry

end solve_r_l985_98578


namespace problem_statement_l985_98544

theorem problem_statement (g : ℝ → ℝ) (m k : ℝ) (h₀ : ∀ x, g x = 5 * x - 3)
  (h₁ : 0 < k) (h₂ : 0 < m)
  (h₃ : ∀ x, |g x - 2| < k ↔ |x - 1| < m) : m ≤ k / 5 :=
sorry

end problem_statement_l985_98544


namespace range_of_a_l985_98579

theorem range_of_a (a : ℝ) : (∃ x y : ℝ, x^2 / a^2 + y^2 / (a + 6) = 1 ∧ (a^2 > a + 6 ∧ a + 6 > 0)) → (a > 3 ∨ (-6 < a ∧ a < -2)) :=
by
  intro h
  sorry

end range_of_a_l985_98579


namespace polynomial_roots_r_eq_18_l985_98531

theorem polynomial_roots_r_eq_18
  (a b c : ℂ) 
  (h_roots : Polynomial.roots (Polynomial.C (0 : ℂ) * Polynomial.X^3 + Polynomial.C (5 : ℂ) * Polynomial.X^2 + Polynomial.C (2 : ℂ) * Polynomial.X + Polynomial.C (-8 : ℂ)) = {a, b, c}) 
  (h_ab_roots : Polynomial.roots (Polynomial.C (0 : ℂ) * Polynomial.X^3 + Polynomial.C p * Polynomial.X^2 + Polynomial.C q * Polynomial.X + Polynomial.C r) = {2 * a + b, 2 * b + c, 2 * c + a}) :
  r = 18 := sorry

end polynomial_roots_r_eq_18_l985_98531


namespace dave_non_working_games_l985_98516

def total_games : ℕ := 10
def price_per_game : ℕ := 4
def total_earnings : ℕ := 32

theorem dave_non_working_games : (total_games - (total_earnings / price_per_game)) = 2 := by
  sorry

end dave_non_working_games_l985_98516


namespace trapezoid_area_l985_98597

open Real

theorem trapezoid_area 
  (r : ℝ) (BM CD AB : ℝ) (radius_nonneg : 0 ≤ r) 
  (BM_positive : 0 < BM) (CD_positive : 0 < CD) (AB_positive : 0 < AB)
  (circle_radius : r = 4) (BM_length : BM = 16) (CD_length : CD = 3) :
  let height := 2 * r
  let base_sum := AB + CD
  let area := height * base_sum / 2
  AB = BM + 8 → area = 108 :=
by
  intro hyp
  sorry

end trapezoid_area_l985_98597


namespace buildings_collapsed_l985_98548

theorem buildings_collapsed (B : ℕ) (h₁ : 2 * B = X) (h₂ : 4 * B = Y) (h₃ : 8 * B = Z) (h₄ : B + 2 * B + 4 * B + 8 * B = 60) : B = 4 :=
by
  sorry

end buildings_collapsed_l985_98548


namespace annual_parking_savings_l985_98562

theorem annual_parking_savings :
  let weekly_rate := 10
  let monthly_rate := 40
  let weeks_in_year := 52
  let months_in_year := 12
  let annual_weekly_cost := weekly_rate * weeks_in_year
  let annual_monthly_cost := monthly_rate * months_in_year
  let savings := annual_weekly_cost - annual_monthly_cost
  savings = 40 := by
{
  sorry
}

end annual_parking_savings_l985_98562


namespace inequality_positives_l985_98572

theorem inequality_positives (x1 x2 x3 x4 x5 : ℝ) (hx1 : 0 < x1) (hx2 : 0 < x2) (hx3 : 0 < x3) (hx4 : 0 < x4) (hx5 : 0 < x5) : 
  (x1 + x2 + x3 + x4 + x5)^2 ≥ 4 * (x1 * x2 + x3 * x4 + x5 * x1 + x2 * x3 + x4 * x5) :=
sorry

end inequality_positives_l985_98572


namespace zoe_remaining_pictures_l985_98545

-- Definitions based on the conditions
def total_pictures : Nat := 88
def colored_pictures : Nat := 20

-- Proof statement
theorem zoe_remaining_pictures : total_pictures - colored_pictures = 68 := by
  sorry

end zoe_remaining_pictures_l985_98545


namespace train_speed_kph_l985_98565

-- Definitions based on conditions
def time_seconds : ℕ := 9
def length_meters : ℕ := 135
def conversion_factor : ℕ := 36 -- 3.6 represented as an integer by multiplying both sides by 10

-- The proof statement
theorem train_speed_kph : (length_meters * conversion_factor / 10 / time_seconds = 54) :=
by
  sorry

end train_speed_kph_l985_98565


namespace range_of_p_l985_98570

theorem range_of_p (p : ℝ) (a_n b_n : ℕ → ℝ)
  (ha : ∀ n, a_n n = -n + p)
  (hb : ∀ n, b_n n = 3^(n-4))
  (C_n : ℕ → ℝ)
  (hC : ∀ n, C_n n = if a_n n ≥ b_n n then a_n n else b_n n)
  (hc : ∀ n : ℕ, n ≥ 1 → C_n n > C_n 4) :
  4 < p ∧ p < 7 :=
sorry

end range_of_p_l985_98570


namespace exists_lattice_midpoint_among_five_points_l985_98567

-- Definition of lattice points
structure LatticePoint where
  x : ℤ
  y : ℤ

open LatticePoint

-- The theorem we want to prove
theorem exists_lattice_midpoint_among_five_points (A B C D E : LatticePoint) :
    ∃ P Q : LatticePoint, P ≠ Q ∧ (P.x + Q.x) % 2 = 0 ∧ (P.y + Q.y) % 2 = 0 := 
  sorry

end exists_lattice_midpoint_among_five_points_l985_98567


namespace find_x_if_friendly_l985_98512

theorem find_x_if_friendly (x : ℚ) :
    (∃ m n : ℚ, m + n = 66 ∧ m = 7 * x ∧ n = -18) →
    x = 12 :=
by
  sorry

end find_x_if_friendly_l985_98512


namespace second_horse_revolutions_l985_98559

noncomputable def circumference (radius : ℝ) : ℝ := 2 * Real.pi * radius
noncomputable def distance_traveled (circumference : ℝ) (revolutions : ℕ) : ℝ := circumference * (revolutions : ℝ)
noncomputable def revolutions_needed (distance : ℝ) (circumference : ℝ) : ℕ := ⌊distance / circumference⌋₊

theorem second_horse_revolutions :
  let r1 := 30
  let r2 := 10
  let revolutions1 := 40
  let c1 := circumference r1
  let c2 := circumference r2
  let d1 := distance_traveled c1 revolutions1
  (revolutions_needed d1 c2) = 120 :=
by
  sorry

end second_horse_revolutions_l985_98559


namespace problem1_sin_cos_problem2_linear_combination_l985_98583

/-- Problem 1: Prove that sin(α) * cos(α) = -2/5 given that the terminal side of angle α passes through (-1, 2) --/
theorem problem1_sin_cos (α : ℝ) (x y : ℝ) (h1 : x = -1) (h2 : y = 2) (h3 : Real.sqrt (x^2 + y^2) ≠ 0) :
  Real.sin α * Real.cos α = -2 / 5 :=
by
  sorry

/-- Problem 2: Prove that 10sin(α) + 3cos(α) = 0 given that the terminal side of angle α lies on the line y = -3x --/
theorem problem2_linear_combination (α : ℝ) (x y : ℝ) (h1 : y = -3 * x) (h2 : (x = -1 ∧ y = 3) ∨ (x = 1 ∧ y = -3)) (h3 : Real.sqrt (x^2 + y^2) ≠ 0) :
  10 * Real.sin α + 3 / Real.cos α = 0 :=
by
  sorry

end problem1_sin_cos_problem2_linear_combination_l985_98583


namespace mike_earnings_first_job_l985_98581

def total_earnings := 160
def hours_second_job := 12
def hourly_wage_second_job := 9
def earnings_second_job := hours_second_job * hourly_wage_second_job
def earnings_first_job := total_earnings - earnings_second_job

theorem mike_earnings_first_job : 
  earnings_first_job = 160 - (12 * 9) := by
  -- omitted proof
  sorry

end mike_earnings_first_job_l985_98581


namespace average_gas_mileage_round_trip_l985_98533

theorem average_gas_mileage_round_trip :
  let distance_to_city := 150
  let mpg_sedan := 25
  let mpg_rental := 15
  let total_distance := 2 * distance_to_city
  let gas_used_outbound := distance_to_city / mpg_sedan
  let gas_used_return := distance_to_city / mpg_rental
  let total_gas_used := gas_used_outbound + gas_used_return
  let avg_gas_mileage := total_distance / total_gas_used
  avg_gas_mileage = 18.75 := by
{
  sorry
}

end average_gas_mileage_round_trip_l985_98533


namespace original_average_is_24_l985_98520

theorem original_average_is_24
  (A : ℝ)
  (h1 : ∀ n : ℕ, n = 7 → 35 * A = 7 * 120) :
  A = 24 :=
by
  sorry

end original_average_is_24_l985_98520


namespace trigonometric_identity_l985_98522

theorem trigonometric_identity
  (α : Real)
  (h : 3 * Real.sin α + Real.cos α = 0) :
  1 / (Real.cos α ^ 2 + 2 * Real.sin α * Real.cos α) = 10 / 3 :=
by
  sorry

end trigonometric_identity_l985_98522


namespace find_larger_number_l985_98523

theorem find_larger_number (hc_f : ℕ) (factor1 factor2 : ℕ)
(h_hcf : hc_f = 63)
(h_factor1 : factor1 = 11)
(h_factor2 : factor2 = 17)
(lcm := hc_f * factor1 * factor2)
(A := hc_f * factor1)
(B := hc_f * factor2) :
max A B = 1071 := by
  sorry

end find_larger_number_l985_98523


namespace tangent_slope_at_1_l985_98551

def f (x : ℝ) : ℝ := x^3 + x^2 + 1

theorem tangent_slope_at_1 : (deriv f 1) = 5 := by
  sorry

end tangent_slope_at_1_l985_98551


namespace range_a_l985_98592

variable (a : ℝ)

def p := (∀ x : ℝ, x^2 + x + a > 0)
def q := ∃ x y : ℝ, x^2 - 2 * a * x + 1 ≤ y

theorem range_a :
  ({a : ℝ | (p a ∧ ¬q a) ∨ (¬p a ∧ q a)} = {a : ℝ | a < -1} ∪ {a : ℝ | 1 / 4 < a ∧ a < 1}) := 
by
  sorry

end range_a_l985_98592


namespace find_a2_b2_l985_98584

theorem find_a2_b2 (a b : ℝ) (h : a^2 * b^2 + a^2 + b^2 + 16 = 10 * a * b) : a^2 + b^2 = 8 :=
by
  sorry

end find_a2_b2_l985_98584


namespace f_m_eq_five_l985_98535

def f (x : ℝ) (a : ℝ) : ℝ :=
  x^3 + a * x + 3

axiom f_neg_m : ∀ (m a : ℝ), f (-m) a = 1

theorem f_m_eq_five (m a : ℝ) (h : f (-m) a = 1) : f m a = 5 :=
  by sorry

end f_m_eq_five_l985_98535


namespace sum_when_max_power_less_500_l985_98561

theorem sum_when_max_power_less_500 :
  ∃ (a b : ℕ), a > 0 ∧ b > 1 ∧ a^b < 500 ∧
  (∀ (a' b' : ℕ), a' > 0 → b' > 1 → a'^b' < 500 → a^b ≥ a'^b') ∧ (a + b = 24) :=
  sorry

end sum_when_max_power_less_500_l985_98561


namespace mosquito_feedings_to_death_l985_98501

theorem mosquito_feedings_to_death 
  (drops_per_feeding : ℕ := 20) 
  (drops_per_liter : ℕ := 5000) 
  (lethal_blood_loss_liters : ℝ := 3) 
  (drops_per_feeding_liters : ℝ := drops_per_feeding / drops_per_liter) 
  (lethal_feedings : ℝ := lethal_blood_loss_liters / drops_per_feeding_liters) :
  lethal_feedings = 750 := 
by
  sorry

end mosquito_feedings_to_death_l985_98501


namespace original_price_of_computer_l985_98552

theorem original_price_of_computer :
  ∃ (P : ℝ), (1.30 * P = 377) ∧ (2 * P = 580) ∧ (P = 290) :=
by
  existsi (290 : ℝ)
  sorry

end original_price_of_computer_l985_98552


namespace circle_integer_points_l985_98587

theorem circle_integer_points (m n : ℤ) (h : ∃ m n : ℤ, m^2 + n^2 = r ∧ 
  ∃ p q : ℤ, m^2 + n^2 = p ∧ ∃ s t : ℤ, m^2 + n^2 = q ∧ ∃ u v : ℤ, m^2 + n^2 = s ∧ 
  ∃ j k : ℤ, m^2 + n^2 = t ∧ ∃ l w : ℤ, m^2 + n^2 = u ∧ ∃ x y : ℤ, m^2 + n^2 = v ∧ 
  ∃ i b : ℤ, m^2 + n^2 = w ∧ ∃ c d : ℤ, m^2 + n^2 = b ) :
  ∃ r, r = 25 := by
    sorry

end circle_integer_points_l985_98587


namespace total_ways_to_choose_gifts_l985_98542

/-- The 6 pairs of zodiac signs -/
def zodiac_pairs : Set (Set String) :=
  {{"Rat", "Ox"}, {"Tiger", "Rabbit"}, {"Dragon", "Snake"}, {"Horse", "Sheep"}, {"Monkey", "Rooster"}, {"Dog", "Pig"}}

/-- The preferences of Students A, B, and C -/
def A_likes : Set String := {"Ox", "Horse"}
def B_likes : Set String := {"Ox", "Dog", "Sheep"}
def C_likes : Set String := {"Rat", "Ox", "Tiger", "Rabbit", "Dragon", "Snake", "Horse", "Sheep", "Monkey", "Rooster", "Dog", "Pig"}

theorem total_ways_to_choose_gifts : 
  True := 
by
  -- We prove that the number of ways is 16
  sorry

end total_ways_to_choose_gifts_l985_98542


namespace cylinder_volume_l985_98525

theorem cylinder_volume (r h : ℝ) (hrh : 2 * Real.pi * r * h = 100 * Real.pi) (h_diag : 4 * r^2 + h^2 = 200) :
  Real.pi * r^2 * h = 250 * Real.pi :=
sorry

end cylinder_volume_l985_98525


namespace solution_set_of_inequality_l985_98593

theorem solution_set_of_inequality :
  { x : ℝ | 3 ≤ |2 * x - 5| ∧ |2 * x - 5| < 9 } = { x : ℝ | (-2 < x ∧ x ≤ 1) ∨ (4 ≤ x ∧ x < 7) } :=
by 
  -- Conditions and steps omitted for the sake of the statement.
  sorry

end solution_set_of_inequality_l985_98593


namespace max_positive_integers_l985_98569

theorem max_positive_integers (f : Fin 2018 → ℤ) (h : ∀ i : Fin 2018, f i > f (i - 1) + f (i - 2)) : 
  ∃ n: ℕ, n = 2016 ∧ (∀ i : ℕ, i < 2018 → f i > 0) ∧ (∀ i : ℕ, i < 2 → f i < 0) := 
sorry

end max_positive_integers_l985_98569


namespace competition_inequality_l985_98524

variable (a b k : ℕ)

-- Conditions
variable (h1 : b % 2 = 1) 
variable (h2 : b ≥ 3)
variable (h3 : ∀ (J1 J2 : ℕ), J1 ≠ J2 → ∃ num_students : ℕ, num_students ≤ a ∧ num_students ≤ k)

theorem competition_inequality (h1: b % 2 = 1) (h2: b ≥ 3) (h3: ∀ (J1 J2 : ℕ), J1 ≠ J2 → ∃ num_students : ℕ, num_students ≤ a ∧ num_students ≤ k) :
  (k: ℝ) / (a: ℝ) ≥ (b-1: ℝ) / (2*b: ℝ) := sorry

end competition_inequality_l985_98524


namespace graph_n_plus_k_odd_l985_98513

-- Definitions and assumptions
variable {V : Type} [Fintype V] [DecidableEq V] (G : SimpleGraph V)
variable (n k : ℕ)
variable (hG : Fintype.card V = n)
variable (hCond : ∀ (S : Finset V), S.card = k → (G.commonNeighborsFinset S).card % 2 = 1)

-- Goal
theorem graph_n_plus_k_odd :
  (n + k) % 2 = 1 :=
sorry

end graph_n_plus_k_odd_l985_98513


namespace tall_mirror_passes_l985_98558

theorem tall_mirror_passes (T : ℕ)
    (s_tall_ref : ℕ)
    (s_wide_ref : ℕ)
    (e_tall_ref : ℕ)
    (e_wide_ref : ℕ)
    (wide_passes : ℕ)
    (total_reflections : ℕ)
    (H1 : s_tall_ref = 10)
    (H2 : s_wide_ref = 5)
    (H3 : e_tall_ref = 6)
    (H4 : e_wide_ref = 3)
    (H5 : wide_passes = 5)
    (H6 : s_tall_ref * T + s_wide_ref * wide_passes + e_tall_ref * T + e_wide_ref * wide_passes = 88) : 
    T = 3 := 
by sorry

end tall_mirror_passes_l985_98558


namespace hyperbola_equation_of_focus_and_asymptote_l985_98515

theorem hyperbola_equation_of_focus_and_asymptote :
  ∃ a b : ℝ, a > 0 ∧ b > 0 ∧ (2 * a) ^ 2 + (2 * b) ^ 2 = 25 ∧ b / a = 2 ∧ 
  (∀ x y : ℝ, (y = 2 * x + 10) → (x = -5) ∧ (y = 0)) ∧ 
  (∀ x y : ℝ, (x ^ 2 / 5 - y ^ 2 / 20 = 1)) :=
by
  sorry

end hyperbola_equation_of_focus_and_asymptote_l985_98515


namespace denis_dartboard_score_l985_98502

theorem denis_dartboard_score :
  ∀ P1 P2 P3 P4 : ℕ,
  P1 = 30 → 
  P2 = 38 → 
  P3 = 41 → 
  P1 + P2 + P3 + P4 = 4 * ((P1 + P2 + P3 + P4) / 4) → 
  P4 = 34 :=
by
  intro P1 P2 P3 P4 hP1 hP2 hP3 hTotal
  have hSum := hP1.symm ▸ hP2.symm ▸ hP3.symm ▸ hTotal
  sorry

end denis_dartboard_score_l985_98502


namespace tan_of_45_deg_l985_98510

theorem tan_of_45_deg : Real.tan (Real.pi / 4) = 1 := by
  sorry

end tan_of_45_deg_l985_98510


namespace Alyssa_spent_in_total_l985_98580

def amount_paid_for_grapes : ℝ := 12.08
def refund_for_cherries : ℝ := 9.85
def total_spent : ℝ := amount_paid_for_grapes - refund_for_cherries

theorem Alyssa_spent_in_total : total_spent = 2.23 := by
  sorry

end Alyssa_spent_in_total_l985_98580


namespace tetrahedron_labeling_impossible_l985_98553

/-- Suppose each vertex of a tetrahedron needs to be labeled with an integer from 1 to 4, each integer being used exactly once.
We need to prove that there are no such arrangements in which the sum of the numbers on the vertices of each face is the same for all four faces.
Arrangements that can be rotated into each other are considered identical. -/
theorem tetrahedron_labeling_impossible :
  ∀ (label : Fin 4 → Fin 5) (h_unique : ∀ v1 v2 : Fin 4, v1 ≠ v2 → label v1 ≠ label v2),
  ∃ (sum_faces : ℕ), sum_faces = 7 ∧ sum_faces % 3 = 1 → False :=
by
  sorry

end tetrahedron_labeling_impossible_l985_98553


namespace sqrt_plus_inv_sqrt_eq_l985_98541

noncomputable def sqrt_plus_inv_sqrt (x : ℝ) : ℝ :=
  Real.sqrt x + 1 / Real.sqrt x

theorem sqrt_plus_inv_sqrt_eq (x : ℝ) (h₁ : 0 < x) (h₂ : x + 1 / x = 50) :
  sqrt_plus_inv_sqrt x = 2 * Real.sqrt 13 := by
  sorry

end sqrt_plus_inv_sqrt_eq_l985_98541


namespace sets_are_equal_l985_98530

theorem sets_are_equal :
  let M := {x | ∃ k : ℤ, x = 2 * k + 1}
  let N := {x | ∃ k : ℤ, x = 4 * k + 1 ∨ x = 4 * k - 1}
  M = N :=
by
  sorry

end sets_are_equal_l985_98530


namespace total_birds_on_fence_l985_98540

theorem total_birds_on_fence (initial_birds additional_birds storks : ℕ) 
  (h1 : initial_birds = 6) 
  (h2 : additional_birds = 4) 
  (h3 : storks = 8) :
  initial_birds + additional_birds + storks = 18 :=
by
  sorry

end total_birds_on_fence_l985_98540


namespace max_probability_sum_15_l985_98534

-- Context and Definitions based on conditions
def S : Finset ℕ := {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}

-- The assertion to be proved:
theorem max_probability_sum_15 (n : ℕ) (h : n ∈ S) :
  n = 7 :=
by
  sorry

end max_probability_sum_15_l985_98534


namespace total_students_in_school_l985_98554

theorem total_students_in_school : 
  ∀ (number_of_deaf_students number_of_blind_students : ℕ), 
  (number_of_deaf_students = 180) → 
  (number_of_deaf_students = 3 * number_of_blind_students) → 
  (number_of_deaf_students + number_of_blind_students = 240) :=
by 
  sorry

end total_students_in_school_l985_98554


namespace range_of_a_l985_98556

open Real

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, x^2 + a * x + 1 ≥ 0) ↔ -2 ≤ a ∧ a ≤ 2 :=
by
  sorry

end range_of_a_l985_98556


namespace gcd_of_given_lengths_l985_98529

def gcd_of_lengths_is_eight : Prop :=
  let lengths := [48, 64, 80, 120]
  ∃ d, d = 8 ∧ (∀ n ∈ lengths, d ∣ n)

theorem gcd_of_given_lengths : gcd_of_lengths_is_eight := 
  sorry

end gcd_of_given_lengths_l985_98529


namespace simplify_expr_l985_98555

variable {x y : ℝ}

theorem simplify_expr (hx : x ≠ 0) (hy : y ≠ 0) :
  ((x^3 + 1) / x) * ((y^3 + 1) / y) - ((x^3 - 1) / y) * ((y^3 - 1) / x) = 2 * x^2 + 2 * y^2 :=
by sorry

end simplify_expr_l985_98555


namespace rhombus_area_l985_98586

-- Definitions
def side_length := 25 -- cm
def diagonal1 := 30 -- cm

-- Statement to prove
theorem rhombus_area (s : ℝ) (d1 : ℝ) (d2 : ℝ) 
  (h_s : s = 25) 
  (h_d1 : d1 = 30)
  (h_side : s^2 = (d1/2)^2 + (d2/2)^2) :
  (d1 * d2) / 2 = 600 :=
by sorry

end rhombus_area_l985_98586


namespace derivative_of_y_l985_98547

noncomputable def y (x : ℝ) : ℝ :=
  1/2 * Real.tanh x + 1/(4 * Real.sqrt 2) * Real.log ((1 + Real.sqrt 2 * Real.tanh x) / (1 - Real.sqrt 2 * Real.tanh x))

theorem derivative_of_y (x : ℝ) : 
  (deriv y x) = 1/(Real.cosh x ^ 2 * (1 - Real.sinh x ^ 2)) := 
by
  sorry

end derivative_of_y_l985_98547


namespace sum_of_sequence_l985_98591

theorem sum_of_sequence (n : ℕ) (h : n ≥ 2) 
  (a : ℕ → ℕ) (S : ℕ → ℕ)
  (h1 : a 1 = 2)
  (h2 : ∀ n : ℕ, n ≥ 2 → S n * S (n-1) + a n = 0) :
  S n = 2 / (2 * n - 1) := by
  sorry

end sum_of_sequence_l985_98591


namespace three_digit_sum_of_factorials_l985_98537

theorem three_digit_sum_of_factorials : 
  ∃ (n : ℕ), (100 ≤ n ∧ n < 1000) ∧ (n = 145) ∧ 
  (∃ (d1 d2 d3 : ℕ), n = d1 * 100 + d2 * 10 + d3 ∧ 
    1 ≤ d1 ∧ d1 < 10 ∧ 1 ≤ d2 ∧ d2 < 10 ∧ 1 ≤ d3 ∧ d3 < 10 ∧ 
    (d1 * d1.factorial + d2 * d2.factorial + d3 * d3.factorial = n)) :=
  by
  sorry

end three_digit_sum_of_factorials_l985_98537


namespace acute_triangle_l985_98563

theorem acute_triangle (a b c : ℝ) (h : a^π + b^π = c^π) : a^2 + b^2 > c^2 := sorry

end acute_triangle_l985_98563


namespace number_of_unlocked_cells_l985_98519

-- Establish the conditions from the problem description.
def total_cells : ℕ := 2004

-- Helper function to determine if a number is a perfect square.
def is_perfect_square (n : ℕ) : Prop :=
  ∃ (k : ℕ), k * k = n

-- Counting the number of perfect squares in the range from 1 to total_cells.
def perfect_squares_up_to (n : ℕ) : ℕ :=
  (Nat.sqrt n)

-- The theorem that needs to be proved.
theorem number_of_unlocked_cells : perfect_squares_up_to total_cells = 44 :=
by
  sorry

end number_of_unlocked_cells_l985_98519


namespace probability_exactly_one_second_class_product_l985_98543

open Nat

/-- Proof problem -/
theorem probability_exactly_one_second_class_product :
  let n := 100 -- total products
  let k := 4   -- number of selected products
  let first_class := 90 -- first-class products
  let second_class := 10 -- second-class products
  let C (n k : ℕ) := Nat.choose n k
  (C second_class 1 * C first_class 3 : ℚ) / C n k = 
  (Nat.choose second_class 1 * Nat.choose first_class 3 : ℚ) / Nat.choose n k :=
by
  -- Mathematically equivalent proof
  sorry

end probability_exactly_one_second_class_product_l985_98543


namespace largest_integer_base8_square_l985_98507

theorem largest_integer_base8_square :
  ∃ (N : ℕ), (N^2 >= 8^3) ∧ (N^2 < 8^4) ∧ (N = 63 ∧ N % 8 = 7) := sorry

end largest_integer_base8_square_l985_98507


namespace total_wheels_in_parking_lot_l985_98536

def num_cars : ℕ := 14
def num_bikes : ℕ := 10
def wheels_per_car : ℕ := 4
def wheels_per_bike : ℕ := 2

theorem total_wheels_in_parking_lot :
  (num_cars * wheels_per_car) + (num_bikes * wheels_per_bike) = 76 :=
by
  sorry

end total_wheels_in_parking_lot_l985_98536


namespace simplifyExpression_l985_98505

theorem simplifyExpression (a b c d : Int) (ha : a = -2) (hb : b = -6) (hc : c = -3) (hd : d = 2) :
  (a + b - c - d = -2 - 6 + 3 - 2) :=
by {
  sorry
}

end simplifyExpression_l985_98505


namespace phillip_remaining_money_l985_98575

def initial_money : ℝ := 95
def cost_oranges : ℝ := 14
def cost_apples : ℝ := 25
def cost_candy : ℝ := 6
def cost_eggs : ℝ := 12
def cost_milk : ℝ := 8
def discount_apples_rate : ℝ := 0.15
def discount_milk_rate : ℝ := 0.10

def discounted_cost_apples : ℝ := cost_apples * (1 - discount_apples_rate)
def discounted_cost_milk : ℝ := cost_milk * (1 - discount_milk_rate)

def total_spent : ℝ := cost_oranges + discounted_cost_apples + cost_candy + cost_eggs + discounted_cost_milk

def remaining_money : ℝ := initial_money - total_spent

theorem phillip_remaining_money : remaining_money = 34.55 := by
  -- Proof here
  sorry

end phillip_remaining_money_l985_98575


namespace ratio_of_second_to_third_l985_98526

theorem ratio_of_second_to_third (A B C : ℕ) (h1 : A + B + C = 98) (h2 : A * 3 = B * 2) (h3 : B = 30) :
  B * 8 = C * 5 :=
by
  sorry

end ratio_of_second_to_third_l985_98526


namespace savings_with_discount_l985_98568

theorem savings_with_discount :
  let original_price := 3.00
  let discount_rate := 0.30
  let discounted_price := original_price * (1 - discount_rate)
  let number_of_notebooks := 7
  let total_cost_without_discount := number_of_notebooks * original_price
  let total_cost_with_discount := number_of_notebooks * discounted_price
  total_cost_without_discount - total_cost_with_discount = 6.30 :=
by
  sorry

end savings_with_discount_l985_98568


namespace initial_amount_of_money_l985_98571

variable (X : ℕ) -- Initial amount of money Lily had in her account

-- Conditions
def spent_on_shirt : ℕ := 7
def spent_in_second_shop : ℕ := 3 * spent_on_shirt
def remaining_after_purchases : ℕ := 27

-- Proof problem: prove that the initial amount of money X is 55 given the conditions
theorem initial_amount_of_money (h : X - spent_on_shirt - spent_in_second_shop = remaining_after_purchases) : X = 55 :=
by
  -- Placeholder to indicate that steps will be worked out in Lean
  sorry

end initial_amount_of_money_l985_98571


namespace mode_I_swaps_mode_II_swaps_l985_98560

-- Define the original and target strings
def original_sign := "MEGYEI TAKARÉKPÉNZTÁR R. T."
def target_sign := "TATÁR GYERMEK A PÉNZT KÉRI."

-- Define a function for adjacent swaps needed to convert original_sign to target_sign
def adjacent_swaps (orig : String) (target : String) : ℕ := sorry

-- Define a function for any distant swaps needed to convert original_sign to target_sign
def distant_swaps (orig : String) (target : String) : ℕ := sorry

-- The theorems we want to prove
theorem mode_I_swaps : adjacent_swaps original_sign target_sign = 85 := sorry

theorem mode_II_swaps : distant_swaps original_sign target_sign = 11 := sorry

end mode_I_swaps_mode_II_swaps_l985_98560


namespace main_inequality_l985_98577

theorem main_inequality (m : ℝ) : (∀ x : ℝ, |2 * x - m| ≤ |3 * x + 6|) ↔ m = -4 := by
  sorry

end main_inequality_l985_98577


namespace measles_cases_1993_l985_98508

theorem measles_cases_1993 :
  ∀ (cases_1970 cases_1986 cases_2000 : ℕ)
    (rate1 rate2 : ℕ),
  cases_1970 = 600000 →
  cases_1986 = 30000 →
  cases_2000 = 600 →
  rate1 = 35625 →
  rate2 = 2100 →
  cases_1986 - 7 * rate2 = 15300 :=
by {
  sorry
}

end measles_cases_1993_l985_98508


namespace oreos_total_l985_98532

variable (Jordan : ℕ)
variable (James : ℕ := 4 * Jordan + 7)

theorem oreos_total (h : James = 43) : 43 + Jordan = 52 :=
sorry

end oreos_total_l985_98532


namespace decagon_interior_angle_measure_l985_98500

-- Define the type for a regular polygon
structure RegularPolygon (n : Nat) :=
  (interior_angle_sum : Nat := (n - 2) * 180)
  (side_count : Nat := n)
  (regularity : Prop := True)  -- All angles are equal

-- Define the degree measure of an interior angle of a regular polygon
def interiorAngle (p : RegularPolygon 10) : Nat :=
  (p.interior_angle_sum) / p.side_count

-- The theorem to be proved
theorem decagon_interior_angle_measure : 
  ∀ (p : RegularPolygon 10), interiorAngle p = 144 := by
  -- The proof will be here, but for now, we use sorry
  sorry

end decagon_interior_angle_measure_l985_98500


namespace scientific_notation_of_29_47_thousand_l985_98594

theorem scientific_notation_of_29_47_thousand :
  (29.47 * 1000 = 2.947 * 10^4) :=
sorry

end scientific_notation_of_29_47_thousand_l985_98594


namespace route_time_difference_l985_98514

-- Define the conditions of the problem
def first_route_time : ℕ := 
  let time_uphill := 6
  let time_path := 2 * time_uphill
  let time_finish := (time_uphill + time_path) / 3
  time_uphill + time_path + time_finish

def second_route_time : ℕ := 
  let time_flat_path := 14
  let time_finish := 2 * time_flat_path
  time_flat_path + time_finish

-- Prove the question
theorem route_time_difference : second_route_time - first_route_time = 18 :=
by
  sorry

end route_time_difference_l985_98514


namespace problem_statement_l985_98521

open Set

-- Definitions based on the problem's conditions
def U : Set ℕ := { x | 0 < x ∧ x ≤ 8 }
def S : Set ℕ := {1, 2, 4, 5}
def T : Set ℕ := {3, 5, 7}
def complement_U_T : Set ℕ := U \ T

-- The Lean 4 statement to prove
theorem problem_statement : S ∩ complement_U_T = {1, 2, 4} :=
by sorry

end problem_statement_l985_98521


namespace garden_fencing_l985_98549

/-- A rectangular garden has a length of 50 yards and the width is half the length.
    Prove that the total amount of fencing needed to enclose the garden is 150 yards. -/
theorem garden_fencing : 
  ∀ (length width : ℝ), 
  length = 50 ∧ width = length / 2 → 
  2 * (length + width) = 150 :=
by
  intros length width
  rintro ⟨h1, h2⟩
  sorry

end garden_fencing_l985_98549


namespace find_x_value_l985_98503

theorem find_x_value {C S x : ℝ}
  (h1 : C = 100 * (1 + x / 100))
  (h2 : S - C = 10 / 9)
  (h3 : S = 100 * (1 + x / 100)):
  x = 10 :=
by
  sorry

end find_x_value_l985_98503


namespace chord_intersects_inner_circle_l985_98506

noncomputable def probability_chord_intersects_inner_circle
  (r1 r2 : ℝ) (h1 : r1 = 2) (h2 : r2 = 5) : ℝ :=
0.098

theorem chord_intersects_inner_circle :
  probability_chord_intersects_inner_circle 2 5 rfl rfl = 0.098 :=
sorry

end chord_intersects_inner_circle_l985_98506


namespace triangle_area_example_l985_98511

noncomputable def area_triangle (BC AB : ℝ) (B : ℝ) : ℝ :=
  (1 / 2) * BC * AB * Real.sin B

theorem triangle_area_example
  (BC AB : ℝ) (B : ℝ)
  (hBC : BC = 2)
  (hAB : AB = 3)
  (hB : B = Real.pi / 3) :
  area_triangle BC AB B = (3 * Real.sqrt 3) / 2 :=
by
  sorry

end triangle_area_example_l985_98511


namespace evaluate_expression_l985_98527

theorem evaluate_expression : -30 + 5 * (9 / (3 + 3)) = -22.5 := sorry

end evaluate_expression_l985_98527


namespace sqrt_product_simplification_l985_98528

theorem sqrt_product_simplification (q : ℝ) : 
  Real.sqrt (15 * q) * Real.sqrt (10 * q^3) * Real.sqrt (14 * q^5) = 10 * q^4 * Real.sqrt (21 * q) := 
by 
  sorry

end sqrt_product_simplification_l985_98528
