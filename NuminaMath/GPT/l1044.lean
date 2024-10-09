import Mathlib

namespace problem1_problem2_l1044_104445

-- Proof Problem 1: Prove that when \( k = 5 \), \( x^2 - 5x + 4 > 0 \) holds for \( \{x \mid x < 1 \text{ or } x > 4\} \).
theorem problem1 (x : ℝ) (h : x^2 - 5 * x + 4 > 0) : x < 1 ∨ x > 4 :=
sorry

-- Proof Problem 2: Prove that the range of values for \( k \) such that \( x^2 - kx + 4 > 0 \) holds for all real numbers \( x \) is \( (-4, 4) \).
theorem problem2 (k : ℝ) : (∀ x : ℝ, x^2 - k * x + 4 > 0) ↔ -4 < k ∧ k < 4 :=
sorry

end problem1_problem2_l1044_104445


namespace n_cube_plus_5n_divisible_by_6_l1044_104439

theorem n_cube_plus_5n_divisible_by_6 (n : ℤ) : 6 ∣ (n^3 + 5 * n) := 
sorry

end n_cube_plus_5n_divisible_by_6_l1044_104439


namespace area_of_triangle_arithmetic_sides_l1044_104475

theorem area_of_triangle_arithmetic_sides 
  (a : ℝ) (h : a > 0) (h_sin : Real.sin (2 * Real.pi / 3) = Real.sqrt 3 / 2) :
  let s₁ := a - 2
  let s₂ := a
  let s₃ := a + 2
  ∃ (a b c : ℝ), 
    a = s₁ ∧ b = s₂ ∧ c = s₃ ∧ 
    Real.sin (2 * Real.pi / 3) = Real.sqrt 3 / 2 → 
    (1/2 * s₁ * s₂ * Real.sin (2 * Real.pi / 3) = 15 * Real.sqrt 3 / 4) :=
by
  sorry

end area_of_triangle_arithmetic_sides_l1044_104475


namespace distance_between_trees_l1044_104422

theorem distance_between_trees (yard_length : ℕ) (num_trees : ℕ)
  (h_yard : yard_length = 400) (h_trees : num_trees = 26) : 
  (yard_length / (num_trees - 1)) = 16 :=
by
  sorry

end distance_between_trees_l1044_104422


namespace at_least_one_inequality_holds_l1044_104476

theorem at_least_one_inequality_holds (x y : ℝ) (h1 : 0 < x) (h2 : 0 < y) (h3 : x + y > 2) : 
  (1 + x) / y < 2 ∨ (1 + y) / x < 2 :=
by
  sorry

end at_least_one_inequality_holds_l1044_104476


namespace jordan_machine_solution_l1044_104455

theorem jordan_machine_solution (x : ℝ) (h : 2 * x + 3 - 5 = 27) : x = 14.5 :=
sorry

end jordan_machine_solution_l1044_104455


namespace sum_of_solutions_eq_zero_l1044_104413

theorem sum_of_solutions_eq_zero (x : ℝ) :
  (∃ x_1 x_2 : ℝ, (|x_1 - 20| + |x_2 + 20| = 2020) ∧ (x_1 + x_2 = 0)) :=
sorry

end sum_of_solutions_eq_zero_l1044_104413


namespace percent_of_a_is_20_l1044_104416

variable {a b c : ℝ}

theorem percent_of_a_is_20 (h1 : c = (x / 100) * a)
                          (h2 : c = 0.1 * b)
                          (h3 : b = 2 * a) :
  c = 0.2 * a := sorry

end percent_of_a_is_20_l1044_104416


namespace jack_received_emails_in_the_morning_l1044_104409

theorem jack_received_emails_in_the_morning
  (total_emails : ℕ)
  (afternoon_emails : ℕ)
  (morning_emails : ℕ) 
  (h1 : total_emails = 8)
  (h2 : afternoon_emails = 5)
  (h3 : total_emails = morning_emails + afternoon_emails) :
  morning_emails = 3 :=
  by
    -- proof omitted
    sorry

end jack_received_emails_in_the_morning_l1044_104409


namespace least_five_digit_perfect_square_and_cube_l1044_104474

theorem least_five_digit_perfect_square_and_cube : ∃ n : ℕ, (10000 ≤ n ∧ n < 100000) ∧ (∃ k1 : ℕ, n = k1 ^ 2) ∧ (∃ k2 : ℕ, n = k2 ^ 3) ∧ n = 15625 :=
by
  sorry

end least_five_digit_perfect_square_and_cube_l1044_104474


namespace cost_effectiveness_l1044_104449

-- Define the variables and conditions
def num_employees : ℕ := 30
def ticket_price : ℝ := 80
def group_discount_rate : ℝ := 0.8
def women_discount_rate : ℝ := 0.5

-- Define the costs for each scenario
def cost_with_group_discount : ℝ := num_employees * ticket_price * group_discount_rate

def cost_with_women_discount (x : ℕ) : ℝ :=
  ticket_price * women_discount_rate * x + ticket_price * (num_employees - x)

-- Formalize the equivalence of cost and comparison logic
theorem cost_effectiveness (x : ℕ) (h : 0 ≤ x ∧ x ≤ num_employees) :
  if x < 12 then cost_with_women_discount x > cost_with_group_discount
  else if x = 12 then cost_with_women_discount x = cost_with_group_discount
  else cost_with_women_discount x < cost_with_group_discount :=
by sorry

end cost_effectiveness_l1044_104449


namespace rectangular_solid_length_l1044_104447

theorem rectangular_solid_length (w h : ℕ) (surface_area : ℕ) (l : ℕ) 
  (hw : w = 4) (hh : h = 1) (hsa : surface_area = 58) 
  (h_surface_area_formula : surface_area = 2 * l * w + 2 * l * h + 2 * w * h) : 
  l = 5 :=
by
  rw [hw, hh, hsa] at h_surface_area_formula
  sorry

end rectangular_solid_length_l1044_104447


namespace total_soda_consumption_l1044_104444

variables (c_soda b_soda c_consumed b_consumed b_remaining carol_final bob_final total_consumed : ℕ)

-- Define the conditions
def carol_soda_size : ℕ := 20
def bob_soda_25_percent_more : ℕ := carol_soda_size + carol_soda_size * 25 / 100
def carol_consumed : ℕ := carol_soda_size * 80 / 100
def bob_consumed : ℕ := bob_soda_25_percent_more * 80 / 100
def carol_remaining : ℕ := carol_soda_size - carol_consumed
def bob_remaining : ℕ := bob_soda_25_percent_more - bob_consumed
def bob_gives_carol : ℕ := bob_remaining / 2 + 3
def carol_final_consumption : ℕ := carol_consumed + bob_gives_carol
def bob_final_consumption : ℕ := bob_consumed - bob_gives_carol
def total_soda_consumed : ℕ := carol_final_consumption + bob_final_consumption

-- The theorem to prove the total amount of soda consumed by Carol and Bob together is 36 ounces
theorem total_soda_consumption : total_soda_consumed = 36 := by {
  sorry
}

end total_soda_consumption_l1044_104444


namespace sum_of_consecutive_multiples_of_4_l1044_104477

theorem sum_of_consecutive_multiples_of_4 (n : ℝ) (h : 4 * n + (4 * n + 8) = 140) :
  4 * n + (4 * n + 4) + (4 * n + 8) = 210 :=
sorry

end sum_of_consecutive_multiples_of_4_l1044_104477


namespace part1_part2_part3a_part3b_l1044_104442

open Real

variable (θ : ℝ) (m : ℝ)

-- Conditions
axiom theta_domain : 0 < θ ∧ θ < 2 * π
axiom quadratic_eq : ∀ x : ℝ, 2 * x^2 - (sqrt 3 + 1) * x + m = 0
axiom roots_eq_theta : ∀ x : ℝ, (x = sin θ ∨ x = cos θ)

-- Proof statements
theorem part1 : 1 - cos θ ≠ 0 → 1 - tan θ ≠ 0 → 
  (sin θ / (1 - cos θ) + cos θ / (1 - tan θ)) = (3 + 5 * sqrt 3) / 4 := sorry

theorem part2 : sin θ * cos θ = m / 2 → m = sqrt 3 / 4 := sorry

theorem part3a : sin θ = sqrt 3 / 2 ∧ cos θ = 1 / 2 → θ = π / 3 := sorry

theorem part3b : sin θ = 1 / 2 ∧ cos θ = sqrt 3 / 2 → θ = π / 6 := sorry

end part1_part2_part3a_part3b_l1044_104442


namespace mean_of_remaining_three_numbers_l1044_104433

theorem mean_of_remaining_three_numbers 
    (a b c d : ℝ)
    (h₁ : (a + b + c + d) / 4 = 92)
    (h₂ : d = 120)
    (h₃ : b = 60) : 
    (a + b + c) / 3 = 82.6666666666 := 
by 
    -- This state suggests adding the constraints added so far for the proof:
    sorry

end mean_of_remaining_three_numbers_l1044_104433


namespace smallest_whole_number_gt_total_sum_l1044_104487

-- Declarations of the fractions involved
def term1 : ℚ := 3 + 1/3
def term2 : ℚ := 4 + 1/6
def term3 : ℚ := 5 + 1/12
def term4 : ℚ := 6 + 1/8

-- Definition of the entire sum
def total_sum : ℚ := term1 + term2 + term3 + term4

-- Statement of the theorem
theorem smallest_whole_number_gt_total_sum : 
  ∀ n : ℕ, (n > total_sum) → (∀ m : ℕ, (m >= 0) → (m > total_sum) → (n ≤ m)) → n = 19 := by
  sorry -- the proof is omitted

end smallest_whole_number_gt_total_sum_l1044_104487


namespace min_value_expression_l1044_104432

open Real

theorem min_value_expression (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) 
  (hxyz : x^2 + y^2 + z^2 = 1) : 
  (∃ (c : ℝ), c = 3 * sqrt 3 / 2 ∧ c ≤ (x / (1 - x^2) + y / (1 - y^2) + z / (1 - z^2))) :=
by
  sorry

end min_value_expression_l1044_104432


namespace composite_expression_l1044_104434

theorem composite_expression (n : ℕ) : ∃ a b : ℕ, a > 1 ∧ b > 1 ∧ (a * b = 6 * 2^(2^(4 * n)) + 1) :=
by
  sorry

end composite_expression_l1044_104434


namespace first_discount_percentage_l1044_104479

theorem first_discount_percentage (x : ℝ) :
  let initial_price := 26.67
  let final_price := 15.0
  let second_discount := 0.25
  (initial_price * (1 - x / 100) * (1 - second_discount) = final_price) → x = 25 :=
by
  intros
  sorry

end first_discount_percentage_l1044_104479


namespace minimize_d_and_distance_l1044_104470

-- Define point and geometric shapes
structure Point :=
  (x : ℝ)
  (y : ℝ)

def Parabola (P : Point) : Prop := P.x^2 = 4 * P.y
def Circle (P1 : Point) : Prop := (P1.x - 2)^2 + (P1.y + 1)^2 = 1

-- Define the point P and point P1
variable (P : Point)
variable (P1 : Point)

-- Condition: P is on the parabola
axiom on_parabola : Parabola P

-- Condition: P1 is on the circle
axiom on_circle : Circle P1

-- Theorem: coordinates of P when the function d + distance(P, P1) is minimized
theorem minimize_d_and_distance :
  P = { x := 2 * Real.sqrt 2 - 2, y := 3 - 2 * Real.sqrt 2 } :=
sorry

end minimize_d_and_distance_l1044_104470


namespace range_of_ab_l1044_104482

noncomputable def circle_equation (x y : ℝ) : Prop := (x^2 + y^2 + 2*x - 4*y + 1 = 0)

noncomputable def line_equation (a b x y : ℝ) : Prop := (2*a*x - b*y - 2 = 0)

def symmetric_with_respect_to (center_x center_y a b : ℝ) : Prop :=
  line_equation a b center_x center_y  -- check if the line passes through the center

theorem range_of_ab (a b : ℝ) (h_symm : symmetric_with_respect_to (-1) 2 a b) : 
  ∃ ab_max : ℝ, ab_max = 1/4 ∧ ∀ ab : ℝ, ab = (a * b) → ab ≤ ab_max :=
sorry

end range_of_ab_l1044_104482


namespace mary_flour_requirement_l1044_104478

theorem mary_flour_requirement (total_flour : ℕ) (added_flour : ℕ) (remaining_flour : ℕ) 
  (h1 : total_flour = 7) 
  (h2 : added_flour = 2) 
  (h3 : remaining_flour = total_flour - added_flour) : 
  remaining_flour = 5 :=
sorry

end mary_flour_requirement_l1044_104478


namespace remainder_of_16_pow_2048_mod_11_l1044_104408

theorem remainder_of_16_pow_2048_mod_11 : (16^2048) % 11 = 4 := by
  sorry

end remainder_of_16_pow_2048_mod_11_l1044_104408


namespace div_by_5_l1044_104428

theorem div_by_5 (n : ℕ) (hn : 0 < n) : (2^(4*n+1) + 3) % 5 = 0 := 
by sorry

end div_by_5_l1044_104428


namespace gcd_consecutive_digits_l1044_104461

theorem gcd_consecutive_digits (a : ℕ) (b : ℕ) (c : ℕ) (d : ℕ) 
  (h₁ : b = a + 1) (h₂ : c = a + 2) (h₃ : d = a + 3) :
  ∃ g, g = gcd (1000 * a + 100 * b + 10 * c + d - (1000 * d + 100 * c + 10 * b + a)) 3096 :=
by {
  sorry
}

end gcd_consecutive_digits_l1044_104461


namespace smallest_n_for_terminating_decimal_l1044_104466

-- Theorem follows the tuple of (question, conditions, correct answer)
theorem smallest_n_for_terminating_decimal (n : ℕ) (h : ∃ k : ℕ, n + 75 = 2^k ∨ n + 75 = 5^k ∨ n + 75 = (2^k * 5^k)) :
  n = 50 :=
by
  sorry -- Proof is omitted

end smallest_n_for_terminating_decimal_l1044_104466


namespace even_and_monotonically_decreasing_l1044_104499

noncomputable def f_B (x : ℝ) : ℝ := 1 / (x^2)

theorem even_and_monotonically_decreasing (x : ℝ) (h : x > 0) :
  (f_B x = f_B (-x)) ∧ (∀ {a b : ℝ}, a < b → a > 0 → b > 0 → f_B a > f_B b) :=
by
  sorry

end even_and_monotonically_decreasing_l1044_104499


namespace extremum_and_equal_values_l1044_104425

theorem extremum_and_equal_values {f : ℝ → ℝ} {a b x_0 x_1 : ℝ} 
    (hf : ∀ x, f x = (x - 1)^3 - a * x + b)
    (h'x0 : deriv f x_0 = 0)
    (hfx1_eq_fx0 : f x_1 = f x_0)
    (hx1_ne_x0 : x_1 ≠ x_0) :
  x_1 + 2 * x_0 = 3 := sorry

end extremum_and_equal_values_l1044_104425


namespace solve_quadratic_equation_solve_linear_equation_l1044_104435

-- Equation (1)
theorem solve_quadratic_equation :
  ∀ x : ℝ, x^2 - 8 * x + 1 = 0 → (x = 4 + Real.sqrt 15 ∨ x = 4 - Real.sqrt 15) :=
by
  sorry

-- Equation (2)
theorem solve_linear_equation :
  ∀ x : ℝ, 3 * x * (x - 1) = 2 - 2 * x → (x = 1 ∨ x = -2/3) :=
by
  sorry

end solve_quadratic_equation_solve_linear_equation_l1044_104435


namespace Lewis_more_items_than_Samantha_l1044_104498

def Tanya_items : ℕ := 4
def Samantha_items : ℕ := 4 * Tanya_items
def Lewis_items : ℕ := 20

theorem Lewis_more_items_than_Samantha : (Lewis_items - Samantha_items) = 4 := by
  sorry

end Lewis_more_items_than_Samantha_l1044_104498


namespace remainder_when_subtracted_l1044_104401

theorem remainder_when_subtracted (s t : ℕ) (hs : s % 6 = 2) (ht : t % 6 = 3) (h : s > t) : (s - t) % 6 = 5 :=
by
  sorry -- Proof not required

end remainder_when_subtracted_l1044_104401


namespace distance_to_place_l1044_104468

variables {r c1 c2 t D : ℝ}

theorem distance_to_place (h : t = (D / (r - c1)) + (D / (r + c2))) :
  D = t * (r^2 - c1 * c2) / (2 * r + c2 - c1) :=
by
  have h1 : D * (r + c2) / (r - c1) * (r - c1) = D * (r + c2) := by sorry
  have h2 : D * (r - c1) / (r + c2) * (r + c2) = D * (r - c1) := by sorry
  have h3 : D * (r + c2) = D * (r + c2) := by sorry
  have h4 : D * (r - c1) = D * (r - c1) := by sorry
  have h5 : t * (r - c1) * (r + c2) = D * (r + c2) + D * (r - c1) := by sorry
  have h6 : t * (r^2 - c1 * c2) = D * (2 * r + c2 - c1) := by sorry
  have h7 : D = t * (r^2 - c1 * c2) / (2 * r + c2 - c1) := by sorry
  exact h7

end distance_to_place_l1044_104468


namespace min_book_corner_cost_l1044_104402

theorem min_book_corner_cost :
  ∃ x : ℕ, 0 ≤ x ∧ x ≤ 30 ∧
  80 * x + 30 * (30 - x) ≤ 1900 ∧
  50 * x + 60 * (30 - x) ≤ 1620 ∧
  860 * x + 570 * (30 - x) = 22320 := sorry

end min_book_corner_cost_l1044_104402


namespace parabola_coeff_sum_l1044_104423

theorem parabola_coeff_sum (a b c : ℤ) (h₁ : a * (1:ℤ)^2 + b * 1 + c = 3)
                                      (h₂ : a * (-1)^2 + b * (-1) + c = 5)
                                      (vertex : ∀ x, a * (x + 1)^2 + 1 = a * x^2 + bx + c) :
a + b + c = 3 := 
sorry

end parabola_coeff_sum_l1044_104423


namespace ribbon_arrangement_count_correct_l1044_104450

-- Definitions for the problem conditions
inductive Color
| red
| yellow
| blue

-- The color sequence from top to bottom
def color_sequence : List Color := [Color.red, Color.blue, Color.yellow, Color.yellow]

-- A function to count the valid arrangements
def count_valid_arrangements (sequence : List Color) : Nat :=
  -- Since we need to prove, we're bypassing the actual implementation with sorry
  sorry

-- The proof statement
theorem ribbon_arrangement_count_correct : count_valid_arrangements color_sequence = 12 :=
by
  sorry

end ribbon_arrangement_count_correct_l1044_104450


namespace quadratic_coefficient_conversion_l1044_104484

theorem quadratic_coefficient_conversion :
  ∀ x : ℝ, (3 * x^2 - 1 = 5 * x) → (3 * x^2 - 5 * x - 1 = 0) :=
by
  intros x h
  rw [←sub_eq_zero, ←h]
  ring

end quadratic_coefficient_conversion_l1044_104484


namespace prank_combinations_l1044_104494

theorem prank_combinations :
  let monday_choices := 1
  let tuesday_choices := 2
  let wednesday_choices := 5
  let thursday_choices := 4
  let friday_choices := 1
  (monday_choices * tuesday_choices * wednesday_choices * thursday_choices * friday_choices) = 40 :=
by
  sorry

end prank_combinations_l1044_104494


namespace equilateral_triangle_sum_l1044_104446

theorem equilateral_triangle_sum (side_length : ℚ) (h_eq : side_length = 13 / 12) :
  3 * side_length = 13 / 4 :=
by
  -- Proof omitted
  sorry

end equilateral_triangle_sum_l1044_104446


namespace maximum_ratio_l1044_104438

-- Define the conditions
def is_two_digit_positive_integer (n : ℕ) : Prop :=
  10 ≤ n ∧ n ≤ 99

def mean_is_45 (x y : ℕ) : Prop :=
  (x + y) / 2 = 45

-- State the theorem
theorem maximum_ratio (x y : ℕ) (hx : is_two_digit_positive_integer x) (hy : is_two_digit_positive_integer y) (h_mean : mean_is_45 x y) : 
  ∃ (k: ℕ), (x / y = k) ∧ k = 8 :=
sorry

end maximum_ratio_l1044_104438


namespace cost_of_two_pans_is_20_l1044_104420

variable (cost_of_pan : ℕ)

-- Conditions
def pots_cost := 3 * 20
def total_cost := 100
def pans_eq_cost := total_cost - pots_cost
def cost_of_pan_per_pans := pans_eq_cost / 4

-- Proof statement
theorem cost_of_two_pans_is_20 
  (h1 : pots_cost = 60)
  (h2 : total_cost = 100)
  (h3 : pans_eq_cost = total_cost - pots_cost)
  (h4 : cost_of_pan_per_pans = pans_eq_cost / 4)
  : 2 * cost_of_pan_per_pans = 20 :=
by sorry

end cost_of_two_pans_is_20_l1044_104420


namespace tan_sum_eq_tan_prod_l1044_104431

noncomputable def tan (x : Real) : Real :=
  Real.sin x / Real.cos x

theorem tan_sum_eq_tan_prod (α β γ : Real) (h : tan α + tan β + tan γ = tan α * tan β * tan γ) :
  ∃ k : Int, α + β + γ = k * Real.pi :=
by
  sorry

end tan_sum_eq_tan_prod_l1044_104431


namespace always_two_real_roots_find_m_l1044_104440

-- Given quadratic equation: x^2 - 4mx + 3m^2 = 0
-- Definitions for the problem
def quadratic_eq (m x : ℝ) : Prop := x^2 - 4 * m * x + 3 * m^2 = 0

-- Q1: Prove that this equation always has two real roots.
theorem always_two_real_roots (m : ℝ) : ∃ x₁ x₂ : ℝ, quadratic_eq m x₁ ∧ quadratic_eq m x₂ :=
by
  sorry

-- Q2: If m > 0 and the difference between the two real roots is 2, find the value of m.
theorem find_m (m : ℝ) (h₁ : m > 0) (h₂ : ∃ x₁ x₂ : ℝ, quadratic_eq m x₁ ∧ quadratic_eq m x₂ ∧ |x₁ - x₂| = 2) : m = 1 :=
by
  sorry

end always_two_real_roots_find_m_l1044_104440


namespace sum_of_three_positive_integers_l1044_104492

theorem sum_of_three_positive_integers (n : ℕ) (h : n ≥ 3) :
  ∃ k : ℕ, k = (n - 1) * (n - 2) / 2 := 
sorry

end sum_of_three_positive_integers_l1044_104492


namespace sum_of_coefficients_l1044_104472

theorem sum_of_coefficients (a b c d : ℝ) (f : ℝ → ℝ)
    (h1 : ∀ x, f (x + 2) = 2*x^3 + 5*x^2 + 3*x + 6)
    (h2 : ∀ x, f x = a*x^3 + b*x^2 + c*x + d) :
  a + b + c + d = 6 :=
by sorry

end sum_of_coefficients_l1044_104472


namespace average_waiting_time_l1044_104463

-- Define the problem conditions
def light_period : ℕ := 3  -- Total cycle time in minutes
def green_time : ℕ := 1    -- Green light duration in minutes
def red_time : ℕ := 2      -- Red light duration in minutes

-- Define the probabilities of each light state
def P_G : ℚ := green_time / light_period
def P_R : ℚ := red_time / light_period

-- Define the expected waiting times given each state
def E_T_G : ℚ := 0
def E_T_R : ℚ := red_time / 2

-- Calculate the expected waiting time using the law of total expectation
def E_T : ℚ := E_T_G * P_G + E_T_R * P_R

-- Convert the expected waiting time to seconds
def E_T_seconds : ℚ := E_T * 60

-- Prove that the expected waiting time in seconds is 40 seconds
theorem average_waiting_time : E_T_seconds = 40 := by
  sorry

end average_waiting_time_l1044_104463


namespace pencil_distribution_l1044_104412

theorem pencil_distribution (x : ℕ) 
  (Alice Bob Charles : ℕ)
  (h1 : Alice = 2 * Bob)
  (h2 : Charles = Bob + 3)
  (h3 : Bob = x)
  (total_pencils : 53 = Alice + Bob + Charles) : 
  Bob = 13 ∧ Alice = 26 ∧ Charles = 16 :=
by
  sorry

end pencil_distribution_l1044_104412


namespace A_eq_B_l1044_104404

namespace SetsEquality

open Set

def A : Set ℝ := {x | ∃ a : ℝ, x = 5 - 4 * a + a^2}
def B : Set ℝ := {y | ∃ b : ℝ, y = 4 * b^2 + 4 * b + 2}

theorem A_eq_B : A = B := by
  sorry

end SetsEquality

end A_eq_B_l1044_104404


namespace length_of_other_parallel_side_l1044_104411

theorem length_of_other_parallel_side 
  (a : ℝ) (h : ℝ) (A : ℝ) (x : ℝ) 
  (h_a : a = 16) (h_h : h = 15) (h_A : A = 270) 
  (h_area_formula : A = 1 / 2 * (a + x) * h) : 
  x = 20 :=
sorry

end length_of_other_parallel_side_l1044_104411


namespace max_points_in_equilateral_property_set_l1044_104485

theorem max_points_in_equilateral_property_set (Γ : Finset (ℝ × ℝ)) :
  (∀ (A B : (ℝ × ℝ)), A ∈ Γ → B ∈ Γ → 
    ∃ C : (ℝ × ℝ), C ∈ Γ ∧ 
    dist A B = dist B C ∧ dist B C = dist C A ∧ dist C A = dist A B) → Γ.card ≤ 3 :=
by
  intro h
  sorry

end max_points_in_equilateral_property_set_l1044_104485


namespace stratified_sampling_correct_l1044_104490

variables (total_employees senior_employees mid_level_employees junior_employees sample_size : ℕ)
          (sampling_ratio : ℚ)
          (senior_sample mid_sample junior_sample : ℕ)

-- Conditions
def company_conditions := 
  total_employees = 450 ∧ 
  senior_employees = 45 ∧ 
  mid_level_employees = 135 ∧ 
  junior_employees = 270 ∧ 
  sample_size = 30 ∧ 
  sampling_ratio = 1 / 15

-- Proof goal
theorem stratified_sampling_correct : 
  company_conditions total_employees senior_employees mid_level_employees junior_employees sample_size sampling_ratio →
  senior_sample = senior_employees * sampling_ratio ∧ 
  mid_sample = mid_level_employees * sampling_ratio ∧ 
  junior_sample = junior_employees * sampling_ratio ∧
  senior_sample + mid_sample + junior_sample = sample_size :=
by sorry

end stratified_sampling_correct_l1044_104490


namespace inequality_solution_l1044_104452

theorem inequality_solution (a x : ℝ) (h : |a + 1| < 3) :
  (-4 < a ∧ a < -2 ∧ (x > -1 ∨ x < 1 + a)) ∨ 
  (a = -2 ∧ (x ∈ Set.univ \ {-1})) ∨ 
  (-2 < a ∧ a < 2 ∧ (x > 1 + a ∨ x < -1)) :=
by sorry

end inequality_solution_l1044_104452


namespace trailing_zeros_50_factorial_l1044_104429

def factorial_trailing_zeros (n : Nat) : Nat :=
  n / 5 + n / 25 -- Count the number of trailing zeros given the algorithm used in solution steps

theorem trailing_zeros_50_factorial : factorial_trailing_zeros 50 = 12 :=
by 
  -- Proof goes here
  sorry

end trailing_zeros_50_factorial_l1044_104429


namespace tim_tasks_per_day_l1044_104415

theorem tim_tasks_per_day (earnings_per_task : ℝ) (days_per_week : ℕ) (weekly_earnings : ℝ) :
  earnings_per_task = 1.2 ∧ days_per_week = 6 ∧ weekly_earnings = 720 → (weekly_earnings / days_per_week / earnings_per_task = 100) :=
by
  sorry

end tim_tasks_per_day_l1044_104415


namespace min_value_a_4b_l1044_104471

theorem min_value_a_4b (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : a * b = a + b) :
  a + 4 * b = 9 :=
sorry

end min_value_a_4b_l1044_104471


namespace exists_infinite_irregular_set_l1044_104483

def is_irregular (A : Set ℤ) :=
  ∀ ⦃x y : ℤ⦄, x ∈ A → y ∈ A → x ≠ y → ∀ ⦃k : ℤ⦄, x + k * (y - x) ≠ x ∧ x + k * (y - x) ≠ y

theorem exists_infinite_irregular_set : ∃ A : Set ℤ, Set.Infinite A ∧ is_irregular A :=
sorry

end exists_infinite_irregular_set_l1044_104483


namespace find_value_of_a2_plus_b2_plus_c2_l1044_104486

variables (a b c : ℝ)

-- Define the conditions
def conditions := (a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0) ∧ (a + b + c = 0) ∧ (a^3 + b^3 + c^3 = a^5 + b^5 + c^5)

-- State the theorem we need to prove
theorem find_value_of_a2_plus_b2_plus_c2 (h : conditions a b c) : a^2 + b^2 + c^2 = 6 / 5 :=
  sorry

end find_value_of_a2_plus_b2_plus_c2_l1044_104486


namespace Alice_fills_needed_l1044_104469

def cups_needed : ℚ := 15/4
def cup_capacity : ℚ := 1/3
def fills_needed : ℚ := 12

theorem Alice_fills_needed : (cups_needed / cup_capacity).ceil = fills_needed := by
  -- Proof is omitted with sorry
  sorry

end Alice_fills_needed_l1044_104469


namespace avg_megabyte_usage_per_hour_l1044_104448

theorem avg_megabyte_usage_per_hour (megabytes : ℕ) (days : ℕ) (hours : ℕ) (avg_mbps : ℕ)
  (h1 : megabytes = 27000)
  (h2 : days = 15)
  (h3 : hours = days * 24)
  (h4 : avg_mbps = megabytes / hours) : 
  avg_mbps = 75 := by
  sorry

end avg_megabyte_usage_per_hour_l1044_104448


namespace sandbox_volume_l1044_104480

def length : ℕ := 312
def width : ℕ := 146
def depth : ℕ := 75
def volume (l w d : ℕ) : ℕ := l * w * d

theorem sandbox_volume : volume length width depth = 3429000 := by
  sorry

end sandbox_volume_l1044_104480


namespace evaluate_star_property_l1044_104430

noncomputable def star (a b : ℕ) : ℕ := b ^ a

theorem evaluate_star_property (a b c m : ℕ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hm : 0 < m) :
  (star a b ≠ star b a) ∧
  (star a (star b c) ≠ star (star a b) c) ∧
  (star a (b ^ m) ≠ star (star a m) b) ∧
  ((star a b) ^ m ≠ star a (m * b)) :=
by
  sorry

end evaluate_star_property_l1044_104430


namespace abs_five_minus_two_e_l1044_104407

noncomputable def e : ℝ := 2.718

theorem abs_five_minus_two_e : |5 - 2 * e| = 0.436 := by
  sorry

end abs_five_minus_two_e_l1044_104407


namespace perpendicular_bisector_c_value_l1044_104464

theorem perpendicular_bisector_c_value :
  (∃ c : ℝ, ∀ x y : ℝ, 
    2 * x - y = c ↔ x = 5 ∧ y = 8) → c = 2 := 
by
  sorry

end perpendicular_bisector_c_value_l1044_104464


namespace find_principal_amount_l1044_104403

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

end find_principal_amount_l1044_104403


namespace relationship_a_b_c_l1044_104481

noncomputable def a := Real.log 3 / Real.log (1/2)
noncomputable def b := Real.log (1/2) / Real.log 3
noncomputable def c := Real.exp (0.3 * Real.log 2)

theorem relationship_a_b_c : 
  a < b ∧ b < c := 
by {
  sorry
}

end relationship_a_b_c_l1044_104481


namespace find_remainder_l1044_104459

theorem find_remainder (x y P Q : ℕ) (hx : 0 < x) (hy : 0 < y) (h : x^4 + y^4 = (P + 13) * (x + y) + Q) : Q = 8 :=
sorry

end find_remainder_l1044_104459


namespace diamond_associative_l1044_104488

def diamond (a b : ℕ) : ℕ := a ^ (b / a)

theorem diamond_associative (a b c : ℕ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c):
  diamond a (diamond b c) = diamond (diamond a b) c :=
sorry

end diamond_associative_l1044_104488


namespace determine_abcd_l1044_104427

theorem determine_abcd (a b c d : ℕ) (h₀ : 0 ≤ a ∧ a ≤ 9) (h₁ : 0 ≤ b ∧ b ≤ 9) 
    (h₂ : 0 ≤ c ∧ c ≤ 9) (h₃ : 0 ≤ d ∧ d ≤ 9) 
    (h₄ : (10 * a + b) / 99 + (1000 * a + 100 * b + 10 * c + d) / 9999 = 27 / 37) :
    1000 * a + 100 * b + 10 * c + d = 3644 :=
by
  sorry

end determine_abcd_l1044_104427


namespace bridge_weight_excess_l1044_104436

theorem bridge_weight_excess :
  ∀ (Kelly_weight Megan_weight Mike_weight : ℕ),
  Kelly_weight = 34 →
  Kelly_weight = 85 * Megan_weight / 100 →
  Mike_weight = Megan_weight + 5 →
  (Kelly_weight + Megan_weight + Mike_weight) - 100 = 19 :=
by
  intros Kelly_weight Megan_weight Mike_weight
  intros h1 h2 h3
  sorry

end bridge_weight_excess_l1044_104436


namespace variance_is_0_02_l1044_104489

def data_points : List ℝ := [9.8, 9.9, 10.1, 10, 10.2]

noncomputable def mean (l : List ℝ) : ℝ :=
  l.sum / l.length

noncomputable def variance (l : List ℝ) : ℝ :=
  let m := mean l
  (l.map (λ x => (x - m) ^ 2)).sum / l.length

theorem variance_is_0_02 : variance data_points = 0.02 :=
by
  sorry

end variance_is_0_02_l1044_104489


namespace nails_to_buy_l1044_104437

-- Define the initial number of nails Tom has
def initial_nails : ℝ := 247

-- Define the number of nails found in the toolshed
def toolshed_nails : ℝ := 144

-- Define the number of nails found in a drawer
def drawer_nails : ℝ := 0.5

-- Define the number of nails given by the neighbor
def neighbor_nails : ℝ := 58.75

-- Define the total number of nails needed for the project
def total_needed_nails : ℝ := 625.25

-- Define the total number of nails Tom already has
def total_existing_nails : ℝ := 
  initial_nails + toolshed_nails + drawer_nails + neighbor_nails

-- Prove that Tom needs to buy 175 more nails
theorem nails_to_buy :
  total_needed_nails - total_existing_nails = 175 := by
  sorry

end nails_to_buy_l1044_104437


namespace probability_xavier_yvonne_not_zelda_wendell_l1044_104405

theorem probability_xavier_yvonne_not_zelda_wendell
  (P_Xavier_solves : ℚ)
  (P_Yvonne_solves : ℚ)
  (P_Zelda_solves : ℚ)
  (P_Wendell_solves : ℚ) :
  P_Xavier_solves = 1/4 →
  P_Yvonne_solves = 1/3 →
  P_Zelda_solves = 5/8 →
  P_Wendell_solves = 1/2 →
  (P_Xavier_solves * P_Yvonne_solves * (1 - P_Zelda_solves) * (1 - P_Wendell_solves)) = 1/64 :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  simp
  sorry

end probability_xavier_yvonne_not_zelda_wendell_l1044_104405


namespace intersect_complement_A_B_eq_l1044_104426

open Set

variable (U : Set ℝ)
variable (A : Set ℝ)
variable (B : Set ℝ)

noncomputable def complement_A : Set ℝ := U \ A
noncomputable def intersection_complement_A_B : Set ℝ := complement_A U A ∩ B

theorem intersect_complement_A_B_eq : 
  U = univ ∧ A = {x : ℝ | x + 1 < 0} ∧ B = {x : ℝ | x - 3 < 0} →
  intersection_complement_A_B U A B = Icc (-1 : ℝ) 3 :=
by
  intro h
  sorry

end intersect_complement_A_B_eq_l1044_104426


namespace sum_a_b_l1044_104458

variable {a b : ℝ}

theorem sum_a_b (hab : a * b = 5) (hrecip : 1 / (a^2) + 1 / (b^2) = 0.6) : a + b = 5 ∨ a + b = -5 :=
sorry

end sum_a_b_l1044_104458


namespace evaluate_expression_l1044_104406

theorem evaluate_expression : 2 + (2 / (2 + (2 / (2 + 3)))) = 17 / 6 := 
by
  sorry

end evaluate_expression_l1044_104406


namespace ratio_of_bases_l1044_104460

-- Definitions for an isosceles trapezoid
def isosceles_trapezoid (s t : ℝ) := ∃ (a b c d : ℝ), s = d ∧ s = a ∧ t = b ∧ (a + c = b + d)

-- Main theorem statement based on conditions and required ratio
theorem ratio_of_bases (s t : ℝ) (h1 : isosceles_trapezoid s t)
  (h2 : s = s) (h3 : t = t) : s / t = 3 / 5 :=
by { sorry }

end ratio_of_bases_l1044_104460


namespace f_2016_is_1_l1044_104453

noncomputable def f : ℤ → ℤ := sorry

axiom h1 : f 1 = 1
axiom h2 : f 2015 ≠ 1
axiom h3 : ∀ a b : ℤ, f (a + b) ≤ max (f a) (f b)
axiom h4 : ∀ x : ℤ, f x = f (-x)

theorem f_2016_is_1 : f 2016 = 1 := 
by 
  sorry

end f_2016_is_1_l1044_104453


namespace graph_union_l1044_104410

-- Definitions of the conditions from part a)
def graph1 (z y : ℝ) : Prop := z^4 - 6 * y^4 = 3 * z^2 - 2

def graph_hyperbola (z y : ℝ) : Prop := z^2 - 3 * y^2 = 2

def graph_ellipse (z y : ℝ) : Prop := z^2 - 2 * y^2 = 1

-- Lean statement to prove the question is equivalent to the answer
theorem graph_union (z y : ℝ) : graph1 z y ↔ (graph_hyperbola z y ∨ graph_ellipse z y) := 
sorry

end graph_union_l1044_104410


namespace finite_set_cardinality_l1044_104496

-- Define the main theorem statement
theorem finite_set_cardinality (m : ℕ) (A : Finset ℤ) (B : ℕ → Finset ℤ)
  (hm : m ≥ 2)
  (hB : ∀ k : ℕ, k ∈ Finset.range m.succ → (B k).sum id = m^k) :
  A.card ≥ m / 2 := 
sorry

end finite_set_cardinality_l1044_104496


namespace cube_root_21952_is_28_l1044_104419

theorem cube_root_21952_is_28 :
  ∃ n : ℕ, n^3 = 21952 ∧ n = 28 :=
sorry

end cube_root_21952_is_28_l1044_104419


namespace powerjet_pumps_250_gallons_in_30_minutes_l1044_104441

theorem powerjet_pumps_250_gallons_in_30_minutes :
  let rate : ℝ := 500
  let time_in_hours : ℝ := 1 / 2
  rate * time_in_hours = 250 :=
by
  sorry

end powerjet_pumps_250_gallons_in_30_minutes_l1044_104441


namespace obtain_2015_in_4_operations_obtain_2015_in_3_operations_l1044_104424

-- Define what an operation is
def operation (cards : List ℕ) : List ℕ :=
  sorry  -- Implementation of this is unnecessary for the statement

-- Check if 2015 can be obtained in 4 operations
def can_obtain_2015_in_4_operations (initial_cards : List ℕ) : Prop :=
  ∃ cards, (operation^[4] initial_cards) = cards ∧ 2015 ∈ cards

-- Check if 2015 can be obtained in 3 operations
def can_obtain_2015_in_3_operations (initial_cards : List ℕ) : Prop :=
  ∃ cards, (operation^[3] initial_cards) = cards ∧ 2015 ∈ cards

theorem obtain_2015_in_4_operations :
  can_obtain_2015_in_4_operations [1, 2] :=
sorry

theorem obtain_2015_in_3_operations :
  can_obtain_2015_in_3_operations [1, 2] :=
sorry

end obtain_2015_in_4_operations_obtain_2015_in_3_operations_l1044_104424


namespace molecular_weight_of_6_moles_l1044_104421

-- Define the molecular weight of the compound
def molecular_weight : ℕ := 1404

-- Define the number of moles
def number_of_moles : ℕ := 6

-- The hypothesis would be the molecular weight condition
theorem molecular_weight_of_6_moles : number_of_moles * molecular_weight = 8424 :=
by sorry

end molecular_weight_of_6_moles_l1044_104421


namespace contractor_male_workers_l1044_104493

noncomputable def number_of_male_workers (M : ℕ) : Prop :=
  let female_wages : ℕ := 15 * 20
  let child_wages : ℕ := 5 * 8
  let total_wages : ℕ := 35 * M + female_wages + child_wages
  let total_workers : ℕ := M + 15 + 5
  (total_wages / total_workers) = 26

theorem contractor_male_workers : ∃ M : ℕ, number_of_male_workers M ∧ M = 20 :=
by
  use 20
  sorry

end contractor_male_workers_l1044_104493


namespace MaryIncomeIs64PercentOfJuanIncome_l1044_104400

variable {J T M : ℝ}

-- Conditions
def TimIncome (J : ℝ) : ℝ := 0.40 * J
def MaryIncome (T : ℝ) : ℝ := 1.60 * T

-- Theorem to prove
theorem MaryIncomeIs64PercentOfJuanIncome (J : ℝ) :
  MaryIncome (TimIncome J) = 0.64 * J :=
by
  sorry

end MaryIncomeIs64PercentOfJuanIncome_l1044_104400


namespace student_missed_20_l1044_104473

theorem student_missed_20 {n : ℕ} (S_correct : ℕ) (S_incorrect : ℕ) 
    (h1 : S_correct = n * (n + 1) / 2)
    (h2 : S_incorrect = S_correct - 20) : 
    S_incorrect = n * (n + 1) / 2 - 20 := 
sorry

end student_missed_20_l1044_104473


namespace fixed_monthly_costs_l1044_104462

theorem fixed_monthly_costs
  (production_cost_per_component : ℕ)
  (shipping_cost_per_component : ℕ)
  (components_per_month : ℕ)
  (lowest_price_per_component : ℕ)
  (total_revenue : ℕ)
  (total_variable_cost : ℕ)
  (F : ℕ) :
  production_cost_per_component = 80 →
  shipping_cost_per_component = 5 →
  components_per_month = 150 →
  lowest_price_per_component = 195 →
  total_variable_cost = components_per_month * (production_cost_per_component + shipping_cost_per_component) →
  total_revenue = components_per_month * lowest_price_per_component →
  total_revenue = total_variable_cost + F →
  F = 16500 :=
by
  intros h1 h2 h3 h4 h5 h6 h7
  sorry

end fixed_monthly_costs_l1044_104462


namespace socks_count_l1044_104497

theorem socks_count
  (black_socks : ℕ) 
  (white_socks : ℕ)
  (H1 : white_socks = 4 * black_socks)
  (H2 : black_socks = 6)
  (H3 : white_socks / 2 = white_socks - (white_socks / 2)) :
  (white_socks / 2) - black_socks = 6 := by
  sorry

end socks_count_l1044_104497


namespace minimum_additional_squares_to_symmetry_l1044_104417

-- Define the type for coordinates in the grid
structure Coord where
  x : Nat
  y : Nat

-- Define the conditions
def initial_shaded_squares : List Coord := [
  ⟨2, 4⟩, ⟨3, 2⟩, ⟨5, 1⟩, ⟨1, 4⟩
]

def grid_size : Coord := ⟨6, 5⟩

def vertical_line_of_symmetry : Nat := 3 -- between columns 3 and 4
def horizontal_line_of_symmetry : Nat := 2 -- between rows 2 and 3

-- Define reflection across lines of symmetry
def reflect_vertical (c : Coord) : Coord :=
  ⟨2 * vertical_line_of_symmetry - c.x, c.y⟩

def reflect_horizontal (c : Coord) : Coord :=
  ⟨c.x, 2 * horizontal_line_of_symmetry - c.y⟩

def reflect_both (c : Coord) : Coord :=
  reflect_vertical (reflect_horizontal c)

-- Define the theorem
theorem minimum_additional_squares_to_symmetry :
  ∃ (additional_squares : Nat), additional_squares = 5 := 
sorry

end minimum_additional_squares_to_symmetry_l1044_104417


namespace strawberries_left_l1044_104491

theorem strawberries_left (picked: ℕ) (eaten: ℕ) (initial_count: picked = 35) (eaten_count: eaten = 2) :
  picked - eaten = 33 :=
by
  sorry

end strawberries_left_l1044_104491


namespace area_triangle_ABF_proof_area_triangle_AFD_proof_l1044_104465

variable (A B C D M F : Type)
variable (area_square : Real) (midpoint_D_CM : Prop) (lies_on_line_BC : Prop)

-- Given conditions
axiom area_ABCD_300 : area_square = 300
axiom M_midpoint_DC : midpoint_D_CM
axiom F_on_line_BC : lies_on_line_BC

-- Define areas for the triangles
def area_triangle_ABF : Real := 300
def area_triangle_AFD : Real := 150

-- Prove that given the conditions, the area of triangle ABF is 300 cm²
theorem area_triangle_ABF_proof : area_square = 300 ∧ midpoint_D_CM ∧ lies_on_line_BC → area_triangle_ABF = 300 :=
by
  intro h
  sorry

-- Prove that given the conditions, the area of triangle AFD is 150 cm²
theorem area_triangle_AFD_proof : area_square = 300 ∧ midpoint_D_CM ∧ lies_on_line_BC → area_triangle_AFD = 150 :=
by
  intro h
  sorry

end area_triangle_ABF_proof_area_triangle_AFD_proof_l1044_104465


namespace ratio_of_numbers_l1044_104451

theorem ratio_of_numbers (A B D M : ℕ) 
  (h1 : A + B + D = M)
  (h2 : Nat.gcd A B = D)
  (h3 : Nat.lcm A B = M)
  (h4 : A ≥ B) : A / B = 3 / 2 :=
by
  sorry

end ratio_of_numbers_l1044_104451


namespace cost_price_6500_l1044_104443

variable (CP SP : ℝ)

-- Condition 1: The selling price is 30% more than the cost price.
def selling_price (CP : ℝ) : ℝ := CP * 1.3

-- Condition 2: The selling price is Rs. 8450.
axiom selling_price_8450 : selling_price CP = 8450

-- Prove that the cost price of the computer table is Rs. 6500.
theorem cost_price_6500 : CP = 6500 :=
by
  sorry

end cost_price_6500_l1044_104443


namespace range_of_a_l1044_104467

variable {x a : ℝ}

def p (x : ℝ) := 2*x^2 - 3*x + 1 ≤ 0
def q (x : ℝ) (a : ℝ) := (x - a) * (x - a - 1) ≤ 0

theorem range_of_a (h : ¬ p x → ¬ q x a) : 0 ≤ a ∧ a ≤ 1/2 := by
  sorry

end range_of_a_l1044_104467


namespace expected_red_pairs_correct_l1044_104457

-- Define the number of red cards and the total number of cards
def red_cards : ℕ := 25
def total_cards : ℕ := 50

-- Calculate the probability that one red card is followed by another red card in a circle of total_cards
def prob_adj_red : ℚ := (red_cards - 1) / (total_cards - 1)

-- The expected number of pairs of adjacent red cards
def expected_adj_red_pairs : ℚ := red_cards * prob_adj_red

-- The theorem to be proved: the expected number of adjacent red pairs is 600/49
theorem expected_red_pairs_correct : expected_adj_red_pairs = 600 / 49 :=
by
  -- Placeholder for the proof
  sorry

end expected_red_pairs_correct_l1044_104457


namespace who_is_werewolf_choose_companion_l1044_104418

-- Define inhabitants with their respective statements
inductive Inhabitant
| A | B | C

-- Assume each inhabitant can be either a knight (truth-teller) or a liar
def is_knight (i : Inhabitant) : Prop := sorry

-- Define statements made by each inhabitant
def A_statement : Prop := ∃ werewolf : Inhabitant, werewolf = Inhabitant.C
def B_statement : Prop := ¬(∃ werewolf : Inhabitant, werewolf = Inhabitant.B)
def C_statement : Prop := ∃ liar1 liar2 : Inhabitant, liar1 ≠ liar2 ∧ liar1 ≠ Inhabitant.C ∧ liar2 ≠ Inhabitant.C

-- Define who is the werewolf (liar)
def is_werewolf (i : Inhabitant) : Prop := ¬is_knight i

-- The given conditions from statements
axiom A_is_knight : is_knight Inhabitant.A ↔ A_statement
axiom B_is_knight : is_knight Inhabitant.B ↔ B_statement
axiom C_is_knight : is_knight Inhabitant.C ↔ C_statement

-- The conclusion: C is the werewolf and thus a liar.
theorem who_is_werewolf : is_werewolf Inhabitant.C :=
by sorry

-- Choosing a companion: 
-- If C is a werewolf, we prefer to pick A as a companion over B or C.
theorem choose_companion (worry_about_werewolf : Bool) : Inhabitant :=
if worry_about_werewolf then Inhabitant.A else sorry

end who_is_werewolf_choose_companion_l1044_104418


namespace find_c_squared_ab_l1044_104456

theorem find_c_squared_ab (a b c : ℝ) (h1 : a^2 * (b + c) = 2008) (h2 : b^2 * (a + c) = 2008) (h3 : a ≠ b) : 
  c^2 * (a + b) = 2008 :=
sorry

end find_c_squared_ab_l1044_104456


namespace starting_weight_of_labrador_puppy_l1044_104495

theorem starting_weight_of_labrador_puppy :
  ∃ L : ℝ,
    (L + 0.25 * L) - (12 + 0.25 * 12) = 35 ∧ 
    L = 40 :=
by
  use 40
  sorry

end starting_weight_of_labrador_puppy_l1044_104495


namespace prime_numbers_satisfy_equation_l1044_104454

noncomputable def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem prime_numbers_satisfy_equation :
  ∃ (p q r : ℕ), is_prime p ∧ is_prime q ∧ is_prime r ∧ (p + q^2 = r^4) ∧ 
  (p = 7) ∧ (q = 3) ∧ (r = 2) :=
by
  sorry

end prime_numbers_satisfy_equation_l1044_104454


namespace eight_bees_have_48_legs_l1044_104414

  def legs_per_bee : ℕ := 6
  def number_of_bees : ℕ := 8
  def total_legs : ℕ := 48

  theorem eight_bees_have_48_legs :
    number_of_bees * legs_per_bee = total_legs :=
  by
    sorry
  
end eight_bees_have_48_legs_l1044_104414
