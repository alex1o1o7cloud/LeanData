import Mathlib

namespace NUMINAMATH_GPT_composition_points_value_l1670_167054

theorem composition_points_value (f g : ℕ → ℕ) (ab cd : ℕ) 
  (h₁ : f 2 = 6) 
  (h₂ : f 3 = 4) 
  (h₃ : f 4 = 2)
  (h₄ : g 2 = 4) 
  (h₅ : g 3 = 2) 
  (h₆ : g 5 = 6) :
  let (a, b) := (2, 6)
  let (c, d) := (3, 4)
  ab + cd = (a * b) + (c * d) :=
by {
  sorry
}

end NUMINAMATH_GPT_composition_points_value_l1670_167054


namespace NUMINAMATH_GPT_smallest_next_divisor_l1670_167045

theorem smallest_next_divisor (n : ℕ) (hn : n % 2 = 0) (h4d : 1000 ≤ n ∧ n < 10000) (hdiv : 221 ∣ n) : 
  ∃ (d : ℕ), d = 238 ∧ 221 < d ∧ d ∣ n :=
by
  sorry

end NUMINAMATH_GPT_smallest_next_divisor_l1670_167045


namespace NUMINAMATH_GPT_lcm_of_9_12_15_is_180_l1670_167090

theorem lcm_of_9_12_15_is_180 :
  Nat.lcm 9 (Nat.lcm 12 15) = 180 :=
by
  sorry

end NUMINAMATH_GPT_lcm_of_9_12_15_is_180_l1670_167090


namespace NUMINAMATH_GPT_find_A_solution_l1670_167074

theorem find_A_solution (A : ℝ) (h : 32 * A^3 = 42592) : A = 11 :=
sorry

end NUMINAMATH_GPT_find_A_solution_l1670_167074


namespace NUMINAMATH_GPT_positive_difference_solutions_l1670_167083

theorem positive_difference_solutions : 
  ∀ (r : ℝ), r ≠ -3 → 
  (∃ r1 r2 : ℝ, (r^2 - 6*r - 20) / (r + 3) = 3*r + 10 → r1 ≠ r2 ∧ 
  |r1 - r2| = 20) :=
by
  sorry

end NUMINAMATH_GPT_positive_difference_solutions_l1670_167083


namespace NUMINAMATH_GPT_value_range_of_f_l1670_167072

noncomputable def f (x : ℝ) : ℝ := 2 + Real.logb 5 (x + 3)

theorem value_range_of_f :
  ∀ x ∈ Set.Icc (-2 : ℝ) 2, f x ∈ Set.Icc (2 : ℝ) 3 := 
by
  sorry

end NUMINAMATH_GPT_value_range_of_f_l1670_167072


namespace NUMINAMATH_GPT_problem1_problem2_l1670_167007

-- Problem 1
theorem problem1 : (π - 1)^0 + 4 * Real.sin (π / 4) - Real.sqrt 8 + abs (-3) = 4 := sorry

-- Problem 2
theorem problem2 (a : ℝ) (ha : a ≠ 1) : (1 - 1 / a) / ((a^2 - 2 * a + 1) / a) = 1 / (a - 1) := sorry

end NUMINAMATH_GPT_problem1_problem2_l1670_167007


namespace NUMINAMATH_GPT_angle_quadrant_l1670_167050

theorem angle_quadrant (theta : ℤ) (h_theta : theta = -3290) : 
  ∃ q : ℕ, q = 4 := 
by 
  sorry

end NUMINAMATH_GPT_angle_quadrant_l1670_167050


namespace NUMINAMATH_GPT_real_roots_range_l1670_167001

theorem real_roots_range (k : ℝ) : 
  (∃ x : ℝ, k*x^2 - 6*x + 9 = 0) ↔ k ≤ 1 :=
sorry

end NUMINAMATH_GPT_real_roots_range_l1670_167001


namespace NUMINAMATH_GPT_fencing_cost_correct_l1670_167080

noncomputable def length : ℝ := 80
noncomputable def diff : ℝ := 60
noncomputable def cost_per_meter : ℝ := 26.50

-- Let's calculate the breadth first
noncomputable def breadth : ℝ := length - diff

-- Calculate the perimeter
noncomputable def perimeter : ℝ := 2 * (length + breadth)

-- Calculate the total cost
noncomputable def total_cost : ℝ := perimeter * cost_per_meter

theorem fencing_cost_correct : total_cost = 5300 := 
by 
  sorry

end NUMINAMATH_GPT_fencing_cost_correct_l1670_167080


namespace NUMINAMATH_GPT_molecular_weight_of_compound_l1670_167038

theorem molecular_weight_of_compound :
  let Cu_atoms := 2
  let C_atoms := 3
  let O_atoms := 5
  let N_atoms := 1
  let atomic_weight_Cu := 63.546
  let atomic_weight_C := 12.011
  let atomic_weight_O := 15.999
  let atomic_weight_N := 14.007
  Cu_atoms * atomic_weight_Cu +
  C_atoms * atomic_weight_C +
  O_atoms * atomic_weight_O +
  N_atoms * atomic_weight_N = 257.127 :=
by
  sorry

end NUMINAMATH_GPT_molecular_weight_of_compound_l1670_167038


namespace NUMINAMATH_GPT_distance_upstream_l1670_167061

/-- Proof that the distance a man swims upstream is 18 km given certain conditions. -/
theorem distance_upstream (c : ℝ) (h1 : 54 / (12 + c) = 3) (h2 : 12 - c = 6) : (12 - c) * 3 = 18 :=
by
  sorry

end NUMINAMATH_GPT_distance_upstream_l1670_167061


namespace NUMINAMATH_GPT_minimum_green_sticks_l1670_167035

def natasha_sticks (m n : ℕ) : ℕ :=
  if (m = 3 ∧ n = 3) then 5 else 0

theorem minimum_green_sticks (m n : ℕ) (grid : m = 3 ∧ n = 3) :
  natasha_sticks m n = 5 :=
by
  sorry

end NUMINAMATH_GPT_minimum_green_sticks_l1670_167035


namespace NUMINAMATH_GPT_find_b_l1670_167044

noncomputable def p (x : ℕ) := 3 * x + 5
noncomputable def q (x : ℕ) (b : ℕ) := 4 * x - b

theorem find_b : ∃ (b : ℕ), p (q 3 b) = 29 ∧ b = 4 := sorry

end NUMINAMATH_GPT_find_b_l1670_167044


namespace NUMINAMATH_GPT_abs_nonneg_rational_l1670_167058

theorem abs_nonneg_rational (a : ℚ) : |a| ≥ 0 :=
sorry

end NUMINAMATH_GPT_abs_nonneg_rational_l1670_167058


namespace NUMINAMATH_GPT_cylindrical_to_rectangular_point_l1670_167037

noncomputable def cylindrical_to_rectangular (r θ z : ℝ) : ℝ × ℝ × ℝ :=
  (r * Real.cos θ, r * Real.sin θ, z)

theorem cylindrical_to_rectangular_point :
  cylindrical_to_rectangular (Real.sqrt 2) (Real.pi / 4) 1 = (1, 1, 1) :=
by
  sorry

end NUMINAMATH_GPT_cylindrical_to_rectangular_point_l1670_167037


namespace NUMINAMATH_GPT_evaluate_expression_l1670_167052

theorem evaluate_expression (c d : ℕ) (hc : c = 4) (hd : d = 2) :
  (c^c - c * (c - d)^c)^c = 136048896 := by
  sorry

end NUMINAMATH_GPT_evaluate_expression_l1670_167052


namespace NUMINAMATH_GPT_power_multiplication_l1670_167019

variable (p : ℝ)  -- Assuming p is a real number

theorem power_multiplication :
  (-p)^2 * (-p)^3 = -p^5 :=
sorry

end NUMINAMATH_GPT_power_multiplication_l1670_167019


namespace NUMINAMATH_GPT_no_sum_2015_l1670_167073

theorem no_sum_2015 (x a : ℤ) : 3 * x + 3 * a ≠ 2015 := by
  sorry

end NUMINAMATH_GPT_no_sum_2015_l1670_167073


namespace NUMINAMATH_GPT_total_money_collected_is_140_l1670_167085

def total_attendees : ℕ := 280
def child_attendees : ℕ := 80
def adult_attendees : ℕ := total_attendees - child_attendees
def adult_ticket_cost : ℝ := 0.60
def child_ticket_cost : ℝ := 0.25

def money_collected_from_adults : ℝ := adult_attendees * adult_ticket_cost
def money_collected_from_children : ℝ := child_attendees * child_ticket_cost
def total_money_collected : ℝ := money_collected_from_adults + money_collected_from_children

theorem total_money_collected_is_140 : total_money_collected = 140 := by
  sorry

end NUMINAMATH_GPT_total_money_collected_is_140_l1670_167085


namespace NUMINAMATH_GPT_problem_solution_l1670_167000

variable (f : ℝ → ℝ)

-- Let f be an odd function
def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

-- f(x) = f(4 - x) for all x in ℝ
def satisfies_symmetry (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (4 - x)

-- f is increasing on [0, 2]
def is_increasing_on_interval (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x → x < y → y ≤ b → f x < f y

theorem problem_solution :
  is_odd_function f →
  satisfies_symmetry f →
  is_increasing_on_interval f 0 2 →
  f 6 < f 4 ∧ f 4 < f 1 :=
by
  intros
  sorry

end NUMINAMATH_GPT_problem_solution_l1670_167000


namespace NUMINAMATH_GPT_angle_same_terminal_side_l1670_167086

theorem angle_same_terminal_side (k : ℤ) : ∃ k : ℤ, 290 = k * 360 - 70 :=
by
  sorry

end NUMINAMATH_GPT_angle_same_terminal_side_l1670_167086


namespace NUMINAMATH_GPT_men_wages_eq_13_5_l1670_167010

-- Definitions based on problem conditions
def wages (men women boys : ℕ) : ℝ :=
  if 9 * men + women + 7 * boys = 216 then
    men
  else 
    0

def equivalent_wage (men_wage women_wage boy_wage : ℝ) : Prop :=
  9 * men_wage = women_wage ∧
  women_wage = 7 * boy_wage

def total_earning (men_wage women_wage boy_wage : ℝ) : Prop :=
  9 * men_wage + 7 * boy_wage = 216

-- Theorem statement
theorem men_wages_eq_13_5 (M_wage W_wage B_wage : ℝ) :
  equivalent_wage M_wage W_wage B_wage →
  total_earning M_wage W_wage B_wage →
  M_wage = 13.5 :=
by 
  intros h_equiv h_total
  sorry

end NUMINAMATH_GPT_men_wages_eq_13_5_l1670_167010


namespace NUMINAMATH_GPT_max_triangle_area_l1670_167095

theorem max_triangle_area :
  ∃ a b c : ℝ, 0 ≤ a ∧ a ≤ 1 ∧ 1 ≤ b ∧ b ≤ 2 ∧ 2 ≤ c ∧ c ≤ 3 ∧ 
  (a + b > c ∧ a + c > b ∧ b + c > a) ∧ (1 ≤ 0.5 * a * b) := sorry

end NUMINAMATH_GPT_max_triangle_area_l1670_167095


namespace NUMINAMATH_GPT_trains_cross_time_l1670_167070

noncomputable def time_to_cross (len1 len2 speed1_kmh speed2_kmh : ℝ) : ℝ :=
  let speed1_ms := speed1_kmh * (5 / 18)
  let speed2_ms := speed2_kmh * (5 / 18)
  let relative_speed_ms := speed1_ms + speed2_ms
  let total_distance := len1 + len2
  total_distance / relative_speed_ms

theorem trains_cross_time :
  time_to_cross 1500 1000 90 75 = 54.55 := by
  sorry

end NUMINAMATH_GPT_trains_cross_time_l1670_167070


namespace NUMINAMATH_GPT_greatest_k_divides_n_l1670_167023

theorem greatest_k_divides_n (n : ℕ) (h_pos : 0 < n) (h_divisors_n : Nat.totient n = 72) (h_divisors_5n : Nat.totient (5 * n) = 90) : ∃ k : ℕ, ∀ m : ℕ, (5^k ∣ n) → (5^(k+1) ∣ n) → k = 3 :=
by
  sorry

end NUMINAMATH_GPT_greatest_k_divides_n_l1670_167023


namespace NUMINAMATH_GPT_f_monotonicity_l1670_167064

noncomputable def f : ℝ → ℝ := sorry -- Definition of the function f(x)

axiom f_symm (x : ℝ) : f (1 - x) = f x

axiom f_derivative (x : ℝ) : (x - 1 / 2) * (deriv f x) > 0

theorem f_monotonicity (x1 x2 : ℝ) (h1 : x1 < x2) (h2 : x1 + x2 > 1) : f x1 < f x2 :=
sorry

end NUMINAMATH_GPT_f_monotonicity_l1670_167064


namespace NUMINAMATH_GPT_value_of_x_l1670_167081

theorem value_of_x (v w z y x : ℤ) 
  (h1 : v = 90)
  (h2 : w = v + 30)
  (h3 : z = w + 21)
  (h4 : y = z + 11)
  (h5 : x = y + 6) : 
  x = 158 :=
by 
  sorry

end NUMINAMATH_GPT_value_of_x_l1670_167081


namespace NUMINAMATH_GPT_find_n_l1670_167098

open Nat

def is_prime (p : ℕ) : Prop :=
  p > 1 ∧ ∀ m : ℕ, m ∣ p → m = 1 ∨ m = p

def twin_primes (p q : ℕ) : Prop :=
  is_prime p ∧ is_prime q ∧ q = p + 2

def is_twins_prime_sum (n p q : ℕ) : Prop :=
  twin_primes p q ∧ is_prime (2^n + p) ∧ is_prime (2^n + q)

theorem find_n :
  ∀ (n : ℕ), (∃ (p q : ℕ), is_twins_prime_sum n p q) → (n = 1 ∨ n = 3) :=
sorry

end NUMINAMATH_GPT_find_n_l1670_167098


namespace NUMINAMATH_GPT_floor_diff_l1670_167088

theorem floor_diff (x : ℝ) (h : x = 13.2) : 
  (Int.floor (x^2) - (Int.floor x) * (Int.floor x) = 5) := by
  sorry

end NUMINAMATH_GPT_floor_diff_l1670_167088


namespace NUMINAMATH_GPT_complex_expression_evaluation_l1670_167013

-- Defining the imaginary unit
def i : ℂ := Complex.I

-- Defining the complex number z
def z : ℂ := 1 - i

-- Stating the theorem to prove
theorem complex_expression_evaluation : z^2 + (2 / z) = 1 - i := by
  sorry

end NUMINAMATH_GPT_complex_expression_evaluation_l1670_167013


namespace NUMINAMATH_GPT_relationship_l1670_167060

-- Given definitions
def S : ℕ := 31
def L : ℕ := 124 - S

-- Proving the relationship
theorem relationship: S + L = 124 ∧ S = 31 → L = S + 62 := by
  sorry

end NUMINAMATH_GPT_relationship_l1670_167060


namespace NUMINAMATH_GPT_f_log2_9_l1670_167089

def f (x : ℝ) : ℝ := sorry

theorem f_log2_9 : 
  (∀ x, f (x + 1) = 1 / f x) → 
  (∀ x, 0 < x ∧ x ≤ 1 → f x = 2^x) → 
  f (Real.log 9 / Real.log 2) = 8 / 9 :=
by
  intros h1 h2
  sorry

end NUMINAMATH_GPT_f_log2_9_l1670_167089


namespace NUMINAMATH_GPT_work_rate_a_b_l1670_167063

/-- a and b can do a piece of work in some days, b and c in 5 days, c and a in 15 days. If c takes 12 days to do the work, 
    prove that a and b together can complete the work in 10 days.
-/
theorem work_rate_a_b
  (A B C : ℚ) 
  (h1 : B + C = 1 / 5)
  (h2 : C + A = 1 / 15)
  (h3 : C = 1 / 12) :
  (A + B = 1 / 10) := 
sorry

end NUMINAMATH_GPT_work_rate_a_b_l1670_167063


namespace NUMINAMATH_GPT_house_to_market_distance_l1670_167020

-- Definitions of the conditions
def distance_to_school : ℕ := 50
def distance_back_home : ℕ := 50
def total_distance_walked : ℕ := 140

-- Statement of the problem
theorem house_to_market_distance :
  distance_to_market = total_distance_walked - (distance_to_school + distance_back_home) :=
by
  sorry

end NUMINAMATH_GPT_house_to_market_distance_l1670_167020


namespace NUMINAMATH_GPT_train_length_is_1400_l1670_167018

theorem train_length_is_1400
  (L : ℝ) 
  (h1 : ∃ speed, speed = L / 100) 
  (h2 : ∃ speed, speed = (L + 700) / 150) :
  L = 1400 :=
by sorry

end NUMINAMATH_GPT_train_length_is_1400_l1670_167018


namespace NUMINAMATH_GPT_construct_convex_hexagon_l1670_167057

-- Definitions of the sides and their lengths
variables {A B C D E F : Type} -- Points of the hexagon
variables {AB BC CD DE EF FA : ℝ}  -- Lengths of the sides
variables (convex_hexagon : Prop) -- the hexagon is convex

-- Hypotheses of parallel and equal opposite sides
variables (H_AB_DE : AB = DE)
variables (H_BC_EF : BC = EF)
variables (H_CD_AF : CD = AF)

-- Define the construction of the hexagon under the given conditions
theorem construct_convex_hexagon
  (convex_hexagon : Prop)
  (H_AB_DE : AB = DE)
  (H_BC_EF : BC = EF)
  (H_CD_AF : CD = AF) : 
  ∃ (A B C D E F : Type), 
    A ≠ B ∧ B ≠ C ∧ C ≠ D ∧ D ≠ E ∧ E ≠ F ∧ F ≠ A ∧ convex_hexagon ∧ 
    (AB = FA) ∧ (AF = CD) ∧ (BC = EF) ∧ (AB = DE) := 
sorry -- Proof omitted

end NUMINAMATH_GPT_construct_convex_hexagon_l1670_167057


namespace NUMINAMATH_GPT_sqrt_of_9_neg_sqrt_of_0_49_pm_sqrt_of_64_div_81_l1670_167005

-- Definition and proof of sqrt(9) = 3
theorem sqrt_of_9 : Real.sqrt 9 = 3 := by
  sorry

-- Definition and proof of -sqrt(0.49) = -0.7
theorem neg_sqrt_of_0_49 : -Real.sqrt 0.49 = -0.7 := by
  sorry

-- Definition and proof of ±sqrt(64/81) = ±(8/9)
theorem pm_sqrt_of_64_div_81 : (Real.sqrt (64 / 81) = 8 / 9) ∧ (Real.sqrt (64 / 81) = -8 / 9) := by
  sorry

end NUMINAMATH_GPT_sqrt_of_9_neg_sqrt_of_0_49_pm_sqrt_of_64_div_81_l1670_167005


namespace NUMINAMATH_GPT_two_real_roots_opposite_signs_l1670_167056

theorem two_real_roots_opposite_signs (a : ℝ) :
  (∃ x y : ℝ, (a * x^2 - (a + 3) * x + 2 = 0) ∧ (a * y^2 - (a + 3) * y + 2 = 0) ∧ (x * y < 0)) ↔ (a < 0) :=
by
  sorry

end NUMINAMATH_GPT_two_real_roots_opposite_signs_l1670_167056


namespace NUMINAMATH_GPT_polynomial_root_theorem_l1670_167096

theorem polynomial_root_theorem
  (α β γ δ p q : ℝ)
  (h₁ : α + β = -p)
  (h₂ : α * β = 1)
  (h₃ : γ + δ = -q)
  (h₄ : γ * δ = 1) :
  (α - γ) * (β - γ) * (α + δ) * (β + δ) = q^2 - p^2 :=
by
  sorry

end NUMINAMATH_GPT_polynomial_root_theorem_l1670_167096


namespace NUMINAMATH_GPT_div_by_73_l1670_167042

theorem div_by_73 (n : ℕ) (h : 0 < n) : (2^(3*n + 6) + 3^(4*n + 2)) % 73 = 0 := sorry

end NUMINAMATH_GPT_div_by_73_l1670_167042


namespace NUMINAMATH_GPT_sin_squared_not_periodic_l1670_167078

noncomputable def sin_squared (x : ℝ) : ℝ := Real.sin (x^2)

theorem sin_squared_not_periodic : 
  ¬ (∃ T > 0, ∀ x ∈ Set.univ, sin_squared (x + T) = sin_squared x) := 
sorry

end NUMINAMATH_GPT_sin_squared_not_periodic_l1670_167078


namespace NUMINAMATH_GPT_carol_age_difference_l1670_167032

theorem carol_age_difference (bob_age carol_age : ℕ) (h1 : bob_age + carol_age = 66)
  (h2 : carol_age = 3 * bob_age + 2) (h3 : bob_age = 16) (h4 : carol_age = 50) :
  carol_age - 3 * bob_age = 2 :=
by
  sorry

end NUMINAMATH_GPT_carol_age_difference_l1670_167032


namespace NUMINAMATH_GPT_power_of_two_with_nines_l1670_167027

theorem power_of_two_with_nines (k : ℕ) (h : k > 1) :
  ∃ (n : ℕ), (2^n % 10^k) / 10^((10 * 5^k + k + 2 - k) / 2) = 9 :=
sorry

end NUMINAMATH_GPT_power_of_two_with_nines_l1670_167027


namespace NUMINAMATH_GPT_base_number_is_two_l1670_167091

theorem base_number_is_two (a : ℝ) (x : ℕ) (h1 : x = 14) (h2 : a^x - a^(x - 2) = 3 * a^12) : a = 2 := by
  sorry

end NUMINAMATH_GPT_base_number_is_two_l1670_167091


namespace NUMINAMATH_GPT_fractions_expressible_iff_prime_l1670_167066

noncomputable def is_good_fraction (a b n : ℕ) : Prop :=
  a > 0 ∧ b > 0 ∧ a + b = n

theorem fractions_expressible_iff_prime (n : ℕ) (hn : n > 1) :
  (∀ (a b : ℕ), b < n → ∃ (k l : ℤ), k * a + l * n = b) ↔ Prime n :=
sorry

end NUMINAMATH_GPT_fractions_expressible_iff_prime_l1670_167066


namespace NUMINAMATH_GPT_DansAgeCalculation_l1670_167016

theorem DansAgeCalculation (D x : ℕ) (h1 : D = 8) (h2 : D + 20 = 7 * (D - x)) : x = 4 :=
by
  sorry

end NUMINAMATH_GPT_DansAgeCalculation_l1670_167016


namespace NUMINAMATH_GPT_saree_final_sale_price_in_inr_l1670_167067

noncomputable def finalSalePrice (initialPrice: ℝ) (discounts: List ℝ) (conversionRate: ℝ) : ℝ :=
  let finalUSDPrice := discounts.foldl (fun acc discount => acc * (1 - discount)) initialPrice
  finalUSDPrice * conversionRate

theorem saree_final_sale_price_in_inr
  (initialPrice : ℝ := 150)
  (discounts : List ℝ := [0.20, 0.15, 0.05])
  (conversionRate : ℝ := 75)
  : finalSalePrice initialPrice discounts conversionRate = 7267.5 :=
by
  sorry

end NUMINAMATH_GPT_saree_final_sale_price_in_inr_l1670_167067


namespace NUMINAMATH_GPT_solve_quadratic_equation_l1670_167011

theorem solve_quadratic_equation (x : ℝ) : x^2 = 100 → x = -10 ∨ x = 10 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_solve_quadratic_equation_l1670_167011


namespace NUMINAMATH_GPT_range_of_a_l1670_167099

theorem range_of_a :
  ∀ a : ℝ, (∃ x : ℝ, 0 ≤ x ∧ x ≤ 1 ∧ x^2 + (1 - a) * x + 3 - a > 0) ↔ a < 3 := 
sorry

end NUMINAMATH_GPT_range_of_a_l1670_167099


namespace NUMINAMATH_GPT_lines_in_n_by_n_grid_l1670_167036

def num_horizontal_lines (n : ℕ) : ℕ := n + 1
def num_vertical_lines (n : ℕ) : ℕ := n + 1
def total_lines (n : ℕ) : ℕ := num_horizontal_lines n + num_vertical_lines n

theorem lines_in_n_by_n_grid (n : ℕ) :
  total_lines n = 2 * (n + 1) := by
  sorry

end NUMINAMATH_GPT_lines_in_n_by_n_grid_l1670_167036


namespace NUMINAMATH_GPT_number_of_sheets_in_stack_l1670_167029

theorem number_of_sheets_in_stack (n : ℕ) (h1 : 2 * n + 2 = 74) : n / 4 = 9 := 
by
  sorry

end NUMINAMATH_GPT_number_of_sheets_in_stack_l1670_167029


namespace NUMINAMATH_GPT_percentage_increase_biographies_l1670_167026

variable (B b n : ℝ)
variable (h1 : b = 0.20 * B)
variable (h2 : b + n = 0.32 * (B + n))

theorem percentage_increase_biographies (B b n : ℝ) (h1 : b = 0.20 * B) (h2 : b + n = 0.32 * (B + n)) :
  n / b * 100 = 88.24 := by
  sorry

end NUMINAMATH_GPT_percentage_increase_biographies_l1670_167026


namespace NUMINAMATH_GPT_range_of_a_l1670_167051

theorem range_of_a (a : ℝ) : (∃ x : ℝ, x^2 - a * x + 1 < 0) ↔ (a > 2 ∨ a < -2) :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_l1670_167051


namespace NUMINAMATH_GPT_right_triangle_inequality_l1670_167093

theorem right_triangle_inequality {a b c : ℝ} (h : c^2 = a^2 + b^2) : 
  a + b ≤ c * Real.sqrt 2 :=
sorry

end NUMINAMATH_GPT_right_triangle_inequality_l1670_167093


namespace NUMINAMATH_GPT_largest_lcm_among_pairs_is_45_l1670_167003

theorem largest_lcm_among_pairs_is_45 :
  max (max (max (max (max (Nat.lcm 15 3) (Nat.lcm 15 5)) (Nat.lcm 15 6)) (Nat.lcm 15 9)) (Nat.lcm 15 10)) (Nat.lcm 15 15) = 45 :=
by
  sorry

end NUMINAMATH_GPT_largest_lcm_among_pairs_is_45_l1670_167003


namespace NUMINAMATH_GPT_A_beats_B_by_63_l1670_167015

variable (A B C : ℕ)

-- Condition: A beats C by 163 meters
def A_beats_C : Prop := A = 1000 - 163
-- Condition: B beats C by 100 meters
def B_beats_C (X : ℕ) : Prop := 1000 - X = 837 + 100
-- Main theorem statement
theorem A_beats_B_by_63 (X : ℕ) (h1 : A_beats_C A) (h2 : B_beats_C X): X = 63 :=
by
  sorry

end NUMINAMATH_GPT_A_beats_B_by_63_l1670_167015


namespace NUMINAMATH_GPT_no_first_quadrant_l1670_167030

theorem no_first_quadrant (a b : ℝ) (h_a : a < 0) (h_b : b < 0) (h_am : (a - b) < 0) :
  ¬∃ x : ℝ, (a - b) * x + b > 0 ∧ x > 0 :=
sorry

end NUMINAMATH_GPT_no_first_quadrant_l1670_167030


namespace NUMINAMATH_GPT_joan_games_l1670_167008

theorem joan_games (games_this_year games_total games_last_year : ℕ) 
  (h1 : games_this_year = 4) 
  (h2 : games_total = 9) 
  (h3 : games_total = games_this_year + games_last_year) :
  games_last_year = 5 :=
by {
  -- The proof goes here
  sorry
}

end NUMINAMATH_GPT_joan_games_l1670_167008


namespace NUMINAMATH_GPT_set_equality_proof_l1670_167087

theorem set_equality_proof :
  (∃ (u : ℤ), ∃ (m n l : ℤ), u = 12 * m + 8 * n + 4 * l) ↔
  (∃ (u : ℤ), ∃ (p q r : ℤ), u = 20 * p + 16 * q + 12 * r) :=
sorry

end NUMINAMATH_GPT_set_equality_proof_l1670_167087


namespace NUMINAMATH_GPT_arithmetic_sequence_sum_first_nine_terms_l1670_167047

variable (a_n : ℕ → ℤ)
variable (S_n : ℕ → ℤ)
variable (d : ℤ)

-- The sequence {a_n} is an arithmetic sequence.
def arithmetic_sequence := ∀ n : ℕ, a_n (n + 1) = a_n n + d

-- The sum of the first n terms of the sequence.
def sum_first_n_terms := ∀ n : ℕ, S_n n = (n * (a_n 1 + a_n n)) / 2

-- Given condition: a_2 = 3 * a_4 - 6
def given_condition := a_n 2 = 3 * a_n 4 - 6

-- The main theorem to prove S_9 = 27
theorem arithmetic_sequence_sum_first_nine_terms (h_arith : arithmetic_sequence a_n d) (h_sum : sum_first_n_terms a_n S_n) (h_condition : given_condition a_n) : 
  S_n 9 = 27 := 
by
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_sum_first_nine_terms_l1670_167047


namespace NUMINAMATH_GPT_exists_collinear_B_points_l1670_167034

noncomputable def intersection (A B C D : Point) : Point :=
sorry

noncomputable def collinearity (P Q R S T : Point) : Prop :=
sorry

def convex_pentagon (A1 A2 A3 A4 A5 : Point) : Prop :=
-- Condition ensuring A1, A2, A3, A4, A5 form a convex pentagon, to be precisely defined
sorry

theorem exists_collinear_B_points :
  ∃ (A1 A2 A3 A4 A5 : Point),
    convex_pentagon A1 A2 A3 A4 A5 ∧
    collinearity
      (intersection A1 A4 A2 A3)
      (intersection A2 A5 A3 A4)
      (intersection A3 A1 A4 A5)
      (intersection A4 A2 A5 A1)
      (intersection A5 A3 A1 A2) :=
sorry

end NUMINAMATH_GPT_exists_collinear_B_points_l1670_167034


namespace NUMINAMATH_GPT_pyramid_height_is_correct_l1670_167021

noncomputable def pyramid_height (perimeter : ℝ) (apex_distance : ℝ) : ℝ :=
  let side_length := perimeter / 4
  let half_diagonal := side_length * Real.sqrt 2 / 2
  Real.sqrt (apex_distance ^ 2 - half_diagonal ^ 2)

theorem pyramid_height_is_correct :
  pyramid_height 40 15 = 5 * Real.sqrt 7 :=
by
  sorry

end NUMINAMATH_GPT_pyramid_height_is_correct_l1670_167021


namespace NUMINAMATH_GPT_students_going_on_field_trip_l1670_167071

-- Define conditions
def van_capacity : Nat := 7
def number_of_vans : Nat := 6
def number_of_adults : Nat := 9

-- Define the total capacity
def total_people_capacity : Nat := number_of_vans * van_capacity

-- Define the number of students
def number_of_students : Nat := total_people_capacity - number_of_adults

-- Prove the number of students is 33
theorem students_going_on_field_trip : number_of_students = 33 := by
  sorry

end NUMINAMATH_GPT_students_going_on_field_trip_l1670_167071


namespace NUMINAMATH_GPT_problem_statement_l1670_167031

def f (x : ℝ) : ℝ := x^3 + 1
def g (x : ℝ) : ℝ := 3 * x - 2

theorem problem_statement : f (g (f (g 2))) = 7189058 := by
  sorry

end NUMINAMATH_GPT_problem_statement_l1670_167031


namespace NUMINAMATH_GPT_required_cups_of_sugar_l1670_167033

-- Define the original ratios
def original_flour_water_sugar_ratio : Rat := 10 / 6 / 3
def new_flour_water_ratio : Rat := 2 * (10 / 6)
def new_flour_sugar_ratio : Rat := (1 / 2) * (10 / 3)

-- Given conditions
def cups_of_water : Rat := 2

-- Problem statement: prove the amount of sugar required
theorem required_cups_of_sugar : ∀ (sugar_cups : Rat),
  original_flour_water_sugar_ratio = 10 / 6 / 3 ∧
  new_flour_water_ratio = 2 * (10 / 6) ∧
  new_flour_sugar_ratio = (1 / 2) * (10 / 3) ∧
  cups_of_water = 2 ∧
  (6 / 12) = (2 / sugar_cups) → sugar_cups = 4 := by
  intro sugar_cups
  sorry

end NUMINAMATH_GPT_required_cups_of_sugar_l1670_167033


namespace NUMINAMATH_GPT_at_least_5_limit_ups_needed_l1670_167004

-- Let's denote the necessary conditions in Lean
variable (a : ℝ) -- the buying price of stock A

-- Initial price after 4 consecutive limit downs
def price_after_limit_downs (a : ℝ) : ℝ := a * (1 - 0.1) ^ 4

-- Condition of no loss after certain limit ups
def no_loss_after_limit_ups (a : ℝ) (x : ℕ) : Prop := 
  price_after_limit_downs a * (1 + 0.1)^x ≥ a
  
theorem at_least_5_limit_ups_needed (a : ℝ) : ∃ x, no_loss_after_limit_ups a x ∧ x ≥ 5 :=
by
  -- We are required to find such x and prove the condition, which has been shown in the mathematical solution
  sorry

end NUMINAMATH_GPT_at_least_5_limit_ups_needed_l1670_167004


namespace NUMINAMATH_GPT_find_digit_for_multiple_of_3_l1670_167097

theorem find_digit_for_multiple_of_3 (d : ℕ) (h : d < 10) : 
  (56780 + d) % 3 = 0 ↔ d = 1 :=
by sorry

end NUMINAMATH_GPT_find_digit_for_multiple_of_3_l1670_167097


namespace NUMINAMATH_GPT_Bruce_Anne_combined_cleaning_time_l1670_167012

-- Define the conditions
def Anne_clean_time : ℕ := 12
def Anne_speed_doubled_time : ℕ := 3
def Bruce_clean_time : ℕ := 6
def Combined_time_with_doubled_speed : ℚ := 1 / 3
def Combined_time_current_speed : ℚ := 1 / 4

-- Prove the problem statement
theorem Bruce_Anne_combined_cleaning_time : 
  (Anne_clean_time = 12) ∧ 
  ((1 / Bruce_clean_time + 1 / 6) = Combined_time_with_doubled_speed) →
  (1 / Combined_time_current_speed) = 4 := 
by
  intro h1
  sorry

end NUMINAMATH_GPT_Bruce_Anne_combined_cleaning_time_l1670_167012


namespace NUMINAMATH_GPT_A_square_or_cube_neg_identity_l1670_167092

open Matrix

theorem A_square_or_cube_neg_identity (A : Matrix (Fin 2) (Fin 2) ℚ)
  (n : ℕ) (hn_nonzero : n ≠ 0) (hA_pow_n : A ^ n = -(1 : Matrix (Fin 2) (Fin 2) ℚ)) :
  A ^ 2 = -(1 : Matrix (Fin 2) (Fin 2) ℚ) ∨ A ^ 3 = -(1 : Matrix (Fin 2) (Fin 2) ℚ) :=
sorry

end NUMINAMATH_GPT_A_square_or_cube_neg_identity_l1670_167092


namespace NUMINAMATH_GPT_std_deviation_above_l1670_167079

variable (mean : ℝ) (std_dev : ℝ) (score1 : ℝ) (score2 : ℝ)
variable (n1 : ℝ) (n2 : ℝ)

axiom h_mean : mean = 74
axiom h_std1 : score1 = 58
axiom h_std2 : score2 = 98
axiom h_cond1 : score1 = mean - n1 * std_dev
axiom h_cond2 : n1 = 2

theorem std_deviation_above (mean std_dev score1 score2 n1 n2 : ℝ)
  (h_mean : mean = 74)
  (h_std1 : score1 = 58)
  (h_std2 : score2 = 98)
  (h_cond1 : score1 = mean - n1 * std_dev)
  (h_cond2 : n1 = 2) :
  n2 = (score2 - mean) / std_dev :=
sorry

end NUMINAMATH_GPT_std_deviation_above_l1670_167079


namespace NUMINAMATH_GPT_problem_theorem_l1670_167048

theorem problem_theorem (x y z : ℤ) 
  (h1 : x = 10 * y + 3)
  (h2 : 2 * x = 21 * y + 1)
  (h3 : 3 * x = 5 * z + 2) : 
  11 * y - x + 7 * z = 219 := 
by
  sorry

end NUMINAMATH_GPT_problem_theorem_l1670_167048


namespace NUMINAMATH_GPT_line_tangent_to_circle_l1670_167039

open Real

theorem line_tangent_to_circle :
    ∃ (x y : ℝ), (3 * x - 4 * y - 5 = 0) ∧ ((x - 1)^2 + (y + 3)^2 - 4 = 0) ∧ 
    (∃ (t r : ℝ), (t = 0 ∧ r ≠ 0) ∧ 
     (3 * t - 4 * (r + t * 3 / 4) - 5 = 0) ∧ ((r + t * 3 / 4 - 1)^2 + (3 * (-1) + t - 3)^2 = 0)) 
  :=
sorry

end NUMINAMATH_GPT_line_tangent_to_circle_l1670_167039


namespace NUMINAMATH_GPT_tickets_bought_l1670_167017

theorem tickets_bought
  (olivia_money : ℕ) (nigel_money : ℕ) (ticket_cost : ℕ) (leftover_money : ℕ)
  (total_money : ℕ) (money_spent : ℕ) 
  (h1 : olivia_money = 112) 
  (h2 : nigel_money = 139) 
  (h3 : ticket_cost = 28) 
  (h4 : leftover_money = 83)
  (h5 : total_money = olivia_money + nigel_money)
  (h6 : total_money = 251)
  (h7 : money_spent = total_money - leftover_money)
  (h8 : money_spent = 168)
  : money_spent / ticket_cost = 6 := 
by
  sorry

end NUMINAMATH_GPT_tickets_bought_l1670_167017


namespace NUMINAMATH_GPT_time_to_cover_length_l1670_167069

-- Define the conditions
def speed_escalator : ℝ := 12
def length_escalator : ℝ := 150
def speed_person : ℝ := 3

-- State the theorem to be proved
theorem time_to_cover_length : (length_escalator / (speed_escalator + speed_person)) = 10 := by
  sorry

end NUMINAMATH_GPT_time_to_cover_length_l1670_167069


namespace NUMINAMATH_GPT_sum_of_inserted_numbers_l1670_167075

variable {x y : ℝ} -- Variables x and y are real numbers

-- Conditions
axiom geometric_sequence_condition : x^2 = 3 * y
axiom arithmetic_sequence_condition : 2 * y = x + 9

-- Goal: Prove that x + y = 45 / 4 (which is 11 1/4)
theorem sum_of_inserted_numbers : x + y = 45 / 4 :=
by
  -- Utilize axioms and conditions
  sorry

end NUMINAMATH_GPT_sum_of_inserted_numbers_l1670_167075


namespace NUMINAMATH_GPT_symmetric_circle_eq_l1670_167014

theorem symmetric_circle_eq (x y : ℝ) :
  (x + 1)^2 + (y - 1)^2 = 1 → x - y = 1 → (x - 2)^2 + (y + 2)^2 = 1 :=
by
  sorry

end NUMINAMATH_GPT_symmetric_circle_eq_l1670_167014


namespace NUMINAMATH_GPT_largest_possible_k_satisfies_triangle_condition_l1670_167059

theorem largest_possible_k_satisfies_triangle_condition :
  ∃ k : ℕ, 
    k = 2009 ∧ 
    ∀ (b r w : Fin 2009 → ℝ), 
    (∀ i : Fin 2009, i ≤ i.succ → b i ≤ b i.succ ∧ r i ≤ r i.succ ∧ w i ≤ w i.succ) → 
    (∃ (j : Fin 2009), 
      b j + r j > w j ∧ b j + w j > r j ∧ r j + w j > b j) :=
sorry

end NUMINAMATH_GPT_largest_possible_k_satisfies_triangle_condition_l1670_167059


namespace NUMINAMATH_GPT_bijective_bounded_dist_l1670_167049

open Int

theorem bijective_bounded_dist {k : ℕ} (f : ℤ → ℤ) 
    (hf_bijective : Function.Bijective f)
    (hf_property : ∀ i j : ℤ, |i - j| ≤ k → |f i - (f j)| ≤ k) :
    ∀ i j : ℤ, |f i - (f j)| = |i - j| := 
sorry

end NUMINAMATH_GPT_bijective_bounded_dist_l1670_167049


namespace NUMINAMATH_GPT_eq_of_op_star_l1670_167009

theorem eq_of_op_star (a b n : ℕ) (ha : a > 0) (hb : b > 0) (hn : n > 0) :
  (a^b^2)^n = a^(bn)^2 ↔ n = 1 := by
sorry

end NUMINAMATH_GPT_eq_of_op_star_l1670_167009


namespace NUMINAMATH_GPT_gwen_money_difference_l1670_167094

theorem gwen_money_difference:
  let money_from_grandparents : ℕ := 15
  let money_from_uncle : ℕ := 8
  money_from_grandparents - money_from_uncle = 7 :=
by
  sorry

end NUMINAMATH_GPT_gwen_money_difference_l1670_167094


namespace NUMINAMATH_GPT_sum_of_products_l1670_167002

variable (a b c : ℝ)

theorem sum_of_products (h1 : a^2 + b^2 + c^2 = 250) (h2 : a + b + c = 16) : 
  ab + bc + ca = 3 :=
sorry

end NUMINAMATH_GPT_sum_of_products_l1670_167002


namespace NUMINAMATH_GPT_cost_price_of_watch_l1670_167053

theorem cost_price_of_watch 
  (CP : ℝ)
  (h1 : 0.88 * CP = SP_loss)
  (h2 : 1.04 * CP = SP_gain)
  (h3 : SP_gain - SP_loss = 140) :
  CP = 875 := 
sorry

end NUMINAMATH_GPT_cost_price_of_watch_l1670_167053


namespace NUMINAMATH_GPT_rectangle_ratio_l1670_167046

theorem rectangle_ratio (s y x : ℝ)
  (h1 : 4 * y * x + s * s = 9 * s * s)
  (h2 : s + y + y = 3 * s)
  (h3 : y = s)
  (h4 : x + s = 3 * s) : 
  (x / y = 2) :=
sorry

end NUMINAMATH_GPT_rectangle_ratio_l1670_167046


namespace NUMINAMATH_GPT_smallest_b_for_factorization_l1670_167006

theorem smallest_b_for_factorization : ∃ (b : ℕ), (∀ p q : ℤ, (x^2 + (b * x) + 2352) = (x + p) * (x + q) → p + q = b ∧ p * q = 2352) ∧ b = 112 := 
sorry

end NUMINAMATH_GPT_smallest_b_for_factorization_l1670_167006


namespace NUMINAMATH_GPT_simple_interest_calculation_l1670_167077

-- Define the principal (P), rate (R), and time (T)
def principal : ℝ := 10000
def rate : ℝ := 0.08
def time : ℝ := 1

-- Define the simple interest formula
def simple_interest (P R T : ℝ) : ℝ := P * R * T

-- The theorem to be proved
theorem simple_interest_calculation : simple_interest principal rate time = 800 :=
by
  -- Proof steps would go here, but this is left as an exercise
  sorry

end NUMINAMATH_GPT_simple_interest_calculation_l1670_167077


namespace NUMINAMATH_GPT_sampling_interval_l1670_167025

theorem sampling_interval 
  (total_population : ℕ) 
  (individuals_removed : ℕ) 
  (population_after_removal : ℕ)
  (sampling_interval : ℕ) :
  total_population = 102 →
  individuals_removed = 2 →
  population_after_removal = total_population - individuals_removed →
  population_after_removal = 100 →
  ∃ s : ℕ, population_after_removal % s = 0 ∧ s = 10 := 
by
  sorry

end NUMINAMATH_GPT_sampling_interval_l1670_167025


namespace NUMINAMATH_GPT_find_E_equals_2023_l1670_167024

noncomputable def proof : Prop :=
  ∃ a b c : ℝ, a ≠ b ∧ (a^2 * (b + c) = 2023) ∧ (b^2 * (c + a) = 2023) ∧ (c^2 * (a + b) = 2023)

theorem find_E_equals_2023 : proof :=
by
  sorry

end NUMINAMATH_GPT_find_E_equals_2023_l1670_167024


namespace NUMINAMATH_GPT_garden_sparrows_l1670_167082

theorem garden_sparrows (ratio_b_s : ℕ) (bluebirds sparrows : ℕ)
  (h1 : ratio_b_s = 4 / 5) (h2 : bluebirds = 28) :
  sparrows = 35 :=
  sorry

end NUMINAMATH_GPT_garden_sparrows_l1670_167082


namespace NUMINAMATH_GPT_simplify_expression_l1670_167068

theorem simplify_expression (x : ℝ) : (x + 3) * (x - 3) = x^2 - 9 :=
by
  -- We acknowledge this is the placeholder for the proof.
  -- This statement follows directly from the difference of squares identity.
  sorry

end NUMINAMATH_GPT_simplify_expression_l1670_167068


namespace NUMINAMATH_GPT_proof_problem_l1670_167043

open Set

noncomputable def U : Set ℝ := Icc (-5 : ℝ) 4

noncomputable def A : Set ℝ := {x : ℝ | -3 ≤ 2 * x + 1 ∧ 2 * x + 1 < 1}

noncomputable def B : Set ℝ := {x : ℝ | x^2 - 2 * x ≤ 0}

-- Definition of the complement of A in U
noncomputable def complement_U_A : Set ℝ := U \ A

-- The final proof statement
theorem proof_problem : (complement_U_A ∩ B) = Icc 0 2 :=
by
  sorry

end NUMINAMATH_GPT_proof_problem_l1670_167043


namespace NUMINAMATH_GPT_concentrate_amount_l1670_167041

def parts_concentrate : ℤ := 1
def parts_water : ℤ := 5
def part_ratio : ℤ := parts_concentrate + parts_water -- Total parts
def servings : ℤ := 375
def volume_per_serving : ℤ := 150
def total_volume : ℤ := servings * volume_per_serving -- Total volume of orange juice
def volume_per_part : ℤ := total_volume / part_ratio -- Volume per part of mixture

theorem concentrate_amount :
  volume_per_part = 9375 :=
by
  sorry

end NUMINAMATH_GPT_concentrate_amount_l1670_167041


namespace NUMINAMATH_GPT_minimum_dot_product_l1670_167028

-- Define point coordinates
structure Point where
  x : ℝ
  y : ℝ

-- Define points A, B, C, D according to the given problem statement
def A : Point := ⟨0, 0⟩
def B : Point := ⟨1, 0⟩
def C : Point := ⟨1, 2⟩
def D : Point := ⟨0, 2⟩

-- Define the condition for points E and F on the sides BC and CD respectively.
def isOnBC (E : Point) : Prop := E.x = 1 ∧ 0 ≤ E.y ∧ E.y ≤ 2
def isOnCD (F : Point) : Prop := F.y = 2 ∧ 0 ≤ F.x ∧ F.x ≤ 1

-- Define the distance constraint for |EF| = 1
def distEF (E F : Point) : Prop :=
  (F.x - E.x)^2 + (F.y - E.y)^2 = 1

-- Define the dot product between vectors AE and AF
def dotProductAEAF (E F : Point) : ℝ :=
  2 * E.y + F.x

-- Main theorem to prove the minimum dot product value
theorem minimum_dot_product (E F : Point) (hE : isOnBC E) (hF : isOnCD F) (hDistEF : distEF E F) :
  dotProductAEAF E F = 5 - Real.sqrt 5 :=
  sorry

end NUMINAMATH_GPT_minimum_dot_product_l1670_167028


namespace NUMINAMATH_GPT_prime_sum_of_composites_l1670_167022

def is_composite (n : ℕ) : Prop := ∃ m k : ℕ, m > 1 ∧ k > 1 ∧ m * k = n
def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n
def can_be_expressed_as_sum_of_two_composites (p : ℕ) : Prop :=
  ∃ a b : ℕ, is_composite a ∧ is_composite b ∧ p = a + b

theorem prime_sum_of_composites :
  can_be_expressed_as_sum_of_two_composites 13 ∧ 
  ∀ p : ℕ, is_prime p ∧ p > 13 → can_be_expressed_as_sum_of_two_composites p :=
by 
  sorry

end NUMINAMATH_GPT_prime_sum_of_composites_l1670_167022


namespace NUMINAMATH_GPT_michael_total_cost_l1670_167055

def peach_pies : ℕ := 5
def apple_pies : ℕ := 4
def blueberry_pies : ℕ := 3

def pounds_per_pie : ℕ := 3

def price_per_pound_peaches : ℝ := 2.0
def price_per_pound_apples : ℝ := 1.0
def price_per_pound_blueberries : ℝ := 1.0

def total_peach_pounds : ℕ := peach_pies * pounds_per_pie
def total_apple_pounds : ℕ := apple_pies * pounds_per_pie
def total_blueberry_pounds : ℕ := blueberry_pies * pounds_per_pie

def cost_peaches : ℝ := total_peach_pounds * price_per_pound_peaches
def cost_apples : ℝ := total_apple_pounds * price_per_pound_apples
def cost_blueberries : ℝ := total_blueberry_pounds * price_per_pound_blueberries

def total_cost : ℝ := cost_peaches + cost_apples + cost_blueberries

theorem michael_total_cost :
  total_cost = 51.0 := by
  sorry

end NUMINAMATH_GPT_michael_total_cost_l1670_167055


namespace NUMINAMATH_GPT_smallest_n_interesting_meeting_l1670_167076

theorem smallest_n_interesting_meeting (m : ℕ) (hm : 2 ≤ m) :
  ∀ (n : ℕ), (n ≤ 3 * m - 1) ∧ (∀ (rep : Finset (Fin (3 * m))), rep.card = n →
  ∃ subrep : Finset (Fin (3 * m)), subrep.card = 3 ∧ ∀ (x y : Fin (3 * m)), x ∈ subrep → y ∈ subrep → x ≠ y → ∃ z : Fin (3 * m), z ∈ subrep ∧ z = x + y) → n = 2 * m + 1 := by
  sorry

end NUMINAMATH_GPT_smallest_n_interesting_meeting_l1670_167076


namespace NUMINAMATH_GPT_unique_painted_cube_l1670_167040

/-- Determine the number of distinct ways to paint a cube where:
  - One side is yellow,
  - Two sides are purple,
  - Three sides are orange.
  Taking into account that two cubes are considered identical if they can be rotated to match. -/
theorem unique_painted_cube :
  ∃ unique n : ℕ, n = 1 ∧
    (∃ (c : Fin 6 → Fin 3), 
      (∃ (i : Fin 6), c i = 0) ∧ 
      (∃ (j k : Fin 6), j ≠ k ∧ c j = 1 ∧ c k = 1) ∧ 
      (∃ (m p q : Fin 6), m ≠ p ∧ m ≠ q ∧ p ≠ q ∧ c m = 2 ∧ c p = 2 ∧ c q = 2)
    ) :=
sorry

end NUMINAMATH_GPT_unique_painted_cube_l1670_167040


namespace NUMINAMATH_GPT_find_x_value_l1670_167062

noncomputable def floor_plus_2x_eq_33 (x : ℝ) : Prop :=
  ∃ n : ℤ, ⌊x⌋ = n ∧ n + 2 * x = 33 ∧  (0 : ℝ) ≤ x - n ∧ x - n < 1

theorem find_x_value : ∀ x : ℝ, floor_plus_2x_eq_33 x → x = 11 :=
by
  intro x
  intro h
  -- Proof skipped, included as 'sorry' to compile successfully.
  sorry

end NUMINAMATH_GPT_find_x_value_l1670_167062


namespace NUMINAMATH_GPT_case1_equiv_case2_equiv_determine_case_l1670_167065

theorem case1_equiv (a c x : ℝ) (hc : c ≠ 0) (hx : x ≠ 0) : 
  ((x + a) / (x + c) = a / c) ↔ (a = c) :=
by sorry

theorem case2_equiv (b d : ℝ) (hb : b ≠ 0) (hd : d ≠ 0) : 
  (b / d = b / d) :=
by sorry

theorem determine_case (a b c d x : ℝ) (hc : c ≠ 0) (hx : x ≠ 0) (hb : b ≠ 0) (hd : d ≠ 0) :
  ¬((x + a) / (x + c) = a / c) ∧ (b / d = b / d) :=
by sorry

end NUMINAMATH_GPT_case1_equiv_case2_equiv_determine_case_l1670_167065


namespace NUMINAMATH_GPT_polynomial_factorization_l1670_167084

theorem polynomial_factorization (x : ℝ) :
  x^6 + 6*x^5 + 15*x^4 + 20*x^3 + 15*x^2 + 6*x + 1 = (x + 1)^6 :=
by {
  -- proof goes here
  sorry
}

end NUMINAMATH_GPT_polynomial_factorization_l1670_167084
