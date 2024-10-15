import Mathlib

namespace NUMINAMATH_GPT_geometric_sequence_general_term_and_arithmetic_sequence_max_sum_l1675_167511

theorem geometric_sequence_general_term_and_arithmetic_sequence_max_sum :
  (∃ a_n : ℕ → ℕ, ∃ b_n : ℕ → ℤ, ∃ T_n : ℕ → ℤ,
    (∀ n, a_n n = 2^(n-1)) ∧
    (a_n 1 + a_n 2 = 3) ∧
    (b_n 2 = a_n 3) ∧
    (b_n 3 = -b_n 5) ∧
    (∀ n, T_n n = n * (b_n 1 + b_n n) / 2) ∧
    (T_n 3 = 12) ∧
    (T_n 4 = 12)) :=
by
  sorry

end NUMINAMATH_GPT_geometric_sequence_general_term_and_arithmetic_sequence_max_sum_l1675_167511


namespace NUMINAMATH_GPT_alpha_values_perpendicular_l1675_167578

theorem alpha_values_perpendicular
  (α : ℝ)
  (h1 : α ∈ Set.Ico 0 (2 * Real.pi))
  (h2 : ∀ (x y : ℝ), x * Real.cos α - y - 1 = 0 → x + y * Real.sin α + 1 = 0 → false):
  α = Real.pi / 4 ∨ α = 5 * Real.pi / 4 :=
by
  sorry

end NUMINAMATH_GPT_alpha_values_perpendicular_l1675_167578


namespace NUMINAMATH_GPT_gcd_117_182_evaluate_polynomial_l1675_167546

-- Problem 1: Prove that GCD of 117 and 182 is 13
theorem gcd_117_182 : Int.gcd 117 182 = 13 := 
by
  sorry

-- Problem 2: Prove that evaluating the polynomial at x = -1 results in 12
noncomputable def f : ℤ → ℤ := λ x => 1 - 9 * x + 8 * x^2 - 4 * x^4 + 5 * x^5 + 3 * x^6

theorem evaluate_polynomial : f (-1) = 12 := 
by
  sorry

end NUMINAMATH_GPT_gcd_117_182_evaluate_polynomial_l1675_167546


namespace NUMINAMATH_GPT_brick_surface_area_l1675_167518

theorem brick_surface_area (l w h : ℝ) (hl : l = 10) (hw : w = 4) (hh : h = 3) : 
  2 * (l * w + l * h + w * h) = 164 := 
by
  sorry

end NUMINAMATH_GPT_brick_surface_area_l1675_167518


namespace NUMINAMATH_GPT_geoboard_quadrilaterals_l1675_167506

-- Definitions of the quadrilaterals as required by the conditions of the problem.
def quadrilateral_area (quad : Type) : ℝ := sorry
def quadrilateral_perimeter (quad : Type) : ℝ := sorry

-- Declaration of Quadrilateral I and II on a geoboard.
def quadrilateral_i : Type := sorry
def quadrilateral_ii : Type := sorry

-- The proof problem statement.
theorem geoboard_quadrilaterals :
  quadrilateral_area quadrilateral_i = quadrilateral_area quadrilateral_ii ∧
  quadrilateral_perimeter quadrilateral_i < quadrilateral_perimeter quadrilateral_ii := by
  sorry

end NUMINAMATH_GPT_geoboard_quadrilaterals_l1675_167506


namespace NUMINAMATH_GPT_solution_set_eq_l1675_167551

def odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = -f (x)

def decreasing_condition (f : ℝ → ℝ) : Prop :=
  ∀ x1 x2 : ℝ, x1 < 0 → x2 < 0 → x1 ≠ x2 → (x1 * f (x1) - x2 * f (x2)) / (x1 - x2) < 0

variable (f : ℝ → ℝ)
variable (h_odd : odd_function f)
variable (h_minus_2_zero : f (-2) = 0)
variable (h_decreasing : decreasing_condition f)

theorem solution_set_eq :
  {x : ℝ | f x > 0} = {x : ℝ | -2 < x ∧ x < 0} ∪ {x : ℝ | 2 < x} :=
sorry

end NUMINAMATH_GPT_solution_set_eq_l1675_167551


namespace NUMINAMATH_GPT_find_x_l1675_167520

theorem find_x (x y z w : ℕ) (h1 : x = y + 8) (h2 : y = z + 15) (h3 : z = w + 25) (h4 : w = 90) : x = 138 :=
by
  sorry

end NUMINAMATH_GPT_find_x_l1675_167520


namespace NUMINAMATH_GPT_solve_equation_l1675_167516

theorem solve_equation (x : ℝ) (h : x ≠ 0 ∧ x ≠ -1) : (x / (x + 1) = 1 + (1 / x)) ↔ (x = -1 / 2) :=
by
  sorry

end NUMINAMATH_GPT_solve_equation_l1675_167516


namespace NUMINAMATH_GPT_determine_time_Toronto_l1675_167589

noncomputable def timeDifferenceBeijingToronto: ℤ := -12

def timeBeijing: ℕ × ℕ := (1, 8) -- (day, hour) format for simplicity: October 1st, 8:00

def timeToronto: ℕ × ℕ := (30, 20) -- Expected result in (day, hour): September 30th, 20:00

theorem determine_time_Toronto :
  timeDifferenceBeijingToronto = -12 →
  timeBeijing = (1, 8) →
  timeToronto = (30, 20) :=
by
  -- proof to be written 
  sorry

end NUMINAMATH_GPT_determine_time_Toronto_l1675_167589


namespace NUMINAMATH_GPT_lateral_surface_area_of_cone_l1675_167510

open Real

theorem lateral_surface_area_of_cone
  (SA : ℝ) (SB : ℝ)
  (cos_angle_SA_SB : ℝ) (angle_SA_base : ℝ)
  (area_SAB : ℝ) :
  cos_angle_SA_SB = 7 / 8 →
  angle_SA_base = π / 4 →
  area_SAB = 5 * sqrt 15 →
  SA = 4 * sqrt 5 →
  SB = SA →
  (1/2) * (sqrt 2 / 2 * SA) * (2 * π * SA) = 40 * sqrt 2 * π :=
sorry

end NUMINAMATH_GPT_lateral_surface_area_of_cone_l1675_167510


namespace NUMINAMATH_GPT_not_necessarily_divisor_l1675_167534

def consecutive_product (k : ℤ) : ℤ := k * (k + 1) * (k + 2) * (k + 3)

theorem not_necessarily_divisor (k : ℤ) (hk : 8 ∣ consecutive_product k) : ¬ (48 ∣ consecutive_product k) :=
sorry

end NUMINAMATH_GPT_not_necessarily_divisor_l1675_167534


namespace NUMINAMATH_GPT_number_of_cats_adopted_l1675_167536

theorem number_of_cats_adopted (c : ℕ) 
  (h1 : 50 * c + 3 * 100 + 2 * 150 = 700) :
  c = 2 :=
by
  sorry

end NUMINAMATH_GPT_number_of_cats_adopted_l1675_167536


namespace NUMINAMATH_GPT_part1_ABC_inquality_part2_ABCD_inquality_l1675_167533

theorem part1_ABC_inquality (a b c ABC : ℝ) : 
  (ABC <= (a^2 + b^2) / 4) -> 
  (ABC <= (b^2 + c^2) / 4) -> 
  (ABC <= (a^2 + c^2) / 4) -> 
    (ABC < (a^2 + b^2 + c^2) / 6) :=
sorry

theorem part2_ABCD_inquality (a b c d ABC BCD CDA DAB ABCD : ℝ) :
  (ABCD = 1/2 * ((ABC) + (BCD) + (CDA) + (DAB))) -> 
  (ABC < (a^2 + b^2 + c^2) / 6) -> 
  (BCD < (b^2 + c^2 + d^2) / 6) -> 
  (CDA < (c^2 + d^2 + a^2) / 6) -> 
  (DAB < (d^2 + a^2 + b^2) / 6) -> 
    (ABCD < (a^2 + b^2 + c^2 + d^2) / 6) :=
sorry

end NUMINAMATH_GPT_part1_ABC_inquality_part2_ABCD_inquality_l1675_167533


namespace NUMINAMATH_GPT_subtraction_of_bases_l1675_167502

def base8_to_base10 (n : Nat) : Nat :=
  match n with
  | 0 => 0
  | _ => (n / 100) * 8^2 + ((n % 100) / 10) * 8^1 + (n % 10) * 8^0

def base7_to_base10 (n : Nat) : Nat :=
  match n with
  | 0 => 0
  | _ => (n / 100) * 7^2 + ((n % 100) / 10) * 7^1 + (n % 10) * 7^0

theorem subtraction_of_bases :
  base8_to_base10 343 - base7_to_base10 265 = 82 :=
by
  sorry

end NUMINAMATH_GPT_subtraction_of_bases_l1675_167502


namespace NUMINAMATH_GPT_maurice_earnings_l1675_167503

theorem maurice_earnings (bonus_per_10_tasks : ℕ → ℕ) (num_tasks : ℕ) (total_earnings : ℕ) :
  (∀ n, n * (bonus_per_10_tasks n) = 6 * n) →
  num_tasks = 30 →
  total_earnings = 78 →
  bonus_per_10_tasks num_tasks / 10 = 3 →
  (total_earnings - (bonus_per_10_tasks num_tasks / 10) * 6) / num_tasks = 2 :=
by
  intros h_bonus h_num_tasks h_total_earnings h_bonus_count
  sorry

end NUMINAMATH_GPT_maurice_earnings_l1675_167503


namespace NUMINAMATH_GPT_meal_total_l1675_167565

noncomputable def meal_price (appetizer entree dessert drink sales_tax tip : ℝ) : ℝ :=
  let total_before_tax := appetizer + (2 * entree) + dessert + (2 * drink)
  let tax_amount := (sales_tax / 100) * total_before_tax
  let subtotal := total_before_tax + tax_amount
  let tip_amount := (tip / 100) * subtotal
  subtotal + tip_amount

theorem meal_total : 
  meal_price 9 20 11 6.5 7.5 22 = 95.75 :=
by
  sorry

end NUMINAMATH_GPT_meal_total_l1675_167565


namespace NUMINAMATH_GPT_finite_transformation_l1675_167545

-- Define the function representing the number transformation
def transform (n : ℕ) : ℕ :=
  if n % 2 = 0 then n / 2 else n + 5

-- Define the predicate stating that the process terminates
def process_terminates (n : ℕ) : Prop :=
  ∃ k : ℕ, k > 0 ∧ transform^[k] n = 1

-- Lean 4 statement for the theorem
theorem finite_transformation (n : ℕ) (h : n > 1) : process_terminates n ↔ ¬ (∃ m : ℕ, m > 0 ∧ n = 5 * m) :=
by
  sorry

end NUMINAMATH_GPT_finite_transformation_l1675_167545


namespace NUMINAMATH_GPT_problem1_l1675_167517

theorem problem1 (x : ℝ) (n : ℕ) (h : x^n = 2) : (3 * x^n)^2 - 4 * (x^2)^n = 20 :=
by
  sorry

end NUMINAMATH_GPT_problem1_l1675_167517


namespace NUMINAMATH_GPT_tetrahedron_edge_length_of_tangent_spheres_l1675_167570

theorem tetrahedron_edge_length_of_tangent_spheres (r : ℝ) (h₁ : r = 2) :
  ∃ s : ℝ, s = 4 :=
by
  sorry

end NUMINAMATH_GPT_tetrahedron_edge_length_of_tangent_spheres_l1675_167570


namespace NUMINAMATH_GPT_average_first_18_even_numbers_l1675_167558

theorem average_first_18_even_numbers : 
  let first_even := 2
  let difference := 2
  let n := 18
  let last_even := first_even + (n - 1) * difference
  let sum := (n / 2) * (first_even + last_even)
  let average := sum / n
  average = 19 :=
by
  -- Definitions
  let first_even := 2
  let difference := 2
  let n := 18
  let last_even := first_even + (n - 1) * difference
  let sum := (n / 2) * (first_even + last_even)
  let average := sum / n
  -- The claim
  show average = 19
  sorry

end NUMINAMATH_GPT_average_first_18_even_numbers_l1675_167558


namespace NUMINAMATH_GPT_linear_eq_solution_l1675_167508

theorem linear_eq_solution (m x : ℝ) (h : |m| = 1) (h1: 1 - m ≠ 0):
  x = -(1/2) :=
sorry

end NUMINAMATH_GPT_linear_eq_solution_l1675_167508


namespace NUMINAMATH_GPT_positive_difference_is_correct_l1675_167597

/-- Angela's compounded interest parameters -/
def angela_initial_deposit : ℝ := 9000
def angela_interest_rate : ℝ := 0.08
def years : ℕ := 25

/-- Bob's simple interest parameters -/
def bob_initial_deposit : ℝ := 11000
def bob_interest_rate : ℝ := 0.09

/-- Compound interest calculation for Angela -/
def angela_balance : ℝ := angela_initial_deposit * (1 + angela_interest_rate) ^ years

/-- Simple interest calculation for Bob -/
def bob_balance : ℝ := bob_initial_deposit * (1 + bob_interest_rate * years)

/-- Difference calculation -/
def balance_difference : ℝ := angela_balance - bob_balance

/-- The positive difference between their balances to the nearest dollar -/
theorem positive_difference_is_correct :
  abs (round balance_difference) = 25890 :=
by
  sorry

end NUMINAMATH_GPT_positive_difference_is_correct_l1675_167597


namespace NUMINAMATH_GPT_area_of_fourth_square_l1675_167529

theorem area_of_fourth_square (AB BC AC CD AD : ℝ) (h_sum_ABC : AB^2 + 25 = 50)
  (h_sum_ACD : 50 + 49 = AD^2) : AD^2 = 99 :=
by
  sorry

end NUMINAMATH_GPT_area_of_fourth_square_l1675_167529


namespace NUMINAMATH_GPT_james_running_increase_l1675_167549

theorem james_running_increase (initial_miles_per_week : ℕ) (percent_increase : ℝ) (total_days : ℕ) (days_in_week : ℕ) :
  initial_miles_per_week = 100 →
  percent_increase = 0.2 →
  total_days = 280 →
  days_in_week = 7 →
  ∃ miles_per_week_to_add : ℝ, miles_per_week_to_add = 3 :=
by
  intros
  sorry

end NUMINAMATH_GPT_james_running_increase_l1675_167549


namespace NUMINAMATH_GPT_a_three_equals_35_l1675_167577

-- Define the mathematical sequences and functions
def S (n : ℕ) : ℕ := 5 * n^2 + 10 * n

def a (n : ℕ) : ℕ := S (n + 1) - S n

-- The proposition we want to prove
theorem a_three_equals_35 : a 2 = 35 := by 
  sorry

end NUMINAMATH_GPT_a_three_equals_35_l1675_167577


namespace NUMINAMATH_GPT_swim_time_CBA_l1675_167532

theorem swim_time_CBA (d t_down t_still t_upstream: ℝ) 
  (h1 : d = 1) 
  (h2 : t_down = 1 / (6 / 5))
  (h3 : t_still = 1)
  (h4 : t_upstream = (4 / 5) / 2)
  (total_time_down : (t_down + t_still) = 1)
  (total_time_up : (t_still + t_down) = 2) :
  (t_upstream * (d - (d / 5))) / 2 = 5 / 2 :=
by sorry

end NUMINAMATH_GPT_swim_time_CBA_l1675_167532


namespace NUMINAMATH_GPT_right_angled_triangle_side_length_l1675_167505

theorem right_angled_triangle_side_length :
  ∃ c : ℕ, (c = 5) ∧ (3^2 + 4^2 = c^2) ∧ (c = 4 + 1) := by
  sorry

end NUMINAMATH_GPT_right_angled_triangle_side_length_l1675_167505


namespace NUMINAMATH_GPT_neg_one_quadratic_residue_iff_l1675_167539

theorem neg_one_quadratic_residue_iff (p : ℕ) [Fact (Nat.Prime p)] (hp : p % 2 = 1) : 
  (∃ x : ℤ, x^2 ≡ -1 [ZMOD p]) ↔ p % 4 = 1 :=
sorry

end NUMINAMATH_GPT_neg_one_quadratic_residue_iff_l1675_167539


namespace NUMINAMATH_GPT_Katya_saves_enough_l1675_167575

theorem Katya_saves_enough {h c_pool_sauna x y : ℕ} (hc : h = 275) (hcs : c_pool_sauna = 250)
  (hx : x = y + 200) (heq : x + y = c_pool_sauna) : (h / (c_pool_sauna - x)) = 11 :=
by
  sorry

end NUMINAMATH_GPT_Katya_saves_enough_l1675_167575


namespace NUMINAMATH_GPT_remainder_div_3973_28_l1675_167563

theorem remainder_div_3973_28 : (3973 % 28) = 9 := by
  sorry

end NUMINAMATH_GPT_remainder_div_3973_28_l1675_167563


namespace NUMINAMATH_GPT_find_m_l1675_167507

theorem find_m (m : ℕ) (h : 10^(m-1) < 2^512 ∧ 2^512 < 10^m): 
  m = 155 :=
sorry

end NUMINAMATH_GPT_find_m_l1675_167507


namespace NUMINAMATH_GPT_cages_needed_l1675_167571

theorem cages_needed (initial_puppies sold_puppies puppies_per_cage : ℕ) (h1 : initial_puppies = 13) (h2 : sold_puppies = 7) (h3 : puppies_per_cage = 2) :
  (initial_puppies - sold_puppies) / puppies_per_cage = 3 := 
by
  sorry

end NUMINAMATH_GPT_cages_needed_l1675_167571


namespace NUMINAMATH_GPT_sequence_an_value_l1675_167535

theorem sequence_an_value (a : ℕ → ℕ) (S : ℕ → ℕ)
  (hS : ∀ n, 4 * S n = (a n - 1) * (a n + 3))
  (h_pos : ∀ n, 0 < a n)
  (n_nondec : ∀ n, a (n + 1) - a n = 2) :
  a 1005 = 2011 := 
sorry

end NUMINAMATH_GPT_sequence_an_value_l1675_167535


namespace NUMINAMATH_GPT_even_function_derivative_l1675_167586

theorem even_function_derivative (f : ℝ → ℝ)
  (h_even : ∀ x, f (-x) = f x)
  (h_deriv_pos : ∀ x > 0, deriv f x = (x - 1) * (x - 2)) : f (-2) < f 1 :=
sorry

end NUMINAMATH_GPT_even_function_derivative_l1675_167586


namespace NUMINAMATH_GPT_LaKeisha_needs_to_mow_more_sqft_l1675_167541

noncomputable def LaKeisha_price_per_sqft : ℝ := 0.10
noncomputable def LaKeisha_book_cost : ℝ := 150
noncomputable def LaKeisha_mowed_sqft : ℕ := 3 * 20 * 15
noncomputable def LaKeisha_earnings_so_far : ℝ := LaKeisha_mowed_sqft * LaKeisha_price_per_sqft

theorem LaKeisha_needs_to_mow_more_sqft (additional_sqft_needed : ℝ) :
  additional_sqft_needed = (LaKeisha_book_cost - LaKeisha_earnings_so_far) / LaKeisha_price_per_sqft → 
  additional_sqft_needed = 600 :=
by
  sorry

end NUMINAMATH_GPT_LaKeisha_needs_to_mow_more_sqft_l1675_167541


namespace NUMINAMATH_GPT_square_of_fourth_power_of_fourth_smallest_prime_l1675_167599

-- Define the fourth smallest prime number
def fourth_smallest_prime : ℕ := 7

-- Define the square of the fourth power of that number
def square_of_fourth_power (n : ℕ) : ℕ := (n^4)^2

-- Prove the main statement
theorem square_of_fourth_power_of_fourth_smallest_prime : square_of_fourth_power fourth_smallest_prime = 5764801 :=
by
  sorry

end NUMINAMATH_GPT_square_of_fourth_power_of_fourth_smallest_prime_l1675_167599


namespace NUMINAMATH_GPT_even_function_order_l1675_167554

noncomputable def f (m : ℝ) (x : ℝ) := (m - 1) * x^2 + 6 * m * x + 2

theorem even_function_order (m : ℝ) (h_even : ∀ x : ℝ, f m (-x) = f m x) : 
  m = 0 ∧ f m (-2) < f m 1 ∧ f m 1 < f m 0 := by
sorry

end NUMINAMATH_GPT_even_function_order_l1675_167554


namespace NUMINAMATH_GPT_intersection_point_of_planes_l1675_167540

theorem intersection_point_of_planes :
  ∃ (x y z : ℚ), 
    3 * x - y + 4 * z = 2 ∧ 
    -3 * x + 4 * y - 3 * z = 4 ∧ 
    -x + y - z = 5 ∧ 
    x = -55 ∧ 
    y = -11 ∧ 
    z = 39 := 
by
  sorry

end NUMINAMATH_GPT_intersection_point_of_planes_l1675_167540


namespace NUMINAMATH_GPT_probability_computation_l1675_167552

noncomputable def probability_inside_sphere : ℝ :=
  let volume_of_cube : ℝ := 64
  let volume_of_sphere : ℝ := (4/3) * Real.pi * (2^3)
  volume_of_sphere / volume_of_cube

theorem probability_computation :
  probability_inside_sphere = Real.pi / 6 :=
by
  sorry

end NUMINAMATH_GPT_probability_computation_l1675_167552


namespace NUMINAMATH_GPT_arithmetic_seq_b3_b6_l1675_167566

theorem arithmetic_seq_b3_b6 (b : ℕ → ℕ) (d : ℕ) 
  (h_seq : ∀ n, b n = b 1 + n * d)
  (h_increasing : ∀ n, b (n + 1) > b n)
  (h_b4_b5 : b 4 * b 5 = 30) :
  b 3 * b 6 = 28 := 
sorry

end NUMINAMATH_GPT_arithmetic_seq_b3_b6_l1675_167566


namespace NUMINAMATH_GPT_sine_tangent_not_possible_1_sine_tangent_not_possible_2_l1675_167527

theorem sine_tangent_not_possible_1 : 
  ¬ (∃ θ : ℝ, Real.sin θ = 0.27413 ∧ Real.tan θ = 0.25719) :=
sorry

theorem sine_tangent_not_possible_2 : 
  ¬ (∃ θ : ℝ, Real.sin θ = 0.25719 ∧ Real.tan θ = 0.27413) :=
sorry

end NUMINAMATH_GPT_sine_tangent_not_possible_1_sine_tangent_not_possible_2_l1675_167527


namespace NUMINAMATH_GPT_gcd_of_987654_and_123456_l1675_167515

theorem gcd_of_987654_and_123456 : Nat.gcd 987654 123456 = 6 := by
  sorry

end NUMINAMATH_GPT_gcd_of_987654_and_123456_l1675_167515


namespace NUMINAMATH_GPT_min_AB_CD_value_l1675_167512

def vector := (ℝ × ℝ)

def dot_product (v1 v2 : vector) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2

def AB_CD (AC BD CB : vector) : ℝ :=
  let AB := (CB.1 + AC.1, CB.2 + AC.2)
  let CD := (CB.1 + BD.1, CB.2 + BD.2)
  dot_product AB CD

theorem min_AB_CD_value : ∀ (AC BD : vector), AC = (1, 2) → BD = (-2, 2) → 
  ∃ CB : vector, AB_CD AC BD CB = -9 / 4 :=
by
  intros AC BD hAC hBD
  sorry

end NUMINAMATH_GPT_min_AB_CD_value_l1675_167512


namespace NUMINAMATH_GPT_Jacob_age_is_3_l1675_167591

def Phoebe_age : ℕ := sorry
def Rehana_age : ℕ := 25
def Jacob_age (P : ℕ) : ℕ := 3 * P / 5

theorem Jacob_age_is_3 (P : ℕ) (h1 : Rehana_age + 5 = 3 * (P + 5)) (h2 : Rehana_age = 25) (h3 : Jacob_age P = 3) : Jacob_age P = 3 := by {
  sorry
}

end NUMINAMATH_GPT_Jacob_age_is_3_l1675_167591


namespace NUMINAMATH_GPT_kara_water_intake_l1675_167576

-- Definitions based on the conditions
def daily_doses := 3
def week1_days := 7
def week2_days := 7
def forgot_doses_day := 2
def total_weeks := 2
def total_water := 160

-- The statement to prove
theorem kara_water_intake :
  let total_doses := (daily_doses * week1_days) + (daily_doses * week2_days - forgot_doses_day)
  ∃ (water_per_dose : ℕ), water_per_dose * total_doses = total_water ∧ water_per_dose = 4 :=
by
  sorry

end NUMINAMATH_GPT_kara_water_intake_l1675_167576


namespace NUMINAMATH_GPT_polygon_sides_eq_eight_l1675_167573

theorem polygon_sides_eq_eight (n : ℕ) (h : (n - 2) * 180 = 3 * 360) : n = 8 := by 
  sorry

end NUMINAMATH_GPT_polygon_sides_eq_eight_l1675_167573


namespace NUMINAMATH_GPT_count_board_configurations_l1675_167528

-- Define the 3x3 board as a type with 9 positions
inductive Position 
| top_left | top_center | top_right
| middle_left | center | middle_right
| bottom_left | bottom_center | bottom_right

-- Define an enum for players' moves
inductive Mark
| X | O | Empty

-- Define a board as a mapping from positions to marks
def Board : Type := Position → Mark

-- Define the win condition for Carl
def win_condition (b : Board) : Prop := 
(b Position.center = Mark.O) ∧ 
((b Position.top_left = Mark.O ∧ b Position.top_center = Mark.O) ∨ 
(b Position.middle_left = Mark.O ∧ b Position.middle_right = Mark.O) ∨ 
(b Position.bottom_left = Mark.O ∧ b Position.bottom_center = Mark.O))

-- Define the condition for a filled board
def filled_board (b : Board) : Prop :=
∀ p : Position, b p ≠ Mark.Empty

-- The proof problem to show the total number of configurations is 30
theorem count_board_configurations : 
  ∃ (n : ℕ), n = 30 ∧
  (∃ b : Board, win_condition b ∧ filled_board b) := 
sorry

end NUMINAMATH_GPT_count_board_configurations_l1675_167528


namespace NUMINAMATH_GPT_rebus_solution_l1675_167538

theorem rebus_solution :
  ∃ (A B C : ℕ), A ≠ 0 ∧ B ≠ 0 ∧ C ≠ 0 ∧ A ≠ B ∧ A ≠ C ∧ B ≠ C ∧
    (A*100 + B*10 + A) + (A*100 + B*10 + C) + (A*100 + C*10 + C) = 1416 ∧ 
    A = 4 ∧ B = 7 ∧ C = 6 :=
by {
  sorry
}

end NUMINAMATH_GPT_rebus_solution_l1675_167538


namespace NUMINAMATH_GPT_page_number_added_twice_l1675_167555

-- Define the sum of natural numbers from 1 to n
def sum_nat (n: ℕ): ℕ := n * (n + 1) / 2

-- Incorrect sum due to one page number being counted twice
def incorrect_sum (n p: ℕ): ℕ := sum_nat n + p

-- Declaring the known conditions as Lean definitions
def n : ℕ := 70
def incorrect_sum_val : ℕ := 2550

-- Lean theorem statement to be proven
theorem page_number_added_twice :
  ∃ p, incorrect_sum n p = incorrect_sum_val ∧ p = 65 := by
  sorry

end NUMINAMATH_GPT_page_number_added_twice_l1675_167555


namespace NUMINAMATH_GPT_nell_gave_cards_l1675_167568

theorem nell_gave_cards (c_original : ℕ) (c_left : ℕ) (cards_given : ℕ) :
  c_original = 528 → c_left = 252 → cards_given = c_original - c_left → cards_given = 276 :=
by
  intros h1 h2 h3
  rw [h1, h2] at h3
  exact h3

end NUMINAMATH_GPT_nell_gave_cards_l1675_167568


namespace NUMINAMATH_GPT_solution_to_quadratic_inequality_l1675_167556

theorem solution_to_quadratic_inequality :
  {x : ℝ | x^2 + 3*x < 10} = {x : ℝ | -5 < x ∧ x < 2} :=
sorry

end NUMINAMATH_GPT_solution_to_quadratic_inequality_l1675_167556


namespace NUMINAMATH_GPT_absent_children_l1675_167500

theorem absent_children (total_children bananas_per_child_if_present bananas_per_child_if_absent children_present absent_children : ℕ) 
  (H1 : total_children = 740)
  (H2 : bananas_per_child_if_present = 2)
  (H3 : bananas_per_child_if_absent = 4)
  (H4 : children_present * bananas_per_child_if_absent = total_children * bananas_per_child_if_present)
  (H5 : children_present = total_children - absent_children) : 
  absent_children = 370 :=
sorry

end NUMINAMATH_GPT_absent_children_l1675_167500


namespace NUMINAMATH_GPT_ryan_sandwiches_l1675_167548

theorem ryan_sandwiches (sandwich_slices : ℕ) (total_slices : ℕ) (h1 : sandwich_slices = 3) (h2 : total_slices = 15) :
  total_slices / sandwich_slices = 5 :=
by
  sorry

end NUMINAMATH_GPT_ryan_sandwiches_l1675_167548


namespace NUMINAMATH_GPT_solve_quadratic_l1675_167524

noncomputable def quadratic_roots (a b c x : ℝ) : Prop := a * x^2 + b * x + c = 0

theorem solve_quadratic : ∀ x : ℝ, quadratic_roots 1 (-4) (-5) x ↔ (x = -1 ∨ x = 5) :=
by
  intro x
  rw [quadratic_roots]
  sorry

end NUMINAMATH_GPT_solve_quadratic_l1675_167524


namespace NUMINAMATH_GPT_quadratic_root_l1675_167513

theorem quadratic_root (k : ℝ) (h : ∃ x : ℝ, x^2 - 2*k*x + k^2 = 0 ∧ x = -1) : k = -1 :=
sorry

end NUMINAMATH_GPT_quadratic_root_l1675_167513


namespace NUMINAMATH_GPT_y_coord_vertex_of_parabola_l1675_167519

-- Define the quadratic equation of the parabola
def parabola (x : ℝ) : ℝ := 2 * x^2 + 16 * x + 29

-- Statement to prove
theorem y_coord_vertex_of_parabola : ∃ (x : ℝ), parabola x = 2 * (x + 4)^2 - 3 := sorry

end NUMINAMATH_GPT_y_coord_vertex_of_parabola_l1675_167519


namespace NUMINAMATH_GPT_bacon_strips_needed_l1675_167564

theorem bacon_strips_needed (plates : ℕ) (eggs_per_plate : ℕ) (bacon_per_plate : ℕ) (customers : ℕ) :
  eggs_per_plate = 2 →
  bacon_per_plate = 2 * eggs_per_plate →
  customers = 14 →
  plates = customers →
  plates * bacon_per_plate = 56 := by
  sorry

end NUMINAMATH_GPT_bacon_strips_needed_l1675_167564


namespace NUMINAMATH_GPT_min_value_of_reciprocal_sum_l1675_167544

theorem min_value_of_reciprocal_sum (x y : ℝ) (h1 : 0 < x) (h2 : 0 < y) (h3 : 2 * x + y = 2) : 
  ∃ (z : ℝ), z = (1 / x + 1 / y) ∧ z = (3 / 2 + Real.sqrt 2) :=
sorry

end NUMINAMATH_GPT_min_value_of_reciprocal_sum_l1675_167544


namespace NUMINAMATH_GPT_coordinates_of_N_l1675_167596

-- Define the given conditions
def M : ℝ × ℝ := (5, -6)
def a : ℝ × ℝ := (1, -2)
def minusThreeA : ℝ × ℝ := (-3, 6)
def vectorMN (N : ℝ × ℝ) : ℝ × ℝ := (N.1 - M.1, N.2 - M.2)

-- Define the required goal
theorem coordinates_of_N (N : ℝ × ℝ) : vectorMN N = minusThreeA → N = (2, 0) :=
by
  sorry

end NUMINAMATH_GPT_coordinates_of_N_l1675_167596


namespace NUMINAMATH_GPT_combination_of_15_3_l1675_167593

open Nat

theorem combination_of_15_3 : choose 15 3 = 455 :=
by
  -- The statement describes that the number of ways to choose 3 books out of 15 is 455
  sorry

end NUMINAMATH_GPT_combination_of_15_3_l1675_167593


namespace NUMINAMATH_GPT_point_not_on_transformed_plane_l1675_167582

def point_A : ℝ × ℝ × ℝ := (4, 0, -3)

def plane_eq (x y z : ℝ) : ℝ := 7 * x - y + 3 * z - 1

def scale_factor : ℝ := 3

def transformed_plane_eq (x y z : ℝ) : ℝ := 7 * x - y + 3 * z - (scale_factor * 1)

theorem point_not_on_transformed_plane :
  transformed_plane_eq 4 0 (-3) ≠ 0 :=
by
  sorry

end NUMINAMATH_GPT_point_not_on_transformed_plane_l1675_167582


namespace NUMINAMATH_GPT_find_m_l1675_167584

theorem find_m 
  (m : ℤ) 
  (h1 : ∀ x y : ℤ, -3 * x + y = m → 2 * x + y = 28 → x = -6) : 
  m = 58 :=
by 
  sorry

end NUMINAMATH_GPT_find_m_l1675_167584


namespace NUMINAMATH_GPT_circles_intersect_at_two_points_l1675_167590

theorem circles_intersect_at_two_points : 
  let C1 := {p : ℝ × ℝ | (p.1 - 3)^2 + p.2^2 = 9}
  let C2 := {p : ℝ × ℝ | p.1^2 + (p.2 - 6)^2 = 36}
  ∃ pts : Finset (ℝ × ℝ), pts.card = 2 ∧ ∀ p ∈ pts, p ∈ C1 ∧ p ∈ C2 := 
sorry

end NUMINAMATH_GPT_circles_intersect_at_two_points_l1675_167590


namespace NUMINAMATH_GPT_simplify_expression_as_single_fraction_l1675_167521

variable (d : ℚ)

theorem simplify_expression_as_single_fraction :
  (5 + 4*d)/9 + 3 = (32 + 4*d)/9 := 
by
  sorry

end NUMINAMATH_GPT_simplify_expression_as_single_fraction_l1675_167521


namespace NUMINAMATH_GPT_max_students_seated_l1675_167504

-- Define the number of seats in the i-th row
def seats_in_row (i : ℕ) : ℕ := 10 + 2 * i

-- Define the maximum number of students that can be seated in the i-th row
def max_students_in_row (i : ℕ) : ℕ := (seats_in_row i + 1) / 2

-- Sum the maximum number of students for all 25 rows
def total_max_students : ℕ := (Finset.range 25).sum max_students_in_row

-- The theorem statement
theorem max_students_seated : total_max_students = 450 := by
  sorry

end NUMINAMATH_GPT_max_students_seated_l1675_167504


namespace NUMINAMATH_GPT_solve_garden_width_l1675_167526

noncomputable def garden_width_problem (w l : ℕ) :=
  (w + l = 30) ∧ (w * l = 200) ∧ (l = w + 8) → w = 11

theorem solve_garden_width (w l : ℕ) : garden_width_problem w l :=
by
  intro h
  -- Omitting the actual proof
  sorry

end NUMINAMATH_GPT_solve_garden_width_l1675_167526


namespace NUMINAMATH_GPT_point_in_quadrant_l1675_167561

theorem point_in_quadrant (a b : ℝ) (h1 : a - b > 0) (h2 : a * b < 0) : 
  (a > 0 ∧ b < 0) ∧ ¬(a > 0 ∧ b > 0) ∧ ¬(a < 0 ∧ b > 0) ∧ ¬(a < 0 ∧ b < 0) := 
by 
  sorry

end NUMINAMATH_GPT_point_in_quadrant_l1675_167561


namespace NUMINAMATH_GPT_product_of_primes_sum_101_l1675_167560

theorem product_of_primes_sum_101 (p q : ℕ) (hp : Nat.Prime p) (hq : Nat.Prime q) (h_sum : p + q = 101) : p * q = 194 := by
  sorry

end NUMINAMATH_GPT_product_of_primes_sum_101_l1675_167560


namespace NUMINAMATH_GPT_attendants_both_tools_l1675_167583

theorem attendants_both_tools (pencil_users pen_users only_one_type total_attendants both_types : ℕ)
  (h1 : pencil_users = 25) 
  (h2 : pen_users = 15) 
  (h3 : only_one_type = 20) 
  (h4 : total_attendants = only_one_type + both_types) 
  (h5 : total_attendants = pencil_users + pen_users - both_types) 
  : both_types = 10 :=
by
  -- Fill in the proof sub-steps here if needed
  sorry

end NUMINAMATH_GPT_attendants_both_tools_l1675_167583


namespace NUMINAMATH_GPT_A_speed_is_10_l1675_167594

noncomputable def A_walking_speed (v t : ℝ) := 
  v * (t + 7) = 140 ∧ v * (t + 7) = 20 * t

theorem A_speed_is_10 (v t : ℝ) 
  (h1 : v * (t + 7) = 140)
  (h2 : v * (t + 7) = 20 * t) :
  v = 10 :=
sorry

end NUMINAMATH_GPT_A_speed_is_10_l1675_167594


namespace NUMINAMATH_GPT_bottle_t_capsules_l1675_167501

theorem bottle_t_capsules 
  (num_capsules_r : ℕ)
  (cost_r : ℝ)
  (cost_t : ℝ)
  (cost_per_capsule_difference : ℝ)
  (h1 : num_capsules_r = 250)
  (h2 : cost_r = 6.25)
  (h3 : cost_t = 3.00)
  (h4 : cost_per_capsule_difference = 0.005) :
  ∃ (num_capsules_t : ℕ), num_capsules_t = 150 := 
by
  sorry

end NUMINAMATH_GPT_bottle_t_capsules_l1675_167501


namespace NUMINAMATH_GPT_solve_system_l1675_167537

theorem solve_system :
  ∃ x y : ℝ, (x + y = 5) ∧ (x + 2 * y = 8) ∧ (x = 2) ∧ (y = 3) :=
by
  sorry

end NUMINAMATH_GPT_solve_system_l1675_167537


namespace NUMINAMATH_GPT_quadratic_roots_l1675_167542

theorem quadratic_roots : ∀ x : ℝ, (x^2 - 6 * x + 5 = 0) ↔ (x = 5 ∨ x = 1) :=
by sorry

end NUMINAMATH_GPT_quadratic_roots_l1675_167542


namespace NUMINAMATH_GPT_difference_of_cubes_l1675_167509

theorem difference_of_cubes (x y : ℕ) (h1 : x = y + 3) (h2 : x + y = 5) : x^3 - y^3 = 63 :=
by sorry

end NUMINAMATH_GPT_difference_of_cubes_l1675_167509


namespace NUMINAMATH_GPT_complex_inequality_l1675_167530

open Complex

noncomputable def condition (a b c : ℂ) := a * Complex.abs (b * c) + b * Complex.abs (c * a) + c * Complex.abs (a * b) = 0

theorem complex_inequality (a b c : ℂ) (h : condition a b c) :
  Complex.abs ((a - b) * (b - c) * (c - a)) ≥ 3 * Real.sqrt 3 * Complex.abs (a * b * c) := 
sorry

end NUMINAMATH_GPT_complex_inequality_l1675_167530


namespace NUMINAMATH_GPT_not_always_greater_quotient_l1675_167543

theorem not_always_greater_quotient (a : ℝ) (b : ℝ) (ha : a ≠ 0) (hb : 0 < b) : ¬ (∀ b < 1, a / b > a) ∧ ¬ (∀ b > 1, a / b > a) :=
by sorry

end NUMINAMATH_GPT_not_always_greater_quotient_l1675_167543


namespace NUMINAMATH_GPT_garden_breadth_l1675_167514

theorem garden_breadth (P L B : ℕ) (h₁ : P = 950) (h₂ : L = 375) (h₃ : P = 2 * (L + B)) : B = 100 := by
  sorry

end NUMINAMATH_GPT_garden_breadth_l1675_167514


namespace NUMINAMATH_GPT_smallest_n_for_congruence_l1675_167572

theorem smallest_n_for_congruence : ∃ n : ℕ, 0 < n ∧ 7^n % 5 = n^4 % 5 ∧ (∀ m : ℕ, 0 < m ∧ 7^m % 5 = m^4 % 5 → n ≤ m) ∧ n = 4 :=
by
  sorry

end NUMINAMATH_GPT_smallest_n_for_congruence_l1675_167572


namespace NUMINAMATH_GPT_Kim_sales_on_Friday_l1675_167598

theorem Kim_sales_on_Friday (tuesday_sales : ℕ) (tuesday_discount_rate : ℝ) 
    (monday_increase_rate : ℝ) (wednesday_increase_rate : ℝ) 
    (thursday_decrease_rate : ℝ) (friday_increase_rate : ℝ) 
    (final_friday_sales : ℕ) :
    tuesday_sales = 800 →
    tuesday_discount_rate = 0.05 →
    monday_increase_rate = 0.50 →
    wednesday_increase_rate = 1.5 →
    thursday_decrease_rate = 0.20 →
    friday_increase_rate = 1.3 →
    final_friday_sales = 1310 :=
by
  sorry

end NUMINAMATH_GPT_Kim_sales_on_Friday_l1675_167598


namespace NUMINAMATH_GPT_find_T_shirts_l1675_167550

variable (T S : ℕ)

-- Given conditions
def condition1 : S = 2 * T := sorry
def condition2 : T + S - (T + 3) = 15 := sorry

-- Prove that number of T-shirts T Norma left in the washer is 9
theorem find_T_shirts (h1 : S = 2 * T) (h2 : T + S - (T + 3) = 15) : T = 9 :=
  by
    sorry

end NUMINAMATH_GPT_find_T_shirts_l1675_167550


namespace NUMINAMATH_GPT_num_students_in_second_class_l1675_167525

theorem num_students_in_second_class 
  (avg1 : ℕ) (num1 : ℕ) (avg2 : ℕ) (overall_avg : ℕ) (n : ℕ) :
  avg1 = 50 → num1 = 30 → avg2 = 60 → overall_avg = 5625 → 
  (num1 * avg1 + n * avg2) = (num1 + n) * overall_avg → n = 50 :=
by sorry

end NUMINAMATH_GPT_num_students_in_second_class_l1675_167525


namespace NUMINAMATH_GPT_graph_not_pass_through_second_quadrant_l1675_167587

theorem graph_not_pass_through_second_quadrant 
    (k : ℝ) (b : ℝ) (h1 : k = 1) (h2 : b = -2) : 
    ¬ ∃ (x y : ℝ), y = k * x + b ∧ x < 0 ∧ y > 0 := 
by
  sorry

end NUMINAMATH_GPT_graph_not_pass_through_second_quadrant_l1675_167587


namespace NUMINAMATH_GPT_smallest_possible_value_of_other_integer_l1675_167579

theorem smallest_possible_value_of_other_integer (x : ℕ) (x_pos : 0 < x) (a b : ℕ) (h1 : a = 77) 
    (h2 : gcd a b = x + 7) (h3 : lcm a b = x * (x + 7)) : b = 22 :=
sorry

end NUMINAMATH_GPT_smallest_possible_value_of_other_integer_l1675_167579


namespace NUMINAMATH_GPT_quadratic_decreasing_l1675_167569

theorem quadratic_decreasing (a : ℝ) (h : ∀ x1 x2 : ℝ, x1 ≤ x2 → x2 ≤ 4 → (x1^2 + 4*a*x1 - 2) ≥ (x2^2 + 4*a*x2 - 2)) : a ≤ -2 := 
by
  sorry

end NUMINAMATH_GPT_quadratic_decreasing_l1675_167569


namespace NUMINAMATH_GPT_range_of_a_l1675_167567

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, (a-1) * x^2 + 2 * (a-1) * x - 4 ≥ 0 -> false) ↔ -3 < a ∧ a ≤ 1 := by
  sorry

end NUMINAMATH_GPT_range_of_a_l1675_167567


namespace NUMINAMATH_GPT_magnitude_v_l1675_167595

open Complex

theorem magnitude_v (u v : ℂ) (h1 : u * v = 20 - 15 * Complex.I) (h2 : Complex.abs u = 5) :
  Complex.abs v = 5 := by
  sorry

end NUMINAMATH_GPT_magnitude_v_l1675_167595


namespace NUMINAMATH_GPT_necessary_and_sufficient_condition_l1675_167531

variable (p q : Prop)

theorem necessary_and_sufficient_condition (hp : p) (hq : q) : ¬p ∨ ¬q = False :=
by {
    -- You are requested to fill out the proof here.
    sorry
}

end NUMINAMATH_GPT_necessary_and_sufficient_condition_l1675_167531


namespace NUMINAMATH_GPT_unique_natural_in_sequences_l1675_167553

def seq_x (n : ℕ) : ℤ := if n = 0 then 10 else if n = 1 then 10 else seq_x (n - 2) * (seq_x (n - 1) + 1) + 1
def seq_y (n : ℕ) : ℤ := if n = 0 then -10 else if n = 1 then -10 else (seq_y (n - 1) + 1) * seq_y (n - 2) + 1

theorem unique_natural_in_sequences (k : ℕ) (i j : ℕ) :
  seq_x i = k → seq_y j ≠ k :=
by
  sorry

end NUMINAMATH_GPT_unique_natural_in_sequences_l1675_167553


namespace NUMINAMATH_GPT_original_denominator_is_nine_l1675_167559

theorem original_denominator_is_nine (d : ℕ) : 
  (2 + 5) / (d + 5) = 1 / 2 → d = 9 := 
by sorry

end NUMINAMATH_GPT_original_denominator_is_nine_l1675_167559


namespace NUMINAMATH_GPT_value_of_a_l1675_167574

def f (a : ℝ) (x : ℝ) : ℝ := a * x^3 + 3 * x^2 + 2

theorem value_of_a (a : ℝ) : 
  (∀ x, deriv (f a) x = 6 * x + 3 * a * x^2) →
  deriv (f a) (-1) = 6 → a = 4 :=
by
  -- Proof will be filled in here
  sorry

end NUMINAMATH_GPT_value_of_a_l1675_167574


namespace NUMINAMATH_GPT_problem_l1675_167592

-- Define the problem
theorem problem {a b c : ℤ} (h1 : a = c + 1) (h2 : b - 1 = a) :
  (a - b) ^ 2 + (b - c) ^ 2 + (c - a) ^ 2 = 6 := 
sorry

end NUMINAMATH_GPT_problem_l1675_167592


namespace NUMINAMATH_GPT_average_of_remaining_two_numbers_l1675_167523

theorem average_of_remaining_two_numbers (a b c d e f : ℝ)
  (h1 : (a + b + c + d + e + f) / 6 = 3.95)
  (h2 : (a + b) / 2 = 3.4)
  (h3 : (c + d) / 2 = 3.85) :
  (e + f) / 2 = 4.6 :=
sorry

end NUMINAMATH_GPT_average_of_remaining_two_numbers_l1675_167523


namespace NUMINAMATH_GPT_evaluate_expression_l1675_167562

theorem evaluate_expression : 3 + 5 * 2^3 - 4 / 2 + 7 * 3 = 62 := 
  by sorry

end NUMINAMATH_GPT_evaluate_expression_l1675_167562


namespace NUMINAMATH_GPT_num_non_divisible_by_3_divisors_l1675_167585

theorem num_non_divisible_by_3_divisors (a b c : ℕ) (h1: 0 ≤ a ∧ a ≤ 2) (h2: 0 ≤ b ∧ b ≤ 2) (h3: 0 ≤ c ∧ c ≤ 1) :
  (3 * 2 = 6) :=
by sorry

end NUMINAMATH_GPT_num_non_divisible_by_3_divisors_l1675_167585


namespace NUMINAMATH_GPT_average_riding_speed_l1675_167547

theorem average_riding_speed
  (initial_reading : ℕ) (final_reading : ℕ) (time_day1 : ℕ) (time_day2 : ℕ)
  (h_initial : initial_reading = 2332)
  (h_final : final_reading = 2552)
  (h_time_day1 : time_day1 = 5)
  (h_time_day2 : time_day2 = 4) :
  (final_reading - initial_reading) / (time_day1 + time_day2) = 220 / 9 :=
by
  sorry

end NUMINAMATH_GPT_average_riding_speed_l1675_167547


namespace NUMINAMATH_GPT_count_lines_in_2008_cube_l1675_167581

def num_lines_through_centers_of_unit_cubes (n : ℕ) : ℕ :=
  n * n * 3 + n * 2 * 3 + 4

theorem count_lines_in_2008_cube :
  num_lines_through_centers_of_unit_cubes 2008 = 12115300 :=
by
  -- The actual proof would go here
  sorry

end NUMINAMATH_GPT_count_lines_in_2008_cube_l1675_167581


namespace NUMINAMATH_GPT_find_a_and_b_min_value_expression_l1675_167580

universe u

-- Part (1): Prove the values of a and b
theorem find_a_and_b :
    (∀ x : ℝ, a * x^2 - 3 * x + 2 > 0 ↔ x < 1 ∨ x > b) →
    a = 1 ∧ b = 2 :=
sorry

-- Part (2): Given a = 1 and b = 2 prove the minimum value of 2x + y + 3
theorem min_value_expression :
    (1 / (x + 1) + 2 / (y + 1) = 1) →
    (x > 0) →
    (y > 0) →
    ∀ x y : ℝ, 2 * x + y + 3 ≥ 8 :=
sorry

end NUMINAMATH_GPT_find_a_and_b_min_value_expression_l1675_167580


namespace NUMINAMATH_GPT_volume_of_prism_is_429_l1675_167588

theorem volume_of_prism_is_429 (x y z : ℝ) (h1 : x * y = 56) (h2 : y * z = 57) (h3 : z * x = 58) : 
  x * y * z = 429 :=
by
  sorry

end NUMINAMATH_GPT_volume_of_prism_is_429_l1675_167588


namespace NUMINAMATH_GPT_simplify_expression_l1675_167522

theorem simplify_expression : 
  ((3 + 4 + 5 + 6 + 7) / 3 + (3 * 6 + 9)^2 / 9) = 268 / 3 := 
by 
  sorry

end NUMINAMATH_GPT_simplify_expression_l1675_167522


namespace NUMINAMATH_GPT_speed_ratio_bus_meets_Vasya_first_back_trip_time_l1675_167557

namespace TransportProblem

variable (d : ℝ) -- distance from point A to B
variable (v_bus : ℝ) -- bus speed
variable (v_Vasya : ℝ) -- Vasya's speed
variable (v_Petya : ℝ) -- Petya's speed

-- Conditions
axiom bus_speed : v_bus * 3 = d
axiom bus_meet_Vasya_second_trip : 7.5 * v_Vasya = 0.5 * d
axiom bus_meet_Petya_at_B : 9 * v_Petya = d
axiom bus_start_time : d / v_bus = 3

theorem speed_ratio: (v_Vasya / v_Petya) = (3 / 5) :=
  sorry

theorem bus_meets_Vasya_first_back_trip_time: ∃ (x: ℕ), x = 11 :=
  sorry

end TransportProblem

end NUMINAMATH_GPT_speed_ratio_bus_meets_Vasya_first_back_trip_time_l1675_167557
