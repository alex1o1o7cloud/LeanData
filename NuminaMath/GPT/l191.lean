import Mathlib

namespace calculation_correct_l191_191679

-- Defining the initial values
def a : ℕ := 20 ^ 10
def b : ℕ := 20 ^ 9
def c : ℕ := 10 ^ 6
def d : ℕ := 2 ^ 12

-- The expression we need to prove
theorem calculation_correct : ((a / b) ^ 3 * c) / d = 1953125 :=
by
  sorry

end calculation_correct_l191_191679


namespace cinco_de_mayo_day_days_between_feb_14_and_may_5_l191_191689

theorem cinco_de_mayo_day {
  feb_14_is_tuesday : ∃ n : ℕ, n % 7 = 2
}: 
∃ n : ℕ, n % 7 = 5 := sorry

theorem days_between_feb_14_and_may_5: 
  ∃ d : ℕ, 
  d = 81 := sorry

end cinco_de_mayo_day_days_between_feb_14_and_may_5_l191_191689


namespace remove_terms_sum_l191_191524

theorem remove_terms_sum :
  let s := (1/3 + 1/5 + 1/7 + 1/9 + 1/11 + 1/13 + 1/15 : ℚ)
  s = 16339/15015 →
  (1/13 + 1/15 = 2061/5005) →
  s - (1/13 + 1/15) = 3/2 :=
by
  intros s hs hremove
  have hrem : (s - (1/13 + 1/15 : ℚ) = 3/2) ↔ (16339/15015 - 2061/5005 = 3/2) := sorry
  exact hrem.mpr sorry

end remove_terms_sum_l191_191524


namespace quadratic_two_distinct_real_roots_l191_191773

theorem quadratic_two_distinct_real_roots : 
  ∀ x : ℝ, ∃ a b c : ℝ, (∀ x : ℝ, (x+1)*(x-1) = 2*x + 3 → x^2 - 2*x - 4 = 0) ∧ 
  (a = 1) ∧ (b = -2) ∧ (c = -4) ∧ (b^2 - 4*a*c > 0) :=
by
  sorry

end quadratic_two_distinct_real_roots_l191_191773


namespace smallest_sum_xyz_l191_191631

theorem smallest_sum_xyz (x y z : ℕ) (h : x * y * z = 40320) : x + y + z ≥ 103 :=
sorry

end smallest_sum_xyz_l191_191631


namespace median_of_trapezoid_l191_191823

theorem median_of_trapezoid (h : ℝ) (x : ℝ) 
  (triangle_area_eq_trapezoid_area : (1 / 2) * 24 * h = ((x + (2 * x)) / 2) * h) : 
  ((x + (2 * x)) / 2) = 12 := by
  sorry

end median_of_trapezoid_l191_191823


namespace candle_problem_l191_191384

-- Define the initial heights and burn rates of the candles
def heightA (t : ℝ) : ℝ := 12 - 2 * t
def heightB (t : ℝ) : ℝ := 15 - 3 * t

-- Lean theorem statement for the given problem
theorem candle_problem : ∃ t : ℝ, (heightA t = (1/3) * heightB t) ∧ t = 7 :=
by
  -- This is to keep the theorem statement valid without the proof
  sorry

end candle_problem_l191_191384


namespace derivative_at_one_is_three_l191_191029

-- Definition of the function
def f (x : ℝ) := (x - 1)^2 + 3 * (x - 1)

-- The statement of the problem
theorem derivative_at_one_is_three : deriv f 1 = 3 := 
  sorry

end derivative_at_one_is_three_l191_191029


namespace sandy_correct_sums_l191_191076

/-- 
Sandy gets 3 marks for each correct sum and loses 2 marks for each incorrect sum.
Sandy attempts 50 sums and obtains 100 marks within a 45-minute time constraint.
If Sandy receives a 1-mark penalty for each sum not completed within the time limit,
prove that the number of correct sums Sandy got is 25.
-/
theorem sandy_correct_sums (c i : ℕ) (h1 : c + i = 50) (h2 : 3 * c - 2 * i - (50 - c) = 100) : c = 25 :=
by
  sorry

end sandy_correct_sums_l191_191076


namespace algebraic_expression_independence_l191_191599

theorem algebraic_expression_independence (a b : ℝ) (h : ∀ x : ℝ, (x^2 + a*x - (b*x^2 - x - 3)) = 3) : a - b = -2 :=
by
  sorry

end algebraic_expression_independence_l191_191599


namespace temperature_celsius_range_l191_191783

theorem temperature_celsius_range (C : ℝ) :
  (∀ C : ℝ, let F_approx := 2 * C + 30;
             let F_exact := (9 / 5) * C + 32;
             abs ((2 * C + 30 - ((9 / 5) * C + 32)) / ((9 / 5) * C + 32)) ≤ 0.05) →
  (40 / 29) ≤ C ∧ C ≤ (360 / 11) :=
by
  intros h
  sorry

end temperature_celsius_range_l191_191783


namespace transport_cost_expression_and_min_cost_l191_191105

noncomputable def total_transport_cost (x : ℕ) (a : ℕ) : ℕ :=
if 2 ≤ a ∧ a ≤ 6 then (5 - a) * x + 23200 else 0

theorem transport_cost_expression_and_min_cost :
  ∀ x : ℕ, ∀ a : ℕ,
  (100 ≤ x ∧ x ≤ 800) →
  (2 ≤ a ∧ a ≤ 6) →
  (total_transport_cost x a = 5 * x + 23200) ∧ 
  (a = 6 → total_transport_cost 800 a = 22400) :=
by
  intros
  -- Provide the detailed proof here.
  sorry

end transport_cost_expression_and_min_cost_l191_191105


namespace least_number_to_subtract_from_724946_l191_191661

def divisible_by_10 (n : ℕ) : Prop :=
  n % 10 = 0

theorem least_number_to_subtract_from_724946 :
  ∃ k : ℕ, k = 6 ∧ divisible_by_10 (724946 - k) :=
by
  sorry

end least_number_to_subtract_from_724946_l191_191661


namespace min_m_value_arithmetic_seq_l191_191721

theorem min_m_value_arithmetic_seq :
  ∀ (a S : ℕ → ℚ) (m : ℕ),
  (∀ n : ℕ, a (n+2) = 5 ∧ a (n+6) = 21) →
  (∀ n : ℕ, S (n+1) = S n + 1 / a (n+1)) →
  (∀ n : ℕ, S (2 * n + 1) - S n ≤ m / 15) →
  ∀ n : ℕ, m = 5 :=
sorry

end min_m_value_arithmetic_seq_l191_191721


namespace minimum_value_expression_l191_191024

theorem minimum_value_expression (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a + b = 1) :
  (∃ x : ℝ, x = 9 ∧ ∀ y : ℝ, y = (1 / a^2 - 1) * (1 / b^2 - 1) → x ≤ y) :=
sorry

end minimum_value_expression_l191_191024


namespace solve_for_x_l191_191751

-- Define the problem
def equation (x : ℝ) : Prop := x + 2 * x + 12 = 500 - (3 * x + 4 * x)

-- State the theorem that we want to prove
theorem solve_for_x : ∃ (x : ℝ), equation x ∧ x = 48.8 := by
  sorry

end solve_for_x_l191_191751


namespace period_of_f_minimum_value_zero_then_a_eq_one_maximum_value_of_f_axis_of_symmetry_l191_191450

noncomputable def f (x a : ℝ) := 2 * Real.sqrt 3 * Real.sin x * Real.cos x + 2 * (Real.cos x) ^ 2 + a

theorem period_of_f : ∀ a : ℝ, ∀ x : ℝ, f (x + π) a = f x a := 
by sorry

theorem minimum_value_zero_then_a_eq_one : (∀ x : ℝ, f x a ≥ 0) → a = 1 := 
by sorry

theorem maximum_value_of_f : a = 1 → (∀ x : ℝ, f x 1 ≤ 4) :=
by sorry

theorem axis_of_symmetry : a = 1 → ∃ k : ℤ, ∀ x : ℝ, 2 * x + π / 6 = k * π + π / 2 ↔ f x 1 = f 0 1 :=
by sorry

end period_of_f_minimum_value_zero_then_a_eq_one_maximum_value_of_f_axis_of_symmetry_l191_191450


namespace fixed_monthly_costs_l191_191532

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

end fixed_monthly_costs_l191_191532


namespace set_of_points_l191_191015

theorem set_of_points : {p : ℝ × ℝ | (2 * p.1 - p.2 = 1) ∧ (p.1 + 4 * p.2 = 5)} = { (1, 1) } :=
by
  sorry

end set_of_points_l191_191015


namespace fill_tank_with_leak_l191_191539

theorem fill_tank_with_leak (P L T : ℝ) 
  (hP : P = 1 / 2)  -- Rate of the pump
  (hL : L = 1 / 6)  -- Rate of the leak
  (hT : T = 3)  -- Time taken to fill the tank with the leak
  : 1 / (P - L) = T := 
by
  sorry

end fill_tank_with_leak_l191_191539


namespace exists_three_with_gcd_d_l191_191018

theorem exists_three_with_gcd_d (n : ℕ) (nums : Fin n.succ → ℕ) (d : ℕ)
  (h1 : n ≥ 2)  -- because n+1 (number of elements nums : Fin n.succ) ≥ 3 given that n ≥ 2
  (h2 : ∀ i, nums i > 0) 
  (h3 : ∀ i, nums i ≤ 100) 
  (h4 : Nat.gcd (nums 0) (Nat.gcd (nums 1) (nums 2)) = d) : 
  ∃ i j k : Fin n.succ, i ≠ j ∧ j ≠ k ∧ k ≠ i ∧ Nat.gcd (nums i) (Nat.gcd (nums j) (nums k)) = d :=
by
  sorry

end exists_three_with_gcd_d_l191_191018


namespace min_cost_per_ounce_l191_191404

theorem min_cost_per_ounce 
  (cost_40 : ℝ := 200) (cost_90 : ℝ := 400)
  (percentage_40 : ℝ := 0.4) (percentage_90 : ℝ := 0.9)
  (desired_percentage : ℝ := 0.5) :
  (∀ (x y : ℝ), 0.4 * x + 0.9 * y = 0.5 * (x + y) → 200 * x + 400 * y / (x + y) = 240) :=
sorry

end min_cost_per_ounce_l191_191404


namespace min_distance_between_M_and_N_l191_191436

noncomputable def f (x : ℝ) := Real.sin x + (1 / 6) * x^3
noncomputable def g (x : ℝ) := x - 1

theorem min_distance_between_M_and_N :
  ∃ (x1 x2 : ℝ), x1 ≥ 0 ∧ x2 ≥ 0 ∧ f x1 = g x2 ∧ (x2 - x1 = 1) :=
sorry

end min_distance_between_M_and_N_l191_191436


namespace greatest_positive_multiple_of_4_l191_191235

theorem greatest_positive_multiple_of_4 {y : ℕ} (h1 : y % 4 = 0) (h2 : y > 0) (h3 : y^3 < 8000) : y ≤ 16 :=
by {
  -- The proof will go here
  -- Sorry is placed here to skip the proof for now
  sorry
}

end greatest_positive_multiple_of_4_l191_191235


namespace compute_zeta_seventh_power_sum_l191_191122

noncomputable def complex_seventh_power_sum : Prop :=
  ∀ (ζ₁ ζ₂ ζ₃ : ℂ), 
    (ζ₁ + ζ₂ + ζ₃ = 1) ∧ 
    (ζ₁^2 + ζ₂^2 + ζ₃^2 = 3) ∧
    (ζ₁^3 + ζ₂^3 + ζ₃^3 = 7) →
    (ζ₁^7 + ζ₂^7 + ζ₃^7 = 71)

theorem compute_zeta_seventh_power_sum : complex_seventh_power_sum :=
by
  sorry

end compute_zeta_seventh_power_sum_l191_191122


namespace evaluate_expression_l191_191392

theorem evaluate_expression (a b : ℝ) (h1 : a = 4) (h2 : b = -1) : -2 * a ^ 2 - 3 * b ^ 2 + 2 * a * b = -43 :=
by
  sorry

end evaluate_expression_l191_191392


namespace smallest_base_for_80_l191_191117

-- Define the problem in terms of inequalities
def smallest_base (n : ℕ) (d : ℕ) :=
  ∃ b : ℕ, b > 1 ∧ b <= (n^(1/d)) ∧ (n^(1/(d+1))) < (b + 1)

-- Assertion that the smallest whole number b such that 80 can be expressed in base b using only three digits
theorem smallest_base_for_80 : ∃ b, smallest_base 80 3 ∧ b = 5 :=
  sorry

end smallest_base_for_80_l191_191117


namespace Jirina_number_l191_191474

theorem Jirina_number (a b c d : ℕ) (h_abcd : 1000 * a + 100 * b + 10 * c + d = 1468) :
  (1000 * a + 100 * b + 10 * c + d) +
  (1000 * a + 100 * d + 10 * c + b) = 3332 ∧ 
  (1000 * a + 100 * b + 10 * c + d)+
  (1000 * c + 100 * b + 10 * a + d) = 7886 :=
by
  sorry

end Jirina_number_l191_191474


namespace least_y_value_l191_191789

theorem least_y_value (y : ℝ) : 2 * y ^ 2 + 7 * y + 3 = 5 → y ≥ -2 :=
by
  intro h
  sorry

end least_y_value_l191_191789


namespace books_cost_l191_191102

theorem books_cost (total_cost_three_books cost_seven_books : ℕ) 
  (h₁ : total_cost_three_books = 45)
  (h₂ : cost_seven_books = 7 * (total_cost_three_books / 3)) : 
  cost_seven_books = 105 :=
  sorry

end books_cost_l191_191102


namespace small_drinking_glasses_count_l191_191061

theorem small_drinking_glasses_count :
  ∀ (large_jelly_beans_per_large_glass small_jelly_beans_per_small_glass total_jelly_beans : ℕ),
  (large_jelly_beans_per_large_glass = 50) →
  (small_jelly_beans_per_small_glass = large_jelly_beans_per_large_glass / 2) →
  (5 * large_jelly_beans_per_large_glass + n * small_jelly_beans_per_small_glass = total_jelly_beans) →
  (total_jelly_beans = 325) →
  n = 3 := by
  sorry

end small_drinking_glasses_count_l191_191061


namespace intersection_M_N_l191_191039

def I : Set ℤ := {0, -1, -2, -3, -4}
def M : Set ℤ := {0, -1, -2}
def N : Set ℤ := {0, -3, -4}

theorem intersection_M_N : M ∩ N = {0} := 
by 
  sorry

end intersection_M_N_l191_191039


namespace ratio_of_area_to_perimeter_l191_191930

noncomputable def side_length := 10
noncomputable def altitude := (side_length * (Real.sqrt 3 / 2))
noncomputable def area := (1 / 2) * side_length * altitude
noncomputable def perimeter := 3 * side_length

theorem ratio_of_area_to_perimeter (s : ℝ) (h : ℝ) (A : ℝ) (P : ℝ) 
  (h1 : s = 10) 
  (h2 : h = s * (Real.sqrt 3 / 2)) 
  (h3 : A = (1 / 2) * s * h) 
  (h4 : P = 3 * s) :
  A / P = 5 * Real.sqrt 3 / 6 := by
  sorry

end ratio_of_area_to_perimeter_l191_191930


namespace find_y_l191_191047

theorem find_y (x y : ℝ) (h₁ : 1.5 * x = 0.3 * y) (h₂ : x = 20) : y = 100 :=
sorry

end find_y_l191_191047


namespace prime_sum_of_primes_unique_l191_191836

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem prime_sum_of_primes_unique (p q : ℕ) (hp : is_prime p) (hq : is_prime q) (h_sum_prime : is_prime (p^q + q^p)) :
  p = 2 ∧ q = 3 :=
sorry

end prime_sum_of_primes_unique_l191_191836


namespace sum_first_60_digits_of_fraction_l191_191796

theorem sum_first_60_digits_of_fraction (n : ℕ) (h : n = 60) :
  let decimal_expansion := "000891".cycle in
  (decimal_expansion.take n).foldr (λ c acc, acc + (c.to_nat - '0'.to_nat)) 0 = 180 :=
by sorry

end sum_first_60_digits_of_fraction_l191_191796


namespace find_x_value_l191_191153

open Real

theorem find_x_value (a b c : ℤ) (x : ℝ) (h : 5 / (a^2 + b * log x) = c) : 
  x = 10^((5 / c - a^2) / b) := 
by 
  sorry

end find_x_value_l191_191153


namespace arithmetic_sequence_property_l191_191468

variable {α : Type*} [LinearOrderedField α]

theorem arithmetic_sequence_property
  (a : ℕ → α) (h1 : a 1 + a 8 = 9) (h4 : a 4 = 3) : a 5 = 6 :=
by
  sorry

end arithmetic_sequence_property_l191_191468


namespace findPerpendicularLine_l191_191124

-- Defining the condition: the line passes through point (-1, 2)
def pointOnLine (x y : ℝ) (a b : ℝ) (c : ℝ) : Prop :=
  a * x + b * y + c = 0

-- Defining the condition: the line is perpendicular to 2x - 3y + 4 = 0
def isPerpendicular (a1 b1 a2 b2 : ℝ) : Prop :=
  a1 * a2 + b1 * b2 = 0

-- The original line equation: 2x - 3y + 4 = 0
def originalLine (x y : ℝ) : Prop :=
  2 * x - 3 * y + 4 = 0

-- The target equation of the line: 3x + 2y - 1 = 0
def targetLine (x y : ℝ) : Prop :=
  3 * x + 2 * y - 1 = 0

theorem findPerpendicularLine :
  (pointOnLine (-1) 2 3 2 (-1)) ∧
  (isPerpendicular 3 2 2 (-3)) →
  (∀ x y, targetLine x y ↔ 3 * x + 2 * y - 1 = 0) :=
by
  sorry

end findPerpendicularLine_l191_191124


namespace negation_of_P_l191_191451

open Real

theorem negation_of_P :
  (¬ (∀ x : ℝ, x > sin x)) ↔ (∃ x : ℝ, x ≤ sin x) :=
by
  sorry

end negation_of_P_l191_191451


namespace value_of_a_plus_b_l191_191449

noncomputable def f (x : ℝ) (a b : ℝ) : ℝ :=
  if 0 ≤ x then Real.sqrt x + 3 else a * x + b

theorem value_of_a_plus_b (a b : ℝ) 
  (h1 : ∀ x1 : ℝ, x1 ≠ 0 → ∃ x2 : ℝ, x1 ≠ x2 ∧ f x1 a b = f x2 a b)
  (h2 : f (2 * a) a b = f (3 * b) a b) :
  a + b = - (Real.sqrt 6) / 2 + 3 :=
by
  sorry

end value_of_a_plus_b_l191_191449


namespace correct_option_l191_191179

variable (f : ℝ → ℝ)
variable (h_diff : ∀ x : ℝ, differentiable_at ℝ f x)
variable (h_cond : ∀ x : ℝ, f x > deriv f x)

theorem correct_option :
  e ^ 2016 * f (-2016) > f 0 ∧ f 2016 < e ^ 2016 * f 0 :=
sorry

end correct_option_l191_191179


namespace division_of_cubics_l191_191423

theorem division_of_cubics (c d : ℕ) (h1 : c = 7) (h2 : d = 3) : 
  (c^3 + d^3) / (c^2 - c * d + d^2) = 10 := by
  sorry

end division_of_cubics_l191_191423


namespace evaluate_expression_l191_191008

theorem evaluate_expression :
  (|(-1 : ℝ)|^2023 + (Real.sqrt 3)^2 - 2 * Real.sin (Real.pi / 6) + (1 / 2)⁻¹ = 5) :=
by
  sorry

end evaluate_expression_l191_191008


namespace seashells_count_l191_191744

theorem seashells_count : 18 + 47 = 65 := by
  sorry

end seashells_count_l191_191744


namespace todd_has_40_left_after_paying_back_l191_191925

def todd_snowcone_problem : Prop :=
  let borrowed := 100
  let repay := 110
  let cost_ingredients := 75
  let snowcones_sold := 200
  let price_per_snowcone := 0.75
  let total_earnings := snowcones_sold * price_per_snowcone
  let remaining_money := total_earnings - repay
  remaining_money = 40

theorem todd_has_40_left_after_paying_back : todd_snowcone_problem :=
by
  -- Add proof here if needed
  sorry

end todd_has_40_left_after_paying_back_l191_191925


namespace prob_same_color_eq_19_div_39_l191_191800

/-- Definition of initial conditions --/
def total_balls : ℕ := 5 + 8
def green_balls : ℕ := 5
def white_balls : ℕ := 8

/-- Helper functions to calculate probabilities --/
def prob_green_first : ℚ := green_balls / total_balls
def prob_green_second : ℚ := (green_balls - 1) / (total_balls - 1)
def prob_both_green : ℚ := prob_green_first * prob_green_second

def prob_white_first : ℚ := white_balls / total_balls
def prob_white_second : ℚ := (white_balls - 1) / (total_balls - 1)
def prob_both_white : ℚ := prob_white_first * prob_white_second

def prob_same_color : ℚ := prob_both_green + prob_both_white

/-- The main proof statement --/
theorem prob_same_color_eq_19_div_39 :
  prob_same_color = 19 / 39 :=
by
  -- This is where the proof would be constructed.
  sorry

end prob_same_color_eq_19_div_39_l191_191800


namespace taxi_ride_distance_l191_191807

variable (t : ℝ) (c₀ : ℝ) (cᵢ : ℝ)

theorem taxi_ride_distance (h_t : t = 18.6) (h_c₀ : c₀ = 3.0) (h_cᵢ : cᵢ = 0.4) : 
  ∃ d : ℝ, d = 8 := 
by 
  sorry

end taxi_ride_distance_l191_191807


namespace range_of_a_l191_191328

theorem range_of_a
    (a : ℝ)
    (h : ∀ x y : ℝ, (x - a) ^ 2 + (y - a) ^ 2 = 4 → x ^ 2 + y ^ 2 = 1) :
    a ∈ (Set.Ioo (-(3 * Real.sqrt 2 / 2)) (-(Real.sqrt 2 / 2)) ∪ Set.Ioo (Real.sqrt 2 / 2) (3 * Real.sqrt 2 / 2)) :=
by
  sorry

end range_of_a_l191_191328


namespace interval_between_births_l191_191914

def youngest_child_age : ℕ := 6

def sum_of_ages (I : ℝ) : ℝ :=
  youngest_child_age + (youngest_child_age + I) + (youngest_child_age + 2 * I) + (youngest_child_age + 3 * I) + (youngest_child_age + 4 * I)

theorem interval_between_births : ∃ (I : ℝ), sum_of_ages I = 60 ∧ I = 3.6 := 
by
  sorry

end interval_between_births_l191_191914


namespace cookies_sum_l191_191278

theorem cookies_sum (C : ℕ) (h1 : C % 6 = 5) (h2 : C % 9 = 7) (h3 : C < 80) :
  C = 29 :=
by sorry

end cookies_sum_l191_191278


namespace length_of_train_l191_191972

def speed_kmh : ℝ := 162
def time_seconds : ℝ := 2.222044458665529
def speed_ms : ℝ := 45  -- from conversion: 162 * (1000 / 3600)

theorem length_of_train :
  (speed_kmh * (1000 / 3600)) * time_seconds = 100 := by
  -- Proof is left out
  sorry 

end length_of_train_l191_191972


namespace equation_three_no_real_roots_l191_191309

theorem equation_three_no_real_roots
  (a₁ a₂ a₃ : ℝ)
  (h₁ : a₁^2 - 4 ≥ 0)
  (h₂ : a₂^2 - 8 < 0)
  (h₃ : a₂^2 = a₁ * a₃) :
  a₃^2 - 16 < 0 :=
sorry

end equation_three_no_real_roots_l191_191309


namespace simplify_root_product_l191_191360

theorem simplify_root_product : 
  (625:ℝ)^(1/4) * (125:ℝ)^(1/3) = 25 :=
by
  have h₁ : (625:ℝ) = 5^4 := by norm_num
  have h₂ : (125:ℝ) = 5^3 := by norm_num
  rw [h₁, h₂]
  norm_num
  sorry

end simplify_root_product_l191_191360


namespace last_four_digits_of_5_pow_9000_l191_191659

theorem last_four_digits_of_5_pow_9000 (h : 5^300 ≡ 1 [MOD 1250]) : 
  5^9000 ≡ 1 [MOD 1250] :=
sorry

end last_four_digits_of_5_pow_9000_l191_191659


namespace apples_eaten_l191_191413

-- Define the number of apples eaten by Anna on Tuesday
def apples_eaten_on_Tuesday : ℝ := 4

theorem apples_eaten (A : ℝ) (h1 : A = apples_eaten_on_Tuesday) 
                      (h2 : 2 * A = 2 * apples_eaten_on_Tuesday) 
                      (h3 : A / 2 = apples_eaten_on_Tuesday / 2) 
                      (h4 : A + (2 * A) + (A / 2) = 14) : 
  A = 4 :=
by {
  sorry
}

end apples_eaten_l191_191413


namespace prime_number_conditions_l191_191250

theorem prime_number_conditions :
  ∃ p n : ℕ, Prime p ∧ p = n^2 + 9 ∧ p = (n+1)^2 - 8 :=
by
  sorry

end prime_number_conditions_l191_191250


namespace total_money_found_l191_191383

def value_of_quarters (n_quarters : ℕ) : ℝ := n_quarters * 0.25
def value_of_dimes (n_dimes : ℕ) : ℝ := n_dimes * 0.10
def value_of_nickels (n_nickels : ℕ) : ℝ := n_nickels * 0.05
def value_of_pennies (n_pennies : ℕ) : ℝ := n_pennies * 0.01

theorem total_money_found (n_quarters n_dimes n_nickels n_pennies : ℕ) :
  n_quarters = 10 →
  n_dimes = 3 →
  n_nickels = 4 →
  n_pennies = 200 →
  value_of_quarters n_quarters + value_of_dimes n_dimes + value_of_nickels n_nickels + value_of_pennies n_pennies = 5.00 := 
by
  intros h_quarters h_dimes h_nickels h_pennies
  sorry

end total_money_found_l191_191383


namespace angles_equal_sixty_degrees_l191_191173

/-- Given a triangle ABC with sides a, b, c and respective angles α, β, γ, and with circumradius R,
if the following equation holds:
    (a * cos α + b * cos β + c * cos γ) / (a * sin β + b * sin γ + c * sin α) = (a + b + c) / (9 * R),
prove that α = β = γ = 60 degrees. -/
theorem angles_equal_sixty_degrees 
  (a b c R : ℝ) 
  (α β γ : ℝ) 
  (h : (a * Real.cos α + b * Real.cos β + c * Real.cos γ) / (a * Real.sin β + b * Real.sin γ + c * Real.sin α) = (a + b + c) / (9 * R)) :
  α = 60 ∧ β = 60 ∧ γ = 60 := 
sorry

end angles_equal_sixty_degrees_l191_191173


namespace abs_y_lt_inequality_sum_l191_191960

-- Problem (1)
theorem abs_y_lt {
  x y : ℝ
} (h1 : |x - y| < 1) (h2 : |2 * x + y| < 1) :
  |y| < 1 := by
  sorry

-- Problem (2)
theorem inequality_sum {
  a b c d : ℝ
} (h1 : a > b) (h2 : b > c) (h3 : c > d) :
  (1 / (a - b) + 1 / (b - c) + 1 / (c - d)) ≥ 9 / (a - d) := by
  sorry

end abs_y_lt_inequality_sum_l191_191960


namespace fraction_equation_correct_l191_191230

theorem fraction_equation_correct : (1 / 2 - 1 / 6) / (1 / 6009) = 2003 := by
  sorry

end fraction_equation_correct_l191_191230


namespace benny_pays_l191_191003

theorem benny_pays (cost_per_lunch : ℕ) (number_of_people : ℕ) (total_cost : ℕ) :
  cost_per_lunch = 8 → number_of_people = 3 → total_cost = number_of_people * cost_per_lunch → total_cost = 24 :=
by
  intros h₁ h₂ h₃
  rw [h₁, h₂] at h₃
  exact h₃

end benny_pays_l191_191003


namespace solve_real_eq_l191_191840

theorem solve_real_eq (x : ℝ) (h₁ : x ≠ 2) (h₂ : x ≠ -2) :
  (x = (23 + Real.sqrt 145) / 6 ∨ x = (23 - Real.sqrt 145) / 6) ↔
  ((x ^ 3 - 3 * x ^ 2) / (x ^ 2 - 4) + 2 * x = -16) :=
by sorry

end solve_real_eq_l191_191840


namespace total_cleaning_validation_l191_191486

-- Define the cleaning frequencies and their vacations
def Michael_bath_week := 2
def Michael_shower_week := 1
def Michael_vacation_weeks := 3

def Angela_shower_day := 1
def Angela_vacation_weeks := 2

def Lucy_bath_week := 3
def Lucy_shower_week := 2
def Lucy_alter_weeks := 4
def Lucy_alter_shower_day := 1
def Lucy_alter_bath_week := 1

def weeks_year := 52
def days_week := 7

-- Calculate Michael's total cleaning times in a year
def Michael_total := (Michael_bath_week * weeks_year) + (Michael_shower_week * weeks_year)
def Michael_vacation_reduction := Michael_vacation_weeks * (Michael_bath_week + Michael_shower_week)
def Michael_cleaning_times := Michael_total - Michael_vacation_reduction

-- Calculate Angela's total cleaning times in a year
def Angela_total := (Angela_shower_day * days_week * weeks_year)
def Angela_vacation_reduction := Angela_vacation_weeks * (Angela_shower_day * days_week)
def Angela_cleaning_times := Angela_total - Angela_vacation_reduction

-- Calculate Lucy's total cleaning times in a year
def Lucy_baths_total := Lucy_bath_week * weeks_year
def Lucy_showers_total := Lucy_shower_week * weeks_year
def Lucy_alter_showers := Lucy_alter_shower_day * days_week * Lucy_alter_weeks
def Lucy_alter_baths_reduction := (Lucy_bath_week - Lucy_alter_bath_week) * Lucy_alter_weeks
def Lucy_cleaning_times := Lucy_baths_total + Lucy_showers_total + Lucy_alter_showers - Lucy_alter_baths_reduction

-- Calculate the total times they clean themselves in 52 weeks
def total_cleaning_times := Michael_cleaning_times + Angela_cleaning_times + Lucy_cleaning_times

-- The proof statement
theorem total_cleaning_validation : total_cleaning_times = 777 :=
by simp [Michael_cleaning_times, Angela_cleaning_times, Lucy_cleaning_times, total_cleaning_times]; sorry

end total_cleaning_validation_l191_191486


namespace coloring_methods_390_l191_191247

def numColoringMethods (colors cells : ℕ) (maxColors : ℕ) : ℕ :=
  if colors = 6 ∧ cells = 4 ∧ maxColors = 3 then 390 else 0

theorem coloring_methods_390 :
  numColoringMethods 6 4 3 = 390 :=
by 
  sorry

end coloring_methods_390_l191_191247


namespace find_a_l191_191169

theorem find_a 
  (a : ℝ)
  (h : ∀ n : ℕ, (n.choose 2) * 2^(5-2) * a^2 = 80 → n = 5) :
  a = 1 ∨ a = -1 :=
by
  sorry

end find_a_l191_191169


namespace sphere_surface_area_l191_191172

theorem sphere_surface_area (a b c : ℝ)
  (h1 : a * b * c = Real.sqrt 6)
  (h2 : a * b = Real.sqrt 2)
  (h3 : b * c = Real.sqrt 3) :
  4 * Real.pi * (Real.sqrt (a^2 + b^2 + c^2) / 2) ^ 2 = 6 * Real.pi :=
sorry

end sphere_surface_area_l191_191172


namespace line_l_passes_through_fixed_point_intersecting_lines_find_k_l191_191589

-- Define the lines
def line_l (k : ℝ) (x y : ℝ) : Prop := k * x - y + 1 + 2 * k = 0
def line_l1 (x y : ℝ) : Prop := 2 * x + 3 * y + 8 = 0
def line_l2 (x y : ℝ) : Prop := x - y - 1 = 0

-- 1. Prove line l passes through the point (-2, 1)
theorem line_l_passes_through_fixed_point (k : ℝ) :
  line_l k (-2) 1 :=
by sorry

-- 2. Given lines l, l1, and l2 intersect at a single point, find k
theorem intersecting_lines_find_k (k : ℝ) :
  (∃ x y : ℝ, line_l k x y ∧ line_l1 x y ∧ line_l2 x y) ↔ k = -3 :=
by sorry

end line_l_passes_through_fixed_point_intersecting_lines_find_k_l191_191589


namespace radian_measure_of_neg_300_degrees_l191_191519

theorem radian_measure_of_neg_300_degrees : (-300 : ℝ) * (Real.pi / 180) = -5 * Real.pi / 3 :=
by
  sorry

end radian_measure_of_neg_300_degrees_l191_191519


namespace find_cake_box_width_l191_191829

-- Define the dimensions of the carton
def carton_length := 25
def carton_width := 42
def carton_height := 60
def carton_volume := carton_length * carton_width * carton_height

-- Define the dimensions of the cake box
def cake_box_length := 8
variable (cake_box_width : ℝ) -- This is the unknown width we need to find
def cake_box_height := 5
def cake_box_volume := cake_box_length * cake_box_width * cake_box_height

-- Maximum number of cake boxes that can be placed in the carton
def max_cake_boxes := 210
def total_cake_boxes_volume := max_cake_boxes * cake_box_volume cake_box_width

-- Theorem to prove
theorem find_cake_box_width : cake_box_width = 7.5 :=
by
  sorry

end find_cake_box_width_l191_191829


namespace factorize_m_minimize_ab_find_abc_l191_191866

-- Problem 1: Factorization
theorem factorize_m (m : ℝ) : m^2 - 6 * m + 5 = (m - 1) * (m - 5) :=
sorry

-- Problem 2: Minimization
theorem minimize_ab (a b : ℝ) (h1 : (a - 2)^2 ≥ 0) (h2 : (b + 5)^2 ≥ 0) :
  ∃ (a b : ℝ), (a - 2)^2 + (b + 5)^2 + 4 = 4 ∧ a = 2 ∧ b = -5 :=
sorry

-- Problem 3: Value of a + b + c
theorem find_abc (a b c : ℝ) (h1 : a - b = 8) (h2 : a * b + c^2 - 4 * c + 20 = 0) :
  a + b + c = 2 :=
sorry

end factorize_m_minimize_ab_find_abc_l191_191866


namespace divisors_of_90_l191_191711

def num_pos_divisors (n : ℕ) : ℕ :=
  let factors := if n = 90 then [(2, 1), (3, 2), (5, 1)] else []
  factors.foldl (fun acc (p, k) => acc * (k + 1)) 1

theorem divisors_of_90 : num_pos_divisors 90 = 12 := by
  sorry

end divisors_of_90_l191_191711


namespace robin_made_more_cupcakes_l191_191571

theorem robin_made_more_cupcakes (initial final sold made: ℕ)
  (h1 : initial = 42)
  (h2 : sold = 22)
  (h3 : final = 59)
  (h4 : initial - sold + made = final) :
  made = 39 :=
  sorry

end robin_made_more_cupcakes_l191_191571


namespace boxes_per_case_l191_191358

-- Define the conditions
def total_boxes : ℕ := 54
def total_cases : ℕ := 9

-- Define the result we want to prove
theorem boxes_per_case : total_boxes / total_cases = 6 := 
by sorry

end boxes_per_case_l191_191358


namespace systematic_sampling_seventeenth_group_l191_191816

theorem systematic_sampling_seventeenth_group :
  ∀ (total_students : ℕ) (sample_size : ℕ) (first_number : ℕ) (interval : ℕ),
  total_students = 800 →
  sample_size = 50 →
  first_number = 8 →
  interval = total_students / sample_size →
  first_number + 16 * interval = 264 :=
by
  intros total_students sample_size first_number interval h1 h2 h3 h4
  sorry

end systematic_sampling_seventeenth_group_l191_191816


namespace cleaning_task_sequences_correct_l191_191054

section ChemistryClass

-- Total number of students
def total_students : ℕ := 15

-- Number of classes in a week
def classes_per_week : ℕ := 5

-- Calculate the number of valid sequences of task assignments
def num_valid_sequences : ℕ := total_students * (total_students - 1) * (total_students - 2) * (total_students - 3) * (total_students - 4)

theorem cleaning_task_sequences_correct :
  num_valid_sequences = 360360 :=
by
  unfold num_valid_sequences
  norm_num
  sorry

end ChemistryClass

end cleaning_task_sequences_correct_l191_191054


namespace find_b_l191_191210

-- Define the conditions as constants
def x := 36 -- angle a in degrees
def y := 44 -- given
def z := 52 -- given
def w := 48 -- angle b we need to find

-- Define the problem as a theorem
theorem find_b : x + w + y + z = 180 :=
by
  -- Substitute the given values and show the sum
  have h : 36 + 48 + 44 + 52 = 180 := by norm_num
  exact h

end find_b_l191_191210


namespace ellipse_minor_axis_length_l191_191271

open Real

theorem ellipse_minor_axis_length :
  ∃ (c : ℝ × ℝ) (a b : ℝ), 
    a > 0 ∧ b > 0 ∧
    (∀ (x y : ℝ), (x, y) ∈ {(0,0), (0,4), (2,0), (2,4), (-1,2)} → 
     ((x - c.1)^2 / a^2 + (y - c.2)^2 / b^2 = 1)) ∧
    b = 4/sqrt(3) ∧ 2 * b = 8 * sqrt(3) / 3
    :=
sorry

end ellipse_minor_axis_length_l191_191271


namespace polynomial_simplification_l191_191494

variable (x : ℝ)

theorem polynomial_simplification : 
  ((3 * x - 2) * (5 * x ^ 12 + 3 * x ^ 11 - 4 * x ^ 9 + x ^ 8)) = 
  (15 * x ^ 13 - x ^ 12 - 6 * x ^ 11 - 12 * x ^ 10 + 11 * x ^ 9 - 2 * x ^ 8) := by
  sorry

end polynomial_simplification_l191_191494


namespace cookies_remaining_percentage_l191_191097

theorem cookies_remaining_percentage: 
  ∀ (total initial_remaining eduardo_remaining final_remaining: ℕ),
  total = 600 → 
  initial_remaining = total - (2 * total / 5) → 
  eduardo_remaining = initial_remaining - (3 * initial_remaining / 5) → 
  final_remaining = eduardo_remaining → 
  (final_remaining * 100) / total = 24 := 
by
  intros total initial_remaining eduardo_remaining final_remaining h_total h_initial_remaining h_eduardo_remaining h_final_remaining
  sorry

end cookies_remaining_percentage_l191_191097


namespace decreasing_function_iff_m_eq_2_l191_191089

theorem decreasing_function_iff_m_eq_2 
    (m : ℝ) : 
    (∀ x : ℝ, 0 < x → (m^2 - m - 1) * x^(-5*m - 3) < (m^2 - m - 1) * (x + 1)^(-5*m - 3)) ↔ m = 2 := 
sorry

end decreasing_function_iff_m_eq_2_l191_191089


namespace square_of_binomial_l191_191653

theorem square_of_binomial (k : ℝ) : (∃ a : ℝ, x^2 - 20 * x + k = (x - a)^2) → k = 100 :=
by {
  sorry
}

end square_of_binomial_l191_191653


namespace find_a_in_terms_of_x_l191_191366

theorem find_a_in_terms_of_x (a b x : ℝ) (h₁ : a ≠ b) (h₂ : a^3 - b^3 = 22 * x^3) (h₃ : a - b = 2 * x) : 
  a = x * (1 + (Real.sqrt (40 / 3)) / 2) ∨ a = x * (1 - (Real.sqrt (40 / 3)) / 2) :=
by
  sorry

end find_a_in_terms_of_x_l191_191366


namespace total_monthly_sales_l191_191108

-- Definitions and conditions
def num_customers_per_month : ℕ := 500
def lettuce_per_customer : ℕ := 2
def price_per_lettuce : ℕ := 1
def tomatoes_per_customer : ℕ := 4
def price_per_tomato : ℕ := 1 / 2

-- Statement to prove
theorem total_monthly_sales : num_customers_per_month * (lettuce_per_customer * price_per_lettuce + tomatoes_per_customer * price_per_tomato) = 2000 := 
by 
  sorry

end total_monthly_sales_l191_191108


namespace minimum_value_quadratic_expression_l191_191568

noncomputable def quadratic_expression (x y : ℝ) : ℝ :=
  3 * x^2 + 4 * x * y + 2 * y^2 - 6 * x + 8 * y + 9

theorem minimum_value_quadratic_expression :
  ∃ (x y : ℝ), quadratic_expression x y = -15 ∧
    ∀ (a b : ℝ), quadratic_expression a b ≥ -15 :=
by sorry

end minimum_value_quadratic_expression_l191_191568


namespace highest_probability_highspeed_rail_l191_191332

def total_balls : ℕ := 10
def beidou_balls : ℕ := 3
def tianyan_balls : ℕ := 2
def highspeed_rail_balls : ℕ := 5

theorem highest_probability_highspeed_rail :
  (highspeed_rail_balls : ℚ) / total_balls > (beidou_balls : ℚ) / total_balls ∧
  (highspeed_rail_balls : ℚ) / total_balls > (tianyan_balls : ℚ) / total_balls :=
by {
  -- Proof skipped
  sorry
}

end highest_probability_highspeed_rail_l191_191332


namespace leah_earned_initially_l191_191607

noncomputable def initial_money (x : ℝ) : Prop :=
  let amount_after_milkshake := (6 / 7) * x
  let amount_left_wallet := (3 / 7) * x
  amount_left_wallet = 12

theorem leah_earned_initially (x : ℝ) (h : initial_money x) : x = 28 :=
by
  sorry

end leah_earned_initially_l191_191607


namespace total_daisies_sold_l191_191009

-- Conditions Definitions
def first_day_sales : ℕ := 45
def second_day_sales : ℕ := first_day_sales + 20
def third_day_sales : ℕ := 2 * second_day_sales - 10
def fourth_day_sales : ℕ := 120

-- Question: Prove that the total sales over the four days is 350.
theorem total_daisies_sold :
  first_day_sales + second_day_sales + third_day_sales + fourth_day_sales = 350 := by
  sorry

end total_daisies_sold_l191_191009


namespace max_value_fraction_ratio_tangent_line_through_point_l191_191441

theorem max_value_fraction_ratio (x y : ℝ) (h : x^2 + y^2 - 2*x - 2 = 0) :
  ∃ (max_value : ℝ), max_value = 2 + sqrt 6 :=
sorry

theorem tangent_line_through_point (x y : ℝ) (h : x^2 + y^2 - 2*x - 2 = 0) :
  ∃ (m : ℝ), m ≠ 0 ∧ (x, y) = (0, sqrt 2) → ∀ (x' y' : ℝ), y' = m * x' + sqrt 2 → x' - sqrt 2 * y' + 2 = 0 :=
sorry

end max_value_fraction_ratio_tangent_line_through_point_l191_191441


namespace negation_of_universal_l191_191086

theorem negation_of_universal:
  ¬(∀ x : ℕ, x^2 > 1) ↔ ∃ x : ℕ, x^2 ≤ 1 :=
by sorry

end negation_of_universal_l191_191086


namespace rational_product_nonpositive_l191_191177

open Classical

theorem rational_product_nonpositive (a b : ℚ) (ha : |a| = a) (hb : |b| ≠ b) : a * b ≤ 0 :=
by
  sorry

end rational_product_nonpositive_l191_191177


namespace find_integer_N_l191_191013

theorem find_integer_N : ∃ N : ℤ, (N ^ 2 ≡ N [ZMOD 10000]) ∧ (N - 2 ≡ 0 [ZMOD 7]) :=
by
  sorry

end find_integer_N_l191_191013


namespace length_of_PQ_l191_191470

theorem length_of_PQ
  (k : ℝ) -- height of the trapezoid
  (PQ RU : ℝ) -- sides of trapezoid PQRU
  (A1 : ℝ := (PQ * k) / 2) -- area of triangle PQR
  (A2 : ℝ := (RU * k) / 2) -- area of triangle PUR
  (ratio_A1_A2 : A1 / A2 = 5 / 2) -- given ratio of areas
  (sum_PQ_RU : PQ + RU = 180) -- given sum of PQ and RU
  : PQ = 900 / 7 :=
by
  sorry

end length_of_PQ_l191_191470


namespace factor_of_5_in_20_fact_l191_191802

noncomputable def fact : ℕ → ℕ
| 0       => 1
| (n + 1) => (n + 1) * fact n

theorem factor_of_5_in_20_fact (n : ℕ) (hn : 0 < n) :
  (5^n ∣ fact 20) ∧ ¬ (5^(n + 1) ∣ fact 20) → n = 4 :=
by
  sorry

end factor_of_5_in_20_fact_l191_191802


namespace max_value_y_l191_191443

theorem max_value_y :
  ∀ x: ℝ, (0 ≤ x ∧ x ≤ 2) → 2^(2 * x - 1) - 3 * 2^x + 5 ≤ 5 / 2 := 
by
  intros x h
  sorry

end max_value_y_l191_191443


namespace required_circle_equation_l191_191841

def circle1 (x y : ℝ) : Prop := x^2 + y^2 - x + y - 2 = 0
def circle2 (x y : ℝ) : Prop := x^2 + y^2 = 5
def line_condition (x y : ℝ) : Prop := 3 * x + 4 * y - 1 = 0
def intersection_points (x y : ℝ) : Prop := circle1 x y ∧ circle2 x y

theorem required_circle_equation : 
  ∃ (h : ℝ × ℝ → Prop), 
    (∀ p, intersection_points p.1 p.2 → h p.1 p.2) ∧ 
    (∃ cx cy r, (∀ x y, h x y ↔ (x - cx)^2 + (y - cy)^2 = r^2) ∧ line_condition cx cy ∧ h x y = ((x + 1)^2 + (y - 1)^2 = 13)) 
:= sorry

end required_circle_equation_l191_191841


namespace largest_expression_l191_191996

def P : ℕ := 3 * 2024 ^ 2025
def Q : ℕ := 2024 ^ 2025
def R : ℕ := 2023 * 2024 ^ 2024
def S : ℕ := 3 * 2024 ^ 2024
def T : ℕ := 2024 ^ 2024
def U : ℕ := 2024 ^ 2023

theorem largest_expression : 
  (P - Q) > (Q - R) ∧ 
  (P - Q) > (R - S) ∧ 
  (P - Q) > (S - T) ∧ 
  (P - Q) > (T - U) :=
by sorry

end largest_expression_l191_191996


namespace probability_of_defective_product_is_0_032_l191_191547

-- Defining the events and their probabilities
def P_H1 : ℝ := 0.30
def P_H2 : ℝ := 0.25
def P_H3 : ℝ := 0.45

-- Defining the probabilities of defects given each production line
def P_A_given_H1 : ℝ := 0.03
def P_A_given_H2 : ℝ := 0.02
def P_A_given_H3 : ℝ := 0.04

-- Summing up the total probabilities
def P_A : ℝ :=
  P_H1 * P_A_given_H1 +
  P_H2 * P_A_given_H2 +
  P_H3 * P_A_given_H3

-- The statement to be proven
theorem probability_of_defective_product_is_0_032 :
  P_A = 0.032 :=
by
  -- Proof would go here
  sorry

end probability_of_defective_product_is_0_032_l191_191547


namespace fixed_monthly_costs_l191_191533

theorem fixed_monthly_costs
  (cost_per_component : ℕ) (shipping_cost : ℕ) 
  (num_components : ℕ) (selling_price : ℚ)
  (F : ℚ) :
  cost_per_component = 80 →
  shipping_cost = 6 →
  num_components = 150 →
  selling_price = 196.67 →
  F = (num_components * selling_price) - (num_components * (cost_per_component + shipping_cost)) →
  F = 16600.5 :=
by
  intros
  sorry

end fixed_monthly_costs_l191_191533


namespace triangle_max_area_l191_191579

noncomputable def max_area_triangle (a b c : ℝ) := 
  (1/2) * b * c * Real.sin (Real.arccos ((b^2 + c^2 - a^2) / (2 * b * c)))

theorem triangle_max_area 
  (a b c : ℝ)
  (h_a : a = Real.sqrt 2)
  (h_bc : b^2 - c^2 = 6)
  : max_area_triangle a b c = Real.sqrt 2 :=
begin
  sorry
end

end triangle_max_area_l191_191579


namespace sabrina_total_leaves_l191_191355

theorem sabrina_total_leaves (num_basil num_sage num_verbena : ℕ)
  (h1 : num_basil = 2 * num_sage)
  (h2 : num_sage = num_verbena - 5)
  (h3 : num_basil = 12) :
  num_sage + num_verbena + num_basil = 29 :=
by
  sorry

end sabrina_total_leaves_l191_191355


namespace ratio_equilateral_triangle_l191_191943

def equilateral_triangle_ratio (s : ℝ) (h : s = 10) : ℝ :=
  let altitude := s * (Real.sqrt 3 / 2)
  let area := (1 / 2) * s * altitude
  let perimeter := 3 * s in
  area / perimeter -- this simplifies to 25\sqrt{3} / 30 or 5\sqrt{3} / 6

theorem ratio_equilateral_triangle : ∀ (s : ℝ), s = 10 → equilateral_triangle_ratio s (by assumption) = 5 * (Real.sqrt 3) / 6 :=
by
  intros s h
  rw h
  sorry

end ratio_equilateral_triangle_l191_191943


namespace total_seashells_l191_191747

-- Conditions
def sam_seashells : Nat := 18
def mary_seashells : Nat := 47

-- Theorem stating the question and the final answer
theorem total_seashells : sam_seashells + mary_seashells = 65 :=
by
  sorry

end total_seashells_l191_191747


namespace area_of_ABHED_is_54_l191_191723

noncomputable def area_ABHED (DA DE : ℝ) (AB EF : ℝ) (BH : ℝ) (congruent : Bool) : ℝ :=
  if congruent then 96 + (7 * Real.sqrt 89 / 2) - (8 * Real.sqrt 73 / 2) else 0

theorem area_of_ABHED_is_54 : 
  area_ABHED 8 8 12 12 7 true = 54 :=
by
  sorry

end area_of_ABHED_is_54_l191_191723


namespace lcm_of_4_6_10_18_l191_191641

theorem lcm_of_4_6_10_18 : Nat.lcm (Nat.lcm 4 6) (Nat.lcm 10 18) = 180 := by
  sorry

end lcm_of_4_6_10_18_l191_191641


namespace hyperbola_equation_of_focus_and_asymptote_l191_191373

theorem hyperbola_equation_of_focus_and_asymptote :
  ∃ a b : ℝ, a > 0 ∧ b > 0 ∧ (2 * a) ^ 2 + (2 * b) ^ 2 = 25 ∧ b / a = 2 ∧ 
  (∀ x y : ℝ, (y = 2 * x + 10) → (x = -5) ∧ (y = 0)) ∧ 
  (∀ x y : ℝ, (x ^ 2 / 5 - y ^ 2 / 20 = 1)) :=
by
  sorry

end hyperbola_equation_of_focus_and_asymptote_l191_191373


namespace nth_position_equation_l191_191737

theorem nth_position_equation (n : ℕ) (h : n > 0) : 9 * (n - 1) + n = 10 * n - 9 := by
  sorry

end nth_position_equation_l191_191737


namespace find_b_l191_191512

theorem find_b (a b c : ℕ) (h1 : a + b + c = 99) (h2 : a + 6 = b - 6) (h3 : b - 6 = 5 * c) : b = 51 :=
sorry

end find_b_l191_191512


namespace teacher_age_is_45_l191_191368

def avg_age_of_students := 14
def num_students := 30
def avg_age_with_teacher := 15
def num_people_with_teacher := 31

def total_age_of_students := avg_age_of_students * num_students
def total_age_with_teacher := avg_age_with_teacher * num_people_with_teacher

theorem teacher_age_is_45 : (total_age_with_teacher - total_age_of_students = 45) :=
by
  sorry

end teacher_age_is_45_l191_191368


namespace find_number_l191_191948

theorem find_number (x : ℝ) : x + 5 * 12 / (180 / 3) = 61 ↔ x = 60 := by
  sorry

end find_number_l191_191948


namespace pages_written_per_month_l191_191622

theorem pages_written_per_month 
  (d : ℕ) (days_in_month : ℕ) (letters_freq : ℕ) (time_per_letter : ℕ) 
  (time_per_page : ℕ) (long_letter_time_ratio : ℕ) (long_letter_time : ℕ) :
  d = 3 →
  days_in_month = 30 →
  letters_freq = 10 →
  time_per_letter = 20 →
  time_per_page = 10 →
  long_letter_time_ratio = 2 →
  long_letter_time = 80 →
  (days_in_month / letters_freq * time_per_letter / time_per_page) +
  (long_letter_time / (time_per_page * long_letter_time_ratio)) = 24 :=
by
  intros h1 h2 h3 h4 h5 h6 h7
  rw [h2, h3, h4, h5, h6, h7]
  simp
  exact rfl

#eval pages_written_per_month

end pages_written_per_month_l191_191622


namespace zeros_in_Q_l191_191881

def R_k (k : ℕ) : ℤ := (7^k - 1) / 6

def Q : ℤ := (7^30 - 1) / (7^6 - 1)

def count_zeros (n : ℤ) : ℕ := sorry

theorem zeros_in_Q : count_zeros Q = 470588 :=
by sorry

end zeros_in_Q_l191_191881


namespace cubes_end_same_digits_l191_191048

theorem cubes_end_same_digits (a b : ℕ) (h : a % 1000 = b % 1000) : (a^3) % 1000 = (b^3) % 1000 := by
  sorry

end cubes_end_same_digits_l191_191048


namespace union_of_A_B_complement_intersection_l191_191591

def U : Set ℝ := Set.univ

def A : Set ℝ := { x | -x^2 + 2*x + 15 ≤ 0 }

def B : Set ℝ := { x | |x - 5| < 1 }

theorem union_of_A_B :
  A ∪ B = { x | x ≤ -3 ∨ x > 4 } :=
by
  sorry

theorem complement_intersection :
  (U \ A) ∩ B = { x | 4 < x ∧ x < 5 } :=
by
  sorry

end union_of_A_B_complement_intersection_l191_191591


namespace amount_spent_on_food_l191_191736

-- We define the conditions given in the problem
def Mitzi_brought_money : ℕ := 75
def ticket_cost : ℕ := 30
def tshirt_cost : ℕ := 23
def money_left : ℕ := 9

-- Define the total amount Mitzi spent
def total_spent : ℕ := Mitzi_brought_money - money_left

-- Define the combined cost of the ticket and T-shirt
def combined_cost : ℕ := ticket_cost + tshirt_cost

-- The proof goal
theorem amount_spent_on_food : total_spent - combined_cost = 13 := by
  sorry

end amount_spent_on_food_l191_191736


namespace initial_plan_days_l191_191401

-- Define the given conditions in Lean
variables (D : ℕ) -- Initial planned days for completing the job
variables (P : ℕ) -- Number of people initially hired
variables (Q : ℕ) -- Number of people fired
variables (W1 : ℚ) -- Portion of the work done before firing people
variables (D1 : ℕ) -- Days taken to complete W1 portion of work
variables (W2 : ℚ) -- Remaining portion of the work done after firing people
variables (D2 : ℕ) -- Days taken to complete W2 portion of work

-- Conditions from the problem
axiom h1 : P = 10
axiom h2 : Q = 2
axiom h3 : W1 = 1 / 4
axiom h4 : D1 = 20
axiom h5 : W2 = 3 / 4
axiom h6 : D2 = 75

-- The main theorem that proves the total initially planned days were 80
theorem initial_plan_days : D = 80 :=
sorry

end initial_plan_days_l191_191401


namespace power_sum_is_integer_l191_191740

theorem power_sum_is_integer (a : ℝ) (n : ℕ) (h_pos : 0 < n)
  (h_k : ∃ k : ℤ, k = a + 1/a) : 
  ∃ m : ℤ, m = a^n + 1/a^n := 
sorry

end power_sum_is_integer_l191_191740


namespace integer_solutions_l191_191161

theorem integer_solutions (m n : ℤ) (h1 : m * (m + n) = n * 12) (h2 : n * (m + n) = m * 3) :
  (m = 4 ∧ n = 2) :=
by sorry

end integer_solutions_l191_191161


namespace total_integers_at_least_eleven_l191_191716

theorem total_integers_at_least_eleven (n neg_count : ℕ) 
  (h1 : neg_count % 2 = 1)
  (h2 : neg_count ≤ 11) :
  n ≥ 11 := 
sorry

end total_integers_at_least_eleven_l191_191716


namespace probability_both_dice_same_color_l191_191799

-- Definitions according to the conditions
def num_sides_total := 30
def num_red_sides := 6
def num_green_sides := 8
def num_blue_sides := 10
def num_golden_sides := 6

-- Definition of the probability calculation for each color
def probability_same_color (n : ℕ) (total : ℕ) : ℚ := (n * n : ℚ) / (total * total : ℚ)

-- Combined probability of both dice showing the same color
def combined_probability :=
    probability_same_color num_red_sides num_sides_total + 
    probability_same_color num_green_sides num_sides_total + 
    probability_same_color num_blue_sides num_sides_total + 
    probability_same_color num_golden_sides num_sides_total

-- The final theorem statement
theorem probability_both_dice_same_color : combined_probability = 59 / 225 := 
    sorry -- Proof is not required

end probability_both_dice_same_color_l191_191799


namespace subway_train_speed_l191_191092

open Nat

-- Define the speed function
def speed (s : ℕ) : ℕ := s^2 + 2*s

-- Define the theorem to be proved
theorem subway_train_speed (t : ℕ) (ht : 0 ≤ t ∧ t ≤ 7) (h_speed : speed 7 - speed t = 28) : t = 5 :=
by
  sorry

end subway_train_speed_l191_191092


namespace count_valid_pairs_l191_191400

theorem count_valid_pairs : 
  ∃ n : ℕ, n = 3 ∧ ∀ (m n : ℕ), m > n → n ≥ 4 → (m + n) ≤ 40 → (m - n)^2 = m + n → (m, n) ∈ [(10, 6), (15, 10), (21, 15)] := 
by {
  sorry 
}

end count_valid_pairs_l191_191400


namespace merchant_profit_after_discount_l191_191267

/-- A merchant marks his goods up by 40% and then offers a discount of 20% 
on the marked price. Prove that the merchant makes a profit of 12%. -/
theorem merchant_profit_after_discount :
  ∀ (CP MP SP : ℝ),
    CP > 0 →
    MP = CP * 1.4 →
    SP = MP * 0.8 →
    ((SP - CP) / CP) * 100 = 12 :=
by
  intros CP MP SP hCP hMP hSP
  sorry

end merchant_profit_after_discount_l191_191267


namespace new_savings_after_expense_increase_l191_191672

theorem new_savings_after_expense_increase
    (monthly_salary : ℝ)
    (initial_saving_percent : ℝ)
    (expense_increase_percent : ℝ)
    (initial_salary : monthly_salary = 20000)
    (saving_rate : initial_saving_percent = 0.1)
    (increase_rate : expense_increase_percent = 0.1) :
    monthly_salary - (monthly_salary * (1 - initial_saving_percent + (1 - initial_saving_percent) * expense_increase_percent)) = 200 :=
by
  sorry

end new_savings_after_expense_increase_l191_191672


namespace union_of_A_and_B_l191_191484

/-- Let the universal set U = ℝ, and let the sets A = {x | x^2 - x - 2 = 0}
and B = {y | ∃ x, x ∈ A ∧ y = x + 3}. We want to prove that A ∪ B = {-1, 2, 5}.
-/
theorem union_of_A_and_B (U : Set ℝ) (A B : Set ℝ) (A_def : ∀ x, x ∈ A ↔ x^2 - x - 2 = 0)
  (B_def : ∀ y, y ∈ B ↔ ∃ x, x ∈ A ∧ y = x + 3) :
  A ∪ B = {-1, 2, 5} :=
sorry

end union_of_A_and_B_l191_191484


namespace sequence_a_n_derived_conditions_derived_sequence_is_even_l191_191305

-- Statement of the first problem
theorem sequence_a_n_derived_conditions (a : ℕ → ℝ) (b : ℕ → ℝ) (n : ℕ)
  (h1 : b 1 = a n)
  (h2 : ∀ k, 2 ≤ k ∧ k ≤ n → b k = a (k - 1) + a k - b (k - 1))
  (h3 : b 1 = 5 ∧ b 2 = -2 ∧ b 3 = 7 ∧ b 4 = 2):
  a 1 = 2 ∧ a 2 = 1 ∧ a 3 = 4 ∧ a 4 = 5 :=
sorry

-- Statement of the second problem
theorem derived_sequence_is_even (a : ℕ → ℝ) (b : ℕ → ℝ) (c : ℕ → ℝ) (n : ℕ)
  (h_even : n % 2 = 0)
  (h1 : b 1 = a n)
  (h2 : ∀ k, 2 ≤ k ∧ k ≤ n → b k = a (k - 1) + a k - b (k - 1))
  (h3 : c 1 = b n)
  (h4 : ∀ k, 2 ≤ k ∧ k ≤ n → c k = b (k - 1) + b k - c (k - 1)):
  ∀ i, 1 ≤ i ∧ i ≤ n → c i = a i :=
sorry

end sequence_a_n_derived_conditions_derived_sequence_is_even_l191_191305


namespace square_of_binomial_l191_191645

theorem square_of_binomial (k : ℝ) : (∃ b : ℝ, x^2 - 20 * x + k = (x + b)^2) -> k = 100 :=
sorry

end square_of_binomial_l191_191645


namespace cakes_initially_made_l191_191280

variables (sold bought total initial_cakes : ℕ)

theorem cakes_initially_made (h1 : sold = 105) (h2 : bought = 170) (h3 : total = 186) :
  initial_cakes = total - (sold - bought) :=
by
  rw [h1, h2, h3]
  sorry

end cakes_initially_made_l191_191280


namespace question1_question2_l191_191708

section

variable (A B C : Set ℝ)
variable (a : ℝ)

-- Condition 1: A = {x | -1 ≤ x < 3}
def setA : Set ℝ := {x | -1 ≤ x ∧ x < 3}

-- Condition 2: B = {x | 2x - 4 ≥ x - 2}
def setB : Set ℝ := {x | x ≥ 2}

-- Condition 3: C = {x | x ≥ a - 1}
def setC (a : ℝ) : Set ℝ := {x | x ≥ a - 1}

-- Question 1: Prove A ∩ B = {x | 2 ≤ x < 3}
theorem question1 : A = setA → B = setB → A ∩ B = {x | 2 ≤ x ∧ x < 3} :=
by intros hA hB; rw [hA, hB]; sorry

-- Question 2: If B ∪ C = C, prove a ∈ (-∞, 3]
theorem question2 : B = setB → C = setC a → (B ∪ C = C) → a ≤ 3 :=
by intros hB hC hBUC; rw [hB, hC] at hBUC; sorry

end

end question1_question2_l191_191708


namespace ratio_of_second_to_first_l191_191828

theorem ratio_of_second_to_first (A1 A2 A3 : ℕ) (h1 : A1 = 600) (h2 : A3 = A1 + A2 - 400) (h3 : A1 + A2 + A3 = 3200) : A2 / A1 = 2 :=
by
  sorry

end ratio_of_second_to_first_l191_191828


namespace length_of_BC_l191_191974

theorem length_of_BC (b : ℝ) (h : b ^ 4 = 125) : 2 * b = 10 :=
sorry

end length_of_BC_l191_191974


namespace ratio_of_area_to_perimeter_of_equilateral_triangle_l191_191936

theorem ratio_of_area_to_perimeter_of_equilateral_triangle (s : ℕ) : s = 10 → (let A := (s^2 * sqrt 3) / 4, P := 3 * s in A / P = 5 * sqrt 3 / 6) :=
by
  intro h,
  sorry

end ratio_of_area_to_perimeter_of_equilateral_triangle_l191_191936


namespace eight_child_cotton_l191_191336

theorem eight_child_cotton {a_1 a_8 d S_8 : ℕ} 
  (h1 : d = 17)
  (h2 : S_8 = 996)
  (h3 : 8 * a_1 + 28 * d = S_8) :
  a_8 = a_1 + 7 * d → a_8 = 184 := by
  intro h4
  subst_vars
  sorry

end eight_child_cotton_l191_191336


namespace equilateral_triangle_area_to_perimeter_ratio_l191_191933

theorem equilateral_triangle_area_to_perimeter_ratio
  (a : ℝ) (h : a = 10) :
  let altitude := a * (Real.sqrt 3 / 2) in
  let area := (1 / 2) * a * altitude in
  let perimeter := 3 * a in
  area / perimeter = 5 * (Real.sqrt 3) / 6 := 
by
  sorry

end equilateral_triangle_area_to_perimeter_ratio_l191_191933


namespace cross_section_perimeter_l191_191203

-- Define the lengths of the diagonals AC and BD.
def length_AC : ℝ := 8
def length_BD : ℝ := 12

-- Define the perimeter calculation for the cross-section quadrilateral
-- that passes through the midpoint E of AB and is parallel to BD and AC.
theorem cross_section_perimeter :
  let side1 := length_AC / 2
  let side2 := length_BD / 2
  let perimeter := 2 * (side1 + side2)
  perimeter = 20 :=
by
  sorry

end cross_section_perimeter_l191_191203


namespace binomial_12_3_eq_220_l191_191982

theorem binomial_12_3_eq_220 : nat.choose 12 3 = 220 :=
by {
  sorry
}

end binomial_12_3_eq_220_l191_191982


namespace distance_between_cities_l191_191637

noncomputable def speed_a : ℝ := 1 / 10
noncomputable def speed_b : ℝ := 1 / 15
noncomputable def time_to_meet : ℝ := 6
noncomputable def distance_diff : ℝ := 12

theorem distance_between_cities : 
  (time_to_meet * (speed_a + speed_b) = 60) →
  time_to_meet * speed_a - time_to_meet * speed_b = distance_diff →
  time_to_meet * (speed_a + speed_b) = 60 :=
by
  intros h1 h2
  sorry

end distance_between_cities_l191_191637


namespace umbrella_numbers_count_l191_191675

theorem umbrella_numbers_count :
  ∃ (A : ℕ → ℕ → ℕ), (A 2 2 + A 2 3 + A 2 4 + A 2 5 = 40) :=
by
  let A := Nat.perm
  use A
  sorry

end umbrella_numbers_count_l191_191675


namespace students_in_class_l191_191759

theorem students_in_class (n : ℕ) (T : ℕ) 
  (average_age_students : T = 16 * n)
  (staff_age : ℕ)
  (increased_average_age : (T + staff_age) / (n + 1) = 17)
  (staff_age_val : staff_age = 49) : n = 32 := 
by
  sorry

end students_in_class_l191_191759


namespace total_oysters_eaten_l191_191781

/-- Squido eats 200 oysters -/
def Squido_eats := 200

/-- Crabby eats at least twice as many oysters as Squido -/
def Crabby_eats := 2 * Squido_eats

/-- Total oysters eaten by Squido and Crabby -/
theorem total_oysters_eaten : Squido_eats + Crabby_eats = 600 := 
by
  sorry

end total_oysters_eaten_l191_191781


namespace candy_amount_in_peanut_butter_jar_l191_191918

-- Definitions of the candy amounts in each jar
def banana_jar := 43
def grape_jar := banana_jar + 5
def peanut_butter_jar := 4 * grape_jar
def coconut_jar := (3 / 2) * banana_jar

-- The statement we need to prove
theorem candy_amount_in_peanut_butter_jar : peanut_butter_jar = 192 := by
  sorry

end candy_amount_in_peanut_butter_jar_l191_191918


namespace betsy_sewing_l191_191830

-- Definitions of conditions
def total_squares : ℕ := 16 + 16
def sewn_percentage : ℝ := 0.25
def sewn_squares : ℝ := sewn_percentage * total_squares
def squares_left : ℝ := total_squares - sewn_squares

-- Proof that Betsy needs to sew 24 more squares
theorem betsy_sewing : squares_left = 24 := by
  -- Sorry placeholder for the actual proof
  sorry

end betsy_sewing_l191_191830


namespace root_exists_in_interval_l191_191289

noncomputable def f (x : ℝ) : ℝ := x^3 - x - 1

theorem root_exists_in_interval :
  ∃ c : ℝ, 1 < c ∧ c < 2 ∧ f c = 0 := 
sorry

end root_exists_in_interval_l191_191289


namespace tan_function_constants_l191_191148

theorem tan_function_constants (a b : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b)
(h_period : b ≠ 0 ∧ ∃ k : ℤ, b * (3 / 2) = k * π) 
(h_pass : a * Real.tan (b * (π / 4)) = 3) : a * b = 2 * Real.sqrt 3 :=
by 
  sorry

end tan_function_constants_l191_191148


namespace maximal_subset_with_property_A_l191_191142

-- Define property A for a subset S ⊆ {0, 1, 2, ..., 99}
def has_property_A (S : Finset ℕ) : Prop := 
  ∀ a b c : ℕ, (a * 10 + b ∈ S) → (b * 10 + c ∈ S) → False

-- Define the set of integers {0, 1, 2, ..., 99}
def numbers_set := Finset.range 100

-- The main statement to be proven
theorem maximal_subset_with_property_A :
  ∃ S : Finset ℕ, S ⊆ numbers_set ∧ has_property_A S ∧ S.card = 25 := 
sorry

end maximal_subset_with_property_A_l191_191142


namespace cubes_painted_on_one_side_l191_191742

def is_cube_painted_on_one_side (l w h : ℕ) (cube_size : ℕ) : ℕ :=
  let top_bottom := (l - 2) * (w - 2) * 2
  let front_back := (l - 2) * (h - 2) * 2
  let left_right := (w - 2) * (h - 2) * 2
  top_bottom + front_back + left_right

theorem cubes_painted_on_one_side (l w h cube_size : ℕ) (h_l : l = 5) (h_w : w = 4) (h_h : h = 3) (h_cube_size : cube_size = 1) :
  is_cube_painted_on_one_side l w h cube_size = 22 :=
by
  sorry

end cubes_painted_on_one_side_l191_191742


namespace least_y_l191_191792

theorem least_y (y : ℝ) : (2 * y ^ 2 + 7 * y + 3 = 5) → y = -2 :=
sorry

end least_y_l191_191792


namespace find_value_of_complex_fraction_l191_191703

-- Define the imaginary unit i
noncomputable def i : ℂ := Complex.I

-- State the theorem
theorem find_value_of_complex_fraction :
  (1 - 2 * i) / (1 + i) = -1 / 2 - 3 / 2 * i := 
sorry

end find_value_of_complex_fraction_l191_191703


namespace area_of_sector_l191_191674

theorem area_of_sector (r : ℝ) (n : ℝ) (h_r : r = 3) (h_n : n = 120) : 
  (n / 360) * π * r^2 = 3 * π :=
by
  rw [h_r, h_n] -- Plugin in the given values first
  norm_num     -- Normalize numerical expressions
  sorry        -- Placeholder for further simplification if needed. 

end area_of_sector_l191_191674


namespace todd_money_after_repay_l191_191924

-- Definitions as conditions
def borrowed_amount : ℤ := 100
def repay_amount : ℤ := 110
def ingredients_cost : ℤ := 75
def snow_cones_sold : ℤ := 200
def price_per_snow_cone : ℚ := 0.75

-- Function to calculate money left after transactions
def money_left_after_repay (borrowed : ℤ) (repay : ℤ) (cost : ℤ) (sold : ℤ) (price : ℚ) : ℚ :=
  let left_after_cost := borrowed - cost
  let earnings := sold * price
  let total_before_repay := left_after_cost + earnings
  total_before_repay - repay

-- The theorem stating the problem
theorem todd_money_after_repay :
  money_left_after_repay borrowed_amount repay_amount ingredients_cost snow_cones_sold price_per_snow_cone = 65 := 
by
  -- This is a placeholder for the actual proof
  sorry

end todd_money_after_repay_l191_191924


namespace min_time_to_cook_cakes_l191_191673

theorem min_time_to_cook_cakes (cakes : ℕ) (pot_capacity : ℕ) (time_per_side : ℕ) 
  (h1 : cakes = 3) (h2 : pot_capacity = 2) (h3 : time_per_side = 5) : 
  ∃ t, t = 15 := by
  sorry

end min_time_to_cook_cakes_l191_191673


namespace product_even_if_sum_odd_l191_191461

theorem product_even_if_sum_odd (a b : ℤ) (h : (a + b) % 2 = 1) : (a * b) % 2 = 0 :=
sorry

end product_even_if_sum_odd_l191_191461


namespace max_regions_two_convex_polygons_l191_191638

theorem max_regions_two_convex_polygons (M N : ℕ) (hM : M > N) :
    ∃ R, R = 2 * N + 2 := 
sorry

end max_regions_two_convex_polygons_l191_191638


namespace john_can_see_jane_for_45_minutes_l191_191605

theorem john_can_see_jane_for_45_minutes :
  ∀ (john_speed : ℝ) (jane_speed : ℝ) (initial_distance : ℝ) (final_distance : ℝ),
  john_speed = 7 →
  jane_speed = 3 →
  initial_distance = 1 →
  final_distance = 2 →
  (initial_distance / (john_speed - jane_speed) + final_distance / (john_speed - jane_speed)) * 60 = 45 :=
by
  intros john_speed jane_speed initial_distance final_distance
  sorry

end john_can_see_jane_for_45_minutes_l191_191605


namespace total_distance_covered_l191_191542

noncomputable def radius : ℝ := 0.242
noncomputable def circumference : ℝ := 2 * Real.pi * radius
noncomputable def number_of_revolutions : ℕ := 500
noncomputable def total_distance : ℝ := circumference * number_of_revolutions

theorem total_distance_covered :
  total_distance = 760 :=
by
  -- sorry Re-enable this line for the solver to automatically skip the proof 
  sorry

end total_distance_covered_l191_191542


namespace frisbee_total_distance_correct_l191_191548

-- Define the conditions
def bess_distance_per_throw : ℕ := 20 * 2 -- 20 meters out and 20 meters back = 40 meters
def bess_number_of_throws : ℕ := 4
def holly_distance_per_throw : ℕ := 8
def holly_number_of_throws : ℕ := 5

-- Calculate total distances
def bess_total_distance : ℕ := bess_distance_per_throw * bess_number_of_throws
def holly_total_distance : ℕ := holly_distance_per_throw * holly_number_of_throws
def total_distance : ℕ := bess_total_distance + holly_total_distance

-- The proof statement
theorem frisbee_total_distance_correct :
  total_distance = 200 :=
by
  -- proof goes here (we use sorry to skip the proof)
  sorry

end frisbee_total_distance_correct_l191_191548


namespace problem_A_eq_7_problem_A_eq_2012_l191_191683

open Nat

-- Problem statement for A = 7
theorem problem_A_eq_7 (n k : ℕ) :
  (n! + 7 * n = n^k) ↔ ((n, k) = (2, 4) ∨ (n, k) = (3, 3)) :=
sorry

-- Problem statement for A = 2012
theorem problem_A_eq_2012 (n k : ℕ) :
  ¬ (n! + 2012 * n = n^k) :=
sorry

end problem_A_eq_7_problem_A_eq_2012_l191_191683


namespace determine_x_l191_191687

theorem determine_x
  (w : ℤ) (z : ℤ) (y : ℤ) (x : ℤ)
  (h₁ : w = 90)
  (h₂ : z = w + 25)
  (h₃ : y = z + 12)
  (h₄ : x = y + 7) : x = 134 :=
by
  sorry

end determine_x_l191_191687


namespace least_y_l191_191791

theorem least_y (y : ℝ) : (2 * y ^ 2 + 7 * y + 3 = 5) → y = -2 :=
sorry

end least_y_l191_191791


namespace equation_of_chord_line_l191_191315

theorem equation_of_chord_line (m n s t : ℝ)
  (h₀ : m > 0) (h₁ : n > 0) (h₂ : s > 0) (h₃ : t > 0)
  (h₄ : m + n = 3)
  (h₅ : m / s + n / t = 1)
  (h₆ : m < n)
  (h₇ : s + t = 3 + 2 * Real.sqrt 2)
  (h₈ : ∃ x1 x2 y1 y2 : ℝ, 
        (x1 + x2) / 2 = m ∧ (y1 + y2) / 2 = n ∧
        x1 ^ 2 / 4 + y1 ^ 2 / 16 = 1 ∧
        x2 ^ 2 / 4 + y2 ^ 2 / 16 = 1) 
  : 2 * m + n - 4 = 0 := sorry

end equation_of_chord_line_l191_191315


namespace fraction_is_percent_of_y_l191_191330

theorem fraction_is_percent_of_y (y : ℝ) (hy : y > 0) : 
  (2 * y / 5 + 3 * y / 10) / y = 0.7 :=
sorry

end fraction_is_percent_of_y_l191_191330


namespace radian_to_degree_equivalent_l191_191154

theorem radian_to_degree_equivalent : 
  (7 / 12) * (180 : ℝ) = 105 :=
by
  sorry

end radian_to_degree_equivalent_l191_191154


namespace sum_of_roots_eq_36_l191_191757

theorem sum_of_roots_eq_36 :
  (∃ x1 x2 x3 : ℝ, (11 - x1) ^ 3 + (13 - x2) ^ 3 = (24 - 2 * x3) ^ 3 ∧ 
  (11 - x2) ^ 3 + (13 - x3) ^ 3 = (24 - 2 * x1) ^ 3 ∧ 
  (11 - x3) ^ 3 + (13 - x1) ^ 3 = (24 - 2 * x2) ^ 3 ∧
  x1 + x2 + x3 = 36) :=
sorry

end sum_of_roots_eq_36_l191_191757


namespace galaxy_destruction_probability_l191_191121

theorem galaxy_destruction_probability :
  let m := 45853
  let n := 65536
  m + n = 111389 :=
by
  sorry

end galaxy_destruction_probability_l191_191121


namespace find_angle_l191_191014

theorem find_angle (x : ℝ) (h1 : 90 - x = (1/2) * (180 - x)) : x = 90 :=
by
  sorry

end find_angle_l191_191014


namespace carmen_burning_candles_l191_191555

theorem carmen_burning_candles (candle_hours_per_night: ℕ) (nights_per_candle: ℕ) (candles_used: ℕ) (total_nights: ℕ) : 
  candle_hours_per_night = 2 →
  nights_per_candle = 8 / candle_hours_per_night →
  candles_used = 6 →
  total_nights = candles_used * (nights_per_candle / candle_hours_per_night) →
  total_nights = 24 :=
by
  intros h1 h2 h3 h4
  sorry

end carmen_burning_candles_l191_191555


namespace minimum_value_expression_l191_191566

theorem minimum_value_expression :
  ∃ x y : ℝ, ∀ x y : ℝ, 3 * x ^ 2 + 4 * x * y + 2 * y ^ 2 - 6 * x + 8 * y + 9 ≥ -10 :=
by
  sorry

end minimum_value_expression_l191_191566


namespace chess_tournament_participants_l191_191457

theorem chess_tournament_participants (n : ℕ) (h : n * (n - 1) / 2 = 378) : n = 28 :=
sorry

end chess_tournament_participants_l191_191457


namespace slope_angle_at_point_l191_191774

-- Define the function f
def f (x : ℝ) : ℝ := x^3 - 4 * x + 8

-- Define the derivative of the function f
def f' (x : ℝ) : ℝ := 3 * x^2 - 4

-- State the problem: Prove the slope angle at point (1, 5) is 135 degrees
theorem slope_angle_at_point (θ : ℝ) (h : θ = 135) :
    f' 1 = -1 := 
by 
    sorry

end slope_angle_at_point_l191_191774


namespace least_m_lcm_l191_191692

theorem least_m_lcm (m : ℕ) (h : m > 0) : Nat.lcm 15 m = Nat.lcm 42 m → m = 70 := by
  sorry

end least_m_lcm_l191_191692


namespace value_of_k_l191_191650

theorem value_of_k (k : ℕ) : (∃ b : ℕ, x^2 - 20 * x + k = (x + b)^2) → k = 100 := by
  sorry

end value_of_k_l191_191650


namespace John_spending_l191_191255

theorem John_spending
  (X : ℝ)
  (h1 : (1/2) * X + (1/3) * X + (1/10) * X + 10 = X) :
  X = 150 :=
by
  sorry

end John_spending_l191_191255


namespace custom_op_evaluation_l191_191454

def custom_op (x y : ℤ) : ℤ := x * y - 3 * x

theorem custom_op_evaluation : custom_op 6 4 - custom_op 4 6 = -6 :=
by
  sorry

end custom_op_evaluation_l191_191454


namespace cos_double_angle_l191_191216

theorem cos_double_angle (θ : ℝ) (h : Real.sin (Real.pi / 2 + θ) = 1 / 3) :
  Real.cos (2 * θ) = -7 / 9 :=
sorry

end cos_double_angle_l191_191216


namespace max_sqrt_expr_l191_191580

variable {x y z : ℝ}

noncomputable def f (x y z : ℝ) : ℝ := Real.sqrt x + Real.sqrt (2 * y) + Real.sqrt (3 * z)

theorem max_sqrt_expr (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (h : x + y + z = 2) : 
  f x y z ≤ 2 * Real.sqrt 3 := by
  sorry

end max_sqrt_expr_l191_191580


namespace identical_sets_l191_191275

def A : Set ℝ := {x : ℝ | ∃ y : ℝ, y = x^2 + 1}
def B : Set ℝ := {y : ℝ | ∃ x : ℝ, y = x^2 + 1}
def C : Set (ℝ × ℝ) := {(x, y) : ℝ × ℝ | y = x^2 + 1}
def D : Set ℝ := {y : ℝ | 1 ≤ y}

theorem identical_sets : B = D :=
by
  sorry

end identical_sets_l191_191275


namespace midpoint_distance_from_school_l191_191676

def distance_school_kindergarten_km := 1
def distance_school_kindergarten_m := 700
def distance_kindergarten_house_m := 900

theorem midpoint_distance_from_school : 
  (1000 * distance_school_kindergarten_km + distance_school_kindergarten_m + distance_kindergarten_house_m) / 2 = 1300 := 
by
  sorry

end midpoint_distance_from_school_l191_191676


namespace function_d_is_odd_l191_191523

-- Definition of an odd function
def is_odd_function (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f (-x) = -f x

-- Given function
def f (x : ℝ) : ℝ := x^3

-- Proof statement
theorem function_d_is_odd : is_odd_function f := 
by sorry

end function_d_is_odd_l191_191523


namespace option_B_option_D_l191_191437

noncomputable def curve_C (x y : ℝ) : Prop := x^2 + y^2 - 2*x - 2 = 0

-- The maximum value of (y + 1) / (x + 1) is 2 + sqrt(6)
theorem option_B (x y : ℝ) (h : curve_C x y) :
  ∃ k, (y + 1) / (x + 1) = k ∧ k = 2 + Real.sqrt 6 :=
sorry

-- A tangent line through the point (0, √2) on curve C has the equation x - √2 * y + 2 = 0
theorem option_D (h : curve_C 0 (Real.sqrt 2)) :
  ∃ a b c, a * 0 + b * Real.sqrt 2 + c = 0 ∧ c = 2 ∧ a = 1 ∧ b = - Real.sqrt 2 :=
sorry

end option_B_option_D_l191_191437


namespace derivative_at_one_is_three_l191_191030

-- Definition of the function
def f (x : ℝ) := (x - 1)^2 + 3 * (x - 1)

-- The statement of the problem
theorem derivative_at_one_is_three : deriv f 1 = 3 := 
  sorry

end derivative_at_one_is_three_l191_191030


namespace quadratic_has_two_distinct_real_roots_l191_191772

noncomputable def discriminant (a b c : ℝ) : ℝ := b * b - 4 * a * c

theorem quadratic_has_two_distinct_real_roots :
  (∀ x : ℝ, (x + 1) * (x - 1) = 2 * x + 3) → discriminant 1 (-2) (-4) > 0 :=
by
  intro h
  -- conditions from the problem
  let a := 1
  let b := -2
  let c := -4
  -- use the discriminant function directly with the values
  have delta := discriminant a b c
  show delta > 0
  sorry

end quadratic_has_two_distinct_real_roots_l191_191772


namespace rect_tiling_l191_191787

theorem rect_tiling (a b : ℕ) : ∃ (w h : ℕ), w = max 1 (2 * a) ∧ h = 2 * b ∧ (∃ f : ℕ → ℕ → (ℕ × ℕ), ∀ i j, (i < w ∧ j < h → f i j = (a, b))) := sorry

end rect_tiling_l191_191787


namespace discounted_price_l191_191375

theorem discounted_price (P : ℝ) (original_price : ℝ) (discount_rate : ℝ)
  (h1 : original_price = 975)
  (h2 : discount_rate = 0.20)
  (h3 : P = original_price - discount_rate * original_price) : 
  P = 780 := 
by
  sorry

end discounted_price_l191_191375


namespace general_term_of_sequence_l191_191997

noncomputable def harmonic_mean {n : ℕ} (p : Fin n → ℝ) : ℝ :=
  n / (Finset.univ.sum (fun i => p i))

theorem general_term_of_sequence (a : ℕ → ℝ) (h : ∀ n : ℕ, harmonic_mean (fun i : Fin n => a (i + 1)) = 1 / (2 * n - 1))
    (h₂ : ∀ n : ℕ, (Finset.range n).sum a = 2 * n^2 - n) :
  ∀ n : ℕ, a n = 4 * n - 3 := by
  sorry

end general_term_of_sequence_l191_191997


namespace sixth_root_of_unity_l191_191894

/- Constants and Variables -/
variable (p q r s t k : ℂ)
variable (nz_p : p ≠ 0) (nz_q : q ≠ 0) (nz_r : r ≠ 0) (nz_s : s ≠ 0) (nz_t : t ≠ 0)
variable (hk1 : p * k^5 + q * k^4 + r * k^3 + s * k^2 + t * k + p = 0)
variable (hk2 : q * k^5 + r * k^4 + s * k^3 + t * k^2 + p * k + q = 0)

/- Theorem to prove -/
theorem sixth_root_of_unity : k^6 = 1 :=
by sorry

end sixth_root_of_unity_l191_191894


namespace JoggerDifference_l191_191546

theorem JoggerDifference (tyson_joggers alexander_joggers christopher_joggers : ℕ)
  (h1 : christopher_joggers = 20 * tyson_joggers)
  (h2 : christopher_joggers = 80)
  (h3 : alexander_joggers = tyson_joggers + 22) :
  christopher_joggers - alexander_joggers = 54 := by
  sorry

end JoggerDifference_l191_191546


namespace solve_abs_inequality_l191_191301

theorem solve_abs_inequality (x : ℝ) (h : abs ((8 - x) / 4) < 3) : -4 < x ∧ x < 20 := 
  sorry

end solve_abs_inequality_l191_191301


namespace problem1_range_of_f_problem2_range_of_m_l191_191448

noncomputable def f (x : ℝ) : ℝ :=
  (Real.log x / Real.log 2 - 2) * (Real.log x / Real.log 4 - 1/2)

theorem problem1_range_of_f :
  Set.range (fun x => f x) ∩ Set.Icc 1 4 = Set.Icc (-1/8 : ℝ) 1 :=
sorry

theorem problem2_range_of_m :
  ∀ x, x ∈ Set.Icc 4 16 → f x > (m : ℝ) * (Real.log x / Real.log 4) ↔ m < 0 :=
sorry

end problem1_range_of_f_problem2_range_of_m_l191_191448


namespace expected_number_of_edges_same_color_3x3_l191_191715

noncomputable def expected_edges_same_color (board_size : ℕ) (blackened_count : ℕ) : ℚ :=
  let total_pairs := 12       -- 6 horizontal pairs + 6 vertical pairs
  let prob_both_white := 1 / 6
  let prob_both_black := 5 / 18
  let prob_same_color := prob_both_white + prob_both_black
  total_pairs * prob_same_color

theorem expected_number_of_edges_same_color_3x3 :
  expected_edges_same_color 3 5 = 16 / 3 :=
by
  sorry

end expected_number_of_edges_same_color_3x3_l191_191715


namespace number_of_sheep_l191_191374

def ratio_sheep_horses (S H : ℕ) : Prop := S / H = 3 / 7
def horse_food_per_day := 230 -- ounces
def total_food_per_day := 12880 -- ounces

theorem number_of_sheep (S H : ℕ) 
  (h1 : ratio_sheep_horses S H) 
  (h2 : H * horse_food_per_day = total_food_per_day) 
  : S = 24 :=
sorry

end number_of_sheep_l191_191374


namespace range_of_m_l191_191310

-- Define the sets A and B
def setA := {x : ℝ | abs (x - 1) < 2}
def setB (m : ℝ) := {x : ℝ | x >= m}

-- State the theorem
theorem range_of_m : ∀ (m : ℝ), (setA ∩ setB m = setA) → m <= -1 :=
by
  sorry

end range_of_m_l191_191310


namespace no_valid_positive_x_l191_191809

theorem no_valid_positive_x
  (π : Real)
  (R H x : Real)
  (hR : R = 5)
  (hH : H = 10)
  (hx_pos : x > 0) :
  ¬π * (R + x) ^ 2 * H = π * R ^ 2 * (H + x) :=
by
  sorry

end no_valid_positive_x_l191_191809


namespace cornflowers_count_l191_191348

theorem cornflowers_count
  (n k : ℕ)
  (total_flowers : 9 * n + 17 * k = 70)
  (equal_dandelions_daisies : 5 * n = 7 * k) :
  (9 * n - 20 - 14 = 2) ∧ (17 * k - 20 - 14 = 0) :=
by
  sorry

end cornflowers_count_l191_191348


namespace oysters_eaten_l191_191779

-- Define the conditions in Lean
def Squido_oysters : ℕ := 200
def Crabby_oysters (Squido_oysters : ℕ) : ℕ := 2 * Squido_oysters

-- Statement to prove
theorem oysters_eaten (Squido_oysters Crabby_oysters : ℕ) (h1 : Crabby_oysters = 2 * Squido_oysters) : 
  Squido_oysters + Crabby_oysters = 600 :=
by
  sorry

end oysters_eaten_l191_191779


namespace ratio_second_part_l191_191813

theorem ratio_second_part (first_part second_part total : ℕ) 
  (h_ratio_percent : 50 = 100 * first_part / total) 
  (h_first_part : first_part = 10) : 
  second_part = 10 := by
  have h_total : total = 2 * first_part := by sorry
  sorry

end ratio_second_part_l191_191813


namespace vectors_parallel_l191_191040

theorem vectors_parallel (x : ℝ) :
    ∀ (a b : ℝ × ℝ × ℝ),
    a = (2, -1, 3) →
    b = (x, 2, -6) →
    (∃ k : ℝ, b = (k * 2, k * -1, k * 3)) →
    x = -4 :=
by
  intro a b ha hb hab
  sorry

end vectors_parallel_l191_191040


namespace inequality_abc_l191_191444

theorem inequality_abc (a b : ℝ) (ha : a > 0) (hb : b > 0) : 
  a / (b ^ (1/2 : ℝ)) + b / (a ^ (1/2 : ℝ)) ≥ a ^ (1/2 : ℝ) + b ^ (1/2 : ℝ) :=
by { sorry }

end inequality_abc_l191_191444


namespace proportional_function_ratio_l191_191081

-- Let k be a constant, and y = kx be a proportional function.
-- We know that f(1) = 3 and f(a) = b where b ≠ 0.
-- We want to prove that a / b = 1 / 3.

theorem proportional_function_ratio (a b k : ℝ) :
  (∀ x, x = 1 → k * x = 3) →
  (∀ x, x = a → k * x = b) →
  b ≠ 0 →
  a / b = 1 / 3 :=
by
  intros h1 h2 h3
  -- the proof will follow but is not required here
  sorry

end proportional_function_ratio_l191_191081


namespace new_persons_joined_l191_191625

theorem new_persons_joined (initial_avg_age new_avg_age initial_total new_avg_age_total final_avg_age final_total : ℝ) 
  (n_initial n_new : ℕ) 
  (h1 : initial_avg_age = 16)
  (h2 : n_initial = 20)
  (h3 : new_avg_age = 15)
  (h4 : final_avg_age = 15.5)
  (h5 : initial_total = initial_avg_age * n_initial)
  (h6 : new_avg_age_total = new_avg_age * (n_new : ℝ))
  (h7 : final_total = initial_total + new_avg_age_total)
  (h8 : final_total = final_avg_age * (n_initial + n_new)) 
  : n_new = 20 :=
by
  sorry

end new_persons_joined_l191_191625


namespace bacteria_growth_time_l191_191668

theorem bacteria_growth_time : 
  (∀ n : ℕ, 2 ^ n = 4096 → (n * 15) / 60 = 3) :=
by
  sorry

end bacteria_growth_time_l191_191668


namespace water_leaving_rate_l191_191082

-- Definitions: Volume of water and time taken
def volume_of_water : ℕ := 300
def time_taken : ℕ := 25

-- Theorem statement: Rate of water leaving the tank
theorem water_leaving_rate : (volume_of_water / time_taken) = 12 := 
by sorry

end water_leaving_rate_l191_191082


namespace ratio_of_runs_l191_191254

theorem ratio_of_runs (A B C : ℕ) (h1 : B = C / 5) (h2 : A + B + C = 95) (h3 : C = 75) :
  A / B = 1 / 3 :=
by sorry

end ratio_of_runs_l191_191254


namespace product_of_roots_of_cubic_l191_191151

theorem product_of_roots_of_cubic :
  let a := 2
  let d := 18
  let product_of_roots := -(d / a)
  product_of_roots = -9 :=
by
  sorry

end product_of_roots_of_cubic_l191_191151


namespace michael_total_earnings_l191_191225

-- Define the cost of large paintings and small paintings
def large_painting_cost : ℕ := 100
def small_painting_cost : ℕ := 80

-- Define the number of large and small paintings sold
def large_paintings_sold : ℕ := 5
def small_paintings_sold : ℕ := 8

-- Calculate Michael's total earnings
def total_earnings : ℕ := (large_painting_cost * large_paintings_sold) + (small_painting_cost * small_paintings_sold)

-- Prove: Michael's total earnings are 1140 dollars
theorem michael_total_earnings : total_earnings = 1140 := by
  sorry

end michael_total_earnings_l191_191225


namespace todd_final_money_l191_191921

noncomputable def todd_initial_money : ℝ := 100
noncomputable def todd_debt : ℝ := 110
noncomputable def todd_spent_on_ingredients : ℝ := 75
noncomputable def snow_cones_sold : ℝ := 200
noncomputable def price_per_snowcone : ℝ := 0.75

theorem todd_final_money :
  let initial_money := todd_initial_money,
      debt := todd_debt,
      spent := todd_spent_on_ingredients,
      revenue := snow_cones_sold * price_per_snowcone,
      remaining := initial_money - spent,
      total_pre_debt := remaining + revenue,
      final_money := total_pre_debt - debt
  in final_money = 65 :=
by
  sorry

end todd_final_money_l191_191921


namespace ratio_equilateral_triangle_l191_191942

def equilateral_triangle_ratio (s : ℝ) (h : s = 10) : ℝ :=
  let altitude := s * (Real.sqrt 3 / 2)
  let area := (1 / 2) * s * altitude
  let perimeter := 3 * s in
  area / perimeter -- this simplifies to 25\sqrt{3} / 30 or 5\sqrt{3} / 6

theorem ratio_equilateral_triangle : ∀ (s : ℝ), s = 10 → equilateral_triangle_ratio s (by assumption) = 5 * (Real.sqrt 3) / 6 :=
by
  intros s h
  rw h
  sorry

end ratio_equilateral_triangle_l191_191942


namespace radius_of_triangle_DEF_l191_191794

noncomputable def radius_of_inscribed_circle (DE DF EF : ℝ) : ℝ :=
  let s := (DE + DF + EF) / 2
  let K := Real.sqrt (s * (s - DE) * (s - DF) * (s - EF))
  K / s

theorem radius_of_triangle_DEF :
  radius_of_inscribed_circle 26 15 17 = 121 / 29 := by
sorry

end radius_of_triangle_DEF_l191_191794


namespace actual_average_height_l191_191626

theorem actual_average_height 
  (incorrect_avg_height : ℝ)
  (num_students : ℕ)
  (incorrect_height : ℝ)
  (correct_height : ℝ)
  (actual_avg_height : ℝ) :
  incorrect_avg_height = 175 →
  num_students = 20 →
  incorrect_height = 151 →
  correct_height = 111 →
  actual_avg_height = 173 :=
by
  sorry

end actual_average_height_l191_191626


namespace range_of_a_l191_191699

def set1 : Set ℝ := {x | x ≤ 2}
def set2 (a : ℝ) : Set ℝ := {x | x > a}
variable (a : ℝ)

theorem range_of_a (h : set1 ∪ set2 a = Set.univ) : a ≤ 2 :=
by sorry

end range_of_a_l191_191699


namespace seven_books_cost_l191_191100

-- Given condition: Three identical books cost $45
def three_books_cost (cost_per_book : ℤ) := 3 * cost_per_book = 45

-- Question to prove: The cost of seven identical books is $105
theorem seven_books_cost (cost_per_book : ℤ) (h : three_books_cost cost_per_book) : 7 * cost_per_book = 105 := 
sorry

end seven_books_cost_l191_191100


namespace largest_K_inequality_l191_191290

noncomputable def largest_K : ℝ := 18

theorem largest_K_inequality (a b c : ℝ) (h_pos_a : a > 0) (h_pos_b : b > 0) (h_pos_c : c > 0) 
(h_cond : a * b + b * c + c * a = a * b * c) :
( (a^a * (b^2 + c^2)) / ((a^a - 1)^2) + (b^b * (c^2 + a^2)) / ((b^b - 1)^2) + (c^c * (a^2 + b^2)) / ((c^c - 1)^2) )
≥ largest_K * ((a + b + c) / (a * b * c - 1)) ^ 2 :=
sorry

end largest_K_inequality_l191_191290


namespace ratio_of_area_to_perimeter_l191_191932

noncomputable def side_length := 10
noncomputable def altitude := (side_length * (Real.sqrt 3 / 2))
noncomputable def area := (1 / 2) * side_length * altitude
noncomputable def perimeter := 3 * side_length

theorem ratio_of_area_to_perimeter (s : ℝ) (h : ℝ) (A : ℝ) (P : ℝ) 
  (h1 : s = 10) 
  (h2 : h = s * (Real.sqrt 3 / 2)) 
  (h3 : A = (1 / 2) * s * h) 
  (h4 : P = 3 * s) :
  A / P = 5 * Real.sqrt 3 / 6 := by
  sorry

end ratio_of_area_to_perimeter_l191_191932


namespace pastries_total_l191_191418

-- We start by defining the conditions
def Calvin_pastries (Frank_pastries Grace_pastries : ℕ) : ℕ := Frank_pastries + 8
def Phoebe_pastries (Frank_pastries Grace_pastries : ℕ) : ℕ := Frank_pastries + 8
def Grace_pastries : ℕ := 30
def have_same_pastries (Calvin_pastries Phoebe_pastries Grace_pastries : ℕ) : Prop :=
  Calvin_pastries + 5 = Grace_pastries ∧ Phoebe_pastries + 5 = Grace_pastries

-- Total number of pastries held by Calvin, Phoebe, Frank, and Grace
def total_pastries (Frank_pastries Calvin_pastries Phoebe_pastries Grace_pastries : ℕ) : ℕ :=
  Frank_pastries + Calvin_pastries + Phoebe_pastries + Grace_pastries

-- The statement to prove
theorem pastries_total (Frank_pastries : ℕ) : 
  have_same_pastries (Calvin_pastries Frank_pastries Grace_pastries) (Phoebe_pastries Frank_pastries Grace_pastries) Grace_pastries → 
  Frank_pastries + Calvin_pastries Frank_pastries Grace_pastries + Phoebe_pastries Frank_pastries Grace_pastries + Grace_pastries = 97 :=
by
  sorry

end pastries_total_l191_191418


namespace required_circle_equation_l191_191842

-- Define the first circle equation
def circle1 (x y : ℝ) : Prop := x^2 + y^2 - x + y - 2 = 0

-- Define the second circle equation
def circle2 (x y : ℝ) : Prop := x^2 + y^2 = 5

-- Define the line equation on which the center of the required circle lies
def center_line (x y : ℝ) : Prop := 3 * x + 4 * y - 1 = 0

-- State the final proof that the equation of the required circle is (x + 1)^2 + (y - 1)^2 = 13 under the given conditions
theorem required_circle_equation (x y : ℝ) :
  ( ∃ (x1 y1 : ℝ), circle1 x1 y1 ∧ circle2 x1 y1 ∧
    (∃ (cx cy r : ℝ), center_line cx cy ∧ (x - cx)^2 + (y - cy)^2 = r^2 ∧ (x1 - cx)^2 + (y1 - cy)^2 = r^2 ∧
      (x + 1)^2 + (y - 1)^2 = 13) )
 := sorry

end required_circle_equation_l191_191842


namespace complementary_angle_percentage_decrease_l191_191507

theorem complementary_angle_percentage_decrease :
  ∀ (α β : ℝ), α + β = 90 → 6 * α = 3 * β → 
  let α' := 1.2 * α in
  let β' := 90 - α' in
  (100 * (β' / β)) = 90 :=
by
  intros α β h_sum h_ratio α' β'
  have α' := 1.2 * α
  have β' := 90 - α'
  sorry

end complementary_angle_percentage_decrease_l191_191507


namespace parametric_to_ordinary_eq_l191_191769

-- Define the parametric equations and the domain of the parameter t
def parametric_eqns (t : ℝ) : ℝ × ℝ := (t + 1, 3 - t^2)

-- Define the target equation to be proved
def target_eqn (x y : ℝ) : Prop := y = -x^2 + 2*x + 2

-- Prove that, given the parametric equations, the target ordinary equation holds
theorem parametric_to_ordinary_eq :
  ∃ (t : ℝ) (x y : ℝ), parametric_eqns t = (x, y) ∧ target_eqn x y :=
by
  sorry

end parametric_to_ordinary_eq_l191_191769


namespace correct_choice_of_f_l191_191031

def f1 (x : ℝ) : ℝ := (x - 1)^2 + 3 * (x - 1)
def f2 (x : ℝ) : ℝ := 2 * (x - 1)
def f3 (x : ℝ) : ℝ := 2 * (x - 1)^2
def f4 (x : ℝ) : ℝ := x - 1

theorem correct_choice_of_f (h : (deriv f1 1 = 3) ∧ (deriv f2 1 ≠ 3) ∧ (deriv f3 1 ≠ 3) ∧ (deriv f4 1 ≠ 3)) : 
  ∀ f, (f = f1 ∨ f = f2 ∨ f = f3 ∨ f = f4) → (deriv f 1 = 3 → f = f1) :=
by sorry

end correct_choice_of_f_l191_191031


namespace inequality_proof_l191_191074

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) : 
  ab * (a + b) + bc * (b + c) + ac * (a + c) ≥ 6 * abc := 
sorry

end inequality_proof_l191_191074


namespace max_number_of_triangles_l191_191212

theorem max_number_of_triangles (num_sides : ℕ) (num_internal_points : ℕ) 
    (total_points : ℕ) (h1 : num_sides = 13) (h2 : num_internal_points = 200) 
    (h3 : total_points = num_sides + num_internal_points) 
    (h4 : ∀ (x y z : point), x ≠ y ∧ y ≠ z ∧ z ≠ x → ¬ collinear x y z) : 
    (total_points.choose 3) = 411 :=
by
  sorry

end max_number_of_triangles_l191_191212


namespace det_M_pow_three_eq_twenty_seven_l191_191845

-- Define a matrix M
variables (M : Matrix (Fin n) (Fin n) ℝ)

-- Given condition: det M = 3
axiom det_M_eq_3 : Matrix.det M = 3

-- State the theorem we aim to prove
theorem det_M_pow_three_eq_twenty_seven : Matrix.det (M^3) = 27 :=
by
  sorry

end det_M_pow_three_eq_twenty_seven_l191_191845


namespace max_omega_value_l191_191036

noncomputable def f (ω : ℝ) (φ : ℝ) (x : ℝ) : ℝ :=
  Real.sin (ω * x + φ)

theorem max_omega_value 
  (ω : ℝ) 
  (φ : ℝ) 
  (hω : 0 < ω) 
  (hφ : |φ| ≤ Real.pi / 2)
  (h_zero : f ω φ (-Real.pi / 4) = 0)
  (h_sym : f ω φ (Real.pi / 4) = f ω φ (-Real.pi / 4))
  (h_monotonic : ∀ x₁ x₂, (Real.pi / 18) < x₁ → x₁ < x₂ → x₂ < (5 * Real.pi / 36) → f ω φ x₁ < f ω φ x₂) :
  ω = 9 :=
  sorry

end max_omega_value_l191_191036


namespace find_other_number_l191_191376

-- Defining the two numbers and their properties
def sum_is_84 (a b : ℕ) : Prop := a + b = 84
def one_is_36 (a b : ℕ) : Prop := a = 36 ∨ b = 36
def other_is_48 (a b : ℕ) : Prop := a = 48 ∨ b = 48

-- The theorem statement
theorem find_other_number (a b : ℕ) (h1 : sum_is_84 a b) (h2 : one_is_36 a b) : other_is_48 a b :=
by {
  sorry
}

end find_other_number_l191_191376


namespace min_value_m_n_l191_191966

noncomputable def log_a (a x : ℝ) := Real.log x / Real.log a

theorem min_value_m_n 
  (a : ℝ) (m n : ℝ)
  (h_a_pos : a > 0) (h_a_ne1 : a ≠ 1)
  (h_mn_pos : m > 0 ∧ n > 0)
  (h_line_eq : 2 * m + n = 1) :
  m + n = 3 + 2 * Real.sqrt 2 :=
by
  sorry

end min_value_m_n_l191_191966


namespace range_of_c_l191_191702

-- Definitions of the propositions p and q
def p (c : ℝ) : Prop := ∀ x y : ℝ, x < y → c^x > c^y
def q (c : ℝ) : Prop := ∃ x : ℝ, x^2 - c^2 ≤ - (1 / 16)

-- Main theorem
theorem range_of_c (c : ℝ) (h1 : c > 0) (h2 : p c) (h3 : q c) : c ≥ 1 / 4 ∧ c < 1 :=
  sorry

end range_of_c_l191_191702


namespace sum_of_three_numbers_l191_191657

theorem sum_of_three_numbers (a b c : ℝ) 
  (h1 : a + b = 35) 
  (h2 : b + c = 57) 
  (h3 : c + a = 62) : 
  a + b + c = 77 :=
by
  sorry

end sum_of_three_numbers_l191_191657


namespace find_a1_l191_191023

variable (a : ℕ → ℚ) (d : ℚ)
variable (S : ℕ → ℚ)
variable (h_seq : ∀ n, a (n + 1) = a n + d)
variable (h_diff : d ≠ 0)
variable (h_prod : (a 2) * (a 3) = (a 4) * (a 5))
variable (h_sum : S 4 = 27)
variable (h_sum_def : ∀ n, S n = n * (a 1 + a n) / 2)

theorem find_a1 : a 1 = 135 / 8 := by
  sorry

end find_a1_l191_191023


namespace minimum_value_quadratic_expression_l191_191569

noncomputable def quadratic_expression (x y : ℝ) : ℝ :=
  3 * x^2 + 4 * x * y + 2 * y^2 - 6 * x + 8 * y + 9

theorem minimum_value_quadratic_expression :
  ∃ (x y : ℝ), quadratic_expression x y = -15 ∧
    ∀ (a b : ℝ), quadratic_expression a b ≥ -15 :=
by sorry

end minimum_value_quadratic_expression_l191_191569


namespace number_of_divisors_l191_191709

-- Defining the given number and its prime factorization as a condition.
def given_number : ℕ := 90

-- Defining the prime factorization.
def prime_factorization (n : ℕ) : List (ℕ × ℕ) :=
  if n = 90 then [(2, 1), (3, 2), (5, 1)] else []

-- The statement to prove that the number of positive divisors of 90 is 12.
theorem number_of_divisors (n : ℕ) (pf : List (ℕ × ℕ)) :
  n = 90 → pf = [(2, 1), (3, 2), (5, 1)] →
  (pf.map (λ p, p.2 + 1)).prod = 12 :=
by
  intros hn hpf
  rw [hn, hpf]
  simp
  sorry

end number_of_divisors_l191_191709


namespace initial_crayons_count_l191_191380

variable (x : ℕ) -- x represents the initial number of crayons

theorem initial_crayons_count (h1 : x + 3 = 12) : x = 9 := 
by sorry

end initial_crayons_count_l191_191380


namespace largest_r_l191_191493

theorem largest_r (a : ℕ → ℕ) (h : ∀ n, 0 < a n ∧ a n ≤ a (n + 2) ∧ a (n + 2) ≤ Int.sqrt (a n ^ 2 + 2 * a (n + 1))) :
  ∃ M, ∀ n ≥ M, a (n + 2) = a n :=
sorry

end largest_r_l191_191493


namespace greatest_prime_factor_391_l191_191387

theorem greatest_prime_factor_391 : ∃ p, Prime p ∧ p ∣ 391 ∧ ∀ q, Prime q ∧ q ∣ 391 → q ≤ p :=
by
  sorry

end greatest_prime_factor_391_l191_191387


namespace binom_12_3_eq_220_l191_191990

theorem binom_12_3_eq_220 : nat.choose 12 3 = 220 :=
sorry

end binom_12_3_eq_220_l191_191990


namespace hyperbola_properties_l191_191462

theorem hyperbola_properties :
  let h := -3
  let k := 0
  let a := 5
  let c := Real.sqrt 50
  ∃ b : ℝ, a^2 + b^2 = c^2 ∧ h + k + a + b = 7 :=
by
  sorry

end hyperbola_properties_l191_191462


namespace problem_solution_l191_191594

theorem problem_solution (b : ℝ) (i : ℂ) (h : i^2 = -1) (h_cond : (2 - i) * (4 * i) = 4 + b * i) : 
  b = 8 := 
by 
  sorry

end problem_solution_l191_191594


namespace sum_of_x_and_y_l191_191464

-- Define the given angles
def angle_A : ℝ := 34
def angle_B : ℝ := 74
def angle_C : ℝ := 32

-- State the theorem
theorem sum_of_x_and_y (x y : ℝ) :
  (680 - x - y) = 720 → (x + y = 40) :=
by
  intro h
  sorry

end sum_of_x_and_y_l191_191464


namespace betsy_remaining_squares_l191_191831

def total_squares := 16 + 16
def percent_sewn := 0.25
def squares_sewn := total_squares * percent_sewn
def remaining_squares := total_squares - squares_sewn

theorem betsy_remaining_squares : remaining_squares = 24 :=
by
  sorry

end betsy_remaining_squares_l191_191831


namespace find_b_if_continuous_l191_191222

noncomputable def f (x : ℝ) (b : ℝ) : ℝ :=
if x ≤ 2 then 5 * x^2 + 4 else b * x + 1

theorem find_b_if_continuous (b : ℝ) : 
  (∀ ε > 0, ∃ δ > 0, ∀ x, abs (x - 2) < δ → abs (f x b - f 2 b) < ε) ↔ b = 23 / 2 :=
by
  sorry

end find_b_if_continuous_l191_191222


namespace store_A_profit_margin_l191_191916

theorem store_A_profit_margin
  (x y : ℝ)
  (hx : x > 0)
  (hy : y > x)
  (h : (y - x) / x + 0.12 = (y - 0.9 * x) / (0.9 * x)) :
  (y - x) / x = 0.08 :=
by {
  sorry
}

end store_A_profit_margin_l191_191916


namespace point_B_value_l191_191491

theorem point_B_value :
  ∃ B : ℝ, (|B + 1| = 4) ∧ (B = 3 ∨ B = -5) := 
by
  sorry

end point_B_value_l191_191491


namespace LCM_quotient_l191_191608

-- Define M as the least common multiple of integers from 12 to 25
def LCM_12_25 : ℕ := Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm 
                       (Nat.lcm 12 13) 14) 15) 16) 17) (Nat.lcm (Nat.lcm 
                       (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm 18 19) 20) 21) 22) 23) 24)

-- Define N as the least common multiple of LCM_12_25, 36, 38, 40, 42, 44, 45
def N : ℕ := Nat.lcm LCM_12_25 (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm 36 38) 40) 42) (Nat.lcm 44 45))

-- Prove that N / LCM_12_25 = 1
theorem LCM_quotient : N / LCM_12_25 = 1 := by
    sorry

end LCM_quotient_l191_191608


namespace fifth_inequality_l191_191489

theorem fifth_inequality :
  1 + (1 / 2^2) + (1 / 3^2) + (1 / 4^2) + (1 / 5^2) + (1 / 6^2) < 11 / 6 :=
sorry

end fifth_inequality_l191_191489


namespace isosceles_triangle_congruent_l191_191242

theorem isosceles_triangle_congruent (A B C C1 : ℝ) 
(h₁ : A = B) 
(h₂ : C = C1) 
: A = B ∧ C = C1 :=
by
  sorry

end isosceles_triangle_congruent_l191_191242


namespace distance_traveled_downstream_l191_191958

noncomputable def speed_boat : ℝ := 20  -- Speed of the boat in still water in km/hr
noncomputable def rate_current : ℝ := 5  -- Rate of current in km/hr
noncomputable def time_minutes : ℝ := 24  -- Time traveled downstream in minutes
noncomputable def time_hours : ℝ := time_minutes / 60  -- Convert time to hours
noncomputable def effective_speed_downstream : ℝ := speed_boat + rate_current  -- Effective speed downstream

theorem distance_traveled_downstream :
  effective_speed_downstream * time_hours = 10 := by {
  sorry
}

end distance_traveled_downstream_l191_191958


namespace unique_digit_sum_l191_191156

theorem unique_digit_sum (Y M E T : ℕ) (h1 : Y ≠ M) (h2 : Y ≠ E) (h3 : Y ≠ T)
    (h4 : M ≠ E) (h5 : M ≠ T) (h6 : E ≠ T) (h7 : 10 * Y + E = YE) (h8 : 10 * M + E = ME)
    (h9 : YE * ME = T * T * T) (hT_even : T % 2 = 0) : 
    Y + M + E + T = 10 :=
  sorry

end unique_digit_sum_l191_191156


namespace cost_of_one_shirt_l191_191455

-- Definitions based on the conditions given
variables (J S : ℝ)

-- First condition: 3 pairs of jeans and 2 shirts cost $69
def condition1 : Prop := 3 * J + 2 * S = 69

-- Second condition: 2 pairs of jeans and 3 shirts cost $61
def condition2 : Prop := 2 * J + 3 * S = 61

-- The theorem to prove that the cost of one shirt is $9
theorem cost_of_one_shirt (J S : ℝ) (h1 : condition1 J S) (h2 : condition2 J S) : S = 9 :=
by
  sorry

end cost_of_one_shirt_l191_191455


namespace angles_equal_l191_191322

theorem angles_equal (A B C : ℝ) (h1 : A + B = 180) (h2 : B + C = 180) : A = C := sorry

end angles_equal_l191_191322


namespace exponentiation_properties_l191_191833

theorem exponentiation_properties:
  (10^6) * (10^2)^3 / 10^4 = 10^8 :=
by
  sorry

end exponentiation_properties_l191_191833


namespace exists_subsets_S_l191_191732

open Finset

def exists_two_subsets_with_equal_sum (S : Finset ℕ) : Prop :=
  ∃ (x y u v : ℕ), {x, y} ⊆ S ∧ {u, v} ⊆ S ∧ x + y = u + v ∧ {x, y} ≠ {u, v}

theorem exists_subsets_S (S : Finset ℕ) (h1 : S ⊆ range 25) (h2 : S.card = 10) :
  exists_two_subsets_with_equal_sum S :=
by
  sorry

end exists_subsets_S_l191_191732


namespace slope_of_line_l191_191643

theorem slope_of_line (x₁ y₁ x₂ y₂ : ℝ) (h₁ : x₁ = 1) (h₂ : y₁ = 3) (h₃ : x₂ = 4) (h₄ : y₂ = -6) : 
  (y₂ - y₁) / (x₂ - x₁) = -3 := by
  sorry

end slope_of_line_l191_191643


namespace dutch_americans_blue_shirts_window_seats_l191_191738

theorem dutch_americans_blue_shirts_window_seats :
  let total_people := 90
  let dutch_fraction := 3 / 5
  let dutch_americans_fraction := 1 / 2
  let window_seats_fraction := 1 / 3
  let blue_shirts_fraction := 2 / 3
  total_people * dutch_fraction * dutch_americans_fraction * window_seats_fraction * blue_shirts_fraction = 6 := by
  sorry

end dutch_americans_blue_shirts_window_seats_l191_191738


namespace product_of_B_coords_l191_191312

structure Point where
  x : ℝ
  y : ℝ

def isMidpoint (M A B : Point) : Prop :=
  M.x = (A.x + B.x) / 2 ∧ M.y = (A.y + B.y) / 2

theorem product_of_B_coords :
  ∀ (M A B : Point), 
  isMidpoint M A B →
  M = ⟨3, 7⟩ →
  A = ⟨5, 3⟩ →
  (B.x * B.y) = 11 :=
by intro M A B hM hM_def hA_def; sorry

end product_of_B_coords_l191_191312


namespace box_max_volume_l191_191155

theorem box_max_volume (x : ℝ) (h1 : 0 < x) (h2 : x < 5) :
    (10 - 2 * x) * (16 - 2 * x) * x ≤ 144 :=
by
  -- The proof will be filled here
  sorry

end box_max_volume_l191_191155


namespace line_y_intercept_l191_191541

theorem line_y_intercept (x1 y1 x2 y2 : ℝ) (h1 : x1 = 2) (h2 : y1 = -3) (h3 : x2 = 6) (h4 : y2 = 9) :
  ∃ b : ℝ, b = -9 := 
by
  sorry

end line_y_intercept_l191_191541


namespace decreasing_f_range_l191_191185

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 0 then a^x else (a - 3) * x + 4 * a

theorem decreasing_f_range (a : ℝ) :
  (∀ x1 x2 : ℝ, x1 ≠ x2 → (f a x1 - f a x2) / (x1 - x2) < 0) → (0 < a ∧ a ≤ 1/4) :=
by
  sorry

end decreasing_f_range_l191_191185


namespace object_travel_distance_in_one_hour_l191_191396

/-- If an object travels at 3 feet per second, then it travels 10800 feet in one hour. -/
theorem object_travel_distance_in_one_hour
  (speed : ℕ) (seconds_in_minute : ℕ) (minutes_in_hour : ℕ)
  (h_speed : speed = 3)
  (h_seconds_in_minute : seconds_in_minute = 60)
  (h_minutes_in_hour : minutes_in_hour = 60) :
  (speed * (seconds_in_minute * minutes_in_hour) = 10800) :=
by
  sorry

end object_travel_distance_in_one_hour_l191_191396


namespace gcd_gx_x_l191_191026

theorem gcd_gx_x (x : ℤ) (hx : 34560 ∣ x) :
  Int.gcd ((3 * x + 4) * (8 * x + 5) * (15 * x + 11) * (x + 17)) x = 20 := 
by
  sorry

end gcd_gx_x_l191_191026


namespace sequence_formula_l191_191017

theorem sequence_formula (S : ℕ → ℕ) (a : ℕ → ℕ) (h₁ : ∀ n, S n = n^2 + 1) :
  (a 1 = 2) ∧ (∀ n ≥ 2, a n = 2 * n - 1) := 
by
  sorry

end sequence_formula_l191_191017


namespace diameter_in_scientific_notation_l191_191239

def diameter : ℝ := 0.00000011
def scientific_notation (d : ℝ) : Prop := d = 1.1e-7

theorem diameter_in_scientific_notation : scientific_notation diameter :=
by
  sorry

end diameter_in_scientific_notation_l191_191239


namespace age_ratio_five_years_later_l191_191593

theorem age_ratio_five_years_later (my_age : ℕ) (son_age : ℕ) (h1 : my_age = 45) (h2 : son_age = 15) :
  (my_age + 5) / gcd (my_age + 5) (son_age + 5) = 5 ∧ (son_age + 5) / gcd (my_age + 5) (son_age + 5) = 2 :=
by
  sorry

end age_ratio_five_years_later_l191_191593


namespace largest_among_four_numbers_l191_191677

theorem largest_among_four_numbers
  (a b : ℝ)
  (h1 : 0 < a)
  (h2 : a < b)
  (h3 : a + b = 1) :
  b > max (max (1/2) (2 * a * b)) (a^2 + b^2) := 
sorry

end largest_among_four_numbers_l191_191677


namespace polynomial_inequality_solution_l191_191010

theorem polynomial_inequality_solution (x : ℝ) :
  (x < 5 - Real.sqrt 29 ∨ x > 5 + Real.sqrt 29) →
  x^3 - 12 * x^2 + 36 * x + 8 > 0 :=
by
  sorry

end polynomial_inequality_solution_l191_191010


namespace greatest_integer_a_l191_191432

-- Define formal properties and state the main theorem.
theorem greatest_integer_a (a : ℤ) : (∀ x : ℝ, ¬(x^2 + (a:ℝ) * x + 15 = 0)) → (a ≤ 7) :=
by
  intro h
  sorry

end greatest_integer_a_l191_191432


namespace field_area_is_36_square_meters_l191_191349

theorem field_area_is_36_square_meters (side_length : ℕ) (h : side_length = 6) : side_length * side_length = 36 :=
by
  sorry

end field_area_is_36_square_meters_l191_191349


namespace students_enjoy_soccer_fraction_l191_191279

theorem students_enjoy_soccer_fraction :
  (∀ (total_students : ℕ),
    let enjoy_soccer := 0.7 * total_students
    let dont_enjoy_soccer := 0.3 * total_students
    let express_enjoyment := 0.75 * enjoy_soccer
    let not_express_enjoyment := 0.25 * enjoy_soccer
    let express_disinterest := 0.85 * dont_enjoy_soccer
    let incorrectly_say_enjoy := 0.15 * dont_enjoy_soccer
    let say_dont_enjoy := not_express_enjoyment + express_disinterest
    not_express_enjoyment / say_dont_enjoy = 13 / 32)
:= sorry

end students_enjoy_soccer_fraction_l191_191279


namespace calculate_otimes_l191_191286

def otimes (x y : ℝ) : ℝ := x^3 - y^2 + x

theorem calculate_otimes (k : ℝ) : 
  otimes k (otimes k k) = -k^6 + 2*k^5 - 3*k^4 + 3*k^3 - k^2 + 2*k := by
  sorry

end calculate_otimes_l191_191286


namespace betsy_to_cindy_ratio_l191_191981

-- Definitions based on the conditions
def cindy_time : ℕ := 12
def tina_time : ℕ := cindy_time + 6
def betsy_time : ℕ := tina_time / 3

-- Theorem statement to prove
theorem betsy_to_cindy_ratio :
  (betsy_time : ℚ) / cindy_time = 1 / 2 :=
by sorry

end betsy_to_cindy_ratio_l191_191981


namespace sequence_2011_l191_191339

theorem sequence_2011 :
  ∀ (a : ℕ → ℤ), (a 1 = 1) →
                  (a 2 = 2) →
                  (∀ n : ℕ, a (n + 2) = a (n + 1) - a n) →
                  a 2011 = 1 :=
by {
  -- Insert proof here
  sorry
}

end sequence_2011_l191_191339


namespace largest_cube_edge_length_l191_191973

theorem largest_cube_edge_length (a : ℕ) : 
  (6 * a ^ 2 ≤ 1500) ∧
  (a * 15 ≤ 60) ∧
  (a * 15 ≤ 25) →
  a ≤ 15 :=
by
  sorry

end largest_cube_edge_length_l191_191973


namespace Q_finishes_in_6_hours_l191_191616

def Q_time_to_finish_job (T_Q : ℝ) : Prop :=
  let P_rate := 1 / 3
  let Q_rate := 1 / T_Q
  let work_together_2hr := 2 * (P_rate + Q_rate)
  let P_alone_work_40min := (2 / 3) * P_rate
  work_together_2hr + P_alone_work_40min = 1

theorem Q_finishes_in_6_hours : Q_time_to_finish_job 6 :=
  sorry -- Proof skipped

end Q_finishes_in_6_hours_l191_191616


namespace time_to_complete_together_l191_191623

theorem time_to_complete_together (sylvia_time carla_time combined_time : ℕ) (h_sylvia : sylvia_time = 45) (h_carla : carla_time = 30) :
  let sylvia_rate := 1 / (sylvia_time : ℚ)
  let carla_rate := 1 / (carla_time : ℚ)
  let combined_rate := sylvia_rate + carla_rate
  let time_to_complete := 1 / combined_rate
  time_to_complete = (combined_time : ℚ) :=
by
  sorry

end time_to_complete_together_l191_191623


namespace charging_time_is_correct_l191_191138

-- Lean definitions for the given conditions
def smartphone_charge_time : ℕ := 26
def tablet_charge_time : ℕ := 53
def phone_half_charge_time : ℕ := smartphone_charge_time / 2

-- Definition for the total charging time based on conditions
def total_charging_time : ℕ :=
  tablet_charge_time + phone_half_charge_time

-- Proof problem statement
theorem charging_time_is_correct : total_charging_time = 66 := by
  sorry

end charging_time_is_correct_l191_191138


namespace multiples_7_not_14_less_350_l191_191321

theorem multiples_7_not_14_less_350 : 
  ∃ n : ℕ, n = 25 ∧ (∀ k : ℕ, k < 350 → (k % 7 = 0 ∧ k % 14 ≠ 0 → k ∈ {7 * m | m : ℕ}) ∨ (k % 14 = 0 → k ∉ {7 * m | m : ℕ})) := 
sorry

end multiples_7_not_14_less_350_l191_191321


namespace abs_add_lt_abs_add_l191_191435

open Real

theorem abs_add_lt_abs_add {a b : ℝ} (h : a * b < 0) : abs (a + b) < abs a + abs b := 
  sorry

end abs_add_lt_abs_add_l191_191435


namespace place_circle_no_overlap_l191_191073

theorem place_circle_no_overlap 
    (rect_width rect_height : ℝ) (num_squares : ℤ) (square_size square_diameter : ℝ)
    (h_rect_dims : rect_width = 20 ∧ rect_height = 25)
    (h_num_squares : num_squares = 120)
    (h_square_size : square_size = 1)
    (h_circle_diameter : square_diameter = 1) : 
  ∃ (x y : ℝ), 0 ≤ x ∧ x ≤ rect_width ∧ 0 ≤ y ∧ y ≤ rect_height ∧ 
    ∀ (square_x square_y : ℝ), 
      0 ≤ square_x ∧ square_x ≤ rect_width - square_size ∧ 
      0 ≤ square_y ∧ square_y ≤ rect_height - square_size → 
      (x - square_x)^2 + (y - square_y)^2 ≥ (square_diameter / 2)^2 :=
sorry

end place_circle_no_overlap_l191_191073


namespace max_triangles_convex_polygon_l191_191213

theorem max_triangles_convex_polygon (vertices : ℕ) (interior_points : ℕ) (total_points : ℕ) : 
  vertices = 13 ∧ interior_points = 200 ∧ total_points = 213 ∧ (∀ (x y z : ℕ), (x < total_points ∧ y < total_points ∧ z < total_points) → x ≠ y ∧ y ≠ z ∧ x ≠ z) →
  (∃ triangles : ℕ, triangles = 411) :=
by
  sorry

end max_triangles_convex_polygon_l191_191213


namespace monotonicity_f_parity_f_max_value_f_min_value_f_l191_191035

noncomputable def f (x : ℝ) : ℝ := x / (x^2 - 4)

-- Monotonicity Proof
theorem monotonicity_f : ∀ {x1 x2 : ℝ}, 2 < x1 → 2 < x2 → x1 < x2 → f x1 > f x2 :=
sorry

-- Parity Proof
theorem parity_f : ∀ x : ℝ, f (-x) = -f x :=
sorry

-- Maximum Value Proof
theorem max_value_f : ∀ {x : ℝ}, x = -6 → f x = -3/16 :=
sorry

-- Minimum Value Proof
theorem min_value_f : ∀ {x : ℝ}, x = -3 → f x = -3/5 :=
sorry

end monotonicity_f_parity_f_max_value_f_min_value_f_l191_191035


namespace max_omega_for_increasing_l191_191202

noncomputable def sin_function (ω : ℕ) (x : ℝ) := Real.sin (ω * x + Real.pi / 6)

def is_monotonically_increasing_on (f : ℝ → ℝ) (a b : ℝ) : Prop := ∀ x y, a ≤ x ∧ x ≤ y ∧ y ≤ b → f x ≤ f y

theorem max_omega_for_increasing : ∀ (ω : ℕ), (0 < ω) →
  is_monotonically_increasing_on (sin_function ω) (Real.pi / 6) (Real.pi / 4) ↔ ω ≤ 9 :=
sorry

end max_omega_for_increasing_l191_191202


namespace solve_problem_l191_191706

namespace Example

-- Definitions based on given conditions
def isEvenFunction (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f x = f (-x)

def condition_2 (f : ℝ → ℝ) : Prop := f 2 = -1

def condition_3 (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f x = -f (2 - x)

-- Main theorem statement
theorem solve_problem (f : ℝ → ℝ)
  (h1 : isEvenFunction f)
  (h2 : condition_2 f)
  (h3 : condition_3 f) : f 2016 = 1 :=
sorry

end Example

end solve_problem_l191_191706


namespace total_students_class_is_63_l191_191487

def num_tables : ℕ := 6
def students_per_table : ℕ := 3
def girls_bathroom : ℕ := 4
def times_canteen : ℕ := 4
def group1_students : ℕ := 4
def group2_students : ℕ := 5
def group3_students : ℕ := 6
def germany_students : ℕ := 2
def france_students : ℕ := 4
def norway_students : ℕ := 3
def italy_students : ℕ := 1

def total_students_in_class : ℕ :=
  (num_tables * students_per_table) +
  girls_bathroom +
  (times_canteen * girls_bathroom) +
  (group1_students + group2_students + group3_students) +
  (germany_students + france_students + norway_students + italy_students)

theorem total_students_class_is_63 : total_students_in_class = 63 :=
  by
    sorry

end total_students_class_is_63_l191_191487


namespace proof_problem_l191_191025

variables (a b c : Line) (alpha beta gamma : Plane)

-- Define perpendicular relationship between line and plane
def perp_line_plane (l : Line) (p : Plane) : Prop := sorry

-- Define parallel relationship between lines
def parallel_lines (l1 l2 : Line) : Prop := sorry

-- Define parallel relationship between planes
def parallel_planes (p1 p2 : Plane) : Prop := sorry

-- Main theorem statement
theorem proof_problem 
  (h1 : perp_line_plane a alpha) 
  (h2 : perp_line_plane b beta) 
  (h3 : parallel_planes alpha beta) : 
  parallel_lines a b :=
sorry

end proof_problem_l191_191025


namespace avery_shirts_count_l191_191978

theorem avery_shirts_count {S : ℕ} (h_total : S + 2 * S + S = 16) : S = 4 :=
by
  sorry

end avery_shirts_count_l191_191978


namespace at_least_12_lyamziks_rowed_l191_191636

-- Define the lyamziks, their weights, and constraints
def LyamzikWeight1 : ℕ := 7
def LyamzikWeight2 : ℕ := 14
def LyamzikWeight3 : ℕ := 21
def LyamzikWeight4 : ℕ := 28
def totalLyamziks : ℕ := LyamzikWeight1 + LyamzikWeight2 + LyamzikWeight3 + LyamzikWeight4
def boatCapacity : ℕ := 10
def maxRowsPerLyamzik : ℕ := 2

-- Question to prove
theorem at_least_12_lyamziks_rowed : totalLyamziks ≥ 12 :=
  by sorry


end at_least_12_lyamziks_rowed_l191_191636


namespace combined_molecular_weight_l191_191111

def atomic_weight_C : ℝ := 12.01
def atomic_weight_Cl : ℝ := 35.45
def atomic_weight_S : ℝ := 32.07
def atomic_weight_F : ℝ := 19.00

def molecular_weight_CCl4 : ℝ := atomic_weight_C + 4 * atomic_weight_Cl
def molecular_weight_SF6 : ℝ := atomic_weight_S + 6 * atomic_weight_F

def weight_moles_CCl4 (moles : ℝ) : ℝ := moles * molecular_weight_CCl4
def weight_moles_SF6 (moles : ℝ) : ℝ := moles * molecular_weight_SF6

theorem combined_molecular_weight : weight_moles_CCl4 9 + weight_moles_SF6 5 = 2114.64 := by
  sorry

end combined_molecular_weight_l191_191111


namespace intersect_tetrahedral_angle_by_plane_l191_191492

noncomputable theory

open Geometry

def tetrahedral_angle_intersection (A B C D S : Point) (a b : Line) (α : Plane) : Prop :=
  convex_tetrahedral_angle A B C D S ∧
  intersects_line (face A S B) (face C S D) a S ∧
  intersects_line (face A S D) (face B S C) b S ∧
  plane_passes_through_lines α a b ∧
  (∀ (P : Point), P ∈ α ∧ P ∈ segment (face A S B) → 
    intersects α (tetrahedral_angle A B C D S) ∧ 
    quadrilateral (cross_section α (tetrahedral_angle A B C D S))).quadrilateral_parallelogram

theorem intersect_tetrahedral_angle_by_plane :
  ∀ (A B C D S : Point) (a b : Line) (α : Plane),
  tetrahedral_angle_intersection A B C D S a b α →
  ∃ (P : Parallelogram), (cross_section α (tetrahedral_angle A B C D S)) = P :=
by
  intros A B C D S a b α H
  sorry

end intersect_tetrahedral_angle_by_plane_l191_191492


namespace share_difference_l191_191145

variables {x : ℕ}

theorem share_difference (h1: 12 * x - 7 * x = 5000) : 7 * x - 3 * x = 4000 :=
by
  sorry

end share_difference_l191_191145


namespace gardner_bakes_brownies_l191_191226

theorem gardner_bakes_brownies : 
  ∀ (cookies cupcakes brownies students sweet_treats_per_student total_sweet_treats total_cookies_and_cupcakes : ℕ),
  cookies = 20 →
  cupcakes = 25 →
  students = 20 →
  sweet_treats_per_student = 4 →
  total_sweet_treats = students * sweet_treats_per_student →
  total_cookies_and_cupcakes = cookies + cupcakes →
  brownies = total_sweet_treats - total_cookies_and_cupcakes →
  brownies = 35 :=
by
  intros cookies cupcakes brownies students sweet_treats_per_student total_sweet_treats total_cookies_and_cupcakes
  intros h1 h2 h3 h4 h5 h6 h7
  sorry

end gardner_bakes_brownies_l191_191226


namespace alice_probability_l191_191273

noncomputable def probability_picking_exactly_three_green_marbles : ℚ :=
  let binom : ℚ := 35 -- binomial coefficient (7 choose 3)
  let prob_green : ℚ := 8 / 15 -- probability of picking a green marble
  let prob_purple : ℚ := 7 / 15 -- probability of picking a purple marble
  binom * (prob_green ^ 3) * (prob_purple ^ 4)

theorem alice_probability :
  probability_picking_exactly_three_green_marbles = 34454336 / 136687500 := by
  sorry

end alice_probability_l191_191273


namespace speed_ratio_of_runners_l191_191614

theorem speed_ratio_of_runners (v_A v_B : ℝ) (c : ℝ)
  (h1 : 0 < v_A ∧ 0 < v_B) -- They run at constant, but different speeds
  (h2 : (v_B / v_A) = (2 / 3)) -- Distance relationship from meeting points
  : v_B / v_A = 2 :=
by
  sorry

end speed_ratio_of_runners_l191_191614


namespace hexagon_area_l191_191087

-- Definitions of the conditions
def DEF_perimeter := 42
def circumcircle_radius := 10
def area_of_hexagon_DE'F'D'E'F := 210

-- The theorem statement
theorem hexagon_area (DEF_perimeter : ℕ) (circumcircle_radius : ℕ) : Prop :=
  DEF_perimeter = 42 → circumcircle_radius = 10 → 
  area_of_hexagon_DE'F'D'E'F = 210

-- Example invocation of the theorem, proof omitted.
example : hexagon_area DEF_perimeter circumcircle_radius :=
by {
  sorry
}

end hexagon_area_l191_191087


namespace min_value_of_xy_ratio_l191_191752

theorem min_value_of_xy_ratio :
  ∃ t : ℝ,
    (t = 2 ∨
    t = ((-1 + Real.sqrt 217) / 12) ∨
    t = ((-1 - Real.sqrt 217) / 12)) ∧
    min (min 2 ((-1 + Real.sqrt 217) / 12)) ((-1 - Real.sqrt 217) / 12) = -1.31 :=
sorry

end min_value_of_xy_ratio_l191_191752


namespace parabola_directrix_l191_191565

theorem parabola_directrix (x y : ℝ) (h_eqn : y = -3 * x^2 + 6 * x - 5) :
  y = -23 / 12 :=
sorry

end parabola_directrix_l191_191565


namespace find_e_l191_191088

theorem find_e (d e f : ℝ) (h1 : f = 5)
  (h2 : -d / 3 = -f)
  (h3 : -f = 1 + d + e + f) :
  e = -26 := 
by
  sorry

end find_e_l191_191088


namespace average_abs_diff_sum_l191_191165

open Finset

-- Define the absolute difference sum over permutations
def abs_diff_sum (s : Permutations (Fin 8)) : ℝ :=
|b_1 - b_2| + |b_3 - b_4| + |b_5 - b_6| + |b_7 - b_8|

-- The main statement
theorem average_abs_diff_sum : 
  ∃ (p q : ℕ), Nat.coprime p q ∧ (p / q : ℝ) = 12 ∧ p + q = 13 := by
  sorry

end average_abs_diff_sum_l191_191165


namespace ratio_of_area_to_perimeter_l191_191940

noncomputable def altitude_of_equilateral_triangle (s : ℝ) : ℝ :=
  s * (Real.sqrt 3 / 2)

noncomputable def area_of_equilateral_triangle (s : ℝ) : ℝ :=
  1 / 2 * s * altitude_of_equilateral_triangle s

noncomputable def perimeter_of_equilateral_triangle (s : ℝ) : ℝ :=
  3 * s

theorem ratio_of_area_to_perimeter (s : ℝ) (h : s = 10) :
    (area_of_equilateral_triangle s) / (perimeter_of_equilateral_triangle s) = 5 * Real.sqrt 3 / 6 :=
  by
  rw [h]
  sorry

end ratio_of_area_to_perimeter_l191_191940


namespace lemma2_l191_191587

noncomputable def f (x a b : ℝ) := |x + a| - |x - b|

lemma lemma1 {x : ℝ} : f x 1 2 > 2 ↔ x > 3 / 2 := 
sorry

theorem lemma2 {a b : ℝ} (h₀ : 0 < a) (h₁ : 0 < b) (h₂ : ∀ x : ℝ, f x a b ≤ 3):
  1 / a + 2 / b = (1 / 3) * (3 + 2 * Real.sqrt 2) := 
sorry

end lemma2_l191_191587


namespace triangle_inequality_property_l191_191257

noncomputable def perimeter (a b c : ℝ) : ℝ := a + b + c

noncomputable def circumradius (a b c : ℝ) (A B C: ℝ) : ℝ := 
  (a * b * c) / (4 * Real.sqrt (A * B * C))

noncomputable def inradius (a b c : ℝ) (A B C: ℝ) : ℝ := 
  Real.sqrt (A * B * C) * perimeter a b c

theorem triangle_inequality_property (a b c A B C : ℝ)
  (h₁ : ∀ {x}, x > 0)
  (h₂ : A ≠ B)
  (h₃ : B ≠ C)
  (h₄ : C ≠ A) :
  ¬ (perimeter a b c ≤ circumradius a b c A B C + inradius a b c A B C) ∧
  ¬ (perimeter a b c > circumradius a b c A B C + inradius a b c A B C) ∧
  ¬ (perimeter a b c / 6 < circumradius a b c A B C + inradius a b c A B C ∨ 
  circumradius a b c A B C + inradius a b c A B C < 6 * perimeter a b c) :=
sorry

end triangle_inequality_property_l191_191257


namespace necessary_and_sufficient_condition_l191_191768

theorem necessary_and_sufficient_condition (a : ℝ) : (∀ x : ℝ, 1 ≤ x ∧ x ≤ 2 → x^2 - a ≤ 0) ↔ (a ≥ 4) :=
by
  sorry

end necessary_and_sufficient_condition_l191_191768


namespace ratio_equilateral_triangle_l191_191944

def equilateral_triangle_ratio (s : ℝ) (h : s = 10) : ℝ :=
  let altitude := s * (Real.sqrt 3 / 2)
  let area := (1 / 2) * s * altitude
  let perimeter := 3 * s in
  area / perimeter -- this simplifies to 25\sqrt{3} / 30 or 5\sqrt{3} / 6

theorem ratio_equilateral_triangle : ∀ (s : ℝ), s = 10 → equilateral_triangle_ratio s (by assumption) = 5 * (Real.sqrt 3) / 6 :=
by
  intros s h
  rw h
  sorry

end ratio_equilateral_triangle_l191_191944


namespace binomial_12_3_eq_220_l191_191983

theorem binomial_12_3_eq_220 : nat.choose 12 3 = 220 :=
by {
  sorry
}

end binomial_12_3_eq_220_l191_191983


namespace greatest_prime_factor_391_l191_191388

theorem greatest_prime_factor_391 : ∃ p, prime p ∧ p ∣ 391 ∧ ∀ q, prime q ∧ q ∣ 391 → q ∣ p := by
  sorry

end greatest_prime_factor_391_l191_191388


namespace min_triangular_faces_l191_191262

theorem min_triangular_faces (l c e m n k : ℕ) (h1 : l > c) (h2 : l + c = e + 2) (h3 : l = c + k) (h4 : e ≥ (3 * m + 4 * n) / 2) :
  m ≥ 6 := sorry

end min_triangular_faces_l191_191262


namespace product_sum_diff_l191_191912

variable (a b : ℝ) -- Real numbers

theorem product_sum_diff (a b : ℝ) : (a + b) * (a - b) = (a + b) * (a - b) :=
by
  sorry

end product_sum_diff_l191_191912


namespace find_value_of_ratio_l191_191483

theorem find_value_of_ratio (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : x / y + y / x = 4) :
  (x + 2 * y) / (x - 2 * y) = Real.sqrt 33 / 3 := 
  sorry

end find_value_of_ratio_l191_191483


namespace angles_on_x_axis_l191_191090

theorem angles_on_x_axis (α : ℝ) : 
  (∃ k : ℤ, α = 2 * k * Real.pi) ∨ (∃ k : ℤ, α = (2 * k + 1) * Real.pi) ↔ 
  ∃ k : ℤ, α = k * Real.pi :=
by
  sorry

end angles_on_x_axis_l191_191090


namespace quadratic_equal_roots_iff_l191_191434

theorem quadratic_equal_roots_iff (k : ℝ) :
  (∃ x : ℝ, x^2 - k * x + 9 = 0 ∧ x^2 - k * x + 9 = 0 ∧ x = x) ↔ k^2 = 36 :=
by
  sorry

end quadratic_equal_roots_iff_l191_191434


namespace tangent_line_to_curve_at_P_l191_191900

noncomputable def tangent_line_at_point (x y : ℝ) := 4 * x - y - 2 = 0

theorem tangent_line_to_curve_at_P :
  (∃ (b: ℝ), ∀ (x: ℝ), b = 2 * 1^2 → tangent_line_at_point 1 2)
:= 
by
  sorry

end tangent_line_to_curve_at_P_l191_191900


namespace average_class_a_average_class_b_expectation_of_X_l191_191561

noncomputable section

def selfStudyTimesClassA : List ℝ := [8, 13, 28, 32, 39]
def selfStudyTimesClassB : List ℝ := [12, 25, 26, 28, 31]
def threshold : ℝ := 26
def average (l : List ℝ) : ℝ := l.sum / l.length

def averageClassA := average selfStudyTimesClassA
def averageClassB := average selfStudyTimesClassB

theorem average_class_a : averageClassA = 24 := by
  sorry

theorem average_class_b : averageClassB = 24.4 := by
  sorry

def binom (n k : ℕ) : ℕ := Nat.choose n k
def probabilityX (x : ℕ) : ℝ := 
  match x with
  | 0 => binom 2 2 * binom 3 2 / (binom 5 2 * binom 5 2)
  | 1 => (binom 2 1 * binom 3 1 * binom 3 2 + binom 2 2 * binom 3 1 * binom 2 1) / (binom 5 2 * binom 5 2)
  | 2 => (binom 2 1 * binom 3 1 * binom 3 1 * binom 2 1 + binom 3 2 * binom 3 2 + binom 2 2 * binom 2 2) / (binom 5 2 * binom 5 2)
  | 3 => (binom 3 2 * binom 3 1 * binom 2 1 + binom 2 1 * binom 3 1 * binom 2 2) / (binom 5 2 * binom 5 2)
  | 4 => binom 3 2 * binom 2 2 / (binom 5 2 * binom 5 2)
  | _ => 0

def expectationX : ℝ := ∑ x in Finset.range 5, x * probabilityX x

theorem expectation_of_X : expectationX = 2 := by
  sorry

end average_class_a_average_class_b_expectation_of_X_l191_191561


namespace GCF_of_LCMs_l191_191069

def GCF : ℕ → ℕ → ℕ := Nat.gcd
def LCM : ℕ → ℕ → ℕ := Nat.lcm

theorem GCF_of_LCMs :
  GCF (LCM 9 21) (LCM 10 15) = 3 :=
by
  sorry

end GCF_of_LCMs_l191_191069


namespace smallest_consecutive_sum_l191_191775

theorem smallest_consecutive_sum (n : ℤ) (h : n + (n + 1) + (n + 2) + (n + 3) + (n + 4) = 210) : 
  n = 40 := 
sorry

end smallest_consecutive_sum_l191_191775


namespace largest_value_of_c_l191_191733

theorem largest_value_of_c : ∀ c : ℝ, (3 * c + 6) * (c - 2) = 9 * c → c ≤ 4 :=
by
  intros c hc
  have : (3 * c + 6) * (c - 2) = 9 * c := hc
  sorry

end largest_value_of_c_l191_191733


namespace speed_man_l191_191206

noncomputable def speedOfMan : ℝ := 
  let d := 437.535 / 1000  -- distance in kilometers
  let t := 25 / 3600      -- time in hours
  d / t                    -- speed in kilometers per hour

theorem speed_man : speedOfMan = 63 := by
  sorry

end speed_man_l191_191206


namespace translated_coordinates_of_B_l191_191057

-- Definitions and conditions
def pointA : ℝ × ℝ := (-2, 3)

def translate_right (x : ℝ) (units : ℝ) : ℝ := x + units
def translate_down (y : ℝ) (units : ℝ) : ℝ := y - units

-- Theorem statement
theorem translated_coordinates_of_B :
  let Bx := translate_right (-2) 3
  let By := translate_down 3 5
  (Bx, By) = (1, -2) :=
by
  -- This is where the proof would go, but we're using sorry to skip the proof steps.
  sorry

end translated_coordinates_of_B_l191_191057


namespace proof_rewritten_eq_and_sum_l191_191231

-- Define the given equation
def given_eq (x : ℝ) : Prop := 64 * x^2 + 80 * x - 72 = 0

-- Define the rewritten form of the equation
def rewritten_eq (x : ℝ) : Prop := (8 * x + 5)^2 = 97

-- Define the correctness of rewriting the equation
def correct_rewrite (x : ℝ) : Prop :=
  given_eq x → rewritten_eq x

-- Define the correct value of a + b + c
def correct_sum : Prop :=
  8 + 5 + 97 = 110

-- The final theorem statement
theorem proof_rewritten_eq_and_sum (x : ℝ) : correct_rewrite x ∧ correct_sum :=
by
  sorry

end proof_rewritten_eq_and_sum_l191_191231


namespace value_of_product_l191_191520

theorem value_of_product : (1/3 * 9 * 1/27 * 81 * 1/243 * 729 * 1/2187 * 6561 * 1/19683 * 59049 = 243) :=
by sorry

end value_of_product_l191_191520


namespace rectangle_width_decrease_l191_191905

theorem rectangle_width_decrease (A L W : ℝ) (h1 : A = L * W) (h2 : 1.5 * L * W' = A) : 
  (W' = (2/3) * W) -> by exact (W - W') / W = 1 / 3 :=
by
  sorry

end rectangle_width_decrease_l191_191905


namespace altitude_segment_length_l191_191516

theorem altitude_segment_length 
  {A B C D E : Type} 
  (BD DC AE y : ℝ) 
  (h1 : BD = 4) 
  (h2 : DC = 6) 
  (h3 : AE = 3) 
  (h4 : 3 / 4 = 9 / (y + 3)) : 
  y = 9 := 
by 
  sorry

end altitude_segment_length_l191_191516


namespace eight_digit_permutations_l191_191195

open Fin

-- Definition of the number of each digit occurrence
def digits_occurrences : List Nat := [2, 2, 2, 2]

-- Definition of the total number of digits
def total_digits : Nat := 8

-- The factorial function defined in Lean
def factorial (n : Nat) : Nat := if n = 0 then 1 else n * factorial (n - 1)

-- The count of unique permutations considering repetitions of digits
def count_permutations : Nat :=
  factorial total_digits / (factorial 2 * factorial 2 * factorial 2 * factorial 2)

-- The main theorem: Proof that the number of different eight-digit integers is 2520
theorem eight_digit_permutations : count_permutations = 2520 :=
by
  sorry

end eight_digit_permutations_l191_191195


namespace inequality_solution_set_range_of_a_l191_191850

def f (x : ℝ) : ℝ := abs (3*x + 2)

theorem inequality_solution_set :
  { x : ℝ | f x < 4 - abs (x - 1) } = { x : ℝ | -5/4 < x ∧ x < 1/2 } :=
by 
  sorry

theorem range_of_a (a : ℝ) (m n : ℝ) (h1 : m + n = 1) (h2 : 0 < m) (h3 : 0 < n) 
  (h4 : ∀ x : ℝ, abs (x - a) - f x ≤ 1 / m + 1 / n) : 
  0 < a ∧ a ≤ 10/3 :=
by 
  sorry

end inequality_solution_set_range_of_a_l191_191850


namespace meaningful_fraction_x_range_l191_191864

theorem meaningful_fraction_x_range (x : ℝ) : (x-2 ≠ 0) ↔ (x ≠ 2) :=
by 
  sorry

end meaningful_fraction_x_range_l191_191864


namespace calc_expression_l191_191068

def r (θ : ℚ) : ℚ := 1 / (1 + θ)
def s (θ : ℚ) : ℚ := θ + 1

theorem calc_expression : s (r (s (r (s (r 2))))) = 24 / 17 :=
by 
  sorry

end calc_expression_l191_191068


namespace min_value_expression_l191_191064

theorem min_value_expression (x y : ℝ) (hx : 0 < x) (hy : 0 < y) :
    let a := 2
    let b := 3
    let term1 := 2*x + 1/(3*y)
    let term2 := 3*y + 1/(2*x)
    (term1 * (term1 - 2023) + term2 * (term2 - 2023)) = -2050529.5 :=
sorry

end min_value_expression_l191_191064


namespace prime_sum_product_l191_191915

theorem prime_sum_product (p q : ℕ) (hp : Nat.Prime p) (hq : Nat.Prime q) (hsum : p + q = 102) (hgt : p > 30 ∨ q > 30) :
  p * q = 2201 := 
sorry

end prime_sum_product_l191_191915


namespace find_k_l191_191175

theorem find_k (a b : ℕ) (k : ℕ) (h1 : 0 < a) (h2 : 0 < b) (h3 : (a^2 + b^2) = k * (a * b - 1)) :
  k = 5 :=
sorry

end find_k_l191_191175


namespace binom_12_3_eq_220_l191_191988

theorem binom_12_3_eq_220 : nat.choose 12 3 = 220 :=
sorry

end binom_12_3_eq_220_l191_191988


namespace distinct_positive_integers_exists_l191_191453

theorem distinct_positive_integers_exists 
(n : ℕ)
(a b : ℕ)
(h1 : a ≠ b)
(h2 : b % a = 0)
(h3 : a > 10^(2 * n - 1) ∧ a < 10^(2 * n))
(h4 : b > 10^(2 * n - 1) ∧ b < 10^(2 * n))
(h5 : ∀ x y : ℕ, a = 10^n * x + y ∧ b = 10^n * y + x ∧ x < y ∧ x / 10^(n - 1) ≠ 0 ∧ y / 10^(n - 1) ≠ 0) :
a = (10^(2 * n) - 1) / 7 ∧ b = 6 * (10^(2 * n) - 1) / 7 := 
by
  sorry

end distinct_positive_integers_exists_l191_191453


namespace find_a4_l191_191307

-- Define the arithmetic sequence and the sum of the first N terms
def arithmetic_seq (a d : ℕ) (n : ℕ) : ℕ := a + (n - 1) * d

-- Sum of the first N terms in an arithmetic sequence
def sum_arithmetic_seq (a d N : ℕ) : ℕ := N * (2 * a + (N - 1) * d) / 2

-- Define the conditions
def condition1 (a d : ℕ) : Prop := a + (a + 2 * d) + (a + 4 * d) = 15
def condition2 (a d : ℕ) : Prop := sum_arithmetic_seq a d 4 = 16

-- Lean 4 statement to prove the value of a_4
theorem find_a4 (a d : ℕ) (h1 : condition1 a d) (h2 : condition2 a d) : arithmetic_seq a d 4 = 7 :=
sorry

end find_a4_l191_191307


namespace william_won_more_rounds_than_harry_l191_191251

def rounds_played : ℕ := 15
def william_won_rounds : ℕ := 10
def harry_won_rounds : ℕ := rounds_played - william_won_rounds
def william_won_more_rounds := william_won_rounds > harry_won_rounds

theorem william_won_more_rounds_than_harry : william_won_rounds - harry_won_rounds = 5 := 
by sorry

end william_won_more_rounds_than_harry_l191_191251


namespace min_teachers_required_l191_191953

-- Define the conditions
def num_english_teachers : ℕ := 9
def num_history_teachers : ℕ := 7
def num_geography_teachers : ℕ := 6
def max_subjects_per_teacher : ℕ := 2

-- The proposition we want to prove
theorem min_teachers_required :
  ∃ (t : ℕ), t = 13 ∧
    t * max_subjects_per_teacher ≥ num_english_teachers + num_history_teachers + num_geography_teachers :=
sorry

end min_teachers_required_l191_191953


namespace ratio_of_area_to_perimeter_of_equilateral_triangle_l191_191938

theorem ratio_of_area_to_perimeter_of_equilateral_triangle (s : ℕ) : s = 10 → (let A := (s^2 * sqrt 3) / 4, P := 3 * s in A / P = 5 * sqrt 3 / 6) :=
by
  intro h,
  sorry

end ratio_of_area_to_perimeter_of_equilateral_triangle_l191_191938


namespace average_birds_monday_l191_191730

variable (M : ℕ)

def avg_birds_monday (M : ℕ) : Prop :=
  let total_sites := 5 + 5 + 10
  let total_birds := 5 * M + 5 * 5 + 10 * 8
  (total_birds = total_sites * 7)

theorem average_birds_monday (M : ℕ) (h : avg_birds_monday M) : M = 7 := by
  sorry

end average_birds_monday_l191_191730


namespace right_triangle_side_sums_l191_191691

theorem right_triangle_side_sums (a b c : ℕ) (h1 : a + b = c + 6) (h2 : a^2 + b^2 = c^2) :
  (a = 7 ∧ b = 24 ∧ c = 25) ∨ (a = 8 ∧ b = 15 ∧ c = 17) ∨ (a = 9 ∧ b = 12 ∧ c = 15) :=
sorry

end right_triangle_side_sums_l191_191691


namespace patio_tiles_l191_191140

theorem patio_tiles (r c : ℕ) (h1 : r * c = 48) (h2 : (r + 4) * (c - 2) = 48) : r = 6 :=
sorry

end patio_tiles_l191_191140


namespace mary_mortgage_payment_l191_191347

theorem mary_mortgage_payment :
  let a1 := 400
  let r := 2
  let n := 11
  let sum_geom_series (a1 r : ℕ) (n : ℕ) : ℕ := (a1 * (1 - r^n)) / (1 - r)
  sum_geom_series a1 r n = 819400 :=
by
  let a1 := 400
  let r := 2
  let n := 11
  let sum_geom_series (a1 r : ℕ) (n : ℕ) : ℕ := (a1 * (1 - r^n)) / (1 - r)
  have h : sum_geom_series a1 r n = 819400 := sorry
  exact h

end mary_mortgage_payment_l191_191347


namespace sum_of_digits_of_2010_l191_191611

noncomputable def sum_of_base6_digits (n : ℕ) : ℕ :=
  (n.digits 6).sum

theorem sum_of_digits_of_2010 : sum_of_base6_digits 2010 = 10 := by
  sorry

end sum_of_digits_of_2010_l191_191611


namespace translated_coordinates_of_B_l191_191335

-- Define the initial coordinates of points A and B
def A : ℝ × ℝ := (1, 1)
def B : ℝ × ℝ := (-2, 0)

-- Define the translated coordinates of point A
def A' : ℝ × ℝ := (4, 0)

-- Define the expected coordinates of point B' after the translation
def B' : ℝ × ℝ := (1, -1)

-- Proof statement
theorem translated_coordinates_of_B (A A' B : ℝ × ℝ) (B' : ℝ × ℝ) :
  A = (1, 1) ∧ A' = (4, 0) ∧ B = (-2, 0) → B' = (1, -1) :=
by
  intros h
  sorry

end translated_coordinates_of_B_l191_191335


namespace closest_perfect_square_to_1042_is_1024_l191_191950

theorem closest_perfect_square_to_1042_is_1024 :
  ∀ n : ℕ, (n = 32 ∨ n = 33) → ((1042 - n^2 = 18) ↔ n = 32):=
by
  intros n hn
  cases hn
  case inl h32 => sorry
  case inr h33 => sorry

end closest_perfect_square_to_1042_is_1024_l191_191950


namespace photo_area_with_frame_l191_191527

-- Define the areas and dimensions given in the conditions
def paper_length : ℕ := 12
def paper_width : ℕ := 8
def frame_width : ℕ := 2

-- Define the dimensions of the photo including the frame
def photo_length_with_frame : ℕ := paper_length + 2 * frame_width
def photo_width_with_frame : ℕ := paper_width + 2 * frame_width

-- The theorem statement proving the area of the wall photo including the frame
theorem photo_area_with_frame :
  (photo_length_with_frame * photo_width_with_frame) = 192 := by
  sorry

end photo_area_with_frame_l191_191527


namespace alpha_plus_beta_l191_191095

theorem alpha_plus_beta :
  (∃ α β : ℝ, 
    (∀ x : ℝ, x ≠ -β ∧ x ≠ 45 → (x - α) / (x + β) = (x^2 - 90 * x + 1980) / (x^2 + 70 * x - 3570))
  ) → (∃ α β : ℝ, α + β = 123) :=
by {
  sorry
}

end alpha_plus_beta_l191_191095


namespace missing_digit_in_mean_of_number_set_l191_191682

noncomputable def number_set : Finset ℝ :=
Finset.range 9 |>.map (λ k, ((8:ℝ) * (10 ^ (k + 1) - 1) / 9))

noncomputable def arithmetic_mean (s : Finset ℝ) : ℝ :=
s.sum / s.card

theorem missing_digit_in_mean_of_number_set : 
  ∀ M : ℝ, M = arithmetic_mean number_set → (∃ d : ℕ, d ∈ (Finset.range 10) ∧ ¬(d : ℝ) ∈ Finset.coe (Finset.image (λ x, (x : ℝ).fract) (LinOrder.range 10)) → d = 0) :=
by
  sorry

end missing_digit_in_mean_of_number_set_l191_191682


namespace sum_a_eq_9_l191_191513

def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

theorem sum_a_eq_9 (a2 a3 a4 a5 a6 a7 : ℤ) 
  (h1 : 0 ≤ a2 ∧ a2 < 2) (h2 : 0 ≤ a3 ∧ a3 < 3) (h3 : 0 ≤ a4 ∧ a4 < 4)
  (h4 : 0 ≤ a5 ∧ a5 < 5) (h5 : 0 ≤ a6 ∧ a6 < 6) (h6 : 0 ≤ a7 ∧ a7 < 7)
  (h_eq : (5 : ℚ) / 7 = (a2 : ℚ) / factorial 2 + (a3 : ℚ) / factorial 3 + (a4 : ℚ) / factorial 4 + 
                         (a5 : ℚ) / factorial 5 + (a6 : ℚ) / factorial 6 + (a7 : ℚ) / factorial 7) :
  a2 + a3 + a4 + a5 + a6 + a7 = 9 := 
sorry

end sum_a_eq_9_l191_191513


namespace algebraic_expression_value_l191_191698

theorem algebraic_expression_value (x y : ℝ) (h : |x - 2| + (y + 3)^2 = 0) : (x + y)^2023 = -1 := by
  sorry

end algebraic_expression_value_l191_191698


namespace find_m_for_parallel_lines_l191_191191

theorem find_m_for_parallel_lines (m : ℝ) :
  (∀ x y : ℝ, (3 + m) * x + 4 * y = 5 - 3 * m) →
  (∀ x y : ℝ, 2 * x + (5 + m) * y = 8) →
  m = -7 :=
by
  sorry

end find_m_for_parallel_lines_l191_191191


namespace quadratic_form_decomposition_l191_191770

theorem quadratic_form_decomposition (a b c : ℝ) (h : ∀ x : ℝ, 8 * x^2 + 64 * x + 512 = a * (x + b) ^ 2 + c) :
  a + b + c = 396 := 
sorry

end quadratic_form_decomposition_l191_191770


namespace solve_equation_l191_191363

theorem solve_equation :
  ∀ x : ℝ, 
    (1 / (x + 8) + 1 / (x + 5) = 1 / (x + 11) + 1 / (x + 4)) ↔ 
      (x = (-3 + Real.sqrt 37) / 2 ∨ x = (-3 - Real.sqrt 37) / 2) := 
by
  intro x
  sorry

end solve_equation_l191_191363


namespace exists_k_simplifies_expression_to_5x_squared_l191_191221

theorem exists_k_simplifies_expression_to_5x_squared :
  ∃ k : ℝ, (∀ x : ℝ, (x - k * x) * (2 * x - k * x) - 3 * x * (2 * x - k * x) = 5 * x^2) :=
by
  sorry

end exists_k_simplifies_expression_to_5x_squared_l191_191221


namespace cos_sum_zero_l191_191553

noncomputable def cos_sum : ℂ :=
  Real.cos (Real.pi / 15) + Real.cos (4 * Real.pi / 15) + Real.cos (7 * Real.pi / 15) + Real.cos (10 * Real.pi / 15)

theorem cos_sum_zero : cos_sum = 0 := by
  sorry

end cos_sum_zero_l191_191553


namespace find_arithmetic_sequence_elements_l191_191471

theorem find_arithmetic_sequence_elements :
  ∃ (a b c : ℤ), -1 < a ∧ a < b ∧ b < c ∧ c < 7 ∧
  (∃ d : ℤ, a = -1 + d ∧ b = -1 + 2 * d ∧ c = -1 + 3 * d ∧ 7 = -1 + 4 * d) :=
sorry

end find_arithmetic_sequence_elements_l191_191471


namespace find_m_b_l191_191118

theorem find_m_b (m b : ℚ) :
  (3 * m - 14 = 2) ∧ (m ^ 2 - 6 * m + 15 = b) →
  m = 16 / 3 ∧ b = 103 / 9 := by
  intro h
  rcases h with ⟨h1, h2⟩
  -- proof steps here
  sorry

end find_m_b_l191_191118


namespace sum_of_29_12_23_is_64_sixtyfour_is_two_to_six_l191_191511

theorem sum_of_29_12_23_is_64: 29 + 12 + 23 = 64 := sorry

theorem sixtyfour_is_two_to_six:
  64 = 2^6 := sorry

end sum_of_29_12_23_is_64_sixtyfour_is_two_to_six_l191_191511


namespace average_speed_palindrome_l191_191294

open Nat

theorem average_speed_palindrome :
  ∀ (initial final : ℕ) (time : ℕ), (initial = 12321) →
    (final = 12421) →
    (time = 3) →
    (∃ speed : ℚ, speed = (final - initial) / time ∧ speed = 33.33) :=
by
  intros initial final time h_initial h_final h_time
  sorry

end average_speed_palindrome_l191_191294


namespace min_value_of_z_l191_191582

theorem min_value_of_z (a x y : ℝ) (h1 : a > 0) (h2 : x ≥ 1) (h3 : x + y ≤ 3) (h4 : y ≥ a * (x - 3)) :
  (∃ (x y : ℝ), 2 * x + y = 1) → a = 1 / 2 :=
by {
  sorry
}

end min_value_of_z_l191_191582


namespace evaluate_fx_plus_2_l191_191218

noncomputable def f (x : ℝ) : ℝ := (x + 1) / (x - 1)

theorem evaluate_fx_plus_2 (x : ℝ) (h : x ^ 2 ≠ 1) : 
  f (x + 2) = (x + 3) / (x + 1) :=
by
  sorry

end evaluate_fx_plus_2_l191_191218


namespace decreasing_interval_implies_range_of_a_l191_191049

noncomputable def f (a x : ℝ) : ℝ := x^2 + 2 * (a - 1) * x + 2

theorem decreasing_interval_implies_range_of_a (a : ℝ)
  (h : ∀ x y : ℝ, x ≤ y → y ≤ 4 → f a x ≥ f a y) : a ≤ -3 :=
by
  sorry

end decreasing_interval_implies_range_of_a_l191_191049


namespace greatest_drop_in_price_l191_191130

def jan_change : ℝ := -0.75
def feb_change : ℝ := 1.50
def mar_change : ℝ := -3.00
def apr_change : ℝ := 2.50
def may_change : ℝ := -0.25
def jun_change : ℝ := 0.80
def jul_change : ℝ := -2.75
def aug_change : ℝ := -1.20

theorem greatest_drop_in_price : 
  mar_change = min (min (min (min (min (min jan_change jul_change) aug_change) may_change) feb_change) apr_change) jun_change :=
by
  -- This statement is where the proof would go.
  sorry

end greatest_drop_in_price_l191_191130


namespace solve_for_k_l191_191205

theorem solve_for_k (k : ℚ) : 
  (∃ x : ℚ, (3 * x - 6 = 0) ∧ (2 * x - 5 * k = 11)) → k = -7/5 :=
by 
  intro h
  cases' h with x hx
  have hx1 : x = 2 := by linarith
  have hx2 : x = 11 / 2 + 5 / 2 * k := by linarith
  linarith

end solve_for_k_l191_191205


namespace percentage_decrease_in_larger_angle_l191_191509

-- Define the angles and conditions
def angles_complementary (A B : ℝ) : Prop := A + B = 90
def angle_ratio (A B : ℝ) : Prop := A / B = 1 / 2
def small_angle_increase (A A' : ℝ) : Prop := A' = A * 1.2
def large_angle_new (A' B' : ℝ) : Prop := A' + B' = 90

-- Prove percentage decrease in larger angle
theorem percentage_decrease_in_larger_angle (A B A' B' : ℝ) 
    (h1 : angles_complementary A B)
    (h2 : angle_ratio A B)
    (h3 : small_angle_increase A A')
    (h4 : large_angle_new A' B')
    : (B - B') / B * 100 = 10 :=
sorry

end percentage_decrease_in_larger_angle_l191_191509


namespace weight_of_new_person_l191_191501

-- Definitions
variable (W : ℝ) -- total weight of original 15 people
variable (x : ℝ) -- weight of the new person
variable (n : ℕ) (avr_increase : ℝ) (original_person_weight : ℝ)
variable (total_increase : ℝ) -- total weight increase

-- Given constants
axiom n_value : n = 15
axiom avg_increase_value : avr_increase = 8
axiom original_person_weight_value : original_person_weight = 45
axiom total_increase_value : total_increase = n * avr_increase

-- Equation stating the condition
axiom weight_replace : W - original_person_weight + x = W + total_increase

-- Theorem (problem translated)
theorem weight_of_new_person : x = 165 := by
  sorry

end weight_of_new_person_l191_191501


namespace find_K_values_l191_191051

theorem find_K_values (K M : ℕ) (h1 : (K * (K + 1)) / 2 = M^2) (h2 : M < 200) (h3 : K > M) :
  K = 8 ∨ K = 49 :=
sorry

end find_K_values_l191_191051


namespace solve_eq1_solve_eq2_l191_191890

-- Define the first equation
def eq1 (x : ℚ) : Prop := x / (x - 1) = 3 / (2*x - 2) - 2

-- Define the valid solution for the first equation
def sol1 : ℚ := 7 / 6

-- Theorem for the first equation
theorem solve_eq1 : eq1 sol1 :=
by
  sorry

-- Define the second equation
def eq2 (x : ℚ) : Prop := (5*x + 2) / (x^2 + x) = 3 / (x + 1)

-- Theorem for the second equation: there is no valid solution
theorem solve_eq2 : ¬ ∃ x : ℚ, eq2 x :=
by
  sorry

end solve_eq1_solve_eq2_l191_191890


namespace intersect_complement_l191_191319

-- Definition of the universal set U
def U : Set ℕ := {1, 2, 3, 4, 5}

-- Definition of set A
def A : Set ℕ := {1, 2, 3}

-- Definition of set B
def B : Set ℕ := {3, 4}

-- Definition of the complement of B in U
def CU (U : Set ℕ) (B : Set ℕ) : Set ℕ := {x | x ∈ U ∧ x ∉ B}

-- Expected result of the intersection
def result : Set ℕ := {1, 2}

-- The proof statement
theorem intersect_complement :
  A ∩ CU U B = result :=
sorry

end intersect_complement_l191_191319


namespace abc_sum_leq_three_l191_191028

open Real

theorem abc_sum_leq_three {a b c : ℝ} (ha : a > 0) (hb : b > 0) (hc : c > 0) (h : a^2 + b^2 + c^2 + a * b * c = 4) :
  a + b + c ≤ 3 :=
sorry

end abc_sum_leq_three_l191_191028


namespace bed_height_l191_191825

noncomputable def bed_length : ℝ := 8
noncomputable def bed_width : ℝ := 4
noncomputable def bags_of_soil : ℕ := 16
noncomputable def soil_per_bag : ℝ := 4
noncomputable def total_volume_of_soil : ℝ := bags_of_soil * soil_per_bag
noncomputable def number_of_beds : ℕ := 2
noncomputable def volume_per_bed : ℝ := total_volume_of_soil / number_of_beds

theorem bed_height :
  volume_per_bed / (bed_length * bed_width) = 1 :=
sorry

end bed_height_l191_191825


namespace solve_for_x_l191_191201

theorem solve_for_x (x : ℕ) (h1 : x > 0) (h2 : x % 6 = 0) (h3 : x^2 > 144) (h4 : x < 30) : x = 18 ∨ x = 24 :=
by
  sorry

end solve_for_x_l191_191201


namespace exists_sum_pair_l191_191578

theorem exists_sum_pair (n : ℕ) (a b : List ℕ) (h₁ : ∀ x ∈ a, x < n) (h₂ : ∀ y ∈ b, y < n) 
  (h₃ : List.Nodup a) (h₄ : List.Nodup b) (h₅ : a.length + b.length ≥ n) : ∃ x ∈ a, ∃ y ∈ b, x + y = n := by
  sorry

end exists_sum_pair_l191_191578


namespace top_face_not_rotated_by_90_l191_191917

-- Define the cube and the conditions of rolling and returning
structure Cube :=
  (initial_top_face_orientation : ℕ) -- an integer representation of the orientation of the top face
  (position : ℤ × ℤ) -- (x, y) coordinates on a 2D plane

def rolls_over_edges (c : Cube) : Cube :=
  sorry -- placeholder for the actual rolling operation

def returns_to_original_position (c : Cube) (original : Cube) : Prop :=
  c.position = original.position ∧ c.initial_top_face_orientation = original.initial_top_face_orientation

-- The main theorem to prove
theorem top_face_not_rotated_by_90 {c : Cube} (original : Cube) :
  returns_to_original_position c original → c.initial_top_face_orientation ≠ (original.initial_top_face_orientation + 1) % 4 :=
sorry

end top_face_not_rotated_by_90_l191_191917


namespace solve_problem_l191_191343

-- Define the constants c and d
variables (c d : ℝ)

-- Define the conditions of the problem
def condition1 : Prop := 
  (∀ x : ℝ, (x + c) * (x + d) * (x + 15) = 0 ↔ x = -c ∨ x = -d ∨ x = -15) ∧
  -4 ≠ -c ∧ -4 ≠ -d ∧ -4 ≠ -15

def condition2 : Prop := 
  (∀ x : ℝ, (x + 3 * c) * (x + 4) * (x + 9) = 0 ↔ x = -4) ∧
  d ≠ -4 ∧ d ≠ -15

-- We need to prove this final result under the given conditions
theorem solve_problem (h1 : condition1 c d) (h2 : condition2 c d) : 100 * c + d = -291 := 
  sorry

end solve_problem_l191_191343


namespace brenda_blisters_l191_191149

theorem brenda_blisters (blisters_per_arm : ℕ) (blisters_rest : ℕ) (arms : ℕ) :
  blisters_per_arm = 60 → blisters_rest = 80 → arms = 2 → 
  blisters_per_arm * arms + blisters_rest = 200 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  sorry

end brenda_blisters_l191_191149


namespace S_n_formula_l191_191980

-- Define S_n as described in the problem
noncomputable def S (n : ℕ) : ℕ :=
∑ i in Finset.range (n - 1),
    ∑ j in Finset.range i.succ.succ \ Finset.range i.succ,
      i * j

-- Proof statement asserting the equality
theorem S_n_formula (n : ℕ) (hn : 0 < n) :
  S n = (n * (n + 1) * (n - 1) * (3 * n + 2)) / 24 :=
sorry

end S_n_formula_l191_191980


namespace geometric_sum_5_l191_191846

noncomputable def geometric_sequence (a : ℕ → ℝ) : Prop :=
∀ n m : ℕ, ∃ r : ℝ, a (n + 1) = a n * r ∧ a (m + 1) = a m * r

theorem geometric_sum_5 (a : ℕ → ℝ) (h1 : geometric_sequence a) (h2 : a 2 * a 4 + 2 * a 3 * a 5 + a 4 * a 6 = 25) (h3 : ∀ n, 0 < a n) :
  a 3 + a 5 = 5 :=
sorry

end geometric_sum_5_l191_191846


namespace distance_between_cities_l191_191120

theorem distance_between_cities (d : ℝ)
  (meeting_point1 : d - 437 + 437 = d)
  (meeting_point2 : 3 * (d - 437) = 2 * d - 237) :
  d = 1074 :=
by
  sorry

end distance_between_cities_l191_191120


namespace percentage_increase_visitors_l191_191277

theorem percentage_increase_visitors 
  (V_Oct : ℕ)
  (V_Nov V_Dec : ℕ)
  (h1 : V_Oct = 100)
  (h2 : V_Dec = V_Nov + 15)
  (h3 : V_Oct + V_Nov + V_Dec = 345) : 
  (V_Nov - V_Oct) * 100 / V_Oct = 15 := 
by 
  sorry

end percentage_increase_visitors_l191_191277


namespace optionB_is_a9_l191_191144

-- Definitions of the expressions
def optionA (a : ℤ) : ℤ := a^3 + a^6
def optionB (a : ℤ) : ℤ := a^3 * a^6
def optionC (a : ℤ) : ℤ := a^10 - a
def optionD (a α : ℤ) : ℤ := α^12 / a^2

-- Theorem stating which option equals a^9
theorem optionB_is_a9 (a α : ℤ) : optionA a ≠ a^9 ∧ optionB a = a^9 ∧ optionC a ≠ a^9 ∧ optionD a α ≠ a^9 :=
by
  sorry

end optionB_is_a9_l191_191144


namespace set_intersection_is_correct_l191_191844

def setA : Set ℝ := {x | x^2 - 4 * x > 0}
def setB : Set ℝ := {x | abs (x - 1) ≤ 2}
def setIntersection : Set ℝ := {x | -1 ≤ x ∧ x < 0}

theorem set_intersection_is_correct :
  setA ∩ setB = setIntersection := 
by
  sorry

end set_intersection_is_correct_l191_191844


namespace binomial_12_3_eq_220_l191_191991

-- Definition of binomial coefficient
def binomial (n k : ℕ) : ℕ :=
  if k ≤ n then Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k)) else 0

-- Theorem to prove binomial(12, 3) = 220
theorem binomial_12_3_eq_220 : binomial 12 3 = 220 := by
  sorry

end binomial_12_3_eq_220_l191_191991


namespace age_solution_l191_191131

theorem age_solution (M S : ℕ) (h1 : M = S + 16) (h2 : M + 2 = 2 * (S + 2)) : S = 14 :=
by sorry

end age_solution_l191_191131


namespace angle_ratio_l191_191466

theorem angle_ratio (A B C : ℝ) (hA : A = 60) (hB : B = 80) (h_sum : A + B + C = 180) : B / C = 2 := by
  sorry

end angle_ratio_l191_191466


namespace arithmetic_expression_eval_l191_191834

theorem arithmetic_expression_eval : (10 - 9 + 8) * 7 + 6 - 5 * (4 - 3 + 2) - 1 = 53 :=
by
  sorry

end arithmetic_expression_eval_l191_191834


namespace bench_press_after_injury_and_training_l191_191062

theorem bench_press_after_injury_and_training
  (p : ℕ) (h1 : p = 500) (h2 : p' : ℕ) (h3 : preduce : ℕ) (h4 : p' = p - preduce) 
  (h5 : preduce = 4 * p / 5) (h6 : q : ℕ) (h7 : q = 3 * p') : 
  q = 300 := by
  sorry

end bench_press_after_injury_and_training_l191_191062


namespace part1_part2_l191_191562

noncomputable def f (x a b : ℝ) : ℝ := |x - a| + |x + b|

theorem part1 (a b : ℝ) (h₀ : a = 1) (h₁ : b = 2) :
  {x : ℝ | f x a b ≤ 5} = {x : ℝ | -3 ≤ x ∧ x ≤ 2} :=
by
  sorry

theorem part2 (a b : ℝ) (h_min_value : ∀ x : ℝ, f x a b ≥ 3) :
  a + b = 3 → (a > 0 ∧ b > 0) →
  (∃ a b : ℝ, a = b ∧ a + b = 3 ∧ (a = b → f x a b = 3)) →
  (∀ a b : ℝ, (a^2/b + b^2/a) ≥ 3) :=
by
  sorry

end part1_part2_l191_191562


namespace max_value_is_one_sixteenth_l191_191112

noncomputable def max_value_expression (t : ℝ) : ℝ :=
  (3^t - 4 * t) * t / 9^t

theorem max_value_is_one_sixteenth : 
  ∃ t : ℝ, max_value_expression t = 1 / 16 :=
sorry

end max_value_is_one_sixteenth_l191_191112


namespace root_range_of_f_eq_zero_solution_set_of_f_le_zero_l191_191849

variable (m : ℝ) (x : ℝ)

def f (x : ℝ) : ℝ := m * x^2 + (2 * m + 1) * x + 2

theorem root_range_of_f_eq_zero (h : ∃ r1 r2 : ℝ, r1 > 1 ∧ r2 < 1 ∧ f r1 = 0 ∧ f r2 = 0) : -1 < m ∧ m < 0 :=
sorry

theorem solution_set_of_f_le_zero : 
  (m = 0 -> ∀ x, f x ≤ 0 ↔ x ≤ - 2) ∧
  (m < 0 -> ∀ x, f x ≤ 0 ↔ -2 ≤ x ∧ x ≤ - (1/m)) ∧
  (0 < m ∧ m < 1/2 -> ∀ x, f x ≤ 0 ↔ - (1/m) ≤ x ∧ x ≤ - 2) ∧
  (m = 1/2 -> ∀ x, f x ≤ 0 ↔ x = - 2) ∧
  (m > 1/2 -> ∀ x, f x ≤ 0 ↔ -2 ≤ x ∧ x ≤ - (1/m)) :=
sorry

end root_range_of_f_eq_zero_solution_set_of_f_le_zero_l191_191849


namespace green_disks_count_l191_191964

-- Definitions of the conditions given in the problem
def total_disks : ℕ := 14
def red_disks (g : ℕ) : ℕ := 2 * g
def blue_disks (g : ℕ) : ℕ := g / 2

-- The theorem statement to prove
theorem green_disks_count (g : ℕ) (h : 2 * g + g + g / 2 = total_disks) : g = 4 :=
sorry

end green_disks_count_l191_191964


namespace range_of_a_l191_191609

variable (a : ℝ)
variable (f : ℝ → ℝ)

-- Conditions
def isOddFunction (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = -f x

def fWhenNegative (f : ℝ → ℝ) (a : ℝ) : Prop :=
  ∀ x : ℝ, x < 0 → f x = 9 * x + a^2 / x + 7

def fNonNegativeCondition (f : ℝ → ℝ) (a : ℝ) : Prop :=
  ∀ x : ℝ, x ≥ 0 → f x ≥ a + 1

-- Theorem to prove
theorem range_of_a (odd_f : isOddFunction f) (f_neg : fWhenNegative f a) 
  (nonneg_cond : fNonNegativeCondition f a) : 
  a ≤ -8 / 7 :=
by
  sorry

end range_of_a_l191_191609


namespace gcd_8251_6105_l191_191786

theorem gcd_8251_6105 : Nat.gcd 8251 6105 = 37 := by
  sorry

end gcd_8251_6105_l191_191786


namespace probability_gold_coin_biased_l191_191005

open Probability

-- Definition of events and probabilities
variables {Ω : Type*} (p : pmf Ω)
variables (A B : set Pred) -- Event A: Gold coin is biased; Event B: Observing the given sequence of coin flips.
variables (P : pred ℝ) -- Probability function

-- Assume mutually exclusive events A and not A.
axiom mutually_exclusive {Ω : Type} (p : pmf Ω) (A : set Ω) :
  P (A ∩ Aᶜ) = 0

-- Assumptions based on conditions from the problem
axiom gold_coin_biased (A : set Ω) : p.heads = 0.6
axiom silver_coin_fair (B : set Ω) : p.heads = 0.5
axiom gold_coin_event (A : set Ω) : P (B | A) = 0.15
axiom gold_coin_biased_prob (A : set Ω) : P (A) = 0.5
axiom silver_coin_event (Aᶜ : set Ω) : P (B | Aᶜ) = 0.12
axiom silver_coin_notbiased_prob (Aᶜ : set Ω) : P (Aᶜ) = 0.5

-- Translate Bayes' theorem to our setup in Lean 
noncomputable def bayes_theorem : ℝ := 
  (P (B | A) * P (A)) / (P (B | A) * P (A) + P (B | Aᶜ) * P (Aᶜ))

-- The final statement asserting the desired probability 
theorem probability_gold_coin_biased : bayes_theorem p A B = 5 / 9 := sorry

end probability_gold_coin_biased_l191_191005


namespace total_apples_picked_l191_191004

theorem total_apples_picked (benny_apples : ℕ) (dan_apples : ℕ) (h_benny : benny_apples = 2) (h_dan : dan_apples = 9) :
  benny_apples + dan_apples = 11 :=
by
  sorry

end total_apples_picked_l191_191004


namespace square_of_binomial_l191_191655

theorem square_of_binomial (k : ℝ) : (∃ a : ℝ, x^2 - 20 * x + k = (x - a)^2) → k = 100 :=
by {
  sorry
}

end square_of_binomial_l191_191655


namespace infection_average_l191_191134

theorem infection_average (x : ℕ) (h : 1 + x + x * (1 + x) = 196) : x = 13 :=
sorry

end infection_average_l191_191134


namespace train_cross_time_l191_191272

noncomputable def speed_kmh := 72
noncomputable def speed_mps : ℝ := speed_kmh * (1000 / 3600)
noncomputable def length_train := 180
noncomputable def length_bridge := 270
noncomputable def total_distance := length_train + length_bridge
noncomputable def time_to_cross := total_distance / speed_mps

theorem train_cross_time :
  time_to_cross = 22.5 := 
sorry

end train_cross_time_l191_191272


namespace equilateral_triangle_ratio_l191_191945

-- Define the side length of the equilateral triangle
def side_length : ℝ := 10

-- Define the altitude of the equilateral triangle
def altitude (a : ℝ) : ℝ := a * (Real.sqrt 3) / 2

-- Define the area of the equilateral triangle
def area (a : ℝ) : ℝ := (a * altitude a) / 2

-- Define the perimeter of the equilateral triangle
def perimeter (a : ℝ) : ℝ := 3 * a

-- Define the ratio of area to perimeter
def ratio (a : ℝ) : ℝ := area a / perimeter a

theorem equilateral_triangle_ratio :
  ratio 10 = (5 * Real.sqrt 3) / 6 :=
by
  sorry

end equilateral_triangle_ratio_l191_191945


namespace dodecahedron_interior_diagonals_count_l191_191835

-- Define a dodecahedron structure
structure Dodecahedron :=
  (vertices : ℕ)
  (edges_per_vertex : ℕ)
  (faces_per_vertex : ℕ)

-- Define the property of a dodecahedron
def dodecahedron_property : Dodecahedron :=
{
  vertices := 20,
  edges_per_vertex := 3,
  faces_per_vertex := 3
}

-- The theorem statement
theorem dodecahedron_interior_diagonals_count (d : Dodecahedron)
  (h1 : d.vertices = 20)
  (h2 : d.edges_per_vertex = 3)
  (h3 : d.faces_per_vertex = 3) : 
  (d.vertices * (d.vertices - d.edges_per_vertex)) / 2 = 160 :=
by
  sorry

end dodecahedron_interior_diagonals_count_l191_191835


namespace scrooge_share_l191_191977

def leftover_pie : ℚ := 8 / 9

def share_each (x : ℚ) : Prop :=
  2 * x + 3 * x = leftover_pie

theorem scrooge_share (x : ℚ):
  share_each x → (2 * x = 16 / 45) := by
  sorry

end scrooge_share_l191_191977


namespace probability_of_A_given_B_l191_191717

-- Definitions of events
def tourist_attractions : List String := ["Pengyuan", "Jiuding Mountain", "Garden Expo Park", "Yunlong Lake", "Pan'an Lake"]

-- Probabilities for each scenario
noncomputable def P_AB : ℝ := 8 / 25
noncomputable def P_B : ℝ := 20 / 25
noncomputable def P_A_given_B : ℝ := 2 / 5

-- Proof statement
theorem probability_of_A_given_B : (P_AB / P_B) = P_A_given_B :=
by
  sorry

end probability_of_A_given_B_l191_191717


namespace total_kids_played_l191_191728

-- Definitions based on conditions
def kidsMonday : Nat := 17
def kidsTuesday : Nat := 15
def kidsWednesday : Nat := 2

-- Total kids calculation
def totalKids : Nat := kidsMonday + kidsTuesday + kidsWednesday

-- Theorem to prove
theorem total_kids_played (Julia : Prop) : totalKids = 34 :=
by
  -- Using sorry to skip the proof
  sorry

end total_kids_played_l191_191728


namespace proper_subset_singleton_l191_191852

theorem proper_subset_singleton : ∀ (P : Set ℕ), P = {0} → (∃ S, S ⊂ P ∧ S = ∅) :=
by
  sorry

end proper_subset_singleton_l191_191852


namespace calculate_rows_l191_191072

-- Definitions based on conditions
def totalPecanPies : ℕ := 16
def totalApplePies : ℕ := 14
def piesPerRow : ℕ := 5

-- The goal is to prove the total rows of pies
theorem calculate_rows : (totalPecanPies + totalApplePies) / piesPerRow = 6 := by
  sorry

end calculate_rows_l191_191072


namespace number_of_possible_values_of_a_l191_191232

theorem number_of_possible_values_of_a :
  ∃ a_count : ℕ, (∃ (a b c d : ℕ), a > b ∧ b > c ∧ c > d ∧ a + b + c + d = 2020 ∧ a^2 - b^2 + c^2 - d^2 = 2020 ∧ a_count = 501) :=
sorry

end number_of_possible_values_of_a_l191_191232


namespace balance_balls_l191_191739

variables (R B O P : ℝ)

-- Conditions
axiom h1 : 4 * R = 8 * B
axiom h2 : 3 * O = 6 * B
axiom h3 : 8 * B = 6 * P

-- Proof problem
theorem balance_balls : 5 * R + 3 * O + 3 * P = 20 * B :=
by sorry

end balance_balls_l191_191739


namespace jenny_hours_left_l191_191472

theorem jenny_hours_left : 
  ∀ (total_hours research_hours proposal_hours : ℕ), 
  total_hours = 20 → 
  research_hours = 10 → 
  proposal_hours = 2 → 
  total_hours - (research_hours + proposal_hours) = 8 :=
by
  intros total_hours research_hours proposal_hours h_total h_research h_proposal
  rw [h_total, h_research, h_proposal]
  norm_num

end jenny_hours_left_l191_191472


namespace total_original_cost_l191_191141

theorem total_original_cost (discounted_price1 discounted_price2 discounted_price3 : ℕ) 
  (discount_rate1 discount_rate2 discount_rate3 : ℚ)
  (h1 : discounted_price1 = 4400)
  (h2 : discount_rate1 = 0.56)
  (h3 : discounted_price2 = 3900)
  (h4 : discount_rate2 = 0.35)
  (h5 : discounted_price3 = 2400)
  (h6 : discount_rate3 = 0.20) :
  (discounted_price1 / (1 - discount_rate1) + discounted_price2 / (1 - discount_rate2) 
    + discounted_price3 / (1 - discount_rate3) = 19000) :=
by
  sorry

end total_original_cost_l191_191141


namespace rectangle_width_decreased_l191_191903

theorem rectangle_width_decreased (L W : ℝ) (hL : L > 0) (hW : W > 0) :
  let L' := 1.5 * L in
  let W' := (L * W) / L' in
  ((W - W') / W) * 100 = 33.3333 :=
by
  sorry

end rectangle_width_decreased_l191_191903


namespace exponential_sequence_term_eq_l191_191843

-- Definitions for the conditions
variable {α : Type} [CommRing α] (q : α)
def a (n : ℕ) : α := q * (q ^ (n - 1))

-- Statement of the problem
theorem exponential_sequence_term_eq : a q 9 = a q 3 * a q 7 := by
  sorry

end exponential_sequence_term_eq_l191_191843


namespace cost_of_cherries_l191_191411

theorem cost_of_cherries (total_spent amount_for_grapes amount_for_cherries : ℝ)
  (h1 : total_spent = 21.93)
  (h2 : amount_for_grapes = 12.08)
  (h3 : amount_for_cherries = total_spent - amount_for_grapes) :
  amount_for_cherries = 9.85 :=
sorry

end cost_of_cherries_l191_191411


namespace bracelet_ratio_l191_191281

-- Definition of the conditions
def initial_bingley_bracelets : ℕ := 5
def kelly_bracelets_given : ℕ := 16 / 4
def total_bracelets_after_receiving := initial_bingley_bracelets + kelly_bracelets_given
def bingley_remaining_bracelets : ℕ := 6
def bingley_bracelets_given := total_bracelets_after_receiving - bingley_remaining_bracelets

-- Lean 4 Statement
theorem bracelet_ratio : bingley_bracelets_given * 3 = total_bracelets_after_receiving := by
  sorry

end bracelet_ratio_l191_191281


namespace age_problem_l191_191406

theorem age_problem (F : ℝ) (M : ℝ) (Y : ℝ)
  (hF : F = 40.00000000000001)
  (hM : M = (2/5) * F)
  (hY : M + Y = (1/2) * (F + Y)) :
  Y = 8.000000000000002 :=
sorry

end age_problem_l191_191406


namespace second_runner_stop_time_l191_191517

-- Definitions provided by the conditions
def pace_first := 8 -- pace of the first runner in minutes per mile
def pace_second := 7 -- pace of the second runner in minutes per mile
def time_elapsed := 56 -- time elapsed in minutes before the second runner stops
def distance_first := time_elapsed / pace_first -- distance covered by the first runner in miles
def distance_second := time_elapsed / pace_second -- distance covered by the second runner in miles
def distance_gap := distance_second - distance_first -- gap between the runners in miles

-- Statement of the proof problem
theorem second_runner_stop_time :
  8 = distance_gap * pace_first :=
by
sorry

end second_runner_stop_time_l191_191517


namespace ratio_of_area_to_perimeter_of_equilateral_triangle_l191_191937

theorem ratio_of_area_to_perimeter_of_equilateral_triangle (s : ℕ) : s = 10 → (let A := (s^2 * sqrt 3) / 4, P := 3 * s in A / P = 5 * sqrt 3 / 6) :=
by
  intro h,
  sorry

end ratio_of_area_to_perimeter_of_equilateral_triangle_l191_191937


namespace find_angle_EHG_l191_191469

noncomputable def angle_EHG (angle_EFG : ℝ) (angle_GHE : ℝ) : ℝ := angle_GHE - angle_EFG
 
theorem find_angle_EHG : 
  ∀ (EF GH : Prop) (angle_EFG angle_GHE : ℝ), (EF ∧ GH) → 
    EF ∧ GH ∧ angle_EFG = 50 ∧ angle_GHE = 80 → angle_EHG angle_EFG angle_GHE = 30 := 
by 
  intros EF GH angle_EFG angle_GHE h1 h2
  sorry

end find_angle_EHG_l191_191469


namespace find_n_l191_191592

noncomputable def problem_statement (m n : ℤ) : Prop :=
  (∀ x : ℝ, x^2 - (m + 2) * x + (m - 2) = 0 → ∃ x1 x2 : ℝ, x1 ≠ 0 ∧ x2 ≠ 0 ∧ x1 * x2 < 0 ∧ x1 > |x2|) ∧
  (∃ r1 r2 : ℚ, r1 * r2 = 2 ∧ m * (r1 * r1 + r2 * r2) = (n - 2) * (r1 + r2) + m^2 - 3)

theorem find_n (m : ℤ) (hm : -2 < m ∧ m < 2) : 
  problem_statement m 5 ∨ problem_statement m (-1) :=
sorry

end find_n_l191_191592


namespace infection_average_l191_191135

theorem infection_average (x : ℕ) (h : 1 + x + x * (1 + x) = 196) : x = 13 :=
sorry

end infection_average_l191_191135


namespace units_digit_of_result_l191_191776

def tens_plus_one (a b : ℕ) : Prop := a = b + 1

theorem units_digit_of_result (a b : ℕ) (h : tens_plus_one a b) :
  ((10 * a + b) - (10 * b + a)) % 10 = 9 :=
by
  -- Let's mark this part as incomplete using sorry.
  sorry

end units_digit_of_result_l191_191776


namespace pizza_cost_difference_l191_191129

theorem pizza_cost_difference :
  let p := 12 -- Cost of plain pizza
  let m := 3 -- Cost of mushrooms
  let o := 4 -- Cost of olives
  let s := 12 -- Total number of slices
  (m + o + p) / s * 10 - (m + o + p) / s * 2 = 12.67 :=
by
  sorry

end pizza_cost_difference_l191_191129


namespace equilateral_triangle_ratio_l191_191947

-- Define the side length of the equilateral triangle
def side_length : ℝ := 10

-- Define the altitude of the equilateral triangle
def altitude (a : ℝ) : ℝ := a * (Real.sqrt 3) / 2

-- Define the area of the equilateral triangle
def area (a : ℝ) : ℝ := (a * altitude a) / 2

-- Define the perimeter of the equilateral triangle
def perimeter (a : ℝ) : ℝ := 3 * a

-- Define the ratio of area to perimeter
def ratio (a : ℝ) : ℝ := area a / perimeter a

theorem equilateral_triangle_ratio :
  ratio 10 = (5 * Real.sqrt 3) / 6 :=
by
  sorry

end equilateral_triangle_ratio_l191_191947


namespace sequence_infinite_pos_neg_l191_191771

theorem sequence_infinite_pos_neg (a : ℕ → ℝ)
  (h : ∀ k : ℕ, a (k + 1) = (k * a k + 1) / (k - a k)) :
  ∃ (P N : ℕ → Prop), (∀ n, P n ↔ 0 < a n) ∧ (∀ n, N n ↔ a n < 0) ∧ 
  (∀ m, ∃ n, n > m ∧ P n) ∧ (∀ m, ∃ n, n > m ∧ N n) := 
sorry

end sequence_infinite_pos_neg_l191_191771


namespace max_marks_l191_191805

theorem max_marks (M : ℝ) (h1 : 0.42 * M = 80) : M = 190 :=
by
  sorry

end max_marks_l191_191805


namespace sabrina_total_leaves_l191_191354

theorem sabrina_total_leaves (num_basil num_sage num_verbena : ℕ)
  (h1 : num_basil = 2 * num_sage)
  (h2 : num_sage = num_verbena - 5)
  (h3 : num_basil = 12) :
  num_sage + num_verbena + num_basil = 29 :=
by
  sorry

end sabrina_total_leaves_l191_191354


namespace total_amount_l191_191965

-- Declare the variables
variables (A B C : ℕ)

-- Introduce the conditions as hypotheses
theorem total_amount (h1 : A = B + 40) (h2 : C = A + 30) (h3 : B = 290) : 
  A + B + C = 980 := 
by {
  sorry
}

end total_amount_l191_191965


namespace total_pastries_l191_191416

-- Defining the initial conditions
def Grace_pastries : ℕ := 30
def Calvin_pastries : ℕ := Grace_pastries - 5
def Phoebe_pastries : ℕ := Grace_pastries - 5
def Frank_pastries : ℕ := Calvin_pastries - 8

-- The theorem we want to prove
theorem total_pastries : 
  Calvin_pastries + Phoebe_pastries + Frank_pastries + Grace_pastries = 97 := by
  sorry

end total_pastries_l191_191416


namespace thomas_total_blocks_l191_191635

-- Definitions according to the conditions
def a1 : Nat := 7
def a2 : Nat := a1 + 3
def a3 : Nat := a2 - 6
def a4 : Nat := a3 + 10
def a5 : Nat := 2 * a2

-- The total number of blocks
def total_blocks : Nat := a1 + a2 + a3 + a4 + a5

-- The proof statement
theorem thomas_total_blocks :
  total_blocks = 55 := 
sorry

end thomas_total_blocks_l191_191635


namespace max_vertex_value_in_cube_l191_191896

def transform_black (v : ℕ) (e1 e2 e3 : ℕ) : ℕ :=
  e1 + e2 + e3

def transform_white (v : ℕ) (d1 d2 d3 : ℕ) : ℕ :=
  d1 + d2 + d3

def max_value_after_transformation (initial_values : Fin 8 → ℕ) : ℕ :=
  -- Combination of transformations and iterations are derived here
  42648

theorem max_vertex_value_in_cube :
  ∀ (initial_values : Fin 8 → ℕ),
  (∀ i, 1 ≤ initial_values i ∧ initial_values i ≤ 8) →
  (∃ (final_value : ℕ), final_value = max_value_after_transformation initial_values) → final_value = 42648 :=
by {
  sorry
}

end max_vertex_value_in_cube_l191_191896


namespace option_B_option_D_l191_191438

noncomputable def curve_C (x y : ℝ) : Prop := x^2 + y^2 - 2*x - 2 = 0

-- The maximum value of (y + 1) / (x + 1) is 2 + sqrt(6)
theorem option_B (x y : ℝ) (h : curve_C x y) :
  ∃ k, (y + 1) / (x + 1) = k ∧ k = 2 + Real.sqrt 6 :=
sorry

-- A tangent line through the point (0, √2) on curve C has the equation x - √2 * y + 2 = 0
theorem option_D (h : curve_C 0 (Real.sqrt 2)) :
  ∃ a b c, a * 0 + b * Real.sqrt 2 + c = 0 ∧ c = 2 ∧ a = 1 ∧ b = - Real.sqrt 2 :=
sorry

end option_B_option_D_l191_191438


namespace pyramid_volume_is_sqrt3_l191_191577

noncomputable def volume_of_pyramid := 
  let base_area : ℝ := 2 * Real.sqrt 3
  let angle_ABC : ℝ := 60
  let BC := 2
  let EC := BC
  let FB := BC / 2
  let height : ℝ := Real.sqrt 3
  let pyramid_volume := 1/3 * EC * FB * height
  pyramid_volume

theorem pyramid_volume_is_sqrt3 : volume_of_pyramid = Real.sqrt 3 :=
by sorry

end pyramid_volume_is_sqrt3_l191_191577


namespace reams_for_haley_correct_l191_191854

-- Definitions: 
-- total reams = 5
-- reams for sister = 3
-- reams for Haley = ?

def total_reams : Nat := 5
def reams_for_sister : Nat := 3
def reams_for_haley : Nat := total_reams - reams_for_sister

-- The proof problem: prove reams_for_haley = 2 given the conditions.
theorem reams_for_haley_correct : reams_for_haley = 2 := by 
  sorry

end reams_for_haley_correct_l191_191854


namespace set_A_roster_l191_191038

def is_nat_not_greater_than_4 (x : ℕ) : Prop := x ≤ 4

def A : Set ℕ := {x | is_nat_not_greater_than_4 x}

theorem set_A_roster : A = {0, 1, 2, 3, 4} := by
  sorry

end set_A_roster_l191_191038


namespace equation1_solution_equation2_solution_l191_191497

theorem equation1_solution (x : ℝ) : (x - 4)^2 - 9 = 0 ↔ (x = 7 ∨ x = 1) := 
sorry

theorem equation2_solution (x : ℝ) : (x + 1)^3 = -27 ↔ (x = -4) := 
sorry

end equation1_solution_equation2_solution_l191_191497


namespace fraction_fliers_afternoon_l191_191252

theorem fraction_fliers_afternoon :
  ∀ (initial_fliers remaining_fliers next_day_fliers : ℕ),
    initial_fliers = 2500 →
    next_day_fliers = 1500 →
    remaining_fliers = initial_fliers - initial_fliers / 5 →
    (remaining_fliers - next_day_fliers) / remaining_fliers = 1 / 4 :=
by
  intros initial_fliers remaining_fliers next_day_fliers
  sorry

end fraction_fliers_afternoon_l191_191252


namespace balls_color_equality_l191_191624

theorem balls_color_equality (r g b: ℕ) (h1: r + g + b = 20) (h2: b ≥ 7) (h3: r ≥ 4) (h4: b = 2 * g) : 
  r = b ∨ r = g :=
by
  sorry

end balls_color_equality_l191_191624


namespace max_stories_on_odd_pages_l191_191530

theorem max_stories_on_odd_pages 
    (stories : Fin 30 -> Fin 31) 
    (h_unique : Function.Injective stories) 
    (h_bounds : ∀ i, stories i < 31)
    : ∃ n, n = 23 ∧ ∃ f : Fin n -> Fin 30, ∀ j, f j % 2 = 1 := 
sorry

end max_stories_on_odd_pages_l191_191530


namespace smaller_cube_edge_length_l191_191963

theorem smaller_cube_edge_length (x : ℝ) 
    (original_edge_length : ℝ := 7)
    (increase_percentage : ℝ := 600) 
    (original_surface_area_formula : ℝ := 6 * original_edge_length^2)
    (new_surface_area_formula : ℝ := (1 + increase_percentage / 100) * original_surface_area_formula) :
  ∃ x : ℝ, 6 * x^2 * (original_edge_length ^ 3 / x ^ 3) = new_surface_area_formula → x = 1 := by
  sorry

end smaller_cube_edge_length_l191_191963


namespace odd_function_f_neg_one_l191_191344

open Real

noncomputable def f (x : ℝ) : ℝ :=
  if h : 0 < x ∧ x < 2 then 2^x else 0 -- Placeholder; actual implementation skipped for simplicity

theorem odd_function_f_neg_one :
  (∀ x, f (-x) = -f x) ∧ (∀ x, (0 < x ∧ x < 2) → f x = 2^x) → 
  f (-1) = -2 :=
by
  intros h
  let odd_property := h.1
  let condition_in_range := h.2
  sorry

end odd_function_f_neg_one_l191_191344


namespace park_area_l191_191957

theorem park_area (length breadth : ℝ) (x : ℝ) 
  (h1 : length = 3 * x) 
  (h2 : breadth = x) 
  (h3 : 2 * length + 2 * breadth = 800) 
  (h4 : 12 * (4 / 60) * 1000 = 800) : 
  length * breadth = 30000 := by
sorry

end park_area_l191_191957


namespace part_a_part_b_part_c_l191_191402

-- Part a
def can_ratings_increase_after_first_migration (QA_before : ℚ) (QB_before : ℚ) (QA_after : ℚ) (QB_after : ℚ) : Prop :=
  QA_before < QA_after ∧ QB_before < QB_after

-- Part b
def can_ratings_increase_after_second_migration (QA_after_first : ℚ) (QB_after_first : ℚ) (QA_after_second : ℚ) (QB_after_second : ℚ) : Prop :=
  QA_after_second ≤ QA_after_first ∨ QB_after_second ≤ QB_after_first

-- Part c
def can_all_ratings_increase_after_reversed_migration (QA_before : ℚ) (QB_before : ℚ) (QC_before : ℚ) (QA_after_first : ℚ) (QB_after_first : ℚ) (QC_after_first : ℚ)
  (QA_after_second : ℚ) (QB_after_second : ℚ) (QC_after_second : ℚ) : Prop :=
  QA_before < QA_after_first ∧ QB_before < QB_after_first ∧ QC_before < QC_after_first ∧
  QA_after_first < QA_after_second ∧ QB_after_first < QB_after_second ∧ QC_after_first <= QC_after_second


-- Lean statements
theorem part_a (QA_before QA_after QB_before QB_after : ℚ) (Q_moved : ℚ) 
  (h : QA_before < QA_after ∧ QA_after < Q_moved ∧ QB_before < QB_after ∧ QB_after < Q_moved) : 
  can_ratings_increase_after_first_migration QA_before QB_before QA_after QB_after := 
by sorry

theorem part_b (QA_after_first QB_after_first QA_after_second QB_after_second : ℚ):
  ¬ can_ratings_increase_after_second_migration QA_after_first QB_after_first QA_after_second QB_after_second := 
by sorry

theorem part_c (QA_before QB_before QC_before QA_after_first QB_after_first QC_after_first
  QA_after_second QB_after_second QC_after_second: ℚ)
  (h: QA_before < QA_after_first ∧ QB_before < QB_after_first ∧ QC_before < QC_after_first ∧ 
      QA_after_first < QA_after_second ∧ QB_after_first < QB_after_second) :
   can_all_ratings_increase_after_reversed_migration QA_before QB_before QC_before QA_after_first QB_after_first QC_after_first QA_after_second QB_after_second QC_after_second :=
by sorry

end part_a_part_b_part_c_l191_191402


namespace diagonal_length_of_cuboid_l191_191263

theorem diagonal_length_of_cuboid
  (a b c : ℝ)
  (h1 : a * b = Real.sqrt 2)
  (h2 : b * c = Real.sqrt 3)
  (h3 : c * a = Real.sqrt 6) : 
  Real.sqrt (a^2 + b^2 + c^2) = Real.sqrt 6 := 
sorry

end diagonal_length_of_cuboid_l191_191263


namespace solve_for_x_l191_191750

theorem solve_for_x (x : ℚ) (h : (7 * x) / (x - 2) + 4 / (x - 2) = 6 / (x - 2)) : x = 2 / 7 :=
sorry

end solve_for_x_l191_191750


namespace tan_alpha_eq_one_l191_191720

noncomputable def rho (theta : ℝ) : ℝ := sorry
noncomputable def parametric_line_x (t α : ℝ) : ℝ := 2 + t * Real.cos α
noncomputable def parametric_line_y (t α : ℝ) : ℝ := 3 + t * Real.sin α

theorem tan_alpha_eq_one (α : ℝ) (t : ℝ) (rho : ℝ → ℝ) :
  (∀ θ, rho θ * Real.sin θ^2 + 4 * Real.sin θ - rho θ = 0) →
  (parametric_line_x t α, parametric_line_y t α) = (0, 1) →
  Real.tan α = 1 :=
by
  simp [parametric_line_x, parametric_line_y, Real.tan]
  sorry

end tan_alpha_eq_one_l191_191720


namespace convex_polyhedron_has_triangular_face_l191_191326

def convex_polyhedron : Type := sorry -- placeholder for the type of convex polyhedra
def face (P : convex_polyhedron) : Type := sorry -- placeholder for the type of faces of a polyhedron
def vertex (P : convex_polyhedron) : Type := sorry -- placeholder for the type of vertices of a polyhedron
def edge (P : convex_polyhedron) : Type := sorry -- placeholder for the type of edges of a polyhedron

-- The number of edges meeting at a specific vertex
def vertex_degree (P : convex_polyhedron) (v : vertex P) : ℕ := sorry

-- Number of edges or vertices on a specific face
def face_sides (P : convex_polyhedron) (f : face P) : ℕ := sorry

-- A polyhedron is convex
def is_convex (P : convex_polyhedron) : Prop := sorry

-- A face is a triangle if it has 3 sides
def is_triangle (P : convex_polyhedron) (f : face P) := face_sides P f = 3

-- The problem statement in Lean 4
theorem convex_polyhedron_has_triangular_face
  (P : convex_polyhedron)
  (h1 : is_convex P)
  (h2 : ∀ v : vertex P, vertex_degree P v ≥ 4) :
  ∃ f : face P, is_triangle P f :=
sorry

end convex_polyhedron_has_triangular_face_l191_191326


namespace equilateral_triangle_area_to_perimeter_ratio_l191_191935

theorem equilateral_triangle_area_to_perimeter_ratio
  (a : ℝ) (h : a = 10) :
  let altitude := a * (Real.sqrt 3 / 2) in
  let area := (1 / 2) * a * altitude in
  let perimeter := 3 * a in
  area / perimeter = 5 * (Real.sqrt 3) / 6 := 
by
  sorry

end equilateral_triangle_area_to_perimeter_ratio_l191_191935


namespace intersection_A_B_l191_191311

def A : Set ℝ := {x | 2*x - 1 ≤ 0}
def B : Set ℝ := {x | 0 < x ∧ x < 1}

theorem intersection_A_B : A ∩ B = {x | 0 < x ∧ x ≤ 1/2} := 
by 
  sorry

end intersection_A_B_l191_191311


namespace equilateral_triangle_ratio_is_correct_l191_191927

noncomputable def equilateral_triangle_area_perimeter_ratio (a : ℝ) (h_eq : a = 10) : ℝ :=
  let altitude := (Real.sqrt 3 / 2) * a
  let area := (1 / 2) * a * altitude
  let perimeter := 3 * a
  area / perimeter

theorem equilateral_triangle_ratio_is_correct :
  equilateral_triangle_area_perimeter_ratio 10 (by rfl) = 5 * Real.sqrt 3 / 6 :=
by
  sorry

end equilateral_triangle_ratio_is_correct_l191_191927


namespace find_smaller_integer_l191_191228

theorem find_smaller_integer (x : ℤ) (h1 : ∃ y : ℤ, y = 2 * x) (h2 : x + 2 * x = 96) : x = 32 :=
sorry

end find_smaller_integer_l191_191228


namespace fraction_value_l191_191243

theorem fraction_value : (3 - (-3)) / (2 - 1) = 6 := 
by
  sorry

end fraction_value_l191_191243


namespace smallest_base_for_80_l191_191116

-- Define the problem in terms of inequalities
def smallest_base (n : ℕ) (d : ℕ) :=
  ∃ b : ℕ, b > 1 ∧ b <= (n^(1/d)) ∧ (n^(1/(d+1))) < (b + 1)

-- Assertion that the smallest whole number b such that 80 can be expressed in base b using only three digits
theorem smallest_base_for_80 : ∃ b, smallest_base 80 3 ∧ b = 5 :=
  sorry

end smallest_base_for_80_l191_191116


namespace binomial_12_3_eq_220_l191_191993

-- Definition of binomial coefficient
def binomial (n k : ℕ) : ℕ :=
  if k ≤ n then Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k)) else 0

-- Theorem to prove binomial(12, 3) = 220
theorem binomial_12_3_eq_220 : binomial 12 3 = 220 := by
  sorry

end binomial_12_3_eq_220_l191_191993


namespace cone_volume_ratio_l191_191559

theorem cone_volume_ratio (rC hC rD hD : ℝ) (h_rC : rC = 10) (h_hC : hC = 20) (h_rD : rD = 20) (h_hD : hD = 10) :
  ((1/3) * π * rC^2 * hC) / ((1/3) * π * rD^2 * hD) = 1/2 :=
by 
  sorry

end cone_volume_ratio_l191_191559


namespace sulfuric_acid_moles_used_l191_191693

-- Definitions and conditions
def iron_moles : ℕ := 2
def iron_ii_sulfate_moles_produced : ℕ := 2
def sulfuric_acid_to_iron_ratio : ℕ := 1

-- Proof statement
theorem sulfuric_acid_moles_used {H2SO4_moles : ℕ} 
  (h_fe_reacts : H2SO4_moles = iron_moles * sulfuric_acid_to_iron_ratio) 
  (h_fe produces: iron_ii_sulfate_moles_produced = iron_moles) : H2SO4_moles = 2 :=
by
  sorry

end sulfuric_acid_moles_used_l191_191693


namespace quadratic_inequality_solution_empty_l191_191241

theorem quadratic_inequality_solution_empty (m : ℝ) :
  (∀ x : ℝ, ((m + 1) * x^2 - m * x + m - 1 < 0) → false) →
  (m ≥ (2 * Real.sqrt 3) / 3 ∨ m ≤ -(2 * Real.sqrt 3) / 3) :=
by
  sorry

end quadratic_inequality_solution_empty_l191_191241


namespace math_quiz_scores_stability_l191_191601

theorem math_quiz_scores_stability :
  let avgA := (90 + 82 + 88 + 96 + 94) / 5
  let avgB := (94 + 86 + 88 + 90 + 92) / 5
  let varA := ((90 - avgA) ^ 2 + (82 - avgA) ^ 2 + (88 - avgA) ^ 2 + (96 - avgA) ^ 2 + (94 - avgA) ^ 2) / 5
  let varB := ((94 - avgB) ^ 2 + (86 - avgB) ^ 2 + (88 - avgB) ^ 2 + (90 - avgB) ^ 2 + (92 - avgB) ^ 2) / 5
  avgA = avgB ∧ varB < varA :=
by
  sorry

end math_quiz_scores_stability_l191_191601


namespace max_value_f_on_interval_l191_191908

noncomputable def f (x : ℝ) : ℝ := Real.exp x - x

theorem max_value_f_on_interval : 
  ∃ x ∈ Set.Icc (-1 : ℝ) 1, (∀ y ∈ Set.Icc (-1 : ℝ) 1, f y ≤ f x) ∧ f x = Real.exp 1 - 1 :=
sorry

end max_value_f_on_interval_l191_191908


namespace bob_same_color_probability_is_1_over_28_l191_191804

def num_marriages : ℕ := 9
def red_marbles : ℕ := 3
def blue_marbles : ℕ := 3
def green_marbles : ℕ := 3

def david_marbles : ℕ := 3
def alice_marbles : ℕ := 3
def bob_marbles : ℕ := 3

def total_ways : ℕ := 1680
def favorable_ways : ℕ := 60
def probability_bob_same_color := favorable_ways / total_ways

theorem bob_same_color_probability_is_1_over_28 : probability_bob_same_color = (1 : ℚ) / 28 := by
  sorry

end bob_same_color_probability_is_1_over_28_l191_191804


namespace mike_picked_12_pears_l191_191729

theorem mike_picked_12_pears (k_picked k_gave_away k_m_together k_left m_left : ℕ) 
  (hkp : k_picked = 47) 
  (hkg : k_gave_away = 46) 
  (hkt : k_m_together = 13)
  (hkl : k_left = k_picked - k_gave_away) 
  (hlt : k_m_left = k_left + m_left) : 
  m_left = 12 := by
  sorry

end mike_picked_12_pears_l191_191729


namespace sum_smallest_largest_l191_191367

theorem sum_smallest_largest (n a : ℕ) (h_even_n : n % 2 = 0) (y x : ℕ)
  (h_y : y = a + n - 1)
  (h_x : x = (a + 3 * (n / 3 - 1)) * (n / 3)) : 
  2 * y = a + (a + 2 * (n - 1)) :=
by
  sorry

end sum_smallest_largest_l191_191367


namespace joggers_difference_l191_191544

-- Define the conditions as per the problem statement
variables (Tyson Alexander Christopher : ℕ)
variable (H1 : Alexander = Tyson + 22)
variable (H2 : Christopher = 20 * Tyson)
variable (H3 : Christopher = 80)

-- The theorem statement to prove Christopher bought 54 more joggers than Alexander
theorem joggers_difference : (Christopher - Alexander) = 54 :=
  sorry

end joggers_difference_l191_191544


namespace minimum_value_expression_l191_191567

theorem minimum_value_expression :
  ∃ x y : ℝ, ∀ x y : ℝ, 3 * x ^ 2 + 4 * x * y + 2 * y ^ 2 - 6 * x + 8 * y + 9 ≥ -10 :=
by
  sorry

end minimum_value_expression_l191_191567


namespace factor_as_complete_square_l191_191704

theorem factor_as_complete_square (k : ℝ) : (∃ a : ℝ, x^2 + k*x + 9 = (x + a)^2) ↔ k = 6 ∨ k = -6 := 
sorry

end factor_as_complete_square_l191_191704


namespace sandra_fathers_contribution_ratio_l191_191886

theorem sandra_fathers_contribution_ratio :
  let saved := 10
  let mother := 4
  let candy_cost := 0.5
  let jellybean_cost := 0.2
  let candies := 14
  let jellybeans := 20
  let remaining := 11
  let total_cost := candies * candy_cost + jellybeans * jellybean_cost
  let total_amount := total_cost + remaining
  let amount_without_father := saved + mother
  let father := total_amount - amount_without_father
  (father / mother) = 2 := by 
  sorry

end sandra_fathers_contribution_ratio_l191_191886


namespace find_a_l191_191297

theorem find_a (a : ℝ) (h : ∫ x in -a..a, (2 * x - 1) = -8) : a = 4 :=
sorry

end find_a_l191_191297


namespace swap_columns_produce_B_l191_191342

variables {n : ℕ} (A : Matrix (Fin n) (Fin n) (Fin n))

def K (B : Matrix (Fin n) (Fin n) (Fin n)) : ℕ :=
  Fintype.card {ij : (Fin n) × (Fin n) // B ij.1 ij.2 = ij.2}

theorem swap_columns_produce_B (A : Matrix (Fin n) (Fin n) (Fin n)) :
  ∃ (B : Matrix (Fin n) (Fin n) (Fin n)), (∀ i, ∃ j, B i j = A i j) ∧ K B ≤ n :=
sorry

end swap_columns_produce_B_l191_191342


namespace binomial_12_3_equals_220_l191_191987

theorem binomial_12_3_equals_220 : Nat.choose 12 3 = 220 := by
  sorry

end binomial_12_3_equals_220_l191_191987


namespace binom_12_3_eq_220_l191_191989

theorem binom_12_3_eq_220 : nat.choose 12 3 = 220 :=
sorry

end binom_12_3_eq_220_l191_191989


namespace sum_of_first_12_terms_l191_191174

noncomputable def arithmetic_seq (a d : ℤ) (n : ℕ) : ℤ := a + n * d

def Sn (a d : ℤ) (n : ℕ) : ℤ := n * (2 * a + (n - 1) * d) / 2

theorem sum_of_first_12_terms (a d : ℤ) (h1 : a + d * 4 = 3 * (a + d * 2))
                             (h2 : a + d * 9 = 14) : Sn a d 12 = 84 := 
by
  sorry

end sum_of_first_12_terms_l191_191174


namespace square_of_binomial_eq_100_l191_191647

-- Given conditions
def is_square_of_binomial (p : ℝ[X]) : Prop :=
  ∃ b : ℝ, p = (X + C b)^2

-- The equivalent proof problem statement
theorem square_of_binomial_eq_100 (k : ℝ) :
  is_square_of_binomial (X^2 - 20 * X + C k) → k = 100 :=
by
  sorry

end square_of_binomial_eq_100_l191_191647


namespace intersection_eq_singleton_zero_l191_191189

-- Definition of the sets M and N
def M : Set ℤ := {0, 1}
def N : Set ℤ := { x | ∃ n : ℤ, x = 2 * n }

-- The theorem stating that the intersection of M and N is {0}
theorem intersection_eq_singleton_zero : M ∩ N = {0} :=
by
  sorry

end intersection_eq_singleton_zero_l191_191189


namespace handshakesCountIsCorrect_l191_191778

-- Define the number of gremlins and imps
def numGremlins : ℕ := 30
def numImps : ℕ := 20

-- Define the conditions based on the problem
def handshakesAmongGremlins : ℕ := (numGremlins * (numGremlins - 1)) / 2
def handshakesBetweenImpsAndGremlins : ℕ := numImps * numGremlins

-- Calculate the total handshakes
def totalHandshakes : ℕ := handshakesAmongGremlins + handshakesBetweenImpsAndGremlins

-- Prove that the total number of handshakes equals 1035
theorem handshakesCountIsCorrect : totalHandshakes = 1035 := by
  sorry

end handshakesCountIsCorrect_l191_191778


namespace distinct_terms_count_l191_191421

theorem distinct_terms_count
  (x y z w p q r s t : Prop)
  (h1 : ¬(x = y ∨ x = z ∨ x = w ∨ y = z ∨ y = w ∨ z = w))
  (h2 : ¬(p = q ∨ p = r ∨ p = s ∨ p = t ∨ q = r ∨ q = s ∨ q = t ∨ r = s ∨ r = t ∨ s = t)) :
  ∃ (n : ℕ), n = 20 := by
  sorry

end distinct_terms_count_l191_191421


namespace rationalize_denominator_l191_191075

theorem rationalize_denominator : (1 / (Real.sqrt 3 + 1)) = ((Real.sqrt 3 - 1) / 2) :=
by
  sorry

end rationalize_denominator_l191_191075


namespace square_of_binomial_eq_100_l191_191649

-- Given conditions
def is_square_of_binomial (p : ℝ[X]) : Prop :=
  ∃ b : ℝ, p = (X + C b)^2

-- The equivalent proof problem statement
theorem square_of_binomial_eq_100 (k : ℝ) :
  is_square_of_binomial (X^2 - 20 * X + C k) → k = 100 :=
by
  sorry

end square_of_binomial_eq_100_l191_191649


namespace total_cost_correct_l191_191534

variables (gravel_cost_per_ton : ℝ) (gravel_tons : ℝ)
variables (sand_cost_per_ton : ℝ) (sand_tons : ℝ)
variables (cement_cost_per_ton : ℝ) (cement_tons : ℝ)

noncomputable def total_cost : ℝ :=
  (gravel_cost_per_ton * gravel_tons) + (sand_cost_per_ton * sand_tons) + (cement_cost_per_ton * cement_tons)

theorem total_cost_correct :
  gravel_cost_per_ton = 30.5 → gravel_tons = 5.91 →
  sand_cost_per_ton = 40.5 → sand_tons = 8.11 →
  cement_cost_per_ton = 55.6 → cement_tons = 4.35 →
  total_cost gravel_cost_per_ton gravel_tons sand_cost_per_ton sand_tons cement_cost_per_ton cement_tons = 750.57 :=
by
  intros h1 h2 h3 h4 h5 h6
  rw [h1, h2, h3, h4, h5, h6]
  norm_num
  sorry

end total_cost_correct_l191_191534


namespace sum_of_permuted_numbers_not_all_ones_l191_191053

theorem sum_of_permuted_numbers_not_all_ones (x y : ℕ) (hx : ∀ d ∈ (Nat.digits 10 x), d ≠ 0)
  (hy : Nat.digits 10 y = List.perm (Nat.digits 10 x)) :
  ∃ d ∈ (Nat.digits 10 (x + y)), d ≠ 1 :=
by
  sorry

end sum_of_permuted_numbers_not_all_ones_l191_191053


namespace min_value_of_2x_plus_4y_l191_191576

noncomputable def minimum_value (x y : ℝ) : ℝ := 2^x + 4^y

theorem min_value_of_2x_plus_4y (x y : ℝ) (h : x + 2 * y = 3) : minimum_value x y = 4 * Real.sqrt 2 :=
by
  sorry

end min_value_of_2x_plus_4y_l191_191576


namespace find_C_coordinates_l191_191211

noncomputable def maximize_angle (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a > b) (x : ℝ) : Prop :=
  ∀ C : ℝ × ℝ, C = (x, 0) → x = Real.sqrt (a * b)

theorem find_C_coordinates (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a > b) :
  maximize_angle a b ha hb hab (Real.sqrt (a * b)) :=
by  sorry

end find_C_coordinates_l191_191211


namespace percentage_increase_decrease_exceeds_original_l191_191597

open Real

theorem percentage_increase_decrease_exceeds_original (p q M : ℝ) (hp : 0 < p) (hq1 : 0 < q) (hq2 : q < 100) (hM : 0 < M) :
  (M * (1 + p / 100) * (1 - q / 100) > M) ↔ (p > (100 * q) / (100 - q)) :=
by
  sorry

end percentage_increase_decrease_exceeds_original_l191_191597


namespace solve_proof_problem_l191_191859

noncomputable def proof_problem (f g : ℝ → ℝ) :=
  ∀ x y : ℝ, f (x + g y) = 2 * x + y → g (x + f y) = x / 2 + y

theorem solve_proof_problem (f g : ℝ → ℝ) (h : ∀ x y : ℝ, f (x + g y) = 2 * x + y) :
  ∀ x y : ℝ, g (x + f y) = x / 2 + y :=
sorry

end solve_proof_problem_l191_191859


namespace episodes_per_wednesday_l191_191224

theorem episodes_per_wednesday :
  ∀ (W : ℕ), (∃ (n_episodes : ℕ) (n_mondays : ℕ) (n_weeks : ℕ), 
    n_episodes = 201 ∧ n_mondays = 67 ∧ n_weeks = 67 
    ∧ n_weeks * W + n_mondays = n_episodes) 
    → W = 2 :=
by
  intro W
  rintro ⟨n_episodes, n_mondays, n_weeks, h1, h2, h3, h4⟩
  -- proof would go here
  sorry

end episodes_per_wednesday_l191_191224


namespace pieces_of_candy_l191_191379

def total_items : ℝ := 3554
def secret_eggs : ℝ := 145.0

theorem pieces_of_candy : (total_items - secret_eggs) = 3409 :=
by 
  sorry

end pieces_of_candy_l191_191379


namespace laptop_cost_l191_191818

theorem laptop_cost (L : ℝ) (smartphone_cost : ℝ) (total_cost : ℝ) (change : ℝ) (n_laptops n_smartphones : ℕ) 
  (hl_smartphone : smartphone_cost = 400) 
  (hl_laptops : n_laptops = 2) 
  (hl_smartphones : n_smartphones = 4) 
  (hl_total : total_cost = 3000)
  (hl_change : change = 200) 
  (hl_total_spent : total_cost - change = 2 * L + 4 * smartphone_cost) : 
  L = 600 :=
by 
  sorry

end laptop_cost_l191_191818


namespace students_move_bricks_l191_191323

variable (a b c : ℕ)

theorem students_move_bricks (h : a * b * c ≠ 0) : 
  (by let efficiency := (c : ℚ) / (a * b);
      let total_work := (a : ℚ);
      let required_time := total_work / efficiency;
      exact required_time = (a^2 * b) / (c^2)) := sorry

end students_move_bricks_l191_191323


namespace largest_value_of_d_l191_191065

noncomputable def maximum_possible_value_of_d 
  (a b c d : ℝ) 
  (h1 : a + b + c + d = 10) 
  (h2 : ab + ac + ad + bc + bd + cd = 17) : ℝ :=
  (5 + Real.sqrt 123) / 2

theorem largest_value_of_d 
  (a b c d : ℝ) 
  (h1 : a + b + c + d = 10) 
  (h2 : ab + ac + ad + bc + bd + cd = 17) : 
  d ≤ maximum_possible_value_of_d a b c d h1 h2 :=
sorry

end largest_value_of_d_l191_191065


namespace range_of_d_l191_191306

noncomputable def sn (n a1 d : ℝ) := (n / 2) * (2 * a1 + (n - 1) * d)

theorem range_of_d (a1 d : ℝ) (h_eq : (sn 2 a1 d) * (sn 4 a1 d) / 2 + (sn 3 a1 d) ^ 2 / 9 + 2 = 0) :
  d ∈ Set.Iic (-Real.sqrt 2) ∪ Set.Ici (Real.sqrt 2) :=
sorry

end range_of_d_l191_191306


namespace increase_in_average_l191_191288

theorem increase_in_average (s1 s2 s3 s4 s5: ℝ)
  (h1: s1 = 92) (h2: s2 = 86) (h3: s3 = 89) (h4: s4 = 94) (h5: s5 = 91):
  ( ((s1 + s2 + s3 + s4 + s5) / 5) - ((s1 + s2 + s3) / 3) ) = 1.4 :=
by
  sorry

end increase_in_average_l191_191288


namespace sector_area_l191_191237

theorem sector_area (α : ℝ) (r : ℝ) (hα : α = π / 3) (hr : r = 6) : 
  1/2 * r^2 * α = 6 * π :=
by {
  sorry
}

end sector_area_l191_191237


namespace stream_current_rate_proof_l191_191898

noncomputable def stream_current_rate (c : ℝ) : Prop :=
  ∃ (c : ℝ), (6 / (8 - c) + 6 / (8 + c) = 2) ∧ c = 4

theorem stream_current_rate_proof : stream_current_rate 4 :=
by {
  -- Proof to be provided here.
  sorry
}

end stream_current_rate_proof_l191_191898


namespace bottles_sold_tuesday_l191_191995

def initial_inventory : ℕ := 4500
def sold_monday : ℕ := 2445
def sold_days_wed_to_sun : ℕ := 50 * 5
def bottles_delivered_saturday : ℕ := 650
def final_inventory : ℕ := 1555

theorem bottles_sold_tuesday : 
  initial_inventory + bottles_delivered_saturday - sold_monday - sold_days_wed_to_sun - final_inventory = 900 := 
by
  sorry

end bottles_sold_tuesday_l191_191995


namespace ratio_men_to_women_on_team_l191_191662

theorem ratio_men_to_women_on_team (M W : ℕ) 
  (h1 : W = M + 6) 
  (h2 : M + W = 24) : 
  M / W = 3 / 5 := 
by 
  sorry

end ratio_men_to_women_on_team_l191_191662


namespace ellipse_circle_inequality_l191_191181

theorem ellipse_circle_inequality
  (a b : ℝ) (x y : ℝ)
  (x1 y1 x2 y2 : ℝ)
  (h_ellipse1 : (x1^2) / (a^2) + (y1^2) / (b^2) = 1)
  (h_ellipse2 : (x2^2) / (a^2) + (y2^2) / (b^2) = 1)
  (h_ab : a > b ∧ b > 0)
  (h_circle : (x - x1) * (x - x2) + (y - y1) * (y - y2) = 0) :
  x^2 + y^2 ≤ (3/2) * a^2 + (1/2) * b^2 :=
sorry

end ellipse_circle_inequality_l191_191181


namespace maximum_s_squared_l191_191822

-- Definitions based on our conditions
def semicircle_radius : ℝ := 5
def diameter_length : ℝ := 10

-- Statement of the problem (no proof, statement only)
theorem maximum_s_squared (A B C : ℝ×ℝ) (AC BC : ℝ) (h : AC + BC = s) :
    (A.2 = 0) ∧ (B.2 = 0) ∧ (dist A B = diameter_length) ∧
    (dist C (5,0) = semicircle_radius) ∧ (s = AC + BC) →
    s^2 ≤ 200 :=
sorry

end maximum_s_squared_l191_191822


namespace x_power_2023_zero_or_neg_two_l191_191227

variable {x : ℂ} -- Assuming x is a complex number to handle general roots of unity.

theorem x_power_2023_zero_or_neg_two 
  (h1 : (x - 1) * (x + 1) = x^2 - 1)
  (h2 : (x - 1) * (x^2 + x + 1) = x^3 - 1)
  (h3 : (x - 1) * (x^3 + x^2 + x + 1) = x^4 - 1)
  (pattern : (x - 1) * (x^5 + x^4 + x^3 + x^2 + x + 1) = 0) :
  x^2023 - 1 = 0 ∨ x^2023 - 1 = -2 :=
by
  sorry

end x_power_2023_zero_or_neg_two_l191_191227


namespace tea_or_coffee_indifference_l191_191425

open Classical

theorem tea_or_coffee_indifference : 
  (∀ (wants_tea wants_coffee : Prop), (wants_tea ∨ wants_coffee) → (¬ wants_tea → wants_coffee) ∧ (¬ wants_coffee → wants_tea)) → 
  (∀ (wants_tea wants_coffee : Prop), (wants_tea ∨ wants_coffee) → (¬ wants_tea → wants_coffee) ∧ (¬ wants_coffee → wants_tea)) :=
by
  sorry

end tea_or_coffee_indifference_l191_191425


namespace triangle_side_AC_l191_191603

theorem triangle_side_AC (B : Real) (BC AB : Real) (AC : Real) (hB : B = 30 * Real.pi / 180) (hBC : BC = 2) (hAB : AB = Real.sqrt 3) : AC = 1 :=
by
  sorry

end triangle_side_AC_l191_191603


namespace video_total_votes_l191_191824

theorem video_total_votes (x : ℕ) (L D : ℕ)
  (h1 : L + D = x)
  (h2 : L - D = 130)
  (h3 : 70 * x = 100 * L) :
  x = 325 :=
by
  sorry

end video_total_votes_l191_191824


namespace solve_system1_solve_system2_l191_191498

-- Define System (1) and prove its solution
theorem solve_system1 (x y : ℝ) (h1 : x = 5 - y) (h2 : x - 3 * y = 1) : x = 4 ∧ y = 1 := by
  sorry

-- Define System (2) and prove its solution
theorem solve_system2 (x y : ℝ) (h1 : x - 2 * y = 6) (h2 : 2 * x + 3 * y = -2) : x = 2 ∧ y = -2 := by
  sorry

end solve_system1_solve_system2_l191_191498


namespace total_cards_correct_l191_191735

-- Define the number of dozens each person has
def dozens_per_person : Nat := 9

-- Define the number of cards per dozen
def cards_per_dozen : Nat := 12

-- Define the number of people
def num_people : Nat := 4

-- Define the total number of Pokemon cards in all
def total_cards : Nat := dozens_per_person * cards_per_dozen * num_people

-- The statement to be proved
theorem total_cards_correct : total_cards = 432 := 
by 
  -- Proof omitted as requested
  sorry

end total_cards_correct_l191_191735


namespace sum_of_squares_l191_191761

theorem sum_of_squares (n : ℕ) (x : ℕ) (h1 : (x + 1)^3 - x^3 = n^2) (h2 : n > 0) : ∃ a b : ℕ, n = a^2 + b^2 :=
by
  sorry

end sum_of_squares_l191_191761


namespace exists_centrally_symmetric_inscribed_convex_hexagon_l191_191598

-- Definition of a convex polygon with vertices
def convex_polygon (W : Type) : Prop := sorry

-- Definition of the unit area condition
def has_unit_area (W : Type) : Prop := sorry

-- Definition of being centrally symmetric
def is_centrally_symmetric (V : Type) : Prop := sorry

-- Definition of being inscribed
def is_inscribed_polygon (V W : Type) : Prop := sorry

-- Definition of a convex hexagon
def convex_hexagon (V : Type) : Prop := sorry

-- Main theorem statement
theorem exists_centrally_symmetric_inscribed_convex_hexagon (W : Type) 
  (hW_convex : convex_polygon W) (hW_area : has_unit_area W) : 
  ∃ V : Type, convex_hexagon V ∧ is_centrally_symmetric V ∧ is_inscribed_polygon V W ∧ sorry :=
  sorry

end exists_centrally_symmetric_inscribed_convex_hexagon_l191_191598


namespace nancy_initial_files_correct_l191_191613

-- Definitions based on the problem conditions
def initial_files (deleted_files : ℕ) (folder_count : ℕ) (files_per_folder : ℕ) : ℕ :=
  (folder_count * files_per_folder) + deleted_files

-- The proof statement
theorem nancy_initial_files_correct :
  initial_files 31 7 7 = 80 :=
by
  sorry

end nancy_initial_files_correct_l191_191613


namespace number_of_persons_l191_191892

theorem number_of_persons (P : ℕ) : 
  (P * 12 * 5 = 30 * 13 * 6) → P = 39 :=
by
  sorry

end number_of_persons_l191_191892


namespace devin_teaching_years_l191_191293

theorem devin_teaching_years :
  let calculus_years := 4
  let algebra_years := 2 * calculus_years
  let statistics_years := 5 * algebra_years
  calculus_years + algebra_years + statistics_years = 52 :=
by
  let calculus_years := 4
  let algebra_years := 2 * calculus_years
  let statistics_years := 5 * algebra_years
  show calculus_years + algebra_years + statistics_years = 52
  sorry

end devin_teaching_years_l191_191293


namespace largest_fraction_l191_191020

theorem largest_fraction (p q r s : ℕ) (hp : 0 < p) (hpq : p < q) (hqr : q < r) (hrs : r < s) : 
  max (max (max (max (↑(p + q) / ↑(r + s)) (↑(p + s) / ↑(q + r))) 
              (↑(q + r) / ↑(p + s))) 
          (↑(q + s) / ↑(p + r))) 
      (↑(r + s) / ↑(p + q)) = (↑(r + s) / ↑(p + q)) :=
sorry

end largest_fraction_l191_191020


namespace total_balls_estimate_l191_191872

theorem total_balls_estimate 
  (total_balls : ℕ) 
  (red_balls : ℕ) 
  (frequency : ℚ)
  (h_red_balls : red_balls = 12)
  (h_frequency : frequency = 0.6) 
  (h_fraction : (red_balls : ℚ) / total_balls = frequency): 
  total_balls = 20 := 
by 
  sorry

end total_balls_estimate_l191_191872


namespace contribution_is_6_l191_191433

-- Defining the earnings of each friend
def earning_1 : ℕ := 18
def earning_2 : ℕ := 22
def earning_3 : ℕ := 30
def earning_4 : ℕ := 35
def earning_5 : ℕ := 45

-- Defining the modified contribution for the highest earner
def modified_earning_5 : ℕ := 40

-- Calculate the total adjusted earnings
def total_earnings : ℕ := earning_1 + earning_2 + earning_3 + earning_4 + modified_earning_5

-- Calculate the equal share each friend should receive
def equal_share : ℕ := total_earnings / 5

-- Calculate the contribution needed from the friend who earned $35 to match the equal share
def contribution_from_earning_4 : ℕ := earning_4 - equal_share

-- Stating the proof problem
theorem contribution_is_6 : contribution_from_earning_4 = 6 := by
  sorry

end contribution_is_6_l191_191433


namespace cookies_remaining_percentage_l191_191096

theorem cookies_remaining_percentage (c : ℕ) (p_n p_e : ℚ) (h_c : c = 600) (h_nicole : p_n = 2/5) (h_eduardo : p_e = 3/5) :
  ∃ p_r : ℚ, p_r = 24 :=
by
  have h_nicole_cookies : ℚ := p_n * c
  have h_remaining_after_nicole : ℚ := c - h_nicole_cookies
  have h_eduardo_cookies : ℚ := p_e * h_remaining_after_nicole
  have h_remaining_after_eduardo : ℚ := h_remaining_after_nicole - h_eduardo_cookies
  have h_percentage_remaining : ℚ := (h_remaining_after_eduardo / c) * 100
  use h_percentage_remaining
  field_simp [h_c, h_nicole, h_eduardo]
  norm_num
  sorry

end cookies_remaining_percentage_l191_191096


namespace inequality_proof_l191_191697

variable {x y z : ℝ}

theorem inequality_proof 
  (hx : x > 0) (hy : y > 0) (hz : z > 0) 
  (hxyz : x + y + z = 1) : 
  (1 / x^2 + x) * (1 / y^2 + y) * (1 / z^2 + z) ≥ (28 / 3)^3 :=
sorry

end inequality_proof_l191_191697


namespace part1_part2_l191_191581

noncomputable def A (x : ℝ) : Prop := x < 0 ∨ x > 2
noncomputable def B (a x : ℝ) : Prop := a ≤ x ∧ x ≤ 3 - 2 * a

-- Part (1)
theorem part1 (a : ℝ) : 
  (∀ x : ℝ, A x ∨ B a x) ↔ (a ≤ 0) := 
sorry

-- Part (2)
theorem part2 (a : ℝ) : 
  (∀ x : ℝ, B a x → (0 ≤ x ∧ x ≤ 2)) ↔ (1 / 2 ≤ a) :=
sorry

end part1_part2_l191_191581


namespace base4_sum_conversion_to_base10_l191_191007

theorem base4_sum_conversion_to_base10 :
  let n1 := 2213
  let n2 := 2703
  let n3 := 1531
  let base := 4
  let sum_base4 := n1 + n2 + n3 
  let sum_base10 :=
    (1 * base^4) + (0 * base^3) + (2 * base^2) + (5 * base^1) + (1 * base^0)
  sum_base10 = 309 :=
by
  sorry

end base4_sum_conversion_to_base10_l191_191007


namespace solve_eq_f_x_plus_3_l191_191187

-- Define the function f with its piecewise definition based on the conditions
noncomputable def f (x : ℝ) : ℝ :=
  if h : x ≥ 0 then x^2 - 3 * x
  else -(x^2 - 3 * (-x))

-- Define the main theorem to find the solution set
theorem solve_eq_f_x_plus_3 (x : ℝ) :
  f x = x + 3 ↔ x = 2 + Real.sqrt 7 ∨ x = -1 ∨ x = -3 :=
by sorry

end solve_eq_f_x_plus_3_l191_191187


namespace angle_sum_l191_191722

theorem angle_sum (x : ℝ) (h1 : 2 * x + x = 90) : x = 30 := 
sorry

end angle_sum_l191_191722


namespace xy_value_l191_191197

theorem xy_value (x y : ℝ) (h : x * (x + y) = x^2 + 12) : x * y = 12 := by
  sorry

end xy_value_l191_191197


namespace find_number_l191_191529

theorem find_number (x : ℤ) (h : 300 + 8 * x = 340) : x = 5 := by
  sorry

end find_number_l191_191529


namespace intersection_point_l191_191299

variable (x y z t : ℝ)

-- Conditions
def line_parametric : Prop := 
  (x = 1 + 2 * t) ∧ 
  (y = 2) ∧ 
  (z = 4 + t)

def plane_equation : Prop :=
  x - 2 * y + 4 * z - 19 = 0

-- Problem statement
theorem intersection_point (h_line: line_parametric x y z t) (h_plane: plane_equation x y z):
  x = 3 ∧ y = 2 ∧ z = 5 :=
by
  sorry

end intersection_point_l191_191299


namespace exists_zero_point_in_interval_l191_191287

noncomputable def f (x : ℝ) : ℝ := x^3 + Real.sin x - 2 * x

theorem exists_zero_point_in_interval :
  ∃ c ∈ Set.Ioo 1 (Real.pi / 2), f c = 0 := 
sorry

end exists_zero_point_in_interval_l191_191287


namespace transform_eq_l191_191784

theorem transform_eq (x y : ℝ) (h : 5 * x - 6 * y = 4) : 
  y = (5 / 6) * x - (2 / 3) :=
  sorry

end transform_eq_l191_191784


namespace find_angle_x_l191_191337

theorem find_angle_x (angle_ABC angle_BAC angle_BCA angle_DCE angle_CED x : ℝ)
  (h1 : angle_ABC + angle_BAC + angle_BCA = 180)
  (h2 : angle_ABC = 70) 
  (h3 : angle_BAC = 50)
  (h4 : angle_DCE + angle_CED = 90)
  (h5 : angle_DCE = angle_BCA) :
  x = 30 :=
by
  sorry

end find_angle_x_l191_191337


namespace find_a10_l191_191467

-- Define the arithmetic sequence
def arithmetic_sequence (a : ℕ → ℤ) : Prop :=
  ∀ n m : ℕ, a (n + 1) - a n = a (m + 1) - a m

-- Given conditions
variables (a : ℕ → ℤ) (h_arith_seq : arithmetic_sequence a)
variables (h_a2 : a 2 = 2) (h_a6 : a 6 = 10)

-- Goal to prove
theorem find_a10 : a 10 = 18 :=
by
  sorry

end find_a10_l191_191467


namespace expand_product_l191_191427

-- Define the problem
theorem expand_product (x : ℝ) : (x - 3) * (x + 4) = x^2 + x - 12 :=
by
  sorry

end expand_product_l191_191427


namespace polyhedron_has_triangular_face_l191_191325

-- Let's define the structure of a polyhedron, its vertices, edges, and faces.
structure Polyhedron :=
(vertices : ℕ)
(edges : ℕ)
(faces : ℕ)

-- Let's assume a function that indicates if a polyhedron is convex.
def is_convex (P : Polyhedron) : Prop := sorry  -- Convexity needs a rigorous formal definition.

-- Define a face of a polyhedron as an n-sided polygon.
structure Face :=
(sides : ℕ)

-- Predicate to check if a face is triangular.
def is_triangle (F : Face) : Prop := F.sides = 3

-- Predicate to check if each vertex has at least four edges meeting at it.
def each_vertex_has_at_least_four_edges (P : Polyhedron) : Prop := 
  sorry  -- This would need a more intricate definition involving the degrees of vertices.

-- We state the theorem using the defined concepts.
theorem polyhedron_has_triangular_face 
(P : Polyhedron) 
(h1 : is_convex P) 
(h2 : each_vertex_has_at_least_four_edges P) :
∃ (F : Face), is_triangle F :=
sorry

end polyhedron_has_triangular_face_l191_191325


namespace number_of_triangles_l191_191012

open Nat

/-- Each side of a square is divided into 8 equal parts, and using the divisions
as vertices (not including the vertices of the square), the number of different 
triangles that can be obtained is 3136. -/
theorem number_of_triangles (n : ℕ := 7) :
  (n * 4).choose 3 - 4 * n.choose 3 = 3136 := 
sorry

end number_of_triangles_l191_191012


namespace circle_passing_origin_l191_191407

theorem circle_passing_origin (a b r : ℝ) :
  ((a^2 + b^2 = r^2) ↔ (∃ (x y : ℝ), (x-a)^2 + (y-b)^2 = r^2 ∧ x = 0 ∧ y = 0)) :=
by
  sorry

end circle_passing_origin_l191_191407


namespace inequality_always_true_l191_191522

theorem inequality_always_true (a : ℝ) (x : ℝ) (h : -1 ≤ a ∧ a ≤ 1) :
  x^2 + (a - 4) * x + 4 - 2 * a > 0 → (x < 1 ∨ x > 3) :=
by {
  sorry
}

end inequality_always_true_l191_191522


namespace average_infection_rate_l191_191133

theorem average_infection_rate (x : ℝ) : 
  (1 + x + x * (1 + x) = 196) → x = 13 :=
by
  intro h
  sorry

end average_infection_rate_l191_191133


namespace quadratic_two_distinct_real_roots_l191_191188

theorem quadratic_two_distinct_real_roots (k : ℝ) (h1 : k ≠ 0) : 
  (∀ Δ > 0, Δ = (-2)^2 - 4 * k * (-1)) ↔ (k > -1) :=
by
  -- Since Δ = 4 + 4k, we need to show that (4 + 4k > 0) ↔ (k > -1)
  sorry

end quadratic_two_distinct_real_roots_l191_191188


namespace paint_gallons_needed_l191_191875

theorem paint_gallons_needed 
    (n_poles : ℕ := 12)
    (height : ℝ := 12)
    (diameter : ℝ := 8)
    (coverage_per_gallon : ℝ := 300)
    (whole_gallons : ℝ := 17) :
    let radius := diameter / 2
    let lateral_surface_area_per_pole := 2 * Real.pi * radius * height
    let top_and_bottom_face_area_per_pole := 2 * Real.pi * radius ^ 2
    let total_surface_area_per_pole := lateral_surface_area_per_pole + top_and_bottom_face_area_per_pole
    let total_paintable_area := n_poles * total_surface_area_per_pole
    let gallons_needed := (total_paintable_area / coverage_per_gallon).ceil in
    gallons_needed = whole_gallons :=
by
    sorry

end paint_gallons_needed_l191_191875


namespace elroy_miles_difference_l191_191159

theorem elroy_miles_difference (earning_rate_last_year earning_rate_this_year total_collection_last_year : ℝ)
  (rate_last_year : earning_rate_last_year = 4)
  (rate_this_year : earning_rate_this_year = 2.75)
  (total_collected : total_collection_last_year = 44) :
  (total_collection_last_year / earning_rate_this_year) - (total_collection_last_year / earning_rate_last_year) = 5 :=
by
  rw [rate_last_year, rate_this_year, total_collected]
  norm_num
  sorry

end elroy_miles_difference_l191_191159


namespace find_percentage_decrease_l191_191506

-- Define the measures of two complementary angles
def angles_complementary (a b : ℝ) : Prop := a + b = 90

-- Given variables
variable (small_angle large_angle : ℝ)

-- Given conditions
def ratio_of_angles (small_angle large_angle : ℝ) : Prop := small_angle / large_angle = 1 / 2

def increased_small_angle (small_angle : ℝ) : ℝ := small_angle * 1.2

noncomputable def new_large_angle (small_angle large_angle : ℝ) : ℝ :=
  90 - increased_small_angle small_angle

def percentage_decrease (original new : ℝ) : ℝ :=
  ((original - new) / original) * 100

-- The theorem we need to prove
theorem find_percentage_decrease
  (h1 : ratio_of_angles small_angle large_angle)
  (h2 : angles_complementary small_angle large_angle) :
  percentage_decrease large_angle (new_large_angle small_angle large_angle) = 10 :=
sorry

end find_percentage_decrease_l191_191506


namespace max_value_fraction_ratio_tangent_line_through_point_l191_191442

theorem max_value_fraction_ratio (x y : ℝ) (h : x^2 + y^2 - 2*x - 2 = 0) :
  ∃ (max_value : ℝ), max_value = 2 + sqrt 6 :=
sorry

theorem tangent_line_through_point (x y : ℝ) (h : x^2 + y^2 - 2*x - 2 = 0) :
  ∃ (m : ℝ), m ≠ 0 ∧ (x, y) = (0, sqrt 2) → ∀ (x' y' : ℝ), y' = m * x' + sqrt 2 → x' - sqrt 2 * y' + 2 = 0 :=
sorry

end max_value_fraction_ratio_tangent_line_through_point_l191_191442


namespace product_value_l191_191391

theorem product_value :
  (1 / 3) * 9 * (1 / 27) * 81 * (1 / 243) * 729 * (1 / 2187) * 6561 = 81 :=
  sorry

end product_value_l191_191391


namespace line_slope_point_l191_191405

theorem line_slope_point (m b : ℝ) (h_slope : m = -4) (h_point : ∃ x y : ℝ, (x, y) = (5, 2) ∧ y = m * x + b) : 
  m + b = 18 := by
  sorry

end line_slope_point_l191_191405


namespace work_completion_time_for_A_l191_191666

-- Define the conditions
def B_completion_time : ℕ := 30
def joint_work_days : ℕ := 4
def work_left_fraction : ℚ := 2 / 3

-- Define the required proof statement
theorem work_completion_time_for_A (x : ℚ) : 
  (4 * (1 / x + 1 / B_completion_time) = 1 / 3) → x = 20 := 
by
  sorry

end work_completion_time_for_A_l191_191666


namespace Wendy_did_not_recycle_2_bags_l191_191639

theorem Wendy_did_not_recycle_2_bags (points_per_bag : ℕ) (total_bags : ℕ) (points_earned : ℕ) (did_not_recycle : ℕ) : 
  points_per_bag = 5 → 
  total_bags = 11 → 
  points_earned = 45 → 
  5 * (11 - did_not_recycle) = 45 → 
  did_not_recycle = 2 :=
by
  intros h_points_per_bag h_total_bags h_points_earned h_equation
  sorry

end Wendy_did_not_recycle_2_bags_l191_191639


namespace hyperbola_eccentricity_proof_l191_191588

noncomputable def hyperbola_eccentricity (a : ℝ) (h_pos: a > 0) : ℝ :=
  let c := Real.sqrt (a^2 + 1) in
  c / a

theorem hyperbola_eccentricity_proof :
  let a := 1 / Real.tan (Real.pi / 6) in
  hyperbola_eccentricity a (by norm_num [Real.tan_pos_pi_div_two]) = 2 * Real.sqrt 3 / 3 :=
sorry

end hyperbola_eccentricity_proof_l191_191588


namespace geometric_sequence_a3_equals_4_l191_191724

variable {a : ℕ → ℝ}

def is_geometric_sequence (a : ℕ → ℝ) := ∃ r : ℝ, ∀ i, a (i+1) = a i * r

theorem geometric_sequence_a3_equals_4 
    (a_seq : is_geometric_sequence a) 
    (a_6_eq : a 6 = 6)
    (a_9_eq : a 9 = 9) : 
    a 3 = 4 := 
sorry

end geometric_sequence_a3_equals_4_l191_191724


namespace inequality_solution_set_l191_191685

noncomputable def solution_set := { x : ℝ | (x < -1 ∨ 1 < x) ∧ x ≠ 4 }

theorem inequality_solution_set : 
  { x : ℝ | (x^2 - 1) / (4 - x)^2 ≥ 0 } = solution_set :=
  by 
    sorry

end inequality_solution_set_l191_191685


namespace part_a_part_b_l191_191660

theorem part_a (x : ℝ) (n : ℕ) (hx_pos : 0 < x) (hx_ne_one : x ≠ 1) (hn_pos : 0 < n) :
  Real.log x < n * (x ^ (1 / n) - 1) ∧ n * (x ^ (1 / n) - 1) < (x ^ (1 / n)) * Real.log x := sorry

theorem part_b (x : ℝ) (hx_pos : 0 < x) (hx_ne_one : x ≠ 1) :
  (Real.log x) = (Real.log x) := sorry

end part_a_part_b_l191_191660


namespace remainder_when_98_mul_102_divided_by_11_l191_191795

theorem remainder_when_98_mul_102_divided_by_11 :
  (98 * 102) % 11 = 1 :=
by
  sorry

end remainder_when_98_mul_102_divided_by_11_l191_191795


namespace find_functions_l191_191160

noncomputable def satisfies_condition (f : ℝ → ℝ) : Prop :=
  ∀ (p q : ℝ), p ≠ q → (f q - f p) / (q - p) * 0 + f p - (f q - f p) / (q - p) * p = p * q

theorem find_functions (f : ℝ → ℝ) (c : ℝ) :
  satisfies_condition f → (∀ x : ℝ, f x = x * (c + x)) :=
by
  intros
  sorry

end find_functions_l191_191160


namespace inequalities_not_hold_range_a_l191_191317

theorem inequalities_not_hold_range_a (a : ℝ) :
  (¬ ∀ x : ℝ, x^2 - a * x + 1 ≤ 0) ∧ (¬ ∀ x : ℝ, a * x^2 + x - 1 > 0) ↔ (-2 < a ∧ a ≤ -1 / 4) :=
by
  sorry

end inequalities_not_hold_range_a_l191_191317


namespace average_of_three_quantities_l191_191897

theorem average_of_three_quantities (a b c d e : ℝ) 
  (h_avg_5 : (a + b + c + d + e) / 5 = 11)
  (h_avg_2 : (d + e) / 2 = 21.5) :
  (a + b + c) / 3 = 4 :=
by
  sorry

end average_of_three_quantities_l191_191897


namespace smallest_base_l191_191115

theorem smallest_base (b : ℕ) : (b^2 ≤ 80 ∧ 80 < b^3) → b = 5 := by
  sorry

end smallest_base_l191_191115


namespace floor_expression_bounds_l191_191430

theorem floor_expression_bounds (x : ℝ) (h : ⌊x * ⌊x / 2⌋⌋ = 12) : 
  4.9 ≤ x ∧ x < 5.1 :=
sorry

end floor_expression_bounds_l191_191430


namespace find_amount_l191_191399

-- Let A be the certain amount.
variable (A x : ℝ)

-- Given conditions
def condition1 (x : ℝ) := 0.65 * x = 0.20 * A
def condition2 (x : ℝ) := x = 150

-- Goal
theorem find_amount (A x : ℝ) (h1 : condition1 A x) (h2 : condition2 x) : A = 487.5 := 
by 
  sorry

end find_amount_l191_191399


namespace plan1_more_cost_effective_than_plan2_l191_191557

variable (x : ℝ)

def plan1_cost (x : ℝ) : ℝ :=
  36 + 0.1 * x

def plan2_cost (x : ℝ) : ℝ :=
  0.6 * x

theorem plan1_more_cost_effective_than_plan2 (h : x > 72) : 
  plan1_cost x < plan2_cost x :=
by
  sorry

end plan1_more_cost_effective_than_plan2_l191_191557


namespace greater_than_neg2_by_1_l191_191393

theorem greater_than_neg2_by_1 : -2 + 1 = -1 := by
  sorry

end greater_than_neg2_by_1_l191_191393


namespace isosceles_right_triangle_area_l191_191000

theorem isosceles_right_triangle_area (a b : ℝ) (h₁ : a = b) (h₂ : a + b = 20) : 
  (1 / 2) * a * b = 50 := 
by 
  sorry

end isosceles_right_triangle_area_l191_191000


namespace num_divisors_90_l191_191710

theorem num_divisors_90 : (∀ (n : ℕ), n = 90 → (factors n).divisors.card = 12) :=
by {
  intro n,
  intro hn,
  sorry
}

end num_divisors_90_l191_191710


namespace integer_ratio_value_l191_191220

theorem integer_ratio_value {x y : ℝ} (h1 : 3 < (x^2 - y^2) / (x^2 + y^2)) (h2 : (x^2 - y^2) / (x^2 + y^2) < 4) (h3 : ∃ t : ℤ, x = t * y) : ∃ t : ℤ, t = 2 :=
by
  sorry

end integer_ratio_value_l191_191220


namespace remainder_when_divided_by_17_l191_191962

theorem remainder_when_divided_by_17
  (N k : ℤ)
  (h : N = 357 * k + 36) :
  N % 17 = 2 :=
by
  sorry

end remainder_when_divided_by_17_l191_191962


namespace find_b_l191_191573

theorem find_b (b : ℚ) (m : ℚ) 
  (h1 : x^2 + b*x + 1/6 = (x + m)^2 + 1/18) 
  (h2 : b < 0) : 
  b = -2/3 := 
sorry

end find_b_l191_191573


namespace expected_value_transformation_l191_191318

variable (X : MassFunction ℚ)

-- Conditions
def valid_distribution (X : MassFunction ℚ) : Prop :=
  X 0 = 0.3 ∧ X 2 = 0.2 ∧ X 4 = 0.5

-- Question
theorem expected_value_transformation :
  valid_distribution X →
  E (5 • X + 4) = 16 :=
by
  sorry

end expected_value_transformation_l191_191318


namespace total_lifespan_l191_191906

theorem total_lifespan (B H F : ℕ)
  (hB : B = 10)
  (hH : H = B - 6)
  (hF : F = 4 * H) :
  B + H + F = 30 := by
  sorry

end total_lifespan_l191_191906


namespace xy_value_l191_191200

theorem xy_value (x y : ℝ) (h : x * (x + y) = x ^ 2 + 12) : x * y = 12 :=
by {
  sorry
}

end xy_value_l191_191200


namespace Yuna_boarding_place_l191_191838

-- Conditions
def Eunji_place : ℕ := 10
def people_after_Eunji : ℕ := 11

-- Proof Problem: Yuna's boarding place calculation
theorem Yuna_boarding_place :
  Eunji_place + people_after_Eunji + 1 = 22 :=
by
  sorry

end Yuna_boarding_place_l191_191838


namespace unique_n_for_solutions_l191_191066

theorem unique_n_for_solutions :
  ∃! (n : ℕ), (∀ (x y z : ℕ), 0 < x ∧ 0 < y ∧ 0 < z ∧ (3 * x + 3 * y + 2 * z = n)) → 
  ((∃ (s : ℕ), s = 10) ∧ (n = 17)) :=
sorry

end unique_n_for_solutions_l191_191066


namespace m_ge_1_l191_191853

open Set

theorem m_ge_1 (m : ℝ) :
  (∀ x, x ∈ {x | x ≤ 1} ∩ {x | ¬ (x ≤ m)} → False) → m ≥ 1 :=
by
  intro h
  sorry

end m_ge_1_l191_191853


namespace seashells_count_l191_191745

theorem seashells_count : 18 + 47 = 65 := by
  sorry

end seashells_count_l191_191745


namespace intersection_complement_l191_191707

open Set

variable (U A B : Set ℕ)
variable (hU : U = {1, 2, 3, 4, 5})
variable (hA : A = {1, 3, 4})
variable (hB : B = {4, 5})

theorem intersection_complement :
  A ∩ (U \ B) = {1, 3} :=
by
  rw [hU, hA, hB]
  ext
  simp
  sorry

end intersection_complement_l191_191707


namespace isabella_hourly_rate_l191_191604

def isabella_hours_per_day : ℕ := 5
def isabella_days_per_week : ℕ := 6
def isabella_weeks : ℕ := 7
def isabella_total_earnings : ℕ := 1050

theorem isabella_hourly_rate :
  (isabella_hours_per_day * isabella_days_per_week * isabella_weeks) * x = isabella_total_earnings → x = 5 := by
  sorry

end isabella_hourly_rate_l191_191604


namespace water_depth_when_upright_l191_191969
-- Import the entire Mathlib library

-- Define the conditions and question as a theorem
theorem water_depth_when_upright (height : ℝ) (diameter : ℝ) (horizontal_depth : ℝ) :
  height = 20 → diameter = 6 → horizontal_depth = 4 → water_depth = 5.3 :=
by
  intro h1 h2 h3
  -- The proof would go here, but we insert sorry to skip it
  sorry

end water_depth_when_upright_l191_191969


namespace seven_books_cost_l191_191098

-- Given condition: Three identical books cost $45
def three_books_cost (cost_per_book : ℤ) := 3 * cost_per_book = 45

-- Question to prove: The cost of seven identical books is $105
theorem seven_books_cost (cost_per_book : ℤ) (h : three_books_cost cost_per_book) : 7 * cost_per_book = 105 := 
sorry

end seven_books_cost_l191_191098


namespace toothpicks_at_150th_stage_l191_191901

theorem toothpicks_at_150th_stage (a₁ d n : ℕ) (h₁ : a₁ = 6) (hd : d = 5) (hn : n = 150) :
  (n * (2 * a₁ + (n - 1) * d)) / 2 = 56775 :=
by
  sorry -- Proof to be completed.

end toothpicks_at_150th_stage_l191_191901


namespace not_entire_field_weedy_l191_191814

-- Define the conditions
def field_divided_into_100_plots : Prop :=
  ∃ (a b : ℕ), a * b = 100

def initial_weedy_plots : Prop :=
  ∃ (weedy_plots : Finset (ℕ × ℕ)), weedy_plots.card = 9

def plot_becomes_weedy (weedy_plots : Finset (ℕ × ℕ)) (p : ℕ × ℕ) : Prop :=
  (p.fst ≠ 0 ∧ (p.fst - 1, p.snd) ∈ weedy_plots) ∧
  (p.snd ≠ 0 ∧ (p.fst, p.snd - 1) ∈ weedy_plots) ∨
  (p.fst ≠ 0 ∧ (p.fst - 1, p.snd) ∈ weedy_plots) ∧
  (p.snd ≠ 100 ∧ (p.fst, p.snd + 1) ∈ weedy_plots) ∨
  (p.fst ≠ 100 ∧ (p.fst + 1, p.snd) ∈ weedy_plots) ∧
  (p.snd ≠ 0 ∧ (p.fst, p.snd - 1) ∈ weedy_plots) ∨
  (p.fst ≠ 100 ∧ (p.fst + 1, p.snd) ∈ weedy_plots) ∧
  (p.snd ≠ 100 ∧ (p.fst, p.snd + 1) ∈ weedy_plots)

-- Theorem statement
theorem not_entire_field_weedy :
  field_divided_into_100_plots →
  initial_weedy_plots →
  (∀ weedy_plots : Finset (ℕ × ℕ), (∀ p : ℕ × ℕ, plot_becomes_weedy weedy_plots p → weedy_plots ∪ {p} = weedy_plots) → weedy_plots.card < 100) :=
  sorry

end not_entire_field_weedy_l191_191814


namespace side_c_possibilities_l191_191873

theorem side_c_possibilities (A : ℝ) (a b c : ℝ) (hA : A = 30) (ha : a = 4) (hb : b = 4 * Real.sqrt 3) :
  c = 4 ∨ c = 8 :=
sorry

end side_c_possibilities_l191_191873


namespace total_distance_traveled_l191_191550

-- Definitions of conditions
def bess_throw_distance : ℕ := 20
def bess_throws : ℕ := 4
def holly_throw_distance : ℕ := 8
def holly_throws : ℕ := 5
def bess_effective_throw_distance : ℕ := 2 * bess_throw_distance

-- Theorem statement
theorem total_distance_traveled :
  (bess_throws * bess_effective_throw_distance + holly_throws * holly_throw_distance) = 200 := 
  by sorry

end total_distance_traveled_l191_191550


namespace total_charge_for_3_6_miles_during_peak_hours_l191_191877

-- Define the initial conditions as constants
def initial_fee : ℝ := 2.05
def charge_per_half_mile_first_2_miles : ℝ := 0.45
def charge_per_two_fifth_mile_after_2_miles : ℝ := 0.35
def peak_hour_surcharge : ℝ := 1.50

-- Define the function to calculate the total charge
noncomputable def total_charge (total_distance : ℝ) (is_peak_hour : Bool) : ℝ :=
  let first_2_miles_charge := if total_distance > 2 then 4 * charge_per_half_mile_first_2_miles else (total_distance / 0.5) * charge_per_half_mile_first_2_miles
  let remaining_distance := if total_distance > 2 then total_distance - 2 else 0
  let after_2_miles_charge := if total_distance > 2 then (remaining_distance / (2 / 5)) * charge_per_two_fifth_mile_after_2_miles else 0
  let surcharge := if is_peak_hour then peak_hour_surcharge else 0
  initial_fee + first_2_miles_charge + after_2_miles_charge + surcharge

-- Prove that total charge of 3.6 miles during peak hours is 6.75
theorem total_charge_for_3_6_miles_during_peak_hours : total_charge 3.6 true = 6.75 := by
  sorry

end total_charge_for_3_6_miles_during_peak_hours_l191_191877


namespace fred_gave_cards_l191_191071

theorem fred_gave_cards (initial_cards : ℕ) (torn_cards : ℕ) 
  (bought_cards : ℕ) (total_cards : ℕ) (fred_cards : ℕ) : 
  initial_cards = 18 → torn_cards = 8 → bought_cards = 40 → total_cards = 84 →
  fred_cards = total_cards - (initial_cards - torn_cards + bought_cards) →
  fred_cards = 34 :=
by
  intros h_initial h_torn h_bought h_total h_fred
  sorry

end fred_gave_cards_l191_191071


namespace charging_time_is_correct_l191_191139

-- Lean definitions for the given conditions
def smartphone_charge_time : ℕ := 26
def tablet_charge_time : ℕ := 53
def phone_half_charge_time : ℕ := smartphone_charge_time / 2

-- Definition for the total charging time based on conditions
def total_charging_time : ℕ :=
  tablet_charge_time + phone_half_charge_time

-- Proof problem statement
theorem charging_time_is_correct : total_charging_time = 66 := by
  sorry

end charging_time_is_correct_l191_191139


namespace isabella_paint_area_l191_191726

-- Lean 4 statement for the proof problem based on given conditions and question:
theorem isabella_paint_area :
  let length := 15
  let width := 12
  let height := 9
  let door_and_window_area := 80
  let number_of_bedrooms := 4
  (2 * (length * height) + 2 * (width * height) - door_and_window_area) * number_of_bedrooms = 1624 :=
by
  sorry

end isabella_paint_area_l191_191726


namespace grover_total_profit_l191_191041

theorem grover_total_profit :
  let boxes := 3
  let masks_per_box := 20
  let price_per_mask := 0.50
  let cost := 15
  let total_masks := boxes * masks_per_box
  let total_revenue := total_masks * price_per_mask
  let total_profit := total_revenue - cost
  total_profit = 15 := by
sorry

end grover_total_profit_l191_191041


namespace todd_money_after_repay_l191_191923

-- Definitions as conditions
def borrowed_amount : ℤ := 100
def repay_amount : ℤ := 110
def ingredients_cost : ℤ := 75
def snow_cones_sold : ℤ := 200
def price_per_snow_cone : ℚ := 0.75

-- Function to calculate money left after transactions
def money_left_after_repay (borrowed : ℤ) (repay : ℤ) (cost : ℤ) (sold : ℤ) (price : ℚ) : ℚ :=
  let left_after_cost := borrowed - cost
  let earnings := sold * price
  let total_before_repay := left_after_cost + earnings
  total_before_repay - repay

-- The theorem stating the problem
theorem todd_money_after_repay :
  money_left_after_repay borrowed_amount repay_amount ingredients_cost snow_cones_sold price_per_snow_cone = 65 := 
by
  -- This is a placeholder for the actual proof
  sorry

end todd_money_after_repay_l191_191923


namespace cows_in_group_l191_191331

variable (c h : ℕ)

/--
In a group of cows and chickens, the number of legs was 20 more than twice the number of heads.
Cows have 4 legs each and chickens have 2 legs each.
Each animal has one head.
-/
theorem cows_in_group (h : ℕ) (hc : 4 * c + 2 * h = 2 * (c + h) + 20) : c = 10 :=
by
  sorry

end cows_in_group_l191_191331


namespace solve_abs_quadratic_eq_l191_191619

theorem solve_abs_quadratic_eq (x : ℝ) (h : |2 * x + 4| = 1 - 3 * x + x ^ 2) :
    x = (5 + Real.sqrt 37) / 2 ∨ x = (5 - Real.sqrt 37) / 2 := by
  sorry

end solve_abs_quadratic_eq_l191_191619


namespace find_d_l191_191217

-- Definitions for the functions f and g
def f (x : ℝ) (c : ℝ) : ℝ := 5 * x + c
def g (x : ℝ) (c : ℝ) : ℝ := c * x + 3

-- Statement to prove the value of d
theorem find_d (c d : ℝ) (h1 : ∀ x : ℝ, f (g x c) c = 15 * x + d) : d = 18 :=
by
  -- inserting custom logic for proof
  sorry

end find_d_l191_191217


namespace halfway_fraction_eq_l191_191765

-- Define the fractions
def one_seventh := 1 / 7
def one_fourth := 1 / 4

-- Define the common denominators
def common_denom_1 := 4 / 28
def common_denom_2 := 7 / 28

-- Define the addition of the common denominators
def addition := common_denom_1 + common_denom_2

-- Define the average of the fractions
noncomputable def average := addition / 2

-- State the theorem
theorem halfway_fraction_eq : average = 11 / 56 :=
by
  -- Provide the steps which will be skipped here
  sorry

end halfway_fraction_eq_l191_191765


namespace milk_production_days_l191_191861

variable {x : ℕ}

def daily_cow_production (x : ℕ) : ℚ := (x + 4) / ((x + 2) * (x + 3))

def total_daily_production (x : ℕ) : ℚ := (x + 5) * daily_cow_production x

def required_days (x : ℕ) : ℚ := (x + 9) / total_daily_production x

theorem milk_production_days : 
  required_days x = (x + 9) * (x + 2) * (x + 3) / ((x + 5) * (x + 4)) := 
by 
  sorry

end milk_production_days_l191_191861


namespace sugar_more_than_flour_l191_191223

def flour_needed : Nat := 9
def sugar_needed : Nat := 11
def flour_added : Nat := 4
def sugar_added : Nat := 0

def flour_remaining : Nat := flour_needed - flour_added
def sugar_remaining : Nat := sugar_needed - sugar_added

theorem sugar_more_than_flour : sugar_remaining - flour_remaining = 6 :=
by
  sorry

end sugar_more_than_flour_l191_191223


namespace area_change_l191_191447

variable (d x : ℝ)

-- Defining the area of the quadrilateral ACED as a function of x.
def area_ACED (d x : ℝ) : ℝ := (2 * d^2 + 4 * d * x - x^2) / (4 * Real.sqrt 3)

noncomputable def area_range (d x : ℝ) : Prop :=
  area_ACED d 0 = d^2 / (2 * Real.sqrt 3) ∧
  area_ACED d d = 5 * d^2 / (4 * Real.sqrt 3) ∧
  (∀ x, (0 ≤ x) ∧ (x ≤ d) → (d^2 / (2 * Real.sqrt 3) <= area_ACED d x) ∧ (area_ACED d x <= 5 * d^2 / (4 * Real.sqrt 3)))

theorem area_change (d : ℝ) : area_range d x :=
  sorry

end area_change_l191_191447


namespace simplify_expression_l191_191495

theorem simplify_expression : 
  (1 / (1 / (1 / 2)^0 + 1 / (1 / 2)^1 + 1 / (1 / 2)^2 + 1 / (1 / 2)^3)) = 1 / 15 :=
by 
  sorry

end simplify_expression_l191_191495


namespace total_seashells_l191_191746

-- Conditions
def sam_seashells : Nat := 18
def mary_seashells : Nat := 47

-- Theorem stating the question and the final answer
theorem total_seashells : sam_seashells + mary_seashells = 65 :=
by
  sorry

end total_seashells_l191_191746


namespace map_line_segments_l191_191011

def point : Type := ℝ × ℝ

def transformation (f : point → point) (p q : point) : Prop := f p = q

def counterclockwise_rotation_180 (p : point) : point := (-p.1, -p.2)

def clockwise_rotation_180 (p : point) : point := (-p.1, -p.2)

theorem map_line_segments :
  (transformation counterclockwise_rotation_180 (3, -2) (-3, 2) ∧
   transformation counterclockwise_rotation_180 (2, -5) (-2, 5)) ∨
  (transformation clockwise_rotation_180 (3, -2) (-3, 2) ∧
   transformation clockwise_rotation_180 (2, -5) (-2, 5)) :=
by
  sorry

end map_line_segments_l191_191011


namespace maximize_x3y4_correct_l191_191480

noncomputable def maximize_x3y4 : ℝ × ℝ :=
  let x := 160 / 7
  let y := 120 / 7
  (x, y)

theorem maximize_x3y4_correct :
  ∃ x y : ℝ, 0 < x ∧ 0 < y ∧ x + y = 40 ∧ (x, y) = maximize_x3y4 ∧ 
  ∀ (x' y' : ℝ), 0 < x' ∧ 0 < y' ∧ x' + y' = 40 → x ^ 3 * y ^ 4 ≥ x' ^ 3 * y' ^ 4 :=
by
  sorry

end maximize_x3y4_correct_l191_191480


namespace triangle_inequality_violation_l191_191109

theorem triangle_inequality_violation (a b c : ℝ) (ha : a = 1) (hb : b = 2) (hc : c = 7) : 
  ¬(a + b > c ∧ a + c > b ∧ b + c > a) := 
by
  rw [ha, hb, hc]
  simp
  sorry

end triangle_inequality_violation_l191_191109


namespace evaluate_fg_of_2_l191_191713

def f (x : ℝ) : ℝ := x ^ 3
def g (x : ℝ) : ℝ := 4 * x + 5

theorem evaluate_fg_of_2 : f (g 2) = 2197 :=
by
  sorry

end evaluate_fg_of_2_l191_191713


namespace central_angle_of_section_l191_191808

theorem central_angle_of_section (A : ℝ) (hA : 0 < A) (prob : ℝ) (hprob : prob = 1 / 4) :
  ∃ θ : ℝ, (θ / 360) = prob :=
by
  use 90
  sorry

end central_angle_of_section_l191_191808


namespace elastic_collision_ball_speed_l191_191537

open Real

noncomputable def final_ball_speed (v_car v_ball : ℝ) : ℝ :=
  let relative_speed := v_ball + v_car
  relative_speed + v_car

theorem elastic_collision_ball_speed :
  let v_car := 5
  let v_ball := 6
  final_ball_speed v_car v_ball = 16 := 
by
  sorry

end elastic_collision_ball_speed_l191_191537


namespace range_of_b_no_common_points_l191_191240

theorem range_of_b_no_common_points (b : ℝ) :
  ¬ (∃ x : ℝ, 2 ^ |x| - 1 = b) ↔ b < 0 :=
by
  sorry

end range_of_b_no_common_points_l191_191240


namespace abc_sum_l191_191080

theorem abc_sum
  (a b c : ℤ)
  (h1 : ∀ x : ℤ, x^2 + 17 * x + 70 = (x + a) * (x + b))
  (h2 : ∀ x : ℤ, x^2 - 19 * x + 84 = (x - b) * (x - c)) :
  a + b + c = 29 := by
  sorry

end abc_sum_l191_191080


namespace total_pastries_l191_191417

-- Defining the initial conditions
def Grace_pastries : ℕ := 30
def Calvin_pastries : ℕ := Grace_pastries - 5
def Phoebe_pastries : ℕ := Grace_pastries - 5
def Frank_pastries : ℕ := Calvin_pastries - 8

-- The theorem we want to prove
theorem total_pastries : 
  Calvin_pastries + Phoebe_pastries + Frank_pastries + Grace_pastries = 97 := by
  sorry

end total_pastries_l191_191417


namespace number_of_pencils_broken_l191_191678

theorem number_of_pencils_broken
  (initial_pencils : ℕ)
  (misplaced_pencils : ℕ)
  (found_pencils : ℕ)
  (bought_pencils : ℕ)
  (final_pencils : ℕ)
  (h_initial : initial_pencils = 20)
  (h_misplaced : misplaced_pencils = 7)
  (h_found : found_pencils = 4)
  (h_bought : bought_pencils = 2)
  (h_final : final_pencils = 16) :
  (initial_pencils - misplaced_pencils + found_pencils + bought_pencils - final_pencils) = 3 := 
by
  sorry

end number_of_pencils_broken_l191_191678


namespace bees_count_on_fifth_day_l191_191260

theorem bees_count_on_fifth_day
  (initial_count : ℕ) (h_initial : initial_count = 1)
  (growth_factor : ℕ) (h_growth : growth_factor = 3) :
  let bees_at_day (n : ℕ) : ℕ := initial_count * (growth_factor + 1) ^ n
  bees_at_day 5 = 1024 := 
by {
  sorry
}

end bees_count_on_fifth_day_l191_191260


namespace minimize_square_sum_l191_191059

theorem minimize_square_sum (x y z : ℝ) (h : x + 2 * y + 3 * z = 1) : 
  ∃ x y z, (x + 2 * y + 3 * z = 1) ∧ (x^2 + y^2 + z^2 ≥ 0) ∧ ((x^2 + y^2 + z^2) = 1 / 14) :=
sorry

end minimize_square_sum_l191_191059


namespace number_of_possible_values_of_S_l191_191734

open Finset

theorem number_of_possible_values_of_S :
  let A := {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
            21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 
            39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 
            57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 
            75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 
            93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 
            109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120 }
  in
  ∃ (S : ℕ), (∀ A ⊆ range 121, A.card = 80 → S = ∑ x in A, id x) ∧
             let S_min := ∑ i in range 1 (80+1), i in
             let S_max := ∑ i in range 41 (120+1), i in
             (max_value : ℕ) (min_value : ℕ), max_value = S_max ∧ min_value = S_min ∧
             max_value - min_value + 1 = 3201 :=
by
  sorry

end number_of_possible_values_of_S_l191_191734


namespace wendy_candy_in_each_box_l191_191385

variable (x : ℕ)

def brother_candy : ℕ := 6
def total_candy : ℕ := 12
def wendy_boxes : ℕ := 2 * x

theorem wendy_candy_in_each_box :
  2 * x + brother_candy = total_candy → x = 3 :=
by
  intro h
  sorry

end wendy_candy_in_each_box_l191_191385


namespace equilateral_triangle_ratio_is_correct_l191_191928

noncomputable def equilateral_triangle_area_perimeter_ratio (a : ℝ) (h_eq : a = 10) : ℝ :=
  let altitude := (Real.sqrt 3 / 2) * a
  let area := (1 / 2) * a * altitude
  let perimeter := 3 * a
  area / perimeter

theorem equilateral_triangle_ratio_is_correct :
  equilateral_triangle_area_perimeter_ratio 10 (by rfl) = 5 * Real.sqrt 3 / 6 :=
by
  sorry

end equilateral_triangle_ratio_is_correct_l191_191928


namespace solution_set_of_inequality_l191_191091

theorem solution_set_of_inequality (x : ℝ) : 
  x^2 + 4 * x - 5 > 0 ↔ (x < -5 ∨ x > 1) :=
sorry

end solution_set_of_inequality_l191_191091


namespace no_xy_term_implies_k_eq_4_l191_191050

theorem no_xy_term_implies_k_eq_4 (k : ℝ) :
  (∀ x y : ℝ, (x + 2 * y) * (2 * x - k * y - 1) = 2 * x^2 + (4 - k) * x * y - x - 2 * k * y^2 - 2 * y) →
  ((4 - k) = 0) →
  k = 4 := 
by
  intros h1 h2
  sorry

end no_xy_term_implies_k_eq_4_l191_191050


namespace ratio_of_toys_l191_191361

theorem ratio_of_toys (total_toys : ℕ) (num_friends : ℕ) (toys_D : ℕ) 
  (h1 : total_toys = 118) 
  (h2 : num_friends = 4) 
  (h3 : toys_D = total_toys / num_friends) : 
  (toys_D / total_toys : ℚ) = 1 / 4 :=
by
  sorry

end ratio_of_toys_l191_191361


namespace minimum_perimeter_is_12_l191_191171

noncomputable def minimum_perimeter_upper_base_frustum
  (a b : ℝ) (h : ℝ) (V : ℝ) : ℝ :=
if h = 3 ∧ V = 63 ∧ (a * b = 9) then
  2 * (a + b)
else
  0 -- this case will never be used

theorem minimum_perimeter_is_12 :
  ∃ a b : ℝ, a * b = 9 ∧ 2 * (a + b) = 12 :=
by
  existsi 3
  existsi 3
  sorry

end minimum_perimeter_is_12_l191_191171


namespace todd_has_40_left_after_paying_back_l191_191926

def todd_snowcone_problem : Prop :=
  let borrowed := 100
  let repay := 110
  let cost_ingredients := 75
  let snowcones_sold := 200
  let price_per_snowcone := 0.75
  let total_earnings := snowcones_sold * price_per_snowcone
  let remaining_money := total_earnings - repay
  remaining_money = 40

theorem todd_has_40_left_after_paying_back : todd_snowcone_problem :=
by
  -- Add proof here if needed
  sorry

end todd_has_40_left_after_paying_back_l191_191926


namespace value_of_f_at_4_l191_191044

noncomputable def f (x : ℝ) (c : ℝ) (d : ℝ) : ℝ :=
  c * x ^ 2 + d * x + 3

theorem value_of_f_at_4 :
  (∃ c d : ℝ, f 1 c d = 3 ∧ f 2 c d = 5) → f 4 1 (-1) = 15 :=
by
  sorry

end value_of_f_at_4_l191_191044


namespace carrot_cakes_in_february_l191_191895

theorem carrot_cakes_in_february :
  (∃ (cakes_in_oct : ℕ) (cakes_in_nov : ℕ) (cakes_in_dec : ℕ) (cakes_in_jan : ℕ) (monthly_increase : ℕ),
      cakes_in_oct = 19 ∧
      cakes_in_nov = 21 ∧
      cakes_in_dec = 23 ∧
      cakes_in_jan = 25 ∧
      monthly_increase = 2 ∧
      cakes_in_february = cakes_in_jan + monthly_increase) →
  cakes_in_february = 27 :=
  sorry

end carrot_cakes_in_february_l191_191895


namespace monotonic_decreasing_interval_l191_191085

noncomputable def y (x : ℝ) : ℝ := x * log x

theorem monotonic_decreasing_interval :
  ∀ x : ℝ, 0 < x ∧ x < real.exp (-1) → deriv y x < 0 :=
by
  intros x h
  have h1 : deriv y x = log x + 1 := sorry
  have h2 : log x + 1 < 0 := sorry
  rw [h1]
  exact h2

end monotonic_decreasing_interval_l191_191085


namespace train_length_l191_191410

theorem train_length (speed_kmh : ℝ) (time_s : ℝ) (conversion_factor : ℝ) (speed_ms : ℝ) (distance_m : ℝ) 
  (h1 : speed_kmh = 36) 
  (h2 : time_s = 28)
  (h3 : conversion_factor = 1000 / 3600) -- convert km/hr to m/s
  (h4 : speed_ms = speed_kmh * conversion_factor)
  (h5 : distance_m = speed_ms * time_s) :
  distance_m = 280 := 
by
  sorry

end train_length_l191_191410


namespace missing_digit_divisible_by_11_l191_191563

theorem missing_digit_divisible_by_11 (A : ℕ) (h : 1 ≤ A ∧ A ≤ 9) (div_11 : (100 + 10 * A + 2) % 11 = 0) : A = 3 :=
sorry

end missing_digit_divisible_by_11_l191_191563


namespace factor_expression_eq_l191_191428

theorem factor_expression_eq (x : ℤ) : 75 * x + 50 = 25 * (3 * x + 2) :=
by
  -- The actual proof is omitted
  sorry

end factor_expression_eq_l191_191428


namespace max_inscribed_triangle_area_l191_191518

theorem max_inscribed_triangle_area (a b : ℝ) (ha : 0 < a) (hb : 0 < b) : 
  ∃ A, A = (3 * Real.sqrt 3 / 4) * a * b := 
sorry

end max_inscribed_triangle_area_l191_191518


namespace rectangle_area_l191_191669

theorem rectangle_area (r : ℝ) (L W : ℝ) (h₀ : r = 7) (h₁ : 2 * r = W) (h₂ : L / W = 3) : 
  L * W = 588 :=
by sorry

end rectangle_area_l191_191669


namespace ellipse_equation_l191_191617

theorem ellipse_equation (a b : ℝ) (h1 : 0 < b) (h2 : b < a) 
  (h3 : ∃ (P : ℝ × ℝ), P = (0, -1) ∧ P.2^2 = b^2) 
  (h4 : ∃ (C2 : ℝ → ℝ → Prop), (∀ x y : ℝ, C2 x y ↔ x^2 + y^2 = 4) ∧ 2 * a = 4) :
  (∀ x y : ℝ, (x^2 / a^2) + (y^2 / b^2) = 1 ↔ (x^2 / 4) + y^2 = 1) :=
by
  sorry

end ellipse_equation_l191_191617


namespace walk_to_bus_stop_time_l191_191248

theorem walk_to_bus_stop_time 
  (S T : ℝ)   -- Usual speed and time
  (D : ℝ)        -- Distance to bus stop
  (T'_delay : ℝ := 9)   -- Additional delay in minutes
  (T_coffee : ℝ := 6)   -- Coffee shop time in minutes
  (reduced_speed_factor : ℝ := 4/5)  -- Reduced speed factor
  (h1 : D = S * T)
  (h2 : D = reduced_speed_factor * S * (T + T'_delay - T_coffee)) :
  T = 12 :=
by
  sorry

end walk_to_bus_stop_time_l191_191248


namespace average_mileage_is_correct_l191_191971

noncomputable def total_distance : ℝ := 150 + 200
noncomputable def sedan_efficiency : ℝ := 25
noncomputable def truck_efficiency : ℝ := 15
noncomputable def sedan_miles : ℝ := 150
noncomputable def truck_miles : ℝ := 200

noncomputable def total_gas_used : ℝ := (sedan_miles / sedan_efficiency) + (truck_miles / truck_efficiency)
noncomputable def average_gas_mileage : ℝ := total_distance / total_gas_used

theorem average_mileage_is_correct :
  average_gas_mileage = 18.1 := 
by
  sorry

end average_mileage_is_correct_l191_191971


namespace necessary_but_not_sufficient_l191_191170

theorem necessary_but_not_sufficient (a b : ℝ) :
  (a ≠ 0) → (ab ≠ 0) ↔ (a ≠ 0) :=
by sorry

end necessary_but_not_sufficient_l191_191170


namespace preimage_of_3_2_eq_l191_191585

noncomputable def f (p : ℝ × ℝ) : ℝ × ℝ :=
  (p.1 * p.2, p.1 + p.2)

theorem preimage_of_3_2_eq (x y : ℝ) :
  f (x, y) = (-3, 2) ↔ (x = 3 ∧ y = -1) ∨ (x = -1 ∧ y = 3) :=
by
  sorry

end preimage_of_3_2_eq_l191_191585


namespace painter_remaining_time_l191_191408

-- Define the initial conditions
def total_rooms : ℕ := 11
def hours_per_room : ℕ := 7
def painted_rooms : ℕ := 2

-- Define the remaining rooms to paint
def remaining_rooms : ℕ := total_rooms - painted_rooms

-- Define the proof problem: the remaining time to paint the rest of the rooms
def remaining_hours : ℕ := remaining_rooms * hours_per_room

theorem painter_remaining_time :
  remaining_hours = 63 :=
sorry

end painter_remaining_time_l191_191408


namespace ratio_of_area_to_perimeter_l191_191931

noncomputable def side_length := 10
noncomputable def altitude := (side_length * (Real.sqrt 3 / 2))
noncomputable def area := (1 / 2) * side_length * altitude
noncomputable def perimeter := 3 * side_length

theorem ratio_of_area_to_perimeter (s : ℝ) (h : ℝ) (A : ℝ) (P : ℝ) 
  (h1 : s = 10) 
  (h2 : h = s * (Real.sqrt 3 / 2)) 
  (h3 : A = (1 / 2) * s * h) 
  (h4 : P = 3 * s) :
  A / P = 5 * Real.sqrt 3 / 6 := by
  sorry

end ratio_of_area_to_perimeter_l191_191931


namespace isosceles_triangle_circum_incenter_distance_l191_191285

variable {R r d : ℝ}

/-- The distance \(d\) between the centers of the circumscribed circle and the inscribed circle of an isosceles triangle satisfies \(d = \sqrt{R(R - 2r)}\) --/
theorem isosceles_triangle_circum_incenter_distance (hR : 0 < R) (hr : 0 < r) 
  (hIso : ∃ (A B C : ℝ × ℝ), (A ≠ B) ∧ (A ≠ C) ∧ (B ≠ C) ∧ (dist A B = dist A C)) 
  : d = Real.sqrt (R * (R - 2 * r)) :=
sorry

end isosceles_triangle_circum_incenter_distance_l191_191285


namespace calculate_expr_l191_191150

theorem calculate_expr : 1 - Real.sqrt 9 = -2 := by
  sorry

end calculate_expr_l191_191150


namespace binomial_12_3_eq_220_l191_191992

-- Definition of binomial coefficient
def binomial (n k : ℕ) : ℕ :=
  if k ≤ n then Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k)) else 0

-- Theorem to prove binomial(12, 3) = 220
theorem binomial_12_3_eq_220 : binomial 12 3 = 220 := by
  sorry

end binomial_12_3_eq_220_l191_191992


namespace simplify_expression_l191_191618

theorem simplify_expression (x : ℝ) (h : x ≠ 1) :
  (x^2 + x) / (x^2 - 2*x + 1) / ((x + 1) / (x - 1)) = x / (x - 1) :=
by
  sorry

end simplify_expression_l191_191618


namespace initial_position_of_M_l191_191353

theorem initial_position_of_M :
  ∃ x : ℤ, (x + 7) - 4 = 0 ∧ x = -3 :=
by sorry

end initial_position_of_M_l191_191353


namespace explicit_formula_of_odd_function_monotonicity_in_interval_l191_191182

-- Using Noncomputable because divisions are involved.
noncomputable def f (x : ℝ) (p q : ℝ) : ℝ := (p * x^2 + 2) / (q - 3 * x)

theorem explicit_formula_of_odd_function (p q : ℝ) 
  (h_odd : ∀ x : ℝ, f x p q = - f (-x) p q) 
  (h_value : f 2 p q = -5/3) : 
  f x 2 0 = -2/3 * (x + 1/x) :=
by sorry

theorem monotonicity_in_interval {x : ℝ} (h_domain : 0 < x ∧ x < 1) : 
  ∀ x1 x2 : ℝ, 0 < x1 ∧ x1 < x2 ∧ x2 < 1 -> f x1 2 0 < f x2 2 0 :=
by sorry

end explicit_formula_of_odd_function_monotonicity_in_interval_l191_191182


namespace polygon_sides_l191_191968

theorem polygon_sides (n : ℕ) (h1 : ∀ n, 180 * (n - 2) = 140 * n) : n = 9 :=
sorry

end polygon_sides_l191_191968


namespace obtain_angle_10_30_l191_191911

theorem obtain_angle_10_30 (a : ℕ) (h : 100 + a = 135) : a = 35 := 
by sorry

end obtain_angle_10_30_l191_191911


namespace min_value_proof_l191_191168

noncomputable def min_value (x y : ℝ) : ℝ := 1 / x + 1 / (2 * y)

theorem min_value_proof (x y : ℝ) (hx : x > 0) (hy : y > 0) (hxy : x + 2 * y = 1) :
  min_value x y = 4 :=
sorry

end min_value_proof_l191_191168


namespace painted_cubes_even_faces_l191_191499

theorem painted_cubes_even_faces :
  let L := 6 -- length of the block
  let W := 2 -- width of the block
  let H := 2 -- height of the block
  let total_cubes := 24 -- the block is cut into 24 1-inch cubes
  let cubes_even_faces := 12 -- the number of 1-inch cubes with even number of blue faces
  -- each cube has a total of 6 faces,
  -- we need to count how many cubes have an even number of painted faces.
  L * W * H = total_cubes → 
  cubes_even_faces = 12 := sorry

end painted_cubes_even_faces_l191_191499


namespace no_all_blue_possible_l191_191615

-- Define initial counts of chameleons
def initial_red : ℕ := 25
def initial_green : ℕ := 12
def initial_blue : ℕ := 8

-- Define the invariant condition
def invariant (r g : ℕ) : Prop := (r - g) % 3 = 1

-- Define the main theorem statement
theorem no_all_blue_possible : ¬∃ r g, r = 0 ∧ g = 0 ∧ invariant r g :=
by {
  sorry
}

end no_all_blue_possible_l191_191615


namespace find_D_c_l191_191954

-- Define the given conditions
def daily_wage_ratio (W_a W_b W_c : ℝ) : Prop :=
  W_a / W_b = 3 / 4 ∧ W_a / W_c = 3 / 5 ∧ W_b / W_c = 4 / 5

def total_earnings (W_a W_b W_c : ℝ) (D_a D_b D_c : ℕ) : ℝ :=
  W_a * D_a + W_b * D_b + W_c * D_c

variables {W_a W_b W_c : ℝ} 
variables {D_a D_b D_c : ℕ} 

-- Given values according to the problem
def W_c_value : ℝ := 110
def D_a_value : ℕ := 6
def D_b_value : ℕ := 9
def total_earnings_value : ℝ := 1628

-- The target proof statement
theorem find_D_c 
  (h_ratio : daily_wage_ratio W_a W_b W_c)
  (h_Wc : W_c = W_c_value)
  (h_earnings : total_earnings W_a W_b W_c D_a_value D_b_value D_c = total_earnings_value) 
  : D_c = 4 := 
sorry

end find_D_c_l191_191954


namespace toms_total_money_l191_191381

def quarter_value : ℕ := 25 -- cents
def dime_value : ℕ := 10 -- cents
def nickel_value : ℕ := 5 -- cents
def penny_value : ℕ := 1 -- cent

def quarters : ℕ := 10
def dimes : ℕ := 3
def nickels : ℕ := 4
def pennies : ℕ := 200

def total_in_cents : ℕ := (quarters * quarter_value) + (dimes * dime_value) + (nickels * nickel_value) + (pennies * penny_value)

def total_in_dollars : ℝ := total_in_cents / 100

theorem toms_total_money : total_in_dollars = 5 := by
  sorry

end toms_total_money_l191_191381


namespace knight_count_l191_191350

theorem knight_count (K L : ℕ) (h1 : K + L = 15) 
  (h2 : ∀ k, k < K → (∃ l, l < L ∧ l = 6)) 
  (h3 : ∀ l, l < L → (K > 7)) : K = 9 :=
by 
  sorry

end knight_count_l191_191350


namespace test_two_categorical_features_l191_191245

-- Definitions based on the problem conditions
def is_testing_method (method : String) : Prop :=
  method = "Three-dimensional bar chart" ∨
  method = "Two-dimensional bar chart" ∨
  method = "Contour bar chart" ∨
  method = "Independence test"

noncomputable def correct_method : String :=
  "Independence test"

-- Theorem statement based on the problem and solution
theorem test_two_categorical_features :
  ∀ m : String, is_testing_method m → m = correct_method :=
by
  sorry

end test_two_categorical_features_l191_191245


namespace integer_satisfying_conditions_l191_191788

theorem integer_satisfying_conditions :
  {a : ℤ | 1 ≤ a ∧ a ≤ 105 ∧ 35 ∣ (a^3 - 1)} = {1, 11, 16, 36, 46, 51, 71, 81, 86} :=
by
  sorry

end integer_satisfying_conditions_l191_191788


namespace total_charging_time_l191_191137

def charge_smartphone_full : ℕ := 26
def charge_tablet_full : ℕ := 53
def charge_phone_half : ℕ := charge_smartphone_full / 2
def charge_tablet : ℕ := charge_tablet_full

theorem total_charging_time : 
  charge_phone_half + charge_tablet = 66 := by
  sorry

end total_charging_time_l191_191137


namespace youngest_person_age_l191_191378

noncomputable def avg_age_seven_people := 30
noncomputable def avg_age_six_people_when_youngest_born := 25
noncomputable def num_people := 7
noncomputable def num_people_minus_one := 6

theorem youngest_person_age :
  let total_age_seven_people := num_people * avg_age_seven_people
  let total_age_six_people := num_people_minus_one * avg_age_six_people_when_youngest_born
  total_age_seven_people - total_age_six_people = 60 :=
by
  let total_age_seven_people := num_people * avg_age_seven_people
  let total_age_six_people := num_people_minus_one * avg_age_six_people_when_youngest_born
  sorry

end youngest_person_age_l191_191378


namespace probability_at_least_6_heads_in_9_flips_l191_191266

theorem probability_at_least_6_heads_in_9_flips : 
  let total_outcomes := 2 ^ 9 in
  let successful_outcomes := Nat.choose 9 6 + Nat.choose 9 7 + Nat.choose 9 8 + Nat.choose 9 9 in
  successful_outcomes.toRational / total_outcomes.toRational = (130 : ℚ) / 512 :=
by
  sorry

end probability_at_least_6_heads_in_9_flips_l191_191266


namespace sabrina_total_leaves_l191_191357

-- Definitions based on conditions
def basil_leaves := 12
def twice_the_sage_leaves (sages : ℕ) := 2 * sages = basil_leaves
def five_fewer_than_verbena (sages verbenas : ℕ) := sages + 5 = verbenas

-- Statement to prove
theorem sabrina_total_leaves (sages verbenas : ℕ) 
    (h1 : twice_the_sage_leaves sages) 
    (h2 : five_fewer_than_verbena sages verbenas) :
    basil_leaves + sages + verbenas = 29 :=
sorry

end sabrina_total_leaves_l191_191357


namespace calculate_1307_squared_l191_191955

theorem calculate_1307_squared : 1307 * 1307 = 1709849 := sorry

end calculate_1307_squared_l191_191955


namespace neg_two_squared_result_l191_191913

theorem neg_two_squared_result : -2^2 = -4 :=
by
  sorry

end neg_two_squared_result_l191_191913


namespace JoggerDifference_l191_191545

theorem JoggerDifference (tyson_joggers alexander_joggers christopher_joggers : ℕ)
  (h1 : christopher_joggers = 20 * tyson_joggers)
  (h2 : christopher_joggers = 80)
  (h3 : alexander_joggers = tyson_joggers + 22) :
  christopher_joggers - alexander_joggers = 54 := by
  sorry

end JoggerDifference_l191_191545


namespace average_speed_of_train_l191_191821

def ChicagoTime (t : String) : Prop := t = "5:00 PM"
def NewYorkTime (t : String) : Prop := t = "10:00 AM"
def TimeDifference (d : Nat) : Prop := d = 1
def Distance (d : Nat) : Prop := d = 480

theorem average_speed_of_train :
  ∀ (d t1 t2 diff : Nat), 
  Distance d → (NewYorkTime "10:00 AM") → (ChicagoTime "5:00 PM") → TimeDifference diff →
  (t2 = 5 ∧ t1 = (10 - diff)) →
  (d / (t2 - t1) = 60) :=
by
  intros d t1 t2 diff hD ht1 ht2 hDiff hTimes
  sorry

end average_speed_of_train_l191_191821


namespace number_eq_180_l191_191665

theorem number_eq_180 (x : ℝ) (h : 64 + 5 * 12 / (x / 3) = 65) : x = 180 :=
sorry

end number_eq_180_l191_191665


namespace twenty_four_multiples_of_4_l191_191093

theorem twenty_four_multiples_of_4 {n : ℕ} : (n = 104) ↔ (∃ k : ℕ, k = 24 ∧ ∀ m : ℕ, (12 ≤ m ∧ m ≤ n) → ∃ t : ℕ, m = 12 + 4 * t ∧ 1 ≤ t ∧ t ≤ 24) := 
by
  sorry

end twenty_four_multiples_of_4_l191_191093


namespace common_points_l191_191910

variable {R : Type*} [LinearOrderedField R]

def eq1 (x y : R) : Prop := x - y + 2 = 0
def eq2 (x y : R) : Prop := 3 * x + y - 4 = 0
def eq3 (x y : R) : Prop := x + y - 2 = 0
def eq4 (x y : R) : Prop := 2 * x - 5 * y + 7 = 0

theorem common_points : ∃ s : Finset (R × R), 
  (∀ p ∈ s, eq1 p.1 p.2 ∨ eq2 p.1 p.2) ∧ (∀ p ∈ s, eq3 p.1 p.2 ∨ eq4 p.1 p.2) ∧ s.card = 6 :=
by
  sorry

end common_points_l191_191910


namespace jenny_hours_left_l191_191473

theorem jenny_hours_left
  (hours_research : ℕ)
  (hours_proposal : ℕ)
  (hours_total : ℕ)
  (h1 : hours_research = 10)
  (h2 : hours_proposal = 2)
  (h3 : hours_total = 20) :
  (hours_total - (hours_research + hours_proposal) = 8) :=
by
  sorry

end jenny_hours_left_l191_191473


namespace max_AMC_expression_l191_191481

theorem max_AMC_expression (A M C : ℕ) (h : A + M + C = 15) : A * M * C + A * M + M * C + C * A ≤ 200 :=
by
  sorry

end max_AMC_expression_l191_191481


namespace function_is_monotonically_increasing_l191_191627

theorem function_is_monotonically_increasing (a : ℝ) :
  (∀ x : ℝ, (x^2 + 2*x + a) ≥ 0) ↔ (1 ≤ a) := 
sorry

end function_is_monotonically_increasing_l191_191627


namespace solve_quadratic_1_solve_quadratic_2_l191_191364

-- Define the first problem
theorem solve_quadratic_1 (x : ℝ) : 3 * x^2 - 4 * x = 2 * x → x = 0 ∨ x = 2 := by
  -- Proof step will go here
  sorry

-- Define the second problem
theorem solve_quadratic_2 (x : ℝ) : x * (x + 8) = 16 → x = -4 + 4 * Real.sqrt 2 ∨ x = -4 - 4 * Real.sqrt 2 := by
  -- Proof step will go here
  sorry

end solve_quadratic_1_solve_quadratic_2_l191_191364


namespace find_value_of_N_l191_191302

theorem find_value_of_N (N : ℝ) : 
  2 * ((3.6 * N * 2.50) / (0.12 * 0.09 * 0.5)) = 1600.0000000000002 → 
  N = 0.4800000000000001 :=
by
  sorry

end find_value_of_N_l191_191302


namespace Feuerbach_theorem_l191_191885

theorem Feuerbach_theorem
  (α β γ : ℝ) (x y z : ℝ)
  (htangent : 
    ∀ (x y z : ℝ), 
      x * cos (α/2) / sin ((β - γ) / 2) + 
      y * cos (β/2) / sin ((α - γ) / 2) + 
      z * cos (γ/2) / sin ((α - β) / 2) = 0) :
  ( ∃ (ξ η ζ : ℝ),
    (ξ : η : ζ = 
      ( sin^2 ((β - γ) / 2) :
        sin^2 ((α - γ) / 2) :
        sin^2 ((α - β) / 2) )) ) :=
begin
  sorry
end

end Feuerbach_theorem_l191_191885


namespace circle_equation_focus_parabola_origin_l191_191999

noncomputable def parabola_focus (p : ℝ) (x y : ℝ) : Prop :=
  y^2 = 4 * p * x

def passes_through_origin (x y : ℝ) : Prop :=
  (0 - x)^2 + (0 - y)^2 = x^2 + y^2

theorem circle_equation_focus_parabola_origin :
  (∃ x y : ℝ, parabola_focus 1 x y ∧ passes_through_origin x y)
    → ∃ k : ℝ, (x^2 - 2 * x + y^2 = k) :=
sorry

end circle_equation_focus_parabola_origin_l191_191999


namespace simplify_root_product_l191_191359

theorem simplify_root_product : 
  (625:ℝ)^(1/4) * (125:ℝ)^(1/3) = 25 :=
by
  have h₁ : (625:ℝ) = 5^4 := by norm_num
  have h₂ : (125:ℝ) = 5^3 := by norm_num
  rw [h₁, h₂]
  norm_num
  sorry

end simplify_root_product_l191_191359


namespace two_point_line_l191_191395

theorem two_point_line (k b : ℝ) (h_k : k ≠ 0) :
  (∀ (x y : ℝ), (y = k * x + b → (x, y) = (0, 0) ∨ (x, y) = (1, 1))) →
  (∀ (x y : ℝ), (y = k * x + b → (x, y) ≠ (2, 0))) :=
by
  sorry

end two_point_line_l191_191395


namespace minimum_bailing_rate_l191_191394

-- Conditions as formal definitions.
def distance_from_shore : ℝ := 3
def intake_rate : ℝ := 20 -- gallons per minute
def sinking_threshold : ℝ := 120 -- gallons
def speed_first_half : ℝ := 6 -- miles per hour
def speed_second_half : ℝ := 3 -- miles per hour

-- Formal translation of the problem using definitions.
theorem minimum_bailing_rate : (distance_from_shore = 3) →
                             (intake_rate = 20) →
                             (sinking_threshold = 120) →
                             (speed_first_half = 6) →
                             (speed_second_half = 3) →
                             (∃ r : ℝ, 18 ≤ r) :=
by
  sorry

end minimum_bailing_rate_l191_191394


namespace find_coeff_sum_l191_191236

def parabola_eq (a b c : ℚ) (y : ℚ) : ℚ := a*y^2 + b*y + c

theorem find_coeff_sum 
  (a b c : ℚ)
  (h_eq : ∀ y, parabola_eq a b c y = - ((y + 6)^2) / 3 + 7)
  (h_pass : parabola_eq a b c 0 = 5) :
  a + b + c = -32 / 3 :=
by
  sorry

end find_coeff_sum_l191_191236


namespace intersection_of_A_and_B_l191_191590

theorem intersection_of_A_and_B :
  let A := {0, 1, 2, 3, 4}
  let B := {x | ∃ n ∈ A, x = 2 * n}
  A ∩ B = {0, 2, 4} :=
by
  sorry

end intersection_of_A_and_B_l191_191590


namespace multiples_of_7_not_14_l191_191320

theorem multiples_of_7_not_14 (n : ℕ) : 
  ∃ count : ℕ, count = 25 ∧ 
    (∀ x, x < 350 → x % 7 = 0 → x % 14 ≠ 0 ↔ ∃ k, x = 7 * k ∧ k % 2 = 1) := 
by
  have count := (finset.range' 7 350).countp (λ x, x % 7 = 0 ∧ x % 14 ≠ 0)
  have h_count : count = 25 := sorry 
  exact ⟨25, h_count, sorry⟩

end multiples_of_7_not_14_l191_191320


namespace jack_total_cost_l191_191806

def plan_base_cost : ℕ := 25

def cost_per_text : ℕ := 8

def free_hours : ℕ := 25

def cost_per_extra_minute : ℕ := 10

def texts_sent : ℕ := 150

def hours_talked : ℕ := 26

def total_cost (base_cost : ℕ) (texts_sent : ℕ) (cost_per_text : ℕ) (hours_talked : ℕ) 
               (free_hours : ℕ) (cost_per_extra_minute : ℕ) : ℕ :=
  base_cost + (texts_sent * cost_per_text) / 100 + 
  ((hours_talked - free_hours) * 60 * cost_per_extra_minute) / 100

theorem jack_total_cost : 
  total_cost plan_base_cost texts_sent cost_per_text hours_talked free_hours cost_per_extra_minute = 43 :=
by
  sorry

end jack_total_cost_l191_191806


namespace larger_number_l191_191956

theorem larger_number (HCF A B : ℕ) (factor1 factor2 : ℕ) (h_HCF : HCF = 23) (h_factor1 : factor1 = 14) (h_factor2 : factor2 = 15) (h_LCM : HCF * factor1 * factor2 = A * B) (h_A : A = HCF * factor2) (h_B : B = HCF * factor1) : A = 345 :=
by
  sorry

end larger_number_l191_191956


namespace employee_pay_l191_191663

variable (X Y : ℝ)

theorem employee_pay (h1: X + Y = 572) (h2: X = 1.2 * Y) : Y = 260 :=
by
  sorry

end employee_pay_l191_191663


namespace sum_of_a_for_quadratic_has_one_solution_l191_191686

noncomputable def discriminant (a : ℝ) : ℝ := (a + 12)^2 - 4 * 3 * 16

theorem sum_of_a_for_quadratic_has_one_solution : 
  (∀ a : ℝ, discriminant a = 0) → 
  (-12 + 8 * Real.sqrt 3) + (-12 - 8 * Real.sqrt 3) = -24 :=
by
  intros h
  simp [discriminant] at h
  sorry

end sum_of_a_for_quadratic_has_one_solution_l191_191686


namespace stephen_hawking_philosophical_implications_l191_191503

/-- Stephen Hawking's statements -/
def stephen_hawking_statement_1 := "The universe was not created by God"
def stephen_hawking_statement_2 := "Modern science can explain the origin of the universe"

/-- Definitions implied by Hawking's statements -/
def unity_of_world_lies_in_materiality := "The unity of the world lies in its materiality"
def thought_and_existence_identical := "Thought and existence are identical"

/-- Combined implication of Stephen Hawking's statements -/
def correct_philosophical_implications := [unity_of_world_lies_in_materiality, thought_and_existence_identical]

/-- Theorem: The correct philosophical implications of Stephen Hawking's statements are ① and ②. -/
theorem stephen_hawking_philosophical_implications :
  (stephen_hawking_statement_1 = "The universe was not created by God") →
  (stephen_hawking_statement_2 = "Modern science can explain the origin of the universe") →
  correct_philosophical_implications = ["The unity of the world lies in its materiality", "Thought and existence are identical"] :=
by
  sorry

end stephen_hawking_philosophical_implications_l191_191503


namespace blocks_differs_in_exactly_two_ways_correct_l191_191531

structure Block where
  material : Bool       -- material: false for plastic, true for wood
  size : Fin 3          -- sizes: 0 for small, 1 for medium, 2 for large
  color : Fin 4         -- colors: 0 for blue, 1 for green, 2 for red, 3 for yellow
  shape : Fin 4         -- shapes: 0 for circle, 1 for hexagon, 2 for square, 3 for triangle
  finish : Bool         -- finish: false for glossy, true for matte

def originalBlock : Block :=
  { material := false, size := 1, color := 2, shape := 0, finish := false }

def differsInExactlyTwoWays (b1 b2 : Block) : Bool :=
  (if b1.material ≠ b2.material then 1 else 0) +
  (if b1.size ≠ b2.size then 1 else 0) +
  (if b1.color ≠ b2.color then 1 else 0) +
  (if b1.shape ≠ b2.shape then 1 else 0) +
  (if b1.finish ≠ b2.finish then 1 else 0) == 2

def countBlocksDifferingInTwoWays : Nat :=
  let allBlocks := List.product
                  (List.product
                    (List.product
                      (List.product
                        [false, true]
                        ([0, 1, 2] : List (Fin 3)))
                      ([0, 1, 2, 3] : List (Fin 4)))
                    ([0, 1, 2, 3] : List (Fin 4)))
                  [false, true]
  (allBlocks.filter
    (λ b => differsInExactlyTwoWays originalBlock
                { material := b.1.1.1.1, size := b.1.1.1.2, color := b.1.1.2, shape := b.1.2, finish := b.2 })).length

theorem blocks_differs_in_exactly_two_ways_correct :
  countBlocksDifferingInTwoWays = 51 :=
  by
    sorry

end blocks_differs_in_exactly_two_ways_correct_l191_191531


namespace correct_division_result_l191_191681

theorem correct_division_result {x : ℕ} (h : 3 * x = 90) : x / 3 = 10 :=
by
  -- placeholder for the actual proof
  sorry

end correct_division_result_l191_191681


namespace intersection_complement_l191_191893

open Set

variable (U P Q : Set ℕ)
variable (H_U : U = {1, 2, 3, 4, 5, 6})
variable (H_P : P = {1, 2, 3, 4})
variable (H_Q : Q = {3, 4, 5})

theorem intersection_complement (hU : U = {1, 2, 3, 4, 5, 6}) (hP : P = {1, 2, 3, 4}) (hQ : Q = {3, 4, 5}) :
  P ∩ (U \ Q) = {1, 2} := by
  sorry

end intersection_complement_l191_191893


namespace area_of_BCD_l191_191477

variables (a b c x y : ℝ)

-- Conditions
axiom h1 : x = (1 / 2) * a * b
axiom h2 : y = (1 / 2) * b * c

-- Conclusion to prove
theorem area_of_BCD (a b c x y : ℝ) (h1 : x = (1 / 2) * a * b) (h2 : y = (1 / 2) * b * c) : 
  (1 / 2) * b * c = y :=
sorry

end area_of_BCD_l191_191477


namespace probability_sum_greater_than_9_l191_191333

def num_faces := 6
def total_outcomes := num_faces * num_faces
def favorable_outcomes := 6
def probability := favorable_outcomes / total_outcomes

theorem probability_sum_greater_than_9 (h : total_outcomes = 36) :
  probability = 1 / 6 :=
by
  sorry

end probability_sum_greater_than_9_l191_191333


namespace area_of_triangle_l191_191352

theorem area_of_triangle (a b : ℝ) 
  (hypotenuse : ℝ) (median : ℝ)
  (h_side : hypotenuse = 2)
  (h_median : median = 1)
  (h_sum : a + b = 1 + Real.sqrt 3) 
  (h_pythagorean :(a^2 + b^2 = 4)): 
  (1/2 * a * b) = (Real.sqrt 3 / 2) := 
sorry

end area_of_triangle_l191_191352


namespace total_students_in_class_l191_191244

theorem total_students_in_class : 
  ∀ (total_candies students_candies : ℕ), 
    total_candies = 901 → students_candies = 53 → 
    students_candies * (total_candies / students_candies) = total_candies ∧ 
    total_candies % students_candies = 0 → 
    total_candies / students_candies = 17 := 
by 
  sorry

end total_students_in_class_l191_191244


namespace books_cost_l191_191101

theorem books_cost (total_cost_three_books cost_seven_books : ℕ) 
  (h₁ : total_cost_three_books = 45)
  (h₂ : cost_seven_books = 7 * (total_cost_three_books / 3)) : 
  cost_seven_books = 105 :=
  sorry

end books_cost_l191_191101


namespace sum_of_cubes_of_ages_l191_191303

noncomputable def dick_age : ℕ := 2
noncomputable def tom_age : ℕ := 5
noncomputable def harry_age : ℕ := 6

theorem sum_of_cubes_of_ages :
  4 * dick_age + 2 * tom_age = 3 * harry_age ∧ 
  3 * harry_age^2 = 2 * dick_age^2 + 4 * tom_age^2 ∧ 
  Nat.gcd (Nat.gcd dick_age tom_age) harry_age = 1 → 
  dick_age^3 + tom_age^3 + harry_age^3 = 349 :=
by
  intros h
  sorry

end sum_of_cubes_of_ages_l191_191303


namespace find_m_l191_191452

-- Definitions for the system of equations and the condition
def system_of_equations (x y m : ℝ) :=
  2 * x + 6 * y = 25 ∧ 6 * x + 2 * y = -11 ∧ x - y = m - 1

-- Statement to prove
theorem find_m (x y m : ℝ) (h : system_of_equations x y m) : m = -8 :=
  sorry

end find_m_l191_191452


namespace inequality_proof_l191_191027

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a^3 + 1 / b^3 - 1) * (b^3 + 1 / c^3 - 1) * (c^3 + 1 / a^3 - 1) ≤ (a * b * c + 1 / (a * b * c) - 1)^3 :=
by
  sorry

end inequality_proof_l191_191027


namespace minimal_colors_l191_191463

def complete_graph (n : ℕ) := Type

noncomputable def color_edges (G : complete_graph 2015) := ℕ → ℕ → ℕ

theorem minimal_colors (G : complete_graph 2015) (color : color_edges G) :
  (∀ {u v w : ℕ} (h1 : u ≠ v) (h2 : v ≠ w) (h3 : w ≠ u), color u v ≠ color v w ∧ color u v ≠ color u w ∧ color u w ≠ color v w) →
  ∃ C: ℕ, C = 2015 := 
sorry

end minimal_colors_l191_191463


namespace sum_div_mult_sub_result_l191_191832

-- Define the problem with conditions and expected answer
theorem sum_div_mult_sub_result :
  3521 + 480 / 60 * 3 - 521 = 3024 :=
by 
  sorry

end sum_div_mult_sub_result_l191_191832


namespace mean_of_remaining_three_numbers_l191_191078

variable {a b c : ℝ}

theorem mean_of_remaining_three_numbers (h1 : (a + b + c + 103) / 4 = 90) : (a + b + c) / 3 = 85.7 :=
by
  -- Sorry placeholder for the proof
  sorry

end mean_of_remaining_three_numbers_l191_191078


namespace polyhedron_has_triangular_face_l191_191324

-- Let's define the structure of a polyhedron, its vertices, edges, and faces.
structure Polyhedron :=
(vertices : ℕ)
(edges : ℕ)
(faces : ℕ)

-- Let's assume a function that indicates if a polyhedron is convex.
def is_convex (P : Polyhedron) : Prop := sorry  -- Convexity needs a rigorous formal definition.

-- Define a face of a polyhedron as an n-sided polygon.
structure Face :=
(sides : ℕ)

-- Predicate to check if a face is triangular.
def is_triangle (F : Face) : Prop := F.sides = 3

-- Predicate to check if each vertex has at least four edges meeting at it.
def each_vertex_has_at_least_four_edges (P : Polyhedron) : Prop := 
  sorry  -- This would need a more intricate definition involving the degrees of vertices.

-- We state the theorem using the defined concepts.
theorem polyhedron_has_triangular_face 
(P : Polyhedron) 
(h1 : is_convex P) 
(h2 : each_vertex_has_at_least_four_edges P) :
∃ (F : Face), is_triangle F :=
sorry

end polyhedron_has_triangular_face_l191_191324


namespace parabola_expression_l191_191459

theorem parabola_expression 
  (a b : ℝ) 
  (h : 9 = a * (-2)^2 + b * (-2) + 5) : 
  2 * a - b + 6 = 8 :=
by
  sorry

end parabola_expression_l191_191459


namespace razorback_shop_jersey_revenue_l191_191077

theorem razorback_shop_jersey_revenue :
  let price_per_tshirt := 67
  let price_per_jersey := 165
  let tshirts_sold := 74
  let jerseys_sold := 156
  jerseys_sold * price_per_jersey = 25740 := by
  sorry

end razorback_shop_jersey_revenue_l191_191077


namespace smallest_integer_in_set_l191_191465

theorem smallest_integer_in_set (n : ℤ) (h : n+4 < 3 * ((n + (n + 1) + (n + 2) + (n + 3) + (n + 4)) / 5)) : n ≥ 0 :=
by sorry

end smallest_integer_in_set_l191_191465


namespace find_number_l191_191521

theorem find_number (x : ℝ) (h : 0.8 * x = (2/5 : ℝ) * 25 + 22) : x = 40 :=
by
  sorry

end find_number_l191_191521


namespace num_students_B_l191_191208

-- Define the given conditions
variables (x : ℕ) -- The number of students who get a B

noncomputable def number_of_A := 2 * x
noncomputable def number_of_C := (12 / 10 : ℤ) * x -- Using (12 / 10) to approximate 1.2 in integers

-- Given total number of students is 42 for integer result
def total_students := 42

-- Lean statement to show number of students getting B is 10
theorem num_students_B : 4.2 * (x : ℝ) = 42 → x = 10 :=
by
  sorry

end num_students_B_l191_191208


namespace gcd_polynomial_example_l191_191314

theorem gcd_polynomial_example (b : ℤ) (h : ∃ k : ℤ, b = 2 * 1177 * k) :
  Int.gcd (3 * b^2 + 34 * b + 76) (b + 14) = 2 :=
by
  sorry

end gcd_polynomial_example_l191_191314


namespace max_n_for_sum_lt_zero_l191_191500

-- Define the arithmetic sequence and associated properties
def is_arithmetic_sequence (a : ℕ → ℚ) : Prop :=
  ∃ d : ℚ, ∀ n : ℕ, a (n + 1) = a n + d

-- Define the sum of the first n terms of the sequence
def sum_of_first_n_terms (a : ℕ → ℚ) (n : ℕ) : ℚ :=
  ∑ k in finset.range n, a k

-- Problem statement
theorem max_n_for_sum_lt_zero
  (a : ℕ → ℚ)
  (h_arith : is_arithmetic_sequence a)
  (h_a1 : a 1 < 0)
  (h_a2015_a2016 : a 2015 + a 2016 > 0)
  (h_a2015_a2016_neg_product : a 2015 * a 2016 < 0) :
  ∃ n : ℕ, n = 4029 ∧ sum_of_first_n_terms a n < 0 ∧ ∀ m : ℕ, m > n → sum_of_first_n_terms a m ≥ 0 :=
sorry

end max_n_for_sum_lt_zero_l191_191500


namespace sum_of_cubes_eq_twice_product_of_roots_l191_191163

theorem sum_of_cubes_eq_twice_product_of_roots (m : ℝ) :
  (∃ a b : ℝ, (3*a^2 + 6*a + m = 0) ∧ (3*b^2 + 6*b + m = 0) ∧ (a ≠ b)) → 
  (a^3 + b^3 = 2 * a * b) → 
  m = 6 :=
by
  intros h_exists sum_eq_twice_product
  sorry

end sum_of_cubes_eq_twice_product_of_roots_l191_191163


namespace total_oysters_eaten_l191_191782

/-- Squido eats 200 oysters -/
def Squido_eats := 200

/-- Crabby eats at least twice as many oysters as Squido -/
def Crabby_eats := 2 * Squido_eats

/-- Total oysters eaten by Squido and Crabby -/
theorem total_oysters_eaten : Squido_eats + Crabby_eats = 600 := 
by
  sorry

end total_oysters_eaten_l191_191782


namespace valid_group_count_l191_191634

theorem valid_group_count (n : ℕ) (h_n : n = 4043) :
  (Nat.choose n 3) - n * (Nat.choose 2021 2) = 
  (Nat.choose 4043 3) - 4043 * (Nat.choose 2021 2) :=
by
  rw h_n
  simp
  sorry

end valid_group_count_l191_191634


namespace find_a_l191_191190

theorem find_a (x y a : ℝ) (h1 : x + 3 * y = 4 - a) 
  (h2 : x - y = -3 * a) (h3 : x + y = 0) : a = 1 :=
sorry

end find_a_l191_191190


namespace repaired_shoes_last_time_l191_191667

theorem repaired_shoes_last_time :
  let cost_of_repair := 13.50
  let cost_of_new := 32.00
  let duration_of_new := 2.0
  let surcharge := 0.1852
  let avg_cost_new := cost_of_new / duration_of_new
  let avg_cost_repair (T : ℝ) := cost_of_repair / T
  (avg_cost_new = (1 + surcharge) * avg_cost_repair 1) ↔ T = 1 := 
by
  sorry

end repaired_shoes_last_time_l191_191667


namespace ticket_distribution_count_l191_191633

theorem ticket_distribution_count :
  let A := 2
  let B := 2
  let C := 1
  let D := 1
  let total_tickets := A + B + C + D
  ∃ (num_dist : ℕ), num_dist = 180 :=
by {
  sorry
}

end ticket_distribution_count_l191_191633


namespace ana_final_salary_l191_191001

def initial_salary : ℝ := 2500
def june_raise : ℝ := initial_salary * 0.15
def june_bonus : ℝ := 300
def salary_after_june : ℝ := initial_salary + june_raise + june_bonus
def july_pay_cut : ℝ := salary_after_june * 0.25
def final_salary : ℝ := salary_after_june - july_pay_cut

theorem ana_final_salary :
  final_salary = 2381.25 := by
  -- sorry is used here to skip the proof
  sorry

end ana_final_salary_l191_191001


namespace equilateral_triangle_condition_l191_191760

-- We define points in a plane and vectors between these points
structure Point where
  x : ℝ
  y : ℝ

-- Vector subtraction
def vector (p q : Point) : Point :=
  { x := q.x - p.x, y := q.y - p.y }

-- The equation required to hold for certain type of triangles
def bisector_eq_zero (A B C A1 B1 C1 : Point) : Prop :=
  let AA1 := vector A A1
  let BB1 := vector B B1
  let CC1 := vector C C1
  AA1.x + BB1.x + CC1.x = 0 ∧ AA1.y + BB1.y + CC1.y = 0

-- Property of equilateral triangle
def is_equilateral (A B C : Point) : Prop :=
  let AB := vector A B
  let BC := vector B C
  let CA := vector C A
  (AB.x^2 + AB.y^2 = BC.x^2 + BC.y^2 ∧ BC.x^2 + BC.y^2 = CA.x^2 + CA.y^2)

-- Main theorem statement
theorem equilateral_triangle_condition (A B C A1 B1 C1 : Point)
  (h : bisector_eq_zero A B C A1 B1 C1) :
  is_equilateral A B C :=
sorry

end equilateral_triangle_condition_l191_191760


namespace solve_for_y_l191_191362

noncomputable def g (y : ℝ) : ℝ := (30 * y + (30 * y + 27)^(1/3))^(1/3)

theorem solve_for_y :
  (∃ y : ℝ, g y = 15) ↔ (∃ y : ℝ, y = 1674 / 15) :=
by
  sorry

end solve_for_y_l191_191362


namespace nat_pow_eq_sub_two_case_l191_191998

theorem nat_pow_eq_sub_two_case (n : ℕ) : (∃ a k : ℕ, k ≥ 2 ∧ 2^n - 1 = a^k) ↔ (n = 0 ∨ n = 1) :=
by
  sorry

end nat_pow_eq_sub_two_case_l191_191998


namespace binomial_12_3_eq_220_l191_191984

theorem binomial_12_3_eq_220 : nat.choose 12 3 = 220 :=
by {
  sorry
}

end binomial_12_3_eq_220_l191_191984


namespace area_of_circle_segment_l191_191110

-- Definitions for the conditions in the problem
def circle_eq (x y : ℝ) : Prop := x^2 - 10 * x + y^2 = 9
def line_eq (x y : ℝ) : Prop := y = x - 5

-- The area of the portion of the circle that lies above the x-axis and to the left of the line y = x - 5
theorem area_of_circle_segment :
  let area_of_circle := 34 * Real.pi
  let portion_fraction := 1 / 8
  portion_fraction * area_of_circle = 4.25 * Real.pi :=
by
  sorry

end area_of_circle_segment_l191_191110


namespace find_divisor_l191_191204

variable (r q d v : ℕ)
variable (h1 : r = 8)
variable (h2 : q = 43)
variable (h3 : d = 997)

theorem find_divisor : d = v * q + r → v = 23 :=
by
  sorry

end find_divisor_l191_191204


namespace correct_choice_of_f_l191_191032

def f1 (x : ℝ) : ℝ := (x - 1)^2 + 3 * (x - 1)
def f2 (x : ℝ) : ℝ := 2 * (x - 1)
def f3 (x : ℝ) : ℝ := 2 * (x - 1)^2
def f4 (x : ℝ) : ℝ := x - 1

theorem correct_choice_of_f (h : (deriv f1 1 = 3) ∧ (deriv f2 1 ≠ 3) ∧ (deriv f3 1 ≠ 3) ∧ (deriv f4 1 ≠ 3)) : 
  ∀ f, (f = f1 ∨ f = f2 ∨ f = f3 ∨ f = f4) → (deriv f 1 = 3 → f = f1) :=
by sorry

end correct_choice_of_f_l191_191032


namespace triangle_inequality_check_l191_191952

theorem triangle_inequality_check (a b c : ℝ) (h1 : a + b > c) (h2 : b + c > a) (h3 : c + a > b) : 
  (a = 5 ∧ b = 8 ∧ c = 12) → (a + b > c ∧ b + c > a ∧ c + a > b) :=
by 
  intros h
  rcases h with ⟨rfl, rfl, rfl⟩
  exact ⟨h1, h2, h3⟩

end triangle_inequality_check_l191_191952


namespace power_equivalence_l191_191042

theorem power_equivalence (m : ℕ) : 16^6 = 4^m → m = 12 :=
by
  sorry

end power_equivalence_l191_191042


namespace initial_people_on_train_l191_191827

theorem initial_people_on_train 
    (P : ℕ)
    (h1 : 116 = P - 4)
    (h2 : P = 120)
    : 
    P = 116 + 4 := by
have h3 : P = 120 := by sorry
exact h3

end initial_people_on_train_l191_191827


namespace correct_options_l191_191439

-- Given condition
def curve (x y : ℝ) : Prop := x^2 + y^2 - 2*x - 2 = 0

-- Option B assertion
def option_B (x y : ℝ) : Prop := (x^2 + y^2 - 4)/((x - 1)^2 + y^2 + 1) ≤ 2 + Real.sqrt 6

-- Option D assertion
def option_D (x y : ℝ) : Prop := x - Real.sqrt 2 * y + 2 = 0

-- Theorem to prove both options B and D are correct under the given condition
theorem correct_options {x y : ℝ} (h : curve x y) : option_B x y ∧ option_D x y := by
  sorry

end correct_options_l191_191439


namespace marbles_leftover_l191_191656

theorem marbles_leftover (g j : ℕ) (hg : g % 8 = 5) (hj : j % 8 = 6) :
  ((g + 5 + j) % 8) = 0 :=
by
  sorry

end marbles_leftover_l191_191656


namespace age_ratio_in_two_years_l191_191671

variable (S M : ℕ)

-- Conditions
def sonCurrentAge : Prop := S = 18
def manCurrentAge : Prop := M = S + 20
def multipleCondition : Prop := ∃ k : ℕ, M + 2 = k * (S + 2)

-- Statement to prove
theorem age_ratio_in_two_years (h1 : sonCurrentAge S) (h2 : manCurrentAge S M) (h3 : multipleCondition S M) : 
  (M + 2) / (S + 2) = 2 := 
by
  sorry

end age_ratio_in_two_years_l191_191671


namespace smallest_n_l191_191793

theorem smallest_n
  (n : ℕ)
  (h1 : n % 2 = 1)
  (h2 : n % 3 = 1)
  (h3 : n % 4 = 1)
  (h4 : n % 5 = 1)
  (h5 : n % 6 = 1)
  (h6 : n % 7 = 1)
  (h7 : 8 ∣ n) :
  n = 1681 :=
  sorry

end smallest_n_l191_191793


namespace midpoint_to_plane_distance_l191_191316

noncomputable def distance_to_plane (A B P: ℝ) (dA dB: ℝ) : ℝ :=
if h : A = B then |dA|
else if h1 : dA + dB = (2 : ℝ) * (dA + dB) / 2 then (dA + dB) / 2
else if h2 : |dB - dA| = (2 : ℝ) * |dB - dA| / 2 then |dB - dA| / 2
else 0

theorem midpoint_to_plane_distance
  (α : Type*)
  (A B P: ℝ)
  {dA dB : ℝ}
  (h_dA : dA = 3)
  (h_dB : dB = 5) :
  distance_to_plane A B P dA dB = 4 ∨ distance_to_plane A B P dA dB = 1 :=
by sorry

end midpoint_to_plane_distance_l191_191316


namespace john_new_bench_press_l191_191063

theorem john_new_bench_press (initial_weight : ℕ) (decrease_percent : ℕ) (retain_percent : ℕ) (training_factor : ℕ) (final_weight : ℕ) 
  (h1 : initial_weight = 500)
  (h2 : decrease_percent = 80)
  (h3 : retain_percent = 20)
  (h4 : training_factor = 3)
  (h5 : final_weight = initial_weight * retain_percent / 100 * training_factor) : 
  final_weight = 300 := 
by sorry

end john_new_bench_press_l191_191063


namespace slope_of_line_l191_191034

theorem slope_of_line : 
  (∀ x y : ℝ, (y = (1/2) * x + 1) → ∃ m : ℝ, m = 1/2) :=
sorry

end slope_of_line_l191_191034


namespace theater_seats_l191_191270

theorem theater_seats
  (A : ℕ) -- Number of adult tickets
  (C : ℕ) -- Number of child tickets
  (hC : C = 63) -- 63 child tickets sold
  (total_revenue : ℕ) -- Total Revenue
  (hRev : total_revenue = 519) -- Total revenue is 519
  (adult_ticket_price : ℕ := 12) -- Price per adult ticket
  (child_ticket_price : ℕ := 5) -- Price per child ticket
  (hRevEq : adult_ticket_price * A + child_ticket_price * C = total_revenue) -- Revenue equation
  : A + C = 80 := sorry

end theater_seats_l191_191270


namespace parallel_lines_m_values_l191_191016

theorem parallel_lines_m_values (m : ℝ) :
  (∀ x y : ℝ, (3 + m) * x + 4 * y = 5 → 2 * x + (5 + m) * y = 8) →
  (m = -1 ∨ m = -7) :=
by
  sorry

end parallel_lines_m_values_l191_191016


namespace max_value_of_k_l191_191572

theorem max_value_of_k:
  ∃ (k : ℕ), 
  (∀ (a b : ℕ → ℕ) (h : ∀ i, a i < b i) (no_share : ∀ i j, i ≠ j → (a i ≠ a j ∧ a i ≠ b j ∧ b i ≠ a j ∧ b i ≠ b j)) (distinct_sums : ∀ i j, i ≠ j → a i + b i ≠ a j + b j) (sum_limit : ∀ i, a i + b i ≤ 3011), 
    k ≤ 3011 ∧ k = 1204) := sorry

end max_value_of_k_l191_191572


namespace books_printed_l191_191727

-- Definitions of the conditions
def book_length := 600
def pages_per_sheet := 8
def total_sheets := 150

-- The theorem to prove
theorem books_printed : (total_sheets * pages_per_sheet / book_length) = 2 := by
  sorry

end books_printed_l191_191727


namespace leos_current_weight_l191_191456

theorem leos_current_weight (L K : ℝ) 
  (h1 : L + 10 = 1.5 * K) 
  (h2 : L + K = 180) : L = 104 := 
by 
  sorry

end leos_current_weight_l191_191456


namespace total_distance_biked_l191_191475

-- Definitions of the given conditions
def biking_time_to_park : ℕ := 15
def biking_time_return : ℕ := 25
def average_speed : ℚ := 6 -- miles per hour

-- Total biking time in minutes, then converted to hours
def total_biking_time_minutes : ℕ := biking_time_to_park + biking_time_return
def total_biking_time_hours : ℚ := total_biking_time_minutes / 60

-- Prove that the total distance biked is 4 miles
theorem total_distance_biked : total_biking_time_hours * average_speed = 4 := 
by
  -- proof will be here
  sorry

end total_distance_biked_l191_191475


namespace tangent_line_at_2_neg3_l191_191763

noncomputable def tangent_line (x : ℝ) : ℝ := (1 + x) / (1 - x)

theorem tangent_line_at_2_neg3 :
  ∃ m b, ∀ x, (tangent_line x = m * x + b) →
  ∃ y, (2 * x - y - 7 = 0) :=
by
  sorry

end tangent_line_at_2_neg3_l191_191763


namespace min_value_of_expression_l191_191033

theorem min_value_of_expression {a b : ℝ} (ha : 0 < a) (hb : 0 < b) (h : 1/a + 2/b = 3) :
  (a + 1) * (b + 2) = 50/9 :=
sorry

end min_value_of_expression_l191_191033


namespace range_of_m_l191_191696

variable (m : ℝ)
def p := ∀ x : ℝ, x ∈ Set.Icc (-1 : ℝ) 1 → x^2 - 2*x - 4*m^2 + 8*m - 2 ≥ 0
def q := ∃ x : ℝ, x ∈ Set.Icc (1 : ℝ) 2 ∧ Real.log (x^2 - m*x + 1) / Real.log (1/2) < -1

theorem range_of_m (hp : p m) (hq : q m) (hl : (p m) ∨ (q m)) (hf : ¬ ((p m) ∧ (q m))) :
  m < 1/2 ∨ m = 3/2 := sorry

end range_of_m_l191_191696


namespace no_real_solutions_for_equation_l191_191889

theorem no_real_solutions_for_equation (x : ℝ) : ¬(∃ x : ℝ, (8 * x^2 + 150 * x - 5) / (3 * x + 50) = 4 * x + 7) :=
sorry

end no_real_solutions_for_equation_l191_191889


namespace extreme_value_proof_l191_191458

noncomputable def extreme_value (x y : ℝ) := 4 * x + 3 * y 

theorem extreme_value_proof 
  (x y : ℝ)
  (hx : 0 < x)
  (hy : 0 < y)
  (h : x + y = 5 * x * y) : 
  extreme_value x y = 3 :=
sorry

end extreme_value_proof_l191_191458


namespace ratio_of_men_to_women_l191_191632

theorem ratio_of_men_to_women
  (M W : ℕ)
  (h1 : W = M + 6)
  (h2 : M + W = 16) :
  M * 11 = 5 * W :=
by
    -- We can explicitly construct the necessary proof here, but according to instructions we add sorry to bypass for now
    sorry

end ratio_of_men_to_women_l191_191632


namespace total_fencing_cost_is_correct_l191_191714

-- Define the fencing cost per side
def costPerSide : Nat := 69

-- Define the number of sides for a square
def sidesOfSquare : Nat := 4

-- Define the total cost calculation for fencing the square
def totalCostOfFencing (costPerSide : Nat) (sidesOfSquare : Nat) := costPerSide * sidesOfSquare

-- Prove that for a given cost per side and number of sides, the total cost of fencing the square is 276 dollars
theorem total_fencing_cost_is_correct : totalCostOfFencing 69 4 = 276 :=
by
    -- Proof goes here
    sorry

end total_fencing_cost_is_correct_l191_191714


namespace manufacturing_cost_of_shoe_l191_191767

theorem manufacturing_cost_of_shoe
  (transportation_cost_per_shoe : ℝ)
  (selling_price_per_shoe : ℝ)
  (gain_percentage : ℝ)
  (manufacturing_cost : ℝ)
  (H1 : transportation_cost_per_shoe = 5)
  (H2 : selling_price_per_shoe = 282)
  (H3 : gain_percentage = 0.20)
  (H4 : selling_price_per_shoe = (manufacturing_cost + transportation_cost_per_shoe) * (1 + gain_percentage)) :
  manufacturing_cost = 230 :=
sorry

end manufacturing_cost_of_shoe_l191_191767


namespace inequality_solution_set_correct_l191_191167

noncomputable def inequality_solution_set (a b c x : ℝ) : Prop :=
  (a > c) → (b + c > 0) → ((x - b < 0 ∧ x < c) ∨ (x > a)) → ((x - c) * (x + b) / (x - a) > 0)

theorem inequality_solution_set_correct (a b c : ℝ) :
  a > c → b + c > 0 → ∀ x, ((a > c) → (b + c > 0) → (((x - b < 0 ∧ x < c) ∨ (x > a)) → ((x - c) * (x + b) / (x - a) > 0))) :=
by
  intros h1 h2 x
  sorry

end inequality_solution_set_correct_l191_191167


namespace tangent_line_equation_at_point_l191_191762

theorem tangent_line_equation_at_point :
  let y := λ x : ℝ, (1 + x) / (1 - x)
  let x₀ := 2
  let y₀ := -3
  let m := 2
  (2 * x₀ - y₀ - 7 = 0) :=
by 
  sorry

end tangent_line_equation_at_point_l191_191762


namespace bag_of_chips_weight_l191_191126

theorem bag_of_chips_weight (c : ℕ) : 
  (∀ (t : ℕ), t = 9) → 
  (∀ (b : ℕ), b = 6) → 
  (∀ (x : ℕ), x = 4 * 6) → 
  (21 * 16 = 336) →
  (336 - 24 * 9 = 6 * c) → 
  c = 20 :=
by
  intros ht hb hx h_weight_total h_weight_chips
  sorry

end bag_of_chips_weight_l191_191126


namespace total_pages_written_is_24_l191_191621

def normal_letter_interval := 3
def time_per_normal_letter := 20
def time_per_page := 10
def additional_time_factor := 2
def time_spent_long_letter := 80
def days_in_month := 30

def normal_letters_written := days_in_month / normal_letter_interval
def pages_per_normal_letter := time_per_normal_letter / time_per_page
def total_pages_normal_letters := normal_letters_written * pages_per_normal_letter

def time_per_page_long_letter := additional_time_factor * time_per_page
def pages_long_letter := time_spent_long_letter / time_per_page_long_letter

def total_pages_written := total_pages_normal_letters + pages_long_letter

theorem total_pages_written_is_24 : total_pages_written = 24 := by
  sorry

end total_pages_written_is_24_l191_191621


namespace problem_statement_l191_191445

noncomputable def necessary_but_not_sufficient_condition (x y : ℝ) (hx : x > 0) : Prop :=
  (x > |y| → x > y) ∧ ¬ (x > y → x > |y|)

theorem problem_statement
  (x y : ℝ)
  (hx : x > 0)
  : necessary_but_not_sufficient_condition x y hx :=
sorry

end problem_statement_l191_191445


namespace xy_value_l191_191198

theorem xy_value (x y : ℝ) (h : x * (x + y) = x^2 + 12) : x * y = 12 := by
  sorry

end xy_value_l191_191198


namespace at_least_6_heads_probability_l191_191265

open_locale big_operators

theorem at_least_6_heads_probability : 
  let outcomes := 2 ^ 9 in
  let total_ways := (Nat.choose 9 6 + Nat.choose 9 7 + Nat.choose 9 8 + Nat.choose 9 9) in
  total_ways / outcomes = 130 / 512 :=
by
  sorry

end at_least_6_heads_probability_l191_191265


namespace math_equivalent_problem_l191_191719

noncomputable def correct_difference (A B C D : ℕ) (incorrect_difference : ℕ) : ℕ :=
  if (B = 3) ∧ (D = 2) ∧ (C = 5) ∧ (incorrect_difference = 60) then
    ((A * 10 + B) - 52)
  else
    0

theorem math_equivalent_problem (A : ℕ) : correct_difference A 3 5 2 60 = 31 :=
by
  sorry

end math_equivalent_problem_l191_191719


namespace black_squares_covered_by_trominoes_l191_191253

def is_odd (n : ℕ) : Prop :=
  n % 2 = 1

noncomputable def min_trominoes (n : ℕ) : ℕ :=
  ((n + 1) ^ 2) / 4

theorem black_squares_covered_by_trominoes (n : ℕ) (h1 : n ≥ 7) (h2 : is_odd n):
  ∀ n : ℕ, ∃ k : ℕ, k = min_trominoes n :=
by
  sorry

end black_squares_covered_by_trominoes_l191_191253


namespace triangle_ABC_properties_l191_191184

noncomputable def f (x : ℝ) : ℝ := Real.exp x + x

theorem triangle_ABC_properties
  (xA xB xC : ℝ)
  (h_seq : xA < xB ∧ xB < xC ∧ 2 * xB = xA + xC)
  : (f xB + (f xA + f xC) / 2 > f ((xA + xC) / 2)) ∧ (f xA ≠ f xB ∧ f xB ≠ f xC) := 
sorry

end triangle_ABC_properties_l191_191184


namespace square_of_binomial_l191_191646

theorem square_of_binomial (k : ℝ) : (∃ b : ℝ, x^2 - 20 * x + k = (x + b)^2) -> k = 100 :=
sorry

end square_of_binomial_l191_191646


namespace line_circle_relationship_l191_191045

theorem line_circle_relationship (m : ℝ) :
  (∃ x y : ℝ, (mx + y - m - 1 = 0) ∧ (x^2 + y^2 = 2)) ∨ 
  (∃ x : ℝ, (x - 1)^2 + (m*(x - 1) + (1 - 1))^2 = 2) :=
by
  sorry

end line_circle_relationship_l191_191045


namespace find_a_l191_191346

theorem find_a (a : ℕ) (h_pos : 0 < a)
  (h_cube : ∀ n : ℕ, 0 < n → ∃ k : ℤ, 4 * ((a : ℤ) ^ n + 1) = k^3) :
  a = 1 :=
sorry

end find_a_l191_191346


namespace total_cars_produced_l191_191680

def CarCompanyA_NorthAmerica := 3884
def CarCompanyA_Europe := 2871
def CarCompanyA_Asia := 1529

def CarCompanyB_NorthAmerica := 4357
def CarCompanyB_Europe := 3690
def CarCompanyB_Asia := 1835

def CarCompanyC_NorthAmerica := 2937
def CarCompanyC_Europe := 4210
def CarCompanyC_Asia := 977

def TotalNorthAmerica :=
  CarCompanyA_NorthAmerica + CarCompanyB_NorthAmerica + CarCompanyC_NorthAmerica

def TotalEurope :=
  CarCompanyA_Europe + CarCompanyB_Europe + CarCompanyC_Europe

def TotalAsia :=
  CarCompanyA_Asia + CarCompanyB_Asia + CarCompanyC_Asia

def TotalProduction := TotalNorthAmerica + TotalEurope + TotalAsia

theorem total_cars_produced : TotalProduction = 26290 := 
by sorry

end total_cars_produced_l191_191680


namespace solve_inequality_l191_191891

theorem solve_inequality (x : ℝ) (h : x ≠ 1) : (x / (x - 1) ≥ 2 * x) ↔ (x ≤ 0 ∨ (1 < x ∧ x ≤ 3 / 2)) :=
by
  sorry

end solve_inequality_l191_191891


namespace man_swims_distance_back_l191_191967

def swimming_speed_still_water : ℝ := 8
def speed_of_water : ℝ := 4
def time_taken_against_current : ℝ := 2
def distance_swum : ℝ := 8

theorem man_swims_distance_back :
  (distance_swum = (swimming_speed_still_water - speed_of_water) * time_taken_against_current) :=
by
  -- The proof will be filled in later.
  sorry

end man_swims_distance_back_l191_191967


namespace round_trip_time_l191_191488

variable (dist : ℝ)
variable (speed_to_work : ℝ)
variable (speed_to_home : ℝ)

theorem round_trip_time (h_dist : dist = 24) (h_speed_to_work : speed_to_work = 60) (h_speed_to_home : speed_to_home = 40) :
    (dist / speed_to_work + dist / speed_to_home) = 1 := 
by 
  sorry

end round_trip_time_l191_191488


namespace total_money_l191_191382

def value_of_quarters (count: ℕ) : ℝ := count * 0.25
def value_of_dimes (count: ℕ) : ℝ := count * 0.10
def value_of_nickels (count: ℕ) : ℝ := count * 0.05
def value_of_pennies (count: ℕ) : ℝ := count * 0.01

theorem total_money (q d n p : ℕ) :
  q = 10 → d = 3 → n = 4 → p = 200 →
  value_of_quarters q + value_of_dimes d + value_of_nickels n + value_of_pennies p = 5.00 :=
by {
  intros,
  sorry
}

end total_money_l191_191382


namespace find_f_of_4_l191_191878

def f (a b c x : ℝ) := a * x^2 + b * x + c

theorem find_f_of_4 {a b c : ℝ} (h1 : f a b c 1 = 3) (h2 : f a b c 2 = 12) (h3 : f a b c 3 = 27) :
  f a b c 4 = 48 := 
sorry

end find_f_of_4_l191_191878


namespace total_charging_time_l191_191136

def charge_smartphone_full : ℕ := 26
def charge_tablet_full : ℕ := 53
def charge_phone_half : ℕ := charge_smartphone_full / 2
def charge_tablet : ℕ := charge_tablet_full

theorem total_charging_time : 
  charge_phone_half + charge_tablet = 66 := by
  sorry

end total_charging_time_l191_191136


namespace morgan_change_l191_191883

theorem morgan_change:
  let hamburger := 5.75
  let onion_rings := 2.50
  let smoothie := 3.25
  let side_salad := 3.75
  let cake := 4.20
  let total_cost := hamburger + onion_rings + smoothie + side_salad + cake
  let payment := 50
  let change := payment - total_cost
  ℝ := by
    exact sorry

end morgan_change_l191_191883


namespace area_of_shape_l191_191229

def points := [(0, 1), (1, 2), (3, 2), (4, 1), (2, 0)]

theorem area_of_shape : 
  let I := 6 -- Number of interior points
  let B := 5 -- Number of boundary points
  ∃ (A : ℝ), A = I + B / 2 - 1 ∧ A = 7.5 := 
  by
    use 7.5
    simp
    sorry

end area_of_shape_l191_191229


namespace min_value_l191_191178

variables (a b c : ℝ)
variable (hpos : a > 0 ∧ b > 0 ∧ c > 0)
variable (hsum : a + b + c = 1)

theorem min_value (hpos : a > 0 ∧ b > 0 ∧ c > 0) (hsum : a + b + c = 1) :
  9 * a^2 + 4 * b^2 + (1/4) * c^2 = 36 / 157 := 
sorry

end min_value_l191_191178


namespace correct_options_l191_191440

-- Given condition
def curve (x y : ℝ) : Prop := x^2 + y^2 - 2*x - 2 = 0

-- Option B assertion
def option_B (x y : ℝ) : Prop := (x^2 + y^2 - 4)/((x - 1)^2 + y^2 + 1) ≤ 2 + Real.sqrt 6

-- Option D assertion
def option_D (x y : ℝ) : Prop := x - Real.sqrt 2 * y + 2 = 0

-- Theorem to prove both options B and D are correct under the given condition
theorem correct_options {x y : ℝ} (h : curve x y) : option_B x y ∧ option_D x y := by
  sorry

end correct_options_l191_191440


namespace find_m_l191_191748

theorem find_m (x y m : ℝ) (opp_sign: y = -x) 
  (h1 : 4 * x + 2 * y = 3 * m) 
  (h2 : 3 * x + y = m + 2) : 
  m = 1 :=
by 
  -- Placeholder for the steps to prove the theorem
  sorry

end find_m_l191_191748


namespace actual_discount_is_expected_discount_l191_191810

-- Define the conditions
def promotional_discount := 20 / 100  -- 20% discount
def vip_card_discount := 10 / 100  -- 10% additional discount

-- Define the combined discount calculation
def combined_discount := (1 - promotional_discount) * (1 - vip_card_discount)

-- Define the expected discount off the original price
def expected_discount := 28 / 100  -- 28% discount

-- Theorem statement proving the combined discount is equivalent to the expected discount
theorem actual_discount_is_expected_discount :
  combined_discount = 1 - expected_discount :=
by
  -- Proof omitted.
  sorry

end actual_discount_is_expected_discount_l191_191810


namespace pair_comparison_l191_191276

theorem pair_comparison :
  (∀ (a b : ℤ), (a, b) = (-2^4, (-2)^4) → a ≠ b) ∧
  (∀ (a b : ℤ), (a, b) = (5^3, 3^5) → a ≠ b) ∧
  (∀ (a b : ℤ), (a, b) = (-(-3), -|-3|) → a ≠ b) ∧
  (∀ (a b : ℤ), (a, b) = ((-1)^2, (-1)^2008) → a = b) :=
by
  sorry

end pair_comparison_l191_191276


namespace find_p_l191_191595

theorem find_p (p : ℕ) : 64^5 = 8^p → p = 10 :=
by
  intro h
  sorry

end find_p_l191_191595


namespace larger_number_of_two_integers_l191_191858

theorem larger_number_of_two_integers (x y : ℤ) (h1 : x * y = 30) (h2 : x + y = 13) : (max x y = 10) :=
by
  sorry

end larger_number_of_two_integers_l191_191858


namespace find_square_number_divisible_by_9_between_40_and_90_l191_191688

theorem find_square_number_divisible_by_9_between_40_and_90 :
  ∃ x : ℕ, (∃ n : ℕ, x = n^2) ∧ (9 ∣ x) ∧ 40 < x ∧ x < 90 ∧ x = 81 :=
by
  sorry

end find_square_number_divisible_by_9_between_40_and_90_l191_191688


namespace range_of_expression_l191_191847

theorem range_of_expression (x y z : ℝ) (h1 : 0 < x) (h2 : 0 < y) (h3 : 0 < z) (h4 : x + y + z = 1) :
  0 < (x * y + y * z + z * x - 2 * x * y * z) ∧ (x * y + y * z + z * x - 2 * x * y * z) ≤ 7 / 27 := by
  sorry

end range_of_expression_l191_191847


namespace triangle_area_is_correct_l191_191246

noncomputable def triangle_area : ℝ :=
  let A := (3, 3)
  let B := (4.5, 7.5)
  let C := (7.5, 4.5)
  1 / 2 * |(A.1 * (B.2 - C.2) + B.1 * (C.2 - A.2) + C.1 * (A.2 - B.2) : ℝ)|

theorem triangle_area_is_correct : triangle_area = 9 := by
  sorry

end triangle_area_is_correct_l191_191246


namespace adoption_complete_in_7_days_l191_191803

-- Define the initial number of puppies
def initial_puppies := 9

-- Define the number of puppies brought in later
def additional_puppies := 12

-- Define the number of puppies adopted per day
def adoption_rate := 3

-- Define the total number of puppies
def total_puppies : Nat := initial_puppies + additional_puppies

-- Define the number of days required to adopt all puppies
def adoption_days : Nat := total_puppies / adoption_rate

-- Prove that the number of days to adopt all puppies is 7
theorem adoption_complete_in_7_days : adoption_days = 7 := by
  -- The exact implementation of the proof is not necessary,
  -- so we use sorry to skip the proof.
  sorry

end adoption_complete_in_7_days_l191_191803


namespace ratio_of_area_to_perimeter_l191_191941

noncomputable def altitude_of_equilateral_triangle (s : ℝ) : ℝ :=
  s * (Real.sqrt 3 / 2)

noncomputable def area_of_equilateral_triangle (s : ℝ) : ℝ :=
  1 / 2 * s * altitude_of_equilateral_triangle s

noncomputable def perimeter_of_equilateral_triangle (s : ℝ) : ℝ :=
  3 * s

theorem ratio_of_area_to_perimeter (s : ℝ) (h : s = 10) :
    (area_of_equilateral_triangle s) / (perimeter_of_equilateral_triangle s) = 5 * Real.sqrt 3 / 6 :=
  by
  rw [h]
  sorry

end ratio_of_area_to_perimeter_l191_191941


namespace impossible_closed_chain_1997_tiles_l191_191284

/- 
Problem Statement:
Given 1997 square tiles placed sequentially on an infinite checkerboard grid such that:
1. Each tile covers one cell,
2. Tiles are numbered from 1 to 1997,
3. Adjacent cells on the checkerboard are of different colors,
4. Tiles with odd numbers land on cells of one color, and tiles with even numbers land on cells of the opposite color,
prove that forming a closed chain with these 1997 tiles is impossible.
-/

theorem impossible_closed_chain_1997_tiles :
  ∀ (n : ℕ) (chain : Fin n → ℕ), n = 1997 →
  (∀ i : Fin (n - 1), adjacent_tiles (chain i) (chain (i + 1))) →
  adjacent_tiles (chain 0) (chain (Fin.ofNat (n - 1))) →
  (∀ i : Fin n, chain i % 2 = i % 2) →
  false :=
by
  intros n chain h1 h2 h3 h4
  sorry


end impossible_closed_chain_1997_tiles_l191_191284


namespace students_distribution_l191_191424

theorem students_distribution (students villages : ℕ) (h_students : students = 4) (h_villages : villages = 3) :
  ∃ schemes : ℕ, schemes = 36 := 
sorry

end students_distribution_l191_191424


namespace binomial_12_3_equals_220_l191_191986

theorem binomial_12_3_equals_220 : Nat.choose 12 3 = 220 := by
  sorry

end binomial_12_3_equals_220_l191_191986


namespace total_distance_traveled_l191_191551

-- Definitions of conditions
def bess_throw_distance : ℕ := 20
def bess_throws : ℕ := 4
def holly_throw_distance : ℕ := 8
def holly_throws : ℕ := 5
def bess_effective_throw_distance : ℕ := 2 * bess_throw_distance

-- Theorem statement
theorem total_distance_traveled :
  (bess_throws * bess_effective_throw_distance + holly_throws * holly_throw_distance) = 200 := 
  by sorry

end total_distance_traveled_l191_191551


namespace journey_time_difference_l191_191127

theorem journey_time_difference :
  let speed := 40  -- mph
  let distance1 := 360  -- miles
  let distance2 := 320  -- miles
  (distance1 / speed - distance2 / speed) * 60 = 60 := 
by
  sorry

end journey_time_difference_l191_191127


namespace smallest_sum_of_squares_l191_191372

theorem smallest_sum_of_squares (x y : ℤ) (h : x^2 - y^2 = 145) : 
  ∃ x y, x^2 - y^2 = 145 ∧ x^2 + y^2 = 433 :=
sorry

end smallest_sum_of_squares_l191_191372


namespace football_banquet_total_food_l191_191002

-- Definitions representing the conditions
def individual_max_food (n : Nat) := n ≤ 2
def min_guests (g : Nat) := g ≥ 160

-- The proof problem statement
theorem football_banquet_total_food : 
  ∀ (n g : Nat), (∀ i, i ≤ g → individual_max_food n) ∧ min_guests g → g * n = 320 := 
by
  intros n g h
  sorry

end football_banquet_total_food_l191_191002


namespace algebra_problem_l191_191694

noncomputable def expression (a b : ℝ) : ℝ :=
  (3 * a + b / 3)⁻¹ * ((3 * a)⁻¹ + (b / 3)⁻¹)

theorem algebra_problem (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) :
  expression a b = (a * b)⁻¹ :=
by
  sorry

end algebra_problem_l191_191694


namespace major_airlines_free_snacks_l191_191259

variable (S : ℝ)

theorem major_airlines_free_snacks (h1 : 0.5 ≤ 1) (h2 : 0.5 = 1) :
  0.5 ≤ S :=
sorry

end major_airlines_free_snacks_l191_191259


namespace ratio_of_spinsters_to_cats_l191_191629

theorem ratio_of_spinsters_to_cats (S C : ℕ) (hS : S = 12) (hC : C = S + 42) : S / gcd S C = 2 ∧ C / gcd S C = 9 :=
by
  -- skip proof (use sorry)
  sorry

end ratio_of_spinsters_to_cats_l191_191629


namespace math_problem_l191_191107

variables {A B : Type} [Fintype A] [Fintype B]
          (p1 p2 : ℝ) (h1 : 1/2 < p1) (h2 : p1 < p2) (h3 : p2 < 1)
          (nA : ℕ) (hA : nA = 3) (nB : ℕ) (hB : nB = 3)

noncomputable def E_X : ℝ := nA * p1
noncomputable def E_Y : ℝ := nB * p2

noncomputable def D_X : ℝ := nA * p1 * (1 - p1)
noncomputable def D_Y : ℝ := nB * p2 * (1 - p2)

theorem math_problem :
  E_X p1 nA = 3 * p1 →
  E_Y p2 nB = 3 * p2 →
  D_X p1 nA = 3 * p1 * (1 - p1) →
  D_Y p2 nB = 3 * p2 * (1 - p2) →
  E_X p1 nA < E_Y p2 nB ∧ D_X p1 nA > D_Y p2 nB :=
by
  sorry

end math_problem_l191_191107


namespace numbers_not_divisible_by_5_or_7_l191_191855

theorem numbers_not_divisible_by_5_or_7 (n : ℕ) (h : n = 999) :
  let num_div_5 := n / 5,
      num_div_7 := n / 7,
      num_div_35 := n / 35,
      total_eliminated := num_div_5 + num_div_7 - num_div_35,
      result := n - total_eliminated in
  result = 686 := by
{
  sorry
}

end numbers_not_divisible_by_5_or_7_l191_191855


namespace smallest_positive_integer_modulo_l191_191390

theorem smallest_positive_integer_modulo {n : ℕ} (h : 19 * n ≡ 546 [MOD 13]) : n = 11 := by
  sorry

end smallest_positive_integer_modulo_l191_191390


namespace complementary_angle_decrease_l191_191508

theorem complementary_angle_decrease (angle1 angle2 : ℝ) (h1 : angle1 + angle2 = 90)
  (h2 : angle1 / 3 = angle2 / 6) (h3 : ∃ x: ℝ, x = 0.2) :
  let new_angle1 := angle1 * 1.2 in 
  let new_angle2 := 90 - new_angle1 in
  (new_angle2 - angle2) / angle2 = -0.10 := sorry

end complementary_angle_decrease_l191_191508


namespace first_dog_walks_two_miles_per_day_l191_191979

variable (x : ℝ)

theorem first_dog_walks_two_miles_per_day  
  (h1 : 7 * x + 56 = 70) : 
  x = 2 := 
by 
  sorry

end first_dog_walks_two_miles_per_day_l191_191979


namespace satisfactory_fraction_l191_191268

theorem satisfactory_fraction :
  let num_students_A := 8
  let num_students_B := 7
  let num_students_C := 6
  let num_students_D := 5
  let num_students_F := 4
  let satisfactory_grades := num_students_A + num_students_B + num_students_C
  let total_students := num_students_A + num_students_B + num_students_C + num_students_D + num_students_F
  satisfactory_grades / total_students = 7 / 10 :=
by
  let num_students_A := 8
  let num_students_B := 7
  let num_students_C := 6
  let num_students_D := 5
  let num_students_F := 4
  let satisfactory_grades := num_students_A + num_students_B + num_students_C
  let total_students := num_students_A + num_students_B + num_students_C + num_students_D + num_students_F
  have h1: satisfactory_grades = 21 := by sorry
  have h2: total_students = 30 := by sorry
  have fraction := (satisfactory_grades: ℚ) / total_students
  have simplified_fraction := fraction = 7 / 10
  exact sorry

end satisfactory_fraction_l191_191268


namespace inequality_hold_l191_191596

theorem inequality_hold {a b : ℝ} (h : a < b) : -3 * a > -3 * b :=
sorry

end inequality_hold_l191_191596


namespace average_infection_rate_l191_191132

theorem average_infection_rate (x : ℝ) : 
  (1 + x + x * (1 + x) = 196) → x = 13 :=
by
  intro h
  sorry

end average_infection_rate_l191_191132


namespace inequality_solution_set_l191_191431

theorem inequality_solution_set :
  ∀ x : ℝ, (1 / (x^2 + 1) > 5 / x + 21 / 10) ↔ x ∈ Set.Ioo (-2 : ℝ) (0 : ℝ) :=
by
  sorry

end inequality_solution_set_l191_191431


namespace square_of_binomial_l191_191644

theorem square_of_binomial (k : ℝ) : (∃ b : ℝ, x^2 - 20 * x + k = (x + b)^2) -> k = 100 :=
sorry

end square_of_binomial_l191_191644


namespace combination_sum_l191_191398

-- Definition of combination, also known as binomial coefficient
def combination (n k : ℕ) : ℕ :=
  Nat.choose n k

-- Theorem statement
theorem combination_sum :
  (combination 8 2) + (combination 8 3) = 84 :=
by
  sorry

end combination_sum_l191_191398


namespace project_selection_probability_l191_191695

/-- Each employee can randomly select one project from four optional assessment projects. -/
def employees : ℕ := 4

def projects : ℕ := 4

def total_events (e : ℕ) (p : ℕ) : ℕ := p^e

def choose_exactly_one_project_not_selected_probability (e : ℕ) (p : ℕ) : ℚ :=
  (Nat.choose p 2 * Nat.factorial 3) / (p^e : ℚ)

theorem project_selection_probability :
  choose_exactly_one_project_not_selected_probability employees projects = 9 / 16 :=
by
  sorry

end project_selection_probability_l191_191695


namespace solve_for_y_l191_191862

theorem solve_for_y (x y : ℝ) (h₁ : x^(2 * y) = 64) (h₂ : x = 8) : y = 1 :=
by
  sorry

end solve_for_y_l191_191862


namespace max_area_of_rectangle_l191_191874

-- Define the parameters and the problem
def perimeter := 150
def half_perimeter := perimeter / 2

theorem max_area_of_rectangle (x : ℕ) (y : ℕ) 
  (h1 : x + y = half_perimeter)
  (h2 : x > 0) (h3 : y > 0) :
  (∃ x y, x * y ≤ 1406) := 
sorry

end max_area_of_rectangle_l191_191874


namespace sector_angle_sector_max_area_l191_191123

-- Part (1)
theorem sector_angle (r l : ℝ) (α : ℝ) :
  2 * r + l = 10 → (1 / 2) * l * r = 4 → α = l / r → α = 1 / 2 :=
by
  intro h1 h2 h3
  sorry

-- Part (2)
theorem sector_max_area (r l : ℝ) (α S : ℝ) :
  2 * r + l = 40 → α = l / r → S = (1 / 2) * l * r →
  (∀ r' l' α' S', 2 * r' + l' = 40 → α' = l' / r' → S' = (1 / 2) * l' * r' → S ≤ S') →
  r = 10 ∧ α = 2 ∧ S = 100 :=
by
  intro h1 h2 h3 h4
  sorry

end sector_angle_sector_max_area_l191_191123


namespace smallest_n_divisible_by_125000_l191_191880

noncomputable def geometric_term_at (a r : ℚ) (n : ℕ) : ℚ :=
  a * r^(n-1)

noncomputable def first_term : ℚ := 5 / 8
noncomputable def second_term : ℚ := 25
noncomputable def common_ratio : ℚ := second_term / first_term

theorem smallest_n_divisible_by_125000 :
  ∃ n : ℕ, n ≥ 7 ∧ geometric_term_at first_term common_ratio n % 125000 = 0 :=
by
  sorry

end smallest_n_divisible_by_125000_l191_191880


namespace mayor_cup_num_teams_l191_191515

theorem mayor_cup_num_teams (x : ℕ) (h : x * (x - 1) / 2 = 21) : 
    ∃ x, x * (x - 1) / 2 = 21 := 
by
  sorry

end mayor_cup_num_teams_l191_191515


namespace number_of_ordered_triples_l191_191856

theorem number_of_ordered_triples :
  ∃ (S : Finset (ℕ × ℕ × ℕ)), 
    (∀ (x y z : ℕ), (x, y, z) ∈ S → Nat.lcm x y = 84 ∧ Nat.lcm x z = 480 ∧ Nat.lcm y z = 630) ∧ 
    S.card = 6 :=
by
  sorry

end number_of_ordered_triples_l191_191856


namespace expression_evaluation_l191_191296

theorem expression_evaluation (a b c : ℤ) 
  (h1 : c = a + 8) 
  (h2 : b = a + 4) 
  (h3 : a = 5) 
  (h4 : a + 2 ≠ 0) 
  (h5 : b - 3 ≠ 0) 
  (h6 : c + 7 ≠ 0) : 
  (a + 3) / (a + 2) * (b - 2) / (b - 3) * (c + 10) / (c + 7) = 23/15 :=
by
  sorry

end expression_evaluation_l191_191296


namespace bottles_not_placed_in_crate_l191_191606

-- Defining the constants based on the conditions
def bottles_per_crate : Nat := 12
def total_bottles : Nat := 130
def crates : Nat := 10

-- Theorem statement based on the question and the correct answer
theorem bottles_not_placed_in_crate :
  total_bottles - (bottles_per_crate * crates) = 10 :=
by
  -- Proof will be here
  sorry

end bottles_not_placed_in_crate_l191_191606


namespace fg_of_3_is_2810_l191_191186

def f (x : ℕ) : ℕ := x^2 + 1
def g (x : ℕ) : ℕ := 2 * x^3 - 1

theorem fg_of_3_is_2810 : f (g 3) = 2810 := by
  sorry

end fg_of_3_is_2810_l191_191186


namespace certain_number_exists_l191_191642

theorem certain_number_exists :
  ∃ x : ℤ, 55 * x % 7 = 6 ∧ x % 7 = 1 := by
  sorry

end certain_number_exists_l191_191642


namespace calculate_enclosed_area_l191_191282

open Real

noncomputable def enclosed_area_parametric_curve_line : ℝ :=
2 * sqrt 3

theorem calculate_enclosed_area : 
  let parametric_x := λ t : ℝ, 2 * (t - sin t),
      parametric_y := λ t : ℝ, 2 * (1 - cos t),
      line_y := 3 in
      (∫ t in (2 * π / 3)..(5 * π / 3), 
          ((parametric_x t * parametric_y' t) - 0) + 
          ((line_y - 0) * ((parametric_x (t + π) - parametric_x t)))
      ) = 2 * sqrt 3 :=
by
  sorry

end calculate_enclosed_area_l191_191282


namespace field_trip_buses_needed_l191_191365

def fifth_graders : Nat := 109
def sixth_graders : Nat := 115
def seventh_graders : Nat := 118
def teachers_per_grade : Nat := 4
def parents_per_grade : Nat := 2
def total_grades : Nat := 3
def seats_per_bus : Nat := 72

def total_students : Nat := fifth_graders + sixth_graders + seventh_graders
def total_chaperones : Nat := (teachers_per_grade + parents_per_grade) * total_grades
def total_people : Nat := total_students + total_chaperones
def buses_needed : Nat := (total_people + seats_per_bus - 1) / seats_per_bus  -- ceiling division

theorem field_trip_buses_needed : buses_needed = 5 := by
  sorry

end field_trip_buses_needed_l191_191365


namespace smallest_ratio_l191_191755

-- Define the system of equations as conditions
def eq1 (x y : ℝ) := x^3 + 3 * y^3 = 11
def eq2 (x y : ℝ) := (x^2 * y) + (x * y^2) = 6

-- Define the goal: proving the smallest value of x/y for the solutions (x, y) is -1.31
theorem smallest_ratio (x y : ℝ) (h1 : eq1 x y) (h2 : eq2 x y) :
  ∃ t : ℝ, t = x / y ∧ ∀ t', t' = x / y → t' ≥ -1.31 :=
sorry

end smallest_ratio_l191_191755


namespace square_of_binomial_l191_191654

theorem square_of_binomial (k : ℝ) : (∃ a : ℝ, x^2 - 20 * x + k = (x - a)^2) → k = 100 :=
by {
  sorry
}

end square_of_binomial_l191_191654


namespace contractor_absent_days_l191_191526

-- Definition of conditions
def total_days : ℕ := 30
def payment_per_work_day : ℝ := 25
def fine_per_absent_day : ℝ := 7.5
def total_payment : ℝ := 490

-- The proof statement
theorem contractor_absent_days : ∃ y : ℕ, (∃ x : ℕ, x + y = total_days ∧ payment_per_work_day * (x : ℝ) - fine_per_absent_day * (y : ℝ) = total_payment) ∧ y = 8 := 
by 
  sorry

end contractor_absent_days_l191_191526


namespace equilateral_triangle_ratio_is_correct_l191_191929

noncomputable def equilateral_triangle_area_perimeter_ratio (a : ℝ) (h_eq : a = 10) : ℝ :=
  let altitude := (Real.sqrt 3 / 2) * a
  let area := (1 / 2) * a * altitude
  let perimeter := 3 * a
  area / perimeter

theorem equilateral_triangle_ratio_is_correct :
  equilateral_triangle_area_perimeter_ratio 10 (by rfl) = 5 * Real.sqrt 3 / 6 :=
by
  sorry

end equilateral_triangle_ratio_is_correct_l191_191929


namespace part1_part2_l191_191586

theorem part1 (a : ℝ) (f : ℝ → ℝ) (h_f : ∀ x, f x = |2 * x - a| + |x - 1|) :
  (∀ x, f x + |x - 1| ≥ 2) → (a ≤ 0 ∨ a ≥ 4) :=
by sorry

theorem part2 (a : ℝ) (f : ℝ → ℝ) (h_a : a < 2) (h_f : ∀ x, f x = |2 * x - a| + |x - 1|) :
  (∀ x, f x ≥ a - 1) → (a = 4 / 3) :=
by sorry

end part1_part2_l191_191586


namespace angle_B_value_l191_191725

theorem angle_B_value (a b c A B : ℝ) (h1 : Real.sqrt 3 * a = 2 * b * Real.sin A) : 
  Real.sin B = Real.sqrt 3 / 2 ↔ (B = Real.pi / 3 ∨ B = 2 * Real.pi / 3) :=
by sorry

noncomputable def find_b_value (a : ℝ) (area : ℝ) (A B c : ℝ) (h1 : a = 6) (h2 : area = 6 * Real.sqrt 3) (h3 : c = 4) (h4 : B = Real.pi / 3 ∨ B = 2 * Real.pi / 3) : 
  ℝ := 
if B = Real.pi / 3 then 2 * Real.sqrt 7 else Real.sqrt 76

end angle_B_value_l191_191725


namespace minimum_value_y_l191_191313

theorem minimum_value_y (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a + b = 2) :
  (∀ x : ℝ, x = (1 / a + 4 / b) → x ≥ 9 / 2) :=
sorry

end minimum_value_y_l191_191313


namespace semifinalists_not_advance_l191_191056

theorem semifinalists_not_advance
  (s : ℕ) (medals : ℕ) (groups : ℕ)
  (h_s : s = 8)
  (h_medals : medals = 3)
  (h_groups : groups = 56) :
  ∑ n in Ico 1 s.succ, if (∑ k in Ico 1 n, k = groups) && (medals = 3) then some 0 else none = some 0 :=
by
  sorry

end semifinalists_not_advance_l191_191056


namespace problem_statement_l191_191219

variable {f : ℝ → ℝ}
variable {a : ℝ}

def odd_function (f : ℝ → ℝ) :=
  ∀ x, f (-x) = -f x

def periodic_function (f : ℝ → ℝ) (p : ℝ) :=
  ∀ x, f (x + p) = f x

theorem problem_statement
  (h_odd : odd_function f)
  (h_periodic : periodic_function f 3)
  (h_f1 : f 1 < 1)
  (h_f2 : f 2 = a) :
  -1 < a ∧ a < 2 :=
sorry

end problem_statement_l191_191219


namespace total_rocks_is_300_l191_191865

-- Definitions of rock types in Cliff's collection
variables (I S M : ℕ) -- I: number of igneous rocks, S: number of sedimentary rocks, M: number of metamorphic rocks
variables (shinyI shinyS shinyM : ℕ) -- shinyI: shiny igneous rocks, shinyS: shiny sedimentary rocks, shinyM: shiny metamorphic rocks

-- Given conditions
def igneous_one_third_shiny (I shinyI : ℕ) := 2 * shinyI = 3 * I
def sedimentary_two_ig_as_sed (S I : ℕ) := S = 2 * I
def metamorphic_twice_as_ig (M I : ℕ) := M = 2 * I
def shiny_igneous_is_40 (shinyI : ℕ) := shinyI = 40
def one_fifth_sed_shiny (S shinyS : ℕ) := 5 * shinyS = S
def three_quarters_met_shiny (M shinyM : ℕ) := 4 * shinyM = 3 * M

-- Theorem statement
theorem total_rocks_is_300 (I S M shinyI shinyS shinyM : ℕ)
  (h1 : igneous_one_third_shiny I shinyI)
  (h2 : sedimentary_two_ig_as_sed S I)
  (h3 : metamorphic_twice_as_ig M I)
  (h4 : shiny_igneous_is_40 shinyI)
  (h5 : one_fifth_sed_shiny S shinyS)
  (h6 : three_quarters_met_shiny M shinyM) :
  (I + S + M) = 300 :=
sorry -- Proof to be completed

end total_rocks_is_300_l191_191865


namespace factor_expression_l191_191564

theorem factor_expression (c : ℝ) : 180 * c ^ 2 + 36 * c = 36 * c * (5 * c + 1) := 
by
  sorry

end factor_expression_l191_191564


namespace product_of_k_values_l191_191482

theorem product_of_k_values (a b c k : ℝ) (h_distinct : a ≠ b ∧ b ≠ c ∧ c ≠ a) (h_eq : a / (1 + b) = k ∧ b / (1 + c) = k ∧ c / (1 + a) = k) : k = -1 :=
by
  sorry

end product_of_k_values_l191_191482


namespace probability_at_least_6_heads_l191_191264

open Finset

noncomputable def binom (n k : ℕ) : ℕ := (finset.range (k + 1)).sum (λ i, if i.choose k = 0 then 0 else n.choose k)

theorem probability_at_least_6_heads : 
  (finset.sum (finset.range 10) (λ k, if k >= 6 then (nat.choose 9 k : ℚ) else 0)) / 2^9 = (130 : ℚ) / 512 :=
by sorry

end probability_at_least_6_heads_l191_191264


namespace circle_area_l191_191600

theorem circle_area (r : ℝ) (h : 3 * (1 / (2 * π * r)) = r) : 
  π * r^2 = 3 / 2 :=
by
  sorry

end circle_area_l191_191600


namespace trapezium_other_side_length_l191_191785

theorem trapezium_other_side_length (a h Area : ℕ) (a_eq : a = 4) (h_eq : h = 6) (Area_eq : Area = 27) : 
  ∃ (b : ℕ), b = 5 := 
by
  sorry

end trapezium_other_side_length_l191_191785


namespace hiking_trip_time_l191_191812

noncomputable def R_up : ℝ := 7
noncomputable def R_down : ℝ := 1.5 * R_up
noncomputable def Distance_down : ℝ := 21
noncomputable def T_down : ℝ := Distance_down / R_down
noncomputable def T_up : ℝ := T_down

theorem hiking_trip_time :
  T_up = 2 := by
      sorry

end hiking_trip_time_l191_191812


namespace sum_of_fourth_powers_l191_191125

theorem sum_of_fourth_powers (a b c : ℝ) (h1 : a + b + c = 0) (h2 : a^2 + b^2 + c^2 = 1) :
  a^4 + b^4 + c^4 = 1 / 2 :=
by sorry

end sum_of_fourth_powers_l191_191125


namespace polygon_triangle_existence_l191_191528

theorem polygon_triangle_existence (n : ℕ) (h₁ : n > 1)
  (h₂ : ∀ (k₁ k₂ : ℕ), k₁ ≠ k₂ → (4 ≤ k₁) → (4 ≤ k₂) → k₁ ≠ k₂) :
  ∃ k, k = 3 :=
by
  sorry

end polygon_triangle_existence_l191_191528


namespace gwen_money_remaining_l191_191570

def gwen_money (initial : ℝ) (spent1 : ℝ) (earned : ℝ) (spent2 : ℝ) : ℝ :=
  initial - spent1 + earned - spent2

theorem gwen_money_remaining :
  gwen_money 5 3.25 1.5 0.7 = 2.55 :=
by
  sorry

end gwen_money_remaining_l191_191570


namespace identify_7_real_coins_l191_191060

theorem identify_7_real_coins (coins : Fin 63 → ℝ) (fakes : Finset (Fin 63)) (h_fakes_count : fakes.card = 7) (real_weight fake_weight : ℝ)
  (h_weights : ∀ i, i ∉ fakes → coins i = real_weight) (h_fake_weights : ∀ i, i ∈ fakes → coins i = fake_weight) (h_lighter : fake_weight < real_weight) :
  ∃ real_coins : Finset (Fin 63), real_coins.card = 7 ∧ (∀ i, i ∈ real_coins → coins i = real_weight) :=
sorry

end identify_7_real_coins_l191_191060


namespace arithmetic_sequence_sum_l191_191478

noncomputable def arithmetic_sequence (a d : ℝ) (n : ℕ) : ℝ := a + (n - 1) * d

noncomputable def sum_arithmetic_sequence (a d : ℝ) (n : ℕ) : ℝ := n * (a + (a + (n - 1) * d)) / 2

theorem arithmetic_sequence_sum:
  ∀ (a₃ a₅ : ℝ), a₃ = 5 → a₅ = 9 →
  ∃ (d a₁ a₇ : ℝ), arithmetic_sequence a₁ d 3 = a₃ ∧ arithmetic_sequence a₁ d 5 = a₅ ∧ a₁ + a₇ = 14 ∧
  sum_arithmetic_sequence a₁ d 7 = 49 :=
by
  intros a₃ a₅ h₁ h₂
  sorry

end arithmetic_sequence_sum_l191_191478


namespace wives_identification_l191_191340

theorem wives_identification (Anna Betty Carol Dorothy MrBrown MrGreen MrWhite MrSmith : ℕ):
  Anna = 2 ∧ Betty = 3 ∧ Carol = 4 ∧ Dorothy = 5 ∧
  (MrBrown = Dorothy ∧ MrGreen = 2 * Carol ∧ MrWhite = 3 * Betty ∧ MrSmith = 4 * Anna) ∧
  (Anna + Betty + Carol + Dorothy + MrBrown + MrGreen + MrWhite + MrSmith = 44) →
  (
    Dorothy = 5 ∧
    Carol = 4 ∧
    Betty = 3 ∧
    Anna = 2 ∧
    MrBrown = 5 ∧
    MrGreen = 8 ∧
    MrWhite = 9 ∧
    MrSmith = 8
  ) :=
by
  intros
  sorry

end wives_identification_l191_191340


namespace compare_magnitudes_l191_191258

theorem compare_magnitudes (a b c d e : ℝ) (h₁ : a > b) (h₂ : b > 0) (h₃ : c < d) (h₄ : d < 0) (h₅ : e < 0) :
  (e / (a - c)) > (e / (b - d)) :=
  sorry

end compare_magnitudes_l191_191258


namespace friend_c_spent_26_l191_191658

theorem friend_c_spent_26 :
  let you_spent := 12
  let friend_a_spent := you_spent + 4
  let friend_b_spent := friend_a_spent - 3
  let friend_c_spent := friend_b_spent * 2
  friend_c_spent = 26 :=
by
  sorry

end friend_c_spent_26_l191_191658


namespace find_marks_in_english_l191_191560

theorem find_marks_in_english 
    (avg : ℕ) (math_marks : ℕ) (physics_marks : ℕ) (chemistry_marks : ℕ) (biology_marks : ℕ) (total_subjects : ℕ)
    (avg_eq : avg = 78) 
    (math_eq : math_marks = 65) 
    (physics_eq : physics_marks = 82) 
    (chemistry_eq : chemistry_marks = 67) 
    (biology_eq : biology_marks = 85) 
    (subjects_eq : total_subjects = 5) : 
    math_marks + physics_marks + chemistry_marks + biology_marks + E = 78 * 5 → 
    E = 91 :=
by sorry

end find_marks_in_english_l191_191560


namespace no_integers_satisfying_polynomials_l191_191741

theorem no_integers_satisfying_polynomials 
: ¬ ∃ (a b c d : ℤ), a * 19^3 + b * 19^2 + c * 19 + d = 1 ∧ a * 62^3 + b * 62^2 + c * 62 + d = 2 := 
by
  sorry

end no_integers_satisfying_polynomials_l191_191741


namespace intercepted_segments_length_l191_191920

theorem intercepted_segments_length {a b c x : ℝ} 
  (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) 
  (h4 : x = a * b * c / (a * b + b * c + c * a)) : 
  x = a * b * c / (a * b + b * c + c * a) :=
by sorry

end intercepted_segments_length_l191_191920


namespace minimum_value_of_sum_of_squares_l191_191460

theorem minimum_value_of_sum_of_squares (x y z : ℝ) (h : 4 * x + 3 * y + 12 * z = 1) : 
  x^2 + y^2 + z^2 ≥ 1 / 169 :=
by
  sorry

end minimum_value_of_sum_of_squares_l191_191460


namespace ratio_Jane_to_John_l191_191215

-- Define the conditions as given in the problem.
variable (J N : ℕ) -- total products inspected by John and Jane
variable (rJ rN rT : ℚ) -- rejection rates for John, Jane, and total

-- Setting up the provided conditions
axiom h1 : rJ = 0.005 -- John rejected 0.5% of the products he inspected
axiom h2 : rN = 0.007 -- Jane rejected 0.7% of the products she inspected
axiom h3 : rT = 0.0075 -- 0.75% of the total products were rejected

-- Prove the ratio of products inspected by Jane to products inspected by John is 5
theorem ratio_Jane_to_John : (rJ * J + rN * N) = rT * (J + N) → N = 5 * J :=
by 
  sorry

end ratio_Jane_to_John_l191_191215


namespace total_flowers_l191_191882

-- Definition of conditions
def minyoung_flowers : ℕ := 24
def yoojung_flowers (y : ℕ) : Prop := minyoung_flowers = 4 * y

-- Theorem statement
theorem total_flowers (y : ℕ) (h : yoojung_flowers y) : minyoung_flowers + y = 30 :=
by sorry

end total_flowers_l191_191882


namespace perpendicular_lines_l191_191951

-- Definitions of conditions
def condition1 (α β γ δ : ℝ) : Prop := α = 90 ∧ α + β = 180 ∧ α + γ = 180 ∧ α + δ = 180
def condition2 (α β γ δ : ℝ) : Prop := α = β ∧ β = γ ∧ γ = δ
def condition3 (α β : ℝ) : Prop := α = β ∧ α + β = 180
def condition4 (α β : ℝ) : Prop := α = β ∧ α + β = 180

-- Main theorem statement
theorem perpendicular_lines (α β γ δ : ℝ) :
  (condition1 α β γ δ ∨ condition2 α β γ δ ∨
   condition3 α β ∨ condition4 α β) → α = 90 :=
by sorry

end perpendicular_lines_l191_191951


namespace joggers_difference_l191_191543

-- Define the conditions as per the problem statement
variables (Tyson Alexander Christopher : ℕ)
variable (H1 : Alexander = Tyson + 22)
variable (H2 : Christopher = 20 * Tyson)
variable (H3 : Christopher = 80)

-- The theorem statement to prove Christopher bought 54 more joggers than Alexander
theorem joggers_difference : (Christopher - Alexander) = 54 :=
  sorry

end joggers_difference_l191_191543


namespace solve_inequality_l191_191620

theorem solve_inequality (a : ℝ) : 
  (a > 0 → {x : ℝ | x < -a / 4 ∨ x > a / 3 } = {x : ℝ | 12 * x^2 - a * x - a^2 > 0}) ∧ 
  (a = 0 → {x : ℝ | x ≠ 0} = {x : ℝ | 12 * x^2 - a * x - a^2 > 0}) ∧ 
  (a < 0 → {x : ℝ | x < a / 3 ∨ x > -a / 4} = {x : ℝ | 12 * x^2 - a * x - a^2 > 0}) :=
sorry

end solve_inequality_l191_191620


namespace alice_average_speed_l191_191143

/-- Alice cycled 40 miles at 8 miles per hour and 20 miles at 40 miles per hour. 
    The average speed for the entire trip --/
theorem alice_average_speed :
  let distance1 := 40
  let speed1 := 8
  let distance2 := 20
  let speed2 := 40
  let total_distance := distance1 + distance2
  let time1 := distance1 / speed1
  let time2 := distance2 / speed2
  let total_time := time1 + time2
  (total_distance / total_time) = (120 / 11) := 
by
  sorry -- proof steps would go here

end alice_average_speed_l191_191143


namespace solve_equation_l191_191162

theorem solve_equation (a b n : ℕ) (p : ℕ) [hp : Fact (Nat.Prime p)] :
  (a > 0) → (b > 0) → (n > 0) → (a ^ 2013 + b ^ 2013 = p ^ n) ↔ 
  ∃ k : ℕ, a = 2 ^ k ∧ b = 2 ^ k ∧ p = 2 ∧ n = 2013 * k + 1 :=
by
  sorry

end solve_equation_l191_191162


namespace proof_4_minus_a_l191_191043

theorem proof_4_minus_a :
  ∀ (a b : ℚ),
    (5 + a = 7 - b) →
    (3 + b = 8 + a) →
    4 - a = 11 / 2 :=
by
  intros a b h1 h2
  sorry

end proof_4_minus_a_l191_191043


namespace knights_in_room_l191_191351

noncomputable def number_of_knights (n : ℕ) : ℕ :=
  if n = 15 then 9 else 0

theorem knights_in_room : ∀ (n : ℕ), 
  (n = 15 ∧ 
  (∀ (k l : ℕ), k + l = n ∧ k ≥ 8 ∧ l ≥ 6 → k = 9)) :=
begin
  intro n,
  split,
  { -- prove n = 15
    sorry,
  },
  { -- prove the number of knights k is 9 when conditions are met
    intros k l h,
    sorry
  }
end

end knights_in_room_l191_191351


namespace second_term_is_correct_l191_191630

noncomputable def arithmetic_sequence_second_term (a d : ℤ) (h1 : a + 9 * d = 15) (h2 : a + 10 * d = 18) : ℤ :=
  a + d

theorem second_term_is_correct (a d : ℤ) (h1 : a + 9 * d = 15) (h2 : a + 10 * d = 18) :
  arithmetic_sequence_second_term a d h1 h2 = -9 :=
sorry

end second_term_is_correct_l191_191630


namespace strawberry_jelly_amount_l191_191887

def totalJelly : ℕ := 6310
def blueberryJelly : ℕ := 4518
def strawberryJelly : ℕ := totalJelly - blueberryJelly

theorem strawberry_jelly_amount : strawberryJelly = 1792 := by
  rfl

end strawberry_jelly_amount_l191_191887


namespace polynomial_at_1_gcd_of_72_120_168_l191_191664

-- Define the polynomial function
def polynomial (x : ℤ) : ℤ := 5 * x^5 + 4 * x^4 + 3 * x^3 + 2 * x^2 + x - 6

-- Assertion that the polynomial evaluated at x = 1 gives 9
theorem polynomial_at_1 : polynomial 1 = 9 := by
  -- Usually, this is where the detailed Horner's method proof would go
  sorry

-- Define the gcd function for three numbers
def gcd3 (a b c : ℤ) : ℤ := Int.gcd (Int.gcd a b) c

-- Assertion that the GCD of 72, 120, and 168 is 24
theorem gcd_of_72_120_168 : gcd3 72 120 168 = 24 := by
  -- Usually, this is where the detailed Euclidean algorithm proof would go
  sorry

end polynomial_at_1_gcd_of_72_120_168_l191_191664


namespace fireworks_number_l191_191819

variable (x : ℕ)
variable (fireworks_total : ℕ := 484)
variable (happy_new_year_fireworks : ℕ := 12 * 5)
variable (boxes_of_fireworks : ℕ := 50 * 8)
variable (year_fireworks : ℕ := 4 * x)

theorem fireworks_number :
    4 * x + happy_new_year_fireworks + boxes_of_fireworks = fireworks_total →
    x = 6 := 
by
  sorry

end fireworks_number_l191_191819


namespace greatest_prime_factor_391_l191_191389

theorem greatest_prime_factor_391 : 
  greatestPrimeFactor 391 = 23 :=
sorry

end greatest_prime_factor_391_l191_191389


namespace sabrina_total_leaves_l191_191356

-- Definitions based on conditions
def basil_leaves := 12
def twice_the_sage_leaves (sages : ℕ) := 2 * sages = basil_leaves
def five_fewer_than_verbena (sages verbenas : ℕ) := sages + 5 = verbenas

-- Statement to prove
theorem sabrina_total_leaves (sages verbenas : ℕ) 
    (h1 : twice_the_sage_leaves sages) 
    (h2 : five_fewer_than_verbena sages verbenas) :
    basil_leaves + sages + verbenas = 29 :=
sorry

end sabrina_total_leaves_l191_191356


namespace intersection_count_l191_191684

theorem intersection_count :
  ∀ {x y : ℝ}, (2 * x - 2 * y + 4 = 0 ∨ 6 * x + 2 * y - 8 = 0) ∧ (y = -x^2 + 2 ∨ 4 * x - 10 * y + 14 = 0) → 
  (x ≠ 0 ∨ y ≠ 2) ∧ (x ≠ -1 ∨ y ≠ 1) ∧ (x ≠ 1 ∨ y ≠ -1) ∧ (x ≠ 2 ∨ y ≠ 2) → 
  ∃! (p : ℝ × ℝ), (p = (0, 2) ∨ p = (-1, 1) ∨ p = (1, -1) ∨ p = (2, 2)) := sorry

end intersection_count_l191_191684


namespace binomial_12_3_equals_220_l191_191985

theorem binomial_12_3_equals_220 : Nat.choose 12 3 = 220 := by
  sorry

end binomial_12_3_equals_220_l191_191985


namespace foil_covered_prism_width_l191_191959

def inner_prism_dimensions (l w h : ℕ) : Prop :=
  w = 2 * l ∧ w = 2 * h ∧ l * w * h = 128

def outer_prism_width (l w h outer_width : ℕ) : Prop :=
  inner_prism_dimensions l w h ∧ outer_width = w + 2

theorem foil_covered_prism_width (l w h outer_width : ℕ) (h_inner_prism : inner_prism_dimensions l w h) :
  outer_prism_width l w h outer_width → outer_width = 10 :=
by
  intro h_outer_prism
  obtain ⟨h_w_eq, h_w_eq_2, h_volume_eq⟩ := h_inner_prism
  obtain ⟨_, h_outer_width_eq⟩ := h_outer_prism
  sorry

end foil_covered_prism_width_l191_191959


namespace larger_square_uncovered_area_l191_191540

theorem larger_square_uncovered_area :
  let side_length_larger := 10
  let side_length_smaller := 4
  let area_larger := side_length_larger ^ 2
  let area_smaller := side_length_smaller ^ 2
  (area_larger - area_smaller) = 84 :=
by
  let side_length_larger := 10
  let side_length_smaller := 4
  let area_larger := side_length_larger ^ 2
  let area_smaller := side_length_smaller ^ 2
  sorry

end larger_square_uncovered_area_l191_191540


namespace pyramid_volume_l191_191269

noncomputable def volume_of_pyramid (length width alt corner_edge : ℝ) :=
  (1 / 3) * (length * width) * (Real.sqrt (corner_edge ^ 2 - (Real.sqrt (length ^ 2 + width ^ 2) / 2) ^ 2))

theorem pyramid_volume :
  volume_of_pyramid 7 9  13.87 15 ≈ 291 := 
begin
  have result : volume_of_pyramid 7 9 13.87 15 = 
    (1 / 3) * (7 * 9) * (Real.sqrt (15 ^ 2 - (Real.sqrt (7 ^ 2 + 9 ^ 2) / 2) ^ 2)),
  exact volume_of_pyramid 7 9 13.87 15,
  linarith
end

end pyramid_volume_l191_191269


namespace books_in_school_libraries_correct_l191_191777

noncomputable def booksInSchoolLibraries : ℕ :=
  let booksInPublicLibrary := 1986
  let totalBooks := 7092
  totalBooks - booksInPublicLibrary

-- Now we create a theorem to check the correctness of our definition
theorem books_in_school_libraries_correct :
  booksInSchoolLibraries = 5106 := by
  sorry -- We skip the proof, as instructed

end books_in_school_libraries_correct_l191_191777


namespace smallest_sum_of_squares_l191_191370

theorem smallest_sum_of_squares (x y : ℕ) (h : x^2 - y^2 = 145) : x^2 + y^2 = 433 :=
sorry

end smallest_sum_of_squares_l191_191370


namespace ratio_of_place_values_l191_191058

-- Definitions based on conditions
def place_value_tens_digit : ℝ := 10
def place_value_hundredths_digit : ℝ := 0.01

-- Statement to prove
theorem ratio_of_place_values :
  (place_value_tens_digit / place_value_hundredths_digit) = 1000 :=
by
  sorry

end ratio_of_place_values_l191_191058


namespace true_statements_count_l191_191274

variable {α : Type*}
variables (M N : Set α)

theorem true_statements_count :
  let P1 := (M ∩ N ⊆ N)
  let P2 := (M ∩ N ⊆ M ∪ N)
  let P3 := (M ∪ N ⊆ N)
  let P4 := (M ⊆ N → M ∩ N = M)
  P1 ∧ P2 ∧ ¬P3 ∧ P4 :=
by
  intro P1 P2 P3 P4
  split
  -- Proof steps would go here
  repeat { sorry }

end true_statements_count_l191_191274


namespace ab_value_l191_191476

variable (a b : ℝ)

theorem ab_value (h1 : a^5 * b^8 = 12) (h2 : a^8 * b^13 = 18) : a * b = 128 / 3 := 
by 
  sorry

end ab_value_l191_191476


namespace pastries_total_l191_191419

-- We start by defining the conditions
def Calvin_pastries (Frank_pastries Grace_pastries : ℕ) : ℕ := Frank_pastries + 8
def Phoebe_pastries (Frank_pastries Grace_pastries : ℕ) : ℕ := Frank_pastries + 8
def Grace_pastries : ℕ := 30
def have_same_pastries (Calvin_pastries Phoebe_pastries Grace_pastries : ℕ) : Prop :=
  Calvin_pastries + 5 = Grace_pastries ∧ Phoebe_pastries + 5 = Grace_pastries

-- Total number of pastries held by Calvin, Phoebe, Frank, and Grace
def total_pastries (Frank_pastries Calvin_pastries Phoebe_pastries Grace_pastries : ℕ) : ℕ :=
  Frank_pastries + Calvin_pastries + Phoebe_pastries + Grace_pastries

-- The statement to prove
theorem pastries_total (Frank_pastries : ℕ) : 
  have_same_pastries (Calvin_pastries Frank_pastries Grace_pastries) (Phoebe_pastries Frank_pastries Grace_pastries) Grace_pastries → 
  Frank_pastries + Calvin_pastries Frank_pastries Grace_pastries + Phoebe_pastries Frank_pastries Grace_pastries + Grace_pastries = 97 :=
by
  sorry

end pastries_total_l191_191419


namespace maximum_abc_value_l191_191422

theorem maximum_abc_value:
  (∀ (a b c : ℝ), (0 < a ∧ a < 3) ∧ (0 < b ∧ b < 3) ∧ (0 < c ∧ c < 3) ∧ (∀ x : ℝ, (x^4 + a * x^3 + b * x^2 + c * x + 1) ≠ 0) → (abc ≤ 18.75)) :=
sorry

end maximum_abc_value_l191_191422


namespace smallest_sum_of_squares_l191_191371

theorem smallest_sum_of_squares (x y : ℤ) (h : x^2 - y^2 = 145) : 
  ∃ x y, x^2 - y^2 = 145 ∧ x^2 + y^2 = 433 :=
sorry

end smallest_sum_of_squares_l191_191371


namespace non_basalt_rocks_total_eq_l191_191094

def total_rocks_in_box_A : ℕ := 57
def basalt_rocks_in_box_A : ℕ := 25

def total_rocks_in_box_B : ℕ := 49
def basalt_rocks_in_box_B : ℕ := 19

def non_basalt_rocks_in_box_A : ℕ := total_rocks_in_box_A - basalt_rocks_in_box_A
def non_basalt_rocks_in_box_B : ℕ := total_rocks_in_box_B - basalt_rocks_in_box_B

def total_non_basalt_rocks : ℕ := non_basalt_rocks_in_box_A + non_basalt_rocks_in_box_B

theorem non_basalt_rocks_total_eq : total_non_basalt_rocks = 62 := by
  -- proof goes here
  sorry

end non_basalt_rocks_total_eq_l191_191094


namespace probability_is_12_over_2907_l191_191403

noncomputable def probability_drawing_red_red_green : ℚ :=
  (3 / 19) * (2 / 18) * (4 / 17)

theorem probability_is_12_over_2907 :
  probability_drawing_red_red_green = 12 / 2907 :=
sorry

end probability_is_12_over_2907_l191_191403


namespace correct_option_c_l191_191797

-- Definitions for the problem context
noncomputable def qualification_rate : ℝ := 0.99
noncomputable def picking_probability := qualification_rate

-- The theorem statement that needs to be proven
theorem correct_option_c : picking_probability = 0.99 :=
sorry

end correct_option_c_l191_191797


namespace graph_triples_l191_191535

theorem graph_triples (V : Finset ℕ) (hV : V.card = 30)
  (h_edges : ∀ v ∈ V, ∃ (E : Finset (Finset ℕ)), E.card = 6 ∧ ∀ e ∈ E, e ⊆ V ∧ v ∉ e ∧ e.card = 2):
  ∃ m, m = 1990 :=
by
  sorry

end graph_triples_l191_191535


namespace math_problem_l191_191640

theorem math_problem :
  8 / 4 - 3^2 + 4 * 2 + (Nat.factorial 5) = 121 :=
by
  sorry

end math_problem_l191_191640


namespace age_difference_l191_191377

theorem age_difference 
  (a b : ℕ) 
  (h1 : 0 ≤ a ∧ a < 10) 
  (h2 : 0 ≤ b ∧ b < 10) 
  (h3 : 10 * a + b + 5 = 3 * (10 * b + a + 5)) : 
  (10 * a + b) - (10 * b + a) = 63 := 
by
  sorry

end age_difference_l191_191377


namespace inequality_transformation_l191_191860

theorem inequality_transformation (f : ℝ → ℝ) (a b : ℝ) (h1 : ∀ x, f x = 2 * x + 3) (h2 : a > 0) (h3 : b > 0) :
  (∀ x, |f x + 5| < a → |x + 3| < b) ↔ b ≤ a / 2 :=
sorry

end inequality_transformation_l191_191860


namespace a_plus_b_eq_three_l191_191022

noncomputable def xi : ℝ → ℝ := sorry  -- Define the discrete random variable ξ
axiom p_a : 2 / 3 = ∑' x, xi x * (if x = a then 1 else 0)
axiom p_b : 1 / 3 = ∑' x, xi x * (if x = b then 1 else 0)
axiom a_lt_b : a < b
axiom expec_xi : ∑' x, xi x * x = 4 / 3
axiom var_xi : ∑' x, xi x * (x - 4 / 3)^2 = 2 / 9

noncomputable def a : ℝ := sorry
noncomputable def b : ℝ := sorry

theorem a_plus_b_eq_three : a + b = 3 := 
sorry

end a_plus_b_eq_three_l191_191022


namespace find_cost_price_l191_191801

-- Given conditions
variables (CP SP1 SP2 : ℝ)
def condition1 : Prop := SP1 = 0.90 * CP
def condition2 : Prop := SP2 = 1.10 * CP
def condition3 : Prop := SP2 - SP1 = 500

-- Prove that CP is 2500 
theorem find_cost_price 
  (CP SP1 SP2 : ℝ)
  (h1 : condition1 CP SP1)
  (h2 : condition2 CP SP2)
  (h3 : condition3 SP1 SP2) : 
  CP = 2500 :=
sorry -- proof not required

end find_cost_price_l191_191801


namespace henry_needs_30_dollars_l191_191194

def henry_action_figures_completion (current_figures total_figures cost_per_figure : ℕ) : ℕ :=
  (total_figures - current_figures) * cost_per_figure

theorem henry_needs_30_dollars : henry_action_figures_completion 3 8 6 = 30 := by
  sorry

end henry_needs_30_dollars_l191_191194


namespace remainder_div_l191_191525

theorem remainder_div (N : ℤ) (k : ℤ) (h : N = 35 * k + 25) : N % 15 = 10 := by
  sorry

end remainder_div_l191_191525


namespace arithmetic_sequence_x_value_l191_191994

theorem arithmetic_sequence_x_value
  (x : ℝ)
  (h₁ : 2 * x - (1 / 3) = (x + 4) - 2 * x) :
  x = 13 / 3 := by
  sorry

end arithmetic_sequence_x_value_l191_191994


namespace bus_patrons_correct_l191_191690

-- Definitions corresponding to conditions
def number_of_golf_carts : ℕ := 13
def patrons_per_cart : ℕ := 3
def car_patrons : ℕ := 12

-- Multiply to get total patrons transported by golf carts
def total_patrons := number_of_golf_carts * patrons_per_cart

-- Calculate bus patrons
def bus_patrons := total_patrons - car_patrons

-- The statement to prove
theorem bus_patrons_correct : bus_patrons = 27 :=
by
  sorry

end bus_patrons_correct_l191_191690


namespace remainder_geometric_series_sum_l191_191291

/-- Define the sum of the geometric series. --/
def geometric_series_sum (n : ℕ) : ℕ :=
  (13^(n+1) - 1) / 12

/-- The given geometric series. --/
def series_sum := geometric_series_sum 1004

/-- Define the modulo operation. --/
def mod_op (a b : ℕ) := a % b

/-- The main statement to prove. --/
theorem remainder_geometric_series_sum :
  mod_op series_sum 1000 = 1 :=
sorry

end remainder_geometric_series_sum_l191_191291


namespace elroy_more_miles_l191_191158

-- Given conditions
def last_year_rate : ℝ := 4
def this_year_rate : ℝ := 2.75
def last_year_collection : ℝ := 44

-- Goals
def last_year_miles : ℝ := last_year_collection / last_year_rate
def this_year_miles : ℝ := last_year_collection / this_year_rate
def miles_difference : ℝ := this_year_miles - last_year_miles

theorem elroy_more_miles :
  miles_difference = 5 := by
  sorry

end elroy_more_miles_l191_191158


namespace books_cost_l191_191103

theorem books_cost (total_cost_three_books cost_seven_books : ℕ) 
  (h₁ : total_cost_three_books = 45)
  (h₂ : cost_seven_books = 7 * (total_cost_three_books / 3)) : 
  cost_seven_books = 105 :=
  sorry

end books_cost_l191_191103


namespace groupC_is_all_polyhedra_l191_191412

inductive GeometricBody
| TriangularPrism : GeometricBody
| QuadrangularPyramid : GeometricBody
| Sphere : GeometricBody
| Cone : GeometricBody
| Cube : GeometricBody
| TruncatedCone : GeometricBody
| HexagonalPyramid : GeometricBody
| Hemisphere : GeometricBody

def isPolyhedron : GeometricBody → Prop
| GeometricBody.TriangularPrism => true
| GeometricBody.QuadrangularPyramid => true
| GeometricBody.Sphere => false
| GeometricBody.Cone => false
| GeometricBody.Cube => true
| GeometricBody.TruncatedCone => false
| GeometricBody.HexagonalPyramid => true
| GeometricBody.Hemisphere => false

def groupA := [GeometricBody.TriangularPrism, GeometricBody.QuadrangularPyramid, GeometricBody.Sphere, GeometricBody.Cone]
def groupB := [GeometricBody.TriangularPrism, GeometricBody.QuadrangularPyramid, GeometricBody.Cube, GeometricBody.TruncatedCone]
def groupC := [GeometricBody.TriangularPrism, GeometricBody.QuadrangularPyramid, GeometricBody.Cube, GeometricBody.HexagonalPyramid]
def groupD := [GeometricBody.Cone, GeometricBody.TruncatedCone, GeometricBody.Sphere, GeometricBody.Hemisphere]

def allPolyhedra (group : List GeometricBody) : Prop :=
  ∀ b, b ∈ group → isPolyhedron b

theorem groupC_is_all_polyhedra : 
  allPolyhedra groupC ∧
  ¬ allPolyhedra groupA ∧
  ¬ allPolyhedra groupB ∧
  ¬ allPolyhedra groupD :=
by
  sorry

end groupC_is_all_polyhedra_l191_191412


namespace exist_odd_a_b_k_l191_191308

theorem exist_odd_a_b_k (m : ℤ) : 
  ∃ (a b k : ℤ), (a % 2 = 1) ∧ (b % 2 = 1) ∧ (k ≥ 0) ∧ (2 * m = a^19 + b^99 + k * 2^1999) :=
by {
  sorry
}

end exist_odd_a_b_k_l191_191308


namespace moral_of_saying_l191_191420

/-!
  Comrade Mao Zedong said: "If you want to know the taste of a pear, you must change the pear and taste it yourself." 
  Prove that this emphasizes "Practice is the source of knowledge" (option C) over the other options.
-/

def question := "What is the moral of Comrade Mao Zedong's famous saying about tasting a pear?"

def options := ["Knowledge is the driving force behind the development of practice", 
                "Knowledge guides practice", 
                "Practice is the source of knowledge", 
                "Practice has social and historical characteristics"]

def correct_answer := "Practice is the source of knowledge"

theorem moral_of_saying : (question, options[2]) ∈ [("What is the moral of Comrade Mao Zedong's famous saying about tasting a pear?", 
                                                      "Practice is the source of knowledge")] := by 
  sorry

end moral_of_saying_l191_191420


namespace complex_fraction_identity_l191_191712

theorem complex_fraction_identity (x y z a b c : ℝ)
  (h1 : x / a + y / b + z / c = 1)
  (h2 : a / x + b / y + c / z = 3) :
  x^2 / a^2 + y^2 / b^2 + z^2 / c^2 = 1 / 3 :=
by 
  sorry

end complex_fraction_identity_l191_191712


namespace sufficient_but_not_necessary_condition_l191_191701

noncomputable def f (x a : ℝ) : ℝ := (x + 1) / x + Real.sin x - a^2

theorem sufficient_but_not_necessary_condition (a : ℝ) (h : a = 1) : 
  (∀ x, f x a + f (-x) a = 0) ↔ (a = 1) ∨ (a = -1) :=
by
  sorry

end sufficient_but_not_necessary_condition_l191_191701


namespace find_cos_E_floor_l191_191055

theorem find_cos_E_floor (EF GH EH FG : ℝ) (E G : ℝ) 
  (h1 : EF = 200) 
  (h2 : GH = 200) 
  (h3 : EH ≠ FG) 
  (h4 : EF + GH + EH + FG = 800) 
  (h5 : E = G) : 
  (⌊1000 * Real.cos E⌋ = 1000) := 
by 
  sorry

end find_cos_E_floor_l191_191055


namespace proof_problem1_proof_problem2_proof_problem3_proof_problem4_l191_191183

noncomputable def f (x : ℝ) (a : ℝ) := Real.exp (2 * a * x)

noncomputable def g (x : ℝ) (k : ℝ) (b : ℝ) := k * x + b

variables {a b k x x1 x2 : ℝ}
variables (h₀ : f 1 a = Real.exp 1)
variables (h₁ : g x k b = k * x + b)
variables (h₂ : ∀ x > 0, f x a > g x k b)
variables (h₃ : Real.exp x1 = k * x1 ∧ Real.exp x2 = k * x2 ∧ x1 < x2)

axiom problem1 : a = 1 / 2
axiom problem2 : b = 0
axiom problem3 : k ∈ Set.Ioo (-Real.exp 0) (Real.exp 1)
axiom problem4 : x1 * x2 < 1

theorem proof_problem1 : f 1 a = Real.exp 1 → a = 1 / 2 :=
sorry

theorem proof_problem2 : (∀ x, g x k b = -g (-x) k b) → b = 0 :=
sorry

theorem proof_problem3 : (∀ x > 0, f x a > g x k b) → k < Real.exp 1 :=
sorry

theorem proof_problem4 : Real.exp x1 = k * x1 ∧ Real.exp x2 = k * x2 ∧ x1 < x2 → x1 * x2 < 1 :=
sorry

end proof_problem1_proof_problem2_proof_problem3_proof_problem4_l191_191183


namespace find_least_number_subtracted_l191_191300

theorem find_least_number_subtracted (n m : ℕ) (h : n = 78721) (h1 : m = 23) : (n % m) = 15 := by
  sorry

end find_least_number_subtracted_l191_191300


namespace survey_respondents_l191_191397

theorem survey_respondents (X Y : ℕ) (hX : X = 150) (hRatio : X / Y = 5) : X + Y = 180 :=
by
  sorry

end survey_respondents_l191_191397


namespace felipe_total_time_l191_191298

-- Given definitions
def combined_time_without_breaks := 126
def combined_time_with_breaks := 150
def felipe_break := 6
def emilio_break := 2 * felipe_break
def carlos_break := emilio_break / 2

theorem felipe_total_time (F E C : ℕ) 
(h1 : F = E / 2) 
(h2 : C = F + E)
(h3 : (F + E + C) = combined_time_without_breaks)
(h4 : (F + felipe_break) + (E + emilio_break) + (C + carlos_break) = combined_time_with_breaks) : 
F + felipe_break = 27 := 
sorry

end felipe_total_time_l191_191298


namespace min_value_of_xy_ratio_l191_191753

theorem min_value_of_xy_ratio :
  ∃ t : ℝ,
    (t = 2 ∨
    t = ((-1 + Real.sqrt 217) / 12) ∨
    t = ((-1 - Real.sqrt 217) / 12)) ∧
    min (min 2 ((-1 + Real.sqrt 217) / 12)) ((-1 - Real.sqrt 217) / 12) = -1.31 :=
sorry

end min_value_of_xy_ratio_l191_191753


namespace convex_polyhedron_has_triangular_face_l191_191327

def convex_polyhedron : Type := sorry -- placeholder for the type of convex polyhedra
def face (P : convex_polyhedron) : Type := sorry -- placeholder for the type of faces of a polyhedron
def vertex (P : convex_polyhedron) : Type := sorry -- placeholder for the type of vertices of a polyhedron
def edge (P : convex_polyhedron) : Type := sorry -- placeholder for the type of edges of a polyhedron

-- The number of edges meeting at a specific vertex
def vertex_degree (P : convex_polyhedron) (v : vertex P) : ℕ := sorry

-- Number of edges or vertices on a specific face
def face_sides (P : convex_polyhedron) (f : face P) : ℕ := sorry

-- A polyhedron is convex
def is_convex (P : convex_polyhedron) : Prop := sorry

-- A face is a triangle if it has 3 sides
def is_triangle (P : convex_polyhedron) (f : face P) := face_sides P f = 3

-- The problem statement in Lean 4
theorem convex_polyhedron_has_triangular_face
  (P : convex_polyhedron)
  (h1 : is_convex P)
  (h2 : ∀ v : vertex P, vertex_degree P v ≥ 4) :
  ∃ f : face P, is_triangle P f :=
sorry

end convex_polyhedron_has_triangular_face_l191_191327


namespace GCF_LCM_computation_l191_191879

-- Definitions and axioms we need
def GCF (a b : ℕ) : ℕ := Nat.gcd a b
def LCM (a b : ℕ) : ℕ := Nat.lcm a b

-- The theorem to prove
theorem GCF_LCM_computation : GCF (LCM 8 14) (LCM 7 12) = 28 :=
by sorry

end GCF_LCM_computation_l191_191879


namespace first_player_wins_game_l191_191338

theorem first_player_wins_game :
  ∀ (strip_length start_position : ℕ)
    (move : ℕ → ℕ)
    (can_move : ℕ → ℕ → Prop),
    strip_length = 2005 →
    start_position = 1003 →
    (∀ k, move k = 2^(k-1)) →
    (∀ pos k, can_move pos k ↔ pos + move k <= strip_length ∨ pos ≥ move k) →
    (∀ (start_position : ℕ) (moves : ℕ → ℕ) (player_move : ℕ → ℕ)
      (valid_move : ℕ → ℕ → Prop)
      (turns : ℕ → Prop),
      start_position = 1003 →
      ∀ n, turns n →
        (player_move n + moves n ≤ strip_length ∨ player_move n ≥ moves n) →
        game_winner = "First").

end first_player_wins_game_l191_191338


namespace frisbee_total_distance_correct_l191_191549

-- Define the conditions
def bess_distance_per_throw : ℕ := 20 * 2 -- 20 meters out and 20 meters back = 40 meters
def bess_number_of_throws : ℕ := 4
def holly_distance_per_throw : ℕ := 8
def holly_number_of_throws : ℕ := 5

-- Calculate total distances
def bess_total_distance : ℕ := bess_distance_per_throw * bess_number_of_throws
def holly_total_distance : ℕ := holly_distance_per_throw * holly_number_of_throws
def total_distance : ℕ := bess_total_distance + holly_total_distance

-- The proof statement
theorem frisbee_total_distance_correct :
  total_distance = 200 :=
by
  -- proof goes here (we use sorry to skip the proof)
  sorry

end frisbee_total_distance_correct_l191_191549


namespace compute_100p_plus_q_l191_191345

-- Given constants p, q under the provided conditions,
-- prove the result: 100p + q = 430 / 3.
theorem compute_100p_plus_q (p q : ℚ) 
  (h1 : ∀ x : ℚ, (x + p) * (x + q) * (x + 20) = 0 → x ≠ -4)
  (h2 : ∀ x : ℚ, (x + 3 * p) * (x + 4) * (x + 10) = 0 → (x = -4 ∨ x ≠ -4)) :
  100 * p + q = 430 / 3 := 
by 
  sorry

end compute_100p_plus_q_l191_191345


namespace sugar_packs_l191_191628

variable (totalSugar : ℕ) (packWeight : ℕ) (sugarLeft : ℕ)

noncomputable def numberOfPacks (totalSugar packWeight sugarLeft : ℕ) : ℕ :=
  (totalSugar - sugarLeft) / packWeight

theorem sugar_packs : numberOfPacks 3020 250 20 = 12 := by
  sorry

end sugar_packs_l191_191628


namespace unit_circle_solution_l191_191334

noncomputable def unit_circle_point_x (α : ℝ) (hα : α ∈ Set.Ioo (Real.pi / 6) (Real.pi / 2)) 
  (hcos : Real.cos (α + Real.pi / 3) = -11 / 13) : ℝ :=
  1 / 26

theorem unit_circle_solution (α : ℝ) (hα : α ∈ Set.Ioo (Real.pi / 6) (Real.pi / 2)) 
  (hcos : Real.cos (α + Real.pi / 3) = -11 / 13) :
  unit_circle_point_x α hα hcos = 1 / 26 :=
by
  sorry

end unit_circle_solution_l191_191334


namespace max_gcd_of_15m_plus_4_and_14m_plus_3_l191_191146

theorem max_gcd_of_15m_plus_4_and_14m_plus_3 (m : ℕ) (hm : 0 < m) :
  ∃ k : ℕ, k = gcd (15 * m + 4) (14 * m + 3) ∧ k = 11 :=
by {
  sorry
}

end max_gcd_of_15m_plus_4_and_14m_plus_3_l191_191146


namespace estimate_total_balls_l191_191869

theorem estimate_total_balls (red_balls : ℕ) (frequency : ℝ) (total_balls : ℕ) 
  (h_red : red_balls = 12) (h_freq : frequency = 0.6) 
  (h_eq : (red_balls : ℝ) / total_balls = frequency) : 
  total_balls = 20 :=
by
  sorry

end estimate_total_balls_l191_191869


namespace find_volume_of_sphere_l191_191409

noncomputable def volume_of_sphere (AB BC AA1 : ℝ) (hAB : AB = 2) (hBC : BC = 2) (hAA1 : AA1 = 2 * Real.sqrt 2) : ℝ :=
  let diagonal := Real.sqrt (AB^2 + BC^2 + AA1^2)
  let radius := diagonal / 2
  (4 * Real.pi * radius^3) / 3

theorem find_volume_of_sphere : volume_of_sphere 2 2 (2 * Real.sqrt 2) (by rfl) (by rfl) (by rfl) = (32 * Real.pi) / 3 :=
by
  sorry

end find_volume_of_sphere_l191_191409


namespace smallest_base_l191_191114

theorem smallest_base (b : ℕ) : (b^2 ≤ 80 ∧ 80 < b^3) → b = 5 := by
  sorry

end smallest_base_l191_191114


namespace train_passing_time_l191_191820

noncomputable def first_train_length : ℝ := 270
noncomputable def first_train_speed_kmh : ℝ := 108
noncomputable def second_train_length : ℝ := 360
noncomputable def second_train_speed_kmh : ℝ := 72

noncomputable def convert_speed_to_mps (speed_kmh : ℝ) : ℝ := speed_kmh * (1000 / 3600)

noncomputable def first_train_speed_mps : ℝ := convert_speed_to_mps first_train_speed_kmh
noncomputable def second_train_speed_mps : ℝ := convert_speed_to_mps second_train_speed_kmh

noncomputable def relative_speed_mps : ℝ := first_train_speed_mps + second_train_speed_mps
noncomputable def total_distance : ℝ := first_train_length + second_train_length
noncomputable def time_to_pass : ℝ := total_distance / relative_speed_mps

theorem train_passing_time : time_to_pass = 12.6 :=
by 
  sorry

end train_passing_time_l191_191820


namespace sector_angle_l191_191502

noncomputable def central_angle_of_sector (r l : ℝ) : ℝ := l / r

theorem sector_angle (r l : ℝ) (h1 : 2 * r + l = 6) (h2 : (1 / 2) * l * r = 2) :
  central_angle_of_sector r l = 1 ∨ central_angle_of_sector r l = 4 :=
by
  sorry

end sector_angle_l191_191502


namespace larger_cookie_raisins_l191_191238

theorem larger_cookie_raisins : ∃ n r, 5 ≤ n ∧ n ≤ 10 ∧ (n - 1) * r + (r + 1) = 100 ∧ r + 1 = 12 :=
by
  sorry

end larger_cookie_raisins_l191_191238


namespace fractions_sum_correct_l191_191554

noncomputable def fractions_sum : ℝ := (3 / 20) + (5 / 200) + (7 / 2000) + 5

theorem fractions_sum_correct : fractions_sum = 5.1785 :=
by
  sorry

end fractions_sum_correct_l191_191554


namespace rate_per_kg_of_grapes_l191_191193

theorem rate_per_kg_of_grapes : 
  ∀ (rate_per_kg_grapes : ℕ), 
    (10 * rate_per_kg_grapes + 9 * 55 = 1195) → 
    rate_per_kg_grapes = 70 := 
by
  intros rate_per_kg_grapes h
  sorry

end rate_per_kg_of_grapes_l191_191193


namespace square_of_binomial_eq_100_l191_191648

-- Given conditions
def is_square_of_binomial (p : ℝ[X]) : Prop :=
  ∃ b : ℝ, p = (X + C b)^2

-- The equivalent proof problem statement
theorem square_of_binomial_eq_100 (k : ℝ) :
  is_square_of_binomial (X^2 - 20 * X + C k) → k = 100 :=
by
  sorry

end square_of_binomial_eq_100_l191_191648


namespace island_knights_liars_two_people_l191_191490

def islanders_knights_and_liars (n : ℕ) : Prop :=
  ∃ (knight liar : ℕ),
    knight + liar = n ∧
    (∀ i : ℕ, 1 ≤ i → i ≤ n → 
      ((i % i = 0 → liar > 0 ∧ knight > 0) ∧ (i % i ≠ 0 → liar > 0)))

theorem island_knights_liars_two_people :
  islanders_knights_and_liars 2 :=
sorry

end island_knights_liars_two_people_l191_191490


namespace prime_add_eq_2001_l191_191700

theorem prime_add_eq_2001 (a b : ℕ) (ha : Nat.Prime a) (hb : Nat.Prime b) (h : a^2 + b = 2003) : a + b = 2001 :=
sorry

end prime_add_eq_2001_l191_191700


namespace pqrsum_l191_191067

-- Given constants and conditions:
variables {p q r : ℝ} -- p, q, r are real numbers
axiom Hpq : p < q -- given condition p < q
axiom Hineq : ∀ x : ℝ, (x > 5 ∨ 7 ≤ x ∧ x ≤ 15) ↔ ( (x - p) * (x - q) / (x - r) ≥ 0) -- given inequality condition

-- Values from the solution:
axiom Hp : p = 7
axiom Hq : q = 15
axiom Hr : r = 5

-- Proof statement:
theorem pqrsum : p + 2 * q + 3 * r = 52 :=
sorry 

end pqrsum_l191_191067


namespace sqrt_of_six_l191_191758

theorem sqrt_of_six : Real.sqrt 6 = Real.sqrt 6 := by
  sorry

end sqrt_of_six_l191_191758


namespace symmetric_scanning_codes_count_l191_191815

noncomputable def countSymmetricScanningCodes : ℕ :=
  let totalConfigs := 32
  let invalidConfigs := 2
  totalConfigs - invalidConfigs

theorem symmetric_scanning_codes_count :
  countSymmetricScanningCodes = 30 :=
by
  -- Here, we would detail the steps, but we omit the actual proof for now.
  sorry

end symmetric_scanning_codes_count_l191_191815


namespace statements_l191_191079

open Real EuclideanGeometry

def curve_C (a : ℝ) (h : a > 1) : Set (ℝ × ℝ) := 
  { P | (dist P (-1, 0)) * (dist P (1, 0)) = a^2 }

theorem statements (a : ℝ) (h : a > 1) :
  ¬(0, 0) ∈ curve_C a h ∧ 
  (∀ P ∈ curve_C a h, (-P.1, -P.2) ∈ curve_C a h) ∧
  (∀ P ∈ curve_C a h, (euclidean_area (triangle_mk (P, (-1, 0), (1, 0)))) ≤ (1 / 2) * a^2) ∧
  (∀ P ∈ curve_C a h, (dist P (-1, 0)) + (dist P (1, 0)) + 2 ≥ 2 * a + 2) :=
by
  sorry

end statements_l191_191079


namespace find_m_l191_191037

variables {m : ℝ}
def vec_a : ℝ × ℝ := (1, m)
def vec_b : ℝ × ℝ := (2, 5)
def vec_c : ℝ × ℝ := (m, 3)
def vec_a_plus_c := (1 + m, 3 + m)
def vec_a_minus_b := (1 - 2, m - 5)

theorem find_m (h : (1 + m) * (m - 5) = -1 * (m + 3)) : m = (3 + Real.sqrt 17) / 2 ∨ m = (3 - Real.sqrt 17) / 2 := 
sorry

end find_m_l191_191037


namespace cards_per_deck_l191_191485

theorem cards_per_deck (decks : ℕ) (cards_per_layer : ℕ) (layers : ℕ) 
  (h_decks : decks = 16) 
  (h_cards_per_layer : cards_per_layer = 26) 
  (h_layers : layers = 32) 
  (total_cards_used : ℕ := cards_per_layer * layers) 
  (number_of_cards_per_deck : ℕ := total_cards_used / decks) :
  number_of_cards_per_deck = 52 :=
by 
  sorry

end cards_per_deck_l191_191485


namespace smallest_ratio_l191_191754

-- Define the system of equations as conditions
def eq1 (x y : ℝ) := x^3 + 3 * y^3 = 11
def eq2 (x y : ℝ) := (x^2 * y) + (x * y^2) = 6

-- Define the goal: proving the smallest value of x/y for the solutions (x, y) is -1.31
theorem smallest_ratio (x y : ℝ) (h1 : eq1 x y) (h2 : eq2 x y) :
  ∃ t : ℝ, t = x / y ∧ ∀ t', t' = x / y → t' ≥ -1.31 :=
sorry

end smallest_ratio_l191_191754


namespace seven_books_cost_l191_191099

-- Given condition: Three identical books cost $45
def three_books_cost (cost_per_book : ℤ) := 3 * cost_per_book = 45

-- Question to prove: The cost of seven identical books is $105
theorem seven_books_cost (cost_per_book : ℤ) (h : three_books_cost cost_per_book) : 7 * cost_per_book = 105 := 
sorry

end seven_books_cost_l191_191099


namespace max_value_of_f_l191_191907

noncomputable def f (x : ℝ) : ℝ := Real.exp x - x

theorem max_value_of_f : ∀ (x : ℝ), x ∈ Set.Icc (-1 : ℝ) (1 : ℝ) → f x ≤ Real.exp 1 - 1 :=
by 
-- The conditions: the function and the interval
intros x hx
-- The interval condition: -1 ≤ x ≤ 1
have h_interval : -1 ≤ x ∧ x ≤ 1 := by 
  cases hx
  split; assumption
-- We prove it directly by showing the evaluated function points
sorry

end max_value_of_f_l191_191907


namespace value_of_k_l191_191651

theorem value_of_k (k : ℕ) : (∃ b : ℕ, x^2 - 20 * x + k = (x + b)^2) → k = 100 := by
  sorry

end value_of_k_l191_191651


namespace simplify_polynomial_l191_191888

theorem simplify_polynomial (x : ℝ) :
  (14 * x ^ 12 + 8 * x ^ 9 + 3 * x ^ 8) + (2 * x ^ 14 - x ^ 12 + 2 * x ^ 9 + 5 * x ^ 5 + 7 * x ^ 2 + 6) =
  2 * x ^ 14 + 13 * x ^ 12 + 10 * x ^ 9 + 3 * x ^ 8 + 5 * x ^ 5 + 7 * x ^ 2 + 6 :=
by
  sorry

end simplify_polynomial_l191_191888


namespace sum_of_numbers_l191_191949

theorem sum_of_numbers : 
  5678 + 6785 + 7856 + 8567 = 28886 := 
by 
  sorry

end sum_of_numbers_l191_191949


namespace cost_price_of_A_l191_191119

-- Assume the cost price of the bicycle for A which we need to prove
def CP_A : ℝ := 144

-- Given conditions
def profit_A_to_B (CP_A : ℝ) := 1.25 * CP_A
def profit_B_to_C (CP_B : ℝ) := 1.25 * CP_B
def SP_C := 225

-- Proof statement
theorem cost_price_of_A : 
  profit_B_to_C (profit_A_to_B CP_A) = SP_C :=
by
  sorry

end cost_price_of_A_l191_191119


namespace ab_value_l191_191209

theorem ab_value (a b : ℚ) (h1 : 3 * a - 8 = 0) (h2 : b = 3) : a * b = 8 :=
by
  sorry

end ab_value_l191_191209


namespace total_sheets_l191_191261

-- Define the conditions
def sheets_in_bundle : ℕ := 10
def bundles : ℕ := 3
def additional_sheets : ℕ := 8

-- Theorem to prove the total number of sheets Jungkook has
theorem total_sheets : bundles * sheets_in_bundle + additional_sheets = 38 := by
  sorry

end total_sheets_l191_191261


namespace no_integer_solutions_l191_191505

theorem no_integer_solutions :
  ¬ ∃ (x y z : ℤ), x^1988 + y^1988 + z^1988 = 7^1990 := by
  sorry

end no_integer_solutions_l191_191505


namespace value_of_k_l191_191652

theorem value_of_k (k : ℕ) : (∃ b : ℕ, x^2 - 20 * x + k = (x + b)^2) → k = 100 := by
  sorry

end value_of_k_l191_191652


namespace pascal_triangle_41_l191_191857

theorem pascal_triangle_41:
  ∃ (n : Nat), ∀ (k : Nat), n = 41 ∧ (Nat.choose n k = 41) :=
sorry

end pascal_triangle_41_l191_191857


namespace final_people_amount_l191_191147

def initial_people : ℕ := 250
def people_left1 : ℕ := 35
def people_joined1 : ℕ := 20
def percentage_left : ℕ := 10
def groups_joined : ℕ := 4
def group_size : ℕ := 15

theorem final_people_amount :
  let intermediate_people1 := initial_people - people_left1;
  let intermediate_people2 := intermediate_people1 + people_joined1;
  let people_left2 := (intermediate_people2 * percentage_left) / 100;
  let rounded_people_left2 := people_left2;
  let intermediate_people3 := intermediate_people2 - rounded_people_left2;
  let total_new_join := groups_joined * group_size;
  let final_people := intermediate_people3 + total_new_join;
  final_people = 272 :=
by sorry

end final_people_amount_l191_191147


namespace uniquePlantsTotal_l191_191207

-- Define the number of plants in each bed
def numPlantsInA : ℕ := 600
def numPlantsInB : ℕ := 500
def numPlantsInC : ℕ := 400

-- Define the number of shared plants between beds
def sharedPlantsAB : ℕ := 60
def sharedPlantsAC : ℕ := 120
def sharedPlantsBC : ℕ := 80
def sharedPlantsABC : ℕ := 30

-- Prove that the total number of unique plants in the garden is 1270
theorem uniquePlantsTotal : 
  numPlantsInA + numPlantsInB + numPlantsInC 
  - sharedPlantsAB - sharedPlantsAC - sharedPlantsBC 
  + sharedPlantsABC = 1270 := 
by sorry

end uniquePlantsTotal_l191_191207


namespace simplify_expression_l191_191749

noncomputable def x : ℝ := Real.sqrt 3 + 1

theorem simplify_expression (x : ℝ) (h : x = Real.sqrt 3 + 1) :
  ((1 - (x / (x + 1))) / ((x^2 - 1) / (x^2 + 2*x + 1))) = Real.sqrt 3 / 3 :=
by
  sorry

end simplify_expression_l191_191749


namespace janet_freelancer_income_difference_l191_191214

theorem janet_freelancer_income_difference :
  let hours_per_week := 40
  let current_job_hourly_rate := 30
  let freelancer_hourly_rate := 40
  let fica_taxes_per_week := 25
  let healthcare_premiums_per_month := 400
  let weeks_per_month := 4
  
  let current_job_weekly_income := hours_per_week * current_job_hourly_rate
  let current_job_monthly_income := current_job_weekly_income * weeks_per_month
  
  let freelancer_weekly_income := hours_per_week * freelancer_hourly_rate
  let freelancer_monthly_income := freelancer_weekly_income * weeks_per_month
  
  let freelancer_monthly_fica_taxes := fica_taxes_per_week * weeks_per_month
  let freelancer_total_additional_costs := freelancer_monthly_fica_taxes + healthcare_premiums_per_month
  
  let freelancer_net_monthly_income := freelancer_monthly_income - freelancer_total_additional_costs
  
  freelancer_net_monthly_income - current_job_monthly_income = 1100 :=
by
  sorry

end janet_freelancer_income_difference_l191_191214


namespace new_student_weight_l191_191256

theorem new_student_weight : 
  ∀ (w_new : ℕ), 
    (∀ (sum_weight: ℕ), 80 + sum_weight - w_new = sum_weight - 18) → 
      w_new = 62 := 
by
  intros w_new h
  sorry

end new_student_weight_l191_191256


namespace power_of_four_l191_191164

-- Definition of the conditions
def prime_factors (x: ℕ): ℕ := 2 * x + 5 + 2

-- The statement we need to prove given the conditions
theorem power_of_four (x: ℕ) (h: prime_factors x = 33) : x = 13 :=
by
  -- Proof goes here
  sorry

end power_of_four_l191_191164


namespace product_of_integers_l191_191902

theorem product_of_integers (a b : ℤ) (h_lcm : Int.lcm a b = 45) (h_gcd : Int.gcd a b = 9) : a * b = 405 :=
by
  sorry

end product_of_integers_l191_191902


namespace last_two_digits_2007_pow_20077_l191_191766

theorem last_two_digits_2007_pow_20077 : (2007 ^ 20077) % 100 = 7 := 
by sorry

end last_two_digits_2007_pow_20077_l191_191766


namespace width_decreased_by_33_percent_l191_191904

theorem width_decreased_by_33_percent {L W : ℝ} (h : L > 0 ∧ W > 0) (h_area : (1.5 * L) * W' = L * W) :
  W' = (2 / 3) * W :=
begin
  sorry -- Proof to be filled in later
end

end width_decreased_by_33_percent_l191_191904


namespace number_of_terms_in_expansion_l191_191552

def first_factor : List Char := ['x', 'y']
def second_factor : List Char := ['u', 'v', 'w', 'z', 's']

theorem number_of_terms_in_expansion :
  first_factor.length * second_factor.length = 10 :=
by
  -- Lean expects a proof here, but the problem statement specifies to use sorry to skip the proof.
  sorry

end number_of_terms_in_expansion_l191_191552


namespace range_of_k_l191_191180

theorem range_of_k (k : ℝ) : (∀ x : ℝ, x^2 - 2 * x + k^2 - 3 > 0) -> (k > 2 ∨ k < -2) :=
by
  sorry

end range_of_k_l191_191180


namespace positive_value_of_m_l191_191166

variable {m : ℝ}

theorem positive_value_of_m (h : ∃ x : ℝ, (3 * x^2 + m * x + 36) = 0 ∧ (∀ y : ℝ, (3 * y^2 + m * y + 36) = 0 → y = x)) :
  m = 12 * Real.sqrt 3 :=
sorry

end positive_value_of_m_l191_191166


namespace largest_among_abc_l191_191574

theorem largest_among_abc
  (x : ℝ) 
  (hx : 0 < x) 
  (hx1 : x < 1)
  (a : ℝ)
  (ha : a = 2 * Real.sqrt x )
  (b : ℝ)
  (hb : b = 1 + x)
  (c : ℝ)
  (hc : c = 1 / (1 - x)) 
  : a < b ∧ b < c :=
by
  sorry

end largest_among_abc_l191_191574


namespace monotonic_increasing_intervals_l191_191504

noncomputable def f (x : ℝ) : ℝ := (Real.cos (x - Real.pi / 6))^2

theorem monotonic_increasing_intervals (k : ℤ) : 
  ∃ t : Set ℝ, t = Set.Ioo (-Real.pi / 3 + k * Real.pi) (Real.pi / 6 + k * Real.pi) ∧ 
    ∀ x y, x ∈ t → y ∈ t → x ≤ y → f x ≤ f y :=
sorry

end monotonic_increasing_intervals_l191_191504


namespace solution_set_inequality_l191_191479

noncomputable def f : ℝ → ℝ := sorry
noncomputable def derivative_f : ℝ → ℝ := sorry -- f' is the derivative of f

-- Conditions
axiom f_domain {x : ℝ} (h1 : 0 < x) : f x ≠ 0
axiom derivative_condition {x : ℝ} (h1 : 0 < x) : f x + x * derivative_f x > 0
axiom initial_value : f 1 = 2

-- Proof that the solution set of the inequality f(x) < 2/x is (0, 1)
theorem solution_set_inequality : ∀ x : ℝ, 0 < x ∧ x < 1 → f x < 2 / x := sorry

end solution_set_inequality_l191_191479


namespace ratio_of_area_to_perimeter_l191_191939

noncomputable def altitude_of_equilateral_triangle (s : ℝ) : ℝ :=
  s * (Real.sqrt 3 / 2)

noncomputable def area_of_equilateral_triangle (s : ℝ) : ℝ :=
  1 / 2 * s * altitude_of_equilateral_triangle s

noncomputable def perimeter_of_equilateral_triangle (s : ℝ) : ℝ :=
  3 * s

theorem ratio_of_area_to_perimeter (s : ℝ) (h : s = 10) :
    (area_of_equilateral_triangle s) / (perimeter_of_equilateral_triangle s) = 5 * Real.sqrt 3 / 6 :=
  by
  rw [h]
  sorry

end ratio_of_area_to_perimeter_l191_191939


namespace probability_interval_l191_191019

noncomputable def normal_distribution : Type :=
{ μ := 2, σ := σ }

axiom prob_X_less_a (a : ℝ) (ha : a < 2) : 𝓝 (X < a) = 0.32

theorem probability_interval (X : normal_distribution) (a : ℝ) (ha : a < 4 - a) :
  P(a < X < 4 - a) = 0.36 :=
by
  sorry.

end probability_interval_l191_191019


namespace ratio_of_red_marbles_l191_191128

theorem ratio_of_red_marbles (total_marbles blue_marbles green_marbles yellow_marbles red_marbles : ℕ)
  (h1 : total_marbles = 164)
  (h2 : blue_marbles = total_marbles / 2)
  (h3 : green_marbles = 27)
  (h4 : yellow_marbles = 14)
  (h5 : red_marbles = total_marbles - (blue_marbles + green_marbles + yellow_marbles)) :
  (red_marbles : ℚ) / total_marbles = (1 : ℚ) / 4 :=
by {
  sorry
}

end ratio_of_red_marbles_l191_191128


namespace smallest_k_condition_exists_l191_191249

theorem smallest_k_condition_exists (k : ℕ) :
    k > 1 ∧ (k % 13 = 1) ∧ (k % 8 = 1) ∧ (k % 3 = 1) → k = 313 :=
by
  sorry

end smallest_k_condition_exists_l191_191249


namespace range_of_a_l191_191584

noncomputable def f : ℝ → ℝ := sorry

variables (a : ℝ)
variable (is_even : ∀ x : ℝ, f (x) = f (-x)) -- f is even
variable (monotonic_incr : ∀ x y : ℝ, 0 ≤ x → x ≤ y → f x ≤ f y) -- f is monotonically increasing in [0, +∞)

theorem range_of_a
  (h : f (Real.log a / Real.log 2) + f (Real.log (1/a) / Real.log 2) ≤ 2 * f 1) : 
  1 / 2 ≤ a ∧ a ≤ 2 :=
by
  sorry

end range_of_a_l191_191584


namespace lcm_ge_ten_times_first_l191_191575

open Nat

theorem lcm_ge_ten_times_first {a : ℕ → ℕ} (h : ∀ i j, i < j → a i < a j) :
    ∀ (a1 : ℕ), a1 = a 0 → (∀ i, i < 10 → a i ∈ ℕ) →
    lcm (finset.range 10).val.map a ≥ 10 * a 0 := 
by
  sorry

end lcm_ge_ten_times_first_l191_191575


namespace diane_money_l191_191837

-- Define the conditions
def total_cost : ℤ := 65
def additional_needed : ℤ := 38
def initial_amount : ℤ := total_cost - additional_needed

-- Theorem statement
theorem diane_money : initial_amount = 27 := by
  sorry

end diane_money_l191_191837


namespace complement_of_intersection_l191_191070

def S : Set ℝ := {-2, -1, 0, 1, 2}
def T : Set ℝ := {x | x + 1 ≤ 2}
def complement (A B : Set ℝ) : Set ℝ := {x ∈ B | x ∉ A}

theorem complement_of_intersection :
  complement (S ∩ T) S = {2} :=
by
  sorry

end complement_of_intersection_l191_191070


namespace pow_div_simplify_l191_191414

theorem pow_div_simplify : (((15^15) / (15^14))^3 * 3^3) / 3^3 = 3375 := by
  sorry

end pow_div_simplify_l191_191414


namespace find_f_l191_191610

-- Define the function f and its conditions
def f (x : ℝ) : ℝ := sorry

axiom f_0 : f 0 = 0
axiom f_xy (x y : ℝ) : f (x * y) = f ((x^2 + y^2) / 2) + 3 * (x - y)^2

-- Theorem to be proved
theorem find_f (x : ℝ) : f x = -6 * x + 3 :=
by sorry -- proof goes here

end find_f_l191_191610


namespace isosceles_triangle_vertex_angle_l191_191602

theorem isosceles_triangle_vertex_angle (α : ℝ) (β : ℝ) (γ : ℝ)
  (h1 : α = β)
  (h2: α = 70) 
  (h3 : α + β + γ = 180) : 
  γ = 40 :=
by {
  sorry
}

end isosceles_triangle_vertex_angle_l191_191602


namespace geometric_sequence_a4_l191_191764

theorem geometric_sequence_a4 (x a_4 : ℝ) (h1 : 2*x + 2 = (3*x + 3) * (2*x + 2) / x)
  (h2 : x = -4 ∨ x = -1) (h3 : x = -4) : a_4 = -27 / 2 :=
by
  sorry

end geometric_sequence_a4_l191_191764


namespace hotel_rooms_count_l191_191976

theorem hotel_rooms_count
  (TotalLamps : ℕ) (TotalChairs : ℕ) (TotalBedSheets : ℕ)
  (LampsPerRoom : ℕ) (ChairsPerRoom : ℕ) (BedSheetsPerRoom : ℕ) :
  TotalLamps = 147 → 
  TotalChairs = 84 → 
  TotalBedSheets = 210 → 
  LampsPerRoom = 7 → 
  ChairsPerRoom = 4 → 
  BedSheetsPerRoom = 10 →
  (TotalLamps / LampsPerRoom = 21) ∧ 
  (TotalChairs / ChairsPerRoom = 21) ∧ 
  (TotalBedSheets / BedSheetsPerRoom = 21) :=
by
  intros
  sorry

end hotel_rooms_count_l191_191976


namespace birds_in_store_l191_191538

/-- 
A pet store had a total of 180 animals, consisting of birds, dogs, and cats. 
Among the birds, 64 talked, and 13 didn't. If there were 40 dogs in the store 
and the number of birds that talked was four times the number of cats, 
prove that there were 124 birds in total.
-/
theorem birds_in_store (total_animals : ℕ) (talking_birds : ℕ) (non_talking_birds : ℕ) 
  (dogs : ℕ) (cats : ℕ) 
  (h1 : total_animals = 180)
  (h2 : talking_birds = 64)
  (h3 : non_talking_birds = 13)
  (h4 : dogs = 40)
  (h5 : talking_birds = 4 * cats) : 
  talking_birds + non_talking_birds + dogs + cats = 180 ∧ 
  talking_birds + non_talking_birds = 124 :=
by
  -- We are skipping the proof itself and focusing on the theorem statement
  sorry

end birds_in_store_l191_191538


namespace angle_sum_and_relation_l191_191106

variable {A B : ℝ}

theorem angle_sum_and_relation (h1 : A + B = 180) (h2 : A = 5 * B) : A = 150 := by
  sorry

end angle_sum_and_relation_l191_191106


namespace number_of_mappings_l191_191152

noncomputable def countMappings (n : ℕ) (X : Type) [Fintype X] [DecidableEq X]
  (f : X → X) (a : X) (h1 : n ≥ 2) 
  (h2 : ∀ x : X, f (f x) = a) (h3 : a ∈ X) : ℕ :=
∑ k in Finset.range (n - 1) \ {0}, Nat.choose (n - 1) k * k ^ (n - k - 1)

theorem number_of_mappings (n : ℕ) (X : Type) [Fintype X] [DecidableEq X] 
  (f : X → X) (a : X) (h1 : n ≥ 2)
  (h2 : ∀ x : X, f (f x) = a) (h3 : a ∈ X) :
  ∃ (k_set : Finset ℕ), k_set = Finset.range (n - 1) \ {0} 
  ∧ countMappings n X f a h1 h2 h3 = ∑ k in k_set, Nat.choose (n - 1) k * k ^ (n - k - 1) :=
sorry

end number_of_mappings_l191_191152


namespace population_growth_proof_l191_191052

noncomputable def population_growth (P0 : ℕ) (P200 : ℕ) (t : ℕ) (x : ℝ) : Prop :=
  P200 = P0 * (1 + 1 / x)^t

theorem population_growth_proof :
  population_growth 6 1000000 200 16 :=
by
  -- Proof goes here
  sorry

end population_growth_proof_l191_191052


namespace mrs_heine_dogs_treats_l191_191612

theorem mrs_heine_dogs_treats (heart_biscuits_per_dog puppy_boots_per_dog total_items : ℕ)
  (h_biscuits : heart_biscuits_per_dog = 5)
  (h_boots : puppy_boots_per_dog = 1)
  (total : total_items = 12) :
  (total_items / (heart_biscuits_per_dog + puppy_boots_per_dog)) = 2 :=
by
  sorry

end mrs_heine_dogs_treats_l191_191612


namespace solve_equation1_solve_equation2_l191_191233

-- Define the equations and the problem.
def equation1 (x : ℝ) : Prop := (3 / (x^2 - 9)) + (x / (x - 3)) = 1
def equation2 (x : ℝ) : Prop := 2 - (1 / (2 - x)) = ((3 - x) / (x - 2))

-- Proof problem for the first equation: Prove that x = -4 is the solution.
theorem solve_equation1 : ∀ x : ℝ, equation1 x → x = -4 :=
by {
  sorry
}

-- Proof problem for the second equation: Prove that there are no solutions.
theorem solve_equation2 : ∀ x : ℝ, ¬equation2 x :=
by {
  sorry
}

end solve_equation1_solve_equation2_l191_191233


namespace hyperbola_trajectory_center_l191_191292

theorem hyperbola_trajectory_center :
  ∀ m : ℝ, ∃ (x y : ℝ), x^2 - y^2 - 6 * m * x - 4 * m * y + 5 * m^2 - 1 = 0 ∧ 2 * x + 3 * y = 0 :=
by
  sorry

end hyperbola_trajectory_center_l191_191292


namespace cost_of_steel_ingot_l191_191514

theorem cost_of_steel_ingot :
  ∃ P : ℝ, 
    (∃ initial_weight : ℝ, initial_weight = 60) ∧
    (∃ weight_increase_percentage : ℝ, weight_increase_percentage = 0.6) ∧
    (∃ ingot_weight : ℝ, ingot_weight = 2) ∧
    (weight_needed = initial_weight * weight_increase_percentage) ∧
    (number_of_ingots = weight_needed / ingot_weight) ∧
    (number_of_ingots > 10) ∧
    (discount_percentage = 0.2) ∧
    (total_cost = 72) ∧
    (discounted_price_per_ingot = P * (1 - discount_percentage)) ∧
    (total_cost = discounted_price_per_ingot * number_of_ingots) ∧
    P = 5 := 
by
  sorry

end cost_of_steel_ingot_l191_191514


namespace totalPizzaEaten_l191_191083

-- Define the conditions
def rachelAte : ℕ := 598
def bellaAte : ℕ := 354

-- State the theorem
theorem totalPizzaEaten : rachelAte + bellaAte = 952 :=
by
  -- Proof omitted
  sorry

end totalPizzaEaten_l191_191083


namespace local_minimum_value_of_f_l191_191863

noncomputable def f (x : ℝ) : ℝ := x - Real.log x

theorem local_minimum_value_of_f : 
  ∃ x : ℝ, x > 0 ∧ (∀ y : ℝ, y > 0 → f y ≥ f x) ∧ f x = 1 :=
by
  sorry

end local_minimum_value_of_f_l191_191863


namespace shortest_distance_reflection_l191_191304

-- Definitions from the conditions of the problem
def point_P : ℝ × ℝ := (1, 0)
def point_Q : ℝ × ℝ := (2, 1)
def line : ℝ × ℝ → Prop := λ P, P.1 - P.2 + 1 = 0

-- Statement that we need to prove
theorem shortest_distance_reflection :
  ∃ (B : ℝ × ℝ), (B.1 + 1) * (B.1 - 1) + (B.2 - 2) * (B.2 + 2) = 0 ∧
  dist B point_Q = sqrt 10 :=
sorry

end shortest_distance_reflection_l191_191304


namespace graphene_scientific_notation_l191_191192

def scientific_notation (n : ℝ) (a : ℝ) (exp : ℤ) : Prop :=
  n = a * 10 ^ exp ∧ 1 ≤ abs a ∧ abs a < 10

theorem graphene_scientific_notation :
  scientific_notation 0.00000000034 3.4 (-10) :=
by {
  sorry
}

end graphene_scientific_notation_l191_191192


namespace calculate_distance_l191_191848

def velocity (t : ℝ) : ℝ := 3 * t^2 + t

theorem calculate_distance : ∫ t in (0 : ℝ)..(4 : ℝ), velocity t = 72 := 
by
  sorry

end calculate_distance_l191_191848


namespace simplify_expression_l191_191415

theorem simplify_expression (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) : 
  ((x + y) ^ 2 - (x - y) ^ 2) / (4 * x * y) = 1 := 
by sorry

end simplify_expression_l191_191415


namespace nature_of_angles_in_DEF_l191_191867

open Real

-- Definitions for point and triangle
structure Triangle :=
(A B C : ℝ × ℝ)

-- Definitions for angles in the plane
def angle (A B C : ℝ × ℝ) : ℝ :=
  let v := (C.1 - B.1, C.2 - B.2)
  let u := (A.1 - B.1, A.2 - B.2)
  Real.angle v u

-- Definitions for the given triangle and properties
def triangle_ABC := Triangle.mk (0, 0) (1, 0) (cos (100 * π / 180), sin (100 * π / 180))
def A := triangle_ABC.A
def B := triangle_ABC.B
def C := triangle_ABC.C

def angle_BAC := angle B A C
def angle_ABC := angle A B C

-- Points of tangency with the incircle (definitions are symbolic)
def D : ℝ × ℝ := sorry
def E : ℝ × ℝ := sorry
def F : ℝ × ℝ := sorry

-- Triangle formed by points of tangency
def triangle_DEF := Triangle.mk D E F

-- Angles in triangle DEF
def angle_DEF := angle D E F
def angle_FED := angle F E D
def angle_EFD := angle E F D

-- The proof problem statement
theorem nature_of_angles_in_DEF :
  angle_DEF < π ∧ angle_FED > π / 2 ∧ angle_EFD < π :=
sorry

end nature_of_angles_in_DEF_l191_191867


namespace find_unsuitable_activity_l191_191975

-- Definitions based on the conditions
def suitable_for_questionnaire (activity : String) : Prop :=
  activity = "D: The radiation produced by various mobile phones during use"

-- Question transformed into a statement to prove in Lean
theorem find_unsuitable_activity :
  suitable_for_questionnaire "D: The radiation produced by various mobile phones during use" :=
by
  sorry

end find_unsuitable_activity_l191_191975


namespace xy_value_l191_191199

theorem xy_value (x y : ℝ) (h : x * (x + y) = x ^ 2 + 12) : x * y = 12 :=
by {
  sorry
}

end xy_value_l191_191199


namespace equilateral_triangle_area_to_perimeter_ratio_l191_191934

theorem equilateral_triangle_area_to_perimeter_ratio
  (a : ℝ) (h : a = 10) :
  let altitude := a * (Real.sqrt 3 / 2) in
  let area := (1 / 2) * a * altitude in
  let perimeter := 3 * a in
  area / perimeter = 5 * (Real.sqrt 3) / 6 := 
by
  sorry

end equilateral_triangle_area_to_perimeter_ratio_l191_191934


namespace smallest_sum_of_squares_l191_191369

theorem smallest_sum_of_squares (x y : ℕ) (h : x^2 - y^2 = 145) : x^2 + y^2 = 433 :=
sorry

end smallest_sum_of_squares_l191_191369


namespace possible_third_side_l191_191868

theorem possible_third_side (x : ℝ) : (3 + 4 > x) ∧ (abs (4 - 3) < x) → (x = 2) :=
by 
  sorry

end possible_third_side_l191_191868


namespace todd_final_money_l191_191922

noncomputable def todd_initial_money : ℝ := 100
noncomputable def todd_debt : ℝ := 110
noncomputable def todd_spent_on_ingredients : ℝ := 75
noncomputable def snow_cones_sold : ℝ := 200
noncomputable def price_per_snowcone : ℝ := 0.75

theorem todd_final_money :
  let initial_money := todd_initial_money,
      debt := todd_debt,
      spent := todd_spent_on_ingredients,
      revenue := snow_cones_sold * price_per_snowcone,
      remaining := initial_money - spent,
      total_pre_debt := remaining + revenue,
      final_money := total_pre_debt - debt
  in final_money = 65 :=
by
  sorry

end todd_final_money_l191_191922


namespace infinite_sum_computation_l191_191558

theorem infinite_sum_computation : 
  ∑' n : ℕ, (3 * (n + 1) + 2) / (n * (n + 1) * (n + 3)) = 10 / 3 :=
by sorry

end infinite_sum_computation_l191_191558


namespace monotonic_decreasing_interval_l191_191084

noncomputable def func (x : ℝ) : ℝ :=
  x * Real.log x

noncomputable def derivative (x : ℝ) : ℝ :=
  Real.log x + 1

theorem monotonic_decreasing_interval :
  { x : ℝ | 0 < x ∧ x < Real.exp (-1) } ⊆ { x : ℝ | derivative x < 0 } :=
by
  sorry

end monotonic_decreasing_interval_l191_191084


namespace oysters_eaten_l191_191780

-- Define the conditions in Lean
def Squido_oysters : ℕ := 200
def Crabby_oysters (Squido_oysters : ℕ) : ℕ := 2 * Squido_oysters

-- Statement to prove
theorem oysters_eaten (Squido_oysters Crabby_oysters : ℕ) (h1 : Crabby_oysters = 2 * Squido_oysters) : 
  Squido_oysters + Crabby_oysters = 600 :=
by
  sorry

end oysters_eaten_l191_191780


namespace surface_area_of_z_eq_xy_over_a_l191_191283

noncomputable def surface_area (a : ℝ) (h : a > 0) : ℝ :=
  (2 * Real.pi / 3) * a^2 * (2 * Real.sqrt 2 - 1)

theorem surface_area_of_z_eq_xy_over_a (a : ℝ) (h : a > 0) :
  surface_area a h = (2 * Real.pi / 3) * a^2 * (2 * Real.sqrt 2 - 1) := 
sorry

end surface_area_of_z_eq_xy_over_a_l191_191283


namespace math_problem_l191_191756

theorem math_problem (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (h : (4*x + y) / (x - 4*y) = -3) : 
  (x + 4*y) / (4*x - y) = 39 / 37 :=
by
  sorry

end math_problem_l191_191756


namespace estimate_total_balls_l191_191870

theorem estimate_total_balls (red_balls : ℕ) (frequency : ℝ) (total_balls : ℕ) 
  (h_red : red_balls = 12) (h_freq : frequency = 0.6) 
  (h_eq : (red_balls : ℝ) / total_balls = frequency) : 
  total_balls = 20 :=
by
  sorry

end estimate_total_balls_l191_191870


namespace negation_of_exists_leq_l191_191909

theorem negation_of_exists_leq (x : ℝ) : ¬ (∃ x : ℝ, x^2 + 2*x + 2 ≤ 0) ↔ ∀ x : ℝ, x^2 + 2*x + 2 > 0 :=
by
  sorry

end negation_of_exists_leq_l191_191909


namespace shopkeeper_loss_percentage_l191_191811

theorem shopkeeper_loss_percentage {cp sp : ℝ} (h1 : cp = 100) (h2 : sp = cp * 1.1) (h_loss : sp * 0.33 = cp * (1 - x / 100)) :
  x = 70 :=
by
  sorry

end shopkeeper_loss_percentage_l191_191811


namespace problem_equiv_proof_l191_191961

-- Define the universe set U
def U : Set ℝ := Set.univ

-- Define the set A based on the given condition
def A : Set ℝ := { x | x^2 + x - 2 ≤ 0 }

-- Define the set B based on the given condition
def B : Set ℝ := { y | ∃ x : ℝ, x ∈ A ∧ y = Real.log (x + 3) / Real.log 2 }

-- Define the complement of B in the universal set U
def complement_B : Set ℝ := { y | y < 0 ∨ y ≥ 2 }

-- Define the set C that is the intersection of A and complement of B
def C : Set ℝ := A ∩ complement_B

-- State the theorem we need to prove
theorem problem_equiv_proof : C = { x | -2 ≤ x ∧ x < 0 } :=
sorry

end problem_equiv_proof_l191_191961


namespace arithmetic_series_sum_l191_191234

theorem arithmetic_series_sum :
  let a := 2
  let d := 3
  let l := 50
  let n := (l - a) / d + 1
  let S := n * (2 * a + (n - 1) * d) / 2
  S = 442 := by
  sorry

end arithmetic_series_sum_l191_191234


namespace rose_part_payment_l191_191743

-- Defining the conditions
def total_cost (T : ℝ) := 0.95 * T = 5700
def part_payment (x : ℝ) (T : ℝ) := x = 0.05 * T

-- The proof problem: Prove that the part payment Rose made is $300
theorem rose_part_payment : ∃ T x, total_cost T ∧ part_payment x T ∧ x = 300 :=
by
  sorry

end rose_part_payment_l191_191743


namespace alpha_value_l191_191176

noncomputable def alpha (x : ℝ) := Real.arccos x

theorem alpha_value (h1 : Real.cos α = -1/6) (h2 : 0 < α ∧ α < Real.pi) : 
  α = Real.pi - alpha (1/6) :=
by
  sorry

end alpha_value_l191_191176


namespace max_dist_to_origin_from_curve_l191_191021

noncomputable def curve (θ : ℝ) : ℝ × ℝ :=
  let x := 3 + Real.sin θ
  let y := Real.cos θ
  (x, y)

theorem max_dist_to_origin_from_curve :
  ∃ M : ℝ × ℝ, (∃ θ : ℝ, M = curve θ) ∧ Real.sqrt (M.fst^2 + M.snd^2) ≤ 4 :=
by
  sorry

end max_dist_to_origin_from_curve_l191_191021


namespace total_balls_estimate_l191_191871

theorem total_balls_estimate 
  (total_balls : ℕ) 
  (red_balls : ℕ) 
  (frequency : ℚ)
  (h_red_balls : red_balls = 12)
  (h_frequency : frequency = 0.6) 
  (h_fraction : (red_balls : ℚ) / total_balls = frequency): 
  total_balls = 20 := 
by 
  sorry

end total_balls_estimate_l191_191871


namespace find_m_l191_191705

theorem find_m (x1 x2 m : ℝ) (h1 : x1 + x2 = -3) (h2 : x1 * x2 = m) (h3 : 1 / x1 + 1 / x2 = 1) : m = -3 :=
by
  sorry

end find_m_l191_191705


namespace shortest_distance_from_origin_l191_191113

noncomputable def shortest_distance_to_circle (x y : ℝ) : ℝ :=
  x^2 + 6 * x + y^2 - 8 * y + 18

theorem shortest_distance_from_origin :
  ∃ (d : ℝ), d = 5 - Real.sqrt 7 ∧ ∀ (x y : ℝ), shortest_distance_to_circle x y = 0 →
    (Real.sqrt ((x - 0)^2 + (y - 0)^2) - Real.sqrt ((x + 3)^2 + (y - 4)^2)) = d := sorry

end shortest_distance_from_origin_l191_191113


namespace grid_shaded_area_l191_191718

theorem grid_shaded_area :
  let grid_side := 12
  let grid_area := grid_side^2
  let radius_small := 1.5
  let radius_large := 3
  let area_small := π * radius_small^2
  let area_large := π * radius_large^2
  let total_area_circles := 3 * area_small + area_large
  let visible_area := grid_area - total_area_circles
  let A := 144
  let B := 15.75
  A = 144 ∧ B = 15.75 ∧ (A + B = 159.75) →
  visible_area = 144 - 15.75 * π :=
by
  intros
  sorry

end grid_shaded_area_l191_191718


namespace olivia_not_sold_bars_l191_191426

theorem olivia_not_sold_bars (cost_per_bar : ℕ) (total_bars : ℕ) (total_money_made : ℕ) :
  cost_per_bar = 3 →
  total_bars = 7 →
  total_money_made = 9 →
  total_bars - (total_money_made / cost_per_bar) = 4 :=
by
  intros h1 h2 h3
  sorry

end olivia_not_sold_bars_l191_191426


namespace ratio_of_sums_equiv_seven_eighths_l191_191046

variable (p q r u v w : ℝ)
variable (hp : 0 < p) (hq : 0 < q) (hr : 0 < r)
variable (hu : 0 < u) (hv : 0 < v) (hw : 0 < w)
variable (h1 : p^2 + q^2 + r^2 = 49)
variable (h2 : u^2 + v^2 + w^2 = 64)
variable (h3 : p * u + q * v + r * w = 56)

theorem ratio_of_sums_equiv_seven_eighths :
  (p + q + r) / (u + v + w) = 7 / 8 :=
by
  sorry

end ratio_of_sums_equiv_seven_eighths_l191_191046


namespace relationship_a_b_l191_191851

open Real

noncomputable def f (x : ℝ) : ℝ := x^2 + x - 2

noncomputable def g (x : ℝ) : ℝ :=
  if x ≤ -2 ∨ x ≥ 1 then 0 else -x^2 - x + 2

theorem relationship_a_b (a b : ℝ) (h_pos : a > 0) :
  (∀ x : ℝ, a * x + b = g x) → (2 * a < b ∧ b < (a + 1)^2 / 4 + 2 ∧ 0 < a ∧ a < 3) :=
sorry

end relationship_a_b_l191_191851


namespace total_oranges_l191_191496

theorem total_oranges (initial_oranges : ℕ) (additional_oranges : ℕ) (weeks : ℕ) (multiplier : ℕ) :
  initial_oranges = 10 → additional_oranges = 5 → weeks = 2 → multiplier = 2 → 
  let first_week := initial_oranges + additional_oranges in
  let next_weeks := weeks * (multiplier * first_week) in
  first_week + next_weeks = 75 :=
begin
  intros h1 h2 h3 h4,
  let first_week := initial_oranges + additional_oranges,
  let next_weeks := weeks * (multiplier * first_week),
  have h_first : first_week = 15, 
  { rw [h1, h2], exact add_comm 10 5 },
  have h_next : next_weeks = 60, 
  { rw [h_first, h3, h4], exact mul_comm 2 30 },
  rw [h_first, h_next],
  exact add_comm 15 60,
end

end total_oranges_l191_191496


namespace time_to_run_up_and_down_l191_191006

/-- Problem statement: Prove that the time it takes Vasya to run up and down a moving escalator 
which moves upwards is 468 seconds, given these conditions:
1. Vasya runs down twice as fast as he runs up.
2. When the escalator is not working, it takes Vasya 6 minutes to run up and down.
3. When the escalator is moving down, it takes Vasya 13.5 minutes to run up and down.
--/
theorem time_to_run_up_and_down (up_speed down_speed : ℝ) (escalator_speed : ℝ) 
  (h1 : down_speed = 2 * up_speed) 
  (h2 : (1 / up_speed + 1 / down_speed) = 6) 
  (h3 : (1 / (up_speed + escalator_speed) + 1 / (down_speed - escalator_speed)) = 13.5) : 
  (1 / (up_speed - escalator_speed) + 1 / (down_speed + escalator_speed)) * 60 = 468 := 
sorry

end time_to_run_up_and_down_l191_191006


namespace least_y_value_l191_191790

theorem least_y_value (y : ℝ) : 2 * y ^ 2 + 7 * y + 3 = 5 → y ≥ -2 :=
by
  intro h
  sorry

end least_y_value_l191_191790


namespace part_a_part_b_l191_191429

theorem part_a (m : ℕ) : m = 1 ∨ m = 2 ∨ m = 4 → (3^m - 1) % (2^m) = 0 := by
  sorry

theorem part_b (m : ℕ) : m = 1 ∨ m = 2 ∨ m = 4 ∨ m = 6 ∨ m = 8 → (31^m - 1) % (2^m) = 0 := by
  sorry

end part_a_part_b_l191_191429


namespace perfect_square_term_l191_191510

def seq (n : ℕ) : ℕ :=
  if n = 0 then 1
  else if n = 1 then 2
  else 4 * seq (n - 1) - seq (n - 2)

theorem perfect_square_term : ∀ n, (∃ k, seq n = k * k) ↔ n = 0 := by
  sorry

end perfect_square_term_l191_191510


namespace range_of_a1_l191_191817

theorem range_of_a1 (a : ℕ → ℝ) (h : ∀ n, a (n + 1) = 1 / (2 - a n)) 
  (h_pos : ∀ n, a (n + 1) > a n) : a 1 < 1 := 
sorry

end range_of_a1_l191_191817


namespace div_add_example_l191_191386

theorem div_add_example : 150 / (10 / 2) + 5 = 35 := by
  sorry

end div_add_example_l191_191386


namespace af_b_lt_bf_a_l191_191446

variable {f : ℝ → ℝ}
variable {a b : ℝ}

theorem af_b_lt_bf_a (h1 : ∀ x y, 0 < x → 0 < y → x < y → f x > f y)
                    (h2 : ∀ x, 0 < x → f x > 0)
                    (h3 : 0 < a)
                    (h4 : 0 < b)
                    (h5 : a < b) :
  a * f b < b * f a :=
sorry

end af_b_lt_bf_a_l191_191446


namespace contrapositive_equivalence_l191_191798

theorem contrapositive_equivalence (P Q : Prop) : (P → Q) ↔ (¬ Q → ¬ P) :=
by sorry

end contrapositive_equivalence_l191_191798


namespace odd_product_probability_lt_one_eighth_l191_191919

theorem odd_product_probability_lt_one_eighth : 
  (∃ p : ℝ, p = (500 / 1000) * (499 / 999) * (498 / 998)) → p < 1 / 8 :=
by
  sorry

end odd_product_probability_lt_one_eighth_l191_191919


namespace problem_a_b_l191_191196

theorem problem_a_b (a b : ℝ) (h₁ : a + b = 10) (h₂ : a - b = 4) : a^2 - b^2 = 40 :=
by
  sorry

end problem_a_b_l191_191196


namespace equal_share_payment_l191_191731

theorem equal_share_payment (A B C : ℝ) (h : A < B ∧ B < C) :
  (B + C - 2 * A) / 3 + (A + B - 2 * C) / 3 = ((A + B + C) * 2 / 3) - B :=
by
  sorry

end equal_share_payment_l191_191731


namespace hyperbola_m_range_l191_191899

-- Define the equation of the hyperbola
def is_hyperbola (m : ℝ) : Prop :=
  ∀ x y : ℝ, (x^2 / (m + 2)) - (y^2 / (m - 1)) = 1

-- State the equivalent range problem
theorem hyperbola_m_range (m : ℝ) :
  is_hyperbola m ↔ (m < -2 ∨ m > 1) :=
by
  sorry

end hyperbola_m_range_l191_191899


namespace intersection_unique_point_l191_191341

def f (x : ℝ) : ℝ := x^3 + 6 * x^2 + 16 * x + 28

theorem intersection_unique_point :
  ∃ a : ℝ, f a = a ∧ a = -4 := sorry

end intersection_unique_point_l191_191341


namespace number_of_seeds_per_row_l191_191556

-- Define the conditions as variables
def rows : ℕ := 6
def total_potatoes : ℕ := 54
def seeds_per_row : ℕ := 9

-- State the theorem
theorem number_of_seeds_per_row :
  total_potatoes / rows = seeds_per_row :=
by
-- We ignore the proof here, it will be provided later
sorry

end number_of_seeds_per_row_l191_191556


namespace elise_spent_on_comic_book_l191_191295

-- Define the initial amount of money Elise had
def initial_amount : ℤ := 8

-- Define the amount saved from allowance
def saved_amount : ℤ := 13

-- Define the amount spent on puzzle
def spent_on_puzzle : ℤ := 18

-- Define the amount left after all expenditures
def amount_left : ℤ := 1

-- Define the total amount of money Elise had after saving
def total_amount : ℤ := initial_amount + saved_amount

-- Define the total amount spent which equals
-- the sum of amount spent on the comic book and the puzzle
def total_spent : ℤ := total_amount - amount_left

-- Define the amount spent on the comic book as the proposition to be proved
def spent_on_comic_book : ℤ := total_spent - spent_on_puzzle

-- State the theorem to prove how much Elise spent on the comic book
theorem elise_spent_on_comic_book : spent_on_comic_book = 2 :=
by
  sorry

end elise_spent_on_comic_book_l191_191295


namespace elroy_more_miles_l191_191157

theorem elroy_more_miles (m_last_year : ℝ) (m_this_year : ℝ) (collect_last_year : ℝ) :
  m_last_year = 4 → m_this_year = 2.75 → collect_last_year = 44 → 
  (collect_last_year / m_this_year - collect_last_year / m_last_year = 5) :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  sorry

end elroy_more_miles_l191_191157


namespace jerry_feathers_count_l191_191876

noncomputable def hawk_feathers : ℕ := 6
noncomputable def eagle_feathers : ℕ := 17 * hawk_feathers
noncomputable def total_feathers : ℕ := hawk_feathers + eagle_feathers
noncomputable def remaining_feathers_after_sister : ℕ := total_feathers - 10
noncomputable def jerry_feathers_left : ℕ := remaining_feathers_after_sister / 2

theorem jerry_feathers_count : jerry_feathers_left = 49 :=
  by
  sorry

end jerry_feathers_count_l191_191876


namespace min_value_ineq_l191_191583

theorem min_value_ineq (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x + y = 1) : 
  (∀ z : ℝ, z = (4 / x + 1 / y) → z ≥ 9) :=
by
  sorry

end min_value_ineq_l191_191583


namespace fruit_seller_stock_l191_191670

-- Define the given conditions
def remaining_oranges : ℝ := 675
def remaining_percentage : ℝ := 0.25

-- Define the problem function
def original_stock (O : ℝ) : Prop :=
  remaining_percentage * O = remaining_oranges

-- Prove the original stock of oranges was 2700 kg
theorem fruit_seller_stock : original_stock 2700 :=
by
  sorry

end fruit_seller_stock_l191_191670


namespace probability_at_least_3_out_of_6_babies_speak_l191_191329

noncomputable def binomial_prob (n : ℕ) (k : ℕ) (p : ℚ) : ℚ :=
  Nat.choose n k * (p^k) * ((1 - p)^(n - k))

noncomputable def prob_at_least_k (total : ℕ) (k : ℕ) (p : ℚ) : ℚ :=
  1 - (Finset.range k).sum (λ i => binomial_prob total i p)

theorem probability_at_least_3_out_of_6_babies_speak :
  prob_at_least_k 6 3 (2/5) = 7120/15625 :=
by
  sorry

end probability_at_least_3_out_of_6_babies_speak_l191_191329


namespace unique_solution_l191_191839

theorem unique_solution (m n : ℤ) (h : 231 * m^2 = 130 * n^2) : m = 0 ∧ n = 0 :=
by {
  sorry
}

end unique_solution_l191_191839


namespace equilateral_triangle_ratio_l191_191946

-- Define the side length of the equilateral triangle
def side_length : ℝ := 10

-- Define the altitude of the equilateral triangle
def altitude (a : ℝ) : ℝ := a * (Real.sqrt 3) / 2

-- Define the area of the equilateral triangle
def area (a : ℝ) : ℝ := (a * altitude a) / 2

-- Define the perimeter of the equilateral triangle
def perimeter (a : ℝ) : ℝ := 3 * a

-- Define the ratio of area to perimeter
def ratio (a : ℝ) : ℝ := area a / perimeter a

theorem equilateral_triangle_ratio :
  ratio 10 = (5 * Real.sqrt 3) / 6 :=
by
  sorry

end equilateral_triangle_ratio_l191_191946


namespace adult_ticket_cost_l191_191104

variable (A : ℝ)

theorem adult_ticket_cost :
  (20 * 6) + (12 * A) = 216 → A = 8 :=
by
  intro h
  sorry

end adult_ticket_cost_l191_191104


namespace probability_of_selecting_quarter_l191_191536

theorem probability_of_selecting_quarter 
  (value_quarters value_nickels value_pennies total_value : ℚ)
  (coin_value_quarter coin_value_nickel coin_value_penny : ℚ) 
  (h1 : value_quarters = 10)
  (h2 : value_nickels = 10)
  (h3 : value_pennies = 10)
  (h4 : coin_value_quarter = 0.25)
  (h5 : coin_value_nickel = 0.05)
  (h6 : coin_value_penny = 0.01)
  (total_coins : ℚ) 
  (h7 : total_coins = (value_quarters / coin_value_quarter) + (value_nickels / coin_value_nickel) + (value_pennies / coin_value_penny)) : 
  (value_quarters / coin_value_quarter) / total_coins = 1 / 31 :=
by
  sorry

end probability_of_selecting_quarter_l191_191536


namespace driving_time_is_correct_l191_191826

-- Define conditions
def flight_departure : ℕ := 20 * 60 -- 8:00 pm in minutes since 0:00
def checkin_time : ℕ := flight_departure - 2 * 60 -- 2 hours early
def latest_leave_time : ℕ := 17 * 60 -- 5:00 pm in minutes since 0:00
def additional_time : ℕ := 15 -- 15 minutes to park and make their way to the terminal

-- Define question
def driving_time : ℕ := checkin_time - additional_time - latest_leave_time

-- Prove the expected answer
theorem driving_time_is_correct : driving_time = 45 :=
by
  -- omitting the proof
  sorry

end driving_time_is_correct_l191_191826


namespace park_bench_problem_l191_191970

/-- A single bench section at a park can hold either 8 adults or 12 children.
When N bench sections are connected end to end, an equal number of adults and 
children seated together will occupy all the bench space.
This theorem states that the smallest positive integer N such that this condition 
is satisfied is 3. -/
theorem park_bench_problem : ∃ N : ℕ, N > 0 ∧ (8 * N = 12 * N) ∧ N = 3 :=
by
  sorry

end park_bench_problem_l191_191970


namespace oldest_son_park_visits_l191_191884

theorem oldest_son_park_visits 
    (season_pass_cost : ℕ)
    (cost_per_trip : ℕ)
    (youngest_son_trips : ℕ) 
    (remaining_value : ℕ)
    (oldest_son_trips : ℕ) : 
    season_pass_cost = 100 →
    cost_per_trip = 4 →
    youngest_son_trips = 15 →
    remaining_value = season_pass_cost - youngest_son_trips * cost_per_trip →
    oldest_son_trips = remaining_value / cost_per_trip →
    oldest_son_trips = 10 := 
by sorry

end oldest_son_park_visits_l191_191884
