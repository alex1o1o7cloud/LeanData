import Mathlib

namespace NUMINAMATH_GPT_half_abs_diff_squares_l2090_209038

/-- Half of the absolute value of the difference of the squares of 23 and 19 is 84. -/
theorem half_abs_diff_squares : (1 / 2 : ℝ) * |(23^2 : ℝ) - (19^2 : ℝ)| = 84 :=
by
  sorry

end NUMINAMATH_GPT_half_abs_diff_squares_l2090_209038


namespace NUMINAMATH_GPT_triangle_area_l2090_209023

variable (a b c k : ℝ)
variable (h1 : a = 2 * k)
variable (h2 : b = 3 * k)
variable (h3 : c = k * Real.sqrt 13)

theorem triangle_area (h_right_triangle : a^2 + b^2 = c^2) : 
  (1 / 2 * a * b) = 3 * k^2 := 
by 
  sorry

end NUMINAMATH_GPT_triangle_area_l2090_209023


namespace NUMINAMATH_GPT_probability_two_points_square_l2090_209065

def gcd (a b c : Nat) : Nat := Nat.gcd (Nat.gcd a b) c  

theorem probability_two_points_square {a b c : ℕ} (hx : gcd a b c = 1)
  (h : (26 - Real.pi) / 32 = (a - b * Real.pi) / c) : a + b + c = 59 :=
by
  sorry

end NUMINAMATH_GPT_probability_two_points_square_l2090_209065


namespace NUMINAMATH_GPT_find_dividend_l2090_209094

theorem find_dividend :
  ∀ (Divisor Quotient Remainder : ℕ), Divisor = 15 → Quotient = 9 → Remainder = 5 → (Divisor * Quotient + Remainder) = 140 :=
by
  intros Divisor Quotient Remainder hDiv hQuot hRem
  subst hDiv
  subst hQuot
  subst hRem
  sorry

end NUMINAMATH_GPT_find_dividend_l2090_209094


namespace NUMINAMATH_GPT_number_of_weavers_is_4_l2090_209013

theorem number_of_weavers_is_4
  (mats1 days1 weavers1 mats2 days2 weavers2 : ℕ)
  (h1 : mats1 = 4)
  (h2 : days1 = 4)
  (h3 : weavers2 = 10)
  (h4 : mats2 = 25)
  (h5 : days2 = 10)
  (h_rate_eq : (mats1 / (weavers1 * days1)) = (mats2 / (weavers2 * days2))) :
  weavers1 = 4 :=
by
  sorry

end NUMINAMATH_GPT_number_of_weavers_is_4_l2090_209013


namespace NUMINAMATH_GPT_greatest_q_minus_r_l2090_209073

theorem greatest_q_minus_r : 
  ∃ (q r : ℕ), 1001 = 17 * q + r ∧ q - r = 43 :=
by
  sorry

end NUMINAMATH_GPT_greatest_q_minus_r_l2090_209073


namespace NUMINAMATH_GPT_problem1_l2090_209044

theorem problem1
  (a b c : ℝ)
  (h1: a > 0)
  (h2: b > 0)
  (h3: c > 0) :
  a + b + c ≥ Real.sqrt (a * b) + Real.sqrt (b * c) + Real.sqrt (c * a) :=
sorry

end NUMINAMATH_GPT_problem1_l2090_209044


namespace NUMINAMATH_GPT_symmetric_point_l2090_209057

theorem symmetric_point (P : ℝ × ℝ) (a b : ℝ) (h1: P = (2, 1)) (h2 : x - y + 1 = 0) :
  (b - 1) = -(a - 2) ∧ (a + 2) / 2 - (b + 1) / 2 + 1 = 0 → (a, b) = (0, 3) := 
sorry

end NUMINAMATH_GPT_symmetric_point_l2090_209057


namespace NUMINAMATH_GPT_inequality_proof_l2090_209002

theorem inequality_proof (x : ℝ) (n : ℕ) (h : 3 * x ≥ -1) : (1 + x) ^ n ≥ 1 + n * x :=
sorry

end NUMINAMATH_GPT_inequality_proof_l2090_209002


namespace NUMINAMATH_GPT_time_spent_on_type_a_problems_l2090_209033

-- Define the conditions
def total_questions := 200
def examination_duration_hours := 3
def type_a_problems := 100
def type_b_problems := total_questions - type_a_problems
def type_a_time_coeff := 2

-- Convert examination duration to minutes
def examination_duration_minutes := examination_duration_hours * 60

-- Variables for time per problem
variable (x : ℝ)

-- The total time spent
def total_time_spent : ℝ := type_a_problems * (type_a_time_coeff * x) + type_b_problems * x

-- Statement we need to prove
theorem time_spent_on_type_a_problems :
  total_time_spent x = examination_duration_minutes → type_a_problems * (type_a_time_coeff * x) = 120 :=
by
  sorry

end NUMINAMATH_GPT_time_spent_on_type_a_problems_l2090_209033


namespace NUMINAMATH_GPT_spherical_to_rectangular_coords_l2090_209040

theorem spherical_to_rectangular_coords
  (ρ θ φ : ℝ)
  (hρ : ρ = 6)
  (hθ : θ = 7 * Real.pi / 4)
  (hφ : φ = Real.pi / 3) :
  let x := ρ * Real.sin φ * Real.cos θ
  let y := ρ * Real.sin φ * Real.sin θ
  let z := ρ * Real.cos φ
  x = -3 * Real.sqrt 6 ∧ y = -3 * Real.sqrt 6 ∧ z = 3 :=
by
  sorry

end NUMINAMATH_GPT_spherical_to_rectangular_coords_l2090_209040


namespace NUMINAMATH_GPT_currency_exchange_rate_l2090_209077

theorem currency_exchange_rate (b g x : ℕ) (h1 : 1 * b * g = b * g) (h2 : 1 = 1) :
  (b + g) ^ 2 + 1 = b * g * x → x = 5 :=
sorry

end NUMINAMATH_GPT_currency_exchange_rate_l2090_209077


namespace NUMINAMATH_GPT_simplify_expression_l2090_209017

theorem simplify_expression (r : ℝ) : (2 * r^2 + 5 * r - 3) + (3 * r^2 - 4 * r + 2) = 5 * r^2 + r - 1 := 
by
  sorry

end NUMINAMATH_GPT_simplify_expression_l2090_209017


namespace NUMINAMATH_GPT_number_multiplied_by_3_l2090_209000

variable (A B C D E : ℝ) -- Declare the five numbers

theorem number_multiplied_by_3 (h1 : (A + B + C + D + E) / 5 = 6.8) 
    (h2 : ∃ X : ℝ, (A + B + C + D + E + 2 * X) / 5 = 9.2) : 
    ∃ X : ℝ, X = 6 := 
  sorry

end NUMINAMATH_GPT_number_multiplied_by_3_l2090_209000


namespace NUMINAMATH_GPT_prob_selected_first_eq_third_l2090_209089

noncomputable def total_students_first := 800
noncomputable def total_students_second := 600
noncomputable def total_students_third := 500
noncomputable def selected_students_third := 25
noncomputable def prob_selected_third := selected_students_third / total_students_third

theorem prob_selected_first_eq_third :
  (selected_students_third / total_students_third = 1 / 20) →
  (prob_selected_third = 1 / 20) :=
by
  intros h
  sorry

end NUMINAMATH_GPT_prob_selected_first_eq_third_l2090_209089


namespace NUMINAMATH_GPT_solve_inequality_l2090_209080

theorem solve_inequality (x : ℝ) :
  (x - 1)^2 < 12 - x ↔ 
  (Real.sqrt 5) ≠ 0 ∧
  (1 - 3 * (Real.sqrt 5)) / 2 < x ∧ 
  x < (1 + 3 * (Real.sqrt 5)) / 2 :=
sorry

end NUMINAMATH_GPT_solve_inequality_l2090_209080


namespace NUMINAMATH_GPT_find_number_of_values_l2090_209093

theorem find_number_of_values (n S : ℕ) (h1 : S / n = 250) (h2 : S + 30 = 251 * n) : n = 30 :=
sorry

end NUMINAMATH_GPT_find_number_of_values_l2090_209093


namespace NUMINAMATH_GPT_sequence_an_general_formula_and_sum_bound_l2090_209076

theorem sequence_an_general_formula_and_sum_bound (a : ℕ → ℝ)
  (S : ℕ → ℝ)
  (b : ℕ → ℝ)
  (T : ℕ → ℝ)
  (h1 : ∀ n, S n = (1 / 4) * (a n + 1) ^ 2)
  (h2 : ∀ n, b n = 1 / (a n * a (n + 1)))
  (h3 : ∀ n, T n = (1 / 2) * (1 - (1 / (2 * n + 1))))
  (h4 : ∀ n, 0 < a n) :
  (∀ n, a n = 2 * n - 1) ∧ (∀ n, T n < 1 / 2) := 
by
  sorry

end NUMINAMATH_GPT_sequence_an_general_formula_and_sum_bound_l2090_209076


namespace NUMINAMATH_GPT_minimum_common_ratio_l2090_209014

theorem minimum_common_ratio (a : ℕ) (n : ℕ) (q : ℝ) (h_pos : ∀ i, i < n → 0 < a * q^i) (h_geom : ∀ i j, i < j → a * q^i < a * q^j) (h_q : 1 < q ∧ q < 2) : q = 6 / 5 :=
by
  sorry

end NUMINAMATH_GPT_minimum_common_ratio_l2090_209014


namespace NUMINAMATH_GPT_squares_difference_l2090_209030

theorem squares_difference :
  1010^2 - 994^2 - 1008^2 + 996^2 = 8016 :=
by
  sorry

end NUMINAMATH_GPT_squares_difference_l2090_209030


namespace NUMINAMATH_GPT_calculate_distance_to_friend_l2090_209056

noncomputable def distance_to_friend (d t : ℝ) : Prop :=
  (d = 45 * (t + 1)) ∧ (d = 45 + 65 * (t - 0.75))

theorem calculate_distance_to_friend : ∃ d t: ℝ, distance_to_friend d t ∧ d = 155 :=
by
  exists 155
  exists 2.4375
  sorry

end NUMINAMATH_GPT_calculate_distance_to_friend_l2090_209056


namespace NUMINAMATH_GPT_find_k_l2090_209036

-- Define the vectors a, b, and c
def vecA : ℝ × ℝ := (2, -1)
def vecB : ℝ × ℝ := (1, 1)
def vecC : ℝ × ℝ := (-5, 1)

-- Define the condition for two vectors being parallel
def parallel (u v : ℝ × ℝ) : Prop :=
  u.1 * v.2 - u.2 * v.1 = 0

-- Define the target statement to be proven
theorem find_k : ∃ k : ℝ, parallel (vecA.1 + k * vecB.1, vecA.2 + k * vecB.2) vecC ∧ k = 1/2 := 
sorry

end NUMINAMATH_GPT_find_k_l2090_209036


namespace NUMINAMATH_GPT_cost_of_blue_cap_l2090_209007

theorem cost_of_blue_cap (cost_tshirt cost_backpack cost_cap total_spent discount: ℝ) 
  (h1 : cost_tshirt = 30) 
  (h2 : cost_backpack = 10) 
  (h3 : discount = 2)
  (h4 : total_spent = 43) 
  (h5 : total_spent = cost_tshirt + cost_backpack + cost_cap - discount) : 
  cost_cap = 5 :=
by sorry

end NUMINAMATH_GPT_cost_of_blue_cap_l2090_209007


namespace NUMINAMATH_GPT_not_beautiful_739_and_741_l2090_209092

-- Define the function g and its properties
variable (g : ℤ → ℤ)

-- Condition: g(x) ≠ x
axiom g_neq_x (x : ℤ) : g x ≠ x

-- Definition of "beautiful"
def beautiful (a : ℤ) : Prop :=
  ∀ x : ℤ, g x = g (a - x)

-- The theorem to prove
theorem not_beautiful_739_and_741 :
  ¬ (beautiful g 739 ∧ beautiful g 741) :=
sorry

end NUMINAMATH_GPT_not_beautiful_739_and_741_l2090_209092


namespace NUMINAMATH_GPT_percent_asian_population_in_West_l2090_209051

-- Define the populations in different regions
def population_NE := 2
def population_MW := 3
def population_South := 4
def population_West := 10

-- Define the total population
def total_population := population_NE + population_MW + population_South + population_West

-- Calculate the percentage of the population in the West
def percentage_in_West := (population_West * 100) / total_population

-- The proof statement
theorem percent_asian_population_in_West : percentage_in_West = 53 := by
  sorry -- proof to be completed

end NUMINAMATH_GPT_percent_asian_population_in_West_l2090_209051


namespace NUMINAMATH_GPT_square_perimeter_l2090_209069

theorem square_perimeter (x : ℝ) (h : x * x + x * x = (2 * Real.sqrt 2) * (2 * Real.sqrt 2)) :
    4 * x = 8 :=
by
  sorry

end NUMINAMATH_GPT_square_perimeter_l2090_209069


namespace NUMINAMATH_GPT_certain_number_l2090_209071

theorem certain_number (N : ℝ) (k : ℝ) 
  (h1 : (1 / 2) ^ 22 * N ^ k = 1 / 18 ^ 22) 
  (h2 : k = 11) 
  : N = 81 := 
by
  sorry

end NUMINAMATH_GPT_certain_number_l2090_209071


namespace NUMINAMATH_GPT_inverse_mod_53_l2090_209048

theorem inverse_mod_53 (h : 17 * 13 % 53 = 1) : 36 * 40 % 53 = 1 :=
by
  -- Given condition: 17 * 13 % 53 = 1
  -- Derived condition: (-17) * -13 % 53 = 1 which is equivalent to 17 * 13 % 53 = 1
  -- So we need to find: 36 * x % 53 = 1 where x = -13 % 53 => x = 40
  sorry

end NUMINAMATH_GPT_inverse_mod_53_l2090_209048


namespace NUMINAMATH_GPT_shaded_region_area_l2090_209085

theorem shaded_region_area (r : ℝ) (π : ℝ) (h1 : r = 5) : 
  4 * ((1/2 * π * r * r) - (1/2 * r * r)) = 50 * π - 50 :=
by 
  sorry

end NUMINAMATH_GPT_shaded_region_area_l2090_209085


namespace NUMINAMATH_GPT_circle_complete_the_square_l2090_209046

/-- Given the equation x^2 - 6x + y^2 - 10y + 18 = 0, show that it can be transformed to  
    (x - 3)^2 + (y - 5)^2 = 4^2 -/
theorem circle_complete_the_square :
  ∀ x y : ℝ, x^2 - 6 * x + y^2 - 10 * y + 18 = 0 ↔ (x - 3)^2 + (y - 5)^2 = 4^2 :=
by
  sorry

end NUMINAMATH_GPT_circle_complete_the_square_l2090_209046


namespace NUMINAMATH_GPT_early_time_l2090_209083

noncomputable def speed1 : ℝ := 5 -- km/hr
noncomputable def timeLate : ℝ := 5 / 60 -- convert minutes to hours
noncomputable def speed2 : ℝ := 10 -- km/hr
noncomputable def distance : ℝ := 2.5 -- km

theorem early_time (speed1 speed2 distance : ℝ) (timeLate : ℝ) :
  (distance / speed1 - timeLate) * 60 - (distance / speed2) * 60 = 10 :=
by
  sorry

end NUMINAMATH_GPT_early_time_l2090_209083


namespace NUMINAMATH_GPT_second_hand_bisect_angle_l2090_209084

theorem second_hand_bisect_angle :
  ∃ x : ℚ, (6 * x - 360 * (x - 1) = 360 * (x - 1) - 0.5 * x) ∧ (x = 1440 / 1427) :=
by
  sorry

end NUMINAMATH_GPT_second_hand_bisect_angle_l2090_209084


namespace NUMINAMATH_GPT_no_such_k_l2090_209047

theorem no_such_k (u : ℕ → ℝ) (v : ℕ → ℝ)
  (h1 : u 0 = 6) (h2 : v 0 = 4)
  (h3 : ∀ n, u (n + 1) = (3 / 5) * u n - (4 / 5) * v n)
  (h4 : ∀ n, v (n + 1) = (4 / 5) * u n + (3 / 5) * v n) :
  ¬ ∃ k, u k = 7 ∧ v k = 2 :=
by
  sorry

end NUMINAMATH_GPT_no_such_k_l2090_209047


namespace NUMINAMATH_GPT_equal_cubes_l2090_209062

theorem equal_cubes (r s : ℤ) (hr : 0 ≤ r) (hs : 0 ≤ s)
  (h : |r^3 - s^3| = |6 * r^2 - 6 * s^2|) : r = s :=
by
  sorry

end NUMINAMATH_GPT_equal_cubes_l2090_209062


namespace NUMINAMATH_GPT_sqrt_18_mul_sqrt_32_eq_24_l2090_209043
  
theorem sqrt_18_mul_sqrt_32_eq_24 : (Real.sqrt 18 * Real.sqrt 32 = 24) :=
  sorry

end NUMINAMATH_GPT_sqrt_18_mul_sqrt_32_eq_24_l2090_209043


namespace NUMINAMATH_GPT_percentage_cost_for_overhead_l2090_209029

theorem percentage_cost_for_overhead
  (P M N : ℝ)
  (hP : P = 48)
  (hM : M = 50)
  (hN : N = 12) :
  (P + M - P - N) / P * 100 = 79.17 := by
  sorry

end NUMINAMATH_GPT_percentage_cost_for_overhead_l2090_209029


namespace NUMINAMATH_GPT_combinations_of_balls_and_hats_l2090_209001

def validCombinations (b h : ℕ) : Prop :=
  6 * b + 4 * h = 100 ∧ h ≥ 2

theorem combinations_of_balls_and_hats : 
  (∃ (n : ℕ), n = 8 ∧ (∀ b h : ℕ, validCombinations b h → validCombinations b h)) :=
by
  sorry

end NUMINAMATH_GPT_combinations_of_balls_and_hats_l2090_209001


namespace NUMINAMATH_GPT_gcd_lcm_sum_eq_90_l2090_209012

def gcd_three (a b c : ℕ) : ℕ := Nat.gcd (Nat.gcd a b) c
def lcm_three (a b c : ℕ) : ℕ := Nat.lcm (Nat.lcm a b) c

theorem gcd_lcm_sum_eq_90 : 
  let A := gcd_three 18 36 72
  let B := lcm_three 18 36 72
  A + B = 90 :=
by
  let A := gcd_three 18 36 72
  let B := lcm_three 18 36 72
  sorry

end NUMINAMATH_GPT_gcd_lcm_sum_eq_90_l2090_209012


namespace NUMINAMATH_GPT_abs_opposite_numbers_l2090_209006

theorem abs_opposite_numbers (m n : ℤ) (h : m + n = 0) : |m + n - 1| = 1 := by
  sorry

end NUMINAMATH_GPT_abs_opposite_numbers_l2090_209006


namespace NUMINAMATH_GPT_proposition_4_correct_l2090_209060

section

variables {Point Line Plane : Type}
variables (m n : Line) (α β γ : Plane)

-- Definitions of perpendicular and parallel relationships
def perpendicular (l : Line) (p : Plane) : Prop := sorry
def parallel (x y : Line) : Prop := sorry

theorem proposition_4_correct (h1 : perpendicular m α) (h2 : perpendicular n α) : parallel m n :=
sorry

end

end NUMINAMATH_GPT_proposition_4_correct_l2090_209060


namespace NUMINAMATH_GPT_area_of_triangle_ABC_l2090_209091

def Point : Type := (ℝ × ℝ)

def A : Point := (0, 0)
def B : Point := (2, 2)
def C : Point := (2, 0)

def triangle_area (p1 p2 p3 : Point) : ℝ :=
  0.5 * abs (p1.1 * (p2.2 - p3.2) + p2.1 * (p3.2 - p1.2) + p3.1 * (p1.2 - p2.2))

theorem area_of_triangle_ABC :
  triangle_area A B C = 2 :=
by
  sorry

end NUMINAMATH_GPT_area_of_triangle_ABC_l2090_209091


namespace NUMINAMATH_GPT_min_distance_PQ_l2090_209075

theorem min_distance_PQ :
  let P_line (ρ θ : ℝ) := ρ * (Real.cos θ + Real.sin θ) = 4
  let Q_circle (ρ θ : ℝ) := ρ^2 = 4 * ρ * Real.cos θ - 3
  ∃ (P Q : ℝ × ℝ), 
    (∃ ρP θP, P = (ρP * Real.cos θP, ρP * Real.sin θP) ∧ P_line ρP θP) ∧
    (∃ ρQ θQ, Q = (ρQ * Real.cos θQ, ρQ * Real.sin θQ) ∧ Q_circle ρQ θQ) ∧
    ∀ R S : ℝ × ℝ, 
      (∃ ρR θR, R = (ρR * Real.cos θR, ρR * Real.sin θR) ∧ P_line ρR θR) →
      (∃ ρS θS, S = (ρS * Real.cos θS, ρS * Real.sin θS) ∧ Q_circle ρS θS) →
      dist P Q ≤ dist R S :=
  sorry

end NUMINAMATH_GPT_min_distance_PQ_l2090_209075


namespace NUMINAMATH_GPT_find_d_l2090_209096

noncomputable def polynomial_d (a b c d : ℤ) (p q r s : ℤ) : Prop :=
  p > 0 ∧ q > 0 ∧ r > 0 ∧ s > 0 ∧
  1 + a + b + c + d = 2024 ∧
  (1 + p) * (1 + q) * (1 + r) * (1 + s) = 2024 ∧
  d = p * q * r * s

theorem find_d (a b c d : ℤ) (h : polynomial_d a b c d 7 10 22 11) : d = 17020 :=
  sorry

end NUMINAMATH_GPT_find_d_l2090_209096


namespace NUMINAMATH_GPT_max_cards_with_digit_three_l2090_209090

/-- There are ten cards each of the digits "3", "4", and "5". Choose any 8 cards such that their sum is 27. 
Prove that the maximum number of these cards that can be "3" is 6. -/
theorem max_cards_with_digit_three (c3 c4 c5 : ℕ) (hc3 : c3 + c4 + c5 = 8) (h_sum : 3 * c3 + 4 * c4 + 5 * c5 = 27) :
  c3 ≤ 6 :=
sorry

end NUMINAMATH_GPT_max_cards_with_digit_three_l2090_209090


namespace NUMINAMATH_GPT_perimeter_division_l2090_209003

-- Define the given conditions
def is_pentagon (n : ℕ) : Prop := n = 5
def side_length (s : ℕ) : Prop := s = 25
def perimeter (P : ℕ) (n s : ℕ) : Prop := P = n * s

-- Define the Lean statement to prove
theorem perimeter_division (n s P x : ℕ) 
  (h1 : is_pentagon n) 
  (h2 : side_length s) 
  (h3 : perimeter P n s) 
  (h4 : P = 125) 
  (h5 : s = 25) : 
  P / x = s → x = 5 := 
by
  sorry

end NUMINAMATH_GPT_perimeter_division_l2090_209003


namespace NUMINAMATH_GPT_nuts_in_tree_l2090_209078

theorem nuts_in_tree (squirrels nuts : ℕ) (h1 : squirrels = 4) (h2 : squirrels = nuts + 2) : nuts = 2 :=
by
  sorry

end NUMINAMATH_GPT_nuts_in_tree_l2090_209078


namespace NUMINAMATH_GPT_quadratic_roots_property_l2090_209098

theorem quadratic_roots_property (a b : ℝ)
  (h1 : a^2 - 2 * a - 1 = 0)
  (h2 : b^2 - 2 * b - 1 = 0)
  (ha_b_sum : a + b = 2)
  (ha_b_product : a * b = -1) :
  a^2 + 2 * b - a * b = 6 :=
sorry

end NUMINAMATH_GPT_quadratic_roots_property_l2090_209098


namespace NUMINAMATH_GPT_isosceles_triangle_perimeter_l2090_209061

-- Define the lengths of the sides of the isosceles triangle
def side1 : ℕ := 12
def side2 : ℕ := 12
def base : ℕ := 17

-- Define the perimeter as the sum of all three sides
def perimeter : ℕ := side1 + side2 + base

-- State the theorem that needs to be proved
theorem isosceles_triangle_perimeter : perimeter = 41 := by
  -- Insert the proof here
  sorry

end NUMINAMATH_GPT_isosceles_triangle_perimeter_l2090_209061


namespace NUMINAMATH_GPT_percentage_not_sophomores_l2090_209015

variable (Total : ℕ) (Juniors Senior : ℕ) (Freshmen Sophomores : ℕ)

-- Conditions
axiom total_students : Total = 800
axiom percent_juniors : (22 / 100) * Total = Juniors
axiom number_seniors : Senior = 160
axiom freshmen_sophomores_relation : Freshmen = Sophomores + 64
axiom total_composition : Freshmen + Sophomores + Juniors + Senior = Total

-- Proof Objective
theorem percentage_not_sophomores :
  (Total - Sophomores) / Total * 100 = 75 :=
by
  -- proof omitted
  sorry

end NUMINAMATH_GPT_percentage_not_sophomores_l2090_209015


namespace NUMINAMATH_GPT_find_teachers_and_students_l2090_209066

-- Mathematical statements corresponding to the problem conditions
def teachers_and_students_found (x y : ℕ) : Prop :=
  (y = 30 * x + 7) ∧ (31 * x = y + 1)

-- The theorem we need to prove
theorem find_teachers_and_students : ∃ x y, teachers_and_students_found x y ∧ x = 8 ∧ y = 247 :=
  by
    sorry

end NUMINAMATH_GPT_find_teachers_and_students_l2090_209066


namespace NUMINAMATH_GPT_sufficient_and_necessary_condition_l2090_209049

variable {a : ℕ → ℝ}
variable {a1 a2 : ℝ}
variable {q : ℝ}

noncomputable def geometric_sequence (a : ℕ → ℝ) (a1 : ℝ) (q : ℝ) : Prop :=
  (∀ n, a n = a1 * q ^ n)

noncomputable def increasing (a : ℕ → ℝ) : Prop :=
  (∀ n, a n < a (n + 1))

theorem sufficient_and_necessary_condition
  (a : ℕ → ℝ) (a1 : ℝ) (q : ℝ)
  (h_geom : geometric_sequence a a1 q)
  (h_a1_pos : a1 > 0)
  (h_a1_lt_a2 : a1 < a1 * q) :
  increasing a ↔ a1 < a1 * q := 
sorry

end NUMINAMATH_GPT_sufficient_and_necessary_condition_l2090_209049


namespace NUMINAMATH_GPT_total_photos_l2090_209042

def initial_photos : ℕ := 100
def photos_first_week : ℕ := 50
def photos_second_week : ℕ := 2 * photos_first_week
def photos_third_and_fourth_weeks : ℕ := 80

theorem total_photos (initial_photos photos_first_week photos_second_week photos_third_and_fourth_weeks : ℕ) :
  initial_photos = 100 ∧
  photos_first_week = 50 ∧
  photos_second_week = 2 * photos_first_week ∧
  photos_third_and_fourth_weeks = 80 →
  initial_photos + photos_first_week + photos_second_week + photos_third_and_fourth_weeks = 330 :=
by
  sorry

end NUMINAMATH_GPT_total_photos_l2090_209042


namespace NUMINAMATH_GPT_corrected_mean_l2090_209025

theorem corrected_mean (mean_initial : ℝ) (num_obs : ℕ) (obs_incorrect : ℝ) (obs_correct : ℝ) :
  mean_initial = 36 → num_obs = 50 → obs_incorrect = 23 → obs_correct = 30 →
  (mean_initial * ↑num_obs + (obs_correct - obs_incorrect)) / ↑num_obs = 36.14 :=
by
  intros h1 h2 h3 h4
  sorry

end NUMINAMATH_GPT_corrected_mean_l2090_209025


namespace NUMINAMATH_GPT_circle_area_difference_l2090_209067

theorem circle_area_difference (r1 r2 : ℝ) (π : ℝ) (A1 A2 diff : ℝ) 
  (hr1 : r1 = 30)
  (hd2 : 2 * r2 = 30)
  (hA1 : A1 = π * r1^2)
  (hA2 : A2 = π * r2^2)
  (hdiff : diff = A1 - A2) :
  diff = 675 * π :=
by 
  sorry

end NUMINAMATH_GPT_circle_area_difference_l2090_209067


namespace NUMINAMATH_GPT_medians_concurrent_l2090_209052

/--
For any triangle ABC, there exists a point G, known as the centroid, such that
the sum of the vectors from G to each of the vertices A, B, and C is the zero vector.
-/
theorem medians_concurrent 
  (A B C : ℝ×ℝ) : 
  ∃ G : ℝ×ℝ, (G -ᵥ A) + (G -ᵥ B) + (G -ᵥ C) = (0, 0) := 
by 
  -- proof will go here
  sorry 

end NUMINAMATH_GPT_medians_concurrent_l2090_209052


namespace NUMINAMATH_GPT_problem_solution_l2090_209022

theorem problem_solution (b : ℝ) (i : ℂ) (h : i^2 = -1) (h_cond : (2 - i) * (4 * i) = 4 + b * i) : 
  b = 8 := 
by 
  sorry

end NUMINAMATH_GPT_problem_solution_l2090_209022


namespace NUMINAMATH_GPT_oil_vinegar_new_ratio_l2090_209018

theorem oil_vinegar_new_ratio (initial_oil initial_vinegar new_vinegar : ℕ) 
    (h1 : initial_oil / initial_vinegar = 3 / 1)
    (h2 : new_vinegar = (2 * initial_vinegar)) :
    initial_oil / new_vinegar = 3 / 2 :=
by
  sorry

end NUMINAMATH_GPT_oil_vinegar_new_ratio_l2090_209018


namespace NUMINAMATH_GPT_length_of_midsegment_l2090_209037

/-- Given a quadrilateral ABCD where sides AB and CD are parallel with lengths 7 and 3 
    respectively, and the other sides BC and DA are of lengths 5 and 4 respectively, 
    prove that the length of the segment joining the midpoints of sides BC and DA is 5. -/
theorem length_of_midsegment (A B C D : ℝ × ℝ)
  (HAB : A.1 = 0 ∧ A.2 = 0 ∧ B.1 = 7 ∧ B.2 = 0)
  (HBC : dist B C = 5)
  (HCD : dist C D = 3)
  (HDA : dist D A = 4)
  (Hparallel : B.2 = 0 ∧ D.2 ≠ 0 → C.2 = D.2) :
  dist ((B.1 + C.1) / 2, (B.2 + C.2) / 2) ((A.1 + D.1) / 2, (A.2 + D.2) / 2) = 5 :=
sorry

end NUMINAMATH_GPT_length_of_midsegment_l2090_209037


namespace NUMINAMATH_GPT_evaluate_expression_l2090_209004

theorem evaluate_expression (x : ℝ) (h : x = 3) : (x^2 - 3 * x - 10) / (x - 5) = 5 :=
by
  sorry

end NUMINAMATH_GPT_evaluate_expression_l2090_209004


namespace NUMINAMATH_GPT_bamboo_break_height_l2090_209005

-- Conditions provided in the problem
def original_height : ℝ := 20  -- 20 chi
def distance_tip_to_root : ℝ := 6  -- 6 chi

-- Function to check if the height of the break satisfies the equation
def equationHolds (x : ℝ) : Prop :=
  (original_height - x) ^ 2 - x ^ 2 = distance_tip_to_root ^ 2

-- Main statement to prove the height of the break is 9.1 chi
theorem bamboo_break_height : equationHolds 9.1 :=
by
  sorry

end NUMINAMATH_GPT_bamboo_break_height_l2090_209005


namespace NUMINAMATH_GPT_two_sin_cos_15_eq_half_l2090_209070

open Real

theorem two_sin_cos_15_eq_half : 2 * sin (π / 12) * cos (π / 12) = 1 / 2 :=
by
  sorry

end NUMINAMATH_GPT_two_sin_cos_15_eq_half_l2090_209070


namespace NUMINAMATH_GPT_min_positive_announcements_l2090_209019

theorem min_positive_announcements (x y : ℕ) 
  (h1 : x * (x - 1) = 110) 
  (h2 : y * (y - 1) + (x - y) * (x - 1 - (y - 1)) = 50) : 
  y >= 5 := 
sorry

end NUMINAMATH_GPT_min_positive_announcements_l2090_209019


namespace NUMINAMATH_GPT_min_T_tiles_needed_l2090_209099

variable {a b c d : Nat}
variable (total_blocks : Nat := a + b + c + d)
variable (board_size : Nat := 8 * 10)
variable (block_size : Nat := 4)
variable (tile_types := ["T_horizontal", "T_vertical", "S_horizontal", "S_vertical"])
variable (conditions : Prop := total_blocks = 20 ∧ a + c ≥ 5)

theorem min_T_tiles_needed
    (h : conditions)
    (covering : total_blocks * block_size = board_size)
    (T_tiles : a ≥ 6) :
    a = 6 := sorry

end NUMINAMATH_GPT_min_T_tiles_needed_l2090_209099


namespace NUMINAMATH_GPT_find_a9_l2090_209082

variable (S : ℕ → ℚ) (a : ℕ → ℚ) (n : ℕ) (d : ℚ)

-- Conditions
axiom sum_first_six : S 6 = 3
axiom sum_first_eleven : S 11 = 18
axiom Sn_definition : ∀ n, S n = (n : ℚ) / 2 * (a 1 + a n)
axiom arithmetic_sequence : ∀ n, a (n + 1) = a 1 + n * d

-- Problem statement
theorem find_a9 : a 9 = 3 := sorry

end NUMINAMATH_GPT_find_a9_l2090_209082


namespace NUMINAMATH_GPT_number_of_students_l2090_209010

theorem number_of_students 
    (N : ℕ) 
    (h_percentage_5 : 28 * N % 100 = 0)
    (h_percentage_4 : 35 * N % 100 = 0)
    (h_percentage_3 : 25 * N % 100 = 0)
    (h_percentage_2 : 12 * N % 100 = 0)
    (h_class_limit : N ≤ 4 * 30) 
    (h_num_classes : 4 * 30 < 120)
    : N = 100 := 
by 
  sorry

end NUMINAMATH_GPT_number_of_students_l2090_209010


namespace NUMINAMATH_GPT_complement_intersection_l2090_209074

theorem complement_intersection (U M N : Set ℕ) (hU : U = {1, 2, 3, 4, 5, 6, 7, 8}) (hM : M = {1, 3, 5, 7}) (hN : N = {2, 5, 8}) :
  (U \ M) ∩ N = {2, 8} :=
by
  sorry

end NUMINAMATH_GPT_complement_intersection_l2090_209074


namespace NUMINAMATH_GPT_problem_statement_l2090_209081

noncomputable def count_valid_numbers : Nat :=
  let digits := [1, 2, 3, 4, 5]
  let repeated_digit_choices := 5
  let positions_for_repeated_digits := Nat.choose 5 2
  let cases_for_tens_and_hundreds :=
    2 * 3 + 2 + 1
  let two_remaining_digits_permutations := 2
  repeated_digit_choices * positions_for_repeated_digits * cases_for_tens_and_hundreds * two_remaining_digits_permutations

theorem problem_statement : count_valid_numbers = 800 := by
  sorry

end NUMINAMATH_GPT_problem_statement_l2090_209081


namespace NUMINAMATH_GPT_unusual_numbers_exist_l2090_209020

noncomputable def n1 : ℕ := 10 ^ 100 - 1
noncomputable def n2 : ℕ := 10 ^ 100 / 2 - 1

theorem unusual_numbers_exist : 
  (n1 ^ 3 % 10 ^ 100 = n1 ∧ n1 ^ 2 % 10 ^ 100 ≠ n1) ∧ 
  (n2 ^ 3 % 10 ^ 100 = n2 ∧ n2 ^ 2 % 10 ^ 100 ≠ n2) :=
by
  sorry

end NUMINAMATH_GPT_unusual_numbers_exist_l2090_209020


namespace NUMINAMATH_GPT_valid_configuration_exists_l2090_209050

noncomputable def unique_digits (digits: List ℕ) := (digits.length = List.length (List.eraseDup digits)) ∧ ∀ (d : ℕ), d ∈ digits ↔ d ∈ [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

theorem valid_configuration_exists :
  ∃ a b c d e f g h i j : ℕ,
  unique_digits [a, b, c, d, e, f, g, h, i, j] ∧
  a * (100 * b + 10 * c + d) * (100 * e + 10 * f + g) = 1000 * h + 100 * i + 10 * 9 + 71 := 
by
  sorry

end NUMINAMATH_GPT_valid_configuration_exists_l2090_209050


namespace NUMINAMATH_GPT_calculate_three_Z_five_l2090_209054

def Z (a b : ℤ) : ℤ := b + 15 * a - a^3

theorem calculate_three_Z_five : Z 3 5 = 23 :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_GPT_calculate_three_Z_five_l2090_209054


namespace NUMINAMATH_GPT_cos_alpha_solution_l2090_209028

open Real

theorem cos_alpha_solution
  (α : ℝ)
  (h1 : π < α)
  (h2 : α < 3 * π / 2)
  (h3 : tan α = 2) :
  cos α = -sqrt (1 / (1 + 2^2)) :=
by
  sorry

end NUMINAMATH_GPT_cos_alpha_solution_l2090_209028


namespace NUMINAMATH_GPT_yanna_change_l2090_209027

theorem yanna_change :
  let shirt_cost := 5
  let sandal_cost := 3
  let num_shirts := 10
  let num_sandals := 3
  let given_amount := 100
  (given_amount - (num_shirts * shirt_cost + num_sandals * sandal_cost)) = 41 :=
by
  sorry

end NUMINAMATH_GPT_yanna_change_l2090_209027


namespace NUMINAMATH_GPT_esther_biking_speed_l2090_209068

theorem esther_biking_speed (d x : ℝ)
  (h_bike_speed : x > 0)
  (h_average_speed : 5 = 2 * d / (d / x + d / 3)) :
  x = 15 :=
by
  sorry

end NUMINAMATH_GPT_esther_biking_speed_l2090_209068


namespace NUMINAMATH_GPT_abc_value_l2090_209035

theorem abc_value (a b c : ℤ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : c ≠ 0) 
  (h4 : a + b + c = 30) 
  (h5 : (1 : ℚ) / a + (1 : ℚ) / b + (1 : ℚ) / c + 504 / (a * b * c) = 1) :
  a * b * c = 1176 := 
sorry

end NUMINAMATH_GPT_abc_value_l2090_209035


namespace NUMINAMATH_GPT_quadratic_condition_l2090_209072

theorem quadratic_condition (a : ℝ) :
  (∃ x : ℝ, (a - 1) * x^2 + 4 * x - 3 = 0) → a ≠ 1 :=
by
  sorry

end NUMINAMATH_GPT_quadratic_condition_l2090_209072


namespace NUMINAMATH_GPT_perpendicular_vectors_l2090_209031

theorem perpendicular_vectors (a b : ℝ × ℝ) (k : ℝ) (c : ℝ × ℝ) 
  (h1 : a = (1, 2)) (h2 : b = (1, 1)) 
  (h3 : c = (1 + k, 2 + k))
  (h4 : b.1 * c.1 + b.2 * c.2 = 0) : 
  k = -3 / 2 :=
by
  sorry

end NUMINAMATH_GPT_perpendicular_vectors_l2090_209031


namespace NUMINAMATH_GPT_directrix_of_parabola_l2090_209045

-- Define the parabola and the line conditions
def parabola (p : ℝ) := ∀ x y : ℝ, y^2 = 2 * p * x
def focus_line (x y : ℝ) := 2 * x + 3 * y - 8 = 0

-- Theorem stating that the directrix of the parabola is x = -4
theorem directrix_of_parabola (p : ℝ) (hx : ∃ x, ∃ y, focus_line x y) (hp : parabola p) :
  ∃ k : ℝ, k = 4 → ∀ x y : ℝ, (-x) = -4 :=
by
  sorry

end NUMINAMATH_GPT_directrix_of_parabola_l2090_209045


namespace NUMINAMATH_GPT_grid_satisfies_conditions_l2090_209079

-- Define the range of consecutive integers
def consecutive_integers : List ℕ := [5, 6, 7, 8, 9, 10, 11, 12, 13]

-- Define a function to check mutual co-primality condition for two given numbers
def coprime (a b : ℕ) : Prop := Nat.gcd a b = 1

-- Define the 3x3 grid arrangement as a matrix
@[reducible]
def grid : Matrix (Fin 3) (Fin 3) ℕ :=
  ![![8, 9, 10],
    ![5, 7, 11],
    ![6, 13, 12]]

-- Define a predicate to check if the grid cells are mutually coprime for adjacent elements
def mutually_coprime_adjacent (m : Matrix (Fin 3) (Fin 3) ℕ) : Prop :=
    ∀ (i j k l : Fin 3), 
    (|i - k| + |j - l| = 1 ∨ |i - k| = |j - l| ∧ |i - k| = 1) → coprime (m i j) (m k l)

-- The main theorem statement
theorem grid_satisfies_conditions : 
  (∀ (i j : Fin 3), grid i j ∈ consecutive_integers) ∧ 
  mutually_coprime_adjacent grid :=
by
  sorry

end NUMINAMATH_GPT_grid_satisfies_conditions_l2090_209079


namespace NUMINAMATH_GPT_solve_for_a_l2090_209039

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
if h : x >= 0 then 4 ^ x else 2 ^ (a - x)

theorem solve_for_a (a : ℝ) (h : a ≠ 1) (h_eq : f a (1 - a) = f a (a - 1)) : a = 1 / 2 := 
by {
  sorry
}

end NUMINAMATH_GPT_solve_for_a_l2090_209039


namespace NUMINAMATH_GPT_combined_solid_sum_faces_edges_vertices_l2090_209024

noncomputable def prism_faces : ℕ := 6
noncomputable def prism_edges : ℕ := 12
noncomputable def prism_vertices : ℕ := 8
noncomputable def new_pyramid_faces : ℕ := 4
noncomputable def new_pyramid_edges : ℕ := 4
noncomputable def new_pyramid_vertex : ℕ := 1

theorem combined_solid_sum_faces_edges_vertices :
  prism_faces - 1 + new_pyramid_faces + prism_edges + new_pyramid_edges + prism_vertices + new_pyramid_vertex = 34 :=
by
  -- proof would go here
  sorry

end NUMINAMATH_GPT_combined_solid_sum_faces_edges_vertices_l2090_209024


namespace NUMINAMATH_GPT_binom_1300_2_eq_844350_l2090_209041

theorem binom_1300_2_eq_844350 : Nat.choose 1300 2 = 844350 :=
by
  sorry

end NUMINAMATH_GPT_binom_1300_2_eq_844350_l2090_209041


namespace NUMINAMATH_GPT_find_number_l2090_209034

theorem find_number :
  (∃ x : ℝ, x * (3 + Real.sqrt 5) = 1) ∧ (x = (3 - Real.sqrt 5) / 4) :=
sorry

end NUMINAMATH_GPT_find_number_l2090_209034


namespace NUMINAMATH_GPT_ten_men_ten_boys_work_time_l2090_209058

theorem ten_men_ten_boys_work_time :
  (∀ (total_work : ℝ) (man_work_rate boy_work_rate : ℝ),
    15 * 10 * man_work_rate = total_work ∧
    20 * 15 * boy_work_rate = total_work →
    (10 * man_work_rate + 10 * boy_work_rate) * 10 = total_work) :=
by
  sorry

end NUMINAMATH_GPT_ten_men_ten_boys_work_time_l2090_209058


namespace NUMINAMATH_GPT_age_difference_proof_l2090_209011

def AlexAge : ℝ := 16.9996700066
def AlexFatherAge (A : ℝ) (F : ℝ) : Prop := F = 2 * A + 4.9996700066
def FatherAgeSixYearsAgo (A : ℝ) (F : ℝ) : Prop := A - 6 = 1 / 3 * (F - 6)

theorem age_difference_proof :
  ∃ (A F : ℝ), A = 16.9996700066 ∧
  (AlexFatherAge A F) ∧
  (FatherAgeSixYearsAgo A F) :=
by
  sorry

end NUMINAMATH_GPT_age_difference_proof_l2090_209011


namespace NUMINAMATH_GPT_find_abscissas_l2090_209097

theorem find_abscissas (x_A x_B : ℝ) (y_A y_B : ℝ) : 
  ((y_A = x_A^2) ∧ (y_B = x_B^2) ∧ (0, 15) = (0,  (5 * y_B + 3 * y_A) / 8) ∧ (5 * x_B + 3 * x_A = 0)) → 
  ((x_A = -5 ∧ x_B = 3) ∨ (x_A = 5 ∧ x_B = -3)) :=
by
  sorry

end NUMINAMATH_GPT_find_abscissas_l2090_209097


namespace NUMINAMATH_GPT_correct_exponential_calculation_l2090_209008

theorem correct_exponential_calculation (a : ℝ) (ha : a ≠ 0) : 
  (a^4)^4 = a^16 :=
by sorry

end NUMINAMATH_GPT_correct_exponential_calculation_l2090_209008


namespace NUMINAMATH_GPT_polynomial_representation_l2090_209016

noncomputable def given_expression (x : ℝ) : ℝ :=
  (3 * x^2 + 4 * x + 8) * (x - 2) - (x - 2) * (x^2 + 5 * x - 72) + (4 * x - 15) * (x - 2) * (x + 6)

theorem polynomial_representation (x : ℝ) :
  given_expression x = 6 * x^3 - 4 * x^2 - 26 * x + 20 :=
sorry

end NUMINAMATH_GPT_polynomial_representation_l2090_209016


namespace NUMINAMATH_GPT_find_t_l2090_209087

def utility (hours_math hours_reading hours_painting : ℕ) : ℕ :=
  hours_math^2 + hours_reading * hours_painting

def utility_wednesday (t : ℕ) : ℕ :=
  utility 4 t (12 - t)

def utility_thursday (t : ℕ) : ℕ :=
  utility 3 (t + 1) (11 - t)

theorem find_t (t : ℕ) (h : utility_wednesday t = utility_thursday t) : t = 2 :=
by
  sorry

end NUMINAMATH_GPT_find_t_l2090_209087


namespace NUMINAMATH_GPT_johns_commute_distance_l2090_209095

theorem johns_commute_distance
  (y : ℝ)  -- distance in miles
  (h1 : 200 * (y / 200) = y)  -- John usually takes 200 minutes, so usual speed is y/200 miles per minute
  (h2 : 320 = (y / (2 * (y / 200))) + (y / (2 * ((y / 200) - 15/60)))) -- Total journey time on the foggy day
  : y = 92 :=
sorry

end NUMINAMATH_GPT_johns_commute_distance_l2090_209095


namespace NUMINAMATH_GPT_pascal_no_divisible_by_prime_iff_form_l2090_209032

theorem pascal_no_divisible_by_prime_iff_form (p : ℕ) (n : ℕ) 
  (hp : Nat.Prime p) :
  (∀ k ≤ n, Nat.choose n k % p ≠ 0) ↔ ∃ s q : ℕ, s ≥ 0 ∧ 0 < q ∧ q < p ∧ n = p^s * q - 1 :=
by
  sorry

end NUMINAMATH_GPT_pascal_no_divisible_by_prime_iff_form_l2090_209032


namespace NUMINAMATH_GPT_rectangle_new_area_l2090_209063

theorem rectangle_new_area (l w : ℝ) (h_area : l * w = 540) : 
  (1.15 * l) * (0.8 * w) = 497 :=
by
  sorry

end NUMINAMATH_GPT_rectangle_new_area_l2090_209063


namespace NUMINAMATH_GPT_find_m_l2090_209055

def is_ellipse (x y m : ℝ) : Prop :=
  (x^2 / (m + 1) + y^2 / m = 1)

def has_eccentricity (e : ℝ) (m : ℝ) : Prop :=
  e = Real.sqrt (1 - m / (m + 1))

theorem find_m (m : ℝ) (h_m : m > 0) (h_ellipse : ∀ x y, is_ellipse x y m) (h_eccentricity : has_eccentricity (1 / 2) m) : m = 3 :=
by
  sorry

end NUMINAMATH_GPT_find_m_l2090_209055


namespace NUMINAMATH_GPT_four_corresponds_to_364_l2090_209088

noncomputable def number_pattern (n : ℕ) : ℕ :=
  match n with
  | 1 => 6
  | 2 => 36
  | 3 => 363
  | 5 => 365
  | 36 => 2
  | _ => 0 -- Assume 0 as the default case

theorem four_corresponds_to_364 : number_pattern 4 = 364 :=
sorry

end NUMINAMATH_GPT_four_corresponds_to_364_l2090_209088


namespace NUMINAMATH_GPT_factorization_correct_l2090_209064

theorem factorization_correct (a : ℝ) : a^2 - 2 * a - 15 = (a + 3) * (a - 5) := 
by
  sorry

end NUMINAMATH_GPT_factorization_correct_l2090_209064


namespace NUMINAMATH_GPT_determine_selling_price_for_daily_profit_determine_max_profit_and_selling_price_l2090_209026

-- Cost price per souvenir
def cost_price : ℕ := 40

-- Minimum selling price
def min_selling_price : ℕ := 44

-- Maximum selling price
def max_selling_price : ℕ := 60

-- Units sold if selling price is min_selling_price
def units_sold_at_min_price : ℕ := 300

-- Units sold decreases by 10 for every 1 yuan increase in selling price
def decrease_in_units (increase : ℕ) : ℕ := 10 * increase

-- Daily profit for a given increase in selling price
def daily_profit (increase : ℕ) : ℕ := (increase + min_selling_price - cost_price) * (units_sold_at_min_price - decrease_in_units increase)

-- Maximum profit calculation
def maximizing_daily_profit (increase : ℕ) : ℕ := (increase + min_selling_price - cost_price) * (units_sold_at_min_price - decrease_in_units increase) 

-- Statement for Problem Part 1
theorem determine_selling_price_for_daily_profit : ∃ P, P = 52 ∧ daily_profit (P - min_selling_price) = 2640 := 
sorry

-- Statement for Problem Part 2
theorem determine_max_profit_and_selling_price : ∃ P, P = 57 ∧ maximizing_daily_profit (P - min_selling_price) = 2890 := 
sorry

end NUMINAMATH_GPT_determine_selling_price_for_daily_profit_determine_max_profit_and_selling_price_l2090_209026


namespace NUMINAMATH_GPT_expected_amoebas_after_one_week_l2090_209009

section AmoebaProblem

-- Definitions from conditions
def initial_amoebas : ℕ := 1
def split_probability : ℝ := 0.8
def days : ℕ := 7

-- Function to calculate expected amoebas
def expected_amoebas (n : ℕ) : ℝ :=
  initial_amoebas * ((2 : ℝ) ^ n) * (split_probability ^ n)

-- Theorem statement
theorem expected_amoebas_after_one_week :
  expected_amoebas days = 26.8435456 :=
by sorry

end AmoebaProblem

end NUMINAMATH_GPT_expected_amoebas_after_one_week_l2090_209009


namespace NUMINAMATH_GPT_pythagorean_theorem_l2090_209053

theorem pythagorean_theorem (a b c : ℕ) (h : a^2 + b^2 = c^2) : a^2 + b^2 = c^2 :=
by
  sorry

end NUMINAMATH_GPT_pythagorean_theorem_l2090_209053


namespace NUMINAMATH_GPT_solution_1_solution_2_l2090_209059

noncomputable def f (x a : ℝ) : ℝ := |x - a| - |x - 3|

theorem solution_1 (x : ℝ) : (f x (-1) >= 2) ↔ (x >= 2) :=
by
  sorry

theorem solution_2 (a : ℝ) : 
  (∃ x : ℝ, f x a <= -(a / 2)) ↔ (a <= 2 ∨ a >= 6) :=
by
  sorry

end NUMINAMATH_GPT_solution_1_solution_2_l2090_209059


namespace NUMINAMATH_GPT_quadratic_roots_range_l2090_209086

theorem quadratic_roots_range (m : ℝ) :
  (∃ x : ℝ, x^2 - (2 * m + 1) * x + m^2 = 0 ∧ (∃ y : ℝ, y ≠ x ∧ y^2 - (2 * m + 1) * y + m^2 = 0)) ↔ m > -1 / 4 :=
by sorry

end NUMINAMATH_GPT_quadratic_roots_range_l2090_209086


namespace NUMINAMATH_GPT_boxes_with_no_items_l2090_209021

-- Definitions of each condition as given in the problem
def total_boxes : Nat := 15
def pencil_boxes : Nat := 8
def pen_boxes : Nat := 5
def marker_boxes : Nat := 3
def pen_pencil_boxes : Nat := 4
def all_three_boxes : Nat := 1

-- The theorem to prove
theorem boxes_with_no_items : 
     (total_boxes - ((pen_pencil_boxes - all_three_boxes)
                     + (pencil_boxes - pen_pencil_boxes - all_three_boxes)
                     + (pen_boxes - pen_pencil_boxes - all_three_boxes)
                     + (marker_boxes - all_three_boxes)
                     + all_three_boxes)) = 5 := 
by 
  -- This is where the proof would go, but we'll use sorry to indicate it's skipped.
  sorry

end NUMINAMATH_GPT_boxes_with_no_items_l2090_209021
