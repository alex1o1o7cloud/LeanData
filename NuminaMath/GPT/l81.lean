import Mathlib

namespace cylinder_volume_l81_8121

variables (a : ℝ) (π_ne_zero : π ≠ 0) (two_ne_zero : 2 ≠ 0) 

theorem cylinder_volume (h1 : ∃ (h r : ℝ), (2 * π * r = 2 * a ∧ h = a) 
                        ∨ (2 * π * r = a ∧ h = 2 * a)) :
  (∃ (V : ℝ), V = a^3 / π) ∨ (∃ (V : ℝ), V = a^3 / (2 * π)) :=
by
  sorry

end cylinder_volume_l81_8121


namespace water_left_in_bucket_l81_8193

theorem water_left_in_bucket :
  ∀ (original_poured water_left : ℝ),
    original_poured = 0.8 →
    water_left = 0.6 →
    ∃ (poured : ℝ), poured = 0.2 ∧ original_poured - poured = water_left :=
by
  intros original_poured water_left ho hw
  apply Exists.intro 0.2
  simp [ho, hw]
  sorry

end water_left_in_bucket_l81_8193


namespace not_function_age_height_l81_8155

theorem not_function_age_height (f : ℕ → ℝ) :
  ¬(∀ (a b : ℕ), a = b → f a = f b) := sorry

end not_function_age_height_l81_8155


namespace parabola_focus_distance_l81_8171

theorem parabola_focus_distance (p : ℝ) (h_pos : p > 0) (A : ℝ × ℝ)
  (h_A_on_parabola : A.2 = 5 ∧ A.1^2 = 2 * p * A.2)
  (h_AF : abs (A.2 - (p / 2)) = 8) : p = 6 :=
by
  sorry

end parabola_focus_distance_l81_8171


namespace part1_real_roots_part2_specific_roots_l81_8136

-- Part 1: Real roots condition
theorem part1_real_roots (m : ℝ) (h : ∃ x : ℝ, x^2 + (2 * m - 1) * x + m^2 = 0) : m ≤ 1/4 :=
by sorry

-- Part 2: Specific roots condition
theorem part2_specific_roots (m : ℝ) (x1 x2 : ℝ) 
  (h1 : x1^2 + (2 * m - 1) * x1 + m^2 = 0) 
  (h2 : x2^2 + (2 * m - 1) * x2 + m^2 = 0) 
  (h3 : x1 * x2 + x1 + x2 = 4) : m = -1 :=
by sorry

end part1_real_roots_part2_specific_roots_l81_8136


namespace jerry_claim_percentage_l81_8169

theorem jerry_claim_percentage
  (salary_years : ℕ)
  (annual_salary : ℕ)
  (medical_bills : ℕ)
  (punitive_multiplier : ℕ)
  (received_amount : ℕ)
  (total_claim : ℕ)
  (percentage_claim : ℕ) :
  salary_years = 30 →
  annual_salary = 50000 →
  medical_bills = 200000 →
  punitive_multiplier = 3 →
  received_amount = 5440000 →
  total_claim = (annual_salary * salary_years) + medical_bills + (punitive_multiplier * ((annual_salary * salary_years) + medical_bills)) →
  percentage_claim = (received_amount * 100) / total_claim →
  percentage_claim = 80 :=
by
  sorry

end jerry_claim_percentage_l81_8169


namespace range_of_m_l81_8102

theorem range_of_m {x m : ℝ} (h : ∀ x, x^2 - 2*x + 2*m - 1 ≥ 0) : m ≥ 1 :=
sorry

end range_of_m_l81_8102


namespace marlon_goals_l81_8113

theorem marlon_goals :
  ∃ g : ℝ,
    (∀ p f : ℝ, p + f = 40 → g = 0.4 * p + 0.5 * f) → g = 20 :=
by
  sorry

end marlon_goals_l81_8113


namespace coins_remainder_l81_8179

theorem coins_remainder (n : ℕ) (h1 : n % 8 = 6) (h2 : n % 7 = 5) : 
  (∃ m : ℕ, (n = m * 9)) :=
sorry

end coins_remainder_l81_8179


namespace shaded_area_l81_8134

-- Define the points as per the problem
structure Point where
  x : ℝ
  y : ℝ

@[simp]
def A : Point := ⟨0, 0⟩
@[simp]
def B : Point := ⟨0, 7⟩
@[simp]
def C : Point := ⟨7, 7⟩
@[simp]
def D : Point := ⟨7, 0⟩
@[simp]
def E : Point := ⟨7, 0⟩
@[simp]
def F : Point := ⟨14, 0⟩
@[simp]
def G : Point := ⟨10.5, 7⟩

-- Define function for area of a triangle given three points
def triangle_area (P Q R : Point) : ℝ :=
  0.5 * abs ((P.x - R.x) * (Q.y - P.y) - (P.x - Q.x) * (R.y - P.y))

-- The theorem stating the area of the shaded region
theorem shaded_area : triangle_area D G H - triangle_area D E H = 24.5 := by
  sorry

end shaded_area_l81_8134


namespace age_6_not_child_l81_8109

-- Definition and assumptions based on the conditions
def billboard_number : ℕ := 5353
def mr_smith_age : ℕ := 53
def children_ages : List ℕ := [1, 2, 3, 4, 5, 7, 8, 9, 10, 11] -- Excluding age 6

-- The theorem to prove that the age 6 is not one of Mr. Smith's children's ages.
theorem age_6_not_child :
  (billboard_number ≡ 53 * 101 [MOD 10^4]) ∧
  (∀ age ∈ children_ages, billboard_number % age = 0) ∧
  oldest_child_age = 11 → ¬(6 ∈ children_ages) :=
sorry

end age_6_not_child_l81_8109


namespace total_digits_in_numbering_pages_l81_8195

theorem total_digits_in_numbering_pages (n : ℕ) (h : n = 100000) : 
  let digits1 := 9 * 1
  let digits2 := (99 - 10 + 1) * 2
  let digits3 := (999 - 100 + 1) * 3
  let digits4 := (9999 - 1000 + 1) * 4
  let digits5 := (99999 - 10000 + 1) * 5
  let digits6 := 6
  (digits1 + digits2 + digits3 + digits4 + digits5 + digits6) = 488895 :=
by
  sorry

end total_digits_in_numbering_pages_l81_8195


namespace retail_price_per_book_l81_8116

theorem retail_price_per_book (n r w : ℝ)
  (h1 : r * n = 48)
  (h2 : w = r - 2)
  (h3 : w * (n + 4) = 48) :
  r = 6 := by
  sorry

end retail_price_per_book_l81_8116


namespace new_area_of_rectangle_l81_8197

theorem new_area_of_rectangle (L W : ℝ) (h : L * W = 600) :
  let new_length := 0.8 * L
  let new_width := 1.05 * W
  new_length * new_width = 504 :=
by 
  sorry

end new_area_of_rectangle_l81_8197


namespace problem_solution_l81_8119

theorem problem_solution :
  -20 + 7 * (8 - 2 / 2) = 29 :=
by 
  sorry

end problem_solution_l81_8119


namespace base3_sum_l81_8174

theorem base3_sum : 
  (1 * 3^0 - 2 * 3^1 - 2 * 3^0 + 2 * 3^2 + 1 * 3^1 - 1 * 3^0 - 1 * 3^3) = (2 * 3^2 + 1 * 3^1 + 0 * 3^0) := 
by 
  sorry

end base3_sum_l81_8174


namespace sealed_envelope_problem_l81_8183

theorem sealed_envelope_problem :
  ∃ (n : ℕ), (10 ≤ n ∧ n < 100) →
  ((n = 12 ∧ (n % 10 ≠ 2) ∧ n ≠ 35 ∧ (n % 10 ≠ 5)) ∨
   (n ≠ 12 ∧ (n % 10 ≠ 2) ∧ n = 35 ∧ (n % 10 = 5))) →
  ¬(n % 10 ≠ 5) :=
by
  sorry

end sealed_envelope_problem_l81_8183


namespace amount_received_from_mom_l81_8154

-- Defining the problem conditions
def receives_from_dad : ℕ := 5
def spends : ℕ := 4
def has_more_from_mom_after_spending (M : ℕ) : Prop := 
  (receives_from_dad + M - spends = receives_from_dad + 2)

-- Lean theorem statement
theorem amount_received_from_mom (M : ℕ) (h : has_more_from_mom_after_spending M) : M = 6 := 
by
  sorry

end amount_received_from_mom_l81_8154


namespace profit_equation_example_l81_8182

noncomputable def profit_equation (a b : ℝ) (x : ℝ) : Prop :=
  a * (1 + x) ^ 2 = b

theorem profit_equation_example :
  profit_equation 250 360 x :=
by
  have : 25 * (1 + x) ^ 2 = 36 := sorry
  sorry

end profit_equation_example_l81_8182


namespace sum_of_cubes_l81_8148

theorem sum_of_cubes (a b : ℝ) (h1 : a + b = 3) (h2 : a * b = 2) : a^3 + b^3 = 9 :=
by
  sorry

end sum_of_cubes_l81_8148


namespace polynomial_multiplication_l81_8161

theorem polynomial_multiplication :
  (5 * X^2 + 3 * X - 4) * (2 * X^3 + X^2 - X + 1) = 
  (10 * X^5 + 11 * X^4 - 10 * X^3 - 2 * X^2 + 7 * X - 4) := 
by {
  sorry
}

end polynomial_multiplication_l81_8161


namespace felicity_gasoline_usage_l81_8172

def gallons_of_gasoline (G D: ℝ) :=
  G = 2 * D

def combined_volume (M D: ℝ) :=
  M = D - 5

def ethanol_consumption (E M: ℝ) :=
  E = 0.35 * M

def biodiesel_consumption (B M: ℝ) :=
  B = 0.65 * M

def distance_relationship_F_A (F A: ℕ) :=
  A = F + 150

def distance_relationship_F_Bn (F Bn: ℕ) :=
  F = Bn + 50

def total_distance (F A Bn: ℕ) :=
  F + A + Bn = 1750

def gasoline_mileage : ℕ := 35

def diesel_mileage : ℕ := 25

def ethanol_mileage : ℕ := 30

def biodiesel_mileage : ℕ := 20

theorem felicity_gasoline_usage : 
  ∀ (F A Bn: ℕ) (G D M E B: ℝ),
  gallons_of_gasoline G D →
  combined_volume M D →
  ethanol_consumption E M →
  biodiesel_consumption B M →
  distance_relationship_F_A F A →
  distance_relationship_F_Bn F Bn →
  total_distance F A Bn →
  G = 56
  := by
    intros
    sorry

end felicity_gasoline_usage_l81_8172


namespace cost_of_meatballs_is_five_l81_8114

-- Define the conditions
def cost_of_pasta : ℕ := 1
def cost_of_sauce : ℕ := 2
def total_cost_of_meal (servings : ℕ) (cost_per_serving : ℕ) : ℕ := servings * cost_per_serving

-- Define the cost of meatballs calculation
def cost_of_meatballs (total_cost pasta_cost sauce_cost : ℕ) : ℕ :=
  total_cost - pasta_cost - sauce_cost

-- State the theorem we want to prove
theorem cost_of_meatballs_is_five :
  cost_of_meatballs (total_cost_of_meal 8 1) cost_of_pasta cost_of_sauce = 5 :=
by
  -- This part will include the proof steps
  sorry

end cost_of_meatballs_is_five_l81_8114


namespace not_divisible_by_5_for_4_and_7_l81_8173

-- Define a predicate that checks if a given number is not divisible by another number
def notDivisibleBy (n k : ℕ) : Prop := ¬ (n % k = 0)

-- Define the expression we are interested in
def expression (b : ℕ) : ℕ := 3 * b^3 - b^2 + b - 1

-- The theorem we want to prove
theorem not_divisible_by_5_for_4_and_7 :
  notDivisibleBy (expression 4) 5 ∧ notDivisibleBy (expression 7) 5 :=
by
  sorry

end not_divisible_by_5_for_4_and_7_l81_8173


namespace avg_people_moving_to_florida_per_hour_l81_8128

theorem avg_people_moving_to_florida_per_hour (people : ℕ) (days : ℕ) (hours_per_day : ℕ) 
  (h1 : people = 3000) (h2 : days = 5) (h3 : hours_per_day = 24) : 
  people / (days * hours_per_day) = 25 := by
  sorry

end avg_people_moving_to_florida_per_hour_l81_8128


namespace least_froods_l81_8115

theorem least_froods (n : ℕ) :
  (∃ n, n ≥ 1 ∧ (n * (n + 1)) / 2 > 20 * n) → (∃ n, n = 40) :=
by {
  sorry
}

end least_froods_l81_8115


namespace smallest_x_plus_y_l81_8104

theorem smallest_x_plus_y (x y : ℕ) (h1 : x ≠ y) (h2 : x > 0) (h3 : y > 0) 
(h4 : 1 / (x:ℝ) + 1 / (y:ℝ) = 1 / 12) : x + y = 49 :=
sorry

end smallest_x_plus_y_l81_8104


namespace sum_base9_to_base9_eq_l81_8192

-- Definition of base 9 numbers
def base9_to_base10 (n : ℕ) : ℕ :=
  let digit1 := n % 10
  let digit2 := (n / 10) % 10
  let digit3 := (n / 100) % 10
  digit1 + 9 * digit2 + 81 * digit3

-- Definition of base 10 to base 9 conversion
def base10_to_base9 (n : ℕ) : ℕ :=
  let digit1 := n % 9
  let digit2 := (n / 9) % 9
  let digit3 := (n / 81) % 9
  digit1 + 10 * digit2 + 100 * digit3

-- The theorem to prove
theorem sum_base9_to_base9_eq :
  let x := base9_to_base10 236
  let y := base9_to_base10 327
  let z := base9_to_base10 284
  base10_to_base9 (x + y + z) = 858 :=
by {
  sorry
}

end sum_base9_to_base9_eq_l81_8192


namespace find_f4_l81_8190

-- Let f be a function from ℝ to ℝ with the following properties:
variable (f : ℝ → ℝ)

-- 1. f(x + 1) is an odd function
axiom f_odd : ∀ x, f (-(x + 1)) = -f (x + 1)

-- 2. f(x - 1) is an even function
axiom f_even : ∀ x, f (-(x - 1)) = f (x - 1)

-- 3. f(0) = 2
axiom f_zero : f 0 = 2

-- Prove that f(4) = -2
theorem find_f4 : f 4 = -2 :=
by
  sorry

end find_f4_l81_8190


namespace exists_power_of_two_with_consecutive_zeros_l81_8126

theorem exists_power_of_two_with_consecutive_zeros (k : ℕ) (hk : k ≥ 1) :
  ∃ n : ℕ, ∃ a b : ℕ, ∃ m : ℕ, 2^n = a * 10^(m + k) + b ∧ 10^(k - 1) ≤ b ∧ b < 10^k ∧ a % 10 ≠ 0 ∧ b % 10 ≠ 0 :=
sorry

end exists_power_of_two_with_consecutive_zeros_l81_8126


namespace draw_probability_l81_8156

variable (P_lose_a win_a : ℝ)
variable (not_lose_a : ℝ := 0.8)
variable (win_prob_a : ℝ := 0.6)

-- Given conditions
def A_not_losing : Prop := not_lose_a = win_prob_a + win_a

-- Main theorem to prove
theorem draw_probability : P_lose_a = 0.2 :=
by
  sorry

end draw_probability_l81_8156


namespace power_function_quadrant_IV_l81_8120

theorem power_function_quadrant_IV (a : ℝ) (h : a ∈ ({-1, 1/2, 2, 3} : Set ℝ)) :
  ∀ x : ℝ, x * x^a ≠ -x * (-x^a) := sorry

end power_function_quadrant_IV_l81_8120


namespace inequality_solution_l81_8108

theorem inequality_solution (x : ℝ) (hx1 : x ≥ -1/2) (hx2 : x ≠ 0) :
  (4 * x^2 / (1 - Real.sqrt (1 + 2 * x))^2 < 2 * x + 9) ↔ 
  (-1/2 ≤ x ∧ x < 0) ∨ (0 < x ∧ x < 45/8) :=
by
  sorry

end inequality_solution_l81_8108


namespace tip_calculation_correct_l81_8137

noncomputable def calculate_tip (total_with_tax : ℝ) (tax_rate : ℝ) (tip_rate : ℝ) : ℝ :=
  let bill_before_tax := total_with_tax / (1 + tax_rate)
  bill_before_tax * tip_rate

theorem tip_calculation_correct :
  calculate_tip 226 0.13 0.15 = 30 := 
by
  sorry

end tip_calculation_correct_l81_8137


namespace chef_dressing_total_volume_l81_8168

theorem chef_dressing_total_volume :
  ∀ (V1 V2 : ℕ) (P1 P2 : ℕ) (total_amount : ℕ),
    V1 = 128 →
    V2 = 128 →
    P1 = 8 →
    P2 = 13 →
    total_amount = V1 + V2 →
    total_amount = 256 :=
by
  intros V1 V2 P1 P2 total_amount hV1 hV2 hP1 hP2 h_total
  rw [hV1, hV2, add_comm, add_comm] at h_total
  exact h_total

end chef_dressing_total_volume_l81_8168


namespace max_x_plus_y_range_y_plus_1_over_x_extrema_x2_minus_2x_plus_y2_plus_1_l81_8147

namespace Geometry

variables {x y : ℝ}

-- Given condition
def satisfies_circle (x y : ℝ) : Prop := x^2 + y^2 - 4 * y + 1 = 0

-- Proof problems
theorem max_x_plus_y (h : satisfies_circle x y) : 
  x + y ≤ 2 + Real.sqrt 6 :=
sorry

theorem range_y_plus_1_over_x (h : satisfies_circle x y) : 
  -Real.sqrt 2 ≤ (y + 1) / x ∧ (y + 1) / x ≤ Real.sqrt 2 :=
sorry

theorem extrema_x2_minus_2x_plus_y2_plus_1 (h : satisfies_circle x y) : 
  8 - 2 * Real.sqrt 15 ≤ x^2 - 2 * x + y^2 + 1 ∧ x^2 - 2 * x + y^2 + 1 ≤ 8 + 2 * Real.sqrt 15 :=
sorry

end Geometry

end max_x_plus_y_range_y_plus_1_over_x_extrema_x2_minus_2x_plus_y2_plus_1_l81_8147


namespace find_g7_l81_8184

-- Given the required functional equation and specific value g(6) = 7
theorem find_g7 (g : ℝ → ℝ) (H1 : ∀ x y : ℝ, g (x + y) = g x + g y) (H2 : g 6 = 7) : g 7 = 49 / 6 := by
  sorry

end find_g7_l81_8184


namespace trapezoid_upper_side_length_l81_8163

theorem trapezoid_upper_side_length (area base1 height : ℝ) (h1 : area = 222) (h2 : base1 = 23) (h3 : height = 12) : 
  ∃ base2, base2 = 14 :=
by
  -- The proof will be provided here.
  sorry

end trapezoid_upper_side_length_l81_8163


namespace greatest_integer_value_l81_8118

theorem greatest_integer_value (x : ℤ) (h : 3 * |x| + 4 ≤ 19) : x ≤ 5 :=
by
  sorry

end greatest_integer_value_l81_8118


namespace Carl_typing_words_l81_8140

variable (typingSpeed : ℕ) (hoursPerDay : ℕ) (days : ℕ)

theorem Carl_typing_words (h1 : typingSpeed = 50) (h2 : hoursPerDay = 4) (h3 : days = 7) :
  (typingSpeed * 60 * hoursPerDay * days) = 84000 := by
  sorry

end Carl_typing_words_l81_8140


namespace move_left_is_negative_l81_8151

theorem move_left_is_negative (movement_right : ℝ) (h : movement_right = 3) : -movement_right = -3 := 
by 
  sorry

end move_left_is_negative_l81_8151


namespace product_of_ab_l81_8185

theorem product_of_ab (a b : ℝ) (h1 : a + b = 3) (h2 : a - b = 7) : a * b = -10 :=
by
  sorry

end product_of_ab_l81_8185


namespace initial_sheep_count_l81_8188

theorem initial_sheep_count 
    (S : ℕ)
    (initial_horses : ℕ := 100)
    (initial_chickens : ℕ := 9)
    (gifted_goats : ℕ := 37)
    (male_animals : ℕ := 53)
    (total_animals_half : ℕ := 106) :
    ((initial_horses + S + initial_chickens) / 2 + gifted_goats = total_animals_half) → 
    S = 29 :=
by
  intro h
  sorry

end initial_sheep_count_l81_8188


namespace find_M_l81_8122

theorem find_M (M : ℕ) (h1 : M > 0) (h2 : M < 10) : 
  5 ∣ (1989^M + M^1989) ↔ M = 1 ∨ M = 4 := by
  sorry

end find_M_l81_8122


namespace find_g_of_conditions_l81_8186

theorem find_g_of_conditions (g : ℝ → ℝ)
  (h : ∀ x y : ℝ, x * g y = 2 * y * g x)
  (g_10 : g 10 = 15) : g 2 = 6 :=
sorry

end find_g_of_conditions_l81_8186


namespace population_of_missing_village_l81_8162

theorem population_of_missing_village 
  (p1 p2 p3 p4 p5 p6 : ℕ) 
  (h1 : p1 = 803) 
  (h2 : p2 = 900) 
  (h3 : p3 = 1100) 
  (h4 : p4 = 1023) 
  (h5 : p5 = 945) 
  (h6 : p6 = 1249) 
  (avg_population : ℕ) 
  (h_avg : avg_population = 1000) :
  ∃ p7 : ℕ, p7 = 980 ∧ avg_population * 7 = p1 + p2 + p3 + p4 + p5 + p6 + p7 :=
by
  sorry

end population_of_missing_village_l81_8162


namespace interest_rate_calc_l81_8105

theorem interest_rate_calc
  (P : ℝ) (A : ℝ) (T : ℝ) (SI : ℝ := A - P)
  (R : ℝ := (SI * 100) / (P * T))
  (hP : P = 750)
  (hA : A = 950)
  (hT : T = 5) :
  R = 5.33 :=
by
  sorry

end interest_rate_calc_l81_8105


namespace cos_pi_over_3_plus_2theta_l81_8138

theorem cos_pi_over_3_plus_2theta 
  (theta : ℝ)
  (h : Real.sin (Real.pi / 3 - theta) = 3 / 4) : 
  Real.cos (Real.pi / 3 + 2 * theta) = 1 / 8 :=
by 
  sorry

end cos_pi_over_3_plus_2theta_l81_8138


namespace sum_of_numbers_l81_8110

/-- Given three numbers in the ratio 1:2:5, with the sum of their squares being 4320,
prove that the sum of the numbers is 96. -/

theorem sum_of_numbers (x : ℝ) (h1 : (x:ℝ) = x) (h2 : 2 * x = 2 * x) (h3 : 5 * x = 5 * x) 
  (h4 : x^2 + (2 * x)^2 + (5 * x)^2 = 4320) :
  x + 2 * x + 5 * x = 96 := 
sorry

end sum_of_numbers_l81_8110


namespace least_value_of_x_l81_8100

theorem least_value_of_x (x : ℝ) : (4 * x^2 + 8 * x + 3 = 1) → (-1 ≤ x) :=
by
  intro h
  sorry

end least_value_of_x_l81_8100


namespace rectangle_area_l81_8139

theorem rectangle_area (x y : ℝ) (h1 : x + y = 5) (h2 : x^2 + y^2 = 15) : x * y = 5 :=
by
  -- Conditions given to us:
  -- 1. (h1) The sum of the sides is 5.
  -- 2. (h2) The sum of the squares of the sides is 15.
  -- We need to prove that the product of the sides is 5.
  sorry

end rectangle_area_l81_8139


namespace rate_of_rainfall_is_one_l81_8158

variable (R : ℝ)
variable (h1 : 2 + 4 * R + 4 * 3 = 18)

theorem rate_of_rainfall_is_one : R = 1 :=
by
  sorry

end rate_of_rainfall_is_one_l81_8158


namespace find_exponent_l81_8117

theorem find_exponent (y : ℝ) (exponent : ℝ) :
  (12^1 * 6^exponent / 432 = y) → (y = 36) → (exponent = 3) :=
by 
  intros h₁ h₂ 
  sorry

end find_exponent_l81_8117


namespace smallest_n_product_exceeds_l81_8130

theorem smallest_n_product_exceeds (n : ℕ) : (5 : ℝ) ^ (n * (n + 1) / 14) > 1000 ↔ n = 7 :=
by sorry

end smallest_n_product_exceeds_l81_8130


namespace square_diagonal_length_l81_8135

theorem square_diagonal_length (rect_length rect_width : ℝ) 
  (h1 : rect_length = 45) 
  (h2 : rect_width = 40) 
  (rect_area := rect_length * rect_width) 
  (square_area := rect_area) 
  (side_length := Real.sqrt square_area) 
  (diagonal := side_length * Real.sqrt 2) :
  diagonal = 60 :=
by
  -- Proof goes here
  sorry

end square_diagonal_length_l81_8135


namespace integer_solutions_l81_8167

theorem integer_solutions (m : ℤ) :
  (∃ x : ℤ, (m * x - 1) / (x - 1) = 2 + 1 / (1 - x)) → 
  (∃ x : ℝ, (m - 1) * x^2 + 2 * x + 1 / 2 = 0) →
  m = 3 :=
by
  sorry

end integer_solutions_l81_8167


namespace johns_total_payment_l81_8177

theorem johns_total_payment :
  let silverware_cost := 20
  let dinner_plate_cost := 0.5 * silverware_cost
  let total_cost := dinner_plate_cost + silverware_cost
  total_cost = 30 := sorry

end johns_total_payment_l81_8177


namespace S_8_arithmetic_sequence_l81_8191

theorem S_8_arithmetic_sequence (a : ℕ → ℝ) (S : ℕ → ℝ)
  (h1 : ∀ n, S n = (n * (a 1 + a n)) / 2)
  (h2 : a 4 = 18 - a 5):
  S 8 = 72 :=
by
  sorry

end S_8_arithmetic_sequence_l81_8191


namespace line_equation_l81_8196

theorem line_equation {x y : ℝ} (h : (x = 1) ∧ (y = -3)) :
  ∃ c : ℝ, x - 2 * y + c = 0 ∧ c = 7 :=
by
  sorry

end line_equation_l81_8196


namespace parabola_hyperbola_focus_l81_8165

theorem parabola_hyperbola_focus (p : ℝ) :
  let parabolaFocus := (p / 2, 0)
  let hyperbolaRightFocus := (2, 0)
  (parabolaFocus = hyperbolaRightFocus) → p = 4 := 
by
  intro h
  sorry

end parabola_hyperbola_focus_l81_8165


namespace stream_speed_l81_8189

variables (v_s t_d t_u : ℝ)
variables (D : ℝ) -- Distance is not provided in the problem but assumed for formulation.

theorem stream_speed (h1 : t_u = 2 * t_d) (h2 : v_s = 54 + t_d / t_u) :
  v_s = 18 := 
by
  sorry

end stream_speed_l81_8189


namespace Sharik_cannot_eat_all_meatballs_within_one_million_flies_l81_8178

theorem Sharik_cannot_eat_all_meatballs_within_one_million_flies:
  (∀ n: ℕ, ∃ i: ℕ, i > n ∧ ((∀ j < i, ∀ k: ℕ, ∃ m: ℕ, (m ≠ k) → (∃ f, f < 10^6) )) → f > 10^6 ) :=
sorry

end Sharik_cannot_eat_all_meatballs_within_one_million_flies_l81_8178


namespace ratio_rate_down_to_up_l81_8146

noncomputable def rate_up (r_up t_up: ℕ) : ℕ := r_up * t_up
noncomputable def rate_down (d_down t_down: ℕ) : ℕ := d_down / t_down
noncomputable def ratio (r_down r_up: ℕ) : ℚ := r_down / r_up

theorem ratio_rate_down_to_up :
  let r_up := 6
  let t_up := 2
  let d_down := 18
  let t_down := 2
  rate_up 6 2 = 12 ∧ rate_down 18 2 = 9 ∧ ratio 9 6 = 3 / 2 :=
by
  sorry

end ratio_rate_down_to_up_l81_8146


namespace larger_angle_is_99_l81_8198

theorem larger_angle_is_99 (x : ℝ) (h1 : 2 * x + 18 = 180) : x + 18 = 99 :=
by
  sorry

end larger_angle_is_99_l81_8198


namespace max_volume_prism_l81_8133

theorem max_volume_prism (a b h : ℝ) (V : ℝ) 
  (h1 : a * h + b * h + a * b = 32) : 
  V = a * b * h → V ≤ 128 * Real.sqrt 3 / 3 := 
by
  sorry

end max_volume_prism_l81_8133


namespace geometric_sequence_properties_l81_8176

theorem geometric_sequence_properties (a b c : ℝ) (r : ℝ) (h : r ≠ 0)
  (h1 : a = r * (-1))
  (h2 : b = r * a)
  (h3 : c = r * b)
  (h4 : -9 = r * c) :
  b = -3 ∧ a * c = 9 :=
by sorry

end geometric_sequence_properties_l81_8176


namespace Dad_steps_l81_8132

variable (d m y : ℕ)

-- Conditions
def condition_1 : Prop := d = 3 → m = 5
def condition_2 : Prop := m = 3 → y = 5
def condition_3 : Prop := m + y = 400

-- Question and Answer
theorem Dad_steps : condition_1 d m → condition_2 m y → condition_3 m y → d = 90 :=
by
  intros
  sorry

end Dad_steps_l81_8132


namespace area_difference_of_tablets_l81_8149

theorem area_difference_of_tablets 
  (d1 d2 : ℝ) (s1 s2 : ℝ)
  (h1 : d1 = 6) (h2 : d2 = 5) 
  (hs1 : d1^2 = 2 * s1^2) (hs2 : d2^2 = 2 * s2^2) 
  (A1 : ℝ) (A2 : ℝ) (hA1 : A1 = s1^2) (hA2 : A2 = s2^2)
  : A1 - A2 = 5.5 := 
sorry

end area_difference_of_tablets_l81_8149


namespace solution_x_y_l81_8125

theorem solution_x_y (x y : ℕ) (hx : 0 < x) (hy : 0 < y) : 
    x^4 - 6 * x^2 + 1 = 7 * 2^y ↔ (x = 3 ∧ y = 2) :=
by {
    sorry
}

end solution_x_y_l81_8125


namespace inconsistent_mixture_volume_l81_8164

theorem inconsistent_mixture_volume :
  ∀ (diesel petrol water total_volume : ℚ),
    diesel = 4 →
    petrol = 4 →
    total_volume = 2.666666666666667 →
    diesel + petrol + water = total_volume →
    false :=
by
  intros diesel petrol water total_volume diesel_eq petrol_eq total_volume_eq volume_eq
  rw [diesel_eq, petrol_eq] at volume_eq
  sorry

end inconsistent_mixture_volume_l81_8164


namespace yen_to_usd_conversion_l81_8107

theorem yen_to_usd_conversion
  (cost_of_souvenir : ℕ)
  (service_charge : ℕ)
  (conversion_rate : ℕ)
  (total_cost_in_yen : ℕ)
  (usd_equivalent : ℚ)
  (h1 : cost_of_souvenir = 340)
  (h2 : service_charge = 25)
  (h3 : conversion_rate = 115)
  (h4 : total_cost_in_yen = cost_of_souvenir + service_charge)
  (h5 : usd_equivalent = (total_cost_in_yen : ℚ) / conversion_rate) :
  total_cost_in_yen = 365 ∧ usd_equivalent = 3.17 :=
by
  sorry

end yen_to_usd_conversion_l81_8107


namespace fractional_eq_solve_simplify_and_evaluate_l81_8143

-- Question 1: Solve the fractional equation
theorem fractional_eq_solve (x : ℝ) (h1 : (x / (x + 1) = (2 * x) / (3 * x + 3) + 1)) : 
  x = -1.5 := 
sorry

-- Question 2: Simplify and evaluate the expression for x = -1
theorem simplify_and_evaluate (x : ℝ)
  (h2 : x ≠ 0) (h3 : x ≠ 2) (h4 : x ≠ -2) :
  (x + 2) / (x^2 - 2*x) - (x - 1) / (x^2 - 4*x + 4) / ((x+2) / (x^3 - 4*x)) = 
  (x - 4) / (x - 2) ∧ 
  (x = -1) → ((x - 4) / (x - 2) = (5 / 3)) := 
sorry

end fractional_eq_solve_simplify_and_evaluate_l81_8143


namespace angle_C_max_perimeter_l81_8152

def triangle_ABC (A B C a b c : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 

def circumradius_2 (r : ℝ) : Prop :=
  r = 2

def satisfies_condition (a b c A B C : ℝ) : Prop :=
  (a - c)*(Real.sin A + Real.sin C) = b*(Real.sin A - Real.sin B)

theorem angle_C (A B C a b c : ℝ) (h₁ : triangle_ABC A B C a b c) 
                 (h₂ : satisfies_condition a b c A B C)
                 (h₃ : circumradius_2 (2 : ℝ)) : 
  C = Real.pi / 3 :=
sorry

theorem max_perimeter (A B C a b c r : ℝ) (h₁ : triangle_ABC A B C a b c)
                      (h₂ : satisfies_condition a b c A B C)
                      (h₃ : circumradius_2 r) : 
  4 * Real.sqrt 3 + 2 * Real.sqrt 3 = 6 * Real.sqrt 3 :=
sorry

end angle_C_max_perimeter_l81_8152


namespace find_f_80_l81_8142

noncomputable def f (x : ℝ) : ℝ := sorry

axiom functional_relation (x y : ℝ) (hx : 0 < x) (hy : 0 < y) : 
  f (x * y) = f x / y^2

axiom f_40 : f 40 = 50

-- Proof that f 80 = 12.5
theorem find_f_80 : f 80 = 12.5 := 
by
  sorry

end find_f_80_l81_8142


namespace half_angle_in_first_quadrant_l81_8150

theorem half_angle_in_first_quadrant {α : ℝ} (h : 0 < α ∧ α < π / 2) : 
  0 < α / 2 ∧ α / 2 < π / 4 :=
by
  sorry

end half_angle_in_first_quadrant_l81_8150


namespace shakes_sold_l81_8170

variable (s : ℕ) -- the number of shakes sold

-- conditions
def shakes_ounces := 4 * s
def cone_ounces := 6
def total_ounces := 14

-- the theorem to prove
theorem shakes_sold : shakes_ounces + cone_ounces = total_ounces → s = 2 := by
  intros h
  -- proof can be filled in here
  sorry

end shakes_sold_l81_8170


namespace array_sum_remainder_mod_9_l81_8145

theorem array_sum_remainder_mod_9 :
  let sum_terms := ∑' r : ℕ, ∑' c : ℕ, (1 / (4 ^ r)) * (1 / (9 ^ c))
  ∃ m n : ℕ, Nat.gcd m n = 1 ∧ sum_terms = m / n ∧ (m + n) % 9 = 5 :=
by
  sorry

end array_sum_remainder_mod_9_l81_8145


namespace area_of_triangle_ABF_l81_8144

theorem area_of_triangle_ABF (A B F : ℝ × ℝ) (hF : F = (1, 0)) (hA_parabola : A.2^2 = 4 * A.1) (hB_parabola : B.2^2 = 4 * B.1) (h_midpoint_AB : (A.1 + B.1) / 2 = 2 ∧ (A.2 + B.2) / 2 = 2) : 
  ∃ area : ℝ, area = 2 :=
sorry

end area_of_triangle_ABF_l81_8144


namespace logarithmic_expression_l81_8180

noncomputable def lg (x : ℝ) : ℝ := Real.log x / Real.log 10

theorem logarithmic_expression :
  let log2 := lg 2
  let log5 := lg 5
  log2 + log5 = 1 →
  (log2^3 + 3 * log2 * log5 + log5^3 = 1) :=
by
  intros log2 log5 h
  sorry

end logarithmic_expression_l81_8180


namespace fifth_term_of_geometric_sequence_l81_8131

theorem fifth_term_of_geometric_sequence
  (a r : ℝ)
  (h1 : a * r^2 = 16)
  (h2 : a * r^6 = 2) : a * r^4 = 8 :=
sorry

end fifth_term_of_geometric_sequence_l81_8131


namespace q_minus_r_max_value_l81_8106

theorem q_minus_r_max_value :
  ∃ (q r : ℕ), 1073 = 23 * q + r ∧ q > 0 ∧ r > 0 ∧ q - r = 31 :=
sorry

end q_minus_r_max_value_l81_8106


namespace smallest_value_3a_plus_1_l81_8112

theorem smallest_value_3a_plus_1 
  (a : ℝ)
  (h : 8 * a^2 + 9 * a + 6 = 2) : 
  ∃ (b : ℝ), b = 3 * a + 1 ∧ b = -2 :=
by 
  sorry

end smallest_value_3a_plus_1_l81_8112


namespace fraction_pow_zero_l81_8123

theorem fraction_pow_zero
  (a : ℤ) (b : ℤ)
  (h_a : a = -325123789)
  (h_b : b = 59672384757348)
  (h_nonzero_num : a ≠ 0)
  (h_nonzero_denom : b ≠ 0) :
  (a / b : ℚ) ^ 0 = 1 :=
by {
  sorry
}

end fraction_pow_zero_l81_8123


namespace _l81_8187

noncomputable def t_value_theorem (a b x d t y : ℕ) (h1 : a + b = x) (h2 : x + d = t) (h3 : t + a = y) (h4 : b + d + y = 16) : t = 8 :=
by sorry

end _l81_8187


namespace steak_entree_cost_l81_8153

theorem steak_entree_cost
  (total_guests : ℕ)
  (steak_factor : ℕ)
  (chicken_entree_cost : ℕ)
  (total_budget : ℕ)
  (H1 : total_guests = 80)
  (H2 : steak_factor = 3)
  (H3 : chicken_entree_cost = 18)
  (H4 : total_budget = 1860) :
  ∃ S : ℕ, S = 25 := by
  -- Proof steps omitted
  sorry

end steak_entree_cost_l81_8153


namespace John_distance_proof_l81_8175

def initial_running_time : ℝ := 8
def increase_percentage : ℝ := 0.75
def initial_speed : ℝ := 8
def speed_increase : ℝ := 4

theorem John_distance_proof : 
  (initial_running_time + initial_running_time * increase_percentage) * (initial_speed + speed_increase) = 168 := 
by
  -- Proof can be completed here
  sorry

end John_distance_proof_l81_8175


namespace simplify_expression_l81_8141

variable {x y z : ℝ}

theorem simplify_expression (h : x^2 - y^2 ≠ 0) (hx : x ≠ 0) (hz : z ≠ 0) :
  (x^2 - y^2)⁻¹ * (x⁻¹ - z⁻¹) = (z - x) * x⁻¹ * z⁻¹ * (x^2 - y^2)⁻¹ := by
  sorry

end simplify_expression_l81_8141


namespace find_cost_per_pound_of_mixture_l81_8181

-- Problem Definitions and Conditions
variable (x : ℝ) -- the variable x represents the pounds of Spanish peanuts used
variable (y : ℝ) -- the cost per pound of the mixture we're trying to find
def cost_virginia_pound : ℝ := 3.50
def cost_spanish_pound : ℝ := 3.00
def weight_virginia : ℝ := 10.0

-- Formula for the cost per pound of the mixture
noncomputable def cost_per_pound_of_mixture : ℝ := (weight_virginia * cost_virginia_pound + x * cost_spanish_pound) / (weight_virginia + x)

-- Proof Problem Statement
theorem find_cost_per_pound_of_mixture (h : cost_per_pound_of_mixture x = y) : 
  y = (weight_virginia * cost_virginia_pound + x * cost_spanish_pound) / (weight_virginia + x) := sorry

end find_cost_per_pound_of_mixture_l81_8181


namespace students_taking_both_courses_l81_8103

theorem students_taking_both_courses (total_students students_french students_german students_neither both_courses : ℕ) 
(h1 : total_students = 94) 
(h2 : students_french = 41) 
(h3 : students_german = 22) 
(h4 : students_neither = 40) 
(h5 : total_students = students_french + students_german - both_courses + students_neither) :
both_courses = 9 :=
by
  -- sorry can be replaced with the actual proof if necessary
  sorry

end students_taking_both_courses_l81_8103


namespace value_of_expression_l81_8159

theorem value_of_expression (x y : ℕ) (h₁ : x = 12) (h₂ : y = 7) : (x - y) * (x + y) = 95 := by
  -- Here we assume all necessary conditions as given:
  -- x = 12 and y = 7
  -- and we prove that (x - y)(x + y) = 95
  sorry

end value_of_expression_l81_8159


namespace find_x_values_l81_8160

noncomputable def tan_inv := Real.arctan (Real.sqrt 3 / 2)

theorem find_x_values (x : ℝ) :
  (-Real.pi < x ∧ x ≤ Real.pi) ∧ (2 * Real.tan x - Real.sqrt 3 = 0) ↔
  (x = tan_inv ∨ x = tan_inv - Real.pi) :=
by
  sorry

end find_x_values_l81_8160


namespace area_constant_k_l81_8199

theorem area_constant_k (l w d : ℝ) (h_ratio : l / w = 5 / 2) (h_diagonal : d = Real.sqrt (l^2 + w^2)) :
  ∃ k : ℝ, (k = 10 / 29) ∧ (l * w = k * d^2) :=
by
  sorry

end area_constant_k_l81_8199


namespace garden_yield_l81_8101

theorem garden_yield
  (steps_length : ℕ)
  (steps_width : ℕ)
  (step_to_feet : ℕ → ℝ)
  (yield_per_sqft : ℝ)
  (h1 : steps_length = 18)
  (h2 : steps_width = 25)
  (h3 : ∀ n : ℕ, step_to_feet n = n * 2.5)
  (h4 : yield_per_sqft = 2 / 3)
  : (step_to_feet steps_length * step_to_feet steps_width) * yield_per_sqft = 1875 :=
by
  sorry

end garden_yield_l81_8101


namespace prove_river_improvement_l81_8157

def river_improvement_equation (x : ℝ) : Prop :=
  4800 / x - 4800 / (x + 200) = 4

theorem prove_river_improvement (x : ℝ) (h : x > 0) : river_improvement_equation x := by
  sorry

end prove_river_improvement_l81_8157


namespace volume_in_cubic_yards_l81_8111

-- Define the conditions
def volume_in_cubic_feet : ℕ := 162
def cubic_feet_per_cubic_yard : ℕ := 27

-- Problem statement in Lean 4
theorem volume_in_cubic_yards : volume_in_cubic_feet / cubic_feet_per_cubic_yard = 6 := 
  by
    sorry

end volume_in_cubic_yards_l81_8111


namespace mb_range_l81_8166
-- Define the slope m and y-intercept b
def m : ℚ := 2 / 3
def b : ℚ := -1 / 2

-- Define the product mb
def mb : ℚ := m * b

-- Prove the range of mb
theorem mb_range : -1 < mb ∧ mb < 0 := by
  unfold mb
  sorry

end mb_range_l81_8166


namespace volume_of_pyramid_l81_8129

variables (a b c : ℝ)

def triangle_face1 (a b : ℝ) : Prop := 1/2 * a * b = 1.5
def triangle_face2 (b c : ℝ) : Prop := 1/2 * b * c = 2
def triangle_face3 (c a : ℝ) : Prop := 1/2 * c * a = 6

theorem volume_of_pyramid (h1 : triangle_face1 a b) (h2 : triangle_face2 b c) (h3 : triangle_face3 c a) :
  1/3 * a * b * c = 2 :=
sorry

end volume_of_pyramid_l81_8129


namespace largest_n_satisfying_conditions_l81_8194

theorem largest_n_satisfying_conditions : 
  ∃ n : ℤ, 200 < n ∧ n < 250 ∧ (∃ k : ℤ, 12 * n = k^2) ∧ n = 243 :=
by
  sorry

end largest_n_satisfying_conditions_l81_8194


namespace part1_part2_l81_8124

-- Define the conditions p and q
def p (a x : ℝ) : Prop := (x - a) * (x - 3 * a) < 0
def q (x : ℝ) : Prop := (x - 2) * (x - 4) < 0 ∧ (x - 3) * (x - 5) > 0

-- Problem Part 1: Prove that if a = 1 and p ∧ q is true, then 2 < x < 3
theorem part1 (x : ℝ) : p 1 x ∧ q x → 2 < x ∧ x < 3 :=
by
  intro h
  sorry

-- Problem Part 2: Prove that if p is a necessary but not sufficient condition for q, then 1 ≤ a ≤ 2
theorem part2 (a : ℝ) : (∀ x, q x → p a x) ∧ (∃ x, p a x ∧ ¬q x) → 1 ≤ a ∧ a ≤ 2 :=
by
  intro h
  sorry

end part1_part2_l81_8124


namespace count_complex_numbers_l81_8127

theorem count_complex_numbers (a b : ℕ) (h1 : a > 0) (h2 : b > 0) (h : a + b ≤ 5) : 
  ∃ n, n = 10 := 
by
  sorry

end count_complex_numbers_l81_8127
