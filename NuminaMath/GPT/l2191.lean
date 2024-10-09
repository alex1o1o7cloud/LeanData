import Mathlib

namespace rohan_house_rent_percentage_l2191_219196

variable (salary savings food entertainment conveyance : ℕ)
variable (spend_on_house : ℚ)

-- Given conditions
axiom h1 : salary = 5000
axiom h2 : savings = 1000
axiom h3 : food = 40
axiom h4 : entertainment = 10
axiom h5 : conveyance = 10

-- Define savings percentage
def savings_percentage (salary savings : ℕ) : ℚ := (savings : ℚ) / salary * 100

-- Define percentage equation
def total_percentage (food entertainment conveyance spend_on_house savings_percentage : ℚ) : ℚ :=
  food + spend_on_house + entertainment + conveyance + savings_percentage

-- Prove that house rent percentage is 20%
theorem rohan_house_rent_percentage : 
  food = 40 → entertainment = 10 → conveyance = 10 → salary = 5000 → savings = 1000 → 
  total_percentage 40 10 10 spend_on_house (savings_percentage 5000 1000) = 100 →
  spend_on_house = 20 := by
  intros
  sorry

end rohan_house_rent_percentage_l2191_219196


namespace town_population_original_l2191_219104

noncomputable def original_population (n : ℕ) : Prop :=
  let increased_population := n + 1500
  let decreased_population := (85 / 100 : ℚ) * increased_population
  decreased_population = n + 1455

theorem town_population_original : ∃ n : ℕ, original_population n ∧ n = 1200 :=
by
  sorry

end town_population_original_l2191_219104


namespace a_equals_5_l2191_219148

def f (x : ℝ) (a : ℝ) : ℝ := x^3 + a*x^2 + 3*x - 9
def f' (x : ℝ) (a : ℝ) : ℝ := 3*x^2 + 2*a*x + 3

theorem a_equals_5 (a : ℝ) : 
  (∃ x : ℝ, x = -3 ∧ f' x a = 0) → a = 5 := 
by
  sorry

end a_equals_5_l2191_219148


namespace prove_a_lt_one_l2191_219127

/-- Given the function f defined as -2 * ln x + 1 / 2 * (x^2 + 1) - a * x,
    where a > 0, if f(x) ≥ 0 holds in the interval (1, ∞)
    and f(x) = 0 has a unique solution, then a < 1. -/
theorem prove_a_lt_one (f : ℝ → ℝ) (a : ℝ) 
    (h1 : ∀ x, f x = -2 * Real.log x + 1 / 2 * (x^2 + 1) - a * x)
    (h2 : a > 0)
    (h3 : ∀ x, x > 1 → f x ≥ 0)
    (h4 : ∃! x, f x = 0) : 
    a < 1 :=
by
  sorry

end prove_a_lt_one_l2191_219127


namespace arithmetic_seq_fraction_l2191_219131

theorem arithmetic_seq_fraction (a : ℕ → ℤ) (d : ℤ) 
  (h1 : ∃ d : ℤ, ∀ n : ℕ, a (n + 1) = a n + d) 
  (h2 : a 1 + a 10 = a 9) 
  (d_ne_zero : d ≠ 0) : 
  (a 1 + a 2 + a 3 + a 4 + a 5 + a 6 + a 7 + a 8 + a 9) / a 10 = 27 / 8 := 
sorry

end arithmetic_seq_fraction_l2191_219131


namespace geometric_progression_common_ratio_l2191_219168

theorem geometric_progression_common_ratio :
  ∃ r : ℝ, (r > 0) ∧ (r^3 + r^2 + r - 1 = 0) :=
by
  sorry

end geometric_progression_common_ratio_l2191_219168


namespace hyperbola_asymptote_ratio_l2191_219174

theorem hyperbola_asymptote_ratio
  (a b : ℝ) (h₁ : a ≠ b) (h₂ : (∀ x y : ℝ, (x^2 / a^2) - (y^2 / b^2) = 1))
  (h₃ : ∀ m n: ℝ, m * n = -1 → ∃ θ: ℝ, θ = 90* (π / 180)): 
  a / b = 1 := 
sorry

end hyperbola_asymptote_ratio_l2191_219174


namespace Jacqueline_gave_Jane_l2191_219177

def total_fruits (plums guavas apples : ℕ) : ℕ :=
  plums + guavas + apples

def fruits_given_to_Jane (initial left : ℕ) : ℕ :=
  initial - left

theorem Jacqueline_gave_Jane :
  let plums := 16
  let guavas := 18
  let apples := 21
  let left := 15
  let initial := total_fruits plums guavas apples
  fruits_given_to_Jane initial left = 40 :=
by
  sorry

end Jacqueline_gave_Jane_l2191_219177


namespace largest_sum_is_7_over_12_l2191_219125

-- Define the five sums
def sum1 : ℚ := 1/3 + 1/4
def sum2 : ℚ := 1/3 + 1/5
def sum3 : ℚ := 1/3 + 1/6
def sum4 : ℚ := 1/3 + 1/9
def sum5 : ℚ := 1/3 + 1/8

-- Define the problem statement
theorem largest_sum_is_7_over_12 : 
  max (max (max sum1 sum2) (max sum3 sum4)) sum5 = 7/12 := 
by
  sorry

end largest_sum_is_7_over_12_l2191_219125


namespace range_of_m_l2191_219129

theorem range_of_m (m : ℝ) : (∀ x : ℝ, |x-3| + |x+4| ≥ |2*m-1|) ↔ (-3 ≤ m ∧ m ≤ 4) := by
  sorry

end range_of_m_l2191_219129


namespace first_number_less_than_twice_second_l2191_219164

theorem first_number_less_than_twice_second (x y z : ℕ) : 
  x + y = 50 ∧ y = 19 ∧ x = 2 * y - z → z = 7 :=
by sorry

end first_number_less_than_twice_second_l2191_219164


namespace triangle_area_l2191_219163

theorem triangle_area (a b c : ℝ) (h1 : a = 14) (h2 : b = 48) (h3 : c = 50) (h4 : a^2 + b^2 = c^2) : 
  (1/2 * a * b) = 336 := 
by 
  rw [h1, h2]
  sorry

end triangle_area_l2191_219163


namespace johns_weekly_earnings_increase_l2191_219197

def combined_percentage_increase (initial final : ℕ) : ℕ :=
  ((final - initial) * 100) / initial

theorem johns_weekly_earnings_increase :
  combined_percentage_increase 40 60 = 50 :=
by
  sorry

end johns_weekly_earnings_increase_l2191_219197


namespace total_distance_of_journey_l2191_219121

-- Definitions corresponding to conditions in the problem
def electric_distance : ℝ := 30 -- The first 30 miles were in electric mode
def gasoline_consumption_rate : ℝ := 0.03 -- Gallons per mile for gasoline mode
def average_mileage : ℝ := 50 -- Miles per gallon for the entire trip

-- Final goal: proving the total distance is 90 miles
theorem total_distance_of_journey (d : ℝ) :
  (d / (gasoline_consumption_rate * (d - electric_distance)) = average_mileage) → d = 90 :=
by
  sorry

end total_distance_of_journey_l2191_219121


namespace return_trip_time_l2191_219137

theorem return_trip_time 
  (d p w : ℝ) 
  (h1 : d = 90 * (p - w))
  (h2 : ∀ t, t = d / p → d / (p + w) = t - 15) : 
  d / (p + w) = 64 :=
by
  sorry

end return_trip_time_l2191_219137


namespace range_of_m_min_value_a2_2b2_3c2_l2191_219126

theorem range_of_m (x m : ℝ) (h : ∀ x : ℝ, abs (x + 3) + abs (x + m) ≥ 2 * m) : m ≤ 1 :=
sorry

theorem min_value_a2_2b2_3c2 (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (h4 : a + b + c = 1) :
  ∃ (a b c : ℝ), a = 6/11 ∧ b = 3/11 ∧ c = 2/11 ∧ a^2 + 2 * b^2 + 3 * c^2 = 6/11 :=
sorry

end range_of_m_min_value_a2_2b2_3c2_l2191_219126


namespace find_n_l2191_219186

theorem find_n : ∃ n : ℕ, 0 ≤ n ∧ n ≤ 8 ∧ n % 9 = 4897 % 9 ∧ n = 1 :=
by
  use 1
  sorry

end find_n_l2191_219186


namespace curve_left_of_line_l2191_219157

theorem curve_left_of_line (x y : ℝ) : x^3 + 2*y^2 = 8 → x ≤ 2 := 
sorry

end curve_left_of_line_l2191_219157


namespace algebra_expression_l2191_219159

theorem algebra_expression (a b : ℝ) (h : a = b + 1) : 3 + 2 * a - 2 * b = 5 :=
sorry

end algebra_expression_l2191_219159


namespace greatest_possible_median_l2191_219198

theorem greatest_possible_median {k m r s t : ℕ} 
  (h_mean : (k + m + r + s + t) / 5 = 18) 
  (h_order : k < m ∧ m < r ∧ r < s ∧ s < t) 
  (h_t : t = 40) :
  r = 23 := sorry

end greatest_possible_median_l2191_219198


namespace both_false_of_not_or_l2191_219136

-- Define propositions p and q
variables (p q : Prop)

-- The condition given: ¬(p ∨ q)
theorem both_false_of_not_or (h : ¬(p ∨ q)) : ¬ p ∧ ¬ q :=
by {
  sorry
}

end both_false_of_not_or_l2191_219136


namespace sequence_recurrence_l2191_219109

theorem sequence_recurrence (a : ℕ → ℝ) (h₀ : a 1 = 1) (h : ∀ n : ℕ, n ≥ 1 → a (n + 1) = (n / (n + 1)) * a n) :
  ∀ n : ℕ, n ≥ 1 → a n = 1 / n :=
by
  intro n hn
  exact sorry

end sequence_recurrence_l2191_219109


namespace students_between_hoseok_and_minyoung_l2191_219100

def num_students : Nat := 13
def hoseok_position_from_right : Nat := 9
def minyoung_position_from_left : Nat := 8

theorem students_between_hoseok_and_minyoung
    (n : Nat)
    (h : n = num_students)
    (p_h : n - hoseok_position_from_right + 1 = 5)
    (p_m : minyoung_position_from_left = 8):
    ∃ k : Nat, k = 2 :=
by
  sorry

end students_between_hoseok_and_minyoung_l2191_219100


namespace fraction_equivalence_l2191_219180

theorem fraction_equivalence (a b : ℝ) (h : ((1 / a) + (1 / b)) / ((1 / a) - (1 / b)) = 2020) : (a + b) / (a - b) = 2020 :=
sorry

end fraction_equivalence_l2191_219180


namespace like_terms_ratio_l2191_219115

theorem like_terms_ratio (m n : ℕ) (h₁ : m - 2 = 2) (h₂ : 3 = 2 * n - 1) : m / n = 2 := 
by
  sorry

end like_terms_ratio_l2191_219115


namespace number_of_ways_to_divide_friends_l2191_219134

theorem number_of_ways_to_divide_friends :
  let friends := 8
  let teams := 4
  (teams ^ friends) = 65536 := by
  sorry

end number_of_ways_to_divide_friends_l2191_219134


namespace tan_x_value_l2191_219135

noncomputable def f (x : ℝ) : ℝ := Real.sin x - Real.cos x

theorem tan_x_value:
  (∀ x : ℝ, deriv f x = 2 * f x) → (∀ x : ℝ, f x = Real.sin x - Real.cos x) → (∀ x : ℝ, Real.tan x = 3) := 
by
  intros h_deriv h_f
  sorry

end tan_x_value_l2191_219135


namespace difference_in_surface_area_l2191_219114

-- Defining the initial conditions
def original_length : ℝ := 6
def original_width : ℝ := 5
def original_height : ℝ := 4
def cube_side : ℝ := 2

-- Define the surface area calculation for a rectangular solid
def surface_area_rectangular_prism (l w h : ℝ) : ℝ :=
  2 * (l * w + l * h + w * h)

-- Define the surface area of the cube
def surface_area_cube (a : ℝ) : ℝ :=
  6 * a * a

-- Define the removed face areas when cube is extracted
def exposed_faces_area (a : ℝ) : ℝ :=
  2 * (a * a)

-- Define the problem statement in Lean
theorem difference_in_surface_area :
  surface_area_rectangular_prism original_length original_width original_height
  - (surface_area_rectangular_prism original_length original_width original_height - surface_area_cube cube_side + exposed_faces_area cube_side) = 12 :=
by
  sorry

end difference_in_surface_area_l2191_219114


namespace percentage_decrease_hours_worked_l2191_219107

theorem percentage_decrease_hours_worked (B H : ℝ) (h₁ : H > 0) (h₂ : B > 0)
  (h_assistant1 : (1.8 * B) = B * 1.8) (h_assistant2 : (2 * (B / H)) = (1.8 * B) / (0.9 * H)) : 
  ((H - (0.9 * H)) / H) * 100 = 10 := 
by
  sorry

end percentage_decrease_hours_worked_l2191_219107


namespace fg_2_eq_9_l2191_219155

def f (x: ℝ) := x^2
def g (x: ℝ) := -4 * x + 5

theorem fg_2_eq_9 : f (g 2) = 9 :=
by
  sorry

end fg_2_eq_9_l2191_219155


namespace size_of_angle_C_l2191_219162

theorem size_of_angle_C 
  (a b c : ℝ) 
  (A B C : ℝ) 
  (h1 : a = 5) 
  (h2 : b + c = 2 * a) 
  (h3 : 3 * Real.sin A = 5 * Real.sin B) : 
  C = 2 * Real.pi / 3 := 
sorry

end size_of_angle_C_l2191_219162


namespace smallest_sum_of_squares_l2191_219122

theorem smallest_sum_of_squares (x y : ℤ) (h : x^2 - y^2 = 231) :
  x^2 + y^2 ≥ 281 :=
sorry

end smallest_sum_of_squares_l2191_219122


namespace no_unhappy_days_l2191_219193

theorem no_unhappy_days (D R : ℕ) : 
  (D^2 + 4) * (R^2 + 4) - 2 * D * (R^2 + 4) - 2 * R * (D^2 + 4) ≥ 0 := by
  sorry

end no_unhappy_days_l2191_219193


namespace solve_n_minus_m_l2191_219112

theorem solve_n_minus_m :
  ∃ m n, 
    (m ≡ 4 [MOD 7]) ∧ 100 ≤ m ∧ m < 1000 ∧ 
    (n ≡ 4 [MOD 7]) ∧ 1000 ≤ n ∧ n < 10000 ∧ 
    n - m = 903 :=
by
  sorry

end solve_n_minus_m_l2191_219112


namespace num_houses_with_digit_7_in_range_l2191_219123

-- Define the condition for a number to contain a digit 7
def contains_digit_7 (n : Nat) : Prop :=
  (n / 10 = 7) || (n % 10 = 7)

-- The main theorem
theorem num_houses_with_digit_7_in_range (h : Nat) (H1 : 1 ≤ h ∧ h ≤ 70) : 
  ∃! n, 1 ≤ n ∧ n ≤ 70 ∧ contains_digit_7 n :=
sorry

end num_houses_with_digit_7_in_range_l2191_219123


namespace only_exprC_cannot_be_calculated_with_square_of_binomial_l2191_219147

-- Definitions of our expressions using their variables
def exprA (a b : ℝ) := (a + b) * (a - b)
def exprB (x : ℝ) := (-x + 1) * (-x - 1)
def exprC (y : ℝ) := (y + 1) * (-y - 1)
def exprD (m : ℝ) := (m - 1) * (-1 - m)

-- Statement that only exprC cannot be calculated using the square of a binomial formula
theorem only_exprC_cannot_be_calculated_with_square_of_binomial :
  (∀ a b : ℝ, ∃ (u v : ℝ), exprA a b = u^2 - v^2) ∧
  (∀ x : ℝ, ∃ (u v : ℝ), exprB x = u^2 - v^2) ∧
  (forall m : ℝ, ∃ (u v : ℝ), exprD m = u^2 - v^2) 
  ∧ (∀ v : ℝ, ¬ ∃ (u : ℝ), exprC v = u^2 ∨ (exprC v = - (u^2))) := sorry

end only_exprC_cannot_be_calculated_with_square_of_binomial_l2191_219147


namespace ellipse_foci_distance_l2191_219143

-- Definitions based on the problem conditions
def ellipse_eq (x y : ℝ) :=
  Real.sqrt (((x - 4)^2) + ((y - 5)^2)) + Real.sqrt (((x + 6)^2) + ((y + 9)^2)) = 22

def focus1 : (ℝ × ℝ) := (4, -5)
def focus2 : (ℝ × ℝ) := (-6, 9)

-- Statement of the problem
noncomputable def distance_between_foci : ℝ :=
  Real.sqrt (((focus1.1 + 6)^2) + ((focus1.2 - 9)^2))

-- Proof statement
theorem ellipse_foci_distance : distance_between_foci = 2 * Real.sqrt 74 := by
  sorry

end ellipse_foci_distance_l2191_219143


namespace max_value_l2191_219165

def is_odd (f : ℝ → ℝ) := ∀ x, f (-x) = -f x
def is_increasing (f : ℝ → ℝ) := ∀ {a b}, a < b → f a < f b

theorem max_value (f : ℝ → ℝ) (x y : ℝ)
  (h_odd : is_odd f)
  (h_increasing : is_increasing f)
  (h_eq : f (x^2 - 2 * x) + f y = 0) :
  2 * x + y ≤ 4 :=
sorry

end max_value_l2191_219165


namespace find_A_l2191_219101

def spadesuit (A B : ℝ) : ℝ := 4 * A + 3 * B - 2

theorem find_A (A : ℝ) : spadesuit A 7 = 40 ↔ A = 21 / 4 :=
by
  sorry

end find_A_l2191_219101


namespace can_construct_length_one_l2191_219142

noncomputable def possible_to_construct_length_one_by_folding (n : ℕ) : Prop :=
  ∃ k ≤ 10, ∃ (segment_constructed : ℝ), segment_constructed = 1

theorem can_construct_length_one : possible_to_construct_length_one_by_folding 2016 :=
by sorry

end can_construct_length_one_l2191_219142


namespace parts_of_cut_square_l2191_219103

theorem parts_of_cut_square (folds_to_one_by_one : ℕ) : folds_to_one_by_one = 9 :=
  sorry

end parts_of_cut_square_l2191_219103


namespace sum_of_first_70_odd_integers_l2191_219108

theorem sum_of_first_70_odd_integers : 
  let sum_even := 70 * (70 + 1)
  let sum_odd := 70 ^ 2
  let diff := sum_even - sum_odd
  diff = 70 → sum_odd = 4900 :=
by
  intros
  sorry

end sum_of_first_70_odd_integers_l2191_219108


namespace area_of_fig_eq_2_l2191_219171

noncomputable def area_of_fig : ℝ :=
  - ∫ x in (2 * Real.pi / 3)..Real.pi, (Real.sin x - Real.sqrt 3 * Real.cos x)

theorem area_of_fig_eq_2 : area_of_fig = 2 :=
by
  sorry

end area_of_fig_eq_2_l2191_219171


namespace find_a_l2191_219105

noncomputable def base25_num : ℕ := 3 * 25^7 + 1 * 25^6 + 4 * 25^5 + 2 * 25^4 + 6 * 25^3 + 5 * 25^2 + 2 * 25^1 + 3 * 25^0

theorem find_a (a : ℤ) (h0 : 0 ≤ a) (h1 : a ≤ 14) : ((base25_num - a) % 12 = 0) → a = 2 := 
sorry

end find_a_l2191_219105


namespace or_false_iff_not_p_l2191_219192

theorem or_false_iff_not_p (p q : Prop) : (p ∨ q → false) ↔ ¬p :=
by sorry

end or_false_iff_not_p_l2191_219192


namespace decrease_in_silver_coins_l2191_219139

theorem decrease_in_silver_coins
  (a : ℕ) (h₁ : 2 * a = 3 * (50 - a))
  (h₂ : a + (50 - a) = 50) :
  (5 * (50 - a) - 3 * a = 10) :=
by
sorry

end decrease_in_silver_coins_l2191_219139


namespace inequality_proof_l2191_219106

theorem inequality_proof
  (a b c : ℝ)
  (h1 : a > 0)
  (h2 : b > 0)
  (h3 : c > 0)
  (h4 : a + b + c = 1) :
  a * (1 + b - c) ^ (1 / 3) + b * (1 + c - a) ^ (1 / 3) + c * (1 + a - b) ^ (1 / 3) ≤ 1 := 
by
  sorry

end inequality_proof_l2191_219106


namespace distance_from_origin_l2191_219187

noncomputable def point_distance (x y : ℝ) := Real.sqrt (x^2 + y^2)

theorem distance_from_origin (x y : ℝ) (h₁ : abs y = 15) (h₂ : Real.sqrt ((x - 2)^2 + (y - 7)^2) = 13) (h₃ : x > 2) :
  point_distance x y = Real.sqrt (334 + 4 * Real.sqrt 105) :=
by
  sorry

end distance_from_origin_l2191_219187


namespace wine_age_proof_l2191_219116

-- Definitions based on conditions
def Age_Carlo_Rosi : ℕ := 40
def Age_Twin_Valley : ℕ := Age_Carlo_Rosi / 4
def Age_Franzia : ℕ := 3 * Age_Carlo_Rosi

-- We'll use a definition to represent the total age of the three brands of wine.
def Total_Age : ℕ := Age_Franzia + Age_Carlo_Rosi + Age_Twin_Valley

-- Statement to be proven
theorem wine_age_proof : Total_Age = 170 :=
by {
  sorry -- Proof goes here
}

end wine_age_proof_l2191_219116


namespace katie_miles_l2191_219173

theorem katie_miles (x : ℕ) (h1 : ∀ y, y = 3 * x → y ≤ 240) (h2 : x + 3 * x = 240) : x = 60 :=
sorry

end katie_miles_l2191_219173


namespace solve_fraction_eq_l2191_219132

theorem solve_fraction_eq (x : ℝ) :
  (1 / ((x - 1) * (x - 2)) + 1 / ((x - 2) * (x - 3)) + 1 / ((x - 3) * (x - 4)) = 1 / 6) ↔ 
  (x = 7 ∨ x = -2) := 
by
  sorry

end solve_fraction_eq_l2191_219132


namespace round_robin_highest_score_l2191_219146

theorem round_robin_highest_score
  (n : ℕ) (hn : n = 16)
  (teams : Fin n → ℕ)
  (games_played : Fin n → Fin n → ℕ)
  (draws : Fin n → Fin n → ℕ)
  (win_points : ℕ := 2)
  (draw_points : ℕ := 1)
  (total_games : ℕ := (n * (n - 1)) / 2) :
  ¬ (∃ max_score : ℕ, ∀ i : Fin n, teams i ≤ max_score ∧ max_score < 16) :=
by sorry

end round_robin_highest_score_l2191_219146


namespace exist_odd_distinct_integers_l2191_219195

theorem exist_odd_distinct_integers (n : ℕ) (h1 : n % 2 = 1) (h2 : n > 3) (h3 : n % 3 ≠ 0) : 
  ∃ a b c : ℕ, a % 2 = 1 ∧ b % 2 = 1 ∧ c % 2 = 1 ∧ a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ 
  3 / (n : ℚ) = 1 / (a : ℚ) + 1 / (b : ℚ) + 1 / (c : ℚ) :=
sorry

end exist_odd_distinct_integers_l2191_219195


namespace find_a_plus_b_l2191_219141

theorem find_a_plus_b 
  (a b : ℝ)
  (f : ℝ → ℝ) 
  (f_def : ∀ x, f x = x^3 + 3 * x^2 + 6 * x + 14)
  (cond_a : f a = 1) 
  (cond_b : f b = 19) :
  a + b = -2 :=
sorry

end find_a_plus_b_l2191_219141


namespace quadratic_inequality_solution_l2191_219184

theorem quadratic_inequality_solution :
  (∀ x : ℝ, x ∈ Set.Ioo ((1 - Real.sqrt 2) / 3) ((1 + Real.sqrt 2) / 3) → -9 * x^2 + 6 * x + 1 < 0) ∧
  (∀ x : ℝ, -9 * x^2 + 6 * x + 1 < 0 → x ∈ Set.Ioo ((1 - Real.sqrt 2) / 3) ((1 + Real.sqrt 2) / 3)) :=
by
  sorry

end quadratic_inequality_solution_l2191_219184


namespace min_value_l2191_219152

variable {α : Type*} [LinearOrderedField α]

-- Define a geometric sequence with strictly positive terms
def is_geometric_sequence (a : ℕ → α) : Prop :=
  ∃ (q : α), q > 0 ∧ ∀ n, a (n + 1) = a n * q

-- Given conditions
variables (a : ℕ → α) (S : ℕ → α)
variables (h_geom : is_geometric_sequence a)
variables (h_pos : ∀ n, a n > 0)
variables (h_a23 : a 2 * a 6 = 4) (h_a3 : a 3 = 1)

-- Sum of the first n terms of a geometric sequence
def sum_first_n (a : ℕ → α) (n : ℕ) : α :=
  if n = 0 then 0
  else a 0 * ((1 - (a 1 / a 0) ^ n) / (1 - (a 1 / a 0)))

-- Statement of the theorem
theorem min_value (a : ℕ → α) (S : ℕ → α) 
  (h_geom : is_geometric_sequence a)
  (h_pos : ∀ n, a n > 0)
  (h_a23 : a 2 * a 6 = 4)
  (h_a3 : a 3 = 1)
  (h_Sn : ∀ n, S n = sum_first_n a n) :
  ∃ n, n = 3 ∧ (S n + 9 / 4) ^ 2 / (2 * a n) = 8 :=
sorry

end min_value_l2191_219152


namespace range_of_a_l2191_219128

noncomputable def f (x : ℝ) : ℝ := 
  if h : x ≤ 1 then x^2 - x + 3 else 0

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, f x ≥ |x / 2 + a|) ↔ -47 / 16 ≤ a ∧ a ≤ 2 := sorry

end range_of_a_l2191_219128


namespace intersection_A_B_intersection_CA_B_intersection_CA_CB_l2191_219194

-- Set definitions
def A := {x : ℝ | -5 ≤ x ∧ x ≤ 3}
def B := {x : ℝ | x < -2 ∨ x > 4}
def C_A := {x : ℝ | x < -5 ∨ x > 3}  -- Complement of A
def C_B := {x : ℝ | -2 ≤ x ∧ x ≤ 4}  -- Complement of B

-- Lean statements proving the intersections
theorem intersection_A_B : {x : ℝ | -5 ≤ x ∧ x ≤ 3} ∩ {x : ℝ | x < -2 ∨ x > 4} = {x : ℝ | -5 ≤ x ∧ x < -2} :=
by sorry

theorem intersection_CA_B : {x : ℝ | x < -5 ∨ x > 3} ∩ {x : ℝ | x < -2 ∨ x > 4} = {x : ℝ | x < -5 ∨ x > 4} :=
by sorry

theorem intersection_CA_CB : {x : ℝ | x < -5 ∨ x > 3} ∩ {x : ℝ | -2 ≤ x ∧ x ≤ 4} = {x : ℝ | 3 < x ∧ x ≤ 4} :=
by sorry

end intersection_A_B_intersection_CA_B_intersection_CA_CB_l2191_219194


namespace optimal_position_station_l2191_219176

-- Definitions for the conditions
def num_buildings := 5
def building_workers (k : ℕ) : ℕ := if k ≤ 5 then k else 0
def distance_between_buildings := 50

-- Function to calculate the total walking distance
noncomputable def total_distance (x : ℝ) : ℝ :=
  |x| + 2 * |x - 50| + 3 * |x - 100| + 4 * |x - 150| + 5 * |x - 200|

-- Theorem statement
theorem optimal_position_station :
  ∃ x : ℝ, (∀ y : ℝ, total_distance x ≤ total_distance y) ∧ x = 150 :=
by
  sorry

end optimal_position_station_l2191_219176


namespace smallest_possible_area_of_2020th_square_l2191_219161

theorem smallest_possible_area_of_2020th_square :
  ∃ A : ℕ, (∃ n : ℕ, n * n = 2019 + A) ∧ A ≠ 1 ∧
  ∀ A' : ℕ, A' > 0 ∧ (∃ n : ℕ, n * n = 2019 + A') ∧ A' ≠ 1 → A ≤ A' :=
by
  sorry

end smallest_possible_area_of_2020th_square_l2191_219161


namespace value_of_x_plus_4_l2191_219102

theorem value_of_x_plus_4 (x : ℝ) (h : 2 * x + 6 = 16) : x + 4 = 9 :=
by
  sorry

end value_of_x_plus_4_l2191_219102


namespace frog_climb_time_l2191_219190

-- Definitions related to the problem
def well_depth : ℕ := 12
def climb_per_cycle : ℕ := 3
def slip_per_cycle : ℕ := 1
def effective_climb_per_cycle : ℕ := climb_per_cycle - slip_per_cycle

-- Time taken for each activity
def time_to_climb : ℕ := 10 -- given as t
def time_to_slip : ℕ := time_to_climb / 3
def total_time_per_cycle : ℕ := time_to_climb + time_to_slip

-- Condition specifying the observed frog position at a certain time
def observed_time : ℕ := 17 -- minutes since 8:00
def observed_position : ℕ := 9 -- meters climbed since it's 3 meters from the top of the well (well_depth - 3)

-- The main theorem stating the total time taken to climb to the top of the well
theorem frog_climb_time : 
  ∃ (k : ℕ), k * effective_climb_per_cycle + climb_per_cycle = well_depth ∧ k * total_time_per_cycle + time_to_climb = 22 := 
sorry

end frog_climb_time_l2191_219190


namespace original_ratio_l2191_219113

theorem original_ratio (F J : ℚ) (hJ : J = 180) (h_ratio : (F + 45) / J = 3 / 2) : F / J = 5 / 4 :=
by
  sorry

end original_ratio_l2191_219113


namespace tricycles_count_l2191_219149

theorem tricycles_count (B T : ℕ) (hB : B = 50) (hW : 2 * B + 3 * T = 160) : T = 20 :=
by
  sorry

end tricycles_count_l2191_219149


namespace new_tv_cost_l2191_219140

/-
Mark bought his first TV which was 24 inches wide and 16 inches tall. It cost $672.
His new TV is 48 inches wide and 32 inches tall.
The first TV was $1 more expensive per square inch compared to his newest TV.
Prove that the cost of his new TV is $1152.
-/

theorem new_tv_cost :
  let width_first_tv := 24
  let height_first_tv := 16
  let cost_first_tv := 672
  let width_new_tv := 48
  let height_new_tv := 32
  let discount_per_square_inch := 1
  let area_first_tv := width_first_tv * height_first_tv
  let cost_per_square_inch_first_tv := cost_first_tv / area_first_tv
  let cost_per_square_inch_new_tv := cost_per_square_inch_first_tv - discount_per_square_inch
  let area_new_tv := width_new_tv * height_new_tv
  let cost_new_tv := cost_per_square_inch_new_tv * area_new_tv
  cost_new_tv = 1152 := by
  sorry

end new_tv_cost_l2191_219140


namespace num_males_in_group_l2191_219172

-- Definitions based on the given conditions
def num_females (f : ℕ) : Prop := f = 16
def num_males_choose_malt (m_malt : ℕ) : Prop := m_malt = 6
def num_females_choose_malt (f_malt : ℕ) : Prop := f_malt = 8
def num_choose_malt (m_malt f_malt n_malt : ℕ) : Prop := n_malt = m_malt + f_malt
def num_choose_coke (c : ℕ) (n_malt : ℕ) : Prop := n_malt = 2 * c
def total_cheerleaders (t : ℕ) (n_malt c : ℕ) : Prop := t = n_malt + c
def num_males (m f t : ℕ) : Prop := m = t - f

theorem num_males_in_group
  (f m_malt f_malt n_malt c t m : ℕ)
  (hf : num_females f)
  (hmm : num_males_choose_malt m_malt)
  (hfm : num_females_choose_malt f_malt)
  (hmalt : num_choose_malt m_malt f_malt n_malt)
  (hc : num_choose_coke c n_malt)
  (ht : total_cheerleaders t n_malt c)
  (hm : num_males m f t) :
  m = 5 := 
sorry

end num_males_in_group_l2191_219172


namespace john_new_earnings_after_raise_l2191_219138

-- Definition of original earnings and raise percentage
def original_earnings : ℝ := 50
def raise_percentage : ℝ := 0.50

-- Calculate raise amount and new earnings after raise
def raise_amount : ℝ := raise_percentage * original_earnings
def new_earnings : ℝ := original_earnings + raise_amount

-- Math proof problem: Prove new earnings after raise equals $75
theorem john_new_earnings_after_raise : new_earnings = 75 := by
  sorry

end john_new_earnings_after_raise_l2191_219138


namespace line_perp_to_plane_contains_line_implies_perp_l2191_219170

variables {Point Line Plane : Type}
variables (m n : Line) (α : Plane)
variables (contains : Plane → Line → Prop) (perp : Line → Line → Prop) (perp_plane : Line → Plane → Prop)

-- Given: 
-- m and n are two different lines
-- α is a plane
-- m ⊥ α (m is perpendicular to the plane α)
-- n ⊂ α (n is contained in the plane α)
-- Prove: m ⊥ n
theorem line_perp_to_plane_contains_line_implies_perp (hm : perp_plane m α) (hn : contains α n) : perp m n :=
sorry

end line_perp_to_plane_contains_line_implies_perp_l2191_219170


namespace bug_total_distance_l2191_219153

/-- 
A bug starts at position 3 on a number line. It crawls to -4, then to 7, and finally to 1.
The total distance the bug crawls is 24 units.
-/
theorem bug_total_distance : 
  let start := 3
  let first_stop := -4
  let second_stop := 7
  let final_position := 1
  let distance := abs (first_stop - start) + abs (second_stop - first_stop) + abs (final_position - second_stop)
  distance = 24 := 
by
  sorry

end bug_total_distance_l2191_219153


namespace joe_avg_speed_l2191_219166

noncomputable def total_distance : ℝ :=
  420 + 250 + 120 + 65

noncomputable def total_time : ℝ :=
  (420 / 60) + (250 / 50) + (120 / 40) + (65 / 70)

noncomputable def avg_speed : ℝ :=
  total_distance / total_time

theorem joe_avg_speed : avg_speed = 53.67 := by
  sorry

end joe_avg_speed_l2191_219166


namespace points_on_curve_is_parabola_l2191_219188

theorem points_on_curve_is_parabola (X Y : ℝ) (h : Real.sqrt X + Real.sqrt Y = 1) :
  ∃ a b c : ℝ, Y = a * X^2 + b * X + c :=
sorry

end points_on_curve_is_parabola_l2191_219188


namespace vec_subtraction_l2191_219183

-- Definitions
def a : ℝ × ℝ := (1, -2)
def b (m : ℝ) : ℝ × ℝ := (m, 4)

-- Condition: a is parallel to b
def are_parallel (a b : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, b = (k * a.1, k * a.2)

-- Main theorem
theorem vec_subtraction (m : ℝ) (h : are_parallel a (b m)) :
  2 • a - b m = (4, -8) :=
sorry

end vec_subtraction_l2191_219183


namespace sequence_sum_a5_a6_l2191_219144

-- Given sequence partial sum definition
def partial_sum (n : ℕ) : ℕ := n^3

-- Definition of sequence term a_n
def a (n : ℕ) : ℕ := partial_sum n - partial_sum (n - 1)

-- Main theorem to prove a_5 + a_6 = 152
theorem sequence_sum_a5_a6 : a 5 + a 6 = 152 :=
by
  sorry

end sequence_sum_a5_a6_l2191_219144


namespace evaluate_expression_l2191_219110

theorem evaluate_expression (a : ℝ) : (a^7 + a^7 + a^7 - a^7) = a^8 :=
by
  sorry

end evaluate_expression_l2191_219110


namespace tom_age_ratio_l2191_219120

theorem tom_age_ratio (T : ℕ) (h1 : T = 3 * (3 : ℕ)) (h2 : T - 5 = 3 * ((T / 3) - 10)) : T / 5 = 9 := 
by
  sorry

end tom_age_ratio_l2191_219120


namespace periodic_minus_decimal_is_correct_l2191_219124

-- Definitions based on conditions

def periodic_63_as_fraction : ℚ := 63 / 99
def decimal_63_as_fraction : ℚ := 63 / 100
def difference : ℚ := periodic_63_as_fraction - decimal_63_as_fraction

-- Lean 4 statement to prove the mathematically equivalent proof problem
theorem periodic_minus_decimal_is_correct :
  difference = 7 / 1100 :=
by
  sorry

end periodic_minus_decimal_is_correct_l2191_219124


namespace degree_le_of_lt_eventually_l2191_219145

open Polynomial

theorem degree_le_of_lt_eventually {P Q : Polynomial ℝ} (h_exists : ∃ N : ℝ, ∀ x : ℝ, x > N → P.eval x < Q.eval x) :
  P.degree ≤ Q.degree :=
sorry

end degree_le_of_lt_eventually_l2191_219145


namespace tangent_line_b_value_l2191_219117

theorem tangent_line_b_value (a k b : ℝ) 
  (h_curve : ∀ x, x^3 + a * x + 1 = 3 ↔ x = 2)
  (h_derivative : k = 3 * 2^2 - 3)
  (h_tangent : 3 = k * 2 + b) : b = -15 :=
sorry

end tangent_line_b_value_l2191_219117


namespace number_of_red_notes_each_row_l2191_219181

-- Definitions for the conditions
variable (R : ℕ) -- Number of red notes in each row
variable (total_notes : ℕ := 100) -- Total number of notes

-- Derived quantities
def total_red_notes := 5 * R
def total_blue_notes := 2 * total_red_notes + 10

-- Statement of the theorem
theorem number_of_red_notes_each_row 
  (h : total_red_notes + total_blue_notes = total_notes) : 
  R = 6 :=
by
  sorry

end number_of_red_notes_each_row_l2191_219181


namespace triangle_inequality_l2191_219133

theorem triangle_inequality (a b c : ℝ) (α : ℝ) 
  (h_triangle_sides : a + b > c ∧ b + c > a ∧ c + a > b) 
  (h_cosine_rule : a^2 = b^2 + c^2 - 2 * b * c * Real.cos α) :
  (2 * b * c * Real.cos α) / (b + c) < (b + c - a) ∧ (b + c - a) < (2 * b * c) / a := 
sorry

end triangle_inequality_l2191_219133


namespace lcm_18_20_l2191_219118

theorem lcm_18_20 : Nat.lcm 18 20 = 180 := by
  sorry

end lcm_18_20_l2191_219118


namespace total_donation_correct_l2191_219160

-- Define the donations to each orphanage
def first_orphanage_donation : ℝ := 175.00
def second_orphanage_donation : ℝ := 225.00
def third_orphanage_donation : ℝ := 250.00

-- State the total donation
def total_donation : ℝ := 650.00

-- The theorem statement to be proved
theorem total_donation_correct :
  first_orphanage_donation + second_orphanage_donation + third_orphanage_donation = total_donation :=
by
  sorry

end total_donation_correct_l2191_219160


namespace derivative_at_zero_l2191_219158

def f (x : ℝ) : ℝ := x^3

theorem derivative_at_zero : deriv f 0 = 0 :=
by
  sorry

end derivative_at_zero_l2191_219158


namespace inradius_inequality_l2191_219154

theorem inradius_inequality
  (r r_A r_B r_C : ℝ) 
  (h_inscribed_circle: r > 0) 
  (h_tangent_circles_A: r_A > 0) 
  (h_tangent_circles_B: r_B > 0) 
  (h_tangent_circles_C: r_C > 0)
  : r ≤ r_A + r_B + r_C :=
  sorry

end inradius_inequality_l2191_219154


namespace intersection_A_B_l2191_219156

open Set

noncomputable def A : Set ℤ := {-1, 0, 1, 2, 3, 4, 5}

noncomputable def B : Set ℤ := {b | ∃ n : ℤ, b = n^2 - 1}

theorem intersection_A_B :
  A ∩ B = {-1, 0, 3} :=
by {
  sorry
}

end intersection_A_B_l2191_219156


namespace part1_subsets_m_0_part2_range_m_l2191_219167

namespace MathProof

variables {α : Type*} {m : ℝ}

def A := {x : ℝ | x^2 + 5 * x - 6 = 0}
def B (m : ℝ) := {x : ℝ | x^2 + 2 * (m + 1) * x + m^2 - 3 = 0}
def subsets (A : Set ℝ) := {s : Set ℝ | s ⊆ A}

theorem part1_subsets_m_0 :
  subsets (A ∪ B 0) = {∅, {-6}, {1}, {-3}, {-6,1}, {-6,-3}, {1,-3}, {-6,1,-3}} :=
sorry

theorem part2_range_m (h : ∀ x, x ∈ B m → x ∈ A) : m ≤ -2 :=
sorry

end MathProof

end part1_subsets_m_0_part2_range_m_l2191_219167


namespace domain_f_l2191_219130

noncomputable def f (x : ℝ) : ℝ := (Real.sqrt x) / ((x^2) - 4)

theorem domain_f : {x : ℝ | 0 ≤ x ∧ x ≠ 2} = {x | 0 ≤ x ∧ x < 2} ∪ {x | x > 2} :=
by sorry

end domain_f_l2191_219130


namespace total_chairs_l2191_219119

-- Define the conditions as constants
def living_room_chairs : ℕ := 3
def kitchen_chairs : ℕ := 6
def dining_room_chairs : ℕ := 8
def outdoor_patio_chairs : ℕ := 12

-- State the goal to prove
theorem total_chairs : 
  living_room_chairs + kitchen_chairs + dining_room_chairs + outdoor_patio_chairs = 29 := 
by
  -- The proof is not required as per instructions
  sorry

end total_chairs_l2191_219119


namespace gcd_45_81_63_l2191_219182

theorem gcd_45_81_63 : Nat.gcd 45 (Nat.gcd 81 63) = 9 := 
sorry

end gcd_45_81_63_l2191_219182


namespace cos_neg_79_pi_over_6_l2191_219150

theorem cos_neg_79_pi_over_6 : 
  Real.cos (-79 * Real.pi / 6) = -Real.sqrt 3 / 2 :=
by
  sorry

end cos_neg_79_pi_over_6_l2191_219150


namespace product_sum_l2191_219179

theorem product_sum (y x z: ℕ) 
  (h1: 2014 + y = 2015 + x) 
  (h2: 2015 + x = 2016 + z) 
  (h3: y * x * z = 504): 
  y * x + x * z = 128 := 
by 
  sorry

end product_sum_l2191_219179


namespace find_g1_l2191_219111

variables {f g : ℝ → ℝ}

def odd_function (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x
def even_function (g : ℝ → ℝ) : Prop := ∀ x, g (-x) = g x

theorem find_g1 (hf : odd_function f)
                (hg : even_function g)
                (h1 : f (-1) + g 1 = 2)
                (h2 : f 1 + g (-1) = 4) :
                g 1 = 3 :=
sorry

end find_g1_l2191_219111


namespace arithmetic_sequence_problem_l2191_219178

theorem arithmetic_sequence_problem 
  (a : ℕ → ℕ)
  (h1 : a 2 + a 3 + a 4 = 15)
  (h2 : (a 1 + 2) * (a 6 + 16) = (a 3 + 4) ^ 2)
  (h_positive : ∀ n, 0 < a n) :
  a 10 = 19 :=
sorry

end arithmetic_sequence_problem_l2191_219178


namespace drivers_distance_difference_l2191_219189

noncomputable def total_distance_driven (initial_distance : ℕ) (speed_A : ℕ) (speed_B : ℕ) (start_delay : ℕ) : ℕ := sorry

theorem drivers_distance_difference
  (initial_distance : ℕ)
  (speed_A : ℕ)
  (speed_B : ℕ)
  (start_delay : ℕ)
  (correct_difference : ℕ)
  (h_initial : initial_distance = 1025)
  (h_speed_A : speed_A = 90)
  (h_speed_B : speed_B = 80)
  (h_start_delay : start_delay = 1)
  (h_correct_difference : correct_difference = 145) :
  total_distance_driven initial_distance speed_A speed_B start_delay = correct_difference :=
sorry

end drivers_distance_difference_l2191_219189


namespace carrots_weight_l2191_219169

-- Let the weight of the carrots be denoted by C (in kg).
variables (C : ℕ)

-- Conditions:
-- The merchant installed 13 kg of zucchini and 8 kg of broccoli.
-- He sold only half of the total, which amounted to 18 kg, so the total weight was 36 kg.
def conditions := (C + 13 + 8 = 36)

-- Prove that the weight of the carrots installed is 15 kg.
theorem carrots_weight (H : C + 13 + 8 = 36) : C = 15 :=
by {
  sorry -- proof to be filled in
}

end carrots_weight_l2191_219169


namespace aerith_seat_l2191_219191

-- Let the seats be numbered 1 through 8
-- Assigned seats for Aerith, Bob, Chebyshev, Descartes, Euler, Fermat, Gauss, and Hilbert
variables (a b c d e f g h : ℕ)

-- Define the conditions described in the problem
axiom Bob_assigned : b = 1
axiom Chebyshev_assigned : c = g + 2
axiom Descartes_assigned : d = f - 1
axiom Euler_assigned : e = h - 4
axiom Fermat_assigned : f = d + 5
axiom Gauss_assigned : g = e + 1
axiom Hilbert_assigned : h = a - 3

-- Provide the proof statement to find whose seat Aerith sits
theorem aerith_seat : a = c := sorry

end aerith_seat_l2191_219191


namespace meetings_percentage_l2191_219185

theorem meetings_percentage
  (workday_hours : ℕ)
  (first_meeting_minutes : ℕ)
  (second_meeting_factor : ℕ)
  (third_meeting_factor : ℕ)
  (total_minutes : ℕ)
  (total_meeting_minutes : ℕ) :
  workday_hours = 9 →
  first_meeting_minutes = 30 →
  second_meeting_factor = 2 →
  third_meeting_factor = 3 →
  total_minutes = workday_hours * 60 →
  total_meeting_minutes = first_meeting_minutes + second_meeting_factor * first_meeting_minutes + third_meeting_factor * first_meeting_minutes →
  (total_meeting_minutes : ℚ) / (total_minutes : ℚ) * 100 = 33.33 :=
by
  sorry

end meetings_percentage_l2191_219185


namespace vampires_after_two_nights_l2191_219151

-- Define the initial conditions and calculations
def initial_vampires : ℕ := 2
def transformation_rate : ℕ := 5
def first_night_vampires : ℕ := initial_vampires * transformation_rate + initial_vampires
def second_night_vampires : ℕ := first_night_vampires * transformation_rate + first_night_vampires

-- Prove that the number of vampires after two nights is 72
theorem vampires_after_two_nights : second_night_vampires = 72 :=
by sorry

end vampires_after_two_nights_l2191_219151


namespace find_mn_solutions_l2191_219199

theorem find_mn_solutions :
  ∀ (m n : ℤ), m^5 - n^5 = 16 * m * n →
  (m = 0 ∧ n = 0) ∨ (m = -2 ∧ n = 2) :=
by
  sorry

end find_mn_solutions_l2191_219199


namespace ab_greater_than_a_plus_b_l2191_219175

theorem ab_greater_than_a_plus_b (a b : ℝ) (h₁ : a ≥ 2) (h₂ : b > 2) : a * b > a + b :=
  sorry

end ab_greater_than_a_plus_b_l2191_219175
